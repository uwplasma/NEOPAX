"""Optimize the new annular root loss through full-time transport.

This pilot starts from a saved root-loss AD report with an existing crossing.
Only the six initial-profile parameters vary; geometry and configuration stay
fixed. The scalar candidate cost is minimized directly, without squaring it
again. Forward trials and accepted-point AD evaluations use fresh processes.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.benchmarks import check_er_root_candidate_descent as descent

CANDIDATE = "audit_root_annular_cost"
PROFILES = ("n0", "T0", "density_shape_power", "temperature_shape_power",
            "density_shape_alpha", "temperature_shape_alpha")
PHYSICS_KEYS = ("resolved_config", "parameter_labels", "parameter_values", "settings",
                "initial_er_root_ad", "profile_contract", "source_digest")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


@dataclass(frozen=True)
class PilotOptions:
    max_iterations: int = 5
    initial_step: float = 0.001
    min_step: float = 1e-7
    max_step: float = 0.01
    max_backtracks: int = 8
    armijo: float = 1e-4
    root_tolerance: float = 1e-4
    gradient_tolerance: float = 1e-12

    def __post_init__(self):
        values = (self.initial_step, self.min_step, self.max_step, self.armijo,
                  self.root_tolerance, self.gradient_tolerance)
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError("Step sizes, tolerances and Armijo factor must be finite and positive.")
        if not self.min_step <= self.initial_step <= self.max_step or self.armijo >= 1:
            raise ValueError("Require min_step <= initial_step <= max_step and armijo < 1.")
        if self.max_iterations <= 0 or self.max_backtracks <= 0:
            raise ValueError("Iteration and backtracking budgets must be positive.")


def comparison(report):
    return next(item for item in report["root_loss_audit"]["comparisons"] if item["name"] == CANDIDATE)


def validate_seed(report, options):
    tasks = descent.plan_checks(report, [options.initial_step], ("annular",), ("profiles",))
    if set(name for name in report["parameter_order"] if ":" not in name) != set(PROFILES):
        raise ValueError("The pilot requires the six standard profile parameters.")
    if report.get("reverse_replay_jacobian_reuse_mode") != "recorded_forward":
        raise ValueError("Use the validated full-time recorded_forward root-loss AD report.")
    if report["root_loss_audit"].get("nearest_downcrossing") is None:
        raise ValueError("This pilot starts from an existing annular downcrossing.")
    return tasks[0]


def direction(report, scales):
    """Normalized cost descent in fixed trust coordinates, with zero geometry move."""
    labels = report["parameter_order"]
    gradient = np.array([comparison(report)["physical_parameter_gradient"][name] for name in labels])
    if gradient.shape != scales.shape or not np.isfinite(gradient).all():
        raise ValueError("Invalid physical cost gradient or coordinate scales.")
    indices = [i for i, name in enumerate(labels) if ":" not in name]
    scaled = gradient[indices] * scales[indices]
    norm = float(np.linalg.norm(scaled))
    if not np.isfinite(norm):
        raise ValueError("Nonfinite scaled profile-gradient norm.")
    result = np.zeros(len(labels))
    if norm > 0:
        result[indices] = -scales[indices] * scaled / norm
    return result, norm


def require_matching_forward(report, endpoint):
    """Compare the independent forward endpoint to the AD primal at this point."""
    audit = report["root_loss_audit"]
    expected, actual = np.asarray(audit["terminal_Er"]), np.asarray(endpoint["terminal_Er"])
    if (not np.array_equal(endpoint["parameter_values"], audit["experiment"]["parameter_values"])
            or expected.shape != actual.shape or not np.isfinite(actual).all()
            or not np.allclose(actual, expected, rtol=1e-6, atol=1e-7)
            or not np.allclose(endpoint["terminal_rho"], audit["terminal_rho"], rtol=1e-10, atol=1e-12)
            or not np.isclose(endpoint["final_time"], audit["terminal_time"], rtol=0, atol=1e-10)
            or not np.isclose(endpoint["metrics"]["candidates"][CANDIDATE]["cost"],
                              comparison(report)["cost"], rtol=1e-6, atol=1e-7)):
        raise ValueError("Independent forward and AD primal disagree; see saved endpoint/report.")


def forward_key(task):
    # Names and first-order predictions do not affect the forward physics.
    return descent.task_digest({key: task[key] for key in PHYSICS_KEYS})


def ad_arguments(template):
    """Use the existing benchmark parser/defaults and recorded setup, not a new AD lane."""
    sys.path.insert(0, str(ROOT / "examples/benchmarks"))
    import benchmark_transport_reverse_ad_only as benchmark

    geometry = ",".join(name.removeprefix("vmec:") for name in template["parameter_order"] if ":" in name)
    args = benchmark.build_argument_parser().parse_args([
        "--reverse-parameter-mode", "profiles_plus_realtime_geometry",
        "--reverse-geometry-parameter", geometry,
        "--realtime-geometry-gradient-path", "reverse_payload", "--objective", "all",
        "--full-transport-shared-payload-smoke", "--root-loss-audit",
        "--initial-Er-root-ad", "jax_selected_root", "--max-reverse-accepted-steps", "1000",
        "--reverse-stage-adjoint-solve-mode", "block",
        "--reverse-rhs-transpose-mode", "explicit_database",
        "--reverse-step-bwd-mode", "reduced_cotangent_call_boundary",
        "--reverse-initial-cache-support-pullback-mode", "scalar",
        "--reverse-rebuild-support-pullback-mode", "separate",
        "--reverse-final-objective-cotangent-mode", "grouped_vjp",
        "--reverse-bootstrap-cotangent-mode", "joint_local_vjp_upar_only",
        "--reverse-segment-start-replay-mode", "minimal",
        "--reverse-segment-primal-record-mode", "reuse_segment_primal_record",
        "--reverse-schedule-artifact-mode", "reuse_static_probe",
        "--reverse-replay-jacobian-reuse-mode", "recorded_forward",
    ])
    for name, value in template.items():
        if name.startswith("reverse_") and hasattr(args, name) and value is not None:
            setattr(args, name, value)
    args.config = template.get("config_path", args.config)
    args.accepted_step_limit = None
    args.reverse_segment_length = template["reverse_segment_length"]
    settings = template["root_loss_audit"]["settings"]
    for name, key in (("target", "target"), ("inner_radius", "inner_radius"),
                      ("outer_radius", "outer_radius"), ("profile_trust_multiplier", "profile_trust_multiplier"),
                      ("ess_alpha", "geometry_ess_alpha")):
        setattr(args, "root_loss_audit_" + name, settings[key])
    return benchmark._resolve_reverse_benchmark_mode_defaults(args, database_full_transport=True)


def run_ad_task(task):
    """Worker only: fixed config, requested initial profiles, established full reverse."""
    import jax.numpy as jnp
    sys.path.insert(0, str(ROOT / "examples/benchmarks"))
    import benchmark_transport_reverse_ad_only as benchmark
    from NEOPAX._orchestrator import build_runtime_context

    template = task["seed_report"]
    cfg = copy.deepcopy(template["root_loss_audit"]["experiment"]["resolved_config"])
    args = ad_arguments(template)
    with benchmark._benchmark_device_context(cfg):
        runtime, state = build_runtime_context(cfg)
        geometry = benchmark._geometry_context_from_config(cfg, args.reverse_geometry_parameter)
        return benchmark._run_realtime_geometry_optimization_api_smoke(
            args=args, config=cfg, geometry_context=geometry,
            baseline_values=jnp.asarray(template["root_loss_audit"]["experiment"]["parameter_values"]),
            baseline_runtime=runtime, baseline_state=state, profile_cfg=cfg.get("profiles", {}),
            neoclassical_cfg=cfg.get("neoclassical", {}),
            parameter_values_override=task["forward_task"]["parameter_values"], write_report=False,
        )


class WorkerFailure(RuntimeError):
    pass


class SubprocessEvaluator:
    def __init__(self, template, baseline_task, out, descent_dir=None):
        self.template, self.baseline_task, self.out = template, baseline_task, out
        self.counter = max((int(p.name.split("_")[1]) for p in out.glob("evaluation_*_task.json")), default=0)
        self.cache = {}
        if descent_dir is not None:
            for path in descent_dir.glob("*_task.json"):
                result_path = path.with_name(path.name.replace("_task.json", ".json"))
                if not result_path.exists():
                    continue
                task, result = json.loads(path.read_text()), json.loads(result_path.read_text())
                if result.get("task_digest") == descent.task_digest(task) and all(key in task for key in PHYSICS_KEYS):
                    self.cache[forward_key(task)] = result

    def task_at(self, parameters):
        task = dict(self.baseline_task, parameter_values=np.asarray(parameters).tolist())
        task["name"] = f"optimization_trial_{self.counter + 1:04d}"
        return task

    def forward(self, parameters):
        task = self.task_at(parameters)
        if forward_key(task) in self.cache:
            print("[root-optimization] reusing matching saved forward endpoint", flush=True)
            return copy.deepcopy(self.cache[forward_key(task)])
        return self._run("forward", task)

    def ad(self, parameters):
        return self._run("ad", self.task_at(parameters))

    def _run(self, kind, forward_task):
        self.counter += 1
        stem = self.out / f"evaluation_{self.counter:04d}_{kind}"
        task_path, result_path, log_path = (stem.with_name(stem.name + suffix)
                                           for suffix in ("_task.json", ".json", ".log"))
        write_json(task_path, dict(kind=kind, forward_task=forward_task, seed_report=self.template))
        print(f"[root-optimization] {kind} evaluation; log={log_path}", flush=True)
        with log_path.open("w") as log:
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker-task", str(task_path),
                                      "--worker-output", str(result_path)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            raise WorkerFailure(f"{kind} worker exited {process.returncode}; see {log_path}")
        return json.loads(result_path.read_text())


def optimize(seed, evaluator, options, *, state=None, save=lambda _: None, emit=lambda _: None):
    """Host-only adaptive descent; all physics is delegated to evaluator."""
    labels = seed["parameter_order"]
    scales = np.asarray(seed["root_loss_audit"]["coordinate_scales"], dtype=float)
    if scales.shape != (len(labels),) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Invalid saved coordinate scales.")
    profile_indices = [i for i, name in enumerate(labels) if ":" not in name]
    target = float(seed["root_loss_audit"]["target"])

    def record(parameters, endpoint, iteration, step, ratio):
        root = endpoint["metrics"]["nearest_downcrossing"]
        return dict(iteration=iteration, cost=endpoint["metrics"]["candidates"][CANDIDATE]["cost"],
                    root=root, root_error=abs(root - target) if root is not None else None,
                    step=step, reduction_ratio=ratio, parameters=list(parameters))

    if state is None:
        point = seed["root_loss_audit"]["experiment"]["parameter_values"]
        endpoint = evaluator.forward(point)
        require_matching_forward(seed, endpoint)
        state = dict(parameters=list(point), report=seed, endpoint=endpoint, next_step=options.initial_step,
                     history=[record(point, endpoint, 0, 0., None)], status="running", converged=False)
        save(state)
    else:
        state = copy.deepcopy(state)
        require_matching_forward(state["report"], state["endpoint"])

    while True:
        root = state["endpoint"]["metrics"]["nearest_downcrossing"]
        if (root is not None and abs(root - target) <= options.root_tolerance
                and state["endpoint"]["metrics"]["wrong_sign_fraction"] == 0):
            state.update(status="root_target_met", converged=True)
            break
        iteration = len(state["history"])
        if iteration > options.max_iterations:
            state.update(status="iteration_limit", converged=False)
            break
        vector, norm = direction(state["report"], scales)
        if norm <= options.gradient_tolerance:
            state.update(status="stationary_before_target", converged=False)
            break
        point = np.asarray(state["parameters"])
        cost = state["history"][-1]["cost"]
        step, accepted = min(state["next_step"], options.max_step), False
        for trial in range(options.max_backtracks):
            if step < options.min_step:
                break
            candidate = point + step * vector
            event = dict(event="trial", iteration=iteration, trial=trial + 1, step=step,
                         parameters=candidate.tolist(), predicted_cost_change=-step * norm)
            try:
                if np.any(candidate[profile_indices] <= 0):
                    raise ValueError("Nonpositive trial profile parameter.")
                endpoint = evaluator.forward(candidate)
                metrics = endpoint["metrics"]
                trial_cost = float(metrics["candidates"][CANDIDATE]["cost"])
                event.update(cost=trial_cost, root=metrics["nearest_downcrossing"])
                if not np.isfinite(trial_cost) or metrics["nearest_downcrossing"] is None:
                    raise ValueError("Trial has a nonfinite cost or loses the existing root.")
                if not np.isclose(endpoint["final_time"], seed["root_loss_audit"]["terminal_time"], rtol=0, atol=1e-10):
                    raise ValueError("Forward trial did not reach the full simulation endpoint.")
                if trial_cost >= cost or trial_cost > cost - options.armijo * step * norm:
                    raise ValueError("Insufficient actual cost reduction.")
                emit({**event, "event": "gradient_started"})
                report = evaluator.ad(candidate)
                if not report.get("all_finite") or report["parameter_order"] != labels:
                    raise ValueError("Invalid accepted-point AD table.")
                if not np.array_equal(report["root_loss_audit"]["experiment"]["parameter_values"], candidate):
                    raise ValueError("AD evaluated the wrong parameter point.")
                direction(report, scales)  # Require a finite next gradient before accepting.
                require_matching_forward(report, endpoint)
            except (WorkerFailure, ValueError, FloatingPointError) as exc:
                emit({**event, "event": "rejected", "reason": str(exc)})
                step *= 0.5
                continue
            ratio = (cost - trial_cost) / (step * norm)
            next_step = step * (1.5 if ratio > 0.75 else 0.5 if ratio < 0.25 else 1.)
            state.update(parameters=candidate.tolist(), report=report, endpoint=endpoint,
                         next_step=min(options.max_step, max(options.min_step, next_step)), status="running")
            state["history"].append(record(candidate, endpoint, iteration, step, ratio))
            emit({**event, "event": "accepted", "reduction_ratio": ratio,
                  "root_error": state["history"][-1]["root_error"]})
            save(state)
            accepted = True
            break
        if not accepted:
            state.update(status="backtracking_exhausted", converged=False, next_step=max(step, options.min_step))
            break
    save(state)
    return state


def save_outputs(out, state, signature):
    write_json(out / "checkpoint.json", dict(signature=signature, state=state))
    summary = {name: state[name] for name in ("status", "converged", "parameters", "next_step", "history")}
    summary.update(candidate=CANDIDATE, target=state["report"]["root_loss_audit"]["target"],
                   geometry_fixed=True, full_transport=True, accepted_iterations=len(state["history"]) - 1)
    write_json(out / "summary.json", summary)
    labels = state["report"]["parameter_order"]
    fields = ["iteration", "cost", "root", "root_error", "step", "reduction_ratio"]
    with (out / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + labels)
        writer.writeheader()
        for row in state["history"]:
            writer.writerow({**{key: row[key] for key in fields}, **dict(zip(labels, row["parameters"], strict=True))})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ad-report", type=Path)
    parser.add_argument("--descent-dir", type=Path, help="Reuse matching saved forward endpoints from the checker.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/profiles_annular_root_optimization")
    parser.add_argument("--resume", action="store_true")
    for name, kind, default in (("max-iterations", int, 5), ("initial-step", float, .001),
                                ("min-step", float, 1e-7), ("max-step", float, .01),
                                ("max-backtracks", int, 8), ("armijo", float, 1e-4),
                                ("root-tolerance", float, 1e-4), ("gradient-tolerance", float, 1e-12)):
        parser.add_argument("--" + name, type=kind, default=default)
    parser.add_argument("--worker-task", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_task:
        task = json.loads(args.worker_task.read_text())
        result = (descent.run_forward_task(task["forward_task"]) if task["kind"] == "forward" else run_ad_task(task))
        write_json(args.worker_output, result)
        return
    if args.ad_report is None:
        parser.error("--ad-report is required")
    options = PilotOptions(**{key: getattr(args, key) for key in asdict(PilotOptions())})
    seed = json.loads(args.ad_report.read_text())
    baseline_task = validate_seed(seed, options)
    options_digest = asdict(options)
    options_digest.pop("max_iterations")  # A resumed pilot may increase its accepted-iteration budget.
    signature = descent.task_digest(dict(seed=seed, options=options_digest,
        physics_source=baseline_task["source_digest"],
        driver_source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reverse_source=hashlib.sha256((ROOT / "examples/benchmarks/benchmark_transport_reverse_ad_only.py").read_bytes()).hexdigest()))
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    checkpoint, state = out / "checkpoint.json", None
    if args.resume:
        saved = json.loads(checkpoint.read_text())
        if saved["signature"] != signature:
            parser.error("Resume inputs/settings/sources differ; use a new output directory.")
        state = saved["state"]
    elif checkpoint.exists():
        parser.error("Checkpoint already exists; use --resume or a new output directory.")
    evaluator = SubprocessEvaluator(seed, baseline_task, out, args.descent_dir)

    def emit(event):
        with (out / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
        print("[root-optimization] " + json.dumps(event, allow_nan=False), flush=True)

    state = optimize(seed, evaluator, options, state=state,
                     save=lambda value: save_outputs(out, value, signature), emit=emit)
    print(f"[root-optimization] status={state['status']} converged={state['converged']} "
          f"accepted_iterations={len(state['history']) - 1} root_error={state['history'][-1]['root_error']} "
          f"summary={out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
