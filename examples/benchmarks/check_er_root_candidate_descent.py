"""Check NEW candidate descent directions using independent full forward solves.

Run the candidate-only reverse benchmark first, then pass its JSON report.
Each forward endpoint gets a fresh subprocess, avoiding accumulated GPU
executables. Completed endpoints can be reused only for identical inputs.
These are finite physical moves, not a frozen-schedule AD/FD oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def task_digest(task):
    return hashlib.sha256(json.dumps(task, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_digest(config):
    """Invalidate cached endpoints after local model or seed changes."""
    digest = hashlib.sha256()
    paths = [Path(__file__), Path(__file__).with_name("_er_root_loss_audit.py"),
             Path(__file__).with_name("compare_er_transition_loss_candidates.py"),
             Path(__file__).with_name("benchmark_transport_realtime_geometry_forward_fd.py")]
    paths += sorted((ROOT / "NEOPAX").rglob("*.py"))
    seed = config.get("geometry", {}).get("vmec_input_file")
    if seed:
        seed = Path(seed)
        paths.append(seed if seed.is_absolute() else ROOT / seed)
    for path in paths:
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def plan_checks(report, steps, candidates=("local", "annular"), groups=("profiles", "geometry")):
    if not report.get("all_finite"):
        raise ValueError("The candidate AD report must have finite values and derivatives.")
    if report.get("accepted_step_limit") is not None:
        raise ValueError("Use a full-time AD report, with no accepted-step prefix limit.")
    audit = report["root_loss_audit"]
    experiment = audit["experiment"]
    if experiment["initial_er_root_ad"] != "jax_selected_root":
        raise ValueError("Use --initial-Er-root-ad jax_selected_root for this physical descent check.")
    if experiment["resolved_config"].get("transport_solver", {}).get("stop_after_accepted_steps") is not None:
        raise ValueError("Remove the configured accepted-step prefix limit for a full-time descent check.")
    labels = report["parameter_order"]
    baseline = np.asarray(experiment["parameter_values"], dtype=float)
    if baseline.shape != (len(labels),) or not np.isfinite(baseline).all():
        raise ValueError("Invalid baseline parameter vector.")
    steps = tuple(float(step) for step in steps)
    if not steps or any(not np.isfinite(step) or step <= 0 for step in steps):
        raise ValueError("Descent step sizes must be finite and positive.")
    if not candidates or not set(candidates) <= {"local", "annular"} or not groups or not set(groups) <= {"profiles", "geometry"}:
        raise ValueError("Choose local/annular candidates and profiles/geometry groups.")
    base = dict(resolved_config=experiment["resolved_config"], parameter_labels=labels,
                settings=audit["settings"], initial_er_root_ad=experiment["initial_er_root_ad"],
                source_digest=source_digest(experiment["resolved_config"]))
    tasks = [dict(base, name="baseline", parameter_values=baseline.tolist())]
    for candidate in candidates:
        name = f"audit_root_{candidate}_cost"
        comparison = next(c for c in audit["comparisons"] if c["name"] == name)
        for group in groups:
            direction = comparison["groups"][group]
            if direction["scaled_gradient_l2"] == 0:
                continue
            vector = np.asarray(direction["physical_descent_direction"], dtype=float)
            if vector.shape != baseline.shape or not np.isfinite(vector).all():
                raise ValueError("Invalid physical descent direction.")
            for step in steps:
                point = baseline + step * vector
                if any(value <= 0 for label, value in zip(labels, point, strict=True) if ":" not in label):
                    raise ValueError("Descent step would make a profile parameter nonpositive; reduce --step-sizes.")
                tasks.append(dict(base, name=f"{candidate}_{group}_{step:g}", candidate=name, group=group,
                                  step_size=step, parameter_values=point.tolist(), prediction=direction))
    return tasks


def run_forward_task(task):
    """Worker only: rebuild the requested physical point and solve adaptively."""
    import copy
    import dataclasses
    import jax
    import jax.numpy as jnp

    # Existing benchmark helpers use sibling imports when run as scripts.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from benchmark_transport_reverse_ad_only import _geometry_context_from_config, _benchmark_device_context
    from benchmark_transport_realtime_geometry_forward_fd import _schedule_rollout
    from NEOPAX._geometry_autodiff import _input_with_boundary_deltas, boundary_param_entries
    from NEOPAX._orchestrator import build_runtime_context
    from NEOPAX._reverse_ad_parameters import PROFILE_PARAMETER_ORDER
    from NEOPAX._reverse_ad_transport import initial_state_for_parameter_vector
    from examples.benchmarks._er_root_loss_audit import RootLossAuditSettings, terminal_metrics
    import tempfile

    cfg = copy.deepcopy(task["resolved_config"])
    labels, values = task["parameter_labels"], task["parameter_values"]
    physical = dict(zip(labels, values, strict=True))
    cfg.setdefault("profiles", {}).update({name: physical[name] for name in PROFILE_PARAMETER_ORDER})
    cfg.setdefault("ambipolarity", {})["er_ambipolar_plot"] = False
    settings = RootLossAuditSettings(**task["settings"])
    geometry_labels = [label for label in labels if ":" in label]
    specs = [tuple([label.removeprefix("vmec:").split(":")[0],
                    *map(int, label.removeprefix("vmec:").split(":")[1:])]) for label in geometry_labels]
    deltas = [physical[label] for label in geometry_labels]
    with _benchmark_device_context(cfg), tempfile.TemporaryDirectory(prefix="neopax_root_descent_") as directory:
        if any(delta != 0 for delta in deltas):
            context = _geometry_context_from_config(cfg, ",".join(f"{f}:{m}:{n}" for f, m, n in specs))
            updated = _input_with_boundary_deltas(context, jnp.asarray(deltas), boundary_param_entries(context, specs))
            input_path = Path(directory) / "input.root_descent"
            updated.to_indata(input_path)
            cfg["geometry"]["vmec_input_file"] = str(input_path)
            cfg["geometry"]["vmec_param_delta"] = 0.0
        runtime, state = build_runtime_context(cfg)
        # Match profile construction/floors in the production reverse builder.
        # Runtime construction has already solved the selected initial Er for
        # these changed profiles/geometry; reuse that concrete root profile.
        initial_er = state.Er
        state = initial_state_for_parameter_vector(
            jnp.asarray([physical[name] for name in PROFILE_PARAMETER_ORDER]), baseline_state=state,
            runtime=runtime, profile_cfg=cfg["profiles"], config=cfg, initial_er_root_ad="off",
        )
        state = dataclasses.replace(state, Er=initial_er)
        _, _, _, rollout, final = _schedule_rollout(cfg, runtime, state, accepted_step_limit=None)
        er = np.asarray(jax.device_get(final.Er), dtype=float)
        rho = np.asarray(jax.device_get(runtime.geometry.rho_grid), dtype=float)
        metrics = terminal_metrics(er, rho, settings)
        return dict(task_digest=task_digest(task), name=task["name"], metrics=metrics,
                    terminal_Er=er.tolist(), terminal_rho=rho.tolist(),
                    final_time=float(jax.device_get(rollout.final_carry.t)),
                    accepted_steps=int(jax.device_get(rollout.accepted_count)),
                    attempts=int(jax.device_get(rollout.attempt_count)), parameter_values=values)


def compare_endpoint(task, baseline, endpoint, audit):
    name = task["candidate"]
    before, after = baseline["metrics"], endpoint["metrics"]
    cost_change = after["candidates"][name]["cost"] - before["candidates"][name]["cost"]
    position_change = (after["position_cost"] - before["position_cost"]
                       if before["position_cost"] is not None and after["position_cost"] is not None else None)
    rate = task["prediction"]["predicted_position_cost_change"]
    root_change = (after["nearest_downcrossing"] - before["nearest_downcrossing"]
                   if before["nearest_downcrossing"] is not None and after["nearest_downcrossing"] is not None else None)
    rms_change = after["annular_Er_rms"] - before["annular_Er_rms"]
    return dict(name=task["name"], candidate=name, group=task["group"], step_size=task["step_size"],
                cost_change=cost_change, zero_cost_change=after["candidates"][name]["zero_cost"] - before["candidates"][name]["zero_cost"],
                sign_cost_change=after["candidates"][name]["sign_cost"] - before["candidates"][name]["sign_cost"],
                position_cost_change=position_change, root_change=root_change,
                predicted_cost_change=task["step_size"] * task["prediction"]["predicted_cost_change_per_unit_descent"],
                predicted_position_cost_change=task["step_size"] * rate if rate is not None else None,
                annular_Er_rms_change=rms_change,
                cost_decreased=cost_change < 0,
                root_position_improved=position_change < 0 if position_change is not None else None,
                cost_decreased_without_position_improvement=(cost_change < 0 and position_change >= 0) if position_change is not None else None,
                root_created=before["nearest_downcrossing"] is None and after["nearest_downcrossing"] is not None,
                root_lost=before["nearest_downcrossing"] is not None and after["nearest_downcrossing"] is None,
                root_bracket_changed=before["nearest_downcrossing_bracket"] != after["nearest_downcrossing_bracket"],
                baseline_root_error=(before["nearest_downcrossing"] - audit["target"]) if before["nearest_downcrossing"] is not None else None,
                endpoint_metrics=after)


def plot_profiles(summary, out, audit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True, constrained_layout=True)
    baseline = summary["baseline"]
    for i, candidate in enumerate(("local", "annular")):
        for j, group in enumerate(("profiles", "geometry")):
            ax = axes[i, j]
            ax.plot(baseline["terminal_rho"], baseline["terminal_Er"], "k--", label="Baseline")
            for result in summary["results"]:
                if result["candidate"] == f"audit_root_{candidate}_cost" and result["group"] == group:
                    endpoint = json.loads((out / f"{result['name']}.json").read_text())
                    ax.plot(endpoint["terminal_rho"], endpoint["terminal_Er"], label=f"Descent step {result['step_size']:g}")
            ax.axvline(audit["target"], color="0.4", linestyle=":", label="Target radius")
            ax.axhline(0, color="0.6", linewidth=0.7)
            ax.set(title=f"{candidate.capitalize()} candidate / {group}", xlabel="rho", ylabel="Er [kV/m]")
            ax.legend(fontsize=8)
    fig.savefig(out / "terminal_Er_comparison.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ad-report", type=Path)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/root_candidate_descent")
    parser.add_argument("--step-sizes", default="0.01", help="Positive trust-coordinate step sizes, comma separated.")
    parser.add_argument("--candidates", default="local,annular")
    parser.add_argument("--groups", default="profiles,geometry")
    parser.add_argument("--worker-task", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_task:
        result = run_forward_task(json.loads(args.worker_task.read_text()))
        args.worker_output.write_text(json.dumps(result, indent=2, allow_nan=False))
        return
    if args.ad_report is None:
        parser.error("--ad-report is required")
    report = json.loads(args.ad_report.read_text())
    try:
        tasks = plan_checks(report, args.step_sizes.split(","), args.candidates.split(","), args.groups.split(","))
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    audit = report["root_loss_audit"]
    summary = dict(ad_report=str(args.ad_report.resolve()), results=[], failures=[],
                   interpretation="Independent adaptive forward solves at finite downhill moves. Not a frozen-schedule AD/FD comparison.")
    baseline = None
    for task in tasks:
        task_path, result_path = out / f"{task['name']}_task.json", out / f"{task['name']}.json"
        task_path.write_text(json.dumps(task, indent=2, allow_nan=False))
        endpoint = json.loads(result_path.read_text()) if result_path.exists() else None
        if endpoint is None or endpoint.get("task_digest") != task_digest(task):
            print(f"[candidate-descent] running {task['name']}; full adaptive forward solve", flush=True)
            with (out / f"{task['name']}.log").open("w") as log:
                process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker-task", str(task_path),
                                          "--worker-output", str(result_path)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if process.returncode:
                summary["failures"].append(dict(name=task["name"], exit_code=process.returncode, log=str(log.name)))
                (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
                print(f"[candidate-descent] FAILED {task['name']}: see {log.name}", flush=True)
                if baseline is None:
                    raise SystemExit(1)
                continue
            endpoint = json.loads(result_path.read_text())
        if baseline is None:
            baseline = endpoint
            reference = np.asarray(audit["terminal_Er"])
            actual = np.asarray(endpoint["terminal_Er"])
            match = (np.allclose(endpoint["terminal_rho"], audit["terminal_rho"], rtol=1e-10, atol=1e-12)
                     and np.allclose(actual, reference, rtol=1e-6, atol=1e-7)
                     and np.isclose(endpoint["final_time"], audit["terminal_time"], rtol=0, atol=1e-10))
            summary["baseline_matches_ad"] = bool(match)
            summary["baseline_Er_max_abs_difference"] = float(np.max(abs(actual - reference)))
            actual_root, ad_root = endpoint["metrics"]["nearest_downcrossing"], audit["nearest_downcrossing"]
            summary["baseline_root_difference_from_ad"] = actual_root - ad_root if actual_root is not None and ad_root is not None else None
            summary["baseline"] = endpoint
            if not match:
                (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
                raise SystemExit("Forward baseline does not reproduce the AD endpoint. See summary.json; no downhill runs attempted.")
        else:
            comparison = compare_endpoint(task, baseline, endpoint, audit)
            summary["results"].append(comparison)
            print("[candidate-descent] " + json.dumps(comparison, allow_nan=False), flush=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(f"Wrote {out / 'summary.json'}", flush=True)
    plot_profiles(summary, out, audit)
    if summary["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
