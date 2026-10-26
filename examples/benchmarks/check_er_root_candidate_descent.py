"""Check candidate/control descent directions using independent full forward solves.

Run the candidate-only reverse benchmark first, then pass its JSON report.
Each forward endpoint gets a fresh subprocess, avoiding accumulated GPU
executables. Completed endpoints can be reused only for identical inputs.
These are finite physical moves, not a frozen-schedule AD/FD oracle.
Profile moves follow the AD map: fixed configuration, changed initial
profile values, and the requested JAX selected-root initializer.
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

CANDIDATES = ("local", "annular", "original", "bootstrap", "original_bootstrap")


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
    if not candidates or not set(candidates) <= set(CANDIDATES) or not groups or not set(groups) <= {"profiles", "geometry"}:
        raise ValueError("Choose local/annular/original/bootstrap/original_bootstrap and profiles/geometry groups.")
    base = dict(resolved_config=experiment["resolved_config"], parameter_labels=labels,
                settings=audit["settings"], initial_er_root_ad=experiment["initial_er_root_ad"],
                profile_contract="ad_initial_profiles_fixed_config_selected_root",
                source_digest=source_digest(experiment["resolved_config"]))
    if "original_objective_settings" in audit:
        base.update(objective_suite="original", original_objective_settings=audit["original_objective_settings"])
    tasks = [dict(base, name="baseline", parameter_values=baseline.tolist())]
    for candidate in candidates:
        name = f"audit_root_{candidate}_cost"
        comparison = next((c for c in audit["comparisons"] if c["name"] == name), None)
        if comparison is None:
            raise ValueError(f"The AD report lacks {name}; prepare a fresh --objective-suite original baseline for original controls.")
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


def compare_initial_roots(baseline_er, selected_er, general_er, implicit_rate, step, rho, annulus):
    """Compare actual selected roots against local fixed-root continuation.

    The implicit rate uses a finite difference of the charge-flux residual
    at the fixed baseline root and its AD derivative with respect to Er.
    It does not differentiate the discrete minimum-entropy selection.
    """
    a, b, c, rate, rho = map(lambda value: np.asarray(value, dtype=float),
                             (baseline_er, selected_er, general_er, implicit_rate, rho))
    band = (rho >= annulus[0]) & (rho <= annulus[1])
    usable = band & np.isfinite(rate)
    observed = (b - a) / step
    error = observed[usable] - rate[usable]
    denominator = max(float(np.linalg.norm(observed[usable])), float(np.linalg.norm(rate[usable])), 1e-30)
    return dict(selected_minus_general_max_abs=float(np.max(abs(b - c))),
                selected_minus_general_annular_max_abs=float(np.max(abs(b[band] - c[band]))),
                selected_initial_sign_change_indices=np.flatnonzero(band & (np.sign(a) != np.sign(b))).tolist(),
                implicit_rate_valid_indices=np.flatnonzero(usable).tolist(),
                annular_rate_error_l2_relative=float(np.linalg.norm(error) / denominator) if np.any(usable) else None,
                annular_rate_error_max_abs=float(np.max(abs(error))) if np.any(usable) else None,
                observed_initial_Er_change=(b - a).tolist(),
                predicted_initial_Er_change=[float(step * value) if np.isfinite(value) else None for value in rate])


def diagnose_profile_initialization(report, steps, candidate, out):
    """Large-machine probe: one runtime build, initial roots only, no transport."""
    import copy
    import jax
    import jax.numpy as jnp

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from benchmark_transport_reverse_ad_only import _benchmark_device_context
    from NEOPAX._orchestrator import build_runtime_context, _maybe_initialize_er_from_ambipolarity
    from NEOPAX._profiles import build_profiles
    from NEOPAX._reverse_ad_parameters import PROFILE_PARAMETER_ORDER
    from NEOPAX._reverse_ad_transport import initial_state_for_parameter_vector
    from NEOPAX._reverse_ad_initial_er import (
        initial_er_charge_flux_residuals, initial_er_charge_flux_residual_er_derivative,
    )

    tasks = plan_checks(report, steps, (candidate,), ("profiles",))
    audit = report["root_loss_audit"]
    cfg = copy.deepcopy(tasks[0]["resolved_config"])
    cfg.setdefault("ambipolarity", {})["er_ambipolar_plot"] = False
    names = report["parameter_order"]
    profile_indices = [names.index(name) for name in PROFILE_PARAMETER_ORDER]
    p0 = np.asarray(tasks[0]["parameter_values"])[profile_indices]
    comparison = next(item for item in audit["comparisons"] if item["name"] == f"audit_root_{candidate}_cost")
    direction = np.asarray(comparison["groups"]["profiles"]["physical_descent_direction"])[profile_indices]
    epsilon = min(1e-6, min(map(float, steps)) / 100)
    out.mkdir(parents=True, exist_ok=True)

    with _benchmark_device_context(cfg):
        print("[profile-initialization] building ONE baseline runtime; no transport solve", flush=True)
        runtime, baseline = build_runtime_context(cfg)

        def state_from_profiles(values, root_mode):
            return initial_state_for_parameter_vector(
                jnp.asarray(values), baseline_state=baseline, runtime=runtime,
                profile_cfg=cfg.get("profiles", {}), config=cfg, initial_er_root_ad=root_mode,
            )

        pre0 = state_from_profiles(p0, "off")
        selected0 = state_from_profiles(p0, "jax_selected_root")
        er0 = selected0.Er
        residual0 = np.asarray(jax.device_get(initial_er_charge_flux_residuals(pre0, er0, runtime=runtime)))
        er_derivative = np.asarray(jax.device_get(initial_er_charge_flux_residual_er_derivative(pre0, er0, runtime=runtime)))
        plus_residual = np.asarray(jax.device_get(initial_er_charge_flux_residuals(
            state_from_profiles(p0 + epsilon * direction, "off"), er0, runtime=runtime)))
        minus_residual = np.asarray(jax.device_get(initial_er_charge_flux_residuals(
            state_from_profiles(p0 - epsilon * direction, "off"), er0, runtime=runtime)))
        usable = np.isfinite(plus_residual) & np.isfinite(minus_residual) & np.isfinite(er_derivative) & (abs(er_derivative) > 1e-30)
        rate = np.full_like(er_derivative, np.nan)
        rate[usable] = -(plus_residual[usable] - minus_residual[usable]) / (2 * epsilon * er_derivative[usable])
        rho = np.asarray(jax.device_get(runtime.geometry.rho_grid))
        temperatures0 = np.asarray(jax.device_get(build_profiles(cfg.get("profiles", {}), runtime.geometry,
                                                               runtime.species.number_species).temperature)) / 1e3
        frozen = np.flatnonzero(~np.asarray(cfg.get("equations", {}).get(
            "toggle_temperature", [True] * runtime.species.number_species), dtype=bool))
        results = dict(
            candidate=candidate, no_transport_solve=True, profile_direction=direction.tolist(), residual_fd_step=epsilon,
            profile_parameter_labels=list(PROFILE_PARAMETER_ORDER), baseline_profile_values=p0.tolist(),
            source_digest=tasks[0]["source_digest"], ambipolarity_config=cfg.get("ambipolarity", {}),
            rho=rho.tolist(), baseline_selected_Er=np.asarray(er0).tolist(),
            baseline_general_Er=np.asarray(baseline.Er).tolist(),
            baseline_charge_flux_residual=[float(value) if np.isfinite(value) else None for value in residual0],
            baseline_charge_flux_Er_derivative=[float(value) if np.isfinite(value) else None for value in er_derivative],
            implicit_initial_Er_rate=[float(value) if np.isfinite(value) else None for value in rate],
            frozen_temperature_species_indices=frozen.tolist(), endpoints=[],
            interpretation="Selected-root differences are compared to local implicit continuation at the baseline root. "
                           "The residual directional derivative is a fixed-root finite difference, not a new full AD benchmark.",
        )
        path = out / "initialization_diagnostic.json"
        path.write_text(json.dumps(results, indent=2, allow_nan=False))
        for task in tasks[1:]:
            p = np.asarray(task["parameter_values"])[profile_indices]
            print(f"[profile-initialization] step={task['step_size']:g}: comparing selected-root entrypoints", flush=True)
            pre = state_from_profiles(p, "off")
            selected = state_from_profiles(p, "jax_selected_root")
            general = _maybe_initialize_er_from_ambipolarity(cfg, runtime, pre)
            observation = compare_initial_roots(er0, selected.Er, general.Er, rate, task["step_size"], rho, audit["annulus"])
            legacy_profiles = {**cfg.get("profiles", {}), **dict(zip(PROFILE_PARAMETER_ORDER, p.tolist(), strict=True))}
            legacy_temperatures = np.asarray(jax.device_get(build_profiles(
                legacy_profiles, runtime.geometry, runtime.species.number_species).temperature)) / 1e3
            observation.update(
                step_size=task["step_size"], profile_values=p.tolist(),
                selected_Er=np.asarray(selected.Er).tolist(), general_Er=np.asarray(general.Er).tolist(),
                legacy_frozen_temperature_max_abs_change_keV=(
                    float(np.max(abs(legacy_temperatures[frozen] - temperatures0[frozen]))) if len(frozen) else 0.),
            )
            results["endpoints"].append(observation)
            path.write_text(json.dumps(results, indent=2, allow_nan=False))
            print("[profile-initialization] " + json.dumps({k: observation[k] for k in (
                "step_size", "selected_minus_general_max_abs", "selected_initial_sign_change_indices",
                "annular_rate_error_l2_relative", "legacy_frozen_temperature_max_abs_change_keV",
            )}), flush=True)
        print(f"Wrote {path}", flush=True)
    return results


def run_forward_task(task):
    """Worker only: rebuild the requested physical point and solve adaptively."""
    import copy
    import jax
    import jax.numpy as jnp

    # Existing benchmark helpers use sibling imports when run as scripts.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from benchmark_transport_reverse_ad_only import _geometry_context_from_config, _benchmark_device_context
    from benchmark_transport_realtime_geometry_forward_fd import _schedule_rollout, _adaptive_rollout_diagnostics
    from NEOPAX._geometry_autodiff import _input_with_boundary_deltas, boundary_param_entries
    from NEOPAX._orchestrator import build_runtime_context
    from NEOPAX._reverse_ad_parameters import PROFILE_PARAMETER_ORDER
    from NEOPAX._reverse_ad_transport import initial_state_for_parameter_vector
    from examples.benchmarks._er_root_loss_audit import RootLossAuditSettings, terminal_metrics
    import tempfile

    cfg = copy.deepcopy(task["resolved_config"])
    labels, values = task["parameter_labels"], task["parameter_values"]
    physical = dict(zip(labels, values, strict=True))
    # The AD benchmark varies initial profile values under a fixed config.
    # In particular, prepare_transport_solver_components builds the prescribed
    # frozen-species temperature from config["profiles"]. Mutating that config
    # here would change additional physics absent from the AD parameter map.
    profile_cfg = cfg.get("profiles", {})
    cfg.setdefault("ambipolarity", {})["er_ambipolar_plot"] = False
    settings = RootLossAuditSettings(**task["settings"])
    original_controls = task.get("objective_suite") == "original"
    if original_controls:
        from examples.benchmarks._er_root_loss_audit import original_objective_settings
        if task.get("original_objective_settings") != original_objective_settings():
            raise ValueError("Original objective constants changed; prepare a matching baseline AD report.")
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
        runtime_initial_er = np.asarray(jax.device_get(state.Er), dtype=float)
        # Match both profile/floor construction and the JAX root primal used
        # by the production AD builder. Agreement at the baseline alone cannot
        # justify substituting a different initializer at perturbed endpoints.
        state = initial_state_for_parameter_vector(
            jnp.asarray([physical[name] for name in PROFILE_PARAMETER_ORDER]), baseline_state=state,
            runtime=runtime, profile_cfg=profile_cfg, config=cfg, initial_er_root_ad=task["initial_er_root_ad"],
        )
        initial_state = {name: np.asarray(jax.device_get(getattr(state, name)), dtype=float).tolist()
                         for name in ("Er", "density", "pressure") if hasattr(state, name)}
        _, _, _, rollout, final = _schedule_rollout(cfg, runtime, state, accepted_step_limit=None)
        er = np.asarray(jax.device_get(final.Er), dtype=float)
        rho = np.asarray(jax.device_get(runtime.geometry.rho_grid), dtype=float)
        metrics = terminal_metrics(er, rho, settings)
        if original_controls:
            from NEOPAX._reverse_ad_transport import bootstrap_current_softmax_abs_scaled
            from examples.benchmarks._er_root_loss_audit import original_transition_values, add_original_metrics
            transition = original_transition_values(jnp.asarray(final.Er), runtime.geometry.rho_grid, settings)
            bootstrap = bootstrap_current_softmax_abs_scaled(final, runtime)
            metrics["target_rho"] = settings.target
            add_original_metrics(metrics, jnp.concatenate((transition, jnp.atleast_1d(bootstrap))))
        return dict(task_digest=task_digest(task), name=task["name"], metrics=metrics,
                    terminal_Er=er.tolist(), terminal_rho=rho.tolist(),
                    final_time=float(jax.device_get(rollout.final_carry.t)),
                    accepted_steps=int(jax.device_get(rollout.accepted_count)),
                    attempts=int(jax.device_get(rollout.attempt_count)), parameter_values=values,
                    initial_state=initial_state, runtime_initial_Er=runtime_initial_er.tolist(),
                    adaptive_schedule=_adaptive_rollout_diagnostics(rollout),
                    profile_contract="ad_initial_profiles_fixed_config_selected_root")


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
    initial_root_changes = None
    if "initial_state" in baseline and "initial_state" in endpoint:
        a = np.asarray(baseline["initial_state"]["Er"])
        b = np.asarray(endpoint["initial_state"]["Er"])
        initial_root_changes = dict(max_abs_change=float(np.max(abs(b - a))),
                                    sign_change_indices=np.flatnonzero(np.sign(a) != np.sign(b)).tolist())
    old, new = before["candidates"][name], after["candidates"][name]
    changes = {key: new[key]-old[key] for key in old.keys() & new.keys() if key.endswith("_cost")}
    return dict(name=task["name"], candidate=name, group=task["group"], step_size=task["step_size"],
                observed_cost_directional_slope=cost_change / task["step_size"],
                predicted_cost_directional_slope=task["prediction"]["predicted_cost_change_per_unit_descent"],
                observed_root_directional_slope=root_change / task["step_size"] if root_change is not None else None,
                predicted_root_directional_slope=task["prediction"].get("predicted_root_change_per_unit_descent"),
                accepted_steps_change=(endpoint["accepted_steps"] - baseline["accepted_steps"])
                    if "accepted_steps" in endpoint and "accepted_steps" in baseline else None,
                cost_change=cost_change, zero_cost_change=changes.get("zero_cost"),
                sign_cost_change=changes.get("sign_cost"), component_cost_changes=changes,
                position_cost_change=position_change, root_change=root_change,
                predicted_cost_change=task["step_size"] * task["prediction"]["predicted_cost_change_per_unit_descent"],
                predicted_position_cost_change=task["step_size"] * rate if rate is not None else None,
                annular_Er_rms_change=rms_change,
                cost_decreased=cost_change < 0,
                inner_mean_Er_change=(after["inner_mean_Er"] - before["inner_mean_Er"]
                                      if "inner_mean_Er" in before and "inner_mean_Er" in after else None),
                inner_positive_fraction_change=(after["inner_positive_fraction"] - before["inner_positive_fraction"]
                                                 if "inner_positive_fraction" in before and "inner_positive_fraction" in after else None),
                wrong_sign_fraction_change=(after["wrong_sign_fraction"] - before["wrong_sign_fraction"]
                                             if "wrong_sign_fraction" in before and "wrong_sign_fraction" in after else None),
                root_position_improved=position_change < 0 if position_change is not None else None,
                cost_decreased_without_position_improvement=(cost_change < 0 and position_change >= 0) if position_change is not None else None,
                root_created=before["nearest_downcrossing"] is None and after["nearest_downcrossing"] is not None,
                root_lost=before["nearest_downcrossing"] is not None and after["nearest_downcrossing"] is None,
                root_bracket_changed=before["nearest_downcrossing_bracket"] != after["nearest_downcrossing_bracket"],
                initial_root_changes=initial_root_changes,
                baseline_root_error=(before["nearest_downcrossing"] - audit["target"]) if before["nearest_downcrossing"] is not None else None,
                endpoint_metrics=after)


def plot_profiles(summary, out, audit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    choices = summary.get("candidates", ("local", "annular"))
    groups = summary.get("groups", ("profiles", "geometry"))
    fig, axes = plt.subplots(len(choices), len(groups), figsize=(5.5*len(groups), 3.5*len(choices)),
                             sharex=True, sharey=True, constrained_layout=True, squeeze=False)
    baseline = summary["baseline"]
    for i, candidate in enumerate(choices):
        for j, group in enumerate(groups):
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
    parser.add_argument("--diagnose-profile-initialization", action="store_true",
                        help="Build one runtime and compare initial-root entrypoints/implicit response; run no transport.")
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
    if args.diagnose_profile_initialization:
        choices = args.candidates.split(",")
        if len(choices) != 1 or choices[0] not in {"local", "annular"}:
            parser.error("Initialization diagnostics require one candidate: --candidates local or annular.")
        try:
            diagnose_profile_initialization(report, args.step_sizes.split(","), choices[0], args.out_dir.resolve())
        except (KeyError, ValueError) as exc:
            parser.error(str(exc))
        return
    try:
        tasks = plan_checks(report, args.step_sizes.split(","), args.candidates.split(","), args.groups.split(","))
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    audit = report["root_loss_audit"]
    summary = dict(ad_report=str(args.ad_report.resolve()), results=[], failures=[],
                   candidates=args.candidates.split(","), groups=args.groups.split(","),
                   profile_contract="ad_initial_profiles_fixed_config_selected_root",
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
