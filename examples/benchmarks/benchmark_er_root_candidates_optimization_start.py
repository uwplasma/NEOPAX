"""Compare new root losses from the usual full-transport optimization start.

Uses the geometry-only example's effective configuration and unoptimized
seed, or its saved optimized geometry. The original seed must have neither
positive Er nor an actual downcrossing in the test annulus. Saved optimized
geometry is observed without requiring the root to be absent.
Physics runs only when this script is executed on the large machine.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_saved_geometry(directory):
    """Read the exported accepted boundary, never reapply its scaled deltas."""
    directory = Path(directory).resolve()
    summary_path = directory / "geometry_phase_summary.json"
    if not summary_path.is_file():
        summary_path = directory / "optimization_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    labels = summary.get("geometry_phase", summary).get("parameter_labels", [])
    if not labels or any(not label.startswith(("RBC:", "ZBS:")) for label in labels):
        raise ValueError("The selected result must contain only geometry DOFs.")
    recorded = summary.get("accepted_geometry_input")
    if recorded is not None:
        path = Path(recorded)
        candidates = [path if path.is_absolute() else directory / path, directory / path.name]
    else:
        candidates = [directory / "input.QI_neopax_database_full_transport_net_power_optimized"]
        modes = summary.get("max_mode_schedule", summary.get("geometry_max_mode_schedule", []))
        if modes:
            candidates.append(directory / f"input.QI_neopax_database_full_transport_net_power_stage_m{modes[-1]}")
    accepted = next((path.resolve() for path in candidates if path.is_file()), None)
    if accepted is None:
        raise FileNotFoundError("The geometry result has no exported accepted VMEC input. "
                                "The original seed must not be substituted for the optimized boundary.")
    modes = summary.get("max_mode_schedule", summary.get("geometry_max_mode_schedule", []))
    if not modes:
        raise ValueError("The geometry result must record its max-mode schedule.")
    metadata = dict(directory=str(directory), summary_path=str(summary_path),
                    summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                    accepted_geometry_input=str(accepted),
                    accepted_geometry_sha256=hashlib.sha256(accepted.read_bytes()).hexdigest())
    return summary, accepted, metadata


def prepare_case(cli, *, example=None, prepare_config=None):
    """Resolve the real example defaults; never build a runtime here."""
    if example is None:
        from examples.optimization import (
            optimize_geometry_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport as example,
        )
    if prepare_config is None:
        from NEOPAX.optimization import _prepare_full_transport_config as prepare_config
    defaults = example.parser().parse_args([])
    saved_summary, saved_metadata = None, None
    if getattr(cli, "geometry_result_dir", None) is not None:
        saved_summary, accepted, saved_metadata = load_saved_geometry(cli.geometry_result_dir)
        defaults.seed_input = accepted
        saved_config = saved_summary.get("config", {})
        neo = saved_config.get("neoclassical", {})
        grid = saved_summary.get("database_grid", [neo.get("ntx_scan_n_theta", defaults.database_n_theta),
            neo.get("ntx_scan_n_zeta", defaults.database_n_phi), neo.get("ntx_scan_n_xi", defaults.database_n_xi)])
        for name, value in zip(("database_n_theta", "database_n_phi", "database_n_xi"), grid, strict=True):
            setattr(defaults, name, int(value))
        defaults.transport_max_steps = int(saved_summary.get("transport_max_steps",
            saved_config.get("transport_solver", {}).get("max_steps", defaults.transport_max_steps)))
        defaults.max_reverse_accepted_steps = int(saved_summary.get("max_reverse_accepted_steps", defaults.max_reverse_accepted_steps))
    for name in ("seed_input", "database_n_theta", "database_n_phi", "database_n_xi",
                 "transport_max_steps", "max_reverse_accepted_steps"):
        value = getattr(cli, name, None)
        if value is not None:
            setattr(defaults, name, value)
    base_config = (copy.deepcopy(saved_summary["config"]) if saved_summary is not None and saved_summary.get("config")
                   else example.transport_config(defaults))
    base_config.setdefault("neoclassical", {}).update(ntx_scan_n_theta=defaults.database_n_theta,
        ntx_scan_n_zeta=defaults.database_n_phi, ntx_scan_n_xi=defaults.database_n_xi)
    base_config.setdefault("transport_solver", {})["max_steps"] = defaults.transport_max_steps
    base_config["transport_solver"]["radau_jacobian_reuse_mode"] = "legacy"
    if saved_summary is not None and "transport_final_time" in saved_summary:
        base_config["transport_solver"]["t_final"] = float(saved_summary["transport_final_time"])
    cfg = prepare_config(base_config, device=example.SOLVER_DEVICE)
    cfg = copy.deepcopy(cfg)
    geom, neo = cfg.setdefault("geometry", {}), cfg.setdefault("neoclassical", {})
    geom["vmec_input_file"] = str(defaults.seed_input.resolve())
    geom["vmec_param_delta"] = 0.
    # The benchmark builder reads these dimensions for its derivative payload;
    # the runtime reads ntx_scan_*. Keep both on the actual optimizer mesh.
    for suffix, value in (("theta", defaults.database_n_theta), ("zeta", defaults.database_n_phi),
                          ("xi", defaults.database_n_xi)):
        neo["ntx_exact_n_" + suffix] = int(value)
    cfg.setdefault("ambipolarity", {})["er_ambipolar_plot"] = False
    general = cfg.setdefault("general", {})
    if general.get("device") == "default":
        general["device"] = "auto"
    if cli.t_final is not None:
        cfg["transport_solver"]["t_final"] = cli.t_final
    if cfg["transport_solver"].get("stop_after_accepted_steps") is not None:
        raise ValueError("This experiment must reach the full configured final time.")
    settings = dict(target=float(defaults.er_transition_rho_target),
                    inner_radius=float(defaults.er_transition_rho_min if cli.inner_radius is None else cli.inner_radius),
                    outer_radius=float(defaults.er_transition_rho_max),
                    profile_trust_multiplier=3.7, geometry_ess_alpha=float(example.ESS_ALPHA))
    stage = (int(saved_summary.get("max_mode_schedule", saved_summary.get("geometry_max_mode_schedule"))[-1])
             if saved_summary is not None else int(example.max_mode_schedule_values()[0]))
    metadata = dict(seed_input=str(defaults.seed_input.resolve()),
                    seed_sha256=hashlib.sha256(defaults.seed_input.read_bytes()).hexdigest(),
                    config_source=str(example.TRANSPORT_CONFIG),
                    t_final=cfg["transport_solver"]["t_final"],
                    transport_max_steps=int(defaults.transport_max_steps),
                    max_reverse_accepted_steps=int(defaults.max_reverse_accepted_steps),
                    database_mesh=[defaults.database_n_theta, defaults.database_n_phi, defaults.database_n_xi],
                    max_mode=stage, mboz=int(example.QI_MBOZ), nboz=int(example.QI_NBOZ),
                    surfaces=[float(v) for v in example.SURFACES], families=example.GEOMETRY_FAMILIES)
    if saved_metadata is not None:
        metadata.update(case_kind="saved_geometry_result", saved_geometry_result=saved_metadata)
    return cfg, settings, metadata


def check_no_electron_region(metrics, final_time, expected_time):
    if not np.isclose(final_time, expected_time, rtol=0, atol=1e-10):
        raise ValueError("Baseline transport did not reach the configured final time.")
    if metrics["nearest_downcrossing"] is not None or metrics["annular_max_Er"] > 0:
        raise ValueError("Baseline already has positive Er or a downcrossing in the test annulus. "
                         "This is not the requested no-electron-root case; inspect baseline_forward_observation.json.")


def run_ad(cli, out):
    """Worker: one baseline AD, with a terminal gate before the reverse sweep."""
    import jax.numpy as jnp
    sys.path.insert(0, str(ROOT / "examples/benchmarks"))
    import benchmark_transport_reverse_ad_only as benchmark
    from NEOPAX import optimization as opt
    from NEOPAX._orchestrator import build_runtime_context
    from NEOPAX._reverse_ad_parameters import PROFILE_PARAMETER_ORDER
    from examples.benchmarks._er_root_loss_audit import (
        ROOT_CREATION_OBJECTIVES, RootLossAuditSettings, experiment_inputs, terminal_metrics,
    )
    from examples.optimization.optimize_profiles_er_root_candidate import ad_arguments, write_json

    cfg, raw_settings, metadata = prepare_case(cli)
    settings = RootLossAuditSettings(**raw_settings)
    context = opt.build_geometry_autodiff_context(
        metadata["seed_input"], param_family="RBC", param_m=1, param_n=0,
        mboz=metadata["mboz"], nboz=metadata["nboz"], surface_s=tuple(metadata["surfaces"]))
    if cli.geometry_parameters:
        specs = benchmark._geometry_param_specs_from_parameter_name(cli.geometry_parameters)
    else:
        parameterization = opt.vmex_boundary_parameterization(
            context, max_mode=metadata["max_mode"], families=metadata["families"],
            scale_mode="ess", ess_alpha=settings.geometry_ess_alpha)
        specs = tuple((spec.family, spec.m, spec.n) for spec in parameterization.specs)
    geometry_labels = [f"vmec:{family}:{m}:{n}" for family, m, n in specs]
    profiles = [benchmark._profile_cfg_scalar_value(cfg["profiles"], name) for name in PROFILE_PARAMETER_ORDER]
    values = profiles + [0.] * len(specs)
    experiment = experiment_inputs(cfg, values, "jax_selected_root")
    template = dict(parameter_order=[*PROFILE_PARAMETER_ORDER, *geometry_labels], reverse_segment_length=50,
                    root_loss_audit=dict(settings=raw_settings, experiment=experiment))
    args = ad_arguments(template)
    args.max_reverse_accepted_steps = metadata["max_reverse_accepted_steps"]
    args.config = metadata["config_source"]
    metadata.update(profile_parameters=dict(zip(PROFILE_PARAMETER_ORDER, profiles, strict=True)),
                    geometry_parameters=geometry_labels, settings=raw_settings,
                    resolved_config=experiment["resolved_config"])
    write_json(out / "case_manifest.json", metadata)
    print("[optimization-start-case] " + json.dumps({k: v for k, v in metadata.items() if k != "resolved_config"}), flush=True)

    checked = []

    def observe(snapshot):
        metrics = terminal_metrics(snapshot["Er"], snapshot["rho"], settings)
        write_json(out / "baseline_forward_observation.json", dict(
            final_time=snapshot["final_time"], terminal_Er=snapshot["Er"].tolist(),
            terminal_rho=snapshot["rho"].tolist(), metrics=metrics))
        print("[optimization-start-baseline] " + json.dumps(metrics), flush=True)
        if metadata.get("case_kind") == "saved_geometry_result":
            if not np.isclose(snapshot["final_time"], metadata["t_final"], rtol=0, atol=1e-10):
                raise ValueError("Baseline transport did not reach the configured final time.")
        else:
            check_no_electron_region(metrics, snapshot["final_time"], metadata["t_final"])
        checked.append(metrics)

    with benchmark._benchmark_device_context(cfg):
        runtime, state = build_runtime_context(cfg)
        result = benchmark._run_realtime_geometry_optimization_api_smoke(
            args=args, config=cfg, geometry_context=context, baseline_values=jnp.asarray(values),
            baseline_runtime=runtime, baseline_state=state, profile_cfg=cfg["profiles"],
            neoclassical_cfg=cfg["neoclassical"], root_audit_objective_labels=ROOT_CREATION_OBJECTIVES,
            terminal_observer_check=observe, write_report=False)
    if not checked:
        raise RuntimeError("The baseline terminal observer was not called; absence of an electron region is unverified.")
    result["optimization_start_case"] = metadata
    result["baseline_case_verified"] = True
    result["baseline_no_electron_region_verified"] = (
        checked[-1]["nearest_downcrossing"] is None and checked[-1]["annular_max_Er"] <= 0)
    write_json(out / "ad_summary.json", result)


def parser():
    out = argparse.ArgumentParser(description=__doc__)
    out.add_argument("--out-dir", type=Path, default=ROOT / "outputs/root_candidates_optimization_start")
    out.add_argument("--ad-only", action="store_true", help="Compute/inspect the shared AD comparison before finite moves.")
    out.add_argument("--ad-report", type=Path, help="Reuse this experiment's saved AD result; do not recompute AD.")
    out.add_argument("--step-sizes", default="0.001")
    out.add_argument("--groups", default="profiles,geometry")
    source = out.add_mutually_exclusive_group()
    source.add_argument("--seed-input", type=Path)
    source.add_argument("--geometry-result-dir", type=Path, help="Use the exported geometry-only result and its saved settings.")
    out.add_argument("--t-final", type=float, help="Default is the optimization's t_final=2, not the old benchmark's 20.")
    out.add_argument("--inner-radius", type=float, help="Default is the optimization's root window lower bound, 0.05.")
    out.add_argument("--geometry-parameters", help="Optional harmonic subset; default matches the optimizer's first mode stage.")
    for name in ("database-n-theta", "database-n-phi", "database-n-xi", "transport-max-steps", "max-reverse-accepted-steps"):
        out.add_argument("--" + name, type=int)
    out.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return out


def main(argv=None):
    command_parser = parser()
    cli = command_parser.parse_args(argv)
    override_names = ("seed_input", "database_n_theta", "database_n_phi", "database_n_xi", "t_final",
                      "inner_radius", "geometry_parameters", "transport_max_steps", "max_reverse_accepted_steps", "geometry_result_dir")
    if cli.ad_report is not None and any(getattr(cli, name) is not None for name in override_names):
        command_parser.error("Physics overrides cannot be combined with --ad-report; its saved configuration defines the case.")
    for name in ("database_n_theta", "database_n_phi", "database_n_xi", "t_final", "transport_max_steps", "max_reverse_accepted_steps"):
        value = getattr(cli, name)
        if value is not None and (not np.isfinite(value) or value <= 0):
            command_parser.error(f"--{name.replace('_', '-')} must be finite and positive.")
    try:
        steps = [float(v) for v in cli.step_sizes.split(",")]
        if not steps or any(not np.isfinite(v) or v <= 0 for v in steps):
            raise ValueError
    except ValueError:
        command_parser.error("--step-sizes must contain finite positive numbers.")
    if not set(cli.groups.split(",")) <= {"profiles", "geometry"}:
        command_parser.error("--groups must select profiles and/or geometry.")
    out = cli.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if cli.worker:
        run_ad(cli, out)
        return
    if cli.ad_report is None:
        report_path = out / "ad_summary.json"
        if report_path.exists():
            raise SystemExit("AD report already exists; pass --ad-report to reuse it or choose a new output directory.")
        # The same CLI, in a child process, releases the baseline AD executables
        # before independent forward tests. This is a diagnostic, not an optimizer.
        command = [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv), "--worker"]
        with (out / "ad.log").open("w") as log:
            print(f"[optimization-start-case] baseline/AD log={out / 'ad.log'}", flush=True)
            with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, bufsize=1) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                returncode = process.wait()
        if returncode:
            raise SystemExit(f"Baseline/AD failed; see {out / 'ad.log'} and baseline_forward_observation.json if present.")
    else:
        report_path = cli.ad_report.resolve()
    report = json.loads(report_path.read_text())
    saved_geometry = report.get("optimization_start_case", {}).get("case_kind") == "saved_geometry_result"
    if not report.get("baseline_no_electron_region_verified") and not (saved_geometry and report.get("baseline_case_verified")):
        raise SystemExit("The report has not verified the requested no-electron-root baseline.")
    for item in report["root_loss_audit"]["comparisons"]:
        print("[optimization-start-candidate] " + json.dumps(item), flush=True)
    print(f"Saved full-time AD comparison: {report_path}", flush=True)
    if not cli.ad_only:
        command = [sys.executable, str(ROOT / "examples/benchmarks/check_er_root_candidate_descent.py"),
                   "--ad-report", str(report_path), "--out-dir", str(out / "descent"),
                   "--step-sizes", cli.step_sizes, "--candidates", "local,annular", "--groups", cli.groups]
        raise SystemExit(subprocess.run(command, cwd=ROOT).returncode)


if __name__ == "__main__":
    main()
