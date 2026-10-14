#!/usr/bin/env python
"""Run combined full-transport optimization from a saved geometry-only result.

No geometry-only optimization is run. The saved geometry coordinates are
restored against their original baseline, retaining the established ESS map.
Memory diagnostics are enabled, CUDA command buffers are disabled by default,
and no additional plotting/transport reruns are performed after optimization.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


def _configure_command_buffers(argv):
    # XLA reads these flags when its backend initializes, so configure the
    # standalone process before importing JAX or either optimization example.
    if "--enable-cuda-command-buffers" not in argv:
        flags = os.environ.get("XLA_FLAGS", "").rstrip()
        os.environ["XLA_FLAGS"] = f"{flags} --xla_gpu_enable_command_buffer=".strip()


if __name__ == "__main__":
    _configure_command_buffers(sys.argv[1:])

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Match the established VMEX/NEOPAX process initialization order.
import vmex as vj  # noqa: E402,F401
from vmex import optimize as vmex_opt  # noqa: E402,F401
from NEOPAX import optimization as opt  # noqa: E402
from examples.optimization import (  # noqa: E402
    optimize_geometry_profiles_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_simple_least_squares
    as combined,
    optimize_geometry_warm_start_profiles_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport
    as warm,
)


def parser():
    out = combined.parser()
    out.description = __doc__
    out.set_defaults(
        seed_input=None, out_dir=ROOT / "outputs" / "combined_from_geometry_result",
        database_n_theta=None, database_n_phi=None, database_n_xi=None,
        profile_trust_multiplier=1.0,
        max_er=None, root_objectives=None, transition_location_objective=None,
        bootstrap_penalty=None, net_power=None,
    )
    out.add_argument("--geometry-result-dir", type=Path, required=True,
                     help="Output directory containing a geometry-only optimization summary.")
    out.add_argument("--diagnose-memory", action=argparse.BooleanOptionalAction, default=True)
    out.add_argument("--enable-cuda-command-buffers", action="store_true",
                     help="Enable CUDA graphs; the default avoids the reported graph-instantiation OOM.")
    return out


def load_geometry_result(directory):
    """Read standalone geometry results or the geometry phase of a warm run."""

    directory = Path(directory).resolve()
    summary_path = directory / "geometry_phase_summary.json"
    if not summary_path.is_file():
        summary_path = directory / "optimization_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    phase = summary.get("geometry_phase", summary)
    modes = tuple(summary.get("geometry_max_mode_schedule", summary.get("max_mode_schedule", ())))
    if not modes:
        raise ValueError("The saved geometry result has no max-mode schedule.")
    labels = tuple(phase["parameter_labels"])
    if not labels or any(not label.startswith(("RBC:", "ZBS:")) for label in labels):
        raise ValueError("The saved result must contain only RBC/ZBS geometry DoFs.")
    coordinates = np.asarray(phase.get("x_scaled", phase.get("x")), dtype=float)
    if coordinates.shape != (len(labels),) or not np.all(np.isfinite(coordinates)):
        raise ValueError("The saved geometry coordinate vector has an invalid shape or value.")
    baseline = phase.get("geometry_coordinate_baseline_input")
    if baseline is None:
        baseline = summary.get("geometry_coordinate_baseline_input")
    if baseline is None:
        baseline = (
            summary["seed_input"] if len(modes) == 1 else str(
                directory / f"input.QI_neopax_database_full_transport_net_power_stage_m{modes[-2]}"
            )
        )
    return SimpleNamespace(
        directory=directory, summary_path=summary_path, summary=summary,
        labels=labels, coordinates=coordinates, max_mode=int(modes[-1]),
        baseline_input=Path(baseline),
        scales=phase.get("coordinate_scales"),
    )


def configure_saved_settings(args, result):
    """Use saved settings unless the caller explicitly overrides a switch/grid."""

    geometry = combined.geometry_example
    config = result.summary.get("config", {})
    neo = config.get("neoclassical", {})
    grid = result.summary.get("database_grid", [
        neo.get("ntx_scan_n_theta", geometry.DATABASE_N_THETA),
        neo.get("ntx_scan_n_zeta", geometry.DATABASE_N_PHI),
        neo.get("ntx_scan_n_xi", geometry.DATABASE_N_XI),
    ])
    for name, value in zip(("database_n_theta", "database_n_phi", "database_n_xi"), grid, strict=True):
        if getattr(args, name) is None:
            setattr(args, name, int(value))
    switches = result.summary.get("objectives", {})
    for name, key, fallback in (
        ("max_er", "max_er", geometry.USE_MAX_ER_OBJECTIVE),
        ("root_objectives", "root_objectives", geometry.USE_ER_TRANSITION_OBJECTIVES),
        ("transition_location_objective", "transition_location", geometry.USE_ER_TRANSITION_LOCATION_OBJECTIVE),
        ("bootstrap_penalty", "bootstrap_penalty", geometry.USE_BOOTSTRAP_PENALTY),
        ("net_power", "net_power", geometry.USE_NET_POWER_OBJECTIVE),
    ):
        if getattr(args, name) is None:
            setattr(args, name, bool(switches.get(key, fallback)))
    if args.seed_input is None:
        args.seed_input = result.baseline_input


def starting_point(problem, result):
    """Append nominal profiles while preserving saved geometry by label."""

    lookup = dict(zip(problem.parameter_labels, np.asarray(jax.device_get(problem.x_scale)), strict=True))
    geometry_labels = tuple(label for label in problem.parameter_labels if label not in combined.PROFILE_PARAMETERS.split(","))
    if geometry_labels != result.labels:
        raise ValueError("The saved geometry parameter layout differs from the current ESS layout.")
    scales = (
        np.asarray(result.scales, dtype=float) if result.scales is not None
        else np.asarray([lookup[label] for label in result.labels], dtype=float)
    )
    return warm._combined_starting_point(
        SimpleNamespace(parameter_labels=result.labels, x_scale=scales),
        SimpleNamespace(x=result.coordinates), problem,
    )


def main():
    args = parser().parse_args()
    saved = load_geometry_result(args.geometry_result_dir)
    configure_saved_settings(args, saved)
    combined._validate_args(args)
    out_dir = args.out_dir.resolve()
    if out_dir == saved.directory:
        raise ValueError("--out-dir must differ from the saved geometry-result directory.")
    out_dir.mkdir(parents=True, exist_ok=True)
    config = (
        copy.deepcopy(saved.summary["config"]) if "config" in saved.summary
        else combined.geometry_example.transport_config(args)
    )
    config.setdefault("neoclassical", {}).update(
        ntx_scan_n_theta=args.database_n_theta,
        ntx_scan_n_zeta=args.database_n_phi,
        ntx_scan_n_xi=args.database_n_xi,
    )
    problem = combined._build_problem(
        args, config=config, vmec_input=args.seed_input.resolve(),
        max_mode=saved.max_mode,
        physical_pitches=combined.geometry_example.PHYSICAL_J_PITCHES,
    )
    problem = combined.TrialSavingProblem(problem, out_dir / "combined_inputs")
    if args.diagnose_memory:
        problem = warm.MemoryDiagnosticProblem(problem, "combined_from_saved_geometry")
    x0 = starting_point(problem, saved)
    problem.input_from_scaled_parameters(x0).to_indata(out_dir / "input.QI_combined_initial")
    print(f"[setup] loaded_geometry_result={saved.summary_path} max_mode={saved.max_mode} "
          f"parameter_count={problem.parameter_count} geometry_stage=skipped "
          f"XLA_FLAGS={os.environ.get('XLA_FLAGS', '')}", flush=True)
    initial = combined.report("combined initial at saved geometry", problem, x0)
    options = combined._least_squares_options(
        args, initial, combined.scaled_bounds(problem),
        combined.optimizer_trust_region_x_scale(problem, args.profile_trust_multiplier),
    )
    result = opt.least_squares(problem, x0=x0, **options)
    profiles = combined.profile_parameter_values(problem, result.x)
    problem.input_from_scaled_parameters(result.x).to_indata(out_dir / "input.QI_combined_optimized")
    (out_dir / "profile_parameters_optimized.json").write_text(json.dumps(profiles, indent=2), encoding="utf-8")
    summary = {
        "geometry_result": str(saved.summary_path), "geometry_baseline_input": str(args.seed_input.resolve()),
        "parameter_labels": list(problem.parameter_labels), "x0": x0.tolist(),
        "x": np.asarray(result.x).tolist(), "initial_cost": warm._cost(initial),
        "cost": float(result.cost), "nfev": int(result.nfev),
        "status": int(result.status), "message": str(result.message),
        "optimized_physical_profiles": profiles, "XLA_FLAGS": os.environ.get("XLA_FLAGS", ""),
    }
    (out_dir / "optimization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[combined result] initial_cost={summary['initial_cost']:.8e} cost={result.cost:.8e} "
          f"nfev={result.nfev} status={result.status} wrote={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
