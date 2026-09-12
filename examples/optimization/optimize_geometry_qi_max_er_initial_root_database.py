#!/usr/bin/env python
"""Standalone database-backed geometry QI + maximum-Er optimization.

This is the NTX scan-database counterpart of
``optimize_geometry_qi_max_er_initial_root.py``.  It uses the validated
fresh-payload database initial-root lane; it does not alter the reverse-AD
benchmark evaluator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from NEOPAX import optimization as opt  # noqa: E402
from NEOPAX._orchestrator import load_config  # noqa: E402


# --------------------------- user settings ---------------------------------
SEED_INPUT = ROOT / "examples" / "inputs" / "input.QI_nfp2_initial"
DATABASE_TRANSPORT_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box.toml"
)
OUT_DIR = ROOT / "outputs" / "geometry_qi_max_er_initial_root_database_optimization"

# Change these defaults here, or pass the matching command-line options.
# ``phi`` is NTX's toroidal ``zeta`` coordinate.
DATABASE_N_THETA = 25
DATABASE_N_PHI = 31
DATABASE_N_XI = 64

SURFACES = np.asarray(
    [1 / 51, 5 / 51, 10 / 51, 15 / 51, 20 / 51, 25 / 51, 30 / 51, 35 / 51, 40 / 51, 45 / 51, 51 / 51],
    dtype=float,
)
QI_MBOZ = 18
QI_NBOZ = 18
MAX_MODE_SCHEDULE = 2
GEOMETRY_FAMILIES = "RBC,ZBS"
SCALE_MODE = "ess"
ESS_ALPHA = 1.2

ASPECT_TARGET = 10.0
IOTA_TARGET = -0.61
MIRROR_TARGET = 0.19
MAX_ER_TARGET = 25.0

# These are the corresponding realtime max-Er script weights.
QI_WEIGHT = 1.0
MAXJ_WEIGHT = 0.0001
ASPECT_WEIGHT = 1.0
IOTA_WEIGHT = 1.0
MIRROR_WEIGHT = 100.0
MAX_ER_WEIGHT = 0.5

NFEV = 40
FTOL = 1.0e-6
XTOL = 1.0e-10
SOLVER_DEVICE = "default"
REVERSE_STAGE_MODE = "database_root_fresh_payload_experiment"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-n-theta", type=int, default=DATABASE_N_THETA)
    parser.add_argument("--database-n-phi", type=int, default=DATABASE_N_PHI)
    parser.add_argument("--database-n-xi", type=int, default=DATABASE_N_XI)
    return parser


def database_config(args: argparse.Namespace) -> dict:
    """Return an in-memory config with this run's database resolution."""

    config = load_config(DATABASE_TRANSPORT_CONFIG)
    neo = config.setdefault("neoclassical", {})
    neo["ntx_scan_n_theta"] = int(args.database_n_theta)
    neo["ntx_scan_n_zeta"] = int(args.database_n_phi)
    neo["ntx_scan_n_xi"] = int(args.database_n_xi)
    return config


qi = opt.geometry.boozer_qi_objective
maxj = opt.geometry.boozer_maxj_objective


def mirror_penalization_value(mirror_ratio):
    return jnp.maximum(mirror_ratio - MIRROR_TARGET, 0.0)


mirror_penalization = opt.transformed_geometry_objective(
    opt.geometry.vmec_mirror_ratio,
    mirror_penalization_value,
    label="mirror_penalization",
)


TERMS = (
    (qi, 0.0, QI_WEIGHT),
    (maxj, 0.0, MAXJ_WEIGHT),
    (mirror_penalization, 0.0, MIRROR_WEIGHT),
    (opt.geometry.vmec_aspect_ratio, ASPECT_TARGET, ASPECT_WEIGHT),
    (opt.geometry.vmec_iota_mean, IOTA_TARGET, IOTA_WEIGHT),
    (opt.transport.softmax_Er, MAX_ER_TARGET, MAX_ER_WEIGHT),
)


def build_problem(config: dict, vmec_input, max_mode: int, args: argparse.Namespace):
    return opt.geometry_initial_er_root_only_least_squares_problem(
        config,
        TERMS,
        vmec_input=vmec_input,
        max_mode=int(max_mode),
        include_profiles=False,
        families=GEOMETRY_FAMILIES,
        scale_mode=SCALE_MODE,
        ess_alpha=ESS_ALPHA,
        mboz=QI_MBOZ,
        nboz=QI_NBOZ,
        surfaces=tuple(float(s) for s in SURFACES),
        n_theta=int(args.database_n_theta),
        n_zeta=int(args.database_n_phi),
        n_xi=int(args.database_n_xi),
        geometry_solver_device=SOLVER_DEVICE,
        device=SOLVER_DEVICE,
        reverse_stage_mode=REVERSE_STAGE_MODE,
    )


def report(tag: str, problem, x) -> None:
    evaluation = problem.evaluate(x)
    values = {
        label: float(np.asarray(jax.device_get(value)))
        for label, value in evaluation.result.objective_values.items()
    }
    residuals = np.asarray(jax.device_get(evaluation.residuals), dtype=float)
    print(f"[{tag}] elapsed_s={evaluation.elapsed_s:.3f} residual_norm={np.linalg.norm(residuals):.6e}")
    for label, value in values.items():
        print(f"  - {label}: {value:.10e}")


def main() -> int:
    args = _parser().parse_args()
    if min(args.database_n_theta, args.database_n_phi, args.database_n_xi) < 1:
        raise ValueError("Database theta, phi, and xi resolutions must all be positive.")
    config = database_config(args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    current_input = SEED_INPUT
    final_problem = final_result = optimized_input = initial_input = None

    for max_mode in (MAX_MODE_SCHEDULE if not np.isscalar(MAX_MODE_SCHEDULE) else (MAX_MODE_SCHEDULE,)):
        print(
            f"\n===== database QI + max-Er, max_mode={max_mode}, "
            f"grid=({args.database_n_theta},{args.database_n_phi},{args.database_n_xi}) =====",
            flush=True,
        )
        problem = build_problem(config, current_input, int(max_mode), args)
        x0 = np.asarray(jax.device_get(problem.x0), dtype=float)
        print(f"[setup] parameter_count={problem.parameter_count} parameters={list(problem.parameter_labels)}")
        if initial_input is None:
            initial_input = problem.input_from_scaled_parameters(x0)
        report("initial", problem, x0)
        result = opt.least_squares(problem, max_nfev=NFEV, ftol=FTOL, xtol=XTOL, verbose=1)
        report(f"stage_m{max_mode}", problem, result.x)
        optimized_input = problem.input_from_scaled_parameters(result.x)
        current_input = OUT_DIR / f"input.QI_neopax_database_max_er_stage_m{max_mode}"
        optimized_input.to_indata(current_input)
        print(f"wrote {current_input}")
        final_problem, final_result = problem, result

    if final_problem is None or final_result is None or optimized_input is None or initial_input is None:
        raise RuntimeError("No optimization stage was executed.")
    initial_input.to_indata(OUT_DIR / SEED_INPUT.name)
    optimized_input.to_indata(OUT_DIR / "input.QI_neopax_database_max_er_optimized")
    summary = {
        "seed_input": str(SEED_INPUT), "database_transport_config": str(DATABASE_TRANSPORT_CONFIG),
        "database_resolution_theta_phi_xi": [args.database_n_theta, args.database_n_phi, args.database_n_xi],
        "reverse_stage_mode": REVERSE_STAGE_MODE, "parameter_labels": list(final_problem.parameter_labels),
        "x": np.asarray(final_result.x, dtype=float).tolist(), "cost": float(final_result.cost),
        "optimality": float(final_result.optimality), "status": int(final_result.status),
        "message": str(final_result.message),
    }
    (OUT_DIR / "optimization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {OUT_DIR / 'optimization_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
