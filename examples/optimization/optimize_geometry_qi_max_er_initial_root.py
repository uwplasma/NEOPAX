#!/usr/bin/env python
"""Geometry QI optimization with an initial ambipolar-Er softmax target."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import vmex as vj  # noqa: E402
from vmex import optimize as vmex_opt  # noqa: E402

from NEOPAX import optimization as opt  # noqa: E402


# --------------------------- parameters ------------------------------------
SEED_INPUT = ROOT / "examples" / "inputs" / "input.QI_nfp2_initial"
TRANSPORT_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml"
)
OUT_DIR = ROOT / "outputs" / "geometry_qi_max_er_initial_root_optimization"

SURFACES = np.asarray(
    [1 / 51, 5 / 51, 10 / 51, 15 / 51, 20 / 51, 25 / 51, 30 / 51, 35 / 51, 40 / 51, 45 / 51, 50 / 51],
    dtype=float,
)
QI_MBOZ = 18
QI_NBOZ = 18

MAX_MODE_SCHEDULE = (1, 2)
GEOMETRY_FAMILIES = "RBC,ZBS"
SCALE_MODE = "ess"
ESS_ALPHA = 1.0

ASPECT_TARGET = 10.0
IOTA_TARGET = -0.61
MIRROR_TARGET = 0.25
MAX_ER_TARGET = 30.0

QI_WEIGHT = 1.0
MAXJ_WEIGHT = 0.1
ASPECT_WEIGHT = 1.0
IOTA_WEIGHT = 100.0
MIRROR_WEIGHT = 100.0
MAX_ER_WEIGHT = 1.0

NFEV = 5
FTOL = 1.0e-6
XTOL = 1.0e-10
GEOMETRY_MAX_ITER = None
SOLVER_DEVICE = "default"

MAKE_WOUT_PLOTS = True


# --------------------------- objective functions ---------------------------
qi = opt.geometry.boozer_qi_objective
qi_maxj_1 = opt.geometry.boozer_maxj_objective
softmax_er = opt.transport.softmax_Er


terms = [
    (qi, 0.0, QI_WEIGHT),
    (qi_maxj_1, 0.0, MAXJ_WEIGHT),
    (opt.geometry.vmec_mirror_ratio, MIRROR_TARGET, MIRROR_WEIGHT),
    (opt.geometry.vmec_aspect_ratio, ASPECT_TARGET, ASPECT_WEIGHT),
    (opt.geometry.vmec_iota_mean, IOTA_TARGET, IOTA_WEIGHT),
    (softmax_er, MAX_ER_TARGET, MAX_ER_WEIGHT),
]


# --------------------------- reporting / outputs ---------------------------
def report(tag, problem, x):
    evaluation = problem.evaluate(x)
    residuals = np.asarray(jax.device_get(evaluation.residuals), dtype=float)
    jacobian = np.asarray(jax.device_get(evaluation.jacobian), dtype=float)
    values = {
        label: float(np.asarray(jax.device_get(value), dtype=float))
        for label, value in evaluation.result.objective_values.items()
    }
    print(f"[{tag}] elapsed_s={evaluation.elapsed_s:.3f}")
    for label, value in values.items():
        print(f"  - {label}: value={value:.16e}")
    print(f"  residual_norm={float(np.linalg.norm(residuals)):.6e}")
    print(f"  jacobian_shape={jacobian.shape}")
    return evaluation


def write_outputs(optimized_input):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_copy = OUT_DIR / SEED_INPUT.name
    optimized_input_path = OUT_DIR / "input.QI_neopax_geometry_max_er_optimized"
    optimized_input.to_indata(seed_copy)
    optimized_input.to_indata(optimized_input_path)
    print(f"wrote {seed_copy}")
    print(f"wrote {optimized_input_path}")

    eq = vmex_opt.solve_equilibrium(optimized_input)
    wout_path = vj.write_wout(OUT_DIR / "wout_QI_neopax_geometry_max_er_optimized.nc", eq.wout)
    print(f"wrote {wout_path}")
    if MAKE_WOUT_PLOTS:
        for _, path in vj.plot_wout(wout_path, OUT_DIR).items():
            print(f"wrote {path}")


# --------------------------- continuation ladder ----------------------------
def main() -> int:
    active_terms = tuple(term for term in terms if float(term[2]) != 0.0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x = None
    current_input = SEED_INPUT
    optimized_input = None
    last_problem = None
    last_result = None

    for max_mode in MAX_MODE_SCHEDULE:
        print(f"\n===== NEOPAX geometry QI + max-Er stage, max_mode={max_mode} =====", flush=True)
        problem = opt.geometry_initial_er_root_only_least_squares_problem(
            TRANSPORT_CONFIG,
            active_terms,
            vmec_input=current_input,
            max_mode=max_mode,
            include_profiles=False,
            families=GEOMETRY_FAMILIES,
            scale_mode=SCALE_MODE,
            ess_alpha=ESS_ALPHA,
            mboz=QI_MBOZ,
            nboz=QI_NBOZ,
            surfaces=tuple(float(s) for s in SURFACES),
            geometry_max_iter=GEOMETRY_MAX_ITER,
            geometry_solver_device=SOLVER_DEVICE,
            device=SOLVER_DEVICE,
        )
        if x is None or len(x) != problem.parameter_count:
            x = np.asarray(jax.device_get(problem.x0), dtype=float)
        print(
            f"[setup] parameter_count={problem.parameter_count} "
            f"parameters={list(problem.parameter_labels)}",
            flush=True,
        )
        report("initial", problem, x)
        last_result = opt.least_squares(
            problem,
            max_nfev=NFEV,
            ftol=FTOL,
            xtol=XTOL,
            verbose=1,
        )
        x = np.asarray(last_result.x, dtype=float)
        report(f"QI + max-Er stage {max_mode}", problem, x)
        optimized_input = problem.input_from_scaled_parameters(x)
        stage_input = OUT_DIR / f"input.QI_neopax_geometry_max_er_stage_m{max_mode}"
        optimized_input.to_indata(stage_input)
        print(f"wrote {stage_input}")
        current_input = stage_input
        x = None
        last_problem = problem

    if optimized_input is None or last_problem is None or last_result is None:
        raise RuntimeError("No optimization stage was executed.")

    summary = {
        "seed_input": str(SEED_INPUT),
        "transport_config": str(TRANSPORT_CONFIG),
        "max_mode_schedule": list(MAX_MODE_SCHEDULE),
        "parameter_labels": list(last_problem.parameter_labels),
        "x": np.asarray(last_result.x, dtype=float).tolist(),
        "cost": float(last_result.cost),
        "optimality": float(last_result.optimality),
        "status": int(last_result.status),
        "message": str(last_result.message),
    }
    summary_path = OUT_DIR / "optimization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")
    write_outputs(optimized_input)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
