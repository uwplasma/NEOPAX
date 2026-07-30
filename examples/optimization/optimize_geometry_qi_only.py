#!/usr/bin/env python
"""Geometry-only QI optimization using NEOPAX reverse-AD internals."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import vmex as vj  # noqa: E402
from vmex import optimize as vmex_opt  # noqa: E402

from NEOPAX import optimization as opt  # noqa: E402


# --------------------------- parameters ------------------------------------
SEED_INPUT = ROOT / "examples" / "inputs" / "input.QI_nfp2_initial"
OUT_DIR = ROOT / "outputs" / "geometry_qi_only_optimization"

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
IOTA_FLOOR = 0.15
MIRROR_TARGET = 0.25

QI_WEIGHT = 1.0
MAXJ_WEIGHT = 0.0
ASPECT_WEIGHT = 1.0
IOTA_WEIGHT = 100.0
MIRROR_WEIGHT = 100.0

QI_NFEV = 10
FTOL = 1.0e-6
XTOL = 1.0e-10
GEOMETRY_MAX_ITER = None
SOLVER_DEVICE = "default"

MAKE_WOUT_PLOTS = True
MAKE_J_POLAR_PLOTS = True


# --------------------------- objective functions ---------------------------
qi = opt.geometry.boozer_qi_objective
qi_maxj_1 = opt.geometry.boozer_maxj_objective


def iota_shortfall_value(mean_iota):
    return jnp.maximum(IOTA_FLOOR - jnp.abs(mean_iota), 0.0)


iota_shortfall = opt.transformed_geometry_objective(
    opt.geometry.vmec_iota_mean,
    iota_shortfall_value,
    label="iota_shortfall",
)


def mirror_penalization_value(mirror_ratio):
    return jnp.maximum(mirror_ratio - MIRROR_TARGET, 0.0)


mirror_penalization = opt.transformed_geometry_objective(
    opt.geometry.vmec_mirror_ratio,
    mirror_penalization_value,
    label="mirror_penalization",
)


qi_terms = [
    (qi, 0.0, QI_WEIGHT),
    # (qi_maxj_1, 0.0, MAXJ_WEIGHT),
    (mirror_penalization, 0.0, MIRROR_WEIGHT),
    (opt.geometry.vmec_aspect_ratio, ASPECT_TARGET, ASPECT_WEIGHT),
    # (iota_shortfall, 0.0, IOTA_WEIGHT),
    (opt.geometry.vmec_iota_mean, IOTA_TARGET, IOTA_WEIGHT),
]


# --------------------------- reporting / plots ------------------------------
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


def plot_j_polar_contours(eq, out_dir, *, lambda_samples=(0.1, 0.3, 0.5, 0.7, 0.9)):
    try:
        import matplotlib.pyplot as plt
        from vmex.core.omnigenity_j import JInvariantQIResidual
    except Exception as exc:
        print(f"skipping J-polar plots: {exc}")
        return

    objective = JInvariantQIResidual(
        SURFACES,
        mboz=QI_MBOZ,
        nboz=QI_NBOZ,
    )
    try:
        out = objective.compute_state(eq.state, eq.runtime)
    except Exception as exc:
        print(f"skipping J-polar plots: {exc}")
        return

    alpha = np.asarray(out["alpha"], dtype=float)
    surfaces = np.asarray(out["surfaces"], dtype=float)
    ji = np.asarray(out["ji"], dtype=float)
    jc = np.asarray(out["jc"], dtype=float)
    lambda_grid = np.power(
        np.arange(objective.n_bounce, dtype=float) / max(objective.n_bounce - 1, 1),
        objective.p_lambda,
    )

    theta = np.concatenate([alpha, alpha[:1] + 2.0 * np.pi])
    theta_grid, radius_grid = np.meshgrid(theta, surfaces, indexing="xy")
    sample_idx = sorted(
        {
            int(np.clip(round(lam * (objective.n_bounce - 1)), 0, objective.n_bounce - 1))
            for lam in lambda_samples
        }
    )

    for name, data in (("ji", ji), ("jc", jc)):
        for idx in sample_idx:
            values = data[:, :, idx]
            values_periodic = np.concatenate([values, values[:, :1]], axis=1)
            fig = plt.figure(figsize=(12, 5))
            ax_polar = fig.add_subplot(1, 2, 1, projection="polar")
            contour = ax_polar.contourf(theta_grid, radius_grid, values_periodic, levels=32, cmap="viridis")
            ax_polar.set_title(f"{name.upper()} polar contour at lambda={lambda_grid[idx]:.2f}")
            ax_polar.set_ylim(float(surfaces.min()), float(surfaces.max()))
            fig.colorbar(contour, ax=ax_polar, pad=0.12, label=name.upper())

            ax_lines = fig.add_subplot(1, 2, 2)
            for isurf, surface in enumerate(surfaces):
                ax_lines.plot(alpha, data[isurf, :, idx], label=f"s={surface:.2f}")
            ax_lines.set_title(f"{name.upper()} vs alpha across surfaces")
            ax_lines.set_xlabel("alpha")
            ax_lines.set_ylabel(name.upper())
            ax_lines.grid(True, alpha=0.3)
            ax_lines.legend(loc="best", ncol=2, fontsize=8)
            fig.tight_layout()
            path = out_dir / f"{name}_polar_lambda_{idx:02d}.png"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {path}")


def write_outputs(optimized_input):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_copy = OUT_DIR / SEED_INPUT.name
    optimized_input_path = OUT_DIR / "input.QI_neopax_geometry_optimized"
    optimized_input.to_indata(seed_copy)
    optimized_input.to_indata(optimized_input_path)
    print(f"wrote {seed_copy}")
    print(f"wrote {optimized_input_path}")

    eq = vmex_opt.solve_equilibrium(optimized_input)
    wout_path = vj.write_wout(OUT_DIR / "wout_QI_neopax_geometry_optimized.nc", eq.wout)
    print(f"wrote {wout_path}")
    if MAKE_WOUT_PLOTS:
        for _, path in vj.plot_wout(wout_path, OUT_DIR).items():
            print(f"wrote {path}")
    if MAKE_J_POLAR_PLOTS:
        plot_j_polar_contours(eq, OUT_DIR)


# --------------------------- continuation ladder ----------------------------
def main() -> int:
    active_terms = tuple(term for term in qi_terms if float(term[2]) != 0.0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x = None
    current_input = SEED_INPUT
    optimized_input = None
    last_problem = None
    last_result = None

    for max_mode in MAX_MODE_SCHEDULE:
        print(f"\n===== NEOPAX geometry-only QI stage, max_mode={max_mode} =====", flush=True)
        problem = opt.geometry_least_squares_problem(
            current_input,
            active_terms,
            max_mode=max_mode,
            parameters=None,
            families=GEOMETRY_FAMILIES,
            scale_mode=SCALE_MODE,
            ess_alpha=ESS_ALPHA,
            mboz=QI_MBOZ,
            nboz=QI_NBOZ,
            surfaces=tuple(float(s) for s in SURFACES),
            max_iter=GEOMETRY_MAX_ITER,
            solver_device=SOLVER_DEVICE,
        )
        if x is None or len(x) != problem.parameter_count:
            x = np.zeros((problem.parameter_count,), dtype=float)
        print(
            f"[setup] parameter_count={problem.parameter_count} "
            f"parameters={list(problem.parameter_labels)}",
            flush=True,
        )
        report("initial", problem, x)
        last_result = opt.least_squares(
            problem,
            max_nfev=QI_NFEV,
            ftol=FTOL,
            xtol=XTOL,
            verbose=1,
        )
        x = np.asarray(last_result.x, dtype=float)
        report(f"QI stage {max_mode}", problem, x)
        optimized_input = problem.input_from_scaled_parameters(x)
        stage_input_path = OUT_DIR / f"input.QI_neopax_geometry_stage_m{max_mode}"
        optimized_input.to_indata(stage_input_path)
        print(f"wrote {stage_input_path}")
        current_input = stage_input_path
        x = None
        last_problem = problem

    summary_path = OUT_DIR / "geometry_qi_only_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "seed_input": str(SEED_INPUT),
        "surfaces": [float(s) for s in SURFACES],
        "mboz": int(QI_MBOZ),
        "nboz": int(QI_NBOZ),
        "terms": [(getattr(term[0], "label", term[0].label), float(term[1]), float(term[2])) for term in active_terms],
        "parameter_labels": [] if last_problem is None else list(last_problem.parameter_labels),
        "final_stage_x_scaled": [] if last_result is None else np.asarray(last_result.x, dtype=float).tolist(),
        "cost": None if last_result is None else float(last_result.cost),
        "optimality": None if last_result is None else float(last_result.optimality),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")
    if optimized_input is not None:
        write_outputs(optimized_input)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
