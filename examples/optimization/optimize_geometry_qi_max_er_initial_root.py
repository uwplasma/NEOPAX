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
MAKE_J_POLAR_PLOTS = True
SAVE_INITIAL_ER_PROFILE_EARLY = False


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
def iteration_diagnostics(evaluation):
    values = {
        label: float(np.asarray(jax.device_get(value), dtype=float))
        for label, value in evaluation.result.objective_values.items()
    }
    residuals = np.asarray(jax.device_get(evaluation.residuals), dtype=float)
    residual_lookup = {
        label: float(residuals[i])
        for i, label in enumerate(evaluation.result.residual_labels)
    }

    def value(*labels):
        for label in labels:
            if label in values:
                return values[label]
        return np.nan

    def residual(*labels):
        for label in labels:
            if label in residual_lookup:
                return residual_lookup[label]
        return np.nan

    er_residual = residual("transport:softmax_Er", "softmax_Er")
    er_cost = 0.5 * er_residual * er_residual
    return (
        f"aspect_ratio={value('geometry:vmec_aspect_ratio', 'vmec_aspect_ratio'):.8e} "
        f"iota_mean={value('geometry:vmec_iota_mean', 'vmec_iota_mean'):.8e} "
        f"mirror_ratio={value('geometry:vmec_mirror_ratio', 'vmec_mirror_ratio'):.8e} "
        f"magnetic_well={value('geometry:vmec_magnetic_well', 'vmec_magnetic_well'):.8e} "
        f"qi_cost={value('geometry:boozer_qi_objective', 'boozer_qi_objective'):.8e} "
        f"maxJ_cost={value('geometry:boozer_maxj_objective', 'boozer_maxj_objective'):.8e} "
        f"softmax_Er={value('transport:softmax_Er', 'softmax_Er'):.8e} "
        f"Er_residual={er_residual:.8e} "
        f"Er_cost={er_cost:.8e}"
    )


def report(tag, problem, x):
    evaluation = problem.evaluate(x)
    residuals = np.asarray(jax.device_get(evaluation.residuals), dtype=float)
    jacobian = np.asarray(jax.device_get(evaluation.jacobian), dtype=float)
    values = {
        label: float(np.asarray(jax.device_get(value), dtype=float))
        for label, value in evaluation.result.objective_values.items()
    }
    print(f"[{tag}] elapsed_s={evaluation.elapsed_s:.3f} {iteration_diagnostics(evaluation)}")
    for label, value in values.items():
        print(f"  - {label}: value={value:.16e}")
    print(f"  residual_norm={float(np.linalg.norm(residuals)):.6e}")
    print(f"  jacobian_shape={jacobian.shape}")
    return evaluation


def save_er_profile(problem, x, out_dir, label):
    rho, er, finite_mask = problem.initial_er_profile_from_scaled_parameters(x)
    rho_np = np.asarray(jax.device_get(rho), dtype=float)
    er_np = np.asarray(jax.device_get(er), dtype=float)
    finite_np = np.asarray(jax.device_get(finite_mask), dtype=bool)
    csv_path = out_dir / f"initial_er_profile_{label}.csv"
    np.savetxt(
        csv_path,
        np.column_stack([rho_np, er_np, finite_np.astype(float)]),
        delimiter=",",
        header="rho,Er,finite_mask",
        comments="",
    )
    print(f"wrote {csv_path}")
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"skipping Er profile plot: {exc}")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rho_np, er_np, marker="o", linewidth=1.5)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("rho")
    ax.set_ylabel("Er")
    ax.set_title(f"Selected initial ambipolar Er profile ({label})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png_path = out_dir / f"initial_er_profile_{label}.png"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png_path}")


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


def write_geometry_artifacts(input_obj, label):
    artifact_dir = OUT_DIR / label
    artifact_dir.mkdir(parents=True, exist_ok=True)
    input_path = artifact_dir / f"input.QI_neopax_geometry_max_er_{label}"
    input_obj.to_indata(input_path)
    print(f"wrote {input_path}")

    eq = vmex_opt.solve_equilibrium(input_obj)
    wout_path = vj.write_wout(artifact_dir / f"wout_QI_neopax_geometry_max_er_{label}.nc", eq.wout)
    print(f"wrote {wout_path}")
    if MAKE_WOUT_PLOTS:
        for _, path in vj.plot_wout(wout_path, artifact_dir).items():
            print(f"wrote {path}")
    if MAKE_J_POLAR_PLOTS:
        plot_j_polar_contours(eq, artifact_dir)
    return artifact_dir


def write_outputs(optimized_input, initial_input, initial_problem, initial_x, final_problem, final_x):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_copy = OUT_DIR / SEED_INPUT.name
    optimized_input_path = OUT_DIR / "input.QI_neopax_geometry_max_er_optimized"
    initial_input.to_indata(seed_copy)
    optimized_input.to_indata(optimized_input_path)
    print(f"wrote {seed_copy}")
    print(f"wrote {optimized_input_path}")
    initial_dir = write_geometry_artifacts(initial_input, "initial")
    optimized_dir = write_geometry_artifacts(optimized_input, "optimized")
    save_er_profile(initial_problem, initial_x, initial_dir, "initial")
    save_er_profile(final_problem, final_x, optimized_dir, "optimized")


# --------------------------- continuation ladder ----------------------------
def main() -> int:
    active_terms = tuple(term for term in terms if float(term[2]) != 0.0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x = None
    current_input = SEED_INPUT
    optimized_input = None
    initial_input = None
    initial_problem = None
    initial_x = None
    initial_er_profile_saved = False
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
        if initial_problem is None:
            initial_problem = problem
            initial_x = np.asarray(x, dtype=float).copy()
            initial_input = problem.input_from_scaled_parameters(x)
            if SAVE_INITIAL_ER_PROFILE_EARLY and not initial_er_profile_saved:
                initial_dir = OUT_DIR / "initial"
                initial_dir.mkdir(parents=True, exist_ok=True)
                save_er_profile(problem, x, initial_dir, "initial")
                initial_er_profile_saved = True
        report("initial", problem, x)
        last_result = opt.least_squares(
            problem,
            max_nfev=NFEV,
            ftol=FTOL,
            xtol=XTOL,
            verbose=1,
            iteration_reporter=iteration_diagnostics,
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

    if (
        optimized_input is None
        or initial_input is None
        or initial_problem is None
        or initial_x is None
        or last_problem is None
        or last_result is None
    ):
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
    write_outputs(
        optimized_input,
        initial_input,
        initial_problem,
        initial_x,
        last_problem,
        np.asarray(last_result.x, dtype=float),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
