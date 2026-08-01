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
MAXJ_WEIGHT = 0.01
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
MAKE_B_AXIS_PLOTS = True
MAKE_BOOZER_B_CONTOUR_PLOTS = True
MAKE_INITIAL_PLOTS = False


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
    (qi_maxj_1, 0.0, MAXJ_WEIGHT),
    (mirror_penalization, 0.0, MIRROR_WEIGHT),
    (opt.geometry.vmec_aspect_ratio, ASPECT_TARGET, ASPECT_WEIGHT),
    # (iota_shortfall, 0.0, IOTA_WEIGHT),
    (opt.geometry.vmec_iota_mean, IOTA_TARGET, IOTA_WEIGHT),
]


# --------------------------- reporting / plots ------------------------------
def iteration_diagnostics(evaluation):
    values = {
        label: float(np.asarray(jax.device_get(value), dtype=float))
        for label, value in evaluation.result.objective_values.items()
    }

    def value(*labels):
        for label in labels:
            if label in values:
                return values[label]
        return np.nan

    return (
        f"aspect_ratio={value('geometry:vmec_aspect_ratio', 'vmec_aspect_ratio'):.8e} "
        f"iota_mean={value('geometry:vmec_iota_mean', 'vmec_iota_mean'):.8e} "
        f"mirror_ratio={value('geometry:vmec_mirror_ratio', 'vmec_mirror_ratio', 'geometry:mirror_penalization', 'mirror_penalization'):.8e} "
        f"magnetic_well={value('geometry:vmec_magnetic_well', 'vmec_magnetic_well'):.8e} "
        f"qi_cost={value('geometry:boozer_qi_objective', 'boozer_qi_objective'):.8e} "
        f"maxJ_cost={value('geometry:boozer_maxj_objective', 'boozer_maxj_objective'):.8e}"
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
            display_name = r"$\mathcal{J}$" if name == "ji" else r"$J_C$"
            title_name = r"$\mathcal{J}$" if name == "ji" else r"$J_C$"
            fig = plt.figure(figsize=(5.4, 5.8))
            ax_polar = fig.add_subplot(1, 1, 1, projection="polar")
            contour = ax_polar.contourf(theta_grid, radius_grid, values_periodic, levels=40, cmap="plasma")
            ax_polar.set_title(f"Second adiabatic invariant, {title_name}", fontsize=15, pad=20)
            ax_polar.set_ylim(0.0, float(surfaces.max()))
            ax_polar.set_thetagrids(np.arange(0, 360, 45), fontsize=8)
            radial_ticks = np.linspace(0.2, float(surfaces.max()), 5)
            ax_polar.set_rticks(radial_ticks)
            ax_polar.set_yticklabels([f"{tick:.1f}" for tick in radial_ticks], fontsize=8)
            ax_polar.set_rlabel_position(45)
            ax_polar.grid(color="white", linewidth=0.8, alpha=0.45)
            colorbar = fig.colorbar(contour, ax=ax_polar, pad=0.12, shrink=0.78)
            colorbar.set_label(display_name, fontsize=11)
            colorbar.ax.tick_params(labelsize=8)
            fig.text(0.5, 0.035, rf"$\lambda$ = {lambda_grid[idx]:.2f}", ha="center", va="center", fontsize=15)
            fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
            path = out_dir / f"{name}_polar_lambda_{idx:02d}.png"
            fig.savefig(path, dpi=320, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {path}")


def plot_b_on_axis(wout, out_dir, label, *, nphi=256):
    try:
        import matplotlib.pyplot as plt
        from vmex.core.plotting import surface_modB
    except Exception as exc:
        print(f"skipping B-axis plot: {exc}")
        return

    phi = np.linspace(0.0, 2.0 * np.pi / int(wout.nfp), int(nphi))
    theta = np.asarray([0.0], dtype=float)
    b_axis = np.asarray(surface_modB(wout, s_index=0, theta=theta, phi=phi), dtype=float).reshape(-1)
    csv_path = out_dir / f"B_axis_{label}.csv"
    np.savetxt(
        csv_path,
        np.column_stack([phi, b_axis]),
        delimiter=",",
        header="phi,B_axis",
        comments="",
    )
    print(f"wrote {csv_path}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(phi, b_axis, linewidth=1.6)
    ax.set_xlabel("phi")
    ax.set_ylabel("|B| on axis")
    ax.set_title(f"|B| on magnetic axis ({label})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png_path = out_dir / f"B_axis_{label}.png"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png_path}")


def plot_b_on_first_flux_surface(wout, out_dir, label, *, nphi=256):
    try:
        import matplotlib.pyplot as plt
        from vmex.core.plotting import surface_modB
    except Exception as exc:
        print(f"skipping first-flux-surface B plot: {exc}")
        return

    phi = np.linspace(0.0, 2.0 * np.pi / int(wout.nfp), int(nphi))
    theta = np.asarray([0.0], dtype=float)
    ns = int(getattr(wout, "ns", 2))
    s_index = 1 if ns > 1 else 0
    b_surface = np.asarray(surface_modB(wout, s_index=s_index, theta=theta, phi=phi), dtype=float).reshape(-1)
    csv_path = out_dir / f"B_first_flux_surface_{label}.csv"
    np.savetxt(
        csv_path,
        np.column_stack([phi, b_surface]),
        delimiter=",",
        header="phi,B_first_flux_surface_theta0",
        comments="",
    )
    print(f"wrote {csv_path}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(phi, b_surface, linewidth=1.6)
    ax.set_xlabel("phi")
    ax.set_ylabel("|B| at first flux surface, theta=0")
    ax.set_title(f"|B| on first flux surface ({label})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png_path = out_dir / f"B_first_flux_surface_{label}.png"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png_path}")


def plot_boozer_b_contours(wout, out_dir, label, *, ntheta=128, nphi=128):
    try:
        import matplotlib.pyplot as plt
        from vmex.core.plotting import surface_modB
    except Exception as exc:
        print(f"skipping Boozer |B| contour plots: {exc}")
        return

    theta = np.linspace(0.0, 2.0 * np.pi, int(ntheta))
    phi = np.linspace(0.0, 2.0 * np.pi / int(wout.nfp), int(nphi))
    ns = int(getattr(wout, "ns", 2))
    surfaces = (("axis", 0), ("first_flux_surface", 1 if ns > 1 else 0))
    for surface_label, s_index in surfaces:
        b_grid = np.asarray(surface_modB(wout, s_index=s_index, theta=theta, phi=phi), dtype=float)
        csv_path = out_dir / f"B_boozer_contour_{surface_label}_{label}.csv"
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
        np.savetxt(
            csv_path,
            np.column_stack([theta_grid.reshape(-1), phi_grid.reshape(-1), b_grid.reshape(-1)]),
            delimiter=",",
            header="theta,phi,B",
            comments="",
        )
        print(f"wrote {csv_path}")

        finite_b = b_grid[np.isfinite(b_grid)]
        if finite_b.size:
            b_min = float(finite_b.min())
            b_max = float(finite_b.max())
            if not b_max > b_min:
                pad = max(abs(b_min), 1.0) * 1.0e-8
                b_min -= pad
                b_max += pad
            levels = np.linspace(b_min, b_max, 28)
        else:
            levels = 28
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        contour = ax.contour(phi, theta, b_grid, levels=levels, cmap="viridis", linewidths=1.0)
        colorbar = fig.colorbar(contour, ax=ax, pad=0.05)
        colorbar.set_label(r"$|B|$ [T]", fontsize=11)
        colorbar.ax.tick_params(labelsize=9)
        ax.set_xlabel(r"toroidal angle $\phi$", fontsize=11)
        ax.set_ylabel(r"poloidal angle $\theta$", fontsize=11)
        ax.set_yticks([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2.0 * np.pi])
        ax.set_yticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
        ax.set_xlim(float(phi.min()), float(phi.max()))
        ax.set_ylim(float(theta.min()), float(theta.max()))
        title_surface = "magnetic axis" if surface_label == "axis" else surface_label.replace("_", " ")
        ax.set_title(rf"$|B|$ on {title_surface} (one field period)", fontsize=12)
        ax.tick_params(axis="both", labelsize=9)
        fig.tight_layout()
        png_path = out_dir / f"B_boozer_contour_{surface_label}_{label}.png"
        fig.savefig(png_path, dpi=320, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {png_path}")


def write_geometry_artifacts(input_obj, label):
    artifact_dir = OUT_DIR / label
    artifact_dir.mkdir(parents=True, exist_ok=True)
    input_path = artifact_dir / f"input.QI_neopax_geometry_{label}"
    input_obj.to_indata(input_path)
    print(f"wrote {input_path}")

    eq = vmex_opt.solve_equilibrium(input_obj)
    wout_path = vj.write_wout(artifact_dir / f"wout_QI_neopax_geometry_{label}.nc", eq.wout)
    print(f"wrote {wout_path}")
    if MAKE_WOUT_PLOTS:
        for _, path in vj.plot_wout(wout_path, artifact_dir).items():
            print(f"wrote {path}")
    if MAKE_B_AXIS_PLOTS:
        plot_b_on_axis(eq.wout, artifact_dir, label)
        plot_b_on_first_flux_surface(eq.wout, artifact_dir, label)
    if MAKE_BOOZER_B_CONTOUR_PLOTS:
        plot_boozer_b_contours(eq.wout, artifact_dir, label)
    if MAKE_J_POLAR_PLOTS:
        plot_j_polar_contours(eq, artifact_dir)


def write_outputs(optimized_input, initial_input):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_copy = OUT_DIR / SEED_INPUT.name
    optimized_input_path = OUT_DIR / "input.QI_neopax_geometry_optimized"
    initial_input.to_indata(seed_copy)
    optimized_input.to_indata(optimized_input_path)
    print(f"wrote {seed_copy}")
    print(f"wrote {optimized_input_path}")
    if MAKE_INITIAL_PLOTS:
        write_geometry_artifacts(initial_input, "initial")
    write_geometry_artifacts(optimized_input, "optimized")


# --------------------------- continuation ladder ----------------------------
def main() -> int:
    active_terms = tuple(term for term in qi_terms if float(term[2]) != 0.0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x = None
    current_input = SEED_INPUT
    optimized_input = None
    initial_input = None
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
        if initial_input is None:
            initial_input = problem.input_from_scaled_parameters(x)
        report("initial", problem, x)
        last_result = opt.least_squares(
            problem,
            max_nfev=QI_NFEV,
            ftol=FTOL,
            xtol=XTOL,
            verbose=1,
            iteration_reporter=iteration_diagnostics,
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
    if optimized_input is not None and initial_input is not None:
        write_outputs(optimized_input, initial_input)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
