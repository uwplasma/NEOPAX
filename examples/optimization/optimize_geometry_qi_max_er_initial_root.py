#!/usr/bin/env python
"""Geometry QI optimization with an initial ambipolar-Er softmax target."""

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
TRANSPORT_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml"
)
OUT_DIR = ROOT / "outputs" / "geometry_qi_max_er_initial_root_optimization"

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

QI_WEIGHT = 1.0
MAXJ_WEIGHT = 0.00005
ASPECT_WEIGHT = 1.0
IOTA_WEIGHT = 1.0
MIRROR_WEIGHT = 100.0
MAX_ER_WEIGHT = 0.6

NFEV = 40
FTOL = 1.0e-6
XTOL = 1.0e-10
GEOMETRY_MAX_ITER = None
SOLVER_DEVICE = "default"
ROOT_OPTIONS = {}

MAKE_WOUT_PLOTS = True
MAKE_J_POLAR_PLOTS = True
MAKE_B_AXIS_PLOTS = True
MAKE_BOOZER_B_CONTOUR_PLOTS = True
MAKE_INITIAL_PLOTS = False
SAVE_INITIAL_ER_PROFILE_EARLY = False


def max_mode_schedule_values():
    if np.isscalar(MAX_MODE_SCHEDULE):
        return (int(MAX_MODE_SCHEDULE),)
    return tuple(int(value) for value in MAX_MODE_SCHEDULE)


# --------------------------- objective functions ---------------------------
qi = opt.geometry.boozer_qi_objective
qi_maxj_1 = opt.geometry.boozer_maxj_objective
softmax_er = opt.transport.softmax_Er


def mirror_penalization_value(mirror_ratio):
    return jnp.maximum(mirror_ratio - MIRROR_TARGET, 0.0)


mirror_penalization = opt.transformed_geometry_objective(
    opt.geometry.vmec_mirror_ratio,
    mirror_penalization_value,
    label="mirror_penalization",
)


terms = [
    (qi, 0.0, QI_WEIGHT),
    (qi_maxj_1, 0.0, MAXJ_WEIGHT),
    (mirror_penalization, 0.0, MIRROR_WEIGHT),
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
    er_left_residual = residual("transport:Er_transition_left", "Er_transition_left")
    er_right_residual = residual("transport:Er_transition_right", "Er_transition_right")
    bootstrap_residual = residual(
        "transport:bootstrap_current_softmax_abs_scaled",
        "transport:bootstrap_current_penalty",
        "bootstrap_current_softmax_abs_scaled",
        "bootstrap_current_penalty",
    )
    return (
        f"aspect_ratio={value('geometry:vmec_aspect_ratio', 'vmec_aspect_ratio'):.8e} "
        f"iota_mean={value('geometry:vmec_iota_mean', 'vmec_iota_mean'):.8e} "
        f"mirror_ratio={value('geometry:vmec_mirror_ratio', 'vmec_mirror_ratio'):.8e} "
        f"mirror_penalty={value('mirror_penalization'):.8e} "
        f"magnetic_well={value('geometry:vmec_magnetic_well', 'vmec_magnetic_well'):.8e} "
        f"qi_cost={value('geometry:boozer_qi_objective', 'boozer_qi_objective'):.8e} "
        f"maxJ_cost={value('geometry:boozer_maxj_objective', 'boozer_maxj_objective'):.8e} "
        f"softmax_Er={value('transport:softmax_Er', 'softmax_Er'):.8e} "
        f"Er_residual={er_residual:.8e} "
        f"Er_cost={er_cost:.8e} "
        f"Er_left={value('transport:Er_transition_left', 'Er_transition_left'):.8e} "
        f"Er_left_residual={er_left_residual:.8e} "
        f"Er_right={value('transport:Er_transition_right', 'Er_transition_right'):.8e} "
        f"Er_right_residual={er_right_residual:.8e} "
        f"bootstrap={value('transport:bootstrap_current_softmax_abs_scaled', 'bootstrap_current_softmax_abs_scaled'):.8e} "
        f"bootstrap_residual={bootstrap_residual:.8e}"
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

    finite_rho = rho_np[finite_np]
    finite_er = er_np[finite_np]
    marker_rho = None
    if finite_rho.size >= 2:
        sign_change = np.flatnonzero(finite_er[:-1] * finite_er[1:] <= 0.0)
        if sign_change.size:
            i = int(sign_change[0])
            denom = finite_er[i + 1] - finite_er[i]
            frac = 0.0 if abs(denom) < 1.0e-30 else -finite_er[i] / denom
            marker_rho = float(finite_rho[i] + np.clip(frac, 0.0, 1.0) * (finite_rho[i + 1] - finite_rho[i]))
        else:
            jump_index = int(np.argmax(np.abs(np.diff(finite_er))))
            marker_rho = float(0.5 * (finite_rho[jump_index] + finite_rho[jump_index + 1]))

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.plot(rho_np, er_np, color="red", linewidth=3.2, solid_capstyle="round")
    if marker_rho is not None:
        ax.axvline(marker_rho, color="black", linewidth=1.8, ymin=0.25, ymax=0.93)
    ax.set_xlabel(r"$\rho$", fontsize=20)
    ax.set_ylabel(r"$E_r[\mathrm{kV}/\mathrm{m}]$", fontsize=20)
    ax.tick_params(axis="both", labelsize=16, width=1.0, length=4)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("0.35")
    ax.margins(x=0.04, y=0.08)
    fig.tight_layout()
    png_path = out_dir / f"initial_er_profile_{label}.png"
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png_path}")


def save_bootstrap_current_profile(problem, x, out_dir, label):
    rho, jboot, finite_mask = problem.bootstrap_current_profile_from_scaled_parameters(x)
    rho_np = np.asarray(jax.device_get(rho), dtype=float)
    jboot_np = np.asarray(jax.device_get(jboot), dtype=float)
    finite_np = np.asarray(jax.device_get(finite_mask), dtype=bool)
    csv_path = out_dir / f"bootstrap_current_profile_{label}.csv"
    np.savetxt(
        csv_path,
        np.column_stack([rho_np, jboot_np, finite_np.astype(float)]),
        delimiter=",",
        header="rho,Jboot_scaled_1e5_A_m2,finite_mask",
        comments="",
    )
    print(f"wrote {csv_path}")
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"skipping bootstrap-current profile plot: {exc}")
        return
    jboot_ka_m2 = 100.0 * jboot_np
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.plot(rho_np, jboot_ka_m2, color="tab:blue", linewidth=3.0, label="bootstrap current")
    ax.axhline(10.0, color="black", linewidth=2.6, label=r"$10 kA m^{-2}$")
    ax.axhline(-10.0, color="black", linewidth=2.6, label=r"$-10 kA m^{-2}$")
    ax.set_xlabel(r"$\rho$", fontsize=20)
    ax.set_ylabel(r"$J^{BOOTSTRAP}[kA m^{-2}]$")
    ax.tick_params(axis="both", labelsize=16, width=1.0, length=4)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("0.35")
    ax.margins(x=0.04, y=0.08)
    ax.legend(loc="best", fontsize=15, frameon=True)
    fig.tight_layout()
    png_path = out_dir / f"bootstrap_current_profile_{label}.png"
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
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

        fig, ax = plt.subplots(figsize=(7, 5))
        contour = ax.contourf(phi, theta, b_grid, levels=40, cmap="viridis")
        fig.colorbar(contour, ax=ax, label="|B|")
        ax.set_xlabel("phi")
        ax.set_ylabel("theta")
        ax.set_title(f"Boozer |B| contour, {surface_label.replace('_', ' ')} ({label})")
        fig.tight_layout()
        png_path = out_dir / f"B_boozer_contour_{surface_label}_{label}.png"
        fig.savefig(png_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {png_path}")


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
    if MAKE_B_AXIS_PLOTS:
        plot_b_on_axis(eq.wout, artifact_dir, label)
        plot_b_on_first_flux_surface(eq.wout, artifact_dir, label)
    if MAKE_BOOZER_B_CONTOUR_PLOTS:
        plot_boozer_b_contours(eq.wout, artifact_dir, label)
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
    if MAKE_INITIAL_PLOTS:
        initial_dir = write_geometry_artifacts(initial_input, "initial")
        save_er_profile(initial_problem, initial_x, initial_dir, "initial")
        save_bootstrap_current_profile(initial_problem, initial_x, initial_dir, "initial")
    optimized_dir = write_geometry_artifacts(optimized_input, "optimized")
    save_er_profile(final_problem, final_x, optimized_dir, "optimized")
    save_bootstrap_current_profile(final_problem, final_x, optimized_dir, "optimized")


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

    max_mode_schedule = max_mode_schedule_values()
    for max_mode in max_mode_schedule:
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
            root_options=ROOT_OPTIONS,
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
        "max_mode_schedule": list(max_mode_schedule),
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
