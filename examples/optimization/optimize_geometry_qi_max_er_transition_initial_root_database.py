#!/usr/bin/env python
"""Standalone database-default QI + targeted ambipolar-Er transition optimization."""

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

import vmex as vj  # noqa: E402
from vmex import optimize as vmex_opt  # noqa: E402

from NEOPAX import optimization as opt  # noqa: E402
from NEOPAX._orchestrator import load_config  # noqa: E402


# --------------------------- user settings ---------------------------------
SEED_INPUT = ROOT / "examples" / "inputs" / "input.QI_nfp2_initial"
DATABASE_TRANSPORT_CONFIG = ROOT / "examples" / "benchmarks" / "Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box.toml"
OUT_DIR = ROOT / "outputs" / "geometry_qi_max_er_transition_initial_root_database_optimization"
DATABASE_N_THETA = 25
DATABASE_N_PHI = 31  # NTX calls the toroidal phi coordinate zeta.
DATABASE_N_XI = 64
SURFACES = np.asarray([1 / 51, 5 / 51, 10 / 51, 15 / 51, 20 / 51, 25 / 51, 30 / 51, 35 / 51, 40 / 51, 45 / 51, 51 / 51], dtype=float)
QI_MBOZ = QI_NBOZ = 18
# ``surrogate`` preserves the established objective and all weights below;
# ``physical`` selects the VMEX-like fixed-pitch, actual-well action.
QI_MAXJ_BACKEND = "surrogate"
PHYSICAL_J_PITCHES = None
PHYSICAL_J_TRAPPING_DEPTHS = (0.35, 0.55, 0.75)
PHYSICAL_J_NALPHA, PHYSICAL_J_POINTS_PER_PERIOD = 5, 24
PHYSICAL_J_NUM_PERIODS, PHYSICAL_J_MAX_WELLS = 6, 16
PHYSICAL_J_QUADRATURE_ORDER, PHYSICAL_MAXJ_TARGET = 16, 0.0
MAX_MODE_SCHEDULE = 2
GEOMETRY_FAMILIES, SCALE_MODE, ESS_ALPHA = "RBC,ZBS", "ess", 1.2
ASPECT_TARGET, IOTA_TARGET, MIRROR_TARGET = 10.0, -0.61, 0.19
ER_TRANSITION_LEFT_INDEX, ER_TRANSITION_RIGHT_INDEX = 25, 26
ER_TRANSITION_LEFT_TARGET, ER_TRANSITION_RIGHT_TARGET = 26.0, -10.0
# Correspond to optimize_geometry_qi_max_er_transition_initial_root.py.
QI_WEIGHT, MAXJ_WEIGHT, ASPECT_WEIGHT, IOTA_WEIGHT, MIRROR_WEIGHT = 1.6, 0.001, 1.0, 1.0, 500.0
ER_TRANSITION_LEFT_WEIGHT = ER_TRANSITION_RIGHT_WEIGHT = 0.09
NFEV, FTOL, XTOL, SOLVER_DEVICE = 40, 1.0e-6, 1.0e-10, "default"
REVERSE_STAGE_MODE = "database_root_fresh_payload_experiment"
ROOT_OPTIONS = {"Er_transition_left_index": ER_TRANSITION_LEFT_INDEX, "Er_transition_right_index": ER_TRANSITION_RIGHT_INDEX}

MAKE_WOUT_PLOTS = True
MAKE_J_POLAR_PLOTS = True
MAKE_B_AXIS_PLOTS = True
MAKE_BOOZER_B_CONTOUR_PLOTS = True
MAKE_INITIAL_PLOTS = True
GEOMETRY_ARTIFACT_STEM = "QI_neopax_database_transition"


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description=__doc__)
    out.add_argument("--database-n-theta", type=int, default=DATABASE_N_THETA)
    out.add_argument("--database-n-phi", type=int, default=DATABASE_N_PHI)
    out.add_argument("--database-n-xi", type=int, default=DATABASE_N_XI)
    return out


def config_for_database(args: argparse.Namespace) -> dict:
    config = load_config(DATABASE_TRANSPORT_CONFIG)
    neo = config.setdefault("neoclassical", {})
    neo.update(ntx_scan_n_theta=int(args.database_n_theta), ntx_scan_n_zeta=int(args.database_n_phi), ntx_scan_n_xi=int(args.database_n_xi))
    return config


def mirror_penalty_value(value):
    return jnp.maximum(value - MIRROR_TARGET, 0.0)


mirror_penalty = opt.transformed_geometry_objective(opt.geometry.vmec_mirror_ratio, mirror_penalty_value, label="mirror_penalization")
TERMS = (
    (opt.geometry.boozer_qi_objective, 0.0, QI_WEIGHT),
    (opt.geometry.boozer_maxj_objective, 0.0, MAXJ_WEIGHT),
    (mirror_penalty, 0.0, MIRROR_WEIGHT),
    (opt.geometry.vmec_aspect_ratio, ASPECT_TARGET, ASPECT_WEIGHT),
    (opt.geometry.vmec_iota_mean, IOTA_TARGET, IOTA_WEIGHT),
    (opt.transport.Er_transition_left, ER_TRANSITION_LEFT_TARGET, ER_TRANSITION_LEFT_WEIGHT),
    (opt.transport.Er_transition_right, ER_TRANSITION_RIGHT_TARGET, ER_TRANSITION_RIGHT_WEIGHT),
)


def qi_maxj_backend_settings(physical_pitches=None):
    return opt.QImaxJBackendSettings(
        backend=QI_MAXJ_BACKEND,
        physical_pitches=PHYSICAL_J_PITCHES if physical_pitches is None else physical_pitches,
        trapping_depths=PHYSICAL_J_TRAPPING_DEPTHS,
        physical_nalpha=PHYSICAL_J_NALPHA,
        physical_points_per_period=PHYSICAL_J_POINTS_PER_PERIOD,
        physical_num_periods=PHYSICAL_J_NUM_PERIODS,
        physical_max_wells=PHYSICAL_J_MAX_WELLS,
        physical_quadrature_order=PHYSICAL_J_QUADRATURE_ORDER,
        physical_maxj_target=PHYSICAL_MAXJ_TARGET,
    )


def build_problem(config: dict, vmec_input, max_mode: int, args: argparse.Namespace, *, physical_pitches=None):
    return opt.geometry_initial_er_root_only_least_squares_problem(
        config, TERMS, vmec_input=vmec_input, max_mode=max_mode, include_profiles=False,
        families=GEOMETRY_FAMILIES, scale_mode=SCALE_MODE, ess_alpha=ESS_ALPHA, mboz=QI_MBOZ, nboz=QI_NBOZ,
        surfaces=tuple(float(item) for item in SURFACES), n_theta=args.database_n_theta,
        n_zeta=args.database_n_phi, n_xi=args.database_n_xi, geometry_solver_device=SOLVER_DEVICE,
        device=SOLVER_DEVICE, root_options=ROOT_OPTIONS, reverse_stage_mode=REVERSE_STAGE_MODE,
        qi_maxj_settings=qi_maxj_backend_settings(physical_pitches),
    )


def iteration_diagnostics(evaluation) -> str:
    values = {
        label: float(np.asarray(jax.device_get(value), dtype=float))
        for label, value in evaluation.result.objective_values.items()
    }
    residuals = np.asarray(jax.device_get(evaluation.residuals), dtype=float)
    residual_lookup = {
        label: float(residuals[index])
        for index, label in enumerate(evaluation.result.residual_labels)
    }

    def value(*labels: str) -> float:
        return next((values[label] for label in labels if label in values), np.nan)

    def component_cost(*labels: str) -> float:
        residual = next(
            (residual_lookup[label] for label in labels if label in residual_lookup),
            np.nan,
        )
        return 0.5 * residual * residual

    return (
        f"total_cost={0.5 * float(np.dot(residuals, residuals)):.8e} "
        f"aspect_ratio={value('geometry:vmec_aspect_ratio', 'vmec_aspect_ratio'):.8e} "
        f"aspect_cost={component_cost('geometry:vmec_aspect_ratio', 'vmec_aspect_ratio'):.8e} "
        f"iota_mean={value('geometry:vmec_iota_mean', 'vmec_iota_mean'):.8e} "
        f"iota_cost={component_cost('geometry:vmec_iota_mean', 'vmec_iota_mean'):.8e} "
        f"mirror_ratio={value('geometry:vmec_mirror_ratio', 'vmec_mirror_ratio'):.8e} "
        f"mirror_penalty={value('mirror_penalization'):.8e} "
        f"mirror_cost={component_cost('mirror_penalization'):.8e} "
        f"qi={value('geometry:boozer_qi_objective', 'boozer_qi_objective'):.8e} "
        f"qi_cost={component_cost('geometry:boozer_qi_objective', 'boozer_qi_objective'):.8e} "
        f"maxJ={value('geometry:boozer_maxj_objective', 'boozer_maxj_objective'):.8e} "
        f"maxJ_cost={component_cost('geometry:boozer_maxj_objective', 'boozer_maxj_objective'):.8e} "
        f"Er_left={value('transport:Er_transition_left', 'Er_transition_left'):.8e} "
        f"Er_left_cost={component_cost('transport:Er_transition_left', 'Er_transition_left'):.8e} "
        f"Er_right={value('transport:Er_transition_right', 'Er_transition_right'):.8e} "
        f"Er_right_cost={component_cost('transport:Er_transition_right', 'Er_transition_right'):.8e}"
    )


def report(tag: str, problem, x) -> None:
    evaluation = problem.evaluate(x)
    values = {label: float(np.asarray(jax.device_get(value))) for label, value in evaluation.result.objective_values.items()}
    residuals = np.asarray(jax.device_get(evaluation.residuals), dtype=float)
    print(
        f"[{tag}] elapsed_s={evaluation.elapsed_s:.3f} "
        f"residual_norm={np.linalg.norm(residuals):.6e} "
        f"{iteration_diagnostics(evaluation)}"
    )
    for label, value in values.items():
        print(f"  - {label}: {value:.10e}")


def save_er_profile(problem, x, out_dir: Path, label: str, *, profiles=None) -> None:
    if profiles is None:
        rho, er, finite_mask = problem.initial_er_profile_from_scaled_parameters(x)
    else:
        rho, er, _current, finite_mask = profiles
    rho_np = np.asarray(jax.device_get(rho), dtype=float)
    er_np = np.asarray(jax.device_get(er), dtype=float)
    finite_np = np.asarray(jax.device_get(finite_mask), dtype=bool)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"initial_er_profile_{label}.csv"
    np.savetxt(
        csv_path,
        np.column_stack((rho_np, er_np, finite_np.astype(float))),
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
            index = int(sign_change[0])
            denominator = finite_er[index + 1] - finite_er[index]
            fraction = 0.0 if abs(denominator) < 1.0e-30 else -finite_er[index] / denominator
            marker_rho = float(
                finite_rho[index]
                + np.clip(fraction, 0.0, 1.0) * (finite_rho[index + 1] - finite_rho[index])
            )
        else:
            jump_index = int(np.argmax(np.abs(np.diff(finite_er))))
            marker_rho = float(0.5 * (finite_rho[jump_index] + finite_rho[jump_index + 1]))
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    ax.plot(rho_np, er_np, color="red", linewidth=3.2, solid_capstyle="round")
    if marker_rho is not None:
        ax.axvline(marker_rho, color="black", linewidth=1.8, ymin=0.25, ymax=0.93)
    ax.set_xlabel(r"$\rho$", fontsize=20)
    ax.set_ylabel(r"$E_r$ [$\mathrm{kV}/\mathrm{m}$]", fontsize=20)
    ax.tick_params(axis="both", labelsize=16, width=1.0, length=4)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("0.35")
    ax.margins(x=0.04, y=0.08)
    fig.tight_layout()
    png_path = out_dir / f"initial_er_profile_{label}.png"
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png_path}")


def save_bootstrap_current_profile(problem, x, out_dir: Path, label: str, *, profiles=None) -> None:
    """Save bootstrap current even when it is not an optimization objective."""

    if profiles is None:
        rho, current, finite_mask = problem.bootstrap_current_profile_from_scaled_parameters(x)
    else:
        rho, _er, current, finite_mask = profiles
    rho_np = np.asarray(jax.device_get(rho), dtype=float)
    current_np = np.asarray(jax.device_get(current), dtype=float)
    finite_np = np.asarray(jax.device_get(finite_mask), dtype=bool)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"bootstrap_current_profile_{label}.csv"
    np.savetxt(
        csv_path,
        np.column_stack((rho_np, current_np, finite_np.astype(float))),
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
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    ax.plot(rho_np, 100.0 * current_np, color="tab:blue", linewidth=3.0, label="bootstrap current")
    ax.axhline(10.0, color="black", linewidth=2.0, label=r"$10\,\mathrm{kA\,m^{-2}}$")
    ax.axhline(-10.0, color="black", linewidth=2.0, label=r"$-10\,\mathrm{kA\,m^{-2}}$")
    ax.set_xlabel(r"$\rho$", fontsize=20)
    ax.set_ylabel(r"$J^{BOOTSTRAP}$ [$\mathrm{kA\,m^{-2}}$]", fontsize=20)
    ax.tick_params(axis="both", labelsize=16)
    ax.legend(loc="best")
    fig.tight_layout()
    png_path = out_dir / f"bootstrap_current_profile_{label}.png"
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png_path}")


def save_transport_profiles(problem, x, out_dir: Path, label: str) -> None:
    """Save Er and bootstrap profiles from one shared database/root calculation."""

    profiles = problem.initial_er_and_bootstrap_current_profiles_from_scaled_parameters(x)
    save_er_profile(problem, x, out_dir, label, profiles=profiles)
    save_bootstrap_current_profile(problem, x, out_dir, label, profiles=profiles)


def plot_j_polar_contours(
    eq, out_dir: Path, *, wout_path=None, physical_pitches=None,
    lambda_samples=(0.1, 0.3, 0.5, 0.7, 0.9)
) -> None:
    if wout_path is not None:
        try:
            from examples.optimization.plot_j_contours_from_wout import (
                plot_vmex_physical_j_contours_from_wout,
            )

            plot_vmex_physical_j_contours_from_wout(
                wout_path, out_dir, surfaces=SURFACES, mboz=QI_MBOZ, nboz=QI_NBOZ,
                trapping_depths=PHYSICAL_J_TRAPPING_DEPTHS,
                physical_pitches=physical_pitches,
            )
        except Exception as exc:
            print(f"skipping VMEX physical-J plots: {exc}")
        return
    """Write polar contours of the second adiabatic invariant and its QI target."""

    try:
        import matplotlib.pyplot as plt
        from vmex.core.omnigenity_j import JInvariantQIResidual
    except Exception as exc:
        print(f"skipping J-polar plots: {exc}")
        return

    objective = JInvariantQIResidual(SURFACES, mboz=QI_MBOZ, nboz=QI_NBOZ)
    try:
        output = objective.compute_state(eq.state, eq.runtime)
    except Exception as exc:
        print(f"skipping J-polar plots: {exc}")
        return

    alpha = np.asarray(output["alpha"], dtype=float)
    surfaces = np.asarray(output["surfaces"], dtype=float)
    ji = np.asarray(output["ji"], dtype=float)
    jc = np.asarray(output["jc"], dtype=float)
    lambda_grid = np.power(
        np.arange(objective.n_bounce, dtype=float) / max(objective.n_bounce - 1, 1),
        objective.p_lambda,
    )
    theta = np.concatenate([alpha, alpha[:1] + 2.0 * np.pi])
    theta_grid, radius_grid = np.meshgrid(theta, surfaces, indexing="xy")
    sample_indices = sorted(
        {
            int(np.clip(round(value * (objective.n_bounce - 1)), 0, objective.n_bounce - 1))
            for value in lambda_samples
        }
    )

    for name, data in (("ji", ji), ("jc", jc)):
        for index in sample_indices:
            values = np.concatenate([data[:, :, index], data[:, :1, index]], axis=1)
            display_name = r"$\mathcal{J}$" if name == "ji" else r"$J_C$"
            fig = plt.figure(figsize=(5.4, 5.8))
            axis = fig.add_subplot(1, 1, 1, projection="polar")
            contour = axis.contourf(theta_grid, radius_grid, values, levels=40, cmap="plasma")
            axis.set_title(f"Second adiabatic invariant, {display_name}", fontsize=15, pad=20)
            axis.set_ylim(0.0, float(surfaces.max()))
            axis.set_thetagrids(np.arange(0, 360, 45), fontsize=8)
            radial_ticks = np.linspace(0.2, float(surfaces.max()), 5)
            axis.set_rticks(radial_ticks)
            axis.set_yticklabels([f"{tick:.1f}" for tick in radial_ticks], fontsize=8)
            axis.set_rlabel_position(45)
            axis.grid(color="white", linewidth=0.8, alpha=0.45)
            colorbar = fig.colorbar(contour, ax=axis, pad=0.12, shrink=0.78)
            colorbar.set_label(display_name, fontsize=11)
            colorbar.ax.tick_params(labelsize=8)
            fig.text(0.5, 0.035, rf"$\lambda$ = {lambda_grid[index]:.2f}", ha="center", fontsize=15)
            fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
            path = out_dir / f"{name}_polar_lambda_{index:02d}.png"
            fig.savefig(path, dpi=320, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {path}")


def plot_b_profiles(wout, out_dir: Path, label: str, *, nphi=256) -> None:
    """Write |B| profiles on the magnetic axis and first flux surface."""

    try:
        import matplotlib.pyplot as plt
        from vmex.core.plotting import surface_modB
    except Exception as exc:
        print(f"skipping |B| profile plots: {exc}")
        return

    phi = np.linspace(0.0, 2.0 * np.pi / int(wout.nfp), int(nphi))
    theta = np.asarray([0.0], dtype=float)
    ns = int(getattr(wout, "ns", 2))
    profiles = (
        ("axis", 0, "|B| on axis"),
        ("first_flux_surface", 1 if ns > 1 else 0, "|B| at first flux surface, theta=0"),
    )
    for surface_label, surface_index, ylabel in profiles:
        values = np.asarray(
            surface_modB(wout, s_index=surface_index, theta=theta, phi=phi), dtype=float
        ).reshape(-1)
        csv_path = out_dir / f"B_{surface_label}_{label}.csv"
        np.savetxt(
            csv_path,
            np.column_stack([phi, values]),
            delimiter=",",
            header=f"phi,B_{surface_label}",
            comments="",
        )
        print(f"wrote {csv_path}")
        fig, axis = plt.subplots(figsize=(7, 4))
        axis.plot(phi, values, linewidth=1.6)
        axis.set_xlabel("phi")
        axis.set_ylabel(ylabel)
        axis.set_title(f"{ylabel} ({label})")
        axis.grid(True, alpha=0.3)
        fig.tight_layout()
        png_path = out_dir / f"B_{surface_label}_{label}.png"
        fig.savefig(png_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {png_path}")


def plot_boozer_b_contours(wout, out_dir: Path, label: str, *, ntheta=128, nphi=128) -> None:
    """Write one-field-period |B| contour plots and their numeric data."""

    try:
        import matplotlib.pyplot as plt
        from vmex.core.plotting import surface_modB
    except Exception as exc:
        print(f"skipping Boozer |B| contour plots: {exc}")
        return

    theta = np.linspace(0.0, 2.0 * np.pi, int(ntheta))
    phi = np.linspace(0.0, 2.0 * np.pi / int(wout.nfp), int(nphi))
    ns = int(getattr(wout, "ns", 2))
    for surface_label, surface_index in (("axis", 0), ("first_flux_surface", 1 if ns > 1 else 0)):
        values = np.asarray(
            surface_modB(wout, s_index=surface_index, theta=theta, phi=phi), dtype=float
        )
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
        csv_path = out_dir / f"B_boozer_contour_{surface_label}_{label}.csv"
        np.savetxt(
            csv_path,
            np.column_stack([theta_grid.ravel(), phi_grid.ravel(), values.ravel()]),
            delimiter=",",
            header="theta,phi,B",
            comments="",
        )
        print(f"wrote {csv_path}")
        finite = values[np.isfinite(values)]
        if finite.size and float(finite.max()) > float(finite.min()):
            levels = np.linspace(float(finite.min()), float(finite.max()), 28)
        elif finite.size:
            padding = max(abs(float(finite.min())), 1.0) * 1.0e-8
            levels = np.linspace(float(finite.min()) - padding, float(finite.max()) + padding, 28)
        else:
            levels = 28
        fig, axis = plt.subplots(figsize=(6.4, 4.8))
        contour = axis.contour(phi, theta, values, levels=levels, cmap="viridis", linewidths=1.0)
        colorbar = fig.colorbar(contour, ax=axis, pad=0.05)
        colorbar.set_label(r"$|B|$ [T]", fontsize=11)
        axis.set_xlabel(r"toroidal angle $\phi$", fontsize=11)
        axis.set_ylabel(r"poloidal angle $\theta$", fontsize=11)
        axis.set_title(f"|B| on {surface_label.replace('_', ' ')} (one field period)", fontsize=12)
        fig.tight_layout()
        png_path = out_dir / f"B_boozer_contour_{surface_label}_{label}.png"
        fig.savefig(png_path, dpi=320, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {png_path}")


def write_geometry_artifacts(input_obj, label: str, *, physical_pitches=None) -> Path:
    """Solve once and write the standard VMEX, J, and |B| diagnostics."""

    artifact_dir = OUT_DIR / label
    artifact_dir.mkdir(parents=True, exist_ok=True)
    input_path = artifact_dir / f"input.{GEOMETRY_ARTIFACT_STEM}_{label}"
    input_obj.to_indata(input_path)
    print(f"wrote {input_path}")
    equilibrium = vmex_opt.solve_equilibrium(input_obj)
    wout_path = vj.write_wout(
        artifact_dir / f"wout_{GEOMETRY_ARTIFACT_STEM}_{label}.nc",
        equilibrium.wout,
    )
    print(f"wrote {wout_path}")
    if MAKE_WOUT_PLOTS:
        for _, path in vj.plot_wout(wout_path, artifact_dir).items():
            print(f"wrote {path}")
    if MAKE_B_AXIS_PLOTS:
        plot_b_profiles(equilibrium.wout, artifact_dir, label)
    if MAKE_BOOZER_B_CONTOUR_PLOTS:
        plot_boozer_b_contours(equilibrium.wout, artifact_dir, label)
    if MAKE_J_POLAR_PLOTS:
        plot_j_polar_contours(
            equilibrium, artifact_dir, wout_path=wout_path,
            physical_pitches=physical_pitches,
        )
    return artifact_dir


def main() -> int:
    args = parser().parse_args()
    if min(args.database_n_theta, args.database_n_phi, args.database_n_xi) < 1:
        raise ValueError("Database theta, phi, and xi resolutions must all be positive.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config, current_input = config_for_database(args), SEED_INPUT
    initial_input = optimized_input = final_problem = final_result = None
    initial_problem = initial_x = None
    frozen_physical_pitches = PHYSICAL_J_PITCHES
    for max_mode in (MAX_MODE_SCHEDULE if not np.isscalar(MAX_MODE_SCHEDULE) else (MAX_MODE_SCHEDULE,)):
        print(f"\n===== database QI + Er transition, max_mode={max_mode}, grid=({args.database_n_theta},{args.database_n_phi},{args.database_n_xi}), J_backend={QI_MAXJ_BACKEND} =====", flush=True)
        problem = build_problem(config, current_input, int(max_mode), args, physical_pitches=frozen_physical_pitches)
        if QI_MAXJ_BACKEND.strip().lower() == "physical" and frozen_physical_pitches is None:
            frozen_physical_pitches = tuple(float(value) for value in problem.context.qi_maxj_physical_pitches)
            print("[setup] frozen_physical_J_pitches_T^-1=" + ",".join(f"{value:.16g}" for value in frozen_physical_pitches), flush=True)
        problem = opt.GeometryInputSavingProblem(
            problem, OUT_DIR / f"geometry_inputs_m{max_mode}"
        )
        x0 = np.asarray(jax.device_get(problem.x0), dtype=float)
        print(f"[setup] parameter_count={problem.parameter_count} parameters={list(problem.parameter_labels)}")
        if initial_input is None:
            initial_input = problem.input_from_scaled_parameters(x0)
            initial_problem = problem
            initial_x = x0.copy()
        report("initial", problem, x0)
        result = opt.least_squares(
            problem,
            max_nfev=NFEV,
            ftol=FTOL,
            xtol=XTOL,
            verbose=1,
            iteration_reporter=iteration_diagnostics,
        )
        report(f"stage_m{max_mode}", problem, result.x)
        optimized_input = problem.input_from_scaled_parameters(result.x)
        current_input = OUT_DIR / f"input.QI_neopax_database_transition_stage_m{max_mode}"
        optimized_input.to_indata(current_input)
        final_problem, final_result = problem, result
    if any(
        value is None
        for value in (
            initial_input,
            optimized_input,
            final_problem,
            final_result,
            initial_problem,
            initial_x,
        )
    ):
        raise RuntimeError("No optimization stage was executed.")
    initial_input.to_indata(OUT_DIR / SEED_INPUT.name)
    optimized_input.to_indata(OUT_DIR / "input.QI_neopax_database_transition_optimized")
    save_transport_profiles(
        initial_problem,
        initial_x,
        OUT_DIR / "initial",
        "initial",
    )
    if MAKE_INITIAL_PLOTS:
        write_geometry_artifacts(
            initial_input, "initial", physical_pitches=frozen_physical_pitches
        )
    save_transport_profiles(
        final_problem,
        np.asarray(final_result.x, dtype=float),
        OUT_DIR / "optimized",
        "optimized",
    )
    write_geometry_artifacts(
        optimized_input, "optimized", physical_pitches=frozen_physical_pitches
    )
    summary = {"seed_input": str(SEED_INPUT), "database_transport_config": str(DATABASE_TRANSPORT_CONFIG), "database_resolution_theta_phi_xi": [args.database_n_theta, args.database_n_phi, args.database_n_xi], "reverse_stage_mode": REVERSE_STAGE_MODE, "parameter_labels": list(final_problem.parameter_labels), "x": np.asarray(final_result.x, dtype=float).tolist(), "cost": float(final_result.cost), "optimality": float(final_result.optimality), "status": int(final_result.status), "message": str(final_result.message)}
    (OUT_DIR / "optimization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {OUT_DIR / 'optimization_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
