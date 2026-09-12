#!/usr/bin/env python
"""Standalone finite-beta database QI + ambipolar-Er-transition optimization.

Set either boolean below to ``True`` to include that finite-beta objective.
The output directory then also receives an initial/final scalar plot.
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

import vmex as vj  # noqa: E402
from vmex import optimize as vmex_opt  # noqa: E402

from NEOPAX import optimization as opt  # noqa: E402
from NEOPAX._orchestrator import load_config  # noqa: E402


# --------------------------- user settings ---------------------------------
SEED_INPUT = ROOT / "examples" / "inputs" / "input.QI_nfp2_initial_finitebeta"
DATABASE_TRANSPORT_CONFIG = ROOT / "examples" / "benchmarks" / "Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box.toml"
OUT_DIR = ROOT / "outputs" / "geometry_qi_max_er_transition_initial_root_database_finitebeta_optimization"
DATABASE_N_THETA, DATABASE_N_PHI, DATABASE_N_XI = 25, 31, 64  # phi is NTX zeta.
SURFACES = np.asarray([1 / 51, 5 / 51, 10 / 51, 15 / 51, 20 / 51, 25 / 51, 30 / 51, 35 / 51, 40 / 51, 45 / 51, 51 / 51], dtype=float)
QI_MBOZ = QI_NBOZ = 18
MAX_MODE_SCHEDULE = (4,)
GEOMETRY_FAMILIES, SCALE_MODE, ESS_ALPHA = "RBC,ZBS", "ess", 1.2
ASPECT_TARGET, IOTA_TARGET, MIRROR_TARGET = 10.0, -0.61, 0.19
ER_TRANSITION_LEFT_INDEX, ER_TRANSITION_RIGHT_INDEX = 25, 26
ER_TRANSITION_LEFT_TARGET, ER_TRANSITION_RIGHT_TARGET = 26.0, -10.0
# Same base weights as optimize_geometry_qi_max_er_transition_initial_root.py.
QI_WEIGHT, MAXJ_WEIGHT, ASPECT_WEIGHT, IOTA_WEIGHT, MIRROR_WEIGHT = 1.6, 0.001, 1.0, 1.0, 500.0
ER_TRANSITION_LEFT_WEIGHT = ER_TRANSITION_RIGHT_WEIGHT = 0.09
# VMEX finite-beta defaults used by optimize_geometry_qi_only_finitebeta.py.
INCLUDE_BETA = True
BETA_TARGET, BETA_WEIGHT = 0.05, 10.0
INCLUDE_DMERC = False
DMERC_TARGET, DMERC_WEIGHT = 0.0, 0.05
NFEV, FTOL, XTOL, SOLVER_DEVICE = 20, 1.0e-6, 1.0e-10, "default"
REVERSE_STAGE_MODE = "database_root_fresh_payload_experiment"
ROOT_OPTIONS = {
    "Er_transition_left_index": ER_TRANSITION_LEFT_INDEX,
    "Er_transition_right_index": ER_TRANSITION_RIGHT_INDEX,
}

MAKE_WOUT_PLOTS = True
MAKE_J_POLAR_PLOTS = True
MAKE_B_AXIS_PLOTS = True
MAKE_BOOZER_B_CONTOUR_PLOTS = True
MAKE_INITIAL_PLOTS = False


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description=__doc__)
    out.add_argument("--database-n-theta", type=int, default=DATABASE_N_THETA)
    out.add_argument("--database-n-phi", type=int, default=DATABASE_N_PHI)
    out.add_argument("--database-n-xi", type=int, default=DATABASE_N_XI)
    out.add_argument("--include-beta", action="store_true", help="Enable the volume-averaged beta target.")
    out.add_argument("--include-dmerc", action="store_true", help="Enable the softmax-Dmerc stability objective.")
    return out


def config_for_database(args: argparse.Namespace) -> dict:
    config = load_config(DATABASE_TRANSPORT_CONFIG)
    config.setdefault("neoclassical", {}).update(ntx_scan_n_theta=int(args.database_n_theta), ntx_scan_n_zeta=int(args.database_n_phi), ntx_scan_n_xi=int(args.database_n_xi))
    return config


def mirror_penalty_value(value):
    return jnp.maximum(value - MIRROR_TARGET, 0.0)


mirror_penalty = opt.transformed_geometry_objective(opt.geometry.vmec_mirror_ratio, mirror_penalty_value, label="mirror_penalization")


def active_terms():
    terms = [
        (opt.geometry.boozer_qi_objective, 0.0, QI_WEIGHT), (opt.geometry.boozer_maxj_objective, 0.0, MAXJ_WEIGHT),
        (mirror_penalty, 0.0, MIRROR_WEIGHT), (opt.geometry.vmec_aspect_ratio, ASPECT_TARGET, ASPECT_WEIGHT),
        (opt.geometry.vmec_iota_mean, IOTA_TARGET, IOTA_WEIGHT),
        (opt.transport.Er_transition_left, ER_TRANSITION_LEFT_TARGET, ER_TRANSITION_LEFT_WEIGHT),
        (opt.transport.Er_transition_right, ER_TRANSITION_RIGHT_TARGET, ER_TRANSITION_RIGHT_WEIGHT),
    ]
    if INCLUDE_BETA:
        terms.append((opt.geometry.vmec_beta_total, BETA_TARGET, BETA_WEIGHT))
    if INCLUDE_DMERC:
        terms.append((opt.geometry.vmec_dmerc_stability_softmax, DMERC_TARGET, DMERC_WEIGHT))
    return tuple(terms)


def build_problem(config: dict, vmec_input, max_mode: int, args: argparse.Namespace):
    return opt.geometry_initial_er_root_only_least_squares_problem(
        config, active_terms(), vmec_input=vmec_input, max_mode=max_mode, include_profiles=False,
        families=GEOMETRY_FAMILIES, scale_mode=SCALE_MODE, ess_alpha=ESS_ALPHA, mboz=QI_MBOZ, nboz=QI_NBOZ,
        surfaces=tuple(float(item) for item in SURFACES), n_theta=args.database_n_theta, n_zeta=args.database_n_phi,
        n_xi=args.database_n_xi, geometry_solver_device=SOLVER_DEVICE, device=SOLVER_DEVICE,
        root_options=ROOT_OPTIONS, reverse_stage_mode=REVERSE_STAGE_MODE,
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
        for label in labels:
            if label in values:
                return values[label]
        return np.nan

    def residual(*labels: str) -> float:
        for label in labels:
            if label in residual_lookup:
                return residual_lookup[label]
        return np.nan

    def component_cost(*labels: str) -> float:
        component_residual = residual(*labels)
        return 0.5 * component_residual * component_residual

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
        f"Er_right_cost={component_cost('transport:Er_transition_right', 'Er_transition_right'):.8e} "
        f"beta_total={value('geometry:vmec_beta_total', 'vmec_beta_total'):.8e} "
        f"beta_cost={component_cost('geometry:vmec_beta_total', 'vmec_beta_total'):.8e} "
        f"softmax_dmerc={value('geometry:vmec_dmerc_stability_softmax', 'vmec_dmerc_stability_softmax'):.8e} "
        f"dmerc_cost={component_cost('geometry:vmec_dmerc_stability_softmax', 'vmec_dmerc_stability_softmax'):.8e}"
    )


def report(tag: str, problem, x) -> dict[str, float]:
    evaluation = problem.evaluate(x)
    residuals = np.asarray(jax.device_get(evaluation.residuals), dtype=float)
    jacobian = np.asarray(jax.device_get(evaluation.jacobian), dtype=float)
    values = {label: float(np.asarray(jax.device_get(value))) for label, value in evaluation.result.objective_values.items()}
    print(f"[{tag}] elapsed_s={evaluation.elapsed_s:.3f} {iteration_diagnostics(evaluation)}")
    for label, value in values.items():
        print(f"  - {label}: {value:.10e}")
    print(f"  residual_norm={float(np.linalg.norm(residuals)):.6e}")
    print(f"  jacobian_shape={jacobian.shape}")
    return values


def save_er_profile(problem, x, out_dir: Path, label: str) -> None:
    rho, er, finite_mask = problem.initial_er_profile_from_scaled_parameters(x)
    rho_np = np.asarray(jax.device_get(rho), dtype=float)
    er_np = np.asarray(jax.device_get(er), dtype=float)
    finite_np = np.asarray(jax.device_get(finite_mask), dtype=bool)
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


def plot_j_polar_contours(eq, out_dir: Path, *, lambda_samples=(0.1, 0.3, 0.5, 0.7, 0.9)) -> None:
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
    theta = np.concatenate((alpha, alpha[:1] + 2.0 * np.pi))
    theta_grid, radius_grid = np.meshgrid(theta, surfaces, indexing="xy")
    sample_indices = sorted(
        {
            int(np.clip(round(value * (objective.n_bounce - 1)), 0, objective.n_bounce - 1))
            for value in lambda_samples
        }
    )
    for name, data in (("ji", ji), ("jc", jc)):
        for index in sample_indices:
            values = np.concatenate((data[:, :, index], data[:, :1, index]), axis=1)
            display_name = r"$\mathcal{J}$" if name == "ji" else r"$J_C$"
            fig = plt.figure(figsize=(5.4, 5.8))
            ax = fig.add_subplot(1, 1, 1, projection="polar")
            contour = ax.contourf(theta_grid, radius_grid, values, levels=40, cmap="plasma")
            ax.set_title(f"Second adiabatic invariant, {display_name}", fontsize=15, pad=20)
            ax.set_ylim(0.0, float(surfaces.max()))
            ax.grid(color="white", linewidth=0.8, alpha=0.45)
            colorbar = fig.colorbar(contour, ax=ax, pad=0.12, shrink=0.78)
            colorbar.set_label(display_name, fontsize=11)
            fig.text(0.5, 0.035, rf"$\lambda$ = {lambda_grid[index]:.2f}", ha="center", fontsize=15)
            fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
            path = out_dir / f"{name}_polar_lambda_{index:02d}.png"
            fig.savefig(path, dpi=320, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {path}")


def plot_b_profiles(wout, out_dir: Path, label: str, *, nphi: int = 256) -> None:
    try:
        import matplotlib.pyplot as plt
        from vmex.core.plotting import surface_modB
    except Exception as exc:
        print(f"skipping B profile plots: {exc}")
        return

    phi = np.linspace(0.0, 2.0 * np.pi / int(wout.nfp), int(nphi))
    theta = np.asarray((0.0,), dtype=float)
    ns = int(getattr(wout, "ns", 2))
    profiles = (
        ("axis", 0, "|B| on axis"),
        ("first_flux_surface", 1 if ns > 1 else 0, "|B| at first flux surface, theta=0"),
    )
    for profile_name, surface_index, ylabel in profiles:
        values = np.asarray(
            surface_modB(wout, s_index=surface_index, theta=theta, phi=phi),
            dtype=float,
        ).reshape(-1)
        csv_path = out_dir / f"B_{profile_name}_{label}.csv"
        np.savetxt(csv_path, np.column_stack((phi, values)), delimiter=",", header="phi,B", comments="")
        print(f"wrote {csv_path}")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(phi, values, linewidth=1.6)
        ax.set_xlabel("phi")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} ({label})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        png_path = out_dir / f"B_{profile_name}_{label}.png"
        fig.savefig(png_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {png_path}")


def plot_boozer_b_contours(wout, out_dir: Path, label: str, *, ntheta: int = 128, nphi: int = 128) -> None:
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
        b_grid = np.asarray(
            surface_modB(wout, s_index=surface_index, theta=theta, phi=phi),
            dtype=float,
        )
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
        csv_path = out_dir / f"B_boozer_contour_{surface_label}_{label}.csv"
        np.savetxt(
            csv_path,
            np.column_stack((theta_grid.ravel(), phi_grid.ravel(), b_grid.ravel())),
            delimiter=",",
            header="theta,phi,B",
            comments="",
        )
        print(f"wrote {csv_path}")
        finite = b_grid[np.isfinite(b_grid)]
        if finite.size and float(finite.max()) > float(finite.min()):
            levels = np.linspace(float(finite.min()), float(finite.max()), 28)
        else:
            levels = 28
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        contour = ax.contour(phi, theta, b_grid, levels=levels, cmap="viridis", linewidths=1.0)
        fig.colorbar(contour, ax=ax, pad=0.05).set_label(r"$|B|$ [T]")
        ax.set_xlabel(r"toroidal angle $\phi$")
        ax.set_ylabel(r"poloidal angle $\theta$")
        ax.set_title(rf"$|B|$ on {surface_label.replace('_', ' ')} (one field period)")
        fig.tight_layout()
        png_path = out_dir / f"B_boozer_contour_{surface_label}_{label}.png"
        fig.savefig(png_path, dpi=320, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {png_path}")


def plot_active_finite_beta_objectives(eq, out_dir: Path, label: str) -> None:
    if not (INCLUDE_DMERC or INCLUDE_BETA):
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"skipping finite-beta objective plots: {exc}")
        return

    if INCLUDE_DMERC:
        try:
            from vmex.core import stability

            s = np.asarray(eq.runtime.setup.s_full, dtype=float)
            dmerc = np.asarray(jax.device_get(stability.d_merc_state(eq.state, eq.runtime)), dtype=float)
            violation = np.asarray(
                jax.device_get(stability.mercier_stability_residual(eq.state, eq.runtime)),
                dtype=float,
            )
            softmax = float(jax.device_get(stability.mercier_stability_softmax(eq.state, eq.runtime)))
            fig, (ax_dmerc, ax_violation) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
            ax_dmerc.plot(s, dmerc, linewidth=1.8, label="DMerc")
            ax_dmerc.axhline(DMERC_TARGET, color="black", linestyle="--", linewidth=1.0, label="target")
            ax_dmerc.set_ylabel("DMerc")
            ax_dmerc.legend(loc="best")
            ax_violation.plot(s[2:-1], violation, linewidth=1.8, color="tab:red")
            ax_violation.set_xlabel("normalized toroidal flux s")
            ax_violation.set_ylabel("smooth violation")
            ax_violation.set_title(f"softmax DMerc = {softmax:.8e}")
            fig.tight_layout()
            path = out_dir / f"softmax_dmerc_{label}.png"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {path}")
        except Exception as exc:
            print(f"skipping softmax-DMerc plot: {exc}")

    if INCLUDE_BETA:
        try:
            beta_total = float(np.asarray(eq.wout.betatotal, dtype=float))
            fig, ax = plt.subplots(figsize=(5.5, 4.0))
            bars = ax.bar(("VMEX total beta", "target"), (beta_total, BETA_TARGET))
            ax.set_ylabel("volume-averaged total beta")
            ax.set_title(f"VMEX betatotal ({label})")
            for bar, value in zip(bars, (beta_total, BETA_TARGET), strict=True):
                ax.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.6g}", ha="center", va="bottom")
            fig.tight_layout()
            path = out_dir / f"beta_total_{label}.png"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {path}")
        except Exception as exc:
            print(f"skipping beta-total plot: {exc}")


def write_geometry_artifacts(input_obj, label: str) -> Path:
    artifact_dir = OUT_DIR / label
    artifact_dir.mkdir(parents=True, exist_ok=True)
    input_path = artifact_dir / f"input.QI_neopax_database_finitebeta_transition_{label}"
    input_obj.to_indata(input_path)
    print(f"wrote {input_path}")
    eq = vmex_opt.solve_equilibrium(input_obj)
    wout_path = vj.write_wout(
        artifact_dir / f"wout_QI_neopax_database_finitebeta_transition_{label}.nc",
        eq.wout,
    )
    print(f"wrote {wout_path}")
    if MAKE_WOUT_PLOTS:
        for _, path in vj.plot_wout(wout_path, artifact_dir).items():
            print(f"wrote {path}")
    if MAKE_B_AXIS_PLOTS:
        plot_b_profiles(eq.wout, artifact_dir, label)
    if MAKE_BOOZER_B_CONTOUR_PLOTS:
        plot_boozer_b_contours(eq.wout, artifact_dir, label)
    if MAKE_J_POLAR_PLOTS:
        plot_j_polar_contours(eq, artifact_dir)
    plot_active_finite_beta_objectives(eq, artifact_dir, label)
    return artifact_dir


def write_outputs(initial_input, optimized_input, initial_problem, initial_x, final_problem, final_x) -> None:
    seed_copy = OUT_DIR / SEED_INPUT.name
    optimized_path = OUT_DIR / "input.QI_neopax_database_finitebeta_transition_optimized"
    initial_input.to_indata(seed_copy)
    optimized_input.to_indata(optimized_path)
    print(f"wrote {seed_copy}")
    print(f"wrote {optimized_path}")
    if MAKE_INITIAL_PLOTS:
        initial_dir = write_geometry_artifacts(initial_input, "initial")
        save_er_profile(initial_problem, initial_x, initial_dir, "initial")
    optimized_dir = write_geometry_artifacts(optimized_input, "optimized")
    save_er_profile(final_problem, final_x, optimized_dir, "optimized")


def main() -> int:
    global INCLUDE_BETA, INCLUDE_DMERC
    args = parser().parse_args()
    INCLUDE_BETA = bool(INCLUDE_BETA or args.include_beta)
    INCLUDE_DMERC = bool(INCLUDE_DMERC or args.include_dmerc)
    if min(args.database_n_theta, args.database_n_phi, args.database_n_xi) < 1:
        raise ValueError("Database theta, phi, and xi resolutions must all be positive.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config, current_input = config_for_database(args), SEED_INPUT
    initial_input = optimized_input = final_problem = final_result = initial_values = final_values = None
    initial_problem = None
    initial_x = None
    for max_mode in MAX_MODE_SCHEDULE:
        print(f"\n===== finite-beta database QI + Er transition, max_mode={max_mode}, grid=({args.database_n_theta},{args.database_n_phi},{args.database_n_xi}), beta={INCLUDE_BETA}, dmerc={INCLUDE_DMERC} =====", flush=True)
        problem = build_problem(config, current_input, max_mode, args)
        x0 = np.asarray(jax.device_get(problem.x0), dtype=float)
        print(f"[setup] parameter_count={problem.parameter_count} parameters={list(problem.parameter_labels)}")
        if initial_input is None:
            initial_input, initial_values = problem.input_from_scaled_parameters(x0), report("initial", problem, x0)
            initial_problem = problem
            initial_x = x0.copy()
        else:
            report("initial", problem, x0)
        result = opt.least_squares(
            problem,
            max_nfev=NFEV,
            ftol=FTOL,
            xtol=XTOL,
            verbose=1,
            iteration_reporter=iteration_diagnostics,
        )
        final_values = report(f"stage_m{max_mode}", problem, result.x)
        optimized_input = problem.input_from_scaled_parameters(result.x)
        current_input = OUT_DIR / f"input.QI_neopax_database_finitebeta_transition_stage_m{max_mode}"
        optimized_input.to_indata(current_input)
        final_problem, final_result = problem, result
    if any(
        value is None
        for value in (
            initial_input,
            optimized_input,
            initial_problem,
            initial_x,
            final_problem,
            final_result,
            initial_values,
            final_values,
        )
    ):
        raise RuntimeError("No optimization stage was executed.")
    write_outputs(
        initial_input,
        optimized_input,
        initial_problem,
        initial_x,
        final_problem,
        np.asarray(final_result.x, dtype=float),
    )
    summary = {"seed_input": str(SEED_INPUT), "database_transport_config": str(DATABASE_TRANSPORT_CONFIG), "database_resolution_theta_phi_xi": [args.database_n_theta, args.database_n_phi, args.database_n_xi], "reverse_stage_mode": REVERSE_STAGE_MODE, "include_beta": INCLUDE_BETA, "include_dmerc": INCLUDE_DMERC, "parameter_labels": list(final_problem.parameter_labels), "x": np.asarray(final_result.x, dtype=float).tolist(), "cost": float(final_result.cost), "optimality": float(final_result.optimality), "status": int(final_result.status), "message": str(final_result.message)}
    (OUT_DIR / "optimization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {OUT_DIR / 'optimization_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
