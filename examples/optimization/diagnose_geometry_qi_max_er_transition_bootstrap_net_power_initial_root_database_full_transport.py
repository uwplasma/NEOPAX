#!/usr/bin/env python
"""Standalone vacuum QI + database full-transport Er/net-power optimization.

Maximum-Er and the two Er-transition/root-position terms may be active at the
same time.  The bootstrap-current penalty is independently selectable.  Every
trial must reach ``t_final=2`` before the 1000-step guard or it is handled as a
failed optimizer trial.  Net power is an independently selectable objective
with a 300 MW target.  This diagnostic variant also enables detailed reverse
segment/step finite checks; it does not modify the production script.
"""

from __future__ import annotations

import argparse
import copy
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
from NEOPAX._orchestrator import load_config, run_config  # noqa: E402


# --------------------------- parameters ------------------------------------
SEED_INPUT = ROOT / "examples" / "inputs" / "input.QI_nfp2_initial"
TRANSPORT_CONFIG = (
    ROOT
    / "examples"
    / "optimization"
    / "full_transport_database_vacuum_t2.toml"
)
OUT_DIR = ROOT / "outputs" / "geometry_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_diagnostic_optimization"

DATABASE_N_THETA = 25
DATABASE_N_PHI = 25  # NTX calls this toroidal coordinate zeta.
DATABASE_N_XI = 33

SURFACES = np.asarray(
    [1 / 51, 5 / 51, 10 / 51, 15 / 51, 20 / 51, 25 / 51,
     30 / 51, 35 / 51, 40 / 51, 45 / 51, 51 / 51],
    dtype=float,
)
QI_MBOZ = 18
QI_NBOZ = 18
# ``surrogate`` preserves the established objective and all weights below;
# ``physical`` selects the VMEX-like fixed-pitch, actual-well action.
QI_MAXJ_BACKEND = "surrogate"
PHYSICAL_J_PITCHES = None
PHYSICAL_J_TRAPPING_DEPTHS = (0.35, 0.55, 0.75)
PHYSICAL_J_NALPHA, PHYSICAL_J_POINTS_PER_PERIOD = 5, 24
PHYSICAL_J_NUM_PERIODS, PHYSICAL_J_MAX_WELLS = 6, 16
PHYSICAL_J_QUADRATURE_ORDER, PHYSICAL_MAXJ_TARGET = 16, 0.0

MAX_MODE_SCHEDULE = 2
GEOMETRY_FAMILIES = "RBC,ZBS"
SCALE_MODE = "ess"
ESS_ALPHA = 1.2

# None means the full transport solve must reach t_final from the TOML.
FULL_TRANSPORT_ACCEPTED_STEP_LIMIT = None
REVERSE_SEGMENT_LENGTH = 50
MAX_REVERSE_ACCEPTED_STEPS = 500
TRANSPORT_MAX_STEPS = 1000
TRANSPORT_FINAL_TIME = 2.0
REVERSE_STAGE_MODE = "database_full_transport_optimization"
REVERSE_SEGMENT_INPUT_DIAGNOSTICS = True

ASPECT_TARGET = 10.0
IOTA_TARGET = -0.61
MIRROR_TARGET = 0.25
MAX_ER_TARGET = 25.0
ER_TRANSITION_LEFT_TARGET = 26.0
ER_TRANSITION_RIGHT_TARGET = -10.0
ER_TRANSITION_LEFT_INDEX = 25
ER_TRANSITION_RIGHT_INDEX = 26
BOOTSTRAP_LIMIT_SCALED = 0.1
NET_POWER_TARGET_MW = 300.0
# The reverse-AD transport objective is the signed volume average in MW/m^3.
# Use the same fixed reference volume established by the full-transport power
# optimization example, while reporting the corresponding total MW explicitly.
NET_POWER_REFERENCE_VOLUME_M3 = 331.0187969899648
NET_POWER_TARGET_MW_M3 = NET_POWER_TARGET_MW / NET_POWER_REFERENCE_VOLUME_M3
# Preserve the established root/bootstrap geometry weights, the established
# maximum-Er weight, and each transport target from the standalone root lanes.
QI_WEIGHT = 1.0
MAXJ_WEIGHT = 0.0
ASPECT_WEIGHT = 1.0
IOTA_WEIGHT = 1.0
MIRROR_WEIGHT = 100.0
MAX_ER_WEIGHT = 5.0
ER_TRANSITION_LEFT_WEIGHT = 0.09
ER_TRANSITION_RIGHT_WEIGHT = 0.09
BOOTSTRAP_WEIGHT = 2.0
NET_POWER_WEIGHT = 100.0

USE_MAX_ER_OBJECTIVE = True
USE_ER_TRANSITION_OBJECTIVES = False
USE_BOOTSTRAP_PENALTY = False
USE_NET_POWER_OBJECTIVE = True

NFEV = 30
FTOL = 1.0e-6
XTOL = 1.0e-10
GEOMETRY_MAX_ITER = None
SOLVER_DEVICE = "default"

MAKE_WOUT_PLOTS = True
MAKE_J_POLAR_PLOTS = True
MAKE_B_AXIS_PLOTS = True
MAKE_BOOZER_B_CONTOUR_PLOTS = True
MAKE_INITIAL_PLOTS = True
MAKE_TRANSPORT_REPORTS = True


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


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description=__doc__)
    out.add_argument("--seed-input", type=Path, default=SEED_INPUT)
    out.add_argument("--out-dir", type=Path, default=OUT_DIR)
    out.add_argument("--max-nfev", type=int, default=NFEV)
    out.add_argument("--database-n-theta", type=int, default=DATABASE_N_THETA)
    out.add_argument("--database-n-phi", type=int, default=DATABASE_N_PHI)
    out.add_argument("--database-n-xi", type=int, default=DATABASE_N_XI)
    out.add_argument(
        "--er-transition-left-index",
        type=int,
        default=ER_TRANSITION_LEFT_INDEX,
        help="Final-time Er radial-cell index for the left transition target.",
    )
    out.add_argument(
        "--er-transition-right-index",
        type=int,
        default=ER_TRANSITION_RIGHT_INDEX,
        help="Final-time Er radial-cell index for the right transition target.",
    )
    out.add_argument(
        "--max-er",
        action=argparse.BooleanOptionalAction,
        default=USE_MAX_ER_OBJECTIVE,
        help="Enable/disable the maximum-Er objective.",
    )
    out.add_argument(
        "--root-objectives",
        action=argparse.BooleanOptionalAction,
        default=USE_ER_TRANSITION_OBJECTIVES,
        help="Enable/disable both Er-transition/root-position objectives.",
    )
    out.add_argument(
        "--bootstrap-penalty",
        action=argparse.BooleanOptionalAction,
        default=USE_BOOTSTRAP_PENALTY,
        help="Enable/disable the bootstrap-current penalty.",
    )
    out.add_argument(
        "--net-power",
        action=argparse.BooleanOptionalAction,
        default=USE_NET_POWER_OBJECTIVE,
        help="Enable/disable the 300 MW net-total-power objective.",
    )
    out.add_argument(
        "--initial-plots",
        action=argparse.BooleanOptionalAction,
        default=MAKE_INITIAL_PLOTS,
        help="Rerun VMEX/NEOPAX and write plots for the initial configuration.",
    )
    return out


def transport_config(args: argparse.Namespace) -> dict:
    config = copy.deepcopy(load_config(TRANSPORT_CONFIG))
    neoclassical = config.setdefault("neoclassical", {})
    neoclassical["ntx_scan_n_theta"] = int(args.database_n_theta)
    neoclassical["ntx_scan_n_zeta"] = int(args.database_n_phi)
    neoclassical["ntx_scan_n_xi"] = int(args.database_n_xi)
    solver = config.setdefault("transport_solver", {})
    solver["t_final"] = float(TRANSPORT_FINAL_TIME)
    solver["max_steps"] = int(TRANSPORT_MAX_STEPS)
    return config


def max_mode_schedule_values():
    if np.isscalar(MAX_MODE_SCHEDULE):
        return (int(MAX_MODE_SCHEDULE),)
    return tuple(int(value) for value in MAX_MODE_SCHEDULE)


# --------------------------- objective functions ---------------------------
def positive_part(value):
    return jnp.maximum(value, 0.0)


mirror_penalization = opt.transformed_geometry_objective(
    opt.geometry.vmec_mirror_ratio,
    lambda mirror_ratio: positive_part(mirror_ratio - MIRROR_TARGET),
    label="mirror_penalization",
)

bootstrap_penalty = opt.transformed_transport_objective(
    opt.transport.bootstrap_current_softmax_abs_scaled,
    lambda bootstrap_softmax_abs_scaled: positive_part(bootstrap_softmax_abs_scaled - BOOTSTRAP_LIMIT_SCALED),
    label="bootstrap_current_penalty",
)

GEOMETRY_TERMS = (
    (opt.geometry.boozer_qi_objective, 0.0, QI_WEIGHT),
    (opt.geometry.boozer_maxj_objective, 0.0, MAXJ_WEIGHT),
    (mirror_penalization, 0.0, MIRROR_WEIGHT),
    (opt.geometry.vmec_aspect_ratio, ASPECT_TARGET, ASPECT_WEIGHT),
    (opt.geometry.vmec_iota_mean, IOTA_TARGET, IOTA_WEIGHT),
)


def active_terms(args: argparse.Namespace):
    terms = list(GEOMETRY_TERMS)
    if args.max_er:
        terms.append((opt.transport.softmax_Er, MAX_ER_TARGET, MAX_ER_WEIGHT))
    if args.root_objectives:
        terms.extend(
            (
                (opt.transport.Er_transition_left, ER_TRANSITION_LEFT_TARGET, ER_TRANSITION_LEFT_WEIGHT),
                (opt.transport.Er_transition_right, ER_TRANSITION_RIGHT_TARGET, ER_TRANSITION_RIGHT_WEIGHT),
            )
        )
    if args.bootstrap_penalty:
        terms.append((bootstrap_penalty, 0.0, BOOTSTRAP_WEIGHT))
    if args.net_power:
        terms.append(
            (
                opt.transport.net_total_power_volume_average_mw_m3,
                NET_POWER_TARGET_MW_M3,
                NET_POWER_WEIGHT,
            )
        )
    if not any(term[0].objective.family == "transport" if hasattr(term[0], "objective") else term[0].family == "transport" for term in terms):
        raise ValueError("Enable at least one full-transport objective.")
    return tuple(terms)


# --------------------------- reporting -------------------------------------
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

    def component_cost(*labels):
        component_residual = residual(*labels)
        return 0.5 * component_residual * component_residual

    net_power_average = value(
        "transport:net_total_power_volume_average_mw_m3",
        "net_total_power_volume_average_mw_m3",
    )
    net_power_mw = net_power_average * NET_POWER_REFERENCE_VOLUME_M3

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
        f"softmax_Er={value('transport:softmax_Er', 'softmax_Er'):.8e} "
        f"Er_cost={component_cost('transport:softmax_Er', 'softmax_Er'):.8e} "
        f"Er_left={value('transport:Er_transition_left', 'Er_transition_left'):.8e} "
        f"Er_left_cost={component_cost('transport:Er_transition_left', 'Er_transition_left'):.8e} "
        f"Er_right={value('transport:Er_transition_right', 'Er_transition_right'):.8e} "
        f"Er_right_cost={component_cost('transport:Er_transition_right', 'Er_transition_right'):.8e} "
        f"bootstrap_penalty={value('bootstrap_current_penalty', 'transport:bootstrap_current_penalty'):.8e} "
        f"bootstrap_cost={component_cost('bootstrap_current_penalty', 'transport:bootstrap_current_penalty'):.8e} "
        f"net_power_average_MW_m3={net_power_average:.8e} "
        f"net_power_MW={net_power_mw:.8e} "
        f"net_power_cost={component_cost('transport:net_total_power_volume_average_mw_m3', 'net_total_power_volume_average_mw_m3'):.8e}"
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


def plot_j_polar_contours(
    eq, out_dir, *, wout_path=None, physical_pitches=None,
    lambda_samples=(0.1, 0.3, 0.5, 0.7, 0.9)
):
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


def plot_b_profiles(wout, out_dir, label, *, nphi=256):
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


def plot_boozer_b_contours(wout, out_dir, label, *, ntheta=128, nphi=128):
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


def write_geometry_artifacts(input_obj, label, out_dir, *, physical_pitches=None):
    artifact_dir = out_dir / label
    artifact_dir.mkdir(parents=True, exist_ok=True)
    input_path = artifact_dir / f"input.QI_neopax_geometry_full_transport_{label}"
    input_obj.to_indata(input_path)
    print(f"wrote {input_path}")

    eq = vmex_opt.solve_equilibrium(input_obj)
    wout_path = vj.write_wout(artifact_dir / f"wout_QI_neopax_geometry_full_transport_{label}.nc", eq.wout)
    print(f"wrote {wout_path}")
    if MAKE_WOUT_PLOTS:
        for _, path in vj.plot_wout(wout_path, artifact_dir).items():
            print(f"wrote {path}")
    if MAKE_B_AXIS_PLOTS:
        plot_b_profiles(eq.wout, artifact_dir, label)
    if MAKE_BOOZER_B_CONTOUR_PLOTS:
        plot_boozer_b_contours(eq.wout, artifact_dir, label)
    if MAKE_J_POLAR_PLOTS:
        plot_j_polar_contours(
            eq, artifact_dir, wout_path=wout_path,
            physical_pitches=physical_pitches,
        )
    return artifact_dir


def write_transport_report(input_obj, label, config, out_dir):
    """Run the forward transport solver and save its usual plot/HDF5 outputs."""

    if not MAKE_TRANSPORT_REPORTS:
        return None
    artifact_dir = out_dir / label / "transport"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    input_path = artifact_dir / f"input.QI_neopax_geometry_full_transport_{label}"
    input_obj.to_indata(input_path)
    config = copy.deepcopy(config)
    config.setdefault("geometry", {})["vmec_input_file"] = str(input_path)
    transport_output = config.setdefault("transport_output", {})
    transport_output["transport_plot"] = True
    transport_output["transport_write_hdf5"] = True
    transport_output["transport_output_dir"] = str(artifact_dir)
    print(f"[transport-report] running {label} forward transport output", flush=True)
    result = run_config(config)
    print(f"[transport-report] wrote usual transport outputs in {artifact_dir}", flush=True)
    return result


def write_outputs(
    optimized_input, initial_input, *, config, out_dir, seed_input,
    make_initial_plots, physical_pitches=None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_copy = out_dir / seed_input.name
    optimized_input_path = out_dir / "input.QI_neopax_database_full_transport_net_power_optimized"
    initial_input.to_indata(seed_copy)
    optimized_input.to_indata(optimized_input_path)
    print(f"wrote {seed_copy}")
    print(f"wrote {optimized_input_path}")
    if make_initial_plots:
        write_geometry_artifacts(
            initial_input, "initial", out_dir, physical_pitches=physical_pitches
        )
        write_transport_report(initial_input, "initial", config, out_dir)
    write_geometry_artifacts(
        optimized_input, "optimized", out_dir, physical_pitches=physical_pitches
    )
    write_transport_report(optimized_input, "optimized", config, out_dir)


# --------------------------- continuation ladder ----------------------------
def main() -> int:
    args = parser().parse_args()
    if args.max_nfev < 1:
        raise ValueError("--max-nfev must be positive.")
    for name in ("database_n_theta", "database_n_phi", "database_n_xi"):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    seed_input = args.seed_input.resolve()
    out_dir = args.out_dir.resolve()
    config = transport_config(args)
    n_radial = int(config.get("geometry", {}).get("n_radial", 51))
    for name in ("er_transition_left_index", "er_transition_right_index"):
        index = int(getattr(args, name))
        if not 0 <= index < n_radial:
            raise ValueError(
                f"--{name.replace('_', '-')} must be in [0, {n_radial}); got {index}."
            )
    terms = active_terms(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    x = None
    current_input = seed_input
    optimized_input = None
    initial_input = None
    last_problem = None
    last_result = None
    frozen_physical_pitches = PHYSICAL_J_PITCHES

    max_mode_schedule = max_mode_schedule_values()
    for max_mode in max_mode_schedule:
        print(
            "\n===== vacuum database QI + max-Er/root/bootstrap/net-power full transport "
            f"stage, max_mode={max_mode}, grid=({args.database_n_theta},"
            f"{args.database_n_phi},{args.database_n_xi}), "
            f"J_backend={QI_MAXJ_BACKEND} =====",
            flush=True,
        )
        problem = opt.geometry_full_transport_least_squares_problem(
            config,
            terms,
            vmec_input=current_input,
            max_mode=max_mode,
            families=GEOMETRY_FAMILIES,
            scale_mode=SCALE_MODE,
            ess_alpha=ESS_ALPHA,
            mboz=QI_MBOZ,
            nboz=QI_NBOZ,
            surfaces=tuple(float(s) for s in SURFACES),
            n_theta=args.database_n_theta,
            n_zeta=args.database_n_phi,
            n_xi=args.database_n_xi,
            geometry_max_iter=GEOMETRY_MAX_ITER,
            geometry_solver_device=SOLVER_DEVICE,
            device=SOLVER_DEVICE,
            accepted_step_limit=FULL_TRANSPORT_ACCEPTED_STEP_LIMIT,
            reverse_segment_length=REVERSE_SEGMENT_LENGTH,
            max_reverse_accepted_steps=MAX_REVERSE_ACCEPTED_STEPS,
            initial_er_root_ad="jax_selected_root",
            er_transition_left_index=args.er_transition_left_index,
            er_transition_right_index=args.er_transition_right_index,
            radau_jacobian_reuse_mode="legacy",
            reverse_stage_adjoint_solve_mode="block",
            reverse_rhs_transpose_mode="explicit_database",
            reverse_stage_cotangent_mode="full",
            reverse_step_bwd_mode="reduced_cotangent_call_boundary",
            reverse_stage_adjoint_memory_mode="default",
            reverse_segment_input_diagnostics=REVERSE_SEGMENT_INPUT_DIAGNOSTICS,
            reverse_stage_mode=REVERSE_STAGE_MODE,
            qi_maxj_settings=qi_maxj_backend_settings(frozen_physical_pitches),
        )
        if QI_MAXJ_BACKEND.strip().lower() == "physical" and frozen_physical_pitches is None:
            frozen_physical_pitches = tuple(
                float(value) for value in problem.context.qi_maxj_physical_pitches
            )
            print(
                "[setup] frozen_physical_J_pitches_T^-1="
                + ",".join(f"{value:.16g}" for value in frozen_physical_pitches),
                flush=True,
            )
        problem = opt.GeometryInputSavingProblem(
            problem,
            out_dir / f"geometry_inputs_m{max_mode}",
            filename_prefix="input.QI_neopax_database_full_transport_net_power_eval",
        )
        if x is None or len(x) != problem.parameter_count:
            x = np.asarray(jax.device_get(problem.x0), dtype=float)
        print(
            f"[setup] parameter_count={problem.parameter_count} "
            f"parameters={list(problem.parameter_labels)}",
            flush=True,
        )
        print(
            "[setup] full_transport "
            f"t_final={TRANSPORT_FINAL_TIME} max_steps={TRANSPORT_MAX_STEPS} "
            f"accepted_step_limit={FULL_TRANSPORT_ACCEPTED_STEP_LIMIT} "
            f"reverse_segment_length={REVERSE_SEGMENT_LENGTH} "
            f"max_reverse_accepted_steps={MAX_REVERSE_ACCEPTED_STEPS} "
            f"reverse_segment_input_diagnostics={REVERSE_SEGMENT_INPUT_DIAGNOSTICS} "
            f"Er_transition_indices=({args.er_transition_left_index},"
            f"{args.er_transition_right_index})",
            flush=True,
        )
        if initial_input is None:
            initial_input = problem.input_from_scaled_parameters(x)
        initial_evaluation = report("initial", problem, x)
        last_result = opt.least_squares(
            problem,
            max_nfev=args.max_nfev,
            ftol=FTOL,
            xtol=XTOL,
            verbose=1,
            iteration_reporter=iteration_diagnostics,
            initial_evaluation=initial_evaluation,
        )
        x = np.asarray(last_result.x, dtype=float)
        report(f"QI + database full-transport/net-power stage {max_mode}", problem, x)
        optimized_input = problem.input_from_scaled_parameters(x)
        stage_input = out_dir / f"input.QI_neopax_database_full_transport_net_power_stage_m{max_mode}"
        optimized_input.to_indata(stage_input)
        print(f"wrote {stage_input}")
        current_input = stage_input
        x = None
        last_problem = problem

    if optimized_input is None or initial_input is None or last_problem is None or last_result is None:
        raise RuntimeError("No optimization stage was executed.")

    summary = {
        "seed_input": str(seed_input),
        "transport_config": str(TRANSPORT_CONFIG),
        "database_grid": [args.database_n_theta, args.database_n_phi, args.database_n_xi],
        "transport_final_time": TRANSPORT_FINAL_TIME,
        "transport_max_steps": TRANSPORT_MAX_STEPS,
        "objectives": {
            "max_er": bool(args.max_er),
            "root_objectives": bool(args.root_objectives),
            "bootstrap_penalty": bool(args.bootstrap_penalty),
            "net_power": bool(args.net_power),
        },
        "net_power_target_mw": NET_POWER_TARGET_MW,
        "net_power_reference_volume_m3": NET_POWER_REFERENCE_VOLUME_M3,
        "max_mode_schedule": list(max_mode_schedule),
        "accepted_step_limit": FULL_TRANSPORT_ACCEPTED_STEP_LIMIT,
        "reverse_segment_length": REVERSE_SEGMENT_LENGTH,
        "max_reverse_accepted_steps": MAX_REVERSE_ACCEPTED_STEPS,
        "parameter_labels": list(last_problem.parameter_labels),
        "x": np.asarray(last_result.x, dtype=float).tolist(),
        "cost": float(last_result.cost),
        "optimality": float(last_result.optimality),
        "status": int(last_result.status),
        "message": str(last_result.message),
    }
    summary_path = out_dir / "optimization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")
    write_outputs(
        optimized_input,
        initial_input,
        config=config,
        out_dir=out_dir,
        seed_input=seed_input,
        make_initial_plots=args.initial_plots,
        physical_pitches=frozen_physical_pitches,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

