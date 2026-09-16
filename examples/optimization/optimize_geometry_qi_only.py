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
from NEOPAX._orchestrator import load_config  # noqa: E402


# --------------------------- parameters ------------------------------------
SEED_INPUT = ROOT / "examples" / "inputs" / "input.QI_nfp2_initial"
OUT_DIR = ROOT / "outputs" / "geometry_qi_only_optimization"

# Backend used only to calculate the post-optimization Er/bootstrap profiles.
# Change this to "realtime" to use the established exact-Lij configuration.
TRANSPORT_PROFILE_BACKEND = "database"
DATABASE_TRANSPORT_CONFIG = ROOT / "examples" / "benchmarks" / "Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box.toml"
REALTIME_TRANSPORT_CONFIG = ROOT / "examples" / "benchmarks" / "Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml"
DATABASE_N_THETA = 25
DATABASE_N_PHI = 31
DATABASE_N_XI = 64

SURFACES = np.asarray(
    [1 / 51, 5 / 51, 10 / 51, 15 / 51, 20 / 51, 25 / 51, 30 / 51, 35 / 51, 40 / 51, 45 / 51, 50 / 51],
    dtype=float,
)
# Physical-J Boozer/action resolution.  These defaults match the coarse action
# stage used by VMEX main's QI_maxJ_continuation optimization.  For its
# resolved-action settings use MBOZ=NBOZ=10 together with the alternative
# values documented below.
QI_MBOZ = 8
QI_NBOZ = 8

# Select the J definition used by the active QI/max-J terms below.
# "surrogate" preserves the established en/local_test objective;
# "physical" uses resolved actual magnetic wells at fixed physical pitch.
QI_MAXJ_BACKEND = "physical"  # Change to "surrogate" for the established objective.
PHYSICAL_J_PITCHES = None  # Optional tuple in T^-1; None selects once at the seed.
# VMEX QI+maximum-J coarse optimization settings.  These are optimization
# settings, not plotting-only settings, and can be edited here explicitly.
# VMEX's resolved-action alternative is:
#   NALPHA=9, POINTS_PER_PERIOD=32, NUM_PERIODS=10,
#   MAX_WELLS=24, QUADRATURE_ORDER=24, with QI_MBOZ=QI_NBOZ=10 above.
PHYSICAL_J_TRAPPING_DEPTHS = (0.35, 0.55, 0.75)
PHYSICAL_J_NALPHA = 5
PHYSICAL_J_POINTS_PER_PERIOD = 24
PHYSICAL_J_NUM_PERIODS = 6
PHYSICAL_J_MAX_WELLS = 16
PHYSICAL_J_QUADRATURE_ORDER = 16
PHYSICAL_MAXJ_TARGET = 0.0

# Dense post-processing resolution used only for the VMEX-style J contour.
# The physical pitch remains exactly the frozen optimization pitch; increasing
# these values does not alter the optimization objective or its derivatives.
PHYSICAL_J_PLOT_NALPHA = 96
PHYSICAL_J_PLOT_POINTS_PER_PERIOD = 64
PHYSICAL_J_PLOT_NUM_PERIODS = 6
PHYSICAL_J_PLOT_MAX_WELLS = 24
PHYSICAL_J_PLOT_QUADRATURE_ORDER = 32

MAX_MODE_SCHEDULE = (1, 2)
# Boundary degrees of freedom.  ``None`` selects the standard VMEX packed
# parameters independently at every stage in ``MAX_MODE_SCHEDULE``: all RBC
# and ZBS harmonics through that max_mode, with the scale direction RBC(0,0)
# fixed.  To optimize an exact list instead, replace ``None`` with NEOPAX
# ``FAMILY:m:n`` labels, for example:
# GEOMETRY_PARAMETERS = ("RBC:0:1", "RBC:1:0", "ZBS:0:1", "ZBS:1:0")
# An explicit list is reused unchanged at every continuation stage.
GEOMETRY_PARAMETERS = None
GEOMETRY_FAMILIES = ("RBC", "ZBS")
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
MAKE_TRANSPORT_PROFILE_PLOTS = True


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


def qi_maxj_backend_settings(physical_pitches=None):
    return opt.QImaxJBackendSettings(
        backend=QI_MAXJ_BACKEND,
        physical_pitches=(
            PHYSICAL_J_PITCHES if physical_pitches is None else physical_pitches
        ),
        trapping_depths=PHYSICAL_J_TRAPPING_DEPTHS,
        physical_nalpha=PHYSICAL_J_NALPHA,
        physical_points_per_period=PHYSICAL_J_POINTS_PER_PERIOD,
        physical_num_periods=PHYSICAL_J_NUM_PERIODS,
        physical_max_wells=PHYSICAL_J_MAX_WELLS,
        physical_quadrature_order=PHYSICAL_J_QUADRATURE_ORDER,
        physical_maxj_target=PHYSICAL_MAXJ_TARGET,
    )


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


def build_transport_profile_problem(vmec_input):
    """Build a diagnostic-only selected-root problem for Er/bootstrap plots."""

    backend = str(TRANSPORT_PROFILE_BACKEND).strip().lower()
    if backend not in {"database", "realtime"}:
        raise ValueError("TRANSPORT_PROFILE_BACKEND must be 'database' or 'realtime'.")
    config_path = DATABASE_TRANSPORT_CONFIG if backend == "database" else REALTIME_TRANSPORT_CONFIG
    config = load_config(config_path)
    resolution_kwargs = {}
    if backend == "database":
        config.setdefault("neoclassical", {}).update(
            ntx_scan_n_theta=int(DATABASE_N_THETA),
            ntx_scan_n_zeta=int(DATABASE_N_PHI),
            ntx_scan_n_xi=int(DATABASE_N_XI),
        )
        resolution_kwargs = {
            "n_theta": int(DATABASE_N_THETA),
            "n_zeta": int(DATABASE_N_PHI),
            "n_xi": int(DATABASE_N_XI),
        }
    max_mode = max(int(value) for value in MAX_MODE_SCHEDULE)
    return opt.geometry_initial_er_root_only_least_squares_problem(
        config,
        ((opt.transport.softmax_Er, 0.0, 1.0),),
        vmec_input=vmec_input,
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
        reverse_stage_mode="off",
        **resolution_kwargs,
    )


def save_er_profile(rho, er, finite_mask, out_dir, label):
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
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    ax.plot(rho_np, er_np, color="red", linewidth=3.2, solid_capstyle="round")
    ax.set_xlabel(r"$\rho$", fontsize=20)
    ax.set_ylabel(r"$E_r$ [$\mathrm{kV}/\mathrm{m}$]", fontsize=20)
    ax.tick_params(axis="both", labelsize=16)
    fig.tight_layout()
    png_path = out_dir / f"initial_er_profile_{label}.png"
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png_path}")


def save_bootstrap_current_profile(rho, current, finite_mask, out_dir, label):
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


def save_transport_profiles_for_input(vmec_input, out_dir, label):
    backend = str(TRANSPORT_PROFILE_BACKEND).strip().lower()
    resolution = (
        f"grid=({DATABASE_N_THETA},{DATABASE_N_PHI},{DATABASE_N_XI})"
        if backend == "database"
        else "grid=from_realtime_config"
    )
    print(f"[transport profiles] label={label} backend={backend} {resolution}", flush=True)
    problem = build_transport_profile_problem(vmec_input)
    rho, er, current, finite_mask = (
        problem.initial_er_and_bootstrap_current_profiles_from_scaled_parameters(problem.x0)
    )
    save_er_profile(rho, er, finite_mask, out_dir, label)
    save_bootstrap_current_profile(rho, current, finite_mask, out_dir, label)


def plot_physical_j_polar_contours(wout_path, out_dir, *, physical_pitches=None):
    """Plot VMEX's resolved actual-well physical J diagnostic."""

    try:
        from examples.optimization.plot_j_contours_from_wout import (
            plot_vmex_physical_j_contours_from_wout,
        )
    except Exception as exc:
        print(f"skipping physical-J polar plots: {exc}")
        return

    try:
        plot_vmex_physical_j_contours_from_wout(
            wout_path,
            out_dir,
            surfaces=tuple(float(value) for value in SURFACES),
            mboz=QI_MBOZ,
            nboz=QI_NBOZ,
            trapping_depths=PHYSICAL_J_TRAPPING_DEPTHS,
            physical_pitches=physical_pitches,
            nalpha=PHYSICAL_J_PLOT_NALPHA,
            points_per_period=PHYSICAL_J_PLOT_POINTS_PER_PERIOD,
            num_periods=PHYSICAL_J_PLOT_NUM_PERIODS,
            max_wells=PHYSICAL_J_PLOT_MAX_WELLS,
            quadrature_order=PHYSICAL_J_PLOT_QUADRATURE_ORDER,
            jit_boozer=True,
        )
    except Exception as exc:
        print(f"skipping physical-J polar plots: {exc}")


def plot_j_polar_contours(
    eq,
    out_dir,
    *,
    wout_path=None,
    lambda_samples=(0.1, 0.3, 0.5, 0.7, 0.9),
    physical_pitches=None,
):
    del eq, lambda_samples
    plot_physical_j_polar_contours(
        wout_path,
        out_dir,
        physical_pitches=physical_pitches,
    )


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


def write_geometry_artifacts(input_obj, label, *, physical_pitches=None):
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
        plot_j_polar_contours(
            eq,
            artifact_dir,
            wout_path=wout_path,
            physical_pitches=physical_pitches,
        )


def write_outputs(optimized_input, initial_input, *, physical_pitches=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_copy = OUT_DIR / SEED_INPUT.name
    optimized_input_path = OUT_DIR / "input.QI_neopax_geometry_optimized"
    initial_input.to_indata(seed_copy)
    optimized_input.to_indata(optimized_input_path)
    print(f"wrote {seed_copy}")
    print(f"wrote {optimized_input_path}")
    if MAKE_TRANSPORT_PROFILE_PLOTS:
        save_transport_profiles_for_input(seed_copy, OUT_DIR / "initial", "initial")
        save_transport_profiles_for_input(
            optimized_input_path, OUT_DIR / "optimized", "optimized"
        )
    if MAKE_INITIAL_PLOTS:
        write_geometry_artifacts(
            initial_input, "initial", physical_pitches=physical_pitches
        )
    write_geometry_artifacts(
        optimized_input, "optimized", physical_pitches=physical_pitches
    )


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
    frozen_physical_pitches = PHYSICAL_J_PITCHES

    for max_mode in MAX_MODE_SCHEDULE:
        print(
            f"\n===== NEOPAX geometry-only QI stage, max_mode={max_mode}, "
            f"J_backend={QI_MAXJ_BACKEND} =====",
            flush=True,
        )
        if str(QI_MAXJ_BACKEND).strip().lower() == "physical":
            print(
                "[setup] physical_J_resolution "
                f"mboz={QI_MBOZ} nboz={QI_NBOZ} "
                f"nalpha={PHYSICAL_J_NALPHA} "
                f"points_per_period={PHYSICAL_J_POINTS_PER_PERIOD} "
                f"num_periods={PHYSICAL_J_NUM_PERIODS} "
                f"max_wells={PHYSICAL_J_MAX_WELLS} "
                f"quadrature_order={PHYSICAL_J_QUADRATURE_ORDER}",
                flush=True,
            )
        problem = opt.geometry_least_squares_problem(
            current_input,
            active_terms,
            max_mode=max_mode,
            parameters=GEOMETRY_PARAMETERS,
            families=GEOMETRY_FAMILIES,
            scale_mode=SCALE_MODE,
            ess_alpha=ESS_ALPHA,
            mboz=QI_MBOZ,
            nboz=QI_NBOZ,
            surfaces=tuple(float(s) for s in SURFACES),
            max_iter=GEOMETRY_MAX_ITER,
            solver_device=SOLVER_DEVICE,
            qi_maxj_settings=qi_maxj_backend_settings(frozen_physical_pitches),
        )
        if (
            str(QI_MAXJ_BACKEND).strip().lower() == "physical"
            and frozen_physical_pitches is None
        ):
            frozen_physical_pitches = tuple(
                float(value) for value in problem.context.qi_maxj_physical_pitches
            )
            print(
                "[setup] frozen_physical_J_pitches_T^-1="
                + ",".join(f"{value:.16g}" for value in frozen_physical_pitches),
                flush=True,
            )
        problem = opt.GeometryInputSavingProblem(
            problem, OUT_DIR / f"geometry_inputs_m{max_mode}"
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
        "qi_maxj_backend": QI_MAXJ_BACKEND,
        "physical_j_pitches_T_inverse": (
            None
            if frozen_physical_pitches is None
            else list(frozen_physical_pitches)
        ),
        "physical_j_trapping_depths": list(PHYSICAL_J_TRAPPING_DEPTHS),
        "physical_j_resolution": {
            "nalpha": int(PHYSICAL_J_NALPHA),
            "points_per_period": int(PHYSICAL_J_POINTS_PER_PERIOD),
            "num_periods": int(PHYSICAL_J_NUM_PERIODS),
            "max_wells": PHYSICAL_J_MAX_WELLS,
            "quadrature_order": int(PHYSICAL_J_QUADRATURE_ORDER),
        },
        "physical_j_plot_resolution": {
            "nalpha": int(PHYSICAL_J_PLOT_NALPHA),
            "points_per_period": int(PHYSICAL_J_PLOT_POINTS_PER_PERIOD),
            "num_periods": int(PHYSICAL_J_PLOT_NUM_PERIODS),
            "max_wells": PHYSICAL_J_PLOT_MAX_WELLS,
            "quadrature_order": int(PHYSICAL_J_PLOT_QUADRATURE_ORDER),
        },
        "physical_maxj_target": float(PHYSICAL_MAXJ_TARGET),
        "transport_profile_backend": TRANSPORT_PROFILE_BACKEND,
        "transport_profile_config": str(
            DATABASE_TRANSPORT_CONFIG
            if str(TRANSPORT_PROFILE_BACKEND).strip().lower() == "database"
            else REALTIME_TRANSPORT_CONFIG
        ),
        "database_resolution_theta_phi_xi": [DATABASE_N_THETA, DATABASE_N_PHI, DATABASE_N_XI],
        "terms": [(getattr(term[0], "label", term[0].label), float(term[1]), float(term[2])) for term in active_terms],
        "parameter_labels": [] if last_problem is None else list(last_problem.parameter_labels),
        "final_stage_x_scaled": [] if last_result is None else np.asarray(last_result.x, dtype=float).tolist(),
        "cost": None if last_result is None else float(last_result.cost),
        "optimality": None if last_result is None else float(last_result.optimality),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")
    if optimized_input is not None and initial_input is not None:
        write_outputs(
            optimized_input,
            initial_input,
            physical_pitches=frozen_physical_pitches,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
