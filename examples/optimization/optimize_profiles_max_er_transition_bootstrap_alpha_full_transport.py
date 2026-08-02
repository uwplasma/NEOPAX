#!/usr/bin/env python
"""Profile-only optimization using full Radau transport reverse AD.

This is the full-time-transport counterpart to the initial-Er-root profile
script: the objectives are evaluated on the final transported state rather
than on the initial ambipolarity solve alone.
"""

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

from NEOPAX import optimization as opt  # noqa: E402
from NEOPAX._constants import elementary_charge  # noqa: E402
from NEOPAX._transport_flux_models import DENSITY_STATE_TO_PHYSICAL  # noqa: E402


# --------------------------- inputs / parameters ---------------------------
TRANSPORT_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml"
)
OUT_DIR = ROOT / "outputs" / "profiles_max_er_transition_bootstrap_alpha_full_transport_optimization"

PROFILE_PARAMETERS = "n0,T0,density_shape_power,temperature_shape_power"
PROFILE_SCALE_MODE = "nominal"
PROFILE_PHYSICAL_LOWER = {
    "n0": 0.05,
    "T0": 0.05,
    "density_shape_power": 0.25,
    "temperature_shape_power": 0.25,
}
PROFILE_PHYSICAL_UPPER = {
    "n0": 20.0,
    "T0": 60.0,
    "density_shape_power": 40.0,
    "temperature_shape_power": 20.0,
}

# Use None to run the full transport interval from the TOML t_final.
FULL_TRANSPORT_ACCEPTED_STEP_LIMIT = None
REVERSE_SEGMENT_LENGTH = 4

MAX_ER_TARGET = 25.0
ER_TRANSITION_LEFT_INDEX = 25
ER_TRANSITION_RIGHT_INDEX = 26
ER_TRANSITION_LEFT_TARGET = 26.0
ER_TRANSITION_RIGHT_TARGET = -10.0
BOOTSTRAP_LIMIT_SCALED = 0.1  # 10 kA/m^2 in the scaled bootstrap-current objective.
ALPHA_POWER_MIN_MW_M3 = 0.6

MAX_ER_WEIGHT = 0.6
ER_TRANSITION_LEFT_WEIGHT = 1.0
ER_TRANSITION_RIGHT_WEIGHT = 1.0
BOOTSTRAP_WEIGHT = 1.0
ALPHA_POWER_WEIGHT = 1.0

NFEV = 10
FTOL = 1.0e-6
XTOL = 1.0e-10
SOLVER_DEVICE = "default"


# --------------------------- objective functions ---------------------------
def positive_part(value):
    return jnp.maximum(value, 0.0)


bootstrap_penalty = opt.transformed_transport_objective(
    opt.transport.bootstrap_current_softmax_abs_scaled,
    lambda value: positive_part(value - BOOTSTRAP_LIMIT_SCALED),
    label="bootstrap_current_penalty",
)

alpha_power_shortfall = opt.transformed_transport_objective(
    opt.transport.alpha_power_volume_average_mw_m3,
    lambda value: positive_part(ALPHA_POWER_MIN_MW_M3 - value),
    label="alpha_power_shortfall",
)

terms = [
    (opt.transport.softmax_Er, MAX_ER_TARGET, MAX_ER_WEIGHT),
    (opt.transport.Er_transition_left, ER_TRANSITION_LEFT_TARGET, ER_TRANSITION_LEFT_WEIGHT),
    (opt.transport.Er_transition_right, ER_TRANSITION_RIGHT_TARGET, ER_TRANSITION_RIGHT_WEIGHT),
    (bootstrap_penalty, 0.0, BOOTSTRAP_WEIGHT),
    (alpha_power_shortfall, 0.0, ALPHA_POWER_WEIGHT),
]


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

    er_residual = residual("transport:softmax_Er", "softmax_Er")
    er_cost = 0.5 * er_residual * er_residual
    return (
        f"softmax_Er={value('transport:softmax_Er', 'softmax_Er'):.8e} "
        f"Er_residual={er_residual:.8e} "
        f"Er_cost={er_cost:.8e} "
        f"Er_left={value('transport:Er_transition_left', 'Er_transition_left'):.8e} "
        f"Er_left_residual={residual('transport:Er_transition_left', 'Er_transition_left'):.8e} "
        f"Er_right={value('transport:Er_transition_right', 'Er_transition_right'):.8e} "
        f"Er_right_residual={residual('transport:Er_transition_right', 'Er_transition_right'):.8e} "
        f"bootstrap_penalty={value('transport:bootstrap_current_penalty', 'bootstrap_current_penalty'):.8e} "
        f"alpha_power={value('transport:alpha_power_volume_average_mw_m3', 'alpha_power_volume_average_mw_m3'):.8e} "
        f"alpha_shortfall={value('transport:alpha_power_shortfall', 'alpha_power_shortfall'):.8e}"
    )


def report(tag, problem, x):
    evaluation = problem.evaluate(x)
    residuals = np.asarray(jax.device_get(evaluation.residuals), dtype=float)
    jacobian = np.asarray(jax.device_get(evaluation.jacobian), dtype=float)
    physical = np.asarray(jax.device_get(jnp.asarray(x) * problem.x_scale), dtype=float)
    print(f"[{tag}] elapsed_s={evaluation.elapsed_s:.3f} {iteration_diagnostics(evaluation)}")
    print(f"  scaled_x={np.asarray(x, dtype=float).tolist()}")
    print(f"  physical_profiles={dict(zip(problem.parameter_labels, physical.tolist(), strict=True))}")
    print(f"  residual_norm={float(np.linalg.norm(residuals)):.6e}")
    print(f"  jacobian_shape={jacobian.shape}")
    return evaluation


def scaled_profile_bounds(problem):
    scales = np.asarray(jax.device_get(problem.x_scale), dtype=float)
    lower = []
    upper = []
    for label, scale in zip(problem.parameter_labels, scales, strict=True):
        lower.append(PROFILE_PHYSICAL_LOWER.get(label, -np.inf) / scale)
        upper.append(PROFILE_PHYSICAL_UPPER.get(label, np.inf) / scale)
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


# --------------------------- plots -----------------------------------------
def save_er_profile(rho_np, er_np, out_dir, label):
    csv_path = out_dir / f"Er_profile_{label}.csv"
    np.savetxt(
        csv_path,
        np.column_stack([rho_np, er_np]),
        delimiter=",",
        header="rho,Er",
        comments="",
    )
    print(f"wrote {csv_path}")
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"skipping Er profile plot: {exc}")
        return

    finite = np.isfinite(er_np)
    marker_rho = None
    if np.count_nonzero(finite) >= 2:
        finite_rho = rho_np[finite]
        finite_er = er_np[finite]
        sign_change = np.flatnonzero(finite_er[:-1] * finite_er[1:] <= 0.0)
        if sign_change.size:
            i = int(sign_change[0])
            denom = finite_er[i + 1] - finite_er[i]
            frac = 0.0 if abs(denom) < 1.0e-30 else -finite_er[i] / denom
            marker_rho = float(finite_rho[i] + np.clip(frac, 0.0, 1.0) * (finite_rho[i + 1] - finite_rho[i]))

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
    png_path = out_dir / f"Er_profile_{label}.png"
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png_path}")


def bootstrap_current_profile(final_state, runtime):
    flux_model = runtime.models.flux
    neoclassical_model = getattr(flux_model, "neoclassical_model", flux_model)
    corrected_fluxes_fn = getattr(neoclassical_model, "evaluate_momentum_corrected_fluxes", None)
    if corrected_fluxes_fn is None:
        raise ValueError("Bootstrap-current profile requires evaluate_momentum_corrected_fluxes.")
    fluxes = corrected_fluxes_fn(final_state)
    upar = fluxes.get("Upar_neo", fluxes.get("Upar", None))
    if upar is None:
        raise ValueError("Momentum-corrected fluxes did not provide Upar_neo or Upar.")
    upar_arr = jnp.asarray(upar, dtype=jnp.asarray(final_state.pressure).dtype)
    charge_qp = jnp.asarray(runtime.species.charge_qp, dtype=upar_arr.dtype)
    current_weights = jnp.sign(charge_qp)
    upar_physical = jnp.asarray(DENSITY_STATE_TO_PHYSICAL, dtype=upar_arr.dtype) * upar_arr
    scale = jnp.asarray(elementary_charge * 1.0e-5, dtype=upar_arr.dtype)
    if int(upar_arr.shape[0]) == int(charge_qp.shape[0]):
        return jnp.sum(upar_physical * current_weights[:, None], axis=0) * scale
    return jnp.sum(upar_physical * current_weights[None, :], axis=1) * scale


def save_bootstrap_current_profile(rho_np, jboot_np, out_dir, label):
    csv_path = out_dir / f"bootstrap_current_profile_{label}.csv"
    np.savetxt(
        csv_path,
        np.column_stack([rho_np, jboot_np]),
        delimiter=",",
        header="rho,Jboot_scaled_1e5_A_m2",
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
    ax.set_ylabel(r"$J^{BOOTSTRAP}[kA m^{-2}]$", fontsize=20)
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


def save_density_temperature_profiles(rho_np, density_np, temperature_np, out_dir, label):
    for name, data, ylabel in (
        ("density", density_np, r"$n[10^{20}m^{-3}]$"),
        ("temperature", temperature_np, r"$T[keV]$"),
    ):
        data = data if data.ndim == 2 else data[None, :]
        csv_path = out_dir / f"{name}_profile_{label}.csv"
        np.savetxt(
            csv_path,
            np.column_stack([rho_np, data.T]),
            delimiter=",",
            header="rho," + ",".join(f"species_{i}" for i in range(data.shape[0])),
            comments="",
        )
        print(f"wrote {csv_path}")
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"skipping {name} profile plot: {exc}")
            continue
        fig, ax = plt.subplots(figsize=(6.4, 6.4))
        for i in range(data.shape[0]):
            ax.plot(rho_np, data[i], linewidth=2.8, label=f"species {i}")
        ax.set_xlabel(r"$\rho$", fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.tick_params(axis="both", labelsize=16, width=1.0, length=4)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color("0.35")
        ax.margins(x=0.04, y=0.08)
        ax.legend(loc="best", fontsize=14, frameon=True)
        fig.tight_layout()
        png_path = out_dir / f"{name}_profile_{label}.png"
        fig.savefig(png_path, dpi=320, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {png_path}")


def write_final_profiles(problem, x):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rho, final_state = problem.final_transport_profiles_from_scaled_parameters(x)
    rho_np = np.asarray(jax.device_get(rho), dtype=float)
    density_np = np.asarray(jax.device_get(final_state.density), dtype=float)
    temperature_np = np.asarray(jax.device_get(final_state.temperature), dtype=float)
    er_np = np.asarray(jax.device_get(final_state.Er), dtype=float)
    jboot_np = np.asarray(jax.device_get(bootstrap_current_profile(final_state, problem.runtime)), dtype=float)
    save_er_profile(rho_np, er_np, OUT_DIR, "optimized")
    save_density_temperature_profiles(rho_np, density_np, temperature_np, OUT_DIR, "optimized")
    save_bootstrap_current_profile(rho_np, jboot_np, OUT_DIR, "optimized")


# --------------------------- solve -----------------------------------------
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    problem = opt.full_transport_profile_least_squares_problem(
        TRANSPORT_CONFIG,
        tuple(term for term in terms if float(term[2]) != 0.0),
        profile_parameters=PROFILE_PARAMETERS,
        profile_scale_mode=PROFILE_SCALE_MODE,
        device=SOLVER_DEVICE,
        accepted_step_limit=FULL_TRANSPORT_ACCEPTED_STEP_LIMIT,
        reverse_segment_length=REVERSE_SEGMENT_LENGTH,
        initial_er_root_ad="jax_selected_root",
        radau_jacobian_reuse_mode="legacy",
        reverse_stage_adjoint_solve_mode="bicgstab",
        reverse_rhs_transpose_mode="explicit_ntx_interpolated",
        reverse_step_bwd_mode="reduced_cotangent",
    )
    x0 = np.asarray(jax.device_get(problem.x0), dtype=float)
    print(f"[setup] parameter_count={problem.parameter_count} parameters={list(problem.parameter_labels)}")
    print(f"[setup] nominal_profile_scales={np.asarray(jax.device_get(problem.x_scale), dtype=float).tolist()}")
    print(
        "[setup] full_transport "
        f"accepted_step_limit={FULL_TRANSPORT_ACCEPTED_STEP_LIMIT} "
        f"reverse_segment_length={REVERSE_SEGMENT_LENGTH}"
    )
    report("initial", problem, x0)
    bounds = scaled_profile_bounds(problem)
    result = opt.least_squares(
        problem,
        max_nfev=NFEV,
        ftol=FTOL,
        xtol=XTOL,
        x_scale=np.ones_like(x0),
        bounds=bounds,
        verbose=1,
        iteration_reporter=iteration_diagnostics,
    )
    x_opt = np.asarray(result.x, dtype=float)
    report("profile-only optimized", problem, x_opt)

    summary = {
        "transport_config": str(TRANSPORT_CONFIG),
        "profile_parameters": list(problem.parameter_labels),
        "profile_scale_mode": PROFILE_SCALE_MODE,
        "accepted_step_limit": FULL_TRANSPORT_ACCEPTED_STEP_LIMIT,
        "reverse_segment_length": REVERSE_SEGMENT_LENGTH,
        "x_scaled": x_opt.tolist(),
        "x_physical": np.asarray(jax.device_get(jnp.asarray(x_opt) * problem.x_scale), dtype=float).tolist(),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "status": int(result.status),
        "message": str(result.message),
    }
    summary_path = OUT_DIR / "optimization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")
    write_final_profiles(problem, x_opt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
