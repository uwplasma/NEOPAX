#!/usr/bin/env python
"""Profile-only optimization for initial-Er targets and bootstrap-current limit."""

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


# --------------------------- inputs / parameters ---------------------------
SEED_INPUT = ROOT / "examples" / "inputs" / "input.QI_nfp2_newNT_opt_hires"
TRANSPORT_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml"
)
OUT_DIR = ROOT / "outputs" / "profiles_max_er_transition_bootstrap_initial_root_optimization"

PROFILE_PARAMETERS = (
    "n0,T0,density_shape_power,temperature_shape_power,"
    "density_shape_alpha,temperature_shape_alpha"
)
PROFILE_SCALE_MODE = "nominal"
PROFILE_PHYSICAL_LOWER = {
    "n0": 2.0,
    "T0": 7.0,
    "density_shape_power": 0.1,
    "temperature_shape_power": 0.1,
    "density_shape_alpha": 0.1,
    "temperature_shape_alpha": 0.1,
}
PROFILE_PHYSICAL_UPPER = {
    "n0": 10.0,
    "T0": 25.0,
    "density_shape_power": 12.0,
    "temperature_shape_power": 12.0,
    "density_shape_alpha": 12.0,
    "temperature_shape_alpha": 12.0,
}

MAX_ER_TARGET = 25.0
ER_TRANSITION_LEFT_INDEX = 25
ER_TRANSITION_RIGHT_INDEX = 26
ER_TRANSITION_LEFT_TARGET = 25.0
ER_TRANSITION_RIGHT_TARGET = -10.0
BOOTSTRAP_LIMIT_SCALED = 0.1  # 10 kA/m^2 in the scaled bootstrap-current objective.

MAX_ER_WEIGHT = 10.0
ER_TRANSITION_LEFT_WEIGHT = 0.1
ER_TRANSITION_RIGHT_WEIGHT = 0.1
BOOTSTRAP_WEIGHT = 10.0

NFEV = 40
FTOL = 1.0e-6
XTOL = 1.0e-10
SOLVER_DEVICE = "default"
ROOT_OPTIONS = {
    "Er_transition_left_index": ER_TRANSITION_LEFT_INDEX,
    "Er_transition_right_index": ER_TRANSITION_RIGHT_INDEX,
}


# --------------------------- objective functions ---------------------------
def bootstrap_penalty_value(bootstrap_softmax_abs_scaled):
    return jnp.maximum(bootstrap_softmax_abs_scaled - BOOTSTRAP_LIMIT_SCALED, 0.0)


bootstrap_penalty = opt.transformed_transport_objective(
    opt.transport.bootstrap_current_softmax_abs_scaled,
    bootstrap_penalty_value,
    label="bootstrap_current_penalty",
)


terms = [
    (opt.transport.softmax_Er, MAX_ER_TARGET, MAX_ER_WEIGHT),
    (opt.transport.Er_transition_left, ER_TRANSITION_LEFT_TARGET, ER_TRANSITION_LEFT_WEIGHT),
    (opt.transport.Er_transition_right, ER_TRANSITION_RIGHT_TARGET, ER_TRANSITION_RIGHT_WEIGHT),
    (bootstrap_penalty, 0.0, BOOTSTRAP_WEIGHT),
]


# --------------------------- reporting -------------------------------------
def iteration_diagnostics(evaluation):
    values = {
        label: float(np.asarray(jax.device_get(value), dtype=float))
        for label, value in evaluation.result.objective_values.items()
    }
    residuals = np.asarray(jax.device_get(evaluation.residuals), dtype=float)
    jacobian = np.asarray(jax.device_get(evaluation.jacobian), dtype=float)
    gradient = jacobian.T @ residuals if jacobian.ndim == 2 else np.asarray([])
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
    bootstrap_penalty_value = value("transport:bootstrap_current_penalty", "bootstrap_current_penalty")
    bootstrap_residual = residual("transport:bootstrap_current_penalty", "bootstrap_current_penalty")
    return (
        f"softmax_Er={value('transport:softmax_Er', 'softmax_Er'):.8e} "
        f"Er_residual={er_residual:.8e} "
        f"Er_cost={er_cost:.8e} "
        f"Er_left={value('transport:Er_transition_left', 'Er_transition_left'):.8e} "
        f"Er_left_residual={residual('transport:Er_transition_left', 'Er_transition_left'):.8e} "
        f"Er_right={value('transport:Er_transition_right', 'Er_transition_right'):.8e} "
        f"Er_right_residual={residual('transport:Er_transition_right', 'Er_transition_right'):.8e} "
        f"bootstrap_penalty={bootstrap_penalty_value:.8e} "
        f"bootstrap_residual={bootstrap_residual:.8e} "
        f"grad={gradient.tolist()}"
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
    print(f"  jacobian={jacobian.tolist()}")
    print(f"  gradient={(jacobian.T @ residuals).tolist()}")
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
def save_er_profile(rho_np, er_np, finite_np, out_dir, label):
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


def save_bootstrap_current_profile(rho_np, jboot_np, finite_np, out_dir, label):
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
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    ax.plot(rho_np, jboot_ka_m2, color="tab:blue", linewidth=3.0, label="bootstrap current")
    ax.axhline(10.0, color="black", linewidth=2.6, label=r"$10 kA m^{-2}$")
    ax.axhline(-10.0, color="black", linewidth=2.6, label=r"$-10 kA m^{-2}$")
    ax.set_xlabel(r"$\rho$", fontsize=20)
    ax.set_ylabel(r"$J^{BOOTSTRAP}$ [$kA m^{-2}$]", fontsize=20)
    ax.tick_params(axis="both", labelsize=16, width=1.0, length=4)
    ax.grid(False)
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
    density_plot = density_np if density_np.ndim == 2 else density_np[None, :]
    temperature_plot = temperature_np if temperature_np.ndim == 2 else temperature_np[None, :]
    for name, data, ylabel in (
        ("density", density_plot, r"$n$ [$10^{20}m^{-3}$]"),
        ("temperature", temperature_plot, r"$T$ [$keV$]"),
    ):
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
        fig, ax = plt.subplots(figsize=(6.8, 5.6))
        for i in range(data.shape[0]):
            ax.plot(rho_np, data[i], linewidth=3.0, label=f"species {i}")
        ax.set_xlabel(r"$\rho$", fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.tick_params(axis="both", labelsize=16, width=1.0, length=4)
        ax.grid(False)
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


def write_profile_artifacts(problem, x, label):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rho, density, temperature, er, finite_mask = problem.initial_root_profiles_from_scaled_parameters(x)
    rho_np = np.asarray(jax.device_get(rho), dtype=float)
    density_np = np.asarray(jax.device_get(density), dtype=float)
    temperature_np = np.asarray(jax.device_get(temperature), dtype=float)
    er_np = np.asarray(jax.device_get(er), dtype=float)
    finite_np = np.asarray(jax.device_get(finite_mask), dtype=bool)
    rho_boot, jboot, finite_boot = problem.bootstrap_current_profile_from_scaled_parameters(x)
    rho_boot_np = np.asarray(jax.device_get(rho_boot), dtype=float)
    if not np.allclose(rho_np, rho_boot_np):
        raise RuntimeError("Bootstrap-current rho grid did not match initial-root rho grid.")
    save_er_profile(rho_np, er_np, finite_np, OUT_DIR, label)
    save_density_temperature_profiles(rho_np, density_np, temperature_np, OUT_DIR, label)
    save_bootstrap_current_profile(
        rho_np,
        np.asarray(jax.device_get(jboot), dtype=float),
        np.asarray(jax.device_get(finite_boot), dtype=bool),
        OUT_DIR,
        label,
    )


def profile_artifact_arrays(problem, x):
    rho, density, temperature, er, finite_mask = problem.initial_root_profiles_from_scaled_parameters(x)
    rho_np = np.asarray(jax.device_get(rho), dtype=float)
    rho_boot, jboot, finite_boot = problem.bootstrap_current_profile_from_scaled_parameters(x)
    rho_boot_np = np.asarray(jax.device_get(rho_boot), dtype=float)
    if not np.allclose(rho_np, rho_boot_np):
        raise RuntimeError("Bootstrap-current rho grid did not match initial-root rho grid.")
    return {
        "rho": rho_np,
        "density": np.asarray(jax.device_get(density), dtype=float),
        "temperature": np.asarray(jax.device_get(temperature), dtype=float),
        "er": np.asarray(jax.device_get(er), dtype=float),
        "finite_er": np.asarray(jax.device_get(finite_mask), dtype=bool),
        "jboot": np.asarray(jax.device_get(jboot), dtype=float),
        "finite_bootstrap": np.asarray(jax.device_get(finite_boot), dtype=bool),
    }


def save_initial_final_profile_overlay(initial, optimized, out_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"skipping initial/final overlay plots: {exc}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    rho = initial["rho"]
    if not np.allclose(rho, optimized["rho"]):
        raise RuntimeError("Initial and optimized rho grids did not match.")

    def _style_axes(ax):
        ax.tick_params(axis="both", labelsize=16, width=1.0, length=4)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color("0.35")
        ax.margins(x=0.04, y=0.08)

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    ax.plot(rho, initial["er"], color="red", linestyle="--", linewidth=3.0, label="initial")
    ax.plot(rho, optimized["er"], color="red", linestyle="-", linewidth=3.2, label="optimized")
    ax.set_xlabel(r"$\rho$", fontsize=20)
    ax.set_ylabel(r"$E_r$ [$\mathrm{kV}/\mathrm{m}$]", fontsize=20)
    _style_axes(ax)
    ax.legend(loc="lower left", fontsize=15, frameon=True)
    fig.tight_layout()
    png_path = out_dir / "initial_final_er_profile.png"
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png_path}")

    for name, ylabel in (
        ("density", r"$n$ [$10^{20}m^{-3}$]"),
        ("temperature", r"$T$ [$keV$]"),
    ):
        initial_data = initial[name]
        optimized_data = optimized[name]
        initial_plot = initial_data if initial_data.ndim == 2 else initial_data[None, :]
        optimized_plot = optimized_data if optimized_data.ndim == 2 else optimized_data[None, :]
        fig, ax = plt.subplots(figsize=(6.8, 5.6))
        for i in range(initial_plot.shape[0]):
            color = f"C{i}"
            ax.plot(rho, initial_plot[i], color=color, linestyle="--", linewidth=2.8, label=f"species {i} initial")
            ax.plot(rho, optimized_plot[i], color=color, linestyle="-", linewidth=3.0, label=f"species {i} optimized")
        ax.set_xlabel(r"$\rho$", fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        _style_axes(ax)
        ax.legend(loc="best", fontsize=12, frameon=True)
        fig.tight_layout()
        png_path = out_dir / f"initial_final_{name}_profile.png"
        fig.savefig(png_path, dpi=320, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {png_path}")

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    ax.plot(
        rho,
        100.0 * initial["jboot"],
        color="tab:blue",
        linestyle="--",
        linewidth=2.8,
        label="initial",
    )
    ax.plot(
        rho,
        100.0 * optimized["jboot"],
        color="tab:blue",
        linestyle="-",
        linewidth=3.0,
        label="optimized",
    )
    ax.axhline(10.0, color="black", linewidth=2.6, label=r"$10 kA m^{-2}$")
    ax.axhline(-10.0, color="black", linewidth=2.6, label=r"$-10 kA m^{-2}$")
    ax.set_xlabel(r"$\rho$", fontsize=20)
    ax.set_ylabel(r"$J^{BOOTSTRAP}$ [$kA m^{-2}$]", fontsize=20)
    _style_axes(ax)
    ax.legend(loc="best", fontsize=15, frameon=True)
    fig.tight_layout()
    png_path = out_dir / "initial_final_bootstrap_current_profile.png"
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png_path}")


# --------------------------- solve -----------------------------------------
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    active_terms = tuple(term for term in terms if float(term[2]) != 0.0)
    problem = opt.geometry_initial_er_root_only_least_squares_problem(
        TRANSPORT_CONFIG,
        active_terms,
        vmec_input=SEED_INPUT,
        max_mode=None,
        include_profiles=True,
        profile_parameters=PROFILE_PARAMETERS,
        profile_scale_mode=PROFILE_SCALE_MODE,
        geometry_solver_device=SOLVER_DEVICE,
        device=SOLVER_DEVICE,
        root_options=ROOT_OPTIONS,
    )
    x0 = np.asarray(jax.device_get(problem.x0), dtype=float)
    print(f"[setup] parameter_count={problem.parameter_count} parameters={list(problem.parameter_labels)}")
    print(f"[setup] nominal_profile_scales={np.asarray(jax.device_get(problem.x_scale), dtype=float).tolist()}")
    print(
        "[setup] active_terms="
        + json.dumps(
            [
                {
                    "objective": getattr(term[0], "label", getattr(term[0], "name", str(term[0]))),
                    "target": float(term[1]),
                    "weight": float(term[2]),
                }
                for term in active_terms
            ]
        ),
        flush=True,
    )
    report("initial", problem, x0)
    write_profile_artifacts(problem, x0, "initial")
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
        "seed_input": str(SEED_INPUT),
        "transport_config": str(TRANSPORT_CONFIG),
        "profile_parameters": list(problem.parameter_labels),
        "profile_scale_mode": PROFILE_SCALE_MODE,
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
    write_profile_artifacts(problem, x_opt, "optimized")
    save_initial_final_profile_overlay(
        profile_artifact_arrays(problem, x0),
        profile_artifact_arrays(problem, x_opt),
        OUT_DIR,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
