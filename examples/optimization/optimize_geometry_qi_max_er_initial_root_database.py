#!/usr/bin/env python
"""Standalone database-default geometry QI + maximum-Er optimization.

This is the NTX scan-database counterpart of
``optimize_geometry_qi_max_er_initial_root.py``.  It uses the validated
fresh-payload database initial-root lane by default, with the validated
realtime exact-Lij lane available as a setting. It does not alter the
reverse-AD benchmark evaluator.
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

from NEOPAX import optimization as opt  # noqa: E402
from NEOPAX._orchestrator import load_config  # noqa: E402


# --------------------------- user settings ---------------------------------
SEED_INPUT = ROOT / "examples" / "inputs" / "input.QI_nfp2_initial"
DATABASE_TRANSPORT_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box.toml"
)
OUT_DIR = ROOT / "outputs" / "geometry_qi_max_er_initial_root_database_optimization"

# Change these defaults here, or pass the matching command-line options.
# ``phi`` is NTX's toroidal ``zeta`` coordinate.
DATABASE_N_THETA = 25
DATABASE_N_PHI = 31
DATABASE_N_XI = 64

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

# These are the corresponding realtime max-Er script weights.
QI_WEIGHT = 1.0
MAXJ_WEIGHT = 0.0001
ASPECT_WEIGHT = 1.0
IOTA_WEIGHT = 1.0
MIRROR_WEIGHT = 100.0
MAX_ER_WEIGHT = 0.5

NFEV = 40
FTOL = 1.0e-6
XTOL = 1.0e-10
SOLVER_DEVICE = "default"
REVERSE_STAGE_MODE = "database_root_fresh_payload_experiment"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-n-theta", type=int, default=DATABASE_N_THETA)
    parser.add_argument("--database-n-phi", type=int, default=DATABASE_N_PHI)
    parser.add_argument("--database-n-xi", type=int, default=DATABASE_N_XI)
    return parser


def database_config(args: argparse.Namespace) -> dict:
    """Return an in-memory config with this run's database resolution."""

    config = load_config(DATABASE_TRANSPORT_CONFIG)
    neo = config.setdefault("neoclassical", {})
    neo["ntx_scan_n_theta"] = int(args.database_n_theta)
    neo["ntx_scan_n_zeta"] = int(args.database_n_phi)
    neo["ntx_scan_n_xi"] = int(args.database_n_xi)
    return config


qi = opt.geometry.boozer_qi_objective
maxj = opt.geometry.boozer_maxj_objective


def mirror_penalization_value(mirror_ratio):
    return jnp.maximum(mirror_ratio - MIRROR_TARGET, 0.0)


mirror_penalization = opt.transformed_geometry_objective(
    opt.geometry.vmec_mirror_ratio,
    mirror_penalization_value,
    label="mirror_penalization",
)


TERMS = (
    (qi, 0.0, QI_WEIGHT),
    (maxj, 0.0, MAXJ_WEIGHT),
    (mirror_penalization, 0.0, MIRROR_WEIGHT),
    (opt.geometry.vmec_aspect_ratio, ASPECT_TARGET, ASPECT_WEIGHT),
    (opt.geometry.vmec_iota_mean, IOTA_TARGET, IOTA_WEIGHT),
    (opt.transport.softmax_Er, MAX_ER_TARGET, MAX_ER_WEIGHT),
)


def build_problem(config: dict, vmec_input, max_mode: int, args: argparse.Namespace):
    return opt.geometry_initial_er_root_only_least_squares_problem(
        config,
        TERMS,
        vmec_input=vmec_input,
        max_mode=int(max_mode),
        include_profiles=False,
        families=GEOMETRY_FAMILIES,
        scale_mode=SCALE_MODE,
        ess_alpha=ESS_ALPHA,
        mboz=QI_MBOZ,
        nboz=QI_NBOZ,
        surfaces=tuple(float(s) for s in SURFACES),
        n_theta=int(args.database_n_theta),
        n_zeta=int(args.database_n_phi),
        n_xi=int(args.database_n_xi),
        geometry_solver_device=SOLVER_DEVICE,
        device=SOLVER_DEVICE,
        reverse_stage_mode=REVERSE_STAGE_MODE,
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
        f"softmax_Er={value('transport:softmax_Er', 'softmax_Er'):.8e} "
        f"Er_cost={component_cost('transport:softmax_Er', 'softmax_Er'):.8e}"
    )


def report(tag: str, problem, x) -> None:
    evaluation = problem.evaluate(x)
    values = {
        label: float(np.asarray(jax.device_get(value)))
        for label, value in evaluation.result.objective_values.items()
    }
    residuals = np.asarray(jax.device_get(evaluation.residuals), dtype=float)
    print(f"[{tag}] elapsed_s={evaluation.elapsed_s:.3f} residual_norm={np.linalg.norm(residuals):.6e}")
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


def main() -> int:
    args = _parser().parse_args()
    if min(args.database_n_theta, args.database_n_phi, args.database_n_xi) < 1:
        raise ValueError("Database theta, phi, and xi resolutions must all be positive.")
    config = database_config(args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    current_input = SEED_INPUT
    final_problem = final_result = optimized_input = initial_input = None
    initial_problem = initial_x = None

    for max_mode in (MAX_MODE_SCHEDULE if not np.isscalar(MAX_MODE_SCHEDULE) else (MAX_MODE_SCHEDULE,)):
        print(
            f"\n===== database QI + max-Er, max_mode={max_mode}, "
            f"grid=({args.database_n_theta},{args.database_n_phi},{args.database_n_xi}) =====",
            flush=True,
        )
        problem = build_problem(config, current_input, int(max_mode), args)
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
        current_input = OUT_DIR / f"input.QI_neopax_database_max_er_stage_m{max_mode}"
        optimized_input.to_indata(current_input)
        print(f"wrote {current_input}")
        final_problem, final_result = problem, result

    if any(
        value is None
        for value in (
            final_problem,
            final_result,
            optimized_input,
            initial_input,
            initial_problem,
            initial_x,
        )
    ):
        raise RuntimeError("No optimization stage was executed.")
    initial_input.to_indata(OUT_DIR / SEED_INPUT.name)
    optimized_input.to_indata(OUT_DIR / "input.QI_neopax_database_max_er_optimized")
    save_transport_profiles(
        initial_problem,
        initial_x,
        OUT_DIR / "initial",
        "initial",
    )
    save_transport_profiles(
        final_problem,
        np.asarray(final_result.x, dtype=float),
        OUT_DIR / "optimized",
        "optimized",
    )
    summary = {
        "seed_input": str(SEED_INPUT), "database_transport_config": str(DATABASE_TRANSPORT_CONFIG),
        "database_resolution_theta_phi_xi": [args.database_n_theta, args.database_n_phi, args.database_n_xi],
        "reverse_stage_mode": REVERSE_STAGE_MODE, "parameter_labels": list(final_problem.parameter_labels),
        "x": np.asarray(final_result.x, dtype=float).tolist(), "cost": float(final_result.cost),
        "optimality": float(final_result.optimality), "status": int(final_result.status),
        "message": str(final_result.message),
    }
    (OUT_DIR / "optimization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {OUT_DIR / 'optimization_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
