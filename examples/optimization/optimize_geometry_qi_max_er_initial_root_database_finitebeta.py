#!/usr/bin/env python
"""Standalone finite-beta database QI + maximum-Er optimization.

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

from NEOPAX import optimization as opt  # noqa: E402
from NEOPAX._orchestrator import load_config  # noqa: E402


# --------------------------- user settings ---------------------------------
SEED_INPUT = ROOT / "examples" / "inputs" / "input.QI_nfp2_initial_finitebeta"
DATABASE_TRANSPORT_CONFIG = ROOT / "examples" / "benchmarks" / "Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box.toml"
OUT_DIR = ROOT / "outputs" / "geometry_qi_max_er_initial_root_database_finitebeta_optimization"
DATABASE_N_THETA, DATABASE_N_PHI, DATABASE_N_XI = 25, 31, 64  # phi is NTX zeta.
SURFACES = np.asarray([1 / 51, 5 / 51, 10 / 51, 15 / 51, 20 / 51, 25 / 51, 30 / 51, 35 / 51, 40 / 51, 45 / 51, 51 / 51], dtype=float)
QI_MBOZ = QI_NBOZ = 18
# Finite-beta continuation includes the m=3 stage, as in the QI-only finite-beta example.
MAX_MODE_SCHEDULE = (1, 2, 3)
GEOMETRY_FAMILIES, SCALE_MODE, ESS_ALPHA = "RBC,ZBS", "ess", 1.2
ASPECT_TARGET, IOTA_TARGET, MIRROR_TARGET, MAX_ER_TARGET = 10.0, -0.61, 0.19, 25.0
# Same base weights as optimize_geometry_qi_max_er_initial_root.py.
QI_WEIGHT, MAXJ_WEIGHT, ASPECT_WEIGHT, IOTA_WEIGHT, MIRROR_WEIGHT, MAX_ER_WEIGHT = 1.0, 0.0001, 1.0, 1.0, 100.0, 0.5
# VMEX finite-beta defaults used by optimize_geometry_qi_only_finitebeta.py.
INCLUDE_BETA = False
BETA_TARGET, BETA_WEIGHT = 0.05, 10.0
INCLUDE_DMERC = False
DMERC_TARGET, DMERC_WEIGHT = 0.0, 0.05
NFEV, FTOL, XTOL, SOLVER_DEVICE = 40, 1.0e-6, 1.0e-10, "default"
REVERSE_STAGE_MODE = "database_root_fresh_payload_experiment"


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
        (opt.geometry.vmec_iota_mean, IOTA_TARGET, IOTA_WEIGHT), (opt.transport.softmax_Er, MAX_ER_TARGET, MAX_ER_WEIGHT),
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
        reverse_stage_mode=REVERSE_STAGE_MODE,
    )


def report(tag: str, problem, x) -> dict[str, float]:
    evaluation = problem.evaluate(x)
    values = {label: float(np.asarray(jax.device_get(value))) for label, value in evaluation.result.objective_values.items()}
    print(f"[{tag}] elapsed_s={evaluation.elapsed_s:.3f} residual_norm={np.linalg.norm(np.asarray(jax.device_get(evaluation.residuals))):.6e}")
    for label, value in values.items():
        print(f"  - {label}: {value:.10e}")
    return values


def plot_finite_beta_scalars(initial: dict[str, float], final: dict[str, float]) -> None:
    labels = []
    if INCLUDE_BETA:
        labels.append(("beta_total", "geometry:vmec_beta_total"))
    if INCLUDE_DMERC:
        labels.append(("softmax_dmerc", "geometry:vmec_dmerc_stability_softmax"))
    if not labels:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"skipping finite-beta scalar plot: {exc}")
        return
    names = [name for name, _ in labels]
    before = [initial.get(key, np.nan) for _, key in labels]
    after = [final.get(key, np.nan) for _, key in labels]
    points = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(5, 1.8 * len(names)), 4))
    ax.bar(points - 0.18, before, 0.36, label="initial")
    ax.bar(points + 0.18, after, 0.36, label="optimized")
    ax.set_xticks(points, names)
    ax.set_title("Active finite-beta objectives")
    ax.legend()
    fig.tight_layout()
    path = OUT_DIR / "finitebeta_objectives_initial_vs_optimized.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"wrote {path}")


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
    for max_mode in MAX_MODE_SCHEDULE:
        print(f"\n===== finite-beta database QI + max-Er, max_mode={max_mode}, grid=({args.database_n_theta},{args.database_n_phi},{args.database_n_xi}), beta={INCLUDE_BETA}, dmerc={INCLUDE_DMERC} =====", flush=True)
        problem = build_problem(config, current_input, max_mode, args)
        x0 = np.asarray(jax.device_get(problem.x0), dtype=float)
        print(f"[setup] parameter_count={problem.parameter_count} parameters={list(problem.parameter_labels)}")
        if initial_input is None:
            initial_input, initial_values = problem.input_from_scaled_parameters(x0), report("initial", problem, x0)
        else:
            report("initial", problem, x0)
        result = opt.least_squares(problem, max_nfev=NFEV, ftol=FTOL, xtol=XTOL, verbose=1)
        final_values = report(f"stage_m{max_mode}", problem, result.x)
        optimized_input = problem.input_from_scaled_parameters(result.x)
        current_input = OUT_DIR / f"input.QI_neopax_database_finitebeta_max_er_stage_m{max_mode}"
        optimized_input.to_indata(current_input)
        final_problem, final_result = problem, result
    if any(value is None for value in (initial_input, optimized_input, final_problem, final_result, initial_values, final_values)):
        raise RuntimeError("No optimization stage was executed.")
    initial_input.to_indata(OUT_DIR / SEED_INPUT.name)
    optimized_input.to_indata(OUT_DIR / "input.QI_neopax_database_finitebeta_max_er_optimized")
    plot_finite_beta_scalars(initial_values, final_values)
    summary = {"seed_input": str(SEED_INPUT), "database_transport_config": str(DATABASE_TRANSPORT_CONFIG), "database_resolution_theta_phi_xi": [args.database_n_theta, args.database_n_phi, args.database_n_xi], "reverse_stage_mode": REVERSE_STAGE_MODE, "include_beta": INCLUDE_BETA, "include_dmerc": INCLUDE_DMERC, "parameter_labels": list(final_problem.parameter_labels), "x": np.asarray(final_result.x, dtype=float).tolist(), "cost": float(final_result.cost), "optimality": float(final_result.optimality), "status": int(final_result.status), "message": str(final_result.message)}
    (OUT_DIR / "optimization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {OUT_DIR / 'optimization_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
