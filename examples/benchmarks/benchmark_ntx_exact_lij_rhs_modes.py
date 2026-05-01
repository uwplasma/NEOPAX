"""Benchmark shared transport rhs modes on the NTX exact-Lij runtime model.

Example:

    python examples/benchmarks/benchmark_ntx_exact_lij_rhs_modes.py

    python examples/benchmarks/benchmark_ntx_exact_lij_rhs_modes.py \
        --config examples/Solve_Transport_Equations/Solve_Transport_equations_noHe_radau.toml \
        --backend theta_newton \
        --warmup 1 \
        --repeat 2
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import jax
import jax.numpy as jnp

import NEOPAX


DEFAULT_CONFIG = Path("examples/Solve_Transport_Equations/Solve_Transport_equations_noHe_radau.toml")


def _leaf_max_abs_delta(reference, other) -> float:
    ref_leaves = jax.tree_util.tree_leaves(reference)
    other_leaves = jax.tree_util.tree_leaves(other)
    if len(ref_leaves) != len(other_leaves):
        return float("nan")
    max_delta = 0.0
    for ref_leaf, other_leaf in zip(ref_leaves, other_leaves):
        ref_arr = jnp.asarray(ref_leaf)
        other_arr = jnp.asarray(other_leaf)
        if ref_arr.shape != other_arr.shape:
            return float("nan")
        delta = float(jnp.max(jnp.abs(other_arr - ref_arr)))
        max_delta = max(max_delta, delta)
    return max_delta


def _accepted_count(result) -> int | None:
    if result.accepted_mask is None:
        return None
    return int(jnp.sum(jnp.asarray(result.accepted_mask, dtype=jnp.int32)))


def _build_config(config_path: Path, *, backend: str, rhs_mode: str, n_theta: int | None, n_zeta: int | None, n_xi: int | None):
    config = NEOPAX.prepare_config(config_path, backend=backend)
    config = copy.deepcopy(config)

    general = config.setdefault("general", {})
    general["mode"] = "transport"

    neoclassical = config.setdefault("neoclassical", {})
    neoclassical["flux_model"] = "ntx_exact_lij_runtime"
    if n_theta is not None:
        neoclassical["ntx_exact_n_theta"] = int(n_theta)
    if n_zeta is not None:
        neoclassical["ntx_exact_n_zeta"] = int(n_zeta)
    if n_xi is not None:
        neoclassical["ntx_exact_n_xi"] = int(n_xi)

    ambipolarity = config.setdefault("ambipolarity", {})
    ambipolarity["er_ambipolar_plot"] = False

    solver = config.setdefault("transport_solver", {})
    solver["transport_solver_backend"] = str(backend)
    solver["rhs_mode"] = str(rhs_mode)
    solver["debug_stage_markers"] = False

    transport_output = config.setdefault("transport_output", {})
    transport_output["transport_plot"] = False
    transport_output["transport_write_hdf5"] = False

    return config


def _run_once(config):
    t_start = time.perf_counter()
    result = NEOPAX.run(config)
    wall_seconds = time.perf_counter() - t_start
    return result, wall_seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Base transport config to benchmark. Default: no-He Radau benchmark case.",
    )
    parser.add_argument(
        "--backend",
        default="radau",
        choices=["radau", "theta", "theta_newton"],
        help="Transport solver backend to benchmark.",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per rhs_mode before timing.")
    parser.add_argument("--repeat", type=int, default=1, help="Timed runs per rhs_mode.")
    parser.add_argument(
        "--rhs-modes",
        nargs="+",
        default=["black_box", "lagged_response"],
        help="RHS modes to compare.",
    )
    parser.add_argument("--ntx-n-theta", type=int, default=None, help="Override ntx_exact_n_theta.")
    parser.add_argument("--ntx-n-zeta", type=int, default=None, help="Override ntx_exact_n_zeta.")
    parser.add_argument("--ntx-n-xi", type=int, default=None, help="Override ntx_exact_n_xi.")
    args = parser.parse_args()

    config_path = Path(args.config)
    mode_results = []
    reference_final_state = None

    print(f"[benchmark] config={config_path}")
    print(f"[benchmark] backend={args.backend}")
    print(f"[benchmark] rhs_modes={args.rhs_modes}")
    if args.ntx_n_theta is not None or args.ntx_n_zeta is not None or args.ntx_n_xi is not None:
        print(
            "[benchmark] ntx resolution overrides:",
            f"n_theta={args.ntx_n_theta}",
            f"n_zeta={args.ntx_n_zeta}",
            f"n_xi={args.ntx_n_xi}",
        )

    for rhs_mode in args.rhs_modes:
        config = _build_config(
            config_path,
            backend=args.backend,
            rhs_mode=rhs_mode,
            n_theta=args.ntx_n_theta,
            n_zeta=args.ntx_n_zeta,
            n_xi=args.ntx_n_xi,
        )

        for _ in range(max(0, args.warmup)):
            _run_once(config)

        timed_runs = []
        last_result = None
        for _ in range(max(1, args.repeat)):
            last_result, wall_seconds = _run_once(config)
            timed_runs.append(wall_seconds)

        if reference_final_state is None and last_result is not None:
            reference_final_state = last_result.final_state

        final_state_delta = None
        if reference_final_state is not None and last_result is not None:
            final_state_delta = _leaf_max_abs_delta(reference_final_state, last_result.final_state)

        mode_results.append(
            {
                "rhs_mode": rhs_mode,
                "mean_wall_seconds": sum(timed_runs) / len(timed_runs),
                "best_wall_seconds": min(timed_runs),
                "n_steps": None if last_result is None or last_result.n_steps is None else int(last_result.n_steps),
                "accepted_steps": None if last_result is None else _accepted_count(last_result),
                "failed": None if last_result is None or last_result.failed is None else bool(last_result.failed),
                "fail_code": None if last_result is None or last_result.fail_code is None else int(last_result.fail_code),
                "final_time": None if last_result is None or last_result.final_time is None else float(last_result.final_time),
                "final_state_max_abs_delta_vs_first": final_state_delta,
            }
        )

    print()
    print("rhs_mode                 mean_s     best_s     n_steps  accepted  failed  fail_code  final_t   max|delta|")
    print("-" * 108)
    for row in mode_results:
        print(
            f"{row['rhs_mode']:<23}"
            f"{row['mean_wall_seconds']:>10.3f}"
            f"{row['best_wall_seconds']:>11.3f}"
            f"{str(row['n_steps']):>12}"
            f"{str(row['accepted_steps']):>10}"
            f"{str(row['failed']):>8}"
            f"{str(row['fail_code']):>11}"
            f"{str(None if row['final_time'] is None else round(row['final_time'], 6)):>10}"
            f"{str(None if row['final_state_max_abs_delta_vs_first'] is None else f'{row['final_state_max_abs_delta_vs_first']:.3e}'):>14}"
        )


if __name__ == "__main__":
    main()
