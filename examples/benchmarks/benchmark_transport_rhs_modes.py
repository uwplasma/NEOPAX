"""Benchmark shared transport rhs modes on selectable NEOPAX flux models.

Example:

    python examples/benchmarks/benchmark_transport_rhs_modes.py

    python examples/benchmarks/benchmark_transport_rhs_modes.py \
        --config examples/Solve_Transport_Equations/Solve_Transport_equations_noHe_radau.toml \
        --backend theta_newton \
        --flux-model ntx_database \
        --warmup 1 \
        --repeat 2
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import NEOPAX


DEFAULT_CONFIG = Path("examples/benchmarks/Solve_Transport_equations_noHe_radau_benchmark.toml")
DEFAULT_FLUX_MODEL = "ntx_exact_lij_runtime"
DEFAULT_ER_INIT_MODE = "keep"
DEFAULT_NTX_EXACT_N_THETA = 5
DEFAULT_NTX_EXACT_N_ZETA = 21
DEFAULT_NTX_EXACT_N_XI = 32
DEFAULT_NTX_EXACT_FACE_RESPONSE_MODE = "interpolate_center_response"
DEFAULT_NTX_EXACT_RADIAL_BATCH_SIZE = None
DEFAULT_NTX_EXACT_SCAN_BATCH_SIZE = None
DEFAULT_NTX_EXACT_RESPONSE_ANCHOR_COUNT = None
DEFAULT_NTX_EXACT_USE_REMAT = False
DEFAULT_RHS_MODES = ["black_box", "lagged_response", "lagged_linear_state"]


def _block_on_result(result) -> None:
    for candidate in (result.final_time, result.n_steps, result.done, result.fail_code):
        if candidate is not None:
            jax.block_until_ready(candidate)
            return
    if result.accepted_mask is not None:
        jax.block_until_ready(result.accepted_mask)


def _tree_to_host_numpy(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), tree)


def _leaf_max_abs_delta(reference, other) -> float:
    ref_leaves = jax.tree_util.tree_leaves(reference)
    other_leaves = jax.tree_util.tree_leaves(other)
    if len(ref_leaves) != len(other_leaves):
        return float("nan")
    max_delta = 0.0
    for ref_leaf, other_leaf in zip(ref_leaves, other_leaves):
        ref_arr = np.asarray(ref_leaf)
        other_arr = np.asarray(other_leaf)
        if ref_arr.shape != other_arr.shape:
            return float("nan")
        delta = float(np.max(np.abs(other_arr - ref_arr)))
        max_delta = max(max_delta, delta)
    return max_delta


def _accepted_count(result) -> int | None:
    if result.accepted_mask is None:
        return None
    return int(jnp.sum(jnp.asarray(result.accepted_mask, dtype=jnp.int32)))


def _build_config(
    config_path: Path,
    *,
    backend: str,
    device: str,
    rhs_mode: str,
    flux_model: str,
    er_init_mode: str,
    n_theta: int | None,
    n_zeta: int | None,
    n_xi: int | None,
    ntx_face_response_mode: str,
    ntx_radial_batch_size: int | None,
    ntx_scan_batch_size: int | None,
    ntx_response_anchor_count: int | None,
    ntx_use_remat: bool,
):
    config = NEOPAX.prepare_config(config_path, backend=backend, device=device)
    config = copy.deepcopy(config)

    general = config.setdefault("general", {})
    general["mode"] = "transport"

    neoclassical = config.setdefault("neoclassical", {})
    if flux_model != "keep":
        neoclassical["flux_model"] = str(flux_model)
    active_flux_model = neoclassical.get("flux_model", "ntx_database")

    profiles = config.setdefault("profiles", {})
    if er_init_mode != "keep":
        profiles["er_initialization_mode"] = str(er_init_mode)

    if active_flux_model == "ntx_exact_lij_runtime":
        if n_theta is None:
            n_theta = DEFAULT_NTX_EXACT_N_THETA
        if n_zeta is None:
            n_zeta = DEFAULT_NTX_EXACT_N_ZETA
        if n_xi is None:
            n_xi = DEFAULT_NTX_EXACT_N_XI
        if n_theta is not None:
            neoclassical["ntx_exact_n_theta"] = int(n_theta)
        if n_zeta is not None:
            neoclassical["ntx_exact_n_zeta"] = int(n_zeta)
        if n_xi is not None:
            neoclassical["ntx_exact_n_xi"] = int(n_xi)
        if ntx_radial_batch_size not in (None, 0):
            neoclassical["ntx_exact_radial_batch_size"] = int(ntx_radial_batch_size)
        if ntx_scan_batch_size not in (None, 0):
            neoclassical["ntx_exact_scan_batch_size"] = int(ntx_scan_batch_size)
        if ntx_response_anchor_count not in (None, 0):
            neoclassical["ntx_exact_response_anchor_count"] = int(ntx_response_anchor_count)
        if ntx_use_remat:
            neoclassical["ntx_exact_use_remat"] = True
        if rhs_mode in {"lagged_response", "lagged_transport_response"}:
            neoclassical["ntx_exact_face_response_mode"] = str(ntx_face_response_mode)

    ambipolarity = config.setdefault("ambipolarity", {})
    ambipolarity["er_ambipolar_plot"] = False
    ambipolarity["er_ambipolar_overlay_reference_er"] = False

    solver = config.setdefault("transport_solver", {})
    solver["transport_solver_backend"] = str(backend)
    solver["rhs_mode"] = str(rhs_mode)
    solver["debug_stage_markers"] = False

    transport_output = config.setdefault("transport_output", {})
    transport_output["transport_plot"] = False
    transport_output["transport_write_hdf5"] = False

    return config, active_flux_model


def _run_once(config):
    t_start = time.perf_counter()
    result = NEOPAX.run(config)
    _block_on_result(result)
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
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Execution device to request from NEOPAX/JAX.",
    )
    parser.add_argument(
        "--flux-model",
        default=DEFAULT_FLUX_MODEL,
        choices=["keep", "ntx_database", "ntx_exact_lij_runtime", "ntx_scan_runtime"],
        help=(
            "Neoclassical flux model to benchmark. "
            "'keep' preserves whatever the base TOML already uses."
        ),
    )
    parser.add_argument(
        "--er-init-mode",
        default=DEFAULT_ER_INIT_MODE,
        choices=["keep", "analytical", "ambipolar_min_entropy"],
        help=(
            "Override profiles.er_initialization_mode for the benchmark. "
            "Default is 'keep', so the benchmark TOML controls whether Er starts "
            "from the ambipolar root finder or an analytical profile."
        ),
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per rhs_mode before timing.")
    parser.add_argument("--repeat", type=int, default=1, help="Timed runs per rhs_mode.")
    parser.add_argument(
        "--rhs-modes",
        nargs="+",
        default=DEFAULT_RHS_MODES,
        help="RHS modes to compare.",
    )
    parser.add_argument(
        "--ntx-n-theta",
        type=int,
        default=None,
        help="Override ntx_exact_n_theta when benchmarking ntx_exact_lij_runtime.",
    )
    parser.add_argument(
        "--ntx-n-zeta",
        type=int,
        default=None,
        help="Override ntx_exact_n_zeta when benchmarking ntx_exact_lij_runtime.",
    )
    parser.add_argument(
        "--ntx-n-xi",
        type=int,
        default=None,
        help="Override ntx_exact_n_xi when benchmarking ntx_exact_lij_runtime.",
    )
    parser.add_argument(
        "--ntx-face-response-mode",
        default=DEFAULT_NTX_EXACT_FACE_RESPONSE_MODE,
        choices=["face_local_response", "interpolate_center_response"],
        help=(
            "Exact-runtime NTX face response mode to use for lagged_response benchmarks. "
            "Black-box mode continues to use the normal live face-local path."
        ),
    )
    parser.add_argument(
        "--ntx-radial-batch-size",
        type=int,
        default=DEFAULT_NTX_EXACT_RADIAL_BATCH_SIZE,
        help=(
            "Exact-runtime NTX radial hybrid batch size. "
            "Unset or 0 keeps the default unbatched lax.map path; values >1 use "
            "lax.map over chunks and vmap within each chunk."
        ),
    )
    parser.add_argument(
        "--ntx-scan-batch-size",
        type=int,
        default=DEFAULT_NTX_EXACT_SCAN_BATCH_SIZE,
        help=(
            "Exact-runtime NTX coefficient-scan batch size across the energy-grid cases. "
            "Unset or 0 keeps the default full vmap over n_x; values >1 apply NTX-style "
            "chunking to reduce peak memory."
        ),
    )
    parser.add_argument(
        "--ntx-response-anchor-count",
        type=int,
        default=DEFAULT_NTX_EXACT_RESPONSE_ANCHOR_COUNT,
        help=(
            "If set below the full transport radial count, build the lagged exact-runtime NTX "
            "response only on that many radial anchor points and interpolate the reduced "
            "transport-moment response back to the full transport grid."
        ),
    )
    parser.add_argument(
        "--ntx-response-anchor-counts",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Run a sweep over multiple exact-runtime lagged-response anchor counts. "
            "Useful for comparisons such as 7 vs 14 anchors. "
            "When set, each anchor count is benchmarked as a separate configuration."
        ),
    )
    parser.add_argument(
        "--ntx-use-remat",
        action="store_true",
        default=DEFAULT_NTX_EXACT_USE_REMAT,
        help=(
            "Enable jax.checkpoint/rematerialization on the local exact-runtime NTX solve path. "
            "This can reduce peak memory at the cost of extra recomputation."
        ),
    )
    parser.add_argument(
        "--compute-final-state-delta",
        action="store_true",
        help=(
            "Compute max|delta| of final_state versus the first rhs_mode. "
            "Disabled by default because materializing the full final state can be very expensive."
        ),
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    sweep_anchor_counts = args.ntx_response_anchor_counts
    if sweep_anchor_counts is None:
        sweep_anchor_counts = [args.ntx_response_anchor_count]

    mode_results = []
    reference_final_state = None

    print(f"[benchmark] config={config_path}")
    print(f"[benchmark] backend={args.backend}")
    print(f"[benchmark] device={args.device}")
    print(f"[benchmark] requested_flux_model={args.flux_model}")
    print(f"[benchmark] er_init_mode={args.er_init_mode}")
    print(f"[benchmark] ntx_face_response_mode={args.ntx_face_response_mode}")
    print(f"[benchmark] ntx_radial_batch_size={args.ntx_radial_batch_size}")
    print(f"[benchmark] ntx_scan_batch_size={args.ntx_scan_batch_size}")
    print(f"[benchmark] ntx_response_anchor_counts={sweep_anchor_counts}")
    print(f"[benchmark] ntx_use_remat={args.ntx_use_remat}")
    print(f"[benchmark] compute_final_state_delta={args.compute_final_state_delta}")
    print(f"[benchmark] rhs_modes={args.rhs_modes}")

    for sweep_anchor_count in sweep_anchor_counts:
        active_flux_model = None
        for rhs_mode in args.rhs_modes:
            config, active_flux_model = _build_config(
                config_path,
                backend=args.backend,
                device=args.device,
                rhs_mode=rhs_mode,
                flux_model=args.flux_model,
                er_init_mode=args.er_init_mode,
                n_theta=args.ntx_n_theta,
                n_zeta=args.ntx_n_zeta,
                n_xi=args.ntx_n_xi,
                ntx_face_response_mode=args.ntx_face_response_mode,
                ntx_radial_batch_size=args.ntx_radial_batch_size,
                ntx_scan_batch_size=args.ntx_scan_batch_size,
                ntx_response_anchor_count=sweep_anchor_count,
                ntx_use_remat=args.ntx_use_remat,
            )

            if rhs_mode == args.rhs_modes[0]:
                print(f"[benchmark] active_flux_model={active_flux_model}")
                if active_flux_model == "ntx_exact_lij_runtime" and (
                    args.ntx_n_theta is not None or args.ntx_n_zeta is not None or args.ntx_n_xi is not None
                ):
                    print(
                        "[benchmark] ntx exact runtime resolution overrides:",
                        f"n_theta={config['neoclassical'].get('ntx_exact_n_theta')}",
                        f"n_zeta={config['neoclassical'].get('ntx_exact_n_zeta')}",
                        f"n_xi={config['neoclassical'].get('ntx_exact_n_xi')}",
                        f"radial_batch_size={config['neoclassical'].get('ntx_exact_radial_batch_size', 0)}",
                        f"scan_batch_size={config['neoclassical'].get('ntx_exact_scan_batch_size', 0)}",
                        f"response_anchor_count={config['neoclassical'].get('ntx_exact_response_anchor_count', 0)}",
                        f"use_remat={config['neoclassical'].get('ntx_exact_use_remat', False)}",
                    )
                if active_flux_model == "ntx_exact_lij_runtime":
                    print(
                        "[benchmark] ntx exact runtime face response mode:",
                        config["neoclassical"].get("ntx_exact_face_response_mode", "face_local_response"),
                    )
                    anchor_count = config["neoclassical"].get("ntx_exact_response_anchor_count", 0)
                    if int(anchor_count) > 0:
                        print(
                            "[benchmark] ntx exact runtime lagged response mode:",
                            f"coarse_anchor_interpolated(anchor_count={anchor_count}, derivatives=Er+log_nu_star)",
                        )
                    else:
                        print(
                            "[benchmark] ntx exact runtime lagged response mode:",
                            "full_radius_response",
                        )

            for _ in range(max(0, args.warmup)):
                print(
                    f"[benchmark] warmup compile/run rhs_mode={rhs_mode}"
                    f" anchor_count={sweep_anchor_count}"
                )
                _run_once(config)

            timed_runs = []
            last_result = None
            for _ in range(max(1, args.repeat)):
                last_result, wall_seconds = _run_once(config)
                timed_runs.append(wall_seconds)

            final_state_delta = None
            if args.compute_final_state_delta and last_result is not None:
                if reference_final_state is None:
                    reference_final_state = _tree_to_host_numpy(last_result.final_state)
                else:
                    final_state_delta = _leaf_max_abs_delta(reference_final_state, _tree_to_host_numpy(last_result.final_state))

            mode_results.append(
                {
                    "rhs_mode": rhs_mode,
                    "flux_model": active_flux_model,
                    "response_anchor_count": sweep_anchor_count,
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
    print("rhs_mode                 anchors     mean_s     best_s     n_steps  accepted  failed  fail_code  final_t   max|delta|")
    print("-" * 118)
    for row in mode_results:
        final_time_str = str(None if row["final_time"] is None else round(row["final_time"], 6))
        final_delta = row["final_state_max_abs_delta_vs_first"]
        final_delta_str = "None" if final_delta is None else f"{final_delta:.3e}"
        print(
            f"{row['rhs_mode']:<23}"
            f"{str(row['response_anchor_count']):>9}"
            f"{row['mean_wall_seconds']:>10.3f}"
            f"{row['best_wall_seconds']:>11.3f}"
            f"{str(row['n_steps']):>12}"
            f"{str(row['accepted_steps']):>10}"
            f"{str(row['failed']):>8}"
            f"{str(row['fail_code']):>11}"
            f"{final_time_str:>10}"
            f"{final_delta_str:>14}"
        )


if __name__ == "__main__":
    main()
