"""Benchmark and compare initial fluxes across database, exact black-box,
and exact lagged-response variants from the same initial state.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import NEOPAX
from NEOPAX._orchestrator import build_runtime_context

DEFAULT_CONFIG = Path("examples/benchmarks/Solve_Transport_equations_noHe_radau_benchmark.toml")


def _build_config(
    config_path: Path,
    *,
    device: str,
    flux_model: str,
    er_init_mode: str,
    anchor_count: int | None,
    n_theta: int | None,
    n_zeta: int | None,
    n_xi: int | None,
) -> dict:
    config = NEOPAX.prepare_config(config_path, mode="transport", device=device)
    config = copy.deepcopy(config)
    config.setdefault("general", {})["mode"] = "transport"
    config.setdefault("profiles", {})["er_initialization_mode"] = str(er_init_mode)
    neo = config.setdefault("neoclassical", {})
    neo["flux_model"] = str(flux_model)
    if flux_model == "ntx_exact_lij_runtime":
        if n_theta is not None:
            neo["ntx_exact_n_theta"] = int(n_theta)
        if n_zeta is not None:
            neo["ntx_exact_n_zeta"] = int(n_zeta)
        if n_xi is not None:
            neo["ntx_exact_n_xi"] = int(n_xi)
        if anchor_count not in (None, 0):
            neo["ntx_exact_response_anchor_count"] = int(anchor_count)
        else:
            neo.pop("ntx_exact_response_anchor_count", None)
        neo["ntx_exact_face_response_mode"] = "interpolate_center_response"
    out = config.setdefault("transport_output", {})
    out["transport_plot"] = False
    out["transport_write_hdf5"] = False
    return config


def _block_on_fluxes(fluxes: dict) -> None:
    for key in ("Gamma", "Q", "Upar"):
        if key in fluxes:
            jax.block_until_ready(fluxes[key])
            return


def _block_on_tree(tree) -> None:
    leaves = jax.tree_util.tree_leaves(tree)
    for leaf in leaves:
        try:
            jax.block_until_ready(leaf)
            return
        except Exception:
            continue


def _flatten_fluxes(fluxes: dict):
    return {key: jnp.asarray(value) for key, value in fluxes.items()}


def _flux_delta(reference: dict, other: dict) -> float:
    max_delta = 0.0
    for key in sorted(reference.keys()):
        if key not in other:
            return float("nan")
        ref_arr = jnp.asarray(reference[key])
        other_arr = jnp.asarray(other[key])
        if ref_arr.shape != other_arr.shape:
            return float("nan")
        delta = float(jnp.max(jnp.abs(other_arr - ref_arr)))
        max_delta = max(max_delta, delta)
    return max_delta


def _evaluate_black_box(flux_model, state):
    t0 = time.perf_counter()
    fluxes = flux_model(state)
    _block_on_fluxes(fluxes)
    return _flatten_fluxes(fluxes), time.perf_counter() - t0


def _evaluate_lagged(flux_model, state):
    t0 = time.perf_counter()
    response = flux_model.build_lagged_response(state)
    _block_on_tree(response)
    build_dt = time.perf_counter() - t0

    t1 = time.perf_counter()
    fluxes = flux_model.evaluate_with_lagged_response(state, response)
    _block_on_fluxes(fluxes)
    eval_dt = time.perf_counter() - t1
    return _flatten_fluxes(fluxes), build_dt, eval_dt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--er-init-mode", default="keep", choices=["keep", "analytical", "ambipolar_min_entropy"])
    parser.add_argument("--anchor-count", type=int, default=7)
    parser.add_argument("--ntx-n-theta", type=int, default=5)
    parser.add_argument("--ntx-n-zeta", type=int, default=21)
    parser.add_argument("--ntx-n-xi", type=int, default=32)
    args = parser.parse_args()

    config_path = Path(args.config)
    print(f"[flux-bench] config={config_path}")
    print(f"[flux-bench] device={args.device}")
    print(f"[flux-bench] er_init_mode={args.er_init_mode}")
    print(f"[flux-bench] anchor_count={args.anchor_count}")

    cases = [
        ("database_black_box", "ntx_database", None, "black_box"),
        ("exact_black_box_full", "ntx_exact_lij_runtime", 0, "black_box"),
        ("exact_lagged_full", "ntx_exact_lij_runtime", 0, "lagged"),
        ("exact_lagged_coarse", "ntx_exact_lij_runtime", args.anchor_count, "lagged"),
    ]

    results = {}
    reference_fluxes = None
    reference_er = None

    for label, flux_model_name, anchor_count, eval_mode in cases:
        config = _build_config(
            config_path,
            device=args.device,
            flux_model=flux_model_name,
            er_init_mode=args.er_init_mode,
            anchor_count=anchor_count,
            n_theta=args.ntx_n_theta,
            n_zeta=args.ntx_n_zeta,
            n_xi=args.ntx_n_xi,
        )
        runtime, state = build_runtime_context(config)
        flux_model = runtime.models.flux
        if eval_mode == "black_box":
            fluxes, total_dt = _evaluate_black_box(flux_model, state)
            build_dt = None
            eval_dt = total_dt
        else:
            fluxes, build_dt, eval_dt = _evaluate_lagged(flux_model, state)
            total_dt = build_dt + eval_dt
        er_arr = jnp.asarray(state.Er)
        if reference_fluxes is None:
            reference_fluxes = fluxes
            reference_er = er_arr
        results[label] = {
            "er": er_arr,
            "fluxes": fluxes,
            "build_dt": build_dt,
            "eval_dt": eval_dt,
            "total_dt": total_dt,
            "er_delta": float(jnp.max(jnp.abs(er_arr - reference_er))),
            "flux_delta": _flux_delta(reference_fluxes, fluxes),
        }

    print()
    print("case                     total_s    build_s     eval_s      max|dEr|     max|dflux|")
    print("-" * 90)
    for label, *_rest in cases:
        row = results[label]
        build_s = "None" if row["build_dt"] is None else f"{row['build_dt']:.3f}"
        print(
            f"{label:<24}"
            f"{row['total_dt']:>9.3f}"
            f"{build_s:>11}"
            f"{row['eval_dt']:>11.3f}"
            f"{row['er_delta']:>14.6e}"
            f"{row['flux_delta']:>15.6e}"
        )


if __name__ == "__main__":
    main()
