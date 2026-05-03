"""Benchmark ambipolar root initialization across flux-model choices.

This isolates the ambipolarity/root-finding path from the transport solver.
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
from NEOPAX._orchestrator import build_runtime_context, run_ambipolarity

DEFAULT_CONFIG = Path("examples/benchmarks/Solve_Transport_equations_noHe_radau_benchmark.toml")
DEFAULT_NTX_SCAN_RHO = [0.12247, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
DEFAULT_NTX_SCAN_NU_V = [1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3]
DEFAULT_NTX_SCAN_ER_TILDE = [0.0, 1.0e-6, 3.0e-6, 1.0e-5, 3.0e-5, 1.0e-4]


def _build_config(
    config_path: Path,
    *,
    device: str,
    flux_model: str,
    n_theta: int | None,
    n_zeta: int | None,
    n_xi: int | None,
) -> dict:
    config = NEOPAX.prepare_config(config_path, mode="ambipolarity", device=device)
    config = copy.deepcopy(config)
    config.setdefault("general", {})["mode"] = "ambipolarity"
    neo = config.setdefault("neoclassical", {})
    neo["flux_model"] = str(flux_model)
    if flux_model == "ntx_exact_lij_runtime":
        if n_theta is not None:
            neo["ntx_exact_n_theta"] = int(n_theta)
        if n_zeta is not None:
            neo["ntx_exact_n_zeta"] = int(n_zeta)
        if n_xi is not None:
            neo["ntx_exact_n_xi"] = int(n_xi)
    elif flux_model == "ntx_scan_runtime":
        neo.setdefault("ntx_scan_rho", list(DEFAULT_NTX_SCAN_RHO))
        neo.setdefault("ntx_scan_nu_v", list(DEFAULT_NTX_SCAN_NU_V))
        neo.setdefault("ntx_scan_er_tilde", list(DEFAULT_NTX_SCAN_ER_TILDE))
        if n_theta is not None:
            neo["ntx_scan_n_theta"] = int(n_theta)
        if n_zeta is not None:
            neo["ntx_scan_n_zeta"] = int(n_zeta)
        if n_xi is not None:
            neo["ntx_scan_n_xi"] = int(n_xi)
    amb = config.setdefault("ambipolarity", {})
    amb["er_ambipolar_plot"] = False
    amb["er_ambipolar_write_hdf5"] = False
    out = config.setdefault("transport_output", {})
    out["transport_plot"] = False
    out["transport_write_hdf5"] = False
    return config


def _block_on_ambipolar_result(result: dict) -> None:
    candidate = result.get("best_root")
    if candidate is not None:
        jax.block_until_ready(candidate)
        return
    candidate = result.get("roots_3")
    if candidate is not None:
        jax.block_until_ready(candidate)


def _run_once(config: dict):
    runtime, state = build_runtime_context(config)
    t0 = time.perf_counter()
    result = run_ambipolarity(config, runtime, state)
    _block_on_ambipolar_result(result)
    dt = time.perf_counter() - t0
    return result, dt


def _best_root_summary(best_root) -> tuple[float, float]:
    arr = jnp.asarray(best_root)
    finite = jnp.isfinite(arr)
    if not bool(jnp.any(finite)):
        return float("nan"), float("nan")
    vals = arr[finite]
    return float(jnp.min(vals)), float(jnp.max(vals))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument(
        "--flux-models",
        nargs="+",
        default=["ntx_database", "ntx_exact_lij_runtime"],
        choices=["ntx_database", "ntx_exact_lij_runtime", "ntx_scan_runtime"],
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--ntx-n-theta", type=int, default=5)
    parser.add_argument("--ntx-n-zeta", type=int, default=21)
    parser.add_argument("--ntx-n-xi", type=int, default=32)
    args = parser.parse_args()

    config_path = Path(args.config)
    print(f"[amb-bench] config={config_path}")
    print(f"[amb-bench] device={args.device}")
    print(f"[amb-bench] flux_models={args.flux_models}")

    rows = []
    for flux_model in args.flux_models:
        config = _build_config(
            config_path,
            device=args.device,
            flux_model=flux_model,
            n_theta=args.ntx_n_theta,
            n_zeta=args.ntx_n_zeta,
            n_xi=args.ntx_n_xi,
        )
        timings = []
        last = None
        for _ in range(max(1, args.repeat)):
            last, dt = _run_once(config)
            timings.append(dt)
        root_min, root_max = _best_root_summary(last["best_root"])
        rows.append(
            {
                "flux_model": flux_model,
                "mean_s": sum(timings) / len(timings),
                "best_s": min(timings),
                "root_min": root_min,
                "root_max": root_max,
            }
        )

    print()
    print("flux_model                mean_s     best_s     root_min      root_max")
    print("-" * 72)
    for row in rows:
        print(
            f"{row['flux_model']:<24}"
            f"{row['mean_s']:>10.3f}"
            f"{row['best_s']:>11.3f}"
            f"{row['root_min']:>13.6e}"
            f"{row['root_max']:>14.6e}"
        )


if __name__ == "__main__":
    main()
