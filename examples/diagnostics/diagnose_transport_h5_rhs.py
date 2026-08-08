#!/usr/bin/env python
"""Diagnose flux/RHS finiteness from a saved transport_solution.h5 snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from NEOPAX._orchestrator import build_runtime_context, load_config, prepare_transport_solver_components
from NEOPAX._state import TransportState


def _as_array(value):
    return jnp.asarray(value)


def _first_bad_flat(value):
    arr = _as_array(value)
    bad = jnp.ravel(jnp.logical_not(jnp.isfinite(arr)))
    if not bool(jax.device_get(jnp.any(bad))):
        return None
    return int(jax.device_get(jnp.argmax(bad)))


def _stats(label: str, value):
    arr = _as_array(value)
    finite = bool(jax.device_get(jnp.all(jnp.isfinite(arr))))
    print(
        f"{label}: finite={finite} shape={tuple(arr.shape)} "
        f"min={float(jax.device_get(jnp.nanmin(arr))):.16e} "
        f"max={float(jax.device_get(jnp.nanmax(arr))):.16e} "
        f"max_abs={float(jax.device_get(jnp.nanmax(jnp.abs(arr)))):.16e} "
        f"first_bad_flat={_first_bad_flat(arr)}"
    )


def _walk_stats(prefix: str, value):
    if value is None:
        print(f"{prefix}: None")
        return
    if isinstance(value, dict):
        for key in sorted(value):
            _walk_stats(f"{prefix}.{key}", value[key])
        return
    if hasattr(value, "__dataclass_fields__"):
        for key in value.__dataclass_fields__:
            if key.startswith("_"):
                continue
            _walk_stats(f"{prefix}.{key}", getattr(value, key))
        return
    try:
        _stats(prefix, value)
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        print(f"{prefix}: skipped ({type(exc).__name__}: {exc})")


def _read_snapshot(path: Path, time_index: int) -> TransportState:
    with h5py.File(path, "r") as h5:
        missing = [name for name in ("density", "pressure", "Er") if name not in h5]
        if missing:
            raise KeyError(f"{path} is missing datasets required for TransportState: {missing}")
        density = jnp.asarray(h5["density"][time_index])
        pressure = jnp.asarray(h5["pressure"][time_index])
        er = jnp.asarray(h5["Er"][time_index])
    return TransportState(density=density, pressure=pressure, Er=er)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="NEOPAX TOML used for the transport run.")
    parser.add_argument("--solution", required=True, type=Path, help="Saved transport_solution.h5 file.")
    parser.add_argument("--time-index", type=int, default=-1, help="Snapshot index to diagnose, default final.")
    parser.add_argument(
        "--enable-ntx-debug",
        action="store_true",
        help="Also enable low-level [ntx-nonfinite] debug prints during this diagnostic.",
    )
    args = parser.parse_args(argv)

    if args.enable_ntx_debug:
        import os

        os.environ["NEOPAX_NTX_NONFINITE_DEBUG"] = "1"

    config = load_config(args.config)
    runtime, initial_state = build_runtime_context(config)
    state = _read_snapshot(args.solution, args.time_index)
    prepared = prepare_transport_solver_components(config, runtime, initial_state)
    equation_system = prepared["equation_system"]
    working_state, _ = equation_system._prepare_working_state(state)

    print(f"[diagnose-h5-rhs] config={args.config.resolve()}")
    print(f"[diagnose-h5-rhs] solution={args.solution.resolve()} time_index={args.time_index}")
    print("[diagnose-h5-rhs] saved state")
    _walk_stats("state", state)
    print("[diagnose-h5-rhs] prepared working state")
    _walk_stats("working_state", working_state)

    print("[diagnose-h5-rhs] direct shared flux")
    direct_flux = runtime.models.flux(working_state)
    _walk_stats("direct_flux", direct_flux)

    print("[diagnose-h5-rhs] direct RHS")
    direct_rhs = equation_system._evaluate_state(state, lagged_response=None)
    _walk_stats("direct_rhs", direct_rhs)

    print("[diagnose-h5-rhs] lagged response built at this snapshot")
    lagged_response = equation_system.build_lagged_response(state)
    _walk_stats("lagged_response", lagged_response)

    print("[diagnose-h5-rhs] lagged shared flux evaluated at same snapshot")
    lagged_flux = runtime.models.flux.evaluate_with_lagged_response(
        working_state,
        lagged_response.flux_response,
    )
    _walk_stats("lagged_flux", lagged_flux)

    print("[diagnose-h5-rhs] lagged RHS evaluated at same snapshot")
    lagged_rhs = equation_system._evaluate_state(state, lagged_response=lagged_response)
    _walk_stats("lagged_rhs", lagged_rhs)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
