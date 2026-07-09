"""Compare frozen-file and realtime-VMEC initial ambipolar Er setup.

This diagnostic intentionally stops after ``build_runtime_context``.  That
builds geometry, NTX support, the initial state, and any configured ambipolar
Er initialization, but it does not run the transport time integrator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from NEOPAX._orchestrator import build_runtime_context, load_config  # noqa: E402


DEFAULT_FROZEN_CONFIG = (
    ROOT / "examples" / "benchmarks" / "Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_benchmark.toml"
)
DEFAULT_REALTIME_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml"
)

GEOMETRY_FIELDS = (
    "a_b",
    "Psia_value",
    "B0",
    "B0prime",
    "B_10",
    "Bsqav",
    "G_PS",
    "I_value",
    "G_value",
    "Vprime",
    "Vprime_half",
    "curvature",
    "epsilon_t",
    "iota",
    "sqrtg00_value",
)


def _stats(left, right) -> dict[str, Any]:
    left_np = np.asarray(jax.device_get(left), dtype=float)
    right_np = np.asarray(jax.device_get(right), dtype=float)
    diff = right_np - left_np
    denom = max(float(np.linalg.norm(left_np)), 1.0e-30)
    return {
        "shape": list(left_np.shape),
        "left_min": float(np.nanmin(left_np)),
        "left_max": float(np.nanmax(left_np)),
        "right_min": float(np.nanmin(right_np)),
        "right_max": float(np.nanmax(right_np)),
        "max_abs": float(np.nanmax(np.abs(diff))),
        "rel_l2": float(np.linalg.norm(diff) / denom),
    }


def _load_for_initial_ambipolarity(path: Path):
    config = load_config(path)
    # Keep this diagnostic read-only with respect to benchmark plot outputs.
    amb_cfg = config.setdefault("ambipolarity", {})
    amb_cfg["er_ambipolar_plot"] = False
    amb_cfg["er_ambipolar_overlay_reference_er"] = False
    runtime, state = build_runtime_context(config)
    return config, runtime, state


def _print_top(title: str, rows: dict[str, dict[str, Any]], *, top: int) -> None:
    print(f"[compare] {title}")
    sorted_rows = sorted(rows.items(), key=lambda item: item[1]["rel_l2"], reverse=True)
    for name, row in sorted_rows[:top]:
        print(
            f"  - {name}: rel_l2={row['rel_l2']:.6e} "
            f"max_abs={row['max_abs']:.6e} "
            f"left=[{row['left_min']:.6e}, {row['left_max']:.6e}] "
            f"right=[{row['right_min']:.6e}, {row['right_max']:.6e}]"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-config", default=str(DEFAULT_FROZEN_CONFIG))
    parser.add_argument("--realtime-config", default=str(DEFAULT_REALTIME_CONFIG))
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    frozen_path = Path(args.frozen_config).resolve()
    realtime_path = Path(args.realtime_config).resolve()
    print(f"[compare] frozen config:  {frozen_path}")
    print(f"[compare] realtime config:{realtime_path}")
    print("[compare] building frozen initial runtime/state")
    _, frozen_runtime, frozen_state = _load_for_initial_ambipolarity(frozen_path)
    print("[compare] building realtime initial runtime/state")
    _, realtime_runtime, realtime_state = _load_for_initial_ambipolarity(realtime_path)

    state_rows = {
        "Er": _stats(frozen_state.Er, realtime_state.Er),
        "density": _stats(frozen_state.density, realtime_state.density),
        "pressure": _stats(frozen_state.pressure, realtime_state.pressure),
    }
    geometry_rows = {
        name: _stats(getattr(frozen_runtime.geometry, name), getattr(realtime_runtime.geometry, name))
        for name in GEOMETRY_FIELDS
    }

    _print_top("initial state frozen vs realtime", state_rows, top=int(args.top))
    _print_top("geometry frozen vs realtime", geometry_rows, top=int(args.top))

    payload = {
        "frozen_config": str(frozen_path),
        "realtime_config": str(realtime_path),
        "initial_state": state_rows,
        "geometry": geometry_rows,
    }
    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[compare] wrote {out}")


if __name__ == "__main__":
    main()
