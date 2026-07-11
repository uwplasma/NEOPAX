"""Compare NEOPAX realtime VMEC state against vmec_jax's public driver state.

This diagnostic deliberately stops before WOUT, NTX, Boozer, and transport.
It answers the primitive question: does NEOPAX's realtime geometry path build
the same VMEC state as the public vmec_jax multigrid driver from the same input?
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from NEOPAX._geometry_autodiff import (  # noqa: E402
    _solve_state_for_single_param,
    build_geometry_autodiff_context,
)
from NEOPAX._orchestrator import load_config  # noqa: E402

DEFAULT_REALTIME_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml"
)


def _numeric_leaves(obj: Any, prefix: str = "") -> dict[str, np.ndarray]:
    leaves: dict[str, np.ndarray] = {}
    if dataclasses.is_dataclass(obj):
        for field in dataclasses.fields(obj):
            name = field.name
            leaves.update(_numeric_leaves(getattr(obj, name), f"{prefix}.{name}" if prefix else name))
        return leaves
    if isinstance(obj, dict):
        for key, value in obj.items():
            leaves.update(_numeric_leaves(value, f"{prefix}.{key}" if prefix else str(key)))
        return leaves
    if isinstance(obj, (tuple, list)):
        for idx, value in enumerate(obj):
            leaves.update(_numeric_leaves(value, f"{prefix}[{idx}]"))
        return leaves
    try:
        arr = np.asarray(jax.device_get(obj))
    except Exception:
        return leaves
    if np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_):
        leaves[prefix or "value"] = arr
    return leaves


def _stats(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape:
        return {
            "shape_mismatch": True,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
        }
    left_f = left.astype(float, copy=False)
    right_f = right.astype(float, copy=False)
    diff = left_f - right_f
    finite = np.isfinite(left_f) & np.isfinite(right_f) & np.isfinite(diff)
    if not np.any(finite):
        return {
            "shape": list(left.shape),
            "max_abs": float("nan"),
            "rel_l2": float("nan"),
            "left_min": float("nan"),
            "left_max": float("nan"),
            "right_min": float("nan"),
            "right_max": float("nan"),
        }
    left_finite = left_f[finite]
    right_finite = right_f[finite]
    diff_finite = diff[finite]
    denom = max(float(np.linalg.norm(left_finite)), float(np.linalg.norm(right_finite)), 1.0e-300)
    return {
        "shape": list(left.shape),
        "max_abs": float(np.max(np.abs(diff_finite))),
        "rel_l2": float(np.linalg.norm(diff_finite) / denom),
        "left_min": float(np.min(left_finite)),
        "left_max": float(np.max(left_finite)),
        "right_min": float(np.min(right_finite)),
        "right_max": float(np.max(right_finite)),
    }


def _compare_states(left_state: Any, right_state: Any) -> dict[str, Any]:
    left_leaves = _numeric_leaves(left_state)
    right_leaves = _numeric_leaves(right_state)
    names = sorted(set(left_leaves) | set(right_leaves))
    rows: dict[str, Any] = {}
    for name in names:
        if name not in left_leaves:
            rows[name] = {"missing": "neopax_realtime"}
            continue
        if name not in right_leaves:
            rows[name] = {"missing": "vmec_driver"}
            continue
        rows[name] = _stats(left_leaves[name], right_leaves[name])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--realtime-config", default=str(DEFAULT_REALTIME_CONFIG))
    parser.add_argument("--solver-device", default=None)
    parser.add_argument("--driver-solver", default=None)
    parser.add_argument("--driver-solver-mode", default=None)
    parser.add_argument(
        "--neopax-lane",
        choices=("forward", "ad"),
        default=None,
        help="Override geometry.vmec_lane for the NEOPAX state build. Use 'ad' to test the implicit VMEC lane.",
    )
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()

    realtime_config = Path(args.realtime_config).expanduser().resolve()
    config = load_config(realtime_config)
    geom_cfg = dict(config.get("geometry", {}))
    context = build_geometry_autodiff_context(
        geom_cfg.get("vmec_input_file"),
        param_family=str(geom_cfg.get("vmec_param_family", "RBC")),
        param_m=int(geom_cfg.get("vmec_m_index", 0)),
        param_n=int(geom_cfg.get("vmec_n_index", 0)),
        mboz=geom_cfg.get("mboz", geom_cfg.get("vmec_mboz")),
        nboz=geom_cfg.get("nboz", geom_cfg.get("vmec_nboz")),
    )
    input_path = Path(context.input_path).expanduser().resolve()
    print(f"[compare] input:           {input_path}")
    print(f"[compare] realtime config: {realtime_config}")
    neopax_lane = str(args.neopax_lane or geom_cfg.get("vmec_lane", "forward")).strip().lower()
    print(f"[compare] building NEOPAX realtime VMEC state lane={neopax_lane}")
    neopax_state = _solve_state_for_single_param(
        context,
        jnp.asarray(float(geom_cfg.get("vmec_param_delta", 0.0)), dtype=jnp.float64),
        lane=neopax_lane,
        max_iter=geom_cfg.get("vmec_max_iter"),
        step_size=geom_cfg.get("vmec_step_size"),
        jacobian_penalty=float(geom_cfg.get("vmec_jacobian_penalty", 1.0e3)),
    )

    import vmec_jax as vj

    run_kwargs: dict[str, Any] = {"verbose": False}
    if args.solver_device is not None:
        run_kwargs["solver_device"] = args.solver_device
    if args.driver_solver is not None:
        run_kwargs["solver"] = args.driver_solver
    if args.driver_solver_mode is not None:
        run_kwargs["solver_mode"] = args.driver_solver_mode
    if hasattr(vj, "run_fixed_boundary"):
        print(f"[compare] running vmec_jax.run_fixed_boundary with kwargs={run_kwargs}")
        driver_run = vj.run_fixed_boundary(input_path, **run_kwargs)
    else:
        solve_kwargs: dict[str, Any] = {
            "mode": "cli",
            "verbose": False,
        }
        inp = vj.VmecInput.from_file(input_path)
        if geom_cfg.get("vmec_max_iter") is not None:
            niter_array = np.asarray(inp.niter_array, dtype=np.int64).copy()
            niter_array[-1] = int(geom_cfg.get("vmec_max_iter"))
            solve_kwargs["niter_array"] = niter_array
        if geom_cfg.get("vmec_step_size") is not None:
            solve_kwargs["time_step"] = float(geom_cfg.get("vmec_step_size"))
        if args.solver_device is not None:
            solve_kwargs["device"] = args.solver_device
        if args.driver_solver_mode is not None:
            solve_kwargs["mode"] = args.driver_solver_mode
        if args.driver_solver is not None:
            print("[compare] --driver-solver is ignored by the current vmec_jax.solve API")
        from vmec_jax.core.multigrid import solve_multigrid

        print(f"[compare] running vmec_jax.solve_multigrid with kwargs={solve_kwargs}")
        driver_run = solve_multigrid(inp, **solve_kwargs)
    driver_state = driver_run.state

    rows = _compare_states(neopax_state, driver_state)
    comparable = {
        name: row
        for name, row in rows.items()
        if "rel_l2" in row and np.isfinite(row["rel_l2"])
    }
    worst = sorted(comparable.items(), key=lambda item: item[1]["rel_l2"], reverse=True)
    print("[compare] worst NEOPAX-realtime-state vs vmec-driver-state differences")
    for name, row in worst[: int(args.top)]:
        print(
            f"  - {name}: rel_l2={row['rel_l2']:.6e} "
            f"max_abs={row['max_abs']:.6e} shape={row['shape']} "
            f"neopax=[{row['left_min']:.6e}, {row['left_max']:.6e}] "
            f"driver=[{row['right_min']:.6e}, {row['right_max']:.6e}]"
        )

    payload = {
        "input_path": str(input_path),
        "realtime_config": str(realtime_config),
        "neopax_lane": neopax_lane,
        "driver_kwargs": run_kwargs,
        "state_variables": rows,
    }
    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"[compare] wrote {output_path}")


if __name__ == "__main__":
    main()
