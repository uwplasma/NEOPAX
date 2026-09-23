"""Compare saved VMEC WOUT variables against realtime vmec_jax WOUT fields.

This diagnostic does not run transport and does not call NTX.  It answers a
more primitive question: do the arrays in a saved ``wout_*.nc`` file match the
``WoutData`` object reconstructed from the realtime VMEC input path?
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.benchmarks.compare_realtime_vmec_initial_ambipolarity import (  # noqa: E402
    DEFAULT_REALTIME_CONFIG,
    _build_realtime_vmec_wout,
)


DEFAULT_FROZEN_WOUT = ROOT / "examples" / "inputs" / "wout_QI_nfp2_newNT_opt_hires_true.nc"
DEFAULT_VARIABLES = (
    "bsupumnc",
    "bsupumns",
    "bsupvmnc",
    "bsupvmns",
    "bsubumnc",
    "bsubumns",
    "bsubvmnc",
    "bsubvmns",
    "gmnc",
    "gmns",
    "bmnc",
    "bmns",
    "iotas",
    "iotaf",
    "rmnc",
    "zmns",
    "lmnc",
)


def _as_numpy(value: Any) -> np.ndarray:
    arr = np.asarray(jax.device_get(value))
    if np.ma.isMaskedArray(arr):
        arr = np.asarray(np.ma.filled(arr, np.nan))
    return np.asarray(arr)


def _finite_stats(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left_f = np.asarray(left, dtype=float)
    right_f = np.asarray(right, dtype=float)
    diff = right_f - left_f
    finite_left = left_f[np.isfinite(left_f)]
    finite_right = right_f[np.isfinite(right_f)]
    finite_diff = diff[np.isfinite(diff)]
    denom = max(float(np.linalg.norm(np.nan_to_num(left_f, nan=0.0))), 1.0e-30)
    return {
        "shape": list(left_f.shape),
        "saved_min": float(np.min(finite_left)) if finite_left.size else float("nan"),
        "saved_max": float(np.max(finite_left)) if finite_left.size else float("nan"),
        "realtime_min": float(np.min(finite_right)) if finite_right.size else float("nan"),
        "realtime_max": float(np.max(finite_right)) if finite_right.size else float("nan"),
        "max_abs": float(np.max(np.abs(finite_diff))) if finite_diff.size else float("nan"),
        "rel_l2": float(np.linalg.norm(np.nan_to_num(diff, nan=0.0)) / denom),
    }


def _shape_mismatch(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    return {
        "shape_mismatch": True,
        "saved_shape": list(left.shape),
        "realtime_shape": list(right.shape),
    }


def _load_netcdf_variables(path: Path) -> dict[str, np.ndarray]:
    try:
        from netCDF4 import Dataset
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError("This diagnostic requires netCDF4 to read saved WOUT files.") from exc

    with Dataset(path, mode="r") as ds:
        return {
            name: _as_numpy(var[:])
            for name, var in ds.variables.items()
        }


def _compare_wout_variables(
    *,
    saved_wout_path: Path,
    realtime_wout,
    variable_names: tuple[str, ...],
    include_all_common: bool,
) -> dict[str, Any]:
    saved_variables = _load_netcdf_variables(saved_wout_path)
    realtime_field_names = {field.name for field in dataclasses.fields(realtime_wout)}
    names = list(variable_names)
    if include_all_common:
        names = sorted(set(names) | (set(saved_variables) & realtime_field_names))

    rows: dict[str, Any] = {}
    for name in names:
        if name not in saved_variables:
            rows[name] = {"missing": "saved_wout"}
            continue
        if not hasattr(realtime_wout, name):
            rows[name] = {"missing": "realtime_wout"}
            continue
        saved = _as_numpy(saved_variables[name])
        realtime = _as_numpy(getattr(realtime_wout, name))
        if saved.shape != realtime.shape:
            rows[name] = _shape_mismatch(saved, realtime)
            continue
        rows[name] = _finite_stats(saved, realtime)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--saved-wout",
        default=str(DEFAULT_FROZEN_WOUT),
        help="Saved WOUT NetCDF file to compare against the realtime WOUT object.",
    )
    parser.add_argument(
        "--realtime-config",
        default=str(DEFAULT_REALTIME_CONFIG),
        help="Realtime geometry TOML used to build the vmec_jax WOUT object.",
    )
    parser.add_argument(
        "--variables",
        default=",".join(DEFAULT_VARIABLES),
        help="Comma-separated WOUT variable names to compare.",
    )
    parser.add_argument(
        "--all-common",
        action="store_true",
        help="Also compare every variable present in both the saved file and realtime WoutData.",
    )
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()

    saved_wout_path = Path(args.saved_wout).expanduser().resolve()
    realtime_config_path = Path(args.realtime_config).expanduser().resolve()
    variable_names = tuple(
        part.strip()
        for part in str(args.variables).split(",")
        if part.strip()
    )

    print(f"[compare] saved WOUT:      {saved_wout_path}")
    print(f"[compare] realtime config: {realtime_config_path}")
    print("[compare] building realtime vmec_jax WOUT object")
    realtime_wout = _build_realtime_vmec_wout(realtime_config_path)
    rows = _compare_wout_variables(
        saved_wout_path=saved_wout_path,
        realtime_wout=realtime_wout,
        variable_names=variable_names,
        include_all_common=bool(args.all_common),
    )

    comparable = {
        name: row
        for name, row in rows.items()
        if "rel_l2" in row
    }
    missing_or_mismatched = {
        name: row
        for name, row in rows.items()
        if "rel_l2" not in row
    }
    worst = sorted(comparable.items(), key=lambda item: item[1]["rel_l2"], reverse=True)

    print("[compare] worst saved-vs-realtime WOUT variable differences")
    for name, row in worst[: int(args.top)]:
        print(
            f"  - {name}: rel_l2={row['rel_l2']:.6e} "
            f"max_abs={row['max_abs']:.6e} shape={row['shape']} "
            f"saved=[{row['saved_min']:.6e}, {row['saved_max']:.6e}] "
            f"realtime=[{row['realtime_min']:.6e}, {row['realtime_max']:.6e}]"
        )
    if missing_or_mismatched:
        print("[compare] missing or shape-mismatched variables")
        for name, row in missing_or_mismatched.items():
            print(f"  - {name}: {row}")

    payload = {
        "saved_wout": str(saved_wout_path),
        "realtime_config": str(realtime_config_path),
        "variables": rows,
    }
    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"[compare] wrote {out}")


if __name__ == "__main__":
    main()
