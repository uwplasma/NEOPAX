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

from NEOPAX._ambipolarity import solve_ambipolarity_roots_radial  # noqa: E402
from NEOPAX._entropy_models import get_entropy_model  # noqa: E402
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
    finite_left = left_np[np.isfinite(left_np)]
    finite_right = right_np[np.isfinite(right_np)]
    finite_diff = diff[np.isfinite(diff)]
    denom = max(float(np.linalg.norm(np.nan_to_num(left_np, nan=0.0))), 1.0e-30)
    return {
        "shape": list(left_np.shape),
        "left_min": float(np.min(finite_left)) if finite_left.size else float("nan"),
        "left_max": float(np.max(finite_left)) if finite_left.size else float("nan"),
        "right_min": float(np.min(finite_right)) if finite_right.size else float("nan"),
        "right_max": float(np.max(finite_right)) if finite_right.size else float("nan"),
        "max_abs": float(np.max(np.abs(finite_diff))) if finite_diff.size else float("nan"),
        "rel_l2": float(np.linalg.norm(np.nan_to_num(diff, nan=0.0)) / denom),
    }


def _load_for_initial_ambipolarity(path: Path, *, initialize_er: bool = True):
    config = load_config(path)
    # Keep this diagnostic read-only with respect to benchmark plot outputs.
    amb_cfg = config.setdefault("ambipolarity", {})
    amb_cfg["er_ambipolar_plot"] = False
    amb_cfg["er_ambipolar_overlay_reference_er"] = False
    if not initialize_er:
        profiles_cfg = config.setdefault("profiles", {})
        profiles_cfg["er_initialization_mode"] = "analytical"
    runtime, state = build_runtime_context(config)
    return config, runtime, state


def _ambipolar_root_diagnostics(path: Path):
    config, runtime, state = _load_for_initial_ambipolarity(path, initialize_er=False)
    amb_cfg = dict(config.get("ambipolarity", {}))
    model_name = str(amb_cfg.get("er_ambipolar_method", "two_stage")).lower()
    entropy_model_name = config.get("neoclassical", {}).get(
        "entropy_model",
        runtime.solver_parameters.get("neoclassical_flux_model", "ntx_database"),
    )
    params = {
        "species": runtime.species,
        "energy_grid": runtime.energy_grid,
        "geometry": runtime.geometry,
        "database": runtime.database,
        "solver_parameters": runtime.solver_parameters,
    }
    roots, entropies, best_roots, n_roots = solve_ambipolarity_roots_radial(
        state=state,
        config=config,
        params=params,
        model_name=model_name,
        flux_model=runtime.models.flux,
        entropy_model=get_entropy_model(entropy_model_name),
        amb_cfg=amb_cfg,
    )
    return {
        "roots": np.asarray(roots, dtype=float),
        "entropies": np.asarray(entropies, dtype=float),
        "best_roots": np.asarray(best_roots, dtype=float),
        "n_roots": np.asarray(n_roots, dtype=int),
    }


def _root_branch_rows(frozen: dict[str, np.ndarray], realtime: dict[str, np.ndarray]) -> dict[str, Any]:
    best_delta = realtime["best_roots"] - frozen["best_roots"]
    n_roots_delta = realtime["n_roots"] - frozen["n_roots"]
    root_delta = realtime["roots"] - frozen["roots"]
    entropy_delta = realtime["entropies"] - frozen["entropies"]
    branch_frozen = np.nanargmin(
        np.where(np.isfinite(frozen["entropies"]), frozen["entropies"], np.inf),
        axis=1,
    )
    branch_realtime = np.nanargmin(
        np.where(np.isfinite(realtime["entropies"]), realtime["entropies"], np.inf),
        axis=1,
    )
    branch_changed = branch_frozen != branch_realtime
    finite_root_delta = root_delta[np.isfinite(root_delta)]
    finite_entropy_delta = entropy_delta[np.isfinite(entropy_delta)]
    worst = np.argsort(np.abs(best_delta))[::-1][:10]
    return {
        "best_roots": _stats(frozen["best_roots"], realtime["best_roots"]),
        "n_roots_changed_count": int(np.sum(n_roots_delta != 0)),
        "selected_branch_changed_count": int(np.sum(branch_changed)),
        "root_branch_max_abs": float(np.max(np.abs(finite_root_delta))) if finite_root_delta.size else float("nan"),
        "entropy_branch_max_abs": (
            float(np.max(np.abs(finite_entropy_delta))) if finite_entropy_delta.size else float("nan")
        ),
        "worst_best_root_deltas": [
            {
                "index": int(i),
                "frozen_best": float(frozen["best_roots"][i]),
                "realtime_best": float(realtime["best_roots"][i]),
                "delta": float(best_delta[i]),
                "frozen_n_roots": int(frozen["n_roots"][i]),
                "realtime_n_roots": int(realtime["n_roots"][i]),
                "frozen_selected_branch": int(branch_frozen[i]),
                "realtime_selected_branch": int(branch_realtime[i]),
                "frozen_roots": frozen["roots"][i].tolist(),
                "realtime_roots": realtime["roots"][i].tolist(),
                "frozen_entropies": frozen["entropies"][i].tolist(),
                "realtime_entropies": realtime["entropies"][i].tolist(),
            }
            for i in worst
        ],
    }


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
    print("[compare] solving frozen ambipolar root branches without transport")
    frozen_roots = _ambipolar_root_diagnostics(frozen_path)
    print("[compare] solving realtime ambipolar root branches without transport")
    realtime_roots = _ambipolar_root_diagnostics(realtime_path)

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
    root_rows = _root_branch_rows(frozen_roots, realtime_roots)
    print("[compare] ambipolar root branches frozen vs realtime")
    print(
        "  - best_roots: "
        f"rel_l2={root_rows['best_roots']['rel_l2']:.6e} "
        f"max_abs={root_rows['best_roots']['max_abs']:.6e}"
    )
    print(f"  - n_roots_changed_count={root_rows['n_roots_changed_count']}")
    print(f"  - selected_branch_changed_count={root_rows['selected_branch_changed_count']}")
    print(f"  - root_branch_max_abs={root_rows['root_branch_max_abs']:.6e}")
    print(f"  - entropy_branch_max_abs={root_rows['entropy_branch_max_abs']:.6e}")
    print("  - worst best-root deltas:")
    for row in root_rows["worst_best_root_deltas"]:
        print(
            f"      i={row['index']} delta={row['delta']:.6e} "
            f"frozen={row['frozen_best']:.6e} realtime={row['realtime_best']:.6e} "
            f"branches={row['frozen_selected_branch']}->{row['realtime_selected_branch']} "
            f"n_roots={row['frozen_n_roots']}->{row['realtime_n_roots']}"
        )

    payload = {
        "frozen_config": str(frozen_path),
        "realtime_config": str(realtime_path),
        "initial_state": state_rows,
        "geometry": geometry_rows,
        "ambipolar_roots": root_rows,
    }
    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[compare] wrote {out}")


if __name__ == "__main__":
    main()
