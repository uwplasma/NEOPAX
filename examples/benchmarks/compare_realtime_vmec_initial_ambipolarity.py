"""Compare frozen-file and realtime-VMEC initial ambipolar Er setup.

This diagnostic intentionally stops after ``build_runtime_context``.  That
builds geometry, NTX support, the initial state, and any configured ambipolar
Er initialization, but it does not run the transport time integrator.
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


def _parse_index_list(value: str | None) -> list[int]:
    if value is None or str(value).strip() == "":
        return []
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def _ambipolar_residual_curves(path: Path, *, indices: list[int], er_grid: np.ndarray) -> dict[str, Any]:
    config, runtime, state = _load_for_initial_ambipolarity(path, initialize_er=False)
    charge_qp = jnp.asarray(runtime.species.charge_qp)
    local_particle_flux = runtime.models.flux.build_local_particle_flux_evaluator(state)
    er_grid_jax = jnp.asarray(er_grid, dtype=jnp.float64)
    state_er = jnp.asarray(state.Er)

    def _evaluate_for_index(index: int):
        index_value = jnp.asarray(index, dtype=jnp.int32)

        def _evaluate_er(er_value):
            if local_particle_flux is not None:
                gamma = local_particle_flux(index_value, er_value)
            else:
                er_vec = state_er.at[index_value].set(jnp.asarray(er_value, dtype=state_er.dtype))
                fluxes = runtime.models.flux(dataclasses.replace(state, Er=er_vec))
                gamma = fluxes.get("Gamma_total") or fluxes.get("Gamma")
                if gamma is None:
                    raise ValueError("Flux model did not return 'Gamma' or 'Gamma_total'.")
                gamma = gamma[:, index_value]
            residual = jnp.sum(charge_qp * gamma)
            entropy = jnp.sum(jnp.abs(gamma))
            return residual, entropy

        residual, entropy = jax.vmap(_evaluate_er)(er_grid_jax)
        residual_np = np.asarray(jax.device_get(residual), dtype=float)
        entropy_np = np.asarray(jax.device_get(entropy), dtype=float)
        finite_residual = residual_np[np.isfinite(residual_np)]
        return {
            "residual": residual_np.tolist(),
            "entropy": entropy_np.tolist(),
            "residual_min": float(np.min(finite_residual)) if finite_residual.size else float("nan"),
            "residual_max": float(np.max(finite_residual)) if finite_residual.size else float("nan"),
            "residual_linf": float(np.max(np.abs(finite_residual))) if finite_residual.size else float("nan"),
        }

    return {str(index): _evaluate_for_index(index) for index in indices}


def _local_flux_probes(path: Path, *, probes_by_index: dict[int, dict[str, float]]) -> dict[str, Any]:
    _config, runtime, state = _load_for_initial_ambipolarity(path, initialize_er=False)
    charge_qp = jnp.asarray(runtime.species.charge_qp)
    local_particle_flux = runtime.models.flux.build_local_particle_flux_evaluator(state)
    if local_particle_flux is None:
        raise ValueError("Flux model did not provide a local particle-flux evaluator.")

    rows: dict[str, Any] = {}
    for index, probes in probes_by_index.items():
        index_value = jnp.asarray(index, dtype=jnp.int32)
        probe_rows: dict[str, Any] = {}
        for name, er_value in probes.items():
            gamma = local_particle_flux(index_value, jnp.asarray(er_value, dtype=state.Er.dtype))
            gamma_np = np.asarray(jax.device_get(gamma), dtype=float)
            residual = float(jax.device_get(jnp.sum(charge_qp * gamma)))
            entropy = float(jax.device_get(jnp.sum(jnp.abs(gamma))))
            probe_rows[str(name)] = {
                "Er": float(er_value),
                "Gamma": gamma_np.tolist(),
                "residual": residual,
                "entropy": entropy,
            }
        rows[str(index)] = probe_rows
    return rows


def _flux_probe_rows(frozen: dict[str, Any], realtime: dict[str, Any]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for index, frozen_index_rows in frozen.items():
        realtime_index_rows = realtime[index]
        probe_rows = {}
        for probe_name, frozen_probe in frozen_index_rows.items():
            realtime_probe = realtime_index_rows[probe_name]
            frozen_gamma = np.asarray(frozen_probe["Gamma"], dtype=float)
            realtime_gamma = np.asarray(realtime_probe["Gamma"], dtype=float)
            gamma_delta = realtime_gamma - frozen_gamma
            probe_rows[probe_name] = {
                "Er": frozen_probe["Er"],
                "frozen": frozen_probe,
                "realtime": realtime_probe,
                "gamma_delta": gamma_delta.tolist(),
                "gamma_delta_linf": float(np.max(np.abs(gamma_delta))) if gamma_delta.size else float("nan"),
                "residual_delta": float(realtime_probe["residual"] - frozen_probe["residual"]),
                "entropy_delta": float(realtime_probe["entropy"] - frozen_probe["entropy"]),
            }
        rows[index] = probe_rows
    return rows


def _path_to_name(path_entry) -> str:
    parts = []
    for item in path_entry:
        key = getattr(item, "key", None)
        idx = getattr(item, "idx", None)
        name = getattr(item, "name", None)
        if key is not None:
            parts.append(str(key))
        elif idx is not None:
            parts.append(str(idx))
        elif name is not None:
            parts.append(str(name))
        else:
            parts.append(str(item))
    return ".".join(parts) if parts else "<root>"


def _selected_array(arr, indices: list[int]):
    arr_np = np.asarray(jax.device_get(arr), dtype=float)
    if arr_np.ndim > 0 and arr_np.shape[0] > max(indices):
        return arr_np[np.asarray(indices, dtype=int)]
    return arr_np


def _tree_selected_stats(left_tree, right_tree, *, indices: list[int], max_rows: int = 40) -> dict[str, Any]:
    left_flat, left_def = jax.tree_util.tree_flatten_with_path(left_tree)
    right_flat, right_def = jax.tree_util.tree_flatten_with_path(right_tree)
    if left_def != right_def:
        left_leaves, left_plain_def = jax.tree_util.tree_flatten(left_tree)
        right_leaves, right_plain_def = jax.tree_util.tree_flatten(right_tree)
        return {
            "tree_structure_equal": False,
            "left_treedef": str(left_plain_def),
            "right_treedef": str(right_plain_def),
            "left_leaf_count": len(left_leaves),
            "right_leaf_count": len(right_leaves),
            "worst": [],
        }

    rows = []
    for (path_entry, left_leaf), (_right_path, right_leaf) in zip(left_flat, right_flat, strict=True):
        if left_leaf is None or right_leaf is None:
            continue
        try:
            left_np = _selected_array(left_leaf, indices)
            right_np = _selected_array(right_leaf, indices)
        except Exception:
            continue
        if left_np.shape != right_np.shape:
            rows.append(
                {
                    "name": _path_to_name(path_entry),
                    "shape": [list(left_np.shape), list(right_np.shape)],
                    "rel_l2": float("inf"),
                    "max_abs": float("inf"),
                }
            )
            continue
        diff = right_np - left_np
        finite_diff = diff[np.isfinite(diff)]
        denom = max(float(np.linalg.norm(np.nan_to_num(left_np, nan=0.0))), 1.0e-30)
        rows.append(
            {
                "name": _path_to_name(path_entry),
                "shape": list(left_np.shape),
                "rel_l2": float(np.linalg.norm(np.nan_to_num(diff, nan=0.0)) / denom),
                "max_abs": float(np.max(np.abs(finite_diff))) if finite_diff.size else float("nan"),
                "left_min": float(np.nanmin(left_np)) if left_np.size else float("nan"),
                "left_max": float(np.nanmax(left_np)) if left_np.size else float("nan"),
                "right_min": float(np.nanmin(right_np)) if right_np.size else float("nan"),
                "right_max": float(np.nanmax(right_np)) if right_np.size else float("nan"),
            }
        )
    rows.sort(key=lambda row: (not np.isfinite(row["rel_l2"]), row["rel_l2"]), reverse=True)
    return {
        "tree_structure_equal": True,
        "worst": rows[:max_rows],
    }


def _ntx_support_rows(frozen_runtime, realtime_runtime, *, indices: list[int]) -> dict[str, Any]:
    def _support(runtime):
        model = runtime.models.flux
        model = getattr(model, "neoclassical_model", model)
        support_fn = getattr(model, "_static_support", None)
        if support_fn is None:
            raise ValueError("Neoclassical model does not expose _static_support().")
        return support_fn()

    frozen_support = _support(frozen_runtime)
    realtime_support = _support(realtime_runtime)
    frozen_center_prepared = frozen_support.center_prepared
    realtime_center_prepared = realtime_support.center_prepared
    return {
        "center_channels": _tree_selected_stats(
            frozen_support.center_channels,
            realtime_support.center_channels,
            indices=indices,
        ),
        "center_prepared": _tree_selected_stats(
            frozen_center_prepared,
            realtime_center_prepared,
            indices=indices,
        ),
        "center_prepared_geometry": _tree_selected_stats(
            frozen_center_prepared.geometry,
            realtime_center_prepared.geometry,
            indices=indices,
        ),
        "center_prepared_d_theta": _tree_selected_stats(
            frozen_center_prepared.d_theta,
            realtime_center_prepared.d_theta,
            indices=indices,
        ),
        "center_prepared_d_zeta": _tree_selected_stats(
            frozen_center_prepared.d_zeta,
            realtime_center_prepared.d_zeta,
            indices=indices,
        ),
    }


def _residual_curve_rows(
    frozen: dict[str, Any],
    realtime: dict[str, Any],
    *,
    er_grid: np.ndarray,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for key, frozen_row in frozen.items():
        realtime_row = realtime[key]
        frozen_residual = np.asarray(frozen_row["residual"], dtype=float)
        realtime_residual = np.asarray(realtime_row["residual"], dtype=float)
        delta = realtime_residual - frozen_residual
        finite_delta = delta[np.isfinite(delta)]
        denom = max(float(np.linalg.norm(np.nan_to_num(frozen_residual, nan=0.0))), 1.0e-30)
        rows[key] = {
            "er_grid": er_grid.tolist(),
            "frozen": frozen_row,
            "realtime": realtime_row,
            "residual_delta_linf": float(np.max(np.abs(finite_delta))) if finite_delta.size else float("nan"),
            "residual_delta_rel_l2": float(np.linalg.norm(np.nan_to_num(delta, nan=0.0)) / denom),
        }
    return rows


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
    parser.add_argument(
        "--residual-curve-indices",
        default=None,
        help="Optional comma-separated radial indices for ambipolar residual curve diagnostics.",
    )
    parser.add_argument("--residual-curve-er-min", type=float, default=-60.0)
    parser.add_argument("--residual-curve-er-max", type=float, default=30.0)
    parser.add_argument("--residual-curve-count", type=int, default=121)
    parser.add_argument(
        "--skip-ntx-support-diagnostics",
        action="store_true",
        help="Do not compare selected NTX center channel/prepared-system leaves.",
    )
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

    residual_curve_rows = None
    flux_probe_rows = None
    ntx_support_rows = None
    residual_indices = _parse_index_list(args.residual_curve_indices)
    if residual_indices:
        er_grid = np.linspace(
            float(args.residual_curve_er_min),
            float(args.residual_curve_er_max),
            int(args.residual_curve_count),
            dtype=float,
        )
        print(
            "[compare] ambipolar residual curves frozen vs realtime: "
            f"indices={residual_indices} er=[{er_grid[0]:.6e}, {er_grid[-1]:.6e}] "
            f"n={er_grid.size}"
        )
        frozen_curves = _ambipolar_residual_curves(frozen_path, indices=residual_indices, er_grid=er_grid)
        realtime_curves = _ambipolar_residual_curves(realtime_path, indices=residual_indices, er_grid=er_grid)
        residual_curve_rows = _residual_curve_rows(frozen_curves, realtime_curves, er_grid=er_grid)
        for key, row in residual_curve_rows.items():
            print(
                f"  - i={key}: residual_delta_rel_l2={row['residual_delta_rel_l2']:.6e} "
                f"residual_delta_linf={row['residual_delta_linf']:.6e} "
                f"frozen_range=[{row['frozen']['residual_min']:.6e}, {row['frozen']['residual_max']:.6e}] "
                f"realtime_range=[{row['realtime']['residual_min']:.6e}, {row['realtime']['residual_max']:.6e}]"
            )
        probes_by_index = {
            int(index): {
                "Er0": 0.0,
                "frozen_best": float(frozen_roots["best_roots"][int(index)]),
                "realtime_best": float(realtime_roots["best_roots"][int(index)]),
            }
            for index in residual_indices
        }
        frozen_flux_probes = _local_flux_probes(frozen_path, probes_by_index=probes_by_index)
        realtime_flux_probes = _local_flux_probes(realtime_path, probes_by_index=probes_by_index)
        flux_probe_rows = _flux_probe_rows(frozen_flux_probes, realtime_flux_probes)
        print("[compare] local particle-flux probes frozen vs realtime")
        for key, probe_set in flux_probe_rows.items():
            for probe_name, row in probe_set.items():
                print(
                    f"  - i={key} {probe_name}: Er={row['Er']:.6e} "
                    f"residual_delta={row['residual_delta']:.6e} "
                    f"gamma_delta_linf={row['gamma_delta_linf']:.6e} "
                    f"frozen_residual={row['frozen']['residual']:.6e} "
                    f"realtime_residual={row['realtime']['residual']:.6e}"
                )
        if not args.skip_ntx_support_diagnostics:
            ntx_support_rows = _ntx_support_rows(
                frozen_runtime,
                realtime_runtime,
                indices=residual_indices,
            )
            print("[compare] selected NTX support diffs frozen vs realtime")
            for group_name, group in ntx_support_rows.items():
                print(f"  - {group_name}: tree_structure_equal={group['tree_structure_equal']}")
                if not group["tree_structure_equal"]:
                    print(
                        "      "
                        f"leaf_count={group.get('left_leaf_count')}->{group.get('right_leaf_count')}"
                    )
                for row in group["worst"][:12]:
                    print(
                        f"      {row['name']}: rel_l2={row['rel_l2']:.6e} "
                        f"max_abs={row['max_abs']:.6e} shape={row['shape']}"
                    )

    payload = {
        "frozen_config": str(frozen_path),
        "realtime_config": str(realtime_path),
        "initial_state": state_rows,
        "geometry": geometry_rows,
        "ambipolar_roots": root_rows,
    }
    if residual_curve_rows is not None:
        payload["ambipolar_residual_curves"] = residual_curve_rows
    if flux_probe_rows is not None:
        payload["local_particle_flux_probes"] = flux_probe_rows
    if ntx_support_rows is not None:
        payload["ntx_support_diagnostics"] = ntx_support_rows
    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[compare] wrote {out}")


if __name__ == "__main__":
    main()
