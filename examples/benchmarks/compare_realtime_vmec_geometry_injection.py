"""Compare frozen VMEC/Boozer geometry injection against realtime VMEC/JAX.

This is a diagnostic for cases where transport diverges before the time
integrator matters, e.g. already during ambipolar initialization.  It builds:

1. the usual frozen VmecBoozer geometry from wout/boozmn files;
2. the realtime VmecBoozer-shaped geometry from vmec_jax -> booz_xform_jax;
3. the frozen and realtime NTX exact-Lij support channels.

No transport solve or AD rule is exercised here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from NEOPAX._geometry_autodiff import (
    _build_neopax_geometry_from_state,
    _solve_state_for_single_param,
    build_geometry_autodiff_context,
    build_ntx_exact_lij_support_from_vmec_state,
)
from NEOPAX._orchestrator import _build_geometry, load_config
from NEOPAX._transport_flux_models import build_ntx_exact_lij_runtime_support


DEFAULT_FROZEN_CONFIG = (
    "examples/benchmarks/"
    "Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_benchmark.toml"
)
DEFAULT_REALTIME_CONFIG = (
    "examples/benchmarks/"
    "Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml"
)


GEOMETRY_FIELDS = (
    "a_b",
    "Psia_value",
    "rho_grid",
    "rho_grid_half",
    "r_grid",
    "r_grid_half",
    "Vprime",
    "Vprime_half",
    "epsilon_t",
    "B0",
    "B_10",
    "iota",
    "R0",
    "curvature",
    "G_PS",
    "sqrtg00_value",
    "Bsqav",
    "I_value",
    "G_value",
)

CHANNEL_FIELDS = (
    "rho",
    "a_b",
    "psia",
    "b00",
    "r00",
    "boozer_i",
    "boozer_g",
    "iota",
    "drds",
    "dr_tildedr",
    "dr_tildeds",
    "fac_reference_to_sfincs_11",
    "fac_reference_to_sfincs_31",
    "fac_reference_to_sfincs_33",
    "fac_sfincs_to_dkes_11",
    "fac_sfincs_to_dkes_31",
    "fac_sfincs_to_dkes_33",
    "fac_dkes_to_d11star",
    "fac_dkes_to_d31star",
    "fac_dkes_to_d33star",
)


def _as_array(value: Any) -> np.ndarray:
    return np.asarray(jnp.asarray(value, dtype=jnp.float64))


def _stats(lhs: Any, rhs: Any) -> dict[str, Any]:
    left = _as_array(lhs).reshape(-1)
    right = _as_array(rhs).reshape(-1)
    if left.shape != right.shape:
        return {
            "shape_frozen": list(left.shape),
            "shape_realtime": list(right.shape),
            "max_abs": None,
            "rel_l2": None,
            "status": "shape-mismatch",
        }
    diff = left - right
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    denom = max(float(np.linalg.norm(left)), 1.0e-30)
    rel_l2 = float(np.linalg.norm(diff) / denom)
    finite = np.isfinite(left) & np.isfinite(right)
    if np.any(finite):
        finite_diff = diff[finite]
        finite_left = left[finite]
        finite_max_abs = float(np.max(np.abs(finite_diff)))
        finite_denom = max(float(np.linalg.norm(finite_left)), 1.0e-30)
        finite_rel_l2 = float(np.linalg.norm(finite_diff) / finite_denom)
    else:
        finite_max_abs = None
        finite_rel_l2 = None
    return {
        "shape": list(left.shape),
        "max_abs": max_abs,
        "rel_l2": rel_l2,
        "finite_count": int(np.sum(finite)),
        "finite_rel_l2": finite_rel_l2,
        "finite_max_abs": finite_max_abs,
        "frozen_first": float(left[0]) if left.size else None,
        "realtime_first": float(right[0]) if right.size else None,
        "frozen_last": float(left[-1]) if left.size else None,
        "realtime_last": float(right[-1]) if right.size else None,
        "status": "ok",
    }


def _print_section(title: str, rows: dict[str, dict[str, Any]], *, top: int) -> None:
    print(f"\n[{title}]")
    sortable = [
        (name, row)
        for name, row in rows.items()
        if row.get("status") == "ok" and row.get("rel_l2") is not None
    ]
    sortable.sort(key=lambda item: (item[1]["rel_l2"], item[1]["max_abs"]), reverse=True)
    for name, row in sortable[:top]:
        finite_rel = row.get("finite_rel_l2")
        finite_rel_text = "nan" if finite_rel is None else f"{finite_rel:.6e}"
        print(
            f"{name:32s} "
            f"rel_l2={row['rel_l2']:.6e} "
            f"finite_rel_l2={finite_rel_text} "
            f"max_abs={row['max_abs']:.6e} "
            f"first=({row['frozen_first']:.6e}, {row['realtime_first']:.6e}) "
            f"last=({row['frozen_last']:.6e}, {row['realtime_last']:.6e})"
        )
    for name, row in rows.items():
        if row.get("status") != "ok":
            print(f"{name:32s} status={row['status']} {row}")


def _resolve_config(path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path = Path(path).expanduser().resolve()
    config = load_config(config_path)
    config["_config_dir"] = str(config_path.parent)
    return config_path, config


def _build_realtime_state_and_geometry(config: dict[str, Any]):
    geom_cfg = dict(config.get("geometry", {}))
    context = build_geometry_autodiff_context(
        geom_cfg.get("vmec_input_file"),
        param_family=str(geom_cfg.get("vmec_param_family", "RBC")),
        param_m=int(geom_cfg.get("vmec_m_index", 0)),
        param_n=int(geom_cfg.get("vmec_n_index", 0)),
        mboz=geom_cfg.get("mboz", geom_cfg.get("vmec_mboz")),
        nboz=geom_cfg.get("nboz", geom_cfg.get("vmec_nboz")),
    )
    lane = str(geom_cfg.get("vmec_lane", "forward")).strip().lower()
    param_delta = jnp.asarray(float(geom_cfg.get("vmec_param_delta", 0.0)), dtype=jnp.float64)
    state = _solve_state_for_single_param(
        context,
        param_delta,
        lane=lane,
        max_iter=geom_cfg.get("vmec_max_iter"),
        step_size=geom_cfg.get("vmec_step_size"),
        jacobian_penalty=float(geom_cfg.get("vmec_jacobian_penalty", 1.0e3)),
    )
    geometry = _build_neopax_geometry_from_state(
        context,
        state,
        n_r=int(geom_cfg.get("n_radial", 51)),
    )
    return context, state, geometry


def _build_supports(frozen_config: dict[str, Any], realtime_config: dict[str, Any], frozen_geometry, realtime_context, realtime_state, realtime_geometry):
    neo_cfg = dict(frozen_config.get("neoclassical", {}))
    n_theta = int(neo_cfg.get("ntx_exact_n_theta", 25))
    n_zeta = int(neo_cfg.get("ntx_exact_n_zeta", 25))
    n_xi = int(neo_cfg.get("ntx_exact_n_xi", 64))
    surface_backend = str(neo_cfg.get("ntx_exact_surface_backend", "auto"))

    frozen_rho_center = jnp.asarray(frozen_geometry.r_grid, dtype=jnp.float64) / jnp.asarray(frozen_geometry.a_b)
    frozen_rho_face = jnp.asarray(frozen_geometry.r_grid_half, dtype=jnp.float64) / jnp.asarray(frozen_geometry.a_b)
    frozen_support = build_ntx_exact_lij_runtime_support(
        frozen_config["geometry"]["vmec_file"],
        frozen_config["geometry"]["boozer_file"],
        frozen_rho_center,
        frozen_rho_face,
        surface_backend=surface_backend,
        n_theta=n_theta,
        n_zeta=n_zeta,
        n_xi=n_xi,
    )

    realtime_neo_cfg = dict(realtime_config.get("neoclassical", {}))
    realtime_support = build_ntx_exact_lij_support_from_vmec_state(
        realtime_context,
        realtime_state,
        realtime_geometry,
        n_theta=int(realtime_neo_cfg.get("ntx_exact_n_theta", n_theta)),
        n_zeta=int(realtime_neo_cfg.get("ntx_exact_n_zeta", n_zeta)),
        n_xi=int(realtime_neo_cfg.get("ntx_exact_n_xi", n_xi)),
    )
    return frozen_support, realtime_support


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-config", default=DEFAULT_FROZEN_CONFIG)
    parser.add_argument("--realtime-config", default=DEFAULT_REALTIME_CONFIG)
    parser.add_argument("--top", type=int, default=12, help="Rows to print per section.")
    parser.add_argument("--json-output", type=str, default=None, help="Optional compact JSON report path.")
    args = parser.parse_args()

    frozen_path, frozen_config = _resolve_config(args.frozen_config)
    realtime_path, realtime_config = _resolve_config(args.realtime_config)
    print(f"[compare] frozen config:  {frozen_path}")
    print(f"[compare] realtime config:{realtime_path}")

    frozen_geometry = _build_geometry(frozen_config)
    realtime_context, realtime_state, realtime_geometry = _build_realtime_state_and_geometry(realtime_config)
    frozen_support, realtime_support = _build_supports(
        frozen_config,
        realtime_config,
        frozen_geometry,
        realtime_context,
        realtime_state,
        realtime_geometry,
    )

    geometry_rows = {
        field: _stats(getattr(frozen_geometry, field), getattr(realtime_geometry, field))
        for field in GEOMETRY_FIELDS
    }
    center_rows = {
        field: _stats(getattr(frozen_support.center_channels, field), getattr(realtime_support.center_channels, field))
        for field in CHANNEL_FIELDS
    }
    face_rows = {
        field: _stats(getattr(frozen_support.face_channels, field), getattr(realtime_support.face_channels, field))
        for field in CHANNEL_FIELDS
    }

    _print_section("geometry frozen vs realtime", geometry_rows, top=int(args.top))
    _print_section("NTX center channels frozen vs realtime", center_rows, top=int(args.top))
    _print_section("NTX face channels frozen vs realtime", face_rows, top=int(args.top))

    if args.json_output:
        out_path = Path(args.json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "frozen_config": str(frozen_path),
            "realtime_config": str(realtime_path),
            "geometry": geometry_rows,
            "ntx_center_channels": center_rows,
            "ntx_face_channels": face_rows,
        }
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\n[compare] wrote {out_path}")


if __name__ == "__main__":
    main()
