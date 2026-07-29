#!/usr/bin/env python
"""Smoke-test the internal realtime-geometry transport reverse-AD optimization path."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from NEOPAX._geometry_autodiff import build_geometry_autodiff_context
from NEOPAX._orchestrator import build_runtime_context, load_config
from NEOPAX._reverse_ad_optimization import (
    build_transport_realtime_geometry_least_squares_runner,
    transport_least_squares_terms,
)
from NEOPAX._reverse_ad_parameters import (
    VmecBoundaryParameterSpec,
    parse_vmec_boundary_parameter_specs,
    reverse_ad_optimization_parameter_set,
)
from NEOPAX._reverse_ad_transport import (
    TRANSPORT_REVERSE_OBJECTIVE_LABELS,
    internal_realtime_geometry_transport_reverse_table_result_builder,
    realtime_geometry_transport_reverse_table_context,
)


PROFILE_PARAMETER_ORDER = ("n0", "T0", "density_shape_power", "temperature_shape_power")


def _profile_values(profile_cfg, dtype):
    return jnp.asarray([float(profile_cfg[name]) for name in PROFILE_PARAMETER_ORDER], dtype=dtype)


def _geometry_context_from_config(config, first_spec):
    geom_cfg = config.get("geometry", {})
    vmec_input_file = geom_cfg.get("vmec_input_file")
    if vmec_input_file is None:
        raise ValueError("geometry.vmec_input_file is required for realtime geometry reverse smoke.")
    family, m, n = first_spec
    return build_geometry_autodiff_context(
        vmec_input_file,
        param_family=family,
        param_m=m,
        param_n=n,
        mboz=geom_cfg.get("mboz", geom_cfg.get("vmec_mboz")),
        nboz=geom_cfg.get("nboz", geom_cfg.get("vmec_nboz")),
    )


def _baseline_geometry_delta_vector(geom_cfg, geometry_specs):
    deltas = np.zeros((len(geometry_specs),), dtype=np.float64)
    configured_delta = float(geom_cfg.get("vmec_param_delta", 0.0))
    if configured_delta != 0.0:
        configured_spec = (
            str(geom_cfg.get("vmec_param_family", "RBC")).strip().upper(),
            int(geom_cfg.get("vmec_param_m", 0)),
            int(geom_cfg.get("vmec_param_n", 0)),
        )
        for i, spec in enumerate(geometry_specs):
            normalized_spec = (str(spec[0]).strip().upper(), int(spec[1]), int(spec[2]))
            if normalized_spec == configured_spec:
                deltas[i] = configured_delta
                break
    return jnp.asarray(deltas, dtype=jnp.float64)


def _objective_names(value: str):
    if str(value).strip().lower() == "all":
        return TRANSPORT_REVERSE_OBJECTIVE_LABELS
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reverse-geometry-parameter", default="RBC:1:0")
    parser.add_argument("--objective", default="all")
    parser.add_argument("--accepted-step-limit", type=int, default=2)
    parser.add_argument("--reverse-segment-length", type=int, default=1)
    parser.add_argument("--initial-Er-root-ad", default="jax_selected_root")
    parser.add_argument("--optimization-api-profile-dofs", choices=("include", "exclude"), default="include")
    parser.add_argument("--reverse-stage-adjoint-solve-mode", default="bicgstab")
    parser.add_argument("--reverse-rhs-transpose-mode", default="explicit_ntx_interpolated")
    parser.add_argument("--reverse-step-bwd-mode", default="reduced_cotangent")
    parser.add_argument("--output", default="outputs/autodiff_transport_lagged_ntx/reverse_ad/internal_optimization_smoke_2step.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    solver_cfg = config.setdefault("transport_solver", {})
    if args.accepted_step_limit is not None:
        solver_cfg["stop_after_accepted_steps"] = int(args.accepted_step_limit)

    runtime, baseline_state = build_runtime_context(config)
    if baseline_state is None:
        raise RuntimeError("transport runtime did not return an initial state.")
    profile_cfg = dict(config.get("profiles", {}))
    neoclassical_cfg = dict(config.get("neoclassical", {}))
    geometry_specs = tuple(spec.as_tuple() for spec in parse_vmec_boundary_parameter_specs(args.reverse_geometry_parameter))
    if not geometry_specs:
        raise ValueError("--reverse-geometry-parameter must select at least one VMEC harmonic.")
    geometry_context = _geometry_context_from_config(config, geometry_specs[0])
    geometry_deltas = _baseline_geometry_delta_vector(config.get("geometry", {}), geometry_specs)
    profile_values = _profile_values(profile_cfg, jnp.asarray(baseline_state.pressure).dtype)
    baseline_values = jnp.concatenate([profile_values, geometry_deltas.astype(profile_values.dtype)])
    include_profiles = args.optimization_api_profile_dofs == "include"
    parameter_set = reverse_ad_optimization_parameter_set(
        include_profiles=include_profiles,
        vmec_boundary=tuple(VmecBoundaryParameterSpec(*spec) for spec in geometry_specs),
    )
    table_context = realtime_geometry_transport_reverse_table_context(
        config=config,
        baseline_values=baseline_values,
        baseline_runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        neoclassical_cfg=neoclassical_cfg,
    )
    table_builder = internal_realtime_geometry_transport_reverse_table_result_builder(
        table_context=table_context,
        geometry_context=geometry_context,
        baseline_geometry_deltas=geometry_deltas,
        n_r=int(config.get("geometry", {}).get("n_radial", 51)),
        n_theta=int(neoclassical_cfg.get("ntx_exact_n_theta", 5)),
        n_zeta=int(neoclassical_cfg.get("ntx_exact_n_zeta", 21)),
        n_xi=int(neoclassical_cfg.get("ntx_exact_n_xi", 33)),
        surface_backend=str(neoclassical_cfg.get("ntx_exact_surface_backend", "vmec")),
        accepted_step_limit=int(args.accepted_step_limit),
        reverse_segment_length=int(args.reverse_segment_length),
        initial_er_root_ad=str(args.initial_Er_root_ad),
        reverse_stage_adjoint_solve_mode=str(args.reverse_stage_adjoint_solve_mode),
        reverse_rhs_transpose_mode=str(args.reverse_rhs_transpose_mode),
        reverse_step_bwd_mode=str(args.reverse_step_bwd_mode),
        progress_label="[autodiff-gate] internal optimization smoke payload pullback",
    )
    objective_names = _objective_names(args.objective)
    runner = build_transport_realtime_geometry_least_squares_runner(
        config,
        objective_names=objective_names,
        parameter_set=parameter_set,
        table_context=table_context,
        table_result_builder=table_builder,
        options={
            "accepted_step_limit": int(args.accepted_step_limit),
            "reverse_segment_length": int(args.reverse_segment_length),
            "initial_er_root_ad": str(args.initial_Er_root_ad),
        },
    )
    print(
        "[autodiff-gate] progress: running internal realtime-geometry optimization smoke",
        flush=True,
    )
    evaluation = runner(transport_least_squares_terms(objective_names))
    residuals_np = np.asarray(jax.device_get(evaluation.residuals), dtype=float)
    jacobian_np = np.asarray(jax.device_get(evaluation.jacobian), dtype=float)
    result = evaluation.result
    print(
        "[autodiff-gate] mode=transport_realtime_geometry_reverse_internal_optimization_smoke "
        f"objective={args.objective} residual_count={len(result.residual_labels)} "
        f"parameter_count={len(result.parameter_labels)} elapsed_s={evaluation.elapsed_s:.3f}",
        flush=True,
    )
    for row_i, label in enumerate(result.residual_labels):
        print(f"  - {label}: residual={residuals_np[row_i]:.16e}")
        for parameter_name, value in zip(result.parameter_labels, jacobian_np[row_i].tolist()):
            print(f"      d{label}/d{parameter_name}: jac={value:.16e}")
    report = {
        "mode": "transport_realtime_geometry_reverse_internal_optimization_smoke",
        "config": str(Path(args.config)),
        "objective_order": list(objective_names),
        "residual_labels": list(result.residual_labels),
        "parameter_order": list(result.parameter_labels),
        "residuals": residuals_np.tolist(),
        "jacobian": jacobian_np.tolist(),
        "accepted_step_limit": int(args.accepted_step_limit),
        "reverse_segment_length": int(args.reverse_segment_length),
        "initial_er_root_ad": str(args.initial_Er_root_ad),
        "elapsed_s": float(evaluation.elapsed_s),
    }
    outpath = ROOT / args.output
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {outpath.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
