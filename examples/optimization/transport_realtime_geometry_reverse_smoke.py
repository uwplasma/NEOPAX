#!/usr/bin/env python
"""Smoke-test the internal realtime-geometry transport reverse-AD optimization path."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

import NEOPAX
from NEOPAX._orchestrator import build_runtime_context
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
    realtime_geometry_transport_reverse_grouped_inputs,
    realtime_geometry_transport_reverse_table_context,
    realtime_geometry_transport_reverse_support_segment_executor,
    run_internal_realtime_geometry_support_segment_probe,
)


PROFILE_PARAMETER_ORDER = ("n0", "T0", "density_shape_power", "temperature_shape_power")


def _prepare_smoke_config(
    config_path: Path,
    *,
    device: str | None,
    ntx_exact_derivative_mode: str | None = None,
    ntx_exact_derivative_field_pullback_mode: str | None = None,
    ntx_exact_derivative_pullback_boundary: str | None = None,
    ntx_exact_derivative_pullback_algebra: str | None = None,
    radau_jacobian_reuse_mode: str | None = None,
) -> dict:
    config = NEOPAX.prepare_config(config_path, device=device)
    config = copy.deepcopy(config)
    config.setdefault("general", {})["mode"] = "transport"
    transport_output = config.setdefault("transport_output", {})
    transport_output["transport_plot"] = False
    transport_output["transport_write_hdf5"] = False
    transport_output["transport_compare_ambipolarity_residual"] = False
    transport_output["transport_scan_ambipolarity_residual"] = False
    solver_cfg = config.setdefault("transport_solver", {})
    solver_cfg["debug_stage_markers"] = False
    solver_cfg["debug_disable_jit"] = False
    solver_cfg["debug_walltime_attempts"] = False
    if radau_jacobian_reuse_mode is not None:
        solver_cfg["radau_jacobian_reuse_mode"] = str(radau_jacobian_reuse_mode)
    if ntx_exact_derivative_mode is not None:
        config.setdefault("neoclassical", {})["ntx_exact_derivative_mode"] = str(ntx_exact_derivative_mode)
    if ntx_exact_derivative_field_pullback_mode is not None:
        config.setdefault("neoclassical", {})[
            "ntx_exact_derivative_field_pullback_mode"
        ] = str(ntx_exact_derivative_field_pullback_mode)
    if ntx_exact_derivative_pullback_boundary is not None:
        config.setdefault("neoclassical", {})[
            "ntx_exact_derivative_pullback_boundary"
        ] = str(ntx_exact_derivative_pullback_boundary)
    if ntx_exact_derivative_pullback_algebra is not None:
        config.setdefault("neoclassical", {})[
            "ntx_exact_derivative_pullback_algebra"
        ] = str(ntx_exact_derivative_pullback_algebra)
    return config


def _baseline_profile_cfg(config: dict) -> dict:
    profiles = copy.deepcopy(config.get("profiles", {}))
    profiles.setdefault("model", "standard_analytical")
    return profiles


def _profile_values(profile_cfg, dtype):
    return jnp.asarray([float(profile_cfg[name]) for name in PROFILE_PARAMETER_ORDER], dtype=dtype)


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
    parser.add_argument("--device", choices=("default", "cpu", "gpu"), default="default")
    parser.add_argument("--reverse-geometry-parameter", default="RBC:1:0")
    parser.add_argument("--reverse-geometry-families", default="RBC,ZBS")
    parser.add_argument("--reverse-geometry-include-zero-harmonics", action="store_true")
    parser.add_argument("--objective", default="all")
    parser.add_argument("--accepted-step-limit", type=int, default=2)
    parser.add_argument("--reverse-segment-length", type=int, default=1)
    parser.add_argument("--initial-Er-root-ad", default="jax_selected_root")
    parser.add_argument("--optimization-api-profile-dofs", choices=("include", "exclude"), default="include")
    parser.add_argument("--ntx-exact-derivative-mode", default="direct")
    parser.add_argument("--ntx-exact-derivative-field-pullback-mode", default="generic_jvp")
    parser.add_argument("--ntx-exact-derivative-pullback-boundary", default=None)
    parser.add_argument("--ntx-exact-derivative-pullback-algebra", default=None)
    parser.add_argument("--radau-jacobian-reuse-mode", default="legacy")
    parser.add_argument("--reverse-stage-adjoint-solve-mode", default="bicgstab")
    parser.add_argument("--reverse-rhs-transpose-mode", default="explicit_ntx_interpolated")
    parser.add_argument("--reverse-stage-cotangent-mode", default="full")
    parser.add_argument("--reverse-step-bwd-mode", default="reduced_cotangent")
    parser.add_argument("--reverse-stage-adjoint-memory-mode", default="default")
    parser.add_argument("--reverse-stage-adjoint-iter-maxiter", type=int, default=40)
    parser.add_argument("--reverse-stage-adjoint-iter-tol", type=float, default=1.0e-10)
    parser.add_argument("--realtime-geometry-component-pullbacks", action="store_true")
    parser.add_argument("--hide-solver-iterations", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = _prepare_smoke_config(
        Path(args.config),
        device=args.device,
        ntx_exact_derivative_mode=args.ntx_exact_derivative_mode,
        ntx_exact_derivative_field_pullback_mode=args.ntx_exact_derivative_field_pullback_mode,
        ntx_exact_derivative_pullback_boundary=args.ntx_exact_derivative_pullback_boundary,
        ntx_exact_derivative_pullback_algebra=args.ntx_exact_derivative_pullback_algebra,
        radau_jacobian_reuse_mode=args.radau_jacobian_reuse_mode,
    )

    t_phase = time.perf_counter()
    runtime, baseline_state = build_runtime_context(config)
    print(
        "[autodiff-gate] progress: realtime geometry runtime build ready "
        f"elapsed_s={time.perf_counter() - t_phase:.3f}",
        flush=True,
    )
    if baseline_state is None:
        raise RuntimeError("transport runtime did not return an initial state.")
    t_phase = time.perf_counter()
    profile_cfg = _baseline_profile_cfg(config)
    neoclassical_cfg = dict(config.get("neoclassical", {}))
    geometry_specs = tuple(spec.as_tuple() for spec in parse_vmec_boundary_parameter_specs(args.reverse_geometry_parameter))
    if not geometry_specs:
        raise ValueError("--reverse-geometry-parameter must select at least one VMEC harmonic.")
    profile_values = _profile_values(profile_cfg, jnp.asarray(baseline_state.pressure).dtype)
    geometry_deltas = _baseline_geometry_delta_vector(config.get("geometry", {}), geometry_specs)
    baseline_values = jnp.concatenate([profile_values, geometry_deltas.astype(profile_values.dtype)])
    include_profiles = args.optimization_api_profile_dofs == "include"
    parameter_set = reverse_ad_optimization_parameter_set(
        include_profiles=include_profiles,
        vmec_boundary=tuple(VmecBoundaryParameterSpec(*spec) for spec in geometry_specs),
    )
    setattr(args, "realtime_geometry_gradient_path", "reverse_payload")
    setattr(args, "skip_realtime_geometry_support_bar_diagnostics", True)
    setattr(args, "initial_er_root_ad", str(args.initial_Er_root_ad))

    def _internal_support_segment_probe(
        *,
        args,
        config,
        baseline_values,
        baseline_runtime,
        baseline_state,
        profile_cfg,
        neoclassical_cfg,
        return_report=False,
    ):
        return run_internal_realtime_geometry_support_segment_probe(
            args=args,
            context=realtime_geometry_transport_reverse_table_context(
                config=config,
                baseline_values=baseline_values,
                baseline_runtime=baseline_runtime,
                baseline_state=baseline_state,
                profile_cfg=profile_cfg,
                neoclassical_cfg=neoclassical_cfg,
            ),
            return_report=return_report,
            suppress_diagnostics=False,
        )

    support_segment_executor = realtime_geometry_transport_reverse_support_segment_executor(
        support_segment_probe=_internal_support_segment_probe,
        config=config,
        baseline_values=baseline_values,
        baseline_runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        neoclassical_cfg=neoclassical_cfg,
    )
    grouped_inputs = realtime_geometry_transport_reverse_grouped_inputs(
        args=args,
        config=config,
        baseline_values=baseline_values,
        baseline_runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        neoclassical_cfg=neoclassical_cfg,
        support_segment_executor=support_segment_executor,
    )
    print(
        "[autodiff-gate] progress: realtime geometry optimization inputs ready "
        f"elapsed_s={time.perf_counter() - t_phase:.3f}",
        flush=True,
    )
    table_context = grouped_inputs.table_context
    run_grouped_report = grouped_inputs.run_grouped_report
    objective_names = _objective_names(args.objective)
    runner = build_transport_realtime_geometry_least_squares_runner(
        config,
        objective_names=objective_names,
        parameter_set=parameter_set,
        table_context=table_context,
        run_grouped_report=run_grouped_report,
        objective_labels=TRANSPORT_REVERSE_OBJECTIVE_LABELS,
        options={
            "quiet": False,
            "accepted_step_limit": int(args.accepted_step_limit),
            "reverse_segment_length": int(args.reverse_segment_length),
            "initial_er_root_ad": str(args.initial_Er_root_ad),
            "reverse_stage_adjoint_solve_mode": str(args.reverse_stage_adjoint_solve_mode),
            "reverse_rhs_transpose_mode": str(args.reverse_rhs_transpose_mode),
            "reverse_stage_cotangent_mode": str(args.reverse_stage_cotangent_mode),
            "reverse_step_bwd_mode": str(args.reverse_step_bwd_mode),
            "reverse_stage_adjoint_memory_mode": str(args.reverse_stage_adjoint_memory_mode),
            "reverse_stage_adjoint_iter_maxiter": int(args.reverse_stage_adjoint_iter_maxiter),
            "reverse_stage_adjoint_iter_tol": float(args.reverse_stage_adjoint_iter_tol),
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
        "ntx_exact_derivative_mode": str(args.ntx_exact_derivative_mode),
        "ntx_exact_derivative_field_pullback_mode": str(args.ntx_exact_derivative_field_pullback_mode),
        "radau_jacobian_reuse_mode": str(args.radau_jacobian_reuse_mode),
        "reverse_stage_adjoint_solve_mode": str(args.reverse_stage_adjoint_solve_mode),
        "reverse_rhs_transpose_mode": str(args.reverse_rhs_transpose_mode),
        "reverse_stage_cotangent_mode": str(args.reverse_stage_cotangent_mode),
        "reverse_step_bwd_mode": str(args.reverse_step_bwd_mode),
        "reverse_stage_adjoint_memory_mode": str(args.reverse_stage_adjoint_memory_mode),
        "reverse_stage_adjoint_iter_maxiter": int(args.reverse_stage_adjoint_iter_maxiter),
        "reverse_stage_adjoint_iter_tol": float(args.reverse_stage_adjoint_iter_tol),
        "elapsed_s": float(evaluation.elapsed_s),
    }
    output_path = args.output
    if output_path is None:
        output_path = (
            "outputs/autodiff_transport_lagged_ntx/reverse_ad/"
            f"internal_optimization_smoke_{int(args.accepted_step_limit)}step.json"
        )
    outpath = ROOT / output_path
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {outpath.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
