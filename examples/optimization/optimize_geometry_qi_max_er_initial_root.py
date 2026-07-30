#!/usr/bin/env python
"""Evaluate geometry objectives plus initial ambipolar-Er objectives for optimization."""

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
from NEOPAX._geometry_autodiff import (
    build_geometry_autodiff_context,
)
from NEOPAX._orchestrator import build_runtime_context
from NEOPAX._reverse_ad_optimization import (
    evaluate_geometry_initial_er_root_only_least_squares_fused,
    geometry,
    transport,
)
from NEOPAX._reverse_ad_parameters import (
    ProfileParameterSpec,
    VmecBoundaryParameterSpec,
    discover_vmec_boundary_parameter_specs,
    parse_profile_parameter_specs,
    parse_vmec_boundary_parameter_specs,
    reverse_ad_optimization_parameter_set,
    vmex_boundary_parameterization,
)
from NEOPAX._reverse_ad_transport import (
    TRANSPORT_REVERSE_PROFILE_PARAMETER_ORDER,
    initial_state_for_parameter_vector,
)


DEFAULT_CONFIG = (
    "examples/benchmarks/"
    "Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml"
)
DEFAULT_VMEC_INPUT = "examples/inputs/input.QI_nfp2_initial"
INITIAL_ROOT_OBJECTIVES = ("softmax_Er", "smooth_root_proxy", "Er2_volume_average", "Er_volume_average")


def _prepare_config(config_path: Path, *, device: str | None, vmec_input: str | None) -> dict:
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
    config.setdefault("neoclassical", {})["ntx_exact_derivative_mode"] = "direct"
    config.setdefault("neoclassical", {})["ntx_exact_derivative_field_pullback_mode"] = "generic_jvp"
    if vmec_input is not None:
        config.setdefault("geometry", {})["vmec_input_file"] = str(vmec_input)
    return config


def _profile_cfg(config: dict) -> dict:
    profiles = copy.deepcopy(config.get("profiles", {}))
    profiles.setdefault("model", "standard_analytical")
    return profiles


def _profile_values(profile_cfg: dict, dtype) -> jax.Array:
    return jnp.asarray(
        [float(profile_cfg[name]) for name in TRANSPORT_REVERSE_PROFILE_PARAMETER_ORDER],
        dtype=dtype,
    )


def _geometry_context(config: dict):
    geom_cfg = config.get("geometry", {})
    vmec_input = geom_cfg.get("vmec_input_file")
    if vmec_input is None:
        raise ValueError("geometry.vmec_input_file is required.")
    return build_geometry_autodiff_context(
        vmec_input,
        param_family="RBC",
        param_m=1,
        param_n=0,
        mboz=int(geom_cfg.get("mboz", geom_cfg.get("vmec_mboz", 18))),
        nboz=int(geom_cfg.get("nboz", geom_cfg.get("vmec_nboz", 18))),
    )


def _geometry_specs(args, geometry_context):
    mode = str(args.geometry_parameters).strip().lower()
    if mode == "all":
        return tuple(
            discover_vmec_boundary_parameter_specs(
                geometry_context,
                families=args.geometry_families,
                nonzero_only=not bool(args.geometry_include_zero_harmonics),
            )
        )
    if mode == "vmex_packed":
        return vmex_boundary_parameterization(
            geometry_context,
            max_mode=int(args.geometry_max_mode),
            families=args.geometry_families,
            scale_mode=args.geometry_scale_mode,
            ess_alpha=float(args.geometry_ess_alpha),
            nonzero_only=not bool(args.geometry_include_zero_harmonics),
        ).specs
    return parse_vmec_boundary_parameter_specs(args.geometry_parameters)


def _baseline_geometry_deltas(config: dict, specs) -> jax.Array:
    geom_cfg = config.get("geometry", {})
    deltas = np.zeros((len(specs),), dtype=np.float64)
    configured_delta = float(geom_cfg.get("vmec_param_delta", 0.0))
    if configured_delta != 0.0:
        configured_spec = (
            str(geom_cfg.get("vmec_param_family", "RBC")).strip().upper(),
            int(geom_cfg.get("vmec_param_m", 0)),
            int(geom_cfg.get("vmec_param_n", 0)),
        )
        for i, spec in enumerate(specs):
            if spec.as_tuple() == configured_spec:
                deltas[i] = configured_delta
                break
    return jnp.asarray(deltas, dtype=jnp.float64)


def _terms(args):
    terms = []
    if args.parameter_mode != "profile_only":
        terms.extend(
            [
                (geometry.boozer_qi_objective, 0.0, args.qi_weight),
                (geometry.boozer_maxj_objective, 0.0, args.maxj_weight),
                (geometry.vmec_aspect_ratio, args.aspect_target, args.aspect_weight),
                (geometry.vmec_iota_mean, args.iota_target, args.iota_weight),
                (geometry.vmec_mirror_ratio, args.mirror_target, args.mirror_weight),
            ]
        )
    if args.parameter_mode != "geometry_only" or args.include_er_objective_for_geometry:
        terms.append((transport.softmax_Er, args.max_er_target, args.max_er_weight))
    return tuple(term for term in terms if float(term[2]) != 0.0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--vmec-input", default=DEFAULT_VMEC_INPUT)
    parser.add_argument("--device", choices=("default", "cpu", "gpu"), default="default")
    parser.add_argument(
        "--parameter-mode",
        choices=("geometry_only", "profile_only", "profiles_plus_geometry"),
        default="geometry_only",
    )
    parser.add_argument("--profile-parameters", default="n0,T0,density_shape_power,temperature_shape_power")
    parser.add_argument("--geometry-parameters", default="RBC:1:0")
    parser.add_argument("--geometry-families", default="RBC,ZBS")
    parser.add_argument("--geometry-include-zero-harmonics", action="store_true")
    parser.add_argument("--geometry-max-mode", type=int, default=2)
    parser.add_argument("--geometry-scale-mode", default="ess")
    parser.add_argument("--geometry-ess-alpha", type=float, default=1.0)
    parser.add_argument("--include-er-objective-for-geometry", dest="include_er_objective_for_geometry", action="store_true", default=True)
    parser.add_argument("--no-er-objective-for-geometry", dest="include_er_objective_for_geometry", action="store_false")
    parser.add_argument("--max-er-target", type=float, default=30.0)
    parser.add_argument("--max-er-weight", type=float, default=1.0)
    parser.add_argument("--qi-weight", type=float, default=1.0)
    parser.add_argument("--maxj-weight", type=float, default=1.0)
    parser.add_argument("--aspect-target", type=float, default=12.0)
    parser.add_argument("--aspect-weight", type=float, default=1.0)
    parser.add_argument("--iota-target", type=float, default=-0.8)
    parser.add_argument("--iota-weight", type=float, default=1.0)
    parser.add_argument("--mirror-target", type=float, default=0.0)
    parser.add_argument("--mirror-weight", type=float, default=1.0)
    parser.add_argument("--geometry-max-iter", type=int, default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    t_phase = time.perf_counter()
    config = _prepare_config(Path(args.config), device=args.device, vmec_input=args.vmec_input)
    context = _geometry_context(config)
    print(
        "[optimization] progress: geometry context ready "
        f"elapsed_s={time.perf_counter() - t_phase:.3f}",
        flush=True,
    )
    t_phase = time.perf_counter()
    runtime, baseline_state = build_runtime_context(config)
    if baseline_state is None:
        raise RuntimeError("transport runtime did not return an initial state.")
    print(
        "[optimization] progress: realtime geometry runtime ready "
        f"elapsed_s={time.perf_counter() - t_phase:.3f}",
        flush=True,
    )

    profiles = _profile_cfg(config)
    profile_values0 = _profile_values(profiles, jnp.asarray(baseline_state.pressure).dtype)
    if args.parameter_mode == "profile_only":
        vmec_specs = ()
        profile_specs = parse_profile_parameter_specs(args.profile_parameters)
    elif args.parameter_mode == "geometry_only":
        vmec_specs = _geometry_specs(args, context)
        profile_specs = ()
    else:
        vmec_specs = _geometry_specs(args, context)
        profile_specs = parse_profile_parameter_specs(args.profile_parameters)
    if profile_specs:
        profile_lookup = {
            name: i for i, name in enumerate(TRANSPORT_REVERSE_PROFILE_PARAMETER_ORDER)
        }
        profile_parameter_values = jnp.asarray(
            [profile_values0[profile_lookup[spec.name]] for spec in profile_specs],
            dtype=profile_values0.dtype,
        )
    else:
        profile_parameter_values = jnp.zeros((0,), dtype=profile_values0.dtype)
    if vmec_specs:
        geometry_parameter_values = _baseline_geometry_deltas(config, vmec_specs)
    else:
        geometry_parameter_values = jnp.zeros((0,), dtype=jnp.float64)
    parameter_set = reverse_ad_optimization_parameter_set(
        include_profiles=bool(profile_specs),
        profiles=tuple(spec.name for spec in profile_specs) if profile_specs else None,
        vmec_boundary=tuple(vmec_specs),
    )
    parameter_value_by_label = {
        **{spec.name: profile_parameter_values[i] for i, spec in enumerate(profile_specs)},
        **{spec.label: geometry_parameter_values[i] for i, spec in enumerate(vmec_specs)},
    }

    def _initial_parameter_value(spec):
        if isinstance(spec, ProfileParameterSpec):
            return parameter_value_by_label[spec.name]
        if isinstance(spec, VmecBoundaryParameterSpec):
            return parameter_value_by_label[spec.label]
        raise TypeError(f"Unsupported optimization parameter spec type: {type(spec).__name__}.")

    parameter_values = jnp.asarray(
        [_initial_parameter_value(spec) for spec in parameter_set.specs],
        dtype=jnp.float64,
    )

    def _pre_root_state_from_profile_values(values):
        return initial_state_for_parameter_vector(
            values,
            config=config,
            initial_er_root_ad="off",
            baseline_state=baseline_state,
            profile_cfg=profiles,
            runtime=runtime,
        )

    active_terms = _terms(args)
    geom_cfg = config.get("geometry", {})
    geometry_solver_device = str(geom_cfg.get("vmec_implicit_solver_device", "default"))
    geometry_max_iter = args.geometry_max_iter
    if geometry_max_iter is None:
        geometry_max_iter = geom_cfg.get("vmec_max_iter")
    neoclassical_cfg = config.get("neoclassical", {})

    print(
        "[optimization] progress: evaluating residuals/Jacobian "
        f"parameter_mode={args.parameter_mode} parameter_count={len(parameter_set.specs)} "
        f"term_count={len(active_terms)}",
        flush=True,
    )
    t_eval = time.perf_counter()
    evaluation = evaluate_geometry_initial_er_root_only_least_squares_fused(
        config,
        parameter_set=parameter_set,
        parameter_values=parameter_values,
        terms=active_terms,
        geometry_context=context,
        runtime=runtime,
        baseline_profile_values=profile_values0,
        pre_root_state_from_profile_values=_pre_root_state_from_profile_values,
        n_r=int(geom_cfg.get("n_radial", 51)),
        n_theta=int(neoclassical_cfg.get("ntx_exact_n_theta", 25)),
        n_zeta=int(neoclassical_cfg.get("ntx_exact_n_zeta", 25)),
        n_xi=int(neoclassical_cfg.get("ntx_exact_n_xi", 64)),
        surface_backend=str(neoclassical_cfg.get("ntx_surface_backend", "vmec")),
        geometry_max_iter=geometry_max_iter,
        geometry_solver_device=geometry_solver_device,
    )
    result = evaluation.result
    residuals = jax.block_until_ready(result.residuals)
    jacobian = jax.block_until_ready(result.jacobian)
    elapsed_s = time.perf_counter() - t_eval
    residuals_np = np.asarray(jax.device_get(residuals), dtype=float)
    jacobian_np = np.asarray(jax.device_get(jacobian), dtype=float)

    print(
        "[optimization] mode=geometry_qi_max_er_initial_root "
        f"parameter_mode={args.parameter_mode} residual_count={len(result.residual_labels)} "
        f"parameter_count={len(result.parameter_labels)} elapsed_s={elapsed_s:.3f}",
        flush=True,
    )
    for row_i, label in enumerate(result.residual_labels):
        print(f"  - {label}: residual={residuals_np[row_i]:.16e}")
        for parameter_name, value in zip(result.parameter_labels, jacobian_np[row_i].tolist()):
            print(f"      d{label}/d{parameter_name}: jac={value:.16e}")

    output_path = args.output
    if output_path is None:
        output_path = (
            "outputs/autodiff_transport_lagged_ntx/reverse_ad/"
            f"geometry_qi_max_er_initial_root_{args.parameter_mode}.json"
        )
    outpath = ROOT / output_path
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(
        json.dumps(
            {
                "mode": "geometry_qi_max_er_initial_root",
                "parameter_mode": args.parameter_mode,
                "config": str(Path(args.config)),
                "vmec_input": str(args.vmec_input),
                "residual_labels": list(result.residual_labels),
                "parameter_labels": list(result.parameter_labels),
                "residuals": residuals_np.tolist(),
                "jacobian": jacobian_np.tolist(),
                "elapsed_s": float(elapsed_s),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {outpath.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
