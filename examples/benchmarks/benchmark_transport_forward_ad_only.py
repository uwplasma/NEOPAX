from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_transport_forward_fd_lane import (  # noqa: E402
    ALLOWED_PARAMETERS,
    DEFAULT_CONFIG,
    OBJECTIVE_LABELS,
    _adaptive_rollout_objectives_realized_schedule_only_for_parameter as _forward_fd_lane_realized_schedule_objectives,
    _baseline_profile_cfg,
    _objective_vector,
    _prepare_benchmark_config,
)
from benchmark_transport_autodiff_lagged_ntx import (  # noqa: E402
    _forward_benchmark_adaptive_rollout_objectives_realized_schedule_only_for_parameter_jvp,
)
from NEOPAX._geometry_autodiff import (  # noqa: E402
    build_geometry_autodiff_context,
    build_runtime_context_for_geometry_param,
)
from NEOPAX._orchestrator import build_runtime_context, prepare_transport_solver_components  # noqa: E402
from NEOPAX._transport_solvers import (  # noqa: E402
    _build_prepared_radau_accepted_rollout,
    _build_prepared_radau_execution_context,
    _radau_adaptive_final_state_rollout,
)


_REALTIME_GEOMETRY_BACKENDS = {"vmec_jax_booz_xform_jax", "vmec_runtime", "vmec_realtime"}


def _report_path(parameter_name: str) -> Path:
    safe_parameter_name = parameter_name.replace(":", "_")
    outdir = ROOT / "outputs" / "autodiff_transport_lagged_ntx" / safe_parameter_name
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / "transport_forward_ad_only_summary.json"


def _to_float_list(values) -> list[float]:
    return np.asarray(jax.device_get(values), dtype=float).tolist()


def _parse_vmec_parameter(parameter_name: str) -> tuple[str, int, int] | None:
    parts = str(parameter_name).split(":")
    if len(parts) != 4 or parts[0].strip().lower() != "vmec":
        return None
    family = parts[1].strip().upper()
    try:
        m = int(parts[2])
        n = int(parts[3])
    except ValueError as exc:
        raise ValueError(
            "VMEC geometry parameters must use the syntax 'vmec:FAMILY:m:n', "
            "for example 'vmec:RBC:1:0'."
        ) from exc
    if not family:
        raise ValueError("VMEC geometry parameter family cannot be empty.")
    return family, m, n


def _require_supported_parameter(parameter_name: str) -> None:
    if parameter_name in ALLOWED_PARAMETERS:
        return
    if _parse_vmec_parameter(parameter_name) is not None:
        return
    allowed = ", ".join(sorted(ALLOWED_PARAMETERS))
    raise ValueError(
        f"Unknown parameter '{parameter_name}'. Use one of {allowed}, "
        "or a realtime-geometry parameter like 'vmec:RBC:1:0'."
    )


def _geometry_context_from_config(config: dict[str, Any], parameter_name: str):
    vmec_parameter = _parse_vmec_parameter(parameter_name)
    if vmec_parameter is None:
        raise ValueError(f"Parameter '{parameter_name}' is not a VMEC geometry parameter.")
    family, m, n = vmec_parameter
    geom_cfg = config.get("geometry", {})
    backend = str(geom_cfg.get("backend", "")).strip().lower()
    if backend not in _REALTIME_GEOMETRY_BACKENDS:
        raise ValueError(
            "VMEC geometry parameters require a realtime geometry backend "
            f"({sorted(_REALTIME_GEOMETRY_BACKENDS)}); got backend={backend!r}."
        )
    vmec_input_file = geom_cfg.get("vmec_input_file")
    if vmec_input_file is None:
        raise ValueError("Realtime geometry parameter mode requires geometry.vmec_input_file.")
    return build_geometry_autodiff_context(
        vmec_input_file,
        param_family=family,
        param_m=m,
        param_n=n,
        mboz=int(geom_cfg.get("mboz", geom_cfg.get("vmec_mboz", 12))),
        nboz=int(geom_cfg.get("nboz", geom_cfg.get("vmec_nboz", 12))),
    )


def _adaptive_rollout_objectives_for_geometry_parameter(
    parameter_delta,
    *,
    config: dict[str, Any],
    geometry_context,
    accepted_step_limit_override: int | None = None,
):
    geom_cfg = config.get("geometry", {})
    runtime, state0 = build_runtime_context_for_geometry_param(
        config,
        geometry_context,
        parameter_delta,
        lane="ad",
        n_r=int(geom_cfg.get("n_radial", 51)),
        max_iter=geom_cfg.get("vmec_max_iter"),
        step_size=geom_cfg.get("vmec_step_size"),
        jacobian_penalty=float(geom_cfg.get("vmec_jacobian_penalty", 1.0e3)),
    )
    prepared_components = prepare_transport_solver_components(config, runtime, state0)
    solver = prepared_components["solver"]
    solve_vector_field = prepared_components["solve_vector_field"]
    prepared_rollout = _build_prepared_radau_accepted_rollout(
        solver=solver,
        state=state0,
        vector_field=solve_vector_field,
        species=runtime.species,
    )
    execution_context = _build_prepared_radau_execution_context(
        solver=solver,
        prepared_rollout=prepared_rollout,
    )
    stop_after_accepted_steps = (
        int(accepted_step_limit_override)
        if accepted_step_limit_override is not None
        else getattr(solver, "stop_after_accepted_steps", None)
    )
    max_total_steps = int(max(1, getattr(solver, "max_steps", 1)))
    if stop_after_accepted_steps is not None:
        max_total_steps = min(
            max_total_steps,
            max(int(stop_after_accepted_steps) * 16, int(stop_after_accepted_steps) + 16),
        )
    rollout = _radau_adaptive_final_state_rollout(
        execution_context,
        prepared_rollout.initial_carry,
        max_total_steps=max_total_steps,
        stop_after_accepted_steps=stop_after_accepted_steps,
    )
    final_state = prepared_rollout.physics_context.unpack_flat(rollout.final_carry.y)
    return _objective_vector(final_state, runtime)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Forward-only adaptive custom-JVP benchmark lane using scratch-compatible helper semantics."
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="Benchmark TOML.")
    parser.add_argument(
        "--parameter",
        type=str,
        default="n0",
        help=(
            "Parameter to differentiate. Use a profile parameter "
            f"({', '.join(sorted(ALLOWED_PARAMETERS))}) or a realtime geometry "
            "parameter such as vmec:RBC:1:0."
        ),
    )
    parser.add_argument("--device", type=str, default=None, help="Optional device override.")
    parser.add_argument(
        "--accepted-step-limit",
        type=int,
        default=None,
        help="Optional accepted-step prefix to stop the adaptive rollout.",
    )
    parser.add_argument(
        "--ntx-exact-derivative-mode",
        default="direct",
        choices=("direct", "custom_vjp"),
        help="NTX exact-runtime derivative mode.",
    )
    parser.add_argument(
        "--radau-jacobian-reuse-mode",
        type=str,
        default=None,
        help="Optional Radau Jacobian reuse mode override, e.g. legacy or retry_only.",
    )
    parser.add_argument(
        "--forward-ad-fusion-mode",
        type=str,
        default="replay",
        choices=("replay", "step"),
        help=(
            "Forward AD implementation. 'replay' is the recovered reference; "
            "'step' is the experimental shared primal/tangent accepted-step path."
        ),
    )
    args = parser.parse_args()
    _require_supported_parameter(args.parameter)

    config = _prepare_benchmark_config(
        Path(args.config),
        device=args.device,
        ntx_exact_derivative_mode=args.ntx_exact_derivative_mode,
        radau_jacobian_reuse_mode=args.radau_jacobian_reuse_mode,
    )
    vmec_parameter = _parse_vmec_parameter(args.parameter)
    if vmec_parameter is None:
        runtime, baseline_state = build_runtime_context(config)
        profile_cfg = _baseline_profile_cfg(config)
        baseline_value = float(profile_cfg[args.parameter])

        if args.forward_ad_fusion_mode == "replay":
            objective_fn = lambda p: _forward_benchmark_adaptive_rollout_objectives_realized_schedule_only_for_parameter_jvp(  # noqa: E731
                p,
                config=config,
                runtime=runtime,
                baseline_state=baseline_state,
                profile_cfg=profile_cfg,
                parameter_name=args.parameter,
                accepted_step_limit_override=args.accepted_step_limit,
            )
        else:
            objective_fn = lambda p: _forward_fd_lane_realized_schedule_objectives(  # noqa: E731
                p,
                config=config,
                runtime=runtime,
                baseline_state=baseline_state,
                profile_cfg=profile_cfg,
                parameter_name=args.parameter,
                accepted_step_limit_override=args.accepted_step_limit,
                derivative_mode="jvp_step",
            )
    else:
        if args.forward_ad_fusion_mode != "replay":
            raise ValueError(
                "Realtime geometry parameters currently support --forward-ad-fusion-mode replay only. "
                "The profile-only step-fused path remains unchanged."
            )
        geometry_context = _geometry_context_from_config(config, args.parameter)
        baseline_value = float(config.get("geometry", {}).get("vmec_param_delta", 0.0))

        objective_fn = lambda p: _adaptive_rollout_objectives_for_geometry_parameter(  # noqa: E731
            p,
            config=config,
            geometry_context=geometry_context,
            accepted_step_limit_override=args.accepted_step_limit,
        )

    print("[autodiff-gate] progress: running forward custom-JVP", flush=True)
    t_forward_ad_start = time.perf_counter()
    objective_values, gradient_ad = jax.jvp(
        objective_fn,
        (jnp.asarray(baseline_value),),
        (jnp.asarray(1.0),),
    )

    objective_np = np.asarray(jax.device_get(objective_values), dtype=float)
    grad_np = np.asarray(jax.device_get(gradient_ad), dtype=float)
    forward_ad_total_s = time.perf_counter() - t_forward_ad_start
    report = {
        "mode": "transport_forward_ad_only",
        "config_path": str(Path(args.config)),
        "parameter_name": args.parameter,
        "parameter_kind": "vmec_geometry" if vmec_parameter is not None else "profile",
        "baseline_value": baseline_value,
        "accepted_step_limit": None if args.accepted_step_limit is None else int(args.accepted_step_limit),
        "ntx_exact_derivative_mode": str(args.ntx_exact_derivative_mode),
        "radau_jacobian_reuse_mode": None if args.radau_jacobian_reuse_mode is None else str(args.radau_jacobian_reuse_mode),
        "forward_ad_fusion_mode": str(args.forward_ad_fusion_mode),
        "forward_ad_total_s": float(forward_ad_total_s),
        "objective_labels": OBJECTIVE_LABELS,
        "objective_values": objective_np.tolist(),
        "gradient_forward_ad": grad_np.tolist(),
    }

    print(
        f"[autodiff-gate] mode=transport_forward_ad_only parameter={args.parameter} "
        f"parameter_kind={'vmec_geometry' if vmec_parameter is not None else 'profile'} "
        f"baseline_value={baseline_value:.6e} "
        f"radau_jacobian_reuse_mode={args.radau_jacobian_reuse_mode} "
        f"forward_ad_fusion_mode={args.forward_ad_fusion_mode} "
        f"forward_ad_total_s={forward_ad_total_s:.6e}"
    )
    print("[autodiff-gate] objective values:")
    for label, value in zip(OBJECTIVE_LABELS, _to_float_list(objective_values)):
        print(f"  - {label}: value={float(value):.16e}")
    print("[autodiff-gate] objective tangents:")
    for label, value in zip(OBJECTIVE_LABELS, _to_float_list(gradient_ad)):
        print(f"  - {label}: ad={float(value):.6e}")

    outpath = _report_path(args.parameter)
    outpath.write_text(json.dumps(report, indent=2))
    print(f"Wrote {outpath.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
