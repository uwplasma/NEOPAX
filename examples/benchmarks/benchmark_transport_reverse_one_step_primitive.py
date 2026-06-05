from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from benchmark_transport_autodiff_lagged_ntx import (  # noqa: E402
    DEFAULT_CONFIG,
    PROFILE_VECTOR_PARAMETERS,
    _baseline_profile_cfg,
    _prepare_benchmark_config,
    _prepare_realized_schedule_profile_vector_rollout_option_a,
)
from NEOPAX._orchestrator import build_runtime_context  # noqa: E402
from NEOPAX._transport_solvers import (  # noqa: E402
    _RadauAcceptedStepReducedOutput,
    _radau_accepted_step_primitive,
    _radau_accepted_step_primitive_pullback,
)


def _report_path() -> Path:
    outdir = ROOT / "outputs" / "autodiff_transport_lagged_ntx" / "one_step_primitive"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / "transport_reverse_one_step_primitive_summary.json"


def _parse_parameter_subset(text: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in str(text).split(",") if item.strip())
    if not values:
        raise ValueError("At least one profile-vector parameter must be provided.")
    invalid = [name for name in values if name not in PROFILE_VECTOR_PARAMETERS]
    if invalid:
        raise ValueError(
            f"Unsupported profile-vector parameter(s): {invalid}. "
            f"Allowed values: {list(PROFILE_VECTOR_PARAMETERS)}"
        )
    return values


def _tree_max_abs(tree) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float64)
    vals = [jnp.max(jnp.abs(jnp.asarray(leaf, dtype=jnp.float64))) for leaf in leaves]
    return jnp.max(jnp.stack(vals))


def _make_reduced_output_bar(carry, mode: str) -> _RadauAcceptedStepReducedOutput:
    if mode == "y-only":
        return _RadauAcceptedStepReducedOutput(
            t_out=jnp.zeros_like(carry.t),
            y_out=jnp.ones_like(carry.y),
            dt_out=jnp.zeros_like(carry.dt),
            prev_stages_out=jnp.zeros_like(carry.prev_stages),
            prev_dt_out=jnp.zeros_like(carry.prev_dt),
            lagged_reference_y_out=jnp.zeros_like(carry.lagged_reference_y),
            prev_theta_final_out=jnp.zeros_like(carry.prev_theta_final),
        )
    if mode == "all-ones":
        return _RadauAcceptedStepReducedOutput(
            t_out=jnp.ones_like(carry.t),
            y_out=jnp.ones_like(carry.y),
            dt_out=jnp.ones_like(carry.dt),
            prev_stages_out=jnp.ones_like(carry.prev_stages),
            prev_dt_out=jnp.ones_like(carry.prev_dt),
            lagged_reference_y_out=jnp.ones_like(carry.lagged_reference_y),
            prev_theta_final_out=jnp.ones_like(carry.prev_theta_final),
        )
    raise ValueError(f"Unsupported bar mode: {mode}")


def _zero_like_value(value):
    if value is None:
        return None

    def _zero_leaf(leaf):
        if leaf is None:
            return None
        arr = jnp.asarray(leaf)
        if jnp.issubdtype(arr.dtype, jnp.bool_):
            return jnp.zeros_like(arr, dtype=jnp.bool_)
        return jnp.zeros_like(arr)

    return jax.tree_util.tree_map(_zero_leaf, value)


def _ablate_reverse_payload(reverse_payload, ablation_mode: str):
    if ablation_mode == "none":
        return reverse_payload

    fields = {}
    for field in dataclasses.fields(reverse_payload):
        name = field.name
        value = getattr(reverse_payload, name)
        zero = False
        if ablation_mode == "stage" and name in {"stage_history"}:
            zero = True
        elif ablation_mode == "lagged" and name in {"lagged_response_in", "lagged_response_cache_in", "rhs_time_ref"}:
            zero = True
        elif ablation_mode == "jacobian" and name in {"jacobian_out"}:
            zero = True
        elif ablation_mode == "lu" and name in {"real_lu_out", "complex_lu_out"}:
            zero = True
        elif ablation_mode == "pivots" and name in {"real_piv_out", "complex_piv_out"}:
            zero = True
        fields[name] = _zero_like_value(value) if zero else value
    return dataclasses.replace(reverse_payload, **fields)


def _structure_leaf_names(structure_mode: str) -> tuple[str, ...]:
    if structure_mode == "full":
        return ()
    if structure_mode == "scalars":
        return (
            "t_in",
            "dt_in",
            "trial_dt",
            "prev_dt_in",
            "prev_theta_final_in",
            "prev_newton_iter_count_in",
            "lagged_response_valid_in",
        )
    if structure_mode == "trial_dt":
        return ("trial_dt",)
    if structure_mode == "t_in":
        return ("t_in",)
    if structure_mode == "dt_in":
        return ("dt_in",)
    if structure_mode == "prev_dt":
        return ("prev_dt_in",)
    if structure_mode == "lagged_valid":
        return ("lagged_response_valid_in",)
    if structure_mode == "newton_count":
        return ("prev_newton_iter_count_in",)
    if structure_mode == "prev_theta":
        return ("prev_theta_final_in",)
    if structure_mode == "stage":
        return ("stage_history",)
    raise ValueError(f"Unsupported structure mode: {structure_mode}")


def _payload_kwargs(reverse_payload) -> dict[str, Any]:
    return {field.name: getattr(reverse_payload, field.name) for field in dataclasses.fields(reverse_payload)}


def _payload_leaf_stats(reverse_payload):
    stats = []
    total_bytes = 0
    groups = {
        "stage": {"stage_history"},
        "lagged": {"lagged_response_in", "lagged_response_cache_in", "rhs_time_ref"},
        "jacobian": {"jacobian_out"},
        "lu": {"real_lu_out", "complex_lu_out"},
        "pivots": {"real_piv_out", "complex_piv_out"},
        "scalars_other": set(),
    }
    group_totals = {key: 0 for key in groups}
    for field in dataclasses.fields(reverse_payload):
        name = field.name
        value = getattr(reverse_payload, name)
        leaves = [leaf for leaf in jax.tree_util.tree_leaves(value) if leaf is not None]
        arrays = [np.asarray(jax.device_get(leaf)) for leaf in leaves]
        bytes_used = int(sum(arr.nbytes for arr in arrays))
        total_bytes += bytes_used
        group_name = "scalars_other"
        for candidate, members in groups.items():
            if name in members:
                group_name = candidate
                break
        group_totals[group_name] += bytes_used
        if arrays:
            combined_elements = int(sum(arr.size for arr in arrays))
            combined_dtype = ",".join(sorted({str(arr.dtype) for arr in arrays}))
            shape_repr = [list(arr.shape) for arr in arrays[:3]]
            if len(arrays) > 3:
                shape_repr.append(["..."])
        else:
            combined_elements = 0
            combined_dtype = "None"
            shape_repr = []
        stats.append(
            {
                "name": name,
                "shape": shape_repr,
                "dtype": combined_dtype,
                "elements": combined_elements,
                "bytes": bytes_used,
                "group": group_name,
            }
        )
    stats.sort(key=lambda item: item["bytes"], reverse=True)
    return {
        "total_bytes": total_bytes,
        "group_totals": group_totals,
        "leaves": stats,
    }


def _compute_one_step_metrics(
    execution_context,
    initial_carry,
    *,
    bar_mode: str,
    execution_mode: str,
    payload_mode: str,
    ablation_mode: str,
    dynamic_structure: str,
):
    kernel_context = execution_context.kernel_context
    physics_context = execution_context.physics_context
    attempt_context = execution_context.attempt_context

    primitive_result = _radau_accepted_step_primitive(
        kernel_context,
        physics_context,
        initial_carry,
        attempt_context,
    )
    reverse_payload = _ablate_reverse_payload(primitive_result.reverse_payload, ablation_mode)
    reduced_output_bar = _make_reduced_output_bar(initial_carry, bar_mode)
    payload_field_names = tuple(field.name for field in dataclasses.fields(reverse_payload))

    def _primitive_only_fn():
        primitive_reduced_bar = _radau_accepted_step_primitive_pullback(
            kernel_context,
            physics_context,
            initial_carry,
            attempt_context,
            reverse_payload,
            reduced_output_bar,
        )
        return {
            "converged": primitive_result.attempt_result.converged,
            "err_norm": primitive_result.attempt_result.err_norm,
            "trial_dt": primitive_result.attempt_result.trial_dt,
            "newton_iter_count": primitive_result.attempt_result.newton_iter_count,
            "primitive_y_bar_max": _tree_max_abs(primitive_reduced_bar.y_out),
            "primitive_dt_bar_abs": jnp.max(jnp.abs(jnp.asarray(primitive_reduced_bar.dt_out, dtype=jnp.float64))),
            "primitive_prev_stages_bar_max": _tree_max_abs(primitive_reduced_bar.prev_stages_out),
        }

    def _primitive_dynamic_fn(reverse_payload, reduced_bar):
        primitive_reduced_bar = _radau_accepted_step_primitive_pullback(
            kernel_context,
            physics_context,
            initial_carry,
            attempt_context,
            reverse_payload,
            reduced_bar,
        )
        return {
            "converged": primitive_result.attempt_result.converged,
            "err_norm": primitive_result.attempt_result.err_norm,
            "trial_dt": primitive_result.attempt_result.trial_dt,
            "newton_iter_count": primitive_result.attempt_result.newton_iter_count,
            "primitive_y_bar_max": _tree_max_abs(primitive_reduced_bar.y_out),
            "primitive_dt_bar_abs": jnp.max(jnp.abs(jnp.asarray(primitive_reduced_bar.dt_out, dtype=jnp.float64))),
            "primitive_prev_stages_bar_max": _tree_max_abs(primitive_reduced_bar.prev_stages_out),
        }

    if dynamic_structure == "full":
        dynamic_leaf_names = payload_field_names
    else:
        dynamic_leaf_names = tuple(name for name in _structure_leaf_names(dynamic_structure) if name in payload_field_names)
    payload_base_kwargs = _payload_kwargs(reverse_payload)

    def _rebuild_payload_from_dynamic(dynamic_values):
        updated = dict(payload_base_kwargs)
        for name, value in zip(dynamic_leaf_names, dynamic_values):
            updated[name] = value
        return dataclasses.replace(reverse_payload, **updated)

    def _primitive_dynamic_payload_only_fn(reverse_payload_dynamic):
        return _primitive_dynamic_fn(reverse_payload_dynamic, reduced_output_bar)

    def _primitive_dynamic_selected_fn(dynamic_values):
        rebuilt_payload = _rebuild_payload_from_dynamic(dynamic_values)
        return _primitive_dynamic_fn(rebuilt_payload, reduced_output_bar)

    if execution_mode == "jit":
        if payload_mode == "closed-over":
            compare_fn = jax.jit(_primitive_only_fn)
            call = lambda: compare_fn()
        elif payload_mode == "dynamic":
            compare_fn = jax.jit(_primitive_dynamic_fn)
            call = lambda: compare_fn(reverse_payload, reduced_output_bar)
        elif payload_mode == "dynamic-payload-only":
            compare_fn = jax.jit(_primitive_dynamic_payload_only_fn)
            call = lambda: compare_fn(reverse_payload)
        elif payload_mode == "dynamic-selected":
            compare_fn = jax.jit(_primitive_dynamic_selected_fn)
            dynamic_values = tuple(getattr(reverse_payload, name) for name in dynamic_leaf_names)
            call = lambda: compare_fn(dynamic_values)
        else:
            raise ValueError(f"Unsupported payload mode: {payload_mode}")
        t0 = time.perf_counter()
        first = call()
        jax.block_until_ready(first["primitive_y_bar_max"])
        compile_plus_execute_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        second = call()
        jax.block_until_ready(second["primitive_y_bar_max"])
        execute_s = time.perf_counter() - t1
    else:
        t0 = time.perf_counter()
        with jax.disable_jit():
            if payload_mode == "closed-over":
                second = _primitive_only_fn()
            elif payload_mode == "dynamic":
                second = _primitive_dynamic_fn(reverse_payload, reduced_output_bar)
            elif payload_mode == "dynamic-payload-only":
                second = _primitive_dynamic_payload_only_fn(reverse_payload)
            elif payload_mode == "dynamic-selected":
                dynamic_values = tuple(getattr(reverse_payload, name) for name in dynamic_leaf_names)
                second = _primitive_dynamic_selected_fn(dynamic_values)
            else:
                raise ValueError(f"Unsupported payload mode: {payload_mode}")
        jax.block_until_ready(second["primitive_y_bar_max"])
        execute_s = time.perf_counter() - t0
        compile_plus_execute_s = execute_s

    result = {key: np.asarray(jax.device_get(value)).item() for key, value in second.items()}
    result["compile_plus_execute_s"] = compile_plus_execute_s
    result["execute_s"] = execute_s
    payload_report = _payload_leaf_stats(reverse_payload)
    payload_report["dynamic_leaf_names"] = list(dynamic_leaf_names)
    payload_report["dynamic_leaf_count"] = int(len(dynamic_leaf_names))
    return result, payload_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the one-step accepted-step primitive reverse rule."
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="Benchmark TOML.")
    parser.add_argument("--device", type=str, default=None, help="Optional device override passed to config preparation.")
    parser.add_argument(
        "--ntx-exact-derivative-mode",
        default="direct",
        choices=("direct", "custom_vjp"),
        help="NTX exact-runtime derivative mode. Use direct for this benchmark.",
    )
    parser.add_argument(
        "--parameters",
        default=",".join(PROFILE_VECTOR_PARAMETERS),
        help="Comma-separated subset of profile-vector parameters to include.",
    )
    parser.add_argument(
        "--accepted-step-limit",
        type=int,
        default=1,
        help="Accepted-step limit used only to prepare the baseline rollout context. Default: 1.",
    )
    parser.add_argument(
        "--bar-mode",
        default="y-only",
        choices=("y-only", "all-ones"),
        help="Cotangent pattern used for the local reduced-output pullback comparison.",
    )
    parser.add_argument(
        "--execution-mode",
        default="eager",
        choices=("eager", "jit"),
        help="Run the one-step comparison eagerly or under JIT. Default: eager.",
    )
    parser.add_argument(
        "--payload-mode",
        default="closed-over",
        choices=("closed-over", "dynamic", "dynamic-payload-only", "dynamic-selected"),
        help="How much of the one-step reverse contract is passed dynamically at runtime. Default: closed-over.",
    )
    parser.add_argument(
        "--payload-ablation",
        default="none",
        choices=("none", "stage", "lagged", "jacobian", "lu", "pivots"),
        help="Zero one payload family before running the reverse benchmark. Default: none.",
    )
    parser.add_argument(
        "--dynamic-structure",
        default="full",
        choices=(
            "full",
            "scalars",
            "trial_dt",
            "t_in",
            "dt_in",
            "prev_dt",
            "lagged_valid",
            "newton_count",
            "prev_theta",
            "stage",
        ),
        help="When using `dynamic-selected`, choose which payload leaves remain dynamic. Default: full.",
    )
    args = parser.parse_args()

    parameter_names = _parse_parameter_subset(args.parameters)
    config = _prepare_benchmark_config(
        Path(args.config),
        device=args.device,
        ntx_exact_derivative_mode=args.ntx_exact_derivative_mode,
    )
    runtime, baseline_state = build_runtime_context(config)
    profile_cfg = _baseline_profile_cfg(config)
    baseline_vector = jnp.asarray([float(profile_cfg[name]) for name in parameter_names], dtype=jnp.float64)

    execution_context, _prepared_rollout, initial_carry, _max_total_steps, _stop_after_accepted_steps, _solver, _solve_vector_field = (
        _prepare_realized_schedule_profile_vector_rollout_option_a(
            baseline_vector,
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_names=parameter_names,
            accepted_step_limit_override=args.accepted_step_limit,
        )
    )

    result, payload_report = _compute_one_step_metrics(
        execution_context,
        initial_carry,
        bar_mode=args.bar_mode,
        execution_mode=args.execution_mode,
        payload_mode=args.payload_mode,
        ablation_mode=args.payload_ablation,
        dynamic_structure=args.dynamic_structure,
    )

    report = {
        "config": str(args.config),
        "device": args.device,
        "ntx_exact_derivative_mode": args.ntx_exact_derivative_mode,
        "parameter_names": list(parameter_names),
        "parameter_values": [float(x) for x in np.asarray(jax.device_get(baseline_vector), dtype=float)],
        "accepted_step_limit": int(args.accepted_step_limit),
        "bar_mode": args.bar_mode,
        "execution_mode": args.execution_mode,
        "payload_mode": args.payload_mode,
        "payload_ablation": args.payload_ablation,
        "dynamic_structure": args.dynamic_structure,
        "payload_report": payload_report,
        "result": result,
    }

    outpath = _report_path()
    outpath.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("[autodiff-gate] mode=transport_reverse_one_step_primitive")
    print(f"[autodiff-gate] parameters={list(parameter_names)}")
    print(
        f"[autodiff-gate] bar_mode={args.bar_mode} "
        f"accepted_step_limit={int(args.accepted_step_limit)} execution_mode={args.execution_mode} "
        f"payload_mode={args.payload_mode} payload_ablation={args.payload_ablation} "
        f"dynamic_structure={args.dynamic_structure}"
    )
    print(
        f"[autodiff-gate] payload_total_bytes={int(payload_report['total_bytes'])} "
        f"group_totals={payload_report['group_totals']} "
        f"dynamic_leaf_names={payload_report['dynamic_leaf_names']}"
    )
    for key, value in result.items():
        if isinstance(value, bool):
            print(f"  - {key}: {value}")
        elif "count" in key:
            print(f"  - {key}: {int(value)}")
        else:
            print(f"  - {key}: {float(value):.6e}")
    print(f"[autodiff-gate] wrote={outpath}")


if __name__ == "__main__":
    main()
