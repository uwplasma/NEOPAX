from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import sys
import time
from pathlib import Path

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
    _RadauAcceptedPrimitivePayloadRollout,
    _RadauAcceptedStepForwardLikeCacheNoStageCotangent,
    _RadauAcceptedStepForwardLikeCotangent,
    _RadauAcceptedStepForwardLikeNoStageCotangent,
    _RadauAcceptedStepReducedOutput,
    _RadauAcceptedStepReversePayload,
    _radau_accepted_step_forward_like_cache_no_stage_pullback,
    _radau_accepted_step_forward_like_pullback,
    _radau_accepted_step_forward_like_no_stage_pullback,
    _radau_accepted_step_primitive,
    _radau_accepted_step_primitive_pullback,
    _radau_adaptive_schedule_rollout,
    _radau_carry_with_forward_only_jvp_fields,
    _radau_contract_reduced_output_bar,
    _radau_forward_like_cache_no_stage_cotangent_from_reduced_output_bar,
    _radau_forward_like_cotangent_from_reduced_output_bar,
    _radau_forward_like_no_stage_cotangent_from_reduced_output_bar,
    _radau_replay_realized_accepted_rollout,
)


def _report_path() -> Path:
    outdir = ROOT / "outputs" / "autodiff_transport_lagged_ntx" / "multi_step_primitive"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / "transport_reverse_multi_step_primitive_summary.json"


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


def _parse_step_counts(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in str(text).split(",") if item.strip())
    if not values:
        raise ValueError("At least one accepted-step count must be provided.")
    if any(value <= 0 for value in values):
        raise ValueError("Accepted-step counts must be positive integers.")
    return values


def _tree_max_abs(tree) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float64)
    vals = [jnp.max(jnp.abs(jnp.asarray(leaf, dtype=jnp.float64))) for leaf in leaves]
    return jnp.max(jnp.stack(vals))


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
        stats.append({"name": name, "bytes": bytes_used, "group": group_name})
    stats.sort(key=lambda item: item["bytes"], reverse=True)
    return {"total_bytes": total_bytes, "group_totals": group_totals, "leaves": stats}


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


def _ablate_reduced_output_bar(
    reduced_output_bar: _RadauAcceptedStepReducedOutput,
    ablation_mode: str,
) -> _RadauAcceptedStepReducedOutput:
    if ablation_mode == "none":
        return reduced_output_bar

    replacements = {}
    if ablation_mode == "prev-stages":
        replacements["prev_stages_out"] = jnp.zeros_like(reduced_output_bar.prev_stages_out)
    elif ablation_mode == "lagged-reference-y":
        replacements["lagged_reference_y_out"] = jnp.zeros_like(reduced_output_bar.lagged_reference_y_out)
    elif ablation_mode == "step-meta":
        replacements["dt_out"] = jnp.zeros_like(reduced_output_bar.dt_out)
        replacements["prev_dt_out"] = jnp.zeros_like(reduced_output_bar.prev_dt_out)
        replacements["prev_theta_final_out"] = jnp.zeros_like(reduced_output_bar.prev_theta_final_out)
    elif ablation_mode == "non-y":
        replacements["t_out"] = jnp.zeros_like(reduced_output_bar.t_out)
        replacements["dt_out"] = jnp.zeros_like(reduced_output_bar.dt_out)
        replacements["prev_stages_out"] = jnp.zeros_like(reduced_output_bar.prev_stages_out)
        replacements["prev_dt_out"] = jnp.zeros_like(reduced_output_bar.prev_dt_out)
        replacements["lagged_reference_y_out"] = jnp.zeros_like(reduced_output_bar.lagged_reference_y_out)
        replacements["prev_theta_final_out"] = jnp.zeros_like(reduced_output_bar.prev_theta_final_out)
    else:
        raise ValueError(f"Unsupported bar ablation mode: {ablation_mode}")
    return dataclasses.replace(reduced_output_bar, **replacements)


def _forward_like_cotangent_metrics(cotangent: _RadauAcceptedStepForwardLikeCotangent) -> dict[str, jax.Array]:
    return {
        "primitive_y_bar_max": _tree_max_abs(cotangent.y),
        "primitive_dt_bar_abs": jnp.max(jnp.abs(jnp.asarray(cotangent.dt, dtype=jnp.float64))),
        "primitive_prev_stages_bar_max": _tree_max_abs(cotangent.prev_stages),
    }


def _forward_like_no_stage_cotangent_metrics(
    cotangent: _RadauAcceptedStepForwardLikeNoStageCotangent,
) -> dict[str, jax.Array]:
    return {
        "primitive_y_bar_max": _tree_max_abs(cotangent.y),
        "primitive_dt_bar_abs": jnp.max(jnp.abs(jnp.asarray(cotangent.dt, dtype=jnp.float64))),
        "primitive_prev_stages_bar_max": jnp.asarray(0.0, dtype=jnp.float64),
    }


def _forward_like_cache_no_stage_cotangent_metrics(
    cotangent: _RadauAcceptedStepForwardLikeCacheNoStageCotangent,
) -> dict[str, jax.Array]:
    return {
        "primitive_y_bar_max": _tree_max_abs(cotangent.y),
        "primitive_dt_bar_abs": jnp.max(jnp.abs(jnp.asarray(cotangent.dt, dtype=jnp.float64))),
        "primitive_prev_stages_bar_max": jnp.asarray(0.0, dtype=jnp.float64),
    }


def _payload_field_names_without_lagged_valid() -> tuple[str, ...]:
    return tuple(
        field.name
        for field in dataclasses.fields(_RadauAcceptedStepReversePayload)
        if field.name != "lagged_response_valid_in"
    )


def _prefix_transport_config(
    config: dict,
    *,
    accepted_step_limit: int,
    max_total_steps_multiplier: int,
) -> dict:
    tuned = copy.deepcopy(config)
    solver_cfg = tuned.setdefault("transport_solver", {})
    solver_cfg["stop_after_accepted_steps"] = int(accepted_step_limit)
    solver_cfg["max_steps"] = max(
        int(accepted_step_limit),
        int(accepted_step_limit) * int(max_total_steps_multiplier),
    )
    return tuned


def _compute_multi_step_metrics(
    execution_context,
    initial_carry,
    *,
    max_total_steps: int,
    stop_after_accepted_steps: int | None,
    bar_mode: str,
    execution_mode: str,
    segment_length: int,
    checkpoint_count: int,
    reverse_probe_mode: str,
    payload_ablation: str,
    bar_ablation: str,
    cotangent_contract: str,
    reverse_compose_mode: str,
):
    schedule_rollout = _radau_adaptive_schedule_rollout(
        execution_context,
        initial_carry,
        max_total_steps=max_total_steps,
        stop_after_accepted_steps=stop_after_accepted_steps,
    )
    accepted_mask_np = np.asarray(
        jax.device_get(jnp.logical_and(schedule_rollout.trace.active_mask, schedule_rollout.trace.accepted_mask)),
        dtype=bool,
    )
    attempted_dts_np = np.asarray(jax.device_get(schedule_rollout.trace.attempted_dts), dtype=float)
    accepted_dts_np = attempted_dts_np[accepted_mask_np]
    next_dts_np = np.asarray(jax.device_get(schedule_rollout.trace.next_dts), dtype=float)[accepted_mask_np]
    next_recent_reject_count_np = np.asarray(
        jax.device_get(schedule_rollout.trace.next_recent_reject_count),
        dtype=np.int32,
    )[accepted_mask_np]
    next_regrowth_cooldown_np = np.asarray(
        jax.device_get(schedule_rollout.trace.next_regrowth_cooldown),
        dtype=np.int32,
    )[accepted_mask_np]
    next_easy_growth_streak_np = np.asarray(
        jax.device_get(schedule_rollout.trace.next_easy_growth_streak),
        dtype=np.int32,
    )[accepted_mask_np]
    next_lagged_response_valid_np = np.asarray(
        jax.device_get(schedule_rollout.trace.next_lagged_response_valid),
        dtype=bool,
    )[accepted_mask_np]
    accepted_count = int(accepted_dts_np.shape[0])
    base_final_output_bar = _ablate_reduced_output_bar(
        _radau_contract_reduced_output_bar(
            _make_reduced_output_bar(schedule_rollout.final_carry, bar_mode),
            cotangent_contract,
        ),
        bar_ablation,
    )
    if cotangent_contract == "forward-like-v1":
        final_reverse_state = _radau_forward_like_cotangent_from_reduced_output_bar(
            base_final_output_bar,
            initial_carry,
        )
    elif cotangent_contract == "forward-like-v2-no-stage":
        final_reverse_state = _radau_forward_like_no_stage_cotangent_from_reduced_output_bar(
            base_final_output_bar,
        )
    elif cotangent_contract == "forward-like-v3-cache-no-stage":
        final_reverse_state = _radau_forward_like_cache_no_stage_cotangent_from_reduced_output_bar(
            base_final_output_bar,
            initial_carry,
        )
    else:
        final_reverse_state = base_final_output_bar
    segment_length = max(1, int(segment_length))
    checkpoint_count = max(0, int(checkpoint_count))

    if accepted_count == 0:
        return {
            "accepted_count": 0,
            "segment_count": 0,
            "checkpoint_count": 0,
            "segment_length": int(segment_length),
            "primitive_y_bar_max": 0.0,
            "primitive_dt_bar_abs": 0.0,
            "primitive_prev_stages_bar_max": 0.0,
            "compile_plus_execute_s": 0.0,
            "execute_s": 0.0,
        }

    segment_ranges = [
        (start_idx, min(start_idx + segment_length, accepted_count))
        for start_idx in range(0, accepted_count, segment_length)
    ]
    segment_starts = [start_idx for start_idx, _ in segment_ranges]
    accepted_dts_host = tuple(float(x) for x in accepted_dts_np.tolist())
    next_dts_host = tuple(float(x) for x in next_dts_np.tolist())
    next_recent_reject_count_host = tuple(int(x) for x in next_recent_reject_count_np.tolist())
    next_regrowth_cooldown_host = tuple(int(x) for x in next_regrowth_cooldown_np.tolist())
    next_easy_growth_streak_host = tuple(int(x) for x in next_easy_growth_streak_np.tolist())
    next_lagged_response_valid_host = tuple(bool(x) for x in next_lagged_response_valid_np.tolist())
    dtype = execution_context.dtype
    replay_cache = {}

    def _accepted_dt_slice(start_idx: int, end_idx: int):
        return jnp.asarray(accepted_dts_host[start_idx:end_idx], dtype=dtype)

    def _accepted_next_dt_slice(start_idx: int, end_idx: int):
        return jnp.asarray(next_dts_host[start_idx:end_idx], dtype=dtype)

    def _accepted_recent_reject_slice(start_idx: int, end_idx: int):
        return jnp.asarray(next_recent_reject_count_host[start_idx:end_idx], dtype=jnp.int32)

    def _accepted_regrowth_slice(start_idx: int, end_idx: int):
        return jnp.asarray(next_regrowth_cooldown_host[start_idx:end_idx], dtype=jnp.int32)

    def _accepted_growth_streak_slice(start_idx: int, end_idx: int):
        return jnp.asarray(next_easy_growth_streak_host[start_idx:end_idx], dtype=jnp.int32)

    def _accepted_lagged_valid_slice(start_idx: int, end_idx: int):
        return jnp.asarray(next_lagged_response_valid_host[start_idx:end_idx], dtype=jnp.bool_)

    def _replay_segment(
        carry_start,
        accepted_active_mask,
        dt_slice,
        next_dt_slice,
        recent_reject_slice,
        regrowth_slice,
        growth_streak_slice,
        lagged_valid_slice,
    ):
        replay = _radau_replay_realized_accepted_rollout(
            execution_context,
            carry_start,
            accepted_active_mask,
            dt_slice,
            next_dt_slice,
            recent_reject_slice,
            regrowth_slice,
            growth_streak_slice,
            lagged_valid_slice,
        )
        return replay.final_carry

    def _replay_from_carry(carry_start, start_idx: int, end_idx: int):
        dt_slice = _accepted_dt_slice(start_idx, end_idx)
        if dt_slice.shape[0] == 0:
            return carry_start
        if execution_mode == "jit":
            length = int(dt_slice.shape[0])
            fn = replay_cache.get(length)
            if fn is None:
                fn = jax.jit(_replay_segment)
                replay_cache[length] = fn
            return fn(
                carry_start,
                jnp.ones((dt_slice.shape[0],), dtype=jnp.bool_),
                dt_slice,
                _accepted_next_dt_slice(start_idx, end_idx),
                _accepted_recent_reject_slice(start_idx, end_idx),
                _accepted_regrowth_slice(start_idx, end_idx),
                _accepted_growth_streak_slice(start_idx, end_idx),
                _accepted_lagged_valid_slice(start_idx, end_idx),
            )
        with jax.disable_jit():
            return _replay_segment(
                carry_start,
                jnp.ones((dt_slice.shape[0],), dtype=jnp.bool_),
                dt_slice,
                _accepted_next_dt_slice(start_idx, end_idx),
                _accepted_recent_reject_slice(start_idx, end_idx),
                _accepted_regrowth_slice(start_idx, end_idx),
                _accepted_growth_streak_slice(start_idx, end_idx),
                _accepted_lagged_valid_slice(start_idx, end_idx),
            )

    def _select_checkpoint_starts() -> tuple[int, ...]:
        if checkpoint_count <= 0 or len(segment_starts) <= 1:
            return ()
        internal_starts = segment_starts[1:]
        if checkpoint_count >= len(internal_starts):
            return tuple(int(x) for x in internal_starts)
        positions = np.linspace(0, len(internal_starts) - 1, num=checkpoint_count, dtype=int)
        selected = []
        seen = set()
        for pos in positions.tolist():
            start_value = int(internal_starts[int(pos)])
            if start_value not in seen:
                selected.append(start_value)
                seen.add(start_value)
        return tuple(selected)

    checkpoint_starts = _select_checkpoint_starts()
    checkpoint_map: dict[int, object] = {0: initial_carry}
    if checkpoint_starts:
        running_carry = initial_carry
        running_start = 0
        for checkpoint_start in checkpoint_starts:
            running_carry = _replay_from_carry(running_carry, running_start, checkpoint_start)
            checkpoint_map[int(checkpoint_start)] = running_carry
            running_start = checkpoint_start

    def _segment_start_carry(segment_start: int):
        available = [idx for idx in checkpoint_map.keys() if idx <= segment_start]
        checkpoint_start = max(available) if available else 0
        carry_start = checkpoint_map[checkpoint_start]
        if checkpoint_start == segment_start:
            return carry_start
        return _replay_from_carry(carry_start, checkpoint_start, segment_start)

    def _collect_segment_payloads(carry_start, dt_segment):
        def _scan_body(carry, dt_value):
            carry_for_step = dataclasses.replace(carry, dt=dt_value)
            primitive_result = _radau_accepted_step_primitive(
                execution_context.kernel_context,
                execution_context.physics_context,
                carry_for_step,
                execution_context.attempt_context,
            )
            next_carry = dataclasses.replace(
                primitive_result.next_carry,
                prev_error=jnp.maximum(
                    primitive_result.attempt_result.err_norm,
                    jnp.asarray(1.0e-12, dtype=dtype),
                ),
                recent_reject_count=jnp.asarray(0, dtype=jnp.int32),
                regrowth_cooldown=jnp.asarray(0, dtype=jnp.int32),
                easy_growth_streak=jnp.asarray(0, dtype=jnp.int32),
            )
            return next_carry, (
                primitive_result.reduced_output,
                primitive_result.reverse_payload,
                dt_value,
            )

        carry_seed = _radau_carry_with_forward_only_jvp_fields(carry_start)
        final_carry, scan_outputs = jax.lax.scan(_scan_body, carry_seed, dt_segment)
        reduced_outputs, reverse_payloads, accepted_dts = scan_outputs
        return _RadauAcceptedPrimitivePayloadRollout(
            final_carry=final_carry,
            accepted_mask=jnp.ones((dt_segment.shape[0],), dtype=jnp.bool_),
            accepted_dts=accepted_dts,
            reduced_outputs=reduced_outputs,
            reverse_payloads=reverse_payloads,
        )

    payload_collect_cache = {}
    last_step_payload_report = None

    def _collect_segment_payloads_compiled(carry_start, dt_segment):
        length = int(dt_segment.shape[0])
        if execution_mode == "jit":
            fn = payload_collect_cache.get(length)
            if fn is None:
                fn = jax.jit(_collect_segment_payloads)
                payload_collect_cache[length] = fn
            return fn(carry_start, dt_segment)
        with jax.disable_jit():
            return _collect_segment_payloads(carry_start, dt_segment)

    carry_template = initial_carry

    payload_dynamic_field_names = _payload_field_names_without_lagged_valid()

    def _build_reverse_segment_fn(payload_template, reverse_payloads_static):
        payload_base_kwargs = {
            field.name: getattr(payload_template, field.name)
            for field in dataclasses.fields(_RadauAcceptedStepReversePayload)
        }

        def _payload_from_dynamic_values(dynamic_values, lagged_valid_value: bool):
            updated = dict(payload_base_kwargs)
            updated["lagged_response_valid_in"] = jnp.asarray(lagged_valid_value, dtype=jnp.bool_)
            for name, value in zip(payload_dynamic_field_names, dynamic_values):
                updated[name] = value
            return dataclasses.replace(payload_template, **updated)

        reversed_payloads = jax.tree_util.tree_map(lambda x: jnp.flip(x, axis=0), reverse_payloads_static)
        lagged_valids = reversed_payloads.lagged_response_valid_in
        reversed_dynamic_values = tuple(getattr(reversed_payloads, name) for name in payload_dynamic_field_names)

        def _segment_reverse_impl(carry_bar):
            lagged_valids = reversed_payloads.lagged_response_valid_in

            def _scan_body(next_bar, xs):
                lagged_valid_value, dynamic_step_values = xs

                def _do_reuse(args):
                    dynamic_values, bar = args
                    reverse_payload = _payload_from_dynamic_values(dynamic_values, True)
                    if cotangent_contract == "forward-like-v1":
                        return _radau_accepted_step_forward_like_pullback(
                            execution_context.kernel_context,
                            execution_context.physics_context,
                            carry_template,
                            execution_context.attempt_context,
                            reverse_payload,
                            bar,
                        )
                    if cotangent_contract == "forward-like-v2-no-stage":
                        return _radau_accepted_step_forward_like_no_stage_pullback(
                            execution_context.kernel_context,
                            execution_context.physics_context,
                            carry_template,
                            execution_context.attempt_context,
                            reverse_payload,
                            bar,
                        )
                    if cotangent_contract == "forward-like-v3-cache-no-stage":
                        return _radau_accepted_step_forward_like_cache_no_stage_pullback(
                            execution_context.kernel_context,
                            execution_context.physics_context,
                            carry_template,
                            execution_context.attempt_context,
                            reverse_payload,
                            bar,
                        )
                    return _radau_accepted_step_primitive_pullback(
                        execution_context.kernel_context,
                        execution_context.physics_context,
                        carry_template,
                        execution_context.attempt_context,
                        reverse_payload,
                        bar,
                    )

                def _do_rebuild(args):
                    dynamic_values, bar = args
                    reverse_payload = _payload_from_dynamic_values(dynamic_values, False)
                    if cotangent_contract == "forward-like-v1":
                        return _radau_accepted_step_forward_like_pullback(
                            execution_context.kernel_context,
                            execution_context.physics_context,
                            carry_template,
                            execution_context.attempt_context,
                            reverse_payload,
                            bar,
                        )
                    if cotangent_contract == "forward-like-v2-no-stage":
                        return _radau_accepted_step_forward_like_no_stage_pullback(
                            execution_context.kernel_context,
                            execution_context.physics_context,
                            carry_template,
                            execution_context.attempt_context,
                            reverse_payload,
                            bar,
                        )
                    if cotangent_contract == "forward-like-v3-cache-no-stage":
                        return _radau_accepted_step_forward_like_cache_no_stage_pullback(
                            execution_context.kernel_context,
                            execution_context.physics_context,
                            carry_template,
                            execution_context.attempt_context,
                            reverse_payload,
                            bar,
                        )
                    return _radau_accepted_step_primitive_pullback(
                        execution_context.kernel_context,
                        execution_context.physics_context,
                        carry_template,
                        execution_context.attempt_context,
                        reverse_payload,
                        bar,
                    )

                next_bar = jax.lax.cond(
                    lagged_valid_value,
                    _do_reuse,
                    _do_rebuild,
                    operand=(dynamic_step_values, next_bar),
                )
                return next_bar, None

            final_bar, _ = jax.lax.scan(
                _scan_body,
                carry_bar,
                (lagged_valids, reversed_dynamic_values),
            )
            return final_bar

        return _segment_reverse_impl

    def _apply_one_step_pullback(reverse_payload, bar):
        if cotangent_contract == "forward-like-v1":
            return _radau_accepted_step_forward_like_pullback(
                execution_context.kernel_context,
                execution_context.physics_context,
                carry_template,
                execution_context.attempt_context,
                reverse_payload,
                bar,
            )
        if cotangent_contract == "forward-like-v2-no-stage":
            return _radau_accepted_step_forward_like_no_stage_pullback(
                execution_context.kernel_context,
                execution_context.physics_context,
                carry_template,
                execution_context.attempt_context,
                reverse_payload,
                bar,
            )
        if cotangent_contract == "forward-like-v3-cache-no-stage":
            return _radau_accepted_step_forward_like_cache_no_stage_pullback(
                execution_context.kernel_context,
                execution_context.physics_context,
                carry_template,
                execution_context.attempt_context,
                reverse_payload,
                bar,
            )
        return _radau_accepted_step_primitive_pullback(
            execution_context.kernel_context,
            execution_context.physics_context,
            carry_template,
            execution_context.attempt_context,
            reverse_payload,
            bar,
        )

    def _build_step_loop_reverse_fns(payload_template):
        payload_base_kwargs = {
            field.name: getattr(payload_template, field.name)
            for field in dataclasses.fields(_RadauAcceptedStepReversePayload)
        }

        def _payload_from_dynamic_values(dynamic_values, lagged_valid_value: bool):
            updated = dict(payload_base_kwargs)
            updated["lagged_response_valid_in"] = jnp.asarray(lagged_valid_value, dtype=jnp.bool_)
            for name, value in zip(payload_dynamic_field_names, dynamic_values):
                updated[name] = value
            return dataclasses.replace(payload_template, **updated)

        def _reuse_impl(dynamic_values, bar):
            reverse_payload = _payload_from_dynamic_values(dynamic_values, True)
            return _apply_one_step_pullback(reverse_payload, bar)

        def _rebuild_impl(dynamic_values, bar):
            reverse_payload = _payload_from_dynamic_values(dynamic_values, False)
            return _apply_one_step_pullback(reverse_payload, bar)

        if execution_mode == "jit":
            return jax.jit(_reuse_impl), jax.jit(_rebuild_impl)
        return _reuse_impl, _rebuild_impl

    def _reverse_segment_compiled(payload_rollout, carry_bar):
        nonlocal last_step_payload_report
        if last_step_payload_report is None:
            last_payload = jax.tree_util.tree_map(lambda x: x[-1], payload_rollout.reverse_payloads)
            last_step_payload_report = _payload_leaf_stats(_ablate_reverse_payload(last_payload, payload_ablation))

        reverse_payloads = _ablate_reverse_payload(payload_rollout.reverse_payloads, payload_ablation)
        step_count = int(payload_rollout.accepted_dts.shape[0])
        if step_count <= 0:
            return carry_bar
        payload_template = jax.tree_util.tree_map(lambda x: x[0], reverse_payloads)
        if reverse_compose_mode == "step-loop":
            reuse_fn, rebuild_fn = _build_step_loop_reverse_fns(payload_template)
            reversed_payloads = jax.tree_util.tree_map(lambda x: jnp.flip(x, axis=0), reverse_payloads)
            lagged_valids = np.asarray(
                jax.device_get(reversed_payloads.lagged_response_valid_in),
                dtype=bool,
            ).tolist()
            reversed_dynamic_values = tuple(getattr(reversed_payloads, name) for name in payload_dynamic_field_names)
            for step_idx, lagged_valid in enumerate(lagged_valids):
                dynamic_values = tuple(value[step_idx] for value in reversed_dynamic_values)
                if execution_mode == "jit":
                    carry_bar = reuse_fn(dynamic_values, carry_bar) if lagged_valid else rebuild_fn(dynamic_values, carry_bar)
                else:
                    with jax.disable_jit():
                        carry_bar = reuse_fn(dynamic_values, carry_bar) if lagged_valid else rebuild_fn(dynamic_values, carry_bar)
            return carry_bar
        fn = _build_reverse_segment_fn(payload_template, reverse_payloads)
        if execution_mode == "jit":
            fn = jax.jit(fn)
            return fn(carry_bar)
        with jax.disable_jit():
            return fn(carry_bar)

    def _reverse_only_once():
        nonlocal last_step_payload_report
        carry_bar = final_reverse_state
        reversed_ranges = list(reversed(segment_ranges))
        for range_idx, (start_idx, end_idx) in enumerate(reversed_ranges):
            segment_start_carry = _segment_start_carry(start_idx)
            dt_segment = _accepted_dt_slice(start_idx, end_idx)
            payload_rollout = _collect_segment_payloads_compiled(segment_start_carry, dt_segment)
            if reverse_probe_mode == "last-step-only":
                last_idx = int(payload_rollout.accepted_dts.shape[0]) - 1
                reverse_payload = jax.tree_util.tree_map(
                    lambda x, idx=last_idx: x[idx],
                    payload_rollout.reverse_payloads,
                )
                reverse_payload = _ablate_reverse_payload(reverse_payload, payload_ablation)
                if last_step_payload_report is None:
                    last_step_payload_report = _payload_leaf_stats(reverse_payload)
                payload_template = reverse_payload
                ablated_reverse_payloads = _ablate_reverse_payload(payload_rollout.reverse_payloads, payload_ablation)
                single_reverse_payloads = dataclasses.replace(
                    ablated_reverse_payloads,
                    **{
                        name: getattr(ablated_reverse_payloads, name)[last_idx : last_idx + 1]
                        for name in payload_dynamic_field_names + ("lagged_response_valid_in",)
                    },
                )
                one_step_fn = _build_reverse_segment_fn(payload_template, single_reverse_payloads)
                if execution_mode == "jit":
                    carry_bar = jax.jit(one_step_fn)(carry_bar)
                else:
                    with jax.disable_jit():
                        carry_bar = one_step_fn(carry_bar)
                break
            carry_bar = _reverse_segment_compiled(payload_rollout, carry_bar)
        if cotangent_contract == "forward-like-v1":
            metric_values = _forward_like_cotangent_metrics(carry_bar)
        elif cotangent_contract == "forward-like-v2-no-stage":
            metric_values = _forward_like_no_stage_cotangent_metrics(carry_bar)
        elif cotangent_contract == "forward-like-v3-cache-no-stage":
            metric_values = _forward_like_cache_no_stage_cotangent_metrics(carry_bar)
        else:
            metric_values = {
                "primitive_y_bar_max": _tree_max_abs(carry_bar.y_out),
                "primitive_dt_bar_abs": jnp.max(jnp.abs(jnp.asarray(carry_bar.dt_out, dtype=jnp.float64))),
                "primitive_prev_stages_bar_max": _tree_max_abs(carry_bar.prev_stages_out),
            }
        return {
            "accepted_count": jnp.asarray(accepted_count, dtype=jnp.int32),
            "segment_count": jnp.asarray(1 if reverse_probe_mode == "last-step-only" else len(segment_ranges), dtype=jnp.int32),
            "checkpoint_count": jnp.asarray(len(checkpoint_starts), dtype=jnp.int32),
            "segment_length": jnp.asarray(segment_length, dtype=jnp.int32),
            "primitive_y_bar_max": metric_values["primitive_y_bar_max"],
            "primitive_dt_bar_abs": metric_values["primitive_dt_bar_abs"],
            "primitive_prev_stages_bar_max": metric_values["primitive_prev_stages_bar_max"],
        }

    t0 = time.perf_counter()
    first = _reverse_only_once()
    jax.block_until_ready(first["primitive_y_bar_max"])
    compile_plus_execute_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    second = _reverse_only_once()
    jax.block_until_ready(second["primitive_y_bar_max"])
    execute_s = time.perf_counter() - t1

    result = {key: np.asarray(jax.device_get(value)).item() for key, value in second.items()}
    result["compile_plus_execute_s"] = compile_plus_execute_s
    result["execute_s"] = execute_s
    result["last_step_payload_report"] = last_step_payload_report
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark checkpointed accepted-step primitive reverse composition over several accepted steps."
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
        "--accepted-step-counts",
        default="1,2,4",
        help="Comma-separated accepted-step counts to benchmark.",
    )
    parser.add_argument(
        "--max-total-steps-multiplier",
        type=int,
        default=8,
        help="Use accepted_step_count * multiplier as the capped max_steps in the benchmark harness. Default: 8.",
    )
    parser.add_argument(
        "--bar-mode",
        default="y-only",
        choices=("y-only", "all-ones"),
        help="Cotangent pattern used for the reduced-output pullback.",
    )
    parser.add_argument(
        "--execution-mode",
        default="jit",
        choices=("eager", "jit"),
        help="Run the segmented reverse composition eagerly or under JIT. Default: jit.",
    )
    parser.add_argument(
        "--segment-length",
        type=int,
        default=8,
        help="Accepted-step segment length used for transient payload collection and reverse. Default: 8.",
    )
    parser.add_argument(
        "--checkpoint-count",
        type=int,
        default=0,
        help="Number of sparse accepted-step checkpoints to store. Use 0 for no stored checkpoints. Default: 0.",
    )
    parser.add_argument(
        "--reverse-probe-mode",
        default="full",
        choices=("full", "last-step-only"),
        help="Run the full segmented reverse or only the very first reverse step at the end of the rollout. Default: full.",
    )
    parser.add_argument(
        "--payload-ablation",
        default="none",
        choices=("none", "stage", "lagged", "jacobian", "lu", "pivots"),
        help="Zero one reverse-payload family before the reverse probe. Default: none.",
    )
    parser.add_argument(
        "--bar-ablation",
        default="none",
        choices=("none", "prev-stages", "lagged-reference-y", "step-meta", "non-y"),
        help="Zero one reduced-output-bar family after each reverse step. Default: none.",
    )
    parser.add_argument(
        "--cotangent-contract",
        default="full",
        choices=("full", "forward-like-v1", "forward-like-v2-no-stage", "forward-like-v3-cache-no-stage"),
        help="Propagated reverse cotangent contract. `forward-like-v1` keeps a smaller forward-like state; `forward-like-v2-no-stage` also removes the propagated stage-history lane; `forward-like-v3-cache-no-stage` follows the forward lagged-cache boundary more closely. Default: full.",
    )
    parser.add_argument(
        "--reverse-compose-mode",
        default="segment-scan",
        choices=("segment-scan", "step-loop"),
        help="How to compose accepted-step reverse calls within each segment. `segment-scan` uses one jitted scan kernel; `step-loop` reuses the one-step reverse kernel step-by-step. Default: segment-scan.",
    )
    args = parser.parse_args()

    parameter_names = _parse_parameter_subset(args.parameters)
    accepted_step_counts = _parse_step_counts(args.accepted_step_counts)
    config = _prepare_benchmark_config(
        Path(args.config),
        device=args.device,
        ntx_exact_derivative_mode=args.ntx_exact_derivative_mode,
    )
    runtime, baseline_state = build_runtime_context(config)
    profile_cfg = _baseline_profile_cfg(config)
    baseline_vector = jnp.asarray([float(profile_cfg[name]) for name in parameter_names], dtype=jnp.float64)

    prefix_reports = []
    for accepted_step_count in accepted_step_counts:
        prefix_config = _prefix_transport_config(
            config,
            accepted_step_limit=accepted_step_count,
            max_total_steps_multiplier=args.max_total_steps_multiplier,
        )
        (
            execution_context,
            _prepared_rollout,
            initial_carry,
            max_total_steps,
            stop_after_accepted_steps,
            _solver,
            _solve_vector_field,
        ) = _prepare_realized_schedule_profile_vector_rollout_option_a(
            baseline_vector,
            config=prefix_config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_names=parameter_names,
            accepted_step_limit_override=accepted_step_count,
        )
        result = _compute_multi_step_metrics(
            execution_context,
            initial_carry,
            max_total_steps=max_total_steps,
            stop_after_accepted_steps=stop_after_accepted_steps,
            bar_mode=args.bar_mode,
            execution_mode=args.execution_mode,
            segment_length=args.segment_length,
            checkpoint_count=args.checkpoint_count,
            reverse_probe_mode=args.reverse_probe_mode,
            payload_ablation=args.payload_ablation,
            bar_ablation=args.bar_ablation,
            cotangent_contract=args.cotangent_contract,
            reverse_compose_mode=args.reverse_compose_mode,
        )
        prefix_reports.append(
            {
                "accepted_step_count": int(accepted_step_count),
                "max_total_steps": int(max_total_steps),
                "result": result,
            }
        )

    report = {
        "config": str(args.config),
        "device": args.device,
        "ntx_exact_derivative_mode": args.ntx_exact_derivative_mode,
        "parameter_names": list(parameter_names),
        "parameter_values": [float(x) for x in np.asarray(jax.device_get(baseline_vector), dtype=float)],
        "accepted_step_counts": [int(x) for x in accepted_step_counts],
        "max_total_steps_multiplier": int(args.max_total_steps_multiplier),
        "bar_mode": args.bar_mode,
        "execution_mode": args.execution_mode,
        "segment_length": int(args.segment_length),
        "checkpoint_count": int(args.checkpoint_count),
        "reverse_probe_mode": args.reverse_probe_mode,
        "payload_ablation": args.payload_ablation,
        "bar_ablation": args.bar_ablation,
        "cotangent_contract": args.cotangent_contract,
        "reverse_compose_mode": args.reverse_compose_mode,
        "prefix_reports": prefix_reports,
    }

    outpath = _report_path()
    outpath.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("[autodiff-gate] mode=transport_reverse_multi_step_primitive")
    print(f"[autodiff-gate] parameters={list(parameter_names)}")
    print(
        f"[autodiff-gate] accepted_step_counts={list(accepted_step_counts)} "
        f"bar_mode={args.bar_mode} execution_mode={args.execution_mode} "
        f"max_total_steps_multiplier={int(args.max_total_steps_multiplier)} "
        f"segment_length={int(args.segment_length)} checkpoint_count={int(args.checkpoint_count)} "
        f"reverse_probe_mode={args.reverse_probe_mode} payload_ablation={args.payload_ablation} "
        f"bar_ablation={args.bar_ablation} cotangent_contract={args.cotangent_contract} "
        f"reverse_compose_mode={args.reverse_compose_mode}"
    )
    for prefix in prefix_reports:
        result = prefix["result"]
        print(
            f"  - accepted_step_count={int(prefix['accepted_step_count'])} "
            f"accepted_count={int(result['accepted_count'])} "
            f"max_total_steps={int(prefix['max_total_steps'])} "
            f"segment_count={int(result['segment_count'])} "
            f"checkpoint_count={int(result['checkpoint_count'])}"
        )
        print(f"    primitive_y_bar_max={float(result['primitive_y_bar_max']):.6e}")
        print(f"    primitive_dt_bar_abs={float(result['primitive_dt_bar_abs']):.6e}")
        print(f"    primitive_prev_stages_bar_max={float(result['primitive_prev_stages_bar_max']):.6e}")
        print(f"    compile_plus_execute_s={float(result['compile_plus_execute_s']):.6e}")
        print(f"    execute_s={float(result['execute_s']):.6e}")
    print(f"[autodiff-gate] wrote={outpath}")


if __name__ == "__main__":
    main()
