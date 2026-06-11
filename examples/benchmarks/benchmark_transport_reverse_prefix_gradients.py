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
    OBJECTIVE_LABELS,
    PROFILE_VECTOR_PARAMETERS,
    _baseline_profile_cfg,
    _extract_fixed_temperature_projection,
    _extract_state_regularization,
    _initial_carry_from_state_with_static_setup,
    _objective_vector_from_final_y,
    _parameterized_initial_state_multi,
    _prepare_benchmark_config,
    _prepare_realized_schedule_profile_vector_rollout_option_a,
)
from NEOPAX._orchestrator import build_runtime_context  # noqa: E402
from NEOPAX._transport_solvers import (  # noqa: E402
    _RadauAcceptedPrimitivePayloadRollout,
    _RadauAcceptedStepForwardLikeCacheNoStageCotangent,
    _RadauAcceptedStepReversePayload,
    _RadauAcceptedStepReducedOutput,
    _radau_accepted_step_forward_like_cache_no_stage_pullback,
    _radau_accepted_step_primitive,
    _radau_adaptive_schedule_rollout,
    _radau_carry_with_forward_only_jvp_fields,
    _radau_collect_realized_accepted_step_payloads,
    _radau_contract_reduced_output_bar,
    _radau_forward_like_cache_no_stage_cotangent_from_reduced_output_bar,
    _radau_reduced_output_from_carry,
    _radau_replay_realized_accepted_rollout,
    _make_solver_state_transform,
)


def _report_path() -> Path:
    outdir = ROOT / "outputs" / "autodiff_transport_lagged_ntx" / "reverse_prefix_gradients"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / "transport_reverse_prefix_gradients_summary.json"


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


def _parse_step_counts(text: str) -> tuple[int | None, ...]:
    raw_items = tuple(item.strip() for item in str(text).split(",") if item.strip())
    if not raw_items:
        raise ValueError("At least one accepted-step prefix or `full` must be provided.")
    values: list[int | None] = []
    for item in raw_items:
        if item.lower() in {"full", "uncapped", "none"}:
            values.append(None)
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("Accepted-step prefixes must be positive integers.")
        values.append(value)
    return tuple(values)


def _make_final_output_bar(
    final_carry,
    final_y_bar,
) -> _RadauAcceptedStepReducedOutput:
    return _RadauAcceptedStepReducedOutput(
        t_out=jnp.zeros_like(final_carry.t),
        y_out=final_y_bar,
        dt_out=jnp.zeros_like(final_carry.dt),
        prev_stages_out=jnp.zeros_like(final_carry.prev_stages),
        prev_dt_out=jnp.zeros_like(final_carry.prev_dt),
        lagged_reference_y_out=jnp.zeros_like(final_carry.lagged_reference_y),
        prev_theta_final_out=jnp.zeros_like(final_carry.prev_theta_final),
    )


def _prefix_transport_config(
    config: dict,
    *,
    accepted_step_limit: int | None,
    max_total_steps_multiplier: int,
) -> dict:
    tuned = copy.deepcopy(config)
    solver_cfg = tuned.setdefault("transport_solver", {})
    if accepted_step_limit is None:
        solver_cfg.pop("stop_after_accepted_steps", None)
    else:
        solver_cfg["stop_after_accepted_steps"] = int(accepted_step_limit)
        solver_cfg["max_steps"] = max(
            int(accepted_step_limit),
            int(accepted_step_limit) * int(max_total_steps_multiplier),
        )
    return tuned


def _objective_basis(count: int, index: int, dtype) -> jax.Array:
    return jnp.asarray(np.eye(count, dtype=np.float64)[index], dtype=dtype)


def _tree_vdot(lhs, rhs):
    def _leaf_vdot(a, b):
        a_arr = jnp.asarray(a)
        b_arr = jnp.asarray(b)
        if not jnp.issubdtype(a_arr.dtype, jnp.inexact):
            return jnp.asarray(0.0, dtype=jnp.float64)
        if not jnp.issubdtype(b_arr.dtype, jnp.inexact):
            return jnp.asarray(0.0, dtype=jnp.float64)
        return jnp.asarray(
            jnp.vdot(jnp.ravel(a_arr), jnp.ravel(b_arr)),
            dtype=jnp.float64,
        )

    leaves = jax.tree_util.tree_map(
        _leaf_vdot,
        lhs,
        rhs,
    )
    return jax.tree_util.tree_reduce(
        lambda acc, x: acc + x,
        leaves,
        initializer=jnp.asarray(0.0, dtype=jnp.float64),
    )


def _inexact_leaf_arrays(tree) -> list[jax.Array]:
    arrays: list[jax.Array] = []
    for leaf in jax.tree_util.tree_leaves(tree):
        arr = jnp.asarray(leaf)
        if jnp.issubdtype(arr.dtype, jnp.inexact):
            arrays.append(jnp.ravel(jnp.asarray(arr, dtype=jnp.float64)))
    return arrays


def _tree_diff_metrics(lhs, rhs) -> dict[str, float]:
    lhs_leaves = _inexact_leaf_arrays(lhs)
    rhs_leaves = _inexact_leaf_arrays(rhs)
    if not lhs_leaves:
        return {"abs_max": 0.0, "abs_l2": 0.0, "rel_max": 0.0}
    lhs_flat = jnp.concatenate(lhs_leaves)
    rhs_flat = jnp.concatenate(rhs_leaves)
    diff = lhs_flat - rhs_flat
    denom = jnp.maximum(jnp.abs(rhs_flat), jnp.asarray(1.0e-30, dtype=rhs_flat.dtype))
    return {
        "abs_max": float(jax.device_get(jnp.max(jnp.abs(diff)))),
        "abs_l2": float(jax.device_get(jnp.linalg.norm(diff))),
        "rel_max": float(jax.device_get(jnp.max(jnp.abs(diff) / denom))),
    }


def _tree_finite_stats(tree) -> dict[str, object]:
    nonfinite_leaf_names: list[str] = []
    max_abs = 0.0
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        arr = jnp.asarray(leaf)
        if not jnp.issubdtype(arr.dtype, jnp.inexact):
            continue
        if not bool(jax.device_get(jnp.all(jnp.isfinite(arr)))):
            nonfinite_leaf_names.append("/".join(str(entry) for entry in path))
            continue
        if arr.size:
            max_abs = max(max_abs, float(jax.device_get(jnp.max(jnp.abs(arr)))))
    return {
        "all_finite": len(nonfinite_leaf_names) == 0,
        "nonfinite_leaf_names": nonfinite_leaf_names,
        "max_abs": max_abs,
    }


def _prepare_prefix_context(
    baseline_vector,
    *,
    config,
    runtime,
    baseline_state,
    profile_cfg,
    parameter_names: tuple[str, ...],
    accepted_step_limit: int | None,
    max_total_steps_multiplier: int,
):
    prefix_config = _prefix_transport_config(
        config,
        accepted_step_limit=accepted_step_limit,
        max_total_steps_multiplier=max_total_steps_multiplier,
    )
    (
        execution_context,
        prepared_rollout,
        initial_carry,
        max_total_steps,
        stop_after_accepted_steps,
        solver,
        solve_vector_field,
    ) = _prepare_realized_schedule_profile_vector_rollout_option_a(
        baseline_vector,
        config=prefix_config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_names=parameter_names,
        accepted_step_limit_override=accepted_step_limit,
    )
    return {
        "prefix_config": prefix_config,
        "execution_context": execution_context,
        "prepared_rollout": prepared_rollout,
        "initial_carry": initial_carry,
        "max_total_steps": max_total_steps,
        "stop_after_accepted_steps": stop_after_accepted_steps,
        "solver": solver,
        "solve_vector_field": solve_vector_field,
    }


def _initial_flat_state_from_params(
    params,
    *,
    runtime,
    baseline_state,
    profile_cfg,
    parameter_names: tuple[str, ...],
    solve_vector_field,
):
    state0 = _parameterized_initial_state_multi(
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        geometry=runtime.geometry,
        n_species=runtime.species.number_species,
        parameter_names=parameter_names,
        parameter_values=params,
    )
    temperature_active_mask, fixed_temperature_profile = _extract_fixed_temperature_projection(
        solve_vector_field
    )
    density_floor, temperature_floor = _extract_state_regularization(solve_vector_field)
    flat_state0, _unpack_flat, _unpack_packed, _pack_state, _project_flat, _project_flat_pullback, _unpack_flat_pullback = _make_solver_state_transform(
        state0,
        runtime.species,
        temperature_active_mask=temperature_active_mask,
        fixed_temperature_profile=fixed_temperature_profile,
        density_floor=density_floor,
        temperature_floor=temperature_floor,
    )
    return flat_state0


def _initial_forward_like_cache_no_stage_cotangent_to_flat_state(
    cotangent: _RadauAcceptedStepForwardLikeCacheNoStageCotangent,
    *,
    execution_context,
    initial_carry,
):
    # Keep dt_bar in the reduced initialization contract to match the forward-like
    # active lanes, even though the current prepared initial dt is solver-static
    # and does not feed back to the profile parameters here.
    dt_bar = jnp.asarray(cotangent.dt)
    flat_state_bar = jnp.asarray(cotangent.y)
    lagged_cache_bar = cotangent.lagged_response_cache
    build_pullback = execution_context.physics_context.build_lagged_response_pullback
    if (lagged_cache_bar is not None) and callable(build_pullback):
        flat_state_bar = flat_state_bar + build_pullback(
            initial_carry.y,
            lagged_cache_bar,
        )
    return flat_state_bar, dt_bar


def _payload_field_names_for_contract() -> tuple[str, ...]:
    excluded = {
        "accepted_y",
        "prev_dt_in",
        "prev_theta_final_in",
        "prev_newton_iter_count_in",
        "lagged_response_valid_in",
    }
    return tuple(
        field.name
        for field in dataclasses.fields(_RadauAcceptedStepReversePayload)
        if field.name not in excluded
    )


def _compute_forward_reference(
    baseline_vector,
    *,
    runtime,
    baseline_state,
    profile_cfg,
    parameter_names: tuple[str, ...],
    prefix_context: dict[str, object],
    compute_jacobian: bool,
):
    execution_context = prefix_context["execution_context"]
    prepared_rollout = prefix_context["prepared_rollout"]
    initial_carry = prefix_context["initial_carry"]
    solver = prefix_context["solver"]
    solve_vector_field = prefix_context["solve_vector_field"]
    payload_rollout = _radau_collect_realized_accepted_step_payloads(
        execution_context,
        initial_carry,
        max_total_steps=prefix_context["max_total_steps"],
        stop_after_accepted_steps=prefix_context["stop_after_accepted_steps"],
    )

    accepted_active_mask = payload_rollout.accepted_mask
    accepted_dts = payload_rollout.accepted_dts
    next_dts = payload_rollout.reduced_outputs.dt_out
    zero_i32 = jnp.zeros_like(accepted_active_mask, dtype=jnp.int32)
    next_lagged_response_valid = jnp.full_like(
        accepted_active_mask,
        bool(execution_context.attempt_context.use_transport_lagged_response),
        dtype=jnp.bool_,
    )

    def _objective_fn(params):
        state0 = _parameterized_initial_state_multi(
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            geometry=runtime.geometry,
            n_species=runtime.species.number_species,
            parameter_names=parameter_names,
            parameter_values=params,
        )
        carry0 = _initial_carry_from_state_with_static_setup(
            solver=solver,
            state=state0,
            solve_vector_field=solve_vector_field,
            species=runtime.species,
            prepared_rollout_static=prepared_rollout,
        )
        replay = _radau_replay_realized_accepted_rollout(
            execution_context,
            carry0,
            accepted_active_mask,
            accepted_dts,
            next_dts,
            zero_i32,
            zero_i32,
            zero_i32,
            next_lagged_response_valid,
        )
        return _objective_vector_from_final_y(
            replay.final_y,
            prepared_rollout=prepared_rollout,
            runtime=runtime,
        )

    t0 = time.perf_counter()
    objective_values = _objective_fn(baseline_vector)
    jacobian = None
    if compute_jacobian:
        jacobian = jax.jacfwd(_objective_fn)(baseline_vector)
        jax.block_until_ready(jacobian)
    else:
        jax.block_until_ready(objective_values)
    elapsed_s = time.perf_counter() - t0
    return objective_values, jacobian, elapsed_s, payload_rollout


def _compute_reverse_candidate(
    baseline_vector,
    *,
    runtime,
    baseline_state,
    profile_cfg,
    parameter_names: tuple[str, ...],
    prefix_context: dict[str, object],
    execution_mode: str,
    segment_length: int,
):
    execution_context = prefix_context["execution_context"]
    prepared_rollout = prefix_context["prepared_rollout"]
    initial_carry = prefix_context["initial_carry"]
    solve_vector_field = prefix_context["solve_vector_field"]
    schedule_rollout = _radau_adaptive_schedule_rollout(
        execution_context,
        initial_carry,
        max_total_steps=prefix_context["max_total_steps"],
        stop_after_accepted_steps=prefix_context["stop_after_accepted_steps"],
    )
    objective_values, final_y_pullback = jax.vjp(
        lambda final_y: _objective_vector_from_final_y(
            final_y,
            prepared_rollout=prepared_rollout,
            runtime=runtime,
        ),
        schedule_rollout.final_carry.y,
    )
    _, initial_flat_state_pullback = jax.vjp(
        lambda params: _initial_flat_state_from_params(
            params,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_names=parameter_names,
            solve_vector_field=solve_vector_field,
        ),
        baseline_vector,
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
    segment_length = max(1, int(segment_length))
    segment_ranges = [
        (start_idx, min(start_idx + segment_length, accepted_count))
        for start_idx in range(0, accepted_count, segment_length)
    ]
    accepted_dts_host = tuple(float(x) for x in accepted_dts_np.tolist())
    next_dts_host = tuple(float(x) for x in next_dts_np.tolist())
    next_recent_reject_count_host = tuple(int(x) for x in next_recent_reject_count_np.tolist())
    next_regrowth_cooldown_host = tuple(int(x) for x in next_regrowth_cooldown_np.tolist())
    next_easy_growth_streak_host = tuple(int(x) for x in next_easy_growth_streak_np.tolist())
    next_lagged_response_valid_host = tuple(bool(x) for x in next_lagged_response_valid_np.tolist())
    dtype = execution_context.dtype
    replay_cache = {}
    payload_collect_cache = {}
    payload_dynamic_field_names = _payload_field_names_for_contract()

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

    def _segment_start_carry(segment_start: int):
        if segment_start == 0:
            return initial_carry
        return _replay_from_carry(initial_carry, 0, segment_start)

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
            return _radau_accepted_step_forward_like_cache_no_stage_pullback(
                execution_context.kernel_context,
                execution_context.physics_context,
                initial_carry,
                execution_context.attempt_context,
                reverse_payload,
                bar,
            )

        def _rebuild_impl(dynamic_values, bar):
            reverse_payload = _payload_from_dynamic_values(dynamic_values, False)
            return _radau_accepted_step_forward_like_cache_no_stage_pullback(
                execution_context.kernel_context,
                execution_context.physics_context,
                initial_carry,
                execution_context.attempt_context,
                reverse_payload,
                bar,
            )

        if execution_mode == "jit":
            return jax.jit(_reuse_impl), jax.jit(_rebuild_impl)
        return _reuse_impl, _rebuild_impl

    jacobian_rows = []
    compile_plus_execute_s = None
    reverse_start_s = time.perf_counter()
    objective_count = int(objective_values.shape[0])
    first_nonfinite_debug = None
    for objective_index in range(objective_count):
        basis = _objective_basis(objective_count, objective_index, objective_values.dtype)
        (final_y_bar,) = final_y_pullback(basis)
        final_output_bar = _radau_contract_reduced_output_bar(
            _make_final_output_bar(schedule_rollout.final_carry, final_y_bar),
            "forward-like-v3-cache-no-stage",
        )
        call_start_s = time.perf_counter()
        carry_bar = _radau_forward_like_cache_no_stage_cotangent_from_reduced_output_bar(
            final_output_bar,
            initial_carry,
        )
        for start_idx, end_idx in reversed(segment_ranges):
            segment_start_carry = _segment_start_carry(start_idx)
            payload_rollout = _collect_segment_payloads_compiled(
                segment_start_carry,
                _accepted_dt_slice(start_idx, end_idx),
            )
            payload_template = jax.tree_util.tree_map(lambda x: x[0], payload_rollout.reverse_payloads)
            reuse_fn, rebuild_fn = _build_step_loop_reverse_fns(payload_template)
            reversed_payloads = jax.tree_util.tree_map(lambda x: jnp.flip(x, axis=0), payload_rollout.reverse_payloads)
            lagged_valids = np.asarray(
                jax.device_get(reversed_payloads.lagged_response_valid_in),
                dtype=bool,
            ).tolist()
            reversed_dynamic_values = tuple(getattr(reversed_payloads, name) for name in payload_dynamic_field_names)
            for step_idx, lagged_valid in enumerate(lagged_valids):
                dynamic_values = tuple(
                    jax.tree_util.tree_map(lambda x, idx=step_idx: x[idx], value)
                    for value in reversed_dynamic_values
                )
                carry_bar = reuse_fn(dynamic_values, carry_bar) if lagged_valid else rebuild_fn(dynamic_values, carry_bar)
        jax.block_until_ready(carry_bar.y)
        call_elapsed_s = time.perf_counter() - call_start_s
        if compile_plus_execute_s is None:
            compile_plus_execute_s = call_elapsed_s
        flat_state_bar, dt_bar = _initial_forward_like_cache_no_stage_cotangent_to_flat_state(
            carry_bar,
            execution_context=execution_context,
            initial_carry=initial_carry,
        )
        # Match the current forward initialization philosophy: keep dt in the
        # reduced contract, but do not force a parameter pullback lane for it
        # until initial dt becomes parameter-sensitive in the same way.
        del dt_bar
        (parameter_bar,) = initial_flat_state_pullback(flat_state_bar)
        if first_nonfinite_debug is None:
            carry_bar_stats = _tree_finite_stats(carry_bar)
            flat_state_bar_stats = _tree_finite_stats(flat_state_bar)
            parameter_bar_stats = _tree_finite_stats(parameter_bar)
            if not (
                carry_bar_stats["all_finite"]
                and flat_state_bar_stats["all_finite"]
                and parameter_bar_stats["all_finite"]
            ):
                first_nonfinite_debug = {
                    "objective_index": int(objective_index),
                    "objective_label": OBJECTIVE_LABELS[objective_index],
                    "carry_bar_stats": carry_bar_stats,
                    "flat_state_bar_stats": flat_state_bar_stats,
                    "parameter_bar_stats": parameter_bar_stats,
                }
        jacobian_rows.append(parameter_bar)
    total_reverse_s = time.perf_counter() - reverse_start_s
    jacobian = jnp.stack(jacobian_rows, axis=0)
    jax.block_until_ready(jacobian)
    return {
        "objective_values": objective_values,
        "jacobian": jacobian,
        "compile_plus_execute_s": float(compile_plus_execute_s or 0.0),
        "total_reverse_s": float(total_reverse_s),
        "accepted_count": accepted_count,
        "jacobian_finite_stats": _tree_finite_stats(jacobian),
        "first_nonfinite_debug": first_nonfinite_debug,
    }


def _per_objective_gradient_report(
    objective_labels: list[str],
    parameter_names: tuple[str, ...],
    forward_jacobian: np.ndarray,
    reverse_jacobian: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for objective_index, objective_label in enumerate(objective_labels):
        forward_row = forward_jacobian[objective_index]
        reverse_row = reverse_jacobian[objective_index]
        parameter_entries = []
        for parameter_index, parameter_name in enumerate(parameter_names):
            fwd = float(forward_row[parameter_index])
            rev = float(reverse_row[parameter_index])
            abs_diff = abs(rev - fwd)
            rel_diff = abs_diff / max(abs(fwd), 1.0e-30)
            parameter_entries.append(
                {
                    "parameter": parameter_name,
                    "forward": fwd,
                    "reverse": rev,
                    "abs_diff": abs_diff,
                    "rel_diff": rel_diff,
                }
            )
        rows.append(
            {
                "objective": objective_label,
                "parameters": parameter_entries,
                "max_abs_diff": max(entry["abs_diff"] for entry in parameter_entries),
                "max_rel_diff": max(entry["rel_diff"] for entry in parameter_entries),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print accepted-step reverse gradients, with optional comparison against the trusted forward AD path."
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="Benchmark TOML.")
    parser.add_argument("--device", type=str, default=None, help="Optional device override passed to config preparation.")
    parser.add_argument(
        "--ntx-exact-derivative-mode",
        default="direct",
        choices=("direct", "custom_vjp"),
        help="NTX exact-runtime derivative mode.",
    )
    parser.add_argument(
        "--parameters",
        default=",".join(PROFILE_VECTOR_PARAMETERS),
        help="Comma-separated subset of profile-vector parameters to include.",
    )
    parser.add_argument(
        "--accepted-step-counts",
        default="1,2,4",
        help="Comma-separated accepted-step prefixes to compare. Use `full` to run the solver's natural realized rollout.",
    )
    parser.add_argument(
        "--comparison-mode",
        default="reverse-only",
        choices=("reverse-only", "both"),
        help="Run reverse gradients alone or compare reverse against the forward accepted-step reference. Default: reverse-only.",
    )
    parser.add_argument(
        "--reverse-execution-mode",
        default="jit",
        choices=("eager", "jit"),
        help="Run the new reverse composition eagerly or under JIT. Default: jit.",
    )
    parser.add_argument(
        "--segment-length",
        type=int,
        default=8,
        help="Accepted-step segment length used by the segmented reverse path. Default: 8.",
    )
    parser.add_argument(
        "--max-total-steps-multiplier",
        type=int,
        default=8,
        help="Cap solver max_steps for each accepted-step prefix to accepted_step_limit * multiplier. Default: 8.",
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

    prefixes = []
    for accepted_step_limit in accepted_step_counts:
        prefix_context = _prepare_prefix_context(
            baseline_vector,
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_names=parameter_names,
            accepted_step_limit=accepted_step_limit,
            max_total_steps_multiplier=args.max_total_steps_multiplier,
        )
        forward_objectives = None
        forward_jacobian = None
        forward_elapsed_s = None
        if args.comparison_mode == "both":
            forward_objectives, forward_jacobian, forward_elapsed_s, _payload_rollout = _compute_forward_reference(
                baseline_vector,
                runtime=runtime,
                baseline_state=baseline_state,
                profile_cfg=profile_cfg,
                parameter_names=parameter_names,
                prefix_context=prefix_context,
                compute_jacobian=True,
            )
        reverse_result = _compute_reverse_candidate(
            baseline_vector,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_names=parameter_names,
            prefix_context=prefix_context,
            execution_mode=args.reverse_execution_mode,
            segment_length=args.segment_length,
        )

        reverse_objectives_np = np.asarray(jax.device_get(reverse_result["objective_values"]), dtype=float)
        reverse_jacobian_np = np.asarray(jax.device_get(reverse_result["jacobian"]), dtype=float)
        accepted_count = int(reverse_result["accepted_count"])
        forward_objectives_np = None
        forward_jacobian_np = None
        objective_metrics = None
        gradient_metrics = None
        gradient_rows = None
        if args.comparison_mode == "both":
            forward_objectives_np = np.asarray(jax.device_get(forward_objectives), dtype=float)
            forward_jacobian_np = np.asarray(jax.device_get(forward_jacobian), dtype=float)
            objective_metrics = _tree_diff_metrics(reverse_objectives_np, forward_objectives_np)
            gradient_metrics = _tree_diff_metrics(reverse_jacobian_np, forward_jacobian_np)
            gradient_rows = _per_objective_gradient_report(
                OBJECTIVE_LABELS[: int(forward_jacobian_np.shape[0])],
                parameter_names,
                forward_jacobian_np,
                reverse_jacobian_np,
            )

        prefixes.append(
            {
                "accepted_step_limit": (
                    None if accepted_step_limit is None else int(accepted_step_limit)
                ),
                "accepted_count": accepted_count,
                "forward_elapsed_s": (
                    None if forward_elapsed_s is None else float(forward_elapsed_s)
                ),
                "reverse_compile_plus_execute_s": float(reverse_result["compile_plus_execute_s"]),
                "reverse_total_s": float(reverse_result["total_reverse_s"]),
                "reverse_jacobian_finite_stats": reverse_result["jacobian_finite_stats"],
                "reverse_first_nonfinite_debug": reverse_result["first_nonfinite_debug"],
                "objective_values_forward": None if forward_objectives_np is None else forward_objectives_np.tolist(),
                "objective_values_reverse": reverse_objectives_np.tolist(),
                "objective_diff": objective_metrics,
                "forward_jacobian": None if forward_jacobian_np is None else forward_jacobian_np.tolist(),
                "reverse_jacobian": reverse_jacobian_np.tolist(),
                "gradient_diff": gradient_metrics,
                "per_objective_gradients": gradient_rows,
            }
        )

    report = {
        "config": str(args.config),
        "device": args.device,
        "ntx_exact_derivative_mode": args.ntx_exact_derivative_mode,
        "parameter_names": list(parameter_names),
        "parameter_values": [float(x) for x in np.asarray(jax.device_get(baseline_vector), dtype=float)],
        "accepted_step_counts": [None if x is None else int(x) for x in accepted_step_counts],
        "comparison_mode": args.comparison_mode,
        "reverse_execution_mode": args.reverse_execution_mode,
        "segment_length": int(args.segment_length),
        "max_total_steps_multiplier": int(args.max_total_steps_multiplier),
        "prefixes": prefixes,
    }

    outpath = _report_path()
    outpath.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("[autodiff-gate] mode=transport_reverse_prefix_gradients")
    print(f"[autodiff-gate] parameters={list(parameter_names)}")
    print(
        f"[autodiff-gate] accepted_step_counts={list(accepted_step_counts)} "
        f"comparison_mode={args.comparison_mode} "
        f"reverse_execution_mode={args.reverse_execution_mode} "
        f"segment_length={int(args.segment_length)} "
        f"max_total_steps_multiplier={int(args.max_total_steps_multiplier)}"
    )
    for prefix in prefixes:
        print(
            f"  - accepted_step_limit={prefix['accepted_step_limit']} "
            f"accepted_count={int(prefix['accepted_count'])} "
            f"reverse_compile_plus_execute_s={float(prefix['reverse_compile_plus_execute_s']):.6e} "
            f"reverse_total_s={float(prefix['reverse_total_s']):.6e}"
        )
        finite_stats = prefix.get("reverse_jacobian_finite_stats")
        if isinstance(finite_stats, dict) and not bool(finite_stats.get("all_finite", True)):
            print(
                "    reverse_jacobian_nonfinite_leaves="
                f"{finite_stats.get('nonfinite_leaf_names', [])}"
            )
        first_nonfinite_debug = prefix.get("reverse_first_nonfinite_debug")
        if isinstance(first_nonfinite_debug, dict) and first_nonfinite_debug:
            print(
                "    first_nonfinite_debug "
                f"objective={first_nonfinite_debug.get('objective_label')} "
                f"carry_all_finite={first_nonfinite_debug.get('carry_bar_stats', {}).get('all_finite')} "
                f"flat_state_all_finite={first_nonfinite_debug.get('flat_state_bar_stats', {}).get('all_finite')} "
                f"parameter_all_finite={first_nonfinite_debug.get('parameter_bar_stats', {}).get('all_finite')}"
            )
        reverse_jacobian = np.asarray(prefix["reverse_jacobian"], dtype=float)
        for objective_index, objective_label in enumerate(OBJECTIVE_LABELS[: int(reverse_jacobian.shape[0])]):
            row = reverse_jacobian[objective_index]
            row_text = " ".join(
                f"{parameter_names[param_index]}={float(row[param_index]):.6e}"
                for param_index in range(len(parameter_names))
            )
            print(f"    reverse_gradient[{objective_label}] {row_text}")
        if prefix["objective_diff"] is not None and prefix["gradient_diff"] is not None:
            print(
                f"    objective_abs_max={float(prefix['objective_diff']['abs_max']):.6e} "
                f"gradient_abs_max={float(prefix['gradient_diff']['abs_max']):.6e} "
                f"gradient_rel_max={float(prefix['gradient_diff']['rel_max']):.6e}"
            )
    print(f"[autodiff-gate] wrote={outpath}")


if __name__ == "__main__":
    main()
