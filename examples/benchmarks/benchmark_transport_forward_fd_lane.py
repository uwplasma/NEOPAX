from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any

import jax

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_transport_autodiff_lagged_ntx import (  # noqa: E402
    ALLOWED_PARAMETERS,
    DEFAULT_CONFIG,
    OBJECTIVE_LABELS,
    _adaptive_rollout_diagnostics,
    _baseline_profile_cfg,
    _fd_step,
    _objective_vector,
    _parameterized_initial_state,
    _prepare_benchmark_config,
    _truncate_rollout_trace_by_accepted_steps,
)
from NEOPAX._orchestrator import prepare_transport_solver_components  # noqa: E402
from NEOPAX._transport_solvers import (  # noqa: E402
    _build_prepared_radau_accepted_rollout,
    _build_prepared_radau_execution_context,
    _extract_fixed_temperature_projection,
    _extract_state_regularization,
    _make_radau_initial_step_state,
    _make_solver_state_transform,
    _project_flat_state_if_needed,
    _radau_adaptive_final_state_rollout,
    _radau_adaptive_final_y_realized_schedule,
    _radau_adaptive_payload_trace_rollout,
    _radau_adaptive_schedule_rollout,
    _radau_carry_from_step_state,
    _radau_eval_rhs,
    _radau_forward_fd_run_prepared_on_realized_trace,
    _radau_forward_fd_run_prepared_on_time_list,
)


def _initial_carry_from_state_with_static_setup(
    *,
    solver,
    state,
    solve_vector_field,
    species,
    prepared_rollout_static,
):
    temperature_active_mask, fixed_temperature_profile = _extract_fixed_temperature_projection(solve_vector_field)
    density_floor, temperature_floor = _extract_state_regularization(solve_vector_field)
    flat_state0, unpack_flat, _unpack_packed, _pack_state, project_flat, _project_flat_pullback, _unpack_flat_pullback = _make_solver_state_transform(
        state,
        species,
        temperature_active_mask=temperature_active_mask,
        fixed_temperature_profile=fixed_temperature_profile,
        density_floor=density_floor,
        temperature_floor=temperature_floor,
    )
    kernel_context = prepared_rollout_static.kernel_context
    physics_context = prepared_rollout_static.physics_context
    initial_carry_static = prepared_rollout_static.initial_carry
    initial_lagged_response = (
        physics_context.build_lagged_response(
            unpack_flat(_project_flat_state_if_needed(flat_state0, project_flat))
        )
        if (kernel_context.use_transport_lagged_response and physics_context.build_lagged_response is not None)
        else None
    )
    initial_rhs = _radau_eval_rhs(
        initial_carry_static.t,
        flat_state0,
        initial_lagged_response,
        physics_context.flat_rhs,
        physics_context.flat_rhs_with_lagged_response,
    )
    step_state0 = _make_radau_initial_step_state(
        initial_carry_static.t,
        flat_state0,
        initial_carry_static.dt,
        kernel_context.dtype,
        initial_rhs,
        kernel_context.num_stages,
        initial_carry_static.real_lu,
        initial_carry_static.real_piv,
        initial_carry_static.complex_lu,
        initial_carry_static.complex_piv,
        initial_lagged_response,
        jax.numpy.asarray(kernel_context.use_transport_lagged_response),
        flat_state0,
    )
    return _radau_carry_from_step_state(step_state0)


def _adaptive_rollout_final_state_for_parameter(
    parameter_value,
    *,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    parameter_name: str,
    use_realized_schedule_jvp: bool = False,
    accepted_step_limit_override: int | None = None,
    use_schedule_trace_only: bool = False,
    use_payload_trace: bool = False,
):
    state0 = _parameterized_initial_state(
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        geometry=runtime.geometry,
        n_species=runtime.species.number_species,
        parameter_name=parameter_name,
        parameter_value=parameter_value,
    )
    if use_realized_schedule_jvp:
        state0_static = _parameterized_initial_state(
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            geometry=runtime.geometry,
            n_species=runtime.species.number_species,
            parameter_name=parameter_name,
            parameter_value=jax.lax.stop_gradient(parameter_value),
        )
        prepared_components_static = prepare_transport_solver_components(config, runtime, state0_static)
        solver = prepared_components_static["solver"]
        solve_vector_field_static = prepared_components_static["solve_vector_field"]
        prepared_rollout_static = _build_prepared_radau_accepted_rollout(
            solver=solver,
            state=state0_static,
            vector_field=solve_vector_field_static,
            species=runtime.species,
        )
        execution_context = _build_prepared_radau_execution_context(
            solver=solver,
            prepared_rollout=prepared_rollout_static,
        )
        initial_carry = _initial_carry_from_state_with_static_setup(
            solver=solver,
            state=state0,
            solve_vector_field=solve_vector_field_static,
            species=runtime.species,
            prepared_rollout_static=prepared_rollout_static,
        )
        max_total_steps = int(max(1, getattr(solver, "max_steps", 1)))
        stop_after_accepted_steps = (
            int(accepted_step_limit_override)
            if accepted_step_limit_override is not None
            else getattr(solver, "stop_after_accepted_steps", None)
        )
        final_y = _radau_adaptive_final_y_realized_schedule(
            execution_context,
            max_total_steps,
            stop_after_accepted_steps,
            initial_carry,
        )
        final_state = prepared_rollout_static.physics_context.unpack_flat(final_y)
        if use_payload_trace:
            rollout = _radau_adaptive_payload_trace_rollout(
                execution_context,
                initial_carry,
                max_total_steps=max_total_steps,
                stop_after_accepted_steps=stop_after_accepted_steps,
            )
        elif use_schedule_trace_only:
            rollout = _radau_adaptive_schedule_rollout(
                execution_context,
                initial_carry,
                max_total_steps=max_total_steps,
                stop_after_accepted_steps=stop_after_accepted_steps,
            )
        else:
            rollout = _radau_adaptive_final_state_rollout(
                execution_context,
                initial_carry,
                max_total_steps=max_total_steps,
                stop_after_accepted_steps=stop_after_accepted_steps,
            )
    else:
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
        max_total_steps = int(max(1, getattr(solver, "max_steps", 1)))
        stop_after_accepted_steps = (
            int(accepted_step_limit_override)
            if accepted_step_limit_override is not None
            else getattr(solver, "stop_after_accepted_steps", None)
        )
        rollout = (
            _radau_adaptive_payload_trace_rollout(
                execution_context,
                prepared_rollout.initial_carry,
                max_total_steps=max_total_steps,
                stop_after_accepted_steps=stop_after_accepted_steps,
            )
            if use_payload_trace
            else _radau_adaptive_schedule_rollout(
                execution_context,
                prepared_rollout.initial_carry,
                max_total_steps=max_total_steps,
                stop_after_accepted_steps=stop_after_accepted_steps,
            )
            if use_schedule_trace_only
            else _radau_adaptive_final_state_rollout(
                execution_context,
                prepared_rollout.initial_carry,
                max_total_steps=max_total_steps,
                stop_after_accepted_steps=stop_after_accepted_steps,
            )
        )
        final_state = prepared_rollout.physics_context.unpack_flat(rollout.final_carry.y)
    return final_state, rollout


def _adaptive_rollout_objectives_realized_schedule_only_for_parameter(
    parameter_value,
    *,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    parameter_name: str,
    accepted_step_limit_override: int | None = None,
    derivative_mode: str = "jvp",
):
    state0 = _parameterized_initial_state(
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        geometry=runtime.geometry,
        n_species=runtime.species.number_species,
        parameter_name=parameter_name,
        parameter_value=parameter_value,
    )
    state0_static = _parameterized_initial_state(
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        geometry=runtime.geometry,
        n_species=runtime.species.number_species,
        parameter_name=parameter_name,
        parameter_value=jax.lax.stop_gradient(parameter_value),
    )
    prepared_components_static = prepare_transport_solver_components(config, runtime, state0_static)
    solver = prepared_components_static["solver"]
    solve_vector_field_static = prepared_components_static["solve_vector_field"]
    prepared_rollout_static = _build_prepared_radau_accepted_rollout(
        solver=solver,
        state=state0_static,
        vector_field=solve_vector_field_static,
        species=runtime.species,
    )
    execution_context = _build_prepared_radau_execution_context(
        solver=solver,
        prepared_rollout=prepared_rollout_static,
    )
    initial_carry = _initial_carry_from_state_with_static_setup(
        solver=solver,
        state=state0,
        solve_vector_field=solve_vector_field_static,
        species=runtime.species,
        prepared_rollout_static=prepared_rollout_static,
    )
    max_total_steps = int(max(1, getattr(solver, "max_steps", 1)))
    stop_after_accepted_steps = (
        int(accepted_step_limit_override)
        if accepted_step_limit_override is not None
        else getattr(solver, "stop_after_accepted_steps", None)
    )
    derivative_mode_key = str(derivative_mode).strip().lower()
    if derivative_mode_key != "jvp":
        raise NotImplementedError("The scratch forward benchmark lane supports derivative_mode='jvp' only.")
    final_y = _radau_adaptive_final_y_realized_schedule(
        execution_context,
        max_total_steps,
        stop_after_accepted_steps,
        initial_carry,
    )
    final_state = prepared_rollout_static.physics_context.unpack_flat(final_y)
    return _objective_vector(final_state, runtime)


def _adaptive_rollout_objectives_for_parameter_on_frozen_trace(
    parameter_value,
    *,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    parameter_name: str,
    frozen_trace,
    replay_mode: str = "attempt",
):
    state0 = _parameterized_initial_state(
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        geometry=runtime.geometry,
        n_species=runtime.species.number_species,
        parameter_name=parameter_name,
        parameter_value=parameter_value,
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
    replay = _radau_forward_fd_run_prepared_on_realized_trace(
        prepared_rollout,
        execution_context,
        frozen_trace,
        replay_mode=replay_mode,
    )
    return _objective_vector(replay["final_state"], runtime), replay


def _adaptive_rollout_objectives_for_parameter_on_time_list(
    parameter_value,
    *,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    parameter_name: str,
    time_list,
):
    state0 = _parameterized_initial_state(
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        geometry=runtime.geometry,
        n_species=runtime.species.number_species,
        parameter_name=parameter_name,
        parameter_value=parameter_value,
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
    replay = _radau_forward_fd_run_prepared_on_time_list(
        prepared_rollout,
        time_list,
    )
    return _objective_vector(replay["final_state"], runtime), replay
