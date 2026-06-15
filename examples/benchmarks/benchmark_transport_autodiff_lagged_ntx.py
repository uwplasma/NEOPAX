"""AD-vs-FD benchmark for the lagged exact-runtime NTX transport solve.

This harness keeps the magnetic geometry fixed and compares JAX automatic
differentiation against central finite differences for smooth final-state
transport objectives while varying one initial-profile parameter.

Default parameter choices are aimed at the standard analytical profile model:

- ``n0``
- ``T0``
- ``density_shape_power``
- ``temperature_shape_power``

Outputs:

- JSON summary
- CSV sweep
- PNG/PDF figure
"""

from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import json
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import NEOPAX
from NEOPAX._orchestrator import build_runtime_context, prepare_transport_solver_components, run_transport
from NEOPAX._profiles import AnalyticalProfileModel
from NEOPAX._transport_flux_models import PRESSURE_SOURCE_STATE_TO_MW_M3
from NEOPAX._transport_solvers import (
    _RadauAcceptedStepAttemptContext,
    _extract_fixed_temperature_projection,
    _extract_state_regularization,
    _make_solver_state_transform,
    _make_radau_initial_step_state,
    _project_flat_state_if_needed,
    _radau_adaptive_final_state_rollout,
    _radau_adaptive_final_y_realized_schedule,
    _radau_adaptive_final_y_realized_schedule_vjp,
    _radau_apply_accepted_step_map,
    _radau_carry_from_step_state,
    _radau_carry_with_forward_only_jvp_fields,
    _radau_debug_compare_zero_tangent_one_step,
    _radau_debug_realized_attempt_replay,
    _radau_dt_sequence_from_time_list,
    _radau_eval_rhs,
    _radau_prepare_lagged_response,
    _radau_stage_residual,
    _execute_radau_accepted_step_attempt,
    _execute_radau_accepted_step_attempt_autodiff,
    _build_prepared_radau_execution_context,
    _build_prepared_radau_accepted_rollout,
    _radau_controller_composed_rollout,
    _radau_controller_forward_only_rollout,
    _radau_prepare_stage_subsolve_inputs_from_carry,
    _radau_run_prepared_on_realized_trace,
    _radau_run_prepared_on_time_list,
    _radau_run_stage_subsolve_standalone_autodiff,
)


DEFAULT_CONFIG = Path(
    "examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_benchmark.toml"
)
ALLOWED_PARAMETERS = {"n0", "T0", "density_shape_power", "temperature_shape_power"}
PROFILE_VECTOR_PARAMETERS = ("n0", "T0", "density_shape_power", "temperature_shape_power")
OBJECTIVE_LABELS = [
    "softmax_Er",
    "smooth_root_proxy",
    "Er2_volume_average",
    "Er_volume_average",
    "electron_temperature_volume_average_keV",
    "total_pressure_volume_average",
    "alpha_power_volume_average_mw_m3",
]
DEFAULT_FD_SWEEP_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)
STANDALONE_SUBSOLVE_LABELS = [
    "stage_sum",
    "stage_l2_norm",
    "final_residual_norm",
    "theta_final",
]
DEFAULT_SMALL_STEP_COUNTS = (2, 3, 5)


def _prepare_benchmark_config(
    config_path: Path,
    *,
    device: str | None,
    ntx_exact_derivative_mode: str | None = None,
) -> dict[str, Any]:
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
    if ntx_exact_derivative_mode is not None:
        config.setdefault("neoclassical", {})["ntx_exact_derivative_mode"] = str(ntx_exact_derivative_mode)
    return config


def _apply_one_step_diagnostic_config(config: dict[str, Any]) -> dict[str, Any]:
    tuned = copy.deepcopy(config)
    solver_cfg = tuned.setdefault("transport_solver", {})
    solver_cfg["stop_after_accepted_steps"] = 1
    current_max_steps = int(solver_cfg.get("max_steps", 20000))
    solver_cfg["max_steps"] = max(current_max_steps, 20)
    return tuned


def _baseline_profile_cfg(config: dict[str, Any]) -> dict[str, Any]:
    profiles = copy.deepcopy(config.get("profiles", {}))
    profiles.setdefault("model", "standard_analytical")
    return profiles


def _parameterized_profile_set(
    profile_cfg: dict[str, Any],
    geometry,
    n_species: int,
    *,
    parameter_name: str,
    parameter_value,
):
    cfg = dict(profile_cfg)
    cfg[parameter_name] = parameter_value

    model = AnalyticalProfileModel(
        n0=cfg.get("n0", 4.21),
        n_edge=cfg.get("n_edge", 0.6),
        T0=cfg.get("T0", 17.8),
        T_edge=cfg.get("T_edge", 0.7),
        c_density=None if cfg.get("c_density") is None else tuple(cfg.get("c_density")),
        c_temperature=None if cfg.get("c_temperature") is None else tuple(cfg.get("c_temperature")),
        density_shape_power=cfg.get("density_shape_power", 2.0),
        temperature_shape_power=cfg.get("temperature_shape_power", 2.0),
        n_scale=cfg.get("n_scale", 1.0),
        T_scale=cfg.get("T_scale", 1.0),
        er0_scale=cfg.get("er0_scale", 100.0),
        er0_peak_rho=cfg.get("er0_peak_rho", 0.8),
        charge_qp=None if cfg.get("charge_qp") is None else tuple(cfg.get("charge_qp")),
    )
    return model.build(geometry, n_species)


def _parameterized_initial_state(
    *,
    baseline_state,
    profile_cfg: dict[str, Any],
    geometry,
    n_species: int,
    parameter_name: str,
    parameter_value,
):
    profile_set = _parameterized_profile_set(
        profile_cfg,
        geometry,
        n_species,
        parameter_name=parameter_name,
        parameter_value=parameter_value,
    )
    density_state = jnp.asarray(profile_set.density, dtype=baseline_state.density.dtype) / 1.0e20
    temperature_state = jnp.asarray(profile_set.temperature, dtype=baseline_state.pressure.dtype) / 1.0e3
    pressure_state = density_state * temperature_state
    return dataclasses.replace(
        baseline_state,
        density=density_state,
        pressure=pressure_state,
    )


def _parameterized_initial_state_multi(
    *,
    baseline_state,
    profile_cfg: dict[str, Any],
    geometry,
    n_species: int,
    parameter_names: tuple[str, ...],
    parameter_values,
):
    values = jnp.asarray(parameter_values)
    if int(values.size) != int(len(parameter_names)):
        raise ValueError(
            f"parameter_values must have length {len(parameter_names)} but got {int(values.size)}."
        )
    cfg = dict(profile_cfg)
    for idx, name in enumerate(parameter_names):
        cfg[name] = values[idx]
    profile_set = _parameterized_profile_set(
        cfg,
        geometry,
        n_species,
        parameter_name=parameter_names[0],
        parameter_value=cfg[parameter_names[0]],
    )
    density_state = jnp.asarray(profile_set.density, dtype=baseline_state.density.dtype) / 1.0e20
    temperature_state = jnp.asarray(profile_set.temperature, dtype=baseline_state.pressure.dtype) / 1.0e3
    pressure_state = density_state * temperature_state
    return dataclasses.replace(
        baseline_state,
        density=density_state,
        pressure=pressure_state,
    )


def _softmax_objective(er_profile: jax.Array, *, beta: float = 16.0) -> jax.Array:
    beta_arr = jnp.asarray(beta, dtype=er_profile.dtype)
    return jax.scipy.special.logsumexp(beta_arr * er_profile) / beta_arr


def _smooth_root_proxy(er_profile: jax.Array, rho_grid: jax.Array, *, beta: float = 24.0, eps: float = 1.0e-4):
    beta_arr = jnp.asarray(beta, dtype=er_profile.dtype)
    eps_arr = jnp.asarray(eps, dtype=er_profile.dtype)
    smooth_abs = jnp.sqrt(er_profile * er_profile + eps_arr * eps_arr)
    weights = jnp.exp(-beta_arr * smooth_abs)
    return jnp.sum(rho_grid * weights) / jnp.maximum(jnp.sum(weights), jnp.asarray(1.0e-30, dtype=er_profile.dtype))


def _volume_average(profile: jax.Array, geometry) -> jax.Array:
    volume = jnp.trapezoid(jnp.asarray(geometry.Vprime), x=jnp.asarray(geometry.r_grid))
    integral = jnp.trapezoid(profile * jnp.asarray(geometry.Vprime), x=jnp.asarray(geometry.r_grid))
    return integral / jnp.maximum(volume, jnp.asarray(1.0e-30, dtype=integral.dtype))


def _alpha_power_volume_average(final_state, runtime) -> jax.Array:
    source_models = runtime.models.source or {}
    pressure_source_model = source_models.get("temperature") if isinstance(source_models, dict) else None
    if pressure_source_model is None:
        return jnp.asarray(0.0, dtype=final_state.pressure.dtype)
    raw_sources = pressure_source_model(final_state)
    alpha_power = raw_sources.get("AlphaPower") if isinstance(raw_sources, dict) else None
    if alpha_power is None:
        return jnp.asarray(0.0, dtype=final_state.pressure.dtype)
    alpha_mw_m3 = PRESSURE_SOURCE_STATE_TO_MW_M3 * jnp.asarray(alpha_power, dtype=final_state.pressure.dtype)
    return _volume_average(alpha_mw_m3, runtime.geometry)


def _electron_temperature_volume_average(final_state, runtime) -> jax.Array:
    species_idx = getattr(runtime.species, "species_idx", {})
    electron_idx = species_idx.get("e", 0)
    temperature = jnp.asarray(final_state.temperature[electron_idx], dtype=final_state.pressure.dtype)
    return _volume_average(temperature, runtime.geometry)


def _total_pressure_volume_average(final_state, runtime) -> jax.Array:
    total_pressure = jnp.sum(jnp.asarray(final_state.pressure, dtype=final_state.pressure.dtype), axis=0)
    return _volume_average(total_pressure, runtime.geometry)


def _objective_vector(final_state, runtime) -> jax.Array:
    er = jnp.asarray(final_state.Er)
    rho = jnp.asarray(runtime.geometry.rho_grid, dtype=er.dtype)
    er2_vol = _volume_average(er * er, runtime.geometry)
    er_vol = _volume_average(er, runtime.geometry)
    te_vol = _electron_temperature_volume_average(final_state, runtime)
    p_tot_vol = _total_pressure_volume_average(final_state, runtime)
    alpha_vol = _alpha_power_volume_average(final_state, runtime)
    return jnp.stack(
        [
            _softmax_objective(er),
            _smooth_root_proxy(er, rho),
            er2_vol,
            er_vol,
            te_vol,
            p_tot_vol,
            alpha_vol,
        ]
    )


def _transport_objectives_for_parameter(
    parameter_value,
    *,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    parameter_name: str,
):
    state0 = _parameterized_initial_state(
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        geometry=runtime.geometry,
        n_species=runtime.species.number_species,
        parameter_name=parameter_name,
        parameter_value=parameter_value,
    )
    result = run_transport(config, runtime, state0)
    final_state = result["final_state"]
    return _objective_vector(final_state, runtime)


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
        prepared_components = prepare_transport_solver_components(config, runtime, state0)
        solve_vector_field = prepared_components["solve_vector_field"]
        prepared_rollout = _build_prepared_radau_accepted_rollout(
            solver=solver,
            state=state0,
            vector_field=solve_vector_field,
            species=runtime.species,
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
            prepared_rollout.initial_carry,
        )
        final_state = prepared_rollout.physics_context.unpack_flat(final_y)
        if use_payload_trace:
            rollout = _radau_adaptive_payload_trace_rollout(
                execution_context,
                prepared_rollout.initial_carry,
                max_total_steps=max_total_steps,
                stop_after_accepted_steps=stop_after_accepted_steps,
            )
        elif use_schedule_trace_only:
            rollout = _radau_adaptive_schedule_rollout(
                execution_context,
                prepared_rollout.initial_carry,
                max_total_steps=max_total_steps,
                stop_after_accepted_steps=stop_after_accepted_steps,
            )
        else:
            rollout = _radau_adaptive_final_state_rollout(
                execution_context,
                prepared_rollout.initial_carry,
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


def _forward_benchmark_adaptive_rollout_final_state_for_parameter(
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
    """Forward-benchmark-owned adaptive rollout helper.

    Keep the forward benchmark lane structurally independent from reverse
    benchmark plumbing, even where the current implementation details happen to
    match.
    """

    prepare_fn = (
        _forward_benchmark_prepare_realized_schedule_scalar_rollout_ad_lane
        if use_realized_schedule_jvp
        else _forward_benchmark_prepare_realized_schedule_scalar_rollout
    )
    (
        execution_context,
        prepared_rollout,
        initial_carry,
        max_total_steps,
        stop_after_accepted_steps,
        _solver,
        _solve_vector_field,
    ) = prepare_fn(
        parameter_value,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        accepted_step_limit_override=accepted_step_limit_override,
    )
    if use_realized_schedule_jvp:
        final_y = _forward_benchmark_adaptive_final_y_realized_schedule(
            execution_context,
            max_total_steps,
            stop_after_accepted_steps,
            initial_carry,
        )
        final_state = prepared_rollout.physics_context.unpack_flat(final_y)
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
        rollout = (
            _radau_adaptive_payload_trace_rollout(
                execution_context,
                initial_carry,
                max_total_steps=max_total_steps,
                stop_after_accepted_steps=stop_after_accepted_steps,
            )
            if use_payload_trace
            else _radau_adaptive_schedule_rollout(
                execution_context,
                initial_carry,
                max_total_steps=max_total_steps,
                stop_after_accepted_steps=stop_after_accepted_steps,
            )
            if use_schedule_trace_only
            else _radau_adaptive_final_state_rollout(
                execution_context,
                initial_carry,
                max_total_steps=max_total_steps,
                stop_after_accepted_steps=stop_after_accepted_steps,
            )
        )
        final_state = prepared_rollout.physics_context.unpack_flat(rollout.final_carry.y)
    return final_state, rollout


def _forward_benchmark_replay_realized_accepted_final_y(
    execution_context,
    carry0,
    accepted_mask,
    dt_sequence,
):
    """Forward-owned lightweight accepted-step replay for the scalar AD lane."""

    kernel_context = execution_context.kernel_context
    physics_context = execution_context.physics_context
    dtype = kernel_context.dtype

    def _scan_body(carry, xs):
        accepted, dt_value = xs

        def _do_step(_):
            carry_for_step = dataclasses.replace(carry, dt=dt_value)
            attempt_context = _RadauAcceptedStepAttemptContext(
                t_final=carry.t + dt_value,
                use_transport_lagged_response=jnp.asarray(kernel_context.use_transport_lagged_response),
            )
            step_map_result = _radau_apply_accepted_step_map(
                kernel_context,
                physics_context,
                _radau_carry_with_forward_only_jvp_fields(carry_for_step),
                attempt_context,
            )
            next_carry = dataclasses.replace(
                step_map_result.next_carry,
                prev_error=jnp.maximum(
                    step_map_result.err_norm,
                    jnp.asarray(1.0e-12, dtype=dtype),
                ),
                recent_reject_count=jnp.asarray(0, dtype=jnp.int32),
                regrowth_cooldown=jnp.asarray(0, dtype=jnp.int32),
                easy_growth_streak=jnp.asarray(0, dtype=jnp.int32),
            )
            return next_carry, step_map_result.accepted_y

        def _skip(_):
            return carry, carry.y

        return jax.lax.cond(accepted, _do_step, _skip, operand=None)

    final_carry, _ = jax.lax.scan(_scan_body, carry0, (accepted_mask, dt_sequence))
    return final_carry.y


@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2))
def _forward_benchmark_adaptive_final_y_realized_schedule(
    execution_context,
    max_total_steps: int,
    stop_after_accepted_steps: int | None,
    carry0,
):
    """Forward-owned custom-JVP final-y helper for the scalar accepted-step lane."""

    rollout = _radau_adaptive_final_state_rollout(
        execution_context,
        carry0,
        max_total_steps=max_total_steps,
        stop_after_accepted_steps=stop_after_accepted_steps,
    )
    return rollout.final_carry.y


@_forward_benchmark_adaptive_final_y_realized_schedule.defjvp
def _forward_benchmark_adaptive_final_y_realized_schedule_jvp(
    execution_context,
    max_total_steps: int,
    stop_after_accepted_steps: int | None,
    primals,
    tangents,
):
    (carry0,) = primals
    (carry0_dot,) = tangents
    rollout = _radau_adaptive_final_state_rollout(
        execution_context,
        carry0,
        max_total_steps=max_total_steps,
        stop_after_accepted_steps=stop_after_accepted_steps,
    )
    accepted_mask = jax.lax.stop_gradient(
        jnp.logical_and(rollout.trace.active_mask, rollout.trace.accepted_mask)
    )
    accepted_dts = jax.lax.stop_gradient(rollout.trace.attempted_dts)

    def _replay(carry_value):
        return _forward_benchmark_replay_realized_accepted_final_y(
            execution_context,
            carry_value,
            accepted_mask,
            accepted_dts,
        )

    primal_out, tangent_out = jax.jvp(_replay, (carry0,), (carry0_dot,))
    return primal_out, tangent_out


def _adaptive_rollout_objectives_for_parameter(
    parameter_value,
    *,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    parameter_name: str,
    use_realized_schedule_jvp: bool = False,
    accepted_step_limit_override: int | None = None,
):
    final_state, rollout = _adaptive_rollout_final_state_for_parameter(
        parameter_value,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        use_realized_schedule_jvp=use_realized_schedule_jvp,
        accepted_step_limit_override=accepted_step_limit_override,
    )
    return _objective_vector(final_state, runtime), rollout


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
    prepared_components = prepare_transport_solver_components(config, runtime, state0)
    solve_vector_field = prepared_components["solve_vector_field"]
    prepared_rollout = _build_prepared_radau_accepted_rollout(
        solver=solver,
        state=state0,
        vector_field=solve_vector_field,
        species=runtime.species,
    )
    max_total_steps = int(max(1, getattr(solver, "max_steps", 1)))
    stop_after_accepted_steps = (
        int(accepted_step_limit_override)
        if accepted_step_limit_override is not None
        else getattr(solver, "stop_after_accepted_steps", None)
    )
    derivative_mode_key = str(derivative_mode).strip().lower()
    if derivative_mode_key == "jvp":
        final_y = _radau_adaptive_final_y_realized_schedule(
            execution_context,
            max_total_steps,
            stop_after_accepted_steps,
            prepared_rollout.initial_carry,
        )
    elif derivative_mode_key == "vjp":
        final_y = _radau_adaptive_final_y_realized_schedule_vjp(
            execution_context,
            max_total_steps,
            stop_after_accepted_steps,
            prepared_rollout.initial_carry,
        )
    else:
        raise ValueError("derivative_mode must be one of {'jvp', 'vjp'}.")
    final_state = prepared_rollout.physics_context.unpack_flat(final_y)
    return _objective_vector(final_state, runtime)


def _forward_benchmark_adaptive_rollout_objectives_realized_schedule_only_for_parameter(
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
    (
        execution_context,
        prepared_rollout,
        initial_carry,
        max_total_steps,
        stop_after_accepted_steps,
        _solver,
        _solve_vector_field,
    ) = _forward_benchmark_prepare_realized_schedule_scalar_rollout_ad_lane(
        parameter_value,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        accepted_step_limit_override=accepted_step_limit_override,
    )
    derivative_mode_key = str(derivative_mode).strip().lower()
    if derivative_mode_key == "jvp":
        final_y = _forward_benchmark_adaptive_final_y_realized_schedule(
            execution_context,
            max_total_steps,
            stop_after_accepted_steps,
            initial_carry,
        )
    elif derivative_mode_key == "vjp":
        final_y = _radau_adaptive_final_y_realized_schedule_vjp(
            execution_context,
            max_total_steps,
            stop_after_accepted_steps,
            initial_carry,
        )
    else:
        raise ValueError("derivative_mode must be one of {'jvp', 'vjp'}.")
    final_state = prepared_rollout.physics_context.unpack_flat(final_y)
    return _objective_vector(final_state, runtime)


def _forward_benchmark_adaptive_rollout_objectives_realized_schedule_only_for_parameter_jvp(
    parameter_value,
    *,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    parameter_name: str,
    accepted_step_limit_override: int | None = None,
):
    """Dedicated forward custom-JVP objective helper for the scalar benchmark lane."""

    return _forward_benchmark_adaptive_rollout_objectives_realized_schedule_only_for_parameter(
        parameter_value,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        accepted_step_limit_override=accepted_step_limit_override,
        derivative_mode="jvp",
    )


def _forward_benchmark_adaptive_realized_schedule_jvp_stage_debug_for_parameter(
    parameter_value,
    *,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    parameter_name: str,
    accepted_step_limit_override: int | None = None,
):
    """Forward scalar JVP stage-local finiteness debug."""

    (
        execution_context,
        prepared_rollout,
        initial_carry,
        max_total_steps,
        stop_after_accepted_steps,
        _solver,
        _solve_vector_field,
    ) = _forward_benchmark_prepare_realized_schedule_scalar_rollout_ad_lane(
        parameter_value,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        accepted_step_limit_override=accepted_step_limit_override,
    )

    def _final_y_from_parameter(pval):
        (
            exec_ctx,
            _prepared_rollout_local,
            initial_carry_local,
            max_total_steps_local,
            stop_after_accepted_steps_local,
            _solver_local,
            _solve_vector_field_local,
        ) = _forward_benchmark_prepare_realized_schedule_scalar_rollout_ad_lane(
            pval,
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=parameter_name,
            accepted_step_limit_override=accepted_step_limit_override,
        )
        return _forward_benchmark_adaptive_final_y_realized_schedule(
            exec_ctx,
            max_total_steps_local,
            stop_after_accepted_steps_local,
            initial_carry_local,
        )

    primal_value = jnp.asarray(parameter_value, dtype=jnp.float64)
    tangent_value = jnp.asarray(1.0, dtype=jnp.float64)
    final_y, final_y_dot = jax.jvp(
        _final_y_from_parameter,
        (primal_value,),
        (tangent_value,),
    )
    final_state = prepared_rollout.physics_context.unpack_flat(final_y)
    final_state_dot = jax.jvp(
        prepared_rollout.physics_context.unpack_flat,
        (final_y,),
        (final_y_dot,),
    )[1]
    objective_primal, objective_tangent = jax.jvp(
        lambda flat_y: _objective_vector(prepared_rollout.physics_context.unpack_flat(flat_y), runtime),
        (final_y,),
        (final_y_dot,),
    )
    return {
        "final_y_all_finite": _tree_all_finite(final_y),
        "final_y_dot_all_finite": _tree_all_finite(final_y_dot),
        "final_state_all_finite": _tree_all_finite(final_state),
        "final_state_dot_all_finite": _tree_all_finite(final_state_dot),
        "objective_primal_all_finite": bool(np.all(np.isfinite(np.asarray(jax.device_get(objective_primal), dtype=float)))),
        "objective_tangent_all_finite": bool(np.all(np.isfinite(np.asarray(jax.device_get(objective_tangent), dtype=float)))),
        "objective_primal": np.asarray(jax.device_get(objective_primal), dtype=float).tolist(),
        "objective_tangent": np.asarray(jax.device_get(objective_tangent), dtype=float).tolist(),
    }


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
    flat_state0, unpack_flat, _unpack_packed, _pack_state, project_flat = _make_solver_state_transform(
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
        jnp.asarray(kernel_context.use_transport_lagged_response),
        flat_state0,
    )
    return _radau_carry_from_step_state(step_state0)


def _forward_benchmark_prepare_realized_schedule_scalar_rollout(
    parameter_value,
    *,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    parameter_name: str,
    accepted_step_limit_override: int | None = None,
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
        state=state0,
        solver=solver,
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
    return (
        execution_context,
        prepared_rollout_static,
        initial_carry,
        max_total_steps,
        stop_after_accepted_steps,
        solver,
        solve_vector_field_static,
    )


def _forward_benchmark_prepare_realized_schedule_scalar_rollout_ad_lane(
    parameter_value,
    *,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    parameter_name: str,
    accepted_step_limit_override: int | None = None,
):
    """Forward-owned scalar AD prepare helper using the prepared initial carry directly."""

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
    prepared_components = prepare_transport_solver_components(config, runtime, state0)
    solve_vector_field = prepared_components["solve_vector_field"]
    prepared_rollout = _build_prepared_radau_accepted_rollout(
        solver=solver,
        state=state0,
        vector_field=solve_vector_field,
        species=runtime.species,
    )
    max_total_steps = int(max(1, getattr(solver, "max_steps", 1)))
    stop_after_accepted_steps = (
        int(accepted_step_limit_override)
        if accepted_step_limit_override is not None
        else getattr(solver, "stop_after_accepted_steps", None)
    )
    return (
        execution_context,
        prepared_rollout,
        prepared_rollout.initial_carry,
        max_total_steps,
        stop_after_accepted_steps,
        solver,
        solve_vector_field,
    )


def _prepare_realized_schedule_profile_vector_rollout_option_a(
    parameter_values,
    *,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    parameter_names: tuple[str, ...] = PROFILE_VECTOR_PARAMETERS,
    accepted_step_limit_override: int | None = None,
):
    parameter_values = jnp.asarray(parameter_values, dtype=jnp.float64)
    state0 = _parameterized_initial_state_multi(
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        geometry=runtime.geometry,
        n_species=runtime.species.number_species,
        parameter_names=parameter_names,
        parameter_values=parameter_values,
    )
    state0_static = _parameterized_initial_state_multi(
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        geometry=runtime.geometry,
        n_species=runtime.species.number_species,
        parameter_names=parameter_names,
        parameter_values=jax.lax.stop_gradient(parameter_values),
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
    solve_vector_field = solve_vector_field_static
    initial_carry = _initial_carry_from_state_with_static_setup(
        state=state0,
        solver=solver,
        solve_vector_field=solve_vector_field,
        species=runtime.species,
        prepared_rollout_static=prepared_rollout_static,
    )
    max_total_steps = int(max(1, getattr(solver, "max_steps", 1)))
    stop_after_accepted_steps = (
        int(accepted_step_limit_override)
        if accepted_step_limit_override is not None
        else getattr(solver, "stop_after_accepted_steps", None)
    )
    return (
        execution_context,
        prepared_rollout_static,
        initial_carry,
        max_total_steps,
        stop_after_accepted_steps,
        solver,
        solve_vector_field_static,
    )


def _forward_benchmark_prepare_realized_schedule_profile_vector_rollout(
    parameter_values,
    *,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    parameter_names: tuple[str, ...] = PROFILE_VECTOR_PARAMETERS,
    accepted_step_limit_override: int | None = None,
):
    parameter_values = jnp.asarray(parameter_values, dtype=jnp.float64)
    state0 = _parameterized_initial_state_multi(
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        geometry=runtime.geometry,
        n_species=runtime.species.number_species,
        parameter_names=parameter_names,
        parameter_values=parameter_values,
    )
    state0_static = _parameterized_initial_state_multi(
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        geometry=runtime.geometry,
        n_species=runtime.species.number_species,
        parameter_names=parameter_names,
        parameter_values=jax.lax.stop_gradient(parameter_values),
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
    prepared_components = prepare_transport_solver_components(config, runtime, state0)
    solve_vector_field = prepared_components["solve_vector_field"]
    prepared_rollout = _build_prepared_radau_accepted_rollout(
        solver=solver,
        state=state0,
        vector_field=solve_vector_field,
        species=runtime.species,
    )
    max_total_steps = int(max(1, getattr(solver, "max_steps", 1)))
    stop_after_accepted_steps = (
        int(accepted_step_limit_override)
        if accepted_step_limit_override is not None
        else getattr(solver, "stop_after_accepted_steps", None)
    )
    return (
        execution_context,
        prepared_rollout,
        prepared_rollout.initial_carry,
        max_total_steps,
        stop_after_accepted_steps,
        solver,
        solve_vector_field,
    )


def _objective_vector_from_final_y(
    final_y,
    *,
    prepared_rollout,
    runtime,
):
    final_state = prepared_rollout.physics_context.unpack_flat(final_y)
    return _objective_vector(final_state, runtime)


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


def _initial_carry_vdot_leaf_filter() -> str | None:
    raw_value = os.environ.get("NEOPAX_TRANSPORT_REVERSE_INITIAL_CARRY_LEAF")
    if raw_value is None:
        return None
    value = raw_value.strip()
    return value or None


def _zero_initial_carry_optional_pytree(tree):
    if tree is None:
        return None

    def _zero_leaf(x):
        arr = jnp.asarray(x)
        if jnp.issubdtype(arr.dtype, jnp.inexact):
            return jnp.zeros_like(arr)
        return jnp.zeros(arr.shape, dtype=jax.dtypes.float0)

    return jax.tree_util.tree_map(_zero_leaf, tree)


def _select_initial_carry_leaf(carry, selected_leaf: str | None):
    if selected_leaf is None:
        return carry
    zeroed = jax.tree_util.tree_map(_zero_initial_carry_optional_pytree, carry)
    if not hasattr(carry, selected_leaf):
        raise ValueError(f"Unknown initial carry leaf '{selected_leaf}'.")
    return dataclasses.replace(
        zeroed,
        **{selected_leaf: getattr(carry, selected_leaf)},
    )


def _local_adjoint_check_enabled() -> bool:
    raw_value = os.environ.get("NEOPAX_TRANSPORT_REVERSE_LOCAL_ADJOINT_CHECK")
    if raw_value is None:
        return False
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _rollout_adjoint_check_enabled() -> bool:
    raw_value = os.environ.get("NEOPAX_TRANSPORT_REVERSE_ROLLOUT_ADJOINT_CHECK")
    if raw_value is None:
        return False
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _rollout_adjoint_check_basis_index() -> int:
    raw_value = os.environ.get("NEOPAX_TRANSPORT_REVERSE_ROLLOUT_ADJOINT_BASIS")
    if raw_value is None:
        return 0
    return max(0, int(raw_value))


def _parameter_carry_diagnostic_enabled() -> bool:
    raw_value = os.environ.get("NEOPAX_TRANSPORT_REVERSE_PARAMETER_CARRY_DIAGNOSTIC")
    if raw_value is None:
        return False
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _host_step_pullback_diagnostic_enabled() -> bool:
    raw_value = os.environ.get("NEOPAX_TRANSPORT_REVERSE_HOST_STEP_PULLBACK_DIAGNOSTIC")
    if raw_value is None:
        return False
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}

def _step_pullback_segment_index() -> int:
    raw_value = os.environ.get("NEOPAX_TRANSPORT_REVERSE_STEP_PULLBACK_SEGMENT")
    if raw_value is None:
        return 1
    return max(0, int(raw_value))


def _step_pullback_step_index() -> int:
    raw_value = os.environ.get("NEOPAX_TRANSPORT_REVERSE_STEP_PULLBACK_STEP")
    if raw_value is None:
        return 10
    return max(0, int(raw_value))


def _host_step_pullback_y_bar_path() -> str | None:
    raw_value = os.environ.get("NEOPAX_TRANSPORT_REVERSE_HOST_STEP_PULLBACK_Y_BAR_PATH")
    if raw_value is None:
        return None
    value = raw_value.strip()
    return value or None


def _host_step_pullback_replay_dy_bar_path() -> str | None:
    raw_value = os.environ.get("NEOPAX_TRANSPORT_REVERSE_HOST_STEP_PULLBACK_REPLAY_DY_BAR_PATH")
    if raw_value is None:
        return None
    value = raw_value.strip()
    return value or None


def _host_step_pullback_replay_inputs_path() -> str | None:
    raw_value = os.environ.get("NEOPAX_TRANSPORT_REVERSE_HOST_STEP_PULLBACK_REPLAY_INPUTS_PATH")
    if raw_value is None:
        return None
    value = raw_value.strip()
    return value or None


def _host_step_pullback_replay_lagged_cache_path() -> str | None:
    raw_value = os.environ.get("NEOPAX_TRANSPORT_REVERSE_HOST_STEP_PULLBACK_REPLAY_LAGGED_CACHE_PATH")
    if raw_value is None:
        return None
    value = raw_value.strip()
    return value or None


def _host_step_pullback_replay_primal_path() -> str | None:
    raw_value = os.environ.get("NEOPAX_TRANSPORT_REVERSE_HOST_STEP_PULLBACK_REPLAY_PRIMAL_PATH")
    if raw_value is None:
        return None
    value = raw_value.strip()
    return value or None


def _run_host_local_step_pullback_diagnostic_for_parameter_vector(
    parameter_values,
    *,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    parameter_names: tuple[str, ...] = PROFILE_VECTOR_PARAMETERS,
    accepted_step_limit_override: int | None = None,
):
    accepted_y_bar_override = None
    y_bar_path = _host_step_pullback_y_bar_path()
    if y_bar_path is not None:
        accepted_y_bar_override = np.load(y_bar_path)
    replay_dy_bar_override = None
    replay_dy_bar_path = _host_step_pullback_replay_dy_bar_path()
    if replay_dy_bar_path is not None:
        replay_dy_bar_override = np.load(replay_dy_bar_path)
    replay_inputs_override = None
    replay_inputs_path = _host_step_pullback_replay_inputs_path()
    if replay_inputs_path is not None:
        replay_inputs_override = dict(np.load(replay_inputs_path))
    replay_lagged_cache_override = None
    replay_lagged_cache_path = _host_step_pullback_replay_lagged_cache_path()
    if replay_lagged_cache_path is not None:
        with open(replay_lagged_cache_path, "rb") as f:
            replay_lagged_cache_override = pickle.load(f)
    replay_primal_override = None
    replay_primal_path = _host_step_pullback_replay_primal_path()
    if replay_primal_path is not None:
        replay_primal_override = dict(np.load(replay_primal_path))
    execution_context, _prepared_rollout, initial_carry, max_total_steps, stop_after_accepted_steps, _solver, _solve_vector_field = (
        _prepare_realized_schedule_profile_vector_rollout_option_a(
            parameter_values,
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_names=parameter_names,
            accepted_step_limit_override=accepted_step_limit_override,
        )
    )
    return _radau_host_local_step_pullback_compare(
        execution_context,
        initial_carry,
        max_total_steps=max_total_steps,
        stop_after_accepted_steps=stop_after_accepted_steps,
        segment_index=_step_pullback_segment_index(),
        step_index=_step_pullback_step_index(),
        accepted_y_bar_override=accepted_y_bar_override,
        replay_dy_bar_override=replay_dy_bar_override,
        replay_inputs_override=replay_inputs_override,
        replay_lagged_cache_override=replay_lagged_cache_override,
        replay_primal_override=replay_primal_override,
    )
def _adaptive_rollout_objectives_realized_schedule_only_for_parameter_vector(
    parameter_values,
    *,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    parameter_names: tuple[str, ...] = PROFILE_VECTOR_PARAMETERS,
    accepted_step_limit_override: int | None = None,
    derivative_mode: str = "jvp",
):
    parameter_values = jnp.asarray(parameter_values, dtype=jnp.float64)
    state0 = _parameterized_initial_state_multi(
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        geometry=runtime.geometry,
        n_species=runtime.species.number_species,
        parameter_names=parameter_names,
        parameter_values=parameter_values,
    )
    state0_static = _parameterized_initial_state_multi(
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        geometry=runtime.geometry,
        n_species=runtime.species.number_species,
        parameter_names=parameter_names,
        parameter_values=jax.lax.stop_gradient(parameter_values),
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
    prepared_components = prepare_transport_solver_components(config, runtime, state0)
    solve_vector_field = prepared_components["solve_vector_field"]
    prepared_rollout = _build_prepared_radau_accepted_rollout(
        solver=solver,
        state=state0,
        vector_field=solve_vector_field,
        species=runtime.species,
    )
    max_total_steps = int(max(1, getattr(solver, "max_steps", 1)))
    stop_after_accepted_steps = (
        int(accepted_step_limit_override)
        if accepted_step_limit_override is not None
        else getattr(solver, "stop_after_accepted_steps", None)
    )
    derivative_mode_key = str(derivative_mode).strip().lower()
    if derivative_mode_key == "jvp":
        final_y = _radau_adaptive_final_y_realized_schedule(
            execution_context,
            max_total_steps,
            stop_after_accepted_steps,
            prepared_rollout.initial_carry,
        )
    elif derivative_mode_key == "vjp":
        final_y = _radau_adaptive_final_y_realized_schedule_vjp(
            execution_context,
            max_total_steps,
            stop_after_accepted_steps,
            prepared_rollout.initial_carry,
        )
    else:
        raise ValueError("derivative_mode must be one of {'jvp', 'vjp'}.")
    final_state = prepared_rollout.physics_context.unpack_flat(final_y)
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
    replay = _radau_run_prepared_on_realized_trace(
        prepared_rollout,
        execution_context,
        frozen_trace,
        replay_mode=replay_mode,
    )
    return _objective_vector(replay["final_state"], runtime), replay


def _forward_benchmark_adaptive_rollout_objectives_for_parameter_on_frozen_trace(
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
    (
        execution_context,
        prepared_rollout,
        initial_carry,
        _max_total_steps,
        _stop_after_accepted_steps,
        _solver,
        _solve_vector_field,
    ) = _forward_benchmark_prepare_realized_schedule_scalar_rollout(
        parameter_value,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
    )
    replay = _radau_run_prepared_on_realized_trace(
        prepared_rollout,
        execution_context,
        frozen_trace,
        replay_mode=replay_mode,
        carry0=initial_carry,
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
    replay = _radau_run_prepared_on_time_list(
        prepared_rollout,
        time_list,
    )
    return _objective_vector(replay["final_state"], runtime), replay


def _compress_accepted_trial_ys(
    trial_ys: jax.Array,
    accepted_mask: jax.Array,
    accepted_count: int,
) -> jax.Array:
    accepted_count = int(accepted_count)
    if accepted_count <= 0:
        return jnp.zeros((0, trial_ys.shape[-1]), dtype=trial_ys.dtype)

    def _scan_body(carry, xs):
        write_idx, out = carry
        accepted, flat_y = xs

        def _write(_):
            next_out = out.at[write_idx].set(flat_y)
            return write_idx + 1, next_out

        next_carry = jax.lax.cond(
            jnp.logical_and(accepted, write_idx < accepted_count),
            _write,
            lambda _: (write_idx, out),
            operand=None,
        )
        return next_carry, None

    init = (
        jnp.asarray(0, dtype=jnp.int32),
        jnp.zeros((accepted_count, trial_ys.shape[-1]), dtype=trial_ys.dtype),
    )
    (_, packed), _ = jax.lax.scan(_scan_body, init, (accepted_mask, trial_ys))
    return packed


def _objective_trajectory_from_flat_ys(
    flat_ys: jax.Array,
    *,
    runtime,
    unpack_flat,
) -> jax.Array:
    def _single(flat_y):
        return _objective_vector(unpack_flat(flat_y), runtime)

    if flat_ys.shape[0] == 0:
        return jnp.zeros((0, len(OBJECTIVE_LABELS)), dtype=jnp.asarray(runtime.geometry.rho_grid).dtype)
    return jax.vmap(_single)(flat_ys)


def _adaptive_rollout_objective_trajectory_on_time_list(
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
    replay = _radau_run_prepared_on_time_list(
        prepared_rollout,
        time_list,
    )
    flat_ys = replay["rollout"].trial_ys
    trajectory = _objective_trajectory_from_flat_ys(
        flat_ys,
        runtime=runtime,
        unpack_flat=prepared_rollout.physics_context.unpack_flat,
    )
    return trajectory, replay


def _adaptive_rollout_objective_trajectory_on_realized_trace(
    parameter_value,
    *,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    parameter_name: str,
    frozen_trace,
    replay_mode: str = "attempt",
    accepted_count: int,
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
    replay = _radau_run_prepared_on_realized_trace(
        prepared_rollout,
        execution_context,
        frozen_trace,
        replay_mode=replay_mode,
    )
    accepted_flat_ys = _compress_accepted_trial_ys(
        replay["rollout"].trial_ys,
        jax.lax.stop_gradient(frozen_trace.accepted_mask),
        accepted_count=accepted_count,
    )
    trajectory = _objective_trajectory_from_flat_ys(
        accepted_flat_ys,
        runtime=runtime,
        unpack_flat=prepared_rollout.physics_context.unpack_flat,
    )
    return trajectory, replay


def _adaptive_rollout_initial_carry_for_parameter(
    parameter_value,
    *,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    parameter_name: str,
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
    return prepared_rollout.initial_carry


def _sample_accepted_step_indices(total_accepted: int, sample_every: int) -> tuple[int, ...]:
    total_accepted = int(total_accepted)
    sample_every = max(1, int(sample_every))
    if total_accepted <= 0:
        return ()
    indices = list(range(0, total_accepted, sample_every))
    if indices[-1] != total_accepted - 1:
        indices.append(total_accepted - 1)
    return tuple(indices)


def _objective_tangent_from_flat_y(
    flat_y,
    flat_y_dot,
    *,
    runtime,
    unpack_flat,
):
    def _obj_from_flat(y_flat):
        return _objective_vector(unpack_flat(y_flat), runtime)

    _, tangent = jax.jvp(_obj_from_flat, (flat_y,), (flat_y_dot,))
    return tangent


def _adaptive_rollout_flat_state_trajectory_on_time_list(
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
    replay = _radau_run_prepared_on_time_list(
        prepared_rollout,
        time_list,
    )
    return replay["rollout"].trial_ys, replay


def _sampled_adaptive_objective_tangent_trajectory(
    *,
    execution_context,
    carry0,
    carry0_dot,
    trace,
    runtime,
    unpack_flat,
    sample_every: int,
):
    sample_indices = _sample_accepted_step_indices(
        int(np.sum(np.asarray(jax.device_get(trace.accepted_mask), dtype=bool))),
        sample_every,
    )
    sample_count = len(sample_indices)
    if sample_count == 0:
        return {
            "sampled_times": jnp.zeros((0,), dtype=execution_context.dtype),
            "sampled_tangents": jnp.zeros((0, len(OBJECTIVE_LABELS)), dtype=execution_context.dtype),
            "sampled_indices": (),
        }

    sample_indices_arr = jnp.asarray(sample_indices, dtype=jnp.int32)
    active_mask = jax.lax.stop_gradient(trace.active_mask)
    accepted_mask = jax.lax.stop_gradient(trace.accepted_mask)
    attempted_dts = jax.lax.stop_gradient(trace.attempted_dts)
    next_dts = jax.lax.stop_gradient(trace.next_dts)
    next_recent_reject_count = jax.lax.stop_gradient(trace.next_recent_reject_count)
    next_regrowth_cooldown = jax.lax.stop_gradient(trace.next_regrowth_cooldown)
    next_easy_growth_streak = jax.lax.stop_gradient(trace.next_easy_growth_streak)
    next_lagged_response_valid = jax.lax.stop_gradient(trace.next_lagged_response_valid)
    step_ts = jax.lax.stop_gradient(trace.step_ts)

    def _accepted_attempt(
        carry_value,
        *,
        dt_value,
        next_dt_value,
        recent_reject_count_value,
        regrowth_cooldown_value,
        easy_growth_streak_value,
        lagged_response_valid_value,
    ):
        carry_for_step = dataclasses.replace(carry_value, dt=dt_value)
        attempt_result = _execute_radau_accepted_step_attempt_autodiff(
            execution_context.kernel_context,
            execution_context.physics_context,
            _radau_carry_with_forward_only_jvp_fields(carry_for_step),
            execution_context.attempt_context,
        )
        project_flat = execution_context.physics_context.project_flat
        accepted_y = project_flat(attempt_result.trial_y) if project_flat is not None else None
        if accepted_y is None:
            accepted_y = attempt_result.trial_y
        return dataclasses.replace(
            attempt_result.carry_after_attempt,
            t=carry_value.t + dt_value,
            y=accepted_y,
            dt=next_dt_value,
            prev_error=jnp.maximum(
                attempt_result.err_norm,
                jnp.asarray(1.0e-12, dtype=execution_context.dtype),
            ),
            prev_stages=attempt_result.stage_history,
            prev_dt=dt_value,
            recent_reject_count=recent_reject_count_value,
            regrowth_cooldown=regrowth_cooldown_value,
            easy_growth_streak=easy_growth_streak_value,
            lagged_response_valid=lagged_response_valid_value,
            jacobian=attempt_result.jacobian_out,
            cache_valid=attempt_result.cache_valid_out,
            cache_dt=attempt_result.cache_dt_out,
            cache_age=attempt_result.cache_age_out,
            real_lu=attempt_result.real_lu_out,
            real_piv=attempt_result.real_piv_out,
            complex_lu=attempt_result.complex_lu_out,
            complex_piv=attempt_result.complex_piv_out,
            prev_theta_final=attempt_result.theta_final,
            prev_newton_iter_count=attempt_result.newton_iter_count,
        )

    def _rejected_attempt(
        carry_value,
        *,
        dt_value,
        next_dt_value,
        recent_reject_count_value,
        regrowth_cooldown_value,
        easy_growth_streak_value,
        lagged_response_valid_value,
    ):
        carry_for_step = dataclasses.replace(jax.lax.stop_gradient(carry_value), dt=dt_value)
        attempt_result = _execute_radau_accepted_step_attempt_autodiff(
            execution_context.kernel_context,
            execution_context.physics_context,
            _radau_carry_with_forward_only_jvp_fields(carry_for_step),
            execution_context.attempt_context,
        )
        return dataclasses.replace(
            carry_value,
            dt=next_dt_value,
            recent_reject_count=recent_reject_count_value,
            regrowth_cooldown=regrowth_cooldown_value,
            easy_growth_streak=easy_growth_streak_value,
            lagged_response_cache=jax.lax.stop_gradient(attempt_result.carry_after_attempt.lagged_response_cache),
            lagged_response_valid=lagged_response_valid_value,
            lagged_reference_y=jax.lax.stop_gradient(attempt_result.carry_after_attempt.lagged_reference_y),
            jacobian=jax.lax.stop_gradient(attempt_result.jacobian_out),
            cache_valid=jax.lax.stop_gradient(attempt_result.cache_valid_out),
            cache_dt=jax.lax.stop_gradient(attempt_result.cache_dt_out),
            cache_age=jax.lax.stop_gradient(attempt_result.cache_age_out),
            real_lu=jax.lax.stop_gradient(attempt_result.real_lu_out),
            real_piv=jax.lax.stop_gradient(attempt_result.real_piv_out),
            complex_lu=jax.lax.stop_gradient(attempt_result.complex_lu_out),
            complex_piv=jax.lax.stop_gradient(attempt_result.complex_piv_out),
            prev_theta_final=jax.lax.stop_gradient(attempt_result.theta_final),
            prev_newton_iter_count=jax.lax.stop_gradient(attempt_result.newton_iter_count),
        )

    zero_tangents = jnp.zeros((sample_count, len(OBJECTIVE_LABELS)), dtype=execution_context.dtype)
    zero_times = jnp.zeros((sample_count,), dtype=execution_context.dtype)

    def _scan_body(scan_state, inputs):
        carry, carry_dot, accepted_seen, sample_write_idx, sampled_times, sampled_tangents = scan_state
        (
            active,
            accepted,
            dt_value,
            next_dt_value,
            recent_reject_count_value,
            regrowth_cooldown_value,
            easy_growth_streak_value,
            lagged_response_valid_value,
            time_value,
        ) = inputs

        def _run_step(step_operand):
            carry_in, carry_dot_in = step_operand

            def _run_accepted(_):
                return jax.jvp(
                    lambda c: _accepted_attempt(
                        c,
                        dt_value=dt_value,
                        next_dt_value=next_dt_value,
                        recent_reject_count_value=recent_reject_count_value,
                        regrowth_cooldown_value=regrowth_cooldown_value,
                        easy_growth_streak_value=easy_growth_streak_value,
                        lagged_response_valid_value=lagged_response_valid_value,
                    ),
                    (carry_in,),
                    (carry_dot_in,),
                )

            def _run_rejected(_):
                return jax.jvp(
                    lambda c: _rejected_attempt(
                        c,
                        dt_value=dt_value,
                        next_dt_value=next_dt_value,
                        recent_reject_count_value=recent_reject_count_value,
                        regrowth_cooldown_value=regrowth_cooldown_value,
                        easy_growth_streak_value=easy_growth_streak_value,
                        lagged_response_valid_value=lagged_response_valid_value,
                    ),
                    (carry_in,),
                    (carry_dot_in,),
                )

            return jax.lax.cond(accepted, _run_accepted, _run_rejected, operand=None)

        def _skip_step(step_operand):
            return step_operand

        next_carry, next_carry_dot = jax.lax.cond(
            active,
            _run_step,
            _skip_step,
            operand=(carry, carry_dot),
        )
        accepted_seen_next = accepted_seen + jnp.where(jnp.logical_and(active, accepted), jnp.asarray(1, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32))
        just_accepted_index = accepted_seen_next - 1
        should_sample = jnp.logical_and(
            jnp.logical_and(active, accepted),
            jnp.logical_and(sample_write_idx < sample_count, just_accepted_index == sample_indices_arr[sample_write_idx]),
        )

        def _write_sample(_):
            obj_tangent = _objective_tangent_from_flat_y(
                next_carry.y,
                next_carry_dot.y,
                runtime=runtime,
                unpack_flat=unpack_flat,
            )
            times_next = sampled_times.at[sample_write_idx].set(time_value)
            tangents_next = sampled_tangents.at[sample_write_idx].set(obj_tangent)
            return sample_write_idx + 1, times_next, tangents_next

        sample_write_idx_next, sampled_times_next, sampled_tangents_next = jax.lax.cond(
            should_sample,
            _write_sample,
            lambda _: (sample_write_idx, sampled_times, sampled_tangents),
            operand=None,
        )
        return (
            next_carry,
            next_carry_dot,
            accepted_seen_next,
            sample_write_idx_next,
            sampled_times_next,
            sampled_tangents_next,
        ), None

    init_state = (
        carry0,
        carry0_dot,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        zero_times,
        zero_tangents,
    )
    final_state, _ = jax.lax.scan(
        _scan_body,
        init_state,
        (
            active_mask,
            accepted_mask,
            attempted_dts,
            next_dts,
            next_recent_reject_count,
            next_regrowth_cooldown,
            next_easy_growth_streak,
            next_lagged_response_valid,
            step_ts,
        ),
    )
    return {
        "sampled_times": final_state[4],
        "sampled_tangents": final_state[5],
        "sampled_indices": sample_indices,
    }


def _sampled_adaptive_state_tangent_trajectory(
    *,
    execution_context,
    carry0,
    carry0_dot,
    trace,
    sample_every: int,
):
    return _sampled_realized_trace_state_tangent_trajectory(
        execution_context=execution_context,
        carry0=carry0,
        carry0_dot=carry0_dot,
        trace=trace,
        sample_every=sample_every,
        use_custom=True,
    )


def _sampled_realized_trace_state_tangent_trajectory(
    *,
    execution_context,
    carry0,
    carry0_dot,
    trace,
    sample_every: int,
    use_custom: bool,
):
    sample_indices = _sample_accepted_step_indices(
        int(np.sum(np.asarray(jax.device_get(trace.accepted_mask), dtype=bool))),
        sample_every,
    )
    sample_count = len(sample_indices)
    if sample_count == 0:
        return {
            "sampled_times": jnp.zeros((0,), dtype=execution_context.dtype),
            "sampled_state_tangents": jnp.zeros((0, carry0.y.shape[0]), dtype=execution_context.dtype),
            "sampled_indices": (),
        }

    sample_indices_arr = jnp.asarray(sample_indices, dtype=jnp.int32)
    active_mask = jax.lax.stop_gradient(trace.active_mask)
    accepted_mask = jax.lax.stop_gradient(trace.accepted_mask)
    attempted_dts = jax.lax.stop_gradient(trace.attempted_dts)
    next_dts = jax.lax.stop_gradient(trace.next_dts)
    next_recent_reject_count = jax.lax.stop_gradient(trace.next_recent_reject_count)
    next_regrowth_cooldown = jax.lax.stop_gradient(trace.next_regrowth_cooldown)
    next_easy_growth_streak = jax.lax.stop_gradient(trace.next_easy_growth_streak)
    next_lagged_response_valid = jax.lax.stop_gradient(trace.next_lagged_response_valid)
    step_ts = jax.lax.stop_gradient(trace.step_ts)

    def _accepted_attempt(
        carry_value,
        *,
        dt_value,
        next_dt_value,
        recent_reject_count_value,
        regrowth_cooldown_value,
        easy_growth_streak_value,
        lagged_response_valid_value,
    ):
        carry_for_step = dataclasses.replace(carry_value, dt=dt_value)
        attempt_fn = _execute_radau_accepted_step_attempt_autodiff if use_custom else _execute_radau_accepted_step_attempt
        attempt_result = attempt_fn(
            execution_context.kernel_context,
            execution_context.physics_context,
            _radau_carry_with_forward_only_jvp_fields(carry_for_step),
            execution_context.attempt_context,
        )
        project_flat = execution_context.physics_context.project_flat
        accepted_y = project_flat(attempt_result.trial_y) if project_flat is not None else None
        if accepted_y is None:
            accepted_y = attempt_result.trial_y
        return dataclasses.replace(
            attempt_result.carry_after_attempt,
            t=carry_value.t + dt_value,
            y=accepted_y,
            dt=next_dt_value,
            prev_error=jnp.maximum(
                attempt_result.err_norm,
                jnp.asarray(1.0e-12, dtype=execution_context.dtype),
            ),
            prev_stages=attempt_result.stage_history,
            prev_dt=dt_value,
            recent_reject_count=recent_reject_count_value,
            regrowth_cooldown=regrowth_cooldown_value,
            easy_growth_streak=easy_growth_streak_value,
            lagged_response_valid=lagged_response_valid_value,
            jacobian=attempt_result.jacobian_out,
            cache_valid=attempt_result.cache_valid_out,
            cache_dt=attempt_result.cache_dt_out,
            cache_age=attempt_result.cache_age_out,
            real_lu=attempt_result.real_lu_out,
            real_piv=attempt_result.real_piv_out,
            complex_lu=attempt_result.complex_lu_out,
            complex_piv=attempt_result.complex_piv_out,
            prev_theta_final=attempt_result.theta_final,
            prev_newton_iter_count=attempt_result.newton_iter_count,
        )

    def _rejected_attempt(
        carry_value,
        *,
        dt_value,
        next_dt_value,
        recent_reject_count_value,
        regrowth_cooldown_value,
        easy_growth_streak_value,
        lagged_response_valid_value,
    ):
        carry_for_step = dataclasses.replace(jax.lax.stop_gradient(carry_value), dt=dt_value)
        attempt_fn = _execute_radau_accepted_step_attempt_autodiff if use_custom else _execute_radau_accepted_step_attempt
        attempt_result = attempt_fn(
            execution_context.kernel_context,
            execution_context.physics_context,
            _radau_carry_with_forward_only_jvp_fields(carry_for_step),
            execution_context.attempt_context,
        )
        return dataclasses.replace(
            carry_value,
            dt=next_dt_value,
            recent_reject_count=recent_reject_count_value,
            regrowth_cooldown=regrowth_cooldown_value,
            easy_growth_streak=easy_growth_streak_value,
            lagged_response_cache=jax.lax.stop_gradient(attempt_result.carry_after_attempt.lagged_response_cache),
            lagged_response_valid=lagged_response_valid_value,
            lagged_reference_y=jax.lax.stop_gradient(attempt_result.carry_after_attempt.lagged_reference_y),
            jacobian=jax.lax.stop_gradient(attempt_result.jacobian_out),
            cache_valid=jax.lax.stop_gradient(attempt_result.cache_valid_out),
            cache_dt=jax.lax.stop_gradient(attempt_result.cache_dt_out),
            cache_age=jax.lax.stop_gradient(attempt_result.cache_age_out),
            real_lu=jax.lax.stop_gradient(attempt_result.real_lu_out),
            real_piv=jax.lax.stop_gradient(attempt_result.real_piv_out),
            complex_lu=jax.lax.stop_gradient(attempt_result.complex_lu_out),
            complex_piv=jax.lax.stop_gradient(attempt_result.complex_piv_out),
            prev_theta_final=jax.lax.stop_gradient(attempt_result.theta_final),
            prev_newton_iter_count=jax.lax.stop_gradient(attempt_result.newton_iter_count),
        )

    zero_tangents = jnp.zeros((sample_count, carry0.y.shape[0]), dtype=execution_context.dtype)
    zero_times = jnp.zeros((sample_count,), dtype=execution_context.dtype)

    def _scan_body(scan_state, inputs):
        carry, carry_dot, accepted_seen, sample_write_idx, sampled_times, sampled_tangents = scan_state
        (
            active,
            accepted,
            dt_value,
            next_dt_value,
            recent_reject_count_value,
            regrowth_cooldown_value,
            easy_growth_streak_value,
            lagged_response_valid_value,
            time_value,
        ) = inputs

        def _run_step(step_operand):
            carry_in, carry_dot_in = step_operand

            def _run_accepted(_):
                return jax.jvp(
                    lambda c: _accepted_attempt(
                        c,
                        dt_value=dt_value,
                        next_dt_value=next_dt_value,
                        recent_reject_count_value=recent_reject_count_value,
                        regrowth_cooldown_value=regrowth_cooldown_value,
                        easy_growth_streak_value=easy_growth_streak_value,
                        lagged_response_valid_value=lagged_response_valid_value,
                    ),
                    (carry_in,),
                    (carry_dot_in,),
                )

            def _run_rejected(_):
                return jax.jvp(
                    lambda c: _rejected_attempt(
                        c,
                        dt_value=dt_value,
                        next_dt_value=next_dt_value,
                        recent_reject_count_value=recent_reject_count_value,
                        regrowth_cooldown_value=regrowth_cooldown_value,
                        easy_growth_streak_value=easy_growth_streak_value,
                        lagged_response_valid_value=lagged_response_valid_value,
                    ),
                    (carry_in,),
                    (carry_dot_in,),
                )

            return jax.lax.cond(accepted, _run_accepted, _run_rejected, operand=None)

        def _skip_step(step_operand):
            return step_operand

        next_carry, next_carry_dot = jax.lax.cond(
            active,
            _run_step,
            _skip_step,
            operand=(carry, carry_dot),
        )
        accepted_seen_next = accepted_seen + jnp.where(jnp.logical_and(active, accepted), jnp.asarray(1, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32))
        just_accepted_index = accepted_seen_next - 1
        should_sample = jnp.logical_and(
            jnp.logical_and(active, accepted),
            jnp.logical_and(sample_write_idx < sample_count, just_accepted_index == sample_indices_arr[sample_write_idx]),
        )

        def _write_sample(_):
            times_next = sampled_times.at[sample_write_idx].set(time_value)
            tangents_next = sampled_tangents.at[sample_write_idx].set(next_carry_dot.y)
            return sample_write_idx + 1, times_next, tangents_next

        sample_write_idx_next, sampled_times_next, sampled_tangents_next = jax.lax.cond(
            should_sample,
            _write_sample,
            lambda _: (sample_write_idx, sampled_times, sampled_tangents),
            operand=None,
        )
        return (
            next_carry,
            next_carry_dot,
            accepted_seen_next,
            sample_write_idx_next,
            sampled_times_next,
            sampled_tangents_next,
        ), None

    init_state = (
        carry0,
        carry0_dot,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        zero_times,
        zero_tangents,
    )
    final_state, _ = jax.lax.scan(
        _scan_body,
        init_state,
        (
            active_mask,
            accepted_mask,
            attempted_dts,
            next_dts,
            next_recent_reject_count,
            next_regrowth_cooldown,
            next_easy_growth_streak,
            next_lagged_response_valid,
            step_ts,
        ),
    )
    return {
        "sampled_times": final_state[4],
        "sampled_state_tangents": final_state[5],
        "sampled_indices": sample_indices,
    }


def _sampled_fixed_dt_state_tangent_trajectory(
    *,
    execution_context,
    carry0,
    carry0_dot,
    dt_sequence,
    sample_every: int,
):
    total_steps = int(np.asarray(jax.device_get(dt_sequence)).shape[0])
    sample_indices = _sample_accepted_step_indices(total_steps, sample_every)
    sample_count = len(sample_indices)
    if sample_count == 0:
        return {
            "sampled_times": jnp.zeros((0,), dtype=execution_context.dtype),
            "sampled_state_tangents": jnp.zeros((0, carry0.y.shape[0]), dtype=execution_context.dtype),
            "sampled_indices": (),
        }

    sample_indices_arr = jnp.asarray(sample_indices, dtype=jnp.int32)
    dt_sequence = jnp.asarray(dt_sequence, dtype=execution_context.dtype)
    cumulative_times = carry0.t + jnp.cumsum(dt_sequence)
    zero_tangents = jnp.zeros((sample_count, carry0.y.shape[0]), dtype=execution_context.dtype)
    zero_times = jnp.zeros((sample_count,), dtype=execution_context.dtype)

    def _accepted_attempt(
        carry_value,
        *,
        dt_value,
    ):
        carry_for_step = dataclasses.replace(carry_value, dt=dt_value)
        attempt_result = _execute_radau_accepted_step_attempt(
            execution_context.kernel_context,
            execution_context.physics_context,
            _radau_carry_with_forward_only_jvp_fields(carry_for_step),
            _RadauAcceptedStepAttemptContext(
                t_final=carry_value.t + dt_value,
                use_transport_lagged_response=jnp.asarray(execution_context.kernel_context.use_transport_lagged_response),
            ),
        )
        project_flat = execution_context.physics_context.project_flat
        accepted_y = project_flat(attempt_result.trial_y) if project_flat is not None else None
        if accepted_y is None:
            accepted_y = attempt_result.trial_y
        return dataclasses.replace(
            attempt_result.carry_after_attempt,
            t=carry_value.t + dt_value,
            y=accepted_y,
            dt=dt_value,
            prev_error=jnp.maximum(
                attempt_result.err_norm,
                jnp.asarray(1.0e-12, dtype=execution_context.dtype),
            ),
            prev_stages=attempt_result.stage_history,
            prev_dt=dt_value,
            recent_reject_count=jnp.asarray(0, dtype=jnp.int32),
            regrowth_cooldown=jnp.asarray(0, dtype=jnp.int32),
            easy_growth_streak=jnp.asarray(0, dtype=jnp.int32),
            lagged_response_valid=jnp.asarray(execution_context.kernel_context.use_transport_lagged_response),
            jacobian=attempt_result.jacobian_out,
            cache_valid=attempt_result.cache_valid_out,
            cache_dt=attempt_result.cache_dt_out,
            cache_age=attempt_result.cache_age_out,
            real_lu=attempt_result.real_lu_out,
            real_piv=attempt_result.real_piv_out,
            complex_lu=attempt_result.complex_lu_out,
            complex_piv=attempt_result.complex_piv_out,
            prev_theta_final=attempt_result.theta_final,
            prev_newton_iter_count=attempt_result.newton_iter_count,
        )

    def _scan_body(scan_state, inputs):
        carry, carry_dot, step_index, sample_write_idx, sampled_times, sampled_tangents = scan_state
        dt_value, time_value = inputs

        next_carry, next_carry_dot = jax.jvp(
            lambda c: _accepted_attempt(c, dt_value=dt_value),
            (carry,),
            (carry_dot,),
        )
        should_sample = jnp.logical_and(
            sample_write_idx < sample_count,
            step_index == sample_indices_arr[sample_write_idx],
        )

        def _write_sample(_):
            times_next = sampled_times.at[sample_write_idx].set(time_value)
            tangents_next = sampled_tangents.at[sample_write_idx].set(next_carry_dot.y)
            return sample_write_idx + 1, times_next, tangents_next

        sample_write_idx_next, sampled_times_next, sampled_tangents_next = jax.lax.cond(
            should_sample,
            _write_sample,
            lambda _: (sample_write_idx, sampled_times, sampled_tangents),
            operand=None,
        )
        return (
            next_carry,
            next_carry_dot,
            step_index + jnp.asarray(1, dtype=jnp.int32),
            sample_write_idx_next,
            sampled_times_next,
            sampled_tangents_next,
        ), None

    init_state = (
        carry0,
        carry0_dot,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        zero_times,
        zero_tangents,
    )
    final_state, _ = jax.lax.scan(
        _scan_body,
        init_state,
        (dt_sequence, cumulative_times),
    )
    return {
        "sampled_times": final_state[4],
        "sampled_state_tangents": final_state[5],
        "sampled_indices": sample_indices,
    }


def _manual_sampled_realized_trace_state_tangent_trajectory(
    *,
    execution_context,
    carry0,
    carry0_dot,
    trace,
    sample_every: int,
    use_custom: bool,
):
    accepted_mask_np = np.asarray(jax.device_get(trace.accepted_mask), dtype=bool)
    active_mask_np = np.asarray(jax.device_get(trace.active_mask), dtype=bool)
    attempted_dts_np = np.asarray(jax.device_get(trace.attempted_dts), dtype=float)
    next_dts_np = np.asarray(jax.device_get(trace.next_dts), dtype=float)
    next_recent_reject_count_np = np.asarray(jax.device_get(trace.next_recent_reject_count))
    next_regrowth_cooldown_np = np.asarray(jax.device_get(trace.next_regrowth_cooldown))
    next_easy_growth_streak_np = np.asarray(jax.device_get(trace.next_easy_growth_streak))
    next_lagged_response_valid_np = np.asarray(jax.device_get(trace.next_lagged_response_valid))
    step_ts_np = np.asarray(jax.device_get(trace.step_ts), dtype=float)

    sample_indices = _sample_accepted_step_indices(int(np.sum(accepted_mask_np)), sample_every)
    if not sample_indices:
        return {
            "sampled_times": jnp.zeros((0,), dtype=execution_context.dtype),
            "sampled_state_tangents": jnp.zeros((0, carry0.y.shape[0]), dtype=execution_context.dtype),
            "sampled_indices": (),
        }

    attempt_fn = _execute_radau_accepted_step_attempt_autodiff if use_custom else _execute_radau_accepted_step_attempt
    project_flat = execution_context.physics_context.project_flat
    carry = carry0
    carry_dot = carry0_dot
    accepted_seen = 0
    sample_write_idx = 0
    sampled_times: list[float] = []
    sampled_tangents: list[np.ndarray] = []

    for idx in range(len(active_mask_np)):
        if not active_mask_np[idx]:
            continue

        accepted = bool(accepted_mask_np[idx])
        dt_value = jnp.asarray(attempted_dts_np[idx], dtype=execution_context.dtype)
        next_dt_value = jnp.asarray(next_dts_np[idx], dtype=execution_context.dtype)
        recent_reject_count_value = jnp.asarray(next_recent_reject_count_np[idx])
        regrowth_cooldown_value = jnp.asarray(next_regrowth_cooldown_np[idx])
        easy_growth_streak_value = jnp.asarray(next_easy_growth_streak_np[idx])
        lagged_response_valid_value = jnp.asarray(next_lagged_response_valid_np[idx])

        def _accepted_attempt(carry_value):
            carry_for_step = dataclasses.replace(carry_value, dt=dt_value)
            attempt_result = attempt_fn(
                execution_context.kernel_context,
                execution_context.physics_context,
                _radau_carry_with_forward_only_jvp_fields(carry_for_step),
                execution_context.attempt_context,
            )
            accepted_y = project_flat(attempt_result.trial_y) if project_flat is not None else None
            if accepted_y is None:
                accepted_y = attempt_result.trial_y
            return dataclasses.replace(
                attempt_result.carry_after_attempt,
                t=carry_value.t + dt_value,
                y=accepted_y,
                dt=next_dt_value,
                prev_error=jnp.maximum(
                    attempt_result.err_norm,
                    jnp.asarray(1.0e-12, dtype=execution_context.dtype),
                ),
                prev_stages=attempt_result.stage_history,
                prev_dt=dt_value,
                recent_reject_count=recent_reject_count_value,
                regrowth_cooldown=regrowth_cooldown_value,
                easy_growth_streak=easy_growth_streak_value,
                lagged_response_valid=lagged_response_valid_value,
                jacobian=attempt_result.jacobian_out,
                cache_valid=attempt_result.cache_valid_out,
                cache_dt=attempt_result.cache_dt_out,
                cache_age=attempt_result.cache_age_out,
                real_lu=attempt_result.real_lu_out,
                real_piv=attempt_result.real_piv_out,
                complex_lu=attempt_result.complex_lu_out,
                complex_piv=attempt_result.complex_piv_out,
                prev_theta_final=attempt_result.theta_final,
                prev_newton_iter_count=attempt_result.newton_iter_count,
            )

        def _rejected_attempt(carry_value):
            carry_for_step = dataclasses.replace(jax.lax.stop_gradient(carry_value), dt=dt_value)
            attempt_result = attempt_fn(
                execution_context.kernel_context,
                execution_context.physics_context,
                _radau_carry_with_forward_only_jvp_fields(carry_for_step),
                execution_context.attempt_context,
            )
            return dataclasses.replace(
                carry_value,
                dt=next_dt_value,
                recent_reject_count=recent_reject_count_value,
                regrowth_cooldown=regrowth_cooldown_value,
                easy_growth_streak=easy_growth_streak_value,
                lagged_response_cache=jax.lax.stop_gradient(attempt_result.carry_after_attempt.lagged_response_cache),
                lagged_response_valid=lagged_response_valid_value,
                lagged_reference_y=jax.lax.stop_gradient(attempt_result.carry_after_attempt.lagged_reference_y),
                jacobian=jax.lax.stop_gradient(attempt_result.jacobian_out),
                cache_valid=jax.lax.stop_gradient(attempt_result.cache_valid_out),
                cache_dt=jax.lax.stop_gradient(attempt_result.cache_dt_out),
                cache_age=jax.lax.stop_gradient(attempt_result.cache_age_out),
                real_lu=jax.lax.stop_gradient(attempt_result.real_lu_out),
                real_piv=jax.lax.stop_gradient(attempt_result.real_piv_out),
                complex_lu=jax.lax.stop_gradient(attempt_result.complex_lu_out),
                complex_piv=jax.lax.stop_gradient(attempt_result.complex_piv_out),
                prev_theta_final=jax.lax.stop_gradient(attempt_result.theta_final),
                prev_newton_iter_count=jax.lax.stop_gradient(attempt_result.newton_iter_count),
            )

        step_fn = _accepted_attempt if accepted else _rejected_attempt
        carry, carry_dot = jax.jvp(step_fn, (carry,), (carry_dot,))

        if accepted:
            if sample_write_idx < len(sample_indices) and accepted_seen == sample_indices[sample_write_idx]:
                sampled_times.append(float(step_ts_np[idx]))
                sampled_tangents.append(np.asarray(jax.device_get(carry_dot.y), dtype=float))
                sample_write_idx += 1
            accepted_seen += 1

    if sampled_tangents:
        tangent_array = jnp.asarray(np.stack(sampled_tangents, axis=0), dtype=execution_context.dtype)
        times_array = jnp.asarray(np.asarray(sampled_times, dtype=float), dtype=execution_context.dtype)
    else:
        tangent_array = jnp.zeros((0, carry0.y.shape[0]), dtype=execution_context.dtype)
        times_array = jnp.zeros((0,), dtype=execution_context.dtype)

    return {
        "sampled_times": times_array,
        "sampled_state_tangents": tangent_array,
        "sampled_indices": sample_indices,
    }


def _manual_realized_trace_state_tangent_checkpoints(
    *,
    execution_context,
    carry0,
    carry0_dot,
    trace,
    accepted_checkpoints: tuple[int, ...],
    use_custom: bool,
):
    checkpoints = tuple(sorted({int(v) for v in accepted_checkpoints if int(v) >= 1}))
    if not checkpoints:
        return {"times": jnp.zeros((0,), dtype=execution_context.dtype), "state_tangents": jnp.zeros((0, carry0.y.shape[0]), dtype=execution_context.dtype), "accepted_indices": ()}

    accepted_mask_np = np.asarray(jax.device_get(trace.accepted_mask), dtype=bool)
    active_mask_np = np.asarray(jax.device_get(trace.active_mask), dtype=bool)
    attempted_dts_np = np.asarray(jax.device_get(trace.attempted_dts), dtype=float)
    next_dts_np = np.asarray(jax.device_get(trace.next_dts), dtype=float)
    next_recent_reject_count_np = np.asarray(jax.device_get(trace.next_recent_reject_count))
    next_regrowth_cooldown_np = np.asarray(jax.device_get(trace.next_regrowth_cooldown))
    next_easy_growth_streak_np = np.asarray(jax.device_get(trace.next_easy_growth_streak))
    next_lagged_response_valid_np = np.asarray(jax.device_get(trace.next_lagged_response_valid))
    step_ts_np = np.asarray(jax.device_get(trace.step_ts), dtype=float)

    attempt_fn = _execute_radau_accepted_step_attempt_autodiff if use_custom else _execute_radau_accepted_step_attempt
    project_flat = execution_context.physics_context.project_flat
    carry = carry0
    carry_dot = carry0_dot
    accepted_seen = 0
    checkpoint_ptr = 0
    sampled_times: list[float] = []
    sampled_tangents: list[np.ndarray] = []

    def _advance_one_attempt(
        carry_value,
        carry_dot_value,
        accepted_value,
        dt_value,
        next_dt_value,
        recent_reject_count_value,
        regrowth_cooldown_value,
        easy_growth_streak_value,
        lagged_response_valid_value,
    ):
        def _accepted_branch(_):
            def _accepted_attempt(carry_inner):
                carry_for_step = dataclasses.replace(carry_inner, dt=dt_value)
                attempt_result = attempt_fn(
                    execution_context.kernel_context,
                    execution_context.physics_context,
                    _radau_carry_with_forward_only_jvp_fields(carry_for_step),
                    execution_context.attempt_context,
                )
                accepted_y = project_flat(attempt_result.trial_y) if project_flat is not None else None
                if accepted_y is None:
                    accepted_y = attempt_result.trial_y
                return dataclasses.replace(
                    attempt_result.carry_after_attempt,
                    t=carry_inner.t + dt_value,
                    y=accepted_y,
                    dt=next_dt_value,
                    prev_error=jnp.maximum(
                        attempt_result.err_norm,
                        jnp.asarray(1.0e-12, dtype=execution_context.dtype),
                    ),
                    prev_stages=attempt_result.stage_history,
                    prev_dt=dt_value,
                    recent_reject_count=recent_reject_count_value,
                    regrowth_cooldown=regrowth_cooldown_value,
                    easy_growth_streak=easy_growth_streak_value,
                    lagged_response_valid=lagged_response_valid_value,
                    jacobian=attempt_result.jacobian_out,
                    cache_valid=attempt_result.cache_valid_out,
                    cache_dt=attempt_result.cache_dt_out,
                    cache_age=attempt_result.cache_age_out,
                    real_lu=attempt_result.real_lu_out,
                    real_piv=attempt_result.real_piv_out,
                    complex_lu=attempt_result.complex_lu_out,
                    complex_piv=attempt_result.complex_piv_out,
                    prev_theta_final=attempt_result.theta_final,
                    prev_newton_iter_count=attempt_result.newton_iter_count,
                )

            return jax.jvp(_accepted_attempt, (carry_value,), (carry_dot_value,))

        def _rejected_branch(_):
            def _rejected_attempt(carry_inner):
                carry_for_step = dataclasses.replace(jax.lax.stop_gradient(carry_inner), dt=dt_value)
                attempt_result = attempt_fn(
                    execution_context.kernel_context,
                    execution_context.physics_context,
                    _radau_carry_with_forward_only_jvp_fields(carry_for_step),
                    execution_context.attempt_context,
                )
                return dataclasses.replace(
                    carry_inner,
                    dt=next_dt_value,
                    recent_reject_count=recent_reject_count_value,
                    regrowth_cooldown=regrowth_cooldown_value,
                    easy_growth_streak=easy_growth_streak_value,
                    lagged_response_cache=jax.lax.stop_gradient(attempt_result.carry_after_attempt.lagged_response_cache),
                    lagged_response_valid=lagged_response_valid_value,
                    lagged_reference_y=jax.lax.stop_gradient(attempt_result.carry_after_attempt.lagged_reference_y),
                    jacobian=jax.lax.stop_gradient(attempt_result.jacobian_out),
                    cache_valid=jax.lax.stop_gradient(attempt_result.cache_valid_out),
                    cache_dt=jax.lax.stop_gradient(attempt_result.cache_dt_out),
                    cache_age=jax.lax.stop_gradient(attempt_result.cache_age_out),
                    real_lu=jax.lax.stop_gradient(attempt_result.real_lu_out),
                    real_piv=jax.lax.stop_gradient(attempt_result.real_piv_out),
                    complex_lu=jax.lax.stop_gradient(attempt_result.complex_lu_out),
                    complex_piv=jax.lax.stop_gradient(attempt_result.complex_piv_out),
                    prev_theta_final=jax.lax.stop_gradient(attempt_result.theta_final),
                    prev_newton_iter_count=jax.lax.stop_gradient(attempt_result.newton_iter_count),
                )

            return jax.jvp(_rejected_attempt, (carry_value,), (carry_dot_value,))

        return jax.lax.cond(accepted_value, _accepted_branch, _rejected_branch, operand=None)

    compiled_advance_one_attempt = jax.jit(_advance_one_attempt)

    for idx in range(len(active_mask_np)):
        if not active_mask_np[idx]:
            continue
        if checkpoint_ptr >= len(checkpoints):
            break

        accepted = bool(accepted_mask_np[idx])
        dt_value = jnp.asarray(attempted_dts_np[idx], dtype=execution_context.dtype)
        next_dt_value = jnp.asarray(next_dts_np[idx], dtype=execution_context.dtype)
        recent_reject_count_value = jnp.asarray(next_recent_reject_count_np[idx])
        regrowth_cooldown_value = jnp.asarray(next_regrowth_cooldown_np[idx])
        easy_growth_streak_value = jnp.asarray(next_easy_growth_streak_np[idx])
        lagged_response_valid_value = jnp.asarray(next_lagged_response_valid_np[idx])
        carry, carry_dot = compiled_advance_one_attempt(
            carry,
            carry_dot,
            jnp.asarray(accepted),
            dt_value,
            next_dt_value,
            recent_reject_count_value,
            regrowth_cooldown_value,
            easy_growth_streak_value,
            lagged_response_valid_value,
        )

        if accepted:
            accepted_seen += 1
            if accepted_seen == checkpoints[checkpoint_ptr]:
                sampled_times.append(float(step_ts_np[idx]))
                sampled_tangents.append(np.asarray(jax.device_get(carry_dot.y), dtype=float))
                checkpoint_ptr += 1

    if sampled_tangents:
        tangent_array = jnp.asarray(np.stack(sampled_tangents, axis=0), dtype=execution_context.dtype)
        times_array = jnp.asarray(np.asarray(sampled_times, dtype=float), dtype=execution_context.dtype)
    else:
        tangent_array = jnp.zeros((0, carry0.y.shape[0]), dtype=execution_context.dtype)
        times_array = jnp.zeros((0,), dtype=execution_context.dtype)
    return {"times": times_array, "state_tangents": tangent_array, "accepted_indices": checkpoints}


def _adaptive_rollout_objectives_for_parameter_on_windowed_frozen_trace(
    parameter_value,
    *,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    parameter_name: str,
    baseline_trace,
    replay_mode: str = "attempt",
    accepted_window_size: int = 10,
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

    total_accepted = int(np.asarray(jax.device_get(jnp.sum(jnp.asarray(baseline_trace.accepted_mask, dtype=jnp.int32)))))
    current_carry = prepared_rollout.initial_carry
    window_summaries: list[dict[str, Any]] = []
    last_replay = None
    first_failing_window_debug = None
    for accepted_start in range(0, total_accepted, int(accepted_window_size)):
        carry_before_window = current_carry
        window_trace = _slice_rollout_trace_by_accepted_window(
            baseline_trace,
            accepted_start=accepted_start,
            accepted_count=min(int(accepted_window_size), total_accepted - accepted_start),
        )
        replay = _radau_run_prepared_on_realized_trace(
            prepared_rollout,
            execution_context,
            window_trace,
            replay_mode=replay_mode,
            carry0=current_carry,
        )
        last_replay = replay
        current_carry = replay["final_carry"]
        window_objectives = _objective_vector(replay["final_state"], runtime)
        window_objectives_np = np.asarray(jax.device_get(window_objectives), dtype=float)
        window_state_finite = _tree_all_finite(replay["final_state"])
        window_summaries.append(
            {
                "accepted_start": int(accepted_start),
                "accepted_count": int(np.sum(np.asarray(jax.device_get(window_trace.accepted_mask), dtype=bool))),
                "attempt_count": int(np.sum(np.asarray(jax.device_get(window_trace.active_mask), dtype=bool))),
                "state_finite": window_state_finite,
            }
        )
        if not window_state_finite:
            first_failing_window_debug = {
                "accepted_start": int(accepted_start),
                "accepted_count": int(np.sum(np.asarray(jax.device_get(window_trace.accepted_mask), dtype=bool))),
                "attempt_count": int(np.sum(np.asarray(jax.device_get(window_trace.active_mask), dtype=bool))),
                "replay_mode": str(replay_mode),
                "nonfinite_debug": _frozen_replay_nonfinite_debug(
                    replay,
                    window_trace,
                    objectives_np=window_objectives_np,
                ),
            }
            if str(replay_mode).strip().lower() == "attempt":
                accepted_replay = _radau_run_prepared_on_realized_trace(
                    prepared_rollout,
                    execution_context,
                    window_trace,
                    replay_mode="accepted",
                    carry0=carry_before_window,
                )
                accepted_objectives_np = np.asarray(
                    jax.device_get(_objective_vector(accepted_replay["final_state"], runtime)),
                    dtype=float,
                )
                first_failing_window_debug["accepted_mode_debug"] = _frozen_replay_nonfinite_debug(
                    accepted_replay,
                    window_trace,
                    objectives_np=accepted_objectives_np,
                )
            break

    if last_replay is None:
        final_state = prepared_rollout.physics_context.unpack_flat(current_carry.y)
        replay_out = {
            "final_state": final_state,
            "final_carry": current_carry,
            "replay_mode": str(replay_mode),
            "window_summaries": window_summaries,
        }
    else:
        replay_out = dict(last_replay)
        replay_out["window_summaries"] = window_summaries
        replay_out["first_failing_window_debug"] = first_failing_window_debug

    return _objective_vector(replay_out["final_state"], runtime), replay_out


def _adaptive_rollout_nan_debug_for_parameter(
    parameter_value,
    *,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    parameter_name: str,
    debug_mode: str = "minimal",
    include_one_step_compare: bool = False,
):
    def _zero_optional_pytree(tree):
        return jax.tree_util.tree_map(
            lambda x: None if x is None else jnp.zeros_like(x),
            tree,
            is_leaf=lambda x: x is None,
        )

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
    max_total_steps = int(max(1, getattr(solver, "max_steps", 1)))
    stop_after_accepted_steps = getattr(solver, "stop_after_accepted_steps", None)
    rollout = _radau_adaptive_final_state_rollout(
        execution_context,
        prepared_rollout.initial_carry,
        max_total_steps=max_total_steps,
        stop_after_accepted_steps=stop_after_accepted_steps,
    )

    def _initial_carry_for_parameter(p):
        state_p = _parameterized_initial_state(
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            geometry=runtime.geometry,
            n_species=runtime.species.number_species,
            parameter_name=parameter_name,
            parameter_value=p,
        )
        prepared_components_p = prepare_transport_solver_components(config, runtime, state_p)
        solve_vector_field_p = prepared_components_p["solve_vector_field"]
        prepared_rollout_p = _build_prepared_radau_accepted_rollout(
            solver=solver,
            state=state_p,
            vector_field=solve_vector_field_p,
            species=runtime.species,
        )
        return prepared_rollout_p.initial_carry

    baseline_value = jnp.asarray(parameter_value)
    _, carry0_dot = jax.jvp(
        _initial_carry_for_parameter,
        (baseline_value,),
        (jnp.asarray(1.0, dtype=baseline_value.dtype),),
    )
    debug = _radau_debug_realized_attempt_replay(
        execution_context,
        prepared_rollout.initial_carry,
        carry0_dot,
        rollout.trace,
    )
    lagged_cache_zeroed_debug = None
    prev_stages_zeroed_debug = None
    prev_stages_and_lagged_zeroed_debug = None
    y_zeroed_debug = None
    prev_error_zeroed_debug = None
    y_and_prev_error_zeroed_debug = None
    all_tangents_zeroed_debug = None
    if debug_mode == "exhaustive":
        all_tangents_zeroed_debug = _radau_debug_realized_attempt_replay(
            execution_context,
            prepared_rollout.initial_carry,
            jax.tree_util.tree_map(
                lambda x: None if x is None else jnp.zeros_like(x),
                carry0_dot,
                is_leaf=lambda x: x is None,
            ),
            rollout.trace,
        )
        lagged_cache_zeroed_debug = _radau_debug_realized_attempt_replay(
            execution_context,
            prepared_rollout.initial_carry,
            dataclasses.replace(
                carry0_dot,
                lagged_response_cache=_zero_optional_pytree(carry0_dot.lagged_response_cache),
            ),
            rollout.trace,
        )
        prev_stages_zeroed_debug = _radau_debug_realized_attempt_replay(
            execution_context,
            prepared_rollout.initial_carry,
            dataclasses.replace(
                carry0_dot,
                prev_stages=jnp.zeros_like(carry0_dot.prev_stages),
            ),
            rollout.trace,
        )
        prev_stages_and_lagged_zeroed_debug = _radau_debug_realized_attempt_replay(
            execution_context,
            prepared_rollout.initial_carry,
            dataclasses.replace(
                carry0_dot,
                prev_stages=jnp.zeros_like(carry0_dot.prev_stages),
                lagged_response_cache=_zero_optional_pytree(carry0_dot.lagged_response_cache),
            ),
            rollout.trace,
        )
        y_zeroed_debug = _radau_debug_realized_attempt_replay(
            execution_context,
            prepared_rollout.initial_carry,
            dataclasses.replace(
                carry0_dot,
                y=jax.tree_util.tree_map(jnp.zeros_like, carry0_dot.y),
            ),
            rollout.trace,
        )
        prev_error_zeroed_debug = _radau_debug_realized_attempt_replay(
            execution_context,
            prepared_rollout.initial_carry,
            dataclasses.replace(
                carry0_dot,
                prev_error=jnp.zeros_like(carry0_dot.prev_error),
            ),
            rollout.trace,
        )
        y_and_prev_error_zeroed_debug = _radau_debug_realized_attempt_replay(
            execution_context,
            prepared_rollout.initial_carry,
            dataclasses.replace(
                carry0_dot,
                y=jax.tree_util.tree_map(jnp.zeros_like, carry0_dot.y),
                prev_error=jnp.zeros_like(carry0_dot.prev_error),
            ),
            rollout.trace,
        )
    local_attempt_window: list[dict[str, Any]] = []
    first_bad_index = int(debug.first_bad_index)
    if first_bad_index >= 0:
        start = max(0, first_bad_index - 2)
        stop = min(int(np.asarray(jax.device_get(rollout.attempt_count))), first_bad_index + 3)
        attempted_dts = np.asarray(jax.device_get(rollout.trace.attempted_dts[start:stop]), dtype=float)
        next_dts = np.asarray(jax.device_get(rollout.trace.next_dts[start:stop]), dtype=float)
        accepted_mask = np.asarray(jax.device_get(rollout.trace.accepted_mask[start:stop]), dtype=bool)
        err_norms = np.asarray(jax.device_get(rollout.trace.err_norms[start:stop]), dtype=float)
        theta_finals = np.asarray(jax.device_get(rollout.trace.theta_finals[start:stop]), dtype=float)
        newton_iter_counts = np.asarray(jax.device_get(rollout.trace.newton_iter_counts[start:stop]), dtype=np.int32)
        cache_valid_next = np.asarray(jax.device_get(rollout.trace.cache_valid_next[start:stop]), dtype=bool)
        for idx in range(start, stop):
            local_idx = idx - start
            local_attempt_window.append(
                {
                    "index": int(idx),
                    "accepted": bool(accepted_mask[local_idx]),
                    "attempted_dt": float(attempted_dts[local_idx]),
                    "next_dt": float(next_dts[local_idx]),
                    "err_norm": float(err_norms[local_idx]),
                    "theta_final": float(theta_finals[local_idx]),
                    "newton_iter_count": int(newton_iter_counts[local_idx]),
                    "cache_valid_next": bool(cache_valid_next[local_idx]),
                    "tangent_finite": bool(debug.tangent_finite_mask[idx]),
                    "dt_dot_abs": float(debug.dt_dot_abs[idx]),
                    "prev_error_dot_abs": float(debug.prev_error_dot_abs[idx]),
                    "density_dot_max_abs": float(debug.density_dot_max_abs[idx]),
                    "pressure_dot_max_abs": float(debug.pressure_dot_max_abs[idx]),
                    "er_dot_max_abs": float(debug.er_dot_max_abs[idx]),
                    "y_dot_max_abs": float(debug.y_dot_max_abs[idx]),
                    "prev_stages_dot_max_abs": float(debug.prev_stages_dot_max_abs[idx]),
                    "lagged_response_cache_dot_max_abs": float(debug.lagged_response_cache_dot_max_abs[idx]),
                    "jacobian_dot_max_abs": float(debug.jacobian_dot_max_abs[idx]),
                    "real_lu_dot_max_abs": float(debug.real_lu_dot_max_abs[idx]),
                    "complex_lu_dot_max_abs": float(debug.complex_lu_dot_max_abs[idx]),
                }
            )
    result = {
        "first_bad_index": first_bad_index,
        "first_bad_was_accepted": bool(debug.first_bad_was_accepted),
        "first_bad_dt": float(debug.first_bad_dt),
        "final_tangent_finite": bool(debug.final_tangent_finite),
        "tangent_finite_mask": list(debug.tangent_finite_mask),
        "local_attempt_window": local_attempt_window,
        "debug_mode": debug_mode,
    }
    if include_one_step_compare:
        zero_tangent_one_step = _radau_debug_compare_zero_tangent_one_step(
            execution_context,
            prepared_rollout.initial_carry,
            rollout.trace,
            target_attempt_index=first_bad_index,
        )
        result["zero_tangent_one_step"] = {
            "target_attempt_index": int(zero_tangent_one_step.target_attempt_index),
            "target_was_accepted": bool(zero_tangent_one_step.target_was_accepted),
            "trial_dt": float(zero_tangent_one_step.trial_dt),
            "custom_trial_y_max_abs": float(zero_tangent_one_step.custom_trial_y_max_abs),
            "custom_stage_history_max_abs": float(zero_tangent_one_step.custom_stage_history_max_abs),
            "custom_finite": bool(zero_tangent_one_step.custom_finite),
            "direct_trial_y_max_abs": float(zero_tangent_one_step.direct_trial_y_max_abs),
            "direct_stage_history_max_abs": float(zero_tangent_one_step.direct_stage_history_max_abs),
            "direct_finite": bool(zero_tangent_one_step.direct_finite),
        }
    if all_tangents_zeroed_debug is not None:
        result["all_tangents_zeroed_debug"] = {
            "first_bad_index": int(all_tangents_zeroed_debug.first_bad_index),
            "first_bad_was_accepted": bool(all_tangents_zeroed_debug.first_bad_was_accepted),
            "first_bad_dt": float(all_tangents_zeroed_debug.first_bad_dt),
            "final_tangent_finite": bool(all_tangents_zeroed_debug.final_tangent_finite),
        }
    if lagged_cache_zeroed_debug is not None:
        result["lagged_cache_zeroed_debug"] = {
            "first_bad_index": int(lagged_cache_zeroed_debug.first_bad_index),
            "first_bad_was_accepted": bool(lagged_cache_zeroed_debug.first_bad_was_accepted),
            "first_bad_dt": float(lagged_cache_zeroed_debug.first_bad_dt),
            "final_tangent_finite": bool(lagged_cache_zeroed_debug.final_tangent_finite),
        }
    if prev_stages_zeroed_debug is not None:
        result["prev_stages_zeroed_debug"] = {
            "first_bad_index": int(prev_stages_zeroed_debug.first_bad_index),
            "first_bad_was_accepted": bool(prev_stages_zeroed_debug.first_bad_was_accepted),
            "first_bad_dt": float(prev_stages_zeroed_debug.first_bad_dt),
            "final_tangent_finite": bool(prev_stages_zeroed_debug.final_tangent_finite),
        }
    if prev_stages_and_lagged_zeroed_debug is not None:
        result["prev_stages_and_lagged_zeroed_debug"] = {
            "first_bad_index": int(prev_stages_and_lagged_zeroed_debug.first_bad_index),
            "first_bad_was_accepted": bool(prev_stages_and_lagged_zeroed_debug.first_bad_was_accepted),
            "first_bad_dt": float(prev_stages_and_lagged_zeroed_debug.first_bad_dt),
            "final_tangent_finite": bool(prev_stages_and_lagged_zeroed_debug.final_tangent_finite),
        }
    if y_zeroed_debug is not None:
        result["y_zeroed_debug"] = {
            "first_bad_index": int(y_zeroed_debug.first_bad_index),
            "first_bad_was_accepted": bool(y_zeroed_debug.first_bad_was_accepted),
            "first_bad_dt": float(y_zeroed_debug.first_bad_dt),
            "final_tangent_finite": bool(y_zeroed_debug.final_tangent_finite),
        }
    if prev_error_zeroed_debug is not None:
        result["prev_error_zeroed_debug"] = {
            "first_bad_index": int(prev_error_zeroed_debug.first_bad_index),
            "first_bad_was_accepted": bool(prev_error_zeroed_debug.first_bad_was_accepted),
            "first_bad_dt": float(prev_error_zeroed_debug.first_bad_dt),
            "final_tangent_finite": bool(prev_error_zeroed_debug.final_tangent_finite),
        }
    if y_and_prev_error_zeroed_debug is not None:
        result["y_and_prev_error_zeroed_debug"] = {
            "first_bad_index": int(y_and_prev_error_zeroed_debug.first_bad_index),
            "first_bad_was_accepted": bool(y_and_prev_error_zeroed_debug.first_bad_was_accepted),
            "first_bad_dt": float(y_and_prev_error_zeroed_debug.first_bad_dt),
            "final_tangent_finite": bool(y_and_prev_error_zeroed_debug.final_tangent_finite),
        }
    return result


def _fd_step(baseline_value: float, *, rel_step: float, abs_step: float) -> float:
    return max(abs_step, rel_step * max(abs(baseline_value), 1.0))


def _accepted_count(mask) -> int | None:
    if mask is None:
        return None
    arr = np.asarray(jax.device_get(mask))
    return int(np.sum(arr))


def _result_scalar(result: dict[str, Any], key: str, *, dtype=None):
    value = result.get(key)
    if value is None:
        return None
    arr = np.asarray(jax.device_get(value))
    if arr.shape == ():
        scalar = arr.item()
        return dtype(scalar) if dtype is not None else scalar
    return arr


def _saved_rollout_signature(result: dict[str, Any]) -> dict[str, Any]:
    accepted_mask = result.get("accepted_mask")
    ts = result.get("ts")
    dts = result.get("dts")
    if accepted_mask is None or ts is None or dts is None:
        return {
            "saved_times": None,
            "saved_step_sizes": None,
        }

    mask_arr = np.asarray(jax.device_get(accepted_mask), dtype=bool)
    ts_arr = np.asarray(jax.device_get(ts), dtype=float)
    dts_arr = np.asarray(jax.device_get(dts), dtype=float)
    valid = mask_arr
    return {
        "saved_times": ts_arr[valid].tolist(),
        "saved_step_sizes": dts_arr[valid].tolist(),
    }


def _sequence_allclose(seq_a, seq_b, *, rtol: float = 1.0e-10, atol: float = 1.0e-12) -> bool | None:
    if seq_a is None or seq_b is None:
        return None
    arr_a = np.asarray(seq_a, dtype=float)
    arr_b = np.asarray(seq_b, dtype=float)
    if arr_a.shape != arr_b.shape:
        return False
    return bool(np.allclose(arr_a, arr_b, rtol=rtol, atol=atol))


def _parse_float_csv(text: str | None) -> tuple[float, ...]:
    if text is None:
        return ()
    values = []
    for chunk in str(text).split(","):
        token = chunk.strip()
        if not token:
            continue
        values.append(float(token))
    return tuple(values)


def _parse_int_csv(text: str | None) -> tuple[int, ...]:
    return tuple(int(round(v)) for v in _parse_float_csv(text))


def _result_diagnostics(result: dict[str, Any]) -> dict[str, Any]:
    accepted_mask = result.get("accepted_mask")
    failed_mask = result.get("failed_mask")
    fail_codes = result.get("fail_codes")
    rollout_signature = _saved_rollout_signature(result)
    diag = {
        "n_steps": None if result.get("n_steps") is None else int(np.asarray(jax.device_get(result["n_steps"]))),
        "accepted_count": _accepted_count(accepted_mask),
        "accepted_mask": None if accepted_mask is None else np.asarray(jax.device_get(accepted_mask), dtype=bool).tolist(),
        "failed_any": False if failed_mask is None else bool(np.any(np.asarray(jax.device_get(failed_mask), dtype=bool))),
        "fail_codes": None if fail_codes is None else np.asarray(jax.device_get(fail_codes)).tolist(),
        "saved_times": rollout_signature["saved_times"],
        "saved_step_sizes": rollout_signature["saved_step_sizes"],
        "last_attempt": {
            "accepted": _result_scalar(result, "last_attempt_accepted", dtype=bool),
            "converged": _result_scalar(result, "last_attempt_converged", dtype=bool),
            "fail_code": _result_scalar(result, "last_attempt_fail_code", dtype=int),
            "newton_iter_count": _result_scalar(result, "last_attempt_newton_iter_count", dtype=int),
            "theta_final": _result_scalar(result, "last_attempt_theta_final", dtype=float),
            "err_norm": _result_scalar(result, "last_attempt_err_norm", dtype=float),
            "final_residual_norm": _result_scalar(result, "last_attempt_final_residual_norm", dtype=float),
            "final_delta_norm": _result_scalar(result, "last_attempt_final_delta_norm", dtype=float),
            "slow_contraction": _result_scalar(result, "last_attempt_slow_contraction", dtype=bool),
            "residual_blowup": _result_scalar(result, "last_attempt_residual_blowup", dtype=bool),
            "newton_nonfinite": _result_scalar(result, "last_attempt_newton_nonfinite", dtype=bool),
        },
    }
    return diag


def _adaptive_rollout_diagnostics(rollout) -> dict[str, Any]:
    trace = rollout.trace
    accepted_mask = np.asarray(jax.device_get(trace.accepted_mask), dtype=bool)
    active_mask = np.asarray(jax.device_get(trace.active_mask), dtype=bool)
    attempted_dts = np.asarray(jax.device_get(trace.attempted_dts), dtype=float)
    next_dts = np.asarray(jax.device_get(trace.next_dts), dtype=float)
    step_ts = np.asarray(jax.device_get(trace.step_ts), dtype=float)
    err_norms = np.asarray(jax.device_get(trace.err_norms), dtype=float)
    return {
        "attempt_count": int(np.asarray(jax.device_get(rollout.attempt_count))),
        "accepted_count": int(np.asarray(jax.device_get(rollout.accepted_count))),
        "completed": bool(np.asarray(jax.device_get(rollout.completed))),
        "failed": bool(np.asarray(jax.device_get(rollout.failed))),
        "fail_code": int(np.asarray(jax.device_get(rollout.fail_code))),
        "accepted_mask": accepted_mask[active_mask].tolist(),
        "attempted_dts": attempted_dts[active_mask].tolist(),
        "next_dts": next_dts[active_mask].tolist(),
        "step_ts": step_ts[active_mask].tolist(),
        "err_norms": err_norms[active_mask].tolist(),
    }


def _adaptive_rollout_summary(rollout) -> dict[str, Any]:
    return {
        "attempt_count": int(np.asarray(jax.device_get(rollout.attempt_count))),
        "accepted_count": int(np.asarray(jax.device_get(rollout.accepted_count))),
        "completed": bool(np.asarray(jax.device_get(rollout.completed))),
        "failed": bool(np.asarray(jax.device_get(rollout.failed))),
        "fail_code": int(np.asarray(jax.device_get(rollout.fail_code))),
    }


def _tree_all_finite(tree) -> bool:
    finite = True
    for leaf in jax.tree_util.tree_leaves(tree, is_leaf=lambda x: x is None):
        if leaf is None:
            continue
        arr = np.asarray(jax.device_get(leaf))
        if np.issubdtype(arr.dtype, np.inexact):
            finite = finite and bool(np.all(np.isfinite(arr)))
    return bool(finite)


def _array_state_finite_mask(values: np.ndarray) -> np.ndarray:
    if values.ndim <= 1:
        return np.isfinite(values)
    reduce_axes = tuple(range(1, values.ndim))
    return np.all(np.isfinite(values), axis=reduce_axes)


def _truncate_rollout_trace_by_accepted_steps(trace, accepted_step_limit: int | None):
    if accepted_step_limit is None:
        return trace
    accepted_step_limit = int(accepted_step_limit)
    if accepted_step_limit <= 0:
        raise ValueError("accepted_step_limit must be positive when provided.")

    accepted_mask = jnp.asarray(trace.accepted_mask, dtype=bool)
    active_mask = jnp.asarray(trace.active_mask, dtype=bool)
    accepted_prefix_count = jnp.cumsum(accepted_mask.astype(jnp.int32))
    keep_mask = jnp.logical_and(active_mask, accepted_prefix_count <= accepted_step_limit)
    return dataclasses.replace(
        trace,
        active_mask=keep_mask,
        accepted_mask=jnp.logical_and(accepted_mask, keep_mask),
    )


def _accepted_time_list_until_attempt_index(trace, inclusive_attempt_index: int) -> list[float]:
    active_mask = np.asarray(jax.device_get(trace.active_mask), dtype=bool)
    accepted_mask = np.asarray(jax.device_get(trace.accepted_mask), dtype=bool)
    step_ts = np.asarray(jax.device_get(trace.step_ts), dtype=float)
    upper = int(inclusive_attempt_index)
    accepted_times = [
        float(step_ts[idx])
        for idx in range(min(len(active_mask), upper + 1))
        if active_mask[idx] and accepted_mask[idx]
    ]
    return accepted_times


def _slice_rollout_trace_by_accepted_window(
    trace,
    *,
    accepted_start: int,
    accepted_count: int,
):
    accepted_start = int(accepted_start)
    accepted_count = int(accepted_count)
    if accepted_start < 0 or accepted_count <= 0:
        raise ValueError("accepted_start must be >= 0 and accepted_count must be > 0.")

    accepted_mask = jnp.asarray(trace.accepted_mask, dtype=bool)
    active_mask = jnp.asarray(trace.active_mask, dtype=bool)
    accepted_prefix_count = jnp.cumsum(accepted_mask.astype(jnp.int32))
    accepted_before = jnp.asarray(accepted_start, dtype=jnp.int32)
    accepted_after = jnp.asarray(accepted_start + accepted_count, dtype=jnp.int32)
    keep_mask = jnp.logical_and(
        active_mask,
        jnp.logical_and(accepted_prefix_count > accepted_before, accepted_prefix_count <= accepted_after),
    )
    return dataclasses.replace(
        trace,
        active_mask=keep_mask,
        accepted_mask=jnp.logical_and(accepted_mask, keep_mask),
    )


def _frozen_replay_nonfinite_debug(
    replay_result: dict[str, Any],
    replay_trace,
    *,
    objectives_np: np.ndarray,
) -> dict[str, Any]:
    rollout = replay_result["rollout"]
    trial_ys = np.asarray(jax.device_get(rollout.trial_ys), dtype=float)
    state_finite_mask = _array_state_finite_mask(trial_ys)
    active_mask = np.asarray(jax.device_get(replay_trace.active_mask), dtype=bool)
    accepted_mask = np.asarray(jax.device_get(replay_trace.accepted_mask), dtype=bool)
    attempted_dts = np.asarray(jax.device_get(replay_trace.attempted_dts), dtype=float)
    next_dts = np.asarray(jax.device_get(replay_trace.next_dts), dtype=float)
    step_ts = np.asarray(jax.device_get(replay_trace.step_ts), dtype=float)
    baseline_err_norms = np.asarray(jax.device_get(replay_trace.err_norms), dtype=float)

    active_indices = np.where(active_mask)[0]
    active_state_finite = state_finite_mask[active_mask]
    bad_positions = np.where(~active_state_finite)[0]
    first_bad_index = int(active_indices[bad_positions[0]]) if bad_positions.size > 0 else -1

    local_attempt_window: list[dict[str, Any]] = []
    if first_bad_index >= 0:
        start = max(0, first_bad_index - 2)
        stop = min(len(active_mask), first_bad_index + 3)
        for idx in range(start, stop):
            if not active_mask[idx]:
                continue
            local_attempt_window.append(
                {
                    "index": int(idx),
                    "accepted": bool(accepted_mask[idx]),
                    "attempted_dt": float(attempted_dts[idx]),
                    "next_dt": float(next_dts[idx]),
                    "time": float(step_ts[idx]),
                    "baseline_err_norm": float(baseline_err_norms[idx]),
                    "replay_state_finite": bool(state_finite_mask[idx]),
                }
            )

    return {
        "final_state_finite": _tree_all_finite(replay_result["final_state"]),
        "objectives_finite": bool(np.all(np.isfinite(objectives_np))),
        "first_bad_index": first_bad_index,
        "first_bad_was_accepted": None if first_bad_index < 0 else bool(accepted_mask[first_bad_index]),
        "first_bad_dt": None if first_bad_index < 0 else float(attempted_dts[first_bad_index]),
        "local_attempt_window": local_attempt_window,
    }


def _write_sweep_csv(
    path: Path,
    *,
    parameter_name: str,
    sweep_values: np.ndarray,
    objective_values: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([parameter_name] + OBJECTIVE_LABELS)
        for value, row in zip(sweep_values, objective_values):
            writer.writerow([float(value)] + [float(v) for v in row])


def _write_figure(report: dict[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    sweep_values = np.asarray(report["sweep_values"], dtype=float)
    sweep_objectives = np.asarray(report["sweep_objectives"], dtype=float)
    baseline_objectives = np.asarray(report["baseline_objectives"], dtype=float)
    grad_ad = np.asarray(report["gradient_autodiff"], dtype=float)
    grad_fd = np.asarray(report["gradient_fd"], dtype=float)
    rel_err = np.asarray(report["gradient_relative_error"], dtype=float)
    rho = np.asarray(report["rho_grid"], dtype=float)
    er_baseline = np.asarray(report["baseline_final_Er"], dtype=float)
    er_minus = np.asarray(report["fd_minus_final_Er"], dtype=float)
    er_plus = np.asarray(report["fd_plus_final_Er"], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.6), constrained_layout=True)

    ax0 = axes[0, 0]
    for idx, label in enumerate(OBJECTIVE_LABELS):
        ax0.plot(sweep_values, sweep_objectives[:, idx], marker="o", linewidth=1.8, label=label)
    ax0.axvline(float(report["baseline_value"]), color="0.35", linestyle="--", linewidth=1.0)
    ax0.set_xlabel(report["parameter_name"])
    ax0.set_ylabel("objective value")
    ax0.set_title("Objective sweep")
    ax0.grid(True, alpha=0.3)
    ax0.legend(fontsize=8)

    ax1 = axes[0, 1]
    ax1.plot(rho, er_baseline, linewidth=2.1, label="baseline")
    ax1.plot(rho, er_minus, linestyle="--", linewidth=1.6, label="- fd step")
    ax1.plot(rho, er_plus, linestyle=":", linewidth=1.8, label="+ fd step")
    ax1.set_xlabel(r"$\rho$")
    ax1.set_ylabel(r"$E_r$")
    ax1.set_title("Final $E_r$ profile comparison")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    ax2 = axes[1, 0]
    ax2.scatter(grad_fd, grad_ad, color="#111827", s=42)
    lo = float(min(np.min(grad_fd), np.min(grad_ad)))
    hi = float(max(np.max(grad_fd), np.max(grad_ad)))
    pad = 0.05 * max(hi - lo, 1.0e-12)
    ax2.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#dc2626", linestyle="--", linewidth=1.2)
    for x, y, label in zip(grad_fd, grad_ad, OBJECTIVE_LABELS):
        ax2.annotate(label, (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax2.set_xlabel("central finite difference")
    ax2.set_ylabel("JAX autodiff")
    ax2.set_title("Derivative parity")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 1]
    x = np.arange(len(OBJECTIVE_LABELS))
    ax3.bar(x, rel_err, color="#2563eb")
    ax3.set_xticks(x, OBJECTIVE_LABELS, rotation=20, ha="right")
    ax3.set_yscale("log")
    ax3.set_ylabel("relative derivative error")
    ax3.set_title("AD vs FD relative error")
    ax3.grid(True, alpha=0.3)
    ax3.text(
        0.03,
        0.95,
        f"passed = {report['passed']}\n"
        f"fd_step = {float(report['fd_step']):.3e}\n"
        f"max rel. err = {float(report['max_relative_error']):.2e}",
        transform=ax3.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.75", "alpha": 0.92},
    )

    fig.suptitle(f"Lagged exact-runtime NTX AD-vs-FD gate: {report['parameter_name']}")
    fig.savefig(out, dpi=220)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def _print_terminal_summary(report: dict[str, Any]) -> None:
    def _fmt_float(value) -> str:
        return "na" if value is None else f"{float(value):.6e}"

    if report.get("small_step_only_check"):
        print(
            f"[autodiff-gate] mode=small_step_only "
            f"parameter={report['parameter_name']} "
            f"baseline_value={report['baseline_value']:.6e} "
            f"fd_step={report['fd_step']:.6e}"
        )
        small_step = report.get("small_step_composition") or []
        print("[autodiff-gate] small-step composition errors:")
        for entry in small_step:
            print(
                f"  - step_count={int(entry['step_count'])} "
                f"step_scale={float(entry['step_scale']):.6e} "
                f"max_rel_err={float(entry['max_relative_error']):.6e}"
            )
        return

    if report.get("controller_only_check"):
        print(
            f"[autodiff-gate] mode=controller_only "
            f"parameter={report['parameter_name']} "
            f"baseline_value={report['baseline_value']:.6e} "
            f"fd_step={report['fd_step']:.6e}"
        )
        controller_step = report.get("controller_step_composition") or []
        print("[autodiff-gate] controller-step composition errors:")
        for entry in controller_step:
            paths = entry.get("controller_paths", {})
            print(
                f"  - step_count={int(entry['step_count'])} "
                f"step_scale={float(entry['step_scale']):.6e} "
                f"max_rel_err={float(entry['max_relative_error']):.6e} "
                f"accepted_equal={paths.get('accepted_mask_equal_minus_plus')} "
                f"attempted_dts_equal={paths.get('attempted_dts_equal_minus_plus')} "
                f"next_dts_equal={paths.get('next_dts_equal_minus_plus')}"
            )
            print(
                f"    baseline attempted_dts={paths.get('baseline', {}).get('attempted_dts')} "
                f"next_dts={paths.get('baseline', {}).get('next_dts')}"
            )
            print(
                f"    fd_minus attempted_dts={paths.get('fd_minus', {}).get('attempted_dts')} "
                f"next_dts={paths.get('fd_minus', {}).get('next_dts')}"
            )
            print(
                f"    fd_plus attempted_dts={paths.get('fd_plus', {}).get('attempted_dts')} "
                f"next_dts={paths.get('fd_plus', {}).get('next_dts')}"
            )
        return

    if report.get("forward_only_controller_check"):
        print(
            f"[autodiff-gate] mode=forward_only_controller "
            f"parameter={report['parameter_name']} "
            f"baseline_value={report['baseline_value']:.6e} "
            f"fd_step={report['fd_step']:.6e}"
        )
        controller_step = report.get("controller_step_composition") or []
        print("[autodiff-gate] controller-step composition errors:")
        for entry in controller_step:
            paths = entry.get("controller_paths", {})
            print(
                f"  - step_count={int(entry['step_count'])} "
                f"step_scale={float(entry['step_scale']):.6e} "
                f"max_rel_err={float(entry['max_relative_error']):.6e} "
                f"accepted_equal={paths.get('accepted_mask_equal_minus_plus')} "
                f"attempted_dts_equal={paths.get('attempted_dts_equal_minus_plus')} "
                f"next_dts_equal={paths.get('next_dts_equal_minus_plus')}"
            )
        return

    if report.get("realized_schedule_ad_debug_fast"):
        print(
            f"[autodiff-gate] mode=realized_schedule_ad_debug_fast "
            f"parameter={report['parameter_name']} "
            f"baseline_value={report['baseline_value']:.6e}"
        )
        path = report.get("rollout_path", {})
        diag = path.get("baseline", {})
        print(
            f"[autodiff-gate] rollout baseline: "
            f"attempt_count={diag.get('attempt_count')} "
            f"accepted_count={diag.get('accepted_count')} "
            f"completed={diag.get('completed')} "
            f"failed={diag.get('failed')} "
            f"fail_code={diag.get('fail_code')}"
        )
        print("[autodiff-gate] AD finiteness:")
        for label, ad in zip(
            report["objective_labels"],
            report["gradient_autodiff"],
        ):
            print(f"  - {label}: ad={float(ad):.6e}")
        nan_debug = report.get("nan_debug")
        if nan_debug is not None:
            print(
                "[autodiff-gate] replay NaN debug: "
                f"first_bad_index={nan_debug.get('first_bad_index')} "
                f"first_bad_was_accepted={nan_debug.get('first_bad_was_accepted')} "
                f"first_bad_dt={_fmt_float(nan_debug.get('first_bad_dt'))} "
                f"final_tangent_finite={nan_debug.get('final_tangent_finite')}"
            )
            zero_tangent_one_step = nan_debug.get("zero_tangent_one_step")
            if zero_tangent_one_step is not None:
                print(
                    "[autodiff-gate] one-step zero-tangent compare: "
                    f"target_attempt_index={zero_tangent_one_step.get('target_attempt_index')} "
                    f"target_was_accepted={zero_tangent_one_step.get('target_was_accepted')} "
                    f"trial_dt={_fmt_float(zero_tangent_one_step.get('trial_dt'))} "
                    f"custom_trial_y_max_abs={_fmt_float(zero_tangent_one_step.get('custom_trial_y_max_abs'))} "
                    f"custom_stage_history_max_abs={_fmt_float(zero_tangent_one_step.get('custom_stage_history_max_abs'))} "
                    f"custom_finite={zero_tangent_one_step.get('custom_finite')} "
                    f"direct_trial_y_max_abs={_fmt_float(zero_tangent_one_step.get('direct_trial_y_max_abs'))} "
                    f"direct_stage_history_max_abs={_fmt_float(zero_tangent_one_step.get('direct_stage_history_max_abs'))} "
                    f"direct_finite={zero_tangent_one_step.get('direct_finite')}"
                )
            local_attempt_window = nan_debug.get("local_attempt_window") or []
            if local_attempt_window:
                print("[autodiff-gate] replay NaN local window:")
                for entry in local_attempt_window:
                    print(
                        "  - "
                        f"index={entry.get('index')} "
                        f"accepted={entry.get('accepted')} "
                        f"attempted_dt={_fmt_float(entry.get('attempted_dt'))} "
                        f"next_dt={_fmt_float(entry.get('next_dt'))} "
                        f"err_norm={_fmt_float(entry.get('err_norm'))} "
                        f"theta_final={_fmt_float(entry.get('theta_final'))} "
                        f"newton_iter_count={entry.get('newton_iter_count')} "
                        f"cache_valid_next={entry.get('cache_valid_next')} "
                        f"tangent_finite={entry.get('tangent_finite')}"
                    )
        return

    if report.get("realized_schedule_frozen_fd_check"):
        print(
            f"[autodiff-gate] mode=realized_schedule_frozen_fd "
            f"parameter={report['parameter_name']} "
            f"baseline_value={report['baseline_value']:.6e} "
            f"fd_step={report['fd_step']:.6e} "
            f"replay_mode={report.get('frozen_replay_mode')} "
            f"ad_mode={report.get('ad_mode')} "
            f"accepted_step_limit={report.get('accepted_step_limit')}"
        )
        path = report.get("rollout_path", {})
        diag = path.get("baseline", {})
        print(
            f"[autodiff-gate] rollout baseline: "
            f"attempt_count={diag.get('attempt_count')} "
            f"accepted_count={diag.get('accepted_count')} "
            f"completed={diag.get('completed')} "
            f"failed={diag.get('failed')} "
            f"fail_code={diag.get('fail_code')}"
        )
        print(
            f"[autodiff-gate] frozen accepted_time_list={report.get('accepted_time_list')}"
        )
        for key in ("frozen_fd_minus", "frozen_fd_plus"):
            diag_frozen = path.get(key, {})
            print(
                f"[autodiff-gate] {key}: "
                f"attempt_count={diag_frozen.get('attempt_count')} "
                f"accepted_count={diag_frozen.get('accepted_count')} "
                f"state_finite={diag_frozen.get('state_finite')} "
                f"objectives_finite={diag_frozen.get('objectives_finite')} "
                f"all_finite={diag_frozen.get('all_finite')}"
            )
            nonfinite_debug = diag_frozen.get("nonfinite_debug")
            if nonfinite_debug is not None:
                print(
                    f"[autodiff-gate] {key} nonfinite debug: "
                    f"first_bad_index={nonfinite_debug.get('first_bad_index')} "
                    f"first_bad_was_accepted={nonfinite_debug.get('first_bad_was_accepted')} "
                    f"first_bad_dt={_fmt_float(nonfinite_debug.get('first_bad_dt'))} "
                    f"final_state_finite={nonfinite_debug.get('final_state_finite')} "
                    f"objectives_finite={nonfinite_debug.get('objectives_finite')}"
                )
                local_window = nonfinite_debug.get("local_attempt_window") or []
                if local_window:
                    print(f"[autodiff-gate] {key} local window:")
                    for entry in local_window:
                        print(
                            "  - "
                            f"index={entry.get('index')} "
                            f"accepted={entry.get('accepted')} "
                            f"time={_fmt_float(entry.get('time'))} "
                            f"attempted_dt={_fmt_float(entry.get('attempted_dt'))} "
                            f"next_dt={_fmt_float(entry.get('next_dt'))} "
                            f"baseline_err_norm={_fmt_float(entry.get('baseline_err_norm'))} "
                            f"replay_state_finite={entry.get('replay_state_finite')}"
                        )
            accepted_mode_debug = diag_frozen.get("accepted_mode_debug")
            if accepted_mode_debug is not None:
                print(
                    f"[autodiff-gate] {key} accepted-only replay check: "
                    f"first_bad_index={accepted_mode_debug.get('first_bad_index')} "
                    f"first_bad_was_accepted={accepted_mode_debug.get('first_bad_was_accepted')} "
                    f"first_bad_dt={_fmt_float(accepted_mode_debug.get('first_bad_dt'))} "
                    f"final_state_finite={accepted_mode_debug.get('final_state_finite')} "
                    f"objectives_finite={accepted_mode_debug.get('objectives_finite')}"
                )
        if report.get("ad_only"):
            print("[autodiff-gate] AD derivative values:")
            for label, ad in zip(
                report["objective_labels"],
                report["gradient_autodiff"],
            ):
                print(f"  - {label}: ad={float(ad):.6e}")
            return
        if report.get("ad_available", True):
            print("[autodiff-gate] objective errors:")
            for label, ad, fd, ae, re in zip(
                report["objective_labels"],
                report["gradient_autodiff"],
                report["gradient_fd"],
                report["gradient_absolute_error"],
                report["gradient_relative_error"],
            ):
                print(
                    f"  - {label}: ad={float(ad):.6e} fd={float(fd):.6e} "
                    f"abs_err={float(ae):.6e} rel_err={float(re):.6e}"
                )
        else:
            print("[autodiff-gate] prefix frozen replay diagnostic: AD skipped; reporting frozen FD only")
            for label, fd in zip(
                report["objective_labels"],
                report["gradient_fd"],
            ):
                print(f"  - {label}: frozen_fd={float(fd):.6e}")
        return

    if report.get("realized_schedule_frozen_replay_localize"):
        print(
            f"[autodiff-gate] mode=realized_schedule_frozen_replay_localize "
            f"parameter={report['parameter_name']} "
            f"baseline_value={report['baseline_value']:.6e} "
            f"fd_step={report['fd_step']:.6e}"
        )
        path = report.get("rollout_path", {})
        diag = path.get("baseline", {})
        print(
            f"[autodiff-gate] rollout baseline: "
            f"attempt_count={diag.get('attempt_count')} "
            f"accepted_count={diag.get('accepted_count')} "
            f"completed={diag.get('completed')} "
            f"failed={diag.get('failed')} "
            f"fail_code={diag.get('fail_code')}"
        )
        print("[autodiff-gate] attempt replay prefix checks:")
        for entry in report.get("attempt_prefix_checks") or []:
            print(
                "  - "
                f"accepted_steps={entry.get('accepted_step_limit')} "
                f"attempt_count={entry.get('attempt_count')} "
                f"accepted_count={entry.get('accepted_count')} "
                f"minus_all_finite={entry.get('fd_minus', {}).get('all_finite')} "
                f"plus_all_finite={entry.get('fd_plus', {}).get('all_finite')} "
                f"all_finite={entry.get('all_finite')}"
            )
        last_passing = report.get("last_passing_attempt_case")
        if last_passing is not None:
            print(
                "[autodiff-gate] last passing attempt prefix: "
                f"accepted_steps={last_passing.get('accepted_step_limit')} "
                f"attempt_count={last_passing.get('attempt_count')}"
            )
        first_failing = report.get("first_failing_attempt_case")
        if first_failing is not None:
            print(
                "[autodiff-gate] first failing attempt prefix: "
                f"accepted_steps={first_failing.get('accepted_step_limit')} "
                f"attempt_count={first_failing.get('attempt_count')}"
            )
        accepted_boundary = report.get("accepted_mode_at_boundary")
        if accepted_boundary is not None:
            print(
                "[autodiff-gate] accepted replay at boundary: "
                f"accepted_steps={accepted_boundary.get('accepted_step_limit')} "
                f"attempt_count={accepted_boundary.get('attempt_count')} "
                f"minus_all_finite={accepted_boundary.get('fd_minus', {}).get('all_finite')} "
                f"plus_all_finite={accepted_boundary.get('fd_plus', {}).get('all_finite')} "
                f"all_finite={accepted_boundary.get('all_finite')}"
            )
        return

    if report.get("realized_schedule_windowed_frozen_fd_check"):
        print(
            f"[autodiff-gate] mode=realized_schedule_windowed_frozen_fd "
            f"parameter={report['parameter_name']} "
            f"baseline_value={report['baseline_value']:.6e} "
            f"fd_step={report['fd_step']:.6e} "
            f"replay_mode={report.get('windowed_replay_mode')} "
            f"accepted_window_size={report.get('accepted_window_size')}"
        )
        path = report.get("rollout_path", {})
        diag = path.get("baseline", {})
        print(
            f"[autodiff-gate] rollout baseline: "
            f"attempt_count={diag.get('attempt_count')} "
            f"accepted_count={diag.get('accepted_count')} "
            f"completed={diag.get('completed')} "
            f"failed={diag.get('failed')} "
            f"fail_code={diag.get('fail_code')}"
        )
        for key in ("windowed_fd_minus", "windowed_fd_plus"):
            diag_windowed = path.get(key, {})
            print(
                f"[autodiff-gate] {key}: "
                f"all_finite={diag_windowed.get('all_finite')}"
            )
            for entry in diag_windowed.get("window_summaries") or []:
                print(
                    "  - "
                    f"accepted_start={entry.get('accepted_start')} "
                    f"accepted_count={entry.get('accepted_count')} "
                    f"attempt_count={entry.get('attempt_count')} "
                    f"state_finite={entry.get('state_finite')}"
                )
            failing_window = diag_windowed.get("first_failing_window_debug")
            if failing_window is not None:
                nf = failing_window.get("nonfinite_debug", {})
                print(
                    f"[autodiff-gate] {key} failing window: "
                    f"accepted_start={failing_window.get('accepted_start')} "
                    f"accepted_count={failing_window.get('accepted_count')} "
                    f"attempt_count={failing_window.get('attempt_count')} "
                    f"first_bad_index={nf.get('first_bad_index')} "
                    f"first_bad_was_accepted={nf.get('first_bad_was_accepted')} "
                    f"first_bad_dt={_fmt_float(nf.get('first_bad_dt'))}"
                )
                local_window = nf.get("local_attempt_window") or []
                if local_window:
                    print(f"[autodiff-gate] {key} local window:")
                    for entry in local_window:
                        print(
                            "  - "
                            f"index={entry.get('index')} "
                            f"accepted={entry.get('accepted')} "
                            f"time={_fmt_float(entry.get('time'))} "
                            f"attempted_dt={_fmt_float(entry.get('attempted_dt'))} "
                            f"next_dt={_fmt_float(entry.get('next_dt'))} "
                            f"baseline_err_norm={_fmt_float(entry.get('baseline_err_norm'))} "
                            f"replay_state_finite={entry.get('replay_state_finite')}"
                        )
                accepted_debug = failing_window.get("accepted_mode_debug")
                if accepted_debug is not None:
                    print(
                        f"[autodiff-gate] {key} accepted-only replay on failing window: "
                        f"first_bad_index={accepted_debug.get('first_bad_index')} "
                        f"first_bad_was_accepted={accepted_debug.get('first_bad_was_accepted')} "
                        f"first_bad_dt={_fmt_float(accepted_debug.get('first_bad_dt'))} "
                        f"final_state_finite={accepted_debug.get('final_state_finite')} "
                        f"objectives_finite={accepted_debug.get('objectives_finite')}"
                    )
        print("[autodiff-gate] objective errors:")
        for label, ad, fd, ae, re in zip(
            report["objective_labels"],
            report["gradient_autodiff"],
            report["gradient_fd"],
            report["gradient_absolute_error"],
            report["gradient_relative_error"],
        ):
                print(
                    f"  - {label}: ad={float(ad):.6e} fd={float(fd):.6e} "
                    f"abs_err={float(ae):.6e} rel_err={float(re):.6e}"
                )
        return

    if report.get("adaptive_vs_frozen_custom_ad_check"):
        print(
            f"[autodiff-gate] mode=adaptive_vs_frozen_custom_ad "
            f"parameter={report['parameter_name']} "
            f"checkpoint_index={report['checkpoint_index']} "
            f"replay_mode={report.get('replay_mode')}"
        )
        print(
            "[autodiff-gate] timings: "
            f"adaptive_seconds={float(report['adaptive_runtime_seconds']):.3f} "
            f"frozen_seconds={float(report['frozen_runtime_seconds']):.3f}"
        )
        print("[autodiff-gate] adaptive vs frozen custom AD:")
        for label, ad_adaptive, ad_frozen, ae, re in zip(
            report["objective_labels"],
            report["gradient_adaptive_custom"],
            report["gradient_frozen_custom"],
            report["gradient_absolute_error"],
            report["gradient_relative_error"],
        ):
            print(
                f"  - {label}: adaptive={float(ad_adaptive):.6e} "
                f"frozen={float(ad_frozen):.6e} "
                f"abs_err={float(ae):.6e} rel_err={float(re):.6e}"
            )
        return

    if report.get("baseline_dt_path_safe_fd_check"):
        print(
            f"[autodiff-gate] mode=baseline_dt_path_safe_fd "
            f"parameter={report['parameter_name']} "
            f"baseline_value={report['baseline_value']:.6e} "
            f"fd_step={report['fd_step']:.6e} "
            f"ad_mode={report.get('ad_mode')} "
            f"safe_attempt_index={report.get('safe_attempt_index')} "
            f"safe_final_time={report.get('safe_final_time'):.6e}"
        )
        path = report.get("rollout_path", {})
        diag = path.get("baseline", {})
        print(
            f"[autodiff-gate] rollout baseline: "
            f"attempt_count={diag.get('attempt_count')} "
            f"accepted_count={diag.get('accepted_count')} "
            f"completed={diag.get('completed')} "
            f"failed={diag.get('failed')} "
            f"fail_code={diag.get('fail_code')}"
        )
        print(
            "[autodiff-gate] fixed-dt replay finiteness: "
            f"baseline={path.get('baseline_fixed_dt_state_finite')} "
            f"fd_minus={path.get('fd_minus_fixed_dt_state_finite')} "
            f"fd_plus={path.get('fd_plus_fixed_dt_state_finite')}"
        )
        print("[autodiff-gate] objective errors:")
        for label, ad, fd, ae, re in zip(
            report["objective_labels"],
            report["gradient_autodiff"],
            report["gradient_fd"],
            report["gradient_absolute_error"],
            report["gradient_relative_error"],
        ):
            print(
                f"  - {label}: ad={float(ad):.6e} fd={float(fd):.6e} "
                f"abs_err={float(ae):.6e} rel_err={float(re):.6e}"
            )
        return

    if report.get("baseline_dt_path_safe_compose_check"):
        print(
            f"[autodiff-gate] mode=baseline_dt_path_safe_compose "
            f"parameter={report['parameter_name']} "
            f"baseline_value={report['baseline_value']:.6e} "
            f"safe_attempt_index={report.get('safe_attempt_index')} "
            f"safe_final_time={report.get('safe_final_time'):.6e}"
        )
        print("[autodiff-gate] objective errors:")
        for label, ad, fd, ae, re in zip(
            report["objective_labels"],
            report["gradient_realized_schedule_autodiff"],
            report["gradient_fixed_dt_direct_autodiff"],
            report["gradient_absolute_error"],
            report["gradient_relative_error"],
        ):
            print(
                f"  - {label}: adaptive_ad={float(ad):.6e} fixed_dt_direct_ad={float(fd):.6e} "
                f"abs_err={float(ae):.6e} rel_err={float(re):.6e}"
            )
        return

    if report.get("baseline_dt_path_safe_compose_scan_check"):
        print(
            f"[autodiff-gate] mode=baseline_dt_path_safe_compose_scan "
            f"parameter={report['parameter_name']} "
            f"baseline_value={report['baseline_value']:.6e} "
            f"safe_attempt_index={report.get('safe_attempt_index')} "
            f"safe_accepted_count={report.get('safe_accepted_count')}"
        )
        path = report.get("rollout_path", {})
        diag = path.get("baseline", {})
        print(
            f"[autodiff-gate] rollout baseline: "
            f"attempt_count={diag.get('attempt_count')} "
            f"accepted_count={diag.get('accepted_count')} "
            f"completed={diag.get('completed')} "
            f"failed={diag.get('failed')} "
            f"fail_code={diag.get('fail_code')}"
        )
        print("[autodiff-gate] prefix compose errors:")
        label_to_index = {label: idx for idx, label in enumerate(report["objective_labels"])}
        er_idx = label_to_index["Er_volume_average"]
        er2_idx = label_to_index["Er2_volume_average"]
        pressure_idx = label_to_index["total_pressure_volume_average"]
        for entry in report.get("entries") or []:
            rel_err = np.asarray(entry["gradient_relative_error"], dtype=float)
            print(
                "  - "
                f"accepted_count={entry.get('accepted_count')} "
                f"final_time={float(entry.get('final_time')):.6e} "
                f"max_rel_err={float(entry.get('max_relative_error')):.6e} "
                f"Er_rel_err={float(rel_err[er_idx]):.6e} "
                f"Er2_rel_err={float(rel_err[er2_idx]):.6e} "
                f"pressure_rel_err={float(rel_err[pressure_idx]):.6e}"
            )
        return

    if report.get("baseline_dt_path_safe_trajectory_compare_check"):
        print(
            f"[autodiff-gate] mode=baseline_dt_path_safe_trajectory_compare "
            f"parameter={report['parameter_name']} "
            f"baseline_value={report['baseline_value']:.6e} "
            f"safe_attempt_index={report.get('safe_attempt_index')} "
            f"safe_accepted_count={report.get('safe_accepted_count')} "
            f"sample_every={report.get('sample_every')} "
            f"safe_final_time={report.get('safe_final_time'):.6e}"
        )
        path = report.get("rollout_path", {})
        diag = path.get("baseline", {})
        print(
            f"[autodiff-gate] rollout baseline: "
            f"attempt_count={diag.get('attempt_count')} "
            f"accepted_count={diag.get('accepted_count')} "
            f"completed={diag.get('completed')} "
            f"failed={diag.get('failed')} "
            f"fail_code={diag.get('fail_code')}"
        )
        print("[autodiff-gate] accepted-step trajectory errors:")
        for entry in report.get("entries") or []:
            print(
                "  - "
                f"accepted_index={entry.get('accepted_index')} "
                f"time={float(entry.get('time')):.6e} "
                f"max_rel_err={float(entry.get('max_relative_error')):.6e} "
                f"Er_rel_err={float(entry.get('Er_relative_error')):.6e} "
                f"Er2_rel_err={float(entry.get('Er2_relative_error')):.6e} "
                f"pressure_rel_err={float(entry.get('pressure_relative_error')):.6e}"
            )
        return

    if report.get("baseline_dt_path_safe_state_trajectory_compare_check"):
        print(
            f"[autodiff-gate] mode=baseline_dt_path_safe_state_trajectory_compare "
            f"parameter={report['parameter_name']} "
            f"baseline_value={report['baseline_value']:.6e} "
            f"safe_attempt_index={report.get('safe_attempt_index')} "
            f"safe_accepted_count={report.get('safe_accepted_count')} "
            f"sample_every={report.get('sample_every')} "
            f"safe_final_time={report.get('safe_final_time'):.6e}"
        )
        path = report.get("rollout_path", {})
        diag = path.get("baseline", {})
        print(
            f"[autodiff-gate] rollout baseline: "
            f"attempt_count={diag.get('attempt_count')} "
            f"accepted_count={diag.get('accepted_count')} "
            f"completed={diag.get('completed')} "
            f"failed={diag.get('failed')} "
            f"fail_code={diag.get('fail_code')}"
        )
        print("[autodiff-gate] accepted-step state-tangent errors:")
        for entry in report.get("entries") or []:
            print(
                "  - "
                f"accepted_index={entry.get('accepted_index')} "
                f"time={float(entry.get('time')):.6e} "
                f"full_state_rel_err={float(entry.get('full_state_relative_error')):.6e} "
                f"density_rel_err={float(entry.get('density_relative_error')):.6e} "
                f"pressure_rel_err={float(entry.get('pressure_relative_error')):.6e} "
                f"Er_rel_err={float(entry.get('Er_relative_error')):.6e}"
            )
        return

    if report.get("realized_trace_safe_state_trajectory_compare_check"):
        print(
            f"[autodiff-gate] mode=realized_trace_safe_state_trajectory_compare "
            f"parameter={report['parameter_name']} "
            f"baseline_value={report['baseline_value']:.6e} "
            f"safe_attempt_index={report.get('safe_attempt_index')} "
            f"safe_accepted_count={report.get('safe_accepted_count')} "
            f"sample_every={report.get('sample_every')} "
            f"safe_final_time={report.get('safe_final_time'):.6e}"
        )
        path = report.get("rollout_path", {})
        diag = path.get("baseline", {})
        print(
            f"[autodiff-gate] rollout baseline: "
            f"attempt_count={diag.get('attempt_count')} "
            f"accepted_count={diag.get('accepted_count')} "
            f"completed={diag.get('completed')} "
            f"failed={diag.get('failed')} "
            f"fail_code={diag.get('fail_code')}"
        )
        print("[autodiff-gate] accepted-step realized-trace state-tangent errors:")
        for entry in report.get("entries", []):
            print(
                "  - "
                f"accepted_index={int(entry.get('accepted_index'))} "
                f"time={float(entry.get('time')):.6e} "
                f"full_state_rel_err={float(entry.get('full_state_relative_error')):.6e} "
                f"density_rel_err={float(entry.get('density_relative_error')):.6e} "
                f"pressure_rel_err={float(entry.get('pressure_relative_error')):.6e} "
                f"Er_rel_err={float(entry.get('Er_relative_error')):.6e}"
            )
        return

    if report.get("realized_trace_sparse_checkpoint_compare_check"):
        print(
            f"[autodiff-gate] mode=realized_trace_sparse_checkpoint_compare "
            f"parameter={report['parameter_name']} "
            f"baseline_value={report['baseline_value']:.6e} "
            f"checkpoints={','.join(str(v) for v in report.get('checkpoint_counts', []))}"
        )
        path = report.get("rollout_path", {})
        diag = path.get("baseline", {})
        print(
            f"[autodiff-gate] rollout baseline: "
            f"attempt_count={diag.get('attempt_count')} "
            f"accepted_count={diag.get('accepted_count')} "
            f"completed={diag.get('completed')} "
            f"failed={diag.get('failed')} "
            f"fail_code={diag.get('fail_code')}"
        )
        print("[autodiff-gate] sparse realized-trace checkpoint errors:")
        for entry in report.get("entries", []):
            print(
                "  - "
                f"accepted_index={int(entry.get('accepted_index'))} "
                f"time={float(entry.get('time')):.6e} "
                f"full_state_rel_err={float(entry.get('full_state_relative_error')):.6e} "
                f"density_rel_err={float(entry.get('density_relative_error')):.6e} "
                f"pressure_rel_err={float(entry.get('pressure_relative_error')):.6e} "
                f"Er_rel_err={float(entry.get('Er_relative_error')):.6e}"
            )
        return

    if report.get("realized_trace_checkpoint_compare_check"):
        print(
            f"[autodiff-gate] mode=realized_trace_checkpoint_compare "
            f"parameter={report['parameter_name']} "
            f"baseline_value={report['baseline_value']:.6e} "
            f"checkpoint_index={report.get('checkpoint_index')} "
            f"checkpoint_time={report.get('checkpoint_time'):.6e}"
        )
        path = report.get("rollout_path", {})
        diag = path.get("baseline", {})
        print(
            f"[autodiff-gate] rollout baseline: "
            f"attempt_count={diag.get('attempt_count')} "
            f"accepted_count={diag.get('accepted_count')} "
            f"completed={diag.get('completed')} "
            f"failed={diag.get('failed')} "
            f"fail_code={diag.get('fail_code')}"
        )
        entry = report.get("comparison", {}).get("trial_y", {})
        print(
            "[autodiff-gate] checkpoint comparison: "
            f"full_rel_err={float(entry.get('full_relative_error')):.6e} "
            f"pressure_rel_err={float(entry.get('pressure_relative_error')):.6e} "
            f"Er_rel_err={float(entry.get('Er_relative_error')):.6e}"
        )
        return

    if report.get("ntx_derivative_mode_compare_check"):
        print(
            f"[autodiff-gate] mode=ntx_derivative_mode_compare "
            f"parameter={report['parameter_name']} "
            f"checkpoint_index={report.get('checkpoint_index')} "
            f"replay_mode={report.get('replay_mode')}"
        )
        print("[autodiff-gate] timings:")
        for mode_name in ("direct", "custom_vjp"):
            timing = report.get("timings", {}).get(mode_name, {})
            print(f"  - {mode_name}: wall_seconds={float(timing.get('wall_seconds', 0.0)):.3f}")
        print("[autodiff-gate] custom AD mode-to-mode objective differences:")
        for label, abs_err, rel_err in zip(
            report["objective_labels"],
            report["custom_ad_mode_difference"]["objective_absolute_error"],
            report["custom_ad_mode_difference"]["objective_relative_error"],
        ):
            print(
                f"  - {label}: "
                f"abs_err={float(abs_err):.6e} "
                f"rel_err={float(rel_err):.6e}"
            )
        for mode_name in ("direct", "custom_vjp"):
            mode_report = report["modes"][mode_name]
            state_cmp = mode_report.get("state_tangent_comparison", {}).get("custom_vs_fd", {})
            print(
                f"[autodiff-gate] {mode_name} custom_vs_fd state: "
                f"full_rel_err={float(state_cmp.get('full_relative_error')):.6e} "
                f"pressure_rel_err={float(state_cmp.get('pressure_relative_error')):.6e} "
                f"Er_rel_err={float(state_cmp.get('Er_relative_error')):.6e}"
            )
        return

    if report.get("realized_trace_checkpoint_frozen_fd_check"):
        mode_name = (
            "realized_trace_checkpoint_fd_stencil"
            if report.get("fd_stencil_check")
            else "realized_trace_checkpoint_frozen_fd"
        )
        print(
            f"[autodiff-gate] mode={mode_name} "
            f"parameter={report['parameter_name']} "
            f"baseline_value={report['baseline_value']:.6e} "
            f"fd_step={report['fd_step']:.6e} "
            f"ntx_derivative_mode={report.get('ntx_exact_derivative_mode', 'direct')} "
            f"checkpoint_index={report.get('checkpoint_index')} "
            f"checkpoint_time={report.get('checkpoint_time'):.6e}"
        )
        path = report.get("rollout_path", {})
        diag = path.get("baseline", {})
        print(
            f"[autodiff-gate] rollout baseline: "
            f"attempt_count={diag.get('attempt_count')} "
            f"accepted_count={diag.get('accepted_count')} "
            f"completed={diag.get('completed')} "
            f"failed={diag.get('failed')} "
            f"fail_code={diag.get('fail_code')}"
        )
        print(
            "[autodiff-gate] objective errors:"
        )
        if report.get("fd_stencil_check"):
            for label, ad, fd_center, fd_five, re_center, re_five in zip(
                report["objective_labels"],
                report["gradient_autodiff"],
                report["gradient_fd"],
                report["gradient_fd_five_point"],
                report["gradient_relative_error"],
                report["gradient_five_point_relative_error"],
            ):
                print(
                    f"  - {label}: "
                    f"custom_ad={float(ad):.6e} "
                    f"fd_center={float(fd_center):.6e} "
                    f"fd_five_point={float(fd_five):.6e} "
                    f"custom_vs_center_rel_err={float(re_center):.6e} "
                    f"custom_vs_five_point_rel_err={float(re_five):.6e}"
                )
        elif report.get("gradient_direct") is None:
            for label, ad, fd, re in zip(
                report["objective_labels"],
                report["gradient_autodiff"],
                report["gradient_fd"],
                report["gradient_relative_error"],
            ):
                print(
                    f"  - {label}: "
                    f"custom_ad={float(ad):.6e} "
                    f"fd={float(fd):.6e} "
                    f"custom_vs_fd_rel_err={float(re):.6e}"
                )
        else:
            for label, ad, direct, fd, ae, re, dae, dre, cde, cdr in zip(
                report["objective_labels"],
                report["gradient_autodiff"],
                report["gradient_direct"],
                report["gradient_fd"],
                report["gradient_absolute_error"],
                report["gradient_relative_error"],
                report["gradient_direct_absolute_error"],
                report["gradient_direct_relative_error"],
                report["gradient_custom_vs_direct_absolute_error"],
                report["gradient_custom_vs_direct_relative_error"],
            ):
                print(
                    f"  - {label}: "
                    f"custom_ad={float(ad):.6e} "
                    f"direct_ad={float(direct):.6e} "
                    f"fd={float(fd):.6e} "
                    f"custom_vs_fd_rel_err={float(re):.6e} "
                    f"direct_vs_fd_rel_err={float(dre):.6e} "
                    f"custom_vs_direct_rel_err={float(cdr):.6e}"
                )
        state_cmp = report.get("state_tangent_comparison", {})
        if state_cmp:
            custom_vs_fd = state_cmp.get("custom_vs_fd", {})
            direct_vs_fd = state_cmp.get("direct_vs_fd", {})
            custom_vs_direct = state_cmp.get("custom_vs_direct", {})
            print("[autodiff-gate] state tangent errors:")
            print(
                "  - custom_vs_fd: "
                f"full_rel_err={float(custom_vs_fd.get('full_relative_error')):.6e} "
                f"pressure_rel_err={float(custom_vs_fd.get('pressure_relative_error')):.6e} "
                f"Er_rel_err={float(custom_vs_fd.get('Er_relative_error')):.6e}"
            )
            if direct_vs_fd is not None:
                print(
                    "  - direct_vs_fd: "
                    f"full_rel_err={float(direct_vs_fd.get('full_relative_error')):.6e} "
                    f"pressure_rel_err={float(direct_vs_fd.get('pressure_relative_error')):.6e} "
                    f"Er_rel_err={float(direct_vs_fd.get('Er_relative_error')):.6e}"
                )
            if custom_vs_direct is not None:
                print(
                    "  - custom_vs_direct: "
                    f"full_rel_err={float(custom_vs_direct.get('full_relative_error')):.6e} "
                    f"pressure_rel_err={float(custom_vs_direct.get('pressure_relative_error')):.6e} "
                    f"Er_rel_err={float(custom_vs_direct.get('Er_relative_error')):.6e}"
                )
        state_cmp_five = report.get("state_tangent_comparison_five_point")
        if state_cmp_five:
            custom_vs_fd_five = state_cmp_five.get("custom_vs_fd_five_point", {})
            center_vs_fd_five = state_cmp_five.get("center_vs_fd_five_point", {})
            print("[autodiff-gate] state tangent errors (five-point):")
            print(
                "  - custom_vs_fd_five_point: "
                f"full_rel_err={float(custom_vs_fd_five.get('full_relative_error')):.6e} "
                f"pressure_rel_err={float(custom_vs_fd_five.get('pressure_relative_error')):.6e} "
                f"Er_rel_err={float(custom_vs_fd_five.get('Er_relative_error')):.6e}"
            )
            print(
                "  - center_vs_fd_five_point: "
                f"full_rel_err={float(center_vs_fd_five.get('full_relative_error')):.6e} "
                f"pressure_rel_err={float(center_vs_fd_five.get('pressure_relative_error')):.6e} "
                f"Er_rel_err={float(center_vs_fd_five.get('Er_relative_error')):.6e}"
            )
            direct_vs_fd_five = state_cmp_five.get("direct_vs_fd_five_point")
            if direct_vs_fd_five is not None:
                print(
                    "  - direct_vs_fd_five_point: "
                    f"full_rel_err={float(direct_vs_fd_five.get('full_relative_error')):.6e} "
                    f"pressure_rel_err={float(direct_vs_fd_five.get('pressure_relative_error')):.6e} "
                    f"Er_rel_err={float(direct_vs_fd_five.get('Er_relative_error')):.6e}"
                )
        return

    if report.get("realized_trace_checkpoint_interpolated_fd_check"):
        print(
            f"[autodiff-gate] mode=realized_trace_checkpoint_interpolated_fd "
            f"parameter={report['parameter_name']} "
            f"baseline_value={report['baseline_value']:.6e} "
            f"fd_step={report['fd_step']:.6e} "
            f"checkpoint_index={report.get('checkpoint_index')} "
            f"checkpoint_time={report.get('checkpoint_time'):.6e}"
        )
        path = report.get("rollout_path", {})
        for key in ("baseline", "fd_minus", "fd_plus"):
            diag = path.get(key, {})
            print(
                f"[autodiff-gate] rollout {key}: "
                f"attempt_count={diag.get('attempt_count')} "
                f"accepted_count={diag.get('accepted_count')} "
                f"completed={diag.get('completed')} "
                f"failed={diag.get('failed')} "
                f"fail_code={diag.get('fail_code')}"
            )
        print("[autodiff-gate] objective errors:")
        for label, ad, fd, ae, re in zip(
            report["objective_labels"],
            report["gradient_autodiff"],
            report["gradient_fd"],
            report["gradient_absolute_error"],
            report["gradient_relative_error"],
        ):
            print(
                f"  - {label}: ad={float(ad):.6e} fd={float(fd):.6e} "
                f"abs_err={float(ae):.6e} rel_err={float(re):.6e}"
            )
        return

    if report.get("baseline_dt_path_first_step_field_compare_check"):
        print(
            f"[autodiff-gate] mode=baseline_dt_path_first_step_field_compare "
            f"parameter={report['parameter_name']} "
            f"baseline_value={report['baseline_value']:.6e} "
            f"safe_attempt_index={report.get('safe_attempt_index')} "
            f"safe_final_time={report.get('safe_final_time'):.6e} "
            f"first_step_time={report.get('first_step_time'):.6e}"
        )
        path = report.get("rollout_path", {})
        diag = path.get("baseline", {})
        print(
            f"[autodiff-gate] rollout baseline: "
            f"attempt_count={diag.get('attempt_count')} "
            f"accepted_count={diag.get('accepted_count')} "
            f"completed={diag.get('completed')} "
            f"failed={diag.get('failed')} "
            f"fail_code={diag.get('fail_code')}"
        )
        for label in ("density", "pressure", "Er"):
            entry = report.get(label, {})
            print(
                f"[autodiff-gate] first-step {label}: "
                f"adaptive_max_abs={float(entry.get('adaptive_max_abs')):.6e} "
                f"direct_max_abs={float(entry.get('direct_max_abs')):.6e} "
                f"error_max_abs={float(entry.get('error_max_abs')):.6e} "
                f"rel_err={float(entry.get('relative_error')):.6e}"
            )
        return

    if report.get("baseline_dt_path_first_step_local_tangent_compare_check"):
        print(
            f"[autodiff-gate] mode=baseline_dt_path_first_step_local_tangent_compare "
            f"parameter={report['parameter_name']} "
            f"baseline_value={report['baseline_value']:.6e} "
            f"safe_attempt_index={report.get('safe_attempt_index')} "
            f"safe_final_time={report.get('safe_final_time'):.6e} "
            f"first_step_time={report.get('first_step_time'):.6e}"
        )
        path = report.get("rollout_path", {})
        diag = path.get("baseline", {})
        print(
            f"[autodiff-gate] rollout baseline: "
            f"attempt_count={diag.get('attempt_count')} "
            f"accepted_count={diag.get('accepted_count')} "
            f"completed={diag.get('completed')} "
            f"failed={diag.get('failed')} "
            f"fail_code={diag.get('fail_code')}"
        )
        for label in ("trial_y", "carry_after_attempt_y", "stage_history"):
            entry = report.get(label, {})
            print(
                f"[autodiff-gate] first-step {label}: "
                f"full_rel_err={float(entry.get('full_relative_error')):.6e} "
                f"pressure_rel_err={float(entry.get('pressure_relative_error')):.6e} "
                f"Er_rel_err={float(entry.get('Er_relative_error')):.6e}"
            )
        return

    if report.get("baseline_dt_path_first_step_exact_local_tangent_compare_check"):
        print(
            f"[autodiff-gate] mode=baseline_dt_path_first_step_exact_local_tangent_compare "
            f"parameter={report['parameter_name']} "
            f"baseline_value={report['baseline_value']:.6e} "
            f"safe_attempt_index={report.get('safe_attempt_index')} "
            f"safe_final_time={report.get('safe_final_time'):.6e} "
            f"first_step_time={report.get('first_step_time'):.6e}"
        )
        path = report.get("rollout_path", {})
        diag = path.get("baseline", {})
        print(
            f"[autodiff-gate] rollout baseline: "
            f"attempt_count={diag.get('attempt_count')} "
            f"accepted_count={diag.get('accepted_count')} "
            f"completed={diag.get('completed')} "
            f"failed={diag.get('failed')} "
            f"fail_code={diag.get('fail_code')}"
        )
        for family in (
            "custom_vs_direct",
            "exact_vs_direct",
            "custom_vs_exact",
            "restricted_direct_vs_direct",
            "custom_vs_restricted_direct",
        ):
            print(f"[autodiff-gate] {family}:")
            section = report.get(family, {})
            for label in ("trial_y", "stage_history"):
                entry = section.get(label, {})
                print(
                    "  - "
                    f"{label}: "
                    f"full_rel_err={float(entry.get('full_relative_error')):.6e} "
                    f"pressure_rel_err={float(entry.get('pressure_relative_error')):.6e} "
                    f"Er_rel_err={float(entry.get('Er_relative_error')):.6e}"
                )
        ablations = report.get("carry_field_ablations") or {}
        if ablations:
            print("[autodiff-gate] carry-field ablations:")
            for label, section in ablations.items():
                direct_entry = section.get("ablated_direct_vs_direct", {}).get("trial_y", {})
                custom_entry = section.get("custom_vs_ablated_direct", {}).get("trial_y", {})
                print(
                    "  - "
                    f"{label}: "
                    f"ablated_direct_Er_rel_err={float(direct_entry.get('Er_relative_error')):.6e} "
                    f"custom_vs_ablated_Er_rel_err={float(custom_entry.get('Er_relative_error')):.6e}"
                )
        return

    if report.get("baseline_dt_path_second_step_carry_ablation_check"):
        print(
            f"[autodiff-gate] mode=baseline_dt_path_second_step_carry_ablation "
            f"parameter={report['parameter_name']} "
            f"baseline_value={report['baseline_value']:.6e} "
            f"first_step_time={report.get('first_step_time'):.6e} "
            f"second_step_time={report.get('second_step_time'):.6e}"
        )
        path = report.get("rollout_path", {})
        diag = path.get("baseline", {})
        print(
            f"[autodiff-gate] rollout baseline: "
            f"attempt_count={diag.get('attempt_count')} "
            f"accepted_count={diag.get('accepted_count')} "
            f"completed={diag.get('completed')} "
            f"failed={diag.get('failed')} "
            f"fail_code={diag.get('fail_code')}"
        )
        print("[autodiff-gate] second-step comparisons:")
        for label in ("custom_vs_direct", "carry_after_step1_custom_vs_direct"):
            entry = report.get(label, {}).get("trial_y", {})
            print(
                "  - "
                f"{label}: "
                f"full_rel_err={float(entry.get('full_relative_error')):.6e} "
                f"pressure_rel_err={float(entry.get('pressure_relative_error')):.6e} "
                f"Er_rel_err={float(entry.get('Er_relative_error')):.6e}"
            )
        ablations = report.get("carry_field_ablations") or {}
        if ablations:
            print("[autodiff-gate] second-step carry-field ablations:")
            for label, section in ablations.items():
                direct_entry = section.get("ablated_direct_vs_direct", {}).get("trial_y", {})
                custom_entry = section.get("custom_vs_ablated_direct", {}).get("trial_y", {})
                print(
                    "  - "
                    f"{label}: "
                    f"ablated_direct_Er_rel_err={float(direct_entry.get('Er_relative_error')):.6e} "
                    f"custom_vs_ablated_Er_rel_err={float(custom_entry.get('Er_relative_error')):.6e}"
                )
        return

    if report.get("baseline_dt_path_third_step_carry_ablation_check"):
        print(
            f"[autodiff-gate] mode=baseline_dt_path_third_step_carry_ablation "
            f"parameter={report['parameter_name']} "
            f"baseline_value={report['baseline_value']:.6e} "
            f"first_step_time={report.get('first_step_time'):.6e} "
            f"second_step_time={report.get('second_step_time'):.6e} "
            f"third_step_time={report.get('third_step_time'):.6e}"
        )
        path = report.get("rollout_path", {})
        diag = path.get("baseline", {})
        print(
            f"[autodiff-gate] rollout baseline: "
            f"attempt_count={diag.get('attempt_count')} "
            f"accepted_count={diag.get('accepted_count')} "
            f"completed={diag.get('completed')} "
            f"failed={diag.get('failed')} "
            f"fail_code={diag.get('fail_code')}"
        )
        print("[autodiff-gate] third-step comparisons:")
        for label in ("custom_vs_direct", "carry_after_step2_custom_vs_direct"):
            entry = report.get(label, {}).get("trial_y", {})
            print(
                "  - "
                f"{label}: "
                f"full_rel_err={float(entry.get('full_relative_error')):.6e} "
                f"pressure_rel_err={float(entry.get('pressure_relative_error')):.6e} "
                f"Er_rel_err={float(entry.get('Er_relative_error')):.6e}"
            )
        helper_consistency = report.get("helper_consistency") or {}
        if helper_consistency:
            print("[autodiff-gate] third-step helper consistency:")
            for label in ("custom_scan_vs_manual", "direct_scan_vs_manual"):
                entry = helper_consistency.get(label, {}).get("trial_y", {})
                print(
                    "  - "
                    f"{label}: "
                    f"full_rel_err={float(entry.get('full_relative_error')):.6e} "
                    f"pressure_rel_err={float(entry.get('pressure_relative_error')):.6e} "
                    f"Er_rel_err={float(entry.get('Er_relative_error')):.6e}"
                )
        ablations = report.get("carry_field_ablations") or {}
        if ablations:
            print("[autodiff-gate] third-step carry-field ablations:")
            for label, section in ablations.items():
                direct_entry = section.get("ablated_direct_vs_direct", {}).get("trial_y", {})
                custom_entry = section.get("custom_vs_ablated_direct", {}).get("trial_y", {})
                print(
                    "  - "
                    f"{label}: "
                    f"ablated_direct_Er_rel_err={float(direct_entry.get('Er_relative_error')):.6e} "
                    f"custom_vs_ablated_Er_rel_err={float(custom_entry.get('Er_relative_error')):.6e}"
                )
        return

    if report.get("realized_trace_sixth_step_carry_ablation_check"):
        print(
            f"[autodiff-gate] mode=realized_trace_sixth_step_carry_ablation "
            f"parameter={report['parameter_name']} "
            f"baseline_value={report['baseline_value']:.6e} "
            f"sixth_step_time={report.get('sixth_step_time'):.6e}"
        )
        path = report.get("rollout_path", {})
        diag = path.get("baseline", {})
        print(
            f"[autodiff-gate] rollout baseline: "
            f"attempt_count={diag.get('attempt_count')} "
            f"accepted_count={diag.get('accepted_count')} "
            f"completed={diag.get('completed')} "
            f"failed={diag.get('failed')} "
            f"fail_code={diag.get('fail_code')}"
        )
        print("[autodiff-gate] sixth-step comparisons:")
        for label in ("custom_vs_direct", "carry_after_step5_custom_vs_direct"):
            entry = report.get(label, {}).get("trial_y", {})
            print(
                "  - "
                f"{label}: "
                f"full_rel_err={float(entry.get('full_relative_error')):.6e} "
                f"pressure_rel_err={float(entry.get('pressure_relative_error')):.6e} "
                f"Er_rel_err={float(entry.get('Er_relative_error')):.6e}"
            )
        helper_consistency = report.get("helper_consistency") or {}
        if helper_consistency:
            print("[autodiff-gate] sixth-step helper consistency:")
            for label in ("custom_scan_vs_manual", "direct_scan_vs_manual"):
                entry = helper_consistency.get(label, {}).get("trial_y", {})
                print(
                    "  - "
                    f"{label}: "
                    f"full_rel_err={float(entry.get('full_relative_error')):.6e} "
                    f"pressure_rel_err={float(entry.get('pressure_relative_error')):.6e} "
                    f"Er_rel_err={float(entry.get('Er_relative_error')):.6e}"
                )
        ablations = report.get("carry_field_ablations") or {}
        if ablations:
            print("[autodiff-gate] sixth-step carry-field ablations:")
            for label, section in ablations.items():
                direct_entry = section.get("ablated_direct_vs_direct", {}).get("trial_y", {})
                custom_entry = section.get("custom_vs_ablated_direct", {}).get("trial_y", {})
                print(
                    "  - "
                    f"{label}: "
                    f"ablated_direct_Er_rel_err={float(direct_entry.get('Er_relative_error')):.6e} "
                    f"custom_vs_ablated_Er_rel_err={float(custom_entry.get('Er_relative_error')):.6e}"
                )
        return

    if report.get("realized_schedule_rollout_check"):
        print(
            f"[autodiff-gate] mode=realized_schedule_rollout "
            f"parameter={report['parameter_name']} "
            f"baseline_value={report['baseline_value']:.6e} "
            f"fd_step={report['fd_step']:.6e}"
        )
        path = report.get("rollout_path", {})
        for key in ("baseline", "fd_minus", "fd_plus"):
            diag = path.get(key, {})
            print(
                f"[autodiff-gate] rollout {key}: "
                f"attempt_count={diag.get('attempt_count')} "
                f"accepted_count={diag.get('accepted_count')} "
                f"completed={diag.get('completed')} "
                f"failed={diag.get('failed')} "
                f"fail_code={diag.get('fail_code')}"
            )
            print(
                f"[autodiff-gate] rollout {key} path: "
                f"accepted_mask={diag.get('accepted_mask')} "
                f"attempted_dts={diag.get('attempted_dts')} "
                f"next_dts={diag.get('next_dts')}"
            )
        print("[autodiff-gate] objective errors:")
        for label, ad, fd, ae, re in zip(
            report["objective_labels"],
            report["gradient_autodiff"],
            report["gradient_fd"],
            report["gradient_absolute_error"],
            report["gradient_relative_error"],
        ):
            print(
                f"  - {label}: ad={float(ad):.6e} fd={float(fd):.6e} "
                f"abs_err={float(ae):.6e} rel_err={float(re):.6e}"
            )
        nan_debug = report.get("nan_debug")
        if nan_debug is not None:
            print(
                "[autodiff-gate] replay NaN debug: "
                f"first_bad_index={nan_debug.get('first_bad_index')} "
                f"first_bad_was_accepted={nan_debug.get('first_bad_was_accepted')} "
                f"first_bad_dt={_fmt_float(nan_debug.get('first_bad_dt'))} "
                f"final_tangent_finite={nan_debug.get('final_tangent_finite')}"
            )
            lagged_cache_zeroed_debug = nan_debug.get("lagged_cache_zeroed_debug")
            if lagged_cache_zeroed_debug is not None:
                print(
                    "[autodiff-gate] replay NaN debug with lagged-cache tangent zeroed: "
                    f"first_bad_index={lagged_cache_zeroed_debug.get('first_bad_index')} "
                    f"first_bad_was_accepted={lagged_cache_zeroed_debug.get('first_bad_was_accepted')} "
                    f"first_bad_dt={_fmt_float(lagged_cache_zeroed_debug.get('first_bad_dt'))} "
                    f"final_tangent_finite={lagged_cache_zeroed_debug.get('final_tangent_finite')}"
                )
            prev_stages_zeroed_debug = nan_debug.get("prev_stages_zeroed_debug")
            if prev_stages_zeroed_debug is not None:
                print(
                    "[autodiff-gate] replay NaN debug with prev-stages tangent zeroed: "
                    f"first_bad_index={prev_stages_zeroed_debug.get('first_bad_index')} "
                    f"first_bad_was_accepted={prev_stages_zeroed_debug.get('first_bad_was_accepted')} "
                    f"first_bad_dt={_fmt_float(prev_stages_zeroed_debug.get('first_bad_dt'))} "
                    f"final_tangent_finite={prev_stages_zeroed_debug.get('final_tangent_finite')}"
                )
            prev_stages_and_lagged_zeroed_debug = nan_debug.get("prev_stages_and_lagged_zeroed_debug")
            if prev_stages_and_lagged_zeroed_debug is not None:
                print(
                    "[autodiff-gate] replay NaN debug with prev-stages and lagged-cache tangents zeroed: "
                    f"first_bad_index={prev_stages_and_lagged_zeroed_debug.get('first_bad_index')} "
                    f"first_bad_was_accepted={prev_stages_and_lagged_zeroed_debug.get('first_bad_was_accepted')} "
                    f"first_bad_dt={_fmt_float(prev_stages_and_lagged_zeroed_debug.get('first_bad_dt'))} "
                    f"final_tangent_finite={prev_stages_and_lagged_zeroed_debug.get('final_tangent_finite')}"
                )
            y_zeroed_debug = nan_debug.get("y_zeroed_debug")
            if y_zeroed_debug is not None:
                print(
                    "[autodiff-gate] replay NaN debug with y tangent zeroed: "
                    f"first_bad_index={y_zeroed_debug.get('first_bad_index')} "
                    f"first_bad_was_accepted={y_zeroed_debug.get('first_bad_was_accepted')} "
                    f"first_bad_dt={_fmt_float(y_zeroed_debug.get('first_bad_dt'))} "
                    f"final_tangent_finite={y_zeroed_debug.get('final_tangent_finite')}"
                )
            prev_error_zeroed_debug = nan_debug.get("prev_error_zeroed_debug")
            if prev_error_zeroed_debug is not None:
                print(
                    "[autodiff-gate] replay NaN debug with prev-error tangent zeroed: "
                    f"first_bad_index={prev_error_zeroed_debug.get('first_bad_index')} "
                    f"first_bad_was_accepted={prev_error_zeroed_debug.get('first_bad_was_accepted')} "
                    f"first_bad_dt={_fmt_float(prev_error_zeroed_debug.get('first_bad_dt'))} "
                    f"final_tangent_finite={prev_error_zeroed_debug.get('final_tangent_finite')}"
                )
            y_and_prev_error_zeroed_debug = nan_debug.get("y_and_prev_error_zeroed_debug")
            if y_and_prev_error_zeroed_debug is not None:
                print(
                    "[autodiff-gate] replay NaN debug with y and prev-error tangents zeroed: "
                    f"first_bad_index={y_and_prev_error_zeroed_debug.get('first_bad_index')} "
                    f"first_bad_was_accepted={y_and_prev_error_zeroed_debug.get('first_bad_was_accepted')} "
                    f"first_bad_dt={_fmt_float(y_and_prev_error_zeroed_debug.get('first_bad_dt'))} "
                    f"final_tangent_finite={y_and_prev_error_zeroed_debug.get('final_tangent_finite')}"
                )
            all_tangents_zeroed_debug = nan_debug.get("all_tangents_zeroed_debug")
            if all_tangents_zeroed_debug is not None:
                print(
                    "[autodiff-gate] replay NaN debug with all tangents zeroed: "
                    f"first_bad_index={all_tangents_zeroed_debug.get('first_bad_index')} "
                    f"first_bad_was_accepted={all_tangents_zeroed_debug.get('first_bad_was_accepted')} "
                    f"first_bad_dt={_fmt_float(all_tangents_zeroed_debug.get('first_bad_dt'))} "
                    f"final_tangent_finite={all_tangents_zeroed_debug.get('final_tangent_finite')}"
                )
            zero_tangent_one_step = nan_debug.get("zero_tangent_one_step")
            if zero_tangent_one_step is not None:
                print(
                    "[autodiff-gate] one-step zero-tangent compare: "
                    f"target_attempt_index={zero_tangent_one_step.get('target_attempt_index')} "
                    f"target_was_accepted={zero_tangent_one_step.get('target_was_accepted')} "
                    f"trial_dt={_fmt_float(zero_tangent_one_step.get('trial_dt'))} "
                    f"custom_trial_y_max_abs={_fmt_float(zero_tangent_one_step.get('custom_trial_y_max_abs'))} "
                    f"custom_stage_history_max_abs={_fmt_float(zero_tangent_one_step.get('custom_stage_history_max_abs'))} "
                    f"custom_finite={zero_tangent_one_step.get('custom_finite')} "
                    f"direct_trial_y_max_abs={_fmt_float(zero_tangent_one_step.get('direct_trial_y_max_abs'))} "
                    f"direct_stage_history_max_abs={_fmt_float(zero_tangent_one_step.get('direct_stage_history_max_abs'))} "
                    f"direct_finite={zero_tangent_one_step.get('direct_finite')}"
                )
            local_attempt_window = nan_debug.get("local_attempt_window") or []
            if local_attempt_window:
                print("[autodiff-gate] replay NaN local window:")
                for entry in local_attempt_window:
                    print(
                        f"  - index={entry.get('index')} "
                        f"accepted={entry.get('accepted')} "
                        f"attempted_dt={_fmt_float(entry.get('attempted_dt'))} "
                        f"next_dt={_fmt_float(entry.get('next_dt'))} "
                        f"err_norm={_fmt_float(entry.get('err_norm'))} "
                        f"theta_final={_fmt_float(entry.get('theta_final'))} "
                        f"newton_iter_count={entry.get('newton_iter_count')} "
                        f"cache_valid_next={entry.get('cache_valid_next')} "
                        f"tangent_finite={entry.get('tangent_finite')} "
                        f"dt_dot_abs={_fmt_float(entry.get('dt_dot_abs'))} "
                        f"prev_error_dot_abs={_fmt_float(entry.get('prev_error_dot_abs'))} "
                        f"density_dot_max_abs={_fmt_float(entry.get('density_dot_max_abs'))} "
                        f"pressure_dot_max_abs={_fmt_float(entry.get('pressure_dot_max_abs'))} "
                        f"er_dot_max_abs={_fmt_float(entry.get('er_dot_max_abs'))} "
                        f"y_dot_max_abs={_fmt_float(entry.get('y_dot_max_abs'))} "
                        f"prev_stages_dot_max_abs={_fmt_float(entry.get('prev_stages_dot_max_abs'))} "
                        f"lagged_cache_dot_max_abs={_fmt_float(entry.get('lagged_response_cache_dot_max_abs'))} "
                        f"jacobian_dot_max_abs={_fmt_float(entry.get('jacobian_dot_max_abs'))} "
                        f"real_lu_dot_max_abs={_fmt_float(entry.get('real_lu_dot_max_abs'))} "
                        f"complex_lu_dot_max_abs={_fmt_float(entry.get('complex_lu_dot_max_abs'))}"
                    )
        return

    if report.get("realized_schedule_direct_ad_compare_check"):
        print(
            f"[autodiff-gate] mode=realized_schedule_direct_ad_compare "
            f"parameter={report['parameter_name']} "
            f"baseline_value={report['baseline_value']:.6e} "
            f"fd_step={report['fd_step']:.6e}"
        )
        path = report.get("rollout_path", {})
        for key in ("baseline", "fd_minus", "fd_plus"):
            diag = path.get(key, {})
            print(
                f"[autodiff-gate] rollout {key}: "
                f"attempt_count={diag.get('attempt_count')} "
                f"accepted_count={diag.get('accepted_count')} "
                f"completed={diag.get('completed')} "
                f"failed={diag.get('failed')} "
                f"fail_code={diag.get('fail_code')}"
            )
        print(
            "[autodiff-gate] fd path parity: "
            f"accepted_mask_equal_minus_plus={path.get('accepted_mask_equal_minus_plus')} "
            f"attempted_dts_equal_minus_plus={path.get('attempted_dts_equal_minus_plus')} "
            f"next_dts_equal_minus_plus={path.get('next_dts_equal_minus_plus')}"
        )
        print("[autodiff-gate] objective errors:")
        for label, cad, dad, fd, cfre, dfre, cdre in zip(
            report["objective_labels"],
            report["gradient_custom_autodiff"],
            report["gradient_direct_autodiff"],
            report["gradient_fd"],
            report["gradient_custom_vs_fd_relative_error"],
            report["gradient_direct_vs_fd_relative_error"],
            report["gradient_custom_vs_direct_relative_error"],
        ):
            print(
                f"  - {label}: "
                f"custom_ad={float(cad):.6e} "
                f"direct_ad={float(dad):.6e} "
                f"fd={float(fd):.6e} "
                f"custom_vs_fd_rel_err={float(cfre):.6e} "
                f"direct_vs_fd_rel_err={float(dfre):.6e} "
                f"custom_vs_direct_rel_err={float(cdre):.6e}"
            )
        return

    print(
        f"[autodiff-gate] mode={'one_step' if report.get('one_step_diagnostic') else 'full_solve'} "
        f"parameter={report['parameter_name']} "
        f"baseline_value={report['baseline_value']:.6e} "
        f"fd_step={report['fd_step']:.6e}"
    )
    path = report.get("solver_path", {})
    for key in ("baseline", "fd_minus", "fd_plus"):
        diag = path.get(key, {})
        last_attempt = diag.get("last_attempt", {})
        print(
            f"[autodiff-gate] path {key}: "
            f"n_steps={diag.get('n_steps')} "
            f"accepted_count={diag.get('accepted_count')} "
            f"failed_any={diag.get('failed_any')}"
        )
        print(
            f"[autodiff-gate] path {key} last_attempt: "
            f"accepted={last_attempt.get('accepted')} "
            f"converged={last_attempt.get('converged')} "
            f"fail_code={last_attempt.get('fail_code')} "
            f"newton_iter_count={last_attempt.get('newton_iter_count')} "
            f"theta_final={_fmt_float(last_attempt.get('theta_final'))} "
            f"err_norm={_fmt_float(last_attempt.get('err_norm'))} "
            f"final_residual_norm={_fmt_float(last_attempt.get('final_residual_norm'))} "
            f"final_delta_norm={_fmt_float(last_attempt.get('final_delta_norm'))}"
        )
        print(
            f"[autodiff-gate] path {key} saved_signature: "
            f"times={diag.get('saved_times')} "
            f"dts={diag.get('saved_step_sizes')}"
        )
    print(
        "[autodiff-gate] fd path parity:",
        f"accepted_mask_equal_minus_plus={path.get('accepted_mask_equal_minus_plus')} "
        f"saved_times_equal_minus_plus={path.get('saved_times_equal_minus_plus')} "
        f"saved_dts_equal_minus_plus={path.get('saved_dts_equal_minus_plus')}",
    )
    print("[autodiff-gate] objective errors:")
    for label, j_ad, j_fd, abs_err, rel_err in zip(
        report["objective_labels"],
        report["gradient_autodiff"],
        report["gradient_fd"],
        report["gradient_absolute_error"],
        report["gradient_relative_error"],
    ):
        print(
            f"  - {label}: "
            f"ad={float(j_ad):.6e} "
            f"fd={float(j_fd):.6e} "
            f"abs_err={float(abs_err):.6e} "
            f"rel_err={float(rel_err):.6e}"
        )
    fd_sweep = report.get("fd_step_sweep")
    if fd_sweep:
        print("[autodiff-gate] fd step sweep:")
        for entry in fd_sweep:
            print(
                f"  - scale={float(entry['scale']):.6e} "
                f"fd_step={float(entry['fd_step']):.6e} "
                f"max_rel_err={float(entry['max_relative_error']):.6e} "
                f"n_steps_minus={entry['n_steps_minus']} "
                f"n_steps_plus={entry['n_steps_plus']} "
                f"saved_times_equal={entry['saved_times_equal_minus_plus']} "
                f"saved_dts_equal={entry['saved_dts_equal_minus_plus']}"
            )
    standalone = report.get("standalone_stage_subsolve")
    if standalone:
        print("[autodiff-gate] standalone stage subsolve errors:")
        for label, j_ad, j_fd, abs_err, rel_err in zip(
            standalone["labels"],
            standalone["gradient_autodiff"],
            standalone["gradient_fd"],
            standalone["gradient_absolute_error"],
            standalone["gradient_relative_error"],
        ):
            print(
                f"  - {label}: "
                f"ad={float(j_ad):.6e} "
                f"fd={float(j_fd):.6e} "
                f"abs_err={float(abs_err):.6e} "
                f"rel_err={float(rel_err):.6e}"
            )
    small_step = report.get("small_step_composition")
    if small_step:
        print("[autodiff-gate] small-step composition errors:")
        for entry in small_step:
            print(
                f"  - step_count={int(entry['step_count'])} "
                f"step_scale={float(entry['step_scale']):.6e} "
                f"max_rel_err={float(entry['max_relative_error']):.6e}"
            )
    controller_step = report.get("controller_step_composition")
    if controller_step:
        print("[autodiff-gate] controller-step composition errors:")
        for entry in controller_step:
            print(
                f"  - step_count={int(entry['step_count'])} "
                f"step_scale={float(entry['step_scale']):.6e} "
                f"max_rel_err={float(entry['max_relative_error']):.6e}"
            )
def _fd_step_sweep_report(
    *,
    runtime,
    config: dict[str, Any],
    baseline_state,
    profile_cfg: dict[str, Any],
    parameter_name: str,
    baseline_value: float,
    gradient_ad: jax.Array,
    fd_step: float,
    step_multipliers: tuple[float, ...],
) -> list[dict[str, Any]]:
    grad_ad_np = np.asarray(jax.device_get(gradient_ad), dtype=float)
    entries: list[dict[str, Any]] = []
    seen_steps: set[float] = set()

    for scale in step_multipliers:
        step_value = float(fd_step * scale)
        if step_value <= 0.0:
            continue
        rounded_key = round(step_value, 18)
        if rounded_key in seen_steps:
            continue
        seen_steps.add(rounded_key)
        minus_value = baseline_value - step_value
        plus_value = baseline_value + step_value
        minus_result = run_transport(
            config,
            runtime,
            _parameterized_initial_state(
                baseline_state=baseline_state,
                profile_cfg=profile_cfg,
                geometry=runtime.geometry,
                n_species=runtime.species.number_species,
                parameter_name=parameter_name,
                parameter_value=jnp.asarray(minus_value),
            ),
        )
        plus_result = run_transport(
            config,
            runtime,
            _parameterized_initial_state(
                baseline_state=baseline_state,
                profile_cfg=profile_cfg,
                geometry=runtime.geometry,
                n_species=runtime.species.number_species,
                parameter_name=parameter_name,
                parameter_value=jnp.asarray(plus_value),
            ),
        )
        objectives_minus = _objective_vector(minus_result["final_state"], runtime)
        objectives_plus = _objective_vector(plus_result["final_state"], runtime)
        gradient_fd = (objectives_plus - objectives_minus) / (2.0 * step_value)
        grad_fd_np = np.asarray(jax.device_get(gradient_fd), dtype=float)
        abs_err = np.abs(grad_ad_np - grad_fd_np)
        rel_err = abs_err / np.maximum(np.abs(grad_fd_np), 1.0e-10)
        minus_diag = _result_diagnostics(minus_result)
        plus_diag = _result_diagnostics(plus_result)
        entries.append(
            {
                "scale": float(scale),
                "fd_step": float(step_value),
                "gradient_fd": grad_fd_np.tolist(),
                "gradient_absolute_error": abs_err.tolist(),
                "gradient_relative_error": rel_err.tolist(),
                "max_relative_error": float(np.max(rel_err)),
                "n_steps_minus": minus_diag["n_steps"],
                "n_steps_plus": plus_diag["n_steps"],
                "saved_times_equal_minus_plus": _sequence_allclose(
                    minus_diag["saved_times"],
                    plus_diag["saved_times"],
                ),
                "saved_dts_equal_minus_plus": _sequence_allclose(
                    minus_diag["saved_step_sizes"],
                    plus_diag["saved_step_sizes"],
                ),
            }
        )
    return entries


def _standalone_stage_subsolve_objectives_for_parameter(
    p,
    *,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    parameter_name: str,
):
    state0 = _parameterized_initial_state(
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        geometry=runtime.geometry,
        n_species=runtime.species.number_species,
        parameter_name=parameter_name,
        parameter_value=p,
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
    subsolve_inputs = _radau_prepare_stage_subsolve_inputs_from_carry(
        prepared_rollout.kernel_context,
        prepared_rollout.physics_context,
        prepared_rollout.initial_carry,
        t_final=solver.t1,
    )
    subsolve_result = _radau_run_stage_subsolve_standalone_autodiff(
        prepared_rollout.kernel_context,
        prepared_rollout.physics_context,
        subsolve_inputs,
    )
    return jnp.stack(
        [
            jnp.sum(subsolve_result.z_final),
            jnp.linalg.norm(subsolve_result.z_final),
            subsolve_result.final_residual_norm,
            subsolve_result.theta_final,
        ]
    )


def _small_step_composition_objectives_for_parameter(
    p,
    *,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    parameter_name: str,
    step_count: int,
    step_scale: float,
):
    state0 = _parameterized_initial_state(
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        geometry=runtime.geometry,
        n_species=runtime.species.number_species,
        parameter_name=parameter_name,
        parameter_value=p,
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
    kernel_context = prepared_rollout.kernel_context
    physics_context = prepared_rollout.physics_context
    carry0 = prepared_rollout.initial_carry
    base_dt = jnp.asarray(step_scale, dtype=kernel_context.dtype) * carry0.dt

    def _scan_body(carry, _):
        carry_for_step = dataclasses.replace(carry, dt=base_dt)
        attempt_context = _RadauAcceptedStepAttemptContext(
            t_final=carry.t + base_dt,
            use_transport_lagged_response=jnp.asarray(kernel_context.use_transport_lagged_response),
        )
        step_map_result = _radau_apply_accepted_step_map(
            kernel_context,
            physics_context,
            carry_for_step,
            attempt_context,
        )
        return step_map_result.next_carry, step_map_result.err_norm

    final_carry, _err_norms = jax.lax.scan(
        _scan_body,
        carry0,
        xs=jnp.arange(int(step_count), dtype=jnp.int32),
    )
    final_state = physics_context.unpack_flat(final_carry.y)
    return _objective_vector(final_state, runtime)


def _small_step_composition_report(
    *,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    parameter_name: str,
    baseline_value: float,
    fd_step: float,
    small_step_counts: tuple[float, ...],
    small_step_scale: float,
) -> list[dict[str, Any]]:
    entries = []
    minus_value = baseline_value - fd_step
    plus_value = baseline_value + fd_step

    print(
        f"[autodiff-gate] realized-schedule progress: preparing baseline rollout "
        f"(parameter={parameter_name}, baseline_value={baseline_value:.6e}, fd_step={fd_step:.6e})",
        flush=True,
    )
    for raw_count in small_step_counts:
        step_count = int(raw_count)
        composition_objective_fn = lambda p: _small_step_composition_objectives_for_parameter(  # noqa: E731
            p,
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=parameter_name,
            step_count=step_count,
            step_scale=small_step_scale,
        )
        comp_ad = jax.jacfwd(composition_objective_fn)(jnp.asarray(baseline_value))
        comp_minus = np.asarray(
            jax.device_get(composition_objective_fn(jnp.asarray(minus_value))),
            dtype=float,
        )
        comp_plus = np.asarray(
            jax.device_get(composition_objective_fn(jnp.asarray(plus_value))),
            dtype=float,
        )
        comp_fd = (comp_plus - comp_minus) / (2.0 * fd_step)
        comp_ad_np = np.asarray(jax.device_get(comp_ad), dtype=float)
        comp_abs_err = np.abs(comp_ad_np - comp_fd)
        comp_rel_err = comp_abs_err / np.maximum(np.abs(comp_fd), 1.0e-10)
        entries.append(
            {
                "step_count": int(step_count),
                "step_scale": float(small_step_scale),
                "gradient_autodiff": comp_ad_np.tolist(),
                "gradient_fd": comp_fd.tolist(),
                "gradient_absolute_error": comp_abs_err.tolist(),
                "gradient_relative_error": comp_rel_err.tolist(),
                "max_relative_error": float(np.max(comp_rel_err)),
                "labels": OBJECTIVE_LABELS,
            }
        )
    return entries


def _controller_rollout_for_parameter(
    parameter_value,
    *,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    parameter_name: str,
    step_count: int,
    step_scale: float,
    forward_only_controller: bool = False,
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
    rollout = (
        _radau_controller_forward_only_rollout(
            execution_context,
            prepared_rollout.initial_carry,
            step_count=step_count,
            dt_scale=step_scale,
        )
        if forward_only_controller
        else _radau_controller_composed_rollout(
            execution_context,
            prepared_rollout.initial_carry,
            step_count=step_count,
            dt_scale=step_scale,
        )
    )
    return rollout, runtime, prepared_rollout


def _controller_composition_objectives_for_parameter(
    p,
    *,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    parameter_name: str,
    step_count: int,
    step_scale: float,
):
    rollout, runtime, prepared_rollout = _controller_rollout_for_parameter(
        p,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        step_count=step_count,
        step_scale=step_scale,
    )
    final_state = prepared_rollout.physics_context.unpack_flat(rollout.final_carry.y)
    return _objective_vector(final_state, runtime)


def _controller_rollout_summary(rollout) -> dict[str, Any]:
    return {
        "accepted_mask": np.asarray(jax.device_get(rollout.accepted_mask), dtype=bool).tolist(),
        "attempted_dts": np.asarray(jax.device_get(rollout.attempted_dts), dtype=float).tolist(),
        "next_dts": np.asarray(jax.device_get(rollout.next_dts), dtype=float).tolist(),
        "err_norms": np.asarray(jax.device_get(rollout.err_norms), dtype=float).tolist(),
    }


def _controller_multi_objectives_for_parameter(
    parameter_value,
    *,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    parameter_name: str,
    step_counts: tuple[int, ...],
    step_scale: float,
    forward_only_controller: bool = False,
):
    max_step_count = int(max(step_counts))
    rollout, runtime, prepared_rollout = _controller_rollout_for_parameter(
        parameter_value,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        step_count=max_step_count,
        step_scale=step_scale,
        forward_only_controller=forward_only_controller,
    )
    unpack_flat = prepared_rollout.physics_context.unpack_flat
    objectives = []
    for step_count in step_counts:
        flat_y = rollout.step_ys[int(step_count) - 1]
        final_state = unpack_flat(flat_y)
        objectives.append(_objective_vector(final_state, runtime))
    return jnp.stack(objectives, axis=0), rollout


def _controller_composition_report(
    *,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    parameter_name: str,
    baseline_value: float,
    fd_step: float,
    small_step_counts: tuple[float, ...],
    small_step_scale: float,
    forward_only_controller: bool = False,
) -> list[dict[str, Any]]:
    step_counts = tuple(int(v) for v in small_step_counts)
    minus_value = baseline_value - fd_step
    plus_value = baseline_value + fd_step
    composition_objective_fn = lambda p: _controller_multi_objectives_for_parameter(  # noqa: E731
        p,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        step_counts=step_counts,
        step_scale=small_step_scale,
        forward_only_controller=forward_only_controller,
    )[0]
    comp_ad = jax.jacfwd(composition_objective_fn)(jnp.asarray(baseline_value))
    comp_minus, minus_rollout = _controller_multi_objectives_for_parameter(
        jnp.asarray(minus_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        step_counts=step_counts,
        step_scale=small_step_scale,
        forward_only_controller=forward_only_controller,
    )
    comp_plus, plus_rollout = _controller_multi_objectives_for_parameter(
        jnp.asarray(plus_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        step_counts=step_counts,
        step_scale=small_step_scale,
        forward_only_controller=forward_only_controller,
    )
    baseline_objectives, baseline_rollout = _controller_multi_objectives_for_parameter(
        jnp.asarray(baseline_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        step_counts=step_counts,
        step_scale=small_step_scale,
        forward_only_controller=forward_only_controller,
    )
    comp_ad_np = np.asarray(jax.device_get(comp_ad), dtype=float)
    comp_minus_np = np.asarray(jax.device_get(comp_minus), dtype=float)
    comp_plus_np = np.asarray(jax.device_get(comp_plus), dtype=float)
    comp_fd_np = (comp_plus_np - comp_minus_np) / (2.0 * fd_step)
    entries = []
    diag_baseline = _controller_rollout_summary(baseline_rollout)
    diag_minus = _controller_rollout_summary(minus_rollout)
    diag_plus = _controller_rollout_summary(plus_rollout)
    for idx, step_count in enumerate(step_counts):
        comp_abs_err = np.abs(comp_ad_np[idx] - comp_fd_np[idx])
        comp_rel_err = comp_abs_err / np.maximum(np.abs(comp_fd_np[idx]), 1.0e-10)
        entries.append(
            {
                "step_count": int(step_count),
                "step_scale": float(small_step_scale),
                "baseline_objectives": np.asarray(jax.device_get(baseline_objectives[idx]), dtype=float).tolist(),
                "gradient_autodiff": comp_ad_np[idx].tolist(),
                "gradient_fd": comp_fd_np[idx].tolist(),
                "gradient_absolute_error": comp_abs_err.tolist(),
                "gradient_relative_error": comp_rel_err.tolist(),
                "max_relative_error": float(np.max(comp_rel_err)),
                "labels": OBJECTIVE_LABELS,
                "controller_paths": {
                    "baseline": {
                        "accepted_mask": diag_baseline["accepted_mask"][:step_count],
                        "attempted_dts": diag_baseline["attempted_dts"][:step_count],
                        "next_dts": diag_baseline["next_dts"][:step_count],
                        "err_norms": diag_baseline["err_norms"][:step_count],
                    },
                    "fd_minus": {
                        "accepted_mask": diag_minus["accepted_mask"][:step_count],
                        "attempted_dts": diag_minus["attempted_dts"][:step_count],
                        "next_dts": diag_minus["next_dts"][:step_count],
                        "err_norms": diag_minus["err_norms"][:step_count],
                    },
                    "fd_plus": {
                        "accepted_mask": diag_plus["accepted_mask"][:step_count],
                        "attempted_dts": diag_plus["attempted_dts"][:step_count],
                        "next_dts": diag_plus["next_dts"][:step_count],
                        "err_norms": diag_plus["err_norms"][:step_count],
                    },
                    "accepted_mask_equal_minus_plus": diag_minus["accepted_mask"][:step_count] == diag_plus["accepted_mask"][:step_count],
                    "attempted_dts_equal_minus_plus": _sequence_allclose(
                        diag_minus["attempted_dts"][:step_count],
                        diag_plus["attempted_dts"][:step_count],
                    ),
                    "next_dts_equal_minus_plus": _sequence_allclose(
                        diag_minus["next_dts"][:step_count],
                        diag_plus["next_dts"][:step_count],
                    ),
                },
            }
        )
    return entries


def _controller_only_report(
    *,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    parameter_name: str,
    baseline_value: float,
    fd_step: float,
    small_step_counts: tuple[float, ...],
    small_step_scale: float,
) -> list[dict[str, Any]]:
    return _controller_composition_report(
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        baseline_value=baseline_value,
        fd_step=fd_step,
        small_step_counts=small_step_counts,
        small_step_scale=small_step_scale,
    )


def build_controller_only_report(
    *,
    config_path: Path,
    parameter_name: str,
    rel_fd_step: float,
    abs_fd_step: float,
    small_step_counts: tuple[float, ...],
    small_step_scale: float,
    device: str | None,
) -> dict[str, Any]:
    if parameter_name not in ALLOWED_PARAMETERS:
        raise ValueError(f"parameter_name must be one of {sorted(ALLOWED_PARAMETERS)}")

    config = _prepare_benchmark_config(config_path, device=device)
    runtime, baseline_state = build_runtime_context(config)
    profile_cfg = _baseline_profile_cfg(config)
    baseline_value = float(profile_cfg[parameter_name])
    fd_step = _fd_step(baseline_value, rel_step=rel_fd_step, abs_step=abs_fd_step)
    controller_step_composition = _controller_only_report(
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        baseline_value=baseline_value,
        fd_step=fd_step,
        small_step_counts=small_step_counts,
        small_step_scale=small_step_scale,
    )
    max_rel_error = max(float(entry["max_relative_error"]) for entry in controller_step_composition)
    return {
        "config_path": str(config_path),
        "controller_only_check": True,
        "parameter_name": parameter_name,
        "baseline_value": baseline_value,
        "fd_step": float(fd_step),
        "controller_step_composition": controller_step_composition,
        "passed": bool(np.isfinite(max_rel_error) and max_rel_error <= 5.0e-2),
        "max_relative_error": float(max_rel_error),
        "objective_labels": OBJECTIVE_LABELS,
    }


def build_forward_only_controller_report(
    *,
    config_path: Path,
    parameter_name: str,
    rel_fd_step: float,
    abs_fd_step: float,
    small_step_counts: tuple[float, ...],
    small_step_scale: float,
    device: str | None,
) -> dict[str, Any]:
    if parameter_name not in ALLOWED_PARAMETERS:
        raise ValueError(f"parameter_name must be one of {sorted(ALLOWED_PARAMETERS)}")

    config = _prepare_benchmark_config(config_path, device=device)
    runtime, baseline_state = build_runtime_context(config)
    profile_cfg = _baseline_profile_cfg(config)
    baseline_value = float(profile_cfg[parameter_name])
    fd_step = _fd_step(baseline_value, rel_step=rel_fd_step, abs_step=abs_fd_step)
    controller_step_composition = _controller_composition_report(
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        baseline_value=baseline_value,
        fd_step=fd_step,
        small_step_counts=small_step_counts,
        small_step_scale=small_step_scale,
        forward_only_controller=True,
    )
    max_rel_error = max(float(entry["max_relative_error"]) for entry in controller_step_composition)
    return {
        "config_path": str(config_path),
        "forward_only_controller_check": True,
        "parameter_name": parameter_name,
        "baseline_value": baseline_value,
        "fd_step": float(fd_step),
        "controller_step_composition": controller_step_composition,
        "passed": bool(np.isfinite(max_rel_error) and max_rel_error <= 5.0e-2),
        "max_relative_error": float(max_rel_error),
        "objective_labels": OBJECTIVE_LABELS,
    }


def build_realized_schedule_rollout_report(
    *,
    config_path: Path,
    parameter_name: str,
    rel_fd_step: float,
    abs_fd_step: float,
    device: str | None,
    include_nan_debug: bool = False,
    nan_debug_mode: str = "minimal",
    nan_debug_include_one_step_compare: bool = False,
) -> dict[str, Any]:
    if parameter_name not in ALLOWED_PARAMETERS:
        raise ValueError(f"parameter_name must be one of {sorted(ALLOWED_PARAMETERS)}")

    config = _prepare_benchmark_config(config_path, device=device)
    runtime, baseline_state = build_runtime_context(config)
    profile_cfg = _baseline_profile_cfg(config)
    baseline_value = float(profile_cfg[parameter_name])
    fd_step = _fd_step(baseline_value, rel_step=rel_fd_step, abs_step=abs_fd_step)
    minus_value = baseline_value - fd_step
    plus_value = baseline_value + fd_step

    objective_fn = lambda p: _adaptive_rollout_objectives_for_parameter(  # noqa: E731
        p,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        use_realized_schedule_jvp=True,
    )[0]

    if accepted_step_limit is None:
        baseline_objectives, baseline_rollout = _adaptive_rollout_objectives_for_parameter(
            jnp.asarray(baseline_value),
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=parameter_name,
            use_realized_schedule_jvp=True,
        )
    else:
        baseline_final_state, baseline_rollout = _adaptive_rollout_final_state_for_parameter(
            jnp.asarray(baseline_value),
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=parameter_name,
            use_realized_schedule_jvp=False,
            accepted_step_limit_override=accepted_step_limit,
        )
        baseline_objectives = _objective_vector(baseline_final_state, runtime)
    print("[autodiff-gate] realized-schedule progress: baseline rollout complete; running fd_minus rollout", flush=True)
    objectives_minus, minus_rollout = _adaptive_rollout_objectives_for_parameter(
        jnp.asarray(minus_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        use_realized_schedule_jvp=True,
    )
    print("[autodiff-gate] realized-schedule progress: fd_minus rollout complete; running fd_plus rollout", flush=True)
    objectives_plus, plus_rollout = _adaptive_rollout_objectives_for_parameter(
        jnp.asarray(plus_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        use_realized_schedule_jvp=True,
    )
    print("[autodiff-gate] realized-schedule progress: fd_plus rollout complete; running AD gradient", flush=True)

    gradient_ad = jax.jacfwd(objective_fn)(jnp.asarray(baseline_value))
    print("[autodiff-gate] realized-schedule progress: AD gradient complete; forming FD gradient", flush=True)
    gradient_fd = (objectives_plus - objectives_minus) / (2.0 * fd_step)

    grad_ad_np = np.asarray(jax.device_get(gradient_ad), dtype=float)
    grad_fd_np = np.asarray(jax.device_get(gradient_fd), dtype=float)
    abs_err = np.abs(grad_ad_np - grad_fd_np)
    rel_err = abs_err / np.maximum(np.abs(grad_fd_np), 1.0e-10)

    baseline_diag = _adaptive_rollout_diagnostics(baseline_rollout)
    minus_diag = _adaptive_rollout_diagnostics(minus_rollout)
    plus_diag = _adaptive_rollout_diagnostics(plus_rollout)
    nan_debug = None
    if include_nan_debug and not np.all(np.isfinite(grad_ad_np)):
        print("[autodiff-gate] realized-schedule progress: AD produced nonfinite values; running NaN localization", flush=True)
        nan_debug = _adaptive_rollout_nan_debug_for_parameter(
            jnp.asarray(baseline_value),
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=parameter_name,
            debug_mode=nan_debug_mode,
            include_one_step_compare=nan_debug_include_one_step_compare,
        )
        print("[autodiff-gate] realized-schedule progress: NaN localization complete", flush=True)

    return {
        "config_path": str(config_path),
        "realized_schedule_rollout_check": True,
        "parameter_name": parameter_name,
        "baseline_value": baseline_value,
        "fd_step": float(fd_step),
        "baseline_objectives": np.asarray(jax.device_get(baseline_objectives), dtype=float).tolist(),
        "gradient_autodiff": grad_ad_np.tolist(),
        "gradient_fd": grad_fd_np.tolist(),
        "gradient_absolute_error": abs_err.tolist(),
        "gradient_relative_error": rel_err.tolist(),
        "max_relative_error": float(np.max(rel_err)),
        "passed": bool(np.all(np.isfinite(rel_err)) and np.max(rel_err) <= 5.0e-2),
        "objective_labels": OBJECTIVE_LABELS,
        "rollout_path": {
            "baseline": baseline_diag,
            "fd_minus": minus_diag,
            "fd_plus": plus_diag,
            "accepted_mask_equal_minus_plus": minus_diag["accepted_mask"] == plus_diag["accepted_mask"],
            "attempted_dts_equal_minus_plus": _sequence_allclose(
                minus_diag["attempted_dts"],
                plus_diag["attempted_dts"],
            ),
            "next_dts_equal_minus_plus": _sequence_allclose(
                minus_diag["next_dts"],
                plus_diag["next_dts"],
            ),
        },
        "nan_debug": nan_debug,
    }


def build_realized_schedule_direct_ad_compare_report(
    *,
    config_path: Path,
    parameter_name: str,
    rel_fd_step: float,
    abs_fd_step: float,
    device: str | None,
) -> dict[str, Any]:
    if parameter_name not in ALLOWED_PARAMETERS:
        raise ValueError(f"parameter_name must be one of {sorted(ALLOWED_PARAMETERS)}")

    config = _prepare_benchmark_config(config_path, device=device)
    runtime, baseline_state = build_runtime_context(config)
    profile_cfg = _baseline_profile_cfg(config)
    baseline_value = float(profile_cfg[parameter_name])
    fd_step = _fd_step(baseline_value, rel_step=rel_fd_step, abs_step=abs_fd_step)
    minus_value = baseline_value - fd_step
    plus_value = baseline_value + fd_step

    custom_objective_fn = lambda p: _adaptive_rollout_objectives_for_parameter(  # noqa: E731
        p,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        use_realized_schedule_jvp=True,
    )[0]
    direct_objective_fn = lambda p: _adaptive_rollout_objectives_for_parameter(  # noqa: E731
        p,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        use_realized_schedule_jvp=False,
    )[0]

    baseline_objectives, baseline_rollout = _adaptive_rollout_objectives_for_parameter(
        jnp.asarray(baseline_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        use_realized_schedule_jvp=True,
    )
    print(
        "[autodiff-gate] realized-schedule-direct-ad progress: baseline rollout complete; running fd_minus rollout",
        flush=True,
    )
    objectives_minus, minus_rollout = _adaptive_rollout_objectives_for_parameter(
        jnp.asarray(minus_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        use_realized_schedule_jvp=True,
    )
    print(
        "[autodiff-gate] realized-schedule-direct-ad progress: fd_minus rollout complete; running fd_plus rollout",
        flush=True,
    )
    objectives_plus, plus_rollout = _adaptive_rollout_objectives_for_parameter(
        jnp.asarray(plus_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        use_realized_schedule_jvp=True,
    )
    print(
        "[autodiff-gate] realized-schedule-direct-ad progress: fd_plus rollout complete; running custom AD",
        flush=True,
    )
    gradient_custom = jax.jacfwd(custom_objective_fn)(jnp.asarray(baseline_value))
    print(
        "[autodiff-gate] realized-schedule-direct-ad progress: custom AD complete; running direct adaptive AD",
        flush=True,
    )
    gradient_direct = jax.jacfwd(direct_objective_fn)(jnp.asarray(baseline_value))
    gradient_fd = (objectives_plus - objectives_minus) / (2.0 * fd_step)

    grad_custom_np = np.asarray(jax.device_get(gradient_custom), dtype=float)
    grad_direct_np = np.asarray(jax.device_get(gradient_direct), dtype=float)
    grad_fd_np = np.asarray(jax.device_get(gradient_fd), dtype=float)
    custom_vs_fd_abs = np.abs(grad_custom_np - grad_fd_np)
    custom_vs_fd_rel = custom_vs_fd_abs / np.maximum(np.abs(grad_fd_np), 1.0e-10)
    direct_vs_fd_abs = np.abs(grad_direct_np - grad_fd_np)
    direct_vs_fd_rel = direct_vs_fd_abs / np.maximum(np.abs(grad_fd_np), 1.0e-10)
    custom_vs_direct_abs = np.abs(grad_custom_np - grad_direct_np)
    custom_vs_direct_rel = custom_vs_direct_abs / np.maximum(np.abs(grad_direct_np), 1.0e-10)

    baseline_diag = _adaptive_rollout_diagnostics(baseline_rollout)
    minus_diag = _adaptive_rollout_diagnostics(minus_rollout)
    plus_diag = _adaptive_rollout_diagnostics(plus_rollout)

    return {
        "config_path": str(config_path),
        "realized_schedule_direct_ad_compare_check": True,
        "parameter_name": parameter_name,
        "baseline_value": baseline_value,
        "fd_step": float(fd_step),
        "baseline_objectives": np.asarray(jax.device_get(baseline_objectives), dtype=float).tolist(),
        "gradient_custom_autodiff": grad_custom_np.tolist(),
        "gradient_direct_autodiff": grad_direct_np.tolist(),
        "gradient_fd": grad_fd_np.tolist(),
        "gradient_custom_vs_fd_absolute_error": custom_vs_fd_abs.tolist(),
        "gradient_custom_vs_fd_relative_error": custom_vs_fd_rel.tolist(),
        "gradient_direct_vs_fd_absolute_error": direct_vs_fd_abs.tolist(),
        "gradient_direct_vs_fd_relative_error": direct_vs_fd_rel.tolist(),
        "gradient_custom_vs_direct_absolute_error": custom_vs_direct_abs.tolist(),
        "gradient_custom_vs_direct_relative_error": custom_vs_direct_rel.tolist(),
        "max_relative_error": float(np.max(custom_vs_fd_rel)),
        "passed": bool(np.all(np.isfinite(custom_vs_fd_rel)) and np.max(custom_vs_fd_rel) <= 5.0e-2),
        "objective_labels": OBJECTIVE_LABELS,
        "rollout_path": {
            "baseline": baseline_diag,
            "fd_minus": minus_diag,
            "fd_plus": plus_diag,
            "accepted_mask_equal_minus_plus": minus_diag["accepted_mask"] == plus_diag["accepted_mask"],
            "attempted_dts_equal_minus_plus": _sequence_allclose(
                minus_diag["attempted_dts"],
                plus_diag["attempted_dts"],
            ),
            "next_dts_equal_minus_plus": _sequence_allclose(
                minus_diag["next_dts"],
                plus_diag["next_dts"],
            ),
        },
    }


def build_realized_schedule_frozen_fd_report(
    *,
    config_path: Path,
    parameter_name: str,
    rel_fd_step: float,
    abs_fd_step: float,
    device: str | None,
    replay_mode: str = "attempt",
    accepted_step_limit: int | None = None,
    keep_adaptive_ad: bool = False,
    ad_only: bool = False,
) -> dict[str, Any]:
    if parameter_name not in ALLOWED_PARAMETERS:
        raise ValueError(f"parameter_name must be one of {sorted(ALLOWED_PARAMETERS)}")

    config = _prepare_benchmark_config(config_path, device=device)
    runtime, baseline_state = build_runtime_context(config)
    profile_cfg = _baseline_profile_cfg(config)
    baseline_value = float(profile_cfg[parameter_name])
    fd_step = _fd_step(baseline_value, rel_step=rel_fd_step, abs_step=abs_fd_step)
    minus_value = baseline_value - fd_step
    plus_value = baseline_value + fd_step

    if ad_only and keep_adaptive_ad:
        if accepted_step_limit is not None:
            print(
                "[autodiff-gate] adaptive AD-only mode ignores "
                "--realized-schedule-frozen-accepted-steps because no frozen replay is built",
                flush=True,
            )
        objective_fn = lambda p: _adaptive_rollout_objectives_realized_schedule_only_for_parameter(  # noqa: E731
            p,
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=parameter_name,
        )
        print(
            "[autodiff-gate] adaptive AD-only progress: running adaptive custom AD gradient",
            flush=True,
        )
        _, gradient_ad = jax.jvp(
            objective_fn,
            (jnp.asarray(baseline_value),),
            (jnp.asarray(1.0),),
        )
        grad_ad_np = np.asarray(jax.device_get(gradient_ad), dtype=float)
        print(
            "[autodiff-gate] adaptive AD-only progress: AD gradient complete",
            flush=True,
        )
        return {
            "config_path": str(config_path),
            "realized_schedule_frozen_fd_check": True,
            "parameter_name": parameter_name,
            "baseline_value": baseline_value,
            "fd_step": float(fd_step),
            "baseline_objectives": None,
            "gradient_autodiff": grad_ad_np.tolist(),
            "gradient_fd": None,
            "gradient_absolute_error": None,
            "gradient_relative_error": None,
            "max_relative_error": float("nan"),
            "passed": True,
            "objective_labels": OBJECTIVE_LABELS,
            "frozen_replay_mode": str(replay_mode),
            "ad_mode": "realized_schedule_jvp",
            "ad_available": True,
            "ad_only": True,
            "keep_adaptive_ad": True,
            "accepted_step_limit": None,
            "accepted_time_list": None,
            "rollout_path": {},
        }

    baseline_objectives, baseline_rollout = _adaptive_rollout_objectives_for_parameter(
        jnp.asarray(baseline_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        use_realized_schedule_jvp=True,
    )
    baseline_diag = _adaptive_rollout_diagnostics(baseline_rollout)
    replay_trace = _truncate_rollout_trace_by_accepted_steps(
        baseline_rollout.trace,
        accepted_step_limit,
    )
    replay_accepted_mask = np.asarray(jax.device_get(replay_trace.accepted_mask), dtype=bool)
    replay_active_mask = np.asarray(jax.device_get(replay_trace.active_mask), dtype=bool)
    accepted_times = np.asarray(jax.device_get(replay_trace.step_ts), dtype=float)
    accepted_mask = np.asarray(jax.device_get(replay_trace.accepted_mask), dtype=bool)
    active_mask = np.asarray(jax.device_get(replay_trace.active_mask), dtype=bool)
    accepted_time_list = accepted_times[np.logical_and(active_mask, accepted_mask)].tolist()
    print(
        "[autodiff-gate] realized-schedule frozen-fd baseline summary: "
        f"attempt_count={baseline_diag['attempt_count']} "
        f"accepted_count={baseline_diag['accepted_count']} "
        f"completed={baseline_diag['completed']} "
        f"failed={baseline_diag['failed']} "
        f"fail_code={baseline_diag['fail_code']}",
        flush=True,
    )
    if accepted_step_limit is None or keep_adaptive_ad:
        objective_fn = lambda p: _adaptive_rollout_objectives_for_parameter(  # noqa: E731
            p,
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=parameter_name,
            use_realized_schedule_jvp=True,
        )[0]
        ad_mode = "realized_schedule_jvp"
    else:
        objective_fn = lambda p: _adaptive_rollout_objectives_for_parameter_on_frozen_trace(  # noqa: E731
            p,
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=parameter_name,
            frozen_trace=replay_trace,
            replay_mode=replay_mode,
        )[0]
        ad_mode = "frozen_trace_direct"
    print(
        "[autodiff-gate] realized-schedule frozen-fd progress: baseline rollout complete; running AD gradient",
        flush=True,
    )
    gradient_ad = jax.jacfwd(objective_fn)(jnp.asarray(baseline_value))
    grad_ad_np = np.asarray(jax.device_get(gradient_ad), dtype=float)
    ad_available = True
    if ad_only:
        print(
            "[autodiff-gate] realized-schedule frozen-fd progress: AD gradient complete; skipping FD replays (--realized-schedule-frozen-ad-only)",
            flush=True,
        )
        return {
            "config_path": str(config_path),
            "realized_schedule_frozen_fd_check": True,
            "parameter_name": parameter_name,
            "baseline_value": baseline_value,
            "fd_step": float(fd_step),
            "baseline_objectives": np.asarray(jax.device_get(baseline_objectives), dtype=float).tolist(),
            "gradient_autodiff": grad_ad_np.tolist(),
            "gradient_fd": None,
            "gradient_absolute_error": None,
            "gradient_relative_error": None,
            "max_relative_error": float("nan"),
            "passed": True,
            "objective_labels": OBJECTIVE_LABELS,
            "frozen_replay_mode": str(replay_mode),
            "ad_mode": ad_mode,
            "ad_available": ad_available,
            "ad_only": True,
            "keep_adaptive_ad": bool(keep_adaptive_ad),
            "accepted_step_limit": None if accepted_step_limit is None else int(accepted_step_limit),
            "accepted_time_list": accepted_time_list,
            "rollout_path": {
                "baseline": baseline_diag,
            },
        }
    print(
        "[autodiff-gate] realized-schedule frozen-fd progress: baseline rollout complete; "
        f"running frozen fd_minus replay ({replay_mode})",
        flush=True,
    )
    objectives_minus, minus_replay = _adaptive_rollout_objectives_for_parameter_on_frozen_trace(
        jnp.asarray(minus_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        frozen_trace=replay_trace,
        replay_mode=replay_mode,
    )
    minus_objectives_np = np.asarray(jax.device_get(objectives_minus), dtype=float)
    minus_replay_finite = _tree_all_finite(minus_replay["final_state"]) and bool(np.all(np.isfinite(minus_objectives_np)))
    minus_nonfinite_debug = None
    minus_accepted_mode_debug = None
    print(
        "[autodiff-gate] realized-schedule frozen-fd fd_minus summary: "
        f"state_finite={_tree_all_finite(minus_replay['final_state'])} "
        f"objectives_finite={bool(np.all(np.isfinite(minus_objectives_np)))} "
        f"all_finite={minus_replay_finite}",
        flush=True,
    )
    print(
        "[autodiff-gate] realized-schedule frozen-fd progress: frozen fd_minus replay complete; "
        f"running frozen fd_plus replay ({replay_mode})",
        flush=True,
    )
    objectives_plus, plus_replay = _adaptive_rollout_objectives_for_parameter_on_frozen_trace(
        jnp.asarray(plus_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        frozen_trace=replay_trace,
        replay_mode=replay_mode,
    )
    plus_objectives_np = np.asarray(jax.device_get(objectives_plus), dtype=float)
    plus_replay_finite = _tree_all_finite(plus_replay["final_state"]) and bool(np.all(np.isfinite(plus_objectives_np)))
    plus_nonfinite_debug = None
    plus_accepted_mode_debug = None
    print(
        "[autodiff-gate] realized-schedule frozen-fd fd_plus summary: "
        f"state_finite={_tree_all_finite(plus_replay['final_state'])} "
        f"objectives_finite={bool(np.all(np.isfinite(plus_objectives_np)))} "
        f"all_finite={plus_replay_finite}",
        flush=True,
    )
    if not minus_replay_finite:
        minus_nonfinite_debug = _frozen_replay_nonfinite_debug(
            minus_replay,
            replay_trace,
            objectives_np=minus_objectives_np,
        )
        if replay_mode == "attempt":
            print(
                "[autodiff-gate] realized-schedule frozen-fd progress: "
                "fd_minus attempt replay failed; checking accepted-only replay on same frozen schedule",
                flush=True,
            )
            minus_objectives_accepted, minus_replay_accepted = _adaptive_rollout_objectives_for_parameter_on_frozen_trace(
                jnp.asarray(minus_value),
                config=config,
                runtime=runtime,
                baseline_state=baseline_state,
                profile_cfg=profile_cfg,
                parameter_name=parameter_name,
                frozen_trace=replay_trace,
                replay_mode="accepted",
            )
            minus_accepted_mode_debug = _frozen_replay_nonfinite_debug(
                minus_replay_accepted,
                replay_trace,
                objectives_np=np.asarray(jax.device_get(minus_objectives_accepted), dtype=float),
            )
    if not plus_replay_finite:
        plus_nonfinite_debug = _frozen_replay_nonfinite_debug(
            plus_replay,
            replay_trace,
            objectives_np=plus_objectives_np,
        )
        if replay_mode == "attempt":
            print(
                "[autodiff-gate] realized-schedule frozen-fd progress: "
                "fd_plus attempt replay failed; checking accepted-only replay on same frozen schedule",
                flush=True,
            )
            plus_objectives_accepted, plus_replay_accepted = _adaptive_rollout_objectives_for_parameter_on_frozen_trace(
                jnp.asarray(plus_value),
                config=config,
                runtime=runtime,
                baseline_state=baseline_state,
                profile_cfg=profile_cfg,
                parameter_name=parameter_name,
                frozen_trace=replay_trace,
                replay_mode="accepted",
            )
            plus_accepted_mode_debug = _frozen_replay_nonfinite_debug(
                plus_replay_accepted,
                replay_trace,
                objectives_np=np.asarray(jax.device_get(plus_objectives_accepted), dtype=float),
            )
    gradient_fd = (objectives_plus - objectives_minus) / (2.0 * fd_step)
    grad_fd_np = np.asarray(jax.device_get(gradient_fd), dtype=float)
    print(
        "[autodiff-gate] realized-schedule frozen-fd progress: frozen fd_plus replay complete; forming frozen FD gradient",
        flush=True,
    )
    abs_err = np.abs(grad_ad_np - grad_fd_np)
    rel_err = abs_err / np.maximum(np.abs(grad_fd_np), 1.0e-10)

    return {
        "config_path": str(config_path),
        "realized_schedule_frozen_fd_check": True,
        "parameter_name": parameter_name,
        "baseline_value": baseline_value,
        "fd_step": float(fd_step),
        "baseline_objectives": np.asarray(jax.device_get(baseline_objectives), dtype=float).tolist(),
        "gradient_autodiff": None if grad_ad_np is None else grad_ad_np.tolist(),
        "gradient_fd": grad_fd_np.tolist(),
        "gradient_absolute_error": None if abs_err is None else abs_err.tolist(),
        "gradient_relative_error": None if rel_err is None else rel_err.tolist(),
        "max_relative_error": float("nan") if rel_err is None else float(np.max(rel_err)),
        "passed": (
            bool(np.all(np.isfinite(rel_err)) and np.max(rel_err) <= 5.0e-2)
            if rel_err is not None
            else bool(minus_replay_finite and plus_replay_finite)
        ),
        "objective_labels": OBJECTIVE_LABELS,
        "frozen_replay_mode": str(replay_mode),
        "ad_mode": ad_mode,
        "ad_available": ad_available,
        "ad_only": False,
        "keep_adaptive_ad": bool(keep_adaptive_ad),
        "accepted_step_limit": None if accepted_step_limit is None else int(accepted_step_limit),
        "accepted_time_list": accepted_time_list,
        "rollout_path": {
            "baseline": baseline_diag,
            "frozen_fd_minus": {
                "replay_mode": minus_replay["replay_mode"],
                "accepted_count": int(np.sum(replay_accepted_mask)),
                "attempt_count": int(np.sum(replay_active_mask)),
                "state_finite": _tree_all_finite(minus_replay["final_state"]),
                "objectives_finite": bool(np.all(np.isfinite(minus_objectives_np))),
                "all_finite": minus_replay_finite,
                "nonfinite_debug": minus_nonfinite_debug,
                "accepted_mode_debug": minus_accepted_mode_debug,
            },
            "frozen_fd_plus": {
                "replay_mode": plus_replay["replay_mode"],
                "accepted_count": int(np.sum(replay_accepted_mask)),
                "attempt_count": int(np.sum(replay_active_mask)),
                "state_finite": _tree_all_finite(plus_replay["final_state"]),
                "objectives_finite": bool(np.all(np.isfinite(plus_objectives_np))),
                "all_finite": plus_replay_finite,
                "nonfinite_debug": plus_nonfinite_debug,
                "accepted_mode_debug": plus_accepted_mode_debug,
            },
        },
    }


def _frozen_replay_prefix_case(
    *,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    parameter_name: str,
    minus_value: float,
    plus_value: float,
    baseline_trace,
    replay_mode: str,
    accepted_step_limit: int,
) -> dict[str, Any]:
    replay_trace = _truncate_rollout_trace_by_accepted_steps(
        baseline_trace,
        accepted_step_limit,
    )
    accepted_mask = np.asarray(jax.device_get(replay_trace.accepted_mask), dtype=bool)
    active_mask = np.asarray(jax.device_get(replay_trace.active_mask), dtype=bool)
    accepted_times = np.asarray(jax.device_get(replay_trace.step_ts), dtype=float)
    accepted_time_list = accepted_times[np.logical_and(active_mask, accepted_mask)].tolist()

    objectives_minus, minus_replay = _adaptive_rollout_objectives_for_parameter_on_frozen_trace(
        jnp.asarray(minus_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        frozen_trace=replay_trace,
        replay_mode=replay_mode,
    )
    minus_objectives_np = np.asarray(jax.device_get(objectives_minus), dtype=float)
    minus_state_finite = _tree_all_finite(minus_replay["final_state"])
    minus_objectives_finite = bool(np.all(np.isfinite(minus_objectives_np)))
    minus_all_finite = bool(minus_state_finite and minus_objectives_finite)

    objectives_plus, plus_replay = _adaptive_rollout_objectives_for_parameter_on_frozen_trace(
        jnp.asarray(plus_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        frozen_trace=replay_trace,
        replay_mode=replay_mode,
    )
    plus_objectives_np = np.asarray(jax.device_get(objectives_plus), dtype=float)
    plus_state_finite = _tree_all_finite(plus_replay["final_state"])
    plus_objectives_finite = bool(np.all(np.isfinite(plus_objectives_np)))
    plus_all_finite = bool(plus_state_finite and plus_objectives_finite)

    return {
        "accepted_step_limit": int(accepted_step_limit),
        "replay_mode": str(replay_mode),
        "attempt_count": int(np.sum(active_mask)),
        "accepted_count": int(np.sum(accepted_mask)),
        "accepted_time_list": accepted_time_list,
        "fd_minus": {
            "state_finite": minus_state_finite,
            "objectives_finite": minus_objectives_finite,
            "all_finite": minus_all_finite,
        },
        "fd_plus": {
            "state_finite": plus_state_finite,
            "objectives_finite": plus_objectives_finite,
            "all_finite": plus_all_finite,
        },
        "all_finite": bool(minus_all_finite and plus_all_finite),
    }


def build_realized_schedule_frozen_replay_localize_report(
    *,
    config_path: Path,
    parameter_name: str,
    rel_fd_step: float,
    abs_fd_step: float,
    device: str | None,
) -> dict[str, Any]:
    if parameter_name not in ALLOWED_PARAMETERS:
        raise ValueError(f"parameter_name must be one of {sorted(ALLOWED_PARAMETERS)}")

    config = _prepare_benchmark_config(config_path, device=device)
    runtime, baseline_state = build_runtime_context(config)
    profile_cfg = _baseline_profile_cfg(config)
    baseline_value = float(profile_cfg[parameter_name])
    fd_step = _fd_step(baseline_value, rel_step=rel_fd_step, abs_step=abs_fd_step)
    minus_value = baseline_value - fd_step
    plus_value = baseline_value + fd_step

    baseline_objectives, baseline_rollout = _adaptive_rollout_objectives_for_parameter(
        jnp.asarray(baseline_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        use_realized_schedule_jvp=True,
    )
    baseline_diag = _adaptive_rollout_diagnostics(baseline_rollout)
    total_accepted = int(np.asarray(jax.device_get(baseline_rollout.accepted_count)))
    print(
        "[autodiff-gate] frozen-replay-localize baseline summary: "
        f"attempt_count={baseline_diag['attempt_count']} "
        f"accepted_count={baseline_diag['accepted_count']} "
        f"completed={baseline_diag['completed']} "
        f"failed={baseline_diag['failed']} "
        f"fail_code={baseline_diag['fail_code']}",
        flush=True,
    )

    attempt_mode_cache: dict[int, dict[str, Any]] = {}

    def _attempt_case(prefix: int) -> dict[str, Any]:
        prefix = int(prefix)
        if prefix not in attempt_mode_cache:
            print(
                "[autodiff-gate] frozen-replay-localize progress: "
                f"checking attempt replay prefix accepted_steps={prefix}",
                flush=True,
            )
            attempt_mode_cache[prefix] = _frozen_replay_prefix_case(
                config=config,
                runtime=runtime,
                baseline_state=baseline_state,
                profile_cfg=profile_cfg,
                parameter_name=parameter_name,
                minus_value=minus_value,
                plus_value=plus_value,
                baseline_trace=baseline_rollout.trace,
                replay_mode="attempt",
                accepted_step_limit=prefix,
            )
        return attempt_mode_cache[prefix]

    low = 1
    high = total_accepted
    first_failing_attempt_case = _attempt_case(high)
    if first_failing_attempt_case["all_finite"]:
        last_passing_attempt_case = first_failing_attempt_case
        first_failing_attempt_case = None
    else:
        last_passing_attempt_case = None
        while low <= high:
            mid = (low + high) // 2
            case_mid = _attempt_case(mid)
            if case_mid["all_finite"]:
                last_passing_attempt_case = case_mid
                low = mid + 1
            else:
                first_failing_attempt_case = case_mid
                high = mid - 1

    accepted_mode_at_boundary = None
    boundary_case = first_failing_attempt_case or last_passing_attempt_case
    if boundary_case is not None:
        boundary_prefix = int(boundary_case["accepted_step_limit"])
        print(
            "[autodiff-gate] frozen-replay-localize progress: "
            f"checking accepted replay at boundary accepted_steps={boundary_prefix}",
            flush=True,
        )
        accepted_mode_at_boundary = _frozen_replay_prefix_case(
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=parameter_name,
            minus_value=minus_value,
            plus_value=plus_value,
            baseline_trace=baseline_rollout.trace,
            replay_mode="accepted",
            accepted_step_limit=boundary_prefix,
        )

    return {
        "config_path": str(config_path),
        "realized_schedule_frozen_replay_localize": True,
        "parameter_name": parameter_name,
        "baseline_value": baseline_value,
        "fd_step": float(fd_step),
        "baseline_objectives": np.asarray(jax.device_get(baseline_objectives), dtype=float).tolist(),
        "objective_labels": OBJECTIVE_LABELS,
        "rollout_path": {
            "baseline": baseline_diag,
        },
        "attempt_prefix_checks": [attempt_mode_cache[key] for key in sorted(attempt_mode_cache)],
        "last_passing_attempt_case": last_passing_attempt_case,
        "first_failing_attempt_case": first_failing_attempt_case,
        "accepted_mode_at_boundary": accepted_mode_at_boundary,
        "passed": bool(first_failing_attempt_case is None),
        "max_relative_error": float("nan"),
    }


def build_realized_schedule_ad_debug_fast_report(
    *,
    config_path: Path,
    parameter_name: str,
    device: str | None,
    include_nan_debug: bool = True,
    nan_debug_mode: str = "minimal",
    nan_debug_include_one_step_compare: bool = False,
) -> dict[str, Any]:
    if parameter_name not in ALLOWED_PARAMETERS:
        raise ValueError(f"parameter_name must be one of {sorted(ALLOWED_PARAMETERS)}")

    config = _prepare_benchmark_config(config_path, device=device)
    runtime, baseline_state = build_runtime_context(config)
    profile_cfg = _baseline_profile_cfg(config)
    baseline_value = float(profile_cfg[parameter_name])

    objective_fn = lambda p: _adaptive_rollout_objectives_for_parameter(  # noqa: E731
        p,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        use_realized_schedule_jvp=True,
    )[0]

    baseline_objectives, baseline_rollout = _adaptive_rollout_objectives_for_parameter(
        jnp.asarray(baseline_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        use_realized_schedule_jvp=True,
    )
    baseline_diag = _adaptive_rollout_summary(baseline_rollout)
    print(
        "[autodiff-gate] fast-ad-debug baseline summary: "
        f"attempt_count={baseline_diag['attempt_count']} "
        f"accepted_count={baseline_diag['accepted_count']} "
        f"completed={baseline_diag['completed']} "
        f"failed={baseline_diag['failed']} "
        f"fail_code={baseline_diag['fail_code']}",
        flush=True,
    )
    print("[autodiff-gate] fast-ad-debug progress: baseline rollout complete; running AD gradient", flush=True)
    gradient_ad = jax.jacfwd(objective_fn)(jnp.asarray(baseline_value))
    print("[autodiff-gate] fast-ad-debug progress: AD gradient complete", flush=True)

    grad_ad_np = np.asarray(jax.device_get(gradient_ad), dtype=float)
    print("[autodiff-gate] fast-ad-debug AD values:", flush=True)
    for label, ad in zip(OBJECTIVE_LABELS, grad_ad_np):
        finite_flag = bool(np.isfinite(ad))
        print(
            f"  - {label}: ad={float(ad):.6e} finite={finite_flag}",
            flush=True,
        )
    nan_debug = None
    if include_nan_debug and not np.all(np.isfinite(grad_ad_np)):
        print("[autodiff-gate] fast-ad-debug progress: AD produced nonfinite values; running NaN localization", flush=True)
        nan_debug = _adaptive_rollout_nan_debug_for_parameter(
            jnp.asarray(baseline_value),
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=parameter_name,
            debug_mode=nan_debug_mode,
            include_one_step_compare=nan_debug_include_one_step_compare,
        )
        print(
            "[autodiff-gate] fast-ad-debug NaN localization result: "
            f"first_bad_index={nan_debug.get('first_bad_index')} "
            f"first_bad_was_accepted={nan_debug.get('first_bad_was_accepted')} "
            f"first_bad_dt={_fmt_float(nan_debug.get('first_bad_dt'))} "
            f"final_tangent_finite={nan_debug.get('final_tangent_finite')}",
            flush=True,
        )
        print("[autodiff-gate] fast-ad-debug progress: NaN localization complete", flush=True)

    return {
        "config_path": str(config_path),
        "realized_schedule_ad_debug_fast": True,
        "parameter_name": parameter_name,
        "baseline_value": baseline_value,
        "baseline_objectives": np.asarray(jax.device_get(baseline_objectives), dtype=float).tolist(),
        "gradient_autodiff": grad_ad_np.tolist(),
        "ad_all_finite": bool(np.all(np.isfinite(grad_ad_np))),
        "passed": bool(np.all(np.isfinite(grad_ad_np))),
        "max_relative_error": float("nan"),
        "objective_labels": OBJECTIVE_LABELS,
        "rollout_path": {
            "baseline": baseline_diag,
        },
        "nan_debug": nan_debug,
    }


def _config_with_t_final(config: dict[str, Any], t_final: float) -> dict[str, Any]:
    tuned = copy.deepcopy(config)
    solver_cfg = tuned.setdefault("transport_solver", {})
    solver_cfg["t_final"] = float(t_final)
    return tuned


def build_baseline_dt_path_safe_fd_report(
    *,
    config_path: Path,
    parameter_name: str,
    rel_fd_step: float,
    abs_fd_step: float,
    device: str | None,
    known_first_bad_attempt_index: int,
    safe_attempt_margin: int,
) -> dict[str, Any]:
    if parameter_name not in ALLOWED_PARAMETERS:
        raise ValueError(f"parameter_name must be one of {sorted(ALLOWED_PARAMETERS)}")

    config = _prepare_benchmark_config(config_path, device=device)
    runtime, baseline_state = build_runtime_context(config)
    profile_cfg = _baseline_profile_cfg(config)
    baseline_value = float(profile_cfg[parameter_name])
    fd_step = _fd_step(baseline_value, rel_step=rel_fd_step, abs_step=abs_fd_step)
    minus_value = baseline_value - fd_step
    plus_value = baseline_value + fd_step

    baseline_objectives_full, baseline_rollout = _adaptive_rollout_objectives_for_parameter(
        jnp.asarray(baseline_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        use_realized_schedule_jvp=True,
    )
    baseline_diag = _adaptive_rollout_diagnostics(baseline_rollout)
    safe_attempt_index = int(known_first_bad_attempt_index) - int(safe_attempt_margin)
    safe_time_list = _accepted_time_list_until_attempt_index(
        baseline_rollout.trace,
        safe_attempt_index,
    )
    if not safe_time_list:
        raise ValueError("Safe baseline dt path is empty; adjust known_first_bad_attempt_index or safe_attempt_margin.")
    safe_final_time = float(safe_time_list[-1])
    config_safe = _config_with_t_final(config, safe_final_time)
    runtime_safe, baseline_state_safe = build_runtime_context(config_safe)
    profile_cfg_safe = _baseline_profile_cfg(config_safe)
    print(
        "[autodiff-gate] baseline-dt-safe-fd baseline summary: "
        f"attempt_count={baseline_diag['attempt_count']} "
        f"accepted_count={baseline_diag['accepted_count']} "
        f"known_first_bad_attempt_index={known_first_bad_attempt_index} "
        f"safe_attempt_index={safe_attempt_index} "
        f"safe_accepted_count={len(safe_time_list)} "
        f"safe_final_time={safe_final_time:.6e}",
        flush=True,
    )

    adaptive_objective_fn = lambda p: _adaptive_rollout_objectives_for_parameter(  # noqa: E731
        p,
        config=config_safe,
        runtime=runtime_safe,
        baseline_state=baseline_state_safe,
        profile_cfg=profile_cfg_safe,
        parameter_name=parameter_name,
        use_realized_schedule_jvp=True,
    )[0]
    fixed_dt_objective_fn = lambda p: _adaptive_rollout_objectives_for_parameter_on_time_list(  # noqa: E731
        p,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        time_list=safe_time_list,
    )[0]

    baseline_objectives, baseline_replay = _adaptive_rollout_objectives_for_parameter_on_time_list(
        jnp.asarray(baseline_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        time_list=safe_time_list,
    )
    baseline_fixed_dt_state_finite = _tree_all_finite(baseline_replay["final_state"])
    print(
        "[autodiff-gate] baseline-dt-safe-fd fixed replay summary: "
        f"baseline_state_finite={baseline_fixed_dt_state_finite}",
        flush=True,
    )
    print("[autodiff-gate] baseline-dt-safe-fd progress: baseline fixed-dt replay complete; running fd_minus replay", flush=True)
    objectives_minus, minus_replay = _adaptive_rollout_objectives_for_parameter_on_time_list(
        jnp.asarray(minus_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        time_list=safe_time_list,
    )
    minus_fixed_dt_state_finite = _tree_all_finite(minus_replay["final_state"])
    print(
        "[autodiff-gate] baseline-dt-safe-fd fd_minus summary: "
        f"state_finite={minus_fixed_dt_state_finite}",
        flush=True,
    )
    print("[autodiff-gate] baseline-dt-safe-fd progress: fd_minus replay complete; running fd_plus replay", flush=True)
    objectives_plus, plus_replay = _adaptive_rollout_objectives_for_parameter_on_time_list(
        jnp.asarray(plus_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        time_list=safe_time_list,
    )
    plus_fixed_dt_state_finite = _tree_all_finite(plus_replay["final_state"])
    print(
        "[autodiff-gate] baseline-dt-safe-fd fd_plus summary: "
        f"state_finite={plus_fixed_dt_state_finite}",
        flush=True,
    )
    print(
        "[autodiff-gate] baseline-dt-safe-fd progress: "
        f"fd_plus replay complete; running realized-schedule AD to safe_final_time={safe_final_time:.6e}",
        flush=True,
    )
    gradient_ad = jax.jacfwd(adaptive_objective_fn)(jnp.asarray(baseline_value))
    print("[autodiff-gate] baseline-dt-safe-fd progress: AD gradient complete; forming FD gradient", flush=True)
    gradient_fd = (objectives_plus - objectives_minus) / (2.0 * fd_step)

    grad_ad_np = np.asarray(jax.device_get(gradient_ad), dtype=float)
    grad_fd_np = np.asarray(jax.device_get(gradient_fd), dtype=float)
    abs_err = np.abs(grad_ad_np - grad_fd_np)
    rel_err = abs_err / np.maximum(np.abs(grad_fd_np), 1.0e-10)

    return {
        "config_path": str(config_path),
        "baseline_dt_path_safe_fd_check": True,
        "parameter_name": parameter_name,
        "baseline_value": baseline_value,
        "fd_step": float(fd_step),
        "known_first_bad_attempt_index": int(known_first_bad_attempt_index),
        "safe_attempt_margin": int(safe_attempt_margin),
        "safe_attempt_index": int(safe_attempt_index),
        "safe_final_time": float(safe_final_time),
        "safe_time_list": safe_time_list,
        "baseline_objectives": np.asarray(jax.device_get(baseline_objectives), dtype=float).tolist(),
        "ad_mode": "realized_schedule_jvp",
        "gradient_autodiff": grad_ad_np.tolist(),
        "gradient_fd": grad_fd_np.tolist(),
        "gradient_absolute_error": abs_err.tolist(),
        "gradient_relative_error": rel_err.tolist(),
        "max_relative_error": float(np.max(rel_err)),
        "passed": bool(np.all(np.isfinite(rel_err)) and np.max(rel_err) <= 5.0e-2),
        "objective_labels": OBJECTIVE_LABELS,
        "rollout_path": {
            "baseline": baseline_diag,
            "baseline_fixed_dt_state_finite": baseline_fixed_dt_state_finite,
            "fd_minus_fixed_dt_state_finite": minus_fixed_dt_state_finite,
            "fd_plus_fixed_dt_state_finite": plus_fixed_dt_state_finite,
        },
    }


def build_baseline_dt_path_safe_compose_report(
    *,
    config_path: Path,
    parameter_name: str,
    rel_fd_step: float,
    abs_fd_step: float,
    device: str | None,
    known_first_bad_attempt_index: int,
    safe_attempt_margin: int,
) -> dict[str, Any]:
    fd_report = build_baseline_dt_path_safe_fd_report(
        config_path=config_path,
        parameter_name=parameter_name,
        rel_fd_step=rel_fd_step,
        abs_fd_step=abs_fd_step,
        device=device,
        known_first_bad_attempt_index=known_first_bad_attempt_index,
        safe_attempt_margin=safe_attempt_margin,
    )

    config = _prepare_benchmark_config(config_path, device=device)
    safe_final_time = float(fd_report["safe_final_time"])
    config_safe = _config_with_t_final(config, safe_final_time)
    runtime, baseline_state = build_runtime_context(config_safe)
    profile_cfg = _baseline_profile_cfg(config_safe)
    baseline_value = float(profile_cfg[parameter_name])
    safe_time_list = list(fd_report["safe_time_list"])

    adaptive_objective_fn = lambda p: _adaptive_rollout_objectives_for_parameter(  # noqa: E731
        p,
        config=config_safe,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        use_realized_schedule_jvp=True,
    )[0]
    fixed_dt_objective_fn = lambda p: _adaptive_rollout_objectives_for_parameter_on_time_list(  # noqa: E731
        p,
        config=config,
        runtime=_runtime,
        baseline_state=_baseline_state,
        profile_cfg=_profile_cfg,
        parameter_name=parameter_name,
        time_list=safe_time_list,
    )[0]
    _runtime, _baseline_state = build_runtime_context(config)
    _profile_cfg = _baseline_profile_cfg(config)
    print(
        "[autodiff-gate] baseline-dt-safe-compose progress: "
        f"running realized-schedule AD and fixed-dt direct AD to safe_final_time={safe_final_time:.6e}",
        flush=True,
    )
    gradient_adaptive = jax.jacfwd(adaptive_objective_fn)(jnp.asarray(baseline_value))
    print("[autodiff-gate] baseline-dt-safe-compose progress: realized-schedule AD complete; running fixed-dt direct AD", flush=True)
    gradient_fixed_dt_direct = jax.jacfwd(fixed_dt_objective_fn)(jnp.asarray(baseline_value))
    grad_adaptive_np = np.asarray(jax.device_get(gradient_adaptive), dtype=float)
    grad_direct_np = np.asarray(jax.device_get(gradient_fixed_dt_direct), dtype=float)
    abs_err = np.abs(grad_adaptive_np - grad_direct_np)
    rel_err = abs_err / np.maximum(np.abs(grad_direct_np), 1.0e-10)

    return {
        "config_path": str(config_path),
        "baseline_dt_path_safe_compose_check": True,
        "parameter_name": parameter_name,
        "baseline_value": baseline_value,
        "safe_final_time": float(safe_final_time),
        "safe_attempt_index": int(fd_report["safe_attempt_index"]),
        "safe_time_list": fd_report["safe_time_list"],
        "gradient_realized_schedule_autodiff": grad_adaptive_np.tolist(),
        "gradient_fixed_dt_direct_autodiff": grad_direct_np.tolist(),
        "gradient_absolute_error": abs_err.tolist(),
        "gradient_relative_error": rel_err.tolist(),
        "max_relative_error": float(np.max(rel_err)),
        "passed": bool(np.all(np.isfinite(rel_err)) and np.max(rel_err) <= 5.0e-2),
        "objective_labels": OBJECTIVE_LABELS,
        "baseline_dt_safe_fd_report": fd_report,
    }


def build_baseline_dt_path_safe_compose_scan_report(
    *,
    config_path: Path,
    parameter_name: str,
    device: str | None,
    known_first_bad_attempt_index: int,
    safe_attempt_margin: int,
    accepted_step_counts: tuple[int, ...],
) -> dict[str, Any]:
    if parameter_name not in ALLOWED_PARAMETERS:
        raise ValueError(f"parameter_name must be one of {sorted(ALLOWED_PARAMETERS)}")

    config = _prepare_benchmark_config(config_path, device=device)
    runtime, baseline_state = build_runtime_context(config)
    profile_cfg = _baseline_profile_cfg(config)
    baseline_value = float(profile_cfg[parameter_name])

    _, baseline_rollout = _adaptive_rollout_final_state_for_parameter(
        jnp.asarray(baseline_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        use_realized_schedule_jvp=False,
        accepted_step_limit_override=max_checkpoint,
    )
    baseline_diag = _adaptive_rollout_diagnostics(baseline_rollout)
    safe_attempt_index = int(known_first_bad_attempt_index) - int(safe_attempt_margin)
    safe_time_list_full = _accepted_time_list_until_attempt_index(
        baseline_rollout.trace,
        safe_attempt_index,
    )
    if not safe_time_list_full:
        raise ValueError("Safe baseline dt path is empty; adjust known_first_bad_attempt_index or safe_attempt_margin.")

    safe_accepted_count = len(safe_time_list_full)
    requested_counts = [int(v) for v in accepted_step_counts if int(v) > 0]
    if not requested_counts:
        requested_counts = [1, 2, 5, 10, safe_accepted_count]
    prefix_counts = sorted({min(v, safe_accepted_count) for v in requested_counts})
    print(
        "[autodiff-gate] baseline-dt-safe-compose-scan baseline summary: "
        f"attempt_count={baseline_diag['attempt_count']} "
        f"accepted_count={baseline_diag['accepted_count']} "
        f"known_first_bad_attempt_index={known_first_bad_attempt_index} "
        f"safe_attempt_index={safe_attempt_index} "
        f"safe_accepted_count={safe_accepted_count}",
        flush=True,
    )

    entries: list[dict[str, Any]] = []
    global_max_rel_error = 0.0
    for accepted_count in prefix_counts:
        prefix_time_list = safe_time_list_full[:accepted_count]
        prefix_final_time = float(prefix_time_list[-1])
        print(
            "[autodiff-gate] baseline-dt-safe-compose-scan progress: "
            f"accepted_count={accepted_count} "
            f"final_time={prefix_final_time:.6e} "
            "running adaptive AD",
            flush=True,
        )
        config_prefix = _config_with_t_final(config, prefix_final_time)
        runtime_prefix, baseline_state_prefix = build_runtime_context(config_prefix)
        profile_cfg_prefix = _baseline_profile_cfg(config_prefix)

        adaptive_objective_fn = lambda p: _adaptive_rollout_objectives_for_parameter(  # noqa: E731
            p,
            config=config_prefix,
            runtime=runtime_prefix,
            baseline_state=baseline_state_prefix,
            profile_cfg=profile_cfg_prefix,
            parameter_name=parameter_name,
            use_realized_schedule_jvp=True,
        )[0]
        fixed_dt_objective_fn = lambda p: _adaptive_rollout_objectives_for_parameter_on_time_list(  # noqa: E731
            p,
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=parameter_name,
            time_list=prefix_time_list,
        )[0]

        gradient_adaptive = jax.jacfwd(adaptive_objective_fn)(jnp.asarray(baseline_value))
        print(
            "[autodiff-gate] baseline-dt-safe-compose-scan progress: "
            f"accepted_count={accepted_count} running fixed-dt direct AD",
            flush=True,
        )
        gradient_fixed_dt_direct = jax.jacfwd(fixed_dt_objective_fn)(jnp.asarray(baseline_value))

        grad_adaptive_np = np.asarray(jax.device_get(gradient_adaptive), dtype=float)
        grad_direct_np = np.asarray(jax.device_get(gradient_fixed_dt_direct), dtype=float)
        abs_err = np.abs(grad_adaptive_np - grad_direct_np)
        rel_err = abs_err / np.maximum(np.abs(grad_direct_np), 1.0e-10)
        max_rel_err = float(np.max(rel_err))
        global_max_rel_error = max(global_max_rel_error, max_rel_err)

        label_to_index = {label: idx for idx, label in enumerate(OBJECTIVE_LABELS)}
        er_idx = label_to_index["Er_volume_average"]
        er2_idx = label_to_index["Er2_volume_average"]
        pressure_idx = label_to_index["total_pressure_volume_average"]
        print(
            "[autodiff-gate] baseline-dt-safe-compose-scan summary: "
            f"accepted_count={accepted_count} "
            f"max_rel_err={max_rel_err:.6e} "
            f"Er_rel_err={float(rel_err[er_idx]):.6e} "
            f"Er2_rel_err={float(rel_err[er2_idx]):.6e} "
            f"pressure_rel_err={float(rel_err[pressure_idx]):.6e}",
            flush=True,
        )

        entries.append(
            {
                "accepted_count": int(accepted_count),
                "final_time": float(prefix_final_time),
                "gradient_realized_schedule_autodiff": grad_adaptive_np.tolist(),
                "gradient_fixed_dt_direct_autodiff": grad_direct_np.tolist(),
                "gradient_absolute_error": abs_err.tolist(),
                "gradient_relative_error": rel_err.tolist(),
                "max_relative_error": max_rel_err,
            }
        )

    return {
        "config_path": str(config_path),
        "baseline_dt_path_safe_compose_scan_check": True,
        "parameter_name": parameter_name,
        "baseline_value": baseline_value,
        "known_first_bad_attempt_index": int(known_first_bad_attempt_index),
        "safe_attempt_margin": int(safe_attempt_margin),
        "safe_attempt_index": int(safe_attempt_index),
        "safe_accepted_count": int(safe_accepted_count),
        "entries": entries,
        "objective_labels": OBJECTIVE_LABELS,
        "max_relative_error": float(global_max_rel_error),
        "passed": bool(np.isfinite(global_max_rel_error) and global_max_rel_error <= 5.0e-2),
        "rollout_path": {"baseline": baseline_diag},
    }


def build_baseline_dt_path_safe_trajectory_compare_report(
    *,
    config_path: Path,
    parameter_name: str,
    device: str | None,
    known_first_bad_attempt_index: int,
    safe_attempt_margin: int,
    sample_every: int,
) -> dict[str, Any]:
    if parameter_name not in ALLOWED_PARAMETERS:
        raise ValueError(f"parameter_name must be one of {sorted(ALLOWED_PARAMETERS)}")

    config = _prepare_benchmark_config(config_path, device=device)
    runtime, baseline_state = build_runtime_context(config)
    profile_cfg = _baseline_profile_cfg(config)
    baseline_value = float(profile_cfg[parameter_name])

    _, baseline_rollout = _adaptive_rollout_final_state_for_parameter(
        jnp.asarray(baseline_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        use_realized_schedule_jvp=False,
        accepted_step_limit_override=checkpoint_index,
    )
    baseline_diag = _adaptive_rollout_diagnostics(baseline_rollout)
    safe_attempt_index = int(known_first_bad_attempt_index) - int(safe_attempt_margin)
    safe_trace = _truncate_rollout_trace_by_accepted_steps(
        baseline_rollout.trace,
        accepted_step_limit=len(_accepted_time_list_until_attempt_index(baseline_rollout.trace, safe_attempt_index)),
    )
    safe_time_list = _accepted_time_list_until_attempt_index(baseline_rollout.trace, safe_attempt_index)
    safe_accepted_count = len(safe_time_list)
    if safe_accepted_count <= 0:
        raise ValueError("Safe baseline dt path is empty; adjust known_first_bad_attempt_index or safe_attempt_margin.")

    print(
        "[autodiff-gate] baseline-dt-safe-trajectory baseline summary: "
        f"attempt_count={baseline_diag['attempt_count']} "
        f"accepted_count={baseline_diag['accepted_count']} "
        f"known_first_bad_attempt_index={known_first_bad_attempt_index} "
        f"safe_attempt_index={safe_attempt_index} "
        f"safe_accepted_count={safe_accepted_count} "
        f"safe_final_time={float(safe_time_list[-1]):.6e}",
        flush=True,
    )

    state0_static = _parameterized_initial_state(
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        geometry=runtime.geometry,
        n_species=runtime.species.number_species,
        parameter_name=parameter_name,
        parameter_value=jax.lax.stop_gradient(jnp.asarray(baseline_value)),
    )
    prepared_components_static = prepare_transport_solver_components(config, runtime, state0_static)
    solver_static = prepared_components_static["solver"]
    solve_vector_field_static = prepared_components_static["solve_vector_field"]
    prepared_rollout_static = _build_prepared_radau_accepted_rollout(
        solver=solver_static,
        state=state0_static,
        vector_field=solve_vector_field_static,
        species=runtime.species,
    )
    execution_context = _build_prepared_radau_execution_context(
        solver=solver_static,
        prepared_rollout=prepared_rollout_static,
    )

    print("[autodiff-gate] baseline-dt-safe-trajectory progress: building adaptive initial carry tangent", flush=True)
    carry0, carry0_dot = jax.jvp(
        lambda p: _adaptive_rollout_initial_carry_for_parameter(
            p,
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=parameter_name,
        ),
        (jnp.asarray(baseline_value),),
        (jnp.asarray(1.0),),
    )

    fixed_dt_fn = lambda p: _adaptive_rollout_objective_trajectory_on_time_list(  # noqa: E731
        p,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        time_list=safe_time_list,
    )[0]

    print("[autodiff-gate] baseline-dt-safe-trajectory progress: running realized-trace adaptive AD trajectory", flush=True)
    adaptive_result = _sampled_adaptive_objective_tangent_trajectory(
        execution_context=execution_context,
        carry0=carry0,
        carry0_dot=carry0_dot,
        trace=safe_trace,
        runtime=runtime,
        unpack_flat=prepared_rollout_static.physics_context.unpack_flat,
        sample_every=sample_every,
    )
    print("[autodiff-gate] baseline-dt-safe-trajectory progress: adaptive AD trajectory complete; running fixed-dt direct AD trajectory", flush=True)
    _, fixed_dt_tangent = jax.jvp(fixed_dt_fn, (jnp.asarray(baseline_value),), (jnp.asarray(1.0),))

    sample_indices = adaptive_result["sampled_indices"]
    adaptive_np = np.asarray(jax.device_get(adaptive_result["sampled_tangents"]), dtype=float)
    fixed_full_np = np.asarray(jax.device_get(fixed_dt_tangent), dtype=float)
    fixed_np = fixed_full_np[np.asarray(sample_indices, dtype=int), :]
    sampled_times_np = np.asarray(jax.device_get(adaptive_result["sampled_times"]), dtype=float)
    abs_err = np.abs(adaptive_np - fixed_np)
    rel_err = abs_err / np.maximum(np.abs(fixed_np), 1.0e-10)

    entries = []
    global_max_rel_error = 0.0
    label_to_index = {label: idx for idx, label in enumerate(OBJECTIVE_LABELS)}
    for idx, sample_idx in enumerate(sample_indices):
        step_rel = rel_err[idx]
        step_max = float(np.max(step_rel))
        global_max_rel_error = max(global_max_rel_error, step_max)
        entries.append(
            {
                "accepted_index": int(sample_idx + 1),
                "time": float(sampled_times_np[idx]),
                "gradient_realized_trace_ad": adaptive_np[idx].tolist(),
                "gradient_fixed_dt_direct_ad": fixed_np[idx].tolist(),
                "gradient_absolute_error": abs_err[idx].tolist(),
                "gradient_relative_error": step_rel.tolist(),
                "max_relative_error": step_max,
                "Er_relative_error": float(step_rel[label_to_index["Er_volume_average"]]),
                "Er2_relative_error": float(step_rel[label_to_index["Er2_volume_average"]]),
                "pressure_relative_error": float(step_rel[label_to_index["total_pressure_volume_average"]]),
            }
        )

    return {
        "config_path": str(config_path),
        "baseline_dt_path_safe_trajectory_compare_check": True,
        "parameter_name": parameter_name,
        "baseline_value": baseline_value,
        "known_first_bad_attempt_index": int(known_first_bad_attempt_index),
        "safe_attempt_margin": int(safe_attempt_margin),
        "safe_attempt_index": int(safe_attempt_index),
        "safe_accepted_count": int(safe_accepted_count),
        "safe_final_time": float(safe_time_list[-1]),
        "sample_every": int(max(1, sample_every)),
        "objective_labels": OBJECTIVE_LABELS,
        "entries": entries,
        "max_relative_error": float(global_max_rel_error),
        "passed": bool(np.isfinite(global_max_rel_error) and global_max_rel_error <= 5.0e-2),
        "rollout_path": {"baseline": baseline_diag},
    }


def build_baseline_dt_path_safe_state_trajectory_compare_report(
    *,
    config_path: Path,
    parameter_name: str,
    device: str | None,
    known_first_bad_attempt_index: int,
    safe_attempt_margin: int,
    sample_every: int,
) -> dict[str, Any]:
    if parameter_name not in ALLOWED_PARAMETERS:
        raise ValueError(f"parameter_name must be one of {sorted(ALLOWED_PARAMETERS)}")

    config = _prepare_benchmark_config(config_path, device=device)
    runtime, baseline_state = build_runtime_context(config)
    profile_cfg = _baseline_profile_cfg(config)
    baseline_value = float(profile_cfg[parameter_name])

    _, baseline_rollout = _adaptive_rollout_final_state_for_parameter(
        jnp.asarray(baseline_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        use_realized_schedule_jvp=False,
        accepted_step_limit_override=checkpoint_index,
    )
    baseline_diag = _adaptive_rollout_diagnostics(baseline_rollout)
    safe_attempt_index = int(known_first_bad_attempt_index) - int(safe_attempt_margin)
    safe_time_list = _accepted_time_list_until_attempt_index(baseline_rollout.trace, safe_attempt_index)
    safe_accepted_count = len(safe_time_list)
    if safe_accepted_count <= 0:
        raise ValueError("Safe baseline dt path is empty; adjust known_first_bad_attempt_index or safe_attempt_margin.")
    safe_trace = _truncate_rollout_trace_by_accepted_steps(
        baseline_rollout.trace,
        accepted_step_limit=safe_accepted_count,
    )

    print(
        "[autodiff-gate] baseline-dt-safe-state-trajectory baseline summary: "
        f"attempt_count={baseline_diag['attempt_count']} "
        f"accepted_count={baseline_diag['accepted_count']} "
        f"known_first_bad_attempt_index={known_first_bad_attempt_index} "
        f"safe_attempt_index={safe_attempt_index} "
        f"safe_accepted_count={safe_accepted_count} "
        f"sample_every={int(max(1, sample_every))} "
        f"safe_final_time={float(safe_time_list[-1]):.6e}",
        flush=True,
    )

    state0_static = _parameterized_initial_state(
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        geometry=runtime.geometry,
        n_species=runtime.species.number_species,
        parameter_name=parameter_name,
        parameter_value=jax.lax.stop_gradient(jnp.asarray(baseline_value)),
    )
    prepared_components_static = prepare_transport_solver_components(config, runtime, state0_static)
    solver_static = prepared_components_static["solver"]
    solve_vector_field_static = prepared_components_static["solve_vector_field"]
    prepared_rollout_static = _build_prepared_radau_accepted_rollout(
        solver=solver_static,
        state=state0_static,
        vector_field=solve_vector_field_static,
        species=runtime.species,
    )
    execution_context = _build_prepared_radau_execution_context(
        solver=solver_static,
        prepared_rollout=prepared_rollout_static,
    )

    carry0, carry0_dot = jax.jvp(
        lambda p: _adaptive_rollout_initial_carry_for_parameter(
            p,
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=parameter_name,
        ),
        (jnp.asarray(baseline_value),),
        (jnp.asarray(1.0),),
    )

    print("[autodiff-gate] baseline-dt-safe-state-trajectory progress: running realized-trace adaptive state tangent trajectory", flush=True)
    adaptive_result = _sampled_adaptive_state_tangent_trajectory(
        execution_context=execution_context,
        carry0=carry0,
        carry0_dot=carry0_dot,
        trace=safe_trace,
        sample_every=sample_every,
    )
    print("[autodiff-gate] baseline-dt-safe-state-trajectory progress: adaptive trajectory complete; running fixed-dt direct state tangent trajectory", flush=True)
    fixed_direct_result = _sampled_fixed_dt_state_tangent_trajectory(
        execution_context=execution_context,
        carry0=carry0,
        carry0_dot=carry0_dot,
        dt_sequence=_radau_dt_sequence_from_time_list(
            safe_time_list,
            t0=prepared_rollout_static.initial_carry.t,
            dtype=prepared_rollout_static.kernel_context.dtype,
        ),
        sample_every=sample_every,
    )

    sample_indices = adaptive_result["sampled_indices"]
    adaptive_np = np.asarray(jax.device_get(adaptive_result["sampled_state_tangents"]), dtype=float)
    fixed_np = np.asarray(jax.device_get(fixed_direct_result["sampled_state_tangents"]), dtype=float)
    sampled_times_np = np.asarray(jax.device_get(adaptive_result["sampled_times"]), dtype=float)
    unpack_flat = prepared_rollout_static.physics_context.unpack_flat

    def _rel_norm(a: np.ndarray, b: np.ndarray) -> float:
        num = float(np.linalg.norm(a - b))
        den = max(float(np.linalg.norm(b)), 1.0e-10)
        return num / den

    entries = []
    global_max_rel_error = 0.0
    for idx, sample_idx in enumerate(sample_indices):
        ad_step = adaptive_np[idx]
        direct_step = fixed_np[idx]
        ad_state = unpack_flat(jnp.asarray(ad_step))
        direct_state = unpack_flat(jnp.asarray(direct_step))
        ad_density = np.asarray(jax.device_get(ad_state.density), dtype=float)
        direct_density = np.asarray(jax.device_get(direct_state.density), dtype=float)
        ad_pressure = np.asarray(jax.device_get(ad_state.pressure), dtype=float)
        direct_pressure = np.asarray(jax.device_get(direct_state.pressure), dtype=float)
        ad_er = np.asarray(jax.device_get(ad_state.Er), dtype=float)
        direct_er = np.asarray(jax.device_get(direct_state.Er), dtype=float)
        full_rel = _rel_norm(ad_step, direct_step)
        density_rel = _rel_norm(ad_density, direct_density)
        pressure_rel = _rel_norm(ad_pressure, direct_pressure)
        er_rel = _rel_norm(ad_er, direct_er)
        global_max_rel_error = max(global_max_rel_error, full_rel, density_rel, pressure_rel, er_rel)
        entries.append(
            {
                "accepted_index": int(sample_idx + 1),
                "time": float(sampled_times_np[idx]),
                "full_state_relative_error": float(full_rel),
                "density_relative_error": float(density_rel),
                "pressure_relative_error": float(pressure_rel),
                "Er_relative_error": float(er_rel),
            }
        )

    return {
        "config_path": str(config_path),
        "baseline_dt_path_safe_state_trajectory_compare_check": True,
        "parameter_name": parameter_name,
        "baseline_value": baseline_value,
        "known_first_bad_attempt_index": int(known_first_bad_attempt_index),
        "safe_attempt_margin": int(safe_attempt_margin),
        "safe_attempt_index": int(safe_attempt_index),
        "safe_accepted_count": int(safe_accepted_count),
        "safe_final_time": float(safe_time_list[-1]),
        "sample_every": int(max(1, sample_every)),
        "entries": entries,
        "max_relative_error": float(global_max_rel_error),
        "passed": bool(np.isfinite(global_max_rel_error) and global_max_rel_error <= 5.0e-2),
        "rollout_path": {"baseline": baseline_diag},
    }


def build_realized_trace_safe_state_trajectory_compare_report(
    *,
    config_path: Path,
    parameter_name: str,
    device: str | None,
    known_first_bad_attempt_index: int,
    safe_attempt_margin: int,
    sample_every: int,
) -> dict[str, Any]:
    if parameter_name not in ALLOWED_PARAMETERS:
        raise ValueError(f"parameter_name must be one of {sorted(ALLOWED_PARAMETERS)}")

    config = _prepare_benchmark_config(config_path, device=device)
    runtime, baseline_state = build_runtime_context(config)
    profile_cfg = _baseline_profile_cfg(config)
    baseline_value = float(profile_cfg[parameter_name])

    _, baseline_rollout = _adaptive_rollout_objectives_for_parameter(
        jnp.asarray(baseline_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        use_realized_schedule_jvp=True,
    )
    baseline_diag = _adaptive_rollout_diagnostics(baseline_rollout)
    safe_attempt_index = int(known_first_bad_attempt_index) - int(safe_attempt_margin)
    safe_time_list = _accepted_time_list_until_attempt_index(baseline_rollout.trace, safe_attempt_index)
    safe_accepted_count = len(safe_time_list)
    if safe_accepted_count <= 0:
        raise ValueError("Safe realized trace path is empty; adjust known_first_bad_attempt_index or safe_attempt_margin.")
    safe_trace = _truncate_rollout_trace_by_accepted_steps(
        baseline_rollout.trace,
        accepted_step_limit=safe_accepted_count,
    )

    print(
        "[autodiff-gate] realized-trace-safe-state-trajectory baseline summary: "
        f"attempt_count={baseline_diag['attempt_count']} "
        f"accepted_count={baseline_diag['accepted_count']} "
        f"known_first_bad_attempt_index={known_first_bad_attempt_index} "
        f"safe_attempt_index={safe_attempt_index} "
        f"safe_accepted_count={safe_accepted_count} "
        f"sample_every={int(max(1, sample_every))} "
        f"safe_final_time={float(safe_time_list[-1]):.6e}",
        flush=True,
    )

    state0_static = _parameterized_initial_state(
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        geometry=runtime.geometry,
        n_species=runtime.species.number_species,
        parameter_name=parameter_name,
        parameter_value=jax.lax.stop_gradient(jnp.asarray(baseline_value)),
    )
    prepared_components_static = prepare_transport_solver_components(config, runtime, state0_static)
    solver_static = prepared_components_static["solver"]
    solve_vector_field_static = prepared_components_static["solve_vector_field"]
    prepared_rollout_static = _build_prepared_radau_accepted_rollout(
        solver=solver_static,
        state=state0_static,
        vector_field=solve_vector_field_static,
        species=runtime.species,
    )
    execution_context = _build_prepared_radau_execution_context(
        solver=solver_static,
        prepared_rollout=prepared_rollout_static,
    )

    carry0, carry0_dot = jax.jvp(
        lambda p: _adaptive_rollout_initial_carry_for_parameter(
            p,
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=parameter_name,
        ),
        (jnp.asarray(baseline_value),),
        (jnp.asarray(1.0),),
    )

    print("[autodiff-gate] realized-trace-safe-state-trajectory progress: running realized-trace custom state tangent trajectory", flush=True)
    custom_result = _sampled_realized_trace_state_tangent_trajectory(
        execution_context=execution_context,
        carry0=carry0,
        carry0_dot=carry0_dot,
        trace=safe_trace,
        sample_every=sample_every,
        use_custom=True,
    )
    print("[autodiff-gate] realized-trace-safe-state-trajectory progress: custom trajectory complete; running realized-trace direct state tangent trajectory", flush=True)
    direct_result = _sampled_realized_trace_state_tangent_trajectory(
        execution_context=execution_context,
        carry0=carry0,
        carry0_dot=carry0_dot,
        trace=safe_trace,
        sample_every=sample_every,
        use_custom=False,
    )

    sample_indices = custom_result["sampled_indices"]
    custom_np = np.asarray(jax.device_get(custom_result["sampled_state_tangents"]), dtype=float)
    direct_np = np.asarray(jax.device_get(direct_result["sampled_state_tangents"]), dtype=float)
    sampled_times_np = np.asarray(jax.device_get(custom_result["sampled_times"]), dtype=float)
    unpack_flat = prepared_rollout_static.physics_context.unpack_flat

    def _rel_norm(a: np.ndarray, b: np.ndarray) -> float:
        num = float(np.linalg.norm(a - b))
        den = max(float(np.linalg.norm(b)), 1.0e-10)
        return num / den

    entries = []
    global_max_rel_error = 0.0
    for idx, sample_idx in enumerate(sample_indices):
        custom_step = custom_np[idx]
        direct_step = direct_np[idx]
        custom_state = unpack_flat(jnp.asarray(custom_step))
        direct_state = unpack_flat(jnp.asarray(direct_step))
        custom_density = np.asarray(jax.device_get(custom_state.density), dtype=float)
        direct_density = np.asarray(jax.device_get(direct_state.density), dtype=float)
        custom_pressure = np.asarray(jax.device_get(custom_state.pressure), dtype=float)
        direct_pressure = np.asarray(jax.device_get(direct_state.pressure), dtype=float)
        custom_er = np.asarray(jax.device_get(custom_state.Er), dtype=float)
        direct_er = np.asarray(jax.device_get(direct_state.Er), dtype=float)
        full_rel = _rel_norm(custom_step, direct_step)
        density_rel = _rel_norm(custom_density, direct_density)
        pressure_rel = _rel_norm(custom_pressure, direct_pressure)
        er_rel = _rel_norm(custom_er, direct_er)
        global_max_rel_error = max(global_max_rel_error, full_rel, density_rel, pressure_rel, er_rel)
        entries.append(
            {
                "accepted_index": int(sample_idx + 1),
                "time": float(sampled_times_np[idx]),
                "full_state_relative_error": float(full_rel),
                "density_relative_error": float(density_rel),
                "pressure_relative_error": float(pressure_rel),
                "Er_relative_error": float(er_rel),
            }
        )

    return {
        "config_path": str(config_path),
        "realized_trace_safe_state_trajectory_compare_check": True,
        "parameter_name": parameter_name,
        "baseline_value": baseline_value,
        "known_first_bad_attempt_index": int(known_first_bad_attempt_index),
        "safe_attempt_margin": int(safe_attempt_margin),
        "safe_attempt_index": int(safe_attempt_index),
        "safe_accepted_count": int(safe_accepted_count),
        "safe_final_time": float(safe_time_list[-1]),
        "sample_every": int(max(1, sample_every)),
        "entries": entries,
        "max_relative_error": float(global_max_rel_error),
        "passed": bool(np.isfinite(global_max_rel_error) and global_max_rel_error <= 5.0e-2),
        "rollout_path": {"baseline": baseline_diag},
    }

def build_realized_trace_sparse_checkpoint_compare_report(
    *,
    config_path: Path,
    parameter_name: str,
    device: str | None,
    checkpoint_counts: tuple[int, ...],
) -> dict[str, Any]:
    if parameter_name not in ALLOWED_PARAMETERS:
        raise ValueError(f"parameter_name must be one of {sorted(ALLOWED_PARAMETERS)}")

    checkpoints = tuple(sorted({int(v) for v in checkpoint_counts if int(v) >= 1}))
    if not checkpoints:
        raise ValueError("checkpoint_counts must contain at least one positive accepted-step index.")
    max_checkpoint = max(checkpoints)

    config = _prepare_benchmark_config(config_path, device=device)
    runtime, baseline_state = build_runtime_context(config)
    profile_cfg = _baseline_profile_cfg(config)
    baseline_value = float(profile_cfg[parameter_name])

    _, baseline_rollout = _adaptive_rollout_objectives_for_parameter(
        jnp.asarray(baseline_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        use_realized_schedule_jvp=True,
        accepted_step_limit_override=max_checkpoint,
    )
    baseline_diag = _adaptive_rollout_diagnostics(baseline_rollout)
    total_accepted = int(np.sum(np.asarray(jax.device_get(baseline_rollout.trace.accepted_mask), dtype=bool)))
    if max_checkpoint > total_accepted:
        raise ValueError(f"Requested checkpoint {max_checkpoint} exceeds accepted-count {total_accepted}.")
    checkpoint_trace = _truncate_rollout_trace_by_accepted_steps(
        baseline_rollout.trace,
        accepted_step_limit=max_checkpoint,
    )

    print(
        "[autodiff-gate] realized-trace-sparse-checkpoints baseline summary: "
        f"attempt_count={baseline_diag['attempt_count']} "
        f"accepted_count={baseline_diag['accepted_count']} "
        f"checkpoints={','.join(str(v) for v in checkpoints)}",
        flush=True,
    )

    state0_static = _parameterized_initial_state(
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        geometry=runtime.geometry,
        n_species=runtime.species.number_species,
        parameter_name=parameter_name,
        parameter_value=jax.lax.stop_gradient(jnp.asarray(baseline_value)),
    )
    prepared_components_static = prepare_transport_solver_components(config, runtime, state0_static)
    solver_static = prepared_components_static["solver"]
    solve_vector_field_static = prepared_components_static["solve_vector_field"]
    prepared_rollout_static = _build_prepared_radau_accepted_rollout(
        solver=solver_static,
        state=state0_static,
        vector_field=solve_vector_field_static,
        species=runtime.species,
    )
    execution_context = _build_prepared_radau_execution_context(
        solver=solver_static,
        prepared_rollout=prepared_rollout_static,
    )

    carry0, carry0_dot = jax.jvp(
        lambda p: _adaptive_rollout_initial_carry_for_parameter(
            p,
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=parameter_name,
        ),
        (jnp.asarray(baseline_value),),
        (jnp.asarray(1.0),),
    )

    print("[autodiff-gate] realized-trace-sparse-checkpoints progress: running custom checkpoints", flush=True)
    custom_result = _manual_realized_trace_state_tangent_checkpoints(
        execution_context=execution_context,
        carry0=carry0,
        carry0_dot=carry0_dot,
        trace=checkpoint_trace,
        accepted_checkpoints=checkpoints,
        use_custom=True,
    )
    print("[autodiff-gate] realized-trace-sparse-checkpoints progress: custom complete; running direct checkpoints", flush=True)
    direct_result = _manual_realized_trace_state_tangent_checkpoints(
        execution_context=execution_context,
        carry0=carry0,
        carry0_dot=carry0_dot,
        trace=checkpoint_trace,
        accepted_checkpoints=checkpoints,
        use_custom=False,
    )

    custom_np = np.asarray(jax.device_get(custom_result["state_tangents"]), dtype=float)
    direct_np = np.asarray(jax.device_get(direct_result["state_tangents"]), dtype=float)
    sampled_times_np = np.asarray(jax.device_get(custom_result["times"]), dtype=float)
    unpack_flat = prepared_rollout_static.physics_context.unpack_flat

    def _rel_norm(a: np.ndarray, b: np.ndarray) -> float:
        num = float(np.linalg.norm(a - b))
        den = max(float(np.linalg.norm(b)), 1.0e-10)
        return num / den

    entries = []
    global_max_rel_error = 0.0
    for idx, accepted_idx in enumerate(checkpoints):
        custom_step = custom_np[idx]
        direct_step = direct_np[idx]
        custom_state = unpack_flat(jnp.asarray(custom_step))
        direct_state = unpack_flat(jnp.asarray(direct_step))
        custom_density = np.asarray(jax.device_get(custom_state.density), dtype=float)
        direct_density = np.asarray(jax.device_get(direct_state.density), dtype=float)
        custom_pressure = np.asarray(jax.device_get(custom_state.pressure), dtype=float)
        direct_pressure = np.asarray(jax.device_get(direct_state.pressure), dtype=float)
        custom_er = np.asarray(jax.device_get(custom_state.Er), dtype=float)
        direct_er = np.asarray(jax.device_get(direct_state.Er), dtype=float)
        full_rel = _rel_norm(custom_step, direct_step)
        density_rel = _rel_norm(custom_density, direct_density)
        pressure_rel = _rel_norm(custom_pressure, direct_pressure)
        er_rel = _rel_norm(custom_er, direct_er)
        global_max_rel_error = max(global_max_rel_error, full_rel, density_rel, pressure_rel, er_rel)
        entries.append(
            {
                "accepted_index": int(accepted_idx),
                "time": float(sampled_times_np[idx]),
                "full_state_relative_error": float(full_rel),
                "density_relative_error": float(density_rel),
                "pressure_relative_error": float(pressure_rel),
                "Er_relative_error": float(er_rel),
            }
        )
        print(
            "[autodiff-gate] sparse checkpoint comparison: "
            f"accepted_index={int(accepted_idx)} "
            f"time={float(sampled_times_np[idx]):.6e} "
            f"full_rel_err={float(full_rel):.6e} "
            f"pressure_rel_err={float(pressure_rel):.6e} "
            f"Er_rel_err={float(er_rel):.6e}",
            flush=True,
        )

    return {
        "config_path": str(config_path),
        "realized_trace_sparse_checkpoint_compare_check": True,
        "parameter_name": parameter_name,
        "baseline_value": baseline_value,
        "checkpoint_counts": [int(v) for v in checkpoints],
        "entries": entries,
        "max_relative_error": float(global_max_rel_error),
        "passed": bool(np.isfinite(global_max_rel_error) and global_max_rel_error <= 5.0e-2),
        "rollout_path": {"baseline": baseline_diag},
    }


def build_baseline_dt_path_first_step_field_compare_report(
    *,
    config_path: Path,
    parameter_name: str,
    device: str | None,
    known_first_bad_attempt_index: int,
    safe_attempt_margin: int,
) -> dict[str, Any]:
    if parameter_name not in ALLOWED_PARAMETERS:
        raise ValueError(f"parameter_name must be one of {sorted(ALLOWED_PARAMETERS)}")

    config = _prepare_benchmark_config(config_path, device=device)
    runtime, baseline_state = build_runtime_context(config)
    profile_cfg = _baseline_profile_cfg(config)
    baseline_value = float(profile_cfg[parameter_name])

    _, baseline_rollout = _adaptive_rollout_objectives_for_parameter(
        jnp.asarray(baseline_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        use_realized_schedule_jvp=True,
    )
    baseline_diag = _adaptive_rollout_diagnostics(baseline_rollout)
    safe_attempt_index = int(known_first_bad_attempt_index) - int(safe_attempt_margin)
    safe_time_list = _accepted_time_list_until_attempt_index(baseline_rollout.trace, safe_attempt_index)
    safe_accepted_count = len(safe_time_list)
    if safe_accepted_count <= 0:
        raise ValueError("Safe baseline dt path is empty; adjust known_first_bad_attempt_index or safe_attempt_margin.")
    safe_trace = _truncate_rollout_trace_by_accepted_steps(
        baseline_rollout.trace,
        accepted_step_limit=safe_accepted_count,
    )

    print(
        "[autodiff-gate] baseline-dt-first-step-field baseline summary: "
        f"attempt_count={baseline_diag['attempt_count']} "
        f"accepted_count={baseline_diag['accepted_count']} "
        f"safe_attempt_index={safe_attempt_index} "
        f"safe_final_time={float(safe_time_list[-1]):.6e}",
        flush=True,
    )

    state0_static = _parameterized_initial_state(
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        geometry=runtime.geometry,
        n_species=runtime.species.number_species,
        parameter_name=parameter_name,
        parameter_value=jax.lax.stop_gradient(jnp.asarray(baseline_value)),
    )
    prepared_components_static = prepare_transport_solver_components(config, runtime, state0_static)
    solver_static = prepared_components_static["solver"]
    solve_vector_field_static = prepared_components_static["solve_vector_field"]
    prepared_rollout_static = _build_prepared_radau_accepted_rollout(
        solver=solver_static,
        state=state0_static,
        vector_field=solve_vector_field_static,
        species=runtime.species,
    )
    execution_context = _build_prepared_radau_execution_context(
        solver=solver_static,
        prepared_rollout=prepared_rollout_static,
    )

    carry0, carry0_dot = jax.jvp(
        lambda p: _adaptive_rollout_initial_carry_for_parameter(
            p,
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=parameter_name,
        ),
        (jnp.asarray(baseline_value),),
        (jnp.asarray(1.0),),
    )

    adaptive_result = _sampled_adaptive_state_tangent_trajectory(
        execution_context=execution_context,
        carry0=carry0,
        carry0_dot=carry0_dot,
        trace=safe_trace,
        sample_every=1,
    )
    _, fixed_dt_tangent = jax.jvp(
        lambda p: _adaptive_rollout_flat_state_trajectory_on_time_list(
            p,
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=parameter_name,
            time_list=safe_time_list,
        )[0],
        (jnp.asarray(baseline_value),),
        (jnp.asarray(1.0),),
    )

    adaptive_first = np.asarray(jax.device_get(adaptive_result["sampled_state_tangents"]), dtype=float)[0]
    fixed_first = np.asarray(jax.device_get(fixed_dt_tangent), dtype=float)[0]
    time_first = float(np.asarray(jax.device_get(adaptive_result["sampled_times"]), dtype=float)[0])
    unpack_flat = prepared_rollout_static.physics_context.unpack_flat
    adaptive_state = unpack_flat(jnp.asarray(adaptive_first))
    fixed_state = unpack_flat(jnp.asarray(fixed_first))

    def _component_report(ad_arr, ref_arr):
        ad_np = np.asarray(jax.device_get(ad_arr), dtype=float)
        ref_np = np.asarray(jax.device_get(ref_arr), dtype=float)
        diff = ad_np - ref_np
        ref_norm = max(float(np.linalg.norm(ref_np)), 1.0e-10)
        return {
            "adaptive_max_abs": float(np.max(np.abs(ad_np))),
            "direct_max_abs": float(np.max(np.abs(ref_np))),
            "error_max_abs": float(np.max(np.abs(diff))),
            "relative_error": float(np.linalg.norm(diff) / ref_norm),
        }

    density_report = _component_report(adaptive_state.density, fixed_state.density)
    pressure_report = _component_report(adaptive_state.pressure, fixed_state.pressure)
    er_report = _component_report(adaptive_state.Er, fixed_state.Er)

    return {
        "config_path": str(config_path),
        "baseline_dt_path_first_step_field_compare_check": True,
        "parameter_name": parameter_name,
        "baseline_value": baseline_value,
        "safe_attempt_index": int(safe_attempt_index),
        "safe_final_time": float(safe_time_list[-1]),
        "first_step_time": time_first,
        "density": density_report,
        "pressure": pressure_report,
        "Er": er_report,
        "max_relative_error": float(max(density_report["relative_error"], pressure_report["relative_error"], er_report["relative_error"])),
        "passed": bool(max(density_report["relative_error"], pressure_report["relative_error"], er_report["relative_error"]) <= 5.0e-2),
        "rollout_path": {"baseline": baseline_diag},
    }


def build_baseline_dt_path_first_step_local_tangent_compare_report(
    *,
    config_path: Path,
    parameter_name: str,
    device: str | None,
    known_first_bad_attempt_index: int,
    safe_attempt_margin: int,
) -> dict[str, Any]:
    if parameter_name not in ALLOWED_PARAMETERS:
        raise ValueError(f"parameter_name must be one of {sorted(ALLOWED_PARAMETERS)}")

    config = _prepare_benchmark_config(config_path, device=device)
    runtime, baseline_state = build_runtime_context(config)
    profile_cfg = _baseline_profile_cfg(config)
    baseline_value = float(profile_cfg[parameter_name])

    _, baseline_rollout = _adaptive_rollout_objectives_for_parameter(
        jnp.asarray(baseline_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        use_realized_schedule_jvp=True,
    )
    baseline_diag = _adaptive_rollout_diagnostics(baseline_rollout)
    safe_attempt_index = int(known_first_bad_attempt_index) - int(safe_attempt_margin)
    safe_time_list = _accepted_time_list_until_attempt_index(baseline_rollout.trace, safe_attempt_index)
    if len(safe_time_list) <= 0:
        raise ValueError("Safe baseline dt path is empty; adjust known_first_bad_attempt_index or safe_attempt_margin.")

    print(
        "[autodiff-gate] baseline-dt-first-step-local baseline summary: "
        f"attempt_count={baseline_diag['attempt_count']} "
        f"accepted_count={baseline_diag['accepted_count']} "
        f"safe_attempt_index={safe_attempt_index} "
        f"safe_final_time={float(safe_time_list[-1]):.6e}",
        flush=True,
    )

    state0_static = _parameterized_initial_state(
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        geometry=runtime.geometry,
        n_species=runtime.species.number_species,
        parameter_name=parameter_name,
        parameter_value=jax.lax.stop_gradient(jnp.asarray(baseline_value)),
    )
    prepared_components_static = prepare_transport_solver_components(config, runtime, state0_static)
    solver_static = prepared_components_static["solver"]
    solve_vector_field_static = prepared_components_static["solve_vector_field"]
    prepared_rollout_static = _build_prepared_radau_accepted_rollout(
        solver=solver_static,
        state=state0_static,
        vector_field=solve_vector_field_static,
        species=runtime.species,
    )
    execution_context = _build_prepared_radau_execution_context(
        solver=solver_static,
        prepared_rollout=prepared_rollout_static,
    )

    carry0, carry0_dot = jax.jvp(
        lambda p: _adaptive_rollout_initial_carry_for_parameter(
            p,
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=parameter_name,
        ),
        (jnp.asarray(baseline_value),),
        (jnp.asarray(1.0),),
    )

    print(
        "[autodiff-gate] baseline-dt-first-step-local progress: running custom and direct first-step tangents",
        flush=True,
    )

    custom_primal, custom_tangent = jax.jvp(
        lambda carry: _execute_radau_accepted_step_attempt_autodiff(
            execution_context.kernel_context,
            execution_context.physics_context,
            carry,
            execution_context.attempt_context,
        ),
        (carry0,),
        (carry0_dot,),
    )
    _, direct_tangent = jax.jvp(
        lambda carry: _execute_radau_accepted_step_attempt(
            execution_context.kernel_context,
            execution_context.physics_context,
            carry,
            execution_context.attempt_context,
        ),
        (carry0,),
        (carry0_dot,),
    )
    carry0_dot_restricted = jax.tree_util.tree_map(
        lambda x: None if x is None else jnp.zeros_like(x),
        carry0_dot,
        is_leaf=lambda x: x is None,
    )
    carry0_dot_restricted = dataclasses.replace(
        carry0_dot_restricted,
        y=carry0_dot.y,
        dt=carry0_dot.dt,
    )
    _, restricted_direct_tangent = jax.jvp(
        lambda carry: _execute_radau_accepted_step_attempt(
            execution_context.kernel_context,
            execution_context.physics_context,
            carry,
            execution_context.attempt_context,
        ),
        (carry0,),
        (carry0_dot_restricted,),
    )

    def _zero_optional_pytree(tree):
        return jax.tree_util.tree_map(
            lambda x: None if x is None else jnp.zeros_like(x),
            tree,
            is_leaf=lambda x: x is None,
        )

    def _direct_tangent_with_ablation(**replacements):
        ablated_dot = dataclasses.replace(carry0_dot, **replacements)
        _, tangent = jax.jvp(
            lambda carry: _execute_radau_accepted_step_attempt(
                execution_context.kernel_context,
                execution_context.physics_context,
                carry,
                execution_context.attempt_context,
            ),
            (carry0,),
            (ablated_dot,),
        )
        return tangent

    unpack_flat = prepared_rollout_static.physics_context.unpack_flat
    num_stages = int(execution_context.kernel_context.num_stages)
    state_dim = int(execution_context.kernel_context.state_dim)

    def _field_relative_errors(ad_state, ref_state):
        ad_pressure = np.asarray(jax.device_get(ad_state.pressure), dtype=float)
        ref_pressure = np.asarray(jax.device_get(ref_state.pressure), dtype=float)
        ad_er = np.asarray(jax.device_get(ad_state.Er), dtype=float)
        ref_er = np.asarray(jax.device_get(ref_state.Er), dtype=float)
        return {
            "pressure_relative_error": float(
                np.linalg.norm(ad_pressure - ref_pressure)
                / max(float(np.linalg.norm(ref_pressure)), 1.0e-10)
            ),
            "Er_relative_error": float(
                np.linalg.norm(ad_er - ref_er)
                / max(float(np.linalg.norm(ref_er)), 1.0e-10)
            ),
        }

    def _flat_component_report(ad_arr, ref_arr):
        ad_flat = np.asarray(jax.device_get(ad_arr), dtype=float)
        ref_flat = np.asarray(jax.device_get(ref_arr), dtype=float)
        ad_state = unpack_flat(jnp.asarray(ad_flat))
        ref_state = unpack_flat(jnp.asarray(ref_flat))
        report = _field_relative_errors(ad_state, ref_state)
        report["full_relative_error"] = float(
            np.linalg.norm(ad_flat - ref_flat) / max(float(np.linalg.norm(ref_flat)), 1.0e-10)
        )
        return report

    def _stage_history_report(ad_stage_history, ref_stage_history):
        ad_hist = np.asarray(jax.device_get(ad_stage_history), dtype=float).reshape((num_stages, state_dim))
        ref_hist = np.asarray(jax.device_get(ref_stage_history), dtype=float).reshape((num_stages, state_dim))
        ad_pressure = []
        ref_pressure = []
        ad_er = []
        ref_er = []
        for stage_idx in range(num_stages):
            ad_state = unpack_flat(jnp.asarray(ad_hist[stage_idx]))
            ref_state = unpack_flat(jnp.asarray(ref_hist[stage_idx]))
            ad_pressure.append(np.asarray(jax.device_get(ad_state.pressure), dtype=float))
            ref_pressure.append(np.asarray(jax.device_get(ref_state.pressure), dtype=float))
            ad_er.append(np.asarray(jax.device_get(ad_state.Er), dtype=float))
            ref_er.append(np.asarray(jax.device_get(ref_state.Er), dtype=float))
        ad_pressure_np = np.stack(ad_pressure, axis=0)
        ref_pressure_np = np.stack(ref_pressure, axis=0)
        ad_er_np = np.stack(ad_er, axis=0)
        ref_er_np = np.stack(ref_er, axis=0)
        return {
            "full_relative_error": float(
                np.linalg.norm(ad_hist - ref_hist) / max(float(np.linalg.norm(ref_hist)), 1.0e-10)
            ),
            "pressure_relative_error": float(
                np.linalg.norm(ad_pressure_np - ref_pressure_np)
                / max(float(np.linalg.norm(ref_pressure_np)), 1.0e-10)
            ),
            "Er_relative_error": float(
                np.linalg.norm(ad_er_np - ref_er_np)
                / max(float(np.linalg.norm(ref_er_np)), 1.0e-10)
            ),
        }

    trial_y_report = _flat_component_report(custom_tangent.trial_y, direct_tangent.trial_y)
    carry_after_attempt_y_report = _flat_component_report(
        custom_tangent.carry_after_attempt.y,
        direct_tangent.carry_after_attempt.y,
    )
    stage_history_report = _stage_history_report(
        custom_tangent.stage_history,
        direct_tangent.stage_history,
    )

    return {
        "config_path": str(config_path),
        "baseline_dt_path_first_step_local_tangent_compare_check": True,
        "parameter_name": parameter_name,
        "baseline_value": baseline_value,
        "safe_attempt_index": int(safe_attempt_index),
        "safe_final_time": float(safe_time_list[-1]),
        "first_step_time": float(np.asarray(jax.device_get(carry0.t + custom_primal.trial_dt), dtype=float)),
        "trial_y": trial_y_report,
        "carry_after_attempt_y": carry_after_attempt_y_report,
        "stage_history": stage_history_report,
        "max_relative_error": float(
            max(
                trial_y_report["Er_relative_error"],
                carry_after_attempt_y_report["Er_relative_error"],
                stage_history_report["Er_relative_error"],
            )
        ),
        "passed": bool(
            max(
                trial_y_report["Er_relative_error"],
                carry_after_attempt_y_report["Er_relative_error"],
                stage_history_report["Er_relative_error"],
            )
            <= 5.0e-2
        ),
        "rollout_path": {"baseline": baseline_diag},
    }


def build_baseline_dt_path_first_step_exact_local_tangent_compare_report(
    *,
    config_path: Path,
    parameter_name: str,
    device: str | None,
    known_first_bad_attempt_index: int,
    safe_attempt_margin: int,
) -> dict[str, Any]:
    if parameter_name not in ALLOWED_PARAMETERS:
        raise ValueError(f"parameter_name must be one of {sorted(ALLOWED_PARAMETERS)}")

    config = _prepare_benchmark_config(config_path, device=device)
    runtime, baseline_state = build_runtime_context(config)
    profile_cfg = _baseline_profile_cfg(config)
    baseline_value = float(profile_cfg[parameter_name])

    _, baseline_rollout = _adaptive_rollout_objectives_for_parameter(
        jnp.asarray(baseline_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        use_realized_schedule_jvp=True,
    )
    baseline_diag = _adaptive_rollout_diagnostics(baseline_rollout)
    safe_attempt_index = int(known_first_bad_attempt_index) - int(safe_attempt_margin)
    safe_time_list = _accepted_time_list_until_attempt_index(baseline_rollout.trace, safe_attempt_index)
    if len(safe_time_list) <= 0:
        raise ValueError("Safe baseline dt path is empty; adjust known_first_bad_attempt_index or safe_attempt_margin.")

    print(
        "[autodiff-gate] baseline-dt-first-step-exact-local baseline summary: "
        f"attempt_count={baseline_diag['attempt_count']} "
        f"accepted_count={baseline_diag['accepted_count']} "
        f"safe_attempt_index={safe_attempt_index} "
        f"safe_final_time={float(safe_time_list[-1]):.6e}",
        flush=True,
    )

    state0_static = _parameterized_initial_state(
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        geometry=runtime.geometry,
        n_species=runtime.species.number_species,
        parameter_name=parameter_name,
        parameter_value=jax.lax.stop_gradient(jnp.asarray(baseline_value)),
    )
    prepared_components_static = prepare_transport_solver_components(config, runtime, state0_static)
    solver_static = prepared_components_static["solver"]
    solve_vector_field_static = prepared_components_static["solve_vector_field"]
    prepared_rollout_static = _build_prepared_radau_accepted_rollout(
        solver=solver_static,
        state=state0_static,
        vector_field=solve_vector_field_static,
        species=runtime.species,
    )
    execution_context = _build_prepared_radau_execution_context(
        solver=solver_static,
        prepared_rollout=prepared_rollout_static,
    )

    carry0, carry0_dot = jax.jvp(
        lambda p: _adaptive_rollout_initial_carry_for_parameter(
            p,
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=parameter_name,
        ),
        (jnp.asarray(baseline_value),),
        (jnp.asarray(1.0),),
    )

    print(
        "[autodiff-gate] baseline-dt-first-step-exact-local progress: running custom, direct, and exact first-step tangents",
        flush=True,
    )

    custom_primal, custom_tangent = jax.jvp(
        lambda carry: _execute_radau_accepted_step_attempt_autodiff(
            execution_context.kernel_context,
            execution_context.physics_context,
            carry,
            execution_context.attempt_context,
        ),
        (carry0,),
        (carry0_dot,),
    )
    _, direct_tangent = jax.jvp(
        lambda carry: _execute_radau_accepted_step_attempt(
            execution_context.kernel_context,
            execution_context.physics_context,
            carry,
            execution_context.attempt_context,
        ),
        (carry0,),
        (carry0_dot,),
    )
    carry0_dot_restricted = jax.tree_util.tree_map(
        lambda x: None if x is None else jnp.zeros_like(x),
        carry0_dot,
        is_leaf=lambda x: x is None,
    )
    carry0_dot_restricted = dataclasses.replace(
        carry0_dot_restricted,
        y=carry0_dot.y,
        dt=carry0_dot.dt,
    )
    _, restricted_direct_tangent = jax.jvp(
        lambda carry: _execute_radau_accepted_step_attempt(
            execution_context.kernel_context,
            execution_context.physics_context,
            carry,
            execution_context.attempt_context,
        ),
        (carry0,),
        (carry0_dot_restricted,),
    )

    def _direct_tangent_with_ablation(**replacements):
        ablated_dot = dataclasses.replace(carry0_dot, **replacements)
        _, tangent = jax.jvp(
            lambda carry: _execute_radau_accepted_step_attempt(
                execution_context.kernel_context,
                execution_context.physics_context,
                carry,
                execution_context.attempt_context,
            ),
            (carry0,),
            (ablated_dot,),
        )
        return tangent

    lagged_response, _, _ = _radau_prepare_lagged_response(
        execution_context.kernel_context,
        carry0,
        execution_context.physics_context.unpack_flat,
        execution_context.physics_context.project_flat,
        execution_context.physics_context.build_lagged_response,
    )

    def _rhs_eval(t_eval, y_eval):
        return _radau_eval_rhs(
            t_eval,
            y_eval,
            lagged_response,
            execution_context.physics_context.flat_rhs,
            execution_context.physics_context.flat_rhs_with_lagged_response,
        )

    f0 = _rhs_eval(carry0.t, carry0.y)
    z_final = custom_primal.stage_history
    h_value = custom_primal.trial_dt
    jacobian_ref = custom_primal.jacobian_out

    def _residual_wrt_z(z_flat):
        return _radau_stage_residual(
            execution_context.kernel_context,
            execution_context.physics_context,
            flat_y=carry0.y,
            t_value=carry0.t,
            h_value=h_value,
            z_flat=z_flat,
            f0=f0,
            jacobian_ref=jacobian_ref,
            lagged_response=lagged_response,
        )

    def _residual_wrt_y_h(flat_y, h_scalar):
        f0_local = _rhs_eval(carry0.t, flat_y)
        return _radau_stage_residual(
            execution_context.kernel_context,
            execution_context.physics_context,
            flat_y=flat_y,
            t_value=carry0.t,
            h_value=h_scalar,
            z_flat=z_final,
            f0=f0_local,
            jacobian_ref=jacobian_ref,
            lagged_response=lagged_response,
        )

    residual_z_jacobian = jax.jacfwd(_residual_wrt_z)(z_final)
    _, residual_source = jax.jvp(
        _residual_wrt_y_h,
        (carry0.y, h_value),
        (carry0_dot.y, carry0_dot.dt),
    )
    exact_dz_flat = -jnp.linalg.solve(residual_z_jacobian, residual_source)
    exact_dz_stages = exact_dz_flat.reshape(
        (execution_context.kernel_context.num_stages, execution_context.kernel_context.state_dim)
    )
    exact_dy_next = (
        carry0_dot.y
        + carry0_dot.dt
        * (
            execution_context.kernel_context.b
            @ z_final.reshape(
                (execution_context.kernel_context.num_stages, execution_context.kernel_context.state_dim)
            )
        )
        + h_value * (execution_context.kernel_context.b @ exact_dz_stages)
    )

    unpack_flat = prepared_rollout_static.physics_context.unpack_flat

    def _field_relative_errors(ad_state, ref_state):
        ad_pressure = np.asarray(jax.device_get(ad_state.pressure), dtype=float)
        ref_pressure = np.asarray(jax.device_get(ref_state.pressure), dtype=float)
        ad_er = np.asarray(jax.device_get(ad_state.Er), dtype=float)
        ref_er = np.asarray(jax.device_get(ref_state.Er), dtype=float)
        return {
            "pressure_relative_error": float(
                np.linalg.norm(ad_pressure - ref_pressure)
                / max(float(np.linalg.norm(ref_pressure)), 1.0e-10)
            ),
            "Er_relative_error": float(
                np.linalg.norm(ad_er - ref_er)
                / max(float(np.linalg.norm(ref_er)), 1.0e-10)
            ),
        }

    def _flat_component_report(ad_arr, ref_arr):
        ad_flat = np.asarray(jax.device_get(ad_arr), dtype=float)
        ref_flat = np.asarray(jax.device_get(ref_arr), dtype=float)
        ad_state = unpack_flat(jnp.asarray(ad_flat))
        ref_state = unpack_flat(jnp.asarray(ref_flat))
        report = _field_relative_errors(ad_state, ref_state)
        report["full_relative_error"] = float(
            np.linalg.norm(ad_flat - ref_flat) / max(float(np.linalg.norm(ref_flat)), 1.0e-10)
        )
        return report

    def _stage_history_report(ad_stage_history, ref_stage_history):
        num_stages = int(execution_context.kernel_context.num_stages)
        state_dim = int(execution_context.kernel_context.state_dim)
        ad_hist = np.asarray(jax.device_get(ad_stage_history), dtype=float).reshape((num_stages, state_dim))
        ref_hist = np.asarray(jax.device_get(ref_stage_history), dtype=float).reshape((num_stages, state_dim))
        ad_pressure = []
        ref_pressure = []
        ad_er = []
        ref_er = []
        for stage_idx in range(num_stages):
            ad_state = unpack_flat(jnp.asarray(ad_hist[stage_idx]))
            ref_state = unpack_flat(jnp.asarray(ref_hist[stage_idx]))
            ad_pressure.append(np.asarray(jax.device_get(ad_state.pressure), dtype=float))
            ref_pressure.append(np.asarray(jax.device_get(ref_state.pressure), dtype=float))
            ad_er.append(np.asarray(jax.device_get(ad_state.Er), dtype=float))
            ref_er.append(np.asarray(jax.device_get(ref_state.Er), dtype=float))
        ad_pressure_np = np.stack(ad_pressure, axis=0)
        ref_pressure_np = np.stack(ref_pressure, axis=0)
        ad_er_np = np.stack(ad_er, axis=0)
        ref_er_np = np.stack(ref_er, axis=0)
        return {
            "full_relative_error": float(
                np.linalg.norm(ad_hist - ref_hist) / max(float(np.linalg.norm(ref_hist)), 1.0e-10)
            ),
            "pressure_relative_error": float(
                np.linalg.norm(ad_pressure_np - ref_pressure_np)
                / max(float(np.linalg.norm(ref_pressure_np)), 1.0e-10)
            ),
            "Er_relative_error": float(
                np.linalg.norm(ad_er_np - ref_er_np)
                / max(float(np.linalg.norm(ref_er_np)), 1.0e-10)
            ),
        }

    def _zero_optional_pytree(tree):
        return jax.tree_util.tree_map(
            lambda x: None if x is None else jnp.zeros_like(x),
            tree,
            is_leaf=lambda x: x is None,
        )

    exact_trial_y = exact_dy_next
    exact_stage_history = exact_dz_flat

    custom_vs_direct = {
        "trial_y": _flat_component_report(custom_tangent.trial_y, direct_tangent.trial_y),
        "stage_history": _stage_history_report(custom_tangent.stage_history, direct_tangent.stage_history),
    }
    exact_vs_direct = {
        "trial_y": _flat_component_report(exact_trial_y, direct_tangent.trial_y),
        "stage_history": _stage_history_report(exact_stage_history, direct_tangent.stage_history),
    }
    custom_vs_exact = {
        "trial_y": _flat_component_report(custom_tangent.trial_y, exact_trial_y),
        "stage_history": _stage_history_report(custom_tangent.stage_history, exact_stage_history),
    }
    restricted_direct_vs_direct = {
        "trial_y": _flat_component_report(restricted_direct_tangent.trial_y, direct_tangent.trial_y),
        "stage_history": _stage_history_report(restricted_direct_tangent.stage_history, direct_tangent.stage_history),
    }
    custom_vs_restricted_direct = {
        "trial_y": _flat_component_report(custom_tangent.trial_y, restricted_direct_tangent.trial_y),
        "stage_history": _stage_history_report(custom_tangent.stage_history, restricted_direct_tangent.stage_history),
    }

    ablation_specs = {
        "prev_stages_zeroed": {
            "prev_stages": jnp.zeros_like(carry0_dot.prev_stages),
        },
        "lagged_response_cache_zeroed": {
            "lagged_response_cache": _zero_optional_pytree(carry0_dot.lagged_response_cache),
        },
        "lagged_reference_y_zeroed": {
            "lagged_reference_y": jnp.zeros_like(carry0_dot.lagged_reference_y),
        },
        "linearization_cache_zeroed": {
            "jacobian": jnp.zeros_like(carry0_dot.jacobian),
            "cache_valid": jnp.zeros_like(carry0_dot.cache_valid),
            "cache_dt": jnp.zeros_like(carry0_dot.cache_dt),
            "cache_age": jnp.zeros_like(carry0_dot.cache_age),
            "real_lu": jnp.zeros_like(carry0_dot.real_lu),
            "real_piv": jnp.zeros(carry0_dot.real_piv.shape, dtype=carry0_dot.real_piv.dtype),
            "complex_lu": jnp.zeros_like(carry0_dot.complex_lu),
            "complex_piv": jnp.zeros(carry0_dot.complex_piv.shape, dtype=carry0_dot.complex_piv.dtype),
        },
        "step_history_meta_zeroed": {
            "prev_error": jnp.zeros_like(carry0_dot.prev_error),
            "prev_dt": jnp.zeros_like(carry0_dot.prev_dt),
            "prev_theta_final": jnp.zeros_like(carry0_dot.prev_theta_final),
            "prev_newton_iter_count": jnp.zeros(carry0_dot.prev_newton_iter_count.shape, dtype=carry0_dot.prev_newton_iter_count.dtype),
        },
    }

    carry_field_ablations = {}
    for label, replacements in ablation_specs.items():
        ablated_direct_tangent = _direct_tangent_with_ablation(**replacements)
        carry_field_ablations[label] = {
            "ablated_direct_vs_direct": {
                "trial_y": _flat_component_report(ablated_direct_tangent.trial_y, direct_tangent.trial_y),
                "stage_history": _stage_history_report(ablated_direct_tangent.stage_history, direct_tangent.stage_history),
            },
            "custom_vs_ablated_direct": {
                "trial_y": _flat_component_report(custom_tangent.trial_y, ablated_direct_tangent.trial_y),
                "stage_history": _stage_history_report(custom_tangent.stage_history, ablated_direct_tangent.stage_history),
            },
        }

    return {
        "config_path": str(config_path),
        "baseline_dt_path_first_step_exact_local_tangent_compare_check": True,
        "parameter_name": parameter_name,
        "baseline_value": baseline_value,
        "safe_attempt_index": int(safe_attempt_index),
        "safe_final_time": float(safe_time_list[-1]),
        "first_step_time": float(np.asarray(jax.device_get(carry0.t + custom_primal.trial_dt), dtype=float)),
        "custom_vs_direct": custom_vs_direct,
        "exact_vs_direct": exact_vs_direct,
        "custom_vs_exact": custom_vs_exact,
        "restricted_direct_vs_direct": restricted_direct_vs_direct,
        "custom_vs_restricted_direct": custom_vs_restricted_direct,
        "carry_field_ablations": carry_field_ablations,
        "max_relative_error": float(
            max(
                custom_vs_direct["trial_y"]["Er_relative_error"],
                custom_vs_direct["stage_history"]["Er_relative_error"],
                exact_vs_direct["trial_y"]["Er_relative_error"],
                exact_vs_direct["stage_history"]["Er_relative_error"],
            )
        ),
        "passed": True,
        "rollout_path": {"baseline": baseline_diag},
    }


def build_baseline_dt_path_second_step_carry_ablation_report(
    *,
    config_path: Path,
    parameter_name: str,
    device: str | None,
) -> dict[str, Any]:
    if parameter_name not in ALLOWED_PARAMETERS:
        raise ValueError(f"parameter_name must be one of {sorted(ALLOWED_PARAMETERS)}")

    config = _prepare_benchmark_config(config_path, device=device)
    runtime, baseline_state = build_runtime_context(config)
    profile_cfg = _baseline_profile_cfg(config)
    baseline_value = float(profile_cfg[parameter_name])

    _, baseline_rollout = _adaptive_rollout_objectives_for_parameter(
        jnp.asarray(baseline_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        use_realized_schedule_jvp=True,
    )
    baseline_diag = _adaptive_rollout_diagnostics(baseline_rollout)

    accepted_mask_np = np.asarray(jax.device_get(baseline_rollout.trace.accepted_mask), dtype=bool)
    accepted_attempt_indices = np.flatnonzero(accepted_mask_np)
    if accepted_attempt_indices.size < 2:
        raise ValueError("Need at least two accepted attempts for second-step carry ablation debug.")
    idx0 = int(accepted_attempt_indices[0])
    idx1 = int(accepted_attempt_indices[1])

    print(
        "[autodiff-gate] baseline-dt-second-step-carry baseline summary: "
        f"attempt_count={baseline_diag['attempt_count']} "
        f"accepted_count={baseline_diag['accepted_count']} "
        f"first_accepted_attempt={idx0} "
        f"second_accepted_attempt={idx1}",
        flush=True,
    )

    state0_static = _parameterized_initial_state(
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        geometry=runtime.geometry,
        n_species=runtime.species.number_species,
        parameter_name=parameter_name,
        parameter_value=jax.lax.stop_gradient(jnp.asarray(baseline_value)),
    )
    prepared_components_static = prepare_transport_solver_components(config, runtime, state0_static)
    solver_static = prepared_components_static["solver"]
    solve_vector_field_static = prepared_components_static["solve_vector_field"]
    prepared_rollout_static = _build_prepared_radau_accepted_rollout(
        solver=solver_static,
        state=state0_static,
        vector_field=solve_vector_field_static,
        species=runtime.species,
    )
    execution_context = _build_prepared_radau_execution_context(
        solver=solver_static,
        prepared_rollout=prepared_rollout_static,
    )

    carry0, carry0_dot = jax.jvp(
        lambda p: _adaptive_rollout_initial_carry_for_parameter(
            p,
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=parameter_name,
        ),
        (jnp.asarray(baseline_value),),
        (jnp.asarray(1.0),),
    )

    attempted_dts = np.asarray(jax.device_get(baseline_rollout.trace.attempted_dts), dtype=float)
    next_dts = np.asarray(jax.device_get(baseline_rollout.trace.next_dts), dtype=float)
    next_recent_reject_count = np.asarray(jax.device_get(baseline_rollout.trace.next_recent_reject_count))
    next_regrowth_cooldown = np.asarray(jax.device_get(baseline_rollout.trace.next_regrowth_cooldown))
    next_easy_growth_streak = np.asarray(jax.device_get(baseline_rollout.trace.next_easy_growth_streak))
    next_lagged_response_valid = np.asarray(jax.device_get(baseline_rollout.trace.next_lagged_response_valid))
    step_ts = np.asarray(jax.device_get(baseline_rollout.trace.step_ts), dtype=float)

    def _zero_optional_pytree(tree):
        return jax.tree_util.tree_map(
            lambda x: None if x is None else jnp.zeros_like(x),
            tree,
            is_leaf=lambda x: x is None,
        )

    def _attempt_update(
        carry_value,
        attempt_fn,
        *,
        dt_value,
        next_dt_value,
        recent_reject_count_value,
        regrowth_cooldown_value,
        easy_growth_streak_value,
        lagged_response_valid_value,
    ):
        carry_for_step = dataclasses.replace(carry_value, dt=jnp.asarray(dt_value, dtype=execution_context.dtype))
        attempt_result = attempt_fn(
            execution_context.kernel_context,
            execution_context.physics_context,
            _radau_carry_with_forward_only_jvp_fields(carry_for_step),
            execution_context.attempt_context,
        )
        project_flat = execution_context.physics_context.project_flat
        accepted_y = project_flat(attempt_result.trial_y) if project_flat is not None else None
        if accepted_y is None:
            accepted_y = attempt_result.trial_y
        return dataclasses.replace(
            attempt_result.carry_after_attempt,
            t=carry_value.t + jnp.asarray(dt_value, dtype=execution_context.dtype),
            y=accepted_y,
            dt=jnp.asarray(next_dt_value, dtype=execution_context.dtype),
            prev_error=jnp.maximum(
                attempt_result.err_norm,
                jnp.asarray(1.0e-12, dtype=execution_context.dtype),
            ),
            prev_stages=attempt_result.stage_history,
            prev_dt=jnp.asarray(dt_value, dtype=execution_context.dtype),
            recent_reject_count=jnp.asarray(recent_reject_count_value),
            regrowth_cooldown=jnp.asarray(regrowth_cooldown_value),
            easy_growth_streak=jnp.asarray(easy_growth_streak_value),
            lagged_response_valid=jnp.asarray(lagged_response_valid_value),
            jacobian=attempt_result.jacobian_out,
            cache_valid=attempt_result.cache_valid_out,
            cache_dt=attempt_result.cache_dt_out,
            cache_age=attempt_result.cache_age_out,
            real_lu=attempt_result.real_lu_out,
            real_piv=attempt_result.real_piv_out,
            complex_lu=attempt_result.complex_lu_out,
            complex_piv=attempt_result.complex_piv_out,
            prev_theta_final=attempt_result.theta_final,
            prev_newton_iter_count=attempt_result.newton_iter_count,
        )

    step0_kwargs = {
        "dt_value": attempted_dts[idx0],
        "next_dt_value": next_dts[idx0],
        "recent_reject_count_value": next_recent_reject_count[idx0],
        "regrowth_cooldown_value": next_regrowth_cooldown[idx0],
        "easy_growth_streak_value": next_easy_growth_streak[idx0],
        "lagged_response_valid_value": next_lagged_response_valid[idx0],
    }
    step1_kwargs = {
        "dt_value": attempted_dts[idx1],
        "next_dt_value": next_dts[idx1],
        "recent_reject_count_value": next_recent_reject_count[idx1],
        "regrowth_cooldown_value": next_regrowth_cooldown[idx1],
        "easy_growth_streak_value": next_easy_growth_streak[idx1],
        "lagged_response_valid_value": next_lagged_response_valid[idx1],
    }

    custom_carry1, custom_carry1_dot = jax.jvp(
        lambda c: _attempt_update(c, _execute_radau_accepted_step_attempt_autodiff, **step0_kwargs),
        (carry0,),
        (carry0_dot,),
    )
    direct_carry1, direct_carry1_dot = jax.jvp(
        lambda c: _attempt_update(c, _execute_radau_accepted_step_attempt, **step0_kwargs),
        (carry0,),
        (carry0_dot,),
    )
    _, custom_step2_result = jax.jvp(
        lambda c: _attempt_update(c, _execute_radau_accepted_step_attempt_autodiff, **step1_kwargs),
        (custom_carry1,),
        (custom_carry1_dot,),
    )
    _, direct_step2_result = jax.jvp(
        lambda c: _attempt_update(c, _execute_radau_accepted_step_attempt, **step1_kwargs),
        (direct_carry1,),
        (direct_carry1_dot,),
    )

    def _direct_step2_with_ablation(**replacements):
        ablated_dot = dataclasses.replace(direct_carry1_dot, **replacements)
        _, tangent = jax.jvp(
            lambda c: _attempt_update(c, _execute_radau_accepted_step_attempt, **step1_kwargs),
            (direct_carry1,),
            (ablated_dot,),
        )
        return tangent

    unpack_flat = prepared_rollout_static.physics_context.unpack_flat

    def _field_relative_errors(ad_state, ref_state):
        ad_pressure = np.asarray(jax.device_get(ad_state.pressure), dtype=float)
        ref_pressure = np.asarray(jax.device_get(ref_state.pressure), dtype=float)
        ad_er = np.asarray(jax.device_get(ad_state.Er), dtype=float)
        ref_er = np.asarray(jax.device_get(ref_state.Er), dtype=float)
        return {
            "pressure_relative_error": float(
                np.linalg.norm(ad_pressure - ref_pressure)
                / max(float(np.linalg.norm(ref_pressure)), 1.0e-10)
            ),
            "Er_relative_error": float(
                np.linalg.norm(ad_er - ref_er)
                / max(float(np.linalg.norm(ref_er)), 1.0e-10)
            ),
        }

    def _flat_component_report(ad_arr, ref_arr):
        ad_flat = np.asarray(jax.device_get(ad_arr), dtype=float)
        ref_flat = np.asarray(jax.device_get(ref_arr), dtype=float)
        ad_state = unpack_flat(jnp.asarray(ad_flat))
        ref_state = unpack_flat(jnp.asarray(ref_flat))
        report = _field_relative_errors(ad_state, ref_state)
        report["full_relative_error"] = float(
            np.linalg.norm(ad_flat - ref_flat) / max(float(np.linalg.norm(ref_flat)), 1.0e-10)
        )
        return report

    custom_vs_direct = {
        "trial_y": _flat_component_report(custom_step2_result.y, direct_step2_result.y),
    }
    carry_after_step1_custom_vs_direct = {
        "trial_y": _flat_component_report(custom_carry1_dot.y, direct_carry1_dot.y),
    }

    ablation_specs = {
        "prev_stages_zeroed": {
            "prev_stages": jnp.zeros_like(direct_carry1_dot.prev_stages),
        },
        "lagged_response_cache_zeroed": {
            "lagged_response_cache": _zero_optional_pytree(direct_carry1_dot.lagged_response_cache),
        },
        "lagged_reference_y_zeroed": {
            "lagged_reference_y": jnp.zeros_like(direct_carry1_dot.lagged_reference_y),
        },
        "linearization_cache_zeroed": {
            "jacobian": jnp.zeros_like(direct_carry1_dot.jacobian),
            "cache_valid": jnp.zeros_like(direct_carry1_dot.cache_valid),
            "cache_dt": jnp.zeros_like(direct_carry1_dot.cache_dt),
            "cache_age": jnp.zeros_like(direct_carry1_dot.cache_age),
            "real_lu": jnp.zeros_like(direct_carry1_dot.real_lu),
            "real_piv": jnp.zeros(direct_carry1_dot.real_piv.shape, dtype=direct_carry1_dot.real_piv.dtype),
            "complex_lu": jnp.zeros_like(direct_carry1_dot.complex_lu),
            "complex_piv": jnp.zeros(direct_carry1_dot.complex_piv.shape, dtype=direct_carry1_dot.complex_piv.dtype),
        },
        "step_history_meta_zeroed": {
            "prev_error": jnp.zeros_like(direct_carry1_dot.prev_error),
            "prev_dt": jnp.zeros_like(direct_carry1_dot.prev_dt),
            "prev_theta_final": jnp.zeros_like(direct_carry1_dot.prev_theta_final),
            "prev_newton_iter_count": jnp.zeros(
                direct_carry1_dot.prev_newton_iter_count.shape,
                dtype=direct_carry1_dot.prev_newton_iter_count.dtype,
            ),
        },
    }

    carry_field_ablations = {}
    for label, replacements in ablation_specs.items():
        ablated_direct_step2 = _direct_step2_with_ablation(**replacements)
        carry_field_ablations[label] = {
            "ablated_direct_vs_direct": {
                "trial_y": _flat_component_report(ablated_direct_step2.y, direct_step2_result.y),
            },
            "custom_vs_ablated_direct": {
                "trial_y": _flat_component_report(custom_step2_result.y, ablated_direct_step2.y),
            },
        }

    return {
        "config_path": str(config_path),
        "baseline_dt_path_second_step_carry_ablation_check": True,
        "parameter_name": parameter_name,
        "baseline_value": baseline_value,
        "first_step_time": float(step_ts[idx0]),
        "second_step_time": float(step_ts[idx1]),
        "custom_vs_direct": custom_vs_direct,
        "carry_after_step1_custom_vs_direct": carry_after_step1_custom_vs_direct,
        "carry_field_ablations": carry_field_ablations,
        "max_relative_error": float(custom_vs_direct["trial_y"]["Er_relative_error"]),
        "passed": bool(custom_vs_direct["trial_y"]["Er_relative_error"] <= 5.0e-2),
        "rollout_path": {"baseline": baseline_diag},
    }


def build_baseline_dt_path_third_step_carry_ablation_report(
    *,
    config_path: Path,
    parameter_name: str,
    device: str | None,
) -> dict[str, Any]:
    if parameter_name not in ALLOWED_PARAMETERS:
        raise ValueError(f"parameter_name must be one of {sorted(ALLOWED_PARAMETERS)}")

    config = _prepare_benchmark_config(config_path, device=device)
    runtime, baseline_state = build_runtime_context(config)
    profile_cfg = _baseline_profile_cfg(config)
    baseline_value = float(profile_cfg[parameter_name])

    _, baseline_rollout = _adaptive_rollout_objectives_for_parameter(
        jnp.asarray(baseline_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        use_realized_schedule_jvp=True,
    )
    baseline_diag = _adaptive_rollout_diagnostics(baseline_rollout)

    accepted_mask_np = np.asarray(jax.device_get(baseline_rollout.trace.accepted_mask), dtype=bool)
    accepted_attempt_indices = np.flatnonzero(accepted_mask_np)
    if accepted_attempt_indices.size < 3:
        raise ValueError("Need at least three accepted attempts for third-step carry ablation debug.")
    idx0 = int(accepted_attempt_indices[0])
    idx1 = int(accepted_attempt_indices[1])
    idx2 = int(accepted_attempt_indices[2])

    print(
        "[autodiff-gate] baseline-dt-third-step-carry baseline summary: "
        f"attempt_count={baseline_diag['attempt_count']} "
        f"accepted_count={baseline_diag['accepted_count']} "
        f"first_accepted_attempt={idx0} "
        f"second_accepted_attempt={idx1} "
        f"third_accepted_attempt={idx2}",
        flush=True,
    )

    state0_static = _parameterized_initial_state(
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        geometry=runtime.geometry,
        n_species=runtime.species.number_species,
        parameter_name=parameter_name,
        parameter_value=jax.lax.stop_gradient(jnp.asarray(baseline_value)),
    )
    prepared_components_static = prepare_transport_solver_components(config, runtime, state0_static)
    solver_static = prepared_components_static["solver"]
    solve_vector_field_static = prepared_components_static["solve_vector_field"]
    prepared_rollout_static = _build_prepared_radau_accepted_rollout(
        solver=solver_static,
        state=state0_static,
        vector_field=solve_vector_field_static,
        species=runtime.species,
    )
    execution_context = _build_prepared_radau_execution_context(
        solver=solver_static,
        prepared_rollout=prepared_rollout_static,
    )

    carry0, carry0_dot = jax.jvp(
        lambda p: _adaptive_rollout_initial_carry_for_parameter(
            p,
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=parameter_name,
        ),
        (jnp.asarray(baseline_value),),
        (jnp.asarray(1.0),),
    )

    attempted_dts = np.asarray(jax.device_get(baseline_rollout.trace.attempted_dts), dtype=float)
    next_dts = np.asarray(jax.device_get(baseline_rollout.trace.next_dts), dtype=float)
    next_recent_reject_count = np.asarray(jax.device_get(baseline_rollout.trace.next_recent_reject_count))
    next_regrowth_cooldown = np.asarray(jax.device_get(baseline_rollout.trace.next_regrowth_cooldown))
    next_easy_growth_streak = np.asarray(jax.device_get(baseline_rollout.trace.next_easy_growth_streak))
    next_lagged_response_valid = np.asarray(jax.device_get(baseline_rollout.trace.next_lagged_response_valid))
    step_ts = np.asarray(jax.device_get(baseline_rollout.trace.step_ts), dtype=float)

    def _zero_optional_pytree(tree):
        return jax.tree_util.tree_map(
            lambda x: None if x is None else jnp.zeros_like(x),
            tree,
            is_leaf=lambda x: x is None,
        )

    def _attempt_update(
        carry_value,
        attempt_fn,
        *,
        dt_value,
        next_dt_value,
        recent_reject_count_value,
        regrowth_cooldown_value,
        easy_growth_streak_value,
        lagged_response_valid_value,
    ):
        carry_for_step = dataclasses.replace(carry_value, dt=jnp.asarray(dt_value, dtype=execution_context.dtype))
        attempt_result = attempt_fn(
            execution_context.kernel_context,
            execution_context.physics_context,
            _radau_carry_with_forward_only_jvp_fields(carry_for_step),
            execution_context.attempt_context,
        )
        project_flat = execution_context.physics_context.project_flat
        accepted_y = project_flat(attempt_result.trial_y) if project_flat is not None else None
        if accepted_y is None:
            accepted_y = attempt_result.trial_y
        return dataclasses.replace(
            attempt_result.carry_after_attempt,
            t=carry_value.t + jnp.asarray(dt_value, dtype=execution_context.dtype),
            y=accepted_y,
            dt=jnp.asarray(next_dt_value, dtype=execution_context.dtype),
            prev_error=jnp.maximum(
                attempt_result.err_norm,
                jnp.asarray(1.0e-12, dtype=execution_context.dtype),
            ),
            prev_stages=attempt_result.stage_history,
            prev_dt=jnp.asarray(dt_value, dtype=execution_context.dtype),
            recent_reject_count=jnp.asarray(recent_reject_count_value),
            regrowth_cooldown=jnp.asarray(regrowth_cooldown_value),
            easy_growth_streak=jnp.asarray(easy_growth_streak_value),
            lagged_response_valid=jnp.asarray(lagged_response_valid_value),
            jacobian=attempt_result.jacobian_out,
            cache_valid=attempt_result.cache_valid_out,
            cache_dt=attempt_result.cache_dt_out,
            cache_age=attempt_result.cache_age_out,
            real_lu=attempt_result.real_lu_out,
            real_piv=attempt_result.real_piv_out,
            complex_lu=attempt_result.complex_lu_out,
            complex_piv=attempt_result.complex_piv_out,
            prev_theta_final=attempt_result.theta_final,
            prev_newton_iter_count=attempt_result.newton_iter_count,
        )

    step0_kwargs = {
        "dt_value": attempted_dts[idx0],
        "next_dt_value": next_dts[idx0],
        "recent_reject_count_value": next_recent_reject_count[idx0],
        "regrowth_cooldown_value": next_regrowth_cooldown[idx0],
        "easy_growth_streak_value": next_easy_growth_streak[idx0],
        "lagged_response_valid_value": next_lagged_response_valid[idx0],
    }
    step1_kwargs = {
        "dt_value": attempted_dts[idx1],
        "next_dt_value": next_dts[idx1],
        "recent_reject_count_value": next_recent_reject_count[idx1],
        "regrowth_cooldown_value": next_regrowth_cooldown[idx1],
        "easy_growth_streak_value": next_easy_growth_streak[idx1],
        "lagged_response_valid_value": next_lagged_response_valid[idx1],
    }
    step2_kwargs = {
        "dt_value": attempted_dts[idx2],
        "next_dt_value": next_dts[idx2],
        "recent_reject_count_value": next_recent_reject_count[idx2],
        "regrowth_cooldown_value": next_regrowth_cooldown[idx2],
        "easy_growth_streak_value": next_easy_growth_streak[idx2],
        "lagged_response_valid_value": next_lagged_response_valid[idx2],
    }

    custom_carry1, custom_carry1_dot = jax.jvp(
        lambda c: _attempt_update(c, use_custom=True, **step0_kwargs),
        (carry0,),
        (carry0_dot,),
    )
    direct_carry1, direct_carry1_dot = jax.jvp(
        lambda c: _attempt_update(c, use_custom=False, **step0_kwargs),
        (carry0,),
        (carry0_dot,),
    )
    custom_carry2, custom_carry2_dot = jax.jvp(
        lambda c: _attempt_update(c, use_custom=True, **step1_kwargs),
        (custom_carry1,),
        (custom_carry1_dot,),
    )
    direct_carry2, direct_carry2_dot = jax.jvp(
        lambda c: _attempt_update(c, use_custom=False, **step1_kwargs),
        (direct_carry1,),
        (direct_carry1_dot,),
    )
    _, custom_step3_result = jax.jvp(
        lambda c: _attempt_update(c, use_custom=True, **step2_kwargs),
        (custom_carry2,),
        (custom_carry2_dot,),
    )
    _, direct_step3_result = jax.jvp(
        lambda c: _attempt_update(c, use_custom=False, **step2_kwargs),
        (direct_carry2,),
        (direct_carry2_dot,),
    )

    def _direct_step3_with_ablation(**replacements):
        ablated_dot = dataclasses.replace(direct_carry2_dot, **replacements)
        _, tangent = jax.jvp(
            lambda c: _attempt_update(c, use_custom=False, **step2_kwargs),
            (direct_carry2,),
            (ablated_dot,),
        )
        return tangent

    unpack_flat = prepared_rollout_static.physics_context.unpack_flat

    def _field_relative_errors(ad_state, ref_state):
        ad_pressure = np.asarray(jax.device_get(ad_state.pressure), dtype=float)
        ref_pressure = np.asarray(jax.device_get(ref_state.pressure), dtype=float)
        ad_er = np.asarray(jax.device_get(ad_state.Er), dtype=float)
        ref_er = np.asarray(jax.device_get(ref_state.Er), dtype=float)
        return {
            "pressure_relative_error": float(
                np.linalg.norm(ad_pressure - ref_pressure)
                / max(float(np.linalg.norm(ref_pressure)), 1.0e-10)
            ),
            "Er_relative_error": float(
                np.linalg.norm(ad_er - ref_er)
                / max(float(np.linalg.norm(ref_er)), 1.0e-10)
            ),
        }

    def _flat_component_report(ad_arr, ref_arr):
        ad_flat = np.asarray(jax.device_get(ad_arr), dtype=float)
        ref_flat = np.asarray(jax.device_get(ref_arr), dtype=float)
        ad_state = unpack_flat(jnp.asarray(ad_flat))
        ref_state = unpack_flat(jnp.asarray(ref_flat))
        report = _field_relative_errors(ad_state, ref_state)
        report["full_relative_error"] = float(
            np.linalg.norm(ad_flat - ref_flat) / max(float(np.linalg.norm(ref_flat)), 1.0e-10)
        )
        return report

    custom_vs_direct = {
        "trial_y": _flat_component_report(custom_step3_result.y, direct_step3_result.y),
    }
    carry_after_step2_custom_vs_direct = {
        "trial_y": _flat_component_report(custom_carry2_dot.y, direct_carry2_dot.y),
    }

    trace_prefix3 = _truncate_rollout_trace_by_accepted_steps(
        baseline_rollout.trace,
        accepted_step_limit=3,
    )
    helper_custom = _sampled_adaptive_state_tangent_trajectory(
        execution_context=execution_context,
        carry0=carry0,
        carry0_dot=carry0_dot,
        trace=trace_prefix3,
        sample_every=1,
    )
    helper_direct = _sampled_fixed_dt_state_tangent_trajectory(
        execution_context=execution_context,
        carry0=carry0,
        carry0_dot=carry0_dot,
        dt_sequence=_radau_dt_sequence_from_time_list(
            _accepted_time_list_until_attempt_index(baseline_rollout.trace, idx2),
            t0=prepared_rollout_static.initial_carry.t,
            dtype=prepared_rollout_static.kernel_context.dtype,
        ),
        sample_every=1,
    )
    helper_custom_np = np.asarray(jax.device_get(helper_custom["sampled_state_tangents"]), dtype=float)
    helper_direct_np = np.asarray(jax.device_get(helper_direct["sampled_state_tangents"]), dtype=float)
    helper_custom_step3 = helper_custom_np[2]
    helper_direct_step3 = helper_direct_np[2]

    helper_consistency = {
        "custom_scan_vs_manual": {
            "trial_y": _flat_component_report(helper_custom_step3, np.asarray(jax.device_get(custom_step3_result.y), dtype=float)),
        },
        "direct_scan_vs_manual": {
            "trial_y": _flat_component_report(helper_direct_step3, np.asarray(jax.device_get(direct_step3_result.y), dtype=float)),
        },
    }

    ablation_specs = {
        "prev_stages_zeroed": {
            "prev_stages": jnp.zeros_like(direct_carry2_dot.prev_stages),
        },
        "prev_dt_zeroed": {
            "prev_dt": jnp.zeros_like(direct_carry2_dot.prev_dt),
        },
        "prev_theta_final_zeroed": {
            "prev_theta_final": jnp.zeros_like(direct_carry2_dot.prev_theta_final),
        },
        "prev_newton_iter_count_zeroed": {
            "prev_newton_iter_count": jnp.zeros(
                direct_carry2_dot.prev_newton_iter_count.shape,
                dtype=direct_carry2_dot.prev_newton_iter_count.dtype,
            ),
        },
        "lagged_response_cache_zeroed": {
            "lagged_response_cache": _zero_optional_pytree(direct_carry2_dot.lagged_response_cache),
        },
        "lagged_reference_y_zeroed": {
            "lagged_reference_y": jnp.zeros_like(direct_carry2_dot.lagged_reference_y),
        },
        "linearization_cache_zeroed": {
            "jacobian": jnp.zeros_like(direct_carry2_dot.jacobian),
            "cache_valid": jnp.zeros_like(direct_carry2_dot.cache_valid),
            "cache_dt": jnp.zeros_like(direct_carry2_dot.cache_dt),
            "cache_age": jnp.zeros_like(direct_carry2_dot.cache_age),
            "real_lu": jnp.zeros_like(direct_carry2_dot.real_lu),
            "real_piv": jnp.zeros(direct_carry2_dot.real_piv.shape, dtype=direct_carry2_dot.real_piv.dtype),
            "complex_lu": jnp.zeros_like(direct_carry2_dot.complex_lu),
            "complex_piv": jnp.zeros(direct_carry2_dot.complex_piv.shape, dtype=direct_carry2_dot.complex_piv.dtype),
        },
        "step_history_meta_zeroed": {
            "prev_error": jnp.zeros_like(direct_carry2_dot.prev_error),
            "prev_dt": jnp.zeros_like(direct_carry2_dot.prev_dt),
            "prev_theta_final": jnp.zeros_like(direct_carry2_dot.prev_theta_final),
            "prev_newton_iter_count": jnp.zeros(
                direct_carry2_dot.prev_newton_iter_count.shape,
                dtype=direct_carry2_dot.prev_newton_iter_count.dtype,
            ),
        },
    }

    carry_field_ablations = {}
    for label, replacements in ablation_specs.items():
        ablated_direct_step3 = _direct_step3_with_ablation(**replacements)
        carry_field_ablations[label] = {
            "ablated_direct_vs_direct": {
                "trial_y": _flat_component_report(ablated_direct_step3.y, direct_step3_result.y),
            },
            "custom_vs_ablated_direct": {
                "trial_y": _flat_component_report(custom_step3_result.y, ablated_direct_step3.y),
            },
        }

    return {
        "config_path": str(config_path),
        "baseline_dt_path_third_step_carry_ablation_check": True,
        "parameter_name": parameter_name,
        "baseline_value": baseline_value,
        "first_step_time": float(step_ts[idx0]),
        "second_step_time": float(step_ts[idx1]),
        "third_step_time": float(step_ts[idx2]),
        "custom_vs_direct": custom_vs_direct,
        "carry_after_step2_custom_vs_direct": carry_after_step2_custom_vs_direct,
        "helper_consistency": helper_consistency,
        "carry_field_ablations": carry_field_ablations,
        "max_relative_error": float(custom_vs_direct["trial_y"]["Er_relative_error"]),
        "passed": bool(custom_vs_direct["trial_y"]["Er_relative_error"] <= 5.0e-2),
        "rollout_path": {"baseline": baseline_diag},
    }


def build_realized_trace_sixth_step_carry_ablation_report(
    *,
    config_path: Path,
    parameter_name: str,
    device: str | None,
) -> dict[str, Any]:
    if parameter_name not in ALLOWED_PARAMETERS:
        raise ValueError(f"parameter_name must be one of {sorted(ALLOWED_PARAMETERS)}")

    config = _prepare_benchmark_config(config_path, device=device)
    runtime, baseline_state = build_runtime_context(config)
    profile_cfg = _baseline_profile_cfg(config)
    baseline_value = float(profile_cfg[parameter_name])

    _, baseline_rollout = _adaptive_rollout_objectives_for_parameter(
        jnp.asarray(baseline_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        use_realized_schedule_jvp=True,
    )
    baseline_diag = _adaptive_rollout_diagnostics(baseline_rollout)

    accepted_mask_np = np.asarray(jax.device_get(baseline_rollout.trace.accepted_mask), dtype=bool)
    accepted_attempt_indices = np.flatnonzero(accepted_mask_np)
    if accepted_attempt_indices.size < 6:
        raise ValueError("Need at least six accepted attempts for sixth-step carry ablation debug.")
    accepted_attempt_indices = [int(v) for v in accepted_attempt_indices[:6]]

    print(
        "[autodiff-gate] realized-trace-sixth-step-carry baseline summary: "
        f"attempt_count={baseline_diag['attempt_count']} "
        f"accepted_count={baseline_diag['accepted_count']} "
        f"sixth_accepted_attempt={accepted_attempt_indices[5]}",
        flush=True,
    )

    state0_static = _parameterized_initial_state(
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        geometry=runtime.geometry,
        n_species=runtime.species.number_species,
        parameter_name=parameter_name,
        parameter_value=jax.lax.stop_gradient(jnp.asarray(baseline_value)),
    )
    prepared_components_static = prepare_transport_solver_components(config, runtime, state0_static)
    solver_static = prepared_components_static["solver"]
    solve_vector_field_static = prepared_components_static["solve_vector_field"]
    prepared_rollout_static = _build_prepared_radau_accepted_rollout(
        solver=solver_static,
        state=state0_static,
        vector_field=solve_vector_field_static,
        species=runtime.species,
    )
    execution_context = _build_prepared_radau_execution_context(
        solver=solver_static,
        prepared_rollout=prepared_rollout_static,
    )

    carry0, carry0_dot = jax.jvp(
        lambda p: _adaptive_rollout_initial_carry_for_parameter(
            p,
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=parameter_name,
        ),
        (jnp.asarray(baseline_value),),
        (jnp.asarray(1.0),),
    )

    attempted_dts = np.asarray(jax.device_get(baseline_rollout.trace.attempted_dts), dtype=float)
    next_dts = np.asarray(jax.device_get(baseline_rollout.trace.next_dts), dtype=float)
    next_recent_reject_count = np.asarray(jax.device_get(baseline_rollout.trace.next_recent_reject_count))
    next_regrowth_cooldown = np.asarray(jax.device_get(baseline_rollout.trace.next_regrowth_cooldown))
    next_easy_growth_streak = np.asarray(jax.device_get(baseline_rollout.trace.next_easy_growth_streak))
    next_lagged_response_valid = np.asarray(jax.device_get(baseline_rollout.trace.next_lagged_response_valid))
    step_ts = np.asarray(jax.device_get(baseline_rollout.trace.step_ts), dtype=float)

    def _zero_optional_pytree(tree):
        return jax.tree_util.tree_map(
            lambda x: None if x is None else jnp.zeros_like(x),
            tree,
            is_leaf=lambda x: x is None,
        )

    def _attempt_update(
        carry_value,
        attempt_fn,
        *,
        dt_value,
        next_dt_value,
        recent_reject_count_value,
        regrowth_cooldown_value,
        easy_growth_streak_value,
        lagged_response_valid_value,
    ):
        carry_for_step = dataclasses.replace(carry_value, dt=jnp.asarray(dt_value, dtype=execution_context.dtype))
        attempt_result = attempt_fn(
            execution_context.kernel_context,
            execution_context.physics_context,
            _radau_carry_with_forward_only_jvp_fields(carry_for_step),
            execution_context.attempt_context,
        )
        project_flat = execution_context.physics_context.project_flat
        accepted_y = project_flat(attempt_result.trial_y) if project_flat is not None else None
        if accepted_y is None:
            accepted_y = attempt_result.trial_y
        return dataclasses.replace(
            attempt_result.carry_after_attempt,
            t=carry_value.t + jnp.asarray(dt_value, dtype=execution_context.dtype),
            y=accepted_y,
            dt=jnp.asarray(next_dt_value, dtype=execution_context.dtype),
            prev_error=jnp.maximum(
                attempt_result.err_norm,
                jnp.asarray(1.0e-12, dtype=execution_context.dtype),
            ),
            prev_stages=attempt_result.stage_history,
            prev_dt=jnp.asarray(dt_value, dtype=execution_context.dtype),
            recent_reject_count=jnp.asarray(recent_reject_count_value),
            regrowth_cooldown=jnp.asarray(regrowth_cooldown_value),
            easy_growth_streak=jnp.asarray(easy_growth_streak_value),
            lagged_response_valid=jnp.asarray(lagged_response_valid_value),
            jacobian=attempt_result.jacobian_out,
            cache_valid=attempt_result.cache_valid_out,
            cache_dt=attempt_result.cache_dt_out,
            cache_age=attempt_result.cache_age_out,
            real_lu=attempt_result.real_lu_out,
            real_piv=attempt_result.real_piv_out,
            complex_lu=attempt_result.complex_lu_out,
            complex_piv=attempt_result.complex_piv_out,
            prev_theta_final=attempt_result.theta_final,
            prev_newton_iter_count=attempt_result.newton_iter_count,
        )

    step_kwargs = []
    for idx in accepted_attempt_indices:
        step_kwargs.append(
            {
                "dt_value": attempted_dts[idx],
                "next_dt_value": next_dts[idx],
                "recent_reject_count_value": next_recent_reject_count[idx],
                "regrowth_cooldown_value": next_regrowth_cooldown[idx],
                "easy_growth_streak_value": next_easy_growth_streak[idx],
                "lagged_response_valid_value": next_lagged_response_valid[idx],
            }
        )

    custom_carry = carry0
    custom_carry_dot = carry0_dot
    direct_carry = carry0
    direct_carry_dot = carry0_dot
    carry_after_step5_custom_dot = None
    carry_after_step5_direct_dot = None
    custom_step6_result = None
    direct_step6_result = None

    for i, kwargs in enumerate(step_kwargs):
        custom_carry, custom_carry_dot = jax.jvp(
            lambda c, kwargs=kwargs: _attempt_update(c, use_custom=True, **kwargs),
            (custom_carry,),
            (custom_carry_dot,),
        )
        direct_carry, direct_carry_dot = jax.jvp(
            lambda c, kwargs=kwargs: _attempt_update(c, use_custom=False, **kwargs),
            (direct_carry,),
            (direct_carry_dot,),
        )
        if i == 4:
            carry_after_step5_custom_dot = custom_carry_dot
            carry_after_step5_direct_dot = direct_carry_dot
        if i == 5:
            custom_step6_result = custom_carry_dot
            direct_step6_result = direct_carry_dot

    def _direct_step6_with_ablation(**replacements):
        ablated_dot = dataclasses.replace(carry_after_step5_direct_dot, **replacements)
        _, tangent = jax.jvp(
            lambda c: _attempt_update(c, use_custom=False, **step_kwargs[5]),
            (direct_carry,),
            (ablated_dot,),
        )
        return tangent

    unpack_flat = prepared_rollout_static.physics_context.unpack_flat

    def _field_relative_errors(ad_state, ref_state):
        ad_pressure = np.asarray(jax.device_get(ad_state.pressure), dtype=float)
        ref_pressure = np.asarray(jax.device_get(ref_state.pressure), dtype=float)
        ad_er = np.asarray(jax.device_get(ad_state.Er), dtype=float)
        ref_er = np.asarray(jax.device_get(ref_state.Er), dtype=float)
        return {
            "pressure_relative_error": float(
                np.linalg.norm(ad_pressure - ref_pressure)
                / max(float(np.linalg.norm(ref_pressure)), 1.0e-10)
            ),
            "Er_relative_error": float(
                np.linalg.norm(ad_er - ref_er)
                / max(float(np.linalg.norm(ref_er)), 1.0e-10)
            ),
        }

    def _flat_component_report(ad_arr, ref_arr):
        ad_flat = np.asarray(jax.device_get(ad_arr), dtype=float)
        ref_flat = np.asarray(jax.device_get(ref_arr), dtype=float)
        ad_state = unpack_flat(jnp.asarray(ad_flat))
        ref_state = unpack_flat(jnp.asarray(ref_flat))
        report = _field_relative_errors(ad_state, ref_state)
        report["full_relative_error"] = float(
            np.linalg.norm(ad_flat - ref_flat) / max(float(np.linalg.norm(ref_flat)), 1.0e-10)
        )
        return report

    custom_vs_direct = {
        "trial_y": _flat_component_report(custom_step6_result.y, direct_step6_result.y),
    }
    carry_after_step5_custom_vs_direct = {
        "trial_y": _flat_component_report(carry_after_step5_custom_dot.y, carry_after_step5_direct_dot.y),
    }

    trace_prefix6 = _truncate_rollout_trace_by_accepted_steps(
        baseline_rollout.trace,
        accepted_step_limit=6,
    )
    helper_custom = _sampled_realized_trace_state_tangent_trajectory(
        execution_context=execution_context,
        carry0=carry0,
        carry0_dot=carry0_dot,
        trace=trace_prefix6,
        sample_every=1,
        use_custom=True,
    )
    helper_direct = _sampled_realized_trace_state_tangent_trajectory(
        execution_context=execution_context,
        carry0=carry0,
        carry0_dot=carry0_dot,
        trace=trace_prefix6,
        sample_every=1,
        use_custom=False,
    )
    helper_custom_np = np.asarray(jax.device_get(helper_custom["sampled_state_tangents"]), dtype=float)
    helper_direct_np = np.asarray(jax.device_get(helper_direct["sampled_state_tangents"]), dtype=float)
    helper_custom_step6 = helper_custom_np[5]
    helper_direct_step6 = helper_direct_np[5]

    helper_consistency = {
        "custom_scan_vs_manual": {
            "trial_y": _flat_component_report(helper_custom_step6, np.asarray(jax.device_get(custom_step6_result.y), dtype=float)),
        },
        "direct_scan_vs_manual": {
            "trial_y": _flat_component_report(helper_direct_step6, np.asarray(jax.device_get(direct_step6_result.y), dtype=float)),
        },
    }

    ablation_specs = {
        "prev_stages_zeroed": {
            "prev_stages": jnp.zeros_like(carry_after_step5_direct_dot.prev_stages),
        },
        "prev_dt_zeroed": {
            "prev_dt": jnp.zeros_like(carry_after_step5_direct_dot.prev_dt),
        },
        "prev_theta_final_zeroed": {
            "prev_theta_final": jnp.zeros_like(carry_after_step5_direct_dot.prev_theta_final),
        },
        "prev_newton_iter_count_zeroed": {
            "prev_newton_iter_count": jnp.zeros(
                carry_after_step5_direct_dot.prev_newton_iter_count.shape,
                dtype=carry_after_step5_direct_dot.prev_newton_iter_count.dtype,
            ),
        },
        "lagged_response_cache_zeroed": {
            "lagged_response_cache": _zero_optional_pytree(carry_after_step5_direct_dot.lagged_response_cache),
        },
        "lagged_reference_y_zeroed": {
            "lagged_reference_y": jnp.zeros_like(carry_after_step5_direct_dot.lagged_reference_y),
        },
        "linearization_cache_zeroed": {
            "jacobian": jnp.zeros_like(carry_after_step5_direct_dot.jacobian),
            "cache_valid": jnp.zeros_like(carry_after_step5_direct_dot.cache_valid),
            "cache_dt": jnp.zeros_like(carry_after_step5_direct_dot.cache_dt),
            "cache_age": jnp.zeros_like(carry_after_step5_direct_dot.cache_age),
            "real_lu": jnp.zeros_like(carry_after_step5_direct_dot.real_lu),
            "real_piv": jnp.zeros(carry_after_step5_direct_dot.real_piv.shape, dtype=carry_after_step5_direct_dot.real_piv.dtype),
            "complex_lu": jnp.zeros_like(carry_after_step5_direct_dot.complex_lu),
            "complex_piv": jnp.zeros(carry_after_step5_direct_dot.complex_piv.shape, dtype=carry_after_step5_direct_dot.complex_piv.dtype),
        },
        "step_history_meta_zeroed": {
            "prev_error": jnp.zeros_like(carry_after_step5_direct_dot.prev_error),
            "prev_dt": jnp.zeros_like(carry_after_step5_direct_dot.prev_dt),
            "prev_theta_final": jnp.zeros_like(carry_after_step5_direct_dot.prev_theta_final),
            "prev_newton_iter_count": jnp.zeros(
                carry_after_step5_direct_dot.prev_newton_iter_count.shape,
                dtype=carry_after_step5_direct_dot.prev_newton_iter_count.dtype,
            ),
        },
    }

    carry_field_ablations = {}
    for label, replacements in ablation_specs.items():
        ablated_direct_step6 = _direct_step6_with_ablation(**replacements)
        carry_field_ablations[label] = {
            "ablated_direct_vs_direct": {
                "trial_y": _flat_component_report(ablated_direct_step6.y, direct_step6_result.y),
            },
            "custom_vs_ablated_direct": {
                "trial_y": _flat_component_report(custom_step6_result.y, ablated_direct_step6.y),
            },
        }

    return {
        "config_path": str(config_path),
        "realized_trace_sixth_step_carry_ablation_check": True,
        "parameter_name": parameter_name,
        "baseline_value": baseline_value,
        "sixth_step_time": float(step_ts[accepted_attempt_indices[5]]),
        "custom_vs_direct": custom_vs_direct,
        "carry_after_step5_custom_vs_direct": carry_after_step5_custom_vs_direct,
        "helper_consistency": helper_consistency,
        "carry_field_ablations": carry_field_ablations,
        "max_relative_error": float(custom_vs_direct["trial_y"]["Er_relative_error"]),
        "passed": bool(custom_vs_direct["trial_y"]["Er_relative_error"] <= 5.0e-2),
        "rollout_path": {"baseline": baseline_diag},
    }


def build_realized_trace_checkpoint_compare_report(
    *,
    config_path: Path,
    parameter_name: str,
    device: str | None,
    checkpoint_index: int,
) -> dict[str, Any]:
    if parameter_name not in ALLOWED_PARAMETERS:
        raise ValueError(f"parameter_name must be one of {sorted(ALLOWED_PARAMETERS)}")

    checkpoint_index = int(checkpoint_index)
    if checkpoint_index <= 0:
        raise ValueError("checkpoint_index must be positive.")

    config = _prepare_benchmark_config(config_path, device=device)
    runtime, baseline_state = build_runtime_context(config)
    profile_cfg = _baseline_profile_cfg(config)
    baseline_value = float(profile_cfg[parameter_name])

    _, baseline_rollout = _adaptive_rollout_objectives_for_parameter(
        jnp.asarray(baseline_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        use_realized_schedule_jvp=True,
    )
    baseline_diag = _adaptive_rollout_diagnostics(baseline_rollout)

    accepted_mask_np = np.asarray(jax.device_get(baseline_rollout.trace.accepted_mask), dtype=bool)
    accepted_attempt_indices = np.flatnonzero(accepted_mask_np)
    if accepted_attempt_indices.size < checkpoint_index:
        raise ValueError(
            f"Need at least {checkpoint_index} accepted attempts for realized-trace checkpoint compare; "
            f"found {accepted_attempt_indices.size}."
        )
    accepted_attempt_indices = [int(v) for v in accepted_attempt_indices[:checkpoint_index]]

    print(
        "[autodiff-gate] realized-trace-checkpoint baseline summary: "
        f"attempt_count={baseline_diag['attempt_count']} "
        f"accepted_count={baseline_diag['accepted_count']} "
        f"checkpoint_index={checkpoint_index} "
        f"checkpoint_attempt={accepted_attempt_indices[-1]}",
        flush=True,
    )

    state0_static = _parameterized_initial_state(
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        geometry=runtime.geometry,
        n_species=runtime.species.number_species,
        parameter_name=parameter_name,
        parameter_value=jax.lax.stop_gradient(jnp.asarray(baseline_value)),
    )
    prepared_components_static = prepare_transport_solver_components(config, runtime, state0_static)
    solver_static = prepared_components_static["solver"]
    solve_vector_field_static = prepared_components_static["solve_vector_field"]
    prepared_rollout_static = _build_prepared_radau_accepted_rollout(
        solver=solver_static,
        state=state0_static,
        vector_field=solve_vector_field_static,
        species=runtime.species,
    )
    execution_context = _build_prepared_radau_execution_context(
        solver=solver_static,
        prepared_rollout=prepared_rollout_static,
    )

    carry0, carry0_dot = jax.jvp(
        lambda p: _adaptive_rollout_initial_carry_for_parameter(
            p,
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=parameter_name,
        ),
        (jnp.asarray(baseline_value),),
        (jnp.asarray(1.0),),
    )

    attempted_dts = np.asarray(jax.device_get(baseline_rollout.trace.attempted_dts), dtype=float)
    next_dts = np.asarray(jax.device_get(baseline_rollout.trace.next_dts), dtype=float)
    next_recent_reject_count = np.asarray(jax.device_get(baseline_rollout.trace.next_recent_reject_count))
    next_regrowth_cooldown = np.asarray(jax.device_get(baseline_rollout.trace.next_regrowth_cooldown))
    next_easy_growth_streak = np.asarray(jax.device_get(baseline_rollout.trace.next_easy_growth_streak))
    next_lagged_response_valid = np.asarray(jax.device_get(baseline_rollout.trace.next_lagged_response_valid))
    step_ts = np.asarray(jax.device_get(baseline_rollout.trace.step_ts), dtype=float)

    def _attempt_update(
        carry_value,
        *,
        attempt_fn,
        dt_value,
        next_dt_value,
        recent_reject_count_value,
        regrowth_cooldown_value,
        easy_growth_streak_value,
        lagged_response_valid_value,
    ):
        carry_for_step = dataclasses.replace(carry_value, dt=jnp.asarray(dt_value, dtype=execution_context.dtype))
        attempt_result = attempt_fn(
            execution_context.kernel_context,
            execution_context.physics_context,
            _radau_carry_with_forward_only_jvp_fields(carry_for_step),
            execution_context.attempt_context,
        )
        project_flat = execution_context.physics_context.project_flat
        accepted_y = project_flat(attempt_result.trial_y) if project_flat is not None else None
        if accepted_y is None:
            accepted_y = attempt_result.trial_y
        return dataclasses.replace(
            attempt_result.carry_after_attempt,
            t=carry_value.t + jnp.asarray(dt_value, dtype=execution_context.dtype),
            y=accepted_y,
            dt=jnp.asarray(next_dt_value, dtype=execution_context.dtype),
            prev_error=jnp.maximum(
                attempt_result.err_norm,
                jnp.asarray(1.0e-12, dtype=execution_context.dtype),
            ),
            prev_stages=attempt_result.stage_history,
            prev_dt=jnp.asarray(dt_value, dtype=execution_context.dtype),
            recent_reject_count=jnp.asarray(recent_reject_count_value),
            regrowth_cooldown=jnp.asarray(regrowth_cooldown_value),
            easy_growth_streak=jnp.asarray(easy_growth_streak_value),
            lagged_response_valid=jnp.asarray(lagged_response_valid_value),
            jacobian=attempt_result.jacobian_out,
            cache_valid=attempt_result.cache_valid_out,
            cache_dt=attempt_result.cache_dt_out,
            cache_age=attempt_result.cache_age_out,
            real_lu=attempt_result.real_lu_out,
            real_piv=attempt_result.real_piv_out,
            complex_lu=attempt_result.complex_lu_out,
            complex_piv=attempt_result.complex_piv_out,
            prev_theta_final=attempt_result.theta_final,
            prev_newton_iter_count=attempt_result.newton_iter_count,
        )

    custom_carry = carry0
    custom_carry_dot = carry0_dot
    direct_carry = carry0
    direct_carry_dot = carry0_dot

    compiled_custom_attempt_update = jax.jit(
        lambda carry_value, carry_dot_value, dt_value, next_dt_value, recent_reject_count_value, regrowth_cooldown_value, easy_growth_streak_value, lagged_response_valid_value: jax.jvp(
            lambda c: _attempt_update(
                c,
                attempt_fn=_execute_radau_accepted_step_attempt_autodiff,
                dt_value=dt_value,
                next_dt_value=next_dt_value,
                recent_reject_count_value=recent_reject_count_value,
                regrowth_cooldown_value=regrowth_cooldown_value,
                easy_growth_streak_value=easy_growth_streak_value,
                lagged_response_valid_value=lagged_response_valid_value,
            ),
            (carry_value,),
            (carry_dot_value,),
        )
    )
    compiled_direct_attempt_update = jax.jit(
        lambda carry_value, carry_dot_value, dt_value, next_dt_value, recent_reject_count_value, regrowth_cooldown_value, easy_growth_streak_value, lagged_response_valid_value: jax.jvp(
            lambda c: _attempt_update(
                c,
                attempt_fn=_execute_radau_accepted_step_attempt,
                dt_value=dt_value,
                next_dt_value=next_dt_value,
                recent_reject_count_value=recent_reject_count_value,
                regrowth_cooldown_value=regrowth_cooldown_value,
                easy_growth_streak_value=easy_growth_streak_value,
                lagged_response_valid_value=lagged_response_valid_value,
            ),
            (carry_value,),
            (carry_dot_value,),
        )
    )

    for idx in accepted_attempt_indices:
        dt_value = jnp.asarray(attempted_dts[idx], dtype=execution_context.dtype)
        next_dt_value = jnp.asarray(next_dts[idx], dtype=execution_context.dtype)
        recent_reject_count_value = jnp.asarray(next_recent_reject_count[idx])
        regrowth_cooldown_value = jnp.asarray(next_regrowth_cooldown[idx])
        easy_growth_streak_value = jnp.asarray(next_easy_growth_streak[idx])
        lagged_response_valid_value = jnp.asarray(next_lagged_response_valid[idx])
        custom_carry, custom_carry_dot = compiled_custom_attempt_update(
            custom_carry,
            custom_carry_dot,
            dt_value,
            next_dt_value,
            recent_reject_count_value,
            regrowth_cooldown_value,
            easy_growth_streak_value,
            lagged_response_valid_value,
        )
        direct_carry, direct_carry_dot = compiled_direct_attempt_update(
            direct_carry,
            direct_carry_dot,
            dt_value,
            next_dt_value,
            recent_reject_count_value,
            regrowth_cooldown_value,
            easy_growth_streak_value,
            lagged_response_valid_value,
        )

    unpack_flat = prepared_rollout_static.physics_context.unpack_flat

    def _field_relative_errors(ad_state, ref_state):
        ad_pressure = np.asarray(jax.device_get(ad_state.pressure), dtype=float)
        ref_pressure = np.asarray(jax.device_get(ref_state.pressure), dtype=float)
        ad_er = np.asarray(jax.device_get(ad_state.Er), dtype=float)
        ref_er = np.asarray(jax.device_get(ref_state.Er), dtype=float)
        return {
            "pressure_relative_error": float(
                np.linalg.norm(ad_pressure - ref_pressure)
                / max(float(np.linalg.norm(ref_pressure)), 1.0e-10)
            ),
            "Er_relative_error": float(
                np.linalg.norm(ad_er - ref_er)
                / max(float(np.linalg.norm(ref_er)), 1.0e-10)
            ),
        }

    def _flat_component_report(ad_arr, ref_arr):
        ad_flat = np.asarray(jax.device_get(ad_arr), dtype=float)
        ref_flat = np.asarray(jax.device_get(ref_arr), dtype=float)
        ad_state = unpack_flat(jnp.asarray(ad_flat))
        ref_state = unpack_flat(jnp.asarray(ref_flat))
        report = _field_relative_errors(ad_state, ref_state)
        report["full_relative_error"] = float(
            np.linalg.norm(ad_flat - ref_flat) / max(float(np.linalg.norm(ref_flat)), 1.0e-10)
        )
        return report

    comparison = {
        "trial_y": _flat_component_report(custom_carry_dot.y, direct_carry_dot.y),
    }

    return {
        "config_path": str(config_path),
        "realized_trace_checkpoint_compare_check": True,
        "parameter_name": parameter_name,
        "baseline_value": baseline_value,
        "checkpoint_index": int(checkpoint_index),
        "checkpoint_time": float(step_ts[accepted_attempt_indices[-1]]),
        "comparison": comparison,
        "max_relative_error": float(comparison["trial_y"]["Er_relative_error"]),
        "passed": bool(comparison["trial_y"]["Er_relative_error"] <= 5.0e-2),
        "rollout_path": {"baseline": baseline_diag},
    }


def build_realized_trace_checkpoint_frozen_fd_report(
    *,
    config_path: Path,
    parameter_name: str,
    rel_fd_step: float,
    abs_fd_step: float,
    device: str | None,
    checkpoint_index: int,
    replay_mode: str = "attempt",
    include_direct_ad: bool = True,
    compute_five_point: bool = False,
    ntx_exact_derivative_mode: str | None = None,
) -> dict[str, Any]:
    if parameter_name not in ALLOWED_PARAMETERS:
        raise ValueError(f"parameter_name must be one of {sorted(ALLOWED_PARAMETERS)}")

    checkpoint_index = int(checkpoint_index)
    if checkpoint_index <= 0:
        raise ValueError("checkpoint_index must be positive.")

    config = _prepare_benchmark_config(
        config_path,
        device=device,
        ntx_exact_derivative_mode=ntx_exact_derivative_mode,
    )
    runtime, baseline_state = build_runtime_context(config)
    profile_cfg = _baseline_profile_cfg(config)
    baseline_value = float(profile_cfg[parameter_name])
    fd_step = _fd_step(baseline_value, rel_step=rel_fd_step, abs_step=abs_fd_step)
    minus_value = baseline_value - fd_step
    plus_value = baseline_value + fd_step
    minus2_value = baseline_value - 2.0 * fd_step
    plus2_value = baseline_value + 2.0 * fd_step

    baseline_final_state, baseline_rollout = _adaptive_rollout_final_state_for_parameter(
        jnp.asarray(baseline_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        use_realized_schedule_jvp=False,
        accepted_step_limit_override=checkpoint_index,
    )
    del baseline_final_state
    baseline_diag = _adaptive_rollout_diagnostics(baseline_rollout)
    accepted_mask_np = np.asarray(jax.device_get(baseline_rollout.trace.accepted_mask), dtype=bool)
    accepted_attempt_indices = np.flatnonzero(accepted_mask_np)
    if accepted_attempt_indices.size < checkpoint_index:
        raise ValueError(
            f"Need at least {checkpoint_index} accepted attempts for realized-trace checkpoint frozen-FD compare; "
            f"found {accepted_attempt_indices.size}."
        )
    accepted_attempt_indices = [int(v) for v in accepted_attempt_indices[:checkpoint_index]]
    replay_trace = _truncate_rollout_trace_by_accepted_steps(
        baseline_rollout.trace,
        accepted_step_limit=checkpoint_index,
    )

    print(
        "[autodiff-gate] realized-trace-checkpoint-frozen-fd baseline summary: "
        f"attempt_count={baseline_diag['attempt_count']} "
        f"accepted_count={baseline_diag['accepted_count']} "
        f"checkpoint_index={checkpoint_index} "
        f"checkpoint_attempt={accepted_attempt_indices[-1]}",
        flush=True,
    )

    state0_static = _parameterized_initial_state(
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        geometry=runtime.geometry,
        n_species=runtime.species.number_species,
        parameter_name=parameter_name,
        parameter_value=jax.lax.stop_gradient(jnp.asarray(baseline_value)),
    )
    prepared_components_static = prepare_transport_solver_components(config, runtime, state0_static)
    solver_static = prepared_components_static["solver"]
    solve_vector_field_static = prepared_components_static["solve_vector_field"]
    prepared_rollout_static = _build_prepared_radau_accepted_rollout(
        solver=solver_static,
        state=state0_static,
        vector_field=solve_vector_field_static,
        species=runtime.species,
    )
    execution_context = _build_prepared_radau_execution_context(
        solver=solver_static,
        prepared_rollout=prepared_rollout_static,
    )

    carry0, carry0_dot = jax.jvp(
        lambda p: _adaptive_rollout_initial_carry_for_parameter(
            p,
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=parameter_name,
        ),
        (jnp.asarray(baseline_value),),
        (jnp.asarray(1.0),),
    )

    attempted_dts = np.asarray(jax.device_get(baseline_rollout.trace.attempted_dts), dtype=float)
    next_dts = np.asarray(jax.device_get(baseline_rollout.trace.next_dts), dtype=float)
    next_recent_reject_count = np.asarray(jax.device_get(baseline_rollout.trace.next_recent_reject_count))
    next_regrowth_cooldown = np.asarray(jax.device_get(baseline_rollout.trace.next_regrowth_cooldown))
    next_easy_growth_streak = np.asarray(jax.device_get(baseline_rollout.trace.next_easy_growth_streak))
    next_lagged_response_valid = np.asarray(jax.device_get(baseline_rollout.trace.next_lagged_response_valid))
    step_ts = np.asarray(jax.device_get(baseline_rollout.trace.step_ts), dtype=float)
    unpack_flat = prepared_rollout_static.physics_context.unpack_flat
    _flat_state0, _unpack_flat_tmp, _unpack_packed_tmp, pack_state, _project_flat_tmp = _make_solver_state_transform(
        baseline_state,
        runtime.species,
    )

    def _attempt_update(
        carry_value,
        attempt_fn,
        *,
        dt_value,
        next_dt_value,
        recent_reject_count_value,
        regrowth_cooldown_value,
        easy_growth_streak_value,
        lagged_response_valid_value,
    ):
        carry_for_step = dataclasses.replace(carry_value, dt=jnp.asarray(dt_value, dtype=execution_context.dtype))
        attempt_result = attempt_fn(
            execution_context.kernel_context,
            execution_context.physics_context,
            _radau_carry_with_forward_only_jvp_fields(carry_for_step),
            execution_context.attempt_context,
        )
        project_flat = execution_context.physics_context.project_flat
        accepted_y = project_flat(attempt_result.trial_y) if project_flat is not None else None
        if accepted_y is None:
            accepted_y = attempt_result.trial_y
        return dataclasses.replace(
            attempt_result.carry_after_attempt,
            t=carry_value.t + jnp.asarray(dt_value, dtype=execution_context.dtype),
            y=accepted_y,
            dt=jnp.asarray(next_dt_value, dtype=execution_context.dtype),
            prev_error=jnp.maximum(
                attempt_result.err_norm,
                jnp.asarray(1.0e-12, dtype=execution_context.dtype),
            ),
            prev_stages=attempt_result.stage_history,
            prev_dt=jnp.asarray(dt_value, dtype=execution_context.dtype),
            recent_reject_count=jnp.asarray(recent_reject_count_value),
            regrowth_cooldown=jnp.asarray(regrowth_cooldown_value),
            easy_growth_streak=jnp.asarray(easy_growth_streak_value),
            lagged_response_valid=jnp.asarray(lagged_response_valid_value),
            jacobian=attempt_result.jacobian_out,
            cache_valid=attempt_result.cache_valid_out,
            cache_dt=attempt_result.cache_dt_out,
            cache_age=attempt_result.cache_age_out,
            real_lu=attempt_result.real_lu_out,
            real_piv=attempt_result.real_piv_out,
            complex_lu=attempt_result.complex_lu_out,
            complex_piv=attempt_result.complex_piv_out,
            prev_theta_final=attempt_result.theta_final,
            prev_newton_iter_count=attempt_result.newton_iter_count,
        )

    compiled_custom_attempt_update = jax.jit(
        lambda carry_value, carry_dot_value, dt_value, next_dt_value, recent_reject_count_value, regrowth_cooldown_value, easy_growth_streak_value, lagged_response_valid_value: jax.jvp(
            lambda c: _attempt_update(
                c,
                _execute_radau_accepted_step_attempt_autodiff,
                dt_value=dt_value,
                next_dt_value=next_dt_value,
                recent_reject_count_value=recent_reject_count_value,
                regrowth_cooldown_value=regrowth_cooldown_value,
                easy_growth_streak_value=easy_growth_streak_value,
                lagged_response_valid_value=lagged_response_valid_value,
            ),
            (carry_value,),
            (carry_dot_value,),
        )
    )
    compiled_direct_attempt_update = jax.jit(
        lambda carry_value, carry_dot_value, dt_value, next_dt_value, recent_reject_count_value, regrowth_cooldown_value, easy_growth_streak_value, lagged_response_valid_value: jax.jvp(
            lambda c: _attempt_update(
                c,
                _execute_radau_accepted_step_attempt,
                dt_value=dt_value,
                next_dt_value=next_dt_value,
                recent_reject_count_value=recent_reject_count_value,
                regrowth_cooldown_value=regrowth_cooldown_value,
                easy_growth_streak_value=easy_growth_streak_value,
                lagged_response_valid_value=lagged_response_valid_value,
            ),
            (carry_value,),
            (carry_dot_value,),
        )
    )

    custom_carry = carry0
    custom_carry_dot = carry0_dot
    direct_carry = carry0
    direct_carry_dot = carry0_dot
    for idx in accepted_attempt_indices:
        dt_value = jnp.asarray(attempted_dts[idx], dtype=execution_context.dtype)
        next_dt_value = jnp.asarray(next_dts[idx], dtype=execution_context.dtype)
        recent_reject_count_value = jnp.asarray(next_recent_reject_count[idx])
        regrowth_cooldown_value = jnp.asarray(next_regrowth_cooldown[idx])
        easy_growth_streak_value = jnp.asarray(next_easy_growth_streak[idx])
        lagged_response_valid_value = jnp.asarray(next_lagged_response_valid[idx])
        custom_carry, custom_carry_dot = compiled_custom_attempt_update(
            custom_carry,
            custom_carry_dot,
            dt_value,
            next_dt_value,
            recent_reject_count_value,
            regrowth_cooldown_value,
            easy_growth_streak_value,
            lagged_response_valid_value,
        )
        if include_direct_ad:
            direct_carry, direct_carry_dot = compiled_direct_attempt_update(
                direct_carry,
                direct_carry_dot,
                dt_value,
                next_dt_value,
                recent_reject_count_value,
                regrowth_cooldown_value,
                easy_growth_streak_value,
                lagged_response_valid_value,
            )

    def _objective_from_flat_y(flat_y):
        return _objective_vector(unpack_flat(flat_y), runtime)

    _, objective_ad = jax.jvp(_objective_from_flat_y, (custom_carry.y,), (custom_carry_dot.y,))
    grad_ad_np = np.asarray(jax.device_get(objective_ad), dtype=float)
    if include_direct_ad:
        _, objective_direct = jax.jvp(_objective_from_flat_y, (direct_carry.y,), (direct_carry_dot.y,))
        grad_direct_np = np.asarray(jax.device_get(objective_direct), dtype=float)
    else:
        grad_direct_np = None

    def _evaluate_frozen_fd(value: float):
        return _adaptive_rollout_objectives_for_parameter_on_frozen_trace(
            jnp.asarray(value),
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=parameter_name,
            frozen_trace=replay_trace,
            replay_mode=replay_mode,
        )

    objectives_minus, minus_replay = _evaluate_frozen_fd(minus_value)
    objectives_plus, plus_replay = _evaluate_frozen_fd(plus_value)
    gradient_fd = (objectives_plus - objectives_minus) / (2.0 * fd_step)
    grad_fd_np = np.asarray(jax.device_get(gradient_fd), dtype=float)
    abs_err = np.abs(grad_ad_np - grad_fd_np)
    rel_err = abs_err / np.maximum(np.abs(grad_fd_np), 1.0e-10)
    if include_direct_ad:
        direct_abs_err = np.abs(grad_direct_np - grad_fd_np)
        direct_rel_err = direct_abs_err / np.maximum(np.abs(grad_fd_np), 1.0e-10)
        custom_vs_direct_abs_err = np.abs(grad_ad_np - grad_direct_np)
        custom_vs_direct_rel_err = custom_vs_direct_abs_err / np.maximum(np.abs(grad_direct_np), 1.0e-10)
    else:
        direct_abs_err = None
        direct_rel_err = None
        custom_vs_direct_abs_err = None
        custom_vs_direct_rel_err = None

    fd_minus_flat = np.asarray(jax.device_get(pack_state(minus_replay["final_state"])), dtype=float)
    fd_plus_flat = np.asarray(jax.device_get(pack_state(plus_replay["final_state"])), dtype=float)
    state_fd_np = (fd_plus_flat - fd_minus_flat) / (2.0 * fd_step)
    state_custom_np = np.asarray(jax.device_get(custom_carry_dot.y), dtype=float)
    state_direct_np = np.asarray(jax.device_get(direct_carry_dot.y), dtype=float) if include_direct_ad else None

    def _flat_component_report(ad_arr, ref_arr):
        ad_flat = np.asarray(jax.device_get(ad_arr), dtype=float)
        ref_flat = np.asarray(jax.device_get(ref_arr), dtype=float)
        ad_state = unpack_flat(jnp.asarray(ad_flat))
        ref_state = unpack_flat(jnp.asarray(ref_flat))
        ad_pressure = np.asarray(jax.device_get(ad_state.pressure), dtype=float)
        ref_pressure = np.asarray(jax.device_get(ref_state.pressure), dtype=float)
        ad_er = np.asarray(jax.device_get(ad_state.Er), dtype=float)
        ref_er = np.asarray(jax.device_get(ref_state.Er), dtype=float)
        return {
            "full_relative_error": float(
                np.linalg.norm(ad_flat - ref_flat) / max(float(np.linalg.norm(ref_flat)), 1.0e-10)
            ),
            "pressure_relative_error": float(
                np.linalg.norm(ad_pressure - ref_pressure)
                / max(float(np.linalg.norm(ref_pressure)), 1.0e-10)
            ),
            "Er_relative_error": float(
                np.linalg.norm(ad_er - ref_er)
                / max(float(np.linalg.norm(ref_er)), 1.0e-10)
            ),
        }

    state_comparison = {
        "custom_vs_fd": _flat_component_report(state_custom_np, state_fd_np),
        "direct_vs_fd": _flat_component_report(state_direct_np, state_fd_np) if include_direct_ad else None,
        "custom_vs_direct": _flat_component_report(state_custom_np, state_direct_np) if include_direct_ad else None,
    }

    grad_fd_five_point_np = None
    grad_five_point_abs_err = None
    grad_five_point_rel_err = None
    state_fd_five_point_np = None
    state_comparison_five_point = None
    state_comparison_center_vs_five_point = None
    minus2_replay_finite = None
    plus2_replay_finite = None
    if compute_five_point:
        objectives_minus2, minus2_replay = _evaluate_frozen_fd(minus2_value)
        objectives_plus2, plus2_replay = _evaluate_frozen_fd(plus2_value)
        gradient_fd_five_point = (
            -objectives_plus2 + 8.0 * objectives_plus - 8.0 * objectives_minus + objectives_minus2
        ) / (12.0 * fd_step)
        grad_fd_five_point_np = np.asarray(jax.device_get(gradient_fd_five_point), dtype=float)
        grad_five_point_abs_err = np.abs(grad_ad_np - grad_fd_five_point_np)
        grad_five_point_rel_err = grad_five_point_abs_err / np.maximum(np.abs(grad_fd_five_point_np), 1.0e-10)

        fd_minus2_flat = np.asarray(jax.device_get(pack_state(minus2_replay["final_state"])), dtype=float)
        fd_plus2_flat = np.asarray(jax.device_get(pack_state(plus2_replay["final_state"])), dtype=float)
        state_fd_five_point_np = (
            -fd_plus2_flat + 8.0 * fd_plus_flat - 8.0 * fd_minus_flat + fd_minus2_flat
        ) / (12.0 * fd_step)
        state_comparison_five_point = {
            "custom_vs_fd_five_point": _flat_component_report(state_custom_np, state_fd_five_point_np),
            "center_vs_fd_five_point": _flat_component_report(state_fd_np, state_fd_five_point_np),
        }
        if include_direct_ad:
            state_comparison_five_point["direct_vs_fd_five_point"] = _flat_component_report(
                state_direct_np,
                state_fd_five_point_np,
            )

        state_comparison_center_vs_five_point = _flat_component_report(state_fd_np, state_fd_five_point_np)
        minus2_replay_finite = _tree_all_finite(minus2_replay["final_state"])
        plus2_replay_finite = _tree_all_finite(plus2_replay["final_state"])

    return {
        "config_path": str(config_path),
        "realized_trace_checkpoint_frozen_fd_check": True,
        "fd_stencil_check": bool(compute_five_point),
        "ntx_exact_derivative_mode": None if ntx_exact_derivative_mode is None else str(ntx_exact_derivative_mode),
        "parameter_name": parameter_name,
        "baseline_value": baseline_value,
        "fd_step": float(fd_step),
        "checkpoint_index": int(checkpoint_index),
        "checkpoint_time": float(step_ts[accepted_attempt_indices[-1]]),
        "gradient_autodiff": grad_ad_np.tolist(),
        "gradient_direct": None if grad_direct_np is None else grad_direct_np.tolist(),
        "gradient_fd": grad_fd_np.tolist(),
        "gradient_fd_five_point": None if grad_fd_five_point_np is None else grad_fd_five_point_np.tolist(),
        "gradient_absolute_error": abs_err.tolist(),
        "gradient_relative_error": rel_err.tolist(),
        "gradient_five_point_absolute_error": None if grad_five_point_abs_err is None else grad_five_point_abs_err.tolist(),
        "gradient_five_point_relative_error": None if grad_five_point_rel_err is None else grad_five_point_rel_err.tolist(),
        "gradient_direct_absolute_error": None if direct_abs_err is None else direct_abs_err.tolist(),
        "gradient_direct_relative_error": None if direct_rel_err is None else direct_rel_err.tolist(),
        "gradient_custom_vs_direct_absolute_error": None if custom_vs_direct_abs_err is None else custom_vs_direct_abs_err.tolist(),
        "gradient_custom_vs_direct_relative_error": None if custom_vs_direct_rel_err is None else custom_vs_direct_rel_err.tolist(),
        "state_tangent_comparison": state_comparison,
        "state_tangent_comparison_five_point": state_comparison_five_point,
        "state_tangent_comparison_center_vs_five_point": state_comparison_center_vs_five_point,
        "max_relative_error": float(np.max(rel_err)),
        "passed": bool(
            np.all(np.isfinite(rel_err))
            and np.max(rel_err) <= 5.0e-2
            and (
                grad_five_point_rel_err is None
                or (np.all(np.isfinite(grad_five_point_rel_err)) and np.max(grad_five_point_rel_err) <= 5.0e-2)
            )
        ),
        "objective_labels": OBJECTIVE_LABELS,
        "replay_mode": str(replay_mode),
        "rollout_path": {
            "baseline": baseline_diag,
            "fd_minus_state_finite": _tree_all_finite(minus_replay["final_state"]),
            "fd_plus_state_finite": _tree_all_finite(plus_replay["final_state"]),
            "fd_minus2_state_finite": minus2_replay_finite,
            "fd_plus2_state_finite": plus2_replay_finite,
        },
    }


def build_realized_trace_checkpoint_interpolated_fd_report(
    *,
    config_path: Path,
    parameter_name: str,
    rel_fd_step: float,
    abs_fd_step: float,
    device: str | None,
    checkpoint_index: int,
) -> dict[str, Any]:
    if parameter_name not in ALLOWED_PARAMETERS:
        raise ValueError(f"parameter_name must be one of {sorted(ALLOWED_PARAMETERS)}")

    checkpoint_index = int(checkpoint_index)
    if checkpoint_index <= 0:
        raise ValueError("checkpoint_index must be positive.")

    config = _prepare_benchmark_config(config_path, device=device)
    runtime, baseline_state = build_runtime_context(config)
    profile_cfg = _baseline_profile_cfg(config)
    baseline_value = float(profile_cfg[parameter_name])
    fd_step = _fd_step(baseline_value, rel_step=rel_fd_step, abs_step=abs_fd_step)
    minus_value = baseline_value - fd_step
    plus_value = baseline_value + fd_step

    _, baseline_rollout = _adaptive_rollout_final_state_for_parameter(
        jnp.asarray(baseline_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        use_realized_schedule_jvp=False,
        accepted_step_limit_override=checkpoint_index,
    )
    baseline_diag = _adaptive_rollout_diagnostics(baseline_rollout)
    accepted_mask_np = np.asarray(jax.device_get(baseline_rollout.trace.accepted_mask), dtype=bool)
    accepted_attempt_indices = np.flatnonzero(accepted_mask_np)
    if accepted_attempt_indices.size < checkpoint_index:
        raise ValueError(
            f"Need at least {checkpoint_index} accepted attempts for realized-trace checkpoint interpolated-FD compare; "
            f"found {accepted_attempt_indices.size}."
        )
    accepted_attempt_indices = [int(v) for v in accepted_attempt_indices[:checkpoint_index]]
    checkpoint_time = float(np.asarray(jax.device_get(baseline_rollout.trace.step_ts), dtype=float)[accepted_attempt_indices[-1]])

    print(
        "[autodiff-gate] realized-trace-checkpoint-interp-fd baseline summary: "
        f"attempt_count={baseline_diag['attempt_count']} "
        f"accepted_count={baseline_diag['accepted_count']} "
        f"checkpoint_index={checkpoint_index} "
        f"checkpoint_attempt={accepted_attempt_indices[-1]}",
        flush=True,
    )

    state0_static = _parameterized_initial_state(
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        geometry=runtime.geometry,
        n_species=runtime.species.number_species,
        parameter_name=parameter_name,
        parameter_value=jax.lax.stop_gradient(jnp.asarray(baseline_value)),
    )
    prepared_components_static = prepare_transport_solver_components(config, runtime, state0_static)
    solver_static = prepared_components_static["solver"]
    solve_vector_field_static = prepared_components_static["solve_vector_field"]
    prepared_rollout_static = _build_prepared_radau_accepted_rollout(
        solver=solver_static,
        state=state0_static,
        vector_field=solve_vector_field_static,
        species=runtime.species,
    )
    execution_context = _build_prepared_radau_execution_context(
        solver=solver_static,
        prepared_rollout=prepared_rollout_static,
    )
    unpack_flat = prepared_rollout_static.physics_context.unpack_flat

    carry0, carry0_dot = jax.jvp(
        lambda p: _adaptive_rollout_initial_carry_for_parameter(
            p,
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=parameter_name,
        ),
        (jnp.asarray(baseline_value),),
        (jnp.asarray(1.0),),
    )

    attempted_dts = np.asarray(jax.device_get(baseline_rollout.trace.attempted_dts), dtype=float)
    next_dts = np.asarray(jax.device_get(baseline_rollout.trace.next_dts), dtype=float)
    next_recent_reject_count = np.asarray(jax.device_get(baseline_rollout.trace.next_recent_reject_count))
    next_regrowth_cooldown = np.asarray(jax.device_get(baseline_rollout.trace.next_regrowth_cooldown))
    next_easy_growth_streak = np.asarray(jax.device_get(baseline_rollout.trace.next_easy_growth_streak))
    next_lagged_response_valid = np.asarray(jax.device_get(baseline_rollout.trace.next_lagged_response_valid))

    def _attempt_update(
        carry_value,
        attempt_fn,
        *,
        dt_value,
        next_dt_value,
        recent_reject_count_value,
        regrowth_cooldown_value,
        easy_growth_streak_value,
        lagged_response_valid_value,
    ):
        carry_for_step = dataclasses.replace(carry_value, dt=jnp.asarray(dt_value, dtype=execution_context.dtype))
        attempt_result = attempt_fn(
            execution_context.kernel_context,
            execution_context.physics_context,
            _radau_carry_with_forward_only_jvp_fields(carry_for_step),
            execution_context.attempt_context,
        )
        project_flat = execution_context.physics_context.project_flat
        accepted_y = project_flat(attempt_result.trial_y) if project_flat is not None else None
        if accepted_y is None:
            accepted_y = attempt_result.trial_y
        return dataclasses.replace(
            attempt_result.carry_after_attempt,
            t=carry_value.t + jnp.asarray(dt_value, dtype=execution_context.dtype),
            y=accepted_y,
            dt=jnp.asarray(next_dt_value, dtype=execution_context.dtype),
            prev_error=jnp.maximum(
                attempt_result.err_norm,
                jnp.asarray(1.0e-12, dtype=execution_context.dtype),
            ),
            prev_stages=attempt_result.stage_history,
            prev_dt=jnp.asarray(dt_value, dtype=execution_context.dtype),
            recent_reject_count=jnp.asarray(recent_reject_count_value),
            regrowth_cooldown=jnp.asarray(regrowth_cooldown_value),
            easy_growth_streak=jnp.asarray(easy_growth_streak_value),
            lagged_response_valid=jnp.asarray(lagged_response_valid_value),
            jacobian=attempt_result.jacobian_out,
            cache_valid=attempt_result.cache_valid_out,
            cache_dt=attempt_result.cache_dt_out,
            cache_age=attempt_result.cache_age_out,
            real_lu=attempt_result.real_lu_out,
            real_piv=attempt_result.real_piv_out,
            complex_lu=attempt_result.complex_lu_out,
            complex_piv=attempt_result.complex_piv_out,
            prev_theta_final=attempt_result.theta_final,
            prev_newton_iter_count=attempt_result.newton_iter_count,
        )

    compiled_custom_attempt_update = jax.jit(
        lambda carry_value, carry_dot_value, dt_value, next_dt_value, recent_reject_count_value, regrowth_cooldown_value, easy_growth_streak_value, lagged_response_valid_value: jax.jvp(
            lambda c: _attempt_update(
                c,
                _execute_radau_accepted_step_attempt_autodiff,
                dt_value=dt_value,
                next_dt_value=next_dt_value,
                recent_reject_count_value=recent_reject_count_value,
                regrowth_cooldown_value=regrowth_cooldown_value,
                easy_growth_streak_value=easy_growth_streak_value,
                lagged_response_valid_value=lagged_response_valid_value,
            ),
            (carry_value,),
            (carry_dot_value,),
        )
    )

    custom_carry = carry0
    custom_carry_dot = carry0_dot
    for idx in accepted_attempt_indices:
        custom_carry, custom_carry_dot = compiled_custom_attempt_update(
            custom_carry,
            custom_carry_dot,
            jnp.asarray(attempted_dts[idx], dtype=execution_context.dtype),
            jnp.asarray(next_dts[idx], dtype=execution_context.dtype),
            jnp.asarray(next_recent_reject_count[idx]),
            jnp.asarray(next_regrowth_cooldown[idx]),
            jnp.asarray(next_easy_growth_streak[idx]),
            jnp.asarray(next_lagged_response_valid[idx]),
        )

    def _objective_from_flat_y(flat_y):
        return _objective_vector(unpack_flat(flat_y), runtime)

    _, objective_ad = jax.jvp(_objective_from_flat_y, (custom_carry.y,), (custom_carry_dot.y,))
    grad_ad_np = np.asarray(jax.device_get(objective_ad), dtype=float)

    def _adaptive_objective_traj_for_value(param_value: float):
        _, rollout = _adaptive_rollout_final_state_for_parameter(
            jnp.asarray(param_value),
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=parameter_name,
            use_realized_schedule_jvp=False,
        )
        accepted_mask = np.asarray(jax.device_get(rollout.trace.accepted_mask), dtype=bool)
        active_mask = np.asarray(jax.device_get(rollout.trace.active_mask), dtype=bool)
        accepted_step_mask = np.logical_and(active_mask, accepted_mask)
        accepted_count = int(np.sum(accepted_step_mask))
        accepted_flat_ys = _compress_accepted_trial_ys(
            rollout.trace.y_end,
            jnp.asarray(accepted_step_mask),
            accepted_count=accepted_count,
        )
        traj = _objective_trajectory_from_flat_ys(
            accepted_flat_ys,
            runtime=runtime,
            unpack_flat=unpack_flat,
        )
        step_ts = np.asarray(jax.device_get(rollout.trace.step_ts), dtype=float)
        accepted_times = step_ts[accepted_step_mask]
        return accepted_times, np.asarray(jax.device_get(traj), dtype=float), rollout

    minus_times, minus_traj, minus_rollout = _adaptive_objective_traj_for_value(minus_value)
    plus_times, plus_traj, plus_rollout = _adaptive_objective_traj_for_value(plus_value)

    def _interp_objectives(times: np.ndarray, values: np.ndarray, target: float) -> np.ndarray:
        if times.size == 0:
            raise ValueError("Empty trajectory for interpolation.")
        if target <= float(times[0]):
            return values[0].copy()
        if target >= float(times[-1]):
            return values[-1].copy()
        cols = [np.interp(target, times, values[:, i]) for i in range(values.shape[1])]
        return np.asarray(cols, dtype=float)

    interp_minus = _interp_objectives(minus_times, minus_traj, checkpoint_time)
    interp_plus = _interp_objectives(plus_times, plus_traj, checkpoint_time)
    grad_fd_np = (interp_plus - interp_minus) / (2.0 * fd_step)
    abs_err = np.abs(grad_ad_np - grad_fd_np)
    rel_err = abs_err / np.maximum(np.abs(grad_fd_np), 1.0e-10)

    return {
        "config_path": str(config_path),
        "realized_trace_checkpoint_interpolated_fd_check": True,
        "parameter_name": parameter_name,
        "baseline_value": baseline_value,
        "fd_step": float(fd_step),
        "checkpoint_index": int(checkpoint_index),
        "checkpoint_time": float(checkpoint_time),
        "gradient_autodiff": grad_ad_np.tolist(),
        "gradient_fd": grad_fd_np.tolist(),
        "gradient_absolute_error": abs_err.tolist(),
        "gradient_relative_error": rel_err.tolist(),
        "max_relative_error": float(np.max(rel_err)),
        "passed": bool(np.all(np.isfinite(rel_err)) and np.max(rel_err) <= 5.0e-2),
        "objective_labels": OBJECTIVE_LABELS,
        "rollout_path": {
            "baseline": baseline_diag,
            "fd_minus": _adaptive_rollout_diagnostics(minus_rollout),
            "fd_plus": _adaptive_rollout_diagnostics(plus_rollout),
        },
    }


def build_ntx_derivative_mode_compare_report(
    *,
    config_path: Path,
    parameter_name: str,
    rel_fd_step: float,
    abs_fd_step: float,
    device: str | None,
    checkpoint_index: int,
    replay_mode: str = "attempt",
    include_direct_ad: bool = True,
    compute_five_point: bool = False,
) -> dict[str, Any]:
    modes = ("direct", "custom_vjp")
    reports: dict[str, dict[str, Any]] = {}
    timings: dict[str, dict[str, float]] = {}

    for mode in modes:
        t0 = time.perf_counter()
        report = build_realized_trace_checkpoint_frozen_fd_report(
            config_path=config_path,
            parameter_name=parameter_name,
            rel_fd_step=rel_fd_step,
            abs_fd_step=abs_fd_step,
            device=device,
            checkpoint_index=checkpoint_index,
            replay_mode=replay_mode,
            include_direct_ad=include_direct_ad,
            compute_five_point=compute_five_point,
            ntx_exact_derivative_mode=mode,
        )
        elapsed = time.perf_counter() - t0
        reports[mode] = report
        timings[mode] = {"wall_seconds": float(elapsed)}

    direct_report = reports["direct"]
    custom_vjp_report = reports["custom_vjp"]

    grad_direct = np.asarray(direct_report["gradient_autodiff"], dtype=float)
    grad_custom_vjp = np.asarray(custom_vjp_report["gradient_autodiff"], dtype=float)
    grad_mode_abs_err = np.abs(grad_direct - grad_custom_vjp)
    grad_mode_rel_err = grad_mode_abs_err / np.maximum(np.abs(grad_direct), 1.0e-10)

    state_direct = direct_report.get("state_tangent_comparison", {}).get("custom_vs_fd", {})
    state_custom_vjp = custom_vjp_report.get("state_tangent_comparison", {}).get("custom_vs_fd", {})

    return {
        "config_path": str(config_path),
        "ntx_derivative_mode_compare_check": True,
        "parameter_name": parameter_name,
        "checkpoint_index": int(checkpoint_index),
        "replay_mode": str(replay_mode),
        "include_direct_ad": bool(include_direct_ad),
        "compute_five_point": bool(compute_five_point),
        "modes": {
            "direct": direct_report,
            "custom_vjp": custom_vjp_report,
        },
        "timings": timings,
        "custom_ad_mode_difference": {
            "objective_absolute_error": grad_mode_abs_err.tolist(),
            "objective_relative_error": grad_mode_rel_err.tolist(),
            "max_objective_relative_error": float(np.max(grad_mode_rel_err)),
            "state_custom_vs_fd_direct": state_direct,
            "state_custom_vs_fd_custom_vjp": state_custom_vjp,
        },
        "objective_labels": OBJECTIVE_LABELS,
        "passed": bool(
            direct_report.get("passed", False)
            and custom_vjp_report.get("passed", False)
            and np.all(np.isfinite(grad_mode_rel_err))
        ),
    }


def build_realized_schedule_windowed_frozen_fd_report(
    *,
    config_path: Path,
    parameter_name: str,
    rel_fd_step: float,
    abs_fd_step: float,
    device: str | None,
    replay_mode: str = "attempt",
    accepted_window_size: int = 10,
) -> dict[str, Any]:
    if parameter_name not in ALLOWED_PARAMETERS:
        raise ValueError(f"parameter_name must be one of {sorted(ALLOWED_PARAMETERS)}")

    config = _prepare_benchmark_config(config_path, device=device)
    runtime, baseline_state = build_runtime_context(config)
    profile_cfg = _baseline_profile_cfg(config)
    baseline_value = float(profile_cfg[parameter_name])
    fd_step = _fd_step(baseline_value, rel_step=rel_fd_step, abs_step=abs_fd_step)
    minus_value = baseline_value - fd_step
    plus_value = baseline_value + fd_step

    objective_fn = lambda p: _adaptive_rollout_objectives_for_parameter(  # noqa: E731
        p,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        use_realized_schedule_jvp=True,
    )[0]

    baseline_objectives, baseline_rollout = _adaptive_rollout_objectives_for_parameter(
        jnp.asarray(baseline_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        use_realized_schedule_jvp=True,
    )
    baseline_diag = _adaptive_rollout_diagnostics(baseline_rollout)
    print(
        "[autodiff-gate] windowed-frozen-fd baseline summary: "
        f"attempt_count={baseline_diag['attempt_count']} "
        f"accepted_count={baseline_diag['accepted_count']} "
        f"completed={baseline_diag['completed']} "
        f"failed={baseline_diag['failed']} "
        f"fail_code={baseline_diag['fail_code']}",
        flush=True,
    )
    print(
        "[autodiff-gate] windowed-frozen-fd progress: baseline rollout complete; "
        f"running windowed fd_minus replay ({replay_mode}, window={accepted_window_size})",
        flush=True,
    )
    objectives_minus, minus_replay = _adaptive_rollout_objectives_for_parameter_on_windowed_frozen_trace(
        jnp.asarray(minus_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        baseline_trace=baseline_rollout.trace,
        replay_mode=replay_mode,
        accepted_window_size=accepted_window_size,
    )
    minus_objectives_np = np.asarray(jax.device_get(objectives_minus), dtype=float)
    minus_replay_finite = _tree_all_finite(minus_replay["final_state"]) and bool(np.all(np.isfinite(minus_objectives_np)))
    print(
        "[autodiff-gate] windowed-frozen-fd fd_minus summary: "
        f"state_finite={_tree_all_finite(minus_replay['final_state'])} "
        f"objectives_finite={bool(np.all(np.isfinite(minus_objectives_np)))} "
        f"all_finite={minus_replay_finite}",
        flush=True,
    )
    print(
        "[autodiff-gate] windowed-frozen-fd progress: windowed fd_minus replay complete; "
        f"running windowed fd_plus replay ({replay_mode}, window={accepted_window_size})",
        flush=True,
    )
    objectives_plus, plus_replay = _adaptive_rollout_objectives_for_parameter_on_windowed_frozen_trace(
        jnp.asarray(plus_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        baseline_trace=baseline_rollout.trace,
        replay_mode=replay_mode,
        accepted_window_size=accepted_window_size,
    )
    plus_objectives_np = np.asarray(jax.device_get(objectives_plus), dtype=float)
    plus_replay_finite = _tree_all_finite(plus_replay["final_state"]) and bool(np.all(np.isfinite(plus_objectives_np)))
    print(
        "[autodiff-gate] windowed-frozen-fd fd_plus summary: "
        f"state_finite={_tree_all_finite(plus_replay['final_state'])} "
        f"objectives_finite={bool(np.all(np.isfinite(plus_objectives_np)))} "
        f"all_finite={plus_replay_finite}",
        flush=True,
    )
    print(
        "[autodiff-gate] windowed-frozen-fd progress: windowed fd_plus replay complete; running AD gradient",
        flush=True,
    )
    gradient_ad = jax.jacfwd(objective_fn)(jnp.asarray(baseline_value))
    print(
        "[autodiff-gate] windowed-frozen-fd progress: AD gradient complete; forming windowed frozen FD gradient",
        flush=True,
    )
    gradient_fd = (objectives_plus - objectives_minus) / (2.0 * fd_step)

    grad_ad_np = np.asarray(jax.device_get(gradient_ad), dtype=float)
    grad_fd_np = np.asarray(jax.device_get(gradient_fd), dtype=float)
    abs_err = np.abs(grad_ad_np - grad_fd_np)
    rel_err = abs_err / np.maximum(np.abs(grad_fd_np), 1.0e-10)

    return {
        "config_path": str(config_path),
        "realized_schedule_windowed_frozen_fd_check": True,
        "parameter_name": parameter_name,
        "baseline_value": baseline_value,
        "fd_step": float(fd_step),
        "baseline_objectives": np.asarray(jax.device_get(baseline_objectives), dtype=float).tolist(),
        "gradient_autodiff": grad_ad_np.tolist(),
        "gradient_fd": grad_fd_np.tolist(),
        "gradient_absolute_error": abs_err.tolist(),
        "gradient_relative_error": rel_err.tolist(),
        "max_relative_error": float(np.max(rel_err)),
        "passed": bool(np.all(np.isfinite(rel_err)) and np.max(rel_err) <= 5.0e-2),
        "objective_labels": OBJECTIVE_LABELS,
        "windowed_replay_mode": str(replay_mode),
        "accepted_window_size": int(accepted_window_size),
        "rollout_path": {
            "baseline": baseline_diag,
            "windowed_fd_minus": {
                "all_finite": minus_replay_finite,
                "window_summaries": minus_replay.get("window_summaries"),
                "first_failing_window_debug": minus_replay.get("first_failing_window_debug"),
            },
            "windowed_fd_plus": {
                "all_finite": plus_replay_finite,
                "window_summaries": plus_replay.get("window_summaries"),
                "first_failing_window_debug": plus_replay.get("first_failing_window_debug"),
            },
        },
    }


def build_adaptive_vs_frozen_custom_ad_report(
    *,
    config_path: Path,
    parameter_name: str,
    device: str | None,
    checkpoint_index: int,
    replay_mode: str = "attempt",
    ntx_exact_derivative_mode: str | None = None,
) -> dict[str, Any]:
    adaptive_started = time.perf_counter()
    adaptive_report = build_realized_schedule_frozen_fd_report(
        config_path=config_path,
        parameter_name=parameter_name,
        rel_fd_step=3.0e-8,
        abs_fd_step=1.0e-10,
        device=device,
        replay_mode=replay_mode,
        accepted_step_limit=checkpoint_index,
        keep_adaptive_ad=True,
        ad_only=True,
    )
    adaptive_seconds = time.perf_counter() - adaptive_started

    frozen_started = time.perf_counter()
    frozen_report = build_realized_trace_checkpoint_compare_report(
        config_path=config_path,
        parameter_name=parameter_name,
        device=device,
        checkpoint_index=checkpoint_index,
    )
    frozen_seconds = time.perf_counter() - frozen_started

    grad_adaptive = np.asarray(adaptive_report["gradient_autodiff"], dtype=float)
    grad_frozen = np.asarray(frozen_report["gradient_autodiff"], dtype=float)
    abs_err = np.abs(grad_adaptive - grad_frozen)
    rel_err = abs_err / np.maximum(np.abs(grad_frozen), 1.0e-10)

    return {
        "config_path": str(config_path),
        "adaptive_vs_frozen_custom_ad_check": True,
        "parameter_name": parameter_name,
        "checkpoint_index": int(checkpoint_index),
        "replay_mode": str(replay_mode),
        "ntx_exact_derivative_mode": None if ntx_exact_derivative_mode is None else str(ntx_exact_derivative_mode),
        "objective_labels": OBJECTIVE_LABELS,
        "gradient_adaptive_custom": grad_adaptive.tolist(),
        "gradient_frozen_custom": grad_frozen.tolist(),
        "gradient_absolute_error": abs_err.tolist(),
        "gradient_relative_error": rel_err.tolist(),
        "max_relative_error": float(np.max(rel_err)),
        "adaptive_runtime_seconds": float(adaptive_seconds),
        "frozen_runtime_seconds": float(frozen_seconds),
        "adaptive_report": adaptive_report,
        "frozen_report": frozen_report,
        "passed": bool(np.all(np.isfinite(rel_err)) and np.max(rel_err) <= 5.0e-2),
    }


def build_small_step_only_report(
    *,
    config_path: Path,
    parameter_name: str,
    rel_fd_step: float,
    abs_fd_step: float,
    small_step_counts: tuple[float, ...],
    small_step_scale: float,
    device: str | None,
) -> dict[str, Any]:
    if parameter_name not in ALLOWED_PARAMETERS:
        raise ValueError(f"parameter_name must be one of {sorted(ALLOWED_PARAMETERS)}")

    config = _prepare_benchmark_config(config_path, device=device)
    runtime, baseline_state = build_runtime_context(config)
    profile_cfg = _baseline_profile_cfg(config)
    baseline_value = float(profile_cfg[parameter_name])
    fd_step = _fd_step(baseline_value, rel_step=rel_fd_step, abs_step=abs_fd_step)
    small_step_composition = _small_step_composition_report(
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        baseline_value=baseline_value,
        fd_step=fd_step,
        small_step_counts=small_step_counts,
        small_step_scale=small_step_scale,
    )
    max_rel_error = max(float(entry["max_relative_error"]) for entry in small_step_composition)
    return {
        "config_path": str(config_path),
        "small_step_only_check": True,
        "parameter_name": parameter_name,
        "baseline_value": baseline_value,
        "fd_step": float(fd_step),
        "small_step_composition": small_step_composition,
        "passed": bool(np.isfinite(max_rel_error) and max_rel_error <= 5.0e-2),
        "max_relative_error": float(max_rel_error),
        "objective_labels": OBJECTIVE_LABELS,
    }


def build_report(
    *,
    config_path: Path,
    parameter_name: str,
    rel_fd_step: float,
    abs_fd_step: float,
    sweep_half_width_rel: float,
    sweep_points: int,
    with_sweep: bool,
    one_step_diagnostic: bool,
    with_fd_step_sweep: bool,
    fd_step_sweep_multipliers: tuple[float, ...],
    with_standalone_stage_subsolve_check: bool,
    with_small_step_composition_check: bool,
    with_controller_composition_check: bool,
    small_step_counts: tuple[float, ...],
    small_step_scale: float,
    device: str | None,
) -> dict[str, Any]:
    if parameter_name not in ALLOWED_PARAMETERS:
        raise ValueError(f"parameter_name must be one of {sorted(ALLOWED_PARAMETERS)}")

    config = _prepare_benchmark_config(config_path, device=device)
    if one_step_diagnostic:
        config = _apply_one_step_diagnostic_config(config)
    runtime, baseline_state = build_runtime_context(config)
    profile_cfg = _baseline_profile_cfg(config)
    baseline_value = float(profile_cfg[parameter_name])

    objective_fn = lambda p: _transport_objectives_for_parameter(  # noqa: E731
        p,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
    )

    fd_step = _fd_step(baseline_value, rel_step=rel_fd_step, abs_step=abs_fd_step)
    minus_value = baseline_value - fd_step
    plus_value = baseline_value + fd_step
    baseline_result = run_transport(
        config,
        runtime,
        _parameterized_initial_state(
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            geometry=runtime.geometry,
            n_species=runtime.species.number_species,
            parameter_name=parameter_name,
            parameter_value=jnp.asarray(baseline_value),
        ),
    )
    minus_result = run_transport(
        config,
        runtime,
        _parameterized_initial_state(
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            geometry=runtime.geometry,
            n_species=runtime.species.number_species,
            parameter_name=parameter_name,
            parameter_value=jnp.asarray(minus_value),
        ),
    )
    plus_result = run_transport(
        config,
        runtime,
        _parameterized_initial_state(
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            geometry=runtime.geometry,
            n_species=runtime.species.number_species,
            parameter_name=parameter_name,
            parameter_value=jnp.asarray(plus_value),
        ),
    )

    baseline_objectives = _objective_vector(baseline_result["final_state"], runtime)
    objectives_minus = _objective_vector(minus_result["final_state"], runtime)
    objectives_plus = _objective_vector(plus_result["final_state"], runtime)
    gradient_ad = jax.jacfwd(objective_fn)(jnp.asarray(baseline_value))
    gradient_fd = (objectives_plus - objectives_minus) / (2.0 * fd_step)

    grad_ad_np = np.asarray(jax.device_get(gradient_ad), dtype=float)
    grad_fd_np = np.asarray(jax.device_get(gradient_fd), dtype=float)
    abs_err = np.abs(grad_ad_np - grad_fd_np)
    rel_err = abs_err / np.maximum(np.abs(grad_fd_np), 1.0e-10)

    if with_sweep:
        sweep_half_width = sweep_half_width_rel * max(abs(baseline_value), 1.0)
        sweep_values = np.linspace(
            baseline_value - sweep_half_width,
            baseline_value + sweep_half_width,
            int(sweep_points),
            dtype=float,
        )
        sweep_objectives = np.stack(
            [
                np.asarray(jax.device_get(objective_fn(jnp.asarray(value))), dtype=float)
                for value in sweep_values
            ],
            axis=0,
        )
    else:
        sweep_values = np.asarray([minus_value, baseline_value, plus_value], dtype=float)
        sweep_objectives = np.stack(
            [
                np.asarray(jax.device_get(objectives_minus), dtype=float),
                np.asarray(jax.device_get(baseline_objectives), dtype=float),
                np.asarray(jax.device_get(objectives_plus), dtype=float),
            ],
            axis=0,
        )

    baseline_diag = _result_diagnostics(baseline_result)
    minus_diag = _result_diagnostics(minus_result)
    plus_diag = _result_diagnostics(plus_result)

    fd_step_sweep = None
    if with_fd_step_sweep:
        fd_step_sweep = _fd_step_sweep_report(
            runtime=runtime,
            config=config,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=parameter_name,
            baseline_value=baseline_value,
            gradient_ad=gradient_ad,
            fd_step=fd_step,
            step_multipliers=fd_step_sweep_multipliers,
        )

    standalone_stage_subsolve = None
    if with_standalone_stage_subsolve_check:
        standalone_objective_fn = lambda p: _standalone_stage_subsolve_objectives_for_parameter(  # noqa: E731
            p,
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=parameter_name,
        )
        standalone_ad = jax.jacfwd(standalone_objective_fn)(jnp.asarray(baseline_value))
        standalone_minus = np.asarray(
            jax.device_get(standalone_objective_fn(jnp.asarray(minus_value))),
            dtype=float,
        )
        standalone_plus = np.asarray(
            jax.device_get(standalone_objective_fn(jnp.asarray(plus_value))),
            dtype=float,
        )
        standalone_fd = (standalone_plus - standalone_minus) / (2.0 * fd_step)
        standalone_ad_np = np.asarray(jax.device_get(standalone_ad), dtype=float)
        standalone_abs_err = np.abs(standalone_ad_np - standalone_fd)
        standalone_rel_err = standalone_abs_err / np.maximum(np.abs(standalone_fd), 1.0e-10)
        standalone_stage_subsolve = {
            "labels": STANDALONE_SUBSOLVE_LABELS,
            "gradient_autodiff": standalone_ad_np.tolist(),
            "gradient_fd": standalone_fd.tolist(),
            "gradient_absolute_error": standalone_abs_err.tolist(),
            "gradient_relative_error": standalone_rel_err.tolist(),
            "max_relative_error": float(np.max(standalone_rel_err)),
        }

    small_step_composition = None
    if with_small_step_composition_check:
        small_step_composition = _small_step_composition_report(
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=parameter_name,
            baseline_value=baseline_value,
            fd_step=fd_step,
            small_step_counts=small_step_counts,
            small_step_scale=small_step_scale,
        )

    controller_step_composition = None
    if with_controller_composition_check:
        controller_step_composition = _controller_composition_report(
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=parameter_name,
            baseline_value=baseline_value,
            fd_step=fd_step,
            small_step_counts=small_step_counts,
            small_step_scale=small_step_scale,
        )

    report = {
        "config_path": str(config_path),
        "one_step_diagnostic": bool(one_step_diagnostic),
        "parameter_name": parameter_name,
        "baseline_value": baseline_value,
        "fd_step": float(fd_step),
        "baseline_objectives": np.asarray(jax.device_get(baseline_objectives), dtype=float).tolist(),
        "gradient_autodiff": grad_ad_np.tolist(),
        "gradient_fd": grad_fd_np.tolist(),
        "gradient_absolute_error": abs_err.tolist(),
        "gradient_relative_error": rel_err.tolist(),
        "max_relative_error": float(np.max(rel_err)),
        "passed": bool(np.all(np.isfinite(rel_err)) and np.max(rel_err) <= 5.0e-2),
        "objective_labels": OBJECTIVE_LABELS,
        "autodiff_reuses_baseline_value_only": True,
        "sweep_values": sweep_values.tolist(),
        "sweep_objectives": sweep_objectives.tolist(),
        "fd_step_sweep": fd_step_sweep,
        "standalone_stage_subsolve": standalone_stage_subsolve,
        "small_step_composition": small_step_composition,
        "controller_step_composition": controller_step_composition,
        "solver_path": {
            "baseline": baseline_diag,
            "fd_minus": minus_diag,
            "fd_plus": plus_diag,
            "accepted_mask_equal_minus_plus": (
                baseline_diag["accepted_mask"] is not None
                and minus_diag["accepted_mask"] is not None
                and plus_diag["accepted_mask"] is not None
                and minus_diag["accepted_mask"] == plus_diag["accepted_mask"]
            ),
            "saved_times_equal_minus_plus": _sequence_allclose(
                minus_diag["saved_times"],
                plus_diag["saved_times"],
            ),
            "saved_dts_equal_minus_plus": _sequence_allclose(
                minus_diag["saved_step_sizes"],
                plus_diag["saved_step_sizes"],
            ),
        },
        "rho_grid": np.asarray(jax.device_get(runtime.geometry.rho_grid), dtype=float).tolist(),
        "baseline_final_Er": np.asarray(jax.device_get(baseline_result["final_state"].Er), dtype=float).tolist(),
        "fd_minus_final_Er": np.asarray(jax.device_get(minus_result["final_state"].Er), dtype=float).tolist(),
        "fd_plus_final_Er": np.asarray(jax.device_get(plus_result["final_state"].Er), dtype=float).tolist(),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--parameter",
        default="n0",
        choices=sorted(ALLOWED_PARAMETERS),
        help="Initial-profile parameter to differentiate against.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--fd-rel-step", type=float, default=1.0e-3)
    parser.add_argument("--fd-abs-step", type=float, default=1.0e-4)
    parser.add_argument("--sweep-half-width-rel", type=float, default=5.0e-2)
    parser.add_argument("--sweep-points", type=int, default=7)
    parser.add_argument("--with-sweep", action="store_true", help="Run extra sweep solves for objective curves.")
    parser.add_argument(
        "--with-fd-step-sweep",
        action="store_true",
        help="Run extra full-solve FD checks at multiple FD step sizes.",
    )
    parser.add_argument(
        "--fd-step-sweep-multipliers",
        default="0.25,0.5,1.0,2.0,4.0",
        help="Comma-separated multipliers applied to the base FD step when --with-fd-step-sweep is enabled.",
    )
    parser.add_argument(
        "--one-step-diagnostic",
        action="store_true",
        help="Stop after one accepted transport step to isolate local AD-vs-FD behavior.",
    )
    parser.add_argument(
        "--with-standalone-stage-subsolve-check",
        action="store_true",
        help="Run an additive AD-vs-FD check on the standalone Radau stage-subsolve primitive.",
    )
    parser.add_argument(
        "--with-small-step-composition-check",
        action="store_true",
        help="Run an additive AD-vs-FD check on a short accepted-step composition map.",
    )
    parser.add_argument(
        "--small-step-only-check",
        action="store_true",
        help="Run only the short accepted-step composition check, without the full baseline/fd solve report.",
    )
    parser.add_argument(
        "--with-controller-composition-check",
        action="store_true",
        help="Run an additive AD-vs-FD check on a short rollout with the real Radau controller dt updates.",
    )
    parser.add_argument(
        "--controller-only-check",
        action="store_true",
        help="Run only the short rollout with real Radau controller dt updates and print controller trajectory diagnostics.",
    )
    parser.add_argument(
        "--forward-only-controller-check",
        action="store_true",
        help="Run only the short rollout with controller dt evolution treated as forward-only between steps.",
    )
    parser.add_argument(
        "--realized-schedule-rollout-check",
        action="store_true",
        help="Run a final-time-only adaptive-rollout check using the first solve-level custom JVP over the primal's realized accepted schedule.",
    )
    parser.add_argument(
        "--realized-schedule-direct-ad-compare-check",
        action="store_true",
        help="Run a final-time adaptive comparison of custom AD, direct adaptive AD, and FD.",
    )
    parser.add_argument(
        "--realized-schedule-frozen-fd-check",
        action="store_true",
        help="Run a cheaper realized-schedule check that compares AD against FD on the baseline frozen Radau replay path instead of two fresh adaptive FD solves.",
    )
    parser.add_argument(
        "--realized-schedule-frozen-replay-localize",
        action="store_true",
        help="Run a fast one-command binary search that localizes the first failing frozen replay prefix and compares attempt vs accepted replay at the failure boundary.",
    )
    parser.add_argument(
        "--realized-schedule-windowed-frozen-fd-check",
        action="store_true",
        help="Compare adaptive AD against FD built from short frozen baseline schedule windows with re-anchoring between windows.",
    )
    parser.add_argument(
        "--baseline-dt-path-safe-fd-check",
        action="store_true",
        help="Compare adaptive AD at a safe truncated final time against FD computed on the baseline accepted dt path up to two attempts before the known frozen-replay failure.",
    )
    parser.add_argument(
        "--baseline-dt-path-safe-compose-check",
        action="store_true",
        help="Compare adaptive realized-schedule AD against direct AD on the same safe baseline accepted dt path.",
    )
    parser.add_argument(
        "--baseline-dt-path-safe-compose-scan-check",
        action="store_true",
        help="Compare adaptive realized-schedule AD against fixed-dt direct AD at multiple accepted-step prefixes of the same safe baseline path in one run.",
    )
    parser.add_argument(
        "--baseline-dt-path-safe-trajectory-compare-check",
        action="store_true",
        help="Heavy opt-in mode: run one realized-trace adaptive AD trajectory and one fixed-dt direct AD trajectory, then compare per-accepted-step objective tangents along the safe baseline path.",
    )
    parser.add_argument(
        "--baseline-dt-path-safe-state-trajectory-compare-check",
        action="store_true",
        help="Heavy opt-in mode: run one realized-trace adaptive state-tangent trajectory and one fixed-dt direct state-tangent trajectory, then compare state-slice mismatches along the safe baseline path.",
    )
    parser.add_argument(
        "--realized-trace-safe-state-trajectory-compare-check",
        action="store_true",
        help="Heavy opt-in mode: run one realized-trace custom state-tangent trajectory and one realized-trace direct state-tangent trajectory on the same safe frozen trace.",
    )
    parser.add_argument(
        "--realized-trace-sixth-step-carry-ablation-check",
        action="store_true",
        help="Dedicated opt-in mode: compare custom vs direct on the sixth accepted realized-trace step and ablate carried step-5 tangent fields one at a time.",
    )
    parser.add_argument(
        "--realized-trace-sparse-checkpoint-compare-check",
        action="store_true",
        help="Dedicated opt-in mode: compare realized-trace custom vs direct only at a small set of accepted-step checkpoints.",
    )
    parser.add_argument(
        "--realized-trace-checkpoint-compare-check",
        action="store_true",
        help="Dedicated opt-in mode: compare realized-trace custom vs direct only at one accepted-step checkpoint.",
    )
    parser.add_argument(
        "--realized-trace-checkpoint-frozen-fd-check",
        action="store_true",
        help="Dedicated opt-in mode: compare baseline realized-trace checkpoint AD against frozen-trace FD at one accepted-step checkpoint.",
    )
    parser.add_argument(
        "--realized-trace-checkpoint-fd-stencil-check",
        action="store_true",
        help="Dedicated opt-in mode: compare baseline realized-trace checkpoint custom AD against frozen-trace center FD and five-point FD at one accepted-step checkpoint.",
    )
    parser.add_argument(
        "--ntx-derivative-mode-compare-check",
        action="store_true",
        help="Dedicated opt-in mode: run the same realized-trace checkpoint frozen-FD benchmark twice, once with NTX direct AD and once with NTX custom_vjp, and compare timings and custom-AD derivatives.",
    )
    parser.add_argument(
        "--adaptive-vs-frozen-custom-ad-check",
        action="store_true",
        help="Dedicated opt-in mode: compare the live adaptive custom-JVP derivative against the frozen accepted-trace custom derivative at one accepted-step checkpoint.",
    )
    parser.add_argument(
        "--realized-trace-checkpoint-interpolated-fd-check",
        action="store_true",
        help="Dedicated opt-in mode: compare baseline realized-trace checkpoint AD against adaptive fd_minus/fd_plus trajectories interpolated to the baseline checkpoint time.",
    )
    parser.add_argument(
        "--skip-direct-ad-in-frozen-check",
        action="store_true",
        help="For --realized-trace-checkpoint-frozen-fd-check, skip the direct-AD reference path and run only custom AD vs FD.",
    )
    parser.add_argument(
        "--ntx-exact-derivative-mode",
        default="direct",
        choices=("direct", "custom_vjp"),
        help="NTX exact-runtime derivative mode used by the benchmark when a single-mode run is requested.",
    )
    parser.add_argument(
        "--realized-trace-checkpoint-index",
        type=int,
        default=10,
        help="Accepted-step checkpoint used by the realized-trace checkpoint compare/frozen-FD/interpolated-FD modes.",
    )
    parser.add_argument(
        "--realized-trace-sparse-checkpoint-counts",
        default="10,20",
        help="Comma-separated accepted-step checkpoints used by --realized-trace-sparse-checkpoint-compare-check.",
    )
    parser.add_argument(
        "--allow-heavy-trajectory-diagnostics",
        action="store_true",
        help="Allow RAM-heavy full-trajectory diagnostics. Without this flag, use cheap checkpoint/localized modes instead.",
    )
    parser.add_argument(
        "--baseline-dt-path-first-step-field-compare-check",
        action="store_true",
        help="Dedicated opt-in mode: compare adaptive vs direct fixed-dt tangent fields at the first accepted step only.",
    )
    parser.add_argument(
        "--baseline-dt-path-first-step-local-tangent-compare-check",
        action="store_true",
        help="Dedicated opt-in mode: compare custom vs direct first-step local tangents for trial_y, carry_after_attempt.y, and stage_history.",
    )
    parser.add_argument(
        "--baseline-dt-path-first-step-exact-local-tangent-compare-check",
        action="store_true",
        help="Dedicated opt-in mode: compare custom and exact-residual one-step local tangents against direct AD for the first accepted step.",
    )
    parser.add_argument(
        "--baseline-dt-path-second-step-carry-ablation-check",
        action="store_true",
        help="Dedicated opt-in mode: compare custom vs direct on the second accepted step and ablate step-1 carry tangents one field at a time.",
    )
    parser.add_argument(
        "--baseline-dt-path-third-step-carry-ablation-check",
        action="store_true",
        help="Dedicated opt-in mode: compare custom vs direct on the third accepted step and ablate step-2 carry tangents one field at a time.",
    )
    parser.add_argument(
        "--baseline-dt-path-safe-trajectory-sample-every",
        type=int,
        default=5,
        help="Accepted-step sampling stride used by --baseline-dt-path-safe-trajectory-compare-check to limit memory and output volume.",
    )
    parser.add_argument(
        "--baseline-dt-path-safe-compose-scan-counts",
        default="1,2,5,10,20,30,40,45",
        help="Comma-separated accepted-step prefixes used by --baseline-dt-path-safe-compose-scan-check.",
    )
    parser.add_argument(
        "--known-first-bad-attempt-index",
        type=int,
        default=76,
        help="Earliest known frozen-FD bad attempt index used to truncate the safe baseline dt path before fd_plus/fd_minus go nonfinite.",
    )
    parser.add_argument(
        "--safe-attempt-margin",
        type=int,
        default=2,
        help="How many attempts before the known bad attempt to stop the safe baseline dt path.",
    )
    parser.add_argument(
        "--realized-schedule-windowed-accepted-window-size",
        type=int,
        default=10,
        help="Accepted-step window size used by --realized-schedule-windowed-frozen-fd-check.",
    )
    parser.add_argument(
        "--realized-schedule-frozen-replay-mode",
        default="attempt",
        choices=("attempt", "accepted"),
        help="Frozen replay mode used by --realized-schedule-frozen-fd-check.",
    )
    parser.add_argument(
        "--realized-schedule-frozen-accepted-steps",
        type=int,
        default=None,
        help="Optional prefix length for --realized-schedule-frozen-fd-check. When set, replay only the first N accepted steps from the baseline realized schedule.",
    )
    parser.add_argument(
        "--realized-schedule-frozen-keep-adaptive-ad",
        action="store_true",
        help="For --realized-schedule-frozen-fd-check, keep AD on the live realized-schedule custom-JVP lane even when --realized-schedule-frozen-accepted-steps truncates the FD replay prefix.",
    )
    parser.add_argument(
        "--realized-schedule-frozen-ad-only",
        action="store_true",
        help="For --realized-schedule-frozen-fd-check, run only the AD derivative and skip the frozen FD replays.",
    )
    parser.add_argument(
        "--realized-schedule-ad-debug-fast",
        action="store_true",
        help="Run the cheapest realized-schedule AD failure-localization path: baseline rollout, AD gradient, and optional minimal NaN localization only.",
    )
    parser.add_argument(
        "--realized-schedule-nan-debug",
        action="store_true",
        help="When --realized-schedule-rollout-check is enabled, run a minimal NaN-localization replay pass if AD returns nonfinite values.",
    )
    parser.add_argument(
        "--realized-schedule-nan-debug-exhaustive",
        action="store_true",
        help="Expand --realized-schedule-nan-debug to run the full replay sweep with all tangent-zeroing variants.",
    )
    parser.add_argument(
        "--realized-schedule-nan-debug-one-step-compare",
        action="store_true",
        help="Add the expensive one-step custom-vs-direct zero-tangent comparison to NaN localization.",
    )
    parser.add_argument(
        "--small-step-counts",
        default="2,3,5",
        help="Comma-separated accepted-step counts used by the full-report short accepted-step composition check.",
    )
    parser.add_argument(
        "--small-step-scale",
        type=float,
        default=0.25,
        help="Scale applied to the initial Radau dt for the small-step composition check.",
    )
    parser.add_argument("--outdir", type=Path, default=Path("outputs/autodiff_transport_lagged_ntx"))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    heavy_trajectory_flags = (
        args.baseline_dt_path_safe_trajectory_compare_check,
        args.baseline_dt_path_safe_state_trajectory_compare_check,
        args.realized_trace_safe_state_trajectory_compare_check,
    )
    if any(heavy_trajectory_flags) and not args.allow_heavy_trajectory_diagnostics:
        raise SystemExit(
            "Refusing to run RAM-heavy trajectory diagnostics without "
            "--allow-heavy-trajectory-diagnostics. "
            "Use cheap checkpoint modes instead, for example: "
            "--realized-trace-checkpoint-compare-check --realized-trace-checkpoint-index 10 "
            "or --realized-trace-sparse-checkpoint-compare-check "
            "--realized-trace-sparse-checkpoint-counts 10,20."
        )

    if args.realized_schedule_ad_debug_fast:
        report = build_realized_schedule_ad_debug_fast_report(
            config_path=args.config,
            parameter_name=args.parameter,
            device=args.device,
            include_nan_debug=args.realized_schedule_nan_debug,
            nan_debug_mode="exhaustive" if args.realized_schedule_nan_debug_exhaustive else "minimal",
            nan_debug_include_one_step_compare=args.realized_schedule_nan_debug_one_step_compare,
        )
    elif args.realized_schedule_rollout_check:
        report = build_realized_schedule_rollout_report(
            config_path=args.config,
            parameter_name=args.parameter,
            rel_fd_step=args.fd_rel_step,
            abs_fd_step=args.fd_abs_step,
            device=args.device,
            include_nan_debug=args.realized_schedule_nan_debug,
            nan_debug_mode="exhaustive" if args.realized_schedule_nan_debug_exhaustive else "minimal",
            nan_debug_include_one_step_compare=args.realized_schedule_nan_debug_one_step_compare,
        )
    elif args.realized_schedule_direct_ad_compare_check:
        report = build_realized_schedule_direct_ad_compare_report(
            config_path=args.config,
            parameter_name=args.parameter,
            rel_fd_step=args.fd_rel_step,
            abs_fd_step=args.fd_abs_step,
            device=args.device,
        )
    elif args.realized_schedule_frozen_fd_check:
        report = build_realized_schedule_frozen_fd_report(
            config_path=args.config,
            parameter_name=args.parameter,
            rel_fd_step=args.fd_rel_step,
            abs_fd_step=args.fd_abs_step,
            device=args.device,
            replay_mode=args.realized_schedule_frozen_replay_mode,
            accepted_step_limit=args.realized_schedule_frozen_accepted_steps,
            keep_adaptive_ad=args.realized_schedule_frozen_keep_adaptive_ad,
            ad_only=args.realized_schedule_frozen_ad_only,
        )
    elif args.realized_schedule_frozen_replay_localize:
        report = build_realized_schedule_frozen_replay_localize_report(
            config_path=args.config,
            parameter_name=args.parameter,
            rel_fd_step=args.fd_rel_step,
            abs_fd_step=args.fd_abs_step,
            device=args.device,
        )
    elif args.realized_schedule_windowed_frozen_fd_check:
        report = build_realized_schedule_windowed_frozen_fd_report(
            config_path=args.config,
            parameter_name=args.parameter,
            rel_fd_step=args.fd_rel_step,
            abs_fd_step=args.fd_abs_step,
            device=args.device,
            replay_mode=args.realized_schedule_frozen_replay_mode,
            accepted_window_size=args.realized_schedule_windowed_accepted_window_size,
        )
    elif args.baseline_dt_path_safe_fd_check:
        report = build_baseline_dt_path_safe_fd_report(
            config_path=args.config,
            parameter_name=args.parameter,
            rel_fd_step=args.fd_rel_step,
            abs_fd_step=args.fd_abs_step,
            device=args.device,
            known_first_bad_attempt_index=args.known_first_bad_attempt_index,
            safe_attempt_margin=args.safe_attempt_margin,
        )
    elif args.baseline_dt_path_safe_compose_check:
        report = build_baseline_dt_path_safe_compose_report(
            config_path=args.config,
            parameter_name=args.parameter,
            rel_fd_step=args.fd_rel_step,
            abs_fd_step=args.fd_abs_step,
            device=args.device,
            known_first_bad_attempt_index=args.known_first_bad_attempt_index,
            safe_attempt_margin=args.safe_attempt_margin,
        )
    elif args.baseline_dt_path_safe_compose_scan_check:
        report = build_baseline_dt_path_safe_compose_scan_report(
            config_path=args.config,
            parameter_name=args.parameter,
            device=args.device,
            known_first_bad_attempt_index=args.known_first_bad_attempt_index,
            safe_attempt_margin=args.safe_attempt_margin,
            accepted_step_counts=_parse_int_csv(args.baseline_dt_path_safe_compose_scan_counts),
        )
    elif args.baseline_dt_path_safe_trajectory_compare_check:
        report = build_baseline_dt_path_safe_trajectory_compare_report(
            config_path=args.config,
            parameter_name=args.parameter,
            device=args.device,
            known_first_bad_attempt_index=args.known_first_bad_attempt_index,
            safe_attempt_margin=args.safe_attempt_margin,
            sample_every=args.baseline_dt_path_safe_trajectory_sample_every,
        )
    elif args.baseline_dt_path_safe_state_trajectory_compare_check:
        report = build_baseline_dt_path_safe_state_trajectory_compare_report(
            config_path=args.config,
            parameter_name=args.parameter,
            device=args.device,
            known_first_bad_attempt_index=args.known_first_bad_attempt_index,
            safe_attempt_margin=args.safe_attempt_margin,
            sample_every=args.baseline_dt_path_safe_trajectory_sample_every,
        )
    elif args.realized_trace_safe_state_trajectory_compare_check:
        report = build_realized_trace_safe_state_trajectory_compare_report(
            config_path=args.config,
            parameter_name=args.parameter,
            device=args.device,
            known_first_bad_attempt_index=args.known_first_bad_attempt_index,
            safe_attempt_margin=args.safe_attempt_margin,
            sample_every=args.baseline_dt_path_safe_trajectory_sample_every,
        )
    elif args.realized_trace_sixth_step_carry_ablation_check:
        report = build_realized_trace_sixth_step_carry_ablation_report(
            config_path=args.config,
            parameter_name=args.parameter,
            device=args.device,
        )
    elif args.realized_trace_checkpoint_compare_check:
        report = build_realized_trace_checkpoint_compare_report(
            config_path=args.config,
            parameter_name=args.parameter,
            device=args.device,
            checkpoint_index=args.realized_trace_checkpoint_index,
        )
    elif args.ntx_derivative_mode_compare_check:
        report = build_ntx_derivative_mode_compare_report(
            config_path=args.config,
            parameter_name=args.parameter,
            rel_fd_step=args.fd_rel_step,
            abs_fd_step=args.fd_abs_step,
            device=args.device,
            checkpoint_index=args.realized_trace_checkpoint_index,
            replay_mode=args.realized_schedule_frozen_replay_mode,
            include_direct_ad=not args.skip_direct_ad_in_frozen_check,
            compute_five_point=False,
        )
    elif args.adaptive_vs_frozen_custom_ad_check:
        report = build_adaptive_vs_frozen_custom_ad_report(
            config_path=args.config,
            parameter_name=args.parameter,
            device=args.device,
            checkpoint_index=args.realized_trace_checkpoint_index,
            replay_mode=args.realized_schedule_frozen_replay_mode,
            ntx_exact_derivative_mode=args.ntx_exact_derivative_mode,
        )
    elif args.realized_trace_checkpoint_frozen_fd_check:
        report = build_realized_trace_checkpoint_frozen_fd_report(
            config_path=args.config,
            parameter_name=args.parameter,
            rel_fd_step=args.fd_rel_step,
            abs_fd_step=args.fd_abs_step,
            device=args.device,
            checkpoint_index=args.realized_trace_checkpoint_index,
            replay_mode=args.realized_schedule_frozen_replay_mode,
            include_direct_ad=not args.skip_direct_ad_in_frozen_check,
            ntx_exact_derivative_mode=args.ntx_exact_derivative_mode,
        )
    elif args.realized_trace_checkpoint_fd_stencil_check:
        report = build_realized_trace_checkpoint_frozen_fd_report(
            config_path=args.config,
            parameter_name=args.parameter,
            rel_fd_step=args.fd_rel_step,
            abs_fd_step=args.fd_abs_step,
            device=args.device,
            checkpoint_index=args.realized_trace_checkpoint_index,
            replay_mode=args.realized_schedule_frozen_replay_mode,
            include_direct_ad=False,
            compute_five_point=True,
            ntx_exact_derivative_mode=args.ntx_exact_derivative_mode,
        )
    elif args.realized_trace_checkpoint_interpolated_fd_check:
        report = build_realized_trace_checkpoint_interpolated_fd_report(
            config_path=args.config,
            parameter_name=args.parameter,
            rel_fd_step=args.fd_rel_step,
            abs_fd_step=args.fd_abs_step,
            device=args.device,
            checkpoint_index=args.realized_trace_checkpoint_index,
        )
    elif args.realized_trace_sparse_checkpoint_compare_check:
        report = build_realized_trace_sparse_checkpoint_compare_report(
            config_path=args.config,
            parameter_name=args.parameter,
            device=args.device,
            checkpoint_counts=_parse_int_csv(args.realized_trace_sparse_checkpoint_counts),
        )
    elif args.baseline_dt_path_first_step_field_compare_check:
        report = build_baseline_dt_path_first_step_field_compare_report(
            config_path=args.config,
            parameter_name=args.parameter,
            device=args.device,
            known_first_bad_attempt_index=args.known_first_bad_attempt_index,
            safe_attempt_margin=args.safe_attempt_margin,
        )
    elif args.baseline_dt_path_first_step_local_tangent_compare_check:
        report = build_baseline_dt_path_first_step_local_tangent_compare_report(
            config_path=args.config,
            parameter_name=args.parameter,
            device=args.device,
            known_first_bad_attempt_index=args.known_first_bad_attempt_index,
            safe_attempt_margin=args.safe_attempt_margin,
        )
    elif args.baseline_dt_path_first_step_exact_local_tangent_compare_check:
        report = build_baseline_dt_path_first_step_exact_local_tangent_compare_report(
            config_path=args.config,
            parameter_name=args.parameter,
            device=args.device,
            known_first_bad_attempt_index=args.known_first_bad_attempt_index,
            safe_attempt_margin=args.safe_attempt_margin,
        )
    elif args.baseline_dt_path_second_step_carry_ablation_check:
        report = build_baseline_dt_path_second_step_carry_ablation_report(
            config_path=args.config,
            parameter_name=args.parameter,
            device=args.device,
        )
    elif args.baseline_dt_path_third_step_carry_ablation_check:
        report = build_baseline_dt_path_third_step_carry_ablation_report(
            config_path=args.config,
            parameter_name=args.parameter,
            device=args.device,
        )
    elif args.forward_only_controller_check:
        report = build_forward_only_controller_report(
            config_path=args.config,
            parameter_name=args.parameter,
            rel_fd_step=args.fd_rel_step,
            abs_fd_step=args.fd_abs_step,
            small_step_counts=_parse_float_csv(args.small_step_counts),
            small_step_scale=args.small_step_scale,
            device=args.device,
        )
    elif args.controller_only_check:
        report = build_controller_only_report(
            config_path=args.config,
            parameter_name=args.parameter,
            rel_fd_step=args.fd_rel_step,
            abs_fd_step=args.fd_abs_step,
            small_step_counts=_parse_float_csv(args.small_step_counts),
            small_step_scale=args.small_step_scale,
            device=args.device,
        )
    elif args.small_step_only_check:
        report = build_small_step_only_report(
            config_path=args.config,
            parameter_name=args.parameter,
            rel_fd_step=args.fd_rel_step,
            abs_fd_step=args.fd_abs_step,
            small_step_counts=_parse_float_csv(args.small_step_counts),
            small_step_scale=args.small_step_scale,
            device=args.device,
        )
    else:
        report = build_report(
            config_path=args.config,
            parameter_name=args.parameter,
            rel_fd_step=args.fd_rel_step,
            abs_fd_step=args.fd_abs_step,
            sweep_half_width_rel=args.sweep_half_width_rel,
            sweep_points=args.sweep_points,
            with_sweep=args.with_sweep,
            one_step_diagnostic=args.one_step_diagnostic,
            with_fd_step_sweep=args.with_fd_step_sweep,
            fd_step_sweep_multipliers=_parse_float_csv(args.fd_step_sweep_multipliers),
            with_standalone_stage_subsolve_check=args.with_standalone_stage_subsolve_check,
            with_small_step_composition_check=args.with_small_step_composition_check,
            with_controller_composition_check=args.with_controller_composition_check,
            small_step_counts=_parse_float_csv(args.small_step_counts),
            small_step_scale=args.small_step_scale,
            device=args.device,
        )

    outdir = args.outdir / args.parameter
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / f"transport_autodiff_{args.parameter}_summary.json"
    csv_path = outdir / f"transport_autodiff_{args.parameter}_sweep.csv"
    fig_path = outdir / f"transport_autodiff_{args.parameter}.png"

    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_terminal_summary(report)
    print(f"Wrote {json_path}")
    if "sweep_values" in report and "sweep_objectives" in report:
        _write_sweep_csv(
            csv_path,
            parameter_name=report["parameter_name"],
            sweep_values=np.asarray(report["sweep_values"], dtype=float),
            objective_values=np.asarray(report["sweep_objectives"], dtype=float),
        )
        print(f"Wrote {csv_path}")
        if not args.no_plot:
            _write_figure(report, fig_path)
            print(f"Wrote {fig_path}")
    passed_value = report.get("passed")
    max_rel_error_value = report.get("max_relative_error")
    max_rel_error_text = (
        f"{float(max_rel_error_value):.3e}"
        if max_rel_error_value is not None
        else "n/a"
    )
    print(
        f"parameter={report['parameter_name']} "
        f"passed={passed_value} "
        f"max_rel_error={max_rel_error_text}"
    )


if __name__ == "__main__":
    main()
