from __future__ import annotations

import copy
import dataclasses
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import NEOPAX  # noqa: E402
from NEOPAX._orchestrator import prepare_transport_solver_components  # noqa: E402
from NEOPAX._profiles import AnalyticalProfileModel  # noqa: E402
from NEOPAX._transport_flux_models import PRESSURE_SOURCE_STATE_TO_MW_M3  # noqa: E402
from NEOPAX._transport_solvers import (  # noqa: E402
    _build_prepared_radau_accepted_rollout,
    _build_prepared_radau_execution_context,
    _accepted_step_limit_reached,
    _custom_loop_active,
    _extract_fixed_temperature_projection,
    _extract_state_regularization,
    _make_radau_initial_step_state,
    _make_solver_state_transform,
    _project_flat_state_if_needed,
    _RadauAcceptedStepAttemptContext,
    _radau_adaptive_final_state_rollout,
    _radau_adaptive_schedule_rollout,
    _radau_adaptive_final_y_realized_schedule,
    _radau_apply_accepted_step_map,
    _radau_carry_from_step_state,
    _radau_eval_rhs,
    _radau_solve_on_fixed_time_map_final_state_only,
    _radau_step_fn_forward_solver,
    _radau_step_state_from_carry,
)


DEFAULT_CONFIG = Path(
    "examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_benchmark.toml"
)
ALLOWED_PARAMETERS = {"n0", "T0", "density_shape_power", "temperature_shape_power"}
OBJECTIVE_LABELS = [
    "softmax_Er",
    "smooth_root_proxy",
    "Er2_volume_average",
    "Er_volume_average",
    "electron_temperature_volume_average_keV",
    "total_pressure_volume_average",
    "alpha_power_volume_average_mw_m3",
]


@dataclasses.dataclass(frozen=True, eq=False)
class _ForwardSolverScheduleTrace:
    accepted_mask: Any
    active_mask: Any
    attempted_dts: Any
    step_ts: Any


@dataclasses.dataclass(frozen=True, eq=False)
class _ForwardSolverScheduleRolloutResult:
    trace: _ForwardSolverScheduleTrace
    attempt_count: Any
    accepted_count: Any
    completed: Any
    failed: Any
    fail_code: Any


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


def _fd_step(baseline_value: float, *, rel_step: float, abs_step: float) -> float:
    return float(max(abs_step, abs(rel_step) * max(abs(baseline_value), 1.0)))


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


def _accepted_time_list_from_trace(trace) -> list[float]:
    active_mask = np.asarray(jax.device_get(trace.active_mask), dtype=bool)
    accepted_mask = np.asarray(jax.device_get(trace.accepted_mask), dtype=bool)
    step_ts = np.asarray(jax.device_get(trace.step_ts), dtype=float)
    keep = np.logical_and(active_mask, accepted_mask)
    return step_ts[keep].tolist()


def _adaptive_rollout_diagnostics(rollout) -> dict[str, Any]:
    return {
        "attempt_count": int(np.asarray(jax.device_get(rollout.attempt_count)).item()),
        "accepted_count": int(np.asarray(jax.device_get(rollout.accepted_count)).item()),
        "completed": bool(np.asarray(jax.device_get(rollout.completed)).item()),
        "failed": bool(np.asarray(jax.device_get(rollout.failed)).item()),
        "fail_code": int(np.asarray(jax.device_get(rollout.fail_code)).item()),
    }


def _forward_solver_schedule_rollout(
    execution_context,
    carry0,
    *,
    max_total_steps: int,
    stop_after_accepted_steps: int | None = None,
) -> _ForwardSolverScheduleRolloutResult:
    """Compact accepted-dt trace using the same step dispatcher as the production solver."""

    dtype = execution_context.dtype
    step_state0 = _radau_step_state_from_carry(
        carry0,
        status=jnp.asarray([0, 0, 0], dtype=jnp.int32),
    )
    xs = jnp.arange(int(max_total_steps), dtype=jnp.int32)

    def _scan_body_with_idx(step_state, step_idx):
        active = jnp.logical_and(
            _custom_loop_active(
                step_state,
                execution_context.attempt_context.t_final,
                step_idx,
                max_total_steps,
            ),
            jnp.logical_not(_accepted_step_limit_reached(step_state, stop_after_accepted_steps)),
        )

        def _run(_):
            next_step_state, step_info = _radau_step_fn_forward_solver(execution_context, step_state, None)
            return next_step_state, (
                jnp.asarray(step_info.accepted),
                jnp.asarray(step_info.dt, dtype=dtype),
                jnp.asarray(step_info.t, dtype=dtype),
            )

        def _skip(_):
            return step_state, (
                jnp.asarray(False),
                jnp.asarray(0.0, dtype=dtype),
                step_state.t,
            )

        next_step_state, wrapped_scan_out = jax.lax.cond(active, _run, _skip, operand=None)
        accepted, attempted_dt, step_t = wrapped_scan_out
        return next_step_state, (
            active,
            accepted,
            attempted_dt,
            step_t,
        )

    final_step_state, scan_outputs = jax.lax.scan(_scan_body_with_idx, step_state0, xs)
    active_mask, accepted_mask, attempted_dts, step_ts = scan_outputs
    trace = _ForwardSolverScheduleTrace(
        accepted_mask=accepted_mask,
        active_mask=active_mask,
        attempted_dts=attempted_dts,
        step_ts=step_ts,
    )
    completed = jnp.logical_or(
        final_step_state.t >= (execution_context.attempt_context.t_final - 1.0e-15),
        _accepted_step_limit_reached(final_step_state, stop_after_accepted_steps),
    )
    failed = final_step_state.status[0] != 0
    fail_code = final_step_state.status[1]
    return _ForwardSolverScheduleRolloutResult(
        trace=trace,
        attempt_count=jnp.sum(active_mask.astype(jnp.int32)),
        accepted_count=jnp.sum(accepted_mask.astype(jnp.int32)),
        completed=completed,
        failed=failed,
        fail_code=fail_code,
    )


def _production_solver_baseline_final_state_and_schedule_for_parameter(
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
    prepared_components = prepare_transport_solver_components(config, runtime, state0)
    solver = prepared_components["solver"]
    solve_vector_field = prepared_components["solve_vector_field"]
    solver_output = solver.solve(state0, solve_vector_field, runtime.species)
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
    rollout = _forward_solver_schedule_rollout(
        execution_context,
        prepared_rollout.initial_carry,
        max_total_steps=max_total_steps,
        stop_after_accepted_steps=stop_after_accepted_steps,
    )
    return solver_output["final_state"], rollout


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
    replay_mode_normalized = str(replay_mode).strip().lower()
    if replay_mode_normalized != "accepted":
        raise ValueError(
            "The forward/FD frozen lane now supports replay_mode='accepted' only. "
            "FD runs use the solver-native fixed accepted-time-map path rather than the old replay machinery."
        )

    accepted_time_list = _accepted_time_list_from_trace(frozen_trace)
    return _adaptive_rollout_objectives_for_parameter_on_time_list(
        parameter_value,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        time_list=accepted_time_list,
    )


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
    execution_context = _build_prepared_radau_execution_context(
        solver=solver,
        prepared_rollout=prepared_rollout,
    )
    replay = _radau_solve_on_fixed_time_map_final_state_only(
        prepared_rollout,
        execution_context,
        time_list,
    )
    return _objective_vector(replay["final_state"], runtime), replay


def _accepted_replay_state_debug_for_parameter(
    parameter_value,
    *,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    parameter_name: str,
    accepted_step_limit: int | None = None,
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
    max_total_steps = int(max(1, getattr(solver, "max_steps", 1)))
    stop_after_accepted_steps = (
        int(accepted_step_limit)
        if accepted_step_limit is not None
        else getattr(solver, "stop_after_accepted_steps", None)
    )
    baseline_rollout = _forward_solver_schedule_rollout(
        execution_context,
        prepared_rollout.initial_carry,
        max_total_steps=max_total_steps,
        stop_after_accepted_steps=stop_after_accepted_steps,
    )
    replay_trace = _truncate_rollout_trace_by_accepted_steps(
        baseline_rollout.trace,
        accepted_step_limit,
    )
    accepted_time_list = _accepted_time_list_from_trace(replay_trace)
    accepted_mask = np.asarray(jax.device_get(replay_trace.accepted_mask), dtype=bool)
    active_mask = np.asarray(jax.device_get(replay_trace.active_mask), dtype=bool)
    attempted_dts = np.asarray(jax.device_get(replay_trace.attempted_dts), dtype=float)
    keep_mask = np.logical_and(active_mask, accepted_mask)
    accepted_dts = attempted_dts[keep_mask]

    adaptive_step_state = _radau_step_state_from_carry(
        prepared_rollout.initial_carry,
        status=jnp.asarray([0, 0, 0], dtype=jnp.int32),
    )
    time_list_carry = prepared_rollout.initial_carry
    realized_states = []
    time_list_states = []
    realized_lagged_valid_in = []
    time_list_lagged_valid_in = []

    for active, accepted, dt_value_np in zip(active_mask.tolist(), accepted_mask.tolist(), attempted_dts.tolist()):
        if not active:
            continue
        dt_value = jnp.asarray(dt_value_np, dtype=prepared_rollout.kernel_context.dtype)

        realized_lagged_valid_in.append(
            bool(np.asarray(jax.device_get(adaptive_step_state.lagged_response_valid)).item())
        )
        adaptive_step_state, step_info = _radau_step_fn_forward_solver(
            execution_context,
            dataclasses.replace(adaptive_step_state, dt=dt_value),
            None,
        )
        if accepted:
            realized_states.append(
                prepared_rollout.physics_context.unpack_flat(step_info.y)
            )

    for dt_value_np in accepted_dts:
        dt_value = jnp.asarray(dt_value_np, dtype=prepared_rollout.kernel_context.dtype)
        time_list_lagged_valid_in.append(bool(np.asarray(jax.device_get(time_list_carry.lagged_response_valid)).item()))
        time_attempt_context = _RadauAcceptedStepAttemptContext(
            t_final=time_list_carry.t + dt_value,
            use_transport_lagged_response=jnp.asarray(prepared_rollout.kernel_context.use_transport_lagged_response),
        )
        time_result = _radau_apply_accepted_step_map(
            prepared_rollout.kernel_context,
            prepared_rollout.physics_context,
            dataclasses.replace(time_list_carry, dt=dt_value),
            time_attempt_context,
        )
        time_list_carry = dataclasses.replace(
            time_result.next_carry,
            prev_error=jnp.maximum(
                time_result.err_norm,
                jnp.asarray(1.0e-12, dtype=prepared_rollout.kernel_context.dtype),
            ),
            recent_reject_count=jnp.asarray(0, dtype=jnp.int32),
            regrowth_cooldown=jnp.asarray(0, dtype=jnp.int32),
            easy_growth_streak=jnp.asarray(0, dtype=jnp.int32),
        )
        time_list_states.append(
            prepared_rollout.physics_context.unpack_flat(time_result.accepted_y)
        )
    return {
        "accepted_time_list": accepted_time_list,
        "baseline_rollout": baseline_rollout,
        "replay_trace": replay_trace,
        "realized_saved_states": realized_states,
        "time_list_saved_states": time_list_states,
        "realized_lagged_valid_in": realized_lagged_valid_in,
        "time_list_lagged_valid_in": time_list_lagged_valid_in,
        "realized_final_state": prepared_rollout.physics_context.unpack_flat(adaptive_step_state.y),
        "time_list_final_state": prepared_rollout.physics_context.unpack_flat(time_list_carry.y),
    }
