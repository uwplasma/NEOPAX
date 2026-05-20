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
    _radau_adaptive_final_state_rollout,
    _radau_adaptive_final_y_realized_schedule,
    _radau_apply_accepted_step_map,
    _radau_debug_realized_attempt_replay,
    _build_prepared_radau_execution_context,
    _build_prepared_radau_accepted_rollout,
    _radau_controller_composed_rollout,
    _radau_controller_forward_only_rollout,
    _radau_prepare_stage_subsolve_inputs_from_carry,
    _radau_run_stage_subsolve_standalone_autodiff,
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
DEFAULT_FD_SWEEP_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)
STANDALONE_SUBSOLVE_LABELS = [
    "stage_sum",
    "stage_l2_norm",
    "final_residual_norm",
    "theta_final",
]
DEFAULT_SMALL_STEP_COUNTS = (2, 3, 5)


def _prepare_benchmark_config(config_path: Path, *, device: str | None) -> dict[str, Any]:
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
        stop_after_accepted_steps = getattr(solver, "stop_after_accepted_steps", None)
        final_y = _radau_adaptive_final_y_realized_schedule(
            execution_context,
            max_total_steps,
            stop_after_accepted_steps,
            prepared_rollout.initial_carry,
        )
        final_state = prepared_rollout.physics_context.unpack_flat(final_y)
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
        stop_after_accepted_steps = getattr(solver, "stop_after_accepted_steps", None)
        rollout = _radau_adaptive_final_state_rollout(
            execution_context,
            prepared_rollout.initial_carry,
            max_total_steps=max_total_steps,
            stop_after_accepted_steps=stop_after_accepted_steps,
        )
        final_state = prepared_rollout.physics_context.unpack_flat(rollout.final_carry.y)
    return final_state, rollout


def _adaptive_rollout_objectives_for_parameter(
    parameter_value,
    *,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    parameter_name: str,
    use_realized_schedule_jvp: bool = False,
):
    final_state, rollout = _adaptive_rollout_final_state_for_parameter(
        parameter_value,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        use_realized_schedule_jvp=use_realized_schedule_jvp,
    )
    return _objective_vector(final_state, runtime), rollout


def _adaptive_rollout_nan_debug_for_parameter(
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
    return {
        "first_bad_index": int(debug.first_bad_index),
        "first_bad_was_accepted": bool(debug.first_bad_was_accepted),
        "first_bad_dt": float(debug.first_bad_dt),
        "final_tangent_finite": bool(debug.final_tangent_finite),
        "tangent_finite_mask": list(debug.tangent_finite_mask),
        "attempted_dts": _adaptive_rollout_diagnostics(rollout)["attempted_dts"],
        "accepted_mask": _adaptive_rollout_diagnostics(rollout)["accepted_mask"],
        "err_norms": _adaptive_rollout_diagnostics(rollout)["err_norms"],
    }


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
    objectives_minus, minus_rollout = _adaptive_rollout_objectives_for_parameter(
        jnp.asarray(minus_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        use_realized_schedule_jvp=True,
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

    gradient_ad = jax.jacfwd(objective_fn)(jnp.asarray(baseline_value))
    gradient_fd = (objectives_plus - objectives_minus) / (2.0 * fd_step)

    grad_ad_np = np.asarray(jax.device_get(gradient_ad), dtype=float)
    grad_fd_np = np.asarray(jax.device_get(gradient_fd), dtype=float)
    abs_err = np.abs(grad_ad_np - grad_fd_np)
    rel_err = abs_err / np.maximum(np.abs(grad_fd_np), 1.0e-10)

    baseline_diag = _adaptive_rollout_diagnostics(baseline_rollout)
    minus_diag = _adaptive_rollout_diagnostics(minus_rollout)
    plus_diag = _adaptive_rollout_diagnostics(plus_rollout)
    nan_debug = None
    if not np.all(np.isfinite(grad_ad_np)):
        nan_debug = _adaptive_rollout_nan_debug_for_parameter(
            jnp.asarray(baseline_value),
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=parameter_name,
        )

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

    if args.realized_schedule_rollout_check:
        report = build_realized_schedule_rollout_report(
            config_path=args.config,
            parameter_name=args.parameter,
            rel_fd_step=args.fd_rel_step,
            abs_fd_step=args.fd_abs_step,
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
    print(
        f"parameter={report['parameter_name']} "
        f"passed={report['passed']} "
        f"max_rel_error={report['max_relative_error']:.3e}"
    )


if __name__ == "__main__":
    main()
