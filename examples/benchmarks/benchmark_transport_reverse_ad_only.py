from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_transport_forward_fd_lane import (  # noqa: E402
    DEFAULT_CONFIG,
    OBJECTIVE_LABELS,
    _adaptive_rollout_diagnostics,
    _baseline_profile_cfg,
    _objective_vector,
    _parameterized_profile_set,
    _prepare_benchmark_config,
)
from NEOPAX._orchestrator import build_runtime_context  # noqa: E402
from NEOPAX._orchestrator import prepare_transport_solver_components  # noqa: E402
from NEOPAX._transport_solvers import (  # noqa: E402
    _build_prepared_radau_accepted_rollout,
    _build_prepared_radau_execution_context,
    _extract_fixed_temperature_projection,
    _extract_state_regularization,
    _make_radau_initial_step_state,
    _make_solver_state_transform,
    _project_flat_state_if_needed,
    _radau_adaptive_final_y_realized_schedule_vjp,
    _radau_adaptive_schedule_rollout,
    _radau_carry_from_step_state,
    _radau_debug_local_accepted_step_transpose,
    _radau_eval_rhs,
)


PARAMETER_ORDER = ("n0", "T0", "density_shape_power", "temperature_shape_power")


@dataclasses.dataclass(frozen=True)
class _ReverseStaticSetup:
    solver: object
    solve_vector_field: object
    prepared_rollout: object
    execution_context: object
    stop_after_accepted_steps: int | None
    max_total_steps: int
    reverse_segment_length: int | None


def _report_path(objective_name: str) -> Path:
    outdir = ROOT / "outputs" / "autodiff_transport_lagged_ntx" / "reverse_ad"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / f"transport_reverse_ad_only_{objective_name}.json"


def _initial_state_for_parameter_vector(
    parameter_values,
    *,
    baseline_state,
    profile_cfg: dict,
    runtime,
):
    cfg = dict(profile_cfg)
    for name, value in zip(PARAMETER_ORDER, parameter_values):
        cfg[name] = value
    profile_set = _parameterized_profile_set(
        cfg,
        runtime.geometry,
        runtime.species.number_species,
        parameter_name=PARAMETER_ORDER[0],
        parameter_value=cfg[PARAMETER_ORDER[0]],
    )
    density_state = jnp.asarray(profile_set.density, dtype=baseline_state.density.dtype) / 1.0e20
    temperature_state = jnp.asarray(profile_set.temperature, dtype=baseline_state.pressure.dtype) / 1.0e3
    pressure_state = density_state * temperature_state
    return dataclasses.replace(
        baseline_state,
        density=density_state,
        pressure=pressure_state,
    )


def _add_trees(lhs, rhs):
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs
    return jax.tree_util.tree_map(lambda a, b: a + b, lhs, rhs)


def _lagged_response_pullback_from_owner(solve_vector_field):
    owner = getattr(solve_vector_field, "__self__", None)
    if owner is None:
        return None
    pullback_fn = getattr(owner, "pullback_build_lagged_response", None)
    return pullback_fn if callable(pullback_fn) else None


def _reverse_initial_carry_from_state_with_static_setup(
    *,
    solver,
    state,
    solve_vector_field,
    species,
    prepared_rollout_static,
):
    """Build the initial carry with a reverse-local model-aware lagged pullback."""

    temperature_active_mask, fixed_temperature_profile = _extract_fixed_temperature_projection(solve_vector_field)
    density_floor, temperature_floor = _extract_state_regularization(solve_vector_field)
    kernel_context = prepared_rollout_static.kernel_context
    physics_context = prepared_rollout_static.physics_context
    initial_carry_static = prepared_rollout_static.initial_carry
    lagged_pullback_fn = _lagged_response_pullback_from_owner(solve_vector_field)

    def _flat_state_from_state(state_value):
        flat_state, *_ = _make_solver_state_transform(
            state_value,
            species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
            density_floor=density_floor,
            temperature_floor=temperature_floor,
        )
        return flat_state

    def _build_state_from_flat(flat_value, unpack_flat, project_flat):
        return unpack_flat(_project_flat_state_if_needed(flat_value, project_flat))

    @jax.custom_vjp
    def _build_initial_carry(state_value):
        flat_state0, unpack_flat, _unpack_packed, _pack_state, project_flat = _make_solver_state_transform(
            state_value,
            species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
            density_floor=density_floor,
            temperature_floor=temperature_floor,
        )
        lagged_state0 = _build_state_from_flat(flat_state0, unpack_flat, project_flat)
        initial_lagged_response = (
            physics_context.build_lagged_response(lagged_state0)
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

    def _build_initial_carry_fwd(state_value):
        flat_state0, unpack_flat, _unpack_packed, _pack_state, project_flat = _make_solver_state_transform(
            state_value,
            species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
            density_floor=density_floor,
            temperature_floor=temperature_floor,
        )
        lagged_state0 = _build_state_from_flat(flat_state0, unpack_flat, project_flat)
        initial_lagged_response = (
            physics_context.build_lagged_response(lagged_state0)
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
        carry0 = _radau_carry_from_step_state(step_state0)
        residual = (state_value, flat_state0, lagged_state0, initial_lagged_response)
        return carry0, residual

    def _build_initial_carry_bwd(residual, carry_bar):
        state_value, flat_state0, lagged_state0, initial_lagged_response = residual
        _, unpack_flat, _unpack_packed, _pack_state, project_flat = _make_solver_state_transform(
            state_value,
            species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
            density_floor=density_floor,
            temperature_floor=temperature_floor,
        )
        flat_bar = jnp.asarray(carry_bar.y)
        flat_bar = flat_bar + jnp.asarray(carry_bar.lagged_reference_y)

        prev_stages_bar = jnp.asarray(carry_bar.prev_stages).reshape((kernel_context.num_stages, -1))
        rhs_bar = jnp.sum(prev_stages_bar, axis=0)
        lagged_bar = carry_bar.lagged_response_cache

        def _rhs_state_pullback_fallback(lagged_response_value):
            def _rhs_from_flat(flat_value):
                return _radau_eval_rhs(
                    initial_carry_static.t,
                    flat_value,
                    lagged_response_value,
                    physics_context.flat_rhs,
                    physics_context.flat_rhs_with_lagged_response,
                )

            _, rhs_pullback = jax.vjp(_rhs_from_flat, flat_state0)
            (rhs_flat_bar,) = rhs_pullback(rhs_bar)
            return rhs_flat_bar

        if physics_context.flat_rhs_state_pullback is not None:
            rhs_flat_bar = physics_context.flat_rhs_state_pullback(
                initial_carry_static.t,
                flat_state0,
                initial_lagged_response,
                rhs_bar,
            )
            if project_flat is not None:
                _, project_pullback = jax.vjp(project_flat, flat_state0)
                (rhs_flat_bar,) = project_pullback(rhs_flat_bar)
        else:
            rhs_flat_bar = _rhs_state_pullback_fallback(initial_lagged_response)
        flat_bar = flat_bar + rhs_flat_bar

        if initial_lagged_response is not None:
            def _rhs_from_flat_and_lagged(flat_value, lagged_value):
                return _radau_eval_rhs(
                    initial_carry_static.t,
                    flat_value,
                    lagged_value,
                    physics_context.flat_rhs,
                    physics_context.flat_rhs_with_lagged_response,
                )

            if physics_context.flat_rhs_lagged_response_pullback is not None:
                rhs_lagged_bar = physics_context.flat_rhs_lagged_response_pullback(
                    initial_carry_static.t,
                    flat_state0,
                    initial_lagged_response,
                    rhs_bar,
                )
            else:
                _, rhs_pullback = jax.vjp(_rhs_from_flat_and_lagged, flat_state0, initial_lagged_response)
                _rhs_flat_bar_unused, rhs_lagged_bar = rhs_pullback(rhs_bar)
            lagged_bar = _add_trees(lagged_bar, rhs_lagged_bar)

            if lagged_pullback_fn is not None:
                lagged_state_bar = lagged_pullback_fn(lagged_state0, lagged_bar)
            else:
                def _build_lagged_from_state(lagged_state_value):
                    return physics_context.build_lagged_response(lagged_state_value)

                _, lagged_pullback = jax.vjp(_build_lagged_from_state, lagged_state0)
                (lagged_state_bar,) = lagged_pullback(lagged_bar)

            def _lagged_state_from_flat(flat_value):
                return _build_state_from_flat(flat_value, unpack_flat, project_flat)

            _, lagged_state_flat_pullback = jax.vjp(_lagged_state_from_flat, flat_state0)
            (lagged_flat_bar,) = lagged_state_flat_pullback(lagged_state_bar)
            flat_bar = flat_bar + lagged_flat_bar

        _, state_pullback = jax.vjp(_flat_state_from_state, state_value)
        (state_bar,) = state_pullback(flat_bar)
        return (state_bar,)

    _build_initial_carry.defvjp(_build_initial_carry_fwd, _build_initial_carry_bwd)
    return _build_initial_carry(state)


def _reverse_objective_for_parameter_vector(
    parameter_values,
    *,
    runtime,
    baseline_state,
    profile_cfg: dict,
    objective_index: int,
    reverse_setup: _ReverseStaticSetup,
):
    state0 = _initial_state_for_parameter_vector(
        parameter_values,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        runtime=runtime,
    )
    initial_carry = _reverse_initial_carry_from_state_with_static_setup(
        solver=reverse_setup.solver,
        state=state0,
        solve_vector_field=reverse_setup.solve_vector_field,
        species=runtime.species,
        prepared_rollout_static=reverse_setup.prepared_rollout,
    )
    final_y = _radau_adaptive_final_y_realized_schedule_vjp(
        reverse_setup.execution_context,
        reverse_setup.max_total_steps,
        reverse_setup.stop_after_accepted_steps,
        reverse_setup.reverse_segment_length,
        initial_carry,
    )
    final_state = reverse_setup.prepared_rollout.physics_context.unpack_flat(final_y)
    return _objective_vector(final_state, runtime)[objective_index]


def _prepare_reverse_static_setup(
    parameter_values,
    *,
    config: dict,
    runtime,
    baseline_state,
    profile_cfg: dict,
    accepted_step_limit_override: int | None = None,
    reverse_segment_length: int | None = None,
    reverse_direct_stage_adjoint: bool = False,
    reverse_stage_adjoint_solve_mode: str = "structured",
    reverse_rhs_transpose_mode: str = "generic",
    reverse_stage_cotangent_mode: str = "full",
    reverse_stage_adjoint_iter_maxiter: int = 40,
    reverse_stage_adjoint_iter_tol: float = 1.0e-10,
) -> _ReverseStaticSetup:
    state0_static = _initial_state_for_parameter_vector(
        parameter_values,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        runtime=runtime,
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
    if reverse_direct_stage_adjoint:
        execution_context = dataclasses.replace(
            execution_context,
            physics_context=dataclasses.replace(
                execution_context.physics_context,
                reverse_direct_stage_adjoint=True,
                reverse_stage_adjoint_solve_mode=str(reverse_stage_adjoint_solve_mode),
                reverse_rhs_transpose_mode=str(reverse_rhs_transpose_mode),
                reverse_stage_cotangent_mode=str(reverse_stage_cotangent_mode),
                reverse_stage_adjoint_iter_maxiter=int(reverse_stage_adjoint_iter_maxiter),
                reverse_stage_adjoint_iter_tol=float(reverse_stage_adjoint_iter_tol),
            ),
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
        schedule_probe = _radau_adaptive_schedule_rollout(
            execution_context,
            prepared_rollout_static.initial_carry,
            max_total_steps=max_total_steps,
            stop_after_accepted_steps=stop_after_accepted_steps,
        )
        actual_attempt_count = int(np.asarray(jax.device_get(schedule_probe.attempt_count)))
        max_total_steps = min(
            max_total_steps,
            max(actual_attempt_count + 2, int(stop_after_accepted_steps)),
        )
        accepted_limit = int(stop_after_accepted_steps)
        active_mask_np = np.asarray(jax.device_get(schedule_probe.trace.active_mask), dtype=bool)
        accepted_mask_np = np.asarray(jax.device_get(schedule_probe.trace.accepted_mask), dtype=bool)
        next_lagged_valid_np = np.asarray(
            jax.device_get(schedule_probe.trace.next_lagged_response_valid),
            dtype=bool,
        )
        accepted_positions = np.nonzero(np.logical_and(active_mask_np, accepted_mask_np))[0][:accepted_limit]
        incoming_valid = bool(np.asarray(jax.device_get(prepared_rollout_static.initial_carry.lagged_response_valid)))
        lagged_branch_schedule: list[bool] = []
        for accepted_position in accepted_positions:
            lagged_branch_schedule.append(bool(incoming_valid))
            incoming_valid = bool(next_lagged_valid_np[int(accepted_position)])
        if len(lagged_branch_schedule) < accepted_limit:
            lagged_branch_schedule.extend([bool(incoming_valid)] * (accepted_limit - len(lagged_branch_schedule)))
        execution_context = dataclasses.replace(
            execution_context,
            physics_context=dataclasses.replace(
                execution_context.physics_context,
                reverse_lagged_branch_schedule=tuple(lagged_branch_schedule),
            ),
        )
    return _ReverseStaticSetup(
        solver=solver,
        solve_vector_field=solve_vector_field_static,
        prepared_rollout=prepared_rollout_static,
        execution_context=execution_context,
        stop_after_accepted_steps=stop_after_accepted_steps,
        max_total_steps=max_total_steps,
        reverse_segment_length=reverse_segment_length,
    )


def _baseline_rollout_for_diagnostics(
    parameter_values,
    *,
    config: dict,
    runtime,
    baseline_state,
    profile_cfg: dict,
    accepted_step_limit_override: int | None = None,
):
    state0 = _initial_state_for_parameter_vector(
        parameter_values,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        runtime=runtime,
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
    return _radau_adaptive_schedule_rollout(
        execution_context,
        prepared_rollout.initial_carry,
        max_total_steps=max_total_steps,
        stop_after_accepted_steps=stop_after_accepted_steps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reverse-only adaptive benchmark lane using the current reverse-capable realized-schedule helper."
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="Benchmark TOML.")
    parser.add_argument(
        "--objective",
        type=str,
        default="softmax_Er",
        choices=OBJECTIVE_LABELS,
        help="Scalar objective for reverse mode. One run returns all profile-parameter gradients.",
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
        "--baseline-diagnostics",
        action="store_true",
        help="Run an extra primal schedule rollout to print attempt/accepted counts before reverse AD.",
    )
    parser.add_argument(
        "--reverse-segment-length",
        type=int,
        default=None,
        help=(
            "Optional reverse checkpoint segment length for accepted-step replay. "
            "Omit for the current unsegmented reference path."
        ),
    )
    parser.add_argument(
        "--reverse-direct-stage-adjoint",
        action="store_true",
        help=(
            "Use the reverse-only structured accepted-step adjoint. This is the default; "
            "the flag is kept as an explicit marker for old command lines."
        ),
    )
    parser.add_argument(
        "--reverse-transpose-fallback",
        action="store_true",
        help=(
            "Use the older transpose-of-forward-tangent helper instead of the "
            "structured reverse accepted-step adjoint. Intended only for comparisons."
        ),
    )
    parser.add_argument(
        "--reverse-stage-adjoint-solve-mode",
        choices=("structured", "bicgstab", "block", "gmres"),
        default="structured",
        help=(
            "Reverse stage-adjoint linear solve. 'structured' uses the Radau "
            "transformed LU transpose approximation and is the lightweight default; "
            "'bicgstab' is the lower-memory exact iterative candidate; 'block' and "
            "'gmres' are correctness oracles but are memory/compile heavy."
        ),
    )
    parser.add_argument(
        "--reverse-rhs-transpose-mode",
        choices=("generic", "explicit_ntx_interpolated"),
        default="generic",
        help=(
            "RHS-state transpose used inside exact reverse stage-adjoint matvecs. "
            "'generic' is the known-good JAX VJP reference; "
            "'explicit_ntx_interpolated' opts into the experimental explicit NTX state pullback."
        ),
    )
    parser.add_argument(
        "--reverse-stage-cotangent-mode",
        choices=(
            "full",
            "zero_lagged",
            "zero_rhs_state",
            "zero_rhs_direct",
            "zero_rhs_flux",
            "zero_stage_solve",
            "zero_step_bwd",
            "force_reuse_bwd",
            "force_rebuild_bwd",
            "branch_schedule_bwd",
            "dynamic_call_bwd",
        ),
        default="full",
        help=(
            "Diagnostic-only branch toggle for exact stage adjoints. 'full' is the normal "
            "reverse lane; 'zero_lagged' drops stage lagged-response cotangents; "
            "'zero_rhs_state' drops stage RHS-state cotangents, including inside the exact "
            "iterative transpose matvec; 'zero_rhs_direct' keeps only shared-flux state "
            "cotangents; 'zero_rhs_flux' keeps only direct equation-assembly state "
            "cotangents; 'zero_stage_solve' bypasses the exact stage-adjoint solve and "
            "residual-input pullback; 'zero_step_bwd' bypasses the accepted-step backward "
            "body inside segmented replay; 'force_reuse_bwd' and 'force_rebuild_bwd' "
            "compile only one lagged-response backward branch for diagnosis. Non-full "
            "modes intentionally change gradients unless the forced branch matches the "
            "realized primal branch for every accepted step; 'branch_schedule_bwd' uses "
            "the baseline realized accepted-step branch schedule statically; "
            "'dynamic_call_bwd' keeps the dynamic branch but puts each branch body behind "
            "a non-inlined compiled call boundary."
        ),
    )
    parser.add_argument(
        "--reverse-stage-adjoint-iter-maxiter",
        type=int,
        default=40,
        help=(
            "Maximum Krylov iterations for exact iterative reverse stage-adjoint "
            "modes ('bicgstab'/'gmres'). Defaults to the current conservative value."
        ),
    )
    parser.add_argument(
        "--reverse-stage-adjoint-iter-tol",
        type=float,
        default=1.0e-10,
        help=(
            "Relative tolerance for exact iterative reverse stage-adjoint modes "
            "('bicgstab'/'gmres'). Defaults to the current conservative value."
        ),
    )
    parser.add_argument(
        "--timing-mode",
        choices=("eager", "jit-warm"),
        default="eager",
        help=(
            "Timing harness. 'eager' preserves the original un-jitted grad timing; "
            "'jit-warm' reports first jit call and second warm execute call separately."
        ),
    )
    parser.add_argument(
        "--warm-repeats",
        type=int,
        default=1,
        help="Number of post-compile warm executions to time when --timing-mode=jit-warm.",
    )
    parser.add_argument(
        "--local-transpose-diagnostic-accepted-step",
        type=int,
        default=None,
        help=(
            "Diagnostic-only mode: run one local accepted-step dot-product transpose "
            "check at this zero-based accepted-step ordinal, then exit."
        ),
    )
    parser.add_argument(
        "--local-transpose-diagnostic-seed-mode",
        type=str,
        default="y",
        choices=(
            "y",
            "prev_stages",
            "lagged_cache",
            "lagged_reference",
            "y_lagged_cache",
            "y_lagged_reference",
            "lagged_cache_reference",
            "all",
        ),
        help="Seed channel for --local-transpose-diagnostic-accepted-step.",
    )
    parser.add_argument(
        "--local-transpose-diagnostic-input-seed-mode",
        type=str,
        default=None,
        choices=(
            "y",
            "prev_stages",
            "lagged_cache",
            "lagged_reference",
            "y_lagged_cache",
            "y_lagged_reference",
            "lagged_cache_reference",
            "all",
        ),
        help="Optional input tangent seed channel. Defaults to --local-transpose-diagnostic-seed-mode.",
    )
    parser.add_argument(
        "--local-transpose-diagnostic-output-seed-mode",
        type=str,
        default=None,
        choices=(
            "y",
            "prev_stages",
            "lagged_cache",
            "lagged_reference",
            "y_lagged_cache",
            "y_lagged_reference",
            "lagged_cache_reference",
            "all",
        ),
        help="Optional output cotangent seed channel. Defaults to --local-transpose-diagnostic-seed-mode.",
    )
    args = parser.parse_args()
    if int(args.reverse_stage_adjoint_iter_maxiter) <= 0:
        raise SystemExit("[autodiff-gate] --reverse-stage-adjoint-iter-maxiter must be positive.")
    if float(args.reverse_stage_adjoint_iter_tol) <= 0.0:
        raise SystemExit("[autodiff-gate] --reverse-stage-adjoint-iter-tol must be positive.")
    if (
        str(args.reverse_rhs_transpose_mode) == "explicit_ntx_interpolated"
        and str(args.reverse_stage_adjoint_solve_mode) == "gmres"
    ):
        raise SystemExit(
            "[autodiff-gate] --reverse-rhs-transpose-mode explicit_ntx_interpolated is not ready for "
            "JAX scipy GMRES. Use bicgstab for this experimental mode while the NTX RHS-state "
            "transpose is being specialized."
        )
    reverse_segment_length = None
    if args.reverse_segment_length is not None:
        reverse_segment_length = int(args.reverse_segment_length)
        if reverse_segment_length <= 0:
            raise SystemExit("[autodiff-gate] --reverse-segment-length must be positive when provided.")
    reverse_direct_stage_adjoint = not bool(args.reverse_transpose_fallback)

    config = _prepare_benchmark_config(
        Path(args.config),
        device=args.device,
        ntx_exact_derivative_mode=args.ntx_exact_derivative_mode,
        radau_jacobian_reuse_mode=args.radau_jacobian_reuse_mode,
    )
    runtime, baseline_state = build_runtime_context(config)
    profile_cfg = _baseline_profile_cfg(config)
    baseline_values = jnp.asarray(
        [float(profile_cfg[name]) for name in PARAMETER_ORDER],
        dtype=jnp.asarray(baseline_state.pressure).dtype,
    )
    objective_index = OBJECTIVE_LABELS.index(args.objective)
    reverse_setup = _prepare_reverse_static_setup(
        baseline_values,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        accepted_step_limit_override=args.accepted_step_limit,
        reverse_segment_length=reverse_segment_length,
        reverse_direct_stage_adjoint=reverse_direct_stage_adjoint,
        reverse_stage_adjoint_solve_mode=str(args.reverse_stage_adjoint_solve_mode),
        reverse_rhs_transpose_mode=str(args.reverse_rhs_transpose_mode),
        reverse_stage_cotangent_mode=str(args.reverse_stage_cotangent_mode),
        reverse_stage_adjoint_iter_maxiter=int(args.reverse_stage_adjoint_iter_maxiter),
        reverse_stage_adjoint_iter_tol=float(args.reverse_stage_adjoint_iter_tol),
    )

    if args.local_transpose_diagnostic_accepted_step is not None:
        accepted_step_index = int(args.local_transpose_diagnostic_accepted_step)
        if accepted_step_index < 0:
            raise SystemExit("[autodiff-gate] --local-transpose-diagnostic-accepted-step must be >= 0.")
        print("[autodiff-gate] progress: running local accepted-step transpose diagnostic", flush=True)
        baseline_rollout = _radau_adaptive_schedule_rollout(
            reverse_setup.execution_context,
            reverse_setup.prepared_rollout.initial_carry,
            max_total_steps=reverse_setup.max_total_steps,
            stop_after_accepted_steps=reverse_setup.stop_after_accepted_steps,
        )
        diagnostic = _radau_debug_local_accepted_step_transpose(
            reverse_setup.execution_context,
            reverse_setup.prepared_rollout.initial_carry,
            baseline_rollout.trace,
            accepted_step_index=accepted_step_index,
            seed_mode=args.local_transpose_diagnostic_seed_mode,
            input_seed_mode=args.local_transpose_diagnostic_input_seed_mode,
            output_seed_mode=args.local_transpose_diagnostic_output_seed_mode,
        )
        diagnostic = jax.device_get(diagnostic)
        input_seed_mode = (
            args.local_transpose_diagnostic_seed_mode
            if args.local_transpose_diagnostic_input_seed_mode is None
            else args.local_transpose_diagnostic_input_seed_mode
        )
        output_seed_mode = (
            args.local_transpose_diagnostic_seed_mode
            if args.local_transpose_diagnostic_output_seed_mode is None
            else args.local_transpose_diagnostic_output_seed_mode
        )
        report = {
            "mode": "transport_reverse_ad_only_local_transpose_diagnostic",
            "config_path": str(Path(args.config)),
            "objective_name": args.objective,
            "accepted_step_limit": None if args.accepted_step_limit is None else int(args.accepted_step_limit),
            "diagnostic_accepted_step_index": accepted_step_index,
            "diagnostic_seed_mode": str(args.local_transpose_diagnostic_seed_mode),
            "diagnostic_input_seed_mode": str(input_seed_mode),
            "diagnostic_output_seed_mode": str(output_seed_mode),
            "target_attempt_index": int(diagnostic.target_attempt_index),
            "found_target": bool(diagnostic.found_target),
            "lagged_response_valid_in": bool(diagnostic.lagged_response_valid_in),
            "local_branch_reuse": bool(diagnostic.local_branch_reuse),
            "lhs_v_dot_ju": float(diagnostic.lhs_v_dot_ju),
            "rhs_jtv_dot_u": float(diagnostic.rhs_jtv_dot_u),
            "abs_err": float(diagnostic.abs_err),
            "rel_err": float(diagnostic.rel_err),
        }
        print(
            "[autodiff-gate] local transpose diagnostic: "
            f"accepted_step_index={accepted_step_index} "
            f"seed_mode={args.local_transpose_diagnostic_seed_mode} "
            f"input_seed_mode={input_seed_mode} "
            f"output_seed_mode={output_seed_mode} "
            f"target_attempt_index={report['target_attempt_index']} "
            f"found_target={report['found_target']} "
            f"lagged_response_valid_in={report['lagged_response_valid_in']} "
            f"local_branch_reuse={report['local_branch_reuse']}"
        )
        print(
            "[autodiff-gate] local transpose diagnostic values: "
            f"lhs_v_dot_ju={report['lhs_v_dot_ju']:.6e} "
            f"rhs_jtv_dot_u={report['rhs_jtv_dot_u']:.6e} "
            f"abs_err={report['abs_err']:.6e} "
            f"rel_err={report['rel_err']:.6e}"
        )
        outpath = _report_path(args.objective)
        outpath.write_text(json.dumps(report, indent=2))
        print(f"Wrote {outpath.relative_to(ROOT)}")
        return

    baseline_diag = None
    if args.baseline_diagnostics:
        print("[autodiff-gate] progress: running baseline adaptive rollout for reverse AD lane", flush=True)
        baseline_rollout = _baseline_rollout_for_diagnostics(
            baseline_values,
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            accepted_step_limit_override=args.accepted_step_limit,
        )
        baseline_diag = _adaptive_rollout_diagnostics(baseline_rollout)

    objective_fn = lambda p: _reverse_objective_for_parameter_vector(  # noqa: E731
        p,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        objective_index=objective_index,
        reverse_setup=reverse_setup,
    )

    print("[autodiff-gate] progress: running reverse custom-VJP", flush=True)
    reverse_compile_plus_execute_s = None
    reverse_execute_s = None
    reverse_execute_times_s: list[float] = []
    t_reverse_start = time.perf_counter()
    if args.timing_mode == "jit-warm":
        grad_fn = jax.jit(jax.grad(objective_fn))
        first_gradient = grad_fn(baseline_values)
        first_gradient = jax.block_until_ready(first_gradient)
        reverse_compile_plus_execute_s = time.perf_counter() - t_reverse_start

        gradient_rev = first_gradient
        for _ in range(max(1, int(args.warm_repeats))):
            t_execute_start = time.perf_counter()
            gradient_rev = grad_fn(baseline_values)
            gradient_rev = jax.block_until_ready(gradient_rev)
            reverse_execute_times_s.append(time.perf_counter() - t_execute_start)
        reverse_execute_s = float(np.mean(reverse_execute_times_s))
        reverse_total_s = reverse_compile_plus_execute_s + float(np.sum(reverse_execute_times_s))
    else:
        gradient_rev = jax.grad(objective_fn)(baseline_values)
        gradient_rev = jax.block_until_ready(gradient_rev)
        reverse_total_s = time.perf_counter() - t_reverse_start
    grad_np = np.asarray(jax.device_get(gradient_rev), dtype=float)
    reverse_checkpoint_count = None
    if reverse_segment_length is not None:
        reverse_checkpoint_base = (
            int(args.accepted_step_limit)
            if args.accepted_step_limit is not None
            else int(reverse_setup.max_total_steps)
        )
        reverse_checkpoint_count = int(
            (reverse_checkpoint_base + int(reverse_segment_length) - 1)
            // int(reverse_segment_length)
        )
    reverse_lagged_branch_schedule = getattr(
        reverse_setup.execution_context.physics_context,
        "reverse_lagged_branch_schedule",
        None,
    )
    reverse_lagged_reuse_count = None
    reverse_lagged_rebuild_count = None
    if reverse_lagged_branch_schedule is not None:
        reverse_lagged_reuse_count = int(sum(bool(value) for value in reverse_lagged_branch_schedule))
        reverse_lagged_rebuild_count = int(len(reverse_lagged_branch_schedule) - reverse_lagged_reuse_count)

    report = {
        "mode": "transport_reverse_ad_only",
        "config_path": str(Path(args.config)),
        "objective_name": args.objective,
        "parameter_order": list(PARAMETER_ORDER),
        "baseline_values": np.asarray(jax.device_get(baseline_values), dtype=float).tolist(),
        "accepted_step_limit": None if args.accepted_step_limit is None else int(args.accepted_step_limit),
        "max_total_steps": int(reverse_setup.max_total_steps),
        "reverse_checkpoint_count": reverse_checkpoint_count,
        "ntx_exact_derivative_mode": str(args.ntx_exact_derivative_mode),
        "radau_jacobian_reuse_mode": None if args.radau_jacobian_reuse_mode is None else str(args.radau_jacobian_reuse_mode),
        "reverse_segment_length": reverse_segment_length,
        "reverse_lagged_reuse_count": reverse_lagged_reuse_count,
        "reverse_lagged_rebuild_count": reverse_lagged_rebuild_count,
        "reverse_direct_stage_adjoint": bool(reverse_direct_stage_adjoint),
        "reverse_stage_adjoint_solve_mode": str(args.reverse_stage_adjoint_solve_mode),
        "reverse_rhs_transpose_mode": str(args.reverse_rhs_transpose_mode),
        "reverse_stage_cotangent_mode": str(args.reverse_stage_cotangent_mode),
        "reverse_stage_adjoint_iter_maxiter": int(args.reverse_stage_adjoint_iter_maxiter),
        "reverse_stage_adjoint_iter_tol": float(args.reverse_stage_adjoint_iter_tol),
        "reverse_transpose_fallback": bool(args.reverse_transpose_fallback),
        "timing_mode": str(args.timing_mode),
        "reverse_total_s": float(reverse_total_s),
        "reverse_compile_plus_execute_s": None if reverse_compile_plus_execute_s is None else float(reverse_compile_plus_execute_s),
        "reverse_execute_s": None if reverse_execute_s is None else float(reverse_execute_s),
        "reverse_execute_times_s": [float(value) for value in reverse_execute_times_s],
        "gradient_reverse_ad": grad_np.tolist(),
        "rollout_path": {
            "baseline": baseline_diag,
        },
    }

    print(
        f"[autodiff-gate] mode=transport_reverse_ad_only objective={args.objective} "
        f"parameters={list(PARAMETER_ORDER)} "
        f"radau_jacobian_reuse_mode={args.radau_jacobian_reuse_mode} "
        f"max_total_steps={reverse_setup.max_total_steps} "
        f"reverse_checkpoint_count={reverse_checkpoint_count} "
        f"reverse_segment_length={reverse_segment_length} "
        f"reverse_lagged_reuse_count={reverse_lagged_reuse_count} "
        f"reverse_lagged_rebuild_count={reverse_lagged_rebuild_count} "
        f"reverse_direct_stage_adjoint={bool(reverse_direct_stage_adjoint)} "
        f"reverse_stage_adjoint_solve_mode={args.reverse_stage_adjoint_solve_mode} "
        f"reverse_rhs_transpose_mode={args.reverse_rhs_transpose_mode} "
        f"reverse_stage_cotangent_mode={args.reverse_stage_cotangent_mode} "
        f"reverse_stage_adjoint_iter_maxiter={args.reverse_stage_adjoint_iter_maxiter} "
        f"reverse_stage_adjoint_iter_tol={args.reverse_stage_adjoint_iter_tol:.6e} "
        f"timing_mode={args.timing_mode} "
        f"reverse_total_s={reverse_total_s:.6e}"
    )
    if reverse_compile_plus_execute_s is not None:
        print(
            f"[autodiff-gate] timing reverse_compile_plus_execute_s={reverse_compile_plus_execute_s:.6e} "
            f"reverse_execute_s_mean={reverse_execute_s:.6e} "
            f"reverse_execute_s_min={min(reverse_execute_times_s):.6e} "
            f"reverse_execute_repeats={len(reverse_execute_times_s)}"
        )
        print(
            "[autodiff-gate] timing reverse_execute_times_s="
            + ",".join(f"{float(value):.6e}" for value in reverse_execute_times_s)
        )
    if baseline_diag is not None:
        print(
            f"[autodiff-gate] rollout baseline: attempt_count={baseline_diag.get('attempt_count')} "
            f"accepted_count={baseline_diag.get('accepted_count')} "
            f"completed={baseline_diag.get('completed')} failed={baseline_diag.get('failed')} "
            f"fail_code={baseline_diag.get('fail_code')}"
        )
    print("[autodiff-gate] reverse gradients:")
    for name, value in zip(PARAMETER_ORDER, grad_np.tolist()):
        print(f"  - d{args.objective}/d{name}: rev={float(value):.6e}")
    outpath = _report_path(args.objective)
    outpath.write_text(json.dumps(report, indent=2))
    print(f"Wrote {outpath.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
