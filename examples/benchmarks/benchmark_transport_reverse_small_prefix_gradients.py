from __future__ import annotations

import argparse
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
    _adaptive_rollout_objectives_realized_schedule_only_for_parameter_vector,
    _baseline_profile_cfg,
    _initial_carry_from_state_with_static_setup,
    _objective_vector,
    _parameterized_initial_state_multi,
    _prepare_benchmark_config,
    _prepare_realized_schedule_profile_vector_rollout_option_a,
)
from NEOPAX._orchestrator import build_runtime_context  # noqa: E402
from NEOPAX._transport_solvers import (  # noqa: E402
    _RadauAcceptedStepReducedOutput,
    _radau_collect_realized_accepted_step_payloads,
    _radau_reduced_output_from_carry,
    _radau_rollout_reverse_from_saved_payloads,
)


def _report_path() -> Path:
    outdir = ROOT / "outputs" / "autodiff_transport_lagged_ntx" / "small_prefix_reverse_gradients"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / "transport_reverse_small_prefix_gradients_summary.json"


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


def _parse_counts(text: str) -> tuple[int, ...]:
    counts = tuple(max(1, int(item.strip())) for item in str(text).split(",") if item.strip())
    if not counts:
        raise ValueError("At least one accepted-step count must be provided.")
    return counts


def _zero_final_output_bar_like(carry, final_y_bar) -> _RadauAcceptedStepReducedOutput:
    return _RadauAcceptedStepReducedOutput(
        t_out=jnp.zeros_like(carry.t),
        y_out=final_y_bar,
        dt_out=jnp.zeros_like(carry.dt),
        prev_stages_out=jnp.zeros_like(carry.prev_stages),
        prev_dt_out=jnp.zeros_like(carry.prev_dt),
        lagged_reference_y_out=jnp.zeros_like(carry.lagged_reference_y),
        prev_theta_final_out=jnp.zeros_like(carry.prev_theta_final),
    )


def _compute_reverse_jacobian_for_prefix(
    baseline_vector,
    *,
    config,
    runtime,
    baseline_state,
    profile_cfg,
    parameter_names,
    accepted_step_limit: int,
):
    (
        execution_context,
        prepared_rollout_static,
        initial_carry,
        max_total_steps,
        stop_after_accepted_steps,
        solver,
        solve_vector_field_static,
    ) = _prepare_realized_schedule_profile_vector_rollout_option_a(
        baseline_vector,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_names=parameter_names,
        accepted_step_limit_override=accepted_step_limit,
    )

    payload_rollout = _radau_collect_realized_accepted_step_payloads(
        execution_context,
        initial_carry,
        max_total_steps=max_total_steps,
        stop_after_accepted_steps=stop_after_accepted_steps,
    )

    def _objective_from_final_y(final_y):
        final_state = prepared_rollout_static.physics_context.unpack_flat(final_y)
        return _objective_vector(final_state, runtime)

    objective_values, objective_pullback = jax.vjp(
        _objective_from_final_y,
        payload_rollout.final_carry.y,
    )

    def _initial_reduced_from_params(parameter_values):
        state0 = _parameterized_initial_state_multi(
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            geometry=runtime.geometry,
            n_species=runtime.species.number_species,
            parameter_names=parameter_names,
            parameter_values=parameter_values,
        )
        carry0 = _initial_carry_from_state_with_static_setup(
            solver=solver,
            state=state0,
            solve_vector_field=solve_vector_field_static,
            species=runtime.species,
            prepared_rollout_static=prepared_rollout_static,
        )
        return _radau_reduced_output_from_carry(carry0)

    _, initial_pullback = jax.vjp(_initial_reduced_from_params, baseline_vector)

    reverse_rows = []
    for objective_index in range(int(objective_values.shape[0])):
        basis = jnp.zeros_like(objective_values).at[objective_index].set(1.0)
        (final_y_bar,) = objective_pullback(basis)
        final_output_bar = _zero_final_output_bar_like(payload_rollout.final_carry, final_y_bar)
        reduced_input_bar = _radau_rollout_reverse_from_saved_payloads(
            execution_context,
            initial_carry,
            payload_rollout,
            final_output_bar,
        )
        (parameter_bar,) = initial_pullback(reduced_input_bar)
        reverse_rows.append(parameter_bar)

    reverse_jacobian = jnp.stack(reverse_rows, axis=0)
    return objective_values, reverse_jacobian


def _compute_prefix_report(
    baseline_vector,
    *,
    config,
    runtime,
    baseline_state,
    profile_cfg,
    parameter_names,
    accepted_step_limit: int,
):
    objective_fn = lambda p: _adaptive_rollout_objectives_realized_schedule_only_for_parameter_vector(  # noqa: E731
        p,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_names=parameter_names,
        accepted_step_limit_override=accepted_step_limit,
        derivative_mode="jvp",
    )

    t0 = time.perf_counter()
    forward_jacobian = jax.jacfwd(objective_fn)(baseline_vector)
    forward_objectives = objective_fn(baseline_vector)
    jax.block_until_ready(forward_jacobian)
    forward_time_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    reverse_objectives, reverse_jacobian = _compute_reverse_jacobian_for_prefix(
        baseline_vector,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_names=parameter_names,
        accepted_step_limit=accepted_step_limit,
    )
    jax.block_until_ready(reverse_jacobian)
    reverse_time_s = time.perf_counter() - t1

    forward_np = np.asarray(jax.device_get(forward_jacobian), dtype=float)
    reverse_np = np.asarray(jax.device_get(reverse_jacobian), dtype=float)
    objectives_forward_np = np.asarray(jax.device_get(forward_objectives), dtype=float)
    objectives_reverse_np = np.asarray(jax.device_get(reverse_objectives), dtype=float)
    abs_err = np.abs(reverse_np - forward_np)
    rel_err = abs_err / np.maximum(np.abs(forward_np), 1.0e-10)

    per_objective = {}
    for objective_index, objective_label in enumerate(OBJECTIVE_LABELS):
        per_parameter = {}
        for parameter_index, parameter_name in enumerate(parameter_names):
            per_parameter[parameter_name] = {
                "forward": float(forward_np[objective_index, parameter_index]),
                "reverse": float(reverse_np[objective_index, parameter_index]),
                "absolute_error": float(abs_err[objective_index, parameter_index]),
                "relative_error": float(rel_err[objective_index, parameter_index]),
            }
        per_objective[objective_label] = {
            "forward_objective_value": float(objectives_forward_np[objective_index]),
            "reverse_objective_value": float(objectives_reverse_np[objective_index]),
            "max_absolute_error": float(np.max(abs_err[objective_index])),
            "max_relative_error": float(np.max(rel_err[objective_index])),
            "parameters": per_parameter,
        }

    return {
        "accepted_step_limit": int(accepted_step_limit),
        "forward_time_s": float(forward_time_s),
        "reverse_time_s": float(reverse_time_s),
        "objective_values_forward": objectives_forward_np.tolist(),
        "objective_values_reverse": objectives_reverse_np.tolist(),
        "forward_jacobian": forward_np.tolist(),
        "reverse_jacobian": reverse_np.tolist(),
        "absolute_error": abs_err.tolist(),
        "relative_error": rel_err.tolist(),
        "max_absolute_error": float(np.max(abs_err)),
        "max_relative_error": float(np.max(rel_err)),
        "per_objective": per_objective,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare small-prefix accepted-step reverse gradients against the trusted forward AD path."
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
        help="Comma-separated accepted-step prefix lengths to compare.",
    )
    args = parser.parse_args()

    parameter_names = _parse_parameter_subset(args.parameters)
    accepted_step_counts = _parse_counts(args.accepted_step_counts)
    config = _prepare_benchmark_config(
        Path(args.config),
        device=args.device,
        ntx_exact_derivative_mode=args.ntx_exact_derivative_mode,
    )
    runtime, baseline_state = build_runtime_context(config)
    profile_cfg = _baseline_profile_cfg(config)
    baseline_vector = jnp.asarray(
        [float(profile_cfg[name]) for name in parameter_names],
        dtype=jnp.float64,
    )

    prefix_reports = []
    for accepted_step_limit in accepted_step_counts:
        prefix_report = _compute_prefix_report(
            baseline_vector,
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_names=parameter_names,
            accepted_step_limit=accepted_step_limit,
        )
        prefix_reports.append(prefix_report)

    report = {
        "config": str(args.config),
        "device": args.device,
        "ntx_exact_derivative_mode": args.ntx_exact_derivative_mode,
        "parameter_names": list(parameter_names),
        "parameter_values": [float(x) for x in np.asarray(jax.device_get(baseline_vector), dtype=float)],
        "accepted_step_counts": list(accepted_step_counts),
        "prefix_reports": prefix_reports,
    }

    outpath = _report_path()
    outpath.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("[autodiff-gate] mode=transport_reverse_small_prefix_gradients", flush=True)
    print(f"[autodiff-gate] parameters={list(parameter_names)}", flush=True)
    print(f"[autodiff-gate] accepted_step_counts={list(accepted_step_counts)}", flush=True)
    for prefix_report in prefix_reports:
        print(
            f"  - accepted_steps={prefix_report['accepted_step_limit']} "
            f"max_abs_err={prefix_report['max_absolute_error']:.6e} "
            f"max_rel_err={prefix_report['max_relative_error']:.6e} "
            f"forward_time_s={prefix_report['forward_time_s']:.6e} "
            f"reverse_time_s={prefix_report['reverse_time_s']:.6e}",
            flush=True,
        )
    print(f"[autodiff-gate] wrote={outpath}", flush=True)


if __name__ == "__main__":
    main()
