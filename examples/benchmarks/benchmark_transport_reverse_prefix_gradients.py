from __future__ import annotations

import argparse
import copy
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
    _initial_carry_from_state_with_static_setup,
    _objective_vector_from_final_y,
    _parameterized_initial_state_multi,
    _prepare_benchmark_config,
    _prepare_realized_schedule_profile_vector_rollout_option_a,
)
from NEOPAX._orchestrator import build_runtime_context  # noqa: E402
from NEOPAX._transport_solvers import (  # noqa: E402
    _RadauAcceptedStepReducedOutput,
    _radau_collect_realized_accepted_step_payloads,
    _radau_reduced_output_from_carry,
    _radau_replay_realized_accepted_rollout,
    _radau_rollout_reverse_from_saved_payloads,
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


def _parse_step_counts(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in str(text).split(",") if item.strip())
    if not values:
        raise ValueError("At least one accepted-step prefix must be provided.")
    if any(value <= 0 for value in values):
        raise ValueError("Accepted-step prefixes must be positive integers.")
    return values


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


def _objective_basis(count: int, index: int, dtype) -> jax.Array:
    return jnp.asarray(np.eye(count, dtype=np.float64)[index], dtype=dtype)


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


def _prepare_prefix_context(
    baseline_vector,
    *,
    config,
    runtime,
    baseline_state,
    profile_cfg,
    parameter_names: tuple[str, ...],
    accepted_step_limit: int,
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


def _compute_forward_reference(
    baseline_vector,
    *,
    runtime,
    baseline_state,
    profile_cfg,
    parameter_names: tuple[str, ...],
    prefix_context: dict[str, object],
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
    jacobian = jax.jacfwd(_objective_fn)(baseline_vector)
    jax.block_until_ready(jacobian)
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
    payload_rollout,
    execution_mode: str,
):
    execution_context = prefix_context["execution_context"]
    prepared_rollout = prefix_context["prepared_rollout"]
    initial_carry = prefix_context["initial_carry"]
    solver = prefix_context["solver"]
    solve_vector_field = prefix_context["solve_vector_field"]

    objective_values, final_y_pullback = jax.vjp(
        lambda final_y: _objective_vector_from_final_y(
            final_y,
            prepared_rollout=prepared_rollout,
            runtime=runtime,
        ),
        payload_rollout.final_carry.y,
    )

    def _initial_reduced_output_from_params(params):
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
        return _radau_reduced_output_from_carry(carry0)

    _, initial_param_pullback = jax.vjp(_initial_reduced_output_from_params, baseline_vector)

    def _reverse_kernel(final_output_bar):
        return _radau_rollout_reverse_from_saved_payloads(
            execution_context,
            initial_carry,
            payload_rollout,
            final_output_bar,
        )

    kernel_fn = jax.jit(_reverse_kernel) if execution_mode == "jit" else _reverse_kernel

    jacobian_rows = []
    compile_plus_execute_s = None
    reverse_start_s = time.perf_counter()
    objective_count = int(objective_values.shape[0])
    for objective_index in range(objective_count):
        basis = _objective_basis(objective_count, objective_index, objective_values.dtype)
        (final_y_bar,) = final_y_pullback(basis)
        final_output_bar = _make_final_output_bar(payload_rollout.final_carry, final_y_bar)
        call_start_s = time.perf_counter()
        reduced_input_bar = kernel_fn(final_output_bar)
        jax.block_until_ready(reduced_input_bar.y_out)
        call_elapsed_s = time.perf_counter() - call_start_s
        if compile_plus_execute_s is None:
            compile_plus_execute_s = call_elapsed_s
        (parameter_bar,) = initial_param_pullback(reduced_input_bar)
        jacobian_rows.append(parameter_bar)
    total_reverse_s = time.perf_counter() - reverse_start_s
    jacobian = jnp.stack(jacobian_rows, axis=0)
    jax.block_until_ready(jacobian)
    return {
        "objective_values": objective_values,
        "jacobian": jacobian,
        "compile_plus_execute_s": float(compile_plus_execute_s or 0.0),
        "total_reverse_s": float(total_reverse_s),
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
        description="Compare short-prefix accepted-step reverse gradients against the trusted forward AD path."
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
        help="Comma-separated accepted-step prefixes to compare.",
    )
    parser.add_argument(
        "--reverse-execution-mode",
        default="jit",
        choices=("eager", "jit"),
        help="Run the new reverse composition eagerly or under JIT. Default: jit.",
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
        forward_objectives, forward_jacobian, forward_elapsed_s, payload_rollout = _compute_forward_reference(
            baseline_vector,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_names=parameter_names,
            prefix_context=prefix_context,
        )
        reverse_result = _compute_reverse_candidate(
            baseline_vector,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_names=parameter_names,
            prefix_context=prefix_context,
            payload_rollout=payload_rollout,
            execution_mode=args.reverse_execution_mode,
        )

        forward_objectives_np = np.asarray(jax.device_get(forward_objectives), dtype=float)
        reverse_objectives_np = np.asarray(jax.device_get(reverse_result["objective_values"]), dtype=float)
        forward_jacobian_np = np.asarray(jax.device_get(forward_jacobian), dtype=float)
        reverse_jacobian_np = np.asarray(jax.device_get(reverse_result["jacobian"]), dtype=float)
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
                "accepted_step_limit": int(accepted_step_limit),
                "forward_elapsed_s": float(forward_elapsed_s),
                "reverse_compile_plus_execute_s": float(reverse_result["compile_plus_execute_s"]),
                "reverse_total_s": float(reverse_result["total_reverse_s"]),
                "objective_values_forward": forward_objectives_np.tolist(),
                "objective_values_reverse": reverse_objectives_np.tolist(),
                "objective_diff": objective_metrics,
                "forward_jacobian": forward_jacobian_np.tolist(),
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
        "accepted_step_counts": [int(x) for x in accepted_step_counts],
        "reverse_execution_mode": args.reverse_execution_mode,
        "max_total_steps_multiplier": int(args.max_total_steps_multiplier),
        "prefixes": prefixes,
    }

    outpath = _report_path()
    outpath.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("[autodiff-gate] mode=transport_reverse_prefix_gradients")
    print(f"[autodiff-gate] parameters={list(parameter_names)}")
    print(
        f"[autodiff-gate] accepted_step_counts={list(accepted_step_counts)} "
        f"reverse_execution_mode={args.reverse_execution_mode} "
        f"max_total_steps_multiplier={int(args.max_total_steps_multiplier)}"
    )
    for prefix in prefixes:
        print(
            f"  - accepted_step_limit={int(prefix['accepted_step_limit'])} "
            f"objective_abs_max={float(prefix['objective_diff']['abs_max']):.6e} "
            f"gradient_abs_max={float(prefix['gradient_diff']['abs_max']):.6e} "
            f"gradient_rel_max={float(prefix['gradient_diff']['rel_max']):.6e}"
        )
    print(f"[autodiff-gate] wrote={outpath}")


if __name__ == "__main__":
    main()
