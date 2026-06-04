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
    PROFILE_VECTOR_PARAMETERS,
    _baseline_profile_cfg,
    _prepare_benchmark_config,
    _prepare_realized_schedule_profile_vector_rollout_option_a,
)
from NEOPAX._orchestrator import build_runtime_context  # noqa: E402
from NEOPAX._transport_solvers import (  # noqa: E402
    _RadauAcceptedStepReducedOutput,
    _radau_collect_realized_accepted_step_payloads,
    _radau_rollout_reverse_from_saved_payloads,
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
):
    payload_rollout = _radau_collect_realized_accepted_step_payloads(
        execution_context,
        initial_carry,
        max_total_steps=max_total_steps,
        stop_after_accepted_steps=stop_after_accepted_steps,
    )
    reduced_output_bar = _make_reduced_output_bar(payload_rollout.final_carry, bar_mode)
    accepted_count = int(
        np.asarray(
            jax.device_get(jnp.sum(payload_rollout.accepted_mask.astype(jnp.int32))),
            dtype=int,
        ).item()
    )

    def _reverse_only_fn():
        reduced_input_bar = _radau_rollout_reverse_from_saved_payloads(
            execution_context,
            initial_carry,
            payload_rollout,
            reduced_output_bar,
        )
        return {
            "accepted_count": jnp.asarray(accepted_count, dtype=jnp.int32),
            "primitive_y_bar_max": _tree_max_abs(reduced_input_bar.y_out),
            "primitive_dt_bar_abs": jnp.max(jnp.abs(jnp.asarray(reduced_input_bar.dt_out, dtype=jnp.float64))),
            "primitive_prev_stages_bar_max": _tree_max_abs(reduced_input_bar.prev_stages_out),
        }

    if execution_mode == "jit":
        reverse_fn = jax.jit(_reverse_only_fn)
        t0 = time.perf_counter()
        first = reverse_fn()
        jax.block_until_ready(first["primitive_y_bar_max"])
        compile_plus_execute_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        second = reverse_fn()
        jax.block_until_ready(second["primitive_y_bar_max"])
        execute_s = time.perf_counter() - t1
    else:
        t0 = time.perf_counter()
        with jax.disable_jit():
            second = _reverse_only_fn()
        jax.block_until_ready(second["primitive_y_bar_max"])
        execute_s = time.perf_counter() - t0
        compile_plus_execute_s = execute_s

    result = {key: np.asarray(jax.device_get(value)).item() for key, value in second.items()}
    result["compile_plus_execute_s"] = compile_plus_execute_s
    result["execute_s"] = execute_s
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the accepted-step primitive reverse composition over several accepted steps."
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
        help="Run the reverse composition eagerly or under JIT. Default: jit.",
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
        "prefix_reports": prefix_reports,
    }

    outpath = _report_path()
    outpath.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("[autodiff-gate] mode=transport_reverse_multi_step_primitive")
    print(f"[autodiff-gate] parameters={list(parameter_names)}")
    print(
        f"[autodiff-gate] accepted_step_counts={list(accepted_step_counts)} "
        f"bar_mode={args.bar_mode} execution_mode={args.execution_mode} "
        f"max_total_steps_multiplier={int(args.max_total_steps_multiplier)}"
    )
    for prefix in prefix_reports:
        result = prefix["result"]
        print(
            f"  - accepted_step_count={int(prefix['accepted_step_count'])} "
            f"accepted_count={int(result['accepted_count'])} "
            f"max_total_steps={int(prefix['max_total_steps'])}"
        )
        print(f"    primitive_y_bar_max={float(result['primitive_y_bar_max']):.6e}")
        print(f"    primitive_dt_bar_abs={float(result['primitive_dt_bar_abs']):.6e}")
        print(f"    primitive_prev_stages_bar_max={float(result['primitive_prev_stages_bar_max']):.6e}")
        print(f"    compile_plus_execute_s={float(result['compile_plus_execute_s']):.6e}")
        print(f"    execute_s={float(result['execute_s']):.6e}")
    print(f"[autodiff-gate] wrote={outpath}")


if __name__ == "__main__":
    main()
