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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_transport_forward_fd_lane import (  # noqa: E402
    ALLOWED_PARAMETERS,
    DEFAULT_CONFIG,
    OBJECTIVE_LABELS,
    _accepted_time_list_from_trace,
    _accepted_replay_state_debug_for_parameter,
    _adaptive_rollout_diagnostics,
    _adaptive_rollout_objectives_for_parameter_on_frozen_trace,
    _baseline_profile_cfg,
    _fd_step,
    _first_accepted_replay_mismatch_for_parameter,
    _objective_vector,
    _production_solver_baseline_final_state_and_schedule_for_parameter,
    _prepare_benchmark_config,
    _single_step_compare_for_parameter,
    _truncate_rollout_trace_by_accepted_steps,
)
from NEOPAX._orchestrator import build_runtime_context  # noqa: E402


def _report_path(parameter_name: str) -> Path:
    outdir = ROOT / "outputs" / "autodiff_transport_lagged_ntx" / parameter_name
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / "transport_frozen_fd_only_summary.json"


def _tree_all_finite(tree) -> bool:
    for leaf in jax.tree_util.tree_leaves(tree):
        arr = np.asarray(jax.device_get(leaf))
        if np.issubdtype(arr.dtype, np.inexact) and not np.all(np.isfinite(arr)):
            return False
    return True


def _fixed_time_replay_diagnostics(replay) -> dict | None:
    if replay is None:
        return None
    rollout = replay.get("rollout") if isinstance(replay, dict) else None
    final_carry = replay.get("final_carry") if isinstance(replay, dict) else None
    if rollout is None:
        return None
    if hasattr(rollout, "attempt_count"):
        return {
            "attempt_count": int(np.asarray(jax.device_get(rollout.attempt_count)).item()),
            "accepted_count": int(np.asarray(jax.device_get(rollout.accepted_count)).item()),
            "completed": bool(np.asarray(jax.device_get(rollout.completed)).item()),
            "failed": bool(np.asarray(jax.device_get(rollout.failed)).item()),
            "fail_code": int(np.asarray(jax.device_get(rollout.fail_code)).item()),
            "final_t": None if final_carry is None else float(np.asarray(jax.device_get(final_carry.t)).item()),
            "fixed_time_mode": replay.get("fixed_time_mode") if isinstance(replay, dict) else None,
        }
    converged_mask = getattr(rollout, "converged_mask", None)
    accepted_dts = getattr(rollout, "accepted_dts", None)
    accepted_count = 0 if accepted_dts is None else int(np.asarray(jax.device_get(accepted_dts)).reshape(-1).shape[0])
    converged_np = None if converged_mask is None else np.asarray(jax.device_get(converged_mask), dtype=bool).reshape(-1)
    all_converged = True if converged_np is None else bool(np.all(converged_np))
    first_nonconverged_index = (
        None
        if converged_np is None or all_converged
        else int(np.argmax(np.logical_not(converged_np)))
    )
    nonconverged_count = (
        None
        if converged_np is None
        else int(np.sum(np.logical_not(converged_np)))
    )
    return {
        "attempt_count": accepted_count,
        "accepted_count": accepted_count,
        "completed": bool(all_converged),
        "failed": bool(not all_converged),
        "fail_code": int(0 if all_converged else 1),
        "final_t": None if final_carry is None else float(np.asarray(jax.device_get(final_carry.t)).item()),
        "fixed_time_mode": replay.get("fixed_time_mode") if isinstance(replay, dict) else None,
        "all_converged": bool(all_converged),
        "nonconverged_count": nonconverged_count,
        "first_nonconverged_index": first_nonconverged_index,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen-FD-only benchmark lane using the solver-native fixed accepted-time-map forward path."
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="Benchmark TOML.")
    parser.add_argument(
        "--parameter",
        type=str,
        default="n0",
        choices=sorted(ALLOWED_PARAMETERS),
        help="Profile parameter to differentiate.",
    )
    parser.add_argument("--fd-rel-step", type=float, default=3.0e-8, help="Relative FD step.")
    parser.add_argument("--fd-abs-step", type=float, default=1.0e-10, help="Absolute FD step.")
    parser.add_argument(
        "--replay-mode",
        type=str,
        default="accepted",
        choices=("accepted",),
        help="Fixed accepted-time-map mode used for FD.",
    )
    parser.add_argument(
        "--fixed-time-lane",
        type=str,
        default="solver",
        choices=("solver", "direct"),
        help=(
            "Fixed-time implementation used by baseline replay and FD endpoints. "
            "'solver' uses the forward-solver fixed-dt schedule path; 'direct' "
            "uses the lower-level accepted-step-map helper."
        ),
    )
    parser.add_argument(
        "--fd-endpoint-lane",
        type=str,
        default="fixed-time",
        choices=("fixed-time", "adaptive"),
        help=(
            "FD endpoint solve lane. 'fixed-time' freezes the baseline accepted time map; "
            "'adaptive' runs fd-/fd+ through the same production adaptive solver lane as the baseline."
        ),
    )
    parser.add_argument("--device", type=str, default=None, help="Optional device override.")
    parser.add_argument(
        "--accepted-step-limit",
        type=int,
        default=None,
        help="Optional accepted-step prefix used to truncate the baseline trace.",
    )
    parser.add_argument(
        "--ntx-exact-derivative-mode",
        default="direct",
        choices=("direct", "custom_vjp"),
        help="NTX exact-runtime derivative mode.",
    )
    parser.add_argument(
        "--radau-jacobian-reuse-mode",
        default="retry_only",
        choices=("retry_only", "retry_refactor_lu", "dt_close", "legacy"),
        help=(
            "Radau Jacobian/LU reuse policy for the adaptive baseline and FD endpoints. "
            "'retry_only' is the current default; 'retry_refactor_lu' reuses retry Jacobians "
            "but always refactors LU; 'dt_close'/'legacy' restores the old cache_valid && dt_close linearization reuse."
        ),
    )
    parser.add_argument(
        "--baseline-replay-debug",
        action="store_true",
        help="Run only the unperturbed baseline accepted replay consistency check.",
    )
    parser.add_argument(
        "--accepted-replay-step-debug",
        action="store_true",
        help="Run only the unperturbed accepted-step state-by-state baseline vs replay comparison.",
    )
    parser.add_argument(
        "--single-step-compare",
        action="store_true",
        help="Run only a one-step production-vs-fixed-dt comparison from the same incoming accepted-step boundary.",
    )
    parser.add_argument(
        "--first-replay-mismatch-debug",
        action="store_true",
        help=(
            "Run a lightweight production-vs-fixed accepted-step replay scan and stop at the "
            "first accepted step whose output mismatch exceeds --first-replay-mismatch-tol."
        ),
    )
    parser.add_argument(
        "--first-replay-mismatch-tol",
        type=float,
        default=1.0e-8,
        help="Absolute mismatch threshold for --first-replay-mismatch-debug.",
    )
    parser.add_argument(
        "--first-replay-max-accepted-steps",
        type=int,
        default=32,
        help="Maximum accepted steps inspected by --first-replay-mismatch-debug.",
    )
    parser.add_argument(
        "--debug-direct-accepted-step-map",
        action="store_true",
        help=(
            "Deprecated compatibility flag. Equivalent to --fixed-time-lane direct."
        ),
    )
    parser.add_argument(
        "--accepted-step-index",
        type=int,
        default=0,
        help="Accepted-step index used by --single-step-compare.",
    )
    args = parser.parse_args()
    fixed_time_lane = "direct" if args.debug_direct_accepted_step_map else str(args.fixed_time_lane).strip().lower()
    fd_endpoint_lane = str(args.fd_endpoint_lane).strip().lower()
    use_direct_accepted_step_map = fixed_time_lane == "direct"

    config = _prepare_benchmark_config(
        Path(args.config),
        device=args.device,
        ntx_exact_derivative_mode=args.ntx_exact_derivative_mode,
        radau_jacobian_reuse_mode=args.radau_jacobian_reuse_mode,
    )
    runtime, baseline_state = build_runtime_context(config)
    profile_cfg = _baseline_profile_cfg(config)
    baseline_value = float(profile_cfg[args.parameter])
    fd_step = _fd_step(baseline_value, rel_step=args.fd_rel_step, abs_step=args.fd_abs_step)
    effective_accepted_step_limit = args.accepted_step_limit
    if args.first_replay_mismatch_debug and effective_accepted_step_limit is None:
        effective_accepted_step_limit = int(max(1, args.first_replay_max_accepted_steps))

    print("[autodiff-gate] progress: running baseline adaptive rollout for frozen FD trace", flush=True)
    t_baseline0 = time.perf_counter()
    baseline_final_state, baseline_rollout = _production_solver_baseline_final_state_and_schedule_for_parameter(
        jnp.asarray(baseline_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=args.parameter,
        accepted_step_limit_override=effective_accepted_step_limit,
    )
    t_baseline1 = time.perf_counter()
    baseline_diag = _adaptive_rollout_diagnostics(baseline_rollout)
    replay_trace = _truncate_rollout_trace_by_accepted_steps(
        baseline_rollout.trace,
        effective_accepted_step_limit,
    )
    baseline_objectives_adaptive = _objective_vector(baseline_final_state, runtime)

    baseline_replay_objectives = None
    baseline_replay_state_finite = None
    baseline_replay_abs_diff = None
    baseline_replay_rel_diff = None
    baseline_replay_er_max_abs_diff = None
    baseline_replay_er_mean_abs_diff = None
    baseline_replay_er_max_rel_diff = None
    baseline_replay_er_mean_rel_diff = None
    accepted_time_list = None
    accepted_dt_sequence = None
    baseline_replay_elapsed_s = None
    accepted_replay_step_debug = None
    single_step_compare = None
    first_replay_mismatch_debug = None
    if args.baseline_replay_debug and str(args.replay_mode).strip().lower() == "accepted":
        accepted_time_list = _accepted_time_list_from_trace(replay_trace)
        accepted_mask_for_report = np.asarray(jax.device_get(replay_trace.accepted_mask), dtype=bool)
        active_mask_for_report = np.asarray(jax.device_get(replay_trace.active_mask), dtype=bool)
        attempted_dts_for_report = np.asarray(jax.device_get(replay_trace.attempted_dts), dtype=float)
        accepted_dt_sequence = attempted_dts_for_report[
            np.logical_and(active_mask_for_report, accepted_mask_for_report)
        ].tolist()
        print("[autodiff-gate] progress: running fixed-time baseline solve (accepted)", flush=True)
        t_replay0 = time.perf_counter()
        baseline_replay_objectives, baseline_replay = _adaptive_rollout_objectives_for_parameter_on_frozen_trace(
            jnp.asarray(baseline_value),
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=args.parameter,
            frozen_trace=replay_trace,
            replay_mode=args.replay_mode,
            use_direct_accepted_step_map_debug=use_direct_accepted_step_map,
        )
        t_replay1 = time.perf_counter()
        baseline_replay_elapsed_s = t_replay1 - t_replay0
        baseline_replay_state_finite = _tree_all_finite(baseline_replay["final_state"])
        baseline_replay_abs_diff = np.abs(
            np.asarray(jax.device_get(baseline_replay_objectives), dtype=float)
            - np.asarray(jax.device_get(baseline_objectives_adaptive), dtype=float)
        )
        baseline_replay_rel_diff = baseline_replay_abs_diff / np.maximum(
            np.abs(np.asarray(jax.device_get(baseline_objectives_adaptive), dtype=float)),
            1.0e-30,
        )
        baseline_er = np.asarray(jax.device_get(baseline_final_state.Er), dtype=float)
        replay_er = np.asarray(jax.device_get(baseline_replay["final_state"].Er), dtype=float)
        er_abs_diff = np.abs(baseline_er - replay_er)
        baseline_replay_er_max_abs_diff = float(np.max(er_abs_diff)) if er_abs_diff.size else 0.0
        baseline_replay_er_mean_abs_diff = float(np.mean(er_abs_diff)) if er_abs_diff.size else 0.0
        baseline_replay_er_max_rel_diff = (
            float(np.max(er_abs_diff) / max(np.max(np.abs(baseline_er)), 1.0e-30))
            if er_abs_diff.size
            else 0.0
        )
        baseline_replay_er_mean_rel_diff = (
            float(np.mean(er_abs_diff) / max(np.mean(np.abs(baseline_er)), 1.0e-30))
            if er_abs_diff.size
            else 0.0
        )
    if args.accepted_replay_step_debug and str(args.replay_mode).strip().lower() == "accepted":
        print("[autodiff-gate] progress: running accepted replay step debug", flush=True)
        t_step_debug0 = time.perf_counter()
        accepted_replay_step_debug = _accepted_replay_state_debug_for_parameter(
            jnp.asarray(baseline_value),
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=args.parameter,
            accepted_step_limit=effective_accepted_step_limit,
        )
        t_step_debug1 = time.perf_counter()
        er_step_max_abs = np.asarray(accepted_replay_step_debug["er_step_max_abs"], dtype=float)
        first_bad_step = int(np.argmax(er_step_max_abs > 1.0e-8)) if er_step_max_abs.size and np.any(er_step_max_abs > 1.0e-8) else -1
        accepted_replay_step_debug = {
            "accepted_count": int(er_step_max_abs.shape[0]),
            "elapsed_s": t_step_debug1 - t_step_debug0,
            "er_step_max_abs": er_step_max_abs.tolist(),
            "first_bad_step": first_bad_step,
            "first_bad_step_max_abs": None if first_bad_step < 0 else float(er_step_max_abs[first_bad_step]),
            "realized_lagged_valid_in": accepted_replay_step_debug["realized_lagged_valid_in"],
            "time_list_lagged_valid_in": accepted_replay_step_debug["time_list_lagged_valid_in"],
            "realized_prev_theta_final": accepted_replay_step_debug["realized_prev_theta_final"],
            "time_list_prev_theta_final": accepted_replay_step_debug["time_list_prev_theta_final"],
            "realized_prev_newton_iter_count": accepted_replay_step_debug["realized_prev_newton_iter_count"],
            "time_list_prev_newton_iter_count": accepted_replay_step_debug["time_list_prev_newton_iter_count"],
            "realized_prev_error": accepted_replay_step_debug["realized_prev_error"],
            "time_list_prev_error": accepted_replay_step_debug["time_list_prev_error"],
            "realized_prev_dt": accepted_replay_step_debug["realized_prev_dt"],
            "time_list_prev_dt": accepted_replay_step_debug["time_list_prev_dt"],
        }
    if args.single_step_compare:
        print("[autodiff-gate] progress: running single-step production vs fixed-dt comparison", flush=True)
        t_single0 = time.perf_counter()
        single_step_compare = _single_step_compare_for_parameter(
            jnp.asarray(baseline_value),
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=args.parameter,
            accepted_step_index=args.accepted_step_index,
        )
        t_single1 = time.perf_counter()
        single_step_compare["elapsed_s"] = t_single1 - t_single0
    if args.first_replay_mismatch_debug:
        print("[autodiff-gate] progress: running lightweight first replay mismatch debug", flush=True)
        t_first_mismatch0 = time.perf_counter()
        first_replay_mismatch_debug = _first_accepted_replay_mismatch_for_parameter(
            jnp.asarray(baseline_value),
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=args.parameter,
            replay_trace=replay_trace,
            mismatch_tol=args.first_replay_mismatch_tol,
            max_accepted_steps=args.first_replay_max_accepted_steps,
        )
        t_first_mismatch1 = time.perf_counter()
        first_replay_mismatch_debug["elapsed_s"] = t_first_mismatch1 - t_first_mismatch0

    adaptive_objectives_np = np.asarray(
        jax.device_get(baseline_objectives_adaptive),
        dtype=float,
    )
    grad_np = None
    objectives_minus_np = None
    objectives_plus_np = None
    fd_midpoint_np = None
    fd_midpoint_abs_diff_np = None
    fd_midpoint_rel_diff_np = None
    minus_replay = None
    plus_replay = None
    t_minus0 = t_minus1 = t_plus0 = t_plus1 = None
    if (
        not args.baseline_replay_debug
        and not args.accepted_replay_step_debug
        and not args.single_step_compare
        and not args.first_replay_mismatch_debug
    ):
        minus_value = baseline_value - fd_step
        plus_value = baseline_value + fd_step

        endpoint_label = "production adaptive" if fd_endpoint_lane == "adaptive" else "fixed-time"
        print(f"[autodiff-gate] progress: running {endpoint_label} fd_minus solve ({args.replay_mode})", flush=True)
        t_minus0 = time.perf_counter()
        if fd_endpoint_lane == "adaptive":
            minus_final_state, minus_rollout = _production_solver_baseline_final_state_and_schedule_for_parameter(
                jnp.asarray(minus_value),
                config=config,
                runtime=runtime,
                baseline_state=baseline_state,
                profile_cfg=profile_cfg,
                parameter_name=args.parameter,
                accepted_step_limit_override=effective_accepted_step_limit,
            )
            objectives_minus = _objective_vector(minus_final_state, runtime)
            minus_replay = {
                "rollout": minus_rollout,
                "final_state": minus_final_state,
                "final_carry": minus_rollout.final_carry,
            }
        else:
            objectives_minus, minus_replay = _adaptive_rollout_objectives_for_parameter_on_frozen_trace(
                jnp.asarray(minus_value),
                config=config,
                runtime=runtime,
                baseline_state=baseline_state,
                profile_cfg=profile_cfg,
                parameter_name=args.parameter,
                frozen_trace=replay_trace,
                replay_mode=args.replay_mode,
                use_direct_accepted_step_map_debug=use_direct_accepted_step_map,
            )
        t_minus1 = time.perf_counter()
        print(f"[autodiff-gate] progress: running {endpoint_label} fd_plus solve ({args.replay_mode})", flush=True)
        t_plus0 = time.perf_counter()
        if fd_endpoint_lane == "adaptive":
            plus_final_state, plus_rollout = _production_solver_baseline_final_state_and_schedule_for_parameter(
                jnp.asarray(plus_value),
                config=config,
                runtime=runtime,
                baseline_state=baseline_state,
                profile_cfg=profile_cfg,
                parameter_name=args.parameter,
                accepted_step_limit_override=effective_accepted_step_limit,
            )
            objectives_plus = _objective_vector(plus_final_state, runtime)
            plus_replay = {
                "rollout": plus_rollout,
                "final_state": plus_final_state,
                "final_carry": plus_rollout.final_carry,
            }
        else:
            objectives_plus, plus_replay = _adaptive_rollout_objectives_for_parameter_on_frozen_trace(
                jnp.asarray(plus_value),
                config=config,
                runtime=runtime,
                baseline_state=baseline_state,
                profile_cfg=profile_cfg,
                parameter_name=args.parameter,
                frozen_trace=replay_trace,
                replay_mode=args.replay_mode,
                use_direct_accepted_step_map_debug=use_direct_accepted_step_map,
            )
        t_plus1 = time.perf_counter()

        gradient_fd = (objectives_plus - objectives_minus) / (2.0 * fd_step)
        grad_np = np.asarray(jax.device_get(gradient_fd), dtype=float)
        objectives_minus_np = np.asarray(jax.device_get(objectives_minus), dtype=float)
        objectives_plus_np = np.asarray(jax.device_get(objectives_plus), dtype=float)
        fd_midpoint_np = 0.5 * (objectives_minus_np + objectives_plus_np)
        fd_midpoint_abs_diff_np = np.abs(fd_midpoint_np - adaptive_objectives_np)
        fd_midpoint_rel_diff_np = fd_midpoint_abs_diff_np / np.maximum(
            np.abs(adaptive_objectives_np),
            1.0e-30,
        )
    minus_diag = _fixed_time_replay_diagnostics(minus_replay)
    plus_diag = _fixed_time_replay_diagnostics(plus_replay)
    minus_state_finite = None if minus_replay is None else _tree_all_finite(minus_replay["final_state"])
    plus_state_finite = None if plus_replay is None else _tree_all_finite(plus_replay["final_state"])
    fd_valid = None
    fd_valid_reason = None
    if minus_diag is not None and plus_diag is not None:
        fd_valid = bool(
            minus_diag.get("completed")
            and not minus_diag.get("failed")
            and plus_diag.get("completed")
            and not plus_diag.get("failed")
        )
        if not fd_valid and fixed_time_lane == "direct":
            baseline_final_t = float(np.asarray(jax.device_get(baseline_rollout.final_carry.t)).item())
            minus_final_t = minus_diag.get("final_t")
            plus_final_t = plus_diag.get("final_t")
            direct_reached_final = bool(
                minus_final_t is not None
                and plus_final_t is not None
                and np.isclose(minus_final_t, baseline_final_t, rtol=0.0, atol=1.0e-12)
                and np.isclose(plus_final_t, baseline_final_t, rtol=0.0, atol=1.0e-12)
            )
            direct_finite = bool(
                minus_state_finite
                and plus_state_finite
                and objectives_minus_np is not None
                and objectives_plus_np is not None
                and np.all(np.isfinite(objectives_minus_np))
                and np.all(np.isfinite(objectives_plus_np))
            )
            if direct_reached_final and direct_finite:
                fd_valid = True
                fd_valid_reason = "direct_fixed_time_reached_final_with_finite_endpoints"
        if not fd_valid:
            grad_np = None
    report = {
        "mode": "transport_frozen_fd_only",
        "config_path": str(Path(args.config)),
        "parameter_name": args.parameter,
        "baseline_value": baseline_value,
        "fd_step": float(fd_step),
        "fd_valid": fd_valid,
        "fd_valid_reason": fd_valid_reason,
        "fd_endpoint_lane": fd_endpoint_lane,
        "replay_mode": str(args.replay_mode),
        "fixed_time_lane": fixed_time_lane,
        "radau_jacobian_reuse_mode": str(args.radau_jacobian_reuse_mode),
        "debug_direct_accepted_step_map": use_direct_accepted_step_map,
        "accepted_step_limit": None if effective_accepted_step_limit is None else int(effective_accepted_step_limit),
        "ntx_exact_derivative_mode": str(args.ntx_exact_derivative_mode),
        "baseline_replay_debug": bool(args.baseline_replay_debug),
        "accepted_replay_step_debug": accepted_replay_step_debug,
        "single_step_compare": single_step_compare,
        "first_replay_mismatch_debug": first_replay_mismatch_debug,
        "objective_labels": OBJECTIVE_LABELS,
        "gradient_fd": None if grad_np is None else grad_np.tolist(),
        "objectives_minus": None if objectives_minus_np is None else objectives_minus_np.tolist(),
        "objectives_plus": None if objectives_plus_np is None else objectives_plus_np.tolist(),
        "fd_midpoint_objectives": None if fd_midpoint_np is None else fd_midpoint_np.tolist(),
        "fd_midpoint_abs_diff_vs_adaptive": None if fd_midpoint_abs_diff_np is None else fd_midpoint_abs_diff_np.tolist(),
        "fd_midpoint_rel_diff_vs_adaptive": None if fd_midpoint_rel_diff_np is None else fd_midpoint_rel_diff_np.tolist(),
        "fd_minus_rollout": minus_diag,
        "fd_plus_rollout": plus_diag,
        "adaptive_objectives": adaptive_objectives_np.tolist(),
        "baseline_replay_objectives": None if baseline_replay_objectives is None else np.asarray(jax.device_get(baseline_replay_objectives), dtype=float).tolist(),
        "baseline_replay_abs_diff": None if baseline_replay_abs_diff is None else baseline_replay_abs_diff.tolist(),
        "baseline_replay_rel_diff": None if baseline_replay_rel_diff is None else baseline_replay_rel_diff.tolist(),
        "baseline_replay_er_max_abs_diff": baseline_replay_er_max_abs_diff,
        "baseline_replay_er_mean_abs_diff": baseline_replay_er_mean_abs_diff,
        "baseline_replay_er_max_rel_diff": baseline_replay_er_max_rel_diff,
        "baseline_replay_er_mean_rel_diff": baseline_replay_er_mean_rel_diff,
        "accepted_time_list": accepted_time_list,
        "accepted_dt_sequence": accepted_dt_sequence,
        "rollout_path": {
            "baseline": baseline_diag,
            "baseline_adaptive_elapsed_s": t_baseline1 - t_baseline0,
            "baseline_replay_elapsed_s": baseline_replay_elapsed_s,
            "baseline_replay_state_finite": baseline_replay_state_finite,
            "fd_minus_elapsed_s": None if t_minus0 is None or t_minus1 is None else (t_minus1 - t_minus0),
            "fd_plus_elapsed_s": None if t_plus0 is None or t_plus1 is None else (t_plus1 - t_plus0),
            "frozen_fd_minus_state_finite": minus_state_finite,
            "frozen_fd_plus_state_finite": plus_state_finite,
        },
    }

    print(
        f"[autodiff-gate] mode=transport_frozen_fd_only parameter={args.parameter} "
        f"baseline_value={baseline_value:.6e} fd_step={fd_step:.6e} replay_mode={args.replay_mode} "
        f"fixed_time_lane={fixed_time_lane} fd_endpoint_lane={fd_endpoint_lane} "
        f"radau_jacobian_reuse_mode={args.radau_jacobian_reuse_mode}"
    )
    print(
        f"[autodiff-gate] rollout baseline: attempt_count={baseline_diag.get('attempt_count')} "
        f"accepted_count={baseline_diag.get('accepted_count')} "
        f"completed={baseline_diag.get('completed')} failed={baseline_diag.get('failed')} "
        f"fail_code={baseline_diag.get('fail_code')}"
    )
    print(
        f"[autodiff-gate] timing baseline_adaptive_s={t_baseline1 - t_baseline0:.6e} "
        f"baseline_replay_s={0.0 if baseline_replay_elapsed_s is None else baseline_replay_elapsed_s:.6e} "
        f"fd_minus_s={0.0 if t_minus0 is None or t_minus1 is None else t_minus1 - t_minus0:.6e} "
        f"fd_plus_s={0.0 if t_plus0 is None or t_plus1 is None else t_plus1 - t_plus0:.6e}"
    )
    if baseline_replay_objectives is not None:
        print(
            f"[autodiff-gate] baseline replay accepted_count={0 if accepted_time_list is None else len(accepted_time_list)} "
            f"state_finite={baseline_replay_state_finite}"
        )
        print(
            "[autodiff-gate] baseline replay Er diffs vs adaptive: "
            f"max_abs_diff={0.0 if baseline_replay_er_max_abs_diff is None else baseline_replay_er_max_abs_diff:.6e} "
            f"mean_abs_diff={0.0 if baseline_replay_er_mean_abs_diff is None else baseline_replay_er_mean_abs_diff:.6e} "
            f"max_rel_diff={0.0 if baseline_replay_er_max_rel_diff is None else baseline_replay_er_max_rel_diff:.6e} "
            f"mean_rel_diff={0.0 if baseline_replay_er_mean_rel_diff is None else baseline_replay_er_mean_rel_diff:.6e}"
        )
        print("[autodiff-gate] baseline replay relative diffs vs adaptive:")
        for label, abs_value, rel_value in zip(
            OBJECTIVE_LABELS,
            baseline_replay_abs_diff.tolist(),
            baseline_replay_rel_diff.tolist(),
        ):
            print(f"  - {label}: rel_diff={float(rel_value):.6e} abs_diff={float(abs_value):.6e}")
    if accepted_replay_step_debug is not None:
        print(
            f"[autodiff-gate] accepted replay step debug: accepted_count={accepted_replay_step_debug['accepted_count']} "
            f"elapsed_s={accepted_replay_step_debug['elapsed_s']:.6e} "
            f"first_bad_step={accepted_replay_step_debug['first_bad_step']} "
            f"first_bad_step_max_abs={0.0 if accepted_replay_step_debug['first_bad_step_max_abs'] is None else accepted_replay_step_debug['first_bad_step_max_abs']:.6e}"
        )
        first_bad_step = accepted_replay_step_debug["first_bad_step"]
        if first_bad_step >= 0:
            print(
                "[autodiff-gate] accepted replay metadata at first bad step: "
                f"realized_theta={accepted_replay_step_debug['realized_prev_theta_final'][first_bad_step]:.6e} "
                f"replay_theta={accepted_replay_step_debug['time_list_prev_theta_final'][first_bad_step]:.6e} "
                f"realized_newton_iter={accepted_replay_step_debug['realized_prev_newton_iter_count'][first_bad_step]} "
                f"replay_newton_iter={accepted_replay_step_debug['time_list_prev_newton_iter_count'][first_bad_step]} "
                f"realized_prev_error={accepted_replay_step_debug['realized_prev_error'][first_bad_step]:.6e} "
                f"replay_prev_error={accepted_replay_step_debug['time_list_prev_error'][first_bad_step]:.6e} "
                f"realized_prev_dt={accepted_replay_step_debug['realized_prev_dt'][first_bad_step]:.6e} "
                f"replay_prev_dt={accepted_replay_step_debug['time_list_prev_dt'][first_bad_step]:.6e}"
            )
    if single_step_compare is not None:
        print(
            f"[autodiff-gate] single-step compare: accepted_step_index={single_step_compare['accepted_step_index']} "
            f"incoming_t={single_step_compare['incoming_t']:.6e} "
            f"accepted_dt={single_step_compare['accepted_dt']:.6e} "
            f"elapsed_s={single_step_compare['elapsed_s']:.6e}"
        )
        print(
            f"[autodiff-gate] single-step diffs: accepted_y={single_step_compare['accepted_y_max_abs_diff']:.6e} "
            f"Er={single_step_compare['state_Er_max_abs_diff']:.6e} "
            f"density={single_step_compare['state_density_max_abs_diff']:.6e} "
            f"pressure={single_step_compare['state_pressure_max_abs_diff']:.6e}"
        )
        print(
            f"[autodiff-gate] single-step carry diffs: prev_stages={single_step_compare['carry_prev_stages_max_abs_diff']:.6e} "
            f"prev_dt={single_step_compare['carry_prev_dt_abs_diff']:.6e} "
            f"prev_error={single_step_compare['carry_prev_error_abs_diff']:.6e} "
            f"theta={single_step_compare['carry_prev_theta_final_abs_diff']:.6e} "
            f"newton_iter_diff={single_step_compare['carry_prev_newton_iter_count_diff']} "
            f"jacobian={single_step_compare['carry_jacobian_max_abs_diff']:.6e} "
            f"real_lu={single_step_compare['carry_real_lu_max_abs_diff']:.6e} "
            f"complex_lu={single_step_compare['carry_complex_lu_max_abs_diff']:.6e} "
            f"lagged_cache={single_step_compare['carry_lagged_cache_max_abs_diff']:.6e} "
            f"lagged_ref={single_step_compare['carry_lagged_reference_y_max_abs_diff']:.6e} "
            f"lagged_valid_equal={single_step_compare['carry_lagged_valid_equal']}"
        )
    if first_replay_mismatch_debug is not None:
        print(
            "[autodiff-gate] first replay mismatch debug: "
            f"found_mismatch={first_replay_mismatch_debug['found_mismatch']} "
            f"checked_accepted_count={first_replay_mismatch_debug['checked_accepted_count']} "
            f"accepted_index={first_replay_mismatch_debug['accepted_index']} "
            f"attempt_index={first_replay_mismatch_debug['attempt_index']} "
            f"elapsed_s={first_replay_mismatch_debug['elapsed_s']:.6e}"
        )
        if first_replay_mismatch_debug.get("accepted_index") is not None:
            print(
                "[autodiff-gate] first replay mismatch incoming: "
                f"t={first_replay_mismatch_debug['incoming_t']:.6e} "
                f"dt={first_replay_mismatch_debug['accepted_dt']:.6e} "
                f"y_diff={first_replay_mismatch_debug['incoming_y_max_abs_diff']:.6e} "
                f"prev_stages_diff={first_replay_mismatch_debug['incoming_prev_stages_max_abs_diff']:.6e} "
                f"jacobian_diff={first_replay_mismatch_debug['incoming_jacobian_max_abs_diff']:.6e} "
                f"real_lu_diff={first_replay_mismatch_debug['incoming_real_lu_max_abs_diff']:.6e} "
                f"complex_lu_diff={first_replay_mismatch_debug['incoming_complex_lu_max_abs_diff']:.6e} "
                f"lagged_cache_diff={first_replay_mismatch_debug['incoming_lagged_cache_max_abs_diff']:.6e}"
            )
            print(
                "[autodiff-gate] first replay mismatch carry: "
                f"production_recent_reject={first_replay_mismatch_debug['incoming_production_recent_reject_count']} "
                f"fixed_recent_reject={first_replay_mismatch_debug['incoming_fixed_recent_reject_count']} "
                f"production_cache_valid={first_replay_mismatch_debug['incoming_production_cache_valid']} "
                f"fixed_cache_valid={first_replay_mismatch_debug['incoming_fixed_cache_valid']} "
                f"production_cache_dt={first_replay_mismatch_debug['incoming_production_cache_dt']:.6e} "
                f"fixed_cache_dt={first_replay_mismatch_debug['incoming_fixed_cache_dt']:.6e} "
                f"production_lagged_valid={first_replay_mismatch_debug['incoming_production_lagged_valid']} "
                f"fixed_lagged_valid={first_replay_mismatch_debug['incoming_fixed_lagged_valid']}"
            )
            print(
                "[autodiff-gate] first replay mismatch step: "
                f"production_accepted={first_replay_mismatch_debug['production_accepted']} "
                f"fixed_accepted={first_replay_mismatch_debug['fixed_accepted']} "
                f"production_jacobian_reused={first_replay_mismatch_debug['production_jacobian_reused']} "
                f"fixed_jacobian_reused={first_replay_mismatch_debug['fixed_jacobian_reused']} "
                f"production_lagged_reused={first_replay_mismatch_debug['production_lagged_reused']} "
                f"fixed_lagged_reused={first_replay_mismatch_debug['fixed_lagged_reused']} "
                f"production_newton_iter={first_replay_mismatch_debug['production_newton_iter_count']} "
                f"fixed_newton_iter={first_replay_mismatch_debug['fixed_newton_iter_count']}"
            )
            print(
                "[autodiff-gate] first replay mismatch outgoing: "
                f"y_diff={first_replay_mismatch_debug['outgoing_y_max_abs_diff']:.6e} "
                f"Er_diff={first_replay_mismatch_debug['outgoing_Er_max_abs_diff']:.6e} "
                f"density_diff={first_replay_mismatch_debug['outgoing_density_max_abs_diff']:.6e} "
                f"pressure_diff={first_replay_mismatch_debug['outgoing_pressure_max_abs_diff']:.6e} "
                f"production_err_norm={first_replay_mismatch_debug['production_err_norm']:.6e} "
                f"fixed_err_norm={first_replay_mismatch_debug['fixed_err_norm']:.6e}"
            )
    if minus_diag is not None and plus_diag is not None:
        print(f"[autodiff-gate] {fd_endpoint_lane} endpoint rollout diagnostics:")
        print(
            "  - minus: "
            f"attempt_count={minus_diag['attempt_count']} "
            f"accepted_count={minus_diag['accepted_count']} "
            f"completed={minus_diag['completed']} "
            f"failed={minus_diag['failed']} "
            f"fail_code={minus_diag['fail_code']} "
            f"final_t={0.0 if minus_diag['final_t'] is None else minus_diag['final_t']:.6e} "
            f"nonconverged_count={minus_diag.get('nonconverged_count')} "
            f"first_nonconverged_index={minus_diag.get('first_nonconverged_index')}"
        )
        print(
            "  - plus: "
            f"attempt_count={plus_diag['attempt_count']} "
            f"accepted_count={plus_diag['accepted_count']} "
            f"completed={plus_diag['completed']} "
            f"failed={plus_diag['failed']} "
            f"fail_code={plus_diag['fail_code']} "
            f"final_t={0.0 if plus_diag['final_t'] is None else plus_diag['final_t']:.6e} "
            f"nonconverged_count={plus_diag.get('nonconverged_count')} "
            f"first_nonconverged_index={plus_diag.get('first_nonconverged_index')}"
        )
        if fd_valid is False:
            print(
                "[autodiff-gate] fd_valid=False: not reporting central FD gradients "
                "because at least one FD endpoint did not complete."
            )
        elif fd_valid_reason is not None:
            print(
                "[autodiff-gate] fd_valid=True: reporting direct fixed-time FD gradients "
                f"because {fd_valid_reason}; inspect nonconverged_count diagnostics above."
            )
    if grad_np is not None:
        print("[autodiff-gate] objective values:")
        for label, value in zip(OBJECTIVE_LABELS, grad_np.tolist()):
            print(f"  - {label}: fd={float(value):.6e}")
    if objectives_minus_np is not None and objectives_plus_np is not None:
        print("[autodiff-gate] fd endpoint objectives:")
        for label, minus_value, plus_value in zip(OBJECTIVE_LABELS, objectives_minus_np.tolist(), objectives_plus_np.tolist()):
            print(
                f"  - {label}: minus={float(minus_value):.6e} "
                f"plus={float(plus_value):.6e} "
                f"delta={float(plus_value - minus_value):.6e}"
            )
    if grad_np is not None:
        if fd_midpoint_np is not None and fd_midpoint_abs_diff_np is not None and fd_midpoint_rel_diff_np is not None:
            print("[autodiff-gate] fd midpoint vs adaptive baseline:")
            for label, midpoint_value, adaptive_value, abs_diff, rel_diff in zip(
                OBJECTIVE_LABELS,
                fd_midpoint_np.tolist(),
                adaptive_objectives_np.tolist(),
                fd_midpoint_abs_diff_np.tolist(),
                fd_midpoint_rel_diff_np.tolist(),
            ):
                print(
                    f"  - {label}: midpoint={float(midpoint_value):.6e} "
                    f"adaptive={float(adaptive_value):.6e} "
                    f"abs_diff={float(abs_diff):.6e} "
                    f"rel_diff={float(rel_diff):.6e}"
                )

    outpath = _report_path(args.parameter)
    outpath.write_text(json.dumps(report, indent=2))
    print(f"Wrote {outpath.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
