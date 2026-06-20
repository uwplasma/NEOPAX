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
    return {
        "attempt_count": int(np.asarray(jax.device_get(rollout.attempt_count)).item()),
        "accepted_count": int(np.asarray(jax.device_get(rollout.accepted_count)).item()),
        "completed": bool(np.asarray(jax.device_get(rollout.completed)).item()),
        "failed": bool(np.asarray(jax.device_get(rollout.failed)).item()),
        "fail_code": int(np.asarray(jax.device_get(rollout.fail_code)).item()),
        "final_t": None if final_carry is None else float(np.asarray(jax.device_get(final_carry.t)).item()),
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
        choices=("retry_only", "retry_or_accepted_dt_close", "accepted_dt_close"),
        help=(
            "Radau Jacobian/LU reuse policy for the adaptive baseline and fixed-time endpoints. "
            "'retry_only' is the current default; 'retry_or_accepted_dt_close' keeps retry reuse "
            "and additionally reuses after accepted steps when the next dt is close to the cached dt."
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

    print("[autodiff-gate] progress: running baseline adaptive rollout for frozen FD trace", flush=True)
    t_baseline0 = time.perf_counter()
    baseline_final_state, baseline_rollout = _production_solver_baseline_final_state_and_schedule_for_parameter(
        jnp.asarray(baseline_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=args.parameter,
        accepted_step_limit_override=args.accepted_step_limit,
    )
    t_baseline1 = time.perf_counter()
    baseline_diag = _adaptive_rollout_diagnostics(baseline_rollout)
    replay_trace = _truncate_rollout_trace_by_accepted_steps(
        baseline_rollout.trace,
        args.accepted_step_limit,
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
    baseline_replay_elapsed_s = None
    accepted_replay_step_debug = None
    single_step_compare = None
    if args.baseline_replay_debug and str(args.replay_mode).strip().lower() == "accepted":
        accepted_time_list = _accepted_time_list_from_trace(replay_trace)
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
            accepted_step_limit=args.accepted_step_limit,
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
    if not args.baseline_replay_debug and not args.accepted_replay_step_debug and not args.single_step_compare:
        minus_value = baseline_value - fd_step
        plus_value = baseline_value + fd_step

        print(f"[autodiff-gate] progress: running fixed-time fd_minus solve ({args.replay_mode})", flush=True)
        t_minus0 = time.perf_counter()
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
        print(f"[autodiff-gate] progress: running fixed-time fd_plus solve ({args.replay_mode})", flush=True)
        t_plus0 = time.perf_counter()
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
    report = {
        "mode": "transport_frozen_fd_only",
        "config_path": str(Path(args.config)),
        "parameter_name": args.parameter,
        "baseline_value": baseline_value,
        "fd_step": float(fd_step),
        "replay_mode": str(args.replay_mode),
        "fixed_time_lane": fixed_time_lane,
        "radau_jacobian_reuse_mode": str(args.radau_jacobian_reuse_mode),
        "debug_direct_accepted_step_map": use_direct_accepted_step_map,
        "accepted_step_limit": None if args.accepted_step_limit is None else int(args.accepted_step_limit),
        "ntx_exact_derivative_mode": str(args.ntx_exact_derivative_mode),
        "baseline_replay_debug": bool(args.baseline_replay_debug),
        "accepted_replay_step_debug": accepted_replay_step_debug,
        "single_step_compare": single_step_compare,
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
        "rollout_path": {
            "baseline": baseline_diag,
            "baseline_adaptive_elapsed_s": t_baseline1 - t_baseline0,
            "baseline_replay_elapsed_s": baseline_replay_elapsed_s,
            "baseline_replay_state_finite": baseline_replay_state_finite,
            "fd_minus_elapsed_s": None if t_minus0 is None or t_minus1 is None else (t_minus1 - t_minus0),
            "fd_plus_elapsed_s": None if t_plus0 is None or t_plus1 is None else (t_plus1 - t_plus0),
            "frozen_fd_minus_state_finite": None if minus_replay is None else _tree_all_finite(minus_replay["final_state"]),
            "frozen_fd_plus_state_finite": None if plus_replay is None else _tree_all_finite(plus_replay["final_state"]),
        },
    }

    print(
        f"[autodiff-gate] mode=transport_frozen_fd_only parameter={args.parameter} "
        f"baseline_value={baseline_value:.6e} fd_step={fd_step:.6e} replay_mode={args.replay_mode} "
        f"fixed_time_lane={fixed_time_lane} radau_jacobian_reuse_mode={args.radau_jacobian_reuse_mode}"
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
    if grad_np is not None:
        if minus_diag is not None and plus_diag is not None:
            print("[autodiff-gate] fixed-time endpoint rollout diagnostics:")
            print(
                "  - minus: "
                f"attempt_count={minus_diag['attempt_count']} "
                f"accepted_count={minus_diag['accepted_count']} "
                f"completed={minus_diag['completed']} "
                f"failed={minus_diag['failed']} "
                f"fail_code={minus_diag['fail_code']} "
                f"final_t={0.0 if minus_diag['final_t'] is None else minus_diag['final_t']:.6e}"
            )
            print(
                "  - plus: "
                f"attempt_count={plus_diag['attempt_count']} "
                f"accepted_count={plus_diag['accepted_count']} "
                f"completed={plus_diag['completed']} "
                f"failed={plus_diag['failed']} "
                f"fail_code={plus_diag['fail_code']} "
                f"final_t={0.0 if plus_diag['final_t'] is None else plus_diag['final_t']:.6e}"
            )
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
