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
    _adaptive_rollout_final_state_for_parameter,
    _adaptive_rollout_objectives_for_parameter_on_frozen_trace,
    _baseline_profile_cfg,
    _fd_step,
    _objective_vector,
    _prepare_benchmark_config,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen-FD-only benchmark lane using scratch-compatible forward/FD helper semantics."
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
        default="attempt",
        choices=("attempt", "accepted"),
        help="Frozen replay mode used for FD.",
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
        "--baseline-replay-debug",
        action="store_true",
        help="Run only the unperturbed baseline accepted replay consistency check.",
    )
    parser.add_argument(
        "--accepted-replay-step-debug",
        action="store_true",
        help="Run only the unperturbed accepted-step state-by-state baseline vs replay comparison.",
    )
    args = parser.parse_args()

    config = _prepare_benchmark_config(
        Path(args.config),
        device=args.device,
        ntx_exact_derivative_mode=args.ntx_exact_derivative_mode,
    )
    runtime, baseline_state = build_runtime_context(config)
    profile_cfg = _baseline_profile_cfg(config)
    baseline_value = float(profile_cfg[args.parameter])
    fd_step = _fd_step(baseline_value, rel_step=args.fd_rel_step, abs_step=args.fd_abs_step)

    print("[autodiff-gate] progress: running baseline adaptive rollout for frozen FD trace", flush=True)
    t_baseline0 = time.perf_counter()
    baseline_final_state, baseline_rollout = _adaptive_rollout_final_state_for_parameter(
        jnp.asarray(baseline_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=args.parameter,
        use_realized_schedule_jvp=False,
        accepted_step_limit_override=args.accepted_step_limit,
        use_schedule_trace_only=True,
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
    accepted_time_list = None
    baseline_replay_elapsed_s = None
    accepted_replay_step_debug = None
    if args.baseline_replay_debug and str(args.replay_mode).strip().lower() == "accepted":
        accepted_time_list = _accepted_time_list_from_trace(replay_trace)
        print("[autodiff-gate] progress: running frozen baseline replay (accepted)", flush=True)
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
        )
        t_replay1 = time.perf_counter()
        baseline_replay_elapsed_s = t_replay1 - t_replay0
        baseline_replay_state_finite = _tree_all_finite(baseline_replay["final_state"])
        baseline_replay_abs_diff = np.abs(
            np.asarray(jax.device_get(baseline_replay_objectives), dtype=float)
            - np.asarray(jax.device_get(baseline_objectives_adaptive), dtype=float)
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
        realized_er = np.asarray([np.asarray(jax.device_get(state.Er), dtype=float) for state in accepted_replay_step_debug["realized_saved_states"]], dtype=float)
        time_list_er = np.asarray([np.asarray(jax.device_get(state.Er), dtype=float) for state in accepted_replay_step_debug["time_list_saved_states"]], dtype=float)
        er_step_max_abs = np.max(np.abs(realized_er - time_list_er), axis=1) if realized_er.size and time_list_er.size else np.asarray([], dtype=float)
        first_bad_step = int(np.argmax(er_step_max_abs > 1.0e-8)) if er_step_max_abs.size and np.any(er_step_max_abs > 1.0e-8) else -1
        accepted_replay_step_debug = {
            "accepted_count": int(realized_er.shape[0]) if realized_er.ndim >= 1 else 0,
            "elapsed_s": t_step_debug1 - t_step_debug0,
            "er_step_max_abs": er_step_max_abs.tolist(),
            "first_bad_step": first_bad_step,
            "first_bad_step_max_abs": None if first_bad_step < 0 else float(er_step_max_abs[first_bad_step]),
            "realized_lagged_valid_in": accepted_replay_step_debug["realized_lagged_valid_in"],
            "time_list_lagged_valid_in": accepted_replay_step_debug["time_list_lagged_valid_in"],
        }

    adaptive_objectives_np = np.asarray(
        jax.device_get(baseline_objectives_adaptive),
        dtype=float,
    )
    grad_np = None
    minus_replay = None
    plus_replay = None
    t_minus0 = t_minus1 = t_plus0 = t_plus1 = None
    if not args.baseline_replay_debug and not args.accepted_replay_step_debug:
        minus_value = baseline_value - fd_step
        plus_value = baseline_value + fd_step

        print(f"[autodiff-gate] progress: running frozen fd_minus replay ({args.replay_mode})", flush=True)
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
        )
        t_minus1 = time.perf_counter()
        print(f"[autodiff-gate] progress: running frozen fd_plus replay ({args.replay_mode})", flush=True)
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
        )
        t_plus1 = time.perf_counter()

        gradient_fd = (objectives_plus - objectives_minus) / (2.0 * fd_step)
        grad_np = np.asarray(jax.device_get(gradient_fd), dtype=float)
    report = {
        "mode": "transport_frozen_fd_only",
        "config_path": str(Path(args.config)),
        "parameter_name": args.parameter,
        "baseline_value": baseline_value,
        "fd_step": float(fd_step),
        "replay_mode": str(args.replay_mode),
        "accepted_step_limit": None if args.accepted_step_limit is None else int(args.accepted_step_limit),
        "ntx_exact_derivative_mode": str(args.ntx_exact_derivative_mode),
        "baseline_replay_debug": bool(args.baseline_replay_debug),
        "accepted_replay_step_debug": accepted_replay_step_debug,
        "objective_labels": OBJECTIVE_LABELS,
        "gradient_fd": None if grad_np is None else grad_np.tolist(),
        "adaptive_objectives": adaptive_objectives_np.tolist(),
        "baseline_replay_objectives": None if baseline_replay_objectives is None else np.asarray(jax.device_get(baseline_replay_objectives), dtype=float).tolist(),
        "baseline_replay_abs_diff": None if baseline_replay_abs_diff is None else baseline_replay_abs_diff.tolist(),
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
        f"baseline_value={baseline_value:.6e} fd_step={fd_step:.6e} replay_mode={args.replay_mode}"
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
        print("[autodiff-gate] baseline replay abs diffs vs adaptive:")
        for label, value in zip(OBJECTIVE_LABELS, baseline_replay_abs_diff.tolist()):
            print(f"  - {label}: abs_diff={float(value):.6e}")
    if accepted_replay_step_debug is not None:
        print(
            f"[autodiff-gate] accepted replay step debug: accepted_count={accepted_replay_step_debug['accepted_count']} "
            f"elapsed_s={accepted_replay_step_debug['elapsed_s']:.6e} "
            f"first_bad_step={accepted_replay_step_debug['first_bad_step']} "
            f"first_bad_step_max_abs={0.0 if accepted_replay_step_debug['first_bad_step_max_abs'] is None else accepted_replay_step_debug['first_bad_step_max_abs']:.6e}"
        )
    if grad_np is not None:
        print("[autodiff-gate] objective values:")
        for label, value in zip(OBJECTIVE_LABELS, grad_np.tolist()):
            print(f"  - {label}: fd={float(value):.6e}")

    outpath = _report_path(args.parameter)
    outpath.write_text(json.dumps(report, indent=2))
    print(f"Wrote {outpath.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
