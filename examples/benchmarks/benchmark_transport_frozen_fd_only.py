from __future__ import annotations

import argparse
import json
import sys
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
    _adaptive_rollout_diagnostics,
    _adaptive_rollout_final_state_for_parameter,
    _adaptive_rollout_objectives_for_parameter_on_frozen_trace,
    _baseline_profile_cfg,
    _fd_step,
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
    _, baseline_rollout = _adaptive_rollout_final_state_for_parameter(
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
    baseline_diag = _adaptive_rollout_diagnostics(baseline_rollout)
    replay_trace = _truncate_rollout_trace_by_accepted_steps(
        baseline_rollout.trace,
        args.accepted_step_limit,
    )

    minus_value = baseline_value - fd_step
    plus_value = baseline_value + fd_step

    print(f"[autodiff-gate] progress: running frozen fd_minus replay ({args.replay_mode})", flush=True)
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
    print(f"[autodiff-gate] progress: running frozen fd_plus replay ({args.replay_mode})", flush=True)
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
        "objective_labels": OBJECTIVE_LABELS,
        "gradient_fd": grad_np.tolist(),
        "rollout_path": {
            "baseline": baseline_diag,
            "frozen_fd_minus_state_finite": _tree_all_finite(minus_replay["final_state"]),
            "frozen_fd_plus_state_finite": _tree_all_finite(plus_replay["final_state"]),
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
    print("[autodiff-gate] objective values:")
    for label, value in zip(OBJECTIVE_LABELS, grad_np.tolist()):
        print(f"  - {label}: fd={float(value):.6e}")

    outpath = _report_path(args.parameter)
    outpath.write_text(json.dumps(report, indent=2))
    print(f"Wrote {outpath.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
