from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_transport_autodiff_lagged_ntx import (  # noqa: E402
    ALLOWED_PARAMETERS,
    DEFAULT_CONFIG,
    OBJECTIVE_LABELS,
    _adaptive_rollout_diagnostics,
    _baseline_profile_cfg,
    _fd_step,
    _forward_benchmark_adaptive_rollout_final_state_for_parameter,
    _forward_benchmark_adaptive_rollout_objectives_for_parameter_on_frozen_trace,
    _forward_benchmark_adaptive_rollout_objectives_realized_schedule_only_for_parameter,
    _prepare_benchmark_config,
    _truncate_rollout_trace_by_accepted_steps,
)
from NEOPAX._orchestrator import build_runtime_context  # noqa: E402


def _report_path(parameter_name: str) -> Path:
    outdir = ROOT / "outputs" / "autodiff_transport_lagged_ntx" / parameter_name
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / "transport_adaptive_ad_vs_frozen_fd_summary.json"


def _tree_all_finite(tree) -> bool:
    for leaf in jax.tree_util.tree_leaves(tree):
        arr = np.asarray(jax.device_get(leaf))
        if np.issubdtype(arr.dtype, np.inexact) and not np.all(np.isfinite(arr)):
            return False
    return True


def _to_float_list(values) -> list[float]:
    return np.asarray(jax.device_get(values), dtype=float).tolist()


def _print_summary(report: dict[str, Any]) -> None:
    print(
        f"[autodiff-gate] mode=adaptive_ad_vs_frozen_fd "
        f"parameter={report['parameter_name']} "
        f"baseline_value={report['baseline_value']:.6e} "
        f"fd_step={report['fd_step']:.6e} "
        f"replay_mode={report['replay_mode']}"
    )
    diag = report["rollout_path"]["baseline"]
    print(
        f"[autodiff-gate] rollout baseline: "
        f"attempt_count={diag.get('attempt_count')} "
        f"accepted_count={diag.get('accepted_count')} "
        f"completed={diag.get('completed')} "
        f"failed={diag.get('failed')} "
        f"fail_code={diag.get('fail_code')}"
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the adaptive custom-JVP transport derivative against central FD on the frozen baseline trace."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Benchmark TOML.",
    )
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
    parser.add_argument("--device", type=str, default=None, help="Optional device override passed to config preparation.")
    parser.add_argument(
        "--accepted-step-limit",
        type=int,
        default=None,
        help="Optional accepted-step prefix to truncate the frozen FD trace.",
    )
    parser.add_argument(
        "--ntx-exact-derivative-mode",
        default="direct",
        choices=("direct", "custom_vjp"),
        help="NTX exact-runtime derivative mode. Use direct for this forward-mode benchmark.",
    )
    parser.add_argument(
        "--adaptive-derivative-mode",
        default="jvp",
        choices=("jvp", "vjp"),
        help="Adaptive realized-schedule derivative mode: forward custom JVP or reverse custom VJP.",
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
    _, baseline_rollout = _forward_benchmark_adaptive_rollout_final_state_for_parameter(
        jnp.asarray(baseline_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=args.parameter,
        use_realized_schedule_jvp=False,
    )
    baseline_diag = _adaptive_rollout_diagnostics(baseline_rollout)
    replay_trace = _truncate_rollout_trace_by_accepted_steps(
        baseline_rollout.trace,
        args.accepted_step_limit,
    )

    objective_fn = lambda p: _forward_benchmark_adaptive_rollout_objectives_realized_schedule_only_for_parameter(  # noqa: E731
        p,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=args.parameter,
        derivative_mode=args.adaptive_derivative_mode,
    )

    print(f"[autodiff-gate] progress: running adaptive custom AD ({args.adaptive_derivative_mode})", flush=True)
    if str(args.adaptive_derivative_mode).strip().lower() == "jvp":
        _, gradient_ad = jax.jvp(
            objective_fn,
            (jnp.asarray(baseline_value),),
            (jnp.asarray(1.0),),
        )
    else:
        gradient_ad = jax.jacrev(objective_fn)(jnp.asarray(baseline_value))

    minus_value = baseline_value - fd_step
    plus_value = baseline_value + fd_step

    print(f"[autodiff-gate] progress: running frozen fd_minus replay ({args.replay_mode})", flush=True)
    objectives_minus, minus_replay = _forward_benchmark_adaptive_rollout_objectives_for_parameter_on_frozen_trace(
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
    objectives_plus, plus_replay = _forward_benchmark_adaptive_rollout_objectives_for_parameter_on_frozen_trace(
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

    grad_ad_np = np.asarray(jax.device_get(gradient_ad), dtype=float)
    grad_fd_np = np.asarray(jax.device_get(gradient_fd), dtype=float)
    abs_err = np.abs(grad_ad_np - grad_fd_np)
    rel_err = abs_err / np.maximum(np.abs(grad_fd_np), 1.0e-10)

    report = {
        "adaptive_ad_vs_frozen_fd_check": True,
        "config_path": str(Path(args.config)),
        "parameter_name": args.parameter,
        "baseline_value": baseline_value,
        "fd_step": float(fd_step),
        "replay_mode": str(args.replay_mode),
        "adaptive_derivative_mode": str(args.adaptive_derivative_mode),
        "accepted_step_limit": None if args.accepted_step_limit is None else int(args.accepted_step_limit),
        "ntx_exact_derivative_mode": str(args.ntx_exact_derivative_mode),
        "objective_labels": OBJECTIVE_LABELS,
        "gradient_autodiff": grad_ad_np.tolist(),
        "gradient_fd": grad_fd_np.tolist(),
        "gradient_absolute_error": abs_err.tolist(),
        "gradient_relative_error": rel_err.tolist(),
        "max_relative_error": float(np.max(rel_err)),
        "passed": bool(np.all(np.isfinite(rel_err)) and np.max(rel_err) <= 5.0e-2),
        "rollout_path": {
            "baseline": baseline_diag,
            "frozen_fd_minus_state_finite": _tree_all_finite(minus_replay["final_state"]),
            "frozen_fd_plus_state_finite": _tree_all_finite(plus_replay["final_state"]),
        },
    }

    _print_summary(report)
    outpath = _report_path(args.parameter)
    outpath.write_text(json.dumps(report, indent=2))
    print(f"Wrote {outpath.relative_to(ROOT)}")
    print(f"parameter={args.parameter} passed={report['passed']} max_rel_error={report['max_relative_error']}")


if __name__ == "__main__":
    main()
