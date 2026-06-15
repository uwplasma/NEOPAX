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
    _forward_benchmark_adaptive_rollout_objectives_realized_schedule_only_for_parameter_jvp,
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


def _replay_diagnostics(replay: dict[str, Any], objectives) -> dict[str, Any]:
    rollout = replay.get("rollout")
    diag = {
        "replay_mode": replay.get("replay_mode"),
        "final_state_finite": _tree_all_finite(replay.get("final_state")),
        "final_carry_finite": _tree_all_finite(replay.get("final_carry")),
        "objectives_finite": bool(np.all(np.isfinite(np.asarray(jax.device_get(objectives), dtype=float)))),
    }
    if rollout is not None:
        for key in ("attempt_count", "accepted_count", "completed", "failed", "fail_code"):
            if hasattr(rollout, key):
                value = getattr(rollout, key)
                diag[key] = np.asarray(jax.device_get(value)).item()
    return diag


def _print_summary(report: dict[str, Any]) -> None:
    print(
        f"[autodiff-gate] mode=adaptive_ad_vs_frozen_fd "
        f"parameter={report['parameter_name']} "
        f"baseline_value={report['baseline_value']:.6e} "
        f"fd_step={report['fd_step']:.6e} "
        f"replay_mode={report['replay_mode']}"
    )
    diag = report["rollout_path"].get("baseline")
    if diag is not None:
        print(
            f"[autodiff-gate] rollout baseline: "
            f"attempt_count={diag.get('attempt_count')} "
            f"accepted_count={diag.get('accepted_count')} "
            f"completed={diag.get('completed')} "
            f"failed={diag.get('failed')} "
            f"fail_code={diag.get('fail_code')}"
        )
    grad_ad = report.get("gradient_autodiff")
    grad_fd = report.get("gradient_fd")
    grad_abs = report.get("gradient_absolute_error")
    grad_rel = report.get("gradient_relative_error")
    print("[autodiff-gate] objective errors:")
    if grad_ad is not None and grad_fd is not None:
        for label, ad, fd, ae, re in zip(
            report["objective_labels"],
            grad_ad,
            grad_fd,
            grad_abs,
            grad_rel,
        ):
            print(
                f"  - {label}: ad={float(ad):.6e} fd={float(fd):.6e} "
                f"abs_err={float(ae):.6e} rel_err={float(re):.6e}"
            )
    elif grad_ad is not None:
        for label, ad in zip(report["objective_labels"], grad_ad):
            print(f"  - {label}: ad={float(ad):.6e}")
    elif grad_fd is not None:
        for label, fd in zip(report["objective_labels"], grad_fd):
            print(f"  - {label}: fd={float(fd):.6e}")
    frozen_diag = report.get("frozen_replay", {})
    if frozen_diag:
        minus = frozen_diag.get("minus", {})
        plus = frozen_diag.get("plus", {})
        print(
            "[autodiff-gate] frozen replay minus: "
            f"objectives_finite={minus.get('objectives_finite')} "
            f"final_state_finite={minus.get('final_state_finite')} "
            f"final_carry_finite={minus.get('final_carry_finite')} "
            f"completed={minus.get('completed')} failed={minus.get('failed')} fail_code={minus.get('fail_code')}"
        )
        print(
            "[autodiff-gate] frozen replay plus: "
            f"objectives_finite={plus.get('objectives_finite')} "
            f"final_state_finite={plus.get('final_state_finite')} "
            f"final_carry_finite={plus.get('final_carry_finite')} "
            f"completed={plus.get('completed')} failed={plus.get('failed')} fail_code={plus.get('fail_code')}"
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
    parser.add_argument(
        "--run-mode",
        default="both",
        choices=("both", "ad", "fd"),
        help="Run adaptive AD only, frozen FD only, or both.",
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

    baseline_rollout = None
    baseline_diag = None
    replay_trace = None
    if args.run_mode in ("both", "fd"):
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

    if str(args.adaptive_derivative_mode).strip().lower() == "jvp":
        objective_fn = lambda p: _forward_benchmark_adaptive_rollout_objectives_realized_schedule_only_for_parameter_jvp(  # noqa: E731
            p,
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=args.parameter,
            accepted_step_limit_override=args.accepted_step_limit,
        )
    else:
        objective_fn = lambda p: _forward_benchmark_adaptive_rollout_objectives_realized_schedule_only_for_parameter(  # noqa: E731
            p,
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=args.parameter,
            accepted_step_limit_override=args.accepted_step_limit,
            derivative_mode=args.adaptive_derivative_mode,
        )

    gradient_ad = None
    if args.run_mode in ("both", "ad"):
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

    objectives_minus = None
    objectives_plus = None
    minus_replay = None
    plus_replay = None
    gradient_fd = None
    if args.run_mode in ("both", "fd"):
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

    grad_ad_np = None if gradient_ad is None else np.asarray(jax.device_get(gradient_ad), dtype=float)
    grad_fd_np = None if gradient_fd is None else np.asarray(jax.device_get(gradient_fd), dtype=float)
    if grad_ad_np is not None and grad_fd_np is not None:
        abs_err = np.abs(grad_ad_np - grad_fd_np)
        rel_err = abs_err / np.maximum(np.abs(grad_fd_np), 1.0e-10)
        max_rel_error = float(np.max(rel_err))
        passed = bool(np.all(np.isfinite(rel_err)) and np.max(rel_err) <= 5.0e-2)
    elif grad_ad_np is not None:
        abs_err = None
        rel_err = None
        max_rel_error = float("nan")
        passed = bool(np.all(np.isfinite(grad_ad_np)))
    elif grad_fd_np is not None:
        abs_err = None
        rel_err = None
        max_rel_error = float("nan")
        passed = bool(np.all(np.isfinite(grad_fd_np)))
    else:
        raise ValueError("run_mode must execute at least one lane.")

    report = {
        "adaptive_ad_vs_frozen_fd_check": True,
        "config_path": str(Path(args.config)),
        "parameter_name": args.parameter,
        "baseline_value": baseline_value,
        "fd_step": float(fd_step),
        "replay_mode": str(args.replay_mode),
        "adaptive_derivative_mode": str(args.adaptive_derivative_mode),
        "run_mode": str(args.run_mode),
        "accepted_step_limit": None if args.accepted_step_limit is None else int(args.accepted_step_limit),
        "ntx_exact_derivative_mode": str(args.ntx_exact_derivative_mode),
        "objective_labels": OBJECTIVE_LABELS,
        "gradient_autodiff": None if grad_ad_np is None else grad_ad_np.tolist(),
        "gradient_fd": None if grad_fd_np is None else grad_fd_np.tolist(),
        "gradient_absolute_error": None if abs_err is None else abs_err.tolist(),
        "gradient_relative_error": None if rel_err is None else rel_err.tolist(),
        "max_relative_error": max_rel_error,
        "passed": passed,
        "rollout_path": {
            "baseline": baseline_diag,
            "frozen_fd_minus_state_finite": None if minus_replay is None else _tree_all_finite(minus_replay["final_state"]),
            "frozen_fd_plus_state_finite": None if plus_replay is None else _tree_all_finite(plus_replay["final_state"]),
        },
        "frozen_replay": None if minus_replay is None or plus_replay is None else {
            "minus": _replay_diagnostics(minus_replay, objectives_minus),
            "plus": _replay_diagnostics(plus_replay, objectives_plus),
        },
    }

    _print_summary(report)
    outpath = _report_path(args.parameter)
    outpath.write_text(json.dumps(report, indent=2))
    print(f"Wrote {outpath.relative_to(ROOT)}")
    print(f"parameter={args.parameter} passed={report['passed']} max_rel_error={report['max_relative_error']}")


if __name__ == "__main__":
    main()
