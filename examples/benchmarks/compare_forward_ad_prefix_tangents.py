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

from NEOPAX._orchestrator import build_runtime_context  # noqa: E402
from examples.benchmarks.benchmark_transport_autodiff_lagged_ntx import (  # noqa: E402
    _forward_benchmark_adaptive_rollout_objectives_realized_schedule_only_for_parameter_jvp,
)
from examples.benchmarks.benchmark_transport_forward_fd_lane import (  # noqa: E402
    DEFAULT_CONFIG,
    OBJECTIVE_LABELS,
    _adaptive_rollout_objectives_realized_schedule_only_for_parameter,
    _baseline_profile_cfg,
    _prepare_benchmark_config,
)


def _central_fd(objective_fn, baseline_value: float, step: float):
    value_minus = objective_fn(jnp.asarray(baseline_value - step))
    value_plus = objective_fn(jnp.asarray(baseline_value + step))
    return (value_plus - value_minus) / jnp.asarray(2.0 * step)


def _to_float_list(values) -> list[float]:
    return [float(x) for x in np.asarray(jax.device_get(values), dtype=float).reshape(-1)]


def _max_rel_error(left, right) -> float:
    left_np = np.asarray(jax.device_get(left), dtype=float)
    right_np = np.asarray(jax.device_get(right), dtype=float)
    denom = np.maximum(np.maximum(np.abs(left_np), np.abs(right_np)), 1.0e-30)
    return float(np.max(np.abs(left_np - right_np) / denom))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare short-prefix forward-AD tangents from the trusted replay lane, "
            "the experimental exact fused lane, and central FD."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--parameter",
        default="n0",
        choices=("n0", "T0", "density_shape_power", "temperature_shape_power"),
    )
    parser.add_argument("--accepted-step-limit", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--ntx-exact-derivative-mode",
        default="direct",
        choices=("direct",),
    )
    parser.add_argument("--radau-jacobian-reuse-mode", type=str, default="legacy")
    parser.add_argument("--fd-rel-step", type=float, default=3.0e-8)
    parser.add_argument("--fd-abs-step", type=float, default=1.0e-10)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("outputs/autodiff_transport_lagged_ntx/forward_ad_prefix_tangent_compare.json"),
    )
    args = parser.parse_args()

    config = _prepare_benchmark_config(
        args.config,
        device=args.device,
        ntx_exact_derivative_mode=args.ntx_exact_derivative_mode,
        radau_jacobian_reuse_mode=args.radau_jacobian_reuse_mode,
    )
    runtime, baseline_state = build_runtime_context(config)
    profile_cfg = _baseline_profile_cfg(config)
    baseline_value = float(profile_cfg[args.parameter])
    fd_step = max(abs(baseline_value) * float(args.fd_rel_step), float(args.fd_abs_step))

    replay_fn = lambda p: _forward_benchmark_adaptive_rollout_objectives_realized_schedule_only_for_parameter_jvp(  # noqa: E731
        p,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=args.parameter,
        accepted_step_limit_override=args.accepted_step_limit,
    )
    exact_fn = lambda p: _adaptive_rollout_objectives_realized_schedule_only_for_parameter(  # noqa: E731
        p,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=args.parameter,
        accepted_step_limit_override=args.accepted_step_limit,
        derivative_mode="jvp_exact",
    )

    print("[autodiff-gate] progress: comparing short-prefix forward AD tangents", flush=True)
    t0 = time.perf_counter()
    replay_values, replay_tangent = jax.jvp(
        replay_fn,
        (jnp.asarray(baseline_value),),
        (jnp.asarray(1.0),),
    )
    exact_values, exact_tangent = jax.jvp(
        exact_fn,
        (jnp.asarray(baseline_value),),
        (jnp.asarray(1.0),),
    )
    fd_tangent = _central_fd(exact_fn, baseline_value, fd_step)
    elapsed_s = time.perf_counter() - t0

    report = {
        "mode": "forward_ad_prefix_tangent_compare",
        "config_path": str(args.config),
        "parameter_name": args.parameter,
        "baseline_value": baseline_value,
        "accepted_step_limit": int(args.accepted_step_limit),
        "fd_step": float(fd_step),
        "ntx_exact_derivative_mode": str(args.ntx_exact_derivative_mode),
        "radau_jacobian_reuse_mode": str(args.radau_jacobian_reuse_mode),
        "elapsed_s": float(elapsed_s),
        "objective_labels": OBJECTIVE_LABELS,
        "replay_objective_values": _to_float_list(replay_values),
        "exact_objective_values": _to_float_list(exact_values),
        "replay_tangent": _to_float_list(replay_tangent),
        "exact_tangent": _to_float_list(exact_tangent),
        "fd_tangent": _to_float_list(fd_tangent),
        "max_rel_exact_vs_replay_value": _max_rel_error(exact_values, replay_values),
        "max_rel_exact_vs_replay_tangent": _max_rel_error(exact_tangent, replay_tangent),
        "max_rel_exact_vs_fd_tangent": _max_rel_error(exact_tangent, fd_tangent),
        "max_rel_replay_vs_fd_tangent": _max_rel_error(replay_tangent, fd_tangent),
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2) + "\n")

    print(
        "[autodiff-gate] mode=forward_ad_prefix_tangent_compare "
        f"parameter={args.parameter} accepted_step_limit={args.accepted_step_limit} "
        f"fd_step={fd_step:.6e} elapsed_s={elapsed_s:.6e}"
    )
    print(
        "[autodiff-gate] max rel: "
        f"exact_vs_replay_value={report['max_rel_exact_vs_replay_value']:.6e} "
        f"exact_vs_replay_tangent={report['max_rel_exact_vs_replay_tangent']:.6e} "
        f"exact_vs_fd_tangent={report['max_rel_exact_vs_fd_tangent']:.6e} "
        f"replay_vs_fd_tangent={report['max_rel_replay_vs_fd_tangent']:.6e}"
    )
    print("[autodiff-gate] tangent comparison:")
    for label, replay, exact, fd in zip(
        OBJECTIVE_LABELS,
        _to_float_list(replay_tangent),
        _to_float_list(exact_tangent),
        _to_float_list(fd_tangent),
    ):
        print(f"  - {label}: replay={replay:.16e} exact={exact:.16e} fd={fd:.16e}")
    print(f"Wrote {args.json_output}")


if __name__ == "__main__":
    main()
