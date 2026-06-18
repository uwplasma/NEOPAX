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
    _adaptive_rollout_objectives_realized_schedule_only_for_parameter,
    _baseline_profile_cfg,
    _production_solver_baseline_final_state_and_schedule_for_parameter,
    _prepare_benchmark_config,
)
from NEOPAX._orchestrator import build_runtime_context  # noqa: E402


def _report_path(parameter_name: str) -> Path:
    outdir = ROOT / "outputs" / "autodiff_transport_lagged_ntx" / parameter_name
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / "transport_forward_ad_only_summary.json"


def _to_float_list(values) -> list[float]:
    return np.asarray(jax.device_get(values), dtype=float).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Forward-only adaptive custom-JVP benchmark lane using scratch-compatible helper semantics."
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="Benchmark TOML.")
    parser.add_argument(
        "--parameter",
        type=str,
        default="n0",
        choices=sorted(ALLOWED_PARAMETERS),
        help="Profile parameter to differentiate.",
    )
    parser.add_argument("--device", type=str, default=None, help="Optional device override.")
    parser.add_argument(
        "--accepted-step-limit",
        type=int,
        default=None,
        help="Optional accepted-step prefix to stop the adaptive rollout.",
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

    print("[autodiff-gate] progress: running baseline adaptive rollout for forward AD lane", flush=True)
    _, baseline_rollout = _production_solver_baseline_final_state_and_schedule_for_parameter(
        jnp.asarray(baseline_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=args.parameter,
        accepted_step_limit_override=args.accepted_step_limit,
    )

    objective_fn = lambda p: _adaptive_rollout_objectives_realized_schedule_only_for_parameter(  # noqa: E731
        p,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=args.parameter,
        accepted_step_limit_override=args.accepted_step_limit,
        derivative_mode="jvp",
    )

    print("[autodiff-gate] progress: running forward custom-JVP", flush=True)
    _, gradient_ad = jax.jvp(
        objective_fn,
        (jnp.asarray(baseline_value),),
        (jnp.asarray(1.0),),
    )

    baseline_diag = _adaptive_rollout_diagnostics(baseline_rollout)
    grad_np = np.asarray(jax.device_get(gradient_ad), dtype=float)
    report = {
        "mode": "transport_forward_ad_only",
        "config_path": str(Path(args.config)),
        "parameter_name": args.parameter,
        "baseline_value": baseline_value,
        "accepted_step_limit": None if args.accepted_step_limit is None else int(args.accepted_step_limit),
        "ntx_exact_derivative_mode": str(args.ntx_exact_derivative_mode),
        "objective_labels": OBJECTIVE_LABELS,
        "gradient_forward_ad": grad_np.tolist(),
        "rollout_path": {
            "baseline": baseline_diag,
        },
    }

    print(
        f"[autodiff-gate] mode=transport_forward_ad_only parameter={args.parameter} "
        f"baseline_value={baseline_value:.6e}"
    )
    print(
        f"[autodiff-gate] rollout baseline: attempt_count={baseline_diag.get('attempt_count')} "
        f"accepted_count={baseline_diag.get('accepted_count')} "
        f"completed={baseline_diag.get('completed')} failed={baseline_diag.get('failed')} "
        f"fail_code={baseline_diag.get('fail_code')}"
    )
    print("[autodiff-gate] objective values:")
    for label, value in zip(OBJECTIVE_LABELS, _to_float_list(gradient_ad)):
        print(f"  - {label}: ad={float(value):.6e}")

    outpath = _report_path(args.parameter)
    outpath.write_text(json.dumps(report, indent=2))
    print(f"Wrote {outpath.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
