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
    _adaptive_rollout_objectives_realized_schedule_only_for_parameter,
    _baseline_profile_cfg,
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
    parser.add_argument(
        "--radau-jacobian-reuse-mode",
        type=str,
        default=None,
        help="Optional Radau Jacobian reuse mode override, e.g. legacy or retry_only.",
    )
    parser.add_argument(
        "--forward-ad-fusion-mode",
        type=str,
        default="replay",
        choices=("replay", "step"),
        help=(
            "Forward AD implementation. 'replay' is the recovered reference; "
            "'step' is the experimental shared primal/tangent accepted-step path."
        ),
    )
    args = parser.parse_args()

    config = _prepare_benchmark_config(
        Path(args.config),
        device=args.device,
        ntx_exact_derivative_mode=args.ntx_exact_derivative_mode,
        radau_jacobian_reuse_mode=args.radau_jacobian_reuse_mode,
    )
    runtime, baseline_state = build_runtime_context(config)
    profile_cfg = _baseline_profile_cfg(config)
    baseline_value = float(profile_cfg[args.parameter])

    objective_fn = lambda p: _adaptive_rollout_objectives_realized_schedule_only_for_parameter(  # noqa: E731
        p,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=args.parameter,
        accepted_step_limit_override=args.accepted_step_limit,
        derivative_mode="jvp_step" if args.forward_ad_fusion_mode == "step" else "jvp",
    )

    print("[autodiff-gate] progress: running forward custom-JVP", flush=True)
    t_forward_ad_start = time.perf_counter()
    _, gradient_ad = jax.jvp(
        objective_fn,
        (jnp.asarray(baseline_value),),
        (jnp.asarray(1.0),),
    )

    grad_np = np.asarray(jax.device_get(gradient_ad), dtype=float)
    forward_ad_total_s = time.perf_counter() - t_forward_ad_start
    report = {
        "mode": "transport_forward_ad_only",
        "config_path": str(Path(args.config)),
        "parameter_name": args.parameter,
        "baseline_value": baseline_value,
        "accepted_step_limit": None if args.accepted_step_limit is None else int(args.accepted_step_limit),
        "ntx_exact_derivative_mode": str(args.ntx_exact_derivative_mode),
        "radau_jacobian_reuse_mode": None if args.radau_jacobian_reuse_mode is None else str(args.radau_jacobian_reuse_mode),
        "forward_ad_fusion_mode": str(args.forward_ad_fusion_mode),
        "forward_ad_total_s": float(forward_ad_total_s),
        "objective_labels": OBJECTIVE_LABELS,
        "gradient_forward_ad": grad_np.tolist(),
    }

    print(
        f"[autodiff-gate] mode=transport_forward_ad_only parameter={args.parameter} "
        f"baseline_value={baseline_value:.6e} "
        f"radau_jacobian_reuse_mode={args.radau_jacobian_reuse_mode} "
        f"forward_ad_fusion_mode={args.forward_ad_fusion_mode} "
        f"forward_ad_total_s={forward_ad_total_s:.6e}"
    )
    print("[autodiff-gate] objective values:")
    for label, value in zip(OBJECTIVE_LABELS, _to_float_list(gradient_ad)):
        print(f"  - {label}: ad={float(value):.6e}")

    outpath = _report_path(args.parameter)
    outpath.write_text(json.dumps(report, indent=2))
    print(f"Wrote {outpath.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
