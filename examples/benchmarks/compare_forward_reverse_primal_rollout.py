"""Compare forward-lane and reverse-lane primal rollouts.

This diagnostic checks whether the forward AD benchmark and reverse AD benchmark
differentiate the same primal trajectory before comparing tangents/adjoints.
No derivative is computed here.
"""

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

from benchmark_transport_forward_fd_lane import (  # noqa: E402
    DEFAULT_CONFIG,
    OBJECTIVE_LABELS,
    _adaptive_rollout_objectives_realized_schedule_only_for_parameter,
    _baseline_profile_cfg,
    _objective_vector,
    _prepare_benchmark_config,
)
from benchmark_transport_autodiff_lagged_ntx import (  # noqa: E402
    _forward_benchmark_adaptive_rollout_objectives_realized_schedule_only_for_parameter_jvp,
)
from benchmark_transport_reverse_ad_only import (  # noqa: E402
    PARAMETER_ORDER,
    _prepare_reverse_static_setup,
    _reverse_initial_carry_for_parameter_vector,
)
from NEOPAX._orchestrator import build_runtime_context  # noqa: E402
from NEOPAX._transport_solvers import (  # noqa: E402
    _radau_adaptive_final_y_realized_schedule,
    _radau_adaptive_final_y_realized_schedule_vjp,
)


def _objective_stats(lhs, rhs) -> list[dict[str, Any]]:
    lhs_np = np.asarray(jax.device_get(lhs), dtype=float)
    rhs_np = np.asarray(jax.device_get(rhs), dtype=float)
    rows = []
    for label, left, right in zip(OBJECTIVE_LABELS, lhs_np.tolist(), rhs_np.tolist()):
        delta = right - left
        rows.append(
            {
                "objective": label,
                "forward_primal": float(left),
                "reverse_primal": float(right),
                "delta_reverse_minus_forward": float(delta),
                "relative_delta": float(abs(delta) / max(abs(left), 1.0e-30)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="Benchmark TOML.")
    parser.add_argument("--device", type=str, default=None, help="Optional device override.")
    parser.add_argument("--accepted-step-limit", type=int, default=None)
    parser.add_argument("--ntx-exact-derivative-mode", default="direct", choices=("direct", "custom_vjp"))
    parser.add_argument(
        "--ntx-exact-derivative-field-pullback-mode",
        default=None,
        help="Accepted for parity with reverse benchmark config preparation.",
    )
    parser.add_argument(
        "--ntx-exact-derivative-pullback-algebra",
        default=None,
        help="Accepted for parity with reverse benchmark config preparation.",
    )
    parser.add_argument("--radau-jacobian-reuse-mode", type=str, default=None)
    parser.add_argument("--reverse-segment-length", type=int, default=None)
    parser.add_argument("--reverse-stage-adjoint-solve-mode", type=str, default="bicgstab")
    parser.add_argument("--reverse-rhs-transpose-mode", type=str, default="explicit_ntx_interpolated")
    parser.add_argument("--reverse-step-bwd-mode", type=str, default="reduced_cotangent")
    parser.add_argument(
        "--json-output",
        type=str,
        default=str(ROOT / "outputs" / "autodiff_transport_lagged_ntx" / "forward_reverse_primal_compare.json"),
    )
    args = parser.parse_args()

    config = _prepare_benchmark_config(
        Path(args.config),
        device=args.device,
        ntx_exact_derivative_mode=args.ntx_exact_derivative_mode,
        ntx_exact_derivative_field_pullback_mode=args.ntx_exact_derivative_field_pullback_mode,
        ntx_exact_derivative_pullback_algebra=args.ntx_exact_derivative_pullback_algebra,
        radau_jacobian_reuse_mode=args.radau_jacobian_reuse_mode,
    )
    runtime, baseline_state = build_runtime_context(config)
    profile_cfg = _baseline_profile_cfg(config)
    baseline_values = jnp.asarray([float(profile_cfg[name]) for name in PARAMETER_ORDER], dtype=jnp.float64)

    forward_objectives = _adaptive_rollout_objectives_realized_schedule_only_for_parameter(
        baseline_values[0],
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=PARAMETER_ORDER[0],
        accepted_step_limit_override=args.accepted_step_limit,
        derivative_mode="jvp",
    )
    forward_objectives = jax.block_until_ready(forward_objectives)

    fused_forward_objective_fn = lambda p: _forward_benchmark_adaptive_rollout_objectives_realized_schedule_only_for_parameter_jvp(  # noqa: E731
        p,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=PARAMETER_ORDER[0],
        accepted_step_limit_override=args.accepted_step_limit,
    )
    fused_forward_plain_objectives = fused_forward_objective_fn(baseline_values[0])
    fused_forward_plain_objectives = jax.block_until_ready(fused_forward_plain_objectives)
    fused_forward_objectives, fused_forward_tangents = jax.jvp(
        fused_forward_objective_fn,
        (baseline_values[0],),
        (jnp.asarray(1.0, dtype=baseline_values.dtype),),
    )
    fused_forward_objectives = jax.block_until_ready(fused_forward_objectives)
    fused_forward_tangents = jax.block_until_ready(fused_forward_tangents)

    reverse_setup = _prepare_reverse_static_setup(
        baseline_values,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        accepted_step_limit_override=args.accepted_step_limit,
        reverse_segment_length=args.reverse_segment_length,
        reverse_direct_stage_adjoint=True,
        reverse_stage_adjoint_solve_mode=args.reverse_stage_adjoint_solve_mode,
        reverse_rhs_transpose_mode=args.reverse_rhs_transpose_mode,
        reverse_step_bwd_mode=args.reverse_step_bwd_mode,
    )
    reverse_initial_carry = _reverse_initial_carry_for_parameter_vector(
        baseline_values,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        reverse_setup=reverse_setup,
    )
    reverse_final_y_vjp = _radau_adaptive_final_y_realized_schedule_vjp(
        reverse_setup.execution_context,
        reverse_setup.max_total_steps,
        reverse_setup.stop_after_accepted_steps,
        reverse_setup.reverse_segment_length,
        reverse_initial_carry,
    )
    reverse_state_vjp = reverse_setup.prepared_rollout.physics_context.unpack_flat(reverse_final_y_vjp)
    reverse_objectives_vjp = jax.block_until_ready(_objective_vector(reverse_state_vjp, runtime))

    reverse_final_y_plain = _radau_adaptive_final_y_realized_schedule(
        reverse_setup.execution_context,
        reverse_setup.max_total_steps,
        reverse_setup.stop_after_accepted_steps,
        reverse_initial_carry,
    )
    reverse_state_plain = reverse_setup.prepared_rollout.physics_context.unpack_flat(reverse_final_y_plain)
    reverse_objectives_plain = jax.block_until_ready(_objective_vector(reverse_state_plain, runtime))

    rows_vjp = _objective_stats(forward_objectives, reverse_objectives_vjp)
    rows_plain = _objective_stats(forward_objectives, reverse_objectives_plain)
    rows_fused_vjp = _objective_stats(fused_forward_objectives, reverse_objectives_vjp)
    rows_fused_plain = _objective_stats(fused_forward_objectives, reverse_objectives_plain)
    rows_fused_plain_call = _objective_stats(fused_forward_plain_objectives, fused_forward_objectives)
    max_rel_vjp = max(row["relative_delta"] for row in rows_vjp)
    max_rel_plain = max(row["relative_delta"] for row in rows_plain)
    max_rel_fused_vjp = max(row["relative_delta"] for row in rows_fused_vjp)
    max_rel_fused_plain = max(row["relative_delta"] for row in rows_fused_plain)

    print("[compare] forward realized-schedule primal vs reverse VJP primal")
    for row in rows_vjp:
        print(
            f"  - {row['objective']}: forward={row['forward_primal']:.16e} "
            f"reverse_vjp={row['reverse_primal']:.16e} "
            f"delta={row['delta_reverse_minus_forward']:.6e} "
            f"rel={row['relative_delta']:.6e}"
        )
    print("[compare] forward realized-schedule primal vs reverse plain primal")
    for row in rows_plain:
        print(
            f"  - {row['objective']}: forward={row['forward_primal']:.16e} "
            f"reverse_plain={row['reverse_primal']:.16e} "
            f"delta={row['delta_reverse_minus_forward']:.6e} "
            f"rel={row['relative_delta']:.6e}"
        )
    print("[compare] fused forward custom-JVP primal vs reverse VJP primal")
    for row in rows_fused_vjp:
        print(
            f"  - {row['objective']}: fused_forward={row['forward_primal']:.16e} "
            f"reverse_vjp={row['reverse_primal']:.16e} "
            f"delta={row['delta_reverse_minus_forward']:.6e} "
            f"rel={row['relative_delta']:.6e}"
        )
    print("[compare] fused forward wrapper plain-call primal vs JVP primal")
    for row in rows_fused_plain_call:
        print(
            f"  - {row['objective']}: plain_call={row['forward_primal']:.16e} "
            f"jvp_primal={row['reverse_primal']:.16e} "
            f"delta={row['delta_reverse_minus_forward']:.6e} "
            f"rel={row['relative_delta']:.6e}"
        )
    print("[compare] fused forward custom-JVP primal vs reverse plain primal")
    for row in rows_fused_plain:
        print(
            f"  - {row['objective']}: fused_forward={row['forward_primal']:.16e} "
            f"reverse_plain={row['reverse_primal']:.16e} "
            f"delta={row['delta_reverse_minus_forward']:.6e} "
            f"rel={row['relative_delta']:.6e}"
        )
    print("[compare] fused forward n0 tangents")
    for label, value in zip(OBJECTIVE_LABELS, np.asarray(jax.device_get(fused_forward_tangents), dtype=float)):
        print(f"  - {label}: tangent={float(value):.16e}")

    payload = {
        "config_path": str(Path(args.config)),
        "accepted_step_limit": args.accepted_step_limit,
        "parameter_order": list(PARAMETER_ORDER),
        "baseline_values": np.asarray(jax.device_get(baseline_values), dtype=float).tolist(),
        "reverse_max_total_steps": int(reverse_setup.max_total_steps),
        "reverse_stop_after_accepted_steps": reverse_setup.stop_after_accepted_steps,
        "forward_vs_reverse_vjp": rows_vjp,
        "forward_vs_reverse_plain": rows_plain,
        "fused_forward_vs_reverse_vjp": rows_fused_vjp,
        "fused_forward_vs_reverse_plain": rows_fused_plain,
        "fused_forward_plain_call_vs_jvp_primal": rows_fused_plain_call,
        "fused_forward_tangent_n0": np.asarray(jax.device_get(fused_forward_tangents), dtype=float).tolist(),
        "max_relative_delta_vjp": float(max_rel_vjp),
        "max_relative_delta_plain": float(max_rel_plain),
        "max_relative_delta_fused_vjp": float(max_rel_fused_vjp),
        "max_relative_delta_fused_plain": float(max_rel_fused_plain),
    }
    out_path = Path(args.json_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[compare] max rel vjp={max_rel_vjp:.6e} plain={max_rel_plain:.6e}")
    print(f"[compare] wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
