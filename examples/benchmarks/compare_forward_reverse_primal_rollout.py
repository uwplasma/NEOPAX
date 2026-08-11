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


def _fused_forward_objective_fn_for_mode(
    mode: str,
    *,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    accepted_step_limit_override: int | None,
):
    if mode == "replay":
        return lambda p: _forward_benchmark_adaptive_rollout_objectives_realized_schedule_only_for_parameter_jvp(  # noqa: E731
            p,
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=PARAMETER_ORDER[0],
            accepted_step_limit_override=accepted_step_limit_override,
        )
    if mode == "step":
        return lambda p: _adaptive_rollout_objectives_realized_schedule_only_for_parameter(  # noqa: E731
            p,
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=PARAMETER_ORDER[0],
            accepted_step_limit_override=accepted_step_limit_override,
            derivative_mode="jvp_step",
        )
    if mode == "accepted_replay":
        return lambda p: _adaptive_rollout_objectives_realized_schedule_only_for_parameter(  # noqa: E731
            p,
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=PARAMETER_ORDER[0],
            accepted_step_limit_override=accepted_step_limit_override,
            derivative_mode="accepted_replay",
        )
    if mode == "exact":
        return lambda p: _adaptive_rollout_objectives_realized_schedule_only_for_parameter(  # noqa: E731
            p,
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=PARAMETER_ORDER[0],
            accepted_step_limit_override=accepted_step_limit_override,
            derivative_mode="jvp_exact",
        )
    raise ValueError(f"Unknown forward AD fusion mode: {mode}")


def _forward_tangents_for_mode(
    mode: str,
    *,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    baseline_value,
    accepted_step_limit_override: int | None,
):
    objective_fn = _fused_forward_objective_fn_for_mode(
        mode,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        accepted_step_limit_override=accepted_step_limit_override,
    )
    plain_objectives = jax.block_until_ready(objective_fn(baseline_value))
    jvp_objectives, tangents = jax.jvp(
        objective_fn,
        (baseline_value,),
        (jnp.asarray(1.0, dtype=jnp.asarray(baseline_value).dtype),),
    )
    return (
        jax.block_until_ready(plain_objectives),
        jax.block_until_ready(jvp_objectives),
        jax.block_until_ready(tangents),
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


def _objective_values_from_reference_payload(payload: dict[str, Any]) -> dict[str, float]:
    raw_values = payload.get("objective_values")
    if not isinstance(raw_values, dict):
        raw_values = payload.get("objective_values_by_objective")
    if isinstance(raw_values, dict):
        return {
            str(label): float(value)
            for label, value in raw_values.items()
            if label in OBJECTIVE_LABELS and value is not None
        }
    if isinstance(raw_values, list):
        return {
            label: float(value)
            for label, value in zip(OBJECTIVE_LABELS, raw_values)
            if value is not None
        }
    return {}


def _reference_objective_consistency_rows(
    reference_objective_values: dict[str, float],
    live_objectives,
) -> tuple[list[dict[str, Any]], float, float]:
    live_np = np.asarray(jax.device_get(live_objectives), dtype=float)
    rows = []
    for label, live_value in zip(OBJECTIVE_LABELS, live_np.tolist()):
        if label not in reference_objective_values:
            continue
        reference_value = float(reference_objective_values[label])
        delta = float(live_value - reference_value)
        rows.append(
            {
                "objective": label,
                "reference_value": reference_value,
                "live_reverse_primal": float(live_value),
                "delta_live_minus_reference": delta,
                "relative_delta": float(
                    abs(delta) / max(abs(float(live_value)), abs(reference_value), 1.0e-30)
                ),
            }
        )
    if not rows:
        return rows, float("nan"), float("nan")
    return (
        rows,
        max(abs(row["delta_live_minus_reference"]) for row in rows),
        max(row["relative_delta"] for row in rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="Benchmark TOML.")
    parser.add_argument("--device", type=str, default=None, help="Optional device override.")
    parser.add_argument("--accepted-step-limit", type=int, default=None)
    parser.add_argument("--ntx-exact-derivative-mode", default="direct", choices=("direct",))
    parser.add_argument(
        "--ntx-exact-derivative-field-pullback-mode",
        default="compact_vjp",
        help="Accepted for parity with reverse benchmark config preparation.",
    )
    parser.add_argument(
        "--ntx-exact-derivative-pullback-algebra",
        default="ntx_helper",
        help="Accepted for parity with reverse benchmark config preparation.",
    )
    parser.add_argument("--radau-jacobian-reuse-mode", type=str, default=None)
    parser.add_argument("--reverse-segment-length", type=int, default=None)
    parser.add_argument("--reverse-stage-adjoint-solve-mode", type=str, default="bicgstab")
    parser.add_argument("--reverse-rhs-transpose-mode", type=str, default="explicit_ntx_interpolated")
    parser.add_argument("--reverse-step-bwd-mode", type=str, default="reduced_cotangent")
    parser.add_argument(
        "--forward-ad-fusion-mode",
        type=str,
        default="replay",
        choices=("replay", "accepted_replay", "step", "exact"),
        help=(
            "Forward custom-JVP tangent path to diagnose. 'replay' is the recovered "
            "reference replay helper; 'accepted_replay' uses the static setup accepted/rejected "
            "trace replay path; 'step' uses the step-fused accepted-step path; 'exact' "
            "differentiates raw accepted-step attempts on the frozen realized schedule."
        ),
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=str(ROOT / "outputs" / "autodiff_transport_lagged_ntx" / "forward_reverse_primal_compare.json"),
    )
    parser.add_argument(
        "--compare-forward-fusion-modes",
        action="store_true",
        help="Also run replay, step, and exact forward custom-JVP tangent paths and print their tangent deltas.",
    )
    parser.add_argument(
        "--primal-only",
        action="store_true",
        help="Skip forward JVP tangent evaluation and only compare primal objective values.",
    )
    parser.add_argument(
        "--reference-gradient-json",
        type=str,
        default=None,
        help=(
            "Optional reverse benchmark JSON with gradient_reverse_ad_by_objective. If provided, "
            "forward n0 tangents are compared against the reverse n0 column."
        ),
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

    if args.primal_only:
        fused_objective_fn = _fused_forward_objective_fn_for_mode(
            str(args.forward_ad_fusion_mode),
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            accepted_step_limit_override=args.accepted_step_limit,
        )
        fused_forward_plain_objectives = jax.block_until_ready(fused_objective_fn(baseline_values[0]))
        fused_forward_objectives = fused_forward_plain_objectives
        fused_forward_tangents = None
    else:
        (
            fused_forward_plain_objectives,
            fused_forward_objectives,
            fused_forward_tangents,
        ) = _forward_tangents_for_mode(
            str(args.forward_ad_fusion_mode),
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            baseline_value=baseline_values[0],
            accepted_step_limit_override=args.accepted_step_limit,
        )

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
    print(f"[compare] fused forward custom-JVP primal vs reverse VJP primal ({args.forward_ad_fusion_mode})")
    for row in rows_fused_vjp:
        print(
            f"  - {row['objective']}: fused_forward={row['forward_primal']:.16e} "
            f"reverse_vjp={row['reverse_primal']:.16e} "
            f"delta={row['delta_reverse_minus_forward']:.6e} "
            f"rel={row['relative_delta']:.6e}"
        )
    print(f"[compare] fused forward wrapper plain-call primal vs JVP primal ({args.forward_ad_fusion_mode})")
    for row in rows_fused_plain_call:
        print(
            f"  - {row['objective']}: plain_call={row['forward_primal']:.16e} "
            f"jvp_primal={row['reverse_primal']:.16e} "
            f"delta={row['delta_reverse_minus_forward']:.6e} "
            f"rel={row['relative_delta']:.6e}"
        )
    print(f"[compare] fused forward custom-JVP primal vs reverse plain primal ({args.forward_ad_fusion_mode})")
    for row in rows_fused_plain:
        print(
            f"  - {row['objective']}: fused_forward={row['forward_primal']:.16e} "
            f"reverse_plain={row['reverse_primal']:.16e} "
            f"delta={row['delta_reverse_minus_forward']:.6e} "
            f"rel={row['relative_delta']:.6e}"
        )
    if fused_forward_tangents is not None:
        print(f"[compare] fused forward n0 tangents ({args.forward_ad_fusion_mode})")
        for label, value in zip(OBJECTIVE_LABELS, np.asarray(jax.device_get(fused_forward_tangents), dtype=float)):
            print(f"  - {label}: tangent={float(value):.16e}")

    forward_fusion_tangent_compare = None
    reference_reverse_n0 = None
    reference_objective_consistency = None
    if args.reference_gradient_json is not None:
        ref_payload = json.loads(Path(args.reference_gradient_json).read_text(encoding="utf-8"))
        reference_objective_values = _objective_values_from_reference_payload(ref_payload)
        if reference_objective_values:
            consistency_rows, max_reference_abs, max_reference_rel = _reference_objective_consistency_rows(
                reference_objective_values,
                reverse_objectives_vjp,
            )
            reference_objective_consistency = {
                "rows": consistency_rows,
                "max_absolute_delta": float(max_reference_abs),
                "max_relative_delta": float(max_reference_rel),
                "warning": bool(max_reference_abs > 1.0e-8 or max_reference_rel > 1.0e-8),
            }
            print(
                "[compare] reference-gradient JSON primal consistency: "
                f"max_abs={max_reference_abs:.6e} max_rel={max_reference_rel:.6e}"
            )
            if reference_objective_consistency["warning"]:
                print(
                    "[compare] WARNING reference-gradient JSON objective values do not match "
                    "the live reverse primal; tangent-vs-reverse comparisons may be stale."
                )
        else:
            reference_objective_consistency = {
                "rows": [],
                "max_absolute_delta": None,
                "max_relative_delta": None,
                "warning": True,
                "message": "reference JSON has no objective_values block to validate against live primal",
            }
            print(
                "[compare] WARNING reference-gradient JSON has no objective_values block; "
                "cannot verify whether gradient comparisons are from the current primal."
            )
        gradient_by_objective = ref_payload.get(
            "gradient_reverse_ad_by_objective",
            ref_payload.get("gradient_by_objective", {}),
        )
        reference_reverse_n0 = {
            label: float(gradient_by_objective[label][PARAMETER_ORDER[0]])
            for label in OBJECTIVE_LABELS
            if label in gradient_by_objective and PARAMETER_ORDER[0] in gradient_by_objective[label]
        }

    if args.compare_forward_fusion_modes and not args.primal_only:
        _, replay_objectives, replay_tangents = _forward_tangents_for_mode(
            "replay",
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            baseline_value=baseline_values[0],
            accepted_step_limit_override=args.accepted_step_limit,
        )
        _, step_objectives, step_tangents = _forward_tangents_for_mode(
            "step",
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            baseline_value=baseline_values[0],
            accepted_step_limit_override=args.accepted_step_limit,
        )
        _, exact_objectives, exact_tangents = _forward_tangents_for_mode(
            "exact",
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            baseline_value=baseline_values[0],
            accepted_step_limit_override=args.accepted_step_limit,
        )
        replay_np = np.asarray(jax.device_get(replay_tangents), dtype=float)
        step_np = np.asarray(jax.device_get(step_tangents), dtype=float)
        exact_np = np.asarray(jax.device_get(exact_tangents), dtype=float)
        objective_rows = _objective_stats(replay_objectives, step_objectives)
        exact_objective_rows = _objective_stats(replay_objectives, exact_objectives)
        forward_fusion_tangent_compare = []
        print("[compare] forward custom-JVP n0 tangent mode comparison")
        for label, replay_value, step_value, exact_value in zip(
            OBJECTIVE_LABELS,
            replay_np.tolist(),
            step_np.tolist(),
            exact_np.tolist(),
        ):
            delta_step_minus_replay = float(step_value - replay_value)
            delta_exact_minus_replay = float(exact_value - replay_value)
            row = {
                "objective": label,
                "replay_tangent": float(replay_value),
                "step_tangent": float(step_value),
                "exact_tangent": float(exact_value),
                "delta_step_minus_replay": delta_step_minus_replay,
                "delta_exact_minus_replay": delta_exact_minus_replay,
                "relative_delta_step_vs_replay": float(
                    abs(delta_step_minus_replay) / max(abs(float(replay_value)), 1.0e-30)
                ),
                "relative_delta_exact_vs_replay": float(
                    abs(delta_exact_minus_replay) / max(abs(float(replay_value)), 1.0e-30)
                ),
            }
            if reference_reverse_n0 is not None and label in reference_reverse_n0:
                reverse_value = reference_reverse_n0[label]
                row["reverse_n0"] = reverse_value
                row["delta_replay_minus_reverse"] = float(replay_value - reverse_value)
                row["delta_step_minus_reverse"] = float(step_value - reverse_value)
                row["delta_exact_minus_reverse"] = float(exact_value - reverse_value)
                row["relative_delta_replay_vs_reverse"] = float(
                    abs(float(replay_value) - reverse_value) / max(abs(reverse_value), 1.0e-30)
                )
                row["relative_delta_step_vs_reverse"] = float(
                    abs(float(step_value) - reverse_value) / max(abs(reverse_value), 1.0e-30)
                )
                row["relative_delta_exact_vs_reverse"] = float(
                    abs(float(exact_value) - reverse_value) / max(abs(reverse_value), 1.0e-30)
                )
                print(
                    f"  - {label}: replay={float(replay_value):.16e} "
                    f"step={float(step_value):.16e} "
                    f"exact={float(exact_value):.16e} "
                    f"reverse={reverse_value:.16e} "
                    f"step-replay={delta_step_minus_replay:.6e} "
                    f"exact-replay={delta_exact_minus_replay:.6e} "
                    f"replay-rev={row['delta_replay_minus_reverse']:.6e} "
                    f"step-rev={row['delta_step_minus_reverse']:.6e} "
                    f"exact-rev={row['delta_exact_minus_reverse']:.6e}"
                )
            else:
                print(
                    f"  - {label}: replay={float(replay_value):.16e} "
                    f"step={float(step_value):.16e} "
                    f"exact={float(exact_value):.16e} "
                    f"step-replay={delta_step_minus_replay:.6e} "
                    f"exact-replay={delta_exact_minus_replay:.6e} "
                    f"rel={row['relative_delta_step_vs_replay']:.6e}"
                )
            forward_fusion_tangent_compare.append(row)
        print("[compare] forward custom-JVP replay primal vs step primal")
        for row in objective_rows:
            print(
                f"  - {row['objective']}: replay={row['forward_primal']:.16e} "
                f"step={row['reverse_primal']:.16e} "
                f"delta={row['delta_reverse_minus_forward']:.6e} "
                f"rel={row['relative_delta']:.6e}"
            )
        print("[compare] forward custom-JVP replay primal vs exact primal")
        for row in exact_objective_rows:
            print(
                f"  - {row['objective']}: replay={row['forward_primal']:.16e} "
                f"exact={row['reverse_primal']:.16e} "
                f"delta={row['delta_reverse_minus_forward']:.6e} "
                f"rel={row['relative_delta']:.6e}"
            )

    payload = {
        "config_path": str(Path(args.config)),
        "accepted_step_limit": args.accepted_step_limit,
        "parameter_order": list(PARAMETER_ORDER),
        "baseline_values": np.asarray(jax.device_get(baseline_values), dtype=float).tolist(),
        "reverse_max_total_steps": int(reverse_setup.max_total_steps),
        "reverse_stop_after_accepted_steps": reverse_setup.stop_after_accepted_steps,
        "forward_ad_fusion_mode": str(args.forward_ad_fusion_mode),
        "forward_vs_reverse_vjp": rows_vjp,
        "forward_vs_reverse_plain": rows_plain,
        "fused_forward_vs_reverse_vjp": rows_fused_vjp,
        "fused_forward_vs_reverse_plain": rows_fused_plain,
        "fused_forward_plain_call_vs_jvp_primal": rows_fused_plain_call,
        "fused_forward_tangent_n0": (
            None
            if fused_forward_tangents is None
            else np.asarray(jax.device_get(fused_forward_tangents), dtype=float).tolist()
        ),
        "reference_gradient_json": args.reference_gradient_json,
        "reference_objective_consistency": reference_objective_consistency,
        "forward_fusion_tangent_compare_n0": forward_fusion_tangent_compare,
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
