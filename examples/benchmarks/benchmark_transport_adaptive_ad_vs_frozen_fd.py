from __future__ import annotations

import argparse
import json
import os
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
    _accepted_time_list_from_trace,
    _adaptive_rollout_diagnostics,
    _adaptive_rollout_objectives_for_parameter_on_time_list,
    _baseline_profile_cfg,
    _fd_step,
    _frozen_replay_nonfinite_debug,
    _forward_benchmark_adaptive_rollout_final_state_for_parameter,
    _forward_benchmark_adaptive_realized_schedule_replay_primal_debug_for_parameter,
    _forward_benchmark_adaptive_realized_schedule_jvp_stage_debug_for_parameter,
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


def _scalar_objectives_finite(values) -> bool:
    arr = np.asarray(jax.device_get(values), dtype=float)
    return bool(np.all(np.isfinite(arr)))


def _ad_lane_local_nan_diagnostics(
    *,
    baseline_value: float,
    objective_fn,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    parameter_name: str,
    accepted_step_limit: int | None,
) -> dict[str, Any]:
    primal_objectives = objective_fn(jnp.asarray(baseline_value))
    final_state, rollout = _forward_benchmark_adaptive_rollout_final_state_for_parameter(
        jnp.asarray(baseline_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        use_realized_schedule_jvp=True,
        accepted_step_limit_override=accepted_step_limit,
    )
    return {
        "primal_objectives": _to_float_list(primal_objectives),
        "primal_objectives_finite": _scalar_objectives_finite(primal_objectives),
        "final_state_finite": _tree_all_finite(final_state),
        "rollout": _adaptive_rollout_diagnostics(rollout),
    }


def _ad_lane_stage_debug(
    *,
    baseline_value: float,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    parameter_name: str,
    accepted_step_limit: int | None,
) -> dict[str, Any]:
    return _forward_benchmark_adaptive_realized_schedule_jvp_stage_debug_for_parameter(
        jnp.asarray(baseline_value),
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
        accepted_step_limit_override=accepted_step_limit,
    )


def _print_summary(report: dict[str, Any]) -> None:
    print(
        f"[autodiff-gate] mode=adaptive_ad_vs_frozen_fd "
        f"parameter={report['parameter_name']} "
        f"baseline_value={report['baseline_value']:.6e} "
        f"fd_step={report['fd_step']:.6e} "
        f"replay_mode={report['replay_mode']}"
    )
    solver_settings = report.get("solver_settings")
    if solver_settings is not None:
        print(
            "[autodiff-gate] solver settings: "
            f"backend={solver_settings.get('backend')} "
            f"integrator={solver_settings.get('integrator')} "
            f"radau_rhs_mode={solver_settings.get('radau_rhs_mode')} "
            f"radau_num_stages={solver_settings.get('radau_num_stages')} "
            f"t0={solver_settings.get('t0')} "
            f"t_final={solver_settings.get('t_final')} "
            f"dt={solver_settings.get('dt')}"
        )
    geometry_settings = report.get("geometry_settings")
    if geometry_settings is not None:
        print(
            "[autodiff-gate] geometry settings: "
            f"backend={geometry_settings.get('backend')} "
            f"vmec_file={geometry_settings.get('vmec_file')} "
            f"boozer_file={geometry_settings.get('boozer_file')} "
            f"ntx_exact_surface_backend={geometry_settings.get('ntx_exact_surface_backend')}"
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
        baseline_fixed = frozen_diag.get("baseline_fixed_dt", {})
        if baseline_fixed:
            print(
                "[autodiff-gate] frozen replay baseline fixed-dt: "
                f"objectives_finite={baseline_fixed.get('objectives_finite')} "
                f"final_state_finite={baseline_fixed.get('final_state_finite')} "
                f"final_carry_finite={baseline_fixed.get('final_carry_finite')}"
            )
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
        minus_nf = minus.get("nonfinite_debug")
        if minus_nf is not None:
            print(
                "[autodiff-gate] frozen replay minus nonfinite debug: "
                f"first_bad_index={minus_nf.get('first_bad_index')} "
                f"first_bad_was_accepted={minus_nf.get('first_bad_was_accepted')} "
                f"first_bad_accepted_ordinal={minus_nf.get('first_bad_accepted_ordinal')} "
                f"first_bad_dt={minus_nf.get('first_bad_dt')} "
                f"final_state_finite={minus_nf.get('final_state_finite')} "
                f"objectives_finite={minus_nf.get('objectives_finite')}"
            )
            for entry in (minus_nf.get("local_attempt_window") or []):
                print(
                    "  "
                    f"index={entry.get('index')} "
                    f"accepted={entry.get('accepted')} "
                    f"attempted_dt={entry.get('attempted_dt')} "
                    f"next_dt={entry.get('next_dt')} "
                    f"time={entry.get('time')} "
                    f"baseline_err_norm={entry.get('baseline_err_norm')} "
                    f"replay_state_finite={entry.get('replay_state_finite')}"
                )
            local_accepted_window = minus_nf.get("local_accepted_window") or []
            if local_accepted_window:
                print("  accepted-window:",)
                for entry in local_accepted_window:
                    print(
                        "  "
                        f"accepted_ordinal={entry.get('accepted_ordinal')} "
                        f"trace_index={entry.get('trace_index')} "
                        f"attempted_dt={entry.get('attempted_dt')} "
                        f"next_dt={entry.get('next_dt')} "
                        f"time={entry.get('time')} "
                        f"baseline_err_norm={entry.get('baseline_err_norm')} "
                        f"replay_state_finite={entry.get('replay_state_finite')}"
                    )
        plus_nf = plus.get("nonfinite_debug")
        if plus_nf is not None:
            print(
                "[autodiff-gate] frozen replay plus nonfinite debug: "
                f"first_bad_index={plus_nf.get('first_bad_index')} "
                f"first_bad_was_accepted={plus_nf.get('first_bad_was_accepted')} "
                f"first_bad_accepted_ordinal={plus_nf.get('first_bad_accepted_ordinal')} "
                f"first_bad_dt={plus_nf.get('first_bad_dt')} "
                f"final_state_finite={plus_nf.get('final_state_finite')} "
                f"objectives_finite={plus_nf.get('objectives_finite')}"
            )
            for entry in (plus_nf.get("local_attempt_window") or []):
                print(
                    "  "
                    f"index={entry.get('index')} "
                    f"accepted={entry.get('accepted')} "
                    f"attempted_dt={entry.get('attempted_dt')} "
                    f"next_dt={entry.get('next_dt')} "
                    f"time={entry.get('time')} "
                    f"baseline_err_norm={entry.get('baseline_err_norm')} "
                    f"replay_state_finite={entry.get('replay_state_finite')}"
                )
            local_accepted_window = plus_nf.get("local_accepted_window") or []
            if local_accepted_window:
                print("  accepted-window:",)
                for entry in local_accepted_window:
                    print(
                        "  "
                        f"accepted_ordinal={entry.get('accepted_ordinal')} "
                        f"trace_index={entry.get('trace_index')} "
                        f"attempted_dt={entry.get('attempted_dt')} "
                        f"next_dt={entry.get('next_dt')} "
                        f"time={entry.get('time')} "
                        f"baseline_err_norm={entry.get('baseline_err_norm')} "
                        f"replay_state_finite={entry.get('replay_state_finite')}"
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
    parser.add_argument(
        "--ad-local-nan-debug",
        action="store_true",
        help="When AD returns nonfinite values, run lane-local primal/final-state diagnostics in this benchmark entrypoint.",
    )
    parser.add_argument(
        "--ad-stage-debug",
        action="store_true",
        help="When AD returns nonfinite values, localize whether the forward scalar JVP first breaks in final_y_dot, unpacked state dot, or objective tangent.",
    )
    parser.add_argument(
        "--ntx-local-pullback-finite-debug",
        action="store_true",
        help="Enable NTX local pullback finite debug prints via NEOPAX_TRANSPORT_NTX_LOCAL_PULLBACK_FINITE_DEBUG=1.",
    )
    args = parser.parse_args()

    if args.ntx_local_pullback_finite_debug:
        os.environ["NEOPAX_TRANSPORT_NTX_LOCAL_PULLBACK_FINITE_DEBUG"] = "1"

    config = _prepare_benchmark_config(
        Path(args.config),
        device=args.device,
        ntx_exact_derivative_mode=args.ntx_exact_derivative_mode,
    )
    runtime, baseline_state = build_runtime_context(config)
    profile_cfg = _baseline_profile_cfg(config)
    baseline_value = float(profile_cfg[args.parameter])
    fd_step = _fd_step(baseline_value, rel_step=args.fd_rel_step, abs_step=args.fd_abs_step)
    solver_cfg = dict(config.get("transport_solver", {}))
    geometry_cfg = dict(config.get("geometry", {}))
    neoclassical_cfg = dict(config.get("neoclassical", {}))

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
            use_schedule_trace_only=True,
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
    ad_local_nan_debug = None
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
        if args.ad_local_nan_debug:
            grad_ad_np_local = np.asarray(jax.device_get(gradient_ad), dtype=float)
            if not np.all(np.isfinite(grad_ad_np_local)):
                print(
                    "[autodiff-gate] progress: AD produced nonfinite values; running lane-local diagnostics",
                    flush=True,
                )
                ad_local_nan_debug = _ad_lane_local_nan_diagnostics(
                    baseline_value=baseline_value,
                    objective_fn=objective_fn,
                    config=config,
                    runtime=runtime,
                    baseline_state=baseline_state,
                    profile_cfg=profile_cfg,
                    parameter_name=args.parameter,
                    accepted_step_limit=args.accepted_step_limit,
                )
                rollout_diag = ad_local_nan_debug["rollout"]
                print(
                    "[autodiff-gate] ad local debug: "
                    f"primal_objectives_finite={ad_local_nan_debug['primal_objectives_finite']} "
                    f"final_state_finite={ad_local_nan_debug['final_state_finite']} "
                    f"attempt_count={rollout_diag.get('attempt_count')} "
                    f"accepted_count={rollout_diag.get('accepted_count')} "
                    f"completed={rollout_diag.get('completed')} "
                    f"failed={rollout_diag.get('failed')} "
                    f"fail_code={rollout_diag.get('fail_code')}",
                    flush=True,
                )
                if args.ad_stage_debug and str(args.adaptive_derivative_mode).strip().lower() == "jvp":
                    stage_debug = _ad_lane_stage_debug(
                        baseline_value=baseline_value,
                        config=config,
                        runtime=runtime,
                        baseline_state=baseline_state,
                        profile_cfg=profile_cfg,
                        parameter_name=args.parameter,
                        accepted_step_limit=args.accepted_step_limit,
                    )
                    print(
                        "[autodiff-gate] ad stage debug: "
                        f"final_y_all_finite={stage_debug['final_y_all_finite']} "
                        f"final_y_dot_all_finite={stage_debug['final_y_dot_all_finite']} "
                        f"final_state_all_finite={stage_debug['final_state_all_finite']} "
                        f"final_state_dot_all_finite={stage_debug['final_state_dot_all_finite']} "
                        f"objective_primal_all_finite={stage_debug['objective_primal_all_finite']} "
                        f"objective_tangent_all_finite={stage_debug['objective_tangent_all_finite']}",
                        flush=True,
                    )
                    replay_primal_debug = _forward_benchmark_adaptive_realized_schedule_replay_primal_debug_for_parameter(
                        jnp.asarray(baseline_value),
                        config=config,
                        runtime=runtime,
                        baseline_state=baseline_state,
                        profile_cfg=profile_cfg,
                        parameter_name=args.parameter,
                        accepted_step_limit_override=args.accepted_step_limit,
                    )
                    print(
                        "[autodiff-gate] ad replay primal debug: "
                        f"attempt_count={replay_primal_debug.get('attempt_count')} "
                        f"accepted_count={replay_primal_debug.get('accepted_count')} "
                        f"first_bad_index={replay_primal_debug.get('first_bad_index')} "
                        f"first_bad_was_accepted={replay_primal_debug.get('first_bad_was_accepted')} "
                        f"first_bad_dt={replay_primal_debug.get('first_bad_dt')} "
                        f"final_state_finite={replay_primal_debug.get('final_state_finite')} "
                        f"objectives_finite={replay_primal_debug.get('objectives_finite')}",
                        flush=True,
                    )
                    local_window = replay_primal_debug.get("local_attempt_window") or []
                    if local_window:
                        print("[autodiff-gate] ad replay primal local window:", flush=True)
                        for entry in local_window:
                            print(
                                "  "
                                f"index={entry.get('index')} "
                                f"accepted={entry.get('accepted')} "
                                f"attempted_dt={entry.get('attempted_dt')} "
                                f"next_dt={entry.get('next_dt')} "
                                f"time={entry.get('time')} "
                                f"baseline_err_norm={entry.get('baseline_err_norm')} "
                                f"replay_state_finite={entry.get('replay_state_finite')}",
                                flush=True,
                            )

    minus_value = baseline_value - fd_step
    plus_value = baseline_value + fd_step

    objectives_minus = None
    objectives_plus = None
    minus_replay = None
    plus_replay = None
    baseline_fixed_dt_replay = None
    baseline_fixed_dt_objectives = None
    gradient_fd = None
    minus_nonfinite_debug = None
    plus_nonfinite_debug = None
    if args.run_mode in ("both", "fd"):
        if str(args.replay_mode).strip().lower() == "accepted":
            accepted_time_list = _accepted_time_list_from_trace(replay_trace)
            baseline_fixed_dt_objectives, baseline_fixed_dt_replay = _adaptive_rollout_objectives_for_parameter_on_time_list(
                jnp.asarray(baseline_value),
                config=config,
                runtime=runtime,
                baseline_state=baseline_state,
                profile_cfg=profile_cfg,
                parameter_name=args.parameter,
                time_list=accepted_time_list,
            )
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
        minus_diag_local = _replay_diagnostics(minus_replay, objectives_minus)
        plus_diag_local = _replay_diagnostics(plus_replay, objectives_plus)
        if not minus_diag_local["final_state_finite"] or not minus_diag_local["objectives_finite"]:
            minus_nonfinite_debug = _frozen_replay_nonfinite_debug(
                minus_replay,
                replay_trace,
                objectives_np=np.asarray(jax.device_get(objectives_minus), dtype=float),
            )
        if not plus_diag_local["final_state_finite"] or not plus_diag_local["objectives_finite"]:
            plus_nonfinite_debug = _frozen_replay_nonfinite_debug(
                plus_replay,
                replay_trace,
                objectives_np=np.asarray(jax.device_get(objectives_plus), dtype=float),
            )

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
        "solver_settings": {
            "backend": solver_cfg.get("transport_solver_backend"),
            "integrator": solver_cfg.get("integrator"),
            "radau_rhs_mode": solver_cfg.get("radau_rhs_mode"),
            "radau_num_stages": solver_cfg.get("radau_num_stages"),
            "t0": solver_cfg.get("t0"),
            "t_final": solver_cfg.get("t_final"),
            "dt": solver_cfg.get("dt"),
        },
        "geometry_settings": {
            "backend": geometry_cfg.get("backend"),
            "vmec_file": geometry_cfg.get("vmec_file"),
            "boozer_file": geometry_cfg.get("boozer_file"),
            "ntx_exact_surface_backend": neoclassical_cfg.get("ntx_exact_surface_backend"),
        },
        "objective_labels": OBJECTIVE_LABELS,
        "gradient_autodiff": None if grad_ad_np is None else grad_ad_np.tolist(),
        "gradient_fd": None if grad_fd_np is None else grad_fd_np.tolist(),
        "ad_local_nan_debug": ad_local_nan_debug,
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
            "baseline_fixed_dt": (
                {}
                if baseline_fixed_dt_replay is None
                else _replay_diagnostics(baseline_fixed_dt_replay, baseline_fixed_dt_objectives)
            ),
            "minus": _replay_diagnostics(minus_replay, objectives_minus) | {"nonfinite_debug": minus_nonfinite_debug},
            "plus": _replay_diagnostics(plus_replay, objectives_plus) | {"nonfinite_debug": plus_nonfinite_debug},
        },
    }

    _print_summary(report)
    outpath = _report_path(args.parameter)
    outpath.write_text(json.dumps(report, indent=2))
    print(f"Wrote {outpath.relative_to(ROOT)}")
    print(f"parameter={args.parameter} passed={report['passed']} max_rel_error={report['max_relative_error']}")


if __name__ == "__main__":
    main()
