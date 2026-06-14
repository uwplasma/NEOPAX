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
    DEFAULT_CONFIG,
    OBJECTIVE_LABELS,
    PROFILE_VECTOR_PARAMETERS,
    _adaptive_rollout_nan_debug_for_parameter,
    _adaptive_rollout_objectives_realized_schedule_only_for_parameter_vector,
    _build_prepared_radau_accepted_rollout,
    _build_prepared_radau_execution_context,
    _forward_benchmark_prepare_realized_schedule_profile_vector_rollout,
    _objective_vector,
    _baseline_profile_cfg,
    _prepare_benchmark_config,
)
from NEOPAX._orchestrator import build_runtime_context  # noqa: E402
from NEOPAX._orchestrator import prepare_transport_solver_components  # noqa: E402
from NEOPAX._transport_solvers import _radau_forward_adaptive_final_y_realized_schedule  # noqa: E402

BENCHMARK_VERSION_TAG = "profile_vector_ad_compare_v2026_06_15_forward_reverse_clean_split"


def _report_path() -> Path:
    outdir = ROOT / "outputs" / "autodiff_transport_lagged_ntx" / "profile_vector"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / "transport_profile_vector_ad_compare_summary.json"


def _print_summary(report: dict[str, Any]) -> None:
    print(
        "[autodiff-gate] "
        f"mode=profile_vector_ad_compare parameters={list(report['parameter_names'])}"
    )
    print("[autodiff-gate] max relative error by parameter:")
    for name, value in zip(report["parameter_names"], report["parameter_max_relative_error"]):
        print(f"  - {name}: max_rel_err={float(value):.6e}")
    print("[autodiff-gate] metric-by-parameter errors:")
    for metric_index, label in enumerate(report["objective_labels"]):
        print(f"  - {label}:")
        for param_index, name in enumerate(report["parameter_names"]):
            fwd = report["jacobian_forward"][metric_index][param_index]
            rev = report["jacobian_reverse"][metric_index][param_index]
            ae = report["jacobian_absolute_error"][metric_index][param_index]
            re = report["jacobian_relative_error"][metric_index][param_index]
            print(
                f"    {name}: fwd={float(fwd):.6e} rev={float(rev):.6e} "
                f"abs_err={float(ae):.6e} rel_err={float(re):.6e}"
            )


def _tree_all_finite(tree) -> bool:
    for leaf in jax.tree_util.tree_leaves(tree):
        arr = np.asarray(jax.device_get(leaf))
        if np.issubdtype(arr.dtype, np.inexact) and not np.all(np.isfinite(arr)):
            return False
    return True


def _forward_objective_stage_debug(
    parameter_value,
    *,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    parameter_name: str,
) -> dict[str, Any]:
    parameter_values = jnp.asarray([parameter_value], dtype=jnp.float64)
    (
        execution_context,
        prepared_rollout,
        initial_carry,
        max_total_steps,
        stop_after_accepted_steps,
        _solver,
        _solve_vector_field,
    ) = _forward_benchmark_prepare_realized_schedule_profile_vector_rollout(
        parameter_values,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_names=(parameter_name,),
    )

    def _final_y_from_param_vector(pvec):
        (
            exec_ctx,
            prepared_rollout_local,
            initial_carry_local,
            max_steps_local,
            stop_after_local,
            _solver_local,
            _svf_local,
        ) = _forward_benchmark_prepare_realized_schedule_profile_vector_rollout(
            pvec,
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_names=(parameter_name,),
        )
        return _radau_forward_adaptive_final_y_realized_schedule(
            exec_ctx,
            max_steps_local,
            stop_after_local,
            initial_carry_local,
        )

    baseline_vec = parameter_values
    basis = jnp.asarray([1.0], dtype=jnp.float64)
    final_y, final_y_dot = jax.jvp(_final_y_from_param_vector, (baseline_vec,), (basis,))
    final_state = prepared_rollout.physics_context.unpack_flat(final_y)
    final_state_dot = jax.jvp(
        prepared_rollout.physics_context.unpack_flat,
        (final_y,),
        (final_y_dot,),
    )[1]
    objective_primal, objective_tangent = jax.jvp(
        lambda flat_y: _objective_vector(prepared_rollout.physics_context.unpack_flat(flat_y), runtime),
        (final_y,),
        (final_y_dot,),
    )
    return {
        "final_y_all_finite": _tree_all_finite(final_y),
        "final_y_dot_all_finite": _tree_all_finite(final_y_dot),
        "final_state_all_finite": _tree_all_finite(final_state),
        "final_state_dot_all_finite": _tree_all_finite(final_state_dot),
        "objective_primal_all_finite": bool(np.all(np.isfinite(np.asarray(jax.device_get(objective_primal), dtype=float)))),
        "objective_tangent_all_finite": bool(np.all(np.isfinite(np.asarray(jax.device_get(objective_tangent), dtype=float)))),
        "objective_primal": np.asarray(jax.device_get(objective_primal), dtype=float).tolist(),
        "objective_tangent": np.asarray(jax.device_get(objective_tangent), dtype=float).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare multi-parameter forward custom-JVP columns against a reverse custom-VJP Jacobian."
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="Benchmark TOML.")
    parser.add_argument("--device", type=str, default=None, help="Optional device override passed to config preparation.")
    parser.add_argument(
        "--ntx-exact-derivative-mode",
        default="direct",
        choices=("direct", "custom_vjp"),
        help="NTX exact-runtime derivative mode. Use direct for this benchmark.",
    )
    parser.add_argument(
        "--ad-mode",
        default="both",
        choices=("both", "forward", "reverse"),
        help="Run forward columns, reverse rows, or both.",
    )
    parser.add_argument(
        "--reverse-replay-device",
        default="default",
        choices=("cpu", "gpu", "default", "auto"),
        help="Device used by the accepted-step reverse replay in custom-VJP mode. Default: default.",
    )
    parser.add_argument(
        "--parameters",
        default=",".join(PROFILE_VECTOR_PARAMETERS),
        help="Comma-separated subset of profile-vector parameters to include.",
    )
    parser.add_argument(
        "--objective-indices",
        default=None,
        help="Optional comma-separated subset of objective row indices to run in reverse mode.",
    )
    parser.add_argument(
        "--forward-nan-debug",
        action="store_true",
        help="If the forward Jacobian column is nonfinite, run accepted-step NaN localization for single-parameter runs.",
    )
    parser.add_argument(
        "--forward-stage-debug",
        action="store_true",
        help="If the forward Jacobian column is nonfinite, report whether nonfinites appear in final_y, unpacked state, or objective mapping.",
    )
    args = parser.parse_args()

    if args.reverse_replay_device != "default":
        os.environ["NEOPAX_TRANSPORT_REVERSE_REPLAY_DEVICE"] = str(args.reverse_replay_device)

    config = _prepare_benchmark_config(
        Path(args.config),
        device=args.device,
        ntx_exact_derivative_mode=args.ntx_exact_derivative_mode,
    )
    runtime, baseline_state = build_runtime_context(config)
    profile_cfg = _baseline_profile_cfg(config)
    parameter_names = tuple(
        name.strip() for name in str(args.parameters).split(",") if name.strip()
    )
    if not parameter_names:
        raise ValueError("At least one parameter must be selected via --parameters.")
    unknown_parameters = [name for name in parameter_names if name not in PROFILE_VECTOR_PARAMETERS]
    if unknown_parameters:
        raise ValueError(f"Unknown parameters requested: {unknown_parameters}")
    baseline_vector = jnp.asarray([float(profile_cfg[name]) for name in parameter_names], dtype=jnp.float64)
    n_params = len(parameter_names)
    objective_indices = (
        tuple(range(len(OBJECTIVE_LABELS)))
        if args.objective_indices is None
        else tuple(int(item.strip()) for item in str(args.objective_indices).split(",") if item.strip())
    )
    if not objective_indices:
        raise ValueError("At least one objective must be selected via --objective-indices.")
    invalid_objective_indices = [idx for idx in objective_indices if idx < 0 or idx >= len(OBJECTIVE_LABELS)]
    if invalid_objective_indices:
        raise ValueError(f"Objective indices out of range: {invalid_objective_indices}")

    objective_fn_jvp = lambda p: _adaptive_rollout_objectives_realized_schedule_only_for_parameter_vector(  # noqa: E731
        p,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_names=parameter_names,
        derivative_mode="jvp",
    )
    objective_fn_vjp = lambda p: _adaptive_rollout_objectives_realized_schedule_only_for_parameter_vector(  # noqa: E731
        p,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_names=parameter_names,
        derivative_mode="vjp",
    )
    jac_fwd = None
    jac_rev = None
    forward_nan_debug = {}
    forward_stage_debug = {}
    if args.ad_mode in ("both", "forward"):
        print("[autodiff-gate] progress: running forward custom-JVP columns", flush=True)
        fwd_columns = []
        for idx in range(n_params):
            basis = np.zeros(n_params, dtype=float)
            basis[idx] = 1.0
            _, tangent = jax.jvp(
                objective_fn_jvp,
                (baseline_vector,),
                (jnp.asarray(basis, dtype=jnp.float64),),
            )
            tangent_arr = np.asarray(jax.device_get(tangent), dtype=float)
            tangent_arr = tangent_arr[np.asarray(objective_indices, dtype=int)]
            if (
                args.forward_nan_debug
                and not np.all(np.isfinite(tangent_arr))
                and n_params == 1
            ):
                parameter_name = parameter_names[idx]
                print(
                    "[autodiff-gate] forward-nan-debug progress: running accepted-step NaN localization "
                    f"for parameter={parameter_name}",
                    flush=True,
                )
                forward_nan_debug[parameter_name] = _adaptive_rollout_nan_debug_for_parameter(
                    baseline_vector[idx],
                    config=config,
                    runtime=runtime,
                    baseline_state=baseline_state,
                    profile_cfg=profile_cfg,
                    parameter_name=parameter_name,
                    debug_mode="minimal",
                    include_one_step_compare=False,
                )
            if (
                args.forward_stage_debug
                and not np.all(np.isfinite(tangent_arr))
                and n_params == 1
            ):
                parameter_name = parameter_names[idx]
                print(
                    "[autodiff-gate] forward-stage-debug progress: checking final_y/state/objective finiteness "
                    f"for parameter={parameter_name}",
                    flush=True,
                )
                forward_stage_debug[parameter_name] = _forward_objective_stage_debug(
                    baseline_vector[idx],
                    config=config,
                    runtime=runtime,
                    baseline_state=baseline_state,
                    profile_cfg=profile_cfg,
                    parameter_name=parameter_name,
                )
            fwd_columns.append(tangent_arr)
        jac_fwd = np.stack(fwd_columns, axis=1)
    if args.ad_mode in ("both", "reverse"):
        print("[autodiff-gate] progress: running reverse custom-VJP Jacobian", flush=True)
        jac_rev = np.asarray(jax.device_get(jax.jacrev(objective_fn_vjp)(baseline_vector)), dtype=float)
        jac_rev = jac_rev[np.asarray(objective_indices, dtype=int), :]

    if jac_fwd is not None and jac_rev is not None:
        abs_err = np.abs(jac_fwd - jac_rev)
        rel_err = abs_err / np.maximum(np.abs(jac_fwd), 1.0e-10)
        param_max_rel = np.max(rel_err, axis=0)
        max_rel_error = float(np.max(rel_err))
        passed = bool(np.all(np.isfinite(rel_err)) and max_rel_error <= 5.0e-2)
    elif jac_fwd is not None:
        abs_err = None
        rel_err = None
        param_max_rel = np.max(np.abs(jac_fwd), axis=0)
        max_rel_error = float("nan")
        passed = bool(np.all(np.isfinite(jac_fwd)))
    elif jac_rev is not None:
        abs_err = None
        rel_err = None
        param_max_rel = np.max(np.abs(jac_rev), axis=0)
        max_rel_error = float("nan")
        passed = bool(np.all(np.isfinite(jac_rev)))
    else:
        raise ValueError("No AD mode selected.")

    report = {
        "profile_vector_ad_compare": True,
        "config_path": str(Path(args.config)),
        "benchmark_version": BENCHMARK_VERSION_TAG,
        "parameter_names": list(parameter_names),
        "baseline_values": np.asarray(jax.device_get(baseline_vector), dtype=float).tolist(),
        "objective_labels": [OBJECTIVE_LABELS[idx] for idx in objective_indices],
        "objective_indices": list(objective_indices),
        "jacobian_forward": None if jac_fwd is None else jac_fwd.tolist(),
        "jacobian_reverse": None if jac_rev is None else jac_rev.tolist(),
        "jacobian_absolute_error": None if abs_err is None else abs_err.tolist(),
        "jacobian_relative_error": None if rel_err is None else rel_err.tolist(),
        "parameter_max_relative_error": param_max_rel.tolist(),
        "max_relative_error": max_rel_error,
        "ntx_exact_derivative_mode": str(args.ntx_exact_derivative_mode),
        "ad_mode": str(args.ad_mode),
        "reverse_replay_device": str(args.reverse_replay_device),
        "forward_nan_debug": forward_nan_debug or None,
        "forward_stage_debug": forward_stage_debug or None,
        "passed": passed,
    }

    if jac_fwd is not None and jac_rev is not None:
        _print_summary(report)
    elif jac_rev is not None:
        print("[autodiff-gate] mode=profile_vector_ad_compare reverse-only", flush=True)
        print("[autodiff-gate] reverse Jacobian rows:")
        for metric_index, label in enumerate(report["objective_labels"]):
            print(f"  - {label}:")
            for param_index, name in enumerate(report["parameter_names"]):
                rev = report["jacobian_reverse"][metric_index][param_index]
                print(f"    {name}: rev={float(rev):.6e}")
    elif jac_fwd is not None:
        print("[autodiff-gate] mode=profile_vector_ad_compare forward-only", flush=True)
        print(f"[autodiff-gate] benchmark_version={BENCHMARK_VERSION_TAG}", flush=True)
        print("[autodiff-gate] forward Jacobian columns:")
        for metric_index, label in enumerate(report["objective_labels"]):
            print(f"  - {label}:")
            for param_index, name in enumerate(report["parameter_names"]):
                fwd = report["jacobian_forward"][metric_index][param_index]
                print(f"    {name}: fwd={float(fwd):.6e}")
        if report.get("forward_nan_debug"):
            for name, debug in report["forward_nan_debug"].items():
                print(
                    "[autodiff-gate] forward-nan-debug "
                    f"parameter={name} first_bad_index={debug.get('first_bad_index')} "
                    f"first_bad_was_accepted={debug.get('first_bad_was_accepted')} "
                    f"first_bad_dt={debug.get('first_bad_dt')} "
                    f"final_tangent_finite={debug.get('final_tangent_finite')}",
                    flush=True,
                )
        if report.get("forward_stage_debug"):
            for name, debug in report["forward_stage_debug"].items():
                print(
                    "[autodiff-gate] forward-stage-debug "
                    f"parameter={name} final_y_all_finite={debug.get('final_y_all_finite')} "
                    f"final_y_dot_all_finite={debug.get('final_y_dot_all_finite')} "
                    f"final_state_all_finite={debug.get('final_state_all_finite')} "
                    f"final_state_dot_all_finite={debug.get('final_state_dot_all_finite')} "
                    f"objective_primal_all_finite={debug.get('objective_primal_all_finite')} "
                    f"objective_tangent_all_finite={debug.get('objective_tangent_all_finite')}",
                    flush=True,
                )
    outpath = _report_path()
    outpath.write_text(json.dumps(report, indent=2))
    print(f"Wrote {outpath.relative_to(ROOT)}")
    print(f"passed={report['passed']} max_rel_error={report['max_relative_error']}")


if __name__ == "__main__":
    main()
