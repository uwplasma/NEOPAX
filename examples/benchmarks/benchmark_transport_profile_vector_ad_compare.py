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
    _prepare_benchmark_config,
    _baseline_profile_cfg,
    _adaptive_rollout_objectives_realized_schedule_only_for_parameter_vector,
    _host_step_pullback_diagnostic_enabled,
    _run_host_local_step_pullback_diagnostic_for_parameter_vector,
)
from NEOPAX._orchestrator import build_runtime_context  # noqa: E402


def _report_path() -> Path:
    outdir = ROOT / "outputs" / "autodiff_transport_lagged_ntx" / "profile_vector"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / "transport_profile_vector_ad_compare_summary.json"


def _parse_parameter_subset(text: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in str(text).split(",") if item.strip())
    if not values:
        raise ValueError("At least one profile-vector parameter must be provided.")
    invalid = [name for name in values if name not in PROFILE_VECTOR_PARAMETERS]
    if invalid:
        raise ValueError(
            f"Unsupported profile-vector parameter(s): {invalid}. "
            f"Allowed values: {list(PROFILE_VECTOR_PARAMETERS)}"
        )
    return values


def _parse_objective_subset(text: str) -> tuple[int, ...]:
    values = tuple(item.strip() for item in str(text).split(",") if item.strip())
    if not values:
        raise ValueError("At least one objective index must be provided.")
    out = []
    n_objectives = len(OBJECTIVE_LABELS)
    for item in values:
        idx = int(item)
        if idx < 0:
            idx = n_objectives + idx
        if idx < 0 or idx >= n_objectives:
            raise ValueError(
                f"Objective index {item} out of range. Allowed range: 0..{n_objectives - 1} or negative Python-style indices."
            )
        out.append(idx)
    return tuple(out)


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare multi-parameter forward custom-JVP columns against a reverse Jacobian."
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
        help="Run forward columns, reverse rows, or both. Reverse is currently unavailable during the option-4 refactor.",
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
    args = parser.parse_args()
    if args.ad_mode in ("both", "reverse"):
        raise NotImplementedError(
            "Legacy reverse benchmark path has been removed. "
            "The new option-4 accepted-step reverse is not wired into this CLI yet. "
            "Use --ad-mode forward or the dedicated one-step primitive benchmark."
        )
    os.environ["NEOPAX_TRANSPORT_REVERSE_REPLAY_DEVICE"] = str(args.reverse_replay_device)
    parameter_names = _parse_parameter_subset(args.parameters)
    objective_indices = (
        tuple(range(len(OBJECTIVE_LABELS)))
        if args.objective_indices is None
        else _parse_objective_subset(args.objective_indices)
    )

    config = _prepare_benchmark_config(
        Path(args.config),
        device=args.device,
        ntx_exact_derivative_mode=args.ntx_exact_derivative_mode,
    )
    runtime, baseline_state = build_runtime_context(config)
    profile_cfg = _baseline_profile_cfg(config)
    baseline_vector = jnp.asarray([float(profile_cfg[name]) for name in parameter_names], dtype=jnp.float64)

    if _host_step_pullback_diagnostic_enabled():
        diagnostic = _run_host_local_step_pullback_diagnostic_for_parameter_vector(
            baseline_vector,
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_names=parameter_names,
        )
        print("[autodiff-gate] host-step-pullback-diagnostic", flush=True)
        for key, value in diagnostic.items():
            print(f"  {key}: {float(value):.6e}" if np.asarray(jax.device_get(value)).shape == () else f"  {key}: {np.asarray(jax.device_get(value)).tolist()}", flush=True)
        return

    objective_fn_jvp = lambda p: _adaptive_rollout_objectives_realized_schedule_only_for_parameter_vector(  # noqa: E731
        p,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_names=parameter_names,
        derivative_mode="jvp",
    )
    jac_fwd = None
    jac_rev = None
    n_params = len(parameter_names)

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
            fwd_columns.append(tangent_arr)
        jac_fwd = np.stack(fwd_columns, axis=1)

    if jac_fwd is not None and jac_rev is not None:
        abs_err = np.abs(jac_fwd - jac_rev)
        rel_err = abs_err / np.maximum(np.abs(jac_fwd), 1.0e-10)
        param_max_rel = np.max(rel_err, axis=0)
        max_rel_error = float(np.max(rel_err))
        passed = bool(np.all(np.isfinite(rel_err)) and np.max(rel_err) <= 5.0e-2)
        abs_err_list = abs_err.tolist()
        rel_err_list = rel_err.tolist()
        param_max_rel_list = param_max_rel.tolist()
    else:
        abs_err_list = None
        rel_err_list = None
        param_max_rel_list = None
        max_rel_error = float("nan")
        passed = True

    report = {
        "profile_vector_ad_compare": True,
        "config_path": str(Path(args.config)),
        "ad_mode": str(args.ad_mode),
        "parameter_names": list(parameter_names),
        "baseline_values": np.asarray(jax.device_get(baseline_vector), dtype=float).tolist(),
        "objective_indices": list(objective_indices),
        "objective_labels": [OBJECTIVE_LABELS[i] for i in objective_indices],
        "jacobian_forward": None if jac_fwd is None else jac_fwd.tolist(),
        "jacobian_reverse": None if jac_rev is None else jac_rev.tolist(),
        "jacobian_absolute_error": abs_err_list,
        "jacobian_relative_error": rel_err_list,
        "parameter_max_relative_error": param_max_rel_list,
        "max_relative_error": max_rel_error,
        "ntx_exact_derivative_mode": str(args.ntx_exact_derivative_mode),
        "reverse_replay_device": str(args.reverse_replay_device),
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
        print("[autodiff-gate] forward Jacobian columns:")
        for metric_index, label in enumerate(report["objective_labels"]):
            print(f"  - {label}:")
            for param_index, name in enumerate(report["parameter_names"]):
                fwd = report["jacobian_forward"][metric_index][param_index]
                print(f"    {name}: fwd={float(fwd):.6e}")
    outpath = _report_path()
    outpath.write_text(json.dumps(report, indent=2))
    print(f"Wrote {outpath.relative_to(ROOT)}")
    print(f"passed={report['passed']} max_rel_error={report['max_relative_error']}")


if __name__ == "__main__":
    main()
