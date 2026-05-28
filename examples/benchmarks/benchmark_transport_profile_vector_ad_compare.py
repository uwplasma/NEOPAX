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
    DEFAULT_CONFIG,
    OBJECTIVE_LABELS,
    PROFILE_VECTOR_PARAMETERS,
    _adaptive_rollout_objectives_realized_schedule_only_for_parameter_vector,
    _baseline_profile_cfg,
    _prepare_benchmark_config,
)
from NEOPAX._orchestrator import build_runtime_context  # noqa: E402


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
    args = parser.parse_args()

    config = _prepare_benchmark_config(
        Path(args.config),
        device=args.device,
        ntx_exact_derivative_mode=args.ntx_exact_derivative_mode,
    )
    runtime, baseline_state = build_runtime_context(config)
    profile_cfg = _baseline_profile_cfg(config)
    baseline_vector = jnp.asarray([float(profile_cfg[name]) for name in PROFILE_VECTOR_PARAMETERS], dtype=jnp.float64)

    objective_fn_jvp = lambda p: _adaptive_rollout_objectives_realized_schedule_only_for_parameter_vector(  # noqa: E731
        p,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_names=PROFILE_VECTOR_PARAMETERS,
        derivative_mode="jvp",
    )
    objective_fn_vjp = lambda p: _adaptive_rollout_objectives_realized_schedule_only_for_parameter_vector(  # noqa: E731
        p,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_names=PROFILE_VECTOR_PARAMETERS,
        derivative_mode="vjp",
    )

    print("[autodiff-gate] progress: running forward custom-JVP columns", flush=True)
    n_params = len(PROFILE_VECTOR_PARAMETERS)
    fwd_columns = []
    for idx in range(n_params):
        basis = np.zeros(n_params, dtype=float)
        basis[idx] = 1.0
        _, tangent = jax.jvp(
            objective_fn_jvp,
            (baseline_vector,),
            (jnp.asarray(basis, dtype=jnp.float64),),
        )
        fwd_columns.append(np.asarray(jax.device_get(tangent), dtype=float))
    jac_fwd = np.stack(fwd_columns, axis=1)

    print("[autodiff-gate] progress: running reverse custom-VJP Jacobian", flush=True)
    jac_rev = np.asarray(jax.device_get(jax.jacrev(objective_fn_vjp)(baseline_vector)), dtype=float)

    abs_err = np.abs(jac_fwd - jac_rev)
    rel_err = abs_err / np.maximum(np.abs(jac_fwd), 1.0e-10)
    param_max_rel = np.max(rel_err, axis=0)

    report = {
        "profile_vector_ad_compare": True,
        "config_path": str(Path(args.config)),
        "parameter_names": list(PROFILE_VECTOR_PARAMETERS),
        "baseline_values": np.asarray(jax.device_get(baseline_vector), dtype=float).tolist(),
        "objective_labels": OBJECTIVE_LABELS,
        "jacobian_forward": jac_fwd.tolist(),
        "jacobian_reverse": jac_rev.tolist(),
        "jacobian_absolute_error": abs_err.tolist(),
        "jacobian_relative_error": rel_err.tolist(),
        "parameter_max_relative_error": param_max_rel.tolist(),
        "max_relative_error": float(np.max(rel_err)),
        "ntx_exact_derivative_mode": str(args.ntx_exact_derivative_mode),
        "passed": bool(np.all(np.isfinite(rel_err)) and np.max(rel_err) <= 5.0e-2),
    }

    _print_summary(report)
    outpath = _report_path()
    outpath.write_text(json.dumps(report, indent=2))
    print(f"Wrote {outpath.relative_to(ROOT)}")
    print(f"passed={report['passed']} max_rel_error={report['max_relative_error']}")


if __name__ == "__main__":
    main()
