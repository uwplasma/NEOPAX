#!/usr/bin/env python
"""Directional value/Jacobian check for live-database initial-root AD.

The check is run at the baseline and at a nonzero VMEC boundary vector.  The
second point is essential: a root evaluator that accidentally retained the
baseline database can appear correct at ``x0`` while returning stale transport
values after an optimizer update.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import optimize_geometry_qi_max_er_transition_bootstrap_initial_root as base  # noqa: E402


DATABASE_TRANSPORT_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box.toml"
)


def _evaluate(problem, x):
    evaluation = problem.evaluate(x)
    return (
        np.asarray(jax.device_get(evaluation.residuals), dtype=float),
        np.asarray(jax.device_get(evaluation.jacobian), dtype=float),
        tuple(evaluation.result.residual_labels),
    )


def _check_point(problem, x, *, parameter_index: int, fd_step: float, label: str):
    residuals, jacobian, labels = _evaluate(problem, x)
    x_plus = np.array(x, copy=True)
    x_minus = np.array(x, copy=True)
    x_plus[parameter_index] += fd_step
    x_minus[parameter_index] -= fd_step
    residuals_plus, _, _ = _evaluate(problem, x_plus)
    residuals_minus, _, _ = _evaluate(problem, x_minus)
    finite_difference = (residuals_plus - residuals_minus) / (2.0 * fd_step)
    ad_column = jacobian[:, parameter_index]
    difference = ad_column - finite_difference
    denominator = np.maximum(np.maximum(np.abs(ad_column), np.abs(finite_difference)), 1.0e-12)
    relative = np.abs(difference) / denominator
    transport_rows = [
        row
        for row, name in enumerate(labels)
        if any(
            marker in str(name)
            for marker in ("softmax_Er", "Er_transition", "bootstrap_current")
        )
    ]
    if not transport_rows:
        raise RuntimeError("The database directional check did not find any transport residual rows.")
    transport_shift = (
        0.0
        if not transport_rows
        else float(np.max(np.abs(residuals_plus[transport_rows] - residuals_minus[transport_rows])))
    )
    print(
        f"[database directional] point={label} "
        f"residual_norm={np.linalg.norm(residuals):.6e} "
        f"fd_jacobian_max_abs={np.max(np.abs(difference)):.6e} "
        f"fd_jacobian_max_relative={np.max(relative):.6e} "
        f"transport_residual_two_sided_shift={transport_shift:.6e}",
        flush=True,
    )
    if transport_shift == 0.0:
        raise AssertionError(
            "Database transport residuals did not change under a boundary perturbation; "
            "the evaluator may be using a stale baseline database."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameter-index", type=int, default=0)
    parser.add_argument("--fd-step", type=float, default=1.0e-5)
    parser.add_argument("--base-offset", type=float, default=1.0e-3)
    args = parser.parse_args()
    if args.fd_step <= 0.0:
        raise ValueError("--fd-step must be positive.")
    if not np.isscalar(base.MAX_MODE_SCHEDULE):
        raise ValueError("This directional check requires one fixed MAX_MODE_SCHEDULE value.")

    base.TRANSPORT_CONFIG = DATABASE_TRANSPORT_CONFIG
    base.REVERSE_STAGE_MODE = "database"
    problem = base.build_transition_bootstrap_initial_root_problem(
        base.SEED_INPUT, int(base.MAX_MODE_SCHEDULE)
    )
    x0 = np.asarray(jax.device_get(problem.x0), dtype=float)
    if not 0 <= args.parameter_index < x0.size:
        raise ValueError(f"--parameter-index must be in [0, {x0.size}); got {args.parameter_index}.")
    _check_point(
        problem,
        x0,
        parameter_index=args.parameter_index,
        fd_step=args.fd_step,
        label="x0",
    )
    x_perturbed = np.array(x0, copy=True)
    x_perturbed[args.parameter_index] += args.base_offset
    _check_point(
        problem,
        x_perturbed,
        parameter_index=args.parameter_index,
        fd_step=args.fd_step,
        label="perturbed",
    )
    print("[database directional] complete: transport values respond at both points.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
