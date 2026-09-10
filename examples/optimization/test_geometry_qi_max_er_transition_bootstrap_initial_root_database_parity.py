#!/usr/bin/env python
"""Parity check for the opt-in persistent database initial-root stage.

The reference is the unchanged ``database`` evaluator.  The trial mode only
adds a persistent JIT boundary around the fixed-table selected-root work; it
still rebuilds the live VMEC-derived database and performs the one recorded
scan transpose outside that boundary.
"""

from __future__ import annotations

import argparse
import io
from contextlib import redirect_stdout
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
SMALL_DATABASE_TRANSPORT_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box_small.toml"
)


def _terms_for_objective_set(objective_set: str):
    selected = []
    for term in base.terms:
        objective = getattr(term[0], "objective", term[0])
        is_transport = objective.family == "transport"
        is_bootstrap = (
            is_transport and objective.name == "bootstrap_current_softmax_abs_scaled"
        )
        if objective_set == "geometry_only" and is_transport:
            continue
        if objective_set == "transport_er_only" and (not is_transport or is_bootstrap):
            continue
        if objective_set == "bootstrap_only" and not is_bootstrap:
            continue
        selected.append(term)
    if not selected:
        raise RuntimeError(f"No terms selected for objective_set={objective_set!r}.")
    return selected


def _build(*, mode: str, transport_config: Path, objective_set: str):
    previous_config = base.TRANSPORT_CONFIG
    previous_mode = base.REVERSE_STAGE_MODE
    previous_terms = base.terms
    base.TRANSPORT_CONFIG = transport_config
    base.REVERSE_STAGE_MODE = mode
    base.terms = _terms_for_objective_set(objective_set)
    try:
        return base.build_transition_bootstrap_initial_root_problem(
            base.SEED_INPUT, int(base.MAX_MODE_SCHEDULE)
        )
    finally:
        base.TRANSPORT_CONFIG = previous_config
        base.REVERSE_STAGE_MODE = previous_mode
        base.terms = previous_terms


def _evaluate(problem, x):
    with redirect_stdout(io.StringIO()):
        result = problem.evaluate(x)
    return jax.block_until_ready((result.residuals, result.jacobian))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--objective-set",
        choices=("all", "geometry_only", "transport_er_only", "bootstrap_only"),
        default="all",
    )
    parser.add_argument(
        "--small-database",
        action="store_true",
        help="Use the reduced (5, 25, 31) database grid for a quicker check.",
    )
    parser.add_argument(
        "--parameter-index", type=int, default=0,
        help="Boundary-vector entry to perturb before both evaluations.",
    )
    parser.add_argument(
        "--parameter-offset", type=float, default=1.0e-3,
        help="Nonzero offset verifies both paths use the current VMEC-derived database.",
    )
    args = parser.parse_args()
    if not np.isscalar(base.MAX_MODE_SCHEDULE):
        raise ValueError("The database parity test requires one fixed MAX_MODE_SCHEDULE value.")

    config = SMALL_DATABASE_TRANSPORT_CONFIG if args.small_database else DATABASE_TRANSPORT_CONFIG
    benchmark = _build(mode="database", transport_config=config, objective_set=args.objective_set)
    trial = _build(
        mode="database_root_experiment", transport_config=config, objective_set=args.objective_set
    )
    x = np.asarray(jax.device_get(benchmark.x0), dtype=float)
    if not 0 <= args.parameter_index < x.size:
        raise ValueError(f"--parameter-index must be in [0, {x.size}); got {args.parameter_index}.")
    x[args.parameter_index] += args.parameter_offset

    reference_residuals, reference_jacobian = _evaluate(benchmark, x)
    trial_residuals, trial_jacobian = _evaluate(trial, x)
    residual_delta = np.asarray(jax.device_get(trial_residuals - reference_residuals), dtype=float)
    jacobian_delta = np.asarray(jax.device_get(trial_jacobian - reference_jacobian), dtype=float)
    reference_jacobian_np = np.asarray(jax.device_get(reference_jacobian), dtype=float)
    denominator = np.maximum(np.abs(reference_jacobian_np), 1.0e-14)
    relative_delta = np.abs(jacobian_delta) / denominator
    relative_index = tuple(
        int(index) for index in np.unravel_index(np.argmax(relative_delta), relative_delta.shape)
    )
    print(
        "[database parity] "
        f"objective_set={args.objective_set} small_database={args.small_database} "
        f"parameter_index={args.parameter_index} parameter_offset={args.parameter_offset:.6e}",
        flush=True,
    )
    print(f"[database parity] residual_max_abs={np.max(np.abs(residual_delta)):.16e}", flush=True)
    print(f"[database parity] jacobian_max_abs={np.max(np.abs(jacobian_delta)):.16e}", flush=True)
    print(
        "[database parity] jacobian_max_relative="
        f"{np.max(relative_delta):.16e} index={relative_index}",
        flush=True,
    )

    # Same equations, root selection, and one-scan fold as ``database``;
    # only the fixed-table root work is compiled.  Keep the established
    # persistent-stage numerical envelope explicit rather than comparing two
    # compilation schedules bit-for-bit.
    np.testing.assert_allclose(trial_residuals, reference_residuals, rtol=1.0e-9, atol=1.0e-10)
    np.testing.assert_allclose(trial_jacobian, reference_jacobian, rtol=2.0e-7, atol=2.0e-8)
    print("[database parity] PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
