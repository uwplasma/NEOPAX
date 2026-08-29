#!/usr/bin/env python
"""Baseline repeat parity harness for the four-row evaluation.

No SciPy iteration is run.  The two problems use the same four-objective terms,
VMEC input, and initial scaled DoF vector.
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

import optimize_geometry_qi_max_er_transition_bootstrap_initial_root as example  # noqa: E402


def _terms_for_objective_set(objective_set: str):
    """Match the memory harness's row selection for an exact parity check."""

    selected = []
    for term in example.terms:
        objective = getattr(term[0], "objective", term[0])
        is_transport = objective.family == "transport"
        if objective_set == "transport_er_only":
            if is_transport and objective.name != "bootstrap_current_softmax_abs_scaled":
                selected.append(term)
        elif objective_set == "er_only":
            if not is_transport or objective.name != "bootstrap_current_softmax_abs_scaled":
                selected.append(term)
        elif objective_set == "all":
            selected.append(term)
        else:  # pragma: no cover - argparse validates the public choices.
            raise ValueError(f"Unsupported objective set {objective_set!r}.")
    return selected


def _build(mode: str = "off", objective_set: str = "all"):
    previous_mode = example.REVERSE_STAGE_MODE
    previous_terms = example.terms
    example.REVERSE_STAGE_MODE = mode
    example.terms = _terms_for_objective_set(objective_set)
    try:
        return example.build_transition_bootstrap_initial_root_problem(
            example.SEED_INPUT, int(example.MAX_MODE_SCHEDULE)
        )
    finally:
        example.REVERSE_STAGE_MODE = previous_mode
        example.terms = previous_terms


def _evaluate(problem, x):
    with redirect_stdout(io.StringIO()):
        result = problem.evaluate(x)
    return jax.block_until_ready((result.residuals, result.jacobian))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "optimization",
            "optimization_root_experiment",
            "optimization_root_strict_experiment",
            "optimization_root_per_radius_experiment",
            "optimization_payload_experiment",
            "optimization_payload_root_experiment",
            "optimization_payload_root_strict_experiment",
            "optimization_payload_root_scan_experiment",
            "optimization_payload_root_scan_geometry_experiment",
            "optimization_payload_reverse_experiment",
        ),
        default="optimization",
        help="Opt-in implementation to compare against the unchanged off benchmark.",
    )
    parser.add_argument(
        "--objective-set",
        choices=("all", "er_only", "transport_er_only"),
        default="all",
        help="Use the same row selection as the memory test.",
    )
    args = parser.parse_args()
    if not np.isscalar(example.MAX_MODE_SCHEDULE):
        raise ValueError("The parity test requires one fixed MAX_MODE_SCHEDULE value.")
    benchmark = _build(objective_set=args.objective_set)
    staged = _build(args.mode, objective_set=args.objective_set)
    x = np.asarray(jax.device_get(benchmark.x0), dtype=float)
    off_residuals, off_jacobian = _evaluate(benchmark, x)
    staged_residuals, staged_jacobian = _evaluate(staged, x)
    residual_delta = np.asarray(jax.device_get(staged_residuals - off_residuals), dtype=float)
    jacobian_delta = np.asarray(jax.device_get(staged_jacobian - off_jacobian), dtype=float)
    off_jacobian_np = np.asarray(jax.device_get(off_jacobian), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        jacobian_relative_delta = np.abs(jacobian_delta) / np.abs(off_jacobian_np)
    jacobian_relative_delta = np.where(
        np.abs(off_jacobian_np) == 0.0,
        np.where(np.abs(jacobian_delta) == 0.0, 0.0, np.inf),
        jacobian_relative_delta,
    )
    jacobian_relative_index = tuple(
        int(index) for index in np.unravel_index(np.argmax(jacobian_relative_delta), jacobian_relative_delta.shape)
    )
    print(f"[parity] mode={args.mode} objective_set={args.objective_set}", flush=True)
    print(f"[parity] residual_max_abs={np.max(np.abs(residual_delta)):.16e}", flush=True)
    print(f"[parity] jacobian_max_abs={np.max(np.abs(jacobian_delta)):.16e}", flush=True)
    print(
        "[parity] jacobian_max_relative="
        f"{np.max(jacobian_relative_delta):.16e} index={jacobian_relative_index}",
        flush=True,
    )
    if args.mode in {
        "optimization",
        "optimization_root_experiment",
        "optimization_payload_experiment",
        "optimization_payload_root_experiment",
        "optimization_payload_root_strict_experiment",
        "optimization_payload_root_scan_experiment",
        "optimization_payload_root_scan_geometry_experiment",
        "optimization_payload_reverse_experiment",
    }:
        # The accepted persistent-root stage has the same equations and
        # branch-selection rules as ``off`` but changes compiled evaluation
        # boundaries. The observed worst relative Jacobian difference is
        # 1.27e-7; keep a narrow explicit acceptance envelope for this opt-in
        # optimizer route. The unchanged ``off`` path remains the benchmark.
        residual_rtol, residual_atol = 1.0e-9, 1.0e-10
        jacobian_rtol, jacobian_atol = 2.0e-7, 2.0e-8
    else:
        residual_rtol, residual_atol = 1.0e-11, 1.0e-12
        jacobian_rtol, jacobian_atol = 1.0e-8, 1.0e-9
    np.testing.assert_allclose(
        staged_residuals, off_residuals, rtol=residual_rtol, atol=residual_atol
    )
    np.testing.assert_allclose(
        staged_jacobian, off_jacobian, rtol=jacobian_rtol, atol=jacobian_atol
    )
    print("[parity] PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
