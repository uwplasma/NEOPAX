#!/usr/bin/env python
"""Compare benchmark and optimization-only initial-root residual/Jacobian rows.

No SciPy iteration is run.  The two problems use the same four-objective terms,
VMEC input, and initial scaled DoF vector.
"""

from __future__ import annotations

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


def _build(mode: str):
    previous = example.REVERSE_STAGE_MODE
    try:
        example.REVERSE_STAGE_MODE = mode
        return example.build_transition_bootstrap_initial_root_problem(
            example.SEED_INPUT, int(example.MAX_MODE_SCHEDULE)
        )
    finally:
        example.REVERSE_STAGE_MODE = previous


def _evaluate(problem, x):
    with redirect_stdout(io.StringIO()):
        result = problem.evaluate(x)
    return jax.block_until_ready((result.residuals, result.jacobian))


def main() -> int:
    if not np.isscalar(example.MAX_MODE_SCHEDULE):
        raise ValueError("The parity test requires one fixed MAX_MODE_SCHEDULE value.")
    benchmark = _build("off")
    staged = _build("vmex_like")
    x = np.asarray(jax.device_get(benchmark.x0), dtype=float)
    off_residuals, off_jacobian = _evaluate(benchmark, x)
    staged_residuals, staged_jacobian = _evaluate(staged, x)
    residual_delta = np.asarray(jax.device_get(staged_residuals - off_residuals), dtype=float)
    jacobian_delta = np.asarray(jax.device_get(staged_jacobian - off_jacobian), dtype=float)
    print(f"[parity] residual_max_abs={np.max(np.abs(residual_delta)):.16e}", flush=True)
    print(f"[parity] jacobian_max_abs={np.max(np.abs(jacobian_delta)):.16e}", flush=True)
    np.testing.assert_allclose(staged_residuals, off_residuals, rtol=1.0e-11, atol=1.0e-12)
    # Separate evaluations can differ in floating-point reduction order on
    # tiny Jacobian entries; this remains far below reverse-AD significance.
    np.testing.assert_allclose(staged_jacobian, off_jacobian, rtol=1.0e-8, atol=1.0e-9)
    print("[parity] PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
