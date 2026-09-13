#!/usr/bin/env python
"""Four-step parity test for the database full-transport segment boundary.

Both evaluations use the selected initial-Er root and deliberately stop after
four accepted Radau steps.  The reference places those steps in one segment;
the trial uses four fixed one-step segments.  This is a boundary test, not a
physical final-time transport run.
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

from NEOPAX import optimization as opt  # noqa: E402
import optimize_geometry_qi_max_er_transition_bootstrap_initial_root as base  # noqa: E402


SMALL_DATABASE_TRANSPORT_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box_small.toml"
)
ACCEPTED_STEP_LIMIT = 4
REFERENCE_SEGMENT_LENGTH = 4
TRIAL_SEGMENT_LENGTH = 1
DATABASE_N_THETA = 5
DATABASE_N_PHI = 25
DATABASE_N_XI = 31


def active_terms():
    return tuple(term for term in base.terms if float(term[2]) != 0.0)


def build_problem(*, reverse_segment_length: int):
    if not np.isscalar(base.MAX_MODE_SCHEDULE):
        raise ValueError("The reduced full-transport test requires one fixed max mode.")
    return opt.geometry_full_transport_least_squares_problem(
        SMALL_DATABASE_TRANSPORT_CONFIG,
        active_terms(),
        vmec_input=base.SEED_INPUT,
        max_mode=int(base.MAX_MODE_SCHEDULE),
        families=base.GEOMETRY_FAMILIES,
        scale_mode=base.SCALE_MODE,
        ess_alpha=base.ESS_ALPHA,
        mboz=base.QI_MBOZ,
        nboz=base.QI_NBOZ,
        surfaces=tuple(float(value) for value in base.SURFACES),
        n_theta=DATABASE_N_THETA,
        n_zeta=DATABASE_N_PHI,
        n_xi=DATABASE_N_XI,
        geometry_solver_device=base.SOLVER_DEVICE,
        device=base.SOLVER_DEVICE,
        accepted_step_limit=ACCEPTED_STEP_LIMIT,
        reverse_segment_length=int(reverse_segment_length),
        max_reverse_accepted_steps=ACCEPTED_STEP_LIMIT,
        initial_er_root_ad="jax_selected_root",
        radau_jacobian_reuse_mode="legacy",
        reverse_stage_adjoint_solve_mode="bicgstab",
        reverse_rhs_transpose_mode="explicit_ntx_interpolated",
        reverse_step_bwd_mode="reduced_cotangent",
    )


def evaluate(problem, x):
    with redirect_stdout(io.StringIO()):
        result = problem.evaluate(x)
    return jax.block_until_ready((result.residuals, result.jacobian))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameter-index", type=int, default=0)
    parser.add_argument("--parameter-offset", type=float, default=1.0e-3)
    args = parser.parse_args()

    reference = build_problem(reverse_segment_length=REFERENCE_SEGMENT_LENGTH)
    trial = build_problem(reverse_segment_length=TRIAL_SEGMENT_LENGTH)
    x = np.array(jax.device_get(reference.x0), dtype=float, copy=True)
    if not 0 <= args.parameter_index < x.size:
        raise ValueError(
            f"--parameter-index must be in [0, {x.size}); got {args.parameter_index}."
        )
    x[args.parameter_index] += float(args.parameter_offset)

    reference_residuals, reference_jacobian = evaluate(reference, x)
    trial_residuals, trial_jacobian = evaluate(trial, x)
    residual_delta = np.asarray(
        jax.device_get(trial_residuals - reference_residuals), dtype=float
    )
    jacobian_delta = np.asarray(
        jax.device_get(trial_jacobian - reference_jacobian), dtype=float
    )
    reference_jacobian_np = np.asarray(jax.device_get(reference_jacobian), dtype=float)
    relative_delta = np.abs(jacobian_delta) / np.maximum(
        np.abs(reference_jacobian_np), 1.0e-14
    )
    relative_index = tuple(
        int(index)
        for index in np.unravel_index(np.argmax(relative_delta), relative_delta.shape)
    )

    print(
        "[database full-transport parity] "
        f"grid=({DATABASE_N_THETA},{DATABASE_N_PHI},{DATABASE_N_XI}) "
        f"accepted_steps={ACCEPTED_STEP_LIMIT} initial_er_root=jax_selected_root "
        f"reference_segment_length={REFERENCE_SEGMENT_LENGTH} "
        f"trial_segment_length={TRIAL_SEGMENT_LENGTH} "
        f"parameter_index={args.parameter_index} "
        f"parameter_offset={args.parameter_offset:.6e}",
        flush=True,
    )
    print(
        "[database full-transport parity] residual_max_abs="
        f"{np.max(np.abs(residual_delta)):.16e}",
        flush=True,
    )
    print(
        "[database full-transport parity] jacobian_max_abs="
        f"{np.max(np.abs(jacobian_delta)):.16e}",
        flush=True,
    )
    print(
        "[database full-transport parity] jacobian_max_relative="
        f"{np.max(relative_delta):.16e} index={relative_index}",
        flush=True,
    )

    np.testing.assert_allclose(
        trial_residuals, reference_residuals, rtol=1.0e-9, atol=1.0e-10
    )
    np.testing.assert_allclose(
        trial_jacobian, reference_jacobian, rtol=2.0e-7, atol=2.0e-8
    )
    print("[database full-transport parity] PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
