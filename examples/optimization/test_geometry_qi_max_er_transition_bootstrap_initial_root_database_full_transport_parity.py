#!/usr/bin/env python
"""Parity of benchmark and optimization database full-transport lanes.

Both lanes use the same benchmark TOML, unperturbed VMEC input, selected
initial-Er root, four accepted Radau steps, and four one-step reverse
segments.  The reference is the unchanged benchmark composition.  The trial
links the accepted optimization-only root boundaries to those same transport
kernels.  This is not an FD test or a physical final-time transport run.
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

from NEOPAX import optimization as opt  # noqa: E402
import optimize_geometry_qi_max_er_transition_bootstrap_initial_root as base  # noqa: E402


SMALL_DATABASE_TRANSPORT_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box_small.toml"
)
ACCEPTED_STEP_LIMIT = 4
REVERSE_SEGMENT_LENGTH = 1
REFERENCE_STAGE_MODE = "benchmark"
TRIAL_STAGE_MODE = "database_root_fresh_payload_experiment"
DATABASE_N_THETA = 5
DATABASE_N_PHI = 25
DATABASE_N_XI = 31
# Defaults of the current validated database full-transport benchmark lane.
# Keep these explicit so this parity test cannot silently fall back to the
# generic/Lij-oriented defaults of the public optimization API.
DATABASE_REVERSE_OPTIONS = {
    "reverse_stage_adjoint_solve_mode": "block",
    "reverse_rhs_transpose_mode": "explicit_database",
    "reverse_rhs_pullback_mode": "separate",
    "reverse_initial_cache_support_pullback_mode": "scalar",
    "reverse_rebuild_support_pullback_mode": "separate",
    "reverse_database_initial_support_mode": "split",
    "reverse_database_support_preparation_mode": "shared",
    "reverse_database_center_geometry_mode": "scalar_jvp",
    "reverse_database_stage_jacobian_mode": "independent",
    "reverse_database_support_objective_mode": "scalar",
    "reverse_database_segment_support_mode": "inline",
    "reverse_database_interpolation_transpose_mode": "legacy_sparse",
    "reverse_final_objective_cotangent_mode": "grouped_vjp",
    "reverse_bootstrap_cotangent_mode": "joint_local_vjp_upar_only",
    "reverse_schedule_artifact_mode": "reuse_static_probe",
    "reverse_segment_start_replay_mode": "minimal",
    "reverse_segment_primal_record_mode": "reuse_segment_primal_record",
    "reverse_stage_cotangent_mode": "full",
    "reverse_step_bwd_mode": "reduced_cotangent_call_boundary",
    "reverse_stage_adjoint_memory_mode": "default",
}


def active_terms():
    return tuple(term for term in base.terms if float(term[2]) != 0.0)


def build_problem(*, reverse_stage_mode: str):
    if not np.isscalar(base.MAX_MODE_SCHEDULE):
        raise ValueError("The reduced full-transport test requires one fixed max mode.")
    problem = opt.geometry_full_transport_least_squares_problem(
        SMALL_DATABASE_TRANSPORT_CONFIG,
        active_terms(),
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
        reverse_segment_length=REVERSE_SEGMENT_LENGTH,
        max_reverse_accepted_steps=ACCEPTED_STEP_LIMIT,
        initial_er_root_ad="jax_selected_root",
        radau_jacobian_reuse_mode="legacy",
        reverse_stage_adjoint_solve_mode="block",
        reverse_rhs_transpose_mode="explicit_database",
        reverse_step_bwd_mode="reduced_cotangent_call_boundary",
        reverse_stage_mode=reverse_stage_mode,
    )
    # Fail before the expensive solve if either lane silently stops matching
    # the current validated database benchmark defaults.
    for name, expected in DATABASE_REVERSE_OPTIONS.items():
        actual = problem.options.get(name)
        if actual != expected:
            raise AssertionError(
                f"Database benchmark option {name!r} is {actual!r}; "
                f"expected {expected!r}."
            )
    return problem


def evaluate(problem, x):
    with redirect_stdout(io.StringIO()):
        result = problem.evaluate(x)
    return jax.block_until_ready((result.residuals, result.jacobian))


def main() -> int:
    reference = build_problem(reverse_stage_mode=REFERENCE_STAGE_MODE)
    trial = build_problem(reverse_stage_mode=TRIAL_STAGE_MODE)
    reference_x0 = np.asarray(jax.device_get(reference.x0), dtype=float)
    trial_x0 = np.asarray(jax.device_get(trial.x0), dtype=float)
    if reference.parameter_labels != trial.parameter_labels:
        raise AssertionError("Reference and trial parameter layouts differ.")
    np.testing.assert_array_equal(trial_x0, reference_x0)
    x = reference_x0

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
        f"segments={ACCEPTED_STEP_LIMIT} segment_length={REVERSE_SEGMENT_LENGTH} "
        f"reference_stage={REFERENCE_STAGE_MODE} trial_stage={TRIAL_STAGE_MODE} "
        "parameter_point=unperturbed_x0 "
        "transport_reverse=block/explicit_database/reduced_cotangent_call_boundary "
        "database_interpolation_transpose=legacy_sparse",
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
