#!/usr/bin/env python
"""Parity of benchmark and optimization database full-transport entry points.

Both lanes use the same benchmark TOML, the validated root-only optimization
seed, selected initial-Er root, 16 accepted Radau steps, and four four-step reverse
segments.  The reference is the unchanged benchmark composition.  The trial
is the optimization-only full-transport selector and must invoke that same
integrated selected-root/transport composition exactly once.  Candidate JIT
boundaries are added only after this no-duplication baseline passes.  This is
not an FD test or a physical final-time transport run.  Its compared rows
explicitly include maximum Er and net power in addition to the root-position
and bootstrap objectives.

By default both workers evaluate the unperturbed point.  With a nonzero
``--parameter-offset``, the optimization worker first evaluates ``x0`` and
then the perturbed point through the same persistent stages.  It also writes
that perturbed VMEX input.  The benchmark worker then builds a fresh problem
whose baseline is that input and evaluates its local ``x0``.  That second mode
detects trial data accidentally retained by an optimization-only JIT boundary
without asking the unchanged benchmark lane to update a live scan runtime.
"""

from __future__ import annotations

import argparse
import io
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SMALL_DATABASE_TRANSPORT_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box_small.toml"
)
ACCEPTED_STEP_LIMIT = 16
REVERSE_SEGMENT_LENGTH = 4
REFERENCE_STAGE_MODE = "benchmark"
TRIAL_STAGE_MODE = "database_full_transport_optimization"
DATABASE_N_THETA = 5
DATABASE_N_PHI = 25
DATABASE_N_XI = 31
ER_TRANSITION_LEFT_INDEX = 25
ER_TRANSITION_RIGHT_INDEX = 26
SOFTMAX_ER_WEIGHT = 0.5
NET_POWER_TARGET_MW = 300.0
NET_POWER_REFERENCE_VOLUME_M3 = 331.0187969899648
NET_POWER_TARGET_MW_M3 = NET_POWER_TARGET_MW / NET_POWER_REFERENCE_VOLUME_M3
NET_POWER_WEIGHT = 1.0
# Defaults of the current validated database full-transport benchmark lane.
# Keep these explicit so this parity test cannot silently fall back to the
# generic/Lij-oriented defaults of the public optimization API.
DATABASE_REVERSE_OPTIONS = {
    "reverse_stage_adjoint_solve_mode": "block",
    "reverse_rhs_transpose_mode": "explicit_database",
    "reverse_rhs_pullback_mode": "separate",
    "reverse_initial_cache_support_pullback_mode": "scalar",
    "reverse_rebuild_support_pullback_mode": "separate",
    "reverse_database_initial_support_mode": "reduced_zero",
    "reverse_database_initial_state_mode": "reduced_zero_rhs",
    "reverse_database_support_preparation_mode": "shared",
    "reverse_database_center_geometry_mode": "scalar_jvp",
    "reverse_database_stage_jacobian_mode": "independent",
    "reverse_database_support_objective_mode": "scalar",
    "reverse_database_segment_support_mode": "inline",
    "reverse_database_interpolation_transpose_mode": "legacy_sparse",
    "reverse_database_root_interpolation_transpose_mode": "legacy_sparse",
    "reverse_database_bootstrap_interpolation_transpose_mode": "legacy_sparse",
    "reverse_final_objective_cotangent_mode": "grouped_joint_vjp",
    "reverse_bootstrap_cotangent_mode": "joint_local_vjp_upar_only",
    "reverse_schedule_artifact_mode": "reuse_static_probe",
    "reverse_segment_start_replay_mode": "minimal",
    "reverse_segment_primal_record_mode": "reuse_segment_primal_record",
    "reverse_stage_cotangent_mode": "full",
    "reverse_step_bwd_mode": "reduced_cotangent_call_boundary",
    "reverse_stage_adjoint_memory_mode": "default",
}


def active_terms():
    from NEOPAX import optimization as opt
    import optimize_geometry_qi_max_er_transition_bootstrap_initial_root as base

    terms = [term for term in base.terms if float(term[2]) != 0.0]
    # The imported ambipolar-root script deliberately disables maximum Er and
    # does not define the full-transport net-power objective. Include both
    # explicitly so parity covers every terminal observable used by the two
    # standalone full-transport optimization scripts.
    geometry_term_count = sum(
        1
        for objective, _target, _weight in terms
        if (
            objective.objective.family
            if hasattr(objective, "objective")
            else objective.family
        )
        == "geometry"
    )
    terms.insert(
        geometry_term_count,
        (opt.transport.softmax_Er, base.MAX_ER_TARGET, SOFTMAX_ER_WEIGHT),
    )
    terms.append(
        (
            opt.transport.net_total_power_volume_average_mw_m3,
            NET_POWER_TARGET_MW_M3,
            NET_POWER_WEIGHT,
        )
    )
    return tuple(terms)


def build_problem(
    *,
    reverse_stage_mode: str,
    vmec_input=None,
    er_transition_left_index: int = ER_TRANSITION_LEFT_INDEX,
    er_transition_right_index: int = ER_TRANSITION_RIGHT_INDEX,
):
    from NEOPAX import optimization as opt
    import optimize_geometry_qi_max_er_transition_bootstrap_initial_root as base

    if not np.isscalar(base.MAX_MODE_SCHEDULE):
        raise ValueError("The reduced full-transport test requires one fixed max mode.")
    problem = opt.geometry_full_transport_least_squares_problem(
        SMALL_DATABASE_TRANSPORT_CONFIG,
        active_terms(),
        # Preserve the already validated selected-root optimization prefix.
        # Without this explicit override, the transport TOML selects the
        # separate 201-surface benchmark seed instead of the root lane's
        # 51-surface seed and can exhaust GPU memory before reaching the root.
        vmec_input=base.SEED_INPUT if vmec_input is None else vmec_input,
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
        er_transition_left_index=er_transition_left_index,
        er_transition_right_index=er_transition_right_index,
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
    if problem.options.get("Er_transition_left_index") != int(
        er_transition_left_index
    ) or problem.options.get("Er_transition_right_index") != int(
        er_transition_right_index
    ):
        raise AssertionError("Full-transport Er transition indices were not retained.")
    return problem


def evaluate(problem, x):
    import jax

    with redirect_stdout(io.StringIO()):
        result = problem.evaluate(x)
    return jax.block_until_ready((result.residuals, result.jacobian))


def _worker(
    stage_name: str,
    output_path: Path,
    *,
    parameter_index: int,
    parameter_offset: float,
    er_transition_left_index: int = ER_TRANSITION_LEFT_INDEX,
    er_transition_right_index: int = ER_TRANSITION_RIGHT_INDEX,
    vmec_input: Path | None = None,
    perturbed_input_output: Path | None = None,
) -> int:
    """Evaluate one lane in its own process and persist only host arrays."""

    import jax

    stage_mode = REFERENCE_STAGE_MODE if stage_name == "reference" else TRIAL_STAGE_MODE
    problem = build_problem(
        reverse_stage_mode=stage_mode,
        vmec_input=vmec_input,
        er_transition_left_index=er_transition_left_index,
        er_transition_right_index=er_transition_right_index,
    )
    x0 = np.array(jax.device_get(problem.x0), dtype=float, copy=True)
    if not 0 <= parameter_index < x0.size:
        raise ValueError(
            f"--parameter-index must be in [0, {x0.size}); got {parameter_index}."
        )
    fresh_perturbed_reference = stage_name == "reference" and vmec_input is not None
    evaluation_point = np.array(x0, copy=True)
    if not fresh_perturbed_reference:
        evaluation_point[parameter_index] += float(parameter_offset)
    primed_at_x0 = stage_name == "trial" and float(parameter_offset) != 0.0
    if primed_at_x0:
        if perturbed_input_output is None:
            raise ValueError(
                "The perturbed trial worker requires --worker-perturbed-input-output."
            )
        problem.input_from_scaled_parameters(evaluation_point).to_indata(
            perturbed_input_output
        )
        # Keep the optimization stage alive across two distinct geometries.
        # The fresh benchmark worker deliberately does not take this step.
        priming_residuals, priming_jacobian = evaluate(problem, x0)
        del priming_residuals, priming_jacobian
    residuals, jacobian = evaluate(problem, evaluation_point)
    np.savez(
        output_path,
        residuals=np.asarray(jax.device_get(residuals), dtype=float),
        jacobian=np.asarray(jax.device_get(jacobian), dtype=float),
        x0=x0,
        evaluation_point=evaluation_point,
        x_scale=np.asarray(jax.device_get(problem.x_scale), dtype=float),
        parameter_labels=np.asarray(problem.parameter_labels, dtype=str),
        objective_labels=np.asarray(
            [term.residual_label for term in problem.terms], dtype=str
        ),
    )
    print(
        f"[database full-transport parity] worker={stage_name} "
        f"stage={stage_mode} primed_at_x0={primed_at_x0} "
        f"fresh_perturbed_reference={fresh_perturbed_reference} wrote={output_path}",
        flush=True,
    )
    return 0


def _run_worker(
    stage_name: str,
    output_path: Path,
    *,
    parameter_index: int,
    parameter_offset: float,
    er_transition_left_index: int = ER_TRANSITION_LEFT_INDEX,
    er_transition_right_index: int = ER_TRANSITION_RIGHT_INDEX,
    vmec_input: Path | None = None,
    perturbed_input_output: Path | None = None,
) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-stage",
        stage_name,
        "--worker-output",
        str(output_path),
        "--parameter-index",
        str(parameter_index),
        "--parameter-offset",
        repr(float(parameter_offset)),
        "--er-transition-left-index",
        str(int(er_transition_left_index)),
        "--er-transition-right-index",
        str(int(er_transition_right_index)),
    ]
    if vmec_input is not None:
        command.extend(("--worker-vmec-input", str(vmec_input)))
    if perturbed_input_output is not None:
        command.extend(
            ("--worker-perturbed-input-output", str(perturbed_input_output))
        )
    subprocess.run(command, check=True)


def _load_worker_output(
    path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    tuple[str, ...],
]:
    with np.load(path, allow_pickle=False) as data:
        return (
            np.asarray(data["residuals"], dtype=float),
            np.asarray(data["jacobian"], dtype=float),
            np.asarray(data["x0"], dtype=float),
            np.asarray(data["evaluation_point"], dtype=float),
            np.asarray(data["x_scale"], dtype=float),
            tuple(str(value) for value in data["parameter_labels"]),
            tuple(str(value) for value in data["objective_labels"]),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker-stage",
        choices=("reference", "trial", "trial_fresh"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-vmec-input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-perturbed-input-output", type=Path, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--parameter-index",
        type=int,
        default=0,
        help="Scaled geometry parameter to perturb in persistent-stage parity mode.",
    )
    parser.add_argument(
        "--parameter-offset",
        type=float,
        default=0.0,
        help=(
            "Offset applied to the selected scaled parameter. A nonzero value "
            "primes the trial at x0 before comparing both lanes at the offset point."
        ),
    )
    parser.add_argument(
        "--diagnose-fresh-trial",
        action="store_true",
        help=(
            "Also evaluate a separate optimization worker initialized directly "
            "at the perturbed point. This isolates persistent-stage reuse from "
            "benchmark-versus-optimization numerical differences."
        ),
    )
    parser.add_argument(
        "--er-transition-left-index",
        type=int,
        default=ER_TRANSITION_LEFT_INDEX,
        help="Final-time Er radial-cell index for the left transition objective.",
    )
    parser.add_argument(
        "--er-transition-right-index",
        type=int,
        default=ER_TRANSITION_RIGHT_INDEX,
        help="Final-time Er radial-cell index for the right transition objective.",
    )
    args = parser.parse_args()
    for name in ("er_transition_left_index", "er_transition_right_index"):
        index = int(getattr(args, name))
        if not 0 <= index < 51:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 51).")
    if args.worker_stage is not None:
        if args.worker_output is None:
            parser.error("--worker-output is required with --worker-stage")
        return _worker(
            args.worker_stage,
            args.worker_output,
            parameter_index=args.parameter_index,
            parameter_offset=args.parameter_offset,
            er_transition_left_index=args.er_transition_left_index,
            er_transition_right_index=args.er_transition_right_index,
            vmec_input=args.worker_vmec_input,
            perturbed_input_output=args.worker_perturbed_input_output,
        )
    if args.worker_output is not None:
        parser.error("--worker-output is only valid with --worker-stage")

    # Keep this parent process free of JAX/NEOPAX imports.  Otherwise it can
    # retain a GPU client/allocation while a supposedly isolated worker runs.
    # Sequential workers release all reference-lane GPU state before the trial
    # starts while preserving exactly the host arrays needed for parity.
    with tempfile.TemporaryDirectory(prefix="neopax_full_transport_parity_") as temp_dir:
        temp_root = Path(temp_dir)
        reference_path = temp_root / "reference.npz"
        trial_path = temp_root / "trial.npz"
        fresh_trial_path = temp_root / "trial_fresh.npz"
        perturbed_input_path = temp_root / "input.full_transport_parity_perturbed"
        if args.parameter_offset != 0.0:
            # The trial must run first so it can materialize the exact VMEX
            # geometry used at x1. The reference process then treats that
            # geometry as its baseline, keeping the benchmark lane untouched.
            _run_worker(
                "trial",
                trial_path,
                parameter_index=args.parameter_index,
                parameter_offset=args.parameter_offset,
                er_transition_left_index=args.er_transition_left_index,
                er_transition_right_index=args.er_transition_right_index,
                perturbed_input_output=perturbed_input_path,
            )
            _run_worker(
                "reference",
                reference_path,
                parameter_index=args.parameter_index,
                parameter_offset=args.parameter_offset,
                er_transition_left_index=args.er_transition_left_index,
                er_transition_right_index=args.er_transition_right_index,
                vmec_input=perturbed_input_path,
            )
            if args.diagnose_fresh_trial:
                _run_worker(
                    "trial_fresh",
                    fresh_trial_path,
                    parameter_index=args.parameter_index,
                    parameter_offset=args.parameter_offset,
                    er_transition_left_index=args.er_transition_left_index,
                    er_transition_right_index=args.er_transition_right_index,
                )
        else:
            _run_worker(
                "reference",
                reference_path,
                parameter_index=args.parameter_index,
                parameter_offset=args.parameter_offset,
                er_transition_left_index=args.er_transition_left_index,
                er_transition_right_index=args.er_transition_right_index,
            )
            _run_worker(
                "trial",
                trial_path,
                parameter_index=args.parameter_index,
                parameter_offset=args.parameter_offset,
                er_transition_left_index=args.er_transition_left_index,
                er_transition_right_index=args.er_transition_right_index,
            )
        (
            reference_residuals,
            reference_jacobian,
            reference_x0,
            reference_point,
            reference_x_scale,
            reference_labels,
            reference_objective_labels,
        ) = _load_worker_output(reference_path)
        (
            trial_residuals,
            trial_jacobian,
            trial_x0,
            trial_point,
            trial_x_scale,
            trial_labels,
            trial_objective_labels,
        ) = _load_worker_output(trial_path)
        fresh_trial_output = (
            _load_worker_output(fresh_trial_path)
            if args.diagnose_fresh_trial and args.parameter_offset != 0.0
            else None
        )

    if args.diagnose_fresh_trial and args.parameter_offset == 0.0:
        print(
            "[database full-transport parity] fresh-trial diagnostic skipped: "
            "--parameter-offset is zero",
            flush=True,
        )

    if fresh_trial_output is not None:
        (
            fresh_trial_residuals,
            fresh_trial_jacobian,
            fresh_trial_x0,
            fresh_trial_point,
            fresh_trial_x_scale,
            fresh_trial_labels,
            fresh_trial_objective_labels,
        ) = fresh_trial_output
        if fresh_trial_labels != trial_labels:
            raise AssertionError("Fresh and reused trial parameter layouts differ.")
        if fresh_trial_objective_labels != trial_objective_labels:
            raise AssertionError("Fresh and reused trial objective layouts differ.")
        np.testing.assert_array_equal(fresh_trial_x0, trial_x0)
        np.testing.assert_array_equal(fresh_trial_point, trial_point)
        np.testing.assert_array_equal(fresh_trial_x_scale, trial_x_scale)
        reuse_residual_delta = trial_residuals - fresh_trial_residuals
        reuse_jacobian_delta = trial_jacobian - fresh_trial_jacobian
        print(
            "[database full-transport parity] fresh-trial reuse diagnostic "
            "comparison=reused_x0_to_x1_minus_fresh_direct_x1 "
            f"residual_max_abs={np.max(np.abs(reuse_residual_delta)):.16e} "
            f"jacobian_max_abs={np.max(np.abs(reuse_jacobian_delta)):.16e}",
            flush=True,
        )
        for row, label in enumerate(trial_objective_labels):
            print(
                "[database full-transport parity] fresh-trial row "
                f"objective={label} row={row} "
                f"residual_abs={abs(reuse_residual_delta[row]):.16e} "
                "jacobian_max_abs="
                f"{np.max(np.abs(reuse_jacobian_delta[row])):.16e}",
                flush=True,
            )

    if reference_labels != trial_labels:
        raise AssertionError("Reference and trial parameter layouts differ.")
    if reference_objective_labels != trial_objective_labels:
        raise AssertionError("Reference and trial objective layouts differ.")
    if args.parameter_offset == 0.0:
        np.testing.assert_array_equal(trial_x0, reference_x0)
        np.testing.assert_array_equal(trial_point, reference_point)
        reference_jacobian_aligned = reference_jacobian
    else:
        if np.any(reference_x_scale == 0.0) or np.any(trial_x_scale == 0.0):
            raise ValueError("Full-transport parity parameter scales must be nonzero.")
        # Each problem reports derivatives in its own scaled coordinates.
        # Convert the fresh-baseline benchmark Jacobian into the original
        # trial problem's scaled coordinates before comparing all columns.
        reference_jacobian_aligned = (
            reference_jacobian / reference_x_scale[None, :]
        ) * trial_x_scale[None, :]
    if fresh_trial_output is not None:
        fresh_reference_residual_delta = (
            fresh_trial_residuals - reference_residuals
        )
        fresh_reference_jacobian_delta = (
            fresh_trial_jacobian - reference_jacobian_aligned
        )
        print(
            "[database full-transport parity] fresh-trial benchmark diagnostic "
            "comparison=fresh_direct_x1_minus_fresh_benchmark_x1 "
            "residual_max_abs="
            f"{np.max(np.abs(fresh_reference_residual_delta)):.16e} "
            "jacobian_max_abs="
            f"{np.max(np.abs(fresh_reference_jacobian_delta)):.16e}",
            flush=True,
        )
        for row, label in enumerate(trial_objective_labels):
            print(
                "[database full-transport parity] fresh-trial benchmark row "
                f"objective={label} row={row} "
                f"residual_abs={abs(fresh_reference_residual_delta[row]):.16e} "
                "jacobian_max_abs="
                f"{np.max(np.abs(fresh_reference_jacobian_delta[row])):.16e}",
                flush=True,
            )
    residual_delta = trial_residuals - reference_residuals
    jacobian_delta = trial_jacobian - reference_jacobian_aligned
    reference_jacobian_np = reference_jacobian_aligned
    relative_delta = np.abs(jacobian_delta) / np.maximum(
        np.abs(reference_jacobian_np), 1.0e-14
    )
    relative_index = tuple(
        int(index)
        for index in np.unravel_index(np.argmax(relative_delta), relative_delta.shape)
    )
    absolute_index = tuple(
        int(index)
        for index in np.unravel_index(
            np.argmax(np.abs(jacobian_delta)), jacobian_delta.shape
        )
    )
    jacobian_rtol = 2.0e-7
    jacobian_atol = 2.0e-8
    jacobian_tolerance = (
        jacobian_atol + jacobian_rtol * np.abs(reference_jacobian_aligned)
    )
    violation_indices = np.argwhere(
        np.abs(jacobian_delta) > jacobian_tolerance
    )

    print(
        "[database full-transport parity] "
        f"grid=({DATABASE_N_THETA},{DATABASE_N_PHI},{DATABASE_N_XI}) "
        f"accepted_steps={ACCEPTED_STEP_LIMIT} initial_er_root=jax_selected_root "
        f"segments={ACCEPTED_STEP_LIMIT // REVERSE_SEGMENT_LENGTH} "
        f"segment_length={REVERSE_SEGMENT_LENGTH} "
        f"reference_stage={REFERENCE_STAGE_MODE} trial_stage={TRIAL_STAGE_MODE} "
        f"parameter_point={'unperturbed_x0' if args.parameter_offset == 0.0 else 'reused_stage_perturbed_x1'} "
        f"parameter_index={args.parameter_index} "
        f"parameter_offset={args.parameter_offset:.6e} "
        f"Er_transition_indices=({args.er_transition_left_index},"
        f"{args.er_transition_right_index}) "
        f"trial_primed_at_x0={args.parameter_offset != 0.0} "
        f"reference_geometry={'original_baseline' if args.parameter_offset == 0.0 else 'fresh_perturbed_baseline'} "
        "jacobian_coordinates=trial_seed_scaled "
        "process_isolation=sequential_workers "
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
        f"{np.max(np.abs(jacobian_delta)):.16e} index={absolute_index}",
        flush=True,
    )
    print(
        "[database full-transport parity] jacobian_max_relative="
        f"{np.max(relative_delta):.16e} index={relative_index}",
        flush=True,
    )
    print(
        "[database full-transport parity] jacobian_violation_count="
        f"{len(violation_indices)}",
        flush=True,
    )
    for row, column in violation_indices:
        print(
            "[database full-transport parity] jacobian_violation "
            f"objective={trial_objective_labels[int(row)]} row={int(row)} "
            f"parameter={trial_labels[int(column)]} column={int(column)} "
            f"trial={trial_jacobian[row, column]:.16e} "
            f"reference={reference_jacobian_aligned[row, column]:.16e} "
            f"abs={abs(jacobian_delta[row, column]):.16e} "
            f"relative={relative_delta[row, column]:.16e}",
            flush=True,
        )

    residual_rtol = 1.0e-9 if args.parameter_offset == 0.0 else 2.0e-7
    residual_atol = 1.0e-10 if args.parameter_offset == 0.0 else 2.0e-8
    np.testing.assert_allclose(
        trial_residuals,
        reference_residuals,
        rtol=residual_rtol,
        atol=residual_atol,
    )
    np.testing.assert_allclose(
        trial_jacobian,
        reference_jacobian_aligned,
        rtol=jacobian_rtol,
        atol=jacobian_atol,
    )
    print("[database full-transport parity] PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
