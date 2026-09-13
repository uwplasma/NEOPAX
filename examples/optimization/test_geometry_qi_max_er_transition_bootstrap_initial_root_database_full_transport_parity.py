#!/usr/bin/env python
"""Parity of benchmark and optimization database full-transport entry points.

Both lanes use the same benchmark TOML, the validated root-only optimization
seed, selected initial-Er root, 16 accepted Radau steps, and four four-step reverse
segments.  The reference is the unchanged benchmark composition.  The trial
is the optimization-only full-transport selector and must invoke that same
integrated selected-root/transport composition exactly once.  Candidate JIT
boundaries are added only after this no-duplication baseline passes.  This is
not an FD test or a physical final-time transport run.
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
    import optimize_geometry_qi_max_er_transition_bootstrap_initial_root as base

    return tuple(term for term in base.terms if float(term[2]) != 0.0)


def build_problem(*, reverse_stage_mode: str):
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
    import jax

    with redirect_stdout(io.StringIO()):
        result = problem.evaluate(x)
    return jax.block_until_ready((result.residuals, result.jacobian))


def _worker(stage_name: str, output_path: Path) -> int:
    """Evaluate one lane in its own process and persist only host arrays."""

    import jax

    stage_mode = REFERENCE_STAGE_MODE if stage_name == "reference" else TRIAL_STAGE_MODE
    problem = build_problem(reverse_stage_mode=stage_mode)
    x0 = np.asarray(jax.device_get(problem.x0), dtype=float)
    residuals, jacobian = evaluate(problem, x0)
    np.savez(
        output_path,
        residuals=np.asarray(jax.device_get(residuals), dtype=float),
        jacobian=np.asarray(jax.device_get(jacobian), dtype=float),
        x0=x0,
        parameter_labels=np.asarray(problem.parameter_labels, dtype=str),
    )
    print(
        f"[database full-transport parity] worker={stage_name} "
        f"stage={stage_mode} wrote={output_path}",
        flush=True,
    )
    return 0


def _run_worker(stage_name: str, output_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-stage",
            stage_name,
            "--worker-output",
            str(output_path),
        ],
        check=True,
    )


def _load_worker_output(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    with np.load(path, allow_pickle=False) as data:
        return (
            np.asarray(data["residuals"], dtype=float),
            np.asarray(data["jacobian"], dtype=float),
            np.asarray(data["x0"], dtype=float),
            tuple(str(value) for value in data["parameter_labels"]),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-stage", choices=("reference", "trial"), help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_stage is not None:
        if args.worker_output is None:
            parser.error("--worker-output is required with --worker-stage")
        return _worker(args.worker_stage, args.worker_output)
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
        _run_worker("reference", reference_path)
        reference_residuals, reference_jacobian, reference_x0, reference_labels = (
            _load_worker_output(reference_path)
        )
        _run_worker("trial", trial_path)
        trial_residuals, trial_jacobian, trial_x0, trial_labels = _load_worker_output(
            trial_path
        )

    if reference_labels != trial_labels:
        raise AssertionError("Reference and trial parameter layouts differ.")
    np.testing.assert_array_equal(trial_x0, reference_x0)
    residual_delta = trial_residuals - reference_residuals
    jacobian_delta = trial_jacobian - reference_jacobian
    reference_jacobian_np = reference_jacobian
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
        f"segments={ACCEPTED_STEP_LIMIT // REVERSE_SEGMENT_LENGTH} "
        f"segment_length={REVERSE_SEGMENT_LENGTH} "
        f"reference_stage={REFERENCE_STAGE_MODE} trial_stage={TRIAL_STAGE_MODE} "
        "parameter_point=unperturbed_x0 "
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
