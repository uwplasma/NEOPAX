#!/usr/bin/env python
"""Audit one-shot geometry/profile block Gauss-Newton candidates.

This diagnostic does not run an optimization and does not modify either the
geometry-only or the existing combined optimization example.  It evaluates
the geometry-only and combined problems at the identical physical seed in
separate worker processes, checks that the combined problem embeds the exact
geometry residual/Jacobian, and then compares five linearized candidates:

1. the established geometry-only step;
2. the geometry-only step followed by a profile correction;
3. a profile-only first step;
4. the profile step followed by a geometry correction; and
5. the ordinary simultaneous geometry/profile step.

The seed comparison requires two expensive worker evaluations.  Optional
``--nonlinear-candidates`` are each evaluated in another isolated worker at
the actual candidate point.  Those evaluations distinguish an affine
Gauss--Newton prediction from a genuine nonlinear VMEC/NTX/transport result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import the existing example as data/building blocks only.  Its main function
# is never called and neither existing optimization path is edited here.
from examples.optimization import (  # noqa: E402
    optimize_geometry_profiles_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport
    as combined_example,
)


AUDIT_OUT_DIR = ROOT / "outputs" / "geometry_profiles_one_shot_block_step_audit"
PROFILE_LABELS = (
    "n0",
    "T0",
    "density_shape_power",
    "temperature_shape_power",
    "density_shape_alpha",
    "temperature_shape_alpha",
)
NONLINEAR_CANDIDATE_NAMES = (
    "geometry_only",
    "geometry_then_profile",
    "profile_first",
    "profile_then_geometry",
    "ordinary_direct_joint",
)


def parser() -> argparse.ArgumentParser:
    out = combined_example.parser()
    out.description = __doc__
    out.set_defaults(
        out_dir=AUDIT_OUT_DIR,
        profile_dofs=True,
        initial_plots=False,
    )
    default_max_mode = (
        int(combined_example.MAX_MODE_SCHEDULE)
        if np.isscalar(combined_example.MAX_MODE_SCHEDULE)
        else int(tuple(combined_example.MAX_MODE_SCHEDULE)[-1])
    )
    out.add_argument("--max-mode", type=int, default=default_max_mode)
    out.add_argument("--geometry-trust-radius", type=float, default=1.0)
    out.add_argument("--profile-trust-radius", type=float, default=1.0)
    out.add_argument("--parity-rtol", type=float, default=1.0e-6)
    out.add_argument("--parity-atol", type=float, default=1.0e-8)
    out.add_argument(
        "--reuse-linear-audit-npz",
        type=Path,
        help=(
            "Reuse residuals and geometry/profile Jacobian blocks from a "
            "previous audit NPZ instead of repeating the two seed "
            "evaluations. Candidate points are rebuilt with the current "
            "trust radii and bounds."
        ),
    )
    out.add_argument(
        "--nonlinear-candidates",
        nargs="+",
        choices=NONLINEAR_CANDIDATE_NAMES,
        default=(),
        help=(
            "Evaluate the selected linearized steps as actual nonlinear "
            "full-transport trials. Each candidate adds one expensive "
            "isolated problem evaluation."
        ),
    )
    out.add_argument(
        "--worker-kind",
        choices=("geometry", "combined", "nonlinear_candidate"),
        help=argparse.SUPPRESS,
    )
    out.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    out.add_argument("--candidate-input", type=Path, help=argparse.SUPPRESS)
    return out


def _validate_args(args: argparse.Namespace) -> None:
    if args.postprocess_existing:
        raise ValueError("--postprocess-existing is not valid for this audit.")
    if not args.profile_dofs:
        raise ValueError("The block-step audit requires --profile-dofs.")
    for name in (
        "database_n_theta",
        "database_n_phi",
        "database_n_xi",
        "max_mode",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    for name in ("geometry_trust_radius", "profile_trust_radius"):
        if float(getattr(args, name)) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.parity_rtol < 0.0 or args.parity_atol < 0.0:
        raise ValueError("Parity tolerances must be nonnegative.")
    if (
        args.reuse_linear_audit_npz is not None
        and not args.reuse_linear_audit_npz.is_file()
    ):
        raise FileNotFoundError(
            f"Saved audit NPZ not found: {args.reuse_linear_audit_npz}"
        )


def _problem_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "vmec_input": args.seed_input.resolve(),
        "max_mode": int(args.max_mode),
        "families": combined_example.GEOMETRY_FAMILIES,
        "scale_mode": combined_example.GEOMETRY_SCALE_MODE,
        "ess_alpha": combined_example.ESS_ALPHA,
        "mboz": combined_example.QI_MBOZ,
        "nboz": combined_example.QI_NBOZ,
        "surfaces": tuple(float(value) for value in combined_example.SURFACES),
        "n_theta": int(args.database_n_theta),
        "n_zeta": int(args.database_n_phi),
        "n_xi": int(args.database_n_xi),
        "geometry_max_iter": combined_example.GEOMETRY_MAX_ITER,
        "geometry_solver_device": combined_example.SOLVER_DEVICE,
        "device": combined_example.SOLVER_DEVICE,
        "accepted_step_limit": combined_example.FULL_TRANSPORT_ACCEPTED_STEP_LIMIT,
        "reverse_segment_length": combined_example.REVERSE_SEGMENT_LENGTH,
        "max_reverse_accepted_steps": (
            combined_example.MAX_REVERSE_ACCEPTED_STEPS
        ),
        "initial_er_root_ad": "jax_selected_root",
        "er_transition_left_index": int(args.er_transition_left_index),
        "er_transition_right_index": int(args.er_transition_right_index),
        "er_transition_rho_min": float(args.er_transition_rho_min),
        "er_transition_rho_max": float(args.er_transition_rho_max),
        "er_transition_rho_target": float(args.er_transition_rho_target),
        "er_transition_temperature_kv_m": float(
            args.er_transition_temperature_kv_m
        ),
        "er_transition_rho_softness": float(args.er_transition_rho_softness),
        "er_transition_softmax_beta": float(args.er_transition_softmax_beta),
        "radau_jacobian_reuse_mode": "legacy",
        "reverse_stage_adjoint_solve_mode": "block",
        "reverse_rhs_transpose_mode": "explicit_database",
        "reverse_stage_cotangent_mode": "full",
        "reverse_step_bwd_mode": "reduced_cotangent_call_boundary",
        "reverse_stage_adjoint_memory_mode": "default",
        "print_final_softmax_er": combined_example.PRINT_FINAL_SOFTMAX_ER,
        "reverse_stage_mode": combined_example.REVERSE_STAGE_MODE,
        "qi_maxj_settings": combined_example.qi_maxj_backend_settings(
            combined_example.PHYSICAL_J_PITCHES
        ),
    }


def _build_problem(args: argparse.Namespace, *, include_profiles: bool):
    config = combined_example.transport_config(args)
    terms = combined_example.active_terms(args)
    kwargs = _problem_kwargs(args)
    kwargs.update(combined_example.optional_profile_problem_kwargs(include_profiles))
    return combined_example.opt.geometry_full_transport_least_squares_problem(
        config,
        terms,
        **kwargs,
    )


def _worker(args: argparse.Namespace) -> int:
    import jax

    include_profiles = args.worker_kind in {"combined", "nonlinear_candidate"}
    problem = _build_problem(args, include_profiles=include_profiles)
    x0 = np.asarray(jax.device_get(problem.x0), dtype=float)
    labels = np.asarray(tuple(problem.parameter_labels), dtype=np.str_)
    candidate_name = "seed"
    values = x0
    if args.worker_kind == "nonlinear_candidate":
        if args.candidate_input is None:
            raise ValueError(
                "--candidate-input is required for a nonlinear candidate worker."
            )
        candidate = _load(args.candidate_input)
        candidate_name = str(np.asarray(candidate["candidate_name"]).item())
        expected_labels = np.asarray(candidate["parameter_labels"], dtype=np.str_)
        if not np.array_equal(labels, expected_labels):
            raise AssertionError(
                "Nonlinear candidate parameter labels do not match the "
                "freshly built combined problem."
            )
        expected_x0 = np.asarray(candidate["x0"], dtype=float)
        if not np.array_equal(x0, expected_x0):
            maximum, relative = _max_error(expected_x0, x0)
            raise AssertionError(
                "Nonlinear candidate seed coordinates do not match the "
                "freshly built combined problem: "
                f"max_abs={maximum:.16e}, max_relative={relative:.16e}."
            )
        values = np.asarray(candidate["candidate_x"], dtype=float)

    evaluation = problem.evaluate(values)
    residuals = np.asarray(jax.device_get(evaluation.residuals), dtype=float)
    jacobian = np.asarray(jax.device_get(evaluation.jacobian), dtype=float)
    scales = np.asarray(jax.device_get(problem.x_scale), dtype=float)
    lower = np.full_like(x0, -np.inf)
    upper = np.full_like(x0, np.inf)
    if include_profiles:
        lower, upper = combined_example.scaled_bounds(problem)
    args.worker_output.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "residuals": residuals,
        "jacobian": jacobian,
        "parameter_labels": labels,
        "x0": x0,
        "evaluated_x": values,
        "coordinate_scales": scales,
        "lower_bounds": np.asarray(lower, dtype=float),
        "upper_bounds": np.asarray(upper, dtype=float),
        "cost": np.asarray(_cost(residuals), dtype=float),
        "elapsed_s": np.asarray(float(evaluation.elapsed_s), dtype=float),
        "finite": np.asarray(bool(np.all(np.isfinite(residuals))), dtype=bool),
    }
    if include_profiles:
        physical_profiles = combined_example.profile_parameter_values(
            problem, values
        )
        output["physical_profile_labels"] = np.asarray(
            tuple(physical_profiles), dtype=np.str_
        )
        output["physical_profile_values"] = np.asarray(
            tuple(physical_profiles.values()), dtype=float
        )
    np.savez(args.worker_output, **output)
    print(
        "[block-step-audit] "
        f"worker={args.worker_kind} candidate={candidate_name} "
        f"cost={_cost(residuals):.16e} residuals={residuals.shape} "
        f"jacobian={jacobian.shape} wrote={args.worker_output}",
        flush=True,
    )
    if args.worker_kind == "nonlinear_candidate":
        print(
            "[block-step-audit] nonlinear candidate physical_profiles="
            f"{physical_profiles}",
            flush=True,
        )
        print(
            "[block-step-audit] nonlinear candidate objectives "
            + combined_example.geometry_example.iteration_diagnostics(
                evaluation
            ),
            flush=True,
        )
    return 0


def _run_worker(
    kind: str,
    output: Path,
    *,
    candidate_input: Path | None = None,
) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        *sys.argv[1:],
        "--worker-kind",
        kind,
        "--worker-output",
        str(output),
    ]
    if candidate_input is not None:
        command.extend(("--candidate-input", str(candidate_input)))
    subprocess.run(command, check=True)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def _combined_coordinate_metadata(
    args: argparse.Namespace,
    labels: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild combined seed coordinates/bounds without an AD evaluation."""

    config = combined_example.transport_config(args)
    profile_config = config.get("profiles", {})
    defaults = {
        "n0": 4.21,
        "T0": 17.8,
        "density_shape_power": 2.0,
        "temperature_shape_power": 2.0,
        "density_shape_alpha": 1.0,
        "temperature_shape_alpha": 1.0,
    }
    baseline = {}
    for name in PROFILE_LABELS:
        value = profile_config.get(name, defaults[name])
        if isinstance(value, (list, tuple)):
            value = value[0]
        baseline[name] = float(value)

    scale_mode = str(combined_example.PROFILE_SCALE_MODE).strip().lower()
    coordinate_mode = str(combined_example.PROFILE_COORDINATE_MODE).strip().lower()
    if scale_mode not in {"identity", "none", "unit", "nominal", "baseline"}:
        raise ValueError(f"Unsupported profile scale mode {scale_mode!r}.")
    if coordinate_mode not in {"absolute", "delta"}:
        raise ValueError(f"Unsupported profile coordinate mode {coordinate_mode!r}.")

    x0 = np.zeros((len(labels),), dtype=float)
    scales = np.full((len(labels),), np.nan, dtype=float)
    lower = np.full((len(labels),), -np.inf, dtype=float)
    upper = np.full((len(labels),), np.inf, dtype=float)
    for index, label in enumerate(labels):
        if label not in PROFILE_LABELS:
            continue
        physical_seed = baseline[label]
        scale = (
            max(abs(physical_seed), 1.0e-12)
            if scale_mode in {"nominal", "baseline"}
            else 1.0
        )
        offset = physical_seed if coordinate_mode == "delta" else 0.0
        x0[index] = (
            0.0 if coordinate_mode == "delta" else physical_seed / scale
        )
        scales[index] = scale
        lower[index] = (
            combined_example.PROFILE_PHYSICAL_LOWER.get(label, -np.inf)
            - offset
        ) / scale
        upper[index] = (
            combined_example.PROFILE_PHYSICAL_UPPER.get(label, np.inf)
            - offset
        ) / scale
    return x0, scales, lower, upper


def _seed_data_from_saved_linear_audit(
    args: argparse.Namespace,
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Adapt either the original or extended audit NPZ to seed worker data."""

    saved = _load(path.resolve())
    required = {
        "residuals",
        "jacobian_geometry",
        "jacobian_profile",
        "combined_parameter_labels",
    }
    missing = sorted(required.difference(saved))
    if missing:
        raise ValueError(
            f"Saved audit NPZ {path} is missing arrays: {missing}."
        )
    labels = tuple(str(value) for value in saved["combined_parameter_labels"])
    profile_indices = np.asarray(
        [index for index, label in enumerate(labels) if label in PROFILE_LABELS],
        dtype=int,
    )
    geometry_indices = np.asarray(
        [index for index, label in enumerate(labels) if label not in PROFILE_LABELS],
        dtype=int,
    )
    if tuple(labels[index] for index in profile_indices) != PROFILE_LABELS:
        raise ValueError(
            "Saved audit does not contain the expected six profile parameters "
            "in canonical order."
        )
    residuals = np.asarray(saved["residuals"], dtype=float)
    jacobian_geometry = np.asarray(saved["jacobian_geometry"], dtype=float)
    jacobian_profile = np.asarray(saved["jacobian_profile"], dtype=float)
    if jacobian_geometry.shape != (residuals.size, geometry_indices.size):
        raise ValueError(
            "Saved geometry Jacobian shape is inconsistent with residuals "
            "and parameter labels."
        )
    if jacobian_profile.shape != (residuals.size, profile_indices.size):
        raise ValueError(
            "Saved profile Jacobian shape is inconsistent with residuals "
            "and parameter labels."
        )
    jacobian_combined = np.zeros((residuals.size, len(labels)), dtype=float)
    jacobian_combined[:, geometry_indices] = jacobian_geometry
    jacobian_combined[:, profile_indices] = jacobian_profile
    x0, scales, lower, upper = _combined_coordinate_metadata(args, labels)
    geometry_labels = np.asarray(
        tuple(labels[index] for index in geometry_indices), dtype=np.str_
    )
    geometry = {
        "residuals": residuals.copy(),
        "jacobian": jacobian_geometry,
        "parameter_labels": geometry_labels,
        "x0": np.zeros((geometry_indices.size,), dtype=float),
        "coordinate_scales": np.full(
            (geometry_indices.size,), np.nan, dtype=float
        ),
        "lower_bounds": np.full((geometry_indices.size,), -np.inf, dtype=float),
        "upper_bounds": np.full((geometry_indices.size,), np.inf, dtype=float),
    }
    combined = {
        "residuals": residuals.copy(),
        "jacobian": jacobian_combined,
        "parameter_labels": np.asarray(labels, dtype=np.str_),
        "x0": x0,
        "coordinate_scales": scales,
        "lower_bounds": lower,
        "upper_bounds": upper,
    }
    return geometry, combined


def _max_error(reference: np.ndarray, trial: np.ndarray) -> tuple[float, float]:
    difference = np.abs(np.asarray(trial) - np.asarray(reference))
    if difference.size == 0:
        return 0.0, 0.0
    index = np.unravel_index(int(np.argmax(difference)), difference.shape)
    maximum = float(difference[index])
    denominator = max(abs(float(np.asarray(reference)[index])), 1.0e-300)
    return maximum, maximum / denominator


def _rank_and_singular_values(matrix: np.ndarray) -> tuple[int, np.ndarray]:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0 or singular_values[0] == 0.0:
        return 0, singular_values
    tolerance = (
        np.finfo(float).eps
        * max(matrix.shape)
        * float(singular_values[0])
    )
    return int(np.count_nonzero(singular_values > tolerance)), singular_values


def _trust_region_step(
    jacobian: np.ndarray,
    residuals: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, float]:
    """Minimum-norm damped least-squares step inside a Euclidean ball."""

    jacobian = np.asarray(jacobian, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    if jacobian.shape[1] == 0:
        return np.empty((0,), dtype=float), 0.0
    u, singular_values, vt = np.linalg.svd(jacobian, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] == 0.0:
        return np.zeros((jacobian.shape[1],), dtype=float), 0.0
    coefficients = u.T @ (-residuals)
    tolerance = (
        np.finfo(float).eps
        * max(jacobian.shape)
        * float(singular_values[0])
    )

    def step(damping: float) -> np.ndarray:
        factors = np.divide(
            singular_values,
            singular_values * singular_values + float(damping),
            out=np.zeros_like(singular_values),
            where=singular_values > tolerance,
        )
        return vt.T @ (factors * coefficients)

    undamped = step(0.0)
    if np.linalg.norm(undamped) <= radius:
        return undamped, 0.0
    lower, upper = 0.0, max(float(singular_values[0] ** 2), 1.0)
    while np.linalg.norm(step(upper)) > radius:
        upper *= 4.0
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        if np.linalg.norm(step(midpoint)) > radius:
            lower = midpoint
        else:
            upper = midpoint
    return step(upper), upper


def _fraction_to_bounds(
    x: np.ndarray,
    step: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    fraction = 1.0
    for value, delta, lo, hi in zip(x, step, lower, upper, strict=True):
        if delta > 0.0 and np.isfinite(hi):
            fraction = min(fraction, (hi - value) / delta)
        elif delta < 0.0 and np.isfinite(lo):
            fraction = min(fraction, (lo - value) / delta)
    return float(np.clip(fraction, 0.0, 1.0))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def _cost(residuals: np.ndarray) -> float:
    return 0.5 * float(np.dot(residuals, residuals))


def _linear_step_audit(
    args: argparse.Namespace,
    geometry: dict[str, np.ndarray],
    combined: dict[str, np.ndarray],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    geometry_labels = tuple(str(value) for value in geometry["parameter_labels"])
    combined_labels = tuple(str(value) for value in combined["parameter_labels"])
    profile_indices = np.asarray(
        [i for i, label in enumerate(combined_labels) if label in PROFILE_LABELS],
        dtype=int,
    )
    geometry_indices = np.asarray(
        [i for i, label in enumerate(combined_labels) if label not in PROFILE_LABELS],
        dtype=int,
    )
    embedded_geometry_labels = tuple(combined_labels[i] for i in geometry_indices)
    if embedded_geometry_labels != geometry_labels:
        raise AssertionError(
            "Combined geometry parameter order does not match geometry-only: "
            f"combined={embedded_geometry_labels}, geometry={geometry_labels}."
        )
    if tuple(combined_labels[i] for i in profile_indices) != PROFILE_LABELS:
        raise AssertionError("Combined profile parameter order is not the expected six-DoF order.")

    residual_geometry = np.asarray(geometry["residuals"], dtype=float)
    residual_combined = np.asarray(combined["residuals"], dtype=float)
    jacobian_geometry = np.asarray(geometry["jacobian"], dtype=float)
    jacobian_combined = np.asarray(combined["jacobian"], dtype=float)
    jacobian_profile = jacobian_combined[:, profile_indices]
    jacobian_embedded_geometry = jacobian_combined[:, geometry_indices]
    if residual_geometry.shape != residual_combined.shape:
        raise AssertionError(
            "Residual shapes differ: "
            f"geometry={residual_geometry.shape}, combined={residual_combined.shape}."
        )
    residual_max_abs, residual_max_relative = _max_error(
        residual_geometry, residual_combined
    )
    jacobian_max_abs, jacobian_max_relative = _max_error(
        jacobian_geometry, jacobian_embedded_geometry
    )
    residual_parity = bool(
        np.allclose(
            residual_geometry,
            residual_combined,
            rtol=args.parity_rtol,
            atol=args.parity_atol,
        )
    )
    jacobian_parity = bool(
        np.allclose(
            jacobian_geometry,
            jacobian_embedded_geometry,
            rtol=args.parity_rtol,
            atol=args.parity_atol,
        )
    )

    geometry_step, geometry_damping = _trust_region_step(
        jacobian_geometry,
        residual_geometry,
        float(args.geometry_trust_radius),
    )
    geometry_model_residual = (
        residual_combined + jacobian_embedded_geometry @ geometry_step
    )
    profile_step, profile_damping = _trust_region_step(
        jacobian_profile,
        geometry_model_residual,
        float(args.profile_trust_radius),
    )
    profile_x0 = np.asarray(combined["x0"], dtype=float)[profile_indices]
    profile_lower = np.asarray(combined["lower_bounds"], dtype=float)[profile_indices]
    profile_upper = np.asarray(combined["upper_bounds"], dtype=float)[profile_indices]
    profile_bound_fraction = _fraction_to_bounds(
        profile_x0, profile_step, profile_lower, profile_upper
    )
    profile_step = profile_step * profile_bound_fraction
    anchored_model_residual = geometry_model_residual + jacobian_profile @ profile_step

    profile_first_step, profile_first_damping = _trust_region_step(
        jacobian_profile,
        residual_combined,
        float(args.profile_trust_radius),
    )
    profile_first_bound_fraction = _fraction_to_bounds(
        profile_x0,
        profile_first_step,
        profile_lower,
        profile_upper,
    )
    profile_first_step = profile_first_step * profile_first_bound_fraction
    profile_first_model_residual = (
        residual_combined + jacobian_profile @ profile_first_step
    )
    profile_then_geometry_step, profile_then_geometry_damping = (
        _trust_region_step(
            jacobian_embedded_geometry,
            profile_first_model_residual,
            float(args.geometry_trust_radius),
        )
    )
    profile_then_geometry_model_residual = (
        profile_first_model_residual
        + jacobian_embedded_geometry @ profile_then_geometry_step
    )

    joint_radius = float(
        np.hypot(args.geometry_trust_radius, args.profile_trust_radius)
    )
    direct_step, direct_damping = _trust_region_step(
        jacobian_combined,
        residual_combined,
        joint_radius,
    )
    combined_x0 = np.asarray(combined["x0"], dtype=float)
    direct_bound_fraction = _fraction_to_bounds(
        combined_x0,
        direct_step,
        np.asarray(combined["lower_bounds"], dtype=float),
        np.asarray(combined["upper_bounds"], dtype=float),
    )
    direct_step = direct_step * direct_bound_fraction
    direct_model_residual = residual_combined + jacobian_combined @ direct_step

    rank_geometry, singular_geometry = _rank_and_singular_values(jacobian_geometry)
    rank_profile, singular_profile = _rank_and_singular_values(jacobian_profile)
    rank_combined, singular_combined = _rank_and_singular_values(jacobian_combined)
    initial_cost = _cost(residual_combined)
    geometry_cost = _cost(geometry_model_residual)
    anchored_cost = _cost(anchored_model_residual)
    direct_cost = _cost(direct_model_residual)
    direct_geometry_step = direct_step[geometry_indices]
    direct_profile_step = direct_step[profile_indices]
    profile_added_reduction = geometry_cost - anchored_cost

    geometry_only_full_step = np.zeros_like(direct_step)
    geometry_only_full_step[geometry_indices] = geometry_step
    geometry_then_profile_full_step = geometry_only_full_step.copy()
    geometry_then_profile_full_step[profile_indices] = profile_step
    profile_first_full_step = np.zeros_like(direct_step)
    profile_first_full_step[profile_indices] = profile_first_step
    profile_then_geometry_full_step = profile_first_full_step.copy()
    profile_then_geometry_full_step[geometry_indices] = (
        profile_then_geometry_step
    )

    candidate_steps = {
        "geometry_only": geometry_only_full_step,
        "geometry_then_profile": geometry_then_profile_full_step,
        "profile_first": profile_first_full_step,
        "profile_then_geometry": profile_then_geometry_full_step,
        "ordinary_direct_joint": direct_step,
    }
    candidate_model_residuals = {
        "geometry_only": geometry_model_residual,
        "geometry_then_profile": anchored_model_residual,
        "profile_first": profile_first_model_residual,
        "profile_then_geometry": profile_then_geometry_model_residual,
        "ordinary_direct_joint": direct_model_residual,
    }

    report = {
        "parameter_counts": {
            "geometry": len(geometry_labels),
            "profile": len(PROFILE_LABELS),
            "combined": len(combined_labels),
        },
        "parity": {
            "residual_pass": residual_parity,
            "residual_max_abs": residual_max_abs,
            "residual_max_relative": residual_max_relative,
            "geometry_jacobian_pass": jacobian_parity,
            "geometry_jacobian_max_abs": jacobian_max_abs,
            "geometry_jacobian_max_relative": jacobian_max_relative,
            "rtol": float(args.parity_rtol),
            "atol": float(args.parity_atol),
        },
        "ranks": {
            "geometry": rank_geometry,
            "profile": rank_profile,
            "combined": rank_combined,
        },
        "singular_values": {
            "geometry": singular_geometry.tolist(),
            "profile": singular_profile.tolist(),
            "combined": singular_combined.tolist(),
        },
        "trust_radii": {
            "geometry": float(args.geometry_trust_radius),
            "profile": float(args.profile_trust_radius),
            "direct_joint": joint_radius,
        },
        "linearized_costs": {
            "initial": initial_cost,
            "geometry_only_step": geometry_cost,
            "geometry_anchored_profile_correction": anchored_cost,
            "profile_first_step": _cost(profile_first_model_residual),
            "profile_then_geometry_step": _cost(
                profile_then_geometry_model_residual
            ),
            "ordinary_direct_joint_step": direct_cost,
        },
        "linearized_reductions": {
            "geometry_only": initial_cost - geometry_cost,
            "profile_added_after_geometry": profile_added_reduction,
            "anchored_total": initial_cost - anchored_cost,
            "profile_first": initial_cost - _cost(profile_first_model_residual),
            "profile_then_geometry": (
                initial_cost - _cost(profile_then_geometry_model_residual)
            ),
            "ordinary_direct_joint": initial_cost - direct_cost,
        },
        "steps": {
            "geometry_only_l2": float(np.linalg.norm(geometry_step)),
            "anchored_profile_l2": float(np.linalg.norm(profile_step)),
            "profile_first_l2": float(np.linalg.norm(profile_first_step)),
            "profile_then_geometry_l2": float(
                np.linalg.norm(profile_then_geometry_step)
            ),
            "ordinary_direct_geometry_l2": float(
                np.linalg.norm(direct_geometry_step)
            ),
            "ordinary_direct_profile_l2": float(np.linalg.norm(direct_profile_step)),
            "ordinary_vs_geometry_geometry_cosine": _cosine(
                geometry_step, direct_geometry_step
            ),
            "ordinary_vs_geometry_geometry_l2_difference": float(
                np.linalg.norm(direct_geometry_step - geometry_step)
            ),
            "geometry_damping": geometry_damping,
            "profile_damping": profile_damping,
            "profile_first_damping": profile_first_damping,
            "profile_then_geometry_damping": profile_then_geometry_damping,
            "ordinary_direct_damping": direct_damping,
            "profile_bound_fraction": profile_bound_fraction,
            "profile_first_bound_fraction": profile_first_bound_fraction,
            "ordinary_direct_bound_fraction": direct_bound_fraction,
        },
        "profile_projected_gradient": {
            "at_seed_l2": float(
                np.linalg.norm(jacobian_profile.T @ residual_combined)
            ),
            "after_geometry_step_l2": float(
                np.linalg.norm(jacobian_profile.T @ geometry_model_residual)
            ),
        },
        "linear_model_geometry_then_profile_adds_reduction": bool(
            residual_parity
            and jacobian_parity
            and profile_added_reduction > max(1.0e-12, 1.0e-12 * initial_cost)
        ),
        "nonlinear_candidate_results": {},
    }
    arrays = {
        "residuals": residual_combined,
        "jacobian_geometry": jacobian_embedded_geometry,
        "jacobian_profile": jacobian_profile,
        "geometry_step": geometry_step,
        "anchored_profile_step": profile_step,
        "ordinary_direct_step": direct_step,
        "profile_first_step": profile_first_step,
        "profile_then_geometry_step": profile_then_geometry_step,
        "geometry_model_residual": geometry_model_residual,
        "anchored_model_residual": anchored_model_residual,
        "profile_first_model_residual": profile_first_model_residual,
        "profile_then_geometry_model_residual": (
            profile_then_geometry_model_residual
        ),
        "ordinary_direct_model_residual": direct_model_residual,
        "combined_parameter_labels": np.asarray(combined_labels, dtype=np.str_),
        "combined_x0": combined_x0,
        "combined_coordinate_scales": np.asarray(
            combined["coordinate_scales"], dtype=float
        ),
        "combined_lower_bounds": np.asarray(combined["lower_bounds"], dtype=float),
        "combined_upper_bounds": np.asarray(combined["upper_bounds"], dtype=float),
        "geometry_indices": geometry_indices,
        "profile_indices": profile_indices,
    }
    for name, step_value in candidate_steps.items():
        arrays[f"candidate_step_{name}"] = step_value
        arrays[f"candidate_x_{name}"] = combined_x0 + step_value
        arrays[f"candidate_model_residual_{name}"] = (
            candidate_model_residuals[name]
        )
    return report, arrays


def _evaluate_nonlinear_candidates(
    args: argparse.Namespace,
    report: dict[str, object],
    arrays: dict[str, np.ndarray],
    temporary_path: Path,
    *,
    seed_evaluations_this_run: int,
) -> None:
    """Evaluate requested candidate points with the actual nonlinear model."""

    initial_cost = float(report["linearized_costs"]["initial"])
    labels = np.asarray(arrays["combined_parameter_labels"], dtype=np.str_)
    x0 = np.asarray(arrays["combined_x0"], dtype=float)
    nonlinear_results = report["nonlinear_candidate_results"]
    for name in dict.fromkeys(args.nonlinear_candidates):
        candidate_x = np.asarray(arrays[f"candidate_x_{name}"], dtype=float)
        model_residual = np.asarray(
            arrays[f"candidate_model_residual_{name}"], dtype=float
        )
        model_cost = _cost(model_residual)
        model_reduction = initial_cost - model_cost
        candidate_input = temporary_path / f"candidate_input_{name}.npz"
        candidate_output = temporary_path / f"candidate_output_{name}.npz"
        np.savez(
            candidate_input,
            candidate_name=np.asarray(name, dtype=np.str_),
            parameter_labels=labels,
            x0=x0,
            candidate_x=candidate_x,
        )
        print(
            "[block-step-audit] running actual nonlinear candidate "
            f"name={name} predicted_cost={model_cost:.16e}",
            flush=True,
        )
        _run_worker(
            "nonlinear_candidate",
            candidate_output,
            candidate_input=candidate_input,
        )
        actual = _load(candidate_output)
        actual_residual = np.asarray(actual["residuals"], dtype=float)
        actual_cost = float(np.asarray(actual["cost"]).item())
        actual_reduction = initial_cost - actual_cost
        agreement_ratio = (
            actual_reduction / model_reduction
            if model_reduction > 0.0
            else float("nan")
        )
        finite = bool(np.asarray(actual["finite"]).item())
        nonlinear_results[name] = {
            "finite": finite,
            "step_l2": float(
                np.linalg.norm(candidate_x - x0)
            ),
            "predicted_linearized_cost": model_cost,
            "predicted_reduction": model_reduction,
            "actual_nonlinear_cost": actual_cost,
            "actual_reduction": actual_reduction,
            "actual_to_predicted_reduction_ratio": agreement_ratio,
            "actual_residual_norm": float(np.linalg.norm(actual_residual)),
            "worker_elapsed_s": float(np.asarray(actual["elapsed_s"]).item()),
            "physical_profiles": dict(
                zip(
                    (
                        str(value)
                        for value in actual["physical_profile_labels"]
                    ),
                    (
                        float(value)
                        for value in actual["physical_profile_values"]
                    ),
                    strict=True,
                )
            ),
        }
        arrays[f"actual_residual_{name}"] = actual_residual
        print(
            "[block-step-audit] actual nonlinear candidate "
            f"name={name} actual_cost={actual_cost:.16e} "
            f"actual_reduction={actual_reduction:.16e} "
            f"agreement_ratio={agreement_ratio:.16e}",
            flush=True,
        )
    report["seed_problem_evaluations_this_run"] = int(
        seed_evaluations_this_run
    )
    report["nonlinear_candidate_evaluations_this_run"] = len(
        nonlinear_results
    )
    report["expensive_problem_evaluations_this_run"] = (
        int(seed_evaluations_this_run) + len(nonlinear_results)
    )


def main() -> int:
    args = parser().parse_args()
    _validate_args(args)
    if args.worker_kind is not None:
        if args.worker_output is None:
            raise ValueError("--worker-output is required with --worker-kind.")
        if (
            args.worker_kind != "nonlinear_candidate"
            and args.candidate_input is not None
        ):
            raise ValueError(
                "--candidate-input is only valid for a nonlinear candidate worker."
            )
        return _worker(args)
    if args.worker_output is not None:
        raise ValueError("--worker-output is only valid with --worker-kind.")
    if args.candidate_input is not None:
        raise ValueError("--candidate-input is only valid with --worker-kind.")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="neopax_block_step_audit_") as temporary:
        temporary_path = Path(temporary)
        if args.reuse_linear_audit_npz is None:
            geometry_path = temporary_path / "geometry.npz"
            combined_path = temporary_path / "combined.npz"
            print(
                "[block-step-audit] running isolated geometry-only seed evaluation",
                flush=True,
            )
            _run_worker("geometry", geometry_path)
            print(
                "[block-step-audit] running isolated combined seed evaluation",
                flush=True,
            )
            _run_worker("combined", combined_path)
            geometry = _load(geometry_path)
            combined = _load(combined_path)
            seed_evaluations_this_run = 2
            seed_source = "fresh_isolated_workers"
        else:
            print(
                "[block-step-audit] reusing saved seed residual/Jacobian "
                f"from {args.reuse_linear_audit_npz.resolve()}",
                flush=True,
            )
            print(
                "[block-step-audit] reuse assumes that the saved NPZ used "
                "the same seed, database grid, objectives, and CLI settings "
                "as this invocation",
                flush=True,
            )
            geometry, combined = _seed_data_from_saved_linear_audit(
                args, args.reuse_linear_audit_npz
            )
            seed_evaluations_this_run = 0
            seed_source = str(args.reuse_linear_audit_npz.resolve())
        report, arrays = _linear_step_audit(args, geometry, combined)
        report["seed_data_source"] = seed_source
        _evaluate_nonlinear_candidates(
            args,
            report,
            arrays,
            temporary_path,
            seed_evaluations_this_run=seed_evaluations_this_run,
        )

    json_path = out_dir / "geometry_profiles_one_shot_block_step_audit.json"
    npz_path = out_dir / "geometry_profiles_one_shot_block_step_audit.npz"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez(npz_path, **arrays)
    print(json.dumps(report, indent=2), flush=True)
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {npz_path}", flush=True)
    if not report["parity"]["residual_pass"]:
        print("[block-step-audit] FAIL: residual embedding parity", flush=True)
        return 2
    if not report["parity"]["geometry_jacobian_pass"]:
        print("[block-step-audit] FAIL: geometry Jacobian embedding parity", flush=True)
        return 2
    nonlinear_results = report["nonlinear_candidate_results"]
    if nonlinear_results and not all(
        result["finite"] for result in nonlinear_results.values()
    ):
        print(
            "[block-step-audit] FAIL: at least one requested nonlinear "
            "candidate produced nonfinite residuals",
            flush=True,
        )
        return 4
    if nonlinear_results:
        print(
            "[block-step-audit] COMPLETE: actual nonlinear candidate costs "
            "were evaluated; inspect nonlinear_candidate_results rather than "
            "the affine costs when choosing a method",
            flush=True,
        )
        return 0
    print(
        "[block-step-audit] COMPLETE: seed embedding and affine steps were "
        "audited, but no nonlinear candidate was requested",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
