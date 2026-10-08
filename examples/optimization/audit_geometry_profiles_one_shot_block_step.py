#!/usr/bin/env python
"""Audit a geometry-anchored one-shot geometry/profile Gauss-Newton step.

This diagnostic does not run an optimization and does not modify either the
geometry-only or the existing combined optimization example.  It evaluates
the geometry-only and combined problems at the identical physical seed in
separate worker processes, checks that the combined problem embeds the exact
geometry residual/Jacobian, and then compares three linearized steps:

1. the established geometry-only step;
2. the ordinary simultaneous geometry/profile step; and
3. the geometry-only step followed by a profile correction using the same
   residual and Jacobian evaluation.

Only the two worker evaluations are expensive.  Every step comparison after
that is host-side linear algebra and does not call VMEC, NTX, or transport.
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
        "--worker-kind",
        choices=("geometry", "combined"),
        help=argparse.SUPPRESS,
    )
    out.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
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

    include_profiles = args.worker_kind == "combined"
    problem = _build_problem(args, include_profiles=include_profiles)
    x0 = np.asarray(jax.device_get(problem.x0), dtype=float)
    evaluation = problem.evaluate(x0)
    residuals = np.asarray(jax.device_get(evaluation.residuals), dtype=float)
    jacobian = np.asarray(jax.device_get(evaluation.jacobian), dtype=float)
    labels = np.asarray(tuple(problem.parameter_labels), dtype=np.str_)
    scales = np.asarray(jax.device_get(problem.x_scale), dtype=float)
    lower = np.full_like(x0, -np.inf)
    upper = np.full_like(x0, np.inf)
    if include_profiles:
        lower, upper = combined_example.scaled_bounds(problem)
    args.worker_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.worker_output,
        residuals=residuals,
        jacobian=jacobian,
        parameter_labels=labels,
        x0=x0,
        coordinate_scales=scales,
        lower_bounds=np.asarray(lower, dtype=float),
        upper_bounds=np.asarray(upper, dtype=float),
    )
    print(
        "[block-step-audit] "
        f"worker={args.worker_kind} residuals={residuals.shape} "
        f"jacobian={jacobian.shape} wrote={args.worker_output}",
        flush=True,
    )
    return 0


def _run_worker(kind: str, output: Path) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        *sys.argv[1:],
        "--worker-kind",
        kind,
        "--worker-output",
        str(output),
    ]
    subprocess.run(command, check=True)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


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

    report = {
        "expensive_problem_evaluations": 2,
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
            "ordinary_direct_joint_step": direct_cost,
        },
        "linearized_reductions": {
            "geometry_only": initial_cost - geometry_cost,
            "profile_added_after_geometry": profile_added_reduction,
            "anchored_total": initial_cost - anchored_cost,
            "ordinary_direct_joint": initial_cost - direct_cost,
        },
        "steps": {
            "geometry_only_l2": float(np.linalg.norm(geometry_step)),
            "anchored_profile_l2": float(np.linalg.norm(profile_step)),
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
            "ordinary_direct_damping": direct_damping,
            "profile_bound_fraction": profile_bound_fraction,
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
        "supports_geometry_anchored_direction": bool(
            residual_parity
            and jacobian_parity
            and profile_added_reduction > max(1.0e-12, 1.0e-12 * initial_cost)
        ),
    }
    arrays = {
        "residuals": residual_combined,
        "jacobian_geometry": jacobian_embedded_geometry,
        "jacobian_profile": jacobian_profile,
        "geometry_step": geometry_step,
        "anchored_profile_step": profile_step,
        "ordinary_direct_step": direct_step,
        "geometry_model_residual": geometry_model_residual,
        "anchored_model_residual": anchored_model_residual,
        "ordinary_direct_model_residual": direct_model_residual,
        "combined_parameter_labels": np.asarray(combined_labels, dtype=np.str_),
    }
    return report, arrays


def main() -> int:
    args = parser().parse_args()
    _validate_args(args)
    if args.worker_kind is not None:
        if args.worker_output is None:
            raise ValueError("--worker-output is required with --worker-kind.")
        return _worker(args)
    if args.worker_output is not None:
        raise ValueError("--worker-output is only valid with --worker-kind.")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="neopax_block_step_audit_") as temporary:
        temporary_path = Path(temporary)
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

    report, arrays = _linear_step_audit(args, geometry, combined)
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
    if report["supports_geometry_anchored_direction"]:
        print(
            "[block-step-audit] PASS: the same seed/Jacobian supports a "
            "cost-reducing profile correction after the geometry step",
            flush=True,
        )
        return 0
    print(
        "[block-step-audit] INCONCLUSIVE: embedding parity passed but the "
        "linearized profile correction did not add meaningful reduction",
        flush=True,
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
