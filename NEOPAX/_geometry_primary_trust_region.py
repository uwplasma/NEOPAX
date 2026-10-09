"""Geometry-primary nonlinear least squares with profile corrections.

This opt-in solver constructs one composite step per accepted nonlinear state:
first an ESS-geometry Gauss--Newton step, then a nominal-profile correction to
the residual left by that fixed geometry step.  A rejected combined trial
contracts the profile correction before the geometry trust radius is touched.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from ._block_trust_region import (
    _block_model_step,
    _cost,
    _projected_gradient,
)


@dataclasses.dataclass(frozen=True, slots=True)
class GeometryPrimaryTrustRegionResult:
    """Result of one geometry-primary/profile-correction optimization."""

    x: np.ndarray
    fun: np.ndarray
    jac: np.ndarray
    cost: float
    optimality: float
    nfev: int
    njev: int
    nit: int
    status: int
    message: str
    success: bool
    geometry_radius: float
    profile_fraction_limit: float
    accepted_steps: int
    rejected_steps: int
    profile_contractions: int
    geometry_contractions: int
    geometry_only_trials: int
    evaluation: object


def geometry_primary_profile_correction_least_squares(
    problem,
    *,
    profile_mask,
    bounds,
    initial_evaluation=None,
    max_nfev: int = 30,
    geometry_initial_radius: float = 1.0,
    geometry_min_radius: float = 1.0e-4,
    geometry_max_radius: float = 4.0,
    profile_initial_fraction_limit: float = 0.10,
    profile_min_fraction_limit: float = 1.0e-3,
    profile_max_fraction_limit: float = 0.25,
    proximal_weight: float = 1.0e-8,
    acceptance_threshold: float = 0.10,
    shrink_threshold: float = 0.25,
    expansion_threshold: float = 0.75,
    shrink_factor: float = 0.25,
    expansion_factor: float = 2.0,
    ftol: float = 1.0e-6,
    xtol: float = 1.0e-10,
    gtol: float = 1.0e-8,
    iteration_reporter: Callable[[object], str] | None = None,
    verbose: int = 1,
) -> GeometryPrimaryTrustRegionResult:
    """Optimize with a fixed geometry baseline plus a profile correction.

    Both substeps use the same residual/Jacobian evaluation.  The profile
    substep sees ``r + J_g d_g`` and cannot alter ``d_g``.  If the combined
    nonlinear trial is rejected, only the profile limit contracts.  Once the
    profile limit reaches its minimum, the exact geometry-only trial is tested;
    only rejection of that trial contracts the geometry radius.
    """

    if int(max_nfev) < 1:
        raise ValueError("max_nfev must be positive.")
    if not (
        0.0
        <= acceptance_threshold
        < shrink_threshold
        < expansion_threshold
        < 1.0
    ):
        raise ValueError(
            "Require 0 <= acceptance < shrink < expansion < 1."
        )
    for name, value in (
        ("geometry_initial_radius", geometry_initial_radius),
        ("geometry_min_radius", geometry_min_radius),
        ("geometry_max_radius", geometry_max_radius),
        ("profile_initial_fraction_limit", profile_initial_fraction_limit),
        ("profile_min_fraction_limit", profile_min_fraction_limit),
        ("profile_max_fraction_limit", profile_max_fraction_limit),
    ):
        if float(value) <= 0.0:
            raise ValueError(f"{name} must be positive.")
    if not (
        float(geometry_min_radius)
        <= float(geometry_initial_radius)
        <= float(geometry_max_radius)
    ):
        raise ValueError(
            "Geometry trust radii must be ordered min <= initial <= max."
        )
    if not (
        float(profile_min_fraction_limit)
        <= float(profile_initial_fraction_limit)
        <= float(profile_max_fraction_limit)
    ):
        raise ValueError(
            "Profile fraction limits must be ordered min <= initial <= max."
        )
    if float(proximal_weight) < 0.0:
        raise ValueError("proximal_weight must be nonnegative.")
    if not 0.0 < float(shrink_factor) < 1.0:
        raise ValueError("shrink_factor must lie strictly between zero and one.")
    if float(expansion_factor) <= 1.0:
        raise ValueError("expansion_factor must be greater than one.")
    for name, value in (("ftol", ftol), ("xtol", xtol), ("gtol", gtol)):
        if float(value) < 0.0:
            raise ValueError(f"{name} must be nonnegative.")

    x = np.asarray(jax.device_get(problem.x0), dtype=float)
    profile_mask = np.asarray(profile_mask, dtype=bool)
    if profile_mask.shape != x.shape:
        raise ValueError(
            "profile_mask must have one entry per optimizer coordinate."
        )
    geometry_mask = ~profile_mask
    if not np.any(profile_mask) or not np.any(geometry_mask):
        raise ValueError(
            "Geometry-primary optimization requires nonempty geometry and "
            "profile blocks."
        )
    profile_indices = np.flatnonzero(profile_mask)
    geometry_indices = np.flatnonzero(geometry_mask)
    lower, upper = (np.asarray(value, dtype=float) for value in bounds)
    if lower.shape != x.shape or upper.shape != x.shape:
        raise ValueError("bounds must have one lower/upper value per coordinate.")
    if np.any(lower > upper) or np.any(x < lower) or np.any(x > upper):
        raise ValueError("Initial coordinates must satisfy ordered bounds.")

    def host_evaluation(evaluation):
        residuals = np.asarray(jax.device_get(evaluation.residuals), dtype=float)
        jacobian = np.asarray(jax.device_get(evaluation.jacobian), dtype=float)
        if residuals.ndim != 1:
            raise ValueError("Least-squares residuals must be one-dimensional.")
        if jacobian.shape != (residuals.size, x.size):
            raise ValueError(
                "Least-squares Jacobian has incompatible shape: "
                f"{jacobian.shape}, expected {(residuals.size, x.size)}."
            )
        if not np.all(np.isfinite(residuals)) or not np.all(np.isfinite(jacobian)):
            raise FloatingPointError("Nonfinite residual or Jacobian evaluation.")
        return residuals, jacobian

    evaluation = (
        problem.evaluate(jnp.asarray(x, dtype=jnp.float64))
        if initial_evaluation is None
        else initial_evaluation
    )
    residuals, jacobian = host_evaluation(evaluation)
    cost = _cost(residuals)
    nfev = 1
    njev = 1
    nit = 0
    accepted_steps = 0
    rejected_steps = 0
    profile_contractions = 0
    geometry_contractions = 0
    geometry_only_trials = 0
    geometry_radius = float(geometry_initial_radius)
    profile_fraction_limit = float(profile_initial_fraction_limit)
    profile_suspended = False
    status = 0
    message = "The maximum number of function evaluations is exceeded."

    while nfev < int(max_nfev):
        gradient = jacobian.T @ residuals
        projected_gradient = _projected_gradient(x, gradient, lower, upper)
        optimality = float(np.linalg.norm(projected_gradient, ord=np.inf))
        if optimality <= float(gtol):
            status = 1
            message = "The projected gradient tolerance is satisfied."
            break

        geometry_step_reduced, geometry_metadata = _block_model_step(
            residuals,
            jacobian[:, geometry_indices],
            x[geometry_indices],
            lower[geometry_indices],
            upper[geometry_indices],
            np.zeros((geometry_indices.size,), dtype=bool),
            geometry_radius=geometry_radius,
            profile_fraction_limit=1.0,
            proximal_weight=proximal_weight,
        )
        step = np.zeros_like(x)
        step[geometry_indices] = geometry_step_reduced
        geometry_model_residuals = residuals + jacobian @ step
        geometry_predicted_reduction = cost - _cost(
            geometry_model_residuals
        )

        if profile_suspended:
            profile_step_reduced = np.zeros((profile_indices.size,), dtype=float)
            profile_metadata = {
                "subproblem_solver": "suspended_geometry_only",
                "profile_step_max_abs": 0.0,
            }
        else:
            profile_step_reduced, profile_metadata = _block_model_step(
                geometry_model_residuals,
                jacobian[:, profile_indices],
                x[profile_indices],
                lower[profile_indices],
                upper[profile_indices],
                np.ones((profile_indices.size,), dtype=bool),
                geometry_radius=1.0,
                profile_fraction_limit=profile_fraction_limit,
                proximal_weight=proximal_weight,
            )
        step[profile_indices] = profile_step_reduced
        model_residuals = residuals + jacobian @ step
        geometry_model_cost = _cost(geometry_model_residuals)
        model_cost = _cost(model_residuals)
        profile_predicted_correction = geometry_model_cost - model_cost
        if profile_predicted_correction < -1.0e-12 * max(cost, 1.0):
            # A numerical subproblem failure must not make the geometry
            # baseline worse. Fall back exactly to the geometry-only step.
            profile_step_reduced = np.zeros_like(profile_step_reduced)
            step[profile_indices] = 0.0
            model_residuals = geometry_model_residuals
            model_cost = geometry_model_cost
            profile_predicted_correction = 0.0
            profile_metadata = {
                "subproblem_solver": "rejected_nonimproving_correction",
                "profile_step_max_abs": 0.0,
            }
        predicted_reduction = cost - model_cost
        if not np.isfinite(predicted_reduction) or predicted_reduction <= 0.0:
            status = 4
            message = "The composite model produced no positive reduction."
            break

        profile_step_max_abs = float(
            np.max(np.abs(profile_step_reduced))
            if profile_step_reduced.size
            else 0.0
        )
        is_geometry_only_trial = bool(profile_step_max_abs == 0.0)
        geometry_only_trials += int(is_geometry_only_trial)
        trial_x = np.clip(x + step, lower, upper)
        trial_evaluation = None
        failure = None
        try:
            trial_evaluation = problem.evaluate(
                jnp.asarray(trial_x, dtype=jnp.float64)
            )
            trial_residuals, trial_jacobian = host_evaluation(trial_evaluation)
            trial_cost = _cost(trial_residuals)
        except Exception as exc:
            failure = str(exc)
            trial_residuals = None
            trial_jacobian = None
            trial_cost = float("inf")
        nfev += 1
        njev += int(trial_evaluation is not None)
        nit += 1
        actual_reduction = cost - trial_cost
        ratio = actual_reduction / predicted_reduction
        accepted = bool(
            np.isfinite(trial_cost)
            and actual_reduction > 0.0
            and ratio >= float(acceptance_threshold)
        )

        if verbose:
            details = ""
            if trial_evaluation is not None and iteration_reporter is not None:
                text = str(iteration_reporter(trial_evaluation)).strip()
                details = f" {text}" if text else ""
            print(
                "[NEOPAX geometry_primary] "
                f"eval={nfev} accepted={accepted} "
                f"geometry_only_trial={is_geometry_only_trial} "
                f"cost={trial_cost:.8e} actual_reduction={actual_reduction:.8e} "
                f"predicted_reduction={predicted_reduction:.8e} "
                f"geometry_predicted_reduction="
                f"{geometry_predicted_reduction:.8e} "
                f"profile_predicted_correction="
                f"{profile_predicted_correction:.8e} ratio={ratio:.8e} "
                f"geometry_radius={geometry_radius:.6e} "
                f"profile_fraction_limit={profile_fraction_limit:.6e} "
                f"geometry_model_step="
                f"{geometry_metadata['subproblem_solver']} "
                f"profile_model_step={profile_metadata['subproblem_solver']} "
                f"geometry_step_l2="
                f"{np.linalg.norm(geometry_step_reduced):.6e} "
                f"profile_step_max_abs={profile_step_max_abs:.6e}"
                f"{details}",
                flush=True,
            )
            if failure is not None:
                print(
                    "[NEOPAX geometry_primary] trial failure: " + failure,
                    flush=True,
                )

        if not accepted:
            rejected_steps += 1
            if not is_geometry_only_trial:
                if profile_fraction_limit > float(
                    profile_min_fraction_limit
                ) * (1.0 + 1.0e-12):
                    profile_fraction_limit = max(
                        float(profile_min_fraction_limit),
                        profile_fraction_limit * float(shrink_factor),
                    )
                    profile_contractions += 1
                else:
                    # The next trial is exactly the retained geometry proposal.
                    profile_suspended = True
            else:
                if geometry_radius <= float(geometry_min_radius) * (
                    1.0 + 1.0e-12
                ):
                    status = 5
                    message = (
                        "The geometry-only trial failed at the minimum "
                        "geometry trust radius."
                    )
                    break
                geometry_radius = max(
                    float(geometry_min_radius),
                    geometry_radius * float(shrink_factor),
                )
                geometry_contractions += 1
                profile_suspended = True
            continue

        previous_cost = cost
        x = trial_x
        evaluation = trial_evaluation
        residuals = trial_residuals
        jacobian = trial_jacobian
        cost = trial_cost
        accepted_steps += 1

        geometry_activity = float(
            np.linalg.norm(geometry_step_reduced) / geometry_radius
        )
        profile_activity = (
            profile_step_max_abs / profile_fraction_limit
            if not is_geometry_only_trial
            else 0.0
        )
        if ratio < float(shrink_threshold):
            if not is_geometry_only_trial:
                new_limit = max(
                    float(profile_min_fraction_limit),
                    profile_fraction_limit * float(shrink_factor),
                )
                profile_contractions += int(new_limit < profile_fraction_limit)
                profile_fraction_limit = new_limit
            else:
                new_radius = max(
                    float(geometry_min_radius),
                    geometry_radius * float(shrink_factor),
                )
                geometry_contractions += int(new_radius < geometry_radius)
                geometry_radius = new_radius
        elif ratio > float(expansion_threshold):
            if geometry_activity >= 0.8:
                geometry_radius = min(
                    float(geometry_max_radius),
                    geometry_radius * float(expansion_factor),
                )
            if profile_activity >= 0.8:
                profile_fraction_limit = min(
                    float(profile_max_fraction_limit),
                    profile_fraction_limit * float(expansion_factor),
                )
        profile_suspended = False

        if actual_reduction <= float(ftol) * max(previous_cost, 1.0):
            status = 2
            message = "The actual cost reduction tolerance is satisfied."
            break
        if np.linalg.norm(step) <= float(xtol) * (
            float(xtol) + np.linalg.norm(x)
        ):
            status = 3
            message = "The step tolerance is satisfied."
            break

    final_gradient = jacobian.T @ residuals
    final_optimality = float(
        np.linalg.norm(
            _projected_gradient(x, final_gradient, lower, upper), ord=np.inf
        )
    )
    return GeometryPrimaryTrustRegionResult(
        x=np.asarray(x, dtype=float),
        fun=np.asarray(residuals, dtype=float),
        jac=np.asarray(jacobian, dtype=float),
        cost=float(cost),
        optimality=final_optimality,
        nfev=int(nfev),
        njev=int(njev),
        nit=int(nit),
        status=int(status),
        message=message,
        success=bool(status in (1, 2, 3)),
        geometry_radius=float(geometry_radius),
        profile_fraction_limit=float(profile_fraction_limit),
        accepted_steps=int(accepted_steps),
        rejected_steps=int(rejected_steps),
        profile_contractions=int(profile_contractions),
        geometry_contractions=int(geometry_contractions),
        geometry_only_trials=int(geometry_only_trials),
        evaluation=evaluation,
    )


__all__ = [
    "GeometryPrimaryTrustRegionResult",
    "geometry_primary_profile_correction_least_squares",
]
