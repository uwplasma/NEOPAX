"""Block-structured nonlinear trust-region least squares.

This module is intentionally opt-in.  It does not replace the established
``optimization.least_squares`` path.  It is used by the combined
geometry/profile experiment where one shared SciPy trust region is not an
appropriate metric for two physically different, locally redundant blocks.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True, slots=True)
class BlockTrustRegionResult:
    """Result of a block-structured nonlinear least-squares run."""

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
    evaluation: object


def _cost(residuals: np.ndarray) -> float:
    residuals = np.asarray(residuals, dtype=float)
    return 0.5 * float(residuals @ residuals)


def _projected_gradient(
    x: np.ndarray,
    gradient: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    projected = np.asarray(gradient, dtype=float).copy()
    tolerance = 1.0e-12 * np.maximum(1.0, np.abs(x))
    at_lower = x <= lower + tolerance
    at_upper = x >= upper - tolerance
    projected[at_lower & (projected > 0.0)] = 0.0
    projected[at_upper & (projected < 0.0)] = 0.0
    return projected


def _block_model_step(
    residuals: np.ndarray,
    jacobian: np.ndarray,
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    profile_mask: np.ndarray,
    *,
    geometry_radius: float,
    profile_fraction_limit: float,
    proximal_weight: float,
) -> tuple[np.ndarray, dict[str, float | bool | str]]:
    """Solve one convex block-constrained Gauss--Newton subproblem."""

    from scipy.optimize import Bounds, minimize

    residuals = np.asarray(residuals, dtype=float)
    jacobian = np.asarray(jacobian, dtype=float)
    x = np.asarray(x, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    profile_mask = np.asarray(profile_mask, dtype=bool)
    geometry_mask = ~profile_mask
    profile_indices = np.flatnonzero(profile_mask)
    geometry_indices = np.flatnonzero(geometry_mask)

    step_lower = lower - x
    step_upper = upper - x
    if profile_indices.size:
        step_lower[profile_indices] = np.maximum(
            step_lower[profile_indices], -float(profile_fraction_limit)
        )
        step_upper[profile_indices] = np.minimum(
            step_upper[profile_indices], float(profile_fraction_limit)
        )

    profile_radius = max(
        float(profile_fraction_limit) * np.sqrt(max(profile_indices.size, 1)),
        np.finfo(float).eps,
    )
    geometry_radius_eff = max(float(geometry_radius), np.finfo(float).eps)

    def objective(step):
        linearized = residuals + jacobian @ step
        value = 0.5 * float(linearized @ linearized)
        if proximal_weight > 0.0:
            if geometry_indices.size:
                normalized = step[geometry_indices] / geometry_radius_eff
                value += 0.5 * float(proximal_weight) * float(
                    normalized @ normalized
                )
            if profile_indices.size:
                normalized = step[profile_indices] / profile_radius
                value += 0.5 * float(proximal_weight) * float(
                    normalized @ normalized
                )
        return value

    def gradient(step):
        linearized = residuals + jacobian @ step
        value = jacobian.T @ linearized
        if proximal_weight > 0.0:
            if geometry_indices.size:
                value[geometry_indices] += (
                    float(proximal_weight)
                    * step[geometry_indices]
                    / geometry_radius_eff**2
                )
            if profile_indices.size:
                value[profile_indices] += (
                    float(proximal_weight)
                    * step[profile_indices]
                    / profile_radius**2
                )
        return value

    constraints = []
    if geometry_indices.size:
        def geometry_constraint(step):
            block = step[geometry_indices]
            return geometry_radius_eff**2 - float(block @ block)

        def geometry_constraint_jacobian(step):
            value = np.zeros_like(step)
            value[geometry_indices] = -2.0 * step[geometry_indices]
            return value

        constraints.append(
            {
                "type": "ineq",
                "fun": geometry_constraint,
                "jac": geometry_constraint_jacobian,
            }
        )
    if profile_indices.size:
        def profile_constraint(step):
            block = step[profile_indices]
            return profile_radius**2 - float(block @ block)

        def profile_constraint_jacobian(step):
            value = np.zeros_like(step)
            value[profile_indices] = -2.0 * step[profile_indices]
            return value

        constraints.append(
            {
                "type": "ineq",
                "fun": profile_constraint,
                "jac": profile_constraint_jacobian,
            }
        )

    result = minimize(
        objective,
        np.zeros_like(x),
        jac=gradient,
        method="SLSQP",
        bounds=Bounds(step_lower, step_upper),
        constraints=constraints,
        options={"ftol": 1.0e-12, "maxiter": 500, "disp": False},
    )
    step = np.asarray(result.x, dtype=float)
    feasibility_tolerance = 1.0e-8
    feasible = bool(
        np.all(step >= step_lower - feasibility_tolerance)
        and np.all(step <= step_upper + feasibility_tolerance)
    )
    if geometry_indices.size:
        feasible = feasible and bool(
            np.linalg.norm(step[geometry_indices])
            <= geometry_radius_eff * (1.0 + feasibility_tolerance)
        )
    if profile_indices.size:
        feasible = feasible and bool(
            np.linalg.norm(step[profile_indices])
            <= profile_radius * (1.0 + feasibility_tolerance)
        )
    if not feasible:
        raise RuntimeError(
            "Block trust-region subproblem returned an infeasible step: "
            f"status={result.status}, message={result.message}."
        )
    metadata: dict[str, float | bool | str] = {
        "success": bool(result.success),
        "message": str(result.message),
        "geometry_step_norm": float(
            np.linalg.norm(step[geometry_indices])
        ),
        "profile_step_norm": float(np.linalg.norm(step[profile_indices])),
        "profile_step_max_abs": float(
            np.max(np.abs(step[profile_indices]))
            if profile_indices.size
            else 0.0
        ),
    }
    return step, metadata


def block_trust_region_least_squares(
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
    proximal_weight: float = 1.0e-10,
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
) -> BlockTrustRegionResult:
    """Run one nonlinear optimization with independent physical block limits.

    Geometry coordinates are assumed to already be ESS-scaled. Profile
    coordinates are assumed to be centered nominal deltas, so a profile step
    of ``0.1`` is a ten-percent change from the seed value. Both blocks enter
    every quadratic model and every accepted nonlinear trial.
    """

    if int(max_nfev) < 1:
        raise ValueError("max_nfev must be positive.")
    if not 0.0 <= acceptance_threshold < shrink_threshold < expansion_threshold < 1.0:
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

    x = np.asarray(jax.device_get(problem.x0), dtype=float)
    profile_mask = np.asarray(profile_mask, dtype=bool)
    if profile_mask.shape != x.shape:
        raise ValueError(
            "profile_mask must have one entry per optimizer coordinate."
        )
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

    if initial_evaluation is None:
        evaluation = problem.evaluate(jnp.asarray(x, dtype=jnp.float64))
    else:
        evaluation = initial_evaluation
    residuals, jacobian = host_evaluation(evaluation)
    cost = _cost(residuals)
    nfev = 1
    njev = 1
    nit = 0
    accepted_steps = 0
    rejected_steps = 0
    geometry_radius = float(geometry_initial_radius)
    profile_fraction_limit = float(profile_initial_fraction_limit)
    geometry_mask = ~profile_mask
    status = 0
    message = "The maximum number of function evaluations is exceeded."

    while nfev < int(max_nfev):
        gradient = jacobian.T @ residuals
        projected_gradient = _projected_gradient(
            x, gradient, lower, upper
        )
        optimality = float(np.linalg.norm(projected_gradient, ord=np.inf))
        if optimality <= float(gtol):
            status = 1
            message = "The projected gradient tolerance is satisfied."
            break

        step, step_metadata = _block_model_step(
            residuals,
            jacobian,
            x,
            lower,
            upper,
            profile_mask,
            geometry_radius=geometry_radius,
            profile_fraction_limit=profile_fraction_limit,
            proximal_weight=proximal_weight,
        )
        model_residuals = residuals + jacobian @ step
        predicted_reduction = cost - _cost(model_residuals)
        if not np.isfinite(predicted_reduction) or predicted_reduction <= 0.0:
            status = 4
            message = "The block model produced no positive predicted reduction."
            break

        trial_x = np.clip(x + step, lower, upper)
        trial_evaluation = None
        failure = None
        try:
            trial_evaluation = problem.evaluate(
                jnp.asarray(trial_x, dtype=jnp.float64)
            )
            trial_residuals, trial_jacobian = host_evaluation(trial_evaluation)
            trial_cost = _cost(trial_residuals)
        except Exception as exc:  # Failed physical trials contract the radii.
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
                "[NEOPAX block_trust_region] "
                f"eval={nfev} accepted={accepted} "
                f"cost={trial_cost:.8e} actual_reduction={actual_reduction:.8e} "
                f"predicted_reduction={predicted_reduction:.8e} "
                f"ratio={ratio:.8e} geometry_radius={geometry_radius:.6e} "
                f"profile_fraction_limit={profile_fraction_limit:.6e} "
                f"geometry_step_l2={step_metadata['geometry_step_norm']:.6e} "
                f"profile_step_max_abs={step_metadata['profile_step_max_abs']:.6e}"
                f"{details}",
                flush=True,
            )
            if failure is not None:
                print(
                    "[NEOPAX block_trust_region] trial failure: " + failure,
                    flush=True,
                )

        if ratio < float(shrink_threshold) or not np.isfinite(ratio):
            if np.any(geometry_mask):
                geometry_radius = max(
                    float(geometry_min_radius),
                    geometry_radius * float(shrink_factor),
                )
            if np.any(profile_mask):
                profile_fraction_limit = max(
                    float(profile_min_fraction_limit),
                    profile_fraction_limit * float(shrink_factor),
                )
        elif ratio > float(expansion_threshold):
            geometry_activity = (
                float(step_metadata["geometry_step_norm"]) / geometry_radius
                if np.any(geometry_mask)
                else 0.0
            )
            profile_activity = (
                float(step_metadata["profile_step_max_abs"])
                / profile_fraction_limit
                if np.any(profile_mask)
                else 0.0
            )
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

        if not accepted:
            rejected_steps += 1
            if (
                geometry_radius <= float(geometry_min_radius)
                and profile_fraction_limit
                <= float(profile_min_fraction_limit)
            ):
                status = 5
                message = "Both block trust limits reached their minima."
                break
            continue

        previous_cost = cost
        x = trial_x
        evaluation = trial_evaluation
        residuals = trial_residuals
        jacobian = trial_jacobian
        cost = trial_cost
        accepted_steps += 1
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
    return BlockTrustRegionResult(
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
        success=bool(status in (1, 2, 3, 4)),
        geometry_radius=float(geometry_radius),
        profile_fraction_limit=float(profile_fraction_limit),
        accepted_steps=int(accepted_steps),
        rejected_steps=int(rejected_steps),
        evaluation=evaluation,
    )


__all__ = ["BlockTrustRegionResult", "block_trust_region_least_squares"]
