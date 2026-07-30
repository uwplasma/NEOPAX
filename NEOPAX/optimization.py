"""Optimization helpers built on NEOPAX reverse-AD internals."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ._geometry_autodiff import build_geometry_autodiff_context
from ._reverse_ad_optimization import (
    LeastSquaresEvaluation,
    LeastSquaresTerm,
    ObjectiveRef,
    assemble_least_squares_result,
    geometry,
    geometry_full_ad_reverse_table,
    normalize_least_squares_terms,
    scale_least_squares_evaluation_columns,
)
from ._reverse_ad_parameters import (
    VmexBoundaryParameterization,
    parse_vmec_boundary_parameter_specs,
    reverse_ad_optimization_parameter_set,
    vmex_boundary_parameterization,
)


@dataclasses.dataclass(frozen=True, slots=True)
class GeometryLeastSquaresProblem:
    """Geometry-only least-squares problem using the validated geometry AD table."""

    context: object
    parameterization: VmexBoundaryParameterization
    terms: tuple[LeastSquaresTerm, ...]
    lane: str = "ad"
    max_iter: int | None = None
    step_size: float | None = None
    solver_device: str | None = "default"

    @property
    def parameter_count(self) -> int:
        return len(self.parameterization.specs)

    @property
    def parameter_labels(self) -> tuple[str, ...]:
        return self.parameterization.labels

    @property
    def x0(self):
        return jnp.zeros((self.parameter_count,), dtype=jnp.float64)

    @property
    def x_scale(self):
        return self.parameterization.x_scale

    def evaluate(self, scaled_parameter_values=None) -> LeastSquaresEvaluation:
        """Evaluate residuals/Jacobian with respect to scaled optimizer variables."""

        if scaled_parameter_values is None:
            scaled_values = self.x0
        else:
            scaled_values = jnp.asarray(scaled_parameter_values, dtype=jnp.float64)
        if tuple(scaled_values.shape) != (self.parameter_count,):
            raise ValueError(
                "scaled_parameter_values must have shape "
                f"({self.parameter_count},); got {tuple(scaled_values.shape)}."
            )
        physical_values = self.parameterization.scaled_to_physical_delta(scaled_values)
        parameter_set = reverse_ad_optimization_parameter_set(
            include_profiles=False,
            vmec_boundary=self.parameterization.specs,
        )
        objective_names = tuple(
            dict.fromkeys(term.objective.name for term in self.terms)
        )
        t_start = time.perf_counter()
        table = geometry_full_ad_reverse_table(
            context=self.context,
            parameter_set=parameter_set,
            objective_names=objective_names,
            parameter_values=physical_values,
            lane=self.lane,
            max_iter=self.max_iter,
            step_size=self.step_size,
            final_vmec_pullback_mode="raw_block_transpose",
            solver_device=self.solver_device,
        )
        result = assemble_least_squares_result(
            self.terms,
            parameter_set=parameter_set,
            backend_results={"geometry": table},
        )
        residuals = jax.block_until_ready(result.residuals)
        jacobian = jax.block_until_ready(result.jacobian)
        evaluation = LeastSquaresEvaluation(
            result=result,
            residuals=residuals,
            jacobian=jacobian,
            elapsed_s=float(time.perf_counter() - t_start),
        )
        return scale_least_squares_evaluation_columns(evaluation, self.x_scale)

    def residuals(self, scaled_parameter_values=None) -> np.ndarray:
        return np.asarray(jax.device_get(self.evaluate(scaled_parameter_values).residuals), dtype=float)

    def jacobian(self, scaled_parameter_values=None) -> np.ndarray:
        return np.asarray(jax.device_get(self.evaluate(scaled_parameter_values).jacobian), dtype=float)


def geometry_least_squares_problem(
    vmec_input,
    terms: Sequence[LeastSquaresTerm | tuple[ObjectiveRef | str, float, float]],
    *,
    max_mode: int | None = None,
    parameters: str | Sequence[str] | None = None,
    families: str | Sequence[str] | None = "RBC,ZBS",
    scale_mode: str = "ess",
    ess_alpha: float = 1.0,
    mboz: int = 18,
    nboz: int = 18,
    lane: str = "ad",
    max_iter: int | None = None,
    step_size: float | None = None,
    solver_device: str | None = "default",
) -> GeometryLeastSquaresProblem:
    """Build a VMEX-style geometry least-squares problem.

    Use ``max_mode`` for VMEX packed boundary DOFs, or pass explicit
    ``parameters`` such as ``"RBC:1:0,ZBS:1:0"`` for diagnostic runs.
    """

    context = build_geometry_autodiff_context(
        vmec_input,
        param_family="RBC",
        param_m=1,
        param_n=0,
        mboz=int(mboz),
        nboz=int(nboz),
    )
    if parameters is not None:
        specs = parse_vmec_boundary_parameter_specs(
            ",".join(parameters) if not isinstance(parameters, str) else parameters
        )
        parameterization = VmexBoundaryParameterization(
            specs=specs,
            scales=tuple(1.0 for _ in specs),
            scale_mode="unit",
        )
    else:
        if max_mode is None:
            raise ValueError("Either max_mode or explicit parameters must be provided.")
        parameterization = vmex_boundary_parameterization(
            context,
            max_mode=int(max_mode),
            families=families,
            scale_mode=scale_mode,
            ess_alpha=float(ess_alpha),
        )
    normalized_terms = normalize_least_squares_terms(terms, default_family="geometry")
    return GeometryLeastSquaresProblem(
        context=context,
        parameterization=parameterization,
        terms=normalized_terms,
        lane=lane,
        max_iter=max_iter,
        step_size=step_size,
        solver_device=solver_device,
    )


def least_squares(problem: GeometryLeastSquaresProblem, **kwargs):
    """Run SciPy least_squares on a NEOPAX geometry least-squares problem."""

    from scipy.optimize import least_squares as scipy_least_squares

    cache: dict[tuple[float, ...], LeastSquaresEvaluation] = {}

    def _key(x):
        return tuple(np.asarray(x, dtype=float).tolist())

    def _evaluate(x):
        key = _key(x)
        evaluation = cache.get(key)
        if evaluation is None:
            evaluation = problem.evaluate(jnp.asarray(x, dtype=jnp.float64))
            cache.clear()
            cache[key] = evaluation
        return evaluation

    def _fun(x):
        return np.asarray(jax.device_get(_evaluate(x).residuals), dtype=float)

    def _jac(x):
        return np.asarray(jax.device_get(_evaluate(x).jacobian), dtype=float)

    x0 = np.asarray(jax.device_get(problem.x0), dtype=float)
    x_scale = np.asarray(jax.device_get(problem.x_scale), dtype=float)
    kwargs.setdefault("x_scale", x_scale)
    return scipy_least_squares(_fun, x0, jac=_jac, **kwargs)


__all__ = [
    "GeometryLeastSquaresProblem",
    "geometry",
    "geometry_least_squares_problem",
    "least_squares",
]
