"""Optimization helpers built on NEOPAX reverse-AD internals."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Sequence
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from ._geometry_autodiff import (
    _input_with_boundary_deltas,
    boundary_param_entries,
    build_geometry_autodiff_context,
)
from ._reverse_ad_optimization import (
    LeastSquaresEvaluation,
    LeastSquaresResult,
    LeastSquaresTerm,
    ObjectiveTableResult,
    ObjectiveRef,
    geometry,
    geometry_full_ad_reverse_table,
    scale_least_squares_evaluation_columns,
)
from ._reverse_ad_parameters import (
    VmexBoundaryParameterization,
    parse_vmec_boundary_parameter_specs,
    reverse_ad_optimization_parameter_set,
    vmex_boundary_parameterization,
)


@dataclasses.dataclass(frozen=True, slots=True)
class GeometryObjectiveTransform:
    """Script-defined scalar geometry objective backed by one AD-table row."""

    base: ObjectiveRef
    value_fn: Callable[[object], object]
    derivative_fn: Callable[[object], object]
    label: str

    @property
    def objective(self) -> ObjectiveRef:
        return self.base


@dataclasses.dataclass(frozen=True, slots=True)
class GeometryLeastSquaresTerm:
    """Geometry least-squares term with an optional scalar chain-rule transform."""

    objective: ObjectiveRef
    target: float
    weight: float
    label: str | None = None
    value_fn: Callable[[object], object] | None = None
    derivative_fn: Callable[[object], object] | None = None

    @property
    def residual_label(self) -> str:
        return self.label or self.objective.label


@dataclasses.dataclass(frozen=True, slots=True)
class GeometryLeastSquaresProblem:
    """Geometry-only least-squares problem using the validated geometry AD table."""

    context: object
    parameterization: VmexBoundaryParameterization
    terms: tuple[GeometryLeastSquaresTerm, ...]
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
        objective_names = tuple(dict.fromkeys(term.objective.name for term in self.terms))
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
        result = _assemble_geometry_least_squares_result(
            self.terms,
            parameter_set=parameter_set,
            table=table,
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

    def input_from_scaled_parameters(self, scaled_parameter_values=None):
        """Return a VMEX input object with the current scaled boundary deltas applied."""

        if scaled_parameter_values is None:
            scaled_values = self.x0
        else:
            scaled_values = jnp.asarray(scaled_parameter_values, dtype=jnp.float64)
        physical_values = self.parameterization.scaled_to_physical_delta(scaled_values)
        entries = boundary_param_entries(self.context, self.parameterization.vmec_tuples)
        return _input_with_boundary_deltas(self.context, physical_values, entries)


def geometry_objective(name: str | ObjectiveRef) -> ObjectiveRef:
    """Return a named geometry objective reference."""

    if isinstance(name, ObjectiveRef):
        if name.family != "geometry":
            raise ValueError(f"Expected a geometry ObjectiveRef, got {name.family!r}.")
        return name
    return ObjectiveRef("geometry", str(name))


def transformed_geometry_objective(
    base: str | ObjectiveRef,
    value_fn: Callable[[object], object],
    derivative_fn: Callable[[object], object] | None = None,
    *,
    label: str,
) -> GeometryObjectiveTransform:
    """Build a scalar transformed geometry objective from one table row."""

    if derivative_fn is None:
        derivative_fn = jax.grad(lambda x: jnp.asarray(value_fn(x), dtype=jnp.float64))
    return GeometryObjectiveTransform(
        base=geometry_objective(base),
        value_fn=value_fn,
        derivative_fn=derivative_fn,
        label=str(label),
    )


def _normalize_geometry_least_squares_terms(
    terms: Sequence[
        GeometryLeastSquaresTerm
        | LeastSquaresTerm
        | tuple[ObjectiveRef | GeometryObjectiveTransform | str, float, float]
    ],
) -> tuple[GeometryLeastSquaresTerm, ...]:
    normalized: list[GeometryLeastSquaresTerm] = []
    for term in terms:
        if isinstance(term, GeometryLeastSquaresTerm):
            normalized.append(term)
            continue
        if isinstance(term, LeastSquaresTerm):
            normalized.append(
                GeometryLeastSquaresTerm(
                    objective=geometry_objective(term.objective),
                    target=float(term.target),
                    weight=float(term.weight),
                    label=term.label,
                )
            )
            continue
        if not isinstance(term, tuple) or len(term) != 3:
            raise TypeError(
                "Geometry terms must be GeometryLeastSquaresTerm instances, "
                "LeastSquaresTerm instances, or (objective, target, weight) tuples."
            )
        objective, target, weight = term
        if isinstance(objective, GeometryObjectiveTransform):
            normalized.append(
                GeometryLeastSquaresTerm(
                    objective=objective.base,
                    target=float(target),
                    weight=float(weight),
                    label=objective.label,
                    value_fn=objective.value_fn,
                    derivative_fn=objective.derivative_fn,
                )
            )
        else:
            normalized.append(
                GeometryLeastSquaresTerm(
                    objective=geometry_objective(objective),
                    target=float(target),
                    weight=float(weight),
                )
            )
    if not normalized:
        raise ValueError("At least one geometry least-squares term is required.")
    for term in normalized:
        if term.weight < 0.0:
            raise ValueError(f"Geometry least-squares weights must be non-negative; got {term.weight}.")
    return tuple(normalized)


def _result_lookup(result: ObjectiveTableResult) -> dict[str, int]:
    return {name: index for index, name in enumerate(result.objective_names)}


def _assemble_geometry_least_squares_result(
    terms: Sequence[GeometryLeastSquaresTerm],
    *,
    parameter_set,
    table: ObjectiveTableResult,
) -> LeastSquaresResult:
    lookup = _result_lookup(table)
    values = jnp.asarray(table.values)
    jacobian_table = jnp.asarray(table.jacobian)
    residual_rows = []
    jacobian_rows = []
    residual_labels = []
    objective_values: dict[str, object] = {}
    label_counts: dict[str, int] = {}
    for term in terms:
        row_index = lookup[term.objective.name]
        base_value = values[row_index]
        base_jacobian = jacobian_table[row_index]
        if term.value_fn is None:
            value = base_value
            chain = jnp.asarray(1.0, dtype=base_value.dtype)
        else:
            value = term.value_fn(base_value)
            chain = term.derivative_fn(base_value)
        scale = jnp.asarray(np.sqrt(float(term.weight)), dtype=base_value.dtype)
        residual_rows.append(scale * (value - jnp.asarray(term.target, dtype=base_value.dtype)))
        jacobian_rows.append(scale * chain * base_jacobian)
        base_label = term.residual_label
        label_count = label_counts.get(base_label, 0)
        label_counts[base_label] = label_count + 1
        residual_label = base_label if label_count == 0 else f"{base_label}#{label_count + 1}"
        residual_labels.append(residual_label)
        objective_values[residual_label] = value
    return LeastSquaresResult(
        residuals=jnp.stack(residual_rows),
        jacobian=jnp.stack(jacobian_rows),
        residual_labels=tuple(residual_labels),
        parameter_labels=parameter_set.vmec_prefixed_labels,
        objective_values=objective_values,
    )


def geometry_least_squares_problem(
    vmec_input,
    terms: Sequence[
        GeometryLeastSquaresTerm
        | LeastSquaresTerm
        | tuple[ObjectiveRef | GeometryObjectiveTransform | str, float, float]
    ],
    *,
    max_mode: int | None = None,
    parameters: str | Sequence[str] | None = None,
    families: str | Sequence[str] | None = "RBC,ZBS",
    scale_mode: str = "ess",
    ess_alpha: float = 1.0,
    mboz: int = 18,
    nboz: int = 18,
    surfaces: Sequence[float] = (0.1, 0.28, 0.46, 0.64, 0.82, 1.0),
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
        surface_s=tuple(float(s) for s in surfaces),
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
    normalized_terms = _normalize_geometry_least_squares_terms(terms)
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
    state: dict[str, object] = {"nres": None, "npar": problem.parameter_count}

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
        try:
            residuals = np.asarray(jax.device_get(_evaluate(x).residuals), dtype=float)
        except Exception as exc:
            if state["nres"] is None:
                raise
            if int(kwargs.get("verbose", 0) or 0):
                print(f"[NEOPAX least_squares] trial solve failed: {exc}")
            return np.full((int(state["nres"]),), 1.0e6, dtype=float)
        residuals = np.where(np.isfinite(residuals), residuals, 1.0e6)
        state["nres"] = int(residuals.size)
        return residuals

    def _jac(x):
        try:
            jacobian = np.asarray(jax.device_get(_evaluate(x).jacobian), dtype=float)
        except Exception as exc:
            if state["nres"] is None:
                raise
            if int(kwargs.get("verbose", 0) or 0):
                print(f"[NEOPAX least_squares] trial jacobian failed: {exc}")
            return np.zeros((int(state["nres"]), int(state["npar"])), dtype=float)
        return np.where(np.isfinite(jacobian), jacobian, 0.0)

    x0 = np.asarray(jax.device_get(problem.x0), dtype=float)
    x_scale = np.asarray(jax.device_get(problem.x_scale), dtype=float)
    kwargs.setdefault("x_scale", x_scale)
    return scipy_least_squares(_fun, x0, jac=_jac, **kwargs)


__all__ = [
    "GeometryLeastSquaresTerm",
    "GeometryObjectiveTransform",
    "GeometryLeastSquaresProblem",
    "geometry",
    "geometry_objective",
    "geometry_least_squares_problem",
    "least_squares",
    "transformed_geometry_objective",
]
