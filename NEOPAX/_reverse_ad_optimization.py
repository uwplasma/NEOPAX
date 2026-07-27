"""Optimization-facing reverse-AD objective-term API.

This module defines the least-squares shell for the production reverse-AD lane.
It intentionally does not own solver math yet.  Backends are provided as
callables so benchmark-validated transport and geometry table implementations
can be wired in later without changing the user-facing term interface.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import jax
import jax.numpy as jnp

from ._reverse_ad_parameters import (
    ProfileParameterSpec,
    ReverseADParameterSet,
    VmecBoundaryParameterSpec,
    parameter_labels,
)
from ._reverse_ad_transport import (
    RealtimeGeometryTransportReverseTableContext,
    RealtimeGeometryTransportReverseTableRequest,
    RealtimeGeometryTransportReverseTableResult,
    TransportReverseReportBuilder,
    TransportReverseReportRunner,
    TransportReverseTableResultBuilder,
    normalize_transport_objective_names,
    realtime_geometry_transport_reverse_table_request,
    transport_realtime_geometry_reverse_table,
)


ObjectiveFamily = Literal["transport", "geometry", "regularization"]


@dataclasses.dataclass(frozen=True, slots=True)
class ObjectiveRef:
    """Reference to a named objective in one reverse-AD backend family."""

    family: ObjectiveFamily
    name: str

    def __post_init__(self) -> None:
        family = str(self.family).strip().lower()
        if family not in {"transport", "geometry", "regularization"}:
            raise ValueError(
                "Objective family must be one of: transport, geometry, regularization; "
                f"got {self.family!r}."
            )
        name = str(self.name).strip()
        if not name:
            raise ValueError("Objective name must be non-empty.")
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "name", name)

    @property
    def label(self) -> str:
        return f"{self.family}:{self.name}"


@dataclasses.dataclass(frozen=True, slots=True)
class LeastSquaresTerm:
    """One VMEX-style least-squares term: `(objective, target, weight)`."""

    objective: ObjectiveRef
    target: float = 0.0
    weight: float = 1.0
    label: str | None = None

    def __post_init__(self) -> None:
        weight = float(self.weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"Least-squares term weight must be finite and non-negative; got {self.weight!r}.")
        target = float(self.target)
        if not math.isfinite(target):
            raise ValueError(f"Least-squares term target must be finite; got {self.target!r}.")
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "weight", weight)

    @property
    def residual_label(self) -> str:
        return self.label or self.objective.label


@dataclasses.dataclass(frozen=True, slots=True)
class ObjectiveTableResult:
    """Objective values and Jacobian rows for one backend group."""

    objective_names: tuple[str, ...]
    values: object
    jacobian: object

    def __post_init__(self) -> None:
        names = tuple(str(name).strip() for name in self.objective_names)
        if not names:
            raise ValueError("ObjectiveTableResult requires at least one objective name.")
        if any(not name for name in names):
            raise ValueError("ObjectiveTableResult objective names must be non-empty.")
        object.__setattr__(self, "objective_names", names)


@dataclasses.dataclass(frozen=True, slots=True)
class LeastSquaresResult:
    """Assembled least-squares residuals and Jacobian."""

    residuals: object
    jacobian: object
    residual_labels: tuple[str, ...]
    parameter_labels: tuple[str, ...]
    objective_values: dict[str, object]


@dataclasses.dataclass(frozen=True, slots=True)
class LeastSquaresEvaluation:
    """Device-ready least-squares result plus timing for optimization callers."""

    result: LeastSquaresResult
    residuals: object
    jacobian: object
    elapsed_s: float


ObjectiveTableBackend = Callable[
    [tuple[str, ...], ReverseADParameterSet, Mapping[str, object]],
    ObjectiveTableResult,
]
TransportRealtimeGeometryLeastSquaresRunner = Callable[
    [Sequence[LeastSquaresTerm | tuple[ObjectiveRef | str, float, float]]],
    LeastSquaresEvaluation,
]
def transport_objective(name: str) -> ObjectiveRef:
    return ObjectiveRef("transport", name)


def geometry_objective(name: str) -> ObjectiveRef:
    return ObjectiveRef("geometry", name)


def regularization_objective(name: str) -> ObjectiveRef:
    return ObjectiveRef("regularization", name)


@dataclasses.dataclass(frozen=True, slots=True)
class ObjectiveNamespace:
    """Attribute/call helper for VMEX-like objective references."""

    family: ObjectiveFamily

    def __getattr__(self, name: str) -> ObjectiveRef:
        return ObjectiveRef(self.family, name)

    def __call__(self, name: str) -> ObjectiveRef:
        return ObjectiveRef(self.family, name)


transport = ObjectiveNamespace("transport")
geometry = ObjectiveNamespace("geometry")
regularization = ObjectiveNamespace("regularization")


def least_squares_term(
    objective: ObjectiveRef | str,
    target: float = 0.0,
    weight: float = 1.0,
    *,
    family: ObjectiveFamily = "transport",
    label: str | None = None,
) -> LeastSquaresTerm:
    """Build a least-squares term from an ObjectiveRef or objective name."""

    objective_ref = objective if isinstance(objective, ObjectiveRef) else ObjectiveRef(family, str(objective))
    return LeastSquaresTerm(objective_ref, target=target, weight=weight, label=label)


def transport_least_squares_terms(
    objective_names: Sequence[str],
    *,
    target: float = 0.0,
    weight: float = 1.0,
) -> tuple[LeastSquaresTerm, ...]:
    """Build uniform transport least-squares terms for a list of objectives."""

    names = normalize_transport_objective_names(objective_names)
    return tuple(
        LeastSquaresTerm(transport(objective_name), target=target, weight=weight)
        for objective_name in names
    )


def normalize_least_squares_terms(
    terms: Sequence[LeastSquaresTerm | tuple[ObjectiveRef | str, float, float]],
    *,
    default_family: ObjectiveFamily = "transport",
) -> tuple[LeastSquaresTerm, ...]:
    """Normalize VMEX-style `(objective, target, weight)` entries."""

    normalized: list[LeastSquaresTerm] = []
    for term in terms:
        if isinstance(term, LeastSquaresTerm):
            normalized.append(term)
            continue
        if not isinstance(term, tuple) or len(term) != 3:
            raise TypeError(
                "Least-squares terms must be LeastSquaresTerm instances or "
                "(objective, target, weight) tuples."
            )
        objective, target, weight = term
        normalized.append(
            least_squares_term(
                objective,
                target=float(target),
                weight=float(weight),
                family=default_family,
            )
        )
    if not normalized:
        raise ValueError("At least one least-squares term is required.")
    return tuple(normalized)


def group_least_squares_terms_by_family(
    terms: Sequence[LeastSquaresTerm],
) -> dict[ObjectiveFamily, tuple[LeastSquaresTerm, ...]]:
    """Group terms by backend family while preserving within-family order."""

    grouped: dict[ObjectiveFamily, list[LeastSquaresTerm]] = {
        "transport": [],
        "geometry": [],
        "regularization": [],
    }
    for term in terms:
        grouped[term.objective.family].append(term)
    return {family: tuple(values) for family, values in grouped.items() if values}


def _unique_objective_names(terms: Sequence[LeastSquaresTerm]) -> tuple[str, ...]:
    """Return objective names once, preserving first-use order."""

    names: list[str] = []
    seen: set[str] = set()
    for term in terms:
        name = term.objective.name
        if name in seen:
            continue
        names.append(name)
        seen.add(name)
    return tuple(names)


def _result_lookup(result: ObjectiveTableResult) -> dict[str, int]:
    if len(set(result.objective_names)) != len(result.objective_names):
        raise ValueError(
            "ObjectiveTableResult objective_names must be unique; got duplicates in "
            f"{result.objective_names!r}."
        )
    return {name: index for index, name in enumerate(result.objective_names)}


def assemble_least_squares_result(
    terms: Sequence[LeastSquaresTerm],
    *,
    parameter_set: ReverseADParameterSet,
    backend_results: Mapping[ObjectiveFamily, ObjectiveTableResult],
    vmec_prefix: bool = True,
) -> LeastSquaresResult:
    """Assemble least-squares residuals/Jacobian from grouped table results."""

    expected_parameter_count = len(parameter_set.specs)
    residual_rows = []
    jacobian_rows = []
    residual_labels = []
    objective_values: dict[str, object] = {}
    residual_label_counts: dict[str, int] = {}
    for term in terms:
        result = backend_results.get(term.objective.family)
        if result is None:
            raise ValueError(f"No backend result was provided for objective family {term.objective.family!r}.")
        lookup = _result_lookup(result)
        if term.objective.name not in lookup:
            available = ", ".join(result.objective_names)
            raise ValueError(
                f"Objective {term.objective.name!r} was requested from family "
                f"{term.objective.family!r}, but backend returned: {available}."
            )
        row_index = lookup[term.objective.name]
        values = jnp.asarray(result.values)
        jacobian_table = jnp.asarray(result.jacobian)
        if values.ndim != 1 or int(values.shape[0]) != len(result.objective_names):
            raise ValueError(
                "ObjectiveTableResult values must have shape (objective_count,); "
                f"got values.shape={values.shape}, objective_count={len(result.objective_names)}."
            )
        if jacobian_table.ndim != 2 or int(jacobian_table.shape[0]) != len(result.objective_names):
            raise ValueError(
                "ObjectiveTableResult jacobian must have shape (objective_count, parameter_count); "
                f"got jacobian.shape={jacobian_table.shape}, objective_count={len(result.objective_names)}."
            )
        if int(jacobian_table.shape[1]) != expected_parameter_count:
            raise ValueError(
                "ObjectiveTableResult jacobian parameter dimension does not match parameter set: "
                f"jacobian.shape={jacobian_table.shape}, parameter_count={expected_parameter_count}."
            )
        value = values[row_index]
        jacobian_row = jacobian_table[row_index]
        scale = jnp.asarray(math.sqrt(float(term.weight)), dtype=value.dtype)
        residual_rows.append(scale * (value - jnp.asarray(term.target, dtype=value.dtype)))
        jacobian_rows.append(scale * jacobian_row)
        base_label = term.residual_label
        label_count = residual_label_counts.get(base_label, 0)
        residual_label_counts[base_label] = label_count + 1
        residual_label = base_label if label_count == 0 else f"{base_label}#{label_count + 1}"
        residual_labels.append(residual_label)
        objective_values[residual_label] = value

    residuals = jnp.stack(residual_rows)
    jacobian = jnp.stack(jacobian_rows)
    return LeastSquaresResult(
        residuals=residuals,
        jacobian=jacobian,
        residual_labels=tuple(residual_labels),
        parameter_labels=parameter_labels(parameter_set.specs, vmec_prefix=vmec_prefix),
        objective_values=objective_values,
    )


def residuals_and_jacobian_reverse_ad(
    config: Mapping[str, object],
    *,
    parameter_set: ReverseADParameterSet,
    terms: Sequence[LeastSquaresTerm | tuple[ObjectiveRef | str, float, float]],
    backends: Mapping[ObjectiveFamily, ObjectiveTableBackend],
    options: Mapping[str, object] | None = None,
) -> LeastSquaresResult:
    """Evaluate least-squares residuals/Jacobian through grouped reverse-AD backends.

    This is the API shell.  The caller supplies backend table functions for now;
    later implementation steps will provide the validated transport and geometry
    backends from NEOPAX internals.
    """

    del config  # The shell keeps the signature stable; backend wiring comes next.
    normalized_terms = normalize_least_squares_terms(terms)
    grouped_terms = group_least_squares_terms_by_family(normalized_terms)
    backend_options = {} if options is None else dict(options)
    backend_results: dict[ObjectiveFamily, ObjectiveTableResult] = {}
    for family, family_terms in grouped_terms.items():
        backend = backends.get(family)
        if backend is None:
            raise NotImplementedError(
                f"No reverse-AD backend is registered for objective family {family!r}."
            )
        objective_names = _unique_objective_names(family_terms)
        backend_results[family] = backend(objective_names, parameter_set, backend_options)
    return assemble_least_squares_result(
        normalized_terms,
        parameter_set=parameter_set,
        backend_results=backend_results,
    )


def evaluate_transport_reverse_table_least_squares(
    config: Mapping[str, object],
    *,
    parameter_set: ReverseADParameterSet,
    terms: Sequence[LeastSquaresTerm | tuple[ObjectiveRef | str, float, float]],
    table_result_builder: TransportReverseTableResultBuilder,
    options: Mapping[str, object] | None = None,
) -> LeastSquaresEvaluation:
    """Evaluate transport least squares from the validated grouped table builder.

    The builder owns the expensive reverse pass.  This helper keeps the
    optimization path on the JAX-native table-result route and avoids the
    report/dictionary adapter unless an outer benchmark explicitly asks for
    reporting.
    """

    backend = transport_reverse_table_result_builder_backend(table_result_builder)
    t_start = time.perf_counter()
    result = residuals_and_jacobian_reverse_ad(
        config,
        parameter_set=parameter_set,
        terms=terms,
        backends={"transport": backend},
        options=options,
    )
    residuals = jax.block_until_ready(result.residuals)
    jacobian = jax.block_until_ready(result.jacobian)
    elapsed_s = time.perf_counter() - t_start
    return LeastSquaresEvaluation(
        result=result,
        residuals=residuals,
        jacobian=jacobian,
        elapsed_s=float(elapsed_s),
    )


def evaluate_transport_realtime_geometry_least_squares(
    config: Mapping[str, object],
    *,
    request: RealtimeGeometryTransportReverseTableRequest,
    terms: Sequence[LeastSquaresTerm | tuple[ObjectiveRef | str, float, float]],
    table_result_builder: TransportReverseTableResultBuilder | None = None,
    run_grouped_report=None,
    objective_labels: Sequence[str] | None = None,
    options: Mapping[str, object] | None = None,
    quiet_default: bool = True,
) -> LeastSquaresEvaluation:
    """Evaluate transport least squares through the realtime-geometry table API."""

    normalized_terms = normalize_least_squares_terms(terms)
    grouped_terms = group_least_squares_terms_by_family(normalized_terms)
    unsupported_families = tuple(family for family in grouped_terms if family != "transport")
    if unsupported_families:
        raise NotImplementedError(
            "evaluate_transport_realtime_geometry_least_squares only supports transport "
            f"terms; got families {unsupported_families!r}."
        )
    requested_term_objectives = _unique_objective_names(grouped_terms.get("transport", ()))
    if requested_term_objectives != request.objective_names:
        raise ValueError(
            "Realtime-geometry transport request objectives must match the transport "
            "least-squares terms in first-use order: "
            f"request={request.objective_names!r}, terms={requested_term_objectives!r}."
        )
    t_start = time.perf_counter()
    table_result = transport_realtime_geometry_reverse_table(
        request=request,
        table_result_builder=table_result_builder,
        run_grouped_report=run_grouped_report,
        objective_labels=objective_labels,
        options=options,
        quiet_default=quiet_default,
    )
    result = residuals_and_jacobian_reverse_ad(
        config,
        parameter_set=request.parameter_set,
        terms=normalized_terms,
        backends={"transport": transport_reverse_table_backend(table_result)},
        options=options,
    )
    residuals = jax.block_until_ready(result.residuals)
    jacobian = jax.block_until_ready(result.jacobian)
    elapsed_s = time.perf_counter() - t_start
    return LeastSquaresEvaluation(
        result=result,
        residuals=residuals,
        jacobian=jacobian,
        elapsed_s=float(elapsed_s),
    )


def build_transport_realtime_geometry_least_squares_runner(
    config: Mapping[str, object],
    *,
    objective_names: Sequence[str],
    parameter_set: ReverseADParameterSet,
    table_context: RealtimeGeometryTransportReverseTableContext,
    table_result_builder: TransportReverseTableResultBuilder | None = None,
    run_grouped_report: TransportReverseReportRunner | None = None,
    objective_labels: Sequence[str] | None = None,
    options: Mapping[str, object] | None = None,
    quiet_default: bool = True,
) -> TransportRealtimeGeometryLeastSquaresRunner:
    """Build a callable least-squares runner for realtime-geometry transport objectives.

    This is the optimizer-facing wiring helper.  The actual heavy reverse
    executor can still be supplied by a benchmark during migration, but request
    construction and canonical evaluation live here.
    """

    request = realtime_geometry_transport_reverse_table_request(
        objective_names=objective_names,
        parameter_set=parameter_set,
        context=table_context,
        options=options,
    )

    def _runner(
        terms: Sequence[LeastSquaresTerm | tuple[ObjectiveRef | str, float, float]],
    ) -> LeastSquaresEvaluation:
        return evaluate_transport_realtime_geometry_least_squares(
            config,
            request=request,
            terms=terms,
            table_result_builder=table_result_builder,
            run_grouped_report=run_grouped_report,
            objective_labels=objective_labels,
            options=options,
            quiet_default=quiet_default,
        )

    return _runner


def transport_reverse_report_to_objective_table_result(
    report: Mapping[str, object],
    objective_names: Sequence[str],
    parameter_set: ReverseADParameterSet,
) -> ObjectiveTableResult:
    """Adapt a validated transport reverse benchmark report into a table result.

    This is a zero-math bridge: it reads the grouped reverse tables already
    produced by the validated realtime-geometry benchmark path and rearranges
    rows/columns into the optimization parameter order.
    """

    requested_objectives = normalize_transport_objective_names(objective_names)

    try:
        objective_values = report["objective_values"]
        profile_gradients = report["profile_gradient_reverse_ad"]
    except KeyError as exc:
        raise ValueError(
            "Transport reverse report must contain objective_values and "
            "profile_gradient_reverse_ad entries."
        ) from exc

    if not isinstance(objective_values, Mapping):
        raise TypeError("report['objective_values'] must be a mapping by objective name.")
    if not isinstance(profile_gradients, Mapping):
        raise TypeError("report['profile_gradient_reverse_ad'] must be a mapping by objective name.")

    geometry_gradients = report.get("geometry_gradient_reverse_ad", {})
    if geometry_gradients is None:
        geometry_gradients = {}
    if not isinstance(geometry_gradients, Mapping):
        raise TypeError("report['geometry_gradient_reverse_ad'] must be a mapping by objective name.")

    values = []
    jacobian_rows = []
    for objective_name in requested_objectives:
        if objective_name not in objective_values:
            available = ", ".join(str(name) for name in objective_values)
            raise ValueError(
                f"Objective {objective_name!r} is missing from transport reverse report values. "
                f"Available objectives: {available}."
            )
        if objective_name not in profile_gradients:
            available = ", ".join(str(name) for name in profile_gradients)
            raise ValueError(
                f"Objective {objective_name!r} is missing from profile gradient report. "
                f"Available objectives: {available}."
            )

        profile_row = profile_gradients[objective_name]
        geometry_row = geometry_gradients.get(objective_name, {})
        if not isinstance(profile_row, Mapping):
            raise TypeError(
                f"report['profile_gradient_reverse_ad'][{objective_name!r}] must be a mapping."
            )
        if not isinstance(geometry_row, Mapping):
            raise TypeError(
                f"report['geometry_gradient_reverse_ad'][{objective_name!r}] must be a mapping."
            )

        values.append(float(objective_values[objective_name]))
        row = []
        for spec in parameter_set.specs:
            if isinstance(spec, ProfileParameterSpec):
                if spec.name not in profile_row:
                    available = ", ".join(str(name) for name in profile_row)
                    raise ValueError(
                        f"Profile parameter {spec.name!r} is missing from objective "
                        f"{objective_name!r} gradient row. Available parameters: {available}."
                    )
                row.append(float(profile_row[spec.name]))
                continue
            if isinstance(spec, VmecBoundaryParameterSpec):
                if spec.vmec_label in geometry_row:
                    row.append(float(geometry_row[spec.vmec_label]))
                elif spec.label in geometry_row:
                    row.append(float(geometry_row[spec.label]))
                else:
                    available = ", ".join(str(name) for name in geometry_row)
                    raise ValueError(
                        f"VMEC parameter {spec.vmec_label!r} is missing from objective "
                        f"{objective_name!r} geometry gradient row. Available parameters: {available}."
                    )
                continue
            raise TypeError(f"Unsupported reverse-AD parameter spec type: {type(spec).__name__}.")
        jacobian_rows.append(jnp.asarray(row))

    return ObjectiveTableResult(
        objective_names=requested_objectives,
        values=jnp.asarray(values),
        jacobian=jnp.stack(jacobian_rows, axis=0),
    )


def transport_reverse_table_result_to_objective_table_result(
    table_result: RealtimeGeometryTransportReverseTableResult,
    objective_names: Sequence[str],
    parameter_set: ReverseADParameterSet,
) -> ObjectiveTableResult:
    """Adapt a JAX-native transport reverse table into an optimization table.

    Unlike the report adapter, this keeps objective values and gradient blocks
    as JAX arrays and never goes through JSON/report dictionaries.
    """

    if not isinstance(table_result, RealtimeGeometryTransportReverseTableResult):
        raise TypeError(
            "table_result must be a RealtimeGeometryTransportReverseTableResult; "
            f"got {type(table_result).__name__}."
        )
    requested_objectives = normalize_transport_objective_names(objective_names)
    objective_lookup = {
        objective_name: objective_i
        for objective_i, objective_name in enumerate(table_result.objective_labels)
    }
    profile_lookup = {
        parameter_name: parameter_i
        for parameter_i, parameter_name in enumerate(table_result.profile_parameter_labels)
    }
    geometry_lookup = {
        parameter_name: parameter_i
        for parameter_i, parameter_name in enumerate(table_result.geometry_parameter_labels)
    }

    objective_values = jnp.asarray(table_result.objective_values)
    profile_gradient_matrix = jnp.asarray(table_result.profile_gradient_matrix)
    geometry_gradient_matrix = jnp.asarray(table_result.geometry_gradient_matrix)
    values = []
    jacobian_rows = []
    for objective_name in requested_objectives:
        if objective_name not in objective_lookup:
            available = ", ".join(table_result.objective_labels)
            raise ValueError(
                f"Objective {objective_name!r} is missing from transport reverse table. "
                f"Available objectives: {available}."
            )
        objective_i = objective_lookup[objective_name]
        values.append(objective_values[objective_i])
        row = []
        for spec in parameter_set.specs:
            if isinstance(spec, ProfileParameterSpec):
                if spec.name not in profile_lookup:
                    available = ", ".join(table_result.profile_parameter_labels)
                    raise ValueError(
                        f"Profile parameter {spec.name!r} is missing from transport reverse table. "
                        f"Available profile parameters: {available}."
                    )
                row.append(profile_gradient_matrix[objective_i, profile_lookup[spec.name]])
                continue
            if isinstance(spec, VmecBoundaryParameterSpec):
                if spec.vmec_label in geometry_lookup:
                    geometry_i = geometry_lookup[spec.vmec_label]
                elif spec.label in geometry_lookup:
                    geometry_i = geometry_lookup[spec.label]
                else:
                    available = ", ".join(table_result.geometry_parameter_labels)
                    raise ValueError(
                        f"VMEC parameter {spec.vmec_label!r} is missing from transport reverse table. "
                        f"Available geometry parameters: {available}."
                    )
                row.append(geometry_gradient_matrix[objective_i, geometry_i])
                continue
            raise TypeError(f"Unsupported reverse-AD parameter spec type: {type(spec).__name__}.")
        jacobian_rows.append(jnp.stack(row))

    return ObjectiveTableResult(
        objective_names=requested_objectives,
        values=jnp.stack(values),
        jacobian=jnp.stack(jacobian_rows, axis=0),
    )


def transport_reverse_table_backend(
    table_result: RealtimeGeometryTransportReverseTableResult,
) -> ObjectiveTableBackend:
    """Return a grouped transport backend backed by a JAX-native table result."""

    def _backend(
        objective_names: tuple[str, ...],
        parameter_set: ReverseADParameterSet,
        options: Mapping[str, object],
    ) -> ObjectiveTableResult:
        del options
        requested_objectives = normalize_transport_objective_names(objective_names)
        return transport_reverse_table_result_to_objective_table_result(
            table_result,
            requested_objectives,
            parameter_set,
        )

    return _backend


def transport_reverse_table_result_builder_backend(
    table_result_builder: TransportReverseTableResultBuilder,
) -> ObjectiveTableBackend:
    """Return a grouped transport backend backed directly by a JAX table builder.

    This is the optimization-facing bridge we want long-term: the caller returns
    the validated grouped reverse table object, and this adapter selects the
    requested objective rows/parameter columns without report dictionaries or
    host-side conversions.
    """

    def _backend(
        objective_names: tuple[str, ...],
        parameter_set: ReverseADParameterSet,
        options: Mapping[str, object],
    ) -> ObjectiveTableResult:
        requested_objectives = normalize_transport_objective_names(objective_names)
        table_result = table_result_builder(requested_objectives, parameter_set, options)
        return transport_reverse_table_result_to_objective_table_result(
            table_result,
            requested_objectives,
            parameter_set,
        )

    return _backend


def transport_reverse_report_backend(report: Mapping[str, object]) -> ObjectiveTableBackend:
    """Return a grouped transport backend backed by a precomputed reverse report."""

    def _backend(
        objective_names: tuple[str, ...],
        parameter_set: ReverseADParameterSet,
        options: Mapping[str, object],
    ) -> ObjectiveTableResult:
        del options
        requested_objectives = normalize_transport_objective_names(objective_names)
        return transport_reverse_report_to_objective_table_result(
            report,
            requested_objectives,
            parameter_set,
        )

    return _backend


def transport_reverse_report_builder_backend(
    report_builder: TransportReverseReportBuilder,
) -> ObjectiveTableBackend:
    """Return a grouped transport backend backed by a validated report builder.

    The builder owns the actual grouped reverse execution.  This adapter only
    enforces the transport objective-table contract and preserves the same
    report-to-table conversion used for benchmark reports.
    """

    def _backend(
        objective_names: tuple[str, ...],
        parameter_set: ReverseADParameterSet,
        options: Mapping[str, object],
    ) -> ObjectiveTableResult:
        requested_objectives = normalize_transport_objective_names(objective_names)
        report = report_builder(requested_objectives, parameter_set, options)
        table_result = report.get("transport_reverse_table_result") if isinstance(report, Mapping) else None
        if isinstance(table_result, RealtimeGeometryTransportReverseTableResult):
            return transport_reverse_table_result_to_objective_table_result(
                table_result,
                requested_objectives,
                parameter_set,
            )
        return transport_reverse_report_to_objective_table_result(
            report,
            requested_objectives,
            parameter_set,
        )

    return _backend


def scalar_loss_and_gradient_from_least_squares(result: LeastSquaresResult):
    """Return `0.5 * r @ r` and `J.T @ r` from a least-squares result."""

    residuals = jnp.asarray(result.residuals)
    jacobian = jnp.asarray(result.jacobian)
    return 0.5 * jnp.vdot(residuals, residuals), jacobian.T @ residuals
