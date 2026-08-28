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
from typing import Any, Literal

import jax
import jax.numpy as jnp

from ._constants import elementary_charge
from ._reverse_ad_parameters import (
    PROFILE_PARAMETER_ORDER,
    ProfileParameterSpec,
    ReverseADParameterSet,
    VmecBoundaryParameterSpec,
    parameter_labels,
)
from ._transport_flux_models import DENSITY_STATE_TO_PHYSICAL, _add_float_delta_tree, _float_delta_tree_like
from ._geometry_autodiff import (
    build_neopax_geometry_and_ntx_exact_lij_support_from_state,
    geometry_full_ad_objective_table_pullback_from_param_vector,
    geometry_observable_names_for_kind,
    geometry_payload_pullback_from_param_vector_raw_block_transpose,
    geometry_raw_block_solve_from_param_vector,
    geometry_raw_block_transpose_from_state_bars,
)
from ._reverse_ad_initial_er import (
    compact_initial_er_ntx_support_pullback_leaves,
    compact_initial_er_state_pullback,
    find_ntx_support_payload,
    initial_er_charge_flux_residual_er_derivative,
    initial_er_charge_flux_residual_scalar,
    initial_er_charge_flux_residuals,
    initial_er_selected_root_profile,
    runtime_with_geometry_payload,
    runtime_with_ntx_support_payload,
)
from ._reverse_ad_transport import (
    RealtimeGeometryTransportReverseTableContext,
    RealtimeGeometryTransportReverseTableRequest,
    RealtimeGeometryTransportReverseTableResult,
    TransportReverseReportBuilder,
    TransportReverseReportRunner,
    TransportReverseTableResultBuilder,
    bootstrap_current_softmax_abs_scaled,
    net_total_power_volume_average,
    normalize_transport_objective_names,
    realtime_geometry_transport_reverse_table_from_payload_cotangents,
    realtime_geometry_transport_reverse_table_request,
    transport_realtime_geometry_reverse_table,
)
from ._optimization_initial_root_stage import (
    InitialErTransportPayloadAdapter,
    raw_block_dynamic_payload,
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
InitialErRootOnlyStateBuilder = Callable[[object], object]
InitialErRootOnlyLeastSquaresRunner = Callable[
    [object, Sequence[LeastSquaresTerm | tuple[ObjectiveRef | str, float, float]]],
    LeastSquaresEvaluation,
]
GeometryInitialErRootOnlyLeastSquaresRunner = Callable[
    [object, Sequence[LeastSquaresTerm | tuple[ObjectiveRef | str, float, float]]],
    LeastSquaresEvaluation,
]


@dataclasses.dataclass(frozen=True, slots=True)
class SharedGeometryTransportPayload:
    """Shared VMEC primal data for mixed optimization objectives.

    This is the first internal building block for fused optimization. It owns the
    one VMEC raw-block solve for the current optimizer parameter vector.
    Geometry/NTX transport payloads are intentionally built locally by the
    objective collector that needs them, so they are not retained while the
    payload-to-state VJP rebuilds its own traceable payload.
    """

    raw_block_solve: Any
    vmec_parameter_values: object
    vmec_specs: tuple[VmecBoundaryParameterSpec, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class ObjectiveCotangentTable:
    """Objective values plus pre-raw-block cotangents for fused optimization.

    Geometry columns are intentionally not assembled here. The optimizer-fused
    path should collect all VMEC-state/payload cotangents from geometry and
    transport objectives, then apply one batched raw-block transpose.
    """

    objective_names: tuple[str, ...]
    values: object
    profile_gradient_matrix: object
    vmec_state_bars: object | None = None
    payload_bars: tuple[Mapping[str, Any], ...] = ()


INITIAL_ER_ROOT_ONLY_OBJECTIVES: tuple[str, ...] = (
    "softmax_Er",
    "net_total_power_volume_average_mw_m3",
    "Er_transition_left",
    "Er_transition_right",
    "Er2_volume_average",
    "Er_volume_average",
)
_BOOTSTRAP_CURRENT_OBJECTIVE = "bootstrap_current_softmax_abs_scaled"
INITIAL_ER_ROOT_ONLY_EXPLICIT_OBJECTIVES: tuple[str, ...] = (
    *INITIAL_ER_ROOT_ONLY_OBJECTIVES,
    _BOOTSTRAP_CURRENT_OBJECTIVE,
)
GEOMETRY_FULL_AD_OBJECTIVE_ALIASES: Mapping[str, str] = {
    "aspect_ratio": "vmec_aspect_ratio",
    "vmec_aspect_ratio": "vmec_aspect_ratio",
    "volume_total": "vmec_volume_total",
    "vmec_volume_total": "vmec_volume_total",
    "iota": "vmec_iota_mean",
    "mean_iota": "vmec_iota_mean",
    "iota_mean": "vmec_iota_mean",
    "vmec_iota_mean": "vmec_iota_mean",
    "magnetic_well": "vmec_magnetic_well",
    "well": "vmec_magnetic_well",
    "vmec_magnetic_well": "vmec_magnetic_well",
    "mirror": "vmec_mirror_ratio",
    "mirror_ratio": "vmec_mirror_ratio",
    "vmec_mirror_ratio": "vmec_mirror_ratio",
    "beta_volume": "vmec_beta_volume",
    "vmec_beta_volume": "vmec_beta_volume",
    "boozer_iota_b_mean": "boozer_iota_b_mean",
    "boozer_b00_mean": "boozer_b00_mean",
    "boozer_buco_b_mean": "boozer_buco_b_mean",
    "boozer_bvco_b_mean": "boozer_bvco_b_mean",
    "boozer_aspect_proxy": "boozer_aspect_proxy",
    "boozer_b10_over_b00_mean": "boozer_b10_over_b00_mean",
    "qi": "boozer_qi_objective",
    "qi_objective": "boozer_qi_objective",
    "boozer_qi_objective": "boozer_qi_objective",
    "maxj": "boozer_maxj_objective",
    "maxj_objective": "boozer_maxj_objective",
    "boozer_maxj_objective": "boozer_maxj_objective",
}


def normalize_initial_er_root_only_objective_names(objective_names: Sequence[str] | str) -> tuple[str, ...]:
    """Return validated Er-only objective names for the initial-root optimization lane."""

    if isinstance(objective_names, str):
        raw_names = tuple(part.strip() for part in objective_names.split(",") if part.strip())
    else:
        raw_names = tuple(str(name).strip() for name in objective_names if str(name).strip())
    if not raw_names:
        raise ValueError("At least one initial-Er root-only objective is required.")
    allowed_objectives = INITIAL_ER_ROOT_ONLY_EXPLICIT_OBJECTIVES
    unknown = tuple(name for name in raw_names if name not in allowed_objectives)
    if unknown:
        allowed = ", ".join(allowed_objectives)
        raise ValueError(
            "Initial-Er root-only optimization supports only Er-related objectives; "
            f"unsupported={unknown!r}, choices are: {allowed}."
        )
    return raw_names


@dataclasses.dataclass(frozen=True, slots=True)
class InitialErRootOnlyReverseTableRequest:
    """Request for Er-root-only objectives without transport time evolution."""

    objective_names: tuple[str, ...]
    parameter_set: ReverseADParameterSet
    parameter_values: object
    runtime: object
    rooted_state_from_parameter_vector: InitialErRootOnlyStateBuilder
    options: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        names = normalize_initial_er_root_only_objective_names(self.objective_names)
        object.__setattr__(self, "objective_names", names)
        object.__setattr__(self, "parameter_values", jnp.asarray(self.parameter_values))


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


def _adapt_objective_table_result(
    result: ObjectiveTableResult,
    *,
    source_parameter_set: ReverseADParameterSet,
    target_parameter_set: ReverseADParameterSet,
    objective_names: Sequence[str],
) -> ObjectiveTableResult:
    """Project a backend table to the objective/parameter layout requested by a caller."""

    requested_objectives = tuple(str(name).strip() for name in objective_names if str(name).strip())
    row_lookup = _result_lookup(result)
    missing_objectives = tuple(name for name in requested_objectives if name not in row_lookup)
    if missing_objectives:
        available = ", ".join(result.objective_names)
        raise ValueError(
            f"Cannot adapt objective table; missing objectives {missing_objectives!r}. "
            f"Available objectives are: {available}."
        )

    source_column_lookup: dict[object, int] = {}
    for column_index, spec in enumerate(source_parameter_set.specs):
        if isinstance(spec, ProfileParameterSpec):
            source_column_lookup[("profile", spec.name)] = column_index
        elif isinstance(spec, VmecBoundaryParameterSpec):
            source_column_lookup[("vmec_boundary", spec.family, spec.m, spec.n)] = column_index
        else:
            raise TypeError(f"Unsupported reverse-AD parameter spec type: {type(spec).__name__}.")

    column_indices: list[int] = []
    for spec in target_parameter_set.specs:
        if isinstance(spec, ProfileParameterSpec):
            key = ("profile", spec.name)
        elif isinstance(spec, VmecBoundaryParameterSpec):
            key = ("vmec_boundary", spec.family, spec.m, spec.n)
        else:
            raise TypeError(f"Unsupported reverse-AD parameter spec type: {type(spec).__name__}.")
        try:
            column_indices.append(source_column_lookup[key])
        except KeyError as exc:
            raise ValueError(f"Cannot adapt objective table; missing parameter column {spec.label!r}.") from exc

    row_indices = [row_lookup[name] for name in requested_objectives]
    values = jnp.asarray(result.values)[jnp.asarray(row_indices, dtype=jnp.int32)]
    jacobian = jnp.asarray(result.jacobian)
    selected_rows = jacobian[jnp.asarray(row_indices, dtype=jnp.int32), :]
    if column_indices:
        selected_columns = selected_rows[:, jnp.asarray(column_indices, dtype=jnp.int32)]
    else:
        selected_columns = jnp.zeros((len(requested_objectives), 0), dtype=selected_rows.dtype)
    return ObjectiveTableResult(
        objective_names=requested_objectives,
        values=values,
        jacobian=selected_columns,
    )


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
    release_backend_graphs = bool(backend_options.pop("release_backend_graphs", False))
    backend_results: dict[ObjectiveFamily, ObjectiveTableResult] = {}
    for family, family_terms in grouped_terms.items():
        backend = backends.get(family)
        if backend is None:
            raise NotImplementedError(
                f"No reverse-AD backend is registered for objective family {family!r}."
            )
        objective_names = _unique_objective_names(family_terms)
        table_result = backend(objective_names, parameter_set, backend_options)
        if release_backend_graphs:
            values, jacobian = jax.block_until_ready((table_result.values, table_result.jacobian))
            table_result = ObjectiveTableResult(
                objective_names=table_result.objective_names,
                values=jnp.asarray(jax.device_get(values)),
                jacobian=jnp.asarray(jax.device_get(jacobian)),
            )
        backend_results[family] = table_result
    return assemble_least_squares_result(
        normalized_terms,
        parameter_set=parameter_set,
        backend_results=backend_results,
    )


def _initial_er_root_only_volume_average(profile, geometry):
    profile_arr = jnp.asarray(profile)
    vprime = jnp.asarray(geometry.Vprime, dtype=profile_arr.dtype)
    r_grid = jnp.asarray(geometry.r_grid, dtype=profile_arr.dtype)
    volume = jnp.trapezoid(vprime, x=r_grid)
    integral = jnp.trapezoid(profile_arr * vprime, x=r_grid)
    return integral / jnp.maximum(volume, jnp.asarray(1.0e-30, dtype=profile_arr.dtype))


def _bootstrap_current_softmax_abs_value_and_flux_bar(
    *,
    state,
    runtime,
    fluxes: Mapping[str, Any],
    beta: float = 128.0,
    eps: float = 1.0e-12,
):
    """Return smooth max(abs(Jboot)) and a sparse flux cotangent.

    This is the compact counterpart to ``bootstrap_current_softmax_abs_scaled``:
    it starts from already-evaluated lagged-response fluxes and creates only the
    ``Upar_neo`` cotangent needed by the NTX compact pullbacks.
    """

    upar = fluxes.get("Upar_neo", fluxes.get("Upar", None))
    if upar is None:
        raise ValueError("bootstrap current objective requires Upar or Upar_neo fluxes.")
    dtype = jnp.asarray(state.pressure).dtype
    charge_qp = jnp.asarray(runtime.species.charge_qp, dtype=dtype)
    current_weights = jnp.sign(charge_qp)
    upar_arr = jnp.asarray(upar, dtype=dtype)
    scale = jnp.asarray(elementary_charge * 1.0e-5, dtype=dtype)
    upar_physical_scale = jnp.asarray(DENSITY_STATE_TO_PHYSICAL, dtype=dtype)
    upar_physical = upar_physical_scale * upar_arr
    if int(upar_arr.shape[0]) == int(charge_qp.shape[0]):
        jboot = jnp.sum(upar_physical * current_weights[:, None], axis=0) * scale
        species_axis_first = True
    else:
        jboot = jnp.sum(upar_physical * current_weights[None, :], axis=1) * scale
        species_axis_first = False

    smooth_abs = jnp.sqrt(jboot * jboot + jnp.asarray(eps, dtype=dtype) ** 2)
    beta_arr = jnp.asarray(beta, dtype=dtype)
    value = jax.scipy.special.logsumexp(beta_arr * smooth_abs) / beta_arr
    smooth_abs_bar = jax.nn.softmax(beta_arr * smooth_abs)
    jboot_bar = smooth_abs_bar * jboot / jnp.maximum(smooth_abs, jnp.asarray(1.0e-30, dtype=dtype))
    if species_axis_first:
        upar_bar = current_weights[:, None] * (upar_physical_scale * scale * jboot_bar)[None, :]
    else:
        upar_bar = (upar_physical_scale * scale * jboot_bar)[:, None] * current_weights[None, :]

    gamma_ref = fluxes.get("Gamma_neo", fluxes.get("Gamma", None))
    q_ref = fluxes.get("Q_neo", fluxes.get("Q", None))
    flux_bar = {
        "Gamma": jnp.zeros_like(jnp.asarray(gamma_ref, dtype=dtype)) if gamma_ref is not None else 0,
        "Q": jnp.zeros_like(jnp.asarray(q_ref, dtype=dtype)) if q_ref is not None else 0,
        "Upar": jnp.zeros_like(upar_arr),
        "Gamma_neo": jnp.zeros_like(jnp.asarray(gamma_ref, dtype=dtype)) if gamma_ref is not None else 0,
        "Q_neo": jnp.zeros_like(jnp.asarray(q_ref, dtype=dtype)) if q_ref is not None else 0,
        "Upar_neo": upar_bar,
    }
    return value, flux_bar


def _bootstrap_current_softmax_abs_value_from_fluxes(
    *,
    state,
    runtime,
    fluxes: Mapping[str, Any],
    beta: float = 128.0,
    eps: float = 1.0e-12,
):
    value, _ = _bootstrap_current_softmax_abs_value_and_flux_bar(
        state=state,
        runtime=runtime,
        fluxes=fluxes,
        beta=beta,
        eps=eps,
    )
    return value


def _compact_bootstrap_current_root_objective_cotangent(
    *,
    rooted_state,
    runtime_for_geometry,
    baseline_geometry,
    baseline_ntx_support,
    geometry_delta0,
    dispatch_cache_probe=None,
):
    """Compact cotangent row for bootstrap current at the selected initial Er root."""

    flux_model = getattr(getattr(runtime_for_geometry, "models", None), "flux", None)
    neoclassical_model = getattr(flux_model, "neoclassical_model", flux_model)
    corrected_fluxes_fn = getattr(neoclassical_model, "evaluate_momentum_corrected_fluxes", None)
    state_pullback_fn = getattr(neoclassical_model, "pullback_momentum_corrected_upar_state_by_radius", None)
    support_pullback_fn = getattr(neoclassical_model, "pullback_momentum_corrected_upar_support_by_radius", None)
    geometry_pullback_fn = getattr(neoclassical_model, "pullback_momentum_corrected_upar_geometry_by_radius", None)
    if not callable(corrected_fluxes_fn):
        raise NotImplementedError(
            "bootstrap_current_softmax_abs_scaled requires realtime NTX "
            "evaluate_momentum_corrected_fluxes for compact root AD."
        )
    if not callable(state_pullback_fn) or not callable(support_pullback_fn) or not callable(geometry_pullback_fn):
        raise NotImplementedError(
            "bootstrap_current_softmax_abs_scaled requires compact corrected-Upar "
            "state, geometry, and support pullbacks on the realtime NTX model."
        )

    def _probe(label: str) -> None:
        if dispatch_cache_probe is not None:
            dispatch_cache_probe(str(label))

    _probe("before_bootstrap_corrected_fluxes")
    corrected_fluxes = corrected_fluxes_fn(rooted_state)
    _probe("after_bootstrap_corrected_fluxes")
    value, flux_bar = _bootstrap_current_softmax_abs_value_and_flux_bar(
        state=rooted_state,
        runtime=runtime_for_geometry,
        fluxes=corrected_fluxes,
    )
    upar_bar = flux_bar["Upar_neo"]
    _probe("before_bootstrap_state_pullback")
    state_bar = state_pullback_fn(rooted_state, upar_bar)
    _probe("after_bootstrap_state_pullback")
    _probe("before_bootstrap_geometry_pullback")
    geometry_bar = geometry_pullback_fn(rooted_state, upar_bar, baseline_geometry, baseline_ntx_support)
    _probe("after_bootstrap_geometry_pullback")
    _probe("before_bootstrap_ntx_support_pullback")
    support_bar_leaves = support_pullback_fn(rooted_state, upar_bar, baseline_ntx_support)
    _probe("after_bootstrap_ntx_support_pullback")
    _, support_treedef = jax.tree_util.tree_flatten(baseline_ntx_support)
    support_bar = support_treedef.unflatten(tuple(support_bar_leaves))
    return value, state_bar, geometry_bar, support_bar


def _row_from_batched_tree(tree, row_index: int):
    return jax.tree_util.tree_map(lambda leaf: leaf[row_index], tree)


def _stack_tree_rows(rows: Sequence[Any]):
    return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves, axis=0), *rows)


def _initial_er_root_only_objective_values(
    state,
    runtime,
    objective_names: Sequence[str],
    *,
    options: Mapping[str, object] | None = None,
):
    names = normalize_initial_er_root_only_objective_names(objective_names)
    opts = {} if options is None else options
    er = jnp.asarray(state.Er)
    geometry = runtime.geometry
    softmax_beta = float(opts.get("softmax_Er_beta", 16.0))
    transition_left_index = int(opts.get("Er_transition_left_index", 20))
    transition_right_index = int(opts.get("Er_transition_right_index", 21))

    def _er_at_index(index: int):
        clipped = max(0, min(int(index), int(er.shape[-1]) - 1))
        return er[clipped]

    def _one(name: str):
        if name == "softmax_Er":
            beta = jnp.asarray(softmax_beta, dtype=er.dtype)
            return jax.scipy.special.logsumexp(beta * er) / beta
        if name == "net_total_power_volume_average_mw_m3":
            return net_total_power_volume_average(state, runtime)
        if name == "Er_transition_left":
            return _er_at_index(transition_left_index)
        if name == "Er_transition_right":
            return _er_at_index(transition_right_index)
        if name == "Er2_volume_average":
            return _initial_er_root_only_volume_average(er * er, geometry)
        if name == "Er_volume_average":
            return _initial_er_root_only_volume_average(er, geometry)
        if name == _BOOTSTRAP_CURRENT_OBJECTIVE:
            return bootstrap_current_softmax_abs_scaled(state, runtime)
        raise ValueError(f"Unsupported initial-Er root-only objective {name!r}.")

    return jnp.stack([_one(name) for name in names])


_DIRECT_INITIAL_ER_OBJECTIVES = frozenset(
    ("softmax_Er", "Er_transition_left", "Er_transition_right")
)


def _direct_initial_er_objective_values_and_state_bars(
    state,
    objective_names: Sequence[str],
    *,
    options: Mapping[str, object] | None = None,
):
    """Return exact Er-only objective values and their state cotangent rows.

    These three objectives depend only on the selected Er vector.  Their
    closed-form rows are algebraically identical to the generic VJP used by
    the benchmark path, but avoid constructing that VJP in the opt-in
    optimization route.  Geometry and NTX-support direct bars are exactly
    zero for this restricted objective set.
    """

    names = tuple(objective_names)
    unsupported = tuple(name for name in names if name not in _DIRECT_INITIAL_ER_OBJECTIVES)
    if unsupported:
        raise ValueError(f"Direct initial-Er rows do not support {unsupported!r}.")
    opts = {} if options is None else options
    er = jnp.asarray(state.Er)
    beta = jnp.asarray(float(opts.get("softmax_Er_beta", 16.0)), dtype=er.dtype)
    left_index = max(0, min(int(opts.get("Er_transition_left_index", 20)), int(er.shape[0]) - 1))
    right_index = max(0, min(int(opts.get("Er_transition_right_index", 21)), int(er.shape[0]) - 1))

    def _zero_leaf(leaf):
        return jnp.zeros_like(jnp.asarray(leaf))

    zero_state = jax.tree_util.tree_map(_zero_leaf, state)
    values = []
    rows = []
    for name in names:
        if name == "softmax_Er":
            values.append(jax.scipy.special.logsumexp(beta * er) / beta)
            er_bar = jax.nn.softmax(beta * er)
        elif name == "Er_transition_left":
            values.append(er[left_index])
            er_bar = jnp.zeros_like(er).at[left_index].set(1.0)
        else:
            values.append(er[right_index])
            er_bar = jnp.zeros_like(er).at[right_index].set(1.0)
        rows.append(dataclasses.replace(zero_state, Er=er_bar))
    return jnp.stack(values), _stack_tree_rows(rows)


@dataclasses.dataclass(frozen=True, slots=True)
class InitialErTransportReverseStage:
    """Persistent, bounded reverse kernels for one initial-Er optimization stage.

    The benchmark route does not construct this object.  Each kernel receives
    all current numerical data explicitly; geometry/support metadata stays in
    ``payload_adapter`` and is never traced as a dynamic JAX argument.
    """

    payload_adapter: InitialErTransportPayloadAdapter
    selected_root: Callable[..., Any]
    state_pullback: Callable[..., Any]
    geometry_pullback: Callable[..., Any]
    support_pullback: Callable[..., Any]


def build_initial_er_transport_reverse_stage(
    *,
    runtime,
    payload_adapter: InitialErTransportPayloadAdapter,
    config: Mapping[str, object],
) -> InitialErTransportReverseStage:
    """Build the three fixed-signature transport reverse kernels once.

    This is intentionally below the optimizer and VMEC boundaries: each
    kernel covers one existing compact reverse operation only.  No objective
    assembly, geometry solve, or payload-to-VMEC pullback is fused into it.
    """

    def _runtime_from_leaves(geometry_leaves, support_leaves):
        payload = payload_adapter.rebuild(geometry_leaves, support_leaves)
        runtime_with_geometry = runtime_with_geometry_payload(runtime, payload["geometry"])
        return runtime_with_ntx_support_payload(
            runtime_with_geometry, payload["ntx_support"]
        )

    def _state_pullback(state, er_profile, residual_bars, geometry_leaves, support_leaves):
        return compact_initial_er_state_pullback(
            residual_scalar_fn=initial_er_charge_flux_residual_scalar,
            state=state,
            er_profile=er_profile,
            residual_bars=residual_bars,
            runtime=_runtime_from_leaves(geometry_leaves, support_leaves),
        )

    def _selected_root(state, geometry_leaves, support_leaves):
        return initial_er_selected_root_profile(
            state,
            config=dict(config),
            runtime=_runtime_from_leaves(geometry_leaves, support_leaves),
        )

    def _geometry_pullback(
        state,
        er_profile,
        residual_bars,
        geometry_delta,
        geometry_leaves,
        support_leaves,
    ):
        payload = payload_adapter.rebuild(geometry_leaves, support_leaves)
        geometry_base = jax.tree_util.tree_map(jax.lax.stop_gradient, payload["geometry"])
        support_base = jax.tree_util.tree_map(jax.lax.stop_gradient, payload["ntx_support"])

        def _residuals_from_delta(delta):
            trial_geometry = _add_float_delta_tree(geometry_base, delta)
            runtime_with_geometry = runtime_with_geometry_payload(runtime, trial_geometry)
            runtime_with_geometry = runtime_with_ntx_support_payload(
                runtime_with_geometry, support_base
            )
            return initial_er_charge_flux_residuals(
                jax.tree_util.tree_map(jax.lax.stop_gradient, state),
                jax.lax.stop_gradient(er_profile),
                runtime=runtime_with_geometry,
            )

        _, pullback = jax.vjp(_residuals_from_delta, geometry_delta)
        return jax.vmap(lambda bars: pullback(bars)[0])(residual_bars)

    def _support_pullback(state, er_profile, residual_bars, geometry_leaves, support_leaves):
        payload = payload_adapter.rebuild(geometry_leaves, support_leaves)
        runtime_with_geometry = runtime_with_geometry_payload(runtime, payload["geometry"])
        return compact_initial_er_ntx_support_pullback_leaves(
            runtime=runtime_with_geometry,
            state=state,
            er_profile=er_profile,
            residual_bars=residual_bars,
            support=payload["ntx_support"],
        )

    return InitialErTransportReverseStage(
        payload_adapter=payload_adapter,
        selected_root=jax.jit(_selected_root, inline=False),
        state_pullback=jax.jit(_state_pullback, inline=False),
        geometry_pullback=jax.jit(_geometry_pullback, inline=False),
        support_pullback=jax.jit(_support_pullback, inline=False),
    )


def initial_er_root_only_reverse_table(
    request: InitialErRootOnlyReverseTableRequest,
) -> ObjectiveTableResult:
    """Evaluate Er-root-only objective values and Jacobian with no time evolution."""

    if _BOOTSTRAP_CURRENT_OBJECTIVE in request.objective_names:
        raise NotImplementedError(
            "bootstrap_current_softmax_abs_scaled must use the geometry-active compact "
            "initial-Er root table; the generic root-only table would trace the full "
            "momentum-corrected NTX evaluator."
        )
    parameter_values = jnp.asarray(request.parameter_values)

    def _values_from_parameter_vector(values):
        rooted_state = request.rooted_state_from_parameter_vector(values)
        return _initial_er_root_only_objective_values(
            rooted_state,
            request.runtime,
            request.objective_names,
            options=request.options,
        )

    objective_values, pullback = jax.vjp(_values_from_parameter_vector, parameter_values)
    objective_basis = jnp.eye(len(request.objective_names), dtype=objective_values.dtype)
    objective_jacobian = jax.vmap(lambda cotangent: pullback(cotangent)[0])(objective_basis)
    return ObjectiveTableResult(
        objective_names=request.objective_names,
        values=objective_values,
        jacobian=objective_jacobian,
    )


def evaluate_initial_er_root_only_least_squares(
    config: Mapping[str, object],
    *,
    request: InitialErRootOnlyReverseTableRequest,
    terms: Sequence[LeastSquaresTerm | tuple[ObjectiveRef | str, float, float]],
    options: Mapping[str, object] | None = None,
) -> LeastSquaresEvaluation:
    """Evaluate least squares for selected initial ambipolar-Er root objectives only."""

    normalized_terms = normalize_least_squares_terms(terms)
    grouped_terms = group_least_squares_terms_by_family(normalized_terms)
    unsupported_families = tuple(family for family in grouped_terms if family != "transport")
    if unsupported_families:
        raise NotImplementedError(
            "evaluate_initial_er_root_only_least_squares only supports transport/Er terms; "
            f"got families {unsupported_families!r}."
        )
    requested_term_objectives = _unique_objective_names(grouped_terms.get("transport", ()))
    if requested_term_objectives != request.objective_names:
        raise ValueError(
            "Initial-Er root-only request objectives must match the transport least-squares "
            "terms in first-use order: "
            f"request={request.objective_names!r}, terms={requested_term_objectives!r}."
        )
    t_start = time.perf_counter()
    table_result = initial_er_root_only_reverse_table(request)
    result = residuals_and_jacobian_reverse_ad(
        config,
        parameter_set=request.parameter_set,
        terms=normalized_terms,
        backends={"transport": lambda _names, _parameter_set, _options: table_result},
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


def build_initial_er_root_only_least_squares_runner(
    config: Mapping[str, object],
    *,
    runtime,
    parameter_set: ReverseADParameterSet,
    rooted_state_from_parameter_vector: InitialErRootOnlyStateBuilder,
    objective_names: Sequence[str] | str | None = None,
    options: Mapping[str, object] | None = None,
) -> InitialErRootOnlyLeastSquaresRunner:
    """Build a TOML-backed runner for ambipolar-root Er objectives without rollout.

    The caller owns TOML/config preparation and supplies a state builder that
    maps the active parameter vector to a state whose ``Er`` is the selected
    ambipolar root.  This keeps the inner AD path independent of file/config
    parsing while preserving the TOML-derived runtime, profiles, geometry, and
    ambipolarity settings.
    """

    normalized_objectives = (
        None
        if objective_names is None
        else normalize_initial_er_root_only_objective_names(objective_names)
    )
    runner_options = {} if options is None else dict(options)

    def _runner(
        parameter_values,
        terms: Sequence[LeastSquaresTerm | tuple[ObjectiveRef | str, float, float]],
    ) -> LeastSquaresEvaluation:
        normalized_terms = normalize_least_squares_terms(terms)
        grouped_terms = group_least_squares_terms_by_family(normalized_terms)
        active_objectives = (
            normalized_objectives
            if normalized_objectives is not None
            else normalize_initial_er_root_only_objective_names(
                _unique_objective_names(grouped_terms.get("transport", ()))
            )
        )
        request = InitialErRootOnlyReverseTableRequest(
            objective_names=active_objectives,
            parameter_set=parameter_set,
            parameter_values=parameter_values,
            runtime=runtime,
            rooted_state_from_parameter_vector=rooted_state_from_parameter_vector,
            options=runner_options,
        )
        return evaluate_initial_er_root_only_least_squares(
            config,
            request=request,
            terms=normalized_terms,
            options=runner_options,
        )

    return _runner


def _add_trees(lhs, rhs):
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs
    return jax.tree_util.tree_map(lambda a, b: a + b, lhs, rhs)


def vmec_parameter_values_from_parameter_vector(
    parameter_set: ReverseADParameterSet,
    parameter_values,
) -> jnp.ndarray:
    """Extract VMEC-boundary columns from a mixed optimization vector."""

    parameter_values_arr = jnp.asarray(parameter_values, dtype=jnp.float64)
    if parameter_values_arr.ndim != 1 or int(parameter_values_arr.shape[0]) != len(parameter_set.specs):
        raise ValueError(
            "parameter_values must be a 1D vector matching the reverse-AD parameter set; "
            f"got shape={parameter_values_arr.shape}, parameter_count={len(parameter_set.specs)}."
        )
    values = [
        parameter_values_arr[i]
        for i, spec in enumerate(parameter_set.specs)
        if isinstance(spec, VmecBoundaryParameterSpec)
    ]
    return jnp.asarray(values, dtype=jnp.float64)


def build_shared_geometry_transport_payload(
    *,
    geometry_context,
    parameter_set: ReverseADParameterSet,
    parameter_values,
    runtime,
    n_r: int,
    n_theta: int,
    n_zeta: int,
    n_xi: int,
    surface_backend: str = "vmec",
    max_iter: int | None = None,
    solver_device: str | None = "default",
) -> SharedGeometryTransportPayload:
    """Build one VMEC solve for mixed geometry/transport optimization.

    This helper supports mixed profile+geometry parameter vectors by extracting
    only VMEC-boundary entries for the VMEC solve. It deliberately lives in
    NEOPAX internals so thin optimization scripts do not own raw-block/payload
    plumbing. The realtime transport payload is not retained here to avoid
    overlapping it with the payload VJP's traceable payload rebuild.
    """

    vmec_specs = tuple(parameter_set.vmec_boundary_specs)
    if not vmec_specs:
        raise ValueError("A shared geometry/transport payload requires at least one VMEC boundary parameter.")
    vmec_values = vmec_parameter_values_from_parameter_vector(parameter_set, parameter_values)
    raw_block_solve = geometry_raw_block_solve_from_param_vector(
        geometry_context,
        vmec_values,
        tuple(spec.as_tuple() for spec in vmec_specs),
        max_iter=max_iter,
        solver_device=solver_device,
    )
    del runtime, n_r, n_theta, n_zeta, n_xi, surface_backend
    return SharedGeometryTransportPayload(
        raw_block_solve=raw_block_solve,
        vmec_parameter_values=vmec_values,
        vmec_specs=vmec_specs,
    )


def initial_er_root_only_objective_cotangent_table(
    *,
    config: Mapping[str, object],
    objective_names: Sequence[str] | str,
    parameter_set: ReverseADParameterSet,
    parameter_values,
    runtime,
    profile_values,
    pre_root_state_from_profile_values: Callable[[object], object],
    support_payload_override=None,
    options: Mapping[str, object] | None = None,
) -> ObjectiveCotangentTable:
    """Return initial-Er root objective values plus profile/payload cotangents.

    This is the compact root-only reverse path stopped before the final
    payload-to-VMEC raw-block pullback. It is intentionally separate from the
    benchmark-good assembled table so fused optimization can collect transport
    and geometry cotangents first, then perform one shared geometry pullback.
    """

    requested_objectives = normalize_initial_er_root_only_objective_names(objective_names)
    parameter_values_arr = jnp.asarray(parameter_values)
    profile_values_arr = jnp.asarray(profile_values)
    if parameter_values_arr.ndim != 1 or int(parameter_values_arr.shape[0]) != len(parameter_set.specs):
        raise ValueError(
            "parameter_values must be a 1D vector matching the reverse-AD parameter set; "
            f"got shape={parameter_values_arr.shape}, parameter_count={len(parameter_set.specs)}."
        )

    support_payload = support_payload_override
    if support_payload is None:
        support_payload = find_ntx_support_payload(runtime)
    if not isinstance(support_payload, dict):
        support_payload = {
            "geometry": runtime.geometry,
            "ntx_support": support_payload,
        }
    baseline_geometry = support_payload["geometry"]
    baseline_ntx_support = support_payload["ntx_support"]
    runtime_for_geometry = runtime_with_geometry_payload(runtime, baseline_geometry)
    runtime_for_geometry = runtime_with_ntx_support_payload(runtime_for_geometry, baseline_ntx_support)
    geometry_delta0 = _float_delta_tree_like(baseline_geometry)

    pre_root_state = pre_root_state_from_profile_values(profile_values_arr)
    er_profile, finite_mask = initial_er_selected_root_profile(
        pre_root_state,
        config=dict(config),
        runtime=runtime_for_geometry,
    )
    er_profile = jnp.asarray(er_profile, dtype=pre_root_state.Er.dtype)
    finite_mask = jnp.asarray(finite_mask, dtype=bool)
    rooted_state = dataclasses.replace(pre_root_state, Er=er_profile)

    generic_objectives = tuple(name for name in requested_objectives if name != _BOOTSTRAP_CURRENT_OBJECTIVE)

    def _values_from_rooted_state_and_geometry(state_value, geometry_delta):
        geometry = _add_float_delta_tree(baseline_geometry, geometry_delta)
        runtime_with_geometry = runtime_with_geometry_payload(runtime_for_geometry, geometry)
        runtime_with_geometry = runtime_with_ntx_support_payload(runtime_with_geometry, baseline_ntx_support)
        return _initial_er_root_only_objective_values(
            state_value,
            runtime_with_geometry,
            generic_objectives,
            options=options,
        )

    objective_count = len(requested_objectives)
    generic_values = None
    generic_rooted_state_bars = None
    generic_direct_geometry_bars = None
    generic_value_lookup = {name: i for i, name in enumerate(generic_objectives)}
    if generic_objectives:
        generic_values, objective_pullback = jax.vjp(
            _values_from_rooted_state_and_geometry,
            rooted_state,
            geometry_delta0,
        )
        generic_basis = jnp.eye(len(generic_objectives), dtype=jnp.asarray(generic_values).dtype)
        generic_rooted_state_bars, generic_direct_geometry_bars = jax.vmap(
            lambda cotangent: objective_pullback(cotangent)
        )(generic_basis)

    bootstrap_row = None
    if _BOOTSTRAP_CURRENT_OBJECTIVE in requested_objectives:
        bootstrap_row = _compact_bootstrap_current_root_objective_cotangent(
            rooted_state=rooted_state,
            runtime_for_geometry=runtime_for_geometry,
            baseline_geometry=baseline_geometry,
            baseline_ntx_support=baseline_ntx_support,
            geometry_delta0=geometry_delta0,
        )

    objective_value_rows = []
    rooted_state_bar_rows = []
    direct_geometry_bar_rows = []
    direct_ntx_support_bar_rows = []
    zero_ntx_support_bar = _float_delta_tree_like(baseline_ntx_support)
    for name in requested_objectives:
        if name == _BOOTSTRAP_CURRENT_OBJECTIVE:
            if bootstrap_row is None:
                raise AssertionError("bootstrap cotangent row was not built.")
            value, rooted_state_bar, direct_geometry_bar, direct_support_bar = bootstrap_row
            objective_value_rows.append(value)
            rooted_state_bar_rows.append(rooted_state_bar)
            direct_geometry_bar_rows.append(direct_geometry_bar)
            direct_ntx_support_bar_rows.append(direct_support_bar)
            continue
        if generic_values is None or generic_rooted_state_bars is None or generic_direct_geometry_bars is None:
            raise AssertionError("generic root objective rows were not built.")
        generic_i = generic_value_lookup[name]
        objective_value_rows.append(generic_values[generic_i])
        rooted_state_bar_rows.append(_row_from_batched_tree(generic_rooted_state_bars, generic_i))
        direct_geometry_bar_rows.append(_row_from_batched_tree(generic_direct_geometry_bars, generic_i))
        direct_ntx_support_bar_rows.append(zero_ntx_support_bar)

    objective_values = jnp.stack(objective_value_rows)
    rooted_state_bars = _stack_tree_rows(rooted_state_bar_rows)
    direct_geometry_bars = _stack_tree_rows(direct_geometry_bar_rows)
    direct_ntx_support_bars = _stack_tree_rows(direct_ntx_support_bar_rows)

    dres_der = initial_er_charge_flux_residual_er_derivative(
        pre_root_state,
        er_profile,
        runtime=runtime_for_geometry,
    )
    safe_dres_der = jnp.where(
        jnp.abs(dres_der) > jnp.asarray(1.0e-30, dtype=dres_der.dtype),
        dres_der,
        jnp.inf,
    )
    residual_bars = jnp.where(
        finite_mask[None, :],
        -jnp.asarray(rooted_state_bars.Er) / safe_dres_der[None, :],
        0.0,
    )
    state_residual_bars = compact_initial_er_state_pullback(
        residual_scalar_fn=initial_er_charge_flux_residual_scalar,
        state=pre_root_state,
        er_profile=er_profile,
        residual_bars=residual_bars,
        runtime=runtime_for_geometry,
    )
    direct_pre_root_state_bars = dataclasses.replace(
        rooted_state_bars,
        Er=jnp.zeros_like(rooted_state_bars.Er),
    )
    pre_root_state_bars = _add_trees(direct_pre_root_state_bars, state_residual_bars)

    profile_specs = tuple(parameter_set.profile_specs)
    if profile_specs:
        _, profile_pullback = jax.vjp(
            pre_root_state_from_profile_values,
            profile_values_arr,
        )
        profile_gradient_all = jax.vmap(lambda state_bar: profile_pullback(state_bar)[0])(
            pre_root_state_bars
        )
        canonical_profile_lookup = {name: i for i, name in enumerate(PROFILE_PARAMETER_ORDER)}
        profile_gradient_matrix = jnp.stack(
            [profile_gradient_all[:, canonical_profile_lookup[spec.name]] for spec in profile_specs],
            axis=1,
        )
    else:
        profile_gradient_matrix = jnp.zeros((objective_count, 0), dtype=jnp.asarray(objective_values).dtype)

    def _residuals_from_geometry_delta(geometry_delta):
        geometry = _add_float_delta_tree(baseline_geometry, geometry_delta)
        runtime_with_geometry = runtime_with_geometry_payload(runtime_for_geometry, geometry)
        runtime_with_geometry = runtime_with_ntx_support_payload(runtime_with_geometry, baseline_ntx_support)
        return initial_er_charge_flux_residuals(
            pre_root_state,
            er_profile,
            runtime=runtime_with_geometry,
        )

    _, geometry_residual_pullback = jax.vjp(
        _residuals_from_geometry_delta,
        geometry_delta0,
    )
    residual_geometry_bars = jax.vmap(lambda residual_bar: geometry_residual_pullback(residual_bar)[0])(
        residual_bars
    )
    geometry_bars = _add_trees(direct_geometry_bars, residual_geometry_bars)

    ntx_runtime = runtime_with_geometry_payload(runtime_for_geometry, baseline_geometry)
    ntx_bar_leaves = compact_initial_er_ntx_support_pullback_leaves(
        runtime=ntx_runtime,
        state=pre_root_state,
        er_profile=er_profile,
        residual_bars=residual_bars,
        support=baseline_ntx_support,
    )
    _, ntx_treedef = jax.tree_util.tree_flatten(baseline_ntx_support)
    ntx_bars = ntx_treedef.unflatten(tuple(ntx_bar_leaves))
    support_bars = []
    for objective_index in range(objective_count):
        geometry_bar = jax.tree_util.tree_map(lambda leaf: leaf[objective_index], geometry_bars)
        ntx_bar = jax.tree_util.tree_map(
            lambda residual_leaf, direct_leaf: residual_leaf[objective_index] + direct_leaf[objective_index],
            ntx_bars,
            direct_ntx_support_bars,
        )
        support_bars.append({"geometry": geometry_bar, "ntx_support": ntx_bar})

    return ObjectiveCotangentTable(
        objective_names=requested_objectives,
        values=objective_values,
        profile_gradient_matrix=profile_gradient_matrix,
        vmec_state_bars=None,
        payload_bars=tuple(support_bars),
    )


def fused_geometry_parameter_matrix_from_cotangent_tables(
    *,
    geometry_context,
    parameter_set: ReverseADParameterSet,
    tables: Sequence[ObjectiveCotangentTable],
    shared_payload: SharedGeometryTransportPayload,
    n_r: int,
    n_theta: int,
    n_zeta: int,
    n_xi: int,
    surface_backend: str = "vmec",
    max_iter: int | None = None,
    solver_device: str | None = "default",
    extra_state_bars_factory=None,
) -> jnp.ndarray:
    """Fuse objective-family cotangents and pull them once to VMEC columns."""

    del parameter_set  # Geometry columns are ordered by shared_payload.vmec_specs.
    if not shared_payload.vmec_specs:
        raise ValueError("Fused geometry pullback requires at least one VMEC boundary parameter.")
    def _zero_state_bar_batch(row_count: int):
        if shared_payload.raw_block_solve is None:
            raise ValueError("Direct VMEC-state cotangent rows require an existing raw-block solve.")
        state = shared_payload.raw_block_solve.state
        return jax.tree_util.tree_map(
            lambda leaf: jnp.broadcast_to(jnp.zeros_like(leaf)[None, ...], (int(row_count),) + leaf.shape),
            state,
        )

    extra_state_bar_batches = []
    payload_bars_for_pullback = []
    row_blocks: list[tuple[str, int]] = []
    for table in tables:
        row_count = len(table.objective_names)
        values = jnp.asarray(table.values)
        if values.ndim != 1 or int(values.shape[0]) != row_count:
            raise ValueError(
                "ObjectiveCotangentTable values must have shape (objective_count,); "
                f"got values.shape={values.shape}, objective_count={row_count}."
            )
        profile_gradient_matrix = jnp.asarray(table.profile_gradient_matrix)
        if profile_gradient_matrix.ndim != 2 or int(profile_gradient_matrix.shape[0]) != row_count:
            raise ValueError(
                "ObjectiveCotangentTable profile_gradient_matrix must have one row per objective; "
                f"got shape={profile_gradient_matrix.shape}, objective_count={row_count}."
            )
        if table.payload_bars and len(table.payload_bars) != row_count:
            raise ValueError(
                "ObjectiveCotangentTable payload_bars must have one cotangent tree per objective; "
                f"got payload_count={len(table.payload_bars)}, objective_count={row_count}."
            )
        if table.payload_bars:
            if table.vmec_state_bars is not None:
                raise NotImplementedError(
                    "Fused payload rows with additional direct VMEC-state bars are not supported yet; "
                    "split this objective family into separate cotangent tables."
                )
            payload_bars_for_pullback.extend(table.payload_bars)
            row_blocks.append(("payload", row_count))
        else:
            table_state_bar = _zero_state_bar_batch(row_count)
            if table.vmec_state_bars is not None:
                table_state_bar = jax.tree_util.tree_map(
                    lambda left, right: left + right,
                    table_state_bar,
                    table.vmec_state_bars,
                )
            extra_state_bar_batches.append(table_state_bar)
            row_blocks.append(("state", row_count))

    if not payload_bars_for_pullback and not extra_state_bar_batches:
        return jnp.zeros((0, len(shared_payload.vmec_specs)), dtype=jnp.float64)

    extra_state_bar_batch = (
        None
        if not extra_state_bar_batches
        else jax.tree_util.tree_map(
            lambda *leaves: jnp.concatenate(leaves, axis=0),
            *extra_state_bar_batches,
        )
    )
    payload_matrix = None
    if payload_bars_for_pullback:
        # Keep payload rows on the benchmark-good compact tangent-contraction
        # path. Passing VMEC-state geometry rows as extra_state_bars disables
        # that compact path and can force a full NTX support VJP.
        payload_matrix = geometry_payload_pullback_from_param_vector_raw_block_transpose(
            geometry_context,
            shared_payload.vmec_parameter_values,
            tuple(spec.as_tuple() for spec in shared_payload.vmec_specs),
            tuple(payload_bars_for_pullback),
            combined_payload=True,
            n_r=int(n_r),
            n_theta=int(n_theta),
            n_zeta=int(n_zeta),
            n_xi=int(n_xi),
            surface_backend=str(surface_backend),
            max_iter=max_iter,
            solver_device=solver_device,
            raw_block_solve=shared_payload.raw_block_solve,
        )
    if extra_state_bars_factory is not None:
        if shared_payload.raw_block_solve is None:
            raise ValueError("Deferred VMEC-state cotangent rows require an existing raw-block solve.")
        deferred_state_bars = extra_state_bars_factory(shared_payload.raw_block_solve)
        extra_state_bar_batch = (
            deferred_state_bars
            if extra_state_bar_batch is None
            else jax.tree_util.tree_map(
                lambda left, right: jnp.concatenate([left, right], axis=0),
                extra_state_bar_batch,
                deferred_state_bars,
            )
        )
        row_blocks.append(("state", int(jax.tree_util.tree_leaves(deferred_state_bars)[0].shape[0])))
    state_matrix = None
    if extra_state_bar_batch is not None:
        if shared_payload.raw_block_solve is None:
            raise ValueError("Direct VMEC-state cotangent rows require an existing raw-block solve.")
        state_matrix = geometry_raw_block_transpose_from_state_bars(
            shared_payload.raw_block_solve,
            extra_state_bar_batch,
            probe_chunk_size=1,
        )
    if payload_matrix is None and state_matrix is None:
        raise ValueError("Direct VMEC-state cotangent rows require an existing raw-block solve.")
    if payload_matrix is None:
        return state_matrix
    if state_matrix is None:
        return payload_matrix

    payload_row0 = 0
    state_row0 = 0
    ordered_blocks = []
    for kind, row_count in row_blocks:
        row_count = int(row_count)
        if kind == "payload":
            ordered_blocks.append(payload_matrix[payload_row0 : payload_row0 + row_count])
            payload_row0 += row_count
        else:
            ordered_blocks.append(state_matrix[state_row0 : state_row0 + row_count])
            state_row0 += row_count
    return jnp.concatenate(ordered_blocks, axis=0)


def objective_table_result_from_cotangent_table(
    table: ObjectiveCotangentTable,
    *,
    parameter_set: ReverseADParameterSet,
    geometry_gradient_matrix,
) -> ObjectiveTableResult:
    """Expand one cotangent table into the full mixed-parameter Jacobian layout."""

    objective_count = len(table.objective_names)
    values = jnp.asarray(table.values)
    profile_gradient_matrix = jnp.asarray(table.profile_gradient_matrix)
    geometry_gradient_matrix = jnp.asarray(geometry_gradient_matrix)
    profile_specs = tuple(parameter_set.profile_specs)
    vmec_specs = tuple(parameter_set.vmec_boundary_specs)
    if values.ndim != 1 or int(values.shape[0]) != objective_count:
        raise ValueError(
            "ObjectiveCotangentTable values must have shape (objective_count,); "
            f"got values.shape={values.shape}, objective_count={objective_count}."
        )
    if profile_gradient_matrix.shape != (objective_count, len(profile_specs)):
        raise ValueError(
            "ObjectiveCotangentTable profile_gradient_matrix shape does not match profile parameters: "
            f"got {profile_gradient_matrix.shape}, expected {(objective_count, len(profile_specs))}."
        )
    if geometry_gradient_matrix.shape != (objective_count, len(vmec_specs)):
        raise ValueError(
            "geometry_gradient_matrix shape does not match VMEC parameters: "
            f"got {geometry_gradient_matrix.shape}, expected {(objective_count, len(vmec_specs))}."
        )

    profile_lookup = {spec: i for i, spec in enumerate(profile_specs)}
    vmec_lookup = {spec: i for i, spec in enumerate(vmec_specs)}
    jacobian_rows = []
    for row_i in range(objective_count):
        columns = []
        for spec in parameter_set.specs:
            if isinstance(spec, ProfileParameterSpec):
                columns.append(profile_gradient_matrix[row_i, profile_lookup[spec]])
            elif isinstance(spec, VmecBoundaryParameterSpec):
                columns.append(geometry_gradient_matrix[row_i, vmec_lookup[spec]])
            else:
                raise TypeError(f"Unsupported reverse-AD parameter spec type: {type(spec).__name__}.")
        jacobian_rows.append(jnp.stack(columns))

    return ObjectiveTableResult(
        objective_names=table.objective_names,
        values=values,
        jacobian=jnp.stack(jacobian_rows, axis=0),
    )


def geometry_active_initial_er_root_only_reverse_table(
    *,
    config: Mapping[str, object],
    objective_names: Sequence[str] | str,
    parameter_set: ReverseADParameterSet,
    parameter_values,
    runtime,
    profile_values,
    pre_root_state_from_profile_values: Callable[[object], object],
    geometry_context,
    baseline_geometry_deltas,
    n_r: int,
    n_theta: int,
    n_zeta: int,
    n_xi: int,
    surface_backend: str = "vmec",
    max_iter: int | None = None,
    solver_device: str | None = "default",
    progress_label: str | None = None,
    raw_block_solve=None,
    support_payload_override=None,
    options: Mapping[str, object] | None = None,
    dispatch_cache_probe=None,
) -> ObjectiveTableResult:
    """Return compact initial-Er objective table for active realtime geometry.

    This is the internalized version of the benchmark's geometry-active
    `--initial-Er-root-only-optimization-smoke` path.  It keeps the selected
    ambipolar branch fixed, uses the compact root residual transposes, and maps
    geometry/support payload cotangents to VMEC boundary columns with the same
    raw-block payload pullback used by the realtime reverse benchmark.
    """

    def _probe(label: str) -> None:
        # Test-only instrumentation. Default/benchmark calls leave this unset.
        if dispatch_cache_probe is not None:
            dispatch_cache_probe(str(label))

    _probe("support_reverse_entry")
    requested_objectives = normalize_initial_er_root_only_objective_names(objective_names)
    parameter_values_arr = jnp.asarray(parameter_values)
    profile_values_arr = jnp.asarray(profile_values)
    vmec_specs = tuple(parameter_set.vmec_boundary_specs)
    if not vmec_specs:
        raise ValueError("geometry_active_initial_er_root_only_reverse_table requires VMEC boundary parameters.")
    if parameter_values_arr.ndim != 1 or int(parameter_values_arr.shape[0]) != len(parameter_set.specs):
        raise ValueError(
            "parameter_values must be a 1D vector matching the reverse-AD parameter set; "
            f"got shape={parameter_values_arr.shape}, parameter_count={len(parameter_set.specs)}."
        )
    baseline_geometry_deltas = jnp.asarray(baseline_geometry_deltas, dtype=jnp.float64)
    if baseline_geometry_deltas.ndim != 1 or int(baseline_geometry_deltas.shape[0]) != len(vmec_specs):
        raise ValueError(
            "baseline_geometry_deltas must match VMEC boundary parameter count; "
            f"got shape={baseline_geometry_deltas.shape}, vmec_parameter_count={len(vmec_specs)}."
        )

    support_payload = support_payload_override
    if support_payload is None:
        support_payload = find_ntx_support_payload(runtime)
    if not isinstance(support_payload, dict):
        support_payload = {
            "geometry": runtime.geometry,
            "ntx_support": support_payload,
        }
    vmec_parameter_values = jnp.asarray(
        [
            parameter_values_arr[i]
            for i, spec in enumerate(parameter_set.specs)
            if isinstance(spec, VmecBoundaryParameterSpec)
        ],
        dtype=jnp.float64,
    )
    use_runtime_payload = raw_block_solve is None or support_payload_override is not None
    if raw_block_solve is not None:
        try:
            use_runtime_payload = bool(
                jnp.allclose(vmec_parameter_values, baseline_geometry_deltas).item()
            )
        except Exception:
            use_runtime_payload = False

    profile_specs = tuple(parameter_set.profile_specs)

    def _initial_er_payload_cotangents_for_current_geometry():
        if use_runtime_payload:
            baseline_geometry = support_payload["geometry"]
            baseline_ntx_support = support_payload["ntx_support"]
        else:
            current_payload = build_neopax_geometry_and_ntx_exact_lij_support_from_state(
                geometry_context,
                raw_block_solve.state,
                n_r=int(n_r),
                n_theta=int(n_theta),
                n_zeta=int(n_zeta),
                n_xi=int(n_xi),
                surface_backend=str(surface_backend),
            )
            baseline_geometry = current_payload["geometry"]
            baseline_ntx_support = current_payload["ntx_support"]
        runtime_for_geometry = runtime_with_geometry_payload(runtime, baseline_geometry)
        runtime_for_geometry = runtime_with_ntx_support_payload(runtime_for_geometry, baseline_ntx_support)
        _probe("after_support_payload_setup")
        geometry_delta0 = _float_delta_tree_like(baseline_geometry)

        pre_root_state = pre_root_state_from_profile_values(profile_values_arr)
        _probe("before_selected_root")
        er_profile, finite_mask = initial_er_selected_root_profile(
            pre_root_state,
            config=dict(config),
            runtime=runtime_for_geometry,
        )
        _probe("after_selected_root")
        er_profile = jnp.asarray(er_profile, dtype=pre_root_state.Er.dtype)
        finite_mask = jnp.asarray(finite_mask, dtype=bool)
        rooted_state = dataclasses.replace(pre_root_state, Er=er_profile)

        generic_objectives = tuple(name for name in requested_objectives if name != _BOOTSTRAP_CURRENT_OBJECTIVE)

        def _values_from_rooted_state_and_geometry(state_value, geometry_delta):
            geometry = _add_float_delta_tree(baseline_geometry, geometry_delta)
            runtime_with_geometry = runtime_with_geometry_payload(runtime_for_geometry, geometry)
            runtime_with_geometry = runtime_with_ntx_support_payload(runtime_with_geometry, baseline_ntx_support)
            return _initial_er_root_only_objective_values(
                state_value,
                runtime_with_geometry,
                generic_objectives,
                options=options,
            )

        objective_count = len(requested_objectives)
        generic_values = None
        generic_rooted_state_bars = None
        generic_direct_geometry_bars = None
        generic_value_lookup = {name: i for i, name in enumerate(generic_objectives)}
        if generic_objectives:
            _probe("before_generic_objective_vjp")
            generic_values, objective_pullback = jax.vjp(
                _values_from_rooted_state_and_geometry,
                rooted_state,
                geometry_delta0,
            )
            _probe("after_generic_objective_vjp")
            generic_basis = jnp.eye(len(generic_objectives), dtype=jnp.asarray(generic_values).dtype)
            _probe("before_generic_objective_vmap")
            generic_rooted_state_bars, generic_direct_geometry_bars = jax.vmap(
                lambda cotangent: objective_pullback(cotangent)
            )(generic_basis)
            _probe("after_generic_objective_vmap")

        bootstrap_row = None
        if _BOOTSTRAP_CURRENT_OBJECTIVE in requested_objectives:
            _probe("before_bootstrap_cotangent")
            bootstrap_row = _compact_bootstrap_current_root_objective_cotangent(
                rooted_state=rooted_state,
                runtime_for_geometry=runtime_for_geometry,
                baseline_geometry=baseline_geometry,
                baseline_ntx_support=baseline_ntx_support,
                geometry_delta0=geometry_delta0,
                dispatch_cache_probe=dispatch_cache_probe,
            )
            _probe("after_bootstrap_cotangent")

        objective_value_rows = []
        rooted_state_bar_rows = []
        direct_geometry_bar_rows = []
        direct_ntx_support_bar_rows = []
        zero_ntx_support_bar = _float_delta_tree_like(baseline_ntx_support)
        for name in requested_objectives:
            if name == _BOOTSTRAP_CURRENT_OBJECTIVE:
                if bootstrap_row is None:
                    raise AssertionError("bootstrap cotangent row was not built.")
                value, rooted_state_bar, direct_geometry_bar, direct_support_bar = bootstrap_row
                objective_value_rows.append(value)
                rooted_state_bar_rows.append(rooted_state_bar)
                direct_geometry_bar_rows.append(direct_geometry_bar)
                direct_ntx_support_bar_rows.append(direct_support_bar)
                continue
            if generic_values is None or generic_rooted_state_bars is None or generic_direct_geometry_bars is None:
                raise AssertionError("generic root objective rows were not built.")
            generic_i = generic_value_lookup[name]
            objective_value_rows.append(generic_values[generic_i])
            rooted_state_bar_rows.append(_row_from_batched_tree(generic_rooted_state_bars, generic_i))
            direct_geometry_bar_rows.append(_row_from_batched_tree(generic_direct_geometry_bars, generic_i))
            direct_ntx_support_bar_rows.append(zero_ntx_support_bar)

        objective_values = jnp.stack(objective_value_rows)
        rooted_state_bars = _stack_tree_rows(rooted_state_bar_rows)
        direct_geometry_bars = _stack_tree_rows(direct_geometry_bar_rows)
        direct_ntx_support_bars = _stack_tree_rows(direct_ntx_support_bar_rows)

        dres_der = initial_er_charge_flux_residual_er_derivative(
            pre_root_state,
            er_profile,
            runtime=runtime_for_geometry,
        )
        safe_dres_der = jnp.where(
            jnp.abs(dres_der) > jnp.asarray(1.0e-30, dtype=dres_der.dtype),
            dres_der,
            jnp.inf,
        )
        residual_bars = jnp.where(
            finite_mask[None, :],
            -jnp.asarray(rooted_state_bars.Er) / safe_dres_der[None, :],
            0.0,
        )
        _probe("before_root_state_pullback")
        state_residual_bars = compact_initial_er_state_pullback(
            residual_scalar_fn=initial_er_charge_flux_residual_scalar,
            state=pre_root_state,
            er_profile=er_profile,
            residual_bars=residual_bars,
            runtime=runtime_for_geometry,
        )
        _probe("after_root_state_pullback")
        direct_pre_root_state_bars = dataclasses.replace(
            rooted_state_bars,
            Er=jnp.zeros_like(rooted_state_bars.Er),
        )
        pre_root_state_bars = _add_trees(direct_pre_root_state_bars, state_residual_bars)

        if profile_specs:
            _probe("before_profile_pullback")
            _, profile_pullback = jax.vjp(
                pre_root_state_from_profile_values,
                profile_values_arr,
            )
            profile_gradient_all = jax.vmap(lambda state_bar: profile_pullback(state_bar)[0])(
                pre_root_state_bars
            )
            _probe("after_profile_pullback")
        else:
            profile_gradient_all = jnp.zeros((objective_count, 0), dtype=jnp.asarray(objective_values).dtype)
        canonical_profile_lookup = {name: i for i, name in enumerate(PROFILE_PARAMETER_ORDER)}
        if profile_specs:
            profile_gradient_matrix_for_payload = jnp.stack(
                [profile_gradient_all[:, canonical_profile_lookup[spec.name]] for spec in profile_specs],
                axis=1,
            )
        else:
            profile_gradient_matrix_for_payload = profile_gradient_all

        def _residuals_from_geometry_delta(geometry_delta):
            geometry = _add_float_delta_tree(baseline_geometry, geometry_delta)
            runtime_with_geometry = runtime_with_geometry_payload(runtime_for_geometry, geometry)
            runtime_with_geometry = runtime_with_ntx_support_payload(runtime_with_geometry, baseline_ntx_support)
            return initial_er_charge_flux_residuals(
                pre_root_state,
                er_profile,
                runtime=runtime_with_geometry,
            )

        _probe("before_root_geometry_pullback")
        _, geometry_residual_pullback = jax.vjp(
            _residuals_from_geometry_delta,
            geometry_delta0,
        )
        _probe("after_root_geometry_vjp")
        residual_geometry_bars = jax.vmap(
            lambda residual_bar: geometry_residual_pullback(residual_bar)[0]
        )(residual_bars)
        _probe("after_root_geometry_pullback")
        geometry_bars = _add_trees(direct_geometry_bars, residual_geometry_bars)

        ntx_runtime = runtime_with_geometry_payload(runtime_for_geometry, baseline_geometry)
        _probe("before_root_ntx_support_pullback")
        ntx_bar_leaves = compact_initial_er_ntx_support_pullback_leaves(
            runtime=ntx_runtime,
            state=pre_root_state,
            er_profile=er_profile,
            residual_bars=residual_bars,
            support=baseline_ntx_support,
        )
        _probe("after_root_ntx_support_pullback")
        _, ntx_treedef = jax.tree_util.tree_flatten(baseline_ntx_support)
        ntx_bars = ntx_treedef.unflatten(tuple(ntx_bar_leaves))
        support_bars = []
        for objective_index in range(objective_count):
            geometry_bar = jax.tree_util.tree_map(lambda leaf: leaf[objective_index], geometry_bars)
            ntx_bar = jax.tree_util.tree_map(
                lambda residual_leaf, direct_leaf: residual_leaf[objective_index] + direct_leaf[objective_index],
                ntx_bars,
                direct_ntx_support_bars,
            )
            support_bars.append({"geometry": geometry_bar, "ntx_support": ntx_bar})
        return objective_values, profile_gradient_matrix_for_payload, tuple(support_bars), objective_count

    (
        objective_values,
        profile_gradient_matrix_for_payload,
        support_bars,
        objective_count,
    ) = _initial_er_payload_cotangents_for_current_geometry()
    objective_values, profile_gradient_matrix_for_payload, support_bars = jax.block_until_ready(
        (objective_values, profile_gradient_matrix_for_payload, support_bars)
    )
    _probe("after_payload_cotangents_ready")

    geometry_param_tuples = tuple(spec.as_tuple() for spec in vmec_specs)
    assembly_result = realtime_geometry_transport_reverse_table_from_payload_cotangents(
        objective_labels=requested_objectives,
        profile_parameter_labels=tuple(spec.name for spec in profile_specs),
        geometry_parameter_labels=tuple(spec.label for spec in vmec_specs),
        objective_values=objective_values,
        profile_gradient_matrix=profile_gradient_matrix_for_payload,
        geometry_context=geometry_context,
        baseline_geometry_deltas=baseline_geometry_deltas,
        geometry_param_specs=geometry_param_tuples,
        support_bars=support_bars,
        support_component_bars_by_name={},
        include_component_pullbacks=False,
        combined_geometry_payload=True,
        n_r=int(n_r),
        n_theta=int(n_theta),
        n_zeta=int(n_zeta),
        n_xi=int(n_xi),
        surface_backend=str(surface_backend),
        max_iter=max_iter,
        solver_device=solver_device,
        progress_label=progress_label,
        raw_block_solve=raw_block_solve,
        return_branch_gradients=False,
    )

    geometry_gradient_matrix = jnp.asarray(assembly_result.table_result.geometry_gradient_matrix)
    profile_gradient_matrix = jnp.asarray(assembly_result.table_result.profile_gradient_matrix)
    jacobian_rows = []
    profile_lookup = {spec.name: i for i, spec in enumerate(profile_specs)}
    geometry_lookup = {spec.label: i for i, spec in enumerate(vmec_specs)}
    for _row_i in range(objective_count):
        columns = []
        for spec in parameter_set.specs:
            if isinstance(spec, ProfileParameterSpec):
                columns.append(profile_gradient_matrix[_row_i, profile_lookup[spec.name]])
            elif isinstance(spec, VmecBoundaryParameterSpec):
                columns.append(geometry_gradient_matrix[_row_i, geometry_lookup[spec.label]])
            else:
                raise TypeError(f"Unsupported reverse-AD parameter spec type: {type(spec).__name__}.")
        jacobian_rows.append(jnp.stack(columns))

    return ObjectiveTableResult(
        objective_names=requested_objectives,
        values=objective_values,
        jacobian=jnp.stack(jacobian_rows, axis=0),
    )


def _optimization_payload_to_vmec_table(
    *,
    objective_labels, profile_parameter_labels, geometry_parameter_labels,
    objective_values, profile_gradient_matrix, geometry_context,
    baseline_geometry_deltas, geometry_param_specs, support_bars,
    support_component_bars_by_name, include_component_pullbacks,
    combined_geometry_payload, return_branch_gradients,
    n_r, n_theta, n_zeta, n_xi, surface_backend, max_iter,
    solver_device, progress_label, raw_block_solve,
):
    """Optimization-only boundary around the established payload pullback."""
    return realtime_geometry_transport_reverse_table_from_payload_cotangents(
        objective_labels=objective_labels,
        profile_parameter_labels=profile_parameter_labels,
        geometry_parameter_labels=geometry_parameter_labels,
        objective_values=objective_values,
        profile_gradient_matrix=profile_gradient_matrix,
        geometry_context=geometry_context,
        baseline_geometry_deltas=baseline_geometry_deltas,
        geometry_param_specs=geometry_param_specs,
        support_bars=support_bars,
        support_component_bars_by_name=support_component_bars_by_name,
        include_component_pullbacks=include_component_pullbacks,
        combined_geometry_payload=combined_geometry_payload,
        n_r=int(n_r), n_theta=int(n_theta), n_zeta=int(n_zeta), n_xi=int(n_xi),
        surface_backend=str(surface_backend), max_iter=max_iter,
        solver_device=solver_device, progress_label=progress_label,
        raw_block_solve=raw_block_solve, return_branch_gradients=return_branch_gradients,
    )


def _optimization_root_to_payload_cotangents(
    *, config, requested_objectives, runtime, profile_values_arr,
    pre_root_state_from_profile_values, geometry_context, n_r, n_theta,
    n_zeta, n_xi, surface_backend, raw_block_solve, support_payload,
    use_runtime_payload, profile_specs, options, boozer_surface_sampling=None,
    r00_boozer_surface_sampling=None,
    transport_reverse_stage: InitialErTransportReverseStage | None = None,
):
    transport_payload_adapter = (
        None if transport_reverse_stage is None else transport_reverse_stage.payload_adapter
    )
    if use_runtime_payload:
        baseline_geometry = support_payload["geometry"]
        baseline_ntx_support = support_payload["ntx_support"]
    else:
        current_payload = build_neopax_geometry_and_ntx_exact_lij_support_from_state(
            geometry_context,
            raw_block_solve.state,
            n_r=int(n_r),
            n_theta=int(n_theta),
            n_zeta=int(n_zeta),
            n_xi=int(n_xi),
            surface_backend=str(surface_backend),
            boozer_surface_sampling=boozer_surface_sampling,
            r00_boozer_surface_sampling=r00_boozer_surface_sampling,
        )
        baseline_geometry = current_payload["geometry"]
        baseline_ntx_support = current_payload["ntx_support"]
    if transport_payload_adapter is not None:
        stage_payload = transport_payload_adapter.rebuild(
            *transport_payload_adapter.dynamic_leaves(
                {"geometry": baseline_geometry, "ntx_support": baseline_ntx_support}
            )
        )
        baseline_geometry = stage_payload["geometry"]
        baseline_ntx_support = stage_payload["ntx_support"]
        geometry_leaves, support_leaves = transport_payload_adapter.dynamic_leaves(
            {"geometry": baseline_geometry, "ntx_support": baseline_ntx_support}
        )
    else:
        geometry_leaves = support_leaves = None
    runtime_for_geometry = runtime_with_geometry_payload(runtime, baseline_geometry)
    runtime_for_geometry = runtime_with_ntx_support_payload(runtime_for_geometry, baseline_ntx_support)
    geometry_delta0 = _float_delta_tree_like(baseline_geometry)
    use_direct_er_rows = transport_payload_adapter is not None

    pre_root_state = pre_root_state_from_profile_values(profile_values_arr)
    if transport_reverse_stage is None:
        er_profile, finite_mask = initial_er_selected_root_profile(
            pre_root_state,
            config=dict(config),
            runtime=runtime_for_geometry,
        )
    else:
        er_profile, finite_mask = transport_reverse_stage.selected_root(
            pre_root_state,
            geometry_leaves,
            support_leaves,
        )
    er_profile = jnp.asarray(er_profile, dtype=pre_root_state.Er.dtype)
    finite_mask = jnp.asarray(finite_mask, dtype=bool)
    rooted_state = dataclasses.replace(pre_root_state, Er=er_profile)

    generic_objectives = tuple(name for name in requested_objectives if name != _BOOTSTRAP_CURRENT_OBJECTIVE)

    def _values_from_rooted_state_and_geometry(state_value, geometry_delta):
        geometry = _add_float_delta_tree(baseline_geometry, geometry_delta)
        runtime_with_geometry = runtime_with_geometry_payload(runtime_for_geometry, geometry)
        runtime_with_geometry = runtime_with_ntx_support_payload(runtime_with_geometry, baseline_ntx_support)
        return _initial_er_root_only_objective_values(
            state_value,
            runtime_with_geometry,
            generic_objectives,
            options=options,
        )

    objective_count = len(requested_objectives)
    generic_values = None
    generic_rooted_state_bars = None
    generic_direct_geometry_bars = None
    generic_value_lookup = {name: i for i, name in enumerate(generic_objectives)}
    if generic_objectives and use_direct_er_rows and all(
        name in _DIRECT_INITIAL_ER_OBJECTIVES for name in generic_objectives
    ):
        generic_values, generic_rooted_state_bars = _direct_initial_er_objective_values_and_state_bars(
            rooted_state,
            generic_objectives,
            options=options,
        )
        generic_direct_geometry_bars = _float_delta_tree_like(baseline_geometry)
        generic_direct_geometry_bars = jax.tree_util.tree_map(
            lambda leaf: jnp.broadcast_to(
                leaf,
                (len(generic_objectives),) + jnp.asarray(leaf).shape,
            ),
            generic_direct_geometry_bars,
        )
    elif generic_objectives:
        generic_values, objective_pullback = jax.vjp(
            _values_from_rooted_state_and_geometry,
            rooted_state,
            geometry_delta0,
        )
        generic_basis = jnp.eye(len(generic_objectives), dtype=jnp.asarray(generic_values).dtype)
        generic_rooted_state_bars, generic_direct_geometry_bars = jax.vmap(
            lambda cotangent: objective_pullback(cotangent)
        )(generic_basis)

    bootstrap_row = None
    if _BOOTSTRAP_CURRENT_OBJECTIVE in requested_objectives:
        bootstrap_row = _compact_bootstrap_current_root_objective_cotangent(
            rooted_state=rooted_state,
            runtime_for_geometry=runtime_for_geometry,
            baseline_geometry=baseline_geometry,
            baseline_ntx_support=baseline_ntx_support,
            geometry_delta0=geometry_delta0,
        )

    objective_value_rows = []
    rooted_state_bar_rows = []
    direct_geometry_bar_rows = []
    direct_ntx_support_bar_rows = []
    zero_ntx_support_bar = _float_delta_tree_like(baseline_ntx_support)
    for name in requested_objectives:
        if name == _BOOTSTRAP_CURRENT_OBJECTIVE:
            if bootstrap_row is None:
                raise AssertionError("bootstrap cotangent row was not built.")
            value, rooted_state_bar, direct_geometry_bar, direct_support_bar = bootstrap_row
            objective_value_rows.append(value)
            rooted_state_bar_rows.append(rooted_state_bar)
            direct_geometry_bar_rows.append(direct_geometry_bar)
            direct_ntx_support_bar_rows.append(direct_support_bar)
            continue
        if generic_values is None or generic_rooted_state_bars is None or generic_direct_geometry_bars is None:
            raise AssertionError("generic root objective rows were not built.")
        generic_i = generic_value_lookup[name]
        objective_value_rows.append(generic_values[generic_i])
        rooted_state_bar_rows.append(_row_from_batched_tree(generic_rooted_state_bars, generic_i))
        direct_geometry_bar_rows.append(_row_from_batched_tree(generic_direct_geometry_bars, generic_i))
        direct_ntx_support_bar_rows.append(zero_ntx_support_bar)

    objective_values = jnp.stack(objective_value_rows)
    rooted_state_bars = _stack_tree_rows(rooted_state_bar_rows)
    direct_geometry_bars = _stack_tree_rows(direct_geometry_bar_rows)
    direct_ntx_support_bars = _stack_tree_rows(direct_ntx_support_bar_rows)

    dres_der = initial_er_charge_flux_residual_er_derivative(
        pre_root_state,
        er_profile,
        runtime=runtime_for_geometry,
    )
    safe_dres_der = jnp.where(
        jnp.abs(dres_der) > jnp.asarray(1.0e-30, dtype=dres_der.dtype),
        dres_der,
        jnp.inf,
    )
    residual_bars = jnp.where(
        finite_mask[None, :],
        -jnp.asarray(rooted_state_bars.Er) / safe_dres_der[None, :],
        0.0,
    )
    if transport_reverse_stage is None:
        state_residual_bars = compact_initial_er_state_pullback(
            residual_scalar_fn=initial_er_charge_flux_residual_scalar,
            state=pre_root_state,
            er_profile=er_profile,
            residual_bars=residual_bars,
            runtime=runtime_for_geometry,
        )
    else:
        state_residual_bars = transport_reverse_stage.state_pullback(
            pre_root_state,
            er_profile,
            residual_bars,
            geometry_leaves,
            support_leaves,
        )
    direct_pre_root_state_bars = dataclasses.replace(
        rooted_state_bars,
        Er=jnp.zeros_like(rooted_state_bars.Er),
    )
    pre_root_state_bars = _add_trees(direct_pre_root_state_bars, state_residual_bars)

    if profile_specs:
        _, profile_pullback = jax.vjp(
            pre_root_state_from_profile_values,
            profile_values_arr,
        )
        profile_gradient_all = jax.vmap(lambda state_bar: profile_pullback(state_bar)[0])(
            pre_root_state_bars
        )
    else:
        profile_gradient_all = jnp.zeros((objective_count, 0), dtype=jnp.asarray(objective_values).dtype)
    canonical_profile_lookup = {name: i for i, name in enumerate(PROFILE_PARAMETER_ORDER)}
    if profile_specs:
        profile_gradient_matrix_for_payload = jnp.stack(
            [profile_gradient_all[:, canonical_profile_lookup[spec.name]] for spec in profile_specs],
            axis=1,
        )
    else:
        profile_gradient_matrix_for_payload = profile_gradient_all

    def _residuals_from_geometry_delta(geometry_delta):
        geometry = _add_float_delta_tree(baseline_geometry, geometry_delta)
        runtime_with_geometry = runtime_with_geometry_payload(runtime_for_geometry, geometry)
        runtime_with_geometry = runtime_with_ntx_support_payload(runtime_with_geometry, baseline_ntx_support)
        return initial_er_charge_flux_residuals(
            pre_root_state,
            er_profile,
            runtime=runtime_with_geometry,
        )

    if transport_reverse_stage is None:
        _, geometry_residual_pullback = jax.vjp(
            _residuals_from_geometry_delta,
            geometry_delta0,
        )
        residual_geometry_bars = jax.vmap(
            lambda residual_bar: geometry_residual_pullback(residual_bar)[0]
        )(residual_bars)
    else:
        residual_geometry_bars = transport_reverse_stage.geometry_pullback(
            pre_root_state,
            er_profile,
            residual_bars,
            geometry_delta0,
            geometry_leaves,
            support_leaves,
        )
    geometry_bars = _add_trees(direct_geometry_bars, residual_geometry_bars)

    ntx_runtime = runtime_with_geometry_payload(runtime_for_geometry, baseline_geometry)
    if transport_reverse_stage is None:
        ntx_bar_leaves = compact_initial_er_ntx_support_pullback_leaves(
            runtime=ntx_runtime,
            state=pre_root_state,
            er_profile=er_profile,
            residual_bars=residual_bars,
            support=baseline_ntx_support,
        )
    else:
        ntx_bar_leaves = transport_reverse_stage.support_pullback(
            pre_root_state,
            er_profile,
            residual_bars,
            geometry_leaves,
            support_leaves,
        )
    _, ntx_treedef = jax.tree_util.tree_flatten(baseline_ntx_support)
    ntx_bars = ntx_treedef.unflatten(tuple(ntx_bar_leaves))
    support_bars = []
    for objective_index in range(objective_count):
        geometry_bar = jax.tree_util.tree_map(lambda leaf: leaf[objective_index], geometry_bars)
        ntx_bar = jax.tree_util.tree_map(
            lambda residual_leaf, direct_leaf: residual_leaf[objective_index] + direct_leaf[objective_index],
            ntx_bars,
            direct_ntx_support_bars,
        )
        support_bars.append({"geometry": geometry_bar, "ntx_support": ntx_bar})
    return objective_values, profile_gradient_matrix_for_payload, tuple(support_bars), objective_count




def geometry_active_initial_er_root_only_reverse_table_optimization(
    *,
    config: Mapping[str, object],
    objective_names: Sequence[str] | str,
    parameter_set: ReverseADParameterSet,
    parameter_values,
    runtime,
    profile_values,
    pre_root_state_from_profile_values: Callable[[object], object],
    geometry_context,
    baseline_geometry_deltas,
    n_r: int,
    n_theta: int,
    n_zeta: int,
    n_xi: int,
    surface_backend: str = "vmec",
    max_iter: int | None = None,
    solver_device: str | None = "default",
    progress_label: str | None = None,
    raw_block_solve=None,
    support_payload_override=None,
    options: Mapping[str, object] | None = None,
    optimization_stage=None,
    transport_reverse_stage: InitialErTransportReverseStage | None = None,
    payload_assembly_stage=None,
) -> ObjectiveTableResult:
    """Return compact initial-Er objective table for active realtime geometry.

    This is the internalized version of the benchmark's geometry-active
    `--initial-Er-root-only-optimization-smoke` path.  It keeps the selected
    ambipolar branch fixed, uses the compact root residual transposes, and maps
    geometry/support payload cotangents to VMEC boundary columns with the same
    raw-block payload pullback used by the realtime reverse benchmark.
    """

    requested_objectives = normalize_initial_er_root_only_objective_names(objective_names)
    parameter_values_arr = jnp.asarray(parameter_values)
    profile_values_arr = jnp.asarray(profile_values)
    vmec_specs = tuple(parameter_set.vmec_boundary_specs)
    if not vmec_specs:
        raise ValueError("geometry_active_initial_er_root_only_reverse_table requires VMEC boundary parameters.")
    if parameter_values_arr.ndim != 1 or int(parameter_values_arr.shape[0]) != len(parameter_set.specs):
        raise ValueError(
            "parameter_values must be a 1D vector matching the reverse-AD parameter set; "
            f"got shape={parameter_values_arr.shape}, parameter_count={len(parameter_set.specs)}."
        )
    baseline_geometry_deltas = jnp.asarray(baseline_geometry_deltas, dtype=jnp.float64)
    if baseline_geometry_deltas.ndim != 1 or int(baseline_geometry_deltas.shape[0]) != len(vmec_specs):
        raise ValueError(
            "baseline_geometry_deltas must match VMEC boundary parameter count; "
            f"got shape={baseline_geometry_deltas.shape}, vmec_parameter_count={len(vmec_specs)}."
        )

    support_payload = support_payload_override
    if support_payload is None:
        support_payload = find_ntx_support_payload(runtime)
    if not isinstance(support_payload, dict):
        support_payload = {
            "geometry": runtime.geometry,
            "ntx_support": support_payload,
        }
    vmec_parameter_values = jnp.asarray(
        [
            parameter_values_arr[i]
            for i, spec in enumerate(parameter_set.specs)
            if isinstance(spec, VmecBoundaryParameterSpec)
        ],
        dtype=jnp.float64,
    )
    use_runtime_payload = raw_block_solve is None or support_payload_override is not None
    if raw_block_solve is not None:
        try:
            use_runtime_payload = bool(
                jnp.allclose(vmec_parameter_values, baseline_geometry_deltas).item()
            )
        except Exception:
            use_runtime_payload = False

    profile_specs = tuple(parameter_set.profile_specs)

    root_operator = _optimization_root_to_payload_cotangents if optimization_stage is None else optimization_stage.root_to_payload
    dynamic_payload = None if raw_block_solve is None else raw_block_dynamic_payload(raw_block_solve)
    root_args = dict(
        config=config,
        requested_objectives=requested_objectives,
        runtime=runtime,
        profile_values_arr=profile_values_arr,
        pre_root_state_from_profile_values=pre_root_state_from_profile_values,
        geometry_context=geometry_context,
        n_r=n_r, n_theta=n_theta, n_zeta=n_zeta, n_xi=n_xi,
        surface_backend=surface_backend,
        support_payload=support_payload,
        use_runtime_payload=use_runtime_payload,
        profile_specs=profile_specs,
        options=options,
        transport_reverse_stage=transport_reverse_stage,
    )
    if optimization_stage is None:
        root_args["raw_block_solve"] = raw_block_solve
        root_result = root_operator(**root_args)
    else:
        root_result = root_operator(dynamic_payload, profile_values_arr)
    (
        objective_values, profile_gradient_matrix_for_payload, support_bars, objective_count,
    ) = root_result
    objective_values, profile_gradient_matrix_for_payload, support_bars = jax.block_until_ready(
        (objective_values, profile_gradient_matrix_for_payload, support_bars)
    )

    geometry_param_tuples = tuple(spec.as_tuple() for spec in vmec_specs)
    payload_args = dict(
        objective_labels=requested_objectives,
        profile_parameter_labels=tuple(spec.name for spec in profile_specs),
        geometry_parameter_labels=tuple(spec.label for spec in vmec_specs),
        objective_values=objective_values,
        profile_gradient_matrix=profile_gradient_matrix_for_payload,
        geometry_context=geometry_context,
        baseline_geometry_deltas=baseline_geometry_deltas,
        geometry_param_specs=geometry_param_tuples,
        support_bars=support_bars,
        support_component_bars_by_name={},
        include_component_pullbacks=False,
        combined_geometry_payload=True,
        n_r=int(n_r),
        n_theta=int(n_theta),
        n_zeta=int(n_zeta),
        n_xi=int(n_xi),
        surface_backend=str(surface_backend),
        max_iter=max_iter,
        solver_device=solver_device,
        progress_label=progress_label,
        raw_block_solve=raw_block_solve,
        return_branch_gradients=False,
    )
    payload_operator = (
        _optimization_payload_to_vmec_table
        if optimization_stage is None and payload_assembly_stage is None
        else (optimization_stage.payload_to_vmec if optimization_stage is not None else payload_assembly_stage.payload_to_vmec)
    )
    if optimization_stage is None and payload_assembly_stage is None:
        assembly_result = payload_operator(**payload_args)
    else:
        assembly_result = payload_operator(
            dynamic_payload,
            baseline_geometry_deltas,
            objective_values,
            profile_gradient_matrix_for_payload,
            support_bars,
        )

    geometry_gradient_matrix = jnp.asarray(assembly_result.table_result.geometry_gradient_matrix)
    profile_gradient_matrix = jnp.asarray(assembly_result.table_result.profile_gradient_matrix)
    jacobian_rows = []
    profile_lookup = {spec.name: i for i, spec in enumerate(profile_specs)}
    geometry_lookup = {spec.label: i for i, spec in enumerate(vmec_specs)}
    for _row_i in range(objective_count):
        columns = []
        for spec in parameter_set.specs:
            if isinstance(spec, ProfileParameterSpec):
                columns.append(profile_gradient_matrix[_row_i, profile_lookup[spec.name]])
            elif isinstance(spec, VmecBoundaryParameterSpec):
                columns.append(geometry_gradient_matrix[_row_i, geometry_lookup[spec.label]])
            else:
                raise TypeError(f"Unsupported reverse-AD parameter spec type: {type(spec).__name__}.")
        jacobian_rows.append(jnp.stack(columns))

    return ObjectiveTableResult(
        objective_names=requested_objectives,
        values=objective_values,
        jacobian=jnp.stack(jacobian_rows, axis=0),
    )



def evaluate_geometry_initial_er_root_only_least_squares(
    config: Mapping[str, object],
    *,
    parameter_set: ReverseADParameterSet,
    parameter_values,
    terms: Sequence[LeastSquaresTerm | tuple[ObjectiveRef | str, float, float]],
    geometry_context,
    runtime=None,
    rooted_state_from_parameter_vector: InitialErRootOnlyStateBuilder | None = None,
    options: Mapping[str, object] | None = None,
    root_options: Mapping[str, object] | None = None,
    geometry_lane: str = "ad",
    geometry_max_iter: int | None = None,
    geometry_step_size: float | None = None,
    geometry_final_vmec_pullback_mode: str = "raw_block_transpose",
    geometry_solver_device: str | None = "default",
) -> LeastSquaresEvaluation:
    """Evaluate grouped geometry objectives plus initial-Er root objectives.

    This is the optimization-facing composition layer.  It intentionally routes
    each family through the already validated grouped reverse table backend:
    one geometry table for all requested VMEC/Boozer objectives and one compact
    initial-Er table for all requested Er-root objectives.
    """

    normalized_terms = normalize_least_squares_terms(terms)
    grouped_terms = group_least_squares_terms_by_family(normalized_terms)
    unsupported_families = tuple(
        family for family in grouped_terms if family not in {"geometry", "transport"}
    )
    if unsupported_families:
        raise NotImplementedError(
            "evaluate_geometry_initial_er_root_only_least_squares supports only "
            f"geometry and transport/initial-Er terms; got families {unsupported_families!r}."
        )
    if "transport" in grouped_terms and (runtime is None or rooted_state_from_parameter_vector is None):
        raise ValueError(
            "Transport initial-Er terms require both runtime and "
            "rooted_state_from_parameter_vector."
        )

    parameter_values_arr = jnp.asarray(parameter_values)
    backend_options = {} if options is None else dict(options)
    backend_options.setdefault("parameter_values", parameter_values_arr)
    backends: dict[ObjectiveFamily, ObjectiveTableBackend] = {}
    if "geometry" in grouped_terms:
        backends["geometry"] = geometry_full_ad_reverse_table_backend(
            context=geometry_context,
            parameter_values=parameter_values_arr,
            lane=geometry_lane,
            max_iter=geometry_max_iter,
            step_size=geometry_step_size,
            final_vmec_pullback_mode=geometry_final_vmec_pullback_mode,
            solver_device=geometry_solver_device,
        )
    if "transport" in grouped_terms:
        root_runner_options = {} if root_options is None else dict(root_options)

        def _transport_backend(
            objective_names: tuple[str, ...],
            parameter_set_inner: ReverseADParameterSet,
            options_inner: Mapping[str, object],
        ) -> ObjectiveTableResult:
            del options_inner
            request = InitialErRootOnlyReverseTableRequest(
                objective_names=normalize_initial_er_root_only_objective_names(objective_names),
                parameter_set=parameter_set_inner,
                parameter_values=parameter_values_arr,
                runtime=runtime,
                rooted_state_from_parameter_vector=rooted_state_from_parameter_vector,
                options=root_runner_options,
            )
            return initial_er_root_only_reverse_table(request)

        backends["transport"] = _transport_backend

    t_start = time.perf_counter()
    result = residuals_and_jacobian_reverse_ad(
        config,
        parameter_set=parameter_set,
        terms=normalized_terms,
        backends=backends,
        options=backend_options,
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


def _active_profile_values_from_parameter_vector(
    parameter_set: ReverseADParameterSet,
    parameter_values,
    baseline_profile_values,
):
    """Return full profile vector with active profile DOFs overwritten."""

    values = jnp.asarray(parameter_values)
    profiles = jnp.asarray(baseline_profile_values)
    if values.ndim != 1 or int(values.shape[0]) != len(parameter_set.specs):
        raise ValueError(
            "parameter_values must be a 1D vector matching the reverse-AD parameter set; "
            f"got shape={values.shape}, parameter_count={len(parameter_set.specs)}."
        )
    if profiles.ndim != 1 or int(profiles.shape[0]) != len(PROFILE_PARAMETER_ORDER):
        raise ValueError(
            "baseline_profile_values must follow PROFILE_PARAMETER_ORDER; "
            f"got shape={profiles.shape}, expected ({len(PROFILE_PARAMETER_ORDER)},)."
        )
    profile_lookup = {name: i for i, name in enumerate(PROFILE_PARAMETER_ORDER)}
    for value, spec in zip(values, parameter_set.specs, strict=True):
        if isinstance(spec, ProfileParameterSpec):
            profiles = profiles.at[profile_lookup[spec.name]].set(value)
    return profiles


def evaluate_geometry_initial_er_root_only_least_squares_fused(
    config: Mapping[str, object],
    *,
    parameter_set: ReverseADParameterSet,
    parameter_values,
    terms: Sequence[LeastSquaresTerm | tuple[ObjectiveRef | str, float, float]],
    geometry_context,
    runtime,
    baseline_profile_values,
    pre_root_state_from_profile_values: Callable[[object], object],
    n_r: int,
    n_theta: int,
    n_zeta: int,
    n_xi: int,
    surface_backend: str = "vmec",
    root_options: Mapping[str, object] | None = None,
    geometry_lane: str = "ad",
    geometry_max_iter: int | None = None,
    geometry_step_size: float | None = None,
    geometry_solver_device: str | None = "default",
) -> LeastSquaresEvaluation:
    """Evaluate geometry + initial-Er root terms with one fused VMEC pullback."""

    root_runner_options = {} if root_options is None else dict(root_options)
    normalized_terms = normalize_least_squares_terms(terms)
    grouped_terms = group_least_squares_terms_by_family(normalized_terms)
    unsupported_families = tuple(
        family for family in grouped_terms if family not in {"geometry", "transport"}
    )
    if unsupported_families:
        raise NotImplementedError(
            "evaluate_geometry_initial_er_root_only_least_squares_fused supports only "
            f"geometry and transport/initial-Er terms; got families {unsupported_families!r}."
        )
    if "geometry" in grouped_terms and not parameter_set.vmec_boundary_specs:
        raise ValueError("Geometry objectives require VMEC boundary parameters.")

    parameter_values_arr = jnp.asarray(parameter_values, dtype=jnp.float64)
    t_start = time.perf_counter()
    shared_payload = None

    cotangent_tables_by_family: dict[ObjectiveFamily, ObjectiveCotangentTable] = {}
    backend_results: dict[ObjectiveFamily, ObjectiveTableResult] = {}
    geometry_cotangent_tables: list[ObjectiveCotangentTable] = []
    deferred_geometry_terms = grouped_terms.get("geometry", ())
    deferred_geometry_table_holder: dict[str, ObjectiveCotangentTable] = {}
    if "transport" in grouped_terms:
        active_profile_values = _active_profile_values_from_parameter_vector(
            parameter_set,
            parameter_values_arr,
            baseline_profile_values,
        )
        if not parameter_set.vmec_boundary_specs:
            transport_table = initial_er_root_only_objective_cotangent_table(
                config=config,
                objective_names=_unique_objective_names(grouped_terms["transport"]),
                parameter_set=parameter_set,
                parameter_values=parameter_values_arr,
                runtime=runtime,
                profile_values=active_profile_values,
                pre_root_state_from_profile_values=pre_root_state_from_profile_values,
                options=root_runner_options,
            )
            cotangent_tables_by_family["transport"] = transport_table
        else:
            vmec_parameter_values = vmec_parameter_values_from_parameter_vector(
                parameter_set,
                parameter_values_arr,
            )
            shared_payload = build_shared_geometry_transport_payload(
                geometry_context=geometry_context,
                parameter_set=parameter_set,
                parameter_values=parameter_values_arr,
                runtime=runtime,
                n_r=int(n_r),
                n_theta=int(n_theta),
                n_zeta=int(n_zeta),
                n_xi=int(n_xi),
                surface_backend=str(surface_backend),
                max_iter=geometry_max_iter,
                solver_device=geometry_solver_device,
            )
            current_support_payload = None
            try:
                use_runtime_payload = bool(
                    jnp.allclose(
                        vmec_parameter_values,
                        jnp.zeros_like(vmec_parameter_values),
                    ).item()
                )
            except Exception:
                use_runtime_payload = False
            if not use_runtime_payload:
                current_support_payload = build_neopax_geometry_and_ntx_exact_lij_support_from_state(
                    geometry_context,
                    shared_payload.raw_block_solve.state,
                    n_r=int(n_r),
                    n_theta=int(n_theta),
                    n_zeta=int(n_zeta),
                    n_xi=int(n_xi),
                    surface_backend=str(surface_backend),
                )
            transport_parameter_set = ReverseADParameterSet(
                profile_specs=tuple(ProfileParameterSpec(name) for name in PROFILE_PARAMETER_ORDER),
                vmec_boundary_specs=tuple(parameter_set.vmec_boundary_specs),
            )
            transport_parameter_values = jnp.concatenate(
                [
                    jnp.asarray(active_profile_values, dtype=parameter_values_arr.dtype),
                    jnp.asarray(vmec_parameter_values, dtype=parameter_values_arr.dtype),
                ],
                axis=0,
            )
            requested_transport_objectives = _unique_objective_names(grouped_terms["transport"])
            transport_table_full = initial_er_root_only_objective_cotangent_table(
                config=config,
                objective_names=requested_transport_objectives,
                parameter_set=transport_parameter_set,
                parameter_values=transport_parameter_values,
                runtime=runtime,
                profile_values=active_profile_values,
                pre_root_state_from_profile_values=pre_root_state_from_profile_values,
                support_payload_override=current_support_payload,
                options=root_runner_options,
            )
            (
                transport_values,
                transport_profile_gradient_matrix,
                transport_payload_bars,
            ) = jax.block_until_ready(
                (
                    transport_table_full.values,
                    transport_table_full.profile_gradient_matrix,
                    transport_table_full.payload_bars,
                )
            )
            del current_support_payload
            source_profile_lookup = {
                spec.name: i for i, spec in enumerate(transport_parameter_set.profile_specs)
            }
            target_profile_gradient_columns = [
                transport_profile_gradient_matrix[:, source_profile_lookup[spec.name]]
                for spec in parameter_set.profile_specs
            ]
            target_profile_gradient_matrix = (
                jnp.stack(target_profile_gradient_columns, axis=1)
                if target_profile_gradient_columns
                else jnp.zeros(
                    (len(requested_transport_objectives), 0),
                    dtype=jnp.asarray(transport_values).dtype,
                )
            )
            transport_table = ObjectiveCotangentTable(
                objective_names=transport_table_full.objective_names,
                values=transport_values,
                profile_gradient_matrix=target_profile_gradient_matrix,
                vmec_state_bars=None,
                payload_bars=transport_payload_bars,
            )
            del transport_table_full
            cotangent_tables_by_family["transport"] = transport_table
            geometry_cotangent_tables.append(transport_table)

    if "geometry" in grouped_terms and not geometry_cotangent_tables:
        if shared_payload is None and parameter_set.vmec_boundary_specs:
            shared_payload = build_shared_geometry_transport_payload(
                geometry_context=geometry_context,
                parameter_set=parameter_set,
                parameter_values=parameter_values_arr,
                runtime=runtime,
                n_r=int(n_r),
                n_theta=int(n_theta),
                n_zeta=int(n_zeta),
                n_xi=int(n_xi),
                surface_backend=str(surface_backend),
                max_iter=geometry_max_iter,
                solver_device=geometry_solver_device,
            )
        if shared_payload is None:
            raise ValueError("Geometry terms require shared VMEC payload data.")
        geometry_table = geometry_full_ad_objective_cotangent_table(
            context=geometry_context,
            parameter_set=parameter_set,
            objective_names=_unique_objective_names(grouped_terms["geometry"]),
            parameter_values=parameter_values_arr,
            shared_payload=shared_payload,
            lane=geometry_lane,
            max_iter=geometry_max_iter,
            step_size=geometry_step_size,
            solver_device=geometry_solver_device,
        )
        cotangent_tables_by_family["geometry"] = geometry_table
        geometry_cotangent_tables.append(geometry_table)

    geometry_gradient_by_table_id: dict[int, object] = {}
    if geometry_cotangent_tables:
        if shared_payload is None:
            raise ValueError("Geometry cotangents were produced without shared VMEC payload data.")

        def _deferred_geometry_state_bars(raw_block_solve):
            geometry_shared_payload = SharedGeometryTransportPayload(
                raw_block_solve=raw_block_solve,
                vmec_parameter_values=shared_payload.vmec_parameter_values,
                vmec_specs=shared_payload.vmec_specs,
            )
            geometry_table = geometry_full_ad_objective_cotangent_table(
                context=geometry_context,
                parameter_set=parameter_set,
                objective_names=_unique_objective_names(deferred_geometry_terms),
                parameter_values=parameter_values_arr,
                shared_payload=geometry_shared_payload,
                lane=geometry_lane,
                max_iter=geometry_max_iter,
                step_size=geometry_step_size,
                solver_device=geometry_solver_device,
            )
            deferred_geometry_table_holder["geometry"] = geometry_table
            return geometry_table.vmec_state_bars

        fused_geometry_matrix = fused_geometry_parameter_matrix_from_cotangent_tables(
            geometry_context=geometry_context,
            parameter_set=parameter_set,
            tables=tuple(geometry_cotangent_tables),
            shared_payload=shared_payload,
            n_r=int(n_r),
            n_theta=int(n_theta),
            n_zeta=int(n_zeta),
            n_xi=int(n_xi),
            surface_backend=str(surface_backend),
            max_iter=geometry_max_iter,
            solver_device=geometry_solver_device,
            extra_state_bars_factory=(
                None
                if not deferred_geometry_terms
                else _deferred_geometry_state_bars
            ),
        )
        if "geometry" in deferred_geometry_table_holder:
            geometry_table = deferred_geometry_table_holder["geometry"]
            cotangent_tables_by_family["geometry"] = geometry_table
            geometry_cotangent_tables.append(geometry_table)
        row0 = 0
        for table in geometry_cotangent_tables:
            row1 = row0 + len(table.objective_names)
            geometry_gradient_by_table_id[id(table)] = fused_geometry_matrix[row0:row1]
            row0 = row1

    for family, table in cotangent_tables_by_family.items():
        geometry_gradient_matrix = geometry_gradient_by_table_id.get(id(table))
        if geometry_gradient_matrix is None:
            geometry_gradient_matrix = jnp.zeros(
                (len(table.objective_names), len(parameter_set.vmec_boundary_specs)),
                dtype=jnp.asarray(table.values).dtype,
            )
        backend_results[family] = objective_table_result_from_cotangent_table(
            table,
            parameter_set=parameter_set,
            geometry_gradient_matrix=geometry_gradient_matrix,
        )

    result = assemble_least_squares_result(
        normalized_terms,
        parameter_set=parameter_set,
        backend_results=backend_results,
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


def evaluate_geometry_initial_er_root_only_least_squares_benchmark_tables(
    config: Mapping[str, object],
    *,
    parameter_set: ReverseADParameterSet,
    parameter_values,
    terms: Sequence[LeastSquaresTerm | tuple[ObjectiveRef | str, float, float]],
    geometry_context,
    runtime,
    baseline_profile_values,
    pre_root_state_from_profile_values: Callable[[object], object],
    n_r: int,
    n_theta: int,
    n_zeta: int,
    n_xi: int,
    surface_backend: str = "vmec",
    geometry_lane: str = "ad",
    geometry_max_iter: int | None = None,
    geometry_step_size: float | None = None,
    geometry_solver_device: str | None = "default",
    root_options: Mapping[str, object] | None = None,
    raw_block_stage=None,
) -> LeastSquaresEvaluation:
    """Evaluate mixed objectives using only benchmark-validated table backends."""

    normalized_terms = normalize_least_squares_terms(terms)
    grouped_terms = group_least_squares_terms_by_family(normalized_terms)
    unsupported_families = tuple(
        family for family in grouped_terms if family not in {"geometry", "transport"}
    )
    if unsupported_families:
        raise NotImplementedError(
            "evaluate_geometry_initial_er_root_only_least_squares_benchmark_tables supports only "
            f"geometry and transport/initial-Er terms; got families {unsupported_families!r}."
        )

    parameter_values_arr = jnp.asarray(parameter_values, dtype=jnp.float64)
    backend_results: dict[ObjectiveFamily, ObjectiveTableResult] = {}
    shared_raw_block_solve = None
    root_runner_options = {} if root_options is None else dict(root_options)
    t_start = time.perf_counter()

    if "transport" in grouped_terms:
        active_profile_values = _active_profile_values_from_parameter_vector(
            parameter_set,
            parameter_values_arr,
            baseline_profile_values,
        )
        requested_transport_objectives = _unique_objective_names(grouped_terms["transport"])
        if parameter_set.vmec_boundary_specs:
            vmec_parameter_values = vmec_parameter_values_from_parameter_vector(
                parameter_set,
                parameter_values_arr,
            )
            transport_parameter_set = ReverseADParameterSet(
                profile_specs=tuple(ProfileParameterSpec(name) for name in PROFILE_PARAMETER_ORDER),
                vmec_boundary_specs=tuple(parameter_set.vmec_boundary_specs),
            )
            transport_parameter_values = jnp.concatenate(
                [
                    jnp.asarray(active_profile_values, dtype=parameter_values_arr.dtype),
                    jnp.asarray(vmec_parameter_values, dtype=parameter_values_arr.dtype),
                ],
                axis=0,
            )
            shared_raw_block_solve = geometry_raw_block_solve_from_param_vector(
                geometry_context,
                vmec_parameter_values,
                tuple(spec.as_tuple() for spec in parameter_set.vmec_boundary_specs),
                max_iter=geometry_max_iter,
                solver_device=geometry_solver_device,
                stage=raw_block_stage,
            )
            transport_result = geometry_active_initial_er_root_only_reverse_table(
                config=config,
                objective_names=requested_transport_objectives,
                parameter_set=transport_parameter_set,
                parameter_values=transport_parameter_values,
                runtime=runtime,
                profile_values=active_profile_values,
                pre_root_state_from_profile_values=pre_root_state_from_profile_values,
                geometry_context=geometry_context,
                baseline_geometry_deltas=jnp.zeros_like(vmec_parameter_values),
                n_r=int(n_r),
                n_theta=int(n_theta),
                n_zeta=int(n_zeta),
                n_xi=int(n_xi),
                surface_backend=str(surface_backend),
                max_iter=geometry_max_iter,
                solver_device=geometry_solver_device,
                progress_label="[optimization] initial-Er root geometry payload pullback:",
                raw_block_solve=shared_raw_block_solve,
                options=root_runner_options,
            )
            transport_values, transport_jacobian = jax.block_until_ready(
                (transport_result.values, transport_result.jacobian)
            )
            backend_results["transport"] = _adapt_objective_table_result(
                ObjectiveTableResult(
                    objective_names=transport_result.objective_names,
                    values=transport_values,
                    jacobian=transport_jacobian,
                ),
                source_parameter_set=transport_parameter_set,
                target_parameter_set=parameter_set,
                objective_names=requested_transport_objectives,
            )
        else:
            transport_table = initial_er_root_only_objective_cotangent_table(
                config=config,
                objective_names=requested_transport_objectives,
                parameter_set=parameter_set,
                parameter_values=parameter_values_arr,
                runtime=runtime,
                profile_values=active_profile_values,
                pre_root_state_from_profile_values=pre_root_state_from_profile_values,
                options=root_runner_options,
            )
            backend_results["transport"] = objective_table_result_from_cotangent_table(
                transport_table,
                parameter_set=parameter_set,
                geometry_gradient_matrix=jnp.zeros(
                    (len(transport_table.objective_names), len(parameter_set.vmec_boundary_specs)),
                    dtype=jnp.asarray(transport_table.values).dtype,
                ),
            )

    if "geometry" in grouped_terms:
        backend_results["geometry"] = geometry_full_ad_reverse_table(
            context=geometry_context,
            parameter_set=parameter_set,
            objective_names=_unique_objective_names(grouped_terms["geometry"]),
            parameter_values=parameter_values_arr,
            lane=geometry_lane,
            max_iter=geometry_max_iter,
            step_size=geometry_step_size,
            final_vmec_pullback_mode="raw_block_transpose",
            solver_device=geometry_solver_device,
            raw_block_solve=shared_raw_block_solve,
        )

    result = assemble_least_squares_result(
        normalized_terms,
        parameter_set=parameter_set,
        backend_results=backend_results,
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


def build_geometry_initial_er_root_only_least_squares_runner(
    config: Mapping[str, object],
    *,
    parameter_set: ReverseADParameterSet,
    geometry_context,
    runtime=None,
    rooted_state_from_parameter_vector: InitialErRootOnlyStateBuilder | None = None,
    options: Mapping[str, object] | None = None,
    root_options: Mapping[str, object] | None = None,
    geometry_lane: str = "ad",
    geometry_max_iter: int | None = None,
    geometry_step_size: float | None = None,
    geometry_final_vmec_pullback_mode: str = "raw_block_transpose",
    geometry_solver_device: str | None = "default",
) -> GeometryInitialErRootOnlyLeastSquaresRunner:
    """Build a runner for geometry terms and initial-Er root-only transport terms."""

    runner_options = {} if options is None else dict(options)
    runner_root_options = {} if root_options is None else dict(root_options)

    def _runner(
        parameter_values,
        terms: Sequence[LeastSquaresTerm | tuple[ObjectiveRef | str, float, float]],
    ) -> LeastSquaresEvaluation:
        return evaluate_geometry_initial_er_root_only_least_squares(
            config,
            parameter_set=parameter_set,
            parameter_values=parameter_values,
            terms=terms,
            geometry_context=geometry_context,
            runtime=runtime,
            rooted_state_from_parameter_vector=rooted_state_from_parameter_vector,
            options=runner_options,
            root_options=runner_root_options,
            geometry_lane=geometry_lane,
            geometry_max_iter=geometry_max_iter,
            geometry_step_size=geometry_step_size,
            geometry_final_vmec_pullback_mode=geometry_final_vmec_pullback_mode,
            geometry_solver_device=geometry_solver_device,
        )

    return _runner


def scale_least_squares_evaluation_columns(
    evaluation: LeastSquaresEvaluation,
    column_scale,
) -> LeastSquaresEvaluation:
    """Scale Jacobian columns for an optimizer parameterization.

    If the inner runner differentiates with respect to physical deltas
    ``p = scale * x``, the optimizer-facing Jacobian with respect to ``x`` is
    ``J_p * scale``.
    """

    jacobian_physical = jnp.asarray(evaluation.jacobian)
    scale = jnp.asarray(column_scale, dtype=jacobian_physical.dtype)
    if scale.ndim != 1:
        raise ValueError(f"column_scale must be one-dimensional; got shape={scale.shape}.")
    if jacobian_physical.ndim != 2:
        raise ValueError(f"least-squares jacobian must be two-dimensional; got shape={jacobian_physical.shape}.")
    if int(scale.shape[0]) != int(jacobian_physical.shape[1]):
        raise ValueError(
            "column_scale length must match the least-squares jacobian parameter count; "
            f"got scale.shape={scale.shape}, jacobian.shape={jacobian_physical.shape}."
        )
    jacobian = jacobian_physical * scale[jnp.newaxis, :]
    result = dataclasses.replace(evaluation.result, jacobian=jacobian)
    return dataclasses.replace(evaluation, result=result, jacobian=jacobian)


def build_scaled_parameter_least_squares_runner(
    runner: GeometryInitialErRootOnlyLeastSquaresRunner | InitialErRootOnlyLeastSquaresRunner,
    *,
    column_scale,
):
    """Wrap a physical-parameter runner with optimizer-scaled parameters."""

    scale = jnp.asarray(column_scale, dtype=jnp.float64)

    def _runner(
        scaled_parameter_values,
        terms: Sequence[LeastSquaresTerm | tuple[ObjectiveRef | str, float, float]],
    ) -> LeastSquaresEvaluation:
        physical_parameter_values = jnp.asarray(scaled_parameter_values, dtype=scale.dtype) * scale
        evaluation = runner(physical_parameter_values, terms)
        return scale_least_squares_evaluation_columns(evaluation, scale)

    return _runner


def evaluate_geometry_initial_er_root_only_least_squares_optimization(
    config: Mapping[str, object],
    *,
    parameter_set: ReverseADParameterSet,
    parameter_values,
    terms: Sequence[LeastSquaresTerm | tuple[ObjectiveRef | str, float, float]],
    geometry_context,
    runtime,
    baseline_profile_values,
    pre_root_state_from_profile_values: Callable[[object], object],
    n_r: int,
    n_theta: int,
    n_zeta: int,
    n_xi: int,
    surface_backend: str = "vmec",
    geometry_lane: str = "ad",
    geometry_max_iter: int | None = None,
    geometry_step_size: float | None = None,
    geometry_solver_device: str | None = "default",
    root_options: Mapping[str, object] | None = None,
    raw_block_stage=None,
    optimization_stage=None,
    transport_reverse_stage: InitialErTransportReverseStage | None = None,
    payload_assembly_stage=None,
) -> LeastSquaresEvaluation:
    """Evaluate mixed objectives using only benchmark-validated table backends."""

    normalized_terms = normalize_least_squares_terms(terms)
    grouped_terms = group_least_squares_terms_by_family(normalized_terms)
    unsupported_families = tuple(
        family for family in grouped_terms if family not in {"geometry", "transport"}
    )
    if unsupported_families:
        raise NotImplementedError(
            "evaluate_geometry_initial_er_root_only_least_squares_benchmark_tables supports only "
            f"geometry and transport/initial-Er terms; got families {unsupported_families!r}."
        )

    parameter_values_arr = jnp.asarray(parameter_values, dtype=jnp.float64)
    backend_results: dict[ObjectiveFamily, ObjectiveTableResult] = {}
    shared_raw_block_solve = None
    root_runner_options = {} if root_options is None else dict(root_options)
    t_start = time.perf_counter()

    if "transport" in grouped_terms:
        active_profile_values = _active_profile_values_from_parameter_vector(
            parameter_set,
            parameter_values_arr,
            baseline_profile_values,
        )
        requested_transport_objectives = _unique_objective_names(grouped_terms["transport"])
        if parameter_set.vmec_boundary_specs:
            vmec_parameter_values = vmec_parameter_values_from_parameter_vector(
                parameter_set,
                parameter_values_arr,
            )
            transport_parameter_set = ReverseADParameterSet(
                profile_specs=tuple(ProfileParameterSpec(name) for name in PROFILE_PARAMETER_ORDER),
                vmec_boundary_specs=tuple(parameter_set.vmec_boundary_specs),
            )
            transport_parameter_values = jnp.concatenate(
                [
                    jnp.asarray(active_profile_values, dtype=parameter_values_arr.dtype),
                    jnp.asarray(vmec_parameter_values, dtype=parameter_values_arr.dtype),
                ],
                axis=0,
            )
            shared_raw_block_solve = geometry_raw_block_solve_from_param_vector(
                geometry_context,
                vmec_parameter_values,
                tuple(spec.as_tuple() for spec in parameter_set.vmec_boundary_specs),
                max_iter=geometry_max_iter,
                solver_device=geometry_solver_device,
                stage=raw_block_stage,
            )
            transport_result = geometry_active_initial_er_root_only_reverse_table_optimization(
                config=config,
                objective_names=requested_transport_objectives,
                parameter_set=transport_parameter_set,
                parameter_values=transport_parameter_values,
                runtime=runtime,
                profile_values=active_profile_values,
                pre_root_state_from_profile_values=pre_root_state_from_profile_values,
                geometry_context=geometry_context,
                baseline_geometry_deltas=jnp.zeros_like(vmec_parameter_values),
                n_r=int(n_r),
                n_theta=int(n_theta),
                n_zeta=int(n_zeta),
                n_xi=int(n_xi),
                surface_backend=str(surface_backend),
                max_iter=geometry_max_iter,
                solver_device=geometry_solver_device,
                progress_label="[optimization] initial-Er root geometry payload pullback:",
                raw_block_solve=shared_raw_block_solve,
                options=root_runner_options,
                optimization_stage=optimization_stage,
                transport_reverse_stage=transport_reverse_stage,
                payload_assembly_stage=payload_assembly_stage,
            )
            transport_values, transport_jacobian = jax.block_until_ready(
                (transport_result.values, transport_result.jacobian)
            )
            backend_results["transport"] = _adapt_objective_table_result(
                ObjectiveTableResult(
                    objective_names=transport_result.objective_names,
                    values=transport_values,
                    jacobian=transport_jacobian,
                ),
                source_parameter_set=transport_parameter_set,
                target_parameter_set=parameter_set,
                objective_names=requested_transport_objectives,
            )
        else:
            transport_table = initial_er_root_only_objective_cotangent_table(
                config=config,
                objective_names=requested_transport_objectives,
                parameter_set=parameter_set,
                parameter_values=parameter_values_arr,
                runtime=runtime,
                profile_values=active_profile_values,
                pre_root_state_from_profile_values=pre_root_state_from_profile_values,
                options=root_runner_options,
            )
            backend_results["transport"] = objective_table_result_from_cotangent_table(
                transport_table,
                parameter_set=parameter_set,
                geometry_gradient_matrix=jnp.zeros(
                    (len(transport_table.objective_names), len(parameter_set.vmec_boundary_specs)),
                    dtype=jnp.asarray(transport_table.values).dtype,
                ),
            )

    if "geometry" in grouped_terms:
        backend_results["geometry"] = geometry_full_ad_reverse_table(
            context=geometry_context,
            parameter_set=parameter_set,
            objective_names=_unique_objective_names(grouped_terms["geometry"]),
            parameter_values=parameter_values_arr,
            lane=geometry_lane,
            max_iter=geometry_max_iter,
            step_size=geometry_step_size,
            final_vmec_pullback_mode="raw_block_transpose",
            solver_device=geometry_solver_device,
            raw_block_solve=shared_raw_block_solve,
        )

    result = assemble_least_squares_result(
        normalized_terms,
        parameter_set=parameter_set,
        backend_results=backend_results,
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


def build_geometry_initial_er_root_only_least_squares_runner(
    config: Mapping[str, object],
    *,
    parameter_set: ReverseADParameterSet,
    geometry_context,
    runtime=None,
    rooted_state_from_parameter_vector: InitialErRootOnlyStateBuilder | None = None,
    options: Mapping[str, object] | None = None,
    root_options: Mapping[str, object] | None = None,
    geometry_lane: str = "ad",
    geometry_max_iter: int | None = None,
    geometry_step_size: float | None = None,
    geometry_final_vmec_pullback_mode: str = "raw_block_transpose",
    geometry_solver_device: str | None = "default",
) -> GeometryInitialErRootOnlyLeastSquaresRunner:
    """Build a runner for geometry terms and initial-Er root-only transport terms."""

    runner_options = {} if options is None else dict(options)
    runner_root_options = {} if root_options is None else dict(root_options)

    def _runner(
        parameter_values,
        terms: Sequence[LeastSquaresTerm | tuple[ObjectiveRef | str, float, float]],
    ) -> LeastSquaresEvaluation:
        return evaluate_geometry_initial_er_root_only_least_squares(
            config,
            parameter_set=parameter_set,
            parameter_values=parameter_values,
            terms=terms,
            geometry_context=geometry_context,
            runtime=runtime,
            rooted_state_from_parameter_vector=rooted_state_from_parameter_vector,
            options=runner_options,
            root_options=runner_root_options,
            geometry_lane=geometry_lane,
            geometry_max_iter=geometry_max_iter,
            geometry_step_size=geometry_step_size,
            geometry_final_vmec_pullback_mode=geometry_final_vmec_pullback_mode,
            geometry_solver_device=geometry_solver_device,
        )

    return _runner


def scale_least_squares_evaluation_columns(
    evaluation: LeastSquaresEvaluation,
    column_scale,
) -> LeastSquaresEvaluation:
    """Scale Jacobian columns for an optimizer parameterization.

    If the inner runner differentiates with respect to physical deltas
    ``p = scale * x``, the optimizer-facing Jacobian with respect to ``x`` is
    ``J_p * scale``.
    """

    jacobian_physical = jnp.asarray(evaluation.jacobian)
    scale = jnp.asarray(column_scale, dtype=jacobian_physical.dtype)
    if scale.ndim != 1:
        raise ValueError(f"column_scale must be one-dimensional; got shape={scale.shape}.")
    if jacobian_physical.ndim != 2:
        raise ValueError(f"least-squares jacobian must be two-dimensional; got shape={jacobian_physical.shape}.")
    if int(scale.shape[0]) != int(jacobian_physical.shape[1]):
        raise ValueError(
            "column_scale length must match the least-squares jacobian parameter count; "
            f"got scale.shape={scale.shape}, jacobian.shape={jacobian_physical.shape}."
        )
    jacobian = jacobian_physical * scale[jnp.newaxis, :]
    result = dataclasses.replace(evaluation.result, jacobian=jacobian)
    return dataclasses.replace(evaluation, result=result, jacobian=jacobian)


def build_scaled_parameter_least_squares_runner(
    runner: GeometryInitialErRootOnlyLeastSquaresRunner | InitialErRootOnlyLeastSquaresRunner,
    *,
    column_scale,
):
    """Wrap a physical-parameter runner with optimizer-scaled parameters."""

    scale = jnp.asarray(column_scale, dtype=jnp.float64)

    def _runner(
        scaled_parameter_values,
        terms: Sequence[LeastSquaresTerm | tuple[ObjectiveRef | str, float, float]],
    ) -> LeastSquaresEvaluation:
        physical_parameter_values = jnp.asarray(scaled_parameter_values, dtype=scale.dtype) * scale
        evaluation = runner(physical_parameter_values, terms)
        return scale_least_squares_evaluation_columns(evaluation, scale)

    return _runner



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


def evaluate_geometry_transport_realtime_geometry_least_squares(
    config: Mapping[str, object],
    *,
    request: RealtimeGeometryTransportReverseTableRequest,
    terms: Sequence[LeastSquaresTerm | tuple[ObjectiveRef | str, float, float]],
    geometry_context,
    parameter_values=None,
    table_result_builder: TransportReverseTableResultBuilder | None = None,
    run_grouped_report=None,
    objective_labels: Sequence[str] | None = None,
    options: Mapping[str, object] | None = None,
    quiet_default: bool = True,
    geometry_lane: str = "ad",
    geometry_max_iter: int | None = None,
    geometry_step_size: float | None = None,
    geometry_final_vmec_pullback_mode: str = "raw_block_transpose",
    geometry_solver_device: str | None = "default",
    share_raw_block_solve: bool = True,
) -> LeastSquaresEvaluation:
    """Evaluate mixed geometry + realtime-transport least-squares terms.

    This is the full time-evolution analogue of the mixed initial-Er-root
    optimizer wiring: transport rows come from the realtime transport reverse
    table, geometry rows come from the validated full-geometry reverse table,
    and callers that use the direct internal transport table builder can pass a
    shared raw-block VMEC solve through both branches.
    """

    normalized_terms = normalize_least_squares_terms(terms)
    grouped_terms = group_least_squares_terms_by_family(normalized_terms)
    unsupported_families = tuple(
        family for family in grouped_terms if family not in {"geometry", "transport"}
    )
    if unsupported_families:
        raise NotImplementedError(
            "evaluate_geometry_transport_realtime_geometry_least_squares supports only "
            f"geometry and transport terms; got families {unsupported_families!r}."
        )

    requested_transport_objectives = _unique_objective_names(grouped_terms.get("transport", ()))
    if requested_transport_objectives and requested_transport_objectives != request.objective_names:
        raise ValueError(
            "Realtime-geometry transport request objectives must match the transport "
            "least-squares terms in first-use order: "
            f"request={request.objective_names!r}, terms={requested_transport_objectives!r}."
        )

    opts = {} if options is None else dict(options)
    timing_diagnostics = bool(opts.get("reverse_table_timing_diagnostics", False))
    outer_start = time.perf_counter()
    previous_phase_time = outer_start

    def _report_outer_phase(phase: str) -> None:
        nonlocal previous_phase_time
        if not timing_diagnostics:
            return
        now = time.perf_counter()
        print(
            "[autodiff-gate] timing: "
            f"phase=optimization_table.{phase} "
            f"elapsed_s={now - previous_phase_time:.3f} "
            f"since_outer_start_s={now - outer_start:.3f} "
            f"gap_since_previous_s={now - previous_phase_time:.3f}",
            flush=True,
        )
        previous_phase_time = now

    parameter_values_arr = (
        jnp.zeros((len(request.parameter_set.specs),), dtype=jnp.float64)
        if parameter_values is None
        else jnp.asarray(parameter_values, dtype=jnp.float64)
    )
    if tuple(parameter_values_arr.shape) != (len(request.parameter_set.specs),):
        raise ValueError(
            "parameter_values must match the realtime-geometry parameter set; "
            f"got {tuple(parameter_values_arr.shape)}, expected ({len(request.parameter_set.specs)},)."
        )
    baseline_profile_values = jnp.asarray(
        request.context.baseline_values[: len(PROFILE_PARAMETER_ORDER)],
        dtype=parameter_values_arr.dtype,
    )
    opts.setdefault(
        "profile_values",
        _active_profile_values_from_parameter_vector(
            request.parameter_set,
            parameter_values_arr,
            baseline_profile_values,
        ),
    )

    shared_raw_block_solve = None
    if (
        bool(share_raw_block_solve)
        and request.parameter_set.vmec_boundary_specs
        and ("geometry" in grouped_terms or table_result_builder is not None)
    ):
        vmec_parameter_values = vmec_parameter_values_from_parameter_vector(
            request.parameter_set,
            parameter_values_arr,
        )
        shared_raw_block_solve = geometry_raw_block_solve_from_param_vector(
            geometry_context,
            vmec_parameter_values,
            tuple(spec.as_tuple() for spec in request.parameter_set.vmec_boundary_specs),
            max_iter=geometry_max_iter,
            solver_device=geometry_solver_device,
        )
        opts.setdefault("raw_block_solve", shared_raw_block_solve)
        opts.setdefault("geometry_raw_block_solve", shared_raw_block_solve)
        try:
            use_runtime_payload = bool(
                jnp.allclose(
                    vmec_parameter_values,
                    jnp.zeros_like(vmec_parameter_values),
                ).item()
            )
        except Exception:
            use_runtime_payload = False
        opts.setdefault("use_runtime_payload", use_runtime_payload)
    _report_outer_phase("shared_raw_block_setup")

    backend_results: dict[ObjectiveFamily, ObjectiveTableResult] = {}
    t_start = time.perf_counter()

    if "transport" in grouped_terms:
        table_result = transport_realtime_geometry_reverse_table(
            request=request,
            table_result_builder=table_result_builder,
            run_grouped_report=run_grouped_report,
            objective_labels=objective_labels,
            options=opts,
            quiet_default=quiet_default,
        )
        backend_results["transport"] = transport_reverse_table_result_to_objective_table_result(
            table_result,
            requested_transport_objectives,
            request.parameter_set,
        )
        _report_outer_phase("transport_table")

    if "geometry" in grouped_terms:
        backend_results["geometry"] = geometry_full_ad_reverse_table(
            context=geometry_context,
            parameter_set=request.parameter_set,
            objective_names=_unique_objective_names(grouped_terms["geometry"]),
            parameter_values=parameter_values_arr,
            lane=geometry_lane,
            max_iter=geometry_max_iter,
            step_size=geometry_step_size,
            final_vmec_pullback_mode=geometry_final_vmec_pullback_mode,
            solver_device=geometry_solver_device,
            raw_block_solve=shared_raw_block_solve,
        )
        _report_outer_phase("geometry_table")

    result = assemble_least_squares_result(
        normalized_terms,
        parameter_set=request.parameter_set,
        backend_results=backend_results,
    )
    residuals = jax.block_until_ready(result.residuals)
    jacobian = jax.block_until_ready(result.jacobian)
    _report_outer_phase("assemble_and_synchronize")
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

    This is the optimizer-facing wiring helper. The heavy reverse executor is
    supplied as a grouped report runner so optimization can use the same
    validated memory/graph behavior as the reverse benchmark.
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


def normalize_geometry_full_ad_objective_names(objective_names: Sequence[str]) -> tuple[str, ...]:
    """Normalize VMEX-like geometry objective names to the full AD table labels."""

    requested = tuple(str(name).strip() for name in objective_names)
    if not requested or any(not name for name in requested):
        raise ValueError("At least one non-empty geometry objective name is required.")
    available = set(geometry_observable_names_for_kind("geometry_full_ad_objectives"))
    normalized = []
    for name in requested:
        canonical = GEOMETRY_FULL_AD_OBJECTIVE_ALIASES.get(name, name)
        if canonical not in available:
            choices = ", ".join(sorted(GEOMETRY_FULL_AD_OBJECTIVE_ALIASES))
            raise ValueError(
                f"Unknown geometry objective {name!r}; use a geometry_full_ad objective "
                f"or one of these aliases: {choices}."
            )
        normalized.append(canonical)
    return tuple(normalized)


def geometry_full_ad_objective_cotangent_basis(
    objective_names: Sequence[str],
) -> tuple[tuple[str, ...], jnp.ndarray]:
    """Return canonical geometry objective names and rows in full table order."""

    requested_objectives = tuple(str(name).strip() for name in objective_names)
    canonical_objectives = normalize_geometry_full_ad_objective_names(requested_objectives)
    full_objective_names = geometry_observable_names_for_kind("geometry_full_ad_objectives")
    objective_cotangents = jnp.zeros((len(canonical_objectives), len(full_objective_names)), dtype=jnp.float64)
    for row_i, canonical_name in enumerate(canonical_objectives):
        objective_cotangents = objective_cotangents.at[row_i, full_objective_names.index(canonical_name)].set(1.0)
    return canonical_objectives, objective_cotangents


def geometry_full_ad_objective_cotangent_table(
    *,
    context,
    parameter_set: ReverseADParameterSet,
    objective_names: Sequence[str],
    parameter_values=None,
    shared_payload: SharedGeometryTransportPayload | None = None,
    lane: str = "ad",
    max_iter: int | None = None,
    step_size: float | None = None,
    solver_device: str | None = "default",
) -> ObjectiveCotangentTable:
    """Return geometry objective values and VMEC-state bars before raw-block pullback."""

    requested_objectives = tuple(str(name).strip() for name in objective_names)
    canonical_objectives, objective_cotangents = geometry_full_ad_objective_cotangent_basis(requested_objectives)
    vmec_specs = tuple(spec for spec in parameter_set.specs if isinstance(spec, VmecBoundaryParameterSpec))
    if not vmec_specs:
        raise ValueError(
            "Geometry objectives require at least one VMEC boundary parameter in the optimization parameter set."
        )
    if parameter_values is None:
        parameter_values_arr = jnp.zeros((len(parameter_set.specs),), dtype=jnp.float64)
    else:
        parameter_values_arr = jnp.asarray(parameter_values, dtype=jnp.float64)
    vmec_param_deltas = vmec_parameter_values_from_parameter_vector(parameter_set, parameter_values_arr)
    full_objective_names = geometry_observable_names_for_kind("geometry_full_ad_objectives")
    raw_block_solve = None if shared_payload is None else shared_payload.raw_block_solve
    values_by_name, vmec_state_bars = geometry_full_ad_objective_table_pullback_from_param_vector(
        context,
        vmec_param_deltas,
        tuple(spec.as_tuple() for spec in vmec_specs),
        objective_cotangents,
        objective_names=full_objective_names,
        lane=lane,
        max_iter=max_iter,
        step_size=step_size,
        final_vmec_pullback_mode="raw_block_transpose",
        solver_device=solver_device,
        raw_block_solve=raw_block_solve,
        return_state_bars=True,
    )
    objective_values = jnp.stack(
        [jnp.asarray(values_by_name[name], dtype=jnp.float64).reshape(()) for name in canonical_objectives]
    )
    profile_specs = tuple(parameter_set.profile_specs)
    profile_gradient_matrix = jnp.zeros((len(requested_objectives), len(profile_specs)), dtype=jnp.float64)
    return ObjectiveCotangentTable(
        objective_names=requested_objectives,
        values=objective_values,
        profile_gradient_matrix=profile_gradient_matrix,
        vmec_state_bars=vmec_state_bars,
        payload_bars=(),
    )


def geometry_full_ad_reverse_table(
    *,
    context,
    parameter_set: ReverseADParameterSet,
    objective_names: Sequence[str],
    parameter_values=None,
    lane: str = "ad",
    max_iter: int | None = None,
    step_size: float | None = None,
    final_vmec_pullback_mode: str = "raw_block_transpose",
    solver_device: str | None = "default",
    raw_block_solve=None,
) -> ObjectiveTableResult:
    """Evaluate the validated full-geometry reverse table for optimization terms.

    The returned Jacobian columns follow ``parameter_set.specs``. Profile columns
    are zero because geometry-only objectives do not depend on transport profile
    DOFs; VMEC-boundary columns are pulled with the same raw-block table path used
    by the geometry AD benchmarks.
    """

    requested_objectives = tuple(str(name).strip() for name in objective_names)
    canonical_objectives, objective_cotangents = geometry_full_ad_objective_cotangent_basis(requested_objectives)
    vmec_specs = tuple(spec for spec in parameter_set.specs if isinstance(spec, VmecBoundaryParameterSpec))
    if not vmec_specs:
        raise ValueError(
            "Geometry objectives require at least one VMEC boundary parameter in the optimization "
            "parameter set. Omit geometry terms for profile-only optimization runs."
        )

    full_objective_names = geometry_observable_names_for_kind("geometry_full_ad_objectives")
    if parameter_values is None:
        parameter_values_arr = jnp.zeros((len(parameter_set.specs),), dtype=jnp.float64)
    else:
        parameter_values_arr = jnp.asarray(parameter_values, dtype=jnp.float64)
    if tuple(parameter_values_arr.shape) != (len(parameter_set.specs),):
        raise ValueError(
            "geometry_full_ad_reverse_table parameter_values must have shape "
            f"({len(parameter_set.specs)},); got {tuple(parameter_values_arr.shape)}."
        )
    vmec_param_deltas = jnp.asarray(
        [parameter_values_arr[i] for i, spec in enumerate(parameter_set.specs) if isinstance(spec, VmecBoundaryParameterSpec)],
        dtype=jnp.float64,
    )

    values_by_name, vmec_gradient_matrix = geometry_full_ad_objective_table_pullback_from_param_vector(
        context,
        vmec_param_deltas,
        tuple(spec.as_tuple() for spec in vmec_specs),
        objective_cotangents,
        objective_names=full_objective_names,
        lane=lane,
        max_iter=max_iter,
        step_size=step_size,
        final_vmec_pullback_mode=final_vmec_pullback_mode,
        solver_device=solver_device,
        raw_block_solve=raw_block_solve,
    )
    objective_values = jnp.stack(
        [jnp.asarray(values_by_name[name], dtype=jnp.float64).reshape(()) for name in canonical_objectives]
    )
    vmec_column_lookup = {spec: column_i for column_i, spec in enumerate(vmec_specs)}
    jacobian_rows = []
    for row_i in range(len(requested_objectives)):
        columns = []
        for spec in parameter_set.specs:
            if isinstance(spec, ProfileParameterSpec):
                columns.append(jnp.asarray(0.0, dtype=jnp.float64))
            elif isinstance(spec, VmecBoundaryParameterSpec):
                columns.append(vmec_gradient_matrix[row_i, vmec_column_lookup[spec]])
            else:
                raise TypeError(f"Unsupported reverse-AD parameter spec type: {type(spec).__name__}.")
        jacobian_rows.append(jnp.stack(columns))
    return ObjectiveTableResult(
        objective_names=requested_objectives,
        values=objective_values,
        jacobian=jnp.stack(jacobian_rows, axis=0),
    )


def geometry_full_ad_reverse_table_backend(
    *,
    context,
    parameter_values=None,
    lane: str = "ad",
    max_iter: int | None = None,
    step_size: float | None = None,
    final_vmec_pullback_mode: str = "raw_block_transpose",
    solver_device: str | None = "default",
    raw_block_solve=None,
) -> ObjectiveTableBackend:
    """Return a geometry backend for ``residuals_and_jacobian_reverse_ad``."""

    def _backend(
        objective_names: tuple[str, ...],
        parameter_set: ReverseADParameterSet,
        options: Mapping[str, object],
    ) -> ObjectiveTableResult:
        opts = {} if options is None else dict(options)
        return geometry_full_ad_reverse_table(
            context=context,
            parameter_set=parameter_set,
            objective_names=objective_names,
            parameter_values=opts.get("parameter_values", parameter_values),
            lane=str(opts.get("geometry_lane", lane)),
            max_iter=opts.get("geometry_max_iter", max_iter),
            step_size=opts.get("geometry_step_size", step_size),
            final_vmec_pullback_mode=str(
                opts.get("geometry_final_vmec_pullback_mode", final_vmec_pullback_mode)
            ),
            solver_device=opts.get("geometry_solver_device", solver_device),
            raw_block_solve=opts.get("geometry_raw_block_solve", raw_block_solve),
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
