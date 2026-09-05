"""Reusable transport reverse-AD report-builder helpers.

This module owns production-facing transport reverse-AD seams that have been
lifted out of benchmark reporting. It provides the compact segmented transport
cotangent path and the VMEC raw-block payload pullback used by optimization
callers, while keeping report formatting outside the AD graph.
"""

from __future__ import annotations

import copy
import contextlib
import dataclasses
import io
import inspect
import os
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ._constants import elementary_charge
from ._geometry_autodiff import (
    boundary_param_entries,
    build_neopax_geometry_and_ntx_exact_lij_support_from_state,
    build_geometry_autodiff_context,
    GeometryRawBlockSolve,
    geometry_payload_pullback_from_param_vector_raw_block_transpose,
)
from ._orchestrator import prepare_transport_solver_components
from ._profiles import AnalyticalProfileModel
from ._reverse_ad_initial_er import (
    compact_initial_er_database_support_bars,
    fold_recorded_ntx_scan_database_bar_groups_into_support,
    compact_initial_er_ntx_support_pullback_leaves,
    compact_initial_er_state_pullback,
    find_ntx_support_payload,
    initial_er_charge_flux_residual_er_derivative,
    initial_er_charge_flux_residual_scalar,
    initial_er_charge_flux_residuals,
    initial_er_selected_root_profile,
    realtime_geometry_payload_for_runtime,
    realtime_geometry_reverse_support_payload_for_runtime,
    runtime_with_geometry_payload,
    runtime_with_ntx_support_payload,
    runtime_with_realtime_geometry_reverse_support_payload,
    runtime_without_recorded_ntx_scan_primal,
)
from ._reverse_ad_parameters import (
    PROFILE_PARAMETER_ORDER,
    ReverseADParameterSet,
    discover_vmec_boundary_parameter_specs,
    normalize_vmec_boundary_families,
    vmec_boundary_tuples,
)
from ._transport_flux_models import (
    DENSITY_STATE_TO_PHYSICAL,
    PRESSURE_SOURCE_STATE_TO_MW_M3,
    compute_net_total_power_volume_average_mw_m3,
    _add_float_delta_tree,
    _float_delta_tree_like,
    _sanitize_float_delta_bar_tree,
)
from ._state import safe_density, safe_temperature
from ._transport_solvers import (
    RADAUSolver,
    NewtonThetaMethodSolver,
    ThetaMethodSolver,
    _RadauAcceptedStepReducedCotangent,
    _ThetaStepState,
    _build_prepared_radau_accepted_rollout,
    _build_prepared_radau_execution_context,
    _extract_fixed_temperature_projection,
    _extract_state_regularization,
    _flat_rhs_factory,
    _flat_rhs_build_support_pullback_factory,
    _flat_rhs_lagged_response_pullback_factory,
    _flat_rhs_lagged_response_support_pullback_factory,
    _flat_rhs_state_pullback_factory,
    _flat_rhs_with_lagged_response_factory,
    _lagged_response_hooks,
    _make_radau_initial_step_state,
    _make_solver_state_transform,
    _radau_adaptive_schedule_rollout,
    _radau_sanitize_support_delta_bar_tree,
    _radau_zero_native_vmec_face_coefficient_bars,
    _theta_basic_accepted_step_attempt,
    _theta_basic_adaptive_schedule_rollout,
    _theta_basic_step_from_attempt_fn,
    _theta_initial_reuse_state,
    _theta_make_attempt_context,
    _theta_newton_accepted_step_attempt,
    _theta_newton_adaptive_schedule_rollout,
    _theta_newton_step_from_attempt_fn,
    _theta_prepare_lagged_response,
    _project_flat_state_if_needed,
    _radau_adaptive_final_y_realized_schedule_vjp_fwd,
    _radau_adaptive_final_y_realized_schedule_vjp_fwd_from_schedule_artifact,
    _radau_align_tangent_tree_to_primal,
    _radau_carry_from_step_state,
    _radau_carry_with_forward_only_jvp_fields,
    _radau_eval_rhs,
    _execute_radau_accepted_step_next_reduced_cotangent_batched_bwd_with_support_call,
    _execute_radau_accepted_step_next_reduced_cotangent_batched_bwd_with_support_from_segment_primal_record_call,
    _radau_segment_reduced_cotangent_bwd_batched_with_support_call,
    _radau_segment_replay_minimal_with_primal_records_call,
    _radau_segment_reduced_cotangent_bwd_batched_with_support_from_primal_records_call,
    _radau_sanitize_support_delta_bar_tree,
    _radau_zero_support_delta_tree_like,
)


def _reverse_tree_debug_enabled() -> bool:
    raw = str(os.environ.get("NEOPAX_REVERSE_TREE_DEBUG", "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _jax_trace_cache_size(jitted_function) -> int | None:
    """Return JAX's in-process trace-cache entry count when exposed.

    This is diagnostics-only host inspection. It is deliberately labeled
    "trace cache": it does not claim to measure the persistent XLA cache or
    device execution time.
    """

    cache_size = getattr(jitted_function, "_cache_size", None)
    if not callable(cache_size):
        return None
    try:
        return int(cache_size())
    except Exception:
        return None


def _logical_tree_nbytes(tree: object) -> int:
    """Logical array payload size without transferring device buffers."""
    total = 0
    for leaf in jax.tree_util.tree_leaves(tree):
        shape = getattr(leaf, "shape", None)
        dtype = getattr(leaf, "dtype", None)
        if shape is None or dtype is None:
            continue
        total += int(np.prod(tuple(shape), dtype=np.int64)) * int(np.dtype(dtype).itemsize)
    return total


def _pytree_path_label(path) -> str:
    """Format a JAX pytree path for host-only reverse diagnostics."""
    parts: list[str] = []
    for key in path:
        if hasattr(key, "name"):
            parts.append(str(key.name))
        elif hasattr(key, "key"):
            parts.append(str(key.key))
        elif hasattr(key, "idx"):
            parts.append(str(key.idx))
        else:
            parts.append(str(key))
    return ".".join(parts)


def _batched_support_first_nonfinite_leaves(
    batched_leaves: Sequence[object],
    leaf_labels: Sequence[str],
    objective_count: int,
) -> tuple[tuple[int, str] | None, ...]:
    """Return each objective row's first bad support leaf without copying bars.

    This transfers only one boolean per objective row and leaf.  It is used
    solely by the host diagnostic path after a segment has already completed.
    """
    if len(batched_leaves) != len(leaf_labels):
        raise ValueError("Support diagnostic leaves do not match their pytree paths.")
    row_finite = []
    for leaf in batched_leaves:
        value = jnp.asarray(leaf)
        if value.dtype == jax.dtypes.float0 or not jnp.issubdtype(value.dtype, jnp.inexact):
            row_finite.append(jnp.ones((objective_count,), dtype=bool))
            continue
        if value.ndim < 1 or int(value.shape[0]) != int(objective_count):
            raise ValueError(
                "Batched support diagnostic leaf does not have objective rows: "
                f"shape={value.shape}, objectives={objective_count}."
            )
        axes = tuple(range(1, value.ndim))
        row_finite.append(
            jnp.isfinite(value)
            if not axes
            else jnp.all(jnp.isfinite(value), axis=axes)
        )
    finite_rows = tuple(np.asarray(value, dtype=bool) for value in jax.device_get(tuple(row_finite)))
    first_bad: list[tuple[int, str] | None] = []
    for objective_i in range(objective_count):
        match = next(
            (
                (leaf_i, leaf_labels[leaf_i])
                for leaf_i, finite in enumerate(finite_rows)
                if not bool(finite[objective_i])
            ),
            None,
        )
        first_bad.append(match)
    return tuple(first_bad)


def _merge_rebuild_ntx_channels_into_generic_payload_bar(
    generic_payload_bar: Mapping[str, object],
    rebuild_payload_bar: Mapping[str, object],
) -> dict[str, object]:
    """Retain generic prepared bars while adding rebuild-only contributions.

    The native VMEC coefficient bridge supplies only the rebuild
    ``face_prepared`` contribution. This helper assembles the complementary
    generic payload for the one ordinary support VJP: its prepared bars are
    retained verbatim, while both NTX runtime-channel trees and the separate
    direct-geometry payload receive their rebuild cotangent.

    ``rebuild_ntx`` is the native replacement boundary, but
    ``rebuild_payload_bar["geometry"]`` is not: it contains the direct
    geometry dependence of the lagged response.  Dropping it loses a real
    transport-to-VMEC contribution for every rebuild step.
    """

    generic_ntx = generic_payload_bar["ntx_support"]
    rebuild_ntx = rebuild_payload_bar["ntx_support"]
    return {
        "geometry": _add_float_delta_tree(
            generic_payload_bar["geometry"],
            rebuild_payload_bar["geometry"],
        ),
        "ntx_support": dataclasses.replace(
            generic_ntx,
            center_channels=_add_float_delta_tree(
                generic_ntx.center_channels,
                rebuild_ntx.center_channels,
            ),
            face_channels=_add_float_delta_tree(
                generic_ntx.face_channels,
                rebuild_ntx.face_channels,
            ),
        ),
    }


_BATCHED_REBUILD_SUPPORT_HOOK_NAMES = {
    "ntx_batched_interpolated_faces": "flat_rhs_build_support_pullback_batched_interpolated_faces",
    "ntx_batched_interpolated_faces_reuse_local_vjp_primal": "flat_rhs_build_support_pullback_batched_interpolated_faces_reuse_local_vjp_primal",
    "ntx_batched_interpolated_faces_multi_rhs_shared_primal": "flat_rhs_build_support_pullback_batched_interpolated_faces_multi_rhs_shared_primal",
    "ntx_batched_interpolated_faces_native_multi_rhs_shared_primal": "flat_rhs_build_support_pullback_batched_interpolated_faces_native_multi_rhs_shared_primal",
    "ntx_batched_interpolated_faces_native_multi_rhs_compact_shared_primal": "flat_rhs_build_support_pullback_batched_interpolated_faces_native_multi_rhs_compact_shared_primal",
    "ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal": "flat_rhs_build_support_pullback_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal",
    "ntx_batched_interpolated_faces_native_multi_rhs_compact_residual_reuse_moment_drds_jvp_shared_primal": "flat_rhs_build_support_pullback_batched_interpolated_faces_native_multi_rhs_compact_residual_reuse_moment_drds_jvp_shared_primal",
    "ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients": "flat_rhs_build_support_pullback_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients",
    "ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_directional_product_rule": "flat_rhs_build_support_pullback_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_directional_product_rule",
    "ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_coefficient_pullback": "flat_rhs_build_support_pullback_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_coefficient_pullback",
    "ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_directional_product_rule_per_energy_call_boundary": "flat_rhs_build_support_pullback_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_directional_product_rule_per_energy_call_boundary",
}

_NATIVE_VMEC_REBUILD_SUPPORT_MODES = frozenset(
    {
        "ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients",
        "ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_directional_product_rule",
        "ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_coefficient_pullback",
        "ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_directional_product_rule_per_energy_call_boundary",
    }
)


def _initial_cache_support_pullback_from_rebuild_dispatch(
    *, physics_context, flat_y, lagged_response_bars, support_payload
):
    """Run the active separate batched rebuild-support hook on the initial edge.

    This deliberately does not select a state/support joint hook.  Its output
    matches the normal rebuild support contract, with the VMEC coefficient
    channel retained separately when the active rebuild selector provides it.
    """

    rebuild_mode = str(
        getattr(physics_context, "reverse_rebuild_support_pullback_mode", "separate")
    ).strip().lower()
    hook_name = _BATCHED_REBUILD_SUPPORT_HOOK_NAMES.get(rebuild_mode)
    if hook_name is None:
        raise ValueError(
            "reverse_initial_cache_support_pullback_mode='rebuild_dispatch' "
            "requires a separate batched rebuild support mode; got "
            f"{rebuild_mode!r}."
        )
    pullback = getattr(physics_context, hook_name, None)
    if pullback is None:
        raise RuntimeError(
            "rebuild_dispatch was requested, but the active transport physics "
            f"context does not expose {hook_name}."
        )
    result = pullback(flat_y, lagged_response_bars, support_payload)
    if rebuild_mode in _NATIVE_VMEC_REBUILD_SUPPORT_MODES:
        support_bars, native_vmec_coefficient_bars = result
        return support_bars, native_vmec_coefficient_bars
    return result, None


def _objective_vector_vjp_rows(objective_vector_fn: Callable[[object], object], primal):
    """Return vector objective values and one input cotangent per output row."""

    values, pullback = jax.vjp(objective_vector_fn, primal)
    values = jnp.asarray(values)
    if values.ndim != 1:
        raise ValueError(
            "Grouped final-objective VJP requires a rank-one objective vector; "
            f"got shape {values.shape}."
        )
    basis = jnp.eye(int(values.shape[0]), dtype=values.dtype)
    return values, jax.vmap(lambda cotangent: pullback(cotangent)[0])(basis)


def _take_batched_pytree_row(tree, row_index: int):
    """Extract one leading objective row from every leaf of a pytree."""

    return jax.tree_util.tree_map(
        lambda value: jnp.asarray(value)[row_index],
        tree,
    )


def _realized_reverse_slot_branches(
    slot_active,
    slot_next_lagged_valid,
    segment_start_lagged_valid: bool,
):
    """Return the exact host dispatch order for a realized segment schedule.

    This is intentionally NumPy-only metadata handling: it never receives a
    state, objective cotangent, support payload, or NTX value. Reverse-time
    dependencies still flow entirely through device-resident step launches.
    """

    active = np.asarray(slot_active, dtype=bool).reshape(-1)
    next_valid = np.asarray(slot_next_lagged_valid, dtype=bool).reshape(-1)
    if active.shape != next_valid.shape:
        raise ValueError("realized reverse schedule arrays must have matching shapes.")
    start_valid = np.concatenate(
        [np.asarray([bool(segment_start_lagged_valid)], dtype=bool), next_valid[:-1]]
    )
    return tuple(
        (slot_index, "reuse" if start_valid[slot_index] else "rebuild")
        for slot_index in range(active.size - 1, -1, -1)
        if active[slot_index]
    )


def _run_realized_reverse_slot_dispatch(
    *,
    slot_active,
    slot_next_lagged_valid,
    segment_start_lagged_valid: bool,
    step_start_carries,
    step_primal_records,
    next_reduced_bars,
    initial_support_bars,
    take_axis0: Callable[[object, int], object],
    step_fn: Callable[[int, str, object, object, object], tuple[object, tuple[object, ...]]],
):
    """Apply static reuse/rebuild launches in exact reverse-time order.

    ``step_fn`` owns the device computation; this helper owns only the tiny
    realized schedule and the dependency-preserving Python dispatch.  Keeping
    this seam explicit permits a no-transport mock oracle for the new mode.
    """

    reduced_value = next_reduced_bars
    support_bars = initial_support_bars
    for slot_index, branch in _realized_reverse_slot_branches(
        slot_active,
        slot_next_lagged_valid,
        segment_start_lagged_valid,
    ):
        reduced_value, step_support_bars = step_fn(
            slot_index,
            branch,
            take_axis0(step_start_carries, slot_index),
            take_axis0(step_primal_records, slot_index),
            reduced_value,
        )
        support_bars = tuple(
            accumulated + increment
            for accumulated, increment in zip(
                support_bars,
                step_support_bars,
                strict=True,
            )
        )
    return reduced_value, support_bars


def _initial_lagged_response_joint_state_and_support_pullback(
    *,
    flat_y,
    cache_lagged_bars,
    rhs_lagged_bars,
    support_payload,
    joint_pullback,
):
    """Apply one joint lagged-response transpose to the total initial bar.

    The initial carry has two paths into its lagged response: its explicit
    cache cotangent and the cotangent induced by the initial Radau RHS/stage
    values.  The joint NTX hook must receive their sum; splitting them drops
    the latter's support/geometry contribution or rebuilds the same local
    transpose twice.
    """

    total_lagged_bars = jax.tree_util.tree_map(
        lambda cache_bar, rhs_bar: cache_bar + rhs_bar,
        cache_lagged_bars,
        rhs_lagged_bars,
    )
    return joint_pullback(flat_y, total_lagged_bars, support_payload)


TransportReverseReport = Mapping[str, object]
TransportReverseReportRunner = Callable[[], TransportReverseReport]
TransportReverseSupportSegmentExecutor = Callable[[object, bool], TransportReverseReport]
TransportReverseSupportSegmentProbe = Callable[..., TransportReverseReport]
TransportReverseArgsUpdater = Callable[[object], object]
TransportReverseReportBuilder = Callable[
    [tuple[str, ...], ReverseADParameterSet, Mapping[str, object] | None],
    TransportReverseReport,
]
TransportReverseTableResultBuilder = Callable[
    [tuple[str, ...], ReverseADParameterSet, Mapping[str, object] | None],
    "RealtimeGeometryTransportReverseTableResult",
]
NtxSupportPayloadBuilder = Callable[[object], object]
ReverseStaticSetupBuilder = Callable[..., object]
GeometryDiagnosticsBuilder = Callable[[object], Mapping[str, object]]


@dataclasses.dataclass(frozen=True, slots=True)
class RealtimeGeometryTransportReverseTableContext:
    """Runtime inputs required by the grouped realtime-geometry reverse table."""

    config: Mapping[str, Any]
    baseline_values: object
    baseline_runtime: object
    baseline_state: object
    profile_cfg: Mapping[str, Any]
    neoclassical_cfg: Mapping[str, Any]


@dataclasses.dataclass(frozen=True, slots=True)
class RealtimeGeometryTransportReverseTableRequest:
    """Non-printing grouped transport reverse table request.

    This request object is intentionally small and benchmark-agnostic.  It
    gives the production lane a stable handoff point before the heavy grouped
    runner itself is moved out of the benchmark.
    """

    objective_names: tuple[str, ...]
    parameter_set: ReverseADParameterSet
    context: RealtimeGeometryTransportReverseTableContext
    options: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "objective_names",
            normalize_transport_objective_names(self.objective_names),
        )
        _validate_transport_reverse_parameter_set(self.parameter_set)
        if not isinstance(self.context, RealtimeGeometryTransportReverseTableContext):
            raise TypeError(
                "context must be a RealtimeGeometryTransportReverseTableContext; "
                f"got {type(self.context).__name__}."
            )


@dataclasses.dataclass(frozen=True, slots=True)
class RealtimeGeometryTransportReverseTableResult:
    """JAX-native grouped transport reverse table result.

    This object is the optimization-facing result boundary: it keeps objective
    values and Jacobian blocks as device arrays until a benchmark/reporting
    layer explicitly asks for host dictionaries.
    """

    objective_labels: tuple[str, ...]
    profile_parameter_labels: tuple[str, ...]
    geometry_parameter_labels: tuple[str, ...]
    objective_values: object
    profile_gradient_matrix: object
    geometry_gradient_matrix: object

    def __post_init__(self) -> None:
        objective_names = tuple(str(name) for name in self.objective_labels)
        profile_names = tuple(str(name) for name in self.profile_parameter_labels)
        geometry_names = tuple(str(name) for name in self.geometry_parameter_labels)
        if not objective_names or any(not name for name in objective_names):
            raise ValueError("objective_labels must contain at least one non-empty label.")
        if len(set(objective_names)) != len(objective_names):
            raise ValueError(f"objective_labels must be unique; got {objective_names!r}.")
        if len(set(profile_names)) != len(profile_names):
            raise ValueError(f"profile_parameter_labels must be unique; got {profile_names!r}.")
        if len(set(geometry_names)) != len(geometry_names):
            raise ValueError(f"geometry_parameter_labels must be unique; got {geometry_names!r}.")
        object.__setattr__(self, "objective_labels", objective_names)
        object.__setattr__(self, "profile_parameter_labels", profile_names)
        object.__setattr__(self, "geometry_parameter_labels", geometry_names)
        objective_values_shape = tuple(getattr(self.objective_values, "shape", ()))
        profile_gradient_shape = tuple(getattr(self.profile_gradient_matrix, "shape", ()))
        geometry_gradient_shape = tuple(getattr(self.geometry_gradient_matrix, "shape", ()))
        if objective_values_shape != (len(objective_names),):
            raise ValueError(
                "objective_values must have shape (objective_count,); "
                f"got {objective_values_shape}, objective_count={len(objective_names)}."
            )
        if profile_gradient_shape != (len(objective_names), len(profile_names)):
            raise ValueError(
                "profile_gradient_matrix must have shape "
                "(objective_count, profile_parameter_count); "
                f"got {profile_gradient_shape}, expected {(len(objective_names), len(profile_names))}."
            )
        if geometry_gradient_shape != (len(objective_names), len(geometry_names)):
            raise ValueError(
                "geometry_gradient_matrix must have shape "
                "(objective_count, geometry_parameter_count); "
                f"got {geometry_gradient_shape}, expected {(len(objective_names), len(geometry_names))}."
            )


@dataclasses.dataclass(frozen=True, slots=True)
class RealtimeGeometryPayloadPullbackResult:
    """VMEC-harmonic pullback result for transport support-payload cotangents."""

    geometry_gradient_matrix: object
    geometry_branch_gradient_matrix: object | None
    ntx_support_branch_gradient_matrix: object | None
    component_gradient_matrices: Mapping[str, object]
    component_geometry_branch_matrices: Mapping[str, object]
    component_ntx_support_branch_matrices: Mapping[str, object]
    pullback_mode: str = "payload_state_raw_block_transpose"


@dataclasses.dataclass(frozen=True, slots=True)
class RealtimeGeometryTransportReverseAssemblyResult:
    """Non-printing grouped transport table plus VMEC payload pullback result."""

    table_result: RealtimeGeometryTransportReverseTableResult
    payload_pullback_result: RealtimeGeometryPayloadPullbackResult


@dataclasses.dataclass(frozen=True, slots=True)
class RealtimeGeometryTransportReverseGroupedInputs:
    """Context plus grouped runner for realtime-geometry transport reverse tables."""

    table_context: RealtimeGeometryTransportReverseTableContext
    run_grouped_report: TransportReverseReportRunner


@dataclasses.dataclass(frozen=True, slots=True)
class RealtimeGeometrySupportSegmentCoreSetup:
    """Reusable setup for the realtime-geometry segmented support reverse core."""

    combined_geometry_payload: bool
    payload_kind: str
    ntx_surface_backend: str
    ntx_support_payload: object
    support_payload: object
    profile_values: object
    support_probe_cotangent_mode: str
    reverse_setup: object
    early_geometry_diagnostics: Mapping[str, object] | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class RealtimeGeometrySupportCotangentResult:
    """Grouped transport profile/support cotangents before VMEC payload pullback."""

    objective_values: object
    profile_gradient_matrix: object
    support_bars: object
    support_component_bars_by_name: Mapping[str, object]
    support_reuse_count: int
    support_rebuild_count: int
    initial_cache_pullback_used: bool
    initial_cache_pullback_skipped: bool
    # Isolated native-NTX VMEC coefficient bars.  These are intentionally
    # outside ``support_bars`` because they are cotangents of the traceable
    # face surfaces, not of support-payload leaves.
    native_vmec_face_coefficient_bars: Mapping[str, object] | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class RealtimeGeometrySupportReverseDependencies:
    """Benchmark-supplied callbacks for the migrated support reverse kernel."""

    initial_er_root_enabled: Callable[[Mapping[str, Any], str], bool]
    initial_state_for_parameter_vector: Callable[..., object]
    state_with_initial_er_root_ad: Callable[..., object]
    reverse_initial_carry_from_state_with_static_setup: Callable[..., object]
    objective_scalar_by_index: Callable[[object, object, int], object]
    add_trees: Callable[[object, object], object]
    initial_er_selected_root_profile: Callable[..., object]
    initial_er_charge_flux_residuals: Callable[..., object]
    initial_er_charge_flux_residual_scalar: Callable[..., object]
    initial_er_charge_flux_residual_er_derivative: Callable[..., object]
    compact_initial_er_state_pullback: Callable[..., object]
    compact_initial_er_ntx_support_pullback_leaves: Callable[..., object]
    runtime_with_geometry_payload: Callable[[object, object], object]
    runtime_with_ntx_support_payload: Callable[[object, object], object]
    runtime_with_realtime_geometry_reverse_support_payload: Callable[[object, object], object]

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if not callable(value):
                raise TypeError(f"{field.name} must be callable.")


@dataclasses.dataclass(frozen=True, slots=True)
class RealtimeGeometryReverseStaticSetup:
    """Static solver reverse setup shared by benchmarks and optimization callers."""

    solver: object
    solve_vector_field: object
    prepared_rollout: object
    execution_context: object
    stop_after_accepted_steps: int | None
    max_total_steps: int
    reverse_segment_length: int | None
    require_final_time: bool = False
    # Optional compact scalar trace from the static schedule probe. This is
    # intentionally not a per-step primal tape or an additional carry.
    schedule_artifact: object | None = None
    schedule_segment_start_carries: object | None = None
    schedule_final_carry: object | None = None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class _ThetaReverseCarry:
    t: Any
    y: Any
    lagged_response_cache: Any
    lagged_response_valid: Any
    lagged_reference_y: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class _ThetaAcceptedStepReducedCotangent:
    y: Any
    lagged_response_cache: Any
    lagged_reference_y: Any


@dataclasses.dataclass(frozen=True, eq=False)
class _ThetaReversePhysicsContext:
    unpack_flat: Any
    pack_flat: Any = None
    project_flat: Any = None
    flat_rhs: Any = None
    flat_rhs_with_lagged_response: Any = None
    pullback_build_lagged_response: Any = None
    flat_rhs_lagged_response_pullback: Any = None
    flat_rhs_state_pullback: Any = None
    flat_rhs_build_support_pullback: Any = None
    flat_rhs_lagged_response_support_pullback: Any = None
    reverse_lagged_branch_schedule: tuple[bool, ...] | None = None


@dataclasses.dataclass(frozen=True, eq=False)
class _ThetaPreparedReverseRollout:
    solver: Any
    state: Any
    vector_field: Any
    species: Any
    temperature_active_mask: Any
    fixed_temperature_profile: Any
    density_floor: Any
    temperature_floor: Any
    initial_carry: _ThetaReverseCarry
    physics_context: _ThetaReversePhysicsContext


@dataclasses.dataclass(frozen=True, eq=False)
class _ThetaReverseExecutionContext:
    solver: Any
    prepared_rollout: _ThetaPreparedReverseRollout
    physics_context: _ThetaReversePhysicsContext


@dataclasses.dataclass(frozen=True, eq=False)
class _ThetaReverseScheduleTrace:
    accepted_mask: Any
    active_mask: Any
    t_start: Any
    y_start: Any
    lagged_response_cache_start: Any
    lagged_response_valid_start: Any
    err_norms: Any
    attempted_dts: Any
    next_dts: Any
    step_ts: Any


def _initial_direct_rhs_support_pullback_batched(
    *, carry0, carry0_bars, kernel_context, flat_rhs_direct_support_pullback, support_payload
):
    """Transpose the one direct RHS evaluation used to construct carry zero."""
    if flat_rhs_direct_support_pullback is None:
        return None
    objective_count = int(jnp.asarray(carry0_bars.prev_stages).shape[0])
    rhs_bars = jnp.sum(
        jnp.asarray(carry0_bars.prev_stages).reshape(
            (objective_count, int(kernel_context.num_stages), -1)
        ),
        axis=1,
    )
    # The retained scan-database support is numerical (geometry plus compact
    # tables) after the scan record has been stripped from the rollout
    # runtime.  It can therefore take the objective axis natively.  Keep the
    # conservative scalar path for live Lij payloads, whose scan-surface
    # metadata has static Boozer arrays that cannot be reconstructed by a
    # generic ``vmap``.
    if isinstance(support_payload, dict) and "database" in support_payload:
        return jax.vmap(
            lambda rhs_bar: flat_rhs_direct_support_pullback(
                carry0.t, carry0.y, rhs_bar, support_payload
            )
        )(rhs_bars)

    # Live support compatibility route: stack only numerical bars after each
    # scalar owner call, preserving static surface metadata exactly.
    support_bars = tuple(
        flat_rhs_direct_support_pullback(
            carry0.t, carry0.y, rhs_bars[index], support_payload
        )
        for index in range(objective_count)
    )
    return jax.tree_util.tree_map(
        lambda *values: jnp.stack(tuple(jnp.asarray(value) for value in values)),
        *support_bars,
    )
    next_recent_reject_count: Any
    next_regrowth_cooldown: Any
    next_easy_growth_streak: Any
    next_lagged_response_valid: Any


@dataclasses.dataclass(frozen=True, eq=False)
class _ThetaReverseScheduleRolloutResult:
    final_step_state: Any
    final_carry: _ThetaReverseCarry
    trace: _ThetaReverseScheduleTrace
    attempt_count: Any
    accepted_count: Any
    completed: Any
    failed: Any
    fail_code: Any


TRANSPORT_REVERSE_OBJECTIVE_LABELS: tuple[str, ...] = (
    "softmax_Er",
    "net_total_power_volume_average_mw_m3",
    "Er_transition_left",
    "Er_transition_right",
    "Er2_volume_average",
    "Er_volume_average",
    "electron_temperature_volume_average_keV",
    "total_pressure_volume_average",
    "alpha_power_volume_average_mw_m3",
    "bootstrap_current_softmax_abs_scaled",
)
TRANSPORT_REVERSE_PROFILE_PARAMETER_ORDER: tuple[str, ...] = (
    "n0",
    "T0",
    "density_shape_power",
    "temperature_shape_power",
)


def initial_er_root_ad_mode(value: str | None) -> str:
    mode = str(value or "off").strip().lower()
    aliases = {
        "none": "off",
        "false": "off",
        "0": "off",
        "jax": "jax_selected_root",
        "selected_jax": "jax_selected_root",
        "jax_selected": "jax_selected_root",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"off", "jax_selected_root"}:
        raise ValueError("initial_er_root_ad must be one of: off, jax_selected_root")
    return mode


def initial_er_root_enabled(config: Mapping[str, Any], mode: str) -> bool:
    mode = initial_er_root_ad_mode(mode)
    if mode == "off":
        return False
    profiles_cfg = config.get("profiles", {})
    init_mode = str(profiles_cfg.get("er_initialization_mode", "analytical")).strip().lower()
    return init_mode in {
        "ambipolar_min_entropy",
        "ambipolar_best_root",
        "ambipolarity_best_root",
    }


def add_trees(lhs, rhs):
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs
    return jax.tree_util.tree_map(lambda a, b: a + b, lhs, rhs)


def parameterized_profile_set(
    profile_cfg: Mapping[str, Any],
    geometry,
    n_species: int,
    *,
    parameter_name: str,
    parameter_value,
):
    cfg = dict(profile_cfg)
    cfg[parameter_name] = parameter_value
    model = AnalyticalProfileModel(
        n0=cfg.get("n0", 4.21),
        n_edge=cfg.get("n_edge", 0.6),
        T0=cfg.get("T0", 17.8),
        T_edge=cfg.get("T_edge", 0.7),
        c_density=None if cfg.get("c_density") is None else tuple(cfg.get("c_density")),
        c_temperature=None if cfg.get("c_temperature") is None else tuple(cfg.get("c_temperature")),
        density_shape_power=cfg.get("density_shape_power", 2.0),
        density_shape_alpha=cfg.get("density_shape_alpha", 1.0),
        temperature_shape_power=cfg.get("temperature_shape_power", 2.0),
        temperature_shape_alpha=cfg.get("temperature_shape_alpha", 1.0),
        n_scale=cfg.get("n_scale", 1.0),
        T_scale=cfg.get("T_scale", 1.0),
        er0_scale=cfg.get("er0_scale", 100.0),
        er0_peak_rho=cfg.get("er0_peak_rho", 0.8),
        charge_qp=None if cfg.get("charge_qp") is None else tuple(cfg.get("charge_qp")),
    )
    return model.build(geometry, n_species)


def initial_state_for_parameter_vector(
    parameter_values,
    *,
    baseline_state,
    profile_cfg: Mapping[str, Any],
    runtime,
    config: Mapping[str, Any] | None = None,
    initial_er_root_ad: str = "off",
):
    """Build the initial profile state from a complete runtime object.

    This public compatibility wrapper retains the existing call signature.
    Reverse profile pullbacks should call the compact helper below directly so
    a large database/support payload is not captured by their JAX trace.
    """
    return initial_state_for_parameter_vector_compact(
        parameter_values,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        geometry=runtime.geometry,
        number_species=runtime.species.number_species,
        config=config,
        initial_er_root_ad=initial_er_root_ad,
        root_runtime=runtime,
    )


def initial_state_for_parameter_vector_compact(
    parameter_values,
    *,
    baseline_state,
    profile_cfg: Mapping[str, Any],
    geometry,
    number_species: int,
    config: Mapping[str, Any] | None = None,
    initial_er_root_ad: str = "off",
    root_runtime=None,
):
    """Build an initial profile state without capturing unrelated runtime data.

    Profile construction depends solely on transport geometry and the species
    count.  In particular it does not use an NTX database, its recorded scan
    primal, or flux-model support payload.  Keeping those out of this function
    is important because this map is itself differentiated at the beginning
    of every segmented reverse sweep.
    """
    cfg = dict(profile_cfg)
    values_arr = jnp.asarray(parameter_values)
    if int(values_arr.shape[0]) == len(PROFILE_PARAMETER_ORDER):
        parameter_order = PROFILE_PARAMETER_ORDER
    else:
        parameter_order = TRANSPORT_REVERSE_PROFILE_PARAMETER_ORDER
    for name, value in zip(parameter_order, values_arr):
        cfg[name] = value
    profile_set = parameterized_profile_set(
        cfg,
        geometry,
        number_species,
        parameter_name=TRANSPORT_REVERSE_PROFILE_PARAMETER_ORDER[0],
        parameter_value=cfg[TRANSPORT_REVERSE_PROFILE_PARAMETER_ORDER[0]],
    )
    density_state = jnp.asarray(profile_set.density, dtype=baseline_state.density.dtype) / 1.0e20
    temperature_state = jnp.asarray(profile_set.temperature, dtype=baseline_state.pressure.dtype) / 1.0e3
    solver_cfg = {} if config is None else dict(config.get("transport_solver", {}))
    fallback_solver_cfg = {} if config is None else dict(config.get("solver", {}))
    density_floor = solver_cfg.get("density_floor", fallback_solver_cfg.get("density_floor", 1.0e-6))
    temperature_floor = solver_cfg.get("temperature_floor", fallback_solver_cfg.get("temperature_floor"))
    # Match _orchestrator._build_state exactly.  In particular, a species at
    # zero configured concentration can still have a fixed, finite configured
    # temperature when it participates in the initial ambipolar root.
    temperature_state = safe_temperature(temperature_state, temperature_floor)
    pressure_state = temperature_state * safe_density(density_state, density_floor)
    state = dataclasses.replace(
        baseline_state,
        density=density_state,
        pressure=pressure_state,
    )
    mode = initial_er_root_ad_mode(initial_er_root_ad)
    if mode != "off":
        if config is None:
            raise ValueError("config is required when initial_er_root_ad is enabled.")
        if root_runtime is None:
            raise ValueError("root_runtime is required when initial_er_root_ad is enabled.")
        state = state_with_initial_er_root_ad(
            state, config=config, runtime=root_runtime, mode=mode
        )
    return state


def initial_er_root_state_bar(state, er_profile, finite_mask, state_bar, *, runtime):
    dres_der = initial_er_charge_flux_residual_er_derivative(
        state,
        er_profile,
        runtime=runtime,
    )
    safe_dres_der = jnp.where(
        jnp.abs(dres_der) > jnp.asarray(1.0e-30, dtype=dres_der.dtype),
        dres_der,
        jnp.inf,
    )
    residual_bar = jnp.where(
        finite_mask,
        -jnp.asarray(state_bar.Er) / safe_dres_der,
        0.0,
    )
    state_residual_bar = compact_initial_er_state_pullback(
        residual_scalar_fn=initial_er_charge_flux_residual_scalar,
        state=state,
        er_profile=er_profile,
        residual_bars=residual_bar,
        runtime=runtime,
    )
    direct_bar = dataclasses.replace(state_bar, Er=jnp.zeros_like(state.Er))
    return add_trees(direct_bar, state_residual_bar)


def state_with_initial_er_root_ad(state, *, config: Mapping[str, Any], runtime, mode: str):
    if not initial_er_root_enabled(config, mode):
        return state

    @jax.custom_vjp
    def _replace_er_with_selected_root(state_inner):
        er_profile, _ = initial_er_selected_root_profile(state_inner, config=dict(config), runtime=runtime)
        return dataclasses.replace(state_inner, Er=er_profile)

    def _replace_er_fwd(state_inner):
        er_profile, finite_mask = initial_er_selected_root_profile(
            state_inner,
            config=dict(config),
            runtime=runtime,
        )
        return dataclasses.replace(state_inner, Er=er_profile), (state_inner, er_profile, finite_mask)

    def _replace_er_bwd(residuals, state_bar):
        state_inner, er_profile, finite_mask = residuals
        return (
            initial_er_root_state_bar(
                state_inner,
                er_profile,
                finite_mask,
                state_bar,
                runtime=runtime,
            ),
        )

    _replace_er_with_selected_root.defvjp(_replace_er_fwd, _replace_er_bwd)
    return _replace_er_with_selected_root(state)


def softmax_objective(er_profile: jax.Array, *, beta: float = 16.0) -> jax.Array:
    beta_arr = jnp.asarray(beta, dtype=er_profile.dtype)
    return jax.scipy.special.logsumexp(beta_arr * er_profile) / beta_arr


def smooth_root_proxy(
    er_profile: jax.Array,
    rho_grid: jax.Array,
    *,
    beta: float = 24.0,
    eps: float = 1.0e-4,
):
    beta_arr = jnp.asarray(beta, dtype=er_profile.dtype)
    eps_arr = jnp.asarray(eps, dtype=er_profile.dtype)
    smooth_abs = jnp.sqrt(er_profile * er_profile + eps_arr * eps_arr)
    weights = jnp.exp(-beta_arr * smooth_abs)
    return jnp.sum(rho_grid * weights) / jnp.maximum(jnp.sum(weights), jnp.asarray(1.0e-30, dtype=er_profile.dtype))


def volume_average(profile: jax.Array, geometry) -> jax.Array:
    volume = jnp.trapezoid(jnp.asarray(geometry.Vprime), x=jnp.asarray(geometry.r_grid))
    integral = jnp.trapezoid(profile * jnp.asarray(geometry.Vprime), x=jnp.asarray(geometry.r_grid))
    return integral / jnp.maximum(volume, jnp.asarray(1.0e-30, dtype=integral.dtype))


def alpha_power_volume_average(final_state, runtime) -> jax.Array:
    source_models = runtime.models.source or {}
    pressure_source_model = source_models.get("temperature") if isinstance(source_models, dict) else None
    if pressure_source_model is None:
        return jnp.asarray(0.0, dtype=final_state.pressure.dtype)
    raw_sources = pressure_source_model(final_state)
    alpha_power = raw_sources.get("AlphaPower") if isinstance(raw_sources, dict) else None
    if alpha_power is None:
        return jnp.asarray(0.0, dtype=final_state.pressure.dtype)
    alpha_mw_m3 = PRESSURE_SOURCE_STATE_TO_MW_M3 * jnp.asarray(alpha_power, dtype=final_state.pressure.dtype)
    return volume_average(alpha_mw_m3, runtime.geometry)


def net_total_power_volume_average(final_state, runtime) -> jax.Array:
    """Signed volume average of alpha, bremsstrahlung, and external power."""
    source_models = runtime.models.source or {}
    pressure_source_model = source_models.get("temperature") if isinstance(source_models, dict) else None
    return compute_net_total_power_volume_average_mw_m3(
        final_state,
        pressure_source_model,
        runtime.geometry,
    )


def electron_temperature_volume_average(final_state, runtime) -> jax.Array:
    species_idx = getattr(runtime.species, "species_idx", {})
    electron_idx = species_idx.get("e", 0)
    temperature = jnp.asarray(final_state.temperature[electron_idx], dtype=final_state.pressure.dtype)
    return volume_average(temperature, runtime.geometry)


def total_pressure_volume_average(final_state, runtime) -> jax.Array:
    total_pressure = jnp.sum(jnp.asarray(final_state.pressure, dtype=final_state.pressure.dtype), axis=0)
    return volume_average(total_pressure, runtime.geometry)


def bootstrap_current_softmax_abs_scaled(
    final_state,
    runtime,
    *,
    beta: float = 128.0,
    eps: float = 1.0e-12,
) -> jax.Array:
    """Direct momentum-corrected smooth max(abs(Jboot)).

    A value of 0.1 corresponds to 10 kA/m^2 in physical current density.
    """

    flux_model = getattr(getattr(runtime, "models", None), "flux", None)
    neoclassical_model = getattr(flux_model, "neoclassical_model", flux_model)
    corrected_fluxes_fn = getattr(neoclassical_model, "evaluate_momentum_corrected_fluxes", None)
    if not callable(corrected_fluxes_fn):
        raise NotImplementedError(
            "bootstrap_current_softmax_abs_scaled requires a realtime NTX model with "
            "evaluate_momentum_corrected_fluxes; refusing to use the static database "
            "or uncorrected lagged Upar path."
        )
    corrected_fluxes = corrected_fluxes_fn(final_state)
    upar = corrected_fluxes.get("Upar_neo", corrected_fluxes.get("Upar", None))
    if upar is None:
        raise ValueError("momentum-corrected realtime NTX fluxes did not return Upar.")
    charge_qp = jnp.asarray(runtime.species.charge_qp, dtype=final_state.pressure.dtype)
    current_weights = jnp.sign(charge_qp)
    upar_arr = jnp.asarray(upar, dtype=final_state.pressure.dtype)
    upar_physical = jnp.asarray(DENSITY_STATE_TO_PHYSICAL, dtype=upar_arr.dtype) * upar_arr
    if int(upar_arr.shape[0]) == int(charge_qp.shape[0]):
        jboot = jnp.sum(upar_physical * current_weights[:, None], axis=0)
    else:
        jboot = jnp.sum(upar_physical * current_weights[None, :], axis=1)
    jboot = jboot * jnp.asarray(elementary_charge * 1.0e-5, dtype=final_state.pressure.dtype)
    smooth_abs = jnp.sqrt(jboot * jboot + jnp.asarray(eps, dtype=jboot.dtype) ** 2)
    beta_arr = jnp.asarray(beta, dtype=jboot.dtype)
    return jax.scipy.special.logsumexp(beta_arr * smooth_abs) / beta_arr


def bootstrap_current_softmax_abs_value_and_upar_bar(
    final_state,
    runtime,
    fluxes: Mapping[str, Any],
    *,
    beta: float = 128.0,
    eps: float = 1.0e-12,
) -> tuple[jax.Array, jax.Array]:
    """Return smooth max(abs(Jboot)) and the compact corrected-Upar cotangent."""

    upar = fluxes.get("Upar_neo", fluxes.get("Upar", None))
    if upar is None:
        raise ValueError("bootstrap current objective requires Upar or Upar_neo fluxes.")
    dtype = jnp.asarray(final_state.pressure).dtype
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
    return value, upar_bar


def objective_scalar_by_index(final_state, runtime, objective_index: int):
    objective_name = TRANSPORT_REVERSE_OBJECTIVE_LABELS[int(objective_index)]
    er = jnp.asarray(final_state.Er)
    if objective_name == "softmax_Er":
        return softmax_objective(er)
    if objective_name == "net_total_power_volume_average_mw_m3":
        return net_total_power_volume_average(final_state, runtime)
    if objective_name == "Er_transition_left":
        return er[max(0, min(20, int(er.shape[-1]) - 1))]
    if objective_name == "Er_transition_right":
        return er[max(0, min(21, int(er.shape[-1]) - 1))]
    if objective_name == "Er2_volume_average":
        return volume_average(er * er, runtime.geometry)
    if objective_name == "Er_volume_average":
        return volume_average(er, runtime.geometry)
    if objective_name == "electron_temperature_volume_average_keV":
        return electron_temperature_volume_average(final_state, runtime)
    if objective_name == "total_pressure_volume_average":
        return total_pressure_volume_average(final_state, runtime)
    if objective_name == "alpha_power_volume_average_mw_m3":
        return alpha_power_volume_average(final_state, runtime)
    if objective_name == "bootstrap_current_softmax_abs_scaled":
        return bootstrap_current_softmax_abs_scaled(final_state, runtime)
    raise ValueError(f"Unknown objective index {objective_index}: {objective_name!r}")


def lagged_response_pullback_from_owner(solve_vector_field):
    owner = getattr(solve_vector_field, "__self__", None)
    if owner is None:
        return None
    pullback_fn = getattr(owner, "pullback_build_lagged_response", None)
    return pullback_fn if callable(pullback_fn) else None


def _require_reverse_solver_radau(solver, capability: str) -> None:
    if isinstance(solver, RADAUSolver):
        return
    solver_name = type(solver).__name__
    raise ValueError(
        f"{capability} currently requires RADAUSolver because the reverse path "
        "uses Radau-private accepted-step rollout/VJP/segment rules; "
        f"got {solver_name}. Add the theta implementation behind this solver-neutral "
        "dispatch layer before enabling theta for this reverse capability."
    )


def _require_reverse_execution_context_radau(execution_context, capability: str) -> None:
    kernel_context = getattr(execution_context, "kernel_context", None)
    if hasattr(kernel_context, "radau_transform"):
        return
    context_name = type(execution_context).__name__
    raise ValueError(
        f"{capability} currently requires a Radau reverse execution context because "
        "the reverse path uses Radau-private accepted-step rollout/VJP/segment rules; "
        f"got {context_name}. Add the theta implementation behind this solver-neutral "
        "dispatch layer before enabling theta for this reverse capability."
    )


def _is_theta_reverse_solver(solver) -> bool:
    return isinstance(solver, (ThetaMethodSolver, NewtonThetaMethodSolver))


def _build_prepared_theta_reverse_rollout(*, solver, state, vector_field, species):
    temperature_active_mask, fixed_temperature_profile = _extract_fixed_temperature_projection(vector_field)
    density_floor, temperature_floor = _extract_state_regularization(vector_field)
    flat_state0, unpack_flat, _unpack_packed, pack_flat, project_flat = _make_solver_state_transform(
        state,
        species,
        temperature_active_mask=temperature_active_mask,
        fixed_temperature_profile=fixed_temperature_profile,
        density_floor=density_floor,
        temperature_floor=temperature_floor,
    )
    args = (species,)
    kwargs = {}
    flat_rhs = _flat_rhs_factory(
        unpack_flat,
        vector_field,
        args,
        kwargs,
        project_flat=project_flat,
    )
    flat_rhs_with_lagged_response = _flat_rhs_with_lagged_response_factory(
        unravel=unpack_flat,
        vector_field=vector_field,
        args=args,
        kwargs=kwargs,
        project_flat=project_flat,
    )
    flat_rhs_lagged_response_pullback = _flat_rhs_lagged_response_pullback_factory(
        unpack_flat,
        vector_field,
        args,
        kwargs,
        project_flat=project_flat,
    )
    flat_rhs_state_pullback = _flat_rhs_state_pullback_factory(
        unpack_flat,
        pack_flat,
        vector_field,
        args,
        kwargs,
        project_flat=project_flat,
    )
    pullback_build_lagged_response = lagged_response_pullback_from_owner(vector_field)
    flat_rhs_build_support_pullback = _flat_rhs_build_support_pullback_factory(
        unpack_flat,
        vector_field,
        args,
        kwargs,
        project_flat=project_flat,
    )
    flat_rhs_lagged_response_support_pullback = _flat_rhs_lagged_response_support_pullback_factory(
        unpack_flat,
        vector_field,
        args,
        kwargs,
        project_flat=project_flat,
    )
    dtype = jnp.asarray(flat_state0).dtype
    initial_carry = _ThetaReverseCarry(
        t=jnp.asarray(getattr(solver, "t0", 0.0), dtype=dtype),
        y=flat_state0,
        lagged_response_cache=None,
        lagged_response_valid=jnp.asarray(False),
        lagged_reference_y=flat_state0,
    )
    return _ThetaPreparedReverseRollout(
        solver=solver,
        state=state,
        vector_field=vector_field,
        species=species,
        temperature_active_mask=temperature_active_mask,
        fixed_temperature_profile=fixed_temperature_profile,
        density_floor=density_floor,
        temperature_floor=temperature_floor,
        initial_carry=initial_carry,
        physics_context=_ThetaReversePhysicsContext(
            unpack_flat=unpack_flat,
            pack_flat=pack_flat,
            project_flat=project_flat,
            flat_rhs=flat_rhs,
            flat_rhs_with_lagged_response=flat_rhs_with_lagged_response,
            pullback_build_lagged_response=pullback_build_lagged_response,
            flat_rhs_lagged_response_pullback=flat_rhs_lagged_response_pullback,
            flat_rhs_state_pullback=flat_rhs_state_pullback,
            flat_rhs_build_support_pullback=flat_rhs_build_support_pullback,
            flat_rhs_lagged_response_support_pullback=flat_rhs_lagged_response_support_pullback,
        ),
    )


def _build_theta_reverse_execution_context(*, solver, prepared_rollout):
    return _ThetaReverseExecutionContext(
        solver=solver,
        prepared_rollout=prepared_rollout,
        physics_context=prepared_rollout.physics_context,
    )


def _theta_solver_with_reverse_probe_limits(solver, *, max_total_steps, stop_after_accepted_steps):
    replacements = {}
    if max_total_steps is not None:
        replacements["max_steps"] = int(max(1, max_total_steps))
    if stop_after_accepted_steps is not None:
        replacements["stop_after_accepted_steps"] = int(max(1, stop_after_accepted_steps))
    if not replacements:
        return solver
    init_params = inspect.signature(type(solver).__init__).parameters
    accepted_kinds = {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }
    solver_kwargs = {
        name: getattr(solver, name)
        for name, param in init_params.items()
        if name != "self"
        and hasattr(solver, name)
        and param.kind in accepted_kinds
    }
    solver_kwargs.update(replacements)
    return type(solver)(**solver_kwargs)


def _theta_reverse_adaptive_schedule_rollout(
    execution_context,
    initial_carry,
    *,
    max_total_steps,
    stop_after_accepted_steps,
):
    """Run theta's forward schedule probe and expose Radau-like metadata.

    Theta and Newton-theta use the shared theta attempt helpers directly so
    reverse setup can see real accepted/rejected schedule rows.  The remaining
    theta reverse work is the realized-schedule VJP/segmented cotangent pass.
    """

    prepared = execution_context.prepared_rollout
    solver = _theta_solver_with_reverse_probe_limits(
        execution_context.solver,
        max_total_steps=max_total_steps,
        stop_after_accepted_steps=stop_after_accepted_steps,
    )
    initial_state = prepared.physics_context.unpack_flat(initial_carry.y)
    if isinstance(solver, ThetaMethodSolver):
        rollout = _theta_basic_adaptive_schedule_rollout(
            solver,
            initial_state,
            prepared.vector_field,
            prepared.species,
            max_total_steps=max_total_steps,
            stop_after_accepted_steps=stop_after_accepted_steps,
        )
        final_carry = _ThetaReverseCarry(
            t=rollout.final_step_state.t,
            y=rollout.final_step_state.y,
            lagged_response_cache=rollout.final_step_state.reuse_state.lagged_response_cache,
            lagged_response_valid=rollout.final_step_state.reuse_state.lagged_response_valid,
            lagged_reference_y=rollout.final_step_state.reuse_state.lagged_reference_y,
        )
        trace = _ThetaReverseScheduleTrace(
            accepted_mask=rollout.trace.accepted_mask,
            active_mask=rollout.trace.active_mask,
            t_start=rollout.trace.t_start,
            y_start=rollout.trace.y_start,
            lagged_response_cache_start=rollout.trace.lagged_response_cache_start,
            lagged_response_valid_start=rollout.trace.lagged_response_valid_start,
            err_norms=rollout.trace.err_norms,
            attempted_dts=rollout.trace.attempted_dts,
            next_dts=rollout.trace.next_dts,
            step_ts=rollout.trace.step_ts,
            next_recent_reject_count=rollout.trace.next_recent_reject_count,
            next_regrowth_cooldown=rollout.trace.next_regrowth_cooldown,
            next_easy_growth_streak=rollout.trace.next_easy_growth_streak,
            next_lagged_response_valid=rollout.trace.next_lagged_response_valid,
        )
        return _ThetaReverseScheduleRolloutResult(
            final_step_state=rollout.final_step_state,
            final_carry=final_carry,
            trace=trace,
            attempt_count=rollout.attempt_count,
            accepted_count=rollout.accepted_count,
            completed=rollout.completed,
            failed=rollout.failed,
            fail_code=rollout.fail_code,
        )
    if isinstance(solver, NewtonThetaMethodSolver):
        rollout = _theta_newton_adaptive_schedule_rollout(
            solver,
            initial_state,
            prepared.vector_field,
            prepared.species,
            max_total_steps=max_total_steps,
            stop_after_accepted_steps=stop_after_accepted_steps,
        )
        final_carry = _ThetaReverseCarry(
            t=rollout.final_step_state.t,
            y=rollout.final_step_state.y,
            lagged_response_cache=rollout.final_step_state.reuse_state.lagged_response_cache,
            lagged_response_valid=rollout.final_step_state.reuse_state.lagged_response_valid,
            lagged_reference_y=rollout.final_step_state.reuse_state.lagged_reference_y,
        )
        trace = _ThetaReverseScheduleTrace(
            accepted_mask=rollout.trace.accepted_mask,
            active_mask=rollout.trace.active_mask,
            t_start=rollout.trace.t_start,
            y_start=rollout.trace.y_start,
            lagged_response_cache_start=rollout.trace.lagged_response_cache_start,
            lagged_response_valid_start=rollout.trace.lagged_response_valid_start,
            err_norms=rollout.trace.err_norms,
            attempted_dts=rollout.trace.attempted_dts,
            next_dts=rollout.trace.next_dts,
            step_ts=rollout.trace.step_ts,
            next_recent_reject_count=rollout.trace.next_recent_reject_count,
            next_regrowth_cooldown=rollout.trace.next_regrowth_cooldown,
            next_easy_growth_streak=rollout.trace.next_easy_growth_streak,
            next_lagged_response_valid=rollout.trace.next_lagged_response_valid,
        )
        return _ThetaReverseScheduleRolloutResult(
            final_step_state=rollout.final_step_state,
            final_carry=final_carry,
            trace=trace,
            attempt_count=rollout.attempt_count,
            accepted_count=rollout.accepted_count,
            completed=rollout.completed,
            failed=rollout.failed,
            fail_code=rollout.fail_code,
        )

    result = solver.solve(prepared.state, prepared.vector_field, prepared.species)
    final_flat, *_ = _make_solver_state_transform(
        result["final_state"],
        prepared.species,
        temperature_active_mask=prepared.temperature_active_mask,
        fixed_temperature_profile=prepared.fixed_temperature_profile,
        density_floor=prepared.density_floor,
        temperature_floor=prepared.temperature_floor,
    )
    accepted_count = jnp.asarray(result.get("n_steps", 0), dtype=jnp.int32)
    attempt_count_value = int(np.asarray(jax.device_get(accepted_count)))
    dtype = jnp.asarray(initial_carry.y).dtype
    active_mask = jnp.ones((attempt_count_value,), dtype=bool)
    accepted_mask = jnp.ones((attempt_count_value,), dtype=bool)
    zero_float_trace = jnp.zeros((attempt_count_value,), dtype=dtype)
    zero_int_trace = jnp.zeros((attempt_count_value,), dtype=jnp.int32)
    lagged_valid = result.get("final_reuse_lagged_response_valid", False)
    if lagged_valid is None:
        lagged_valid = False
    lagged_valid_trace = jnp.full((attempt_count_value,), lagged_valid, dtype=bool)
    final_carry = _ThetaReverseCarry(
        t=result["final_time"],
        y=final_flat,
        lagged_response_cache=None,
        lagged_response_valid=jnp.asarray(lagged_valid, dtype=bool),
        lagged_reference_y=final_flat,
    )
    trace = _ThetaReverseScheduleTrace(
        accepted_mask=accepted_mask,
        active_mask=active_mask,
        t_start=zero_float_trace,
        y_start=jnp.broadcast_to(final_flat[None, ...], (attempt_count_value,) + jnp.shape(final_flat)),
        lagged_response_cache_start=None,
        lagged_response_valid_start=lagged_valid_trace,
        err_norms=zero_float_trace,
        attempted_dts=zero_float_trace,
        next_dts=zero_float_trace,
        step_ts=zero_float_trace,
        next_recent_reject_count=zero_int_trace,
        next_regrowth_cooldown=zero_int_trace,
        next_easy_growth_streak=zero_int_trace,
        next_lagged_response_valid=lagged_valid_trace,
    )
    return _ThetaReverseScheduleRolloutResult(
        final_step_state=None,
        final_carry=final_carry,
        trace=trace,
        attempt_count=accepted_count,
        accepted_count=accepted_count,
        completed=result["done"],
        failed=result["failed"],
        fail_code=result["fail_code"],
    )


def _build_prepared_reverse_accepted_rollout(*, solver, state, vector_field, species):
    # Solver-neutral seam: theta support should be added here, not in callers.
    if _is_theta_reverse_solver(solver):
        return _build_prepared_theta_reverse_rollout(
            solver=solver,
            state=state,
            vector_field=vector_field,
            species=species,
        )
    _require_reverse_solver_radau(solver, "prepared reverse accepted rollout")
    return _build_prepared_radau_accepted_rollout(
        solver=solver,
        state=state,
        vector_field=vector_field,
        species=species,
    )


def _build_prepared_reverse_execution_context(*, solver, prepared_rollout):
    if _is_theta_reverse_solver(solver):
        return _build_theta_reverse_execution_context(
            solver=solver,
            prepared_rollout=prepared_rollout,
        )
    _require_reverse_solver_radau(solver, "prepared reverse execution context")
    return _build_prepared_radau_execution_context(
        solver=solver,
        prepared_rollout=prepared_rollout,
    )


def _reverse_adaptive_schedule_rollout(
    execution_context,
    initial_carry,
    *,
    max_total_steps,
    stop_after_accepted_steps,
    capture_segment_length=None,
):
    if isinstance(execution_context, _ThetaReverseExecutionContext):
        return _theta_reverse_adaptive_schedule_rollout(
            execution_context,
            initial_carry,
            max_total_steps=max_total_steps,
            stop_after_accepted_steps=stop_after_accepted_steps,
        )
    _require_reverse_execution_context_radau(execution_context, "reverse adaptive schedule rollout")
    return _radau_adaptive_schedule_rollout(
        execution_context,
        initial_carry,
        max_total_steps=max_total_steps,
        stop_after_accepted_steps=stop_after_accepted_steps,
        capture_segment_length=capture_segment_length,
    )


def _theta_adaptive_final_y_realized_schedule_vjp_fwd(
    execution_context,
    max_total_steps,
    stop_after_accepted_steps,
    reverse_segment_length,
    initial_carry,
):
    rollout = _theta_reverse_adaptive_schedule_rollout(
        execution_context,
        initial_carry,
        max_total_steps=max_total_steps,
        stop_after_accepted_steps=stop_after_accepted_steps,
    )
    active_mask = jax.lax.stop_gradient(rollout.trace.active_mask)
    accepted_mask = jax.lax.stop_gradient(rollout.trace.accepted_mask)
    attempted_dts = jax.lax.stop_gradient(rollout.trace.attempted_dts)
    next_dts = jax.lax.stop_gradient(rollout.trace.next_dts)
    next_recent_reject_count = jax.lax.stop_gradient(rollout.trace.next_recent_reject_count)
    next_regrowth_cooldown = jax.lax.stop_gradient(rollout.trace.next_regrowth_cooldown)
    next_easy_growth_streak = jax.lax.stop_gradient(rollout.trace.next_easy_growth_streak)
    next_lagged_response_valid = jax.lax.stop_gradient(rollout.trace.next_lagged_response_valid)
    segment_start_carries = None
    segmented_final_carry = None
    segmented_replay_arrays = None
    if reverse_segment_length is not None and int(reverse_segment_length) > 0:
        segment_length = int(reverse_segment_length)
        accepted_limit = int(stop_after_accepted_steps) if stop_after_accepted_steps is not None else int(max_total_steps)
        segment_count = (accepted_limit + segment_length - 1) // segment_length
        padded_count = segment_count * segment_length
        accepted_active_mask = jnp.logical_and(active_mask, accepted_mask)
        accepted_count = jnp.minimum(
            jnp.sum(accepted_active_mask.astype(jnp.int32)),
            jnp.asarray(accepted_limit, dtype=jnp.int32),
        )
        accepted_positions = jnp.nonzero(
            accepted_active_mask,
            size=accepted_limit,
            fill_value=0,
        )[0]

        def _compact_and_pad(values):
            compact = jnp.take(values, accepted_positions, axis=0)
            pad_count = padded_count - accepted_limit
            if pad_count == 0:
                return compact
            pad_values = jnp.repeat(compact[-1:], pad_count, axis=0)
            return jnp.concatenate([compact, pad_values], axis=0)

        def _compact_and_pad_tree(tree):
            if tree is None:
                return None
            return jax.tree_util.tree_map(_compact_and_pad, tree)

        replay_active_mask = jnp.concatenate(
            [
                jnp.arange(accepted_limit, dtype=jnp.int32) < accepted_count,
                jnp.zeros((padded_count - accepted_limit,), dtype=jnp.bool_),
            ],
            axis=0,
        )
        replay_attempted_dts = _compact_and_pad(attempted_dts)
        replay_next_dts = _compact_and_pad(next_dts)
        replay_next_recent_reject_count = _compact_and_pad(next_recent_reject_count)
        replay_next_regrowth_cooldown = _compact_and_pad(next_regrowth_cooldown)
        replay_next_easy_growth_streak = _compact_and_pad(next_easy_growth_streak)
        replay_next_lagged_response_valid = _compact_and_pad(next_lagged_response_valid)
        segmented_replay_arrays = (
            replay_active_mask.reshape((segment_count, segment_length)),
            replay_attempted_dts.reshape((segment_count, segment_length)),
            replay_next_dts.reshape((segment_count, segment_length)),
            replay_next_recent_reject_count.reshape((segment_count, segment_length)),
            replay_next_regrowth_cooldown.reshape((segment_count, segment_length)),
            replay_next_easy_growth_streak.reshape((segment_count, segment_length)),
            replay_next_lagged_response_valid.reshape((segment_count, segment_length)),
        )
        compact_start_carries = _ThetaReverseCarry(
            t=_compact_and_pad(jax.lax.stop_gradient(rollout.trace.t_start)),
            y=_compact_and_pad(jax.lax.stop_gradient(rollout.trace.y_start)),
            lagged_response_cache=_compact_and_pad_tree(
                jax.lax.stop_gradient(rollout.trace.lagged_response_cache_start)
            ),
            lagged_response_valid=_compact_and_pad(
                jax.lax.stop_gradient(rollout.trace.lagged_response_valid_start)
            ),
            lagged_reference_y=_compact_and_pad(jax.lax.stop_gradient(rollout.trace.y_start)),
        )
        segment_start_carries = _ThetaReverseCarry(
            t=compact_start_carries.t.reshape((segment_count, segment_length) + compact_start_carries.t.shape[1:])[:, 0],
            y=compact_start_carries.y.reshape((segment_count, segment_length) + compact_start_carries.y.shape[1:])[:, 0],
            lagged_response_cache=(
                None
                if compact_start_carries.lagged_response_cache is None
                else jax.tree_util.tree_map(
                    lambda value: value.reshape((segment_count, segment_length) + value.shape[1:])[:, 0],
                    compact_start_carries.lagged_response_cache,
                )
            ),
            lagged_response_valid=compact_start_carries.lagged_response_valid.reshape(
                (segment_count, segment_length) + compact_start_carries.lagged_response_valid.shape[1:]
            )[:, 0],
            lagged_reference_y=compact_start_carries.lagged_reference_y.reshape(
                (segment_count, segment_length) + compact_start_carries.lagged_reference_y.shape[1:]
            )[:, 0],
        )
        segmented_final_carry = rollout.final_carry
    residuals = (
        initial_carry,
        active_mask,
        accepted_mask,
        attempted_dts,
        next_dts,
        next_recent_reject_count,
        next_regrowth_cooldown,
        next_easy_growth_streak,
        next_lagged_response_valid,
        segment_start_carries,
        segmented_final_carry,
        segmented_replay_arrays,
    )
    return rollout.final_carry.y, residuals


def _reverse_adaptive_final_y_realized_schedule_vjp_fwd(
    execution_context,
    max_total_steps,
    stop_after_accepted_steps,
    reverse_segment_length,
    initial_carry,
):
    if isinstance(execution_context, _ThetaReverseExecutionContext):
        return _theta_adaptive_final_y_realized_schedule_vjp_fwd(
            execution_context,
            max_total_steps,
            stop_after_accepted_steps,
            reverse_segment_length,
            initial_carry,
        )
    _require_reverse_execution_context_radau(execution_context, "reverse realized-schedule VJP forward")
    return _radau_adaptive_final_y_realized_schedule_vjp_fwd(
        execution_context,
        max_total_steps,
        stop_after_accepted_steps,
        reverse_segment_length,
        initial_carry,
    )


def _reverse_zero_support_delta_tree_like(solver, support_payload):
    if _is_theta_reverse_solver(solver):
        return _radau_zero_support_delta_tree_like(support_payload)
    _require_reverse_solver_radau(solver, "reverse support-payload zero cotangent")
    return _radau_zero_support_delta_tree_like(support_payload)


def _reverse_align_tangent_tree_to_primal(execution_context, tangent_tree, primal_tree):
    if isinstance(execution_context, _ThetaReverseExecutionContext):
        return _radau_align_tangent_tree_to_primal(tangent_tree, primal_tree)
    _require_reverse_execution_context_radau(execution_context, "reverse tangent-tree alignment")
    return _radau_align_tangent_tree_to_primal(tangent_tree, primal_tree)


def _theta_lagged_rhs_support_pullback(
    physics_context,
    *,
    t_value,
    y_value,
    lagged_response,
    rhs_bar,
    support_payload,
):
    if (
        lagged_response is None
        or support_payload is None
        or physics_context.flat_rhs_lagged_response_support_pullback is None
    ):
        return _radau_zero_support_delta_tree_like(support_payload)
    return _radau_sanitize_support_delta_bar_tree(
        support_payload,
        physics_context.flat_rhs_lagged_response_support_pullback(
            t_value,
            y_value,
            lagged_response,
            rhs_bar,
            support_payload,
        ),
    )


def _theta_support_cotangent_for_vjp(primal, bar):
    def _sanitize_leaf(primal_leaf, bar_leaf):
        arr = jnp.asarray(primal_leaf)
        if jnp.issubdtype(arr.dtype, jnp.inexact):
            return jnp.asarray(bar_leaf, dtype=arr.dtype)
        return jnp.zeros(arr.shape, dtype=jax.dtypes.float0)

    return jax.tree_util.tree_map(_sanitize_leaf, primal, bar)


def _theta_segment_reduced_cotangent_bwd_batched_with_support_call(
    execution_context,
    cotangent_mode,
    reduced_bars,
    segment_start_carry,
    segment_arrays,
    support_payload,
):
    requested_mode = str(cotangent_mode).strip().lower()
    # Theta has a one-state implicit residual, so the normal "full" reverse lane
    # dispatches to the theta residual-transpose implementation directly.  The
    # theta_* names are kept only as diagnostics/aliases for isolating pieces.
    mode = "theta_implicit_transpose_probe" if requested_mode == "full" else requested_mode
    zero_step_bwd = mode in {"zero_step_bwd", "step_bwd_zero", "zero_accepted_step_bwd"}
    objective_count = jnp.asarray(reduced_bars.y).shape[0]
    zero_support_leaves = tuple(jax.tree_util.tree_leaves(_radau_zero_support_delta_tree_like(support_payload)))
    zero_support_bar_leaves = tuple(
        jnp.broadcast_to(jnp.asarray(leaf)[None, ...], (objective_count,) + jnp.asarray(leaf).shape)
        for leaf in zero_support_leaves
    )
    if zero_step_bwd:
        zero_reduced_bars = _ThetaAcceptedStepReducedCotangent(
            y=jnp.zeros(
                (objective_count,) + jnp.shape(segment_start_carry.y),
                dtype=jnp.asarray(segment_start_carry.y).dtype,
            ),
            lagged_response_cache=_reverse_align_tangent_tree_to_primal(
                execution_context,
                None,
                reduced_bars.lagged_response_cache,
            ),
            lagged_reference_y=jnp.zeros_like(reduced_bars.lagged_reference_y),
        )
        return zero_reduced_bars, zero_support_bar_leaves
    state_only_modes = {
        "state_only",
        "final_state",
        "theta_state_only",
        "theta_zero_lagged",
        "theta_compact_support_probe",
        "theta_implicit_transpose_probe",
    }
    if mode in state_only_modes:
        solver = execution_context.solver
        prepared = execution_context.prepared_rollout
        physics_context = prepared.physics_context
        unpack_flat = physics_context.unpack_flat
        vector_field = prepared.vector_field
        species = prepared.species
        dtype = jnp.asarray(segment_start_carry.y).dtype
        state_dim = jnp.asarray(segment_start_carry.y).shape[0]
        project_flat = physics_context.project_flat
        flat_rhs = physics_context.flat_rhs
        flat_rhs_with_lagged_response = physics_context.flat_rhs_with_lagged_response
        build_lagged_response, _ = _lagged_response_hooks(vector_field)
        rhs_mode = str(getattr(solver, "rhs_mode", "black_box")).strip().lower()
        use_lagged_linear_response = rhs_mode == "lagged_linear_state"
        use_transport_lagged_response = rhs_mode in {"lagged_transport_response", "lagged_response"}
        predictor_mode = getattr(solver, "predictor_mode", "linearized")
        theta = jnp.asarray(solver.theta_implicit, dtype=dtype)
        identity_n = jnp.eye(state_dim, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        t_final = jnp.asarray(solver.t1, dtype=dtype)

        def _make_theta_replay_step_fn(
            flat_rhs_replay,
            flat_rhs_with_lagged_response_replay,
            build_lagged_response_replay=build_lagged_response,
        ):
            if isinstance(solver, ThetaMethodSolver):
                n_linearized_solves = 1 + (solver.n_corrector_steps if solver.use_predictor_corrector else 0)

                def _attempt_fn(attempt_context):
                    return _theta_basic_accepted_step_attempt(
                        attempt_context,
                        predictor_mode=predictor_mode,
                        n_linearized_solves=n_linearized_solves,
                        theta=theta,
                        one=one,
                        identity_n=identity_n,
                        flat_rhs=flat_rhs_replay,
                        flat_rhs_with_lagged_response=flat_rhs_with_lagged_response_replay,
                        use_lagged_linear_response=use_lagged_linear_response,
                        project_flat=project_flat,
                        dtype=dtype,
                        tol=jnp.asarray(solver.tol, dtype=dtype),
                    )

                def _step_fn(step_state):
                    return _theta_basic_step_from_attempt_fn(
                        step_state,
                        attempt_fn=_attempt_fn,
                        t_final=t_final,
                        flat_rhs=flat_rhs_replay,
                        build_lagged_response=build_lagged_response_replay,
                        unpack_flat=unpack_flat,
                        project_flat=project_flat,
                        use_transport_lagged_response=use_transport_lagged_response,
                        dtype=dtype,
                    )

                return _step_fn

            n_linearized_solves = 1 + (solver.n_corrector_steps if solver.use_predictor_corrector else 0)
            lagged_response_reuse_mode = str(getattr(solver, "lagged_response_reuse_mode", "retry_only")).strip().lower()
            lagged_response_reuse_rtol = jnp.asarray(getattr(solver, "lagged_response_reuse_rtol", 5.0e-2), dtype=dtype)
            lagged_response_reuse_atol = jnp.asarray(getattr(solver, "lagged_response_reuse_atol", 1.0e-8), dtype=dtype)
            delta_reduction_factor = jnp.asarray(solver.delta_reduction_factor, dtype=dtype)
            tau_min = jnp.asarray(solver.tau_min, dtype=dtype)
            jacobian_reuse_rtol = jnp.asarray(getattr(solver, "jacobian_reuse_rtol", 0.1), dtype=dtype)
            max_jacobian_age = jnp.asarray(getattr(solver, "max_jacobian_age", 8), dtype=jnp.int32)
            freeze_attempt_linearization = str(
                getattr(solver, "jacobian_reuse_mode", "refresh_each_iteration")
            ).strip().lower() == "freeze_attempt"

            def _attempt_fn(attempt_context):
                return _theta_newton_accepted_step_attempt(
                    attempt_context,
                    predictor_mode=predictor_mode,
                    n_linearized_solves=n_linearized_solves,
                    theta=theta,
                    one=one,
                    identity_n=identity_n,
                    flat_rhs=flat_rhs_replay,
                    flat_rhs_with_lagged_response=flat_rhs_with_lagged_response_replay,
                    use_lagged_linear_response=use_lagged_linear_response,
                    use_transport_lagged_response=use_transport_lagged_response,
                    lagged_response_reuse_mode=lagged_response_reuse_mode,
                    jacobian_reuse_rtol=jacobian_reuse_rtol,
                    max_jacobian_age=max_jacobian_age,
                    delta_reduction_factor=delta_reduction_factor,
                    tau_min=tau_min,
                    project_flat=project_flat,
                    dtype=dtype,
                    tol=jnp.asarray(solver.tol, dtype=dtype),
                    maxiter=jnp.asarray(solver.maxiter, dtype=jnp.int32),
                )

            def _step_fn(step_state):
                return _theta_newton_step_from_attempt_fn(
                    step_state,
                    attempt_fn=_attempt_fn,
                    t_final=t_final,
                    flat_rhs=flat_rhs_replay,
                    build_lagged_response=build_lagged_response_replay,
                    unpack_flat=unpack_flat,
                    project_flat=project_flat,
                    use_transport_lagged_response=use_transport_lagged_response,
                    lagged_response_reuse_mode=lagged_response_reuse_mode,
                    lagged_response_reuse_rtol=lagged_response_reuse_rtol,
                    lagged_response_reuse_atol=lagged_response_reuse_atol,
                    dt_min=jnp.asarray(solver.min_step, dtype=dtype),
                    dt_max=jnp.asarray(solver.max_step, dtype=dtype),
                    safety_factor=jnp.asarray(solver.safety_factor, dtype=dtype),
                    min_step_factor=jnp.asarray(solver.min_step_factor, dtype=dtype),
                    max_step_factor=jnp.asarray(solver.max_step_factor, dtype=dtype),
                    controller_mode=str(getattr(solver, "controller_mode", "current")).strip().lower(),
                    delta_reduction_factor=delta_reduction_factor,
                    freeze_attempt_linearization=freeze_attempt_linearization,
                    dtype=dtype,
                )

            return _step_fn

        _step_fn = _make_theta_replay_step_fn(flat_rhs, flat_rhs_with_lagged_response)

        (
            active_mask,
            attempted_dts,
            next_dts,
            next_recent_reject_count,
            next_regrowth_cooldown,
            next_easy_growth_streak,
            next_lagged_response_valid,
        ) = segment_arrays

        if mode == "theta_implicit_transpose_probe":
            if not use_transport_lagged_response:
                raise NotImplementedError(
                    "theta_implicit_transpose_probe currently targets lagged transport-response theta runs."
                )

            def _initial_theta_step_state_from_carry(carry_value):
                reuse_state = dataclasses.replace(
                    _theta_initial_reuse_state(state_dim, dtype),
                    lagged_response_cache=carry_value.lagged_response_cache,
                    lagged_response_available=jnp.asarray(carry_value.lagged_response_cache is not None),
                    lagged_response_valid=carry_value.lagged_response_valid,
                    lagged_reference_y=carry_value.lagged_reference_y,
                )
                return _ThetaStepState(
                    t=carry_value.t,
                    y=carry_value.y,
                    dt=attempted_dts[0],
                    status=jnp.asarray([0, 0, 0], dtype=jnp.int32),
                    prev_error=jnp.asarray(1.0, dtype=dtype),
                    prev_dt=jnp.asarray(0.0, dtype=dtype),
                    recent_reject_count=jnp.asarray(0, dtype=jnp.int32),
                    regrowth_cooldown=jnp.asarray(0, dtype=jnp.int32),
                    easy_growth_streak=jnp.asarray(0, dtype=jnp.int32),
                    prev_theta_final=jnp.asarray(0.0, dtype=dtype),
                    prev_newton_iter_count=jnp.asarray(0, dtype=jnp.int32),
                    reuse_state=reuse_state,
                )

            def _collect_start_states(carry, slot_values):
                active, dt_value, next_dt_value, recent_reject, cooldown, streak, lagged_valid = slot_values
                carry = dataclasses.replace(carry, dt=dt_value)

                def _run(_):
                    next_state, _info = _step_fn(carry)
                    next_reuse = dataclasses.replace(
                        next_state.reuse_state,
                        lagged_response_valid=lagged_valid,
                    )
                    return dataclasses.replace(
                        next_state,
                        dt=next_dt_value,
                        recent_reject_count=recent_reject,
                        regrowth_cooldown=cooldown,
                        easy_growth_streak=streak,
                        reuse_state=next_reuse,
                    )

                return jax.lax.cond(active, _run, lambda _: carry, operand=None), carry

            _segment_final_state, step_start_states = jax.lax.scan(
                _collect_start_states,
                _initial_theta_step_state_from_carry(segment_start_carry),
                segment_arrays,
            )

            def _rhs_state_pullback(t_value, y_value, lagged_response_value, rhs_bar):
                if lagged_response_value is not None and physics_context.flat_rhs_state_pullback is not None:
                    return physics_context.flat_rhs_state_pullback(
                        t_value,
                        y_value,
                        lagged_response_value,
                        rhs_bar,
                    )

                def _rhs_from_y(y_inner):
                    return _radau_eval_rhs(
                        t_value,
                        y_inner,
                        lagged_response_value,
                        flat_rhs,
                        flat_rhs_with_lagged_response,
                    )

                _, rhs_pullback = jax.vjp(_rhs_from_y, y_value)
                (y_bar,) = rhs_pullback(rhs_bar)
                return y_bar

            def _rhs_lagged_pullback(t_value, y_value, lagged_response_value, rhs_bar):
                if lagged_response_value is None:
                    return None
                if physics_context.flat_rhs_lagged_response_pullback is not None:
                    return physics_context.flat_rhs_lagged_response_pullback(
                        t_value,
                        y_value,
                        lagged_response_value,
                        rhs_bar,
                    )

                def _rhs_from_lagged(lagged_inner):
                    return flat_rhs_with_lagged_response(t_value, y_value, lagged_inner)

                _, lagged_pullback = jax.vjp(_rhs_from_lagged, lagged_response_value)
                (lagged_bar,) = lagged_pullback(rhs_bar)
                return lagged_bar

            def _batched_zero_like_tree(tree):
                if tree is None:
                    return None
                return jax.tree_util.tree_map(
                    lambda leaf: jnp.zeros(
                        (objective_count,) + jnp.asarray(leaf).shape,
                        dtype=jnp.asarray(leaf).dtype,
                    ),
                    tree,
                )

            def _single_step_bwd(carry, slot_xs):
                slot_bars, support_bar_leaves = carry
                step_state, slot_values = slot_xs
                (
                    active,
                    dt_value,
                    _next_dt_value,
                    _recent_reject,
                    _cooldown,
                    _streak,
                    _lagged_valid,
                ) = slot_values

                def _zero_step(_):
                    return slot_bars, support_bar_leaves

                def _do_step(_):
                    carry_for_step = dataclasses.replace(step_state, dt=dt_value)
                    next_state, _info = _step_fn(carry_for_step)
                    h_value = jnp.asarray(dt_value, dtype=dtype)
                    t_old = carry_for_step.t
                    t_new = t_old + h_value
                    y_old = carry_for_step.y
                    y_new = next_state.y

                    if isinstance(solver, ThetaMethodSolver):
                        attempt_context = _theta_make_attempt_context(
                            carry_for_step,
                            t_final=t_final,
                            flat_rhs=flat_rhs,
                            build_lagged_response=build_lagged_response,
                            unpack_flat=unpack_flat,
                            project_flat=project_flat,
                            use_transport_lagged_response=use_transport_lagged_response,
                        )
                        lagged_response = attempt_context.lagged_response
                        lagged_response_reused = jnp.asarray(False)
                        lagged_reference_y = attempt_context.lagged_reference_y
                    else:
                        lagged_response, lagged_reference_y, lagged_response_reused = _theta_prepare_lagged_response(
                            carry_for_step,
                            use_transport_lagged_response=use_transport_lagged_response,
                            lagged_response_reuse_mode=lagged_response_reuse_mode,
                            lagged_response_reuse_rtol=lagged_response_reuse_rtol,
                            lagged_response_reuse_atol=lagged_response_reuse_atol,
                            unpack_flat=unpack_flat,
                            project_flat=project_flat,
                            build_lagged_response=build_lagged_response,
                        )

                    def _rhs_new(y_value):
                        return _radau_eval_rhs(
                            t_new,
                            y_value,
                            lagged_response,
                            flat_rhs,
                            flat_rhs_with_lagged_response,
                        )

                    jac_new = jax.jacfwd(_rhs_new)(y_new)
                    system_t = (jnp.eye(state_dim, dtype=dtype) - h_value * theta * jac_new).T
                    lambda_rows = jax.vmap(
                        lambda rhs: jnp.linalg.solve(system_t, jnp.asarray(rhs, dtype=dtype))
                    )(slot_bars.y)

                    f_old_coeff = h_value * (one - theta)
                    f_new_coeff = h_value * theta
                    old_rhs_bars = f_old_coeff * lambda_rows
                    new_rhs_bars = f_new_coeff * lambda_rows

                    old_rhs_y_bars = jax.vmap(
                        lambda rhs_bar: _rhs_state_pullback(t_old, y_old, None, rhs_bar)
                    )(old_rhs_bars)
                    y_old_bars = lambda_rows + old_rhs_y_bars

                    lagged_rhs_bars = jax.vmap(
                        lambda rhs_bar: _rhs_lagged_pullback(t_new, y_new, lagged_response, rhs_bar)
                    )(new_rhs_bars)
                    support_rhs_bars = jax.vmap(
                        lambda rhs_bar: _theta_lagged_rhs_support_pullback(
                            physics_context,
                            t_value=t_new,
                            y_value=y_new,
                            lagged_response=lagged_response,
                            rhs_bar=rhs_bar,
                            support_payload=support_payload,
                        )
                    )(new_rhs_bars)
                    support_bar_leaves_next = tuple(
                        accumulated + increment
                        for accumulated, increment in zip(
                            support_bar_leaves,
                            jax.tree_util.tree_leaves(support_rhs_bars),
                            strict=True,
                        )
                    )

                    total_lagged_bars = _radau_align_tangent_tree_to_primal(
                        slot_bars.lagged_response_cache,
                        lagged_response,
                    )
                    total_lagged_bars = jax.tree_util.tree_map(
                        lambda lhs, rhs: lhs + rhs,
                        total_lagged_bars,
                        lagged_rhs_bars,
                    )
                    reference_bars = jnp.asarray(slot_bars.lagged_reference_y, dtype=dtype)

                    def _reuse_lagged(_):
                        return (
                            y_old_bars,
                            _radau_align_tangent_tree_to_primal(
                                total_lagged_bars,
                                carry_for_step.reuse_state.lagged_response_cache,
                            ),
                            reference_bars,
                            support_bar_leaves_next,
                        )

                    def _rebuild_lagged(_):
                        if physics_context.pullback_build_lagged_response is not None:
                            projected_y = _project_flat_state_if_needed(y_old, project_flat)
                            rebuild_state = unpack_flat(projected_y)
                            rebuild_state_bars = jax.vmap(
                                lambda lagged_bar: physics_context.pullback_build_lagged_response(
                                    rebuild_state,
                                    lagged_bar,
                                )
                            )(total_lagged_bars)
                            rebuild_flat_bars = jax.vmap(physics_context.pack_flat)(rebuild_state_bars)
                            if project_flat is not None:
                                _, project_pullback = jax.vjp(project_flat, y_old)
                                rebuild_flat_bars = jax.vmap(lambda bar: project_pullback(bar)[0])(rebuild_flat_bars)
                        else:
                            def _build_from_flat(flat_inner):
                                projected_inner = _project_flat_state_if_needed(flat_inner, project_flat)
                                return build_lagged_response(unpack_flat(projected_inner))

                            _, build_pullback = jax.vjp(_build_from_flat, y_old)
                            rebuild_flat_bars = jax.vmap(lambda lagged_bar: build_pullback(lagged_bar)[0])(
                                total_lagged_bars
                            )
                        y_old_bars_rebuild = y_old_bars + rebuild_flat_bars + reference_bars
                        if physics_context.flat_rhs_build_support_pullback is not None:
                            rebuild_support_bars = jax.vmap(
                                lambda lagged_bar: _radau_sanitize_support_delta_bar_tree(
                                    support_payload,
                                    physics_context.flat_rhs_build_support_pullback(
                                        y_old,
                                        lagged_bar,
                                        support_payload,
                                    ),
                                )
                            )(total_lagged_bars)
                            support_bar_leaves_rebuild = tuple(
                                accumulated + increment
                                for accumulated, increment in zip(
                                    support_bar_leaves_next,
                                    jax.tree_util.tree_leaves(rebuild_support_bars),
                                    strict=True,
                                )
                            )
                        else:
                            support_bar_leaves_rebuild = support_bar_leaves_next
                        return (
                            y_old_bars_rebuild,
                            _batched_zero_like_tree(carry_for_step.reuse_state.lagged_response_cache),
                            jnp.zeros_like(reference_bars),
                            support_bar_leaves_rebuild,
                        )

                    y_prev, lagged_prev, reference_prev, support_bar_leaves_out = jax.lax.cond(
                        lagged_response_reused,
                        _reuse_lagged,
                        _rebuild_lagged,
                        operand=None,
                    )
                    return (
                        _ThetaAcceptedStepReducedCotangent(
                            y=y_prev,
                            lagged_response_cache=lagged_prev,
                            lagged_reference_y=reference_prev,
                        ),
                        support_bar_leaves_out,
                    )

                return jax.lax.cond(active, _do_step, _zero_step, operand=None)

            start_bars, support_bar_leaves = jax.lax.scan(
                _single_step_bwd,
                (reduced_bars, zero_support_bar_leaves),
                (step_start_states, segment_arrays),
                reverse=True,
            )[0]
            return start_bars, support_bar_leaves

        @jax.custom_vjp
        def _compact_lagged_rhs_with_support(t_value, y_value, lagged_response_value, support_value):
            return flat_rhs_with_lagged_response(t_value, y_value, lagged_response_value)

        def _compact_lagged_rhs_with_support_fwd(t_value, y_value, lagged_response_value, support_value):
            rhs_value = flat_rhs_with_lagged_response(t_value, y_value, lagged_response_value)
            return rhs_value, (t_value, y_value, lagged_response_value, support_value)

        def _compact_lagged_rhs_with_support_bwd(residual, rhs_bar):
            t_value, y_value, lagged_response_value, support_value = residual
            rhs_bar = jnp.asarray(rhs_bar, dtype=jnp.asarray(y_value).dtype)
            if physics_context.flat_rhs_state_pullback is not None:
                y_bar = physics_context.flat_rhs_state_pullback(
                    t_value,
                    y_value,
                    lagged_response_value,
                    rhs_bar,
                )
            else:
                def _rhs_from_y(y_inner):
                    return flat_rhs_with_lagged_response(
                        t_value,
                        y_inner,
                        lagged_response_value,
                    )

                _, y_pullback = jax.vjp(_rhs_from_y, y_value)
                (y_bar,) = y_pullback(rhs_bar)
            if physics_context.flat_rhs_lagged_response_pullback is not None:
                lagged_bar = physics_context.flat_rhs_lagged_response_pullback(
                    t_value,
                    y_value,
                    lagged_response_value,
                    rhs_bar,
                )
            else:
                def _rhs_from_lagged(lagged_inner):
                    return flat_rhs_with_lagged_response(
                        t_value,
                        y_value,
                        lagged_inner,
                    )

                _, lagged_pullback = jax.vjp(_rhs_from_lagged, lagged_response_value)
                (lagged_bar,) = lagged_pullback(rhs_bar)
            support_bar = _theta_lagged_rhs_support_pullback(
                physics_context,
                t_value=t_value,
                y_value=y_value,
                lagged_response=lagged_response_value,
                rhs_bar=rhs_bar,
                support_payload=support_value,
            )
            support_bar = _theta_support_cotangent_for_vjp(support_value, support_bar)
            return (
                jnp.zeros_like(t_value),
                y_bar,
                lagged_bar,
                support_bar,
            )

        _compact_lagged_rhs_with_support.defvjp(
            _compact_lagged_rhs_with_support_fwd,
            _compact_lagged_rhs_with_support_bwd,
        )

        def _build_lagged_response_from_flat(flat_y_value):
            if build_lagged_response is None:
                return None
            projected_y = _project_flat_state_if_needed(flat_y_value, project_flat)
            return build_lagged_response(unpack_flat(projected_y))

        @jax.custom_vjp
        def _compact_build_lagged_response_with_support(flat_y_value, support_value):
            return _build_lagged_response_from_flat(flat_y_value)

        def _compact_build_lagged_response_with_support_fwd(flat_y_value, support_value):
            lagged_value = _build_lagged_response_from_flat(flat_y_value)
            return lagged_value, (flat_y_value, support_value, lagged_value)

        def _compact_build_lagged_response_with_support_bwd(residual, lagged_bar):
            flat_y_value, support_value, lagged_value = residual
            if lagged_value is None:
                return (
                    jnp.zeros_like(flat_y_value),
                    _theta_support_cotangent_for_vjp(
                        support_value,
                        _radau_zero_support_delta_tree_like(support_value),
                    ),
                )
            projected_y = _project_flat_state_if_needed(flat_y_value, project_flat)
            state_value = unpack_flat(projected_y)
            if physics_context.pullback_build_lagged_response is not None:
                state_bar = physics_context.pullback_build_lagged_response(
                    state_value,
                    lagged_bar,
                )
                flat_y_bar = physics_context.pack_flat(state_bar)
                if project_flat is not None:
                    _, project_pullback = jax.vjp(project_flat, flat_y_value)
                    (flat_y_bar,) = project_pullback(flat_y_bar)
            else:
                def _build_from_flat(flat_inner):
                    projected_inner = _project_flat_state_if_needed(flat_inner, project_flat)
                    return build_lagged_response(unpack_flat(projected_inner))

                _, build_pullback = jax.vjp(_build_from_flat, flat_y_value)
                (flat_y_bar,) = build_pullback(lagged_bar)
            if physics_context.flat_rhs_build_support_pullback is not None:
                support_bar = physics_context.flat_rhs_build_support_pullback(
                    flat_y_value,
                    lagged_bar,
                    support_value,
                )
            else:
                support_bar = _radau_zero_support_delta_tree_like(support_value)
            support_bar = _theta_support_cotangent_for_vjp(support_value, support_bar)
            return flat_y_bar, support_bar

        _compact_build_lagged_response_with_support.defvjp(
            _compact_build_lagged_response_with_support_fwd,
            _compact_build_lagged_response_with_support_bwd,
        )

        def _make_final_reduced_from_start(step_fn_for_replay):
            def _final_reduced_from_start(
                start_y,
                start_lagged_cache,
                start_lagged_reference_y,
            ):
                reuse_state = dataclasses.replace(
                    _theta_initial_reuse_state(state_dim, dtype),
                    lagged_response_cache=start_lagged_cache,
                    lagged_response_available=jnp.asarray(start_lagged_cache is not None),
                    lagged_response_valid=segment_start_carry.lagged_response_valid,
                    lagged_reference_y=start_lagged_reference_y,
                )
                step_state0 = _ThetaStepState(
                    t=segment_start_carry.t,
                    y=start_y,
                    dt=attempted_dts[0],
                    status=jnp.asarray([0, 0, 0], dtype=jnp.int32),
                    prev_error=jnp.asarray(1.0, dtype=dtype),
                    prev_dt=jnp.asarray(0.0, dtype=dtype),
                    recent_reject_count=jnp.asarray(0, dtype=jnp.int32),
                    regrowth_cooldown=jnp.asarray(0, dtype=jnp.int32),
                    easy_growth_streak=jnp.asarray(0, dtype=jnp.int32),
                    prev_theta_final=jnp.asarray(0.0, dtype=dtype),
                    prev_newton_iter_count=jnp.asarray(0, dtype=jnp.int32),
                    reuse_state=reuse_state,
                )

                def _slot(carry, slot_values):
                    active, dt_value, next_dt_value, recent_reject, cooldown, streak, lagged_valid = slot_values
                    carry = dataclasses.replace(carry, dt=dt_value)

                    def _run(_):
                        next_state, _info = step_fn_for_replay(carry)
                        next_reuse = dataclasses.replace(
                            next_state.reuse_state,
                            lagged_response_valid=lagged_valid,
                        )
                        return dataclasses.replace(
                            next_state,
                            dt=next_dt_value,
                            recent_reject_count=recent_reject,
                            regrowth_cooldown=cooldown,
                            easy_growth_streak=streak,
                            reuse_state=next_reuse,
                        )

                    return jax.lax.cond(active, _run, lambda _: carry, operand=None), None

                final_state, _ = jax.lax.scan(
                    _slot,
                    step_state0,
                    (
                        active_mask,
                        attempted_dts,
                        next_dts,
                        next_recent_reject_count,
                        next_regrowth_cooldown,
                        next_easy_growth_streak,
                        next_lagged_response_valid,
                    ),
                )
                return _ThetaAcceptedStepReducedCotangent(
                    y=final_state.y,
                    lagged_response_cache=final_state.reuse_state.lagged_response_cache,
                    lagged_reference_y=final_state.reuse_state.lagged_reference_y,
                )

            return _final_reduced_from_start

        _final_reduced_from_start = _make_final_reduced_from_start(_step_fn)

        def _take_batched_tree_axis0(tree, index):
            if tree is None:
                return None
            return jax.tree_util.tree_map(lambda value: value[index], tree)

        def _zero_tree_like(tree):
            if tree is None:
                return None
            return jax.tree_util.tree_map(jnp.zeros_like, tree)

        def _single_reduced_pullback(objective_index):
            reduced_lagged_bar_i = _take_batched_tree_axis0(
                reduced_bars.lagged_response_cache,
                objective_index,
            )
            if mode == "theta_zero_lagged":
                reduced_lagged_bar_i = _zero_tree_like(reduced_lagged_bar_i)
            reduced_bar_i = _ThetaAcceptedStepReducedCotangent(
                y=reduced_bars.y[objective_index],
                lagged_response_cache=reduced_lagged_bar_i,
                lagged_reference_y=reduced_bars.lagged_reference_y[objective_index],
            )

            if segment_start_carry.lagged_response_cache is None:
                def _final_reduced_from_start_no_cache(start_y, start_lagged_reference_y):
                    return _final_reduced_from_start(
                        start_y,
                        None,
                        start_lagged_reference_y,
                    )

                _, pullback = jax.vjp(
                    _final_reduced_from_start_no_cache,
                    segment_start_carry.y,
                    segment_start_carry.lagged_reference_y,
                )
                start_y_bar_i, start_reference_bar_i = pullback(reduced_bar_i)
                start_cache_bar_i = None
            else:
                _, pullback = jax.vjp(
                    _final_reduced_from_start,
                    segment_start_carry.y,
                    segment_start_carry.lagged_response_cache,
                    segment_start_carry.lagged_reference_y,
                )
                start_y_bar_i, start_cache_bar_i, start_reference_bar_i = pullback(reduced_bar_i)
            return _ThetaAcceptedStepReducedCotangent(
                y=start_y_bar_i,
                lagged_response_cache=start_cache_bar_i,
                lagged_reference_y=start_reference_bar_i,
            )

        objective_indices = jnp.arange(objective_count, dtype=jnp.int32)
        if mode == "theta_compact_support_probe":
            def _single_support_pullback(objective_index):
                reduced_bar_i = _ThetaAcceptedStepReducedCotangent(
                    y=reduced_bars.y[objective_index],
                    lagged_response_cache=_take_batched_tree_axis0(
                        reduced_bars.lagged_response_cache,
                        objective_index,
                    ),
                    lagged_reference_y=reduced_bars.lagged_reference_y[objective_index],
                )

                def _make_support_step_fn(support_value):
                    def _flat_rhs_with_lagged_response_support(t_value, y_value, lagged_response_value):
                        return _compact_lagged_rhs_with_support(
                            t_value,
                            y_value,
                            lagged_response_value,
                            support_value,
                        )

                    def _build_lagged_response_support(state_value):
                        return _compact_build_lagged_response_with_support(
                            physics_context.pack_flat(state_value),
                            support_value,
                        )

                    return _make_theta_replay_step_fn(
                        flat_rhs,
                        _flat_rhs_with_lagged_response_support,
                        build_lagged_response_replay=_build_lagged_response_support,
                    )

                if segment_start_carry.lagged_response_cache is None:
                    def _final_reduced_from_start_no_cache_support(
                        start_y,
                        start_lagged_reference_y,
                        support_value,
                    ):
                        support_final_reduced_from_start = _make_final_reduced_from_start(
                            _make_support_step_fn(support_value)
                        )
                        return support_final_reduced_from_start(
                            start_y,
                            None,
                            start_lagged_reference_y,
                        )

                    _, pullback = jax.vjp(
                        _final_reduced_from_start_no_cache_support,
                        segment_start_carry.y,
                        segment_start_carry.lagged_reference_y,
                        support_payload,
                    )
                    start_y_bar_i, start_reference_bar_i, support_bar_i = pullback(reduced_bar_i)
                    start_cache_bar_i = None
                else:
                    def _final_reduced_from_start_support(
                        start_y,
                        start_lagged_cache,
                        start_lagged_reference_y,
                        support_value,
                    ):
                        support_final_reduced_from_start = _make_final_reduced_from_start(
                            _make_support_step_fn(support_value)
                        )
                        return support_final_reduced_from_start(
                            start_y,
                            start_lagged_cache,
                            start_lagged_reference_y,
                        )

                    _, pullback = jax.vjp(
                        _final_reduced_from_start_support,
                        segment_start_carry.y,
                        segment_start_carry.lagged_response_cache,
                        segment_start_carry.lagged_reference_y,
                        support_payload,
                    )
                    start_y_bar_i, start_cache_bar_i, start_reference_bar_i, support_bar_i = pullback(reduced_bar_i)
                return (
                    _ThetaAcceptedStepReducedCotangent(
                        y=start_y_bar_i,
                        lagged_response_cache=start_cache_bar_i,
                        lagged_reference_y=start_reference_bar_i,
                    ),
                    _radau_sanitize_support_delta_bar_tree(support_payload, support_bar_i),
                )

            start_bars, support_bars = jax.vmap(_single_support_pullback)(objective_indices)
            return start_bars, tuple(jax.tree_util.tree_leaves(support_bars))

        start_bars = jax.vmap(_single_reduced_pullback)(objective_indices)
        return start_bars, zero_support_bar_leaves

    raise NotImplementedError(
        "theta full-transport reverse has real schedule/segment forward artifacts, "
        "but this cotangent diagnostic mode is not implemented for theta. "
        "Use reverse_stage_cotangent_mode='full' for the solver-selected theta "
        "implicit-transpose lane, or one of zero_step_bwd/theta_state_only/"
        "theta_compact_support_probe/theta_implicit_transpose_probe for diagnostics."
    )


def _reverse_segment_reduced_cotangent_bwd_batched_with_support_call(
    execution_context,
    cotangent_mode,
    reduced_bars,
    segment_start_carry,
    segment_arrays,
    support_payload,
):
    if isinstance(execution_context, _ThetaReverseExecutionContext):
        return _theta_segment_reduced_cotangent_bwd_batched_with_support_call(
            execution_context,
            cotangent_mode,
            reduced_bars,
            segment_start_carry,
            segment_arrays,
            support_payload,
        )
    _require_reverse_execution_context_radau(
        execution_context,
        "reverse segmented support-payload cotangent sweep",
    )
    return _radau_segment_reduced_cotangent_bwd_batched_with_support_call(
        execution_context,
        cotangent_mode,
        reduced_bars,
        segment_start_carry,
        segment_arrays,
        support_payload,
    )


def _reverse_reduced_cotangent(
    execution_context,
    *,
    y,
    lagged_response_cache,
    lagged_reference_y,
):
    if isinstance(execution_context, _ThetaReverseExecutionContext):
        return _ThetaAcceptedStepReducedCotangent(
            y=y,
            lagged_response_cache=lagged_response_cache,
            lagged_reference_y=lagged_reference_y,
        )
    _require_reverse_execution_context_radau(execution_context, "reverse reduced cotangent")
    return _RadauAcceptedStepReducedCotangent(
        y=y,
        lagged_response_cache=lagged_response_cache,
        lagged_reference_y=lagged_reference_y,
    )


def _theta_initial_carry_from_state_with_static_setup(
    *,
    solver,
    state,
    solve_vector_field,
    species,
    prepared_rollout_static,
):
    temperature_active_mask, fixed_temperature_profile = _extract_fixed_temperature_projection(solve_vector_field)
    density_floor, temperature_floor = _extract_state_regularization(solve_vector_field)

    def _flat_state_from_state(state_value):
        flat_state, unpack_flat, _unpack_packed, _pack_state, project_flat = _make_solver_state_transform(
            state_value,
            species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
            density_floor=density_floor,
            temperature_floor=temperature_floor,
        )
        return flat_state, unpack_flat, project_flat

    flat_state0, unpack_flat, project_flat = _flat_state_from_state(state)
    dtype = jnp.asarray(flat_state0).dtype
    build_lagged_response, _ = _lagged_response_hooks(solve_vector_field)
    rhs_mode = str(getattr(solver, "rhs_mode", "black_box")).strip().lower()
    use_transport_lagged_response = rhs_mode in {"lagged_transport_response", "lagged_response"}
    if use_transport_lagged_response and build_lagged_response is not None:
        lagged_reference_y = _project_flat_state_if_needed(flat_state0, project_flat)
        lagged_response_cache = build_lagged_response(unpack_flat(lagged_reference_y))
        lagged_response_valid = jnp.asarray(True)
    else:
        lagged_reference_y = flat_state0
        lagged_response_cache = None
        lagged_response_valid = jnp.asarray(False)
    return _ThetaReverseCarry(
        t=jnp.asarray(getattr(solver, "t0", 0.0), dtype=dtype),
        y=flat_state0,
        lagged_response_cache=lagged_response_cache,
        lagged_response_valid=lagged_response_valid,
        lagged_reference_y=lagged_reference_y,
    )


def reverse_initial_carry_from_state_with_static_setup(
    *,
    solver,
    state,
    solve_vector_field,
    species,
    prepared_rollout_static,
    return_native_joint_pullback: bool = False,
    return_native_split_joint_pullback: bool = False,
    return_native_split_joint_no_prepared_carry: bool = False,
    return_native_split_joint_fused_rhs_pullback: bool = False,
):
    """Build the initial carry with the validated reverse-local lagged pullback."""

    if _is_theta_reverse_solver(solver):
        return _theta_initial_carry_from_state_with_static_setup(
            solver=solver,
            state=state,
            solve_vector_field=solve_vector_field,
            species=species,
            prepared_rollout_static=prepared_rollout_static,
        )

    _require_reverse_solver_radau(solver, "reverse initial carry")
    temperature_active_mask, fixed_temperature_profile = _extract_fixed_temperature_projection(solve_vector_field)
    density_floor, temperature_floor = _extract_state_regularization(solve_vector_field)
    kernel_context = prepared_rollout_static.kernel_context
    physics_context = prepared_rollout_static.physics_context
    initial_carry_static = prepared_rollout_static.initial_carry
    lagged_pullback_fn = lagged_response_pullback_from_owner(solve_vector_field)

    def _flat_state_from_state(state_value):
        flat_state, *_ = _make_solver_state_transform(
            state_value,
            species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
            density_floor=density_floor,
            temperature_floor=temperature_floor,
        )
        return flat_state

    def _build_state_from_flat(flat_value, unpack_flat, project_flat):
        return unpack_flat(_project_flat_state_if_needed(flat_value, project_flat))

    @jax.custom_vjp
    def _build_initial_carry(state_value):
        flat_state0, unpack_flat, _unpack_packed, _pack_state, project_flat = _make_solver_state_transform(
            state_value,
            species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
            density_floor=density_floor,
            temperature_floor=temperature_floor,
        )
        lagged_state0 = _build_state_from_flat(flat_state0, unpack_flat, project_flat)
        initial_lagged_response = (
            physics_context.build_lagged_response(lagged_state0)
            if (kernel_context.use_transport_lagged_response and physics_context.build_lagged_response is not None)
            else None
        )
        initial_rhs = _radau_eval_rhs(
            initial_carry_static.t,
            flat_state0,
            initial_lagged_response,
            physics_context.flat_rhs,
            physics_context.flat_rhs_with_lagged_response,
        )
        step_state0 = _make_radau_initial_step_state(
            initial_carry_static.t,
            flat_state0,
            initial_carry_static.dt,
            kernel_context.dtype,
            initial_rhs,
            kernel_context.num_stages,
            initial_carry_static.real_lu,
            initial_carry_static.real_piv,
            initial_carry_static.complex_lu,
            initial_carry_static.complex_piv,
            initial_lagged_response,
            jnp.asarray(kernel_context.use_transport_lagged_response),
            flat_state0,
        )
        return _radau_carry_from_step_state(step_state0)

    def _build_initial_carry_fwd(state_value):
        flat_state0, unpack_flat, _unpack_packed, _pack_state, project_flat = _make_solver_state_transform(
            state_value,
            species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
            density_floor=density_floor,
            temperature_floor=temperature_floor,
        )
        lagged_state0 = _build_state_from_flat(flat_state0, unpack_flat, project_flat)
        initial_lagged_response = (
            physics_context.build_lagged_response(lagged_state0)
            if (kernel_context.use_transport_lagged_response and physics_context.build_lagged_response is not None)
            else None
        )
        initial_rhs = _radau_eval_rhs(
            initial_carry_static.t,
            flat_state0,
            initial_lagged_response,
            physics_context.flat_rhs,
            physics_context.flat_rhs_with_lagged_response,
        )
        step_state0 = _make_radau_initial_step_state(
            initial_carry_static.t,
            flat_state0,
            initial_carry_static.dt,
            kernel_context.dtype,
            initial_rhs,
            kernel_context.num_stages,
            initial_carry_static.real_lu,
            initial_carry_static.real_piv,
            initial_carry_static.complex_lu,
            initial_carry_static.complex_piv,
            initial_lagged_response,
            jnp.asarray(kernel_context.use_transport_lagged_response),
            flat_state0,
        )
        carry0 = _radau_carry_from_step_state(step_state0)
        residual = (state_value, flat_state0, lagged_state0, initial_lagged_response)
        return carry0, residual

    def _build_initial_carry_bwd(residual, carry_bar):
        state_value, flat_state0, lagged_state0, initial_lagged_response = residual
        _, unpack_flat, _unpack_packed, _pack_state, project_flat = _make_solver_state_transform(
            state_value,
            species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
            density_floor=density_floor,
            temperature_floor=temperature_floor,
        )
        flat_bar = jnp.asarray(carry_bar.y)
        flat_bar = flat_bar + jnp.asarray(carry_bar.lagged_reference_y)

        prev_stages_bar = jnp.asarray(carry_bar.prev_stages).reshape((kernel_context.num_stages, -1))
        rhs_bar = jnp.sum(prev_stages_bar, axis=0)
        lagged_bar = carry_bar.lagged_response_cache

        def _tree_max_abs(tree):
            values = []
            for leaf in jax.tree_util.tree_leaves(tree):
                arr = jnp.asarray(leaf)
                if arr.dtype == jax.dtypes.float0:
                    continue
                if jnp.issubdtype(arr.dtype, jnp.number):
                    values.append(jnp.max(jnp.abs(arr)))
            if not values:
                return jnp.asarray(0.0, dtype=flat_state0.dtype)
            return jnp.max(jnp.stack([jnp.asarray(value, dtype=flat_state0.dtype) for value in values]))

        def _zero_flat_bar():
            return jnp.zeros_like(flat_state0)

        def _rhs_state_pullback_fallback(lagged_response_value):
            def _rhs_from_flat(flat_value):
                return _radau_eval_rhs(
                    initial_carry_static.t,
                    flat_value,
                    lagged_response_value,
                    physics_context.flat_rhs,
                    physics_context.flat_rhs_with_lagged_response,
                )

            _, rhs_pullback = jax.vjp(_rhs_from_flat, flat_state0)
            (rhs_flat_bar,) = rhs_pullback(rhs_bar)
            return rhs_flat_bar

        def _nonzero_rhs_state_pullback(_):
            if physics_context.flat_rhs_state_pullback is not None:
                rhs_flat_bar_value = physics_context.flat_rhs_state_pullback(
                    initial_carry_static.t,
                    flat_state0,
                    initial_lagged_response,
                    rhs_bar,
                )
                if project_flat is not None:
                    _, project_pullback = jax.vjp(project_flat, flat_state0)
                    (rhs_flat_bar_value,) = project_pullback(rhs_flat_bar_value)
                return rhs_flat_bar_value
            return _rhs_state_pullback_fallback(initial_lagged_response)

        rhs_flat_bar = jax.lax.cond(
            _tree_max_abs(rhs_bar) > 0.0,
            _nonzero_rhs_state_pullback,
            lambda _: _zero_flat_bar(),
            operand=None,
        )
        flat_bar = flat_bar + rhs_flat_bar

        if initial_lagged_response is not None:
            def _rhs_from_flat_and_lagged(flat_value, lagged_value):
                return _radau_eval_rhs(
                    initial_carry_static.t,
                    flat_value,
                    lagged_value,
                    physics_context.flat_rhs,
                    physics_context.flat_rhs_with_lagged_response,
                )

            def _zero_lagged_bar():
                return _radau_align_tangent_tree_to_primal(None, initial_lagged_response)

            def _nonzero_rhs_lagged_pullback(_):
                if physics_context.flat_rhs_lagged_response_pullback is not None:
                    return physics_context.flat_rhs_lagged_response_pullback(
                        initial_carry_static.t,
                        flat_state0,
                        initial_lagged_response,
                        rhs_bar,
                    )
                _, rhs_pullback = jax.vjp(_rhs_from_flat_and_lagged, flat_state0, initial_lagged_response)
                _rhs_flat_bar_unused, rhs_lagged_bar_value = rhs_pullback(rhs_bar)
                return rhs_lagged_bar_value

            rhs_lagged_bar = jax.lax.cond(
                _tree_max_abs(rhs_bar) > 0.0,
                _nonzero_rhs_lagged_pullback,
                lambda _: _zero_lagged_bar(),
                operand=None,
            )
            lagged_bar = add_trees(lagged_bar, rhs_lagged_bar)

            if lagged_pullback_fn is not None:
                lagged_state_bar = lagged_pullback_fn(lagged_state0, lagged_bar)
            else:
                def _nonzero_lagged_state_pullback(_):
                    def _build_lagged_from_state(lagged_state_value):
                        return physics_context.build_lagged_response(lagged_state_value)

                    _, lagged_pullback = jax.vjp(_build_lagged_from_state, lagged_state0)
                    (lagged_state_bar_value,) = lagged_pullback(lagged_bar)
                    return lagged_state_bar_value

                lagged_state_bar = jax.lax.cond(
                    _tree_max_abs(lagged_bar) > 0.0,
                    _nonzero_lagged_state_pullback,
                    lambda _: jax.tree_util.tree_map(jnp.zeros_like, lagged_state0),
                    operand=None,
                )

            def _lagged_state_from_flat(flat_value):
                return _build_state_from_flat(flat_value, unpack_flat, project_flat)

            def _nonzero_lagged_state_flat_pullback(_):
                _, lagged_state_flat_pullback = jax.vjp(_lagged_state_from_flat, flat_state0)
                (lagged_flat_bar_value,) = lagged_state_flat_pullback(lagged_state_bar)
                return lagged_flat_bar_value

            lagged_flat_bar = jax.lax.cond(
                _tree_max_abs(lagged_state_bar) > 0.0,
                _nonzero_lagged_state_flat_pullback,
                lambda _: _zero_flat_bar(),
                operand=None,
            )
            flat_bar = flat_bar + lagged_flat_bar

        _, state_pullback = jax.vjp(_flat_state_from_state, state_value)
        (state_bar,) = state_pullback(flat_bar)
        return (state_bar,)

    def _native_joint_batched_pullback(residual, carry_bars, support_payload):
        """Manual initial-carry reverse using one batched state/support hook.

        This private closure is intentionally available only through the
        opt-in return path below.  It mirrors the established custom-VJP
        algebra until the initial lagged-response dependency, then replaces
        that state-only pullback with the native joint state/support hook.
        """

        state_value, flat_state0, lagged_state0, initial_lagged_response = residual
        if initial_lagged_response is None:
            raise NotImplementedError(
                "Native joint initial-carry pullback requires an initial lagged response."
            )
        use_split_joint = bool(
            return_native_split_joint_pullback
            or return_native_split_joint_no_prepared_carry
            or return_native_split_joint_fused_rhs_pullback
        )
        joint_pullback = getattr(
            physics_context,
            (
                "flat_rhs_build_state_and_ntx_support_pullback_batched_interpolated_faces_"
                "native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients"
                "_no_prepared_carry"
                if return_native_split_joint_no_prepared_carry
                else "flat_rhs_build_state_and_ntx_support_pullback_batched_interpolated_faces_"
                "native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients"
                if use_split_joint
                else "flat_rhs_build_state_and_support_pullback_batched_interpolated_faces_"
                "native_multi_rhs_reuse_moment_drds_jvp_shared_primal"
            ),
            None,
        )
        if joint_pullback is None:
            raise NotImplementedError(
                "Native joint initial-carry pullback requires the native flattened "
                "state/support hook."
            )
        _, unpack_flat, _unpack_packed, _pack_state, project_flat = _make_solver_state_transform(
            state_value,
            species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
            density_floor=density_floor,
            temperature_floor=temperature_floor,
        )

        def _tree_max_abs(tree):
            values = []
            for leaf in jax.tree_util.tree_leaves(tree):
                arr = jnp.asarray(leaf)
                if arr.dtype == jax.dtypes.float0:
                    continue
                if jnp.issubdtype(arr.dtype, jnp.number):
                    values.append(jnp.max(jnp.abs(arr)))
            if not values:
                return jnp.asarray(0.0, dtype=flat_state0.dtype)
            return jnp.max(jnp.stack([jnp.asarray(value, dtype=flat_state0.dtype) for value in values]))

        def _one_direct_and_lagged_bar(carry_bar):
            direct_flat_bar = (
                jnp.asarray(carry_bar.y) + jnp.asarray(carry_bar.lagged_reference_y)
            )
            prev_stages_bar = jnp.asarray(carry_bar.prev_stages).reshape(
                (kernel_context.num_stages, -1)
            )
            rhs_bar = jnp.sum(prev_stages_bar, axis=0)

            if return_native_split_joint_fused_rhs_pullback:
                fused_rhs_pullback = getattr(
                    physics_context,
                    "flat_rhs_state_and_lagged_response_pullback",
                    None,
                )
                if fused_rhs_pullback is None:
                    raise NotImplementedError(
                        "Fused native initial pullback requires the fixed-lagged "
                        "RHS state/response hook."
                    )

                def _zero_fused_rhs_bar(_):
                    return (
                        jnp.zeros_like(flat_state0),
                        _radau_align_tangent_tree_to_primal(
                            None, initial_lagged_response
                        ),
                    )

                def _nonzero_fused_rhs_bar(_):
                    return fused_rhs_pullback(
                        initial_carry_static.t,
                        flat_state0,
                        initial_lagged_response,
                        rhs_bar,
                    )

                rhs_state_bar, rhs_lagged_bar = jax.lax.cond(
                    _tree_max_abs(rhs_bar) > 0.0,
                    _nonzero_fused_rhs_bar,
                    _zero_fused_rhs_bar,
                    operand=None,
                )
                if project_flat is not None:
                    _, project_pullback = jax.vjp(project_flat, flat_state0)
                    rhs_state_bar = project_pullback(rhs_state_bar)[0]
                return direct_flat_bar + rhs_state_bar, rhs_lagged_bar

            def _zero_flat_bar():
                return jnp.zeros_like(flat_state0)

            def _rhs_state_pullback_fallback(lagged_response_value):
                def _rhs_from_flat(flat_value):
                    return _radau_eval_rhs(
                        initial_carry_static.t,
                        flat_value,
                        lagged_response_value,
                        physics_context.flat_rhs,
                        physics_context.flat_rhs_with_lagged_response,
                    )

                _, rhs_pullback = jax.vjp(_rhs_from_flat, flat_state0)
                return rhs_pullback(rhs_bar)[0]

            def _nonzero_rhs_state_pullback(_):
                if physics_context.flat_rhs_state_pullback is not None:
                    rhs_flat_bar_value = physics_context.flat_rhs_state_pullback(
                        initial_carry_static.t,
                        flat_state0,
                        initial_lagged_response,
                        rhs_bar,
                    )
                    if project_flat is not None:
                        _, project_pullback = jax.vjp(project_flat, flat_state0)
                        rhs_flat_bar_value = project_pullback(rhs_flat_bar_value)[0]
                    return rhs_flat_bar_value
                return _rhs_state_pullback_fallback(initial_lagged_response)

            direct_flat_bar = direct_flat_bar + jax.lax.cond(
                _tree_max_abs(rhs_bar) > 0.0,
                _nonzero_rhs_state_pullback,
                lambda _: _zero_flat_bar(),
                operand=None,
            )

            def _zero_lagged_bar(_):
                return _radau_align_tangent_tree_to_primal(
                    None, initial_lagged_response
                )

            def _nonzero_rhs_lagged_pullback(_):
                if physics_context.flat_rhs_lagged_response_pullback is not None:
                    return physics_context.flat_rhs_lagged_response_pullback(
                        initial_carry_static.t,
                        flat_state0,
                        initial_lagged_response,
                        rhs_bar,
                    )

                def _rhs_from_flat_and_lagged(flat_value, lagged_value):
                    return _radau_eval_rhs(
                        initial_carry_static.t,
                        flat_value,
                        lagged_value,
                        physics_context.flat_rhs,
                        physics_context.flat_rhs_with_lagged_response,
                    )

                _, rhs_pullback = jax.vjp(
                    _rhs_from_flat_and_lagged,
                    flat_state0,
                    initial_lagged_response,
                )
                return rhs_pullback(rhs_bar)[1]

            rhs_lagged_bar = jax.lax.cond(
                _tree_max_abs(rhs_bar) > 0.0,
                _nonzero_rhs_lagged_pullback,
                _zero_lagged_bar,
                operand=None,
            )
            return direct_flat_bar, rhs_lagged_bar

        direct_flat_bars, rhs_lagged_bars = jax.vmap(
            _one_direct_and_lagged_bar
        )(carry_bars)
        if use_split_joint:
            # Do not call the direct geometry transpose here.  This closure is
            # itself differentiated/traced by the initial carry custom VJP;
            # an ordinary Python call would therefore inline VMEC geometry
            # into the native NTX graph.  Return the total lagged cotangent so
            # the outer orchestration can run that independent transpose as a
            # separate executable after this NTX-only contraction is ready.
            total_lagged_bars = jax.tree_util.tree_map(
                lambda cache_bar, rhs_bar: cache_bar + rhs_bar,
                carry_bars.lagged_response_cache,
                rhs_lagged_bars,
            )
            (
                lagged_flat_bars,
                support_bars,
                native_vmec_coefficient_bars,
            ) = joint_pullback(flat_state0, total_lagged_bars, support_payload)
        else:
            lagged_flat_bars, support_bars = (
                _initial_lagged_response_joint_state_and_support_pullback(
                flat_y=flat_state0,
                cache_lagged_bars=carry_bars.lagged_response_cache,
                rhs_lagged_bars=rhs_lagged_bars,
                support_payload=support_payload,
                joint_pullback=joint_pullback,
                )
            )
        if project_flat is not None:
            _, project_pullback = jax.vjp(project_flat, flat_state0)
            lagged_flat_bars = jax.vmap(project_pullback)(lagged_flat_bars)[0]
        total_flat_bars = direct_flat_bars + lagged_flat_bars
        _, state_pullback = jax.vjp(_flat_state_from_state, state_value)
        state_bars = jax.vmap(lambda flat_bar: state_pullback(flat_bar)[0])(
            total_flat_bars
        )
        if use_split_joint:
            return (
                state_bars,
                support_bars,
                native_vmec_coefficient_bars,
                total_lagged_bars,
            )
        return state_bars, support_bars

    _build_initial_carry.defvjp(_build_initial_carry_fwd, _build_initial_carry_bwd)
    if (
        return_native_joint_pullback
        or return_native_split_joint_pullback
        or return_native_split_joint_no_prepared_carry
        or return_native_split_joint_fused_rhs_pullback
    ):
        carry0, residual = _build_initial_carry_fwd(state)

        def _pullback(carry_bars, support_payload):
            return _native_joint_batched_pullback(
                residual, carry_bars, support_payload
            )

        return carry0, _pullback
    return _build_initial_carry(state)


def prepare_reverse_static_setup(
    parameter_values,
    *,
    config: Mapping[str, Any],
    runtime,
    baseline_state,
    profile_cfg: Mapping[str, Any],
    initial_er_root_ad: str = "off",
    accepted_step_limit_override: int | None = None,
    reverse_segment_length: int | str | None = None,
    reverse_direct_stage_adjoint: bool = False,
    reverse_stage_adjoint_solve_mode: str = "structured",
    reverse_rhs_transpose_mode: str = "generic",
    reverse_rhs_pullback_mode: str = "separate",
    reverse_initial_cache_support_pullback_mode: str = "scalar",
    reverse_rebuild_support_pullback_mode: str = "separate",
    reverse_segment_jit_diagnostics: bool = False,
    reverse_segment_input_diagnostics: bool = False,
    reverse_rebuild_component_timing: bool = False,
    reverse_phase_timing_diagnostics: bool = False,
    reverse_segment_profile_annotations: bool = False,
    reverse_segment_start_replay_mode: str = "legacy",
    reverse_segment_primal_record_mode: str = "reconstruct",
    reverse_final_objective_cotangent_mode: str = "scalar",
    reverse_bootstrap_cotangent_mode: str = "separate",
    reverse_stage_cotangent_mode: str = "full",
    reverse_step_bwd_mode: str = "current",
    reverse_stage_adjoint_memory_mode: str = "default",
    reverse_stage_adjoint_iter_maxiter: int = 40,
    reverse_stage_adjoint_iter_tol: float = 1.0e-10,
    reverse_stage_adjoint_woodbury_rank: int = 24,
    reverse_single_segment_vjp_forward_mode: str = "legacy",
    reverse_schedule_artifact_mode: str = "legacy",
    max_reverse_accepted_steps: int | None = None,
) -> RealtimeGeometryReverseStaticSetup:
    state0_static = initial_state_for_parameter_vector(
        parameter_values,
        config=config,
        initial_er_root_ad=initial_er_root_ad,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        runtime=runtime,
    )
    prepared_components_static = prepare_transport_solver_components(dict(config), runtime, state0_static)
    solver = prepared_components_static["solver"]
    reverse_segment_start_replay_mode = str(reverse_segment_start_replay_mode).strip().lower()
    if reverse_segment_start_replay_mode not in {"legacy", "minimal"}:
        raise ValueError(
            "reverse_segment_start_replay_mode must be one of {'legacy', 'minimal'}."
        )
    if reverse_segment_start_replay_mode == "minimal" and not isinstance(solver, RADAUSolver):
        raise ValueError(
            "reverse_segment_start_replay_mode='minimal' is currently implemented only for RADAUSolver."
        )
    if reverse_segment_start_replay_mode == "minimal" and not reverse_direct_stage_adjoint:
        raise ValueError(
            "reverse_segment_start_replay_mode='minimal' requires reverse_direct_stage_adjoint=True."
        )
    reverse_segment_primal_record_mode = str(
        reverse_segment_primal_record_mode
    ).strip().lower()
    if reverse_segment_primal_record_mode not in {
        "reconstruct",
        "reuse_segment_primal_record",
    }:
        raise ValueError(
            "reverse_segment_primal_record_mode must be one of "
            "{'reconstruct', 'reuse_segment_primal_record'}."
        )
    if (
        reverse_segment_primal_record_mode == "reuse_segment_primal_record"
        and reverse_segment_start_replay_mode != "minimal"
    ):
        raise ValueError(
            "reverse_segment_primal_record_mode='reuse_segment_primal_record' "
            "requires reverse_segment_start_replay_mode='minimal'."
        )
    reverse_final_objective_cotangent_mode = str(
        reverse_final_objective_cotangent_mode
    ).strip().lower()
    if reverse_final_objective_cotangent_mode not in {"scalar", "grouped_vjp"}:
        raise ValueError(
            "reverse_final_objective_cotangent_mode must be one of "
            "{'scalar', 'grouped_vjp'}."
        )
    reverse_bootstrap_cotangent_mode = str(reverse_bootstrap_cotangent_mode).strip().lower()
    if reverse_bootstrap_cotangent_mode not in {
        "separate",
        "joint_local_vjp",
        "joint_local_vjp_upar_only",
    }:
        raise ValueError(
            "reverse_bootstrap_cotangent_mode must be one of "
            "{'separate', 'joint_local_vjp', 'joint_local_vjp_upar_only'}."
        )
    reverse_rebuild_support_pullback_mode = str(
        reverse_rebuild_support_pullback_mode
    ).strip().lower()
    if reverse_rebuild_support_pullback_mode not in {
        "separate",
        "separate_reuse_local_vjp_primal",
        "separate_reuse_local_vjp_primal_geometry_only_prepared",
        "separate_reuse_local_vjp_primal_geometry_implicit_ntx_two_directional",
        "separate_reuse_local_vjp_primal_support_only_ntx_implicit",
        "separate_reuse_local_vjp_primal_factorized_ntx_two_directional",
        "ntx_batched_interpolated_faces",
        "ntx_batched_interpolated_faces_reuse_local_vjp_primal",
        "ntx_batched_interpolated_faces_multi_rhs_shared_primal",
        "ntx_batched_interpolated_faces_native_multi_rhs_shared_primal",
        "ntx_batched_interpolated_faces_native_multi_rhs_compact_shared_primal",
        "ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal",
        "ntx_batched_interpolated_faces_native_multi_rhs_compact_residual_reuse_moment_drds_jvp_shared_primal",
        "ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients",
        "ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_directional_product_rule",
        "ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_coefficient_pullback",
        "ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_directional_product_rule_per_energy_call_boundary",
        "ntx_joint_implicit_interpolated_faces",
        "ntx_joint_implicit_interpolated_faces_packed_support_adjoint",
        "ntx_joint_implicit_interpolated_faces_reuse_local_vjp_primal",
        "ntx_joint_implicit_interpolated_faces_reuse_local_vjp_primal_compact_prepared_carry",
    }:
        raise ValueError(
            "reverse_rebuild_support_pullback_mode must be one of "
            "{'separate', 'separate_reuse_local_vjp_primal', "
            "'separate_reuse_local_vjp_primal_geometry_only_prepared', "
            "'separate_reuse_local_vjp_primal_geometry_implicit_ntx_two_directional', "
            "'separate_reuse_local_vjp_primal_support_only_ntx_implicit', "
            "'separate_reuse_local_vjp_primal_factorized_ntx_two_directional', "
            "'ntx_batched_interpolated_faces', "
            "'ntx_batched_interpolated_faces_reuse_local_vjp_primal', "
            "'ntx_batched_interpolated_faces_multi_rhs_shared_primal', "
            "'ntx_batched_interpolated_faces_native_multi_rhs_shared_primal', "
            "'ntx_batched_interpolated_faces_native_multi_rhs_compact_shared_primal', "
            "'ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal', "
            "'ntx_batched_interpolated_faces_native_multi_rhs_compact_residual_reuse_moment_drds_jvp_shared_primal', "
            "'ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients', "
            "'ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_directional_product_rule', "
            "'ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_coefficient_pullback', "
            "'ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_directional_product_rule_per_energy_call_boundary', "
            "'ntx_joint_implicit_interpolated_faces', "
            "'ntx_joint_implicit_interpolated_faces_packed_support_adjoint', "
            "'ntx_joint_implicit_interpolated_faces_reuse_local_vjp_primal', "
            "'ntx_joint_implicit_interpolated_faces_reuse_local_vjp_primal_compact_prepared_carry'}."
        )
    if (
        reverse_rebuild_support_pullback_mode
        in {
            "ntx_batched_interpolated_faces",
            "ntx_batched_interpolated_faces_reuse_local_vjp_primal",
            "ntx_batched_interpolated_faces_multi_rhs_shared_primal",
            "ntx_batched_interpolated_faces_native_multi_rhs_shared_primal",
            "ntx_batched_interpolated_faces_native_multi_rhs_compact_shared_primal",
            "ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal",
            "ntx_batched_interpolated_faces_native_multi_rhs_compact_residual_reuse_moment_drds_jvp_shared_primal",
            "ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients",
            "ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_directional_product_rule",
            "ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_coefficient_pullback",
            "ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_directional_product_rule_per_energy_call_boundary",
            "ntx_joint_implicit_interpolated_faces",
            "ntx_joint_implicit_interpolated_faces_packed_support_adjoint",
            "ntx_joint_implicit_interpolated_faces_reuse_local_vjp_primal",
            "ntx_joint_implicit_interpolated_faces_reuse_local_vjp_primal_compact_prepared_carry",
        }
        and not reverse_direct_stage_adjoint
    ):
        raise ValueError(
            "batched reverse_rebuild_support_pullback_mode "
            "requires reverse_direct_stage_adjoint=True."
        )
    solve_vector_field_static = prepared_components_static["solve_vector_field"]
    prepared_rollout_static = _build_prepared_reverse_accepted_rollout(
        solver=solver,
        state=state0_static,
        vector_field=solve_vector_field_static,
        species=runtime.species,
    )
    execution_context = _build_prepared_reverse_execution_context(
        solver=solver,
        prepared_rollout=prepared_rollout_static,
    )
    if reverse_direct_stage_adjoint:
        execution_context = dataclasses.replace(
            execution_context,
            physics_context=dataclasses.replace(
                execution_context.physics_context,
                reverse_direct_stage_adjoint=True,
                reverse_stage_adjoint_solve_mode=str(reverse_stage_adjoint_solve_mode),
                reverse_rhs_transpose_mode=str(reverse_rhs_transpose_mode),
                reverse_rhs_pullback_mode=str(reverse_rhs_pullback_mode),
                reverse_initial_cache_support_pullback_mode=str(
                    reverse_initial_cache_support_pullback_mode
                ),
                reverse_rebuild_support_pullback_mode=str(
                    reverse_rebuild_support_pullback_mode
                ),
                reverse_segment_jit_diagnostics=bool(reverse_segment_jit_diagnostics),
                reverse_segment_input_diagnostics=bool(reverse_segment_input_diagnostics),
                reverse_rebuild_component_timing=bool(reverse_rebuild_component_timing),
                reverse_phase_timing_diagnostics=bool(reverse_phase_timing_diagnostics),
                reverse_segment_profile_annotations=bool(reverse_segment_profile_annotations),
                reverse_segment_start_replay_mode=str(reverse_segment_start_replay_mode),
                reverse_segment_primal_record_mode=str(reverse_segment_primal_record_mode),
                reverse_final_objective_cotangent_mode=(
                    reverse_final_objective_cotangent_mode
                ),
                reverse_bootstrap_cotangent_mode=reverse_bootstrap_cotangent_mode,
                reverse_stage_cotangent_mode=str(reverse_stage_cotangent_mode),
                reverse_step_bwd_mode=str(reverse_step_bwd_mode),
                reverse_stage_adjoint_memory_mode=str(reverse_stage_adjoint_memory_mode),
                reverse_stage_adjoint_iter_maxiter=int(reverse_stage_adjoint_iter_maxiter),
                reverse_stage_adjoint_iter_tol=float(reverse_stage_adjoint_iter_tol),
                reverse_stage_adjoint_woodbury_rank=int(reverse_stage_adjoint_woodbury_rank),
                reverse_single_segment_vjp_forward_mode=str(reverse_single_segment_vjp_forward_mode),
            ),
        )
    stop_after_accepted_steps = (
        int(accepted_step_limit_override)
        if accepted_step_limit_override is not None
        else getattr(solver, "stop_after_accepted_steps", None)
    )
    requested_full_final_time = stop_after_accepted_steps is None
    max_total_steps = int(max(1, getattr(solver, "max_steps", 1)))
    reverse_segment_auto_quarter = (
        isinstance(reverse_segment_length, str)
        and str(reverse_segment_length).strip().lower()
        in {"auto", "auto_quarter", "quarter", "accepted_quarter"}
    )
    reverse_segment_length_eff = None
    if reverse_segment_length is not None and not reverse_segment_auto_quarter:
        reverse_segment_length_eff = int(reverse_segment_length)
    needs_schedule_probe = (
        stop_after_accepted_steps is not None
        or reverse_segment_auto_quarter
        or (requested_full_final_time and reverse_segment_length_eff is not None)
        or (requested_full_final_time and max_reverse_accepted_steps is not None)
    )
    schedule_artifact = None
    schedule_artifact_mode = str(reverse_schedule_artifact_mode).strip().lower()
    if schedule_artifact_mode not in {"legacy", "reuse_static_probe"}:
        raise ValueError(
            "Unknown reverse_schedule_artifact_mode "
            f"'{reverse_schedule_artifact_mode}'."
        )
    if schedule_artifact_mode == "reuse_static_probe" and not needs_schedule_probe:
        raise ValueError(
            "reverse_schedule_artifact_mode='reuse_static_probe' requires a static schedule probe."
        )
    if (
        schedule_artifact_mode == "reuse_static_probe"
        and str(reverse_single_segment_vjp_forward_mode).strip().lower()
        == "reuse_adaptive_rollout"
    ):
        raise ValueError(
            "reuse_static_probe already removes the adaptive rollout and cannot be combined "
            "with reverse_single_segment_vjp_forward_mode='reuse_adaptive_rollout'."
        )
    if needs_schedule_probe:
        probe_max_total_steps = max_total_steps
        probe_stop_after_accepted_steps = stop_after_accepted_steps
        if requested_full_final_time and max_reverse_accepted_steps is not None:
            probe_stop_after_accepted_steps = int(max_reverse_accepted_steps)
        if probe_stop_after_accepted_steps is not None:
            probe_max_total_steps = min(
                max_total_steps,
                max(
                    int(probe_stop_after_accepted_steps) * 16,
                    int(probe_stop_after_accepted_steps) + 16,
                ),
            )
        max_total_steps = min(
            max_total_steps,
            max(
                int(probe_stop_after_accepted_steps) * 16,
                int(probe_stop_after_accepted_steps) + 16,
            ),
        ) if probe_stop_after_accepted_steps is not None else max_total_steps
        schedule_probe = _reverse_adaptive_schedule_rollout(
            execution_context,
            prepared_rollout_static.initial_carry,
            max_total_steps=probe_max_total_steps,
            stop_after_accepted_steps=probe_stop_after_accepted_steps,
            capture_segment_length=reverse_segment_length_eff,
        )
        actual_attempt_count = int(np.asarray(jax.device_get(schedule_probe.attempt_count)))
        active_mask_np = np.asarray(jax.device_get(schedule_probe.trace.active_mask), dtype=bool)
        accepted_mask_np = np.asarray(jax.device_get(schedule_probe.trace.accepted_mask), dtype=bool)
        accepted_count_np = int(np.sum(np.logical_and(active_mask_np, accepted_mask_np)))
        if requested_full_final_time and max_reverse_accepted_steps is not None:
            final_time = float(np.asarray(jax.device_get(schedule_probe.final_carry.t)))
            target_time = float(getattr(solver, "t1", final_time))
            if final_time < target_time - 1.0e-12 and accepted_count_np >= int(max_reverse_accepted_steps):
                raise RuntimeError(
                    "full transport reverse trial exceeded optimization accepted-step guard "
                    f"(accepted_steps={accepted_count_np}, max_reverse_accepted_steps="
                    f"{int(max_reverse_accepted_steps)}, final_time={final_time:.16e}, "
                    f"target_time={target_time:.16e}); treating trial as failed for optimization."
                )
        if reverse_segment_auto_quarter:
            reverse_segment_length_eff = max(1, (accepted_count_np + 3) // 4)
        if requested_full_final_time and reverse_segment_length_eff is not None:
            stop_after_accepted_steps = max(1, accepted_count_np)
        replay_accepted_limit = (
            int(stop_after_accepted_steps)
            if stop_after_accepted_steps is not None
            else int(probe_stop_after_accepted_steps)
            if probe_stop_after_accepted_steps is not None
            else 1
        )
        max_total_steps = min(
            probe_max_total_steps,
            max(actual_attempt_count + 2, replay_accepted_limit),
        )
        if stop_after_accepted_steps is None:
            accepted_limit = int(max_total_steps)
        else:
            accepted_limit = int(stop_after_accepted_steps)
        next_lagged_valid_np = np.asarray(
            jax.device_get(schedule_probe.trace.next_lagged_response_valid),
            dtype=bool,
        )
        accepted_positions = np.nonzero(np.logical_and(active_mask_np, accepted_mask_np))[0][:accepted_limit]
        incoming_valid = bool(np.asarray(jax.device_get(prepared_rollout_static.initial_carry.lagged_response_valid)))
        lagged_branch_schedule: list[bool] = []
        for accepted_position in accepted_positions:
            lagged_branch_schedule.append(bool(incoming_valid))
            incoming_valid = bool(next_lagged_valid_np[int(accepted_position)])
        if len(lagged_branch_schedule) < accepted_limit:
            lagged_branch_schedule.extend([bool(incoming_valid)] * (accepted_limit - len(lagged_branch_schedule)))
        execution_context = dataclasses.replace(
            execution_context,
            physics_context=dataclasses.replace(
                execution_context.physics_context,
                reverse_lagged_branch_schedule=tuple(lagged_branch_schedule),
            ),
        )
        if schedule_artifact_mode == "reuse_static_probe":
            # Keep only the scalar adaptive schedule.  Do not retain
            # final_step_state, final_carry, or any accepted-step carries/stages.
            # Match the graph length used by the legacy manual VJP forward;
            # the initial probe may have used a deliberately larger guard.
            schedule_artifact = jax.tree_util.tree_map(
                lambda value: value[:max_total_steps],
                schedule_probe.trace,
            )
    return RealtimeGeometryReverseStaticSetup(
        solver=solver,
        solve_vector_field=solve_vector_field_static,
        prepared_rollout=prepared_rollout_static,
        execution_context=execution_context,
        stop_after_accepted_steps=stop_after_accepted_steps,
        max_total_steps=max_total_steps,
        reverse_segment_length=reverse_segment_length_eff,
        require_final_time=bool(requested_full_final_time and reverse_segment_length_eff is not None),
        schedule_artifact=schedule_artifact,
        schedule_segment_start_carries=(
            schedule_probe.segment_start_carries
            if schedule_artifact is not None else None
        ),
        schedule_final_carry=(
            schedule_probe.final_carry if schedule_artifact is not None else None
        ),
    )


def default_realtime_geometry_support_reverse_dependencies() -> RealtimeGeometrySupportReverseDependencies:
    """Return the production/default dependency bundle for the compact reverse rule."""

    return RealtimeGeometrySupportReverseDependencies(
        initial_er_root_enabled=initial_er_root_enabled,
        initial_state_for_parameter_vector=initial_state_for_parameter_vector,
        state_with_initial_er_root_ad=state_with_initial_er_root_ad,
        reverse_initial_carry_from_state_with_static_setup=reverse_initial_carry_from_state_with_static_setup,
        objective_scalar_by_index=objective_scalar_by_index,
        add_trees=add_trees,
        initial_er_selected_root_profile=initial_er_selected_root_profile,
        initial_er_charge_flux_residuals=initial_er_charge_flux_residuals,
        initial_er_charge_flux_residual_scalar=initial_er_charge_flux_residual_scalar,
        initial_er_charge_flux_residual_er_derivative=initial_er_charge_flux_residual_er_derivative,
        compact_initial_er_state_pullback=compact_initial_er_state_pullback,
        compact_initial_er_ntx_support_pullback_leaves=compact_initial_er_ntx_support_pullback_leaves,
        runtime_with_geometry_payload=runtime_with_geometry_payload,
        runtime_with_ntx_support_payload=runtime_with_ntx_support_payload,
        runtime_with_realtime_geometry_reverse_support_payload=(
            runtime_with_realtime_geometry_reverse_support_payload
        ),
    )


def realtime_geometry_transport_reverse_table_context(
    *,
    config: Mapping[str, Any],
    baseline_values,
    baseline_runtime,
    baseline_state,
    profile_cfg: Mapping[str, Any],
    neoclassical_cfg: Mapping[str, Any],
) -> RealtimeGeometryTransportReverseTableContext:
    """Build the grouped realtime-geometry transport reverse table context."""

    return RealtimeGeometryTransportReverseTableContext(
        config=config,
        baseline_values=baseline_values,
        baseline_runtime=baseline_runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        neoclassical_cfg=neoclassical_cfg,
    )


def prepare_realtime_geometry_support_segment_core_setup(
    *,
    args,
    config: Mapping[str, Any],
    baseline_values,
    baseline_runtime,
    baseline_state,
    profile_cfg: Mapping[str, Any],
    neoclassical_cfg: Mapping[str, Any],
    parameter_order: Sequence[str],
    find_ntx_support_payload: NtxSupportPayloadBuilder,
    prepare_reverse_static_setup: ReverseStaticSetupBuilder,
    geometry_volume_diagnostics: GeometryDiagnosticsBuilder | None = None,
) -> RealtimeGeometrySupportSegmentCoreSetup:
    """Prepare reusable inputs for a segmented realtime-geometry support pullback.

    This keeps the benchmark-owned heavy reverse sweep unchanged while moving
    the stable support-payload selection and reverse static setup into the
    internal transport reverse-AD API.
    """

    if not callable(find_ntx_support_payload):
        raise TypeError("find_ntx_support_payload must be callable.")
    if not callable(prepare_reverse_static_setup):
        raise TypeError("prepare_reverse_static_setup must be callable.")
    if geometry_volume_diagnostics is not None and not callable(geometry_volume_diagnostics):
        raise TypeError("geometry_volume_diagnostics must be callable when provided.")
    parameter_labels = tuple(str(name) for name in parameter_order)
    # ``support_segment_probe`` is not a support-only local-VJP check.  It
    # executes the production segmented reverse accumulation, then inspects
    # its payload cotangents before the VMEC pullback.  It must therefore use
    # the same combined geometry/NTX tree as ``reverse_payload``; otherwise
    # the native rebuild bridge receives a bare NTX support tree while its
    # merge contract requires ``{"geometry", "ntx_support"}``.
    combined_geometry_payload = str(args.realtime_geometry_gradient_path) in {
        "reverse_payload",
        "support_segment_probe",
    }
    ntx_surface_backend = str(neoclassical_cfg.get("ntx_exact_surface_backend", "booz"))
    runtime_payload = realtime_geometry_payload_for_runtime(baseline_runtime)
    payload_kind = str(runtime_payload["kind"])
    if payload_kind == "ntx_exact":
        # Preserve the established exact tree and call boundary unchanged.
        ntx_support_payload = find_ntx_support_payload(baseline_runtime)
        support_payload = (
            {"geometry": baseline_runtime.geometry, "ntx_support": ntx_support_payload}
            if combined_geometry_payload
            else ntx_support_payload
        )
    elif payload_kind == "ntx_scan_runtime":
        if not combined_geometry_payload:
            raise ValueError(
                "ntx_scan_runtime reverse requires "
                "--realtime-geometry-gradient-path reverse_payload."
            )
        rebuild_mode = str(
            getattr(args, "reverse_rebuild_support_pullback_mode", "separate")
        ).strip().lower()
        initial_cache_mode = str(
            getattr(args, "reverse_initial_cache_support_pullback_mode", "scalar")
        ).strip().lower()
        if rebuild_mode != "separate" or initial_cache_mode != "scalar":
            raise ValueError(
                "ntx_scan_runtime currently supports only the generic "
                "reverse support selectors: "
                "--reverse-rebuild-support-pullback-mode separate and "
                "--reverse-initial-cache-support-pullback-mode scalar. "
                "The ntx_* selectors require a prepared exact-NTX system."
            )
        final_cotangent_mode = str(
            getattr(args, "reverse_final_objective_cotangent_mode", "scalar")
        ).strip().lower()
        if final_cotangent_mode not in {"scalar", "grouped_vjp"}:
            raise ValueError(
                "ntx_scan_runtime requires "
                "--reverse-final-objective-cotangent-mode scalar or grouped_vjp."
            )
        ntx_support_payload = None
        support_payload = realtime_geometry_reverse_support_payload_for_runtime(
            baseline_runtime
        )
        rhs_transpose_mode = str(
            getattr(args, "reverse_rhs_transpose_mode", "generic")
        ).strip().lower()
        if rhs_transpose_mode in {
            "explicit_database",
            "database",
            "explicit_black_box_database",
        }:
            centre_mode = str(
                getattr(baseline_runtime.models.flux, "center_flux_mode", "")
            ).strip().lower()
            if centre_mode != "direct":
                raise ValueError(
                    "reverse_rhs_transpose_mode='explicit_database' requires "
                    "the black-box database forward contract "
                    "center_flux_mode='direct'; got "
                    f"{centre_mode!r}."
                )
            if "database" not in support_payload:
                raise ValueError(
                    "reverse_rhs_transpose_mode='explicit_database' requires "
                    "a recorded runtime database support payload. Enable "
                    "--ntx-scan-coefficient-reverse-mode structured and "
                    "--ntx-scan-record-primal."
                )
    else:
        raise NotImplementedError(
            "Realtime geometry segmented reverse currently supports "
            f"ntx_exact and ntx_scan_runtime payloads; got {payload_kind!r}."
        )
    profile_values = baseline_values[: len(parameter_labels)]
    support_probe_cotangent_mode = str(args.reverse_stage_cotangent_mode)
    reverse_setup = prepare_reverse_static_setup(
        profile_values,
        config=config,
        runtime=baseline_runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        initial_er_root_ad=args.initial_er_root_ad,
        accepted_step_limit_override=args.accepted_step_limit,
        reverse_segment_length=args.reverse_segment_length,
        reverse_direct_stage_adjoint=True,
        reverse_stage_adjoint_solve_mode=args.reverse_stage_adjoint_solve_mode,
        reverse_rhs_transpose_mode=args.reverse_rhs_transpose_mode,
        reverse_rhs_pullback_mode=getattr(args, "reverse_rhs_pullback_mode", "separate"),
        reverse_initial_cache_support_pullback_mode=getattr(
            args,
            "reverse_initial_cache_support_pullback_mode",
            "scalar",
        ),
        reverse_rebuild_support_pullback_mode=getattr(
            args,
            "reverse_rebuild_support_pullback_mode",
            "separate",
        ),
        reverse_segment_input_diagnostics=bool(
            getattr(args, "reverse_segment_input_diagnostics", False)
        ),
        reverse_segment_start_replay_mode=getattr(
            args,
            "reverse_segment_start_replay_mode",
            "legacy",
        ),
        reverse_segment_primal_record_mode=getattr(
            args,
            "reverse_segment_primal_record_mode",
            "reconstruct",
        ),
        reverse_final_objective_cotangent_mode=getattr(
            args,
            "reverse_final_objective_cotangent_mode",
            "scalar",
        ),
        reverse_bootstrap_cotangent_mode=getattr(
            args,
            "reverse_bootstrap_cotangent_mode",
            "separate",
        ),
        reverse_stage_cotangent_mode=support_probe_cotangent_mode,
        reverse_step_bwd_mode=args.reverse_step_bwd_mode,
        reverse_stage_adjoint_memory_mode=args.reverse_stage_adjoint_memory_mode,
        reverse_stage_adjoint_iter_maxiter=args.reverse_stage_adjoint_iter_maxiter,
        reverse_stage_adjoint_iter_tol=args.reverse_stage_adjoint_iter_tol,
        reverse_stage_adjoint_woodbury_rank=getattr(args, "reverse_stage_adjoint_woodbury_rank", 24),
        reverse_single_segment_vjp_forward_mode=getattr(
            args,
            "reverse_single_segment_vjp_forward_mode",
            "legacy",
        ),
        reverse_schedule_artifact_mode=getattr(
            args,
            "reverse_schedule_artifact_mode",
            "legacy",
        ),
    )
    if (
        str(getattr(args, "reverse_schedule_artifact_mode", "legacy")).strip().lower()
        == "reuse_static_probe"
        and getattr(reverse_setup, "schedule_artifact", None) is None
    ):
        raise RuntimeError(
            "reuse_static_probe was requested, but reverse static setup returned no schedule artifact."
        )
    early_geometry_diagnostics = (
        None
        if geometry_volume_diagnostics is None
        else geometry_volume_diagnostics(baseline_runtime.geometry)
    )
    return RealtimeGeometrySupportSegmentCoreSetup(
        combined_geometry_payload=combined_geometry_payload,
        payload_kind=payload_kind,
        ntx_surface_backend=ntx_surface_backend,
        ntx_support_payload=ntx_support_payload,
        support_payload=support_payload,
        profile_values=profile_values,
        support_probe_cotangent_mode=support_probe_cotangent_mode,
        reverse_setup=reverse_setup,
        early_geometry_diagnostics=early_geometry_diagnostics,
    )


@contextlib.contextmanager
def _reverse_profile_scope(reverse_setup: RealtimeGeometryReverseStaticSetup, name: str):
    """Emit an XProf label only for the opt-in reverse trace mode."""
    enabled = bool(
        getattr(
            reverse_setup.execution_context.physics_context,
            "reverse_segment_profile_annotations",
            False,
        )
    )
    if enabled:
        with jax.named_scope(name):
            yield
    else:
        yield


def realtime_geometry_reverse_all_objectives_support_payload_bar_for_parameter_vector(
    parameter_values,
    *,
    config: Mapping[str, Any],
    runtime,
    baseline_state,
    profile_cfg: Mapping[str, Any],
    reverse_setup,
    support_payload,
    initial_er_root_ad: str = "off",
    objective_labels: Sequence[str],
    dependencies: RealtimeGeometrySupportReverseDependencies,
    progress_prefix: str = "[autodiff-gate]",
):
    """Return all objective values, profile gradients, and support cotangents.

    This is the migrated implementation of the realtime-geometry grouped
    support reverse kernel. Benchmark-specific objective/profile/root helpers
    are still supplied explicitly through `dependencies` so this move does not
    alter the JAX graph or numerical path.
    """

    if not isinstance(dependencies, RealtimeGeometrySupportReverseDependencies):
        raise TypeError(
            "dependencies must be a RealtimeGeometrySupportReverseDependencies instance."
        )
    objective_labels = tuple(str(label) for label in objective_labels)
    if not objective_labels:
        raise ValueError("objective_labels must contain at least one objective.")
    if reverse_setup.reverse_segment_length is None or int(reverse_setup.reverse_segment_length) <= 0:
        raise ValueError("support payload reverse probe requires --reverse-segment-length.")
    step_bwd_mode = str(
        getattr(reverse_setup.execution_context.physics_context, "reverse_step_bwd_mode", "current")
    ).strip().lower()
    if step_bwd_mode not in {
        "reduced_cotangent",
        "reduced_cotangent_call_boundary",
        "reduced_cotangent_call_boundary_common_branch_hoist",
        "reduced_cotangent_call_boundary_common_branch_hoist_rebuild_call",
        "reduced_cotangent_call_boundary_common_branch_hoist_common_call_rebuild_call",
        "reduced_cotangent_host_static_branches",
        "reduced_cotangent_lean_replay",
        "reduced_cotangent_recompute_replay",
        "lean_replay",
        "recompute_replay",
        "reduced",
        "state_only",
        "final_state",
    }:
        raise ValueError("support payload reverse probe requires a reduced-cotangent reverse step bwd mode.")

    def _zero_tangent_like(x):
        arr = jnp.asarray(x)
        if jnp.issubdtype(arr.dtype, jnp.inexact):
            return jnp.zeros_like(arr)
        return jnp.zeros(arr.shape, dtype=jax.dtypes.float0)

    def _take_tree_axis0(tree, index: int):
        return jax.tree_util.tree_map(lambda value: value[index], tree)

    host_static_branch_dispatch = step_bwd_mode == "reduced_cotangent_host_static_branches"
    if host_static_branch_dispatch:
        record_mode = str(
            getattr(
                reverse_setup.execution_context.physics_context,
                "reverse_segment_primal_record_mode",
                "reconstruct",
            )
        ).strip().lower()
        replay_mode = str(
            getattr(
                reverse_setup.execution_context.physics_context,
                "reverse_segment_start_replay_mode",
                "legacy",
            )
        ).strip().lower()
        if record_mode != "reuse_segment_primal_record" or replay_mode != "minimal":
            raise ValueError(
                "reduced_cotangent_host_static_branches requires "
                "reverse_segment_primal_record_mode='reuse_segment_primal_record' "
                "and reverse_segment_start_replay_mode='minimal'."
            )

    def _batched_zero_tangent_tree_like(primal_tree, batch_size: int):
        zero_tree = _reverse_align_tangent_tree_to_primal(
            reverse_setup.execution_context,
            None,
            primal_tree,
        )
        return jax.tree_util.tree_map(
            lambda leaf: jnp.broadcast_to(
                jnp.asarray(leaf)[None, ...],
                (batch_size,) + jnp.asarray(leaf).shape,
            ),
            zero_tree,
        )

    initial_er_root_enabled = dependencies.initial_er_root_enabled(config, initial_er_root_ad)

    # Do not close this small profile map over the complete runtime: a scan
    # database runtime contains the large coefficient tables and optional
    # recorded scan primal, neither of which is an input to profile creation.
    profile_geometry = runtime.geometry
    profile_number_species = runtime.species.number_species

    def _state_from_profiles(p):
        return initial_state_for_parameter_vector_compact(
            p,
            config=config,
            initial_er_root_ad="off",
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            geometry=profile_geometry,
            number_species=profile_number_species,
        )

    phase_timing_diagnostics = bool(
        getattr(
            reverse_setup.execution_context.physics_context,
            "reverse_phase_timing_diagnostics",
            False,
        )
    )
    phase_start = time.perf_counter()
    pre_root_initial_state, profile_state_pullback = jax.vjp(_state_from_profiles, parameter_values)
    profile_state_vjp_elapsed = None
    if phase_timing_diagnostics:
        # Do not conflate the profile pytree VJP with the separately executed
        # selected-root primal below.  This synchronization is diagnostic-only
        # and mirrors the component timings printed later in the Lij path.
        pre_root_initial_state = jax.block_until_ready(pre_root_initial_state)
        profile_state_vjp_elapsed = time.perf_counter() - phase_start
    # The reverse boundary below implements the selected-root implicit
    # pullback explicitly.  Keep the forward root result here so that
    # boundary does not repeat the same radial root solve just to recover its
    # primal value and finite-root mask.
    initial_er_root_primal = None
    selected_root_primal_elapsed = None
    if initial_er_root_enabled:
        selected_root_start = time.perf_counter()
        initial_er_root_primal = dependencies.initial_er_selected_root_profile(
            pre_root_initial_state,
            config=config,
            runtime=runtime,
        )
        initial_state = dataclasses.replace(
            pre_root_initial_state,
            Er=jnp.asarray(
                initial_er_root_primal[0], dtype=pre_root_initial_state.Er.dtype
            ),
        )
        if phase_timing_diagnostics:
            initial_state = jax.block_until_ready(initial_state)
            selected_root_primal_elapsed = time.perf_counter() - selected_root_start
    else:
        initial_state = pre_root_initial_state
    initial_state = jax.block_until_ready(initial_state)
    print(
        f"{progress_prefix} progress: support reverse profile-state vjp ready "
        f"elapsed_s={time.perf_counter() - phase_start:.3f}",
        flush=True,
    )
    if phase_timing_diagnostics:
        root_time_text = (
            "off"
            if selected_root_primal_elapsed is None
            else f"{selected_root_primal_elapsed:.3f}"
        )
        profile_time_text = (
            "n/a"
            if profile_state_vjp_elapsed is None
            else f"{profile_state_vjp_elapsed:.3f}"
        )
        print(
            f"{progress_prefix} diagnostic: initial-state construction "
            f"profile_state_vjp_compile_plus_execute_s={profile_time_text} "
            f"selected_root_primal_compile_plus_execute_s={root_time_text} "
            f"assembly_and_sync_s={time.perf_counter() - phase_start - (profile_state_vjp_elapsed or 0.0) - (selected_root_primal_elapsed or 0.0):.3f}",
            flush=True,
        )

    def _carry_from_state(state_value):
        return dependencies.reverse_initial_carry_from_state_with_static_setup(
            solver=reverse_setup.solver,
            state=state_value,
            solve_vector_field=reverse_setup.solve_vector_field,
            species=runtime.species,
            prepared_rollout_static=reverse_setup.prepared_rollout,
        )

    initial_cache_support_pullback_mode = str(
        getattr(
            reverse_setup.execution_context.physics_context,
            "reverse_initial_cache_support_pullback_mode",
            "scalar",
        )
    ).strip().lower()
    use_native_joint_initial_carry_pullback = (
        initial_cache_support_pullback_mode
        == "ntx_native_joint_state_and_support"
    )
    use_native_split_joint_initial_carry_pullback = (
        initial_cache_support_pullback_mode
        == "ntx_native_joint_state_and_ntx_support_split_geometry_vmec"
    )
    use_native_split_joint_no_prepared_carry_initial_carry_pullback = (
        initial_cache_support_pullback_mode
        == "ntx_native_joint_state_and_ntx_support_split_geometry_vmec_no_prepared_carry"
    )
    use_native_split_joint_fused_rhs_initial_carry_pullback = (
        initial_cache_support_pullback_mode
        == "ntx_native_joint_state_and_ntx_support_split_geometry_vmec_fused_rhs"
    )
    use_rebuild_dispatch_initial_cache_pullback = (
        initial_cache_support_pullback_mode == "rebuild_dispatch"
    )
    phase_start = time.perf_counter()
    if (
        use_native_joint_initial_carry_pullback
        or use_native_split_joint_initial_carry_pullback
        or use_native_split_joint_no_prepared_carry_initial_carry_pullback
        or use_native_split_joint_fused_rhs_initial_carry_pullback
    ):
        initial_carry, initial_state_pullback = (
            dependencies.reverse_initial_carry_from_state_with_static_setup(
                solver=reverse_setup.solver,
                state=initial_state,
                solve_vector_field=reverse_setup.solve_vector_field,
                species=runtime.species,
                prepared_rollout_static=reverse_setup.prepared_rollout,
                return_native_joint_pullback=True,
                return_native_split_joint_pullback=(
                    use_native_split_joint_initial_carry_pullback
                ),
                return_native_split_joint_no_prepared_carry=(
                    use_native_split_joint_no_prepared_carry_initial_carry_pullback
                ),
                return_native_split_joint_fused_rhs_pullback=(
                    use_native_split_joint_fused_rhs_initial_carry_pullback
                ),
            )
        )
        initial_carry_vjp_label = "native joint forward"
    else:
        initial_carry, initial_state_pullback = jax.vjp(
            _carry_from_state, initial_state
        )
        initial_carry_vjp_label = "vjp"
    initial_carry = jax.block_until_ready(initial_carry)
    print(
        f"{progress_prefix} progress: support reverse initial carry "
        f"{initial_carry_vjp_label} ready "
        f"elapsed_s={time.perf_counter() - phase_start:.3f}",
        flush=True,
    )

    phase_start = time.perf_counter()
    schedule_artifact = getattr(reverse_setup, "schedule_artifact", None)
    if schedule_artifact is None:
        final_y, residuals = _reverse_adaptive_final_y_realized_schedule_vjp_fwd(
            reverse_setup.execution_context,
            reverse_setup.max_total_steps,
            reverse_setup.stop_after_accepted_steps,
            reverse_setup.reverse_segment_length,
            initial_carry,
        )
    else:
        print(
            f"{progress_prefix} progress: support reverse reusing static schedule artifact "
            "(no second adaptive rollout; no per-step carry tape)",
            flush=True,
        )
        if isinstance(reverse_setup.execution_context, _ThetaReverseExecutionContext):
            raise ValueError(
                "reuse_static_probe is currently implemented only for the Radau reverse path."
            )
        final_y, residuals = _radau_adaptive_final_y_realized_schedule_vjp_fwd_from_schedule_artifact(
            reverse_setup.execution_context,
            reverse_setup.max_total_steps,
            reverse_setup.stop_after_accepted_steps,
            reverse_setup.reverse_segment_length,
            initial_carry,
            schedule_artifact,
            final_carry=getattr(reverse_setup, "schedule_final_carry", None),
            segment_start_carries_artifact=getattr(
                reverse_setup, "schedule_segment_start_carries", None
            ),
        )
    final_y, residuals = jax.block_until_ready((final_y, residuals))
    print(
        f"{progress_prefix} progress: support reverse realized-schedule vjp forward ready "
        f"elapsed_s={time.perf_counter() - phase_start:.3f}",
        flush=True,
    )
    if phase_timing_diagnostics:
        print(
            f"{progress_prefix} diagnostic: realized-schedule residual construction "
            f"compile_plus_replay_execute_s={time.perf_counter() - phase_start:.3f} "
            "(fixed accepted schedule; this is not an adaptive rerun)",
            flush=True,
        )

    (
        carry0,
        active_mask,
        accepted_mask,
        attempted_dts,
        next_dts,
        next_recent_reject_count,
        next_regrowth_cooldown,
        next_easy_growth_streak,
        next_lagged_response_valid,
        segment_start_carries,
        segmented_final_carry,
        segmented_replay_arrays,
    ) = residuals
    del (
        active_mask,
        accepted_mask,
        attempted_dts,
        next_dts,
        next_recent_reject_count,
        next_regrowth_cooldown,
        next_easy_growth_streak,
        next_lagged_response_valid,
    )
    if segment_start_carries is None or segmented_final_carry is None or segmented_replay_arrays is None:
        raise ValueError("support payload reverse probe requires segmented reverse residuals.")

    final_y_for_objective = segmented_final_carry.y
    if bool(getattr(reverse_setup, "require_final_time", False)):
        final_time = jnp.asarray(segmented_final_carry.t)
        target_time = jnp.asarray(getattr(reverse_setup.solver, "t1", final_time), dtype=final_time.dtype)
        final_time_ready, target_time_ready = jax.block_until_ready((final_time, target_time))
        if float(final_time_ready) < float(target_time_ready) - 1.0e-12:
            raise RuntimeError(
                "full transport reverse trial did not reach solver final time "
                f"(final_time={float(final_time_ready):.16e}, target_time={float(target_time_ready):.16e}); "
                "treating trial as failed for optimization."
            )

    objective_count = int(len(objective_labels))
    objective_values_rows = []
    final_y_bar_rows = []
    objective_payload_bar_rows = []
    combined_geometry_payload = isinstance(support_payload, dict) and "geometry" in support_payload
    zero_payload_bar = _reverse_zero_support_delta_tree_like(reverse_setup.solver, support_payload)
    final_objective_cotangent_mode = str(
        getattr(
            reverse_setup.execution_context.physics_context,
            "reverse_final_objective_cotangent_mode",
            "scalar",
        )
    ).strip().lower()
    if final_objective_cotangent_mode not in {"scalar", "grouped_vjp"}:
        raise ValueError(
            "Unknown reverse_final_objective_cotangent_mode "
            f"{final_objective_cotangent_mode!r}."
        )
    bootstrap_cotangent_mode = str(
        getattr(
            reverse_setup.execution_context.physics_context,
            "reverse_bootstrap_cotangent_mode",
            "separate",
        )
    ).strip().lower()
    if bootstrap_cotangent_mode not in {
        "separate",
        "joint_local_vjp",
        "joint_local_vjp_upar_only",
    }:
        raise ValueError(
            "Unknown reverse_bootstrap_cotangent_mode "
            f"{bootstrap_cotangent_mode!r}."
        )
    bootstrap_objective_name = "bootstrap_current_softmax_abs_scaled"
    ordinary_objective_indices = tuple(
        objective_i
        for objective_i, objective_name in enumerate(objective_labels)
        if objective_name != bootstrap_objective_name
    )
    grouped_objective_values: dict[int, object] = {}
    grouped_final_y_bars: dict[int, object] = {}
    grouped_geometry_bars: dict[int, object] = {}
    phase_timing_diagnostics = bool(
        getattr(
            reverse_setup.execution_context.physics_context,
            "reverse_phase_timing_diagnostics",
            False,
        )
    )
    final_objective_state_elapsed = 0.0
    final_objective_geometry_elapsed = 0.0
    final_objective_bootstrap_elapsed = 0.0
    phase_start = time.perf_counter()
    if final_objective_cotangent_mode == "grouped_vjp" and ordinary_objective_indices:
        # The scalar reference path constructs one VJP for every ordinary
        # terminal objective.  Group them here, but leave bootstrap on its
        # compact NTX-specific rule below.
        def _ordinary_objective_vector_from_final_y(final_y_value):
            final_state = reverse_setup.prepared_rollout.physics_context.unpack_flat(
                final_y_value
            )
            return jnp.stack(
                tuple(
                    dependencies.objective_scalar_by_index(final_state, runtime, objective_i)
                    for objective_i in ordinary_objective_indices
                ),
                axis=0,
            )

        component_start = time.perf_counter()
        ordinary_values, ordinary_final_y_bars = _objective_vector_vjp_rows(
            _ordinary_objective_vector_from_final_y,
            final_y_for_objective,
        )
        if phase_timing_diagnostics:
            ordinary_values, ordinary_final_y_bars = jax.block_until_ready(
                (ordinary_values, ordinary_final_y_bars)
            )
            final_objective_state_elapsed += time.perf_counter() - component_start
        grouped_objective_values = {
            objective_i: ordinary_values[row_i]
            for row_i, objective_i in enumerate(ordinary_objective_indices)
        }
        grouped_final_y_bars = {
            objective_i: ordinary_final_y_bars[row_i]
            for row_i, objective_i in enumerate(ordinary_objective_indices)
        }
        if combined_geometry_payload:
            final_state_for_geometry = (
                reverse_setup.prepared_rollout.physics_context.unpack_flat(
                    final_y_for_objective
                )
            )
            geometry = support_payload["geometry"]
            geometry_delta0 = _float_delta_tree_like(geometry)

            def _ordinary_objective_vector_from_geometry_delta(geometry_delta):
                runtime_with_geometry = dataclasses.replace(
                    runtime,
                    geometry=_add_float_delta_tree(geometry, geometry_delta),
                )
                return jnp.stack(
                    tuple(
                        dependencies.objective_scalar_by_index(
                            final_state_for_geometry,
                            runtime_with_geometry,
                            objective_i,
                        )
                        for objective_i in ordinary_objective_indices
                    ),
                    axis=0,
                )

            component_start = time.perf_counter()
            _, ordinary_geometry_bars = _objective_vector_vjp_rows(
                _ordinary_objective_vector_from_geometry_delta,
                geometry_delta0,
            )
            if phase_timing_diagnostics:
                ordinary_geometry_bars = jax.block_until_ready(ordinary_geometry_bars)
                final_objective_geometry_elapsed += time.perf_counter() - component_start
            grouped_geometry_bars = {
                objective_i: _take_batched_pytree_row(ordinary_geometry_bars, row_i)
                for row_i, objective_i in enumerate(ordinary_objective_indices)
            }
    for objective_i in range(objective_count):
        objective_name = objective_labels[objective_i]
        if (
            objective_name == bootstrap_objective_name
            and (
                "ntx_support" in support_payload
                or "database" in support_payload
            )
        ):
            component_start = time.perf_counter()
            final_state_for_bootstrap = reverse_setup.prepared_rollout.physics_context.unpack_flat(
                final_y_for_objective
            )
            flux_model = getattr(getattr(runtime, "models", None), "flux", None)
            neoclassical_model = getattr(flux_model, "neoclassical_model", flux_model)
            database_payload = "database" in support_payload
            # A recorded runtime-scan payload owns the already-built radial
            # database.  The static scan model still carries its source
            # ``Monoenergetic`` configuration, which is not a centre-flux
            # interpolation table.  Bind the recorded table before obtaining
            # either bootstrap primal or compact pullback methods.
            if database_payload:
                with_payload = getattr(neoclassical_model, "with_support_payload", None)
                if not callable(with_payload):
                    raise NotImplementedError(
                        "Recorded database bootstrap AD requires a realtime NTX "
                        "model with support-payload binding."
                    )
                neoclassical_model = with_payload(support_payload)
            corrected_fluxes_fn = getattr(neoclassical_model, "evaluate_momentum_corrected_fluxes", None)
            upar_only_fn = getattr(
                neoclassical_model,
                "evaluate_momentum_corrected_upar_only",
                None,
            )
            state_pullback_fn = getattr(neoclassical_model, "pullback_momentum_corrected_upar_state_by_radius", None)
            database_pullback_fn = getattr(
                neoclassical_model,
                "pullback_momentum_corrected_upar_database_by_radius",
                None,
            )
            support_pullback_fn = getattr(
                neoclassical_model,
                "pullback_momentum_corrected_upar_support_by_radius",
                None,
            )
            geometry_pullback_fn = getattr(
                neoclassical_model,
                "pullback_momentum_corrected_upar_geometry_by_radius",
                None,
            )
            joint_pullback_fn = getattr(
                neoclassical_model,
                "pullback_momentum_corrected_upar_state_support_geometry_by_radius",
                None,
            )
            use_upar_only_primal = (
                bootstrap_cotangent_mode == "joint_local_vjp_upar_only"
            )
            if not use_upar_only_primal and not callable(corrected_fluxes_fn):
                raise NotImplementedError(
                    "bootstrap_current_softmax_abs_scaled requires realtime NTX "
                    "evaluate_momentum_corrected_fluxes for compact full-transport AD."
                )
            if use_upar_only_primal and not callable(upar_only_fn):
                raise NotImplementedError(
                    "reverse_bootstrap_cotangent_mode='joint_local_vjp_upar_only' "
                    "requires evaluate_momentum_corrected_upar_only on the realtime NTX model."
                )
            if (
                bootstrap_cotangent_mode == "separate"
                and (
                    not callable(state_pullback_fn)
                    or (
                        database_payload
                        and not callable(database_pullback_fn)
                    )
                    or (not database_payload and not callable(support_pullback_fn))
                )
            ):
                raise NotImplementedError(
                    "bootstrap_current_softmax_abs_scaled requires compact corrected-Upar "
                    "state and support pullbacks on the realtime NTX model."
                )
            corrected_fluxes = (
                {"Upar": upar_only_fn(final_state_for_bootstrap)}
                if use_upar_only_primal
                else corrected_fluxes_fn(final_state_for_bootstrap)
            )
            objective_value, upar_bar = bootstrap_current_softmax_abs_value_and_upar_bar(
                final_state_for_bootstrap,
                runtime,
                corrected_fluxes,
            )
            use_joint_bootstrap_pullback = (
                bootstrap_cotangent_mode
                in {"joint_local_vjp", "joint_local_vjp_upar_only"}
                and combined_geometry_payload
                and not database_payload
            )
            if (
                bootstrap_cotangent_mode
                in {"joint_local_vjp", "joint_local_vjp_upar_only"}
                and not combined_geometry_payload
            ):
                raise NotImplementedError(
                    "joint local bootstrap modes require "
                    "the combined realtime geometry payload."
                )
            if use_joint_bootstrap_pullback:
                if not callable(joint_pullback_fn):
                    raise NotImplementedError(
                    "joint local bootstrap modes require "
                        "the compact joint corrected-Upar pullback on the realtime NTX model."
                    )
                geometry = support_payload["geometry"]
                ntx_support = support_payload["ntx_support"]
                (
                    final_state_bar,
                    support_bar_leaves,
                    geometry_objective_bar,
                ) = joint_pullback_fn(
                    final_state_for_bootstrap,
                    upar_bar,
                    geometry,
                    ntx_support,
                )
            else:
                final_state_bar = state_pullback_fn(final_state_for_bootstrap, upar_bar)
            _, unpack_pullback = jax.vjp(
                reverse_setup.prepared_rollout.physics_context.unpack_flat,
                final_y_for_objective,
            )
            final_y_bar_rows.append(unpack_pullback(final_state_bar)[0])
            objective_values_rows.append(objective_value)
            if combined_geometry_payload:
                if database_payload:
                    d11_bar, d13_bar, d33_bar = database_pullback_fn(
                        final_state_for_bootstrap, upar_bar
                    )
                    database = support_payload["database"]
                    database_bar = dataclasses.replace(
                        _float_delta_tree_like(database),
                        D11_log=d11_bar,
                        D13=d13_bar,
                        D33=d33_bar,
                    )
                    geometry = support_payload["geometry"]
                    if not callable(geometry_pullback_fn):
                        raise NotImplementedError(
                            "Recorded database bootstrap AD requires the compact "
                            "fixed-database corrected-Upar geometry pullback."
                        )
                    geometry_objective_bar = geometry_pullback_fn(
                        final_state_for_bootstrap,
                        upar_bar,
                        geometry,
                    )
                    # The recorded scan payload also contains the direct
                    # ``channels`` and ``surfaces`` branches.  Bootstrap has
                    # no direct contribution to either one at this stage,
                    # but every objective row must retain the identical
                    # support pytree so the subsequent batched segment sweep
                    # and one-time database fold can stack it safely.
                    bootstrap_payload_bar = dict(zero_payload_bar)
                    bootstrap_payload_bar.update(
                        {
                            "geometry": _sanitize_float_delta_bar_tree(
                                geometry, geometry_objective_bar
                            ),
                            "database": _sanitize_float_delta_bar_tree(
                                database, database_bar
                            ),
                        }
                    )
                    objective_payload_bar_rows.append(bootstrap_payload_bar)
                    if phase_timing_diagnostics:
                        objective_payload_bar_rows[-1] = jax.block_until_ready(
                            objective_payload_bar_rows[-1]
                        )
                    continue
                if not use_joint_bootstrap_pullback and not callable(geometry_pullback_fn):
                    raise NotImplementedError(
                        "bootstrap_current_softmax_abs_scaled requires compact corrected-Upar "
                        "geometry pullback for combined realtime geometry payloads."
                    )
                if not use_joint_bootstrap_pullback:
                    geometry = support_payload["geometry"]
                    ntx_support = support_payload["ntx_support"]
                    geometry_objective_bar = geometry_pullback_fn(
                        final_state_for_bootstrap,
                        upar_bar,
                        geometry,
                        ntx_support,
                    )
                    support_bar_leaves = support_pullback_fn(
                        final_state_for_bootstrap,
                        upar_bar,
                        ntx_support,
                    )
                _, ntx_treedef = jax.tree_util.tree_flatten(ntx_support)
                objective_payload_bar_rows.append(
                    {
                        "geometry": _sanitize_float_delta_bar_tree(geometry, geometry_objective_bar),
                        "ntx_support": ntx_treedef.unflatten(tuple(support_bar_leaves)),
                    }
                )
            else:
                support_bar_leaves = support_pullback_fn(
                    final_state_for_bootstrap,
                    upar_bar,
                    support_payload,
                )
                _, support_treedef = jax.tree_util.tree_flatten(support_payload)
                objective_payload_bar_rows.append(
                    support_treedef.unflatten(tuple(support_bar_leaves))
                )
            if phase_timing_diagnostics:
                objective_value, final_y_bar_rows[-1], objective_payload_bar_rows[-1] = (
                    jax.block_until_ready(
                        (
                            objective_value,
                            final_y_bar_rows[-1],
                            objective_payload_bar_rows[-1],
                        )
                    )
                )
                objective_values_rows[-1] = objective_value
                final_objective_bootstrap_elapsed += time.perf_counter() - component_start
            continue

        if final_objective_cotangent_mode == "grouped_vjp":
            objective_value = grouped_objective_values[objective_i]
            final_y_bar = grouped_final_y_bars[objective_i]
        else:
            component_start = time.perf_counter()
            def _objective_from_final_y(final_y_value, objective_index=objective_i):
                final_state = reverse_setup.prepared_rollout.physics_context.unpack_flat(final_y_value)
                return dependencies.objective_scalar_by_index(final_state, runtime, objective_index)

            objective_value, objective_pullback = jax.vjp(
                _objective_from_final_y, final_y_for_objective
            )
            final_y_bar = objective_pullback(jnp.ones_like(objective_value))[0]
            if phase_timing_diagnostics:
                final_y_bar = jax.block_until_ready(final_y_bar)
                final_objective_state_elapsed += time.perf_counter() - component_start
        objective_values_rows.append(objective_value)
        final_y_bar_rows.append(final_y_bar)
        if combined_geometry_payload and (
            "ntx_support" in support_payload or "database" in support_payload
        ):
            if final_objective_cotangent_mode == "grouped_vjp":
                geometry_objective_bar = grouped_geometry_bars[objective_i]
            else:
                component_start = time.perf_counter()
                final_state_for_geometry = reverse_setup.prepared_rollout.physics_context.unpack_flat(
                    final_y_for_objective
                )
                geometry = support_payload["geometry"]
                geometry_delta0 = _float_delta_tree_like(geometry)

                def _objective_from_geometry_delta(geometry_delta, objective_index=objective_i):
                    runtime_with_geometry = dataclasses.replace(
                        runtime,
                        geometry=_add_float_delta_tree(geometry, geometry_delta),
                    )
                    return dependencies.objective_scalar_by_index(
                        final_state_for_geometry,
                        runtime_with_geometry,
                        objective_index,
                    )

                _, geometry_objective_pullback = jax.vjp(
                    _objective_from_geometry_delta, geometry_delta0
                )
                (geometry_objective_bar,) = geometry_objective_pullback(
                    jnp.ones_like(objective_value)
                )
                if phase_timing_diagnostics:
                    geometry_objective_bar = jax.block_until_ready(geometry_objective_bar)
                    final_objective_geometry_elapsed += time.perf_counter() - component_start
            # Ordinary terminal objectives read only the final transport
            # state and runtime geometry.  In particular they do not depend
            # directly on the recorded database, scan channels, or scan
            # surfaces.  Keeping this as a geometry-only VJP is the database
            # analogue of the established exact-NTX boundary: differentiating
            # the whole recorded payload here would trace the large table
            # leaves once per objective and defeat the one-time scan fold.
            objective_payload_bar = dict(zero_payload_bar)
            objective_payload_bar["geometry"] = _sanitize_float_delta_bar_tree(
                geometry, geometry_objective_bar
            )
            objective_payload_bar_rows.append(objective_payload_bar)
        elif combined_geometry_payload:
            if final_objective_cotangent_mode == "grouped_vjp":
                raise NotImplementedError(
                    "ntx_scan_runtime currently requires "
                    "reverse_final_objective_cotangent_mode='scalar'."
                )
            component_start = time.perf_counter()
            final_state_for_support = reverse_setup.prepared_rollout.physics_context.unpack_flat(
                final_y_for_objective
            )
            support_delta0 = _float_delta_tree_like(support_payload)

            def _objective_from_support_delta(support_delta, objective_index=objective_i):
                runtime_with_support = (
                    dependencies.runtime_with_realtime_geometry_reverse_support_payload(
                        runtime,
                        _add_float_delta_tree(support_payload, support_delta),
                    )
                )
                return dependencies.objective_scalar_by_index(
                    final_state_for_support,
                    runtime_with_support,
                    objective_index,
                )

            _, support_objective_pullback = jax.vjp(
                _objective_from_support_delta, support_delta0
            )
            (support_objective_bar,) = support_objective_pullback(
                jnp.ones_like(objective_value)
            )
            if phase_timing_diagnostics:
                support_objective_bar = jax.block_until_ready(support_objective_bar)
                final_objective_geometry_elapsed += time.perf_counter() - component_start
            objective_payload_bar_rows.append(
                _sanitize_float_delta_bar_tree(
                    support_payload, support_objective_bar
                )
            )
        else:
            objective_payload_bar_rows.append(zero_payload_bar)
    objective_values = jnp.stack(objective_values_rows, axis=0)
    final_y_bars = jnp.stack(final_y_bar_rows, axis=0)

    cotangent_mode = str(
        getattr(reverse_setup.execution_context.physics_context, "reverse_stage_cotangent_mode", "full")
    ).strip().lower()
    segment_count = int(jax.tree_util.tree_leaves(segmented_replay_arrays)[0].shape[0])

    reduced_bars = _reverse_reduced_cotangent(
        reverse_setup.execution_context,
        y=final_y_bars,
        lagged_response_cache=_batched_zero_tangent_tree_like(
            segmented_final_carry.lagged_response_cache,
            objective_count,
        ),
        lagged_reference_y=jnp.zeros(
            (objective_count,) + jnp.shape(segmented_final_carry.lagged_reference_y),
            dtype=jnp.asarray(segmented_final_carry.lagged_reference_y).dtype,
        ),
    )
    support_path_leaves, support_treedef = jax.tree_util.tree_flatten_with_path(
        zero_payload_bar
    )
    _zero_support_leaves = tuple(leaf for _path, leaf in support_path_leaves)
    support_leaf_labels = tuple(
        _pytree_path_label(path) for path, _leaf in support_path_leaves
    )
    native_vmec_mode = (
        str(
            getattr(
                reverse_setup.execution_context.physics_context,
                "reverse_rebuild_support_pullback_mode",
                "separate",
            )
        ).strip().lower()
        in _NATIVE_VMEC_REBUILD_SUPPORT_MODES
    )
    native_vmec_zero_bars = (
        _radau_zero_native_vmec_face_coefficient_bars(support_payload)
        if native_vmec_mode
        else None
    )
    if native_vmec_zero_bars is None:
        native_vmec_treedef = None
        native_vmec_zero_leaves = tuple()
    else:
        native_vmec_zero_leaves, native_vmec_treedef = jax.tree_util.tree_flatten(
            native_vmec_zero_bars
        )
    objective_payload_bar_leaves = tuple(
        jax.tree_util.tree_leaves(payload_bar)
        for payload_bar in objective_payload_bar_rows
    )
    expected_support_leaf_count = len(_zero_support_leaves)
    normalized_objective_payload_bar_leaves = []
    for objective_i, leaves in enumerate(objective_payload_bar_leaves):
        objective_name = objective_labels[objective_i]
        if len(leaves) != expected_support_leaf_count:
            if _reverse_tree_debug_enabled():
                print(
                    "[autodiff-gate] support-payload-bar-structure mismatch "
                    f"objective={objective_name} leaf_count={len(leaves)} "
                    f"expected_leaf_count={expected_support_leaf_count}",
                    flush=True,
                )
            raise ValueError(
                "Objective support-payload bar leaf-count mismatch for "
                f"{objective_name}: got {len(leaves)}, expected {expected_support_leaf_count}."
            )
        normalized_leaves = []
        for leaf_i, (expected_leaf, leaf) in enumerate(zip(_zero_support_leaves, leaves, strict=True)):
            expected_arr = jnp.asarray(expected_leaf)
            leaf_arr = jnp.asarray(leaf)
            if leaf_arr.dtype == jax.dtypes.float0 or leaf_arr.shape == ():
                normalized_leaves.append(jnp.zeros_like(expected_arr))
                continue
            if leaf_arr.shape != expected_arr.shape:
                if _reverse_tree_debug_enabled():
                    print(
                        "[autodiff-gate] support-payload-bar-structure mismatch "
                        f"objective={objective_name} leaf={leaf_i} "
                        f"shape={leaf_arr.shape} expected_shape={expected_arr.shape} "
                        f"dtype={leaf_arr.dtype} expected_dtype={expected_arr.dtype}",
                        flush=True,
                    )
                raise ValueError(
                    "Objective support-payload bar leaf-shape mismatch for "
                    f"{objective_name} leaf {leaf_i}: got {leaf_arr.shape}, "
                    f"expected {expected_arr.shape}."
                )
            normalized_leaves.append(jnp.asarray(leaf_arr, dtype=expected_arr.dtype))
        normalized_objective_payload_bar_leaves.append(tuple(normalized_leaves))
    objective_payload_bar_leaves = tuple(normalized_objective_payload_bar_leaves)
    support_bar_leaves = tuple(
        jnp.stack(
            [
                jnp.asarray(objective_payload_bar_leaves[objective_i][leaf_i])
                for objective_i in range(objective_count)
            ],
            axis=0,
        )
        for leaf_i in range(len(_zero_support_leaves))
    )
    if native_vmec_zero_leaves:
        support_bar_leaves = (
            *support_bar_leaves,
            *tuple(
                jnp.broadcast_to(
                    jnp.asarray(leaf)[None, ...],
                    (objective_count,) + jnp.asarray(leaf).shape,
                )
                for leaf in native_vmec_zero_leaves
            ),
        )
    objective_values, final_y_bars, support_bar_leaves = jax.block_until_ready(
        (objective_values, final_y_bars, support_bar_leaves)
    )
    print(
        f"{progress_prefix} progress: support reverse final-objective cotangents ready "
        f"elapsed_s={time.perf_counter() - phase_start:.3f} "
        f"ordinary_mode={final_objective_cotangent_mode} "
        f"bootstrap_mode={bootstrap_cotangent_mode}",
        flush=True,
    )
    if phase_timing_diagnostics:
        print(
            f"{progress_prefix} diagnostic: final-objective cotangent components "
            f"state_vjp_s={final_objective_state_elapsed:.3f} "
            f"geometry_vjp_s={final_objective_geometry_elapsed:.3f} "
            f"bootstrap_compact_s={final_objective_bootstrap_elapsed:.3f} "
            f"assembly_and_sync_s={time.perf_counter() - phase_start - final_objective_state_elapsed - final_objective_geometry_elapsed - final_objective_bootstrap_elapsed:.3f} "
            f"(synchronized existing work; no duplicate objective VJPs)",
            flush=True,
        )
    objective_support_bar_leaves = support_bar_leaves
    step_support_bar_leaves_accum = tuple(jnp.zeros_like(leaf) for leaf in support_bar_leaves)
    initial_cache_support_bar_leaves_accum = tuple(jnp.zeros_like(leaf) for leaf in support_bar_leaves)
    support_reuse_count = 0
    support_rebuild_count = 0
    segment_jit_diagnostics = bool(
        getattr(
            reverse_setup.execution_context.physics_context,
            "reverse_segment_jit_diagnostics",
            False,
        )
    )
    segment_input_diagnostics = bool(
        getattr(
            reverse_setup.execution_context.physics_context,
            "reverse_segment_input_diagnostics",
            False,
        )
    )
    rebuild_component_timing = bool(
        getattr(
            reverse_setup.execution_context.physics_context,
            "reverse_rebuild_component_timing",
            False,
        )
    )
    # This diagnostic deliberately uses the existing first real segment as
    # the compile-plus-execute measurement, then repeats that *same* segment
    # once with its original input to expose its warm device time.  The repeat
    # is discarded; it is intentionally opt-in because it costs one segment.
    phase_timing_segment_warm_pending = phase_timing_diagnostics
    if rebuild_component_timing:
        physics_context = reverse_setup.execution_context.physics_context
        rebuild_mode = str(
            getattr(physics_context, "reverse_rebuild_support_pullback_mode", "separate")
        ).strip().lower()
        separate_rebuild_modes = {
            "separate",
            "separate_reuse_local_vjp_primal",
            "separate_reuse_local_vjp_primal_geometry_only_prepared",
            "separate_reuse_local_vjp_primal_geometry_implicit_ntx_two_directional",
        }
        native_batched_rebuild_modes = {
            "ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal",
            "ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients",
            "ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_directional_product_rule",
        }
        if rebuild_mode not in separate_rebuild_modes | native_batched_rebuild_modes:
            raise ValueError(
                "reverse_rebuild_component_timing currently requires a standard "
                "separate rebuild-support mode or the native multi-RHS grouped mode."
            )
        if (
            rebuild_mode in separate_rebuild_modes
            and physics_context.flat_rhs_build_support_pullback is None
        ):
            raise RuntimeError(
                "reverse_rebuild_component_timing requires the standard rebuild-support pullback hook."
            )
        native_batched_support_pullback = None
        if rebuild_mode == (
            "ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal"
        ):
            native_batched_support_pullback = (
                physics_context.flat_rhs_build_support_pullback_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal
            )
        elif rebuild_mode == (
            "ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients"
        ):
            native_batched_support_pullback = (
                physics_context.flat_rhs_build_support_pullback_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients
            )
        elif rebuild_mode == (
            "ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_directional_product_rule"
        ):
            native_batched_support_pullback = (
                physics_context.flat_rhs_build_support_pullback_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_directional_product_rule
            )
        elif rebuild_mode == (
            "ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_coefficient_pullback"
        ):
            native_batched_support_pullback = (
                physics_context.flat_rhs_build_support_pullback_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_coefficient_pullback
            )
        if rebuild_mode in native_batched_rebuild_modes and native_batched_support_pullback is None:
            raise RuntimeError(
                "reverse_rebuild_component_timing requires the selected native "
                "multi-RHS rebuild-support pullback hook."
            )
        diagnostic_segment_index = segment_count - 1
        diagnostic_carry = _take_tree_axis0(segment_start_carries, diagnostic_segment_index)
        diagnostic_cache_bars = _batched_zero_tangent_tree_like(
            diagnostic_carry.lagged_response_cache,
            objective_count,
        )
        diagnostic_cache_valid = bool(
            np.asarray(jax.device_get(diagnostic_carry.lagged_response_valid))
        )
        if diagnostic_cache_valid:
            raise RuntimeError(
                "reverse_rebuild_component_timing selected a reuse carry; "
                "the final segment must start on a rebuild carry."
            )

        def _state_transpose_batched(flat_y, cache_bars):
            rebuild_state = physics_context.unpack_flat(
                _project_flat_state_if_needed(flat_y, physics_context.project_flat)
            )
            return jax.vmap(
                lambda cache_bar: physics_context.pullback_build_lagged_response(
                    rebuild_state,
                    cache_bar,
                    reverse_stage_cotangent_mode=cotangent_mode,
                    reverse_segment_profile_annotations=False,
                )
            )(cache_bars)

        # ``support_payload`` contains NTX geometry metadata with static leaves
        # (for example Fourier-mode arrays).  Passing it as a JIT argument makes
        # JAX replace those leaves with sentinels during tracing.  The production
        # segment path captures this payload in its closure, so do the same here.
        def _support_transpose_batched(
            flat_y,
            cache_bars,
            *,
            inner_timing_component: str = "full",
        ):
            if rebuild_mode in native_batched_rebuild_modes:
                if inner_timing_component != "full":
                    raise ValueError(
                        "native multi-RHS component timing exposes only the exact "
                        "combined support transpose."
                    )
                # This is the production native branch verbatim: the selected
                # NTX helper already receives the complete objective RHS batch.
                # Do not wrap it in another vmap or sanitize it as a scalar tree.
                return tuple(
                    jax.tree_util.tree_leaves(
                        native_batched_support_pullback(
                            flat_y,
                            cache_bars,
                            support_payload,
                        )
                    )
                )
            # Match the production `separate` branch exactly: flatten the
            # sanitized support cotangent *inside* vmap.  Returning the raw
            # support tree would ask vmap to batch NTX static metadata such as
            # Fourier-mode labels, which is not an operation the real segment
            # ever performs.
            def _one_support_transpose(cache_bar):
                support_bar = physics_context.flat_rhs_build_support_pullback(
                    flat_y,
                    cache_bar,
                    support_payload,
                    reverse_segment_profile_annotations_override=False,
                    reuse_local_vjp_primal_anchor_response=(
                        rebuild_mode
                        in {
                            "separate_reuse_local_vjp_primal",
                            "separate_reuse_local_vjp_primal_geometry_only_prepared",
                            "separate_reuse_local_vjp_primal_geometry_implicit_ntx_two_directional",
                        }
                    ),
                    geometry_only_prepared_pullback=(
                        rebuild_mode
                        == "separate_reuse_local_vjp_primal_geometry_only_prepared"
                    ),
                    geometry_implicit_ntx_two_directional_pullback=(
                        rebuild_mode
                        == "separate_reuse_local_vjp_primal_geometry_implicit_ntx_two_directional"
                    ),
                    reverse_rebuild_inner_timing_component=inner_timing_component,
                )
                return tuple(
                    jax.tree_util.tree_leaves(
                        _radau_sanitize_support_delta_bar_tree(
                            support_payload,
                            support_bar,
                        )
                    )
                )

            return jax.vmap(
                _one_support_transpose
            )(cache_bars)

        def _time_rebuild_component(name, compiled_fn, *component_args):
            compile_start = time.perf_counter()
            component_result = compiled_fn(*component_args)
            component_result = jax.block_until_ready(component_result)
            print(
                f"{progress_prefix} diagnostic: reverse rebuild component "
                f"segment={diagnostic_segment_index + 1}/{segment_count} "
                f"name={name} compile_plus_execute_s={time.perf_counter() - compile_start:.3f}",
                flush=True,
            )
            execution_start = time.perf_counter()
            component_result = compiled_fn(*component_args)
            jax.block_until_ready(component_result)
            print(
                f"{progress_prefix} diagnostic: reverse rebuild component "
                f"segment={diagnostic_segment_index + 1}/{segment_count} "
                f"name={name} warm_execute_s={time.perf_counter() - execution_start:.3f}",
                flush=True,
            )
            return component_result

        # This is the exact final realised segment with only the *rebuild*
        # state/support transposes disabled.  It retains segment replay, the
        # block stage-adjoint solve, fixed-lagged RHS state/support transposes,
        # and all objective batching.  It therefore measures the portion that
        # cannot be inferred from the isolated NTX support timer above.
        #
        # The alternate immutable execution context is diagnostic-only and is
        # captured as a static JIT argument.  It never reaches the production
        # reverse sweep below.
        diagnostic_without_rebuild_context = dataclasses.replace(
            reverse_setup.execution_context,
            physics_context=dataclasses.replace(
                physics_context,
                reverse_stage_cotangent_mode="zero_rebuild_pullback",
            ),
        )
        diagnostic_segment_arrays = _take_tree_axis0(
            segmented_replay_arrays,
            diagnostic_segment_index,
        )

        diagnostic_record_mode = str(
            getattr(physics_context, "reverse_segment_primal_record_mode", "reconstruct")
        ).strip().lower()

        def _segment_reverse_without_rebuild_transposes(
            segment_bars,
            segment_carry,
            segment_arrays,
            support_value,
        ):
            return _radau_segment_reduced_cotangent_bwd_batched_with_support_call(
                diagnostic_without_rebuild_context,
                cotangent_mode,
                segment_bars,
                segment_carry,
                segment_arrays,
                support_value,
            )

        print(
            f"{progress_prefix} progress: timing one isolated standard rebuild "
            f"segment={diagnostic_segment_index + 1}/{segment_count} objectives={objective_count} "
            f"production_record_mode={diagnostic_record_mode} "
            "(diagnostic extra work; values are not added to the reverse result)",
            flush=True,
        )
        print(
            f"{progress_prefix} diagnostic: reverse rebuild timing mode={rebuild_mode} "
            "component modules are measured separately and are not additive",
            flush=True,
        )
        if diagnostic_record_mode == "reuse_segment_primal_record":
            # The production segment kernel deliberately fuses these two scans.
            # Separate them only here so their exact record-mode shapes can be
            # inspected. They are not additive production timings.
            record_replay_result = _time_rebuild_component(
                "record_replay_minimal_with_primal_records",
                lambda segment_carry, segment_arrays: _radau_segment_replay_minimal_with_primal_records_call(
                    reverse_setup.execution_context,
                    segment_carry,
                    segment_arrays,
                ),
                diagnostic_carry,
                diagnostic_segment_arrays,
            )
            _, diagnostic_step_start_carries, diagnostic_step_primal_records = record_replay_result
            record_payload_bytes = _logical_tree_nbytes(diagnostic_step_primal_records)
            record_slot_count = int(jax.tree_util.tree_leaves(diagnostic_segment_arrays)[0].shape[0])
            print(
                f"{progress_prefix} diagnostic: reverse segment primal record "
                f"segment={diagnostic_segment_index + 1}/{segment_count} "
                f"logical_payload_bytes={record_payload_bytes} "
                f"logical_payload_mib={record_payload_bytes / (1024 ** 2):.3f} "
                f"slots={record_slot_count} "
                f"logical_bytes_per_slot={record_payload_bytes / max(record_slot_count, 1):.1f} "
                "(logical array payload only; this is not a peak-memory measurement)",
                flush=True,
            )
            _time_rebuild_component(
                "record_consuming_reverse_without_rebuild_transposes",
                lambda segment_bars, step_start_carries, step_primal_records, segment_arrays, support_value:
                _radau_segment_reduced_cotangent_bwd_batched_with_support_from_primal_records_call(
                    diagnostic_without_rebuild_context,
                    cotangent_mode,
                    segment_bars,
                    step_start_carries,
                    step_primal_records,
                    segment_arrays,
                    support_value,
                ),
                reduced_bars,
                diagnostic_step_start_carries,
                diagnostic_step_primal_records,
                diagnostic_segment_arrays,
                support_payload,
            )
        else:
            print(
                f"{progress_prefix} diagnostic: record-specific timings skipped "
                "because production_record_mode is not reuse_segment_primal_record",
                flush=True,
            )
        _time_rebuild_component(
            "segment_reverse_without_rebuild_transposes",
            _segment_reverse_without_rebuild_transposes,
            reduced_bars,
            diagnostic_carry,
            diagnostic_segment_arrays,
            support_payload,
        )
        _time_rebuild_component(
            "state_transpose",
            jax.jit(_state_transpose_batched),
            diagnostic_carry.y,
            diagnostic_cache_bars,
        )
        _time_rebuild_component(
            (
                "native_multi_rhs_support_transpose"
                if rebuild_mode in native_batched_rebuild_modes
                else "support_transpose"
            ),
            jax.jit(_support_transpose_batched),
            diagnostic_carry.y,
            diagnostic_cache_bars,
        )
        if rebuild_mode in {
            "separate_reuse_local_vjp_primal",
            "separate_reuse_local_vjp_primal_geometry_only_prepared",
            "separate_reuse_local_vjp_primal_geometry_implicit_ntx_two_directional",
        }:
            # These two launches deliberately introduce diagnostic-only XLA
            # boundaries.  They use the exact production shapes and device
            # operations, but are not part of the normal fused reverse path.
            # No XProf/CUPTI trace, host callback, or retained reverse payload
            # is involved.
            _time_rebuild_component(
                "anchor_value_transpose_local_ntx_vjp_and_accumulation",
                jax.jit(
                    lambda flat_y, cache_bars: _support_transpose_batched(
                        flat_y,
                        cache_bars,
                        inner_timing_component="local_ntx_vjp_and_accumulation",
                    )
                ),
                diagnostic_carry.y,
                diagnostic_cache_bars,
            )
            _time_rebuild_component(
                "local_ntx_vjp_primal_only",
                jax.jit(
                    lambda flat_y, cache_bars: _support_transpose_batched(
                        flat_y,
                        cache_bars,
                        inner_timing_component="local_ntx_vjp_primal",
                    )
                ),
                diagnostic_carry.y,
                diagnostic_cache_bars,
            )
            # Partition the existing local VJP transpose by its three
            # transport response fields.  Each launch retains the production
            # primal/VJP construction and anchor accumulation, but zeros the
            # other response cotangents on device.  This is diagnostic-only:
            # it deliberately does not alter the full combined pullback used
            # by the reverse sweep.
            for component_name, inner_component in (
                ("local_ntx_vjp_transport_only", "local_ntx_vjp_transport_only"),
                ("local_ntx_vjp_d_er_only", "local_ntx_vjp_d_er_only"),
                ("local_ntx_vjp_d_log_nu_star_only", "local_ntx_vjp_d_log_nu_star_only"),
            ):
                _time_rebuild_component(
                    component_name,
                    jax.jit(
                        lambda flat_y, cache_bars, inner_component=inner_component: _support_transpose_batched(
                            flat_y,
                            cache_bars,
                            inner_timing_component=inner_component,
                        )
                    ),
                    diagnostic_carry.y,
                    diagnostic_cache_bars,
                )
            _time_rebuild_component(
                "coordinate_rho_transpose_only",
                jax.jit(
                    lambda flat_y, cache_bars: _support_transpose_batched(
                        flat_y,
                        cache_bars,
                        inner_timing_component="coordinate_rho_transpose",
                    )
                ),
                diagnostic_carry.y,
                diagnostic_cache_bars,
            )
    print(
        f"{progress_prefix} progress: support reverse segmented cotangent sweep start "
        f"segments={segment_count} segment_length="
        f"{int(jax.tree_util.tree_leaves(segmented_replay_arrays)[0].shape[1])} "
        f"objectives={objective_count} cotangent_mode={cotangent_mode}",
        flush=True,
    )

    def _batched_reduced_first_nonfinite_rows(value):
        """Return one nonfinite reduced-cotangent leaf label per objective row.

        This is host-only diagnostic metadata.  Leaves without the leading
        objective axis are structural carry fields and are intentionally
        ignored.
        """
        path_leaves, _ = jax.tree_util.tree_flatten_with_path(value)
        row_finite = []
        labels = []
        for path, leaf in path_leaves:
            array = jnp.asarray(leaf)
            if (
                array.dtype == jax.dtypes.float0
                or not jnp.issubdtype(array.dtype, jnp.inexact)
                or array.ndim < 1
                or int(array.shape[0]) != int(objective_count)
            ):
                continue
            axes = tuple(range(1, array.ndim))
            row_finite.append(
                jnp.isfinite(array)
                if not axes
                else jnp.all(jnp.isfinite(array), axis=axes)
            )
            labels.append(_pytree_path_label(path))
        if not row_finite:
            return tuple(None for _ in range(objective_count))
        finite_rows = tuple(
            np.asarray(row, dtype=bool) for row in jax.device_get(tuple(row_finite))
        )
        return tuple(
            next(
                (label for label, finite in zip(labels, finite_rows, strict=True) if not finite[row_i]),
                None,
            )
            for row_i in range(objective_count)
        )

    def _run_host_static_branch_segment(
        segment_carry,
        segment_arrays,
        segment_next_reduced_bars,
        *,
        capture_actual_cotangent_diagnostics: bool = False,
    ):
        """Run one realized segment through two static device executables.

        Only the already-realized boolean schedule crosses to the host.  The
        carries, primal records, objective-RHS bars, support bars, and every
        NTX calculation remain device arrays.  This deliberately removes the
        dynamic reuse/rebuild ``lax.cond`` and its enclosing reverse scan from
        the compiled reverse module rather than adding another inner call.
        """

        _, step_start_carries, step_primal_records = (
            _radau_segment_replay_minimal_with_primal_records_call(
                reverse_setup.execution_context,
                segment_carry,
                segment_arrays,
            )
        )
        slot_active = np.asarray(jax.device_get(segment_arrays[0]), dtype=bool).reshape(-1)
        slot_next_lagged_valid = np.asarray(
            jax.device_get(segment_arrays[6]), dtype=bool
        ).reshape(-1)
        segment_support_bars = tuple(
            jnp.zeros_like(leaf) for leaf in support_bar_leaves
        )
        def _run_static_step(
            slot_index,
            branch,
            step_start_carry,
            step_primal_record,
            reduced_value,
        ):
            carry_for_step = _radau_carry_with_forward_only_jvp_fields(
                dataclasses.replace(step_start_carry, dt=segment_arrays[1][slot_index])
            )
            return (
                _execute_radau_accepted_step_next_reduced_cotangent_batched_bwd_with_support_from_segment_primal_record_call(
                    reverse_setup.execution_context.kernel_context,
                    reverse_setup.execution_context.physics_context,
                    reverse_setup.execution_context.attempt_context,
                    branch,
                    carry_for_step,
                    step_primal_record,
                    reduced_value,
                    support_payload,
                )
            )
        if not capture_actual_cotangent_diagnostics:
            return jax.block_until_ready(
                _run_realized_reverse_slot_dispatch(
                    slot_active=slot_active,
                    slot_next_lagged_valid=slot_next_lagged_valid,
                    segment_start_lagged_valid=bool(
                        np.asarray(jax.device_get(segment_carry.lagged_response_valid))
                    ),
                    step_start_carries=step_start_carries,
                    step_primal_records=step_primal_records,
                    next_reduced_bars=segment_next_reduced_bars,
                    initial_support_bars=segment_support_bars,
                    take_axis0=_take_tree_axis0,
                    step_fn=_run_static_step,
                )
            )

        # A nonfinite segment is replayed only in this opt-in host diagnostic.
        # It uses the production step executable and the exact realized branch
        # order, but transfers only finite/nonfinite metadata after each step.
        reduced_value = segment_next_reduced_bars
        support_value = segment_support_bars
        rows = []
        for slot_index, branch in _realized_reverse_slot_branches(
            slot_active,
            slot_next_lagged_valid,
            bool(np.asarray(jax.device_get(segment_carry.lagged_response_valid))),
        ):
            reduced_input_bad = _batched_reduced_first_nonfinite_rows(reduced_value)
            reduced_value, step_support_value = _run_static_step(
                slot_index,
                branch,
                _take_tree_axis0(step_start_carries, slot_index),
                _take_tree_axis0(step_primal_records, slot_index),
                reduced_value,
            )
            reduced_value, step_support_value = jax.block_until_ready(
                (reduced_value, step_support_value)
            )
            support_value = tuple(
                accumulated + increment
                for accumulated, increment in zip(
                    support_value, step_support_value, strict=True
                )
            )
            rows.append(
                {
                    "slot_index": int(slot_index),
                    "branch": str(branch),
                    "input_reduced_bad": reduced_input_bad,
                    "output_reduced_bad": _batched_reduced_first_nonfinite_rows(reduced_value),
                    "step_support_bad": _batched_support_first_nonfinite_leaves(
                        step_support_value[: len(_zero_support_leaves)],
                        support_leaf_labels,
                        objective_count,
                    ),
                    "cumulative_support_bad": _batched_support_first_nonfinite_leaves(
                        support_value[: len(_zero_support_leaves)],
                        support_leaf_labels,
                        objective_count,
                    ),
                }
            )
        return reduced_value, support_value, tuple(rows)

    phase_start = time.perf_counter()
    actual_cotangent_nonfinite_segment_diagnosed = False
    for segment_index in range(segment_count - 1, -1, -1):
        segment_phase_start = time.perf_counter()
        if segment_jit_diagnostics:
            cache_before = (
                _jax_trace_cache_size(_radau_segment_reduced_cotangent_bwd_batched_with_support_call),
                _jax_trace_cache_size(
                    _execute_radau_accepted_step_next_reduced_cotangent_batched_bwd_with_support_call
                ),
            )
        segment_start_carry = _take_tree_axis0(segment_start_carries, segment_index)
        segment_arrays = _take_tree_axis0(segmented_replay_arrays, segment_index)
        segment_reduced_bars_input = reduced_bars
        if host_static_branch_dispatch:
            reduced_bars, segment_support_bar_leaves = _run_host_static_branch_segment(
                segment_start_carry,
                segment_arrays,
                reduced_bars,
            )
        else:
            reduced_bars, segment_support_bar_leaves = (
                _reverse_segment_reduced_cotangent_bwd_batched_with_support_call(
                    reverse_setup.execution_context,
                    cotangent_mode,
                    reduced_bars,
                    segment_start_carry,
                    segment_arrays,
                    support_payload,
                )
            )
            reduced_bars, segment_support_bar_leaves = jax.block_until_ready(
                (reduced_bars, segment_support_bar_leaves)
            )
        segment_bad_rows = None
        if segment_input_diagnostics:
            segment_bad_rows = _batched_support_first_nonfinite_leaves(
                segment_support_bar_leaves[: len(_zero_support_leaves)],
                support_leaf_labels,
                objective_count,
            )
            for objective_i, first_bad in enumerate(segment_bad_rows):
                if first_bad is None:
                    continue
                leaf_i, leaf_label = first_bad
                print(
                    f"{progress_prefix} diagnostic: support reverse segment "
                    f"{segment_index + 1}/{segment_count} first_nonfinite "
                    f"objective={objective_labels[objective_i]} "
                    f"leaf={leaf_i}:{leaf_label}",
                    flush=True,
                )
            if (
                not actual_cotangent_nonfinite_segment_diagnosed
                and any(row is not None for row in segment_bad_rows)
            ):
                actual_cotangent_nonfinite_segment_diagnosed = True
                (
                    diagnostic_reduced_bars,
                    diagnostic_support_bar_leaves,
                    diagnostic_step_rows,
                ) = _run_host_static_branch_segment(
                    segment_start_carry,
                    segment_arrays,
                    segment_reduced_bars_input,
                    capture_actual_cotangent_diagnostics=True,
                )
                diagnostic_reduced_bars, diagnostic_support_bar_leaves = (
                    jax.block_until_ready(
                        (diagnostic_reduced_bars, diagnostic_support_bar_leaves)
                    )
                )
                for row in diagnostic_step_rows:
                    for objective_i, objective_name in enumerate(objective_labels):
                        input_bad = row["input_reduced_bad"][objective_i]
                        output_bad = row["output_reduced_bad"][objective_i]
                        step_bad = row["step_support_bad"][objective_i]
                        cumulative_bad = row["cumulative_support_bad"][objective_i]
                        if (
                            input_bad is None
                            and output_bad is None
                            and step_bad is None
                            and cumulative_bad is None
                        ):
                            continue
                        step_label = (
                            None
                            if step_bad is None
                            else f"{step_bad[0]}:{step_bad[1]}"
                        )
                        cumulative_label = (
                            None
                            if cumulative_bad is None
                            else f"{cumulative_bad[0]}:{cumulative_bad[1]}"
                        )
                        print(
                            f"{progress_prefix} diagnostic: support reverse segment "
                            f"{segment_index + 1}/{segment_count} actual-cotangent "
                            f"slot={row['slot_index']} branch={row['branch']} "
                            f"objective={objective_name} "
                            f"input_reduced_bad={input_bad} "
                            f"output_reduced_bad={output_bad} "
                            f"step_support_bad={step_label} "
                            f"cumulative_support_bad={cumulative_label}",
                            flush=True,
                        )
        if phase_timing_segment_warm_pending and not host_static_branch_dispatch:
            first_call_elapsed = time.perf_counter() - segment_phase_start
            warm_start = time.perf_counter()
            warm_reduced_bars, warm_support_bar_leaves = (
                _reverse_segment_reduced_cotangent_bwd_batched_with_support_call(
                    reverse_setup.execution_context,
                    cotangent_mode,
                    segment_reduced_bars_input,
                    segment_start_carry,
                    segment_arrays,
                    support_payload,
                )
            )
            # Keep no reference to the diagnostic output after synchronization.
            jax.block_until_ready((warm_reduced_bars, warm_support_bar_leaves))
            print(
                f"{progress_prefix} diagnostic: reverse segment "
                f"{segment_index + 1}/{segment_count} first_call_compile_plus_execute_s={first_call_elapsed:.3f} "
                f"warm_execute_s={time.perf_counter() - warm_start:.3f} "
                "(one discarded duplicate segment; no gradient/result change)",
                flush=True,
            )
            phase_timing_segment_warm_pending = False
        if segment_jit_diagnostics:
            cache_after = (
                _jax_trace_cache_size(_radau_segment_reduced_cotangent_bwd_batched_with_support_call),
                _jax_trace_cache_size(
                    _execute_radau_accepted_step_next_reduced_cotangent_batched_bwd_with_support_call
                ),
            )
        support_bar_leaves = tuple(
            accumulated + increment
            for accumulated, increment in zip(support_bar_leaves, segment_support_bar_leaves)
        )
        step_support_bar_leaves_accum = tuple(
            accumulated + increment
            for accumulated, increment in zip(step_support_bar_leaves_accum, segment_support_bar_leaves)
        )
        segment_active = np.asarray(jax.device_get(segment_arrays[0]), dtype=bool).reshape(-1)
        # ``segment_arrays[6]`` is the validity *after* each slot.  The
        # reverse branch instead selects reuse/rebuild from the validity in
        # the carry entering that slot.  Shift the trace and prepend the
        # segment-start carry so the printed counts describe the branch that
        # actually ran.
        segment_next_lagged_valid = np.asarray(
            jax.device_get(segment_arrays[6]), dtype=bool
        ).reshape(-1)
        segment_start_lagged_valid = np.concatenate(
            [
                np.asarray(
                    [bool(np.asarray(jax.device_get(segment_start_carry.lagged_response_valid)))],
                    dtype=bool,
                ),
                segment_next_lagged_valid[:-1],
            ]
        )
        active_start_lagged_valid = segment_start_lagged_valid[segment_active]
        segment_support_reuse_count = int(np.count_nonzero(active_start_lagged_valid))
        segment_support_rebuild_count = int(
            active_start_lagged_valid.size - segment_support_reuse_count
        )
        support_reuse_count += segment_support_reuse_count
        support_rebuild_count += segment_support_rebuild_count
        print(
            f"{progress_prefix} progress: support reverse segment "
            f"{segment_index + 1}/{segment_count} ready "
            f"elapsed_s={time.perf_counter() - segment_phase_start:.3f} "
            f"active_steps={int(np.count_nonzero(segment_active))} "
            f"support_reuse={segment_support_reuse_count} "
            f"support_rebuild={segment_support_rebuild_count}",
            flush=True,
        )
        if segment_input_diagnostics:
            # The segment output was block_until_ready above. These reads are
            # therefore diagnostic metadata transfers, not a synchronization
            # inserted into the device reverse calculation.
            segment_dt_np = np.asarray(
                jax.device_get(segment_arrays[1]),
                dtype=float,
            ).reshape(-1)
            active_dt_np = segment_dt_np[segment_active]
            active_pattern = "".join("1" if flag else "0" for flag in segment_active)
            lagged_pattern = "".join(
                "R" if valid else "B"
                for active, valid in zip(segment_active, segment_start_lagged_valid, strict=True)
                if active
            )
            incoming_cache_valid = bool(
                np.asarray(jax.device_get(segment_start_carry.cache_valid))
            )
            incoming_cache_age = int(
                np.asarray(jax.device_get(segment_start_carry.cache_age))
            )
            dt_min = float(np.min(active_dt_np)) if active_dt_np.size else float("nan")
            dt_max = float(np.max(active_dt_np)) if active_dt_np.size else float("nan")
            print(
                f"{progress_prefix} diagnostic: support reverse segment "
                f"{segment_index + 1}/{segment_count} inputs "
                f"active_pattern={active_pattern} lagged_pattern={lagged_pattern or '-'} "
                f"dt_min={dt_min:.6e} dt_max={dt_max:.6e} "
                f"incoming_cache_valid={incoming_cache_valid} "
                f"incoming_cache_age={incoming_cache_age} "
                "(host diagnostic after device completion)",
                flush=True,
            )
        if segment_jit_diagnostics:
            print(
                f"{progress_prefix} diagnostic: support reverse segment "
                f"{segment_index + 1}/{segment_count} jax_trace_cache "
                f"outer={cache_before[0]}->{cache_after[0]} "
                f"step_call={cache_before[1]}->{cache_after[1]} "
                "(host diagnostic; not an XLA persistent-cache metric)",
                flush=True,
            )
    reduced_bars, support_bar_leaves = jax.block_until_ready((reduced_bars, support_bar_leaves))
    print(
        f"{progress_prefix} progress: support reverse segmented cotangent sweep ready "
        f"elapsed_s={time.perf_counter() - phase_start:.3f} "
        f"support_reuse={support_reuse_count} support_rebuild={support_rebuild_count}",
        flush=True,
    )

    # The normal support-payload contract ends here.  The experimental native
    # VMEC bars were carried in parallel through the segment scan solely so
    # their transpose can be applied below; do not expose them to the legacy
    # initial-cache/root support pullbacks.
    native_vmec_face_coefficient_bars = None
    if native_vmec_treedef is not None:
        native_leaf_start = len(_zero_support_leaves)
        native_vmec_face_coefficient_bars = native_vmec_treedef.unflatten(
            tuple(support_bar_leaves[native_leaf_start:])
        )
        support_bar_leaves = tuple(support_bar_leaves[:native_leaf_start])
        objective_support_bar_leaves = tuple(
            objective_support_bar_leaves[:native_leaf_start]
        )
        step_support_bar_leaves_accum = tuple(
            step_support_bar_leaves_accum[:native_leaf_start]
        )
        initial_cache_support_bar_leaves_accum = tuple(
            initial_cache_support_bar_leaves_accum[:native_leaf_start]
        )

    initial_lagged_response_valid = bool(np.asarray(jax.device_get(carry0.lagged_response_valid)))
    build_support_pullback = reverse_setup.execution_context.physics_context.flat_rhs_build_support_pullback
    if initial_cache_support_pullback_mode not in {
        "scalar",
        "ntx_batched_interpolated_faces",
        "ntx_native_joint_state_and_support",
        "ntx_native_joint_state_and_ntx_support_split_geometry_vmec",
        "ntx_native_joint_state_and_ntx_support_split_geometry_vmec_no_prepared_carry",
        "ntx_native_joint_state_and_ntx_support_split_geometry_vmec_fused_rhs",
        "rebuild_dispatch",
    }:
        raise ValueError(
            "Unknown reverse_initial_cache_support_pullback_mode "
            f"{initial_cache_support_pullback_mode!r}."
        )
    allow_initial_cache_support_pullback = cotangent_mode in {
        "full",
        "full_initial_cache_support_pullback",
        "initial_cache_support_pullback",
    }
    if (
        use_native_joint_initial_carry_pullback
        or use_native_split_joint_initial_carry_pullback
        or use_native_split_joint_no_prepared_carry_initial_carry_pullback
        or use_native_split_joint_fused_rhs_initial_carry_pullback
    ) and not initial_lagged_response_valid:
        raise RuntimeError(
            "ntx_native_joint_state_and_support requires a valid initial "
            "lagged response."
        )
    if use_native_split_joint_initial_carry_pullback and getattr(
        reverse_setup.execution_context.physics_context,
        "flat_rhs_build_state_and_ntx_support_pullback_batched_interpolated_faces_"
        "native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients",
        None,
    ) is None:
        raise RuntimeError(
            "split native initial pullback was requested, but the active transport "
            "physics context does not expose the compact NTX/VMEC hook."
        )
    if use_native_split_joint_no_prepared_carry_initial_carry_pullback and getattr(
        reverse_setup.execution_context.physics_context,
        "flat_rhs_build_state_and_ntx_support_pullback_batched_interpolated_faces_"
        "native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_"
        "no_prepared_carry",
        None,
    ) is None:
        raise RuntimeError(
            "compact split native initial pullback was requested, but the active "
            "transport physics context does not expose the compact NTX/VMEC hook."
        )
    if use_native_split_joint_fused_rhs_initial_carry_pullback and getattr(
        reverse_setup.execution_context.physics_context,
        "flat_rhs_state_and_lagged_response_pullback",
        None,
    ) is None:
        raise RuntimeError(
            "fused split native initial pullback was requested, but the active "
            "transport physics context does not expose the fused fixed-lagged "
            "RHS state/response hook."
        )
    if use_native_joint_initial_carry_pullback and getattr(
        reverse_setup.execution_context.physics_context,
        "flat_rhs_build_state_and_support_pullback_batched_interpolated_faces_"
        "native_multi_rhs_reuse_moment_drds_jvp_shared_primal",
        None,
    ) is None:
        raise RuntimeError(
            "ntx_native_joint_state_and_support was requested, but the active "
            "transport physics context does not expose the native joint "
            "state/support pullback."
        )
    initial_cache_pullback_used = False
    initial_cache_pullback_skipped = False
    initial_cache_support_warm_call = None
    if (
        initial_lagged_response_valid
        and build_support_pullback is not None
        and allow_initial_cache_support_pullback
        and not use_native_joint_initial_carry_pullback
        and not use_native_split_joint_initial_carry_pullback
        and not use_native_split_joint_no_prepared_carry_initial_carry_pullback
        and not use_native_split_joint_fused_rhs_initial_carry_pullback
    ):
        phase_start = time.perf_counter()
        with _reverse_profile_scope(
            reverse_setup, "reverse_post_sweep/initial_cache_support_pullback"
        ):
            initial_native_vmec_bars = None
            if use_rebuild_dispatch_initial_cache_pullback:
                if phase_timing_diagnostics:
                    initial_cache_support_warm_call = lambda: _initial_cache_support_pullback_from_rebuild_dispatch(
                        physics_context=reverse_setup.execution_context.physics_context,
                        flat_y=carry0.y,
                        lagged_response_bars=reduced_bars.lagged_response_cache,
                        support_payload=support_payload,
                    )
                initial_cache_support_bars, initial_native_vmec_bars = (
                    _initial_cache_support_pullback_from_rebuild_dispatch(
                        physics_context=reverse_setup.execution_context.physics_context,
                        flat_y=carry0.y,
                        lagged_response_bars=reduced_bars.lagged_response_cache,
                        support_payload=support_payload,
                    )
                )
            elif initial_cache_support_pullback_mode == "ntx_batched_interpolated_faces":
                batched_pullback = getattr(
                    reverse_setup.execution_context.physics_context,
                    "flat_rhs_build_support_pullback_batched_interpolated_faces",
                    None,
                )
                if batched_pullback is None:
                    raise RuntimeError(
                        "ntx_batched_interpolated_faces was requested, but the active transport "
                        "physics context does not expose the NTX batched support pullback."
                    )
                if phase_timing_diagnostics:
                    initial_cache_support_warm_call = lambda: batched_pullback(
                        carry0.y,
                        reduced_bars.lagged_response_cache,
                        support_payload,
                    )
                initial_cache_support_bars = batched_pullback(
                    carry0.y,
                    reduced_bars.lagged_response_cache,
                    support_payload,
                )
            else:
                if phase_timing_diagnostics:
                    initial_cache_support_warm_call = lambda: jax.lax.map(
                        lambda lagged_bar: build_support_pullback(
                            carry0.y,
                            lagged_bar,
                            support_payload,
                        ),
                        reduced_bars.lagged_response_cache,
                    )
                initial_cache_support_bars = jax.lax.map(
                    lambda lagged_bar: build_support_pullback(
                        carry0.y,
                        lagged_bar,
                        support_payload,
                    ),
                    reduced_bars.lagged_response_cache,
                )
        initial_cache_support_bars = jax.block_until_ready(initial_cache_support_bars)
        initial_cache_support_bar_leaves = jax.tree_util.tree_leaves(initial_cache_support_bars)
        if initial_native_vmec_bars is not None:
            if native_vmec_face_coefficient_bars is None:
                raise RuntimeError(
                    "rebuild_dispatch received native VMEC coefficient bars, but "
                    "the active reverse sweep did not allocate the matching channel."
                )
            native_vmec_face_coefficient_bars = jax.tree_util.tree_map(
                lambda accumulated, increment: accumulated + increment,
                native_vmec_face_coefficient_bars,
                jax.block_until_ready(initial_native_vmec_bars),
            )
            # The native VMEC channel replaces the face-prepared contribution
            # of this lagged rebuild.  Keep this payload in the same rebuild
            # accumulator as segment rebuilds, so the final bridge merges its
            # runtime/direct-geometry leaves but does not add its generic
            # prepared-face bar a second time.
            step_support_bar_leaves_accum = tuple(
                accumulated + increment
                for accumulated, increment in zip(
                    step_support_bar_leaves_accum,
                    initial_cache_support_bar_leaves,
                )
            )
        else:
            support_bar_leaves = tuple(
                accumulated + increment
                for accumulated, increment in zip(
                    support_bar_leaves, initial_cache_support_bar_leaves)
            )
            initial_cache_support_bar_leaves_accum = tuple(
                accumulated + increment
                for accumulated, increment in zip(
                    initial_cache_support_bar_leaves_accum,
                    initial_cache_support_bar_leaves,
                )
            )
        initial_cache_pullback_used = True
        initial_cache_compile_plus_execute_elapsed = time.perf_counter() - phase_start
        print(
            f"{progress_prefix} progress: support reverse initial-cache support pullback ready "
            f"elapsed_s={initial_cache_compile_plus_execute_elapsed:.3f} "
            f"mode={initial_cache_support_pullback_mode}",
            flush=True,
        )
        if initial_cache_support_warm_call is not None:
            warm_start = time.perf_counter()
            jax.block_until_ready(initial_cache_support_warm_call())
            print(
                f"{progress_prefix} diagnostic: initial-cache support pullback "
                f"compile_plus_execute_s={initial_cache_compile_plus_execute_elapsed:.3f} "
                f"warm_execute_s={time.perf_counter() - warm_start:.3f} "
                "(one discarded duplicate; no gradient/result change)",
                flush=True,
            )
    elif (
        initial_lagged_response_valid
        and build_support_pullback is not None
        and not use_native_joint_initial_carry_pullback
        and not use_native_split_joint_initial_carry_pullback
        and not use_native_split_joint_no_prepared_carry_initial_carry_pullback
        and not use_native_split_joint_fused_rhs_initial_carry_pullback
    ):
        initial_cache_pullback_skipped = True

    def _full_carry_bar_from_reduced(reduced_bar):
        return dataclasses.replace(
            jax.tree_util.tree_map(_zero_tangent_like, carry0),
            y=reduced_bar.y,
            lagged_response_cache=reduced_bar.lagged_response_cache,
            lagged_reference_y=reduced_bar.lagged_reference_y,
        )

    phase_start = time.perf_counter()
    carry0_bars = jax.vmap(_full_carry_bar_from_reduced)(reduced_bars)
    carry0_bars = jax.block_until_ready(carry0_bars)
    print(
        f"{progress_prefix} progress: support reverse reduced carry bars expanded ready "
        f"elapsed_s={time.perf_counter() - phase_start:.3f}",
        flush=True,
    )
    # Black-box transport has no initial lagged cache.  Its initial carry
    # still contains one direct RHS evaluation, whose support cotangent is the
    # sum of the stored stage bars (the same contraction used by the initial
    # carry custom VJP).  Keep this separate from the state VJP below.
    physics_context = reverse_setup.execution_context.physics_context
    direct_initial_support_bar_leaves = None
    if (
        not bool(getattr(reverse_setup.prepared_rollout.kernel_context, "use_transport_lagged_response", False))
        and getattr(physics_context, "flat_rhs_direct_support_pullback", None) is not None
    ):
        phase_start = time.perf_counter()
        direct_initial_support_bars = _initial_direct_rhs_support_pullback_batched(
            carry0=carry0,
            carry0_bars=carry0_bars,
            kernel_context=reverse_setup.prepared_rollout.kernel_context,
            flat_rhs_direct_support_pullback=physics_context.flat_rhs_direct_support_pullback,
            support_payload=support_payload,
        )
        direct_initial_support_bars = jax.block_until_ready(direct_initial_support_bars)
        direct_initial_support_bar_leaves = tuple(
            jax.tree_util.tree_leaves(direct_initial_support_bars)
        )
        support_bar_leaves = tuple(
            accumulated + increment
            for accumulated, increment in zip(
                support_bar_leaves, direct_initial_support_bar_leaves, strict=True
            )
        )
        print(
            f"{progress_prefix} progress: support reverse initial direct-RHS support pullback ready "
            f"elapsed_s={time.perf_counter() - phase_start:.3f}",
            flush=True,
        )
    phase_start = time.perf_counter()
    initial_native_ntx_elapsed = None
    initial_direct_geometry_elapsed = None
    initial_state_warm_call = None
    with _reverse_profile_scope(reverse_setup, "reverse_post_sweep/initial_state_pullback"):
        if (
            use_native_joint_initial_carry_pullback
            or use_native_split_joint_initial_carry_pullback
            or use_native_split_joint_no_prepared_carry_initial_carry_pullback
            or use_native_split_joint_fused_rhs_initial_carry_pullback
        ):
            if not allow_initial_cache_support_pullback:
                raise ValueError(
                    "ntx_native_joint_state_and_support requires the full initial "
                    "cache support pullback cotangent mode."
                )
            initial_joint_result = initial_state_pullback(carry0_bars, support_payload)
            if (
                use_native_split_joint_initial_carry_pullback
                or use_native_split_joint_no_prepared_carry_initial_carry_pullback
                or use_native_split_joint_fused_rhs_initial_carry_pullback
            ):
                (
                    initial_state_bars,
                    initial_joint_ntx_support_bars,
                    initial_joint_native_vmec_bars,
                    initial_joint_total_lagged_bars,
                ) = initial_joint_result
                # Materialize the native NTX-only contraction before entering
                # the direct VMEC geometry transpose.  These are deliberately
                # two top-level JAX dispatches, not two Python calls within a
                # traced custom-VJP rule.
                (
                    initial_state_bars,
                    initial_joint_ntx_support_bars,
                    initial_joint_native_vmec_bars,
                    initial_joint_total_lagged_bars,
                ) = jax.block_until_ready(
                    (
                        initial_state_bars,
                        initial_joint_ntx_support_bars,
                        initial_joint_native_vmec_bars,
                        initial_joint_total_lagged_bars,
                    )
                )
                initial_native_ntx_elapsed = time.perf_counter() - phase_start
                direct_geometry_pullback = getattr(
                    getattr(reverse_setup.solve_vector_field, "__self__", None),
                    "pullback_build_lagged_response_direct_geometry_payload_"
                    "batched_interpolated_faces",
                    None,
                )
                if not callable(direct_geometry_pullback):
                    raise RuntimeError(
                        "split native initial pullback requires the direct geometry hook."
                    )
                initial_joint_geometry_bars = direct_geometry_pullback(
                    initial_state,
                    initial_joint_total_lagged_bars,
                    support_payload,
                )
                initial_joint_geometry_bars = jax.block_until_ready(
                    initial_joint_geometry_bars
                )
                initial_direct_geometry_elapsed = (
                    time.perf_counter() - phase_start - initial_native_ntx_elapsed
                )
                initial_joint_support_bars = {
                    "ntx_support": initial_joint_ntx_support_bars,
                    "geometry": initial_joint_geometry_bars,
                }
            else:
                initial_state_bars, initial_joint_support_bars = initial_joint_result
                initial_joint_native_vmec_bars = None
        else:
            initial_state_warm_call = lambda: jax.vmap(
                lambda carry0_bar: initial_state_pullback(carry0_bar)[0]
            )(carry0_bars)
            initial_state_bars = initial_state_warm_call()
            initial_joint_support_bars = None
            initial_joint_native_vmec_bars = None
    if initial_joint_support_bars is not None:
        initial_state_bars, initial_joint_support_bars = jax.block_until_ready(
            (initial_state_bars, initial_joint_support_bars)
        )
        initial_joint_support_bar_leaves = jax.tree_util.tree_leaves(
            initial_joint_support_bars
        )
        support_bar_leaves = tuple(
            accumulated + increment
            for accumulated, increment in zip(
                support_bar_leaves, initial_joint_support_bar_leaves
            )
        )
        initial_cache_support_bar_leaves_accum = tuple(
            accumulated + increment
            for accumulated, increment in zip(
                initial_cache_support_bar_leaves_accum,
                initial_joint_support_bar_leaves,
            )
        )
        if initial_joint_native_vmec_bars is not None:
            if native_vmec_face_coefficient_bars is None:
                raise RuntimeError(
                    "split native initial pullback returned VMEC coefficient bars, "
                    "but the active rebuild mode did not allocate that channel."
                )
            native_vmec_face_coefficient_bars = jax.tree_util.tree_map(
                lambda accumulated, increment: accumulated + increment,
                native_vmec_face_coefficient_bars,
                initial_joint_native_vmec_bars,
            )
        initial_cache_pullback_used = True
    else:
        initial_state_bars = jax.block_until_ready(initial_state_bars)
    initial_state_compile_plus_execute_elapsed = time.perf_counter() - phase_start
    print(
        f"{progress_prefix} progress: support reverse initial "
        f"{'joint state/support' if initial_joint_support_bars is not None else 'state'} pullback ready "
        f"elapsed_s={initial_state_compile_plus_execute_elapsed:.3f}",
        flush=True,
    )
    if phase_timing_diagnostics and initial_state_warm_call is not None:
        warm_start = time.perf_counter()
        jax.block_until_ready(initial_state_warm_call())
        print(
            f"{progress_prefix} diagnostic: initial state pullback "
            f"compile_plus_execute_s={initial_state_compile_plus_execute_elapsed:.3f} "
            f"warm_execute_s={time.perf_counter() - warm_start:.3f} "
            "(one discarded duplicate; no gradient/result change)",
            flush=True,
        )
    if initial_native_ntx_elapsed is not None:
        print(
            f"{progress_prefix} diagnostic: split initial pullback "
            f"native_ntx_state_support_s={initial_native_ntx_elapsed:.3f} "
            f"direct_geometry_s={initial_direct_geometry_elapsed:.3f}",
            flush=True,
        )
    initial_er_root_support_bars = None
    if initial_er_root_enabled:
        phase_start = time.perf_counter()
        if initial_er_root_primal is None:
            raise RuntimeError(
                "Initial-Er root reverse boundary requires the forward selected-root primal."
            )
        er_profile, finite_mask = initial_er_root_primal

        er_profile = jnp.asarray(er_profile, dtype=pre_root_initial_state.Er.dtype)
        finite_mask = jnp.asarray(finite_mask, dtype=bool)

        root_linearization_start = time.perf_counter()
        dres_der = dependencies.initial_er_charge_flux_residual_er_derivative(
            pre_root_initial_state,
            er_profile,
            runtime=runtime,
        )
        safe_dres_der = jnp.where(
            jnp.abs(dres_der) > jnp.asarray(1.0e-30, dtype=dres_der.dtype),
            dres_der,
            jnp.inf,
        )
        residual_bars = jnp.where(
            finite_mask[None, :],
            -jnp.asarray(initial_state_bars.Er) / safe_dres_der[None, :],
            0.0,
        )
        root_linearization_elapsed = time.perf_counter() - root_linearization_start

        root_state_pullback_start = time.perf_counter()
        state_residual_bars = dependencies.compact_initial_er_state_pullback(
            residual_scalar_fn=dependencies.initial_er_charge_flux_residual_scalar,
            state=pre_root_initial_state,
            er_profile=er_profile,
            residual_bars=residual_bars,
            runtime=runtime,
        )
        if phase_timing_diagnostics:
            state_residual_bars = jax.block_until_ready(state_residual_bars)
        root_state_pullback_elapsed = time.perf_counter() - root_state_pullback_start
        direct_initial_state_bars = dataclasses.replace(
            initial_state_bars,
            Er=jnp.zeros_like(initial_state_bars.Er),
        )
        pre_root_initial_state_bars = dependencies.add_trees(
            direct_initial_state_bars,
            state_residual_bars,
        )

        root_geometry_pullback_elapsed = None
        root_ntx_support_pullback_elapsed = None
        if combined_geometry_payload and "ntx_support" in support_payload:
            geometry = support_payload["geometry"]
            ntx_support = support_payload["ntx_support"]
            geometry_delta0 = _float_delta_tree_like(geometry)

            def _residuals_from_geometry_delta(geometry_delta):
                runtime_with_geometry = dependencies.runtime_with_geometry_payload(
                    runtime,
                    _add_float_delta_tree(geometry, geometry_delta),
                )
                runtime_with_geometry = dependencies.runtime_with_ntx_support_payload(
                    runtime_with_geometry,
                    ntx_support,
                )
                return dependencies.initial_er_charge_flux_residuals(
                    pre_root_initial_state,
                    er_profile,
                    runtime=runtime_with_geometry,
                )

            root_geometry_pullback_start = time.perf_counter()
            _, geometry_pullback = jax.vjp(_residuals_from_geometry_delta, geometry_delta0)
            geometry_bars = jax.vmap(lambda residual_bar: geometry_pullback(residual_bar)[0])(
                residual_bars
            )
            if phase_timing_diagnostics:
                geometry_bars = jax.block_until_ready(geometry_bars)
            root_geometry_pullback_elapsed = (
                time.perf_counter() - root_geometry_pullback_start
            )
            ntx_runtime = dependencies.runtime_with_geometry_payload(runtime, geometry)
            root_ntx_support_pullback_start = time.perf_counter()
            ntx_bar_leaves = dependencies.compact_initial_er_ntx_support_pullback_leaves(
                runtime=ntx_runtime,
                state=pre_root_initial_state,
                er_profile=er_profile,
                residual_bars=residual_bars,
                support=ntx_support,
            )
            if phase_timing_diagnostics:
                ntx_bar_leaves = jax.block_until_ready(ntx_bar_leaves)
            root_ntx_support_pullback_elapsed = (
                time.perf_counter() - root_ntx_support_pullback_start
            )
            initial_er_root_support_bars = (
                tuple(jax.tree_util.tree_leaves(geometry_bars)) + tuple(ntx_bar_leaves)
            )
        elif combined_geometry_payload and "database" in support_payload:
            # Recorded scan database: transpose charge-weighted particle flux
            # directly to the three tables, then preserve only the explicit
            # geometry derivative in a database-fixed VJP.  The resulting
            # table bars are folded through the retained scan once with the
            # rest of the transport reverse support bars.
            root_ntx_support_pullback_start = time.perf_counter()
            database_bars = compact_initial_er_database_support_bars(
                runtime=runtime,
                state=pre_root_initial_state,
                er_profile=er_profile,
                residual_bars=residual_bars,
                support=support_payload,
            )
            if phase_timing_diagnostics:
                database_bars = jax.block_until_ready(database_bars)
            root_ntx_support_pullback_elapsed = (
                time.perf_counter() - root_ntx_support_pullback_start
            )
            geometry = support_payload["geometry"]
            geometry_delta0 = _float_delta_tree_like(geometry)

            def _residuals_from_geometry_delta(geometry_delta):
                payload = dict(support_payload)
                payload["geometry"] = _add_float_delta_tree(geometry, geometry_delta)
                runtime_with_geometry = (
                    dependencies.runtime_with_realtime_geometry_reverse_support_payload(
                        runtime, payload
                    )
                )
                return dependencies.initial_er_charge_flux_residuals(
                    pre_root_initial_state, er_profile, runtime=runtime_with_geometry
                )

            root_geometry_pullback_start = time.perf_counter()
            _, geometry_pullback = jax.vjp(
                _residuals_from_geometry_delta, geometry_delta0
            )
            geometry_bars = jax.vmap(
                lambda residual_bar: geometry_pullback(residual_bar)[0]
            )(residual_bars)
            if phase_timing_diagnostics:
                geometry_bars = jax.block_until_ready(geometry_bars)
            root_geometry_pullback_elapsed = time.perf_counter() - root_geometry_pullback_start

            def _batched_zero(tree):
                return jax.tree_util.tree_map(
                    lambda leaf: jnp.zeros(
                        (residual_bars.shape[0],) + jnp.asarray(leaf).shape,
                        dtype=(
                            jnp.asarray(leaf).dtype
                            if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.inexact)
                            else jnp.float64
                        ),
                    ),
                    tree,
                )

            batched_support_bars = {
                "geometry": geometry_bars,
                "channels": _batched_zero(support_payload["channels"]),
                "surfaces": _batched_zero(support_payload["surfaces"]),
                "database": database_bars,
            }
            initial_er_root_support_bars = tuple(
                jax.tree_util.tree_leaves(batched_support_bars)
            )
        elif combined_geometry_payload:
            # Compatibility path for unrecorded scan payloads.
            support_delta0 = _float_delta_tree_like(support_payload)

            def _residuals_from_support_delta(support_delta):
                runtime_with_support = (
                    dependencies.runtime_with_realtime_geometry_reverse_support_payload(
                        runtime,
                        _add_float_delta_tree(support_payload, support_delta),
                    )
                )
                return dependencies.initial_er_charge_flux_residuals(
                    pre_root_initial_state,
                    er_profile,
                    runtime=runtime_with_support,
                )

            root_geometry_pullback_start = time.perf_counter()
            _, support_pullback = jax.vjp(
                _residuals_from_support_delta, support_delta0
            )
            batched_support_bars = jax.vmap(
                lambda residual_bar: support_pullback(residual_bar)[0]
            )(residual_bars)
            if phase_timing_diagnostics:
                batched_support_bars = jax.block_until_ready(batched_support_bars)
            root_geometry_pullback_elapsed = (
                time.perf_counter() - root_geometry_pullback_start
            )
            initial_er_root_support_bars = tuple(
                jax.tree_util.tree_leaves(batched_support_bars)
            )
        else:
            root_ntx_support_pullback_start = time.perf_counter()
            initial_er_root_support_bars = dependencies.compact_initial_er_ntx_support_pullback_leaves(
                runtime=runtime,
                state=pre_root_initial_state,
                er_profile=er_profile,
                residual_bars=residual_bars,
                support=support_payload,
            )
            if phase_timing_diagnostics:
                initial_er_root_support_bars = jax.block_until_ready(
                    initial_er_root_support_bars
                )
            root_ntx_support_pullback_elapsed = (
                time.perf_counter() - root_ntx_support_pullback_start
            )
        pre_root_initial_state_bars, initial_er_root_support_bars = jax.block_until_ready(
            (pre_root_initial_state_bars, initial_er_root_support_bars)
        )
        print(
            f"{progress_prefix} progress: initial-Er root boundary compact pullback ready "
            f"elapsed_s={time.perf_counter() - phase_start:.3f}",
            flush=True,
        )
        if phase_timing_diagnostics:
            def _root_component_time(value):
                return "n/a" if value is None else f"{value:.3f}"

            print(
                f"{progress_prefix} diagnostic: initial-Er root boundary components "
                "reused_forward_selected_root_s=0.000 "
                f"root_linearization_s={root_linearization_elapsed:.3f} "
                f"state_residual_transpose_s={root_state_pullback_elapsed:.3f} "
                f"direct_geometry_transpose_s={_root_component_time(root_geometry_pullback_elapsed)} "
                f"ntx_support_transpose_s={_root_component_time(root_ntx_support_pullback_elapsed)} "
                "(diagnostic synchronization only; no duplicate pullbacks or result change)",
                flush=True,
            )
        initial_state_bars = pre_root_initial_state_bars

    phase_start = time.perf_counter()
    with _reverse_profile_scope(reverse_setup, "reverse_post_sweep/profile_parameter_pullback"):
        gradient_matrix = jax.vmap(
            lambda state_bar: profile_state_pullback(state_bar)[0]
        )(initial_state_bars)
    gradient_matrix = jax.block_until_ready(gradient_matrix)
    print(
        f"{progress_prefix} progress: support reverse profile parameter pullback ready "
        f"elapsed_s={time.perf_counter() - phase_start:.3f}",
        flush=True,
    )
    if initial_er_root_support_bars is not None:
        raw_initial_er_root_support_bar_leaves = tuple(initial_er_root_support_bars)
        if len(raw_initial_er_root_support_bar_leaves) != len(support_bar_leaves):
            raise ValueError(
                "Initial-Er root support pullback produced "
                f"{len(raw_initial_er_root_support_bar_leaves)} leaves, but support payload expects "
                f"{len(support_bar_leaves)} leaves."
            )
        for leaf_i, (accumulated, increment) in enumerate(
            zip(support_bar_leaves, raw_initial_er_root_support_bar_leaves, strict=True)
        ):
            if jnp.asarray(increment).shape != jnp.asarray(accumulated).shape:
                raise ValueError(
                    "Initial-Er root support pullback leaf shape mismatch at "
                    f"leaf {leaf_i}: got {jnp.asarray(increment).shape}, "
                    f"expected {jnp.asarray(accumulated).shape}."
                )
        initial_er_root_support_bar_leaves = tuple(
            jnp.zeros_like(accumulated)
            if jnp.asarray(increment).dtype == jax.dtypes.float0
            else jnp.asarray(increment)
            for accumulated, increment in zip(
                support_bar_leaves,
                raw_initial_er_root_support_bar_leaves,
            )
        )
        support_bar_leaves = tuple(
            accumulated + increment
            for accumulated, increment in zip(support_bar_leaves, initial_er_root_support_bar_leaves)
        )
    # The native VMEC-coefficient route replaces only the *rebuild face
    # prepared-system* contribution.  Objective, initial-cache, and
    # initial-Er-root prepared bars still require the ordinary payload VJP.
    # Keep those generic bars, but fold the rebuild runtime-channel bars into
    # them before that one VJP.  This is deliberately a single full support
    # pullback: a previous version used a second channels-only Boozer VJP,
    # which both dropped the generic prepared bars and caused a large peak
    # allocation after the segment sweep.
    if native_vmec_treedef is None:
        support_bars = tuple(
            support_treedef.unflatten(
                [jnp.asarray(leaf)[objective_i] for leaf in support_bar_leaves]
            )
            for objective_i in range(objective_count)
        )
    else:
        zero_initial_er_root_leaves = tuple(
            jnp.zeros_like(leaf) for leaf in support_bar_leaves
        )
        generic_support_bar_leaves = tuple(
            objective_bar + initial_cache_bar + initial_er_root_bar
            for objective_bar, initial_cache_bar, initial_er_root_bar in zip(
                objective_support_bar_leaves,
                initial_cache_support_bar_leaves_accum,
                (
                    initial_er_root_support_bar_leaves
                    if initial_er_root_support_bars is not None
                    else zero_initial_er_root_leaves
                ),
                strict=True,
            )
        )
        generic_support_bars = tuple(
            support_treedef.unflatten(
                [jnp.asarray(leaf)[objective_i] for leaf in generic_support_bar_leaves]
            )
            for objective_i in range(objective_count)
        )
        rebuild_support_bars = tuple(
            support_treedef.unflatten(
                [jnp.asarray(leaf)[objective_i] for leaf in step_support_bar_leaves_accum]
            )
            for objective_i in range(objective_count)
        )

        support_bars = tuple(
            _merge_rebuild_ntx_channels_into_generic_payload_bar(
                generic_bar, rebuild_bar
            )
            for generic_bar, rebuild_bar in zip(
                generic_support_bars, rebuild_support_bars, strict=True
            )
        )
        # Do not retain the large generic/rebuild payload trees while the
        # later VMEC raw-block pullback is staged.
        del generic_support_bars, rebuild_support_bars, generic_support_bar_leaves
    component_support_bars_by_name = {
        "objective_explicit": tuple(
            support_treedef.unflatten(
                [jnp.asarray(leaf)[objective_i] for leaf in objective_support_bar_leaves]
            )
            for objective_i in range(objective_count)
        ),
        "transport_rhs": tuple(
            support_treedef.unflatten(
                [jnp.asarray(leaf)[objective_i] for leaf in step_support_bar_leaves_accum]
            )
            for objective_i in range(objective_count)
        ),
        "initial_cache": tuple(
            support_treedef.unflatten(
                [jnp.asarray(leaf)[objective_i] for leaf in initial_cache_support_bar_leaves_accum]
            )
            for objective_i in range(objective_count)
        ),
    }
    if direct_initial_support_bar_leaves is not None:
        component_support_bars_by_name["initial_direct_rhs"] = tuple(
            support_treedef.unflatten(
                [jnp.asarray(leaf)[objective_i] for leaf in direct_initial_support_bar_leaves]
            )
            for objective_i in range(objective_count)
        )
    if initial_er_root_support_bars is not None:
        component_support_bars_by_name["initial_er_root"] = tuple(
            support_treedef.unflatten(
                [jnp.asarray(leaf)[objective_i] for leaf in initial_er_root_support_bar_leaves]
            )
            for objective_i in range(objective_count)
        )
    if combined_geometry_payload and "ntx_support" in support_payload:
        phase_start = time.perf_counter()
        geometry = support_payload["geometry"]
        geometry_delta0 = _float_delta_tree_like(geometry)

        def _initial_state_from_geometry_delta(geometry_delta):
            runtime_with_geometry = dataclasses.replace(
                runtime,
                geometry=_add_float_delta_tree(geometry, geometry_delta),
            )
            return dependencies.initial_state_for_parameter_vector(
                parameter_values,
                config=config,
                initial_er_root_ad="off",
                baseline_state=baseline_state,
                profile_cfg=profile_cfg,
                runtime=runtime_with_geometry,
            )

        _, initial_geometry_pullback = jax.vjp(_initial_state_from_geometry_delta, geometry_delta0)
        with _reverse_profile_scope(
            reverse_setup, "reverse_post_sweep/initial_profile_geometry_pullback"
        ):
            initial_geometry_bars = jax.vmap(
                lambda state_bar: initial_geometry_pullback(state_bar)[0]
            )(initial_state_bars)
        initial_geometry_bars = jax.block_until_ready(initial_geometry_bars)
        print(
            f"{progress_prefix} progress: support reverse initial-profile geometry pullback ready "
            f"elapsed_s={time.perf_counter() - phase_start:.3f}",
            flush=True,
        )
        component_support_bars_by_name["initial_profile"] = tuple(
            {
                "geometry": _sanitize_float_delta_bar_tree(
                    support_payload["geometry"],
                    jax.tree_util.tree_map(lambda value: value[objective_i], initial_geometry_bars),
                ),
                "ntx_support": zero_payload_bar["ntx_support"],
            }
            for objective_i in range(objective_count)
        )
        support_bars = tuple(
            {
                "geometry": _sanitize_float_delta_bar_tree(
                    support_payload["geometry"],
                    dependencies.add_trees(
                        support_bar["geometry"],
                        jax.tree_util.tree_map(lambda value: value[objective_i], initial_geometry_bars),
                    ),
                ),
                "ntx_support": support_bar["ntx_support"],
            }
            for objective_i, support_bar in enumerate(support_bars)
        )
    elif combined_geometry_payload:
        # Keep the initial-profile contribution in the same scan support tree
        # as the reverse-step and selected-root contributions.  This is a
        # generic VJP of the ordinary initialization path, not an additional
        # NTX adjoint or an independent database cotangent.
        phase_start = time.perf_counter()
        support_delta0 = _float_delta_tree_like(support_payload)

        def _initial_state_from_support_delta(support_delta):
            runtime_with_support = (
                dependencies.runtime_with_realtime_geometry_reverse_support_payload(
                    runtime,
                    _add_float_delta_tree(support_payload, support_delta),
                )
            )
            return dependencies.initial_state_for_parameter_vector(
                parameter_values,
                config=config,
                initial_er_root_ad="off",
                baseline_state=baseline_state,
                profile_cfg=profile_cfg,
                runtime=runtime_with_support,
            )

        _, initial_support_pullback = jax.vjp(
            _initial_state_from_support_delta, support_delta0
        )
        initial_support_bars = jax.vmap(
            lambda state_bar: initial_support_pullback(state_bar)[0]
        )(initial_state_bars)
        initial_support_bars = jax.block_until_ready(initial_support_bars)
        print(
            f"{progress_prefix} progress: support reverse initial-profile scan payload pullback ready "
            f"elapsed_s={time.perf_counter() - phase_start:.3f}",
            flush=True,
        )
        initial_support_rows = tuple(
            _take_tree_axis0(initial_support_bars, objective_i)
            for objective_i in range(objective_count)
        )
        component_support_bars_by_name["initial_profile"] = tuple(
            _sanitize_float_delta_bar_tree(support_payload, initial_support_row)
            for initial_support_row in initial_support_rows
        )
        support_bars = tuple(
            _sanitize_float_delta_bar_tree(
                support_payload,
                dependencies.add_trees(support_bar, initial_support_row),
            )
            for support_bar, initial_support_row in zip(
                support_bars, initial_support_rows, strict=True
            )
        )
    return (
        objective_values,
        gradient_matrix,
        support_bars,
        component_support_bars_by_name,
        support_reuse_count,
        support_rebuild_count,
        initial_cache_pullback_used,
        initial_cache_pullback_skipped,
        native_vmec_face_coefficient_bars,
    )


def realtime_geometry_support_cotangents_from_parameter_vector(
    *,
    reverse_all_objectives_support_payload_bar: Callable[..., object] | None = None,
    profile_values,
    config: Mapping[str, Any],
    baseline_runtime,
    baseline_state,
    profile_cfg: Mapping[str, Any],
    reverse_setup,
    support_payload,
    initial_er_root_ad: str,
    block_until_ready: bool = True,
) -> RealtimeGeometrySupportCotangentResult:
    """Run the grouped all-objective transport support-cotangent pullback.

    By default this uses the internal compact reverse dependencies. A callback
    can still be supplied by legacy benchmark adapters, but optimization callers
    should use the default internal path.
    """

    if reverse_all_objectives_support_payload_bar is None:
        def reverse_all_objectives_support_payload_bar(*args, **kwargs):
            return realtime_geometry_reverse_all_objectives_support_payload_bar_for_parameter_vector(
                *args,
                objective_labels=TRANSPORT_REVERSE_OBJECTIVE_LABELS,
                dependencies=default_realtime_geometry_support_reverse_dependencies(),
                **kwargs,
            )
    if not callable(reverse_all_objectives_support_payload_bar):
        raise TypeError("reverse_all_objectives_support_payload_bar must be callable.")
    callback_result = reverse_all_objectives_support_payload_bar(
        profile_values,
        config=config,
        runtime=baseline_runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        reverse_setup=reverse_setup,
        support_payload=support_payload,
        initial_er_root_ad=initial_er_root_ad,
    )
    if not isinstance(callback_result, tuple) or len(callback_result) not in {8, 9}:
        raise TypeError(
            "reverse_all_objectives_support_payload_bar must return an 8- or 9-tuple: "
            "(objective_values, profile_gradient_matrix, support_bars, "
            "support_component_bars_by_name, support_reuse_count, "
            "support_rebuild_count, initial_cache_pullback_used, "
            "initial_cache_pullback_skipped[, native_vmec_face_coefficient_bars])."
        )
    (
        objective_values,
        profile_gradient_matrix,
        support_bars,
        support_component_bars_by_name,
        support_reuse_count,
        support_rebuild_count,
        initial_cache_pullback_used,
        initial_cache_pullback_skipped,
        *native_vmec_result,
    ) = callback_result
    native_vmec_face_coefficient_bars = (
        native_vmec_result[0] if native_vmec_result else None
    )
    if block_until_ready:
        (
            objective_values,
            profile_gradient_matrix,
            support_bars,
            support_component_bars_by_name,
            native_vmec_face_coefficient_bars,
        ) = (
            jax.block_until_ready(
                (
                    objective_values,
                    profile_gradient_matrix,
                    support_bars,
                    support_component_bars_by_name,
                    native_vmec_face_coefficient_bars,
                )
            )
        )
    return RealtimeGeometrySupportCotangentResult(
        objective_values=objective_values,
        profile_gradient_matrix=profile_gradient_matrix,
        support_bars=support_bars,
        support_component_bars_by_name=support_component_bars_by_name,
        support_reuse_count=int(support_reuse_count),
        support_rebuild_count=int(support_rebuild_count),
        initial_cache_pullback_used=bool(initial_cache_pullback_used),
        initial_cache_pullback_skipped=bool(initial_cache_pullback_skipped),
        native_vmec_face_coefficient_bars=native_vmec_face_coefficient_bars,
    )


def realtime_geometry_transport_reverse_support_segment_executor(
    *,
    support_segment_probe: TransportReverseSupportSegmentProbe,
    config: Mapping[str, Any],
    baseline_values,
    baseline_runtime,
    baseline_state,
    profile_cfg: Mapping[str, Any],
    neoclassical_cfg: Mapping[str, Any],
) -> TransportReverseSupportSegmentExecutor:
    """Build an executor wrapper around a segmented reverse probe.

    The supplied probe may be the internal grouped implementation or a
    benchmark diagnostic wrapper. This owns the reusable calling convention:
    grouped optimization paths must request an internal report and receive the
    shared runtime/config inputs.
    """

    if not callable(support_segment_probe):
        raise TypeError("support_segment_probe must be callable.")
    context = realtime_geometry_transport_reverse_table_context(
        config=config,
        baseline_values=baseline_values,
        baseline_runtime=baseline_runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        neoclassical_cfg=neoclassical_cfg,
    )

    def _executor(builder_args, return_report: bool) -> TransportReverseReport:
        if not return_report:
            raise ValueError("Grouped optimization builders require return_report=True.")
        return run_realtime_geometry_support_segment_reverse_table_core(
            support_segment_probe=support_segment_probe,
            args=builder_args,
            context=context,
            suppress_output=False,
        )

    return _executor


def _format_geometry_param_spec(param_spec: tuple[str, int, int]) -> str:
    family, m, n = param_spec
    return f"vmec:{str(family).strip().upper()}:{int(m)}:{int(n)}"


def _parse_reverse_geometry_parameter(parameter_name: str) -> tuple[str, int, int]:
    parts = str(parameter_name).split(":")
    if len(parts) != 3:
        raise ValueError(
            "Reverse geometry parameters must use FAMILY:m:n, for example 'RBC:1:0'."
        )
    family = parts[0].strip().upper()
    if not family:
        raise ValueError("Reverse geometry parameter family cannot be empty.")
    try:
        return family, int(parts[1]), int(parts[2])
    except ValueError as exc:
        raise ValueError(
            "Reverse geometry parameters must use integer m/n values, "
            "for example 'RBC:1:0'."
        ) from exc


def _geometry_context_from_reverse_args(config: Mapping[str, Any], args):
    geometry_parameter = str(getattr(args, "reverse_geometry_parameter", "RBC:1:0"))
    if geometry_parameter.strip().lower() == "all":
        family, m, n = ("RBC", 0, 0)
    else:
        family, m, n = _parse_reverse_geometry_parameter(geometry_parameter)
    geom_cfg = config.get("geometry", {})
    vmec_input_file = geom_cfg.get("vmec_input_file")
    if vmec_input_file is None:
        raise ValueError("Realtime geometry reverse mode requires geometry.vmec_input_file.")
    return build_geometry_autodiff_context(
        vmec_input_file,
        param_family=family,
        param_m=m,
        param_n=n,
        mboz=int(geom_cfg.get("mboz", geom_cfg.get("vmec_mboz", 12))),
        nboz=int(geom_cfg.get("nboz", geom_cfg.get("vmec_nboz", 12))),
    )


def _geometry_param_specs_from_reverse_args(args, geometry_context) -> tuple[tuple[str, int, int], ...]:
    geometry_parameter = str(getattr(args, "reverse_geometry_parameter", "RBC:1:0")).strip()
    if geometry_parameter.lower() != "all":
        return (_parse_reverse_geometry_parameter(geometry_parameter),)
    families = normalize_vmec_boundary_families(
        getattr(args, "reverse_geometry_families", "RBC,ZBS")
    )
    try:
        specs = discover_vmec_boundary_parameter_specs(
            geometry_context,
            families=families,
            nonzero_only=not bool(getattr(args, "reverse_geometry_include_zero_harmonics", False)),
        )
    except ValueError as exc:
        raise ValueError(
            "No VMEC boundary harmonics matched the requested all-harmonic selector. "
            "Try enabling zero harmonics or changing the requested families."
        ) from exc
    return vmec_boundary_tuples(specs)


def _baseline_geometry_delta_vector_for_specs(
    geom_cfg: Mapping[str, Any],
    geometry_param_specs: Sequence[tuple[str, int, int]],
) -> jax.Array:
    deltas = np.zeros((len(geometry_param_specs),), dtype=np.float64)
    configured_delta = float(geom_cfg.get("vmec_param_delta", 0.0))
    if configured_delta != 0.0:
        configured_spec = (
            str(geom_cfg.get("vmec_param_family", "RBC")).strip().upper(),
            int(geom_cfg.get("vmec_param_m", 0)),
            int(geom_cfg.get("vmec_param_n", 0)),
        )
        for i, spec in enumerate(geometry_param_specs):
            normalized_spec = (str(spec[0]).strip().upper(), int(spec[1]), int(spec[2]))
            if normalized_spec == configured_spec:
                deltas[i] = configured_delta
                break
    return jnp.asarray(deltas, dtype=jnp.float64)


def _array_finite_summary(value) -> dict[str, Any]:
    arr = np.asarray(jax.device_get(jnp.asarray(value)))
    finite = np.isfinite(arr)
    finite_values = arr[finite]
    first_nonfinite_index = None
    if not bool(np.all(finite)):
        first_nonfinite_index = [int(i) for i in np.argwhere(~finite)[0].tolist()]
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "all_finite": bool(np.all(finite)),
        "nan_count": int(np.isnan(arr).sum()),
        "posinf_count": int(np.isposinf(arr).sum()),
        "neginf_count": int(np.isneginf(arr).sum()),
        "finite_min": None if finite_values.size == 0 else float(np.min(finite_values)),
        "finite_max": None if finite_values.size == 0 else float(np.max(finite_values)),
        "first_nonfinite_index": first_nonfinite_index,
    }


def geometry_volume_diagnostics(geometry) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for name in (
        "a_b",
        "R0",
        "rho_grid",
        "rho_grid_half",
        "r_grid",
        "r_grid_half",
        "Vprime",
        "Vprime_half",
        "overVprime",
    ):
        if hasattr(geometry, name):
            diagnostics[name] = _array_finite_summary(getattr(geometry, name))
    if hasattr(geometry, "Vprime") and hasattr(geometry, "r_grid"):
        volume = jnp.trapezoid(
            jnp.asarray(geometry.Vprime),
            x=jnp.asarray(geometry.r_grid),
        )
        diagnostics["integrated_volume"] = _array_finite_summary(volume)
        diagnostics["integrated_volume_value"] = float(np.asarray(jax.device_get(volume)))
    return diagnostics


def run_internal_realtime_geometry_support_segment_probe(
    *,
    args,
    context: RealtimeGeometryTransportReverseTableContext,
    return_report: bool = True,
    suppress_diagnostics: bool = True,
) -> TransportReverseReport:
    """Run the benchmark-matched grouped realtime-geometry reverse table internally.

    This is the all-objective grouped path used by optimization callers. It
    keeps the compact support cotangent sweep and raw-block VMEC payload
    pullback together, without importing benchmark modules.
    """

    if not return_report:
        raise ValueError("Internal grouped realtime-geometry reverse requires return_report=True.")
    if str(getattr(args, "objective", "all")) != "all":
        raise ValueError("Internal grouped realtime-geometry reverse currently requires objective='all'.")
    config = context.config
    baseline_runtime = context.baseline_runtime
    baseline_state = context.baseline_state
    profile_cfg = context.profile_cfg
    neoclassical_cfg = context.neoclassical_cfg
    core_setup = prepare_realtime_geometry_support_segment_core_setup(
        args=args,
        config=config,
        baseline_values=context.baseline_values,
        baseline_runtime=baseline_runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        neoclassical_cfg=neoclassical_cfg,
        parameter_order=TRANSPORT_REVERSE_PROFILE_PARAMETER_ORDER,
        find_ntx_support_payload=find_ntx_support_payload,
        prepare_reverse_static_setup=prepare_reverse_static_setup,
        geometry_volume_diagnostics=None if suppress_diagnostics else geometry_volume_diagnostics,
    )
    t_start = time.perf_counter()
    t_phase = time.perf_counter()
    support_cotangent_result = realtime_geometry_support_cotangents_from_parameter_vector(
        profile_values=core_setup.profile_values,
        config=config,
        baseline_runtime=baseline_runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        reverse_setup=core_setup.reverse_setup,
        support_payload=core_setup.support_payload,
        initial_er_root_ad=getattr(args, "initial_er_root_ad", "off"),
    )
    if not suppress_diagnostics:
        print(
            "[autodiff-gate] progress: transport reverse profile/support cotangents complete "
            f"elapsed_s={time.perf_counter() - t_phase:.3f}",
            flush=True,
        )
    geom_cfg = config.get("geometry", {})
    geometry_context = _geometry_context_from_reverse_args(config, args)
    geometry_param_specs = _geometry_param_specs_from_reverse_args(args, geometry_context)
    geometry_param_entries = boundary_param_entries(geometry_context, geometry_param_specs)
    geometry_param_labels = tuple(_format_geometry_param_spec(spec) for spec in geometry_param_specs)
    baseline_geometry_deltas = _baseline_geometry_delta_vector_for_specs(
        geom_cfg,
        geometry_param_specs,
    )
    t_phase = time.perf_counter()
    assembly_result = realtime_geometry_transport_reverse_table_from_payload_cotangents(
        objective_labels=TRANSPORT_REVERSE_OBJECTIVE_LABELS,
        profile_parameter_labels=TRANSPORT_REVERSE_PROFILE_PARAMETER_ORDER,
        geometry_parameter_labels=geometry_param_labels,
        objective_values=support_cotangent_result.objective_values,
        profile_gradient_matrix=support_cotangent_result.profile_gradient_matrix,
        geometry_context=geometry_context,
        baseline_geometry_deltas=baseline_geometry_deltas,
        geometry_param_specs=geometry_param_specs,
        support_bars=tuple(support_cotangent_result.support_bars),
        support_component_bars_by_name=support_cotangent_result.support_component_bars_by_name,
        native_vmec_face_coefficient_bars=support_cotangent_result.native_vmec_face_coefficient_bars,
        include_component_pullbacks=bool(getattr(args, "realtime_geometry_component_pullbacks", False)),
        combined_geometry_payload=core_setup.combined_geometry_payload,
        payload_kind=core_setup.payload_kind,
        scan_rho=neoclassical_cfg.get("ntx_scan_rho"),
        scan_surface_backend=str(neoclassical_cfg.get("ntx_scan_surface_backend", "vmec")),
        n_r=int(geom_cfg.get("n_radial", 51)),
        n_theta=int(neoclassical_cfg.get("ntx_exact_n_theta", 25)),
        n_zeta=int(neoclassical_cfg.get("ntx_exact_n_zeta", 25)),
        n_xi=int(neoclassical_cfg.get("ntx_exact_n_xi", 64)),
        surface_backend=core_setup.ntx_surface_backend,
        max_iter=geom_cfg.get("vmec_max_iter"),
        solver_device=str(geom_cfg.get("vmec_implicit_solver_device", "default")),
        progress_label=None if suppress_diagnostics else "[autodiff-gate] realtime geometry payload pullback:",
        return_branch_gradients=False,
    )
    if not suppress_diagnostics:
        print(
            "[autodiff-gate] progress: geometry support pullback complete "
            f"mode={assembly_result.payload_pullback_result.pullback_mode} "
            f"elapsed_s={time.perf_counter() - t_phase:.3f}",
            flush=True,
        )
    table_result = assembly_result.table_result
    metadata_entries = realtime_geometry_transport_reverse_metadata_entries(
        parameter_mode=str(getattr(args, "reverse_parameter_mode", "profiles_plus_realtime_geometry")),
        config_path=str(getattr(args, "config", "")),
        objective_labels=TRANSPORT_REVERSE_OBJECTIVE_LABELS,
        profile_parameter_labels=TRANSPORT_REVERSE_PROFILE_PARAMETER_ORDER,
        profile_values=core_setup.profile_values,
        geometry_parameter_labels=geometry_param_labels,
        geometry_parameter_entries=geometry_param_entries,
        baseline_geometry_deltas=baseline_geometry_deltas,
        geometry_parameter_specs=geometry_param_specs,
        geometry_parameter_selector=str(getattr(args, "reverse_geometry_parameter", "RBC:1:0")),
        accepted_step_limit=None
        if getattr(args, "accepted_step_limit", None) is None
        else int(getattr(args, "accepted_step_limit")),
        reverse_segment_length=None
        if getattr(args, "reverse_segment_length", None) is None
        else int(getattr(args, "reverse_segment_length")),
        reverse_stage_cotangent_mode_requested=str(getattr(args, "reverse_stage_cotangent_mode", "full")),
        reverse_stage_cotangent_mode_effective=core_setup.support_probe_cotangent_mode,
        ntx_exact_derivative_mode=str(getattr(args, "ntx_exact_derivative_mode", "direct")),
        ntx_exact_derivative_field_pullback_mode=str(
            getattr(args, "ntx_exact_derivative_field_pullback_mode", "compact_vjp")
        ),
        ntx_exact_surface_backend=str(neoclassical_cfg.get("ntx_exact_surface_backend", "booz")),
        realtime_geometry_gradient_path=str(getattr(args, "realtime_geometry_gradient_path", "reverse_payload")),
        realtime_geometry_component_pullbacks=bool(getattr(args, "realtime_geometry_component_pullbacks", False)),
        realtime_geometry_support_bar_diagnostics_skipped=bool(suppress_diagnostics),
        realtime_geometry_derivative_complete=bool(core_setup.combined_geometry_payload),
        geometry_support_pullback_mode=assembly_result.payload_pullback_result.pullback_mode,
        realtime_geometry_diagnostics={},
        support_payload_summary={},
        support_bar_summary_by_objective={},
        support_bar_l2_by_objective={},
        support_bar_branch_diagnostics_by_objective={},
        support_reuse_count=support_cotangent_result.support_reuse_count,
        support_rebuild_count=support_cotangent_result.support_rebuild_count,
        support_initial_cache_pullback_used=support_cotangent_result.initial_cache_pullback_used,
        support_initial_cache_pullback_skipped=support_cotangent_result.initial_cache_pullback_skipped,
        elapsed_s=time.perf_counter() - t_start,
    )
    return {
        **metadata_entries,
        **transport_reverse_table_report_entries(table_result=table_result),
        "geometry_gradient_reverse_ad_by_branch": None,
        "geometry_gradient_reverse_ad_by_component": {},
        "geometry_gradient_reverse_ad_final_state_components": {},
        "geometry_gradient_reverse_ad_by_component_and_branch": {},
        "transport_reverse_table_result": table_result,
    }


def run_realtime_geometry_support_segment_reverse_table_core(
    *,
    support_segment_probe: TransportReverseSupportSegmentProbe,
    args,
    context: RealtimeGeometryTransportReverseTableContext,
    suppress_output: bool = True,
) -> TransportReverseReport:
    """Run the segmented support probe as a non-printing reverse table core.

    This function owns the internal core calling convention: execute the probe
    with `return_report=True`, thread the table context, and require the
    JAX-native table result in the returned report.
    """

    if not callable(support_segment_probe):
        raise TypeError("support_segment_probe must be callable.")

    def _call_probe() -> TransportReverseReport:
        return support_segment_probe(
            args=args,
            config=context.config,
            baseline_values=context.baseline_values,
            baseline_runtime=context.baseline_runtime,
            baseline_state=context.baseline_state,
            profile_cfg=context.profile_cfg,
            neoclassical_cfg=context.neoclassical_cfg,
            return_report=True,
        )

    if suppress_output:
        with contextlib.redirect_stdout(io.StringIO()):
            report = _call_probe()
    else:
        report = _call_probe()
    table_result = report.get("transport_reverse_table_result") if isinstance(report, Mapping) else None
    if not isinstance(table_result, RealtimeGeometryTransportReverseTableResult):
        raise TypeError(
            "Segmented realtime-geometry reverse core did not return a "
            "RealtimeGeometryTransportReverseTableResult under "
            "'transport_reverse_table_result'."
        )
    return report


def _copy_args_with_objective_all(args):
    copied_args = copy.copy(args)
    setattr(copied_args, "objective", "all")
    return copied_args


def realtime_geometry_transport_reverse_grouped_inputs(
    *,
    args,
    config: Mapping[str, Any],
    baseline_values,
    baseline_runtime,
    baseline_state,
    profile_cfg: Mapping[str, Any],
    neoclassical_cfg: Mapping[str, Any],
    support_segment_executor: TransportReverseSupportSegmentExecutor,
    update_args_for_all_objectives: TransportReverseArgsUpdater | None = None,
) -> RealtimeGeometryTransportReverseGroupedInputs:
    """Build shared context and grouped runner for realtime-geometry reverse tables.

    The support executor owns the segmented reverse sweep. This helper owns the
    reusable context construction and objective='all' grouped-runner contract.
    """

    table_context = realtime_geometry_transport_reverse_table_context(
        config=config,
        baseline_values=baseline_values,
        baseline_runtime=baseline_runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        neoclassical_cfg=neoclassical_cfg,
    )
    update_args = (
        _copy_args_with_objective_all
        if update_args_for_all_objectives is None
        else update_args_for_all_objectives
    )
    return RealtimeGeometryTransportReverseGroupedInputs(
        table_context=table_context,
        run_grouped_report=realtime_geometry_transport_reverse_grouped_runner(
            args=args,
            support_segment_executor=support_segment_executor,
            update_args_for_all_objectives=update_args,
        ),
    )


def realtime_geometry_transport_reverse_table_request(
    *,
    objective_names: Sequence[str],
    parameter_set: ReverseADParameterSet,
    context: RealtimeGeometryTransportReverseTableContext,
    options: Mapping[str, object] | None = None,
) -> RealtimeGeometryTransportReverseTableRequest:
    """Build and validate a grouped realtime-geometry reverse table request."""

    return RealtimeGeometryTransportReverseTableRequest(
        objective_names=tuple(str(name) for name in objective_names),
        parameter_set=parameter_set,
        context=context,
        options=options,
    )


def realtime_geometry_transport_reverse_table_result(
    *,
    objective_labels: Sequence[str],
    profile_parameter_labels: Sequence[str],
    geometry_parameter_labels: Sequence[str],
    objective_values,
    profile_gradient_matrix,
    geometry_gradient_matrix,
) -> RealtimeGeometryTransportReverseTableResult:
    """Build the JAX-native grouped transport reverse table result."""

    return RealtimeGeometryTransportReverseTableResult(
        objective_labels=tuple(str(name) for name in objective_labels),
        profile_parameter_labels=tuple(str(name) for name in profile_parameter_labels),
        geometry_parameter_labels=tuple(str(name) for name in geometry_parameter_labels),
        objective_values=objective_values,
        profile_gradient_matrix=profile_gradient_matrix,
        geometry_gradient_matrix=geometry_gradient_matrix,
    )


def _validate_transport_reverse_parameter_set(parameter_set: ReverseADParameterSet) -> None:
    """Validate the transport reverse parameter layout placeholder.

    The current report runner still owns parameter handling.  Keeping this
    explicit hook makes the production seam ready for step 2 without changing
    the public helper signature.
    """

    if not isinstance(parameter_set, ReverseADParameterSet):
        raise TypeError(
            "parameter_set must be a ReverseADParameterSet; "
            f"got {type(parameter_set).__name__}."
        )


def normalize_transport_objective_names(
    objective_names: Sequence[str],
    *,
    objective_labels: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Validate and normalize grouped transport objective names."""

    requested_objectives = tuple(str(name).strip() for name in objective_names)
    if not requested_objectives or any(not name for name in requested_objectives):
        raise ValueError("At least one non-empty transport objective name is required.")
    if objective_labels is not None:
        available_labels = tuple(str(name) for name in objective_labels)
        missing = tuple(name for name in requested_objectives if name not in available_labels)
        if missing:
            available = ", ".join(available_labels)
            raise ValueError(
                "Grouped transport reverse report builder received unknown "
                f"transport objectives {missing!r}. Available objectives: {available}."
            )
    return requested_objectives


def build_realtime_geometry_transport_reverse_report(
    *,
    objective_names: Sequence[str],
    parameter_set: ReverseADParameterSet,
    objective_labels: Sequence[str],
    run_grouped_report: TransportReverseReportRunner,
    options: Mapping[str, object] | None = None,
    quiet_default: bool = True,
) -> TransportReverseReport:
    """Build a grouped realtime-geometry transport reverse report.

    This is the production-facing seam for the validated all-objective reverse
    path.  For now the heavy reverse execution is supplied as
    `run_grouped_report`; later extraction can move that runner into this
    module without changing the optimization backend contract.
    """

    _validate_transport_reverse_parameter_set(parameter_set)
    normalize_transport_objective_names(objective_names, objective_labels=objective_labels)
    quiet_option = quiet_default if options is None else options.get("quiet", quiet_default)
    quiet = bool(quiet_option)
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            return run_grouped_report()
    return run_grouped_report()


def run_realtime_geometry_transport_reverse_table(
    *,
    objective_names: Sequence[str],
    parameter_set: ReverseADParameterSet,
    objective_labels: Sequence[str],
    run_grouped_report: TransportReverseReportRunner,
    options: Mapping[str, object] | None = None,
    quiet_default: bool = True,
) -> TransportReverseReport:
    """Run the validated grouped realtime-geometry transport reverse table.

    This is the named internal execution seam for the production optimization
    lane. For realtime-geometry transport, the grouped report runner is the
    validated memory/graph behavior used by the benchmark. Direct table-result
    builders should only become the default after matching this path.
    """

    return build_realtime_geometry_transport_reverse_report(
        objective_names=objective_names,
        parameter_set=parameter_set,
        objective_labels=objective_labels,
        run_grouped_report=run_grouped_report,
        options=options,
        quiet_default=quiet_default,
    )


def build_realtime_geometry_transport_reverse_table_result(
    *,
    objective_names: Sequence[str],
    parameter_set: ReverseADParameterSet,
    objective_labels: Sequence[str],
    run_grouped_report: TransportReverseReportRunner,
    options: Mapping[str, object] | None = None,
    quiet_default: bool = True,
) -> RealtimeGeometryTransportReverseTableResult:
    """Run the validated grouped path and extract its JAX-native table result."""

    report = build_realtime_geometry_transport_reverse_report(
        objective_names=objective_names,
        parameter_set=parameter_set,
        objective_labels=objective_labels,
        run_grouped_report=run_grouped_report,
        options=options,
        quiet_default=quiet_default,
    )
    table_result = report.get("transport_reverse_table_result") if isinstance(report, Mapping) else None
    if not isinstance(table_result, RealtimeGeometryTransportReverseTableResult):
        raise TypeError(
            "Grouped realtime-geometry reverse runner did not return a "
            "RealtimeGeometryTransportReverseTableResult under "
            "'transport_reverse_table_result'."
        )
    return table_result


def transport_realtime_geometry_reverse_table(
    *,
    request: RealtimeGeometryTransportReverseTableRequest | None = None,
    objective_names: Sequence[str] | None = None,
    parameter_set: ReverseADParameterSet | None = None,
    context: RealtimeGeometryTransportReverseTableContext | None = None,
    table_result_builder: TransportReverseTableResultBuilder | None = None,
    run_grouped_report: TransportReverseReportRunner | None = None,
    objective_labels: Sequence[str] | None = None,
    options: Mapping[str, object] | None = None,
    quiet_default: bool = True,
) -> RealtimeGeometryTransportReverseTableResult:
    """Evaluate the realtime-geometry transport reverse table.

    This is the stable internal API boundary for optimization callers. For the
    realtime-geometry transport lane, `run_grouped_report` is currently the
    benchmark-matched path. A direct `table_result_builder` is still accepted
    for controlled experiments, but it is not the validated default.
    """

    if request is None:
        if objective_names is None:
            raise ValueError("objective_names must be provided when request is omitted.")
        if parameter_set is None:
            raise ValueError("parameter_set must be provided when request is omitted.")
        if context is None:
            raise ValueError("context must be provided when request is omitted.")
        request = realtime_geometry_transport_reverse_table_request(
            objective_names=objective_names,
            parameter_set=parameter_set,
            context=context,
            options=options,
        )
    else:
        if objective_names is not None or parameter_set is not None or context is not None:
            raise ValueError(
                "Pass either request or objective_names/parameter_set/context, not both."
            )
        if options is not None:
            request = dataclasses.replace(request, options=options)

    if table_result_builder is not None and run_grouped_report is not None:
        raise ValueError("Pass only one of table_result_builder or run_grouped_report.")
    if table_result_builder is not None:
        return table_result_builder(
            request.objective_names,
            request.parameter_set,
            request.options,
        )
    if run_grouped_report is None:
        raise ValueError("Either table_result_builder or run_grouped_report must be provided.")
    if objective_labels is None:
        raise ValueError("objective_labels must be provided with run_grouped_report.")
    return build_realtime_geometry_transport_reverse_table_result(
        objective_names=request.objective_names,
        parameter_set=request.parameter_set,
        objective_labels=objective_labels,
        run_grouped_report=run_grouped_report,
        options=request.options,
        quiet_default=quiet_default,
    )


def internal_realtime_geometry_transport_reverse_table_result_builder(
    *,
    table_context: RealtimeGeometryTransportReverseTableContext,
    geometry_context,
    baseline_geometry_deltas=None,
    combined_geometry_payload: bool = True,
    n_r: int = 51,
    n_theta: int = 25,
    n_zeta: int = 25,
    n_xi: int = 64,
    surface_backend: str = "vmec",
    max_iter=None,
    solver_device: str = "default",
    accepted_step_limit: int | None = None,
    reverse_segment_length: int | None = 1,
    initial_er_root_ad: str = "off",
    reverse_stage_adjoint_solve_mode: str = "bicgstab",
    reverse_rhs_transpose_mode: str = "explicit_ntx_interpolated",
    reverse_rhs_pullback_mode: str = "separate",
    reverse_initial_cache_support_pullback_mode: str = "scalar",
    reverse_rebuild_support_pullback_mode: str = "separate",
    reverse_segment_jit_diagnostics: bool = False,
    reverse_segment_input_diagnostics: bool = False,
    reverse_rebuild_component_timing: bool = False,
    reverse_phase_timing_diagnostics: bool = False,
    reverse_segment_profile_annotations: bool = False,
    reverse_segment_start_replay_mode: str = "legacy",
    reverse_segment_primal_record_mode: str = "reconstruct",
    reverse_final_objective_cotangent_mode: str = "scalar",
    reverse_bootstrap_cotangent_mode: str = "separate",
    reverse_stage_cotangent_mode: str = "full",
    reverse_step_bwd_mode: str = "reduced_cotangent",
    reverse_stage_adjoint_memory_mode: str = "default",
    reverse_stage_adjoint_iter_maxiter: int = 40,
    reverse_stage_adjoint_iter_tol: float = 1.0e-10,
    reverse_stage_adjoint_woodbury_rank: int = 24,
    reverse_schedule_artifact_mode: str = "legacy",
    max_reverse_accepted_steps: int | None = None,
    realtime_geometry_component_pullbacks: bool = False,
    progress_label: str | None = None,
    raw_block_solve: GeometryRawBlockSolve | None = None,
) -> TransportReverseTableResultBuilder:
    """Build an experimental direct full transport reverse table builder.

    This is useful for extraction work, but the grouped runner remains the
    validated optimization path until this builder is proven to match the
    benchmark memory behavior.
    """

    if not isinstance(table_context, RealtimeGeometryTransportReverseTableContext):
        raise TypeError("table_context must be a RealtimeGeometryTransportReverseTableContext.")
    baseline_profile_values = jnp.asarray(
        table_context.baseline_values[: len(TRANSPORT_REVERSE_PROFILE_PARAMETER_ORDER)]
    )

    def _row_indices(objective_names: Sequence[str]) -> tuple[int, ...]:
        labels = tuple(TRANSPORT_REVERSE_OBJECTIVE_LABELS)
        lookup = {name: i for i, name in enumerate(labels)}
        return tuple(lookup[str(name)] for name in normalize_transport_objective_names(objective_names, objective_labels=labels))

    def _builder(
        objective_names: tuple[str, ...],
        parameter_set: ReverseADParameterSet,
        options: Mapping[str, object] | None,
    ) -> RealtimeGeometryTransportReverseTableResult:
        _validate_transport_reverse_parameter_set(parameter_set)
        opts = {} if options is None else dict(options)
        timing_diagnostics = bool(opts.get("reverse_table_timing_diagnostics", False))
        table_builder_start = time.perf_counter()
        previous_phase_time = table_builder_start

        def _report_table_builder_phase(phase: str) -> None:
            nonlocal previous_phase_time
            if not timing_diagnostics:
                return
            now = time.perf_counter()
            prefix = progress_label or "[autodiff-gate]"
            print(
                f"{prefix} timing: phase=table_builder.{phase} "
                f"elapsed_s={now - previous_phase_time:.3f} "
                f"since_builder_start_s={now - table_builder_start:.3f} "
                f"gap_since_previous_s={now - previous_phase_time:.3f}",
                flush=True,
            )
            previous_phase_time = now

        active_accepted_step_limit = opts.get("accepted_step_limit", accepted_step_limit)
        active_reverse_segment_length = opts.get("reverse_segment_length", reverse_segment_length)
        active_max_reverse_accepted_steps = opts.get(
            "max_reverse_accepted_steps",
            max_reverse_accepted_steps,
        )
        active_initial_er_root_ad = str(opts.get("initial_er_root_ad", initial_er_root_ad))
        active_raw_block_solve = opts.get("raw_block_solve", raw_block_solve)
        active_component_pullbacks = bool(
            opts.get(
                "realtime_geometry_component_pullbacks",
                realtime_geometry_component_pullbacks,
            )
        )
        active_profile_values = jnp.asarray(
            opts.get("profile_values", baseline_profile_values),
            dtype=baseline_profile_values.dtype,
        )
        active_runtime = table_context.baseline_runtime
        active_support_payload = None
        use_runtime_payload = bool(opts.get("use_runtime_payload", active_raw_block_solve is None))
        _report_table_builder_phase("prepare_builder_inputs")
        if active_raw_block_solve is not None and not use_runtime_payload:
            active_support_payload = build_neopax_geometry_and_ntx_exact_lij_support_from_state(
                geometry_context,
                active_raw_block_solve.state,
                n_r=int(opts.get("n_r", n_r)),
                n_theta=int(opts.get("n_theta", n_theta)),
                n_zeta=int(opts.get("n_zeta", n_zeta)),
                n_xi=int(opts.get("n_xi", n_xi)),
                surface_backend=str(opts.get("surface_backend", surface_backend)),
            )
            active_runtime = runtime_with_geometry_payload(
                active_runtime,
                active_support_payload["geometry"],
            )
            active_runtime = runtime_with_ntx_support_payload(
                active_runtime,
                active_support_payload["ntx_support"],
            )
        # Keep the heavy retained scan record outside all generic segment
        # VJPs.  The database itself remains in the support payload, while
        # the original runtime below is retained solely for the final one-time
        # database-to-scan transpose.
        recorded_scan_runtime = active_runtime
        active_runtime = runtime_without_recorded_ntx_scan_primal(active_runtime)
        _report_table_builder_phase("prepare_runtime_payload")
        active_reverse_setup = prepare_reverse_static_setup(
            active_profile_values,
            config=table_context.config,
            runtime=active_runtime,
            baseline_state=table_context.baseline_state,
            profile_cfg=table_context.profile_cfg,
            initial_er_root_ad=active_initial_er_root_ad,
            accepted_step_limit_override=(
                None if active_accepted_step_limit is None else int(active_accepted_step_limit)
            ),
            reverse_segment_length=(
                None
                if active_reverse_segment_length is None
                else active_reverse_segment_length
                if isinstance(active_reverse_segment_length, str)
                else int(active_reverse_segment_length)
            ),
            reverse_direct_stage_adjoint=True,
            reverse_stage_adjoint_solve_mode=str(
                opts.get("reverse_stage_adjoint_solve_mode", reverse_stage_adjoint_solve_mode)
            ),
            reverse_rhs_transpose_mode=str(opts.get("reverse_rhs_transpose_mode", reverse_rhs_transpose_mode)),
            reverse_rhs_pullback_mode=str(opts.get("reverse_rhs_pullback_mode", reverse_rhs_pullback_mode)),
            reverse_initial_cache_support_pullback_mode=str(
                opts.get(
                    "reverse_initial_cache_support_pullback_mode",
                    reverse_initial_cache_support_pullback_mode,
                )
            ),
            reverse_rebuild_support_pullback_mode=str(
                opts.get(
                    "reverse_rebuild_support_pullback_mode",
                    reverse_rebuild_support_pullback_mode,
                )
            ),
            reverse_segment_jit_diagnostics=bool(
                opts.get("reverse_segment_jit_diagnostics", reverse_segment_jit_diagnostics)
            ),
            reverse_segment_input_diagnostics=bool(
                opts.get("reverse_segment_input_diagnostics", reverse_segment_input_diagnostics)
            ),
            reverse_rebuild_component_timing=bool(
                opts.get("reverse_rebuild_component_timing", reverse_rebuild_component_timing)
            ),
            reverse_phase_timing_diagnostics=bool(
                opts.get("reverse_phase_timing_diagnostics", reverse_phase_timing_diagnostics)
            ),
            reverse_segment_profile_annotations=bool(
                opts.get(
                    "reverse_segment_profile_annotations",
                    reverse_segment_profile_annotations,
                )
            ),
            reverse_segment_start_replay_mode=str(
                opts.get("reverse_segment_start_replay_mode", reverse_segment_start_replay_mode)
            ),
            reverse_segment_primal_record_mode=str(
                opts.get(
                    "reverse_segment_primal_record_mode",
                    reverse_segment_primal_record_mode,
                )
            ),
            reverse_final_objective_cotangent_mode=str(
                opts.get(
                    "reverse_final_objective_cotangent_mode",
                    reverse_final_objective_cotangent_mode,
                )
            ),
            reverse_bootstrap_cotangent_mode=str(
                opts.get(
                    "reverse_bootstrap_cotangent_mode",
                    reverse_bootstrap_cotangent_mode,
                )
            ),
            reverse_stage_cotangent_mode=str(opts.get("reverse_stage_cotangent_mode", reverse_stage_cotangent_mode)),
            reverse_step_bwd_mode=str(opts.get("reverse_step_bwd_mode", reverse_step_bwd_mode)),
            reverse_stage_adjoint_memory_mode=str(
                opts.get("reverse_stage_adjoint_memory_mode", reverse_stage_adjoint_memory_mode)
            ),
            reverse_stage_adjoint_iter_maxiter=int(
                opts.get("reverse_stage_adjoint_iter_maxiter", reverse_stage_adjoint_iter_maxiter)
            ),
            reverse_stage_adjoint_iter_tol=float(
                opts.get("reverse_stage_adjoint_iter_tol", reverse_stage_adjoint_iter_tol)
            ),
            reverse_stage_adjoint_woodbury_rank=int(
                opts.get("reverse_stage_adjoint_woodbury_rank", reverse_stage_adjoint_woodbury_rank)
            ),
            reverse_schedule_artifact_mode=str(
                opts.get("reverse_schedule_artifact_mode", reverse_schedule_artifact_mode)
            ),
            max_reverse_accepted_steps=(
                None
                if active_max_reverse_accepted_steps is None
                else int(active_max_reverse_accepted_steps)
            ),
        )
        _report_table_builder_phase("prepare_reverse_static_setup")
        if (
            str(opts.get("reverse_schedule_artifact_mode", reverse_schedule_artifact_mode))
            .strip()
            .lower()
            == "reuse_static_probe"
            and active_reverse_setup.schedule_artifact is None
        ):
            raise RuntimeError(
                "reuse_static_probe was requested, but the internal table builder returned no schedule artifact."
            )
        if active_support_payload is not None:
            # A raw-block state supplies the established exact-Lij payload.
            ntx_support_payload = active_support_payload["ntx_support"]
            support_payload = (
                {
                    "geometry": active_support_payload["geometry"],
                    "ntx_support": ntx_support_payload,
                }
                if combined_geometry_payload
                else ntx_support_payload
            )
        elif str(realtime_geometry_payload_for_runtime(recorded_scan_runtime)["kind"]) == "ntx_scan_runtime":
            # A live scan model owns no prepared exact-NTX support tree.  Its
            # differentiable inputs are geometry, channels and scan surfaces;
            # the recorded route additionally exposes its already-built
            # interpolation database as a table-only support leaf.
            if not combined_geometry_payload:
                raise ValueError(
                    "ntx_scan_runtime reverse requires the combined realtime "
                    "geometry payload."
                )
            ntx_support_payload = None
            support_payload = realtime_geometry_reverse_support_payload_for_runtime(
                recorded_scan_runtime
            )
        else:
            ntx_support_payload = find_ntx_support_payload(active_runtime)
            support_payload = (
                {
                    "geometry": active_runtime.geometry,
                    "ntx_support": ntx_support_payload,
                }
                if combined_geometry_payload
                else ntx_support_payload
            )
        _report_table_builder_phase("prepare_support_payload")
        support_result = realtime_geometry_support_cotangents_from_parameter_vector(
            profile_values=active_profile_values,
            config=table_context.config,
            baseline_runtime=active_runtime,
            baseline_state=table_context.baseline_state,
            profile_cfg=table_context.profile_cfg,
            reverse_setup=active_reverse_setup,
            support_payload=support_payload,
            initial_er_root_ad=active_initial_er_root_ad,
        )
        support_result = jax.block_until_ready(support_result)
        _report_table_builder_phase("transport_support_cotangents")
        rows = _row_indices(objective_names)
        support_bars = tuple(support_result.support_bars[i] for i in rows)
        native_vmec_face_coefficient_bars = (
            None
            if support_result.native_vmec_face_coefficient_bars is None
            else jax.tree_util.tree_map(
                lambda value: jnp.asarray(value)[jnp.asarray(rows, dtype=jnp.int32)],
                support_result.native_vmec_face_coefficient_bars,
            )
        )
        component_bars = {
            name: tuple(values[i] for i in rows)
            for name, values in support_result.support_component_bars_by_name.items()
        }
        # The recorded scan route accumulates a database bar during all
        # objective/segment VJPs.  Fold it once here, before the ordinary
        # VMEC payload transpose.  Legacy payloads have no database leaf and
        # are returned unchanged by the helper.
        component_names = tuple(component_bars)
        folded_groups = fold_recorded_ntx_scan_database_bar_groups_into_support(
            recorded_scan_runtime,
            (support_bars, *(component_bars[name] for name in component_names)),
        )
        support_bars = folded_groups[0]
        component_bars = {
            name: folded_groups[index + 1]
            for index, name in enumerate(component_names)
        }
        all_objective_values = jnp.asarray(support_result.objective_values)
        all_profile_gradient_matrix = jnp.asarray(support_result.profile_gradient_matrix)
        if int(all_profile_gradient_matrix.shape[1]) == len(PROFILE_PARAMETER_ORDER):
            all_profile_parameter_labels = PROFILE_PARAMETER_ORDER
        else:
            all_profile_parameter_labels = TRANSPORT_REVERSE_PROFILE_PARAMETER_ORDER
        profile_lookup = {
            name: i for i, name in enumerate(all_profile_parameter_labels)
        }
        profile_cols = tuple(profile_lookup[spec.name] for spec in parameter_set.profile_specs)
        selected_objective_values = all_objective_values[jnp.asarray(rows, dtype=jnp.int32)]
        selected_profile_matrix = all_profile_gradient_matrix[
            jnp.asarray(rows, dtype=jnp.int32),
            :,
        ]
        if profile_cols:
            selected_profile_matrix = selected_profile_matrix[:, jnp.asarray(profile_cols, dtype=jnp.int32)]
        else:
            selected_profile_matrix = selected_profile_matrix[:, :0]
        vmec_specs = tuple(spec.as_tuple() for spec in parameter_set.vmec_boundary_specs)
        if baseline_geometry_deltas is None:
            active_baseline_geometry_deltas = jnp.zeros((len(vmec_specs),), dtype=jnp.float64)
        else:
            active_baseline_geometry_deltas = jnp.asarray(baseline_geometry_deltas, dtype=jnp.float64)
        _report_table_builder_phase("prepare_transport_table_inputs")
        assembly = realtime_geometry_transport_reverse_table_from_payload_cotangents(
            objective_labels=objective_names,
            profile_parameter_labels=tuple(spec.label for spec in parameter_set.profile_specs),
            geometry_parameter_labels=tuple(spec.vmec_label for spec in parameter_set.vmec_boundary_specs),
            objective_values=selected_objective_values,
            profile_gradient_matrix=selected_profile_matrix,
            geometry_context=geometry_context,
            baseline_geometry_deltas=active_baseline_geometry_deltas,
            geometry_param_specs=vmec_specs,
            support_bars=support_bars,
            support_component_bars_by_name=component_bars,
            native_vmec_face_coefficient_bars=native_vmec_face_coefficient_bars,
            include_component_pullbacks=active_component_pullbacks,
            combined_geometry_payload=combined_geometry_payload,
            n_r=int(opts.get("n_r", n_r)),
            n_theta=int(opts.get("n_theta", n_theta)),
            n_zeta=int(opts.get("n_zeta", n_zeta)),
            n_xi=int(opts.get("n_xi", n_xi)),
            surface_backend=str(opts.get("surface_backend", surface_backend)),
            max_iter=opts.get("max_iter", max_iter),
            solver_device=str(opts.get("solver_device", solver_device)),
            progress_label=progress_label,
            return_branch_gradients=bool(opts.get("return_branch_gradients", False)),
            raw_block_solve=active_raw_block_solve,
        )
        _report_table_builder_phase("transport_payload_to_geometry_table")
        if active_component_pullbacks:
            component_matrices = assembly.payload_pullback_result.component_gradient_matrices
            geometry_labels = tuple(spec.vmec_label for spec in parameter_set.vmec_boundary_specs)
            prefix = progress_label or "[autodiff-gate]"
            print(
                f"{prefix} diagnostic: reverse geometry component pullbacks "
                "(compare objective_explicit to FD fd_explicit_geometry and "
                "final_state_components_sum to FD fd_final_state_geometry)",
                flush=True,
            )
            for objective_i, objective_name in enumerate(objective_names):
                for geometry_i, geometry_label in enumerate(geometry_labels):
                    components = {
                        name: float(jax.device_get(matrix[objective_i, geometry_i]))
                        for name, matrix in component_matrices.items()
                    }
                    dynamic_sum = sum(
                        value for name, value in components.items()
                        if name != "objective_explicit"
                    )
                    component_text = " ".join(
                        f"{name}={value:.6e}"
                        for name, value in components.items()
                    )
                    print(
                        f"{prefix} diagnostic: objective={objective_name} "
                        f"parameter={geometry_label} {component_text} "
                        f"final_state_components_sum={dynamic_sum:.6e}",
                        flush=True,
                    )
        return assembly.table_result

    return _builder


def realtime_geometry_payload_pullback_result(
    *,
    geometry_context,
    baseline_geometry_deltas,
    geometry_param_specs: Sequence[tuple[str, int, int]],
    support_bars: Sequence[object],
    support_component_bars_by_name: Mapping[str, Sequence[object]] | None = None,
    native_vmec_face_coefficient_bars: Mapping[str, object] | None = None,
    include_component_pullbacks: bool = False,
    combined_geometry_payload: bool = True,
    payload_kind: str = "ntx_exact",
    scan_rho=None,
    scan_surface_backend: str = "vmec",
    n_r: int = 51,
    n_theta: int = 25,
    n_zeta: int = 25,
    n_xi: int = 64,
    surface_backend: str = "booz",
    max_iter=None,
    solver_device: str = "default",
    progress_label: str | None = None,
    raw_block_solve: GeometryRawBlockSolve | None = None,
    return_branch_gradients: bool = True,
    dispatch_cache_probe=None,
    prepared_payload_static=None,
    prepared_active_payload_leaves=None,
    return_gradient_matrix: bool = False,
) -> RealtimeGeometryPayloadPullbackResult:
    """Pull transport support-payload cotangents back to VMEC boundary harmonics.

    This is the raw-block-transpose path validated by the geometry benchmarks.
    It only moves the benchmark's existing orchestration into an internal helper;
    callers still supply the already-computed support cotangents.
    """

    component_bars = {} if support_component_bars_by_name is None else dict(support_component_bars_by_name)
    support_component_names = tuple(component_bars) if include_component_pullbacks else tuple()
    geometry_pullback_payload_bars = tuple(support_bars)
    for component_name in support_component_names:
        geometry_pullback_payload_bars = (
            *geometry_pullback_payload_bars,
            *tuple(component_bars[component_name]),
        )

    # Component reports add payload RHS rows, whereas the native coefficient
    # route currently represents the total transport-support transpose only.
    # Pad those diagnostic-only component rows with zeros so the raw-block
    # bridge retains its one-RHS-per-payload-row contract.
    if native_vmec_face_coefficient_bars is not None and support_component_names:
        native_vmec_face_coefficient_bars = jax.tree_util.tree_map(
            lambda value: jnp.concatenate(
                (
                    jnp.asarray(value),
                    jnp.zeros(
                        (
                            len(geometry_pullback_payload_bars) - len(tuple(support_bars)),
                        )
                        + jnp.asarray(value).shape[1:],
                        dtype=jnp.asarray(value).dtype,
                    ),
                ),
                axis=0,
            ),
            native_vmec_face_coefficient_bars,
        )

    geometry_gradient_result = geometry_payload_pullback_from_param_vector_raw_block_transpose(
        geometry_context,
        baseline_geometry_deltas,
        tuple(geometry_param_specs),
        geometry_pullback_payload_bars,
        combined_payload=combined_geometry_payload,
        payload_kind=payload_kind,
        scan_rho=scan_rho,
        scan_surface_backend=scan_surface_backend,
        n_r=int(n_r),
        n_theta=int(n_theta),
        n_zeta=int(n_zeta),
        n_xi=int(n_xi),
        surface_backend=surface_backend,
        max_iter=max_iter,
        solver_device=str(solver_device),
        progress_label=progress_label,
        return_branch_gradients=bool(return_branch_gradients),
        raw_block_solve=raw_block_solve,
        native_vmec_face_coefficient_bars=native_vmec_face_coefficient_bars,
        dispatch_cache_probe=dispatch_cache_probe,
        prepared_static=prepared_payload_static,
        prepared_active_payload_leaves=prepared_active_payload_leaves,
    )
    geometry_gradient_result = jax.block_until_ready(geometry_gradient_result)
    objective_count = int(len(tuple(support_bars)))

    def _split_component_rows(matrix):
        if matrix is None:
            return None, {}
        total_matrix = matrix[:objective_count]
        component_matrices = {}
        row0 = objective_count
        for component_name in support_component_names:
            row1 = row0 + objective_count
            component_matrices[component_name] = matrix[row0:row1]
            row0 = row1
        return total_matrix, component_matrices

    if isinstance(geometry_gradient_result, Mapping):
        # The public report predates the live scan and keeps the historical
        # ``ntx_support_branch`` field name.  Select the actual internal
        # payload branch here; the combined matrix remains the authoritative
        # geometry derivative in both cases.
        support_branch_key = (
            "ntx_scan_runtime"
            if str(payload_kind).strip().lower() == "ntx_scan_runtime"
            else "ntx_support"
        )
        geometry_gradient_matrix, component_gradient_matrices = _split_component_rows(
            geometry_gradient_result["combined"]
        )
        geometry_branch_gradient_matrix, component_geometry_branch_matrices = _split_component_rows(
            geometry_gradient_result.get("geometry")
        )
        ntx_support_branch_gradient_matrix, component_ntx_support_branch_matrices = _split_component_rows(
            geometry_gradient_result.get(support_branch_key)
        )
    else:
        geometry_gradient_matrix, component_gradient_matrices = _split_component_rows(
            geometry_gradient_result
        )
        geometry_branch_gradient_matrix = None
        ntx_support_branch_gradient_matrix = None
        component_geometry_branch_matrices = {}
        component_ntx_support_branch_matrices = {}

    if return_gradient_matrix:
        # Optimization-only numerical boundary: returning the raw matrix
        # avoids constructing host/report dataclasses inside a JIT. The
        # default public path still returns the unchanged rich result below.
        return geometry_gradient_matrix

    return RealtimeGeometryPayloadPullbackResult(
        geometry_gradient_matrix=geometry_gradient_matrix,
        geometry_branch_gradient_matrix=geometry_branch_gradient_matrix,
        ntx_support_branch_gradient_matrix=ntx_support_branch_gradient_matrix,
        component_gradient_matrices=component_gradient_matrices,
        component_geometry_branch_matrices=component_geometry_branch_matrices,
        component_ntx_support_branch_matrices=component_ntx_support_branch_matrices,
    )


def realtime_geometry_transport_reverse_table_from_payload_cotangents(
    *,
    objective_labels: Sequence[str],
    profile_parameter_labels: Sequence[str],
    geometry_parameter_labels: Sequence[str],
    objective_values,
    profile_gradient_matrix,
    geometry_context,
    baseline_geometry_deltas,
    geometry_param_specs: Sequence[tuple[str, int, int]],
    support_bars: Sequence[object],
    support_component_bars_by_name: Mapping[str, Sequence[object]] | None = None,
    native_vmec_face_coefficient_bars: Mapping[str, object] | None = None,
    include_component_pullbacks: bool = False,
    combined_geometry_payload: bool = True,
    payload_kind: str = "ntx_exact",
    scan_rho=None,
    scan_surface_backend: str = "vmec",
    n_r: int = 51,
    n_theta: int = 25,
    n_zeta: int = 25,
    n_xi: int = 64,
    surface_backend: str = "booz",
    max_iter=None,
    solver_device: str = "default",
    progress_label: str | None = None,
    raw_block_solve: GeometryRawBlockSolve | None = None,
    return_branch_gradients: bool = True,
    dispatch_cache_probe=None,
    prepared_payload_static=None,
    prepared_active_payload_leaves=None,
    return_raw_matrices: bool = False,
) -> RealtimeGeometryTransportReverseAssemblyResult:
    """Assemble the JAX transport reverse table from support-payload cotangents."""

    payload_pullback_result = realtime_geometry_payload_pullback_result(
        geometry_context=geometry_context,
        baseline_geometry_deltas=baseline_geometry_deltas,
        geometry_param_specs=geometry_param_specs,
        support_bars=support_bars,
        support_component_bars_by_name=support_component_bars_by_name,
        native_vmec_face_coefficient_bars=native_vmec_face_coefficient_bars,
        include_component_pullbacks=include_component_pullbacks,
        combined_geometry_payload=combined_geometry_payload,
        payload_kind=payload_kind,
        scan_rho=scan_rho,
        scan_surface_backend=scan_surface_backend,
        n_r=n_r,
        n_theta=n_theta,
        n_zeta=n_zeta,
        n_xi=n_xi,
        surface_backend=surface_backend,
        max_iter=max_iter,
        solver_device=solver_device,
        progress_label=progress_label,
        raw_block_solve=raw_block_solve,
        return_branch_gradients=bool(return_branch_gradients),
        dispatch_cache_probe=dispatch_cache_probe,
        prepared_payload_static=prepared_payload_static,
        prepared_active_payload_leaves=prepared_active_payload_leaves,
        return_gradient_matrix=return_raw_matrices,
    )
    if return_raw_matrices:
        return (
            jnp.asarray(objective_values),
            jnp.asarray(profile_gradient_matrix),
            jnp.asarray(payload_pullback_result),
        )
    table_result = realtime_geometry_transport_reverse_table_result(
        objective_labels=objective_labels,
        profile_parameter_labels=profile_parameter_labels,
        geometry_parameter_labels=geometry_parameter_labels,
        objective_values=objective_values,
        profile_gradient_matrix=profile_gradient_matrix,
        geometry_gradient_matrix=payload_pullback_result.geometry_gradient_matrix,
    )
    return RealtimeGeometryTransportReverseAssemblyResult(
        table_result=table_result,
        payload_pullback_result=payload_pullback_result,
    )


def transport_reverse_table_report_entries(
    *,
    table_result: RealtimeGeometryTransportReverseTableResult | None = None,
    objective_labels: Sequence[str] | None = None,
    profile_parameter_labels: Sequence[str] | None = None,
    geometry_parameter_labels: Sequence[str] | None = None,
    objective_values=None,
    profile_gradient_matrix=None,
    geometry_gradient_matrix=None,
) -> dict[str, object]:
    """Assemble reusable non-printing transport reverse table report entries."""

    if table_result is not None:
        objective_labels = table_result.objective_labels
        profile_parameter_labels = table_result.profile_parameter_labels
        geometry_parameter_labels = table_result.geometry_parameter_labels
        objective_values = table_result.objective_values
        profile_gradient_matrix = table_result.profile_gradient_matrix
        geometry_gradient_matrix = table_result.geometry_gradient_matrix
    if (
        objective_labels is None
        or profile_parameter_labels is None
        or geometry_parameter_labels is None
        or objective_values is None
        or profile_gradient_matrix is None
        or geometry_gradient_matrix is None
    ):
        raise TypeError(
            "transport_reverse_table_report_entries requires either table_result or "
            "all explicit labels and arrays."
        )
    objective_names = tuple(str(name) for name in objective_labels)
    profile_names = tuple(str(name) for name in profile_parameter_labels)
    geometry_names = tuple(str(name) for name in geometry_parameter_labels)
    if len(set(objective_names)) != len(objective_names):
        raise ValueError(f"objective_labels must be unique; got {objective_names!r}.")
    if len(set(profile_names)) != len(profile_names):
        raise ValueError(f"profile_parameter_labels must be unique; got {profile_names!r}.")
    if len(set(geometry_names)) != len(geometry_names):
        raise ValueError(f"geometry_parameter_labels must be unique; got {geometry_names!r}.")
    objective_values_np = np.asarray(jax.device_get(objective_values), dtype=float)
    profile_gradient_np = np.asarray(jax.device_get(profile_gradient_matrix), dtype=float)
    geometry_gradient_np = np.asarray(jax.device_get(geometry_gradient_matrix), dtype=float)

    objective_count = len(objective_names)
    if objective_values_np.shape != (objective_count,):
        raise ValueError(
            "objective_values must have shape (objective_count,); "
            f"got {objective_values_np.shape}, objective_count={objective_count}."
        )
    if profile_gradient_np.shape != (objective_count, len(profile_names)):
        raise ValueError(
            "profile_gradient_matrix must have shape "
            "(objective_count, profile_parameter_count); "
            f"got {profile_gradient_np.shape}, expected {(objective_count, len(profile_names))}."
        )
    if geometry_gradient_np.shape != (objective_count, len(geometry_names)):
        raise ValueError(
            "geometry_gradient_matrix must have shape "
            "(objective_count, geometry_parameter_count); "
            f"got {geometry_gradient_np.shape}, expected {(objective_count, len(geometry_names))}."
        )

    objective_finite_np = np.isfinite(objective_values_np)
    return {
        "objective_values": {
            name: float(value)
            for name, value in zip(objective_names, objective_values_np.tolist())
        },
        "objective_finite": {
            name: bool(value)
            for name, value in zip(objective_names, objective_finite_np.tolist())
        },
        "profile_gradient_all_finite_by_objective": {
            objective_name: bool(np.all(np.isfinite(profile_gradient_np[objective_i])))
            for objective_i, objective_name in enumerate(objective_names)
        },
        "geometry_gradient_all_finite_by_objective": {
            objective_name: bool(np.all(np.isfinite(geometry_gradient_np[objective_i])))
            for objective_i, objective_name in enumerate(objective_names)
        },
        "profile_gradient_reverse_ad": {
            objective_name: {
                parameter_name: float(value)
                for parameter_name, value in zip(
                    profile_names,
                    profile_gradient_np[objective_i].tolist(),
                )
            }
            for objective_i, objective_name in enumerate(objective_names)
        },
        "geometry_gradient_reverse_ad": {
            objective_name: {
                geometry_label: float(value)
                for geometry_label, value in zip(
                    geometry_names,
                    geometry_gradient_np[objective_i].tolist(),
                )
            }
            for objective_i, objective_name in enumerate(objective_names)
        },
    }


def realtime_geometry_transport_reverse_metadata_entries(
    *,
    parameter_mode: str,
    config_path: str,
    objective_labels: Sequence[str],
    profile_parameter_labels: Sequence[str],
    profile_values,
    geometry_parameter_labels: Sequence[str],
    geometry_parameter_entries: Sequence[Mapping[str, object]],
    baseline_geometry_deltas,
    geometry_parameter_specs: Sequence[tuple[str, int, int]],
    geometry_parameter_selector: str,
    accepted_step_limit: int | None,
    reverse_segment_length: int | None,
    reverse_stage_cotangent_mode_requested: str,
    reverse_stage_cotangent_mode_effective: str,
    ntx_exact_derivative_mode: str,
    ntx_exact_derivative_field_pullback_mode: str,
    ntx_exact_surface_backend: str,
    realtime_geometry_gradient_path: str,
    realtime_geometry_component_pullbacks: bool,
    realtime_geometry_support_bar_diagnostics_skipped: bool,
    realtime_geometry_derivative_complete: bool,
    geometry_support_pullback_mode: str,
    realtime_geometry_diagnostics: Mapping[str, object],
    support_payload_summary: Mapping[str, object],
    support_bar_summary_by_objective: Mapping[str, object],
    support_bar_l2_by_objective: Mapping[str, object],
    support_bar_branch_diagnostics_by_objective: Mapping[str, object],
    support_reuse_count: int,
    support_rebuild_count: int,
    support_initial_cache_pullback_used: bool,
    support_initial_cache_pullback_skipped: bool,
    elapsed_s: float,
) -> dict[str, object]:
    """Assemble generic realtime-geometry transport reverse report metadata."""

    objective_names = tuple(str(name) for name in objective_labels)
    profile_names = tuple(str(name) for name in profile_parameter_labels)
    geometry_names = tuple(str(name) for name in geometry_parameter_labels)
    geometry_deltas = np.asarray(jax.device_get(baseline_geometry_deltas), dtype=float).tolist()
    if len(geometry_names) != len(geometry_parameter_entries) or len(geometry_names) != len(geometry_deltas):
        raise ValueError(
            "Geometry parameter labels, entries, and baseline deltas must have matching lengths; "
            f"got labels={len(geometry_names)}, entries={len(geometry_parameter_entries)}, "
            f"deltas={len(geometry_deltas)}."
        )
    return {
        "mode": "transport_reverse_ad_only",
        "parameter_mode": str(parameter_mode),
        "config_path": str(config_path),
        "objective_name": "all",
        "objective_order": list(objective_names),
        "parameter_order": list(profile_names),
        "profile_baseline_values": np.asarray(jax.device_get(profile_values), dtype=float).tolist(),
        "geometry_baseline_values": {
            geometry_label: float(entry["baseline_coefficient"]) + float(delta)
            for geometry_label, entry, delta in zip(
                geometry_names,
                geometry_parameter_entries,
                geometry_deltas,
            )
        },
        "geometry_parameter_order": list(geometry_names),
        "geometry_parameter_specs": [
            {"family": family, "m": int(m), "n": int(n)}
            for family, m, n in geometry_parameter_specs
        ],
        "geometry_parameter_selector": str(geometry_parameter_selector),
        "geometry_parameter_count": int(len(geometry_parameter_specs)),
        "accepted_step_limit": None if accepted_step_limit is None else int(accepted_step_limit),
        "reverse_segment_length": None if reverse_segment_length is None else int(reverse_segment_length),
        "reverse_stage_cotangent_mode_requested": str(reverse_stage_cotangent_mode_requested),
        "reverse_stage_cotangent_mode_effective": str(reverse_stage_cotangent_mode_effective),
        "ntx_exact_derivative_mode": str(ntx_exact_derivative_mode),
        "ntx_exact_derivative_field_pullback_mode": str(ntx_exact_derivative_field_pullback_mode),
        "ntx_exact_surface_backend": str(ntx_exact_surface_backend),
        "realtime_geometry_gradient_path": str(realtime_geometry_gradient_path),
        "realtime_geometry_component_pullbacks": bool(realtime_geometry_component_pullbacks),
        "realtime_geometry_support_bar_diagnostics_skipped": bool(
            realtime_geometry_support_bar_diagnostics_skipped
        ),
        "realtime_primal_runtime_builder": "build_runtime_context",
        "realtime_geometry_derivative_boundary": (
            "runtime_geometry_and_ntx_exact_lij_support_payload"
            if realtime_geometry_derivative_complete
            else "ntx_exact_lij_support_payload_only_diagnostic"
        ),
        "realtime_geometry_derivative_complete": bool(realtime_geometry_derivative_complete),
        "geometry_support_pullback_mode": str(geometry_support_pullback_mode),
        "realtime_geometry_support_bwd_mode": "grouped_batched_fused_support",
        "realtime_geometry_diagnostics": realtime_geometry_diagnostics,
        "support_payload_summary": support_payload_summary,
        "support_bar_summary_by_objective": support_bar_summary_by_objective,
        "support_bar_l2_by_objective": support_bar_l2_by_objective,
        "support_bar_branch_diagnostics_by_objective": support_bar_branch_diagnostics_by_objective,
        "support_reuse_count": int(support_reuse_count),
        "support_rebuild_count": int(support_rebuild_count),
        "support_initial_cache_pullback_used": bool(support_initial_cache_pullback_used),
        "support_initial_cache_pullback_skipped": bool(support_initial_cache_pullback_skipped),
        "elapsed_s": float(elapsed_s),
    }


def realtime_geometry_transport_reverse_diagnostic_gradient_entries(
    *,
    objective_labels: Sequence[str],
    geometry_parameter_labels: Sequence[str],
    geometry_gradient_matrix_np,
    geometry_branch_gradient_matrix_np=None,
    ntx_support_branch_gradient_matrix_np=None,
    component_gradient_np_by_name: Mapping[str, object] | None = None,
    component_geometry_branch_np_by_name: Mapping[str, object] | None = None,
    component_ntx_support_branch_np_by_name: Mapping[str, object] | None = None,
    include_component_pullbacks: bool = False,
) -> dict[str, object]:
    """Assemble branch/component geometry-gradient diagnostics from host arrays."""

    objective_names = tuple(str(name) for name in objective_labels)
    geometry_names = tuple(str(name) for name in geometry_parameter_labels)
    component_gradient_np_by_name = (
        {} if component_gradient_np_by_name is None else component_gradient_np_by_name
    )
    component_geometry_branch_np_by_name = (
        {} if component_geometry_branch_np_by_name is None else component_geometry_branch_np_by_name
    )
    component_ntx_support_branch_np_by_name = (
        {} if component_ntx_support_branch_np_by_name is None else component_ntx_support_branch_np_by_name
    )
    if geometry_branch_gradient_matrix_np is None or ntx_support_branch_gradient_matrix_np is None:
        branch_entries = None
    else:
        branch_entries = {
            objective_name: {
                "geometry": {
                    geometry_label: float(value)
                    for geometry_label, value in zip(
                        geometry_names,
                        geometry_branch_gradient_matrix_np[objective_i].tolist(),
                    )
                },
                "ntx_support": {
                    geometry_label: float(value)
                    for geometry_label, value in zip(
                        geometry_names,
                        ntx_support_branch_gradient_matrix_np[objective_i].tolist(),
                    )
                },
                "combined": {
                    geometry_label: float(value)
                    for geometry_label, value in zip(
                        geometry_names,
                        geometry_gradient_matrix_np[objective_i].tolist(),
                    )
                },
            }
            for objective_i, objective_name in enumerate(objective_names)
        }
    if not include_component_pullbacks:
        return {
            "geometry_gradient_reverse_ad_by_branch": branch_entries,
            "geometry_gradient_reverse_ad_by_component": {},
            "geometry_gradient_reverse_ad_final_state_components": {},
            "geometry_gradient_reverse_ad_by_component_and_branch": {},
        }
    component_entries = {
        objective_name: {
            component_name: {
                geometry_label: float(value)
                for geometry_label, value in zip(
                    geometry_names,
                    component_matrix[objective_i].tolist(),
                )
            }
            for component_name, component_matrix in component_gradient_np_by_name.items()
        }
        for objective_i, objective_name in enumerate(objective_names)
    }
    final_state_component_entries = {
        objective_name: {
            geometry_label: float(
                sum(
                    component_matrix[objective_i, geometry_i]
                    for component_name, component_matrix in component_gradient_np_by_name.items()
                    if component_name != "objective_explicit"
                )
            )
            for geometry_i, geometry_label in enumerate(geometry_names)
        }
        for objective_i, objective_name in enumerate(objective_names)
    }
    component_branch_entries = {
        objective_name: {
            component_name: {
                "geometry": {
                    geometry_label: float(value)
                    for geometry_label, value in zip(
                        geometry_names,
                        component_geometry_branch_np_by_name.get(component_name, component_matrix)[
                            objective_i
                        ].tolist(),
                    )
                },
                "ntx_support": {
                    geometry_label: float(value)
                    for geometry_label, value in zip(
                        geometry_names,
                        component_ntx_support_branch_np_by_name.get(component_name, component_matrix)[
                            objective_i
                        ].tolist(),
                    )
                },
                "combined": {
                    geometry_label: float(value)
                    for geometry_label, value in zip(
                        geometry_names,
                        component_matrix[objective_i].tolist(),
                    )
                },
            }
            for component_name, component_matrix in component_gradient_np_by_name.items()
        }
        for objective_i, objective_name in enumerate(objective_names)
    }
    return {
        "geometry_gradient_reverse_ad_by_branch": branch_entries,
        "geometry_gradient_reverse_ad_by_component": component_entries,
        "geometry_gradient_reverse_ad_final_state_components": final_state_component_entries,
        "geometry_gradient_reverse_ad_by_component_and_branch": component_branch_entries,
    }


def realtime_geometry_transport_reverse_grouped_runner(
    *,
    args,
    support_segment_executor: TransportReverseSupportSegmentExecutor,
    update_args_for_all_objectives: TransportReverseArgsUpdater,
) -> TransportReverseReportRunner:
    """Build the grouped all-objective runner used by transport reverse builders.

    The supplied executor still owns the benchmark-validated segmented reverse
    math. This wrapper owns the grouped contract: force objective='all', request
    an internal report return, and require a JAX-native table result.
    """

    if not callable(support_segment_executor):
        raise TypeError("support_segment_executor must be callable.")
    if not callable(update_args_for_all_objectives):
        raise TypeError("update_args_for_all_objectives must be callable.")

    def _run_grouped_report() -> TransportReverseReport:
        grouped_args = update_args_for_all_objectives(args)
        report = support_segment_executor(grouped_args, True)
        table_result = report.get("transport_reverse_table_result") if isinstance(report, Mapping) else None
        if not isinstance(table_result, RealtimeGeometryTransportReverseTableResult):
            raise TypeError(
                "Grouped realtime-geometry reverse executor did not return a "
                "RealtimeGeometryTransportReverseTableResult under "
                "'transport_reverse_table_result'."
            )
        return report

    return _run_grouped_report


def grouped_transport_reverse_report_builder(
    *,
    objective_labels: Sequence[str],
    run_grouped_report: TransportReverseReportRunner,
    table_context: RealtimeGeometryTransportReverseTableContext | None = None,
    quiet_default: bool = True,
) -> TransportReverseReportBuilder:
    """Build a grouped transport report builder around a validated runner.

    `run_grouped_report` should execute the validated all-objective reverse
    path and return its report dictionary. Objective selection is handled later
    by the table adapter; this builder only validates that all requested names
    exist and controls whether benchmark-style progress output is suppressed.
    """

    labels = tuple(str(name) for name in objective_labels)

    def _builder(
        objective_names: tuple[str, ...],
        parameter_set: ReverseADParameterSet,
        options: Mapping[str, object] | None,
    ) -> TransportReverseReport:
        if table_context is not None:
            request = realtime_geometry_transport_reverse_table_request(
                objective_names=objective_names,
                parameter_set=parameter_set,
                context=table_context,
                options=options,
            )
            objective_names = request.objective_names
            parameter_set = request.parameter_set
            options = request.options
        return run_realtime_geometry_transport_reverse_table(
            objective_names=objective_names,
            parameter_set=parameter_set,
            objective_labels=labels,
            run_grouped_report=run_grouped_report,
            options=options,
            quiet_default=quiet_default,
        )

    return _builder


def grouped_transport_reverse_table_result_builder(
    *,
    objective_labels: Sequence[str],
    run_grouped_report: TransportReverseReportRunner,
    table_context: RealtimeGeometryTransportReverseTableContext | None = None,
    quiet_default: bool = True,
) -> TransportReverseTableResultBuilder:
    """Build a grouped transport table-result builder around a validated runner."""

    labels = tuple(str(name) for name in objective_labels)

    def _builder(
        objective_names: tuple[str, ...],
        parameter_set: ReverseADParameterSet,
        options: Mapping[str, object] | None,
    ) -> RealtimeGeometryTransportReverseTableResult:
        if table_context is not None:
            return transport_realtime_geometry_reverse_table(
                objective_names=objective_names,
                parameter_set=parameter_set,
                context=table_context,
                run_grouped_report=run_grouped_report,
                objective_labels=labels,
                options=options,
                quiet_default=quiet_default,
            )
        return build_realtime_geometry_transport_reverse_table_result(
            objective_names=objective_names,
            parameter_set=parameter_set,
            objective_labels=labels,
            run_grouped_report=run_grouped_report,
            options=options,
            quiet_default=quiet_default,
        )

    return _builder
