"""Reusable transport reverse-AD report-builder helpers.

This module owns production-facing transport reverse-AD seams that have been
lifted out of benchmark reporting. The segmented transport cotangent runner is
still supplied by the benchmark for now, while validated downstream pieces such
as the VMEC raw-block payload pullback live here.
"""

from __future__ import annotations

import copy
import contextlib
import dataclasses
import io
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ._geometry_autodiff import geometry_payload_pullback_from_param_vector_raw_block_transpose
from ._reverse_ad_parameters import ReverseADParameterSet
from ._transport_flux_models import (
    _add_float_delta_tree,
    _float_delta_tree_like,
    _sanitize_float_delta_bar_tree,
)
from ._transport_solvers import (
    _RadauAcceptedStepReducedCotangent,
    _radau_adaptive_final_y_realized_schedule_vjp_fwd,
    _radau_align_tangent_tree_to_primal,
    _radau_segment_reduced_cotangent_bwd_batched_with_support_call,
    _radau_zero_support_delta_tree_like,
)


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
    initial_er_charge_flux_residual_er_derivative: Callable[..., object]
    compact_initial_er_ntx_support_pullback_leaves: Callable[..., object]
    runtime_with_geometry_payload: Callable[[object, object], object]
    runtime_with_ntx_support_payload: Callable[[object, object], object]

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if not callable(value):
                raise TypeError(f"{field.name} must be callable.")


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
    combined_geometry_payload = str(args.realtime_geometry_gradient_path) == "reverse_payload"
    ntx_surface_backend = str(neoclassical_cfg.get("ntx_exact_surface_backend", "booz"))
    ntx_support_payload = find_ntx_support_payload(baseline_runtime)
    support_payload = (
        {"geometry": baseline_runtime.geometry, "ntx_support": ntx_support_payload}
        if combined_geometry_payload
        else ntx_support_payload
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
        reverse_stage_cotangent_mode=support_probe_cotangent_mode,
        reverse_step_bwd_mode=args.reverse_step_bwd_mode,
        reverse_stage_adjoint_memory_mode=args.reverse_stage_adjoint_memory_mode,
        reverse_stage_adjoint_iter_maxiter=args.reverse_stage_adjoint_iter_maxiter,
        reverse_stage_adjoint_iter_tol=args.reverse_stage_adjoint_iter_tol,
    )
    early_geometry_diagnostics = (
        None
        if geometry_volume_diagnostics is None
        else geometry_volume_diagnostics(baseline_runtime.geometry)
    )
    return RealtimeGeometrySupportSegmentCoreSetup(
        combined_geometry_payload=combined_geometry_payload,
        ntx_surface_backend=ntx_surface_backend,
        ntx_support_payload=ntx_support_payload,
        support_payload=support_payload,
        profile_values=profile_values,
        support_probe_cotangent_mode=support_probe_cotangent_mode,
        reverse_setup=reverse_setup,
        early_geometry_diagnostics=early_geometry_diagnostics,
    )


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

    def _batched_zero_tangent_tree_like(primal_tree, batch_size: int):
        zero_tree = _radau_align_tangent_tree_to_primal(None, primal_tree)
        return jax.tree_util.tree_map(
            lambda leaf: jnp.broadcast_to(
                jnp.asarray(leaf)[None, ...],
                (batch_size,) + jnp.asarray(leaf).shape,
            ),
            zero_tree,
        )

    initial_er_root_enabled = dependencies.initial_er_root_enabled(config, initial_er_root_ad)

    def _state_from_profiles(p):
        return dependencies.initial_state_for_parameter_vector(
            p,
            config=config,
            initial_er_root_ad="off",
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
        )

    phase_start = time.perf_counter()
    pre_root_initial_state, profile_state_pullback = jax.vjp(_state_from_profiles, parameter_values)
    initial_state = (
        dependencies.state_with_initial_er_root_ad(
            pre_root_initial_state,
            config=config,
            runtime=runtime,
            mode=initial_er_root_ad,
        )
        if initial_er_root_enabled
        else pre_root_initial_state
    )
    initial_state = jax.block_until_ready(initial_state)
    print(
        f"{progress_prefix} progress: support reverse profile-state vjp ready "
        f"elapsed_s={time.perf_counter() - phase_start:.3f}",
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

    phase_start = time.perf_counter()
    initial_carry, initial_state_pullback = jax.vjp(_carry_from_state, initial_state)
    initial_carry = jax.block_until_ready(initial_carry)
    print(
        f"{progress_prefix} progress: support reverse initial carry vjp ready "
        f"elapsed_s={time.perf_counter() - phase_start:.3f}",
        flush=True,
    )

    phase_start = time.perf_counter()
    final_y, residuals = _radau_adaptive_final_y_realized_schedule_vjp_fwd(
        reverse_setup.execution_context,
        reverse_setup.max_total_steps,
        reverse_setup.stop_after_accepted_steps,
        reverse_setup.reverse_segment_length,
        initial_carry,
    )
    final_y, residuals = jax.block_until_ready((final_y, residuals))
    print(
        f"{progress_prefix} progress: support reverse realized-schedule vjp forward ready "
        f"elapsed_s={time.perf_counter() - phase_start:.3f}",
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

    objective_count = int(len(objective_labels))
    objective_values_rows = []
    final_y_bar_rows = []
    objective_payload_bar_rows = []
    combined_geometry_payload = isinstance(support_payload, dict) and "geometry" in support_payload
    zero_payload_bar = _radau_zero_support_delta_tree_like(support_payload)
    for objective_i in range(objective_count):
        def _objective_from_final_y(final_y_value, objective_index=objective_i):
            final_state = reverse_setup.prepared_rollout.physics_context.unpack_flat(final_y_value)
            return dependencies.objective_scalar_by_index(final_state, runtime, objective_index)

        objective_value, objective_pullback = jax.vjp(_objective_from_final_y, final_y_for_objective)
        objective_values_rows.append(objective_value)
        final_y_bar_rows.append(objective_pullback(jnp.ones_like(objective_value))[0])
        if combined_geometry_payload:
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

            _, geometry_objective_pullback = jax.vjp(_objective_from_geometry_delta, geometry_delta0)
            (geometry_objective_bar,) = geometry_objective_pullback(jnp.ones_like(objective_value))
            objective_payload_bar_rows.append(
                {
                    "geometry": _sanitize_float_delta_bar_tree(geometry, geometry_objective_bar),
                    "ntx_support": zero_payload_bar["ntx_support"],
                }
            )
        else:
            objective_payload_bar_rows.append(zero_payload_bar)
    objective_values = jnp.stack(objective_values_rows, axis=0)
    final_y_bars = jnp.stack(final_y_bar_rows, axis=0)

    cotangent_mode = str(
        getattr(reverse_setup.execution_context.physics_context, "reverse_stage_cotangent_mode", "full")
    ).strip().lower()
    segment_count = int(jax.tree_util.tree_leaves(segmented_replay_arrays)[0].shape[0])

    reduced_bars = _RadauAcceptedStepReducedCotangent(
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
    _zero_support_leaves, support_treedef = jax.tree_util.tree_flatten(zero_payload_bar)
    objective_payload_bar_leaves = tuple(
        jax.tree_util.tree_leaves(payload_bar)
        for payload_bar in objective_payload_bar_rows
    )
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
    objective_support_bar_leaves = support_bar_leaves
    step_support_bar_leaves_accum = tuple(jnp.zeros_like(leaf) for leaf in support_bar_leaves)
    initial_cache_support_bar_leaves_accum = tuple(jnp.zeros_like(leaf) for leaf in support_bar_leaves)
    support_reuse_count = 0
    support_rebuild_count = 0
    phase_start = time.perf_counter()
    for segment_index in range(segment_count - 1, -1, -1):
        segment_start_carry = _take_tree_axis0(segment_start_carries, segment_index)
        segment_arrays = _take_tree_axis0(segmented_replay_arrays, segment_index)
        reduced_bars, segment_support_bar_leaves = (
            _radau_segment_reduced_cotangent_bwd_batched_with_support_call(
                reverse_setup.execution_context,
                cotangent_mode,
                reduced_bars,
                segment_start_carry,
                segment_arrays,
                support_payload,
            )
        )
        support_bar_leaves = tuple(
            accumulated + increment
            for accumulated, increment in zip(support_bar_leaves, segment_support_bar_leaves)
        )
        step_support_bar_leaves_accum = tuple(
            accumulated + increment
            for accumulated, increment in zip(step_support_bar_leaves_accum, segment_support_bar_leaves)
        )
        segment_lagged_valid = np.asarray(jax.device_get(segment_arrays[6])).reshape(-1)
        support_reuse_count += int(np.count_nonzero(segment_lagged_valid))
        support_rebuild_count += int(segment_lagged_valid.size - np.count_nonzero(segment_lagged_valid))
    reduced_bars, support_bar_leaves = jax.block_until_ready((reduced_bars, support_bar_leaves))
    print(
        f"{progress_prefix} progress: support reverse segmented cotangent sweep ready "
        f"elapsed_s={time.perf_counter() - phase_start:.3f}",
        flush=True,
    )

    initial_lagged_response_valid = bool(np.asarray(jax.device_get(carry0.lagged_response_valid)))
    build_support_pullback = reverse_setup.execution_context.physics_context.flat_rhs_build_support_pullback
    allow_initial_cache_support_pullback = cotangent_mode in {
        "full",
        "full_initial_cache_support_pullback",
        "initial_cache_support_pullback",
    }
    initial_cache_pullback_used = False
    initial_cache_pullback_skipped = False
    if initial_lagged_response_valid and build_support_pullback is not None and allow_initial_cache_support_pullback:
        initial_cache_support_bars = jax.lax.map(
            lambda lagged_bar: build_support_pullback(
                carry0.y,
                lagged_bar,
                support_payload,
            ),
            reduced_bars.lagged_response_cache,
        )
        initial_cache_support_bar_leaves = jax.tree_util.tree_leaves(initial_cache_support_bars)
        support_bar_leaves = tuple(
            accumulated + increment
            for accumulated, increment in zip(support_bar_leaves, initial_cache_support_bar_leaves)
        )
        initial_cache_support_bar_leaves_accum = tuple(
            accumulated + increment
            for accumulated, increment in zip(
                initial_cache_support_bar_leaves_accum,
                initial_cache_support_bar_leaves,
            )
        )
        initial_cache_pullback_used = True
    elif initial_lagged_response_valid and build_support_pullback is not None:
        initial_cache_pullback_skipped = True

    def _full_carry_bar_from_reduced(reduced_bar):
        return dataclasses.replace(
            jax.tree_util.tree_map(_zero_tangent_like, carry0),
            y=reduced_bar.y,
            lagged_response_cache=reduced_bar.lagged_response_cache,
            lagged_reference_y=reduced_bar.lagged_reference_y,
        )

    carry0_bars = jax.vmap(_full_carry_bar_from_reduced)(reduced_bars)
    initial_state_bars = jax.vmap(lambda carry0_bar: initial_state_pullback(carry0_bar)[0])(carry0_bars)
    initial_er_root_support_bars = None
    if initial_er_root_enabled:
        phase_start = time.perf_counter()
        er_profile, finite_mask = dependencies.initial_er_selected_root_profile(
            pre_root_initial_state,
            config=config,
            runtime=runtime,
        )

        er_profile = jnp.asarray(er_profile, dtype=pre_root_initial_state.Er.dtype)
        finite_mask = jnp.asarray(finite_mask, dtype=bool)

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

        _, state_residual_pullback = jax.vjp(
            lambda state_value: dependencies.initial_er_charge_flux_residuals(
                state_value,
                er_profile,
                runtime=runtime,
            ),
            pre_root_initial_state,
        )
        state_residual_bars = jax.vmap(lambda residual_bar: state_residual_pullback(residual_bar)[0])(
            residual_bars
        )
        direct_initial_state_bars = dataclasses.replace(
            initial_state_bars,
            Er=jnp.zeros_like(initial_state_bars.Er),
        )
        pre_root_initial_state_bars = dependencies.add_trees(
            direct_initial_state_bars,
            state_residual_bars,
        )

        if combined_geometry_payload:
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

            _, geometry_pullback = jax.vjp(_residuals_from_geometry_delta, geometry_delta0)
            geometry_bars = jax.vmap(lambda residual_bar: geometry_pullback(residual_bar)[0])(
                residual_bars
            )
            ntx_runtime = dependencies.runtime_with_geometry_payload(runtime, geometry)
            ntx_bar_leaves = dependencies.compact_initial_er_ntx_support_pullback_leaves(
                runtime=ntx_runtime,
                state=pre_root_initial_state,
                er_profile=er_profile,
                residual_bars=residual_bars,
                support=ntx_support,
            )
            initial_er_root_support_bars = (
                tuple(jax.tree_util.tree_leaves(geometry_bars)) + tuple(ntx_bar_leaves)
            )
        else:
            initial_er_root_support_bars = dependencies.compact_initial_er_ntx_support_pullback_leaves(
                runtime=runtime,
                state=pre_root_initial_state,
                er_profile=er_profile,
                residual_bars=residual_bars,
                support=support_payload,
            )
        pre_root_initial_state_bars, initial_er_root_support_bars = jax.block_until_ready(
            (pre_root_initial_state_bars, initial_er_root_support_bars)
        )
        print(
            f"{progress_prefix} progress: initial-Er root boundary compact pullback ready "
            f"elapsed_s={time.perf_counter() - phase_start:.3f}",
            flush=True,
        )
        initial_state_bars = pre_root_initial_state_bars

    gradient_matrix = jax.vmap(lambda state_bar: profile_state_pullback(state_bar)[0])(initial_state_bars)
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
    support_bars = tuple(
        support_treedef.unflatten(
            [jnp.asarray(leaf)[objective_i] for leaf in support_bar_leaves]
        )
        for objective_i in range(objective_count)
    )
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
    if initial_er_root_support_bars is not None:
        component_support_bars_by_name["initial_er_root"] = tuple(
            support_treedef.unflatten(
                [jnp.asarray(leaf)[objective_i] for leaf in initial_er_root_support_bar_leaves]
            )
            for objective_i in range(objective_count)
        )
    if combined_geometry_payload:
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
        initial_geometry_bars = jax.vmap(
            lambda state_bar: initial_geometry_pullback(state_bar)[0]
        )(initial_state_bars)
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
    return (
        objective_values,
        gradient_matrix,
        support_bars,
        component_support_bars_by_name,
        support_reuse_count,
        support_rebuild_count,
        initial_cache_pullback_used,
        initial_cache_pullback_skipped,
    )


def realtime_geometry_support_cotangents_from_parameter_vector(
    *,
    reverse_all_objectives_support_payload_bar: Callable[..., object],
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

    The callback is still supplied by the benchmark while the implementation is
    migrated. This helper owns the stable internal result shape and optional
    device synchronization without adding another reverse pass.
    """

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
    if not isinstance(callback_result, tuple) or len(callback_result) != 8:
        raise TypeError(
            "reverse_all_objectives_support_payload_bar must return an 8-tuple: "
            "(objective_values, profile_gradient_matrix, support_bars, "
            "support_component_bars_by_name, support_reuse_count, "
            "support_rebuild_count, initial_cache_pullback_used, "
            "initial_cache_pullback_skipped)."
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
    ) = callback_result
    if block_until_ready:
        objective_values, profile_gradient_matrix, support_bars, support_component_bars_by_name = (
            jax.block_until_ready(
                (
                    objective_values,
                    profile_gradient_matrix,
                    support_bars,
                    support_component_bars_by_name,
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
    """Build the temporary executor wrapper around a segmented reverse probe.

    The supplied probe still owns the heavy segmented reverse sweep during
    migration.  This wrapper owns the reusable calling convention: grouped
    optimization paths must request an internal report and receive the shared
    runtime/config inputs.
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
            suppress_output=True,
        )

    return _executor


def run_realtime_geometry_support_segment_reverse_table_core(
    *,
    support_segment_probe: TransportReverseSupportSegmentProbe,
    args,
    context: RealtimeGeometryTransportReverseTableContext,
    suppress_output: bool = True,
) -> TransportReverseReport:
    """Run the segmented support probe as a non-printing reverse table core.

    The heavy probe callback is still supplied externally during migration.
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

    The benchmark still supplies `support_segment_executor`, which owns the
    segmented reverse sweep during migration.  This helper owns the reusable
    context construction and objective='all' grouped-runner contract.
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
    lane.  At this stage the heavy runner is still supplied by the benchmark;
    later steps can move that runner here without changing callers.
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

    This is the stable internal API boundary for optimization callers.  The
    expensive segmented reverse runner can still be supplied from the benchmark
    while that code is being migrated, but callers interact with a
    `RealtimeGeometryTransportReverseTableRequest` and receive the JAX-native
    table result directly.
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


def realtime_geometry_payload_pullback_result(
    *,
    geometry_context,
    baseline_geometry_deltas,
    geometry_param_specs: Sequence[tuple[str, int, int]],
    support_bars: Sequence[object],
    support_component_bars_by_name: Mapping[str, Sequence[object]] | None = None,
    include_component_pullbacks: bool = False,
    combined_geometry_payload: bool = True,
    n_r: int = 51,
    n_theta: int = 25,
    n_zeta: int = 25,
    n_xi: int = 64,
    surface_backend: str = "booz",
    max_iter=None,
    solver_device: str = "default",
    progress_label: str | None = None,
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

    geometry_gradient_result = geometry_payload_pullback_from_param_vector_raw_block_transpose(
        geometry_context,
        baseline_geometry_deltas,
        tuple(geometry_param_specs),
        geometry_pullback_payload_bars,
        combined_payload=combined_geometry_payload,
        n_r=int(n_r),
        n_theta=int(n_theta),
        n_zeta=int(n_zeta),
        n_xi=int(n_xi),
        surface_backend=surface_backend,
        max_iter=max_iter,
        solver_device=str(solver_device),
        progress_label=progress_label,
        return_branch_gradients=True,
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
        geometry_gradient_matrix, component_gradient_matrices = _split_component_rows(
            geometry_gradient_result["combined"]
        )
        geometry_branch_gradient_matrix, component_geometry_branch_matrices = _split_component_rows(
            geometry_gradient_result.get("geometry")
        )
        ntx_support_branch_gradient_matrix, component_ntx_support_branch_matrices = _split_component_rows(
            geometry_gradient_result.get("ntx_support")
        )
    else:
        geometry_gradient_matrix, component_gradient_matrices = _split_component_rows(
            geometry_gradient_result
        )
        geometry_branch_gradient_matrix = None
        ntx_support_branch_gradient_matrix = None
        component_geometry_branch_matrices = {}
        component_ntx_support_branch_matrices = {}

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
    include_component_pullbacks: bool = False,
    combined_geometry_payload: bool = True,
    n_r: int = 51,
    n_theta: int = 25,
    n_zeta: int = 25,
    n_xi: int = 64,
    surface_backend: str = "booz",
    max_iter=None,
    solver_device: str = "default",
    progress_label: str | None = None,
) -> RealtimeGeometryTransportReverseAssemblyResult:
    """Assemble the JAX transport reverse table from support-payload cotangents."""

    payload_pullback_result = realtime_geometry_payload_pullback_result(
        geometry_context=geometry_context,
        baseline_geometry_deltas=baseline_geometry_deltas,
        geometry_param_specs=geometry_param_specs,
        support_bars=support_bars,
        support_component_bars_by_name=support_component_bars_by_name,
        include_component_pullbacks=include_component_pullbacks,
        combined_geometry_payload=combined_geometry_payload,
        n_r=n_r,
        n_theta=n_theta,
        n_zeta=n_zeta,
        n_xi=n_xi,
        surface_backend=surface_backend,
        max_iter=max_iter,
        solver_device=solver_device,
        progress_label=progress_label,
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
