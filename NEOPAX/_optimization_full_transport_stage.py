"""Optimization-only persistent boundaries for full-transport reverse.

The benchmark reverse lane constructs a fresh Radau execution context for
each geometry.  Its segment replay call deliberately treats that context as
static, which is appropriate for a one-shot benchmark but creates one JAX
cache entry per optimizer evaluation.  This module keeps the benchmark calls
unchanged and provides persistent live-database, replay, and database-backward
scans whose geometry/database arrays are explicit dynamic inputs.  The
backward scan reuses the fixed solver structure and rebuilds only its
support-dependent direct-state transpose, rather than tracing a complete
rollout/context build.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ._geometry_autodiff import (
    _build_neopax_geometry_from_state,
    build_ntx_runtime_scan_inputs_from_vmec_state,
)
from ._optimization_initial_root_stage import FloatingPayloadLeafLayout
from ._reverse_ad_initial_er import (
    _replace_geometry_and_fresh_database_payload_in_model,
    find_ntx_database_transport_model_in_model,
    find_ntx_runtime_scan_model_in_model,
    runtime_with_fresh_ntx_database_payload,
    runtime_with_geometry_payload,
)
from ._transport_equations import build_equation_system
from ._transport_flux_models import (
    NTXRuntimeScanTransportModel,
)
from ._transport_solvers import (
    _radau_adaptive_schedule_rollout,
    _build_prepared_radau_accepted_rollout,
    _build_prepared_radau_execution_context,
    _extract_fixed_temperature_projection,
    _extract_state_regularization,
    _flat_rhs_direct_database_payload_pullback_batched_factory,
    _flat_rhs_direct_database_payload_pullback_factory,
    _flat_rhs_factory,
    _make_solver_state_transform,
    _radau_database_segment_reduced_cotangent_bwd_with_table_support_call,
    _radau_segment_replay_minimal_with_primal_records_call,
)


def _tree_signature(
    tree: Any,
) -> tuple[Any, tuple[tuple[tuple[int, ...], str, bool], ...]]:
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    signature = []
    for leaf in leaves:
        array = jnp.asarray(leaf)
        signature.append(
            (
                tuple(array.shape),
                str(array.dtype),
                bool(getattr(array, "weak_type", False)),
            )
        )
    return treedef, tuple(signature)


def _replace_runtime_scan_model(model, replacement):
    """Replace only the live NTX scan owner in a composite flux model."""

    if model is None or not dataclasses.is_dataclass(model) or isinstance(model, type):
        return model, False
    if isinstance(model, NTXRuntimeScanTransportModel):
        return replacement, True
    updates = {}
    changed = False
    for field in dataclasses.fields(model):
        value = getattr(model, field.name)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            new_value, child_changed = _replace_runtime_scan_model(value, replacement)
            if child_changed:
                updates[field.name] = new_value
                changed = True
    if not changed:
        return model, False
    return dataclasses.replace(model, **updates), True


def _scan_primal_record_payload(record):
    """Expose only the registered PyTree contents of an NTX scan record."""

    return (
        tuple(record.surfaces),
        tuple(record.prepared),
        record.Es,
        record.nu_v,
    )


def _scan_primal_record_from_payload(template, payload):
    """Restore NTX's host-side record wrapper and fixed grid metadata."""

    surfaces, prepared, es, nu_v = payload
    return dataclasses.replace(
        template,
        surfaces=tuple(surfaces),
        prepared=tuple(prepared),
        Es=es,
        nu_v=nu_v,
    )


@dataclasses.dataclass
class DatabaseFullTransportRuntimeOptimizationStage:
    """Persistent live NTX forward scan for changing optimizer geometries.

    Geometry and the VMEC-derived scan payload are rebuilt for every trial.
    Only the fixed NTX grid/configuration and compiled scan identity persist.
    This is deliberately an optimization-only owner; the benchmark runtime
    construction remains unchanged.
    """

    runtime_template: Any
    geometry_context: Any
    n_r: int
    rho_scan: Any
    surface_backend: str
    scan_model_template: NTXRuntimeScanTransportModel
    scan_payload_layout: FloatingPayloadLeafLayout
    scan_payload_signature: Any
    compiled_scan: Any

    def runtime_for_vmec_state(self, state_vmec):
        geometry = _build_neopax_geometry_from_state(
            self.geometry_context,
            state_vmec,
            n_r=int(self.n_r),
        )
        channels, surfaces = build_ntx_runtime_scan_inputs_from_vmec_state(
            self.geometry_context,
            state_vmec,
            geometry,
            rho_scan=self.rho_scan,
            surface_backend=self.surface_backend,
        )
        scan_payload = {"channels": channels, "surfaces": surfaces}
        self.scan_payload_layout.validate_static_structure(scan_payload)
        scan_payload_leaves = self.scan_payload_layout.floating_leaves(scan_payload)
        if _tree_signature(scan_payload_leaves) != self.scan_payload_signature:
            raise ValueError(
                "Full-transport optimization live-scan payload shape or dtype "
                "changed within a stage."
            )
        database, scan_primal_record_payload, raw_scan = self.compiled_scan(
            scan_payload_leaves
        )
        scan_primal_record = _scan_primal_record_from_payload(
            self.scan_model_template.scan_primal_record,
            scan_primal_record_payload,
        )
        active_scan_model = self.scan_model_template.with_runtime_scan_payload(
            geometry=geometry,
            channels=channels,
            scan_surfaces=tuple(surfaces),
            database=database,
        )
        active_scan_model = dataclasses.replace(
            active_scan_model,
            scan_primal_record=scan_primal_record,
            scan_primal=raw_scan,
        )
        runtime_with_geometry = runtime_with_geometry_payload(
            self.runtime_template,
            geometry,
        )
        flux_model, changed = _replace_runtime_scan_model(
            runtime_with_geometry.models.flux,
            active_scan_model,
        )
        if not changed:
            raise ValueError(
                "Full-transport optimization runtime has no live NTX scan model."
            )
        return dataclasses.replace(
            runtime_with_geometry,
            database=database,
            models=dataclasses.replace(
                runtime_with_geometry.models,
                flux=flux_model,
            ),
        )

    def cache_size(self) -> int | None:
        cache_size = getattr(self.compiled_scan, "_cache_size", None)
        if not callable(cache_size):
            return None
        try:
            return int(cache_size())
        except Exception:
            return None


def build_database_full_transport_runtime_optimization_stage(
    *,
    runtime,
    geometry_context,
    n_r: int,
) -> DatabaseFullTransportRuntimeOptimizationStage:
    """Build one persistent wrapper around the existing live NTX scan."""

    scan_model = find_ntx_runtime_scan_model_in_model(runtime.models.flux)
    if scan_model is None:
        raise ValueError(
            "Full-transport runtime optimization requires an NTX runtime scan model."
        )
    if scan_model.channels is None or scan_model.scan_surfaces is None:
        raise ValueError(
            "Full-transport runtime optimization requires preloaded scan inputs."
        )
    if (
        not bool(scan_model.record_scan_primal)
        or scan_model.scan_primal_record is None
        or scan_model.scan_primal is None
    ):
        raise ValueError(
            "Full-transport runtime optimization requires a recorded scan primal."
        )
    scan_payload = {
        "channels": scan_model.channels,
        "surfaces": scan_model.scan_surfaces,
    }
    scan_payload_layout = FloatingPayloadLeafLayout.from_template(scan_payload)
    scan_payload_leaves = scan_payload_layout.floating_leaves(scan_payload)

    def _scan_kernel(active_scan_payload_leaves):
        active_payload = scan_payload_layout.rebuild(active_scan_payload_leaves)
        active_scan_model = scan_model.with_runtime_scan_payload(
            geometry=scan_model.geometry,
            channels=active_payload["channels"],
            scan_surfaces=tuple(active_payload["surfaces"]),
            database=None,
        )
        database, scan_primal_record, raw_scan = (
            active_scan_model._build_runtime_database_and_record()
        )
        # NTX deliberately keeps the record itself as an ordinary host-side
        # dataclass.  Its contents are valid JAX PyTrees, so return those
        # numerical fields across this optimization-only JIT and restore the
        # unchanged record type/grid immediately outside the boundary.
        scan_primal_record_payload = _scan_primal_record_payload(
            scan_primal_record
        )
        return database, scan_primal_record_payload, raw_scan

    return DatabaseFullTransportRuntimeOptimizationStage(
        runtime_template=runtime,
        geometry_context=geometry_context,
        n_r=int(n_r),
        rho_scan=np.asarray(jax.device_get(scan_model.rho_scan), dtype=float),
        surface_backend=str(scan_model.surface_backend),
        scan_model_template=scan_model,
        scan_payload_layout=scan_payload_layout,
        scan_payload_signature=_tree_signature(scan_payload_leaves),
        compiled_scan=jax.jit(_scan_kernel, inline=False),
    )


def _equation_system_with_fresh_database_payload(
    template,
    support_payload,
    *,
    static_ntss_density_indices=None,
):
    """Bind an already-current database without rescaling the template table."""

    if not (
        isinstance(support_payload, dict)
        and set(support_payload) == {"geometry", "database"}
        and all(
            hasattr(template, name)
            for name in (
                "config",
                "species",
                "shared_flux_model",
                "source_models",
                "solver_cfg",
                "boundary_models",
            )
        )
    ):
        return template.with_realtime_geometry_support_payload(support_payload)
    geometry = support_payload["geometry"]
    database = support_payload["database"]
    flux_model, changed = _replace_geometry_and_fresh_database_payload_in_model(
        template.shared_flux_model,
        geometry,
        database,
    )
    if not changed:
        raise ValueError(
            "No fixed NTX database model was found in the full-transport equation system."
        )
    equations = tuple(
        build_equation_system(
            config=template.config,
            species=template.species,
            field=geometry,
            flux_model=flux_model,
            source_models=template.source_models,
            solver_cfg=template.solver_cfg,
            boundary_models=template.boundary_models,
        )
    )
    if static_ntss_density_indices is not None:
        # ``build_electric_field_equation`` represents these fixed species
        # indices as a JAX array. Rebuilding the equations inside the
        # optimization replay JIT would therefore turn the indices into a
        # tracer, while the established Radau setup correctly treats them as
        # structural metadata. Restore the concrete template value here.
        equations = tuple(
            dataclasses.replace(
                equation,
                ntss_density_indices=static_ntss_density_indices,
            )
            if getattr(equation, "name", None) == "Er"
            and hasattr(equation, "ntss_density_indices")
            else equation
            for equation in equations
        )
    return dataclasses.replace(
        template,
        equations=equations,
        density_equation=next(
            (eq for eq in equations if getattr(eq, "name", None) == "density"),
            None,
        ),
        temperature_equation=next(
            (eq for eq in equations if getattr(eq, "name", None) == "temperature"),
            None,
        ),
        er_equation=next(
            (eq for eq in equations if getattr(eq, "name", None) == "Er"),
            None,
        ),
        shared_flux_model=flux_model,
    )


@dataclasses.dataclass
class DatabaseFullTransportBootstrapOptimizationStage:
    """Persistent terminal-bootstrap boundary for one optimization stage."""

    support_layout: FloatingPayloadLeafLayout
    compiled_bootstrap: Any
    support_floating_signature: Any

    def evaluate(self, final_y, support_payload):
        self.support_layout.validate_static_structure(support_payload)
        support_leaves = self.support_layout.floating_leaves(support_payload)
        if _tree_signature(support_leaves) != self.support_floating_signature:
            raise ValueError(
                "Full-transport bootstrap support shape or dtype changed "
                "within an optimization stage."
            )
        return self.compiled_bootstrap(final_y, support_leaves)

    def cache_size(self) -> int | None:
        cache_size = getattr(self.compiled_bootstrap, "_cache_size", None)
        if not callable(cache_size):
            return None
        try:
            return int(cache_size())
        except Exception:
            return None


def build_database_full_transport_bootstrap_optimization_stage(
    *,
    runtime,
    reverse_setup,
    support_payload,
) -> DatabaseFullTransportBootstrapOptimizationStage:
    """Retain the existing database bootstrap calculation behind one JIT."""

    from ._reverse_ad_transport import (
        _database_bootstrap_interpolation_bar,
        bootstrap_current_softmax_abs_value_and_upar_bar,
    )
    from ._transport_flux_models import _sanitize_float_delta_bar_tree

    if not (
        isinstance(support_payload, dict)
        and set(support_payload) == {"geometry", "database"}
    ):
        raise ValueError(
            "Full-transport bootstrap optimization requires exactly "
            "{'geometry', 'database'} support."
        )
    support_layout = FloatingPayloadLeafLayout.from_template(support_payload)
    support_leaves = support_layout.floating_leaves(support_payload)
    unpack_flat = reverse_setup.prepared_rollout.physics_context.unpack_flat
    runtime_template = runtime

    def _bootstrap(final_y, active_support_leaves):
        active_support = support_layout.rebuild(active_support_leaves)
        geometry = active_support["geometry"]
        database = active_support["database"]
        active_runtime = runtime_with_fresh_ntx_database_payload(
            runtime_template,
            geometry=geometry,
            database=database,
        )
        database_model = find_ntx_database_transport_model_in_model(
            active_runtime.models.flux
        )
        if database_model is None:
            raise ValueError(
                "Full-transport bootstrap stage requires an NTX database model."
            )
        final_state = unpack_flat(final_y)
        corrected_fluxes = {
            "Upar": database_model.evaluate_momentum_corrected_upar_only(
                final_state
            )
        }
        objective_value, upar_bar = (
            bootstrap_current_softmax_abs_value_and_upar_bar(
                final_state,
                active_runtime,
                corrected_fluxes,
            )
        )
        state_bar, geometry_bar = (
            database_model.pullback_momentum_corrected_upar_state_geometry_by_radius(
                final_state,
                upar_bar,
                geometry,
            )
        )
        database_bar = _database_bootstrap_interpolation_bar(
            database,
            final_state,
            upar_bar,
            mode="legacy_sparse",
            table_pullback=None,
            coordinate_pullback=None,
            sparse_pullback=(
                database_model.pullback_momentum_corrected_upar_database_support_legacy_sparse_by_radius
            ),
        )
        _, unpack_pullback = jax.vjp(unpack_flat, final_y)
        final_y_bar = unpack_pullback(state_bar)[0]
        return (
            objective_value,
            final_y_bar,
            {
                "geometry": _sanitize_float_delta_bar_tree(
                    geometry, geometry_bar
                ),
                "database": _sanitize_float_delta_bar_tree(
                    database, database_bar
                ),
            },
        )

    return DatabaseFullTransportBootstrapOptimizationStage(
        support_layout=support_layout,
        compiled_bootstrap=jax.jit(_bootstrap, inline=False),
        support_floating_signature=_tree_signature(support_leaves),
    )


@dataclasses.dataclass
class DatabaseFullTransportReplayOptimizationStage:
    """One persistent replay executable for a fixed optimization stage.

    Only structural objects are retained.  The initial transport state,
    current VMEC/database support, segment carry, and schedule arrays are
    supplied afresh on every call.
    """

    support_layout: FloatingPayloadLeafLayout
    initial_state_unpack: Any
    solver: Any
    species: Any
    equation_system_template: Any
    compiled_replay: Any
    compiled_database_bwd: Any
    compiled_schedule_probe: Any
    support_floating_signature: Any
    initial_state_signature: Any
    segment_carry_signature: Any
    segment_arrays_signature: Any

    def replay(
        self,
        *,
        initial_flat_state,
        support_payload,
        segment_start_carry,
        segment_arrays,
    ):
        self.support_layout.validate_static_structure(support_payload)
        support_floating_leaves = self.support_layout.floating_leaves(support_payload)
        if _tree_signature(support_floating_leaves) != self.support_floating_signature:
            raise ValueError(
                "Full-transport optimization support shape, dtype, or weak type "
                "changed within a stage."
            )
        if _tree_signature(initial_flat_state) != self.initial_state_signature:
            raise ValueError(
                "Full-transport optimization initial-state layout changed within a stage."
            )
        if _tree_signature(segment_start_carry) != self.segment_carry_signature:
            raise ValueError(
                "Full-transport optimization segment-carry layout changed within a stage."
            )
        if _tree_signature(segment_arrays) != self.segment_arrays_signature:
            raise ValueError(
                "Full-transport optimization segment schedule changed shape or "
                "dtype within a stage."
            )
        return self.compiled_replay(
            initial_flat_state,
            support_floating_leaves,
            segment_start_carry,
            segment_arrays,
        )

    def cache_size(self) -> int | None:
        cache_size = getattr(self.compiled_replay, "_cache_size", None)
        if not callable(cache_size):
            return None
        try:
            return int(cache_size())
        except Exception:
            return None

    def database_segment_bwd(
        self,
        *,
        initial_flat_state,
        support_payload,
        cotangent_mode,
        segment_reduced_bars,
        step_start_carries,
        step_primal_records,
        segment_arrays,
    ):
        """Run the exact benchmark scan through one lean persistent JIT."""

        self.support_layout.validate_static_structure(support_payload)
        support_floating_leaves = self.support_layout.floating_leaves(support_payload)
        if _tree_signature(support_floating_leaves) != self.support_floating_signature:
            raise ValueError(
                "Full-transport optimization support shape, dtype, or weak type "
                "changed within a stage."
            )
        if _tree_signature(initial_flat_state) != self.initial_state_signature:
            raise ValueError(
                "Full-transport optimization initial-state layout changed within a stage."
            )
        if _tree_signature(segment_arrays) != self.segment_arrays_signature:
            raise ValueError(
                "Full-transport optimization segment schedule changed shape or "
                "dtype within a stage."
            )
        return self.compiled_database_bwd(
            support_floating_leaves,
            str(cotangent_mode),
            segment_reduced_bars,
            step_start_carries,
            step_primal_records,
            segment_arrays,
        )

    def database_bwd_cache_size(self) -> int | None:
        cache_size = getattr(self.compiled_database_bwd, "_cache_size", None)
        if not callable(cache_size):
            return None
        try:
            return int(cache_size())
        except Exception:
            return None

    def schedule_probe(
        self,
        *,
        initial_carry,
        support_payload,
        max_total_steps,
        stop_after_accepted_steps,
        capture_segment_length,
    ):
        """Run the exact adaptive schedule through one persistent JIT."""

        self.support_layout.validate_static_structure(support_payload)
        support_floating_leaves = self.support_layout.floating_leaves(support_payload)
        if _tree_signature(support_floating_leaves) != self.support_floating_signature:
            raise ValueError(
                "Full-transport optimization support shape, dtype, or weak type "
                "changed within a stage."
            )
        if _tree_signature(initial_carry.y) != self.initial_state_signature:
            raise ValueError(
                "Full-transport optimization initial-state layout changed within a stage."
            )
        return self.compiled_schedule_probe(
            initial_carry.y,
            support_floating_leaves,
            initial_carry,
            int(max_total_steps),
            (
                None
                if stop_after_accepted_steps is None
                else int(stop_after_accepted_steps)
            ),
            (
                None
                if capture_segment_length is None
                else int(capture_segment_length)
            ),
        )

    def schedule_probe_cache_size(self) -> int | None:
        cache_size = getattr(self.compiled_schedule_probe, "_cache_size", None)
        if not callable(cache_size):
            return None
        try:
            return int(cache_size())
        except Exception:
            return None


def build_database_full_transport_replay_optimization_stage(
    *,
    reverse_setup,
    species,
    support_payload,
    segment_start_carry,
    segment_arrays,
) -> DatabaseFullTransportReplayOptimizationStage:
    """Build the persistent database replay without changing benchmark math."""

    # Imported lazily to avoid a module cycle: the production reverse module
    # owns this established database-hook configurator, while optimization.py
    # imports this stage before importing the reverse transport entry points.
    from ._reverse_ad_transport import _configure_database_reverse_performance

    equation_system_template = getattr(reverse_setup.solve_vector_field, "__self__", None)
    replace_support = getattr(
        equation_system_template, "with_realtime_geometry_support_payload", None
    )
    if not callable(replace_support):
        raise TypeError(
            "Database full-transport optimization requires a composed equation "
            "system with with_realtime_geometry_support_payload(...)."
        )
    replay_body = getattr(
        _radau_segment_replay_minimal_with_primal_records_call, "__wrapped__", None
    )
    if not callable(replay_body):
        raise RuntimeError("The established Radau segment replay body is unavailable.")
    database_bwd_body = getattr(
        _radau_database_segment_reduced_cotangent_bwd_with_table_support_call,
        "__wrapped__",
        None,
    )
    if not callable(database_bwd_body):
        raise RuntimeError(
            "The established Radau database segment backward body is unavailable."
        )
    support_layout = FloatingPayloadLeafLayout.from_template(support_payload)
    support_floating_leaves = support_layout.floating_leaves(support_payload)
    solver = reverse_setup.solver
    template_physics_context = reverse_setup.execution_context.physics_context
    reverse_control_values = {
        field.name: getattr(template_physics_context, field.name)
        for field in dataclasses.fields(template_physics_context)
        if field.name.startswith("reverse_")
    }
    template_er_equation = getattr(equation_system_template, "er_equation", None)
    template_ntss_density_indices = getattr(
        template_er_equation, "ntss_density_indices", None
    )
    static_ntss_density_indices = (
        None
        if template_ntss_density_indices is None
        else np.array(
            jax.device_get(template_ntss_density_indices),
            dtype=np.int32,
            copy=True,
        ).reshape((-1,))
    )
    initial_flat_state = reverse_setup.prepared_rollout.initial_carry.y
    current_initial_state = (
        reverse_setup.execution_context.physics_context.unpack_flat(initial_flat_state)
    )
    zero_state_template = jax.tree_util.tree_map(jnp.zeros_like, current_initial_state)
    temperature_active_mask, fixed_temperature_profile = (
        _extract_fixed_temperature_projection(reverse_setup.solve_vector_field)
    )
    density_floor, temperature_floor = _extract_state_regularization(
        reverse_setup.solve_vector_field
    )
    _, initial_state_unpack, _, _, _ = _make_solver_state_transform(
        zero_state_template,
        species,
        temperature_active_mask=temperature_active_mask,
        fixed_temperature_profile=fixed_temperature_profile,
        density_floor=density_floor,
        temperature_floor=temperature_floor,
    )

    def _active_execution_context(active_initial_flat_state, support_floating_leaves):
        active_support = support_layout.rebuild(support_floating_leaves)
        active_equation_system = _equation_system_with_fresh_database_payload(
            equation_system_template,
            active_support,
            static_ntss_density_indices=static_ntss_density_indices,
        )
        active_initial_state = initial_state_unpack(active_initial_flat_state)
        active_rollout = _build_prepared_radau_accepted_rollout(
            solver=solver,
            state=active_initial_state,
            vector_field=active_equation_system.vector_field,
            species=species,
        )
        active_execution_context = _build_prepared_radau_execution_context(
            solver=solver,
            prepared_rollout=active_rollout,
        )
        # Recreate any mode-dependent database pullback callbacks from the
        # current equation system.  Copying those callbacks from the template
        # would retain its old geometry/database, defeating this dynamic JIT
        # boundary.  The reverse_* values themselves are structural controls,
        # so transplant all of them from the already-validated benchmark
        # execution context after the fresh callbacks have been constructed.
        active_physics_context = _configure_database_reverse_performance(
            active_execution_context.physics_context,
            vector_field=active_equation_system.vector_field,
            species=species,
            initial_support_mode=template_physics_context.reverse_database_initial_support_mode,
            initial_state_mode=template_physics_context.reverse_database_initial_state_mode,
            support_preparation_mode=template_physics_context.reverse_database_support_preparation_mode,
            center_geometry_mode=template_physics_context.reverse_database_center_geometry_mode,
            stage_jacobian_mode=template_physics_context.reverse_database_stage_jacobian_mode,
            support_objective_mode=template_physics_context.reverse_database_support_objective_mode,
            segment_support_mode=template_physics_context.reverse_database_segment_support_mode,
            interpolation_transpose_mode=template_physics_context.reverse_database_interpolation_transpose_mode,
            root_interpolation_transpose_mode=template_physics_context.reverse_database_root_interpolation_transpose_mode,
            bootstrap_interpolation_transpose_mode=template_physics_context.reverse_database_bootstrap_interpolation_transpose_mode,
        )
        active_execution_context = dataclasses.replace(
            active_execution_context,
            physics_context=dataclasses.replace(
                active_physics_context,
                **reverse_control_values,
            ),
        )
        return active_execution_context, active_support

    def _replay_kernel(
        active_initial_flat_state,
        support_floating_leaves,
        active_segment_start_carry,
        active_segment_arrays,
    ):
        active_execution_context, _ = _active_execution_context(
            active_initial_flat_state,
            support_floating_leaves,
        )
        # Invoke the exact body underlying the benchmark JIT.  The outer JIT
        # is the sole optimization cache owner, and all trial data above are
        # explicit numerical arguments to it.
        return replay_body(
            active_execution_context,
            active_segment_start_carry,
            active_segment_arrays,
        )

    def _schedule_probe_kernel(
        active_initial_flat_state,
        support_floating_leaves,
        active_initial_carry,
        max_total_steps,
        stop_after_accepted_steps,
        capture_segment_length,
    ):
        active_execution_context, _ = _active_execution_context(
            active_initial_flat_state,
            support_floating_leaves,
        )
        return _radau_adaptive_schedule_rollout(
            active_execution_context,
            active_initial_carry,
            max_total_steps=max_total_steps,
            stop_after_accepted_steps=stop_after_accepted_steps,
            capture_segment_length=capture_segment_length,
        )

    template_execution_context = reverse_setup.execution_context
    template_unpack_flat = template_physics_context.unpack_flat
    template_unpack_bar = getattr(
        template_unpack_flat,
        "cotangent",
        template_unpack_flat,
    )
    template_pack_flat = template_physics_context.pack_flat
    template_project_flat = template_physics_context.project_flat

    def _direct_state_pullback_with_equation_system(
        active_equation_system,
        t_value,
        flat_y,
        _lagged_response,
        rhs_bar_flat,
    ):
        """Exact benchmark direct-state transpose with live support.

        The ordinary benchmark closure binds this operation to its one fixed
        equation system.  Optimization instead reconstructs only the current
        equation-system payload needed by the same transpose.  Solver setup,
        rollout preparation, and the Radau execution context remain static.
        """

        pullback_fn = getattr(
            active_equation_system,
            "pullback_direct_rhs_state",
            None,
        )
        if not callable(pullback_fn):
            raise RuntimeError(
                "Database full-transport optimization requires the exact "
                "direct-RHS state pullback."
            )
        projected_flat_y = (
            flat_y
            if template_project_flat is None
            else template_project_flat(flat_y)
        )
        state_y = template_unpack_flat(projected_flat_y)
        rhs_bar_state = template_unpack_bar(
            jnp.asarray(rhs_bar_flat, dtype=jnp.asarray(flat_y).dtype)
        )
        state_bar = pullback_fn(
            t_value,
            state_y,
            species,
            rhs_bar_state,
        )
        if state_bar is None:
            raise ValueError("Direct black-box RHS state pullback returned None.")
        projected_bar = template_pack_flat(state_bar)
        if template_project_flat is None:
            return projected_bar
        _, project_pullback = jax.vjp(template_project_flat, flat_y)
        return project_pullback(projected_bar)[0]

    def _database_bwd_kernel(
        support_floating_leaves,
        cotangent_mode,
        segment_reduced_bars,
        step_start_carries,
        step_primal_records,
        active_segment_arrays,
    ):
        active_support = support_layout.rebuild(support_floating_leaves)
        active_equation_system = _equation_system_with_fresh_database_payload(
            equation_system_template,
            active_support,
            static_ntss_density_indices=static_ntss_density_indices,
        )
        # The exact block database stage adjoint forms its state Jacobian from
        # ``physics_context.flat_rhs``.  Keeping the template callable here
        # would therefore differentiate the first optimization geometry even
        # though replay and support cotangents use the current trial payload.
        # Rebuild just this numerical RHS closure from the live equation
        # system; the benchmark kernel and all solver/root structure remain
        # unchanged and static.
        active_flat_rhs = _flat_rhs_factory(
            template_unpack_flat,
            active_equation_system.vector_field,
            (species,),
            {},
            project_flat=template_project_flat,
        )

        def _active_direct_state_pullback(
            t_value,
            flat_y,
            lagged_response,
            rhs_bar_flat,
        ):
            return _direct_state_pullback_with_equation_system(
                active_equation_system,
                t_value,
                flat_y,
                lagged_response,
                rhs_bar_flat,
            )

        # The split table/geometry support transpose is a bound equation-system
        # method.  It accepts the live support tree, but its equation-geometry
        # branch still belongs to the equation owner on which it was created.
        # Rebind that callback to the current equation system as well; otherwise
        # a persistent stage returns the first geometry's transport derivatives
        # even though its primal, table and state transpose are current.
        split_options = {
            name: value
            for name, value, default in (
                (
                    "center_geometry_mode",
                    template_physics_context.reverse_database_center_geometry_mode,
                    "scalar_jvp",
                ),
                (
                    "support_preparation_mode",
                    template_physics_context.reverse_database_support_preparation_mode,
                    "shared",
                ),
                (
                    "interpolation_transpose_mode",
                    template_physics_context.reverse_database_interpolation_transpose_mode,
                    "established",
                ),
            )
            if value != default
        }
        split_support_pullback = _flat_rhs_direct_database_payload_pullback_factory(
            template_unpack_flat,
            active_equation_system.vector_field,
            (species,),
            {},
            "pullback_direct_rhs_database_split_support_payload",
            project_flat=template_project_flat,
            pullback_options=split_options,
        )
        if split_support_pullback is None:
            raise ValueError(
                "Database full-transport optimization requires the fixed-"
                "database split support hook."
            )
        split_callback_values = {
            "flat_rhs_direct_database_split_support_pullback": split_support_pullback
        }
        if (
            template_physics_context.reverse_database_support_objective_mode
            == "batched_split"
        ):
            batched_split_support_pullback = (
                _flat_rhs_direct_database_payload_pullback_batched_factory(
                    template_unpack_flat,
                    active_equation_system.vector_field,
                    (species,),
                    {},
                    "pullback_direct_rhs_database_split_support_payload_batched",
                    project_flat=template_project_flat,
                    pullback_options=split_options,
                )
            )
            if batched_split_support_pullback is None:
                raise ValueError(
                    "Database full-transport optimization requires the batched "
                    "fixed-database split support hook."
                )
            split_callback_values[
                "flat_rhs_direct_database_split_support_pullback_batched"
            ] = batched_split_support_pullback
        active_physics_context = dataclasses.replace(
            template_physics_context,
            flat_rhs=active_flat_rhs,
            flat_rhs_direct_black_box_state_pullback=(
                _active_direct_state_pullback
            ),
            **split_callback_values,
        )
        active_execution_context = dataclasses.replace(
            template_execution_context,
            physics_context=active_physics_context,
        )
        # This is the exact body beneath the benchmark segment JIT.  Unlike
        # the rejected boundary, no prepared rollout or complete physics
        # context is rebuilt inside this scan.
        return database_bwd_body(
            active_execution_context,
            cotangent_mode,
            segment_reduced_bars,
            step_start_carries,
            step_primal_records,
            active_segment_arrays,
            active_support,
        )

    compiled_replay = jax.jit(_replay_kernel, inline=False)
    compiled_schedule_probe = jax.jit(
        _schedule_probe_kernel,
        static_argnums=(3, 4, 5),
        inline=False,
    )
    compiled_database_bwd = jax.jit(
        _database_bwd_kernel,
        static_argnums=(1,),
        inline=False,
    )
    return DatabaseFullTransportReplayOptimizationStage(
        support_layout=support_layout,
        initial_state_unpack=initial_state_unpack,
        solver=solver,
        species=species,
        equation_system_template=equation_system_template,
        compiled_replay=compiled_replay,
        compiled_database_bwd=compiled_database_bwd,
        compiled_schedule_probe=compiled_schedule_probe,
        support_floating_signature=_tree_signature(support_floating_leaves),
        initial_state_signature=_tree_signature(initial_flat_state),
        segment_carry_signature=_tree_signature(segment_start_carry),
        segment_arrays_signature=_tree_signature(segment_arrays),
    )
