"""Optimization-only persistent boundaries for full-transport replay.

The benchmark reverse lane constructs a fresh Radau execution context for
each geometry.  Its segment replay call deliberately treats that context as
static, which is appropriate for a one-shot benchmark but creates one JAX
cache entry per optimizer evaluation.  This module keeps the benchmark call
unchanged and provides the optimization lane with one compiled replay whose
geometry/database arrays are explicit dynamic inputs.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ._optimization_initial_root_stage import FloatingPayloadLeafLayout
from ._reverse_ad_initial_er import (
    _replace_geometry_and_fresh_database_payload_in_model,
)
from ._transport_equations import build_equation_system
from ._transport_solvers import (
    _build_prepared_radau_accepted_rollout,
    _build_prepared_radau_execution_context,
    _extract_fixed_temperature_projection,
    _extract_state_regularization,
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
        """Run the benchmark database segment body through one stable JIT."""

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
            initial_flat_state,
            support_floating_leaves,
            str(cotangent_mode),
            segment_reduced_bars,
            step_start_carries,
            step_primal_records,
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

    def database_bwd_cache_size(self) -> int | None:
        cache_size = getattr(self.compiled_database_bwd, "_cache_size", None)
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

    def _database_bwd_kernel(
        active_initial_flat_state,
        support_floating_leaves,
        cotangent_mode,
        segment_reduced_bars,
        step_start_carries,
        step_primal_records,
        active_segment_arrays,
    ):
        active_execution_context, active_support = _active_execution_context(
            active_initial_flat_state,
            support_floating_leaves,
        )
        # This is the exact body beneath the benchmark database backward JIT.
        # The optimization JIT owns the stable static context; all geometry,
        # database, carry, record, and schedule values remain dynamic inputs.
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
    compiled_database_bwd = jax.jit(
        _database_bwd_kernel,
        static_argnums=(2,),
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
        support_floating_signature=_tree_signature(support_floating_leaves),
        initial_state_signature=_tree_signature(initial_flat_state),
        segment_carry_signature=_tree_signature(segment_start_carry),
        segment_arrays_signature=_tree_signature(segment_arrays),
    )
