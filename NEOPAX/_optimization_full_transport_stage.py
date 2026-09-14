"""Optimization-only persistent boundaries for full-transport reverse.

The benchmark reverse lane constructs a fresh Radau execution context for
each geometry.  Its segment replay call deliberately treats that context as
static, which is appropriate for a one-shot benchmark but creates one JAX
cache entry per optimizer evaluation.  This module keeps the benchmark calls
unchanged and provides persistent replay and database-backward scans whose
geometry/database arrays are explicit dynamic inputs.  The backward scan
reuses the fixed solver structure and rebuilds only its support-dependent
direct-state transpose, rather than tracing a complete rollout/context build.
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
    runtime_with_fresh_ntx_database_payload,
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
class DatabaseFullTransportSupportOptimizationStage:
    """Persistent small VJP boundaries outside the segmented Radau sweep.

    The stage retains only structural templates and compiled callables.  The
    current profile vector, state cotangents, final state, geometry, and
    database leaves remain explicit arguments on every optimization trial.
    """

    support_layout: FloatingPayloadLeafLayout
    geometry_layout: FloatingPayloadLeafLayout
    support_floating_signature: Any
    geometry_floating_signature: Any
    ordinary_objective_indices: tuple[int, ...]
    compiled_profile_primal: Any
    compiled_profile_pullback: Any
    compiled_root_pullback: Any
    compiled_grouped_final_objective: Any
    compiled_flat_state_pullback: Any

    def _support_leaves(self, support_payload):
        self.support_layout.validate_static_structure(support_payload)
        leaves = self.support_layout.floating_leaves(support_payload)
        if _tree_signature(leaves) != self.support_floating_signature:
            raise ValueError(
                "Full-transport support boundary shape, dtype, or weak type "
                "changed within a stage."
            )
        return leaves

    def _geometry_leaves(self, geometry):
        self.geometry_layout.validate_static_structure(geometry)
        leaves = self.geometry_layout.floating_leaves(geometry)
        if _tree_signature(leaves) != self.geometry_floating_signature:
            raise ValueError(
                "Full-transport terminal geometry shape, dtype, or weak type "
                "changed within a stage."
            )
        return leaves

    def profile_state_primal(self, parameter_values, support_payload):
        return self.compiled_profile_primal(
            parameter_values,
            self._geometry_leaves(support_payload["geometry"]),
        )

    def profile_parameter_pullback(
        self, parameter_values, state_bars, support_payload
    ):
        return self.compiled_profile_pullback(
            parameter_values,
            state_bars,
            self._geometry_leaves(support_payload["geometry"]),
        )

    def root_pullback(
        self,
        state,
        er_profile,
        finite_mask,
        initial_state_bars,
        support_payload,
    ):
        return self.compiled_root_pullback(
            state,
            er_profile,
            finite_mask,
            initial_state_bars,
            self._support_leaves(support_payload),
        )

    def grouped_final_objective_rows(
        self, final_y, geometry, objective_indices
    ):
        if tuple(int(index) for index in objective_indices) != self.ordinary_objective_indices:
            raise ValueError(
                "Full-transport terminal objective rows changed within a stage."
            )
        return self.compiled_grouped_final_objective(
            final_y, self._geometry_leaves(geometry)
        )

    def flat_state_pullback(self, final_y, state_bar):
        return self.compiled_flat_state_pullback(final_y, state_bar)

    def dependencies_for(self, base_dependencies, support_payload):
        """Bind this stage to the current trial leaves without tracing them."""

        return dataclasses.replace(
            base_dependencies,
            profile_state_primal=lambda parameter_values: self.profile_state_primal(
                parameter_values, support_payload
            ),
            profile_parameter_pullback=lambda parameter_values, state_bars: (
                self.profile_parameter_pullback(
                    parameter_values, state_bars, support_payload
                )
            ),
            database_initial_er_root_pullback=(
                lambda state, er_profile, finite_mask, initial_state_bars, _support: (
                    self.root_pullback(
                        state,
                        er_profile,
                        finite_mask,
                        initial_state_bars,
                        support_payload,
                    )
                )
            ),
            grouped_joint_final_objective_vjp_rows=(
                self.grouped_final_objective_rows
            ),
            flat_state_pullback=self.flat_state_pullback,
        )

    def cache_sizes(self) -> tuple[int | None, ...]:
        sizes = []
        for compiled in (
            self.compiled_profile_primal,
            self.compiled_profile_pullback,
            self.compiled_root_pullback,
            self.compiled_grouped_final_objective,
            self.compiled_flat_state_pullback,
        ):
            cache_size = getattr(compiled, "_cache_size", None)
            try:
                sizes.append(None if not callable(cache_size) else int(cache_size()))
            except Exception:
                sizes.append(None)
        return tuple(sizes)


def build_database_full_transport_support_optimization_stage(
    *,
    config,
    runtime,
    baseline_state,
    profile_cfg,
    reverse_setup,
    support_payload,
    ordinary_objective_indices,
    dependencies,
    initial_state_for_parameter_vector_compact,
    compact_initial_er_database_support_bars,
    compact_initial_er_database_geometry_bars,
    objective_vector_joint_vjp_rows,
    float_delta_tree_like,
    add_float_delta_tree,
) -> DatabaseFullTransportSupportOptimizationStage:
    """Compile the measured non-segment operations once per optimizer stage."""

    if not (
        isinstance(support_payload, dict)
        and set(support_payload) == {"geometry", "database"}
    ):
        raise ValueError(
            "Full-transport support optimization requires exactly "
            "{'geometry', 'database'} support."
        )
    support_layout = FloatingPayloadLeafLayout.from_template(support_payload)
    support_leaves = support_layout.floating_leaves(support_payload)
    geometry_layout = FloatingPayloadLeafLayout.from_template(
        support_payload["geometry"]
    )
    geometry_leaves = geometry_layout.floating_leaves(
        support_payload["geometry"]
    )
    runtime_template = runtime
    number_species = runtime.species.number_species
    unpack_flat = reverse_setup.prepared_rollout.physics_context.unpack_flat
    root_interpolation_mode = str(
        reverse_setup.execution_context.physics_context
        .reverse_database_root_interpolation_transpose_mode
    )
    ordinary_objective_indices = tuple(
        int(index) for index in ordinary_objective_indices
    )

    def _active_support(active_support_leaves):
        return support_layout.rebuild(active_support_leaves)

    def _active_runtime(active_support_leaves):
        active_support = _active_support(active_support_leaves)
        return runtime_with_fresh_ntx_database_payload(
            runtime_template,
            geometry=active_support["geometry"],
            database=active_support["database"],
        )

    def _profile_state(active_parameters, active_geometry_leaves):
        active_geometry = geometry_layout.rebuild(active_geometry_leaves)
        return initial_state_for_parameter_vector_compact(
            active_parameters,
            config=config,
            initial_er_root_ad="off",
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            geometry=active_geometry,
            number_species=number_species,
        )

    def _profile_pullback(
        active_parameters, active_state_bars, active_geometry_leaves
    ):
        _, pullback = jax.vjp(
            lambda values: _profile_state(values, active_geometry_leaves),
            active_parameters,
        )
        return jax.vmap(lambda state_bar: pullback(state_bar)[0])(
            active_state_bars
        )

    def _root_pullback(
        active_state,
        er_profile,
        finite_mask,
        initial_state_bars,
        active_support_leaves,
    ):
        active_support = _active_support(active_support_leaves)
        active_runtime = _active_runtime(active_support_leaves)
        dres_der = dependencies.initial_er_charge_flux_residual_er_derivative(
            active_state,
            er_profile,
            runtime=active_runtime,
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
        state_residual_bars = dependencies.compact_initial_er_state_pullback(
            residual_scalar_fn=dependencies.initial_er_charge_flux_residual_scalar,
            state=active_state,
            er_profile=er_profile,
            residual_bars=residual_bars,
            runtime=active_runtime,
        )
        direct_initial_state_bars = dataclasses.replace(
            initial_state_bars,
            Er=jnp.zeros_like(initial_state_bars.Er),
        )
        pre_root_initial_state_bars = dependencies.add_trees(
            direct_initial_state_bars,
            state_residual_bars,
        )
        database_bars = compact_initial_er_database_support_bars(
            runtime=active_runtime,
            state=active_state,
            er_profile=er_profile,
            residual_bars=residual_bars,
            support=active_support,
            interpolation_transpose_mode=root_interpolation_mode,
        )

        geometry_bars = compact_initial_er_database_geometry_bars(
            runtime=active_runtime,
            state=active_state,
            er_profile=er_profile,
            residual_bars=residual_bars,
            support=active_support,
        )
        return (
            pre_root_initial_state_bars,
            residual_bars,
            {"geometry": geometry_bars, "database": database_bars},
        )

    def _grouped_final_objective(active_final_y, active_geometry_leaves):
        active_geometry = geometry_layout.rebuild(active_geometry_leaves)
        geometry_delta0 = float_delta_tree_like(active_geometry)

        def _objective_vector(final_y_value, geometry_delta):
            final_state = unpack_flat(final_y_value)
            active_runtime = dataclasses.replace(
                runtime_template,
                geometry=add_float_delta_tree(active_geometry, geometry_delta),
            )
            return jnp.stack(
                tuple(
                    dependencies.objective_scalar_by_index(
                        final_state, active_runtime, objective_index
                    )
                    for objective_index in ordinary_objective_indices
                ),
                axis=0,
            )

        return objective_vector_joint_vjp_rows(
            _objective_vector,
            active_final_y,
            geometry_delta0,
        )

    def _flat_state_pullback(final_y, state_bar):
        _, pullback = jax.vjp(unpack_flat, final_y)
        return pullback(state_bar)[0]

    return DatabaseFullTransportSupportOptimizationStage(
        support_layout=support_layout,
        geometry_layout=geometry_layout,
        support_floating_signature=_tree_signature(support_leaves),
        geometry_floating_signature=_tree_signature(geometry_leaves),
        ordinary_objective_indices=ordinary_objective_indices,
        compiled_profile_primal=jax.jit(_profile_state, inline=False),
        compiled_profile_pullback=jax.jit(_profile_pullback, inline=False),
        compiled_root_pullback=jax.jit(_root_pullback, inline=False),
        compiled_grouped_final_objective=jax.jit(
            _grouped_final_objective, inline=False
        ),
        compiled_flat_state_pullback=jax.jit(
            _flat_state_pullback, inline=False
        ),
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

    template_execution_context = reverse_setup.execution_context
    template_unpack_flat = template_physics_context.unpack_flat
    template_unpack_bar = getattr(
        template_unpack_flat,
        "cotangent",
        template_unpack_flat,
    )
    template_pack_flat = template_physics_context.pack_flat
    template_project_flat = template_physics_context.project_flat

    def _direct_state_pullback_with_support(
        active_support,
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

        active_equation_system = _equation_system_with_fresh_database_payload(
            equation_system_template,
            active_support,
            static_ntss_density_indices=static_ntss_density_indices,
        )
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

        def _active_direct_state_pullback(
            t_value,
            flat_y,
            lagged_response,
            rhs_bar_flat,
        ):
            return _direct_state_pullback_with_support(
                active_support,
                t_value,
                flat_y,
                lagged_response,
                rhs_bar_flat,
            )

        active_physics_context = dataclasses.replace(
            template_physics_context,
            flat_rhs_direct_black_box_state_pullback=(
                _active_direct_state_pullback
            ),
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
        support_floating_signature=_tree_signature(support_floating_leaves),
        initial_state_signature=_tree_signature(initial_flat_state),
        segment_carry_signature=_tree_signature(segment_start_carry),
        segment_arrays_signature=_tree_signature(segment_arrays),
    )
