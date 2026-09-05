"""Reverse-AD helpers for differentiable initial-Er ambipolar roots."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from ._ambipolarity import solve_ambipolarity_roots_radial_jax
from ._entropy_models import get_entropy_model
from ._monoenergetic import database_with_geometry_scale
from ._transport_flux_models import (
    DENSITY_STATE_TO_PHYSICAL,
    NTXDatabaseTransportModel,
    NTXRuntimeScanTransportModel,
    _add_float_delta_tree,
    _collisionality_kind,
    _float_delta_tree_like,
    build_evaluated_transport_state,
    get_Thermodynamical_Forces_A1,
    get_Thermodynamical_Forces_A2,
    get_Thermodynamical_Forces_A3,
    get_v_thermal,
)


def initial_er_root_setup(config: dict, runtime):
    """Return the TOML/runtime-backed setup for selected ambipolar-Er AD."""

    amb_cfg = dict(config.get("ambipolarity", {}))
    # Keep the production TOML unchanged, but use JAX-side chunking for the
    # AD boundary so the root profile stays traced without fusing all radii and
    # scan points into one large NTX kernel.
    amb_cfg["er_ambipolar_blocksize"] = int(amb_cfg.get("er_ambipolar_blocksize", 1) or 1)
    amb_cfg["er_ambipolar_scan_batch_mode"] = "hybrid"
    model_name = str(amb_cfg.get("er_ambipolar_method", "two_stage")).lower()
    entropy_model_name = config.get("neoclassical", {}).get(
        "entropy_model",
        runtime.solver_parameters.get("neoclassical_flux_model", "ntx_database"),
    )
    entropy_model = get_entropy_model(entropy_model_name)
    params = {
        "species": runtime.species,
        "energy_grid": runtime.energy_grid,
        "geometry": runtime.geometry,
        "database": runtime.database,
        "solver_parameters": runtime.solver_parameters,
    }
    return amb_cfg, model_name, entropy_model, params


def initial_er_selected_root_profile(state, *, config: dict, runtime):
    """Return the selected ambipolar Er profile and finite-root mask."""

    amb_cfg, model_name, entropy_model, params = initial_er_root_setup(config, runtime)
    _, _, best_roots, _ = solve_ambipolarity_roots_radial_jax(
        state=state,
        config=config,
        params=params,
        model_name=model_name,
        flux_model=runtime.models.flux,
        entropy_model=entropy_model,
        amb_cfg=amb_cfg,
    )
    best_roots = jnp.asarray(best_roots, dtype=state.Er.dtype)
    finite_mask = jnp.isfinite(best_roots)
    return jnp.where(finite_mask, best_roots, state.Er), finite_mask


def initial_er_charge_flux_residuals(state, er_profile, *, runtime):
    """Return charge-weighted particle-flux residuals at the selected root."""

    charge_qp = jnp.asarray(runtime.species.charge_qp)
    state_with_er = dataclasses.replace(state, Er=er_profile)
    local_particle_flux = runtime.models.flux.build_local_particle_flux_evaluator(state_with_er)
    if local_particle_flux is None:
        raise ValueError("Initial-Er root AD requires a local particle-flux evaluator.")

    def _residual_i(i):
        gamma = local_particle_flux(i, er_profile[i])
        return jnp.sum(charge_qp * gamma)

    indices = jnp.arange(jnp.asarray(er_profile).shape[0], dtype=jnp.int32)
    return jax.lax.map(_residual_i, indices)


def initial_er_charge_flux_residual_scalar(state, er_profile, radius_index, *, runtime):
    """Return one scalar charge-flux residual for compact transposition."""

    charge_qp = jnp.asarray(runtime.species.charge_qp)
    state_with_er = dataclasses.replace(state, Er=er_profile)
    local_particle_flux = runtime.models.flux.build_local_particle_flux_evaluator(state_with_er)
    if local_particle_flux is None:
        raise ValueError("Initial-Er root AD requires a local particle-flux evaluator.")
    gamma = local_particle_flux(radius_index, er_profile[radius_index])
    return jnp.sum(charge_qp * gamma)


def initial_er_charge_flux_residual_er_derivative(state, er_profile, *, runtime):
    """Return d residual / d Er at each radius for selected-root AD."""

    charge_qp = jnp.asarray(runtime.species.charge_qp)
    er_profile = jnp.asarray(er_profile, dtype=state.Er.dtype)

    def _residual_i_er(i, er_value):
        er_eval = er_profile.at[i].set(er_value)
        state_with_er = dataclasses.replace(state, Er=er_eval)
        local_particle_flux = runtime.models.flux.build_local_particle_flux_evaluator(state_with_er)
        if local_particle_flux is None:
            raise ValueError("Initial-Er root AD requires a local particle-flux evaluator.")
        gamma = local_particle_flux(i, er_value)
        return jnp.sum(charge_qp * gamma)

    indices = jnp.arange(er_profile.shape[0], dtype=jnp.int32)
    return jax.lax.map(
        lambda i: jax.grad(lambda er_value: _residual_i_er(i, er_value))(er_profile[i]),
        indices,
    )


def _replace_ntx_support_payload_in_model(model, support):
    if model is None or not dataclasses.is_dataclass(model) or isinstance(model, type):
        return model, False
    if hasattr(model, "with_support_payload"):
        return model.with_support_payload(support), True
    updates = {}
    changed = False
    for field in dataclasses.fields(model):
        value = getattr(model, field.name)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            new_value, child_changed = _replace_ntx_support_payload_in_model(value, support)
            if child_changed:
                updates[field.name] = new_value
                changed = True
    if not changed:
        return model, False
    return dataclasses.replace(model, **updates), True


def runtime_with_ntx_support_payload(runtime, support):
    """Return runtime with an explicit NTX exact-runtime support payload."""

    flux_model, changed = _replace_ntx_support_payload_in_model(runtime.models.flux, support)
    if not changed:
        raise ValueError("Could not find an NTX exact-runtime model that accepts an explicit support payload.")
    return dataclasses.replace(
        runtime,
        models=dataclasses.replace(runtime.models, flux=flux_model),
    )


def _replace_geometry_payload_in_model(model, geometry):
    if model is None or not dataclasses.is_dataclass(model) or isinstance(model, type):
        return model, False
    if isinstance(model, NTXDatabaseTransportModel):
        return dataclasses.replace(
            model,
            geometry=geometry,
            database=database_with_geometry_scale(model.database, geometry.a_b),
        ), True
    updates = {}
    changed = False
    for field in dataclasses.fields(model):
        value = getattr(model, field.name)
        if field.name in {"geometry", "field"}:
            if value is not geometry:
                updates[field.name] = geometry
                changed = True
            continue
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            new_value, child_changed = _replace_geometry_payload_in_model(value, geometry)
            if child_changed:
                updates[field.name] = new_value
                changed = True
    if not changed:
        return model, False
    return dataclasses.replace(model, **updates), True


def find_database_payload_in_model(model):
    """Return the rebuilt database owned by a nested database flux model."""

    if isinstance(model, NTXDatabaseTransportModel):
        return model.database
    if dataclasses.is_dataclass(model) and not isinstance(model, type):
        for field in dataclasses.fields(model):
            found = find_database_payload_in_model(getattr(model, field.name))
            if found is not None:
                return found
    return None


def find_ntx_runtime_scan_model_in_model(model):
    """Return the nested live NTX scan model, if the runtime owns one."""

    if isinstance(model, NTXRuntimeScanTransportModel):
        return model
    if dataclasses.is_dataclass(model) and not isinstance(model, type):
        for field in dataclasses.fields(model):
            found = find_ntx_runtime_scan_model_in_model(getattr(model, field.name))
            if found is not None:
                return found
    return None


def realtime_geometry_payload_for_runtime(runtime):
    """Return the additive tagged geometry payload for a supported runtime.

    This is intentionally not yet consumed by the established exact reverse
    setup.  It gives database reverse setup a pure model capability boundary
    without changing the legacy exact payload shape.
    """

    runtime_scan = find_ntx_runtime_scan_model_in_model(runtime.models.flux)
    if runtime_scan is not None:
        return {
            "kind": "ntx_scan_runtime",
            "geometry": runtime.geometry,
            "channels": runtime_scan.channels,
            "surfaces": runtime_scan.scan_surfaces,
            "database": runtime_scan.database,
        }
    database = find_database_payload_in_model(runtime.models.flux)
    if database is not None:
        return {
            "kind": "ntx_database",
            "geometry": runtime.geometry,
            "database": database,
        }
    return {
        "kind": "ntx_exact",
        "geometry": runtime.geometry,
        "ntx_support": find_ntx_support_payload(runtime),
    }


def realtime_geometry_reverse_support_payload_for_runtime(runtime):
    """Return only the differentiable support leaves for a runtime payload.

    A live runtime scan deliberately excludes its cached database: it is
    regenerated from geometry/channels/surfaces by the NTX scan model during
    the support VJP.  Keeping the cache out prevents it becoming an unrelated
    independent cotangent leaf.
    """

    payload = realtime_geometry_payload_for_runtime(runtime)
    if payload["kind"] == "ntx_scan_runtime":
        return {
            "geometry": payload["geometry"],
            "channels": payload["channels"],
            "surfaces": payload["surfaces"],
        }
    if payload["kind"] == "ntx_exact":
        return {
            "geometry": payload["geometry"],
            "ntx_support": payload["ntx_support"],
        }
    if payload["kind"] == "ntx_database":
        return {
            "geometry": payload["geometry"],
            "database": payload["database"],
        }
    raise ValueError(f"Unknown realtime geometry payload kind {payload['kind']!r}.")


def _replace_database_payload_in_model(model, database):
    if model is None or not dataclasses.is_dataclass(model) or isinstance(model, type):
        return model, False
    if isinstance(model, NTXDatabaseTransportModel):
        return dataclasses.replace(model, database=database), True
    updates = {}
    changed = False
    for field in dataclasses.fields(model):
        value = getattr(model, field.name)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            replacement, child_changed = _replace_database_payload_in_model(value, database)
            if child_changed:
                updates[field.name] = replacement
                changed = True
    return (dataclasses.replace(model, **updates), True) if changed else (model, False)


def _replace_ntx_runtime_scan_payload_in_model(model, payload):
    if model is None or not dataclasses.is_dataclass(model) or isinstance(model, type):
        return model, False
    if isinstance(model, NTXRuntimeScanTransportModel):
        return model.with_runtime_scan_payload(
            geometry=payload["geometry"],
            channels=payload["channels"],
            scan_surfaces=payload["surfaces"],
            database=payload.get("database"),
        ), True
    updates = {}
    changed = False
    for field in dataclasses.fields(model):
        value = getattr(model, field.name)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            replacement, child_changed = _replace_ntx_runtime_scan_payload_in_model(value, payload)
            if child_changed:
                updates[field.name] = replacement
                changed = True
    return (dataclasses.replace(model, **updates), True) if changed else (model, False)


def runtime_with_realtime_geometry_payload(runtime, payload):
    """Replace a runtime from the tagged exact/database geometry payload.

    The tag is Python setup metadata, not a traced JAX value.  This helper is
    additive: the established exact callers continue to use their existing
    geometry/support replacement functions unchanged.
    """

    if not isinstance(payload, dict):
        raise TypeError("realtime geometry payload must be a mapping.")
    kind = str(payload.get("kind", "")).strip().lower()
    if kind == "ntx_exact":
        return runtime_with_ntx_support_payload(
            runtime_with_geometry_payload(runtime, payload["geometry"]),
            payload["ntx_support"],
        )
    if kind == "ntx_database":
        geometry = payload["geometry"]
        database = payload["database"]
        runtime_with_geometry = runtime_with_geometry_payload(runtime, geometry)
        flux_model, changed = _replace_database_payload_in_model(
            runtime_with_geometry.models.flux,
            database,
        )
        if not changed:
            raise ValueError("No NTX database transport model was found in the runtime.")
        return dataclasses.replace(
            runtime_with_geometry,
            database=database,
            models=dataclasses.replace(runtime_with_geometry.models, flux=flux_model),
        )
    if kind == "ntx_scan_runtime":
        flux_model, changed = _replace_ntx_runtime_scan_payload_in_model(
            runtime.models.flux,
            payload,
        )
        if not changed:
            raise ValueError("No live NTX runtime scan model was found in the runtime.")
        return dataclasses.replace(
            runtime,
            geometry=payload["geometry"],
            database=payload.get("database"),
            models=dataclasses.replace(runtime.models, flux=flux_model),
        )
    raise ValueError(f"Unknown realtime geometry payload kind {kind!r}.")


def runtime_with_realtime_geometry_reverse_support_payload(runtime, support_payload):
    """Rebuild ``runtime`` from the differentiable reverse support leaves.

    Unlike :func:`runtime_with_realtime_geometry_payload`, this accepts the
    payload tree owned by a reverse VJP.  In particular, a live NTX scan has
    no database leaf here: the database is regenerated from the supplied
    geometry, channels, and surfaces.  This keeps the cache on the primal
    side of the contract and gives later reverse boundaries one model-aware
    replacement function.
    """

    if not isinstance(support_payload, dict):
        raise TypeError("realtime geometry reverse support payload must be a mapping.")
    payload = realtime_geometry_payload_for_runtime(runtime)
    kind = str(payload["kind"])
    if kind == "ntx_scan_runtime":
        required = {"geometry", "channels", "surfaces"}
        missing = required.difference(support_payload)
        if missing:
            raise ValueError(
                "Live NTX scan reverse support payload is missing "
                f"{sorted(missing)!r}."
            )
        return runtime_with_realtime_geometry_payload(
            runtime,
            {
                "kind": kind,
                "geometry": support_payload["geometry"],
                "channels": support_payload["channels"],
                "surfaces": support_payload["surfaces"],
                # Deliberately clear the old cache. ``with_runtime_scan_payload``
                # rebuilds it through the live NTX scan when the model is used.
                "database": None,
            },
        )
    if kind == "ntx_exact":
        required = {"geometry", "ntx_support"}
        missing = required.difference(support_payload)
        if missing:
            raise ValueError(
                "Exact NTX reverse support payload is missing "
                f"{sorted(missing)!r}."
            )
        return runtime_with_realtime_geometry_payload(
            runtime,
            {"kind": kind, **support_payload},
        )
    if kind == "ntx_database":
        required = {"geometry", "database"}
        missing = required.difference(support_payload)
        if missing:
            raise ValueError(
                "NTX database reverse support payload is missing "
                f"{sorted(missing)!r}."
            )
        return runtime_with_realtime_geometry_payload(
            runtime,
            {"kind": kind, **support_payload},
        )
    raise ValueError(f"Unknown realtime geometry payload kind {kind!r}.")


def runtime_with_geometry_payload(runtime, geometry):
    """Return runtime with transport geometry payload replaced everywhere needed."""

    flux_model, _changed = _replace_geometry_payload_in_model(runtime.models.flux, geometry)
    database = find_database_payload_in_model(flux_model)
    return dataclasses.replace(
        runtime,
        geometry=geometry,
        database=runtime.database if database is None else database,
        models=dataclasses.replace(runtime.models, flux=flux_model),
    )


def find_ntx_support_payload_in_model(model):
    """Return the nested NTX support payload from a flux model."""

    support = getattr(model, "support", None)
    if support is not None and hasattr(model, "with_support_payload"):
        return support
    if dataclasses.is_dataclass(model) and not isinstance(model, type):
        for field in dataclasses.fields(model):
            found = find_ntx_support_payload_in_model(getattr(model, field.name))
            if found is not None:
                return found
    return None


def find_ntx_support_payload(runtime):
    """Return the preloaded NTX exact-runtime support payload from a runtime."""

    support = find_ntx_support_payload_in_model(runtime.models.flux)
    if support is None:
        raise ValueError("No preloaded NTX exact-runtime support payload was found in the realtime runtime.")
    return support


def find_ntx_exact_support_model(model):
    """Return the nested NTX exact-runtime model that owns prepared Lij solves."""

    if (
        model is not None
        and callable(getattr(model, "with_support_payload", None))
        and callable(getattr(model, "_solve_lij_prepared_local", None))
    ):
        return model
    if dataclasses.is_dataclass(model) and not isinstance(model, type):
        for field in dataclasses.fields(model):
            found = find_ntx_exact_support_model(getattr(model, field.name))
            if found is not None:
                return found
    return None


def compact_initial_er_ntx_support_pullback_leaves(
    *,
    runtime,
    state,
    er_profile,
    residual_bars,
    support,
):
    """Compact support pullback for the initial-Er ambipolar residual.

    This mirrors the local NTX particle-flux evaluator but transposes only the
    per-radius prepared support and drds entries. It avoids a full-payload VJP
    through ``build_local_particle_flux_evaluator`` and returns flat support-bar
    leaves in the ``NTXExactLijRuntimeSupport`` pytree order expected by the
    realtime-geometry reverse payload path.
    """

    model = find_ntx_exact_support_model(runtime.models.flux)
    if model is None:
        raise ValueError("Could not find an NTX exact-runtime model for compact initial-Er support pullback.")

    er_profile = jnp.asarray(er_profile, dtype=state.Er.dtype)
    residual_bars = jnp.asarray(residual_bars, dtype=state.Er.dtype)
    if residual_bars.ndim != 2:
        raise ValueError(
            "compact initial-Er support pullback expects residual_bars with shape "
            "(objective_count, radial_count)."
        )
    if er_profile.ndim != 1:
        raise ValueError("compact initial-Er support pullback expects a 1D er_profile.")
    if int(residual_bars.shape[1]) != int(er_profile.shape[0]):
        raise ValueError(
            "compact initial-Er support pullback residual radial dimension does not match er_profile: "
            f"residual_bars.shape={residual_bars.shape}, er_profile.shape={er_profile.shape}."
        )
    objective_count = int(residual_bars.shape[0])
    # Match ``NTXExactLijRuntimeTransportModel.build_local_particle_flux_evaluator``
    # exactly.  In particular, the wHe initial state has an inactive He
    # density at its configured floor.  The compact rule previously used the
    # global default floor and manually reconstructed gradients, whereas the
    # primal evaluator uses this model's configured floors, fixed-species
    # projection, and boundary treatment.  That mismatch made the compact
    # drds transpose nonfinite although the primal root residual was finite.
    evaluated = build_evaluated_transport_state(
        state,
        model.geometry,
        bc_density=model.bc_density,
        bc_temperature=model.bc_temperature,
        density_floor=model.density_floor,
        temperature_floor=model.temperature_floor,
    )
    density = evaluated.center.density
    temperature = evaluated.center.temperature
    v_thermal = get_v_thermal(model.species.mass, temperature)
    species_indices = jnp.arange(int(model.species.number_species), dtype=jnp.int32)
    charge_qp = jnp.asarray(runtime.species.charge_qp, dtype=state.Er.dtype)
    collisionality_kind = _collisionality_kind(model.collisionality_model)
    dndr_all = evaluated.density_grad_center
    dTdr_all = evaluated.temperature_grad_center

    radius_indices = jnp.arange(er_profile.shape[0], dtype=jnp.int32)

    def _batched_zero_tree_leaves(tree):
        return tuple(
            jnp.broadcast_to(
                jnp.zeros_like(jnp.asarray(leaf, dtype=jnp.float64))
                if not jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.inexact)
                else jnp.zeros_like(jnp.asarray(leaf)),
                (objective_count,) + jnp.asarray(leaf).shape,
            )
            for leaf in jax.tree_util.tree_leaves(tree)
        )

    center_channels_bar = jax.tree_util.tree_map(
        lambda leaf: jnp.broadcast_to(
            jnp.zeros_like(jnp.asarray(leaf, dtype=jnp.float64))
            if not jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.inexact)
            else jnp.zeros_like(jnp.asarray(leaf)),
            (objective_count,) + jnp.asarray(leaf).shape,
        ),
        support.center_channels,
    )
    center_prepared_bar_leaves = _batched_zero_tree_leaves(support.center_prepared)
    face_channels_bar_leaves = _batched_zero_tree_leaves(support.face_channels)
    face_prepared_bar_leaves = _batched_zero_tree_leaves(support.face_prepared)

    def _split_flat_vector(flat, sizes, shapes, treedef):
        leaves = []
        offset = 0
        for size, shape in zip(sizes, shapes, strict=True):
            leaves.append(jnp.reshape(flat[offset : offset + size], shape))
            offset += size
        return treedef.unflatten(leaves), flat[offset]

    def _accumulate_radius(carry, radius_index):
        channels_carry, prepared_leaf_carry = carry
        prepared = jax.tree_util.tree_map(
            lambda arr: jax.lax.dynamic_index_in_dim(arr, radius_index, axis=0, keepdims=False),
            support.center_prepared,
        )
        drds_value = jax.lax.dynamic_index_in_dim(
            support.center_channels.drds,
            radius_index,
            axis=0,
            keepdims=False,
        )
        er_scalar = jax.lax.dynamic_index_in_dim(er_profile, radius_index, axis=0, keepdims=False)
        temperature_local = jax.lax.dynamic_index_in_dim(temperature, radius_index, axis=1, keepdims=False)
        density_local = jax.lax.dynamic_index_in_dim(density, radius_index, axis=1, keepdims=False)
        vthermal_local = jax.lax.dynamic_index_in_dim(v_thermal, radius_index, axis=1, keepdims=False)
        gamma_bars = residual_bars[:, radius_index, None] * charge_qp[None, :]
        prepared_delta0 = _float_delta_tree_like(prepared)
        prepared_delta_leaves0, prepared_delta_treedef = jax.tree_util.tree_flatten(prepared_delta0)
        prepared_delta_shapes = tuple(jnp.asarray(leaf).shape for leaf in prepared_delta_leaves0)
        prepared_delta_sizes = tuple(int(jnp.asarray(leaf).size) for leaf in prepared_delta_leaves0)
        flat_delta0 = jnp.concatenate(
            [jnp.ravel(jnp.asarray(leaf)) for leaf in prepared_delta_leaves0]
            + [jnp.ravel(jnp.zeros_like(drds_value))]
        )

        def _gamma_from_local_support_flat(flat_delta):
            prepared_delta, drds_delta = _split_flat_vector(
                flat_delta,
                prepared_delta_sizes,
                prepared_delta_shapes,
                prepared_delta_treedef,
            )
            prepared_value = _add_float_delta_tree(prepared, prepared_delta)
            drds_local = drds_value + drds_delta
            er_local_profile = jnp.asarray(er_profile).at[radius_index].set(er_scalar)
            lij = jax.vmap(
                lambda species_index: model._solve_lij_prepared_local(
                    prepared_value,
                    drds_value=drds_local,
                    species_index=species_index,
                    er_value=er_scalar,
                    temperature_local=temperature_local,
                    density_local=density_local,
                    vthermal_local=vthermal_local,
                    collisionality_kind=collisionality_kind,
                    derivative_mode_override="direct",
                )
            )(species_indices)
            a1 = jax.vmap(
                lambda charge, density_a, temperature_a, dndr_a, dTdr_a: get_Thermodynamical_Forces_A1(
                    charge,
                    density_a,
                    temperature_a,
                    dndr_a,
                    dTdr_a,
                    er_local_profile,
                )
            )(model.species.charge, density, temperature, dndr_all, dTdr_all)
            a2 = jax.vmap(get_Thermodynamical_Forces_A2)(temperature, dTdr_all)
            a3 = get_Thermodynamical_Forces_A3(er_local_profile)
            density_phys = DENSITY_STATE_TO_PHYSICAL * density_local
            return -density_phys * (
                lij[:, 0, 0] * jax.lax.dynamic_index_in_dim(a1, radius_index, axis=1, keepdims=False)
                + lij[:, 0, 1] * jax.lax.dynamic_index_in_dim(a2, radius_index, axis=1, keepdims=False)
                + lij[:, 0, 2] * jax.lax.dynamic_index_in_dim(a3, radius_index, axis=0, keepdims=False)
            )

        local_jacobian = jax.jacrev(_gamma_from_local_support_flat)(flat_delta0)
        flat_bars = jnp.tensordot(gamma_bars, local_jacobian, axes=([1], [0]))
        prepared_flat_size = int(sum(prepared_delta_sizes))
        drds_bars = flat_bars[:, prepared_flat_size]

        updated_prepared_leaves = []
        offset = 0
        for carry_leaf, size, shape in zip(
            prepared_leaf_carry,
            prepared_delta_sizes,
            prepared_delta_shapes,
            strict=True,
        ):
            local_bar = jnp.reshape(flat_bars[:, offset : offset + size], (objective_count,) + shape)
            updated_prepared_leaves.append(carry_leaf.at[:, radius_index].add(local_bar))
            offset += size

        return (
            dataclasses.replace(
                channels_carry,
                drds=channels_carry.drds.at[:, radius_index].add(drds_bars),
            ),
            tuple(updated_prepared_leaves),
        ), None

    (center_channels_bar, center_prepared_bar_leaves), _ = jax.lax.scan(
        _accumulate_radius,
        (center_channels_bar, center_prepared_bar_leaves),
        radius_indices,
    )
    return (
        tuple(jax.tree_util.tree_leaves(center_channels_bar))
        + face_channels_bar_leaves
        + tuple(center_prepared_bar_leaves)
        + face_prepared_bar_leaves
    )


def compact_initial_er_state_pullback(
    *,
    residual_scalar_fn,
    state,
    er_profile,
    residual_bars,
    runtime,
):
    """Compact state pullback for the initial-Er ambipolar residual.

    The generic rule forms a VJP for the full radial residual vector and then
    batches over objective cotangents. That is the memory-heavy path that can
    OOM after the transport cotangent sweep. This transposes one scalar radial
    residual at a time and contracts all objective residual bars immediately,
    matching the compact/local behavior used for the NTX support payload.
    """

    er_profile = jnp.asarray(er_profile, dtype=state.Er.dtype)
    residual_bars = jnp.asarray(residual_bars, dtype=state.Er.dtype)
    if residual_bars.ndim == 1:
        residual_bars = residual_bars[None, :]
        squeeze_result = True
    elif residual_bars.ndim == 2:
        squeeze_result = False
    else:
        raise ValueError(
            "compact initial-Er state pullback expects residual_bars with shape "
            "(radial_count,) or (objective_count, radial_count)."
        )
    if er_profile.ndim != 1:
        raise ValueError("compact initial-Er state pullback expects a 1D er_profile.")
    if int(residual_bars.shape[1]) != int(er_profile.shape[0]):
        raise ValueError(
            "compact initial-Er state pullback residual radial dimension does not match er_profile: "
            f"residual_bars.shape={residual_bars.shape}, er_profile.shape={er_profile.shape}."
        )

    objective_count = int(residual_bars.shape[0])

    def _zero_batched_like(leaf):
        arr = jnp.asarray(leaf)
        if not jnp.issubdtype(arr.dtype, jnp.inexact):
            arr = arr.astype(jnp.float64)
        return jnp.broadcast_to(jnp.zeros_like(arr), (objective_count,) + arr.shape)

    state_bar0 = jax.tree_util.tree_map(_zero_batched_like, state)
    radius_indices = jnp.arange(er_profile.shape[0], dtype=jnp.int32)

    def _add_batched_trees(lhs, rhs):
        return jax.tree_util.tree_map(lambda a, b: a + b, lhs, rhs)

    def _accumulate_radius(carry, radius_index):
        _, residual_pullback = jax.vjp(
            lambda state_value: residual_scalar_fn(
                state_value,
                er_profile,
                radius_index,
                runtime=runtime,
            ),
            state,
        )
        (state_bar_i,) = residual_pullback(jnp.asarray(1.0, dtype=er_profile.dtype))
        weights = residual_bars[:, radius_index]

        def _scale_leaf(leaf):
            leaf_arr = jnp.asarray(leaf)
            return weights.reshape((objective_count,) + (1,) * leaf_arr.ndim) * leaf_arr

        return _add_batched_trees(carry, jax.tree_util.tree_map(_scale_leaf, state_bar_i)), None

    state_bars, _ = jax.lax.scan(_accumulate_radius, state_bar0, radius_indices)
    if squeeze_result:
        return jax.tree_util.tree_map(lambda leaf: leaf[0], state_bars)
    return state_bars
