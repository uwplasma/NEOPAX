"""Exact local checks for the opt-in geometry-only NTX support pullback."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import ntx

from NEOPAX._transport_flux_models import (
    NTXExactLijRuntimeTransportModel,
    _sanitize_float_delta_bar_tree,
)


def _small_runtime_model():
    """Construct only the moment-weight state needed by the local adapter."""

    model = object.__new__(NTXExactLijRuntimeTransportModel)
    weights = jnp.asarray([1.0])
    object.__setattr__(
        model,
        "energy_grid",
        SimpleNamespace(
            v_norm=1.0,
            xWeights=weights,
            L11_weight=weights,
            L12_weight=weights,
            L22_weight=weights,
            L13_weight=weights,
            L23_weight=weights,
            L33_weight=weights,
        ),
    )
    return model


def _assert_float_tree_allclose(actual, expected):
    for actual_leaf, expected_leaf in zip(
        jax.tree_util.tree_leaves(actual),
        jax.tree_util.tree_leaves(expected),
        strict=True,
    ):
        if jnp.issubdtype(jnp.asarray(expected_leaf).dtype, jnp.inexact):
            assert jnp.allclose(actual_leaf, expected_leaf, rtol=1e-9, atol=1e-11)


def test_geometry_implicit_local_support_pullback_matches_prepared_support_path():
    """The new representation changes no active support or response value."""

    model = _small_runtime_model()
    prepared = ntx.prepare_monoenergetic_system(
        ntx.example_surface(),
        ntx.GridSpec(5, 5, 4),
    )
    args = dict(
        drds_value=jnp.asarray(1.2),
        reference_nu_hat=jnp.asarray([1.0e-2]),
        reference_epsi_hat=jnp.asarray([1.0e-3]),
        vth_a=jnp.asarray([1.1]),
        field_bars=(
            jnp.asarray(0.0),
            jnp.asarray([0.3, -0.2, 0.1, 0.4, -0.1, 0.2]),
            jnp.asarray([-0.3, 0.1, 0.2, -0.2, 0.3, -0.1]),
            jnp.asarray([0.2, 0.4, -0.3, 0.1, 0.2, -0.4]),
        ),
    )

    prepared_result = model._pullback_interpolated_moment_prepared_support_and_drds_only(
        prepared,
        **args,
    )
    geometry_result = model._pullback_interpolated_moment_prepared_support_and_drds_only(
        prepared,
        **args,
        geometry_implicit_ntx_two_directional=True,
    )

    _assert_float_tree_allclose(
        geometry_result[0].geometry,
        prepared_result[0].geometry,
    )
    _assert_float_tree_allclose(geometry_result[1], prepared_result[1])
    _assert_float_tree_allclose(geometry_result[2], prepared_result[2])

    fixed_leaf_pairs = zip(
        jax.tree_util.tree_leaves(geometry_result[0]),
        jax.tree_util.tree_leaves(prepared),
        strict=True,
    )
    for bar_leaf, primal_leaf in fixed_leaf_pairs:
        if jnp.issubdtype(jnp.asarray(primal_leaf).dtype, jnp.inexact):
            assert jnp.all(jnp.isfinite(jnp.asarray(bar_leaf)))


def test_geometry_implicit_local_support_pullback_keeps_objective_axis_device_batchable():
    """The adapter has no scalar objective loop or static-leaf batching error."""

    model = _small_runtime_model()
    prepared = ntx.prepare_monoenergetic_system(
        ntx.example_surface(),
        ntx.GridSpec(5, 5, 4),
    )
    base_field_bars = (
        jnp.asarray(0.0),
        jnp.asarray([0.3, -0.2, 0.1, 0.4, -0.1, 0.2]),
        jnp.asarray([-0.3, 0.1, 0.2, -0.2, 0.3, -0.1]),
        jnp.asarray([0.2, 0.4, -0.3, 0.1, 0.2, -0.4]),
    )

    def _one_objective(scale):
        result = model._pullback_interpolated_moment_prepared_support_and_drds_only(
            prepared,
            drds_value=jnp.asarray(1.2),
            reference_nu_hat=jnp.asarray([1.0e-2]),
            reference_epsi_hat=jnp.asarray([1.0e-3]),
            vth_a=jnp.asarray([1.1]),
            field_bars=tuple(scale * value for value in base_field_bars),
            geometry_implicit_ntx_two_directional=True,
        )
        return (
            tuple(
                jax.tree_util.tree_leaves(
                    _sanitize_float_delta_bar_tree(prepared, result[0])
                )
            ),
            result[1],
            result[2],
        )

    batched_bars = jax.vmap(_one_objective)(jnp.asarray([0.5, 1.5]))
    for leaf in jax.tree_util.tree_leaves(batched_bars):
        assert leaf.shape[0] == 2
