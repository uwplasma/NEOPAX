"""Exact local checks for the opt-in geometry-only NTX support pullback."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import ntx

from NEOPAX._transport_flux_models import (
    NTXExactLijRuntimeTransportModel,
    _sanitize_float_delta_bar_tree,
)
from NEOPAX._transport_solvers import _radau_prepare_lagged_response_with_compact_coefficient_record


def _small_runtime_model(n_energy=1):
    """Construct only the moment-weight state needed by the local adapter."""

    model = object.__new__(NTXExactLijRuntimeTransportModel)
    weights = jnp.linspace(0.8, 1.2, n_energy)
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
    object.__setattr__(model, "derivative_mode", "direct")
    object.__setattr__(model, "scan_batch_size", None)
    object.__setattr__(model, "use_remat", False)
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


def test_mock_multi_rhs_anchor_adapter_uses_one_local_primal_context():
    """The planned adapter keeps primal state local and batches only RHS work.

    This is intentionally a pure-array mock: it models the ownership and axis
    contract of the future NEOPAX/NTX anchor adapter without constructing an
    NTX system or invoking any solver.
    """

    call_count = {"local_primal": 0}

    def _build_local_primal_context(anchor_input):
        call_count["local_primal"] += 1
        return (
            2.0 * anchor_input,
            anchor_input + 3.0,
            anchor_input - 5.0,
        )

    def _support_adjoint_from_context(context, rhs_bar):
        base, d_er, d_log_nu = context
        return {
            "prepared": rhs_bar * (base + d_er),
            "drds": rhs_bar * (base - d_log_nu),
        }

    def _multi_rhs_anchor_adapter(anchor_input, rhs_bars):
        # This is the intended adapter ownership: one local primal context,
        # with only cotangent-dependent work carrying the leading RHS axis.
        local_context = _build_local_primal_context(anchor_input)
        return jax.vmap(
            lambda rhs_bar: _support_adjoint_from_context(local_context, rhs_bar)
        )(rhs_bars)

    anchor_input = jnp.asarray(1.25)
    rhs_bars = jnp.asarray([0.5, -1.0, 2.0, 0.25])
    actual = _multi_rhs_anchor_adapter(anchor_input, rhs_bars)
    assert call_count["local_primal"] == 1
    assert actual["prepared"].shape == (4,)
    assert actual["drds"].shape == (4,)

    expected = jax.vmap(
        lambda rhs_bar: _support_adjoint_from_context(
            (2.0 * anchor_input, anchor_input + 3.0, anchor_input - 5.0),
            rhs_bar,
        )
    )(rhs_bars)
    _assert_float_tree_allclose(actual, expected)

    traced = jax.make_jaxpr(_multi_rhs_anchor_adapter)(anchor_input, rhs_bars)
    assert "scan[" not in str(traced)
    assert "map[" not in str(traced)


def test_multi_rhs_prepared_support_adapter_matches_scalar_local_pullbacks():
    """Exact local gate for the unselected multi-RHS NTX adapter.

    The ordinary helper is evaluated once per RHS only as the test oracle.
    The implementation under test receives the complete RHS batch and must
    preserve the scalar prepared, ``drds``, and interpolation-primal values.
    This test intentionally has no transport rollout, filesystem output, or
    profiling side effect.
    """

    model = _small_runtime_model(n_energy=1)
    prepared = ntx.prepare_monoenergetic_system(
        ntx.example_surface(),
        ntx.GridSpec(5, 5, 4),
    )
    scalar_field_bars = (
        (
            jnp.asarray(0.0),
            jnp.asarray([0.3, -0.2, 0.1, 0.4, -0.1, 0.2]),
            jnp.asarray([-0.3, 0.1, 0.2, -0.2, 0.3, -0.1]),
            jnp.asarray([0.2, 0.4, -0.3, 0.1, 0.2, -0.4]),
        ),
        (
            jnp.asarray(0.0),
            jnp.asarray([-0.1, 0.2, 0.3, -0.4, 0.5, -0.2]),
            jnp.asarray([0.4, -0.3, 0.2, 0.1, -0.5, 0.3]),
            jnp.asarray([-0.2, 0.3, 0.4, -0.1, 0.2, 0.5]),
        ),
    )
    batched_field_bars = tuple(
        jnp.stack([field_bars[field_index] for field_bars in scalar_field_bars])
        for field_index in range(4)
    )
    args = dict(
        drds_value=jnp.asarray(1.2),
        reference_nu_hat=jnp.asarray([1.0e-2]),
        reference_epsi_hat=jnp.asarray([1.0e-3]),
        vth_a=jnp.asarray([1.1]),
    )
    actual_prepared, actual_drds, actual_primal = (
        model._pullback_interpolated_moment_prepared_support_and_drds_only_multi_rhs(
            prepared,
            field_bars=batched_field_bars,
            **args,
        )
    )
    for rhs_index, field_bars in enumerate(scalar_field_bars):
        expected_prepared, expected_drds, expected_primal = (
            model._pullback_interpolated_moment_prepared_support_and_drds_only(
                prepared,
                field_bars=field_bars,
                **args,
            )
        )
        _assert_float_tree_allclose(
            jax.tree_util.tree_map(lambda value: value[rhs_index], actual_prepared),
            expected_prepared,
        )
        _assert_float_tree_allclose(actual_drds[rhs_index], expected_drds)
        _assert_float_tree_allclose(actual_primal, expected_primal)


def test_compact_local_coefficient_record_matches_ordinary_response():
    """The record adapter exposes existing coefficient primitives exactly."""

    model = _small_runtime_model(n_energy=2)
    prepared = ntx.prepare_monoenergetic_system(
        ntx.example_surface(),
        ntx.GridSpec(5, 5, 4),
    )
    args = dict(
        drds_value=jnp.asarray(1.2),
        nu_hat_a=jnp.asarray([1.0e-2, 1.8e-2]),
        epsi_hat_a=jnp.asarray([1.0e-3, 2.0e-3]),
        vth_a=jnp.asarray([1.1, 1.2]),
    )

    reference = model._interpolated_moment_reduced_local_outputs_from_primitives(
        prepared,
        **args,
    )
    actual, record = (
        model._interpolated_moment_reduced_local_outputs_with_coefficient_record_from_primitives(
            prepared,
            **args,
        )
    )

    _assert_float_tree_allclose(actual, reference)
    assert record.coefficient_scan.shape == (2, 5)
    assert record.dcoefficient_scan_d_er.shape == (2, 5)
    assert record.dcoefficient_scan_d_log_nu_star.shape == (2, 5)
    assert all(
        jnp.all(jnp.isfinite(leaf))
        for leaf in jax.tree_util.tree_leaves(record)
    )

    shaped_response, shaped_record = jax.eval_shape(
        lambda nu_hat_a, epsi_hat_a, vth_a: model._interpolated_moment_reduced_local_outputs_with_coefficient_record_from_primitives(
            prepared,
            drds_value=jnp.asarray(1.2),
            nu_hat_a=nu_hat_a,
            epsi_hat_a=epsi_hat_a,
            vth_a=vth_a,
        ),
        args["nu_hat_a"],
        args["epsi_hat_a"],
        args["vth_a"],
    )
    assert jax.tree_util.tree_structure(shaped_response) == jax.tree_util.tree_structure(reference)
    assert jax.tree_util.tree_structure(shaped_record) == jax.tree_util.tree_structure(record)
    assert shaped_record.coefficient_scan.shape == record.coefficient_scan.shape


def test_compact_record_lagged_preparation_preserves_rebuild_and_reuse_contract():
    """The experimental Radau hook returns one paired result without carry growth."""

    kernel_context = SimpleNamespace(use_transport_lagged_response=True)
    zero_record = {"coefficients": jnp.zeros((1, 1, 1, 5))}

    def build_with_record(state):
        return {
            "response": state + 2.0
        }, {"coefficients": jnp.broadcast_to(state[None, None, :, None], (1, 1, 1, 5))}

    rebuild_carry = SimpleNamespace(
        y=jnp.asarray([3.0]),
        lagged_response_valid=jnp.asarray(False),
        lagged_response_cache={"response": jnp.asarray([-1.0])},
        lagged_reference_y=jnp.asarray([-2.0]),
    )
    response, reference_y, reused, record = _radau_prepare_lagged_response_with_compact_coefficient_record(
        kernel_context,
        rebuild_carry,
        lambda y: y,
        None,
        build_with_record,
        lambda: zero_record,
    )
    assert jnp.allclose(response["response"], jnp.asarray([5.0]))
    assert jnp.allclose(reference_y, rebuild_carry.y)
    assert not bool(reused)
    assert jnp.allclose(record["coefficients"], jnp.full((1, 1, 1, 5), 3.0))

    reuse_carry = SimpleNamespace(
        y=jnp.asarray([4.0]),
        lagged_response_valid=jnp.asarray(True),
        lagged_response_cache={"response": jnp.asarray([7.0])},
        lagged_reference_y=jnp.asarray([6.0]),
    )
    response, reference_y, reused, record = _radau_prepare_lagged_response_with_compact_coefficient_record(
        kernel_context,
        reuse_carry,
        lambda y: y,
        None,
        build_with_record,
        lambda: zero_record,
    )
    assert jnp.allclose(response["response"], reuse_carry.lagged_response_cache["response"])
    assert jnp.allclose(reference_y, reuse_carry.lagged_reference_y)
    assert bool(reused)
    _assert_float_tree_allclose(record, zero_record)
