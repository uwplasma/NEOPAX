"""Exact local checks for the opt-in geometry-only NTX support pullback."""

from types import SimpleNamespace
import dataclasses
import inspect
from pathlib import Path

import jax
import jax.numpy as jnp
import ntx
import pytest

from NEOPAX._transport_flux_models import (
    CombinedTransportFluxModel,
    NTXExactLijRuntimeSupport,
    NTXExactLijRuntimeTransportModel,
    NTXRuntimeScanChannels,
    _sanitize_float_delta_bar_tree,
)
from NEOPAX._neoclassical import _collisionality_kind
from NEOPAX._energy_grid_models import StandardLaguerreEnergyGrid
from NEOPAX._species import Species
from NEOPAX._state import TransportState, get_v_thermal
from NEOPAX._transport_equations import ComposedEquationSystem
from NEOPAX._transport_solvers import (
    _flat_rhs_build_support_pullback_batched_interpolated_faces_factory,
    _flat_rhs_state_and_lagged_response_pullback_factory,
    _lagged_response_build_state_and_support_pullback_batched_interpolated_faces_hook,
    _radau_prepare_lagged_response_with_compact_coefficient_record,
)
from NEOPAX._geometry_autodiff import (
    _native_vmec_coefficient_tangent_contraction,
    _ntx_runtime_channel_payload_bars,
)
from NEOPAX._reverse_ad_transport import (
    _initial_cache_support_pullback_from_rebuild_dispatch,
    _initial_lagged_response_joint_state_and_support_pullback,
    _merge_rebuild_ntx_channels_into_generic_payload_bar,
    _objective_vector_vjp_rows,
    _take_batched_pytree_row,
)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _MockSupportBar:
    center_channels: object
    face_channels: object
    center_prepared: object
    face_prepared: object


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _ToyBootstrapGeometry:
    """Minimal differentiable geometry payload for the terminal-VJP gate."""

    scale: object


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _TestMomentumGeometry:
    """Minimum JAX-pytree geometry accepted by the momentum matrix JIT."""

    a_b: object
    r_grid: object
    r_grid_half: object
    Bsqav: object


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _ToyBootstrapChannels:
    """Only the sparse ``drds`` channel touched by the compact helper."""

    drds: object


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _ToyBootstrapSupport:
    center_channels: object
    face_channels: object
    center_prepared: object
    face_prepared: object


@dataclasses.dataclass(frozen=True)
class _ToyBootstrapModel:
    """Small analytic model exercising the real compact-helper contracts."""

    geometry: object
    support: object

    def with_support_payload(self, support):
        return dataclasses.replace(self, support=support)

    def _momentum_corrected_upar_one_radius(self, state, radius_index, *, support=None):
        support = self.support if support is None else support
        prepared = support.center_prepared[0][radius_index]
        drds = support.center_channels.drds[radius_index]
        # Keep state, sparse support, and geometry genuinely coupled, while
        # remaining a no-rollout analytic fixture.
        return (
            state.pressure[:, radius_index] * (1.0 + prepared)
            + state.density[:, radius_index] * drds
            + self.geometry.scale * (prepared + 0.5 * drds)
        )


def _small_runtime_model(n_energy=1):
    """Construct only the moment-weight state needed by the local adapter."""

    model = object.__new__(NTXExactLijRuntimeTransportModel)
    weights = jnp.linspace(0.8, 1.2, n_energy)
    object.__setattr__(
        model,
        "energy_grid",
        SimpleNamespace(
            # The production energy grid carries one normalized speed per
            # energy.  Keep that axis in the small two-energy fixture so its
            # ``lax.map`` reference path has the same contract.
            v_norm=jnp.linspace(0.9, 1.1, n_energy),
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


def _small_vmec_prepared():
    """A traceable VMEC prepared system for native-bridge contract gates."""

    surface = ntx.VmecSurface(
        path=Path("native-bridge-fixture.nc"),
        requested_psi_n=0.2,
        psi_n=0.2,
        nfp=2,
        ns=3,
        mpol=2,
        ntor=1,
        total_mode_count=2,
        loaded_mode_count=2,
        iota=jnp.asarray(0.6),
        m=jnp.asarray([0, 1]),
        n=jnp.asarray([0, 1]),
        b_cos=jnp.asarray([1.0, 0.1]),
        jacobian_cos=jnp.asarray([1.0, 0.02]),
        b_sub_theta_cos=jnp.asarray([0.2, 0.01]),
        b_sub_zeta_cos=jnp.asarray([1.1, 0.03]),
        b_sup_theta_cos=jnp.asarray([0.3, 0.04]),
        b_sup_zeta_cos=jnp.asarray([1.2, 0.05]),
        b0=jnp.asarray(1.0),
        psi_a_hat=jnp.asarray(1.0),
        phi_edge=jnp.asarray(1.0),
        r_n=jnp.asarray(0.5),
        r_hat=jnp.asarray(0.5),
        dpsi_hat_dr_hat=jnp.asarray(1.0),
        dr_hat_dpsi_hat=jnp.asarray(1.0),
        transport_psi_scale=jnp.asarray(1.0),
    )
    return surface, ntx.prepare_monoenergetic_system(surface, ntx.GridSpec(5, 5, 4))


def _assert_float_tree_allclose(actual, expected, *, rtol=1e-9, atol=1e-11):
    for leaf_index, (actual_leaf, expected_leaf) in enumerate(zip(
        jax.tree_util.tree_leaves(actual),
        jax.tree_util.tree_leaves(expected),
        strict=True,
    )):
        if jnp.issubdtype(jnp.asarray(expected_leaf).dtype, jnp.inexact):
            if not jnp.allclose(actual_leaf, expected_leaf, rtol=rtol, atol=atol):
                difference = jnp.abs(actual_leaf - expected_leaf)
                scale = jnp.maximum(jnp.abs(expected_leaf), 1.0e-30)
                raise AssertionError(
                    "float tree mismatch: "
                    f"leaf={leaf_index} "
                    f"max_abs={float(jnp.max(difference)):.16e} "
                    f"max_rel={float(jnp.max(difference / scale)):.16e}"
                )


def test_joint_momentum_corrected_upar_pullback_matches_separate_helpers():
    """One local bootstrap VJP preserves the three established contracts.

    This is deliberately analytic and contains no transport rollout, VMEC
    solve, profiler, or filesystem activity.  It exercises the exact sparse
    prepared-plus-``drds`` support layout used by the production helper.
    """

    geometry = _ToyBootstrapGeometry(scale=jnp.asarray(1.7))
    support = _ToyBootstrapSupport(
        center_channels=_ToyBootstrapChannels(
            drds=jnp.asarray([0.8, 1.1]),
        ),
        face_channels=(jnp.asarray([0.2, -0.4]),),
        center_prepared=(jnp.asarray([0.3, -0.25]),),
        face_prepared=(jnp.asarray([0.1, 0.5, -0.2]),),
    )
    model = _ToyBootstrapModel(geometry=geometry, support=support)
    state = TransportState(
        density=jnp.asarray([[1.0, 1.2], [0.9, 1.1]]),
        pressure=jnp.asarray([[1.3, 1.5], [1.1, 1.4]]),
        Er=jnp.asarray([2.0e-4, 2.5e-4]),
    )
    upar_bar = jnp.asarray([[0.4, -0.3], [-0.2, 0.5]])

    state_expected = (
        NTXExactLijRuntimeTransportModel.pullback_momentum_corrected_upar_state_by_radius(
            model, state, upar_bar
        )
    )
    support_expected = (
        NTXExactLijRuntimeTransportModel.pullback_momentum_corrected_upar_support_by_radius(
            model, state, upar_bar, support
        )
    )
    geometry_expected = (
        NTXExactLijRuntimeTransportModel.pullback_momentum_corrected_upar_geometry_by_radius(
            model, state, upar_bar, geometry, support
        )
    )
    state_actual, support_actual, geometry_actual = (
        NTXExactLijRuntimeTransportModel.pullback_momentum_corrected_upar_state_support_geometry_by_radius(
            model, state, upar_bar, geometry, support
        )
    )

    _assert_float_tree_allclose(state_actual, state_expected, rtol=1e-12, atol=1e-12)
    _assert_float_tree_allclose(support_actual, support_expected, rtol=1e-12, atol=1e-12)
    _assert_float_tree_allclose(geometry_actual, geometry_expected, rtol=1e-12, atol=1e-12)


def test_native_vmec_coefficient_bridge_extracts_runtime_channels_only():
    """The diagnostic channel projection cannot retain prepared leaves."""

    support_bars = (
        SimpleNamespace(
            center_channels=(jnp.asarray([1.0]),),
            face_channels=(jnp.asarray([2.0]),),
            center_prepared="center-prepared-must-not-pass",
            face_prepared="face-prepared-must-not-pass",
        ),
        SimpleNamespace(
            center_channels=(jnp.asarray([3.0]),),
            face_channels=(jnp.asarray([4.0]),),
            center_prepared="second-center-prepared-must-not-pass",
            face_prepared="second-face-prepared-must-not-pass",
        ),
    )

    actual = _ntx_runtime_channel_payload_bars(support_bars)
    assert len(actual) == 2
    _assert_float_tree_allclose(
        actual,
        (
            (support_bars[0].center_channels, support_bars[0].face_channels),
            (support_bars[1].center_channels, support_bars[1].face_channels),
        ),
    )
    assert "prepared" not in repr(actual).lower()


def test_native_vmec_bridge_merges_rebuild_channels_and_direct_geometry():
    """Native coefficients replace only face-prepared, never direct geometry."""

    generic_ntx = _MockSupportBar(
        center_channels=(jnp.asarray([1.0, -2.0]),),
        face_channels=(jnp.asarray([3.0, 4.0]),),
        center_prepared="retain-center-prepared",
        face_prepared="retain-face-prepared",
    )
    rebuild_ntx = _MockSupportBar(
        center_channels=(jnp.asarray([0.2, 0.3]),),
        face_channels=(jnp.asarray([-0.4, 0.5]),),
        center_prepared="native-replaces-only-this-center-branch",
        face_prepared="native-replaces-only-this-face-branch",
    )
    merged = _merge_rebuild_ntx_channels_into_generic_payload_bar(
        {"geometry": {"direct": jnp.asarray([1.3, -0.4])}, "ntx_support": generic_ntx},
        {"geometry": {"direct": jnp.asarray([-0.2, 0.7])}, "ntx_support": rebuild_ntx},
    )
    _assert_float_tree_allclose(
        merged["geometry"], {"direct": jnp.asarray([1.1, 0.3])}
    )
    assert merged["ntx_support"].center_prepared == "retain-center-prepared"
    assert merged["ntx_support"].face_prepared == "retain-face-prepared"
    _assert_float_tree_allclose(
        merged["ntx_support"].center_channels,
        (jnp.asarray([1.2, -1.7]),),
    )
    _assert_float_tree_allclose(
        merged["ntx_support"].face_channels,
        (jnp.asarray([2.6, 4.5]),),
    )


def test_grouped_objective_vjp_rows_match_scalar_objective_gradients():
    """The grouped terminal-rule algebra preserves independent objective rows."""

    primal = jnp.asarray([0.4, -0.7, 1.2])

    def objective_vector(value):
        return jnp.asarray(
            [
                value[0] ** 2 + value[1],
                value[0] * value[2] - jnp.sin(value[1]),
                jnp.sum(value**3),
            ]
        )

    values, grouped_bars = _objective_vector_vjp_rows(objective_vector, primal)
    expected_bars = jnp.stack(
        [jax.grad(lambda value, row=row: objective_vector(value)[row])(primal) for row in range(3)]
    )
    assert jnp.allclose(values, objective_vector(primal), rtol=1e-12, atol=1e-12)
    assert jnp.allclose(grouped_bars, expected_bars, rtol=1e-12, atol=1e-12)


def test_take_batched_pytree_row_slices_dataclass_leaves():
    """Grouped geometry bars are pytrees, not directly subscriptable arrays."""

    rows = _MockSupportBar(
        center_channels=(jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),),
        face_channels=(jnp.asarray([[5.0], [6.0]]),),
        center_prepared=jnp.asarray([[7.0, 8.0], [9.0, 10.0]]),
        face_prepared=jnp.asarray([[11.0], [12.0]]),
    )
    actual = _take_batched_pytree_row(rows, 1)
    expected = _MockSupportBar(
        center_channels=(jnp.asarray([3.0, 4.0]),),
        face_channels=(jnp.asarray([6.0]),),
        center_prepared=jnp.asarray([9.0, 10.0]),
        face_prepared=jnp.asarray([12.0]),
    )
    _assert_float_tree_allclose(actual, expected)


def test_initial_joint_lagged_pullback_retains_rhs_induced_support_bar():
    """The initial support bar must use cache plus initial-RHS lagged bars."""

    # Small exact analogue of the initial-carry graph:
    # lagged = 2 * state + 3 * support; rhs = 5 * lagged;
    # carry = (state + rhs, lagged-cache).  The carry cotangent reaches the
    # lagged response through both the cache and RHS paths.
    state = jnp.asarray([0.4, -0.7])
    support = jnp.asarray([1.1, -0.2])
    carry_y_bar = jnp.asarray([0.6, -0.3])
    cache_bar = jnp.asarray([-0.4, 0.9])

    def carry_from_state_and_support(state_value, support_value):
        lagged = 2.0 * state_value + 3.0 * support_value
        return state_value + 5.0 * lagged, lagged

    _, full_pullback = jax.vjp(carry_from_state_and_support, state, support)
    expected_state_bar, expected_support_bar = full_pullback((carry_y_bar, cache_bar))
    rhs_lagged_bar = 5.0 * carry_y_bar

    def joint_pullback(flat_y, total_lagged_bars, support_payload):
        del flat_y, support_payload
        return 2.0 * total_lagged_bars, 3.0 * total_lagged_bars

    joint_state_bar, joint_support_bar = (
        _initial_lagged_response_joint_state_and_support_pullback(
            flat_y=state,
            cache_lagged_bars=cache_bar,
            rhs_lagged_bars=rhs_lagged_bar,
            support_payload=support,
            joint_pullback=joint_pullback,
        )
    )
    # Add the direct carry-to-state contribution, as the real initial-carry
    # reverse does before its lagged-response transpose.
    joint_state_bar = joint_state_bar + carry_y_bar
    assert jnp.allclose(joint_state_bar, expected_state_bar, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(joint_support_bar, expected_support_bar, rtol=1e-12, atol=1e-12)

    # The present split support-only path would omit this real contribution.
    assert not jnp.allclose(3.0 * cache_bar, expected_support_bar)


def test_initial_rebuild_dispatch_uses_active_hook_and_keeps_vmec_channel_separate():
    """The initial-edge selector mirrors the separate rebuild hook contract."""

    calls = []

    def native_hook(flat_y, lagged_bars, support):
        calls.append((flat_y, lagged_bars, support))
        return {"support": lagged_bars + support}, {"b_cos": 2.0 * lagged_bars}

    context = SimpleNamespace(
        reverse_rebuild_support_pullback_mode=(
            "ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_"
            "shared_primal_with_vmec_coefficients"
        ),
        flat_rhs_build_support_pullback_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients=native_hook,
    )
    support_bars, native_bars = _initial_cache_support_pullback_from_rebuild_dispatch(
        physics_context=context,
        flat_y=jnp.asarray([1.0, 2.0]),
        lagged_response_bars=jnp.asarray([0.3, -0.4]),
        support_payload=jnp.asarray([0.5, 0.7]),
    )
    assert len(calls) == 1
    assert jnp.allclose(support_bars["support"], jnp.asarray([0.8, 0.3]))
    assert jnp.allclose(native_bars["b_cos"], jnp.asarray([0.6, -0.8]))


def test_initial_rebuild_dispatch_selects_direct_native_vmec_hook():
    """The direct product-rule selector cannot fall back to the old hook."""

    calls = []

    def direct_hook(flat_y, lagged_bars, support):
        calls.append((flat_y, lagged_bars, support))
        return {"support": lagged_bars + support}, {"b_cos": 3.0 * lagged_bars}

    context = SimpleNamespace(
        reverse_rebuild_support_pullback_mode=(
            "ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_"
            "shared_primal_with_vmec_coefficients_direct_directional_product_rule"
        ),
        flat_rhs_build_support_pullback_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_directional_product_rule=direct_hook,
    )
    support_bars, native_bars = _initial_cache_support_pullback_from_rebuild_dispatch(
        physics_context=context,
        flat_y=jnp.asarray([1.0, 2.0]),
        lagged_response_bars=jnp.asarray([0.3, -0.4]),
        support_payload=jnp.asarray([0.5, 0.7]),
    )
    assert len(calls) == 1
    assert jnp.allclose(support_bars["support"], jnp.asarray([0.8, 0.3]))
    assert jnp.allclose(native_bars["b_cos"], jnp.asarray([0.9, -1.2]))


def test_initial_rebuild_dispatch_matches_ordinary_batched_support_contract():
    """Non-native initial dispatch is exactly its active rebuild hook."""

    def ordinary_hook(flat_y, lagged_bars, support):
        return {
            "state_marker": flat_y * 0.0,
            "support": 1.7 * lagged_bars - 0.25 * support,
        }

    context = SimpleNamespace(
        reverse_rebuild_support_pullback_mode="ntx_batched_interpolated_faces",
        flat_rhs_build_support_pullback_batched_interpolated_faces=ordinary_hook,
    )
    flat_y = jnp.asarray([0.4, -0.8])
    lagged_bars = jnp.asarray([[0.3, -0.4], [0.2, 0.6]])
    support = jnp.asarray([1.1, -0.7])
    expected = ordinary_hook(flat_y, lagged_bars, support)
    actual, native_bars = _initial_cache_support_pullback_from_rebuild_dispatch(
        physics_context=context,
        flat_y=flat_y,
        lagged_response_bars=lagged_bars,
        support_payload=support,
    )
    _assert_float_tree_allclose(actual, expected)
    assert native_bars is None


def test_native_vmec_coefficient_tangent_contraction_matches_vjp_duality():
    """The compact tangent term equals the corresponding coefficient VJP dot."""

    coefficient_bars = (
        jnp.asarray([[0.2, -0.4], [0.7, 0.1], [-0.3, 0.5]]),
        jnp.asarray([[[0.1, 0.2], [-0.5, 0.3]], [[0.4, -0.2], [0.6, 0.1]], [[-0.3, 0.7], [0.2, -0.6]]]),
        jnp.asarray([0.6, -0.2, 0.9]),
    )
    coefficient_tangents = (
        jnp.asarray([1.5, -0.8]),
        jnp.asarray([[0.3, -0.1], [0.9, 0.2]]),
        jnp.asarray(1.7),
    )
    actual = _native_vmec_coefficient_tangent_contraction(
        coefficient_bars, coefficient_tangents
    )
    expected = jnp.asarray(
        [
            jnp.vdot(coefficient_bars[0][row], coefficient_tangents[0])
            + jnp.vdot(coefficient_bars[1][row], coefficient_tangents[1])
            + jnp.vdot(coefficient_bars[2][row], coefficient_tangents[2])
            for row in range(3)
        ]
    )
    assert jnp.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_native_vmec_bridge_forwards_bridge_only_flag_to_local_ntx_helper():
    """Catch a missing bridge-only keyword before a compiled Radau segment."""

    captured = {}

    class _Model:
        def _pullback_interpolated_moment_prepared_support_and_drds_only_multi_rhs(
            self, prepared, **kwargs
        ):
            captured["prepared"] = prepared
            captured.update(kwargs)
            return "native-result"

    method = (
        NTXExactLijRuntimeTransportModel.
        _pullback_interpolated_moment_prepared_support_and_drds_only_native_multi_rhs_reuse_moment_drds_jvp_with_vmec_coefficients
    )
    # This is the next hop used by the selected direct-directional rebuild
    # mode.  Keep the keyword at this boundary: otherwise the failure occurs
    # only after the expensive Radau reverse-segment trace has started.
    batched_method = (
        NTXExactLijRuntimeTransportModel.
        pullback_build_lagged_response_support_payload_batched_interpolated_faces_multi_rhs_shared_primal
    )
    assert "native_vmec_coefficient_bars_only" in inspect.signature(method).parameters
    assert "native_vmec_direct_directional_product_rule" in inspect.signature(method).parameters
    assert (
        "native_vmec_direct_directional_product_rule"
        in inspect.signature(batched_method).parameters
    )
    result = method(
        _Model(),
        "prepared",
        drds_value="drds",
        reference_nu_hat="nu",
        reference_epsi_hat="epsi",
        vth_a="vth",
        field_bars="bars",
        return_case_bars=True,
        native_vmec_coefficient_bars_only=True,
        native_vmec_direct_directional_product_rule=True,
    )
    assert result == "native-result"
    assert captured["return_native_vmec_coefficient_bars"] is True
    assert captured["native_vmec_coefficient_bars_only"] is True
    assert captured["native_vmec_direct_directional_product_rule"] is True
    assert captured["return_case_bars"] is True


def test_native_multi_rhs_composite_forwarding_hook_is_exposed_to_radau():
    """The vector-field owner must expose the native inner NTX hook.

    Radau owns a :class:`CombinedTransportFluxModel` method, rather than the
    exact-NTX component method directly.  This pure mock prevents a missing
    outer forwarding wrapper from becoming a late reverse-segment failure.
    """

    calls = []

    class _InnerNTX:
        def pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_shared_primal(
            self,
            state,
            response_bars,
            support,
        ):
            calls.append((state, response_bars, support))
            return "native-result"

    composite = object.__new__(CombinedTransportFluxModel)
    object.__setattr__(composite, "neoclassical_model", _InnerNTX())
    response = SimpleNamespace(neoclassical_response="ntx-bars")

    assert composite.pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_shared_primal(
        "state",
        response,
        "support",
        ignored_outer_keyword=True,
    ) == "native-result"
    assert calls == [("state", "ntx-bars", "support")]


def test_native_multi_rhs_reused_drds_composite_forwarding_hook_is_exposed_to_radau():
    """The new selector reaches only its dedicated inner hook."""

    calls = []

    class _InnerNTX:
        def pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal(
            self, state, response_bars, support,
        ):
            calls.append((state, response_bars, support))
            return "reused-drds-result"

    composite = object.__new__(CombinedTransportFluxModel)
    object.__setattr__(composite, "neoclassical_model", _InnerNTX())
    response = SimpleNamespace(neoclassical_response="ntx-bars")
    assert composite.pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal(
        "state", response, "support", ignored_outer_keyword=True,
    ) == "reused-drds-result"
    assert calls == [("state", "ntx-bars", "support")]


def test_native_multi_rhs_equation_system_forwarding_hook_is_exposed_to_radau():
    """The real Radau vector-field owner forwards the native hook once."""

    calls = []

    class _SharedFlux:
        def pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_shared_primal(
            self,
            state,
            response_bars,
            support,
        ):
            calls.append((state, response_bars, support))
            return "native-result"

    equations = object.__new__(ComposedEquationSystem)
    object.__setattr__(equations, "shared_flux_model", _SharedFlux())
    object.__setattr__(
        equations,
        "_split_realtime_geometry_payload",
        lambda support: (support, None),
    )
    object.__setattr__(equations, "_prepare_working_state", lambda state: (state, None))
    object.__setattr__(equations, "_shared_flux_call_kwargs", lambda kwargs: {})
    response = SimpleNamespace(flux_response="flux-bars")

    assert equations.pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_shared_primal(
        "state",
        response,
        "support",
        ignored_outer_keyword=True,
    ) == "native-result"
    assert calls == [("state", "flux-bars", "support")]


def test_native_multi_rhs_reused_drds_equation_system_forwarding_hook_is_exposed_to_radau():
    """The Radau vector-field owner reaches the dedicated reuse hook."""

    calls = []

    class _SharedFlux:
        def pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal(
            self, state, response_bars, support,
        ):
            calls.append((state, response_bars, support))
            return "reused-drds-result"

    equations = object.__new__(ComposedEquationSystem)
    object.__setattr__(equations, "shared_flux_model", _SharedFlux())
    object.__setattr__(equations, "_split_realtime_geometry_payload", lambda support: (support, None))
    object.__setattr__(equations, "_prepare_working_state", lambda state: (state, None))
    object.__setattr__(equations, "_shared_flux_call_kwargs", lambda kwargs: {})
    response = SimpleNamespace(flux_response="flux-bars")
    assert equations.pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal(
        "state", response, "support", ignored_outer_keyword=True,
    ) == "reused-drds-result"
    assert calls == [("state", "flux-bars", "support")]


def test_direct_directional_vmec_composite_and_equation_hooks_are_exposed_to_radau():
    """Prevent an inner-only direct selector from failing after setup."""

    calls = []

    class _InnerNTX:
        def pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_directional_product_rule(
            self, state, response_bars, support,
        ):
            calls.append((state, response_bars, support))
            return "direct-result"

    composite = object.__new__(CombinedTransportFluxModel)
    object.__setattr__(composite, "neoclassical_model", _InnerNTX())
    combined_response = SimpleNamespace(neoclassical_response="ntx-bars")
    assert composite.pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_directional_product_rule(
        "state", combined_response, "support", ignored_outer_keyword=True,
    ) == "direct-result"

    equations = object.__new__(ComposedEquationSystem)
    object.__setattr__(equations, "shared_flux_model", composite)
    object.__setattr__(equations, "_split_realtime_geometry_payload", lambda support: (support, None))
    object.__setattr__(equations, "_prepare_working_state", lambda state: (state, None))
    object.__setattr__(equations, "_shared_flux_call_kwargs", lambda kwargs: {})
    equation_response = SimpleNamespace(flux_response=combined_response)
    assert equations.pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_directional_product_rule(
        "state", equation_response, "support", ignored_outer_keyword=True,
    ) == "direct-result"
    assert calls == [
        ("state", "ntx-bars", "support"),
        ("state", "ntx-bars", "support"),
    ]


def test_native_split_joint_no_prepared_carry_hook_selects_only_its_wrapper():
    """The compact initial mode cannot silently select the older joint hook."""

    calls = []

    class _Owner:
        def vector_field(self):
            return None

        def pullback_build_lagged_response_state_and_ntx_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_no_prepared_carry(
            self, state, bars, support, **kwargs,
        ):
            calls.append((state, bars, support, kwargs))
            return "compact-result"

    owner = _Owner()
    hook = _lagged_response_build_state_and_support_pullback_batched_interpolated_faces_hook(
        owner.vector_field,
        native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_no_prepared_carry=True,
    )
    assert callable(hook)
    assert hook("state", "bars", "support") == "compact-result"
    assert calls == [("state", "bars", "support", {})]


def test_fused_fixed_lagged_state_response_factory_calls_one_owner_hook():
    """The initial-only selector cannot silently revert to two RHS hooks."""

    calls = []

    class _Owner:
        def vector_field(self):
            return None

        def pullback_evaluate_with_lagged_response_state_and_response(
            self, t_value, state, *args, lagged_response, rhs_bar, **kwargs
        ):
            calls.append((t_value, state, lagged_response, rhs_bar, args, kwargs))
            return state * 2.0, {"lagged": rhs_bar * -3.0}

    owner = _Owner()
    pullback = _flat_rhs_state_and_lagged_response_pullback_factory(
        unravel=lambda value: value,
        pack_flat=lambda value: value,
        vector_field=owner.vector_field,
        args=(),
        kwargs={},
    )
    state_bar, lagged_bar = pullback(
        jnp.asarray(0.25),
        jnp.asarray([1.0, -2.0]),
        {"lagged": jnp.asarray([4.0, 5.0])},
        jnp.asarray([0.3, -0.7]),
    )
    assert jnp.allclose(state_bar, jnp.asarray([2.0, -4.0]))
    assert jnp.allclose(lagged_bar["lagged"], jnp.asarray([-0.9, 2.1]))
    assert len(calls) == 1


def test_native_multi_rhs_compact_residual_equation_system_forwarding_hook_is_exposed_to_radau():
    """The split-residual selector must be exposed by the vector-field owner."""

    calls = []

    class _SharedFlux:
        def pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_compact_residual_reuse_moment_drds_jvp_shared_primal(
            self, state, response_bars, support,
        ):
            calls.append((state, response_bars, support))
            return "compact-residual-result"

    equations = object.__new__(ComposedEquationSystem)
    object.__setattr__(equations, "shared_flux_model", _SharedFlux())
    object.__setattr__(equations, "_split_realtime_geometry_payload", lambda support: (support, None))
    object.__setattr__(equations, "_prepare_working_state", lambda state: (state, None))
    object.__setattr__(equations, "_shared_flux_call_kwargs", lambda kwargs: {})
    response = SimpleNamespace(flux_response="flux-bars")
    assert equations.pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_compact_residual_reuse_moment_drds_jvp_shared_primal(
        "state", response, "support", ignored_outer_keyword=True,
    ) == "compact-residual-result"
    assert calls == [("state", "flux-bars", "support")]


def test_native_multi_rhs_compact_residual_factory_finds_vector_field_owner_hook():
    """Catch a missing owner wrapper before a compiled reverse segment starts."""

    calls = []

    class _SharedFlux:
        def pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_compact_residual_reuse_moment_drds_jvp_shared_primal(
            self, state, response_bars, support,
        ):
            calls.append((state, response_bars, support))
            return "compact-residual-result"

    equations = object.__new__(ComposedEquationSystem)
    object.__setattr__(equations, "shared_flux_model", _SharedFlux())
    object.__setattr__(equations, "_split_realtime_geometry_payload", lambda support: (support, None))
    object.__setattr__(equations, "_prepare_working_state", lambda state: (state, None))
    object.__setattr__(equations, "_shared_flux_call_kwargs", lambda kwargs: {})
    pullback = _flat_rhs_build_support_pullback_batched_interpolated_faces_factory(
        unravel=lambda value: value,
        vector_field=equations.vector_field,
        args=(),
        kwargs={},
        native_multi_rhs_compact_residual_reuse_moment_drds_jvp_shared_primal=True,
    )
    assert callable(pullback)
    assert pullback("state", SimpleNamespace(flux_response="flux-bars"), "support") == "compact-residual-result"
    assert calls == [("state", "flux-bars", "support")]


def test_native_multi_rhs_compact_residual_physical_wrapper_forwards_its_flag():
    """The outer physical hook must reach the compact-residual local adapter."""

    calls = []

    class _PhysicalLike:
        def pullback_build_lagged_response_support_payload_batched_interpolated_faces_multi_rhs_shared_primal(
            self, state, response_bars, support, **kwargs,
        ):
            calls.append((state, response_bars, support, kwargs))
            return "compact-residual-result"

    method = NTXExactLijRuntimeTransportModel.pullback_build_lagged_response_support_payload_batched_interpolated_faces_multi_rhs_shared_primal
    assert "native_compact_residual_ntx_rhs" in inspect.signature(method).parameters
    wrapper = NTXExactLijRuntimeTransportModel.pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_compact_residual_reuse_moment_drds_jvp_shared_primal
    assert wrapper(_PhysicalLike(), "state", "response-bars", "support") == "compact-residual-result"
    assert calls == [(
        "state",
        "response-bars",
        "support",
        {
            "native_factorized_ntx_rhs": True,
            "native_compact_residual_ntx_rhs": True,
            "reuse_joint_moment_drds_jvp": True,
        },
    )]


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

    prepared = ntx.prepare_monoenergetic_system(
        ntx.example_surface(),
        ntx.GridSpec(5, 5, 4),
    )
    scalar_field_bars = (
        (
            jnp.asarray(0.13),
            jnp.asarray([0.3, -0.2, 0.1, 0.4, -0.1, 0.2]),
            jnp.asarray([-0.3, 0.1, 0.2, -0.2, 0.3, -0.1]),
            jnp.asarray([0.2, 0.4, -0.3, 0.1, 0.2, -0.4]),
        ),
        (
            jnp.asarray(-0.31),
            jnp.asarray([-0.1, 0.2, 0.3, -0.4, 0.5, -0.2]),
            jnp.asarray([0.4, -0.3, 0.2, 0.1, -0.5, 0.3]),
            jnp.asarray([-0.2, 0.3, 0.4, -0.1, 0.2, 0.5]),
        ),
        (
            jnp.asarray(0.21),
            jnp.asarray([0.2, -0.4, 0.5, -0.3, 0.1, 0.3]),
            jnp.asarray([-0.5, 0.2, -0.1, 0.3, 0.4, -0.2]),
            jnp.asarray([0.1, -0.3, 0.2, 0.5, -0.4, 0.3]),
        ),
    )
    batched_field_bars = tuple(
        jnp.stack([field_bars[field_index] for field_bars in scalar_field_bars])
        for field_index in range(4)
    )
    # The transport benchmark uses an energy scan, whereas the original
    # gate covered only one energy.  Exercise both shapes: an error in the
    # energy reduction can leave profile bars apparently correct while
    # corrupting a geometry-only prepared leaf.
    for n_energy in (1, 2):
        model = _small_runtime_model(n_energy=n_energy)
        args = dict(
            drds_value=jnp.asarray(1.2),
            reference_nu_hat=jnp.linspace(1.0e-2, 1.8e-2, n_energy),
            reference_epsi_hat=jnp.linspace(1.0e-3, 2.0e-3, n_energy),
            vth_a=jnp.linspace(1.1, 1.2, n_energy),
        )
        actual_prepared, actual_drds, actual_primal = (
            model._pullback_interpolated_moment_prepared_support_and_drds_only_multi_rhs(
                prepared,
                field_bars=batched_field_bars,
                **args,
            )
        )
        native_prepared, native_drds, native_primal = (
            model._pullback_interpolated_moment_prepared_support_and_drds_only_native_multi_rhs(
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
            _assert_float_tree_allclose(
                jax.tree_util.tree_map(lambda value: value[rhs_index], native_prepared),
                expected_prepared,
            )
            _assert_float_tree_allclose(native_drds[rhs_index], expected_drds)
            _assert_float_tree_allclose(native_primal, expected_primal)


def test_compact_and_reused_drds_native_multi_rhs_adapters_match_full_native_adapter():
    """Opt-in native payload and drds reductions preserve local cotangents."""

    model = _small_runtime_model(n_energy=2)
    prepared = ntx.prepare_monoenergetic_system(
        ntx.example_surface(), ntx.GridSpec(5, 5, 4)
    )
    field_bars = (
        jnp.asarray([0.13, -0.31, 0.21]),
        jnp.asarray([[0.3, -0.2, 0.1, 0.4, -0.1, 0.2],
                     [-0.1, 0.2, 0.3, -0.4, 0.5, -0.2],
                     [0.2, -0.4, 0.5, -0.3, 0.1, 0.3]]),
        jnp.asarray([[-0.3, 0.1, 0.2, -0.2, 0.3, -0.1],
                     [0.4, -0.3, 0.2, 0.1, -0.5, 0.3],
                     [-0.5, 0.2, -0.1, 0.3, 0.4, -0.2]]),
        jnp.asarray([[0.2, 0.4, -0.3, 0.1, 0.2, -0.4],
                     [-0.2, 0.3, 0.4, -0.1, 0.2, 0.5],
                     [0.1, -0.3, 0.2, 0.5, -0.4, 0.3]]),
    )
    args = dict(
        drds_value=jnp.asarray(1.2),
        reference_nu_hat=jnp.asarray([1.0e-2, 1.8e-2]),
        reference_epsi_hat=jnp.asarray([1.0e-3, 2.0e-3]),
        # Production supplies one species thermal speed; the local energy
        # scan must broadcast this scalar before attaching the RHS axis.
        vth_a=jnp.asarray(1.1),
        field_bars=field_bars,
        return_case_bars=True,
    )
    full = model._pullback_interpolated_moment_prepared_support_and_drds_only_native_multi_rhs(
        prepared, **args
    )
    compact = model._pullback_interpolated_moment_prepared_support_and_drds_only_native_multi_rhs_compact(
        prepared, **args
    )
    reused_drds = model._pullback_interpolated_moment_prepared_support_and_drds_only_native_multi_rhs_reuse_moment_drds_jvp(
        prepared, **args
    )
    compact_residual = model._pullback_interpolated_moment_prepared_support_and_drds_only_native_multi_rhs_compact_residual_reuse_moment_drds_jvp(
        prepared, **args
    )
    for candidate in (compact, reused_drds, compact_residual):
        _assert_float_tree_allclose(candidate[0], full[0])
        _assert_float_tree_allclose(candidate[1], full[1])
        _assert_float_tree_allclose(candidate[2], full[2])
        _assert_float_tree_allclose(candidate[3], full[3])


@pytest.mark.parametrize(
    ("field_index", "component_name"),
    (
        (1, "transport_moments"),
        (2, "dtransport_moments_d_er"),
        (3, "dtransport_moments_d_log_nu_star"),
    ),
)
def test_native_vmec_face_rebuild_component_oracle_matches_prepared_vjp(
    field_index,
    component_name,
):
    """Compare each actual NEOPAX low-dot component at the replacement boundary.

    This is the missing integration-level oracle: it retains the production
    local multi-energy/moment assembly but stops before interpolation, a Radau
    step, a VMEC solve, or any filesystem output.  It asks whether the native
    coefficient return is exactly the `face_prepared -> VmecSurface` VJP for
    one response component.
    """

    model = _small_runtime_model(n_energy=2)
    surface, prepared = _small_vmec_prepared()
    all_field_bars = (
        jnp.asarray([0.0, 0.0]),
        jnp.asarray(
            [[0.3, -0.2, 0.1, 0.4, -0.1, 0.2],
             [-0.1, 0.2, 0.3, -0.4, 0.5, -0.2]]
        ),
        jnp.asarray(
            [[-0.3, 0.1, 0.2, -0.2, 0.3, -0.1],
             [0.4, -0.3, 0.2, 0.1, -0.5, 0.3]]
        ),
        jnp.asarray(
            [[0.2, 0.4, -0.3, 0.1, 0.2, -0.4],
             [-0.2, 0.3, 0.4, -0.1, 0.2, 0.5]]
        ),
    )
    field_bars = tuple(
        value if index == field_index else jnp.zeros_like(value)
        for index, value in enumerate(all_field_bars)
    )
    args = dict(
        drds_value=jnp.asarray(1.2),
        reference_nu_hat=jnp.asarray([1.0e-2, 1.8e-2]),
        reference_epsi_hat=jnp.asarray([1.0e-3, 2.0e-3]),
        vth_a=jnp.asarray(1.1),
        field_bars=field_bars,
        return_case_bars=True,
    )
    prepared_bar, _drds_bar, _primal, _case_bars, native_bars = (
        model._pullback_interpolated_moment_prepared_support_and_drds_only_native_multi_rhs_reuse_moment_drds_jvp_with_vmec_coefficients(
            prepared,
            native_vmec_coefficient_bars_only=False,
            **args,
        )
    )
    _, surface_pullback = jax.vjp(
        lambda surface_value: ntx.prepare_monoenergetic_system(
            surface_value, prepared.grid
        ),
        surface,
    )
    names = (
        "b_cos",
        "jacobian_cos",
        "b_sub_theta_cos",
        "b_sub_zeta_cos",
        "b_sup_theta_cos",
        "b_sup_zeta_cos",
        "b0",
    )
    expected = tuple(
        surface_pullback(
            jax.tree_util.tree_map(lambda value: value[rhs_index], prepared_bar)
        )[0]
        for rhs_index in range(int(jnp.asarray(field_bars[0]).shape[0]))
    )
    for name in names:
        expected_value = jnp.stack([getattr(value, name) for value in expected])
        actual_value = jnp.asarray(native_bars[name])
        if not jnp.allclose(actual_value, expected_value, rtol=1e-10, atol=1e-12):
            difference = jnp.abs(actual_value - expected_value)
            scale = jnp.maximum(jnp.abs(expected_value), 1.0e-30)
            raise AssertionError(
                f"{component_name} {name}: "
                f"max_abs={float(jnp.max(difference)):.16e} "
                f"max_rel={float(jnp.max(difference / scale)):.16e}"
            )
    # The native bridge exposes precisely the seven coefficient fields above.
    # Prove that every other differentiable VmecSurface field has no local
    # prepared-system cotangent in the explicit-epsi_hat transport contract;
    # otherwise this bridge would silently omit a state-dependent channel.
    for name in (
        "requested_psi_n",
        "psi_n",
        "nfp",
        "iota",
        "psi_a_hat",
        "phi_edge",
        "r_n",
        "r_hat",
        "dpsi_hat_dr_hat",
        "dr_hat_dpsi_hat",
        "aminor_p",
        "psi_p",
        "transport_psi_scale",
    ):
        values = tuple(getattr(value, name) for value in expected)
        if values[0] is None:
            continue
        expected_value = jnp.stack([jnp.asarray(value) for value in values])
        if jnp.issubdtype(expected_value.dtype, jnp.inexact):
            assert jnp.allclose(expected_value, 0.0, rtol=0.0, atol=1e-12), (
                component_name,
                name,
                expected_value,
            )


@pytest.mark.parametrize(
    ("field_index", "component_name"),
    (
        (1, "transport_moments"),
        (2, "dtransport_moments_d_er"),
        (3, "dtransport_moments_d_log_nu_star"),
    ),
)
def test_native_vmec_face_rebuild_er_drds_chain_matches_local_response_vjp(
    field_index,
    component_name,
):
    """Exercise ``Er * drds -> epsi_hat`` through the native local reverse.

    Unlike the coefficient-only oracle above, this starts with the physical
    local inputs and includes the return-case-bars chain back through
    ``_pullback_local_scan_inputs_and_drds_from_primitives``.  It remains a
    tiny in-memory two-energy check and does not construct a transport state
    or solve VMEC.
    """

    model = _small_runtime_model(n_energy=2)
    # The actual code reads this only if a velocity floor is configured.
    object.__setattr__(model, "er_v_floor", None)
    # ``_nu_over_vnew_local`` is jitted, so the fixture must use NEOPAX's
    # registered Species pytree rather than a host-only SimpleNamespace.
    object.__setattr__(
        model,
        "species",
        Species(
            # Although this gate evaluates ion index zero only, lax.cond
            # traces the Zeff collisionality branch too.  That branch needs
            # a physical electron lane, so retain the smallest valid pair.
            number_species=2,
            species_indices=jnp.asarray([0, 1]),
            mass_mp=jnp.asarray([2.0, 5.446e-4]),
            charge_qp=jnp.asarray([1.0, -1.0]),
            names=("ion", "e"),
        ),
    )
    _surface, prepared = _small_vmec_prepared()
    drds_value = jnp.asarray(1.2)
    er_value = jnp.asarray(-2.7e-3)
    temperature_local = jnp.asarray([1.4, 1.1])
    density_local = jnp.asarray([0.9, 0.9])
    # Match the production helper exactly.  In particular, temperatures are
    # stored in keV and Species.mass is in SI units.
    vthermal_local = get_v_thermal(model.species.mass, temperature_local)
    collisionality_kind = _collisionality_kind("default")
    all_field_bars = (
        jnp.asarray([0.0, 0.0]),
        jnp.asarray(
            [[0.3, -0.2, 0.1, 0.4, -0.1, 0.2],
             [-0.1, 0.2, 0.3, -0.4, 0.5, -0.2]]
        ),
        jnp.asarray(
            [[-0.3, 0.1, 0.2, -0.2, 0.3, -0.1],
             [0.4, -0.3, 0.2, 0.1, -0.5, 0.3]]
        ),
        jnp.asarray(
            [[0.2, 0.4, -0.3, 0.1, 0.2, -0.4],
             [-0.2, 0.3, 0.4, -0.1, 0.2, 0.5]]
        ),
    )
    field_bars = tuple(
        value if index == field_index else jnp.zeros_like(value)
        for index, value in enumerate(all_field_bars)
    )
    nu_hat, epsi_hat, vth_a = model._interpolated_moment_local_scan_primitives(
        drds_value=drds_value,
        species_index=0,
        er_value=er_value,
        temperature_local=temperature_local,
        density_local=density_local,
        vthermal_local=vthermal_local,
        collisionality_kind=collisionality_kind,
    )
    (
        native_prepared,
        native_direct_drds,
        _native_primal,
        (native_nu_bar, native_epsi_bar, native_vth_a_bar),
    ) = model._pullback_interpolated_moment_prepared_support_and_drds_only_native_multi_rhs_reuse_moment_drds_jvp(
        prepared,
        drds_value=drds_value,
        reference_nu_hat=nu_hat,
        reference_epsi_hat=epsi_hat,
        vth_a=vth_a,
        field_bars=field_bars,
        return_case_bars=True,
    )
    (
        native_primitive_drds,
        native_er,
        _native_temperature,
        _native_density,
    ) = jax.vmap(
        lambda nu_bar, epsi_bar, vth_bar: model._pullback_local_scan_inputs_and_drds_from_primitives(
            drds_value=drds_value,
            species_index=0,
            er_value=er_value,
            temperature_local=temperature_local,
            density_local=density_local,
            collisionality_kind=collisionality_kind,
            reference_nu_hat_bar=nu_bar,
            reference_epsi_hat_bar=epsi_bar,
            vth_a_bar=vth_bar,
        )
    )(native_nu_bar, native_epsi_bar, native_vth_a_bar)

    def _local_response(prepared_value, drds_local, er_local):
        return model._build_interpolated_moment_response_local(
            prepared_value,
            drds_value=drds_local,
            species_index=0,
            er_value=er_local,
            temperature_local=temperature_local,
            density_local=density_local,
            vthermal_local=vthermal_local,
            collisionality_kind=collisionality_kind,
        )

    _value, pullback = jax.vjp(_local_response, prepared, drds_value, er_value)
    for rhs_index in range(int(jnp.asarray(field_bars[0]).shape[0])):
        _expected_prepared, expected_drds, expected_er = pullback(
            tuple(value[rhs_index] for value in field_bars)
        )
        # The fixed-epsi component oracle already establishes the prepared
        # surface path.  This gate isolates the previously omitted physical
        # ``Er * drds -> epsi_hat`` case chain; some unrelated prepared leaves
        # carry NaNs for this deliberately tiny collision fixture.
        assert jnp.allclose(
            native_direct_drds[rhs_index] + native_primitive_drds[rhs_index],
            expected_drds,
            rtol=1e-10,
            atol=1e-12,
        ), component_name
        assert jnp.allclose(
            native_er[rhs_index], expected_er, rtol=1e-10, atol=1e-12
        ), component_name


def test_native_joint_local_multi_rhs_matches_explicit_state_support_vjp():
    """Native joint local bars equal explicit state/prepared VJPs per RHS.

    This is the numerical gate for the unselected initial-carry adapter.  It
    deliberately stops at one prepared system: there is no support build,
    interpolation, Radau step, VMEC solve, profile, or filesystem output.
    The Python loops are oracle-only; the implementation under test receives
    the complete on-device RHS batch at once.
    """

    model = _small_runtime_model(n_energy=2)
    object.__setattr__(model, "er_v_floor", None)
    object.__setattr__(
        model,
        "species",
        Species(
            number_species=2,
            species_indices=jnp.asarray([0, 1]),
            mass_mp=jnp.asarray([2.0, 5.446e-4]),
            charge_qp=jnp.asarray([1.0, -1.0]),
            names=("ion", "e"),
        ),
    )
    _surface, prepared = _small_vmec_prepared()
    drds_value = jnp.asarray(1.2)
    er_value = jnp.asarray(-2.7e-3)
    temperature_local = jnp.asarray([1.4, 1.1])
    density_local = jnp.asarray([0.9, 0.9])
    collisionality_kind = _collisionality_kind("default")
    rhs_fields = (
        jnp.asarray([0.17, -0.23]),
        jnp.asarray(
            [[0.3, -0.2, 0.1, 0.4, -0.1, 0.2],
             [-0.1, 0.2, 0.3, -0.4, 0.5, -0.2]]
        ),
        jnp.asarray(
            [[-0.3, 0.1, 0.2, -0.2, 0.3, -0.1],
             [0.4, -0.3, 0.2, 0.1, -0.5, 0.3]]
        ),
        jnp.asarray(
            [[0.2, 0.4, -0.3, 0.1, 0.2, -0.4],
             [-0.2, 0.3, 0.4, -0.1, 0.2, 0.5]]
        ),
    )
    # Species is the leading axis expected by the local joint helper; RHS is
    # next, and remains the matrix dimension passed to NTX.
    field_bars = tuple(
        jnp.stack((values, -0.7 * values), axis=0) for values in rhs_fields
    )
    actual = (
        model._pullback_interpolated_moment_response_local_fields_and_prepared_support_and_drds_flat_prepared(
            prepared,
            drds_value=drds_value,
            er_value=er_value,
            temperature_local=temperature_local,
            density_local=density_local,
            collisionality_kind=collisionality_kind,
            field_bars=field_bars,
            native_factorized_ntx_rhs=True,
            reuse_joint_moment_drds_jvp=True,
        )
    )

    def _add_trees(*trees):
        def _add_leaves(*leaves):
            first = jnp.asarray(leaves[0])
            if first.dtype == jax.dtypes.float0:
                return first
            return sum(leaves[1:], first)

        return jax.tree_util.tree_map(_add_leaves, *trees)

    rhs_count = int(field_bars[0].shape[1])
    expected_rows = []
    for rhs_index in range(rhs_count):
        species_rows = []
        for species_index in range(int(model.species.number_species)):
            def _local_response(
                prepared_value,
                drds_local,
                er_local,
                temperature_value,
                density_value,
            ):
                return model._build_interpolated_moment_response_local(
                    prepared_value,
                    drds_value=drds_local,
                    species_index=species_index,
                    er_value=er_local,
                    temperature_local=temperature_value,
                    density_local=density_value,
                    vthermal_local=get_v_thermal(model.species.mass, temperature_value),
                    collisionality_kind=collisionality_kind,
                )

            _, pullback = jax.vjp(
                _local_response,
                prepared,
                drds_value,
                er_value,
                temperature_local,
                density_local,
            )
            species_rows.append(
                pullback(tuple(field[species_index, rhs_index] for field in field_bars))
            )
        expected_rows.append(_add_trees(*species_rows))

    def _stack_rhs_leaves(*leaves):
        first = jnp.asarray(leaves[0])
        if first.dtype == jax.dtypes.float0:
            return first
        return jnp.stack(leaves)

    expected_prepared = jax.tree_util.tree_map(
        _stack_rhs_leaves, *(row[0] for row in expected_rows)
    )
    expected_state = tuple(
        jnp.stack([row[index] for row in expected_rows])
        for index in range(1, 5)
    )
    _assert_float_tree_allclose(actual[:4], expected_state, rtol=1e-9, atol=1e-11)
    _assert_float_tree_allclose(
        actual[4], tuple(jax.tree_util.tree_leaves(expected_prepared)),
        rtol=1e-9,
        atol=1e-11,
    )


def test_native_joint_local_vmec_contract_omits_only_zero_prepared_carry():
    """The compact joint local contract retains all non-prepared channels.

    This is deliberately a single local NTX system.  It proves the new carry
    reduction changes only the generic prepared return, which the native VMEC
    coefficient channel replaces, without starting a support build or a
    transport rollout.
    """

    model = _small_runtime_model(n_energy=2)
    object.__setattr__(model, "er_v_floor", None)
    object.__setattr__(
        model,
        "species",
        Species(
            number_species=2,
            species_indices=jnp.asarray([0, 1]),
            mass_mp=jnp.asarray([2.0, 5.446e-4]),
            charge_qp=jnp.asarray([1.0, -1.0]),
            names=("ion", "e"),
        ),
    )
    _surface, prepared = _small_vmec_prepared()
    field_values = (
        jnp.asarray([0.17, -0.23]),
        jnp.asarray([[0.3, -0.2, 0.1, 0.4, -0.1, 0.2], [-0.1, 0.2, 0.3, -0.4, 0.5, -0.2]]),
        jnp.asarray([[-0.3, 0.1, 0.2, -0.2, 0.3, -0.1], [0.4, -0.3, 0.2, 0.1, -0.5, 0.3]]),
        jnp.asarray([[0.2, 0.4, -0.3, 0.1, 0.2, -0.4], [-0.2, 0.3, 0.4, -0.1, 0.2, 0.5]]),
    )
    field_bars = tuple(jnp.stack((values, -0.7 * values), axis=0) for values in field_values)
    kwargs = dict(
        drds_value=jnp.asarray(1.2),
        er_value=jnp.asarray(-2.7e-3),
        temperature_local=jnp.asarray([1.4, 1.1]),
        density_local=jnp.asarray([0.9, 0.9]),
        collisionality_kind=_collisionality_kind("default"),
        field_bars=field_bars,
        native_factorized_ntx_rhs=True,
        reuse_joint_moment_drds_jvp=True,
        return_native_vmec_coefficient_bars=True,
    )
    full = model._pullback_interpolated_moment_response_local_fields_and_prepared_support_and_drds_flat_prepared(
        prepared, **kwargs
    )
    compact = model._pullback_interpolated_moment_response_local_fields_and_prepared_support_and_drds_flat_prepared(
        prepared, omit_generic_prepared_carry=True, **kwargs
    )
    _assert_float_tree_allclose(compact[:4], full[:4])
    assert compact[4] == ()
    _assert_float_tree_allclose(compact[5], full[5])


def test_native_vmec_face_rebuild_accumulation_matches_generic_prepared_vjp():
    """Compare the real native replacement boundary without a transport rollout.

    The earlier component oracle proves one local ``face_prepared`` system.
    This gate adds the production face-anchor interpolation, species sum, and
    objective/RHS batching, then checks that converting the generic stacked
    prepared cotangent to VMEC surface coefficients equals the native stacked
    coefficient return.  It deliberately has no VMEC solve or raw-block
    pullback: the two sides receive identical fixed traceable surfaces.
    """

    surface, _prepared = _small_vmec_prepared()
    grid = ntx.GridSpec(5, 5, 4)
    geometry = _TestMomentumGeometry(
        a_b=jnp.asarray(1.0),
        r_grid=jnp.asarray([0.3, 0.7]),
        r_grid_half=jnp.asarray([0.1, 0.5, 0.9]),
        Bsqav=jnp.asarray([1.2, 1.3]),
    )

    def _channels(rho):
        rho = jnp.asarray(rho, dtype=jnp.float64)
        ones = jnp.ones_like(rho)
        return NTXRuntimeScanChannels.from_mapping(
            rho,
            {
                "a_b": 1.0, "psia": 1.0, "b00": ones, "r00": ones,
                "boozer_i": ones, "boozer_g": ones, "iota": ones,
                "drds": ones, "dr_tildedr": ones, "dr_tildeds": ones,
                "fac_reference_to_sfincs_11": ones,
                "fac_reference_to_sfincs_31": ones,
                "fac_reference_to_sfincs_33": ones,
                "fac_sfincs_to_dkes_11": ones,
                "fac_sfincs_to_dkes_31": ones,
                "fac_sfincs_to_dkes_33": ones,
                "fac_dkes_to_d11star": ones,
                "fac_dkes_to_d31star": ones,
                "fac_dkes_to_d33star": ones,
            },
        )

    # Unlike the earlier carrier-only version of this gate, each face must
    # have a distinct prepared VMEC surface.  The production support builder
    # does exactly that, and a re-derived native coefficient rule could agree
    # on repeated copies while failing once Fourier data vary with radius.
    face_surfaces = tuple(
        dataclasses.replace(
            surface,
            b_cos=surface.b_cos + jnp.asarray([0.03 * face, -0.01 * face]),
            jacobian_cos=surface.jacobian_cos + jnp.asarray([0.02 * face, 0.004 * face]),
            b_sub_theta_cos=surface.b_sub_theta_cos + jnp.asarray([0.01 * face, -0.003 * face]),
            b_sub_zeta_cos=surface.b_sub_zeta_cos + jnp.asarray([-0.02 * face, 0.005 * face]),
            b_sup_theta_cos=surface.b_sup_theta_cos + jnp.asarray([0.015 * face, 0.002 * face]),
            b_sup_zeta_cos=surface.b_sup_zeta_cos + jnp.asarray([-0.01 * face, 0.003 * face]),
            b0=surface.b0 + jnp.asarray(0.02 * face),
        )
        for face in range(3)
    )
    face_prepared_values = tuple(
        ntx.prepare_monoenergetic_system(face_surface, grid)
        for face_surface in face_surfaces
    )

    def _stack_prepared(values):
        return jax.tree_util.tree_map(
            lambda *leaves: (
                None if leaves[0] is None
                else jnp.stack([jnp.asarray(leaf) for leaf in leaves], axis=0)
            ),
            *values,
        )

    support = NTXExactLijRuntimeSupport(
        center_channels=_channels(geometry.r_grid),
        face_channels=_channels(geometry.r_grid_half),
        center_prepared=_stack_prepared(face_prepared_values[:2]),
        face_prepared=_stack_prepared(face_prepared_values),
        grid=grid,
    )
    species = Species(
        # The hard-coded three-Sonine momentum correction is square only
        # for three kinetic species; this mirrors the production contract.
        number_species=3,
        species_indices=jnp.asarray([0, 1, 2]),
        mass_mp=jnp.asarray([5.446e-4, 2.0, 3.0]),
        charge_qp=jnp.asarray([-1.0, 1.0, 2.0]),
        names=("e", "D", "He"),
    )
    # ``get_Matrix`` is JIT compiled, so this needs the same JAX-pytree
    # energy-grid object used by production rather than a SimpleNamespace.
    energy_grid = StandardLaguerreEnergyGrid(n_x=2)
    model = NTXExactLijRuntimeTransportModel(
        species=species, energy_grid=energy_grid, geometry=geometry,
        vmec_file=None, boozer_file=None, support=support,
        center_response_mode="interpolate_from_faces", response_anchor_count=2,
    )
    state = TransportState(
        density=jnp.asarray([[1.0, 1.15], [1.0, 1.15], [0.8, 0.9]]),
        pressure=jnp.asarray([[1.3, 1.61], [1.1, 1.38], [0.7, 0.87]]),
        Er=jnp.asarray([2.0e-4, 2.5e-4]),
    )
    # The bootstrap-only primal may omit Gamma/Q/qpar/Upar2, but must retain
    # exactly the regularized Upar consumed by the production objective.
    _assert_float_tree_allclose(
        model.evaluate_momentum_corrected_upar_only(state),
        model.evaluate_momentum_corrected_fluxes(state)["Upar"],
        rtol=1e-10,
        atol=1e-12,
    )
    response_bars = jax.tree_util.tree_map(
        lambda value: jnp.stack(
            (jnp.full_like(jnp.asarray(value), 0.17),
             jnp.full_like(jnp.asarray(value), -0.23)),
            axis=0,
        ),
        model.build_lagged_response(state),
    )
    generic = model.pullback_build_lagged_response_support_payload_batched_interpolated_faces_multi_rhs_shared_primal(
        state, response_bars, support,
        native_factorized_ntx_rhs=True,
        reuse_joint_moment_drds_jvp=True,
    )
    native_support, native_bars = (
        model.pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients(
            state, response_bars, support,
        )
    )
    _assert_float_tree_allclose(
        native_support.face_channels, generic.face_channels,
        rtol=1e-10, atol=1e-12,
    )
    # This selector deliberately carries the face-prepared contribution only
    # through ``native_bars``.  A nonzero generic prepared payload here would
    # double-count it when the later VMEC-state bridge is applied.
    _assert_float_tree_allclose(
        native_support.face_prepared,
        jax.tree_util.tree_map(jnp.zeros_like, native_support.face_prepared),
        rtol=0.0,
        atol=0.0,
    )
    names = (
        "b_cos", "jacobian_cos", "b_sub_theta_cos", "b_sub_zeta_cos",
        "b_sup_theta_cos", "b_sup_zeta_cos", "b0",
    )
    expected_by_name = {name: [] for name in names}
    for rhs_index in range(2):
        expected_by_face = {name: [] for name in names}
        for face_index in range(3):
            _, surface_pullback = jax.vjp(
                lambda surface_value: ntx.prepare_monoenergetic_system(
                    surface_value, grid
                ),
                face_surfaces[face_index],
            )
            prepared_bar = jax.tree_util.tree_map(
                lambda value: jnp.asarray(value)[rhs_index, face_index],
                generic.face_prepared,
            )
            surface_bar = surface_pullback(prepared_bar)[0]
            for name in names:
                expected_by_face[name].append(jnp.asarray(getattr(surface_bar, name)))
        for name in names:
            expected_by_name[name].append(jnp.stack(expected_by_face[name], axis=0))
    for name in names:
        assert jnp.allclose(
            native_bars[name], jnp.stack(expected_by_name[name], axis=0),
            rtol=1e-10, atol=1e-12,
        ), name

    # This is the exact post-NTX replacement used by the geometry bridge.
    # The generic route contracts ``face_prepared`` bars with a JVP of
    # ``prepare_monoenergetic_system(face_surface(state))``.  The native route
    # contracts the parallel coefficient bars with the JVP of just those same
    # face coefficients.  Use one common, two-direction state tangent and
    # require row-by-row equality for the complete two-RHS accumulation.
    def _surfaces_from_mock_state(mock_state):
        first, second = mock_state
        return tuple(
            dataclasses.replace(
                face_surface,
                b_cos=face_surface.b_cos + first * jnp.asarray([0.11, -0.07]),
                jacobian_cos=face_surface.jacobian_cos + second * jnp.asarray([0.05, 0.03]),
                b_sub_theta_cos=face_surface.b_sub_theta_cos + first * jnp.asarray([-0.02, 0.04]),
                b_sub_zeta_cos=face_surface.b_sub_zeta_cos + second * jnp.asarray([0.06, -0.01]),
                b_sup_theta_cos=face_surface.b_sup_theta_cos + first * jnp.asarray([0.03, 0.02]),
                b_sup_zeta_cos=face_surface.b_sup_zeta_cos + second * jnp.asarray([-0.04, 0.05]),
                b0=face_surface.b0 + first * 0.09 - second * 0.06,
            )
            for face_surface in face_surfaces
        )

    def _prepared_from_mock_state(mock_state):
        return _stack_prepared(
            tuple(
                ntx.prepare_monoenergetic_system(face_surface, grid)
                for face_surface in _surfaces_from_mock_state(mock_state)
            )
        )

    def _coefficient_tuple_from_mock_state(mock_state):
        surfaces = _surfaces_from_mock_state(mock_state)
        return tuple(
            jnp.stack([jnp.asarray(getattr(face_surface, name)) for face_surface in surfaces])
            for name in names
        )

    mock_state = (jnp.asarray(0.0), jnp.asarray(0.0))
    mock_tangent = (jnp.asarray(0.37), jnp.asarray(-0.29))
    _, prepared_tangent = jax.jvp(
        _prepared_from_mock_state,
        (mock_state,),
        (mock_tangent,),
    )
    prepared_bar_leaves = jax.tree_util.tree_leaves(generic.face_prepared)
    prepared_tangent_leaves = jax.tree_util.tree_leaves(prepared_tangent)
    generic_jvp_contraction = []
    for rhs_index in range(2):
        total = jnp.asarray(0.0)
        for bar_leaf, tangent_leaf in zip(
            prepared_bar_leaves, prepared_tangent_leaves, strict=True
        ):
            tangent_array = jnp.asarray(tangent_leaf)
            if jnp.issubdtype(tangent_array.dtype, jnp.inexact):
                total = total + jnp.vdot(jnp.asarray(bar_leaf)[rhs_index], tangent_array)
        generic_jvp_contraction.append(total)
    native_jvp_contraction = _native_vmec_coefficient_tangent_contraction(
        tuple(jnp.asarray(native_bars[name]) for name in names),
        jax.jvp(
            _coefficient_tuple_from_mock_state,
            (mock_state,),
            (mock_tangent,),
        )[1],
    )
    assert jnp.allclose(
        native_jvp_contraction,
        jnp.stack(generic_jvp_contraction),
        rtol=1e-10,
        atol=1e-12,
    )


def test_native_multi_rhs_support_retains_local_drds_case_chain():
    """The native support rule must include ``drds -> epsi_hat`` exactly.

    The former native selector only compared against a support-only NTX
    oracle, which deliberately holds the local case fixed.  This miniature
    response instead makes ``epsi_hat`` depend on ``drds`` and compares the
    corrected native case bars plus their primitive transpose to the actual
    local response VJP.  It is in-memory only and has no transport rollout.
    """

    model = _small_runtime_model(n_energy=2)
    prepared = ntx.prepare_monoenergetic_system(
        ntx.example_surface(),
        ntx.GridSpec(5, 5, 4),
    )
    epsi_per_drds = jnp.asarray([1.0e-3, 1.6e-3])

    def _mock_local_primitives(*, drds_value, **_unused):
        return (
            jnp.asarray([1.0e-2, 1.8e-2]),
            epsi_per_drds * drds_value,
            jnp.asarray(1.1),
        )

    object.__setattr__(model, "_interpolated_moment_local_scan_primitives", _mock_local_primitives)
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
        jnp.stack([bars[field_index] for bars in scalar_field_bars])
        for field_index in range(4)
    )
    drds = jnp.asarray(1.2)
    reference_nu_hat, reference_epsi_hat, vth_a = _mock_local_primitives(
        drds_value=drds,
    )
    (
        native_prepared,
        native_direct_drds,
        _native_primal,
        (_native_nu_hat, native_epsi_hat, _native_vth_a),
    ) = model._pullback_interpolated_moment_prepared_support_and_drds_only_native_multi_rhs(
        prepared,
        drds_value=drds,
        reference_nu_hat=reference_nu_hat,
        reference_epsi_hat=reference_epsi_hat,
        vth_a=vth_a,
        field_bars=batched_field_bars,
        return_case_bars=True,
    )
    native_drds = native_direct_drds + jnp.sum(native_epsi_hat * epsi_per_drds, axis=1)

    def _response(prepared_value, drds_value):
        return model._build_interpolated_moment_response_local(
            prepared_value,
            drds_value=drds_value,
            species_index=0,
            er_value=jnp.asarray(0.0),
            temperature_local=jnp.asarray([1.0]),
            density_local=jnp.asarray([1.0]),
            vthermal_local=jnp.asarray([1.1]),
            collisionality_kind="unused",
        )

    # The complete RHS below is linear.  Gate its three NTX-dependent output
    # channels independently so a failure identifies the low-dot term that
    # needs correction, rather than merely reporting a summed prepared bar.
    component_names = (
        "transport_moments",
        "dtransport_moments_d_er",
        "dtransport_moments_d_log_nu_star",
    )
    reference_field_bars = scalar_field_bars[0]
    for field_index, component_name in zip((1, 2, 3), component_names, strict=True):
        component_field_bars = tuple(
            value if index == field_index else jnp.zeros_like(value)
            for index, value in enumerate(reference_field_bars)
        )
        component_batch = tuple(value[None, ...] for value in component_field_bars)
        (
            generic_component_prepared,
            _generic_component_direct_drds,
            _generic_component_primal,
        ) = model._pullback_interpolated_moment_prepared_support_and_drds_only_multi_rhs(
            prepared,
            drds_value=drds,
            reference_nu_hat=reference_nu_hat,
                reference_epsi_hat=reference_epsi_hat,
                vth_a=vth_a,
                field_bars=component_batch,
                include_second_direction_base_prepared=False,
            )
        (
            native_component_prepared,
            _native_component_direct_drds,
            _native_component_primal,
            _native_component_case_bars,
        ) = model._pullback_interpolated_moment_prepared_support_and_drds_only_native_multi_rhs(
            prepared,
            drds_value=drds,
            reference_nu_hat=reference_nu_hat,
            reference_epsi_hat=reference_epsi_hat,
            vth_a=vth_a,
            field_bars=component_batch,
            return_case_bars=True,
        )
        _response_value, component_pullback = jax.vjp(_response, prepared, drds)
        expected_component_prepared, _expected_component_drds = component_pullback(
            component_field_bars
        )
        try:
            _assert_float_tree_allclose(
                jax.tree_util.tree_map(
                    lambda value: value[0], generic_component_prepared
                ),
                expected_component_prepared,
                rtol=5e-9,
            )
        except AssertionError as error:
            raise AssertionError(
                "Generic corrected low-dot mismatch in "
                f"{component_name}; the remaining issue is shared algebra, "
                "not native matrix-RHS packing."
            ) from error
        try:
            _assert_float_tree_allclose(
                jax.tree_util.tree_map(
                    lambda value: value[0], native_component_prepared
                ),
                expected_component_prepared,
                rtol=5e-9,
            )
        except AssertionError as error:
            raise AssertionError(
                "Native matrix-RHS prepared-support low-dot mismatch in "
                f"{component_name}."
            ) from error
        native_component_epsi_hat = _native_component_case_bars[1]
        native_component_drds = (
            _native_component_direct_drds
            + jnp.sum(native_component_epsi_hat * epsi_per_drds, axis=1)
        )
        try:
            _assert_float_tree_allclose(
                native_component_drds[0],
                _expected_component_drds,
                rtol=5e-9,
            )
        except AssertionError as error:
            raise AssertionError(
                "Native matrix-RHS drds support-chain mismatch in "
                f"{component_name}."
            ) from error

    for rhs_index, field_bars in enumerate(scalar_field_bars):
        _response_value, pullback = jax.vjp(_response, prepared, drds)
        expected_prepared, expected_drds = pullback(field_bars)
        _assert_float_tree_allclose(
            jax.tree_util.tree_map(lambda value: value[rhs_index], native_prepared),
            expected_prepared,
            rtol=5e-9,
        )
        _assert_float_tree_allclose(
            native_drds[rhs_index], expected_drds, rtol=5e-9
        )


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
