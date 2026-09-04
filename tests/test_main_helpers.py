import dataclasses
import collections
import types
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import pytest
from ntx import GridSpec, example_surface, prepare_monoenergetic_system

import NEOPAX._reverse_ad_transport as reverse_transport_module
import NEOPAX._reverse_ad_initial_er as initial_er_module
from NEOPAX._orchestrator import (
    _build_database,
    _build_flux_model,
    _load_user_extensions,
    _load_ntss_reference_profiles,
    _normalize_solver_config,
    _resolve_reference_path,
    Models,
    RuntimeContext,
)
from NEOPAX._reverse_ad_initial_er import (
    fold_recorded_ntx_scan_database_bar_into_support,
    fold_recorded_ntx_scan_database_bars_into_support,
    realtime_geometry_payload_for_runtime,
    realtime_geometry_reverse_support_payload_for_runtime,
    runtime_without_recorded_ntx_scan_primal,
    runtime_with_geometry_payload,
    runtime_with_realtime_geometry_payload,
    runtime_with_realtime_geometry_reverse_support_payload,
)
from NEOPAX._reverse_ad_transport import (
    prepare_realtime_geometry_support_segment_core_setup,
    realtime_geometry_payload_pullback_result,
)
from NEOPAX._monoenergetic import (
    MONOENERGETIC_KIND_GENERIC,
    MONOENERGETIC_KIND_PREPROCESSED_3D_NTSS1D_FIXED,
    MONOENERGETIC_KIND_PREPROCESSED_3D_RADIAL_NTSS1D,
    load_monoenergetic_database,
    database_with_geometry_scale,
    monoenergetic_database_kind,
)
from NEOPAX._database import Monoenergetic
from NEOPAX._database_preprocessed import (
    PreprocessedMonoenergetic3D,
    PreprocessedMonoenergetic3DNTSSRadius,
    PreprocessedMonoenergetic3DNTSSRadiusNTSS1D,
    PreprocessedMonoenergetic3DNTSSRadiusNTSS1DFixedNU,
)
from NEOPAX._interpolators_preprocessed import (
    _bilinear,
    get_Dij_preprocessed_3d_ntss_radius,
    radial_preprocessed_interpolation_stencil,
    radial_preprocessed_interpolation_table_bar,
)
from NEOPAX._neoclassical import (
    get_Neoclassical_Fluxes,
    pullback_preprocessed_radial_database_fluxes,
)
from NEOPAX._monoenergetic_interpolators import monoenergetic_interpolation_kernel
from NEOPAX._interpolators import get_Dij, monoenergetic_interpolation_table_bar
from NEOPAX._source_models import get_source_model
from NEOPAX._species import Species
from NEOPAX._state import TransportState
from NEOPAX._transport_flux_models import (
    NTXDatabaseTransportModel,
    NTXExactLijRuntimeTransportModel,
    NTXExactLijRuntimeSupport,
    NTXFullStateQuadraticPreparedCoefficientResponse,
    NTXQuadraticPreparedCoefficientResponse,
    NTXRuntimeScanChannels,
    _as_float_array,
    NTXRuntimeScanTransportModel,
    _sanitize_float_delta_bar_tree,
    _ntx_runtime_scan_to_neopax_monoenergetic,
    build_face_transport_state,
    build_ntx_exact_lij_runtime_transport_model,
    build_ntx_runtime_scan_channels,
    build_ntx_runtime_scan_transport_model,
    get_transport_flux_model,
)


def _tiny_ntx_runtime_channels(rho):
    values = jnp.asarray(rho, dtype=jnp.float64)
    ones = jnp.ones_like(values)
    return NTXRuntimeScanChannels.from_mapping(
        values,
        {
            "a_b": 1.0,
            "psia": 1.0,
            "b00": ones,
            "r00": ones,
            "boozer_i": ones,
            "boozer_g": ones,
            "iota": ones,
            "drds": ones,
            "dr_tildedr": ones,
            "dr_tildeds": ones,
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


def test_runtime_scan_axis_validation_is_trace_safe():
    """Recorded database support VJPs may carry the scan axes as tracers."""

    actual = jax.jit(lambda values: _as_float_array(values, name="rho_scan"))(
        jnp.asarray([0.25, 0.5])
    )
    assert jnp.allclose(actual, jnp.asarray([0.25, 0.5]))
    with pytest.raises(ValueError, match="rho_scan contains non-finite"):
        _as_float_array(jnp.asarray([0.25, jnp.nan]), name="rho_scan")


def test_runtime_scan_axis_range_validation_is_trace_safe():
    model = build_ntx_runtime_scan_transport_model(
        species="species", energy_grid="grid", geometry="geometry",
        vmec_file=None, boozer_file=None,
        ntx_scan_rho=[0.25, 0.5], ntx_scan_nu_v=[1.0e-4, 1.0e-3],
        ntx_scan_er_tilde=[0.0, 1.0e-4], prebuild_database=False,
    )
    actual = jax.jit(
        lambda rho: dataclasses.replace(model, rho_scan=rho)._scan_axes()[0]
    )(jnp.asarray([0.25, 0.5]))
    assert jnp.allclose(actual, jnp.asarray([0.25, 0.5]))
    with pytest.raises(ValueError, match="rho_scan values must satisfy"):
        dataclasses.replace(model, rho_scan=jnp.asarray([0.0, 0.5]))._scan_axes()


def test_reverse_initial_state_preserves_fixed_temperature_at_density_floor(monkeypatch):
    """The reverse initial root must see the same floor regularization as forward."""

    baseline_state = TransportState(
        density=jnp.ones((4, 2)),
        pressure=jnp.ones((4, 2)),
        Er=jnp.zeros((2,)),
    )
    profile_set = types.SimpleNamespace(
        density=jnp.asarray(
            [[2.0e20, 2.0e20], [1.0e20, 1.0e20], [1.0e20, 1.0e20], [0.0, 0.0]]
        ),
        temperature=jnp.asarray(
            [[8.0e3, 8.0e3], [7.0e3, 7.0e3], [7.0e3, 7.0e3], [0.7e3, 0.7e3]]
        ),
    )
    runtime = types.SimpleNamespace(
        geometry=object(),
        species=types.SimpleNamespace(number_species=4),
    )
    monkeypatch.setattr(
        reverse_transport_module,
        "parameterized_profile_set",
        lambda *args, **kwargs: profile_set,
    )

    state = reverse_transport_module.initial_state_for_parameter_vector(
        jnp.asarray([4.21, 17.8, 2.0, 2.0]),
        baseline_state=baseline_state,
        profile_cfg={},
        runtime=runtime,
        config={"transport_solver": {"density_floor": 1.0e-6, "temperature_floor": 1.0e-6}},
    )

    assert jnp.allclose(state.density[3], 0.0)
    assert jnp.allclose(state.pressure[3], 0.7e-6)
    assert jnp.allclose(state.temperature[3], 0.7)


def _repeat_ntx_prepared(prepared, count):
    return jax.tree_util.tree_map(
        lambda *values: None if values[0] is None else jnp.stack(values, axis=0),
        *([prepared] * count),
    )


def test_ntx_exact_runtime_lagged_face_response_matches_reference_and_finite_difference():
    """The exact NTX face response must agree locally with its live face flux."""
    geometry = types.SimpleNamespace(
        a_b=1.0,
        r_grid=jnp.asarray([0.3, 0.7]),
        r_grid_half=jnp.asarray([0.1, 0.5, 0.9]),
    )
    species = Species(
        number_species=2,
        species_indices=jnp.asarray([0, 1]),
        mass_mp=jnp.asarray([5.446e-4, 2.0]),
        charge_qp=jnp.asarray([-1.0, 1.0]),
        names=("e", "D"),
    )
    energy_grid = types.SimpleNamespace(
        xWeights=jnp.asarray([0.2, 0.3, 0.5]),
        L11_weight=jnp.asarray([1.0, 0.8, 1.2]),
        L12_weight=jnp.asarray([0.1, -0.2, 0.3]),
        L22_weight=jnp.asarray([0.9, 1.1, 0.7]),
        L13_weight=jnp.asarray([0.4, 0.5, 0.6]),
        L23_weight=jnp.asarray([-0.3, 0.2, 0.1]),
        L33_weight=jnp.asarray([1.3, 0.6, 0.9]),
        v_norm=jnp.asarray([1.7, 1.8, 1.9]),
    )
    prepared = prepare_monoenergetic_system(example_surface(), GridSpec(3, 3, 2))
    support = NTXExactLijRuntimeSupport(
        center_channels=_tiny_ntx_runtime_channels(geometry.r_grid),
        face_channels=_tiny_ntx_runtime_channels(geometry.r_grid_half),
        center_prepared=_repeat_ntx_prepared(prepared, 2),
        face_prepared=_repeat_ntx_prepared(prepared, 3),
        grid=GridSpec(3, 3, 2),
    )
    model = NTXExactLijRuntimeTransportModel(
        species=species,
        energy_grid=energy_grid,
        geometry=geometry,
        vmec_file=None,
        boozer_file=None,
        support=support,
        center_response_mode="interpolate_from_faces",
        response_anchor_count=2,
    )
    state0 = TransportState(
        density=jnp.asarray([[1.0, 1.15], [1.0, 1.15]]),
        pressure=jnp.asarray([[1.3, 1.61], [1.1, 1.38]]),
        Er=jnp.asarray([2.0e-4, 2.5e-4]),
    )
    direction = TransportState(
        density=jnp.asarray([[0.03, -0.02], [0.02, -0.01]]),
        pressure=jnp.asarray([[0.04, -0.03], [0.03, -0.02]]),
        Er=jnp.asarray([1.0e-5, -0.8e-5]),
    )

    def face_fluxes_from_state(state):
        faces = build_face_transport_state(state, geometry)
        return model.evaluate_face_fluxes(state, faces)

    response = model.build_lagged_response(state0)
    recorded_response, compact_record = model.build_lagged_response_with_compact_coefficient_record(state0)
    for ordinary_leaf, recorded_leaf in zip(
        jax.tree_util.tree_leaves(response),
        jax.tree_util.tree_leaves(recorded_response),
        strict=True,
    ):
        if jnp.issubdtype(jnp.asarray(ordinary_leaf).dtype, jnp.inexact):
            assert jnp.allclose(recorded_leaf, ordinary_leaf, rtol=1.0e-9, atol=1.0e-11)
    assert compact_record.face_anchor_coefficients.coefficient_scan.shape == (2, 2, 3, 5)
    assert compact_record.face_anchor_coefficients.dcoefficient_scan_d_er.shape == (2, 2, 3, 5)
    assert compact_record.face_anchor_coefficients.dcoefficient_scan_d_log_nu_star.shape == (2, 2, 3, 5)
    lagged_at_reference = model.evaluate_with_lagged_response(state0, response)
    direct_at_reference = face_fluxes_from_state(state0)
    for name in ("Gamma", "Q", "Upar"):
        assert jnp.allclose(
            lagged_at_reference[f"{name}_faces"],
            direct_at_reference[name],
            rtol=3.0e-6,
            atol=1.0e-12,
        )

    epsilon = jnp.asarray(1.0e-4)
    state_plus = jax.tree_util.tree_map(
        lambda value, delta: value + epsilon * delta,
        state0,
        direction,
    )
    state_minus = jax.tree_util.tree_map(
        lambda value, delta: value - epsilon * delta,
        state0,
        direction,
    )
    finite_difference = jax.tree_util.tree_map(
        lambda plus, minus: (plus - minus) / (2.0 * epsilon),
        face_fluxes_from_state(state_plus),
        face_fluxes_from_state(state_minus),
    )
    lagged_direction = jax.tree_util.tree_map(
        lambda value, reference: value - reference,
        model.evaluate_with_lagged_response(state_plus, response),
        lagged_at_reference,
    )
    for name in ("Gamma", "Q", "Upar"):
        assert jnp.allclose(
            lagged_direction[f"{name}_faces"] / epsilon,
            finite_difference[name],
            rtol=2.0e-2,
            atol=1.0e-8,
        )


@pytest.mark.parametrize("response_anchor_count", (2, 3))
def test_ntx_exact_runtime_quadratic_lagged_response_matches_live_reference(response_anchor_count):
    """Quadratic realtime payloads work for reduced and full radial layouts."""
    geometry = types.SimpleNamespace(
        a_b=1.0,
        r_grid=jnp.asarray([0.3, 0.7]),
        r_grid_half=jnp.asarray([0.1, 0.5, 0.9]),
    )
    species = Species(
        number_species=2,
        species_indices=jnp.asarray([0, 1]),
        mass_mp=jnp.asarray([5.446e-4, 2.0]),
        charge_qp=jnp.asarray([-1.0, 1.0]),
        names=("e", "D"),
    )
    energy_grid = types.SimpleNamespace(
        xWeights=jnp.asarray([0.2, 0.3, 0.5]),
        L11_weight=jnp.asarray([1.0, 0.8, 1.2]),
        L12_weight=jnp.asarray([0.1, -0.2, 0.3]),
        L22_weight=jnp.asarray([0.9, 1.1, 0.7]),
        L13_weight=jnp.asarray([0.4, 0.5, 0.6]),
        L23_weight=jnp.asarray([-0.3, 0.2, 0.1]),
        L33_weight=jnp.asarray([1.3, 0.6, 0.9]),
        v_norm=jnp.asarray([1.7, 1.8, 1.9]),
    )
    prepared = prepare_monoenergetic_system(example_surface(), GridSpec(3, 3, 2))
    support = NTXExactLijRuntimeSupport(
        center_channels=_tiny_ntx_runtime_channels(geometry.r_grid),
        face_channels=_tiny_ntx_runtime_channels(geometry.r_grid_half),
        center_prepared=_repeat_ntx_prepared(prepared, 2),
        face_prepared=_repeat_ntx_prepared(prepared, 3),
        grid=GridSpec(3, 3, 2),
    )
    model = NTXExactLijRuntimeTransportModel(
        species=species,
        energy_grid=energy_grid,
        geometry=geometry,
        vmec_file=None,
        boozer_file=None,
        support=support,
        center_response_mode="interpolate_from_faces",
        response_anchor_count=response_anchor_count,
        lagged_response_taylor_order=2,
    )
    state = TransportState(
        density=jnp.asarray([[1.0, 1.15], [1.0, 1.15]]),
        pressure=jnp.asarray([[1.3, 1.61], [1.1, 1.38]]),
        Er=jnp.asarray([2.0e-4, 2.5e-4]),
    )
    response = model.build_lagged_response(state)
    assert isinstance(response.face_response, NTXQuadraticPreparedCoefficientResponse)
    assert response.center_response is None
    with pytest.raises(NotImplementedError, match="linear-response reverse-replay"):
        model.build_lagged_response_with_compact_coefficient_record(state)

    # A reduced radial payload interpolates coefficient data onto the omitted
    # face.  It is therefore valid, but cannot be an anchor-identity test.
    if response_anchor_count != 3:
        lagged = model.evaluate_with_lagged_response(state, response)
        for value in lagged.values():
            assert bool(jnp.all(jnp.isfinite(value)))
        return

    faces = build_face_transport_state(state, geometry)
    direct = model.evaluate_face_fluxes(state, faces)
    lagged = model.evaluate_with_lagged_response(state, response)
    for name in ("Gamma", "Q", "Upar"):
        assert jnp.allclose(
            lagged[f"{name}_faces"], direct[name], rtol=3.0e-6, atol=1.0e-12
        )

    full_state_model = dataclasses.replace(model, full_state_quadratic_response=True)
    full_state_response = dataclasses.replace(
        response,
        face_response=NTXFullStateQuadraticPreparedCoefficientResponse(
            reference_state=state,
            coefficient_response=response.face_response,
        ),
    )
    assert isinstance(full_state_response.face_response, NTXFullStateQuadraticPreparedCoefficientResponse)
    full_state = full_state_model.evaluate_with_lagged_response(state, full_state_response)
    for name in ("Gamma", "Q", "Upar"):
        assert jnp.allclose(
            full_state[f"{name}_faces"], direct[name], rtol=3.0e-6, atol=1.0e-12
        )

    direct_full_state_model = dataclasses.replace(
        model,
        center_response_mode="center_local_response",
        full_state_quadratic_response=True,
    )
    direct_full_state_response = direct_full_state_model.build_lagged_response(state)
    assert isinstance(
        direct_full_state_response.center_response,
        NTXFullStateQuadraticPreparedCoefficientResponse,
    )
    assert isinstance(
        direct_full_state_response.face_response,
        NTXFullStateQuadraticPreparedCoefficientResponse,
    )
    direct_full_state = direct_full_state_model.evaluate_with_lagged_response(
        state, direct_full_state_response
    )
    direct_center = direct_full_state_model(state)
    for name in ("Gamma", "Q", "Upar"):
        assert jnp.allclose(
            direct_full_state[name], direct_center[name], rtol=3.0e-6, atol=1.0e-12
        )
        assert jnp.allclose(
            direct_full_state[f"{name}_faces"], direct[name], rtol=3.0e-6, atol=1.0e-12
        )

    direction = TransportState(
        density=jnp.asarray([[0.04, -0.03], [-0.02, 0.01]]),
        pressure=jnp.asarray([[0.06, -0.04], [-0.03, 0.02]]),
        Er=jnp.asarray([2.0e-5, -1.5e-5]),
    )

    direct_full_state_tangent = (
        direct_full_state_model.evaluate_with_lagged_response_tangent(
            state, direction, direct_full_state_response
        )
    )
    for name in ("Gamma", "Q", "Upar"):
        assert name in direct_full_state_tangent
        assert f"{name}_faces" in direct_full_state_tangent

    def _state_at(scale):
        return TransportState(
            density=state.density + scale * direction.density,
            pressure=state.pressure + scale * direction.pressure,
            Er=state.Er + scale * direction.Er,
        )

    def _relative_flux_error(scale):
        perturbed = _state_at(scale)
        direct_perturbed = model.evaluate_face_fluxes(
            perturbed, build_face_transport_state(perturbed, geometry)
        )
        full_perturbed = full_state_model.evaluate_with_lagged_response(
            perturbed, full_state_response
        )
        errors = tuple(
            jnp.max(jnp.abs(full_perturbed[f"{name}_faces"] - direct_perturbed[name]))
            / (1.0 + jnp.max(jnp.abs(direct_perturbed[name])))
            for name in ("Gamma", "Q", "Upar")
        )
        return jnp.max(jnp.asarray(errors))

    # The full-state payload is a quadratic Taylor model.  Away from its
    # anchor its direct-flux defect must decrease cubically as the same state
    # displacement is halved.
    error_full = _relative_flux_error(1.0)
    error_half = _relative_flux_error(0.5)
    assert float(error_full) < 2.0e-2
    assert float(error_half / error_full) < 0.3


def test_ntx_exact_fused_lowdot_local_pullback_matches_ntx_helper():
    """The opt-in fused local NTX path must preserve the scalar local bars."""
    energy_grid = types.SimpleNamespace(
        xWeights=jnp.asarray([0.2, 0.3, 0.5]),
        L11_weight=jnp.asarray([1.0, 0.8, 1.2]),
        L12_weight=jnp.asarray([0.1, -0.2, 0.3]),
        L22_weight=jnp.asarray([0.9, 1.1, 0.7]),
        L13_weight=jnp.asarray([0.4, 0.5, 0.6]),
        L23_weight=jnp.asarray([-0.3, 0.2, 0.1]),
        L33_weight=jnp.asarray([1.3, 0.6, 0.9]),
        v_norm=jnp.asarray([1.7, 1.8, 1.9]),
    )
    common = dict(
        species=object(),
        energy_grid=energy_grid,
        geometry=object(),
        vmec_file=None,
        boozer_file=None,
    )
    reference = NTXExactLijRuntimeTransportModel(**common)
    fused = reference.with_derivative_pullback_algebra("ntx_helper_lowdot_fused")
    prepared = prepare_monoenergetic_system(example_surface(), GridSpec(5, 5, 4))
    field_bars = (
        jnp.asarray(0.2),
        jnp.asarray([0.4, -0.2, 0.1, 0.3, -0.5, 0.2]),
        jnp.asarray([-0.3, 0.2, 0.4, -0.1, 0.5, 0.2]),
        jnp.asarray([0.1, 0.3, -0.2, 0.5, -0.4, 0.2]),
    )
    kwargs = dict(
        prepared=prepared,
        drds_value=jnp.asarray(1.2),
        reference_nu_hat=jnp.asarray([1.0e-2, 1.5e-2, 2.0e-2]),
        reference_epsi_hat=jnp.asarray([1.0e-3, -2.0e-3, 1.5e-3]),
        vth_a=jnp.asarray(2.3),
        field_bars=field_bars,
    )
    reference_bars = reference._pullback_interpolated_moment_reduced_local_outputs(**kwargs)
    fused_bars = fused._pullback_interpolated_moment_reduced_local_outputs(**kwargs)
    for fused_bar, reference_bar in zip(fused_bars, reference_bars, strict=True):
        assert jnp.allclose(fused_bar, reference_bar, rtol=1.0e-9, atol=1.0e-11)


def test_ntx_exact_factorized_two_directional_local_response_matches_generic_jvps():
    """The isolated rebuild primitive must match the existing local response."""
    energy_grid = types.SimpleNamespace(
        xWeights=jnp.asarray([0.2, 0.3, 0.5]),
        L11_weight=jnp.asarray([1.0, 0.8, 1.2]),
        L12_weight=jnp.asarray([0.1, -0.2, 0.3]),
        L22_weight=jnp.asarray([0.9, 1.1, 0.7]),
        L13_weight=jnp.asarray([0.4, 0.5, 0.6]),
        L23_weight=jnp.asarray([-0.3, 0.2, 0.1]),
        L33_weight=jnp.asarray([1.3, 0.6, 0.9]),
        v_norm=jnp.asarray([1.7, 1.8, 1.9]),
    )
    model = NTXExactLijRuntimeTransportModel(
        species=object(),
        energy_grid=energy_grid,
        geometry=object(),
        vmec_file=None,
        boozer_file=None,
    )
    prepared = prepare_monoenergetic_system(example_surface(), GridSpec(3, 3, 2))
    nu_hat = jnp.asarray([1.0e-2, 1.5e-2, 2.0e-2])
    epsi_hat = jnp.asarray([1.0e-3, -2.0e-3, 1.5e-3])
    vth_a = jnp.asarray(2.3)
    drds = jnp.asarray(1.2)

    def _response(prepared_value, drds_value, *, factorized):
        return model._interpolated_moment_reduced_local_outputs_from_primitives(
            prepared_value,
            drds_value=drds_value,
            nu_hat_a=nu_hat,
            epsi_hat_a=epsi_hat,
            vth_a=vth_a,
            use_factorized_ntx_two_directional_prepared_vjp=factorized,
        )

    reference = _response(prepared, drds, factorized=False)
    factorized = jax.jit(
        lambda prepared_value, drds_value: _response(
            prepared_value, drds_value, factorized=True
        )
    )(prepared, drds)
    for actual, expected in zip(factorized, reference, strict=True):
        assert jnp.allclose(actual, expected, rtol=1.0e-9, atol=1.0e-11)


def test_ntx_exact_support_only_prepared_pullback_matches_joint_helper():
    """The isolated rebuild helper preserves prepared, ``drds``, and primal fields."""
    energy_grid = types.SimpleNamespace(
        xWeights=jnp.asarray([0.2, 0.3, 0.5]),
        L11_weight=jnp.asarray([1.0, 0.8, 1.2]),
        L12_weight=jnp.asarray([0.1, -0.2, 0.3]),
        L22_weight=jnp.asarray([0.9, 1.1, 0.7]),
        L13_weight=jnp.asarray([0.4, 0.5, 0.6]),
        L23_weight=jnp.asarray([-0.3, 0.2, 0.1]),
        L33_weight=jnp.asarray([1.3, 0.6, 0.9]),
        v_norm=jnp.asarray([1.7, 1.8, 1.9]),
    )
    model = NTXExactLijRuntimeTransportModel(
        species=object(),
        energy_grid=energy_grid,
        geometry=object(),
        vmec_file=None,
        boozer_file=None,
    )
    prepared = prepare_monoenergetic_system(example_surface(), GridSpec(5, 5, 4))
    kwargs = dict(
        prepared=prepared,
        drds_value=jnp.asarray(1.2),
        reference_nu_hat=jnp.asarray([1.0e-2, 1.5e-2, 2.0e-2]),
        reference_epsi_hat=jnp.asarray([1.0e-3, -2.0e-3, 1.5e-3]),
        vth_a=jnp.asarray(2.3),
        field_bars=(
            jnp.asarray(0.2),
            jnp.asarray([0.4, -0.2, 0.1, 0.3, -0.5, 0.2]),
            jnp.asarray([-0.3, 0.2, 0.4, -0.1, 0.5, 0.2]),
            jnp.asarray([0.1, 0.3, -0.2, 0.5, -0.4, 0.2]),
        ),
    )
    joint = model._pullback_interpolated_moment_reduced_local_outputs_with_prepared_support_and_drds(
        **kwargs
    )
    support_only = jax.jit(
        lambda: model._pullback_interpolated_moment_prepared_support_and_drds_only(**kwargs)
    )()
    expected_primal = model._interpolated_moment_reduced_local_outputs_from_primitives(
        prepared,
        drds_value=kwargs["drds_value"],
        nu_hat_a=kwargs["reference_nu_hat"],
        epsi_hat_a=kwargs["reference_epsi_hat"],
        vth_a=kwargs["vth_a"],
    )
    actual_prepared, actual_drds, actual_primal = support_only
    for actual_leaf, expected_leaf in zip(
        jax.tree_util.tree_leaves(actual_prepared),
        jax.tree_util.tree_leaves(joint[3]),
        strict=True,
    ):
        if jnp.issubdtype(jnp.asarray(expected_leaf).dtype, jnp.inexact):
            assert jnp.allclose(actual_leaf, expected_leaf, rtol=1.0e-9, atol=1.0e-11)
    assert jnp.allclose(actual_drds, joint[4], rtol=1.0e-9, atol=1.0e-11)
    for actual_field, expected_field in zip(actual_primal, expected_primal, strict=True):
        assert jnp.allclose(actual_field, expected_field, rtol=1.0e-9, atol=1.0e-11)

    batched_field_bars = tuple(
        jnp.stack([field_bar, field_bar], axis=0)
        for field_bar in kwargs["field_bars"]
    )
    def _sanitized_support_only(first_bar, second_bar, third_bar, fourth_bar):
        prepared_bar, drds_bar, primal_response = (
            model._pullback_interpolated_moment_prepared_support_and_drds_only(
                prepared,
                drds_value=kwargs["drds_value"],
                reference_nu_hat=kwargs["reference_nu_hat"],
                reference_epsi_hat=kwargs["reference_epsi_hat"],
                vth_a=kwargs["vth_a"],
                field_bars=(first_bar, second_bar, third_bar, fourth_bar),
            )
        )
        return (
            *jax.tree_util.tree_leaves(
                _sanitize_float_delta_bar_tree(prepared, prepared_bar)
            ),
            drds_bar,
            *primal_response,
        )

    batched_support_only = jax.jit(jax.vmap(_sanitized_support_only))(
        *batched_field_bars
    )
    expected_batched = tuple(
        jnp.broadcast_to(value, (2,) + jnp.asarray(value).shape)
        for value in _sanitized_support_only(*kwargs["field_bars"])
    )
    for actual_leaf, expected_leaf in zip(batched_support_only, expected_batched, strict=True):
        if jnp.issubdtype(jnp.asarray(expected_leaf).dtype, jnp.inexact):
            assert jnp.allclose(actual_leaf, expected_leaf, rtol=1.0e-9, atol=1.0e-11)


def test_joint_local_pullback_primal_output_preserves_existing_bars():
    """The opt-in joint helper must add only the local primal response."""
    model = NTXExactLijRuntimeTransportModel(
        species=types.SimpleNamespace(number_species=1, mass=jnp.asarray([1.0])),
        energy_grid=object(),
        geometry=object(),
        vmec_file=None,
        boozer_file=None,
    )
    prepared = {"coefficient": jnp.asarray([2.0, -1.0])}
    primal_response = (
        jnp.asarray([0.25]),
        jnp.asarray([[1.0, -2.0]]),
        jnp.asarray([[3.0, 4.0]]),
        jnp.asarray([[5.0, 6.0]]),
    )
    object.__setattr__(model, "_interpolated_moment_local_scan_primitives", lambda **_kwargs: (
        jnp.asarray(1.0),
        jnp.asarray(2.0),
        jnp.asarray(3.0),
    ))
    object.__setattr__(model, "_pullback_interpolated_moment_reduced_local_outputs_with_prepared_support_and_drds", (
        lambda prepared, **_kwargs: (
            jnp.asarray(7.0),
            jnp.asarray(8.0),
            jnp.asarray(9.0),
            {"coefficient": jnp.asarray([10.0, 11.0])},
            jnp.asarray(12.0),
        )
    ))
    object.__setattr__(model, "_pullback_local_scan_inputs_and_drds_from_primitives", lambda **_kwargs: (
        jnp.asarray(13.0),
        jnp.asarray(14.0),
        jnp.asarray(15.0),
        jnp.asarray(16.0),
    ))
    object.__setattr__(model, "_interpolated_moment_reduced_local_outputs_from_primitives", (
        lambda *_args, **_kwargs: primal_response
    ))
    field_bars = (
        jnp.asarray([0.1]),
        jnp.asarray([[0.2, 0.3]]),
        jnp.asarray([[0.4, 0.5]]),
        jnp.asarray([[0.6, 0.7]]),
    )
    common = dict(
        prepared=prepared,
        drds_value=jnp.asarray(1.0),
        er_value=jnp.asarray(2.0),
        temperature_local=jnp.asarray(3.0),
        density_local=jnp.asarray(4.0),
        collisionality_kind="none",
        field_bars=field_bars,
    )
    reference = model._pullback_interpolated_moment_response_local_fields_and_prepared_support_and_drds_flat_prepared(
        **common
    )
    actual = model._pullback_interpolated_moment_response_local_fields_and_prepared_support_and_drds_flat_prepared(
        **common,
        return_primal_response=True,
    )
    for actual_leaf, reference_leaf in zip(actual[:5], reference, strict=True):
        if isinstance(actual_leaf, tuple):
            for actual_subleaf, reference_subleaf in zip(actual_leaf, reference_leaf, strict=True):
                assert jnp.allclose(actual_subleaf, reference_subleaf)
        else:
            assert jnp.allclose(actual_leaf, reference_leaf)
    for actual_field, expected_field in zip(actual[5], primal_response, strict=True):
        assert jnp.allclose(actual_field, expected_field)


def test_scalar_joint_local_pullback_mock_retains_outer_objective_batch_only():
    """The scalar joint primitive is trace-once under an outer RHS batch.

    This is deliberately a tiny mocked contract test.  It guards the intended
    next rebuild route: state and support bars are returned together by the
    scalar local primitive, while the caller alone owns the objective/RHS
    axis.  No production NTX solve, transport rollout, profiling, or file
    output is involved.
    """
    model = NTXExactLijRuntimeTransportModel(
        species=types.SimpleNamespace(number_species=1, mass=jnp.asarray([1.0])),
        energy_grid=object(),
        geometry=object(),
        vmec_file=None,
        boozer_file=None,
    )
    prepared = {"coefficient": jnp.asarray([2.0, -1.0])}
    calls = {"joint_lowdot_contract": 0}

    object.__setattr__(model, "_interpolated_moment_local_scan_primitives", lambda **_kwargs: (
        jnp.asarray(1.0),
        jnp.asarray(2.0),
        jnp.asarray(3.0),
    ))

    def _joint_lowdot_contract(prepared_value, **_kwargs):
        calls["joint_lowdot_contract"] += 1
        return (
            jnp.asarray(7.0),
            jnp.asarray(8.0),
            jnp.asarray(9.0),
            {"coefficient": jnp.asarray([10.0, 11.0])},
            jnp.asarray(12.0),
        )

    object.__setattr__(
        model,
        "_pullback_interpolated_moment_reduced_local_outputs_with_prepared_support_and_drds",
        _joint_lowdot_contract,
    )
    object.__setattr__(model, "_pullback_local_scan_inputs_and_drds_from_primitives", lambda **_kwargs: (
        jnp.asarray(13.0),
        jnp.asarray(14.0),
        jnp.asarray(15.0),
        jnp.asarray(16.0),
    ))

    common = dict(
        prepared=prepared,
        drds_value=jnp.asarray(1.0),
        er_value=jnp.asarray(2.0),
        temperature_local=jnp.asarray(3.0),
        density_local=jnp.asarray(4.0),
        collisionality_kind="none",
    )
    one_rhs = (
        jnp.asarray([0.1]),
        jnp.asarray([[0.2, 0.3]]),
        jnp.asarray([[0.4, 0.5]]),
        jnp.asarray([[0.6, 0.7]]),
    )

    def _one_objective(*field_bars):
        return model._pullback_interpolated_moment_response_local_fields_and_prepared_support_and_drds_flat_prepared(
            **common,
            field_bars=field_bars,
        )

    single = _one_objective(*one_rhs)
    assert calls["joint_lowdot_contract"] == 1
    assert jnp.allclose(single[0], 25.0)  # implicit + direct drds bars
    assert jnp.allclose(single[1], 14.0)
    assert jnp.allclose(single[2], 15.0)
    assert jnp.allclose(single[3], 16.0)
    assert jnp.allclose(single[4][0], jnp.asarray([10.0, 11.0]))

    objective_count = 20
    calls["joint_lowdot_contract"] = 0
    batched_rhs = tuple(
        jnp.broadcast_to(value, (objective_count,) + value.shape)
        for value in one_rhs
    )
    batched = jax.jit(jax.vmap(_one_objective))(*batched_rhs)

    # Python mock calls occur while JAX traces the one scalar body.  This
    # proves the objective axis is supplied by the outer vmap rather than a
    # host loop or an inner objective-specific helper construction.
    assert calls["joint_lowdot_contract"] == 1
    for actual, expected in zip(batched[:4], single[:4], strict=True):
        assert actual.shape[0] == objective_count
        assert jnp.allclose(actual, jnp.broadcast_to(expected, actual.shape))
    assert batched[4][0].shape[0] == objective_count
    assert jnp.allclose(
        batched[4][0],
        jnp.broadcast_to(single[4][0], batched[4][0].shape),
    )


def test_scalar_joint_local_pullback_mock_matches_existing_state_pullback():
    """Joint local bars retain the established scalar state-bar contract."""
    model = NTXExactLijRuntimeTransportModel(
        species=types.SimpleNamespace(number_species=1, mass=jnp.asarray([1.0])),
        energy_grid=object(),
        geometry=object(),
        vmec_file=None,
        boozer_file=None,
    )
    prepared = {"coefficient": jnp.asarray([2.0, -1.0])}
    object.__setattr__(model, "_interpolated_moment_local_scan_primitives", lambda **_kwargs: (
        jnp.asarray(1.0),
        jnp.asarray(2.0),
        jnp.asarray(3.0),
    ))
    object.__setattr__(model, "_pullback_interpolated_moment_reduced_local_outputs", lambda *_args, **_kwargs: (
        jnp.asarray(8.0),
        jnp.asarray(9.0),
        jnp.asarray(10.0),
    ))
    object.__setattr__(model, "_pullback_local_scan_inputs_from_primitives", lambda **_kwargs: (
        jnp.asarray(14.0),
        jnp.asarray(15.0),
        jnp.asarray(16.0),
    ))
    object.__setattr__(model, "_pullback_interpolated_moment_reduced_local_outputs_with_prepared_support_and_drds", (
        lambda prepared_value, **_kwargs: (
            jnp.asarray(8.0),
            jnp.asarray(9.0),
            jnp.asarray(10.0),
            {"coefficient": jnp.asarray([10.0, 11.0])},
            jnp.asarray(12.0),
        )
    ))
    object.__setattr__(model, "_pullback_local_scan_inputs_and_drds_from_primitives", lambda **_kwargs: (
        jnp.asarray(25.0),
        jnp.asarray(14.0),
        jnp.asarray(15.0),
        jnp.asarray(16.0),
    ))
    field_bars = (
        jnp.asarray([0.1]),
        jnp.asarray([[0.2, 0.3]]),
        jnp.asarray([[0.4, 0.5]]),
        jnp.asarray([[0.6, 0.7]]),
    )
    common = dict(
        prepared=prepared,
        drds_value=jnp.asarray(1.0),
        er_value=jnp.asarray(2.0),
        temperature_local=jnp.asarray(3.0),
        density_local=jnp.asarray(4.0),
        collisionality_kind="none",
        field_bars=field_bars,
    )

    state_bars = model._pullback_interpolated_moment_response_local_fields(**common)
    joint_bars = (
        model._pullback_interpolated_moment_response_local_fields_and_prepared_support_and_drds_flat_prepared(
            **common
        )
    )

    for joint_state_bar, state_bar in zip(joint_bars[1:4], state_bars, strict=True):
        assert jnp.allclose(joint_state_bar, state_bar)
    # The scalar joint contract includes both the primitive-mediated drds bar
    # and the direct transport-moment drds bar.
    assert jnp.allclose(joint_bars[0], 37.0)
    assert jnp.allclose(joint_bars[4][0], jnp.asarray([10.0, 11.0]))


def test_native_joint_local_adapter_keeps_objective_rhs_inside_ntx_mock():
    """The native joint adapter calls NTX once per species, not per RHS.

    This is intentionally a pure mocked layout gate.  It verifies the only
    structural property needed before a remote numerical oracle: the local
    helper receives a species-major, objective-batched field cotangent and
    returns objective-major state/support bars.  No NTX solve, transport
    rollout, device compilation, or file output is involved.
    """
    model = NTXExactLijRuntimeTransportModel(
        species=types.SimpleNamespace(number_species=2, mass=jnp.asarray([1.0, 2.0])),
        energy_grid=types.SimpleNamespace(v_norm=jnp.asarray([1.0])),
        geometry=object(),
        vmec_file=None,
        boozer_file=None,
    )
    calls = []
    object.__setattr__(
        model,
        "_interpolated_moment_local_scan_primitives",
        lambda **_kwargs: (jnp.asarray([1.0]), jnp.asarray([2.0]), jnp.asarray(3.0)),
    )

    def _native_support_only(_prepared, *, field_bars, **_kwargs):
        calls.append(tuple(jnp.asarray(value).shape for value in field_bars))
        rhs_count = field_bars[0].shape[0]
        rhs = jnp.arange(rhs_count, dtype=jnp.float64)
        return (
            {"coefficient": jnp.stack((rhs + 1.0, rhs + 2.0), axis=1)},
            rhs + 0.5,
            None,
            (
                jnp.broadcast_to((rhs + 2.0)[:, None], (rhs_count, 1)),
                jnp.broadcast_to((rhs + 3.0)[:, None], (rhs_count, 1)),
                rhs + 4.0,
            ),
        )

    object.__setattr__(
        model,
        "_pullback_interpolated_moment_prepared_support_and_drds_only_multi_rhs",
        _native_support_only,
    )
    object.__setattr__(
        model,
        "_pullback_local_scan_inputs_and_drds_from_primitives",
        lambda *, reference_nu_hat_bar, reference_epsi_hat_bar, vth_a_bar, **_kwargs: (
            5.0 * jnp.sum(reference_nu_hat_bar),
            7.0 * jnp.sum(reference_epsi_hat_bar),
            11.0 * vth_a_bar,
            13.0 * vth_a_bar,
        ),
    )
    rhs_count = 3
    field_bars = tuple(
        jnp.arange(2 * rhs_count * 2, dtype=jnp.float64).reshape(2, rhs_count, 2)
        for _ in range(4)
    )
    result = model._pullback_interpolated_moment_response_local_fields_and_prepared_support_and_drds_flat_prepared(
        {"coefficient": jnp.asarray([0.0, 0.0])},
        drds_value=jnp.asarray(1.0),
        er_value=jnp.asarray(2.0),
        temperature_local=jnp.asarray([3.0, 4.0]),
        density_local=jnp.asarray([5.0, 6.0]),
        collisionality_kind="none",
        field_bars=field_bars,
        native_factorized_ntx_rhs=True,
        reuse_joint_moment_drds_jvp=True,
    )
    # ``vmap`` traces the species body once.  Its native field argument still
    # has the complete RHS leading axis, proving objective batching was not
    # placed outside NTX.
    assert calls == [((rhs_count, 2),) * 4]
    drds_bar, er_bar, temperature_bar, density_bar, prepared_leaves = result
    rhs = jnp.arange(rhs_count, dtype=jnp.float64)
    assert jnp.allclose(drds_bar, 2.0 * (5.0 * (rhs + 2.0) + rhs + 0.5))
    assert jnp.allclose(er_bar, 2.0 * 7.0 * (rhs + 3.0))
    assert jnp.allclose(temperature_bar, 2.0 * 11.0 * (rhs + 4.0))
    assert jnp.allclose(density_bar, 2.0 * 13.0 * (rhs + 4.0))
    assert prepared_leaves[0].shape == (rhs_count, 2)


def test_joint_lowdot_scalar_rhs_layout_does_not_hide_rhs_axis_from_anchor_scan():
    """Reject a merely relocated objective ``vmap`` as a compile optimisation.

    The rejected joint prepared-lowdot route carries objective-batched support
    leaves through its anchor scan.  A tempting rewrite is to make the anchor
    function scalar in the objective RHS and wrap it in ``vmap`` outside the
    scan.  JAX's scan batching rule pushes that outer axis back into the scan
    carry, so this rewrite alone cannot make the segment HLO smaller.

    This is a pure-array, in-memory structural test: no transport or NTX
    solver is constructed or executed, and ``make_jaxpr`` does not compile for
    a device or write files.
    """
    objective_count = 3
    anchor_count = 4
    rhs = jnp.arange(objective_count * anchor_count, dtype=jnp.float64).reshape(
        objective_count, anchor_count
    )
    anchors = jnp.arange(anchor_count, dtype=jnp.int32)

    def _joint_with_batched_scan_carry(rhs_values):
        def _body(carry, anchor):
            state_bar, support_bar = carry
            local_bar = jax.lax.dynamic_index_in_dim(rhs_values, anchor, axis=1)
            return (
                state_bar + local_bar[:, None],
                {"geometry": support_bar["geometry"] + local_bar[:, None]},
            ), None

        return jax.lax.scan(
            _body,
            (
                jnp.zeros((objective_count, 2), dtype=rhs_values.dtype),
                {"geometry": jnp.zeros((objective_count, 5), dtype=rhs_values.dtype)},
            ),
            anchors,
        )[0]

    def _scalar_rhs_then_outer_vmap(rhs_values):
        def _one_rhs(one_rhs):
            def _body(carry, anchor):
                state_bar, support_bar = carry
                local_bar = jax.lax.dynamic_index_in_dim(one_rhs, anchor, axis=0)
                return (
                    state_bar + local_bar,
                    {"geometry": support_bar["geometry"] + local_bar},
                ), None

            return jax.lax.scan(
                _body,
                (
                    jnp.zeros((2,), dtype=one_rhs.dtype),
                    {"geometry": jnp.zeros((5,), dtype=one_rhs.dtype)},
                ),
                anchors,
            )[0]

        return jax.vmap(_one_rhs)(rhs_values)

    def _scan_carry_shapes(function):
        closed = jax.make_jaxpr(function)(rhs)
        scan_equations = [eqn for eqn in closed.jaxpr.eqns if eqn.primitive.name == "scan"]
        assert len(scan_equations) == 1
        scan = scan_equations[0]
        carry_count = scan.params["num_carry"]
        body = scan.params["jaxpr"].jaxpr
        return tuple(
            tuple(invar.aval.shape)
            for invar in body.invars[:carry_count]
        )

    # The proposed outer-vmap arrangement has exactly the same batched carry
    # shapes: moving the vmap syntactically is not a valid production change.
    assert _scan_carry_shapes(_joint_with_batched_scan_carry) == ((3, 2), (3, 5))
    assert _scan_carry_shapes(_scalar_rhs_then_outer_vmap) == ((3, 2), (3, 5))


def test_normalize_solver_config_prefers_transport_solver_section():
    config = {
        "transport_solver": {
            "transport_solver_backend": "theta_newton",
            "density_floor": 2.5e-6,
        },
        "solver": {
            "integrator": "radau",
        },
        "neoclassical": {"flux_model": "ntx_database"},
        "turbulence": {"flux_model": "none"},
    }

    out = _normalize_solver_config(config)
    assert out["transport_solver_backend"] == "theta_newton"
    assert out["integrator"] == "theta_newton"
    assert out["neoclassical_flux_model"] == "ntx_database"
    assert out["turbulence_flux_model"] == "none"
    assert out["density_floor"] == 2.5e-6
    assert out["Er_relax"] == 1.0
    assert out["DEr"] == 1.0


def test_normalize_solver_config_falls_back_to_legacy_solver_section():
    config = {
        "solver": {
            "integrator": "radau",
        },
        "neoclassical": {"flux_model": "none"},
        "turbulence": {"flux_model": "turbulent_power_analytical"},
    }

    out = _normalize_solver_config(config)
    assert out["transport_solver_backend"] == "radau"
    assert out["integrator"] == "radau"
    assert out["density_floor"] == 1.0e-6
    assert out["turbulence_flux_model"] == "turbulent_power_analytical"


def test_normalize_solver_config_t3d_outer_uses_lagged_response_for_any_flux_model():
    config = {
        "transport_solver": {"transport_solver_backend": "theta_t3d_outer"},
        "turbulence": {"flux_model": "turbulent_power_analytical"},
    }

    out = _normalize_solver_config(config)
    assert out["theta_rhs_mode"] == "lagged_transport_response"


@pytest.mark.parametrize(
    "rhs_mode",
    [
        "black_box",
        "lagged_linear_state",
    ],
)
def test_normalize_solver_config_t3d_outer_rejects_non_transport_lagged_rhs(rhs_mode):
    with pytest.raises(ValueError, match="lagged_transport_response"):
        _normalize_solver_config(
            {
                "transport_solver": {
                    "transport_solver_backend": "theta_t3d_outer",
                    "theta_rhs_mode": rhs_mode,
                },
            }
        )


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("direct", "direct"),
        ("direct_center", "direct"),
        ("interpolate_from_faces", "interpolate_from_faces"),
        ("interpolate_faces", "interpolate_from_faces"),
    ],
)
def test_normalize_solver_config_resolves_universal_center_flux_mode(configured, expected):
    config = {
        "transport_solver": {"center_flux_mode": configured},
        "neoclassical": {"flux_model": "ntx_scan_runtime"},
    }

    assert _normalize_solver_config(config)["transport_center_flux_mode"] == expected


def test_normalize_solver_config_maps_exact_ntx_center_response_mode_as_compatibility_alias():
    config = {
        "neoclassical": {
            "flux_model": "ntx_exact_lij_runtime",
            "ntx_exact_center_response_mode": "interpolate_from_faces",
        },
    }

    assert (
        _normalize_solver_config(config)["transport_center_flux_mode"]
        == "interpolate_from_faces"
    )


def test_normalize_solver_config_rejects_conflicting_universal_and_exact_ntx_center_modes():
    config = {
        "transport_solver": {"center_flux_mode": "direct"},
        "neoclassical": {
            "flux_model": "ntx_exact_lij_runtime",
            "ntx_exact_center_response_mode": "interpolate_from_faces",
        },
    }

    with pytest.raises(ValueError, match="conflicts"):
        _normalize_solver_config(config)


def test_normalize_solver_config_rejects_center_flux_mode_in_old_transport_flux_table():
    with pytest.raises(ValueError, match="has moved"):
        _normalize_solver_config(
            {"transport_flux": {"center_flux_mode": "interpolate_from_faces"}}
        )


def test_resolve_reference_path_handles_relative_paths(tmp_path, monkeypatch):
    ref = tmp_path / "ref.h5"
    ref.write_bytes(b"test")
    monkeypatch.chdir(tmp_path)

    resolved = _resolve_reference_path("ref.h5")
    assert resolved == ref.resolve()


def test_resolve_reference_path_returns_none_for_missing_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _resolve_reference_path("missing.h5") is None


def test_load_monoenergetic_database_dispatches_from_mode(monkeypatch):
    geometry = types.SimpleNamespace(a_b=1.2)

    monkeypatch.setattr(
        "NEOPAX._monoenergetic.PreprocessedMonoenergetic3DNTSSRadiusNTSS1DFixedNU.read_ntx",
        classmethod(lambda cls, a_b, ntx_file: {"kind": "fixed", "a_b": a_b, "file": ntx_file}),
    )

    out = load_monoenergetic_database(
        geometry,
        "db.h5",
        "preprocessed_3d_ntss1d_fixed",
    )

    assert out == {"kind": "fixed", "a_b": 1.2, "file": "db.h5"}


def test_build_database_uses_shared_monoenergetic_loader(monkeypatch):
    captured = {}

    def fake_loader(geometry, ntx_file, interpolation_mode):
        captured["geometry"] = geometry
        captured["file"] = ntx_file
        captured["mode"] = interpolation_mode
        return "database"

    monkeypatch.setattr("NEOPAX._orchestrator.load_monoenergetic_database", fake_loader)

    geometry = types.SimpleNamespace(a_b=1.5)
    config = {"neoclassical": {"neoclassical_file": "scan.h5", "interpolation_mode": "preprocessed_ntss"}}
    out = _build_database(config, geometry)

    assert out == "database"
    assert captured == {"geometry": geometry, "file": "scan.h5", "mode": "preprocessed_ntss"}


def test_monoenergetic_database_kind_defaults_to_generic():
    assert monoenergetic_database_kind(object()) == MONOENERGETIC_KIND_GENERIC


def test_monoenergetic_database_kind_prefers_most_specific_subclass():
    fixed = object.__new__(PreprocessedMonoenergetic3DNTSSRadiusNTSS1DFixedNU)
    ntss1d = object.__new__(PreprocessedMonoenergetic3DNTSSRadiusNTSS1D)
    assert monoenergetic_database_kind(fixed) == MONOENERGETIC_KIND_PREPROCESSED_3D_NTSS1D_FIXED
    assert monoenergetic_database_kind(ntss1d) == MONOENERGETIC_KIND_PREPROCESSED_3D_RADIAL_NTSS1D


def test_database_with_geometry_scale_rebuilds_generic_scale_coordinates():
    database = Monoenergetic(
        a_b=jnp.asarray(2.0),
        rho=jnp.asarray([0.1, 0.3, 0.6, 0.9, 1.0]),
        nu_log=jnp.asarray([-2.0, -1.0]),
        Er_list=jnp.asarray([[1.0, 2.0]] * 5),
        D11_log=jnp.zeros((5, 2, 2)),
        D13=jnp.zeros((5, 2, 2)),
        D33=jnp.zeros((5, 2, 2)),
    )

    actual = database_with_geometry_scale(database, jnp.asarray(4.0))

    assert jnp.allclose(actual.a_b, 4.0)
    assert jnp.allclose(actual.Er_list, database.Er_list + jnp.log10(0.5))
    assert jnp.allclose(actual.low_limit_r, 4.0e-3)
    assert jnp.allclose(actual.r1_lim, 4.0 * database.rho[1])
    assert jnp.allclose(actual.rnm1, 4.0 * database.rho[-1])
    assert actual.D11_log is database.D11_log

    _, tangent = jax.jvp(
        lambda scale: database_with_geometry_scale(database, scale).Er_list,
        (jnp.asarray(2.0),),
        (jnp.asarray(0.2),),
    )
    assert jnp.allclose(tangent, -0.2 / (2.0 * jnp.log(10.0)))


def test_database_with_geometry_scale_rebuilds_preprocessed_coordinates():
    database = PreprocessedMonoenergetic3D(
        a_b=jnp.asarray(2.0),
        rho=jnp.asarray([0.2, 0.5, 0.8]),
        r_grid=jnp.asarray([0.4, 1.0, 1.6]),
        nu_log=jnp.asarray([-2.0]),
        Er_grid=jnp.asarray([[1.0, 2.0]] * 3),
        D11_log=jnp.zeros((3, 1, 2)),
        D13=jnp.zeros((3, 1, 2)),
        D33=jnp.zeros((3, 1, 2)),
        Er_lower_limit=jnp.asarray(1.0e-8),
        low_limit_r=jnp.asarray(2.0e-3),
        del_r=jnp.asarray(1.0e-3),
    )

    actual = database_with_geometry_scale(database, jnp.asarray(4.0))

    assert jnp.allclose(actual.r_grid, 4.0 * database.rho)
    assert jnp.allclose(actual.Er_grid, database.Er_grid + jnp.log10(0.5))
    assert jnp.allclose(actual.low_limit_r, 4.0e-3)
    assert actual.D33 is database.D33


def test_runtime_geometry_replacement_rebuilds_database_scale():
    database = Monoenergetic(
        a_b=jnp.asarray(2.0),
        rho=jnp.asarray([0.1, 0.3, 0.6, 0.9, 1.0]),
        nu_log=jnp.asarray([-2.0]),
        Er_list=jnp.asarray([[1.0]] * 5),
        D11_log=jnp.zeros((5, 1, 1)),
        D13=jnp.zeros((5, 1, 1)),
        D33=jnp.zeros((5, 1, 1)),
    )
    old_geometry = types.SimpleNamespace(a_b=jnp.asarray(2.0))
    new_geometry = types.SimpleNamespace(a_b=jnp.asarray(4.0))
    model = NTXDatabaseTransportModel(
        species="species",
        energy_grid="grid",
        geometry=old_geometry,
        database=database,
    )
    runtime = RuntimeContext(
        species="species",
        energy_grid="grid",
        geometry=old_geometry,
        database=database,
        solver_parameters={},
        models=Models(flux=model),
    )

    actual = runtime_with_geometry_payload(runtime, new_geometry)

    assert actual.geometry is new_geometry
    assert actual.models.flux.geometry is new_geometry
    assert jnp.allclose(actual.models.flux.database.a_b, 4.0)
    assert actual.database is actual.models.flux.database
    assert jnp.allclose(
        actual.models.flux.database.Er_list,
        database.Er_list + jnp.log10(0.5),
    )


def test_realtime_geometry_payload_tags_database_without_exact_support_lookup():
    database = Monoenergetic(
        a_b=jnp.asarray(2.0),
        rho=jnp.asarray([0.1, 0.3, 0.6, 0.9, 1.0]),
        nu_log=jnp.asarray([-2.0]),
        Er_list=jnp.asarray([[1.0]] * 5),
        D11_log=jnp.zeros((5, 1, 1)),
        D13=jnp.zeros((5, 1, 1)),
        D33=jnp.zeros((5, 1, 1)),
    )
    geometry = types.SimpleNamespace(a_b=jnp.asarray(2.0))
    model = NTXDatabaseTransportModel("species", "grid", geometry, database)
    runtime = RuntimeContext(
        species="species",
        energy_grid="grid",
        geometry=geometry,
        database=database,
        solver_parameters={},
        models=Models(flux=model),
    )

    actual = realtime_geometry_payload_for_runtime(runtime)

    assert actual["kind"] == "ntx_database"
    assert actual["geometry"] is geometry
    assert actual["database"] is database
    assert "ntx_support" not in actual


def test_runtime_with_tagged_database_payload_replaces_geometry_and_database():
    database = Monoenergetic(
        a_b=jnp.asarray(2.0),
        rho=jnp.asarray([0.1, 0.3, 0.6, 0.9, 1.0]),
        nu_log=jnp.asarray([-2.0]),
        Er_list=jnp.asarray([[1.0]] * 5),
        D11_log=jnp.zeros((5, 1, 1)),
        D13=jnp.zeros((5, 1, 1)),
        D33=jnp.zeros((5, 1, 1)),
    )
    old_geometry = types.SimpleNamespace(a_b=jnp.asarray(2.0))
    new_geometry = types.SimpleNamespace(a_b=jnp.asarray(4.0))
    model = NTXDatabaseTransportModel("species", "grid", old_geometry, database)
    runtime = RuntimeContext(
        species="species", energy_grid="grid", geometry=old_geometry,
        database=database, solver_parameters={}, models=Models(flux=model),
    )
    new_database = database_with_geometry_scale(database, new_geometry.a_b)

    actual = runtime_with_realtime_geometry_payload(
        runtime,
        {"kind": "ntx_database", "geometry": new_geometry, "database": new_database},
    )

    assert actual.geometry is new_geometry
    assert actual.database is new_database
    assert actual.models.flux.geometry is new_geometry
    assert actual.models.flux.database is new_database


def test_monoenergetic_interpolation_kernel_defaults_to_generic():
    assert monoenergetic_interpolation_kernel(object()) is get_Dij


def test_load_ntss_reference_profiles_interpolates_scalar_and_species_profiles(tmp_path, monkeypatch):
    path = tmp_path / "profiles.h5"
    with h5py.File(path, "w") as f:
        f["r"] = jnp.array([0.0, 0.5, 1.0])
        f["Er"] = jnp.array([0.0, 1.0, 2.0])
        f["ne"] = jnp.array([10.0, 20.0, 30.0])
        f["nD"] = jnp.array([1.0, 2.0, 3.0])
        f["Te"] = jnp.array([100.0, 200.0, 300.0])
        f["TD"] = jnp.array([400.0, 500.0, 600.0])
        f["Tt"] = jnp.array([700.0, 800.0, 900.0])
        f["Vr"] = jnp.ones(3)
        f["FluxQe"] = jnp.array([7.0, 8.0, 9.0])
        f["FluxQI"] = jnp.array([4.0, 5.0, 6.0])

    monkeypatch.chdir(tmp_path)
    rho = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
    out = _load_ntss_reference_profiles("profiles.h5", rho)

    assert jnp.allclose(out["Er"], jnp.array([0.0, 0.5, 1.0, 1.5, 2.0]))
    assert jnp.allclose(out["density"]["e"], jnp.array([10.0, 15.0, 20.0, 25.0, 30.0]))
    assert jnp.allclose(out["density"]["D"], jnp.array([1.0, 1.5, 2.0, 2.5, 3.0]))
    assert jnp.allclose(out["density"]["T"], jnp.array([1.0, 1.5, 2.0, 2.5, 3.0]))
    assert jnp.allclose(out["temperature"]["e"], jnp.array([100.0, 150.0, 200.0, 250.0, 300.0]))
    assert jnp.allclose(out["temperature"]["D"], jnp.array([400.0, 450.0, 500.0, 550.0, 600.0]))
    assert jnp.allclose(out["temperature"]["T"], jnp.array([700.0, 750.0, 800.0, 850.0, 900.0]))
    assert jnp.allclose(out["flux_species"]["Q_total"]["e"], jnp.array([7.0, 7.5, 8.0, 8.5, 9.0]))


def test_load_user_extensions_imports_python_modules(monkeypatch):
    imported = []

    def fake_import_module(name):
        imported.append(name)
        return types.SimpleNamespace(__name__=name)

    monkeypatch.setattr("NEOPAX._orchestrator.importlib.import_module", fake_import_module)
    _load_user_extensions({"extensions": {"python_modules": ["pkg.a", "pkg.b"]}})
    assert imported == ["pkg.a", "pkg.b"]


def test_load_user_extensions_imports_python_files_relative_to_config_dir(tmp_path):
    mod_path = tmp_path / "user_models.py"
    mod_path.write_text("MARKER = 1\n", encoding="utf-8")
    _load_user_extensions(
        {
            "_config_dir": str(tmp_path),
            "extensions": {"python_files": ["user_models.py"]},
        }
    )


def test_load_user_extensions_registers_custom_models_from_python_file(tmp_path):
    mod_path = tmp_path / "user_models.py"
    mod_path.write_text(
        "\n".join(
            [
                "import dataclasses",
                "import jax.numpy as jnp",
                "import NEOPAX",
                "",
                "@dataclasses.dataclass(frozen=True, eq=False)",
                "class FileFluxModel:",
                "    def __call__(self, state, geometry=None, params=None):",
                "        del geometry, params",
                "        base = jnp.ones_like(state.density)",
                "        return {'Gamma': base, 'Q': 2.0 * base, 'Upar': jnp.zeros_like(base)}",
                "",
                "@dataclasses.dataclass(frozen=True, eq=False)",
                "class FileSourceModel:",
                "    def __call__(self, state):",
                "        return {'pressure_source': jnp.ones_like(state.pressure)}",
                "",
                "NEOPAX.register_transport_flux_model('file_registered_flux', FileFluxModel)",
                "NEOPAX.register_source_model('file_registered_source', FileSourceModel)",
            ]
        ),
        encoding="utf-8",
    )
    _load_user_extensions(
        {
            "_config_dir": str(tmp_path),
            "extensions": {"python_files": ["user_models.py"]},
        }
    )

    flux_builder = get_transport_flux_model("file_registered_flux")
    source_builder = get_source_model("file_registered_source")
    assert flux_builder is not None
    assert source_builder is not None


def test_load_user_extensions_registers_custom_models_from_python_module(tmp_path, monkeypatch):
    pkg_dir = tmp_path / "userpkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text(
        "\n".join(
            [
                "import dataclasses",
                "import jax.numpy as jnp",
                "import NEOPAX",
                "",
                "@dataclasses.dataclass(frozen=True, eq=False)",
                "class ModuleFluxModel:",
                "    def __call__(self, state, geometry=None, params=None):",
                "        del geometry, params",
                "        base = jnp.ones_like(state.density)",
                "        return {'Gamma': base, 'Q': 3.0 * base, 'Upar': jnp.zeros_like(base)}",
                "",
                "@dataclasses.dataclass(frozen=True, eq=False)",
                "class ModuleSourceModel:",
                "    def __call__(self, state):",
                "        return {'pressure_source': 2.0 * jnp.ones_like(state.pressure)}",
                "",
                "NEOPAX.register_transport_flux_model('module_registered_flux', ModuleFluxModel)",
                "NEOPAX.register_source_model('module_registered_source', ModuleSourceModel)",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    _load_user_extensions({"extensions": {"python_modules": ["userpkg"]}})

    flux_builder = get_transport_flux_model("module_registered_flux")
    source_builder = get_source_model("module_registered_source")
    assert flux_builder is not None
    assert source_builder is not None


def test_build_flux_model_passes_runtime_ntx_scan_inputs(monkeypatch):
    captured = {}

    def fake_get_transport_flux_model(name):
        def factory(*args, **kwargs):
            captured[name] = {"args": args, "kwargs": kwargs}
            return f"{name}_instance"

        return factory

    monkeypatch.setattr("NEOPAX._orchestrator.get_transport_flux_model", fake_get_transport_flux_model)
    monkeypatch.setattr(
        "NEOPAX._orchestrator.build_transport_flux_model",
        lambda neo, turb, classical, include_turbulent_particle_flux=True, **kwargs: {
            "neo": neo,
            "turb": turb,
            "classical": classical,
            "include_turbulent_particle_flux": include_turbulent_particle_flux,
            **kwargs,
        },
    )

    out = _build_flux_model(
        {
            "geometry": {
                "vmec_file": "wout.nc",
                "boozer_file": "boozmn.nc",
            },
            "neoclassical": {
                "flux_model": "ntx_scan_runtime",
                "ntx_scan_rho": [0.25, 0.5],
                "ntx_scan_nu_v": [1.0e-4, 1.0e-3],
                "ntx_scan_er_tilde": [0.0, 1.0e-4],
            },
            "turbulence": {"flux_model": "none"},
            "classical": {"flux_model": "none"},
        },
        species="species",
        energy_grid="grid",
        geometry="geometry",
        database="db",
        source_models=None,
    )

    assert out["neo"] == "ntx_scan_runtime_instance"
    assert captured["ntx_scan_runtime"]["kwargs"]["vmec_file"] == "wout.nc"
    assert captured["ntx_scan_runtime"]["kwargs"]["boozer_file"] == "boozmn.nc"
    assert captured["ntx_scan_runtime"]["kwargs"]["ntx_scan_rho"] == [0.25, 0.5]
    assert out["center_flux_mode"] == "direct"


@pytest.mark.parametrize(
    ("center_flux_mode", "expected_exact_mode"),
    [
        ("direct", "center_local_response"),
        ("interpolate_from_faces", "interpolate_from_faces"),
    ],
)
def test_build_flux_model_passes_runtime_ntx_exact_lij_inputs(
    monkeypatch,
    center_flux_mode,
    expected_exact_mode,
):
    captured = {}

    def fake_get_transport_flux_model(name):
        def factory(*args, **kwargs):
            captured[name] = {"args": args, "kwargs": kwargs}
            return f"{name}_instance"

        return factory

    monkeypatch.setattr("NEOPAX._orchestrator.get_transport_flux_model", fake_get_transport_flux_model)
    monkeypatch.setattr(
        "NEOPAX._orchestrator.build_transport_flux_model",
        lambda neo, turb, classical, include_turbulent_particle_flux=True, **kwargs: {
            "neo": neo,
            "turb": turb,
            "classical": classical,
            "include_turbulent_particle_flux": include_turbulent_particle_flux,
            **kwargs,
        },
    )

    out = _build_flux_model(
        {
            "geometry": {
                "vmec_file": "wout.nc",
                "boozer_file": "boozmn.nc",
            },
            "neoclassical": {
                "flux_model": "ntx_exact_lij_runtime",
                "ntx_exact_n_theta": 19,
                "ntx_exact_n_zeta": 21,
                "ntx_exact_n_xi": 48,
            },
            "transport_solver": {
                "density_floor": 2.5e-6,
                "temperature_floor": 7.5e-6,
                "center_flux_mode": center_flux_mode,
            },
            "turbulence": {"flux_model": "none"},
            "classical": {"flux_model": "none"},
        },
        species="species",
        energy_grid="grid",
        geometry="geometry",
        database="db",
        source_models=None,
    )

    assert out["neo"] == "ntx_exact_lij_runtime_instance"
    assert captured["ntx_exact_lij_runtime"]["kwargs"]["vmec_file"] == "wout.nc"
    assert captured["ntx_exact_lij_runtime"]["kwargs"]["boozer_file"] == "boozmn.nc"
    assert captured["ntx_exact_lij_runtime"]["kwargs"]["ntx_exact_n_theta"] == 19
    assert captured["ntx_exact_lij_runtime"]["kwargs"]["density_floor"] == 2.5e-6
    assert captured["ntx_exact_lij_runtime"]["kwargs"]["temperature_floor"] == 7.5e-6
    assert (
        captured["ntx_exact_lij_runtime"]["kwargs"]["ntx_exact_center_response_mode"]
        == expected_exact_mode
    )
    assert out["center_flux_mode"] == center_flux_mode


def test_build_ntx_runtime_scan_transport_model_can_skip_prebuild():
    model = build_ntx_runtime_scan_transport_model(
        species="species",
        energy_grid="grid",
        geometry="geometry",
        vmec_file="wout.nc",
        boozer_file="boozmn.nc",
        ntx_scan_rho=[0.25, 0.5],
        ntx_scan_nu_v=[1.0e-4, 1.0e-3],
        ntx_scan_er_tilde=[0.0, 1.0e-4],
        prebuild_database=False,
    )

    assert isinstance(model, NTXRuntimeScanTransportModel)
    assert model.database is None
    assert model.vmec_file == "wout.nc"
    assert model.boozer_file == "boozmn.nc"


def test_ntx_runtime_scan_model_accepts_live_scan_inputs_without_files():
    """Realtime VMEC may supply explicit scan surfaces instead of files."""
    surfaces = (object(), object())
    model = build_ntx_runtime_scan_transport_model(
        species="species",
        energy_grid="grid",
        geometry="geometry",
        vmec_file=None,
        boozer_file=None,
        ntx_scan_rho=[0.25, 0.5],
        ntx_scan_nu_v=[1.0e-4, 1.0e-3],
        ntx_scan_er_tilde=[0.0, 1.0e-4],
        ntx_scan_channels=_tiny_ntx_runtime_channels([0.25, 0.5]),
        ntx_scan_surfaces=surfaces,
        prebuild_database=False,
    )

    assert model.vmec_file is None
    assert model.boozer_file is None
    assert model._scan_surfaces(None, jnp.asarray([0.25, 0.5])) == surfaces


def test_ntx_runtime_scan_payload_replacement_clears_only_stale_database():
    """A realtime scan replacement retains live inputs and never file loaders."""
    old_surfaces = (object(), object())
    new_surfaces = (object(), object())
    old_channels = _tiny_ntx_runtime_channels([0.25, 0.5])
    new_channels = _tiny_ntx_runtime_channels([0.25, 0.5])
    model = build_ntx_runtime_scan_transport_model(
        species="species",
        energy_grid="grid",
        geometry="old_geometry",
        vmec_file=None,
        boozer_file=None,
        ntx_scan_rho=[0.25, 0.5],
        ntx_scan_nu_v=[1.0e-4, 1.0e-3],
        ntx_scan_er_tilde=[0.0, 1.0e-4],
        ntx_scan_channels=old_channels,
        ntx_scan_surfaces=old_surfaces,
        prebuild_database=False,
    )
    stale_database = object()
    model = dataclasses.replace(model, database=stale_database)

    actual = model.with_runtime_scan_payload(
        geometry="new_geometry",
        channels=new_channels,
        scan_surfaces=new_surfaces,
    )

    assert actual.geometry == "new_geometry"
    assert actual.channels is new_channels
    assert actual.scan_surfaces == new_surfaces
    assert actual.database is None
    assert actual.vmec_file is None
    assert actual.boozer_file is None


def test_tagged_realtime_payload_round_trips_live_ntx_scan_model():
    """The tagged reverse seam keeps live NTX scan inputs, not a static DB."""
    surfaces = (object(), object())
    channels = _tiny_ntx_runtime_channels([0.25, 0.5])
    model = build_ntx_runtime_scan_transport_model(
        species="species", energy_grid="grid", geometry="old_geometry",
        vmec_file=None, boozer_file=None,
        ntx_scan_rho=[0.25, 0.5], ntx_scan_nu_v=[1.0e-4, 1.0e-3],
        ntx_scan_er_tilde=[0.0, 1.0e-4], ntx_scan_channels=channels,
        ntx_scan_surfaces=surfaces, prebuild_database=False,
    )
    database = object()
    model = dataclasses.replace(model, database=database)
    runtime = RuntimeContext(
        species="species", energy_grid="grid", geometry="old_geometry",
        database=database, solver_parameters={}, models=Models(flux=model),
    )

    payload = realtime_geometry_payload_for_runtime(runtime)
    assert payload["kind"] == "ntx_scan_runtime"
    assert payload["channels"] is channels
    assert payload["surfaces"] == surfaces
    assert payload["database"] is database
    support_payload = realtime_geometry_reverse_support_payload_for_runtime(runtime)
    assert set(support_payload) == {"geometry", "channels", "surfaces"}
    assert support_payload["channels"] is channels
    assert support_payload["surfaces"] == surfaces

    new_surfaces = (object(), object())
    new_database = object()
    actual = runtime_with_realtime_geometry_payload(
        runtime,
        {
            **payload,
            "geometry": "new_geometry",
            "surfaces": new_surfaces,
            "database": new_database,
        },
    )
    assert actual.geometry == "new_geometry"
    assert actual.database is new_database
    assert actual.models.flux.channels is channels
    assert actual.models.flux.scan_surfaces == new_surfaces
    assert actual.models.flux.database is new_database


def test_live_ntx_scan_reverse_support_replacement_clears_cached_database():
    """The reverse payload cannot accidentally treat the scan cache as input."""

    surfaces = (object(), object())
    channels = _tiny_ntx_runtime_channels([0.25, 0.5])
    model = build_ntx_runtime_scan_transport_model(
        species="species", energy_grid="grid", geometry="old_geometry",
        vmec_file=None, boozer_file=None,
        ntx_scan_rho=[0.25, 0.5], ntx_scan_nu_v=[1.0e-4, 1.0e-3],
        ntx_scan_er_tilde=[0.0, 1.0e-4], ntx_scan_channels=channels,
        ntx_scan_surfaces=surfaces, prebuild_database=False,
    )
    cached_database = object()
    runtime = RuntimeContext(
        species="species", energy_grid="grid", geometry="old_geometry",
        database=cached_database, solver_parameters={},
        models=Models(flux=dataclasses.replace(model, database=cached_database)),
    )

    actual = runtime_with_realtime_geometry_reverse_support_payload(
        runtime,
        {"geometry": "new_geometry", "channels": channels, "surfaces": surfaces},
    )

    assert actual.geometry == "new_geometry"
    assert actual.models.flux.database is None
    assert actual.database is None


def test_reverse_setup_selects_live_scan_payload_without_exact_support_lookup():
    """The segment probe uses the combined live scan tree without exact lookup."""

    surfaces = (object(), object())
    channels = _tiny_ntx_runtime_channels([0.25, 0.5])
    model = build_ntx_runtime_scan_transport_model(
        species="species", energy_grid="grid", geometry="geometry",
        vmec_file=None, boozer_file=None,
        ntx_scan_rho=[0.25, 0.5], ntx_scan_nu_v=[1.0e-4, 1.0e-3],
        ntx_scan_er_tilde=[0.0, 1.0e-4], ntx_scan_channels=channels,
        ntx_scan_surfaces=surfaces, prebuild_database=False,
    )
    runtime = RuntimeContext(
        species="species", energy_grid="grid", geometry="geometry",
        database=None, solver_parameters={}, models=Models(flux=model),
    )
    args = types.SimpleNamespace(
        realtime_geometry_gradient_path="support_segment_probe",
        reverse_stage_cotangent_mode="full",
        initial_er_root_ad="off",
        accepted_step_limit=None,
        reverse_segment_length=1,
        reverse_stage_adjoint_solve_mode="block",
        reverse_rhs_transpose_mode="generic",
        reverse_step_bwd_mode="reduced_cotangent_call_boundary",
        reverse_stage_adjoint_memory_mode="legacy",
        reverse_stage_adjoint_iter_maxiter=1,
        reverse_stage_adjoint_iter_tol=1.0e-8,
        reverse_rebuild_support_pullback_mode="separate",
        reverse_initial_cache_support_pullback_mode="scalar",
        reverse_segment_input_diagnostics=True,
        reverse_segment_start_replay_mode="minimal",
        reverse_segment_primal_record_mode="reuse_segment_primal_record",
        reverse_final_objective_cotangent_mode="scalar",
        reverse_bootstrap_cotangent_mode="joint_local_vjp_upar_only",
    )
    captured = {}

    def _unexpected_exact_lookup(_runtime):
        raise AssertionError("scan setup must not request an exact NTX support payload")

    def _prepare(_profile_values, **kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(schedule_artifact=None)

    setup = prepare_realtime_geometry_support_segment_core_setup(
        args=args, config={}, baseline_values=jnp.asarray([]), baseline_runtime=runtime,
        baseline_state="state", profile_cfg={}, neoclassical_cfg={}, parameter_order=(),
        find_ntx_support_payload=_unexpected_exact_lookup,
        prepare_reverse_static_setup=_prepare,
    )

    assert setup.payload_kind == "ntx_scan_runtime"
    assert set(setup.support_payload) == {"geometry", "channels", "surfaces"}
    assert captured["runtime"] is runtime
    assert captured["reverse_initial_cache_support_pullback_mode"] == "scalar"
    assert captured["reverse_rebuild_support_pullback_mode"] == "separate"
    assert captured["reverse_segment_start_replay_mode"] == "minimal"
    assert captured["reverse_segment_primal_record_mode"] == "reuse_segment_primal_record"
    assert captured["reverse_bootstrap_cotangent_mode"] == "joint_local_vjp_upar_only"


def test_payload_transpose_forwards_live_scan_contract(monkeypatch):
    """The outer transport wrapper forwards scan metadata to the VMEC seam."""

    captured = {}

    def _fake_transpose(*_args, **kwargs):
        captured.update(kwargs)
        return jnp.asarray([[1.0]])

    monkeypatch.setattr(
        "NEOPAX._reverse_ad_transport.geometry_payload_pullback_from_param_vector_raw_block_transpose",
        _fake_transpose,
    )
    result = realtime_geometry_payload_pullback_result(
        geometry_context="context",
        baseline_geometry_deltas=jnp.asarray([0.0]),
        geometry_param_specs=(("RBC", 1, 0),),
        support_bars=({"geometry": "g", "ntx_scan_runtime": "scan"},),
        payload_kind="ntx_scan_runtime",
        scan_rho=(0.25, 0.5),
        scan_surface_backend="vmec",
    )

    assert captured["payload_kind"] == "ntx_scan_runtime"
    assert captured["scan_rho"] == (0.25, 0.5)
    assert captured["scan_surface_backend"] == "vmec"
    assert jnp.allclose(result.geometry_gradient_matrix, jnp.asarray([[1.0]]))


def test_live_ntx_scan_payload_rebuild_keeps_channel_jvp(monkeypatch):
    """A support payload regenerates the database through the live NTX seam."""

    captured_builder_kwargs = {}

    @dataclasses.dataclass(frozen=True)
    class _FakeScan:
        rho: object
        nu_v: object
        Er: object
        drds: object
        D11: object
        D13: object
        D33: object
        Er_tilde: object = None
        Er_to_Ertilde: object = None
        dr_tildedr: object = None
        dr_tildeds: object = None
        a_b: object = None
        psia: object = None
        b00: object = None
        r00: object = None
        boozer_i: object = None
        boozer_g: object = None
        iota: object = None
        fac_reference_to_sfincs_11: object = None
        fac_reference_to_sfincs_31: object = None
        fac_reference_to_sfincs_33: object = None
        fac_monkes_to_sfincs_11: object = None
        fac_monkes_to_sfincs_31: object = None
        fac_monkes_to_sfincs_33: object = None
        fac_sfincs_to_dkes_11: object = None
        fac_sfincs_to_dkes_31: object = None
        fac_sfincs_to_dkes_33: object = None
        fac_dkes_to_d11star: object = None
        fac_dkes_to_d31star: object = None
        fac_dkes_to_d33star: object = None

    class _FakeNTX:
        class GridSpec:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        @staticmethod
        def build_ntx_neopax_scan_from_surfaces(surfaces, *, rho, nu_v, Er, drds, **kwargs):
            captured_builder_kwargs.update(kwargs)
            assert len(surfaces) == int(rho.shape[0])
            shape = (rho.shape[0], nu_v.shape[0], Er.shape[1])
            return _FakeScan(
                rho=rho,
                nu_v=nu_v,
                Er=Er,
                drds=drds,
                D11=jnp.broadcast_to(drds[:, None, None], shape),
                D13=jnp.broadcast_to(2.0 * drds[:, None, None], shape),
                D33=jnp.ones(shape),
            )

    monkeypatch.setattr("NEOPAX._transport_flux_models._import_ntx", lambda: _FakeNTX)
    surfaces = (object(), object())
    channels = _tiny_ntx_runtime_channels([0.25, 0.5])
    model = build_ntx_runtime_scan_transport_model(
        species="species", energy_grid="grid", geometry="geometry",
        vmec_file=None, boozer_file=None,
        ntx_scan_rho=[0.25, 0.5], ntx_scan_nu_v=[1.0e-4, 1.0e-3],
        ntx_scan_er_tilde=[0.0, 1.0e-4], ntx_scan_channels=channels,
        ntx_scan_surfaces=surfaces, prebuild_database=False,
    )

    def _database_from_ab(a_b):
        updated_channels = dataclasses.replace(channels, a_b=a_b)
        return model.with_support_payload(
            {"geometry": "geometry", "channels": updated_channels, "surfaces": surfaces}
        ).database

    database, tangent = jax.jvp(
        lambda a_b: _database_from_ab(a_b).Er_list,
        (jnp.asarray(2.0),),
        (jnp.asarray(0.2),),
    )
    assert jnp.all(jnp.isfinite(database))
    # The zero ``Er_tilde`` column is clamped at the database's log-space
    # floor, so its derivative is correctly zero.  The non-clamped column
    # retains the analytic ``-da_b / (a_b log(10))`` radius chain.
    expected_tangent = jnp.where(
        jnp.asarray([0.0, 1.0e-4])[None, :] != 0.0,
        -0.2 / (2.0 * jnp.log(10.0)),
        0.0,
    )
    assert jnp.allclose(tangent, expected_tangent)
    assert captured_builder_kwargs["coefficient_reverse_mode"] == "generic"


def test_live_ntx_scan_explicit_database_support_does_not_rebuild(monkeypatch):
    """The recorded reverse support leaf must bypass the NTX scan builder."""
    calls = {"count": 0}

    class _FakeNTX:
        class GridSpec:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        @staticmethod
        def build_ntx_neopax_scan_from_surfaces(*_args, **_kwargs):
            calls["count"] += 1
            raise AssertionError("explicit recorded database support must not rebuild the scan")

    monkeypatch.setattr("NEOPAX._transport_flux_models._import_ntx", lambda: _FakeNTX)
    surfaces = (object(), object())
    channels = _tiny_ntx_runtime_channels([0.25, 0.5])
    database = object()
    model = build_ntx_runtime_scan_transport_model(
        species="species", energy_grid="grid", geometry="geometry",
        vmec_file=None, boozer_file=None,
        ntx_scan_rho=[0.25, 0.5], ntx_scan_nu_v=[1.0e-4, 1.0e-3],
        ntx_scan_er_tilde=[0.0, 1.0e-4], ntx_scan_channels=channels,
        ntx_scan_surfaces=surfaces, prebuild_database=False,
    )

    result = model.with_support_payload(
        {"geometry": "geometry", "channels": channels, "surfaces": surfaces, "database": database}
    )
    assert result.database is database
    # This is the path used by the direct black-box RHS and its compact
    # pullbacks.  It must preserve the explicit database rather than invoking
    # the runtime NTX scan builder a second time.
    assert result._database_model().database is database
    assert calls == {"count": 0}


def test_recorded_ntx_database_bar_is_folded_once_into_scan_support(monkeypatch):
    calls = {"count": 0}

    class _RecordedScan:
        def recorded_runtime_database_support_bar(self, database_bar):
            calls["count"] += 1
            assert database_bar == jnp.asarray(5.0)
            return {
                "channels": jnp.asarray(2.0),
                "surfaces": jnp.asarray(3.0),
            }

    monkeypatch.setattr(
        initial_er_module,
        "find_ntx_runtime_scan_model_in_model",
        lambda _flux: _RecordedScan(),
    )
    runtime = types.SimpleNamespace(models=types.SimpleNamespace(flux=object()))
    actual = fold_recorded_ntx_scan_database_bar_into_support(
        runtime,
        {
            "geometry": jnp.asarray(7.0),
            "channels": jnp.asarray(11.0),
            "surfaces": jnp.asarray(13.0),
            "database": jnp.asarray(5.0),
        },
    )
    assert set(actual) == {"geometry", "channels", "surfaces"}
    assert jnp.allclose(actual["channels"], 13.0)
    assert jnp.allclose(actual["surfaces"], 16.0)
    assert calls == {"count": 1}


def test_recorded_ntx_database_bars_use_one_batched_scan_pullback(monkeypatch):
    calls = {"count": 0}

    class _RecordedScan:
        def recorded_runtime_database_support_bar(self, database_bar):
            calls["count"] += 1
            return {
                "channels": 2.0 * database_bar,
                "surfaces": 3.0 * database_bar,
            }

    monkeypatch.setattr(
        initial_er_module,
        "find_ntx_runtime_scan_model_in_model",
        lambda _flux: _RecordedScan(),
    )
    runtime = types.SimpleNamespace(models=types.SimpleNamespace(flux=object()))
    actual = fold_recorded_ntx_scan_database_bars_into_support(
        runtime,
        (
            {"geometry": jnp.asarray(7.0), "channels": jnp.asarray(11.0), "surfaces": jnp.asarray(13.0), "database": jnp.asarray(5.0)},
            {"geometry": jnp.asarray(17.0), "channels": jnp.asarray(19.0), "surfaces": jnp.asarray(23.0), "database": jnp.asarray(29.0)},
        ),
    )
    assert len(actual) == 2
    assert jnp.allclose(actual[0]["channels"], 21.0)
    assert jnp.allclose(actual[0]["surfaces"], 28.0)
    assert jnp.allclose(actual[1]["channels"], 77.0)
    assert jnp.allclose(actual[1]["surfaces"], 110.0)
    # ``vmap`` traces the retained transpose once instead of Python-looping
    # through the two objective rows.
    assert calls == {"count": 1}


def test_recorded_live_ntx_scan_support_exposes_only_the_existing_database():
    """The recorded route is opt-in and never asks the support VJP to rebuild."""

    surfaces = (object(), object())
    channels = _tiny_ntx_runtime_channels([0.25, 0.5])
    database = object()
    model = build_ntx_runtime_scan_transport_model(
        species="species", energy_grid="grid", geometry="geometry",
        vmec_file=None, boozer_file=None,
        ntx_scan_rho=[0.25, 0.5], ntx_scan_nu_v=[1.0e-4, 1.0e-3],
        ntx_scan_er_tilde=[0.0, 1.0e-4], ntx_scan_channels=channels,
        ntx_scan_surfaces=surfaces,
        ntx_scan_coefficient_reverse_mode="structured",
        ntx_scan_record_primal=True,
        prebuild_database=False,
    )
    # The actual record is opaque to this payload-seam test; it simply marks
    # that the forward scan recorded the retained prepared primal.
    model = dataclasses.replace(
        model, database=database, scan_primal_record=object(), scan_primal=object()
    )
    runtime = RuntimeContext(
        species="species", energy_grid="grid", geometry="geometry",
        database=database, solver_parameters={}, models=Models(flux=model),
    )

    support = realtime_geometry_reverse_support_payload_for_runtime(runtime)

    assert set(support) == {"geometry", "channels", "surfaces", "database"}
    assert support["database"] is database
    assert support["channels"] is channels
    assert support["surfaces"] is surfaces


def test_recorded_live_ntx_scan_primal_is_not_captured_by_reverse_runtime():
    """The heavy prepared record is retained only for the post-sweep fold."""

    channels = _tiny_ntx_runtime_channels([0.25, 0.5])
    database = object()
    model = build_ntx_runtime_scan_transport_model(
        species="species", energy_grid="grid", geometry="geometry",
        vmec_file=None, boozer_file=None,
        ntx_scan_rho=[0.25, 0.5], ntx_scan_nu_v=[1.0e-4, 1.0e-3],
        ntx_scan_er_tilde=[0.0, 1.0e-4], ntx_scan_channels=channels,
        ntx_scan_surfaces=(object(), object()),
        ntx_scan_coefficient_reverse_mode="structured",
        ntx_scan_record_primal=True,
        prebuild_database=False,
    )
    record = object()
    raw_scan = object()
    runtime = RuntimeContext(
        species="species", energy_grid="grid", geometry="geometry",
        database=database, solver_parameters={},
        models=Models(flux=dataclasses.replace(
            model, database=database, scan_primal_record=record, scan_primal=raw_scan
        )),
    )

    stripped = runtime_without_recorded_ntx_scan_primal(runtime)
    stripped_model = stripped.models.flux
    assert stripped_model.database is database
    assert stripped_model.channels is channels
    assert stripped_model.scan_surfaces == runtime.models.flux.scan_surfaces
    assert stripped_model.scan_primal_record is None
    assert stripped_model.scan_primal is None
    # The original remains available to execute the single final transpose.
    assert runtime.models.flux.scan_primal_record is record
    assert runtime.models.flux.scan_primal is raw_scan


def test_live_ntx_scan_payload_can_select_structured_coefficient_reverse_mode():
    """The scan builder mode is a narrow opt-in, with generic kept as default."""

    model = build_ntx_runtime_scan_transport_model(
        species="species",
        energy_grid="grid",
        geometry="geometry",
        vmec_file=None,
        boozer_file=None,
        ntx_scan_rho=[0.25, 0.5],
        ntx_scan_nu_v=[1.0e-4, 1.0e-3],
        ntx_scan_er_tilde=[0.0, 1.0e-4],
        ntx_scan_coefficient_reverse_mode="structured",
        prebuild_database=False,
    )
    assert model.coefficient_reverse_mode == "structured"


def test_ntx_runtime_scan_database_keeps_radius_local_er_axis():
    scan = types.SimpleNamespace(
        rho=jnp.asarray([0.25, 0.5]),
        nu_v=jnp.asarray([1.0e-4, 1.0e-3]),
        Er=jnp.asarray(
            [
                [1.0e-6, 2.0e-6],
                [3.0e-6, 6.0e-6],
            ]
        ),
        drds=jnp.asarray([2.0, 4.0]),
        D11=jnp.ones((2, 2, 2)),
        D13=2.0 * jnp.ones((2, 2, 2)),
        D33=3.0 * jnp.ones((2, 2, 2)),
    )

    database = _ntx_runtime_scan_to_neopax_monoenergetic(scan, a_b=2.0)

    expected_er_list = jnp.log10(jnp.maximum(1.0e-8, jnp.abs(scan.Er) / (2.0 * scan.rho[:, None])))
    assert jnp.allclose(database.Er_list, expected_er_list)
    assert not jnp.allclose(database.Er_list[1], database.Er_list[0] + jnp.log10(scan.rho[0] / scan.rho[1]))
    assert jnp.allclose(10.0 ** database.D11_log, scan.D11 * scan.drds[:, None, None] ** 2)
    assert jnp.allclose(database.D13, scan.D13 * scan.drds[:, None, None])
    assert jnp.allclose(database.D33, scan.D33 * scan.nu_v[None, :, None])


def test_build_ntx_exact_lij_runtime_transport_model_can_skip_preload():
    model = build_ntx_exact_lij_runtime_transport_model(
        species="species",
        energy_grid="grid",
        geometry="geometry",
        vmec_file="wout.nc",
        boozer_file="boozmn.nc",
        preload_support=False,
    )

    assert isinstance(model, NTXExactLijRuntimeTransportModel)
    assert model.support is None
    assert model.vmec_file == "wout.nc"
    assert model.boozer_file == "boozmn.nc"
    assert model.lagged_response_taylor_order == 1


def test_build_ntx_exact_lij_runtime_transport_model_accepts_quadratic_feature_gate():
    model = build_ntx_exact_lij_runtime_transport_model(
        species="species",
        energy_grid="grid",
        geometry="geometry",
        vmec_file="wout.nc",
        boozer_file="boozmn.nc",
        lagged_response_taylor_order=2,
        preload_support=False,
    )

    assert model.lagged_response_taylor_order == 2
    with pytest.raises(ValueError, match="must be 1 or 2"):
        model.with_lagged_response_taylor_order(3)


def test_build_ntx_runtime_scan_channels_uses_loader(monkeypatch):
    monkeypatch.setattr(
        "NEOPAX._transport_flux_models._load_ntx_vmec_boozer_channels",
        lambda vmec_file, boozer_file, rho: {
            "a_b": 1.5,
            "psia": 2.5,
            "b00": rho + 1.0,
            "r00": rho + 2.0,
            "boozer_i": rho + 3.0,
            "boozer_g": rho + 4.0,
            "iota": rho + 5.0,
            "drds": rho + 6.0,
            "dr_tildedr": rho + 7.0,
            "dr_tildeds": rho + 8.0,
            "fac_reference_to_sfincs_11": rho + 9.0,
            "fac_reference_to_sfincs_31": rho + 10.0,
            "fac_reference_to_sfincs_33": rho + 11.0,
            "fac_sfincs_to_dkes_11": rho + 12.0,
            "fac_sfincs_to_dkes_31": rho + 13.0,
            "fac_sfincs_to_dkes_33": rho + 14.0,
            "fac_dkes_to_d11star": rho + 15.0,
            "fac_dkes_to_d31star": rho + 16.0,
            "fac_dkes_to_d33star": rho + 17.0,
        },
    )

    channels = build_ntx_runtime_scan_channels("wout.nc", "boozmn.nc", [0.25, 0.5])

    assert isinstance(channels, NTXRuntimeScanChannels)
    assert jnp.allclose(channels.rho, jnp.array([0.25, 0.5]))
    assert channels.a_b == 1.5
    assert jnp.allclose(channels.dr_tildeds, jnp.array([8.25, 8.5]))


def test_build_ntx_runtime_scan_transport_model_can_preload_channels(monkeypatch):
    sentinel = NTXRuntimeScanChannels(
        rho=jnp.array([0.25, 0.5]),
        a_b=1.0,
        psia=2.0,
        b00=jnp.array([1.0, 1.1]),
        r00=jnp.array([2.0, 2.1]),
        boozer_i=jnp.array([3.0, 3.1]),
        boozer_g=jnp.array([4.0, 4.1]),
        iota=jnp.array([5.0, 5.1]),
        drds=jnp.array([6.0, 6.1]),
        dr_tildedr=jnp.array([7.0, 7.1]),
        dr_tildeds=jnp.array([8.0, 8.1]),
        fac_reference_to_sfincs_11=jnp.array([9.0, 9.1]),
        fac_reference_to_sfincs_31=jnp.array([10.0, 10.1]),
        fac_reference_to_sfincs_33=jnp.array([11.0, 11.1]),
        fac_sfincs_to_dkes_11=jnp.array([12.0, 12.1]),
        fac_sfincs_to_dkes_31=jnp.array([13.0, 13.1]),
        fac_sfincs_to_dkes_33=jnp.array([14.0, 14.1]),
        fac_dkes_to_d11star=jnp.array([15.0, 15.1]),
        fac_dkes_to_d31star=jnp.array([16.0, 16.1]),
        fac_dkes_to_d33star=jnp.array([17.0, 17.1]),
    )
    monkeypatch.setattr(
        "NEOPAX._transport_flux_models.build_ntx_runtime_scan_channels",
        lambda vmec_file, boozer_file, rho_scan: sentinel,
    )

    model = build_ntx_runtime_scan_transport_model(
        species="species",
        energy_grid="grid",
        geometry="geometry",
        vmec_file="wout.nc",
        boozer_file="boozmn.nc",
        ntx_scan_rho=[0.25, 0.5],
        ntx_scan_nu_v=[1.0e-4, 1.0e-3],
        ntx_scan_er_tilde=[0.0, 1.0e-4],
        preload_channels=True,
        prebuild_database=False,
    )

    assert model.database is None
    assert model.channels is sentinel


def test_ntx_runtime_scan_transport_model_delegates_face_and_local_evaluators(monkeypatch):
    model = NTXRuntimeScanTransportModel(
        species="species",
        energy_grid="grid",
        geometry="geometry",
        vmec_file="wout.nc",
        boozer_file="boozmn.nc",
        rho_scan=[0.25],
        nu_v_scan=[1.0e-4],
        er_tilde_scan=[0.0],
        database=None,
    )
    calls = []

    monkeypatch.setattr(
        NTXRuntimeScanTransportModel,
        "_build_runtime_database",
        lambda self: "runtime_db",
    )

    def fake_build_local(self, state):
        calls.append(("local", self.database, state))
        return "local_eval"

    def fake_face(self, state, face_state, **kwargs):
        calls.append(("face", self.database, state, face_state, kwargs))
        return "face_eval"

    def fake_build_lagged(self, state, **kwargs):
        calls.append(("build_lagged", self.database, state, kwargs))
        return "face_lagged_response"

    def fake_eval_lagged(self, state, lagged_response, **kwargs):
        calls.append(("eval_lagged", self.database, state, lagged_response, kwargs))
        return "lagged_face_fluxes"

    def fake_pullback_lagged(self, state, lagged_response_bar, **kwargs):
        calls.append(("pullback_lagged", self.database, state, lagged_response_bar, kwargs))
        return "state_bar"

    monkeypatch.setattr(
        "NEOPAX._transport_flux_models.NTXDatabaseTransportModel.build_local_particle_flux_evaluator",
        fake_build_local,
    )
    monkeypatch.setattr(
        "NEOPAX._transport_flux_models.NTXDatabaseTransportModel.evaluate_face_fluxes",
        fake_face,
    )
    monkeypatch.setattr(
        "NEOPAX._transport_flux_models.NTXDatabaseTransportModel.build_lagged_response",
        fake_build_lagged,
    )
    monkeypatch.setattr(
        "NEOPAX._transport_flux_models.NTXDatabaseTransportModel.evaluate_with_lagged_response",
        fake_eval_lagged,
    )
    monkeypatch.setattr(
        "NEOPAX._transport_flux_models.NTXDatabaseTransportModel.pullback_build_lagged_response",
        fake_pullback_lagged,
    )

    assert model.build_local_particle_flux_evaluator("state") == "local_eval"
    assert model.evaluate_face_fluxes("state", "face_state", marker=True) == "face_eval"
    assert model.build_lagged_response("state", marker=True) == "face_lagged_response"
    assert model.evaluate_with_lagged_response("state", "response", marker=True) == "lagged_face_fluxes"
    assert model.pullback_build_lagged_response("state", "response_bar", marker=True) == "state_bar"
    assert calls[0] == ("local", "runtime_db", "state")
    assert calls[1] == ("face", "runtime_db", "state", "face_state", {"marker": True})
    assert calls[2] == ("build_lagged", "runtime_db", "state", {"marker": True})
    assert calls[3] == ("eval_lagged", "runtime_db", "state", "response", {"marker": True})
    assert calls[4] == ("pullback_lagged", "runtime_db", "state", "response_bar", {"marker": True})


def test_ntx_runtime_scan_transport_model_with_scan_inputs_preserves_channels_for_same_rho():
    channels = NTXRuntimeScanChannels(
        rho=jnp.array([0.25, 0.5]),
        a_b=1.0,
        psia=2.0,
        b00=jnp.array([1.0, 1.1]),
        r00=jnp.array([2.0, 2.1]),
        boozer_i=jnp.array([3.0, 3.1]),
        boozer_g=jnp.array([4.0, 4.1]),
        iota=jnp.array([5.0, 5.1]),
        drds=jnp.array([6.0, 6.1]),
        dr_tildedr=jnp.array([7.0, 7.1]),
        dr_tildeds=jnp.array([8.0, 8.1]),
        fac_reference_to_sfincs_11=jnp.array([9.0, 9.1]),
        fac_reference_to_sfincs_31=jnp.array([10.0, 10.1]),
        fac_reference_to_sfincs_33=jnp.array([11.0, 11.1]),
        fac_sfincs_to_dkes_11=jnp.array([12.0, 12.1]),
        fac_sfincs_to_dkes_31=jnp.array([13.0, 13.1]),
        fac_sfincs_to_dkes_33=jnp.array([14.0, 14.1]),
        fac_dkes_to_d11star=jnp.array([15.0, 15.1]),
        fac_dkes_to_d31star=jnp.array([16.0, 16.1]),
        fac_dkes_to_d33star=jnp.array([17.0, 17.1]),
    )
    model = NTXRuntimeScanTransportModel(
        species="species",
        energy_grid="grid",
        geometry="geometry",
        vmec_file="wout.nc",
        boozer_file="boozmn.nc",
        rho_scan=[0.25, 0.5],
        nu_v_scan=[1.0e-4, 1.0e-3],
        er_tilde_scan=[0.0, 1.0e-4],
        channels=channels,
        database="cached_db",
    )

    updated = model.with_scan_inputs(
        nu_v_scan=[2.0e-4, 2.0e-3],
        er_tilde_scan=[1.0e-5, 2.0e-4],
    )

    assert updated.channels is channels
    assert updated.database is None
    assert updated.nu_v_scan == [2.0e-4, 2.0e-3]
    assert updated.er_tilde_scan == [1.0e-5, 2.0e-4]


def test_ntx_runtime_scan_transport_model_with_scan_inputs_drops_channels_for_new_rho():
    channels = NTXRuntimeScanChannels(
        rho=jnp.array([0.25, 0.5]),
        a_b=1.0,
        psia=2.0,
        b00=jnp.array([1.0, 1.1]),
        r00=jnp.array([2.0, 2.1]),
        boozer_i=jnp.array([3.0, 3.1]),
        boozer_g=jnp.array([4.0, 4.1]),
        iota=jnp.array([5.0, 5.1]),
        drds=jnp.array([6.0, 6.1]),
        dr_tildedr=jnp.array([7.0, 7.1]),
        dr_tildeds=jnp.array([8.0, 8.1]),
        fac_reference_to_sfincs_11=jnp.array([9.0, 9.1]),
        fac_reference_to_sfincs_31=jnp.array([10.0, 10.1]),
        fac_reference_to_sfincs_33=jnp.array([11.0, 11.1]),
        fac_sfincs_to_dkes_11=jnp.array([12.0, 12.1]),
        fac_sfincs_to_dkes_31=jnp.array([13.0, 13.1]),
        fac_sfincs_to_dkes_33=jnp.array([14.0, 14.1]),
        fac_dkes_to_d11star=jnp.array([15.0, 15.1]),
        fac_dkes_to_d31star=jnp.array([16.0, 16.1]),
        fac_dkes_to_d33star=jnp.array([17.0, 17.1]),
    )
    model = NTXRuntimeScanTransportModel(
        species="species",
        energy_grid="grid",
        geometry="geometry",
        vmec_file="wout.nc",
        boozer_file="boozmn.nc",
        rho_scan=[0.25, 0.5],
        nu_v_scan=[1.0e-4, 1.0e-3],
        er_tilde_scan=[0.0, 1.0e-4],
        channels=channels,
        database="cached_db",
    )

    updated = model.with_scan_inputs(rho_scan=[0.2, 0.6])

    assert updated.channels is None
    assert updated.database is None
    assert updated.rho_scan == [0.2, 0.6]


def test_build_ntx_exact_lij_runtime_transport_model_can_preload_support(monkeypatch):
    monkeypatch.setattr(
        "NEOPAX._transport_flux_models.build_ntx_exact_lij_runtime_support",
        lambda *args, **kwargs: "sentinel_support",
    )

    model = build_ntx_exact_lij_runtime_transport_model(
        species="species",
        energy_grid="grid",
        geometry=types.SimpleNamespace(
            a_b=1.0,
            r_grid=jnp.array([0.25, 0.5]),
            r_grid_half=jnp.array([0.125, 0.375, 0.625]),
        ),
        vmec_file="wout.nc",
        boozer_file="boozmn.nc",
        preload_support=True,
    )

    assert model.support == "sentinel_support"


@pytest.mark.parametrize("radius", (0.01, 0.2, 0.55, 0.95))
def test_radial_preprocessed_stencil_reconstructs_established_interpolation(radius):
    """The compact stencil is exactly the production radial interpolation."""

    rho = jnp.asarray([0.1, 0.3, 0.5, 0.7, 0.9])
    nu_v = jnp.asarray([1.0e-3, 1.0e-2, 1.0e-1])
    er = jnp.asarray([[1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1]])
    shape = (rho.size, nu_v.size, er.shape[1])
    base = jnp.reshape(jnp.arange(int(jnp.prod(jnp.asarray(shape))), dtype=jnp.float64), shape)
    database = PreprocessedMonoenergetic3DNTSSRadius.read_data(
        a_b=1.0,
        rho=rho,
        nu_v=nu_v,
        Er=er,
        drds=jnp.ones_like(rho),
        D11=1.0 + base,
        D13=2.0 + base,
        D33=3.0 + base,
    )
    grid_nu = jnp.asarray(2.0e-2)
    grid_er = jnp.asarray(3.0e-3)
    stencil = radial_preprocessed_interpolation_stencil(
        jnp.asarray(radius), grid_nu, grid_er, database
    )

    def _reconstruct(table):
        def _one_surface(ir, ier, tz):
            return _bilinear(
                table[ir, stencil.nu_index, ier],
                table[ir, stencil.nu_index, ier + 1],
                table[ir, stencil.nu_index + 1, ier],
                table[ir, stencil.nu_index + 1, ier + 1],
                stencil.nu_fraction,
                tz,
            )

        values = jax.vmap(_one_surface)(
            stencil.radial_indices, stencil.er_indices, stencil.er_fractions
        )
        return jnp.sum(stencil.radial_weights * values)

    expected = get_Dij_preprocessed_3d_ntss_radius(
        jnp.asarray(radius), grid_nu, grid_er, database
    )
    actual = jnp.asarray(
        (
            _reconstruct(database.D11_log),
            _reconstruct(database.D13),
            _reconstruct(database.D33),
        )
    )
    assert jnp.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize("radius", (0.01, 0.55, 0.95))
def test_radial_preprocessed_stencil_table_transpose_matches_generic_vjp(radius):
    """The explicit 16-entry scatter is the established table VJP exactly."""

    rho = jnp.asarray([0.1, 0.3, 0.5, 0.7, 0.9])
    nu_v = jnp.asarray([1.0e-3, 1.0e-2, 1.0e-1])
    er = jnp.asarray([[1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1]])
    shape = (rho.size, nu_v.size, er.shape[1])
    base = jnp.reshape(jnp.arange(int(jnp.prod(jnp.asarray(shape))), dtype=jnp.float64), shape)
    database = PreprocessedMonoenergetic3DNTSSRadius.read_data(
        a_b=1.0, rho=rho, nu_v=nu_v, Er=er, drds=jnp.ones_like(rho),
        D11=1.0 + base, D13=2.0 + base, D33=3.0 + base,
    )
    grid_nu = jnp.asarray(2.0e-2)
    grid_er = jnp.asarray(3.0e-3)
    local_bar = jnp.asarray(-0.37)
    stencil = radial_preprocessed_interpolation_stencil(
        jnp.asarray(radius), grid_nu, grid_er, database
    )

    def _interpolate(table):
        return get_Dij_preprocessed_3d_ntss_radius(
            jnp.asarray(radius), grid_nu, grid_er,
            dataclasses.replace(database, D13=table),
        )[1]

    _, generic_pullback = jax.vjp(_interpolate, database.D13)
    expected = generic_pullback(local_bar)[0]
    actual = radial_preprocessed_interpolation_table_bar(
        stencil, local_bar, database.D13
    )
    assert jnp.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize("radius", (0.12, 0.52, 0.88))
def test_legacy_monoenergetic_table_transpose_matches_generic_vjp(radius):
    """The explicit C1 bicubic/radial Monoenergetic transpose is exact."""

    rho = jnp.asarray([0.1, 0.3, 0.5, 0.7, 0.9])
    nu_log = jnp.asarray([-3.0, -2.0, -1.0, 0.0])
    er_row = jnp.asarray([-5.0, -4.0, -3.0, -2.0])
    er_list = jnp.broadcast_to(er_row, (rho.size, er_row.size))
    shape = (rho.size, nu_log.size, er_row.size)
    base = jnp.reshape(jnp.arange(int(jnp.prod(jnp.asarray(shape))), dtype=jnp.float64), shape)
    database = Monoenergetic(
        a_b=1.0, rho=rho, nu_log=nu_log, Er_list=er_list,
        D11_log=-3.0 + 0.001 * base, D13=0.2 + 0.002 * base,
        D33=0.4 + 0.003 * base,
    )
    grid_nu = jnp.asarray(2.4e-2)
    grid_er = jnp.asarray(2.0e-4)
    local_bar = jnp.asarray(-0.37)

    def _interpolate(table):
        return get_Dij(
            jnp.asarray(radius), grid_nu, grid_er,
            dataclasses.replace(database, D13=table),
        )[1]

    _, generic_pullback = jax.vjp(_interpolate, database.D13)
    expected = generic_pullback(local_bar)[0]
    actual = monoenergetic_interpolation_table_bar(
        jnp.asarray(radius), grid_nu, grid_er, local_bar, database.D13, database
    )
    assert jnp.allclose(actual, expected, rtol=2.0e-11, atol=2.0e-12), (
        float(jnp.max(jnp.abs(actual - expected))),
        float(jnp.max(jnp.abs(expected))),
    )


def test_radial_database_flux_table_transpose_matches_generic_vjp():
    """The compact black-box centre rule is the established database VJP."""

    rho = jnp.asarray([0.1, 0.3, 0.5, 0.7, 0.9])
    nu_v = jnp.asarray([1.0e-3, 1.0e-2, 1.0e-1])
    er = jnp.asarray([[1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1]])
    shape = (rho.size, nu_v.size, er.shape[1])
    base = jnp.reshape(jnp.arange(int(jnp.prod(jnp.asarray(shape))), dtype=jnp.float64), shape)
    database = PreprocessedMonoenergetic3DNTSSRadius.read_data(
        a_b=1.0, rho=rho, nu_v=nu_v, Er=er, drds=jnp.ones_like(rho),
        D11=1.0 + 0.01 * base, D13=0.2 + 0.001 * base, D33=0.3 + 0.002 * base,
    )
    geometry = collections.namedtuple(
        "CompactTransposeGeometry", "r_grid r_grid_half dr full_grid_indices"
    )(
        jnp.asarray([0.2, 0.4, 0.6, 0.8]),
        jnp.asarray([0.1, 0.3, 0.5, 0.7, 0.9]),
        jnp.asarray(0.2),
        jnp.arange(4, dtype=jnp.int32),
    )
    species = Species(
        number_species=2,
        species_indices=jnp.asarray([0, 1]),
        mass_mp=jnp.asarray([5.446e-4, 2.0]),
        charge_qp=jnp.asarray([-1.0, 1.0]),
        names=("e", "D"),
    )
    energy_grid = collections.namedtuple(
        "CompactTransposeEnergyGrid",
        "xWeights L11_weight L12_weight L22_weight L13_weight L23_weight L33_weight v_norm",
    )(
        jnp.asarray([0.25, 0.75]), jnp.asarray([1.0, 0.7]),
        jnp.asarray([0.1, -0.2]), jnp.asarray([0.8, 1.2]),
        jnp.asarray([0.4, 0.5]), jnp.asarray([-0.3, 0.2]),
        jnp.asarray([1.1, 0.6]), jnp.asarray([1.2, 1.8]),
    )
    density = jnp.asarray([[1.0, 1.05, 1.1, 1.15], [0.9, 0.95, 1.0, 1.05]])
    temperature = jnp.asarray([[2.0, 2.1, 2.2, 2.3], [1.6, 1.7, 1.8, 1.9]])
    er_center = jnp.asarray([1.0e-4, -1.2e-4, 1.5e-4, -1.8e-4])
    gamma_bar = jnp.asarray([[0.2, -0.1, 0.3, -0.4], [-0.3, 0.5, -0.2, 0.1]])
    q_bar = -0.7 * gamma_bar
    upar_bar = 0.4 * gamma_bar

    def _fluxes(d11_log, d13, d33):
        _, gamma, q, upar = get_Neoclassical_Fluxes(
            species, energy_grid, geometry,
            dataclasses.replace(database, D11_log=d11_log, D13=d13, D33=d33),
            er_center, temperature, density,
        )
        return gamma, q, upar

    _, generic_pullback = jax.vjp(
        _fluxes, database.D11_log, database.D13, database.D33
    )
    zero_bar = jnp.zeros_like(gamma_bar)
    for channel, output_bars in (
        ("Gamma", (gamma_bar, zero_bar, zero_bar)),
        ("Q", (zero_bar, q_bar, zero_bar)),
        ("Upar", (zero_bar, zero_bar, upar_bar)),
        ("joint", (gamma_bar, q_bar, upar_bar)),
    ):
        expected = generic_pullback(output_bars)
        actual = pullback_preprocessed_radial_database_fluxes(
            species, energy_grid, geometry, database, er_center, temperature, density,
            *output_bars,
        )
        for table_name, actual_table_bar, expected_table_bar in zip(
            ("D11_log", "D13", "D33"), actual, expected, strict=True
        ):
            assert jnp.allclose(actual_table_bar, expected_table_bar, rtol=2.0e-10, atol=2.0e-10), (
                channel,
                table_name,
                float(jnp.max(jnp.abs(actual_table_bar - expected_table_bar))),
                float(jnp.max(jnp.abs(expected_table_bar))),
            )

    # The separate direct-state boundary must be exactly the established
    # local database flux VJP.  It is intentionally tested independently of
    # the table transpose above: Radau uses this path at every stage whereas
    # table bars are accumulated and folded only once after the sweep.
    model = NTXDatabaseTransportModel(
        species=species,
        energy_grid=energy_grid,
        geometry=geometry,
        database=database,
    )
    state = TransportState(
        density=density,
        pressure=density * temperature,
        Er=er_center,
    )
    model_flux_bar = {"Gamma": gamma_bar, "Q": q_bar, "Upar": upar_bar}
    _, state_pullback = jax.vjp(lambda state_value: model(state_value), state)
    (expected_state_bar,) = state_pullback(model_flux_bar)
    actual_state_bar = model.pullback_direct_rhs_state(state, model_flux_bar)
    for actual_leaf, expected_leaf in zip(
        jax.tree_util.tree_leaves(actual_state_bar),
        jax.tree_util.tree_leaves(expected_state_bar),
        strict=True,
    ):
        if jnp.issubdtype(jnp.asarray(expected_leaf).dtype, jnp.inexact):
            assert jnp.allclose(actual_leaf, expected_leaf, rtol=2.0e-10, atol=2.0e-10)


def test_legacy_monoenergetic_flux_table_transpose_matches_generic_vjp():
    """The black-box centre rule remains exact for scan-generated tables."""

    rho = jnp.asarray([0.1, 0.3, 0.5, 0.7, 0.9])
    nu_log = jnp.asarray([-5.0, -3.0, -1.0, 1.0])
    er_list = jnp.broadcast_to(jnp.asarray([-8.0, -5.0, -2.0, 1.0]), (rho.size, 4))
    base = jnp.reshape(jnp.arange(80, dtype=jnp.float64), (5, 4, 4))
    database = Monoenergetic(
        a_b=1.0, rho=rho, nu_log=nu_log, Er_list=er_list,
        D11_log=-3.0 + 0.001 * base, D13=0.2 + 0.001 * base,
        D33=0.3 + 0.002 * base,
    )
    geometry = collections.namedtuple(
        "LegacyMonoTransposeGeometry", "r_grid r_grid_half dr full_grid_indices"
    )(
        jnp.asarray([0.2, 0.4, 0.6, 0.8]),
        jnp.asarray([0.1, 0.3, 0.5, 0.7, 0.9]),
        jnp.asarray(0.2), jnp.arange(4, dtype=jnp.int32),
    )
    species = Species(
        number_species=2, species_indices=jnp.asarray([0, 1]),
        mass_mp=jnp.asarray([5.446e-4, 2.0]), charge_qp=jnp.asarray([-1.0, 1.0]),
        names=("e", "D"),
    )
    energy_grid = collections.namedtuple(
        "LegacyMonoTransposeEnergyGrid",
        "xWeights L11_weight L12_weight L22_weight L13_weight L23_weight L33_weight v_norm",
    )(
        jnp.asarray([0.25, 0.75]), jnp.asarray([1.0, 0.7]),
        jnp.asarray([0.1, -0.2]), jnp.asarray([0.8, 1.2]),
        jnp.asarray([0.4, 0.5]), jnp.asarray([-0.3, 0.2]),
        jnp.asarray([1.1, 0.6]), jnp.asarray([1.2, 1.8]),
    )
    density = jnp.asarray([[1.0, 1.05, 1.1, 1.15], [0.9, 0.95, 1.0, 1.05]])
    temperature = jnp.asarray([[2.0, 2.1, 2.2, 2.3], [1.6, 1.7, 1.8, 1.9]])
    er_center = jnp.asarray([1.0e-4, -1.2e-4, 1.5e-4, -1.8e-4])
    gamma_bar = jnp.asarray([[0.2, -0.1, 0.3, -0.4], [-0.3, 0.5, -0.2, 0.1]])
    q_bar, upar_bar = -0.7 * gamma_bar, 0.4 * gamma_bar

    def _fluxes(d11_log, d13, d33):
        _, gamma, q, upar = get_Neoclassical_Fluxes(
            species, energy_grid, geometry,
            dataclasses.replace(database, D11_log=d11_log, D13=d13, D33=d33),
            er_center, temperature, density,
        )
        return gamma, q, upar

    _, generic_pullback = jax.vjp(_fluxes, database.D11_log, database.D13, database.D33)
    expected = generic_pullback((gamma_bar, q_bar, upar_bar))
    actual = pullback_preprocessed_radial_database_fluxes(
        species, energy_grid, geometry, database, er_center, temperature, density,
        gamma_bar, q_bar, upar_bar,
    )
    for actual_table_bar, expected_table_bar in zip(actual, expected, strict=True):
        assert jnp.allclose(actual_table_bar, expected_table_bar, rtol=3.0e-10, atol=3.0e-10), (
            float(jnp.max(jnp.abs(actual_table_bar - expected_table_bar))),
            float(jnp.max(jnp.abs(expected_table_bar))),
        )
