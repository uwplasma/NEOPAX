"""NTSS work-source checks on small arrays; no transport or NTX solves."""

import dataclasses
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from NEOPAX._state import TransportState
from NEOPAX._temperature_constraints import helium_work_match_indices
from NEOPAX._transport_equations import (
    ComposedEquationSystem,
    PARTICLE_FLUX_PHYSICAL_TO_STATE,
    TemperatureEquation,
    build_equation_system,
)


def species(names=("e", "D", "T", "He")):
    charge = {"e": -1., "D": 1., "T": 1., "He": 2.}
    return SimpleNamespace(
        names=names, number_species=len(names),
        charge_qp=jnp.asarray([charge[name] for name in names]),
        ion_indices=tuple(i for i, name in enumerate(names) if name != "e"),
    )


def state():
    density = jnp.ones((4, 2))
    return TransportState(density=density, pressure=3 * density, Er=jnp.asarray([2., -4.]))


def fluxes():
    # -Gamma_e + Gamma_D + Gamma_T + 2 Gamma_He = 0 at each point.
    neo = jnp.asarray([[9., 11.], [1., 2.], [2., 3.], [3., 3.]]) * 1e20
    return {
        "Gamma": neo + jnp.asarray([[10., 20.], [30., 40.], [50., 60.], [70., 80.]]) * 1e20,
        "Gamma_neo": neo,
        "Q": jnp.zeros((4, 2)), "Q_faces": jnp.zeros((4, 3)),
    }


def equation(flag=None, names=("e", "D", "T", "He"), **kwargs):
    settings = dict(
        dr_cells=jnp.ones(2), Vprime=jnp.ones(2), Vprime_half=jnp.ones(3),
        flux_model=None, flux_faces_builder=lambda gamma, *_: gamma,
        temperature_ghost_builder=lambda temperature: jnp.pad(temperature, ((0, 0), (1, 1)), mode="edge"),
        charge_qp=species(names).charge_qp,
        active_species_mask=jnp.asarray([name != "He" for name in names]),
        species=species(names), include_neo_convection=False,
        include_turbulent_convection=False, include_classical_convection=False,
        heat_flux_reconstruction="closure_face_flux", convection_reconstruction="closure_face_flux",
    )
    if flag is not None:
        settings["match_workHe"] = flag
    settings.update(kwargs)
    return TemperatureEquation(**settings)


@pytest.mark.parametrize("flag", [None, False])
def test_disabled_work_match_preserves_the_original_array_calculation(flag):
    current = state()
    shared = fluxes()
    eq = equation(flag)
    # Exactly the pre-option arithmetic, including total Gamma and the raw He row.
    expected = (
        eq.charge_qp[:, None] * PARTICLE_FLUX_PHYSICAL_TO_STATE
        * shared["Gamma"] * current.Er[None, :]
    )
    actual = jax.jit(lambda s, f: eq._work_rhs(s, f, None))(current, shared)
    np.testing.assert_array_equal(actual, expected)
    rhs_expected = (2. / 3.) * expected * eq.active_species_mask[:, None]
    np.testing.assert_array_equal(eq(current, shared), rhs_expected)


def test_matched_work_agrees_with_ntss_signed_sources_and_ambipolar_sum():
    shared = fluxes()
    actual = equation(True)._work_rhs(state(), shared, None)
    # Direct NTSS integer-point source values for the two field signs.
    expected = np.asarray([[-18., 44.], [8., -20.], [10., -24.], [0., 0.]])
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_allclose(jnp.sum(actual, axis=0), 0., atol=1e-14)
    np.testing.assert_array_equal(shared["Gamma_neo"], fluxes()["Gamma_neo"])
    np.testing.assert_array_equal(shared["Gamma"], fluxes()["Gamma"])


def test_matched_work_uses_neo_flux_and_is_independent_of_total_flux():
    eq = equation(True)
    original = fluxes()
    changed = {**original, "Gamma": 1e8 * original["Gamma"]}
    np.testing.assert_array_equal(
        eq._work_rhs(state(), original, None), eq._work_rhs(state(), changed, None)
    )


def test_nonambipolar_work_sum_retains_the_full_charge_current():
    shared = fluxes()
    shared["Gamma_neo"] = shared["Gamma_neo"].at[0].add(-.5e20)
    eq = equation(True)
    actual = jnp.sum(eq._work_rhs(state(), shared, None), axis=0)
    charge_flux = jnp.sum(eq.charge_qp[:, None] * shared["Gamma_neo"], axis=0)
    expected = PARTICLE_FLUX_PHYSICAL_TO_STATE * state().Er * charge_flux
    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_matched_work_follows_species_names_instead_of_fixed_row_numbers():
    permutation = jnp.asarray([3, 2, 0, 1])
    names = ("He", "T", "e", "D")
    shared = {name: value[permutation] for name, value in fluxes().items()}
    actual = equation(True, names)._work_rhs(state(), shared, None)
    expected = equation(True)._work_rhs(state(), fluxes(), None)[permutation]
    np.testing.assert_array_equal(actual, expected)


def test_matched_work_can_use_face_primary_neo_payload_at_centres():
    eq = equation(True)
    faces = jnp.asarray([[8., 10., 12.], [0., 2., 4.], [1., 3., 5.], [3., 3., 3.]]) * 1e20
    shared = {"Gamma_neo_faces": faces}
    expected = eq._work_rhs(state(), {"Gamma_neo": .5 * (faces[:, :-1] + faces[:, 1:])}, None)
    np.testing.assert_array_equal(eq._work_rhs(state(), shared, None), expected)


def test_matched_work_selects_local_ntss_source_even_if_face_work_was_configured():
    eq = equation(True, work_term_reconstruction="face_completed")
    assert not eq._use_face_completed_work_term()
    np.testing.assert_array_equal(
        eq._work_rhs(state(), fluxes(), None), equation(True)._work_rhs(state(), fluxes(), None)
    )


def test_missing_neo_channel_is_not_silently_replaced_by_total_flux():
    with pytest.raises(ValueError, match="Gamma_neo"):
        equation(True)._work_rhs(state(), {"Gamma": fluxes()["Gamma"]}, None)


def test_global_work_disable_still_returns_zero_with_matching_enabled():
    actual = equation(True, include_work_term=False)._work_rhs(state(), {}, None)
    np.testing.assert_array_equal(actual, jnp.zeros((4, 2)))


@pytest.mark.parametrize("value", ["false", 1, None])
def test_work_match_requires_a_boolean(value):
    with pytest.raises(ValueError, match="match_workHe"):
        helium_work_match_indices({"equations": {"match_workHe": value}}, species().names)


def test_configuration_rejects_an_independent_helium_energy_equation():
    with pytest.raises(ValueError, match="toggle_temperature=false for He"):
        helium_work_match_indices({"equations": {"match_workHe": True}}, species().names)
    with pytest.raises(ValueError, match="requires species"):
        helium_work_match_indices({"equations": {"match_workHe": True}}, ("e", "D", "T"))


def test_work_match_is_independent_of_temperature_matching_and_wired_by_factory():
    field = SimpleNamespace(
        r_grid=jnp.asarray([.25, .75]), r_grid_half=jnp.asarray([0., .5, 1.]), dr=.5,
        Vprime=jnp.ones(2), Vprime_half=jnp.ones(3),
    )
    for temperature_match in (False, True):
        cfg = {"equations": {
            "match_workHe": True, "match_THe": temperature_match,
            "toggle_temperature": [True, True, True, False],
            "toggle_density": [False] * 4, "toggle_Er": False,
        }}
        result = build_equation_system(cfg, species(), field, flux_model=None)
        assert len(result) == 1
        assert result[0].match_workHe is True


def test_pressure_factor_and_debug_components_apply_redistribution_once():
    eq = equation(True)
    expected_work = np.asarray([[-18., 44.], [8., -20.], [10., -24.], [0., 0.]])
    actual = eq(state(), fluxes())
    diagnostic = eq.debug_components(state(), fluxes())
    np.testing.assert_allclose(actual, (2. / 3.) * expected_work)
    np.testing.assert_array_equal(diagnostic["pressure_rhs"], actual)
    np.testing.assert_allclose(diagnostic["work_rhs"], expected_work)


def test_ad_of_matched_work_agrees_with_fd_for_flux_and_field_perturbations():
    eq = equation(True)
    current = state()
    neo = fluxes()["Gamma_neo"] * 1e-20
    weights = jnp.asarray([[.2, .3], [.4, -.5], [.7, -.9], [10., 20.]])

    def objective(gamma, er):
        shared = {**fluxes(), "Gamma_neo": gamma * 1e20}
        return jnp.sum(eq(dataclasses.replace(current, Er=er), shared) * weights)

    direction_g = jnp.asarray([[.1, -.2], [.3, .4], [-.5, .6], [.7, -.8]])
    direction_e = jnp.asarray([.2, -.3])
    gradient_g, gradient_e = jax.jit(jax.grad(objective, argnums=(0, 1)))(neo, current.Er)
    actual = jnp.vdot(gradient_g, direction_g) + jnp.vdot(gradient_e, direction_e)
    step = 1e-5
    expected = (
        objective(neo + step * direction_g, current.Er + step * direction_e)
        - objective(neo - step * direction_g, current.Er - step * direction_e)
    ) / (2 * step)
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)


def test_shared_flux_reverse_routes_donor_cotangents_back_to_helium_neo_flux():
    current = state()
    eq = equation(True)
    system = ComposedEquationSystem(equations=(eq,), temperature_equation=eq, species=species())
    pressure_bar = jnp.zeros((4, 2)).at[1].set(1.).at[2].set(2.)
    rhs_bar = TransportState(
        density=jnp.zeros_like(current.density), pressure=pressure_bar, Er=jnp.zeros(2),
    )
    actual = jax.jit(lambda f: system.pullback_shared_fluxes(current, f, rhs_bar))(fluxes())
    expected = np.zeros((4, 2))
    expected[1] = (2. / 3.) * np.asarray(current.Er)
    expected[2] = (4. / 3.) * np.asarray(current.Er)
    expected[3] = 2. * np.asarray(current.Er)
    np.testing.assert_allclose(actual["Gamma_neo"] * 1e20, expected, atol=1e-13)
    np.testing.assert_array_equal(actual["Gamma"], np.zeros((4, 2)))
