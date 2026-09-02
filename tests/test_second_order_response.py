import jax.numpy as jnp
from types import SimpleNamespace

from NEOPAX._second_order_response import (
    DirectionalSecondOrderJet,
    compose_ntx_coefficient_quadratic,
    divide,
    evaluate,
    maximum_with_constant_floor,
    multiply,
    seed,
)
from NEOPAX._boundary_conditions import BoundaryConditionModel
from NEOPAX._state import TransportState
from NEOPAX._state import get_v_thermal
from NEOPAX._species import Species
from NEOPAX._transport_flux_models import (
    _build_evaluated_transport_state_directional,
    _face_profile_directional,
    _jet_vthermal_from_temperature,
    _nu_over_vnew_local_directional_default,
    build_evaluated_transport_state,
)
from NEOPAX._neoclassical import _nu_over_vnew_local


def test_directional_second_order_jet_product_and_reciprocal_are_exact_to_second_order():
    """Written algebra preserves an analytic identity without generic AD."""
    value = jnp.asarray([2.0, 3.0])
    delta = jnp.asarray([0.15, -0.12])
    basis = seed(value, delta)
    response = divide(multiply(basis, basis), basis)
    assert jnp.allclose(evaluate(response), value + delta, rtol=1.0e-12, atol=1.0e-12)


def test_directional_second_order_jet_reciprocal_has_cubic_remainder():
    def error(scale):
        value = jnp.asarray([2.0])
        response = divide(1.0, seed(value, jnp.asarray([scale])))
        return jnp.abs(evaluate(response) - 1.0 / (value + scale))[0]

    errors = jnp.asarray([error(0.10), error(0.05), error(0.025)])
    assert float(errors[1] / errors[0]) < 0.18
    assert float(errors[2] / errors[1]) < 0.18


def test_ntx_coefficient_composition_includes_coordinate_chain_terms():
    """The cached NTX Hessian is composed with state-coordinate curvature."""
    nu = seed(jnp.asarray([1.0]), jnp.asarray([0.2]), jnp.asarray([0.3]))
    epsi = seed(jnp.asarray([2.0]), jnp.asarray([-0.4]), jnp.asarray([0.5]))
    response = compose_ntx_coefficient_quadratic(
        reference_coefficients=jnp.asarray([[7.0]]),
        dcoefficients_d_nu_hat=jnp.asarray([[11.0]]),
        dcoefficients_d_epsi_hat=jnp.asarray([[13.0]]),
        d2coefficients_d_nu_hat2=jnp.asarray([[17.0]]),
        d2coefficients_d_nu_hat_d_epsi_hat=jnp.asarray([[19.0]]),
        d2coefficients_d_epsi_hat2=jnp.asarray([[23.0]]),
        nu_hat=nu,
        epsi_hat=epsi,
    )
    expected_first = 11.0 * 0.2 + 13.0 * -0.4
    expected_second = 11.0 * 0.3 + 13.0 * 0.5 + 17.0 * 0.2**2 + 2.0 * 19.0 * 0.2 * -0.4 + 23.0 * (-0.4)**2
    assert jnp.allclose(response.first, jnp.asarray([[expected_first]]))
    assert jnp.allclose(response.second, jnp.asarray([[expected_second]]))


def test_floor_jet_freezes_clamped_anchor_branch():
    response = maximum_with_constant_floor(
        seed(jnp.asarray([2.0, 0.2]), jnp.asarray([0.3, 0.4])),
        1.0,
    )
    assert jnp.allclose(response.value, jnp.asarray([2.0, 1.0]))
    assert jnp.allclose(response.first, jnp.asarray([0.3, 0.0]))
    assert jnp.allclose(response.second, jnp.zeros(2))


def test_face_jet_uses_homogeneous_explicit_dirichlet_tangent():
    face_centers = jnp.asarray([0.0, 1.0, 2.0, 3.0])
    profile = seed(jnp.asarray([2.0, 3.0, 5.0]), jnp.asarray([0.4, -0.2, 0.1]))
    bc = BoundaryConditionModel(
        dr=1.0,
        left_type="dirichlet",
        right_type="dirichlet",
        left_value=jnp.asarray(7.0),
        right_value=jnp.asarray(11.0),
    )
    faces = _face_profile_directional(profile, face_centers, bc_model=bc)
    assert jnp.allclose(faces.value, jnp.asarray([7.0, 2.5, 4.0, 11.0]))
    assert jnp.allclose(faces.first, jnp.asarray([0.0, 0.1, -0.05, 0.0]))


def test_face_jet_uses_homogeneous_explicit_neumann_tangent():
    face_centers = jnp.asarray([0.0, 1.0, 2.0, 3.0])
    profile = seed(jnp.asarray([2.0, 3.0, 5.0]), jnp.asarray([0.4, -0.2, 0.1]))
    bc = BoundaryConditionModel(
        dr=1.0,
        left_type="neumann",
        right_type="neumann",
        left_gradient=jnp.asarray(4.0),
        right_gradient=jnp.asarray(-6.0),
    )
    faces = _face_profile_directional(profile, face_centers, bc_model=bc)
    # The prescribed gradients affect values but have no state derivative.
    assert jnp.allclose(faces.first, jnp.asarray([0.475, 0.1, -0.05, 0.1375]))


def test_evaluated_state_jet_matches_second_directional_difference():
    geometry = SimpleNamespace(r_grid_half=jnp.asarray([0.0, 0.4, 1.0, 1.8]))
    state = TransportState(
        density=jnp.asarray([[2.0, 2.4, 3.2], [1.5, 1.7, 1.9]]),
        pressure=jnp.asarray([[4.0, 5.3, 7.7], [2.7, 3.4, 4.4]]),
        Er=jnp.asarray([0.1, -0.3, 0.5]),
    )
    direction = TransportState(
        density=jnp.asarray([[0.2, -0.1, 0.3], [-0.2, 0.1, 0.05]]),
        pressure=jnp.asarray([[0.5, 0.3, -0.2], [0.1, -0.15, 0.2]]),
        Er=jnp.asarray([0.03, -0.02, 0.04]),
    )
    response = _build_evaluated_transport_state_directional(state, direction, geometry)
    eps = 1.0e-3
    def direct(sign):
        displaced = TransportState(
            density=state.density + sign * eps * direction.density,
            pressure=state.pressure + sign * eps * direction.pressure,
            Er=state.Er + sign * eps * direction.Er,
        )
        return build_evaluated_transport_state(displaced, geometry).face.temperature

    central_first = (direct(1.0) - direct(-1.0)) / (2.0 * eps)
    central_second = (direct(1.0) - 2.0 * direct(0.0) + direct(-1.0)) / eps**2
    assert jnp.allclose(response.face.temperature.first, central_first, rtol=2.0e-6, atol=2.0e-6)
    assert jnp.allclose(response.face.temperature.second, central_second, rtol=2.0e-4, atol=2.0e-4)


def test_default_local_collision_jet_matches_second_directional_difference():
    species = Species(
        number_species=2,
        species_indices=jnp.asarray([0, 1]),
        mass_mp=jnp.asarray([5.446e-4, 2.0]),
        charge_qp=jnp.asarray([-1.0, 1.0]),
        names=("e", "D"),
    )
    density = seed(jnp.asarray([2.0, 1.5]), jnp.asarray([0.1, -0.04]))
    temperature = seed(jnp.asarray([4.0, 3.0]), jnp.asarray([0.2, 0.1]))
    reference_vthermal = get_v_thermal(species.mass, temperature.value)
    vthermal = _jet_vthermal_from_temperature(reference_vthermal, temperature)
    energy_norm = jnp.asarray([0.5, 1.0, 1.8])
    vnew = DirectionalSecondOrderJet(
        energy_norm * vthermal.value[0],
        energy_norm * vthermal.first[0],
        energy_norm * vthermal.second[0],
    )
    response = _nu_over_vnew_local_directional_default(species, 0, vnew, density, temperature, vthermal)
    eps = 1.0e-3

    def direct(scale):
        density_value = density.value + scale * direction_density
        temperature_value = temperature.value + scale * direction_temperature
        vthermal_value = get_v_thermal(species.mass, temperature_value)
        return _nu_over_vnew_local(
            species, 0, energy_norm * vthermal_value[0],
            density_value, temperature_value, vthermal_value, 0,
        )

    direction_density = density.first
    direction_temperature = temperature.first
    first = (direct(eps) - direct(-eps)) / (2.0 * eps)
    second = (direct(eps) - 2.0 * direct(0.0) + direct(-eps)) / eps**2
    assert jnp.allclose(response.first, first, rtol=3.0e-5, atol=3.0e-5)
    assert jnp.allclose(response.second, second, rtol=3.0e-3, atol=3.0e-3)
