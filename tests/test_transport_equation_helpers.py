import dataclasses
from types import SimpleNamespace

import jax.numpy as jnp

from NEOPAX._boundary_conditions import BoundaryConditionModel
from NEOPAX._ambipolarity import _ambipolar_root_grid_has_axis_state_entry
from NEOPAX._state import TransportState
from NEOPAX._transport_equations import (
    ElectricFieldEquation,
    PARTICLE_FLUX_PHYSICAL_TO_STATE,
    TemperatureEquation,
    _expand_density_rhs_to_full_shape,
    apply_er_dirichlet_boundary_state,
    enforce_quasi_neutrality,
    project_fixed_temperature_species,
)


@dataclasses.dataclass(frozen=True)
class DummySpecies:
    charge_qp: jnp.ndarray
    names: tuple[str, ...]
    ion_indices: tuple[int, ...]


def test_enforce_quasi_neutrality_reconstructs_electron_density():
    species = DummySpecies(
        charge_qp=jnp.array([-1.0, 1.0, 2.0]),
        names=("e", "D", "He"),
        ion_indices=(1, 2),
    )
    state = TransportState(
        density=jnp.array(
            [
                [0.0, 0.0],
                [3.0, 4.0],
                [1.0, 2.0],
            ]
        ),
        pressure=jnp.ones((3, 2)),
        Er=jnp.zeros(2),
    )

    out = enforce_quasi_neutrality(state, species)
    assert jnp.allclose(out.density[0], jnp.array([5.0, 8.0]))
    assert jnp.allclose(out.density[1:], state.density[1:])


def test_project_fixed_temperature_species_only_changes_inactive_rows():
    state = TransportState(
        density=jnp.array([[2.0, 2.0], [3.0, 3.0]]),
        pressure=jnp.array([[20.0, 24.0], [100.0, 200.0]]),
        Er=jnp.zeros(2),
    )
    active_mask = jnp.array([True, False])
    fixed_temperature = jnp.array([[9.0, 9.0], [7.0, 7.0]])

    out = project_fixed_temperature_species(
        state,
        temperature_active_mask=active_mask,
        fixed_temperature_profile=fixed_temperature,
    )

    assert jnp.allclose(out.pressure[0], state.pressure[0])
    assert jnp.allclose(out.pressure[1], jnp.array([21.0, 21.0]))


def test_apply_er_dirichlet_boundary_state_clamps_endpoints():
    bc = BoundaryConditionModel(
        dr=1.0,
        left_type="dirichlet",
        right_type="dirichlet",
        left_value=jnp.array([1.5]),
        right_value=jnp.array([-2.0]),
    )
    state = TransportState(
        density=jnp.ones((1, 4)),
        pressure=jnp.ones((1, 4)),
        Er=jnp.array([0.0, 3.0, 4.0, 5.0]),
    )

    out = apply_er_dirichlet_boundary_state(state, bc)
    assert jnp.allclose(out.Er, jnp.array([1.5, 3.0, 4.0, -2.0]))


def test_expand_density_rhs_to_full_shape_inserts_zero_electron_row():
    species = DummySpecies(
        charge_qp=jnp.array([-1.0, 1.0, 1.0]),
        names=("e", "D", "T"),
        ion_indices=(1, 2),
    )
    reduced_rhs = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    template = jnp.zeros((3, 2))

    out = _expand_density_rhs_to_full_shape(reduced_rhs, template, species)
    assert jnp.allclose(out, jnp.array([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]]))


def test_expand_density_rhs_to_full_shape_returns_zero_template_on_mismatch():
    template = jnp.ones((3, 2))
    bad_rhs = jnp.ones((4,))
    out = _expand_density_rhs_to_full_shape(bad_rhs, template, species=None)
    assert jnp.allclose(out, jnp.zeros_like(template))


def test_ambipolar_axis_skip_uses_state_centres_not_axis_face():
    cell_centred_geometry = SimpleNamespace(
        r_grid=jnp.asarray([0.01, 0.03]),
        r_grid_half=jnp.asarray([0.0, 0.02, 0.04]),
    )
    axis_node_geometry = SimpleNamespace(r_grid=jnp.asarray([0.0, 0.02]))

    assert not _ambipolar_root_grid_has_axis_state_entry(cell_centred_geometry)
    assert _ambipolar_root_grid_has_axis_state_entry(axis_node_geometry)


def test_face_completed_ambipolar_term_cell_centres_completed_face_scalar():
    """The opt-in Er source uses supplied model faces, not reconstructed centres."""

    state = TransportState(
        density=jnp.ones((1, 2)),
        pressure=jnp.ones((1, 2)),
        Er=jnp.zeros(2),
    )
    gamma_faces = jnp.asarray([[2.0, 4.0, 8.0]])
    equation = ElectricFieldEquation(
        dr_cells=jnp.ones(2),
        Vprime=jnp.ones(2),
        Vprime_half=jnp.ones(3),
        flux_model=None,
        species_mass=jnp.ones(1),
        charge_qp=jnp.ones(1),
        permitivity_prefactor=jnp.ones(2),
        gamma_faces_builder=lambda gamma: jnp.zeros((1, gamma.shape[1] + 1)),
        er_diffusive_flux_builder=lambda er: jnp.zeros(er.shape[0] + 1),
        source_mode="ambipolar_face_completed",
        permitivity_mode="ntss_like_midpoint",
        ntss_B0_mid=1.0,
        ntss_psfactor_mid=1.0,
        ntss_density_indices=jnp.asarray([0]),
    )
    # Deliberately inconsistent centre Gamma: if the implementation silently
    # reconstructed faces from it, this assertion would fail.
    charge_flux, ambi_term = equation._charge_flux_and_ambi_term(
        state,
        Gamma=jnp.asarray([[100.0, 200.0]]),
        plasma_permitivity=jnp.ones(2),
        Gamma_faces=gamma_faces,
    )
    expected_charge_flux = jnp.asarray([3.0, 6.0])
    expected_coeff = 95780.0
    assert jnp.allclose(charge_flux, expected_charge_flux)
    assert jnp.allclose(ambi_term, expected_coeff * expected_charge_flux * 1.0e-20)


def test_floating_er_edge_prefers_native_face_flux_over_center_reconstruction():
    """The floating condition must use the model's outer face flux directly."""

    state = TransportState(
        density=jnp.ones((1, 2)),
        pressure=jnp.ones((1, 2)),
        Er=jnp.zeros(2),
    )
    equation = ElectricFieldEquation(
        dr_cells=jnp.ones(2),
        Vprime=jnp.ones(2),
        Vprime_half=jnp.ones(3),
        flux_model=None,
        species_mass=jnp.ones(1),
        charge_qp=jnp.ones(1),
        permitivity_prefactor=jnp.ones(2),
        # A deliberately incompatible fallback makes it clear that the
        # native `Gamma_faces` payload, not reconstruction, is selected.
        gamma_faces_builder=lambda gamma: jnp.zeros((1, gamma.shape[1] + 1)),
        er_diffusive_flux_builder=lambda er: jnp.zeros(er.shape[0] + 1),
        source_mode="ambipolar_local",
        permitivity_mode="ntss_like_midpoint",
        boundary_mode="floating_ambipolar_edge",
        ntss_B0_mid=1.0,
        ntss_psfactor_mid=1.0,
        ntss_density_indices=jnp.asarray([0]),
    )

    rhs = equation(
        state,
        fluxes={
            "Gamma": jnp.asarray([[100.0, 200.0]]),
            "Gamma_faces": jnp.asarray([[2.0, 4.0, 8.0]]),
        },
    )
    assert jnp.allclose(rhs[-1], -95780.0 * 8.0e-20)


def test_floating_er_edge_node_keeps_last_cell_diffusion_and_evolves_face_node():
    """The NTSS-like mode must not replace the last FV-cell equation."""
    state = TransportState(
        density=jnp.ones((1, 2)),
        pressure=jnp.ones((1, 2)),
        Er=jnp.asarray([1.0, 2.0]),
    )
    equation = ElectricFieldEquation(
        dr_cells=jnp.ones(2),
        Vprime=jnp.ones(2),
        Vprime_half=jnp.ones(3),
        flux_model=None,
        species_mass=jnp.ones(1),
        charge_qp=jnp.ones(1),
        permitivity_prefactor=jnp.ones(2),
        gamma_faces_builder=lambda gamma: jnp.zeros((1, gamma.shape[1] + 1)),
        # A nonzero outer diffusive face makes a replacement of rhs[-1]
        # immediately detectable: conservative_update gives [-0, -3].
        er_diffusive_flux_builder=lambda er, er_edge: jnp.asarray([0.0, 0.0, 3.0]),
        source_mode="ambipolar_local",
        permitivity_mode="ntss_like_midpoint",
        boundary_mode="floating_ambipolar_edge_node",
        ntss_B0_mid=1.0,
        ntss_psfactor_mid=1.0,
        ntss_density_indices=jnp.asarray([0]),
    )
    fluxes = {
        "Gamma": jnp.zeros((1, 2)),
        "Gamma_faces": jnp.asarray([[0.0, 0.0, 8.0]]),
    }
    rhs = equation(state, fluxes=fluxes, er_edge_override=jnp.asarray(5.0))

    # The node is deliberately not a public TransportState field.  It is a
    # Radau-local scalar so ordinary forward/reverse state pytrees remain the
    # same three leaves.
    assert not hasattr(state, "Er_edge")
    assert jnp.allclose(rhs, jnp.asarray([0.0, -3.0]))
    assert jnp.allclose(
        equation.edge_rhs(state, fluxes=fluxes, er_edge_override=jnp.asarray(5.0)),
        -95780.0 * 8.0e-20,
    )


def test_face_completed_work_term_cell_centres_completed_face_product():
    """Work interpolation must average ``q Gamma_face Er_face`` as one scalar."""

    state = TransportState(
        density=jnp.ones((1, 2)),
        pressure=jnp.ones((1, 2)),
        Er=jnp.asarray([1.0, 2.0]),
    )
    equation = TemperatureEquation(
        dr_cells=jnp.ones(2),
        Vprime=jnp.ones(2),
        Vprime_half=jnp.ones(3),
        flux_model=None,
        flux_faces_builder=lambda gamma: gamma,
        temperature_ghost_builder=lambda temperature: temperature,
        charge_qp=jnp.asarray([1.0]),
        active_species_mask=jnp.asarray([True]),
        er_faces_builder=lambda unused_state: jnp.asarray([10.0, 20.0, 30.0]),
        include_work_term=True,
        work_term_reconstruction="face_completed",
    )
    work_rhs = equation._work_rhs(
        state,
        fluxes={"Gamma": jnp.asarray([[100.0, 200.0]])},
        face_fluxes={"Gamma_faces": jnp.asarray([[2.0, 4.0, 8.0]])},
    )
    # centre values are 0.5 * [2*10 + 4*20, 4*20 + 8*30].
    expected = PARTICLE_FLUX_PHYSICAL_TO_STATE * jnp.asarray([[50.0, 160.0]])
    assert jnp.allclose(work_rhs, expected)
