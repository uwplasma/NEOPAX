import dataclasses

import jax.numpy as jnp

from NEOPAX._boundary_conditions import BoundaryConditionModel
from NEOPAX._state import TransportState
from NEOPAX._transport_equations import (
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
