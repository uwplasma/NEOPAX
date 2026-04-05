"""
Example: Evolving only the Er (electric field) equation in NEOPAX, mimicking Solve_Er_General (torax-style).
"""

import jax.numpy as jnp
from NEOPAX._state import TransportState
from NEOPAX._species import Species, get_species_idx
from NEOPAX._transport_equations import ElectricFieldEquation
from NEOPAX._constants import MASS_RATIO_D, MASS_RATIO_T, MASS_RATIO_E

# --- 1. Define species and state (D, T, e) ---
species_names = ("D", "T", "e")

charges = jnp.array([1.0, 1.0, -1.0])
masses = jnp.array([MASS_RATIO_D, MASS_RATIO_T, MASS_RATIO_E])
n_radial = 8
r = jnp.linspace(0, 1, n_radial)

# Profiles (parabolic)
density_profile = 0.2 + 0.8 * (1 - r**2)
temperature_profile = 1.0 + 9.0 * (1 - r**2)

# Only Er is evolved, so densities and temperatures are fixed
# (but must be present in state for the equation)
density = jnp.stack([density_profile, density_profile, jnp.zeros_like(density_profile)])
temperature = jnp.stack([temperature_profile, temperature_profile, temperature_profile])
Er = jnp.zeros(n_radial)

state = TransportState(
    density=density,
    temperature=temperature,
    Er=Er,
    species_names=species_names,
    is_evolved=None  # Not used here
)

species = Species(
    number_species=3,
    species_indices=jnp.arange(3),
    mass_mp=masses,
    charge_qp=charges,
    names=species_names,
    is_frozen=None
)

# --- 2. Construct the Er equation only ---
er_equation = ElectricFieldEquation()

# Example: call the equation (mock flux_models, source_models, field, solver_parameters)
# In a real case, these would be constructed from your physics setup
class DummyField:
    dr = jnp.ones(n_radial)
    Vprime = jnp.ones(n_radial)
    Vprime_half = jnp.ones(n_radial)
    r_grid = r

class DummySolverParams:
    evolve_Er = True
    Er_relax = 1.0
    DEr = 1.0

flux_models = {"Gamma_total": jnp.zeros((3, n_radial))}
source_models = None
field = DummyField()
solver_parameters = DummySolverParams()

# Compute the Er RHS (time derivative)
Er_rhs = er_equation(state, flux_models, source_models, field, solver_parameters)
print("Er RHS:", Er_rhs)

# This example shows how to construct and use only the Er equation for evolution.
# In a real simulation, you would pass this to an ODE solver (e.g., diffrax) for time integration.
