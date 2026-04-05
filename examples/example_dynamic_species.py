"""
Example: Constructing and checking a NEOPAX transport framework with dynamic species and quasi-neutrality (torax-style).
"""
import jax.numpy as jnp
from jaxtyping import Array, Float
from NEOPAX._state import TransportState
from NEOPAX._species import Species

# --- 1. Define dynamic species ---

# No helium, only D, T, e
species_names = ("D", "T", "e")
charges = jnp.array([1.0, 1.0, -1.0])  # D, T, e
masses = jnp.array([2.014, 3.016, 0.000548])  # in proton mass units
is_ion = jnp.array([True, True, False])

# --- 2. Create initial state ---

# Example profiles similar to Solve_Er_General (parabolic, edge drop)
n_radial = 8
n_species = len(species_names)
r = jnp.linspace(0, 1, n_radial)

# Central density 1.0, edge 0.2 (parabolic)
density_profile = 0.2 + 0.8 * (1 - r**2)
# Central temperature 10.0, edge 1.0 (parabolic)
temperature_profile = 1.0 + 9.0 * (1 - r**2)

density = jnp.stack([density_profile, density_profile, jnp.zeros_like(density_profile)])  # D, T, e (e not evolved)
temperature = jnp.stack([temperature_profile, temperature_profile, temperature_profile])
Er = jnp.zeros(n_radial)

state = TransportState(
    density=density,
    temperature=temperature,
    Er=Er,
    species_names=species_names,
    is_evolved=is_ion  # only ions are evolved
)

species = Species(
    number_species=n_species,
    species_indices=jnp.arange(n_species),
    mass_mp=masses,
    charge_qp=charges,
    names=species_names,
    is_frozen=~is_ion
)

# --- 3. Enforce quasi-neutrality for electrons ---
ion_indices = jnp.where(is_ion)[0]
ion_charges = charges[ion_indices]
ion_densities = state.density[ion_indices]
n_e = jnp.sum(ion_charges[:, None] * ion_densities, axis=0)  # shape (n_radial,)

print("Quasi-neutral electron density:", n_e)

# --- 4. Example: Lookup by species name ---
def get_species_idx(name, names):
    return names.index(name)

idx_D = get_species_idx("D", species_names)
print("D index:", idx_D)
print("D density:", state.density[idx_D])

# --- 5. Ready for use in source/transport models ---
# (Models can now use state.species_names and dynamic lookup)
