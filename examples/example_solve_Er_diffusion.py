"""
Example: Solve electric field diffusion (Er) for fixed D, T, e profiles (mimics old Solve_Er_General).
"""
import jax.numpy as jnp
from NEOPAX._state import TransportState
from NEOPAX._species import Species, get_species_idx
from NEOPAX._transport_equations import ElectricFieldEquation
from NEOPAX._constants import MASS_RATIO_D, MASS_RATIO_T, MASS_RATIO_E
import diffrax

# --- 1. Define species and fixed profiles ---
species_names = ("D", "T", "e")
charges = jnp.array([1.0, 1.0, -1.0])
masses = jnp.array([MASS_RATIO_D, MASS_RATIO_T, MASS_RATIO_E])
n_radial = 8
r = jnp.linspace(0, 1, n_radial)

# Parabolic profiles for D, T, e (e density is zero, not evolved)
density_profile = 0.2 + 0.8 * (1 - r**2)
temperature_profile = 1.0 + 9.0 * (1 - r**2)
density = jnp.stack([density_profile, density_profile, jnp.zeros_like(density_profile)])
temperature = jnp.stack([temperature_profile, temperature_profile, temperature_profile])
Er0 = jnp.zeros(n_radial)

state0 = TransportState(
    density=density,
    temperature=temperature,
    Er=Er0,
    species_names=species_names,
    is_evolved=None
)

species = Species(
    number_species=3,
    species_indices=jnp.arange(3),
    mass_mp=masses,
    charge_qp=charges,
    names=species_names,
    is_frozen=None
)

# --- 2. Set up the Er equation and dummy models ---
er_equation = ElectricFieldEquation()

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

# --- 3. Define ODE system for Er only ---
def er_rhs(t, Er_flat, args):
    # Unflatten Er
    Er = Er_flat.reshape(n_radial)
    # State with updated Er
    state = TransportState(
        density=density,
        temperature=temperature,
        Er=Er,
        species_names=species_names,
        is_evolved=None
    )
    return er_equation(state, flux_models, source_models, field, solver_parameters)

# --- 4. Integrate with diffrax ---
term = diffrax.ODETerm(er_rhs)
solver = diffrax.Kvaerno5()
t0 = 0.0
t1 = 1.0
dt = 0.01
saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, 11))
sol = diffrax.diffeqsolve(
    term,
    solver,
    t0,
    t1,
    dt,
    Er0,
    args=None,
    saveat=saveat
)

print("Er(t) solution:", sol.ys)
