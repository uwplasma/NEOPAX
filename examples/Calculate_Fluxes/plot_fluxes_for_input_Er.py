"""
Plot neoclassical particle fluxes for a given input electric field profile using NEOPAX analytic profiles and the Er profile from file.
No ambipolarity root-finding is performed.
"""

import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
import h5py as h5
import interpax
import NEOPAX
from NEOPAX._energy_grid_models import StandardLaguerreEnergyGrid
from NEOPAX._geometry_models import get_geometry_model
from NEOPAX._transport_flux_models import get_transport_flux_model
from NEOPAX._state import TransportState

# --- Paths and files ---
current_path = os.path.dirname(os.path.realpath(__file__))
input_path = os.path.join(current_path, '../inputs/')
output_path = os.path.join(current_path, './')
os.makedirs(output_path, exist_ok=True)

vmec_file = os.path.join(input_path, 'wout_QI_nfp2_newNT_opt_hires.nc')
boozer_file = os.path.join(input_path, 'boozermn_wout_QI_nfp2_newNT_opt_hires.nc')
neoclassical_file = os.path.join(input_path, 'Dij_NEOPAX_FULL_S_NEW_Er_Opt.h5')
Er_file = os.path.join(input_path, 'NTSS_Initial_Er_Opt.h5')

n_species = 3
n_radial = 51
n_x = 4

energy_grid = StandardLaguerreEnergyGrid(n_x=n_x)
geometry = get_geometry_model("vmec_booz", n_r=n_radial, vmec=vmec_file, booz=boozer_file)
rho = geometry.rho_grid

def analytic_profile(val0, val_edge, power=2.0):
    return (val0 - val_edge) * (1 - (rho ) ** power) + val_edge

# --- Analytic profiles (match Solve_Er_General) ---
ne0, te0, ni0, ti0 = 4.21e20, 17.8e3, 4.21e20, 17.8e3
neb, teb, nib, tib = 0.6e20, 0.7e3, 0.6e20, 0.7e3
deuterium_ratio, tritium_ratio = 0.5, 0.5

ne = analytic_profile(ne0, neb, power=10.0)
nD = deuterium_ratio * analytic_profile(ni0, nib, power=10.0)
nT = tritium_ratio * analytic_profile(ni0, nib, power=10.0)
Te = analytic_profile(te0, teb, power=2.0)
TD = analytic_profile(ti0, tib, power=2.0)
TT = analytic_profile(ti0, tib, power=2.0)

density = jnp.stack([ne, nD, nT])
temperature = jnp.stack([Te, TD, TT])

# --- Load Er profile from file (as in Fluxes_Calculation_comparison) ---
with h5.File(Er_file, 'r') as f:
    Er_interp = interpax.Interpolator1D(f['r'][()], f['Er'][()], extrap=True)
    Er = Er_interp(geometry.r_grid)

# --- Species object ---
mass = jnp.array([1 / 1836.15267343, 2, 3])
charge = jnp.array([-1, 1, 1])
species = NEOPAX.Species(
    number_species=n_species,
    species_indices=jnp.arange(n_species),
    mass_mp=mass,
    charge_qp=charge,
)

database = NEOPAX.Monoenergetic.read_monkes(geometry.a_b, neoclassical_file)


# --- Calculate neoclassical fluxes for the input Er profile (direct call) ---
Gamma_neo = NEOPAX.get_Neoclassical_Fluxes(
    species, energy_grid, geometry, database, Er, temperature, density
)[1]  # [1] is Gamma

# --- Calculate fluxes using the transport flux model ---
state = TransportState(
    density=density,
    temperature=temperature,
    Er=Er
)
params = {
    "species": species,
    "energy_grid": energy_grid,
    "geometry": geometry,
    "database": database,
}
transport_flux_model = get_transport_flux_model("monkes_database")
fluxes_model = transport_flux_model(state, geometry=geometry, params=params)
Gamma_model = fluxes_model["Gamma"]

# --- Plot comparison ---
plt.figure(dpi=120)
plt.plot(rho, Gamma_neo[0], label='Electron flux (direct)')
plt.plot(rho, Gamma_neo[1], label='Deuterium flux (direct)')
plt.plot(rho, Gamma_neo[2], label='Tritium flux (direct)')
plt.plot(rho, Gamma_model[0], '--', label='Electron flux (model)')
plt.plot(rho, Gamma_model[1], '--', label='Deuterium flux (model)')
plt.plot(rho, Gamma_model[2], '--', label='Tritium flux (model)')
plt.xlabel(r'$\rho$')
plt.ylabel(r'$\Gamma_s$')
plt.title('Neoclassical Particle Fluxes vs Input $E_r$ (direct vs model)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'neo_fluxes_vs_input_Er_comparison.pdf'))
plt.show()
