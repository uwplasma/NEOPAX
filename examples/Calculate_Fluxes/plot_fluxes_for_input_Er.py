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

grid = NEOPAX.Grid.create_standard(n_radial, n_x, n_species)
field = NEOPAX.Field.read_vmec_booz(n_radial, vmec_file, boozer_file)
rho = field.rho_grid

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
    Er = Er_interp(field.r_grid)

# --- Species object ---
mass = jnp.array([1 / 1836.15267343, 2, 3])
charge = jnp.array([-1, 1, 1])
species = NEOPAX.Species(
    number_species=n_species,
    species_indices=jnp.arange(n_species),
    mass_mp=mass,
    charge_qp=charge,
)

database = NEOPAX.Monoenergetic.read_monkes(field.a_b, neoclassical_file)

# --- Calculate neoclassical fluxes for the input Er profile ---
Gamma_neo = NEOPAX.get_Neoclassical_Fluxes(
    species, grid, field, database, Er, temperature, density
)[1]  # [1] is Gamma

# --- Plot ---
plt.figure(dpi=120)
plt.plot(rho, Gamma_neo[0], label='Electron flux')
plt.plot(rho, Gamma_neo[1], label='Deuterium flux')
plt.plot(rho, Gamma_neo[2], label='Tritium flux')
plt.xlabel(r'$\rho$')
plt.ylabel(r'$\Gamma_s$')
plt.title('Neoclassical Particle Fluxes vs Input $E_r$ (from file)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'neo_fluxes_vs_input_Er.pdf'))
plt.show()
