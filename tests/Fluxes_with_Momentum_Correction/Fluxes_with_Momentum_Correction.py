#Importing modules
import os
current_path = os.path.dirname(os.path.realpath(__name__))
import sys
sys.path.insert(1, os.path.join(current_path,'../../NEOPAX'))
import jax.numpy as jnp
import jax
from jax import config
# to use higher precision
config.update("jax_enable_x64", True)
import h5py as h5
import interpax
from _parameters import n_radial, n_species 
from _field import a_b
from _grid import r_grid, r0, r_final
from _species import Species 
import matplotlib.pyplot as plt
from _grid import full_grid_indeces,species_indeces, rho_grid
from scipy.constants import elementary_charge

#Opening NTSS file for solution with and without momentum correction
file_Initial=h5.File('../inputs/NTSS_Test_Initial_Er_Opt.h5','r')
file_Initial_Momentum=h5.File('../inputs/NTSS_Test_Initial_Momentum_Corr_Er_Opt.h5','r')
Er_Initial=interpax.Interpolator1D(file_Initial['r'][()],file_Initial['Er'][()],extrap=True)


#Defining species data
ne0 = 4.21e20
te0 = 17.8e+3
ni0 = 4.21e20
ti0 = 17.8e+3
neb = 0.6e20
teb = 0.7e+3
nib = 0.6e20
tib = 0.7e+3

n_scale=1.#0.65       
T_scale=1.#0.7       
deuterium_ratio=0.5    
tritium_ratio=0.5    

#Define species profiles 
Te_initial_function= lambda r: T_scale*((te0-teb)*(1-(r/a_b)**2)+teb)
ne_initial_function =lambda r: n_scale*((ne0-neb)*(1-(r/a_b)**10.)+neb)
TD_initial_function= lambda r: T_scale*((ti0-tib)*(1-(r/a_b)**2)+tib)
nD_initial_function = lambda r: n_scale*deuterium_ratio*((ni0-nib)*(1-(r/a_b)**10.)+nib)
TT_initial_function= lambda r: T_scale*((ti0-tib)*(1-(r/a_b)**2)+tib)
nT_initial_function = lambda r: n_scale*tritium_ratio*((ni0-nib)*(1-(r/a_b)**10.)+nib)
Er_initial_function = lambda x: Er_Initial(x)

#Er0 = SpatialDiscretisation.discretise_fn(r0, r_final, n_radial, Er_initial_function)


#Initialize species
Te_initial=Te_initial_function(r_grid)
ne_initial=ne_initial_function(r_grid)
TD_initial=TD_initial_function(r_grid)
nD_initial=nD_initial_function(r_grid)
TT_initial=TT_initial_function(r_grid)
nT_initial=nT_initial_function(r_grid)
Er_initial=Er_initial_function(r_grid)
temperature_initial=jnp.zeros((n_species,n_radial))
density_initial=jnp.zeros((n_species,n_radial))
temperature_initial=temperature_initial.at[0,:].set(Te_initial)
temperature_initial=temperature_initial.at[1,:].set(TD_initial)
temperature_initial=temperature_initial.at[2,:].set(TT_initial)
density_initial=density_initial.at[0,:].set(ne_initial)
density_initial=density_initial.at[1,:].set(nD_initial)
density_initial=density_initial.at[2,:].set(nT_initial)
mass=jnp.array([1 / 1836.15267343,2,3])
charge=jnp.array([-1,1,1])


#Initialçize Global species
global Global_Species   
Global_species=Species(n_species,n_radial,mass,charge,temperature_initial,density_initial,Er_initial)

#Calculating fluxes with momentum correction and without momentum correction
from _neoclassical import get_Neoclassical_Fluxes_With_Momentum_Correction, get_Neoclassical_Fluxes
Lij,Gamma,Q,Upar=get_Neoclassical_Fluxes(Global_species,species_indeces,full_grid_indeces)
Gamma_mom,Q_mom,Upar_mom,qpar_mom,Upar2_mom=get_Neoclassical_Fluxes_With_Momentum_Correction(Global_species,species_indeces,full_grid_indeces)


#Comparing bootstrap current
fig,ax=plt.subplots(dpi=120)
plt.ylabel('$J^{BS}$')
plt.xlabel('$\\rho$')
ax.plot(file_Initial['r'][()]/a_b,file_Initial['J_bs'][()],label='NTSS, without corr')
ax.plot(file_Initial_Momentum['r'][()]/a_b,file_Initial_Momentum['J_bs'][()],label='NTSS, with corr')
ax.plot(rho_grid,elementary_charge*(-Upar[0]+Upar[1]+Upar[2]),label='NEOPAX, without corr')
ax.plot(rho_grid,(-Upar_mom[:,0]+Upar_mom[:,1]+Upar_mom[:,2])*elementary_charge, label='NEOPAX, with corr')
plt.legend()
plt.savefig('J_Bootstrap_with_momentum.pdf')
plt.close()

file_Initial.close()
file_Initial_Momentum.close()