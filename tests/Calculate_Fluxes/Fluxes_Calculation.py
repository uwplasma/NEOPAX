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

#reading NTSS data for comparison, for an optimised electron root equilibrium
file_Initial=h5.File('../inputs/NTSS_Test_Initial_Er_Opt.h5','r')
Er_Initial=interpax.Interpolator1D(file_Initial['r'][()],file_Initial['Er'][()],extrap=True)


#Initialising species data
ne0 = 4.21e20
te0 = 17.8e+3
ni0 = 4.21e20
ti0 = 17.8e+3
neb = 0.6e20
teb = 0.7e+3
nib = 0.6e20
tib = 0.7e+3

n_scale=1.      
T_scale=1.      
deuterium_ratio=0.5    
tritium_ratio=0.5    

#Define species profiles 
Te_initial_function= lambda r: T_scale*((te0-teb)*(1-(r/a_b)**2)+teb)
ne_initial_function =lambda r: n_scale*((ne0-neb)*(1-(r/a_b)**10.)+neb)
TD_initial_function= lambda r: T_scale*((ti0-tib)*(1-(r/a_b)**2)+tib)
nD_initial_function = lambda r: n_scale*deuterium_ratio*((ni0-nib)*(1-(r/a_b)**10.)+nib)
TT_initial_function= lambda r: T_scale*((ti0-tib)*(1-(r/a_b)**2)+tib)
nT_initial_function = lambda r: n_scale*tritium_ratio*((ni0-nib)*(1-(r/a_b)**10.)+nib)
#Using electric field from NTSS solution for comparison
Er_initial_function = lambda x: Er_Initial(x)

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


#Initialçize Global species, which updated at every timestep
global Global_Species   
Global_species=Species(n_species,n_radial,mass,charge,temperature_initial,density_initial,Er_initial)

from _neoclassical import get_Neoclassical_Fluxes, Bsqav, B_00
Lij,Gamma,Q,Upar=get_Neoclassical_Fluxes(Global_species,species_indeces,full_grid_indeces)

fig,ax=plt.subplots(dpi=120)
plt.ylabel('$n_a$')
plt.xlabel('$\\rho$')
ax.plot(rho_grid,Global_species.density[0],label='electrons')
ax.plot(rho_grid,Global_species.density[1],label='deuterium')
ax.plot(rho_grid,Global_species.density[2],label='tritium')
plt.legend()
plt.savefig('densities.pdf')

fig,ax=plt.subplots(dpi=120)
plt.ylabel('$T_a$')
plt.xlabel('$\\rho$')
ax.plot(rho_grid,Global_species.temperature[0],label='electrons')
ax.plot(rho_grid,Global_species.temperature[1],label='deuterium')
ax.plot(rho_grid,Global_species.temperature[2],label='tritum')
plt.legend()
plt.savefig('temperatures.pdf')

fig,ax=plt.subplots(dpi=120)
plt.ylabel('$E_r$')
plt.xlabel('$\\rho$')
ax.plot(rho_grid,Global_species.Er)
plt.legend()
plt.savefig('Electric_field.pdf')

fig,ax=plt.subplots(dpi=120)
plt.ylabel('$L11_e$')
plt.xlabel('$\\rho$')
ax.plot(file_Initial['r'][()]/a_b,file_Initial['D11e'][()],label='NTSS')
ax.plot(rho_grid,Lij[0,:,0,0],label='NEOPAX')
plt.legend()
plt.savefig('L11e.pdf')

fig,ax=plt.subplots(dpi=120)
plt.ylabel('$L12_e$')
plt.xlabel('$\\rho$')
ax.plot(file_Initial['r'][()]/a_b,file_Initial['D12e'][()],label='NTSS')
ax.plot(rho_grid,(Lij[0,:,0,1]-1.5*Lij[0,:,0,0]),label='NEOPAX')
plt.legend()
plt.savefig('L12e.pdf')

fig,ax=plt.subplots(dpi=120)
plt.ylabel('$L21_e$')
plt.xlabel('$\\rho$')
ax.plot(file_Initial['r'][()]/a_b,file_Initial['D21e'][()],label='NTSS')
ax.plot(rho_grid,(Lij[0,:,1,0]),label='NEOPAX')
plt.legend()
plt.savefig('L21e.pdf')

fig,ax=plt.subplots(dpi=120)
plt.ylabel('$L22_e$')
plt.xlabel('$\\rho$')
ax.plot(file_Initial['r'][()]/a_b,file_Initial['D22e'][()],label='NTSS')
ax.plot(rho_grid,(Lij[0,:,1,1]-1.5*Lij[0,:,1,0]),label='NEOPAX')
plt.legend()
plt.savefig('L22e.pdf')

fig,ax=plt.subplots(dpi=120)
plt.ylabel('$L31_e$')
plt.xlabel('$\\rho$')
ax.plot(file_Initial['r'][()]/a_b,file_Initial['D31e'][()],label='NTSS')
ax.plot(rho_grid,Lij[0,:,2,0],label='NEOPAX')
plt.legend()
plt.savefig('L31e.pdf')

fig,ax=plt.subplots(dpi=120)
plt.ylabel('$L32_e$')
plt.xlabel('$\\rho$')
ax.plot(file_Initial['r'][()]/a_b,file_Initial['D32e'][()],label='NTSS')
ax.plot(rho_grid,(Lij[0,:,2,1]-1.5*Lij[0,:,2,0]),label='NEOPAX')
plt.legend()
plt.savefig('L32e.pdf')

#NTSS outputs only in electron conductivity units
fig,ax=plt.subplots(dpi=120)
plt.ylabel('$L33_e$')
plt.xlabel('$\\rho$')
ax.plot(file_Initial['r'][()]/a_b,file_Initial['Sigma'][()],label='NTSS')
ax.plot(rho_grid,jnp.absolute(Lij[0,:,2,2]*(-elementary_charge*2/jnp.sqrt(jnp.pi)*Global_species.density[0]/Global_species.temperature[0])),label='NEOPAX')
plt.legend()
plt.savefig('L33e.pdf')

fig,ax=plt.subplots(dpi=120)
#Deuterium 
plt.ylabel('$L11_D$')
plt.xlabel('$\\rho$')
ax.plot(file_Initial['r'][()]/a_b,file_Initial['D11d'][()],label='NTSS')
ax.plot(rho_grid,Lij[1,:,0,0],label='NEOPAX')
plt.legend()
plt.savefig('L11D.pdf')

fig,ax=plt.subplots(dpi=120)
plt.ylabel('$L12_D$')
plt.xlabel('$\\rho$')
ax.plot(file_Initial['r'][()]/a_b,file_Initial['D12d'][()],label='NTSS')
ax.plot(rho_grid,(Lij[1,:,0,1]-1.5*Lij[1,:,0,0]),label='NEOPAX')
plt.legend()
plt.savefig('L12D.pdf')


fig,ax=plt.subplots(dpi=120)
plt.ylabel('$L21_D$')
plt.xlabel('$\\rho$')
ax.plot(file_Initial['r'][()]/a_b,file_Initial['D21d'][()],label='NTSS')
ax.plot(rho_grid,(Lij[1,:,1,0]),label='NEOPAX')
plt.legend()
plt.savefig('L21D.pdf')

fig,ax=plt.subplots(dpi=120)
plt.ylabel('$L22_D$')
plt.xlabel('$\\rho$')
ax.plot(file_Initial['r'][()]/a_b,file_Initial['D22d'][()],label='NTSS')
ax.plot(rho_grid,(Lij[1,:,1,1]-1.5*Lij[1,:,1,0]),label='NEOPAX')
plt.legend()
plt.savefig('L22D.pdf')

fig,ax=plt.subplots(dpi=120)
plt.ylabel('$L31_D$')
plt.xlabel('$\\rho$')
ax.plot(file_Initial['r'][()]/a_b,file_Initial['D31d'][()],label='NTSS')
ax.plot(rho_grid,Lij[1,:,2,0],label='NEOPAX')
plt.legend()
plt.savefig('L31D.pdf')

fig,ax=plt.subplots(dpi=120)
plt.ylabel('$L32_D$')
plt.xlabel('$\\rho$')
ax.plot(file_Initial['r'][()]/a_b,file_Initial['D32d'][()],label='NTSS')
ax.plot(rho_grid,(Lij[1,:,2,1]-1.5*Lij[1,:,2,0]),label='NEOPAX')
plt.legend()
plt.savefig('L32D.pdf')

plt.close() #for freeing memory due to the large quantity of  plots

#Tritium
fig,ax=plt.subplots(dpi=120)
plt.ylabel('$L11_T$')
plt.xlabel('$\\rho$')
ax.plot(file_Initial['r'][()]/a_b,file_Initial['D11t'][()],label='NTSS')
ax.plot(rho_grid,Lij[2,:,0,0],label='NEOPAX')
plt.legend()
plt.savefig('L11T.pdf')

fig,ax=plt.subplots(dpi=120)
plt.ylabel('$L12_T$')
plt.xlabel('$\\rho$')
ax.plot(file_Initial['r'][()]/a_b,file_Initial['D12t'][()],label='NTSS')
ax.plot(rho_grid,(Lij[2,:,0,1]-1.5*Lij[2,:,0,0]),label='NEOPAX')
plt.legend()
plt.savefig('L12T.pdf')

fig,ax=plt.subplots(dpi=120)
plt.ylabel('$L21_T$')
plt.xlabel('$\\rho$')
ax.plot(file_Initial['r'][()]/a_b,file_Initial['D21t'][()],label='NTSS')
ax.plot(rho_grid,(Lij[2,:,1,0]),label='NEOPAX')
plt.legend()
plt.savefig('L21T.pdf')

fig,ax=plt.subplots(dpi=120)
plt.ylabel('$L22_T$')
plt.xlabel('$\\rho$')
ax.plot(file_Initial['r'][()]/a_b,file_Initial['D22t'][()],label='NTSS')
ax.plot(rho_grid,(Lij[2,:,1,1]-1.5*Lij[2,:,1,0]),label='NEOPAX')
plt.legend()
plt.savefig('L22T.pdf')

fig,ax=plt.subplots(dpi=120)
plt.ylabel('$L31_T$')
plt.xlabel('$\\rho$')
ax.plot(file_Initial['r'][()]/a_b,file_Initial['D31t'][()],label='NTSS')
ax.plot(rho_grid,Lij[2,:,2,0],label='NEOPAX')
plt.legend()
plt.savefig('L31T.pdf')

fig,ax=plt.subplots(dpi=120)
plt.ylabel('$L32_T$')
plt.xlabel('$\\rho$')
ax.plot(file_Initial['r'][()]/a_b,file_Initial['D32t'][()],label='NTSS')
ax.plot(rho_grid,(Lij[2,:,2,1]+1.5*Lij[2,:,2,0]),label='NEOPAX')
plt.legend()
plt.savefig('L32T.pdf')

fig,ax=plt.subplots(dpi=120)
plt.ylabel('$J^{BS}$')
plt.xlabel('$\\rho$')
ax.plot(file_Initial['r'][()]/a_b,file_Initial['J_bs'][()],label='NTSS')
ax.plot(rho_grid,elementary_charge*(-Upar[0]+Upar[1]+Upar[2]),label='NEOPAX')
plt.legend()
plt.savefig('J_Bootstrap.pdf')

fig,ax=plt.subplots(dpi=120)
plt.ylabel('$J^{BS}_e$')
plt.xlabel('$\\rho$')
ax.plot(file_Initial['r'][()]/a_b,file_Initial['J_bse'][()],label='NTSS')
ax.plot(rho_grid,elementary_charge*(-Upar[0]),label='NEOPAX')
plt.legend()
plt.savefig('J_Bootstrap_e.pdf')

fig,ax=plt.subplots(dpi=120)
plt.ylabel('$J^{BS}_i$')
plt.xlabel('$\\rho$')
ax.plot(file_Initial['r'][()]/a_b,file_Initial['J_bsi'][()],label='NTSS')
ax.plot(rho_grid,elementary_charge*(Upar[1]+Upar[2]),label='NEOPAX')
plt.legend()
plt.savefig('J_Bootstrap_Ion.pdf')

plt.close()
file_Initial.close()
