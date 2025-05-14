#Importing modules
import os
current_path = os.path.dirname(os.path.realpath(__name__))
import sys
sys.path.insert(1, os.path.join(current_path,'../../'))
import jax.numpy as jnp
import jax
jax.config.update('jax_platform_name', 'gpu')
# before execute any computation / allocation
print('Using device:', jax.devices())
from jax import config
# to use higher precision
config.update("jax_enable_x64", True)
import toml
import interpax
import h5py as h5
import NEOPAX
import matplotlib.pyplot as plt
from NEOPAX._constants import elementary_charge

#Opening NTSS file for solution with and without momentum correction
file_Initial=h5.File('../inputs/NTSS_Er_Opt_Large_Initial.h5','r')
file_Initial_Momentum=h5.File('../inputs/NTSS_Er_Opt_Large_Initial_Moments.h5','r')
Er_Initial=interpax.Interpolator1D(file_Initial['r'][()],file_Initial['Er'][()],extrap=True)


vmec_file=os.path.join(current_path,'../inputs/wout_QI_nfp2_newNT_opt_hires.nc')
boozer_file=os.path.join(current_path, '../inputs/boozermn_wout_QI_nfp2_newNT_opt_hires.nc')
neoclassical_file='../inputs/Dij_NEOPAX_FULL_S_NEW_Er_Opt_Large.h5'
neoclassical_option= 1

#Create grid
n_species=3
Nx=64
n_radial=51
grid=NEOPAX.Grid.create_standard(n_radial,Nx,n_species)

#Get magnetic configuration related quantities from class Field
field=NEOPAX.Field.read_vmec_booz(n_radial,vmec_file,boozer_file)

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

#Boundaries for temperature and density (to be improved)
T_edge=jnp.array(0.7*1.e+3)
n_edge=jnp.array(0.6e+20)

#Define species profiles 
Te_initial_function= lambda r: T_scale*((te0-teb)*(1-(r/field.a_b)**2)+teb)
ne_initial_function =lambda r: n_scale*((ne0-neb)*(1-(r/field.a_b)**10.)+neb)
TD_initial_function= lambda r: T_scale*((ti0-tib)*(1-(r/field.a_b)**2)+tib)
nD_initial_function = lambda r: n_scale*deuterium_ratio*((ni0-nib)*(1-(r/field.a_b)**10.)+nib)
TT_initial_function= lambda r: T_scale*((ti0-tib)*(1-(r/field.a_b)**2)+tib)
nT_initial_function = lambda r: n_scale*tritium_ratio*((ni0-nib)*(1-(r/field.a_b)**10.)+nib)
Er_initial_function = lambda x: Er_Initial(x)

#Er0 = SpatialDiscretisation.discretise_fn(r0, r_final, n_radial, Er_initial_function)

#Initialize species
Te_initial=Te_initial_function(field.r_grid)
ne_initial=ne_initial_function(field.r_grid)
TD_initial=TD_initial_function(field.r_grid)
nD_initial=nD_initial_function(field.r_grid)
TT_initial=TT_initial_function(field.r_grid)
nT_initial=nT_initial_function(field.r_grid)
Er_initial=Er_initial_function(field.r_grid)


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

#Initialçize species
Global_species=NEOPAX.Species(n_species,n_radial,grid.species_indeces,mass,charge,temperature_initial,density_initial,Er_initial,field.r_grid,field.r_grid_half,field.dr,field.Vprime_half, field.overVprime,n_edge,T_edge)


#Read monoenergetic database, from MONKES 
database=NEOPAX.Monoenergetic.read_monkes(field.a_b,neoclassical_file)


#Calculating fluxes with momentum correction and without momentum correction

Lij,Gamma,Q,Upar=NEOPAX.get_Neoclassical_Fluxes(Global_species,grid,field,database)
Gamma_mom,Q_mom,Upar_mom,qpar_mom,Upar2_mom=NEOPAX.get_Neoclassical_Fluxes_With_Momentum_Correction(Global_species,grid,field,database)



#Comparing bootstrap current
fig,ax=plt.subplots(dpi=120)
plt.ylabel('$J^{BS}$')
plt.xlabel('$\\rho$')
ax.plot(file_Initial['r'][()]/field.a_b,file_Initial['J_bs'][()],label='NTSS, without corr')
ax.plot(file_Initial_Momentum['r'][()]/field.a_b,file_Initial_Momentum['J_bs'][()],label='NTSS, with corr')
ax.plot(field.rho_grid,elementary_charge*(-Upar[0]+Upar[1]+Upar[2]),label='JAX, without corr')
ax.plot(field.rho_grid,(-Upar_mom[:,0]+Upar_mom[:,1]+Upar_mom[:,2])*elementary_charge, label='JAX, with corr')
plt.legend()
plt.savefig('J_Bootstrap_with_momentum.pdf')
plt.close()

#Comparing bootstrap current for electrons
fig,ax=plt.subplots(dpi=120)
plt.ylabel('$J^{BS}$')
plt.xlabel('$\\rho$')
ax.plot(file_Initial['r'][()]/field.a_b,file_Initial['J_bse'][()],label='NTSS, without corr')
ax.plot(file_Initial_Momentum['r'][()]/field.a_b,file_Initial_Momentum['J_bse'][()],label='NTSS, with corr')
ax.plot(field.rho_grid,elementary_charge*(-Upar[0]),label='JAX, without corr')
ax.plot(field.rho_grid,(-Upar_mom[:,0])*elementary_charge, label='JAX, with corr')
plt.legend()
plt.savefig('J_Bootstrap_with_momentum_electrons.pdf')
plt.close()

#Comparing bootstrap current for ions
fig,ax=plt.subplots(dpi=120)
plt.ylabel('$J^{BS}$')
plt.xlabel('$\\rho$')
ax.plot(file_Initial['r'][()]/field.a_b,file_Initial['J_bsi'][()],label='NTSS, without corr')
ax.plot(file_Initial_Momentum['r'][()]/field.a_b,file_Initial_Momentum['J_bsi'][()],label='NTSS, with corr')
ax.plot(field.rho_grid,elementary_charge*(Upar[1]+Upar[2]),label='JAX, without corr')
ax.plot(field.rho_grid,(Upar_mom[:,1]+Upar_mom[:,2])*elementary_charge, label='JAX, with corr')
plt.legend()
plt.savefig('J_Bootstrap_with_momentum_IONS.pdf')
plt.close()

#Comparing heat fluxes for ions 
fig,ax=plt.subplots(dpi=120)
plt.ylabel('$Q_i$')
plt.xlabel('$\\rho$')
ax.plot(file_Initial['r'][()]/field.a_b,file_Initial['FluxQiNeo'][()],label='NTSS, without corr')
ax.plot(file_Initial_Momentum['r'][()]/field.a_b,file_Initial_Momentum['FluxQiNeo'][()],label='NTSS, with corr')
ax.plot(field.rho_grid,(Q[1]+Q[2])*1e-3*1.e-20/62.422,label='JAX, without corr')
#ax.plot(rho_grid,(Q_mom[:,1]+Q_mom[:,2]), label='JAX, with corr')
plt.legend()
plt.savefig('Q_heat_with_momentum_ions.pdf')
plt.close()

#Comparing heat fluxes for ions 
fig,ax=plt.subplots(dpi=120)
plt.ylabel('$Q_e$')
plt.xlabel('$\\rho$')
ax.plot(file_Initial['r'][()]/field.a_b,file_Initial['FluxQeNeo'][()],label='NTSS, without corr')
ax.plot(file_Initial_Momentum['r'][()]/field.a_b,file_Initial_Momentum['FluxQeNeo'][()],label='NTSS, with corr')
ax.plot(field.rho_grid,Q[0]*1e-3*1.e-20/62.422,label='JAX, without corr')
#ax.plot(rho_grid,Q_mom[:,0]/Global_species.density[0]/Global_species.temperature[0], label='JAX, with corr')
plt.legend()
plt.savefig('Q_heat_with_momentum_electrons.pdf')
plt.close()

#Comparing particle fluxes for ions 
fig,ax=plt.subplots(dpi=120)
plt.ylabel('$\\Gamma$')
plt.xlabel('$\\rho$')
ax.plot(file_Initial['r'][()]/field.a_b,file_Initial['FluxNeo'][()],label='NTSS, without corr')
ax.plot(file_Initial_Momentum['r'][()]/field.a_b,file_Initial_Momentum['FluxNeo'][()],label='NTSS, with corr')
ax.plot(field.rho_grid,1.e-20*(Gamma[0]),label='JAX, without corr')
#ax.plot(rho_grid,Gamma_mom[:,0], label='JAX, with corr')
plt.legend()
plt.savefig('Gamma_with_momentum_electrons.pdf')
plt.close()



file_Initial.close()
file_Initial_Momentum.close()