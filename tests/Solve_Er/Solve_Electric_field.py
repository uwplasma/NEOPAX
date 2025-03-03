#Initialization
import os
current_path = os.path.dirname(os.path.realpath(__name__))
import sys
sys.path.insert(1, os.path.join(current_path,'../../'))
import jax
import jax.numpy as jnp
jax.config.update('jax_platform_name', 'gpu')
print(jax.devices()) # TFRT_CPU_0
from jax import config
# to use higher precision
config.update("jax_enable_x64", True)
import h5py as h5
import interpax
import matplotlib.pyplot as plt
import NEOPAX


#Reading NTSS data file for electric field solution
file_Final=h5.File('../inputs/NTSS_TEST_Final.h5','r')
Er_Final=interpax.Interpolator1D(file_Final['r'][()],file_Final['Er'][()],extrap=True)
file_Initial=h5.File('../inputs/NTSS_Test_Initial_Er_Opt.h5','r')
Er_Initial=interpax.Interpolator1D(file_Initial['r'][()],file_Initial['Er'][()],extrap=True)
file_Initial.close()
file_Final.close()

vmec_file=os.path.join(current_path,'../inputs/wout_QI_nfp2_newNT_opt_hires.nc')
boozer_file=os.path.join(current_path, '../inputs/boozermn_wout_QI_nfp2_newNT_opt_hires.nc')
neoclassical_file='../inputs/Dij_NEOPAX_FULL_S_NEW_Er_Opt.h5'
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



#Initial profiles
ne0=Global_species.density[0]*1.e-20
nD0=Global_species.density[1]*1.e-20
nT0=Global_species.density[2]*1.e-20
Pe0=Global_species.temperature[0]*1.e-3*ne0
PD0=Global_species.temperature[1]*1.e-3*nD0
PT0=Global_species.temperature[2]*1.e-3*nT0
Er0=Global_species.Er

#Define initial coupled solutions to diffrax (can solve tupples of spatial descritizations!!! already tested)
y0=(Er0,Pe0,PD0,PT0,ne0,nD0,nT0)#,ne_initial,nT_initial,nD_initial,nHe_initial in r=rho*a_b!!!!
args=Global_species,grid,field,database


sol=NEOPAX.solve_transport_equations(y0,args)

fig,ax=plt.subplots(dpi=120)

plt.xlabel('$\\rho$')
plt.xlabel('$E_r [kV/m $')
ax.plot(field.rho_grid,sol.ys[0][0,:],'g',label='0.0s')
ax.plot(field.rho_grid,sol.ys[0][-1,:],label='final')
ax.plot(field.rho_grid,Er_Initial(field.r_grid),'',label='NTSS Initial')
#ax.plot(rho_grid,Er_Final(r_grid),'r',label='NTSS Final')
plt.legend()
plt.savefig('Er_Neopax.pdf')

