#Initialization
import os
current_path = os.path.dirname(os.path.realpath(__name__))
import jax
import jax.numpy as jnp
jax.config.update('jax_platform_name', 'cpu')
print(jax.devices()) # TFRT_CPU_0
from jax import config
# to use higher precision
config.update("jax_enable_x64", True)
import h5py as h5
import matplotlib.pyplot as plt
import NEOPAX
from NEOPAX._constants import elementary_charge

vmec_file=os.path.join(current_path,'./inputs/wout_QI_nfp2_newNT_opt_hires.nc')
boozer_file=os.path.join(current_path, './inputs/boozermn_wout_QI_nfp2_newNT_opt_hires.nc')
neoclassical_file='./inputs/Dij_NEOPAX_FULL_S_NEW_Er_Opt.h5'
neoclassical_option= 1

#Solver parameters 
parameters=NEOPAX.Solver_Parameters()

#Create grid
n_species=3
Nx=4
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
T_edge=jnp.array([0.7*1.e+3,0.7*1.e+3,0.7*1.e+3])
n_edge=jnp.array([0.6e+20,deuterium_ratio*0.6e+20,tritium_ratio*0.6e+20])

#Define species profiles 
Te_initial_function= lambda r: T_scale*((te0-teb)*(1-(r/field.a_b)**2)+teb)
ne_initial_function =lambda r: n_scale*((ne0-neb)*(1-(r/field.a_b)**10.)+neb)
TD_initial_function= lambda r: T_scale*((ti0-tib)*(1-(r/field.a_b)**2)+tib)
nD_initial_function = lambda r: n_scale*deuterium_ratio*((ni0-nib)*(1-(r/field.a_b)**10.)+nib)
TT_initial_function= lambda r: T_scale*((ti0-tib)*(1-(r/field.a_b)**2)+tib)
nT_initial_function = lambda r: n_scale*tritium_ratio*((ni0-nib)*(1-(r/field.a_b)**10.)+nib)
Er_initial_function = lambda x: 100.*x*(0.8-x)

#Er0 = SpatialDiscretisation.discretise_fn(r0, r_final, n_radial, Er_initial_function)

#Initialize species
Te_initial=Te_initial_function(field.r_grid)
ne_initial=ne_initial_function(field.r_grid)
TD_initial=TD_initial_function(field.r_grid)
nD_initial=nD_initial_function(field.r_grid)
TT_initial=TT_initial_function(field.r_grid)
nT_initial=nT_initial_function(field.r_grid)
Er_initial=Er_initial_function(field.rho_grid)


temperature_initial=jnp.zeros((n_species,n_radial))
density_initial=jnp.zeros((n_species,n_radial))
temperature_initial=temperature_initial.at[0,:].set(Te_initial)
temperature_initial=temperature_initial.at[1,:].set(TD_initial)
temperature_initial=temperature_initial.at[2,:].set(TT_initial)
density_initial=density_initial.at[0,:].set(ne_initial)
density_initial=density_initial.at[1,:].set(nD_initial)
density_initial=density_initial.at[2,:].set(nT_initial)

fr=0.0
temperature_initial=temperature_initial.at[:,0].set((4.*temperature_initial.at[:,1].get()-temperature_initial.at[:,2].get()-fr*2*field.dr)/(3.0))   
temperature_initial=temperature_initial.at[:,-1].set(T_edge)
density_initial=density_initial.at[:,0].set((4.*density_initial.at[:,1].get()-density_initial.at[:,2].get()-fr*2*field.dr)/(3.0))   
density_initial=density_initial.at[:,-1].set(n_edge)
mass=jnp.array([1 / 1836.15267343,2,3])
charge=jnp.array([-1,1,1])

#Initialçize species
Global_species=NEOPAX.Species(n_species,n_radial,grid.species_indeces,mass,charge,temperature_initial,density_initial,Er_initial,field.r_grid,field.r_grid_half,field.dr,field.Vprime_half, field.overVprime,n_edge,T_edge)


#Read monoenergetic database, from MONKES 
database=NEOPAX.Monoenergetic.read_monkes(field.a_b,neoclassical_file)

turbulent=NEOPAX.Turbulence.from_analytical_model(Global_species,jnp.zeros(3))

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
args=Global_species,grid,field,database,turbulent,parameters


sol=NEOPAX.solve_transport_equations(y0,args)

Er_final=sol.ys[0][-1,:]


Er_final_solution=jnp.array([-1.22315383e-28,  1.34148416e+00,  2.68049898e+00,  4.01457838e+00,  5.34141786e+00,  6.65883724e+00,  7.96491378e+00,  9.25819838e+00,
  1.05379801e+01,  1.18045717e+01,  1.30596103e+01,  1.43065483e+01,  1.55528346e+01,  1.68249640e+01,  1.81298768e+01,  1.94465836e+01,
  2.07675719e+01,  2.20864651e+01,  2.33955143e+01,  2.46796001e+01,  2.59154377e+01,  2.71024752e+01,  2.82296743e+01,  2.92845702e+01,
  3.02548772e+01,  3.11273140e+01,  3.18774798e+01,  3.25045423e+01,  3.29713771e+01,  3.31124574e+01,  3.20962431e+01,  2.57181512e+01,
 -5.76493735e+00, -6.66467296e+00, -7.18522100e+00, -7.75864110e+00, -8.40553406e+00, -9.14665133e+00, -1.00175170e+01, -1.10641749e+01,
 -1.22968727e+01, -1.37198451e+01, -1.53452514e+01, -1.72394724e+01, -1.94734517e+01, -2.21161817e+01, -2.51286207e+01, -2.84218098e+01,
 -3.28134826e+01, -4.04411762e+01, -5.49779014e+01])


assert not jnp.all(jnp.abs(Er_final-Er_final_solution)) < 1.e-6, "Electric field solution matches correct value."










