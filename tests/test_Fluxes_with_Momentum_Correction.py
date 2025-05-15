#Importing modules
import jax.numpy as jnp
import jax
jax.config.update('jax_platform_name', 'cpu')
# before execute any computation / allocation
print('Using device:', jax.devices())
from jax import config
# to use higher precision
config.update("jax_enable_x64", True)
import interpax
import h5py as h5
import NEOPAX
import matplotlib.pyplot as plt
from NEOPAX._constants import elementary_charge

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 18
FONT_SIZE = 24

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=FONT_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=FONT_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=FONT_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


#Opening NTSS file for solution with and without momentum correction
file_Initial=h5.File('./tests/inputs/NTSS_W7X_Initial.h5','r')
file_Initial_Momentum=h5.File('./tests/inputs/NTSS_W7X_Initial_Momentum.h5','r')
Er_Initial=interpax.Interpolator1D(file_Initial['r'][()],file_Initial['Er'][()],extrap=True)


vmec_file='./tests/inputs/wout_W7-X_standard_configuration.nc'
boozer_file= './tests/inputs/boozmn_wout_W7-X_standard_configuration.nc'
neoclassical_file='./tests/inputs/Dij_NEOPAX_FULL_S_NEW_W7X.h5'
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
T_edge=jnp.array([0.7*1.e+3,0.7*1.e+3,0.7*1.e+3])
n_edge=jnp.array([0.6e+20,deuterium_ratio*0.6e+20,tritium_ratio*0.6e+20])

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



J_final=(-Upar_mom[:,0]+Upar_mom[:,1]+Upar_mom[:,2])*elementary_charge

J_final_solution=jnp.array([      0.,            5490.01149652,   15723.81318418,   34574.91308781,64887.17372981,  108450.88898812,  165983.13877999,  237387.46345921,
  321587.90346994,  416175.72179143,  518472.77319291,  624510.3142208,  729934.96988671,  815973.11170985,  877139.83742598,  926288.65759796,
  963878.52631876,  991093.77976222, 1009832.80846102, 1022279.55758274, 1029733.34148606, 1029899.2568709,  1021832.29332657, 1008456.81159438,
  994577.01664654,  986048.62449312, 1003396.57863455, 1038623.31006343, 1090188.02497005, 1153088.43944538, 1220619.5355514,  1287086.31167344,
 1345705.97875629, 1398339.01918713, 1444801.58088483, 1482720.44647127, 1511650.04093041, 1530628.21933747, 1538721.02794647, 1538253.40668978,
 1533457.1424207,  1532368.16711923, 1546331.62980544, 1584620.60009108, 1622778.96964537, 1579475.24450098, 1405819.70164514, 1119183.09254559,
  760928.70423782,  358191.32235386,  -13257.68692882])


Final=jnp.abs(J_final-J_final_solution)

assert jnp.all(Final[1:19]) < 1.e-6 ,"Electric field solution matches correct value."
assert jnp.all(Final[20::]) < 1.e-6 , "Electric field solution matches correct value."

file_Initial.close()
file_Initial_Momentum.close()