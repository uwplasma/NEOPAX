#Importing modules
import pytest
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


def test_fluxes_with_momentum_correction():
  #Opening NTSS file for solution with and without momentum correction
  file_Initial=h5.File('./tests/inputs/NTSS_W7X_Initial.h5','r')
  Er_Initial=interpax.Interpolator1D(file_Initial['r'][()],file_Initial['Er'][()],extrap=True)

  vmec_file='./tests/inputs/wout_W7-X_standard_configuration.nc'
  boozer_file='./tests/inputs/boozmn_wout_W7-X_standard_configuration.nc'
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



  J_final=elementary_charge*(-Upar[0]+Upar[1]+Upar[2])

  J_final_solution=jnp.array([      0.,         45168.07982563,   95688.8335667,   155825.74970629,  228581.28274038,  315657.68784945,  417446.43776736,  533218.09844003,
    661089.10371784,  797792.20906534,  939562.66801993, 1081403.75730233, 1217938.51171592, 1326921.75818245, 1402303.70751863, 1459721.67655899,
  1500096.34683585, 1525191.88607759, 1537429.402792,   1540022.41980644, 1534879.80792856, 1516742.23711296, 1485087.24827067, 1445172.12561103,
  1405434.79877314, 1375563.00896985, 1375873.40553034, 1400463.96764785, 1447399.89938824, 1509482.97625722, 1576839.96538312, 1640499.07468469,
  1685269.78960362, 1715367.54473189, 1734068.28948583, 1742016.50079477, 1743352.64694388, 1741526.94511271, 1742317.19748327, 1748390.1280721,
  1756702.56910076, 1773933.69814733, 1817689.98340421, 1915766.4127553, 2047554.44333294, 2051217.90138083, 1819614.4363332,  1377410.70531666,
    817668.10002162,  289681.33337484,  -77014.35955482])


  Final=jnp.abs(J_final-J_final_solution)


  assert jnp.all(Final[1:19]) < 1.e-6 ,"Electric field solution matches correct value."
  assert jnp.all(Final[20:]) < 1.e-6 , "Electric field solution matches correct value."

if __name__ == "__main__":
    pytest.main()