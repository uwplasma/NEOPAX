import jax.numpy as jnp
import jax
from jax import config
# to use higher precision
config.update("jax_enable_x64", True)
import orthax
from _parameters import n_radial, Nx, n_species, Sonine_expansion
from _field import a_b


#This should go to grids
#defining xgrid quadrature points, and weights for needed entries of transport matrices
xgrid=orthax.laguerre.laggauss(Nx)
x=xgrid[0]
xWeights=xgrid[1]
#Define normalised v grid points
v_norm=jnp.sqrt(x)
#Define weights for velocity integrals of energy convolution
L11_weight=jnp.power(x,2.) #The same mount of v factors as in MONKES paper n_v-1, as one of the factors is used to get the factor d(v^2)= dx
L12_weight=jnp.power(x,3.)
L22_weight=jnp.power(x,4.) 
L13_weight=jnp.power(x,1.5)
L23_weight=jnp.power(x,2.5)
L33_weight=jnp.power(x,1.)
####Extra weights for momentum correction Lij is a 5x5 matrix
L24_weight=jnp.power(x,3.5)
L25_weight=jnp.power(x,4.5)
L43_weight=jnp.power(x,2.)
L44_weight=jnp.power(x,3.)
L45_weight=jnp.power(x,4.)
L55_weight=jnp.power(x,5.)


#Radial grids
r0 = 0.
r_final = a_b 
rho_grid=jnp.linspace(0., 1., n_radial)
rho_grid_half=jnp.linspace((rho_grid[0]+rho_grid[1])*0.5, (rho_grid[0]+rho_grid[1])*0.5+rho_grid[-1], n_radial)
r_grid_half=rho_grid_half*a_b
r_grid=rho_grid*a_b
dr=r_grid[2]-r_grid[1]

species_indeces=jnp.arange(n_species)
full_grid_indeces=jnp.arange(n_radial)
sonine_indeces=jnp.arange(len(Sonine_expansion))