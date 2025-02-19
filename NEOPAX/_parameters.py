#Parameters
import jax.numpy as jnp
import jax
from jax import config
# to use higher precision
config.update("jax_enable_x64", True)
import h5py as h5


vmec='../inputs/wout_QI_nfp2_newNT_opt_hires.nc'
booz='../inputs/boozermn_wout_QI_nfp2_newNT_opt_hires.nc'
monkes_file='../inputs/Dij_NEOPAX_FULL_S_NEW_Er_Opt.h5'
momentum_correction_flag=False
Sonine_expansion=jnp.array([1.0,0.4,8.0/35.0]) #order 2 Sonine expansion
n_species=3 #Numbe rof species, electrons will be first, with deuterium second and tritium third, then impurities
n_radial=51
r0=0.
r_final=1.17

Nx=64    #Energy grid for convolution 
# Temporal discretisation
t0 = 0
t_final =20.  
dt = 0.0001
#timesteps at which to save solution
#saveat = diffrax.SaveAt(ts=jnp.array([0,1.e-5,1.e-4,1.e-3,1.e-2,1.e-1,1.e+0,10.0,t_final]))
#saveat = diffrax.SaveAt(ts=jnp.array([0,1.e-5,1.e-4,3.e-4,6.e-4,9.e-4,1.e-3,2.e-3,3.e-3,4.e-3,5.e-3,6.e-3,8.e-3,1.e-2,1.5e-2,2.e-2,t_final]))
ts=jnp.array([0,1.e-5,1.e-4,1.e-3,2.e-3,3.e-3,4.e-3,5.e-3,6.e-3,7.e-3,8.e-3,9.e-3, 1.e-2,1.5e-2,2.e-2,2.5e-2,3.e-2,3.5e-2,1.e-1,1.05e-1,1.1e-1,1.15e-1,1.2e-1,1.25e-1,1.3e-1,1.35e-1,1.4e-1,1.45e-1,1.5e-1,1.55e-1,1.6e-1,1.65e-1,1.7e-1,1.75e-1,1.8e-1,1.85e-1,1.9e-1,1.95e-1,t_final])
# Tolerances for  diffrax 
rtol = 1e-5
atol = 1e-5
#Electric field equation parameters
DEr=jnp.array(0.0) #electric field diffusion coefficient
Er_Relax=jnp.array(0.1)  #Relaxation time of the Electric field
on_OmegaC=jnp.array(0.0)  #Keep zero!!, parameter for testing 
#Turn on/off evolution of quantities (0.0=off)
on_Er=jnp.array(1.0)
on_ne=jnp.array(0.0)
on_nD=jnp.array(0.0)
on_nT=jnp.array(0.0)
on_nHe=jnp.array(0.0)
on_Pe=jnp.array(0.0)
on_PD=jnp.array(0.0)
on_PT=jnp.array(0.0)
chi=jnp.ones(n_species)*0.0065
T_edge=jnp.array(0.7*1.e+3)
edlenPe=jnp.array(0.05)
edlenPD=jnp.array(0.05)
edlenPT=jnp.array(0.05)
n_edge=jnp.array(0.6e+20)

