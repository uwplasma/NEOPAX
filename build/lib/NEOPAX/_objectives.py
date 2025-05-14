import h5py as h5 
import numpy as np
import jax
from jax import config, jit
# to use higher precision
config.update("jax_enable_x64", True)
import os
import jax.numpy as jnp
from jax import jit
import matplotlib.pyplot as plt
import interpax
from ._constants import Boltzmann, elementary_charge, epsilon_0, hbar, proton_mass
import optimistix as optx
from typing import Callable
import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Float  # https://github.com/google/jaxtyping
import jax.experimental.host_callback as hcb
from ._species import collisionality


#Module for neoclassical objective functions definition, using NEOPAX, MONKEX, etc...
def Electron_root_from_Monkes(species,grid,field,nl):
    #Pitch angle resolution
    nl=40
    #Define the electric field to use (checking this to match sfincs!!!!!! this needs to be checked)
    Er=17.8e+3#*(2./1.19*0.29)

    D11i=np.zeros(len(v_ion))
    D11e=np.zeros(len(v_electron))

    #get collisionalities
    #Use neoclassical to redefine this
    L11i_fac=ions.density(r=0.29)*2./jnp.sqrt(jnp.pi)*(ions.species.mass/ions.species.charge)**2*vth_ion**3
    L11e_fac=electrons.density(r=0.29)*2./jnp.sqrt(jnp.pi)*(electrons.species.mass/electrons.species.charge)**2*vth_electron**3

    #Calculate Dij over the x grid for ions and electrons     
    Dij_ions, _, _ = jax.vmap(monkes.monoenergetic_dke_solve_internal,in_axes=(None,None,0,None))(field, Er, nu_v_ions, nl)
    Dij_electrons, _, _ = jax.vmap(monkes.monoenergetic_dke_solve_internal,in_axes=(None,None,0,None))(field, Er, nu_v_electrons, nl)


    #Now to calculate L11 we just need to use the xWeights 
    L11i=L11i_fac*jnp.sum(grid.L11i_weight*grid.xWeights*Dij_ions.at[:,0,0].get())
    L11e=L11e_fac*jnp.sum(grid.L11e_weight*grid.xWeights*Dij_electrons.at[:,0,0].get())

    return L11i/L11e


