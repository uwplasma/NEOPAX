import jax.numpy as jnp
from jax import config
# to use higher precision
config.update("jax_enable_x64", True)
from jax import jit

import equinox as eqx
from jaxtyping import Array, Float, Int  # https://github.com/google/jaxtyping


class Solver_Parameters(eqx.Module):
    #Solver parameters class
    momentum_correction_flag: int
    # Temporal discretisation
    t0: float
    t_final: float  
    dt: float
    #timesteps at which to save solution
    ts_list: Float[Array, "..."]
    # Tolerances for  diffrax solver
    rtol: float
    atol: float
    #Electric field equation parameters
    DEr: float #electric field diffusion coefficient
    Er_relax: float  #Relaxation time of the Electric field
    on_OmegaC: float   #Keep zero!!, parameter for testing  
    #Turn on/off evolution of quantities (0.0=off)
    on_Er: float 
    on_ne: float
    on_nD: float
    on_nT: float
    on_nHe: float
    on_Pe: float
    on_PD: float
    on_PT: float
    #Heat diffusivities
    chi: Float[Array, "..."]  
     
    def __init__(self,t0=None,t_final=None,dt=None,ts_list=None,rtol=None,
                        atol=None,momentum_correction_flag=None,DEr=None,Er_relax=None,
                        on_Er=None,on_ne=None,on_nD=None,on_nT=None,on_nHe=None,on_Pe=None,on_PD=None,on_PT=None,
                        chi=None,on_OmegaC=None):
        

        if t0 is None:
            self.t0 = 0.
        else:        
            self.t0=t0

        if t_final is None:
            self.t_final = 20.
        else:        
            self.t_final=t_final            

        if dt is None:
            self.dt = 0.0001
        else:        
            self.dt=dt   

        if ts_list is None:
            self.ts_list = jnp.array([0,1.e-5,1.e-4,1.e-3,2.e-3,3.e-3,4.e-3,5.e-3,
                                      6.e-3,7.e-3,8.e-3,9.e-3, 1.e-2,1.5e-2,2.e-2,2.5e-2,3.e-2,3.5e-2,1.e-1,1.05e-1,1.1e-1,
                                      1.15e-1,1.2e-1,1.25e-1,1.3e-1,1.35e-1,1.4e-1,1.45e-1,1.5e-1,1.55e-1,1.6e-1,1.65e-1,1.7e-1,
                                      1.75e-1,1.8e-1,1.85e-1,1.9e-1,1.95e-1,20.])
        else:        
            self.ts_list=ts_list   


        if atol is None:
            self.atol = 1.e-5
        else:
            self.atol=atol  

        if rtol is None:
            self.rtol = 1.e-5
        else:
            self.rtol=rtol 


        if momentum_correction_flag is None:
            self.momentum_correction_flag = False
        else:
            self.momentum_correction_flag=momentum_correction_flag

        if DEr is None:
            self.DEr = 2.
        else:
            self.DEr=DEr 


        if Er_relax is None:
            self.Er_relax = 0.1
        else:
            self.Er_relax=Er_relax

        if on_Er is None:
            self.on_Er = 1
        else:
            self.on_Er=on_Er

        if on_ne is None:
            self.on_ne = 0
        else:
            self.on_ne=on_ne

        if on_nD is None:
            self.on_nD = 0
        else:
            self.on_nD=on_nD

        if on_nT is None:
            self.on_nT = 0
        else:
            self.on_nT=on_nT

        if on_nHe is None:
            self.on_nHe = 0
        else:
            self.on_nHe=on_nHe

        if on_Pe is None:
            self.on_Pe = 0
        else:
            self.on_Pe=on_Pe

        if on_PD is None:
            self.on_PD = 0
        else:
            self.on_PD=on_PD

        if on_PT is None:
            self.on_PT = 0
        else:
            self.on_PT=on_PT
                                                
        if chi is None:
            self.chi=jnp.ones(3)*0.0065
        else:
            self.chi=chi

        if on_OmegaC is None:
            self.on_OmegaC = 0
        else:
            self.on_OmegaC=on_OmegaC
