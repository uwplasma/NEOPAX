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
from scipy.constants import Boltzmann, elementary_charge, epsilon_0, hbar, proton_mass
import optimistix as optx
from typing import Callable
import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax.lax as lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import Array, Float  # https://github.com/google/jaxtyping
import jax.experimental.host_callback as hcb
from _species import Species
#from _field import a_b, dVdr, iota
from _grid import rho_grid, rho_grid_half, r_grid, r_grid_half,species_indeces,full_grid_indeces,dr,a_b
from _neoclassical import get_Neoclassical_Fluxes
from _parameters import n_species,n_radial, on_Er, on_Pe, on_PD,on_PT, on_nD,on_nT,on_nHe, DEr, Er_Relax
from _parameters import n_edge, T_edge
from _parameters import t0, t_final, ts,rtol,atol, dt 
from _species import Species 
from _physics import get_plasma_permitivity


#Define the source terms, Er and pressures are in r nor rho!!!! This serves to construct vector field for diffrax
@jit
def sources(species: Species):
    #Evolve species profiles (TODO find a better way for doing the update) 
    Er_half=interpax.Interpolator1D(r_grid,species.Er,extrap=True)(r_grid_half)     
    #Get neoclassical fluxes 
    Lij,Gamma,Q,Upar=get_Neoclassical_Fluxes(species,species_indeces,full_grid_indeces)
    #     #Apply momentum correction 
    Gamma = Gamma.at[:,0].set(0.0)
    ##Q_e = Q_e.at[0].set(0.0)
    ##Q_D = Q_D.at[0].set(0.0)
    ##Q_T = Q_T.at[0].set(0.0)
    ##Gamma_e_half=interpax.Interpolator1D(r_grid,Gamma_e,extrap=True)(r_grid_half)
    ##Gamma_D_half=interpax.Interpolator1D(r_grid,Gamma_D,extrap=True)(r_grid_half)
    ##Gamma_T_half=interpax.Interpolator1D(r_grid,Gamma_T,extrap=True)(r_grid_half)
    ##Q_e_half=interpax.Interpolator1D(r_grid,Gamma_e,extrap=True)(r_grid_half)
    ##Q_D_half=interpax.Interpolator1D(r_grid,Gamma_D,extrap=True)(r_grid_half)
    ##Q_T_half=interpax.Interpolator1D(r_grid,Gamma_T,extrap=True)(r_grid_half)
    #Permitivity is interpolated in rho
    plasma_permitivity=jax.vmap(get_plasma_permitivity,in_axes=(None,0))(species,r_grid)
    ambi_term=(-Gamma[0]+Gamma[1]+Gamma[2])*elementary_charge*1.e-3/plasma_permitivity#(density*psi_fac)
    #ambi_term=ambi_term.at[-1].set(1.82104167e+05)
    #ambi_term=ambi_term.at[-1].set(2.65723598e+05)
    #ambi_term=ambi_term*elementary_charge*1.e-3/(4.21* 1.e+20*proton_mass)#*B00**2#(density*psi_fac)
    #jax.debug.print("AMBI_ER {Er_VALS_AMBI} ", Er_VALS_AMBI=Er.vals)
    #jax.debug.print("AMBI_ER PLS SEE {ambi_term} , {Er_VALS_AMBI}, {Gamma_D}, {Gamma_e}, {Gamma_T}  ", ambi_term=ambi_term, Er_VALS_AMBI=Er.vals, Gamma_D=Gamma_D, Gamma_e=Gamma_e, Gamma_T=Gamma_T)
    SourceEr=on_Er*Er_Relax*(DEr*species.diffusion_Er-ambi_term)
    SourceEr=SourceEr.at[0].set(0.)
    ####Derivatives for convective terms
    ###PeD=jax.vmap(Power_Exchange,in_axes=(0,None,None),out_axes=0)(r_grid,global_species[0],global_species[1])
    ###PeT=jax.vmap(Power_Exchange,in_axes=(0,None,None),out_axes=0)(r_grid,global_species[0],global_species[2])  
    ###PDT=jax.vmap(Power_Exchange,in_axes=(0,None,None),out_axes=0)(r_grid,global_species[1],global_species[2])  
    ###DTreactionRate,HeSource,AlphaPower=jax.vmap(get_DT_Reaction,in_axes=(0,None,None),out_axes=(0,0,0))(r_grid,global_species[1],global_species[2])
    ###fraction_Palpha_e=jax.vmap(FusionPowerFractionElectrons,in_axes=(0,None),out_axes=0)(r_grid,global_species[0])
    ###PBrems,Zeff=jax.vmap(P_rad,in_axes=(0,None,None,None),out_axes=(0,0))(r_grid,global_species[0],global_species[1],global_species[2])
    ###PTotal=AlphaPower-PBrems
    ###Integrated_Ptotal=jnp.sum(Vprime_half*PTotal*dr)
    ###Pe_Exchange=PeD+PeT
    ###PD_Exchange=-PeD+PDT
    ###PT_Exchange=-PeT-PDT
    ####conv_Pe=SpatialDiscretisation(r0,r_final,Vprime_half*(Q_e_half+Te_new(r_grid_half)*Gamma_e_half-chi_e*dTedr.vals*1.e+20*jnp.power(Integrated_Ptotal,0.75))*1.e-20*1.e-3)
    ####onv_PD=SpatialDiscretisation(r0,r_final,Vprime_half*(Q_D_half+TD_new(r_grid_half)*Gamma_D_half-chi_D*dTDdr.vals*1.e+20*jnp.power(Integrated_Ptotal,0.75))*1.e-20*1.e-3) 
    ###conv_PT=SpatialDiscretisation(r0,r_final,Vprime_half*(Q_T_half+TT_new(r_grid_half)*Gamma_T_half-chi_T*dTTdr.vals*1.e+20*jnp.power(Integrated_Ptotal,0.75))*1.e-20*1.e-3)
    ###diffusion_Pe=0.0#gradient_no(conv_Pe).vals*overVprime
    ###diffusion_PD=0.0#gradient_no(conv_PD).vals*overVprime
    ###diffusion_PT=0.0#gradient_no(conv_PT).vals*overVprime
    ###SourcePe=on_Te*2./3.*(-Er.vals*Gamma_e*1.e-20+Pe_Exchange+AlphaPower*fraction_Palpha_e-PBrems-diffusion_Pe)
    ###SourcePD=on_TD*2./3.*(Er.vals*Gamma_D*1.e-20+PD_Exchange+AlphaPower*(1.-fraction_Palpha_e)*0.5-diffusion_PD)
    ###SourcePT=on_TT*2./3.*(Er.vals*Gamma_T*1.e-20+PT_Exchange+AlphaPower*(1.-fraction_Palpha_e)*0.5-diffusion_PT)
    SourcePe=on_Pe*Gamma[0]
    SourcePD=on_PD*Gamma[1]
    SourcePT=on_PT*Gamma[2]
    ###Source_He=on_ne*[(PT_Exchange+AlphaPower*(1.-fraction_Palpha_e)*0.5)]
    ###Source_nD=on_nD*(PT_Exchange+AlphaPower*(1.-fraction_Palpha_e)*0.5)
    ###Source_nT=on_nT*(PT_Exchange+AlphaPower*(1.-fraction_Palpha_e)*0.5)
    SourcenD=on_nD*Gamma[1]
    SourcenT=on_nT*Gamma[2] 
    #Apply drichlet for pressure and density
    SourcePe=SourcePe.at[0].set(0.)
    SourcePD=SourcePD.at[0].set(0.)
    SourcePT=SourcePT.at[0].set(0.)
    SourcePe=SourcePe.at[-1].set(0.)
    SourcePD=SourcePD.at[-1].set(0.)
    SourcePT=SourcePT.at[-1].set(0.)
    #Sourcene=Sourcene.at[-1].set(0.)
    #SourcenD=SourcenD.at[-1].set(0.)
    #SourcenT=SourcePT.at[-1].set(0.)
    #SourcePe=on_Te*2./3.*(diffusion_Pe-Er.vals*Gamma_e*1.e-20)
    #SourcePD=on_TD*2./3.*(diffusion_PD+Er.vals*Gamma_D*1.e-20)
    #SourcePT=on_TT*2./3.*(diffusion_PT+Er.vals*Gamma_T*1.e-20)
    #jax.debug.print("Pe {Source_Pe} ", Source_Pe=Pe.vals/ne_initial.vals)
    #jax.debug.print("Er {Er} ", Er=Er.vals)
    #jax.debug.print("PD {Source_PD} ", Source_PD=PD.vals/nD_initial.vals)
    #jax.debug.print("PT {Source_PT} ", Source_PT=PT.vals/nT_initial.vals)
    #jax.debug.print("AMBI_ER PLS SEE  {SourcePe}, {SourcePD}, {SourcePT}  ", ambi_term=SourceEr,  SourcePe=SourcePe, SourcePD=SourcePD, SourcePT=SourcePT)
    #Auxiliary Spatial discretizations for convective terms (this is missing the last term of turbulent fluxes)
    return SourceEr,SourcePe,SourcePD,SourcePT,SourcenD,SourcenD,SourcenT

@jit
def vector_field(t, y,args):
    Er,Pe,PD,PT,ne,nD,nT=y
    n_r,n_s,mass,charge=args
    #print('Here')
    jax.debug.print(" {t} ", t=t)
    #jax.debug.print(" {dt} ", dt=dt)
    #print('Here')
    #Er.vals=Er.vals.at[-1].set(-48.)
    #We can apply boundary conditions here for now until I find a ,ore optimised way, example:
    #Er_boundary=Er.vals
    #Er_boundary=Er_boundary.at[-1].set(Er_edge)
    #Er=SpatialDiscretisation(r0,r_final,Er_boundary)
    fr=0.0
    #Boundary conditions for densities
    ##NHe_boundary=NHe.vals
    ##NHe_boundary=NHe_boundary.at[0].set((4.*NHe_boundary.at[1].get()-NHe_boundary.at[2].get()-fr*2*dr)/(3.0))   
    ##NHe_boundary=NHe_boundary.at[-1].set(NHe_edge*1.e-20*1.e-3)
    ##NHe_boundary=NHe_boundary.at[-1].set((4.*NHe_boundary.at[-2].get()-NHe_boundary.at[-3].get())/(3.0+2*edlenPe*dr))   
    ##NHe_new=SpatialDiscretisation(r0,r_final,NHe_boundary)
    ##ND_boundary=PD.vals
    ##ND_boundary=ND_boundary.at[-1].set(ND_edge*1.e-20*1.e-3)
    ##ND_boundary=ND_boundary.at[-1].set((4.*ND_boundary.at[-2].get()-ND_boundary.at[-3].get())/(3.0+2*edlenPD*dr))
    ##ND_boundary=ND_boundary.at[0].set((4.*ND_boundary.at[1].get()-ND_boundary.at[2].get()-fr*2*dr)/(3.0))   
    ##ND_new=SpatialDiscretisation(r0,r_final,ND_boundary)
    ##NT_boundary=NT.vals
    ##NT_boundary=NT_boundary.at[-1].set(NT_edge*1.e-20*1.e-3)
    ##NT_boundary=NT_boundary.at[0].set((4.*NT_boundary.at[1].get()-NT_boundary.at[2].get()-fr*2*dr)/(3.0))   
    ##NT_boundary=NT_boundary.at[-1].set((4.*NT_boundary.at[-2].get()-NT_boundary.at[-3].get())/(3.0+2*edlenPT*dr))
    ##NT_new=SpatialDiscretisation(r0,r_final,NT_boundary)
    #Updat ne from quasi-neutrality
    ###ne_sol=Z*ND+Z*NT+Z*NHe
    #Boundary conditions Pressures
    Pe_new=Pe
    ##Pe_new=Pe_new.at[0].set((4.*Pe_new.at[1].get()-Pe_new.at[2].get()-fr*2*dr)/(3.0))   
    ##Pe_new=Pe_new.at[-1].set(n_edge*T_edge*1.e-20*1.e-3)
    #Pe_boundary=Pe_boundary.at[-1].set((4.*Pe_boundary.at[-2].get()-Pe_boundary.at[-3].get())/(3.0+2*edlenPe*dr))   
    PD_new=PD
    #P#D_new=PD_new.at[0].set((4.*PD_new.at[1].get()-PD_new.at[2].get()-fr*2*dr)/(3.0))   
    #P#D_new=PD_new.at[-1].set(n_edge*T_edge*1.e-20*1.e-3)
    #PD_boundary=PD_boundary.at[-1].set((4.*PD_boundary.at[-2].get()-PD_boundary.at[-3].get())/(3.0+2*edlenPD*dr))
    PT_new=PT
    ##PT_new=PT_new.at[0].set((4.*PT_new.at[1].get()-PT_new.at[2].get()-fr*2*dr)/(3.0))   
    ##PT_new=PT_new.at[-1].set(n_edge*T_edge*1.e-20*1.e-3)
    #PT_boundary=PT_boundary.at[-1].set((4.*PT_boundary.at[-2].get()-PT_boundary.at[-3].get())/(3.0+2*edlenPT*dr))
    ne_new=ne
    nD_new=nD
    nT_new=nT
    #temperature=jnp.zeros((n_r,n_species))
    #density=jnp.zeros((n_r,n_species))
    #temperature=temperature.at[0,:].set(Pe_new/ne_new)
    #temperature=temperature.at[1,:].set(PD_new/nD_new)
    #temperature=temperature.at[2,:].set(PT_new/nT_new)
    temperature=jnp.vstack([Pe_new/ne_new,PD_new/nD_new,PT_new/nT_new])
    temperature=temperature*1.e+3
    #density=density.at[0,:].set(ne_new)
    #density=density.at[1,:].set(nD_new)
    #density=density.at[2,:].set(nT_new)
    density=jnp.vstack([ne_new,nD_new,nT_new])
    density=density*1.e+20
    species_new=Species(n_r,n_s,mass,charge,temperature,density,Er)
    jax.debug.print("Pe {Pe} ", Pe=Pe_new/ne_new)
    jax.debug.print("Er {Er} ", Er=Er)
    #hcb.id_print((t,Pe.vals))
    return sources(species_new)

def solve_transport_equations(y0,args):
    term = diffrax.ODETerm(vector_field)
    saveat = diffrax.SaveAt(ts=ts)
    stepsize_controller = diffrax.PIDController(pcoeff=0.3, icoeff=0.4, rtol=rtol, atol=atol, dtmax=None,dtmin=None)
    #solver = diffrax.Tsit5()
    solver=diffrax.Kvaerno5()
    #solver=diffrax.Kvaerno5(root_finder=diffrax._root_finder.VeryChord())
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0,
        t_final,
        dt,
        y0,
        args=args,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=None
    )
    return sol


