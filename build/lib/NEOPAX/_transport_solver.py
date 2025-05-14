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
from ._species import Species, gradient_no
from ._neoclassical import get_Neoclassical_Fluxes,get_Neoclassical_Fluxes_With_Momentum_Correction
from ._physics import get_plasma_permitivity, Power_Exchange, get_DT_Reaction, FusionPowerFractionElectrons, P_rad



#Define the source terms, Er and pressures are in r nor rho!!!! This serves to construct vector field for diffrax
@jit
def sources(species: Species,grid,field,database,turbulent,solver_parameters):
    #Evolve species profiles (TODO find a better way for doing the update) 
    #Er_half=interpax.Interpolator1D(field.r_grid,species.Er,extrap=True)(field.r_grid_half)     
    #Get neoclassical fluxes 
    _,Gamma,Q,Upar=get_Neoclassical_Fluxes(species,grid,field,database)
    #     #Apply momentum correction 
    #Gamma = Gamma.at[:,0].set(Gamma.at[:,1].get())
    #Q = Q.at[:,0].set(Q.at[:,1].get())
    Gamma = Gamma.at[:,0].set(0.)
    #Q = Q.at[:,0].set(0.)    
    #Gamma_e_half=interpax.Interpolator1D(field.r_grid,Gamma[0],extrap=True)(field.r_grid_half)
    #Gamma_D_half=interpax.Interpolator1D(field.r_grid,Gamma[1],extrap=True)(field.r_grid_half)
    #Gamma_T_half=interpax.Interpolator1D(field.r_grid,Gamma[2],extrap=True)(field.r_grid_half)
    #Q_e_half=interpax.Interpolator1D(field.r_grid,Q[0],extrap=True)(field.r_grid_half)
    #Q_D_half=interpax.Interpolator1D(field.r_grid,Q[1],extrap=True)(field.r_grid_half)
    #Q_T_half=interpax.Interpolator1D(field.r_grid,Q[2],extrap=True)(field.r_grid_half)
    #Te_half=interpax.Interpolator1D(field.r_grid,species.temperature[0,:],extrap=True)(field.r_grid_half)
    #TD_half=interpax.Interpolator1D(field.r_grid,species.temperature[1,:],extrap=True)(field.r_grid_half)
    #TT_half=interpax.Interpolator1D(field.r_grid,species.temperature[2,:],extrap=True)(field.r_grid_half)
    #dTedr_half=interpax.Interpolator1D(field.r_grid,species.dTdr[0,:],extrap=True)(field.r_grid_half)
    #dTDdr_half=interpax.Interpolator1D(field.r_grid,species.dTdr[1,:],extrap=True)(field.r_grid_half)
    #dTTdr_half=interpax.Interpolator1D(field.r_grid,species.dTdr[2,:],extrap=True)(field.r_grid_half)
    #Permitivity is interpolated in rho
    plasma_permitivity=jax.vmap(get_plasma_permitivity,in_axes=(None,None,0))(species,field,field.r_grid)
    ambi_term=(-Gamma[0]+Gamma[1]+Gamma[2])*elementary_charge*1.e-3/plasma_permitivity#(density*psi_fac)
    #jax.debug.print("AMBI_ER {Er_VALS_AMBI} ", Er_VALS_AMBI=Er.vals)
    #jax.debug.print("AMBI_ER PLS SEE {ambi_term} , {Er_VALS_AMBI}, {Gamma_D}, {Gamma_e}, {Gamma_T}  ", ambi_term=ambi_term, Er_VALS_AMBI=Er.vals, Gamma_D=Gamma_D, Gamma_e=Gamma_e, Gamma_T=Gamma_T)
    SourceEr=solver_parameters.on_Er*solver_parameters.Er_relax*(solver_parameters.DEr*species.diffusion_Er-ambi_term)
    SourceEr=SourceEr.at[0].set(0.)
    ####Derivatives for convective terms
    #PeD=jax.vmap(Power_Exchange,in_axes=(None,None,None,0),out_axes=0)(species,0,1,grid.full_grid_indeces)  
    #PeT=jax.vmap(Power_Exchange,in_axes=(None,None,None,0),out_axes=0)(species,0,2,grid.full_grid_indeces)   
    #PDT=jax.vmap(Power_Exchange,in_axes=(None,None,None,0),out_axes=0)(species,1,2,grid.full_grid_indeces)  
    #DTreactionRate,HeSource,AlphaPower=jax.vmap(get_DT_Reaction,in_axes=(None,0),out_axes=(0,0,0))(species,grid.full_grid_indeces)
    #fraction_Palpha_e=jax.vmap(FusionPowerFractionElectrons,in_axes=(None,0),out_axes=0)(species,grid.full_grid_indeces)
    #PBrems,Zeff=jax.vmap(P_rad,in_axes=(None,0),out_axes=(0,0))(species,grid.full_grid_indeces)
    #PTotal=AlphaPower-PBrems
    #Vprime_full=interpax.Interpolator1D(field.r_grid_half,species.Vprime_half,extrap=True)(field.r_grid)
    #Integrated_Ptotal=jnp.sum(Vprime_full*PTotal*field.dr)
    #Integrated_Ptotal_half=interpax.Interpolator1D(field.r_grid,Integrated_Ptotal,extrap=True)(field.r_grid)
    #Integrated_Ptotal=jnp.sum(species.Vprime_half*PTotal*field.dr)
    #Pe_Exchange=PeD+PeT
    #PD_Exchange=-PeD+PDT
    #PT_Exchange=-PeT-PDT
    #conv_Pe=Vprime_full*(Q[0]+species.temperature[0,:]*Gamma[0]-solver_parameters.chi[0]*species.dTdr[0,:]*1.e+20*jnp.power(Integrated_Ptotal,0.75))*1.e-20*1.e-3
    #conv_PD=Vprime_full*(Q[1]+species.temperature[1,:]*Gamma[1]-solver_parameters.chi[1]*species.dTdr[1,:]*1.e+20*jnp.power(Integrated_Ptotal,0.75))*1.e-20*1.e-3 
    #conv_PT=Vprime_full*(Q[2]+species.temperature[2,:]*Gamma[2]-solver_parameters.chi[2]*species.dTdr[2,:]*1.e+20*jnp.power(Integrated_Ptotal,0.75))*1.e-20*1.e-3
    #conv_Pe_half=interpax.Interpolator1D(field.r_grid,conv_Pe,extrap=True)(field.r_grid_half)
    #conv_PD_half=interpax.Interpolator1D(field.r_grid,conv_PD,extrap=True)(field.r_grid_half)
    #conv_PT_half=interpax.Interpolator1D(field.r_grid,conv_PT,extrap=True)(field.r_grid_half)
    #conv_Pe=species.Vprime_half*(Q_e_half+Te_half*Gamma_e_half-chi[0]*dTedr_half*1.e+20*jnp.power(Integrated_Ptotal,0.75))*1.e-20*1.e-3
    #conv_PD=species.Vprime_half*(Q_D_half+TD_half*Gamma_D_half-chi[1]*dTDdr_half*1.e+20*jnp.power(Integrated_Ptotal,0.75))*1.e-20*1.e-3 
    #conv_PT=species.Vprime_half*(Q_T_half+TT_half*Gamma_T_half-chi[2]*dTTdr_half*1.e+20*jnp.power(Integrated_Ptotal,0.75))*1.e-20*1.e-3
    #Qe_anom=turbulent.Qa_turb[0]
    #QD_anom=turbulent.Qa_turb[1]
    #QT_anom=turbulent.Qa_turb[2]
    #conv_Pe=species.Vprime_half*(Q_e_half+Te_half*Gamma_e_half+Qe_anom)*1.e-20*1.e-3
    #conv_PD=species.Vprime_half*(Q_D_half+TD_half*Gamma_D_half+QD_anom)*1.e-20*1.e-3 
    #conv_PT=species.Vprime_half*(Q_T_half+TT_half*Gamma_T_half+QT_anom)*1.e-20*1.e-3
    #diffusion_Pe=gradient_no(conv_Pe_half,field.dr)*species.overVprime
    #diffusion_PD=gradient_no(conv_PD_half,field.dr)*species.overVprime
    #diffusion_PT=gradient_no(conv_PT_half,field.dr)*species.overVprime
    #diffusion_Pe=gradient_no(conv_Pe,field.dr)*species.overVprime
    #diffusion_PD=gradient_no(conv_PD,field.dr)*species.overVprime
    #diffusion_PT=gradient_no(conv_PT,field.dr)*species.overVprime
    #SourcePe=on_Pe*2./3.*(-species.Er*Gamma[0]*1.e-20+Pe_Exchange+AlphaPower*fraction_Palpha_e-PBrems-diffusion_Pe)
    #SourcePD=on_PD*2./3.*(species.Er*Gamma[1]*1.e-20+PD_Exchange+AlphaPower*(1.-fraction_Palpha_e)*0.5-diffusion_PD)
    #SourcePT=on_PT*2./3.*(species.Er*Gamma[2]*1.e-20+PT_Exchange+AlphaPower*(1.-fraction_Palpha_e)*0.5-diffusion_PT)
    SourcePe=solver_parameters.on_Pe*Gamma[0]
    SourcePD=solver_parameters.on_PD*Gamma[1]
    SourcePT=solver_parameters.on_PT*Gamma[2]
    ###Source_He=on_ne*[(PT_Exchange+AlphaPower*(1.-fraction_Palpha_e)*0.5)]
    ###Source_nD=on_nD*(PT_Exchange+AlphaPower*(1.-fraction_Palpha_e)*0.5)
    ###Source_nT=on_nT*(PT_Exchange+AlphaPower*(1.-fraction_Palpha_e)*0.5)
    SourcenD=solver_parameters.on_nD*Gamma[1]
    SourcenT=solver_parameters.on_nT*Gamma[2] 
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
    #jax.debug.print("AMBI_ER PLS SEE  {ambi_term}  ", ambi_term=SourceEr)
    #Auxiliary Spatial discretizations for convective terms (this is missing the last term of turbulent fluxes)
    return SourceEr,SourcePe,SourcePD,SourcePT,SourcenD,SourcenD,SourcenT

@jit
def vector_field(t, y,args):
    Er,Pe,PD,PT,ne,nD,nT=y
    Initial_Species,grid,field,database,turbulent,solver_parameters=args
    #print('Here')
    ##jax.debug.print(" {t} ", t=t)
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
    Pe_new=Pe_new.at[0].set((4.*Pe_new.at[1].get()-Pe_new.at[2].get()-fr*2*field.dr)/(3.0))   
    Pe_new=Pe_new.at[-1].set(Initial_Species.n_edge[0]*Initial_Species.T_edge[0]*1.e-20*1.e-3)
    #Pe_boundary=Pe_boundary.at[-1].set((4.*Pe_boundary.at[-2].get()-Pe_boundary.at[-3].get())/(3.0+2*edlenPe*dr))   
    PD_new=PD
    PD_new=PD_new.at[0].set((4.*PD_new.at[1].get()-PD_new.at[2].get()-fr*2*field.dr)/(3.0))   
    PD_new=PD_new.at[-1].set(Initial_Species.n_edge[1]*Initial_Species.T_edge[1]*1.e-20*1.e-3)
    #PD_boundary=PD_boundary.at[-1].set((4.*PD_boundary.at[-2].get()-PD_boundary.at[-3].get())/(3.0+2*edlenPD*dr))
    PT_new=PT
    PT_new=PT_new.at[0].set((4.*PT_new.at[1].get()-PT_new.at[2].get()-fr*2*field.dr)/(3.0))   
    PT_new=PT_new.at[-1].set(Initial_Species.n_edge[2]*Initial_Species.T_edge[2]*1.e-20*1.e-3)
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
    species_new=Species(Initial_Species.number_species,
                        Initial_Species.radial_points,
                        Initial_Species.species_indeces,
                        Initial_Species.mass_mp,
                        Initial_Species.charge_qp,
                        temperature,
                        density,
                        Er,
                        field.r_grid,
                        field.r_grid_half,
                        field.dr,
                        field.Vprime_half,
                        field.overVprime,
                        Initial_Species.n_edge,
                        Initial_Species.T_edge)

    #jax.debug.print("Pe {Pe} ", Pe=Pe_new/ne_new)
    #jax.debug.print("Er {Er} ", Er=Er)
    #hcb.id_print((t,Pe.vals))
    return sources(species_new,grid,field,database,turbulent,solver_parameters)



#Define the source terms, Er and pressures are in r nor rho!!!! This serves to construct vector field for diffrax
@jit
def sources_momentum_correction(species: Species,grid,field,database,turbulent,solver_parameters):
    #Evolve species profiles (TODO find a better way for doing the update) 
    #Er_half=interpax.Interpolator1D(field.r_grid,species.Er,extrap=True)(field.r_grid_half)     
    #Get neoclassical fluxes 
    _,Gamma,Q,Upar=get_Neoclassical_Fluxes_With_Momentum_Correction(species,grid,field,database)
    #     #Apply momentum correction 
    #Gamma = Gamma.at[:,0].set(Gamma.at[:,1].get())
    #Q = Q.at[:,0].set(Q.at[:,1].get())
    Gamma = Gamma.at[:,0].set(0.)
    #Q = Q.at[:,0].set(0.)    
    #Gamma_e_half=interpax.Interpolator1D(field.r_grid,Gamma[0],extrap=True)(field.r_grid_half)
    #Gamma_D_half=interpax.Interpolator1D(field.r_grid,Gamma[1],extrap=True)(field.r_grid_half)
    #Gamma_T_half=interpax.Interpolator1D(field.r_grid,Gamma[2],extrap=True)(field.r_grid_half)
    #Q_e_half=interpax.Interpolator1D(field.r_grid,Q[0],extrap=True)(field.r_grid_half)
    #Q_D_half=interpax.Interpolator1D(field.r_grid,Q[1],extrap=True)(field.r_grid_half)
    #Q_T_half=interpax.Interpolator1D(field.r_grid,Q[2],extrap=True)(field.r_grid_half)
    #Te_half=interpax.Interpolator1D(field.r_grid,species.temperature[0,:],extrap=True)(field.r_grid_half)
    #TD_half=interpax.Interpolator1D(field.r_grid,species.temperature[1,:],extrap=True)(field.r_grid_half)
    #TT_half=interpax.Interpolator1D(field.r_grid,species.temperature[2,:],extrap=True)(field.r_grid_half)
    #dTedr_half=interpax.Interpolator1D(field.r_grid,species.dTdr[0,:],extrap=True)(field.r_grid_half)
    #dTDdr_half=interpax.Interpolator1D(field.r_grid,species.dTdr[1,:],extrap=True)(field.r_grid_half)
    #dTTdr_half=interpax.Interpolator1D(field.r_grid,species.dTdr[2,:],extrap=True)(field.r_grid_half)
    #Permitivity is interpolated in rho
    plasma_permitivity=jax.vmap(get_plasma_permitivity,in_axes=(None,None,0))(species,field,field.r_grid)
    ambi_term=(-Gamma[0]+Gamma[1]+Gamma[2])*elementary_charge*1.e-3/plasma_permitivity#(density*psi_fac)
    #jax.debug.print("AMBI_ER {Er_VALS_AMBI} ", Er_VALS_AMBI=Er.vals)
    #jax.debug.print("AMBI_ER PLS SEE {ambi_term} , {Er_VALS_AMBI}, {Gamma_D}, {Gamma_e}, {Gamma_T}  ", ambi_term=ambi_term, Er_VALS_AMBI=Er.vals, Gamma_D=Gamma_D, Gamma_e=Gamma_e, Gamma_T=Gamma_T)
    SourceEr=solver_parameters.on_Er*solver_parameters.Er_relax*(solver_parameters.DEr*species.diffusion_Er-ambi_term)
    SourceEr=SourceEr.at[0].set(0.)
    ####Derivatives for convective terms
    #PeD=jax.vmap(Power_Exchange,in_axes=(None,None,None,0),out_axes=0)(species,0,1,grid.full_grid_indeces)  
    #PeT=jax.vmap(Power_Exchange,in_axes=(None,None,None,0),out_axes=0)(species,0,2,grid.full_grid_indeces)   
    #PDT=jax.vmap(Power_Exchange,in_axes=(None,None,None,0),out_axes=0)(species,1,2,grid.full_grid_indeces)  
    #DTreactionRate,HeSource,AlphaPower=jax.vmap(get_DT_Reaction,in_axes=(None,0),out_axes=(0,0,0))(species,grid.full_grid_indeces)
    #fraction_Palpha_e=jax.vmap(FusionPowerFractionElectrons,in_axes=(None,0),out_axes=0)(species,grid.full_grid_indeces)
    #PBrems,Zeff=jax.vmap(P_rad,in_axes=(None,0),out_axes=(0,0))(species,grid.full_grid_indeces)
    #PTotal=AlphaPower-PBrems
    #Vprime_full=interpax.Interpolator1D(field.r_grid_half,species.Vprime_half,extrap=True)(field.r_grid)
    #Integrated_Ptotal=jnp.sum(Vprime_full*PTotal*field.dr)
    #Integrated_Ptotal_half=interpax.Interpolator1D(field.r_grid,Integrated_Ptotal,extrap=True)(field.r_grid)
    #Integrated_Ptotal=jnp.sum(species.Vprime_half*PTotal*field.dr)
    #Pe_Exchange=PeD+PeT
    #PD_Exchange=-PeD+PDT
    #PT_Exchange=-PeT-PDT
    #conv_Pe=Vprime_full*(Q[0]+species.temperature[0,:]*Gamma[0]-solver_parameters.chi[0]*species.dTdr[0,:]*1.e+20*jnp.power(Integrated_Ptotal,0.75))*1.e-20*1.e-3
    #conv_PD=Vprime_full*(Q[1]+species.temperature[1,:]*Gamma[1]-solver_parameters.chi[1]*species.dTdr[1,:]*1.e+20*jnp.power(Integrated_Ptotal,0.75))*1.e-20*1.e-3 
    #conv_PT=Vprime_full*(Q[2]+species.temperature[2,:]*Gamma[2]-solver_parameters.chi[2]*species.dTdr[2,:]*1.e+20*jnp.power(Integrated_Ptotal,0.75))*1.e-20*1.e-3
    #conv_Pe_half=interpax.Interpolator1D(field.r_grid,conv_Pe,extrap=True)(field.r_grid_half)
    #conv_PD_half=interpax.Interpolator1D(field.r_grid,conv_PD,extrap=True)(field.r_grid_half)
    #conv_PT_half=interpax.Interpolator1D(field.r_grid,conv_PT,extrap=True)(field.r_grid_half)
    #conv_Pe=species.Vprime_half*(Q_e_half+Te_half*Gamma_e_half-chi[0]*dTedr_half*1.e+20*jnp.power(Integrated_Ptotal,0.75))*1.e-20*1.e-3
    #conv_PD=species.Vprime_half*(Q_D_half+TD_half*Gamma_D_half-chi[1]*dTDdr_half*1.e+20*jnp.power(Integrated_Ptotal,0.75))*1.e-20*1.e-3 
    #conv_PT=species.Vprime_half*(Q_T_half+TT_half*Gamma_T_half-chi[2]*dTTdr_half*1.e+20*jnp.power(Integrated_Ptotal,0.75))*1.e-20*1.e-3
    #Qe_anom=turbulent.Qa_turb[0]
    #QD_anom=turbulent.Qa_turb[1]
    #QT_anom=turbulent.Qa_turb[2]
    #conv_Pe=species.Vprime_half*(Q_e_half+Te_half*Gamma_e_half+Qe_anom)*1.e-20*1.e-3
    #conv_PD=species.Vprime_half*(Q_D_half+TD_half*Gamma_D_half+QD_anom)*1.e-20*1.e-3 
    #conv_PT=species.Vprime_half*(Q_T_half+TT_half*Gamma_T_half+QT_anom)*1.e-20*1.e-3
    #diffusion_Pe=gradient_no(conv_Pe_half,field.dr)*species.overVprime
    #diffusion_PD=gradient_no(conv_PD_half,field.dr)*species.overVprime
    #diffusion_PT=gradient_no(conv_PT_half,field.dr)*species.overVprime
    #diffusion_Pe=gradient_no(conv_Pe,field.dr)*species.overVprime
    #diffusion_PD=gradient_no(conv_PD,field.dr)*species.overVprime
    #diffusion_PT=gradient_no(conv_PT,field.dr)*species.overVprime
    #SourcePe=on_Pe*2./3.*(-species.Er*Gamma[0]*1.e-20+Pe_Exchange+AlphaPower*fraction_Palpha_e-PBrems-diffusion_Pe)
    #SourcePD=on_PD*2./3.*(species.Er*Gamma[1]*1.e-20+PD_Exchange+AlphaPower*(1.-fraction_Palpha_e)*0.5-diffusion_PD)
    #SourcePT=on_PT*2./3.*(species.Er*Gamma[2]*1.e-20+PT_Exchange+AlphaPower*(1.-fraction_Palpha_e)*0.5-diffusion_PT)
    SourcePe=solver_parameters.on_Pe*Gamma[0]
    SourcePD=solver_parameters.on_PD*Gamma[1]
    SourcePT=solver_parameters.on_PT*Gamma[2]
    ###Source_He=on_ne*[(PT_Exchange+AlphaPower*(1.-fraction_Palpha_e)*0.5)]
    ###Source_nD=on_nD*(PT_Exchange+AlphaPower*(1.-fraction_Palpha_e)*0.5)
    ###Source_nT=on_nT*(PT_Exchange+AlphaPower*(1.-fraction_Palpha_e)*0.5)
    SourcenD=solver_parameters.on_nD*Gamma[1]
    SourcenT=solver_parameters.on_nT*Gamma[2] 
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
    #jax.debug.print("AMBI_ER PLS SEE  {ambi_term}  ", ambi_term=SourceEr)
    #Auxiliary Spatial discretizations for convective terms (this is missing the last term of turbulent fluxes)
    return SourceEr,SourcePe,SourcePD,SourcePT,SourcenD,SourcenD,SourcenT

@jit
def vector_field_momentum_correction(t, y,args):
    Er,Pe,PD,PT,ne,nD,nT=y
    Initial_Species,grid,field,database,turbulent,solver_parameters=args
    #print('Here')
    ##jax.debug.print(" {t} ", t=t)
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
    Pe_new=Pe_new.at[0].set((4.*Pe_new.at[1].get()-Pe_new.at[2].get()-fr*2*field.dr)/(3.0))   
    Pe_new=Pe_new.at[-1].set(Initial_Species.n_edge[0]*Initial_Species.T_edge[0]*1.e-20*1.e-3)
    #Pe_boundary=Pe_boundary.at[-1].set((4.*Pe_boundary.at[-2].get()-Pe_boundary.at[-3].get())/(3.0+2*edlenPe*dr))   
    PD_new=PD
    PD_new=PD_new.at[0].set((4.*PD_new.at[1].get()-PD_new.at[2].get()-fr*2*field.dr)/(3.0))   
    PD_new=PD_new.at[-1].set(Initial_Species.n_edge[1]*Initial_Species.T_edge[1]*1.e-20*1.e-3)
    #PD_boundary=PD_boundary.at[-1].set((4.*PD_boundary.at[-2].get()-PD_boundary.at[-3].get())/(3.0+2*edlenPD*dr))
    PT_new=PT
    PT_new=PT_new.at[0].set((4.*PT_new.at[1].get()-PT_new.at[2].get()-fr*2*field.dr)/(3.0))   
    PT_new=PT_new.at[-1].set(Initial_Species.n_edge[2]*Initial_Species.T_edge[2]*1.e-20*1.e-3)
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
    species_new=Species(Initial_Species.number_species,
                        Initial_Species.radial_points,
                        Initial_Species.species_indeces,
                        Initial_Species.mass_mp,
                        Initial_Species.charge_qp,
                        temperature,
                        density,
                        Er,
                        field.r_grid,
                        field.r_grid_half,
                        field.dr,
                        field.Vprime_half,
                        field.overVprime,
                        Initial_Species.n_edge,
                        Initial_Species.T_edge)

    #jax.debug.print("Pe {Pe} ", Pe=Pe_new/ne_new)
    #jax.debug.print("Er {Er} ", Er=Er)
    #hcb.id_print((t,Pe.vals))
    return sources_momentum_correction(species_new,grid,field,database,turbulent,solver_parameters)











def solve_transport_equations(y0,args):
    solver_parameters=args[-1]    
    if solver_parameters.momentum_correction_flag==True: 
        term = diffrax.ODETerm(vector_field_momentum_correction)
    else:
        term = diffrax.ODETerm(vector_field)

    saveat = diffrax.SaveAt(ts=solver_parameters.ts_list)
    stepsize_controller = diffrax.PIDController(pcoeff=0.3, icoeff=0.4, rtol=solver_parameters.rtol, atol=solver_parameters.atol, dtmax=None,dtmin=None)
    #solver = diffrax.Tsit5()
    solver=diffrax.Kvaerno5()
    #solver=diffrax.Kvaerno5(root_finder=diffrax._root_finder.VeryChord())
    sol = diffrax.diffeqsolve(
        term,
        solver,
        solver_parameters.t0,
        solver_parameters.t_final,
        solver_parameters.dt,
        y0,
        args=args,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=None,
        #progress_meter=diffrax.TqdmProgressMeter(),
    )
    return sol


