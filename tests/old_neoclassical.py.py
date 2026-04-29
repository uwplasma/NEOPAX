import h5py as h5
import jax.numpy as jnp
import jax
from jax import config
# to use higher precision
config.update("jax_enable_x64", True)
from jax import jit
import jax.numpy as jnp
import lineax
from ._constants import elementary_charge, epsilon_0
from ._species import collisionality
from ._interpolators import get_Dij


@jit
def get_Lij_matrix(species,grid,field,database,index_species,r_index):
    #def scan_Dij(_,pair):
    #    nu_dummy,Er_dummy=pair
    #    return None,get_Dij(r_grid[r_index], nu_dummy, Er_dummy)
    
    #For no momentum correction, Lij is just a 3 x 3 matrix for each species at each radial position
    Lij=jnp.zeros((3,3))
    #Thermal velocities
    vth_a=species.v_thermal[index_species,r_index]
    #velocities for convolution
    v_new_a=grid.v_norm*vth_a
    #Er's for convolution
    #Omega_c=on_OmegaC*species.charge[index_species]*B0[r_index]/species.mass[index_species]
    Er_vnew_a=species.Er[r_index]*1.e+3/v_new_a#-on_OmegaC*B0prime[r_index]/(2.*Omega_c)*v_new_a
    n=species.density[index_species,r_index]
    T=species.temperature[index_species,r_index]
    #same species collisionalities
    ##nu_vnew_T=collisionality(species_a_loc, v_new_a, *global_species_loc)/v_new_a    
    nu_vnew_a=collisionality(index_species, species,v_new_a, r_index)/v_new_a 
    #L11_fac_T=nT*2./jnp.sqrt(jnp.pi)*(tritium_loc.species.mass/tritium_loc.species.charge)**2*vth_T**3
    #L11_fac_a=n/jnp.sqrt(jnp.pi)*(species.mass[index_species]/species.charge[index_species])**2*vth_a**3
    #L13_fac_a=n/jnp.sqrt(jnp.pi)*(species.mass[index_species]/species.charge[index_species])*vth_a**2#*B00(r_grid[r_index])
    #L33_fac_a=n/jnp.sqrt(jnp.pi)*vth_a#*B00(r_grid[r_index])
    L11_fac_a=-1./jnp.sqrt(jnp.pi)*(species.mass[index_species]/species.charge[index_species])**2*vth_a**3
    L13_fac_a=-1./jnp.sqrt(jnp.pi)*(species.mass[index_species]/species.charge[index_species])*vth_a**2#*B00(r_grid[r_index])
    L33_fac_a=-1./jnp.sqrt(jnp.pi)*vth_a#*B00(r_grid[r_index])    
    #Interpolate D11's, D13's and D33's 
    Dij=jax.vmap(get_Dij,in_axes=(None,0,0,None))(field.r_grid[r_index],nu_vnew_a,Er_vnew_a,database)
    #_, Dij = jax.lax.scan(scan_Dij, None, (nu_vnew_a, Er_vnew_a))
    D11_a=-10**Dij.at[:,0].get()###+4.*nu_vnew_a/3.
    D13_a=-Dij.at[:,1].get()
    D33_a=-jnp.true_divide(Dij.at[:,2].get(),nu_vnew_a)
    Lij=Lij.at[0,0].set(L11_fac_a*jnp.sum(grid.L11_weight*grid.xWeights*(D11_a)))
    Lij=Lij.at[0,1].set(L11_fac_a*jnp.sum(grid.L12_weight*grid.xWeights*(D11_a)))
    Lij=Lij.at[1,0].set(Lij.at[0,1].get())
    Lij=Lij.at[1,1].set(L11_fac_a*jnp.sum(grid.L22_weight*grid.xWeights*(D11_a)))
    Lij=Lij.at[0,2].set(L13_fac_a*jnp.sum(grid.L13_weight*grid.xWeights*(D13_a)))
    Lij=Lij.at[1,2].set(L13_fac_a*jnp.sum(grid.L23_weight*grid.xWeights*(D13_a)))
    Lij=Lij.at[2,0].set(-Lij.at[0,2].get())
    Lij=Lij.at[2,1].set(-Lij.at[1,2].get())
    Lij=Lij.at[2,2].set(L33_fac_a*jnp.sum(grid.L33_weight*grid.xWeights*(D33_a)))
    return Lij


@jit
#Get_fluxes with no momentum correction
def get_Neoclassical_Fluxes(species,grid,field,database):
    def get_Neoclassical_Fluxes_internal(species_internal,a,Lij):
        A1=species_internal.A1[a]
        A2=species_internal.A2[a]
        A3=species_internal.A3
        T_a=species_internal.temperature[a]
        n_a=species_internal.density[a]
        Gamma=-n_a*(Lij.at[a,:,0,0].get()*A1+Lij.at[a,:,0,1].get()*A2+Lij.at[a,:,0,2].get()*A3)
        Q=-T_a*n_a*(Lij.at[a,:,1,0].get()*A1+Lij.at[a,:,1,1].get()*A2+Lij.at[a,:,1,2].get()*A3)
        Upar=-n_a*(Lij.at[a,:,2,0].get()*A1+Lij.at[a,:,2,1].get()*A2+Lij.at[a,:,2,2].get()*A3)
        return Gamma, Q, Upar

    Lij=jax.vmap(jax.vmap(get_Lij_matrix, in_axes=(None,None,None,None,None, 0)), in_axes=(None,None,None,None,0, None))(species,grid,field,database,species.species_indeces,grid.full_grid_indeces)
    Lij=Lij.at[:,0,:,:].set(Lij.at[:,1,:,:].get())
    Gamma,Q,Upar=jax.vmap(get_Neoclassical_Fluxes_internal, in_axes=(None,0, None))(species,species.species_indeces,Lij)
    return Lij,Gamma,Q,Upar



#####FOR MOMEMTUM CORRECTION
@jit
def get_Lij_matrix_with_momentum_correction(species,grid,field,database,index_species,r_index):
    #For no momentum correction, Lij is just a 3 x 3 matrix for each species at each radial position
    Lij=jnp.zeros((5,5))
    Eij=jnp.zeros((5,5))
    nu_weighted_average=jnp.zeros(3)
    #Thermal velocities
    vth_a=species.v_thermal[index_species,r_index]
    #velocities for convolution
    v_new_a=grid.v_norm*vth_a
    #Er's for convolution
    #Omega_c=on_OmegaC*species.charge[index_species]*B0[r_index]/species.mass[index_species]
    Er_vnew_a=species.Er[r_index]*1.e+3/v_new_a#-on_OmegaC*B0prime[r_index]/(2.*Omega_c)*v_new_a
    n=species.density[index_species,r_index]
    T=species.temperature[index_species,r_index]
    #same species collisionalities
    ##nu_vnew_T=collisionality(species_a_loc, v_new_a, *global_species_loc)/v_new_a 
    nu_a=collisionality(index_species, species,v_new_a, r_index)#/v_new_a   
    nu_vnew_a=nu_a/v_new_a
    #L11_fac_T=nT*2./jnp.sqrt(jnp.pi)*(tritium_loc.species.mass/tritium_loc.species.charge)**2*vth_T**3
    #L11_fac_a=n/jnp.sqrt(jnp.pi)*(species.mass[index_species]/species.charge[index_species])**2*vth_a**3
    #L13_fac_a=n/jnp.sqrt(jnp.pi)*(species.mass[index_species]/species.charge[index_species])*vth_a**2#*B00(r_grid[r_index])
    #L33_fac_a=n/jnp.sqrt(jnp.pi)*vth_a#*B00(r_grid[r_index])
    L11_fac_a=-1./jnp.sqrt(jnp.pi)*(species.mass[index_species]/species.charge[index_species])**2*vth_a**3
    L13_fac_a=-1./jnp.sqrt(jnp.pi)*(species.mass[index_species]/species.charge[index_species])*vth_a**2#*B00(r_grid[r_index])
    L33_fac_a=-1./jnp.sqrt(jnp.pi)*vth_a#*B00(r_grid[r_index])    
    #Interpolate D11's, D13's and D33's 
    Dij=jax.vmap(get_Dij,in_axes=(None,0,0,None))(field.r_grid[r_index],nu_vnew_a,Er_vnew_a,database)
    D11_a=-10**Dij.at[:,0].get()###+4.*nu_vnew_a/3.
    D13_a=-Dij.at[:,1].get()
    D33_a=-jnp.true_divide(Dij.at[:,2].get(),nu_vnew_a)
    Lij=Lij.at[0,0].set(L11_fac_a*jnp.sum(grid.L11_weight*grid.xWeights*(D11_a)))
    Lij=Lij.at[0,1].set(L11_fac_a*jnp.sum(grid.L12_weight*grid.xWeights*(D11_a)))
    Lij=Lij.at[1,0].set(Lij.at[0,1].get())
    Lij=Lij.at[1,1].set(L11_fac_a*jnp.sum(grid.L22_weight*grid.xWeights*(D11_a)))
    Lij=Lij.at[0,2].set(L13_fac_a*jnp.sum(grid.L13_weight*grid.xWeights*(D13_a)))
    Lij=Lij.at[1,2].set(L13_fac_a*jnp.sum(grid.L23_weight*grid.xWeights*(D13_a)))
    Lij=Lij.at[2,0].set(-Lij.at[0,2].get())
    Lij=Lij.at[2,1].set(-Lij.at[1,2].get())
    Lij=Lij.at[2,2].set(L33_fac_a*jnp.sum(grid.L33_weight*grid.xWeights*(D33_a)))
    #Entries of the Lij matrix related with momentum correction
    Lij=Lij.at[0,3].set(Lij.at[1,2].get())
    Lij=Lij.at[1,3].set(L13_fac_a*jnp.sum(grid.L24_weight*grid.xWeights*(D13_a)))       
    Lij=Lij.at[0,4].set(Lij.at[1,3].get()) 
    Lij=Lij.at[1,4].set(L13_fac_a*jnp.sum(grid.L25_weight*grid.xWeights*(D13_a))) 
    Lij=Lij.at[3,0].set(-Lij.at[0,3].get())
    Lij=Lij.at[4,0].set(-Lij.at[0,4].get())    
    Lij=Lij.at[3,1].set(-Lij.at[1,3].get())    
    Lij=Lij.at[4,1].set(-Lij.at[1,4].get())    
    Lij=Lij.at[3,2].set(L33_fac_a*jnp.sum(grid.L43_weight*grid.xWeights*(D33_a)))
    Lij=Lij.at[2,3].set(Lij.at[3,2].get())    
    Lij=Lij.at[3,3].set(L33_fac_a*jnp.sum(grid.L44_weight*grid.xWeights*(D33_a)))
    Lij=Lij.at[2,4].set(Lij.at[3,3].get())    
    Lij=Lij.at[4,2].set(Lij.at[3,3].get())        
    Lij=Lij.at[3,4].set(L33_fac_a*jnp.sum(grid.L45_weight*grid.xWeights*(D33_a))) 
    Lij=Lij.at[4,3].set(Lij.at[3,4].get())          
    Lij=Lij.at[4,4].set(L33_fac_a*jnp.sum(grid.L55_weight*grid.xWeights*(D33_a))) 
    #collisionality weighted velocity integrals matrix Eij 
    Eij=Eij.at[0,2].set(L13_fac_a*jnp.sum(grid.L13_weight*nu_a*grid.xWeights*(D13_a)))
    Eij=Eij.at[1,2].set(L13_fac_a*jnp.sum(grid.L23_weight*nu_a*grid.xWeights*(D13_a)))
    Eij=Eij.at[2,0].set(-Eij.at[0,2].get())
    Eij=Eij.at[2,1].set(-Eij.at[1,2].get())   
    Eij=Eij.at[2,2].set(L33_fac_a*jnp.sum(grid.L33_weight*nu_a*grid.xWeights*(D33_a)))    
    ##
    Eij=Eij.at[0,3].set(Eij.at[1,2].get())
    Eij=Eij.at[1,3].set(L13_fac_a*jnp.sum(grid.L24_weight*nu_a*grid.xWeights*(D13_a)))  
    Eij=Eij.at[0,4].set(Eij.at[1,3].get())  
    Eij=Eij.at[1,4].set(L13_fac_a*jnp.sum(grid.L25_weight*nu_a*grid.xWeights*(D13_a))) 
    Eij=Eij.at[3,0].set(-Eij.at[0,3].get())
    Eij=Eij.at[4,0].set(-Eij.at[0,4].get())    
    Eij=Eij.at[3,1].set(-Eij.at[1,3].get())    
    Eij=Eij.at[4,1].set(-Eij.at[1,4].get())    
    Eij=Eij.at[3,2].set(L33_fac_a*jnp.sum(grid.L43_weight*nu_a*grid.xWeights*(D33_a)))
    Eij=Eij.at[2,3].set(Eij.at[3,2].get())    
    Eij=Eij.at[3,3].set(L33_fac_a*jnp.sum(grid.L44_weight*nu_a*grid.xWeights*(D33_a)))
    Eij=Eij.at[2,4].set(Eij.at[3,3].get())    
    Eij=Eij.at[4,2].set(Eij.at[3,3].get())        
    Eij=Eij.at[3,4].set(L33_fac_a*jnp.sum(grid.L45_weight*nu_a*grid.xWeights*(D33_a))) 
    Eij=Eij.at[4,3].set(Eij.at[3,4].get())          
    Eij=Eij.at[4,4].set(L33_fac_a*jnp.sum(grid.L55_weight*nu_a*grid.xWeights*(D33_a)))     
    #Velocity average of collisionalities
    nu_weighted_average=nu_weighted_average.at[0].set(jnp.sum(nu_a*grid.L13_weight*grid.xWeights))
    nu_weighted_average=nu_weighted_average.at[1].set(jnp.sum(nu_a*grid.L23_weight*grid.xWeights))
    nu_weighted_average=nu_weighted_average.at[2].set(jnp.sum(nu_a*grid.L24_weight*grid.xWeights))    
    return Lij,Eij,nu_weighted_average


#Calculate Cm and Cn, which represent the collision operator terms for each pair of species~
#This uses Hinton/Braginskii,... maybe in the future would be worth to try and see if this can be upgraded to a better model
@jit
def get_Collision_Operator_terms(species,a,b,r_index):
    CM_ab=jnp.zeros((3,3))
    CN_ab=jnp.zeros((3,3))
    T_e=species.temperature[0,r_index]
    n_e=species.density[0,r_index]
    mass_a=species.mass[a]
    temperature_a=species.temperature[a,r_index]
    density_b=species.density[b,r_index]
    v_ratio_squared_ba=jnp.power(species.v_thermal[b,r_index],2)/jnp.power(species.v_thermal[a,r_index],2)
    mass_ratio_ab=mass_a/species.mass[b]
    charge_ratio_ab=species.charge[a]/species.charge[b]
    term=jnp.sqrt(1.0+v_ratio_squared_ba)
    CM_ab=CM_ab.at[0,0].set(-(1.0 +  mass_ratio_ab)/jnp.power(term,3))
    CM_ab=CM_ab.at[0,1].set(1.5*CM_ab.at[0,0].get()/(1.0+v_ratio_squared_ba))
    CM_ab=CM_ab.at[0,2].set(1.25*CM_ab.at[0,1].get()/(1.0+v_ratio_squared_ba))
    CM_ab=CM_ab.at[1,0].set(CM_ab.at[0,1].get())
    CM_ab=CM_ab.at[1,1].set(-(3.25 + 4.0*v_ratio_squared_ba + 7.5*jnp.power(v_ratio_squared_ba,2))/jnp.power(term,5))
    CM_ab=CM_ab.at[1,2].set(-(4.3125 + 6.0*v_ratio_squared_ba + 15.75*jnp.power(v_ratio_squared_ba,2))/jnp.power(term,7))
    CM_ab=CM_ab.at[2,0].set(CM_ab.at[0,2].get())
    CM_ab=CM_ab.at[2,1].set(CM_ab.at[1,2].get())
    CM_ab=CM_ab.at[2,2].set(-(6.765625 + 17.0*v_ratio_squared_ba+ 57.375*jnp.power(v_ratio_squared_ba,2) + 28.0*jnp.power(v_ratio_squared_ba,3) + 21.875*jnp.power(v_ratio_squared_ba,4))/jnp.power(term,9))
    CN_ab=CN_ab.at[0,0].set(-CM_ab.at[0,0].get())
    CN_ab=CN_ab.at[0,1].set(-v_ratio_squared_ba*CM_ab.at[0,1].get())
    CN_ab=CN_ab.at[0,2].set(-v_ratio_squared_ba*v_ratio_squared_ba*CM_ab.at[0,2].get())
    CN_ab=CN_ab.at[1,0].set(-CM_ab.at[1,0].get())
    CN_ab=CN_ab.at[1,1].set(6.75*mass_ratio_ab/jnp.power(term,5))
    CN_ab=CN_ab.at[1,2].set(14.0625*mass_ratio_ab*v_ratio_squared_ba/jnp.power(term,7))
    CN_ab=CN_ab.at[2,0].set(-CM_ab.at[2,0].get())
    CN_ab=CN_ab.at[2,1].set(CN_ab.at[1,2].get()*v_ratio_squared_ba/mass_ratio_ab)
    CN_ab=CN_ab.at[2,2].set(2625.0/64.0*mass_ratio_ab*v_ratio_squared_ba/jnp.power(term,9))
    lnlambda=32.2+1.15*jnp.log10(jnp.power(T_e,2)/n_e) 
    tau_ab=3.0*jnp.power(epsilon_0/charge_ratio_ab,2)/elementary_charge*jnp.sqrt(mass_a/elementary_charge*jnp.power(2.0*jnp.pi*temperature_a,3))/(elementary_charge*density_b*lnlambda)
    return CM_ab,CN_ab,tau_ab


@jit 
def get_rhs(species,a,r_index,Lij):
    rhs=jnp.zeros(3)  ###For reference this has density in it so it has values of 1.e20 and the first value is zero         
    rhs=rhs.at[0].set(-Lij.at[2,0].get()*species.A1[a,r_index]-Lij.at[2,1].get()*species.A2[a,r_index]-Lij.at[2,2].get()*species.A3[r_index])
    rhs=rhs.at[1].set(-(2.5*Lij.at[2,0].get()-Lij.at[3,0].get())*species.A1[a,r_index]
                    -(2.5*Lij.at[2,1].get()-Lij.at[3,1].get())*species.A2[a,r_index]
                    -(2.5*Lij.at[2,2].get()-Lij.at[3,2].get())*species.A3[r_index])
    rhs=rhs.at[2].set(-(4.375*Lij.at[2,0].get()-3.5*Lij.at[3,0].get()+0.5*Lij.at[4,0].get())*species.A1[a,r_index]
                    -(4.375*Lij.at[2,1].get()-3.5*Lij.at[3,1].get()+0.5*Lij.at[4,1].get())*species.A2[a,r_index]
                    -(4.375*Lij.at[2,2].get()-3.5*Lij.at[3,2].get()+0.5*Lij.at[4,2].get())*species.A3[r_index])
    return rhs




@jit
#Auxiliar ffunction to get sum matrix for one species a
def get_sum(a,j,k,CM,CN,tau):
    sum_kn=jnp.sum(jnp.true_divide(CM.at[a,:,j,k].get(),tau.at[a,:].get()))+CN.at[a,a,j,k].get()/tau.at[a,a].get()
    return sum_kn

@jit
#Auxiliar ffunction to get sum matrix for one species a
def get_sum_total(a,CM,CN,tau):
    sum_kn=jnp.zeros((3,3))
    sum_kn=sum_kn.at[0,0].set(jnp.sum(jnp.true_divide(CM.at[a,:,0,0].get(),tau.at[a,:].get()))+CN.at[a,a,0,0].get()/tau.at[a,a].get())
    sum_kn=sum_kn.at[0,1].set(jnp.sum(jnp.true_divide(CM.at[a,:,0,1].get(),tau.at[a,:].get()))+CN.at[a,a,0,1].get()/tau.at[a,a].get())
    sum_kn=sum_kn.at[0,2].set(jnp.sum(jnp.true_divide(CM.at[a,:,0,2].get(),tau.at[a,:].get()))+CN.at[a,a,0,2].get()/tau.at[a,a].get())
    sum_kn=sum_kn.at[1,0].set(jnp.sum(jnp.true_divide(CM.at[a,:,1,0].get(),tau.at[a,:].get()))+CN.at[a,a,1,0].get()/tau.at[a,a].get())
    sum_kn=sum_kn.at[1,1].set(jnp.sum(jnp.true_divide(CM.at[a,:,1,1].get(),tau.at[a,:].get()))+CN.at[a,a,1,1].get()/tau.at[a,a].get())
    sum_kn=sum_kn.at[1,2].set(jnp.sum(jnp.true_divide(CM.at[a,:,1,2].get(),tau.at[a,:].get()))+CN.at[a,a,1,2].get()/tau.at[a,a].get())
    sum_kn=sum_kn.at[2,0].set(jnp.sum(jnp.true_divide(CM.at[a,:,2,0].get(),tau.at[a,:].get()))+CN.at[a,a,2,0].get()/tau.at[a,a].get())     
    sum_kn=sum_kn.at[2,1].set(jnp.sum(jnp.true_divide(CM.at[a,:,2,1].get(),tau.at[a,:].get()))+CN.at[a,a,2,1].get()/tau.at[a,a].get())     
    sum_kn=sum_kn.at[2,2].set(jnp.sum(jnp.true_divide(CM.at[a,:,2,2].get(),tau.at[a,:].get()))+CN.at[a,a,2,2].get()/tau.at[a,a].get())     
    return sum_kn


@jit
#auxilir matrix to construct matrix for species a
def get_A_matrix(grid,a,b,coeff,nucoeff,CN,sum,tau,factor):
    I=jnp.identity(3)
    A=jnp.zeros((3,3))
    A=(I-factor*jnp.multiply(jnp.matmul(jnp.transpose(coeff),sum)+nucoeff,grid.Sonine_expansion))*I.at[a,b].get()-factor*jnp.multiply(jnp.matmul(jnp.transpose(coeff),CN.at[a,b,:,:].get()/tau.at[a,b].get()),grid.Sonine_expansion)*(1.-I.at[a,b].get())      #Different species part
    #A=A.at[0,0].set(I.at[a,b].get()*(1.-factor*(jnp.sum(coeff.at[:,0].get()*sum.at[:,0].get())+nucoeff.at[0,0].get())*Sonine_expansion.at[0].get())-factor*jnp.sum(coeff.at[:,0].get()*CN.at[a,b,:,0].get())/tau.at[a,b].get()*Sonine_expansion.at[0].get()*(1.-I.at[a,b].get()))
    #A=A.at[0,1].set(I.at[a,b].get()*(-factor*(jnp.sum(coeff.at[:,0].get()*sum.at[:,1].get())+nucoeff.at[1,0].get())*Sonine_expansion.at[1].get())-factor*jnp.sum(coeff.at[:,0].get()*CN.at[a,b,:,1].get())/tau.at[a,b].get()*Sonine_expansion.at[1].get()*(1.-I.at[a,b].get()))
    #A=A.at[0,2].set(I.at[a,b].get()*(-factor*(jnp.sum(coeff.at[:,0].get()*sum.at[:,2].get())+nucoeff.at[2,0].get())*Sonine_expansion.at[2].get())-factor*jnp.sum(coeff.at[:,0].get()*CN.at[a,b,:,2].get())/tau.at[a,b].get()*Sonine_expansion.at[2].get()*(1.-I.at[a,b].get()))
    #A=A.at[1,0].set(I.at[a,b].get()*(-factor*(jnp.sum(coeff.at[:,1].get()*sum.at[:,0].get())+nucoeff.at[0,1].get())*Sonine_expansion.at[0].get())-factor*jnp.sum(coeff.at[:,1].get()*CN.at[a,b,:,0].get())/tau.at[a,b].get()*Sonine_expansion.at[0].get()*(1.-I.at[a,b].get()))
    #A=A.at[1,1].set(I.at[a,b].get()*(1.-factor*(jnp.sum(coeff.at[:,1].get()*sum.at[:,1].get())+nucoeff.at[1,1].get())*Sonine_expansion.at[1].get())-factor*jnp.sum(coeff.at[:,1].get()*CN.at[a,b,:,1].get())/tau.at[a,b].get()*Sonine_expansion.at[1].get()*(1.-I.at[a,b].get()))
    #A=A.at[1,2].set(I.at[a,b].get()*(-factor*(jnp.sum(coeff.at[:,1].get()*sum.at[:,2].get())+nucoeff.at[2,1].get())*Sonine_expansion.at[2].get())-factor*jnp.sum(coeff.at[:,1].get()*CN.at[a,b,:,2].get())/tau.at[a,b].get()*Sonine_expansion.at[2].get()*(1.-I.at[a,b].get()))
    #A=A.at[2,0].set(I.at[a,b].get()*(-factor*(jnp.sum(coeff.at[:,2].get()*sum.at[:,0].get())+nucoeff.at[0,2].get())*Sonine_expansion.at[0].get())-factor*jnp.sum(coeff.at[:,2].get()*CN.at[a,b,:,0].get())/tau.at[a,b].get()*Sonine_expansion.at[0].get()*(1.-I.at[a,b].get()))
    #A=A.at[2,1].set(I.at[a,b].get()*(-factor*(jnp.sum(coeff.at[:,2].get()*sum.at[:,1].get())+nucoeff.at[1,2].get())*Sonine_expansion.at[1].get())-factor*jnp.sum(coeff.at[:,2].get()*CN.at[a,b,:,1].get())/tau.at[a,b].get()*Sonine_expansion.at[1].get()*(1.-I.at[a,b].get()))
    #A=A.at[2,2].set(I.at[a,b].get()*(1.-factor*(jnp.sum(coeff.at[:,2].get()*sum.at[:,2].get())+nucoeff.at[2,2].get())*Sonine_expansion.at[2].get())-factor*jnp.sum(coeff.at[:,2].get()*CN.at[a,b,:,2].get())/tau.at[a,b].get()*Sonine_expansion.at[2].get()*(1.-I.at[a,b].get()))
    return A

@jit
#auxilir matrix to construct matrix for species a
def get_correction_matrix(species,grid,a,b,coeff,nucoeff,CM,CN,sum,tau,factor,correction,r_index):
    I=jnp.identity(3)
    A=-factor*jnp.matmul(jnp.matmul(jnp.transpose(coeff),sum)+nucoeff,jnp.multiply(grid.Sonine_expansion,correction.at[b].get()))*I.at[a,b].get()-factor*jnp.matmul(jnp.matmul(jnp.transpose(coeff),CN.at[a,b,:,:].get()/tau.at[a,b].get()),jnp.multiply(grid.Sonine_expansion,correction.at[b].get()))*(1.-I.at[a,b].get())      #Different species part
    add1=(1.-I.at[a,b].get())*(species.dndr[a,r_index]/species.density[a,r_index]+species.dTdr[a,r_index]/species.temperature[a,r_index]
    -species.charge[a]/species.charge[b]*species.temperature[b,r_index]/species.temperature[a,r_index]
    *(species.dndr[b,r_index]/species.density[b,r_index]+species.dndr[b,r_index]/species.density[b,r_index]))*CM.at[a,b,0,0].get()/tau.at[a,b].get()
    add2=(1.-I.at[a,b].get())*(species.dndr[a,r_index]/species.density[a,r_index]+species.dTdr[a,r_index]/species.temperature[a,r_index]
    -species.charge[a]/species.charge[b]*species.temperature[b,r_index]/species.temperature[a,r_index]
    *(species.dndr[b,r_index]/species.density[b,r_index]+species.dndr[b,r_index]/species.density[b,r_index]))*(2.5*CM.at[a,b,0,0].get()-CM.at[a,b,1,0].get())/tau.at[a,b].get()
    add3=(1.-I.at[a,b].get())*species.charge[a]/species.charge[b]*species.temperature[b,r_index]/species.temperature[a,r_index]*(species.dTdr[b,r_index]/species.temperature[b,r_index])*CN.at[a,b,0,1].get()/tau.at[a,b].get()
    add4=(1.-I.at[a,b].get())*species.charge[a]/species.charge[b]*species.temperature[b,r_index]/species.temperature[a,r_index]*(species.dTdr[b,r_index]/species.temperature[b,r_index])*(2.5*CN.at[a,b,0,1].get()-CN.at[a,b,1,1].get())/tau.at[a,b].get()
    return A,add1,add2,add3,add4


@jit 
def get_Matrix(species,grid,field,a,r_index,Lij,Eij,CM_ab,CN_ab,tau):
    coeff=jnp.zeros((3,3))
    nucoeff=jnp.zeros((3,3))
    #Get coeff matrices for species a
    coeff=coeff.at[0,0].set(Lij.at[2,2].get())
    coeff=coeff.at[0,1].set(2.5*Lij.at[2,2].get()-Lij.at[3,2].get())
    coeff=coeff.at[0,2].set(4.375*Lij.at[2,2].get()-3.5*Lij.at[3,2].get()+0.5*Lij.at[3,3].get())
    coeff=coeff.at[1,0].set(Lij.at[2,2].get()-0.4*Lij.at[3,2].get())
    coeff=coeff.at[1,1].set(2.5*Lij.at[2,2].get()-2.0*Lij.at[3,2].get()+0.4*Lij.at[3,3].get())
    coeff=coeff.at[1,2].set(4.375*Lij.at[2,2].get()-5.25*Lij.at[3,2].get()+1.9*Lij.at[3,3].get()-0.2*Lij.at[3,4].get())
    coeff=coeff.at[2,0].set(Lij.at[2,2].get()-0.8*Lij.at[3,2].get()+4.0*Lij.at[3,3].get()/35.0)
    coeff=coeff.at[2,1].set(2.5*Lij.at[2,2].get()-3.0*Lij.at[3,2].get()+38.0*Lij.at[3,3].get()/35.0-4.0*Lij.at[3,4].get()/35.0)
    coeff=coeff.at[2,2].set(4.375*Lij.at[2,2].get()-7.0*Lij.at[3,2].get()+3.8*Lij.at[3,3].get()-0.8*Lij.at[3,4].get()+2.0*Lij.at[4,4].get()/35.0)
    nucoeff=nucoeff.at[0,0].set(Eij.at[2,2].get())
    nucoeff=nucoeff.at[0,1].set(2.5*Eij.at[2,2].get()-Eij.at[3,2].get())
    nucoeff=nucoeff.at[0,2].set(4.375*Eij.at[2,2].get()-3.5*Eij.at[3,2].get()+0.5*Eij.at[3,3].get())
    nucoeff=nucoeff.at[1,0].set(nucoeff.at[0,1].get())
    nucoeff=nucoeff.at[1,1].set(6.25*Eij.at[2,2].get()-5.0*Eij.at[3,2].get()+Eij.at[3,3].get())
    nucoeff=nucoeff.at[1,2].set(10.9375*Eij.at[2,2].get()-13.125*Eij.at[3,2].get()+4.75*Eij.at[3,3].get()-0.5*Eij.at[3,4].get())
    nucoeff=nucoeff.at[2,0].set(nucoeff.at[0,2].get())
    nucoeff=nucoeff.at[2,1].set(nucoeff.at[1,2].get())
    nucoeff=nucoeff.at[2,2].set(19.140625*Eij.at[2,2].get()-30.625*Eij.at[3,2].get()+16.625*Eij.at[3,3].get()-3.5*Eij.at[3,4].get()+0.25*Eij.at[4,4].get())
    #Get sum matrix to be used for constructing A in A*x=rhs system, its norder+1 x norder+1 matrix, thus 3x3 in this case
    sum=jax.vmap(jax.vmap(get_sum,in_axes=(None,None,0,None,None,None)),in_axes=(None,0,None,None,None,None))(a,grid.sonine_indeces,grid.sonine_indeces,CM_ab,CN_ab,tau)
    #sum=get_sum_total(a,CM_ab,CN_ab,tau)
    #Get a 3x3 for each species
    factor=2./jnp.power(species.v_thermal[a,r_index],2)/field.Bsqav[r_index] #Define bsqav
    M=jax.vmap(get_A_matrix,in_axes=(None,None,0,None,None,None,None,None,None))(grid,a,species.species_indeces,coeff,nucoeff,CN_ab,sum,tau,factor)
    M=jax.lax.reshape(M,(M.shape[0],M.shape[1]*M.shape[2]),(1,0,2))
    ##construct rhs 3x1 vector for species a, sizer is norder+1=3             
    return M 


@jit 
def get_corrected_fluxes(species,grid,field,a,r_index,Lij,Eij,nu_av,CM_ab,CN_ab,tau,correction):
    coeff=jnp.zeros((3,3))
    nucoeff=jnp.zeros((3,3))
    #Get coeff matrices for the correction
    coeff=coeff.at[0,0].set(Lij.at[2,0].get())
    coeff=coeff.at[0,1].set(Lij.at[3,0].get())
    coeff=coeff.at[1,0].set(Lij.at[2,0].get()-0.4*Lij.at[2,1].get())
    coeff=coeff.at[1,1].set(Lij.at[3,0].get()-0.4*Lij.at[3,1].get())
    coeff=coeff.at[2,0].set(Lij.at[2,0].get()-0.8*Lij.at[2,1].get()+4.0/35.0*Lij.at[3,1].get())
    coeff=coeff.at[2,1].set(Lij.at[3,0].get()-0.8*Lij.at[3,1].get()+4.0/35.0*Lij.at[4,1].get())
    nucoeff=nucoeff.at[0,0].set(Eij.at[2,0].get())
    nucoeff=nucoeff.at[0,1].set(Eij.at[3,0].get())
    nucoeff=nucoeff.at[1,0].set(2.5*Eij.at[2,0].get()-Eij.at[2,1].get())
    nucoeff=nucoeff.at[1,1].set(2.5*Eij.at[3,0].get()-Eij.at[3,1].get())
    nucoeff=nucoeff.at[2,0].set(4.375*Eij.at[2,0].get()-3.5*Eij.at[2,1].get()+0.5*Eij.at[3,1].get())
    nucoeff=nucoeff.at[2,1].set(4.375*Eij.at[3,0].get()-3.5*Eij.at[3,1].get()+0.5*Eij.at[4,1].get())
    #Get sum matrix yet again
    sum=jax.vmap(jax.vmap(get_sum,in_axes=(None,None,0,None,None,None)),in_axes=(None,0,None,None,None,None))(a,grid.sonine_indeces,grid.sonine_indeces,CM_ab,CN_ab,tau)
    #Get a 3x3 for each species
    factor=2./jnp.power(species.v_thermal[a,r_index],2)/field.Bsqav[r_index] #Define bsqav
    #get vector for correction
    M,add1,add2,add3,add4=jax.vmap(get_correction_matrix,in_axes=(None,None,None,0,None,None,None,None,None,None,None,None,None))(species,grid,a,species.species_indeces,coeff,nucoeff,CM_ab,CN_ab,sum,tau,factor,correction,r_index)
    C=jnp.sum(M,axis=0) #should be a (n_species,3) and we sum on all species
    ADD1=jnp.sum(add1)
    ADD2=jnp.sum(add2)
    ADD3=jnp.sum(add3)
    ADD4=jnp.sum(add4)            
    ##calculate corrected fluxes for species a
    Gamma=species.density[a,r_index]*(-(Lij.at[0,0].get()*species.A1[a,r_index]+Lij.at[0,1].get()*species.A2[a,r_index]+Lij.at[0,2].get()*species.A3[r_index])
    +C.at[0].get())#+species.mass[a]*species.temperature[a,r_index]*G_PS[r_index]/elementary_charge/jnp.power(species.charge[a]*B0,2)*(ADD1-species.A2[a,r_index]*sum.at[0,1].get()
    #-ADD3+species.A1[a,r_index]*nu_av.at[0].get()/1.5+species.A2[a,r_index]*nu_av.at[1].get()/1.5))
    Q=species.temperature[a,r_index]*species.density[a,r_index]*(-(Lij.at[1,0].get()*species.A1[a,r_index]+Lij.at[1,1].get()*species.A2[a,r_index]+Lij.at[1,2].get()*species.A3[r_index])
    +C.at[1].get())#+species.mass[a]*species.temperature[a,r_index]*G_PS[r_index]/elementary_charge/jnp.power(species.charge[a]*B0,2)*(ADD2-species.A2[a,r_index]*(2.5*sum.at[0,1].get()-sum.at[1,1].get())
    #-ADD4+species.A1[a,r_index]*nu_av.at[1].get()/1.5+species.A2[a,r_index]*nu_av.at[2].get()/1.5))
    Upar=correction.at[a,0].get()*species.density[a,r_index]
    qpar=correction.at[a,1].get()
    Upar2=correction.at[a,2].get()
    return Gamma,Q,Upar,qpar,Upar2


@jit
#Get momentum correction at one radial position
def get_momentum_Correction(species,grid,field,r_index,Lij,Eij,nu_av):
    #Get collisional operator expansion matrix for a radial position
    CM_ab,CN_ab,tau=jax.vmap(jax.vmap(get_Collision_Operator_terms,in_axes=(None,None,0, None)), in_axes=(None,0, None,None))(species,species.species_indeces,species.species_indeces,r_index)    
    #construct the linear system M*solution = rhs to be solved
    #Construct rhs vector
    rhs=jax.vmap(get_rhs,in_axes=(None,0,None,0))(species,species.species_indeces,r_index,Lij) #Confirm this
    rhs=jnp.reshape(rhs,rhs.shape[0]*rhs.shape[1]) #Unravel into order x [nspecies*order] array
    #Construct matrix M=
    M=jax.vmap(get_Matrix,in_axes=(None,None,None,0,None,0,0,None,None,None))(species,grid,field,species.species_indeces,r_index,Lij,Eij,CM_ab,CN_ab,tau) #Confirm this
    S=lineax.MatrixLinearOperator(jnp.reshape(M,(M.shape[0]*M.shape[1],M.shape[2])))
    #Solve linear system using lineax to get the correction 
    solution=lineax.linear_solve(S,rhs)
    corr=jnp.reshape(solution.value,(CM_ab.shape[0],CM_ab.shape[-1]))
    #Now we need to get corrected fluxes
    #Then we apply correction to fluxes for each species in a function similar to the one for getting matrix M 
    Gamma,Q,Upar,qpar,Upar2 =jax.vmap(get_corrected_fluxes,in_axes=(None,None,None,0,None,0,0,0,None,None,None,None))(species,grid,field,species.species_indeces,r_index,Lij,Eij,nu_av,CM_ab,CN_ab,tau,corr) #Confirm this
    return Gamma,Q,Upar,qpar,Upar2





@jit
#Get_fluxes with no momentum correction
def get_Neoclassical_Fluxes_With_Momentum_Correction(species,grid,field,database):
    #def get_Neoclassical_Fluxes_with_Momentum_Correction_internal(species_internal,a,Lij_a):
    ###Construct and solve linear system to get momentum correction
    Lij,Eij,nu_weighted_average=jax.vmap(jax.vmap(get_Lij_matrix_with_momentum_correction, in_axes=(None,None,None,None,None, 0)), in_axes=(None,None,None,None,0, None))(species,grid,field,database,species.species_indeces,grid.full_grid_indeces)
    Lij=Lij.at[:,0,:,:].set(Lij.at[:,1,:,:].get())
    Eij=Eij.at[:,0,:,:].set(Eij.at[:,1,:,:].get())
    correction=jax.vmap(get_momentum_Correction,in_axes=(None,None,None,0,1,1,1))(species,grid,field,grid.full_grid_indeces,Lij,Eij,nu_weighted_average)
    #R loop for calculating matrix and solving system
    #Gamma,Q,Upar=jax.vmap(get_Neoclassical_Fluxes_with_Momentum_Correction_internal, in_axes=(None,0, 0))(species,s_indeces,Lij)
    return correction#,Lij,Eij,nu_weighted_average






