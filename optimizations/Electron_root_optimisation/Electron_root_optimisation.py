#Importing modules
import os
current_path = os.path.dirname(os.path.realpath(__name__))
import sys
sys.path.insert(1, os.path.join(current_path,'../../'))
import jax.numpy as jnp
import jax
jax.config.update('jax_platform_name', 'gpu')
# before execute any computation / allocation
print('Using device:', jax.devices())
from jax import config
# to use higher precision
config.update("jax_enable_x64", True)
import h5py as h5
import NEOPAX
import monkes
import matplotlib.pyplot as plt
from simsopt.mhd import Boozer





def Electron_root_objective(vmec_file,nt,nz,nl,Er,rho):
    #if(mpi.)
    #Run boozer file from vmec file
    b=Boozer()
    b.read_wout(vmec_file)
    b.run()
    output_file=vmec_file.split('/')[-1]
    boozer_file="boozer_"+str(output_file)
    b.write_boozmn(boozer_file)
    #Create grid
    n_species=2
    Nx=4
    n_radial=51

    grid=NEOPAX.Grid.create_standard(n_radial,Nx,n_species)

    #Get magnetic configuration related quantities from class Field
    field_neopax=NEOPAX.Field.read_vmec_booz(n_radial,vmec_file,boozer_file)
    field_monkes=monkes.Field.from_vmec(vmec_file, rho**2, nt, nz)
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


    #Define species profiles 
    Te_initial_function= lambda r: T_scale*((te0-teb)*(1-(r/field_neopax.a_b)**2)+teb)
    ne_initial_function =lambda r: n_scale*((ne0-neb)*(1-(r/field_neopax.a_b)**10.)+neb)
    Ti_initial_function= lambda r: T_scale*((ti0-tib)*(1-(r/field_neopax.a_b)**2)+tib)
    ni_initial_function = lambda r: n_scale*deuterium_ratio*((ni0-nib)*(1-(r/field_neopax.a_b)**10.)+nib)
    Er_initial_function = lambda x: field_neopax.r_grid*0.

    #Er0 = SpatialDiscretisation.discretise_fn(r0, r_final, n_radial, Er_initial_function)

    #Initialize species
    Te_initial=Te_initial_function(grid.rho_grid)
    ne_initial=ne_initial_function(grid.rho_grid)
    Ti_initial=Ti_initial_function(grid.rho_grid)
    ni_initial=ni_initial_function(grid.rho_grid)
    Er_initial=grid.rho_grid*0.


    temperature_initial=jnp.zeros((n_species,n_radial))
    density_initial=jnp.zeros((n_species,n_radial))
    temperature_initial=temperature_initial.at[0,:].set(Te_initial)
    temperature_initial=temperature_initial.at[1,:].set(Ti_initial)
    density_initial=density_initial.at[0,:].set(ne_initial)
    density_initial=density_initial.at[1,:].set(ni_initial)
    mass=jnp.array([1 / 1836.15267343,2])
    charge=jnp.array([-1,1])

    #Initialçize species
    species=NEOPAX.Species(n_species,n_radial,grid.species_indeces,mass,charge,temperature_initial,density_initial,Er_initial,
                           field_neopax.r_grid,
                           field_neopax.r_grid_half,
                           field_neopax.dr,
                           field_neopax.Vprime_half,
                           field_neopax.overVprime,0.,0.)

    #Module for neoclassical objective functions definition, using NEOPAX, MONKEX, etc...
    def Electron_root_from_Monkes(species,grid,field_neo,field_monk,nl,Er,r_value):
        arr=rho-field_neo.rho_grid
        r_index = jnp.argmax(jnp.where(arr <= 0, arr, -jnp.inf))+1
        #Pitch angle resolution

        #get collisionalities
        v_new_e=grid.v_norm*species.v_thermal[0,r_index]
        nu_vnew_e=NEOPAX.collisionality(0, species,v_new_e, r_index)/v_new_e
        v_new_i=grid.v_norm*species.v_thermal[1,r_index]
        nu_vnew_i=NEOPAX.collisionality(1, species,v_new_i, r_index)/v_new_i

        L11_fac_e=-1./jnp.sqrt(jnp.pi)*(species.mass[0]/species.charge[0])**2*species.v_thermal[0,r_index]**3
        L11_fac_i=-1./jnp.sqrt(jnp.pi)*(species.mass[1]/species.charge[1])**2*species.v_thermal[1,r_index]**3
        #L13_fac_D=-1./jnp.sqrt(jnp.pi)*(species.mass[index_species]/species.charge[index_species])*vth_a**2#*B00(r_grid[r_index])
        #L33_fac_a=-1./jnp.sqrt(jnp.pi)*vth_a#*B00(r_grid[r_index])    

        #Calculate Dij over the x grid for ions and electrons     
        Dij_ions, _, _ = jax.vmap(monkes.monoenergetic_dke_solve_internal,in_axes=(None,None,0,None))(field_monk, Er, nu_vnew_i, nl)
        Dij_electrons, _, _ = jax.vmap(monkes.monoenergetic_dke_solve_internal,in_axes=(None,None,0,None))(field_monk, Er, nu_vnew_e, nl)

        #Now to calculate L11 we just need to use the xWeights 
        L11i=L11_fac_i*jnp.sum(grid.L11_weight*grid.xWeights*Dij_ions.at[:,0,0].get())
        L11e=L11_fac_e*jnp.sum(grid.L11_weight*grid.xWeights*Dij_electrons.at[:,0,0].get())

        return L11i/L11e
    
    cost=jax.jit(Electron_root_from_Monkes,device=jax.devices('cpu')[mpi.world_rank])(species,grid,field_monkes,nl,Er,rho).block_until_ready()
    #cost=jax.jit(Electron_root_from_Monkes,device=jax.devices('cpu')[mpi.world_rank])(species,grid,field_monkes,nl,Er,rho)


vmec_file=os.path.join(current_path,'../inputs/wout_QI_nfp2_newNT_opt_hires.nc')
nt=31
nz=31
nl=40
Er=17.8e+3
rho=0.29
electron_root=Electron_root_objective(vmec_file,nt,nz,nl,Er,rho)
