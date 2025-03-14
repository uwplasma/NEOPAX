#Importing modules
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=1'
current_path = os.path.dirname(os.path.realpath(__name__))
import sys
sys.path.insert(1, os.path.join(current_path,'../../'))
sys.path.insert(1, os.path.join(current_path,'../../../'))
import jax.numpy as jnp
import jax
jax.config.update('jax_platform_name', 'cpu')
# before execute any computation / allocation
print('Using device:', jax.devices())
from jax import config
# to use higher precision
config.update("jax_enable_x64", True)
import h5py as h5
import NEOPAX
import monkes
import matplotlib.pyplot as plt
from simsopt.mhd import Vmec
import booz_xform as bx
from simsopt.util import MpiPartition, proc0_print
from simsopt._core.optimizable import make_optimizable
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_mpi_solve
from qi_functions import QuasiIsodynamicResidual, MirrorRatioPen, MaxElongationPen    
import time
from functools import partial
import numpy as np
import mpi4jax


mpi=MpiPartition(1)

proc0_print("Running Electron_root_optimization.py")
proc0_print("=============================================")

input_file= os.path.join(current_path,'../inputs/input.QI_nfp2')

comm_jax = mpi.comm_world.Clone()


#Set VMEC and boundary variations for optimisation
vmec = Vmec(input_file, mpi=mpi)

# Define parameter space of optimisation (VMEC):
surf = vmec.boundary
surf.fix_all()
max_mode = 2
surf.fixed_range(mmin=0, mmax=max_mode,
                 nmin=-max_mode, nmax=max_mode, fixed=False)
surf.fix("rc(0,0)")  # Major radius

proc0_print('Parameter space:', surf.dof_names)

vmec.keep_all_files = True
b=bx.Booz_xform()



def Electron_root_objective(v,rho,nt,nz,nl,mpi):
    ida=comm_jax.Get_rank()
    token = mpi4jax.send(ida,dest=0 ,comm=comm_jax)
    #Run boozer file from vmec file
    output_file=v.output_file.split('/')[-1]
    boozer_file="boozer_"+str(output_file)
    #if not os.path.exists(boozer_file):
    b.read_wout(v.output_file)
    b.run()
    b.write_boozmn(boozer_file)
    #else: 
        #Create grid
    n_species=2
    Nx=4
    n_radial=51

    grid=NEOPAX.Grid.create_standard(n_radial,Nx,n_species)

    #Get magnetic configuration related quantities from class Field
    field_neopax=NEOPAX.Field.read_vmec_booz(n_radial,v.output_file,boozer_file)
    field_monkes=monkes.Field.from_vmec_s(v.output_file, rho**2, nt, nz)
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
    deuterium_ratio=1.0    


    #Define species profiles 
    Te_initial_function= lambda r: T_scale*((te0-teb)*(1-(r/field_neopax.a_b)**2)+teb)
    ne_initial_function =lambda r: n_scale*((ne0-neb)*(1-(r/field_neopax.a_b)**10.)+neb)
    Ti_initial_function= lambda r: T_scale*((ti0-tib)*(1-(r/field_neopax.a_b)**2)+tib)
    ni_initial_function = lambda r: n_scale*deuterium_ratio*((ni0-nib)*(1-(r/field_neopax.a_b)**10.)+nib)
    Er_initial_function = lambda x: field_neopax.r_grid*0.

    #Initialize species
    Te_initial=Te_initial_function(field_neopax.r_grid)
    ne_initial=ne_initial_function(field_neopax.r_grid)
    Ti_initial=Ti_initial_function(field_neopax.r_grid)
    ni_initial=ni_initial_function(field_neopax.r_grid)
    Er_initial=field_neopax.r_grid*0.


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
    def Electron_root_from_Monkes(species,grid,field_neo,field_monk,nl,r_value):
        arr=r_value-field_neo.rho_grid
        r_index = jnp.argmax(jnp.where(arr <= 0, arr, -jnp.inf))+1
        #get collisionalities
        v_new_e=grid.v_norm*species.v_thermal[0,r_index]
        nu_vnew_e=NEOPAX.collisionality(0, species,v_new_e, r_index)/v_new_e
        v_new_i=grid.v_norm*species.v_thermal[1,r_index]
        nu_vnew_i=NEOPAX.collisionality(1, species,v_new_i, r_index)/v_new_i
        #Er=T_species so that Er/qT=1 at each radial position required
        Er_vnew_e=species.temperature[0,r_index]/v_new_e*field_neo.a_b/(r_value*2.)
        Er_vnew_i=species.temperature[1,r_index]/v_new_i*field_neo.a_b/(r_value*2.)
        #jax.debug.print("r {r} ", r=r_index)
        #jax.debug.print("nu_e {Er_i} ", Er_i=nu_vnew_e)
        #jax.debug.print("nu_i {Er_i} ", Er_i=nu_vnew_i)
        #jax.debug.print("Ere{Er_e} ", Er_e=Er_vnew_e)
        #jax.debug.print("Eri {Er_i} ", Er_i=Er_vnew_i)
        L11_fac_e=-species.density[0,r_index]/jnp.sqrt(jnp.pi)*(species.mass[0]/species.charge[0])**2*species.v_thermal[0,r_index]**3
        L11_fac_i=-species.density[1,r_index]/jnp.sqrt(jnp.pi)*(species.mass[1]/species.charge[1])**2*species.v_thermal[1,r_index]**3
        #L13_fac_D=-1./jnp.sqrt(jnp.pi)*(species.mass[index_species]/species.charge[index_species])*vth_a**2#*B00(r_grid[r_index])
        #L33_fac_a=-1./jnp.sqrt(jnp.pi)*vth_a#*B00(r_grid[r_index])    

            #Calculate Dij over the x grid for ions and electrons 
        D11e=jnp.zeros(len(v_new_e))
        D11i=jnp.zeros(len(v_new_i))
        for j in range(len(v_new_e)):
            Dij_ions, _, _ = monkes.monoenergetic_dke_solve_internal(field_monk, nl, Er_vnew_i[j], nu_vnew_i[j])
            Dij_electrons, _, _ = monkes.monoenergetic_dke_solve_internal(field_monk,  nl, Er_vnew_e[j], nu_vnew_e[j])
            D11i=D11i.at[j].set(Dij_ions[0,0])
            D11e=D11e.at[j].set(Dij_electrons[0,0])

            #Now to calculate L11 we just need to use the xWeights 
        L11i=jax.block_until_ready(L11_fac_i*jnp.sum(grid.L11_weight*grid.xWeights*D11i))
        L11e=jax.block_until_ready(L11_fac_e*jnp.sum(grid.L11_weight*grid.xWeights*D11e))
        jax.debug.print("Eri {Er_i} ", Er_i=L11i)
        jax.debug.print("Eri {Er_i} ", Er_i=L11e)
        jax.debug.print("Eri {Er_i} ", Er_i=L11i/L11e)
        return L11i/L11e
    
    cost=jax.jit(Electron_root_from_Monkes,device=jax.devices('cpu')[mpi.rank_world],static_argnames=['nl'])(species,grid,field_neopax,field_monkes,nl,rho).block_until_ready()
    new_ida, token = mpi4jax.recv(ida, source=comm_jax.Get_rank(),comm=comm_jax, token=token)
        #mpi.comm_world.Barrier()
    return cost



#Wraper of cost function
def WrapCostFunction(v: Vmec,rho: float,nt,nz,nl,mpi: MpiPartition):
    try: 
        v.run()
    except Exception as e:
        print(e)
        return -1#return_number_if_vmec_or_sfincs_not_converged

    try:
        #if(mpi.proc0_groups):
        result = Electron_root_objective(v,rho,nt,nz,nl,mpi)
    except Exception as e:
        print(e)
        result = -1  #return_number_if_vmec_or_sfincs_not_converged
    return result





#Parameters for Qi functions
LENGTHBOUND_QI = 4.5 # Threshold for the length of each coil
CC_THRESHOLD_QI = 0.12 # Threshold for the coil-to-coil distance penalty in the objective function
CURVATURE_THRESHOLD_QI = 10 # Threshold for the curvature penalty in the objective function
MSC_THRESHOLD_QI = 10 # Threshold for the mean squared curvature penalty in the objective function
ncoils_QI = 8 # Number of coils per half field period
quasisymmetry_helicity_n_QI = -1 # Toroidal quasisymmetry integer N in |B|
include_iota_target_QI = False # Specify if iota should be added to the objective function
iota_QI = -0.61 # Target rotational transform iota, using this to avoid rational surfaces
weight_iota_QI=1.  #weight for iota penalty cost function
aspect_ratio_target_QI = 10  # Target aspect ratio
elongation_weight = 1e2       
mirror_weight = 1e2
qsqi_weight=1.
snorms = [1/24, 5/24, 9/24, 13/24] # Flux surfaces at which the penalty will be calculated
nphi_QI=141 # Number of points along measured along each well
nalpha_QI=27 # Number of wells measured
nBj_QI=51 # Number of bounce points measured
mpol_QI=18 # Poloidal modes in Boozer transformation
ntor_QI=18 # Toroidal modes in Boozer transformation
nphi_out_QI=2000 # size of return array if arr_out_QI = True
arr_out_QI=True # If True, returns (nphi_out*nalpha) values, each of which is the difference
maximum_elongation = 6 # Defines the maximum elongation allowed in the QI elongation objective function
maximum_mirror = 0.19 # Defines the maximum mirror ratio of |B| allowed in the QI elongation objective function

#Radial positions for electron root cost function calculation
rho1=0.2
rho2=0.29
rho3=0.35
nt=25
nz=25
nl=64




#start=time.time()
#electron_root=WrapCostFunction(vmec,rho2,nt,nz,nl,mpi)
#final=time.time()
#print(final-start,electron_root)



#QI cost function (A. Goodman)
optQI = partial(QuasiIsodynamicResidual,snorms=snorms, nphi=nphi_QI, nalpha=nalpha_QI, nBj=nBj_QI, mpol=mpol_QI, ntor=ntor_QI, nphi_out=nphi_out_QI, arr_out=arr_out_QI)
qi = make_optimizable(optQI, vmec)
#Elongation penalty in case it is necessary
partial_MaxElongationPen = partial(MaxElongationPen,t=maximum_elongation)
optElongation = make_optimizable(partial_MaxElongationPen, vmec)
#Mirror ration penalty
partial_MirrorRatioPen = partial(MirrorRatioPen,t=maximum_mirror)
optMirror = make_optimizable(partial_MirrorRatioPen, vmec)


optNEO1=partial(WrapCostFunction,rho=rho1,nt=nt,nz=nz,nl=nl,mpi=mpi)
optNEO2=partial(WrapCostFunction,rho=rho2,nt=nt,nz=nz,nl=nl,mpi=mpi)
optNEO3=partial(WrapCostFunction,rho=rho3,nt=nt,nz=nz,nl=nl,mpi=mpi)
#Electron root cost functions at different radial positions (3 in this case)
neo_opt1 = make_optimizable(optNEO1,vmec)
neo_opt2 = make_optimizable(optNEO2,vmec)
neo_opt3 = make_optimizable(optNEO3,vmec)
proc0_print("Electron Root r3 before optimization:", neo_opt3.J())

#Optimise total cost function (apect ration, iota, Electron root, QI, and mirror ration in this case)
prob = LeastSquaresProblem.from_tuples([(vmec.aspect, aspect_ratio_target_QI, 1),(vmec.mean_iota, iota_QI, weight_iota_QI),
                                        (neo_opt1.J,0,1),(neo_opt2.J,0,1),(neo_opt3.J,0,1),(qi.J, 0, qsqi_weight),(optMirror.J, 0, mirror_weight)])
#prob = LeastSquaresProblem.from_tuples([(vmec.aspect, aspect_ratio_target_QI, 1),(vmec.mean_iota, iota_QI, weight_iota_QI),
#                                        (qi.J, 0, qsqi_weight),(optMirror.J, 0, mirror_weight)])

#proc0_print("r1:",neo_opt1.J())
#proc0_print("r2:",neo_opt2.J())
#proc0_print("r3:",neo_opt3.J())
# Make sure all procs participate in computing the objective:
prob.objective()

#Print the relevant individual cost functions (initial)
proc0_print("QI objective before optimization:", qi.J())
#proc0_print("Elogation objective before optimization:", optElongation.J())
proc0_print("Electron Root r3 before optimization:", neo_opt3.J())
proc0_print("Electron Root r3 before optimization:", neo_opt3.J())
proc0_print("Mirror objective before optimization:", optMirror.J())
proc0_print("Electron Root r1 before optimization:", neo_opt1.J())
proc0_print("Electron Root r2 before optimization:", neo_opt2.J())
proc0_print("Total objective before optimization:", prob.objective())


# Evaluate cost function with least squares
least_squares_mpi_solve(prob, mpi, grad=True, rel_step=1e-5, abs_step=1e-8)

# Make sure all procs participate in computing the objective:
prob.objective()

#Print the relevant individual cost functions (final)
proc0_print("Final aspect ratio:", vmec.aspect())
proc0_print("QI objective after optimization:", qi.J())
#proc0_print("Elogation objective after optimization:", optElongation.J())
proc0_print("Mirror objective after optimization:", optMirror.J())
proc0_print("Electron Root r1 after optimization:", neo_opt1.J())
proc0_print("Electron Root r2 after optimization:", neo_opt2.J())
proc0_print("Electron Root r3 after optimization:", neo_opt3.J())
proc0_print("Total objective after optimization:", prob.objective())

proc0_print("End of Electron_root_optimization")
proc0_print("============================================")