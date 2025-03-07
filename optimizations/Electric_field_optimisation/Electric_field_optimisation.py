#Importing modules
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=1'
current_path = os.path.dirname(os.path.realpath(__name__))
import sys
sys.path.insert(1, os.path.join(current_path,'../../'))
sys.path.insert(1, os.path.join(current_path,'../../../'))
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
from simsopt.mhd import Vmec
import booz_xform as bx
from simsopt.util import MpiPartition, proc0_print
from simsopt._core.optimizable import make_optimizable
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_mpi_solve
from qi_functions import QuasiIsodynamicResidual, MirrorRatioPen, MaxElongationPen    
import time
import interpax
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



def create_monkes(v,field_neopax,mpi):
    #ida=comm_jax.Get_rank()
    #token = mpi4jax.send(ida,dest=0 ,comm=comm_jax)

    #Resolution parameters
    nt = 25
    nz = 25
    #Pitch angle resolution
    nl=60  

    nu_v=jnp.array([1.e-5,3.e-5,1.e-4,3.e-4,1.e-3,3.e-3,1.e-2,3.e-2,1.e-1,3.e-1])
    rho=jnp.array([0.12247,0.25,0.375,0.5,0.625,0.75,0.875])
    Er_tilde=np.array([0.0,1.e-6,3.e-6,1.e-5,3.e-5,1.e-4,3.e-4,1.e-3,3.e-3,1.e-2])


    B00_rho=interpax.Interpolator1D(field_neopax.rho_grid,field_neopax.B0)(rho)
    drds=field_neopax.a_b/(rho*2.)
    dr_tildedr=2.*field_neopax.Psia_value/(field_neopax.a_b**2*B00_rho)*field_neopax.a_b/(rho*2.)
    dr_tildeds=dr_tildedr*drds

    #Ceate arrays for different electric field representations
    #Es=np.zeros((len(rho),len(Er_tilde)))
    Er=np.zeros((len(rho),len(Er_tilde)))

    #Create arrays for Dij's monoenergetic scan data 
    D11=jnp.zeros((len(rho),len(nu_v),len(Er_tilde)))
    D13=jnp.zeros((len(rho),len(nu_v),len(Er_tilde)))
    D31=jnp.zeros((len(rho),len(nu_v),len(Er_tilde)))
    D33=jnp.zeros((len(rho),len(nu_v),len(Er_tilde)))


    #Loop for every collisionality and electric field value to obtain the monoenergetic scan 
    #Use internal solve as we do not care for species information in the monoenergetic database
    #Using normal for loop here as it serves only for benchmark-> put into vmap to speed up in real calculations
    for si in range(len(rho)):
        field_monkes = monkes.Field.from_vmec_s(v.output_file, rho[si]**2, nt, nz)     
        for j in range(len(nu_v)):       
            for i in range(len(Er_tilde)):
                #Here we use input 
                Es=Er_tilde[i]*dr_tildeds[si]*B00_rho[si] #Notice we need to multiply by B00 and dr_tildeds factors due to IPP DKES weird coordinates
                Er[si,i]=Er_tilde[i]*dr_tildedr[si]*B00_rho[si]
                #Calculate one Dij matrix 
                #Dij, _, _ = jax.jit(monkes.monoenergetic_dke_solve_internal,device=jax.devices('gpu')[0])(field_monkes, nl=nl, Erhat=Es,nuhat=nu_v[j])
                Dij, _, _ = jax.jit(monkes.monoenergetic_dke_solve_internal,device=jax.devices('cpu')[mpi.rank_world])(field_monkes, nl=nl, Erhat=Es,nuhat=nu_v[j])
                D11[si,j,i]=Dij[0,0]
                D13[si,j,i]=Dij[0,2]
                D31[si,j,i]=Dij[2,0]
                D33[si,j,i]=Dij[2,2]    
                print(si,j,i)
                print(Dij)


    return NEOPAX.NEOPAX.Monoenergetic.read_data(field_neopax.a_b,rho,nu_v,Er,drds,D11,D13,D33)



def Get_Er_profile(v,mpi):
    #Run booz_xform
    output_file=v.output_file.split('/')[-1]
    boozer_file="boozer_"+str(output_file)
    #if not os.path.exists(boozer_file):
    b.read_wout(v.output_file)
    b.run()
    b.write_boozmn(boozer_file)
    #Create grid
    n_species=3
    Nx=4
    n_radial=51
    grid=NEOPAX.Grid.create_standard(n_radial,Nx,n_species)

    #Get magnetic configuration related quantities from class Field
    field_neopax=NEOPAX.Field.read_vmec_booz(n_radial,v.output_file,boozer_file)

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
    T_edge=jnp.array(0.7*1.e+3)
    n_edge=jnp.array(0.6e+20)

    #Define species profiles 
    Te_initial_function= lambda r: T_scale*((te0-teb)*(1-(r/field_neopax.a_b)**2)+teb)
    ne_initial_function =lambda r: n_scale*((ne0-neb)*(1-(r/field_neopax.a_b)**10.)+neb)
    TD_initial_function= lambda r: T_scale*((ti0-tib)*(1-(r/field_neopax.a_b)**2)+tib)
    nD_initial_function = lambda r: n_scale*deuterium_ratio*((ni0-nib)*(1-(r/field_neopax.a_b)**10.)+nib)
    TT_initial_function= lambda r: T_scale*((ti0-tib)*(1-(r/field_neopax.a_b)**2)+tib)
    nT_initial_function = lambda r: n_scale*tritium_ratio*((ni0-nib)*(1-(r/field_neopax.a_b)**10.)+nib)
    Er_initial_function = lambda x: 100.*x*(0.8-x)

    #Er0 = SpatialDiscretisation.discretise_fn(r0, r_final, n_radial, Er_initial_function)

    #Initialize species
    Te_initial=Te_initial_function(field_neopax.r_grid)
    ne_initial=ne_initial_function(field_neopax.r_grid)
    TD_initial=TD_initial_function(field_neopax.r_grid)
    nD_initial=nD_initial_function(field_neopax.r_grid)
    TT_initial=TT_initial_function(field_neopax.r_grid)
    nT_initial=nT_initial_function(field_neopax.r_grid)
    Er_initial=Er_initial_function(field_neopax.rho_grid)


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
    Global_species=NEOPAX.Species(n_species,n_radial,grid.species_indeces,mass,charge,temperature_initial,density_initial,Er_initial,
                                  field_neopax.r_grid,field_neopax.r_grid_half,field_neopax.dr,field_neopax.Vprime_half, field_neopax.overVprime,n_edge,T_edge)


    #Read monoenergetic database, from MONKES 
    database=create_monkes(v,field_neopax)

    #Initial profiles
    ne0=Global_species.density[0]*1.e-20
    nD0=Global_species.density[1]*1.e-20
    nT0=Global_species.density[2]*1.e-20
    Pe0=Global_species.temperature[0]*1.e-3*ne0
    PD0=Global_species.temperature[1]*1.e-3*nD0
    PT0=Global_species.temperature[2]*1.e-3*nT0
    Er0=Global_species.Er

    #Define initial coupled solutions to diffrax (can solve tupples of spatial descritizations!!! already tested)
    y0=(Er0,Pe0,PD0,PT0,ne0,nD0,nT0)#,ne_initial,nT_initial,nD_initial,nHe_initial in r=rho*a_b!!!!
    args=Global_species,grid,field_neopax,database

    #sol=jax.jit(NEOPAX.solve_transport_equations,,device=jax.devices('gpu')[0])(y0,args)
    sol=jax.jit(NEOPAX.solve_transport_equations,device=jax.devices('cpu')[mpi.rank_world])(y0,args)
    return jnp.max(sol.ys[0][-1])




#Wraper of cost function
def WrapCostFunction(v: Vmec,rho: float,nt,nz,nl,mpi: MpiPartition):
    try: 
        v.run()
    except Exception as e:
        print(e)
        return -1#return_number_if_vmec_or_sfincs_not_converged

    try:
        #if(mpi.proc0_groups):
        #result=jax.jit(Electron_root_from_Monkes,device=jax.devices('cpu')[mpi.rank_world],static_argnames=0)(v).block_until_ready()
        #result=jax.jit(Get_Er_profile,device=jax.devices('gpu'),static_argnames=0)(v).block_until_ready()
        #new_ida, token = mpi4jax.recv(ida, source=comm_jax.Get_rank(),comm=comm_jax, token=token)        
    except Exception as e:
        print(e)
        result = -1  #return_number_if_vmec_or_sfincs_not_converged
    return 





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






#QI cost function (A. Goodman)
optQI = partial(QuasiIsodynamicResidual,snorms=snorms, nphi=nphi_QI, nalpha=nalpha_QI, nBj=nBj_QI, mpol=mpol_QI, ntor=ntor_QI, nphi_out=nphi_out_QI, arr_out=arr_out_QI)
qi = make_optimizable(optQI, vmec)
#Elongation penalty in case it is necessary
partial_MaxElongationPen = partial(MaxElongationPen,t=maximum_elongation)
optElongation = make_optimizable(partial_MaxElongationPen, vmec)
#Mirror ration penalty
partial_MirrorRatioPen = partial(MirrorRatioPen,t=maximum_mirror)
optMirror = make_optimizable(partial_MirrorRatioPen, vmec)



#optNEO3=partial(WrapCostFunction,rho=rho3,nt=nt,nz=nz,nl=nl,mpi=mpi)
#Electron root cost functions at different radial positions (3 in this case)
neo_Er = make_optimizable(WrapCostFunction,vmec)

#Optimise total cost function (apect ration, iota, Electron root, QI, and mirror ration in this case)
prob = LeastSquaresProblem.from_tuples([(vmec.aspect, aspect_ratio_target_QI, 1),(vmec.mean_iota, iota_QI, weight_iota_QI),
                                        (neo_Er.J,20.,1),(qi.J, 0, qsqi_weight),(optMirror.J, 0, mirror_weight)])
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
proc0_print("Electron Root r3 before optimization:", neo_Er.J())
proc0_print("Mirror objective before optimization:", optMirror.J())
proc0_print("Total objective before optimization:", prob.objective())


# Evaluate cost function with least squares
#least_squares_mpi_solve(prob, mpi, grad=True, rel_step=1e-5, abs_step=1e-8)

# Make sure all procs participate in computing the objective:
prob.objective()

#Print the relevant individual cost functions (final)
proc0_print("Final aspect ratio:", vmec.aspect())
proc0_print("QI objective after optimization:", qi.J())
#proc0_print("Elogation objective after optimization:", optElongation.J())
proc0_print("Mirror objective after optimization:", optMirror.J())
proc0_print("Electron Root r1 after optimization:", neo_Er.J())
proc0_print("Total objective after optimization:", prob.objective())

proc0_print("End of Electron_root_optimization")
proc0_print("============================================")