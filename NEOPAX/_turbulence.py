import os
import h5py as h5
import jax
import jax.numpy as jnp
from jax import config
# to use higher precision
config.update("jax_enable_x64", True)
from jaxtyping import Array, Float  # https://github.com/google/jaxtyping
import equinox as eqx
import interpax
current_path = os.path.dirname(os.path.realpath(__name__))
import numpy as np
from ._constants import proton_mass, elementary_charge

#Monoenergetic database class
class Turbulence(eqx.Module):
    """Turbulent transport quantities

    """

    # note: assumes (psi, theta, zeta) coordinates, not (rho, theta, zeta)
    nspecies: int
    nradial: int
    Qa_turb :  Float[Array, "..."] #Float[Array, "nspecies x nradial"]  #int = eqx.field(static=True)

    def __init__(
        self,
        nspecies,
        nradial,
        Qa_turb,
    ):

        self.nspecies=nspecies
        self.nradial=nradial
        self.Qa_turb=Qa_turb



    @classmethod
    def from_analytical_model(cls,
        species,
        chi,                   
    ):

        Qa_turb=jnp.zeros((species.number_species,species.radial_points))
        for i in range(species.number_species):
            Qa_turb=Qa_turb.at[i].set(-chi[i]*species.dTdr[i,:])


        data = {}
        data["nspecies"]=species.number_species
        data["nradial"] = species.radial_points
        data["Qa_turb"] = Qa_turb


        return cls(**data)


    @classmethod
    def from_gx_inductiva(cls,
        species,
        field,
        vmec,
       # rho,
    ):
        """Construct Turbulent Heat Flux  

        Parameters
        ----------
        vmec_wout path : path-like
            Path to vmec wout file.
        """

        import inductiva
        import sys
        import os
        from netCDF4 import Dataset
        import h5py as h5
        my_gpu = inductiva.resources.machine_groups.get_by_name("eduardo-gpu-rtx4070")
        #gpu_machine_group = inductiva.resources.MachineGroup("g2-standard-4")


        #Write input sfincs
        def write_GX_input(wout_path,s=0.64,Ti=1.0,ni=1.0,Ti_prime=3.0,ni_prime=1.0,
                        ntheta=96,nperiod=1,ny=64,nx=64,nhermite=8,nlaguerre=4,nspecies=1,beta=0.0,alpha=0.0,t_max=500.0,ne=1.0,Te=1.0):

            return f"""
        # This test runs a nonlinear ITG turbulence calculation using W7-X geometry.
        # This test uses a Boltzmann adiabatic electron response.
        # Parameters similar to the ones used by M. Landremann ML database of GX

        debug = false

        [Dimensions]
        ntheta = 48            # number of points along field line (theta) per 2pi segment
        #ntheta = {ntheta}
        nperiod = 1            # number of 2pi segments along field line is 2*nperiod-1
        ny = {ny}                # number of real-space grid-points in y
        nx = {nx}                # number of real-space grid-points in x

        nhermite = {nhermite}           # number of hermite moments (v_parallel resolution)
        nlaguerre = {nlaguerre}          # number of laguerre moments (mu B resolution)
        nspecies = {nspecies}           # number of evolved kinetic species (adiabatic electrons don't count towards nspecies)
        
        [Domain]
        y0 = 21.0                 # controls box length in y (in units of rho_ref) and minimum ky, so that ky_min*rho_ref = 1/y0 
        boundary = "fix aspect"   # use twist-shift boundary conditions along field line, and cut field line so that x0 ~ y0
        #boundary = "periodic"
        #x0 = 10.0
        
        [Physics]
        beta = {beta}                     # reference normalized pressure, beta = n_ref T_ref / ( B_ref^2 / (8 pi))
        nonlinear_mode = true           # this is a nonlinear calculation

        [Time]
        t_max = {t_max}        # run to t = t_max (units of L_ref/vt_ref)
        dt = 0.1             # maximum timestep (units of L_ref/vt_ref), will be adapted dynamically
        cfl = 0.9            # safety cushion factor on timestep size
        scheme = "rk3"       # use SSPx3 timestepping scheme

        [Initialization]
        ikpar_init = 0                  # parallel wavenumber of initial perturbation
        init_field = "density"          # initial condition set in density
        init_amp = 1.0e-3               # amplitude of initial condition

        [Geometry]
        geo_option = "vmec"           # use VMEC geometry
        #geo_option = "eik"
        # Name of the vmec file
        vmec_file = "{wout_path}"
        #geo_file = "{wout_path}"

        #Field line label alpha_t = theta - iota * zeta. alpha = 0.0 usually corresponds to a
        #field line on the outboard side
        alpha = 0.0

        # Number of poloidal turns (will be reduced for boundary = "fix aspect")
        # The field line goes from (-npol*PI, npol*PI]
        npol = 1

        # Normalized toroidal flux (or s) is how vmec labels surfaces.
        # s goes from [0,1] where 0 is the magnetic axis and 1 is the
        # last closed flux surface.
        desired_normalized_toroidal_flux = {s}

        # it is okay to have extra species data here; only the first nspecies elements of each item are used
        [species]
        z     = [ 1.0,      -1.0     ]         # charge (normalized to Z_ref)
        mass  = [ 1.0,       2.7e-4  ]         # mass (normalized to m_ref)
        dens  = [ {ni},       {ne}     ]         # density (normalized to dens_ref)
        temp  = [ {Ti},       {Te}     ]         # temperature (normalized to T_ref)
        tprim = [ {Ti_prime},       0.0     ]         # temperature gradient, L_ref/L_T
        fprim = [ {ni_prime},       0.0     ]         # density gradient, L_ref/L_n
        vnewk = [ 0.01,      0.0     ]         # collision frequency
        type  = [ "ion",  "electron" ]         # species type
        
        [Boltzmann]
        add_Boltzmann_species = true    # use a Boltzmann species
        Boltzmann_type = "electrons"    # the Boltzmann species will be electrons
        tau_fac = 1.0                   # temperature ratio, T_i/T_e

        [Dissipation]
        closure_model = "none"          # no closure assumptions (just truncation)
        hypercollisions = true          # use hypercollision model (with default parameters)
        hyper = true                    # use hyperdiffusion
        D_hyper = 0.05                  # coefficient of hyperdiffusion

        [Restart]
        # restart = true
        save_for_restart = true

        [Diagnostics]
        nwrite = 50                    # write diagnostics every nwrite timesteps
        free_energy = true             # compute and write free energy spectra (Wg, Wphi, Phi**2)
        fluxes = true                  # compute and write flux spectra (Q, Gamma)
        fields = false                  # write fields on the grid
        moments = false                 # write moments on the grid


        """


        #Write input file in a folder
        def write_input_file(filepath: str, content: str):
            with open(filepath, "w") as f:
                f.write(content)


        def heat_flux(data, ispec=0, navgfac=0.5, label=None, plot=True, fig=None, Lref="a", refsp=None):
            # read data from file
            t = data.variables['time'][:]
            try:
                q = data.groups['Fluxes'].variables['qflux'][:,ispec]
            except:
                print('Error: heat flux data was not written. Make sure to use \'fluxes = true\' in the input file.')
            species_type = data.groups['Inputs'].groups['Species'].variables['species_type'][ispec]
            if species_type == 0:
                species_tag = "i"
            elif species_type == 1:
                species_tag = "e"
            if refsp == None:
                refsp = species_tag

            # compute time-average and std dev
            istart_avg = int(len(t)*navgfac)
            qavg = np.mean(q[istart_avg:])
            qstd = np.std(q[istart_avg:])
            return qavg,qstd 

        indeces=jnp.array([5,12,25,37,42])
        Qi_turb=np.zeros(len(indeces))
        rho=jnp.zeros(len(indeces))

        drds=field.a_b/(2.*field.rho_grid)
        Ti=species.temperature[1,:]/species.temperature[1,0]
        ni=species.density[1,:]/species.density[1,0]
        Te=species.temperature[0,:]/species.temperature[1,0]
        ne=species.density[0,:]/species.density[1,0]
        Ti_prime=-species.dTdr[1,:]/species.temperature[1,0]*drds
        ni_prime=-species.dndr[1,:]/species.density[1,0]*drds
        wout_path=vmec.split('/')[-1]
        print(wout_path)
        workdir=os.path.join(current_path, "inputs/")
        for i in range(len(indeces)):

            rho=rho.at[i].set(field.rho_grid[indeces[i]])
            s=field.rho_grid[indeces[i]]**2
            gx_content_body = write_GX_input(wout_path, s=s, Ti=Ti[indeces[i]], 
                                             ni=ni[indeces[i]], 
                                             Ti_prime=Ti_prime[indeces[i]],
                                             ni_prime=ni_prime[indeces[i]], 
                                             ne=ne[indeces[i]],
                                             Te=Te[indeces[i]])

            input_file='input_test_0p'+str(s).split('.')[1]+'.in'
            write_input_file(os.path.join(workdir, input_file), gx_content_body)

            gx = inductiva.simulators.GX()
            task = gx.run(input_dir=workdir,sim_config_filename=input_file,on=my_gpu)
            #task = gx.run(input_dir=workdir,sim_config_filename=input_file,on=gpu_machine_group)

            task.wait()

            output_dir=task.download_outputs()
            print('Output_dir',output_dir)

            task.print_summary()


            #print()
            output_name=input_file.split('.')[0]+'.nc'
            
            filename=os.path.join(output_dir,output_name)


            try:
                data = Dataset(filename, mode='r')
            except:
                print('File not found')

            nspec = data.dimensions['s'].size

            for ispec in np.arange(nspec):
                Qi_turb[i],_=heat_flux(data, ispec=ispec, refsp="i")
            

        Gx_file=h5.File('Gx_data.h5','w')
        Gx_file['rho']=rho
        Gx_file['Qi_turb']=Qi_turb
        Gx_file.close()

        vth_r=species.v_thermal[1,0]*jnp.sqrt(2.) #sqrt(2) factor for accounting for deuterium mass
        Omega_c_r=elementary_charge*field.B0[0]/proton_mass
        T_r=species.temperature[1,0]
        n_r=species.density[1,0]
        L_r=field.a_b
        GB_normalization=vth_r**2/Omega_c_r**2/L_r**2*vth_r*n_r*T_r
        Qi_turb_interpolated=interpax.Interpolator1D(rho*field.a_b,jnp.array(Qi_turb),extrap=True)(field.r_grid)*GB_normalization
        Qa_turb=jnp.zeros((species.number_species,species.radial_points))
        Qa_turb=Qa_turb.at[0,:].set(Qi_turb_interpolated*2.)
        Qa_turb=Qa_turb.at[1,:].set(Qi_turb_interpolated)
        Qa_turb=Qa_turb.at[2,:].set(Qi_turb_interpolated)

        data = {}
        data["nspecies"]=species.number_species
        data["nradial"] = species.radial_points
        data["Qa_turb"] = Qa_turb

        return cls(**data)

