import jax.numpy as jnp
from jax import config
# to use higher precision
config.update("jax_enable_x64", True)

import dataclasses
from jaxtyping import Array, Float  # https://github.com/google/jaxtyping


@dataclasses.dataclass(eq=False)
class Solver_Parameters:
    #Solver parameters class
    momentum_correction_flag: int
    integrator: str
    transport_solver_family: str
    transport_solver_backend: str
    nonlinear_solver_tol: float
    nonlinear_solver_maxiter: int
    anderson_history: int
    theta_implicit: float
    use_predictor_corrector: bool
    n_corrector_steps: int
    theta_ptc_enabled: bool
    theta_ptc_dt_min_factor: float
    theta_ptc_dt_max_factor: float
    theta_ptc_growth: float
    theta_ptc_shrink: float
    theta_line_search_enabled: bool
    theta_line_search_contraction: float
    theta_line_search_min_alpha: float
    theta_line_search_c: float
    theta_max_step_retries: int
    theta_linear_solver: str
    theta_gmres_tol: float
    theta_gmres_maxiter: int
    theta_trust_region_enabled: bool
    theta_trust_radius: float
    theta_homotopy_steps: int
    theta_differentiable_mode: bool
    er_ambipolar_scan_min: float
    er_ambipolar_scan_max: float
    er_ambipolar_n_scan: int
    er_ambipolar_tol: float
    er_ambipolar_maxiter: int
    er_ambipolar_n_coarse: int
    er_ambipolar_n_fine: int
    er_ambipolar_method: str
    neoclassical_transport_model: str
    turbulent_transport_model: str
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
    er_mode: str
    use_ap_er_preconditioner: bool
    on_OmegaC: float   #Keep zero!!, parameter for testing  
    # Generic evolution controls for any number of species.
    evolve_Er: bool
    evolve_density: Float[Array, "..."]
    evolve_temperature: Float[Array, "..."]
    density_floor: float | Array | None
    temperature_floor: float | Array | None
    # Turbulent transport coefficients
    chi_temperature: Float[Array, "..."]
    chi_density: Float[Array, "..."]
     
    def __init__(self,t0=None,t_final=None,dt=None,ts_list=None,rtol=None,
                        atol=None,momentum_correction_flag=None,DEr=None,Er_relax=None,
                        er_mode=None,use_ap_er_preconditioner=None,
                        evolve_Er=None,evolve_density=None,evolve_temperature=None,
                        density_floor=None,temperature_floor=None,
                        integrator=None,transport_solver_family=None,transport_solver_backend=None,
                        nonlinear_solver_tol=None,nonlinear_solver_maxiter=None,anderson_history=None,
                        theta_implicit=None,use_predictor_corrector=None,n_corrector_steps=None,
                        theta_ptc_enabled=None,theta_ptc_dt_min_factor=None,theta_ptc_dt_max_factor=None,
                        theta_ptc_growth=None,theta_ptc_shrink=None,
                        theta_line_search_enabled=None,theta_line_search_contraction=None,
                        theta_line_search_min_alpha=None,theta_line_search_c=None,
                        theta_max_step_retries=None,
                        theta_linear_solver=None,theta_gmres_tol=None,theta_gmres_maxiter=None,
                        theta_trust_region_enabled=None,theta_trust_radius=None,
                        theta_homotopy_steps=None,
                        theta_differentiable_mode=None,
                        er_ambipolar_scan_min=None,er_ambipolar_scan_max=None,er_ambipolar_n_scan=None,
                        er_ambipolar_tol=None,er_ambipolar_maxiter=None,
                        er_ambipolar_n_coarse=None,er_ambipolar_n_fine=None,
                        er_ambipolar_method=None,
                        neoclassical_transport_model=None,turbulent_transport_model=None,
                        chi_temperature=None,chi_density=None,on_OmegaC=None):
        

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

        if integrator is None:
            self.integrator = "diffrax_kvaerno5"
        else:
            self.integrator = str(integrator)

        if transport_solver_family is None:
            self.transport_solver_family = "auto"
        else:
            self.transport_solver_family = str(transport_solver_family)

        if transport_solver_backend is None:
            # Preserve legacy behavior where `integrator` picks the active backend.
            self.transport_solver_backend = str(self.integrator)
        else:
            self.transport_solver_backend = str(transport_solver_backend)

        if nonlinear_solver_tol is None:
            self.nonlinear_solver_tol = 1.e-8
        else:
            self.nonlinear_solver_tol = float(nonlinear_solver_tol)

        if nonlinear_solver_maxiter is None:
            self.nonlinear_solver_maxiter = 50
        else:
            self.nonlinear_solver_maxiter = int(nonlinear_solver_maxiter)

        if anderson_history is None:
            self.anderson_history = 5
        else:
            self.anderson_history = int(anderson_history)

        if theta_implicit is None:
            self.theta_implicit = 1.0
        else:
            self.theta_implicit = float(theta_implicit)

        if use_predictor_corrector is None:
            self.use_predictor_corrector = True
        else:
            self.use_predictor_corrector = bool(use_predictor_corrector)

        if n_corrector_steps is None:
            self.n_corrector_steps = 1
        else:
            self.n_corrector_steps = int(n_corrector_steps)

        if theta_ptc_enabled is None:
            self.theta_ptc_enabled = True
        else:
            self.theta_ptc_enabled = bool(theta_ptc_enabled)

        if theta_ptc_dt_min_factor is None:
            self.theta_ptc_dt_min_factor = 1.0e-4
        else:
            self.theta_ptc_dt_min_factor = float(theta_ptc_dt_min_factor)

        if theta_ptc_dt_max_factor is None:
            self.theta_ptc_dt_max_factor = 1.0e3
        else:
            self.theta_ptc_dt_max_factor = float(theta_ptc_dt_max_factor)

        if theta_ptc_growth is None:
            self.theta_ptc_growth = 1.5
        else:
            self.theta_ptc_growth = float(theta_ptc_growth)

        if theta_ptc_shrink is None:
            self.theta_ptc_shrink = 0.5
        else:
            self.theta_ptc_shrink = float(theta_ptc_shrink)

        if theta_line_search_enabled is None:
            self.theta_line_search_enabled = True
        else:
            self.theta_line_search_enabled = bool(theta_line_search_enabled)

        if theta_line_search_contraction is None:
            self.theta_line_search_contraction = 0.5
        else:
            self.theta_line_search_contraction = float(theta_line_search_contraction)

        if theta_line_search_min_alpha is None:
            self.theta_line_search_min_alpha = 1.0e-4
        else:
            self.theta_line_search_min_alpha = float(theta_line_search_min_alpha)

        if theta_line_search_c is None:
            self.theta_line_search_c = 1.0e-4
        else:
            self.theta_line_search_c = float(theta_line_search_c)

        if theta_max_step_retries is None:
            self.theta_max_step_retries = 8
        else:
            self.theta_max_step_retries = int(theta_max_step_retries)

        if theta_linear_solver is None:
            self.theta_linear_solver = "direct"
        else:
            self.theta_linear_solver = str(theta_linear_solver)

        if theta_gmres_tol is None:
            self.theta_gmres_tol = 1.0e-8
        else:
            self.theta_gmres_tol = float(theta_gmres_tol)

        if theta_gmres_maxiter is None:
            self.theta_gmres_maxiter = 200
        else:
            self.theta_gmres_maxiter = int(theta_gmres_maxiter)

        if theta_trust_region_enabled is None:
            self.theta_trust_region_enabled = False
        else:
            self.theta_trust_region_enabled = bool(theta_trust_region_enabled)

        if theta_trust_radius is None:
            self.theta_trust_radius = 1.0
        else:
            self.theta_trust_radius = float(theta_trust_radius)

        if theta_homotopy_steps is None:
            self.theta_homotopy_steps = 1
        else:
            self.theta_homotopy_steps = int(theta_homotopy_steps)

        if theta_differentiable_mode is None:
            self.theta_differentiable_mode = False
        else:
            self.theta_differentiable_mode = bool(theta_differentiable_mode)

        if er_ambipolar_scan_min is None:
            self.er_ambipolar_scan_min = -20.0
        else:
            self.er_ambipolar_scan_min = float(er_ambipolar_scan_min)

        if er_ambipolar_scan_max is None:
            self.er_ambipolar_scan_max = 20.0
        else:
            self.er_ambipolar_scan_max = float(er_ambipolar_scan_max)

        if er_ambipolar_n_scan is None:
            self.er_ambipolar_n_scan = 96
        else:
            self.er_ambipolar_n_scan = int(er_ambipolar_n_scan)

        if er_ambipolar_tol is None:
            self.er_ambipolar_tol = 1.e-6
        else:
            self.er_ambipolar_tol = float(er_ambipolar_tol)

        if er_ambipolar_maxiter is None:
            self.er_ambipolar_maxiter = 30
        else:
            self.er_ambipolar_maxiter = int(er_ambipolar_maxiter)

        if er_ambipolar_n_coarse is None:
            self.er_ambipolar_n_coarse = 24
        else:
            self.er_ambipolar_n_coarse = int(er_ambipolar_n_coarse)

        if er_ambipolar_n_fine is None:
            self.er_ambipolar_n_fine = 8
        else:
            self.er_ambipolar_n_fine = int(er_ambipolar_n_fine)

        if er_ambipolar_method is None:
            self.er_ambipolar_method = "jit_multires"
        else:
            self.er_ambipolar_method = str(er_ambipolar_method)

        if neoclassical_transport_model is None:
            self.neoclassical_transport_model = "neoclassical"
        else:
            self.neoclassical_transport_model = str(neoclassical_transport_model)

        if turbulent_transport_model is None:
            self.turbulent_transport_model = "from_turbulence_state"
        else:
            self.turbulent_transport_model = str(turbulent_transport_model)

        if DEr is None:
            self.DEr = 2.
        else:
            self.DEr=DEr 


        if Er_relax is None:
            self.Er_relax = 0.1
        else:
            self.Er_relax=Er_relax

        if er_mode is None:
            self.er_mode = "diffusion"
        else:
            self.er_mode = str(er_mode)

        if use_ap_er_preconditioner is None:
            self.use_ap_er_preconditioner = False
        else:
            self.use_ap_er_preconditioner = bool(use_ap_er_preconditioner)

        if evolve_Er is None:
            self.evolve_Er = True
        else:
            self.evolve_Er = bool(evolve_Er)

        if evolve_density is None:
            self.evolve_density = jnp.asarray([], dtype=bool)
        else:
            self.evolve_density = jnp.asarray(evolve_density, dtype=bool)

        if evolve_temperature is None:
            self.evolve_temperature = jnp.asarray([], dtype=bool)
        else:
            self.evolve_temperature = jnp.asarray(evolve_temperature, dtype=bool)

        if density_floor is None:
            self.density_floor = None
        else:
            self.density_floor = jnp.asarray(density_floor)

        if temperature_floor is None:
            self.temperature_floor = None
        else:
            self.temperature_floor = jnp.asarray(temperature_floor)
                                                
        if chi_temperature is None:
            self.chi_temperature = jnp.ones(3) * 0.5
        else:
            self.chi_temperature = jnp.asarray(chi_temperature)

        if chi_density is None:
            self.chi_density = jnp.zeros(3)
        else:
            self.chi_density = jnp.asarray(chi_density)

        if on_OmegaC is None:
            self.on_OmegaC = 0
        else:
            self.on_OmegaC=on_OmegaC
