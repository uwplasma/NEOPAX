import jax.numpy as jnp
from jax.lax import cond
from functools import partial
from jax.random import normal
from jax import lax, jit, vmap, config
from jax.debug import print as jprint
try: import tomllib
except ModuleNotFoundError: import pip._vendor.tomli as tomllib
config.update("jax_enable_x64", True)

def load_parameters(input_file):
    """
    Load parameters from a TOML file.

    Parameters:
    ----------
    input_file : str
        Path to the TOML file containing simulation parameters.

    Returns:
    -------
    parameters : dict
        Dictionary containing simulation parameters.
    """
    parameters = tomllib.load(open(input_file, "rb"))
    input_parameters = parameters['input_parameters']
    species_parameters = parameters['species_parameters']
    grid_parameters = parameters['grid_parameters']
    solver_parameters = parameters['solver_parameters']    
    return input_parameters,species_parameters, grid_parameters, solver_parameters

def initialize_simulation_parameters(user_parameters={}):
    """
    Initialize the simulation parameters for a particle-in-cell simulation, 
    combining user-provided values with predefined defaults. This function 
    ensures all required parameters are set and automatically calculates 
    derived parameters based on the inputs.

    The function uses lambda functions to define derived parameters that 
    depend on other parameters. These lambda functions are evaluated after 
    merging user-provided parameters with the defaults, ensuring derived 
    parameters are consistent with any overrides.

    Parameters:
    ----------
    user_parameters : dict
        Dictionary containing user-specified parameters. Any parameter not provided
        will default to predefined values.

    Returns:
    -------
    parameters : dict
        Dictionary containing all simulation parameters, with user-provided values
        overriding defaults.
    """
    # Define all default parameters in a single dictionary
    default_parameters = {
        # Basic simulation settings
        "length": 1e-2,                           # Dimensions of the simulation box
        "amplitude_perturbation_x": 5e-4,          # Amplitude of sinusoidal (sin) perturbation in x
        "wavenumber_electrons": 8,    # Wavenumber of sinusoidal electron density perturbation in x (factor of 2pi/length)
        "wavenumber_ions": 0,    # Wavenumber of sinusoidal ion density perturbation in x (factor of 2pi/length)
        "grid_points_per_Debye_length": 2,        # dx over Debye length
        "vth_electrons_over_c": 0.05,             # Thermal velocity of electrons over speed of light
        "ion_temperature_over_electron_temperature": 0.01, # Temperature ratio of ions to electrons
        "timestep_over_spatialstep_times_c": 1.0, # dt * speed_of_light / dx
        "seed": 1701,                             # Random seed for reproducibility
        "electron_drift_speed": 0,                # Drift speed of electrons
        "ion_drift_speed":      0,                # Drift speed of ions
        "velocity_plus_minus_electrons": False,   # create two groups of electrons moving in opposite directions
        "velocity_plus_minus_ions": False,        # create two groups of electrons moving in opposite directions
        "print_info": True,                       # Print information about the simulation
        
        # Boundary conditions
        "particle_BC_left":  0,                   # Left boundary condition for particles
        "particle_BC_right": 0,                   # Right boundary condition for particles
        "field_BC_left":     0,                   # Left boundary condition for fields
        "field_BC_right":    0,                   # Right boundary condition for fields
        
        # External fields (initialized to zero)
        "external_electric_field_amplitude": 0, # Amplitude of sinusoidal (cos) perturbation in x
        "external_electric_field_wavenumber": 0, # Wavenumber of sinusoidal (cos) perturbation in x (factor of 2pi/length)
        "external_magnetic_field_amplitude": 0, # Amplitude of sinusoidal (cos) perturbation in x
        "external_magnetic_field_wavenumber": 0, # Wavenumber of sinusoidal (cos) perturbation in x (factor of 2pi/length)
    }

    # Merge user-provided parameters into the default dictionary
    parameters = {**default_parameters, **user_parameters}
    
    # Compute derived parameters based on user-provided or default values
    for key, value in parameters.items():
        if callable(value):  # If the value is a lambda function, compute it
            parameters[key] = value(parameters)

    return parameters