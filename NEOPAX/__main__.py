"""Main command line interface to ESSOS."""
import sys
# try: import tomllib
# except ModuleNotFoundError: import pip._vendor.tomli as tomllib
from time import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import near_axis, BiotSavart
from essos.dynamics import Tracing
from essos.optimization import optimize_loss_function
from essos.objective_functions import loss_coils_for_nearaxis


def main(cl_args=sys.argv[1:]):
    """
    Run the main ESSOS code from the command line.
    """
    # if len(cl_args) == 0:
    #     print("Using standard input parameters.")
    #     output = 0
    # else:
    #     parameters = tomllib.load(open(cl_args[0], "rb"))
    #     output = 0
        
    # Optimization parameters
    max_coil_length = 5.0
    max_coil_curvature = 4
    order_Fourier_series_coils = 5
    number_coil_points = order_Fourier_series_coils*10
    maximum_function_evaluations = 100
    number_coils_per_half_field_period = 3
    tolerance_optimization = 1e-8

    # Initialize Near-Axis field
    rc=jnp.array([1, 0.045])
    zs=jnp.array([0,-0.045])
    etabar=-0.9
    nfp=3
    field = near_axis(rc=rc, zs=zs, etabar=etabar, nfp=nfp)

    # Initialize coils
    current_on_each_coil = 17e5*field.B0/nfp/2
    number_of_field_periods = nfp
    major_radius_coils = field.R0[0]
    minor_radius_coils = major_radius_coils/1.5
    curves = CreateEquallySpacedCurves(n_curves=number_coils_per_half_field_period,
                                    order=order_Fourier_series_coils,
                                    R=major_radius_coils, r=minor_radius_coils,
                                    n_segments=number_coil_points,
                                    nfp=number_of_field_periods, stellsym=True)
    coils_initial = Coils(curves=curves, currents=[current_on_each_coil]*number_coils_per_half_field_period)

    # Optimize coils
    print(f'Optimizing coils with {maximum_function_evaluations} function evaluations.')
    time0 = time()
    coils_optimized = optimize_loss_function(loss_coils_for_nearaxis, initial_dofs=coils_initial.x,
                                    coils=coils_initial, tolerance_optimization=tolerance_optimization,
                                    maximum_function_evaluations=maximum_function_evaluations, field_nearaxis=field,
                                    max_coil_length=max_coil_length, max_coil_curvature=max_coil_curvature,)
    print(f"Optimization took {time()-time0:.2f} seconds")

    # Trace fieldlines
    nfieldlines = 6
    num_steps = 1000
    tmax = 150
    trace_tolerance = 1e-7

    R0 = jnp.linspace(field.R0[0], 1.05*field.R0[0], nfieldlines)
    Z0 = jnp.zeros(nfieldlines)
    phi0 = jnp.zeros(nfieldlines)
    initial_xyz=jnp.array([R0*jnp.cos(phi0), R0*jnp.sin(phi0), Z0]).T

    time0 = time()
    tracing_initial = Tracing(field=BiotSavart(coils_initial), model='FieldLine', initial_conditions=initial_xyz,
                    maxtime=tmax, timesteps=num_steps, tol_step_size=trace_tolerance)
    tracing_optimized = Tracing(field=BiotSavart(coils_optimized), model='FieldLine', initial_conditions=initial_xyz,
                    maxtime=tmax, timesteps=num_steps, tol_step_size=trace_tolerance)
    print(f"Tracing took {time()-time0:.2f} seconds")

    # Plot coils, before and after optimization
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    coils_initial.plot(ax=ax1, show=False)
    field.plot(ax=ax1, show=False, alpha=0.2)
    tracing_initial.plot(ax=ax1, show=False)
    coils_optimized.plot(ax=ax2, show=False)
    field.plot(ax=ax2, show=False, alpha=0.2)
    tracing_optimized.plot(ax=ax2, show=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])