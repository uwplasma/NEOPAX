Input File Reference
====================

NEOPAX is primarily configured through TOML input files. The most common input
files in the repository follow the same broad structure, with sections for
geometry, physics models, species, profiles, equations, solver settings, and
output controls.

This page is not meant to be an exhaustive key-by-key schema for every option
in the codebase. Its purpose is to document the main sections that appear in
normal workflows and explain what each section controls.


Top-Level Structure
-------------------

A typical NEOPAX input file contains some subset of the following sections:

- ``[general]``
- ``[geometry]``
- ``[neoclassical]``
- ``[turbulence]``
- ``[classical]``
- ``[ambipolarity]``
- ``[energy_grid]``
- ``[species]``
- ``[profiles]``
- ``[boundary.*]``
- ``[equations]``
- ``[sources]``
- ``[sources.parameters.*]``
- ``[transport_solver]``
- ``[transport_output]``
- ``[fluxes]``

Not every mode uses every section.


``[general]``
-------------

This section selects the top-level runtime mode.

Common key:

- ``mode``

Supported values:

- ``transport``
- ``ambipolarity``
- ``fluxes``
- ``sources``

Example:

.. code-block:: toml

    [general]
    mode = "transport"


``[geometry]``
--------------

This section defines the equilibrium files and radial resolution used to build
the stellarator geometry model.

Common keys:

- ``vmec_file``
- ``boozer_file``
- ``n_radial``

Example:

.. code-block:: toml

    [geometry]
    vmec_file = "./examples/inputs/wout_QI_nfp2_newNT_opt_hires.nc"
    boozer_file = "./examples/inputs/boozermn_wout_QI_nfp2_newNT_opt_hires.nc"
    n_radial = 51


``[neoclassical]``, ``[turbulence]``, ``[classical]``
------------------------------------------------------

These sections select the transport model used for each physics contribution.

Common keys include:

- ``flux_model``
- model-specific file paths or coefficients
- interpolation settings for database-driven models

Typical patterns:

.. code-block:: toml

    [neoclassical]
    flux_model = "ntx_database"
    entropy_model = "ntx_database"
    neoclassical_file = "./examples/inputs/Dij_NEOPAX_FULL_S_NEW_Er_Opt.h5"
    interpolation_mode = "preprocessed_3d_radial_ntss1d"

.. code-block:: toml

    [turbulence]
    flux_model = "turbulent_power_analytical"
    chi_density = [6.5e-05, 6.5e-05, 6.5e-05]
    chi_temperature = [0.0065, 0.0065, 0.0065]

.. code-block:: toml

    [classical]
    flux_model = "none"

When using file-based fluxes:

.. code-block:: toml

    [turbulence]
    flux_model = "fluxes_r_file"
    fluxes_file = "./outputs/my_fluxes.h5"
    debug_heat_flux_scale = 1.0


``[ambipolarity]``
------------------

This section configures the radial-electric-field root-finding workflow used in
``mode = "ambipolarity"`` and, optionally, for transport initialization.

Common keys include:

- ``er_ambipolar_method``
- scan bounds
- coarse/refined scan counts
- tolerances
- block size
- plotting/output options

Example:

.. code-block:: toml

    [ambipolarity]
    er_ambipolar_method = "two_stage"
    er_ambipolar_scan_min = -50.0
    er_ambipolar_scan_max = 50.0
    er_ambipolar_n_coarse = 80
    er_ambipolar_n_refine = 8
    er_ambipolar_max_roots = 3
    er_ambipolar_tol = 1.0e-6


``[energy_grid]``
-----------------

This section controls the monoenergetic or reduced velocity-space quadrature
used by the neoclassical pipeline.

Common key:

- ``n_x``

Example:

.. code-block:: toml

    [energy_grid]
    n_x = 3


``[species]``
-------------

This section defines the species composition and their basic physical
properties.

Common keys:

- ``n_species``
- ``names``
- ``mass_mp``
- ``charge_qp``

Example:

.. code-block:: toml

    [species]
    n_species = 3
    names = ["e", "D", "T"]
    mass_mp = [0.000544617, 2.0, 3.0]
    charge_qp = [-1.0, 1.0, 1.0]


``[profiles]``
--------------

This section defines the initial density, temperature, and electric-field
profiles.

Common keys include:

- profile model selection
- central/edge density and temperature
- species scaling factors
- shape parameters
- electric-field initialization mode

Example:

.. code-block:: toml

    [profiles]
    model = "standard_analytical"
    n0 = 4.21
    n_edge = 0.6
    T0 = 17.8
    T_edge = 0.7
    c_density = [1.0, 0.5, 0.5]
    c_temperature = [1.0, 1.0, 1.0]
    density_shape_power = 10.0
    temperature_shape_power = 2.0
    er_initialization_mode = "ambipolar_min_entropy"


``[boundary.*]``
----------------

Boundary conditions are split by field and side.

Typical subtrees include:

- ``[boundary.density.left]``
- ``[boundary.density.right]``
- ``[boundary.temperature.left]``
- ``[boundary.temperature.right]``
- ``[boundary.Er.left]``
- ``[boundary.Er.right]``

Supported boundary types commonly include:

- ``dirichlet``
- ``neumann``
- ``robin``
- ``floating_ambipolar_edge`` for ``E_r`` right boundary handling

Example:

.. code-block:: toml

    [boundary.Er.left]
    type = "dirichlet"
    value = 0.0

    [boundary.Er.right]
    type = "floating_ambipolar_edge"


``[equations]``
---------------

This section toggles which state blocks are actually evolved.

Common keys:

- ``toggle_density``
- ``toggle_temperature``
- ``toggle_Er``

Example:

.. code-block:: toml

    [equations]
    toggle_density = [false, false, false]
    toggle_temperature = [true, true, true]
    toggle_Er = true


``[sources]`` and ``[sources.parameters.*]``
--------------------------------------------

The ``[sources]`` section selects which source models are active. Additional
source-specific options live in ``[sources.parameters.<source_name>]`` blocks.

Example:

.. code-block:: toml

    [sources]
    temperature = ["power_exchange", "dt_reaction", "fusion_power_fraction_electrons", "bremsstrahlung_radiation"]

    [sources.parameters.power_exchange]
    mode = "all"
    coulomb_log_mode = "ntssfusion"

    [sources.parameters.bremsstrahlung_radiation]
    coefficient_mode = "ntssfusion"
    delta_zeff = 0.0


``[transport_solver]``
----------------------

This section controls the transport-time integrator and its numerical
parameters.

Common keys include:

- ``transport_solver_backend``
- ``integrator``
- ``theta_implicit``
- ``dt``
- ``t0``
- ``t_final``
- ``min_step``
- ``max_step``
- ``max_steps``
- ``nonlinear_solver_tol``
- ``nonlinear_solver_maxiter``
- ``save_n``
- density/temperature floors
- transport-physics toggles such as work/convection switches

Example:

.. code-block:: toml

    [transport_solver]
    transport_solver_backend = "theta_newton"
    integrator = "theta_newton"
    dt = 5.0e-5
    t0 = 0.0
    t_final = 20.0
    nonlinear_solver_tol = 1.0e-7
    nonlinear_solver_maxiter = 20
    save_n = 10


``[transport_output]``
----------------------

This section controls plotting, HDF5 writing, and transport-diagnostics output.

Common keys include:

- ``transport_plot``
- ``transport_write_hdf5``
- ``transport_output_dir``
- ``transport_plot_n_times``
- reference overlay files and labels

Example:

.. code-block:: toml

    [transport_output]
    transport_plot = true
    transport_write_hdf5 = false
    transport_output_dir = "./outputs/transport_noHe_theta"
    transport_plot_n_times = -1


``[fluxes]``
------------

This section is mainly used in ``mode = "fluxes"``.

Common keys include:

- ``fluxes_plot``
- ``fluxes_write_hdf5``
- ``fluxes_output_dir``
- optional reference overlays

Example:

.. code-block:: toml

    [fluxes]
    fluxes_plot = true
    fluxes_write_hdf5 = true
    fluxes_output_dir = "./outputs/my_flux_run"


Mode-to-Section Summary
-----------------------

Different runtime modes use different subsets of the input file.

``transport``
^^^^^^^^^^^^^

Most commonly uses:

- ``[general]``
- ``[geometry]``
- transport-model sections
- ``[energy_grid]``
- ``[species]``
- ``[profiles]``
- ``[boundary.*]``
- ``[equations]``
- ``[sources]``
- ``[transport_solver]``
- ``[transport_output]``

``ambipolarity``
^^^^^^^^^^^^^^^^

Most commonly uses:

- ``[general]``
- ``[geometry]``
- ``[neoclassical]``
- ``[ambipolarity]``
- ``[energy_grid]``
- ``[species]``
- ``[profiles]``

``fluxes``
^^^^^^^^^^

Most commonly uses:

- ``[general]``
- ``[geometry]``
- transport-model sections
- ``[energy_grid]``
- ``[species]``
- ``[profiles]``
- ``[fluxes]``

``sources``
^^^^^^^^^^^

Most commonly uses:

- ``[general]``
- ``[geometry]``
- ``[species]``
- ``[profiles]``
- ``[sources]``


See Also
--------

- :doc:`methods_of_use`
- :doc:`worked_examples`
