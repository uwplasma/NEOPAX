Worked Examples
===============

This page collects a few representative NEOPAX workflows using the example
input files already shipped in the repository. The goal is not to document
every option in each file, but to show the normal ways users interact with the
code: transport evolution, ambipolar root finding, flux evaluation, and source
evaluation.


Transport Example
-----------------

Example input files:

- ``examples/Solve_Transport_Equations/Solve_Transport_equations_noHe_theta.toml``
- ``examples/Solve_Transport_Equations/Solve_Transport_equations_noHe_radau.toml``
- ``examples/Solve_Transport_Equations/Solve_Transport_equations_noHe_diffrax_kvaerno.toml``

These examples evolve 1D transport equations for the active profile state using
different time-integration backends.

CLI usage:

.. code-block:: console

    NEOPAX examples/Solve_Transport_Equations/Solve_Transport_equations_noHe_theta.toml

Module usage:

.. code-block:: console

    python -m NEOPAX examples/Solve_Transport_Equations/Solve_Transport_equations_noHe_radau.toml

Direct Python API:

.. code-block:: python

    import NEOPAX

    result = NEOPAX.run(
        "examples/Solve_Transport_Equations/Solve_Transport_equations_noHe_radau.toml",
        backend="radau",
    )

    final_state = result.final_state
    time_grid = result.time_grid

Useful overrides during development:

- ``--backend theta_newton``
- ``--dt 1e-4``
- ``--t-final 5.0``
- ``--output-dir ./outputs/my_transport_run``
- ``--set turbulence.debug_heat_flux_scale=0.5``


Ambipolarity Example
--------------------

Example input file:

- ``examples/Solve_Ambipolarity/ambiplarity_benchmark.toml``

This mode scans radial electric field values and identifies ambipolar roots,
including the entropy-based root selection logic when multiple roots are
present.

CLI usage:

.. code-block:: console

    NEOPAX examples/Solve_Ambipolarity/ambiplarity_benchmark.toml

Direct Python API:

.. code-block:: python

    import NEOPAX

    result = NEOPAX.run("examples/Solve_Ambipolarity/ambiplarity_benchmark.toml")

    roots = result.raw_result.get("roots_3")
    best_root = result.raw_result.get("best_root")

Typical reasons to use this mode:

- inspect ambipolar root structure before transport evolution
- initialize ``E_r`` from an ambipolar branch
- benchmark root-finding settings and neoclassical databases


Fluxes Example
--------------

Example input file:

- ``examples/examples/Calculate_Fluxes/calculate_flux_w7x.toml``

This mode evaluates the currently configured flux model on the input state and
optionally writes plots and HDF5 output.

CLI usage:

.. code-block:: console

    NEOPAX examples/examples/Calculate_Fluxes/calculate_flux_w7x.toml

CLI usage with an override:

.. code-block:: console

    NEOPAX examples/examples/Calculate_Fluxes/calculate_flux_w7x.toml --set fluxes.fluxes_write_hdf5=true

Direct Python API:

.. code-block:: python

    import NEOPAX

    result = NEOPAX.run("examples/examples/Calculate_Fluxes/calculate_flux_w7x.toml")
    fluxes = result.raw_result

This mode is especially useful for:

- validating neoclassical, turbulent, or classical flux models in isolation
- comparing file-driven fluxes against database-driven models
- generating reference flux profiles for debugging transport runs


Sources Example
---------------

The ``sources`` mode evaluates configured source models without running the full
transport solve. If you already have a normal transport input file, the
simplest way to explore sources-only behavior is to override the mode at
runtime.

CLI usage:

.. code-block:: console

    NEOPAX examples/Solve_Transport_Equations/Solve_Transport_equations_noHe_theta.toml --mode sources

Direct Python API:

.. code-block:: python

    import NEOPAX

    result = NEOPAX.run(
        "examples/Solve_Transport_Equations/Solve_Transport_equations_noHe_theta.toml",
        mode="sources",
    )

    sources = result.raw_result

This is useful for:

- checking source magnitudes and signs before transport runs
- isolating heating, radiation, and exchange terms
- validating source parameter changes independently from transport stiffness


Choosing A Usage Path
---------------------

For the same example file, NEOPAX supports three complementary ways of running:

- Console script: fastest for command-line use and batch scripts
- ``python -m NEOPAX``: useful when you want the module path explicitly
- Direct API: best for notebooks, programmatic workflows, parameter scans, and
  future autodiff-oriented use

In practice:

- use the CLI when the TOML file is the main source of truth
- use ``NEOPAX.run(...)`` when Python should remain in control of the workflow
- use ``NEOPAX.prepare_config(...)`` plus ``NEOPAX.run_config(...)`` when you
  want an explicit configuration object before execution


See Also
--------

- :doc:`methods_of_use`
- :doc:`input_file_reference`
- :doc:`overview`
