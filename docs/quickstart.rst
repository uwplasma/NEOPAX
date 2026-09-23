Quickstart
==========

Run a transport input file with the module entry point:

.. code-block:: console

   python -m NEOPAX examples/Solve_Transport_Equations/Solve_Transport_equations_noHe_theta.toml

The command-line entry point is equivalent:

.. code-block:: console

   NEOPAX examples/Solve_Transport_Equations/Solve_Transport_equations_noHe_theta.toml

Most production runs are TOML driven.  The TOML selects:

- geometry and radial grid
- species and initial profiles
- neoclassical, turbulent, and classical flux models
- source models
- equations to evolve
- solver backend and output directory

Common workflows
----------------

Transport solve:

.. code-block:: console

   python -m NEOPAX examples/Solve_Transport_Equations/Solve_Transport_equations_wHe_radau.toml

Ambipolarity-only solve:

.. code-block:: console

   python -m NEOPAX my_case.toml --mode ambipolarity

Override solver settings from the CLI:

.. code-block:: console

   python -m NEOPAX my_case.toml --backend radau --t-final 0.1

For a fuller CLI/API comparison, see :doc:`methods_of_use`.

