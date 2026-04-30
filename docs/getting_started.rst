Getting started
=====

.. _installation:

Installation
------------

To use NEOPAX, there is no need to install it.
You can simply clone the repository and install it using the following command:

.. code-block:: console

    $ git clone https://github.com/uwplasma/NEOPAX.git
    $ cd NEOPAX
    $ pip install .

Run an example
--------------

To run the one of the examples, use the following command:

.. code-block:: console

    python examples/Calculate_Fluxes/Fluxes_Calculation.py

More examples are in subfolders of the `examples` folder.

CLI and API entry paths
-----------------------

After installation, NEOPAX can be run through either the console-script path,
the module path, or the direct Python API.

Console script:

.. code-block:: console

    NEOPAX examples/Solve_Transport_Equations/Solve_Transport_equations_noHe_theta.toml

Module path:

.. code-block:: console

    python -m NEOPAX examples/Solve_Transport_Equations/Solve_Transport_equations_noHe_theta.toml

Direct Python API:

.. code-block:: python

    import NEOPAX

    config = NEOPAX.load_config("examples/Solve_Transport_Equations/Solve_Transport_equations_noHe_theta.toml")
    result = NEOPAX.run_config(config)

The direct Python API is the preferred entry path when NEOPAX is used inside a
larger JAX workflow or when preserving a programmatic/autodiff-friendly call
path matters.

For a more complete description of:

- CLI overrides
- ``NEOPAX.run(...)``
- ``NEOPAX.prepare_config(...)``
- ``NEOPAX.run_config(...)``
- and recommended usage patterns

see :doc:`methods_of_use`.

For configuration details and example-driven walkthroughs, see also:

- :doc:`input_file_reference`
- :doc:`worked_examples`
