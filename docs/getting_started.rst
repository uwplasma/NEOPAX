Getting started
===============

.. _installation:

Installation
------------

NEOPAX requires Python 3.11 or newer. VMEX and NTX are installed from their
upstream ``main`` branches; Boozer is also a declared dependency. Clone and
install NEOPAX with:

.. code-block:: console

    $ git clone https://github.com/uwplasma/NEOPAX.git
    $ cd NEOPAX
    $ pip install .

Geometry and NTX imports follow the active Python environment, including any
explicit ``PYTHONPATH``. NEOPAX does not prepend sibling dependency checkouts
or fall back to the retired ``vmec_jax`` package. For local dependency
development, install the intended checkouts in that environment, for example
``python -m pip install -e ../vmex-neopax-main -e ../NTX-neopax-main``.

The full transport reverse path requires the NEOPAX utility additions in
VMEX and NTX (block response, QI/max-J surrogate, factorized derivatives and
recorded-scan transposes). Until those utility PRs are merged into ``main``,
use the prepared dependency checkouts and install NEOPAX with
``python -m pip install --no-deps -e .`` after installing its requirements.

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
- :doc:`custom_models`

If you want to add your own transport flux or source models, NEOPAX also
supports Python-side registration and TOML-driven extension loading through an
``[extensions]`` section. See :doc:`custom_models`.
