Methods of Use
==============

NEOPAX supports two main ways of running the code:

- a **CLI path** for normal production runs, scripting, and quick overrides
- a **direct Python API path** for notebooks, automation, and JAX-oriented
  programmatic workflows

These two paths share the same underlying runtime logic. The CLI is a thin
configuration override layer on top of the same core execution functions used
by the Python API.


Recommended Use Cases
---------------------

Use the CLI when:

- you already have a TOML case file
- you want quick command-line overrides
- you are launching batch jobs or shell scripts
- you want the most familiar user-facing workflow

Use the direct Python API when:

- you want to build or modify configs in memory
- you are embedding NEOPAX inside a larger Python workflow
- you want to inspect returned objects directly
- you want to preserve a cleaner path for future autodiff-sensitive workflows


CLI Path
--------

Console-script entry
^^^^^^^^^^^^^^^^^^^^

After installation, the package exposes console scripts:

.. code-block:: console

    NEOPAX my_case.toml

or:

.. code-block:: console

    neopax my_case.toml

Module entry
^^^^^^^^^^^^

The module entry path is equivalent:

.. code-block:: console

    python -m NEOPAX my_case.toml

Common CLI overrides
^^^^^^^^^^^^^^^^^^^^

The current CLI supports common overrides for frequently changed runtime
parameters.

Examples:

.. code-block:: console

    NEOPAX my_case.toml --mode fluxes

.. code-block:: console

    NEOPAX my_case.toml --vmec-file ./wout.nc --boozer-file ./booz.nc --n-radial 65

.. code-block:: console

    NEOPAX my_case.toml --n-x 5 --backend radau --dt 1e-4 --t-final 10.0

.. code-block:: console

    NEOPAX my_case.toml --output-dir ./outputs/debug_run

Generic dotted overrides
^^^^^^^^^^^^^^^^^^^^^^^^

For anything not covered by a dedicated CLI flag, use repeated ``--set``
arguments:

.. code-block:: console

    NEOPAX my_case.toml --set turbulence.debug_heat_flux_scale=0.5

.. code-block:: console

    NEOPAX my_case.toml --set transport_solver.dt=1e-4 --set transport_solver.throw=true

.. code-block:: console

    NEOPAX my_case.toml --set general.mode=ambipolarity

The CLI override layer currently understands simple scalars:

- booleans like ``true`` / ``false``
- integers
- floats
- strings
- ``none`` / ``null``

CLI design note
^^^^^^^^^^^^^^^

The CLI is intentionally not a second physics runtime. It only:

1. loads a TOML config
2. applies overrides
3. calls the same core NEOPAX execution path as the Python API


Direct Python API
-----------------

High-level convenience API
^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest direct entry point is:

.. code-block:: python

    import NEOPAX

    result = NEOPAX.run(
        "examples/Solve_Transport_Equations/Solve_Transport_equations_noHe_theta.toml",
        backend="radau",
        n_radial=65,
        n_x=5,
        dt=1e-4,
    )

This mirrors the common CLI overrides while staying inside Python.

What ``NEOPAX.run(...)`` returns
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``NEOPAX.run(...)`` returns a structured ``RunResult`` object with fields such
as:

- ``mode``
- ``config``
- ``raw_result``
- ``final_state``
- ``saved_states``
- ``time_grid``
- ``saved_step_sizes``
- ``accepted_mask``
- ``failed_mask``
- ``fail_codes``
- ``n_steps``
- ``done``
- ``failed``
- ``fail_code``
- ``final_time``
- ``rho``
- ``output_dir``

For a transport run, the most commonly used fields are:

.. code-block:: python

    import NEOPAX

    result = NEOPAX.run("my_case.toml", backend="theta_newton")

    final_state = result.final_state
    ts = result.time_grid
    ys = result.saved_states
    accepted = result.accepted_mask

Configuration-first API
^^^^^^^^^^^^^^^^^^^^^^^

If you want more explicit control, use the lower-level API:

.. code-block:: python

    import NEOPAX

    config = NEOPAX.load_config("my_case.toml")
    config = NEOPAX.prepare_config(
        config,
        mode="transport",
        n_radial=51,
        set_values=["turbulence.debug_heat_flux_scale=0.5"],
    )
    result = NEOPAX.run_config(config)

This path is useful when:

- the config is being edited in memory
- overrides are coming from another Python layer
- you want to keep the config object explicit

Path summary
^^^^^^^^^^^^

The direct Python API currently has three useful levels:

- ``NEOPAX.load_config(path)``
  - load TOML only
- ``NEOPAX.prepare_config(config_or_path, ...)``
  - apply common overrides without running
- ``NEOPAX.run(config_or_path, ...)``
  - convenience run wrapper returning ``RunResult``
- ``NEOPAX.run_config(config)``
  - execute an already-prepared config directly


Which API Should You Prefer?
----------------------------

For most users:

- use ``NEOPAX ...`` or ``python -m NEOPAX ...``

For programmatic workflows:

- use ``NEOPAX.run(...)``

For advanced scripting or future JAX-centric compositions:

- use ``NEOPAX.prepare_config(...)`` plus ``NEOPAX.run_config(...)``

This layering keeps the package usable both as a practical CLI tool and as a
clean Python library.


Extending NEOPAX
----------------

NEOPAX also supports custom flux and source model registration from Python.

For TOML-driven runs, custom model modules can also be loaded through an
``[extensions]`` section before model resolution.

See:

- :doc:`custom_models`
