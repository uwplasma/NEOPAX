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


Runtime NTX Scan From Python
----------------------------

For the ``ntx_scan_runtime`` neoclassical model, the direct Python path now
supports a more explicit split between:

- static VMEC/Boozer-derived setup
- array-valued scan inputs such as ``rho_scan``, ``nu_v_scan``, and
  ``er_tilde_scan``

This is useful when Python is orchestrating repeated evaluations and you want
to avoid rebuilding the static channel data each time.

Example:

.. code-block:: python

    import NEOPAX

    channels = NEOPAX.build_ntx_runtime_scan_channels(
        "examples/inputs/wout_QI_nfp2_newNT_opt_hires.nc",
        "examples/inputs/boozermn_wout_QI_nfp2_newNT_opt_hires.nc",
        [0.25, 0.5, 0.75],
    )

    model = NEOPAX.build_ntx_runtime_scan_transport_model(
        species=species,
        energy_grid=energy_grid,
        geometry=geometry,
        vmec_file="examples/inputs/wout_QI_nfp2_newNT_opt_hires.nc",
        boozer_file="examples/inputs/boozermn_wout_QI_nfp2_newNT_opt_hires.nc",
        ntx_scan_rho=[0.25, 0.5, 0.75],
        ntx_scan_nu_v=[1.0e-5, 1.0e-4, 1.0e-3],
        ntx_scan_er_tilde=[0.0, 1.0e-5, 3.0e-5, 1.0e-4],
        ntx_scan_channels=channels,
        prebuild_database=False,
    )

At that point:

- the static channels are already attached to the model
- the runtime monoenergetic database is still deferred
- evaluation can build the database later through the model path

For repeated scans in Python, the same model can also be reused with updated
scan inputs:

.. code-block:: python

    model_2 = model.with_scan_inputs(
        nu_v_scan=[2.0e-5, 2.0e-4, 2.0e-3],
        er_tilde_scan=[0.0, 2.0e-5, 6.0e-5, 2.0e-4],
    )

If the ``rho`` grid is unchanged, the preloaded static channels are retained.
If the ``rho`` grid changes, the model drops those cached channels so they can
be rebuilt consistently for the new scan grid.

The same general direct-Python idea now also appears in other built-in models:

- ``FluxesRFileTransportModel.with_q_scale(...)``
- ``AnalyticalTurbulentTransportModel.with_transport_coeffs(...)``
- ``PowerAnalyticalTurbulentTransportModel.with_transport_coeffs(...)``
- ``CombinedSourceModel.with_added_sources(...)``

These helpers are intended to keep small model updates at the model-object
level instead of forcing orchestration code to rebuild everything from scratch.

See also:

- ``examples/custom_models/ntx_runtime_scan_direct_api_example.py``


Extending NEOPAX
----------------

NEOPAX also supports custom flux and source model registration from Python.

For TOML-driven runs, custom model modules can also be loaded through an
``[extensions]`` section before model resolution.

See:

- :doc:`custom_models`
