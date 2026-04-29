Overview
========

NEOPAX is a JAX-native stellarator transport package built around modular flux,
source, geometry, and solver components. It is designed to support both
physics studies and numerical experimentation without forcing all workflows
through a single monolithic execution path.

At a high level, NEOPAX can be used as:

- a predictive 1D radial transport solver
- an ambipolarity root-finding tool for :math:`E_r`
- a flux-evaluation driver on fixed profiles
- a source-evaluation driver on fixed profiles

The active runtime entry points are selected through ``mode`` in the input
configuration:

- ``transport``
- ``ambipolarity``
- ``fluxes``
- ``sources``


Capabilities And Advantages
---------------------------

NEOPAX currently offers the following strengths.

- **JAX-native execution**
  - the main transport, ambipolarity, source, and flux pathways operate on
    JAX arrays and are compatible with JIT-oriented workflows
- **Modular physics composition**
  - neoclassical, turbulent, and classical transport contributions are composed
    through a shared flux-model interface
- **Multiple solver backends**
  - Diffrax backends, a custom Radau solver, and theta-based implicit solvers
    are all available through the same runtime orchestration layer
- **Boundary-aware finite-volume transport**
  - transport equations are assembled from cell-centered profiles with explicit
    face reconstruction and boundary-condition handling
- **Ambipolarity workflow integrated with transport**
  - the package supports both standalone ambipolarity runs and ambipolar
    initialization inside transport calculations
- **Registry-driven model selection**
  - flux models and source models are selected from registries rather than
    hard-coded branches scattered through the solver
- **Practical diagnostics and output modes**
  - NEOPAX supports plots, HDF5 outputs, debug summaries, and comparison
    against reference profiles


Equations Solved
----------------

NEOPAX currently solves a radial 1D transport system built from density,
pressure/temperature, and radial-electric-field evolution equations.

Density equation
^^^^^^^^^^^^^^^^

For each independently evolved species, the density equation is assembled in
conservative finite-volume form as

.. math::

   \partial_t n_s = \mathcal{D}_r\!\left(\Gamma_s\right) + S^{(n)}_s,

where:

- :math:`\Gamma_s` is the particle flux
- :math:`\mathcal{D}_r(\cdot)` is the radial finite-volume divergence operator
- :math:`S^{(n)}_s` is the assembled density-source contribution

The implementation supports:

- face-flux reconstruction from cell-centered fluxes
- direct model-supplied face fluxes
- masking of non-independent species in quasi-neutral configurations

Temperature / pressure equation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The thermal equation is evolved in pressure form, with thermal conduction,
convective transport, source terms, and an optional work term:

.. math::

   \partial_t p_s =
   \frac{2}{3}
   \left[
      \mathcal{D}_r\!\left(Q_s + T_s \Gamma_s^{\mathrm{neo}}
      + T_s \Gamma_s^{\mathrm{turb}}
      + T_s \Gamma_s^{\mathrm{class}}\right)
      + S^{(p)}_s
      + q_s \Gamma_s E_r
   \right].

In the code this is assembled from:

- conductive heat flux :math:`Q_s`
- optional neoclassical convective transport
- optional turbulent convective transport
- optional classical convective transport
- assembled pressure-source components
- optional work term :math:`q_s \Gamma_s E_r`

This design allows the user to switch individual transport contributions on and
off while keeping the transport discretization structure unchanged.

Radial electric field equation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :math:`E_r` equation is formulated as a relaxation-diffusion equation
driven by ambipolar charge balance:

.. math::

   \partial_t E_r =
   \tau_{E_r}
   \left[
      D_{E_r}\,\mathcal{D}_r\!\left(F_{E_r}\right)
      - \mathcal{A}
   \right],

where:

- :math:`F_{E_r}` is the diffusive electric-field flux
- :math:`D_{E_r}` is the configured electric-field diffusivity factor
- :math:`\mathcal{A}` is the ambipolar source term built from the charge-weighted
  particle fluxes

Supported modes include:

- local ambipolar source construction
- transport-centered ambipolar source construction
- multiple permitivity scalings, including NTSS-like midpoint options
- standard and floating-edge ambipolar boundary handling


Algorithms And Numerical Methods
--------------------------------

Spatial discretization
^^^^^^^^^^^^^^^^^^^^^^

NEOPAX uses a radial finite-volume discretization based on:

- cell-centered state profiles on ``r_grid``
- face-centered fluxes on ``r_grid_half``
- conservative divergence assembly using ``Vprime`` and ``Vprime_half``

Depending on the equation and reconstruction settings, NEOPAX can use:

- reconstructed face fluxes from cell-centered model outputs
- face-flux closures supplied directly by the transport model
- NTSS-like face state construction
- ghost-cell-aware boundary reconstruction

Time integration
^^^^^^^^^^^^^^^^

The active solver backends currently include:

- ``theta``
  - fixed-form implicit theta method
- ``theta_newton``
  - Newton-based implicit theta method with predictor support
- ``radau``
  - custom implicit Radau backend
- Diffrax backends
  - including ``diffrax_kvaerno5``, ``diffrax_tsit5``, and ``diffrax_dopri5``

The common runtime builder is ``build_time_solver(...)`` in
``NEOPAX/_transport_solvers.py``.

Ambipolarity algorithms
^^^^^^^^^^^^^^^^^^^^^^^

The ambipolarity subsystem supports:

- initialization of ambipolar best roots for transport startup
- standalone radial root solving in ``mode = "ambipolarity"``
- local particle-flux evaluators to reduce the cost of repeated
  :math:`E_r` evaluation
- multiple initializer and branch-tracking pathways inside
  ``NEOPAX/_ambipolarity.py``


Flux Models
-----------

Flux models are assembled through the transport-flux registry in
``NEOPAX/_transport_flux_models.py``. The main runtime pattern is:

- build one neoclassical model
- build one turbulent model
- optionally build one classical model
- combine them through ``CombinedTransportFluxModel``

This gives a unified interface returning species-resolved particle and heat
fluxes, with optional face-flux evaluation support.

Built-in transport flux models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The currently registered transport flux models include:

- ``monkes_database``
  - database-driven neoclassical model
- ``monkes_database_with_momentum``
  - database-driven neoclassical model with momentum-correction pathway
- ``turbulent_analytical``
  - analytical turbulent transport model
- ``turbulent_power_analytical``
  - analytical turbulent transport model normalized to total power
- ``ntss_power_over_n``
  - power-normalized analytical turbulent model aligned with NTSS-style usage
- ``fluxes_r_file``
  - fluxes loaded from a radial profile file
- ``none``
  - zero-flux model

Important architectural points
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- neoclassical, turbulent, and classical contributions are kept separate until
  composition
- face-flux evaluation is supported explicitly for transport assembly
- turbulent particle transport can be disabled independently of turbulent heat
  transport in the combined model


Source Models
-------------

Source models are assembled through the registry in
``NEOPAX/_source_models.py``. Density and temperature equations consume source
models through a shared component-assembly pathway.

Built-in source models
^^^^^^^^^^^^^^^^^^^^^^

The currently registered source models include:

- ``fusion_power_fraction_electrons``
- ``dt_reaction``
- ``power_exchange``
- ``bremsstrahlung_radiation``
- ``analytic``
- ``example_state``

The source framework also supports ``CombinedSourceModel``, which merges
multiple named source contributions into one model output.

Current source handling features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- density-source assembly by named component
- pressure-source assembly by named component
- species-aware component routing
- active-temperature masking for power exchange
- direct use in ``mode = "sources"`` for source-only evaluation workflows


Benchmarks And Current Validation
---------------------------------

The current validation picture has two complementary parts:

- **automated regression tests**
  - the active test suite covers boundary conditions, flux-file transport
    models, finite-volume discretization helpers, geometry volume integration,
    main helper utilities, orchestration and solver helpers, source models,
    state/cell-variable helpers, transport equation helpers, and transport
    state handling
- **physics and solver benchmarking**
  - the repository contains benchmark-oriented planning and solver-performance
    notes in ``PLAN.md``
  - current benchmark work especially tracks ambipolarity behavior and solver
    backend performance

Available benchmark figure
^^^^^^^^^^^^^^^^^^^^^^^^^^

The repository currently includes the ambipolar root benchmark figure below.

.. figure:: benchmark/figures/Er_ambipolar_roots.png
   :width: 85%
   :align: center
   :alt: Ambipolar radial electric field roots benchmark

   Current ambipolar-root benchmark figure stored in the repository. This is
   the main benchmark plot currently available in ``docs/benchmark/figures``.


Current Documentation Gaps
--------------------------

This overview is intentionally the first consolidation page, not the final
manual. The most useful next documentation expansions would be:

- a dedicated input-file reference
- a page for transport solver backends and recommended use cases
- a page for flux-model configuration examples
- a page for source-model configuration examples
- a benchmark page collecting more figures as they are added
