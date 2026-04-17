# NEOPAX Code Summary

## What this project is

NEOPAX is a Python/JAX research code for stellarator neoclassical and transport calculations. The repo is organized around three main workflows:

- `ambipolarity`: find radial ambipolar electric-field roots `Er(r)`
- `fluxes`: evaluate neoclassical/turbulent transport fluxes for a given state
- `transport`: evolve density, temperature, and `Er` in time

The package is built to be JAX-friendly: most core state/model objects are dataclasses or pytrees, and the code is clearly being refactored toward a more modular, Torax-style architecture.

## High-level execution flow

The main orchestrator is [NEOPAX/main.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/main.py). Its job is:

1. Read a TOML config.
2. Build the core runtime objects:
   - `Species`
   - energy grid
   - geometry from VMEC/BOOZ files
   - monoenergetic database
   - initial profiles / `TransportState`
3. Build transport/source/boundary/equation models from registries.
4. Dispatch by mode:
   - ambipolarity root-finding
   - direct flux computation
   - time integration of transport equations

There is also `python -m NEOPAX` support via [NEOPAX/__main__.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/__main__.py:1).

## Core data structures

- [NEOPAX/_state.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_state.py)
  - Defines `TransportState`, the main evolving state.
  - Holds `density`, `temperature`, and `Er`.
  - Includes `get_v_thermal()` helper.

- [NEOPAX/_species.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_species.py)
  - Defines `Species`.
  - Stores masses, charges, names, and helper properties like ion indices.
  - Contains collision/collisionality utilities and thermodynamic-force helpers.

- [NEOPAX/_database.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_database.py)
  - Defines `Monoenergetic`.
  - Reads MONKES-style monoenergetic transport data from HDF5.
  - Precomputes log-scaled transport coefficient arrays and electric-field coordinates.

## Geometry and grids

- [NEOPAX/_geometry_models.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_geometry_models.py)
  - Registry-based geometry model selection.
  - Main implementation is `VmecBoozer`.
  - Loads VMEC/BOOZ netCDF files and constructs radial grids plus geometry quantities like `Vprime`, `iota`, `B0`, curvature-related terms, etc.

- [NEOPAX/_energy_grid_models.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_energy_grid_models.py)
  - Registry-based energy grid builder.
  - Main model is `StandardLaguerreEnergyGrid`, using Laguerre quadrature and Sonine-moment weights.

- [NEOPAX/_profiles.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_profiles.py)
  - Builds initial profiles.
  - Supports analytical and prescribed profile models.
  - Returns a `ProfileSet` used to initialize `TransportState`.

## Physics and flux models

- [NEOPAX/_neoclassical.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_neoclassical.py)
  - Main neoclassical physics engine.
  - Builds `Lij` matrices and computes neoclassical particle/heat/parallel-flow fluxes.
  - Includes a momentum-correction path as well.

- [NEOPAX/_transport_flux_models.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_transport_flux_models.py)
  - Transport-flux model registry and composition layer.
  - Main models:
    - `MonkesDatabaseTransportModel`
    - `AnalyticalTurbulentTransportModel`
    - `ZeroTransportModel`
    - `CombinedTransportFluxModel`
  - This is the main abstraction that combines neoclassical, turbulent, and classical contributions.

- [NEOPAX/_turbulence.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_turbulence.py)
  - Analytical turbulent flux helper(s).

- [NEOPAX/_entropy_models.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_entropy_models.py)
  - Registry for entropy/proxy functions used by ambipolar root selection.

## Ambipolarity solver

- [NEOPAX/_ambipolarity.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_ambipolarity.py)
  - Implements several root-finding strategies for `Er`:
    - coarse/fine two-stage search
    - adaptive bracketing
    - multistart clustering utilities
  - For each radial location it builds:
    - `Gamma_func(Er)` from the transport flux model
    - an entropy proxy for root ranking
  - Also contains plotting and HDF5 output helpers for ambipolar-root scans.

The code is written to keep the inner numerical routines JAX-friendly where possible, with some plotting/output handling done outside the main differentiable path.

## Transport equations and numerics

- [NEOPAX/_transport_equations.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_transport_equations.py)
  - Defines registry-based equation objects:
    - `DensityEquation`
    - `TemperatureEquation`
    - `ElectricFieldEquation`
  - Builds equations from config and combines them through `ComposedEquationSystem`.
  - Includes quasi-neutrality enforcement and the `Er` diffusion/ambipolar source term logic.

- [NEOPAX/_fem.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_fem.py)
  - Finite-volume helpers like `conservative_update()` and face/cell conversions.

- [NEOPAX/_cell_variable.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_cell_variable.py)
  - Reconstruction helpers for face values/gradients.
  - Supports linear and WENO3-style reconstruction logic.

- [NEOPAX/_boundary_conditions.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_boundary_conditions.py)
  - Modular BC layer.
  - Supports Dirichlet, Neumann, and Robin conditions.
  - Has a newer `BoundaryConditionModel` abstraction used by the transport-equation builders.

## Solver layer

- [NEOPAX/_transport_solvers.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_transport_solvers.py)
  - Central solver registry and backend selection.
  - Supports several solver families:
    - Diffrax ODE solvers
    - custom `RADAUSolver`
    - predictor-corrector / Heun
    - Newton / Anderson / Broyden steady-state solvers
    - theta-method solvers
  - `build_time_solver()` chooses the backend from config.

This is one of the most active and complex parts of the repo. A lot of the architecture documentation is centered here.

## Sources

- [NEOPAX/_sources.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_sources.py)
  - Physics source-term formulas such as DT reaction, power exchange, and bremsstrahlung.

- [NEOPAX/_source_models.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_source_models.py)
  - Registry/composition layer for source models.
  - Builds density/temperature source combinations from config.

## Tests and examples

- `tests/`
  - Covers FVM reconstruction/discretization, BC handling, ambipolarity, flux calculations, and solver behavior.
  - `test_theta_solver_benchmarks.py` is especially important for solver validation and compares theta, Radau, and Diffrax/Kvaerno5 on stiff toy problems.
  - Some tests appear to target older APIs as well, so the suite reflects both current and legacy code paths.

- `examples/`
  - Contains runnable scripts for:
    - solving `Er`
    - computing fluxes
    - momentum-correction comparisons
    - config-driven runs from TOML

## Current architecture state

The repo is partly in transition between an older monolithic style and a newer modular style.

What looks modern/current:

- `TransportState`-based state handling
- registry-based model selection
- modular geometry/energy/source/flux/equation/solver builders
- JAX pytrees and explicit config-driven orchestration

What looks legacy or in-progress:

- duplicate/older files such as `*_old.py`, multiple `copy` files, and older tests
- `main.py` contains duplicated transport-branch logic and some refactor leftovers
- some modules still mix old assumptions with the newer state-based design

So the repo is usable, but it should be read as an actively refactored research code rather than a fully cleaned production package.

## Practical reading order

For understanding the current codebase quickly, the best order is:

1. [NEOPAX/main.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/main.py)
2. [NEOPAX/_state.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_state.py) and [NEOPAX/_species.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_species.py)
3. [NEOPAX/_geometry_models.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_geometry_models.py), [NEOPAX/_energy_grid_models.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_energy_grid_models.py), [NEOPAX/_profiles.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_profiles.py)
4. [NEOPAX/_transport_flux_models.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_transport_flux_models.py) and [NEOPAX/_neoclassical.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_neoclassical.py)
5. [NEOPAX/_transport_equations.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_transport_equations.py), [NEOPAX/_fem.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_fem.py), [NEOPAX/_cell_variable.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_cell_variable.py)
6. [NEOPAX/_transport_solvers.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_transport_solvers.py)
7. [NEOPAX/_ambipolarity.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_ambipolarity.py)

## Short takeaway

NEOPAX is a JAX-based stellarator transport code centered on modular construction of geometry, species/state, monoenergetic transport databases, flux models, transport equations, and solver backends. The main scientific core lives in the neoclassical flux calculation plus the transport-equation/solver stack, while the overall codebase is clearly undergoing an active refactor toward cleaner registry-based composition.
