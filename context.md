# Planned Efficiency & Best-Practice Refactors (Torax Comparison, 2026)

Based on a comparison with torax (https://github.com/google-deepmind/torax), the following areas are identified for further efficiency, modularity, and user/developer experience improvements:

1. **Solver Modularity and Multiple Backends**
  - Modularize the solver interface to support multiple time integrators and nonlinear solvers (e.g., Diffrax, Newton, jaxopt, predictor-corrector).
  - Enable easy benchmarking and optimal solver selection for different regimes.

2. **Boundary Condition (BC) Abstraction**
  - Ensure all BCs are implemented as modular, composable objects.
  - Allow user-supplied and runtime-registered BCs, not just hardcoded logic.

3. **Source/Transport Model Registration**
  - Refactor sources and transport models to be pluggable and composable at runtime (e.g., via a registry or composition pattern).

4. **Persistent JAX Compilation Cache**
  - Add support for JAX persistent compilation cache to reduce recompilation time for repeated runs or parameter scans.

5. **Error Checking and Debugging Flags**
  - Add runtime flags for enabling/disabling error checking, type assertions, and debugging, toggleable without code changes.

6. **Output/Logging/Progress Bar**
  - Add a progress bar, structured logging, and user-configurable output directories for simulation runs.

7. **Post-Simulation In-Memory State**
  - Support in-memory state retention after simulation for rapid reruns and parameter studies.

8. **Test Coverage and Assertion Checks**
  - Expand the test suite and add optional strict assertion/type checking for development and CI.

9. **Documentation and Tutorials**
  - Expand documentation and provide example configs/tutorials for new users, matching torax's standards.

**Action Plan:**
- Review and prioritize these items for staged implementation.
- Track progress in this document and update as features are completed.
# Refactor Status Update (March 31, 2026)

**Current Status:**
- All major routines in `_neoclassical.py` and `_species.py` have been refactored to remove any attribute access to temperature, density, or derived quantities from the Species class.
- All such routines now require explicit arguments for temperature, density, v_thermal, A1, A2, A3, charge, and other derived quantities, or receive them via the modular state dataclass.
- The FVM, source, and BC logic is fully modular and operates on the state dataclass, with robust ghost cell and boundary condition handling (torax-style).
- The codebase is now JAX-friendly, JIT-compatible, and supports diffrax integration with the state as a PyTree dataclass.
- All legacy logic that assumed persistent storage of derived quantities in Species/state has been removed in favor of pure function computation.
- All error checks pass for the refactored files, and the code is ready for further physics/model extensions.

# NEOPAX Refactor Context and Guidelines

## Refactor Plan Overview

- **Spatial Discretization:**
  - Refactor all transport equations (starting with the electric field $E_r$) to use finite volume method (FVM) routines, inspired by Torax.
  - Move all spatial derivatives and flux calculations to use modular utilities in `_fem.py`.
  - Ensure conservative updates and robust boundary condition handling.

- **Time Integration:**
  - Support both Diffrax and Torax-like solvers for time integration.
  - Keep time integration modular and independent from spatial discretization.

- **General Guidelines:**
  - All code should be:
    - Jitable (compatible with `jax.jit`)
    - JAX-friendly (using JAX arrays and functions)
    - Fully differentiable (for optimization and autodiff)
    - Modular (separate spatial, temporal, and physics logic)
    - Memory efficient (on both CPU and GPU)
    - Stable (robust to stiff problems, root transitions, and sharp gradients)
    - Fast (suitable for optimization of input geometries and output quantities like fusion power, etc.)

- **Extensibility:**
  - The framework should allow easy addition of new physics, boundary conditions, and solver options.

## Goals
- Achieve robust, high-performance, and research-friendly plasma transport simulations.
- Enable fast optimization and sensitivity analysis for fusion device design.

## 2026 Modular State & FVM Refactor Plan (Torax-style)

- **State as JAX dataclass:**
  - All evolving variables (density, temperature, Er, etc.) will be fields in a JAX dataclass (PyTree-compatible).
  - This enables efficient JIT, vmap, and type safety, and is fully compatible with diffrax.

- **vector_field and sources:**
  - Refactor to accept and return the state dataclass, not flat arrays.
  - All FVM, source, and BC logic will operate on the state object.

- **Modular FVM, sources, and BCs:**
  - Move FVM update, source, and BC logic into composable functions/classes.
  - Implement BCs as modular, callable objects that operate on the state.
  - Allow registration/composition of different source/BC models (as in torax’s CombinedTransportModel).

- **Model-level vectorization:**
  - Design updates to operate on the entire state object, enabling more efficient batching and JIT.

- **Immutable state:**
  - Avoid in-place updates; always return new state objects for better JAX compatibility and JIT performance.

- **Diffrax compatibility:**
  - The state dataclass will be used as the ODE state in diffrax, which supports PyTrees natively.

- **Equation selection:**
  - With this structure, it will be easy to select which equations to evolve: simply include/exclude fields in the state dataclass and update the vector_field logic accordingly.

- **Action plan:**
  1. Define the state dataclass.
  2. Refactor vector_field and sources to use it.
  3. Modularize FVM/source/BC logic.
  4. Compose models and BCs.
  5. Update diffrax integration.
  6. Test and validate.

- **Benefits:**
  - Efficiency, modularity, extensibility, and JAX/diffrax-friendliness, matching torax best practices.

## Additional Refactor Steps (2026)

- Review and update _neoclassical.py and _species.py:
  - Remove any logic that assumes temperature and density are stored in Species.
  - Update all functions to accept temperature and density as explicit arguments or via the new TransportState dataclass.
  - Ensure all neoclassical and physics routines are compatible with the new modular state structure.

- Update plan:
  7. Refactor _neoclassical.py and _species.py for new state structure.
