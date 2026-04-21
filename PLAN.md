# NEOPAX Memory and Speed Rewrite Plan

## Current next upgrade: face-evaluated closure fluxes for the pressure equation

### Problem to solve

The pressure equation currently uses closure-provided heat fluxes `Q` on cell centers and then interpolates those centered fluxes to faces before taking the conservative divergence. This is especially fragile near the sharp `Er` transition and is the leading suspected cause of the temperature oscillations seen in both `RADAU` and `Kvaerno5`.

We explicitly do **not** want to replace the neoclassical/turbulent closures with an inferred diffusive surrogate. The fix must preserve the actual closure models and instead evaluate those models on face states from the beginning.

### Upgrade plan

1. Add a shared face-state builder in [NEOPAX/_transport_flux_models.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_transport_flux_models.py)
   - reconstruct `density`, `pressure/temperature`, and `Er` on `r_grid_half`
   - keep boundary handling consistent with the current cell-centered code
   - keep everything JAX/JIT/trace friendly

2. Add an optional model-side face-flux API
   - extend transport models with a method like `evaluate_face_fluxes(...)`
   - keep the current `__call__(state)` centered path unchanged
   - make the new API return physically modeled `Gamma_face`, `Q_face`, `Upar_face`

3. Implement the turbulent face-flux path first
   - this is the simplest and lowest-risk model to move to faces
   - use reconstructed face states and face gradients to get exact model-consistent `Gamma_face` and `Q_face`

4. Implement the neoclassical face-flux path next
   - evaluate the actual neoclassical closure on face states, not an inferred effective `chi`
   - this likely requires allowing `Lij`/database evaluation at arbitrary radial coordinates on the face grid
   - compute `A1`, `A2`, `A3` on faces and form `Gamma_face`, `Q_face` directly

5. Keep legacy and new paths side by side during validation
   - legacy: centered closure flux + face interpolation
   - new: closure evaluated directly on faces
   - expose the switch in config only after both model paths exist

6. Only then switch the pressure equation to consume face fluxes directly
   - `TemperatureEquation` should use closure-provided `Q_face` when available
   - keep the legacy fallback for comparison and regression testing

7. Validate in this order
   - fixed-`Er`, pressure-only case
   - compare temperature smoothness and `q_divergence`
   - then rerun the coupled pressure + `Er` case

### Implementation status

- [x] shared face-state builder
- [x] model-side face-flux API
- [x] turbulent face fluxes
- [x] neoclassical face fluxes
- [x] pressure equation direct face-flux consumption
- [ ] validation on pressure-only fixed-`Er` case

## Current next upgrade: quasi-neutral density equation construction

### Problem to solve

The density equation currently has two inconsistencies:

1. The electron density equation is effectively disabled by forcing electron `density_rhs = 0`, while ion densities may evolve.
2. Quasi-neutrality is enforced mainly as an output cleanup, instead of being built into the density ODE seen by the solver at every RHS evaluation.

That means the internal solver state can drift away from quasi-neutrality during time stepping, even though the final/saved outputs are projected back afterward. This is not the correct construction if electrons are meant to remain part of the state while still satisfying the quasi-neutrality constraint dynamically.

The next refinement is to move from the current "dependent electron RHS inside the
ODE system" construction to an NTSS-style algebraic update:

- keep electron density in the transport state/output
- evolve only the independent ion/impurity density rows
- reconstruct electron density algebraically from quasi-neutrality for the
  working state and for accepted/output states

\[
n_e = -\frac{1}{Z_e}\sum_{i \ne e} Z_i n_i
\]

This keeps the full public state shape unchanged, but removes the dependent
electron density row from the coupled implicit solve. It should reduce solver
coupling and be friendlier to JAX stiff integrators on the He-coupled cases.

### Upgrade plan

1. Enforce quasi-neutrality on the working state before every RHS evaluation
   - in [NEOPAX/_transport_equations.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_transport_equations.py)
   - build a quasi-neutral working state before evaluating shared fluxes and source terms
   - this ensures all closures see a physically consistent electron density

2. Remove the electron density row from the solved density subsystem
   - compute independent ion/impurity density RHSs only
   - keep the electron density RHS at zero inside the returned ODE state
   - do not feed a dependent electron density equation into the implicit solver

3. Reconstruct a quasi-neutral working state before every RHS evaluation
   - the closures and equation assembly should always see a quasi-neutral `n_e`
   - `n_e` must be rebuilt algebraically from the current ion/impurity state

4. Preserve toggle semantics for partially evolved species
   - fixed species must keep `rhs = 0`
   - if only some ions/impurities evolve, the electron density used in the
     working state must still be reconstructed from the currently evolved
     charged species
   - examples:
     - only `He` evolves, `D/T` fixed
     - only one fuel ion evolves
     - all ions evolve

5. Keep the density equation JAX/JIT/trace friendly
   - no Python-side constraint loop in the hot path
   - keep fixed shapes and array-only algebra
   - no dynamic indexing or host-side solves

6. Avoid redundant direct electron density transport work
   - treat electrons as a dependent density species inside the density equation itself
   - compute density flux divergence and density sources only for independent species
   - do not compute or solve a direct electron density transport/source row
   - keep full state shapes fixed to stay JAX/JIT friendly

7. Keep output-side quasi-neutrality reconstruction
   - accepted/saved/final states should rebuild `n_e` from the ion/impurity
     state before plotting/output and before the next user-visible state
   - this remains the mechanism that updates the stored electron profile between
     timesteps, without solving an electron density ODE row

8. Validate in this order
   - He-only density evolution with fixed `D/T`
   - confirm electron density updates according to quasi-neutrality after
     accepted steps
   - compare no-oscillation behavior with the new `Gamma_face` density flux path
   - then test mixed-species density toggles and coupled transport

### Implementation status

- [x] density face-flux path (`Gamma_face`) added
- [x] density toggle masking added
- [x] enforce quasi-neutral working state during RHS evaluation
- [x] skip redundant direct electron density transport/source work
- [x] remove dependent electron density RHS from the solved ODE system
- [x] keep electron density updated algebraically on accepted/output states
- [ ] validate He-only density evolution

## Goal

Reduce compilation memory, runtime memory, and total wall-clock time for both `mode = "transport"` and `mode = "ambipolarity"` while preserving these constraints:

- modular architecture
- JAX-friendly data flow
- `jax.jit` compatibility
- differentiability from final solution back to initial state through time

This plan is focused on removing large traced closures, avoiding repeated construction inside traced paths, reducing dense Jacobian cost where possible, and controlling saved state/output volume.

## Main problems observed in the current code

### 1. Large compile closures and duplicate setup

- [NEOPAX/main.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/main.py) contains duplicated transport logic and repeated config/model assembly.
- `transport` currently builds large runtime dictionaries and passes them into solver calls.
- `build_equation_system_from_config()` rebuilds geometry, database, and flux models from config instead of consuming already-built runtime objects.
- This increases XLA trace size, compile memory, and recompilation risk.

### 2. Expensive implicit-solver linearization

- [NEOPAX/_transport_solvers.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_transport_solvers.py) uses dense `jax.jacfwd(...)` in several places.
- `RADAUSolver` forms dense Jacobians inside each implicit stage solve.
- `ThetaNewtonSolver` also relies on dense Jacobian formation for nonlinear solves.
- Dense Jacobians are often the biggest source of compile-time and runtime memory growth.

### 3. Ambipolarity repeatedly evaluates expensive flux closures

- [NEOPAX/_ambipolarity.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_ambipolarity.py) builds scalar root-finding closures around the full transport flux model.
- Each root search repeatedly calls `Gamma_func` and `grad(Gamma_func)`, which is expensive because each evaluation touches the full neoclassical flux stack.
- The radial solver currently uses `vmap` across radii or block loops, but the scalar closure itself still carries too much work.

### 4. Unnecessary tracing/output work

- [NEOPAX/_transport_equations.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_transport_equations.py) contains `jax.debug.print(...)` inside `ComposedEquationSystem.__call__`.
- [NEOPAX/_transport_solvers.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_transport_solvers.py) has extra debug `print(...)` calls in solver paths.
- Several solver paths save full trajectories by default or keep large intermediate histories.
- These increase compile size, host-device traffic, and memory pressure.

### 5. Repeated interpolation/model work inside tight loops

- Geometry/database/interpolation helpers are used deep inside neoclassical evaluation.
- Some functions create interpolator objects or re-derive quantities in paths that should ideally be precomputed once.
- That hurts both compile time and runtime.

## Optimization principles

The rewrite should follow these rules:

1. Build once, run many.
   All geometry, grids, databases, boundary models, source models, flux models, and equation objects should be built exactly once before entering a jitted solve.

2. Keep traced arguments minimal and typed.
   Replace config dicts and general runtime dicts with small dataclasses/pytrees containing only the arrays/scalars actually needed during the solve.

3. Separate setup from numerics.
   File I/O, config parsing, plotting, HDF5 writing, and object assembly must stay outside the jitted numerical core.

4. Prefer matrix-free or structured linearization over dense full Jacobians.
   Dense `jacfwd` should be a fallback, not the default path.

5. Save less by default.
   Support final-state-only execution and sparse output schedules to reduce memory.

6. Preserve differentiability.
   Any optimization must keep a differentiable path from final state to initial state for the intended solver modes.

## Rewrite plan

## Phase 1: Unify and shrink the transport/ambipolarity runtime interface

### Objectives

- remove duplicate orchestration logic
- stop rebuilding geometry/database/models in equation builders
- shrink compile closures

### Changes

1. Refactor [NEOPAX/main.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/main.py) into a single setup pipeline:
   - parse config
   - build `Species`
   - build geometry
   - build energy grid
   - load database
   - build initial `TransportState`
   - build source/flux/boundary/equation models once
   - dispatch to `transport` or `ambipolarity`

2. Introduce small runtime dataclasses such as:
   - `StaticRuntime`
   - `TransportModels`
   - `TransportNumerics`

3. Change [NEOPAX/_transport_equations.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_transport_equations.py):
   - `build_equation_system_from_config()` should stop reading files or reconstructing geometry/database/models
   - it should accept already-built model objects and numerics

4. Change [NEOPAX/_ambipolarity.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_ambipolarity.py):
   - `solve_ambipolarity_roots_from_config()` should consume the same prebuilt runtime/model objects
   - no second round of model construction inside the ambipolarity path

### Expected gains

- smaller XLA programs
- lower compile-time memory
- less recompilation from config-driven branching
- cleaner modular structure

## Phase 2: Remove tracing noise and unnecessary saved state

### Objectives

- reduce compile graph size
- lower runtime memory
- avoid host callbacks and debug overhead

### Changes

1. Remove `jax.debug.print(...)` from jitted equation evaluation in [NEOPAX/_transport_equations.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_transport_equations.py).

2. Remove solver debug prints from [NEOPAX/_transport_solvers.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_transport_solvers.py).

3. Add explicit output policies to solvers:
   - `save_mode = "final_only"` default
   - optional sparse checkpoints
   - optional full trajectory only when requested

4. Ensure ambipolarity stores only:
   - requested root summary
   - optional plot/HDF5 data outside the compiled path

### Expected gains

- immediate compile-size reduction
- lower device memory pressure
- faster steady-state and time integration runs

## Phase 3: Make the transport vector field lean and cache-friendly

### Objectives

- reduce runtime cost per RHS evaluation
- reduce memory traffic
- keep the vector field purely numerical

### Changes

1. Build equation objects with precomputed constant arrays:
   - `dr_cells`
   - `Vprime`
   - `Vprime_half`
   - BC face constraints
   - any static masks or species index arrays

2. Precompute and store any reusable interpolation coordinates from geometry/database rather than rebuilding helper objects in inner loops.

3. Revisit [NEOPAX/_transport_flux_models.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_transport_flux_models.py):
   - ensure model `__call__` only depends on state plus compact prebuilt static data
   - remove any accidental argument mismatches or hidden unused paths

4. Use a stable state layout:
   - fixed dtypes
   - fixed shapes
   - no config-dependent optional branches inside the compiled RHS

### Expected gains

- faster RHS evaluation
- fewer recompiles
- lower memory bandwidth cost during long integrations

## Phase 4: Reduce implicit solver memory cost

### Objectives

- address the likely biggest compilation killer: dense Jacobian formation

### Changes

1. In [NEOPAX/_transport_solvers.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_transport_solvers.py), introduce solver linearization modes:
   - `dense_jacfwd` as fallback/reference
   - matrix-free JVP/VJP-based Newton-Krylov path
   - optional block/diagonal approximation path for specific equations such as `Er`

2. For `ThetaNewtonSolver`:
   - prioritize matrix-free Newton updates using `jax.jvp` / `jax.linearize`
   - use iterative linear solves (for example GMRES) instead of assembling dense full Jacobians

3. For `RADAUSolver`:
   - avoid stagewise dense Jacobian assembly where possible
   - reuse linearization information across stages/iterations when valid
   - support lower-memory nonlinear solve mode for large transport states

4. Keep a differentiable solver mode:
   - smooth iteration logic
   - no hard host-side branching in the main solve path
   - retain a reference differentiable backend for validation

### Expected gains

- major compile-memory reduction
- major runtime-memory reduction
- better scaling to larger radial grids and more species

## Phase 5: Specialize the ambipolarity path for radial root-finding

### Objectives

- make ambipolarity cheaper than transport, not a thin wrapper around the full transport stack

### Changes

1. Split the ambipolarity flux evaluation into a dedicated radial kernel:
   - input: one radius index or a small block of radii
   - state dependence only through local quantities needed for `Gamma(Er)`
   - avoid carrying the full transport-equation machinery

2. Rework [NEOPAX/_ambipolarity.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_ambipolarity.py) so `Gamma_func` and entropy functions are built from a compact local evaluator instead of a full transport model closure.

3. Cache or precompute radial quantities that do not depend on the trial `Er`.

4. Keep block processing for memory control, but make block size a first-class optimization parameter.

5. Replace repeated `grad(Gamma_func)` calls with a more efficient derivative path where possible:
   - `jax.grad` on a compact local scalar function
   - or derivative-free safeguarded updates if gradient cost dominates and differentiability of the final selected root is still acceptable for the target workflow

### Expected gains

- much lower compile memory in `ambipolarity`
- less repeated flux work
- faster radial root scans

## Phase 6: Optimize neoclassical evaluation without breaking modularity

### Objectives

- speed up the real physics hotspot

### Changes

1. Audit [NEOPAX/_neoclassical.py](/abs/path/d:/PostDocsProxima/Github_5/NEOPAX/NEOPAX/_neoclassical.py) for:
   - repeated interpolation object creation
   - repeated shape conversions
   - repeated species-wise constants that can be precomputed

2. Precompute species- and geometry-derived constants once in setup dataclasses.

3. Where safe, fuse small helper calls to reduce trace fragmentation and dispatch overhead.

4. Add optional checkpointing/rematerialization only where it lowers peak memory more than it hurts runtime.

### Expected gains

- faster flux evaluation for both transport and ambipolarity
- better scaling with radial resolution

## Phase 7: Validation and profiling

### Objectives

- make sure the rewrite improves memory and speed without changing physics or breaking gradients

### Validation tasks

1. Baseline before changes:
   - compile time
   - peak memory
   - runtime
   - output equivalence

2. Compare before/after on:
   - `mode = "transport"`
   - `mode = "ambipolarity"`
   - small and moderate radial grids

3. Verify differentiability:
   - `jax.grad` or `jax.jvp` from final transport state to initial state
   - solver-specific regression tests for differentiable mode

4. Keep or extend tests around:
   - solver accuracy
   - FVM consistency
   - BC behavior
   - ambipolar root consistency

## Prioritized implementation order

The highest-value order is:

1. unify setup and remove duplicate runtime/model construction
2. remove debug printing and reduce saved trajectories
3. shrink transport/ambipolar runtime objects to compact dataclasses
4. specialize ambipolarity around a local radial flux evaluator
5. replace dense implicit-solver Jacobians with matrix-free linearization
6. optimize neoclassical inner loops
7. benchmark and validate gradients

## Deliverables

- refactored orchestration with one build path for shared runtime objects
- compact runtime/model dataclasses replacing large dict-based traced closures
- lower-memory transport solver modes
- lower-memory ambipolarity root-finding path
- updated tests/benchmarks for memory, runtime, and differentiability
- this rewritten `PLAN.md`

## Success criteria

The rewrite is successful if:

- both `transport` and `ambipolarity` compile without being killed on the target cases
- peak compile memory is substantially lower than the current code
- runtime memory is lower because full trajectories and dense Jacobians are no longer the default
- wall-clock time improves measurably
- the code remains modular, JAX-friendly, jittable, and differentiable from final state back to initial state

## Session Status

### Implemented so far

- `main.py` was refactored into a shared runtime-build path with `RuntimeContext` / `Models`.
- `transport` and `ambipolarity` now consume prebuilt runtime objects instead of rebuilding geometry/database/models repeatedly.
- solver output was reduced so final-state-only execution does not keep full trajectories by default.
- transport shared-flux reuse is only enabled for multi-equation solves.
- ambipolarity now supports an optional modular local particle-flux fast path implemented by flux models, with a generic fallback.
- `RADAUSolver` was substantially reworked:
  - matrix-free GMRES path added
  - adaptive control changed from step-doubling to an embedded low-order estimator (`radau_error_estimator = "embedded2"`)
  - simplified Newton option added (`radau_newton_strategy = "simplified"`) so the stage linearization is reused within a timestep
  - Radau constants now follow the state dtype instead of forcing `float64`
- debug hooks were added for diagnosis:
  - `debug_print_er = true` prints accepted-step `Er` values from the custom Radau solver
  - `debug_stage_markers = true` prints plain Python markers before and after `solver.solve(...)`
  - `debug_disable_jit = true` forces eager execution for diagnosis

### Current observed state

- the example `examples/Solve_Er_General/diffusion_Er.toml` is an `Er`-only transport case
- the process is still sometimes killed
- when it is killed, the user reports:
  - no `jax.debug.print` output
  - no visible GPU activity
  - longer wall time before failure
- this strongly suggests the failure is still happening during JAX tracing / XLA compilation, before the first executed accepted timestep

### Latest diagnostic evidence

- running
  - `python -m NEOPAX examples/Solve_Er_General/diffusion_Er.toml`
  with
  - `debug_stage_markers = true`
  - `debug_disable_jit = false`
  prints:

```text
[NEOPAX] transport setup complete: backend=radau n_equations=1 state_size=357
[NEOPAX] entering solver.solve(...)
```

- this confirms the failure happens after setup and after entering the solver path
- there are still no accepted-step `Er` debug prints before the process is killed in normal JIT mode
- this is strong evidence that the custom Radau solver is being killed during tracing / XLA compilation, or at the very start of compiled execution before the first accepted step

- running the same case with:
  - `debug_disable_jit = true`
  prints:

```text
[NEOPAX] transport setup complete: backend=radau n_equations=1 state_size=357
[NEOPAX] entering solver.solve(...)
[NEOPAX] debug_disable_jit=true, forcing eager execution for diagnosis
```

- this means the code does get into `solver.solve(...)` in eager mode and does not immediately die in the same way
- the likely interpretation is:
  - normal mode failure is dominated by compile-time memory / compile-time cost
  - eager mode is slow, and may still be struggling on the first implicit step, but it is not showing the same immediate compile-time death pattern

### Current example solver block

The example TOML currently includes these Radau-related options:

```toml
[transport_solver]
integrator = "radau"
t0 = 0.0
t_final = 20.0
dt = 1.0e-2
rtol = 1.0e-7
atol = 1.0e-7
max_step = 1.0
min_step = 1.0e-14
nonlinear_solver_tol = 1.0e-8
nonlinear_solver_maxiter = 50
radau_linear_solver = "gmres"
radau_gmres_tol = 1.0e-8
radau_gmres_maxiter = 200
radau_error_estimator = "embedded2"
radau_newton_strategy = "simplified"
debug_print_er = true
debug_stage_markers = true
debug_disable_jit = false
save_n = 10
```

### Most likely diagnosis

- the old worst offender, Radau step-doubling, was removed and did help
- however, full custom adaptive implicit Radau is still compiling as a very large traced program
- for this case, the remaining compile-memory hotspot is likely still the custom Radau Newton/GMRES stage solve and adaptive control graph, not the `Er` equation assembly itself

### Recommended next steps for the next session

1. Run once with:
   - `debug_stage_markers = true`
   - `debug_disable_jit = false`
   to confirm whether the last visible message is `entering solver.solve(...)`.

2. Run once with:
   - `debug_disable_jit = true`
   to check whether the solver can execute in eager mode and reach accepted-step `Er` prints.

3. Since this was now observed, treat the problem as compile-memory dominated unless new evidence contradicts it.

4. In eager mode, add more detailed temporary diagnostics if needed:
   - accepted vs rejected step prints
   - current `dt`
   - Newton residual norm
   - whether GMRES hits its iteration cap
   This will help separate “slow but working” from “numerically struggling.”

5. The next major implementation should then be a lighter stiff backend, preferably:
   - Rosenbrock / Rosenbrock-W, or
   - SDIRK

6. Keep the current Radau path as:
   - a reference solver
   - an optional high-accuracy backend
   but do not rely on it as the primary solver for the memory-constrained target cases.

### Recommendation to carry forward

If the next session confirms compile-time death before execution, the best use of effort is likely:

1. add a Rosenbrock-type backend with adaptive timestep and matrix-free linear solves
2. keep it JAX-friendly, jittable, and differentiable
3. use that backend as the main low-memory stiff solver for transport
4. leave custom Radau available for validation / comparison rather than as the default workhorse

## Current Transport/Er Status

### Stable direction reached

- The `Er` transport path is now behaving much better with:
  - `integrator = "diffrax_kvaerno5"`
  - `Er_source_mode = "ambipolar_local"`
  - `DEr = 0.0` for pure ambipolar relaxation tests
- The centered ambipolar source was producing an odd-even / checkerboard instability in the pure-source case.
- The local ambipolar source removes that face-to-cell averaging step and gives a much more physical relaxation toward the expected branch.
- `ambipolar_local` was therefore made the default source mode for the `Er` equation.

### Important implementation notes

- `TransportState` is currently treated as normalized:
  - density in `1e20 m^-3`
  - temperature in `keV`
  - `Er` in `kV/m`
- Flux-producing physics is converted back to physical units internally where needed.
- A critical bug was fixed in the current repo where normalized state values were being converted to physical units twice in the neoclassical path; that double conversion was a major cause of the later centered-path blow-up.
- Another critical fix was applied in the neoclassical flux assembly so final `Gamma`, `Q`, and `Upar` are returned in physical units instead of staying scaled by normalized state factors.
- For `DEr = 0`, the `Er` equation now bypasses diffusion entirely instead of computing diffusion and multiplying by zero. This avoids `0 * NaN` contamination and makes the pure ambipolar test mathematically faithful.

### Initialization improvement already added

- Transport can now optionally initialize `Er` from the ambipolar min-entropy branch before time integration.
- This is selected from TOML via:

```toml
[profiles]
er_initialization_mode = "ambipolar_min_entropy"
```

- The initializer computes the ambipolar `best_roots` profile using the same root-finder / entropy-selection machinery as `mode = "ambipolarity"` and replaces the initial `state.Er` with that branch.

## Next Ambipolarity Upgrade Path

### Goal

Make the ambipolarity root finder significantly faster and more memory efficient while keeping it:

- differentiable
- JAX-jittable
- JAX-friendly
- low-closure / low-memory
- suitable for reuse as an `Er` initialization tool for transport

### Recommended direction

Upgrade the current `two_stage` method rather than replacing it immediately.

Reason:

- this is a scalar root problem in `Er` at each radius
- the main cost is repeated expensive `Gamma(Er)` evaluations
- a more structured scalar solver should beat a heavier generic nonlinear method
- continuity in radius can be exploited strongly

### Proposed upgraded method

Implement a continuation-based ambipolar root finder, keeping the current method as a fallback/reference.

Working name:

- `two_stage_continuation`

Core idea:

1. Solve the first radius with the current robust search/bracketing logic.
2. For each next radius, use the selected root from the previous radius as the first guess or bracket center.
3. Only widen the search when the local continuation guess fails.
4. Only evaluate entropy on the small set of candidate roots that survive the local stage.

### Why this should help

- far fewer global trial points per radius
- much smaller repeated closure work
- better branch tracking across radius
- lower memory than broad multistart everywhere
- faster initialization for transport when using `ambipolar_min_entropy`

### Specific improvements to implement

1. Radius continuation
- use `best_root[i-1]` as the first guess for `i`
- optionally use a narrow bracket around that value

2. Adaptive bracketing
- start with a small local search window
- widen only when sign changes / convergence checks fail

3. Cheaper entropy selection
- do not evaluate entropy on a large cloud of guesses
- evaluate entropy only on the final small candidate set

4. Blocked radial execution
- keep block processing as a first-class option for memory control
- avoid vmapping all radii at once if compile/runtime memory grows too much

5. Compact local evaluator
- keep using a compact local particle-flux evaluator
- avoid carrying unnecessary transport-equation machinery into the scalar root closure

### Constraints to preserve

The upgraded method must remain:

- differentiable through the chosen numerical path where intended
- `jax.jit` compatible
- free of Python-side per-iteration logic inside the core jitted numerical kernel where practical
- careful about traced closure size
- as memory-light as possible for repeated radial scans

### What not to do first

- do not jump immediately to a heavier generic root solver
- do not reintroduce large multistart scans at every radius if continuation can avoid them
- do not make the local `Gamma(Er)` closure depend on the full transport machinery more than necessary

### Recommended implementation order for the next session

1. Keep current `two_stage` as reference/fallback.
2. Add a continuation-based ambipolar solver variant.
3. Expose it by TOML as a new ambipolarity method option.
4. Benchmark:
  - compile memory
  - runtime
  - branch consistency
  - agreement with current `two_stage`
5. If it is clearly better, make it the preferred method for:
  - `mode = "ambipolarity"`
  - `er_initialization_mode = "ambipolar_min_entropy"`

### Success criterion for this upgrade

The ambipolar upgrade is successful if:

- it produces the same selected branch as current `two_stage` on the reference case
- it uses less compile/runtime memory
- it is faster enough to be practical as a transport initializer
- it stays differentiable, jittable, and JAX-friendly
