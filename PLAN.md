## NEOPAX Solver Efficiency Refactor Plan

### Current Status Snapshot
- Completed:
  - Diffrax flat-state RHS refactor in `_transport_solvers.py`
  - packed-array quasi-neutral / fixed-temperature projection helpers in `_transport_solvers.py`
  - cached equation-object dispatch in `ComposedEquationSystem`
- Benchmarked on:
  - `examples/Solve_Er_General/transport_pressure_Er_debug_kvaerno5_temp_Er.toml`
- Best observed improvement so far:
  - Kvaerno total solve time dropped from roughly `622 s` to roughly `351-359 s`
  - accepted/rejected/total steps dropped from roughly `928/288/1216` to `315/78/393`
- Recent conclusion:
  - the first solver-wrapper refactor gave the major win
  - later projection/composed-system cleanups were structurally good, but did not produce another measurable timing improvement for the benchmark case
  - the main remaining cost is now likely inside the traced Diffrax implicit stage work and/or the shared flux-model evaluation, not the higher-level orchestration layer

### Goals
- Reduce `diffrax_kvaerno5` JIT compile time without changing the transport physics, boundary-condition semantics, solver tolerances, or active-equation configuration.
- Reduce runtime overhead for `diffrax_kvaerno5`, `radau`, and `rosenbrock` by making the RHS and state transforms lighter.
- Preserve the current modular equation architecture as the source of truth, while adding an optimized execution path for the solver hot loop.

### Physics / Math Invariants To Preserve
- Keep the current quasi-neutral treatment:
  - evolve only independent ion/impurity density rows
  - reconstruct `n_e` algebraically from quasi-neutrality
  - do not reintroduce an independently evolved electron density RHS
- Keep the fixed-temperature species semantics:
  - if `toggle_temperature[a] = false`, use `P_a = n_a * T_fixed_a` in the working state and unpacked/output state
  - if `toggle_temperature[a] = true`, keep the normal coupled `T_a = P_a / n_a`
- Keep the corrected NTSS-style Robin / `DECAYLEN` meaning:
  - boundary `value` only matters for Dirichlet
  - Robin/`DECAYLEN` uses the current boundary state with `du/dr = +/- u/L`
- Keep the existing equation objects (`DensityEquation`, `TemperatureEquation`, `ErEquation`, `ComposedEquationSystem`) as the physics reference implementation.

### Phase 1: Profile And Minimize The Diffrax Hot Path
- Status: mostly completed
- Refactor `DiffraxSolver.solve(...)` to use the same flat-state machinery already used by the custom solvers:
  - `_make_solver_state_transform(...)`
  - `_flat_rhs_factory(...)`
- Remove repeated full dataclass unpack/repack inside the traced Diffrax RHS where possible.
- Avoid repeated calls to:
  - `enforce_quasi_neutrality(...)`
  - `project_fixed_temperature_species(...)`
  inside every traced stage via dataclass reconstruction.
- Target result:
  - smaller `jit_diffeqsolve` graph
  - lower compile time
  - lower per-step RHS overhead

Completed in this phase:
- `DiffraxSolver.solve(...)` now runs on flat packed arrays
- Diffrax no longer rebuilds full `TransportState` objects inside every traced stage
- Diffrax output reconstruction still preserves the same public state semantics after the solve

Observed result:
- major timing improvement for the Kvaerno `temperature + Er` benchmark

### Phase 2: Add Array-Level Projection Helpers
- Status: completed, but no additional measurable benchmark win yet
- Implement packed/flat-array helpers for:
  - quasi-neutral electron reconstruction
  - fixed-temperature pressure projection
- Use these helpers in solver hot loops before reconstructing full `TransportState` objects.
- Keep the dataclass-based projection helpers for readability/debug/reference paths.
- Ensure the array-level and dataclass-level implementations stay mathematically identical.

Completed in this phase:
- packed-array projection helper added in `_transport_solvers.py`
- Diffrax, Radau, and Rosenbrock now all use the packed projection path

Observed result:
- cleaner architecture
- no meaningful extra timing improvement on the current Kvaerno benchmark beyond Phase 1

### Phase 3: Specialize The Composed Vector Field For Active Equation Sets
- Status: partially completed
- Add an optimized execution adapter for common active equation combinations, starting with:
  - `temperature + Er`
  - `density + temperature + Er`
- Avoid the generic hot-loop costs of:
  - looping over `self.equations`
  - building `eq_outputs` dictionaries
  - filling missing fields with zeros
  - generic postprocessing branches when the active set is already known
- Keep `ComposedEquationSystem` as the canonical general implementation.
- Use the specialized path only as an execution optimization layer.

Completed in this phase:
- cached direct equation dispatch added to `ComposedEquationSystem`
- removed per-call equation lookup/dictionary assembly from the hot path

Observed result:
- cleaner hot path
- no meaningful extra timing improvement on the current benchmark

Still open in this phase:
- a truly dedicated `temperature + Er` execution path, rather than only a lighter generic composed path
- optionally later, a dedicated `density + temperature + Er` execution path

### Phase 4: Preserve Modularity And Avoid Logic Duplication
- Status: ongoing
- Centralize shared solver-state transforms in `_transport_solvers.py`.
- Keep projection and packing logic in shared helpers rather than per-backend custom code.
- Keep debug and reference paths readable even if the optimized path is lower level.
- Add clear comments near any optimized path that explain:
  - which physics semantics are preserved
  - which reference helper it mirrors

Current assessment:
- modularity has been preserved so far
- equation objects remain the physics source of truth
- future dedicated execution paths should still be written as thin adapters, not duplicate physics implementations

### Phase 5: Benchmark And Validate
- Status: active
- Benchmark first on:
  - `examples/Solve_Er_General/transport_pressure_Er_debug_kvaerno5_temp_Er.toml`
- Compare before/after for:
  - XLA compile time
  - total solve wall time
  - Diffrax accepted/rejected/total steps
- Confirm no change in:
  - temperature profiles
  - `Er` evolution summary
  - boundary-condition behavior after the Robin fix
- After Kvaerno validation, confirm the same refactor benefits:
  - `radau`
  - `rosenbrock`

Current benchmark record:
- Original Kvaerno-style benchmark:
  - total solve about `622 s`
  - compile warning phase about `156 s`
  - steps about `928 accepted / 288 rejected / 1216 total`
- After Phase 1 refactor:
  - total solve about `351 s`
  - compile warning phase about `123 s`
  - steps `315 accepted / 78 rejected / 393 total`
- After later refactors:
  - total solve remained about `357-359 s`
  - compile warning phase remained about `131-133 s`
  - steps stayed `315 / 78 / 393`

Interpretation:
- major easy win already captured
- remaining time is probably dominated by:
  - Diffrax implicit stage machinery
  - shared flux-model / RHS cost
  - compile cost inside `jit_diffeqsolve`

### Phase 6: Optional Follow-Up Improvements
- Status: not started
- Enable and document persistent JAX compilation cache for repeated local runs.
- Add lightweight timing instrumentation around:
  - RHS evaluation
  - solver setup
  - Diffrax solve call
- Only if needed later, revisit:
  - smarter initial `dt0`
  - reduced rejected steps
  after the structural compile/runtime overhead has been lowered.

### Phase 6B: NTSS Consistency Alignment
- Status: not started
- Goal:
  - tighten agreement between NEOPAX and NTSS on the specific implementation details already identified as likely sources of physics mismatch, without breaking the current modular/JAX-friendly architecture

Primary targets in this phase:
- Match NTSS boundary-condition closure more tightly:
  - preserve the already-corrected NTSS-style `DECAYLEN` / Robin meaning
  - replace the current simple ghost-point Neumann/Robin edge formulas with NTSS-style second-order one-sided boundary closure where appropriate
  - ensure the same boundary algebra is used consistently across NEOPAX BC application paths
- Match endpoint flux behavior more tightly:
  - review how NTSS derives edge fluxes from edge-constrained state/gradient
  - reduce mismatch between NEOPAX’s ghost/face reconstruction path and NTSS’s edge-state-first boundary treatment
  - identify which equations should remain state/gradient-driven versus flux-driven at the edge
- Check and correct source/turbulence normalization details against NTSS:
  - revisit the `POWER_OVER_N` / `turbulent_power_analytical` density normalization path
  - confirm the exact `total_power` definition used in the intended NTSS comparison branch
  - re-check pressure-source assembly term-by-term against the chosen NTSS reference branch where needed

Implementation constraints for this phase:
- keep the code JAX/JIT/trace-friendly
- avoid duplicating physics logic in parallel code paths
- where possible, centralize the NTSS-aligned boundary algebra so ghost-cell and face-constraint paths do not diverge

Expected validation in this phase:
- compare near-edge profile shape against NTSS for temperature cases with `DecayLen`
- compare endpoint gradient/flux behavior, not only cell-centered profiles
- re-check the previously suspicious turbulent heat-flux amplitude after normalization consistency fixes

### Phase 7: Radau Modernization Track
- Status: in progress
- Keep this as a Radau-specific branch of the solver plan, distinct from the main Kvaerno efficiency work.
- Use the classic Hairer/NTSS Radau implementation and the newer Julia `AdaptiveRadau` / `OrdinaryDiffEqFIRK` implementation as design references.

Priority items for this phase:
- Implement a Hairer-style transformed-stage linear solve for Radau:
  - avoid treating the 3-stage Radau Newton system as one monolithic `3n` solve
  - transform the stage system into block structure (one real `n x n` block plus one real `2n x 2n` block for the 3-stage case)
  - target lower memory pressure, smaller linear algebra workspaces, and lower compile/runtime cost
- Prefer direct factorization for the transformed Radau blocks where practical:
  - make the transformed-block direct solve the primary NTSS/Hairer-style path
  - use GMRES mainly as a fallback for cases where direct transformed-block solves are not practical
  - avoid equating "direct solve" with a naive dense solve of the full coupled `3n` system; the target is direct factorization of the transformed blocks
- Improve Radau Newton convergence and rejection logic:
  - track contraction more explicitly
  - detect bad Newton progress earlier
  - shrink `dt` earlier when Newton is diverging instead of exhausting iterations
  - use a more mature acceptance-history-aware step update policy
- Improve Radau Jacobian / factorization reuse policy:
  - go beyond the current simple `dt_close + age` cache
  - make reuse/update decisions more consistent with Hairer-style Radau practice

Lower-priority items for later:
- investigate whether any banded/block structure in the transport Jacobian can be exploited in the Radau linear solve
- revisit Radau linear-solver policy (`gmres` vs direct/auto) after the transformed-stage solve exists
- only after the above, consider larger-scope features such as adaptive-order Radau or arbitrary-order tableau generation

Additional NTSS-inspired implementation ideas to evaluate for JAX/JIT-friendly adoption:
- make Jacobian refresh/factorization reuse decisions more Hairer-like:
  - reuse factorized transformed blocks across accepted steps when the step size/state change criteria allow it
  - separate "reuse Jacobian" from "reuse factorization" conceptually in the solver design
- preserve the transformed-stage solve as a static-shape, array-level path:
  - no Python-side dynamic matrix assembly in the hot loop
  - keep all transformed block operators representable as JAX arrays / pure matvecs
- investigate whether limited banded-structure approximations or block-structured dense paths can be exploited without breaking JAX compilation behavior
- avoid iterative-Krylov-first designs in the primary Radau path when NTSS-style direct transformed-block factorization is feasible
- compare NTSS-style step acceptance heuristics and Newton restart policies against the current custom implementation for possible low-overhead adoption

Current assessment for this phase:
- the transformed-stage solve is the highest-value missing Radau algorithmic feature
- the NTSS/Hairer-style direct transformed-block factorization should be considered part of that same main feature, not an optional extra
- adaptive-order Radau is interesting, but not the first thing to port into NEOPAX
- given the current state sizes (for example `state_size ~ 459` in the benchmarked `temperature + Er` case), Radau may remain a secondary high-accuracy path rather than the main default solver target

Completed in this phase so far:
- added a transformed-stage simplified-Newton solve for the custom 3-stage Radau path in `_transport_solvers.py`
  - the simplified-Newton path no longer treats the Radau stage system as one monolithic `3n` solve
  - it now uses a real transformed-stage block structure (`n` block + `2n` block)
- the transformed-stage path remains JAX/JIT/trace/differentiation friendly:
  - static Radau transform constants are precomputed outside the hot loop
  - the traced step loop still uses JAX arrays and pure functions
- added a first contraction-aware Newton rejection improvement:
  - the Newton loop now tracks previous residual norm and a simple contraction estimate
  - clearly bad Newton progress can now trigger earlier rejection/shrinkage instead of only exhausting the maximum iteration budget
- improved Jacobian reuse behavior in the simplified-Newton Radau path:
  - Jacobian recomputation is now lazy when the cache is valid
  - rejected-step retries now keep the freshly computed Jacobian/cache state instead of falling back to an older cache entry

Observed result so far:
- the lean Radau `temperature + Er` benchmark no longer gets killed immediately after entering `solver.solve(...)`
- with transformed blocks plus direct linear solves on the transformed system, the benchmark completed with:
  - total `solver.solve(...)` time about `764 s`
  - compile warning phase about `3m29s`
  - `n_steps = 1140`
  - `failed_any = False`
- interpretation:
  - robustness improved materially
  - step count only improved modestly, so NEOPAX Radau still appears weaker than NTSS mainly in step-control / acceptance policy rather than in the core transformed linear algebra
  - compile time is now a clearer bottleneck, so future Radau work should prefer NTSS-like controller improvements and hot-path simplification over adding more generic branching

Current benchmark picture for the physical `temperature + Er` case:
- `diffrax_kvaerno5` on the same physical case completes in about `328.7 s`
  - Diffrax stats on that case:
    - `num_steps = 296`
    - `num_accepted_steps = 239`
    - `num_rejected_steps = 57`
- current best custom Radau benchmark on that case is about:
  - compile `~4m10s`
  - total `solver.solve(...)` `~567.2 s`
  - `n_steps = 1229`
  - `failed_any = False`

Additional completed work in this phase:
- stripped the active Radau implementation down to the NTSS-style path only:
  - simplified Newton
  - transformed stage blocks
  - direct LU solves
  - no active GMRES / full-Newton alternative path in the traced Radau step
- removed one stale duplicate standard-Radau implementation and cleaned misleading Rosenbrock leftovers in `_transport_solvers.py`
- restored the Radau path to a clean benchmarkable state after a failed intrusive diagnostics experiment

Negative experiments / conclusions to remember:
- a more aggressive history-loosened Jacobian/LU reuse policy reduced average step cost but pushed step count from about `871` to about `1234`, and total solve time got worse; this was reverted
- intrusive Radau diagnostics threaded through the hot loop badly inflated compile/runtime and were removed
- small carry-packing / loop-cleanup changes gave only marginal gains; they are not enough by themselves to close the compile gap to Diffrax
- the older lower-step Radau regime (`~871` steps) was not the best overall runtime regime; newer controller tuning produced a better total runtime while remaining on a higher-step branch

Current NTSS / standard-method interpretation:
- keep using standard Radau IIA / Hairer-Wanner method structure as the numerical base
- borrow NTSS-inspired controller heuristics where they clearly help this transport problem
- do not chase exact NTSS line-by-line mimicry when it depends on codebase-specific surrounding assumptions

Next method-focused work still missing before the larger refactor:
- one more narrow pass on Radau step-control / acceptance policy, staying conservative:
  - focus on more standard / NTSS-like acceptance and `h` evolution
  - avoid broad reuse-loosening experiments
  - keep the active path mathematically the same Radau IIA method
- short immediate benchmark step before the structural refactor:
  - rerun the same physical Radau case with `save_n = 1`
  - use that as the cleanest config-side test of how much compile time is coming from save-buffer bookkeeping in the custom Radau loop

New structural milestone for compile-time reduction:
- add a dedicated Diffrax-like structural refactor track for custom Radau after the next narrow method pass:
  - extract a compact `radau_step(...)` kernel
  - separate solver state from save/output bookkeeping more cleanly
  - shrink the visible loop carry toward a single solver-state pytree/dataclass
  - simplify save handling so it resembles the cleaner `SaveAt`-style separation used by `DiffraxSolver.solve(...)`
  - keep the same solver numerics:
    - same Radau IIA tableau
    - same transformed simplified Newton path
    - same error estimator / controller semantics unless explicitly retuned

Reason for ordering:
- controller quality still appears to be the main remaining runtime weakness versus NTSS / state-of-the-art behavior
- but compile time is now large enough that a Diffrax-like structural cleanup should follow soon after the next narrow controller pass, not be postponed indefinitely
- current evidence suggests:
  - controller tuning can still improve total solve time even when step count stays high
  - compile time remains too large, so the `save_n = 1` benchmark is the last simple config-side check before moving into the structural refactor
  - memory/stability likely improved enough for the lean benchmark to finish
  - runtime is still too slow, so more work is needed on Newton/factorization/step-control efficiency
- after adding lazy Jacobian reuse, preserved reject-retry cache state, and explicit transformed-block LU-factor reuse, the same lean benchmark improved further to:
  - total `solver.solve(...)` time about `574 s`
  - compile warning phase about `3m59s`
  - `n_steps = 1039`
  - `failed_any = False`
- updated interpretation:
  - transformed-block factorization reuse produced a real runtime win
  - step count only improved modestly, so NEOPAX Radau still appears weaker than NTSS mainly in step-control / acceptance policy rather than in the core transformed linear algebra
  - compile time is now a clearer bottleneck, so future Radau work should prefer NTSS-like controller improvements and hot-path simplification over adding more generic branching
- subsequent modernization on the fuller physical `temperature + Er` case established a cleaner current baseline:
  - `diffrax_kvaerno5` on the same physical case completes in about `328.7 s`
- current best custom Radau benchmark on that case is about:
    - compile `~4m10s`
    - total `solver.solve(...)` time `~567.2 s`
    - `n_steps = 1229`
    - `failed_any = False`
- compile-focused findings from recent experiments:
  - removing the carried `done` flag and trimming active-loop bookkeeping produced real wins
  - unifying the save/non-save loop bodies did not help and was reverted
  - tiny carry-packing / readability cleanups are now close to neutral in timing impact
  - stripping the active Radau implementation down to the NTSS-style path only:
    - simplified Newton
    - transformed blocks
    - direct LU solves
    was still worth doing to reduce solver generality before more controller work
- controller-focused findings from recent experiments:
  - a stronger accepted-step `h` holding rule improved the current physical benchmark without reducing step count materially
  - this suggests the next likely wins are in making accepted steps cheaper via better Jacobian/LU reuse rather than expecting large step-count drops from each small controller tweak
  - a later relaxation of accepted-step growth caps improved the fuller physical benchmark further:
    - compile improved to about `4m10s`
    - total solve time improved to about `567.2 s`
    - `n_steps` stayed high at about `1229`
  - interpretation:
    - the current controller is now in a faster high-step regime
    - total runtime improved, but the method still does not look state-of-the-art in step economy
    - compile time is still much larger than `diffrax_kvaerno5`
  - a later history-loosened Jacobian/LU reuse experiment was reverted after proving too aggressive:
    - compile improved to about `4m06s`
    - total solve time worsened to about `614.5 s`
    - `n_steps` jumped from about `871` to about `1234`
  - interpretation:
    - the experiment likely reduced average cost per step
    - but it harmed controller behavior enough to make overall efficiency worse
    - the active baseline therefore remains the earlier `~4m16s / ~596.3 s / 871 steps` state

NTSS comparison findings relevant to this phase:
- NTSS `CRadau/radau.cpp` does not use GMRES in its main Radau path
  - it uses direct decomposition/solve routines on transformed block systems (`decomr_`, `decomc_`, `slvrad_`, `slvrar_`, `slvrai_`)
- NTSS reject/retry handling keeps the newest Jacobian when possible
  - on a rejected step, if the Jacobian is still considered valid (`caljac` true), NTSS retries from the matrix/decomposition path instead of forcing a full Jacobian rebuild
  - this matches the intended direction of the recent NEOPAX change that preserves `jacobian_out`/cache state across rejected retries
- NTSS still goes beyond the current NEOPAX implementation by combining Jacobian reuse with more mature decomposition reuse and step-history logic
- NTSS-style behavior appears highly tolerance/controller sensitive in practice:
  - a similar NTSS Radau run reportedly needs about `54793` steps at one setting
  - but only about `260` steps with `rtol = 1.e-3`
  - so future NEOPAX comparisons should continue to track controller/tolerance regime, not just method name

Important next improvement steps for this phase:
- add explicit transformed-block factorization reuse across accepted/rejected retries when:
  - the cached Jacobian is reused
  - the transformed-block direct path is active
  - the step-size change remains within a safe reuse window
- separate the concepts of:
  - Jacobian reuse
  - transformed-block decomposition/factorization reuse
  so NEOPAX can move closer to NTSS/Hairer behavior without overcoupling cache decisions
- reduce compile-heavy branching inside the traced Radau loop where possible
  - recent runtime improvements have come with heavier `jit_while` compile cost
  - future Newton/retry logic should prefer static-shape, low-branch formulations
- tighten the Radau step controller in a more NTSS-like direction:
  - remember recent rejected steps
  - cap post-rejection step regrowth
  - apply stronger shrinkage after repeated rejected retries
  - use Newton contraction quality to limit aggressive accepted-step growth after difficult solves
- refine Jacobian and LU refresh policy on the stripped standard path:
  - allow longer reuse after easy accepted-step streaks with stable `h`
  - force earlier refresh after recent rejection or poor `theta`
  - keep this narrow so the current compile gains are not given back
- consider simplifying the active Radau backend around the NTSS-like path:
  - `simplified` Newton
  - transformed blocks
  - direct factorization
  while keeping any legacy alternatives outside the primary benchmarked execution path if compile time continues to dominate
- immediate next benchmark step:
  - run `examples/Solve_Er_General/transport_pressure_Er_debug_radau_temp_Er.toml` with `save_n = 1`
  - compare against the current `save_n = 10` reference:
    - compile `~4m10s`
    - total solve `~567.2 s`
    - `n_steps = 1229`
  - if compile remains too large after that, begin the planned Diffrax-like structural refactor for custom Radau
- after factorization reuse is in place, re-benchmark the lean `temperature + Er` Radau case and compare:
  - compile time
  - total solve time
  - total step count
  - whether runtime improvements continue without another major compile regression

Still open in this phase:
- see whether more NTSS-like step-control logic can reduce `n_steps` materially relative to the new `~574 s / 1039 steps` baseline
- revisit transformed-block direct-vs-GMRES policy after more timing data
- investigate whether factorization reuse can be made more explicit and more NTSS-like
- decide later whether any remaining compile gap to `kvaerno5` is worth a more opinionated dedicated Radau kernel beyond the current stripped standard path
- next-session handoff:
  - keep using `examples/Solve_Er_General/transport_pressure_Er_debug_radau_temp_Er.toml` for Radau comparisons
  - compare against the current custom-Radau reference:
    - compile `~4m10s`
    - total solve `~567.2 s`
    - `n_steps = 1229`
  - keep `diffrax_kvaerno5` on the same physical case as the external comparison point:
    - total solve `~328.7 s`
  - immediate next action:
    - benchmark the same Radau case with `save_n = 1`
    - then decide whether to start the Diffrax-like structural Radau refactor
    - `296` total steps (`239` accepted / `57` rejected)

### Next Recommended Steps
1. Investigate shared flux-model / RHS cost, especially the neoclassical face/state evaluation path.
2. Add or document persistent JAX compilation cache for repeated local benchmarking.
3. If solver-wrapper work continues, implement a truly dedicated `temperature + Er` vector field path instead of only optimizing the generic composed system.
4. Re-benchmark after any flux-path optimization before touching more solver scaffolding.
5. Only after Kvaerno is understood better, validate whether the same infrastructure materially helps `radau` and `rosenbrock`.
6. For custom Radau specifically, prioritize transformed-stage linear algebra and improved Newton/rejection logic before broader feature work.
7. For custom Radau, continue benchmarking the lean `temperature + Er` case after each modernization step before moving back to the fuller coupled case.

### Immediate Execution Order
1. Profile or simplify the shared flux-model evaluation path.
2. Enable persistent JAX compilation cache for repeated benchmark runs.
3. If needed, add a truly dedicated `temperature + Er` execution path.
4. Re-benchmark Kvaerno after each substantial change.
5. Then test whether the same changes improve `radau` and `rosenbrock`.
6. If custom Radau becomes the focus, start with the Hairer-style transformed-stage solve, then improve Jacobian reuse and Newton step control.
7. Next Radau-specific checkpoint:
   - rerun `examples/Solve_Er_General/transport_pressure_Er_debug_radau_temp_Er.toml`
   - compare against the current transformed-direct baseline (`~574 s`, compile `~3m59s`, `1039` steps)

### Phase 8: Replace Custom Rosenbrock With Standard RODAS5P
- Status: not started
- Goal:
  - retire the current custom Rosenbrock backend and replace it with a literature-standard high-order Rosenbrock-Wanner method better suited to NEOPAX stiff transport problems

Why this phase exists:
- the current custom Rosenbrock path was useful as an experimental backend, but it has not been competitive enough in robustness/performance
- a standard `RODAS5P`-style method is a much better target than continuing to evolve a nonstandard Rosenbrock variant
- `RODAS5P` is closer to the class of mature stiff solvers expected for this problem class, while still being distinct from `Kvaerno5`

Primary tasks in this phase:
- remove or fully replace the current custom Rosenbrock step formulation in `_transport_solvers.py`
- implement a standard `RODAS5P`-style Rosenbrock-Wanner method with:
  - literature-standard tableau/coefficient set
  - embedded error estimator
  - adaptive step size control
  - proper Jacobian reuse policy
  - `save_n` support consistent with the other custom solvers
- keep the implementation JAX/JIT/trace/differentiation friendly:
  - static tableau constants outside the hot loop
  - array-level stage/state operations
  - no Python-side per-step control in traced loops
- support the same NEOPAX solver-state transform machinery:
  - flat packed state path
  - quasi-neutral projection semantics
  - fixed-temperature projection semantics

Secondary tasks:
- evaluate whether the `RODAS5P` path should prefer direct solves, GMRES, or a hybrid policy for NEOPAX state sizes
- benchmark `RODAS5P` against:
  - `diffrax_kvaerno5`
  - custom `radau`
  on the same lean `temperature + Er` benchmark first

Validation targets:
- confirm the new backend is materially more standard and robust than the current custom Rosenbrock implementation
- compare compile time, wall time, step count, and survival on the lean benchmark case
- only after lean-case validation, test fuller coupled cases
