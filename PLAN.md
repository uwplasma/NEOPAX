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

Known issue / parity note:
- ambipolar `Er` initialization still shows a small NTSS mismatch concentrated in the last one to two radial points, even after:
  - pre-applying BC-corrected density/temperature state before ambipolar root finding
  - passing right-boundary value/gradient constraints into the local `monkes_database` ambipolar flux evaluator
- current diagnosis:
  - the bulk ambipolar profile is now close, so this is likely an outer-edge reconstruction / edge-root-evaluation parity issue rather than a full-profile neoclassical mismatch
  - transport RHS uses the boundary-aware face-flux path, while ambipolar initialization still relies on the local `get_Neoclassical_Fluxes(...)` path with internally reconstructed gradients
  - the next likely parity improvement is a more NTSS-like outer-edge ambipolar evaluation for the last one to two points, or an optional face-based ambipolar evaluator for parity studies
- separate known issue / parity note:
  - transport-mode ambipolar `Er` initialization can still disagree with explicit `mode = "ambipolarity"` root-solving for the same starting profiles/configuration
  - this mismatch appears to affect the root-transition position in the interior, not only the outermost boundary points
  - user recollection suggests the discrepancy may predate the recent BC-gradient / boundary-preprocessing changes, so those edits should not be assumed to be the original cause
  - next diagnostic target:
    - compare transport-mode `best_roots`, `er_init`, and plotted `t=0` `Er` directly against the explicit ambipolarity-mode `best_roots` for the same effective runtime settings

### Phase 7: Radau Modernization Track
- Status: in progress
- Keep this as a Radau-specific branch of the solver plan, distinct from the main Kvaerno efficiency work.
- Use the classic Hairer/NTSS Radau implementation and the newer Julia `AdaptiveRadau` / `OrdinaryDiffEqFIRK` implementation as design references.

Updated assessment:
- the recent custom 3-stage lean Radau work found a good stable baseline, but did not yet reduce step count below roughly `713` on the benchmark transport case
- the strongest remaining gap to NTSS/Hairer is now believed to be in:
  - predictor quality
  - adaptive step-size controller quality
  - Jacobian / LU reuse heuristics
  - and, longer-term, adaptive-order / adaptive-stage Radau capability
- compile-focused micro-refactors have mostly hit diminishing returns; the next high-value Radau work should be more algorithmic and more Hairer-like

Priority items for this phase:
- Implement a more Hairer/NTSS-like adaptive step-size controller:
  - add a predictive / Gustafsson-style controller option for the custom Radau path
  - compare it against the current lean controller on the transport benchmark
  - target fewer rejected / unnecessarily small accepted steps rather than compile-only gains
- Implement a stronger Hairer-style collocation predictor:
  - use extrapolated collocation / prior accepted-step stage information more faithfully
  - keep zero-start / fallback behavior available for difficult Newton regimes
  - target fewer Newton iterations and larger accepted steps
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

New explicit sub-track: Adaptive Radau
- Goal:
  - investigate an adaptive-order / adaptive-stage Radau path closer to the broader Hairer code family, instead of staying permanently fixed at the current 3-stage method
- Scope:
  - start from the Hairer/NTSS stage-count controls already visible in the code/comments
  - evaluate whether NEOPAX should support:
    - fixed 3-stage Radau as the default practical path
    - optional adaptive-stage Radau for higher-accuracy / fewer-step regimes
- Initial questions for this sub-track:
  - how much of the NTSS step-count advantage is coming from better controller/predictor tuning versus higher stage/order selection
  - whether adaptive-stage Radau can be implemented in a JAX-friendly way without exploding compile cost
  - whether a small menu of static-shape Radau variants (for example fixed 3-stage and fixed higher-stage options) is a better first step than fully dynamic stage switching
- Recommendation:
  - treat adaptive Radau as an explicit planned phase item, but only after the predictor/controller improvements above are benchmarked
  - do not let adaptive-stage work block the nearer-term controller/predictor work

Current updated recommendation after fixed-stage benchmarking:
- fixed higher-stage Radau is now working well in NEOPAX:
  - `3` stages: about `713` steps, about `301 s`
  - `5` stages: about `209-215` steps, about `250-266 s`
  - `7` stages: about `163-166` steps, about `250-256 s`
- additional exploratory higher-stage runs are also now in a strong regime:
  - `9` stages: about `143-149` steps, about `232-250 s`
  - `11` stages: about `147-151` steps, about `237-241 s`
- current interpretation:
  - `3 -> 5 -> 7` gave a strong monotonic improvement
  - `9` and `11` are both high-performing, but `11` does not clearly beat `9`
  - `9` stages currently look like the best balance / likely sweet spot on this benchmark
- `9`-stage Radau should therefore be treated as the current custom Radau reference configuration for this benchmark regime
- the most JAX-friendly adaptive-order direction is now:
  - a discrete fixed-family adaptive Radau over `3 / 5 / 7 / 9` stages
  - not a single fully dynamic shape-changing solver kernel

Proposed adaptive-order design:
- keep separate fixed-stage solver kernels for:
  - `3` stages
  - `5` stages
  - `7` stages
- switch order only at the outer accepted-step level
- choose the next order based on:
  - recent error ratio
  - rejected-step history
  - Newton difficulty / contraction quality
  - step-size smoothness
- practical first policy:
  - start from `5`
  - drop to `3` if Newton/rejections become difficult
  - raise to `7` in smooth low-error regions

Expected implementation / resource profile:
- runtime state memory should increase only modestly if only the active-order cache is kept live
- compile/cache memory will increase because JAX will likely compile separate kernels for `3`, `5`, and `7`
- to control memory growth, avoid carrying all stage-specific reuse caches simultaneously unless benchmarking later proves that beneficial

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
- adaptive-order / adaptive-stage Radau is now a planned later sub-track, but still should not be the first Radau feature ported into NEOPAX
- given the current state sizes (for example `state_size ~ 459` in the benchmarked `temperature + Er` case), Radau may remain a secondary high-accuracy path rather than the main default solver target

Near-term execution order for this phase:
1. benchmark and preserve the current best fixed-3-stage custom Radau baseline
2. implement a more Hairer-like predictive/Gustafsson controller
3. improve the extrapolated collocation predictor
4. revisit Jacobian / factorization reuse policy with NTSS-style thresholds
5. only then evaluate adaptive-stage / adaptive-order Radau designs

Current checkpoint for this phase:
- keep two explicit Radau controller modes:
  - `radau_controller_mode = "standard"`
    - the more NTSS-like / history-aware controller path
  - `radau_controller_mode = "lean"`
    - the compile-experiment / lower-step controller path
- default the solver back to `standard` unless `lean` is requested explicitly
- recent lean benchmark on
  - `examples/Solve_Er_General/transport_pressure_Er_debug_radau_temp_Er.toml`
  produced:
  - compile about `4m34s`
  - total first-run `host_return_elapsed_s ~ 600.8 s`
  - `n_steps = 713`
  - `failed_any = False`
- interpretation:
  - compile time did not materially improve
  - but the execution regime became much less conservative in step count
  - so lean remains worth preserving as an alternate controller mode, not as the only Radau path

Current compile-focused conclusion:
- the main remaining compile burden is not in:
  - save buffering
  - outer loop-carry shape
  - small cache-state cleanups
- the main burden is more likely in the traced inner Radau step itself:
  - Jacobian construction path
  - transformed-stage solve staging
  - Newton loop
  - residual/stage evaluation structure

Latest compile-focused implementation step:
- simplified the 3-stage residual evaluation inside the custom Radau step from an explicit 3-call stack build to a batched `jax.vmap(...)` stage evaluation
- goal:
  - make the inner traced step more uniform and less manually unrolled
  - preserve the same Radau numerics while moving the implementation shape a bit closer to a library-style batched stage evaluation

Immediate next Radau compile steps:
1. benchmark the current `standard` controller mode again after the batched stage-evaluation refactor
2. benchmark the same case with explicit `radau_controller_mode = "lean"` to keep the two-path comparison honest
3. if compile is still flat, stop micro-tuning wrappers and focus only on:
   - Jacobian construction structure
   - transformed linear-solve staging
   - Newton/residual kernel regularity

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
- a combined Hairer-inspired tuning batch on the fixed 3-stage custom Radau path was clearly negative and has been reverted:
  - ingredients tested together:
    - stronger stage-history/drift predictor
    - predictive / PI-style controller term
    - NTSS-like reuse step-ratio window and "keep same dt" heuristic
  - observed result on the transport benchmark:
    - compile improved into roughly the `~2m05s` to `~2m10s` range
    - but `n_steps` exploded from the good baseline of about `713` to about `3167`
    - total solve time degraded to about `362-382 s`
  - conclusion:
    - this batch made the custom Radau controller much too conservative / step-inefficient
    - future Radau tuning should test only one lever at a time:
      - predictor only
      - controller only
      - reuse policy only
    - do not reapply this combined tuning batch as a package
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

### Phase 9: Ambipolarity / Interpolation Compile Cleanup
- Status: noted for later
- Goal:
  - reduce compile-time noise and overhead in the ambipolar initialization / radial root-finding path, separate from the current Radau solver work

Why this phase was added:
- during `ambipolar two_stage radial solve`, XLA reported slow constant folding inside:
  - `interpax/_fd_derivs.py`
  - through the path:
    - `evaluate_block`
    - `get_Neoclassical_Fluxes`
    - `get_Lij_matrix`
    - `get_Dij`
    - `interpolation_small_r`
    - `interpolator_nu_Er_general`
- this appears before `solver.solve(...)`, so it is not a Radau regression
- it is likely compile overhead in the neoclassical interpolation / derivative setup used by ambipolar root finding

Primary tasks for this later phase:
- inspect where `interpax` derivative/interpolator objects are being created or specialized during ambipolar root finding
- determine whether any interpolation setup can be:
  - cached
  - hoisted
  - or made more static-shape / less recompilation-prone
- benchmark ambipolar initialization time separately from transport solve time after any changes

Scope note:
- do not mix this work into the current Radau optimization track
- treat it as a separate compile/initialization cleanup phase once the current solver improvements are checkpointed

### Phase 10: Shared Transport-Response Decomposition For Theta / Radau
- Status: planned
- Goal:
  - add a solver-facing transport decomposition layer, closer in spirit to Trinity3D and partly to TORAX, so both `theta` and `radau` can optionally run on a frozen local transport-response model instead of only on the current black-box assembled RHS

Why this phase was added:
- current NEOPAX modularity stops at:
  - `models/equations -> final semidiscrete RHS`
- this is excellent for generic ODE solvers, but it hides:
  - base flux vectors
  - source vectors
  - local transport response / Jacobian-like structure
- both the future TORAX-like theta direction and a more transport-aware Radau path would benefit from an intermediate layer that exposes:
  - local flux decomposition
  - local source decomposition
  - local linear response information
- Trinity3D suggests a more reachable first target than a full TORAX `Block1DCoeffs` abstraction:
  - compute base fluxes/sources
  - compute perturbative local response / Jacobian-like operator
  - assemble a transport linearization used by the solver

Primary design idea:
- add a new optional solver mode where, for one step attempt or one outer nonlinear iterate:
  - base transport flux/source terms are computed at a reference state
  - a local transport response is built around that state
  - that local response is frozen for the internal solve
- use the same decomposition for:
  - a Trinity/TORAX-like theta solve
  - a lagged-response Radau solve

Intended mathematical form:
- locally approximate transport fluxes/sources as:
  - `Q ~= Q0 + J_Q * delta y`
  - `Gamma ~= Gamma0 + J_Gamma * delta y`
- assemble from this a discrete transport response object containing items equivalent in spirit to:
  - base flux/source vectors
  - geometry/grid divergence factors
  - local implicit response matrix / operator
- keep this frozen over:
  - one theta nonlinear iterate
  - or one Radau step attempt, if running in the lagged-response mode

Primary tasks in this phase:
- define a NEOPAX transport-response interface sitting between:
  - current equation/physics evaluation
  - solver implementation
- first target a Trinity3D-like decomposition rather than a full TORAX-style coefficient object:
  - base particle/heat flux vectors
  - source vectors
  - local response/Jacobian-like operator
- make this decomposition available from the active transport state without breaking the current black-box RHS path
- keep the current assembled-RHS path as a reference/validation mode

Theta-specific tasks:
- add a transport-response-backed theta mode in addition to the current generic-RHS theta implementation
- let the solver:
  - build one frozen local transport response per nonlinear iterate
  - solve the resulting theta linearized system
- use this as the path intended to move closer to Trinity3D/TORAX behavior

Radau-specific tasks:
- add an optional lagged-response Radau mode
- in this mode:
  - one local transport response is built per step attempt
  - that response is frozen across all internal Radau stages and Newton iterations of that attempt
- note carefully:
  - the current custom Radau already freezes the black-box RHS Jacobian over a step attempt
  - the new mode would go further by freezing the underlying transport-response model itself, not only `df/dy`

Expected benefits:
- create one shared decomposition usable by multiple solvers
- reduce expensive within-step transport-model reevaluations for:
  - theta
  - and especially Radau in future expensive-flux settings
- make NEOPAX much better prepared for:
  - perturbative/GK-style flux models
  - Trinity3D-like response construction
  - fuller TORAX-like coefficient abstractions later if desired

Expected tradeoffs / risks:
- this is an approximation layer:
  - the true nonlinear flux response may vary across Radau stages within a step
- therefore a frozen-response Radau mode may sacrifice some of the full nonlinear collocation fidelity of the current black-box RHS Radau path
- likely outcome:
  - still better transient fidelity than theta
  - cheaper than fully refreshed stage-by-stage nonlinear flux updates
  - but not identical to the current fully nonlinear residual evaluation path

Validation plan for this phase:
- compare, for both theta and Radau:
  - current black-box RHS mode
  - new frozen transport-response mode
- track:
  - compile time
  - total solve time
  - step count / accepted steps
  - physics agreement in transient and final state
- pay special attention to cases with:
  - hysteresis / multiple branches
  - strongly state-dependent sources
  - threshold-like transport behavior

Implementation ordering recommendation:
1. define the shared transport-response decomposition API
2. implement the decomposition in a Trinity3D-like direct response form first
3. wire it into theta before Radau
4. only then add the lagged-response Radau mode and benchmark whether it preserves enough of Radau's transient advantages to justify its complexity

### Theta Solver Upgrade Notes
- Status: planned
- Scope:
  - capture near-term robustness and adaptive-timestep improvements for the current `theta_newton` path, distinct from the larger shared transport-response refactor above

Current baseline:
- `theta_newton` now has:
  - Newton line search
  - timestep retry / backtracking on failed Newton convergence
  - accepted-step timestep adaptation based on nonlinear iteration count
- this is meaningfully more robust than the earlier fixed-step-like behavior, but still not as mature as TORAX's overall theta/Newton ecosystem

Candidate upgrades:
- split line-search reduction from timestep-backtracking reduction
  - rationale:
    - `theta_delta_reduction_factor` is currently doing two jobs:
      - shrinking the Newton step during line search
      - shrinking the outer timestep during backtracking
    - this should become two distinct controls
  - TORAX similarity:
    - similar in spirit to TORAX
    - TORAX conceptually distinguishes `delta x` line search from `dt` backtracking

- use a better accepted-step adaptation metric than nonlinear iteration count alone
  - possible signals:
    - final residual norm
    - ratio of final to initial residual
    - total line-search severity
    - whether any retry/backtracking happened before acceptance
  - rationale:
    - current controller is practical, but still heuristic and relatively blind
  - TORAX similarity:
    - partly similar
    - TORAX clearly uses nonlinear-solve quality to decide whether a timestep attempt is acceptable
    - but this exact accepted-step growth policy would be a NEOPAX-specific design choice

- add hysteresis / short memory before re-growing `dt`
  - rationale:
    - reduce grow-fail-shrink oscillation
    - only allow aggressive regrowth after several easy accepted steps
  - TORAX similarity:
    - not directly from TORAX docs
    - mainly a NEOPAX-specific solver-controller refinement

- add explicit unphysical-state rejection checks inside the theta Newton path
  - examples:
    - reject negative temperatures before projection
    - reject obviously nonphysical pressure states
    - optionally reject extreme `Er` excursions if needed
  - rationale:
    - make acceptance logic more explicit instead of relying only on floors/projection and residual behavior
  - TORAX similarity:
    - similar to TORAX
    - TORAX docs explicitly mention rejecting Newton steps that produce unphysical states

- add a transport-informed timestep cap / calculator for theta
  - possible direction:
    - optional `theta_timestep_mode = "fixed" | "nonlinear" | "chi_transport"`
    - in `chi_transport`, cap or propose `dt` using a transport timescale estimate based on the maximum effective heat diffusivity
  - rationale:
    - move beyond pure failure-driven timestep adaptation
    - better protect against fast transients in stiff transport regimes
  - TORAX similarity:
    - strongly similar to TORAX
    - TORAX has a separate adaptive timestep method based on `chi_max`

- add richer theta diagnostics
  - suggested outputs:
    - min/max accepted `dt`
    - number of retries
    - number of reduced-`dt` recoveries
    - number of hard failures at `min_step`
    - typical nonlinear iteration counts
  - rationale:
    - make controller tuning and solver assessment much easier
  - TORAX similarity:
    - not specifically TORAX-derived
    - mainly NEOPAX-specific usability / debugging infrastructure

Recommended near-term execution order:
1. split line-search reduction from timestep-backtracking reduction
2. add explicit unphysical-state rejection checks
3. improve the accepted-step controller using residual-based quality metrics
4. add hysteresis for `dt` regrowth
5. add an optional transport-informed `dt` cap / calculator
6. expand theta diagnostics after the controller shape stabilizes

### Phase 11: Neoclassical Interpolation Parity Against NTSS
- Status: in progress
- Goal:
  - isolate and reduce the ambipolar root mismatch between NEOPAX and NTSS by comparing and evolving the monoenergetic interpolation model without destabilizing the current generic runtime path

Why this phase was added:
- the original `generic` monoenergetic interpolation path was fast enough, but it produced clear ambipolar root mismatches against NTSS
- an experimental direct `ntss_like` runtime branch became too expensive to compile under JAX and was therefore not a viable production path
- a separate preprocessed interpolation approach proved that:
  - interpolation structure was a major source of the mismatch
  - a JIT-friendly alternative can be much faster than the current generic path
  - the remaining discrepancy is now in specific NTSS branch behavior rather than generic smoothing alone

Current interpolation modes and findings:
- `generic`
  - current baseline NEOPAX path
  - kept intact as the control/reference implementation
- `preprocessed_3d`
  - separate, JIT-friendly 3D interpolation over:
    - `r`
    - `log10(nu)`
    - `log10(|Er| / r)`
  - much faster than the generic path
  - for `n_scale = 1`, profile/root agreement versus NTSS became nearly complete
  - for scaled profile tests such as `n_scale = 0.6`, `T_scale = 0.75`, the branch-transition mismatch improved but remained
- `ntss_preprocessed`
  - separate, faster NTSS-oriented path
  - uses:
    - grouped collisionality bands
    - `Er = 0` handling
    - 3-point / 4-point interpolation in `xnu`
    - `INPOLD`-style interpolation in `Er`
    - NTSS-style radial interpolation structure
  - compile/runtime became practical enough to iterate on
  - still does not fully match NTSS in the transition region or in the last radii

Key implementation updates completed in this phase:
- restored the original generic path after earlier NTSS-like experiments started polluting compile time
- added `preprocessed_3d` as a separate opt-in mode without modifying the generic baseline
- added `ntss_preprocessed` as a separate opt-in mode without modifying the generic baseline
- fixed multiple JAX-static-shape issues in the NTSS-oriented path:
  - removed dynamic `jnp.arange(noi)` usage
  - removed tracer-driven dynamic slicing
  - rewrote the runtime kernel into fixed-shape arithmetic and masked access
- updated ambipolarity-only runs to avoid solving the full radial root problem twice
  - previously ambipolarity mode solved once during Er initialization and again during explicit ambipolarity execution
  - ambipolarity mode now skips the initialization solve
- added the NTSS no-data guards in the positive-field grouped-collisionality interpolation
- added fit-aware logic in the loader/runtime when optional fit arrays are present in the HDF5

Important conclusions so far:
- the remaining mismatch is not due to a plotting bug
  - when the last radii are missing in the plot, the solver is returning non-finite roots there
- the remaining mismatch is not primarily a simple `r` versus `rho` bug
  - the fast preprocessed paths use physical minor radius and the `|Er| / r` normalization in the same basic form as NTSS
- the strongest remaining discrepancies are now in specific NTSS branch formulas rather than in broad interpolation architecture

Confirmed remaining mismatches versus NTSS:
- the positive-field high-PS / high-collisionality branch is still simplified
  - current `high_branch()` is not yet the NTSS `dkftte` fallback
- the low-`nu`, finite-`Er` fit logic is only active when the optional fit arrays are present in the source HDF5
- if those fit arrays are absent in the MONKES/HDF5 file, the newly added fit-aware branches fall back to defaults and therefore do not change results materially
- last-radii root loss is still present
  - this is currently understood as a solver/interpolation edge mismatch, not a plotting truncation issue

Radius-normalization audit summary:
- for the active fast paths, the core normalization is consistent with NTSS:
  - `r = a_b * rho`
  - `xer = log10(max(xref_l, |Er| / r))`
- two caveats remain:
  - NTSS supports a radial scaling factor `fac_sc_r`, which NEOPAX does not currently expose
  - axis treatment is not yet identical:
    - NTSS enforces a finite effective radius near `r = 0`
    - NEOPAX still treats the axis more directly

Validation results recorded so far:
- `preprocessed_3d`
  - major speed win over the baseline generic path
  - strong agreement with NTSS at unscaled profiles
  - partial but incomplete improvement in the transition region under scaled profiles
- `ntss_preprocessed`
  - practical JIT behavior achieved
  - still mismatches in:
    - root-existence interval around the mid-radius branch transition
    - missing final tail roots at the last radii

Primary remaining tasks in this phase:
- inspect the actual HDF5 keys of the production MONKES file and confirm whether the optional fit arrays are present:
  - `lc_fit`
  - `ag11_0`
  - `ag11_sq`
  - `aefld_u`
  - `aex_er`
  - `akn`
  - `air`
  - `xrm`
- if those fit arrays are absent:
  - document clearly that full NTSS parity cannot be reached from the current file alone using only table interpolation
- port the NTSS high-PS fallback (`dkftte` / `dkfttc`) or build a faithful equivalent for the positive-field `high_branch`
- audit collisionality parity between NEOPAX and NTSS before further interpolation-only changes:
  - verify how `cmul = nu / v` is constructed in both codes
  - compare the effective `xnu = log10(cmul)` at selected radii and profile scalings
  - check whether the current NEOPAX Coulomb-log / multi-species collisionality model is shifting the interpolation band relative to NTSS
  - pay special attention to scaled cases where `n / T^(3/2)` changes strongly
- compare pointwise `D11 / D13 / D33` values at selected `(r, nu, Er)` points near the branch-transition region
- add tail diagnostics for the last radii:
  - explicit debug of `best_root[-5:]`
  - `n_roots_all[-5:]`
  - and grouped branch-selection behavior near the edge

### Phase 11B: Optimize The NTSS 1D Kernel
- Status: planned
- Goal:
  - reduce compile/runtime cost of the NTSS-like `Er`-axis interpolation while preserving the root behavior that appears to come specifically from the NTSS 1D interpolation kernel

Why this phase was added:
- recent hybrid interpolation tests showed that:
  - `preprocessed_3d_ntss_radial_ntss1d` cuts compile time significantly relative to full `preprocessed_ntss`
  - and still preserves roughly the same ambipolar root transition
- this strongly suggests the NTSS-like 1D interpolation in the `Er` direction is one of the main ingredients controlling the NTSS-like root location
- by contrast, simplifying that 1D kernel too aggressively moved the root transition back toward the plain `preprocessed_3d` result

Current interpretation:
- likely compile/runtime cost drivers are now:
  - the general `INPOLD`-style 1D interpolation kernel
  - boundary/extrapolation handling inside that kernel
  - repeated per-surface `Er`-axis work for every energy/radius query
- likely high-value target:
  - preserve the NTSS-like 1D behavior for `D11`
  - while making the kernel more JAX-friendly and more specialized to the actual preprocessed database layout

Primary tasks in this phase:
- profile and isolate the hot cost of the current `_inpold_fixed` / `_inpold_segment` path under the hybrid mode
- design a more JAX-friendly NTSS-like 1D kernel specialized to the actual preprocessed case:
  - fixed small `Er` grid size
  - fewer dynamic branches
  - static-shape friendly internal logic
- test precomputing reusable `Er`-axis interpolation metadata in the database object
- evaluate whether most of the benefit is carried by `D11` alone:
  - test a variant with NTSS-like 1D interpolation for `D11`
  - and simpler interpolation retained for `D13` / `D33`
- compare against the current three key modes:
  - `preprocessed_3d_ntss_radial`
  - `preprocessed_3d_ntss_radial_ntss1d`
  - `preprocessed_ntss`

Validation targets for this phase:
- first-call compile time
- total wall time for the ambipolar benchmark setup
- root-transition position
- agreement with the current `preprocessed_ntss` reference behavior

Implementation ordering recommendation:
1. keep `preprocessed_ntss` unchanged as the accuracy reference
2. use `preprocessed_3d_ntss_radial_ntss1d` as the optimization playground
3. optimize the NTSS-like 1D kernel before touching the NTSS radial stencil again
4. only if needed later, revisit whether a partial `D11`-only NTSS 1D path gives most of the physics benefit at lower cost

Implementation ordering recommendation:
1. keep `generic` unchanged as the baseline control path
2. use `preprocessed_3d` as the fast generic comparison path
3. continue refining `ntss_preprocessed` rather than reintroducing a branch-heavy runtime `ntss_like` path
4. check the production HDF5 for optional fit arrays before further fit-branch work
5. audit `nu / v` parity between NEOPAX and NTSS in the scaled-profile mismatch case
6. if needed, port `dkftte` / `dkfttc` next, as that is now the largest known missing NTSS branch
