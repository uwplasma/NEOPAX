# Benchmark Lagged Approach

## Goal

Separate the effects that were being mixed together in the exact-runtime NTX transport benchmarks:

1. raw NTX lagged-response build/evaluation cost
2. Radau retry / rejection behavior
3. physical differences in the initial `Er` and fluxes between exact-runtime variants and the database interpolation baseline
4. solver-side Newton / convergence-status behavior after the initial exact-runtime RHS is made finite

The next session should start from the current conclusions below, not from the earlier assumption that the main problem was radial batching or coarse-anchor interpolation itself.

## Current Conclusions

### 1. `lax.map` over 51 radii is not the core problem

The full `51`-radius exact-runtime lagged path was not primarily failing because of `jax.lax.map` or because it was evaluating too many radii.

Single-attempt benchmarks showed that the major slowdown was being contaminated by repeated solver attempts and rejections.

### 2. The first exact-runtime transport failure was an axis-cell issue

The first decisive exact-runtime transport diagnosis was:

- exact-runtime center neoclassical flux was nonfinite only at:
  - `[species, radius] = [*, 0]`
- all downstream nonfinite quantities propagated from that axis cell

This affected both:

- exact-runtime `lagged_response`
- exact-runtime `black_box`

So the first failure was **not** specific to the lagged-response surrogate.

### 3. The database path is axis-safe because it uses small-radius regularization

The database interpolation path does not simply evaluate a raw coefficient at `r = 0`.

Instead, its monoenergetic interpolation kernels use explicit small-radius regularization:

- clamp the sensitive small-`r` evaluation radius
- regularize `Er / r`-type quantities
- use a small-`r` continuation built from the first three radial database surfaces

So the database path stays finite at the axis even when the direct exact-runtime local solve did not.

### 4. Exact-runtime axis regularization has now been added

The exact-runtime center path now uses a small-`r` continuation for radial index `0`, built from radii `1,2,3`, instead of trusting the raw direct exact solve at the axis.

This regularization is applied to:

- live exact-runtime center `Lij`
- full-radius lagged center response
- coarse-anchor lagged center response

This fix is differentiable because it is pure JAX algebra on arrays, with no stop-gradient or Python-side traced branching.

### 5. After the axis fix, the initial exact-runtime RHS is finite

After applying the axis regularization:

- exact-runtime full-radius `lagged_response`
  - center fluxes finite
  - temperature / pressure / `Er` initial probe finite
  - `finite_f0 = True`
  - `finite_z0 = True`

- exact-runtime `black_box` with `interpolate_center_response`
  - center fluxes finite
  - initial probe finite
  - `finite_f0 = True`
  - `finite_z0 = True`
  - `finite_initial_residual = True`

So the original “exact-runtime transport is nonfinite at `t0`” issue is no longer the main blocker.

### 6. A benchmark-script face-mode bug was found and fixed

Earlier `black_box` comparisons were misleading because:

- `--ntx-face-response-mode` was only being written into the config for `lagged_response`
- `black_box` runs silently kept the model default:
  - `face_local_response`

This has now been fixed.

So the correct apples-to-apples `black_box` comparison is:

- exact-runtime `black_box`
- with `--ntx-face-response-mode interpolate_center_response`

### 7. The remaining blocker is now solver-side Radau convergence logic

The most recent exact-runtime results show that the remaining failure is no longer a nonfinite initial RHS.

#### Exact-runtime `black_box` with `interpolate_center_response`

- initial probe finite
- `finite_f0 = True`
- `finite_z0 = True`
- `finite_initial_residual = True`
- `nonfinite_stage_state = False`
- `nonfinite_stage_residual = False`
- still:
  - `accepted = False`
  - `converged = False`
  - `fail_code = 1`
  - `err_norm ≈ 3.8e-09`
  - `diverged = True`

So this is now a **Radau Newton / convergence-status rejection**, not a physics nonfinite and not an error-estimator rejection.

#### Exact-runtime full-radius `lagged_response`

- initial probe finite
- `finite_f0 = True`
- `finite_z0 = True`
- but:
  - `finite_initial_residual = False`
  - `nonfinite_stage_residual = True`
  - `nonfinite_stage_state = False`
  - `accepted = False`
  - `converged = False`
  - `fail_code = 1`
  - `err_norm ≈ 6.6e+03`

So the lagged path still has an additional stage-residual failure beyond the now-fixed axis issue.

## Updated Main Hypothesis

The current problem is no longer:

- too many exact-runtime radii
- `lax.map` itself
- or the original axis singularity

The current problem is now split into two solver-focused questions:

1. Why does exact-runtime `black_box` get rejected by Radau even when:
   - initial state is finite
   - initial residual is finite
   - `err_norm` is already tiny

2. Why does exact-runtime `lagged_response` still differ from exact-runtime `black_box` in the stage-residual failure signature?

## Updated Status: Explicit Newton Replication

The newer exact-runtime `black_box` debugging now shows a much stronger result than the older notes above.

For exact-runtime `black_box` with `interpolate_center_response`:

- initial transport probe is finite
- `rhs0` is finite
- Radau predictor `z0` is finite
- all stage states at `z0` are finite
- all stage RHS evaluations at `z0` are finite
- all stage residuals at `z0` are finite
- the benchmark-side explicit Newton replication remains finite through multiple updates:
  - Jacobian finite
  - transformed Radau real/complex block matrices finite
  - LU factors finite
  - `delta0`, `z1`, `residual1` finite
  - `delta1`, `z2`, `residual2` finite

Yet the custom Radau step still reports:

- `accepted = False`
- `converged = False`
- `nonfinite_stage_residual = True`
- `finite_initial_residual = False`
- `newton_nonfinite = True`

So the remaining blocker is now best understood as:

- a **custom Radau solver bookkeeping / control-flow bug**
- not a transport physics nonfinite
- not an NTX runtime-model failure
- and not evidence against the lagged-response approximation itself

### Confirmed bookkeeping bug already fixed

One concrete issue was identified in `_apply_radau_lean_timestep_controller(...)._reject(...)`:

- rejected-but-not-terminal attempts were still returning `step_info.fail_code = 1` or `2`
- even when the persistent step-state correctly had `fail_code = 0`

That misleading `step_info.fail_code` behavior has now been corrected.

### Current hypothesis

The likely remaining issue is inside the actual custom Radau solver path itself, for example:

1. the `while_loop` path diverges from the explicit benchmark-side Newton reconstruction
2. per-iteration bookkeeping of `finite_initial_residual`, `nonfinite_stage_residual`, or `newton_nonfinite` is stale or inconsistent
3. a later in-loop state mutation is occurring that the explicit replicated Newton probe does not yet capture

## Refactor Plan: NTSS-Style Newton Rejection

The next Radau update should make the nonlinear-step rejection logic look more like NTSS/Hairer Radau and less like a generic "slow contraction means divergence" heuristic.

### Target behavior

1. Base the contraction estimate on the **Newton correction norm** instead of the residual norm.
   - NTSS uses the update-size sequence (`dyno`, `dynold`) for `theta`
   - our old logic used a residual-ratio heuristic, which was too sensitive near convergence

2. Replace the hard `slow_contraction -> diverged` trigger with an NTSS-style predicted-defect estimate.
   - compute `theta`
   - if `theta < 0.99`, estimate the remaining Newton defect
   - reject the step and shrink `dt` only when that predicted defect is too large
   - if `theta >= 0.99`, treat that as a controlled nonlinear rejection, not a "nonfinite physics" event

3. Keep real failure classes separate.
   - nonfinite state / residual
   - residual blowup
   - predicted slow-Newton rejection
   - max-iteration exhaustion

4. Keep the benchmark-side Newton probe in sync with the solver logic.
   - the explicit probe and the actual solver must use the same `theta` and rejection formulas
   - otherwise debugging output becomes misleading

### Expected outcome

After this refactor, exact-runtime Radau runs that are finite and nearly converged should no longer be mislabeled as nonfinite/diverged just because of a late bad contraction ratio.

The most likely remaining outcomes should become:

- accepted step
- controlled timestep rejection with a sensible shrink factor
- or a genuine nonfinite / blowup classification

instead of the current mixed diagnostic state.

## Updated Status: Correct `black_box` Branch

One important benchmark bug was found after the earlier notes:

- the benchmark wrapper was not actually overriding `radau_rhs_mode`
- so some earlier "black_box" runs were still executing `lagged_response`

This has now been fixed, and the true exact-runtime `black_box` one-step benchmark is now trustworthy.

### Current trusted `black_box` result

For exact-runtime `black_box` with `interpolate_center_response`, the real in-solver Newton trace now shows:

- finite initial residual
- finite stage states
- finite stage residuals
- finite Newton updates
- no residual blowup
- no nonfinite state
- final residual still small (`~1e-6`)

The step is currently rejected only because the final NTSS-style contraction estimate reaches:

- `theta > 1`

so the rejection is now a **controlled nonlinear rejection**, not a nonfinite physics failure and not stale solver bookkeeping.

That is a much better state than before:

- the old fake nonfinite classification is gone
- the actual branch (`black_box`) is being exercised
- the remaining question is now solver policy, not solver corruption

## Next Step

The next development step should be a focused acceptance-policy experiment on top of the now-correct NTSS-style Newton rejection logic.

### Goal

Decide whether transport should:

1. remain **strictly NTSS-like**
   - reject whenever the final contraction predictor says Newton is no longer reliable

2. or allow a **relaxed near-converged acceptance**
   - if the step is finite
   - no residual blowup
   - no nonfinite state
   - and final nonlinear residual is already below a configurable relaxed tolerance

### Recommended implementation experiment

Add a configurable relaxed acceptance gate for Radau, for example:

- keep NTSS-style predictor logic unchanged
- but after Newton, if:
  - `converged == False`
  - `diverged == True` only because of slow contraction
  - `nonfinite_stage_state == False`
  - `nonfinite_stage_residual == False`
  - `residual_blowup == False`
  - `newton_nonfinite == False`
  - `final_residual_norm <= relaxed_accept_tol`

then classify the step as accepted anyway.

This should be treated as a policy toggle rather than a replacement of the NTSS logic.

### Why this is the right next step

At this point the main uncertainty is no longer:

- batching
- exact-runtime NTX correctness
- axis regularization
- lagged-response corruption
- or fake solver nonfinite bookkeeping

The remaining uncertainty is:

- whether strict NTSS-style rejection is too conservative for the transport problem when the nonlinear residual is already very small

So the next benchmark should answer a policy question, not a debugging question.

## Benchmark Set A: One Step Attempt Only

Run timing benchmarks for exactly one Radau step attempt, without mixing in repeated retries.

Compare:

1. exact-runtime `black_box`
   - `interpolate_center_response`
2. exact-runtime full-radius `lagged_response`
3. exact-runtime coarse-anchor `lagged_response`

For each case, record:

- wall time
- GPU utilization pattern
- whether the single attempt was accepted or rejected
- `converged`
- `fail_code`
- `err_norm`
- `finite_f0`
- `finite_z0`
- `finite_initial_residual`
- `nonfinite_stage_state`
- `nonfinite_stage_residual`

Important:

- benchmark one **attempt**
- not one **accepted** step

## Benchmark Set B: One Accepted Step

After Set A, benchmark the current “one accepted step” setup for:

1. exact-runtime `black_box`
2. exact-runtime full-radius `lagged_response`
3. exact-runtime coarse-anchor `lagged_response`

For each case, count:

- how many attempts were needed before the first accepted step
- how many times `build_lagged_response(...)` was called
- how many lagged RHS evaluations were made

This will show whether the apparent slowdown is mostly:

- raw NTX lagged cost
- or repeated retries / rebuilds after the current convergence issues are resolved

## Physics Comparison: Initial State

For the initial transport state and first step attempt, compare:

1. database interpolation black-box baseline
2. exact-runtime `black_box` with `interpolate_center_response`
3. exact-runtime full-radius `lagged_response`
4. exact-runtime coarse-anchor `lagged_response`

Compare:

- initial `Er`
- neoclassical fluxes
- total shared fluxes
- optionally reduced transport moments / `Lij`

Questions:

1. Are the initial `Er` values effectively identical?
2. Are the initial fluxes effectively identical?
3. If not, where do they first diverge?
4. Is the divergence already present before Radau iteration begins?

## Ambipolarity Initialization Note

Important initialization detail:

- ambipolar `Er` root initialization uses the **active run flux model** for flux/root evaluation
- entropy ranking still uses:
  - `neoclassical.entropy_model`

In the benchmark overrides this means:

1. root candidate flux evaluations:
   - active transport flux model
2. entropy-based root selection:
   - configured entropy model

The ambipolar root itself now appears acceptable provided the scan minimum is wide enough, e.g.:

- `er_ambipolar_scan_min = -52.0`

## Rejection Diagnosis: New Priority

The next highest-value diagnostics are now solver-side, not NTX-axis-side.

Expose or inspect:

- Newton iteration count used
- final residual norm
- final update norm
- convergence tolerance used
- exact logic that sets `diverged = True`
- whether the solver is simply hitting `max_iter`
- whether the divergence heuristic is too aggressive for the exact-runtime RHS scaling

New solver controls now available for these tests:

- `radau_newton_divergence_mode = "legacy" | "conservative"`
- `radau_newton_residual_norm = "raw" | "rms"`

Recommended next comparison:

1. exact-runtime `black_box` with:
   - `radau_newton_divergence_mode = "legacy"`
   - `radau_newton_residual_norm = "raw"`
2. exact-runtime `black_box` with:
   - `radau_newton_divergence_mode = "conservative"`
   - `radau_newton_residual_norm = "rms"`

This should show whether the remaining rejection is primarily caused by:

- the aggressive slow-contraction divergence heuristic
- or residual-norm scaling

## Interpretation Rule

If exact-runtime `black_box` and exact-runtime lagged-response variants all produce finite and physically reasonable initial states, then:

- there is no longer a strong case that the remaining rejection is caused by the initial exact-runtime flux calculation itself

That points more strongly to:

- Radau Newton convergence logic
- convergence scaling / tolerances
- or a lagged-stage residual construction issue

## Specific Comparisons To Do

### Timing

1. exact-runtime `black_box`, one attempt
2. exact-runtime full `51` lagged response, one attempt
3. exact-runtime coarse `7` lagged response, one attempt
4. database black-box baseline, one attempt

### Initial-State Physics

1. database black-box vs exact-runtime `black_box`
2. database black-box vs exact-runtime full lagged
3. database black-box vs exact-runtime coarse lagged
4. exact-runtime `black_box` vs exact-runtime full lagged
5. exact-runtime full lagged vs exact-runtime coarse lagged

### Solver Convergence Diagnosis

For the first single attempt, determine whether the remaining rejection comes mainly from:

1. Newton max-iteration exhaustion
2. overly aggressive divergence detection
3. a scaling issue in the convergence test
4. a lagged-stage residual problem specific to `lagged_response`

## Desired Outputs For Next Session

Prepare a small table or note with:

- approach
- one-attempt wall time
- one-attempt accepted/rejected
- `converged`
- `fail_code`
- `err_norm`
- finite/nonfinite diagnostics
- Newton iteration count
- final residual/update norms
- initial `Er` comparison
- initial flux comparison

## Open Questions

1. Why does exact-runtime `black_box` still get `fail_code = 1` even with:
   - finite `f0`
   - finite `z0`
   - finite initial residual
   - tiny `err_norm`
2. Why does exact-runtime full-radius `lagged_response` still differ from exact-runtime `black_box` in the stage-residual signature?
3. Is the remaining issue in Radau primarily:
   - Newton iteration limit
   - divergence heuristic
   - or convergence scaling?
4. After the axis regularization, are there any remaining exact-runtime physics mismatches at `t0`, or is the problem now entirely solver-side?

## Update: Monoenergetic Database vs Exact NTX Audit

The newer NTX/NEOPAX diagnostics changed the picture substantially.

### What is now ruled out

The following comparisons were checked and were consistent:

1. raw NTX scan vs raw NTX prepared/direct solve
2. raw NTX HDF5 file nodes vs raw exact prepared NTX at the same stored `rho`, `nu_v`, and field node
3. NEOPAX database load/bridge at an exact stored node

So the remaining mismatch is **not**:

- an NTX solver bug
- an NTX prepared-geometry bug
- a raw HDF5 export bug
- or a simple file-load/bridge bug at stored nodes

### New key result

A multi-node comparison of:

- database-queried `D11/D13/D33`
- vs exact prepared NTX `D11/D13/D33`

at exact stored nodes of:

- `rho`
- `nu_v`
- `Er`

still showed clear mismatches.

That means the earlier “this is only an off-node interpolation error” hypothesis was wrong.

### Most likely root cause

The strongest current suspect is the database field-axis construction itself.

Both the NEOPAX loader and the NTX bridge currently construct the stored field axis using the first-radius `Er` row for all radii:

- `Er[0, k]`

instead of the radius-local values:

- `Er[j, k]`

This appears in:

1. `NEOPAX/_database.py`
2. `NTX/src/ntx/_neopax_bridge.py`

So the database field coordinate used at query time is not actually the same per-radius field-node set present in the raw NTX scan file.

### Practical consequence

This explains why:

1. raw file node vs exact NTX can match
2. but database query vs exact NTX can still disagree even at nominal “same-node” comparisons

### Recommended next fix

Patch the field-axis construction so it uses the radius-local field values:

- use `Er[j, k]`, not `Er[0, k]`

in:

1. `NEOPAX/_database.py`
2. `NTX/src/ntx/_neopax_bridge.py`

Then rerun the node-level database-vs-exact `Dij` audit before changing any more solver logic.

## Correction / Superseding Note

This note supersedes the previous "Most likely root cause" subsection above.

### Corrected status

The remaining problem is **not yet** tied to a confirmed single-line bug.

What is still solid:

1. raw NTX scan vs raw NTX prepared/direct solve is consistent
2. raw NTX file nodes vs raw exact prepared NTX is consistent
3. NEOPAX database load/bridge at exact stored nodes is internally consistent
4. database-queried vs exact NTX `Dij` can still mismatch, including at selected nodes

### Important correction to the earlier field-axis hypothesis

The previous suspicion about using `Er[0,k]` instead of `Er[j,k]` is probably **not** the main explanation for the current NTX-built file.

For this builder:

- `Er = Er_tilde * dr_tildedr * B00`
- `dr_tildedr = 2 * Psia / (a_b^2 * B00)`

so the `B00` factor cancels and `Er` becomes effectively radius-independent for a fixed `Er_tilde` grid.

Therefore, in this dataset:

- `Er[j,k] ~= Er[0,k]`

and that specific detail is likely benign here.

### Focused objective for the next session

The next session should focus on the **database-query/evaluation path**, because the mismatch:

1. is already present at the `Dij` level
2. is not explained by raw NTX, raw file contents, or a simple bridge-at-node inconsistency
3. is not fixed by low-`Er` flooring, high-`Er` edge behavior alone, or interpolation mode choice alone

So the concrete next goal is to trace:

1. how runtime query coordinates are built from `rho`, `nu/v`, and `Er/v`
2. how the database path decides a query is "at a node"
3. how queried `D11/D13/D33` are reconstructed before comparison to exact NTX

### Recommended next action

Before changing NTX or rebuilding the scan:

1. inspect `NEOPAX/_interpolators_preprocessed.py`
2. inspect `NEOPAX/_interpolators.py`
3. inspect `NEOPAX/_database.py`
4. trace one or two scalar node cases all the way through the database-query path

Only after that should any code patch be attempted.

## New diagnostic correction

During follow-up inspection, a concrete issue was found in several benchmark scripts:

- the preprocessed database kernels expect `grid_x = r = a_b * rho`
- but some node/scalar benchmark scripts were passing `rho` from the file instead
- and some exact-side benchmark solves were also using physical `r` where `rho = sqrt(s)` was required to build the VMEC surface

So some of the previous "database vs exact mismatch at nodes" evidence was contaminated by the diagnostics themselves.

### Scripts corrected

The following benchmark scripts were corrected to distinguish:

1. `rho_surface = sqrt(s)` for VMEC/NTX surface construction
2. `r = a_b * rho_surface` for database-kernel queries

Corrected scripts:

- `examples/benchmarks/benchmark_compare_database_nodes_vs_exact.py`
- `examples/benchmarks/benchmark_scalar_database_vs_exact.py`
- `examples/benchmarks/benchmark_isolate_database_interpolation_axes.py`
- `examples/benchmarks/benchmark_compare_database_interpolation_modes.py`

### Immediate next step

Before continuing any new database-physics hypothesis, rerun the corrected node/scalar database-vs-exact audits.

Only after those reruns should we decide whether the remaining mismatch is:

1. still present at true nodes
2. mainly off-node
3. or largely a diagnostic artifact from the earlier mixed `rho` vs `r` usage

## Latest rerun status after correcting `rho` vs `r`

After rerunning the corrected diagnostics, the picture improved substantially.

### What changed

The earlier catastrophic "node mismatch" was mostly a diagnostic artifact from mixing:

- `rho_surface = sqrt(s)`
- and physical radius `r = a_b * rho_surface`

After correcting that in the benchmark scripts:

1. `D11` at nodes became much closer to exact NTX
2. scalar off-node comparisons improved significantly
3. the main remaining discrepancy is no longer a general database failure

### Current residual mismatch

The remaining differences are now concentrated mainly in:

1. `D13`
2. secondarily `D33`
3. much less in `D11`

This is true both:

- at selected exact nodes
- and in the scalar off-node audit

### Important mode-comparison result

The corrected node comparison was rerun with:

1. `generic`
2. `preprocessed_3d`

and they showed very similar residual mismatches.

So the remaining issue is **not mainly caused by the interpolation algorithm family** itself.

What this means:

- changing between `generic` and `preprocessed_3d` does not remove the residual problem
- the common issue must live in something they share

### Shared layers now most suspect

The remaining candidates are the shared database-side conventions:

1. the bridge/scaling convention for `D13`
2. the bridge/scaling convention for `D33`
3. the exact-side reconstruction used for comparison

### Focused next objective

The next session should focus on tracing `D13` through the whole chain at:

1. one exact stored node
2. one nearby off-node scalar case

For each case, compare:

1. raw file `D13`
2. bridged stored `D13`
3. database queried `D13`
4. exact NTX reconstructed `D13`

Do the same for `D33` if needed, but `D13` is now the highest-priority residual discrepancy.

## Current State: Exact Runtime Radau Retry Behavior

The most important new result is that exact realtime NTX `black_box` transport is now confirmed to be **advanceable by Radau**, even under the old `nonlinear_solver_tol = 1e-7` acceptance setting.

### What was shown

Using:

- exact realtime NTX
- `rhs_mode = black_box`
- retry-enabled benchmark run
- `max_steps = 20`
- `stop_after_accepted_steps = 1`

the solver did the following:

1. first attempted step was rejected
2. several subsequent retries were also rejected
3. after timestep reduction, a later retry converged and the step was accepted

So the previous single-attempt rejection result should be interpreted as:

- "the original trial step was too large for clean Newton convergence"

not:

- "the solver cannot advance the exact realtime NTX problem"

### Concrete accepted-step result

For the successful retry sequence at low exact-runtime resolution (`n_theta = 5`, `n_zeta = 21`, `n_xi = 33`):

- multiple rejected attempts occurred first
- the accepted step finally converged with:
  - `accepted = True`
  - `converged = True`
  - `err_norm ~ 9.4e-13`
  - `final_delta_norm ~ 7.1e-15`
  - `final_residual_norm = 0`
  - `theta_final ~ 8.4e-05`
- the accepted time advance was:
  - `final_time = 3.125e-10`

This means the current Radau logic is conservative but not fundamentally broken for exact realtime NTX.

### Updated interpretation

The main remaining issue is now best framed as:

1. **efficiency / conservatism**
   - too many rejected attempts before acceptance
2. **policy**
   - whether the first few nearly-converged attempts should be accepted earlier
3. **memory**
   - higher exact-runtime resolutions can hit GPU OOM before or during the solver step

It is no longer accurate to describe the exact realtime `black_box` case as simply "failing to converge."

## Current State: `fnewt` Logic

The custom Radau solver has now been extended so that the NTSS/Hairer-style `fnewt` predictor scale is no longer hardcoded.

### New solver toggle

The following TOML control now exists:

- `radau_newton_fnewt_mode = "tol" | "hairer"`

Accepted aliases for the Hairer-style mode also include:

- `"hairer_like"`
- `"ntss"`

### Mode meanings

1. `radau_newton_fnewt_mode = "tol"`
   - uses `nonlinear_solver_tol` directly as the predictor scale
   - this matches the previous simplified NEOPAX behavior

2. `radau_newton_fnewt_mode = "hairer"`
   - computes `fnewt` from the classic Hairer/NTSS formula using:
     - `rtol`
     - machine precision
     - Radau stage count

### Why this matters

Before this change:

- NEOPAX used:
  - a very strict direct Newton acceptance tolerance (`nonlinear_solver_tol`)
  - but a separate hardcoded predictor scale (`predictor_fnewt = 3e-2`)

That mismatch was awkward.

Now we can test three distinct policies more cleanly:

1. strict direct tolerance with `fnewt` tied to `tol`
2. Hairer-like predictor scaling from `rtol`
3. later, if desired, a relaxed near-converged acceptance override

## Current State: Newton Tolerance Modes

The previous `radau_newton_fnewt_mode` change by itself was **not** enough to make the custom solver match Hairer/NTSS Newton stopping behavior.

The reason is:

- `radau_newton_fnewt_mode` only changed how `predictor_fnewt` was computed
- it did **not** initially change the actual Newton stopping rule
- the true Newton stop condition in NEOPAX was still:
  - residual norm compared against `nonlinear_solver_tol`

This has now been corrected.

### New solver toggle

The solver now has a second explicit control:

- `radau_newton_tol_mode = "residual" | "hairer"`

Accepted aliases for the Hairer-style mode also include:

- `"hairer_like"`
- `"ntss"`

### Mode meanings

1. `radau_newton_tol_mode = "residual"`
   - keep the NEOPAX-style Newton stop condition
   - Newton continues while:
     - residual norm is above `nonlinear_solver_tol`
     - or update norm is above `nonlinear_solver_tol`

2. `radau_newton_tol_mode = "hairer"`
   - use a Hairer/NTSS-style Newton stop condition
   - Newton stops based on a **scaled Newton correction metric**
   - this metric is compared against `predictor_fnewt`
   - it does **not** use the raw residual norm as the primary Newton stopping test

### Important conceptual correction

In Hairer/NTSS Radau, the Newton solver is not controlled by:

- a raw nonlinear residual norm threshold

Instead it is controlled by:

- a tolerance-like quantity `fnewt`
- compared against a scaled correction/update metric

So the current NEOPAX Hairer mode is intended to mirror that structure, not to reinterpret `fnewt` as a residual tolerance.

## Current State: First Hairer-Mode Result

The first retry-enabled exact realtime NTX benchmark using:

- `radau_newton_tol_mode = "hairer"`
- `radau_newton_fnewt_mode = "hairer"`

showed a large behavioral change.

### Observed behavior

For the same low-resolution exact-runtime case:

- `n_theta = 5`
- `n_zeta = 21`
- `n_xi = 33`

the step was accepted immediately on the first attempt, with:

- `accepted = True`
- `converged = True`
- `newton_iter_count = 2`
- `final_time = 1.0e-8`
- `final_residual_norm ~ 2.8e-6`
- `final_delta_norm ~ 7.6e-5`
- `predictor_fnewt ~ 4.77e-4`

### Interpretation

This is **not** a contradiction.

It means:

- the raw residual was still larger than the old direct tolerance `1e-7`
- but the Hairer-style scaled correction metric was already small enough to satisfy the Newton stop rule

So the new Hairer-mode acceptance is substantially less strict than the old direct residual-based Newton stopping rule.

### Practical consequence

For this exact realtime NTX case, the solver now appears able to:

1. keep the original trial step size
2. avoid the long retry cascade
3. accept the first step under a more NTSS-like Newton stopping rule

This is currently the strongest evidence that the previous retry-heavy behavior was at least partly caused by using a NEOPAX-specific residual stopping criterion rather than a Hairer-style correction stopping criterion.

## Update: Slow-Theta Must Not Override Satisfied Hairer Convergence

While reviewing the custom Radau Hairer-mode logic, an additional acceptance-policy bug became clear.

The solver already computes the intended Hairer/NTSS-style Newton stop metric:

- a scaled correction metric (`newton_metric`)
- compared against `predictor_fnewt`

However, the previous implementation still allowed:

- `slow_contraction = True`

to force:

- `diverged = True`

even on an iteration where:

- `newton_metric <= predictor_fnewt`

This is stricter than the intended Hairer/NTSS behavior.

The correct interpretation is:

- `theta` / slow contraction is a predictor of whether continuing Newton is worthwhile
- but it should not overrule an iterate that has already satisfied the Newton correction criterion

So the Radau logic is now being updated so that:

- `slow_contraction` remains a diagnostic / retry predictor
- but it only forces rejection when `newton_metric > predictor_fnewt`

This should make the behavior more faithful to the intended Hairer/NTSS acceptance semantics and should reduce spurious rejected attempts near convergence.

## Update: Low-Overhead NTSS-Like Controller Hysteresis

The accepted-step controller was still using a very lean one-step error-growth rule, which made it prone to:

- regrowing `dt` too aggressively immediately after difficult regions
- repeated grow-fail-shrink oscillation
- spending too long near zones that require sustained small timesteps

To move this closer to the previously discussed NTSS/Hairer direction without adding JAX-unfriendly complexity, the controller has now been extended with scalar-history logic only:

- mild PI-style history using the previous accepted-step error
- post-rejection regrowth cooldown
- stronger shrinkage after repeated rejected retries
- accepted-step growth caps based on Newton difficulty (`theta`, iteration count, slow contraction)
- an easy-step streak that only allows aggressive growth after several clearly comfortable accepted steps

This keeps the implementation efficient for JAX because it adds only a few scalar fields to the Radau step state and uses `jnp.where(...)`-style controller algebra rather than dynamic branching or large history buffers.

The controller has now been pushed one step closer to the missing Hairer/NTSS ingredient by making the accepted-step growth formula itself more explicitly PI-like:

- current accepted error
- previous accepted error
- and a light step-ratio damping term that suppresses immediate further growth after a recent accepted-step expansion

This is still not a full classic Gustafsson controller, but it is closer in spirit than the earlier pure one-step error rule and remains cheap to trace/JIT in JAX.

An additional refinement has now been applied because the first hysteresis pass made `dt` recovery too sticky after difficult regions.

The post-rejection recovery is now less fixed and more responsive:

- shorter regrowth cooldown
- faster cooldown decay after genuinely easy accepted steps
- a larger temporary recovery cap when the next accepted step is already clearly easy again
- lighter growth damping after recent accepted-step expansion

So the controller still avoids the old overshoot/retry pattern, but it should now let `dt` recover more naturally once the stiff region has really passed.

## Update: Preserve Current Controller And Add Explicit Gustafsson Option

To keep the recent solver behavior reproducible while still moving toward a more state-of-the-art adaptive Radau controller, the timestep policy is now being split into explicit modes:

- `radau_controller_mode = "current"`
  - preserves the existing NEOPAX controller logic
  - includes the current scalar-history hysteresis, Newton-quality growth caps, and recent-retry safeguards

- `radau_controller_mode = "current_legacy"`
  - preserves the earlier Newton-quality-aware timestep policy from before the later accepted-step tuning
  - keeps:
    - Newton-quality influence from `theta`, `slow_contraction`, and iteration count
    - post-rejection cooldown
    - easy-step streak / release logic
  - but drops the later extra accepted-step damping around recent accepted-step expansions
  - intended as the closest comparison point to the earlier behavior where easy Newton solves influenced `dt` more directly

- `radau_controller_mode = "gustafsson"`
  - uses a cleaner predictive accepted-step update
  - combines a PI-style term with a light Gustafsson-like predictive proposal based on previous accepted-step history
  - keeps only a light Newton-quality cap and a lighter post-rejection regrowth limiter

- `radau_controller_mode = "hairer_lean"`
  - keeps the newer Hairer-side Newton convergence logic
  - but uses a much simpler accepted-step update closer to the older/original NEOPAX behavior
  - accepted-step growth is driven mainly by a one-step error controller:
    - `growth ~ safety_factor * err_norm^(-alpha)`
  - disables the newer cooldown / streak / Newton-quality regrowth heuristics
  - intended for direct comparison when the more elaborate controller modes become too sticky

This keeps the current path available for direct comparison and makes future benchmarking much easier:

- current controller behavior remains available unchanged
- the more modern predictive controller can now be tested independently

## Update: Reuse Lagged NTX Response Across Rejected Retries

Another inefficiency was identified in the lagged-response Radau path:

- the lagged NTX response is built from the base state at the start of an attempted timestep
- if that timestep attempt is rejected, the retry starts from the same base state
- so rebuilding the lagged response on every rejected retry is unnecessary

The Radau step state now caches exactly one lagged-response object for the current base state:

- reused across rejected retries from the same `t_n, y_n`
- invalidated after an accepted step advances the base state

This keeps the memory footprint bounded while removing repeated lagged-response rebuild cost in retry-heavy regions.

## Update: Preserve Current Predictor And Add Stronger Collocation Option

The Radau stage predictor is also now being split into explicit modes so that predictor experiments can be benchmarked independently of controller changes:

- `radau_predictor_mode = "current"`
  - preserves the existing stage-history blend

- `radau_predictor_mode = "collocation"`
  - uses the previous accepted-step stage history more aggressively
  - applies a collocation-style correction so the predicted stages are shifted to match the current step's fresh base slope `f(t_n, y_n)`

This is intended as the next low-risk step toward a more mature Hairer/NTSS-like predictor while keeping the current predictor available as a stable comparison baseline.

## Current Implemented Timestep / Retry Upgrades

At this point the custom Radau path includes the following timestep-control and retry-related changes:

1. Hairer/NTSS-style Newton stop rule

- `radau_newton_tol_mode = "hairer"`
- Newton convergence is controlled by a scaled correction metric (`newton_metric`)
- this metric is compared against `predictor_fnewt`
- the raw nonlinear residual is no longer the primary Newton stop criterion in Hairer mode

2. Slow-theta rejection fix

- `slow_contraction` no longer overrides an already-satisfied Hairer Newton convergence test
- a step is not rejected just because the contraction predictor is pessimistic when `newton_metric <= predictor_fnewt`

3. Improved attempt diagnostics

- the per-attempt Radau debug line now prints:
  - `t_start`
  - `dt_try`
  - `accepted`
  - `failed`
  - `fail_code`
  - `converged`
  - `err_norm`
- this makes it much clearer when a retry happened because:
  - Newton failed
  - versus Newton converged but the embedded timestep error estimate rejected the step

4. Current-controller scalar-history hysteresis

- mild PI-like accepted-step growth update
- post-rejection regrowth moderation
- stronger shrinkage after repeated rejected retries
- Newton-quality growth caps
- easy-step streak logic for regrowth release

This remains available as:

- `radau_controller_mode = "current"`

5. Optional predictive controller path

- a separate more modern controller option now exists:
  - `radau_controller_mode = "gustafsson"`
- this uses:
  - PI-style history
  - a light predictive / Gustafsson-style proposal
  - lighter post-rejection moderation
  - Newton-quality growth caps as safety only

5b. Optional lean/original-style controller path

- a separate simpler controller option now exists:
  - `radau_controller_mode = "hairer_lean"`
- this preserves:
  - Hairer Newton stopping
  - slow-theta non-override fix
- while reverting the accepted-step timestep evolution closer to the earlier simple NEOPAX rule
- this is useful when:
  - the Newton-side Hairer changes are clearly helping
  - but the newer accepted-step controller heuristics are making `dt` recover too slowly or shrink after easy accepted steps

5c. Optional earlier Newton-quality-aware controller path

- a separate intermediate controller option now exists:
  - `radau_controller_mode = "current_legacy"`
- this is meant to approximate the earlier phase where:
  - Hairer-style Newton acceptance was already in place
  - slow/theta was being used as a timestep-quality signal
  - but the later accepted-step tuning had not yet shifted the balance as much

6. Optional stronger stage predictor

- `radau_predictor_mode = "current"`
  - preserves the existing stage-history blend

- `radau_predictor_mode = "collocation"`
  - uses previous accepted-stage history more aggressively
  - shifts that predictor using the fresh base slope at the current step start

7. Reuse of lagged NTX response across rejected retries

- the lagged-response Radau path now caches exactly one lagged NTX response for the current base state
- if a timestep attempt is rejected and retried from the same `t_n, y_n`, that cached lagged response is reused
- after an accepted step advances the state, the cached lagged response is invalidated

This removes unnecessary lagged-response rebuild cost in retry-heavy regions while keeping memory bounded to one cached lagged response.

8. Optional global lagged-response reuse across accepted steps

- a new toggle now exists:
  - `lagged_response_reuse_mode = "retry_only" | "global_state_drift"`
- default remains:
  - `retry_only`
- the new optional mode:
  - `global_state_drift`
  keeps the cached lagged response after an accepted step when the accepted state is still close to the cached lagged-response reference state

The current global reuse criterion is:

- compute a normalized drift metric on the packed flat transport state:
  - `|| y_new - y_ref || / (atol + rtol * scale(y_ref, y_new))`
- reuse the cached lagged response if that global metric is `<= 1`

Associated knobs:

- `lagged_response_reuse_rtol`
- `lagged_response_reuse_atol`

This is intentionally simpler than a flux-based or per-radius policy:

- it uses state drift, not flux drift
- it reuses or rebuilds the whole lagged response as one object
- it is the lowest-risk extension of the already-implemented retry-only cache

## Remaining State-Of-The-Art Controller / Predictor Upgrades

The current implementation is meaningfully closer to a mature adaptive Radau solver, but it is still not the final state-of-the-art controller path.

The most relevant remaining upgrades are:

1. A cleaner, more faithful Gustafsson accepted-step controller

- reduce reliance on handcrafted cooldown/streak heuristics
- let accepted-step growth be driven more directly by a polished history-aware formula

2. Stronger collocation / extrapolated stage prediction

- use previous accepted-stage history more faithfully
- target fewer Newton iterations and larger accepted `dt`

3. Better reuse across retries

- especially reuse of transformed-block factorizations / decompositions when:
  - the Jacobian is still valid
  - `dt` changes only moderately

4. Smoother Newton-quality weighting in timestep growth

- replace some current hard caps / thresholds with more continuous quality weighting based on:
  - `theta_final`
  - `newton_iter_count`
  - recent contraction difficulty

5. Possible future comparison against another stiff family

- especially variable-order BDF
- not because Radau is wrong, but to learn whether the remaining step-count limit is mostly:
  - controller/predictor quality
  - or a deeper method-family issue for this transport problem

Current practical recommendation:

- keep the stiff transient solvability of Radau
- benchmark:
  - `radau_controller_mode = "hairer_lean"` vs `radau_controller_mode = "current"`
  - `radau_controller_mode = "current"` vs `radau_controller_mode = "gustafsson"`
  - `radau_predictor_mode = "current"` vs `radau_predictor_mode = "collocation"`
- then decide whether the next gain should come from:
  - controller refinement
  - stronger predictor logic
  - or retry-side factorization reuse

## Important Diagnostic Caveat

The benchmark-side `initial_probe.radau.while_loop` block is now likely out of sync with the real solver after the Newton tolerance refactor.

In the first Hairer-mode run:

- the real solver trace showed:
  - converged in 2 Newton iterations
  - accepted first attempt
- but the benchmark-side explicit Newton probe still reported the old longer rejection-style trace

So at the moment:

- the real `[radau-solver] ...` trace is the trustworthy source for actual solver behavior
- the benchmark-side explicit Newton replication needs to be updated to use the same Newton stop rule and tolerance mode as the real solver

Until that probe is synchronized, it should not be used as the authoritative source when `radau_newton_tol_mode = "hairer"`.

## Current State: Memory Limits

The higher-resolution exact realtime NTX tests revealed two distinct memory pressure points.

### 1. Ambipolar post-init debug diagnostic

With `debug_initial_finiteness = true`, the code performs an extra ambipolar scan diagnostic **after** successful root finding.

This can OOM independently of the actual ambipolar solve.

So:

- successful `best_roots` computation does **not** guarantee the debug scan will fit

### 2. Actual Radau transport solve

At higher exact-runtime resolutions, the solver can also OOM during:

- `solver.solve(...)`
- inside the custom Radau saved loop

This is a real transport-step memory issue, separate from ambipolarity.

### Practical consequence

For exact realtime NTX there are now two partially separate resource questions:

1. can ambipolar initialization fit?
2. can the actual Radau Newton step fit?

This is why low-resolution exact-runtime runs are currently the safest place to compare acceptance-policy changes.

## Possible Paths

The next development work can proceed along several reasonable paths.

### Path 1: Stay strict, accept retries

Keep:

- strict `nonlinear_solver_tol`
- current slow-contraction rejection behavior
- no relaxed acceptance override

Interpretation:

- exact realtime NTX is valid
- the solver is just conservative and may need several retries

Cost:

- expensive wall time
- many rejected attempts
- poor practicality at higher resolution

### Path 2: Compare `fnewt` policies directly

Run the same exact-runtime retry-enabled benchmark with:

1. `radau_newton_fnewt_mode = "tol"`
2. `radau_newton_fnewt_mode = "hairer"`

Goal:

- determine whether Hairer-style predictor scaling reduces unnecessary retries
- without changing the actual Newton convergence tolerance

This is the cleanest next apples-to-apples solver-policy comparison.

### Path 2b: Compare Newton tolerance modes directly

Now that the solver also supports a true Hairer-style Newton stop rule, the more decisive comparison is:

1. `radau_newton_tol_mode = "residual"`
2. `radau_newton_tol_mode = "hairer"`

with `radau_newton_fnewt_mode` chosen consistently in each case.

Goal:

- determine whether the retry cascade is mainly caused by the old residual-based Newton stopping rule
- or whether additional acceptance-policy changes are still needed even after switching to Hairer-style correction-based stopping

### Path 3: Add relaxed near-converged acceptance

Keep the predictor logic, but allow acceptance when:

- all states are finite
- no blowup occurred
- no nonfinite stage residual occurred
- final residual is already sufficiently small

Interpretation:

- more pragmatic transport policy
- less faithful to strict Hairer/NTSS rejection semantics

This path should only be evaluated after Path 2, otherwise the source of improvement becomes ambiguous.

### Path 4: Focus on memory rather than acceptance policy

Use lower exact-runtime resolution and/or memory-saving settings to stabilize experimentation:

- reduce `ntx_exact_n_theta/n_zeta/n_xi`
- use ambipolar scan batching
- consider rematerialization / batch-mode settings
- avoid heavy debug diagnostics during high-resolution runs

This path is required if the main goal becomes realistic production-resolution transport rather than solver-policy diagnosis.

## Recommended Near-Term Order

The most useful order for the next session is:

1. keep a low-resolution exact realtime NTX case that is known to fit
2. compare:
   - `radau_newton_fnewt_mode = "tol"`
   - `radau_newton_fnewt_mode = "hairer"`
3. compare:
   - `radau_newton_tol_mode = "residual"`
   - `radau_newton_tol_mode = "hairer"`
4. measure:
   - number of rejected attempts
   - accepted-step wall time
   - accepted `dt`
   - final accepted Newton iteration count
   - final accepted raw residual norm
5. synchronize the benchmark-side explicit Newton probe with the real solver logic
6. only then decide whether a relaxed near-converged acceptance gate is still needed

This keeps the next comparisons interpretable and avoids mixing:

- acceptance-policy changes
- predictor-scale changes
- and memory-driven resolution changes

## Possible Future Lagged-Response Updates

### Reuse cached lagged response across accepted steps

Possible next optimization:

- keep the cached lagged NTX response even after an accepted step
- rebuild it only when the accepted state has drifted enough that the frozen lagged model is no longer trustworthy

Why this may help:

- `build_lagged_response(state)` depends on the transport state, not on `dt`
- if successive accepted states are close, rebuilding the full lagged NTX response every step may be unnecessary
- this is the natural extension of the already-implemented reuse across rejected retries

Suggested control approach:

- use a dedicated lagged-response reuse tolerance, separate from the ODE timestep `rtol`
- monitor relative drift in important quantities such as:
  - `Er`
  - pressure / temperature
  - possibly density
  - ideally collisionality-related inputs such as `log_nu_star`
- force rebuild if solver quality degrades, e.g.:
  - more rejected attempts
  - larger `theta`
  - larger Newton iteration count

This is conceptually similar to:

- frozen-Jacobian / chord-method reuse
- quasi-Newton refresh logic
- trust-region style model-validity checks

### Whole-response reuse is the safest next step

With the current NTX call structure, the most practical next update is:

- cache one lagged response for the current reference state
- reuse it across accepted steps while a drift criterion is satisfied
- rebuild the whole response once that criterion fails

Why this is attractive:

- it fits the current `build_lagged_response(state)` and `evaluate_with_lagged_response(state, lagged_response)` API well
- it keeps implementation and memory overhead bounded
- it is much lower risk than partially rebuilding only selected radii

### Per-radius or per-anchor selective refresh

Possible later refinement:

- refresh only the radii or anchors whose local state drift exceeds a threshold
- keep the rest of the cached lagged response unchanged

Why this is harder:

- the current lagged-response API builds a coherent full response object for all radii / anchors
- there is no existing public interface for "rebuild only this subset of radii"
- selective refresh would require masked replacement logic and more careful cache bookkeeping

Still, it appears feasible because:

- the full-radius branch is already built radius-by-radius internally
- the coarse/interpolated branch is already built anchor-by-anchor internally

Best implementation order:

1. whole-response reuse across accepted steps
2. optional per-radius / per-anchor masked refresh only if benchmarks justify the extra complexity

## Update: Radau / Lagged-Response Design Contract

The intended contract has now been clarified and should guide the next refactor steps:

- the custom Radau solver machinery should behave the same for:
  - database / normal transport RHS
  - exact-runtime lagged-response transport RHS
- the only default difference should be:
  - how selected flux components are evaluated

In practice this means:

- same Newton / predictor / cache / LU / timestep-controller path
- same Jacobian reuse policy
- same retry behavior
- only the selected flux model components should switch to their lagged-response evaluation path

For the current NTX case this means:

- neoclassical fluxes may be supplied by the NTX lagged response
- turbulence and classical should remain on their normal paths unless they also provide lagged responses
- the full transport RHS remains the full transport RHS, but built from the same component-wise flux selection logic

This is important because the older implementation had drifted away from that contract:

- lagged-response mode was not just changing flux evaluation
- it was also changing the Newton linearization / reuse behavior

## Update: Radau Refactor Started

The first pass of the custom-Radau refactor has now started in `_transport_solvers.py`.

Current changes already made:

1. `rhs_mode = "lagged_response"` now uses the selected lagged flux/RHS path consistently for:
   - `f0`
   - stage residual evaluation
   - Newton Jacobian construction at the current state

2. the custom Radau step no longer automatically disables the standard Jacobian reuse path just because:
   - `use_transport_lagged_response = True`

3. the initial stage-history seed (`prev_stages`) is now initialized from the same selected RHS path, instead of always from the direct non-lagged RHS

4. attempt diagnostics now include:
   - `jacobian_reused=True|False`
   - alongside the existing:
     - `lagged_reused=True|False`

The purpose of this first pass is to bring lagged-response Radau back toward:

- same solver structure as the normal/database path
- different flux source only

## What Is Still Missing

The refactor is not finished yet. The following items still need to be checked and/or completed:

1. Confirm that the lagged-response path now really matches the normal/database Radau path structurally.

What still needs verification from runs:

- Jacobian reuse frequency
- LU reuse behavior
- accepted/rejected-step pattern
- whether the lagged path is still paying a hidden solver-side cost not present in the normal path

2. Verify whether the remaining large walltime is dominated by:

- the NTX lagged-response build itself
- the lagged-response stage evaluations
- or some remaining solver-side mismatch

This now needs to be measured after the Radau-path refactor, not before it.

3. Decide whether the Newton Jacobian source in lagged mode should remain:

- the full selected transport RHS Jacobian

or whether a separate advanced option should later exist for:

- a lagged Jacobian
- a frozen Jacobian
- or a more aggressive reuse policy

Important:

- this should be a separate future option
- it should not be implicitly bundled into `rhs_mode = "lagged_response"`

4. Check whether any other solver backends still have the old lagged/non-lagged structural mismatch.

The current work has focused on the custom Radau path only.

5. Update benchmark interpretation after reruns.

The next benchmark reruns should explicitly record:

- `lagged_reused`
- `jacobian_reused`
- walltime per attempt / per accepted step
- whether accepted-step lagged-response reuse is active

Only after that should we decide whether the dominant remaining issue is:

- lagged-response build cost
- solver-side Jacobian cost
- or insufficient lagged-response reuse across accepted steps

## Update: Radau / Lagged-Response Structural Fix Appears To Help

The latest rerun of the exact-runtime NTX lagged-response Radau transport case now looks much healthier.

Observed behavior:

- steps are accepted immediately
- retry behavior is no longer pathological
- walltime per attempt dropped substantially after the first attempt
- the run now completes in a reasonable total time

Representative pattern from the current run:

- first attempt still carries noticeable startup cost
- subsequent accepted attempts are much cheaper
- the custom Radau path no longer shows the previous catastrophic lagged-response slowdown

This strongly suggests that the recent structural Radau-path refactor was important:

- lagged-response mode now behaves much more like the normal/database Radau path
- the lagged flux evaluation is no longer dragging along the old solver-side mismatch in the same way

### Important interpretation

This does **not** mean the exact-runtime NTX lagged path is using a database fallback.

The current successful benchmark still uses:

- `flux_model = "ntx_exact_lij_runtime"`

with no database-backed neoclassical model selected.

So the improved timing should be interpreted as:

- the Radau/lagged-response structural mismatch was a real issue
- fixing that mismatch improved the runtime substantially

## Update: Main Remaining Issue Is Now Timestep Policy

With retry behavior and lagged-response timing looking much better, the next main issue is no longer the lagged-response structural path itself.

The current main concern is now:

- the timestep policy appears to take too many accepted steps compared with earlier benchmarks

In the latest successful runs:

- steps are easy
- Newton convergence is comfortable
- but the accepted-step pattern still needs to be compared against earlier simpler-controller benchmarks

## Specific Follow-Up Note: Reconstruct The Earlier Good Timestep Policy

The next timestep-policy investigation should focus especially on identifying which earlier controller/policy combination was active in the benchmark phase where:

- the solver avoided the false non-convergence / divergence classification
- the problem was actually converged
- `theta` was slow or pessimistic
- and yet the overall step count was lower than in the current runs

This needs special emphasis because the current situation is now:

- retry behavior seems good
- timing behavior seems much better
- but the number of accepted steps is still too large compared with the earlier simpler-controller benchmarks

### Recommended next policy question

Reconstruct and compare the timestep policy that was active when:

1. Hairer-style Newton acceptance had already prevented the old false non-convergence flagging
2. `theta` was being treated as a caution signal rather than a hard rejection override
3. the accepted-step controller was still simpler and apparently produced fewer total steps

This should be treated as a concrete benchmark question, not just a code-history question.

### Practical benchmark note

The next comparison should explicitly record:

- controller mode
- predictor mode
- accepted-step count
- rejected-attempt count
- average accepted `dt`
- total runtime

with special attention to which earlier controller/predictor combination best reproduces the lower-step behavior seen in the previous simpler benchmarks.

## Update: Latest Benchmark Outcome

The latest transport benchmarks changed the picture again in an important way.

### 1. The Radau / lagged-response structural refactor helped substantially

After the recent custom-Radau refactor, the exact-runtime NTX lagged-response case now behaves much better:

- immediate accepted steps are possible
- retry behavior is no longer pathological
- walltime per attempt is dramatically lower than in the earlier broken lagged-response runs
- the run no longer shows the previous catastrophic solver-side slowdown

This is strong evidence that the earlier lagged-response slowdown was not just "raw NTX cost", but was strongly amplified by a solver-path mismatch in the custom Radau implementation.

### 2. Exact-runtime NTX is still being used

The improved timing should **not** be interpreted as a database fallback.

The successful lagged-response runs still use:

- `flux_model = "ntx_exact_lij_runtime"`

with no database-backed neoclassical model selected.

So the current interpretation is:

- exact-runtime NTX lagged-response path is active
- the Radau-path structural fix removed a large artificial penalty

### 3. The main remaining issue is now timestep-policy stagnation

The latest database benchmark reruns show that the current accepted-step controller can still stall the timestep badly even when:

- Newton is converged
- the step is accepted
- `err_norm` is extremely small
- there are no nonfinite states or residuals

Representative pattern:

- `converged = True`
- `err_norm ~ 1e-13`
- `newton_iter_count = 7`
- `theta ~ 2.4e-02`
- `accepted = True`
- but:
  - `growth = 1.0`
  - `next_dt = dt_try`

and this can persist for thousands of accepted steps, causing the timestep to freeze near:

- `dt ~ 1.48e-08`

### 4. Why the timestep is freezing

The current controller logic still classifies an accepted step as "difficult" if:

- `newton_iter_count >= 6`

even when the step is clearly converged and the error estimate is tiny.

Then, in the current non-lean controller modes, difficult accepted steps can be forced to:

- `growth = 1.0`

instead of being allowed to regrow.

So the current bottleneck is no longer:

- false non-convergence
- lagged-response rebuild cost
- or solver-path corruption

It is now:

- overly conservative accepted-step growth policy for difficult-but-converged Newton solves

### 5. Important comparison result

This behavior also explains why the older/original timestep/Newton policy seemed to use fewer steps:

- the older policy was effectively more permissive
- it did not penalize accepted 6-7 iteration Newton solves as strongly
- so it allowed larger regrowth or at least avoided long flat `growth = 1.0` plateaus

This is now the main reason to revisit the older simpler controller logic.

## Updated Priority

The current priority is now:

1. keep the Radau / lagged-response structural fix
2. keep the Hairer-side Newton convergence improvements
3. revisit the accepted-step timestep policy

with special focus on reconstructing which earlier controller behavior gave:

- correct convergence classification
- but fewer accepted steps

## Immediate Recommended Comparison

The next comparison should focus on:

- `radau_controller_mode = "current_legacy"`
- `radau_controller_mode = "hairer_lean"`
- and the earlier simpler benchmark behavior

while keeping:

- `radau_newton_tol_mode = "hairer"`
- `radau_newton_fnewt_mode = "hairer"`

fixed

The question is no longer whether the solver can converge safely.

The question is now:

- which accepted-step policy avoids freezing `dt` on difficult-but-converged steps while preserving the good convergence classification.

## Update: Confirmed Controller / Predictor Comparison Results

For the exact-runtime NTX lagged-response transport case with:

- `radau_rhs_mode = "lagged_response"`
- `lagged_response_reuse_mode = "global_state_drift"`
- `lagged_response_reuse_rtol = 1.0e-2`
- `lagged_response_reuse_atol = 1.0e-8`
- `ntx_exact_face_response_mode = "interpolate_center_response"`
- `ntx_exact_response_anchor_count = 48`
- `t_final = 1.0e-2`

the following controller / predictor combinations have now been explicitly observed:

### Confirmed results

1. `hairer_lean + collocation`

- `radau_newton_tol_mode = "hairer"`
- `radau_newton_fnewt_mode = "hairer"`
- `radau_controller_mode = "hairer_lean"`
- `radau_predictor_mode = "collocation"`
- result:
  - `n_steps = 139`

2. `hairer_lean + current`

- `radau_newton_tol_mode = "hairer"`
- `radau_newton_fnewt_mode = "hairer"`
- `radau_controller_mode = "hairer_lean"`
- `radau_predictor_mode = "current"`
- result:
  - `n_steps = 147`

3. `current_legacy + current`

- `radau_newton_tol_mode = "hairer"`
- `radau_newton_fnewt_mode = "hairer"`
- `radau_controller_mode = "current_legacy"`
- `radau_predictor_mode = "current"`
- result:
  - `n_steps = 253`

4. `current_legacy + collocation`

- `radau_newton_tol_mode = "hairer"`
- `radau_newton_fnewt_mode = "hairer"`
- `radau_controller_mode = "current_legacy"`
- `radau_predictor_mode = "collocation"`
- result:
  - `n_steps = 226`

5. `current + collocation`

- `radau_newton_tol_mode = "hairer"`
- `radau_newton_fnewt_mode = "hairer"`
- `radau_controller_mode = "current"`
- `radau_predictor_mode = "collocation"`
- result:
  - `n_steps = 238`

6. `current + current`

- `radau_newton_tol_mode = "hairer"`
- `radau_newton_fnewt_mode = "hairer"`
- `radau_controller_mode = "current"`
- `radau_predictor_mode = "current"`
- result:
  - `n_steps = 240`

7. `gustafsson + current`

- `radau_newton_tol_mode = "hairer"`
- `radau_newton_fnewt_mode = "hairer"`
- `radau_controller_mode = "gustafsson"`
- `radau_predictor_mode = "current"`
- result:
  - `n_steps = 271`

8. `gustafsson + collocation`

- `radau_newton_tol_mode = "hairer"`
- `radau_newton_fnewt_mode = "hairer"`
- `radau_controller_mode = "gustafsson"`
- `radau_predictor_mode = "collocation"`
- result:
  - `n_steps = 275`

### Current ranking

Best to worst so far, for minimizing accepted timesteps:

1. `hairer_lean + collocation` -> `139`
2. `hairer_lean + current` -> `147`
3. `current_legacy + collocation` -> `226`
4. `current + collocation` -> `238`
5. `current + current` -> `240`
6. `current_legacy + current` -> `253`
7. `gustafsson + current` -> `271`
8. `gustafsson + collocation` -> `275`

### Current practical conclusion

For this exact-runtime NTX lagged-response benchmark family:

- the accepted-step controller is the dominant lever
- `hairer_lean` is clearly better than `current_legacy` and `gustafsson` for minimizing accepted steps
- within the better controller family, `collocation` gives a modest additional gain

So the current best-known configuration is:

- `radau_newton_tol_mode = "hairer"`
- `radau_newton_fnewt_mode = "hairer"`
- `radau_controller_mode = "hairer_lean"`
- `radau_predictor_mode = "collocation"`
- `lagged_response_reuse_mode = "global_state_drift"`
- `lagged_response_reuse_rtol = 1.0e-2`
- `lagged_response_reuse_atol = 1.0e-8`

## Update: Experimental Predictor Upgrades Added

Two new experimental Radau predictor modes have now been added as isolated extensions to the stage-predictor path, without changing the controller structure, Newton termination logic, or lagged-response plumbing.

### New predictor modes

1. `radau_predictor_mode = "extrapolated_collocation"`

- starts from the existing `collocation` predictor
- adds a bounded step-ratio extrapolation based on the previous accepted stage pattern
- is intended to be a higher-order warm start than plain `collocation`, while staying purely algebraic and cheap

2. `radau_predictor_mode = "jacobian_linearized"`

- starts from the `extrapolated_collocation` predictor
- applies one linearized collocation-style correction using the cached Jacobian when available
- falls back automatically to `extrapolated_collocation` when no reusable Jacobian cache is available

### Implementation notes

- the additions were kept local to `_make_radau_stage_predictor(...)`
- the new modes only consume already-available inputs:
  - previous accepted stages
  - previous / current timestep
  - Radau collocation matrix
  - cached Jacobian when available
- this keeps the change minimally invasive and friendly to JAX tracing and differentiability
- no extra Python-side state or imperative bookkeeping was introduced
- the Jacobian-informed mode is intentionally cache-aware rather than forcing a new Jacobian build just for prediction

### Current intent

These predictor modes are experimental next-step candidates for reducing Newton work and possibly accepted timestep count further, while preserving the current Radau / lagged-response solver structure that now behaves well.

## Update: Experimental Predictor Results

The first two experimental predictor upgrades did not beat plain `collocation` in the current exact-runtime NTX lagged-response benchmark family.

### Observed outcomes

1. `hairer_lean + extrapolated_collocation`

- `radau_predictor_mode = "extrapolated_collocation"`
- result:
  - `n_steps = 150`

2. `hairer_lean + jacobian_linearized`

- `radau_predictor_mode = "jacobian_linearized"`
- result:
  - `n_steps = 149`

### Interpretation

- both experimental modes were slightly worse than plain `collocation`
- `extrapolated_collocation` likely overused stage-history extrapolation when the local collocation correction was already sufficient
- `jacobian_linearized` did not reduce step count enough to compensate for its extra per-predictor work

So the current practical conclusion remains:

- `collocation` is still the best predictor seen so far for this benchmark family

## Update: New Gated Predictor Option

A new experimental predictor mode has been added:

- `radau_predictor_mode = "dt_ratio_gated_collocation"`

Intent:

- start from the existing `collocation` predictor
- only blend toward the more aggressive `extrapolated_collocation` predictor when `dt_new / dt_old` stays close to `1`
- fall back smoothly toward plain `collocation` when the step-ratio change is larger

Implementation notes:

- the gate is smooth and algebraic, based on the bounded timestep ratio
- this keeps the mode JAX-friendly and differentiability-friendly
- the change remains local to the predictor path and does not modify controller or Newton logic

## Update: Restored Predictor Baseline And New Low-Risk Experiments

The temporary clipped shared stage-ratio scaling in the predictor core was the likely source of the distorted `current` / `collocation` reordering. After restoring the original raw stage-ratio scaling for the existing modes, the expected ordering came back:

- `collocation` -> `139` steps
- `current` -> `147` steps
- `dt_ratio_gated_collocation` -> `151` steps

This strongly suggests that future predictor experiments should preserve the existing `collocation` structure and only add small, local modifications.

### Newly implemented experimental predictor modes

1. `radau_predictor_mode = "collocation_correction_gated"`

- keeps the restored raw stage-history scaling
- keeps the same collocation structure
- only gates the fresh correction term using the mismatch between:
  - previous first-stage slope proxy
  - current base slope `f0`

2. `radau_predictor_mode = "newton_quality_gated_collocation"`

- keeps the restored raw stage-history scaling
- keeps the same collocation structure
- gates the fresh correction term using previous-step Newton quality:
  - previous `theta_final`
  - previous `newton_iter_count`

### Design intent

- do not modify the live `current` or `collocation` behavior
- do not reintroduce aggressive global extrapolation
- keep the experiments cheap, JAX-friendly, and differentiability-friendly
- only modulate the collocation correction when history looks less trustworthy

## Update: New Predictor Results After Restoring Raw Scaling

With the shared raw stage-ratio scaling restored, the following additional predictor results were observed for the same exact-runtime NTX lagged-response benchmark family:

1. `collocation_correction_gated`

- `radau_predictor_mode = "collocation_correction_gated"`
- result:
  - `n_steps = 154`
  - `synchronized_elapsed_s = 279.434`

2. `newton_quality_gated_collocation`

- `radau_predictor_mode = "newton_quality_gated_collocation"`
- result:
  - `n_steps = 140`
  - `synchronized_elapsed_s = 245.212`

### Interpretation

- `collocation_correction_gated` is not promising so far
- `newton_quality_gated_collocation` did not beat plain `collocation` on accepted-step count
- however, `newton_quality_gated_collocation` is notable because it achieved:
  - one extra accepted step compared with plain `collocation`
  - but lower total synchronized solver time

For reference, the restored plain `collocation` baseline is:

- `n_steps = 139`
- `synchronized_elapsed_s = 256.126`

So at the moment:

- `collocation` remains the best predictor for minimizing accepted steps
- `newton_quality_gated_collocation` is the strongest experimental variant so far if walltime, not just step count, is considered

## Next High-Leverage Directions For Large Step Reduction

Given the latest results, the most promising next gains are now more likely to come from timestep-control and error-estimation policy than from further predictor micro-variants.

### Guardrail

Any future experiment in this area should preserve the current best-known baseline unchanged:

- `radau_newton_tol_mode = "hairer"`
- `radau_newton_fnewt_mode = "hairer"`
- `radau_controller_mode = "hairer_lean"`
- `radau_predictor_mode = "collocation"`

Meaning:

- do not modify the live behavior of `hairer_lean + collocation`
- only add new logic under explicit new opt-in modes or flags
- keep additions JAX-efficient, tracing-friendly, and differentiability-friendly

### 1. More Aggressive Hairer-Lean Variant For Clearly Healthy Accepted Steps

This is likely the strongest controller-side direction.

Idea:

- keep the basic `hairer_lean` structure
- but introduce a new opt-in controller mode that allows stronger regrowth when a step is clearly healthy, for example when:
  - accepted
  - low `err_norm`
  - low `theta_final`
  - few Newton iterations
  - no slow contraction

Why this is promising:

- controller choice already produced large step-count differences
- a slightly more permissive healthy-step regrowth rule could reduce accepted steps much more than additional predictor tuning

Implementation guidance:

- add as a new controller mode rather than editing `hairer_lean`
- keep it algebraic and branch-light
- avoid nonlocal host-side bookkeeping

### 2. Inspect Whether The Embedded Error Estimator Is Slightly Conservative

This is one of the most plausible large-leverage directions.

Idea:

- check whether `embedded2` systematically reports error norms that are conservative for this transport problem
- if so, add a new opt-in estimator or acceptance variant rather than changing the existing default

Why this is promising:

- if the embedded estimator is slightly conservative, the controller will never exploit larger stable timesteps regardless of predictor quality
- this can cap performance much more strongly than nonlinear warm-start details

Implementation guidance:

- keep the existing `embedded2` path intact
- only introduce new estimator behavior behind a new explicit mode
- preserve array-only computations and avoid host-side postprocessing

### 3. Inspect Whether The Error Norm Is Dominated By A Specific State Block

This is probably the most physics-specific high-payoff diagnostic.

Idea:

- determine whether one subsystem, especially `Er`, dominates the timestep error norm
- if so, consider adding a new opt-in error-scaling mode with better-balanced state weighting

Why this is promising:

- one stiff or sensitive component may be throttling the global timestep for the whole coupled transport solve
- rebalancing the norm can reduce accepted steps much more than small predictor changes

Implementation guidance:

- do not change the existing error norm in place
- add a new weighting / scaling mode only if diagnostics justify it
- keep the scaling differentiable and JAX-friendly

### 4. Add A Slightly Faster Recovery Path After One Conservative Accepted Step

This is a narrower controller refinement.

Idea:

- when a step is accepted and the solver immediately shows healthy Newton behavior again, allow faster recovery of timestep growth instead of staying conservative for too long

Why this is promising:

- the current best controller may still spend too long regaining large `dt` after a cautious step
- improving recovery can cut accepted steps without changing the base acceptance semantics

Implementation guidance:

- add as an opt-in controller variant
- avoid long-history state or host-side logic
- keep the state additions minimal if any are needed

### 5. Lower-Priority Direction: Small Blend Retuning Inside The Existing Collocation Family

This is now lower priority than controller / estimator work.

Idea:

- if future tuning returns to the predictor, only consider very small blend retuning around the successful `collocation` structure
- do not revisit aggressive extrapolation as the first next step

Why this is lower priority:

- predictor variants so far only moved the results by a few steps
- the main remaining gains likely live elsewhere

### Recommended Priority Order

1. Diagnose whether the error norm is dominated by a specific state block
2. Check whether `embedded2` is conservative for this benchmark family
3. Add a new opt-in aggressive-when-healthy `hairer_lean` variant
4. Only afterward revisit additional predictor tuning if still needed

## Update: First Controller-Side Implementation

The first controller-side implementation from the next-step list has now been added as a new opt-in mode:

- `radau_controller_mode = "hairer_lean_aggressive"`

### Design

- preserves the existing `hairer_lean` mode unchanged
- reuses the same base Hairer-lean timestep-growth formula
- only changes behavior on clearly healthy accepted steps

More specifically, the new mode can enforce a modest minimum regrowth floor when all of the following are true:

- the step is accepted
- recovery-quality Newton behavior is present
- `err_norm` is still comfortably small
- the step is not classified as very difficult

### Intent

- test whether a slightly more assertive regrowth policy can reduce accepted steps
- avoid changing the existing best-known `hairer_lean + collocation` baseline
- keep the implementation algebraic, cheap, JAX-friendly, and differentiability-friendly

### Important guardrail

- plain `radau_controller_mode = "hairer_lean"` remains unchanged
- this new behavior is only active when the new mode is explicitly selected

### Result so far

- `hairer_lean_aggressive` was tested with `collocation`
- observed result:
  - `n_steps = 160`

So in its current form this controller variant is not promising for the accepted-step minimization objective.

## Update: First Error-Estimator-Side Implementation

The first estimator-side implementation has now been added as opt-in variants of the existing embedded Radau error estimator.

### New modes

1. `radau_error_estimator = "embedded2"`

- existing baseline
- unchanged behavior

2. `radau_error_estimator = "embedded2_mean_scale"`

- keeps the same embedded Radau error vector
- changes only the normalization scale from:
  - `max(|y_n|, |y_{n+1}|)`
  - to
  - `0.5 * (|y_n| + |y_{n+1}|)`

Intent:

- test whether the current endpoint-max scaling is slightly conservative for this benchmark family

3. `radau_error_estimator = "embedded2_blend_scale"`

- keeps the same embedded Radau error vector
- uses a mild blended normalization scale:
  - `0.75 * max + 0.25 * mean`

Intent:

- provide a softer estimator experiment that is closer to the baseline than the full mean-scale variant

### Guardrail

- the embedded error vector itself is unchanged
- the current `embedded2` baseline path is unchanged
- the new behavior is only active when the new estimator modes are explicitly selected
- the implementation is array-only, JAX-friendly, and differentiability-friendly

### Results so far

For the restored `hairer_lean + collocation` baseline:

- `embedded2`
  - `n_steps = 139`
  - `synchronized_elapsed_s = 256.126`
- `embedded2_blend_scale`
  - `n_steps = 150`
  - `synchronized_elapsed_s = 260.866`
- `embedded2_mean_scale`
  - `n_steps = 154`
  - `synchronized_elapsed_s = 262.731`

So far:

- both estimator-side scaling variants are worse than plain `embedded2`
- `embedded2_blend_scale` is less bad than `embedded2_mean_scale`
- neither compensates with a better total runtime
- plain `embedded2` remains the preferred baseline

## Update: NTSS Source Comparison

To understand what has and has not really been tried already, the local NTSS Radau source was inspected directly:

- `NTSS/CRadau/radau.cpp`
- `NTSS/CRadau/CRadau.cpp`
- `NTSS/CRadau/CRadau.h`

### 1. Controller: NTSS default is not the same as our `gustafsson`

NTSS uses the classic Hairer/Wanner Radau step controller with:

- a Newton-iteration-aware safety factor:
  - `fac = min(safe, ((2*nit+1)*safe)/(newt+2*nit))`
- a bounded error-based quotient:
  - `quot = max(facr, min(facl, err^expo / fac))`
- and then:
  - `hnew = h / quot`

There is also an optional Gustafsson predictive controller, but in the NTSS defaults:

- `IWork[7] = 0`

so that predictive controller is **off by default**.

This means:

- NTSS default is not equivalent to the NEOPAX `gustafsson` controller mode
- `hairer_lean` is closer in spirit to the NTSS default than `gustafsson`
- but `hairer_lean` is still simpler than NTSS because it does not currently reproduce:
  - the Newton-iteration-aware `fac` formula
  - the `facl/facr` bounded quotient logic exactly
  - the accepted-step keep/freeze window based on `hnew / hold`

### 2. Predictor: NTSS uses dense-output stage extrapolation

The NTSS predictor is not the same as the current NEOPAX `collocation` predictor.

NTSS behavior is:

- if this is the first step, or a startup / reset condition is active:
  - initialize stage increments from zero
- otherwise:
  - extrapolate the next stage increments from the previous successful step using:
    - dense-output / continuation coefficients
    - the previous accepted step size
    - the new `h / hold` ratio

So the most important genuinely untried NTSS-like predictor idea is:

- a dense-output-based stage predictor built from the previous accepted Radau solution

This is **not** the same as:

- `collocation`
- `dt_ratio_gated_collocation`
- the earlier experimental extrapolated/Jacobian predictor variants

### 3. Newton rejection logic is already much closer to NTSS/Hairer

NTSS uses the familiar contraction machinery based on:

- `theta`
- `faccon = theta / (1 - theta)` when `theta < 0.99`
- a predicted remaining defect estimate
- rejection/shrink only when that predicted defect is too large

This is the part of the solver where NEOPAX has already moved much closer to the NTSS/Hairer logic.

So compared with controller/predictor/error-estimator work:

- Newton stopping / rejection is no longer the biggest untried gap

### 4. Embedded error: NTSS does not use our simple `embedded2` path

NTSS does not appear to use the current NEOPAX-style direct embedded error vector:

- `err_vec = h * (embedded_f0_weight * f0 + b_error @ stages)`

Instead, the NTSS code calls dedicated embedded-estimator routines such as:

- `estrad_`
- `estrav_`

and it also builds the scaling vector differently, in a more canonical componentwise form:

- `scal[i] = atol1 + rtol1 * abs(y[i])`

with `rtol1` itself modified internally in the Hairer/NTSS style.

This means:

- `embedded2_mean_scale` and `embedded2_blend_scale` were not NTSS-like experiments
- a closer NTSS-inspired estimator path remains untried

## What Has Really Been Tried vs. What Remains Open

### Already tried in a meaningfully similar spirit

- Hairer-like Newton stop metric and contraction-based nonlinear acceptance
- simpler accepted-step controller behavior via `hairer_lean`
- optional Gustafsson-like predictive controller behavior
- several local stage-predictor enrichments around the existing `collocation` predictor
- mild error-norm rescalings on top of the current `embedded2` estimator

### Not yet really tried

1. A closer NTSS default controller reproduction

- keep predictive Gustafsson off by default
- implement the iteration-aware `fac / quot / hnew` logic more faithfully
- include the accepted-step keep/freeze window behavior

2. A true NTSS-like dense-output predictor

- use previous accepted-step dense-output / continuation coefficients
- extrapolate the next stage increments from the previous successful Radau solution

3. A closer NTSS-like embedded estimator / scaling path

- not just rescaling the current `embedded2`
- but using a more faithful componentwise `scal[i]` construction and, if practical, a closer analogue of the NTSS embedded-estimator logic

## Updated Priority After The NTSS Inspection

Given the benchmark results so far and the NTSS source comparison, the most promising remaining directions for step reduction now look like:

1. A closer NTSS-style default controller, added as a new opt-in mode
2. A true NTSS-like dense-output predictor, also opt-in
3. A closer NTSS-style embedded estimator / scaling path

All of these should remain subject to the same guardrail:

- do not modify the established `hairer_lean + collocation + embedded2` baseline in place
- only add new opt-in modes so that the current best benchmark path stays stable

## Update: First NTSS-Like Controller Mode

A new opt-in controller mode has now been added:

- `radau_controller_mode = "hairer_ntss"`

Intent:

- move closer to the default non-predictive Hairer/NTSS controller behavior
- keep the current `hairer_lean` path unchanged
- test the controller before attempting a more NTSS-like predictor or estimator

Current implementation characteristics:

- uses a bounded Hairer-style `fac / quot / hnew` update
- includes a Newton-iteration-aware safety factor
- keeps predictive Gustafsson behavior off
- includes an NTSS-like accepted-step keep window:
  - if the proposed accepted-step growth stays in the mild range near the current step size, keep the same step size instead of changing it

Important guardrail:

- this is a new mode only
- the existing `hairer_lean + collocation + embedded2` baseline is unchanged

### Result so far

- `hairer_ntss + collocation + embedded2`
  - `n_steps = 207`
  - `synchronized_elapsed_s = 302.276`

So this first NTSS-like controller attempt is clearly worse than the established baseline and should be treated as a negative result for accepted-step minimization.

## Update: First NTSS-Like Dense Predictor Mode

A new opt-in predictor mode has now been added:

- `radau_predictor_mode = "ntss_dense_output"`

Intent:

- make a first pass at the most important still-untried NTSS-like predictor idea
- keep the current `collocation` predictor unchanged
- reuse only cheap local Radau history already available in the step state

Current implementation characteristics:

- builds a small dense-output-style extrapolation from:
  - the previous accepted stage history
  - the current start-of-step slope `f0`
  - the step ratio `h / hold`
- extrapolates the previous stage polynomial forward to the new collocation nodes
- falls back to `collocation` if the dense extrapolated guess is not finite

Important guardrail:

- this is a new predictor mode only
- `radau_predictor_mode = "collocation"` is unchanged
- no extra predictor-side Jacobian machinery is introduced

### Result so far

- `hairer_lean + ntss_dense_output + embedded2`
  - `n_steps = 139`
  - `synchronized_elapsed_s = 267.904`

So this first NTSS-like dense predictor matches the current best accepted-step count from plain `collocation`, but does not yet beat it.

## Update: First NTSS-Like Embedded-Scaling Mode

A new opt-in error-estimator mode has now been added:

- `radau_error_estimator = "embedded2_ntss_scale"`

Intent:

- keep the current `embedded2` error vector unchanged for now
- test the most accessible NTSS/Hairer-like difference first:
  - a more canonical componentwise scaling style

Current implementation characteristics:

- keeps the same embedded Radau error vector as `embedded2`
- switches the norm scaling to use the candidate state magnitude only:
  - `scale_i = atol + rtol_eff * abs(y_{n+1,i})`
- uses an NTSS/Hairer-style effective relative tolerance:
  - `rtol_eff = 0.1 * rtol^expmns`
  - with the same `expmns = (s + 1) / (2s)` stage-count-dependent exponent structure already used in the Newton `fnewt` logic

Important guardrail:

- this is a new estimator mode only
- `radau_error_estimator = "embedded2"` remains unchanged
- this is not yet a full reproduction of NTSS `estrad_` / `estrav_`
- it is the first isolated scaling-side approximation only

### Result so far

- `hairer_lean + collocation + embedded2_ntss_scale`
  - `n_steps = 112`
  - `synchronized_elapsed_s = 258.703`

This is the first NTSS-inspired modification that clearly beats the restored baseline on accepted-step count.

Comparison against the previous best baseline:

- `hairer_lean + collocation + embedded2`
  - `n_steps = 139`
  - `synchronized_elapsed_s = 256.126`

So the current leading interpretation is:

- the main missing gain did not come from the controller or predictor
- it came from the error-estimator scaling
- the earlier baseline error normalization was likely too conservative for this benchmark family

## Updated Best-Known Configuration

The current best-known accepted-step result is:

- `radau_newton_tol_mode = "hairer"`
- `radau_newton_fnewt_mode = "hairer"`
- `radau_controller_mode = "hairer_lean"`
- `radau_predictor_mode = "collocation"`
- `radau_error_estimator = "embedded2_ntss_scale"`

with:

- `n_steps = 112`

## Recommended Follow-Up Matrix

Since the new estimator mode appears to be the main breakthrough, the next most useful comparisons are:

1. `hairer_lean + ntss_dense_output + embedded2_ntss_scale`

- most interesting next test
- checks whether the NTSS-like predictor adds anything on top of the new estimator scaling

2. `hairer_ntss + collocation + embedded2_ntss_scale`

- lower priority
- mainly tests whether the controller still remains too conservative even after the estimator improvement

3. `hairer_ntss + ntss_dense_output + embedded2_ntss_scale`

- completeness test for the fuller NTSS-like bundle
- currently less likely to beat the best `hairer_lean` path

Current expectation:

- `embedded2_ntss_scale` is probably the main win
- `ntss_dense_output` is worth testing next
- `hairer_ntss` is still more likely to hurt than help on accepted-step count

## Update: Dense Predictor With NTSS-Style Error Scaling

The most important follow-up comparison has now been run:

- `hairer_lean + ntss_dense_output + embedded2_ntss_scale`
  - `n_steps = 112`
  - `synchronized_elapsed_s = 252.216`

Comparison against the current best-known baseline with the same estimator:

- `hairer_lean + collocation + embedded2_ntss_scale`
  - `n_steps = 112`
  - `synchronized_elapsed_s = 258.703`

So at the moment:

- `ntss_dense_output` does not further reduce accepted-step count beyond the current best `112`
- however, it does appear slightly better on walltime in this run

Current interpretation:

- the NTSS-style estimator scaling remains the dominant improvement
- the NTSS-like dense predictor is at least compatible with that improvement
- but it has not yet shown an additional accepted-step reduction beyond the estimator gain itself

## Update: NTSS-Style Error-Scaling Refinements

To understand which part of the successful NTSS-style estimator scaling is doing the useful work, two additional opt-in variants have now been added:

1. `radau_error_estimator = "embedded2_ntss_max_scale"`

- keeps the NTSS-style effective relative tolerance:
  - `rtol_eff = 0.1 * rtol^expmns`
- but uses `max(|y_n|, |y_{n+1}|)` instead of candidate-only scaling

Intent:

- test whether the main win came from the NTSS-style `rtol_eff` alone
- while restoring the more conservative max-based state scale

2. `radau_error_estimator = "embedded2_ntss_blend_scale"`

- keeps the same NTSS-style `rtol_eff`
- uses a mild blend between:
  - candidate-only scaling
  - max-based scaling

Intent:

- test whether the best result can be retained while slightly regularizing the pure candidate-only scaling

Guardrail:

- `embedded2_ntss_scale` remains unchanged
- these are follow-up variants only
- the embedded error vector itself is still unchanged across all these modes

### Results so far

For the restored `hairer_lean + collocation` benchmark family:

- `embedded2_ntss_scale`
  - `n_steps = 112`
  - `synchronized_elapsed_s = 258.703`
- `embedded2_ntss_max_scale`
  - `n_steps = 139`
  - `synchronized_elapsed_s = 257.475`
- `embedded2_ntss_blend_scale`
  - `n_steps = 152`
  - `synchronized_elapsed_s = 281.461`

Current interpretation:

- `embedded2_ntss_max_scale` collapses almost exactly back to the old baseline behavior
- `embedded2_ntss_blend_scale` is worse than both `embedded2_ntss_scale` and the old baseline
- this strongly suggests the main win comes specifically from the combination of:
  - NTSS-style `rtol_eff`
  - and candidate-only scaling
- the NTSS-style effective relative tolerance alone is not enough

## Update: First Transport-Structured NTSS Estimator

A new opt-in transport-structured estimator mode has now been added:

- `radau_error_estimator = "embedded2_ntss_transport_scale"`

Intent:

- preserve the successful NTSS-style candidate-only scaling for the full state
- keep density and pressure fully active in the norm
- only give the `Er` block a state-dependent floor-aware scale, since it is the most likely delicate block in this transport structure

Current implementation characteristics:

- density block:
  - `scale_n = atol + rtol_eff * abs(n_next)`
- pressure block:
  - `scale_p = atol + rtol_eff * abs(p_next)`
- `Er` block:
  - `scale_Er = atol + rtol_eff * max(abs(Er_next), Er_floor)`
- with:
  - `Er_floor = max(0.1 * rms(Er_next), 1e-3)`

Why this is structured but still general:

- if density or temperature/pressure become important, they still fully contribute through their own candidate-state scales
- the estimator is not `Er`-only
- it simply prevents very small local `Er` values from collapsing the normalization scale too aggressively

Guardrail:

- `embedded2_ntss_scale` remains unchanged
- this new behavior is opt-in only
- the implementation stays array-only, JAX-friendly, and differentiability-friendly

### Result so far

- `hairer_lean + collocation + embedded2_ntss_transport_scale`
  - `n_steps = 68`
  - `synchronized_elapsed_s = 234.904`

This is the strongest accepted-step reduction seen so far.

Comparison against the previous best estimator result:

- `hairer_lean + collocation + embedded2_ntss_scale`
  - `n_steps = 112`
  - `synchronized_elapsed_s = 258.703`

Current interpretation:

- the transport-structured NTSS estimator appears to be the current best mode for accepted-step reduction
- Newton is often doing a bit more work per accepted step, because the larger accepted timesteps are harder nonlinear solves
- however, the overall step count and runtime both improved significantly

### Validation note

So far, the visible differences in `Er` appear to be minimal and likely dominated by the fact that different solver modes save output at different accepted timesteps.

That means the next required validation should be:

- compare the solutions at fixed physical save times
- either by forcing a common output-time grid
- or by interpolating one run onto the other run's saved times

This is the right next check before treating `embedded2_ntss_transport_scale` as the preferred default beyond benchmark step-count optimization.

### Most promising next estimator refinements

Given the current results, the remaining useful estimator-side work is now likely to be very focused rather than generic.

The most promising next refinements are:

1. Tune the `Er` floor formula inside `embedded2_ntss_transport_scale`

- the current implementation uses:
  - `Er_floor = max(0.1 * rms(Er_next), 1e-3)`
- this is a reasonable first choice, but it is clearly tunable

Most useful follow-up variants:

- weaker `Er` floor
  - e.g. `0.05 * rms(Er_next)`
- stronger `Er` floor
  - e.g. `0.2 * rms(Er_next)`
- different absolute floors
  - e.g. `1e-4`, `1e-3`, `1e-2`

2. Tie the `Er` floor to the transport physics scale `DEr`

Possible forms:

- `Er_floor = c * DEr`
- or
- `Er_floor = max(c1 * rms(Er_next), c2 * DEr)`

Why this is attractive:

- `DEr` is already a meaningful scale in the transport setup
- this would make the estimator tuning more physically interpretable than using only an RMS-based heuristic

3. Separate tolerance strength by block if needed

If simple `Er` floor tuning is not enough, a next refinement could be:

- keep the current density / pressure candidate-only NTSS scaling
- but give `Er` its own effective tolerance strength

For example:

- different `atol_Er`
- or different `rtol_eff_Er`

Current recommendation:

- do not return to generic max/mean/blend estimator experiments first
- the highest-signal next estimator work is now tuning the transport-structured `Er` scaling specifically

## Update: First Transport-Weighted Predictor Mode

A new opt-in predictor mode has now been added:

- `radau_predictor_mode = "collocation_transport_weighted"`

Intent:

- preserve the successful `collocation` predictor structure
- make the fresh collocation correction block-aware
- let different transport blocks respond differently when their stage history looks stale

Current implementation characteristics:

- density block:
  - blockwise correction gate with a high minimum trust level
- pressure block:
  - blockwise correction gate with near-full trust by default
- `Er` block:
  - blockwise correction gate with a lower minimum trust level
  - intended to damp `Er` correction more strongly when the previous `Er` stage history looks stale

This means the predictor still has the same overall form:

- previous stage guess
- plus a collocation-style fresh correction

but the correction amplitude is now allowed to differ by block.

Guardrail:

- this is a new mode only
- `radau_predictor_mode = "collocation"` is unchanged
- the implementation stays array-only, JAX-friendly, and differentiability-friendly

### Additional predictor variant now available

A combined dense + transport-weighted predictor mode has also now been added:

- `radau_predictor_mode = "transport_weighted_dense"`

Intent:

- keep the transport-weighted block-aware correction idea
- blend it with the NTSS-like dense predictor enhancement
- allow dense-history information to help more where the blockwise history looks trustworthy

Guardrail:

- this is a separate opt-in mode only
- `collocation_transport_weighted` and `ntss_dense_output` remain unchanged

### Result so far

- `hairer_lean + collocation_transport_weighted + embedded2_ntss_transport_scale`
  - `n_steps = 64`
  - `synchronized_elapsed_s = 211.281`

Comparison against the previous best result:

- `hairer_lean + collocation + embedded2_ntss_transport_scale`
  - `n_steps = 68`
  - `synchronized_elapsed_s = 234.904`

Current interpretation:

- the transport-weighted predictor appears to improve further on top of the transport-structured NTSS estimator
- the predictor is allowing even fewer accepted steps than the already-strong `embedded2_ntss_transport_scale` baseline
- Newton per-step effort can still be somewhat higher on these larger accepted timesteps
- but both accepted-step count and total synchronized runtime improved further

Current best-known configuration is now:

- `radau_newton_tol_mode = "hairer"`
- `radau_newton_fnewt_mode = "hairer"`
- `radau_controller_mode = "hairer_lean"`
- `radau_predictor_mode = "collocation_transport_weighted"`
- `radau_error_estimator = "embedded2_ntss_transport_scale"`

with:

- `n_steps = 64`

### Most promising next predictor refinements

Given the latest results, the remaining predictor-side improvements are likely to be incremental and should stay in the same transport-aware spirit.

The most promising next predictor ideas are:

1. A transport-weighted dense-output predictor

- combine the NTSS-like dense predictor with the transport-weighted block-aware correction logic
- this is the cleanest next untried predictor-side extension

Motivation:

- `ntss_dense_output` alone matched the old collocation baseline on step count
- `collocation_transport_weighted` improved further once the transport-structured estimator was in place
- combining the two is the most natural next predictor experiment

2. An estimator-aware transport-weighted predictor

- keep the transport-weighted collocation structure
- make the `Er` correction trust depend more explicitly on the same small-scale sensitivity that motivated the transport-structured estimator

Motivation:

- the successful estimator result suggests that `Er` scaling is the most delicate blockwise ingredient
- the predictor could use the same block sensitivity to decide how strongly to trust `Er` history

3. A Newton-quality transport-weighted predictor

- keep the transport-weighted predictor
- additionally modulate block trust using previous Newton quality

Motivation:

- if the previous accepted step converged but required many iterations, some blocks may still have stale history
- this could be most useful when a single block is driving extra Newton work

Current recommendation:

- do not return to broad global extrapolation or Jacobian-enriched predictor experiments first
- the best next predictor-side work is now transport-aware and block-aware rather than generic

## Update: First Transport-Aware `hairer_lean` Controller Mode

A new opt-in controller mode has now been added:

- `radau_controller_mode = "hairer_lean_transport"`

Intent:

- preserve the successful `hairer_lean` controller structure
- keep the same base growth law
- only add a transport-aware regrowth nudge when the normalized local error is clearly dominated by the `Er` block while density and pressure remain comparatively easy

Current implementation characteristics:

- computes lightweight blockwise normalized local error RMS values from the existing embedded error estimate
- uses those only inside the new mode
- if:
  - the step is accepted
  - `Er` clearly dominates the blockwise normalized local error
  - density and pressure remain easy
  - Newton behavior is still healthy enough
- then:
  - apply a modest regrowth floor on top of the usual `hairer_lean` growth

Guardrail:

- `radau_controller_mode = "hairer_lean"` is unchanged
- this new behavior is opt-in only
- no host-side logic or non-JAX control path is introduced

## Update: First Symmetric Transport-Weighted `hairer_lean` Controller Mode

A new opt-in controller mode has now been added:

- `radau_controller_mode = "hairer_lean_transport_weighted"`

Intent:

- keep the same successful `hairer_lean` base growth law
- use the same blockwise normalized local error decomposition already available from the embedded error estimate
- respond symmetrically to whichever transport block is dominating, rather than only special-casing `Er`

Current implementation characteristics:

- computes smooth block weights from:
  - density error contribution
  - pressure error contribution
  - `Er` error contribution
- builds a smooth localization score from those block weights
- only when:
  - the step is accepted
  - recovery-quality Newton behavior is still present
  - the difficulty looks localized rather than global
- applies a mild weighted regrowth floor on top of the usual `hairer_lean` growth

Guardrail:

- this is a new mode only
- `radau_controller_mode = "hairer_lean"` remains unchanged
- the change is still algebraic, array-only, JAX-friendly, and differentiability-friendly

## Update: Discounted Transport-Aware `hairer_lean` Controller Mode

A new opt-in controller mode has now been added:

- `radau_controller_mode = "hairer_lean_transport_discounted"`

Intent:

- preserve the successful `hairer_lean` accepted-step growth law
- do **not** add extra regrowth floors on accepted steps
- only shrink less aggressively when the blockwise diagnostics indicate that the difficulty is localized rather than global

Current implementation characteristics:

- uses the same blockwise normalized local error decomposition already computed for the transport-aware controller experiments
- identifies localized difficulty using a smooth block-concentration score
- when localized difficulty is present:
  - applies a milder retry shrink
  - applies a less severe rejection cap on the replacement timestep

This means the mode is:

- discounting pessimism on localized difficulty
- rather than encouraging extra growth on accepted steps

Guardrail:

- this is a new mode only
- `radau_controller_mode = "hairer_lean"` remains unchanged
- the implementation is still algebraic, array-only, JAX-friendly, and differentiability-friendly

### Result so far

- `hairer_lean_transport_discounted + collocation_transport_weighted + embedded2_ntss_transport_scale`
  - `n_steps = 62`
  - `synchronized_elapsed_s = 211.968`

Comparison against the previous strongest configurations:

- `hairer_lean + collocation_transport_weighted + embedded2_ntss_transport_scale`
  - `n_steps = 64`
  - `synchronized_elapsed_s = 211.281`
- `hairer_lean_transport_weighted + collocation_transport_weighted + embedded2_ntss_transport_scale`
  - `n_steps = 63`
  - `synchronized_elapsed_s = 222.810`

Current interpretation:

- the discounted transport-aware controller is the strongest controller result so far
- it achieved the lowest accepted-step count seen so far
- unlike the growth-encouraging transport-aware controllers, it did so without a major runtime penalty
- this supports the working hypothesis that the remaining controller gain comes more from discounting false pessimism on localized difficulty than from forcing stronger accepted-step regrowth

## Updated Best-Known Configuration

The current best-known accepted-step result is now:

- `radau_newton_tol_mode = "hairer"`
- `radau_newton_fnewt_mode = "hairer"`
- `radau_controller_mode = "hairer_lean_transport_discounted"`
- `radau_predictor_mode = "collocation_transport_weighted"`
- `radau_error_estimator = "embedded2_ntss_transport_scale"`

with:

- `n_steps = 62`
- `synchronized_elapsed_s = 211.968`

## Most Promising Remaining Experiments

At this point the stack is already heavily transport-structured, so the remaining gains are likely to be incremental rather than dramatic.

The most promising next experiments are:

1. `transport_weighted_dense` with the discounted transport-aware controller

- this predictor was previously worse than `collocation_transport_weighted` when paired with plain `hairer_lean`
- but it has not yet been tested together with:
  - `hairer_lean_transport_discounted`
  - `embedded2_ntss_transport_scale`

Recommended comparison:

- `radau_controller_mode = "hairer_lean_transport_discounted"`
- `radau_predictor_mode = "transport_weighted_dense"`
- `radau_error_estimator = "embedded2_ntss_transport_scale"`

Observed result:

- `hairer_lean_transport_discounted + transport_weighted_dense + embedded2_ntss_transport_scale`
  - `n_steps = 63`
  - `synchronized_elapsed_s = 210.381`

Comparison against the current best step-count configuration:

- `hairer_lean_transport_discounted + collocation_transport_weighted + embedded2_ntss_transport_scale`
  - `n_steps = 62`
  - `synchronized_elapsed_s = 211.968`

Current interpretation:

- `transport_weighted_dense` is essentially tied on runtime and may even be slightly faster in this run
- but `collocation_transport_weighted` still remains the best predictor for minimizing accepted steps

2. Tune the `Er` floor inside `embedded2_ntss_transport_scale`

This is still the most promising estimator-side refinement.

Most useful variants:

- weaker floor factor
  - e.g. `0.05 * rms(Er_next)`
- stronger floor factor
  - e.g. `0.2 * rms(Er_next)`
- `DEr`-based floor
  - e.g. `max(c1 * rms(Er_next), c2 * DEr)`

3. Refine the discounted controller using blockwise Newton difficulty

The current discounted controller discounts pessimism based on localized blockwise error structure.

A possible refinement would be:

- only discount pessimism when the localized difficult block also looks Newton-benign

This is lower priority than the two items above, but it is the most plausible next controller refinement if more controller work is attempted.

## Current Recommendation

If only one next test is run, the strongest next candidate is:

- `hairer_lean_transport_discounted + transport_weighted_dense + embedded2_ntss_transport_scale`

After that, the highest-value remaining refinement is still estimator-side tuning of the transport-structured `Er` floor.

## Update: First General Block-Floor Estimator Mode

A new opt-in estimator mode has now been added:

- `radau_error_estimator = "embedded2_ntss_block_floor_scale"`

Intent:

- generalize the successful transport-structured scaling idea beyond only `Er`
- allow density, pressure, and `Er` all to protect themselves if their candidate-state scale becomes too small

Current implementation characteristics:

- density block:
  - `floor_n = max(0.05 * rms(n_next), 1e-4)`
- pressure block:
  - `floor_p = max(0.05 * rms(p_next), 1e-4)`
- `Er` block:
  - `floor_Er = max(0.1 * rms(Er_next), 1e-3)`

Then each block uses:

- `scale_block = atol + rtol_eff * max(abs(block_next), floor_block)`

Guardrail:

- this is a new estimator mode only
- `embedded2_ntss_transport_scale` remains unchanged
- this is the first general block-floor version, not yet the more physics-aware `DEr`-based refinement

### Result so far

- `hairer_lean_transport_weighted + collocation_transport_weighted + embedded2_ntss_transport_scale`
  - `n_steps = 63`
  - `synchronized_elapsed_s = 222.810`

Comparison against the previous best practical configuration:

- `hairer_lean + collocation_transport_weighted + embedded2_ntss_transport_scale`
  - `n_steps = 64`
  - `synchronized_elapsed_s = 211.281`

Current interpretation:

- the weighted transport-aware controller achieved the lowest accepted-step count seen so far
- but only by a very small margin (`64 -> 63`)
- and it did so with worse total synchronized runtime

So the current split is:

- best raw accepted-step count:
  - `hairer_lean_transport_weighted + collocation_transport_weighted + embedded2_ntss_transport_scale`
- best practical near-best configuration:
  - `hairer_lean + collocation_transport_weighted + embedded2_ntss_transport_scale`

### Most promising next estimator refinements

Even after the strong success of `embedded2_ntss_transport_scale`, there is still some focused estimator-side work worth considering.

The highest-signal next estimator refinements are:

1. Tune the `Er` floor inside `embedded2_ntss_transport_scale`

- current form:
  - `Er_floor = max(0.1 * rms(Er_next), 1e-3)`

Most useful follow-up variants:

- weaker `Er` floor
  - e.g. `0.05 * rms(Er_next)`
- stronger `Er` floor
  - e.g. `0.2 * rms(Er_next)`
- different absolute floors
  - e.g. `1e-4`, `1e-3`, `1e-2`

2. Tie the `Er` floor to the transport physics scale `DEr`

Possible forms:

- `Er_floor = c * DEr`
- or
- `Er_floor = max(c1 * rms(Er_next), c2 * DEr)`

Motivation:

- `DEr` is already a meaningful scale in the transport setup
- this would make the estimator more physically interpretable than a purely RMS-based heuristic

3. Give `Er` its own effective tolerance strength if needed

If floor tuning is not enough, the next refinement could be:

- keep density and pressure candidate-only NTSS scaling unchanged
- but give the `Er` block a separate effective tolerance strength

For example:

- different `atol_Er`
- or different `rtol_eff_Er`

Current recommendation:

- do not return to generic max/mean/blend experiments
- the highest-value remaining estimator work is now refining the transport-structured `Er` scaling specifically
