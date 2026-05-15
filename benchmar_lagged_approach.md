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
