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
