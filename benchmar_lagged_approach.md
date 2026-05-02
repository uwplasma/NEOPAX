# Benchmark Lagged Approach

## Goal

Separate three effects that are currently mixed together:

1. raw NTX lagged-response build/evaluation cost
2. Radau retry / rejection behavior
3. physical differences in the initial `Er` and fluxes between lagged-response variants and the database interpolation baseline

The next session should focus on benchmarking and diagnosing these separately before making more model changes.

## Main Hypothesis

The current timings are being contaminated by repeated Radau step attempts.

So we need to compare:

- full `n_radial = 51` lagged response
- coarse-anchor lagged response

under a benchmark that isolates:

- one single step attempt
- whether that attempt is accepted or rejected

and then compare the initial-state `Er` and produced fluxes against the usual database interpolation path.

## Current Findings

The latest single-attempt tests already showed two important facts:

1. `jax.lax.map` over the full `51` radii is probably **not** the primary problem.
2. The main issue appears to be that the initial Radau step attempt is being **rejected**.

Evidence:

- in single-attempt mode, both:
  - coarse-anchor lagged response
  - full `51`-radius lagged response
  ended with:
  - `n_steps = 0`
  - `final_t = 0.0`
- that means the first attempt did not advance time
- this points to step rejection rather than simply “too many radii” or “`lax.map` is too slow”

So the next session should prioritize:

- diagnosing why the first Radau attempt is rejected
- comparing the initial `Er` and fluxes against the database interpolation baseline

before making more NTX batching changes.

## Benchmark Set A: One Step Attempt Only

Run timing benchmarks for exactly one Radau step attempt, without mixing in repeated retries.

Compare:

1. full `n_radial = 51` lagged response
   - current exact-runtime lagged path
   - `lax.map` / unbatched radial mode
   - and, if still available/desired, batched `vmap` radial mode

2. coarse-anchor lagged response
   - e.g. `anchor_count = 7`
   - same solver settings

For each case, record:

- wall time
- GPU utilization pattern
- whether the single attempt was accepted or rejected
- nonlinear iteration count if available
- error-estimator quantity if available

Important:

- this benchmark should stop after one **attempt**
- not after one **accepted** step

Reason:

- the current `stop_after_accepted_steps = 1` benchmark still allows multiple rejected attempts
- that contaminates the comparison between lagged-response variants

Important interpretation:

- if `n_steps = 0` and `final_t = 0.0`, the one attempt was rejected
- that is now the main condition to diagnose

## Benchmark Set B: One Accepted Step

After Set A, also benchmark the current "one accepted step" setup for the same two lagged approaches:

1. full `n_radial = 51`
2. coarse-anchor (`7`, optionally `14`)

For each case, count:

- how many attempts were needed before the first accepted step
- how many times `build_lagged_response(...)` was called
- how many lagged RHS evaluations were made

This will show whether the apparent slowdown is mostly:

- raw NTX lagged cost
- or repeated retries / rebuilds

## Physics Comparison: Initial Step State

For the initial transport state and initial step attempt, compare the following across:

1. normal database interpolation black-box path
2. full `n_radial = 51` lagged exact-runtime path
3. coarse-anchor lagged exact-runtime path

Compare:

- initial `Er`
- neoclassical fluxes
- total shared fluxes
- any intermediate quantities needed to explain differences:
  - `Gamma`
  - `Q`
  - `Upar`
  - optionally reduced transport moments / `Lij`

Questions to answer:

1. Are the initial `Er` values effectively identical?
2. Are the initial fluxes effectively identical?
3. If not, where do they first diverge?
4. Is the divergence already present before Radau iteration begins?
5. If the initial `Er` and fluxes are effectively identical to the database interpolation path, why is the lagged path still rejecting the first attempt?

Important initialization detail:

- the ambipolar `Er` root initialization uses the **active run flux model** for flux/root evaluation
- in the current benchmark override, that means:
  - `flux_model = ntx_exact_lij_runtime`
- but the entropy ranking used to choose among valid roots still comes from:
  - `neoclassical.entropy_model = ntx_database`

So the initialization split is:

1. root candidate flux evaluations:
   - active transport flux model
2. entropy-based root selection:
   - configured entropy model

## Interpretation Rule

If the initial `Er` and fluxes from:

- full `51` lagged exact-runtime
- coarse-anchor lagged exact-runtime

are both effectively identical to the database interpolation baseline, then:

- there is no obvious physics reason for the lagged approach to trigger repeated rejections if the normal database interpolation run does not

That would point more strongly to:

- solver interaction
- lagged-response linearization behavior
- or implementation inefficiency / instability

instead of raw model disagreement.

## Specific Comparisons To Do

### Timing

1. full `51` lagged response, one attempt
2. coarse `7` lagged response, one attempt
3. full `51` lagged response, one accepted step
4. coarse `7` lagged response, one accepted step

Optional:

5. coarse `14` lagged response, one attempt
6. coarse `14` lagged response, one accepted step

### Initial-State Physics

1. database interpolation black-box vs full `51` lagged
2. database interpolation black-box vs coarse `7` lagged
3. full `51` lagged vs coarse `7` lagged

### Rejection Diagnosis

For the first single attempt, determine whether rejection comes mainly from:

1. nonlinear solve non-convergence
2. error-estimator rejection after convergence
3. an initial `dt` that is too aggressive for the lagged path

Expose, if possible:

- converged / not converged
- `err_norm`
- rejection reason / fail code
- number of lagged RHS evaluations in the attempt

### Ambipolarity Comparison

Run ambipolarity-mode benchmarks directly, using the same exact-runtime and coarse-anchor settings, to compare against the transport-step benchmarks.

Goals:

1. measure the cost of the initial `Er` root solve separately from Radau transport stepping
2. compare the roots and flux evaluations obtained with:
   - database interpolation
   - full `51` exact-runtime lagged-related settings
   - coarse-anchor exact-runtime lagged-related settings
3. determine whether the initialization itself is already producing differences that can later trigger Radau rejection

Suggested comparisons:

1. ambipolarity mode with database interpolation baseline
2. ambipolarity mode with exact-runtime full radial response
3. ambipolarity mode with exact-runtime coarse-anchor response where applicable

Compare:

- best-root `Er` profile
- candidate roots if available
- ambipolar flux values near the chosen root
- runtime / GPU usage

## Desired Outputs For Next Session

Prepare a small table or note with:

- approach
- one-attempt wall time
- one-attempt accepted/rejected
- one-accepted-step wall time
- number of retries before first accepted step
- initial `Er` comparison
- initial flux comparison

## Open Questions

1. Can we cleanly benchmark exactly one Radau attempt without altering the physical model?
2. Can we expose attempt counters and rejection reasons in the benchmark output?
3. Are the lagged-response variants reproducing the same initial flux state as the database interpolation baseline?
4. If yes, why does Radau still reject more often?
5. If no, which quantity first causes the divergence?
6. Is the rejection happening even when full `51` `lax.map` response is used, confirming that the problem is not just the coarse-anchor reduction?
