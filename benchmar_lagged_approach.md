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
