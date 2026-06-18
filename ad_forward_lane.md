# AD Forward Lane Plan

## Goal

Recover a correct and efficient forward-AD benchmark lane for transport that is independent from the reverse lane and is consistent with the restored forward solver / FD baseline behavior.

The desired evolution is:

1. first recover correct derivatives with a **primal adaptive solve + accepted-step replay JVP** strategy
2. then move to the final **fused primal-step + tangent-propagation** strategy

## Guiding Principle

The forward AD lane should follow the forward solver lane, not the reverse lane.

That means:

- the primal baseline must come from the same forward adaptive solver path now being recovered for the solver / FD lanes
- forward AD should not route through reverse replay or generalized accepted-step payload machinery if that widens the graph or changes behavior
- if needed, forward AD should have its own specialized helpers even if that duplicates some logic

## Current Context

Recent forward-lane recovery work established:

- the plain forward solver path and the FD fixed-time lane should both use the same forward-primal NTX treatment
- the FD baseline should use the lighter adaptive schedule trace instead of the payload-heavy rollout
- the fixed-time FD lane should follow the same lagged-response state evolution as the adaptive forward lane

This means the forward AD lane should now be rebuilt on top of this same recovered primal forward map.

## Phase A: Recover Correct AD with Primal Map + Replay

### Objective

Recover correct forward AD derivatives before attempting the more ambitious fused primal+tangent strategy.

### Intended Structure

1. Run the primal adaptive forward solve on the same lane used by the restored forward solver / FD baseline.
2. Extract the realized accepted-step map from the lightweight schedule trace.
3. Run forward-mode differentiation on the realized accepted-step replay lane.
4. Compare those AD derivatives against the frozen-FD benchmark.

### What this phase should use

- primal adaptive solve:
  - same forward solver lane as the current recovered solver / FD baseline
- realized accepted-step trace:
  - only the lightweight accepted-step schedule metadata
- replay/JVP:
  - accepted-step replay only
  - no reverse payload machinery
  - no generalized heavy trace objects

### What this phase should avoid

- payload-heavy rollout traces for baseline collection
- reverse-lane helper reuse when it changes forward behavior
- generalized AD wrappers that are broader than what the forward lane needs

## Phase A Tasks

### Step 1: Baseline recovery

Recover the forward AD benchmark baseline so it uses the same primal adaptive forward lane as the restored FD baseline.

Desired result:

- the baseline run is just the trusted forward-primal solve plus a lightweight schedule trace
- no giant payload trace
- no alternate solver wrapper that differs from the normal forward lane

### Step 2: Accepted-step replay recovery

Make the forward AD replay lane use the realized accepted-step map from the recovered primal baseline.

Desired result:

- replay runs only on the saved accepted-step map
- replay uses the same forward fixed-time-map solver philosophy as the FD lane
- lagged-response reuse/rebuild treatment is consistent with the forward solver

### Step 3: JVP recovery

Recover the JVP application on top of the realized accepted-step replay.

Desired result:

- derivatives are finite
- derivatives are numerically plausible
- derivatives can be compared against the FD lane with the same accepted-step cutoff

### Step 4: Validation against FD

Compare the recovered forward AD derivatives against the FD-only benchmark.

Desired result:

- pressure / temperature / alpha metrics match well first
- `Er`-related metrics are then brought into agreement by further replay-lane alignment if needed

## Phase B: Move to Fused Primal + Tangent Propagation

### Objective

Replace the two-phase primal-baseline + replay-JVP strategy with a single forward lane that propagates primal and tangent together during the adaptive solve.

### Intended Structure

- primal step and tangent step advance together
- accepted-step logic is handled in the forward lane itself
- no separate replay extraction pass is required for the main benchmark

### Important constraint

Do not attempt this until Phase A has recovered correct derivatives.

The fused strategy should be an optimization/refactor of a correct forward AD lane, not a replacement for the initial debugging path.

## Recommended Order

1. recover primal baseline on the normal forward solver lane
2. recover accepted-step replay lane
3. recover correct forward AD derivatives with replay JVP
4. benchmark against FD
5. only then implement fused primal+tangent propagation

## Success Criteria

### Phase A success

- forward AD baseline uses the same primal forward lane as solver / FD
- replay uses only the realized accepted-step map
- AD derivatives are finite and numerically consistent with FD

### Phase B success

- fused forward AD reproduces the Phase A derivative values
- fused lane is cheaper than the two-phase replay strategy
- forward lane remains independent of reverse lane machinery

## Next Recommended Action

Start with Phase A, Step 1:

- recover the forward AD baseline so it uses the same lightweight adaptive primal lane as the restored FD benchmark baseline
