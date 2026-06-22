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

## 2026-06-22 Status Update: Forward AD Recovered

The scalar forward-AD lane has been recovered for the `T0` benchmark case.

### What was fixed

The forward-AD custom JVP had regressed into a payload-heavy path:

- `_radau_adaptive_final_y_realized_schedule_jvp(...)`
- was calling `_radau_adaptive_final_state_rollout(...)`
- which materialized large per-attempt payloads, including state snapshots,
  stage/carry data, lagged-cache data, and LU/cache fields

This produced the large HLO/input-output memory signature and OOM even for
short accepted-step prefix tests.

The custom-JVP primal schedule capture now uses the compact schedule rollout:

- `_radau_adaptive_schedule_rollout(...)`

This restored the lightweight realized-schedule forward-AD behavior:

- primal adaptive solve is performed inside the custom JVP
- the realized schedule/control trace is treated as nondifferentiable
- tangent propagation replays the realized accepted-step map
- no external baseline run is required for forward AD
- no reverse-lane payload machinery is used

### Validation command

```bash
python ./examples/benchmarks/benchmark_transport_forward_ad_only.py \
  --ntx-exact-derivative-mode direct \
  --parameter T0 \
  --radau-jacobian-reuse-mode legacy
```

### Recovered forward-AD values

```text
softmax_Er                         -2.160399e+01
smooth_root_proxy                   2.070900e-05
Er2_volume_average                 -2.765750e+01
Er_volume_average                   2.291385e+00
electron_temperature_volume_average 3.571291e-01
total_pressure_volume_average       1.835267e+00
alpha_power_volume_average          7.221955e-02
```

These match the old trusted benchmark regime and agree closely with the
recovered frozen-FD accepted-time benchmark.

### Current FD reference state

The recovered FD reference uses:

```bash
python ./examples/benchmarks/benchmark_transport_frozen_fd_only.py \
  --ntx-exact-derivative-mode direct \
  --parameter T0 \
  --fd-rel-step 3e-8 \
  --fd-abs-step 1e-10 \
  --replay-mode accepted \
  --fixed-time-lane direct \
  --radau-jacobian-reuse-mode legacy
```

Important interpretation:

- the FD lane runs a separate baseline adaptive solve to get the accepted
  `dt` list
- `fd-` and `fd+` then recompute physics from their own perturbed initial
  carries on that accepted time grid
- baseline carry history is not copied into the FD endpoints
- the direct fixed-time lane may report a few Newton nonconvergence flags, but
  it reaches `t_final` with finite endpoint objectives and recovers the old
  FD scale

### Next Implementation Step: Fused Forward AD

The current recovered forward AD is correct but still two-phase inside the
custom JVP:

1. primal adaptive solve records the compact realized schedule
2. accepted-step replay propagates the tangent

The next optimization is a fused forward-AD lane:

- propagate primal and tangent together during the adaptive solve
- controller decisions remain primal-only
- tangent does not affect accept/reject or next-`dt` decisions
- the fused result must reproduce the recovered forward-AD values above
- only after matching the recovered values should the replay-based JVP be
  retired or downgraded to a diagnostic/reference path

### Next Milestone After Fused Forward AD: Reverse AD Recovery

After fused forward AD is validated, recover the reverse AD lane against the
forward-AD reference values above.

Reverse AD should be treated as a separate lane:

- no changes to the recovered forward solver / FD / forward-AD lanes unless
  explicitly required and reviewed
- reverse should match the accepted-step AD contract used by forward AD
- validation order should be:
  1. reverse vs recovered forward AD on short prefixes
  2. reverse vs recovered forward AD on full `T0`
  3. reverse vs recovered FD only as a secondary sanity check

The forward-AD row above is now the primary numerical reference for the next
reverse-AD recovery work.
