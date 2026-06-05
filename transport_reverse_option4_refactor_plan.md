# Transport Reverse Option 4 Refactor Plan

## Goal

Refactor the transport reverse AD path so that reverse mode differentiates the
same accepted-step map that the current forward AD path already differentiates,
without relying on the current replay-heavy rollout reverse machinery.

Primary success criterion:

- reverse AD must match the current forward AD values for
  `benchmark_transport_profile_vector_ad_compare.py`
  under the accepted-step realized-schedule contract

Secondary success criteria:

- eliminate the current replay/checkpoint OOM failure mode
- remove dependence on broad solver-internal replay for reverse correctness
- preserve the current forward AD path unchanged during this refactor

## Hard Contract

This refactor must preserve:

- differentiate the accepted-step composition only
- the accepted-step schedule/time map is fixed by the primal forward pass
- reverse mode must target the same accepted-step time map as forward mode
- rejected attempts may contribute nondifferentiated support/replay metadata only
- rejected attempts must not become differentiated transitions

Additional temporary rule for this refactor:

- do not change the current forward AD path unless explicitly approved later
- use forward AD as the numerical reference that reverse must match

## Why This Refactor Is Needed

The current reverse path is still hybrid:

- local accepted-step reverse pieces are hand-derived
- but rollout reverse still depends on replay/checkpoint scaffolding
- reverse still reconstructs local primal accepted-step context
- correctness has already shown sensitivity to reconstructed
  `trial_dt` / `stage_history`
- memory has repeatedly blown up in the outer reverse replay/scan path

So the remaining problem is not just checkpoint plumbing. The main issue is that
reverse still does not treat the accepted-step map itself as the true
differentiated primitive.

## Target Architecture

The final architecture should look like this:

1. Forward accepted-step primitive

- keep the current accepted-step forward AD map
- this remains the authoritative `JVP` / forward-AD object
- it should continue to be validated against FD as it is today

2. Reverse accepted-step primitive

- introduce a first-class reverse rule for the same accepted-step map
- this rule must be defined directly on the accepted-step primitive
- it must not depend on broad replay of full solver internals

3. Rollout composition

- rollout reverse is just composition of accepted-step reverse rules
- accepted-step schedule comes from the primal rollout trace
- rejected attempts remain schedule-support metadata only

4. Minimal saved state

- reverse should only require the minimal per-accepted-step state actually needed
  by the accepted-step reverse rule
- this should be explicit and versioned in code

## What The Accepted-Step Primitive Should Be

Define a single accepted-step primitive with:

- inputs:
  - accepted-step carry at step `n`
  - accepted-step `dt_n`
  - solver/physics contexts
  - fixed accepted-step support metadata from the realized forward pass

- outputs:
  - accepted-step carry at step `n+1`
  - reduced replay-state outputs used by rollout composition

Recommended differentiated outputs for the primitive:

- `t_out`
- `y_out`
- `dt_out`
- `prev_stages_out`
- `prev_dt_out`
- `lagged_reference_y_out`
- `prev_theta_final_out`

These match the current replay-state concept and the forward-path contract much
better than full carry replay.

## Minimal Per-Step Reverse Payload

The reverse primitive should not reconstruct arbitrary local primal state from
scratch if the exact reverse depends on it.

Instead, define a minimal explicit per-accepted-step reverse payload. This is
the most important design point of the refactor.

Candidate required payload fields:

- `t_in`
- `y_in`
- `dt_in`
- `accepted_y`
- `trial_y`
- `trial_dt`
- `stage_history`
- `prev_dt_in`
- `prev_theta_final_in`
- `prev_newton_iter_count_in`
- `lagged_response_valid_in`
- `lagged_reference_y_in`

Candidate optional payload fields:

- `rhs_time_ref`
- `jacobian_out`
- any exact reduced stage-linear-solve artifacts if they are truly needed by the
  local reverse rule

Bias for the refactor:

- save exact local primal information that the accepted-step reverse actually
  depends on
- do not save broad full-carry / full-cache / full-checkpoint objects unless
  forced

Important note:

- earlier diagnostics already showed that local replay correctness is sensitive
  to `trial_dt` and especially `stage_history`
- so these should be treated as first-class reverse payload candidates, not as
  reconstructible conveniences

## Recommended Refactor Strategy

### Phase 0. Freeze the forward reference

Before changing reverse further:

- treat the current forward AD path as frozen
- record one or more reference forward-AD outputs for:
  - the current reverse benchmark command
  - optionally a small accepted-step-count narrowed benchmark

Do not optimize or rewrite the forward path in this phase.

### Phase 1. Make the accepted-step primitive explicit

Create or formalize a single accepted-step primitive boundary in code:

- one module-scope primal function
- one explicit reduced output state
- one explicit payload builder

Required deliverables:

- a dataclass for reduced accepted-step primal outputs
- a dataclass for accepted-step reverse payload
- one builder that populates the payload from the exact primal accepted-step run

### Phase 2. Build the local reverse rule directly on the primitive

Replace the current hybrid local replay reverse with a clean primitive-level
reverse rule:

- input:
  - accepted-step payload
  - reduced output cotangent
- output:
  - reduced input cotangent

This reverse rule should be the only authoritative local reverse.

Important design rule:

- do not let the primitive reverse depend on outer rollout checkpoint plumbing
- outer rollout logic should call this primitive reverse, not partially
  reconstruct it

### Phase 3. Compose primitives across the accepted-step rollout

At rollout level:

- run the primal forward pass once to obtain:
  - accepted-step schedule
  - per-step accepted-step reverse payloads
- backward pass then walks the accepted steps in reverse order and applies the
  primitive reverse rule step by step

This rollout reverse should become simple:

- no broad full-carry replay
- no full realized-attempt reverse
- no reconstruction of ambiguous local primal internals during backward

### Phase 4. Reduce saved payload to the true minimum

Once correctness is achieved:

- measure which payload fields are actually required
- trim redundant fields
- only then revisit memory/performance optimization

Do not try to prematurely minimize payload before correctness is stable.

## Existing Reverse Path: What Can Be Removed

If this refactor succeeds, it should allow removal or retirement of most of the
current rollout-level reverse replay complexity, especially:

- broad custom-VJP reverse replay/checkpoint logic for the accepted-step map
- reduced replay-state reverse scans whose main purpose is reconstructive replay
- ad hoc checkpoint-trace / checkpoint-carry reconstruction layers created only
  to make the current hybrid reverse fit in memory

Recommended removal policy:

1. keep old reverse path behind a temporary debug switch while the new path is validated
2. once the new primitive reverse matches forward AD on the benchmark, delete the old rollout reverse path

## Tests and Validation Plan

### A. Local primitive tests

For one accepted step:

- compare primitive reverse against direct `jax.vjp` only on a tiny trusted
  local reference if feasible
- validate exact agreement for the reduced accepted-step output state

Required checks:

- `y_out` cotangent
- `dt_out` cotangent
- `prev_stages_out` cotangent
- `lagged_reference_y_out` cotangent
- `prev_theta_final_out` cotangent

### B. Short rollout agreement against forward AD

For a small number of accepted steps:

- run current forward AD
- run new reverse AD
- require close agreement of full parameter gradients

This is the most important early end-to-end test.

### C. Production benchmark comparison

Use the main benchmark:

```bash
python ./examples/benchmarks/benchmark_transport_profile_vector_ad_compare.py --ntx-exact-derivative-mode direct --ad-mode both --objective-indices 0
```

Success criterion:

- reverse row matches current forward AD row closely

### D. Only after AD-vs-AD agreement, revisit FD

Because the current goal of this refactor is to make reverse match the already
trusted forward AD path, the validation order should be:

1. reverse vs forward AD
2. then forward/reverse vs FD

Do not block the architecture refactor on immediate FD comparison.

## Most Important Design Rules

1. The accepted-step primitive is the differentiated object.

2. The reverse primitive should consume exact local primal payload, not a fresh
   guessed reconstruction of local internals.

3. The rollout reverse should be simple composition of primitive reverse rules.

4. Rejected attempts remain support-only schedule metadata.

5. Forward AD remains the reference during this refactor.

6. Memory optimization should happen after the primitive reverse is correct, not
   by adding more outer replay plumbing first.

## Forward AD: Temporary Policy

During this reverse refactor:

- do not change the current forward AD path
- do not change its accepted-step contract
- do not “optimize” it preemptively

Potential forward-path speedup ideas may be written down separately, but they
should not be implemented before explicit approval.

## Optional Forward-Path Speedup Ideas To Document Only

These are allowed to be documented, but not implemented yet:

- reduce unnecessary saved leaves in forward accepted-step payload construction
- expose a lighter reduced accepted-step output object for benchmarking
- reuse exact local linearization artifacts more systematically in forward JVP
- trim redundant stage-eval tangent work in the accepted-step JVP

These are notes only for later.

## Concrete First Refactor Tasks

1. Introduce explicit dataclasses:
   - reduced accepted-step output
   - accepted-step reverse payload

2. Implement a single payload builder at the accepted-step primal boundary.

3. Implement a new primitive reverse function:
   - payload in
   - reduced output cotangent in
   - reduced input cotangent out

4. Implement a simple rollout reverse that:
   - uses primal saved accepted-step payloads
   - walks them backward
   - applies the primitive reverse

5. Compare the new reverse against the existing forward AD on:
   - one-step
   - short rollout
   - main profile-vector benchmark

6. Only after agreement:
   - remove or retire the old rollout reverse replay path

## Recommended Definition of Done

This refactor is done when all of the following are true:

- reverse differentiates the same accepted-step map as forward
- reverse matches current forward AD on the benchmark row values
- the old giant replay/checkpoint OOM path is no longer required
- the reverse implementation no longer depends on reconstructing ambiguous local
  primal accepted-step internals during backward
- the code expresses the accepted-step primitive and its reverse rule directly

## 2026-06-05 Current Resume Point

### Current best diagnosis

- The local accepted-step custom reverse rule itself is viable under JIT when
  the reverse payload is closed over as a residual-like constant.
- The same rule OOMs under JIT when the reverse payload is passed as a dynamic
  runtime argument.
- Therefore the current blocker is the **dynamic runtime reverse-payload
  contract**, not checkpointing or rollout composition by themselves.

### What this means for the option-4 plan

The next work should stay focused on the one-step primitive level:

1. reduce the dynamic payload contract
2. make the runtime reverse look more like the forward path in using a narrow
   active contract
3. only after dynamic one-step JIT works, resume segmented multi-step rollout
   scaling

### Current benchmark surfaces

Working closed-over one-step:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_one_step_primitive.py --ntx-exact-derivative-mode direct --execution-mode jit --payload-mode closed-over
```

Failing dynamic one-step:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_one_step_primitive.py --ntx-exact-derivative-mode direct --execution-mode jit --payload-mode dynamic
```

Segmented multi-step benchmark now exists, but still inherits the dynamic
payload blocker:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_multi_step_primitive.py --ntx-exact-derivative-mode direct --accepted-step-counts 20000 --execution-mode jit --max-total-steps-multiplier 1 --segment-length 8 --checkpoint-count 0
```

### Immediate next tests

Use the one-step dynamic payload ablations first:

- `--payload-ablation none`
- `--payload-ablation stage`
- `--payload-ablation lagged`
- `--payload-ablation jacobian`
- `--payload-ablation lu`
- `--payload-ablation pivots`

These are now the highest-value tests before any more rollout work.

## Function-by-Function Implementation Checklist

This section maps the refactor onto the current code in:

- `NEOPAX/_transport_solvers.py`

### Current anchor points

The current main anchors are:

- `_RadauAcceptedStepCarry`
- `_RadauReplayState`
- `_RadauAcceptedStepAttemptResult`
- `_RadauAcceptedStepAttemptContext`
- `_execute_radau_accepted_step_attempt(...)`
- `_execute_radau_accepted_step_attempt_autodiff(...)`
- `_radau_apply_accepted_step_replay_state_pullback_linearized(...)`
- `_radau_replay_realized_accepted_carry_trace(...)`
- `_radau_replay_realized_accepted_carry_pullback(...)`
- `_radau_adaptive_final_y_realized_schedule_jvp(...)`
- `_radau_adaptive_final_y_realized_schedule_vjp_*`

### Step 1. Introduce explicit primitive dataclasses

Add new dataclasses near the current accepted-step dataclass block:

1. `_RadauAcceptedStepReducedOutput`
   - explicit reduced accepted-step output state
   - target fields:
     - `t_out`
     - `y_out`
     - `dt_out`
     - `prev_stages_out`
     - `prev_dt_out`
     - `lagged_reference_y_out`
     - `prev_theta_final_out`

2. `_RadauAcceptedStepReversePayload`
   - exact per-step primal payload for reverse
   - first-pass payload should include:
     - `t_in`
     - `y_in`
     - `dt_in`
     - `accepted_y`
     - `trial_y`
     - `trial_dt`
     - `stage_history`
     - `prev_dt_in`
     - `prev_theta_final_in`
     - `prev_newton_iter_count_in`
     - `lagged_response_valid_in`
     - `lagged_reference_y_in`
   - optional second-pass fields if needed:
     - `rhs_time_ref`
     - `jacobian_out`
     - exact reduced stage-linear-solve artifacts

Implementation note:

- do not reuse the old checkpoint/replay trace dataclasses for this
- the new reverse payload should represent the accepted-step primitive directly

### Step 2. Add one canonical reduced-output builder

Add a helper that converts a primal accepted-step result into the new reduced
output object.

Recommended helper:

- `_radau_accepted_step_reduced_output_from_primal(...)`

Inputs:

- `carry_in`
- `attempt_result`
- any required projected `accepted_y`

Output:

- `_RadauAcceptedStepReducedOutput`

Purpose:

- make both forward and reverse talk about the same explicit accepted-step
  output object
- stop depending on ad hoc conversions through `_RadauReplayState`

### Step 3. Add one canonical reverse-payload builder

Add a helper:

- `_radau_accepted_step_reverse_payload_from_primal(...)`

This helper should sit immediately next to the primitive boundary, meaning:

- it should consume the exact primal accepted-step result from
  `_execute_radau_accepted_step_attempt(...)`
- it should save the exact local data the reverse will use later

Important rule:

- build this payload at the primal accepted-step boundary
- do not reconstruct it later from rollout replay/checkpoint logic

### Step 4. Define the primitive forward object explicitly

Add a single accepted-step primitive helper that represents the differentiated
object.

Recommended helper:

- `_radau_accepted_step_primitive(...)`

Inputs:

- accepted-step carry
- attempt context
- solver/physics context

Outputs:

- reduced accepted-step output
- accepted-step reverse payload
- optionally the next carry if convenient for rollout composition

Important rule:

- this helper defines the accepted-step primitive boundary for reverse
- do not make rollout reverse depend on broader replay objects than this helper

### Step 5. Re-express the local reverse on the primitive

Replace the current hybrid local replay-state pullback:

- `_radau_apply_accepted_step_replay_state_pullback_linearized(...)`

with a cleaner primitive reverse rule, for example:

- `_radau_accepted_step_primitive_pullback(...)`

Inputs:

- accepted-step reverse payload
- reduced accepted-step output cotangent

Outputs:

- reduced input cotangent for the accepted-step primitive

What this means concretely:

- the payload becomes the source of exact `trial_dt` / `stage_history`
- stop rebuilding ambiguous local primal context inside the reverse path
- stop using rollout replay/checkpoint structure to discover local reverse data

Temporary migration rule:

- the old `_radau_apply_accepted_step_replay_state_pullback_linearized(...)`
  may remain while bringing up the new primitive reverse
- but the new primitive reverse should become the only authoritative local
  reverse once validated

### Step 6. Keep the forward AD path unchanged

Do not refactor these during the reverse redesign:

- `_execute_radau_accepted_step_attempt_autodiff(...)`
- `_execute_radau_accepted_step_attempt_autodiff_jvp(...)`
- `_radau_adaptive_final_y_realized_schedule_jvp(...)`

Allowed change:

- at most, wrap or mirror their reduced output naming so reverse compares
  against the same explicit accepted-step object

Not allowed in this phase:

- changing forward tangent mathematics
- changing accepted-step forward AD contract
- forward “optimizations”

### Step 7. Build a new rollout reverse from saved per-step payloads

Introduce a new rollout-level reverse path that uses:

- primal accepted-step schedule
- primal saved accepted-step reverse payloads

Recommended helper family:

- `_radau_collect_accepted_step_reverse_payloads(...)`
- `_radau_rollout_reverse_from_saved_payloads(...)`

Design:

1. primal forward pass produces:
   - accepted-step schedule
   - accepted-step reverse payload for each accepted step

2. backward pass:
   - walks accepted steps backward
   - applies `_radau_accepted_step_primitive_pullback(...)`

Key point:

- this rollout reverse should not need the current replay/checkpoint carry
  reconstruction machinery for correctness

### Step 8. Treat the current replay/checkpoint reverse as legacy

The following current functions should be treated as legacy scaffolding that can
be retired once the new path is validated:

- `_radau_replay_realized_accepted_carry_trace(...)`
- `_radau_replay_realized_accepted_carry_pullback(...)`
- `_radau_replay_realized_attempt_checkpoint_carries(...)`
- `_radau_adaptive_final_y_realized_schedule_vjp_fwd(...)`
- `_radau_adaptive_final_y_realized_schedule_vjp_bwd(...)`

Migration suggestion:

1. keep them behind a temporary fallback/debug switch
2. add the new saved-payload reverse path beside them
3. once reverse matches forward AD reliably, delete or retire the legacy path

### Step 9. Validation order in code

Implement tests/checks in this order:

1. one accepted-step primitive check
   - compare primitive reverse against a tiny trusted local reference if feasible

2. short accepted-step rollout check
   - compare new reverse vs current forward AD

3. production benchmark check
   - `benchmark_transport_profile_vector_ad_compare.py --ad-mode both`

Only after these pass:

4. remove old reverse replay path

### Step 10. Explicit non-goals for this refactor

Do not spend refactor time on:

- more checkpoint-interval tuning
- more reduced checkpoint-trace packing schemes
- replaying rejected-step differentiation
- forward-path performance changes

These can all wait until the primitive reverse exists and matches forward AD.
