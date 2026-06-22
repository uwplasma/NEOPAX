## Reverse Auto-Diff Plan

## 2026-06-22 Resume Note: Forward AD Reference Recovered

Before resuming reverse AD work, use the recovered forward-AD lane as the
primary reference.

Validated command:

```bash
python ./examples/benchmarks/benchmark_transport_forward_ad_only.py \
  --ntx-exact-derivative-mode direct \
  --parameter T0 \
  --radau-jacobian-reuse-mode legacy
```

Recovered forward-AD values:

```text
softmax_Er                         -2.160399e+01
smooth_root_proxy                   2.070900e-05
Er2_volume_average                 -2.765750e+01
Er_volume_average                   2.291385e+00
electron_temperature_volume_average 3.571291e-01
total_pressure_volume_average       1.835267e+00
alpha_power_volume_average          7.221955e-02
```

The forward-AD fix was to use the compact adaptive schedule rollout inside
`_radau_adaptive_final_y_realized_schedule_jvp(...)`, not the payload-heavy
rollout. Reverse recovery should not disturb this recovered forward path.

Recommended order from here:

1. finish/validate the fused forward-AD optimization
2. then recover reverse AD against the forward-AD values above
3. use frozen FD only as a secondary sanity check

Reverse AD remains a separate lane and should not reuse or modify forward/FD
helpers in ways that change their recovered behavior.

### Goal
Make the transport reverse path mirror the same accepted-step AD contract that forward mode already uses:
- same primal replay path
- same lagged-response reuse/rebuild branch
- same active accepted-step boundary

The main current mismatch is that forward mode treats lagged response in a compressed, branch-aware way, while reverse mode is still trying to VJP overly large lagged-response objects.

### Current diagnosis

#### Forward mode already treats lagged response explicitly
In `NEOPAX/_transport_solvers.py`, forward mode does not differentiate the lagged response as one giant black-box object.

It splits the effect into two pieces:
- lagged-response rebuild or reuse:
  - if `lagged_response_valid` is `True`, reuse `dlagged_response_cache`
  - if `False`, rebuild via `jax.jvp(build_lagged_response, y, dy)`
- lagged-response evaluation effect:
  - tangent through `evaluate_with_lagged_response(...)` is represented by `lagged_eval_tangent`

This means the correct reverse path must mirror the same compressed contract, not generic VJP through the full cached object.

#### Reverse mode is currently too generic
The current reverse implementation in `NEOPAX/_transport_solvers.py` uses generic VJPs like:
- `jax.vjp(_stage_evals_from_lagged, lagged_response)`
- `jax.vjp(_build_from_flat, carry_in.y)`

This is structurally too broad and is the most likely cause of the large memory blowup when the lagged-response adjoint is restored.

#### NTX already has custom derivative machinery
In `NTX/src/ntx/_solver_prepared.py`, NTX provides:
- `solve_prepared_coefficient_vector_vjp(...)`

with explicit adjoint support in:
- `NTX/src/ntx/_solver_adjoint.py`

and matching tests in:
- `NTX/tests/test_solver.py`

So for NTX exact mode, the reverse path should use this custom derivative lane instead of relying on raw direct reverse AD through the full prepared solve.

### Plan

#### 1. Lock the reverse contract
Reverse must mirror the same accepted-step AD contract as forward:
- same primal replay path
- same reuse/rebuild branch on `lagged_response_valid`
- same active boundary:
  - `y`
  - `dt`
  - `prev_stages`
  - `lagged_reference_y`
  - lagged-response-cache contribution only through the compressed forward model

#### 2. Split lagged-response reverse by branch
Do not use one generic `jax.vjp` over the whole `lagged_response`.

Handle separately:
- cache reused
- cache rebuilt

#### 3. Rebuild branch: use model-specific build pullback
For rebuild, reverse the forward `build_lagged_response(state)` contract, not `_stage_evals_from_lagged` as one giant object.

For NTX exact mode:
- force `ntx_exact_derivative_mode = custom_vjp`
- use the NTX prepared coefficient custom-VJP path for rebuild contributions

#### 4. Reuse branch: implement reduced pullbacks for lagged response types
Add explicit pullbacks for the cached response types actually used by forward mode:
- `NTXInterpolatedMomentResponse`
- `NTXPreparedCoefficientResponse`
- `JVPTransportFluxResponse`
- `CombinedTransportLaggedResponse`

These pullbacks should return only:
- cotangent to the cached response object
- cotangent to `lagged_reference_y` and/or `y` where appropriate

They should avoid VJP-ing the whole cached object through stage evaluation.

#### 5. Implement the NTX interpolated branch first
Start with `NTXInterpolatedMomentResponse`, because it is the cleanest high-value target.

Its forward map is affine in the cached fields, so the reverse should be cheap and explicit.

This is the best first target for eliminating the worst memory blowup on the reused NTX lagged-cache path.

#### 6. Add NTX prepared/exact response pullback next
For the non-interpolated NTX cached response:
- use the NTX prepared coefficient custom-VJP where possible
- avoid generic VJP over the full radius/species cached response tree

#### 7. Make combined lagged-response reverse recursive
For `CombinedTransportLaggedResponse`, reverse should dispatch model-by-model:
- neoclassical
- turbulent
- classical

and sum their cotangents, instead of treating the combined object as one black-box pytree.

#### 8. Keep the accepted-step outer reverse structure
Do not redesign checkpointing again yet.

Keep:
- segmented replay
- reverse `fori_loop`
- explicit stage-solve transpose

Only replace the lagged-response pullback block inside:
- `_radau_apply_accepted_step_replay_state_pullback_linearized(...)`

#### 9. Validate in the smallest useful order
Use the smallest useful benchmark first:

```bash
NEOPAX_TRANSPORT_REVERSE_REPLAY_OUTPUT=y python ./examples/benchmarks/benchmark_transport_profile_vector_ad_compare.py --ntx-exact-derivative-mode direct --ad-mode reverse --objective-indices 0
```

Then compare that one reverse row against the corresponding forward row.

Only after the numbers look sane:
- remove `NEOPAX_TRANSPORT_REVERSE_REPLAY_OUTPUT=y`
- then expand to more objective rows

#### 10. Avoid broad tests until lagged-response reverse is model-aware
The next work should focus on correctness of the lagged-response reverse contract, not more global replay refactors.

### Implementation order
1. Add a lagged-response pullback dispatch helper by response type.
2. Implement `NTXInterpolatedMomentResponse` reverse.
3. Implement `CombinedTransportLaggedResponse` recursive dispatch.
4. Hook that dispatch into `_radau_apply_accepted_step_replay_state_pullback_linearized(...)`.
5. Then evaluate whether the NTX rebuild branch also needs stronger forced `custom_vjp` plumbing in the exact-mode setup.

### Architectural conclusion
The right architecture is:
- checkpointed exact discrete outer replay
- plus model-aware local reverse rules for lagged-response objects

not:
- full generic reverse AD through the entire lagged-response cache

This is also the closest match to what forward mode is already doing.

### Current Status: Later-Step Payload `nan` Bug Is Fixed

The later-step diagnostic established two things:

1. the earlier one-step dynamic JIT OOM was narrowed to passing
   `lagged_response_valid_in` as a dynamic JIT argument
2. the later-step `nan` bug came from the old payload collector design, not
   from the accepted-step reverse rule itself

The old collector did:

- adaptive schedule rollout first
- then separate accepted-only replay to rebuild payloads

That replay path produced corrupted payloads even though the true adaptive
forward trace at the same accepted step stayed finite.

The fix was to rewrite:

- `_radau_collect_realized_accepted_step_payloads(...)`

so it now walks the **real adaptive attempt loop** and records accepted-step
primitive payloads directly from the exact attempt context used by forward
mode, instead of reconstructing them in a second accepted-only replay.

### What Is Now Confirmed

The diagnostic command:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_one_step_primitive.py --ntx-exact-derivative-mode direct --execution-mode jit --payload-mode closed-over --payload-source last-from-rollout --rollout-accepted-step-limit 128 --rollout-max-total-steps-multiplier 4 --payload-capture-device cpu
```

now returns finite values for the later-step one-step pullback, for example:

- `primitive_dt_bar_abs = 2.401439e+03`
- `primitive_prev_stages_bar_max = 1.311968e-02`
- `primitive_y_bar_max = 1.385477e+02`

and the saved payload is now finite:

- `first_nonfinite_payload_step = None`
- `selected_payload_nonfinite_leaves = []`

So the later-step payload `nan` bug is fixed.

### Root Cause That Was Confirmed

At the previously first bad accepted step:

- `accepted_index = 91`
- `trace_index = 149`
- `dt = 4.343504268706003e-05`

the comparison showed:

- payload rebuilt from the **true adaptive forward trace carry** was finite
- payload produced by the **old replay collector** was nonfinite

for the unstable fields:

- `accepted_y`
- `trial_y`
- `stage_history`
- `jacobian_out`
- `real_lu_out`
- `complex_lu_out`

This confirmed that the bug was the replay-collector context mismatch, not the
forward/primal accepted-step map itself.

### Current State Of The Refactor

Resolved enough:

- one-step dynamic JIT OOM is reduced by branch-specializing
  `lagged_response_valid_in`
- later-step payload `nan` bug is fixed by collecting payloads from the true
  adaptive attempt path

Still open:

- the full many-step reverse memory/composition problem
- final GPU-feasible rollout reverse architecture

### Next Steps

The CPU-heavy `last-from-rollout` probe has done its job and should no longer be
the main test path.

From here, the next tests should be GPU-only rollout-composition checks, for
example:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_multi_step_primitive.py --ntx-exact-derivative-mode direct --accepted-step-counts 8,16,32,64 --execution-mode jit --max-total-steps-multiplier 1 --segment-length 8 --checkpoint-count 0
```

Then, if scaling looks sane:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_multi_step_primitive.py --ntx-exact-derivative-mode direct --accepted-step-counts 128 --execution-mode jit --max-total-steps-multiplier 1 --segment-length 8 --checkpoint-count 0
```

The active question again is now:

- does the corrected payload path compose over many accepted steps without
  reopening the rollout-level OOM?

### Latest confirmed findings

#### 1. Reuse-only narrowed reverse removes the giant OOM
Command:

### Contract refinement after the first forward-like reverse passes

The first two forward-like reverse contracts were useful, but they still kept
one important mismatch with the forward accepted-step tangent rule:

- forward tangent input contract is driven by:
  - `dy`
  - `dh`
  - `dlagged_response_cache`
- while the reduced reverse contracts still propagated:
  - `lagged_reference_y`

That is not the cleanest transpose of the reduced accepted-step map. It keeps
the reverse path tied to `build_lagged_response_pullback(...)` and
`lagged_reference_y` propagation more than forward mode does.

The next contraction pass is therefore:

- `forward-like-v3-cache-no-stage`

with propagated reverse state:

- `y`
- `dt`
- `lagged_response_cache`

and no propagated stage-history lane.

This is closer to the forward accepted-step boundary because rebuild
contributions from lagged response stay local to the step where they happen,
while reuse contributions propagate through the compressed lagged-cache lane
instead of through `lagged_reference_y`.

This does **not** yet prove the multi-step JIT OOM is solved, but it is the
correct next structural pass because it removes a remaining contract mismatch
instead of adding another diagnostic workaround.

```bash
NEOPAX_TRANSPORT_REVERSE_REPLAY_OUTPUT=y NEOPAX_TRANSPORT_REVERSE_REUSE_ONLY=1 python ./examples/benchmarks/benchmark_transport_profile_vector_ad_compare.py --ntx-exact-derivative-mode direct --ad-mode reverse --objective-indices 0
```

Result:
- memory drops to about `4.23 GiB`
- run completes
- reverse values are still absurd:
  - `n0: -8.216707e+31`
  - `T0: 5.389379e+31`
  - `density_shape_power: -9.917097e+29`
  - `temperature_shape_power: 1.692342e+32`

Conclusion:
- the rebuild-branch reverse is the dominant memory blocker
- reuse-only path is not the source of the giant OOM
- correctness is still wrong even when the OOM is removed

#### 2. Initial-carry leaf filter was added
In `examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py`:
- env:
  `NEOPAX_TRANSPORT_REVERSE_INITIAL_CARRY_LEAF`

This filters the final contraction
`carry0_tangent • carry0_bar`
to a single initial-carry leaf.

#### 3. `y` leaf alone reproduces the absurd values
Command:

```bash
NEOPAX_TRANSPORT_REVERSE_REPLAY_OUTPUT=y NEOPAX_TRANSPORT_REVERSE_REUSE_ONLY=1 NEOPAX_TRANSPORT_REVERSE_INITIAL_CARRY_LEAF=y python ./examples/benchmarks/benchmark_transport_profile_vector_ad_compare.py --ntx-exact-derivative-mode direct --ad-mode reverse --objective-indices 0
```

Result:
- same absurd values as the full reuse-only contraction

Conclusion:
- the bad signal is already in the `y` contribution

#### 4. `lagged_response_cache` contraction is zero
Command:

```bash
NEOPAX_TRANSPORT_REVERSE_REPLAY_OUTPUT=y NEOPAX_TRANSPORT_REVERSE_REUSE_ONLY=1 NEOPAX_TRANSPORT_REVERSE_INITIAL_CARRY_LEAF=lagged_response_cache python ./examples/benchmarks/benchmark_transport_profile_vector_ad_compare.py --ntx-exact-derivative-mode direct --ad-mode reverse --objective-indices 0
```

Result:
- all sensitivities printed as exactly zero

Conclusion:
- the absurd reverse values are not coming from the final
  `lagged_response_cache` carry contraction

#### 5. Local reuse-only accepted-step adjoint is consistent
Added env:

```bash
NEOPAX_TRANSPORT_REVERSE_LOCAL_ADJOINT_CHECK=1
```

This checks local one-step adjoint consistency at the accepted-step replay-state
boundary.

Command:

```bash
NEOPAX_TRANSPORT_REVERSE_REPLAY_OUTPUT=y NEOPAX_TRANSPORT_REVERSE_REUSE_ONLY=1 NEOPAX_TRANSPORT_REVERSE_LOCAL_ADJOINT_CHECK=1 python ./examples/benchmarks/benchmark_transport_profile_vector_ad_compare.py --ntx-exact-derivative-mode direct --ad-mode reverse --objective-indices 0
```

Result:
- `[autodiff-gate] local-adjoint-check lhs=1.074419e-01 rhs=1.074419e-01 abs_err=1.537659e-14`

Conclusion:
- the local reuse-only accepted-step `y -> accepted_y` pullback is internally
  consistent
- the core one-step local reverse is probably not the source of the
  `1e31` scale explosion

#### 6. Full rollout adjoint check is too expensive
Added env:
- `NEOPAX_TRANSPORT_REVERSE_ROLLOUT_ADJOINT_CHECK=1`
- optional `NEOPAX_TRANSPORT_REVERSE_ROLLOUT_ADJOINT_BASIS=<int>`

This attempts a full `carry -> final_y` adjoint consistency check via
`jax.jvp(_final_y_from_carry, ...)`.

Observed result:
- OOM from the diagnostic itself while building checkpoint carries in
  `_radau_replay_realized_checkpoint_carries(...)`

Interpretation:
- not a new solver regression
- rollout-level forward-mode probing through the replayed rollout is too
  expensive to use as the main next discriminator

#### 7. Cheaper next diagnostic already added
Added env:

```bash
NEOPAX_TRANSPORT_REVERSE_PARAMETER_CARRY_DIAGNOSTIC=1
```

This prints, for each parameter basis:
- `||carry0_bar.y||`
- `max(abs(carry0_bar.y))`
- `||carry0_tangent.y||`
- `max(abs(carry0_tangent.y))`
- `vdot(carry0_tangent.y, carry0_bar.y)`

Purpose:
- determine whether the bad scale is already in `carry0_bar.y`
- or whether the parameter-to-initial-carry `y` tangent is what makes the
  final scalar contraction explode

### Current best diagnosis

- Giant memory OOM:
  - caused by the rebuild-branch reverse being traced in the normal narrowed
    reverse
- Huge wrong gradients in reuse-only mode:
  - not due to `lagged_response_cache` final contraction
  - not obviously due to the local one-step accepted-step adjoint, since the
    local adjoint check passes to `1e-14`
  - likely one layer outward:
    - outer realized-schedule reverse accumulation
    - or parameter-to-initial-carry tangent/contraction layer

### Current next test

```bash
NEOPAX_TRANSPORT_REVERSE_REPLAY_OUTPUT=y NEOPAX_TRANSPORT_REVERSE_REUSE_ONLY=1 NEOPAX_TRANSPORT_REVERSE_LOCAL_ADJOINT_CHECK=1 NEOPAX_TRANSPORT_REVERSE_PARAMETER_CARRY_DIAGNOSTIC=1 python ./examples/benchmarks/benchmark_transport_profile_vector_ad_compare.py --ntx-exact-derivative-mode direct --ad-mode reverse --objective-indices 0
```

### New localization result

#### Parameter/carry diagnostic
Run with:

```bash
NEOPAX_TRANSPORT_REVERSE_REPLAY_OUTPUT=y NEOPAX_TRANSPORT_REVERSE_REUSE_ONLY=1 NEOPAX_TRANSPORT_REVERSE_LOCAL_ADJOINT_CHECK=1 NEOPAX_TRANSPORT_REVERSE_PARAMETER_CARRY_DIAGNOSTIC=1 python ./examples/benchmarks/benchmark_transport_profile_vector_ad_compare.py --ntx-exact-derivative-mode direct --ad-mode reverse --objective-indices 0
```

Result:
- `carry0_tangent.y` stays ordinary-sized for all parameter bases
- `carry0_bar.y` is already huge:
  - `carry0_bar_y_l2=1.463782e+33`
  - `carry0_bar_y_max=7.577226e+32`

Conclusion:
- the blow-up is in the rollout cotangent itself
- not in the parameter-to-initial-carry tangent map

#### Replay segment diagnostic
Added env:

```bash
NEOPAX_TRANSPORT_REVERSE_SEGMENT_DIAGNOSTIC=1
```

Run with:

```bash
NEOPAX_TRANSPORT_REVERSE_REPLAY_OUTPUT=y NEOPAX_TRANSPORT_REVERSE_REUSE_ONLY=1 NEOPAX_TRANSPORT_REVERSE_SEGMENT_DIAGNOSTIC=1 python ./examples/benchmarks/benchmark_transport_profile_vector_ad_compare.py --ntx-exact-derivative-mode direct --ad-mode reverse --objective-indices 0
```

Decisive result:
- for reverse segments `312` down through `3`, replay-state `y` cotangent stays
  exactly at:
  - `l2 = 1.0`
  - `max = 1.0`
- blow-up begins only in the first few reverse segments:
  - `seg=2`:
    - after: `l2=5.638296e+21`, `max=2.882411e+21`
  - `seg=1`:
    - after: `l2=4.333196e+26`, `max=1.972291e+26`
  - `seg=0`:
    - after: `l2=1.463782e+33`, `max=7.577226e+32`

Updated conclusion:
- the instability is highly localized
- it is not a diffuse accumulation over the whole replay
- the next debugging target should be the earliest reverse segments only,
  especially:
  - payload reconstruction in segment `0-2`
  - `_radau_replay_realized_accepted_carry_pullback(...)`
  - how replay-state bars are threaded across those first segments

### Most important current result

Run with:

```bash
NEOPAX_TRANSPORT_REVERSE_REPLAY_OUTPUT=y NEOPAX_TRANSPORT_REVERSE_REUSE_ONLY=1 NEOPAX_TRANSPORT_REVERSE_STEP_PULLBACK_CHECK=1 python ./examples/benchmarks/benchmark_transport_profile_vector_ad_compare.py --ntx-exact-derivative-mode direct --ad-mode reverse --objective-indices 0
```

Decisive result:

- `step-pullback-check seg=1 step=10 lhs=1.843063e+23 rhs=1.577166e+22 abs_err=1.685347e+23`

Interpretation:

- the remaining bug is in the local replay-step pullback itself at later carries
- it is not just outer replay composition
- the earlier initial-carry local adjoint check was too narrow and does not validate the problematic early replay steps
- the stage-solve transpose fix was real and should stay, but it was not the final bug

### Most important next steps

1. Treat the issue as a local replay-step pullback mismatch.
   - Main target:
     - `_radau_apply_accepted_step_replay_state_pullback_linearized(...)`
   - Proven failing site:
     - `seg=1 step=10`

2. Stop adding broad diagnostics unless they directly validate a patch.
   - The current step-level adjoint check is enough.
   - Use it as the main correctness probe.

3. Refactor the local replay-step reverse to mirror the forward tangent helper more literally.
   - Forward source of truth:
     - `_radau_accepted_step_y_tangent_from_primal_linearized(...)`
   - Reverse should be derived from that same reduced map, not maintained as separate handwritten algebra that can drift.

4. Split the local reverse into explicit submaps.
   - output projection pullback
   - accepted-`y` accumulation pullback
   - stage linear solve transpose
   - Jacobian/time-source accumulation pullback
   - lagged-response contribution pullback

5. Replace the current handwritten `dy_bar` / `dh_bar` assembly with the transpose of the same reduced forward tangent map.
   - The now-proven mismatch site is the local algebra after `stage_rhs_bar`.

6. Keep the reuse-only narrowed test while fixing correctness.
   - Continue using:
     - `NEOPAX_TRANSPORT_REVERSE_REPLAY_OUTPUT=y`
     - `NEOPAX_TRANSPORT_REVERSE_REUSE_ONLY=1`
     - `--objective-indices 0`

7. Primary validation after each patch:
   - rerun the same `STEP_PULLBACK_CHECK=1` command
   - success criterion:
     - `lhs` and `rhs` agree closely for `seg=1 step=10`

8. After the local replay-step pullback matches, re-check segment growth.
   - rerun:
     - `NEOPAX_TRANSPORT_REVERSE_SEGMENT_DIAGNOSTIC=1`
   - expectation:
     - the early-segment `y_bar` explosion should disappear or drop sharply

9. Only after reuse-only correctness is fixed, return to rebuild OOM work.
   - rebuild branch is still the memory blocker for the full reverse path
   - but it is not the current correctness blocker
## 2026-06-02 reverse AD latest state

- Active debug mode:
  - `NEOPAX_TRANSPORT_REVERSE_REPLAY_OUTPUT=y`
  - `NEOPAX_TRANSPORT_REVERSE_REUSE_ONLY=1`
  - row `objective_index=0`
- Do **not** widen out yet; rebuild/full reverse is still a separate memory issue.

### Summary

- Local accepted-step `dy/dh` adjoint math is correct in a host-side isolated check.
- Stage-solve transpose fix is still valid and should remain.
- Zeroing `lagged_response_cache_bar` had no effect on the blow-up.
- Therefore the remaining issue is most likely in **replay-state threading/mapping across steps**, not in local accepted-`y` algebra.

### Host-side isolated check

Command:

```bash
NEOPAX_TRANSPORT_REVERSE_HOST_STEP_PULLBACK_DIAGNOSTIC=1 NEOPAX_TRANSPORT_REVERSE_STEP_PULLBACK_SEGMENT=1 NEOPAX_TRANSPORT_REVERSE_STEP_PULLBACK_STEP=10 python ./examples/benchmarks/benchmark_transport_profile_vector_ad_compare.py --ntx-exact-derivative-mode direct --ad-mode reverse --objective-indices 0
```

Result:

- `lhs = rhs = 3.060000e+02`
- `dy_diff_l2 = 0`
- `dh_diff = 0`
- `stage_rhs_diff_l2 = 0`
- lagged terms all zero in this probe

Meaning:

- The frozen-lagged reduced local accepted-`y` pullback is correct.

### Cheap in-replay step-window diagnostic

Command:

```bash
NEOPAX_TRANSPORT_REVERSE_REPLAY_OUTPUT=y NEOPAX_TRANSPORT_REVERSE_REUSE_ONLY=1 NEOPAX_TRANSPORT_REVERSE_STEP_PULLBACK_CHECK=1 NEOPAX_TRANSPORT_REVERSE_STEP_PULLBACK_SEGMENT=1 NEOPAX_TRANSPORT_REVERSE_STEP_PULLBACK_STEP=10 NEOPAX_TRANSPORT_REVERSE_STEP_PULLBACK_WINDOW=2 python ./examples/benchmarks/benchmark_transport_profile_vector_ad_compare.py --ntx-exact-derivative-mode direct --ad-mode reverse --objective-indices 0
```

Important output:

- `seg=1 step=12 accepted_y_bar_l2=1.106325e+22`
- `seg=1 step=11 accepted_y_bar_l2=1.029448e+22`
- `seg=1 step=10 accepted_y_bar_l2=9.430782e+21`
- step check still fails:
  - `lhs=1.843063e+23`
  - `rhs=1.577166e+22`
  - `abs_err=1.685347e+23`
- blow-up then jumps:
  - `seg=1 step=9 accepted_y_bar_l2=3.927721e+23`
  - `seg=1 step=8 accepted_y_bar_l2=6.889240e+23`

Meaning:

- Real replay cotangent is already huge before step `10`.
- First sharp amplification within segment `1` occurs at step `10 -> 9`.

### Lagged-cache ablation

Command:

```bash
NEOPAX_TRANSPORT_REVERSE_REPLAY_OUTPUT=y NEOPAX_TRANSPORT_REVERSE_REUSE_ONLY=1 NEOPAX_TRANSPORT_REVERSE_STEP_PULLBACK_CHECK=1 NEOPAX_TRANSPORT_REVERSE_STEP_PULLBACK_SEGMENT=1 NEOPAX_TRANSPORT_REVERSE_STEP_PULLBACK_STEP=10 NEOPAX_TRANSPORT_REVERSE_STEP_PULLBACK_WINDOW=2 NEOPAX_TRANSPORT_REVERSE_ZERO_LAGGED_CACHE_BAR=1 python ./examples/benchmarks/benchmark_transport_profile_vector_ad_compare.py --ntx-exact-derivative-mode direct --ad-mode reverse --objective-indices 0
```

Result:

- No meaningful change in the step-window cotangent growth.

Meaning:

- `lagged_response_cache_bar` is not the culprit.

### Next session: immediate command to run

Run the same cheap replay-step window command again, now that replay-state leaf norms were added:

```bash
NEOPAX_TRANSPORT_REVERSE_REPLAY_OUTPUT=y NEOPAX_TRANSPORT_REVERSE_REUSE_ONLY=1 NEOPAX_TRANSPORT_REVERSE_STEP_PULLBACK_CHECK=1 NEOPAX_TRANSPORT_REVERSE_STEP_PULLBACK_SEGMENT=1 NEOPAX_TRANSPORT_REVERSE_STEP_PULLBACK_STEP=10 NEOPAX_TRANSPORT_REVERSE_STEP_PULLBACK_WINDOW=2 python ./examples/benchmarks/benchmark_transport_profile_vector_ad_compare.py --ntx-exact-derivative-mode direct --ad-mode reverse --objective-indices 0
```

Watch for:

- `dt_bar`
- `prev_dt_bar`
- `prev_theta_bar`
- `prev_stages_l2`
- `lagged_ref_l2`

Goal:

- detect whether a non-`y` replay-state leaf spikes first and is then feeding back into `accepted_y_bar`.

## 2026-06-03 reverse-AD update: local adjoint cleared, forward-recorded primal mismatch remains

### Current narrowed lab

We are still debugging under:

```bash
NEOPAX_TRANSPORT_REVERSE_REPLAY_OUTPUT=y NEOPAX_TRANSPORT_REVERSE_REUSE_ONLY=1
```

with:

```bash
python ./examples/benchmarks/benchmark_transport_profile_vector_ad_compare.py --ntx-exact-derivative-mode direct --ad-mode reverse --objective-indices 0
```

### What is no longer the main suspect

These have effectively been ruled out as the primary remaining bug:

- local accepted-step `dy/dh` adjoint algebra
- replay-branch post-processing after the local helper
- replay carry inputs:
  - `y`, `t`, `dt`
  - `prev_stages`
  - `lagged_reference_y`
  - `jacobian`
  - `real_lu`, `complex_lu`, pivots
  - cache flags / cache age / cache dt
  - `lagged_response_valid`
  - `prev_theta_final`
  - `prev_newton_iter_count`
  - `lagged_response_cache`

### Same-trace replay result

At `seg=1 step=10`, the cheap replay check now shows:

- `accepted_y_bar_l2 = 9.430782e+21`
- `lhs = 1.843063e+23`
- `rhs = 1.577166e+22`
- `abs_err = 1.685347e+23`
- `direct_dy_l2 = 3.927721e+23`
- `replay_dy_l2 = 3.927721e+23`
- `direct_vs_replay_dy_diff_l2 = 0`

Meaning:

- inside the same trace, the replay path and the direct local helper agree
- so the large local `dy_bar` is coming from the local helper call context used in replay, not from later mutation

### Host vs replay still disagree strongly

Host-side local diagnostic with replay-captured context still gives:

- `dy_manual_l2 = 9.430782e+21`
- `replay_dy_bar_l2 = 3.927721e+23`
- `replay_vs_host_dy_diff_l2 = 3.883529e+23`

So the host-local reconstruction and replay-local reconstruction are still not using the same effective local primal step, even after matching all carried input fields we compared.

### New critical clue

The newly added replay-primal comparison shows:

- `replay_primal_trial_dt_diff = 7.404452e-06`
- `replay_primal_stage_history_diff_l2 = 4.228100e+07`
- `replay_primal_stage_history_diff_max = 3.782329e+07`
- `replay_primal_jacobian_out_diff_l2 = 0`
- `replay_primal_rhs_time_ref_diff_l2 = 0`

This is now the most important result.

Interpretation:

- the remaining mismatch is localized to the **recomputed local primal step**, especially:
  - `trial_dt`
  - `stage_history`
- reverse appears to be differentiating through a local primal accepted-step reconstruction that does not exactly match the forward accepted step at this replay location

### Exact next direction

Use the host diagnostic with the replay captures:

```bash
NEOPAX_TRANSPORT_REVERSE_HOST_STEP_PULLBACK_DIAGNOSTIC=1 NEOPAX_TRANSPORT_REVERSE_STEP_PULLBACK_SEGMENT=1 NEOPAX_TRANSPORT_REVERSE_STEP_PULLBACK_STEP=10 NEOPAX_TRANSPORT_REVERSE_HOST_STEP_PULLBACK_Y_BAR_PATH=outputs/autodiff_transport_lagged_ntx/profile_vector/step10_y_bar.npy NEOPAX_TRANSPORT_REVERSE_HOST_STEP_PULLBACK_REPLAY_DY_BAR_PATH=outputs/autodiff_transport_lagged_ntx/profile_vector/step10_dy_bar_replay.npy NEOPAX_TRANSPORT_REVERSE_HOST_STEP_PULLBACK_REPLAY_INPUTS_PATH=outputs/autodiff_transport_lagged_ntx/profile_vector/step10_inputs_replay.npz NEOPAX_TRANSPORT_REVERSE_HOST_STEP_PULLBACK_REPLAY_LAGGED_CACHE_PATH=outputs/autodiff_transport_lagged_ntx/profile_vector/step10_lagged_cache_replay.pkl NEOPAX_TRANSPORT_REVERSE_HOST_STEP_PULLBACK_REPLAY_PRIMAL_PATH=outputs/autodiff_transport_lagged_ntx/profile_vector/step10_primal_replay.npz python ./examples/benchmarks/benchmark_transport_profile_vector_ad_compare.py --ntx-exact-derivative-mode direct --ad-mode reverse --objective-indices 0
```

Then inspect the newly added forward-recorded checks:

- `forward_recorded_prev_stages_diff_*`
- `forward_recorded_prev_dt_diff`
- `forward_recorded_next_y_diff_*`

If these are large, the likely fix is:

- do not rely on fresh recomputation of the local primal accepted step for this reverse path
- instead use forward-recorded step data, or an exact reconstruction from stored forward payloads, for the reverse local pullback context

## 2026-06-05 option-4 reverse status

### What is now working

- The new one-step accepted-step custom reverse rule works under JIT when the
  reverse payload is closed over as a residual-like constant.

Command:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_one_step_primitive.py --ntx-exact-derivative-mode direct --execution-mode jit --payload-mode closed-over
```

Observed result:

- compile plus execute about `2.89e+01 s`
- steady execute about `1.58e-03 s`
- no OOM

### What is now proven to fail

The same one-step custom reverse rule OOMs under JIT when the reverse payload
is passed as a dynamic runtime argument.

Command:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_one_step_primitive.py --ntx-exact-derivative-mode direct --execution-mode jit --payload-mode dynamic
```

Observed result:

- OOM at about `49.1 GiB`

### Main conclusion

The current blocker is now sharply localized:

- not checkpointing
- not segmenting
- not rollout composition by itself
- not the one-step local reverse algebra in the residual/closed-over form

The blocker is the **dynamic runtime reverse-payload contract** for the
accepted-step custom reverse rule.

### Multi-step benchmark conclusion

The segmented reverse benchmark was refactored so it no longer stores a full
accepted-step payload tape for the whole run. It now uses:

- one lightweight adaptive schedule rollout
- transient per-segment payload collection
- optional sparse checkpoints

However, the full run still fails because the first jitted one-step reverse
call in the multi-step harness is exactly the dynamic-payload form above.

So the next work is **not** another checkpoint tweak. It is reducing the
runtime dynamic payload contract.

### New targeted benchmark controls

`examples/benchmarks/benchmark_transport_reverse_one_step_primitive.py` now supports:

- `--payload-mode closed-over`
- `--payload-mode dynamic`
- `--payload-ablation none|stage|lagged|jacobian|lu|pivots`

and prints:

- `payload_total_bytes`
- grouped payload byte totals

### Immediate next tests

Run these in order:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_one_step_primitive.py --ntx-exact-derivative-mode direct --execution-mode jit --payload-mode dynamic --payload-ablation none
```

```bash
python ./examples/benchmarks/benchmark_transport_reverse_one_step_primitive.py --ntx-exact-derivative-mode direct --execution-mode jit --payload-mode dynamic --payload-ablation stage
```

```bash
python ./examples/benchmarks/benchmark_transport_reverse_one_step_primitive.py --ntx-exact-derivative-mode direct --execution-mode jit --payload-mode dynamic --payload-ablation lagged
```

```bash
python ./examples/benchmarks/benchmark_transport_reverse_one_step_primitive.py --ntx-exact-derivative-mode direct --execution-mode jit --payload-mode dynamic --payload-ablation jacobian
```

```bash
python ./examples/benchmarks/benchmark_transport_reverse_one_step_primitive.py --ntx-exact-derivative-mode direct --execution-mode jit --payload-mode dynamic --payload-ablation lu
```

```bash
python ./examples/benchmarks/benchmark_transport_reverse_one_step_primitive.py --ntx-exact-derivative-mode direct --execution-mode jit --payload-mode dynamic --payload-ablation pivots
```

### Next plan for reducing OOM

1. Identify which dynamic payload family triggers the `~49 GiB` compile blowup.
2. Treat that family as the main refactor target for the runtime reverse
   contract.
3. Reduce the dynamic payload so reverse mode follows the same narrowed active
   contract philosophy as forward mode.
4. Only after the one-step `dynamic` payload form works under JIT, return to
   segmented multi-step rollout scaling.

### 2026-06-05 update: single-family payload ablations did not move the OOM

The following one-step dynamic-payload ablations were tried:

- `--payload-ablation stage`
- `--payload-ablation jacobian`
- `--payload-ablation lu`

Observed result:

- all still OOM at essentially the same `~49.1 GiB`

Interpretation:

- the dynamic-payload blowup is not dominated by one obvious payload family
- it is more likely caused by the dynamic-payload calling convention or the
  overall dynamic pytree contract itself

### Revised immediate next tests

Stop spending time on more single-family zero ablations first.

Instead test dynamic-argument structure directly:

1. dynamic `reverse_payload`, but closed-over `reduced_output_bar`
2. partially dynamic payload:
   - close over most payload leaves
   - pass only one or a few small leaves dynamically
3. single-leaf dynamic probes

Purpose:

- determine whether any dynamic payload pytree triggers the bad compile
- or whether only large dynamic pytrees do

### 2026-06-08 update: forward tangent vs reverse cotangent contract audit

The current objective remains:

- differentiate only accepted steps
- keep reverse on the same accepted-step primal path as forward mode
- make the reverse accepted-step contract as close as possible to the forward
  accepted-step tangent contract
- reduce reverse memory by shrinking the propagated cotangent state rather than
  saving broader payloads

#### Accepted-step contract table

| Field / lane | Forward tangent status | Reverse cotangent status | Current mismatch | Refactor direction |
| --- | --- | --- | --- | --- |
| `y` / `y_out` | active through `dy` in `_radau_extract_tangent_inputs_from_carry(...)` | active as `y_out` in `_RadauAcceptedStepReducedOutput` | aligned | keep active |
| `dt` / `dt_out` | active through `dh` in `_radau_extract_tangent_inputs_from_carry(...)` | active as `dt_out` and also mixed into `t_out` / `prev_dt_out` bars | reverse is broader | keep only one propagated time-step cotangent lane if possible |
| `lagged_response_cache` | active through `dlagged_response_cache` in `_radau_extract_tangent_inputs_from_carry(...)` | not propagated as an outer replay-state cotangent; handled locally via payload and branch pullback | partially aligned | keep local branch pullback, avoid widening rollout state |
| `t` / `t_out` | not a first-class extracted tangent input; effect is folded into local tangent algebra through `dh` and `rhs_time_ref` | propagated as explicit `t_out` cotangent | reverse is broader | candidate to fold into `dt`/local algebra instead of propagating |
| `prev_stages` / `prev_stages_out` | not an extracted tangent input; forward carry-ablation/debug repeatedly audits whether this lane can be zeroed with limited effect | propagated as full `prev_stages_out` cotangent over the reverse rollout | reverse is broader and likely expensive | top candidate to contract or recompute locally |
| `prev_dt` / `prev_dt_out` | not an extracted tangent input; forward debug explicitly zeroes `prev_dt` in later-step carry ablations | propagated as full `prev_dt_out` cotangent over the reverse rollout | reverse is broader | candidate to contract into the main `dt` lane |
| `lagged_reference_y` / `lagged_reference_y_out` | not an extracted tangent input; forward debug repeatedly zeroes this lane and branch logic already compresses lagged handling into reuse vs rebuild | propagated as full `lagged_reference_y_out` cotangent over the reverse rollout | reverse is broader | top candidate to contract; prefer local branch handling |
| `prev_theta_final` / `prev_theta_final_out` | not an extracted tangent input; forward debug explicitly zeroes this lane | propagated as full `prev_theta_final_out` cotangent over the reverse rollout | reverse is broader | top candidate to drop from propagated cotangent |
| `prev_newton_iter_count` | not an extracted tangent input; forward debug explicitly zeroes this lane | not propagated as outer replay-state cotangent in reverse | aligned enough | keep local only |
| controller bookkeeping (`prev_error`, reject counters, cooldowns, etc.) | explicitly `stop_gradient` in `_radau_carry_with_forward_only_jvp_fields(...)` | not propagated as outer reverse-state cotangents | aligned | keep masked |
| Jacobian / LU / cache validity state | explicitly `stop_gradient` in `_radau_carry_with_forward_only_jvp_fields(...)` | not propagated as outer replay-state cotangents; used only as local payload for transpose solve | aligned | keep local only |

#### Main conclusion from the audit

The local reverse algebra is already reasonably close to the transpose of the
forward tangent algebra:

- both use the accepted-step linearization
- both branch on lagged reuse vs rebuild
- both avoid broad reverse AD through the raw Newton/LU internals

The remaining gap is the **propagated rollout-level contract**:

- forward extracts a very small tangent-input object:
  - `dy`
  - `dh`
  - `dlagged_response_cache`
- reverse still propagates a replay-state-shaped cotangent with seven lanes:
  - `t_out`
  - `y_out`
  - `dt_out`
  - `prev_stages_out`
  - `prev_dt_out`
  - `lagged_reference_y_out`
  - `prev_theta_final_out`

So the next refactor should target the reverse propagated cotangent state,
not the accepted-step primal path and not more broad payload-family ablations.

#### Refactor order implied by the audit

1. Introduce an explicit reverse cotangent contraction helper.
2. Add an experimental "forward-like" contraction mode that keeps only the
   lanes closest to the forward tangent contract.
3. Start by contracting:
   - `prev_theta_final_out`
   - `prev_dt_out`
   - `lagged_reference_y_out`
4. Then evaluate whether `prev_stages_out` can also be reduced or folded
   locally.
5. Keep the accepted-step-only reverse composition structure unchanged while
   tightening the propagated cotangent boundary.

### 2026-06-09 update: giant-graph OOM separated from local rebuild-pullback OOM

The latest benchmark passes isolated two distinct issues:

1. the old `segment-scan` reverse composition built a giant monolithic JIT
   graph and OOMed around `141 GiB`
2. after switching to the non-giant-graph `step-loop` composition, the
   remaining failure is a local later-step rebuild pullback that still OOMs
   around `49 GiB`

This means the outer composition diagnosis is complete enough for now. The
remaining hotspot is inside the local rebuild reverse rule.

#### What is now confirmed

- `step-loop` removes the catastrophic `141 GiB` scan-composition blowup
- the failing later reverse step can be isolated to the final segment of the
  `accepted_step_count=16` rollout
- the exact failing step is:
  - `step_index_within_segment = 7`
  - `lagged_valid_in = False`
  - `dt_in = trial_dt = 1.404447e-05`
- the incoming lagged-cache cotangent tree for the failing step is not much
  broader than the nearby passing step:
  - passing captured step `6`:
    - `lagged_valid_in = True`
    - `total_bytes = 30192`
    - `leaf_count = 12`
    - `non_none_leaf_count = 11`
    - `max_abs = 4.721807e+03`
  - failing captured step `7`:
    - `lagged_valid_in = False`
    - `total_bytes = 30192`
    - `leaf_count = 12`
    - `non_none_leaf_count = 11`
    - `max_abs = 5.675512e+03`

So the failure is not explained by a sudden broadening of the incoming
lagged-cache cotangent tree.

#### Strongest current proof

Using the temporary benchmark flag:

- `--rebuild-pullback-ablation zero-build-lagged`

the previously failing isolated captured next step now runs successfully, with:

- `compile_plus_execute_s = 2.881095e+00`
- `execute_s = 1.679732e-03`

This is strong evidence that the local hotspot is specifically the rebuild
contribution through:

- `physics_context.build_lagged_response_pullback(carry_in.y, combined_cache_bar)`

inside:

- `_radau_accepted_step_forward_like_cache_no_stage_pullback(...)`

in `NEOPAX/_transport_solvers.py`.

### Next implementation steps

The next work should stop focusing on benchmark plumbing and instead refactor
the rebuild branch itself.

1. Add a durable local debug split in the rebuild branch so the benchmark can
   separately report:
   - accepted-step transpose cost without rebuild pullback
   - rebuild pullback cost alone
2. Refactor `_radau_accepted_step_forward_like_cache_no_stage_pullback(...)`
   so the rebuild branch no longer calls the generic whole-object
   `build_lagged_response_pullback(...)` directly for the full lagged response.
3. Introduce a model-aware rebuild-adjoint hook for lagged-response building,
   analogous to the existing model-aware lagged-response reuse pullback
   direction.
4. Implement the first specialized rebuild rule for the transport lagged
   response types actually active in this benchmark.
5. Keep `forward-like-v3-cache-no-stage` as the active reverse contract while
   doing this, so contract changes and rebuild-adjoint changes remain separated.
6. Validate in this order:
   - isolated captured failing step
   - full final segment of the `accepted_step_count=16` rollout
   - full `accepted_step_count=16` step-loop run
   - only then resume larger-scale rollout composition experiments

### Architectural conclusion after this round

The remaining design target is:

- keep the narrowed accepted-step reverse contract
- keep the non-giant-graph reverse composition direction
- replace the generic rebuild lagged-response pullback with a compressed,
  model-aware rebuild adjoint

That is now the real blocker for the reverse accepted-step path.

## 2026-06-22 Status Update: Reverse Recovery After Forward/FD Lane Lessons

The current reverse-only benchmark interface has been reshaped to match the
natural reverse-mode contract:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --ntx-exact-derivative-mode direct \
  --objective softmax_Er \
  --accepted-step-limit 1 \
  --radau-jacobian-reuse-mode legacy
```

This means:

- one scalar objective is selected with `--objective`
- one reverse sweep should return gradients for all profile parameters:
  - `n0`
  - `T0`
  - `density_shape_power`
  - `temperature_shape_power`

### Current failure sequence

The first OOM happened while raw reverse AD differentiated through the initial
lagged-response construction:

```text
_reverse_objective_for_parameter_vector
  -> _initial_carry_from_state_with_static_setup
  -> physics_context.build_lagged_response(...)
```

A reverse-lane-only custom initial-carry VJP was added in
`benchmark_transport_reverse_ad_only.py` so the initial lagged-response
cotangent can use the existing model-aware:

```text
pullback_build_lagged_response(...)
```

instead of raw reverse AD through NTX initialization.

After that, the OOM moved to the active solve-level reverse path:

```text
_radau_adaptive_final_y_realized_schedule_vjp_bwd
  -> jax.vjp(_replay, carry0)
  -> _radau_replay_realized_accepted_rollout(...)
```

A narrow one-accepted-step special case was added:

```text
_radau_replay_first_realized_accepted_step(...)
```

so `accepted_step_limit == 1` does not reverse through the whole accepted replay
`lax.scan`.

The next OOM then moved into the local accepted-step custom-JVP transpose:

```text
_radau_accepted_step_attempt_tangent_from_primal(...)
  -> _lagged_output_tangent()
  -> jax.lax.cond(...)
```

The immediate cause is that JAX traces both lagged-response branches. Even when
the first accepted step reuses the initialized lagged cache, the rebuild branch
still traces:

```text
jax.jvp(build_lagged_response, ...)
```

which reopens the large NTX lagged-response graph.

### Important branch-policy correction

Do **not** force lagged rebuild in reverse.

Do **not** force lagged reuse in the general reverse path either.

The correct reverse rule must follow the primal accepted-step branch:

```text
if lagged_response_valid:
    use the reuse-branch pullback
else:
    use the rebuild-branch pullback
```

The temporary forced-reuse accepted-step wrapper is diagnostic-only for the
`accepted_step_limit == 1` smoke test, because the initialized carry should have
a valid lagged-response cache. It is not the intended general reverse
implementation and must not be used to validate later accepted steps where the
primal path may rebuild lagged response.

### 2026-06-22 forced-reuse smoke-test update

After the forced-reuse diagnostic removed the local NTX rebuild trace, the
one-step reverse smoke test reached the pullback but failed with an
`AssertionError` at:

```text
_radau_adaptive_final_y_realized_schedule_vjp_bwd
  -> pullback(final_y_bar)
```

The likely cause was a custom-JVP tangent/primal pytree mismatch for the
lagged-response cache in the diagnostic forced-reuse branch. A narrow structural
normalization helper was added:

```text
_radau_align_tangent_tree_to_primal(...)
```

and is used only when `force_lagged_response_reuse=True`. This does not change
the normal branch-aware lagged path and does not make forced reuse a valid
general reverse strategy.

The same smoke-test path was then tightened further: when
`force_lagged_response_reuse=True`, the accepted-step tangent now bypasses the
`lax.cond` that chooses the exact lagged-cache tangent branch. This prevents the
unused exact branch from being traced/transposed during the diagnostic one-step
reverse run.

### Next correct implementation step

Replace the lagged-response tangent/reverse handling inside the accepted-step
reverse path with a branch-aware implementation that does not trace the heavy
opposite branch.

The intended design is:

- reuse branch:
  - propagate cotangent through the cached lagged response / lagged reference
    path
- rebuild branch:
  - call the model-aware `pullback_build_lagged_response(...)`
  - avoid raw `jax.jvp(build_lagged_response, ...)` or generic VJP through the
    full NTX object

This is the direct reverse analogue of what was relearned from FD and forward
AD recovery:

- production primal adaptive solve determines the accepted schedule and branch
  decisions
- the differentiated map follows accepted steps only
- branch/control decisions are primal metadata
- local derivative rules must be specialized enough that unused heavy branches
  are not traced into the compiled reverse graph
