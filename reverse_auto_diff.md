## Reverse Auto-Diff Plan

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
