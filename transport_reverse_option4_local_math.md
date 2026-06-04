# Transport Reverse Option 4: Local Accepted-Step Math

This note fixes the intended local reverse target for the transport option-4
refactor.

## Contract

We differentiate the composition of **accepted steps only**, along the
accepted-step schedule fixed by the primal forward pass.

For one accepted step we treat the primal step as the AD primitive:

- input reduced state:
  - `t_n`
  - `y_n`
  - `h_n`
  - `Z_prev`
  - `h_prev`
  - `y_lag_prev`
  - `theta_prev`
- fixed support metadata:
  - lagged-response valid flag
  - any nondifferentiated reuse/controller bookkeeping
- output reduced state:
  - `t_{n+1}`
  - `y_{n+1}`
  - `h_{n+1}^{accepted}`
  - `Z_n`
  - `h_n`
  - `y_lag_n`
  - `theta_n`

The reverse rule should return cotangents only for the reduced input state,
not for replay-only cache leaves such as LU factors, Jacobian caches, or
lagged-response cached objects themselves.

## Forward Reduced Algebra

The current forward accepted-step custom-JVP is organized around:

- accepted-state tangent inputs:
  - `dy`
  - `dh`
  - `dlag`
- stage linearized solve:
  - `dZ = A^{-1} rhs(dy, dh, dlag)`
- accepted state tangent:
  - `dy_next = dy + dh (b^T Z) + h (b^T dZ)`

with:

- `Z` = converged stage history of the accepted step
- `A` = local stage linear operator frozen at the primal accepted step
- `b` = Radau output weights
- `h` = accepted trial step size

The accepted-step reduced outputs are then:

- `t_out = t_in + h`
- `y_out = P(trial_y)`
- `dt_out = h`
- `prev_stages_out = Z`
- `prev_dt_out = h`
- `lagged_reference_y_out =`
  - `lagged_reference_y_in` if lagged-response reuse is active
  - `y_in` otherwise
- `prev_theta_final_out = theta_final`

## Minimal Reverse Structure

Let the reduced output cotangent be:

- `t_bar_out`
- `y_bar_out`
- `h_bar_out`
- `Z_bar_out`
- `hprev_bar_out`
- `ylag_bar_out`
- `theta_bar_out`

The reverse rule should be built as:

1. Pull back `y_bar_out` through the accepted-step tangent map only.
2. Add direct reduced-output contributions:
   - `t_bar_in += t_bar_out`
   - `h_bar_total += h_bar_out + hprev_bar_out`
   - `Z_bar += Z_bar_out`
   - `theta_bar_in += theta_bar_out`
   - `ylag_bar_direct` from the lagged-reference output branch
3. Combine all contributions into the reduced input cotangent.

## Accepted-y Pullback

For the linearized accepted-step map:

- `dy_next = dy + h (b^T dZ) + dh (b^T Z)`

the transpose equations are:

- `trial_y_bar = P'^T y_bar_out`
- `dZ_bar = h b \otimes trial_y_bar`
- `rhs_bar = A^{-T} vec(dZ_bar)`
- `dy_bar_from_state = trial_y_bar + sum_s rhs_bar[s] J_s`
- `dh_bar_from_state = <trial_y_bar, b^T Z> + <rhs_bar, source_h>`

where:

- `J_s` is the frozen Jacobian action already present in the current solver
  approximation
- `source_h` is the local stage RHS sensitivity to `h`

This is exactly the reduced reverse counterpart of the current forward custom
JVP algebra, and it should remain the backbone of the option-4 local custom
rule.

## Lagged-Response Dependence

The accepted-state tangent also depends on the lagged-response tangent:

- `dlag_out =`
  - `dlag_in` if reuse is active
  - `B_y dy` otherwise
- `lagged_eval_tangent = E_lag dlag_out`

So the transpose contribution from the stage solve gives a lagged-response
cotangent:

- `lag_bar = E_lag^T rhs_bar`

Then this must be pushed back to the reduced input:

- reuse branch:
  - contributes to `ylag_bar_in` only through the cached lagged reference path
- rebuild branch:
  - contributes to `y_bar_in` through the rebuild map

Important:

- an efficient option-4 rule should **not** propagate cotangents for the
  cached lagged-response object itself, because that object is not part of the
  reduced input contract
- it should only push that dependence back to:
  - `y_in`
  - or `lagged_reference_y_in`

Current reduced implementation target:

- first compute `lag_bar = E_lag^T rhs_bar` inside the accepted-`y` pullback
- then apply a separate builder pullback
  - `ylag_bar_in += B_ref^T lag_bar` in the reuse branch
  - `y_bar_in += B_y^T lag_bar` in the rebuild branch

The backward payload should also carry the exact frozen local linearization
data needed by the accepted-`y` pullback:

- `trial_y`
- `trial_dt`
- `stage_history`
- `J`
- transposed-stage-solve factors / pivots

This lets reverse use the saved local accepted-step payload directly instead of
rerunning the accepted-step primal attempt inside backward.

This is better than differentiating the composed closure

- `flat_y -> build_lagged_response(flat_y) -> stage_eval(lagged_response)`

inside the primitive pullback, because the stage-evaluation transpose and the
builder transpose are then handled as two smaller objects instead of one large
nested reverse problem.

## Why The Current Direct Rule Is Still Heavy

Even after removing the old replay-state helper from the active primitive
pullback, the current local rule can still be too expensive under JIT if it
contains nested transformed subproblems:

- pullback through `P(trial_y)` via `jax.vjp`
- pullback from lagged-response dependence to flat state via a composed
  stage-eval-plus-builder `jax.vjp`

The first reduction already implemented is:

- explicit transpose for the output projection `P`

The next reductions implemented are:

- separate builder pullback for lagged-response reconstruction, so the
  primitive rule consumes `lag_bar` directly instead of reopening the stage
  evaluation closure
- saved local linearization payload for backward, so the primitive pullback no
  longer reruns the accepted-step primal attempt

So the current implementation is:

- structurally closer to option 4
- but not yet the final efficient custom rule

## Next Efficiency Targets

The next local reductions should focus on replacing nested `jax.vjp` calls
inside the primitive pullback with smaller explicit operators where possible.

Priority:

1. keep explicit output projection pullback
   - this is already the right direction and should remain
2. finish lagged-response builder pullback split
   - use `lag_bar` from the accepted-`y` pullback
   - then apply builder-only pullback to `y_in` or `lagged_reference_y_in`
3. keep the stage transpose itself
   - this is the right local linear algebra core of the rule
4. only after the builder split, consider response-type-specific builder
   pullbacks
   - these may reduce the remaining fallback `vjp` further if needed

## Current Status

- old benchmark-facing rollout reverse path: removed
- new primitive pullback: direct, no longer calls the old replay-state helper
- explicit output projection transpose: implemented
- lagged-response transpose split:
  - stage-eval transpose to `lag_bar`: implemented
  - builder-only pullback from `lag_bar` to reduced input: implemented
- backward payload now carries saved local linearization data, so the primitive
  rule no longer reruns `_execute_radau_accepted_step_attempt(...)`
- remaining blocker: the builder pullback still uses a generic fallback path in
  the transport-equations layer, and that may still dominate the one-step JIT
  graph
