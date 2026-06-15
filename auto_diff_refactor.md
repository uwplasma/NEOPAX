# Forward AD Refactor Plan

## Goal

Restore the forward transport AD benchmark lane to the older intended
accepted-step-only custom-JVP behavior, while keeping it independent from the
reverse and FD lanes.

This note starts with **Step 1: audit** of the current code path.

## Step 1 Audit: Current forward AD lane

### Scalar benchmark entrypoint

Current scalar benchmark:

- `examples/benchmarks/benchmark_transport_adaptive_ad_vs_frozen_fd.py`

AD-only path:

1. build benchmark config with `_prepare_benchmark_config(...)`
2. build `runtime, baseline_state = build_runtime_context(config)`
3. define:
   - `objective_fn = _forward_benchmark_adaptive_rollout_objectives_realized_schedule_only_for_parameter(...)`
4. execute:
   - `jax.jvp(objective_fn, ...)` when `--adaptive-derivative-mode jvp`
   - `jax.jacrev(objective_fn)` only when `--adaptive-derivative-mode vjp`

Important:

- `--replay-mode` does **not** affect the AD lane
- after the recent cleanup, `--run-mode ad` skips:
  - baseline FD trace generation
  - frozen replay trace construction
  - FD minus/plus replay

### Current forward helper used by scalar AD

Forward AD benchmark helper:

- `examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py`
- `_forward_benchmark_adaptive_rollout_objectives_realized_schedule_only_for_parameter(...)`

That helper:

1. calls `_forward_benchmark_prepare_realized_schedule_scalar_rollout(...)`
2. which:
   - constructs `state0`
   - constructs `state0_static = stop_gradient(parameter_value)`
   - prepares solver/runtime components from `state0_static`
   - rebuilds a differentiated `initial_carry` with
     `_initial_carry_from_state_with_static_setup(...)`
3. then, in `jvp` mode, calls:
   - `_radau_adaptive_final_y_realized_schedule(...)`

### Current solver-level custom-JVP path

Solver function:

- `NEOPAX/_transport_solvers.py`
- `_radau_adaptive_final_y_realized_schedule(...)`

Its custom JVP:

- `_radau_adaptive_final_y_realized_schedule_jvp(...)`

Current behavior:

1. run a **full primal adaptive rollout** with:
   - `_radau_adaptive_final_state_rollout(...)`
2. record the full attempt trace
3. build a replay tangent lane from that trace

The primal adaptive trace currently records:

- `active_mask`
- `accepted_mask`
- `attempted_dts`
- `next_dts`
- `next_recent_reject_count`
- `next_regrowth_cooldown`
- `next_easy_growth_streak`
- `next_lagged_response_valid`

### What the custom JVP currently freezes

Inside `_radau_adaptive_final_y_realized_schedule_jvp(...)`:

- `active_mask` is reduced to:
  - `logical_and(trace.active_mask, trace.accepted_mask)`
- but the following arrays are still frozen from the full primal adaptive run:
  - `attempted_dts`
  - `next_dts`
  - `next_recent_reject_count`
  - `next_regrowth_cooldown`
  - `next_easy_growth_streak`
  - `next_lagged_response_valid`

These are passed into:

- `_radau_replay_realized_accepted_rollout(...)`

### Why the current lane is not a clean accepted-step-only AD lane

The replay function used by the current custom JVP is:

- `NEOPAX/_transport_solvers.py`
- `_radau_replay_realized_accepted_rollout(...)`

This replay is accepted-attempt-only in one sense:

- only entries marked active in the accepted mask perform a differentiated step

However, each accepted replay step still writes frozen controller-history data
from the primal attempt trace back into the carry:

- `dt = next_dt_value`
- `recent_reject_count = recent_reject_count_value`
- `regrowth_cooldown = regrowth_cooldown_value`
- `easy_growth_streak = easy_growth_streak_value`
- `lagged_response_valid = lagged_response_valid_value`

So the current forward tangent lane:

- does **not** directly differentiate through rejected attempts
- but **does** depend on frozen metadata produced by the full primal adaptive
  attempt history

This is weaker than the intended benchmark contract:

- "differentiate only through the realized accepted-step physical map"

### Runtime implication

The current AD JVP path is expensive because it does:

1. a full adaptive primal rollout
2. then a second accepted-step replay pass for the tangent

So even though the tangent replay is restricted to accepted attempts, the AD
lane still pays for the full primal adaptive attempt logic first.

This is consistent with the observed runtime being much heavier than a minimal
accepted-step-only custom-JVP benchmark should be.

### Historical reference from previous review

The pre-reverse reference boundary already recorded in `auto_diff.md` is:

- clean pre-reverse commit:
  - `e888c02ba85b6ad790730d6c4e66d0bbd77f1cee`
- first checked reverse-era commit:
  - `412dfae7f552bd5ee51357acbb51d8506cd26ea1`

The previously recorded interpretation from that history is:

- in `e888c02...`, the forward custom-JVP lane used a lighter
  accepted-step replay contract
- it was closer to:
  - accepted-mask / accepted-dt replay
  - direct differentiation from the prepared initial carry
- and was less entangled with reverse-era replay infrastructure

### Concrete audit conclusion

The current forward AD lane is **not** a pure accepted-step-only benchmark lane.

More precisely:

- it is **not** differentiating directly through raw rejected attempts
- but it **is** conditioned by frozen controller/rejection-history outputs from
  the full adaptive primal rollout
- and it still uses a full adaptive primal solve as part of the custom-JVP
  implementation

Therefore the current lane does **not** fully comply with the old intended
benchmark behavior.

## Refactor target implied by the audit

The target forward-only contract should be:

1. parameterized initial state
2. forward-owned prepared initial carry
3. forward-owned realized accepted-step map
4. forward-owned custom JVP over that accepted-step map
5. final objective

and it should avoid:

- reverse-lane replay helpers
- FD replay helpers
- frozen rejected-step/controller-history metadata beyond what is strictly
  needed to replay the accepted-step physical map

## Step 2 Target Contract: Historical forward-only benchmark lane

This step pins the refactor target to the trusted pre-reverse benchmark
behavior already saved in `auto_diff.md`.

### Clean reference boundary

The recorded pre-reverse boundary is:

- clean pre-reverse commit:
  - `e888c02ba85b6ad790730d6c4e66d0bbd77f1cee`
- first checked reverse-era commit:
  - `412dfae7f552bd5ee51357acbb51d8506cd26ea1`

So the restored forward lane should be judged against the code behavior at or
before `e888c02...`, not against later reverse-era helper structure.

### Historical forward-lane facts already recovered

From the earlier history review captured in `auto_diff.md`:

- the old forward custom-JVP lane used the lightweight accepted-step replay
  contract:
  - `accepted_mask`
  - `attempted_dts`
  - `_radau_replay_realized_accepted_rollout(...)`
- the old scalar benchmark helper differentiated through
  `prepared_rollout.initial_carry` directly
- it did **not** rebuild that lane through the newer
  `_initial_carry_from_state_with_static_setup(...)` path for the scalar
  accepted-step benchmark

This gives a practical target shape:

1. build a primal adaptive rollout once to discover the realized accepted-step
   schedule
2. extract only the accepted-step replay contract needed by the old forward JVP
3. run the forward tangent lane from the prepared initial carry
4. do **not** inject reverse-era controller-history state into that tangent
   replay unless a strict old-forward dependency is proven

### Historical numerical target

The old trusted saved scalar `T0` benchmark is:

- parameter: `T0`
- `fd_step = 5.340000e-07`
- replay mode: `attempt`

Trusted values:

- `softmax_Er`
  - `ad = -2.160399e+01`
  - `fd = -2.161529e+01`
- `smooth_root_proxy`
  - `ad = 2.070900e-05`
  - `fd = 2.073464e-05`
- `Er2_volume_average`
  - `ad = -2.765750e+01`
  - `fd = -2.767012e+01`
- `Er_volume_average`
  - `ad = 2.291385e+00`
  - `fd = 2.291084e+00`
- `electron_temperature_volume_average_keV`
  - `ad = 3.571291e-01`
  - `fd = 3.571291e-01`
- `total_pressure_volume_average`
  - `ad = 1.835267e+00`
  - `fd = 1.835267e+00`
- `alpha_power_volume_average_mw_m3`
  - `ad = 7.221955e-02`
  - `fd = 7.220444e-02`

These are the primary numerical target for the restored scalar forward lane.

### Structural contract for the refactor

The restored forward-only lane should satisfy all of the following:

1. the AD benchmark path is owned by forward-only helpers
2. the AD tangent path starts from the prepared initial carry used by the
   trusted old benchmark
3. the tangent replay contract is restricted to the realized accepted-step map
4. the forward AD lane does not depend on reverse pullback helpers
5. the forward AD lane does not depend on FD replay orchestration
6. any reverse-specific or FD-specific helper reuse is replaced by explicit
   forward-owned wrappers or copied forward-only helpers if necessary

### Immediate refactor implications

This means the next implementation step should focus on:

- isolating the old forward-owned "prepared rollout / prepared initial carry"
  contract
- splitting that contract away from the current mixed helper stack
- ensuring the scalar benchmark JVP path no longer depends on the newer carry
  reconstruction shape unless that shape is required to reproduce the trusted
  old values

### What Step 2 does not assume yet

This contract does **not** yet claim:

- whether the final implementation should use a fully fused accepted-step
  tangent lane or a prepare-then-replay structure
- whether the vector benchmark should immediately share the exact same helper
  stack
- whether the FD lane should be changed at the same time

Those belong to later implementation and validation steps.

## Step 3 Preview: Implementation checklist

When we start the code refactor, the work should proceed in this order:

1. identify the smallest forward-owned helper set needed for:
   - parameterized state construction
   - prepared rollout construction
   - prepared initial carry extraction
   - accepted-step replay objective evaluation
2. split those helpers into a clearly forward-only lane in
   `benchmark_transport_autodiff_lagged_ntx.py`
3. make the scalar AD benchmark call only that forward-owned lane
4. remove scalar AD dependence on mixed forward/reverse/FD orchestration helpers
5. validate scalar `T0` against the trusted table before touching the vector
   benchmark lane

### Initial validation target

The first implementation checkpoint is not "all scripts pass".

It is:

- scalar forward AD for `T0`
- scalar frozen FD for `T0`
- values near the trusted table saved above
- accepted/attempt counts rechecked against the historical forward benchmark

Only after that should the refactor continue into broader benchmark cleanup.

## Step 3 Progress: forward-only scalar AD lane extraction

The first code pass for Step 3 is now in place in:

- `examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py`

### What was changed

1. added a forward-owned lightweight accepted-step replay helper:
   - `_forward_benchmark_replay_realized_accepted_final_y(...)`
2. added a forward-owned scalar custom-JVP final-y helper:
   - `_forward_benchmark_adaptive_final_y_realized_schedule(...)`
3. added a forward-owned scalar prepare helper that uses the prepared initial
   carry directly:
   - `_forward_benchmark_prepare_realized_schedule_scalar_rollout_ad_lane(...)`
4. rewired the scalar forward AD objective helper to use that new AD-only
   prepare helper
5. rewired the scalar forward AD rollout helper so
   `use_realized_schedule_jvp=True` uses the new AD-only prepare path

### What this improves structurally

This separates the scalar forward AD lane from the mixed helper stack in two
important ways:

- the scalar JVP path no longer goes through the solver-level mixed accepted
  replay helper that writes frozen controller-history metadata into the replay
  carry
- the scalar AD lane no longer starts from the carry rebuilt through
  `_initial_carry_from_state_with_static_setup(...)`; it now starts from the
  prepared initial carry again

### What is deliberately not changed yet

This pass does **not** yet:

- change the FD lane
- change the reverse lane
- change the vector benchmark lane
- replace the scalar VJP lane
- remove the older mixed scalar helper used by the FD frozen-trace path

So this is an intentional partial split:

- scalar forward JVP lane: moved to forward-owned helpers
- scalar FD lane: unchanged
- reverse lane: unchanged

### Next implementation target

The next code step should focus on Step 4:

- remove or bypass any remaining scalar forward AD path pieces that still rely
  on the mixed forward/FD/reverse helper structure
- then validate the restored scalar `T0` values against the trusted table

## Step 4 Progress: bypass remaining mixed scalar forward branches

The next cleanup pass is now also in place.

### What was changed

1. added a dedicated scalar forward-JVP objective wrapper:
   - `_forward_benchmark_adaptive_rollout_objectives_realized_schedule_only_for_parameter_jvp(...)`
2. rewired
   - `examples/benchmarks/benchmark_transport_adaptive_ad_vs_frozen_fd.py`
   so that `--adaptive-derivative-mode jvp` uses that dedicated forward-JVP
   helper directly instead of the mixed derivative-mode branch
3. wired `accepted_step_limit_override=args.accepted_step_limit` into the AD
   objective path in that scalar benchmark script

### Why this matters

Before this pass, the scalar benchmark script still:

- called the mixed derivative-mode objective helper even for the JVP lane
- and did not pass the CLI accepted-step truncation into the AD objective lane

After this pass:

- scalar forward JVP uses a dedicated forward-only benchmark entrypoint
- scalar AD truncation now follows the same accepted-step-limit knob exposed to
  the benchmark CLI

### Step 4 status

For the scalar benchmark lane, the main remaining mixed forward-branch behavior
has now been bypassed.

The next meaningful checkpoint is no longer more structural splitting.

It is:

- run the scalar forward AD and scalar frozen-FD checks again
- compare against the trusted `T0` table
- only then decide whether further scalar cleanup is still needed before moving
  into vector-lane restoration
