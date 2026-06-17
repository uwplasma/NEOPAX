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

## Current State: accepted-step FD lane diagnosis

This section records the latest scalar frozen-FD investigation state for the
`T0` lane so the next session can resume from the current narrowed diagnosis.

### Current command under investigation

```bash
python ./examples/benchmarks/benchmark_transport_adaptive_ad_vs_frozen_fd.py --ntx-exact-derivative-mode direct --parameter T0 --adaptive-derivative-mode jvp --run-mode fd --fd-rel-step 3e-8 --fd-abs-step 1e-10 --replay-mode accepted
```

### Confirmed runtime path

The benchmark now prints solver settings directly. The current run confirms the
forward/frozen-FD lane is using:

- `backend=radau`
- `integrator=radau`
- `radau_rhs_mode=lagged_response`
- `radau_num_stages=7`
- `t0=0.0`
- `t_final=0.01`
- `dt=1e-05`

### Key accepted-FD finding

The important ambiguity is now resolved:

- the accepted-step FD lane is still wrong **before perturbing `T0`**
- the unperturbed baseline fixed-dt replay is already nonfinite

Observed line:

- `frozen replay baseline fixed-dt: objectives_finite=False final_state_finite=False final_carry_finite=False`

So this is **not** currently a finite-difference step-size issue.

### Localized failure location

For baseline fixed accepted-step replay and for both `fd_minus` / `fd_plus`,
the first bad accepted step is:

- `first_bad_index=147`
- `first_bad_was_accepted=True`
- `first_bad_accepted_ordinal=90`
- `first_bad_dt=2.0831628433823396e-05`

accepted-window around the first bad accepted step:

- accepted ordinal `88`, trace index `144`, finite
- accepted ordinal `89`, trace index `145`, finite
- accepted ordinal `90`, trace index `147`, nonfinite
- accepted ordinal `91`, trace index `150`, nonfinite
- accepted ordinal `92`, trace index `152`, nonfinite

This means the accepted-step replay diverges starting at accepted step 90 of
the baseline chronology.

### Important structural conclusion

Because the baseline fixed accepted-step replay already fails:

- the accepted-step FD bug is currently in the frozen accepted-step replay
  contract itself
- it is **not** yet a perturbation-size issue
- it is **not** evidence that rejected steps are still being replayed

### Latest implementation fix applied

The latest identified mismatch was in
`NEOPAX/_transport_solvers.py`:

- `_radau_fixed_dt_accepted_rollout(...)`
- `_radau_replay_realized_accepted_step_map_rollout(...)`

These helpers had been carrying `step_map_result.next_carry` too literally,
while the real adaptive accepted branch preserves the lagged-response
cache/reference fields differently after acceptance.

They were updated to make the fixed accepted-step replay mirror the adaptive
accepted branch's lagged-cache carry semantics more closely:

- preserve `lagged_response_cache` from the incoming carry
- preserve `lagged_reference_y` from the incoming carry
- recompute `lagged_response_valid` using the same
  `global_state_drift` reuse criterion structure used by the adaptive accepted
  branch

This is the latest patch that still needs rerun validation.

### Latest benchmark-script diagnostics added

The scalar benchmark script now also records:

- solver settings in the summary
- baseline fixed-dt replay finiteness for `--replay-mode accepted`
- first bad accepted-step ordinal
- accepted-step local window around the first bad accepted step

So the same scalar FD command above is now the correct probe to validate the
accepted-step frozen replay contract.

### Exact next-session first command

Rerun exactly:

```bash
python ./examples/benchmarks/benchmark_transport_adaptive_ad_vs_frozen_fd.py --ntx-exact-derivative-mode direct --parameter T0 --adaptive-derivative-mode jvp --run-mode fd --fd-rel-step 3e-8 --fd-abs-step 1e-10 --replay-mode accepted
```

Primary success criterion for that rerun:

- `frozen replay baseline fixed-dt: objectives_finite=True final_state_finite=True final_carry_finite=True`

If that becomes finite, then:

- the accepted-step frozen replay contract is much closer to correct
- only after that should `fd_minus` / `fd_plus` values be interpreted again

If baseline fixed-dt is still nonfinite, the next debugging target should stay
inside the accepted-step fixed replay semantics rather than revisiting FD step
size or AD comparisons.

## Current State: accepted-step FD replay is finite but still not the old benchmark

The latest scalar frozen-FD command under active investigation is now:

```bash
python ./examples/benchmarks/benchmark_transport_frozen_fd_only.py --ntx-exact-derivative-mode direct --parameter T0 --fd-rel-step 3e-8 --fd-abs-step 1e-10 --replay-mode accepted --accepted-step-limit 115
```

### Latest confirmed output

Observed output:

- baseline adaptive rollout:
  - `attempt_count=188`
  - `accepted_count=115`
  - `completed=True`
  - `failed=False`
  - `fail_code=0`
- timings:
  - `baseline_adaptive_s=8.417482e+02`
  - `fd_minus_s=2.111428e+03`
  - `fd_plus_s=2.111278e+03`
- finite-difference values:
  - `softmax_Er = -8.261394e+01`
  - `smooth_root_proxy = 1.630489e-05`
  - `Er2_volume_average = 3.384430e+00`
  - `Er_volume_average = 1.676171e+01`
  - `electron_temperature_volume_average_keV = 3.529213e-01`
  - `total_pressure_volume_average = 1.639403e+00`
  - `alpha_power_volume_average_mw_m3 = 4.440607e-02`

### Interpretation

This is an important narrowing relative to the earlier nonfinite replay state:

- the accepted-step frozen replay is now finite
- but it is still not reproducing the old trusted forward/FD benchmark values
- and it is still far too expensive

In particular:

- the thermodynamic metrics are at least in the rough neighborhood
- the `E_r`-sensitive metrics are still clearly wrong
- each replay (`fd_minus`, `fd_plus`) is much slower than the adaptive baseline
  solve, which should not happen if replay is really just the usual forward
  solver on the baseline accepted `dt` sequence

### Structural conclusion

The current problem is no longer:

- nonfinite accepted replay
- FD step-size selection
- accidental replay of rejected steps

The current problem is now most likely:

- the fixed accepted-time replay contract still does **not** match the usual
  forward solver semantics closely enough, especially for the `E_r` /
  lagged-response path
- and/or the benchmark lane is rebuilding too much setup for each of
  `fd_minus` / `fd_plus`

### Important review result from the latest code pass

The recent wrapper change:

- `benchmark_transport_forward_fd_lane.py`
  - accepted replay now calls `_radau_run_prepared_on_time_list(...)`
  - instead of `_radau_forward_fd_run_prepared_on_time_list(...)`

turned out to be mostly a behavior-neutral isolation step, because the two
accepted-step rollout helpers are currently almost identical in solver logic.

So:

- the wrapper swap is safe
- but it is not the real fix

### Next steps

The next refactor should stay strictly inside the forward/FD lane and make the
accepted replay semantics match the ordinary forward pass more literally.

1. Compare the old trusted forward/FD accepted-step contract from
   `NEOPAX_reverse_scratch` against the current `NEOPAX` forward/FD lane.
   Focus only on:
   - accepted time-list construction
   - carry entering accepted replay
   - lagged-response / cache / `E_r`-relevant carry fields

2. Refactor the frozen-FD accepted replay lane so it behaves as:
   - the usual forward transport solver
   - with the baseline accepted `dt` sequence imposed
   - without adaptive accept/reject controller evolution

3. Reduce duplicated setup work in the benchmark lane:
   - avoid rebuilding more than necessary between `fd_minus` and `fd_plus`
   - preserve the forward/FD lane as separate from reverse-specific replay code

4. Revalidate with:

```bash
python ./examples/benchmarks/benchmark_transport_frozen_fd_only.py --ntx-exact-derivative-mode direct --parameter T0 --fd-rel-step 3e-8 --fd-abs-step 1e-10 --replay-mode accepted --accepted-step-limit 115
```

Primary success criteria:

- replay time drops materially below the current `~2111 s` per replay
- `E_r`-sensitive FD metrics move back toward the historical trusted values
- no reverse-lane code paths are involved in the forward/FD benchmark
