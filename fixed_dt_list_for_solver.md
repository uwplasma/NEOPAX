# Fixed `dt` List For Solver Plan

## Goal

Add a solver-native fixed-time-step-list mode for the forward / FD lane in the
current `NEOPAX` solver.

Required behavior:

- baseline run stays on the usual adaptive solver path
- `fd-` and `fd+` use the same solver path as baseline
- the only change for `fd-` / `fd+` is that the accepted `dt` values are given
  externally instead of chosen by the adaptive controller
- do **not** use the current replay machinery for the normal forward / FD lane

This means the fixed-time mode must be a **sibling** of the current adaptive
rollout, not a benchmark-local replay wrapper.

## Non-goals

- do not change reverse mode in this refactor
- do not change the standard adaptive solver behavior
- do not keep relying on `frozen trace` / `replay` helper semantics for the
  normal FD workflow
- do not force baseline `next_lagged_response_valid` or controller-history
  decisions into the FD lane

## Current problem

The current FD refactor is still wrong structurally:

- baseline uses the full adaptive schedule rollout path
- fixed-time FD uses the same inner accepted-step primal, but **not** the same
  outer solver path
- therefore baseline and `fd-` / `fd+` are not yet evolving carry/state through
  the same full solve wrapper

That mismatch is likely why:

- FD timing still looks suspicious
- the electric-field-related metrics do not match the old intended benchmark
- GPU work finishes and the remaining behavior looks inconsistent with a simple
  “same solver, fixed `dt` list” interpretation

## Refactor target

We want two solver-native rollout modes:

1. Adaptive baseline mode
   - current production path
   - chooses `dt` internally

2. Fixed accepted-`dt` mode
   - same step-state / carry evolution as baseline
   - same accepted-step forward solve math as baseline
   - does **not** run adaptive `dt` selection
   - instead consumes a provided `dt_sequence`

The FD benchmark should become:

1. run adaptive baseline once
2. extract accepted `dt` list from baseline
3. run `fd-` with perturbed initial parameter and fixed `dt` list
4. run `fd+` with perturbed initial parameter and fixed `dt` list

No replay helpers in the normal path.

## Implementation plan

### Step 1. Isolate the exact baseline solver path

Audit and preserve the current adaptive baseline path in:

- `NEOPAX/_transport_solvers.py`
- especially:
  - `_radau_adaptive_schedule_rollout(...)`
  - `_radau_attempt_step_with_payload(...)`
  - `_apply_radau_lean_timestep_controller(...)`
  - `_radau_step_state_from_carry(...)`
  - `_radau_carry_from_step_state(...)`

Requirement:

- the new fixed-time mode must reuse the same accepted-step state evolution
  logic as this path

### Step 2. Add a solver-native fixed-`dt` schedule rollout

Create a new solver function in `NEOPAX/_transport_solvers.py`, conceptually:

- `_radau_fixed_dt_schedule_rollout(...)`

Desired structure:

- clone the skeleton of `_radau_adaptive_schedule_rollout(...)`
- operate on `_RadauStepState`
- iterate over a provided `dt_sequence`
- for each supplied `dt`, run the same step attempt path used by baseline

Important:

- do not replace the outer solver path with `_radau_apply_accepted_step_map(...)`
- do not use `_radau_run_prepared_on_realized_trace(...)`
- do not use benchmark-local replay logic

### Step 3. Define the fixed-`dt` accept-step behavior

For each supplied `dt_value`:

1. set the step state's `dt` to `dt_value`
2. run the same accepted-step attempt code as baseline
3. reuse the same accepted-step carry/state update logic as baseline
4. clear only the adaptive-controller evolution that should not persist in the
   fixed-time lane

This is the key design constraint:

- same forward solve path
- same lagged-response numerical path
- no adaptive `next_dt` decision

### Step 4. Make the fixed-time lane explicit about controller bypass

The fixed-time mode should explicitly document what is bypassed:

- `next_dt` growth / shrink selection
- reject / retry adaptive branching as a timestep-selection mechanism

But it should still define what happens if a fixed `dt` step is not viable.

Decision to make during implementation:

Option A:
- fail immediately if the supplied fixed step does not produce a valid accepted
  step

Option B:
- keep enough solver diagnostics to show where the fixed schedule became invalid

Recommended first pass:

- fail clearly and report the step index / `dt` / solver diagnostics

### Step 5. Add prepared-rollout entrypoint for fixed `dt` solves

Add a prepared helper that forwards into the new solver-native mode, e.g.:

- `_radau_solve_on_fixed_dt_list_final_state_only(...)`

This helper should:

- convert `time_list` to `dt_sequence`
- call the new fixed-`dt` schedule rollout
- return:
  - `time_list`
  - `dt_sequence`
  - `final_state`
  - `final_carry`
  - rollout diagnostics

It should **not** be named or documented as replay.

### Step 6. Rewire the forward / FD benchmark lane

Update:

- `examples/benchmarks/benchmark_transport_forward_fd_lane.py`
- `examples/benchmarks/benchmark_transport_frozen_fd_only.py`

So that:

- baseline still runs through the usual adaptive solver
- accepted times are extracted from baseline
- `fd-` / `fd+` call only the new solver-native fixed-`dt` mode
- no normal-path dependence remains on the current replay helpers

### Step 7. Keep the old replay machinery out of the normal FD path

The following should no longer be part of the normal forward / FD workflow:

- `_radau_forward_fd_run_prepared_on_realized_trace(...)`
- `_radau_run_prepared_on_realized_trace(...)`
- forced controller-history replay
- forced `next_lagged_response_valid` replay

If these helpers must remain for legacy debugging, they should be clearly
marked as:

- legacy
- debug-only
- not used by the normal FD benchmark lane

### Step 8. Validation checks

After implementing the new mode, validate in this order:

1. Baseline adaptive solve diagnostics
   - accepted count
   - attempt count
   - final metrics

2. Baseline adaptive vs fixed-time baseline
   - run the same initial state through the fixed `dt` list
   - check final-state and metric agreement
   - this must match closely before using `fd-` / `fd+`

3. FD timing
   - `fd-` and `fd+` should not take dramatically more time than baseline
   - fixed-time mode should be comparable or cheaper than baseline

4. FD values
   - compare against the old trusted benchmark behavior
   - especially:
     - `softmax_Er`
     - `Er2_volume_average`
     - `Er_volume_average`
     - pressure / temperature metrics

## Expected outcome

At the end of this refactor:

- baseline remains the current standard adaptive solver
- FD becomes a true fixed-time sibling solver lane
- the solver, not benchmark-local replay code, owns the fixed-time semantics
- `fd-` / `fd+` do what was originally requested:
  - same solver path as baseline
  - same forward numerical treatment
  - only the `dt` list is externally prescribed

## Current solver-path findings (2026-06-18)

These notes were added after a detailed code comparison between:

- current `NEOPAX`
- `NEOPAX copy 6`

with focus on the plain forward solver path around NTX lagged-response usage.

### What was checked

- `RADAUSolver.solve(...)`
- `_run_saved_loop(...)`
- `_make_radau_initial_step_state(...)`
- `_radau_step_fn(...)`
- `_radau_step_fn_forward_solver(...)`
- `_radau_attempt_step_lean(...)`
- `_radau_attempt_step_forward_solver(...)`
- `_execute_radau_accepted_step_attempt(...)`
- `_radau_single_step_primal(...)`
- `_radau_prepare_lagged_response(...)`
- `TransportEquations.build_lagged_response(...)`
- `TransportEquations.evaluate_with_lagged_response(...)`
- `NTXExactLijRuntimeTransportModel.build_lagged_response(...)`
- `NTXExactLijRuntimeTransportModel.evaluate_with_lagged_response(...)`

### Main findings

1. The NTX lagged-response primal path itself looks effectively the same
   between current `NEOPAX` and `copy 6`.
   - The lagged-response hook chain, equation-level lagged build/eval, and
     NTX runtime lagged build/eval were not the source of an obvious solver
     divergence.

2. The plain solver orchestration is no longer identical.
   - `copy 6` plain adaptive solve goes through:
     - `_radau_step_fn(...)`
     - `_radau_attempt_step_lean(...)`
     - `_execute_radau_accepted_step_attempt(...)`
   - current plain adaptive solve goes through:
     - `_radau_step_fn_forward_solver(...)`
     - `_radau_attempt_step_forward_solver(...)`
     - `_radau_single_step_primal(...)`

3. That orchestration split is structural, not obviously numerical.
   - `_radau_attempt_step_forward_solver(...)` effectively inlines the same
     primal one-step call and controller application that
     `_radau_attempt_step_lean(...)` gets via
     `_execute_radau_accepted_step_attempt(...)`.
   - No clear numerical difference was identified from that wrapper split
     alone.

4. The accepted-step carry and kernel context are effectively the same.
   - `_RadauAcceptedStepCarry`
   - `_RadauAcceptedStepAttemptResult`
   - `_RadauAcceptedStepKernelContext`
   do not show a meaningful forward-runtime widening relative to `copy 6`.

5. The accepted-step physics context is wider in current `NEOPAX`.
   - Current `_RadauAcceptedStepPhysicsContext` carries extra closures:
     - `project_flat_pullback`
     - `build_lagged_response_pullback`
     - `flat_rhs_with_shared_fluxes`
   - `copy 6` only carries the smaller forward-facing fields:
     - `unpack_flat`
     - `project_flat`
     - `build_lagged_response`
     - `flat_rhs`
     - `flat_rhs_with_lagged_response`

6. `_radau_single_step_primal(...)` does not appear to use those extra fields.
   - In the current code, the primal step only touches:
     - `unpack_flat`
     - `project_flat`
     - `build_lagged_response`
     - `flat_rhs`
     - `flat_rhs_with_lagged_response`
   - So the extra fields are not changing the primal math directly.

### Best current hypothesis

The strongest remaining code-level suspect for the forward solver compile-time
inflation is not the NTX lagged-response numerical path itself, but the fact
that the current plain solver closes over a wider `physics_context` than the
older forward lane did, even though the extra reverse/shared-flux closures are
not used by `_radau_single_step_primal(...)`.

This is still a hypothesis. It has **not** yet been validated by refactoring
the forward solver to use a smaller forward-only context and then rerunning the
timing comparison.

### Recommended next step

Create a strictly forward-only accepted-step physics context for the plain
solver lane, carrying only:

- `unpack_flat`
- `project_flat`
- `build_lagged_response`
- `flat_rhs`
- `flat_rhs_with_lagged_response`

Keep reverse / shared-flux / pullback closures in separate AD-facing contexts.

Then test whether the rebuild / reuse compile times of the plain adaptive
solver drop relative to the current path.

## Current FD lane state (2026-06-19)

The frozen-FD benchmark lane was narrowed further.

### Confirmed current routing

- baseline solve:
  - still uses the adaptive production forward solver lane
- accepted-time extraction:
  - comes from `_accepted_time_list_from_trace(...)`
  - uses only `active_mask & accepted_mask`
  - so rejected attempts are **not** directly inserted into the frozen
    accepted time list
- `fd-` and `fd+`:
  - now use the direct fixed-`dt` accepted-step-map path by default
  - they do **not** go through the adaptive-controller wrapper anymore

### Relevant code state

In `examples/benchmarks/benchmark_transport_forward_fd_lane.py`:

- `_adaptive_rollout_objectives_for_parameter_on_frozen_trace(...)`
- `_adaptive_rollout_objectives_for_parameter_on_time_list(...)`
- `_solve_on_fixed_time_map_direct_accepted_step_map_debug(...)`

In `examples/benchmarks/benchmark_transport_frozen_fd_only.py`:

- normal `fd-` / `fd+` now call the frozen-trace helper with:
  - `use_direct_accepted_step_map_debug=True`

This means the FD derivative lane is now:

- baseline adaptive solve once
- accepted time list extracted from baseline
- `fd-` direct accepted-step-map fixed-`dt` solve
- `fd+` direct accepted-step-map fixed-`dt` solve

### Baseline replay debug behavior

The CLI flag:

- `--debug-direct-accepted-step-map`

now matters only for the unperturbed baseline replay debug branch.

Command:

```bash
python ./examples/benchmarks/benchmark_transport_frozen_fd_only.py --ntx-exact-derivative-mode direct --parameter T0 --replay-mode accepted --baseline-replay-debug --debug-direct-accepted-step-map
```

runs only:

- baseline adaptive solve
- one unperturbed fixed-`dt` baseline replay
- final-objective / `E_r` comparison

It does **not** run:

- `fd-`
- `fd+`
- accepted-step state debug
- single-step compare

### Current open issue

Even after:

- reducing the baseline replay mismatch
- and moving `fd-` / `fd+` onto the direct accepted-step-map fixed-`dt` path

the FD derivatives can still behave incorrectly, especially in the `E_r`
metrics.

So the remaining bug is no longer:

- obvious rejected-step leakage into the time list
- or `fd-` / `fd+` going through a different helper family than intended

The next likely targets are:

- exact accepted-time-map semantics
- direct fixed-`dt` accepted-step-map carry evolution under perturbation
- any remaining asymmetry between unperturbed baseline replay and perturbed
  `fd-` / `fd+`
