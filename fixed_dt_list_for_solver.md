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

## Current FD lane state (2026-06-20)

This note supersedes the 2026-06-19 routing summary above for the current
working tree.

### Latest observed failing case

Command family:

```bash
python ./examples/benchmarks/benchmark_transport_frozen_fd_only.py --ntx-exact-derivative-mode direct --parameter T0 --fd-rel-step 1e-8 --fd-abs-step 1e-12 --replay-mode accepted --fixed-time-lane solver
python ./examples/benchmarks/benchmark_transport_frozen_fd_only.py --ntx-exact-derivative-mode direct --parameter T0 --fd-rel-step 3e-8 --fd-abs-step 1e-12 --replay-mode accepted --fixed-time-lane solver
```

Observed behavior:

- baseline adaptive solve completes:
  - `attempt_count=158`
  - `accepted_count=99`
  - `completed=True`
- `fd_plus` completes the full fixed-time map:
  - `accepted_count=99`
  - `final_t=1.0e-02`
- `fd_minus` does **not** complete:
  - for `fd_rel=1e-8`: `accepted_count=96`, `completed=False`,
    `final_t=9.910954e-03`
  - for `fd_rel=3e-8`: `accepted_count=45`, `completed=False`,
    `final_t=1.838964e-03`

Therefore the huge printed central-FD gradients in these runs are not valid
finite-difference derivatives. They mix:

- one complete `fd_plus` endpoint
- one incomplete `fd_minus` endpoint

The benchmark should not report a normal FD gradient when either endpoint
fails to consume the full prescribed fixed-time map.

### Current implementation issue

The current fixed-time solver lane is still not literally "the production
forward solver with an external `dt` list".

In `NEOPAX/_transport_solvers.py`,
`_radau_forward_solver_fixed_dt_schedule_rollout(...)` currently:

1. calls `_radau_single_step_primal(...)` directly
2. checks only nonlinear/subsolve health for fixed-time acceptance
3. manually constructs the next `_RadauStepState`

This duplicates part of the production accepted-step update that normally
lives behind:

- `_radau_attempt_step_forward_solver(...)`
- `_apply_radau_lean_timestep_controller(...)`

That duplication is a real mismatch risk for:

- `prev_error`
- `prev_stages`
- `prev_dt`
- `prev_theta_final`
- `prev_newton_iter_count`
- lagged-response cache validity
- Jacobian/LU cache state
- controller-history fields

The normal adaptive solver lane itself should remain untouched.

### Current conclusion

The immediate problem is not just "Er is sensitive".

The latest data show a clearer failure mode:

- the fixed-time `fd_minus` endpoint fails before `t_final`
- the benchmark still computes and prints central FD values
- those values are therefore contaminated by an incomplete endpoint

The next code change should first make this invalid state explicit:

- if either endpoint has `completed=False` or `failed=True`, print the endpoint
  diagnostics and mark the FD gradient as invalid
- do not present the central difference as a normal derivative in that case

### Recommended next implementation step

Refactor only the fixed-time FD lane so it reuses the production forward
attempt/update path as much as possible.

Target behavior:

1. baseline remains the standard adaptive production solver
2. accepted `dt` values are extracted from the baseline trace
3. fixed-time `fd-` / `fd+` run the same forward attempt/update machinery
4. the only fixed-time override is the next prescribed `dt`
5. no adaptive timestep selection is used to choose the next `dt`
6. standard adaptive solver behavior is not changed

In other words:

- reuse production forward accepted-step state evolution
- bypass only adaptive `next_dt` selection
- do not hand-build the accepted step state in the fixed-time lane

### Validation order

After the refactor:

1. Run unperturbed baseline vs fixed-time baseline:

   ```bash
   python ./examples/benchmarks/benchmark_transport_frozen_fd_only.py --ntx-exact-derivative-mode direct --parameter T0 --replay-mode accepted --baseline-replay-debug --fixed-time-lane solver
   ```

2. Run a tiny FD step and require both endpoints to complete:

   ```bash
   python ./examples/benchmarks/benchmark_transport_frozen_fd_only.py --ntx-exact-derivative-mode direct --parameter T0 --fd-rel-step 1e-10 --fd-abs-step 1e-12 --replay-mode accepted --fixed-time-lane solver
   ```

3. Only after both endpoints complete, inspect FD values for larger `h`:

   ```bash
   python ./examples/benchmarks/benchmark_transport_frozen_fd_only.py --ntx-exact-derivative-mode direct --parameter T0 --fd-rel-step 1e-8 --fd-abs-step 1e-12 --replay-mode accepted --fixed-time-lane solver
   python ./examples/benchmarks/benchmark_transport_frozen_fd_only.py --ntx-exact-derivative-mode direct --parameter T0 --fd-rel-step 3e-8 --fd-abs-step 1e-12 --replay-mode accepted --fixed-time-lane solver
   ```

Do not resume forward-AD comparison until the FD fixed-time lane has a valid
completed-minus/completed-plus baseline.

## Current FD replay state (2026-06-21)

This section records the latest narrowing of the accepted-time FD / replay
debugging path.

### Current benchmark TOML

The default TOML for `benchmark_transport_frozen_fd_only.py` is:

```text
examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_benchmark.toml
```

Relevant settings:

```toml
[transport_solver]
transport_solver_backend = "radau"
integrator = "radau"
radau_rhs_mode = "lagged_response"
radau_num_stages = 7
radau_controller_mode = "hairer_lean_transport_discounted"
radau_predictor_mode = "collocation_transport_weighted"
lagged_response_reuse_mode = "global_state_drift"
t_final = 0.01
dt = 1.0e-5
rtol = 1.0e-6
atol = 1.0e-8
max_steps = 20000
```

The profile baseline values used by the scalar FD benchmark are:

```toml
n0 = 4.21
T0 = 17.8
density_shape_power = 10.0
temperature_shape_power = 2.0
```

### Latest baseline-vs-fixed-time replay result

Command:

```bash
python ./examples/benchmarks/benchmark_transport_frozen_fd_only.py --ntx-exact-derivative-mode direct --parameter T0 --replay-mode accepted --baseline-replay-debug --fixed-time-lane solver
```

Observed result:

- baseline adaptive:
  - `attempt_count=158`
  - `accepted_count=99`
  - `completed=True`
  - `failed=False`
- fixed-time baseline replay:
  - `accepted_count=99`
  - `state_finite=True`
- mismatch:
  - `Er max_abs_diff=1.912787e-06`
  - `Er mean_abs_diff=4.108936e-08`
  - `Er max_rel_diff=7.333392e-08`
  - `Er mean_rel_diff=3.359405e-09`
  - `softmax_Er rel_diff=2.508350e-10`
  - `Er2_volume_average rel_diff=2.145113e-09`
  - `Er_volume_average rel_diff=1.035160e-08`

This mismatch did **not** change when `nonlinear_solver_tol` was tightened to
`1.0e-12`.

### Newton tolerance conclusion

The TOML uses:

```toml
radau_newton_tol_mode = "hairer"
radau_newton_fnewt_mode = "hairer"
```

Therefore convergence is not checked simply as:

```text
final_residual_norm <= nonlinear_solver_tol
```

The active Hairer-style check is:

```text
newton_metric_final <= predictor_fnewt
```

where `predictor_fnewt` is derived mainly from `rtol`, machine epsilon, and
`radau_num_stages`.

Changing only `nonlinear_solver_tol` is therefore not expected to tighten the
active Newton stopping condition in this TOML.

### Jacobian/LU reuse modes

A new benchmark/solver mode was added:

```text
radau_jacobian_reuse_mode = "retry_refactor_lu"
```

Current meanings:

- `retry_only`
  - reuse Jacobian only on same-`t_i` retry after a rejected attempt
  - reuse LU only if retry and `dt_close`
- `retry_refactor_lu`
  - reuse Jacobian only on same-`t_i` retry after a rejected attempt
  - always refactor LU for the current trial `dt`
- `dt_close` / `legacy`
  - old behavior
  - reuse Jacobian and LU whenever `cache_valid && dt_close`

Command:

```bash
python ./examples/benchmarks/benchmark_transport_frozen_fd_only.py --ntx-exact-derivative-mode direct --parameter T0 --replay-mode accepted --baseline-replay-debug --fixed-time-lane solver --radau-jacobian-reuse-mode retry_refactor_lu
```

Observed result was identical to `retry_only`:

- `Er max_abs_diff=1.912787e-06`
- `Er mean_abs_diff=4.108936e-08`

So LU reuse from a nearby rejected `dt` is **not** the observed mismatch source.

### Fully adaptive FD endpoint test

An explicit endpoint lane was added:

```bash
--fd-endpoint-lane adaptive
```

This runs `fd-` and `fd+` through the same production adaptive solve helper as
the baseline, rather than the fixed accepted-time map.

Result:

- this is worse for the old accepted-history benchmark target
- `fd-` and `fd+` choose different adaptive histories:
  - example: baseline `accepted_count=99`
  - `fd_minus accepted_count=122` or `115`
  - `fd_plus accepted_count=114` or `106`
- therefore this endpoint lane is useful only as a diagnostic that full
  adaptive FD is not the old intended comparison object

### Current narrowed diagnosis

The accepted-time replay mismatch is not explained by:

- loose Newton tolerance via `nonlinear_solver_tol`
- LU reuse at nearby rejected `dt`
- fully adaptive endpoint FD

The remaining issue is that accepted-time replay freezes only the accepted
`dt` list, while the production accepted attempt may also depend on carry /
cache context produced by previous rejected attempts.

However, the current TOML narrows the likely channels:

- lagged response:
  - should not change during retries at the same `t_i` because accepted `y_i`
    does not advance
  - it only rebuilds at an attempt start if `lagged_response_valid=False`
  - accepted branch uses the global state drift test against
    `lagged_reference_y`
- `prev_stages`:
  - stored only on accepted steps
  - used directly in `_make_radau_stage_predictor(...)` to construct `z0`
- `prev_dt`:
  - stored only on accepted steps
  - used to scale `prev_stages` in the predictor via `h / prev_dt`
  - also used in adaptive controller growth formulas
- `prev_error`:
  - stored only on accepted steps
  - used by adaptive controller growth formulas
  - should not affect the physical state if fixed `dt` is truly enforced
- `prev_theta_final` and `prev_newton_iter_count`:
  - can be updated by rejected attempts
  - but the active TOML predictor mode is `collocation_transport_weighted`, not
    `newton_quality_gated_collocation`, so these should not currently drive
    `z0`

### Next useful diagnostic

The next diagnostic should be lightweight and should avoid storing large arrays.

Goal:

- stop at the first accepted step where fixed-time replay differs from the
  production adaptive accepted state beyond a chosen tolerance
- print only the incoming carry/context differences for that step

Fields to compare:

- accepted `dt`
- incoming `t`
- incoming `y` max/relative diff
- `prev_stages` max diff
- `prev_dt` diff
- `prev_error` diff
- `lagged_response_valid`
- `lagged_reference_y` max diff
- `cache_valid`
- `cache_dt`
- `recent_reject_count`
- whether Jacobian was reused
- whether LU was reused/refactored

The main question for the next session is:

```text
Which incoming carry/context field first differs between production adaptive
accepted steps and fixed accepted-time replay?
```
