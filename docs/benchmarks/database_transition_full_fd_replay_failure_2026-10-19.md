# Full-time FD replay failure before the bootstrap objective

The first full-time FD comparison failed on the profile minus endpoint. The
traceback enters `_objectives_on_realtime_geometry_frozen_trace`, then
`_objective_vector`, then `bootstrap_current_softmax_abs_scaled`, and finally
the momentum-corrected flux evaluator. Lineax reports nonfinite linear-system
inputs. The visible error is a downstream objective failure; it does not
identify the first invalid replay step or prove which physical field or
bootstrap coefficient became nonfinite.

The corresponding [full-time AD reference](database_transition_objectives_ad_recorded_reuse_full_t20_2026-10-19.md)
completed successfully at 20 seconds / 290 accepted steps, with all 144
derivatives finite and six exactly matching replay endpoints. This FD failure
does not establish a failure of that reverse sweep or an AD/FD gradient error.

## Confirmed FD defects

The `reverse_frozen_metadata` endpoint helper used the old replay reuse policy:
the FD schedule did not record the adaptive forward `jacobian_reused` flags,
and its accepted-step helper did not apply the corresponding forward rebuild
mask. The new AD benchmark does apply that mask. Converged implicit-step
derivatives do not require identical Newton iteration counts or cached LU
history, but a reconstructed step must solve its stage equations successfully.

More seriously, the FD helper wrote `True` into every active slot of
`converged_mask`, discarding the accepted-step solver's actual convergence
flag. Its completeness check verified only final time. The accepted-map
helper advances time even for an unconverged or nonfinite trial, so reaching
20 seconds did not prove a valid FD endpoint. Such a result could reach the
bootstrap objective and obscure the original failed step behind a Lineax
callback error.

The previous command selected the existing FD contract without correcting
these implementation gaps. The saved 16-step comparisons remain evidence for
their tested prefix; they did not exercise every later full-time replay step.

## Correction

Only `benchmark_transport_realtime_geometry_forward_fd.py` is changed:

- For `reverse_frozen_metadata`, record one forward Newton-reuse boolean per
  attempt, then force Jacobian rebuilds on the accepted slots where the forward
  rebuilt. Rejected attempts remain excluded from the FD endpoint replay.
- Keep the baseline and perturbed endpoint's predictor/controller/Jacobian/LU
  metadata tied to the unperturbed replay, preserving the existing FD contract.
- Check actual Newton convergence and finite carry/stage arrays for both solves
  at each accepted step. After a bad step, skip subsequent slots and retain
  the last valid carry.
- Raise before objective evaluation with the endpoint label, baseline versus
  perturbed role, first failed accepted-step number, attempt slot, start time,
  timestep and actual Newton convergence flag.
- Reject an incomplete/nonfinite adaptive baseline and invalid final replay
  state; print baseline accepted/attempt counts and final time.
- Record `fd_replay_jacobian_reuse_mode=recorded_forward` in the FD JSON and
  execution banner when the frozen-metadata contract is selected.

No new command-line mode, reverse-AD equation, production transport solver,
root objective or bootstrap objective is introduced or changed. The initial
callback traceback did not identify the first failed replay step. The remote
rerun below establishes that the rebuild-mask correction alone was insufficient.

## Remote rerun: perturbed Newton failure at accepted step 83

The user reran the corrected `n0` FD command and obtained:

```text
[fd-baseline] accepted_steps=290 attempts=495 final_time=2.0000000000000000e+01 recorded_newton_setup=True
Invalid FD accepted-step replay: endpoint=n0:minus role=perturbed_endpoint accepted_step=83 attempt_slot=156 t_start=5.5758482511664553e-03 dt=2.1380474331935400e-08 newton_converged=False.
```

Exit status was 1, wall time 24:52.00 and maximum host RSS 10,160,588 KiB.
The adaptive baseline reached the required final time and the unperturbed
replay solve at slot 156 was valid. The nearby minus-parameter solve did not
converge under the frozen numerical setup. This error does not itself establish
nonfinite physical state, an incorrect AD derivative, or a bootstrap defect.
The failing step occurs early in physical time despite being step 83.

### Same-timestep Newton recovery

The FD helper now retries an invalid baseline or endpoint Newton solve once,
from the original incoming physical state at the identical accepted timestep.
The retry invalidates the cached Jacobian/LU setup and clears stage predictor
history. It retains the physical state, lagged physical payload, model, stage
equations, tolerances, iteration budget and accepted schedule. The unsuccessful
trial state is discarded. After a successful solve, the endpoint continues to
receive the baseline's frozen metadata as before.

This is a bounded nonlinear-solver restart inside an existing accepted slot.
It neither inserts a transport step nor replays/differentiates a rejected
adaptive attempt. Cached Newton matrices and predictors are acceleration
metadata; converged implicit-step sensitivities do not require identical
acceleration history at neighboring FD parameters.

The `[fd-replay]` banner reports baseline and endpoint restart counts, and the
FD JSON additionally records endpoint restart step numbers and
`fixed_timesteps=True`. If the restart remains invalid, the guard still stops
before objective evaluation and now includes iteration count, residual norm
and correction norm from the final unsuccessful solve. The subsequent remote
run below confirms that this recovery also fails at the same slot.

## Remote cold restart also fails: capture the actual incoming state

The next user run again has 290 accepted baseline steps / 495 attempts at
20 seconds. The cold restart fails at accepted step 83 / attempt slot 156:

```text
t_start=5.5758482511664553e-03 dt=2.1380474331935400e-08
newton_converged=False cold_restart=True newton_iterations=2
residual_norm=1.0663013333199972e+06 delta_norm=5.2937721381344218e+06
```

Exit status is 1, wall time 30:48.13, maximum host RSS 16,313,880 KiB.
Two iterations did not exhaust the configured 20-iteration budget. The
available log does not identify the early-stop flag, failing equation,
physical species or radial location. A stale cache alone is therefore an
insufficient explanation, and the scalar recovery regression is not proof
that the plasma case is repaired. The actual remaining cause is unresolved.

The FD helper now saves a compact NPZ automatically when that failure occurs.
Because the checked scan freezes at its first bad slot, its final two carries
are precisely the failed slot's incoming baseline and endpoint carries. Saving
them requires no per-step carry history, extra replay, or JAX host callback.
The snapshot also records the two initial packed states, solver/model config,
profile values, timestep, forward rebuild decision and scalar failure data.
It is written under `outputs/autodiff_transport_lagged_ntx/fd_replay_failures/`.
The past failed process cannot be reconstructed from the scalar error text;
one new capture is needed to obtain these arrays.

After syncing, capture only the first 83 accepted steps on the GPU machine:

```bash
cd ~/NEOPAX
bash docs/benchmarks/capture_database_transition_fd_step83_2026-10-19.sh
```

This is a diagnostic prefix, not a 20-second AD/FD comparison. The expected
replay error still returns nonzero and prints `[fd-replay-snapshot] wrote ...`.
It avoids continuing the adaptive baseline to 20 seconds for each investigation.

To diagnose the most recently saved step-83 profile snapshot:

```bash
cd ~/NEOPAX
neopax_snapshot=$(ls -t outputs/autodiff_transport_lagged_ntx/fd_replay_failures/n0_minus_step_0083_*.npz | head -n 1)
PYTHONPATH="$HOME/VMEX:$HOME/NTX/src" \
python examples/benchmarks/diagnose_transport_fd_replay_step.py "$neopax_snapshot"
```

The probe reconstructs runtime geometry/database and solver context, then
restores the saved carries after checking their tree layout, shapes and dtypes.
It solves only that one slot with the baseline recorded setup, endpoint
recorded setup and endpoint cold setup. It reports incoming physical fields
per species, initial RHS and trial/stage/Jacobian summaries, real Newton matrix
conditioning and the actual divergence/contraction/nonfinite flags. It writes
a `.diagnosis.json` beside the NPZ. It never executes an adaptive rollout,
the preceding accepted steps, objectives or reverse AD. Runtime/model setup
still costs time. This diagnostic currently supports profile snapshots and
refuses CPU execution before constructing a physics runtime.

No Newton tolerance, accepted timestep, FD perturbation size, physical floor,
objective or production AD behavior was changed to hide the failure.

## Validation and remote retry

`tests/test_fd_frozen_metadata_replay.py` uses a one-variable ODE and mocked
step failures. It checks central differences against the analytic full-time
derivative, accepted-slot filtering and rebuilds, baseline/endpoint failure
attribution, rejection of missing metadata, and convergence/finite/time guards.
Recovery regressions verify that a failed trial is discarded, the original
physical state/timestep is retained, and restart counts identify the recovered
accepted slot. A scalar piecewise RHS reproduces a frozen-Jacobian Newton
failure and checks the cold-restart solution against the independent Radau
stability-function endpoint.
Existing mocked objective-routing and parser tests are included in validation.
No plasma transport, VMEC, NTX or FD benchmark is run locally.

Validation command:

```bash
JAX_PLATFORMS=cpu python -m pytest \
  tests/test_fd_frozen_metadata_replay.py \
  tests/test_reverse_benchmark_root_objectives.py \
  tests/test_database_reverse_benchmark_plumbing.py -q
```

Before the same-timestep recovery addition, this suite passed **65 tests and
33 subtests in 23.76 seconds**. The tests execute only scalar ODEs, synthetic
states and mocked benchmark setup. They do not establish that the remote
full-time FD endpoint is now valid.

After adding same-timestep recovery, validation passes **67 tests and 33
subtests**, run as 13 scalar/recovery tests (20.52 seconds) and 54 mocked
objective-routing/parser tests plus 33 subtests (18.47 seconds). The scalar
piecewise-RHS case confirms failure with the frozen baseline Jacobian and
successful recovery with a fresh Jacobian at the unchanged timestep. Its
physical endpoint agrees with the independent Radau stability-function value.

After the cold-restart failure, the next remote action is the bounded prefix
capture and saved-step probe above. Another full 20-second FD retry is not
justified until the actual failing state has been diagnosed. Full-time FD
errors remain pending.

Snapshot/probe validation: **74 tests and 33 subtests passed in 25.17 seconds**
across the same three files. This includes NPZ round trips with
`allow_pickle=False`, rejection of changed carry layouts/shapes/dtypes,
preservation of the original exception if saving fails, a scalar single-step
probe that forbids adaptive rollout, and CPU rejection before physics setup.
