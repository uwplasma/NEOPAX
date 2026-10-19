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
root objective or bootstrap objective is introduced or changed. The GPU
failure's exact first bad step is still unknown from the supplied traceback.
Remote verification is needed to establish whether applying the recorded
rebuild decisions resolves that failure or exposes an additional cause.

## Validation and remote retry

`tests/test_fd_frozen_metadata_replay.py` uses a one-variable ODE and mocked
step failures. It checks central differences against the analytic full-time
derivative, accepted-slot filtering and rebuilds, baseline/endpoint failure
attribution, rejection of missing metadata, and convergence/finite/time guards.
Existing mocked objective-routing and parser tests are included in validation.
No plasma transport, VMEC, NTX or FD benchmark is run locally.

Validation command:

```bash
JAX_PLATFORMS=cpu python -m pytest \
  tests/test_fd_frozen_metadata_replay.py \
  tests/test_reverse_benchmark_root_objectives.py \
  tests/test_database_reverse_benchmark_plumbing.py -q
```

Result: **65 passed, 33 subtests passed in 23.76 seconds**. The tests execute
only scalar ODEs, synthetic states and mocked benchmark setup. They do not
establish that the remote full-time FD endpoint is now valid.

After syncing the correction, retry just `--parameter n0` with the same
full-time FD command and flags. The baseline banner should show 290 accepted
steps, final time 20 seconds and recorded Newton setup. If the replay remains
invalid, the new error identifies its first bad accepted step before bootstrap
evaluation. Complete this one-column check before running the remaining
seven columns; full-time FD errors remain pending.
