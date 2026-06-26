# Reverse AD Current State

## 2026-06-24 handoff

Current target:
- Recover the reverse AD lane for `benchmark_transport_reverse_ad_only.py`.
- Keep reverse AD independent from the production forward solver, forward AD lane, and FD lane.
- Match the known 16-accepted-step forward AD gradients for `objective=softmax_Er`.

Known 16-step forward AD reference values:
- `dsoftmax_Er/dn0 = -3.759631e+00`
- `dsoftmax_Er/dT0 = 3.054047e+00`
- `dsoftmax_Er/ddensity_shape_power = -8.518430e-02`
- `dsoftmax_Er/dtemperature_shape_power = 3.214063e+00`

Recent reverse state:
- `block` exact stage adjoint matched the forward values, but was memory-heavy.
- `structured` was lighter conceptually but did not match the forward values.
- `bicgstab` was added as an exact iterative stage-adjoint mode, but initially still had similar memory because it used generic full-residual VJPs.

Latest implementation change:
- `_radau_solve_exact_stage_residual_transpose_iterative` no longer builds a generic VJP over the full flattened stage residual.
- It now applies the exact Radau transpose matvec directly:
  `(dR/dz)^T lambda_j = lambda_j - h * sum_i a_ij * J_i^T lambda_i`.
- `_radau_exact_stage_residual_input_pullback` was added so exact reverse branches no longer use generic full-residual VJP for `(y, dt, lagged_response)` bars.
- The exact reverse branch now avoids:
  - dense block matrix construction in `bicgstab`,
  - generic VJP over stage residual wrt `z`,
  - generic VJP over full residual wrt `(y, dt, lagged_response)`.

Next test to run:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --ntx-exact-derivative-mode direct \
  --objective softmax_Er \
  --accepted-step-limit 16 \
  --radau-jacobian-reuse-mode legacy \
  --timing-mode jit-warm \
  --reverse-segment-length 4 \
  --reverse-stage-adjoint-solve-mode bicgstab
```

Expected correctness target:
- The gradients should remain close to the forward AD reference values above.

Expected performance/memory question:
- If memory is still high after the latest `bicgstab` fix, the remaining likely sources are per-stage RHS pullbacks themselves or full reverse segment checkpoints, not the previous generic residual VJP machinery.

Verification done:
- `compile(...)` no-write syntax check passed for `NEOPAX/_transport_solvers.py`.
- `compile(...)` no-write syntax check passed for `examples/benchmarks/benchmark_transport_reverse_ad_only.py`.
- Three implementation-review passes were done after the latest patch:
  - exact reverse branch wiring was checked,
  - stale generic residual VJP sites were searched,
  - benchmark reporting and syntax were checked.

Current status:
- Ready for the next session to run the `bicgstab` test command above.
- Do not change the production forward solver, forward AD lane, or FD lane while testing this reverse AD memory/correctness fix.

## 2026-06-25 handoff

Current correctness status:
- The reverse AD lane matches the known 16-accepted-step forward AD reference for `objective=softmax_Er`.
- The matching values are:
  - `dsoftmax_Er/dn0 = -3.759631e+00`
  - `dsoftmax_Er/dT0 = 3.054047e+00`
  - `dsoftmax_Er/ddensity_shape_power = -8.518430e-02`
  - `dsoftmax_Er/dtemperature_shape_power = 3.214064e+00`
- This was confirmed with `reverse_stage_adjoint_solve_mode=bicgstab`.

Recent performance/memory result:
- The explicit NTX interpolated RHS-transpose path is now correct, but it did **not** materially reduce memory.
- Latest explicit command:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --ntx-exact-derivative-mode direct \
  --objective softmax_Er \
  --accepted-step-limit 16 \
  --radau-jacobian-reuse-mode legacy \
  --timing-mode jit-warm \
  --reverse-segment-length 4 \
  --reverse-stage-adjoint-solve-mode bicgstab \
  --reverse-rhs-transpose-mode explicit_ntx_interpolated
```

- Latest explicit output:
  - `reverse_compile_plus_execute_s = 8.886653e+02`
  - `reverse_execute_s_mean = 2.358889e+02`
  - gradients matched the reference values above.
  - RAM was still about the same high level as the generic path, around half of a 30 GB host in the observed run.

Conclusion from the explicit NTX test:
- The state transpose inside the NTX interpolated RHS is not the dominant remaining memory source.
- Remaining likely memory sources are:
  - segment-level `jax.vjp(_segment_replay, segment_start_carry)`,
  - equation-level direct working-state VJPs,
  - shared-flux-to-RHS assembly VJPs,
  - lagged-response cotangent paths.

Latest implementation change after that conclusion:
- The reverse segmented checkpoint path was changed to compact to accepted attempts before checkpointing.
- Previously, segments were formed over the full attempt trace. For the 16-step case this printed `max_total_steps=27` and `reverse_checkpoint_count=7` with `reverse_segment_length=4`.
- Rejected attempts do not update the derivative carry in this benchmark reverse contract, so storing/replaying rejected-attempt slots inside checkpoint segments was dead structure.
- The new path:
  - gathers accepted attempt positions from the primal trace,
  - pads only the accepted-step sequence,
  - uses `_radau_replay_realized_accepted_rollout(...)` for segment forward storage,
  - uses `_radau_replay_realized_accepted_rollout(...)` for segment backward replay,
  - keeps the primal adaptive solve unchanged.
- The benchmark report was updated so `reverse_checkpoint_count` is based on `accepted_step_limit` when present. For `--accepted-step-limit 16 --reverse-segment-length 4`, it should now print `reverse_checkpoint_count=4`.

Files touched in the latest change:
- `NEOPAX/_transport_solvers.py`
  - `_radau_adaptive_final_y_realized_schedule_vjp_fwd(...)`
  - `_radau_adaptive_final_y_realized_schedule_vjp_bwd(...)`
- `examples/benchmarks/benchmark_transport_reverse_ad_only.py`
  - `reverse_checkpoint_count` reporting logic.

Verification done after latest patch:
- In-memory no-write syntax check passed:

```bash
python -c "from pathlib import Path; files=['NEOPAX/NEOPAX/_transport_solvers.py','NEOPAX/examples/benchmarks/benchmark_transport_reverse_ad_only.py']; [compile(Path(f).read_text(encoding='utf-8'), f, 'exec') for f in files]; print('compile ok')"
```

- Normal `compileall` was not used as evidence because this Windows checkout hit a `__pycache__` permission error.

Next test to run:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --ntx-exact-derivative-mode direct \
  --objective softmax_Er \
  --accepted-step-limit 16 \
  --radau-jacobian-reuse-mode legacy \
  --timing-mode jit-warm \
  --reverse-segment-length 4 \
  --reverse-stage-adjoint-solve-mode bicgstab \
  --reverse-rhs-transpose-mode explicit_ntx_interpolated
```

What to check in the next run:
- `reverse_checkpoint_count` should be `4`.
- Gradients should remain:
  - `n0 = -3.759631e+00`
  - `T0 = 3.054047e+00`
  - `density_shape_power = -8.518430e-02`
  - `temperature_shape_power = 3.214064e+00`
- Compare RAM and `reverse_execute_s_mean` against the previous explicit result:
  - previous `reverse_execute_s_mean = 2.358889e+02`
  - previous RAM was still roughly half of 30 GB.

If memory is still unchanged:
- Do not keep specializing the NTX interpolated state transpose first; it already proved correct but non-dominant.
- Next likely target is removing broad equation-level VJPs in:
  - `ComposedEquationSystem.pullback_evaluate_with_lagged_response_state(...)`
  - `ComposedEquationSystem.pullback_shared_fluxes(...)`
- In particular, check the generic VJPs around:
  - `direct_working_state_pullback = jax.vjp(...)`
  - per-equation `_density_map`, `_pressure_map`, `_er_map` VJPs.
- Also check whether `flat_rhs_lagged_response_pullback` still stages a large lagged-response VJP in the hot path.

## 2026-06-26 reverse accepted-step bwd localization

Current correct 16 accepted-step reference for:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --ntx-exact-derivative-mode direct \
  --objective softmax_Er \
  --accepted-step-limit 16 \
  --radau-jacobian-reuse-mode legacy \
  --timing-mode jit-warm \
  --reverse-segment-length 4 \
  --reverse-stage-adjoint-solve-mode bicgstab \
  --reverse-rhs-transpose-mode explicit_ntx_interpolated
```

is:

```text
dsoftmax_Er/dn0 = -3.759631e+00
dsoftmax_Er/dT0 = 3.054047e+00
dsoftmax_Er/ddensity_shape_power = -8.518430e-02
dsoftmax_Er/dtemperature_shape_power = 3.214064e+00
```

Latest diagnostic conclusions:

- `zero_rhs_flux` kept only the direct equation-assembly RHS-state cotangent.
  - It changed gradients, proving the mode was active.
  - Memory and warmed execution stayed essentially unchanged.
- `zero_rhs_direct` kept only the shared-flux/NTX RHS-state cotangent.
  - It changed gradients, proving the mode was active.
  - Memory and warmed execution again stayed essentially unchanged.
- `zero_stage_solve` bypassed the exact stage-adjoint solve and residual-input pullback.
  - Gradients became zero, as expected for this diagnostic.
  - `reverse_compile_plus_execute_s` dropped materially, but warmed execution was still about `2.36e+02 s`.
  - Interpretation: the exact stage solve contributes to compile graph size, but not most of the warmed execution plateau.
- `zero_step_bwd` bypassed the accepted-step backward body inside the segmented reverse scan.
  - Gradients became zero, as expected.
  - `reverse_compile_plus_execute_s = 2.839014e+02`
  - `reverse_execute_s_mean = 5.637258e+01`
  - RAM dropped to around the low 20-25% host range in the observed graph.
  - Interpretation: the remaining memory/runtime plateau is inside the accepted-step backward body.
- `force_reuse_bwd` forced the reuse branch in accepted-step bwd.
  - It reduced time/memory compared with full dynamic bwd.
  - It was wrong because this run has rebuild slots:
    - `reverse_lagged_reuse_count = 4`
    - `reverse_lagged_rebuild_count = 12`
  - Gradients changed to:

```text
dsoftmax_Er/dn0 = -3.661887e+00
dsoftmax_Er/dT0 = 2.967747e+00
dsoftmax_Er/ddensity_shape_power = -8.274362e-02
dsoftmax_Er/dtemperature_shape_power = 3.124692e+00
```

- `dynamic_call_bwd` kept the dynamic reuse/rebuild branch but placed each branch body behind a non-inlined `jax.jit(..., inline=False)` call boundary.
  - It was correct:

```text
dsoftmax_Er/dn0 = -3.759631e+00
dsoftmax_Er/dT0 = 3.054047e+00
dsoftmax_Er/ddensity_shape_power = -8.518430e-02
dsoftmax_Er/dtemperature_shape_power = 3.214063e+00
```

  - It did not materially reduce memory or warmed execution:
    - `reverse_compile_plus_execute_s = 8.459905e+02`
    - `reverse_execute_s_mean = 2.389553e+02`
  - Interpretation: the dynamic branch remains correct, but XLA still effectively carries the heavy branch bodies. The call boundary is not enough.

Important implementation/strategy note:

- The final reverse lane should not expose user-controlled reuse/rebuild forcing.
- The solver should choose reuse vs rebuild from the primal/replayed accepted-step logic.
- Saving one compact branch bit per accepted step is compatible with checkpointing; saving full branch-local arrays is not.
- True checkpointing should still store only checkpoint carries plus compact metadata such as:
  - accepted `dt[k]`,
  - accepted active mask,
  - incoming lagged branch bit,
  - minimal status bits if needed.
- Binomial/Revolve checkpointing is a later improvement to the checkpoint placement/recompute schedule. It does not by itself remove the heavy accepted-step bwd body.

Additional failed experiment:

- `branch_schedule_bwd` used the baseline realized accepted-step branch schedule as compact metadata, but implemented it with a Python/static unrolled reverse loop.
  - This exploded host memory and the process was killed.
  - Interpretation: naive static unrolling duplicates the heavy accepted-step bwd graph and is not viable.
  - This mode was removed from executable code and from the benchmark CLI choices.
  - Any future branch-scheduled implementation must avoid unrolling the heavy per-step body. It would need a genuinely compact custom primitive/call strategy or grouped kernels that do not duplicate the whole graph.

Current code diagnostics/modes that remain:

- `zero_rhs_direct`
- `zero_rhs_flux`
- `zero_stage_solve`
- `zero_step_bwd`
- `force_reuse_bwd`
- `force_rebuild_bwd`
- `dynamic_call_bwd`

Next direction:

- Do not continue with naive static branch unrolling.
- The dynamic branch remains correct but expensive; `dynamic_call_bwd` proved a non-inlined JIT call boundary did not reduce memory/time.
- The next viable optimization should keep the dynamic solver decision while reducing the accepted-step bwd body itself, or introduce a real custom primitive/call boundary that does not inline/duplicate the reuse and rebuild bwd bodies.
- Keep `zero_step_bwd` as the main localization diagnostic: it proves the accepted-step bwd body is the hot path.
