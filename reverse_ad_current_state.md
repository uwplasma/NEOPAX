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
- `zero_prev_stages_bwd` dropped cotangents through the next-step predictor stage history.
  - It preserved the correct gradients.
  - It did not reduce memory or warmed execution and made compile+execute worse:
    - `reverse_compile_plus_execute_s = 1.109335e+03`
    - `reverse_execute_s_mean = 2.401833e+02`
  - This diagnostic was removed from executable code/CLI because it adds noise without helping memory.

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
- `zero_rebuild_pullback`
- `zero_step_bwd`
- `force_reuse_bwd`
- `force_rebuild_bwd`
- `dynamic_call_bwd`

Next direction:

- Do not continue with naive static branch unrolling.
- The dynamic branch remains correct but expensive; `dynamic_call_bwd` proved a non-inlined JIT call boundary did not reduce memory/time.
- The next viable optimization should keep the dynamic solver decision while reducing the accepted-step bwd body itself, or introduce a real custom primitive/call boundary that does not inline/duplicate the reuse and rebuild bwd bodies.
- Keep `zero_step_bwd` as the main localization diagnostic: it proves the accepted-step bwd body is the hot path.

2026-06-26 fixed-replay cleanup experiment:

- Tried an internal `fixed_replay` path in `_execute_radau_accepted_step_next_carry_vjp_lagged_branch_bwd`.
  - The idea was to avoid constructing cotangents for replay-only fields such as the accepted `dt` inside the branch bwd.
  - It preserved gradients but did not improve memory or warmed execution.
  - Test result:
    - `reverse_compile_plus_execute_s = 1.028776e+03`
    - `reverse_execute_s_mean = 2.396921e+02`
  - This was removed/reverted because it added complexity without helping.

Current reference command:

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

- Expected correctness target remains:
  - `dsoftmax_Er/dn0 = -3.759631e+00`
  - `dsoftmax_Er/dT0 = 3.054047e+00`
  - `dsoftmax_Er/ddensity_shape_power = -8.518430e-02`
  - `dsoftmax_Er/dtemperature_shape_power = 3.214064e+00`

Next optimization target:

- The cost is not fixed replay field cotangents or `prev_stages` cotangents.
- The remaining hot path is the accepted-step bwd body itself.
- `zero_step_bwd` remains the most important localization result because it drops execution to roughly `5.6e+01` seconds.
- Further improvement needs to reduce the actual stage/RHS/lagged adjoint work or package it behind a truly smaller custom primitive/checkpoint strategy; masking carry fields after or inside the step is not enough.

2026-06-26 diagnostic addition:

- Added `--reverse-stage-cotangent-mode zero_rebuild_pullback`.
  - This skips only `pullback_build_lagged_response` in rebuild branches.
  - It intentionally changes gradients.
  - It is meant to isolate whether the 12 rebuild branches are a major compile/memory/runtime source.
- Test result:
  - `reverse_compile_plus_execute_s = 8.447915e+02`
  - `reverse_execute_s_mean = 2.101824e+02`
  - Gradients changed, as expected:
    - `dsoftmax_Er/dn0 = -3.656530e+00`
    - `dsoftmax_Er/dT0 = 3.282627e+00`
    - `dsoftmax_Er/ddensity_shape_power = -8.275152e-02`
    - `dsoftmax_Er/dtemperature_shape_power = 4.491006e+00`
  - Interpretation: rebuild pullback is a real contributor, but not the whole hot path.
- Follow-up implementation:
  - In `NTXExactLijTransportModel.pullback_build_lagged_response`, replaced a local `jax.linear_transpose(jax.jvp(...))` for the simple map `(density, pressure) -> (safe_density, pressure / safe_density)` with the explicit algebraic pullback.
  - This is general NTX pullback cleanup, not a benchmark-specific shortcut.
- Full correctness/performance test after that cleanup:
  - Correct gradients were preserved.
  - It did not help performance:
    - `reverse_compile_plus_execute_s = 1.102192e+03`
    - `reverse_execute_s_mean = 2.429593e+02`
  - This cleanup was reverted because it made the full path slightly worse.
- Added `--reverse-stage-cotangent-mode zero_rebuild_anchor_fields`.
  - This keeps only the direct `reference_er` part of the NTX interpolated rebuild pullback.
  - It skips the anchor-field interpolation/local moment-response pullback inside rebuild.
  - It intentionally changes gradients.
  - Goal: determine whether the rebuild cost comes mostly from the local NTX interpolated anchor-field transpose.
- Test result:
  - `reverse_compile_plus_execute_s = 8.805706e+02`
  - `reverse_execute_s_mean = 2.075852e+02`
  - Gradients changed, as expected:
    - `dsoftmax_Er/dn0 = -2.659934e+00`
    - `dsoftmax_Er/dT0 = 2.395321e+00`
    - `dsoftmax_Er/ddensity_shape_power = -6.021805e-02`
    - `dsoftmax_Er/dtemperature_shape_power = 3.289570e+00`
  - Interpretation: almost all of the rebuild-pullback runtime saving comes from skipping the interpolated anchor/local moment-response pullback; the direct `reference_er` piece is cheap.
- Added `--reverse-stage-cotangent-mode zero_rebuild_local_moment_pullback`.
  - This keeps the interpolation transpose to anchor bars.
  - It skips only the per-anchor/species local NTX moment-response pullback.
  - It intentionally changes gradients.
  - Goal: separate interpolation transpose cost from local moment-response physics pullback cost.
- Implementation started on the real local moment-response pullback reduction:
  - Replaced the local `linear_transpose(jvp(get_v_thermal))` inside `_pullback_local_scan_inputs_from_primitives` with the analytic pullback `temperature_bar += vthermal_bar * vthermal / (2 * temperature)`.
  - Full test preserved gradients but did not improve performance:
    - `reverse_compile_plus_execute_s = 1.107230e+03`
    - `reverse_execute_s_mean = 2.434195e+02`
  - This was reverted because it added code without reducing the full-path graph cost.
- Next implementation attempt:
  - Replaced the full black-box transpose of `_local_scan_inputs` with a hybrid pullback.
  - `nu_hat` still uses a generic VJP through `_nu_over_vnew_local` to preserve the active collisionality model.
  - `epsi_hat` and `vth_a` are transposed explicitly, including finite-`drds` and `er_v_floor` activity.
  - This is a larger graph reduction than the scalar `get_v_thermal` substitution and should be tested on the full path.
- Validation already done:
  - Syntax compile passed for `_transport_flux_models.py`, `_transport_solvers.py`, `_transport_equations.py`, and `benchmark_transport_reverse_ad_only.py`.
  - `git diff --check` passed, with only line-ending warnings.
- Next full correctness/performance command:

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

- Keep/revert criteria:
  - Keep only if the gradients remain at the correctness target:
    - `dsoftmax_Er/dn0 = -3.759631e+00`
    - `dsoftmax_Er/dT0 = 3.054047e+00`
    - `dsoftmax_Er/ddensity_shape_power = -8.518430e-02`
    - `dsoftmax_Er/dtemperature_shape_power = 3.214064e+00`
  - Also require a meaningful improvement in compile, warmed execution, or memory. If it preserves gradients but remains around `reverse_execute_s_mean ~= 2.4e+02` and `reverse_compile_plus_execute_s ~= 1.1e+03`, revert it like the previous scalar pullback attempts.
  - If gradients change, inspect the explicit `epsi_hat`/`er_v_floor` activity logic first; likely issue would be the floor boundary or finite-`drds` branch in the hybrid pullback.

Current useful localization facts:

- `zero_step_bwd` remains the strongest localization: accepted-step bwd is the main hot path.
- `zero_rebuild_pullback` and `zero_rebuild_anchor_fields` showed rebuild pullback is a contributor, and most of that contribution is in the interpolated anchor/local moment-response part.
- Micro-optimizations to simple scalar transposes did not help; the current hybrid local-scan pullback is the first larger graph-reduction attempt after those negative results.

2026-06-27 handoff:

- Current code state includes the hybrid `_local_scan_inputs` pullback attempt in the NTX interpolated local moment-response path.
- This attempt is intended to reduce reverse graph size generically, not to special-case the active benchmark TOML.
- Full correctness/performance test result:
  - Gradients remained correct:
    - `dsoftmax_Er/dn0 = -3.759631e+00`
    - `dsoftmax_Er/dT0 = 3.054047e+00`
    - `dsoftmax_Er/ddensity_shape_power = -8.518430e-02`
    - `dsoftmax_Er/dtemperature_shape_power = 3.214064e+00`
  - Timing did not meaningfully improve:
    - `reverse_total_s = 1.273964e+03`
    - `reverse_compile_plus_execute_s = 1.032273e+03`
    - `reverse_execute_s_mean = 2.416912e+02`
    - `reverse_execute_s_min = 2.416912e+02`
  - Resource graph still showed high host RAM during compile and a long GPU execution plateau; memory looked broadly similar to the prior full path.
- Conclusion: the hybrid `_local_scan_inputs` pullback preserves correctness but fails the keep criteria because memory and warmed execution remain essentially unchanged.
- Next action should be to revert the hybrid local-scan pullback attempt and return to the last known-correct full path before choosing the next optimization target.
- Git currently reports a stale `.git/index.lock` warning in this checkout; do not remove it unless intentionally cleaning up Git state.

2026-06-30 handoff:

- Recent memory-reduction diagnostics did not identify a cheap subfield fix.
- `remat_rebuild_pullback` was tested on the 16-step reverse lane.
  - It preserved the correct gradients:
    - `dsoftmax_Er/dn0 = -3.759631e+00`
    - `dsoftmax_Er/dT0 = 3.054047e+00`
    - `dsoftmax_Er/ddensity_shape_power = -8.518430e-02`
    - `dsoftmax_Er/dtemperature_shape_power = 3.214064e+00`
  - It made performance much worse:
    - `reverse_total_s = 1.919843e+03`
    - `reverse_compile_plus_execute_s = 1.285655e+03`
    - `reverse_execute_s_mean = 6.341880e+02`
  - It did not reduce the RAM plateau enough to justify keeping.
  - This mode was reverted/removed from the intended working state.
- `zero_rebuild_derivative_fields` was tested on the 16-step reverse lane.
  - It intentionally changed gradients and also made performance worse:
    - `reverse_total_s = 1.916134e+03`
    - `reverse_compile_plus_execute_s = 1.288350e+03`
    - `reverse_execute_s_mean = 6.277836e+02`
    - `dsoftmax_Er/dn0 = -3.770725e+00`
    - `dsoftmax_Er/dT0 = 3.060427e+00`
    - `dsoftmax_Er/ddensity_shape_power = -8.536199e-02`
    - `dsoftmax_Er/dtemperature_shape_power = 3.223957e+00`
  - This rules out the derivative-field rebuild pullbacks as the main memory target.
  - This diagnostic mode was reverted/removed from the intended working state.
- An attempted NTX `solve_prepared_coefficient_vector_vjp(...)` boundary for the reverse local transport-moment pullback failed before performance testing.
  - Failure:
    - `UnexpectedTracerError`
    - `prepared` was passed as an argument marked by NTX as `nondiff_argnums=(0,)`, but in this reverse rebuild path `prepared` contains JAX tracers.
  - This confirms the warning in the older notes: the existing NTX custom-VJP API cannot be called directly from the traced reverse rebuild path.
  - The attempted branch was reverted/removed from the intended working state.
- Current evidence:
  - There is not enough evidence to claim that a new NTX coefficient-solve boundary will reduce the graph.
  - There is enough evidence to justify it only as a small proof-of-concept experiment.
  - Do not start a large refactor assuming it will help.
- Next recommended step:
  - Prototype a tiny JAX-compatible coefficient-solve boundary where only truly static metadata is nondifferentiable and the prepared array payload is passed as ordinary JAX operands.
  - Compare compile/JAXPR/HLO or a 2-step reverse run before generalizing.
  - Only continue if this prototype measurably reduces graph size, compile time, RAM, or warmed execution.
- Avoid next session:
  - Do not rerun `remat_rebuild_pullback`.
  - Do not rerun `zero_rebuild_derivative_fields`.
  - Do not rerun the existing NTX `custom_vjp` path inside reverse rebuild without changing the NTX/NEOPAX boundary API.

2026-07-01 handoff:

- A JAX-compatible NTX coefficient-solve boundary proof was tested as an opt-in `array_custom_vjp` mode.
- Implementation shape:
  - The first attempt passed `prepared` as a differentiable custom-VJP argument and failed with:
    - `TypeError: object of type 'object' has no len()`
    - Cause: JAX reconstructed a zero cotangent PyTree for NTX geometry metadata and triggered `GeometryOnGrid.__post_init__`.
  - A closure-based boundary, where `prepared` was captured and only `(nu_hat, epsi_hat)` were VJP arguments, fixed correctness.
- 2 accepted-step reuse-only result:
  - Command used `--accepted-step-limit 2`, `--reverse-segment-length 2`, `--reverse-stage-adjoint-solve-mode bicgstab`, `--reverse-rhs-transpose-mode explicit_ntx_interpolated`, and `--reverse-ntx-coeff-boundary array_custom_vjp`.
  - Branch counts:
    - `reverse_lagged_reuse_count = 2`
    - `reverse_lagged_rebuild_count = 0`
  - Gradients matched the known 2-step targets:
    - `dsoftmax_Er/dn0 = -3.578617e-01`
    - `dsoftmax_Er/dT0 = 3.010300e-01`
    - `dsoftmax_Er/ddensity_shape_power = -7.886159e-03`
    - `dsoftmax_Er/dtemperature_shape_power = 1.779140e-01`
  - Timing:
    - `reverse_total_s = 6.671918e+02`
    - `reverse_compile_plus_execute_s = 6.592605e+02`
    - `reverse_execute_s_mean = 7.931390e+00`
  - Interpretation:
    - Warm execution improved in the small reuse-only case compared with the recent default roughly `18 s`.
    - Compile time and compile-memory class did not improve.
- 16 accepted-step mixed reuse/rebuild result:
  - Command used `--accepted-step-limit 16`, `--reverse-segment-length 4`, `--reverse-stage-adjoint-solve-mode bicgstab`, `--reverse-rhs-transpose-mode explicit_ntx_interpolated`, and `--reverse-ntx-coeff-boundary array_custom_vjp`.
  - Branch counts:
    - `reverse_lagged_reuse_count = 4`
    - `reverse_lagged_rebuild_count = 12`
  - Gradients matched the full 16-step correctness target:
    - `dsoftmax_Er/dn0 = -3.759631e+00`
    - `dsoftmax_Er/dT0 = 3.054047e+00`
    - `dsoftmax_Er/ddensity_shape_power = -8.518430e-02`
    - `dsoftmax_Er/dtemperature_shape_power = 3.214064e+00`
  - Timing:
    - `reverse_total_s = 1.176691e+03`
    - `reverse_compile_plus_execute_s = 9.279624e+02`
    - `reverse_execute_s_mean = 2.487290e+02`
  - Resource graph:
    - RAM plateau still looked broadly similar to the previous full reverse path.
  - Interpretation:
    - The coefficient boundary is correct but is not a real compile-time or memory fix.
    - It did not improve the actual 16-step mixed reuse/rebuild target and was slightly worse than recent full-path warmed execution timings.
  - Action taken:
    - Removed the `array_custom_vjp` / `reverse_ntx_coefficient_boundary` experiment and the benchmark CLI flag to avoid keeping a misleading optimization path.
- Current corrected conclusion:
  - The real compile-time and memory problem is not the NTX coefficient solve boundary.
  - The decisive localization remains `zero_step_bwd`: the accepted-step backward body is the dominant hot path.
  - Static branch scheduling / Python unrolling is not the path forward; it previously exploded host memory.
  - `dynamic_call_bwd` did not materially reduce memory or warmed execution.
- Next real compilation-time and memory-reduction target:
  - Work inside `_execute_radau_accepted_step_next_carry_vjp_lagged_branch_bwd`, not in the NTX coefficient pullback.
  - Remove broad nested `jax.vjp(...)` calls from the accepted-step backward body where possible.
  - First concrete targets:
    - The stage-lagged RHS pullback in `_direct_stage_cache_adjoint`, currently built through a broad VJP of `_stage_evals_from_lagged`.
    - Projector pullbacks from `jax.vjp(physics_context.project_flat, ...)` if they are still nontrivial in the active path.
    - Rebuild `pullback_build_lagged_response(...)` only after the stage-lagged RHS pullback is narrowed, because rebuild-only subfield masking did not solve memory by itself.
  - If explicit/manual accepted-step pullbacks still leave compile RAM unchanged, the remaining real solution is compilation granularity:
    - Stop compiling the whole reverse rollout as one monolithic `jax.grad(objective_fn)`.
    - Move toward separately compiled segment/accepted-step adjoint kernels orchestrated outside a single XLA module.
    - This is a larger design change, but it is the credible path if the accepted-step bwd graph remains too large.

Reverse AD compile-time and memory reduction plan:

1. Freeze the current baseline.
   - Use the known-correct 16-step mixed reuse/rebuild command:
     - `--ntx-exact-derivative-mode direct`
     - `--objective softmax_Er`
     - `--accepted-step-limit 16`
     - `--radau-jacobian-reuse-mode legacy`
     - `--timing-mode jit-warm`
     - `--reverse-segment-length 4`
     - `--reverse-stage-adjoint-solve-mode bicgstab`
     - `--reverse-rhs-transpose-mode explicit_ntx_interpolated`
   - Correct gradient target:
     - `dsoftmax_Er/dn0 = -3.759631e+00`
     - `dsoftmax_Er/dT0 = 3.054047e+00`
     - `dsoftmax_Er/ddensity_shape_power = -8.518430e-02`
     - `dsoftmax_Er/dtemperature_shape_power = 3.214064e+00`
   - Current representative timings are still too heavy:
     - compile plus first execute is roughly `9e+02` to `1.1e+03 s`
     - warm execute is roughly `2.4e+02 s`
     - RAM plateau remains high during the monolithic JIT compile.

2. Target accepted-step backward internals first.
   - Do not return to static branch scheduling or Python/static unrolling:
     - static schedule/unroll previously exploded host memory.
     - `dynamic_call_bwd` did not materially improve memory or warmed execution.
   - Do not continue coefficient-boundary experiments for memory:
     - the `array_custom_vjp` proof was correct but did not improve the real 16-step compile/RAM target.
   - Work inside `_execute_radau_accepted_step_next_carry_vjp_lagged_branch_bwd`.
   - Keep `zero_step_bwd` as the main localization evidence: accepted-step bwd is the hot path.

3. First concrete implementation target: stage-lagged RHS pullback.
   - Current broad path in `_direct_stage_cache_adjoint` builds a nested VJP around `_stage_evals_from_lagged`:
     - `_, lagged_pullback = jax.vjp(_stage_evals_from_lagged, lagged_response)`
     - `(residual_lagged_bar,) = lagged_pullback(stage_rhs_bar)`
   - Replace this with an explicit/manual lagged-response pullback when available, using existing `physics_context.flat_rhs_lagged_response_pullback`-style machinery.
   - Keep the default path unchanged behind a mode flag during testing.
   - Suggested mode name:
     - `reverse_stage_lagged_pullback_mode = "generic" | "explicit"`
   - Suggested benchmark flag:
     - `--reverse-stage-lagged-pullback-mode explicit`

4. Test sequence for the explicit stage-lagged pullback.
   - First run a 2-step test to catch correctness or tracing failures cheaply.
   - Then run the real 16-step mixed reuse/rebuild case.
   - Keep criteria:
     - gradients match the 16-step correctness target,
     - compile time or compile RAM meaningfully improves, or
     - warm execution meaningfully improves without memory regression.
   - Revert criteria:
     - gradients change without a clear, intentional reason,
     - compile RAM is unchanged and execution worsens,
     - implementation adds branch/mode clutter without evidence of helping the main target.

5. Second concrete target if stage-lagged pullback is not enough.
   - Inspect and narrow remaining broad pullbacks inside accepted-step bwd:
     - `jax.vjp(physics_context.project_flat, ...)`
     - rebuild `pullback_build_lagged_response(...)`
   - Only target rebuild after the stage-lagged RHS pullback is narrowed, because rebuild-only subfield masking did not solve memory by itself.

6. Larger fallback if accepted-step manual pullbacks do not reduce compile RAM.
   - Change compilation granularity instead of continuing local micro-optimizations.
   - Avoid compiling the whole reverse rollout as one monolithic `jax.grad(objective_fn)` XLA module.
   - Move toward separately compiled segment or accepted-step adjoint kernels, orchestrated outside one giant compiled graph.
   - This is a larger design change, but it is the credible path if the accepted-step bwd graph remains too large after explicit pullbacks.

2026-07-01 explicit stage-lagged pullback result:

- Implemented an opt-in `--reverse-stage-lagged-pullback-mode explicit` proof that replaced the broad `_stage_evals_from_lagged` VJP inside `_direct_stage_cache_adjoint` with the existing `physics_context.flat_rhs_lagged_response_pullback` hook.
- 16-step mixed reuse/rebuild result:
  - Command used:
    - `--accepted-step-limit 16`
    - `--reverse-segment-length 4`
    - `--reverse-stage-adjoint-solve-mode bicgstab`
    - `--reverse-rhs-transpose-mode explicit_ntx_interpolated`
    - `--reverse-stage-lagged-pullback-mode explicit`
  - Branch counts:
    - `reverse_lagged_reuse_count = 4`
    - `reverse_lagged_rebuild_count = 12`
  - Gradients remained correct:
    - `dsoftmax_Er/dn0 = -3.759631e+00`
    - `dsoftmax_Er/dT0 = 3.054047e+00`
    - `dsoftmax_Er/ddensity_shape_power = -8.518430e-02`
    - `dsoftmax_Er/dtemperature_shape_power = 3.214064e+00`
  - Timing:
    - `reverse_total_s = 1.291324e+03`
    - `reverse_compile_plus_execute_s = 1.049949e+03`
    - `reverse_execute_s_mean = 2.413754e+02`
  - Resource graph:
    - RAM plateau remained in the same broad class as the previous full reverse path.
- Conclusion:
  - This explicit lagged-response pullback is correct but does not solve compile time or memory.
  - It fails the keep criteria because compile plus execute worsened and warm execution/RAM did not materially improve.
  - The opt-in mode and CLI flag were removed to avoid carrying another dead memory path.
- Updated next-step conclusion:
  - At this point several local graph-narrowing attempts have preserved correctness but failed to reduce compile RAM:
    - NTX coefficient boundary
    - explicit stage-lagged RHS pullback
    - rebuild subfield masking/remat attempts
    - dynamic branch call boundary
  - The credible next implementation target is compilation granularity, not another local pullback micro-optimization:
    - stop using one monolithic `jax.grad(objective_fn)` for the whole reverse rollout,
    - split reverse execution into separately compiled segment/accepted-step adjoint kernels,
    - orchestrate the reverse sweep outside a single XLA module while preserving the realized accepted-step schedule.

2026-07-01 reduced-cotangent accepted-step bwd attempt:

- Implemented an opt-in `--reverse-step-bwd-mode reduced_cotangent` path.
- This is not the old option-4 path and does not save full per-step lagged/RHS payloads.
- The new path adds `_RadauAcceptedStepReducedCotangent` and makes the segmented reverse scan carry only:
  - final-state cotangent `y`,
  - lagged-response cache cotangent,
  - lagged-reference-y cotangent.
- It structurally omits the `next_carry_bar.prev_stages` cotangent path inside the accepted-step bwd body.
  - This is based on the earlier `zero_prev_stages_bwd` diagnostic, which preserved the final-state objective gradients but did not reduce memory because it masked too late.
- Reuse vs rebuild is still chosen dynamically from the replayed/primal accepted-step carry via `residual_carry.lagged_response_valid`.
- First test should be the 2-step correctness run:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --ntx-exact-derivative-mode direct \
  --objective softmax_Er \
  --accepted-step-limit 2 \
  --radau-jacobian-reuse-mode legacy \
  --timing-mode jit-warm \
  --reverse-segment-length 2 \
  --reverse-stage-adjoint-solve-mode bicgstab \
  --reverse-rhs-transpose-mode explicit_ntx_interpolated \
  --reverse-step-bwd-mode reduced_cotangent
```

- Expected 2-step gradients from the forward/reverse reference:
  - `dsoftmax_Er/dn0 = -3.578617e-01`
  - `dsoftmax_Er/dT0 = 3.010300e-01`
  - `dsoftmax_Er/ddensity_shape_power = -7.886159e-03`
  - `dsoftmax_Er/dtemperature_shape_power = 1.779140e-01`
- If the 2-step run matches, test the 16-step mixed reuse/rebuild case with the same `--reverse-step-bwd-mode reduced_cotangent` flag.

Reduced-cotangent test results:

- 2 accepted steps, reuse-only:
  - `reverse_lagged_reuse_count = 2`
  - `reverse_lagged_rebuild_count = 0`
  - Correct gradients:
    - `dsoftmax_Er/dn0 = -3.578617e-01`
    - `dsoftmax_Er/dT0 = 3.010300e-01`
    - `dsoftmax_Er/ddensity_shape_power = -7.886159e-03`
    - `dsoftmax_Er/dtemperature_shape_power = 1.779140e-01`
  - Timing:
    - `reverse_total_s = 7.967464e+02`
    - `reverse_compile_plus_execute_s = 7.893813e+02`
    - `reverse_execute_s_mean = 7.365115e+00`
  - Interpretation:
    - Correctness is good.
    - Warm execution improved for the 2-step reuse-only case.
    - Compile time/RAM still did not improve enough.
- 16 accepted steps, mixed reuse/rebuild:
  - `reverse_lagged_reuse_count = 4`
  - `reverse_lagged_rebuild_count = 12`
  - Correct gradients:
    - `dsoftmax_Er/dn0 = -3.759631e+00`
    - `dsoftmax_Er/dT0 = 3.054047e+00`
    - `dsoftmax_Er/ddensity_shape_power = -8.518430e-02`
    - `dsoftmax_Er/dtemperature_shape_power = 3.214064e+00`
  - Timing:
    - `reverse_total_s = 1.202889e+03`
    - `reverse_compile_plus_execute_s = 9.634324e+02`
    - `reverse_execute_s_mean = 2.394570e+02`
  - Resource graph:
    - RAM plateau remained in the same broad class as the full/current mixed run.
  - Interpretation:
    - Reduced-cotangent is correct but does not solve the real 16-step mixed reuse/rebuild memory/runtime plateau.
    - The dominant remaining cost is still inside the accepted-step bwd body, especially the rebuild/stage machinery that reduced-cotangent still invokes.
- Follow-up rebuild cleanup:
  - Replaced the per-anchor generic `jax.linear_transpose(jax.jvp(...))` for the simple local map `(density, pressure) -> (safe_density, pressure / safe_density)` inside `NTXExactLijRuntimeTransportModel.pullback_build_lagged_response` with an explicit algebraic pullback.
  - 16-step reduced-cotangent result after cleanup:
    - Correct gradients preserved:
      - `dsoftmax_Er/dn0 = -3.759631e+00`
      - `dsoftmax_Er/dT0 = 3.054047e+00`
      - `dsoftmax_Er/ddensity_shape_power = -8.518430e-02`
      - `dsoftmax_Er/dtemperature_shape_power = 3.214064e+00`
    - Timing:
      - `reverse_total_s = 1.177058e+03`
      - `reverse_compile_plus_execute_s = 9.363880e+02`
      - `reverse_execute_s_mean = 2.406704e+02`
    - Resource graph:
      - RAM plateau remained in the same broad class.
  - Interpretation:
    - This explicit algebraic cleanup is correct but not enough to reduce the real mixed-case memory/runtime plateau.

Split local-scan pullback attempt:

- Replaced the broad `_pullback_local_scan_inputs_from_primitives` transpose over the full `_local_scan_inputs(er, T, n, vthermal)` map with:
  - a VJP over the smaller `nu_hat(T, n)` collisionality submap,
  - explicit algebraic pullbacks for `epsi_hat(Er, vth)` and `vth(T)`.
- The first version incorrectly used `jax.linear_transpose` directly on nonlinear `nu_hat(T, n)`, which raised an `AssertionError` during reverse compilation.
- Fixed by using `jax.vjp` for the nonlinear `nu_hat` submap while keeping the explicit `epsi_hat/vth` pullback.
- Next test is the same 16-step reduced-cotangent command.

2026-07-02 XLA allocation evidence and interpolated-response builder change:

- XLA dump inspection for the 2-step reverse compile showed that the dominant `jit__lambda` module is not primarily a checkpoint-array issue.
- The dominant preallocated temp was:
  - `allocation 20123`, size about `1.66 GiB`.
- The large live buffers were NTX coefficient/factorization tensors, including:
  - `f64[33,3,4,105,105]`
  - `f64[396,105,105]`
  - `f64[3,4,3,105,105]`
  - `f64[12,210,105]`
- HLO metadata mapped these buffers to NTX factorization internals:
  - `/home/exouser/NTX/src/ntx/_solver_factorization.py`, especially LU factorization/solve and scan dynamic-update-slice regions.
  - The call chain included `vmap(jvp(vmap()))` and `vmap(jvp(vmap(jit(lu_factor/lu_solve))))`.
- Interpretation:
  - The remaining memory/compile problem is not mainly the Radau reduced-cotangent carrier.
  - The largest graph comes from differentiating or linearizing batched NTX coefficient solves over the 33-point energy scan.
  - Any real memory reduction now has to stop the reverse compile from materializing full `33 x ... x 105 x 105` NTX factorization arrays.

Latest code change:

- Changed `NTXExactLijRuntimeTransportModel._build_interpolated_moment_response_local`.
- It no longer uses one full-scan `jax.linearize(...)` over:
  - `_transport_moments_from_inputs(prepared, reference_nu_hat, reference_epsi_hat, ...)`
- It now calls the existing reduced helper:
  - `_interpolated_moment_reduced_local_outputs_from_primitives(...)`
- The intent is to keep the same four interpolated-response outputs:
  - `reference_log_nu_star`
  - `reference_transport_moments`
  - `dtransport_moments_d_er`
  - `dtransport_moments_d_log_nu_star`
- But the derivative fields should be built through the existing per-energy helpers rather than through a monolithic full-scan linearization.
- This is solver-general and not benchmark-specific.

Validation done locally:

- No-write Python compile check passed:

```bash
python -c "from pathlib import Path; source=Path('NEOPAX/_transport_flux_models.py').read_text(); compile(source, 'NEOPAX/_transport_flux_models.py', 'exec'); print('compile-ok')"
```

- `python -m py_compile NEOPAX/_transport_flux_models.py` was not usable on Windows because writing `NEOPAX/__pycache__` failed with access denied.
- `git status --short` shows only:
  - `M NEOPAX/_transport_flux_models.py`

Current caveat:

- `NEOPAX/_transport_flux_models.py` also contains the previous split local-scan pullback work in `_pullback_local_scan_inputs_from_primitives`.
- That change should be interpreted separately from the latest interpolated-response-builder change.
- If the next benchmark fails, check whether the failure is from:
  - the previous local-scan pullback path, or
  - the new builder path.

Next run:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --ntx-exact-derivative-mode direct \
  --objective softmax_Er \
  --accepted-step-limit 2 \
  --radau-jacobian-reuse-mode legacy \
  --timing-mode jit-warm \
  --reverse-segment-length 2 \
  --reverse-stage-adjoint-solve-mode bicgstab \
  --reverse-rhs-transpose-mode explicit_ntx_interpolated \
  --reverse-step-bwd-mode reduced_cotangent
```

Expected 2-step gradients:

- `dsoftmax_Er/dn0 = -3.578617e-01`
- `dsoftmax_Er/dT0 = 3.010300e-01`
- `dsoftmax_Er/ddensity_shape_power = -7.886159e-03`
- `dsoftmax_Er/dtemperature_shape_power = 1.779140e-01`

What to inspect after the run:

- Correctness first:
  - gradients should match the expected 2-step values above.
- Compile/memory second:
  - compare compile time against the recent 2-step reduced-cotangent baseline, roughly `6.5e+02 s` compile-plus-execute / about `7.3 s` warm execution.
  - visually compare RAM plateau against the prior 2-step graphs, which reached roughly the same broad `35-45%` class.
- If compile RAM/time does not improve, rerun the XLA dump and check whether `allocation 20123` still contains:
  - `f64[33,3,4,105,105]`
  - `source_file="/home/exouser/NTX/src/ntx/_solver_factorization.py"`
- If those tensors are gone or reduced, the patch moved the right graph component.
- If those tensors remain unchanged, the remaining source is likely the reverse pullback of derivative fields, especially:
  - `_pullback_dtransport_moments_d_er_from_scan_primitives`
  - `_pullback_dtransport_moments_d_log_nu_star_from_scan_primitives`
  - both still use `jax.jvp` of `_pullback_transport_moments_from_scan_primitives`.

2026-07-02 result of interpolated-response builder split:

- The builder split was tested and should be treated as a failed optimization, not a keeper.
- 2-step correctness was preserved:
  - `dsoftmax_Er/dn0 = -3.578617e-01`
  - `dsoftmax_Er/dT0 = 3.010300e-01`
  - `dsoftmax_Er/ddensity_shape_power = -7.886159e-03`
  - `dsoftmax_Er/dtemperature_shape_power = 1.779140e-01`
- But compile/runtime and XLA memory did not improve:
  - `reverse_total_s = 6.901410e+02`
  - `reverse_compile_plus_execute_s = 6.679523e+02`
  - `reverse_execute_s_mean = 2.218874e+01`
  - compile-only later showed `reverse_compile_s = 6.468943e+02`
- XLA dump after the builder split still showed:
  - `Total bytes used: 2782859927 (2.59GiB)`
  - `allocation 19886: size 1.67GiB, preallocated-temp`
  - metadata still pointing at `/home/exouser/NTX/src/ntx/_solver_factorization.py`
  - remaining `vmap(jvp...)` / `transpose(jvp...)` paths through the NTX factorization and local scan regions.
- Conclusion:
  - Splitting `_build_interpolated_moment_response_local` through `_interpolated_moment_reduced_local_outputs_from_primitives` did not remove the dominant factorization graph.
  - It made warm execution worse in the 2-step lane and did not solve RAM.
  - The builder function was restored to the earlier `jax.linearize` implementation.
- Next real target:
  - Do not keep adding outer checkpoint/replay structure for this issue.
  - The dominant problem is still the NTX coefficient/factorization derivative boundary.
  - Focus on removing or replacing the remaining `jax.jvp` / `transpose(jvp)` routes that enter `_solver_factorization.py`, especially the derivative-field pullbacks:
    - `_pullback_dtransport_moments_d_er_from_scan_primitives`
    - `_pullback_dtransport_moments_d_log_nu_star_from_scan_primitives`
  - The replacement must be solver-general, JAX-compatible, and not benchmark/TOML-specific.

2026-07-03 NTX custom-JVP clarification:

- Clean/current NTX `HEAD` at commit:
  - `0e8443fa9e8efa9a8fba2a06bb89a188ab49df0a`
- The committed NTX code has a custom VJP entry point:
  - `solve_prepared_coefficient_vector_vjp`
- The committed NTX code does **not** have a custom JVP entry point:
  - `solve_prepared_coefficient_vector_jvp`
- This was verified with:
  - `git show HEAD:src/ntx/_solver_prepared.py`
  - `git show HEAD:src/ntx/solver.py`
  - `git show HEAD:src/ntx/__init__.py`
- The `custom_jvp` symbol seen locally was from an uncommitted dirty patch in `D:\PostDocsProxima\Github_5\NTX`, not from clean NTX.
- Dirty NTX files containing that experimental custom-JVP patch:
  - `src/ntx/_solver_prepared.py`
  - `src/ntx/solver.py`
  - `src/ntx/__init__.py`
  - `src/ntx/core/__init__.py`
- The Linux runtime check confirmed clean NTX behavior:

```text
>>> import ntx
>>> print(ntx.__file__)
/home/exouser/NTX/src/ntx/__init__.py
>>> print("public custom_jvp:", hasattr(ntx, "solve_prepared_coefficient_vector_jvp"))
public custom_jvp: False
>>> from ntx import _solver_prepared
>>> print("private custom_jvp:", hasattr(_solver_prepared, "solve_prepared_coefficient_vector_jvp"))
private custom_jvp: False
>>> print("private file:", _solver_prepared.__file__)
private file: /home/exouser/NTX/src/ntx/_solver_prepared.py
```

Interpretation:

- Do not assume NTX currently provides a custom JVP rule.
- `--ntx-exact-derivative-mode custom_jvp` cannot run against clean/current NTX unless the experimental NTX custom-JVP patch is intentionally applied.
- The available clean NTX option is `custom_vjp`, but that is not equivalent to a custom JVP for the derivative-field paths that use JVP/tangent information.
- The previous statement "NTX has custom_jvp" was incorrect unless explicitly referring to the dirty local NTX patch.

Next-session guidance:

- If the goal is to test whether a custom JVP reduces the reverse-AD compile/memory graph, first apply/sync the NTX custom-JVP patch deliberately, then run the `custom_jvp` benchmark.
- If the goal is to stay on clean NTX, do not test `--ntx-exact-derivative-mode custom_jvp`; focus instead on a NEOPAX-side solution that does not require a missing NTX custom-JVP symbol.
- In either case, keep the optimization solver-general and avoid benchmark/TOML-specific shortcuts.

## 2026-07-04 next plan: compact NTX reverse boundary

Current conclusion:

- The reverse AD gradients are correct in the reduced-cotangent/bicgstab lane.
- Further outer checkpointing, static branch schedules, segment-call boundaries, and host-segment orchestration did not materially reduce compile memory or execution time.
- `reverse_segment_length=1` with host-segment backward proved that the remaining heavy graph is already inside one accepted-step backward kernel.
- Therefore, the next real memory/compile-time lever is not the outer Radau checkpoint structure.
- The next target is the NTX response/factorization derivative boundary inside the accepted-step reverse kernel.

Known correct baseline command:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --ntx-exact-derivative-mode direct \
  --objective softmax_Er \
  --accepted-step-limit 16 \
  --radau-jacobian-reuse-mode legacy \
  --timing-mode jit-warm \
  --reverse-segment-length 4 \
  --reverse-stage-adjoint-solve-mode bicgstab \
  --reverse-rhs-transpose-mode explicit_ntx_interpolated \
  --reverse-step-bwd-mode reduced_cotangent
```

Expected 16-step gradients:

- `dsoftmax_Er/dn0 = -3.759631e+00`
- `dsoftmax_Er/dT0 = 3.054047e+00`
- `dsoftmax_Er/ddensity_shape_power = -8.518430e-02`
- `dsoftmax_Er/dtemperature_shape_power = 3.214064e+00`

Plan:

1. Freeze the outer reverse structure.
   - Keep the correct reduced-cotangent lane as the baseline.
   - Do not keep trying checkpoint/segment variants as the main solution.
   - Treat host-segment modes as diagnostics only, not as the final memory fix.

2. Trace the exact NTX reverse call path.
   - Start from `NEOPAX/_transport_flux_models.py`.
   - Follow the path used by Radau stage/RHS reverse adjoints to build the NTX response pullback.
   - Identify the remaining route that stages `jax.jvp`, `transpose(jvp)`, or generic VJP through NTX factorization internals.
   - The XLA evidence to eliminate is the large NTX factorization allocation with shapes like:
     - `f64[33,3,4,105,105]`
     - `f64[396,105,105]`
     - `f64[3,4,3,105,105]`
     - `f64[12,210,105]`
   - The source metadata previously pointed at `/home/exouser/NTX/src/ntx/_solver_factorization.py`.

3. Replace the generic derivative route with a compact NTX reverse primitive.
   - Use the clean NTX custom VJP path where possible, especially `solve_prepared_coefficient_vector_vjp`.
   - Do not recreate the failed `vjp_basis_derivatives` approach; that expanded the graph and caused a large compile-memory warning.
   - Do not use forward-AD tangent/JVP logic in the reverse lane.
   - The new path should accept reverse cotangents and return cotangents for the relevant NTX inputs/coefficient-response quantities.

4. Add a guarded mode only if needed.
   - A possible flag name is:

```bash
--reverse-ntx-response-pullback-mode compact_vjp
```

   - The implementation must be solver-general and not tied to the benchmark TOML.
   - If clean NTX does not expose enough API for the compact reverse pullback, add a small NTX-side helper instead of reimplementing NTX internals in NEOPAX.
   - Keep any fallback explicit and easy to remove after verification.

5. Validate in two stages.
   - First run the 2-step correctness/memory probe:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --ntx-exact-derivative-mode direct \
  --objective softmax_Er \
  --accepted-step-limit 2 \
  --radau-jacobian-reuse-mode legacy \
  --timing-mode jit-warm \
  --reverse-segment-length 2 \
  --reverse-stage-adjoint-solve-mode bicgstab \
  --reverse-rhs-transpose-mode explicit_ntx_interpolated \
  --reverse-step-bwd-mode reduced_cotangent
```

   - Expected 2-step gradients:
     - `dsoftmax_Er/dn0 = -3.578617e-01`
     - `dsoftmax_Er/dT0 = 3.010300e-01`
     - `dsoftmax_Er/ddensity_shape_power = -7.886159e-03`
     - `dsoftmax_Er/dtemperature_shape_power = 1.779140e-01`
   - Only run the 16-step benchmark after the 2-step compile graph or RAM improves.

Success criteria:

- Gradients remain unchanged.
- XLA dump no longer shows the NTX factorization derivative tensors as the dominant allocation.
- Compile RAM drops meaningfully before the GPU execution plateau.
- Compile time improves or at least stops growing while preserving correctness.

Hard constraints:

- Do not mix the forward-AD tangent lane into reverse AD.
- Do not add benchmark/TOML-specific shortcuts.
- Do not solve this by saving full lagged-response/RHS structures at every step; that defeats checkpointing.
- Do not return to static branch unrolling; it already caused host-memory blowup.
- Do not keep failed diagnostic modes as default behavior.

## 2026-07-04 implementation start: compact derivative-field pullback

Implemented first compact NTX reverse-boundary attempt:

- NEOPAX now has `ntx_exact_derivative_field_pullback_mode`.
- Default remains `generic_jvp`.
- New opt-in mode is `compact_vjp`.
- The reverse benchmark exposes it as:

```bash
--ntx-exact-derivative-field-pullback-mode compact_vjp
```

What changed:

- `NEOPAX/_transport_flux_models.py` routes derivative-field pullbacks for:
  - `_pullback_dtransport_moments_d_er_from_scan_primitives`
  - `_pullback_dtransport_moments_d_log_nu_star_from_scan_primitives`
- The compact path calls an NTX helper:
  - `solve_prepared_coefficient_vector_derivative_vjp`
- NTX now exports that helper from:
  - `src/ntx/_solver_prepared.py`
  - `src/ntx/solver.py`
  - `src/ntx/__init__.py`
  - `src/ntx/core/__init__.py`

Important caveat:

- The current compact path still recomputes the primal coefficient vector once in NEOPAX to construct the transport-moment coefficient cotangent.
- Therefore the first test is mainly a compile-memory probe.
- If XLA compile memory improves but warm execution does not, the next optimization is to remove that duplicate solve.
- If compile memory does not improve, the compact NTX helper did not cut the dominant factorization derivative graph and should be revised before 16-step testing.

Local verification:

- No-write syntax checks passed for the modified NEOPAX files.
- No-write syntax checks passed for the modified NTX files.
- `git diff --check` reported only line-ending warnings, no whitespace errors.
- Top-level NTX import could not be checked in the Windows environment because `rich` is not installed there; the Linux benchmark environment should still test the actual import path.

Next run:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --ntx-exact-derivative-mode direct \
  --ntx-exact-derivative-field-pullback-mode compact_vjp \
  --objective softmax_Er \
  --accepted-step-limit 2 \
  --radau-jacobian-reuse-mode legacy \
  --timing-mode jit-warm \
  --reverse-segment-length 2 \
  --reverse-stage-adjoint-solve-mode bicgstab \
  --reverse-rhs-transpose-mode explicit_ntx_interpolated \
  --reverse-step-bwd-mode reduced_cotangent
```

Expected 2-step gradients:

- `dsoftmax_Er/dn0 = -3.578617e-01`
- `dsoftmax_Er/dT0 = 3.010300e-01`
- `dsoftmax_Er/ddensity_shape_power = -7.886159e-03`
- `dsoftmax_Er/dtemperature_shape_power = 1.779140e-01`

What to check:

- First, confirm the gradients match the expected 2-step values.
- Second, compare compile RAM and compile-plus-execute time against the recent 2-step reduced-cotangent baseline without `compact_vjp`.
- Only move to the 16-step run if the 2-step compile-memory graph improves.

## 2026-07-05 updated conclusion: graph-boundary attempts did not reduce GPU compile memory

Recent 2-step tests preserved the expected gradients but did not reduce compile
memory or compile time enough:

- `compact_vjp` with the current NTX helper remained correct, but compile time
  and RAM stayed in the same problematic band.
- A NEOPAX-local fused derivative pullback that removed the duplicate coefficient
  solve was also correct, but it made the compiled graph larger/slower. This was
  reverted and should not be pursued further as the main optimization.
- `ntx_exact_derivative_pullback_boundary=per_energy_jit` was active in the run
  output, but it also made compile time worse and did not lower RAM. Nested
  `jax.jit(..., inline=False)` is not an effective opaque boundary here.

Latest negative `per_energy_jit` run:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --ntx-exact-derivative-mode direct \
  --ntx-exact-derivative-field-pullback-mode compact_vjp \
  --ntx-exact-derivative-pullback-boundary per_energy_jit \
  --objective softmax_Er \
  --accepted-step-limit 2 \
  --radau-jacobian-reuse-mode legacy \
  --timing-mode jit-warm \
  --reverse-segment-length 2 \
  --reverse-stage-adjoint-solve-mode bicgstab \
  --reverse-rhs-transpose-mode explicit_ntx_interpolated \
  --reverse-step-bwd-mode reduced_cotangent
```

Observed:

- `reverse_compile_plus_execute_s = 8.511400e+02`
- `reverse_execute_s_mean = 2.083876e+01`
- gradients matched the expected 2-step values:
  - `dsoftmax_Er/dn0 = -3.578617e-01`
  - `dsoftmax_Er/dT0 = 3.010300e-01`
  - `dsoftmax_Er/ddensity_shape_power = -7.886159e-03`
  - `dsoftmax_Er/dtemperature_shape_power = 1.779140e-01`
- RAM still reached the same high compile-memory band.

Conclusion:

- The main problem is not one duplicate coefficient solve.
- The main problem is that XLA sees too much of the NTX derivative-pullback
  algebra inside the reverse kernel.
- Nested JIT boundaries do not hide that algebra enough.
- Further optimization should target the algebraic contract, not more inlining,
  static branching, or nested JIT wrappers.

## Next-session plan: bespoke GPU algebra for a narrow scalar reverse pullback

Goal:

- Keep the path GPU/JIT compatible.
- Reduce the size and live state of the NTX derivative-pullback graph.
- Preserve the reverse-lane semantics and the existing gradients.
- Avoid benchmark/TOML-specific shortcuts.

Core idea:

- Stop exposing a generic "derivative of coefficient-solve VJP" object to the
  reverse kernel.
- Instead derive a NEOPAX-specific compact scalar pullback for the actual
  transport-moment contribution:

```text
transport_moment_bar -> coefficient_bar -> nu_hat_bar, epsi_hat_bar
```

Target contract:

```text
inputs:
  prepared, nu_hat, epsi_hat, drds, energy_index, transport_moments_bar

outputs:
  nu_hat_bar, epsi_hat_bar
```

For derivative-field pullbacks:

```text
inputs:
  prepared, nu_hat, epsi_hat, nu_hat_dot, epsi_hat_dot,
  drds, energy_index, transport_moments_bar

outputs:
  base_nu_hat_bar, base_epsi_hat_bar,
  tangent_nu_hat_bar, tangent_epsi_hat_bar
```

But the implementation should avoid building large generic tangent/cotangent
structures where possible. Accumulate scalar contractions directly.

Implementation steps:

1. Inspect NTX's current `solve_prepared_coefficient_vector_derivative_vjp`
   algebra and mark which arrays are only needed as intermediate stacked
   structures:
   - `f1_dot`, `f3_dot`
   - `g1_dot`, `g3_dot`
   - `lambda1_dot`, `lambda3_dot`
   - `source*_dot`
   - `adjoint_rhs*_dot`

2. Derive a smaller NEOPAX-side helper that contracts parameter-gradient terms
   in the `k` loop instead of materializing more full stacked mode arrays than
   necessary.

3. Keep the D11 floor semantics exactly. The coefficient-bar still depends on
   the actual coefficient vector because of the active floor mask.

4. First implement this as an opt-in mode, e.g.

```bash
--ntx-exact-derivative-field-pullback-mode compact_scalar_vjp
```

or, if reusing the existing option is cleaner:

```bash
--ntx-exact-derivative-pullback-algebra scalar_contract
```

5. Test only 2 accepted steps first:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --ntx-exact-derivative-mode direct \
  --ntx-exact-derivative-field-pullback-mode compact_vjp \
  --objective softmax_Er \
  --accepted-step-limit 2 \
  --radau-jacobian-reuse-mode legacy \
  --timing-mode jit-warm \
  --reverse-segment-length 2 \
  --reverse-stage-adjoint-solve-mode bicgstab \
  --reverse-rhs-transpose-mode explicit_ntx_interpolated \
  --reverse-step-bwd-mode reduced_cotangent
```

Expected 2-step gradients remain:

- `dsoftmax_Er/dn0 = -3.578617e-01`
- `dsoftmax_Er/dT0 = 3.010300e-01`
- `dsoftmax_Er/ddensity_shape_power = -7.886159e-03`
- `dsoftmax_Er/dtemperature_shape_power = 1.779140e-01`

Success criteria:

- gradients match the expected values,
- compile memory drops visibly below the current high band,
- compile-plus-execute time improves or at least does not worsen,
- no reliance on host callbacks, Python loops outside JIT, or CPU-only fallback.

Do not pursue next:

- duplicate-solve fusion by inlining more NTX algebra into NEOPAX,
- nested `per_energy_jit` as the main solution,
- static reuse/rebuild branch specialization,
- saving full lagged/RHS structures across checkpoints,
- benchmark-specific assumptions about the active TOML or D11 floor branch.

## Refined next-step plan: scalar-contract pullback

This has not yet been tried. Previous attempts simplified nearby pieces, but
they did not replace the generic derivative-of-VJP structure with a direct
scalar-contract reverse rule.

Plan:

1. Freeze the target math from the current correct path:

```text
coefficient_vector -> coefficient_bar -> NTX derivative VJP -> scalar bars
```

2. Use the current compact path as the reference implementation. Preserve:

- D11 floor behavior,
- accepted/rejected step semantics,
- reverse-lane-only implementation,
- no benchmark/TOML-specific shortcuts.

3. Inspect NTX `solve_prepared_coefficient_vector_derivative_vjp` and classify
   the large intermediates:

- `f1_dot`, `f3_dot`
- `g1_dot`, `g3_dot`
- `lambda1_dot`, `lambda3_dot`
- `source*_dot`
- `adjoint_rhs*_dot`

4. Derive direct scalar contractions for the NEOPAX transport-moment pullback.
   The goal is to avoid building a full generic derivative-pullback object when
   only scalar bars for `nu_hat` and `epsi_hat` are needed.

5. Implement as an opt-in mode, for example:

```bash
--ntx-exact-derivative-pullback-algebra scalar_contract
```

or equivalent local naming if it fits the code better.

6. First benchmark command:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --ntx-exact-derivative-mode direct \
  --ntx-exact-derivative-field-pullback-mode compact_vjp \
  --ntx-exact-derivative-pullback-algebra scalar_contract \
  --objective softmax_Er \
  --accepted-step-limit 2 \
  --radau-jacobian-reuse-mode legacy \
  --timing-mode jit-warm \
  --reverse-segment-length 2 \
  --reverse-stage-adjoint-solve-mode bicgstab \
  --reverse-rhs-transpose-mode explicit_ntx_interpolated \
  --reverse-step-bwd-mode reduced_cotangent
```

Expected 2-step gradients:

- `dsoftmax_Er/dn0 = -3.578617e-01`
- `dsoftmax_Er/dT0 = 3.010300e-01`
- `dsoftmax_Er/ddensity_shape_power = -7.886159e-03`
- `dsoftmax_Er/dtemperature_shape_power = 1.779140e-01`

Success criteria:

- gradients match the expected values,
- compile memory visibly drops,
- compile-plus-execute time does not worsen,
- no CPU fallback, host callback, or full lagged/RHS checkpointing.

## 2026-07-06 correction: scalar-contract and NTX-local variants are now tried

Important correction:

- The earlier note saying the scalar-contract pullback had not yet been tried
  is now stale.
- We have now tried several exact NTX/NEOPAX scalar-contract variants.
- They preserved gradients, and some reduced compile time, but they did not
  reduce the host RAM plateau enough to count as the memory solution.

Recent correct 2-step reference command:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --ntx-exact-derivative-mode direct \
  --ntx-exact-derivative-field-pullback-mode compact_vjp \
  --ntx-exact-derivative-pullback-algebra scalar_contract_lowdot \
  --objective softmax_Er \
  --accepted-step-limit 2 \
  --radau-jacobian-reuse-mode legacy \
  --timing-mode jit-warm \
  --reverse-segment-length 2 \
  --reverse-stage-adjoint-solve-mode bicgstab \
  --reverse-rhs-transpose-mode explicit_ntx_interpolated \
  --reverse-step-bwd-mode reduced_cotangent
```

Expected 2-step gradients:

- `dsoftmax_Er/dn0 = -3.578617e-01`
- `dsoftmax_Er/dT0 = 3.010300e-01`
- `dsoftmax_Er/ddensity_shape_power = -7.886159e-03`
- `dsoftmax_Er/dtemperature_shape_power = 1.779140e-01`

Tried and should not be repeated as the main memory solution:

- `scalar_contract`
  - Correct direction, but not enough by itself.
- `scalar_contract_lowdot`
  - Best recent compile-time behavior in the 2-step lane.
  - Still leaves RAM in the same broad high band.
- `scalar_contract_lowdot_recompute`
  - Did not reduce RAM meaningfully.
- `scalar_contract_lowdot_ntx`
  - NTX helper version with the same exact gradients.
  - A packed two-direction version did not reduce RAM.
  - A later direction-scan version also preserved gradients but still showed
    the same RAM class:
    - latest result: `reverse_total_s = 5.032938e+02`
    - `reverse_compile_plus_execute_s = 4.832360e+02`
    - `reverse_execute_s_mean = 2.005789e+01`
  - Conclusion: the two derivative directions are not the dominant live-memory
    source.
- `scalar_contract_matrix_free`
  - It reduced some compile/memory symptoms but changed gradients and depends
    on iterative tolerance behavior.
  - Do not use it as the correctness-preserving path.
- `ntx_exact_scan_batch_size=1`
  - Increased time/memory in the observed run.
  - Do not use batching as the current target.
- `recompute_vjp` prepared-solve boundary
  - Preserved gradients and looked similar; not proven to solve memory.
- `scan_rebuild_local_moment_pullback`, `scan_rebuild_anchor_pullback`,
  `reduced_cotangent_lean_replay`, `reduced_cotangent_recompute_replay`
  - Correct or near-correct local structural tests, but no decisive RAM drop.

Current implementation note:

- The working tree currently contains an experimental NTX helper path for
  `scalar_contract_lowdot_ntx`.
- It should be considered diagnostic, not the recommended production path,
  unless later XLA dump evidence proves it reduces HLO allocation size.
- The benchmark script also has `--objective all`, which computes all metric
  derivatives with `jax.jacrev`. This is useful for derivative reporting, but
  it is expected to be heavier than a single scalar objective and is not a
  memory-reduction path.

Updated conclusion:

- The dominant memory is not caused by:
  - two derivative-direction duplication,
  - coefficient-solve boundary placement,
  - local scan/replay cotangent packaging,
  - NTX scan batching,
  - or outer segment/host-call boundaries.
- The RAM plateau is more consistent with large arrays made live by the
  accepted-step reverse body and NTX factorization/pullback algebra as a whole.
- The next useful work must be evidence-driven from XLA allocation reports, not
  another nearby algebra variant.

Next-session plan:

1. First freeze the current best correctness/compile baseline.

   Recommended baseline command:

   ```bash
   python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
     --ntx-exact-derivative-mode direct \
     --ntx-exact-derivative-field-pullback-mode compact_vjp \
     --ntx-exact-derivative-pullback-algebra scalar_contract_lowdot \
     --objective softmax_Er \
     --accepted-step-limit 2 \
     --radau-jacobian-reuse-mode legacy \
     --timing-mode jit-warm \
     --reverse-segment-length 2 \
     --reverse-stage-adjoint-solve-mode bicgstab \
     --reverse-rhs-transpose-mode explicit_ntx_interpolated \
     --reverse-step-bwd-mode reduced_cotangent
   ```

2. Dump XLA for the baseline and inspect the biggest allocation names before
   changing code again.

   Do not guess from the RAM graph alone. Use the memory-usage report and grep
   the largest allocation/user chain.

3. If the largest allocations are still LU/factorization arrays from NTX:

   - Do not try matrix-free/tolerance-dependent Krylov variants.
   - Do not use scan batching yet.
   - Instead prototype an exact factorization-storage reduction:
     - avoid storing both LU and original dense blocks if both are live,
     - check whether saved lower/upper coupling blocks can be recomputed
       inside the reverse scan from smaller primitives,
     - or split the factorized forward/backward triangular contractions so
       only one mode block is live at a time.
   - Success criterion is an XLA memory report reduction, not just wall-clock
     variation.

4. If the largest allocations are in the accepted-step/rebuild graph rather
   than NTX LU arrays:

   - Return to `_execute_radau_accepted_step_next_reduced_cotangent_bwd`.
   - Target the rebuild branch body itself, not branch scheduling.
   - Do not static-unroll realized branch schedules; that previously caused
     host-memory blowup.

5. Keep the all-metric derivative feature separate.

   To get derivatives for every metric:

   ```bash
   python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
     --ntx-exact-derivative-mode direct \
     --ntx-exact-derivative-field-pullback-mode compact_vjp \
     --ntx-exact-derivative-pullback-algebra scalar_contract_lowdot \
     --objective all \
     --accepted-step-limit 2 \
     --radau-jacobian-reuse-mode legacy \
     --timing-mode jit-warm \
     --reverse-segment-length 2 \
     --reverse-stage-adjoint-solve-mode bicgstab \
     --reverse-rhs-transpose-mode explicit_ntx_interpolated \
     --reverse-step-bwd-mode reduced_cotangent
   ```

   This is for derivative coverage, not for memory reduction.

Do not pursue next:

- more `scalar_contract_lowdot_ntx` direction-packing or direction-scan
  variants,
- `scalar_contract_ntx_pullback`,
- matrix-free/tolerance-dependent algebra,
- NTX scan batching,
- static branch schedule/unroll,
- more per-energy nested JIT wrappers,
- host-segment orchestration as the final solution.

## 2026-07-07 immediate next-session update: target stage/RHS workspace, not more NTX algebra variants

Latest correction:

- The saved lower/upper NTX coupling-block recompute idea was tested and should
  be treated as rejected.
- It was expected to reduce memory only if those saved coupling blocks were the
  dominant peak allocation.
- The observed run showed more time and no memory reduction, and the XLA memory
  report points instead to a dominant `preallocated-temp` allocation.
- Therefore, the peak is more likely a live temporary workspace from the
  accepted-step reverse body / stage-RHS transpose solve, not ordinary
  persistent lower/upper arrays.

Immediate next target:

- Stop making new `scalar_contract_*` NTX derivative-algebra variants for now.
- Work inside the reverse accepted-step body, especially:
  - `_execute_radau_accepted_step_next_reduced_cotangent_bwd`
  - `_radau_solve_exact_stage_residual_transpose_iterative`
- The goal is to shorten the lifetime of stage residual / RHS transpose
  temporaries and accumulate carry/parameter bars immediately instead of
  keeping a broad stage-cotangent workspace live through the whole backward
  step.

Proposed first implementation:

- Add an opt-in reverse step mode such as
  `reduced_cotangent_stream_rhs`.
- In that mode, stream the stage/RHS transpose contribution:
  - solve/accumulate one RHS or cotangent component at a time,
  - immediately add its contribution into reduced carry/parameter bars,
  - avoid materializing or keeping the full stage residual cotangent workspace
    live across the NTX derivative pullback.
- This must stay reverse-only. Do not modify the production forward solver,
  FD lane, or forward-AD lane.

Success criteria:

- Correctness first: match the current trusted `objective all` gradients from
  `scalar_contract_lowdot` + `reduced_cotangent`.
- Memory second: require a real drop in the XLA memory report, especially the
  dominant `preallocated-temp` allocation, not only a visual RAM-graph change.
- Runtime should not regress badly; if memory is unchanged and runtime worsens,
  revert the mode like the rejected recompute-block path.

First benchmark after implementation:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --ntx-exact-derivative-mode direct \
  --ntx-exact-derivative-field-pullback-mode compact_vjp \
  --ntx-exact-derivative-pullback-algebra scalar_contract_lowdot \
  --objective all \
  --accepted-step-limit 2 \
  --radau-jacobian-reuse-mode legacy \
  --timing-mode jit-warm \
  --reverse-segment-length 2 \
  --reverse-stage-adjoint-solve-mode bicgstab \
  --reverse-rhs-transpose-mode explicit_ntx_interpolated \
  --reverse-step-bwd-mode reduced_cotangent_stream_rhs
```

Do not repeat as memory solutions:

- `scalar_contract_lowdot_recompute_blocks`
- additional `scalar_contract_lowdot_ntx` packing/scan variants
- `scalar_contract_matrix_free`
- `ntx_exact_scan_batch_size=1`
- per-energy nested JIT boundaries
- static branch schedules

## 2026-07-08 next plan: add realtime-geometry parameters to the reverse baseline

Goal:

- Extend the current reverse baseline so that, when a TOML uses realtime
  geometry, the differentiated parameter vector can include VMEC boundary
  geometry parameters in addition to the existing initial-profile parameters.
- This is not a pure-geometry AD-vs-FD test. The pure `vmec_jax` and
  `booz_xform_jax` derivative checks already exist.
- The target is the transport reverse baseline using the solver/runtime path
  that computes geometry through `vmec_jax -> booz_xform_jax`, analogous in
  spirit to the existing lagged NTX runtime path.

Current relevant code paths:

- `NEOPAX._orchestrator.build_runtime_context(config)` already detects:

  ```toml
  [geometry]
  backend = "vmec_jax_booz_xform_jax"
  vmec_param_family = "RBC"
  vmec_param_m = 1
  vmec_param_n = 0
  vmec_param_delta = 0.0
  ```

- For that backend it routes to:
  - `build_geometry_autodiff_context(...)`
  - `build_runtime_context_for_geometry_param(...)`
- The realtime VMEC benchmark config is:
  - `examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml`
- The current reverse benchmark still hardcodes only profile parameters:

  ```python
  PARAMETER_ORDER = ("n0", "T0", "density_shape_power", "temperature_shape_power")
  ```

- Therefore, even with realtime geometry in the TOML, the current reverse
  benchmark differentiates profiles on a fixed already-built geometry.

Implementation plan:

1. Keep the profile-only reverse baseline unchanged.

   - Do not change the meaning of the existing `PARAMETER_ORDER`.
   - Do not change the non-realtime geometry path.
   - Do not touch the production forward solver, forward-AD lane, or FD lane.

2. Add an opt-in reverse benchmark mode for realtime geometry parameters.

   Suggested CLI shape:

   ```bash
   --reverse-parameter-mode profiles_plus_realtime_geometry
   --reverse-geometry-parameter RBC:1:0
   ```

   Initial implementation can support one geometry parameter only.

3. Add a small parameter-spec layer in
   `examples/benchmarks/benchmark_transport_reverse_ad_only.py`.

   - Profile specs map to the existing four profile values.
   - Geometry specs map to one scalar VMEC boundary perturbation delta.
   - Printed/report parameter order should become, for example:

     ```text
     ["n0", "T0", "density_shape_power", "temperature_shape_power", "vmec:RBC:1:0"]
     ```

4. For realtime-geometry mode, build the geometry AD context once outside the
   differentiated objective.

   - Read the realtime config's `[geometry]` section.
   - Use `build_geometry_autodiff_context(...)` with the requested
     `family/m/n`.
   - Validate that `geometry.backend` is one of:
     - `vmec_jax_booz_xform_jax`
     - `vmec_runtime`
     - `vmec_realtime`
   - Refuse the mode for ordinary frozen `vmec_booz` geometry configs.

5. Inside the differentiated objective, split the parameter vector.

   - Profile values continue to define the initial density/temperature
     profiles, as today.
   - The geometry value becomes `param_delta` for
     `build_runtime_context_for_geometry_param(...)`.
   - Then rebuild the profile initial state using the geometry from the
     newly-built runtime.

   Desired flow:

   ```text
   p -> profile_values, geometry_delta
     -> runtime, baseline_state = build_runtime_context_for_geometry_param(...)
     -> state0 = current profile-state builder(profile_values, runtime.geometry)
     -> prepare reverse/static solver pieces for that runtime
     -> run reverse objective
   ```

6. Start with a correctness-first path.

   - The current optimized reverse custom-VJP setup precomputes
     `reverse_setup.execution_context`.
   - That precomputed setup assumes geometry/runtime are static and therefore
     is not automatically valid for differentiating geometry.
   - First implementation should prioritize a mathematically honest small test,
     even if slower, before trying to fold geometry into the optimized
     reduced-cotangent reverse path.

7. First test command after implementation:

   ```bash
   python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
     --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
     --reverse-parameter-mode profiles_plus_realtime_geometry \
     --reverse-geometry-parameter RBC:1:0 \
     --ntx-exact-derivative-mode direct \
     --ntx-exact-derivative-field-pullback-mode compact_vjp \
     --ntx-exact-derivative-pullback-algebra scalar_contract_lowdot \
     --objective all \
     --accepted-step-limit 2 \
     --radau-jacobian-reuse-mode legacy \
     --timing-mode jit-warm \
     --reverse-segment-length 2 \
     --reverse-stage-adjoint-solve-mode bicgstab \
     --reverse-rhs-transpose-mode explicit_ntx_interpolated \
     --reverse-step-bwd-mode reduced_cotangent
   ```

Success criteria:

- The report prints both profile and geometry parameters in `parameter_order`.
- Profile derivatives remain consistent with the existing profile-only reverse
  baseline when geometry delta is at zero.
- The new geometry column is nonzero for objectives that depend on geometry.
- The same config can still run without the new mode and reproduce the current
  profile-only behavior.

Do not do in the first pass:

- Do not add multiple geometry parameters at once.
- Do not try to optimize reverse memory in the same patch.
- Do not reuse pure-geometry benchmark code as the final solver-level test.
- Do not change `vmec_jax` or `booz_xform_jax`.

Forward-AD companion plan:

- The matching forward-AD geometry plan is recorded in `ad_forward_lane.md`.
- Forward AD should use the same pure-geometry-tested helpers, especially
  `build_geometry_autodiff_context(...)` and
  `build_runtime_context_for_geometry_param(...)`.
- Forward AD should remain a separate lane: profile parameters keep the current
  `--parameter n0|T0|...` behavior, while the first geometry extension should be
  an opt-in single-parameter form such as `--parameter vmec:RBC:1:0`.

## 2026-07-08 Lane State Handoff

This section records the current state after the all-objective reverse-AD and
realtime-geometry discussions.  Use it as the first checkpoint in the next
session.

### Forward Solver Lane

- Baseline production forward solver should remain independent from AD
  experiments.
- Frozen-geometry NTX forward solver TOMLs are still the reference for
  profile-only transport behavior.
- Realtime-geometry forward solver TOML exists for the VMEC/JAX +
  Booz-xform/JAX path; VMEC/Booz should be built once for the runtime geometry,
  while NTX is still called repeatedly during transport flux evaluations.
- No reverse-AD multi-RHS changes should be allowed to change forward-solver
  behavior.

### Forward-AD Lane

- `benchmark_transport_forward_ad_only.py` remains the reference forward-AD
  lane for profile derivatives.
- Known profile-only 16-step softmax reference values used for reverse
  comparison:

  ```text
  dsoftmax_Er/dn0                     = -3.759631e+00
  dsoftmax_Er/dT0                     =  3.054047e+00
  dsoftmax_Er/ddensity_shape_power    = -8.518430e-02
  dsoftmax_Er/dtemperature_shape_power=  3.214063e+00
  ```

- The forward-AD geometry extension should stay opt-in and use the already
  tested geometry helpers:
  `build_geometry_autodiff_context(...)` and
  `build_runtime_context_for_geometry_param(...)`.
- Do not import reverse-AD helpers into the forward-AD benchmark.

### Reverse-AD Profile Lane

- Correct profile-only reverse baseline for all objectives at 2 accepted steps
  matches the forward-AD values, including the tiny `smooth_root_proxy`
  derivatives when printed at high precision.
- The correct 2-step all-objective reverse command is:

  ```bash
  python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
    --ntx-exact-derivative-mode direct \
    --ntx-exact-derivative-field-pullback-mode compact_vjp \
    --ntx-exact-derivative-pullback-algebra scalar_contract_lowdot_ntx \
    --objective all \
    --accepted-step-limit 2 \
    --radau-jacobian-reuse-mode legacy \
    --timing-mode jit-warm \
    --reverse-segment-length 2 \
    --reverse-stage-adjoint-solve-mode bicgstab \
    --reverse-rhs-transpose-mode explicit_ntx_interpolated \
    --reverse-step-bwd-mode reduced_cotangent
  ```

- The 16-step all-objective baseline without the new multi-RHS mode was
  expensive:

  ```text
  reverse_total_s                 = 2.305885e+03
  reverse_compile_plus_execute_s  = 1.413693e+03
  reverse_execute_s_mean          = 8.921922e+02
  ```

- This 16-step all-objective run produced the expected full gradient matrix and
  should be treated as the current correctness reference for profile-only
  reverse AD.

### Reverse-AD All-Objective Multi-RHS Experiment

- A new opt-in mode was added:

  ```text
  --reverse-all-objectives-mode multi_rhs_reduced
  ```

- The name `reduced` means reduced cotangent contract, not partial
  implementation.  The reduced contract carries only:

  ```text
  y
  lagged_response_cache
  lagged_reference_y
  ```

- The intended call chain for the requested mode is:

  ```text
  _reverse_all_objectives_multi_rhs_reduced_for_parameter_vector
    -> _radau_segment_reduced_cotangent_bwd_batched_call
    -> _execute_radau_accepted_step_next_reduced_cotangent_batched_bwd
    -> _radau_solve_exact_stage_residual_transpose_batched
    -> _radau_solve_exact_stage_residual_transpose_batched_iterative
  ```

- This is intended to batch the objective RHS axis inside the reduced
  stage-adjoint solve rather than only wrapping scalar objective pullbacks.
- The batched BiCGSTAB keeps independent row scalars/convergence state for
  each objective RHS while sharing the loop body and batched transpose matvec.
- The 2-step test with this mode completed and matched the current correct
  all-objective gradients:

  ```text
  reverse_all_objectives_mode      = multi_rhs_reduced
  reverse_total_s                 = 4.780329e+02
  reverse_compile_plus_execute_s  = 4.579891e+02
  reverse_execute_s_mean          = 2.004384e+01
  ```

- The 2-step timing is not decisive for performance because compile and launch
  overhead dominate.  The meaningful test is 16 accepted steps.

Next command to run:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --ntx-exact-derivative-mode direct \
  --ntx-exact-derivative-field-pullback-mode compact_vjp \
  --ntx-exact-derivative-pullback-algebra scalar_contract_lowdot_ntx \
  --objective all \
  --reverse-all-objectives-mode multi_rhs_reduced \
  --accepted-step-limit 16 \
  --radau-jacobian-reuse-mode legacy \
  --timing-mode jit-warm \
  --reverse-segment-length 4 \
  --reverse-stage-adjoint-solve-mode bicgstab \
  --reverse-rhs-transpose-mode explicit_ntx_interpolated \
  --reverse-step-bwd-mode reduced_cotangent
```

Success criteria for the 16-step multi-RHS test:

- Gradients match the previous 16-step all-objective reverse reference.
- `reverse_execute_s_mean` is materially below the previous
  `8.921922e+02 s` if the multi-RHS path is reducing repeated reverse work.
- If execution time remains close to the old value, the remaining duplicated
  cost is likely inside the RHS transpose/pullback graph or XLA lowering of the
  batched objective axis, not in the benchmark-level objective loop.

### Reverse-AD Geometry Lane

- The correctness-first reverse geometry path is separate from the optimized
  profile-only realized-schedule custom VJP path.
- The geometry path should use the pure-geometry-tested VMEC/JAX and
  Booz-xform/JAX helpers.
- Geometry differentiation should initially support a single opt-in VMEC
  boundary parameter, for example:

  ```text
  --reverse-parameter-mode profiles_plus_realtime_geometry
  --reverse-geometry-parameter RBC:1:0
  ```

- Do not mix geometry differentiation work with reverse profile memory
  optimization in the same patch unless explicitly requested.

### Guardrails For Next Session

- Do not change NTX unless explicitly requested.
- Do not change forward solver or forward-AD behavior while testing reverse
  all-objective performance.
- Do not treat `vmap` over scalar pullbacks as a completed multi-RHS adjoint.
- If a new performance path does not improve a small test, do not assume the
  16-step behavior without checking whether the small test is dominated by
  compile/launch overhead.
- If XLA dumps are needed, dump only memory reports or clean `/tmp` afterward;
  previous full dumps risked filling disk.

## 2026-07-09: Profile Reverse AD, Forward AD, And FD Trace State

Current profile-only reverse AD baseline command for all objectives at 16
accepted steps:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --ntx-exact-derivative-mode direct \
  --ntx-exact-derivative-field-pullback-mode compact_vjp \
  --ntx-exact-derivative-pullback-algebra scalar_contract_lowdot_ntx \
  --objective all \
  --accepted-step-limit 16 \
  --radau-jacobian-reuse-mode legacy \
  --timing-mode jit-warm \
  --reverse-segment-length 4 \
  --reverse-stage-adjoint-solve-mode bicgstab \
  --reverse-rhs-transpose-mode explicit_ntx_interpolated \
  --reverse-step-bwd-mode reduced_cotangent
```

Latest reverse run used `reverse_all_objectives_mode=jacrev` and reported:

| Timing | Value |
|---|---:|
| `reverse_total_s` | `2.289904e+03` |
| `reverse_compile_plus_execute_s` | `1.395459e+03` |
| `reverse_execute_s_mean` | `8.944451e+02` |

Latest reverse 16-step gradients:

| Objective | `d/dn0` | `d/dT0` | `d/ddensity_shape_power` | `d/dtemperature_shape_power` |
|---|---:|---:|---:|---:|
| `softmax_Er` | `-3.7596310226172802e+00` | `3.0540468749022844e+00` | `-8.5184303176153636e-02` | `3.2140636924368615e+00` |
| `smooth_root_proxy` | `4.8792550932823947e-04` | `-1.7940595936161225e-04` | `-8.3536017747148126e-06` | `1.7962929118689044e-02` |
| `Er2_volume_average` | `-4.3482590469415534e+00` | `2.4602122740846760e+01` | `1.3374406622306281e+00` | `-3.2083840582997361e+01` |
| `Er_volume_average` | `-1.7368746602804679e+00` | `8.2053365692759939e-01` | `-4.3801264628593156e-02` | `-4.1902415244173707e-01` |
| `electron_temperature_volume_average_keV` | `3.1848844709979965e-03` | `3.5041382074642197e-01` | `-1.8160988923259945e-04` | `1.5035492692713084e+00` |
| `total_pressure_volume_average` | `7.9065130127052745e+00` | `1.8293319419864902e+00` | `2.3977706000108417e-01` | `7.6009325304638091e+00` |
| `alpha_power_volume_average_mw_m3` | `2.7390490116182176e-01` | `8.1429102705099329e-02` | `2.3111658399500357e-03` | `2.7792821656363359e-01` |

Known-good primal check:

```bash
python ./examples/benchmarks/compare_forward_reverse_primal_rollout.py \
  --ntx-exact-derivative-mode direct \
  --ntx-exact-derivative-field-pullback-mode compact_vjp \
  --ntx-exact-derivative-pullback-algebra scalar_contract_lowdot_ntx \
  --accepted-step-limit 16 \
  --radau-jacobian-reuse-mode legacy \
  --reverse-segment-length 4 \
  --reverse-stage-adjoint-solve-mode bicgstab \
  --reverse-rhs-transpose-mode explicit_ntx_interpolated \
  --reverse-step-bwd-mode reduced_cotangent
```

This reported exact equality between forward realized-schedule primal,
reverse-VJP primal, and reverse plain primal for all objectives. Therefore the
current forward/reverse discrepancy is not a primal rollout mismatch.

FD diagnostic state:

- `_truncate_rollout_trace_by_accepted_steps()` was fixed so the frozen trace
  stops at the Nth accepted attempt instead of keeping later rejected attempts.
- The FD diagnostic was then found to use the plain forward lane for frozen
  replay while the AD comparisons use the AD realized-schedule lane.
- `benchmark_transport_adaptive_ad_vs_frozen_fd.py` now has
  `--fd-replay-lane {ad,plain}`, defaulting to `ad`.
- `benchmark_transport_autodiff_lagged_ntx.py` now lets frozen-trace objective
  replay opt into the AD lane through `use_ad_lane=True`.
- Reproduce old behavior with `--fd-replay-lane plain` if needed.

Next FD command to run:

```bash
python ./examples/benchmarks/benchmark_transport_adaptive_ad_vs_frozen_fd.py \
  --ntx-exact-derivative-mode direct \
  --parameter n0 \
  --accepted-step-limit 16 \
  --run-mode fd \
  --fd-rel-step 3e-8 \
  --fd-abs-step 1e-10
```

Expected next interpretation:

- The JSON/output should show `fd_replay_lane=ad`.
- If the frozen replay baseline objectives move to the forward/reverse AD
  primal values, the previous FD mismatch was a lane mismatch.
- If they do not move, the remaining difference is a real frozen-trace FD
  sensitivity issue rather than a plain-vs-AD replay issue.

## 2026-07-10: Realtime Geometry To AD State

Current issue:

- Frozen transport benchmark uses the VMEC WOUT file path for NTX surfaces:
  `ntx_exact_surface_backend = "vmec"` in
  `examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_benchmark.toml`.
- That frozen path calls
  `ntx.surface_from_vmec_jax_vmec_wout_file(vmec_file, s=rho**2)`.
- Realtime geometry path currently builds a vmec_jax `WoutData` object through
  `vmec_jax.wout.wout_minimal_from_fixed_boundary(...)`, then calls
  `ntx.surface_from_vmec_jax_vmec_wout(wout, s=rho**2)`.
- The public vmec_jax WOUT writer path is not a separate complete VMEC2000
  writer: `write_wout_from_fixed_boundary_run(...)` routes through
  `wout_from_fixed_boundary_run(...)`, which routes through
  `wout_minimal_from_fixed_boundary(...)`.
- Therefore, writing the realtime vmec_jax WOUT to NetCDF and reading it back
  through NTX should test whether the mismatch is already present in the
  vmec_jax reconstructed WOUT data, rather than being introduced by NEOPAX's
  in-memory injection.

Important guardrails:

- Do not change NTX for this diagnostic.
- Do not change reverse AD or forward AD profile baselines while debugging
  realtime geometry injection.
- The frozen VMEC-WOUT backend remains the historical profile-only transport
  baseline until a deliberate backend migration is approved.

New diagnostic added:

```bash
python ./examples/benchmarks/compare_realtime_vmec_initial_ambipolarity.py \
  --json-output outputs/realtime_initial_ambipolarity_roundtrip_wout_compare.json \
  --residual-curve-indices 0,29 \
  --residual-curve-er-min -40 \
  --residual-curve-er-max 25 \
  --residual-curve-count 81 \
  --raw-vmec-surface-diagnostics \
  --roundtrip-realtime-wout-diagnostics \
  --roundtrip-realtime-wout-output outputs/realtime_vmec_jax_minimal_wout_roundtrip.nc
```

What to inspect:

- In the printed output and JSON, compare
  `realtime_direct_vs_roundtrip` against `frozen_vs_realtime_roundtrip`.
- If `realtime_direct_vs_roundtrip` is tiny while
  `frozen_vs_realtime_roundtrip` remains at the same level as the raw
  frozen-vs-realtime mismatch, then NEOPAX is not losing information during
  in-memory injection; the mismatch is already in the vmec_jax reconstructed
  WOUT fields.
- If `realtime_direct_vs_roundtrip` is not tiny, then the issue is in WOUT
  writing/reading conventions and should be debugged before transport tests.

Known previous evidence before this roundtrip check:

- Initial ambipolar Er mismatch was already visible before transport time
  integration.
- Pressure and density matched to near roundoff.
- Raw NTX VMEC surface comparison showed the largest mismatch in
  `b_sup_theta_cos` / contravariant magnetic-field coefficients.
- This points to VMEC-WOUT Nyquist magnetic-field reconstruction, not to
  profile values, transport stepping, or reverse AD.

## 2026-07-10: Forward AD `exact` Mode Primal Correction

Observed run:

```bash
python ./examples/benchmarks/benchmark_transport_forward_ad_only.py \
  --ntx-exact-derivative-mode direct \
  --parameter n0 \
  --accepted-step-limit 16 \
  --radau-jacobian-reuse-mode legacy \
  --forward-ad-fusion-mode exact
```

Before the latest patch, this still printed the wrong exact-mode primal
objectives, for example:

- `softmax_Er: value=2.0695082896894014e+01`
- `smooth_root_proxy: value=9.0006241686889227e-03`
- `Er2_volume_average: value=2.4368404307592147e+02`
- `Er_volume_average: value=-3.4299625917040317e+00`

These are the same wrong-primal signature seen earlier for the `exact` path.
The expected reverse/FD realized-schedule primal for this 16-step case is
closer to:

- `softmax_Er ~= 2.069247668676e+01`
- `smooth_root_proxy ~= 8.907308004081e-03`
- `Er2_volume_average ~= 2.437187070410e+02`
- `Er_volume_average ~= -3.430424752652e+00`

Cause:

- `--forward-ad-fusion-mode exact` does **not** dispatch to the standard
  replay custom-JVP path.
- It dispatches through
  `_radau_adaptive_final_y_realized_schedule_exact_jvp(...)`.
- Its custom-JVP rule called
  `_radau_adaptive_final_y_realized_schedule_fused_jvp(...,
  raw_attempt_jvp=True)` and returned that fused-loop primal output.
- That fused-loop primal was not the trusted realized-schedule primal.

Patch applied:

- In `NEOPAX/_transport_solvers.py`,
  `_radau_adaptive_final_y_realized_schedule_exact_jvp_rule(...)` now computes
  `primal_rollout = _radau_adaptive_schedule_rollout(...)` and returns
  `primal_rollout.final_carry.y` as `primal_out`.
- The exact tangent is still produced by
  `_radau_adaptive_final_y_realized_schedule_fused_jvp(...,
  raw_attempt_jvp=True)`.
- This is deliberately narrow: it fixes the `exact` mode printed objective
  values first, without changing reverse AD or the standard forward-AD replay
  lane.

Next command:

```bash
python ./examples/benchmarks/benchmark_transport_forward_ad_only.py \
  --ntx-exact-derivative-mode direct \
  --parameter n0 \
  --accepted-step-limit 16 \
  --radau-jacobian-reuse-mode legacy \
  --forward-ad-fusion-mode exact
```

Expected interpretation:

- If objective values move to the reverse/FD realized-schedule primal, the
  `exact` mode primal-return bug is fixed.
- If tangents still disagree, the remaining issue is isolated to the
  raw-attempt exact tangent propagation, not to the primal rollout.
- Do not change reverse AD for this; reverse already matched FD much better for
  this 16-step profile case.

## 2026-07-14: Realtime Geometry Reverse Coupling Boundary

Current separation rule:

- Frozen-geometry/profile reverse benchmarks must keep using the existing
  reduced reverse accepted-step path.
- Realtime VMEC geometry transport benchmarks must not pass traced geometry
  through the profile path's static `physics_context` custom-VJP argument.

Reason:

- The profile reverse primitive treats the Radau `physics_context` as
  nondifferentiable/static.
- That is correct for frozen geometry, where NTX/geometry support is fixed.
- It is not correct for realtime geometry, where VMEC-derived field tables and
  NTX support depend on the differentiated VMEC boundary parameter.

Implementation checkpoint:

- `benchmark_transport_reverse_ad_only.py` keeps
  `--reverse-parameter-mode profiles` on the old profile reverse path.
- A temporary JVP-column bridge was tried for
  `--reverse-parameter-mode profiles_plus_realtime_geometry`, but this is not a
  valid path because `vmec_jax.core.implicit.solve_implicit(...)` is
  `custom_vjp`-only and rejects forward-mode AD.
- The realtime-geometry branch now fails explicitly with
  `realtime_geometry_gradient_path = "pending_reverse_geometry_payload"` rather
  than accidentally trying either:
  - profile replay VJP with traced static `physics_context`, or
  - forward-mode JVP through VMEC implicit solve.

Final optimized target:

- Add a separate geometry-aware accepted-step/RHS transpose that has an explicit
  differentiable geometry or NTX-support payload.
- That final path should return both the transport carry cotangent and the
  geometry/NTX-support cotangent, then pull the latter back through
  `VMEC params -> VMEC state -> VMEC field tables -> NTX support`.
- Do not retrofit this into the frozen/profile custom VJP; keep the profile
  benchmark protected from realtime-geometry experiments.

### Next-session start point

Do not restart by trying a JVP bridge.

- The rejected command was the realtime transport reverse benchmark:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --reverse-parameter-mode profiles_plus_realtime_geometry \
  --reverse-geometry-parameter RBC:1:0 \
  --ntx-exact-derivative-mode direct \
  --ntx-exact-derivative-field-pullback-mode generic_jvp \
  --objective all \
  --accepted-step-limit 2 \
  --radau-jacobian-reuse-mode legacy \
  --timing-mode jit-warm \
  --reverse-segment-length 1 \
  --reverse-stage-adjoint-solve-mode bicgstab \
  --reverse-rhs-transpose-mode explicit_ntx_interpolated \
  --reverse-step-bwd-mode reduced_cotangent
```

- The JVP-column fallback failed with:
  `TypeError: can't apply forward-mode autodiff (jvp) to a custom_vjp function`.
- The failing custom-VJP function is
  `vmec_jax.core.implicit.solve_implicit(...)`, which is reverse-mode only.
- Therefore, the correct next implementation is reverse-mode only:
  reuse the existing Radau/profile reverse math, but add a new realtime-geometry
  payload channel through the RHS/accepted-step transpose.

Concrete implementation target:

1. Keep `--reverse-parameter-mode profiles` on the existing static
   `physics_context` reduced reverse path.
2. For `profiles_plus_realtime_geometry`, introduce a separate opt-in path whose
   accepted-step/RHS transpose has an explicit differentiable geometry payload.
   This must follow the same object-boundary philosophy as the transport reverse
   path: build Python/static geometry objects outside the differentiated graph,
   and differentiate only the JAX array payload held by those objects.
3. The new backward should return both:
   - `carry_bar`, for the existing profile/transport-state part;
   - `geometry_payload_bar`, for VMEC/NTX support.
4. The payload should be based on the same realtime geometry objects and helper
   boundaries used by the geometry-objective benchmarks, not on rebuilding
   support objects inside the transport derivative graph.
5. Pull `geometry_payload_bar` back through the already-tested realtime geometry
   array construction path:
   `geometry_delta -> solve_implicit custom_vjp -> VMEC state -> VMEC field-table arrays -> NTX surface/support arrays`.
6. Only after this passes the 2-accepted-step single-geometry-parameter test,
   extend to multiple geometry harmonics and then optimize the multi-RHS/objective
   handling.

Payload design constraint:

- Do not construct `Geometry`, `VmecSurface`, `NTXExactLijRuntimeSupport`, or
  other Python/dataclass support containers dynamically inside the
  differentiated transport objective.
- Instead, prebuild the static scaffolding once from the realtime geometry setup
  and pass/update a pytree of differentiable arrays through the RHS/flux
  evaluation.
- This is the same pattern as the profile transport reverse lane: the object
  structure is static; only numerical arrays are differentiated.

Implementation update:

- `NTXExactLijRuntimeSupport` is now registered as a JAX pytree with `grid`
  static and the scan/prepared arrays as differentiable data fields.
- `NTXExactLijRuntimeTransportModel.with_support_payload(...)` is the explicit
  model-level injection point for a realtime geometry NTX support payload.
- `benchmark_transport_reverse_ad_only.py` now has an opt-in
  `--realtime-geometry-gradient-path payload_boundary_probe` mode. This mode
  does **not** compute geometry gradients; it checks the boundary we need for the
  real implementation:
  - find the preloaded realtime NTX support payload,
  - swap that payload back into the already-built flux model,
  - rerun the accepted-step prefix,
  - report baseline-vs-swapped objective deltas and a support-pytree summary.
- The default realtime geometry path remains
  `--realtime-geometry-gradient-path reverse_payload` and still fails
  explicitly until the Radau accepted-step transpose returns a geometry payload
  cotangent.

Checkpoint command:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --reverse-parameter-mode profiles_plus_realtime_geometry \
  --reverse-geometry-parameter RBC:1:0 \
  --realtime-geometry-gradient-path payload_boundary_probe \
  --ntx-exact-derivative-mode direct \
  --ntx-exact-derivative-field-pullback-mode generic_jvp \
  --objective all \
  --accepted-step-limit 2 \
  --radau-jacobian-reuse-mode legacy \
  --timing-mode jit-warm \
  --reverse-segment-length 1 \
  --reverse-stage-adjoint-solve-mode bicgstab \
  --reverse-rhs-transpose-mode explicit_ntx_interpolated \
  --reverse-step-bwd-mode reduced_cotangent
```

Expected checkpoint output:

- `realtime_geometry_gradient_path=payload_boundary_probe`
- nonzero `payload_array_leaves`
- `max_objective_abs_delta` and `max_objective_rel_delta` close to roundoff
  when explicit payload injection is equivalent to the originally preloaded
  realtime support.
- JSON written to
  `outputs/autodiff_transport_lagged_ntx/reverse_ad/transport_reverse_ad_only_realtime_geometry_payload_boundary.json`.

Observed checkpoint result:

- The payload boundary probe found `payload_array_leaves=134` and
  `payload_total_array_bytes=20492928`.
- For finite objectives, baseline and swapped-payload values matched exactly:
  - `softmax_Er = 1.6049719305982389e+01`
  - `smooth_root_proxy = 3.9999999999999997e-01`
- The remaining objectives were `nan` for both baseline and swapped payload:
  `Er2_volume_average`, `Er_volume_average`,
  `electron_temperature_volume_average_keV`, `total_pressure_volume_average`,
  and `alpha_power_volume_average_mw_m3`.
- Interpretation: the support payload injection boundary is behaving for the
  finite outputs. The `nan` max-delta was a probe-reporting issue caused by
  finite and nonfinite objectives being reduced together, not evidence of a
  payload mismatch.
- The probe now reports finite-only maxima as
  `max_finite_objective_abs_delta` and `max_finite_objective_rel_delta`, plus
  an explicit `nonfinite_objectives` list.

Next implementation target:

- Add a realtime-only Radau reverse primitive whose differentiable input is
  `(carry0, geometry_support_payload)` instead of just `carry0`.
- Its forward pass should use the same realized accepted schedule machinery.
- Its backward pass should return both:
  - `carry0_bar`, following the existing reduced profile reverse path;
  - `support_payload_bar`, accumulated from flux/RHS transposes.
- After that, use a single VJP to pull `support_payload_bar` through
  `vmec_jax solve_implicit -> VMEC field tables -> NTX prepared/support arrays`
  back to VMEC boundary harmonics.
- Keep this implementation behind
  `--reverse-parameter-mode profiles_plus_realtime_geometry`; do not alter the
  frozen/profile-only reverse benchmark path.

Support-pullback plumbing checkpoint:

- Added local support-payload pullback hooks through:
  - `NTXExactLijRuntimeTransportModel`
  - `CombinedTransportFluxModel`
  - `ComposedEquationSystem`
  - Radau flat-RHS factory/context setup
- These hooks are optional and unused by the frozen/profile-only reverse path.
- Added benchmark mode
  `--realtime-geometry-gradient-path support_pullback_probe`.
- This probe does not compute transport-objective gradients yet. It checks the
  local cotangent path needed by the full realtime reverse primitive:
  `rhs_bar -> flux_bar -> NTX support_bar`.

Support-pullback checkpoint command:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --reverse-parameter-mode profiles_plus_realtime_geometry \
  --reverse-geometry-parameter RBC:1:0 \
  --realtime-geometry-gradient-path support_pullback_probe \
  --ntx-exact-derivative-mode direct \
  --ntx-exact-derivative-field-pullback-mode generic_jvp \
  --objective all \
  --accepted-step-limit 2 \
  --radau-jacobian-reuse-mode legacy \
  --timing-mode jit-warm \
  --reverse-segment-length 1 \
  --reverse-stage-adjoint-solve-mode bicgstab \
  --reverse-rhs-transpose-mode explicit_ntx_interpolated \
  --reverse-step-bwd-mode reduced_cotangent
```

Expected result:

- `realtime_geometry_gradient_path=support_pullback_probe`
- finite `support_bar_l2`
- `support_bar_all_finite=True`
- JSON written to
  `outputs/autodiff_transport_lagged_ntx/reverse_ad/transport_reverse_ad_only_realtime_geometry_support_pullback.json`.

### 2026-07-15: Realtime Geometry Reverse Handoff

Current status:

- The payload boundary probe passed:
  - `payload_array_leaves=134`
  - `payload_total_array_bytes=20492928`
  - `max_finite_objective_abs_delta=0.000000e+00`
  - `max_finite_objective_rel_delta=0.000000e+00`
- Interpretation: explicitly extracting and reinserting the realtime NTX support
  payload is forward-equivalent for finite objectives. This validates the
  static-object / differentiable-array-payload boundary.
- The short two-step prefix still has nonfinite objectives for:
  `Er2_volume_average`, `Er_volume_average`,
  `electron_temperature_volume_average_keV`, `total_pressure_volume_average`,
  and `alpha_power_volume_average_mw_m3`. These are identical on baseline and
  swapped payload, so they are not caused by the payload boundary.

What was added:

- `NTXExactLijRuntimeSupport` is a JAX pytree with `grid` static.
- `NTXExactLijRuntimeTransportModel.with_support_payload(...)` is the explicit
  payload injection hook.
- Benchmark-local payload replacement helpers in
  `benchmark_transport_reverse_ad_only.py`.
- Optional support-payload pullback plumbing through:
  - `NTXExactLijRuntimeTransportModel`
  - `CombinedTransportFluxModel`
  - `ComposedEquationSystem`
  - Radau flat-RHS factory / physics context
- `--realtime-geometry-gradient-path payload_boundary_probe`
- `--realtime-geometry-gradient-path support_pullback_probe`

Important negative result:

- A generic VJP through `build_lagged_response(...)` with respect to the full
  NTX support payload OOMed:
  - XLA estimated roughly `115 GiB` minimum rematerialized memory.
  - The stack entered
    `pullback_build_lagged_response_support_payload -> NTX build_lagged_response`.
- Do **not** use the broad generic support VJP through lagged-response
  construction as the final path.
- The probe now skips that build path by default. It is only run with:
  `--support-pullback-probe-include-build`.

Correct next-step framing:

- Do not rewrite Radau, NTX, or VMEC from scratch.
- Do not replace the known-good frozen/profile reverse lane.
- Do not try a forward-mode/JVP bridge through VMEC implicit solve.
- Do not hide realtime geometry inside static `physics_context` and expect a
  profile-only VJP to see it.

What we need to connect:

```text
existing profile reverse AD machinery
  -> transport/RHS/NTX cotangents
  -> missing realtime NTX support payload cotangent
  -> existing geometry AD machinery
  -> VMEC harmonic gradients
```

The two working endpoints already exist:

- Profile reverse AD already computes profile gradients for frozen/static
  geometry.
- Geometry AD already maps geometry-objective cotangents through
  `vmec_jax solve_implicit -> VMEC/Boozer/NTX arrays -> RBC/ZBS`.

The missing middle link is:

```text
transport objective cotangent -> cotangent wrt realtime NTX support payload
```

Recommended next implementation:

1. Keep `--reverse-parameter-mode profiles` exactly on the current
   frozen/profile reverse path.
2. For `--reverse-parameter-mode profiles_plus_realtime_geometry`, create a
   sibling realtime-only reduced reverse wrapper. It should reuse the existing
   accepted-step / segmented replay reverse machinery but expose an additional
   differentiable payload argument.
3. The realtime wrapper should conceptually differentiate:

```text
(carry0, support_payload) -> final_y
```

   not just:

```text
carry0 -> final_y
```

4. Its backward should return:

```text
(carry0_bar, support_payload_bar)
```

5. Use the existing profile parameter mapping for `carry0_bar`.
6. Pull `support_payload_bar` back through the existing realtime geometry AD
   construction path to obtain VMEC harmonic gradients.
7. Only after this works for `accepted_step_limit=2` and `RBC:1:0`, compare
   against finite differences using the same accepted/replayed schedule
   philosophy as the frozen profile benchmark.

Most recent useful command:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --reverse-parameter-mode profiles_plus_realtime_geometry \
  --reverse-geometry-parameter RBC:1:0 \
  --realtime-geometry-gradient-path payload_boundary_probe \
  --ntx-exact-derivative-mode direct \
  --ntx-exact-derivative-field-pullback-mode generic_jvp \
  --objective all \
  --accepted-step-limit 2 \
  --radau-jacobian-reuse-mode legacy \
  --timing-mode jit-warm \
  --reverse-segment-length 1 \
  --reverse-stage-adjoint-solve-mode bicgstab \
  --reverse-rhs-transpose-mode explicit_ntx_interpolated \
  --reverse-step-bwd-mode reduced_cotangent
```

Result:

```text
max_finite_objective_abs_delta=0.000000e+00
max_finite_objective_rel_delta=0.000000e+00
```

Avoid rerunning the support build probe unless intentionally diagnosing OOM:

```bash
--support-pullback-probe-include-build
```

### 2026-07-15 continuation: support segment cotangent bridge

We started the next bridge rather than rewriting the realtime reverse path.
The frozen/profile-only reverse lane is still meant to use the existing
`--reverse-parameter-mode profiles` code path.

New implementation pieces:

- `NEOPAX/_transport_solvers.py` now has a sibling reduced accepted-step bwd
  helper that returns both the normal reduced carry cotangent and a cotangent
  wrt the explicit realtime NTX support payload:

```text
_execute_radau_accepted_step_next_reduced_cotangent_bwd_with_support(...)
_radau_segment_reduced_cotangent_bwd_with_support_call(...)
```

- `examples/benchmarks/benchmark_transport_reverse_ad_only.py` now has a
  realtime-only probe mode:

```text
--realtime-geometry-gradient-path support_segment_probe
```

What this probe tests:

```text
objective cotangent
  -> realized-schedule reduced Radau reverse replay
  -> support_payload_bar
```

This is the missing transport half of the realtime geometry bridge. It does
not yet pull `support_payload_bar` through VMEC/NTX support construction to
`RBC/ZBS`; that remains the next step.

First command to run next:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --reverse-parameter-mode profiles_plus_realtime_geometry \
  --reverse-geometry-parameter RBC:1:0 \
  --realtime-geometry-gradient-path support_segment_probe \
  --ntx-exact-derivative-mode direct \
  --ntx-exact-derivative-field-pullback-mode generic_jvp \
  --objective softmax_Er \
  --accepted-step-limit 2 \
  --radau-jacobian-reuse-mode legacy \
  --timing-mode jit-warm \
  --reverse-segment-length 1 \
  --reverse-stage-adjoint-solve-mode bicgstab \
  --reverse-rhs-transpose-mode explicit_ntx_interpolated \
  --reverse-step-bwd-mode reduced_cotangent
```

Expected useful signal:

```text
realtime_geometry_gradient_path=support_segment_probe
support_bar_l2=<finite, preferably nonzero>
support_bar_all_finite=True
```

First failure seen on the GPU run:

```text
TypeError: object of type 'object' has no len()
  NTX/src/ntx/_geometry_types.py, VmecSurface.__post_init__
```

Cause:

- The generic support-payload VJP was differentiating the whole support pytree,
  including integer/static NTX geometry leaves such as `m`, `n`, and related
  metadata. During transpose tracing those can appear as object/float0-style
  cotangents, but NTX dataclass validation expects normal array-like primals
  and calls `len(self.m)`.

Fix applied:

- Support-payload pullbacks now use a float-delta boundary:

```text
support_delta -> with_support_payload(add_float_delta(primal_support, support_delta))
```

- Floating support leaves receive derivatives.
- Non-floating/static leaves stay equal to the primal support values.
- This preserves NTX dataclass construction while keeping the differentiable
  geometry quantities active.

Patched locations:

```text
NEOPAX/_transport_flux_models.py
NEOPAX/_transport_equations.py
```

Rerun the same `support_segment_probe` command after this fix.

If this succeeds, next implementation is:

1. Build a VJP for the existing traceable realtime support construction:

```text
geometry_delta -> support_payload
```

2. Feed `support_payload_bar` into that VJP to obtain the first VMEC harmonic
   cotangent, initially for `RBC:1:0`.
3. Then add profile gradients from the same reduced reverse pass and only then
   compare to a frozen/replayed finite-difference geometry perturbation.

### 2026-07-16 continuation: realtime fixed-Er boundary probe

Current intended semantics:

- Frozen/profile reverse AD computes the baseline initial `Er` with the normal
  ambipolar solve, then treats that `Er` as fixed for profile derivatives.
- Realtime geometry reverse should first match this behavior: compute baseline
  realtime geometry and baseline ambipolar `Er`, then copy that fixed `Er`
  into the differentiated transport rollout.
- Differentiating through the ambipolar initialization is a later implicit-root
  layer, documented separately in `ambipolarity_ad_plan.md`.

Important code changes in `examples/benchmarks/benchmark_transport_reverse_ad_only.py`:

- Added scalar selected-objective evaluation so `--objective softmax_Er` does
  not build the full objective vector and accidentally expose unrelated
  nonfinite objectives.
- Realtime baseline setup now uses normal `build_runtime_context_for_geometry_param`
  ambipolar initialization for the probe baseline.
- The experimental realtime rollout helper accepts `fixed_initial_er` and copies
  `stop_gradient(baseline_state.Er)` into the profile/geometry transport initial
  state. This preserves fixed-initial-`Er` semantics for geometry perturbations.
- Added `--realtime-geometry-gradient-path initial_carry_boundary_probe`.
  This does not compute `d objective / d geometry`; it returns diagnostics for
  the compact cotangent wrt the initial transport state.

Latest successful diagnostic command:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --reverse-parameter-mode profiles_plus_realtime_geometry \
  --reverse-geometry-parameter RBC:1:0 \
  --realtime-geometry-gradient-path initial_carry_boundary_probe \
  --ntx-exact-derivative-mode direct \
  --ntx-exact-derivative-field-pullback-mode generic_jvp \
  --objective softmax_Er \
  --accepted-step-limit 2 \
  --radau-jacobian-reuse-mode legacy \
  --timing-mode jit-warm \
  --reverse-segment-length 1 \
  --reverse-stage-adjoint-solve-mode bicgstab \
  --reverse-rhs-transpose-mode explicit_ntx_interpolated \
  --reverse-step-bwd-mode reduced_cotangent
```

Latest output before the most recent patch:

```text
objective=softmax_Er
value=2.4573910204527036e-01
reduced_y_bar_all_finite=True
reduced_lagged_cache_bar_all_finite=True
state_bar_all_finite=False
no_lagged_cache_state_bar_all_finite=False
y_only_state_bar_all_finite=False
lagged_reference_only_state_bar_all_finite=False
lagged_cache_only_state_bar_all_finite=False
```

Interpretation:

- The realized-schedule reduced Radau reverse replay is finite.
- The NaNs are produced inside the benchmark-local initial-carry custom VJP.
- Since `y_only`, `lagged_reference_only`, and `lagged_cache_only` all became
  nonfinite, the likely poison is a zero-cotangent pullback inside the
  initial-carry transpose, not the geometry/support bridge itself.

Most recent patch to test next:

- In `_reverse_initial_carry_from_state_with_static_setup(...)._build_initial_carry_bwd`,
  explicit zero-cotangent guards were added around:

```text
rhs_bar -> flat_rhs_state_pullback
rhs_bar -> flat_rhs_lagged_response_pullback
lagged_bar -> pullback_build_lagged_response
```

Reason:

- In the boundary probe, `prev_stages_bar` can be zero, so the initial RHS
  contribution is mathematically zero.
- Calling the RHS transpose with an all-zero cotangent can still produce
  `0 * nan = nan` through inactive/unneeded paths.
- The patch returns explicit zero cotangents for those zero-cotangent branches.

Next run:

Rerun the same `initial_carry_boundary_probe` command above.

Expected useful signal:

```text
reduced_y_bar_all_finite=True
reduced_lagged_cache_bar_all_finite=True
y_only_state_bar_all_finite=True
```

If `y_only_state_bar_all_finite` remains false, inspect the direct state packing
pullback from `_make_solver_state_transform` next. If `y_only` becomes finite
but full `state_bar_all_finite` remains false, continue isolating the lagged
reference/cache pullback branches.
