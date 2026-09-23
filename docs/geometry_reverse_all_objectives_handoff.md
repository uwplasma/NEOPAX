# Geometry Reverse All-Objectives Handoff

Goal: restore the `geometry_full_ad_objectives` all-objective reverse table so it gives the correct compact reverse derivatives, especially the `boozer_qi_objective` row.

Hard requirements:
- Use reverse AD only for the benchmarked reverse rule.
- Do not replace the result with forward JVP, finite differences, tolerance tuning, or scalar objective sweeps.
- Do not contaminate the forward solver realtime/frozen geometry paths.
- Preserve the existing compact VMEC implicit RHS/multi-RHS reverse machinery.

Trusted scalar diagnostic:
`python ./examples/benchmarks/compare_geometry_qi_frozen_linearized_fd.py --vmec-input ./examples/inputs/input.QI_nfp2_newNT_opt_hires_true --parameter RBC:1:0 --objective boozer_qi_objective --reference-fd=5.909044e-01 --multigrid`

Trusted reference values from the earlier scalar check:
- `forward_jvp=6.9347871114079501e-03`
- `reverse_state_dot_tangent=6.9347871114111004e-03`
- `implicit_reverse_param_grad=7.0168957131144793e-03`

Interpretation:
- `reverse_state_dot_tangent` is `<dQI/dVMEC_state, dVMEC_state/dRBC:1:0>`.
- `implicit_reverse_param_grad` is the QI state cotangent sent through the VMEC-JAX implicit pullback.
- The all-objective table QI row should recover approximately the same scale, not the unrelated full nonlinear FD reference `5.909044e-01`.

Current code state:
- File: `NEOPAX/_geometry_autodiff.py`
- Function: `geometry_full_ad_objective_table_pullback_from_param_vector`
- The table default `final_vmec_pullback_mode` is back to `"vmap"` so the non-QI objective rows use `_implicit_state_pullback_multi_rhs_with_assemble_rhs`.
- The QI row is no longer mixed into the grouped table `state_bar`.
- The QI row is computed separately:
  `state -> checkpointed _vmec_booz_qi_scalar_objective_from_state -> qi_state_pullback -> implicit.implicit_state_pullback_multi_rhs -> insert QI row`.
- This recovers the older compact RHS row-replacement structure found in commit `97a490d`, with one change: `jax.checkpoint(qi_scalar_from_state)` is used because the uncheckpointed QI VJP OOMed at `booz_xform_from_inputs` while allocating 83.41 MiB.

Important recent observations:
- Latest failure before rerun was a VMEX/JAX pytree registration compatibility issue, not a reverse-rule issue: updated VMEX failed while registering `ImplicitSolution` with `drop=("runtime",)`. The fix is in VMEX `vmex/core/transforms.py`: when `drop` fields are requested, use an explicit `register_pytree_node` so dropped fields such as `runtime` are excluded from the pytree and restored from defaults on unflatten.
- Reduced Boozer QI cotangent path gave too-small QI row values such as `1.625793e-03` and `4.143428e-03`.
- Full Boozer objective-block cotangent path moved too high, e.g. `9.683326e-03` to `1.350655e-02`.
- Restoring the exact older uncheckpointed QI state-VJP row replacement OOMed at:
  `qi_value, qi_state_pullback = jax.vjp(qi_scalar_from_state, state)`.
- The current hybrid should be tested next: same row replacement, but checkpointed QI VJP.

Next command to run:
`python ./examples/benchmarks/benchmark_geometry_vmec_booz_fd_vs_ad.py --mode geometry_full_ad_objectives --vmec-input ./examples/inputs/input.QI_nfp2_newNT_opt_hires_true --param-specs RBC:1:0,ZBS:1:0 --fd-rel-step 3e-7 --fd-abs-step 1e-10 --ad-backend implicit --fd-lane ad --reverse-derivative-mode objective_table --skip-fd-check`

Expected next-session decision:
- If the QI row is near the scalar target (`~6.9e-03` to `~7.0e-03` for `RBC:1:0`) and avoids OOM, keep this compact row-replacement rule and then clean comments/naming.
- If checkpointed QI still OOMs, recover the older lower-memory Boozer/QI trace setup rather than changing the math.
- If checkpointed QI runs but gives a wrong derivative, remove checkpointing and investigate why checkpoint changes the QI state cotangent; do not substitute FD/JVP.

Before editing:
`git diff -- NEOPAX/_geometry_autodiff.py examples/benchmarks/benchmark_geometry_vmec_booz_fd_vs_ad.py docs/geometry_reverse_all_objectives_handoff.md`

## 2026-07-21 update: raw-block pair vs GMRES-only pair

The latest scalar QI diagnostic showed that the previous `~6.6e-03` / `~6.9e-03`
target was not an operator-independent truth. It came from the preconditioned
GMRES-only forward tangent path at a particular linear-solve budget. Increasing
the forward linear budget moved that value, so it should not be used as the
sole oracle for the raw-block reverse rule.

Current key command:

```bash
python ./examples/benchmarks/compare_geometry_qi_frozen_linearized_fd.py \
  --vmec-input ./examples/inputs/input.QI_nfp2_newNT_opt_hires_true \
  --parameter RBC:1:0 \
  --objective boozer_qi_objective \
  --reference-fd=5.909044e-01 \
  --multigrid \
  --forward-linear-solve-mode raw_block \
  --forward-linear-maxiter 300 \
  --adjoint-maxiter 300
```

Observed raw-block forward result:

- `frozen_linearized_fd=-3.0868613803103507e-03`
- `forward_jvp=-3.0867443678659922e-03`

Earlier raw-block transpose reverse result:

- `raw_block_transpose_reverse_param_grad=-3.0867443806652517e-03`

Interpretation:

- The raw-block forward FD/JVP and raw-block transpose reverse are internally
  consistent.
- The previous mismatch was caused by comparing different linearized operator
  families:
  - preconditioned GMRES-only forward tangent,
  - raw block-transpose reverse.
- For the raw-block pair, the comparison should be:
  `frozen_linearized_fd` / `forward_jvp` against
  `raw_block_transpose_reverse_param_grad`.
- For the preconditioned pair, the comparison should instead be:
  preconditioned forward GMRES against preconditioned transpose GMRES, with
  convergence/preconditioning fixed.

Relation to VMEX QI optimization:

- VMEX optimization examples call `opt.least_squares(..., jac="implicit")`.
- Inside that implicit Jacobian path the default is `jac_solver="block"`.
- That block optimization path is closer to the raw-block diagnostic than to
  the drifting GMRES-only diagnostic, but it is not identical:
  - it first solves the raw block-tridiagonal system,
  - then applies a short preconditioned GMRES correction from that raw-block
    initial guess:

```text
A_raw dz0 = -F_raw,p dp
dz = GMRES(A_pre, -F_pre,p dp, x0=dz0, max_restarts=min(3, cfg.adjoint_maxiter))
dJ = J_z dz + J_p dp
```

Recommended next test:

- Add or expose a diagnostic forward mode matching the optimization path:
  `raw_block_plus_correction`.
- Compare it against a matching reverse rule:
  a preconditioned transpose solve using the raw block-transpose solution as
  a real preconditioner or robust warm-start, not as an unrelated raw-only
  gradient.
- Keep these as opt-in diagnostics until the operator pairing is validated.

Do not change the production forward solver or the default VMEX optimization
path while doing this comparison.

Implementation note:

- Added opt-in VMEX helper `implicit_state_tangent_block_corrected(...)`.
- Added NEOPAX diagnostic switch `--forward-linear-solve-mode block_corrected`.
- This mirrors the VMEX optimization `jac_solver="block"` single-direction
  tangent pattern:

```text
A_raw dz0 = -F_raw,p dp
dz = GMRES(A_pre, -F_pre,p dp, x0=dz0, max_restarts=min(3, cfg.adjoint_maxiter))
dJ = J_z dz + J_p dp
```

- This helper is not used by default and should not affect existing VMEX
  optimization behavior or NEOPAX forward solver paths.
