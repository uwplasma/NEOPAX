# Shared Payload Efficiency

## Current Reverse Bottleneck

The full-transport shared-payload reverse path is correctness-sensitive at the
Radau accepted-step adjoint solve. The expensive section is the segmented
cotangent sweep, especially the per-step Radau stage-adjoint solve used by:

- `_radau_solve_exact_stage_residual_transpose(...)`
- `_radau_solve_exact_stage_residual_transpose_batched(...)`
- `_execute_radau_accepted_step_next_reduced_cotangent_bwd(...)`
- `_execute_radau_accepted_step_next_reduced_cotangent_batched_bwd_with_support(...)`

The current good production-like benchmark uses:

```text
--reverse-stage-adjoint-solve-mode bicgstab
--reverse-rhs-transpose-mode explicit_ntx_interpolated
--reverse-step-bwd-mode reduced_cotangent
--reverse-segment-length 4
--objective all
```

This path is the reference for the current reverse AD vs frozen FD tables.

## Existing Solve Modes

### `structured`

`structured` applies the transpose of the already-factorized Radau Newton
linear solve:

```text
T^T block_solve_transpose(T^{-T} rhs)
```

This is compact and fast, but it is not the exact transpose solve for the
converged nonlinear Radau stage residual. Recent structured-mode checks showed
good geometry-only rows but bad transport objective derivatives, especially for
`Er_volume_average`, `Er2_volume_average`, `softmax_Er`, pressure, alpha power,
and electron temperature. Therefore `structured` is not acceptable as the
production full-transport shared-payload reverse rule.

### `bicgstab` / `gmres`

These solve the exact stage-residual transpose system matrix-free. For Radau
stages,

```text
R_i = z_i - f_i(t_i, y + h sum_j A_ij z_j)
```

so the exact reverse solve is

```text
(dR/dz)^T lambda = rhs
lambda_i - h sum_j A_ji J_j^T lambda_j = rhs_i
```

The matrix-free matvec avoids dense Jacobian construction, but each Krylov
iteration applies stage RHS pullbacks. That makes the reverse cotangent sweep
too slow and contributes to very large compile/runtime cost.

### `block`

`block` is exact, but currently constructs full stage Jacobian blocks using
`jax.jacfwd`, materializes the dense `(num_stages * state_dim)^2` system, and
then calls `jnp.linalg.solve`. This is useful as a diagnostic but is not compact
enough for the shared-payload benchmark or optimization.

## Needed Rule

We need a new compact exact block rule, conceptually:

```text
--reverse-stage-adjoint-solve-mode exact_block_compact
```

This mode name is now wired in the benchmark CLI. The first implementation is
an exact dense-block correctness prototype behind the new mode name; it is not
yet the final lower-memory compact factorization.

The target behavior is:

- exact like `bicgstab` / `gmres`
- compact like `structured`
- no dense full stage Jacobian materialization
- no Krylov loop over repeated RHS VJPs
- compatible with the batched all-objective shared-payload table
- compatible with support payload cotangents for realtime geometry / NTX support
- correctness-checked against the current frozen FD comparison tables before
  being used for optimization

## Implementation Target

The implementation should replace the expensive exact solve inside:

```text
_radau_solve_exact_stage_residual_transpose(...)
_radau_solve_exact_stage_residual_transpose_batched(...)
```

but only for a new explicit mode, leaving current `bicgstab`, `gmres`,
`structured`, and diagnostic `block` behavior available during validation.

Step 2 has isolated the exact one-step transpose operator into:

```text
_radau_exact_stage_residual_transpose_matvec(...)
_radau_exact_stage_residual_transpose_matvec_batched(...)
```

The existing `bicgstab` / `gmres` paths now call these helpers, so the operator
that `exact_block_compact` must invert is named and reusable without changing
the current reference solve modes.

Step 3 has isolated the dense exact stage-residual matrix into:

```text
_radau_exact_stage_residual_matrix(...)
```

and added a private comparison helper:

```text
_radau_exact_stage_residual_transpose_matvec_diagnostic(...)
```

This lets us compare the compact extracted transpose matvec against the dense
`block` matrix action without changing `bicgstab`, `gmres`, `structured`, or
`block` semantics. The diagnostic is intentionally private for now; it is a
guardrail for implementing `exact_block_compact`, not a production mode.

Step 4 wires that guardrail into the reverse benchmark as an exit-after-check
diagnostic:

```text
--local-transpose-diagnostic-accepted-step 0 --stage-matvec-diagnostic
```

This runs the baseline adaptive rollout, reconstructs the requested accepted
step, runs that accepted-step primal once, and prints/writes compact-vs-dense
stage transpose matvec metrics:

```text
compact_l2, dense_l2, diff_l2, rel_err, max_abs_diff
```

It still does not activate `exact_block_compact` or change the shared-payload
gradient path.

`exact_block_compact` can now be selected, but it intentionally refuses to run
unless a non-dense compact solve hook is present. The previous dense fallback is
disabled so this mode cannot accidentally materialize the full
`(num_stages * state_dim)^2` matrix while being called "compact".

The important algebraic constraint is:

```text
structured: factors I - h * eig(A) * J_ref
exact:      factors delta_ij I - h * A_ij * J_i
```

where `J_ref = df(t_n, y_n)/dy`, but the exact nonlinear Radau stage residual
uses one `J_i = df(t_i, y_i)/dy_i` for each converged stage state. The Radau
stage transform diagonalizes the stage coupling only when the same Jacobian is
used at every stage. Therefore the exact compact non-iterative rule cannot just
reuse the existing structured LU without changing the derivative; that is why
`structured` is fast but not accurate enough for the full-transport objective
table.

The desired rule should use the accepted primal step's already-built Radau /
Newton linearization objects where possible, but solve the true nonlinear
stage-residual transpose system:

```text
lambda_i - h sum_j A_ji J_j^T lambda_j = rhs_i
```

not merely the approximate transformed Newton linear-solve transpose used by
`structured`.

The next real implementation step is a true exact compact representation of
the per-stage Jacobian block system. That means either:

- adding explicit sparse/banded RHS Jacobian assembly hooks and doing one
  direct sparse/block factorization per accepted step with all objective RHS
  columns solved together, or
- finding/proving a transport-specific block factorization of the per-stage
  `J_i` system that avoids materializing the full dense block.

Using the current matrix-free VJP operator plus a direct inverse is not enough:
VJP gives compact actions of `(dR/dz)^T`, but a non-iterative inverse still
needs some factored representation of that operator.

## Compact Block Direction

The promising direct compact rule is radial-block factorization, not dense
stage-block factorization.

The current flat ordering is:

```text
[density(:), pressure(:), Er(:)]
```

and the exact dense stage matrix is naturally written stage-major:

```text
stage 0 all state, stage 1 all state, ...
```

For a compact direct solve we should instead view the unknowns as radial-cell
blocks:

```text
cell k: all evolved variables at cell k for all Radau stages
```

With face-primary finite-volume transport, most RHS couplings are local in
radius:

- density/temperature flux-divergence terms couple through neighboring faces
- center/source/work terms are cell-local, plus species coupling
- Er ambipolar terms use center fluxes, interpolated or locally evaluated from
  the same face/center response data
- quasi-neutrality and active-temperature projection add local species mixing

This suggests an exact block-banded system in radius. The block size is roughly:

```text
num_stages * variables_per_radial_cell
```

rather than:

```text
num_stages * total_state_dim
```

The implementation target for `solve_exact_stage_transpose_compact(...)` is:

1. Build per-stage local RHS derivative blocks in radial-block form, including
   equation assembly and lagged flux-response derivatives.
2. Assemble the exact transpose stage residual in radial block-banded form.
3. Factor it once per accepted step with a block-banded/direct sweep.
4. Solve all objective RHS columns in that one factorization.
5. Validate against `bicgstab` first, then against frozen-root FD.

This is the analogue of the compact VMEC/NTX block solve: not one objective at
a time, not Krylov iteration, and not dense global materialization.

Initial infrastructure now exists in `_transport_solvers.py`:

- `_radau_block_tridiagonal_solve(...)` solves block-tridiagonal systems with
  one or many RHS columns, including transpose systems, without global dense
  materialization.
- `_radau_infer_radial_block_layout(...)` infers the packed density/pressure/Er
  per-cell variable layout from the Radau kernel context.
- `_radau_stage_stack_to_radial_blocks(...)` and
  `_radau_radial_blocks_to_stage_stack(...)` convert between the current
  stage-major ordering and the planned radial-block ordering.

The next code step is to assemble the actual lower/diagonal/upper radial blocks
for the exact per-stage transport RHS. Until those blocks exist,
`exact_block_compact` correctly refuses to run.

The existing local stage diagnostic now also reports whether the exact dense
operator is compatible with this radial block-tridiagonal assumption:

```text
--local-transpose-diagnostic-accepted-step 0 --stage-matvec-diagnostic
```

The relevant output fields are:

```text
radial_block_count
radial_block_dim
radial_off_tridiagonal_l2
radial_off_tridiagonal_rel_l2
radial_off_tridiagonal_max_abs
```

Interpretation:

- If `radial_off_tridiagonal_rel_l2` and `radial_off_tridiagonal_max_abs` are
  near roundoff, a block-tridiagonal radial factorization is structurally
  correct for that configuration.
- If they are not small, then some term is coupling beyond nearest radial
  neighbors and the compact direct solve must use a wider radial bandwidth or
  isolate that nonlocal term.

The block-tridiagonal solve utility was checked against an explicitly assembled
dense toy system for both primal and transpose solves; the relative errors were
approximately `9.1e-17` and `6.3e-17`.

## Validation Gate

Before promotion, the new mode must be compared against:

- the current good `bicgstab` shared-payload reverse AD table
- the frozen-linearized FD table with frozen initial-Er root branch
- both profile parameters and realtime geometry parameters:
  `n0`, `T0`, `density_shape_power`, `temperature_shape_power`,
  `density_shape_alpha`, `temperature_shape_alpha`, `RBC:1:0`, `ZBS:1:0`

The new mode is only acceptable if transport objective derivatives match the
current good reverse/FD values at the same level as the `bicgstab` reference.
