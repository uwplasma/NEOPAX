# Shared Payload Efficiency

## 2026-08-13 Exact `block` vs `bicgstab` 16-Step Timing And Values

Run configuration:

```text
examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml
--reverse-parameter-mode profiles_plus_realtime_geometry
--reverse-geometry-parameter RBC:1:0,ZBS:1:0
--realtime-geometry-gradient-path reverse_payload
--objective all
--accepted-step-limit 16
--reverse-segment-length 4
--reverse-rhs-transpose-mode explicit_ntx_interpolated
--reverse-step-bwd-mode reduced_cotangent
--full-transport-shared-payload-smoke
```

Both runs used the same schedule and support counts:

```text
segments=4, segment_length=4, residual_count=16, parameter_count=8
support_reuse=4, support_rebuild=12
```

Timing comparison:

| Component | `block` elapsed_s | `bicgstab` elapsed_s | `block - bicgstab` |
| --- | ---: | ---: | ---: |
| realtime geometry runtime build | `217.353` | `217.606` | `-0.253` |
| realtime geometry solver components | `31.484` | `31.609` | `-0.125` |
| support reverse profile-state vjp | `31.286` | `31.208` | `+0.078` |
| support reverse initial carry vjp | `22.740` | `22.504` | `+0.236` |
| support reverse realized-schedule vjp forward | `431.152` | `430.285` | `+0.867` |
| support reverse segmented cotangent sweep | `1758.204` | `2014.368` | `-256.164` |
| support reverse initial-cache support pullback | `643.112` | `643.382` | `-0.270` |
| support reverse reduced carry bars expanded | `0.171` | `0.166` | `+0.005` |
| support reverse initial state pullback | `81.139` | `80.978` | `+0.161` |
| initial-Er root boundary compact pullback | `54.356` | `51.509` | `+2.847` |
| support reverse profile parameter pullback | `0.196` | `0.204` | `-0.008` |
| objective-table final VMEC parameter pullback | `13.697` | `13.853` | `-0.156` |
| total run | `3453.605` | `3705.434` | `-251.829` |

Per-segment timing:

| Segment | `block` elapsed_s | `bicgstab` elapsed_s | Support reuse/rebuild |
| --- | ---: | ---: | --- |
| `4/4` | `985.473` | `1196.844` | `0 / 4` |
| `3/4` | `111.377` | `126.113` | `2 / 2` |
| `2/4` | `330.747` | `345.568` | `2 / 2` |
| `1/4` | `330.607` | `345.841` | `0 / 4` |

Notes:

- `block` was faster than `bicgstab` by `251.829 s` total, about `6.8%`.
- The segmented cotangent sweep was faster by `256.164 s`, about `12.7%`.
- The first executed reverse segment still dominates because it includes a large XLA compile. The slow-compile alarm was about `4m04s` for `block` and `5m55s` for `bicgstab`.
- The large `support reverse initial-cache support pullback` cost is essentially unchanged (`643 s`) and is outside the stage-adjoint solve mode choice.
- Observed RAM monitor trace for the `bicgstab` run sat close to `47%` during the reverse work and dropped to about `10%` when the process finished. This is not a code-instrumented peak. The currently tested Woodbury paths did not improve this in practice: they used similar memory and were slower, because they still go through dense/SVD validation machinery. Therefore this RAM trace is evidence that the present exact-stage reverse family is heavy, not evidence that the current Woodbury implementation is already better.

Derivative comparison:

Across the two printed optimization API residual/Jacobian tables:

```text
common entries compared: 112
max absolute difference: 1.659860e-06
max relative difference: 1.150239e-04
```

The worst relative differences are on tiny `smooth_root_proxy` derivatives near
machine zero. The largest meaningful absolute differences are:

| Entry | `block` | `bicgstab` | Abs diff | Rel diff |
| --- | ---: | ---: | ---: | ---: |
| `d transport:softmax_Er / dtemperature_shape_power` | `3.5362624276528196e+00` | `3.5362607677924291e+00` | `1.659860e-06` | `4.693831e-07` |
| `d transport:softmax_Er / dn0` | `-4.7372071028091174e+00` | `-4.7372059445518024e+00` | `1.158257e-06` | `2.445022e-07` |
| `d transport:Er2_volume_average / dtemperature_shape_power` | `5.4946436756445031e+01` | `5.4946438108771410e+01` | `1.352326e-06` | `2.461170e-08` |
| `d transport:Er2_volume_average / dn0` | `-6.1442935693371233e+01` | `-6.1442936636985735e+01` | `9.436145e-07` | `1.535762e-08` |
| `d transport:softmax_Er / dT0` | `3.9104307061781909e+00` | `3.9104300296804584e+00` | `6.764977e-07` | `1.729982e-07` |

Conclusion:

- `block` and `bicgstab` are numerically equivalent for this benchmark at the
  printed precision.
- This makes `block` a valid exact reference for checking future compact direct-solve paths.
- `block` is only modestly faster, and still not acceptable as the final compact
  optimization path because it materializes dense stage Jacobians.

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

## 2026-08-14 Memory Monitor Note

The latest attached monitor screenshot for the 30 GB machine shows RAM usage
approximately flat near half of total memory during the 16-step shared-payload
reverse run. Record this as:

```text
observed_ram_monitor_fraction ~= 0.5
observed_ram_monitor_total_gb = 30
observed_ram_monitor_used_gb ~= 15
```

This is a visual monitor estimate, not an instrumented peak RSS or GPU memory
measurement. It should be used only for rough comparison with other runs on the
same machine.

The pasted log attached to this memory screenshot was a 16-step `block` run:

```text
--reverse-stage-adjoint-solve-mode block
--accepted-step-limit 16
--reverse-segment-length 4
support reverse segmented cotangent sweep elapsed_s=1421.110
total elapsed_s=3116.764
```

The command for the current Woodbury implementation, using rank 24, is:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --reverse-parameter-mode profiles_plus_realtime_geometry \
  --reverse-geometry-parameter RBC:1:0,ZBS:1:0 \
  --realtime-geometry-gradient-path reverse_payload \
  --objective all \
  --accepted-step-limit 16 \
  --radau-jacobian-reuse-mode legacy \
  --timing-mode jit-warm \
  --reverse-segment-length 4 \
  --reverse-stage-adjoint-solve-mode woodbury_matvec_compact \
  --reverse-stage-adjoint-woodbury-rank 24 \
  --reverse-rhs-transpose-mode explicit_ntx_interpolated \
  --reverse-step-bwd-mode reduced_cotangent \
  --initial-Er-root-ad jax_selected_root \
  --full-transport-shared-payload-smoke
```

## 2026-08-14 `woodbury_matvec_compact` Rank-24 16-Step Run

Run mode:

```text
--reverse-stage-adjoint-solve-mode woodbury_matvec_compact
--reverse-stage-adjoint-woodbury-rank 24
--accepted-step-limit 16
--reverse-segment-length 4
```

Observed timings:

| Component | `woodbury_matvec_compact` elapsed_s | Newer `block` elapsed_s | Older `bicgstab` elapsed_s |
| --- | ---: | ---: | ---: |
| realtime geometry runtime build | `217.798` | `218.060` | `217.606` |
| realtime geometry solver components | `31.707` | `31.674` | `31.609` |
| support reverse profile-state vjp | `31.365` | `31.274` | `31.208` |
| support reverse initial carry vjp | `22.592` | `22.630` | `22.504` |
| support reverse realized-schedule vjp forward | `433.376` | `431.732` | `430.285` |
| support reverse segmented cotangent sweep | `1840.513` | `1421.110` | `2014.368` |
| support reverse initial-cache support pullback | `649.143` | `646.174` | `643.382` |
| support reverse initial state pullback | `81.362` | `78.676` | `80.978` |
| initial-Er root boundary compact pullback | `51.377` | `54.287` | `51.509` |
| objective-table final VMEC parameter pullback | `14.159` | `13.517` | `13.853` |
| total run | `3542.966` | `3116.764` | `3705.434` |

Timing interpretation:

- Compared with the newer `block` run attached with the memory screenshot,
  `woodbury_matvec_compact` is slower by `426.202 s` total and `419.403 s` in
  the segmented cotangent sweep.
- Compared with the older saved `bicgstab` timing table, it is faster by
  `162.468 s` total and `173.855 s` in the segmented cotangent sweep.
- This confirms the current rank-24 Woodbury mode is a correctness experiment,
  not the final efficiency win. It still pays too much compile/runtime overhead
  from the matvec/SVD-style construction path.

Derivative comparison against the newer exact `block` run:

```text
common printed Jacobian entries compared: 128
max absolute difference: 9.510020660172813e-04
```

Largest absolute differences:

| Entry | `woodbury_matvec_compact` | newer `block` | Abs diff | Rel diff |
| --- | ---: | ---: | ---: | ---: |
| `d transport:Er2_volume_average / dvmec:RBC:1:0` | `-6.4027761396697815e+02` | `-6.4027856496904417e+02` | `9.510021e-04` | `1.485294e-06` |
| `d transport:Er2_volume_average / dvmec:ZBS:1:0` | `2.3683363067131035e+02` | `2.3683437561688766e+02` | `7.449456e-04` | `3.145428e-06` |
| `d transport:softmax_Er / dvmec:RBC:1:0` | `-5.9293842886185210e+01` | `-5.9294180972770484e+01` | `3.380866e-04` | `5.701851e-06` |
| `d transport:softmax_Er / dvmec:ZBS:1:0` | `1.7363305438804826e+01` | `1.7363571489179520e+01` | `2.660504e-04` | `1.532233e-05` |
| `d transport:Er2_volume_average / dtemperature_shape_power` | `5.4946272544225238e+01` | `5.4946436756445031e+01` | `1.642122e-04` | `2.988587e-06` |

The values are close, but the Woodbury truncation error is larger than the
`block` vs `bicgstab` difference saved above (`1.659860e-06` max absolute
difference). Rank 24 is therefore good as a diagnostic approximation, but it
has not yet replaced the exact solver path for production correctness.

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

The current code now separates the Woodbury solve machinery from how the
low-rank factors are obtained:

- `_radau_solve_radial_block_tridiagonal_plus_low_rank(...)` performs the
  block-tridiagonal solve and rank-`k` Woodbury correction for one or many RHS
  columns.
- The existing `woodbury_compact` and `woodbury_matvec_compact` modes still
  obtain `U,V` from dense/SVD validation paths. They are correctness oracles,
  not the final optimization path.
- `woodbury_er_coeff_compact` is reserved as the production target mode. It
  requires an equation-system `exact_stage_transpose_low_rank_factors(...)`
  hook and refuses to fall back to the dense/SVD validation paths.
- The remaining production step is to build the Er-ambipolar-coefficient
  low-rank factors directly, avoiding both dense global matrix construction and
  in-JIT SVD.

The production hook must return both pieces:

```text
B = exact block-tridiagonal radial stage-transpose base
U,V = analytic low-rank Er-ambipolar-coefficient off-band correction
```

The analytic `U,V` part is constrained by the `ntss_like_midpoint` formula
above. The still-missing implementation is the non-dense assembly of the
block-tridiagonal base `B`; using the current dense/SVD or full-basis matvec
paths to obtain `B` would reproduce the slow compile/runtime behavior we are
trying to remove.

Packed-layout note: the reverse prepared-rollout path must use component sizes
from the packed solver state, not from the public full state. With quasi-neutral
packing, electron density is removed and Er remains in the flat state. Using the
full public density size mislabels the packed Er tail as density/extra structure
and makes radial block diagnostics misleading. This was corrected before
continuing the compact block-factor work.

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

## Local Stage Diagnostic: wHe Realtime Geometry

Command shape:

```text
benchmark_transport_reverse_ad_only.py
  --config examples/benchmarks/Solve_Transport_equations_wHe_radau_ntx_exact_lagged_runtime_vmec_realtime_geometry_benchmark.toml
  --accepted-step-limit 1
  --reverse-segment-length 1
  --reverse-stage-adjoint-solve-mode bicgstab
  --local-transpose-diagnostic-accepted-step 0
  --stage-matvec-diagnostic
```

Current result after applying the projected-state pullback correction:

```text
compact_l2=1.028406e+00
dense_l2=1.028406e+00
compact_vs_dense_diff_l2=1.842066e-16
compact_vs_dense_rel_err=1.791186e-16
generic_vjp_rel_err=1.870954e-16
explicit_rhs_state_rel_err=2.512159e-16
split_rhs_state_rel_err=4.742437e-16
```

This confirms that the compact local transpose and explicit split pullbacks now
match the dense/generic reference path to roundoff for the `wHe` realtime
geometry diagnostic. The previous large mismatch in this case was caused by
skipping the transpose of the state projection when building compact flat-state
RHS pullbacks.

The radial off-band attribution for this case is:

```text
state_dim=408
block_count=51
block_dim=56
off_tridiagonal_l2=1.740960e-01
off_tridiagonal_rel_l2=3.200945e-03
off_tridiagonal_max_abs=8.184844e-03
density_off_l2=0.000000e+00
pressure_off_l2=0.000000e+00
er_off_l2=1.740715e-01
```

The Er subterm split shows that the off-band coupling comes from the Er
ambipolar coefficient path, not density, pressure, Er diffusion, or the
charge-flux term:

```text
diffusion_off_l2=4.205719e-20
ambipolar_off_l2=1.740715e-01
coeff_off_l2=1.740715e-01
charge_flux_off_l2=0.000000e+00
```

For the benchmark cases using `Er_permittivity_mode = "ntss_like_midpoint"`,
this coefficient path is global because the Er ambipolar RHS uses one midpoint
ion-density scalar:

```text
ni_mid = sum_a n_a[r_mid]
coeffG = 95780 * B0_mid^2 / (ni_mid * psfactor_mid)
Er_rhs_ambipolar = -Er_relax * coeffG * charge_flux * 1e-20
```

Therefore all Er radial rows depend on the same midpoint-density scalar. That
is the mathematical source of the off-block-tridiagonal radial coupling and
also why a low-rank Woodbury correction is the right compact direct solve
structure.

Low-rank Woodbury solve diagnostics for the same one-step system:

```text
rank12_rel=2.730600e-06
rank16_rel=3.417019e-08
rank24_rel=1.103071e-09
rank32_rel=8.940431e-10
rank48_rel=4.622310e-10
```

For this case, rank 24 is already effectively at the useful accuracy floor of
the diagnostic. Higher rank improves only marginally because the remaining
error is dominated by numerical conditioning/roundoff rather than missing
low-rank modes.

Next compact-solver guard added on this branch:

```text
local stage radial-band builder check:
  block_dim=...
  permutation_max_abs_diff=...
  diff_l2=...
  rel_err=...
  max_abs_diff=...
```

This compares the new compact band assembly, built from compact transpose
matvecs and storing only lower/diagonal/upper radial block bands, against the
dense radial transpose oracle. For the compact direct solver to be trusted, the
band-builder `rel_err` should be at roundoff, like the existing compact-vs-dense
transpose checks. If this passes, the remaining production work is only the
analytic Er-ambipolar-coefficient low-rank factor path.
