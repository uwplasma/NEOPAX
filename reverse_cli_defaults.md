# Realtime exact-NTX reverse CLI defaults

This records the benchmark CLI defaults for the realtime-geometry exact-NTX
reverse path. It changes reverse implementation modes only: it does not set
an accepted-step limit, segment length, objectives, geometry parameters,
timing mode, or Radau Jacobian reuse policy.

## Previous CLI defaults

| Option | Previous CLI default |
|---|---|
| `--initial-Er-root-ad` | `off` |
| `--ntx-exact-derivative-pullback-algebra` | `ntx_helper` |
| `--reverse-stage-adjoint-solve-mode` | `structured` |
| `--reverse-rhs-transpose-mode` | `generic` |
| `--reverse-initial-cache-support-pullback-mode` | `scalar` |
| `--reverse-rebuild-support-pullback-mode` | `separate` |
| `--reverse-segment-start-replay-mode` | `legacy` |
| `--reverse-segment-primal-record-mode` | `reconstruct` |
| `--reverse-schedule-artifact-mode` | `legacy` |
| `--reverse-step-bwd-mode` | `current` |
| `--reverse-final-objective-cotangent-mode` | `scalar` |
| `--reverse-bootstrap-cotangent-mode` | `separate` |

Every prior option remains selectable explicitly. They are the broad
reference lane and remain appropriate for non-exact/database configurations.

## Previous standard exact realtime CLI

```text
--initial-Er-root-ad jax_selected_root
--ntx-exact-derivative-pullback-algebra ntx_helper_lowdot_fused
--reverse-stage-adjoint-solve-mode block
--reverse-rhs-transpose-mode explicit_ntx_interpolated
--reverse-step-bwd-mode reduced_cotangent_call_boundary
--reverse-initial-cache-support-pullback-mode ntx_batched_interpolated_faces
--reverse-rebuild-support-pullback-mode separate_reuse_local_vjp_primal
--reverse-segment-start-replay-mode minimal
--reverse-segment-primal-record-mode reuse_segment_primal_record
--reverse-schedule-artifact-mode reuse_static_probe
--reverse-final-objective-cotangent-mode scalar
--reverse-bootstrap-cotangent-mode separate
```

This is the conservative explicit fallback for comparisons with the grouped
native VMEC-coefficient implementation.

## Current fastest exact realtime CLI defaults

```text
--initial-Er-root-ad jax_selected_root
--ntx-exact-derivative-pullback-algebra ntx_helper_lowdot_fused
--reverse-stage-adjoint-solve-mode block
--reverse-rhs-transpose-mode explicit_ntx_interpolated
--reverse-step-bwd-mode reduced_cotangent_call_boundary
--reverse-initial-cache-support-pullback-mode ntx_batched_interpolated_faces
--reverse-rebuild-support-pullback-mode ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_directional_product_rule
--reverse-segment-start-replay-mode minimal
--reverse-segment-primal-record-mode reuse_segment_primal_record
--reverse-schedule-artifact-mode reuse_static_probe
--reverse-final-objective-cotangent-mode scalar
--reverse-bootstrap-cotangent-mode joint_local_vjp_upar_only
```

The direct-directional rebuild selector preserves the grouped NTX
factorisation, matrix-RHS adjoint, and VMEC-coefficient bridge. It changes
only the two post-adjoint directional primitive contractions. Its full
benchmark timing and derivative table still need recording; it is the
requested default mode, not yet a completed performance claim.

`--reverse-schedule-artifact-mode reuse_static_probe` still requires
`--full-transport-shared-payload-smoke`, by design.  A command that is not
using that shared-payload benchmark path must explicitly select
`--reverse-schedule-artifact-mode legacy`.

For `ntx_database`, do not rely on these exact-NTX defaults. Its separate
reverse payload route is tracked in `database_ntx_path_reverse.md`.
