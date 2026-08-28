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

## Validated per-phase benchmark baseline (2026-08-28)

This is the accepted mode inventory for the cache-disabled, 16 accepted-step,
four-step-segment exact-NTX realtime-VMEC benchmark.  It is a phase-by-phase
record, rather than an assertion that every phase is already optimal.

| Reverse phase | Selected implementation | Measured/validated status |
|---|---|---|
| Initial primal and selected Er root | One selected-root solve retained for the later manual root boundary | The root profile and finite mask are reused; no second root solve is permitted. |
| Final objective cotangents | `scalar` ordinary objective VJPs + `joint_local_vjp_upar_only` bootstrap | About 55–57 s total; bootstrap compact remains the dominant ~46 s component. |
| Segmented Radau sweep | `reduced_cotangent_call_boundary`, `minimal` replay, `reuse_segment_primal_record` | Retains the established grouped objective/RHS execution behaviour. |
| Rebuild support transpose | `ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_directional_product_rule` | Accepted execution-mode baseline; retains grouped NTX factorisation and matrix-RHS adjoint. |
| Initial cache support transpose | `ntx_batched_interpolated_faces` | Current reference. The `rebuild_dispatch` and wide joint alternatives regressed time and/or memory. |
| Initial carry/state transpose | Standard reduced-carry state pullback | Current reference; no native-wide joint replacement is selected. |
| Initial-Er root boundary | Explicit compact implicit pullback with the retained forward selected-root primal | 93.393 s → 53.215 s in the phase diagnostic. Remaining components: root linearization 9.936 s, state 12.646 s, geometry 13.590 s, NTX support 16.979 s. |
| Profile parameter and initial-profile geometry accumulation | Existing compact pullbacks | Sub-second in the benchmark. |

The root reuse is implementation behaviour, not a CLI selector.  It keeps
the same selected-root implicit derivative and adds only the already-present
Er profile plus its radial finite-mask as live primal data; it does not retain
an NTX primal/factorisation/root-iteration tape.

For a diagnostic run, append `--reverse-phase-timing-diagnostics` to the
current fastest exact realtime CLI above.  Diagnostic synchronization changes
only timing observability and performs no duplicate mathematical pullbacks.
