# Reverse AD timing work: current state and plan

## Scope and non-negotiable constraints

- Preserve the exact transport/reverse mathematics.
- Do not change timestep selection, accepted-step/rebuild criteria, resolution, or objective definitions.
- Do not add a full accepted-step tape or unbounded checkpoint memory.
- Do not move reverse work to the host or serialise the objective batch.
- Every experimental route is an explicit opt-in. The established best path stays unchanged.
- No GPU benchmark is requested until a small mocked structural/equivalence test supports the hypothesis. Tests must not write profiles, XLA dumps, or other temporary output.

## Current best exact bounded-memory path (2026-08-22)

The benchmark run below is now the current best measured exact transport-reverse
configuration. It keeps only the active four-step segment's primal records;
it is not a full tape and does not alter the accepted schedule, equations, or
objective batch.

```text
--reverse-stage-adjoint-solve-mode block
--reverse-rhs-transpose-mode explicit_ntx_interpolated
--reverse-step-bwd-mode reduced_cotangent_call_boundary
--reverse-initial-cache-support-pullback-mode ntx_batched_interpolated_faces
--reverse-rebuild-support-pullback-mode separate_reuse_local_vjp_primal
--reverse-segment-start-replay-mode minimal
--reverse-segment-primal-record-mode reuse_segment_primal_record
--ntx-exact-derivative-pullback-algebra ntx_helper_lowdot_fused
--reverse-schedule-artifact-mode reuse_static_probe
```

Measured 16-step, cache-disabled transport-reverse timings:

| Phase | Time |
|---|---:|
| Segment 4/4 (four rebuilds; compile plus execution) | `590.420 s` |
| Segment 3/4 (three reuse, one rebuild) | `77.001 s` |
| Segment 2/4 (one reuse, three rebuilds) | `228.237 s` |
| Segment 1/4 (one reuse, three rebuilds) | `227.755 s` |
| Segmented cotangent sweep total | `1123.413 s` |

For comparison, the former reconstructing best path reported `838.014 s` for
the analogous first rebuild-heavy segment. The record mode therefore removes
an observed `247.594 s` from that phase. The returned transport derivatives
match the saved reverse table to ordinary floating-point roundoff.

The later `jit_scan` slow-compile alarm in that run is **unattributed**. Its
terminal position is not evidence that it belongs to the later geometry phase:
compiler stderr can be delayed relative to progress stdout. Do not assign it
to transport or geometry until it is explicitly named.

## Former best exact path

Before enabling the segment-local primal record, the current empirical baseline
was the configuration below. It produced the approximately **2m11** first
reverse-segment compilation reported in the benchmark runs.

```text
--reverse-stage-adjoint-solve-mode block
--reverse-rhs-transpose-mode explicit_ntx_interpolated
--reverse-step-bwd-mode reduced_cotangent_call_boundary
--reverse-initial-cache-support-pullback-mode ntx_batched_interpolated_faces
--reverse-rebuild-support-pullback-mode separate_reuse_local_vjp_primal
--reverse-segment-start-replay-mode minimal
--ntx-exact-derivative-pullback-algebra ntx_helper_lowdot_fused
--reverse-schedule-artifact-mode reuse_static_probe
```

Important: this is the best *measured* path, not a claim that it contains no remaining duplicated work.

## Measured timings used for prioritisation

From the component diagnostic of the established path (four-step segment, all-objective benchmark):

| Component | Warm execution time |
|---|---:|
| Segment reverse excluding rebuild transposes | ~137 s |
| State transpose diagnostic | ~1.7 s |
| Rebuild support transpose | ~56 s |
| Local NTX VJP primal-only diagnostic | ~18 s |
| Local NTX VJP transport-only diagnostic | ~47 s |

The component measurements overlap and must not be added. They identify local NTX rebuild work as the main remaining rebuild target. Other observed fixed costs include the realized-schedule VJP forward (~220 s) and final-objective cotangent construction (~80 s).

## Findings from code inspection

### Established separate rebuild path

For each rebuild cotangent, the current best path performs two local NTX reverse routes:

1. `pullback_build_lagged_response` computes state bars through
   `_pullback_interpolated_moment_response_local_fields`.
2. `pullback_build_lagged_response_support_payload` computes prepared-support/drds bars through a separate local `jax.vjp`.

Both traverse the same anchors and use the same local NTX quantities. The support route correctly reuses its local-VJP primal for interpolation-coordinate work; this is why `separate_reuse_local_vjp_primal` is the current best mode.

### Rejected experimental joint mode

`ntx_joint_implicit_interpolated_faces_reuse_local_vjp_primal` is isolated and must not be used as the baseline.

Its implementation mistakenly returned the objective-independent local primal response from inside an objective `vmap`, then selected one lane. This widened the segment graph with redundant objective lanes and was consistent with the observed ~3m58 compilation. Correcting that mistake is necessary if the experiment is ever revisited, but it is **not** the current timing plan and is not expected to solve the 2m11 baseline cost by itself.

### Previously tested routes not to repeat unchanged

- BiCGSTAB, Woodbury/analytical Woodbury, compact/explicit-NTX-Jacobian and colored-midpoint stage variants: slower than block.
- Existing `ntx_batched_interpolated_faces`: NEOPAX-level batching only; no measured gain.
- Existing joint batched state/support variants, packed-support variants, support-only implicit variants, and factorized-two-directional variants: worse compilation and/or execution.
- Full tapes/checkpoint growth, host processing, serial objective scans, and changing numerical settings are out of scope.

## Scalar-joint local rebuild: investigation result

Do **not** batch a new joint helper over objectives internally.

The initially proposed scalar helper would, for one existing objective lane and one anchor:

1. consumes the existing scalar interpolated-response cotangent;
2. calls NTX's existing `solve_prepared_coefficient_vector_lowdot_two_pullbacks_with_prepared_and_aux` once;
3. returns both state bars (`Er`, `T`, `n`) and prepared-support/drds bars from that one local factorization;
4. reuse the established local primal only for the interpolation-coordinate transpose, without returning it through an objective batch.

The existing outer objective `jax.vmap` would remain unchanged. Therefore the route could support any number of objectives/RHS columns without host work or a sequential objective loop.

**This route is currently blocked and must not be wired.** The scalar local helper's
`return_primal_response=True` branch calls
`_interpolated_moment_reduced_local_outputs_from_primitives` after the joint
lowdot pullback. That is a new local NTX response evaluation; it is not a
free use of the accepted-step/replay primal. Therefore the proposed scalar
joint route would still add duplicated NTX work for interpolation-coordinate
transpose and could regress compilation/execution exactly as the rejected
batched joint mode did.

To make a scalar joint route valid, the reverse step would need the actual
anchor-local primal response from its already reconstructed forward step.
The current lagged-response payload retains interpolated target values, not
those anchor values. Supplying them would require a new segment-bounded
primal record, with its own memory audit. Do not implement this unless that
record can be shown bounded and materially smaller than the current duplicated
NTX work.

## Required mocked tests before a GPU benchmark

Use tiny mocked NTX/model objects and in-memory JAX arrays only. No XLA/profile/output dumping.

1. **Call-count structure:** for one anchor and any objective count (1, 2, 10, 20), one scalar lane invokes the mocked lowdot helper once and returns both categories of bars.
2. **Batch structure:** the existing outer `jax.vmap` is retained; no `lax.map`, host loop, or inner objective `vmap` contains the primal response.
3. **Exact mock equivalence:** scalar-joint state/support bars equal the sum of the current separate mock state and support routes.
4. **Shape/carry check:** the primal response has no leading objective axis and is not added to an accepted-step payload.
5. **Baseline isolation:** dispatch for `separate_reuse_local_vjp_primal` is unchanged.

Only if all five pass should a new named opt-in selector be wired. Then run one cache-disabled benchmark against the baseline above and compare both first-segment compilation and warm rebuild/segment timing. Reject the selector if either regresses materially.
