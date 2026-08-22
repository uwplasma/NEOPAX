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

## Current priority: prepared-support transpose only (2026-08-22)

The record-mode diagnostic resolves the previous ambiguity. For the current
best configuration and the final four-step segment:

| Isolated diagnostic boundary | Warm time |
|---|---:|
| Minimal replay producing segment primal records | `68.888 s` |
| Record-consuming reverse with rebuild transposes disabled | `1.287 s` |
| Rebuild state transpose | `1.675 s` |
| Rebuild prepared-support transpose | `56.956 s` |

The four-step record has `42,194,372` logical bytes (`40.240 MiB`), or about
`10.06 MiB` per slot. It is bounded by segment length, but retaining it for
all 1000 accepted steps would be about `10 GiB`; a full tape remains out of
scope.

Consequences:

- The record mode already removed the duplicate accepted-step reconstruction.
  The remaining record-consuming non-rebuild reverse is only `1.287 s`.
- The required once-per-segment minimal replay remains `68.888 s`; eliminating
  it requires full-tape-like retention and is not acceptable.
- Joining the state and support routes can save at most the `1.675 s` state
  transpose, while prior joint modes materially worsened compilation. It is
  not a priority.
- The only measured warm rebuild target is the `56.956 s` prepared-support
  transpose. This is device work, not host work or geometry construction.

### Plan before any further NTX implementation

1. **Trace the current generic support transpose exactly.** Map the NEOPAX
   `_response_from_support_delta` VJP to its NTX prepared-system factorisation,
   transpose solves, tangent/JVP work, and objective-batch placement.
2. **Compare against already rejected NTX routes.** In particular, compare the
   current generic VJP with
   `solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_and_aux`
   and the factorized two-direction mode. Determine the concrete operations
   that made those routes slower; do not rerun either unchanged.
3. **Design one new isolated helper only if it removes a proven operation.** It
   must keep the existing scalar objective lane and outer device `vmap`, use
   bounded memory, avoid extra primal response evaluation, and not widen the
   compiled graph with an objective axis.
4. **Prove the candidate with small in-memory CPU mocks.** Check exact bars,
   call count, arbitrary objective counts, no host loop/serial scan, and no
   effect on `separate_reuse_local_vjp_primal`.
5. **Only then wire an opt-in NEOPAX selector and provide one cache-disabled
   GPU benchmark command.** Reject it if first-segment compilation or warm
   support-transpose timing regresses.

### Step 1 result: current generic support transpose graph

The current-best selector `separate_reuse_local_vjp_primal` does **not** use
NTX's prepared implicit-adjoint API for its prepared-support bars. For one
anchor and one existing scalar objective lane it builds:

1. `_response_from_support_delta(prepared_delta, drds_delta)`, which applies
   the deltas to the local prepared system and evaluates all species.
2. For each species, `_build_interpolated_moment_response_local`, whose four
   outputs are: `log_nu_star`, base transport moments, `dtransport/dEr`, and
   `dtransport/dlog(nu*)`.
3. The base moments perform one energy scan of raw
   `ntx.solve_prepared_coefficient_vector`; each derivative field is a
   separate JVP through that scan. Thus the local primal contains a base NTX
   solve plus two differentiated solve paths.
4. `jax.vjp` then transposes that entire three-path graph with respect to the
   complete prepared tree and `drds`.

`_ntx_prepared_coefficient_vector_solver` returns the raw differentiable
`ntx.solve_prepared_coefficient_vector`, not
`solve_prepared_coefficient_vector_vjp`. Therefore this generic prepared
support VJP does not reach NTX's explicit prepared-system implicit-adjoint
rule; it differentiates the raw solver/JVP graph. This explains the measured
split: about `18 s` local primal plus about `39 s` additional transpose work,
for the `56.956 s` support transpose.

The outer objective `jax.vmap` is in NEOPAX around this scalar pullback. There
is no host loop or serial objective scan in the current best mode.

### Step 2 result: why the previous specialised support helper regressed

The prior opt-in
`solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_and_aux`
does reuse one *primal factorisation* per energy, but it does not make the
prepared-support derivative a single adjoint operation. Its exact algebra is:

1. base coefficient pullback: two factorised adjoint fields (`lambda1`,
   `lambda3`) plus the complete prepared gradient;
2. first directional field (`d/dEr`): forward directional fields, two
   directional adjoint fields, and base/directional prepared gradients;
3. second directional field (`d/dlog(nu*)`): the same forward and directional
   adjoint work again.

The response contains all three coefficient paths (base, `d/dEr`, and
`d/dlog(nu*)`). Prepared support bars for the latter two are derivatives of
the adjoint equation itself, so their directional adjoint fields are required;
they cannot be reconstructed from the state/case bars returned by the state
route.

Thus a joint state+support helper can share the base primal factorisation and
base adjoint fields, but it cannot remove the two directional prepared-support
adjoints. This is why the already-tested support-only/factorized helpers can
have a larger graph and worse compilation even though they eliminate generic
raw-solver AD. Do not retry them unchanged.

The measured state route is only `1.675 s`, so sharing its base adjoint with
support cannot explain or remove the `56.956 s` support cost. The next helper
must remove a specific operation *inside* those directional prepared-support
adjoints, not merely join the state and support APIs.

### Step 3 source passes: candidate boundary and rejection criteria

Three further source passes establish the following before any implementation.

1. **Exact directional control flow.** The support-only helper calls
   `_scan_direction_pullbacks`, which uses a two-element `lax.scan` over the
   fixed `d/dEr` and `d/dlog(nu*)` directions.  Inside each scan lane it
   separately computes the forward directional fields and two directional
   transpose fields (`lambda1_dot`, `lambda3_dot`).  The current
   `packed_support_directional_adjoint` condition is below the early
   `support_only` return and is therefore not used by this helper.
2. **Existing packed mode is not the same experiment.** The existing packed
   NTX API is selected only by the joint state-plus-support NEOPAX rebuild
   modes. It uses `include_prepared=True`, `support_only=False`, and carries
   the larger joint state/support result.  That is the previously rejected
   packed joint route; it does not test packing the two directions inside the
   scalar support-only helper.
3. **RHS shape and scope.** NTX already has a paired directional-adjoint
   primitive. It concatenates two field RHS along a fixed trailing RHS axis,
   calls the factorized transpose solve once, and splits the result. The
   proposed narrow experiment would use that fixed width-two axis only for
   the two physical derivative directions. It would not add an objective
   axis, alter the existing outer objective `vmap`, add a host loop, or scale
   its internal RHS width with objective count. It would, however, make
   temporary directional-adjoint arrays two columns wide, so a small bounded
   memory and compilation increase remains possible.

The initially proposed **support-only packed-two-direction helper is rejected
before implementation.** The code-level check shows that the existing paired
primitive packs the two *field* adjoints (`lambda1`, `lambda3`) for one
direction; it does not pack the two physical directions.  The two directions
have different coefficient cotangents, base adjoints, and directional RHS, so
an exact all-at-once variant needs four RHS columns, not two.  Passing the
existing packed flag through the support-only branch would merely repeat the
already-tested field-packing experiment in a different wrapper, while packing
all four RHS would widen the critical compiled graph.  Neither removes an NTX
solve or a directional adjoint equation. Do not implement either route.

The next implementation step must therefore be chosen only after identifying
a route that removes work rather than rearranging the same four exact adjoint
RHS. The required mock test for this rejected candidate is unnecessary.

### Further feasibility audit: factor retention and compile specialization

Two further paths were checked before selecting another implementation.

1. **Retaining NTX factors from the segment replay is not safe.** The NTX
   factorization stores LU, lower, and upper dense block bands.  For this
   benchmark a block is `5 * 21 = 105` square and there are `33 + 1 = 34`
   blocks. In float64 this is about `8.6 MiB` for one energy/species/anchor
   factor payload before coefficient and auxiliary arrays. With four energies,
   three species, and 48 anchors, this is about `4.8 GiB` per accepted step.
   Retaining it even for a four-step segment is an OOM risk. It is rejected.
   The existing segment primal record intentionally stores only about
   `10.06 MiB/step`, not these NTX factors.
2. **The remaining bounded-memory compile target is known branch
   specialization.** The production segment kernel receives the realized
   `lagged_response_valid` schedule as array data, then every active slot
   executes `lax.cond(valid, reuse_branch, rebuild_branch)`. XLA must compile
   both branches for every slot even when the realised segment pattern is
   already known (for example `BBBB` or `RRBB`). The source has static
   `force_reuse_bwd` and `force_rebuild_bwd` variants, but no per-slot static
   pattern dispatcher.

The compile-specific low-memory direction, kept separate from the support-rule
work, is an opt-in **static lagged-pattern segment dispatcher**. It would select one of the finite
`2**segment_length` exact segment kernels from the already-realized schedule,
so each compiled kernel contains only the needed reuse/rebuild branch at each
slot. The state/support equations, objective device batch, checkpoint record,
and NTX calls on the realised path stay unchanged. It targets the very large
segment HLO/compile cost, not the irreducible warm `~57 s` support transpose
for every actual rebuild. It must be assessed with a tiny in-memory JAX
structure test before it is wired because the finite set of pattern-specific
executables can trade one huge compile for several smaller first-use compiles.

### Support custom-rule audit after rejecting packed directions

The support-rule investigation remains the active work; branch specialization
is not its replacement.

The existing support-only NTX helper has one identifiable but limited
NEOPAX-side inefficiency. For every energy it builds three tiny generic VJPs
of the explicit six-moment prefactor (base, `d/dEr`, and `d/dlog(nu*)`) and
two JVPs of its direct-`drds` bar. The coefficient part already has an
analytic pullback helper. The direct-`drds` formula is also explicit:
away from the D11 floor it is the weighted sum of
`-2 * coefficient[0] * drds` and `-coefficient[2]`, with the same D11 active
mask as the established coefficient pullback.

This can be made into an isolated analytic-moment variant and exactly tested,
but it does **not** remove an NTX factorization, solve, or prepared-support
directional adjoint. It is therefore expected to reduce a small nested-AD/
compile subgraph only, not the measured `~57 s` warm support transpose. It is
not a credible standalone solution to the reverse timing target.

The expensive, irreducible exact work in the current support rule is instead:

1. the base prepared-system adjoint for the transport moments; and
2. one mixed prepared/case adjoint for each of `d/dEr` and `d/dlog(nu*)`.

Those mixed adjoints are mathematically required because the requested output
contains derivatives of the NTX response with respect to the case while its
prepared system varies. The existing factorized and support-only implicit
helpers already replace the generic raw AD with this explicit algebra. Their
larger compiled graph is why they regressed. There is currently no verified
custom-rule modification that removes any of these three NTX operations while
preserving exactness and without retaining the multi-gigabyte factor payload.

### New isolated candidate: fixed-grid prepared support VJP

The generic current-best support VJP is broader than the actual runtime
parameter dependency. `PreparedMonoenergeticSystem` contains `surface`,
`grid`, `geometry`, `d_theta`, and `d_zeta`. In the runtime-support builder,
`d_theta` and `d_zeta` are rebuilt only from fixed `GridSpec`; they do not
depend on VMEC/Boozer geometry parameters. The generic prepared-tree VJP still
receives them as differentiable inputs, so it forms dense matrix cotangents
whose upstream tangent is exactly zero.

`GeometryOnGrid` contains the runtime-dependent sampled operator quantities.
The new NEOPAX selector takes only `(GeometryOnGrid, drds)` as VJP inputs and
keeps the source surface, fixed grid, `d_theta`, and `d_zeta` in the closure.
It returns the usual prepared-tree bar with zeros at those fixed leaves, so
the outer support contract and objective device batch are unchanged.

```text
--reverse-rebuild-support-pullback-mode \
  separate_reuse_local_vjp_primal_geometry_only_prepared
```

This is opt-in only. It adds no NTX solve, factor record, host work, or
objective serialisation. It targets the dense fixed-operator transpose and
its graph size; it does not claim to remove the required NTX response work.
A small NTX equivalence test compares the grouped geometry-only rule with a
full VJP holding other prepared leaves fixed. The selector must be compared
with the cache-disabled current-best benchmark before becoming preferred.
