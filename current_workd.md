# Reverse AD timing work: current state and plan

## Current next work: local NTX multi-RHS shared-adjoint investigation (2026-08-23)

### Objective

For one rebuilt NTX anchor, avoid independently constructing the local
implicit-solve context for the state (`nu_hat`/`epsi_hat`) and prepared-support
geometry pullbacks. The target is one unbatched local NTX factorisation and
primal mode solve, with all objective cotangents handled as device-resident
right-hand-side columns. This must remain exact and bounded to the active
anchor/rebuild invocation: nothing is retained across accepted steps or
segments.

### Constraints

- `separate_reuse_local_vjp_primal` remains byte-for-byte dispatch-equivalent
  unless a new selector is explicitly chosen.
- No full factor tape, host callback, host objective loop, `lax.map`, or serial
  objective scan.
- The number of objectives is an input dimension, not a fixed benchmark
  assumption; the implementation must support any positive RHS-column count.
- Do not repeat the rejected NEOPAX joint state/support wrapper, which carried
  objective-batched support pytrees through the anchor scan and enlarged the
  segment compilation.

### Implementation plan

1. Add in-memory NTX-only tests on a tiny prepared system. They compare a
   candidate multi-RHS local implicit pullback against `vmap` of the existing
   scalar helper for one, two, and several RHS columns, including all returned
   state and prepared-support bars. The test also inspects the traced program
   to ensure factorisation inputs are unbatched and no host/serial objective
   path is introduced.
2. Only if that gate passes, add a new private NTX multi-RHS helper. It must
   factorise once, solve the primal and the two physical derivative directions
   once, and use batched linear-system RHS only for cotangent-dependent
   transpose solves.
3. Add a private NEOPAX per-anchor adapter that consumes this helper and emits
   the existing state and support accumulation inputs. It must not return the
   objective-independent primal response through an objective `vmap` and must
   not replace the current best dispatch.
4. Add exact small-model NEOPAX equivalence tests and only then expose a new
   opt-in rebuild-support selector. A cache-disabled GPU benchmark is permitted
   only after those tests pass; reject the selector if either compilation or
   warm support-transpose time regresses.

### Native matrix-RHS prerequisite (step 1, implemented; unselected)

The former ``multi_rhs_shared_primal`` helper was not a native matrix-RHS
implementation: after sharing the primal residual, it used ``jax.vmap`` to
re-enter the complete scalar low-dot support pullback once per objective.
That selector is rejected and remains unchanged.

Step 1 of the replacement is now an NTX-private factorized primitive,
``_solve_factorized_multi_rhs_directional_adjoint_field_pair``.  It accepts
field adjoints with axes ``(mode, unknown, rhs, physical_direction)``, packs
only ``rhs * physical_direction`` as trailing columns of the one
block-tridiagonal transpose solve, and restores the axes immediately.  It
contains no objective loop, ``lax.map``, or local factor retention.  A tiny
prepared-system test compares it with the explicit packed-column reference;
the test has been added but is intentionally not run in the local workspace.

The next implementation step is to route the base and directional prepared
support adjoint fields through this primitive before invoking the final
prepared-gradient contractions.  No NEOPAX dispatch or CLI selector has been
changed by this prerequisite.

### Native field construction (step 2, implemented; unselected)

NTX now has the separate helper
``solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_native_multi_rhs_and_aux``.
It performs one primal/factorized local solve, two objective-independent
physical forward directions, and packs the following objective-dependent
transpose fields as matrix RHS columns:

- the base coefficient adjoints;
- the two directional base adjoints; and
- the two directional ``lambda_dot`` adjoints.

Only after those factorized solves does it ``vmap`` the final exact
prepared-gradient algebra over the RHS axis.  It never vmaps or scans the
scalar implicit low-dot helper.  The helper is exported by NTX only; there is
no NEOPAX adapter, selector, CLI choice, or benchmark dispatch yet.

Two tiny prepared-system pytest gates have been added: one validates the raw
packed directional transpose against explicitly packed columns, and one
compares all five returned prepared-support trees against the established
scalar helper for 1, 2, and 4 RHS columns.  They must be run on the remote
NTX environment before this helper is wired into NEOPAX.

Remote NTX gate result (2026-08-23): **passed**.  The packed-column and native
support tests completed as ``4 passed`` (1, 2, and 4 RHS support comparisons).

### Private NEOPAX adapter (step 3, implemented; unselected)

The existing rejected multi-RHS adapter now has an explicit private companion,
``_pullback_interpolated_moment_prepared_support_and_drds_only_native_multi_rhs``.
It calls only the new native NTX helper.  The established adapter retains its
original helper and default argument; reverse dispatch, physics contexts, and
CLI validation do not reference the companion.

The existing one-energy local test now also compares this native adapter with
the scalar support-only adapter for prepared bars, ``drds`` bars, and the
unbatched primal interpolation response.  This remains a local tiny-model
test: no transport rollout, persistent cache, profile, or output path is
involved.  Only after both NTX and NEOPAX exact gates pass may an opt-in
rebuild-support selector be considered.

### Important feasibility gate

The prior source audit shows that merely joining NEOPAX's current state and
support wrappers can save at most the already-small state transpose. The
expensive prepared-support path additionally differentiates the two response
derivative fields. Therefore this work proceeds only if the NTX multi-RHS
helper removes a proven repeated local factorisation/transpose operation. If
the tiny traced test shows that JAX already shares it, no selector will be
implemented and this direction will be recorded as rejected.

### Status: NTX helper and private NEOPAX adapter implemented; not selectable

The first implementation is now in NTX only:
`solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_multi_rhs_and_aux`.
It calls the existing low-dot core once with `return_primal_residuals=True`,
then vmaps only the cotangent-dependent support-adjoint portion while passing
the same unbatched local primal/factor residual to every RHS. It returns the
same five prepared-support trees and auxiliary output, each with a leading RHS
axis. The residual is local to the call and is not returned.

An in-memory tiny-grid test compares it with the established scalar
support-only helper for 1, 2, and 4 RHS columns and passes. It uses no profile,
XLA dump, cache, or output directory. The helper is exported by NTX but no
NEOPAX selector calls it yet; the established best path is unchanged.

### Three-pass adapter audit (2026-08-23)

1. The NTX helper has the required bounded lifetime: its `primal_residuals`
   include the local modes/factors and directional primal modes, but are
   created before the RHS `vmap`, closed over only by that local call, and are
   neither returned nor stored in a segment payload. The exact support adjoint
   remains per RHS, which is necessary because each objective supplies a
   different cotangent.
2. The existing NEOPAX `ntx_batched_interpolated_faces` route cannot simply be
   reused: it builds a generic raw-solver `jax.vjp` and batches that pullback.
   A new private adapter is required to construct the three coefficient-bar
   batches from interpolated response bars and call the new NTX helper at each
   anchor. It will carry only the already-required batched support bars through
   the anchor scan.
3. The prior `separate_reuse_local_vjp_primal_factorized_ntx_two_directional`
   mode is not equivalent to this helper. It changes the response forward
   primitive to a factorized three-output custom-VJP, then differentiates it
   generically. It does not supply one `primal_residuals` object to all support
   RHS adjoints. The new helper therefore removes a concrete repeated local
   primal/factorisation operation that that earlier selector did not remove.

The private NEOPAX adapter is now present as
`_pullback_interpolated_moment_prepared_support_and_drds_only_multi_rhs`, and
the corresponding unselected face-support method is
`pullback_build_lagged_response_support_payload_batched_interpolated_faces_multi_rhs_shared_primal`.
For one anchor/species it passes the complete objective RHS batch to the NTX
helper, receives an unbatched primal response for interpolation, and returns
only objective-batched prepared/``drds`` bars. It does not alter any existing
selector, CLI option, or dispatch.

The remaining gates are exact equivalence and compilation risk. The adapter
must be compared on an in-memory small model with the current scalar support
pullback. A cache-disabled GPU benchmark and any CLI selector remain forbidden
until that comparison passes; reject the selector if either compilation or
warm support-transpose time regresses.

The exact small-model NEOPAX test has passed on the remote machine. It uses one
energy and NTX's existing ``5 x 5 x 4`` tiny prepared system; it compared the
multi-RHS adapter with the scalar helper for two RHS columns and did not
execute a transport rollout, profile, cache, or output writer.

The resulting selector is now available only as
``ntx_batched_interpolated_faces_multi_rhs_shared_primal``. It creates no
fallback and does not modify the current best mode. It is structurally wired
through the dedicated batched support hook but remains unbenchmarked; the
first cache-disabled benchmark must compare objective/gradient values before
timings are interpreted.

### Benchmark result: multi-RHS shared-primal selector rejected (2026-08-23)

The cache-disabled 16-step benchmark completed its segmented sweep with this
new selector and regressed relative to the established record-mode best:

| Segment | New multi-RHS selector | Established record-mode best |
|---|---:|---:|
| `4/4` (four rebuilds; compile plus execution) | `869.372 s` | `590.420 s` |
| `3/4` (one rebuild) | `75.270 s` | `77.001 s` |
| `2/4` (three rebuilds) | `222.447 s` | `228.237 s` |
| `1/4` (three rebuilds) | `222.942 s` | `227.755 s` |
| Sweep | `1390.039 s` | `1123.413 s` |

The first-segment slow-compile alarm was about `4m11s`, versus the former
roughly `2m11s` record-mode baseline. The modest warm improvements in the
later segments do not offset this regression.

Source review explains why: the helper creates one local primal/factor
residual, but then uses ``jax.vmap(_one_rhs)``. Each lane re-enters the
scalar low-dot core and still forms the four exact cotangent-dependent
adjoint field families (`lambda1`, `lambda3`, `lambda1_dot`, and
`lambda3_dot`). It is therefore not the promised single matrix-RHS adjoint
solve; it only shares the objective-independent primal setup. The wider
mapped graph accounts for the compile regression. This selector is rejected
for timing and must not replace the current best mode.

## Compact coefficient-record implementation status (2026-08-22)

The first internal portion is implemented but **not selectable** by the
benchmark, so it cannot affect the established best path. An opt-in NTX
companion builder returns the normal lagged response and a compact record of
only the base, `d/dEr`, and `d/dlog(nu*)` coefficient vectors. A separate
reverse-minimal Radau replay helper invokes that builder only at the ordinary
lagged-response rebuild point; reuse and inactive slots return zero records.

This uses a new private segment-record pytree, leaving the current
`reuse_segment_primal_record` signature, storage, and dispatch unchanged. The
in-memory checks cover local response/record equality, full small-model
ordinary-versus-recorded lagged-response equality, and the Radau
rebuild/reuse paired-result contract. All pass in WSL `mygpuenv`, without
profiles, XLA dumps, or temporary output. The next step is a record-consuming
support-transpose helper; there is intentionally no CLI selector or GPU run
until that consumer has an equivalence test.

### Consumer feasibility audit

The compact coefficient arrays alone cannot power an exact prepared-support
transpose. NTX's support-only implicit rule needs the full primal mode vectors
and factorized block system (`f1_full`, `f3_full`, LU/pivots, lower, and upper)
to form the three base/directional adjoints. The saved five-value coefficient
vectors cannot reconstruct those quantities. Rebuilding them is exactly the
current support-transpose work, while retaining them across a segment is the
previously rejected multi-GiB factor-tape approach. Consequently, no consumer
or selector will be added for this record: doing so would either duplicate the
current NTX work or violate the bounded-memory requirement. The disconnected
record implementation remains isolated while the next timing direction is
selected from a different proven operation.

## Native matrix-RHS exactness correction (2026-08-23; unselected)

The native matrix-RHS path was fast in later segments but did not reproduce
the geometry/support derivatives.  A small local discriminator isolated the
failure to the ``dtransport_moments_d_er`` channel, including the scalar
prepared-support low-dot oracle.  The missing quantity is the directional
coefficient cotangent:

``d/dcase [ pullback(moment_from_coefficients, moment_bar) ]``.

The low-dot implementation differentiated the NTX primal and adjoint fields
but held that cotangent constant.  The native path consequently omitted the
mixed moment/NTX term; it was not a native-RHS packing error.

The correction is intentionally limited to the unselected native adapter:

1. NEOPAX forms the two directional coefficient-cotangent batches alongside
   the already-existing directional ``drds`` batches.
2. NTX accepts these as an optional fifth callback result and carries them
   through the directional coefficient pullback and prepared-gradient JVP.
3. Existing four-item callbacks receive zero tangents, so
   ``separate_reuse_local_vjp_primal`` retains its prior callback path and
   dispatch.

This adds algebraic tangent contractions only; it adds no NTX primal solve,
factorisation, host work, objective loop, checkpoint record, or persistent
output.  Required next gate: the CPU-only in-memory native-versus-actual-local
VJP test for one and two energies, followed by the remote GPU benchmark only
if exactness passes.

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

### Follow-up candidate: geometry-only *implicit* support pullback

The fixed-grid generic-VJP selector above was measured and rejected: it did
not improve the current-best path.  It remains isolated and is not the next
candidate.

There is a distinct, untested route already available in NTX:
`solve_prepared_coefficient_vector_lowdot_two_pullbacks_with_geometry_and_aux`.
It uses NTX's explicit grouped implicit-adjoint algebra, but returns only
`GeometryOnGrid` bars rather than complete `PreparedMonoenergeticSystem` bars.
For the runtime support builder this is sufficient: the remaining prepared
leaves (`surface`, `grid`, `d_theta`, and `d_zeta`) are fixed by the configured
grid and have zero runtime-geometry cotangent.  NEOPAX can reconstruct the
usual prepared-tree bar with those leaves set to zero.

This differs from the rejected `prepared_support_only` and factorized modes:
those called the same grouped core with `include_prepared=True`, forcing
complete prepared and directional-prepared gradient construction.  The
geometry helper takes the `include_geometry=True` path, using
`_geometry_gradient_from_adjoint` and
`_directional_geometry_gradient_from_adjoint` instead.  It retains the exact
base, `d/dEr`, and `d/dlog(nu*)` implicit adjoints—so it does not promise to
remove those mathematically required solves—but avoids building cotangents for
the fixed prepared operator leaves.

An in-memory NTX test now proves that this helper's five geometry bars are the
exact geometry projection of the full prepared helper, and that its returned
primal coefficient/tangent auxiliary is identical.  No production NEOPAX
selector has been added yet.

If implemented, it must be an opt-in mode (suggested name:
`separate_reuse_local_vjp_primal_geometry_implicit_ntx_two_directional`) and
must preserve the existing scalar objective lane and outer device `vmap`.  It
must first receive a NEOPAX mock/full-output equivalence test, then one
cache-disabled benchmark.  Reject it if it regresses first-segment compilation
or warm support-transpose timing relative to
`separate_reuse_local_vjp_primal`.

#### Implementation plan for the geometry-only implicit candidate

1. **Keep the baseline dispatch unchanged.** Add a new rebuild selector only;
   `separate_reuse_local_vjp_primal` remains byte-for-byte on its existing
   generic-VJP branch.  Validate that the new selector requires the current
   interpolated face response, no centre-response cotangent, and
   `reuse_local_vjp_primal_anchor_response=True`, exactly as the existing
   support-only implicit branch does.
2. **Add a geometry-only support adapter in NEOPAX.** Mirror
   `_pullback_interpolated_moment_prepared_support_and_drds_only`, retaining
   its coefficient bars, explicit direct-`drds` terms, energy `lax.map`, and
   returned primal response.  Replace only the NTX call with
   `solve_prepared_coefficient_vector_lowdot_two_pullbacks_with_geometry_and_aux`.
   Sum its base/first/second geometry bars and construct the required prepared
   bar as `zero_like(prepared)` with that summed value in `.geometry`.
3. **Preserve device batching.** The geometry adapter returns numeric leaves
   before the existing species `vmap`, using the current sanitisation/rebuild
   convention.  It does not introduce a Python loop, a `lax.map` over
   objectives, an objective RHS axis, or a retained factor/timestep payload.
4. **Add proof tests before exposing the CLI.** The completed NTX test proves
   geometry projection and auxiliary equality.  Add NEOPAX tests for (a) the
   reconstructed prepared bar having zero fixed leaves, (b) equality of the
   complete local response/pullback against the current full-prepared helper
   on a small real prepared NTX system, and (c) arbitrary batched objective
   leading dimensions without a shape or species-vmap change.
5. **Wire the selector and verify one benchmark.** Add it to static setup,
   benchmark choices, and progress output.  Run the current cache-disabled
   16-step record-mode command with only the new selector substituted.  Compare
   derivatives, first-segment compile time, warm support-transpose timing, and
   total sweep.  Reject the selector unless it improves the support transpose
   without a material compile regression.

#### Implementation and proof status

The initial use of NTX's public geometry helper was rejected during testing:
that helper returns case/profile bars as well as geometry bars, so it would
perform unused contractions.  Instead NTX now has a new isolated public helper
`solve_prepared_coefficient_vector_lowdot_two_pullbacks_geometry_support_only_and_aux`.
It invokes the existing grouped core with `include_geometry=True` and
`support_only=True`; it returns exactly the five geometry bars plus auxiliary.

While adding it, the exact comparison found that the existing support-only
branch selected `_directional_prepared_gradient_from_adjoint` unconditionally.
The branch now selects `_directional_geometry_gradient_from_adjoint` when
`include_geometry=True`.  This does not alter the established prepared-only
support helper.  Focused in-memory tests pass:

* NTX: three geometry/full and geometry-support-only equivalence tests;
* NEOPAX: the complete local support adapter agrees with the prepared-support
  path for active geometry bars, direct `drds` bar, and reused primal response.
  A second NEOPAX test applies the adapter to two objective cotangent lanes with
  `jax.vmap` after the production static-leaf sanitisation; it preserves the
  leading objective axis without a host or serial-objective path.

The new NEOPAX selector is wired and remains opt-in.  It was benchmarked in
the cache-disabled comparison recorded below.

#### Benchmark result: reject geometry-only implicit support mode

The cache-disabled 16-step record-mode benchmark was run with
`separate_reuse_local_vjp_primal_geometry_implicit_ntx_two_directional`.
It selected the intended mode, but regressed materially:

* first reverse-segment compilation alarm: about `3m12s`;
* final `4/4` segment: `776.249 s`;
* established record-mode best final `4/4` segment: about `590.420 s`.

The exactness and batching tests remain useful, but this selector is rejected
for performance.  Keep it opt-in for reference; do not use it as a baseline or
modify the current-best generic local-VJP path because of this experiment.

#### Planned opt-in experiment: compact segment coefficient records

This is distinct from the rejected implicit helper modes.  The established
generic support VJP reconstructs, for each local anchor, the NTX coefficient
primal and its two case-direction coefficient tangents before transposing
them with respect to support.  The ordinary segment replay computes those
values as solver/JVP intermediates while constructing its lagged response,
but the current response API discards them.  The experiment is to add an
opt-in local-response adapter that returns those existing intermediates in a
compact record containing only, for rebuilt anchors:

* the base coefficient vector;
* the `d/dEr` coefficient tangent; and
* the `d/dlog(nu*)` coefficient tangent.

For the present shape (48 anchors, 3 species, 4 energies, 5 coefficients),
these three float64 records are approximately 70 KiB per rebuilt accepted
step.  They are intentionally not NTX LU/factor payloads (about 4.8 GiB per
step in the earlier audit) and are retained only in the existing bounded
segment record, never over the complete accepted-step trajectory.

Implementation gates:

1. Audit the current replay/primal-record structures to prove the three
   arrays can be emitted from the already-executed local response path without
   a second NTX call and with a static, segment-bounded shape.
2. Add a new opt-in record mode.  The default and
   `separate_reuse_local_vjp_primal` must remain unchanged.
3. Add a reverse-only support-adjoint interface that consumes the recorded
   coefficient arrays, rebuilds only the required local operator/factorisation
   for support bars, and does not rerun the base plus two tangent forward
   solves.  It must retain the existing outer device objective `vmap` and add
   no host or serial-objective operation.
4. Prove exact local bars and complete Radau support bars against the current
   generic VJP with in-memory tests.  Test arbitrary objective-batch sizes.
5. Benchmark cache-disabled only after these checks pass.  The immediate
   measured target is the current local-primal component (~18 s per rebuild),
   not an unsupported promise to remove the whole ~57 s support transpose.

#### Step 1 audit result: record source and boundary

The existing `reuse_segment_primal_record` already retains a bounded
`_RadauAcceptedStepSegmentPrimalRecord` per slot and consumes it directly in
the segment reverse scan.  It currently stores the accepted trial/stage data,
Radau factors, and `NTXExactLijLaggedResponse`, but no NTX coefficient scan.
The response cache itself contains only interpolated moment values
(`log(nu*)`, moments, `d/dEr`, and `d/dlog(nu*)`), not anchor coefficient
vectors.

`_build_axis_lagged_response` calls
`_build_interpolated_moment_response_local` at each anchor.  That local
function evaluates one base coefficient scan and two JVP solve paths, but it
returns only four moment fields.  Therefore appending a record at the Radau
layer would require a second NTX evaluation and is rejected.

The valid record source is instead an opt-in variant of that local response
function: it must expose the base coefficient scan and the two coefficient
tangent scans *from the same base/JVP evaluations* used to form the ordinary
four moment fields.  The normal response builder and current-best reverse
mode remain unchanged.  The next design step is to prove this adapter can
preserve exact moment values and static output shape before wiring it into the
segment record.

#### Step 2 implementation: local compact-record adapter

`_interpolated_moment_reduced_local_outputs_with_coefficient_record_from_primitives`
now exists as a private, opt-in companion of the ordinary local response
builder.  It returns the unchanged four response fields plus
`_NTXInterpolatedMomentCoefficientRecord` containing the base, `d/dEr`, and
`d/dlog(nu*)` coefficient scans.  Its two tangents use the same derivative
mode override as the ordinary derivative-field routines; their moment fields
are then formed directly from the emitted coefficient tangents.

The ordinary `_build_interpolated_moment_response_local` path has not been
modified or redirected.  A focused in-memory test compares all four returned
moment fields to the ordinary local builder at `1e-9` relative tolerance over
two energy nodes, checks the three `(n_energy, 5)` record arrays, and traces
the record through `jax.eval_shape` to validate its registered static pytree
structure.  The adapter is not yet wired into `NTXExactLijLaggedResponse`,
the Radau segment record, or any benchmark selector.  Therefore it cannot
affect the current best benchmark.

#### Step 3a implementation: isolated response-plus-record builder

The NTX runtime model now has an opt-in
`build_lagged_response_with_compact_coefficient_record(state)` companion.  It
creates the ordinary interpolated face response and a separate private
`_NTXExactLijLaggedResponseCoefficientRecord` from the same local base/JVP
work.  The existing `NTXExactLijLaggedResponse` dataclass, normal
`build_lagged_response`, carry cache, and all active selectors remain
unchanged.

The companion is deliberately limited to `interpolate_from_faces` and the
interpolated response-anchor lane.  It does not pretend that direct full-radius
coefficient responses feed the current support transpose.  The new record is
not yet passed into Radau; the next required step is a dedicated, opt-in
segment-replay hook and a reverse support-adjoint consumer.  That hook must
keep the record bounded by segment length and must never add it to the normal
lagged-response carry.

During the post-implementation source review, the rho=0 anchor case required
one correction. The ordinary builder skips that local NTX solve and
regularizes only response fields from the next three anchors. The compact
record now contains an explicit zero placeholder at that slot rather than an
extrapolated coefficient vector. A later consumer must use the existing
interpolation transpose (which gives that raw axis slot zero cotangent); this
preserves the no-extra-solve invariant and prevents a nonphysical axis record
from entering the support adjoint.

### Current next work: make the prepared-lowdot joint path structurally small

#### Re-audit conclusion

The intended mathematical primitive already exists in NTX:
`solve_prepared_coefficient_vector_lowdot_two_pullbacks_with_prepared_and_aux`.
For one local NTX system and one response cotangent it returns both the usual
case/state bars and the prepared-support bars from the same primal
factorisation and grouped base/two-directional adjoint calculation.

The established best reverse mode does **not** use that joint primitive.  It
uses the case-only lowdot helper in the scalar state transpose and a separate
generic JAX VJP in the scalar support transpose.  This is the remaining local
duplication.

The previously benchmarked joint modes did use the joint primitive, but in a
costly layout.  At each anchor they apply the local joint helper under an
objective `vmap`, then carry the resulting objective-batched complete prepared
support pytree through the anchor `lax.scan`.  This widens the whole segment
HLO with every differentiable prepared leaf and with the objective axis.  The
cache-disabled regressions (roughly 3--4 minute first-segment compile and
worse total reverse timings) reject that *layout*, not the mathematical joint
lowdot rule.

#### Plan and gates

1. **Mock structural proof before production changes.** Create an in-memory,
   numeric-pytree mock that has the same nesting, objective axis, anchor scan,
   scatter accumulation, and output packing as the joint support route, but
   replaces NTX solves by small array algebra. Compare Jaxpr/abstract carry
   shapes for:
   - the rejected anchor-scan-with-objective-batched-support layout; and
   - a proposed scalar-per-RHS joint layout.
   This test must use no real NTX/transport solve, no GPU compile, no profile,
   and no filesystem output. If batching still places the objective axis on
   the scan carry, the proposed rewrite is rejected before production code.

2. **Only if the mock establishes a smaller carry, add an opt-in compact
   scalar joint adapter.** It will call the existing prepared-lowdot helper
   once per local scalar RHS and return numeric state bars plus only active
   prepared geometry/support leaves. It must not return validated NTX
   dataclasses through `vmap`, and it must not allocate/carry a full generic
   prepared delta tree for inactive/static leaves.

3. **Wire it at the Radau rebuild boundary without changing the best mode.**
   The new selector must replace the *pair* of rebuild state/support calls for
   that selector only. `separate_reuse_local_vjp_primal` remains unchanged.
   Objective batching stays on device; there is no host loop, serial objective
   scan, full accepted-path tape, or factorisation record.

4. **Exactness gates.** With small in-memory fixtures, compare the new joint
   adapter against the sum of the established state-lowdot and scalar generic
   support paths for: state bars, every active support leaf, direct `drds`,
   interpolation-coordinate bars, and arbitrary objective batch sizes.

5. **Benchmark gate.** Only after all tests pass, run one cache-disabled
   16-step benchmark on the remote machine. Reject the selector if either the
   first segment compilation or warm support-transpose timing regresses. The
   compact coefficient-record route remains a separate fallback: it can remove
   repeated forward directional solves from the current best support path, but
   it does not combine state and support into one prepared-lowdot call.

#### Step 1 result: scalar-RHS relocation alone is rejected

The new pure-array structural test
`test_joint_lowdot_scalar_rhs_layout_does_not_hide_rhs_axis_from_anchor_scan`
compares the rejected anchor-major objective-batched route with the tempting
rewrite that places `vmap` outside a scalar anchor scan.  It inspects only
JAX's abstract scan body shapes.  Both layouts have the same scan carries:
state `(n_rhs, 2)` and support `(n_rhs, 5)` in the mock.  This is expected:
JAX's batching rule pushes the outer `vmap` axis into `lax.scan`.

Therefore merely relocating the objective `vmap` cannot reduce compilation or
memory and must not be implemented as a production selector.  A viable joint
path has to change what is carried/accumulated, not merely where `vmap` is
written.  The next investigation must determine whether the prepared-lowdot
support contribution can be reduced directly into a compact runtime-geometry
cotangent, rather than carrying the complete prepared-support tree through
the anchor scan.

#### Step 2 result: do not pull prepared support through VMEC per rebuild

That proposed compaction is not valid for this path.  The final
`geometry_payload_pullback_from_param_vector_raw_block_transpose` deliberately
receives the *summed* NTX support-payload cotangent after the complete transport
reverse sweep.  It builds the payload-to-VMEC-state VJP once and then applies
the VMEC raw-block transpose once for all objective RHS.

The NTX prepared leaves are not merely fixed scaffolding: `surface`,
`geometry`, `d_theta`, and `d_zeta` are built by
`prepare_monoenergetic_system` from VMEC/Boozer surfaces.  Collapsing their
bars to a runtime-geometry cotangent inside each Radau rebuild would require
executing that VMEC/Boozer payload pullback inside every rebuilt step.  That
would duplicate the expensive geometry graph and defeats the current
one-final-pullback design.

The external payload pullback already filters inactive float leaves after the
sweep.  Repacking those same leaves earlier would not remove their floating
array volume or the objective axis from the segment scan.  Hence this route is
rejected without production changes.  The remaining viable exact optimisation
is to reduce the *local NTX work* before those necessary payload bars are
formed (for example the bounded compact coefficient-record consumer), not to
move the VMEC payload contraction into the segment.

#### Step 3 audit: what the local joint prepared-lowdot helper already shares

The local joint helper is not doing a second ordinary NTX primal or a second
base implicit adjoint. Its grouped core calls `_prepared_implicit_vjp_primal`
once, producing the base coefficient modes, factorisation, `f1_full`, and
`f3_full`. The base prepared support bar and ordinary state bars both reuse
the same `lambda1` and `lambda3` transpose solves.

The additional work is required by the response representation itself. The
interpolated response has a base moment field plus `d/dEr` and
`d/dlog(nu*)` fields. Exact prepared-support differentiation therefore needs
five distinct contractions: the base support bar; a base and directional bar
for `d/dEr`; and a base and directional bar for `d/dlog(nu*)`.

The directional terms require full directional NTX mode fields and directional
adjoints (`f1_dot`, `f3_dot`, `lambda1_dot`, and `lambda3_dot`). The fast
case-only lowdot route needs only low-order mode contractions for state bars,
so it does not construct those full fields. They cannot be made free merely
by placing geometry in the same call.

This explains the rejected joint modes: they did share the base factorisation,
but exposed necessary full directional geometry algebra and prepared bars in
an already large objective-batched segment graph. There is no source evidence
of an accidental second base NTX solve to remove in that helper.

#### Step 2 implementation: compact numeric prepared-support scan carry

The existing joint adapter now accepts the private static flag
`compact_prepared_support_carry`. It leaves the scalar joint lowdot call
unchanged: that call still returns state/case bars, direct `drds`, and all
prepared bars together. Only its anchor-scan accumulator changes.

Instead of carrying a pytree of objective-batched `face_prepared` arrays, the
compact branch carries one numeric array with layout
`(objective, face-radius, sum(local prepared-leaf sizes))`. Each anchor packs
the existing local prepared bars and applies one radius scatter; after the
scan, the exact original prepared pytree is reconstructed with its original
bar dtypes and shapes. It adds no tape, factor payload, host work, or serial
objective loop.

The new wrapper is exposed only through the opt-in selector
`ntx_joint_implicit_interpolated_faces_reuse_local_vjp_primal_compact_prepared_carry`.
`separate_reuse_local_vjp_primal` and all existing joint modes retain their
previous implementations.  The selector takes the joint Radau branch, so it
does not invoke either separate rebuild hook.

The dependency-free mock
`tests/test_joint_lowdot_compact_carry_mock.py` verifies that the packed
accumulation/unpacking is exactly equal to the original per-leaf scatter.  It
passed locally without an NTX or transport solve.  `py_compile` and `git diff
--check` also pass.  This establishes layout equivalence only; the remote
benchmark remains the gate for whether the reduced scan carry helps the XLA
compile or warm rebuild time.

#### Dispatch repair after the first remote attempt

The first compact-selector benchmark stopped before the reverse segment
compiled.  Its `RuntimeError` was caused by a missing forwarding method on
the outer `TransportEquationSystem`: the Radau vector field is bound to that
object, not directly to the flux model.  The compact method existed on the
flux-model layers but was therefore invisible to the hook resolver.  The
outer forwarding method is now present and forwards to the existing
local-primal-reuse implementation with the compact flag.  It does not alter
the default or the current best selector.

### Active plan: joint lowdot replaces the separate rebuild support transpose

This plan uses the already exact local primitive
`solve_prepared_coefficient_vector_lowdot_two_pullbacks_with_prepared_and_aux`.
It is **not** a new NTX differentiation rule and it does not change the
current best selector.

For a new opt-in rebuild selector, every rebuilt local NTX response must use
the joint primitive once and return both state/case bars and prepared
geometry/support bars.  That selector must not dispatch either
`pullback_build_lagged_response` followed by
`flat_rhs_build_support_pullback`, nor the generic `jax.vjp` support path.
The support bars are accumulated through the segment exactly as today and are
passed once, after the reverse sweep, to the existing payload/VMEC pullback.

1. **Dispatch proof.** Introduce a narrow joint-only selector and a pure
   mock/structural test that records the invoked rebuild hooks.  It must prove
   the selector calls the joint lowdot hook and never calls the separate
   state or support hooks.  The current best selector is a control and must
   remain byte-for-byte on its existing dispatch branch.

2. **Compact joint adapter.** Refactor only the NEOPAX adapter around the
   existing scalar joint lowdot call.  It may batch numeric leaves on device,
   but it must not return or carry static NTX dataclass metadata, invoke a
   host loop, or use a serial objective scan.  The adapter returns the same
   state bar and active float support leaves as the two established separate
   paths.  It must retain the axis-anchor zero rule and the existing single
   final geometry payload pullback.

3. **Exactness tests.** On small in-memory fixtures, compare all state leaves,
   all active prepared/support leaves, `drds`, and interpolation-coordinate
   bars against the sum of current state-lowdot plus generic-support paths for
   one, two, and four objective RHS.  These tests use no full transport run,
   profile dump, or XLA dump.

4. **Benchmark gate.** Only after the above passes, run one remote
   cache-disabled 16-step benchmark.  Reject the selector if first-segment
   compilation or warm rebuilt-support time regresses.  No default or current
   best mode will be changed based on this experiment.

#### Step 1 result: existing joint selector already has the required dispatch

No new Radau selector is required for the dispatch part.  The existing
`ntx_joint_implicit_interpolated_faces` family enters the
`joint_ntx_rebuild_pullback` branch in `_transport_solvers.py`.  That branch
calls exactly one joint hook,
`flat_rhs_build_state_and_support_pullback_batched_interpolated_faces`, and
receives `(rebuild_flat_bars, rebuild_support_bars)` together.

The separate rebuild state hook is in the `else` branch and is therefore not
called.  The generic/batched separate support hook is guarded by
`not joint_ntx_rebuild_pullback` and is likewise not called.  The following
`elif joint_ntx_rebuild_pullback` only adds the already-returned support bars
to the segment accumulator.  Thus the existing joint selector already has
the required "no separate geometry/support transpose" semantics.  The work
is to replace only its costly NEOPAX joint-adapter layout, not to create a
second Radau dispatch mode.

#### Compact joint-carry benchmark result: reject for performance

The cache-disabled 16-step, segment-length-4 benchmark reached the repaired
compact selector but regressed: its first-segment compilation was about 3m36
(established best: about 2m11) and segment 4/4 was 953s (best: about 590s).
Keep this selector isolated; it is not a timing improvement and must not
replace the current best mode.

### Next plan: true matrix-RHS joint lowdot support extension

The current best rebuild path is exact but constructs two independent local
NTX reverse graphs: the fast lowdot state/profile path and the generic
prepared-support VJP.  The rejected joint selectors removed the generic VJP
but instead wrapped a complete scalar prepared-support lowdot graph in an
objective `vmap`.  They therefore built the expensive full directional
prepared algebra once per objective.

The next isolated mode must not repeat either structure.

1. **Residual audit.** Identify the base, Er-direction, and log-nu-direction
   adjoint fields already formed by the fast state lowdot calculation.  Keep
   them as numeric, local-to-one-anchor residuals only; do not add a segment
   tape or change the existing helper.

2. **NTX matrix-RHS extension.** Add a separate NTX helper whose objective
   RHS is a trailing matrix axis in the factorized adjoint sweeps.  It reuses
   the state lowdot base factorization and base adjoints, then calculates only
   the full directional fields and directional adjoints which prepared
   geometry/support derivatives require.  It returns numeric prepared/support
   leaves and direct `drds` bars; it must contain no outer objective `vmap` of
   a scalar full helper.

3. **NEOPAX adapter.** Add one new rebuild selector that calls this helper
   once per local anchor/species and accumulates its numeric leaves on device.
   It must bypass the generic support VJP but leave every established mode,
   including `separate_reuse_local_vjp_primal`, unchanged.

4. **Proof and gate.** Use only small in-memory array/mock tests locally to
   compare one, two, and four RHS against the sum of current best state and
   support bars.  After those pass, run one remote cache-disabled 16-step,
   segment-length-4 benchmark.  Reject the selector on any compile or warm
   rebuild regression.

#### Step 1 audit result: exact reusable boundary

The fast `ntx_helper_lowdot_fused` state path invokes
`solve_prepared_coefficient_vector_lowdot_two_pullbacks` once per local
energy/species/objective.  Its core forms one local primal factorization,
full primal fields, two forward *low-mode* directions, and a matrix adjoint
for the two directional coefficient bars.  It deliberately contracts and
discards the base adjoint instead of materialising it, and returns no
factorization or adjoint residual to NEOPAX.

The current separate support path independently invokes
`jax.vjp(_response_from_support_delta, ...)`.  That creates a second local
prepared-system primal/factorization and prepared-support reverse graph.  So
the best current path really does repeat the NTX implicit system work.

The rejected joint helper avoided that separate generic VJP, but called the
full prepared-support lowdot core through an objective `vmap`; it therefore
materialised full directional fields and prepared bars for each objective.
The earlier `multi_rhs_shared_primal` helper also does not solve this: it
shares only primal residuals, then `vmap`s scalar support adjoints.

The exact new helper must expose and reuse: (a) the primal factorization and
full primal fields, (b) a materialised base lambda pair, and (c) the existing
two-direction lambda matrix.  It must add only the full directional primal
fields and lambda-dot fields required for prepared bars, with objective RHS
as factor-solve columns rather than an outer scalar-helper `vmap`.

#### Native matrix-RHS implementation and local gates

The NTX implementation is now exposed only through the new explicit rebuild
selector:

`ntx_batched_interpolated_faces_native_multi_rhs_shared_primal`

It is separate from the rejected older
`ntx_batched_interpolated_faces_multi_rhs_shared_primal` selector.  The new
NTX helper packs the implicit adjoint columns in the factorized adjoint solve
and applies an objective batch only to the final prepared-gradient
contraction.  The selector does not change any default or the established
best `separate_reuse_local_vjp_primal` path.

Exact small-system gates have passed remotely:

* NTX native matrix-RHS packing and support-only helper: 4 tests passed.
* NEOPAX private prepared-support adapter versus the scalar local pullbacks:
  1 test passed.

These are algebra/axis checks only.  A cache-disabled transport benchmark is
still required to determine whether the native packed graph improves or
regresses compilation and execution.  Do not promote it over the established
best mode without that comparison.

#### Wiring correction before benchmark

The first native-selector benchmark correctly stopped before XLA compilation:
the Radau vector field is bound to `ComposedEquationSystem`, with the
`CombinedTransportFluxModel` and inner `NTXExactLijRuntimeTransportModel`
below it.  The first correction added the composite forwarding method but
missed the equation-system forwarding method, so the Radau factory still
resolved the native hook to `None`.  Both outer forwarding methods are now
present and delegate only to the inner native method, matching the existing
batched-mode wrappers.  Pure mock tests check each forwarding boundary.  No
established selector is changed.

#### Native matrix-RHS benchmark result: exact run completes, reject for total timing

The cache-disabled 16-step, segment-length-4 native matrix-RHS selector
completed and returned finite full transport/geometry rows.  It reduced the
already-compiled rebuild-heavy segments, but increased first-segment compile
cost enough to lose overall:

| Metric | Established best | Native matrix-RHS |
| --- | ---: | ---: |
| First segment XLA compile | about 2m11 | 5m00 |
| Segment 4/4 | 590s | 923s |
| Segment 3/4 | 77s | 46s |
| Segment 2/4 | 228s | 136s |
| Segment 1/4 | 228s | 136s |
| Reverse segment sweep | 1123s | 1241s |

Thus the packed solves reduce warm rebuild execution but enlarge the enclosing
segment XLA graph.  Keep the selector isolated and do not use it as the
current best.  Any follow-up must preserve the packed runtime algebra while
moving it behind a compilation boundary or reducing the generated segment
graph; simply promoting this selector would regress total time.

The completed run also rejects this selector on exactness.  Against the
established `separate_reuse_local_vjp_primal` run, all 60 transport
profile-parameter rows (`n0`, `T0`, and the four shape parameters) printed
bitwise identically, but every VMEC transport row differed.  Representative
relative differences were: `softmax_Er/RBC` 7.46e-2,
`softmax_Er/ZBS` 2.37e-1, `Er2_volume_average/RBC` 3.07e-2,
`Er2_volume_average/ZBS` 7.53e-2, `Er_volume_average/RBC` 1.42e-1,
and `Er_transition_left/ZBS` 9.62e0.  Therefore the small local adapter
oracle did not cover the complete support-to-VMEC geometry chain used by the
new outer batched route.  Do not use or further benchmark this selector
without a whole-support geometry equivalence test.

#### Root cause of the native VMEC mismatch and corrected next step

The native selector calls NTX's deliberately narrow
``prepared_support_only`` helper.  That helper returns a prepared-system bar
and the direct ``drds`` bar, but intentionally discards the cotangents of the
local NTX case ``(nu_hat, epsi_hat)`` and its ``vth_a`` scale.  The established
``separate_reuse_local_vjp_primal`` path instead differentiates the complete
local response with respect to ``(prepared, drds)``.  Its ``drds`` transpose
therefore includes the chain

``drds -> (nu_hat, epsi_hat, vth_a) -> local NTX response``.

This missing chain is the concrete reason the native run preserves all
profile rows but changes only the VMEC rows.  It is not a numerical tolerance
issue and it does not require another NTX primal or adjoint solve.

The correction remains isolated: extend the existing native matrix-RHS
low-dot helper to return its already-available batched case bars; then apply
their NEOPAX transpose with
``_pullback_local_scan_inputs_and_drds_from_primitives`` and add that result
to the direct ``drds`` bar.  The exact local test must compare this corrected
result against the real generic local ``jax.vjp`` of the response (including
the primitive chain), with one and two energies and multiple RHS.  Only after
that gate passes may it be benchmarked.

The compile-time task is separate: this exactness correction should retain
the native execution saving but cannot by itself promise a smaller HLO.  Once
the local VJP gate is exact, measure a device-only non-inline call boundary
on the corrected native helper using a small mock lowering before asking for
another transport benchmark.  Do not use a host callback, a full tape, or a
serial objective scan.

### Current plan: exact native shared-field support path, then compile reduction

**Scope guard.** The established exact bounded-memory selector remains
`separate_reuse_local_vjp_primal`.  All work below stays behind the isolated
native selector.  It must not add a host transfer, a Python/objective loop,
a persistent per-step tape, or an additional NTX factorization/adjoint solve.

1. **Find the exact missing prepared-support term.** The new local VJP gate
   established that the scalar `prepared_support_only` algebra already differs
   from the true local response VJP, before native matrix-RHS packing is used.
   Split that gate into base, `d/dEr`, and `d/dlog(nu)` output cotangents and
   compare each prepared bar to the real local VJP.  This identifies the one
   omitted/misweighted low-dot support term without a transport rollout.

2. **Correct the algebra at the shared-field level.** Express the identified
   term from the already available native primal, base-adjoint, directional-
   adjoint, and lambda-dot fields.  Pack all objective cotangents as trailing
   matrix-RHS columns.  The result must match the full local VJP for both one
   and two energies, multiple RHS, and the `drds -> epsi_hat` primitive chain.
   The scalar routine remains a test oracle; it is not used in the benchmark
   mode.

3. **Only then address compilation structure.** Before another transport run,
   lower a small in-memory native kernel and compare its generated JAX/XLA
   structure against the current native selector.  A compile reduction is
   acceptable only if it preserves the one shared matrix-RHS factorization and
   does not inline a full prepared-support reverse graph once per Radau slot.
   Candidate: split the already-known realized lagged pattern into static
   reuse/rebuild segment variants, so XLA does not compile both giant branches
   in every reverse slot.  This is bounded by segment patterns, independent of
   total accepted-step count.

4. **Promotion gate.** Run small CPU algebra tests, then the GPU transport
   benchmark only after exactness passes.  Reject the isolated selector unless
   it reproduces the established derivative table and improves total compiled
   and warm reverse timing.  No default changes before that result.

### Current plan: compact native matrix-RHS return path

The corrected native selector now reproduces the established transport
derivatives, and reduces later rebuild-heavy segment execution, but it raises
peak memory and first-segment compilation.  The established
`separate_reuse_local_vjp_primal` selector remains unchanged and is still the
current total-time baseline.

### Deferred, retained option: static lagged-pattern specialization

Static reuse/rebuild-pattern segment dispatch remains a valid compile-only
candidate, but it is **not the next experiment**.  The immediate target is to
retain the one grouped native NTX adjoint that gives the now-exact
approximately `46s`/`135s` warm execution, while reducing its compiled graph
and live temporary footprint toward the lower-compile/memory reference mode.
The reference mode is a graph-shape comparison only: do **not** revert to
separate NTX state/support adjoints or a separate geometry JVP.  The next
investigation must instead compact/fuse the grouped adjoint's internal
post-adjoint contractions and output temporaries, with small in-memory
lowerings and exact local derivative tests.  Do not implement static pattern
specialization until that grouped-adjoint graph-boundary comparison is done.

1. **Compact the native temporary contract.** In a new isolated helper, retain
   the same primal/factorization and matrix-RHS adjoint fields, but return one
   final prepared bar rather than five directional prepared trees. Retain the
   existing small case-component contract, so NEOPAX uses its proven case-chain
   combination unchanged.
   Implemented as isolated selector
   `ntx_batched_interpolated_faces_native_multi_rhs_compact_shared_primal`.
2. **Stream energy accumulation.** Replace the mapped stack of large
   energy-by-objective prepared bars with a `lax.scan` carry.  The objective
   axis remains matrix-RHS batched; this only prevents energy intermediates
   from remaining live until a later reduction. **Deferred:** the first scan
   implementation did not reproduce the mapped native case bars, so it is not
   selected. The compact selector retains the proven 5-to-1 prepared-return
   reduction while its energy reduction stays on the exact mapped reference.
3. **Local gates first.** Use only small in-memory CPU mock equivalence tests
   and lowering-size checks.  Require identical bars and no additional NTX
   factorization, no host transfer, no persistent timestep tape, and no
   objective serial loop.
4. **Compile structure second.** If the compact graph is smaller, test an
   isolated native call boundary in the mock lowering.  Treat static
   reuse/rebuild segment-pattern specialization as a separate subsequent
   compile-only change; it is bounded by `2**segment_length`, not accepted-step
   count.

### Timing attribution correction: native `drds`-JVP reuse (2026-08-24)

The lower warm timings first appeared in the **derivative-corrected**
`ntx_batched_interpolated_faces_native_multi_rhs_shared_primal` mode, after
the sequence of native support corrections for retained directional coefficient
cotangents, the `log(nu*)` base term, and the objective-batched `log(nu*)`
support-coordinate pullback.  That corrected selector predates both the
compact-return and `drds`-JVP-reuse variants.

| Selector / result | First-segment compile | 3/4 (one rebuild) | 2/4 and 1/4 (three rebuilds) |
| --- | ---: | ---: | ---: |
| Derivative-corrected `...native_multi_rhs_shared_primal` | about `5m00s` | `46s` | `136s` |
| `...native_multi_rhs_reuse_moment_drds_jvp_shared_primal` | about `4m34s` | `45.950s` | `135.024s`, `134.821s` |

Therefore the later low times are a property of the existing native
matrix-RHS execution framework, **not evidence that reusing the two local
`drds` JVPs caused a further warm-execution reduction**.  The reuse variation
only removes a source-level duplicate directional moment-JVP expression.  In
the full segment compilation it has not reduced peak RAM (about 55% of 30 GiB)
or materially reduced first-segment compilation (still about 4--5 minutes).
It remains isolated and must not be presented as the cause of the native
warm-segment gain.

#### Derivative gate for the `drds`-JVP-reuse native selector

The completed cache-disabled `...native_multi_rhs_reuse_moment_drds_jvp_shared_primal`
run was compared row-by-row with the completed **derivative-corrected native
shared-primal** run (the reference that already had the approximately
`46s`/`136s` warm timings).  All 128 Jacobian rows have matching labels.  The
largest relative difference is `7.73e-11`
(`geometry:boozer_qi_objective / dvmec:ZBS:1:0`); no row exceeds `1e-10`
relative difference.  Thus this isolated variation preserves the corrected
native derivative table to floating-point roundoff.  The older native run
before the later derivative corrections is deliberately not a comparison
oracle because its VMEC geometry rows differ materially.

#### Mandatory base for subsequent grouped-adjoint work

All subsequent compile-graph and peak-memory work must start from the
derivative-validated isolated selector
`ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal`.
It retains the one grouped native NTX adjoint, the matrix-RHS execution
structure, and the validated `45.950s`/`135.024s`/`134.821s` warm segment
measurements.  Do not redirect this work to `separate_reuse_local_vjp_primal`,
the compact-native selectors, or a joint/geometry-only selector.  Those modes
are comparison references only and must remain independent opt-ins.

### Current grouped-adjoint compile/RAM finding (2026-08-24)

The current base selector retains one grouped NTX primal/factorization and
matrix-RHS implicit-adjoint solve in
`_lowdot_two_pullback_native_multi_rhs_adjoint_fields`.  That part is the
source of the validated warm execution and must remain unchanged.

The compile/RAM expansion occurs **after** that grouped solve.  The native
helper `solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_native_multi_rhs_and_aux`
still executes:

```text
jax.vmap(_one_rhs)
  base:        _prepared_gradient_from_adjoint(...)
  d/dEr:       _directional_prepared_gradient_from_adjoint(...)
  d/dlog(nu):  _directional_prepared_gradient_from_adjoint(...)
```

Each of those prepared-gradient routines contains a generic JAX VJP over the
complete prepared residual.  Thus the factorized NTX adjoint is grouped, but
the complete prepared-support transpose is still generated as an
objective-batched AD graph.  The compact-return selector combines the three
large prepared trees only **after** they have been constructed, and the
`drds`-JVP reuse removes only an upstream moment contraction.  Neither can
shrink this dominant graph.

The next viable modification is therefore a new, isolated **native
multi-RHS prepared-support transpose** in NTX.  It must accept the existing
grouped base/directional adjoint fields with a trailing RHS axis and directly
form only the final exact prepared/support bar (plus the existing small
case/`drds` bars).  It must replace—not wrap—the `_one_rhs` `vmap` above.
It must keep one grouped NTX adjoint, avoid an objective loop/host transfer,
and be first proven against the present helper using small in-memory tests and
Jaxpr/lowering-size checks.  Only then should NEOPAX select it as a new opt-in
variant of the validated `drds`-reuse mode.

### Active implementation plan: compact grouped native prepared transpose

**Starting point.** All work starts from the derivative-validated
`ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal`
mode.  That selector and every existing selector remain unchanged until the
new path passes every gate below.

1. **Contract and oracle.** Define an NTX-private helper whose inputs are the
   existing grouped base/directional primal and adjoint fields with a trailing
   objective-RHS axis.  Its output is exactly the current final combined
   prepared bar, direct `drds` bar, primal response, and compact case bars.
   Add tiny in-memory tests against the present native helper for one and
   multiple RHS columns; test both physical directions and all differentiable
   prepared leaves.
2. **Replace the expensive finalisation only.** Implement a native RHS-axis
   prepared-support contraction that replaces the current `jax.vmap(_one_rhs)`
   over one base and two directional generic prepared-residual AD transposes.
   It must retain the one grouped primal/factorisation/matrix-RHS adjoint
   build, use no host/objective loop, and construct only the final combined
   prepared bar rather than five full objective-batched prepared trees.
3. **Graph gate before transport wiring.** On a small in-memory fixture,
   compare the helper's Jaxpr/lowering structure with the current native helper:
   no objective-vmapped generic prepared VJP, no additional factorisation or
   NTX solve, and no widened input/output contract.  Exact values must match
   at floating-point roundoff.
4. **Isolated NEOPAX selector.** Add a separately named opt-in rebuild mode
   that calls the new helper but retains the current `drds`-reuse primitive and
   case-chain handling.  Do not change defaults or the current validated mode.
5. **Promotion gate.** Run the same remote cache-disabled 16-step benchmark;
   accept only if derivatives match the 128-row reference and compile/RAM fall
   without regressing the current roughly `46s`/`135s` warm rebuild timings.

#### Implementation status (2026-08-24)

Step 1 is complete locally.  NTX now has the private
`_combined_prepared_gradient_from_adjoint_multi_rhs_oracle` and a small
one-/three-RHS test.  It consumes the already grouped primal and
matrix-RHS-adjoint fields and verifies the final combined prepared bar against
the existing compact native contract.  The test passed in WSL without a
transport rollout, cache/profile output, or file dump.  No benchmark selector
calls this helper yet, so the validated `drds`-reuse mode and every other mode
are unchanged.

The source audit also establishes an important constraint for Step 2:
`jax.vmap(_one_rhs)` is a device-batched transform, not a Python/serial loop.
Simply replacing it by a scan would increase warm execution.  Likewise, an
`inline=False` call boundary could reduce the parent Radau module's live graph,
but it can add device-call/fusion overhead and therefore is not a guaranteed
warm-time improvement.  The only no-extra-NTX-solve route that can reduce both
the temporary graph and preserve the grouped computation is an explicit
RHS-axis transpose for the fixed prepared residual.  Step 2 will derive that
transpose from the existing residual/operator algebra; it must not merely wrap
the current generic VJP.

#### Step 2 implementation in progress

The first exact implementation is now in NTX, still unselected:

* `_compact_prepared_residual_inputs` isolates the residual's true dynamic
  dependence: geometry for the direct coefficient/source terms, `d_theta`,
  `d_zeta`, and `block_parameters` for dense block construction.
* `_compact_prepared_gradient_from_adjoint` differentiates the fixed residual
  only to that tuple; `_compact_prepared_bar_to_prepared` performs the small
  final chain back to `PreparedMonoenergeticSystem` once.
* The two low-dot directional terms use the same split residual algebra, then
  add their compact bars before that final prepared pullback.
* A new NTX opt-in helper
  `...native_multi_rhs_compact_residual_and_aux` implements the full
  matrix-RHS contract. It is not yet exposed in NEOPAX, so it cannot alter the
  current benchmark path.
* The compact pytree structure is derived directly from the compact residual
  input tuple—not by executing a throw-away scalar residual VJP—so this path
  does not add an unconsumed transpose while tracing the matrix-RHS kernel.

Small one-/three-RHS exact tests compare (a) each split base/directional
prepared cotangent and (b) the new helper's full compact result—including its
case components—against the established native compact contract. They pass
locally. The next task is the in-memory graph gate; only if that shows the
expected removal of the prepared-construction chain from the dense residual
transpose will a new NEOPAX opt-in mode be wired.

#### Graph gate and NEOPAX-local wiring result

The small in-memory Jaxpr gate passes: the split helper has fewer traced
equations than the existing compact native helper while retaining the same
output arity.  This is the intended structural reduction; it does not add an
NTX factorization, an adjoint solve, a host transfer, or an objective scan.

The helper is now routed through the NEOPAX **local** interpolated-moment
adapter, including the validated `reuse_joint_moment_drds_jvp=True` chain. A
CPU small-model NEOPAX equality test confirms its prepared bar, `drds` bar,
primal response, and case components match the established native helper.
It remains deliberately absent from `reverse_rebuild_support_pullback_mode`:
the next isolated change is the outer reverse selector/factory wiring, after
which the remote cache-disabled derivative and timing gate can be run.

#### Outer selector wiring complete

The new isolated selector is now:

`ntx_batched_interpolated_faces_native_multi_rhs_compact_residual_reuse_moment_drds_jvp_shared_primal`

It is accepted by the benchmark parser and reverse setup validation, selects a
dedicated flattened support-pullback factory/hook, forwards through the
composite transport model, and invokes the split-residual native NTX adapter
with the existing `drds`-reuse setting. Existing selector names, defaults, and
their factory paths remain unchanged. Static compilation and the CPU local
adapter equality gate pass. The next action is the remote cache-disabled
16-step benchmark, comparing its derivative table and compile/RAM/warm segment
times to the validated `...native_multi_rhs_reuse_moment_drds_jvp_shared_primal`
reference.

#### Three-pass audit (2026-08-24)

1. **Mathematics:** the compact residual includes the direct coefficient term,
   both source terms, every lower/diagonal/upper block contribution, the
   nullspace-row conditioning, and both low-dot directions.  The final compact
   input pullback sums direct-geometry and block-parameter paths back to the
   same prepared leaves as the original all-in-one VJP.
2. **RHS/temporary contract:** objective columns remain in one device `vmap`.
   The split path does not introduce a Python loop, `lax.scan`, host transfer,
   extra factorization, or extra NTX adjoint solve. One-/three-RHS exact gates
   cover its full compact prepared and case-bar contract.
3. **Integration:** the NTX re-exports and NEOPAX local adapter compile clean.
   The isolated outer selector is now present in reverse-mode validation, the
   dispatch factory, both prepared-Radau context construction paths, and the
   benchmark parser. Existing selector strings and their branches remain
   unchanged, so the current derivative-validated selector cannot be
   contaminated.

### Corrected next plan: native batched prepared-support transpose

#### Result of the compact-residual experiment

The isolated selector
`ntx_batched_interpolated_faces_native_multi_rhs_compact_residual_reuse_moment_drds_jvp_shared_primal`
now executes through the complete outer forwarding chain.  Its first remote
segment compiled in approximately `4m44s`, its one-rebuild warm segment was
`45.331s`, and its observed peak host RAM remained approximately the same as
the validated native baseline.  Therefore it preserves the useful grouped
matrix-RHS execution but **does not** meet the compile/RAM objective.  It must
remain an independent opt-in experiment and must not replace the baseline.

The precise reason is that the helper first forms compact RHS cotangents with
`jax.vmap(_one_rhs_compact_residual)`, then performs a second
`jax.vmap(_one_compact_bar_to_prepared)`.  That latter transform applies the
compact-input-to-full-prepared VJP once per objective column and XLA inlines
it into the same Radau segment module.  Splitting the residual inputs alone
does not remove the objective-batched final support-transpose graph.

#### Non-negotiable mathematical contract

Do **not** replace the final support transpose by one scalar/combined
cotangent: that would return only a weighted sum of objective gradients and
would be wrong whenever more than one objective is requested.  The desired
operation is the exact native matrix-RHS transpose

```text
prepared_support_bars = J_prepared^T @ C
```

where `C` has one column per requested objective.  The output must retain the
leading RHS/objective axis, but the implementation must form it directly,
without a generic prepared VJP or JVP being transformed once per column.

#### Implementation plan

1. **Freeze the reference contract.** Keep
   `ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal`
   as the only baseline.  Add/retain small NTX one- and three-RHS tests that
   compare the final prepared bars, `drds` bars, primal response, and all ten
   case components at roundoff.  No existing selector or default changes.

2. **Map the prepared-support dependency exactly.** Starting from the already
   grouped fields returned by `_lowdot_two_pullback_native_multi_rhs_adjoint_fields`,
   classify each contribution to `J_prepared^T @ C`: direct coefficient
   geometry term, source term, lower/diagonal/upper block terms, `d_theta`,
   `d_zeta`, and `block_parameters`/geometry chain.  Verify the nullspace row
   and both low-dot directions before writing a new rule.

3. **Implement a private native RHS-axis transpose in NTX.** It must consume
   the trailing-RHS primal and adjoint fields and use matrix contractions over
   that axis to construct the final prepared/support bars.  It must replace
   both compact-residual `vmap`s; it may not call
   `_prepared_gradient_from_adjoint`,
   `_directional_prepared_gradient_from_adjoint`, or
   `_compact_prepared_bar_to_prepared` under an objective `vmap`.  It must add
   no factorization, primal solve, implicit-adjoint solve, host transfer, or
   Python objective loop.

4. **Handle the compact-input chain natively.** The hard part is the shared
   `block_parameters`/geometry mapping.  Supply a dedicated batched transpose
   for that mapping, again returning one bar per RHS.  Do not materialize an
   explicit full Jacobian and do not use `scan` as a memory workaround: both
   would either increase memory or sacrifice the validated warm execution.

5. **Prove graph and value properties locally.** On small in-memory fixtures,
   compare the new helper against the frozen reference for one and three RHS.
   Add a structural Jaxpr gate that verifies no objective-batched generic
   prepared VJP/JVP remains and that no additional factorized NTX solve is
   introduced.  These are CPU/mock tests only; do not run a transport rollout
   locally or emit XLA/profile/output files.

6. **Wire a new isolated NEOPAX selector only after the gates pass.** Preserve
   the existing `drds`-reuse/matrix-RHS route and use a distinct selector
   name.  Remote acceptance requires: the 128-row derivative table matches
   the validated reference, compile time and RAM decrease materially, and the
   approximately `46s`/`135s` warm segment timings do not regress.

An `inline=False` call boundary remains only a separately measurable fallback:
it can hide graph size from the parent segment but may add call/fusion overhead
and does not by itself remove the repeated transpose.  Do not implement it as
the substitute for Step 3.

#### Step 2 progress: direct dense-block RHS transpose

The first production component of the replacement is now implemented privately
in NTX as `_fixed_residual_block_coefficient_bars_multi_rhs`.  It computes the
lower/diagonal/upper dense-block coefficient cotangents directly with the RHS
axis retained as `(mode, rhs, coefficient, spatial)`.  It includes both
primal fields and the nullspace-row replacement exactly, and makes no
factorization, solve, generic VJP, JVP, or objective `vmap` call.

A small CPU-only one-/three-RHS test compares that component against an
independent generic packed-block VJP oracle at roundoff.  It passes.  This is
only the block-residual portion: source/direct-coefficient geometry terms and
the block-parameter-to-prepared chain still need their own native
RHS-preserving transposes before a selectable NEOPAX mode is justified.

### Next step: VMEC-only native geometry transpose

Keep `ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal`
as the derivative-validated reference; do not modify it. The new route stays a
separate opt-in experiment.

The NTX primal, factorization, and grouped matrix-RHS implicit adjoints are
already shared. The remaining compile/RAM graph is the generic reverse of the
local map `VMEC Fourier coefficients -> sampled NTX geometry -> transport`.
It is not a VMEC equilibrium solve or VMEC adjoint.

1. Freeze a small VMEC primitive-bar reference against the generic geometry
   VJP.
2. Add native RHS-axis pullbacks for `radial_drift_spatial`, `volume_prime`,
   and `b2_mean`.
3. Apply each VMEC Fourier grid-evaluation transpose directly to batched field
   bars.
4. Map those bars to selected runtime parameters, retaining the RHS axis.
5. Run one-/three-RHS VMEC equality gates against the grouped-native reference.
6. Only then add a distinct selector; accept remotely only if the 128-row
   derivative table matches, warm timing remains ~46s/~135s, and compile/RAM
   decrease materially.

Boozer stays on its established generic route initially. No step may add an
NTX solve/factorization, objective scan, host transfer, or default change.

#### VMEC coefficient rule: base and low-dot contract now gated

The private NTX VMEC rule is now complete through the local coefficient
boundary.  It directly contracts the fixed residual blocks, source terms,
and direct transport coefficients to VMEC sampled-field bars, then applies
the Fourier transposes to the six traceable VMEC coefficient arrays.  The
low-dot companion accepts the existing primal/directional fields and has no
NTX solve or factorization of its own.

Two CPU-only one-/three-RHS gates now compare that rule with the established
generic chain

```text
VMEC coefficient arrays -> prepare_monoenergetic_system -> compact prepared VJP
```

for both the base cotangent and its low-dot directional derivative.  Both pass
at roundoff.  These tests use only small in-memory VMEC fixtures; they do not
run transport, use a GPU, or emit files.

The next integration boundary is not the NTX solve.  The benchmark uses
`ntx_exact_surface_backend = "vmec"`, and its payload reverse currently
reverses the traceable VMEC field-table construction from the local coefficient
arrays back to the converged VMEC state generically.  A new opt-in route must
therefore insert the native coefficient bars at that *state-bar* boundary and
then reuse the existing raw-block VMEC transpose.  It cannot simply replace
`face_prepared` with a coefficient dictionary: the existing payload contract
requires a VMEC-state cotangent.  The next implementation task is a batched
native transpose of the traceable VMEC field-table/interpolation map.  The
current best selector remains unchanged until that state-bar equality gate
passes.

#### Native VMEC coefficient-return discrepancy (2026-08-25)

The private NTX opt-in return `return_vmec_coefficient_bars=True` was compared
strictly with the existing prepared-system VJP chained back to the same small
`VmecSurface`. The base and directional helper gates pass independently at
roundoff, but the combined low-dot return differs on this ill-conditioned
fixture (the `b_cos` bar is roughly `1e-7` relative). The discrepancy is in
the post-adjoint VMEC coefficient contraction; it adds no NTX primal,
factorization, or implicit-adjoint solve.

This route is not wired to NEOPAX and must remain unselected. Do not relax the
strict equality gate or replace the native contraction with the full generic
prepared VJP, because that would defeat the compile/RAM objective. First
obtain an end-to-end AD/FD comparison on the existing derivative-validated
native selector. Only if it shows a material geometry-gradient error should we
derive a native scalar contraction that reproduces the generic reduction
structure while retaining the shared NTX solves.

#### In-progress AD comparison bridge (2026-08-25)

The isolated comparison route is now being threaded as a *parallel* result:
the existing support payload bar remains unchanged, while the NTX native
low-dot helper returns a separate batched dictionary of face VMEC coefficient
bars.  NEOPAX can map that dictionary through the exact traceable
VMEC-state-to-face-coefficient VJP and add the resulting state bar before the
existing raw-block transpose.  The local NTX, flux-model, composed-equation,
and raw-block endpoints are implemented and remain unselected.

The segmented Radau carrier is now complete.  It accumulates the six native
coefficient-bar arrays in parallel with (not inside) the ordinary support tree,
then the existing raw-block VMEC payload transpose maps them from all face
surfaces to the converged VMEC state.  The comparison selector is deliberately
separate:

```text
ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients
```

It uses the same grouped low-dot NTX solve/factorization as the current
`reuse_moment_drds_jvp_shared_primal` mode. The special NTX helper now also
has a bridge-only return: it still forms the same grouped matrix-RHS adjoints
and native coefficient bars, but omits the generic prepared-gradient
contraction and returns a shape-compatible zero prepared payload to preserve
the current Radau ABI. Existing modes and defaults are untouched.

#### Repair: preserve generic post-sweep prepared bars (2026-08-25)

The first VMEC bridge integration was invalid. It treated native coefficient
bars as a replacement for the *entire* generic support pullback and ran a
second channels-only Boozer VJP. That dropped the generic `face_prepared`
cotangents from objective/initial-cache/initial-Er-root paths and caused the
observed post-sweep Boozer OOM.

The repair keeps the selector isolated and uses one ordinary support VJP:

```text
generic objective + cache + root payload bars
  + rebuild center/face runtime-channel bars
  -> one normal NTX-support-to-VMEC-state VJP

native rebuild face-coefficient bars
  -> direct coefficient-to-VMEC-state VJP
```

Thus only the rebuild `face_prepared` contribution is replaced; all generic
prepared contributions remain exact. No extra NTX solve/factorization or
channels-only Boozer VJP is introduced. The current reference drds-reuse
selector remains untouched. Next gate: run the isolated selector remotely and
compare its AD table against the reference before interpreting compile/RAM.

#### Grouped final-objective terminal cotangents (2026-08-25)

The `support reverse final-objective cotangents` phase was independent of the
rebuild selector and still constructed one final-state VJP and one
explicit-geometry VJP for each ordinary objective. A separate opt-in
`reverse_final_objective_cotangent_mode="grouped_vjp"` now constructs one
vector-valued VJP for all non-bootstrap objectives and maps its pullback over
the on-device objective basis. Bootstrap remains on its existing compact Upar
state/support/geometry rule. `scalar` remains the default/reference mode.

The mechanical vector-VJP row extraction has a small pure equality gate;
before timing it remotely, compare grouped and scalar objective/gradient rows.

#### Next work: complete native VMEC-surface replacement contract (2026-08-25)

The remote run of

```text
ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients
```

completed without the post-sweep OOM and retained the desired rebuild execution
times, but its transport `RBC`/`ZBS` derivatives differ materially from the
validated generic `reuse_moment_drds_jvp_shared_primal` reference.  This is
not evidence of a second NTX adjoint or a duplicate coefficient term.

The selector passes `native_vmec_coefficient_bars_only=True` into NTX.  NTX
therefore returns a zero generic `face_prepared` cotangent and returns native
bars only for six Fourier arrays (`b_cos`, `jacobian_cos`, `b_sub_theta_cos`,
`b_sub_zeta_cos`, `b_sup_theta_cos`, and `b_sup_zeta_cos`).  Those six bars
replace the rebuild `face_prepared` contribution.  The generic objective,
initial-cache, and initial-Er-root support bars remain present by design.

The first concrete audit finding is that the local fixed residual depends on
the six sampled Fourier fields *and* the independent `b0` scalar.  The old
native conversion silently discarded the latter. Other VMEC-surface leaves
are retained in the full-surface gate and must prove to have zero local
prepared cotangent before they are excluded. The bridge therefore remains
experimental until it matches the reference/FD table.

Plan:

1. **Define the exact replacement boundary.** Enumerate every dynamic
   `VmecSurface` leaf consumed by the VMEC preparation graph and add a small
   NTX gate that compares the *entire* native low-dot surface bar (base plus
   both directional terms, arbitrary RHS count) with the ordinary prepared
   VJP.  The existing six-coefficient tests are retained but are insufficient.
2. **Extend the native NTX return contract.** Return the missing scalar
   VMEC-surface cotangents through the same fixed-adjoint/low-dot algebra as
   the six coefficient bars.  Do not reinstate the full generic prepared-bar
   tree or add another NTX primal/factorization/adjoint.
3. **Route the full compact surface bar in NEOPAX.** Replace only rebuild
   `face_prepared` with this complete native surface contract; retain ordinary
   objective, cache, root, and runtime-channel support cotangents.  Contract
   the full native surface bar against the existing implicit VMEC state
   tangents on the compact path.
4. **Prove equivalence before a remote transport run.** Add small in-memory
   NTX and NEOPAX gates for base/directional/full-low-dot equality and for the
   merge boundary.  No rollout, GPU benchmark, profile dump, or temporary
   XLA output is permitted in local tests.
5. **Remote validation sequence.** First compare the full AD table with the
   validated generic drds-reuse reference; then run the existing RBC/ZBS FD
   comparison.  Only if both agree will this selector become a candidate for
   compile/RAM optimisation.  The reference selector remains untouched.

Implementation status: the first concrete missing leaf, `VmecSurface.b0`, is
now carried by the native NTX dictionary and through the NEOPAX face-surface
tangent contraction. `b0` is an independent scalar used directly in the NTX
transport coefficients; it was present in the native primitive cotangent but
was silently discarded by the old Fourier-only conversion. Base,
directional, and combined-contract tests now include it. The remaining
full-surface gate is deliberately strict; no tolerance has been relaxed.

Audit correction: `transport_psi_scale` is a dynamic `VmecSurface` leaf, but
it is *not* consumed by this local prepared NTX solve.  NEOPAX resolves and
passes `MonoenergeticCase(epsi_hat=...)` explicitly at every local runtime
call, so NTX does not invoke its alternative `er_hat / transport_psi_scale`
path.  It must therefore remain zero in the strict replacement-contract
gate; adding a native bar for it would be dead graph payload, not a missing
derivative.

Current repair under test: the native combined low-dot helper previously
projected its base, epsilon-directional, and nu-directional primitive bars to
VMEC coefficients separately, then added the three coefficient pytrees.  The
strict complete-contract gate found a `b_cos` mismatch of about `9.8e-6` on a
value of about `55`, despite each isolated base/directional gate passing.
The helper now sums those three primitive-bar dictionaries first and executes
one sampled-field/Fourier transpose.  This mirrors the combined prepared VJP
contraction order and removes two duplicate VMEC reverse projections without
changing the NTX primal, forward-direction solves, or implicit adjoints.

Result of the first gate: this reduction-order change did **not** satisfy the
strict complete-contract equality.  It moved the synthetic `b_cos` result
only from `55.13340248` to `55.13340163`, while the generic prepared VJP is
`55.13339269` (about `1.6e-7` relative).  Do not use this small discrepancy to
justify a transport benchmark or relax the strict gate.  It predates `b0` and
is far too small to explain the material RBC/ZBS transport-row discrepancy;
the next audit must isolate the three low-dot components at the actual fused
fields and separately verify the newly restored `b0` state chain.

#### `b0` reverse-payload result (2026-08-25)

The native-VMEC selector was rerun after routing the native `b0` cotangent:

```text
ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients
```

The desired warm reverse execution behaviour remained, but the material
transport-objective VMEC rows did **not** move from the earlier native result.
The validated generic `reuse_moment_drds_jvp_shared_primal` reference and the
new native result are:

| objective | RBC reference | RBC native | ZBS reference | ZBS native |
| --- | ---: | ---: | ---: | ---: |
| `softmax_Er` | -59.29418 | -47.16818 | 17.36357 | 27.70411 |
| `Er_transition_left` | -20.32353 | -11.96894 | 0.14571 | 7.91715 |
| `Er_transition_right` | -22.52000 | -13.86698 | 1.01124 | 9.05793 |
| `Er2_volume_average` | -640.27857 | -579.81062 | 236.83438 | 327.35410 |
| `Er_volume_average` | -20.50225 | -19.55786 | 13.01206 | 13.98852 |
| `bootstrap_current_softmax_abs_scaled` | -1.78901 | -1.79714 | -6.25244 | -6.26012 |

All profile rows and pure geometry-objective rows are unchanged.  Thus `b0`
was a genuine missing local native channel, but its contribution is negligible
for this case and cannot explain the material RBC/ZBS transport discrepancy.
The selector remains an execution-time experiment only; do not use it for a
validated geometry gradient until the remaining rebuild-to-VMEC bridge term is
identified and the table agrees with both the generic reference and FD.

#### Replacement-boundary audit (2026-08-25)

The subsequent code audit rules out three proposed explanations for the
remaining material transport-row mismatch:

1. **Face sampling is not drifting.** Both the ordinary support builder and
   `_native_vmec_face_coefficients_from_state` derive the same centre/face
   sampling from `n_r`, apply `_positive_transport_s_values_from_rho`, and use
   `_traceable_vmec_surfaces_from_state` for the `vmec` backend.
2. **The generic payload is not being discarded wholesale.** In native mode,
   `_merge_rebuild_ntx_channels_into_generic_payload_bar` retains the ordinary
   objective, initial-cache, and initial-root prepared bars.  It adds only the
   rebuild runtime-channel bars and replaces only rebuild `face_prepared`.
3. **There is no additional active scalar surface channel to add.** The local
   prepared NTX residual reads the VMEC surface through sampled `b`, its
   derivatives, Jacobian/covariant/contravariant fields, radial drift, and
   `b0`.  The explicit runtime case supplies `epsi_hat`, so
   `transport_psi_scale` has no local path; `iota` is retained metadata in
   `GeometryOnGrid` but is not read by the fixed NTX operator or coefficient
   formulas for a `VmecSurface`.

The unresolved problem is therefore narrower and more important: the native
path is a re-derived fixed-residual/low-dot transpose, not merely a different
unflattening of the generic prepared VJP.  Its strict combined low-dot VMEC
test already disagrees with the generic prepared VJP at about `1.6e-7`
relative.  That isolated error is too small to explain the transport table,
but it proves that the native replacement has not yet been established as an
exact contract.  The next required gate is a small **NEOPAX face-rebuild
integration oracle**: compare the complete generic `face_prepared` bar,
chained through the same traceable VMEC face surfaces, against the native
coefficient bar for each low-dot component (primal response, `d/dEr`, and
`d/dlog(nu*)`) before any further remote transport run.  This identifies
whether the material loss arises inside the local multi-energy/low-dot
assembly or only when its face bar is inserted into the payload bridge.

**Gate result (CPU, 2026-08-25):** all three fixed-`epsi_hat` component cases
passed with the expected-failure marker disabled (`--runxfail`). This proves
only that the native multi-energy low-dot coefficient bars equal the generic
`face_prepared -> VmecSurface` VJP when the local NTX electric-field input is
held fixed. It does **not** cover the physical `Er * drds -> epsi_hat` chain.
Since the material table disagreement is limited to Er-dependent objectives,
that omitted chain is the primary suspect; a separate local Er/drds oracle is
required before assigning the mismatch to the downstream payload bridge.

**Er/drds-chain gate result (CPU, 2026-08-25):** the three-component oracle
(`transport_moments`, `d/dEr`, and `d/dlog(nu*)`) passed with
`--runxfail`, using the production `get_v_thermal` convention and the full
`Er * drds -> epsi_hat` reverse chain.  Therefore the native local NTX case
bars, their `drds` contribution, and their `Er` contribution agree with the
generic local response VJP.  The Er-only transport-table disagreement is
**not** an omitted or duplicated `Er -> epsi_hat` factor.  The remaining
candidate boundary is after this local result: converting the native VMEC
coefficient bars into the traceable VMEC-state/payload cotangent, or merging
that cotangent with the generic rebuild payload.

**Expanded local surface-contract result (CPU, 2026-08-25):** the strict
`native_vmec_face_rebuild_component_oracle` was rerun for all three actual
low-dot response components and passed (`3 passed`).  It verifies the native
coefficient return against the ordinary `face_prepared -> VmecSurface` VJP,
including `b0`, and verifies that every other omitted differentiable
`VmecSurface` field has a zero bar in the explicit-`epsi_hat` contract.  In
combination with the Er/drds-chain gate, this rules out local NTX algebra,
the `Er * drds -> epsi_hat` chain, and an omitted sampled surface field as
the source of the material transport-objective discrepancy.

**Audit correction:** the fast generic drds-reuse selector and the native
VMEC-coefficient selector both explicitly use
`include_second_direction_base_prepared=False`.  The compact native contract
therefore does not drop a term that the validated reference retains; do not
add that term.

**Multi-face rebuild-accumulation audit (CPU, 2026-08-25):** the initial
`native_vmec_face_rebuild_accumulation` pass used three face indices but
accidentally broadcast the same prepared VMEC surface to each one.  It did
prove the RHS/anchor carrier ordering and that the native helper returns a
zero `face_prepared` payload, but it did **not** prove the radial face-surface
case used in production.  The gate now constructs three distinct sampled VMEC
coefficient surfaces and independently VJPs each face.  It continues to
exercise two selected anchors, two species, two energy points, two objective
RHS rows, and the production anchor interpolation transpose.  This distinct-
surface gate is the next required small CPU test.

Only after it passes can the local custom pullback be considered established
through the complete rebuild support boundary.  If it passes, the remaining
unvalidated boundary is the Radau segment carrier/rebuild split or contraction
of the accumulated native face-coefficient bars against the traceable
VMEC-state tangent.  If it fails, the mismatch is inside the native custom
low-dot coefficient rule for radially varying VMEC surfaces.

### Next work: joint initial-carry reverse (2026-08-25)

The current post-sweep initial work is structurally split despite sharing the
same initial lagged response:

1. `initial_cache_support_pullback` applies the batched support transpose to
   `reduced_bars.lagged_response_cache`, producing support/geometry bars.
2. The initial-carry custom VJP first adds the cache cotangent to the
   lagged-response cotangent induced by the initial Radau RHS/stage bars, then
   pulls that total only to the initial state.

This means the support transpose sees only one part of the total lagged
cotangent while the state route separately traverses the same initial
lagged-response dependency.  The measured costs are approximately 137 s and
138 s, whereas the forward initial-carry construction is about 28 s.

Plan:

1. Define a small, in-memory joint-contract oracle.  Given initial state,
   support, cache bars, and RHS-induced lagged bars, it must reproduce the
   VJP of an explicit state-and-support initial-carry function.  It must use no rollout,
   profiler, dump, or temporary output.
2. Add an opt-in initial-carry custom rule that exposes the **total** lagged
   cotangent once, then applies one batched NTX support transpose and derives
   the accompanying state bar from the same local response rule.
3. Keep the existing initial-cache mode and every rebuild selector untouched.
   The new route is selected independently and falls back to the established
   path by default.
4. Add separate timers for (a) RHS-to-lagged cotangent formation, (b) the
   shared initial lagged transpose, and (c) state/support result assembly.
   A remote benchmark is permitted only after the small equality oracle
   passes.

The essential constraint is that this may not merely wrap the current two
large pullbacks: that would preserve both NTX graphs and can increase compile
time and RAM.  The shared primitive must be below the split, at the initial
lagged-response pullback boundary.

**Follow-up audit correction:** the required joint primitive already exists.
`_flat_rhs_build_state_and_support_pullback_batched_interpolated_faces_factory`
wraps `pullback_build_lagged_response_state_and_support_payload_...`, which
returns both batched state bars and support bars from a single local NTX
implicit-adjoint construction.  The segment implementation can select this
hook, but the post-sweep initial-carry path does not: its custom VJP uses the
state-only `lagged_pullback_fn`, while `initial_cache_support_pullback` calls
the support-only hook separately.  Therefore the next change is not a new
NTX rule.  It is an initial-carry-specific adapter which exposes the total
initial lagged cotangent (`cache bar + RHS/stage-induced lagged bar`) to this
existing joint hook and then routes its two outputs to the existing state and
support accumulators.  This is the only route worth implementing; it avoids
the previous failed wide-joint graph by reusing the already bounded local
joint adapter.

**Three-pass feasibility audit:** this adapter is not yet usable for the
current validated fast selector.  The existing joint hooks cover only the
generic/reuse-local-VJP support variants.  The selected fast route,
`ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_`
`shared_primal_with_vmec_coefficients`, exposes a support-only multi-RHS
contract; it has no corresponding joint state-and-support hook.  Routing the
initial carry through an existing generic joint hook would therefore change
the selected NTX algebra and could reproduce the compile/RAM regressions seen
for previous joint segment selectors.

Also, the current split path sends the total lagged cotangent to the
state-only pullback, but sends only the explicit cache cotangent to the
support-only pullback.  A mathematically complete joint rule would add the
initial-RHS-induced support contribution.  It must be compared against a
generic state-and-support VJP before it can be called an optimisation of the
validated reference.  The next safe implementation step is consequently a
small native multi-RHS joint-adapter oracle, not benchmark wiring.

**Implementation status (2026-08-25):** the native joint local helper now
keeps the objective axis inside the NTX matrix-RHS call. The first draft was
incorrect because it still wrapped that helper in an outer objective `vmap`;
that would have constructed scalar NTX pullbacks and could not deliver the
intended factor reuse. The corrected path transposes only
`(objective, species, ...) -> (species, objective, ...)` at the local anchor
boundary, while preserving objective-major output bars. It remains unselected
and is not wired into the initial-carry reverse yet.

The local CPU gate
`test_native_joint_local_adapter_keeps_objective_rhs_inside_ntx_mock` passed
on 2026-08-25 (`1 passed, 33 deselected`, 10.13 s). It is a mocked in-memory
layout/contract test: it proves one native call receives all RHS per species
and that output state/support bars retain the objective axis. It does not run
an NTX solve or a transport rollout. The next required gate is a compact
state-and-support numerical oracle suitable for the remote machine; do not
select the new hook before that equality check passes.

**Numerical gate prepared, not run locally:**
`test_native_joint_local_multi_rhs_matches_explicit_state_support_vjp` compares
the new two-species/two-RHS native local joint output (prepared, `drds`, Er,
temperature, and density bars) to explicit per-species/per-RHS `jax.vjp` calls
of the same local response. It has no transport rollout or filesystem side
effects, but does execute real NTX solves. Per the local-test constraint it
must run only on the remote test machine, not WSL.

**Joint-hook propagation status:** the validated native local joint helper is
now exposed separately through `CombinedTransportFluxModel`,
`ComposedEquationSystem`, and the flattened solver physics context. The
existing rebuild selectors do not select it. The pure forwarding gate
`test_combined_native_joint_lagged_pullback_selects_native_hook_only` passed
on 2026-08-25 (`1 passed, 29 deselected`, 8.54 s). The remaining work is to
replace the post-sweep pair of initial-cache support and initial-state
pullbacks with a single manual initial-carry adapter that passes the total
lagged cotangent to this hook.

**Initial joint adapter implementation (2026-08-25):** a new, unselected
`reverse_initial_cache_support_pullback_mode` value,
`ntx_native_joint_state_and_support`, now constructs the initial carry through
the same primal path but retains a private manual reverse closure. After the
segmented sweep it forms the exact total initial lagged cotangent
`cache + RHS/stage-induced`, calls the existing native multi-RHS joint hook
once, and accumulates its state and support results together. The legacy
`scalar` and `ntx_batched_interpolated_faces` initial modes retain their exact
existing benchmark-local VJP and support-only code paths; the benchmark
delegates to the new core adapter only for the explicit new selector. The
shared-total-bar algebra gate
`test_initial_joint_lagged_pullback_retains_rhs_induced_support_bar` passed on
CPU with no transport rollout, profile, or filesystem output (`1 passed`,
2026-08-25). A remote full reverse/FD comparison is still required before
this opt-in selector can be considered validated for the benchmark.

**First remote selection attempt:** the new mode reached the post-sweep
adapter after completing all four reverse segments, then stopped before the
native joint hook because its zero RHS-lagged `lax.cond` branch had a
zero-argument signature. The branch now accepts the required operand. This is
a control-flow wiring fix only; it does not change either cotangent formula or
the established modes. Re-run the same remote command after updating NEOPAX.

**Outcome / do not select (2026-08-26):** that joint initial-carry mode did
run after the control-flow fix, but took 664.694 s for the combined initial
state/support pullback (versus roughly 137 s + 138 s for the established
split path) and raised peak RAM substantially.  It is therefore a rejected,
opt-in experiment.  The cause is structural: the native joint hook includes
the direct outer-geometry VJP in the same objective-batched compiled graph.
It is not an improvement and must not replace the default or drds-reuse mode.

### Next work: initial edge using the established rebuild dispatch (2026-08-26)

The desired route is *not* the rejected wide joint hook.  The initial cache
is a zero-time graph edge:

```
y0 -- build_lagged_response --> cache0 -- initial RHS --> carry0
```

The reverse segments correctly stop at independent bars for `y0` and
`cache0`; one transpose of the `y0 -> cache0` edge is mathematically required
after the sweep.  It does not require a root solve, a virtual accepted step,
or a second primal NTX solve.

Plan for a new, unselected mode, tentatively named
`rebuild_dispatch`:

1. Audit the exact split rebuild route used by the active
   `reverse_rebuild_support_pullback_mode`: the established state custom
   pullback plus that mode's support pullback.  Define a small structural
   oracle that proves the initial adapter supplies the same `state`, cache
   cotangent, and support payload as a rebuild edge.
2. Add an **initial-only** selector that reuses that dispatch outside the
   segment JIT.  State remains the validated state-only lagged custom VJP;
   support is selected from the active rebuild-mode hook.  Do not call the
   native state-and-support joint hook and do not put direct geometry inside a
   combined objective-batched graph.
3. Handle native VMEC coefficient bars explicitly at this boundary.  The
   current fast rebuild mode returns its VMEC face coefficients out of band;
   the initial adapter must merge them through the existing geometry bridge,
   rather than accidentally exposing them as generic support leaves.
4. Compare the new mode against the current split initial route on small
   in-memory tests.  Only after equality passes, request one remote benchmark
   to measure initial-cache, initial-state, JIT time, and peak RAM.  Defaults,
   the drds-reuse rebuild mode, and the rejected joint mode remain unchanged.

Expected scope: this can remove the *generic* initial support graph and reuse
the fast rebuild support implementation.  It cannot remove the one required
initial-cache transpose or promise a one-NTX-adjoint result; whether it lowers
the observed ~275 s must be measured before any claim of improvement.

**Implementation status:** `rebuild_dispatch` is now an explicit, unselected
initial-cache selector.  It dispatches through the active separate batched
rebuild support hook, while leaving the initial state custom VJP unchanged.
For the current VMEC-coefficient rebuild mode, it preserves the out-of-band
face-coefficient bars and accumulates them with the segment-produced channel
before the existing raw-block geometry bridge.  The small no-rollout dispatch
gate passed on WSL CPU (`1 passed, 30 deselected`, 2026-08-26).  A numerical
equivalence test against the existing initial support selector is required
before a remote benchmark.

**Initial-dispatch contract gate:** the selector is now checked for both
contracts without an NTX solve or a transport rollout: (a) the ordinary
batched result is passed through unchanged, and (b) the VMEC variant returns
the ordinary support payload and out-of-band coefficient payload separately.
The two CPU-only mock tests passed (`2 passed, 30 deselected`, 2026-08-26).
They establish dispatch and carrier semantics; the first remote run must still
compare the resulting AD table with the established drds-reuse reference.

**Three-pass review (2026-08-26):**

1. The selector maps only the already-exposed separate batched rebuild hooks;
   unsupported scalar or joint rebuild selectors fail explicitly instead of
   silently changing algebra.  Existing initial selectors do not enter this
   branch.
2. The native VMEC result initially exposed a double-counting hazard: its
   normal payload includes a generic prepared-face bar while its separate
   coefficient channel replaces that same rebuild-face term.  The selector
   now routes that payload into `step_support_bar_leaves_accum`, exactly like
   a segment rebuild, and keeps it out of the generic initial-cache
   accumulator.  The final existing merge therefore retains direct geometry
   and runtime channels once, with face coefficients supplied once by the
   native channel.
3. The selector remains outside the Radau segment JIT and does not change the
   initial state custom VJP, root path, defaults, or the rejected native joint
   mode.  Python compilation and whitespace checks pass; the small dispatch
   gate remains CPU-only and passed.  The remaining required test is numerical
   equivalence of a real initial response for the non-native batched contract,
   followed separately by the native VMEC coefficient route.
