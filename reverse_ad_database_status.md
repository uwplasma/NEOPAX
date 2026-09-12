# Database reverse-AD status

## Latest checkpoint - 2026-09-12

### Continued support-cost audit

The focused performance-mode, CLI plumbing, shared-stage Jacobian and support
reuse gate was rerun: **68 passed in 22.51 s**. Production interpolation is
unchanged and matches the forward runtime database; the TOML's
`ntss_like_midpoint` setting belongs to Er permittivity, not interpolation.
The user confirmed that preserving forward/TOML behaviour is the requirement.
See [the active support-path audit](docs/benchmarks/database_reverse_support_cost_audit.md)
for the confirmed generic interpolation dispatch and the standalone fixture's
`7 x 16 x 12` file-table versus `7 x 16 x 11` runtime-table distinction.
No new full 16/4 timing or peak-RSS result has been obtained.

The isolated stage/support cost probe now defaults to one component per fresh
process, saves lowered IR before compilation, and records fixture dimensions,
input fingerprints and synchronized samples. Its control/metadata/comparison
gate passed **22 tests in 0.75 s**, without VMEC or JAX compilation. The next
larger-machine command is in the support audit linked above.

A bounded coefficient-axis batching micro-probe matched all nine finite query
cases and reduced its compiled graph/time, but increased temporary GPU memory
(9.358 -> 10.376 MB). It is diagnostic-only and has not been installed in the
production support path. Radius-coordinate batching is still only a proposed
next experiment, not an implemented or measured speedup.

### Resumed: isolated performance selectors and initial-support candidate

Implementation has resumed after recording the completed run below. Current
defaults remain `split / shared / scalar_jvp`; full-transport-only CLI controls
also select the earlier `generic / separate / radial_vjp` paths.
An opt-in `reduced_zero` initial-support mode avoids tracing the support
transpose of structurally zero initial predictor-stage cotangents. It does not
skip the initial state, ambipolar-root, profile, or recorded-scan pullbacks.

See [performance modes and exact commands](docs/benchmarks/database_reverse_performance_modes.md).
The current 22.264 s warm segment reference has not yet been improved by these
selectors. Component-level exact-block/Jacobian timing is being evaluated
separately; no 6 s full-reverse or reduced peak-RSS claim is established.
No default Radau derivative, root optimization JIT boundary, or reusable cache
has been changed.

The fourth control, `--reverse-database-stage-jacobian-mode`, defaults to
`independent` and adds the opt-in `shared` exact-Jacobian reuse candidate.
The completed local 51-radius stage-pair probe gave exactly equal finite
adjoints, warm medians 0.681442 -> 0.415849 s, and compiled temporary **device**
storage 2.429 -> 1.972 GB. These are not full four-step segment timings or
process host-RSS values. Raw warm samples and limitations are archived in
[`database_stage_cost_2026-09-12.json`](docs/benchmarks/database_stage_cost_2026-09-12.json).
The subsequent support compilation in that multi-kernel probe hit the local
WSL host limit; no successful support timing or new full-run peak RSS exists.

Validation so far: the focused shared-stage, initial/root preservation,
selector/callback and support-reuse gate passed **67 tests in 23.76 s**.
The real-table centre geometry plus existing Radau/root-JIT gate separately
passed **19 tests in 207.21 s**. No new expensive full 16/4 run was launched.
After the final CLI guard, **23 CLI/Radau/root-JIT cases passed in 13.20 s**.
The actual production shared-J helper also passed a five-radius GPU real-table
exact-equality check. The remaining support-compiler problem was confirmed by
a fresh small probe approaching the local host limit; it was safely stopped,
not promoted as a successful timing/memory result. No probe process remains.

### Completed performance run recorded (preceding pause)

The latest completed no-diagnostics 16/4 run is now saved with all 17 residuals,
136 Jacobian entries, all progress-phase timings, the executed command, and
resource counters at the top of
[`reverse_ad_vs_fd_database.md`](reverse_ad_vs_fd_database.md).
Verbatim baseline and new run archives are in `docs/benchmarks/`.
This result supersedes the pending-result statements below.

| Completed-run metric | Before | After |
| --- | ---: | ---: |
| Whole-process wall time | 1:19:27 | 1:18:20 |
| Benchmark internal elapsed | 3805.329 s | 3776.534 s |
| Peak host RSS | 15051068 KiB (14.354 GiB) | 14486496 KiB (13.815 GiB) |
| Warm four-step segment mean | 22.813 s | 22.264 s |
| Segment sweep, including first-call overhead | 1136.374 s | 1076.337 s |
| Initial direct-RHS support | 296.198 s | 568.883 s |
| Final recorded-scan fold | 770.232 s | 707.702 s |

Overall: 67 s (1.405%) less wall time and 551.340 MiB (3.751%) less peak host
RSS in this comparison. The initial-support phase is still 272.685 s slower.
The earlier +112.525 s figure described an incomplete printed-phase subtotal,
not the final wall time. No large or repeatable speedup is established.

All 17 residuals and all 102 profile-column entries match exactly at printed
precision. All 136 derivatives are finite. Maximum transport-Jacobian relative
change is 2.806420e-14; maximum across all rows is 9.929343e-11. The existing
AD-FD conclusions are unchanged, including the remaining bootstrap geometry
relative errors of approximately 2.404e-3 (RBC) and 3.413e-3 (ZBS).

The user paused implementation to record these results. No production code or
CLI options were changed in this recording step. The next authorized task,
when resumed, is to restore an independently selectable pre-change performance
baseline and isolate current candidates behind CLI flags before more tuning.
The target is approximately 6 s warm database reverse against approximately
2 s forward; the Lij 220 s / 600 s example is a ratio reference, not an absolute
database timing comparison. The 22.264 s warm segment result does not meet it.
Preserve the existing derivative fixes, root optimization JIT boundaries,
unrelated changes and caches; do not revert whole commits.

This checkpoint supersedes the historical implementation/failure notes below.
The detailed derivative tables and latest performance audit are in
[`reverse_ad_vs_fd_database.md`](reverse_ad_vs_fd_database.md).

### Current task and constraints

Reduce both host memory and execution/compilation time for the black-box
runtime-database full-transport reverse benchmark: 16 accepted steps,
4-step segments, all objectives, profiles plus `RBC:1:0` and `ZBS:1:0`.
Preserve the validated derivatives, full nonlocal coupling, finite Radau
Jacobian contract, root optimization lane and reusable JAX caches.
Do not clear caches to trade speed for memory; do not revert whole commits.
Colored solve modes may remain isolated options, but are not the active plan.

Keep `block` / `explicit_database` / `grouped_vjp` as the benchmark reference.
The active derivative split includes local physical-geometry contributions
and database table/query-coordinate contributions; the early table-only
description below is historical, not the current contract.

### Latest verification and pending candidates

- Latest user full-project gate: **8 passed in 185.95 s (3:05)** for
  `tests/test_database_center_geometry.py` and
  `tests/test_database_support_reuse.py`. This confirms the centre-geometry
  and shared-support regressions in the actual project environment. It is
  not a full test-suite pass or a new GPU timing/AD-FD measurement.
- User reported the original four candidate tests passing in 13.50 s.
- The subsequent audit fixed an integration gap in the optional
  `block_database_multi_rhs` selector: its solve used the finite generic
  Jacobian, but its outgoing state pullback could select the compact database
  state VJP. Both exact block layouts now use the same finite Jacobian contract.
- The user then ran the actual repository selector regressions:
  **7 passed, 122 deselected in 10.60 s**. Coverage includes both selectors'
  matrix/carry dispatch and a JIT-compiled coupled nonlinear three-stage
  solve plus input pullback against an independent residual VJP.
- The optional `grouped_joint_vjp` candidate shares ordinary terminal
  state/geometry objective work; it leaves bootstrap unchanged. The initial
  direct-RHS support reuse is covered by the earlier four-test result.
- No new long benchmark, AD/FD comparison or measured speed/RSS improvement
  is established by these small tests.

The earlier selector regression command was:

```bash
JAX_PLATFORMS=cpu PYTHONPATH="$HOME/VMEX:$HOME/NTX/src" \
python -m pytest tests/test_solvers.py -q -p no:cacheprovider \
-k 'database_block_multi_rhs or database_exact_block_solve_and_carry_pullback or database_plain_block' \
--maxfail=1
```

### Performance conclusion and next action

Do not claim that `block_database_multi_rhs` avoids ten LU factorizations:
installed JAX 0.5.0 CPU lowering showed that the original mapped `block`
already shares one factorization across objective rows. The explicit-column
layout remains optional, with no demonstrated performance advantage.

The reference run's warm four-step reverse segments took 22.907, 23.099 and
22.433 s; the first segment took 1067.933 s including first-call overhead.
Peak host RSS was 15051068 KiB (14.354 GiB). These are pre-measurement
reference values, not improvements from the candidates.

### Implemented next step: call-local shared support preparation

The three built-in database support partials now share one preparation inside
`pullback_direct_rhs_database_split_support_payload`. Owner binding, working
state, center fluxes and the two distinct native-face payloads are prepared
once rather than three times; the equation-to-flux VJP is built once rather
than twice. The table/query-coordinate, local flux geometry and equation
geometry contraction rules are unchanged. Nothing is retained between steps.

The private prepared record is used only when the three built-in public
methods are unoverridden and the equation system is concrete. Standalone
methods, partial fixtures and public-hook overrides keep the old path. The
root-only lane and Radau matrix/state-adjoint code are not changed by this step.
The existing distinction between `eidx=None` for equation-to-flux bars and
the actual prepared electron index for equation-geometry bars is preserved.

Four new cases in `tests/test_database_support_reuse.py` cover JIT/vmap
equivalence against the three standalone partials, dynamic state/support,
separate density/temperature faces, coordinate bars, trace counts and all
three public-hook overrides. These plus five existing boundary regressions
passed in an exact-source CPU harness: **9 passed in 5.26 s**. This is not a
full repository-import test or production GPU validation.

Small JAX 0.5.0 CPU lowering comparison: 557 -> 401 traced equations;
compiled temporary buffers remain 856 bytes in both versions. Optimized
graphs are essentially the same size. This establishes reduced tracing work,
not a demonstrated warm-segment speedup or lower benchmark peak RSS. No caches
were cleared, no dense derivative replacement was introduced, and no physics
contributions were removed. The ~23 s warm-segment cost remains unresolved.

User full-project validation is now recorded: **6 passed, 100 deselected in
14.06 s**, selecting the shared-primal tests and the existing explicit-boundary
and split-versus-generic VJP assembly tests. This validates that integration;
it does not measure the production benchmark's speed or RSS.

Next: continue examining the actual compiled support JIT / stage-scan work
for the warm cost.
Keep density/temperature closures distinct and retain all geometry/table
coordinate bars. Do not substitute the frozen forward Jacobian or request a
long run solely to test the unsupported factorization-count hypothesis.

### Direct-centre physical-mesh derivative follow-up

The built-in `NTXDatabaseTransportModel` centre flux uses local geometry only
through `r_grid`, `r_grid_half`, and `dr`. The established constrained geometry
map derives all three from scalar `a_b` with fixed normalized coordinates.
The implementation replaces the per-radius flattened-geometry VJP with
one scalar JVP of the existing direct-centre evaluator, followed by contraction
with the Gamma/Q/Upar objective bars. It keeps the entire `Monoenergetic`
payload fixed, including its scale and coordinate fields. Their sibling
pullback is unchanged. No Radau, face, root, source, or cache changes.

This dispatch is restricted to the exact built-in model and a complete
physical-mesh dataclass. Custom evaluators, nonphysical geometry stand-ins and
legacy preprocessed databases retain the original VJP body. All returned
geometry leaves retain their objective batch dimensions. Nonfinite diagnostics
remain available; no nonfinite value is replaced with zero.

New regression file: `tests/test_database_center_geometry.py`. Its four cases
passed in an isolated actual-source CPU harness: **4 passed in 175.42 s**,
using JAX 0.5.0, interpax 0.3.7 and equinox 0.11.12. These compare against both
the original radius VJP and the actual direct-centre forward VJP, including
ten independent objective rows, heat-only cotangents and a custom evaluator
with an additional geometry dependency. The existing shared-support harness
also passed again: **9 passed in 5.06 s**. Full project imports and the GPU
benchmark are not validated locally by this harness.

Representative-shape CPU measurements used four species, 51 radii, four
energies, a 7x16x11 `Monoenergetic` table, and ten independent objective bars.
State, geometry, cotangents and database were dynamic JIT arguments; the
table was not constant-folded. Profiles, table entries and quadrature weights
were synthetic, evaluated by the actual collision/interpolation/flux kernels.
The final host-memory comparison used separate fresh processes (no cache
clearing). Warm numbers are medians of nine synchronized calls.

| CPU component measurement | Original radius VJP | Scalar mesh JVP |
| --- | ---: | ---: |
| Trace/lower time | 4.088 s | 2.813 s |
| Compilation time | 45.288 s | 4.095 s |
| Warm execution | 35.039 ms | 3.720 ms |
| Peak process host RSS | 2,158,880 KiB (2.06 GiB) | 865,552 KiB (0.83 GiB) |
| Compiled temporary buffers | 780,576 B | 1,566,632 B |
| Compiled output buffers | 20,864 B | 20,864 B |

The temporary-array increase is about 0.75 MiB; it is **not** a host-memory
saving. The measured whole-process host peak, including compilation, falls
by approximately 60%, while compilation and warm time also fall. Bounded
radius-map variants were investigated: they reduced temporary arrays but
retained roughly 45-54 s compilation and were slower than the all-radii JVP.
Those variants are not retained. The final implementation is the vectorized
scalar JVP that passed the four real-kernel tests above; representative-shape
scalar/ten-row comparisons also remained finite and matched the old VJP.

These are isolated CPU component measurements, not the full GPU 16/4 run.
They do not predict a 60% drop in the benchmark's 14.35 GiB RSS or establish
the new total segment time. The remaining table/query-coordinate transposes,
terminal objectives, initial support/root and final recorded scan fold still
require their own audit/measurements. Keep the existing benchmark command and
physics options; no new mode or diagnostic flag is required for this change.

The user has now passed the full-project regression gate below:
**8 passed in 185.95 s (3:05)**. This supersedes the isolated-import limitation
for these eight tests, without changing the scope of the CPU component
performance measurements above.

```bash
JAX_PLATFORMS=cpu PYTHONPATH="$HOME/VMEX:$HOME/NTX/src" \
python -m pytest \
  tests/test_database_center_geometry.py \
  tests/test_database_support_reuse.py \
  -q -p no:cacheprovider --maxfail=1
```

Next validation: the unchanged no-diagnostic 16/4 GPU benchmark, comparing
all objective derivatives to the saved reference, first/warm segment timings
and `/usr/bin/time -v` peak host RSS. No new long-run result has been supplied
after this optimization yet. No production code was changed when recording
this eight-test result.

### Workspace handoff

Branch: `en/reverse_ad_improvement`. During this implementation HEAD advanced
concurrently from `936fe39` to `7ac3d92`; that commit includes the main shared
preparation and new tests. Do not revert it: it also contains unrelated
optimization work. The final compact-hook capability guard and these updated
notes, centre-kernel optimization and its tests are later working-tree edits
at this checkpoint. No commit/push was
performed by the implementation agent.

## Historical notes (superseded where inconsistent with the checkpoint)

## Scope

This note records the current `ntx_scan_runtime` / black-box database
reverse-AD implementation.  It does not change or describe the established
Lij (`ntx_exact_lagged_runtime`) reverse path.

## Intended reverse boundary

```text
VMEC geometry
  -> one recorded NTX scan
  -> fixed D11/D13/D33 database tables
  -> initial root, forward transport, terminal objectives, reverse segments
  -> accumulated table cotangents
  -> one recorded NTX scan transpose
  -> VMEC cotangent
```

Database transport segments retain only fixed tables.  They do not retain the
scan owner, scan primal, prepared scan systems, or a callable database builder.

## Current implementation

- The database direct-RHS support transpose returns table bars only.
- The database segment reverse uses the same bounded Radau replay and
  reduced-cotangent architecture as the Lij lane, with a fixed-table support
  leaf.
- The retired database-only raw transport-metric geometry VJP and deferred
  post-segment geometry sweep have been removed from the active reverse path.
- The database initial-Er root now also produces table bars only.  It does not
  run a separate local-radius raw geometry pullback.
- Table bars from all reverse contributions are folded through the retained
  scan once, after the complete reverse sweep.

## Diagnosed failure and correction

The old path produced nonfinite `r_grid_half` cotangents in the deferred raw
transport-metric geometry VJP (`fixed_flux_equation`).  Segment diagnostics
showed that all stage-adjoint and fixed-table cotangent values were finite;
the nonfinite value arose only in that retired geometry sweep.

The first one-step run after removing that sweep passed all segment finite
diagnostics and reached the initial-Er root boundary.  It then stopped with:

```text
Initial-Er root support pullback leaf shape mismatch at leaf 19:
got (), expected (10,)
```

Cause: the newly zeroed root geometry tree did not have the leading objective
axis.  The current unvalidated correction broadcasts that zero tree to
`(objective_count, ...)` before it is combined with database table bars.
This is a shape correction only; it does not restore a geometry VJP.

The next one-step run passed that shape boundary and completed the one batched
recorded-scan fold.  It then ran out of GPU memory in the *final VMEC
payload-to-state transpose*, after the scan fold:

```text
runtime_scan_payload_from_state
  -> 56 active float payload leaves
  -> raw payload-to-state VJP batch
  -> attempted 3.73 GiB allocation
```

The scan fold itself completed; this is not a scan reverse or table-bar OOM.
The raw fallback built one retained payload VJP per objective row.  The
database scan lane had the existing compact JVP/tangent-contraction route
explicitly disabled.  That guard has now been removed for the combined scan
payload.  The new route keeps geometry, scan channels, and scan surfaces as
one coupled function and computes, for each VMEC parameter tangent,
the JVP/bar contraction.  By VJP/JVP duality this is the same derivative as
the raw payload-state transpose, without materializing its large state-bar
batch.  This correction is not yet GPU-validated.

The first compact-JVP attempt reached that route and exposed a trace-safety
bug, not a physics or memory failure: the static `ntx_scan_rho` validation
used `bool(jnp.all(...))` inside the JVP.  Scan locations are configuration
coordinates, so they are now validated and captured as a host NumPy constant
before the differentiated payload function is entered.  VMEC state still
drives geometry, channels, and surfaces; only the fixed scan coordinate axis
is excluded from differentiation.  This correction is not yet GPU-validated.

The next compact-JVP attempt passed the scan-coordinate validation and then
found a second static-data leak while deriving the Boozer `R00` normalization:
the scan surface-index metadata was recomputed from `context.static.s` inside
the JVP.  That metadata is now precomputed from the fixed scan radii and
VMEC grid outside the differentiated payload function and passed to the R00
builder, together with already-prepared Boozer constants/mode metadata.  The
R00 values themselves remain functions of the VMEC state.  This correction is
not yet GPU-validated.

## Validation completed

```text
database_direct_rhs_support_stops_at_fixed_table_boundary                 PASS
database_initial_root_support_batches_charge_weighted_particle_bars       PASS
recorded_ntx_database_bar_groups_share_one_batched_scan_pullback          PASS
```

The three tests passed together on CPU (`3 passed, 147 deselected`).  Python
syntax checks passed after the subsequent objective-axis broadcast correction.

## Required next validation

Run the one-step GPU diagnostic benchmark with
`--reverse-segment-input-diagnostics` using the database black-box TOML.  The
expected result is:

- no `database-geometry-reverse` output;
- no `database post-segment geometry sweep` output;
- all database reverse trace values finite;
- no initial-root support leaf-shape error;
- root timing diagnostics report `direct_geometry_transpose_s=n/a`.
- `compact_payload_tangent_contract=True` after the recorded scan fold;
- no raw payload-state-bar allocation / `RESOURCE_EXHAUSTED` error.

Only after that passes should the 16-step / four-segment GPU benchmark be run.

## 2026-09-09 update: compact native face-geometry boundary

This section supersedes the older statement above that the database reverse
has no local geometry contribution.  The correct fixed-table decomposition is
now:

```text
database RHS bar
  -> table bar                    -> accumulated -> one recorded scan VJP
  -> direct centre-flux geometry  -> local VMEC/transport geometry bar
  -> direct native face geometry  -> local VMEC/transport geometry bar
  -> fixed-flux equation geometry -> local VMEC/transport geometry bar
```

Only the table bar reaches the recorded scan.  The three geometry terms are
local and must never capture the scan owner.

### Face geometry correction

The first bounded implementation performed a `jax.vjp` through a newly built
`ComposedEquationSystem` for each native density/temperature face.  It was
finite after the axis-mesh correction but still used about 16–17 GB host RAM
on a 30 GB machine because every local VJP retained equation/source assembly.

The active replacement is
`pullback_direct_face_flux_geometry_by_radius`:

- routed through the database, runtime-scan, and combined flux models;
- evaluates only the native fixed-table database face flux primitive;
- excludes `ComposedEquationSystem`, sources, finite-volume equation
  assembly, and the scan owner;
- follows the same local, bounded scan-over-radius structure as the existing
  direct-centre database geometry primitive;
- is selected only by the database split support hook, never the Lij path.

### Physical radial-mesh rule

The database direct-flux primitive varies the physical VMEC mesh only through
`a_b`:

```text
r_grid = rho_grid * a_b
r_grid_half = rho_grid_half * a_b
dr = r_grid_half[1] - r_grid_half[0]
```

`r_grid_half[0]` is the fixed magnetic axis and is not an independent
tangent direction.  Leaving it free produced nonfinite
`geometry.r_grid_half` bars by evaluating the radial database off its
physical manifold.  This constraint applies to direct database fluxes
(centre and faces), not the fixed-flux equation-assembly geometry VJP.

### Latest validation

```text
database_face_geometry_pullback_keeps_table_fixed_on_physical_mesh       PASS
database_face_geometry_pullback_selects_compact_model_boundary            PASS
database_face_table_pullback_rebinds_only_fixed_database_leaf             PASS
batched_database_stage_table_pullback_accepts_flattened_radau_rows        PASS
recorded_ntx_database_bar_groups_share_one_batched_scan_pullback          PASS

5 passed, 282 deselected in 13.67s
```

### Required next measurement

Run the 16 accepted-step / 4-segment GPU diagnostic benchmark.  Require both:

1. no nonfinite `geometry.r_grid_half` segment bar; and
2. a host-RAM peak below the previous roughly 16–17 GB.

The tests establish dispatch and boundary correctness; they do not yet claim
a measured benchmark memory reduction.

## 2026-09-11 current full-transport diagnostic state

The one-step, one-segment database black-box full-transport reverse completed
after the compact face-state transpose was added.  This run used both
`NEOPAX_DATABASE_STATE_VJP_DIAGNOSTICS=1` and
`NEOPAX_DATABASE_GEOMETRY_VJP_DIAGNOSTICS=1`.

### Finite-status result

The previously failing local boundaries were finite throughout the attached
run:

- fixed-flux equation geometry: `a_b_nonfinite=0`,
  `r_grid_half_nonfinite=0`;
- direct centre-flux geometry: `a_b_nonfinite=0`,
  `r_grid_half_nonfinite=0`;
- compact face-flux geometry: `a_b_nonfinite=0`;
- final recorded scan fold:
  `raw_block_param_bar_all_finite=True`, with
  `raw_block_param_bar_l2=4.578905e+02`.

No `nan`, `inf`, traceback, or nonfinite diagnostic was emitted.  Thus the
compact face-state correction has removed the earlier nonfinite path for this
one-step diagnostic; it has not been rolled back and the direct-state Radau
correction remains active.

### Timing result and remaining blocker

The run is finite but still far too expensive, so it is not yet a successful
16-step production validation:

```text
one active-step segmented cotangent sweep       1948.746 s
initial direct-RHS support pullback              342.165 s
initial state pullback                           162.343 s
initial-Er root compact pullback                 267.393 s
final recorded-scan fold                         733.465 s
```

The active compact contract remains intact:
`ntx_scan_runtime_active_float_leaves=63`,
`compact_payload_tangent_contract=True`, and one batched recorded-scan
transpose.  The next task is therefore performance/memory auditing of these
retained reverse payloads and local face-state calculations, not another
change to the finite geometry/table ownership or a generic scan VJP fallback.
