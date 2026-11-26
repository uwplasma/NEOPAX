# Captured NTSS1D baseline failure: geometry-to-field normalization

Run: `outputs/ntss1d_baseline_failure_FBW8sj`, NEOPAX `7c87820`, JAX
0.10.0, CUDA, FP64. Builds 1 and 2 passed the original baseline-support
guard. Build 3 failed it. All three run the original density-shape-alpha
initialization without transport evolution or FD endpoints. Analysis uses
the transferred actual scan arrays and compiler text, with no local JAX
import or new coefficient solve.

## Location of the reproduced difference

Builds 1 and 2 have identical transport geometry, channels, native surfaces,
all seven prepared NTX systems, scan arguments and raw coefficient arrays.

Build 3 has the same config and loaded Python source hashes. Its 147 native
VMEC surface fields, all seven prepared NTX systems, derivative matrices,
input types and collisionalities match build 1 exactly. Nine transport
geometry fields differ, beginning with Boozer-derived quantities including
`B0`. Five of the seven channel `b00` values differ, with maximum absolute
difference `1.7763568394002505e-15`.

The only changed numerical argument supplied to the NTX scans is `Es`:
29 of 77 independent radius/field values change. Repetition over the 16
collision frequencies gives 464 changed scan rows. These are exactly the
464 changed D13 positions. Identical-input rows reproduce their coefficients.

| Radius index | Changed Es rows | Changed D13 rows |
| --- | ---: | ---: |
| 0 | 80 | 80 |
| 1 | 0 | 0 |
| 2 | 144 | 144 |
| 3 | 0 | 0 |
| 4 | 32 | 32 |
| 5 | 144 | 144 |
| 6 | 64 | 64 |

At `(6,0,7)`, `Es` changes from `0x1.c035ba9ccdabep-7` to
`0x1.c035ba9ccdabfp-7`. Raw D13 changes from `2.89369301593999` to
`2.890787864651262`. The maximum normalized D13 table difference is
`1.9552908040003647e-3`.

## Arithmetic propagation through the normalization

The saved scalar values reproduce the current operations byte-for-byte:

```text
dr_tildedr = 2 * psia / (a_b**2 * b00)
dr_tildeds = dr_tildedr * drds
Es = (Er_tilde * dr_tildeds) * b00
```

In exact arithmetic, `b00` cancels. Input-dependent rounding in the division
and subsequent multiplications leaves last-bit field differences instead.
Using the algebraically cancelled expression on the saved scalar inputs
produces identical fields in all three builds, because `psia`, `a_b` and
`drds` are identical. This is an offline arithmetic check, not a rerun or
validation of a patched NEOPAX forward or derivative path.

## Compiler comparison and limits

All 71 captured preparation/scan StableHLO programs align between the builds.
The 70 preparation programs have identical arithmetic/backend text after
debug metadata removal. Scan backend dispatch choices differ, including
transpose fusions, but its 1,232 mathematical HLO instructions match after
removing backend dispatch and debug metadata. Builds 1 and 2 also have
dispatch differences while giving identical full raw arrays. Thus the
presence of different kernels alone does not establish the failure's cause.

This capture locates an upstream input change rather than different NTX
results for the same complete numerical inputs. It does not identify the
operation that first changes the Boozer-derived geometry. The Boozer
transform and its upstream input construction were not captured here.
Cancelling `b00` in the field formula alone would not make the strict
geometry guard pass, since those geometry arrays also differ.

No LU correction, solver change, guard relaxation or baseline substitution
has been applied. This explains the newly captured failure path, not every
historical failure: the older `fixed_projection_ad_fd16_gu6ZO2` failure
had matching saved geometry and all 1,232 D13 entries differed, including
zero-field nodes. It cannot be assigned the same complete cause from the
older final-table-only capture.

## Capture the missing upstream boundary

`--geometry-input-only-dir DIR --geometry-input-compiler-ir` now observes
the original profile FD initializer and stops at its pre-NTX scan-input
boundary. It keeps the original equilibrium solve and its compilation
history. It runs no NTX coefficient scan, initial-root construction,
transport evolution, adjoint or FD endpoint.

`DIR/geometry/boundaries.json` lists the actual geometry calls, VMEC-to-Boozer
input construction, fixed-order Fourier projections, and Boozer transforms.
Each boundary file retains the actual input/output arrays; the geometry
boundary also retains the original equilibrium state. Copies occur only
after the original prefix returns or stops. Each compiler folder contains
the programs compiled during the actual call, without another lowering or
replay. `loaded_sources.json` records the loaded NEOPAX, VMEX, Boozer and
SOLVAX Python sources, including the installed Boozer API rather than an
assumed local checkout version.

The existing NTX compiler-capture mode also includes these upstream
observations when its NEOPAX geometry module is loaded. Normal execution
does not activate them. The geometry-only early return uses the existing
pre-transport marker so the fixed-mean wrapper correctly recognizes that
no time solver was reached. The reference guard remains unchanged.

Local NumPy/mock tests check original arguments, result identity, deferred
copies, failure evidence, hook restoration and the original CLI prefix's
stop before NTX. No correction or algebraic rewrite is activated by this
diagnostic. It is required to distinguish a change in the Boozer transform
inputs from a change inside the transform or its later interpolation.

## Upstream capture result, 2026-11-26

Archive `boozer_geometry_drift_0uRP9E.tar.gz` contains three independent
geometry-only initializations at NEOPAX `781fa68`. Every one of the 23
captured boundaries matches byte-for-byte across all three builds: the
full equilibrium state, both VMEC-to-Boozer input bundles, 18 Fourier
projections, and the two Boozer transforms' arguments and outputs. The
six equilibrium-state arrays also match the archived AD compact state.
All recorded loaded-source hashes agree between builds.

The 297 captured compiler records align. Nine records differ in compiled
backend text per build pair, including transpose-loop unroll choices and
reduction dispatch. These differences produced identical captured results.
Both Boozer mathematical programs match, and their cuBLASLt algorithm
selections agree. Consequently these passing captures do not establish
that the observed compiler differences cause the earlier geometry drift.
They did not reproduce that drift.

The exact installed Boozer API has SHA256
`0ebfddebdcd4522cb2c08aa334e92397a79a719befb43dce6244be6a0190b3cb`.
Its full transform uses ordinary Fourier matrix contractions, including
cuBLASLt FP64 products in the captured program. The split magnetic-only
projection in the source is a different path. Neither this source review
nor three equal outputs proves or disproves nondeterminism in those
products. The first operation responsible for the failed build remains
unidentified; no correction is established.

### Identical-input Boozer replay

`check_saved_boozer_repeatability.py` takes one completed actual Boozer-call
capture. Two fresh, sequential CUDA processes each call the original
`booz_xform_from_inputs` API 20 times using the same restored argument
objects. It retains the original input shapes, dtypes, static constants,
surface indices, and weak array types. In particular, the captured angular
grids are weakly typed; silently strengthening them would change the JIT
signature. Byte/type checks run before the first transform, and recorded
Boozer source hashes must match the loaded code.

The replay saves first outputs, the first changed output if any, all
output comparisons, and actual compiler programs. It requires exactly
one compilation of `booz_xform_jax_impl` per worker. Within-worker changes
would therefore be repeated-execution differences, not different input
arrays or different JIT signatures. Between-process and archived-output
comparisons are reported separately. A passing replay would narrow this
control only; it would not invalidate the earlier failed initialization
or establish AD/FD gradient accuracy.

No extra outer JIT, compiler options, solver changes, equilibrium, NTX,
root, transport, or derivative calculation is introduced. Local tests
use NumPy and mocks only. Both real saved Boozer captures passed the local
argument/schema preflight before the CUDA replay below.

### CUDA replay result

Run `outputs/boozer_saved_repeat_YQODRN`, NEOPAX `18c5b99`, JAX 0.10.0,
CUDA FP64, completed with exit status 0. Each of two fresh processes
compiled the original Boozer function once and evaluated it 20 times.
All 40 outputs match the original saved capture byte-for-byte. There
are no within-process or between-process output differences. This
control provides no evidence of a nondeterministic original Boozer
call for these arguments. It does not reproduce or repair the earlier
failed full initialization.

Source review distinguishes the transform from its consumers:
`_build_neopax_geometry_from_state` takes `bmnc_b[:, mode00]` from the
full Boozer output and interpolates it from `sample_rho` onto the
transport cell centers to form `B0`. Then
`_build_ntx_runtime_channels_from_geometry` interpolates `B0` onto the
NTX radii to form channel `b00`. Both use the configured
`interpax.Interpolator1D`; the 40-call Boozer replay does not execute
either interpolation. These operations are candidates to distinguish
in a failed capture, not established causes.

The original initialization's `--ntx-scan-audit-compiler-ir` mode now
also captures the actual geometry and Boozer boundaries. A build
that fails its original support guard is needed to compare transform
inputs, transform outputs and resulting geometry in that execution.
The existing passing geometry captures and isolated replay cannot
provide that missing failed boundary. No numerical correction,
guard relaxation or solver-method change follows from these passing
results.

### Original initialization result and remaining FD columns

Run `outputs/original_geometry_failure_Vpk7oO` completed three original
NTSS1D initializations with geometry/Boozer and actual NTX scan capture
enabled. All three passed the original support guard: transport geometry
and coefficient tables match archived AD exactly. Each fresh initial Er
also matches saved AD with maximum absolute difference `0.0`. At `(6,0,7)`
all three report `Es=0x1.c035ba9ccdabep-7` and raw D13
`2.89369301593999`. These runs stop before transport and FD endpoints.
They did not reproduce the earlier failure; they do not establish a fix.

The original generic comparison already contains all 104 pairs. The saved
NTSS1D run contains AD16 and four completed profile FD columns, or 52 pairs.
Its missing FD columns are density shape alpha, temperature shape alpha,
RBC:1:0 and ZBS:1:0. The NEOPAX package and AD benchmark source are unchanged
between archived `6cbe394` and `18c5b99`; FD benchmark changes concern
diagnostic observation and early stopping. The ordinary runner still
rejects resuming an archived run after its source fingerprint changes.

`complete_saved_transport_fd.py` instead creates a separate follow-up
against saved AD. It copies completed summaries/snapshots, records their
provenance, and uses the existing command builder for only the missing
FD columns. The original directory is untouched. Saved config, FD
increments, step limit and solver-mean scope are retained. Installed
package versions must match the saved AD metadata. Original baseline
support and exact state/time guards remain in effect. Its output table
combines archived pairs and follow-up pairs; it is explicitly labeled
as a saved-AD comparison, not a fresh all-worker run.

`--ntx-scan-audit-continue` retains actual upstream geometry/Boozer and NTX
baseline evidence, then proceeds through the same original FD calculation.
It requires an AD reference and cannot combine with the existing early-stop
audit option. Captured baseline arrays are not substituted or used to
relax a guard. The capture context exits before transport, releasing its
temporary retained records. If a future worker fails a guard, its actual
upstream evidence is retained without rerunning the failed initialization.

Local NumPy/AST/mock checks cover one-ULP guard failure in both audit
modes, unchanged transport arguments on continuation, four-worker planning,
baseline validation and preservation of the original run. A plan-only
read of the real archived case selects exactly the four missing FD columns.
No local JAX import or physics calculation was performed.
