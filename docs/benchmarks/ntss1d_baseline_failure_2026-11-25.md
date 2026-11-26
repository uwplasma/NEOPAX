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
