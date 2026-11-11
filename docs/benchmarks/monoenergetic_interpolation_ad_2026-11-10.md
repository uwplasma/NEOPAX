# Monoenergetic interpolation AD preparation

This work is on `en/v0_main`. The cleanup checkout is not modified. It
prepares the eight registered coefficient-interpolation methods for query,
table and coordinate AD, without changing their interpolation rules or
promoting one physical interpolation model over another.

## Derivative implementation

For a scalar query `q = (r, nu/v, Er/v)` and database leaves `D`, the rule is

\[
\dot f = f_{q_1}\dot q_1 + f_{q_2}\dot q_2 + f_{q_3}\dot q_3
       + (\partial_D f)\dot D.
\]

The three scalar query derivatives are evaluated together before multiplying the
state directions. A dense 459-direction state Jacobian therefore does not
expand those directions through every table stencil and limiter. Active
table, knot and fit directions differentiate the original JAX expression.
Nothing is stopped or approximated. Integer and Boolean metadata receive
proper `float0` tangent zeros, rather than numerical integer zeros.

`generic_3d` additionally uses a local 4x4x4 native C1 tensor stencil instead
of constructing three full-table Interpax interpolators. Its values, knot
and coefficient VJPs, extrapolation and mixed Hessians are compared with
Interpax in the small-array tests. Its electric-field coordinate convention
is unchanged.

The fixed-width NTSS method now explicitly rejects electric-field tables
with any count other than 12. This prevents silently applying its hard-coded
12-node formulas to the 11-node live-scan benchmark. The generic root-only
sparse-coordinate shortcut is also restricted to the generic method;
the cubic 3D and no-radius variants retain their own complete coordinate VJP.
The cubic 3D array loader also now uses immutable, vectorized scaling, matching
the file loader; its previous invalid shape lookup and in-place writes prevented
building this database from JAX arrays.

## Component comparison

Run with JAX 0.10.0, x64 and a GPU. Each method/implementation runs in a fresh
process. No transport, ambipolar root, NTX or VMEC solve runs. The fixture
uses the saved `Dij_NTX_QI_Er_Optimized.h5` coefficients, a common synthetic
51-radius/four-species/four-energy profile and synthetic NTSS fit geometry.
These measurements are not a physical equilibrium or NTSS agreement check.

```bash
JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=true \
python examples/benchmarks/benchmark_monoenergetic_interpolation_methods.py \
  --repeats 5 \
  --out-dir outputs/interpolation_methods_ad
```

The `uncontracted` reference differentiates the same interpolation expression
without the outer query contraction. For `generic_3d` this reference already
contains the new native tensor primitive; its timing is not the old Interpax
3D timing. The independent Interpax comparison is numerical, in the tests.

The checks cover query JVPs versus central FD at decreasing perturbations,
query VJPs versus JVPs, coefficient VJPs versus FD, zero-radius/floor,
negative-field/extrapolation cases, and seven-stage particle/heat flux state
Jacobians. Returned arrays must agree before/after at `rtol=atol=3e-10`.
FD points avoid knots: a central difference crossing two interpolation
branches is not a derivative of either individual branch. NTSS coefficient
FD also needs a perturbation-convergence check because its limiters depend
on the coefficient values.

CSV files distinguish compiled temporary workspace, allocator peak/reserved
memory, compilation time and synchronized repeated execution. A default
2 GiB temporary-workspace ceiling prevents executing an oversized local
reference Jacobian. Such a reference is recorded as skipped and replaced by
three single-direction JVP checks against the optimized full Jacobian. It
is not reported as a completed full reference execution. The optimized full
Jacobian must fit the ceiling. Worker failures and timeouts remain failures.
No XLA flags are required by the kernels.

## Measured GPU results

Measured on an RTX 5060 Laptop GPU, JAX/jaxlib 0.10.0 and x64, with default
CUDA compiler settings. The final optimized timings use 15 warm repetitions;
the saved uncontracted reference uses five. The common flux component forms
seven dense 408-by-459 state Jacobians, with the database supplied as a dynamic
argument. All eight optimized components completed.

| Method | Temporary MiB, before -> after | Warm ms, before -> after | Optimized peak allocation, MiB |
| --- | ---: | ---: | ---: |
| `generic` | 317.2 -> 152.6 | 21.76 -> 12.73 | 592.1 |
| `generic_3d` | 147.7 -> 152.6 | 14.11 -> 12.43 | 592.1 |
| `generic_loger_no_r` | 323.3 -> 152.9 | 77.67 -> 12.99 | 592.1 |
| `preprocessed_3d` | 193.2 -> 212.1 | 16.78 -> 13.15 | 572.1 |
| `preprocessed_3d_radial` | 521.3 -> 152.6 | 30.97 -> 12.17 | 738.7 |
| `preprocessed_3d_radial_ntss1d` | 781.8 -> 152.6 | 39.59 -> 16.60 | 592.2 |
| `preprocessed_3d_ntss1d_fixed` | 521.3 -> 152.6 | 35.51 -> 13.28 | 1256.3 |
| `preprocessed_ntss` | 3830.9* -> 156.1 | not executed* -> 17.14 | 1613.2 |

*The uncontracted full NTSS Jacobian exceeded the 2 GiB temporary limit. Its
compiled workspace was measured, but it was not executed; three original
single-direction JVPs instead agreed with contractions of the optimized matrix.
The other seven full Jacobian matrices were compared directly. Query values,
query JVPs/VJPs, edge cases and coefficient-direction derivatives also agreed
before/after. The largest scaled array difference across these comparisons was
2.38e-12 (scale `max(1, abs(reference))`).

Temporary workspace is not total GPU memory. The recorded allocation peaks
include preparation, query checks and compilation; NTSS compilation still has
a larger peak. Allocator reservation is recorded separately in the
[raw CSV measurements](monoenergetic_interpolation_ad_2026-11-10.csv).
The trilinear method uses 10% more temporary workspace, but its allocation peak
drops from 1508.7 to 572.1 MiB and warm execution improves by 22%. The cubic
3D temporary increase is 3%, with a 25% lower allocation peak. Those tradeoffs
were retained. An earlier generic rule that evaluated scalar partials separately
was discarded after doubling warm execution time; the joint rule above is the
selected implementation.

The final generic initial-root coordinate pullback retained a 288.3 MiB
allocation peak, 6.83 s cold execution and 17.77 ms warm execution. Its saved
arrays match the pre-task native reference at `rtol=atol=3e-10`.

Validation also passed 38 focused small-array tests: 35 interpolation, boundary,
helper and root-coordinate checks, plus three database safety checks. The added
joint-rule mixed-Hessian assertion was rerun separately and passed. These are
bounded component checks, with no local transport simulation.

## Full transport AD benchmark boundary

The existing live `ntx_scan_runtime` reverse benchmark constructs a generic
`Monoenergetic` database directly. Its recorded-scan transpose, initial-root
and bootstrap support adapters are not changed into eight interchangeable
database formats by these kernel changes. Editing the TOML interpolation
name does not switch that live-scan benchmark.

`preprocessed_ntss` also has geometry-dependent preprocessing/fit metadata;
its full realtime-geometry reconstruction is explicitly unsupported by
`database_with_geometry_scale`. Passing a kernel derivative check does not
validate differentiating that NumPy preprocessing. Validate a separate
format-to-scan adapter before a full realtime geometry AD comparison.

The full generic benchmark remains the end-to-end validation case. Component
speedups and reduced workspace are not a claim that its complete remote
reverse run has passed.
