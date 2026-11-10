# Native monoenergetic interpolation

Implemented in `en/v0_main` on 10 November 2026. The generic database query
uses a native JAX implementation; no XLA guard-padding setting is required.
The old `en/v0_new` and cleanup checkouts were not edited.

## Implementation

`NEOPAX/_native_interpolation.py` implements the same local C1 cubic rule as
Interpax's default `cubic` interpolation. End slopes equal the adjacent
secant; interior slopes equal the mean of the two neighboring secants.
Polynomial extrapolation uses the boundary interval and its existing slopes.

Each scalar query uses a local 4x4 coefficient stencil. One set of interval
calculations is shared by D11-log, D13 and D33.
Tensor-product Hermite interpolation is evaluated in increment/Horner form,
forming differences before multiplication to limit cancellation in nearly
constant tables. It constructs neither full-table slope arrays nor the
generic 16x16 bicubic coefficient transformation per query. Native JAX AD
retains derivatives with respect to queries, table entries and knot positions,
including higher derivatives. No stopped-gradient cache or new custom AD rule
was introduced.

Cell lookup uses an arithmetic estimate plus an exact correction for strictly
increasing, sufficiently uniform grids. Interpolation weights still use the
actual knots. Other grids use a comparison-based search for small arrays or
an unrolled binary search for larger arrays. This avoids repeated looping
searches on the GPU without approximating the interpolation rule.

`get_Dij` also uses a scalar branch to evaluate its selected small-radius,
interior or edge closure. Previously all three closures ran and their outputs
were masked. Under a `vmap` whose branch predicate is batched, JAX may still
evaluate multiple branches; no universal branch-execution saving is claimed.
The inactive division at the magnetic axis is made finite before applying
the unchanged electric-field floor.

Within each radial closure, its three or four surface queries are evaluated
together, with the coefficient channels on one trailing axis. The existing
radial interpolation formulas are unchanged. This reduces repeated pullback
intermediates; it does not split the state queries into smaller batches.

The generic database surface queries and their 1D helper queries use these
native routines. The optional direct 3D and geometry interpolation paths
still use Interpax, so it remains a package dependency. Preprocessed database
formats retain their own interpolation formulas.

The initial-Er coordinate pullback uses three reusable compiled stages:

1. Prepare the physical queries and unit particle-flux coefficient
   derivatives, retaining sequential radial evaluation. The derivative uses
   Gamma's first Lij row directly, including the log-D11 and parallel-force
   terms, instead of building a full matrix VJP.
2. Transpose the recorded interpolation stencils to their local electric-field
   knots and radial scale. A four-knot native forward Jacobian replaces the
   previous generic bicubic coefficient transformation. Only the accumulated
   per-species/radius coordinate basis is dense.
3. Contract that basis with the objective cotangents and construct the
   established database-bar payload.

The state, geometry, species, energy-grid, database and boundary-value arrays
are dynamic inputs. Boundary type strings are static metadata. The stages
avoid a new state-capturing compilation at every call without stopping
derivatives or changing the interpolation rule. BoundaryConditionModel is
registered as a JAX dataclass so its numerical fields remain differentiable
inputs rather than cache-key identities.
Model subclasses and custom boundary objects retain the existing path,
preserving their override and boundary protocols.

## GPU comparison

Both measurements used JAX/jaxlib 0.10.0, Interpax 0.3.16, x64, the same RTX
5060 Laptop GPU (8 GiB, driver 592.25, WSL), and fresh sequential processes.
`XLA_FLAGS` was empty in both. Persistent compilation cache and JAX
preallocation were disabled in both. Neither XLA correctness guards nor
autotuning were disabled. Both final workers use the same scalar radial
selection and grouped radial-surface evaluation. The Interpax control uses
the independently differentiated generic VJP with reusable compilation;
the native worker uses the three sparse stages. Coordinate timings therefore
compare the complete pullback implementations, while warm query timings
compare the interpolation kernels themselves.

The physical coefficient file was `examples/inputs/Dij_NTX_QI_Er_Optimized.h5`,
with shape **7 radii x 16 collisionalities x 12 electric fields**. The state
had 51 radial cells, four species, four energy nodes and 13 cotangent rows.
The state is synthetic; the coefficient data are saved QI data. No
equilibrium, NTX scan, root search or time-dependent transport solve ran.

| Measurement | Interpax | Native | Change |
| --- | ---: | ---: | ---: |
| Initial-Er coordinate VJP, first call including compilation | 10.534 s | 6.692 s | 36% lower time |
| Coordinate VJP, median of the next two calls | 95.097 ms | 18.425 ms | 81% lower |
| Peak JAX allocation through that operation | 400.1 MiB | 288.3 MiB | 28% lower |
| Allocator pool through that operation | 514 MiB | 514 MiB | Unchanged |
| Warm forward query, 51 queries | 0.642 ms | 0.461 ms | 28% lower |
| Warm query Jacobian, 51 queries | 0.964 ms | 0.539 ms | 44% lower |

Warm timings are medians of 500 synchronized calls after five warmups.
Timings vary on this laptop: the forward interquartile ranges were
0.495-1.067 ms (Interpax) and 0.407-0.570 ms (native); Jacobian ranges were
0.626-1.676 ms and 0.462-0.727 ms respectively. These component observations
are not a full solver speedup measurement.
First-call timing includes compilation; warmed query timing measures a
different, smaller operation. Peak memory comes from JAX's allocator counters,
not total driver/compiled-code memory. It is not the whole simulation's
memory use. Raw measurements are in the adjacent CSV.

The value and derivative comparisons passed `rtol=atol=3e-10`:

| Result | Maximum `abs(new-old)/max(1,abs(old))` | Relative L2 difference |
| --- | ---: | ---: |
| Geometry-scale coordinate bar | 1.46e-12 | 8.25e-14 |
| Electric-field-grid coordinate bars | 1.85e-11 | 1.19e-15 |
| Interpolated coefficients | 8.80e-16 | 2.83e-16 |
| Query Jacobian | 1.82e-14 | 7.09e-15 |

In the earlier nearly-flat synthetic coefficient stress fixture, the native
implementation before the final radial and lookup changes differed from
Interpax by up to 8.3e-8
relatively in individual bars and did not pass the original 3e-10 test.
Its strict comparison was not relaxed. The independent saved-QI comparison
above and small-array derivative tests establish the agreement claimed here;
bitwise identity for every dataset is not claimed.

Before grouping radial surfaces, the original Interpax measurement peaked
at 1,552.1 MiB and the initial native implementation at 624.2 MiB. The final
native implementation reaches 288.3 MiB, about 81% below that original
Interpax measurement. The CSV distinguishes these earlier measurements
from the final comparison above. The final snapshots also pass the same
3e-10 comparison against the original, ungrouped Interpax snapshots.

The focused regression run passed **32 checks**, including native
interpolation boundaries/extrapolation, VJPs for all inputs, JVPs, Hessians,
finite differences, dense/sparse database transposes and flux-transpose
consistency, boundary conditions and coordinate-pullback compilation reuse.
The integrated stages also match the original VJP for all five collisionality
models and for a mixed derivative with respect to profile pressure.
The cell-lookup check includes the strict spacing safeguard and compares
against NumPy's right-sided search at knots,
adjacent representable values, repeated knots, extreme extrapolations and
large-offset grids whose coordinates round to duplicates.

## Reproduce

The Python benchmark launches the two fresh worker processes itself:

```bash
cd "$HOME/NEOPAX" || exit 1
env -u XLA_FLAGS -u JAX_COMPILATION_CACHE_DIR \
  JAX_PLATFORMS=cuda,cpu \
  JAX_ENABLE_COMPILATION_CACHE=0 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$HOME/jax_env/bin/python" \
  examples/benchmarks/benchmark_native_database_interpolation.py \
  --out-dir outputs/native_database_interpolation \
  --repeats 500 \
  --coordinate-repeats 3
```

It writes CSV measurements and NPZ value/derivative snapshots, then checks
agreement. `--database` selects another saved NTX HDF5 file. Run the original
full reverse-AD benchmark separately to assess complete solver memory and
gradients on the target vGPU; it has not been run locally.

## Additional memory reduction

Grouping the radial-surface queries reduced the native coordinate-pullback
peak from **624.2 MiB to 400.1 MiB**, another **36%** reduction. The observed
allocator pool fell from **1,026 MiB to 514 MiB** with preallocation disabled.
These are component measurements, not whole-process or full-simulation
memory requirements. The grouped Interpax control also reaches about
400 MiB: this additional memory saving comes from the radial evaluation
structure, rather than the native cubic formula alone.

Staging barriers between interpolation passes and contracting a smaller
species Jacobian before objective rows did not reduce the peak. Packed
geometry temporaries increased it to about 464 MiB. Precomputed slopes or
cubic coefficients offered no further memory saving and increased setup
cost. Those variants were rejected. The shared radial evaluation, faster
exact cell lookup and compact coordinate stages are enabled with the native stencil.
The adjacent memory-candidate CSV preserves these component measurements.

### What remains in the 400 MiB peak

A probe of the grouped implementation before the reusable wrapper intercepted
the `jit_scan` compilation, before its executable ran. The JAX allocation
peak had already reached 419,564,288
bytes (400.1 MiB). The compiled executable reported 128,976 bytes (about
126 KiB) of runtime temporary workspace, 68,292 bytes of arguments and
8,856 bytes of outputs. After the coordinate pullback, live JAX arrays
occupied 502,272 bytes, while the allocator pool remained 514 MiB.

Thus that measured 400 MiB is a compilation-time peak, not 400 MiB of
interpolation arrays retained at runtime. Further reductions should target
the compiler's temporary-buffer pressure in the local flux/coordinate
pullback. Removing a few stored interpolation arrays alone is unlikely to
produce another large reduction. The coordinate-stencil factorization below
targets this compilation pressure; these figures do not describe full
transport memory.

### Earlier coordinate-pullback graph experiments

All five candidates used fresh sequential GPU processes, the same saved QI
table/state/cotangents, and default XLA settings. Every candidate passed the
saved value and derivative comparison at `rtol=atol=3e-10`.

| Candidate | First coordinate pullback | Median of next two calls | Peak JAX allocation |
| --- | ---: | ---: | ---: |
| Existing grouped implementation | 8.573 s | 3.639 s | 400.1 MiB |
| Profile work moved outside local VJP | 12.264 s | 3.527 s | 400.1 MiB |
| Local species Jacobians, then objective contraction | 8.999 s | 3.477 s | 400.1 MiB |
| Both of the preceding changes | 12.203 s | 3.343 s | 400.1 MiB |
| Reusable compilation with dynamic numerical inputs | 6.367 s | 0.024 s | 400.1 MiB |

None reduced the memory peak meaningfully. The reusable-function candidate
was retained for its timing improvement with memory essentially unchanged;
the other three graph changes were rejected. The approximately 150-fold
repeat-call improvement applies to this isolated coordinate pullback, not
the entire reverse-AD calculation. These repeated calls used identical
inputs; a separate small-array regression also checks changed numerical
inputs against the original implementation and verifies no new JIT cache
entry when their shapes, dtypes and static metadata stay the same.
The final production wrapper independently measured 6.464 s for its first
pullback and 23.319 ms for repeated calls, with all saved coefficient and
derivative comparisons passing again. The custom-boundary and subclass
paths also pass the original-body comparison.

The adjacent coordinate-graph CSV contains these first/repeated-call times
and allocator measurements. The benchmark's `--coordinate-repeats` option
measures repeated pullbacks separately from its warmed interpolation-query
timings.

### Sparse coordinate-stage experiments

These probes change the derivative factorization while preserving the same
particle-flux, boundary, radial interpolation and coordinate contracts.
All passed comparison with the saved generic native VJP at `rtol=atol=3e-10`.

| Prototype | Peak JAX allocation | First call | Repeated call median |
| --- | ---: | ---: | ---: |
| Split stages, all radii vectorized in preparation | 832.2 MiB | 8.508 s | 4.114 ms |
| Split stages, sequential radial preparation | 480.1 MiB | 7.266 s | 17.550 ms |
| Sequential preparation with compact particle contraction | 400.2 MiB | 7.152 s | 17.823 ms |
| Compact particle contraction and native local forward Jacobian | 288.3 MiB | 6.635 s | 16.292 ms |
| Same, with a scalar local reverse derivative | 288.2 MiB | 7.079 s | 20.978 ms |

The native local forward Jacobian was selected: it reduces the previous
400 MiB compilation peak by about 28%, with comparable first-call time
and faster repeated execution. The scalar reverse alternative offered no
meaningful extra memory saving. The allocator pool still reached 514 MiB
in these probes; the lower live-allocation peak does not imply an equal
reduction in memory reserved from the driver.

The **224 MiB preparation measurement was one stage only**. With the older
sparse coordinate transpose, that stage was followed by a 400 MiB peak,
so the complete operation still peaked at 400 MiB. Replacing the coordinate
transpose reduced the complete peak to 288 MiB; the full operation has not
been measured at 224 MiB.

The production integration independently measured a 288.3 MiB peak,
6.692 s first call and 18.425 ms repeated calls. It passed both the fresh
generic Interpax VJP comparison and comparison with the original ungrouped
Interpax snapshots. Against the preceding native reusable VJP, its first
call is about 4% slower and its repeated calls about 21% faster, with 28%
less peak allocation.
