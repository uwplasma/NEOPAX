# Trace the first differing normalization input

The repeated generic baseline builds used identical configurations but
occasionally produced different derived normalization inputs. Saved native
VMEC surfaces agree while Boozer-derived transport R0, a_b and B00 differ
at the last-bit level. This identifies an upstream boundary, not the exact
operation responsible. Sensitivity of D13 to Er alone does not identify a
solver bug.

`check_generic_ntx_database_baselines.py --trace-inputs` runs three independent
processes named `baseline_1`, `baseline_2` and `baseline_3`. All use the same
unperturbed FD-baseline configuration. No gradient or FD perturbation is
computed. The ordinary database-comparison mode is retained.

## Captured boundaries

The trace records, in actual call order:

1. The equilibrium state supplied to geometry construction.
2. The actual VMEX volume/vp values used to calculate the normalization radius.
3. Inputs, constants, grids and outputs of each production Boozer transform.
4. The resulting NEOPAX transport geometry.
5. The native scan surfaces and normalization channels.
6. The Er and Es arrays, and the Er-to-Er_tilde conversion.

Instrumentation is confined to this benchmark, with temporary wrappers that
restore the original functions on exit. Arguments and numerical operations
are forwarded unchanged. Wrappers retain references to immutable device
arrays and copy them after reaching the boundary; they add no device
reductions or callbacks to the computational graph. The volume observer is
active only during geometry/input construction, outside the equilibrium solve.
Execution stops before NTX coefficient or database construction, root
initialization and transport.

Each worker saves `normalization_inputs_*.npz`, `state_reference.npz`,
`metadata.json` and `run.log`. Metadata records input/config/state hashes,
package versions and source hashes, including loaded VMEX modules and the
Boozer JAX API. The parent refuses comparisons across changed inputs or
sources. `comparison.csv` records matching arrays, differences and exact
hexadecimal floating-point values. The console reports `first_changed_stage`
for each comparison with baseline 1.

## First run: reuse the saved implicit equilibrium

This control holds every equilibrium array fixed across processes and
performs **no equilibrium solve**. It tests reproducibility of the downstream
geometry construction; it does not certify reproducibility of a fresh VMEX
solve. The saved `implicit` fields match the benchmark's configured `ad`
primal lane. This is not a comparison between VMEX forward and AD lanes.

Run on the large GPU machine after pushing/pulling the diagnostic changes:

```bash
(
  cd "$HOME/NEOPAX" || exit 1
  git switch en/v0_main || exit 1
  git pull --ff-only origin en/v0_main || exit 1
  mkdir -p outputs
  set -o pipefail

  env JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=true \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "$HOME/jax_env/bin/python" \
    examples/benchmarks/check_generic_ntx_database_baselines.py \
    --trace-inputs \
    --reuse-states outputs/vmex_primal_consistency/states.npz \
    --state-prefix implicit \
    --out-dir outputs/normalization_trace_fixed_state_20261116 \
    2>&1 | tee outputs/normalization_trace_fixed_state_20261116.log
) || printf '\nNormalization trace stopped; terminal kept open.\n'
```

Use a new empty output directory. The parent starts GPU workers sequentially,
with compilation caches disabled as in the original repeat experiment. It
uses no additional XLA flags. GPU absence is an error; there is no CPU physics
fallback. The fixed state must have compatible dimensions and represent the
same input deck; the older states file has no provenance hash, so the trace
records the file and current input hashes without claiming historical identity.

An existing plain `state_reference.npz` from this diagnostic is also accepted
by `--reuse-states`.

## Interpret the first differing stage

- Different volume values from identical states locate variation in the
  volume/geometry preparation, ahead of Boozer normalization.
- Matching state/volume but differing Boozer inputs locate it in VMEX-to-Boozer
  preparation, rather than the transform itself.
- Identical Boozer inputs, constants and grids but differing outputs isolate
  variation to the transform execution. They do not alone identify a specific
  compiler, contraction or backend algorithm.
- Matching Boozer outputs but differing geometry/channels narrow the issue to
  normalization or interpolation of those geometry values.
- All three repeats matching means the issue was not reproduced under this
  frozen-state control. It does not rule out intermittent variation. A later
  run without `--reuse-states` can trace fresh equilibria; that mode performs
  one equilibrium solve per process and should only be used if needed.

No production solver, interpolation, physics or AD rule is changed. Eight
NumPy/mock tests check observation order, unchanged arguments/array references,
early termination, wrapper restoration, saved-state validation, exact-bit
reporting and the three-process driver. No real JAX or physical solves run
in these local tests. GPU results are pending.
