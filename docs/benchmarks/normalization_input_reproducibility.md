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

## Trace the actual ZBS FD setup

The earlier saved-input replay bypassed runtime construction. The original
input trace repeated `build_runtime_context(config)` three times; it did not
exercise the parameter-specific frozen-linearized geometry FD builder.
Matching either check therefore does not rule out a difference in that builder.

The saved-array readout from `outputs/ntss1d_ad_fd16_WFTo4W` confirms that
the ZBS baseline difference is already present upstream of transport:

| Saved field | Maximum absolute AD/FD difference |
| --- | ---: |
| geometry a_b | `2.220446049250313e-16` |
| geometry R0 | `1.7763568394002505e-15` |
| geometry B0 | `3.552713678800501e-15` |
| database D11_log | `1.4103830707767884e-06` |
| database D13 | `6.40592250133909e-03` |
| database D33 | `4.3873442773900706e-04` |
| packed initial state | `6.953623297517808e-03` |

All printed finite masks match. D13's largest difference is at `(6, 0, 7)`,
one of the previously isolated sensitive NTX points. This readout locates
an upstream geometry difference; it does not identify the operation that
introduced it or prove that a_b alone accounts for the table difference.

The FD script selects a geometry context from `--parameter`, prepares a VMEC
solve and its tangent, then builds its zero-offset runtime from
`state_star + 0 * state_tangent`. The normal AD baseline calls
`build_runtime_context(config)`. Both VMEX primal entry points ultimately use
the same `_callback_solve` in the inspected VMEX source. Different public
function names alone do not establish different equilibrium algorithms.
The outer AD script reuses its known initial Er profile during preliminary
setup. This is not the initial state of the optimization API evaluation:
the production evaluator builds profiles and selects a fresh root again.
Confusing these stages led to the failed cached-root patch described below.
Neither root setup can explain an already-different coefficient table.

`--trace-inputs --fd-geometry-parameter ZBS:1:0` now compares one normal AD
setup with two independent calls to the **actual** geometry FD helpers.
It accepts either generic or radial NTSS1D interpolation and preserves the
configured choice. The new comparison runs the original VMEC tangent setup
but stops before NTX coefficient solves, root initialization and transport.
It changes no production solver, interpolation, baseline or replay arithmetic.

Trace stage names are aligned at the common geometry-equilibrium boundary;
the scalar solver's extra return event is omitted in this comparison only.
The trace also captures the actual VMEC parameters/configuration immediately
before each primal call, and its returned state before the FD tangent setup.
It can therefore distinguish different solve inputs, different returned
equilibria, and a difference introduced by zero-offset state assembly. The
observers forward the original objects to the original VMEC entry points.
Both workers retain state, geometry, channel and native surface arrays, plus
input fields and source/version hashes. This can locate the first changed
boundary without assuming the support aggregate identifies its source.
Four NumPy/mock regressions verify selection of the original FD helpers,
alignment of the common boundaries, unmodified VMEC calls and the AD/FD/FD
worker sequence.
All 13 tests in the diagnostic's NumPy-only test module pass.

### GPU result: exact paths agree in this run

The remote run at commit `a8313d8`, saved in
`outputs/ntss1d_zbs_input_paths_k4RToS`, completed all three independent workers:

| Comparison | Differing numerical leaves | First differing bit stage |
| --- | ---: | --- |
| AD vs original ZBS FD setup 1 | 0 | none |
| AD vs original ZBS FD setup 2 | 0 | none |
| Original ZBS FD setup 1 vs 2 | 0 | none |

This is a non-reproduction, not a fix of the earlier failing baseline.
It shows that the parameter-specific setup paths can agree exactly; their
structural differences alone do not establish a consistently different ZBS
geometry calculation. It does not certify the subsequent NTX solves, roots
or transport, which were intentionally skipped.

The follow-up NumPy comparison reports identical TOML settings and all 25
fresh traced geometry fields matching the saved AD baseline numerically.
Against the earlier ZBS FD baseline, 18 of those fields differ. R0 differs
by one ULP (`0x1.82d1eabcb3d91p+3` versus `0x1.82d1eabcb3d90p+3`), and a_b
by one ULP in the opposite direction (`0x1.2d85eb2e9829ap+0` versus
`0x1.2d85eb2e9829bp+0`). This establishes that the fresh matching setups
reproduce the earlier AD geometry, rather than a third geometry. That
readout checks numerical equality; it does not separately compare signed
zeros or NaN payloads with the old snapshots.

The trace preserves the arithmetic but retains intermediate device-array
references until capture. This changes buffer lifetimes compared with the
original benchmark. An observer effect or execution-context-dependent
roundoff remains unproven. No numerical solver change is justified by this
matching result.

### Capture the original FD script without intermediate observers

`benchmark_transport_realtime_geometry_forward_fd.py --geometry-input-only-dir`
runs its unchanged baseline setup calls and installs only a stop at the
completed scan-input boundary. No VMEC, volume, field, Boozer or Fourier
operation is wrapped, and no intermediate arrays are retained during
geometry construction. The normal transport configuration is preserved.
After geometry and scan inputs are complete, the stop records their existing
values and the supplied equilibrium, then returns before NTX coefficient
construction, initialization roots, transport or FD endpoints.

`--geometry-input-reference` compares these four common boundaries against
the already-saved detailed trace. Stage numbers are normalized for comparison;
array values and byte comparisons are unchanged. Three additional NumPy
regressions verify the unobserved geometry builder, restoration/early stop,
unmodified arguments, stage-number alignment and the original FD setup
branch with and without capture. All 16 tests in the module pass.

The original-script GPU control subsequently completed in
`outputs/zbs_original_inputs_0TjYww`, producing
`normalization_inputs_bffsdunw.npz`. Compared with the saved matching trace,
the equilibrium supplied to geometry construction, completed geometry,
scan channels and native scan surfaces all match numerically and bitwise:
`differing leaves=0`, `first_changed_stage=none`,
`first_changed_bit_stage=none`. The pasted excerpt does not include Git HEAD.

This reproduces the saved AD geometry without the detailed trace's
intermediate observers. It does not support identifying those observers
as the cause of the earlier differing ZBS evaluation. The cause remains
unidentified, and no physics, interpolation or time-solver change follows
from this result. The next distinct control, if needed, is the existing
original-script `--baseline-only-dir`: complete NTX construction, root
initialization and the 16-step baseline, then compare saved geometry,
tables and initial state with the archived AD. It needs no new code and
does not compute derivatives or FD endpoints. Repeating the already-matching
pre-NTX captures is not an additional localization step.

The complete original-script baseline then completed in
`outputs/zbs_complete_baseline_UWRwqd`: 16 accepted steps, 23 attempts,
final time `4.1195452419569767e-04`, with Newton setup recorded. The
NumPy-only comparison against the archived AD run reports:

| Captured group | Fields compared | Differing fields |
| --- | ---: | ---: |
| Geometry | 25 | 0 |
| Interpolation database | 19 | 0 |
| Packed initial state, final state and final time | 3 | 0 |

This comparator checks array bytes as well as shapes and dtypes. The
complete baseline agrees, including the NTX table and root-initialized
state, without the pre-NTX stop or intermediate observers. No AD or FD
endpoint was computed in this control. The earlier mismatching evaluation
remains unexplained; this result does not establish a backend fix or
validate a derivative. No further repetition of the same baseline controls
is needed for this matching case. A single ZBS FD16 column can now be
checked against the existing AD report, verifying its own fresh baseline
before interpreting the gradient differences.

### FD16 recheck: matching tables, remaining initialization drift

The next ZBS FD16 run, `outputs/ntss1d_zbs_fd16_recheck_Rhdsg8`, reports
zero geometry/database support difference, but an initial-state difference
of `7.105427357601002e-15`. Its final state differs by
`2.6708448785939254e-09` and final time by `5.796795335410998e-13`.
Both perturbed endpoints converge for all 16 fixed accepted steps with no
Newton retries. The baseline comparison correctly remains unvalidated.
The [saved gradient table](ntss1d_zbs_fd16_recheck_2026-11-19.md) records
the numerical differences with this qualification.

The follow-up NumPy-only readout confirms identical packed density/pressure
blocks and exactly three differing Er entries at radial indices 45, 46
and 49, differing by one, two and one ULP respectively. This localizes the
remaining initial-state mismatch to Er. It does not compare the FD runtime's
root before the repeated solve with its root afterward.

### Cached-root patch withdrawn: the outer setup was the wrong reference

Commit `e0a90cc` changed FD to reuse its runtime's preliminary root, based
on `_run_realtime_geometry_reverse_mode` doing so in the outer AD script.
The GPU control in `outputs/zbs_root_reuse_baseline_HXmxwc` disproves the
claimed baseline alignment. Geometry (25 fields) and tables (19 fields)
match, but 14 initial Er entries differ by up to
`1.2069774596730198e-03`. Final state differs by
`3.0840242807084195e-04` and final time by `6.513128733174819e-10`.
This is a failed patch, not a verified fix of the previous ULP drift.

The actual AD path is:

1. The outer runtime uses the TOML root setup, including scan mode `vmap`.
2. `prepare_reverse_static_setup` in `_reverse_ad_transport.py` invokes the
   production `initial_state_for_parameter_vector` constructor to record
   the adaptive baseline. This rebuilds profiles and selects a fresh root.
3. `realtime_geometry_reverse_all_objectives_support_payload_bar_for_parameter_vector`
   also builds a pre-root profile state and evaluates the selected-root
   primal separately when preparing the pullback. The shared
   `initial_er_root_setup` overrides the root scan mode to `hybrid`.

The AD terminal snapshot records the prepared static initial carry, not
the outer script's preliminary root. The forward states and their root
initialization must not be inferred from the outer setup call alone.
The differing batch choices use the same intended root equation; their
numerical effect on these particular saved states still needs a direct
comparison, rather than an assumption based on the source.

The corrective FD change removes cached-root reuse and invokes the **same
production constructor as AD's adaptive schedule setup** for the enabled
runtime-scan baseline. The benchmark's duplicate constructor remains on
the existing endpoint, root-only and other-model paths. No root or time
solver algorithm is modified. The earlier large geometry variation and
the remaining derivative differences are not identified by this correction.

The previous regressions checked the cache-argument plumbing against an
incorrect expected contract. They have been replaced with tests that
execute the actual AD schedule-constructor call and compare it with FD's
constructor selection, using different preliminary and selected roots so
wrong cache reuse is detected. Additional checks cover the shared hybrid
setup and unchanged legacy/endpoint routes. All 25 NumPy/mock tests in the
two relevant diagnostic modules pass. No local JAX or physics solve ran.
Numerical verification of the corrective patch is pending. A focused GPU
comparison can restore the existing saved geometry/tables and compare the
runtime and evaluator root initializers without VMEC, an NTX scan or transport.

### Saved-root replay restored the wrong interpolation class

The subsequent root-only probe reported a maximum difference of
`7.163548879044086e-01` for both recomputed initializers. That probe used
`restore_runtime`, whose original implementation always instantiated
`Monoenergetic`. This is correct for generic tables but wrong for this
NTSS1D case: production builds `MonoenergeticRuntimeNTSS1D`. The query
dispatcher selects the interpolation kernel from the database class,
not the runtime wrapper's `interpolation_mode` string. Identical saved
arrays therefore did not imply the same root equation in that probe.

The diagnostic now restores the configured class, checks its dispatch kind
and prints both. A NumPy/mock regression executes the actual dispatcher,
checks all saved fields and preserves the generic restoration path.
Unsupported modes fail before model construction. This changes diagnostic
restoration only. The reported `0.716` root shift cannot establish a
production initializer or batching defect. The earlier cached-root failure
in the complete original FD entry point remains valid and separate.

After publishing the change, run the original FD entry point on the GPU:

```bash
(
  cd "$HOME/NEOPAX" || exit 1
  git pull --ff-only origin en/v0_main || exit 1
  neopax_out=$(mktemp -d outputs/zbs_original_inputs_XXXXXX) || exit 1
  neopax_ref=(outputs/ntss1d_zbs_input_paths_k4RToS/ad/normalization_inputs_*.npz)
  [ "${#neopax_ref[@]}" -eq 1 ] && [ -f "${neopax_ref[0]}" ] || exit 1
  set -o pipefail
  env -u PYTHONPATH -u LD_LIBRARY_PATH -u JAX_PLATFORM_NAME \
    -u JAX_COMPILATION_CACHE_DIR -u XLA_FLAGS -u XLA_PYTHON_CLIENT_ALLOCATOR \
    -u NEOPAX_GRADIENT_AUDIT_DIR \
    JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=true \
    JAX_ENABLE_COMPILATION_CACHE=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    NTX_DISABLE_COMPILATION_CACHE=1 VMEX_COMPILATION_CACHE=disabled \
    NEOPAX_REVERSE_STEP_DIAGNOSTICS=1 NEOPAX_TEST_RUNTIME_NTSS1D=1 \
    "$HOME/jax_env/bin/python" \
    examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py \
    --config examples/benchmarks/Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_ntss1d.toml \
    --parameter ZBS:1:0 --vmec-response-mode raw_block \
    --accepted-step-limit 16 --radau-jacobian-reuse-mode legacy \
    --geometry-fd-lane frozen_linearized --replay-mode accepted \
    --fd-accepted-replay-contract reverse_frozen_metadata \
    --initial-Er-root-ad jax_selected_root \
    --initial-Er-root-fd-root-lane frozen_linearized --diagnose-fd-endpoint \
    --geometry-input-only-dir "$neopax_out" \
    --geometry-input-reference "${neopax_ref[0]}" \
    2>&1 | tee "$neopax_out.log"
) || printf '\nOriginal-script input capture stopped; terminal kept open.\n'
```

This runs no accepted steps despite retaining the original benchmark options.
It tests whether the original script without intermediate observers reproduces
the saved AD geometry or the differing ZBS geometry. A matching result would
again be a non-reproduction; a mismatch would be localized to the first
reported common boundary. It does not establish a library/backend cause.

After publishing the diagnostic, run on the large machine:

```bash
(
  cd "$HOME/NEOPAX" || exit 1
  git pull --ff-only origin en/v0_main || exit 1
  neopax_out=$(mktemp -d outputs/ntss1d_zbs_input_paths_XXXXXX) || exit 1
  set -o pipefail
  "$HOME/jax_env/bin/python" \
    examples/benchmarks/check_generic_ntx_database_baselines.py \
    --config examples/benchmarks/Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_ntss1d.toml \
    --trace-inputs --trace-projections \
    --fd-geometry-parameter ZBS:1:0 \
    --out-dir "$neopax_out" \
    2>&1 | tee "$neopax_out.log"
) || printf '\nZBS input-path trace stopped; terminal kept open.\n'
```

Read `first_changed_stage`, `first_changed_bit_stage` and `comparison.csv`.
The script name retains "generic" for the existing database comparison;
this input-only trace preserves the explicitly selected NTSS1D configuration.

Each worker saves `normalization_inputs_*.npz`, `state_reference.npz`,
`metadata.json` and `run.log`. Metadata records input/config/state hashes,
package versions and source hashes, including loaded VMEX modules and the
Boozer JAX API. The parent refuses comparisons across changed inputs or
sources. `comparison.csv` records matching arrays, differences and exact
hexadecimal floating-point values. The console reports `first_changed_stage`
for each comparison with baseline 1. Numerical equality permits matching
NaNs and treats signed zeros as equal; the enhanced comparator also records
`bits_match` and `first_changed_bit_stage` separately.

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

The original observer changes no production numerical operations. Nine
NumPy/mock tests check observation order, unchanged arguments/array references,
early termination, wrapper restoration, saved-state validation, exact-bit
reporting and the three-process driver. No real JAX or physical solves run
in these local tests. The fixed-state GPU run at `821c5b8` has completed:
all three pairwise comparisons report zero differing captured leaves.
See the [recorded fixed-state result](normalization_trace_fixed_state_2026-11-16.md).

## Next run: include fresh equilibrium construction

The fixed-state trace did not reproduce the variation. The same diagnostic
can now include the missing equilibrium-construction stage by omitting
`--reuse-states`. This performs **three fresh equilibrium solves**, one per
independent process, while retaining the same configuration, wrappers and
pre-NTX stopping boundary. No NTX coefficient, root, transport or derivative
solve runs. The mode is already present in `821c5b8`; no new executable code
needs to be pushed for this check.

```bash
(
  cd "$HOME/NEOPAX" || exit 1
  mkdir -p outputs
  set -o pipefail
  env JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=true \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "$HOME/jax_env/bin/python" \
    examples/benchmarks/check_generic_ntx_database_baselines.py \
    --trace-inputs \
    --out-dir outputs/normalization_trace_fresh_state_20261116 \
    2>&1 | tee outputs/normalization_trace_fresh_state_20261116.log
) || printf '\nFresh-equilibrium trace stopped; terminal kept open.\n'
```

This is a reproducibility check of the existing configured primal equilibrium
lane. It does not switch VMEX lanes or differentiate a new solve. The first
differing captured boundary is evidence of where variation first appears in
that run, not proof of a particular compiler or physical-model defect.

## Fresh run reproduced the variation; axis correction

The [fresh-state run](normalization_trace_fresh_state_2026-11-16.md) reports
no numerical differences in returned states or volume inputs. Baseline 3
first differs in the Boozer input preparation, with 63 differing captured
leaves. Thus that experiment localizes the first numerical difference ahead
of the Boozer transform. It does not yet identify the primitive producing
the smaller last-bit differences.

Inspection found a separate, concrete amplification bug in the batched
NEOPAX adapter: odd-m axis coefficients were divided by `1e-30`. Projected
zeros contain roundoff, so this can produce enormous R/Z coefficients in
the first half-mesh row. The adapter now uses an axis denominator of one,
matching `vmex.core.boozer_tables.boozer_input_tables`. The ordinary nonaxis
mesh formula and even-m interpolation are retained. Three NumPy regressions
exercise the actual production interpolation kernel without importing JAX.
The detailed GPU trace at `f45cdec` completed: all three fresh runs match in
every captured numerical value and byte pattern. See the
[post-fix result](normalization_trace_axis_fix_2026-11-16.md). This supports
reproducibility in that experiment, not a claim that every intermittent
input difference or downstream AD/FD discrepancy is fixed.

`--trace-projections` adds observation of the real-space R/Z/lambda fields,
Jacobian, metrics and magnetic fields consumed by the adapter, plus the exact
operands and results of its `rtz,tzm->rm` Fourier contractions. These are
references to existing arrays, with no additional field calculation or
projection. Snapshots are larger. The observer does not change contraction
order or select a different backend algorithm.

After pushing the focused adapter correction and observer update, rerun the
fresh-state trace with this extra flag. This retains the original fresh-solve
call history and performs three equilibrium solves, but no NTX coefficient,
root, transport or derivative solve:

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
    --trace-projections \
    --out-dir outputs/normalization_trace_projection_axis_fix_20261116 \
    2>&1 | tee outputs/normalization_trace_projection_axis_fix_20261116.log
) || printf '\nDetailed input trace stopped; terminal kept open.\n'
```

Matching projection operands with differing results would isolate variation
in the contraction execution. Earlier differences in real-space preparation
would point upstream of that contraction. Both are distinct from the known
axis amplification, and neither should be inferred before inspecting output.

## Next: generated tables without observation wrappers

The fixed inputs must now be checked in the ordinary full database build.
The existing mode below uses one AD-baseline configuration and two identical
FD-baseline configurations. These labels describe setup/scan-recording
settings only; no differentiation or FD perturbation is run. This performs
three complete equilibrium/NTX database builds and stops before roots or
transport. It uses the corrected source already in `f45cdec`, so no further
executable code needs to be pushed. Historical pre-fix reference snapshots
are deliberately not supplied.

```bash
(
  cd "$HOME/NEOPAX" || exit 1
  mkdir -p outputs
  set -o pipefail
  env JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=true \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "$HOME/jax_env/bin/python" \
    examples/benchmarks/check_generic_ntx_database_baselines.py \
    --out-dir outputs/generic_ntx_baselines_axis_fix_20261116 \
    2>&1 | tee outputs/generic_ntx_baselines_axis_fix_20261116.log
) || printf '\nDatabase comparison stopped; terminal kept open.\n'
```

Passing this check would establish matching generated baselines for that run,
not AD/FD gradient accuracy. Fresh paired AD16/FD16 validation remains a
separate subsequent check.

The ordinary post-fix database comparison has now completed: all three builds
match in all captured leaves and all D11_log/D13/D33 table bytes. See the
[recorded result](generic_ntx_baselines_axis_fix_2026-11-16.md). The supplied
output is duplicated and is recorded once. The
[fresh generic AD16/FD16 instructions](axis_fix_ad_fd16.md) preserve both sides
and verify the actual transport baselines before recording derivative errors.
