# Eight-DOF transport AD/FD comparison with NTSS1D interpolation

The standard `Solve_Transport_equations_wHe_radau.toml` reads a saved database
using `preprocessed_3d_radial_ntss1d`. The recent realtime-geometry AD/FD
benchmarks instead build live NTX tables and use the `generic` interpolation
rule. They therefore do not test the same interpolation formula.

The new input
`examples/benchmarks/Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_ntss1d.toml`
copies the existing black-box AD/FD benchmark and changes only:

```toml
[neoclassical]
interpolation_mode = "preprocessed_3d_radial_ntss1d"
```

It retains the existing live NTX scan, its physical coefficient values and
field nodes, realtime VMEX geometry, initial profiles, sources, 51-point
transport grid and Radau configuration. This compares the interpolation on
the same realtime benchmark; it does not substitute the saved NTSS-comparison
database, equilibrium or other transport settings. The VMEX response remains
`raw_block`, the `v0_new`-style default.

## Forward and derivative contracts

`MonoenergeticRuntimeNTSS1D` retains the same 19 independent database leaves
as the generic live-scan database. Its radial and electric-field coordinate
views supply the existing
`get_Dij_preprocessed_3d_ntss_radius_ntss1d` kernel. No alternate NTSS formula
or duplicate coefficient tables are introduced. The interpolation selection
is propagated through the forward scan conversion and both scalar and
batched recorded scan transposes.

The NTSS1D electric-field interpolation has data-dependent slope limiting.
Its table derivative is obtained by transposing that actual kernel, for
centre fluxes, face fluxes and terminal bootstrap current. A two-knot linear
electric-field stencil would differentiate a different function.

The runner selects `established` for the transport, selected-root and bootstrap
table-transpose boundaries. `legacy_sparse` is specific to the generic
interpolator and is rejected for this new live NTSS1D representation. The
accepted-step map, recorded-forward Newton/Jacobian decisions, segmented
transport reverse and one final recorded NTX scan transpose are retained.

Both AD and FD summaries record `database_interpolation_mode`. FD retains
the frozen-linearized geometry and selected-root lanes and the
`reverse_frozen_metadata` accepted-replay contract.

## Local verification and remote execution

The local tests use NumPy and extracted production code, without JAX imports
or physical solves. They check identical coefficient leaves, derived
coordinates, the sole TOML difference, recorded-scan wiring and forward
agreement with the existing preprocessed NTSS1D representation at 36 queries
spanning radial branches, collisionality clamps and electric-field clamps.

The GPU checks remain to be run on the large machine. The Python runner first
executes three small checks: agreement with the established preprocessed
kernel, coefficient-table pullbacks versus FD, and AD/FD of the complete
scan-conversion/coordinate chain. These checks run no VMEX equilibrium, NTX
coefficient or transport solver. The full benchmark workers start only after
the checks pass.

Then the runner executes one AD worker and eight FD workers sequentially for:

- `n0`, `T0`, `density_shape_power`, `temperature_shape_power`,
  `density_shape_alpha`, `temperature_shape_alpha`;
- `RBC:1:0`, `ZBS:1:0`.

The default is 16 accepted steps, four-step reverse segments, FD relative
increment `3e-7` and absolute increment `1e-10`. Results, logs and snapshots
are copied into separate worker directories. A rerun reuses completed worker
reports when the saved configuration, source digest and increments agree.
Use a new output directory after changing dependencies.

On the large machine, after pushing and pulling the changes:

```bash
(
  cd "$HOME/NEOPAX" || exit 1
  git switch en/v0_main || exit 1
  git pull --ff-only origin en/v0_main || exit 1
  neopax_run=outputs/ntss1d_ad_fd16_20261116
  mkdir -p "$neopax_run" || exit 1
  set -o pipefail

  "$HOME/jax_env/bin/python" \
    examples/benchmarks/benchmark_transport_ntss1d_ad_fd.py \
    --out-dir "$neopax_run" \
    --accepted-step-limit 16 \
    2>&1 | tee "$neopax_run/comparison.log"
) || printf '\nComparison stopped after an error; terminal kept open.\n'
```

`--plan-only` prints all nine benchmark commands without importing JAX,
creating output files or launching workers.

The output `ad_vs_fd.csv` contains 104 transport-gradient pairs. Numerical
columns use scientific notation and relative differences are
`abs(AD-FD)/abs(FD)`. The AD report also retains the geometry-objective rows.
Each FD column is checked against the saved AD initial state, terminal state,
final time, geometry and database arrays. Their maximum absolute differences
and bitwise-match flags are saved. Objective values are checked with a
roundoff allowance of `1e-12*(1+abs(AD_value))`. A
`comparison_baseline_matches=False` row cannot establish an AD derivative
error at a common baseline; inspect the recorded differences first.

No full GPU benchmark or numerical JAX derivative check was run locally.

## GPU preflight failure and interpolation guard fix, 2026-11-16

The remote run at `822607f` stopped in the kernel checks before launching any
transport workers. The forward comparison and scan-conversion/coordinate AD
check passed. The coefficient-table pullback check found a nonfinite result;
the original assertion did not print its coefficient/query subcase.
The [supplied log](ntss1d_preflight_failure_2026-11-16.txt) is retained.

Source-only NumPy execution reproduced unsafe arithmetic for the upper-clamp
query `(radius=1.6, nu/v=0.002, Er/v=0.1)`: `_inpold_fixed` evaluated `0/0`
inside a `where` whose returned value was zero. Evaluating both conditional
branches also exposed nonfinite arithmetic in the inactive interior branch,
where the padded knot was `inf`. These computations can contaminate AD even
when the selected forward value is finite.

The fix guards the denominator before division, replaces only unused padding
with finite increasing coordinates inside the polynomial, and makes inactive
repeated-knot radial weights finite. The existing interpolation formula and
determinant cutoff are retained. Geometry and transport solver code is
unchanged.

The local regression executes both sides of the conditionals with NumPy
floating-point exceptions enabled. Six local interpolation checks pass;
the three GPU checks are skipped locally. Against the pre-fix capture,
[all 108 forward coefficient values](ntss1d_guard_forward_check_2026-11-16.csv)
are bitwise identical. This is forward/arithmetic verification, not a GPU AD
result. The GPU tests now label each coefficient/query subcase and also check
the coordinate chain with batched interior, exact-node and clamped queries.

After pushing/pulling the correction, rerun the Python comparison with a new
directory, for example `outputs/ntss1d_ad_fd16_guard_fix_20261116`. The old
directory records the previous source digest and failed preflight. The runner
again starts transport AD/FD only after its GPU kernel checks pass.

## Completed remote comparison at 417fd67

All nine preflight checks now pass, including the three GPU derivative checks.
The complete AD16 and eight FD16 workers also finish successfully.
The [saved results and all relative differences](ntss1d_ad_fd16_2026-11-16.md) distinguish
remaining geometry-gradient differences from the temperature_shape_power
baseline coefficient-table mismatch. Kernel checks passing is not a
certificate of complete transport-gradient agreement.

## Both interpolators after the fixed-order SOLVAX component checks

The SOLVAX candidate's seven component checks have now passed remotely across
the full run and the corrected FP32 fallback rerun. This validates the FP64
product, nested compilation, JVP/VJP, mapped operands and a small generated
solve/gradient, not the full NEOPAX Jacobian. The next comparison uses
NEOPAX `en/v0_main`, VMEX/NTX main-utility branches and SOLVAX
`en/neopax-reproducible-products` on JAX 0.10, without a compilation cache.

The existing runner now accepts the generic `black_box` TOML as well as its
NTSS1D counterpart. The configurations differ only in interpolation mode.
Each uses 51 radial points, NTX `25x31x64`, the same VMEX `ad` equilibrium and
`raw_block` response, sixteen accepted steps, all eight DOFs and all thirteen
transport objectives. The original relative/absolute FD increments remain
`3e-7` and `1e-10`. Different interpolation modes have separate baselines and
schedules; AD and FD are compared only within each mode.

Each FD worker uses the established `--ad-baseline-reference` contract. It
independently builds its runtime and coefficient tables and must match the AD
geometry/database and density/pressure exactly before taking the saved
unperturbed AD Er. The original fresh root is retained in the audit and its
difference is printed. A different initial state, terminal state or final time
stops the worker before perturbed endpoints. Saved AD coefficient tables are
never substituted into FD. The selected-root branch is still differentiated
through the existing frozen-linearized root lane.

The two geometry columns additionally use the saved AD equilibrium and raw
parameter tangent through `--ad-geometry-reference`, with the previously
validated extended-precision native geometry increments and `--device auto`.
This retains the original perturbation size and native geometry formulas.
The accepted-step replay still freezes the recorded Newton/Jacobian metadata.
No transport or derivative implementation changes in this runner update.

The saved-array comparison now checks bytes, shapes and dtypes, including
identical nonfinite diagnostics and signed zero. A zero finite maximum
difference alone cannot label differing arrays a match. A baseline mismatch
retains the CSV and stops subsequent workers. Per-mode outputs contain 104
gradient pairs, including AD/FD objective values and baseline-match flags,
with relative differences `abs(AD-FD)/abs(FD)` in scientific notation. The CSV
now identifies `database_interpolation_mode`. Worker logs retain wall time
and maximum RSS through `/usr/bin/time -v` when available.

Seven local NumPy/mock tests pass without importing JAX. They check both
nine-command plans, the sole TOML difference, cache-disabled worker settings,
saved-reference integrity, matching/nonmatching arrays (including NaNs, signed
zero and dtype differences), and full mocked execution of both modes with
104 rows or immediate rejection of a drifted first FD baseline. The fresh
NEOPAX GPU comparisons remain pending. Use fresh output directories after
dependency changes; existing comparisons must not be resumed on a new stack.

## First difference between independent forward baselines

The first fresh generic FD worker in `fixed_products_ad_fd16_NBobrI` stopped
before perturbations. Geometry, coefficient tables and the packed initial
state match AD exactly. After sixteen accepted steps, the terminal packed
state differs by `1.1010374834086178e-8` and time by
`2.141341737173924e-12`. This is a failed bitwise reproducibility check, not
an FD gradient result or an out-of-memory failure. The existing baseline
guard has no numerical tolerance; the solver's local error tolerances are
separate. No guard or transport/AD algorithm is changed here.

Both independent baselines reach `_radau_adaptive_schedule_rollout`, but
their setup differs. AD converts the scan wrapper to its fixed-table model
and captures checkpoint carries every four accepted steps. FD retains the
wrapper and does not capture checkpoint carries. These differences have not
been established as the cause of the remaining drift. Copying AD's timestep
map into FD would not identify that cause.

The saved-input control uses the failed worker's own tables and packed
initial state, after checking them against the AD capture. It executes four
sequential processes, each doing only an ordinary sixteen-step forward solve:

| Variant | Transport model | Context setup | Checkpoint capture |
| --- | --- | --- | --- |
| `fd` | Restored scan wrapper | FD preparation | None |
| `fd_fixed_model` | Fixed-table model, same database | FD preparation | None |
| `ad_no_checkpoints` | Fixed-table model, same database | Production AD static preparation | None |
| `ad` | Fixed-table model, same database | Production AD static preparation | Every four accepted steps |

All four compute their own adaptive schedule. The AD variants call the real
`prepare_reverse_static_setup`, replacing only initial-state construction
with the already-saved common physical state, so no root is solved. Their
reverse-context flags match the paired run; no VJP or reverse sweep executes.
The no-checkpoint control retains the same scalar Newton-reuse trace output.
The full initial carry, numeric context and original initial RHS (stored in
the initial predictor) are captured. `comparison.json` identifies the first
different scalar attempt and reports exact endpoint comparisons to both
archived baselines. An initial predictor difference alone need not explain
the trajectory: with `prev_dt=0`, the stage predictor ignores that history.

```bash
python examples/benchmarks/check_saved_transport_baseline_repeatability.py \
  --trace-ad-fd \
  --run-dir outputs/fixed_products_ad_fd16_NBobrI/generic \
  --out-dir outputs/independent_forward_trace \
  --accepted-step-limit 16
```

Use the same GPU environment and dependency branches as the archived paired
run, with a fresh output directory. The parent reuses the paired runner's
worker environment (GPU, x64, compilation caches disabled, no XLA flags).
Workers exit sequentially to release GPU
memory. No equilibrium, NTX scan, root solve, perturbed endpoint or reverse AD
run is launched. The restored wrapper omits the retained scan record; if the
controls do not reproduce the archived endpoints, their differences alone
cannot establish the cause of the original run. GPU validation is pending.
The initial diagnostic's first worker stopped before solver preparation:
the output-unpacking helper applied the default `1e-6` density floor to the
raw zero-He initial condition. Restoration now decodes the packed arrays
without projections and reconstructs only the omitted electron density.
The normal solver's projections remain unchanged. A local saved w-He capture
has 51 zero-He entries and now roundtrips byte-for-byte without importing JAX.
This corrects diagnostic restoration, not the original AD/FD transport drift.
Failed workers also print their traceback in the parent log automatically.
Sixteen NumPy/mock tests cover restoration, all four control paths, first
attempt ordering and exact bits, archive comparisons and worker failure.
The restoration regression executes the production packing and density-floor
functions with NumPy, including zero, signed-zero and sub-floor densities.

### Completed independent controls: first timestep divergence

Remote run `independent_forward_trace_OoidNV` completed all four controls.
Its full `comparison.json` reports:

| Comparison | First `next_dt` difference | First attempted-`dt` difference | Final time max abs | Final state max abs |
| --- | --- | --- | --- | --- |
| Scan wrapper vs fixed-table model | None | None | `0` | `0` |
| Fixed-table FD setup vs AD setup without checkpoints | Attempt 23 | Attempt 24 | `2.141341737e-12` | `1.101037483e-8` |
| AD setup without vs with checkpoints | Attempt 7 | Attempt 8 | `8.654673874e-12` | `4.450080127e-8` |

The checkpoint pair's saved initial carry and numeric context match exactly.
All controls retain identical active/accepted and Jacobian-reuse masks. The
first saved scalar difference is an error estimate at attempt 2:
`2.0413663173335623e-14` vs `2.0413663173335626e-14`. The controller clamps
the error used in its growth formula to at least `1e-12`, so that particular
last-bit difference alone does not establish the cause of the timestep split.
The wrapper's initial-RHS difference also does not change its trajectory.

The archived AD and FD endpoints are not reproduced exactly. These controls
isolate sensitivity of the ordinary numerical forward execution to its
surrounding graph; they do not identify a faulty arithmetic operation, prove
that checkpoint capture alone explains the archived run, or validate AD/FD
gradients. The next comparison retains the original compiled programs to
distinguish GPU product choices and arithmetic reassociation from a source
input difference. The source now captures the context actually passed to the
adaptive probe, before static setup attaches reverse-sweep schedule metadata.
First timestep differences and their contemporaneous error estimates are also
printed directly, rather than only the first differing diagnostic scalar.

Add `--capture-compiler-ir` to the same `--trace-ad-fd` command. The observer
returns the original executable from JAX's compiler, without recompilation,
extra JIT nesting, compiler options or XLA flags. It retains the largest
original scan program in each worker as `adaptive_stablehlo.mlir` and
`adaptive_compiled_hlo.txt`; `compiler_programs.json` lists observed scan
compilations. This diagnostic uses JAX's internal compiler API, inspected in
the local JAX 0.10 source. Its pass-through behavior and context capture pass
17 NumPy/mock tests; GPU IR capture remains pending. No transport method,
AD rule, model or baseline tolerance is changed.

### Original compiler programs: scalar reduction dispatch differs

The downloaded `forward_compiler_ir.tar.gz` contains completed run
`forward_compiler_ir_davv7R`. Offline NumPy/stdlib inspection imports no JAX
and executes no physics. The exported StableHLO text for `fd`,
`fd_fixed_model` and `ad_no_checkpoints` is byte-identical: 6,467,817 bytes,
SHA256 `226a2cca350b0e247253b6e824cafa69ee70441d08296ace57e661e11aaebf5a`.
Their compiled programs differ. All four cuBLAS matmuls select algorithm 0,
while several scalar reductions switch between native input-fusion and
Triton kernels. For `fd` versus `fd_fixed_model`, the changed sum-of-squares
kernels include:

| Compiled fusion | Input reduction | FD implementation | Fixed-model FD implementation |
| --- | --- | --- | --- |
| `input_reduce_fusion.36` | 204 FP64 entries squared and summed | Native | Triton, one warp |
| `input_reduce_fusion.38` | 51 FP64 entries squared and summed | Native | Triton, one warp |
| `input_reduce_fusion.39` | First 153 entries of a 408-entry FP64 vector, squared and summed | Native | Triton, one warp |

The pair's saved initial carry is identical, its accept/reject and
Jacobian-reuse masks match, and its terminal difference reproduces the
original failure's `1.1010374834086178e-8` state and
`2.141341737173924e-12` time gaps. In this new compilation, AD without and
with checkpoints have identical timesteps and terminal states; AD also
matches its archived endpoint exactly. Checkpoint recording alone is
therefore not a consistent explanation. Backend selection in ordinary
forward numerical execution is now a concrete discrepancy. These artifacts
do not yet prove that the three listed reductions alone cause all the drift:
some interpolation transpose fusions also choose different implementations.

The diagnostic-only `--solver-mean-reduction fixed` control specifies a fixed
GPU implementation of FP64, one-dimensional, default-argument means in
`NEOPAX._transport_solvers`. It uses `solvax._reproducible.schur_product`
already available in `en/neopax-reproducible-products`: summing a vector via
an all-ones row product, then dividing by its length. The same mean, RMS,
predictor and controller formulas remain in place. A module-local proxy
leaves JAX's global `mean`, the physics modules, non-FP64 means and explicit
axis/keyword reductions unchanged. No production solver source or default
changes; the control is restored on exit. Metadata reports the reduction
sizes actually traced. The existing native behavior is the default.

Run the same saved-input control in a fresh directory with both
`--solver-mean-reduction fixed` and `--capture-compiler-ir`. All four workers
still compute their own adaptive schedule. The comparison reports native
archived endpoints for reference: fixed floating-point reduction order can
change those last bits, so the candidate must first be judged by agreement
between the independent fixed-mean controls. GPU validation, performance
and the subsequent full AD/FD comparison remain pending. Eighteen local
NumPy/mock tests pass, including the actual production error-norm formula,
proxy scope restoration, unchanged fallback reductions and CLI dispatch.

### Fixed-mean control passed; paired AD/FD verification is next

Remote run `forward_fixed_means_5XfmCX`, downloaded as
`forward_fixed_means.tar.gz`, completed all four independent controls.
Offline NumPy inspection found exact shape/dtype/byte equality in every
saved trajectory array across all four variants, including error estimates,
accept/reject and Jacobian-reuse masks, attempted/next timesteps, step times,
final packed state and final time. All initial carries match. The maximum
state and time differences are both `0.000000000e+00`; final time is
`3.669427918687531e-4`. The AD endpoint also matches its native archived
endpoint exactly. The old FD archive retains the earlier `1.101037483e-8`
state and `2.141341737e-12` time differences; it is not the new common point.
Compiler output confirms the fixed product kernel, and all workers report
mean sizes `51, 153, 204, 408, 2856`. No AD or perturbed FD was evaluated in
this control, so gradient correctness and performance remain unverified.

The paired runner now accepts explicit `--solver-mean-reduction fixed` for
both interpolation configurations. It applies the same tested scope to the
ordinary AD and FD script entry points for the entire worker, including the
forward preparation, accepted replay and backward sweep. The original main
functions, parameter/objective lists, perturbation sizes, VMEX lane, frozen
root/Newton contracts and all baseline guards are unchanged. No production
NEOPAX package file or default is modified. Omitting the option retains the
native benchmark. Each worker is a fresh process, so a mean policy is never
switched after JIT compilation in a reused process.

The runner owns `NEOPAX_SOLVER_MEAN_REDUCTION`; a caller's inherited value is
cleared before the requested mode is set. Successful fixed workers record
the reduction sizes actually traced in `solver_mean_reduction.json`; the
runner rejects missing/empty evidence before continuing. Saved settings
include the mean mode, and CSV rows identify it, preventing native/fixed
results from being mixed on resume. Dry-run commands show the fixed scope.

```bash
python examples/benchmarks/benchmark_transport_ntss1d_ad_fd.py \
  --config examples/benchmarks/Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box.toml \
  --solver-mean-reduction fixed --accepted-step-limit 16 \
  --out-dir outputs/generic_fixed_means_ad_fd16

python examples/benchmarks/benchmark_transport_ntss1d_ad_fd.py \
  --solver-mean-reduction fixed --accepted-step-limit 16 \
  --out-dir outputs/ntss1d_fixed_means_ad_fd16
```

Use fresh directories. This is one AD16 plus eight FD16 workers per
interpolator, 104 gradient pairs per mode. Full GPU AD/FD runs are pending.
Nineteen saved-input/entry-point tests and eight paired-runner tests pass
locally without importing JAX. They cover both mean modes and interpolators,
strict failure on baseline drift, fixed-scope execution evidence and resume
rejection when changing the mean mode. This integration is an opt-in
benchmark verification step, not promotion of a production solver policy.

### Full fixed-mean benchmark still fails before FD perturbations

Run `fixed_means_ad_fd16_iHwp8o` used commit `c34eda9`. Generic AD16
completed with finite gradients and zero replayed state-endpoint differences.
Wall time was `51:28.00` and maximum host RSS was `9,603,072` KiB.
The first FD worker (`n0`) used the fixed scope too, but its independently
computed baseline differed by `1.7493235972190074e-11` in final time and
`8.994706846010558e-8` in packed state. The exact baseline guard stopped
before either perturbed endpoint; the remaining DOFs and NTSS1D did not run.
There are no valid AD/FD gradient comparisons from this run. This is not OOM.

The earlier four-worker success was conditional on restoring saved runtime
arrays and compiling in that diagnostic's context. It did not capture the
original full script calls. Fixed one-dimensional means therefore remain an
incomplete control; no additional numerical policy or baseline tolerance is
changed on the strength of that result. The current evidence does not locate
the first remaining differing arithmetic operation in the full benchmark.

`--trace-original-ad-fd` now launches just the ordinary AD entry point and
the first FD entry point with the failed run's configuration, original AD
root reference, original flags and chosen mean mode. Each script performs
its live preparation and its own adaptive forward solve. A pass-through
observer calls the original adaptive helper once with identical input
objects/options and captures its original compiler output. Afterward it
saves the initial carry/numeric execution context and active trajectory,
then exits before the transport backward sweep or FD perturbations.
No restored runtime or shared adaptive schedule is used. Fresh equilibrium,
NTX database and root preparation still occur, so this is not a NumPy-only
check. Workers run sequentially. Both fresh endpoints are compared against
their respective failed-run archives; failed reproduction remains visible.

```bash
python examples/benchmarks/check_saved_transport_baseline_repeatability.py \
  --trace-original-ad-fd --solver-mean-reduction fixed \
  --run-dir outputs/fixed_means_ad_fd16_iHwp8o/generic \
  --out-dir outputs/original_fixed_mean_forwards \
  --accepted-step-limit 16
```

Captures are under `ad/forward` and `fd/forward`: `setup.npz`, `native.npz`,
compiler IR and metadata, with `comparison.json` at the output root.
`NEOPAX_ORIGINAL_FORWARD_DIR` is reserved for this explicit diagnostic;
normal paired workers clear any inherited value. No benchmark summary is
created for a forward-only capture. The observer neither adds a JIT nor
changes compiler flags. NumPy/mock tests check original-call identity,
unchanged options, early exit/restoration, live-script dispatch and visible
archive differences. Full GPU capture remains pending.

### Original-call capture completed; the differing archived AD program is absent

Archive `original_fixed_mean_forwards_a7u5YD.tar.gz` was inspected locally
with NumPy and the standard library only. Both real entry points completed
16 accepted steps in 25 attempts. Every initial-carry field and every saved
trajectory array matches byte-for-byte between AD and FD, including error
estimates, timestep choices and final state/time. Common final time is
`3.6694279357176053e-4`. Numeric execution contexts differ only in expected
reverse-AD selectors, not forward solver parameters.

| Comparison | Maximum packed-state difference | Final-time difference |
| --- | --- | --- |
| Fresh AD vs fresh FD | `0` | `0` |
| Fresh FD vs archived FD (`fixed_means_ad_fd16_iHwp8o`) | `0` | `0` |
| Fresh AD vs archived AD (`fixed_means_ad_fd16_iHwp8o`) | `8.994706846e-8` | `1.749323597e-11` |

The new compiled programs are not byte-identical. Among uniquely matched
fusion operations, two mid-radius interpolation kernels use Triton in AD
and the native GPU emitter in FD. Their complete trajectories nevertheless
agree, so those dispatch differences alone do not establish the cause of
the older drift. Both programs contain the fixed solver-reduction kernels.
The source also confirms that this database path's terminal AD snapshot
uses the adaptive probe's retained final carry: the subsequent backward
sweep does not generate a different primal endpoint.

The archive contains the two new matching programs, not the compiler output
or initial cache from the original differing AD run. It cannot identify the
first arithmetic difference in that older execution. None of these captures
evaluated perturbed FD endpoints or completed the requested gradient table.

The existing observer can now record once and return the identical rollout
object, allowing the original benchmark to continue. Add `--capture-forward-ir`
to the paired runner for this mode. Captures are retained under each worker's
`forward` directory, including when its later baseline guard fails. Successful
workers require nonempty compiler output and exact agreement between the
captured forward point and their actual terminal audit snapshot. The
forward-only command still stops at the same boundary by default. The
normal runner clears inherited capture controls, and saved settings distinguish
the requested recording mode. No integrator, stage count, controller,
tolerance, nonlinear/linear method, numerical formula or acceptance criterion
is changed by this recording option.

```bash
python examples/benchmarks/benchmark_transport_ntss1d_ad_fd.py \
  --config examples/benchmarks/Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box.toml \
  --solver-mean-reduction fixed --capture-forward-ir \
  --accepted-step-limit 16 --out-dir outputs/generic_full_forward_capture
```

Use a fresh output directory. This runs the full AD/FD comparison, rather
than another forward-only control, and retains the compiled programs that
actually produced its reported primals. Existing strict baseline guards
remain active. Twenty-one saved-baseline tests and ten paired-runner tests
pass locally without importing JAX; full GPU validation is pending.

### Failing FD capture isolates a retunable stage sum

Archive `full_capture_ad_fd16_8NznN1_audit.tar.gz` contains the successful
`fd_n0` baseline and failed `fd_T0` baseline. Their complete saved setup
arrays match byte-for-byte. Their exported StableHLO files also match
byte-for-byte (SHA256
`f889af4e8f89d861d8f720c927c59b923bd935aa0798d3bf11f3b2caab10d9c0`).
After ignoring GPU dispatch configuration and the fusion-emitter kind, all
compiled HLO lines match. This is an independent compilation difference,
not a different DOF-specific mathematical program.

The first recorded difference is the attempt-7 error estimate:
`3.837114984735867e-2` versus `3.837114984737251e-2`. The next timestep
changes immediately, and the timestep attempted at slot 8 then differs.
Both runs retain identical accept/reject and Jacobian-reuse masks.
Terminal differences are `8.994706846010558e-8` in packed state and
`1.7493235972190074e-11` in time. The successful n0 comparison has 13
valid gradient pairs; the other seven FD columns remain unvalidated.

The endpoint's seven-by-408 weighted stage reduction is compiled as
`fused_reduce.66`: the successful n0 job uses the native emitter; the T0
job uses Triton. The other switched emitter kinds belong to pure
transposes, integer predicates and the interpolation-grid uniformity
predicate. All eight DOFs reach the shared `_radau_step_endpoint`; the
selected parameter is not an argument to this helper. This is a shared
compilation risk, not a dedicated CUDA implementation for each DOF.

The focused correction replaces generic weighted-stage dots with an
explicit balanced addition tree, separated by optimization barriers. The
same helper is used for the endpoint and the embedded stage-error sum.
The endpoint increment multiplication and final state addition are also
separated. The equations remain
`y_next = y + h * sum_i(b_i*K_i)` and
`err = h * (embedded_f0_weight*f0 + sum_i(b_error_i*K_i))`.
There is no change to the configured integrator, tableau, stage count,
controller/error formula, tolerances, nonlinear/linear method or acceptance
criteria. No dependency, compiler flag or persistent cache is introduced.
This fixes the identified retunable reduction; sufficiency for the full
benchmark still requires GPU validation.

`tests/test_radau_endpoint.py` loads the production helpers without importing
the NEOPAX package. Two local NumPy tests check the equation and FP error
bound for stage counts 1/2/3/5/7 and FP32/FP64, plus storage-layout invariance.
Three opt-in CUDA tests check exact endpoint bits across JIT/auxiliary/
transpose/batched contexts, JVP/VJP and second derivatives, and report
isolated runtime and compiled buffers against the previous endpoint.
These tests run no VMEC, NTX, root or transport solve.

```bash
JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=true \
XLA_PYTHON_CLIENT_PREALLOCATE=false NEOPAX_TEST_RADAU_ENDPOINT_GPU=1 \
python tests/test_radau_endpoint.py -v
```

Commit `ff7cc30` passed all five endpoint checks on the remote CUDA machine
with JAX 0.10.0 in 6.000 seconds. Exact endpoint bits agree across the tested
JIT/layout/batched contexts; JVP, VJP and second derivatives pass as well.

| Isolated compiled endpoint | Previous native reference | Fixed tree |
| --- | --- | --- |
| Median call time | `1.400935e-4 s` | `1.329005e-4 s` |
| Temporary device buffers | `32` bytes | `36,160` bytes |
| Generated code | `5,096` bytes | `11,752` bytes |

These timings are comparable in scale; they do not establish a full-solver
speedup or peak-memory change. The isolated temporary-buffer increase is
`36,128` bytes (about 35 KiB). The reported compile times are not a fair
cold-compile comparison: the fixed endpoint was already compiled by the
preceding reproducibility test. Full paired AD/FD validation remains pending.
Use a fresh output directory for the full benchmark because the numerical
source fingerprint has changed. The fixed-mean benchmark control and exact
baseline guards remain as before; this correction does not itself promote
the experimental mean policy to a production default.

### Full fixed-stage-sum run: six generic profile columns complete

Run `outputs/fixed_stage_sums_ad_fd16_KEiUFg` completed generic AD16 and
all six profile FD16 columns. Each independently computed baseline has
exactly matching initial state, final state, final time, geometry and
database arrays. All 78 gradient pairs are finite. Relative errors below
are `abs(AD-FD)/abs(FD)`, calculated from the full-precision printed values.

| Profile DOF | softmax Er | Bootstrap | Largest of 13 objective errors |
| --- | ---: | ---: | ---: |
| n0 | `5.108973e-9` | `1.995435e-8` | `3.135474e-7` |
| T0 | `9.041301e-10` | `1.111838e-9` | `3.343179e-8` |
| density_shape_power | `5.973674e-8` | `7.773003e-8` | `1.206375e-6` |
| temperature_shape_power | `3.512750e-8` | `5.171317e-9` | `3.512750e-8` |
| density_shape_alpha | `8.255685e-8` | `2.867540e-7` | `1.325398e-5` |
| temperature_shape_alpha | `9.260181e-9` | `1.146692e-8` | `1.837706e-8` |

The largest error is `Er_transition_right` versus `density_shape_alpha`.
These results establish exact baseline agreement for these six columns;
they do not establish zero gradient error or validate the geometry columns.

RBC FD stopped at the geometry/database support guard before transport
stepping or any parameter perturbation. Fresh `a_b` differs by
`2.220446049250313e-16`, `R0` by `1.7763568394002505e-15`, and D13 by up to
`4.790996367580513e-3`. Its saved `fd_geometry_linearization` and
`fd_reference_support` captures must be compared against this same run's
AD captures before attributing a cause. The log does not establish whether
the equilibrium state, geometry evaluation context or compiled arithmetic
first differs. ZBS and the NTSS1D comparison were not reached. This failure
does not provide a new measurement of the Radau stage-sum correction.

The uploaded `fixed_stage_sums_geometry_audit.tar.gz` confirms byte equality
for all 23 saved VMEC state/parameter/mask arrays and the six RBC state-tangent
arrays. A NumPy evaluation of the zero-step state expression changes no bits,
including signed zeros. The state is also identical to the earlier successful
`zbs_auto_geometry_WGuHfQ` input capture. All 147 stored native-surface fields
in the new FD support match that older input capture; 15 of 19 normalization
channel fields differ. The new AD transport geometry matches all 25 fields
of the older capture, while the new FD geometry differs in 18. This narrows
the remaining failure to transport-geometry evaluation and its normalization;
the archive does not retain the intermediate Boozer projections or their
compiled programs and therefore cannot identify the changing operation.

`check_saved_geometry_build.py` observes the original AD and RBC-FD entrypoints
through that boundary. The existing VMEX callback returns the saved state/mask
and validates the complete parameter arrays, preserving its ordinary caller
and placement. A separate guard disables equilibrium solves. Observation stops
before NTX coefficients, roots, transport or derivatives. The trace retains
volume inputs, VMEX fields, Fourier projection operands/results, Boozer
inputs/outputs, state-placement descriptors and actual Boozer/einsum compiler
output. It neither adds a JIT nor changes compilation options. Both geometry
outputs must reproduce their respective archived targets exactly before the
trace can localize that archived failure.

```bash
python examples/benchmarks/check_saved_geometry_build.py \
  --run-dir outputs/fixed_stage_sums_ad_fd16_KEiUFg/black_box \
  --out-dir outputs/fixed_stage_sums_geometry_build_capture
```

The three new NumPy/mock checks cover exact callback inputs (including signed
zero), preservation of the original compiler call/executable and alignment
of repeated trace boundaries. The existing trace-routing tests also run
without JAX; their mock FD arguments now include the current saved-geometry
and scan-audit options. No production calculation or baseline guard changes.
Remote reproduction and compiler analysis remain outstanding.

The uploaded `saved_geometry_build_audit.tar.gz` from
`outputs/saved_geometry_build_YD2V2G` did not reproduce the archived AD
geometry. Both captured paths agree byte-for-byte at all shared numerical
boundaries and match the archived FD geometry. Their six state arrays have
identical GPU placement, weak types and layouts. All thirteen captured
einsum/Boozer StableHLO programs match; after omitting diagnostic metadata,
the only optimized-instruction differences are transpose unroll factors.
Thus this control identifies no changing arithmetic operation and is not
validation of a correction. Its exit status 1 is the reproduction guard.

Restoring the saved callback output also omitted the original equilibrium
solve's compilation/execution history. `--fresh-equilibrium` now keeps that
configured AD solve completely untouched, including its callback, refinement
and device selection. FD retains the recorded linearization as in the failed
benchmark. Both paths still stop before NTX, roots and transport. The new
metadata also checks whether the first primal's actual geometry-input state
matches the later recorded derivative state used by FD. That equality is an
investigation result, not an assumption: reproducing the two archived geometry
outputs is the condition for attributing their difference. A mock
regression verifies that the fresh option leaves the original solve and
callback objects in place, while the saved option prevents an equilibrium
solve. This option changes only the diagnostic, not production methods.

```bash
python examples/benchmarks/check_saved_geometry_build.py \
  --fresh-equilibrium \
  --run-dir outputs/fixed_stage_sums_ad_fd16_KEiUFg/black_box \
  --out-dir outputs/fixed_stage_sums_geometry_original_history
```

### Fourier contraction isolated by comparing the two uploaded captures

In `saved_geometry_original_history_3vVKpe`, both the original AD equilibrium
and FD's restored state match the recorded derivative state byte-for-byte.
Both new paths produce the archived AD geometry, so this capture does not
reproduce the archived FD geometry. Its exit status 1 is the archive guard,
not a solve failure. The AD solve took `3:08.72`, with `3,363,488 KiB` maximum
host RSS; the saved FD build took `0:27.51`, with `1,892,188 KiB` host RSS.

Comparing the **FD path across the two captures** nevertheless reproduces
both original geometry values and isolates the first changing operation:

| Boundary | Earlier FD capture | New FD capture |
| --- | --- | --- |
| Equilibrium, VMEX fields, volume inputs | Exact common bytes | Exact common bytes |
| First projection values `(200,18,16)` and basis `(18,16,128)` | Exact common bytes | Exact common bytes |
| StableHLO for the projection | Same SHA256 | Same SHA256 |
| Matrix products | Four identical cuBLAS partial products, algorithm 0 | Same |
| Sum of four partial products | Triton reduction | Native GPU reduction |
| Projection output | 2,999 differing entries | Maximum absolute gap `4.547473508864641e-13` |

All 18 observed Fourier projections differ; their inputs match. The first
Boozer input coefficient differences follow these contractions. The compiled
Boozer-transform programs themselves match byte-for-byte between the two FD
captures. Thus the changing arithmetic is in NEOPAX's
`_neopax_vmex_boozer_table_inputs_from_state.project`, before the Boozer
transform. This conclusion uses matching-input FD-to-FD captures and does not
depend on claiming either individual AD/FD pair reproduced both archives.

The focused production correction flattens the same angular contraction and
uses `solvax._reproducible.schur_product`, already provided by the installed
`en/neopax-reproducible-products` candidate. Its CUDA FP64 kernel owns the
accumulation order. The projection remains
`weights[m] * sum_tz(values[r,t,z] * basis[t,z,m])`; its basis, grid, weights,
radial interpolation and ordinary bilinear derivatives are unchanged. No
time integrator, root or equilibrium method, tolerance or acceptance rule is
changed. CPU and non-FP64 products use SOLVAX's ordinary JAX product fallback.

`tests/test_vmec_boozer_projection.py` checks the actual production helper
and weighted even/odd caller through AST loading. Its NumPy check covers
FP32/FP64 and noncontiguous logical grids. Three opt-in CUDA checks exercise
bit equality across JIT/layout/batched contexts, value/JVP/VJP/second
derivatives, and isolated timing/compiled buffers. Set
`NEOPAX_VMEC_PROJECTION_CAPTURE` to either uploaded `geometry_build_*.npz`
to use the real operands instead of synthetic arrays. No physical solve
runs in these tests. GPU validation remains pending.

```bash
NEOPAX_TEST_VMEC_PROJECTION_GPU=1 JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=true \
python tests/test_vmec_boozer_projection.py -v

python examples/benchmarks/check_saved_geometry_build.py \
  --fresh-equilibrium --require-current-match \
  --run-dir outputs/fixed_stage_sums_ad_fd16_KEiUFg/black_box \
  --out-dir outputs/fixed_projection_geometry_check
```

The post-correction mode requires complete, exactly matching current AD/FD
geometry, channels, native surfaces, projections and equilibrium states.
Previous geometry rounding is retained as an archive reference rather than
the success criterion for changed source. The full AD/FD benchmark's strict
new-baseline guards remain untouched. Full comparisons still need fresh run
directories for all eight DOFs and both interpolators after the short checks
pass; neither archive analysis nor isolated GPU checks validate those gradients.
