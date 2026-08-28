# Optimization Path Plan

## Verified Payload-Stage Plan — 2026-08-28

### Measured cause

For the fixed four-objective geometry + initial-root case
(`maxEr`, `Er_left`, `Er_right`, `J_bootstrap`), one complete repeated
evaluation produces the following JAX dispatch-cache growth:

| Existing boundary | Entries added per evaluation |
| --- | ---: |
| shared raw VMEX solve | 1 |
| selected initial-`Er` root | 1 |
| payload-to-VMEC reverse table | **13** |
| final objective/table assembly | 3 |

The process RSS rises by about 0.68 GiB per evaluation while `jax.live_arrays`
is flat and the evaluation still performs exactly one raw VMEC solve and one
selected-root solve. The initial boundary attribution to the payload-to-VMEC
table was too coarse: an internal probe shows that table is entered after most
of the new entries already exist. The dominant source is instead the preceding
transport-support/root reverse that constructs the payload cotangents; it is
not geometry-vector extraction, VMEC parameter setup, or duplicated
geometry/root calculations.

The payload table calls the established
`geometry_payload_pullback_from_param_vector_raw_block_transpose` route.
The measurements show that only the final few entries occur in this boundary;
the larger part is created before entry. Reusing fixed Boozer/grid metadata
did not change the count or RSS slope, so that hypothesis is rejected. The
next step is diagnostic-only instrumentation around the existing compact
support/root reverse that creates the payload cotangents.

### Invariants

- The benchmark reverse-AD path is unchanged: same payload builder, custom
  derivative rules, raw-block transpose, root selection, numerical values,
  and benchmark interfaces.
- Each mixed evaluation retains one shared VMEC solve and one selected-root
  evaluation. No geometry calculation is duplicated per objective or DoF.
- No optimizer-wide fused JIT and no JIT is introduced inside VMEC, Boozer,
  NTX, flux, or root physics functions.
- The fixed TOML selects the flux model before a stage is built. The stage is
  keyed by model/layout and is rebuilt only when a new problem is built with a
  different TOML/model/grid/VMEC parameter layout.
- The first implementation is only the exact-Lij four-objective initial-root
  case. Full time-dependent transport is a separate later stage because its
  accepted-step topology can vary.

### Implementation plan

1. **Remove rejected experiments from the proposed optimization route.**
   The vector-only and split raw-parameter JIT experiments did not reduce the
   cache slope. Do not make them the example default.

2. **Identify the exact internal payload boundary before factoring anything.**
   The earlier structural-artifact reuse experiment is not a solution and
   must not become the optimization default. Instrument the existing payload
   pullback first; only the measured owner may be made persistent.

3. **Keep all trial data dynamic.**
   The persistent callable accepts the current raw VMEC state/parameters/mask,
   current support cotangent batch, and current objective/profile values. It
   must never retain an evaluation's state, geometry payload, NTX payload,
   root solution, support bars, or optimizer vector.

4. **Use the existing reverse operations unchanged.**
   The staged callable invokes the same existing geometry/NTX payload mapping,
   its existing VJP/JVP rules, and the same final
   `implicit_state_pullback_multi_rhs_raw_block_transpose`. It changes object
   lifetime and tracing boundaries only; it does not alter reverse-AD math or
   benchmark reference values.

5. **Expose it only through an explicit optimization mode.**
   `off` remains the benchmark/default evaluator. The optimization example
   switches to the stage only after acceptance checks pass. The public
   benchmark scripts and their payload functions keep their present behavior.

6. **Validate before measuring memory.**
   Compare residuals and the complete four-row Jacobian against `off` at the
   same vector. Use the existing tolerance (absolute residual parity; tiny
   floating reduction-level Jacobian differences only). Confirm one raw solve
   and one root solve.

7. **Acceptance memory run.**
   After parity, repeat the three-evaluation memory command. The payload-table
   cache count must remain fixed after warmup; only then assess RSS slope and
   warm execution time. If the payload count is flat but RSS still rises, the
   next diagnostic starts *inside* the already persistent payload operator,
   rather than adding a wider JIT.

### Expected performance/correctness effect

The stage has a one-time compilation/setup cost for its fixed layout. Warm
execution should avoid creating the 13 payload dispatch entries per optimizer
evaluation and should not add a large traced graph: VMEC, Boozer, NTX, and root
algorithms remain behind their existing reverse/custom-VJP boundaries. Because
the mathematical functions and cotangents are identical, derivative benchmark
values are expected to remain unchanged within existing floating-point
reduction tolerance.

## Current Restart Plan: Bound Existing Reverse Kernels

### Objective

Eliminate the repeated-RSS slope in geometry + initial-ambipolar-`Er`/bootstrap
least-squares optimization while retaining the benchmark-validated reverse-AD
mathematics and one shared VMEC geometry solve per trial.

### Non-negotiable constraints

- The benchmark/main reverse-AD path remains unchanged: no changed control flow,
  default arguments, formulas, solver settings, custom derivative rules, or
  benchmark numerical references.
- Geometry is evaluated once per mixed trial and shared by geometry and
  initial-root/ambipolarity/bootstrap objectives.
- Do not add an optimizer-wide fused JIT or JIT inside local VMEC/Boozer/NTX/
  root physics functions.
- Initial scope is the fixed exact-Lij, four-objective case: `maxEr`,
  `Er_left`, `Er_right`, and `J_bootstrap`. Full transport is a later,
  independent stage.

### Why the previous staged attempt is rejected

Retained Python wrappers do not retain JAX executables. They preserved
numerical parity but remained slower and showed RSS growth. Wrapping the whole
root/geometry/Boozer path in an outer JIT is also rejected: it forces setup
operations outside their established benchmark/custom-VJP structure into one
new trace.

### Implementation sequence

1. Remove/disable the incomplete optimization-only outer-JIT route and its
   associated experimental Boozer-precompute plumbing. Preserve the benchmark
   route verbatim.
2. Audit VMEX persistent optimizer callbacks and the corresponding *existing*
   NEOPAX reverse kernels/factories. Identify which executable/cache object is
   recreated or retained per repeated initial-root evaluation.
3. Add diagnostics around the existing correct path only: RSS, live JAX-array
   count, available JAX trace-cache sizes, and identities/counts of existing
   reverse kernel objects.
4. Build a new optimization-only stage that creates and retains those existing
   lower-level benchmark kernels once per fixed TOML/model/objective/VMEC
   layout. Do not copy VMEC/Boozer/NTX/root physics and do not wrap those
   preparation paths in a new outer JIT.
5. Each trial supplies fresh dynamic geometry/support/root data to the retained
   kernels. The existing payload-to-VMEC pullback remains the final reverse
   boundary.

### Required acceptance checks

Before any memory comparison:

1. Four-objective residual/Jacobian parity against the unchanged benchmark
   path.
2. Explicit confirmation of one shared `raw_block_solve` per mixed trial.
3. Stable retained-kernel/cache identity across three repeated calls.

Then compare eight repeated calls of the benchmark/default route and the new
stage. Accept the stage only if RSS no longer has a material per-evaluation
slope and warm execution is not regressed.

### Step 2 audit result

The exact initial-root benchmark path currently has no retained, outer root
executable to reuse. `initial_er_selected_root_profile` calls
`solve_ambipolarity_roots_radial_jax`, whose fixed-shape work is expressed with
`jax.vmap`/`jax.lax.map`; it does **not** create `jax.jit` objects. The separate
host-facing ambipolarity entrypoint does create short-lived `jax.jit` closures,
but it is not called by this reverse-AD path and is not a candidate for this
fix.

The VMEC raw-block static setup is already retained by
`GeometryRawBlockStage`, built once when the least-squares problem is created.
For every mixed evaluation the benchmark evaluator creates exactly one dynamic
`GeometryRawBlockSolve`, then passes that same solve first through the
initial-root transport table and then through the geometry table. Therefore a
new root wrapper, a new outer JIT, or a second geometry cache would not address
the observed slope and must not be introduced.

The next diagnostic step must instead locate native/JAX allocations made below
the already-correct reverse boundaries, while proving the one-solve topology
on every measured call.

### Step 3 diagnostic result and implementation target

The two-call checkpoint run isolated the retained memory to the existing VMEX
raw-block solve: its RSS had risen by `+909.7 MiB` before the selected-root
call, while the selected-root checkpoint was unchanged.  The final
payload-to-VMEC pullback released part of the temporary working set, leaving
the observed `+659.1 MiB` process-RSS slope.  Clearing JAX caches and trimming
only free native heap removed that slope, so this is native callback/executable
cache retention, not live JAX arrays or a retained physical solution.

VMEX's one-shot `solve_implicit_with_aux` constructs a fresh
`functools.partial(_host_solve_and_mask, cfg)` on each call.  This was a
plausible callback-cache candidate, but an optimization-only stable-callback
experiment had exact residual/Jacobian parity and **still** grew by `+679 MiB`
on its second measured call.  It is therefore rejected and removed; callback
identity alone is not the retaining cache.  The one-shot benchmark call
remains unchanged.

The next implementation is the actual VMEX-style boundary: retain one small
`jax.jit` kernel per optimization stage for
`ImplicitParams -> solve_implicit_with_aux -> (VMEC state, dof mask)`.  It
contains no Boozer, NTX, ambipolarity, bootstrap, or reverse payload work;
those remain on their benchmark paths.  This changes only compilation
lifetime, not the raw-block primal or transpose mathematics, and requires
residual/Jacobian parity before its memory measurement.

Date: 2026-07-30

## Guiding Rule

Do not contaminate benchmark paths while developing the optimization path.

The benchmark scripts and benchmark-validated internal paths are validation references. Optimization-specific plumbing must live in NEOPAX internals and thin optimization scripts. If a change would affect a benchmark path or a benchmark-validated function, stop and confirm before changing it.

## Goal

Build a VMEX-like optimization interface for NEOPAX objectives where user scripts mostly declare:

- input VMEC geometry,
- active profile and/or geometry parameters,
- objective terms, targets, and weights,
- optimizer settings.

The script should not manually build raw-block solves, NTX payloads, backend dictionaries, or reverse table plumbing.

Example user-facing shape:

```python
terms = [
    (geometry.boozer_qi_objective, 0.0, QI_WEIGHT),
    (geometry.boozer_maxj_objective, 0.0, MAXJ_WEIGHT),
    (geometry.vmec_aspect_ratio, ASPECT_TARGET, ASPECT_WEIGHT),
    (geometry.vmec_iota_mean, IOTA_TARGET, IOTA_WEIGHT),
    (geometry.vmec_mirror_ratio, MIRROR_TARGET, MIRROR_WEIGHT),
    (transport.softmax_Er, 30.0, ER_WEIGHT),
]

problem = build_geometry_transport_optimization_problem(
    config_path=...,
    vmec_input=...,
    parameterization=...,
    terms=terms,
)
```

## Parameter Scope

The optimizer must support the full parameter vector:

```text
x = [profile DOFs, VMEC geometry DOFs]
```

Required modes:

- `profile_only`
- `geometry_only`
- `profiles_plus_geometry`

Profile DOFs include the validated transport reverse profile parameters, such as:

- `n0`
- `T0`
- `density_shape_power`
- `temperature_shape_power`

Geometry DOFs include VMEC boundary harmonics and VMEX-like parameterizations:

- explicit specs such as `RBC:1:0`
- packed/scaled parameterizations such as `vmex_packed`
- discovered harmonic sets such as all nonzero `RBC/ZBS` modes

## Shared Primal Point

For each optimizer parameter vector, build one shared primal point:

```text
profile DOFs -> pre-root transport state
geometry DOFs -> VMEC raw-block solve/state
VMEC state -> NEOPAX geometry payload
VMEC state -> NTX support payload
pre-root state + geometry/NTX payload -> selected initial Er root
```

The shared object should hold:

```text
raw_block_solve
vmec_state
dof_mask
param_entries
geometry payload
ntx_support payload
pre-root profile state
rooted initial-Er state
```

## Objective Families

Geometry objectives depend on geometry DOFs:

```text
QI, maxJ, aspect, iota, mirror, well, etc.
```

Transport initial-Er objectives depend on profile and geometry DOFs:

```text
profile DOFs -> profile values/root residual/objective
geometry DOFs -> geometry payload/NTX support/root residual/objective
```

The currently tested geometry-only case is only one slice of the full problem.

## Correct Reverse Structure

Do not run geometry and transport as two independent heavy pullback graphs that each perform their own final VMEC raw-block transpose.

Instead split each objective into:

```text
objective value
profile cotangent contribution
VMEC-state / geometry-payload cotangent contribution
direct parameter contribution if needed
```

Then fuse the final geometry pullback:

```text
geometry objective VMEC-state cotangents
+ transport Er-root geometry/NTX payload cotangents
-> one batched raw-block transpose
-> VMEC harmonic Jacobian columns
```

Profile columns are handled separately:

```text
transport Er-root residual cotangents
-> compact profile/state pullback
-> profile Jacobian columns
```

Geometry-only objectives get zero profile columns.

## Final Jacobian Assembly

Assemble all rows into:

```text
J = [profile columns | geometry columns]
```

For each residual row:

```text
geometry objective:
  profile columns = 0
  geometry columns = fused raw-block result

transport Er-root objective:
  profile columns = compact profile pullback
  geometry columns = fused raw-block result
```

This gives the optimizer the same mathematical object as a monolithic least-squares graph:

```text
residual vector r
Jacobian matrix J = dr/dx
```

without retaining multiple heavy VMEC/Boozer/NTX pullback graphs at once.

## Validation Matrix

Validate incrementally:

- root-only benchmark unchanged,
- full transport reverse benchmark unchanged,
- geometry-only objective table unchanged,
- `profile_only + Er-root`,
- `geometry_only + Er-root`,
- `geometry_only + QI/maxJ/aspect/iota/mirror`,
- `geometry_only + QI + Er-root`,
- `profiles_plus_geometry + Er-root`,
- `profiles_plus_geometry + QI + Er-root`,
- `vmex_packed` geometry DOFs.

Each validation should compare against the corresponding benchmark or FD/frozen-linearized reference where available.

## Step 1: Protected Reference State

Status: completed as a guardrail; implementation Step 1 started below.

Before implementing the fused optimization evaluator, keep the validated benchmark behavior as the reference state. Do not edit benchmark scripts to make optimization pass.

Protected benchmark/reference commands:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --reverse-parameter-mode profiles_plus_realtime_geometry \
  --reverse-geometry-parameter RBC:1:0 \
  --objective all \
  --initial-Er-root-ad jax_selected_root \
  --initial-Er-root-only-optimization-smoke
```

This validates the compact initial-Er root-only geometry-active path. The known-good `RBC:1:0` Jacobian entries are:

```text
transport:softmax_Er                         -5.1293330713e+01
transport:smooth_root_proxy                   2.2505242252e-09
transport:Er2_volume_average                 -1.8476397879e+02
transport:Er_volume_average                  -2.0622727790e+01
```

```bash
python ./examples/optimization/transport_realtime_geometry_reverse_smoke.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --reverse-geometry-parameter RBC:1:0 \
  --objective all \
  --accepted-step-limit 2 \
  --reverse-segment-length 1 \
  --initial-Er-root-ad jax_selected_root \
  --ntx-exact-derivative-mode direct \
  --ntx-exact-derivative-field-pullback-mode generic_jvp \
  --radau-jacobian-reuse-mode legacy \
  --reverse-stage-adjoint-solve-mode bicgstab \
  --reverse-rhs-transpose-mode explicit_ntx_interpolated \
  --reverse-step-bwd-mode reduced_cotangent \
  --hide-solver-iterations
```

This validates the 2-step internal realtime-geometry transport reverse path. The latest good `RBC:1:0` geometry-column entries were:

```text
transport:softmax_Er                         -5.1317446794e+01
transport:smooth_root_proxy                   4.1615832428e-09
transport:Er2_volume_average                 -1.8107833322e+02
transport:Er_volume_average                  -2.0555620527e+01
transport:electron_temperature_volume_average -1.3670544542e-02
transport:total_pressure_volume_average      -7.7574038513e-02
transport:alpha_power_volume_average         -1.7350582044e-03
```

Step 1 completion criteria:

- benchmark scripts remain unmodified,
- protected reference commands and expected values are recorded,
- optimization work proceeds through NEOPAX internals and thin scripts only,
- any future change to benchmark-validated internal behavior requires an explicit decision before editing.

## Implementation Steps

### Step 1: Internal Shared Primal Builder

Status: implemented initial API.

Add a NEOPAX-internal shared primal object for mixed geometry/transport optimization. It must:

- accept a full mixed optimization vector `[profile DOFs | geometry DOFs]`,
- extract VMEC-boundary entries without assuming `geometry_only`,
- solve VMEC once through the raw-block-compatible lane,
- build NEOPAX geometry and NTX support once,
- provide a detached payload for objective-value/root computations,
- retain the raw-block solve for the final VMEC harmonic pullback.

Initial implementation:

```python
build_shared_geometry_transport_payload(...)
```

returns:

```python
SharedGeometryTransportPayload(
    raw_block_solve=...,
    payload=...,
    detached_payload=...,
    runtime_with_payload=...,
    vmec_parameter_values=...,
    vmec_specs=...,
)
```

### Step 2: Objective Cotangent Collection API

Status: implemented initial internal collectors.

Refactor geometry and transport initial-Er objective paths so they can return objective values plus cotangents to shared VMEC state/payload, instead of immediately doing separate raw-block parameter pullbacks.

Initial internal pieces:

- `ObjectiveCotangentTable`
- `geometry_full_ad_objective_cotangent_basis(...)`
- `geometry_full_ad_objective_cotangent_table(...)`
- `initial_er_root_only_objective_cotangent_table(...)`

These are additive and are not wired into benchmark-good runs yet. The geometry cotangent table returns VMEC-state cotangents before the final raw-block transpose. The initial-Er root-only cotangent table returns compact profile columns and geometry/NTX payload cotangents before the final payload-to-VMEC pullback.

### Step 3: Fused Raw-Block Geometry Pullback

Status: implemented initial internal path and wired into the optimization smoke script.

Concatenate VMEC-state/payload cotangents from geometry objectives and transport Er-root objectives, then run one batched raw-block transpose for all geometry columns.

Initial internal pieces:

- `geometry_raw_block_transpose_from_state_bars(...)`
- `geometry_payload_pullback_from_param_vector_raw_block_transpose(..., return_state_bars=True)`
- `fused_geometry_parameter_matrix_from_cotangent_tables(...)`

These are opt-in and not wired into benchmark-good runs yet.

The optimization-facing script now calls:

```python
evaluate_geometry_initial_er_root_only_least_squares_fused(...)
```

instead of manually constructing separate geometry and transport backends.
This keeps raw-block solve, realtime geometry/NTX payload construction,
cotangent collection, and final fused geometry pullback inside NEOPAX internals.

Current script status:

- `examples/optimization/optimize_geometry_qi_max_er_initial_root.py` is now a thin smoke/evaluation script.
- It still owns TOML/config preparation, term declaration, parameter selection, and output formatting.
- It no longer owns raw-block solve construction, shared geometry/NTX payload construction, backend dictionaries, or separate assembled pullback calls.
- It exposes `profile_only`, `geometry_only`, and `profiles_plus_geometry` modes, with the initial parameter vector assembled in `ReverseADParameterSet.specs` order.
- The benchmark-good initial-Er root-only graph shape is the reference behavior.
- For geometry-active transport/root objectives, the optimization evaluator must internally call the same table shape used by the benchmark-good path: all `INITIAL_ER_ROOT_ONLY_OBJECTIVES`, canonical profile columns, and the active VMEC columns. The optimizer-facing result then slices to the requested objective rows and requested parameter columns.
- The previous smaller-looking optimization call shape, e.g. only `softmax_Er` and only geometry columns, reached the same payload-to-state-bar code but changed the compiled graph and OOMed at `_state_bar_batch_from_payload_branch`.
- The current implementation adds `_adapt_objective_table_result(...)` and routes geometry-active transport/root optimization through the benchmark-style table before projection. This does not touch benchmark-good scripts.
- Shared/fused geometry work still needs to be built around the benchmark-style transport table, not by replacing it with a different payload map. Avoid passing retained payload objects or pre-existing raw-block solves into the transport/root pullback until that exact graph is validated.

### Step 4: Profile Column Assembly

Status: pending.

Add profile-gradient columns for transport objectives and zero profile columns for geometry-only objectives.

### Step 5: Thin VMEX-Like Problem Builder

Status: pending.

Expose an internal builder that returns a clean residual/Jacobian callable. The example script should only declare terms, parameters, input files, and optimizer settings.

### Step 6: Validation Gates

Status: pending.

Validate:

- protected root-only benchmark unchanged,
- protected 2-step transport reverse smoke unchanged,
- geometry objective table unchanged,
- mixed one-harmonic `QI + Er`,
- mixed full geometry terms plus Er,
- profile-only and profiles-plus-geometry modes,
- packed/all-harmonic geometry parameterization.

## Implementation Notes

The current optimization script is a debugging scaffold and is too plumbing-heavy. Its raw-block solve and payload construction should move into NEOPAX internals.

The long-term script should be thin and VMEX-like. It should not manually wire:

- `shared_raw_block_solve`,
- `shared_geometry_payload`,
- backend dictionaries,
- root-only reverse tables,
- payload pullback internals.

Those belong in the internal optimization problem builder.

## Current State: 2026-07-30 07:25

Goal:

- Build a VMEX-like optimization path for geometry objectives plus initial-Er/root transport objectives.
- Keep benchmark-good AD lanes untouched.
- Use the same validated internal pieces as the benchmarks:
  - geometry objective table for QI/maxJ/aspect/iota/mirror,
  - compact initial-Er root cotangent construction,
  - compact payload-to-VMEC raw-block transpose,
  - raw-block VMEC parameter pullback.

Known-good references:

- Geometry objective benchmark with `compare_geometry_qi_frozen_linearized_fd.py` verifies the optimization-internal geometry table:
  - `boozer_qi_objective` on `input.QI_nfp2_initial`, `RBC:1:0`: internal reverse table matches forward JVP at relative error about `2e-11`.
  - `boozer_maxj_objective` on `input.QI_nfp2_initial`, `RBC:1:0`: internal reverse table matches forward JVP at relative error about `4e-11`.
- Initial-Er root-only benchmark smoke verifies compact root AD and compact payload pullback for geometry-active transport rows.
- Two-step internal realtime-geometry optimization smoke verifies transport time-evolution AD internals for `profiles_plus_realtime_geometry`.

Current optimization script under test:

```bash
python ./examples/optimization/optimize_geometry_qi_max_er_initial_root.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --vmec-input ./examples/inputs/input.QI_nfp2_initial \
  --parameter-mode geometry_only \
  --geometry-parameters RBC:1:0 \
  --max-er-target 30 \
  --max-er-weight 1 \
  --qi-weight 1 \
  --maxj-weight 0 \
  --aspect-weight 0 \
  --iota-weight 0 \
  --mirror-weight 0
```

Current code change:

- Only `NEOPAX/_reverse_ad_optimization.py` is modified.
- The optimization fused evaluator now blocks/materializes transport root cotangents before entering payload-to-VMEC pullback, matching the benchmark-good memory staging.
- The optimization fused evaluator now requests only transport objective rows present in the least-squares terms. For the command above, this is only `softmax_Er`, not all four root-only objectives.
- For zero VMEC geometry deltas, the fused evaluator now uses the runtime geometry/NTX payload for root cotangents instead of building a duplicate current payload.
- For nonzero VMEC geometry deltas, the fused evaluator builds the current support payload from the shared raw-block VMEC state, uses it for root cotangents, then drops it before the payload-to-VMEC VJP.

Current unresolved problem:

- The optimization script still OOMs before QI/geometry objective table construction.
- The OOM occurs inside `geometry_payload_pullback_from_param_vector_raw_block_transpose(...)`, during `_state_bar_batch_from_payload_branch("ntx_support", ...)`.
- The failing allocation is in `booz_xform_jax` called from `_boozer_rmnc00_from_state_at_rho(...)` while constructing the VJP of `build_ntx_exact_lij_support_from_vmec_state(...)`.
- The stack shows the failure happens in the transport/root payload-to-VMEC VJP, not in the final fused raw-block transpose and not in the QI objective table.

Likely discrepancy still to investigate:

- The benchmark-good root-only path enters payload-to-VMEC pullback through `realtime_geometry_transport_reverse_table_from_payload_cotangents(...)`, receives parameter gradients immediately, and does not need to retain any fused geometry-objective state-bar machinery.
- The fused optimization path enters the lower-level `geometry_payload_pullback_from_param_vector_raw_block_transpose(...)` directly so that geometry-objective state bars can later be appended before one final raw-block transpose.
- Even with QI deferred, this direct fused path may still retain extra Python/JAX references around the payload cotangent table or shared payload compared with the benchmark wrapper.

Next debugging steps:

- Compare live objects held by `evaluate_geometry_initial_er_root_only_least_squares_fused(...)` at the moment it calls `fused_geometry_parameter_matrix_from_cotangent_tables(...)` with the benchmark-good root-only call into `geometry_active_initial_er_root_only_reverse_table(...)`.
- Check whether the fused path can call a benchmark-style compact helper that returns both:
  - transport parameter-gradient rows, and
  - optionally a compact raw-block solve/state-bar interface for appending geometry rows,
  without forcing the NTX support VJP to coexist with geometry objective data.
- If one final raw-block transpose cannot be made memory-safe, the fallback should be an explicit optimization option that uses benchmark-table behavior for transport/root and geometry separately. That fallback is correct and benchmark-equivalent but not fully fused, so it should be a deliberate mode, not hidden as the default fused path.

## Current State: 2026-07-30 Working Baseline Recovery

What worked:

- `examples/optimization/optimize_geometry_qi_only.py` is now the working VMEX-like geometry-only optimizer shape.
- It calls `NEOPAX.optimization.geometry_least_squares_problem(...)`.
- That public wrapper calls the internal `geometry_full_ad_reverse_table(...)`, the same geometry objective reverse table validated by the frozen-linearized FD/JVP benchmarks.
- The geometry-only script keeps user-facing objective terms in the script and avoids benchmark imports or raw-block plumbing.

What was going wrong:

- `examples/optimization/optimize_geometry_qi_max_er_initial_root.py` had become a CLI-heavy diagnostic script.
- Its mixed `QI + initial-Er root` path called the experimental fused payload evaluator:

```text
evaluate_geometry_initial_er_root_only_least_squares_fused(...)
-> geometry_payload_pullback_from_param_vector_raw_block_transpose(...)
-> _state_bar_batch_from_payload_branch("ntx_support", ...)
```

- That path OOMed before the geometry objective table could help, so the immediate issue was not QI/maxJ itself. It was the transport/root payload-to-VMEC support VJP being entered with the wrong optimization graph shape.
- The benchmark-good initial-Er root-only path still worked because it goes through the compact benchmark-validated table assembly path:

```text
geometry_active_initial_er_root_only_reverse_table(...)
-> realtime_geometry_transport_reverse_table_from_payload_cotangents(...)
```

What was redone now:

- Added `NEOPAX.optimization.GeometryInitialErRootLeastSquaresProblem`.
- Added `NEOPAX.optimization.geometry_initial_er_root_only_least_squares_problem(...)`.
- This wrapper intentionally calls `evaluate_geometry_initial_er_root_only_least_squares_benchmark_tables(...)`, not the experimental fused payload evaluator.
- Rewrote `examples/optimization/optimize_geometry_qi_max_er_initial_root.py` into the same style as the working geometry-only script:
  - constants at the top,
  - objective terms in the script,
  - no benchmark imports,
  - no direct raw-block/payload plumbing,
  - no CLI flags.
- Benchmark scripts were not edited.

Important limitation:

- This is the correct recovered baseline for optimization behavior, but it is not the final single-pullback fused implementation.
- Mixed transformed geometry penalties are rejected for now instead of silently applying the wrong chain rule. Geometry-only transformed penalties still work through `geometry_least_squares_problem(...)`.
- The mixed script currently uses a direct `vmec_mirror_ratio` target rather than a one-sided mirror penalty.

Next implementation target:

- Redo fused mixed optimization only after the recovered baseline is validated.
- The fused version must be built from the benchmark-good table behavior, not by replacing it with a new lower-level payload VJP shape.
- The desired future fuse is:

```text
benchmark-good transport/root compact table
+ geometry objective VMEC-state cotangents
-> one memory-safe compact geometry raw-block transpose
```

- Until that exact graph is validated, the default optimization wrapper should remain benchmark-table based so optimization correctness and memory behavior match the trusted references.

## Current State: 2026-07-30 Compact Optimization Pullback Debug

Immediate objective:

- Make `examples/optimization/optimize_geometry_qi_max_er_initial_root.py` run the VMEX-like mixed optimization setup without OOM and without contaminating benchmark scripts.
- Preserve benchmark-good derivative behavior and revalidate any benchmark/internal path touched by the compact pullback changes.

Current script under test:

```bash
python ./examples/optimization/optimize_geometry_qi_max_er_initial_root.py
```

The script currently starts with:

```text
max_mode=1
parameter_count=8
parameters=[
  RBC:0:1, RBC:1:-1, RBC:1:0, RBC:1:1,
  ZBS:0:1, ZBS:1:-1, ZBS:1:0, ZBS:1:1
]
```

Current code changes:

- Only `NEOPAX/_geometry_autodiff.py` is modified in the latest working tree checkpoint.
- The compact optimization payload pullback now attempts a tangent-contraction route:

```text
VMEC boundary parameter tangent
-> implicit_state_tangent_raw_block(...)
-> payload JVP
-> cotangent dot tangent
-> objective-by-parameter geometry Jacobian block
```

- This is intended to avoid the previous OOM-prone route:

```text
payload cotangents
-> generic payload VJP to VMEC state bars
-> raw-block transpose to boundary parameters
```

- The fallback state-bar/VJP route remains available for benchmark diagnostics or callers that request branch/state bars.
- The compact tangent route activates only when:

```text
combined_payload=True
return_branch_gradients=False
return_state_bars=False
extra_state_bars is None
extra_state_bars_factory is None
implicit exposes implicit_state_tangent_raw_block
```

Important diagnostic print expected on the intended path:

```text
compact_payload_tangent_contract=True
```

What the latest attached run showed:

- The optimization did enter the compact tangent path.
- It no longer failed at the earlier `jax.vjp(...)` OOM site.
- It failed inside `geometry_setup["tangent_contraction"]`, specifically while `jax.jvp(...)` evaluated `_build_neopax_geometry_from_state(...)`.
- First failure after the compact path change:

```text
jnp.unique(...) ConcretizationTypeError
```

- Follow-up failure after replacing `jnp.unique` locally:

```text
TracerArrayConversionError at np.asarray(context.static.s[1:], dtype=float)
```

Root cause of those two failures:

- The compact tangent path calls `_build_neopax_geometry_from_state(...)` under `jax.jvp(...)` and `jax.lax.map(...)`.
- Any dynamic uniqueness or NumPy conversion inside that function can see JAX tracers.
- The sampling radii are static sampling metadata, not physical differentiable values, so they should be computed outside the transformed payload function.

Latest fix:

- Added `_neopax_geometry_requested_sample_rho(context, n_r=...)`.
- `geometry_payload_pullback_from_param_vector_raw_block_transpose(...)` now precomputes `geometry_requested_sample_rho` before defining the JVP'd `geometry_from_state(...)`.
- `_build_neopax_geometry_from_state(...)` accepts `requested_sample_rho=...` so the transformed geometry function no longer computes sample uniqueness from traced values.
- `r00_support_rho` for NTX support now uses static sorted unique sample grids (`rho_center_sample`, `rho_face_sample`) so interpolation coordinates stay well-defined.

Checks passed after latest fix:

```bash
python -m py_compile NEOPAX/_geometry_autodiff.py NEOPAX/_reverse_ad_optimization.py NEOPAX/optimization.py examples/optimization/optimize_geometry_qi_max_er_initial_root.py
git diff --check -- NEOPAX/_geometry_autodiff.py
```

Validation still needed:

1. Rerun the mixed optimization script:

```bash
python ./examples/optimization/optimize_geometry_qi_max_er_initial_root.py
```

2. Confirm it prints:

```text
compact_payload_tangent_contract=True
```

3. If it completes the initial evaluation, compare the transport/root derivative rows against benchmark-good values for at least `RBC:1:0`.

4. Recheck benchmark/internal derivative references because `_geometry_autodiff.py` changed:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --reverse-parameter-mode profiles_plus_realtime_geometry \
  --reverse-geometry-parameter RBC:1:0 \
  --objective all \
  --initial-Er-root-ad jax_selected_root \
  --initial-Er-root-only-optimization-smoke
```

```bash
python ./examples/optimization/transport_realtime_geometry_reverse_smoke.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --reverse-geometry-parameter RBC:1:0 \
  --objective all \
  --accepted-step-limit 2 \
  --reverse-segment-length 1 \
  --initial-Er-root-ad jax_selected_root \
  --ntx-exact-derivative-mode direct \
  --ntx-exact-derivative-field-pullback-mode generic_jvp \
  --radau-jacobian-reuse-mode legacy \
  --reverse-stage-adjoint-solve-mode bicgstab \
  --reverse-rhs-transpose-mode explicit_ntx_interpolated \
  --reverse-step-bwd-mode reduced_cotangent \
  --hide-solver-iterations
```

Current caution:

- The compact tangent contraction is mathematically the same directional contraction,

```text
payload_bar dot d(payload)/d(parameter)
```

but it computes it by JVP rather than by VJP-to-state followed by raw-block transpose.
- Because that changes AD execution order, derivative values must be rechecked against the benchmark-good references before this path is trusted for optimization.
- Do not modify benchmark scripts as a workaround. If further failures occur, fix the internal compact path or explicitly fall back to benchmark-table behavior.

## Current State: 2026-07-31 Compact Optimization Pullback Running

Latest status:

- `examples/optimization/optimize_geometry_qi_max_er_initial_root.py` now runs past the previous OOM and tracer/static-metadata failures.
- The optimization path enters the intended compact tangent-contraction route:

```text
[optimization] initial-Er root geometry payload pullback: compact_payload_tangent_contract=True
[optimization] initial-Er root geometry payload pullback: raw_block_param_bar_all_finite=True
```

Observed initial run excerpt:

```text
max_mode=1
parameter_count=8
parameters=[
  RBC:0:1, RBC:1:-1, RBC:1:0, RBC:1:1,
  ZBS:0:1, ZBS:1:-1, ZBS:1:0, ZBS:1:1
]

initial elapsed_s=303.518
geometry:boozer_qi_objective        2.0475085800096847e-02
geometry:boozer_maxj_objective      5.5724338478421238e-02
geometry:vmec_mirror_ratio          2.0637638636935299e-01
geometry:vmec_aspect_ratio          9.9971624846684968e+00
geometry:vmec_iota_mean            -5.4821690487569552e-01
transport:softmax_Er                2.1889949774620024e-01
residual_norm                       2.979072e+01
jacobian_shape                      (6, 8)
```

Least-squares iterations started and printed:

```text
[NEOPAX least_squares] eval=1 cost=4.437434e+02 residual_norm=2.979072e+01
[NEOPAX least_squares] eval=2 cost=6.942995e+03 residual_norm=1.178388e+02
[NEOPAX least_squares] eval=3 cost=4.449269e+02 residual_norm=2.983042e+01
```

Static/JVP transform fixes applied in `NEOPAX/_geometry_autodiff.py`:

- Precompute geometry Boozer sample radii outside the JVP'd geometry builder.
- Precompute Boozer surface indices/sample rho outside the JVP'd geometry builder.
- Precompute Boozer constants/grids outside the JVP'd geometry builder.
- Precompute Boozer `(m,n)=(0,0)` and `(1,0)` mode indices outside the JVP'd geometry builder.
- Keep fallback behavior for normal/non-compact callers.

Why this was needed:

- The compact optimization route computes:

```text
VMEC parameter tangent -> state tangent -> JVP(payload) -> bar dot tangent
```

- Therefore `_build_neopax_geometry_from_state(...)` runs under JAX transforms.
- Static setup that was harmless in concrete primal/benchmark staging becomes illegal under JVP if it calls Python `int`, `float`, `bool`, NumPy conversion, or dynamic `jnp.unique` on traced values.

Immediate validation still required:

1. Let the optimization script finish or stop at the planned `NFEV`.
2. Re-run the root-only benchmark smoke:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --reverse-parameter-mode profiles_plus_realtime_geometry \
  --reverse-geometry-parameter RBC:1:0 \
  --objective all \
  --initial-Er-root-ad jax_selected_root \
  --initial-Er-root-only-optimization-smoke
```

3. Re-run the 2-step internal realtime-geometry reverse smoke:

```bash
python ./examples/optimization/transport_realtime_geometry_reverse_smoke.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --reverse-geometry-parameter RBC:1:0 \
  --objective all \
  --accepted-step-limit 2 \
  --reverse-segment-length 1 \
  --initial-Er-root-ad jax_selected_root \
  --ntx-exact-derivative-mode direct \
  --ntx-exact-derivative-field-pullback-mode generic_jvp \
  --radau-jacobian-reuse-mode legacy \
  --reverse-stage-adjoint-solve-mode bicgstab \
  --reverse-rhs-transpose-mode explicit_ntx_interpolated \
  --reverse-step-bwd-mode reduced_cotangent \
  --hide-solver-iterations
```

4. Compare derivative rows against saved benchmark-good values before trusting optimization results.

Open performance note:

- The first evaluation still has large compile/setup cost.
- Subsequent evaluations are much faster for Boozer/J-invariant pieces, but the VMEC implicit solve and transport/root compact payload tangent contraction still dominate.
- Future work should consider caching/static staging around repeated optimization evaluations without changing the validated AD graph.

## Shared Root-Only Payload Smoke: Validated Single-Harmonic Benchmark

Command:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --reverse-parameter-mode profiles_plus_realtime_geometry \
  --reverse-geometry-parameter RBC:1:0 \
  --objective all \
  --initial-Er-root-ad jax_selected_root \
  --initial-Er-root-only-optimization-smoke \
  --initial-Er-root-shared-payload-compare-smoke
```

Result:

- Completed without OOM.
- Runtime build: `309.876 s`.
- Shared root-only smoke elapsed time: `485.368 s`.
- Output written to
  `outputs/autodiff_transport_lagged_ntx/reverse_ad/transport_reverse_ad_only_initial_er_root_shared_payload_smoke.json`.
- The run used the shared/fused root-only payload path and produced `12` residual rows over `5`
  parameters: `n0`, `T0`, `density_shape_power`, `temperature_shape_power`, and
  `vmec:RBC:1:0`.

Key `d/dvmec:RBC:1:0` rows:

| Row | Value | d/dRBC:1:0 |
| --- | ---: | ---: |
| `transport:softmax_Er` | `2.0479476664720302e+01` | `-5.1293330714789889e+01` |
| `transport:smooth_root_proxy` | `8.0591612259350128e-11` | `2.2505242252037452e-09` |
| `transport:Er_transition_left` | `1.7657228878480385e+01` | `-2.0079770095990593e+01` |
| `transport:Er_transition_right` | `1.8321801837429653e+01` | `-2.2278454387928971e+01` |
| `transport:Er2_volume_average` | `2.5947838715347029e+02` | `-1.8476397875085149e+02` |
| `transport:Er_volume_average` | `-3.5568787373760746e+00` | `-2.0622727790532899e+01` |
| `geometry:boozer_qi_objective` | `1.1913614798130393e-03` | `1.7261150817519652e-02` |
| `geometry:boozer_maxj_objective` | `5.1725507614818367e-02` | `-8.0802156302056483e-01` |
| `geometry:vmec_aspect_ratio` | `1.0015330918957178e+01` | `-5.4006784187006147e+00` |
| `geometry:vmec_iota_mean` | `-5.9365259966101458e-01` | `2.4405140609263865e-01` |
| `geometry:vmec_magnetic_well` | `-2.7476128749679612e-02` | `-1.1090116065531674e-02` |
| `geometry:vmec_mirror_ratio` | `2.1100247521308457e-01` | `-5.9979060848576748e-01` |

Interpretation:

- This validates the root-only shared-payload benchmark path for the single-harmonic
  `RBC:1:0` case.
- Geometry rows have zero profile columns, as expected.
- Transport/root rows include both profile columns and the VMEC harmonic column, as expected.
- This does not by itself validate the full Radau time-evolution shared path; that remains covered by
  the separate full-transport shared payload smoke.

### Corrected Current QI Reference

The older shared-payload table above predates the current Boozer/J-invariant QI
formulation. The corrected standalone frozen-linearized QI reference for the
same high-resolution VMEC input is:

```bash
python ./examples/benchmarks/compare_geometry_qi_frozen_linearized_fd.py \
  --vmec-input ./examples/inputs/input.QI_nfp2_newNT_opt_hires_true \
  --parameter RBC:1:0 \
  --objective boozer_qi_objective \
  --multigrid \
  --forward-linear-solve-mode raw_block \
  --forward-linear-maxiter 300 \
  --adjoint-maxiter 300
```

Observed:

```text
baseline value       = 3.2109136309506803e-03
frozen_linearized_fd = 8.9996041589633161e-02
forward_jvp          = 8.9989229074526111e-02
rel_err_linfd_vs_jvp = 7.570367e-05
```

This agrees with the current shared-payload geometry row:

```text
geometry:boozer_qi_objective value          = 3.2109136309506734e-03
d geometry:boozer_qi_objective / d RBC:1:0 = 8.9989229070800647e-02
```

Saved derivative references:

| Quantity | d/dRBC:1:0 |
| --- | ---: |
| `boozer_qi_objective` frozen-linearized FD | `8.9996041589633161e-02` |
| `boozer_qi_objective` forward JVP / AD target | `8.9989229074526111e-02` |
| `boozer_qi_objective` shared-payload reverse row | `8.9989229070800647e-02` |

Conclusion: for the current QI implementation, the saved FD reference is
`8.9996041589633161e-02`; the implicit/JVP AD target is
`8.9989229074526111e-02`, with frozen-linearized FD agreement at about
`7.6e-05` relative error.

### Corrected Current maxJ Reference

The corrected standalone frozen-linearized maxJ reference for the same
high-resolution VMEC input is:

```bash
python ./examples/benchmarks/compare_geometry_qi_frozen_linearized_fd.py \
  --vmec-input ./examples/inputs/input.QI_nfp2_newNT_opt_hires_true \
  --parameter RBC:1:0 \
  --objective boozer_maxj_objective \
  --multigrid \
  --forward-linear-solve-mode raw_block \
  --forward-linear-maxiter 300 \
  --adjoint-maxiter 300
```

Observed:

```text
baseline value       = 1.3389843913753766e-01
frozen_linearized_fd = -1.1591364898936340e+00
forward_jvp          = -1.1593167988198267e+00
rel_err_linfd_vs_jvp = 1.555303e-04
```

This agrees with the current shared-payload geometry row:

```text
geometry:boozer_maxj_objective value          = 1.3389843913753852e-01
d geometry:boozer_maxj_objective / d RBC:1:0 = -1.1593167987379616e+00
```

Saved derivative references:

| Quantity | d/dRBC:1:0 |
| --- | ---: |
| `boozer_maxj_objective` frozen-linearized FD | `-1.1591364898936340e+00` |
| `boozer_maxj_objective` forward JVP / AD target | `-1.1593167988198267e+00` |
| `boozer_maxj_objective` shared-payload reverse row | `-1.1593167987379616e+00` |

Conclusion: for the current maxJ implementation, the saved FD reference is
`-1.1591364898936340e+00`; the implicit/JVP AD target is
`-1.1593167988198267e+00`, with frozen-linearized FD agreement at about
`1.6e-04` relative error. The shared-payload reverse row matches the JVP target
to about `7e-11` relative error.
