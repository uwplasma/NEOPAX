# Optimization Path Plan

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
