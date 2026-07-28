# Reverse-AD Optimization Lane Plan

## Goal

Create a production-quality NEOPAX reverse-AD differentiation lane for optimization while keeping
the existing benchmarks as validation clients.

The lane should support:

- profile parameters,
- realtime VMEC geometry parameters,
- all active VMEC boundary harmonics when geometry is enabled,
- transport objectives,
- VMEC/Boozer/QI/MaxJ geometry objectives,
- scalar weighted losses for optimization,
- full objective tables/Jacobians for diagnostics.

The guiding rule is:

```text
one reusable reverse-AD lane, many benchmark gates
```

Benchmarks should not remain the source of production math.

## Non-Contamination Principle

This work should not change benchmark defaults or forward-solver behavior.

The current benchmarks should become clients of the internal lane gradually:

```text
benchmark parses CLI
benchmark builds options
benchmark calls NEOPAX reverse-AD API
benchmark prints diagnostics / FD comparisons
```

Benchmark-only logic must stay benchmark-only:

- finite-difference perturbation logic,
- split payload FD diagnostics,
- component pullback diagnostics,
- branch split diagnostics,
- support-bar l2/finiteness tree summaries,
- timing experiments,
- one-off OOM probes,
- old failed solver experiments.

The production lane should contain only the validated differentiable path.

## Geometry Parameter Requirement

When geometry is active, the default optimization parameterization should account for all VMEC
boundary harmonics present in the VMEC object, not just a single hardcoded `RBC:1:0`.

The current benchmark already has a prototype of this behavior:

```text
--reverse-geometry-parameter all
--reverse-geometry-families RBC,ZBS
--reverse-geometry-include-zero-harmonics
```

The reusable API should promote this into a first-class parameter discovery helper:

```text
discover_vmec_boundary_parameter_specs(context, families=("RBC", "ZBS"), nonzero_only=True)
```

Expected behavior:

- read mode numbers from `context.static.modes.m` and `context.static.modes.n`,
- read boundary coefficients from `context.boundary.rbc` and `context.boundary.zbs`,
- include every finite nonzero coefficient by default,
- optionally include zero coefficients when explicitly requested,
- return stable labels such as `RBC:1:0`, `ZBS:1:0`, etc.,
- use the same ordering for values, gradients, reports, and optimizer vectors.

The optimizer API should allow:

```text
geometry_parameters = "all"
geometry_families = ("RBC", "ZBS")
geometry_nonzero_only = true
```

and should also allow an explicit list for small tests:

```text
geometry_parameters = ["RBC:1:0", "ZBS:1:0"]
```

## Proposed API

Add an internal module, for example:

```text
NEOPAX/_reverse_ad_optimization.py
```

Initial public/internal entrypoints:

```python
value_and_grad_reverse_ad(config, parameter_spec, objective_spec, *, options)
objective_table_and_jacobian_reverse_ad(config, parameter_specs, objective_specs, *, options)
```

The least-squares residual/Jacobian path should be the first production target, because it matches
the validated all-objective reverse-table machinery and the VMEX optimization-script style:

```python
residuals, jacobian = residuals_and_jacobian_reverse_ad(
    config,
    parameter_spec={
        "profiles": ("n0", "T0", "density_shape_power", "temperature_shape_power"),
        "geometry": "all",
        "geometry_families": ("RBC", "ZBS"),
    },
    objective_spec={
        "transport": {"softmax_Er": 1.0, "total_pressure_volume_average": 1.0},
        "geometry": {"boozer_qi_objective": 1.0},
    },
    options={
        "vmec_pullback_mode": "raw_block_transpose",
        "initial_er_root_ad": "jax_selected_root",
    },
)
```

Each user-facing term has the VMEX-like form:

```python
(objective, target, weight)
```

and maps to:

```text
residual_i = sqrt(weight_i) * (objective_i - target_i)
J_i        = sqrt(weight_i) * d objective_i / d parameters
```

Scalar weighted loss is a convenience derived from this:

```text
loss = 0.5 * residuals @ residuals
grad = jacobian.T @ residuals
```

The raw objective-table path should be retained for diagnostics and multi-objective algorithms:

```python
values, jacobian, labels = objective_table_and_jacobian_reverse_ad(...)
```

## Core Pieces To Extract

Extract only validated reusable pieces from benchmark scripts.

Good candidates:

- profile parameter packing/unpacking,
- VMEC harmonic parsing/formatting,
- all-harmonic VMEC parameter discovery,
- baseline geometry-delta vector construction,
- realtime geometry support payload pullback orchestration,
- compact initial-Er root support pullback,
- raw-block VMEC pullback selection,
- objective label/order management.

Existing validated core helpers to reuse or wrap:

```text
NEOPAX/_geometry_autodiff.py
  geometry_payload_pullback_from_param_vector_raw_block_transpose(...)
  geometry_full_ad_objective_table_pullback_from_param_vector(...)
  build_runtime_context_for_geometry_param(...)
```

Benchmark-local helpers that should be promoted carefully:

```text
examples/benchmarks/benchmark_transport_reverse_ad_only.py
  _all_geometry_param_specs_from_context(...)
  _geometry_param_specs_from_args(...)
  _compact_initial_er_ntx_support_pullback_leaves(...)
```

Do not promote benchmark reporting or probe scaffolding.

## Objective Blocks

Use objective blocks rather than one monolithic objective function.

Transport block:

```text
profile/geometry parameters
  -> runtime + initial state
  -> accepted replay / reverse payload
  -> transport objective values
  -> profile bars + geometry payload bars
  -> profile gradients + VMEC harmonic gradients
```

Geometry block:

```text
VMEC harmonic parameters
  -> VMEX implicit equilibrium
  -> Boozer / QI / MaxJ / VMEC geometry objectives
  -> raw_block_transpose VMEC pullback
  -> VMEC harmonic gradients
```

Combined scalar loss:

```text
weighted transport objectives
+ weighted geometry objectives
+ optional regularization
```

For optimization scripts, the interface should feel like VMEX examples:

```python
terms = [
    (transport.softmax_Er, 20.0, 1.0),
    (transport.total_pressure_volume_average, 35.0, 0.1),
    (geometry.boozer_qi_objective, 0.0, QI_WEIGHT),
    (geometry.aspect_ratio, ASPECT_TARGET, ASPECT_WEIGHT),
    (geometry.iota_shortfall, 0.0, IOTA_WEIGHT),
]
```

Internally, terms must be grouped by backend:

```text
transport terms -> one grouped transport reverse table
geometry terms  -> one grouped geometry/QI reverse table
regularization  -> analytic rows
```

Do not run one reverse pass per term.

## Defaults

Initial safe defaults:

```text
reverse_ad enabled only by explicit API/TOML option
vmec_pullback_mode = raw_block_transpose
geometry_parameters = explicit list unless user requests all
initial_er_root_ad = off
```

After profile root-on FD/reverse validation is explicitly recorded:

```text
initial_er_root_ad = jax_selected_root
```

can become the recommended optimization mode.

## TOML Shape

Suggested TOML configuration:

```toml
[autodiff]
mode = "reverse_ad"
vmec_pullback_mode = "raw_block_transpose"
initial_er_root_ad = "off"

[optimization.parameters]
profiles = ["n0", "T0", "density_shape_power", "temperature_shape_power"]
geometry = "all"
geometry_families = ["RBC", "ZBS"]
geometry_nonzero_only = true

[optimization.terms]
transport = [
  {objective = "softmax_Er", target = 20.0, weight = 1.0},
  {objective = "total_pressure_volume_average", target = 35.0, weight = 0.1},
]
geometry = [
  {objective = "boozer_qi_objective", target = 0.0, weight = 1.0},
]
```

This should be parsed by a thin optimization/AD wrapper, not by the core forward solver.

## Validation Gates

Before using this as an optimization lane, keep these gates:

1. Geometry-only VMEC/Boozer/QI FD vs reverse with `raw_block_transpose`.
2. Transport realtime geometry FD vs reverse, `RBC:1:0`, root off.
3. Transport realtime geometry FD vs reverse, `RBC:1:0`, root on.
4. Profile-only reverse baseline, root off.
5. Profile FD vs reverse, root on.
6. All-harmonic smoke test with a small objective set.
7. Forward solver smoke test to confirm no forward-path contamination.

Known current status:

- realtime-geometry `RBC:1:0` root-on reverse matches FD-on for main transport objectives,
- compact initial-Er root pullback is validated for all objectives in the reverse benchmark,
- profile-only root-off baseline is validated,
- full profile root-on FD/reverse table still needs to be explicitly recorded before making root-on
  the default optimization mode.

## Efficiency Phase

Once correctness is stable, time/memory work should target:

1. primal/reverse schedule fusion,
2. direct contraction of support bars to selected VMEC harmonics,
3. avoiding materialization of full support cotangent trees when only selected harmonics are needed,
4. scalar weighted-loss wrapper derived from the residual/Jacobian path,
5. optional objective blocking only as a controlled memory/time tradeoff, not as a correctness
   workaround.

Do not introduce Python loops over objectives as the production strategy.

## Implementation Order

1. Add parameter-spec dataclasses/helpers for profiles and VMEC harmonics. **Done:**
   `NEOPAX/_reverse_ad_parameters.py` now defines typed profile and VMEC boundary specs,
   canonical labels, parsing for `RBC:1:0` and `vmec:RBC:1:0`, mixed spec splitting,
   a stable profile-then-VMEC `ReverseADParameterSet`, profile-value packing, and VMEC tuple
   conversion for existing geometry helpers.
2. Extract all-harmonic VMEC discovery into NEOPAX core. **Done:**
   `NEOPAX/_reverse_ad_parameters.py` now provides
   `discover_vmec_boundary_parameter_specs(...)` and
   `normalize_vmec_boundary_families(...)`. The realtime reverse benchmark's
   `--reverse-geometry-parameter all` path delegates to this core helper while
   preserving its existing tuple interface.
3. Extract compact initial-Er root pullback into a reusable internal helper. **Done:**
   `NEOPAX/_reverse_ad_initial_er.py` now owns
   `compact_initial_er_ntx_support_pullback_leaves(...)` and
   `find_ntx_exact_support_model(...)`. The realtime reverse benchmark keeps
   local wrapper names and delegates to the core helper, preserving the existing
   benchmark call path and output behavior. The core helper also validates that
   `residual_bars` has shape `(objective_count, radial_count)` and that the
   radial dimension matches the selected `er_profile`.
4. Add objective-term dataclasses and a least-squares residual/Jacobian API shell. **Done:**
   `NEOPAX/_reverse_ad_optimization.py` now defines `ObjectiveRef`, `LeastSquaresTerm`,
   `ObjectiveTableResult`, `LeastSquaresResult`, VMEX-like term normalization, grouping by
   backend family, namespace helpers such as `transport.softmax_Er` and
   `geometry.boozer_qi_objective`, residual/Jacobian assembly with table-shape guards, and scalar
   loss/gradient conversion. It accepts backend table callables but does not yet move or call the
   heavy transport reverse loop.
5. Add grouped transport reverse-table backend for transport terms using the validated benchmark
   path. **In progress:** `_reverse_ad_optimization.py` now provides
   `transport_reverse_report_to_objective_table_result(...)` and
   `transport_reverse_report_backend(...)`, which adapt the already-validated grouped reverse
   benchmark report into an `ObjectiveTableResult` without changing the reverse math. It also now
   provides `transport_reverse_report_builder_backend(...)`, which calls a grouped report builder
   and adapts the result in one backend call. `NEOPAX/_reverse_ad_transport.py` now owns
   `build_realtime_geometry_transport_reverse_report(...)`,
   `grouped_transport_reverse_report_builder(...)`, and shared transport objective-name validation.
   The realtime-geometry reverse benchmark still keeps the validated
   `_run_realtime_geometry_support_segment_probe(...)` function in place, but its
   `_make_realtime_geometry_support_segment_report_builder(...)` now delegates through
   `build_realtime_geometry_transport_reverse_report(...)`. This gives the least-squares API a
   correct transport table bridge for profile + VMEC columns. Step 2 has started:
   `transport_reverse_table_report_entries(...)` now lives in `_reverse_ad_transport.py` and owns
   the reusable non-printing objective/profile-gradient/geometry-gradient table report assembly.
   The remaining work is to move the heavy grouped reverse execution itself out of the benchmark
   module into production internals and then have the CLI call that internal helper.
   Update: the shared `transport_reverse_table_report_entries(...)` helper now accepts JAX device
   arrays directly and owns the `jax.device_get(...)`/NumPy conversion for objective values,
   profile gradients, and geometry gradients. The benchmark keeps only benchmark-specific host
   conversions for branch diagnostics and top-k reporting.
   Update: the realtime-geometry reverse benchmark now builds its optimization-facing grouped
   report bridge through `grouped_transport_reverse_report_builder(...)`, so the benchmark no
   longer open-codes the generic report-builder wrapper. The validated heavy runner is still kept
   in the benchmark file for now, as requested.
   Update: `NEOPAX/_reverse_ad_transport.py` now exposes
   `run_realtime_geometry_transport_reverse_table(...)` as the named internal execution seam for
   the grouped realtime-geometry transport reverse table. It currently delegates to the same
   benchmark-supplied validated grouped runner exactly once; no heavy math has moved yet.
   Update: step 2 has started with a stable request/context handoff:
   `RealtimeGeometryTransportReverseTableContext`,
   `RealtimeGeometryTransportReverseTableRequest`,
   `realtime_geometry_transport_reverse_table_context(...)`, and
   `realtime_geometry_transport_reverse_table_request(...)` now live in
   `_reverse_ad_transport.py`. The benchmark report-builder bridge passes this context into the
   grouped builder, so the next extraction can move execution code without threading benchmark
   arguments manually. This is still a no-math-change bridge and calls the same grouped runner once.
   Update: the generic, non-printing realtime-geometry transport reverse metadata assembly now
   lives in `realtime_geometry_transport_reverse_metadata_entries(...)`. The benchmark still owns
   branch/component diagnostics, printing, JSON writing, and the heavy JAX reverse execution. This
   helper only assembles the same report keys that were already host-side metadata.
   Update: branch/component geometry-gradient report assembly now lives in
   `realtime_geometry_transport_reverse_diagnostic_gradient_entries(...)`. It consumes the same
   already-materialized host arrays the benchmark previously used, so it does not add JAX work,
   device transfers, or reverse passes. The benchmark still owns the expensive reverse execution
   and user-facing printing.
   Update: `_reverse_ad_transport.py` now defines the JAX-native
   `RealtimeGeometryTransportReverseTableResult` plus
   `realtime_geometry_transport_reverse_table_result(...)`. The benchmark creates this object
   immediately after the grouped JAX objective/profile/geometry arrays are available and before
   any host-report conversion. This is the future optimization-facing table boundary.
   Update: `_reverse_ad_optimization.py` can now consume that JAX-native transport table directly
   via `transport_reverse_table_result_to_objective_table_result(...)` and
   `transport_reverse_table_backend(...)`. This avoids routing optimization through benchmark
   report dictionaries/NumPy conversion when a table result is available.
   Update: the benchmark bridge now includes `"transport_reverse_table_result"` only on the
   internal `return_report=True` path. `transport_reverse_report_builder_backend(...)` prefers this
   JAX-native table when present and falls back to dictionary reports for older callers. The JSON
   and printed benchmark reports remain unchanged.
   Update: `_reverse_ad_optimization.py` now also exposes
   `transport_reverse_table_result_builder_backend(...)`, which accepts a builder returning the
   JAX-native `RealtimeGeometryTransportReverseTableResult` directly. This is the clean production
   optimization seam: it avoids report dictionaries/NumPy conversion entirely while keeping the
   benchmark-compatible report-builder fallback available during migration.
   Update: `_reverse_ad_transport.py` now provides
   `grouped_transport_reverse_table_result_builder(...)` and
   `build_realtime_geometry_transport_reverse_table_result(...)`. The realtime-geometry reverse
   benchmark exposes `_make_realtime_geometry_support_segment_table_result_builder(...)`, a direct
   table-result builder around the same validated grouped runner. This lets optimization tests use
   `transport_reverse_table_result_builder_backend(...)` without depending on the report-builder
   fallback.
   Update: `benchmark_transport_reverse_ad_only.py` now has an explicit
   `--optimization-api-smoke` path for `profiles_plus_realtime_geometry` + `reverse_payload`. It
   builds a `ReverseADParameterSet`, wraps the direct table-result builder with
   `transport_reverse_table_result_builder_backend(...)`, and evaluates
   `residuals_and_jacobian_reverse_ad(...)`. This smoke exits after writing a separate
   `optimization_api_smoke` report, so the normal benchmark reports and math remain unchanged.
   Update: the backend construction/timing/blocking portion of that smoke now lives in
   `_reverse_ad_optimization.py` as `evaluate_transport_reverse_table_least_squares(...)`.
   The benchmark still chooses benchmark-specific objectives, prints, and writes JSON, but it no
   longer owns the direct table-backend evaluation mechanics.
   Update: `_reverse_ad_transport.py` now exposes
   `transport_realtime_geometry_reverse_table(...)`, the explicit internal table API boundary for
   optimization callers. It consumes a `RealtimeGeometryTransportReverseTableRequest` (or the same
   fields directly) and returns the JAX-native table result. During migration it can still use the
   benchmark-supplied grouped runner, but callers no longer need to know about report dictionaries.
   Update: `_reverse_ad_optimization.py` now exposes
   `evaluate_transport_realtime_geometry_least_squares(...)`, which calls
   `transport_realtime_geometry_reverse_table(...)` directly and times the full table evaluation
   plus residual/Jacobian assembly. The benchmark `--optimization-api-smoke` path now uses this
   canonical request-based helper. The helper also checks that request objective names match the
   transport least-squares terms, so optimization scripts cannot accidentally evaluate a different
   objective set than the table request.
   Update: `_reverse_ad_optimization.py` now also exposes
   `build_transport_realtime_geometry_least_squares_runner(...)`. This factory builds the
   `RealtimeGeometryTransportReverseTableRequest` internally and returns a callable
   `terms -> LeastSquaresEvaluation`. The benchmark `--optimization-api-smoke` now uses this
   runner factory, so request construction and least-squares wiring are no longer benchmark-owned.
   Update: `_reverse_ad_transport.py` now exposes
   `realtime_geometry_transport_reverse_grouped_inputs(...)`, which builds the
   `RealtimeGeometryTransportReverseTableContext` and grouped report runner from a supplied
   segmented executor. The benchmark now only supplies the temporary executor callback around
   `_run_realtime_geometry_support_segment_probe(...)`; objective='all' grouping and context
   construction are internal.
   Update: `_reverse_ad_transport.py` now also exposes
   `realtime_geometry_transport_reverse_support_segment_executor(...)`, which wraps a supplied
   segmented probe callback with the grouped-executor calling convention. The benchmark now passes
   `_run_realtime_geometry_support_segment_probe` as the callback; enforcing `return_report=True`
   and threading runtime/config/profile inputs are internal.
   Update: `_reverse_ad_transport.py` now exposes
   `run_realtime_geometry_support_segment_reverse_table_core(...)`, a non-printing internal core
   boundary around the supplied segmented probe callback. The heavy probe implementation is still
   in the benchmark, but grouped optimization execution now goes through an internal core that
   enforces `return_report=True`, suppresses probe output, threads table context, and validates the
   returned JAX-native table result.
   Update: step 2 has started moving validated heavy pieces out of the benchmark:
   `_reverse_ad_transport.py` now owns `realtime_geometry_payload_pullback_result(...)`, the
   raw-block-transpose VMEC payload pullback orchestration from transport support cotangents to
   geometry harmonics. The benchmark now delegates this piece to internals and keeps only
   diagnostics/reporting around the returned matrices.
   Update: `_reverse_ad_transport.py` now also owns
   `realtime_geometry_transport_reverse_table_from_payload_cotangents(...)`, which assembles the
   JAX-native table result from objective values, profile gradients, support payload cotangents,
   and the raw-block VMEC payload pullback. The benchmark computes the segmented transport support
   cotangents, then delegates non-printing table assembly to this internal helper.
   Update: `_reverse_ad_transport.py` now owns the grouped-runner contract through
   `realtime_geometry_transport_reverse_grouped_runner(...)`: force the executor to run the
   all-objective internal report path and require that the report contains a JAX-native
   `transport_reverse_table_result`. The benchmark still supplies the actual segmented support
   executor, but no longer owns this grouped-runner contract.
   Update: `_reverse_ad_transport.py` now owns the stable setup immediately before the segmented
   realtime-geometry support reverse sweep through
   `prepare_realtime_geometry_support_segment_core_setup(...)`. This includes support payload
   selection, profile-value slicing, reverse static setup construction, backend metadata, and early
   geometry diagnostics capture. The benchmark still owns the actual segmented support pullback
   implementation.
   Update: `_reverse_ad_transport.py` now also owns the all-objective grouped support-cotangent
   orchestration through `realtime_geometry_support_cotangents_from_parameter_vector(...)` and
   `RealtimeGeometrySupportCotangentResult`. It calls the existing benchmark-supplied grouped
   reverse callback exactly once, performs the same device synchronization, and returns the stable
   internal result shape consumed by the payload-to-VMEC table assembly.
   Update: the active all-objective segmented support reverse kernel now routes through
   `_reverse_ad_transport.py` via
   `realtime_geometry_reverse_all_objectives_support_payload_bar_for_parameter_vector(...)` and a
   benchmark-supplied `RealtimeGeometrySupportReverseDependencies` callback bundle. The benchmark
   still owns the helper callbacks, but the old all-objective benchmark body has been removed; the
   benchmark now keeps only a thin dependency-wrapper for that path.
6. Move remaining low-risk dependency callbacks out of the benchmark only when they can be made
   production-safe without changing numerics.
7. Keep the single-objective support probe benchmark-owned unless optimization explicitly needs it.
   It is still valuable as a diagnostic path.
8. Add a non-benchmark smoke/regression for the internal transport reverse table API using fake or
   reduced objects first.
9. Run one full GPU validation with `--optimization-api-smoke` after the API seams are stable.
10. Add grouped geometry/QI reverse-table backend using `raw_block_transpose`.
11. Add combined residual/Jacobian assembly for transport + geometry + regularization terms.
12. Add scalar weighted-loss convenience wrapper as `0.5 * r @ r`.
13. Only after validation, expose TOML-driven optimization mode.
14. Add VMEX-style packed/scaled VMEC boundary parameterization for optimization scripts.
   Update: `_reverse_ad_parameters.py` now exposes `vmex_boundary_parameterization(...)`,
   `vmex_packed_boundary_parameter_specs(...)`, VMEX independent-mode validation, and ESS-style
   boundary scales. This is parameter-layer only: explicit benchmark specs such as `RBC:1:0` remain
   unchanged, while optimization callers can opt into VMEX-like packed DOFs that exclude
   `m=0,n<=0` and include initially-zero harmonics up to `max_mode`.
15. Add an ambipolar-root-only optimization lane for Er-related objectives without transport time
    evolution.
   Update: `_reverse_ad_optimization.py` now exposes
   `build_initial_er_root_only_least_squares_runner(...)`,
   `evaluate_initial_er_root_only_least_squares(...)`, and
   `initial_er_root_only_reverse_table(...)`. This path evaluates only
   `softmax_Er`, `smooth_root_proxy`, `Er2_volume_average`, and `Er_volume_average` from a
   caller-supplied TOML/runtime-backed selected-root state builder. It does not run the Radau
   transport time map.
16. Add explicit profile-DOF selection for optimization parameter sets.
   Update: `_reverse_ad_parameters.py` now exposes `reverse_ad_optimization_parameter_set(...)`
   with `include_profiles=True/False`. The realtime-geometry optimization smoke also accepts
   `--optimization-api-profile-dofs include|exclude`; `exclude` gives geometry-only optimization
   while still evaluating transport objectives.

## Key Invariants

- Geometry-active optimization must account for all selected VMEC harmonics.
- `raw_block_transpose` remains the default VMEC harmonic pullback for validated reverse geometry.
- User-facing optimization terms can look like VMEX least-squares terms, but internally they must be
  grouped into reverse-table blocks.
- Benchmarks remain validation clients, not production math owners.
- Forward solver behavior is unchanged unless an explicit reverse-AD/optimization path is selected.
- Root branch selection is not differentiated; only the selected-root residual path is differentiated.
- VMEX-packed optimization excludes fixed/non-independent boundary modes (`m=0,n<=0`) by
  construction; explicit diagnostic/benchmark parameter specs remain permissive.
- The initial-Er root-only optimization lane is not a zero-step transport solve. It is a separate
  Er-objective table over the selected initial ambipolar root, with TOML-derived runtime/state
  supplied by the caller.
- Excluding profile DOFs changes only the optimization parameter vector columns. It does not freeze
  profiles inside the physics evaluation; profiles still come from the TOML/baseline state unless
  another selected parameter changes them.
