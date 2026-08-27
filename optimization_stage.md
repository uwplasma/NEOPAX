# Optimization-stage plan: existing payload boundaries only

## Goal

Eliminate a sustained RSS increase across NEOPAX least-squares evaluations
without changing the established reverse-AD mathematics, numerical benchmark
results, forward solvers, initial-Er root solve, or NTX exact-Lij flux model.

The reference route is the existing benchmark route:

```text
geometry_active_initial_er_root_only_reverse_table
  -> realtime_geometry_transport_reverse_table_from_payload_cotangents
  -> realtime_geometry_payload_pullback_result
  -> geometry_payload_pullback_from_param_vector_raw_block_transpose
  -> VMEC raw-block transpose
```

## Fixed rules

1. Initial-Er root solving and `build_local_particle_flux_evaluator` are not
   changed.  No JIT, adapter, or alternate function is inserted below the
   existing payload-cotangent boundary.
2. The exact NTX-Lij flux equations and all forward transport solver code are
   not changed.
3. The benchmark's existing NTX-support/geometry pullback and VMEC raw-block
   transpose remain the only reverse boundaries that staging may touch.
4. The default benchmark path is literally the pre-stage call path.  It does
   not receive a stage argument, wrapper, lambda, or changed closure.
5. Geometry is solved exactly once per mixed geometry + initial-root
   evaluation and shared by all objective rows.
6. A stage may retain static configuration and compiled boundary executables;
   it may never retain a trial's VMEC state, payload cotangents, numerical VJP
   pullback, rooted state, or optimizer DoFs.

## Actual boundary to stage

The transport-specific heavy work is the NTX-support state pullback assembled
inside `geometry_payload_pullback_from_param_vector_raw_block_transpose`, in
particular the `ntx_support` branch of
`_state_bar_batch_from_payload_branch`.  That branch maps the existing
support-payload cotangent rows back to VMEC-state rows, then the existing VMEC
raw-block transpose maps those rows to geometry DoFs.

The initial-root calculation only supplies payload cotangents to this path. It
is not a staging boundary.

## Stage shape

`GeometryPayloadPullbackStage` is built once when the geometry + initial-root
least-squares problem is built.  It contains only:

- geometry context and fixed VMEC parameter layout;
- the existing `GeometryRawBlockStage` configuration;
- fixed grid/resolution/model configuration selected before the run;
- a bounded cache of existing payload-boundary executables keyed only by the
  structural payload layout and active-leaf signature.

It does not select a new physics model or alter TOML/CLI modes.  TOML and the
existing CLI/config preparation choose the model, derivative options, root
method, and solver settings exactly as before.

For each trial, the staged boundary receives dynamically:

```text
VMEC state / raw-block solve
payload cotangent rows
geometry DoF vector
```

It returns the same VMEC-state cotangent rows and applies the same existing
raw-block transpose.  A numerical VJP pullback is rebuilt for the current
primal values as required for correctness; only executable/static-function
lifecycle may be reused.

## Implementation order

1. Restore and lock the old root/flux/default benchmark path.  This is a
   prerequisite, not a compatibility branch.
2. Extract only the current `ntx_support` payload-state pullback setup into a
   private helper with explicit arguments for the current VMEC state and
   batched payload cotangent leaves.  Copy no physics formula and do not alter
   the existing custom-VJP rule; move the existing boundary as-is.
3. Add `GeometryPayloadPullbackStage` around that helper.  Its optional
   compiled boundary uses the same existing payload pullback rule and has only
   structural/static closure inputs.
4. Thread the optional stage only through
   `realtime_geometry_payload_pullback_result` and
   `geometry_payload_pullback_from_param_vector_raw_block_transpose`.
   No root/flux function signature changes.
5. Make optimization explicitly opt in to the stage.  Benchmark scripts keep
   the pre-stage route by default.  Do not auto-enable a stage from objective
   type.
6. Verify staged versus unstaged objective values, payload gradients, and
   geometry Jacobian rows for max-Er, Er-left, Er-right, and bootstrap current.
7. Only after equality checks pass, use one large-machine repeated-evaluation
   run to check cold compile time, warm time, and RSS slope.

## TOML/CLI behavior

The stage is a lifecycle setting, not a physics setting.  It must be enabled
explicitly by the optimization caller (eventually an optimization CLI option)
and must use the already resolved TOML/CLI configuration.  The initial scope
requires the existing `ntx_exact_lij_runtime` payload route.  If a selected
model has no established equivalent payload reverse boundary, stage creation
must fail clearly rather than switch models or equations.

## Full transport later

Full Radau transport is out of scope until the geometry + initial-root stage
is correct.  Its adaptive accepted-step count is represented by masks inside a
fixed `max_total_steps` scan, so it can later use a stage with the same
principle: fixed structural configuration, dynamic numerical state/masks, and
only its already benchmarked custom-VJP/payload boundaries.

## Acceptance criteria

- no root, flux, forward-solver, or benchmark-default-path change;
- staged/unstaged residuals and Jacobian rows agree within existing benchmark
  tolerances;
- one shared geometry solve per transport evaluation;
- no sustained post-warmup RSS growth in the four-objective acceptance run;
- cold compilation and warm execution are reported separately.
