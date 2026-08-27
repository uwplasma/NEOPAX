# VMEX-like staged optimization: geometry + initial ambipolar-Er root

## Scope

Implement an opt-in staged path for geometry optimization with initial-root
ambipolarity objectives: max-Er, Er-left, Er-right, and bootstrap current,
alongside existing geometry objectives.  Full time-evolved transport is a
separate later phase.

## Hard constraints

- The benchmark path and its functions are not edited.
- Root solving, NTX-Lij flux/model functions, forward solvers, and VMEC custom
  rules are not copied or changed.
- Geometry is solved once per evaluation and shared by all objective rows.
- The stage never forms one fused graph for the whole optimization.
- TOML/CLI continue selecting physics/solver modes.  The stage is an explicit
  optimizer lifecycle option only.

## Architecture

The optimization-only module contains two copied orchestration operators,
mirroring VMEX's persistent optimizer residual/Jacobian callables:

```text
GeometryInitialRootOptimizationStage
  root_to_payload_operator
    current profile/state + current geometry/support payload
      -> existing selected-root and compact reverse rules
      -> values, profile rows, payload cotangent rows

  payload_to_vmec_operator
    current payload cotangent rows + current shared raw VMEC solve
      -> existing payload pullback and VMEC raw-block transpose
      -> geometry Jacobian rows
```

The copied code is orchestration only.  The operators call the exact existing
root, NTX, payload, and VMEC routines.  They are built once per fixed
optimization stage and receive changing DoFs/state/payload vectors as dynamic
inputs.  This is the NEOPAX analogue of VMEX's persistent `rows_jit` and
`jac_jit`, while retaining two bounded operators rather than one giant graph.

## Mode and model selection

`reverse_stage_mode="off"` is the default.  `"vmex_like"` is explicit and
only becomes available after both operators are installed and parity tested.
The initial implementation supports the already benchmarked
`ntx_exact_lij_runtime` path only.  A TOML-selected model without a matching
implemented staged adapter fails clearly; it never falls back to another
model.

## Implementation order

1. Add the private stage API and fixed-layout validation.
2. Copy the existing initial-root-to-payload orchestration, preserving every
   call to the benchmarked root/NTX rules.
3. Copy the existing payload-to-VMEC orchestration, preserving the raw-block
   transpose call.
4. Build and cache the two operator callables once per optimization stage.
5. Wire only `reverse_stage_mode="vmex_like"` through the geometry +
   initial-root optimizer.  Benchmark functions remain untouched.
6. Compare staged/benchmark values and Jacobian rows at fixed inputs.
7. Run one large-machine four-objective acceptance test for cold compilation,
   warm timing, and RSS slope.

## Acceptance

The staged four-objective result must match the unchanged benchmark path
within established tolerances, retain one shared geometry solve, and show no
sustained post-warmup RSS increase.  If it changes benchmark numbers or does
not fix the RSS slope, it is not enabled by default.
