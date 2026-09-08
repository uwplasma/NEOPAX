# Database reverse-AD status

## Scope

This note records the current `ntx_scan_runtime` / black-box database
reverse-AD implementation.  It does not change or describe the established
Lij (`ntx_exact_lagged_runtime`) reverse path.

## Intended reverse boundary

```text
VMEC geometry
  -> one recorded NTX scan
  -> fixed D11/D13/D33 database tables
  -> initial root, forward transport, terminal objectives, reverse segments
  -> accumulated table cotangents
  -> one recorded NTX scan transpose
  -> VMEC cotangent
```

Database transport segments retain only fixed tables.  They do not retain the
scan owner, scan primal, prepared scan systems, or a callable database builder.

## Current implementation

- The database direct-RHS support transpose returns table bars only.
- The database segment reverse uses the same bounded Radau replay and
  reduced-cotangent architecture as the Lij lane, with a fixed-table support
  leaf.
- The retired database-only raw transport-metric geometry VJP and deferred
  post-segment geometry sweep have been removed from the active reverse path.
- The database initial-Er root now also produces table bars only.  It does not
  run a separate local-radius raw geometry pullback.
- Table bars from all reverse contributions are folded through the retained
  scan once, after the complete reverse sweep.

## Diagnosed failure and correction

The old path produced nonfinite `r_grid_half` cotangents in the deferred raw
transport-metric geometry VJP (`fixed_flux_equation`).  Segment diagnostics
showed that all stage-adjoint and fixed-table cotangent values were finite;
the nonfinite value arose only in that retired geometry sweep.

The first one-step run after removing that sweep passed all segment finite
diagnostics and reached the initial-Er root boundary.  It then stopped with:

```text
Initial-Er root support pullback leaf shape mismatch at leaf 19:
got (), expected (10,)
```

Cause: the newly zeroed root geometry tree did not have the leading objective
axis.  The current unvalidated correction broadcasts that zero tree to
`(objective_count, ...)` before it is combined with database table bars.
This is a shape correction only; it does not restore a geometry VJP.

## Validation completed

```text
database_direct_rhs_support_stops_at_fixed_table_boundary                 PASS
database_initial_root_support_batches_charge_weighted_particle_bars       PASS
recorded_ntx_database_bar_groups_share_one_batched_scan_pullback          PASS
```

The three tests passed together on CPU (`3 passed, 147 deselected`).  Python
syntax checks passed after the subsequent objective-axis broadcast correction.

## Required next validation

Run the one-step GPU diagnostic benchmark with
`--reverse-segment-input-diagnostics` using the database black-box TOML.  The
expected result is:

- no `database-geometry-reverse` output;
- no `database post-segment geometry sweep` output;
- all database reverse trace values finite;
- no initial-root support leaf-shape error;
- root timing diagnostics report `direct_geometry_transpose_s=n/a`.

Only after that passes should the 16-step / four-segment GPU benchmark be run.
