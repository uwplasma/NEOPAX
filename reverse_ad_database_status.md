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

The next one-step run passed that shape boundary and completed the one batched
recorded-scan fold.  It then ran out of GPU memory in the *final VMEC
payload-to-state transpose*, after the scan fold:

```text
runtime_scan_payload_from_state
  -> 56 active float payload leaves
  -> raw payload-to-state VJP batch
  -> attempted 3.73 GiB allocation
```

The scan fold itself completed; this is not a scan reverse or table-bar OOM.
The raw fallback built one retained payload VJP per objective row.  The
database scan lane had the existing compact JVP/tangent-contraction route
explicitly disabled.  That guard has now been removed for the combined scan
payload.  The new route keeps geometry, scan channels, and scan surfaces as
one coupled function and computes, for each VMEC parameter tangent,
the JVP/bar contraction.  By VJP/JVP duality this is the same derivative as
the raw payload-state transpose, without materializing its large state-bar
batch.  This correction is not yet GPU-validated.

The first compact-JVP attempt reached that route and exposed a trace-safety
bug, not a physics or memory failure: the static `ntx_scan_rho` validation
used `bool(jnp.all(...))` inside the JVP.  Scan locations are configuration
coordinates, so they are now validated and captured as a host NumPy constant
before the differentiated payload function is entered.  VMEC state still
drives geometry, channels, and surfaces; only the fixed scan coordinate axis
is excluded from differentiation.  This correction is not yet GPU-validated.

The next compact-JVP attempt passed the scan-coordinate validation and then
found a second static-data leak while deriving the Boozer `R00` normalization:
the scan surface-index metadata was recomputed from `context.static.s` inside
the JVP.  That metadata is now precomputed from the fixed scan radii and
VMEC grid outside the differentiated payload function and passed to the R00
builder, together with already-prepared Boozer constants/mode metadata.  The
R00 values themselves remain functions of the VMEC state.  This correction is
not yet GPU-validated.

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
- `compact_payload_tangent_contract=True` after the recorded scan fold;
- no raw payload-state-bar allocation / `RESOURCE_EXHAUSTED` error.

Only after that passes should the 16-step / four-segment GPU benchmark be run.

## 2026-09-09 update: compact native face-geometry boundary

This section supersedes the older statement above that the database reverse
has no local geometry contribution.  The correct fixed-table decomposition is
now:

```text
database RHS bar
  -> table bar                    -> accumulated -> one recorded scan VJP
  -> direct centre-flux geometry  -> local VMEC/transport geometry bar
  -> direct native face geometry  -> local VMEC/transport geometry bar
  -> fixed-flux equation geometry -> local VMEC/transport geometry bar
```

Only the table bar reaches the recorded scan.  The three geometry terms are
local and must never capture the scan owner.

### Face geometry correction

The first bounded implementation performed a `jax.vjp` through a newly built
`ComposedEquationSystem` for each native density/temperature face.  It was
finite after the axis-mesh correction but still used about 16–17 GB host RAM
on a 30 GB machine because every local VJP retained equation/source assembly.

The active replacement is
`pullback_direct_face_flux_geometry_by_radius`:

- routed through the database, runtime-scan, and combined flux models;
- evaluates only the native fixed-table database face flux primitive;
- excludes `ComposedEquationSystem`, sources, finite-volume equation
  assembly, and the scan owner;
- follows the same local, bounded scan-over-radius structure as the existing
  direct-centre database geometry primitive;
- is selected only by the database split support hook, never the Lij path.

### Physical radial-mesh rule

The database direct-flux primitive varies the physical VMEC mesh only through
`a_b`:

```text
r_grid = rho_grid * a_b
r_grid_half = rho_grid_half * a_b
dr = r_grid_half[1] - r_grid_half[0]
```

`r_grid_half[0]` is the fixed magnetic axis and is not an independent
tangent direction.  Leaving it free produced nonfinite
`geometry.r_grid_half` bars by evaluating the radial database off its
physical manifold.  This constraint applies to direct database fluxes
(centre and faces), not the fixed-flux equation-assembly geometry VJP.

### Latest validation

```text
database_face_geometry_pullback_keeps_table_fixed_on_physical_mesh       PASS
database_face_geometry_pullback_selects_compact_model_boundary            PASS
database_face_table_pullback_rebinds_only_fixed_database_leaf             PASS
batched_database_stage_table_pullback_accepts_flattened_radau_rows        PASS
recorded_ntx_database_bar_groups_share_one_batched_scan_pullback          PASS

5 passed, 282 deselected in 13.67s
```

### Required next measurement

Run the 16 accepted-step / 4-segment GPU diagnostic benchmark.  Require both:

1. no nonfinite `geometry.r_grid_half` segment bar; and
2. a host-RAM peak below the previous roughly 16–17 GB.

The tests establish dispatch and boundary correctness; they do not yet claim
a measured benchmark memory reduction.
