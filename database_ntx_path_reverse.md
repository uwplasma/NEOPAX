# Reverse AD for realtime VMEC + `ntx_scan_runtime`

## Target

The target is the existing forward model selected by
`[neoclassical].flux_model = "ntx_scan_runtime"` when geometry is rebuilt
from realtime VMEC. It is **not** the static file-backed `ntx_database`
model.

For every rebuilt geometry the forward path already does:

```text
VMEC state -> geometry -> live scan channels + scan surfaces
          -> ntx.build_ntx_neopax_scan_from_surfaces
          -> NEOPAX Monoenergetic database -> transport flux
```

The reverse path must transpose that same chain. The cached database is a
primal cache, not an independent differentiable support input.

## Invariants

* Leave all established exact-runtime NTX selectors and the current fastest
  exact reverse mode unchanged.
* Do not make files, VMEC solves, profile output, XLA dumps, or real transport
  rollouts part of unit tests.
* Do not silently use an exact-NTX prepared-system rule for a scan-runtime
  model. The scan has no preloaded exact support payload.
* Preserve arbitrary objective count: no solution may assume ten objectives.

## Completed seam work

1. `NTXRuntimeScanTransportModel.with_runtime_scan_payload()` replaces live
   geometry, channels, and scan surfaces and clears the stale cache unless a
   new database is deliberately supplied.
2. `with_support_payload()` rebuilds the live scan database from those leaves.
3. `realtime_geometry_payload_for_runtime()` identifies a scan runtime with
   the tagged payload `ntx_scan_runtime`; its reverse support payload contains
   only `{geometry, channels, surfaces}` and intentionally excludes `database`.
4. `runtime_with_realtime_geometry_payload()` can reconstruct a runtime from
   that tagged payload.
5. `runtime_with_realtime_geometry_reverse_support_payload()` is the shared
   reverse-only replacement adapter. For a scan it accepts only geometry,
   channels, and surfaces and deliberately clears the cached database.
6. CPU-only mock oracles pass for payload replacement and for the `a_b` JVP
   through live scan reconstruction. The latter correctly treats the
   zero-`Er_tilde` log-floor column as having zero tangent.

## Next implementation sequence

### 1. Make reverse setup model-aware

Refactor `prepare_realtime_geometry_support_segment_core_setup()` so it
obtains a tagged payload through the capability helper rather than immediately
calling `find_ntx_support_payload()`. Exact runtimes must continue to produce
their existing `{"geometry", "ntx_support"}` tree unchanged. Scan runtimes
must produce `{geometry, channels, surfaces}`.

At this stage reject exact-only rebuild and initial-cache selectors for a scan
runtime with a clear setup-time error. This is preferable to tracing into a
prepared-NTX hook that the model does not own.

### 2. Generalize the runtime replacement dependency

The reverse callbacks currently call
`runtime_with_ntx_support_payload(runtime_with_geometry_payload(...), ...)`.
Introduce a sibling tagged replacement dependency using
`runtime_with_realtime_geometry_payload()`. Route scan payloads through it;
leave the exact callback branch untouched.

### 3. Generalize the outer VMEC payload transpose

The final geometry-to-VMEC raw-block transpose currently extracts an
`ntx_support` bar. Add a scan branch that receives the bars of
`geometry/channels/surfaces`, rebuilds the runtime through the tagged helper,
and applies the existing VMEC payload VJP. The scan database must remain
internal to this function so no cache cotangent is created.

Implemented: the scan branch now transposes one combined
`{geometry, channels, surfaces}` mapping. It deliberately does **not** split
geometry into the exact path's separate direct-geometry and `ntx_support`
branches: the live database reconstruction depends on all three leaves, so a
split would both omit the scan geometry chain and double-count it.

First validate this with a mocked scan builder against a direct JAX VJP of the
same local chain. Include the `Er_tilde -> Er_list` and `drds` channels.

### 4. Add generic initial-boundary handling

The selected-root/initial-cache helpers are exact-support-specific. For the
scan runtime, use a generic VJP of the same tagged runtime reconstruction and
the initial residual/lagged-response calculation. Do not reuse the exact
compact prepared-NTX helper.

### 5. Structural integration checks

CPU/mocked tests must establish:

* reverse setup chooses scan payloads without calling exact support lookup;
* an exact-only selector is rejected at setup for a scan runtime;
* scan payload VJP equals direct local JAX VJP; and
* the exact runtime keeps its existing payload and selector route.

### 6. Remote validation

Only after the structural tests pass, run a small scan-runtime reverse smoke
on the benchmark machine, then compare one profile and one VMEC-coefficient
AD derivative to central FD before running `--objective all`.

## Black-box direct-RHS support reverse

Black-box transport has no lagged response cache.  Its geometry support bar
must therefore come from the direct RHS at every Radau stage, not from the
lagged rebuild/cache transpose.

Implemented first seam:

1. Radau now has a separate optional `flat_rhs_direct_support_pullback`.
   It is called only when the realized stage has no lagged response; all
   existing fixed-lagged exact-NTX paths are unchanged.
2. The equation system reconstructs the live support payload before taking
   the direct RHS VJP.  In a composite, only the neoclassical payload-owning
   model is replaced; turbulence and classical models remain present and can
   contribute their direct geometry dependence through the rebuilt equation
   system.
3. The contract is model-capability based: models exposing
   `with_support_payload` receive their appropriate payload (exact NTX gets
   `ntx_support`; live scan gets `{geometry, channels, surfaces}`); models
   without that capability remain generic geometry participants.

Still required before a remote benchmark:

* wire the black-box support payload through the public realtime-geometry
  reverse setup (the current exact/scan lagged setup does not yet select it);
* add the exact-NTX direct-RHS prepared-lowdot override.  Until then exact
  black-box uses the correct generic RHS VJP, not the performance rule used
  by lagged exact NTX;
* validate one direct-RHS stage against a direct local VJP and then an AD/FD
  coefficient check remotely.
