# Realtime-geometry reverse path for `ntx_database`

## Objective

Enable the existing segmented transport reverse-AD table and benchmark driver
to work with a realtime VMEC geometry and
`[neoclassical].flux_model = "ntx_database"`.

The desired user interface is unchanged: the existing
`benchmark_transport_reverse_ad_only.py` accepts a database TOML and uses the
same profile/geometry parameter options.  The flux-model selection in the
TOML determines whether the reverse uses the database route or the existing
exact-runtime NTX route.

## Scope and invariants

* Do not alter the existing exact-runtime NTX reverse selectors, including
  the current drds-reuse VMEC-coefficient mode and its opt-in experiments.
* Do not load a database file, write a file, or invoke VMEC inside a traced
  JAX function.
* The monoenergetic coefficient tables are fixed data.  Only the pure,
  geometry-dependent database wrapper is differentiated.
* Use the existing generic Radau state/rebuild reverse machinery initially.
  Database interpolation has no prepared NTX factorization, low-dot NTX
  adjoint, or `ntx_batched_*` support hook to reuse.
* Preserve the existing exact-runtime behaviour byte-for-byte when the TOML
  selects `ntx_exact_lij_runtime`.

## Current state

`build_runtime_context_for_vmec_state()` already creates realtime geometry,
then calls `_build_database(config, geometry)`, and builds the model selected
by `neoclassical.flux_model`.  Therefore a forward transport solve with
`ntx_database` and realtime geometry is structurally supported.

The reverse payload API is exact-runtime-specific today:

```text
{"geometry": geometry, "ntx_support": preloaded_exact_ntx_support}
```

`find_ntx_support_payload()`, `runtime_with_ntx_support_payload()`, and the
native NTX support hooks require `NTXExactLijRuntimeTransportModel`.  They
must not be called for `NTXDatabaseTransportModel`.

The loaded database stores immutable interpolation tables and a geometry scale
(`a_b`) plus derived limits.  Replacing only `runtime.geometry` would leave
that geometry-dependent database wrapper stale.  This is the central gap to
close.

## Plan

## Implementation status

* Step 1 / scale-only portion of step 2 is implemented locally: generic and
  preprocessed table objects now have a pure `a_b` replacement helper, and
  `runtime_with_geometry_payload()` uses it for `NTXDatabaseTransportModel`.
* The tagged runtime-payload selector is also implemented locally but is not
  connected to the exact reverse setup yet; it identifies database runtime
  payloads without attempting an exact-NTX support lookup.
* The symmetric tagged runtime replacement is implemented too: it replaces
  both geometry and the database wrapper for database payloads, and delegates
  to the established geometry-plus-support replacement for exact payloads.
* It deliberately does not yet route database configurations through the
  reverse setup.  That remains behind the small-oracle tests below.
* `preprocessed_ntss` is explicitly unsupported for realtime replacement
  until its non-scale geometry fit channels can be reconstructed exactly.

### 1. Freeze a small database runtime oracle

Create an in-memory CPU test using a small `NTXDatabaseTransportModel` and a
small synthetic `Geometry` payload.

Verify, independently for the database flux response and for the complete
transport RHS:

1. replacing the geometry payload changes the database runtime exactly as an
   explicitly reconstructed model does;
2. a JVP in a geometry-scale direction matches finite differences; and
3. a VJP/JVP duality check holds for the database geometry payload.

This test must not create a database file, execute a VMEC solve, or run a
transport rollout.

### 2. Define a pure database payload replacement

Add a narrowly scoped helper that takes an already loaded database and an
updated geometry and returns an equivalent database pytree whose
geometry-dependent values are rebuilt from the updated `geometry.a_b`.

The helper must rebuild every derived quantity whose loader constructor
derives it from `a_b`; it must not use `dataclasses.replace()` if that would
retain stale derived limits.  It must preserve the raw arrays (`rho`,
`nu_log`, field grid, `D11_log`, `D13`, `D33`, and the interpolation-kind
specific tables) by reference/array value.

Add exact small tests against construction through the corresponding database
class constructor for each database interpolation kind used by supported
TOMLs.  Unsupported kinds must fail clearly before entering the reverse
graph.

### 3. Introduce a model-agnostic realtime geometry payload contract

Replace the hard-coded assumption that every realtime payload contains
`ntx_support` with a tagged internal contract:

```text
exact runtime: {"kind": "ntx_exact", "geometry": ..., "ntx_support": ...}
database:      {"kind": "ntx_database", "geometry": ..., "database": ...}
```

The tag is static Python control flow at setup time, never a traced JAX value.
Provide symmetric pure runtime replacement helpers:

* exact: existing geometry + NTX-support replacement;
* database: geometry + rebuilt database replacement.

Keep the public `runtime_with_geometry_payload()` behaviour unchanged for
existing callers.  The new database helper is additive.

### 4. Route generic database support through the existing segmented reverse

At reverse static setup, select from the TOML model capability:

* `ntx_exact_lij_runtime`: retain the current support payload and permitted
  exact-NTX selectors;
* `ntx_database`: use the database payload and generic support/state VJPs.

Reject `ntx_batched_*`, `ntx_joint_*`, and NTX-specific initial-cache modes
for database configurations with a direct error describing that they require
a prepared exact-NTX system.  Allow the generic defaults (`separate`, scalar
or generic initial-cache pullbacks) without special-casing the Radau sweep.

The final objective cotangent and initial-Er root paths must use the same
database runtime replacement, so their geometry bars include database-scale
dependence and cannot silently use baseline-database data.

### 5. Add no-rollout integration tests

Use a small mocked/CPU database runtime to verify:

1. the reverse setup selects the database contract rather than calling
   `find_ntx_support_payload()`;
2. an exact-NTX-only rebuild selector is rejected for a database runtime;
3. the database geometry payload VJP matches a direct JAX VJP of the same
   local database flux calculation; and
4. exact-runtime setup still selects the existing `ntx_support` contract.

These tests are structural and in-memory.  They must not launch a full Radau
benchmark or save profiles/dumps.

### 6. Remote validation sequence

Only after the gates above pass, run on the benchmark machine:

1. a database TOML forward/reverse smoke with a small accepted-step limit;
2. reverse AD versus central FD for one profile parameter and one VMEC
   boundary coefficient; and
3. the normal `--objective all` database benchmark using generic reverse
   pullbacks.

Compare objective values and derivative signs/magnitudes before attempting
database-specific performance work.  The exact-runtime benchmark and its
known AD-vs-FD table remain the reference for the separate exact path only.

## Non-goals for the first implementation

* Reusing the exact NTX low-dot implicit adjoint for a database model.
* Adding native matrix-RHS NTX modes to database interpolation.
* Changing database resolution, interpolation policy, VMEC rebuild criteria,
  or accepted-step schedule to obtain timings.
* Optimizing the database reverse before its geometry derivative is validated.
