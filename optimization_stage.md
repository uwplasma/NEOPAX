# Persistent reverse-AD stage for optimization

## Purpose

Stop the observed per-evaluation RSS growth in NEOPAX optimization runs while
preserving the existing reverse-AD equations, numerical results, and
benchmark-good reverse route.

This is deliberately **not** an optimizer-wide fused JIT.  It follows the
useful VMEX principle: create bounded residual/reverse kernels once per
optimization stage, then pass changing numerical inputs to those kernels on
each least-squares evaluation.

The first implementation scope is the already implemented
`ntx_exact_lij_runtime` reverse-AD path and the geometry + initial-Er-root
objectives:

- maximum Er;
- left and right Er transition objectives;
- bootstrap current;
- geometry objectives in the same least-squares problem.

Full time-evolved transport is a later phase of this same design.  It must not
be implemented by reusing an initial-root-only kernel in a different
mathematical context.

## Non-negotiable invariants

1. Geometry is evaluated once per transport/root evaluation and its result is
   shared by all geometry and initial-root residuals.
2. The staged and unstaged routes use the same equations and reverse payload
   assembly.  This work does not alter reverse-AD mathematics or target
   numerical values.
3. The benchmark-good public route remains the reference:

   ```text
   geometry_active_initial_er_root_only_reverse_table
     -> realtime_geometry_transport_reverse_table_from_payload_cotangents
     -> realtime_geometry_payload_pullback_result
   ```

4. Normal reverse-AD benchmarks retain their current behavior when no stage is
   supplied.  They must not silently opt into a different derivative path.
5. There is one source of truth for transport, ambipolarity, and payload
   reverse mathematics.  Do not create copied optimization-only physics
   functions.
6. A model whose reverse-AD adapter has not been implemented must fail clearly.
   It must never silently fall back to another model or derivative route.

## Why a stage is needed

Current initial-root residual helpers call
`build_local_particle_flux_evaluator(state_with_er)`.  The resulting local
function captures the changing state in a Python/JAX closure.  Repeating this
construction inside reverse transformations can retain new native compiled or
AD-executable state per least-squares evaluation.

Wrapping that existing closure in a larger `jax.jit` is not a solution: it
would still capture a different state per evaluation and can create an even
larger graph.  Nor is the plan to rewrite every custom VJP as `jax.grad`.

Instead, the few repeated reverse boundaries must expose pure functions with
explicit numerical inputs, for example:

```python
local_charge_residual(
    transport_state, er_profile, support_payload, static_model_data
) -> residual
```

The stage owns bounded JVP/VJP/Jacobian kernels for these functions.  Changing
DoFs, profiles, Er, VMEC state, or Radau acceptance masks is data flowing into
an existing kernel; it does not create a new stage.

## Model selection and adapter boundary

Introduce a private `TransportReverseStageAdapter` protocol.  Adapter choice
is made once, during construction of the optimization stage, from the TOML
selected flux model and its static layout.

Initial registry behavior:

```text
ntx_exact_lij_runtime -> ExactLijReverseStageAdapter
ntx_database          -> clear NotImplementedError until its AD path exists
ntx_runtime_scan      -> clear NotImplementedError until its AD path exists
other model           -> clear NotImplementedError
```

This distinction is essential.  `NTXExactLijRuntimeSupport` and
`compact_initial_er_ntx_support_pullback_leaves` are exact-Lij-specific;
database and runtime-scan models use different local evaluator/support
construction.  They must later receive dedicated adapters which implement the
same protocol, not be forced through the exact-Lij payload type.

The protocol has separate entry points for:

- initial-Er root local residual and its compact reverse contribution;
- full transport vector-field/support work and its reverse contribution;
- static metadata used to key/validate the stage.

Sharing static model data or cached kernels between these operations is fine.
Sharing an initial-root mathematical kernel with full transport is not.

## Public API shape

Keep the existing reverse entry points and add an optional private stage
argument at their narrowest useful boundary:

```python
geometry_active_initial_er_root_only_reverse_table(..., stage=None)
evaluate_geometry_initial_er_root_only_least_squares_benchmark_tables(..., stage=None)
realtime_geometry_transport_reverse_table_from_payload_cotangents(..., stage=None)
realtime_geometry_payload_pullback_result(..., stage=None)
```

`stage=None` takes the present implementation route.  A supplied stage invokes
the selected adapter only at the repeated residual/payload boundary; it does
not replace the outer reverse-table assembly.

The optimization problem builds one `GeometryTransportReverseStage` once per
optimization stage and passes it to evaluations.  The stage is private
optimization lifecycle machinery, not a second physics implementation.

## Implementation phases

### Phase 1 — map and freeze the exact-Lij boundary

1. Record the exact current call chain for the four initial-root objectives.
2. Identify every state-capturing evaluator/VJP construction within that
   chain, separately from the existing compact exact-Lij payload pullback.
3. Add equality tests for residuals and Jacobian rows at representative
   parameter vectors before changing behavior.

### Phase 2 — pure exact-Lij adapter operations

1. Add exact-Lij adapter operations accepting state, Er, support, and other
   changing values explicitly.
2. Make these operations call the existing exact-Lij flux/root mathematics;
   do not duplicate formulas or alter root selection rules.
3. Keep the current state-capturing helper as the unstaged compatibility path
   until staged/unstaged agreement is verified.

### Phase 3 — persistent bounded kernels

1. Implement `GeometryTransportReverseStage` with the exact-Lij adapter and
   static configuration validation.
2. Create the bounded residual and reverse kernels once at stage construction.
3. Ensure DoFs and all changing numerical values are dynamic array inputs.
4. Do not place the entire SciPy least-squares loop, geometry solve, or all
   optimization iterations inside one JIT.

### Phase 4 — integrate the initial-root least-squares route

1. Thread the optional stage through the existing benchmark-good reverse table
   route.
2. Construct the stage once in the geometry + initial-Er-root least-squares
   problem.
3. Preserve the existing one shared raw geometry solve for mixed geometry and
   root objectives.
4. Leave benchmark callers unstaged by default.

### Phase 5 — validate exact-Lij initial-root optimization

Before requesting a user-scale optimization run, perform local small-scope
comparisons of staged and unstaged results:

- objective/residual values;
- Jacobian rows and selected directional derivatives;
- payload cotangents where exposed;
- one cold-stage compile time and repeated warm evaluation time.

Then run one four-objective repeated-evaluation acceptance test.  Report RSS
after each evaluation, but treat the result as an acceptance criterion rather
than a diagnostic detour.  Acceptance requires no sustained material RSS slope
after warmup and agreement within established derivative tolerances.

## Full transport extension

The Radau implementation uses a fixed-size `lax.scan(max_total_steps)` with
`active_mask` and `accepted_mask`.  Therefore a different number of accepted
steps is dynamic data and does not, by itself, require a new stage or
compilation.

When extending to full transport:

1. add a full-transport entry point to the exact-Lij adapter;
2. use the same static-stage key, including solver layout and
   `max_total_steps`;
3. keep adaptive acceptance/failure masks dynamic;
4. compare staged and unstaged full-transport reverse outputs before enabling
   staged optimization;
5. later add database and runtime-scan adapters only after their own
   reverse-AD implementations and benchmark comparisons exist.

## Performance and correctness expectations

- Normal benchmark path: no expected compilation or execution change when
  `stage=None`.
- Staged optimization path: a deliberate one-time compile/setup cost per
  static stage is expected.
- Warm staged evaluations should avoid repeated graph/executable retention;
  this is the memory objective, not a guaranteed speedup.
- A changed TOML static configuration (model kind, grid/layout/species,
  solver shape, or `max_total_steps`) creates a new stage before a run.
- Changed optimization DoFs, profiles, Er values, VMEC state, and accepted-step
  count must not create a new stage.
- Any change that would alter benchmark derivative values or reverse-AD
  mathematics requires explicit review before implementation.
