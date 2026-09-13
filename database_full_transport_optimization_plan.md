# Database full-transport optimization contract

## Frozen lanes

- Do not edit or replace the reverse-AD benchmark lane.
- Do not edit or replace the validated database initial-Er/root-only
  optimization lane or any of its accepted JIT boundaries.
- A root-only optimization must not construct, compile, or execute transport
  time evolution.

## Opt-in full-transport composition

When full transport is selected, one optimizer evaluation must execute:

```text
one current VMEC raw-block solve
  -> the existing selected initial-Er root
  -> the existing benchmark Radau forward transport
  -> the existing benchmark segmented transport reverse
  -> one selected-root implicit cotangent passage
  -> the existing payload/database-to-VMEC transpose
  -> the optimization least-squares adapter
```

The transport continuation must call the existing benchmark reverse functions.
It must not disable the root derivative and add a manual correction, run a
second transport reverse sweep, reconstruct transport derivatives, or change
benchmark defaults.

## Validation order

1. Establish a no-duplication optimization entry point that calls the
   unchanged integrated benchmark composition once.
2. Compare that entry point with the benchmark reference on the small
   `(5,25,31)` database using 16 accepted steps, segment length 4, and four
   reverse segments.
3. Repeat one fixed optimization evaluation after warmup and measure RSS,
   live JAX arrays, and existing segment dispatch caches.
4. Only if retained memory still grows, test one optimization-only transport
   JIT boundary at a time.  Every candidate must pass parity before it can be
   kept.

## Current measured status

The no-duplication full-transport baseline is numerically repeatable, but it
is not yet memory-bounded across optimizer evaluations.  The small database
test used `(n_theta,n_phi,n_xi)=(5,25,31)`, 16 accepted steps, segment length
4, one warmup, and repeated evaluation of the same 24-parameter point.

After trial 0 established the RSS baseline, trial 1 reported:

```text
rss_delta=+466.2 MiB
live_jax_arrays=735 -> 847
segment_cache=(2, 2, 0) -> (2, 3, 0)
residual_repeat_max_abs=0.000e+00
jacobian_repeat_max_abs=1.435e-08
```

This is retained compilation/array state, not an acceptable flat-memory
result.  Trial 2 is not required to establish the failure.

## Located cache-key cause

Both of the following transport kernels declare `execution_context` as a
static JIT argument:

- `_radau_segment_replay_minimal_with_primal_records_call`
- `_radau_database_segment_reduced_cotangent_bwd_with_table_support_call`

`prepare_reverse_static_setup()` currently constructs a new
`_RadauSolveExecutionContext` for every optimizer evaluation.  That dataclass
uses identity equality (`eq=False`), so each new instance is a distinct JAX
static cache key.  The warmup-to-trial transition recompiled the database
segment kernel, and the measured trial-0-to-trial-1 cache growth specifically
showed the minimal replay cache growing again.

The per-segment diagnostic text `jax_trace_cache outer=0->0 step_call=0->0`
does not measure these two database-specific call boundaries and therefore
does not contradict the cache growth above.

## Required next implementation boundary

Add an optimization-only persistent transport segment stage that owns a
stable static execution context and accepts fresh numeric values for:

- segment-start carries;
- segment arrays;
- replay records;
- reduced cotangents; and
- geometry/database support payload leaves.

The stage must cover both minimal replay and database segment reverse.  It
must call the existing benchmark kernels/equations, leave the benchmark and
validated root-only lanes unchanged, and pass benchmark parity before its
memory result is accepted.

## Next-step implementation sequence

1. Freeze the current benchmark reference, root-only optimization lane, and
   no-duplication full-transport baseline.  Do not add diagnostic hooks to the
   benchmark path.
2. In optimization-only test plumbing, capture the two consecutive
   `_RadauSolveExecutionContext` instances and classify their fields as:
   invariant static callables/configuration, invariant numeric values, or
   genuinely changing numeric values.  Reuse is allowed only after confirming
   that fresh geometry/database values already enter through the dynamic
   support/carry arguments rather than hidden context fields.
3. Add the first candidate persistent stage around
   `_radau_segment_replay_minimal_with_primal_records_call` only.  The stage
   owns one validated static execution context and accepts fresh
   segment-start carries and segment arrays on every evaluation.  It calls the
   existing replay kernel and does not reproduce its equations.
4. Select that stage only for
   `reverse_stage_mode="database_full_transport_optimization"`.  The
   `benchmark` mode and every initial-Er/root-only mode remain on their current
   paths.
5. Add unit checks that the stage is built once per optimization problem,
   receives fresh numeric inputs, rejects a changed tree/shape/dtype layout,
   and is never constructed by root-only problems.
6. Run full-transport parity against the unchanged benchmark reference on the
   small `(5,25,31)` database with 16 accepted steps and four segments of
   length 4.  Reject the candidate if residual/Jacobian parity fails.
7. If parity passes, run one warmup plus at least three repeated evaluations.
   Acceptance requires the minimal-replay cache size and live-JAX-array count
   to remain flat after warmup, with no repeated evaluation-sized RSS growth.
8. Inspect the database reverse-segment cache independently.  Only if it still
   grows after the replay fix, add a second persistent stage around
   `_radau_database_segment_reduced_cotangent_bwd_with_table_support_call`,
   then repeat the same unit, parity, and memory gates.

No candidate is retained merely because it lowers RSS.  Numerical parity and
the one-VMEC/one-root/one-forward/one-reverse composition remain mandatory.
