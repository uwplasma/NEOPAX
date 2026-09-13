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
