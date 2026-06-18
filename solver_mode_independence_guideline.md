# Solver Mode Independence Guideline

## Guideline

The transport solver must keep these three lanes independent, even if that
requires duplicated helper functions:

1. Plain forward solver lane
2. Forward AD lane
3. Reverse AD lane

Do **not** route the normal adaptive solve through AD-facing or reverse-facing
bookkeeping just to share code.

## Required design rule

The plain forward solver is the reference production path.

- It should carry only the bookkeeping needed to advance the adaptive solve.
- It should not depend on replay payload objects.
- It should not depend on custom-JVP or custom-VJP wrapper boundaries.
- It should not be forced through AD-oriented result containers if a smaller
  forward-only path is possible.

The forward AD lane may specialize around:

- realized accepted-step schedules
- tangent-only masking
- compact rollout traces
- benchmark-specific debug hooks

The reverse AD lane may specialize around:

- payload capture
- pullback/replay contracts
- accepted-step reverse composition
- reverse-only diagnostics and memory controls

## Non-goals

- Do not preserve shared helper structure at the expense of the production
  forward solver.
- Do not let benchmark needs reshape the standard adaptive solve path.
- Do not let reverse replay or payload requirements widen the forward solver
  contract.

## Immediate refactor target

1. Keep `RADAUSolver.solve(...)` on a forward-only hot path.
2. Remove AD/replay bookkeeping from the accepted-step attempt path used by the
   normal adaptive solve whenever it is not required for forward runtime.
3. Keep explicit, separate entrypoints for:
   - forward solve
   - forward AD rollout
   - reverse replay / pullback

## Acceptance check

Any future solver refactor should be rejected if it causes one of these:

- the normal adaptive solve starts calling reverse-only helpers
- the normal adaptive solve starts creating payload objects only needed by AD
- forward benchmark behavior changes because reverse/refactor plumbing was
  inserted into the production solve lane
