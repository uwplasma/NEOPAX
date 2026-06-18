# Forward Pass / Solver Efficiency Plan

## Goal

Bring the current `NEOPAX` forward pass / solver lane back toward the cheaper `NEOPAX_MAIN` compile behavior for NTX lagged-response rebuilds, while keeping:

- the standard adaptive forward solver path numerically intact
- reverse-lane machinery independent
- benchmark/debug-only hooks out of the normal forward hot path

## Current Findings

The highest-signal remaining differences between current `NEOPAX` and `NEOPAX_MAIN` are now in the NTX forward helper path, not mainly in the adaptive Radau controller core.

Already restored:

- the forward NTX local interpolated-moment builder now uses the simpler `jax.linearize(...)` pushforward structure again, closer to `NEOPAX_MAIN`

Still likely widening the forward compile graph:

1. `_local_scan_inputs(...)` in `NEOPAX/_transport_flux_models.py`

- current code still includes extra finite-guard / dtype-selection / debug-path structure that `NEOPAX_MAIN` does not have
- this is still on the forward primal NTX rebuild path

2. Derivative-mode plumbing in the forward NTX helper stack

- current code still threads `derivative_mode_override` through forward-primal helper calls
- `NEOPAX_MAIN` uses a simpler direct forward path here
- even if runtime mode resolves to `"direct"`, the extra branching/plumbing may still widen tracing/compilation

3. Small helper indirection still present on the forward path

- current forward builder still passes through `_interpolated_moment_local_scan_primitives(...)`
- this probably matters less than items 1 and 2, but it is still a `MAIN` mismatch

## Working Hypothesis

The remaining excessive compile time is more likely coming from the NTX flux-model forward helper stack than from the adaptive Radau controller logic itself.

In particular:

- the forward primal rebuild path should be made as close as possible to the old `NEOPAX_MAIN` direct path
- reverse-support helpers should remain available, but should not widen the normal forward compile graph

## Plan

### Step 1: Simplify forward `_local_scan_inputs(...)`

Goal:

- compare current `_local_scan_inputs(...)` against `NEOPAX_MAIN`
- remove or isolate any forward-hot-path-only overhead that was introduced for reverse/debug support

Focus items:

- `jnp.isfinite(drds_value)` / `jnp.where(...)` guarding
- `jnp.result_type(...)` widening
- debug callback hooks behind `NEOPAX_TRANSPORT_NTX_LOCAL_PULLBACK_FINITE_DEBUG`

Desired result:

- the normal forward solver path uses the leanest possible local scan input construction
- reverse/debug safety hooks, if still needed, are kept out of the default forward primal lane

### Step 2: Recover a direct forward NTX evaluator path

Goal:

- make the forward-primal NTX transport-moment / coefficient-scan path structurally closer to `NEOPAX_MAIN`

Focus items:

- avoid threading `derivative_mode_override` through the normal forward helper stack
- if needed, add a separate forward-direct helper path instead of sharing a generalized path with reverse support

Desired result:

- forward solver / primal lagged rebuild uses a dedicated direct NTX path
- reverse-related derivative-mode plumbing does not contaminate forward tracing

### Step 3: Recheck timings on the same rebuild probe

Goal:

- rerun the same forward/solver case that showed the expensive rebuild timing
- compare whether rebuild cost moves closer to the older expected behavior

Success criteria:

- rebuild timing drops materially relative to the current partially restored state
- no regressions in forward solver numerics

### Step 4: Only if needed, revisit solver-side wrappers

Goal:

- only after steps 1-3, inspect whether any remaining forward-only solver wrappers are still widening compilation

This is intentionally lower priority because:

- current evidence points more strongly to the NTX flux-model forward path than to the adaptive controller machinery

## Constraints / Guidelines

- Do not contaminate the standard adaptive forward solver path with reverse-lane machinery
- Keep forward, reverse, and FD lanes independent even if that requires duplicated specialized helpers
- Do not change the normal adaptive solver behavior unless the change is explicitly for restoring the old forward behavior
- Prefer restoring a specialized forward path over routing forward through generalized AD/replay helpers

## Next Recommended Action

Start with Step 1:

- compare and simplify `_local_scan_inputs(...)` for the forward-primal NTX path
- keep any reverse/debug-specific protections separate from the default forward lane
