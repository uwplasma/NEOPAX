# Differentiating Initial Ambipolar Er

## Goal

Add differentiable initial ambipolar-Er construction for the transport AD benchmarks without
contaminating the already validated reverse paths.

The missing derivative is:

```text
profiles + geometry/support -> initial ambipolar Er root -> initial transport state -> rollout -> objectives
```

This should work for:

```text
profile-only / frozen-geometry reverse AD
profiles_plus_realtime_geometry reverse AD
forward FD comparison benchmarks
```

The default behavior should remain unchanged unless the new option is explicitly enabled.

## Current Validated Pieces

The following pieces should not be changed while adding this feature:

```text
Radau accepted-replay reverse rule
realtime geometry/support payload cotangent split
VMEC raw-block transpose pullback
NTX support payload pullback
profile-only frozen-geometry reverse behavior when initial-Er AD is disabled
```

The realtime geometry reverse path now matches frozen-linearized accepted-replay FD for the 16-step
benchmark, so the ambipolarity root derivative should be added as an initial-state boundary
contribution, not as a Radau solver change.

## Proposed User-Facing Switch

Add an explicit opt-in flag/config override, for example:

```bash
--initial-Er-root-ad implicit
```

or:

```bash
--differentiate-initial-ambipolar-Er
```

Default:

```text
off / current behavior
```

This keeps all existing validated commands unchanged.

## Mathematical Rule

At each radius, the selected ambipolar root satisfies:

```text
R(E_r, n, T, geometry/support) = 0
```

Forward sensitivity:

```text
dE_r/dp = - (dR/dE_r)^(-1) dR/dp
```

Reverse sensitivity:

```text
lambda = E_r_bar / (dR/dE_r)
p_bar += - lambda * dR/dp
```

For a coupled multi-radius root:

```text
lambda = solve((dR/dE_r)^T lambda = E_r_bar)
p_bar += - (dR/dp)^T lambda
```

If the root is independent at each radius, use the scalar rule per radius.

## Branch Handling

The root finder may identify multiple ambipolar roots. The reverse rule should not differentiate
through the branch search/selection logic.

Forward:

```text
find candidate roots
select root with current policy
save selected root and branch/root index
```

Reverse:

```text
hold selected branch fixed
differentiate the residual equation at the selected root
```

If a perturbation changes the selected branch, centered FD may disagree with the reverse derivative.
That is expected: the map is nonsmooth at a branch switch.

Diagnostics should include:

```text
selected root index per radius
selected root value
dR/dEr at selected root
min |dR/dEr|
optional FD branch-switch warning
```

## Integration Point

Add the derivative at the initial-state construction boundary.

Current structure:

```text
parameters -> initial transport state -> accepted replay rollout -> objectives
```

New optional structure:

```text
parameters + geometry/support
  -> initial profile values
  -> initial ambipolar Er root
  -> initial transport state
  -> accepted replay rollout
  -> objectives
```

Reverse boundary:

```text
initial_state_bar
  -> existing profile/state pullback
  -> optional initial ambipolar-Er implicit pullback
```

For frozen geometry:

```text
profile bars are added to profile gradients
geometry/support bars are not requested or are discarded
```

For realtime geometry:

```text
profile bars are added to profile gradients
geometry/support bars are added to the realtime support payload cotangents
existing raw_block_transpose maps payload bars to VMEC harmonic gradients
```

Do not create a new VMEC pullback path.

## Suggested Helper Shape

Implement a compact helper around the existing initial-Er root construction:

```text
initial_ambipolar_Er_root_pullback(
    selected_Er,
    profiles,
    geometry_or_support_payload,
    Er_bar,
)
```

Return:

```text
profile_parameter_bar contribution
geometry_payload_bar contribution
diagnostics
```

If integrating directly into initial-state construction is cleaner, use:

```text
build_initial_state_with_ambipolar_Er_ad(...)
```

but keep the opt-in behavior explicit.

## Benchmark Plan

1. Isolated initial-Er root benchmark, profile parameter, frozen geometry.

```text
parameter -> initial Er only
compare FD vs implicit reverse
```

2. Isolated initial-Er root benchmark, realtime geometry parameter such as `RBC:1:0`.

```text
VMEC harmonic -> geometry/support payload -> initial Er only
compare FD vs implicit reverse
```

3. Full profile-only/frozen-geometry reverse benchmark with option off.

Expected:

```text
identical to current validated result
```

4. Full profile-only/frozen-geometry reverse benchmark with option on.

Expected:

```text
changes only by the initial-Er derivative contribution
FD with same option should match
```

5. Full realtime-geometry reverse benchmark with option off.

Expected:

```text
identical to current validated result
```

6. Full realtime-geometry reverse benchmark with option on.

Expected:

```text
FD and reverse match when selected ambipolar branches are stable
branch-switch diagnostics explain mismatches if roots change branch
```

## Non-Goals

Do not differentiate through:

```text
root bracketing/search loops
discrete branch selection
FD perturbation logic
Radau accepted/rejected schedule selection
```

Do not smooth branch selection unless a separate smooth objective/root policy is explicitly chosen.

## Open Questions

1. Is the current initial-Er ambipolar solve independent per radius or coupled across radius?
2. Where is the canonical residual function used for the initial-Er root?
3. Does the existing root finder already expose selected branch/root indices?
4. Does initial-Er construction currently depend on the preloaded NTX support payload, runtime
   geometry, or both?
5. Should branch-switch FD diagnostics be added to the FD benchmark first, or only after the implicit
   root rule exists?
