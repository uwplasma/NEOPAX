# Realtime Transport Geometry Reverse AD Current State

Date: 2026-07-23

## Goal

Extend the validated profile reverse AD lane to realtime VMEC geometry without changing forward-solver math, frozen-geometry profile reverse behavior, or the realtime VMEC forward construction.

The realtime geometry reverse lane should:

- use the same primal geometry construction as the realtime forward solver,
- recover the frozen-geometry/profile reverse lane as the geometry-frozen limit,
- pull transport-objective cotangents back to VMEC harmonics,
- eventually print gradients for all selected VMEC harmonics.

## Current Reference Commands

Realtime geometry FD with accepted replay and split-payload diagnostic:

```bash
python ./examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --parameter RBC:1:0 \
  --geometry-fd-lane frozen_linearized \
  --accepted-step-limit 16 \
  --radau-jacobian-reuse-mode legacy \
  --replay-mode accepted \
  --split-payload-fd-diagnostic
```

Realtime geometry reverse AD:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --reverse-parameter-mode profiles_plus_realtime_geometry \
  --reverse-geometry-parameter RBC:1:0 \
  --realtime-geometry-gradient-path reverse_payload \
  --ntx-exact-derivative-mode direct \
  --ntx-exact-derivative-field-pullback-mode generic_jvp \
  --objective all \
  --accepted-step-limit 16 \
  --radau-jacobian-reuse-mode legacy \
  --timing-mode jit-warm \
  --reverse-segment-length 4 \
  --reverse-stage-adjoint-solve-mode bicgstab \
  --reverse-rhs-transpose-mode explicit_ntx_interpolated \
  --reverse-step-bwd-mode reduced_cotangent
```

## Current FD Evidence

The latest FD split for `RBC:1:0`, 16 accepted steps, frozen-linearized geometry lane:

```text
objective finite-difference gradients:
  softmax_Er                         -4.739435e+01
  smooth_root_proxy                  -2.922456e-02
  Er2_volume_average                 -1.672623e+02
  Er_volume_average                  -1.809277e+01
  electron_temperature_volume_average -2.246442e-02
  total_pressure_volume_average      -7.747999e-02
  alpha_power_volume_average          1.253217e-03

fixed-final-state explicit geometry finite-difference gradients:
  softmax_Er                          0.000000e+00
  smooth_root_proxy                   0.000000e+00
  Er2_volume_average                  3.543218e-01
  Er_volume_average                  -5.370992e-02
  electron_temperature_volume_average -1.286930e-02
  total_pressure_volume_average      -7.753834e-02
  alpha_power_volume_average         -1.924327e-03

baseline-geometry final-state finite-difference gradients:
  softmax_Er                         -4.739435e+01
  smooth_root_proxy                  -2.922456e-02
  Er2_volume_average                 -1.676166e+02
  Er_volume_average                  -1.803906e+01
  electron_temperature_volume_average -9.595126e-03
  total_pressure_volume_average       5.835553e-05
  alpha_power_volume_average          3.177544e-03

geometry-only final-state finite-difference gradients:
  softmax_Er                         -1.721271e+01
  smooth_root_proxy                  -1.910128e-02
  Er2_volume_average                 -1.913312e+02
  Er_volume_average                  -2.719813e+00
  electron_temperature_volume_average -4.539670e-03
  total_pressure_volume_average      -5.428072e-05
  alpha_power_volume_average          7.258337e-04

NTX-support-only final-state finite-difference gradients:
  softmax_Er                         -3.018171e+01
  smooth_root_proxy                  -1.012344e-02
  Er2_volume_average                  2.371468e+01
  Er_volume_average                  -1.531924e+01
  electron_temperature_volume_average -5.055458e-03
  total_pressure_volume_average       1.126255e-04
  alpha_power_volume_average          2.451711e-03
```

Interpretation:

- The explicit objective geometry term is correct and dominates pressure.
- FD says pressure final-state sensitivity is nearly zero:
  `-5.428072e-05 + 1.126255e-04 = 5.834478e-05`.
- FD says Te/pressure/alpha mismatches are not from missing NTX support alone.
- The FD split is consistent internally:
  explicit geometry + baseline-geometry final-state = full FD.

## Current Reverse Evidence

### Before the `field`/`geometry` payload fix

The reverse component split before the `field`/`geometry` fix showed the pressure mismatch:

```text
total_pressure_volume_average:
  objective_explicit      ~= -7.753767e-02
  transport_rhs           ~=  7.320592e-03
  initial_cache           ~=  2.196721e-05
  final_state_components  ~=  7.342559e-03
  total reverse           ~= -7.019511e-02
```

FD says the corresponding final-state contribution should be:

```text
baseline-geometry final-state FD ~= 5.835553e-05
```

Therefore the remaining bug is not the explicit objective geometry pullback and not simply a missing lagged NTX rebuild term. The bad term is the reverse final-state/support cotangent accumulation, specifically the transport-RHS/support contribution feeding `step_support_bar_leaves_accum`.

### FD payload-reconstruction diagnostic

The FD script with `--split-payload-fd-diagnostic` now directly compares:

```text
prepared runtime geometry map:
  prepare_transport_solver_components(perturbed_runtime).equation_system

reverse payload reconstruction map:
  baseline_equation_system.with_geometry_payload(perturbed_geometry)
```

Before the fix, pressure differed:

```text
local RHS geometry finite-difference component sums:
  - pressure: fd_prepared_geometry=1.630660e+03
              fd_with_geometry_payload=1.709301e+03
              diff=7.864125e+01
```

Root cause: `ComposedEquationSystem.with_geometry_payload(...)` only replaced nested dataclass
fields named `geometry`. The turbulent power model used in the realtime benchmark is
`PowerAnalyticalTurbulentTransportModel`, which stores the same transport geometry object as
field name `field`. Therefore the reverse payload reconstruction left this turbulence model on
baseline geometry while the prepared FD map used perturbed geometry.

Patch:

```python
if field.name in {"geometry", "field"}:
    updates[field.name] = geometry
```

After syncing that patch, the FD diagnostic confirmed the maps agree locally:

```text
local RHS geometry finite-difference component sums:
  - density:  diff=0.000000e+00
  - pressure: diff=0.000000e+00
  - Er:       diff=0.000000e+00
```

This confirms the stale turbulence-geometry reconstruction bug is fixed.

### Reverse after the `field`/`geometry` payload fix

The post-fix reverse run moved much closer to FD:

```text
softmax_Er:
  FD = -4.739435e+01
  AD = -4.739231e+01

Er2_volume_average:
  FD = -1.672623e+02
  AD = -1.672543e+02

Er_volume_average:
  FD = -1.809277e+01
  AD = -1.809122e+01

electron_temperature_volume_average_keV:
  FD = -2.246442e-02
  AD = -2.327820e-02

total_pressure_volume_average:
  FD = -7.747999e-02
  AD = -7.605590e-02

alpha_power_volume_average_mw_m3:
  FD =  1.253217e-03
  AD =  1.374368e-03
```

The branch split shows NTX support and explicit objective geometry are already consistent:

```text
pressure NTX:
  FD ~= 1.126255e-04
  AD ~= 1.124851e-04

pressure explicit:
  FD ~= -7.753834e-02
  AD ~= -7.753767e-02
```

The remaining mismatch is in the geometry-only final-state/transport-RHS branch:

```text
FD geometry-only final-state pressure ~= -5.428072e-05
AD transport_rhs.geometry pressure     ~=  1.369281e-03
```

### Current patch waiting for test

Because the post-fix run still has `support_rebuild_count=12`, a remaining missing direct-geometry
term may come from lagged-response rebuilds. The previous lagged-rebuild attempt was done before
the `field`/`geometry` fix and therefore used a stale turbulence geometry map. A narrower reverse-only
patch has now been added:

```text
ComposedEquationSystem.pullback_build_lagged_response_support_payload(...)
```

For combined realtime payloads it now computes:

```text
ntx_support_bar = existing NTX/support build-lagged-response pullback
geometry_bar    = VJP of with_geometry_payload(...).build_lagged_response(state)
```

This should be tested with the same reverse command. It does not alter the forward solver path.

## Suspected Code Path

The suspicious reverse path is:

```text
benchmark_transport_reverse_ad_only.py
  _reverse_all_objectives_support_payload_bar_for_parameter_vector
    -> _radau_segment_reduced_cotangent_bwd_batched_with_support_call
      -> _execute_radau_accepted_step_next_reduced_cotangent_batched_bwd_with_support
        -> _radau_exact_stage_residual_support_pullback
        -> physics_context.flat_rhs_lagged_response_support_pullback
        -> ComposedEquationSystem.pullback_evaluate_with_lagged_response_support_payload
```

The specific accumulated reverse component is:

```text
step_support_bar_leaves_accum
```

This component is printed as:

```text
transport_rhs
```

and contributes to:

```text
final_state_components_sum
```

## Timing Diagnostics Added

`examples/benchmarks/benchmark_transport_reverse_ad_only.py` currently has diagnostic sync timing around:

- realtime geometry runtime build,
- solver component preparation,
- support reverse profile-state VJP,
- support reverse initial-carry VJP,
- support reverse realized-schedule VJP forward residual capture,
- segmented cotangent sweep.

These diagnostics only add `jax.block_until_ready` and prints; they are intended to identify why reverse appears slower before the segment reverse compile. They should not change math.

## Important Caution

The lagged-rebuild geometry pullback must remain reverse-only. Do not rebuild or perturb the
production forward runtime/schedule to test this. The safe comparison remains:

```text
FD accepted replay with frozen linearized geometry
vs
reverse payload cotangent through the same accepted replay schedule
```

## Next Steps

1. Run the reverse command again after the `pullback_build_lagged_response_support_payload`
   geometry-branch patch.
2. Compare reverse `geometry_branch` and `ntx_support_branch` final-state components against FD:
   - `fd_final_state_geometry_branch`
   - `fd_final_state_ntx_support_branch`
3. Inspect whether `_execute_radau_accepted_step_next_reduced_cotangent_batched_bwd_with_support` is using the same lagged response and stage states as the accepted replay FD.
4. If needed, add a local single-step FD-vs-reverse support diagnostic for the pressure RHS/support cotangent, not a full-run workaround.
