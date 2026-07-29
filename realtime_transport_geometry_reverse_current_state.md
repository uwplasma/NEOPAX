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

## 2026-07-24 Confirmed Reverse/FD Match

The latest reverse run with the narrow direct-geometry lagged-response pullback now matches the
frozen-linearized accepted-replay FD decomposition for `RBC:1:0`, `accepted-step-limit=16`.

Reverse command:

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

Reference FD command:

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

Full objective gradients:

```text
softmax_Er:                         FD=-4.739435e+01  reverse=-4.739373e+01
smooth_root_proxy:                  FD=-2.922456e-02  reverse=-3.002664e-02
Er2_volume_average:                 FD=-1.672623e+02  reverse=-1.672605e+02
Er_volume_average:                  FD=-1.809277e+01  reverse=-1.809261e+01
electron_temperature_volume_average FD=-2.246442e-02  reverse=-2.246430e-02
total_pressure_volume_average:      FD=-7.747999e-02  reverse=-7.747942e-02
alpha_power_volume_average:         FD= 1.253217e-03  reverse= 1.253233e-03
```

The old pressure/temperature/alpha mismatch is resolved. The key final-state component comparison:

```text
electron_temperature final-state: FD=-9.595126e-03  reverse=-9.595115e-03
pressure final-state:             FD= 5.835553e-05  reverse= 5.824472e-05
alpha final-state:                FD= 3.177544e-03  reverse= 3.177545e-03
```

The branch split also agrees:

```text
electron_temperature geometry branch:   FD=-4.539670e-03  reverse=-4.534735e-03
electron_temperature NTX branch:        FD=-5.055458e-03  reverse=-4.762568e-03 plus initial-cache NTX=-2.928711e-04

pressure geometry branch:               FD=-5.428072e-05  reverse= transport_rhs.geometry 1.096956e-04 plus initial-cache.geometry -1.639422e-04
pressure NTX branch:                    FD= 1.126255e-04  reverse= transport_rhs.ntx_support 9.052378e-05 plus initial-cache.ntx_support 2.196751e-05

alpha geometry branch:                  FD= 7.258337e-04  reverse= transport_rhs.geometry 7.330155e-04 plus initial-cache.geometry -7.181233e-06
alpha NTX branch:                       FD= 2.451711e-03  reverse= transport_rhs.ntx_support 2.302918e-03 plus initial-cache.ntx_support 1.487927e-04
```

Conclusion: the derivative math for realtime geometry reverse payload is now consistent with the
accepted-replay frozen-linearized FD benchmark. The remaining issue is performance/compilation:

```text
runtime build:                         224.414 s
support reverse realized-schedule VJP: 442.729 s
segmented cotangent sweep:            1875.798 s
total elapsed:                        3224.061 s
```

The next work should focus on reducing the cost of
`_radau_segment_reduced_cotangent_bwd_batched_with_support_call` and the support cotangent sweep,
without changing the now-matching reverse math.

## 2026-07-24 Efficiency Notes

The largest confirmed cost is still the all-objective support cotangent sweep:

```text
support reverse segmented cotangent sweep: 1875.798 s
geometry payload -> VMEC pullback:          321.047 s
```

The segment sweep is expensive because the current all-objective path carries a batched support
cotangent tree through each accepted Radau step:

```text
7 objectives x 159 support/geometry payload leaves x each segment step
```

Inside each accepted-step reverse, `_execute_radau_accepted_step_next_reduced_cotangent_batched_bwd_with_support`
does a `vmap` over objective rows for the support-payload pullback. This saves re-running the replay
for each objective, but creates a very large XLA graph and high peak memory.

The final payload-to-VMEC pullback was also doing diagnostic work by default. It sent the total
support bars plus component bars:

```text
total, objective_explicit, transport_rhs, initial_cache, initial_profile
```

With branch diagnostics enabled, this multiplies the VMEC raw-block RHS rows. A benchmark flag was
added to make those component pullbacks opt-in:

```bash
--realtime-geometry-component-pullbacks
```

Default behavior now skips the component rows and keeps only the production gradients plus the
geometry-vs-NTX branch split. Use the flag only when re-debugging the decomposition.

Latest default run without `--realtime-geometry-component-pullbacks`:

```text
runtime build:                         222.214 s
support reverse realized-schedule VJP: 439.673 s
segmented cotangent sweep:            1463.032 s
geometry payload -> VMEC pullback:     457.899 s
total elapsed:                        2903.023 s
```

The gradients remained unchanged and matched FD:

```text
softmax_Er:                         reverse=-4.739373e+01
smooth_root_proxy:                  reverse=-3.002664e-02
Er2_volume_average:                 reverse=-1.672605e+02
Er_volume_average:                  reverse=-1.809261e+01
electron_temperature_volume_average reverse=-2.246430e-02
total_pressure_volume_average:      reverse=-7.747942e-02
alpha_power_volume_average:         reverse= 1.253233e-03
```

The first run after adding the flag printed empty `components:` lines and
`final_state_components_sum=0.000000e+00`. That was only a reporting artifact:
component pullbacks were intentionally skipped. The report construction has been fixed so component
dictionaries are empty unless `--realtime-geometry-component-pullbacks` is explicitly enabled.

Another benchmark-only option was added to avoid support-bar diagnostic scans:

```bash
--skip-realtime-geometry-support-bar-diagnostics
```

This does not skip the support/geometry payload cotangents used for the derivative. It only skips
the l2/finiteness tree summaries, branch diagnostics, and JSON diagnostic payload for those support
bars. The VMEC payload pullback still performs its own active-leaf and finiteness checks.

Recommended next efficiency work:

1. Add an objective-block-size option for the realtime support sweep, e.g. process 1-2 objective
   rows per compiled kernel instead of all 7. This should reduce compile size and peak memory,
   with a controllable runtime tradeoff.
2. Add a production mode that contracts support bars directly to selected VMEC harmonics in chunks,
   rather than materializing and retaining full support cotangent trees for all objectives.
3. Keep the validated math path fixed: accepted replay schedule, reverse payload split
   `{geometry, ntx_support}`, and raw-block VMEC transpose.

## 2026-07-25 07:46 Europe/Lisbon - Compact Initial-Er Ambipolar Root Pullback

Current status: the realtime-geometry reverse benchmark now has a benchmark-local compact
custom rule for the initial-Er ambipolar root boundary contribution. The goal was to keep the
full all-objective reverse behavior, avoid the generic support VJP memory/time blow-up, and not
contaminate the forward solver path.

Validated command:

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
  --reverse-step-bwd-mode reduced_cotangent \
  --skip-realtime-geometry-support-bar-diagnostics \
  --initial-Er-root-ad jax_selected_root
```

Important timing from the validated 16-step run:

```text
runtime build:                              229.139 s
solver components:                           31.411 s
support reverse profile-state vjp:           31.444 s
support reverse initial carry vjp:           20.789 s
support reverse realized-schedule forward:  440.787 s
support reverse segmented cotangent sweep: 1509.917 s
initial-Er root boundary compact pullback:  193.092 s
transport reverse cotangents complete:     2733.853 s
geometry support pullback:                  357.038 s
total elapsed:                             3091.119 s
```

For comparison, the earlier generic root-boundary support VJP took about `2067.233 s` by itself,
and the generic total was about `4906.118 s`. The compact rule therefore cut the root-boundary
piece by about `10.7x` and reduced the total run by roughly `1815 s` (`~37%`) in this benchmark.

Validated reverse geometry gradients for `RBC:1:0` with the compact initial-Er root rule:

```text
softmax_Er:                         reverse=-6.262655e+01
smooth_root_proxy:                  reverse=-3.759502e-04
Er2_volume_average:                 reverse=-2.711843e+02
Er_volume_average:                  reverse=-2.364607e+01
electron_temperature_volume_average reverse=-2.244850e-02
total_pressure_volume_average:      reverse=-7.752044e-02
alpha_power_volume_average:         reverse= 1.207392e-03
```

Corresponding split into direct geometry and NTX-support branches:

```text
softmax_Er:                         geometry=-2.293057e+01  ntx_support=-3.969599e+01
smooth_root_proxy:                  geometry=-1.141335e-04  ntx_support=-2.618166e-04
Er2_volume_average:                 geometry=-2.818629e+02  ntx_support= 1.067862e+01
Er_volume_average:                  geometry=-3.431351e+00  ntx_support=-2.021472e+01
electron_temperature_volume_average geometry=-1.752381e-02  ntx_support=-4.924686e-03
total_pressure_volume_average:      geometry=-7.785517e-02  ntx_support= 3.347248e-04
alpha_power_volume_average:         geometry=-1.260815e-03  ntx_support= 2.468207e-03
```

FD-on reference from the matching realtime-geometry FD run with `--initial-Er-root-ad jax_selected_root`:

```text
softmax_Er:                         fd=-6.262725e+01
smooth_root_proxy:                  fd=-5.576405e-04
Er2_volume_average:                 fd=-2.712344e+02
Er_volume_average:                  fd=-2.364403e+01
electron_temperature_volume_average fd=-2.244865e-02
total_pressure_volume_average:      fd=-7.752101e-02
alpha_power_volume_average:         fd= 1.207392e-03
```

Conclusion: the compact root pullback preserves the full all-objective reverse path and matches
the FD-on derivatives closely for the main objectives. The `smooth_root_proxy` derivative should
be treated cautiously because the proxy definition was changed/restored during debugging; if it
matters, re-run FD and reverse from the same exact code state before using that one as a gate.

Implementation notes:

- The benchmark-local helper is `_compact_initial_er_ntx_support_pullback_leaves` in
  `examples/benchmarks/benchmark_transport_reverse_ad_only.py`.
- It avoids a generic VJP through the full NTX support tree and avoids constructing batched NTX
  dataclasses, which caused static metadata/object-shape failures.
- It scans over radii with `jax.lax.scan`, builds valid unbatched local prepared supports, computes
  a local `jacrev` of the ambipolar particle-flux residual with respect to flattened local support
  leaves plus local `drds`, and contracts all objective residual cotangents at once with
  `jnp.tensordot`.
- It returns flat support-bar leaves in the `NTXExactLijRuntimeSupport` treedef order and keeps
  face-channel/prepared bars zero for this root-boundary contribution.
- A guard checks root support-bar leaf count and shape against the existing support-bar leaves
  before adding them, so tree mismatches fail loudly.
- This is benchmark-side only and should not change the production forward solver behavior.

Validation checks performed after the edit:

```text
python -m py_compile examples\benchmarks\benchmark_transport_reverse_ad_only.py
git diff --check -- examples\benchmarks\benchmark_transport_reverse_ad_only.py
```

Both checks passed.

Recommended next steps:

1. Keep this compact rule as the active path for all-objective realtime-geometry reverse tests with
   `--initial-Er-root-ad jax_selected_root`.
2. If more runtime reduction is needed, focus next on primal/reverse schedule fusion or support-bar
   contraction to selected VMEC harmonics, not on per-objective loopholes.
3. Re-check `smooth_root_proxy` only after deciding the final proxy definition, since that objective
   was intentionally in flux.

Efficiency phase note: with the realtime-geometry reverse math now matching the FD-on benchmark for
the main objectives, the next work can shift from correctness recovery to time/memory efficiency.
The main remaining cost centers are the realized-schedule reverse sweep and the final geometry
support pullback, so future changes should target those without changing the validated derivative
path.

## Optimization API Extraction State

The validated realtime-geometry reverse path is being lifted out of
`examples/benchmarks/benchmark_transport_reverse_ad_only.py` into internal NEOPAX APIs without
changing the derivative math.

Current extracted modules:

- `NEOPAX/_reverse_ad_parameters.py`
  Defines `ProfileParameterSpec`, `VmecBoundaryParameterSpec`, `ReverseADParameterSet`, VMEC
  harmonic discovery helpers, and stable mixed parameter ordering.
- `NEOPAX/_reverse_ad_initial_er.py`
  Owns the compact initial-Er ambipolar root support pullback that replaced the slow generic VJP.
- `NEOPAX/_reverse_ad_transport.py`
  Owns the JAX-native realtime-geometry transport table boundary:
  `RealtimeGeometryTransportReverseTableRequest`,
  `RealtimeGeometryTransportReverseTableResult`,
  `transport_realtime_geometry_reverse_table(...)`,
  `realtime_geometry_payload_pullback_result(...)`,
  `realtime_geometry_transport_reverse_table_from_payload_cotangents(...)`, and the grouped-runner
  contract. It also owns `realtime_geometry_transport_reverse_grouped_inputs(...)`, which builds
  the table context and grouped report runner from a supplied segmented executor, plus
  `realtime_geometry_transport_reverse_support_segment_executor(...)`, which wraps a supplied
  segmented probe callback, and `run_realtime_geometry_support_segment_reverse_table_core(...)`,
  which is the non-printing internal core boundary around that callback.
- `NEOPAX/_reverse_ad_optimization.py`
  Owns VMEX-style least-squares terms and residual/Jacobian assembly:
  `LeastSquaresTerm`, `transport_least_squares_terms(...)`,
  `evaluate_transport_realtime_geometry_least_squares(...)`,
  `build_transport_realtime_geometry_least_squares_runner(...)`, and
  `scalar_loss_and_gradient_from_least_squares(...)`.

Current canonical optimization smoke stack:

```text
benchmark smoke / future optimizer script
  -> build_transport_realtime_geometry_least_squares_runner(...)
  -> RealtimeGeometryTransportReverseTableRequest
  -> evaluate_transport_realtime_geometry_least_squares(...)
  -> transport_realtime_geometry_reverse_table(...)
  -> validated grouped reverse table result
  -> residuals + Jacobian
```

Current benchmark status:

- `--optimization-api-smoke` in `benchmark_transport_reverse_ad_only.py` now calls
  `build_transport_realtime_geometry_least_squares_runner(...)` and then evaluates the returned
  runner on the benchmark-selected terms.
- The benchmark still supplies the temporary segmented probe callback
  `_run_realtime_geometry_support_segment_probe(...)`; internals now wrap it into the grouped
  executor used by optimization smoke paths.
- The benchmark still owns printing and JSON report writing.
- The normal benchmark CLI behavior remains intact.

Validation performed for the extracted API layer:

```text
python -m py_compile NEOPAX\_reverse_ad_optimization.py NEOPAX\_reverse_ad_transport.py \
  examples\benchmarks\benchmark_transport_reverse_ad_only.py NEOPAX\_reverse_ad_parameters.py \
  NEOPAX\_reverse_ad_initial_er.py
git diff --check -- NEOPAX/_reverse_ad_optimization.py NEOPAX/_reverse_ad_transport.py \
  examples/benchmarks/benchmark_transport_reverse_ad_only.py plan_reverse_ad_optimization_lane.md
```

Additional lightweight smokes passed:

- direct table-builder API path,
- grouped-runner API path,
- grouped-builder-through-table-API path,
- least-squares evaluator with profile + VMEC parameter ordering,
- repeated transport objective terms using one table row and multiple residual rows,
- request/term mismatch guard,
- non-transport term rejection for the transport-specific evaluator.

No full GPU benchmark has been rerun after the API extraction-only changes.

Completed extraction step:

The benchmark-specific construction of:

```text
geometry specs from args
ReverseADParameterSet
RealtimeGeometryTransportReverseTableContext
RealtimeGeometryTransportReverseTableRequest
run_grouped_report
```

has been further internalized. The internal transport helper now builds the
`RealtimeGeometryTransportReverseTableContext` and grouped report runner, while the internal
optimization factory builds the `RealtimeGeometryTransportReverseTableRequest` and returns a
least-squares runner. The internal transport core now enforces `return_report=True`, suppresses
probe output, threads table context, and validates the returned JAX-native table result. The
benchmark still owns geometry-spec selection from CLI args and still supplies the temporary
segmented probe callback `_run_realtime_geometry_support_segment_probe(...)` until that final heavy
runner is migrated.

Concrete next steps:

1. Move the actual segmented executor out of the benchmark.
   Context construction, objective='all' grouping, request construction, and least-squares runner
   creation are internal. The non-printing internal core boundary is also in place. The remaining
   benchmark-owned heavy wiring is the probe implementation
   `_run_realtime_geometry_support_segment_probe(...)`.

2. Optimizer-facing runner factory is now present.
   Current shape:

   ```python
   build_transport_realtime_geometry_least_squares_runner(...)
   ```

   returning a callable that evaluates:

   ```text
   least-squares terms -> LeastSquaresEvaluation
   ```

3. Keep benchmark behavior unchanged.
   `--optimization-api-smoke` calls the new runner factory, and normal benchmark CLI/reporting
   remains as-is.

4. Validate with lightweight smokes.
   Check request construction, parameter ordering, objective mismatch guards, and direct/grouped
   runner equivalence.

5. Run one full GPU smoke only after wiring is stable.
   Use `--optimization-api-smoke` with the known realtime geometry command to confirm no numerical
   or path drift.

After that, the next larger migration step is moving the actual segmented grouped executor out of
the benchmark.

Latest extraction step:

The stable setup immediately before the segmented realtime-geometry support reverse sweep has now
been moved into `_reverse_ad_transport.py` as
`prepare_realtime_geometry_support_segment_core_setup(...)` and
`RealtimeGeometrySupportSegmentCoreSetup`. This internal helper owns:

```text
reverse_payload vs NTX-only payload selection
NTX surface backend metadata
profile parameter value slicing
reverse static setup construction
early geometry diagnostics capture
```

The benchmark still owns the actual heavy segmented support pullback function
`_run_realtime_geometry_support_segment_probe(...)`; it now delegates the reusable setup and then
continues with the same local variables and same reverse math as before. No full GPU benchmark has
been rerun after this extraction-only change.

Latest extraction step:

The all-objective grouped support-cotangent orchestration has also been moved into
`_reverse_ad_transport.py` as `realtime_geometry_support_cotangents_from_parameter_vector(...)` and
`RealtimeGeometrySupportCotangentResult`. This helper owns the stable internal result shape for:

```text
objective_values
profile_gradient_matrix
support_bars
support_component_bars_by_name
support_reuse_count / support_rebuild_count
initial-cache pullback flags
```

The helper still calls the same benchmark-supplied grouped reverse callback exactly once and then
performs the same `jax.block_until_ready` synchronization that the benchmark previously performed
inline. The low-level segmented JAX reverse implementation is still in the benchmark; only the
orchestration/result boundary moved inward.

Latest extraction step:

The runtime path for the all-objective segmented support reverse kernel now routes through
`_reverse_ad_transport.py` via
`realtime_geometry_reverse_all_objectives_support_payload_bar_for_parameter_vector(...)`. The
benchmark builds a `RealtimeGeometrySupportReverseDependencies` bundle containing the still
benchmark-owned objective/profile/root/runtime helper callbacks, then immediately calls the
internal kernel. This preserves the same JAX operations and the same single grouped reverse pass,
but moves the active implementation into NEOPAX internals.

Cleanup completed for this step: the old all-objective benchmark body was removed after the
internal routing pass. The benchmark now keeps the single-objective support helper and profile-only
helpers for their existing CLI modes, while the all-objective support kernel is a thin dependency
wrapper around the internal implementation.

## Current Checkpoint - Reverse AD Optimization Extraction

Current code state:

- The active all-objective realtime-geometry support reverse kernel lives in
  `NEOPAX/_reverse_ad_transport.py` as
  `realtime_geometry_reverse_all_objectives_support_payload_bar_for_parameter_vector(...)`.
- The benchmark function
  `_reverse_all_objectives_support_payload_bar_for_parameter_vector(...)` is now a thin wrapper.
  It builds `RealtimeGeometrySupportReverseDependencies` from benchmark-local helper callbacks and
  immediately calls the internal kernel.
- The old duplicated all-objective benchmark body has been removed.
- The benchmark still intentionally owns the profile-only reduced reverse helper and the
  single-objective support probe helper because those support existing CLI/debug modes.
- `realtime_geometry_support_cotangents_from_parameter_vector(...)` owns the grouped
  all-objective support-cotangent result boundary and still calls the wrapper exactly once.
- `realtime_geometry_transport_reverse_table_from_payload_cotangents(...)` owns the support payload
  to VMEC-harmonic table assembly using the validated raw-block transpose path.
- The optimization-facing least-squares runner factory exists in `_reverse_ad_optimization.py`.

Validation status after the cleanup:

```text
python -m py_compile NEOPAX\_reverse_ad_transport.py \
  examples\benchmarks\benchmark_transport_reverse_ad_only.py \
  NEOPAX\_reverse_ad_optimization.py NEOPAX\_reverse_ad_parameters.py \
  NEOPAX\_reverse_ad_initial_er.py

git diff --check -- NEOPAX/_reverse_ad_transport.py \
  examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  realtime_transport_geometry_reverse_current_state.md \
  plan_reverse_ad_optimization_lane.md
```

Both checks passed. No full GPU benchmark has been rerun after the extraction/cleanup-only
changes.

Current known warnings:

- Git reports that `examples/benchmarks/benchmark_transport_reverse_ad_only.py` and this markdown
  file may be converted from LF to CRLF the next time Git touches them. This is line-ending
  bookkeeping, not a Python/JAX behavior change.

Next steps:

1. Move the dependency callbacks out of the benchmark where appropriate.
   Start with low-risk helpers such as objective label handling or state/profile construction only
   if they can be moved without changing numerics.

2. Decide whether the single-objective support probe should remain benchmark-only.
   It is still useful as a diagnostic path, so the safe default is to keep it in the benchmark unless
   optimization needs it.

3. Add a non-benchmark smoke/regression for the internal API.
   Prefer fake/small objects first so it does not require a full GPU transport run.

4. Run one full GPU validation only after the internal API seams are stable.
   Use the known `profiles_plus_realtime_geometry` command with `--optimization-api-smoke` and
   compare against the saved FD/reverse values.

5. After validation, expose a production-facing optimization entry point.
   The likely shape is a VMEX-style least-squares runner accepting transport terms plus VMEC
   geometry terms, grouped internally into reverse-table blocks.

## Current Checkpoint - Initial-Er Root-Only Optimization Smoke

Date: 2026-07-29

Current focus:

- Validate the ambipolar initial-Er root-only optimization smoke using the same compact reverse
  structure as the working realtime-geometry transport reverse path.
- The root-only smoke is selected with `--initial-Er-root-only-optimization-smoke`; it is separate
  from the already validated full reverse transport benchmark path.

Bug found:

- The geometry-active root-only smoke still contained a generic payload VJP through selected-root
  construction. This was not equivalent to the good benchmark behavior and could trigger tracer
  escape/OOM patterns.
- The bad pattern was effectively `jax.vjp(payload_delta -> runtime -> selected root -> objectives)`.

Correction applied:

- The geometry-active root-only smoke now uses the same split root-boundary structure as the
  validated full reverse path:
  - build the pre-root initial state at the baseline runtime,
  - evaluate the selected Er root once,
  - take objective cotangents with respect to rooted state and direct geometry terms,
  - convert `Er` cotangents to ambipolar residual cotangents via `dR/dEr`,
  - use `compact_initial_er_state_pullback(...)` for state residual bars,
  - use `_compact_initial_er_ntx_support_pullback_leaves(...)` for NTX-support residual bars,
  - use `realtime_geometry_transport_reverse_table_from_payload_cotangents(...)` for the raw-block
    VMEC harmonic pullback.
- A follow-up wrapper bug was fixed: `RealtimeGeometryTransportReverseTableResult` has
  `profile_gradient_matrix` and `geometry_gradient_matrix`, not `.jacobian`. The smoke wrapper now
  concatenates `[profile_gradient_matrix | geometry_gradient_matrix]` in optimization parameter
  order.

Validation so far:

```text
python -m py_compile examples\benchmarks\benchmark_transport_reverse_ad_only.py
git diff --check -- examples/benchmarks/benchmark_transport_reverse_ad_only.py
```

Both checks passed.

Latest user run before the wrapper fix:

```text
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --reverse-parameter-mode profiles_plus_realtime_geometry \
  --reverse-geometry-parameter RBC:1:0 \
  --objective all \
  --initial-Er-root-ad jax_selected_root \
  --initial-Er-root-only-optimization-smoke
```

Observed behavior:

- The compact raw-block geometry payload pullback completed.
- The run then failed only at final smoke-result assembly with:
  `AttributeError: 'RealtimeGeometryTransportReverseTableResult' object has no attribute 'jacobian'`.
- That assembly bug is now patched.

Next test to run:

```text
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --reverse-parameter-mode profiles_plus_realtime_geometry \
  --reverse-geometry-parameter RBC:1:0 \
  --objective all \
  --initial-Er-root-ad jax_selected_root \
  --initial-Er-root-only-optimization-smoke
```

Expected result:

- It should get past the previous `.jacobian` AttributeError.
- It should print root-only objective values and Jacobian rows for the active profile and geometry
  optimization parameters.
- This smoke still builds the realtime-geometry runtime, but it should not prepare or run the Radau
  time-evolution solver.

## Optimization Lane Step 1 Checkpoint

Date: 2026-07-29

Current state:

- `_reverse_ad_optimization.py` now has a geometry full-AD objective-table backend:
  `geometry_full_ad_reverse_table(...)` and `geometry_full_ad_reverse_table_backend(...)`.
- `benchmark_geometry_vmec_booz_fd_vs_ad.py` now uses that internal backend for
  `geometry_full_ad_objectives` objective-table reverse mode, then adapts the result back to the
  same benchmark output format as before.
- Geometry objective aliases now accept VMEX-like names such as `qi`, `maxj`, `aspect_ratio`,
  `iota`, `well`, and `mirror`, mapping them to the canonical
  `geometry_full_ad_objectives` table names.
- The geometry backend uses the validated geometry benchmark path:
  `geometry_full_ad_objective_table_pullback_from_param_vector(...)` with
  `final_vmec_pullback_mode="raw_block_transpose"` by default.
- Geometry objectives now require at least one VMEC boundary parameter in the optimization
  parameter set. Profile-only optimization runs should omit geometry terms rather than receiving
  silently-zero geometry residuals.
- The optimization-plan file now records the active benchmark tools for geometry AD,
  frozen-linearized geometry FD, root-only Er optimization smoke, and two-step realtime
  transport optimization smoke.

Validation run locally:

```text
python -m py_compile NEOPAX\_reverse_ad_optimization.py
git diff --check -- NEOPAX/_reverse_ad_optimization.py plan_reverse_ad_optimization_lane.md
```

Both checks passed. The only messages were Git line-ending warnings on Windows.

Geometry AD benchmark command for the optimization seed:

```text
python ./examples/benchmarks/benchmark_geometry_vmec_booz_fd_vs_ad.py \
  --mode geometry_full_ad_objectives \
  --vmec-input ./examples/inputs/input.QI_nfp2_initial \
  --param-specs RBC:1:0,ZBS:1:0 \
  --fd-rel-step 3e-7 \
  --fd-abs-step 1e-10 \
  --ad-backend implicit \
  --fd-lane ad \
  --reverse-derivative-mode objective_table \
  --final-vmec-pullback-mode raw_block_transpose \
  --skip-fd-check
```

Next steps:

- Add combined residual/Jacobian assembly for mixed transport + geometry least-squares terms.
- Add the scalar weighted-loss convenience wrapper `0.5 * r @ r`.
- Wire VMEX-style packed/scaled geometry parameterization into the optimization runner, converting
  scaled optimizer coordinates to physical VMEC boundary deltas before calling the geometry table.
- Add first example scripts:
  geometry-only QI/maxJ/aspect/iota/mirror/softmaxEr optimization from
  `examples/inputs/input.QI_nfp2_initial`, and profile-only Er/root-transition tuning.
- Keep benchmark files as validation clients; do not change their default numerical behavior.

## 16-Step Transport Optimization API Smoke

Date: 2026-07-29

Command:

```text
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
  --reverse-step-bwd-mode reduced_cotangent \
  --skip-realtime-geometry-support-bar-diagnostics \
  --initial-Er-root-ad jax_selected_root \
  --optimization-api-smoke
```

Result:

- Runtime build ready: `222.776 s`.
- Solver components ready: `31.350 s`.
- Optimization API smoke elapsed: `3126.731 s`.
- Device: GPU, `baseline_values_device=cuda:0`.
- Residual count: `7`.
- Parameter count: `5`.

Jacobian rows:

```text
transport:softmax_Er
  dn0=-5.0838282415568408e+00
  dT0=4.1306249928548100e+00
  ddensity_shape_power=-1.1591644624461980e-01
  dtemperature_shape_power=4.2417414653234697e+00
  dvmec:RBC:1:0=-6.2626550584360913e+01

transport:smooth_root_proxy
  dn0=-5.8593750000000000e-03
  dT0=-7.7788585014637937e-05
  ddensity_shape_power=-1.3650089969452495e-07
  dtemperature_shape_power=2.3783611385449557e-03
  dvmec:RBC:1:0=-3.7595015613984747e-04

transport:Er2_volume_average
  dn0=-6.8729867376835294e+00
  dT0=3.6065316917000459e+01
  ddensity_shape_power=3.7772605808640574e+00
  dtemperature_shape_power=-9.9007831215974562e+00
  dvmec:RBC:1:0=-2.7118432937475063e+02

transport:Er_volume_average
  dn0=-2.3765676300577470e+00
  dT0=1.0952038126424903e+00
  ddensity_shape_power=-1.0966510378096389e-01
  dtemperature_shape_power=-3.1844641881012514e-01
  dvmec:RBC:1:0=-2.3646073953442198e+01

transport:electron_temperature_volume_average_keV
  dn0=3.2044199169065646e-03
  dT0=3.5042845972161463e-01
  ddensity_shape_power=-1.7862537077364612e-04
  dtemperature_shape_power=1.5035680183819486e+00
  dvmec:RBC:1:0=-2.2448497948865054e-02

transport:total_pressure_volume_average
  dn0=7.9065460031525578e+00
  dT0=1.8293618203570214e+00
  ddensity_shape_power=2.3978045308837445e-01
  dtemperature_shape_power=7.6008795466190691e+00
  dvmec:RBC:1:0=-7.7520440608481067e-02

transport:alpha_power_volume_average_mw_m3
  dn0=2.7390173486511626e-01
  dT0=8.1435927075356962e-02
  ddensity_shape_power=2.3105926577552259e-03
  dtemperature_shape_power=2.7792303880127184e-01
  dvmec:RBC:1:0=1.2073916077413038e-03
```

Interpretation:

- This confirms the 16-step reverse-AD optimization API smoke reaches the same validated numerical
  regime as the benchmark reverse path.
- The geometry column for `RBC:1:0` matches the previously validated full 16-step values, including
  the ambipolar selected-root AD path.
- The result was written to
  `outputs/autodiff_transport_lagged_ntx/reverse_ad/transport_reverse_ad_only_optimization_api_smoke.json`.

Comparison to the matching 16-step realtime-geometry frozen-linearized FD run:

```text
python ./examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --parameter RBC:1:0 \
  --geometry-fd-lane frozen_linearized \
  --accepted-step-limit 16 \
  --radau-jacobian-reuse-mode legacy \
  --replay-mode accepted \
  --initial-Er-root-ad jax_selected_root
```

| Objective | 16-step FD d/dRBC:1:0 | 16-step reverse optimization API d/dRBC:1:0 | abs diff | rel diff |
| --- | ---: | ---: | ---: | ---: |
| `softmax_Er` | `-6.2627250000000000e+01` | `-6.2626550584360913e+01` | `6.994156e-04` | `1.116791e-05` |
| `smooth_root_proxy` | `-5.5764050000000004e-04` | `-3.7595015613984747e-04` | `1.816903e-04` | `3.258198e-01` |
| `Er2_volume_average` | `-2.7123439999999999e+02` | `-2.7118432937475063e+02` | `5.007063e-02` | `1.846028e-04` |
| `Er_volume_average` | `-2.3644030000000001e+01` | `-2.3646073953442198e+01` | `2.043953e-03` | `8.644691e-05` |
| `electron_temperature_volume_average_keV` | `-2.2448650000000001e-02` | `-2.2448497948865054e-02` | `1.520511e-07` | `6.773286e-06` |
| `total_pressure_volume_average` | `-7.7521010000000001e-02` | `-7.7520440608481067e-02` | `5.693915e-07` | `7.344996e-06` |
| `alpha_power_volume_average_mw_m3` | `1.2073920000000001e-03` | `1.2073916077413038e-03` | `3.922587e-10` | `3.248810e-07` |

Conclusion:

- The reverse optimization API agrees with 16-step FD for the main transport and geometry-sensitive
  objectives.
- `smooth_root_proxy` has a relatively large relative difference because the derivative magnitude is
  very small and the smooth proxy is sensitive to the root/sign-transition construction. Its
  absolute difference remains small.

## Geometry-Objective Frozen VMEX Benchmark Through Internals

Current status:

- `compare_geometry_qi_frozen_linearized_fd.py` now keeps the same frozen-linearized FD/JVP logic
  for geometry objectives, but its production reverse-equivalent line is routed through
  `geometry_full_ad_reverse_table(...)`.
- This means QI/maxJ/aspect/iota/mirror frozen VMEX comparisons now exercise the same
  optimization-facing internal geometry objective table used by future least-squares scripts.
- The low-level same-baseline diagnostics remain in the benchmark as extra checks, but they are no
  longer the only reverse quantity available in that script.

Geometry-objective frozen FD vs internal optimization-table AD command:

```text
python ./examples/benchmarks/compare_geometry_qi_frozen_linearized_fd.py \
  --vmec-input ./examples/inputs/input.QI_nfp2_initial \
  --parameter RBC:1:0 \
  --objective boozer_qi_objective \
  --multigrid \
  --forward-linear-solve-mode raw_block \
  --forward-linear-maxiter 300 \
  --adjoint-maxiter 300
```

Expected new line in the output:

```text
[geometry-qi-linearized-fd] optimization_internal_reverse_table value=... raw_block_transpose_param_grad=... rel_err_internal_reverse_vs_jvp=...
```

This line is the one to compare against `frozen_linearized_fd` / `forward_jvp` when checking the
optimization-facing geometry objective path.
