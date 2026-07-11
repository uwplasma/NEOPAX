# AD Forward Lane Plan

## Goal

Recover a correct and efficient forward-AD benchmark lane for transport that is independent from the reverse lane and is consistent with the restored forward solver / FD baseline behavior.

The desired evolution is:

1. first recover correct derivatives with a **primal adaptive solve + accepted-step replay JVP** strategy
2. then move to the final **fused primal-step + tangent-propagation** strategy

## Guiding Principle

The forward AD lane should follow the forward solver lane, not the reverse lane.

That means:

- the primal baseline must come from the same forward adaptive solver path now being recovered for the solver / FD lanes
- forward AD should not route through reverse replay or generalized accepted-step payload machinery if that widens the graph or changes behavior
- if needed, forward AD should have its own specialized helpers even if that duplicates some logic

## Current Context

Recent forward-lane recovery work established:

- the plain forward solver path and the FD fixed-time lane should both use the same forward-primal NTX treatment
- the FD baseline should use the lighter adaptive schedule trace instead of the payload-heavy rollout
- the fixed-time FD lane should follow the same lagged-response state evolution as the adaptive forward lane

This means the forward AD lane should now be rebuilt on top of this same recovered primal forward map.

## Phase A: Recover Correct AD with Primal Map + Replay

### Objective

Recover correct forward AD derivatives before attempting the more ambitious fused primal+tangent strategy.

### Intended Structure

1. Run the primal adaptive forward solve on the same lane used by the restored forward solver / FD baseline.
2. Extract the realized accepted-step map from the lightweight schedule trace.
3. Run forward-mode differentiation on the realized accepted-step replay lane.
4. Compare those AD derivatives against the frozen-FD benchmark.

### What this phase should use

- primal adaptive solve:
  - same forward solver lane as the current recovered solver / FD baseline
- realized accepted-step trace:
  - only the lightweight accepted-step schedule metadata
- replay/JVP:
  - accepted-step replay only
  - no reverse payload machinery
  - no generalized heavy trace objects

### What this phase should avoid

- payload-heavy rollout traces for baseline collection
- reverse-lane helper reuse when it changes forward behavior
- generalized AD wrappers that are broader than what the forward lane needs

## Phase A Tasks

### Step 1: Baseline recovery

Recover the forward AD benchmark baseline so it uses the same primal adaptive forward lane as the restored FD baseline.

Desired result:

- the baseline run is just the trusted forward-primal solve plus a lightweight schedule trace
- no giant payload trace
- no alternate solver wrapper that differs from the normal forward lane

### Step 2: Accepted-step replay recovery

Make the forward AD replay lane use the realized accepted-step map from the recovered primal baseline.

Desired result:

- replay runs only on the saved accepted-step map
- replay uses the same forward fixed-time-map solver philosophy as the FD lane
- lagged-response reuse/rebuild treatment is consistent with the forward solver

### Step 3: JVP recovery

Recover the JVP application on top of the realized accepted-step replay.

Desired result:

- derivatives are finite
- derivatives are numerically plausible
- derivatives can be compared against the FD lane with the same accepted-step cutoff

### Step 4: Validation against FD

Compare the recovered forward AD derivatives against the FD-only benchmark.

Desired result:

- pressure / temperature / alpha metrics match well first
- `Er`-related metrics are then brought into agreement by further replay-lane alignment if needed

## Phase B: Move to Fused Primal + Tangent Propagation

### Objective

Replace the two-phase primal-baseline + replay-JVP strategy with a single forward lane that propagates primal and tangent together during the adaptive solve.

### Intended Structure

- primal step and tangent step advance together
- accepted-step logic is handled in the forward lane itself
- no separate replay extraction pass is required for the main benchmark

### Important constraint

Do not attempt this until Phase A has recovered correct derivatives.

The fused strategy should be an optimization/refactor of a correct forward AD lane, not a replacement for the initial debugging path.

## Recommended Order

1. recover primal baseline on the normal forward solver lane
2. recover accepted-step replay lane
3. recover correct forward AD derivatives with replay JVP
4. benchmark against FD
5. only then implement fused primal+tangent propagation

## Success Criteria

### Phase A success

- forward AD baseline uses the same primal forward lane as solver / FD
- replay uses only the realized accepted-step map
- AD derivatives are finite and numerically consistent with FD

### Phase B success

- fused forward AD reproduces the Phase A derivative values
- fused lane is cheaper than the two-phase replay strategy
- forward lane remains independent of reverse lane machinery

## Next Recommended Action

Start with Phase A, Step 1:

- recover the forward AD baseline so it uses the same lightweight adaptive primal lane as the restored FD benchmark baseline

## 2026-06-22 Status Update: Forward AD Recovered

The scalar forward-AD lane has been recovered for the `T0` benchmark case.

### What was fixed

The forward-AD custom JVP had regressed into a payload-heavy path:

- `_radau_adaptive_final_y_realized_schedule_jvp(...)`
- was calling `_radau_adaptive_final_state_rollout(...)`
- which materialized large per-attempt payloads, including state snapshots,
  stage/carry data, lagged-cache data, and LU/cache fields

This produced the large HLO/input-output memory signature and OOM even for
short accepted-step prefix tests.

The custom-JVP primal schedule capture now uses the compact schedule rollout:

- `_radau_adaptive_schedule_rollout(...)`

This restored the lightweight realized-schedule forward-AD behavior:

- primal adaptive solve is performed inside the custom JVP
- the realized schedule/control trace is treated as nondifferentiable
- tangent propagation replays the realized accepted-step map
- no external baseline run is required for forward AD
- no reverse-lane payload machinery is used

### Validation command

```bash
python ./examples/benchmarks/benchmark_transport_forward_ad_only.py \
  --ntx-exact-derivative-mode direct \
  --parameter T0 \
  --radau-jacobian-reuse-mode legacy
```

### Recovered forward-AD values

```text
softmax_Er                         -2.160399e+01
smooth_root_proxy                   2.070900e-05
Er2_volume_average                 -2.765750e+01
Er_volume_average                   2.291385e+00
electron_temperature_volume_average 3.571291e-01
total_pressure_volume_average       1.835267e+00
alpha_power_volume_average          7.221955e-02
```

These match the old trusted benchmark regime and agree closely with the
recovered frozen-FD accepted-time benchmark.

### Current FD reference state

The recovered FD reference uses:

```bash
python ./examples/benchmarks/benchmark_transport_frozen_fd_only.py \
  --ntx-exact-derivative-mode direct \
  --parameter T0 \
  --fd-rel-step 3e-8 \
  --fd-abs-step 1e-10 \
  --replay-mode accepted \
  --fixed-time-lane direct \
  --radau-jacobian-reuse-mode legacy
```

Important interpretation:

- the FD lane runs a separate baseline adaptive solve to get the accepted
  `dt` list
- `fd-` and `fd+` then recompute physics from their own perturbed initial
  carries on that accepted time grid
- baseline carry history is not copied into the FD endpoints
- the direct fixed-time lane may report a few Newton nonconvergence flags, but
  it reaches `t_final` with finite endpoint objectives and recovers the old
  FD scale

### Next Implementation Step: Fused Forward AD

The current recovered forward AD is correct but still two-phase inside the
custom JVP:

1. primal adaptive solve records the compact realized schedule
2. accepted-step replay propagates the tangent

The next optimization is a fused forward-AD lane:

- propagate primal and tangent together during the adaptive solve
- controller decisions remain primal-only
- tangent does not affect accept/reject or next-`dt` decisions
- the fused result must reproduce the recovered forward-AD values above
- only after matching the recovered values should the replay-based JVP be
  retired or downgraded to a diagnostic/reference path

### Next Milestone After Fused Forward AD: Reverse AD Recovery

After fused forward AD is validated, recover the reverse AD lane against the
forward-AD reference values above.

Reverse AD should be treated as a separate lane:

- no changes to the recovered forward solver / FD / forward-AD lanes unless
  explicitly required and reviewed
- reverse should match the accepted-step AD contract used by forward AD
- validation order should be:
  1. reverse vs recovered forward AD on short prefixes
  2. reverse vs recovered forward AD on full `T0`
  3. reverse vs recovered FD only as a secondary sanity check

The forward-AD row above is now the primary numerical reference for the next
reverse-AD recovery work.

## 2026-06-22 Status Update: Step-Fused Forward AD Timing

The experimental step-compute-fused forward AD mode was implemented and tested
for the full `T0` case:

```bash
python ./examples/benchmarks/benchmark_transport_forward_ad_only.py \
  --ntx-exact-derivative-mode direct \
  --parameter T0 \
  --radau-jacobian-reuse-mode legacy \
  --forward-ad-fusion-mode step
```

Reference replay timing:

```text
forward_ad_fusion_mode=replay forward_ad_total_s=1.098723e+03
```

Step-fused timing:

```text
forward_ad_fusion_mode=step forward_ad_total_s=8.988777e+02
```

This is about a `1.22x` speedup, or roughly an `18%` wall-time reduction.

The step-fused gradients match the recovered replay/reference values closely:

```text
softmax_Er                         -2.160408e+01
smooth_root_proxy                   2.070901e-05
Er2_volume_average                 -2.765769e+01
Er_volume_average                   2.291391e+00
electron_temperature_volume_average 3.571291e-01
total_pressure_volume_average       1.835267e+00
alpha_power_volume_average          7.221950e-02
```

The reference replay values were:

```text
softmax_Er                         -2.160399e+01
smooth_root_proxy                   2.070901e-05
Er2_volume_average                 -2.765749e+01
Er_volume_average                   2.291385e+00
electron_temperature_volume_average 3.571291e-01
total_pressure_volume_average       1.835267e+00
alpha_power_volume_average          7.221955e-02
```

### Possible Further Efficiency Improvements

The current step-fused mode is correct-looking and meaningfully faster, but
there are likely still gains available inside the accepted-step tangent
assembly:

- avoid extra lagged/RHS work inside
  `_radau_accepted_step_attempt_tangent_from_primal(...)`
- reuse quantities already available from the primal accepted-step result where
  mathematically valid
- specialize the common approximate-tangent / lagged-response-reuse path
- reduce compile size from carrying all exact/rebuild branches through every
  accepted step
- add lightweight counters for tangent-path branch usage before optimizing
  blindly
- consider fusing metric evaluation only after the full-state tangent path is
  validated and stable

The most likely next optimization target is
`_radau_accepted_step_attempt_tangent_from_primal(...)`, specifically checking
which lagged response, RHS, and stage-evaluation quantities are recomputed even
though equivalent primal data already exists.

## 2026-07-08 Next Plan: Add Realtime-Geometry Parameters to Forward AD

Goal:

- Extend the forward-AD transport benchmark so realtime VMEC/Boozer geometry
  parameters can be differentiated in the full transport solve.
- Keep the existing profile-parameter lane unchanged.
- Reuse the geometry helper functions already tested by the pure geometry gate;
  do not duplicate VMEC/Boozer derivative logic inside the transport benchmark.

Current relevant code paths:

- Pure geometry gate:
  `examples/benchmarks/benchmark_geometry_vmec_booz_fd_vs_ad.py`
- Geometry AD helpers:
  `NEOPAX/_geometry_autodiff.py`
  - `build_geometry_autodiff_context(...)`
  - `solve_geometry_state_ad(...)`
  - `build_runtime_context_for_geometry_param(...)`
- Runtime entry point:
  `NEOPAX/_orchestrator.py::build_runtime_context(config)`
- Realtime geometry benchmark config:
  `examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml`

Implementation plan:

1. Keep the profile-only forward-AD path unchanged.

   Existing commands such as:

   ```bash
   python ./examples/benchmarks/benchmark_transport_forward_ad_only.py \
     --ntx-exact-derivative-mode direct \
     --parameter n0 \
     --accepted-step-limit 2 \
     --radau-jacobian-reuse-mode legacy
   ```

   must continue to use the current profile-only machinery and reproduce the
   current values.

2. Add an opt-in geometry parameter syntax for forward AD.

   First-pass syntax:

   ```text
   --parameter vmec:RBC:1:0
   ```

   This keeps forward AD naturally directional: one `jax.jvp` direction per
   run, exactly like the current `--parameter n0` / `--parameter T0` behavior.

3. Require realtime geometry for `vmec:*` parameters.

   Accept geometry parameters only when the config uses one of:

   - `vmec_jax_booz_xform_jax`
   - `vmec_runtime`
   - `vmec_realtime`

   Refuse ordinary frozen geometry configs with a clear error. A frozen VMEC
   geometry file cannot produce derivatives with respect to VMEC boundary
   coefficients inside the transport solve.

4. Build the geometry AD context once outside the differentiated objective.

   Use the same helper path as the pure geometry gate:

   ```text
   build_geometry_autodiff_context(...)
   ```

   The transport benchmark should not directly call low-level `vmec_jax` or
   `booz_xform_jax` derivative internals.

5. Inside the forward-AD objective, branch only on the parameter type.

   Profile parameter path:

   ```text
   p -> current profile-state builder -> current forward AD transport objective
   ```

   Geometry parameter path:

   ```text
   delta
     -> build_runtime_context_for_geometry_param(..., lane="ad")
     -> use configured profile inputs on the new runtime geometry
     -> current forward AD transport objective
   ```

   This composes the tested geometry AD lane with the tested transport forward
   AD lane, instead of mixing reverse-AD helper code into forward AD.

6. Use the same geometry helper for FD validation, but with the forward lane.

   For geometry finite-difference checks:

   ```text
   delta +/- h
     -> build_runtime_context_for_geometry_param(..., lane="forward")
     -> transport objective
   ```

   This mirrors the pure geometry gate split: AD geometry for JVP, forward
   geometry solves for FD perturbations.

7. First test command after implementation:

   ```bash
   python ./examples/benchmarks/benchmark_transport_forward_ad_only.py \
     --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
     --ntx-exact-derivative-mode direct \
     --parameter vmec:RBC:1:0 \
     --accepted-step-limit 2 \
     --radau-jacobian-reuse-mode legacy
   ```

Success criteria:

- `--parameter n0`, `--parameter T0`, and the other profile parameters remain
  unchanged.
- `--parameter vmec:RBC:1:0` runs only with realtime geometry configs.
- The geometry tangent is produced through
  `build_runtime_context_for_geometry_param(..., lane="ad")`.
- A matching FD check uses
  `build_runtime_context_for_geometry_param(..., lane="forward")`.
- The solver-level geometry derivatives are nonzero for geometry-dependent
  objectives and can be compared against a small FD perturbation before adding
  more geometry parameters.

Do not do in the first pass:

- Do not edit `vmec_jax` or `booz_xform_jax`.
- Do not make forward AD depend on reverse-AD helper code.
- Do not duplicate pure-geometry derivative logic inside the transport
  benchmark.
- Do not silently treat frozen geometry configs as differentiable geometry
  configs.
- Do not add multiple geometry parameters until the single-parameter path is
  validated.

## 2026-07-08 Forward AD 16-Step Profile References

These values were produced with the profile-only forward AD benchmark at
`--accepted-step-limit 16`, `--ntx-exact-derivative-mode direct`, and
`--radau-jacobian-reuse-mode legacy`.

Commands:

```bash
python ./examples/benchmarks/benchmark_transport_forward_ad_only.py \
  --ntx-exact-derivative-mode direct \
  --parameter n0 \
  --accepted-step-limit 16 \
  --radau-jacobian-reuse-mode legacy

python ./examples/benchmarks/benchmark_transport_forward_ad_only.py \
  --ntx-exact-derivative-mode direct \
  --parameter T0 \
  --accepted-step-limit 16 \
  --radau-jacobian-reuse-mode legacy

python ./examples/benchmarks/benchmark_transport_forward_ad_only.py \
  --ntx-exact-derivative-mode direct \
  --parameter density_shape_power \
  --accepted-step-limit 16 \
  --radau-jacobian-reuse-mode legacy

python ./examples/benchmarks/benchmark_transport_forward_ad_only.py \
  --ntx-exact-derivative-mode direct \
  --parameter temperature_shape_power \
  --accepted-step-limit 16 \
  --radau-jacobian-reuse-mode legacy
```

Forward AD runtimes:

| Parameter | Baseline value | `forward_ad_total_s` |
|---|---:|---:|
| `n0` | `4.210000e+00` | `1.304289e+03` |
| `T0` | `1.780000e+01` | `1.305469e+03` |
| `density_shape_power` | `1.000000e+01` | `1.312365e+03` |
| `temperature_shape_power` | `2.000000e+00` | `1.305829e+03` |

Forward AD derivatives:

| Objective | `d/dn0` | `d/dT0` | `d/ddensity_shape_power` | `d/dtemperature_shape_power` |
|---|---:|---:|---:|---:|
| `softmax_Er` | `-3.763715e+00` | `3.057234e+00` | `-8.528369e-02` | `3.220282e+00` |
| `smooth_root_proxy` | `4.214696e-04` | `-1.730687e-04` | `-3.686619e-06` | `1.672156e-02` |
| `Er2_volume_average` | `-4.377065e+00` | `2.463177e+01` | `1.339075e+00` | `-3.204200e+01` |
| `Er_volume_average` | `-1.738659e+00` | `8.214770e-01` | `-4.387205e-02` | `-4.177935e-01` |
| `electron_temperature_volume_average_keV` | `3.187853e-03` | `3.504164e-01` | `-1.821264e-04` | `1.503562e+00` |
| `total_pressure_volume_average` | `7.906522e+00` | `1.829334e+00` | `2.397771e-01` | `7.600936e+00` |
| `alpha_power_volume_average_mw_m3` | `2.739043e-01` | `8.142878e-02` | `2.311157e-03` | `2.779267e-01` |

These are the current forward-AD references for comparing against 16-step
profile-only reverse AD.

## 2026-07-09: Realtime VMEC/Boozer Geometry Injection State

Goal: the realtime VMEC/JAX plus Booz-xform/JAX geometry path should mimic the
frozen `wout/boozermn` geometry path, with the same solver/profile settings
except for the geometry block.

Current comparison command:

```bash
python ./examples/benchmarks/compare_realtime_vmec_geometry_injection.py \
  --json-output outputs/realtime_geometry_injection_compare_after_b0_interior_halfgrid.json
```

Important patches/state so far:

- `_build_neopax_geometry_from_state()` uses public
  `booz_xform_from_inputs(... )["gmnc_b"]` instead of the private NTX
  `_booz_xform_gmnc_from_inputs()` helper, because the private helper no longer
  matches the current `booz_xform_jax._surface_transform()` signature.
- Axis extension for `iota`, `I_value`, and `G_value` now uses the first
  surface sample instead of zero-axis padding.
- Current best local bridge variant uses transport interior half-grid support:
  `sample_rho = rho_grid_half[1:-1]`.
- Current local bridge uses `B0 = b0_interp(r_grid)`.
- Current `B_10` handling uses raw `B10/B00` with axis set to zero.

Latest frozen-vs-realtime geometry comparison, using the current interior
half-grid variant:

| Quantity | `rel_l2` | `max_abs` | Note |
|---|---:|---:|---|
| `curvature` | `2.393569e-01` | `3.469863e-01` | Still large |
| `B0` | `2.132623e-02` | `1.038158e+00` | Last point differs strongly |
| `Bsqav` | `3.721296e-02` | `1.838360e-01` | Still too large |
| `B_10` | `1.017722e-02` | `6.629283e-04` | Much closer than before |
| `epsilon_t` | `1.327986e-02` | n/a | Still non-negligible |
| `R0` | `5.403005e-03` | n/a | Small but visible |
| `Vprime`, `Vprime_half`, `r_grid`, `a_b` | about `2.7e-03` | n/a | Similar small mismatch |

NTX center/face channels are mostly close:

- `iota` is about `1.8e-04` relative.
- `fac_reference_to_sfincs_11` is about `1.4e-05` relative.
- `a_b` is about `2.7e-03` relative.
- `r00` is about `5.25e-03` relative.
- `boozer_i` shows huge relative error only because the absolute values are
  around `1e-15`; this is not currently suspected to be the main error source.
- `fac_dkes_to_d33star` still has a shape mismatch (`[1]` versus `[51]` or
  `[52]`), but this is not currently suspected to drive the electric-field
  mismatch.
- `drds`/`dr_tildeds` have axis-related `inf`/`nan` in relative metrics, but
  finite relative values are around `2.7e-03`.

Current interpretation:

- The realtime bridge is not solved yet.
- Do not continue blind interpolation tweaks until the raw Boozer diagnostic is
  run.
- The strongest current hypothesis is radial support mismatch: frozen
  `VmecBoozer` uses VMEC/Boozer half-mesh support from the file, while the
  realtime bridge currently uses a transport half-grid support. This can make
  the first Boozer surface differ substantially, for example frozen support
  near `rho ~= sqrt(s_half)` while realtime transport half-grid starts much
  closer to the axis.
- If raw Boozer coefficients differ on matched support, the remaining issue is
  in Booz-xform/JAX inputs, resolution, or normalization rather than NEOPAX
  interpolation.

New opt-in raw Boozer diagnostic:

```bash
python ./examples/benchmarks/compare_realtime_vmec_geometry_injection.py \
  --json-output outputs/realtime_geometry_injection_compare_raw_boozer.json \
  --raw-boozer-diagnostics
```

The diagnostic compares:

- `raw_b00`
- `raw_gmn00`
- `raw_buco`
- `raw_bvco`
- `raw_iota`
- `raw_b10`
- `raw_b10_over_b00`
- `boozer_support_rho`

Decision rule for the next session:

- If `boozer_support_rho` differs strongly, fix the realtime Boozer support
  mapping so it mirrors the frozen VMEC/Boozer support without building too
  many surfaces and OOMing.
- If `boozer_support_rho` matches but raw Boozer coefficients differ, inspect
  Booz-xform/JAX inputs, resolution, normalization, and VMEC/JAX state fields.
- If raw coefficients match but NEOPAX geometry fields differ, focus only on
  `_build_neopax_geometry_from_state()` interpolation/formulas.

## 2026-07-11: Forward AD `exact` Fusion Next Step

Current command under discussion:

```bash
python ./examples/benchmarks/benchmark_transport_forward_ad_only.py \
  --ntx-exact-derivative-mode direct \
  --ntx-exact-derivative-field-pullback-mode compact_vjp \
  --ntx-exact-derivative-pullback-algebra scalar_contract_lowdot_ntx \
  --parameter n0 \
  --accepted-step-limit 16 \
  --radau-jacobian-reuse-mode legacy \
  --forward-ad-fusion-mode exact
```

Latest `n0` profile result with `--forward-ad-fusion-mode exact`:

| Objective | Forward AD `exact` tangent |
|---|---:|
| `softmax_Er` | `-3.770213e+00` |
| `smooth_root_proxy` | `4.274729e-04` |
| `Er2_volume_average` | `-4.419701e+00` |
| `Er_volume_average` | `-1.741342e+00` |
| `electron_temperature_volume_average_keV` | `3.191177e-03` |
| `total_pressure_volume_average` | `7.906533e+00` |
| `alpha_power_volume_average_mw_m3` | `2.739037e-01` |

Expected `n0` reverse/FD realized-schedule target is approximately:

| Objective | Reverse/FD target |
|---|---:|
| `softmax_Er` | `-3.75963e+00` |
| `smooth_root_proxy` | `4.7e-04` order, FD-step sensitive |
| `Er2_volume_average` | `-4.348e+00` to `-4.349e+00` |
| `Er_volume_average` | `-1.7369e+00` |
| `electron_temperature_volume_average_keV` | `3.1849e-03` |
| `total_pressure_volume_average` | `7.90651e+00` |
| `alpha_power_volume_average_mw_m3` | `2.73905e-01` |

Diagnosis:

- This is not an NTX pullback-algebra issue.
- Reverse AD and frozen FD remain the reference for the profile-only
  realized-schedule target.
- `--forward-ad-fusion-mode exact` currently dispatches through
  `derivative_mode="jvp_exact"`.
- That reaches `_radau_adaptive_final_y_realized_schedule_exact_jvp(...)`.
- Its JVP rule calls
  `_radau_adaptive_final_y_realized_schedule_fused_jvp(...,
  raw_attempt_jvp=True)`.
- Therefore the current `exact` mode is a **raw-attempt JVP** diagnostic, not
  the accepted-map fused JVP needed to match reverse/FD.
- The needed fused lane must differentiate the same realized accepted-step map
  as reverse/FD, including:
  - accepted-only tangent propagation,
  - zero tangent contribution from rejected attempts,
  - the same projected accepted-step state update,
  - the same forward-only treatment of controller/cache/Jacobian/LU fields,
  - the same final carry/state used by the trusted realized-schedule primal.

Next implementation target:

1. Do not change reverse AD.
2. Do not change the recovered `replay` mode reference.
3. Add a new explicit forward mode, for example
   `--forward-ad-fusion-mode accepted_exact`, instead of overloading the current
   raw-attempt `exact` diagnostic.
4. Implement `accepted_exact` by making the JVP tangent use the same accepted
   replay slot/map as reverse's realized accepted-step replay, not
   `raw_attempt_jvp=True`.
5. Validate only `n0`, `accepted-step-limit 16`, `radau_jacobian_reuse_mode=legacy`
   first.
6. Only after `n0` matches reverse/FD should the other profile parameters be
   rerun.

Acceptance criterion for the next pass:

- `accepted_exact` objective values must match the trusted reverse/FD
  realized-schedule primal for the 16-step case.
- `accepted_exact` tangents for `n0` should move from the current raw-attempt
  values toward the reverse/FD targets above, especially `softmax_Er`,
  `Er2_volume_average`, and `Er_volume_average`.
- If pressure and alpha stay close but the electric-field objectives remain
  off, inspect the accepted-step projection and lagged-response/cache tangent
  threading inside the fused accepted-map replay.
