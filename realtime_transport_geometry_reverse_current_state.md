# Realtime Transport Geometry Reverse AD Current State

Date: 2026-07-22

## Goal

Extend the already-benchmarked profile reverse AD lane to realtime VMEC geometry, without contaminating:

- the forward transport solver,
- the frozen geometry/profile reverse benchmark,
- the realtime forward solver geometry construction.

The realtime geometry reverse lane should use the same primal geometry construction as the realtime forward solver and should pull transport-objective cotangents back to VMEC boundary harmonics.

## Current Baseline Commands

Reverse AD realtime geometry/profile benchmark:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --reverse-parameter-mode profiles_plus_realtime_geometry \
  --reverse-geometry-parameter RBC:1:0 \
  --realtime-geometry-gradient-path reverse_payload \
  --ntx-exact-derivative-mode direct \
  --ntx-exact-derivative-field-pullback-mode generic_jvp \
  --objective all \
  --accepted-step-limit 2 \
  --radau-jacobian-reuse-mode legacy \
  --timing-mode jit-warm \
  --reverse-segment-length 1 \
  --reverse-stage-adjoint-solve-mode bicgstab \
  --reverse-rhs-transpose-mode explicit_ntx_interpolated \
  --reverse-step-bwd-mode reduced_cotangent
```

Realtime geometry forward FD comparison:

```bash
python ./examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --parameter RBC:1:0 \
  --geometry-fd-lane frozen_linearized \
  --accepted-step-limit 2 \
  --radau-jacobian-reuse-mode legacy \
  --replay-mode accepted
```

Profile FD comparison:

```bash
python ./examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --parameter n0 \
  --accepted-step-limit 2 \
  --radau-jacobian-reuse-mode legacy \
  --replay-mode accepted
```

## Fixed / Verified

- Realtime geometry quantities are finite in reverse:
  - `a_b = 1.1778251713416479`
  - `R0 = 12.088114677251474`
  - `integrated_volume = 331.0187969899648`
  - `r_grid`, `Vprime`, `Vprime_half`, `overVprime` all finite.

- NTX surface backend for the realtime TOML is `vmec`.

- The reverse profile gradients match realtime FD profile gradients for `n0` on the accepted-step replay benchmark.

- The old NaN issue in VMEC traceable NTX support was fixed by making the derived `bmag` construction safe near non-positive roundoff:
  - file: `NEOPAX/_geometry_autodiff.py`
  - branch: `_vmec_jax_wout_surface_with_frozen_sampling`

- The reverse geometry payload now prints branch and component splits:
  - `geometry_branch`
  - `ntx_support_branch`
  - `objective_explicit`
  - `transport_rhs`
  - `initial_cache`
  - `initial_profile`

- The reverse all-harmonic infrastructure has been started:
  - `--reverse-geometry-parameter all`
  - `--reverse-geometry-families`
  - `--reverse-geometry-include-zero-harmonics`
  - print limiting via top-k.

## Current Reverse vs FD Numbers

FD geometry benchmark for `RBC:1:0`, frozen-linearized geometry lane, accepted replay:

```text
softmax_Er                         -4.184389e+00
smooth_root_proxy                   1.875871e-12
Er2_volume_average                  1.648737e+01
Er_volume_average                  -3.687997e+00
electron_temperature_volume_average -2.088060e-02
total_pressure_volume_average      -1.168713e-01
alpha_power_volume_average         -2.485351e-03
```

Reverse AD `RBC:1:0` from latest component-split run:

```text
softmax_Er                         -4.180080e+00
smooth_root_proxy                   2.075771e-12
Er2_volume_average                  1.863731e+01
Er_volume_average                  -3.708129e+00
electron_temperature_volume_average -1.381016e-02
total_pressure_volume_average      -7.726593e-02
alpha_power_volume_average         -1.759807e-03
```

`softmax_Er` and `Er_volume_average` are close. `Er2`, `Te`, pressure, and alpha still differ.

## Important Diagnostic Result

The remaining mismatch is not mainly from NTX support for the non-Er objectives. The latest split shows:

```text
electron_temperature:
  objective_explicit = -1.295823e-02
  transport_rhs      = -5.271512e-04
  initial_cache      = -3.247714e-04
  initial_profile    = ~0

total_pressure:
  objective_explicit = -7.766514e-02
  transport_rhs      =  4.557137e-04
  initial_cache      = -5.650149e-05
  initial_profile    = ~0

alpha_power:
  objective_explicit = -1.927415e-03
  transport_rhs      =  6.560537e-05
  initial_cache      =  1.020021e-04
  initial_profile    = ~0
```

For these objectives, the discrepancy is concentrated in the direct explicit geometry objective path, especially the volume-average use of `Vprime` and `r_grid`.

The added initial analytical profile geometry pullback is effectively zero for `RBC:1:0` in this setup. This is consistent with the profile builder using normalized radius `x = r_grid / r_grid[-1]`, so pure/minor-radius-like changes mostly cancel.

## Latest Added Diagnostic

`examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py` now prints a fixed-final-state explicit geometry FD diagnostic for realtime geometry parameters:

```text
fixed-final-state explicit geometry finite-difference gradients
```

This evaluates:

```text
fixed final transport state + perturbed realtime geometry objective only
```

It should be compared directly against the reverse component:

```text
objective_explicit
```

If this FD diagnostic does not match `objective_explicit`, the bug is specifically in the explicit geometry-objective pullback for volume averages / `Vprime` / `r_grid`.

## Ambipolar Root-Finder AD Status

The transport RHS ambipolarity path is already differentiated:

- `Gamma` enters density via `div Gamma`.
- `Q` enters pressure via `div Q`.
- `Gamma` enters Er via ambipolar charge flux.
- NTX support payload cotangents receive the corresponding `Gamma/Q/Upar` bars.

The missing piece for full realtime geometry AD is initial Er root-finder differentiation:

```text
VMEC harmonic -> ambipolar root finder -> initial Er -> rollout -> objective
```

Current FD and reverse geometry comparison intentionally freeze initial Er for the frozen-linearized geometry FD lane, so this branch is not active in the current comparison.

When enabling this, do not differentiate through scan/root selection. Use a frozen selected-root implicit rule:

```text
G_i(Er_i, state, geometry, support) = sum_s q_s Gamma_s(i, Er_i) = 0
dEr_i/dp = - (dG_i/dp) / (dG_i/dEr_i)
```

Reverse form:

```text
G_bar_i = - Er_bar_i / (dG_i/dEr_i)
```

Then pull `G_bar_i` through the local ambipolar residual into geometry/support/profile quantities.

## Next Step

Run the updated FD geometry benchmark and inspect the new explicit-geometry-only FD printout:

```bash
python ./examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --parameter RBC:1:0 \
  --geometry-fd-lane frozen_linearized \
  --accepted-step-limit 2 \
  --radau-jacobian-reuse-mode legacy \
  --replay-mode accepted
```

Compare:

```text
fixed-final-state explicit geometry FD
```

against reverse:

```text
objective_explicit
```

If they match, the remaining full FD mismatch is in replayed transport geometry coupling. If they do not match, fix the explicit geometry objective pullback first.
