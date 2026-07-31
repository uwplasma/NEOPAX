# Optimization AD vs FD References

Reference parameter:

- VMEC harmonic: `RBC:1:0`

This file collects the saved optimization-facing AD rows and the available FD
references. Root-only ambipolarity and full time-evolution transport are kept
separate because they are different maps.

## Geometry Objectives: Shared-Payload AD vs Frozen FD

Configuration:

- VMEC input: `examples/inputs/input.QI_nfp2_newNT_opt_hires_true`
- FD benchmark: `examples/benchmarks/compare_geometry_qi_frozen_linearized_fd.py`
- Geometry lane: frozen-linearized VMEC FD
- AD target: implicit forward JVP / raw-block transpose reverse

| Objective | Shared-payload AD value | Shared-payload AD `d/dRBC:1:0` | Frozen FD `d/dRBC:1:0` | JVP target `d/dRBC:1:0` | `rel(shared, FD)` | `rel(shared, JVP)` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `geometry:boozer_qi_objective` | `3.2109136309506734e-03` | `8.9989229070800647e-02` | `8.9996041589633161e-02` | `8.9989229074526111e-02` | `7.569798e-05` | `4.139899e-11` |
| `geometry:boozer_maxj_objective` | `1.3389843913753852e-01` | `-1.1593167987379616e+00` | `-1.1591364898936340e+00` | `-1.1593167988198267e+00` | `1.555366e-04` | `7.061503e-11` |

Conclusion:

- The shared-payload optimization-facing geometry rows match the AD/JVP target
  to about `1e-10` relative or better.
- The FD mismatch is small and consistent with the frozen-linearized FD
  tolerance/step.

## Geometry Objectives Without Saved FD Rows

These rows were printed by the shared root-only payload smoke, but standalone FD
references are not currently saved in the local docs/outputs.

| Objective | Shared-payload AD value | Shared-payload AD `d/dRBC:1:0` | FD status |
| --- | ---: | ---: | --- |
| `geometry:vmec_aspect_ratio` | `1.0015330918957178e+01` | `-5.4006784187006147e+00` | not saved yet |
| `geometry:vmec_iota_mean` | `-5.9365259966101458e-01` | `2.4405140609263865e-01` | not saved yet |
| `geometry:vmec_magnetic_well` | `-2.7476128749679612e-02` | `-1.1090116065531674e-02` | not saved yet |
| `geometry:vmec_mirror_ratio` | `2.1100247521308457e-01` | `-5.9979060848576748e-01` | not saved yet |

## Root-Only Ambipolarity: Shared-Payload AD Rows

Configuration:

- Config: `examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml`
- Path: initial-Er root only, no Radau time evolution
- Shared-payload smoke flag: `--initial-Er-root-shared-payload-compare-smoke`

Command:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --reverse-parameter-mode profiles_plus_realtime_geometry \
  --reverse-geometry-parameter RBC:1:0 \
  --objective all \
  --initial-Er-root-ad jax_selected_root \
  --initial-Er-root-only-optimization-smoke \
  --initial-Er-root-shared-payload-compare-smoke
```

| Objective | Shared-payload AD value | Shared-payload AD `d/dRBC:1:0` | Root-only FD status |
| --- | ---: | ---: | --- |
| `transport:softmax_Er` | `2.0479476664720302e+01` | `-5.1293330714789889e+01` | not saved yet |
| `transport:smooth_root_proxy` | `8.0591612259350128e-11` | `2.2505242252037452e-09` | not saved yet |
| `transport:Er_transition_left` | `1.7657228878480385e+01` | `-2.0079770095990593e+01` | not saved yet |
| `transport:Er_transition_right` | `1.8321801837429653e+01` | `-2.2278454387928971e+01` | not saved yet |
| `transport:Er2_volume_average` | `2.5947838715347029e+02` | `-1.8476397875085149e+02` | not saved yet |
| `transport:Er_volume_average` | `-3.5568787373760746e+00` | `-2.0622727790532899e+01` | not saved yet |

Note:

- I did not find saved root-only FD values in the local repo docs/outputs.
- Do not compare these root-only rows against the 16-step time-evolution FD
  table below; they are different objective maps.

## Full Transport: Known 16-Step FD vs Reverse AD

Configuration:

- Config: `examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml`
- Path: full realtime geometry transport with 16 accepted Radau steps
- Initial-Er root AD: `jax_selected_root`
- Geometry FD lane: frozen-linearized

FD command:

```bash
python ./examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --parameter RBC:1:0 \
  --geometry-fd-lane frozen_linearized \
  --accepted-step-limit 16 \
  --radau-jacobian-reuse-mode legacy \
  --replay-mode accepted \
  --initial-Er-root-ad jax_selected_root
```

Reverse AD command:

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
  --initial-Er-root-ad jax_selected_root \
  --optimization-api-smoke
```

| Objective | 16-step FD `d/dRBC:1:0` | 16-step reverse AD `d/dRBC:1:0` | Abs diff | Rel diff |
| --- | ---: | ---: | ---: | ---: |
| `transport:softmax_Er` | `-6.2627250000000000e+01` | `-6.2626550584360913e+01` | `6.994156e-04` | `1.116791e-05` |
| `transport:smooth_root_proxy` | `-5.5764050000000004e-04` | `-3.7595015613984747e-04` | `1.816903e-04` | `3.258198e-01` |
| `transport:Er2_volume_average` | `-2.7123439999999999e+02` | `-2.7118432937475063e+02` | `5.007063e-02` | `1.846028e-04` |
| `transport:Er_volume_average` | `-2.3644030000000001e+01` | `-2.3646073953442198e+01` | `2.043953e-03` | `8.644691e-05` |
| `transport:electron_temperature_volume_average_keV` | `-2.2448650000000001e-02` | `-2.2448497948865054e-02` | `1.520511e-07` | `6.773286e-06` |
| `transport:total_pressure_volume_average` | `-7.7521010000000001e-02` | `-7.7520440608481067e-02` | `5.693915e-07` | `7.344996e-06` |
| `transport:alpha_power_volume_average_mw_m3` | `1.2073920000000001e-03` | `1.2073916077413038e-03` | `3.922587e-10` | `3.248810e-07` |

Conclusion:

- The known 16-step full-transport reverse AD rows match FD well for the main
  objectives.
- `smooth_root_proxy` has small absolute error but large relative error because
  the derivative itself is very small and sensitive to the smooth sign/root
  proxy construction.
