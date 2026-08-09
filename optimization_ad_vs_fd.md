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

## Root-Only Ambipolarity: Shared-Payload AD vs FD

Configuration:

- Config: `examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml`
- Path: initial-Er root only, no Radau time evolution
- Shared-payload smoke flag: `--initial-Er-root-shared-payload-compare-smoke`
- FD benchmark flag: `--initial-Er-root-only-fd`

AD command:

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

FD command:

```bash
python ./examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --parameter RBC:1:0 \
  --geometry-fd-lane frozen_linearized \
  --radau-jacobian-reuse-mode legacy \
  --initial-Er-root-ad jax_selected_root \
  --initial-Er-root-only-fd \
  --root-only-objective all
```

| Objective | Shared-payload AD value | Shared-payload AD `d/dRBC:1:0` | Root-only FD value | Root-only FD `d/dRBC:1:0` | Abs diff | Rel diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `transport:softmax_Er` | `2.0479476664720302e+01` | `-5.1293330925305824e+01` | `2.0479476574395946e+01` | `-5.1295189999999998e+01` | `1.859075e-03` | `3.624267e-05` |
| `transport:smooth_root_proxy` | `8.0591612259350128e-11` | `2.2505242258459596e-09` | `8.0591615122458892e-11` | `2.1395930000000002e-09` | `1.109312e-10` | `5.184688e-02` |
| `transport:Er_transition_left` | `1.7657228878480385e+01` | `-2.0079770135329596e+01` | `1.7657228840367615e+01` | `-2.0054269999999999e+01` | `2.550014e-02` | `1.271556e-03` |
| `transport:Er_transition_right` | `1.8321801837429653e+01` | `-2.2278454432960640e+01` | `1.8321801790781215e+01` | `-2.2277920000000002e+01` | `5.344330e-04` | `2.398936e-05` |
| `transport:Er2_volume_average` | `2.5947838715347029e+02` | `-1.8476398068474273e+02` | `2.5946722947647578e+02` | `-1.8458140000000000e+02` | `1.825807e-01` | `9.891608e-04` |
| `transport:Er_volume_average` | `-3.5568787373760746e+00` | `-2.0622727802587601e+01` | `-3.5566293393826456e+00` | `-2.0614270000000001e+01` | `8.457803e-03` | `4.102887e-04` |
| `transport:bootstrap_current_softmax_abs_scaled` | `1.3565451497857757e+00` | `-1.7061866715282692e+00` | `1.3565451791686551e+00` | `-1.7058819999999999e+00` | `3.046715e-04` | `1.786006e-04` |

Saved shared-payload AD profile/geometry derivative matrix:

| Objective | Value | `d/dn0` | `d/dT0` | `d/ddensity_shape_power` | `d/dtemperature_shape_power` | `d/dRBC:1:0` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `transport:softmax_Er` | `2.0479476664720302e+01` | `-4.3986401826353054e+00` | `3.6880196309403508e+00` | `-9.7242344747717077e-02` | `2.2791745736947240e+00` | `-5.1293330925305824e+01` |
| `transport:smooth_root_proxy` | `8.0591612259350128e-11` | `5.1222741603851318e-09` | `-1.8115024386721430e-10` | `2.1916977510205021e-20` | `7.1122704574200364e-09` | `2.2505242258459596e-09` |
| `transport:Er_transition_left` | `1.7657228878480385e+01` | `-1.4233550838687483e+00` | `1.8190927365914935e+00` | `-1.2379524908495901e-02` | `-7.2635479673638610e+00` | `-2.0079770135329596e+01` |
| `transport:Er_transition_right` | `1.8321801837429653e+01` | `-1.6478080431210720e+00` | `1.9848280655116410e+00` | `-1.7743713163044173e-02` | `-6.3407456901369939e+00` | `-2.2278454432960640e+01` |
| `transport:Er2_volume_average` | `2.5947838715347029e+02` | `-1.5277014691028512e+00` | `3.3690713479802625e+01` | `3.9870316779057813e+00` | `-1.5516698692271390e+01` | `-1.8476398068474273e+02` |
| `transport:Er_volume_average` | `-3.5568787373760746e+00` | `-2.0739439797793513e+00` | `9.3014742407775575e-01` | `-1.0160552610600834e-01` | `-8.1985062107760931e-01` | `-2.0622727802587601e+01` |
| `transport:bootstrap_current_softmax_abs_scaled` | `1.3565451497857757e+00` | `-2.1467313139312133e-03` | `2.1623657522770276e-01` | `-1.2293719020797775e-02` | `1.4395745357398442e+00` | `-1.7061866715282692e+00` |
| `geometry:boozer_qi_objective` | `3.2109136309506734e-03` | `0.0000000000000000e+00` | `0.0000000000000000e+00` | `0.0000000000000000e+00` | `0.0000000000000000e+00` | `8.9989229071298027e-02` |
| `geometry:boozer_maxj_objective` | `1.3389843913753852e-01` | `0.0000000000000000e+00` | `0.0000000000000000e+00` | `0.0000000000000000e+00` | `0.0000000000000000e+00` | `-1.1593167987268771e+00` |
| `geometry:vmec_aspect_ratio` | `1.0015330918957178e+01` | `0.0000000000000000e+00` | `0.0000000000000000e+00` | `0.0000000000000000e+00` | `0.0000000000000000e+00` | `-5.4006784187006147e+00` |
| `geometry:vmec_iota_mean` | `-5.9365259966101458e-01` | `0.0000000000000000e+00` | `0.0000000000000000e+00` | `0.0000000000000000e+00` | `0.0000000000000000e+00` | `2.4405140609263865e-01` |
| `geometry:vmec_magnetic_well` | `-2.7476128749679612e-02` | `0.0000000000000000e+00` | `0.0000000000000000e+00` | `0.0000000000000000e+00` | `0.0000000000000000e+00` | `-1.1090116065531674e-02` |
| `geometry:vmec_mirror_ratio` | `2.1153803467163693e-01` | `0.0000000000000000e+00` | `0.0000000000000000e+00` | `0.0000000000000000e+00` | `0.0000000000000000e+00` | `-5.9359094714046601e-01` |

T0 root-only FD comparison:

| Objective | Shared-payload AD `d/dT0` | Root-only FD `d/dT0` | Abs diff | Rel diff |
| --- | ---: | ---: | ---: | ---: |
| `transport:softmax_Er` | `3.6880196309403508e+00` | `3.6880290000000000e+00` | `9.369060e-06` | `2.540397e-06` |
| `transport:smooth_root_proxy` | `-1.8115024386721430e-10` | `-1.4743910000000000e-10` | `3.371114e-11` | `2.286445e-01` |
| `transport:Er_transition_left` | `1.8190927365914935e+00` | `1.8190790000000001e+00` | `1.373659e-05` | `7.551399e-06` |
| `transport:Er_transition_right` | `1.9848280655116410e+00` | `1.9848350000000001e+00` | `6.934488e-06` | `3.493735e-06` |
| `transport:Er2_volume_average` | `3.3690713479802625e+01` | `3.3684800000000000e+01` | `5.913480e-03` | `1.755534e-04` |
| `transport:Er_volume_average` | `9.3014742407775575e-01` | `9.2992190000000003e-01` | `2.255241e-04` | `2.425194e-04` |
| `transport:bootstrap_current_softmax_abs_scaled` | `2.1623657522770276e-01` | `2.1623760000000000e-01` | `1.024772e-06` | `4.739103e-06` |

n0 root-only frozen-linearized-root FD comparison:

| Objective | Shared-payload AD `d/dn0` | Root-only FD `d/dn0` | Abs diff | Rel diff | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `transport:softmax_Er` | `-4.3986401826353054e+00` | `-4.3985670000000003e+00` | `7.318264e-05` | `1.663784e-05` | ok |
| `transport:smooth_root_proxy` | `5.1222741603851318e-09` | `-1.4856530000000001e-10` | `5.270839e-09` | not meaningful | tiny derivative/sign noise |
| `transport:Er_transition_left` | `-1.4233550838687483e+00` | `-1.4233980000000002e+00` | `4.291613e-05` | `3.015048e-05` | ok |
| `transport:Er_transition_right` | `-1.6478080431210720e+00` | `-1.6477620000000000e+00` | `4.604312e-05` | `2.794282e-05` | ok |
| `transport:Er2_volume_average` | `-1.5277014691028512e+00` | `-1.5276720000000001e+00` | `2.946910e-05` | `1.929020e-05` | ok |
| `transport:Er_volume_average` | `-2.0739439797793513e+00` | `-2.0739440000000000e+00` | `2.022065e-08` | `9.749853e-09` | ok |
| `transport:bootstrap_current_softmax_abs_scaled` | `-2.1467313139312133e-03` | `-2.1472470000000001e-03` | `5.156861e-07` | `2.401615e-04` | ok |

Density-shape-power root-only FD comparison:

| Objective | Shared-payload AD `d/ddensity_shape_power` | Root-only FD `d/ddensity_shape_power` | Abs diff | Rel diff |
| --- | ---: | ---: | ---: | ---: |
| `transport:softmax_Er` | `-9.7242344747717077e-02` | `-9.7335660000000000e-02` | `9.331525e-05` | `9.586954e-04` |
| `transport:smooth_root_proxy` | `2.1916977510205021e-20` | `0.0000000000000000e+00` | `2.191698e-20` | not meaningful |
| `transport:Er_transition_left` | `-1.2379524908495901e-02` | `-1.5275880000000000e-02` | `2.896355e-03` | `1.896032e-01` |
| `transport:Er_transition_right` | `-1.7743713163044173e-02` | `-1.7620520000000000e-02` | `1.231932e-04` | `6.991460e-03` |
| `transport:Er2_volume_average` | `3.9870316779057813e+00` | `3.9871900000000000e+00` | `1.583221e-04` | `3.970769e-05` |
| `transport:Er_volume_average` | `-1.0160552610600834e-01` | `-1.0177440000000000e-01` | `1.688739e-04` | `1.659296e-03` |
| `transport:bootstrap_current_softmax_abs_scaled` | `-1.2293719020797775e-02` | `-1.2295100000000000e-02` | `1.380979e-06` | `1.123195e-04` |

Temperature-shape-power root-only frozen-linearized-root FD comparison:

| Objective | Shared-payload AD `d/dtemperature_shape_power` | Root-only FD `d/dtemperature_shape_power` | Abs diff | Rel diff | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `transport:softmax_Er` | `2.2791745736947240e+00` | `2.2789550000000001e+00` | `2.195737e-04` | `9.634841e-05` | ok |
| `transport:smooth_root_proxy` | `7.1122704574200364e-09` | `7.1122720000000004e-09` | `1.542580e-15` | `2.168899e-07` | ok |
| `transport:Er_transition_left` | `-7.2635479673638610e+00` | `-7.2632740000000002e+00` | `2.739674e-04` | `3.771954e-05` | ok |
| `transport:Er_transition_right` | `-6.3407456901369939e+00` | `-6.3408630000000001e+00` | `1.173099e-04` | `1.850061e-05` | ok |
| `transport:Er2_volume_average` | `-1.5516698692271390e+01` | `-1.5516820000000000e+01` | `1.213077e-04` | `7.817821e-06` | ok |
| `transport:Er_volume_average` | `-8.1985062107760931e-01` | `-8.1985240000000001e-01` | `1.778922e-06` | `2.169808e-06` | ok |
| `transport:bootstrap_current_softmax_abs_scaled` | `1.4395745357398442e+00` | `1.4395700000000000e+00` | `4.535740e-06` | `3.150760e-06` | ok |

Note:

- The root-only FD check now uses the full realtime-geometry FD script with
  `--initial-Er-root-only-fd`, so it reuses the same geometry/profile FD setup
  but stops before Radau time evolution.
- `smooth_root_proxy` has a larger relative difference because both AD and FD
  derivatives are `O(1e-9)`. Its absolute difference is `1.1e-10`.
- The bootstrap current row now has a nontrivial value and matches FD to about
  `1.8e-4` relative for `d/dRBC:1:0`.
- The `temperature_shape_power` frozen-linearized root FD check validates all
  rows, including `Er2_volume_average` and `Er_volume_average`.
- The `n0` frozen-linearized root FD check also validates all rows. The earlier
  selected-root `n0` FD check had the same branch-reselection contamination in
  the Er volume-average rows and should be kept as a branch-sensitivity
  diagnostic, not as the AD-branch reference.
- The earlier selected-root FD lane reselected roots at `p +/- h` and produced
  large mismatches for the two Er volume-average rows; that lane is useful as a
  branch-sensitivity diagnostic but is not the AD-branch reference.

## Full Transport: Known 16-Step FD vs Reverse AD

Configuration:

- Config: `examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml`
- Path: full realtime geometry transport with 16 accepted Radau steps
- Initial-Er root AD: `jax_selected_root`
- Geometry FD lane: frozen-linearized

## Full Transport: 2-Step Shared-Payload AD

Configuration:

- Config: `examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml`
- Path: full realtime geometry transport with `2` accepted Radau steps
- Initial-Er root AD: `jax_selected_root`
- Reverse segment length: `1`
- Mode: `--full-transport-shared-payload-smoke`
- Output: `outputs/autodiff_transport_lagged_ntx/reverse_ad/transport_reverse_ad_only_full_transport_shared_payload_smoke.json`
- Elapsed time: `1267.576 s`

Command:

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
  --reverse-step-bwd-mode reduced_cotangent \
  --initial-Er-root-ad jax_selected_root \
  --full-transport-shared-payload-smoke
```

| Objective | Value | `d/dn0` | `d/dT0` | `d/ddensity_shape_power` | `d/dtemperature_shape_power` | `d/dRBC:1:0` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `transport:softmax_Er` | `2.0474480434757560e+01` | `-4.4010821718627282e+00` | `3.6893021316932324e+00` | `-9.7245517683807015e-02` | `2.2784341729715321e+00` | `-5.1317447005281466e+01` |
| `transport:smooth_root_proxy` | `1.9565517403925628e-10` | `0.0000000000000000e+00` | `-4.2240800049036898e-10` | `-1.0873396947539999e-13` | `1.7568065363844018e-08` | `4.1615832435909986e-09` |
| `transport:Er2_volume_average` | `2.5789275767390154e+02` | `-1.1130507692496927e+00` | `3.3314765761704457e+01` | `3.9716409094558180e+00` | `-1.6627182341155759e+01` | `-1.8107833510078140e+02` |
| `transport:Er_volume_average` | `-3.5463367280578586e+00` | `-2.0727795330781582e+00` | `9.2860232007956911e-01` | `-1.0172552313027816e-01` | `-8.2119695491996003e-01` | `-2.0555620539844501e+01` |
| `transport:electron_temperature_volume_average_keV` | `6.4471445508615828e+00` | `4.7658840766284793e-04` | `3.4852955165267474e-01` | `9.8558205532291165e-06` | `1.4942008613521125e+00` | `-1.3670544339803796e-02` |
| `transport:total_pressure_volume_average` | `3.3551072628692104e+01` | `7.9013769812319090e+00` | `1.8280094754912617e+00` | `2.3976273858151381e-01` | `7.5981644339917755e+00` | `-7.7574037477612573e-02` |
| `transport:alpha_power_volume_average_mw_m3` | `5.7786228569831921e-01` | `2.7421038425018640e-01` | `8.1604646824932223e-02` | `2.3155465380623520e-03` | `2.7875538302773323e-01` | `-1.7350581859371954e-03` |

Saved 2-step FD lookup:

- I do not currently find a saved full realtime-geometry 2-step FD table for `d/dRBC:1:0` and all objectives in the local docs/outputs.
- The saved 2-step references I found are profile-only/`softmax_Er` forward AD checks, e.g. `dsoftmax_Er/dn0 = -3.578618e-01`, `dsoftmax_Er/dT0 = 3.010300e-01`, `dsoftmax_Er/ddensity_shape_power = -7.886158e-03`, `dsoftmax_Er/dtemperature_shape_power = 1.779141e-01`. Those are not the matching realtime-geometry `RBC:1:0` FD reference for this table.
- The matching FD command to generate the missing 2-step geometry reference is:

```bash
python ./examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --parameter RBC:1:0 \
  --geometry-fd-lane frozen_linearized \
  --accepted-step-limit 2 \
  --radau-jacobian-reuse-mode legacy \
  --replay-mode accepted \
  --initial-Er-root-ad jax_selected_root
```

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

### 16-Step Full-Transport Shared-Payload Smoke

Current shared-payload smoke run:

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
  --initial-Er-root-ad jax_selected_root \
  --full-transport-shared-payload-smoke
```

Run metadata:

- `mode = transport_reverse_ad_only_full_transport_shared_payload_smoke`
- `residual_count = 7`
- `parameter_count = 5`
- `elapsed_s = 3531.475`
- output JSON: `outputs/autodiff_transport_lagged_ntx/reverse_ad/transport_reverse_ad_only_full_transport_shared_payload_smoke.json`
- This same 7-row reverse table was re-attached later as the current
  16-step full-transport shared-payload reverse result. It predates/omits
  `Er_transition_left`, `Er_transition_right`, and
  `bootstrap_current_softmax_abs_scaled` in the full-transport reverse table.

Full AD table:

| Objective | Value | `d/dn0` | `d/dT0` | `d/ddensity_shape_power` | `d/dtemperature_shape_power` | `d/dRBC:1:0` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `transport:softmax_Er` | `2.0694998241267641e+01` | `-5.0838282415568408e+00` | `4.1306249928548100e+00` | `-1.1591644624461980e-01` | `4.2417414653234697e+00` | `-6.2626550836918256e+01` |
| `transport:smooth_root_proxy` | `1.9868729727533445e-02` | `-5.8593750000000000e-03` | `-7.7788585014637937e-05` | `-1.3650089969452495e-07` | `2.3783611385449557e-03` | `-3.7595015565651822e-04` |
| `transport:Er2_volume_average` | `2.4372053202139412e+02` | `-6.8729867376835294e+00` | `3.6065316917000459e+01` | `3.7772605808640574e+00` | `-9.9007831215974562e+00` | `-2.7118433140008011e+02` |
| `transport:Er_volume_average` | `-3.4309746025765144e+00` | `-2.3765676300577470e+00` | `1.0952038126424903e+00` | `-1.0966510378096389e-01` | `-3.1844641881012514e-01` | `-2.3646073978935458e+01` |
| `transport:electron_temperature_volume_average_keV` | `6.4597967687544342e+00` | `3.2044199169065646e-03` | `3.5042845972161463e-01` | `-1.7862537077364612e-04` | `1.5035680183819486e+00` | `-2.2448497735886055e-02` |
| `transport:total_pressure_volume_average` | `3.3559238356010034e+01` | `7.9065460031525578e+00` | `1.8293618203570214e+00` | `2.3978045308837445e-01` | `7.6008795466190691e+00` | `-7.7520439508832056e-02` |
| `transport:alpha_power_volume_average_mw_m3` | `5.7709114832136932e-01` | `2.7390173486511626e-01` | `8.1435927075356962e-02` | `2.3105926577552259e-03` | `2.7792303880127184e-01` | `1.2073916329946855e-03` |

Comparison against saved 16-step FD `d/dRBC:1:0`:

| Objective | 16-step FD `d/dRBC:1:0` | Shared-payload AD `d/dRBC:1:0` | Abs diff | Rel diff |
| --- | ---: | ---: | ---: | ---: |
| `transport:softmax_Er` | `-6.2627250000000000e+01` | `-6.2626550836918256e+01` | `6.991631e-04` | `1.116388e-05` |
| `transport:smooth_root_proxy` | `-5.5764050000000004e-04` | `-3.7595015565651822e-04` | `1.816903e-04` | `3.258199e-01` |
| `transport:Er2_volume_average` | `-2.7123439999999999e+02` | `-2.7118433140008011e+02` | `5.006860e-02` | `1.845953e-04` |
| `transport:Er_volume_average` | `-2.3644030000000001e+01` | `-2.3646073978935458e+01` | `2.043979e-03` | `8.644799e-05` |
| `transport:electron_temperature_volume_average_keV` | `-2.2448650000000001e-02` | `-2.2448497735886055e-02` | `1.522641e-07` | `6.782774e-06` |
| `transport:total_pressure_volume_average` | `-7.7521010000000001e-02` | `-7.7520439508832056e-02` | `5.704912e-07` | `7.359181e-06` |
| `transport:alpha_power_volume_average_mw_m3` | `1.2073920000000001e-03` | `1.2073916329946855e-03` | `3.670053e-10` | `3.039653e-07` |

Comparison against existing saved 16-step profile references:

- These saved profile references are profile-only 16-step references from
  `ad_forward_lane.md`, not a matching `profiles_plus_realtime_geometry`
  frozen-FD table for this exact shared-payload run.
- The pressure/temperature/alpha rows remain very close.
- The Er/root-sensitive rows are not expected to match these profile-only
  references exactly because this shared-payload run includes realtime geometry
  and `jax_selected_root` coupling.

| Objective | Parameter | Shared-payload AD | Saved profile reference | Abs diff | Rel diff |
| --- | --- | ---: | ---: | ---: | ---: |
| `transport:softmax_Er` | `n0` | `-5.0838282415568408e+00` | `-3.7637150000000000e+00` | `1.320113e+00` | `3.507472e-01` |
| `transport:softmax_Er` | `T0` | `4.1306249928548100e+00` | `3.0572340000000000e+00` | `1.073391e+00` | `3.510988e-01` |
| `transport:softmax_Er` | `density_shape_power` | `-1.1591644624461980e-01` | `-8.5283690000000000e-02` | `3.063276e-02` | `3.591867e-01` |
| `transport:softmax_Er` | `temperature_shape_power` | `4.2417414653234697e+00` | `3.2202820000000001e+00` | `1.021459e+00` | `3.171956e-01` |
| `transport:smooth_root_proxy` | `n0` | `-5.8593750000000000e-03` | `4.2146960000000000e-04` | `6.280845e-03` | `1.490225e+01` |
| `transport:smooth_root_proxy` | `T0` | `-7.7788585014637937e-05` | `-1.7306870000000001e-04` | `9.528011e-05` | `5.505335e-01` |
| `transport:smooth_root_proxy` | `density_shape_power` | `-1.3650089969452495e-07` | `-3.6866190000000000e-06` | `3.550118e-06` | `9.629879e-01` |
| `transport:smooth_root_proxy` | `temperature_shape_power` | `2.3783611385449557e-03` | `1.6721560000000000e-02` | `1.434320e-02` | `8.577518e-01` |
| `transport:Er2_volume_average` | `n0` | `-6.8729867376835294e+00` | `-4.3770650000000000e+00` | `2.495922e+00` | `5.701815e-01` |
| `transport:Er2_volume_average` | `T0` | `3.6065316917000459e+01` | `2.4631770000000000e+01` | `1.143355e+01` | `4.641791e-01` |
| `transport:Er2_volume_average` | `density_shape_power` | `3.7772605808640574e+00` | `1.3390750000000000e+00` | `2.438186e+00` | `1.820799e+00` |
| `transport:Er2_volume_average` | `temperature_shape_power` | `-9.9007831215974562e+00` | `-3.2042000000000002e+01` | `2.214122e+01` | `6.910062e-01` |
| `transport:Er_volume_average` | `n0` | `-2.3765676300577470e+00` | `-1.7386590000000000e+00` | `6.379086e-01` | `3.668970e-01` |
| `transport:Er_volume_average` | `T0` | `1.0952038126424903e+00` | `8.2147700000000001e-01` | `2.737268e-01` | `3.332125e-01` |
| `transport:Er_volume_average` | `density_shape_power` | `-1.0966510378096389e-01` | `-4.3872050000000003e-02` | `6.579305e-02` | `1.499659e+00` |
| `transport:Er_volume_average` | `temperature_shape_power` | `-3.1844641881012514e-01` | `-4.1779350000000000e-01` | `9.934708e-02` | `2.377899e-01` |
| `transport:electron_temperature_volume_average_keV` | `n0` | `3.2044199169065646e-03` | `3.1878530000000000e-03` | `1.656692e-05` | `5.196895e-03` |
| `transport:electron_temperature_volume_average_keV` | `T0` | `3.5042845972161463e-01` | `3.5041640000000002e-01` | `1.205972e-05` | `3.441542e-05` |
| `transport:electron_temperature_volume_average_keV` | `density_shape_power` | `-1.7862537077364612e-04` | `-1.8212640000000001e-04` | `3.501029e-06` | `1.922307e-02` |
| `transport:electron_temperature_volume_average_keV` | `temperature_shape_power` | `1.5035680183819486e+00` | `1.5035620000000001e+00` | `6.018382e-06` | `4.002747e-06` |
| `transport:total_pressure_volume_average` | `n0` | `7.9065460031525578e+00` | `7.9065220000000000e+00` | `2.400315e-05` | `3.035867e-06` |
| `transport:total_pressure_volume_average` | `T0` | `1.8293618203570214e+00` | `1.8293340000000000e+00` | `2.782036e-05` | `1.520792e-05` |
| `transport:total_pressure_volume_average` | `density_shape_power` | `2.3978045308837445e-01` | `2.3977710000000000e-01` | `3.353088e-06` | `1.398419e-05` |
| `transport:total_pressure_volume_average` | `temperature_shape_power` | `7.6008795466190691e+00` | `7.6009360000000000e+00` | `5.645338e-05` | `7.427161e-06` |
| `transport:alpha_power_volume_average_mw_m3` | `n0` | `2.7390173486511626e-01` | `2.7390430000000000e-01` | `2.565135e-06` | `9.365080e-06` |
| `transport:alpha_power_volume_average_mw_m3` | `T0` | `8.1435927075356962e-02` | `8.1428780000000006e-02` | `7.147075e-06` | `8.776964e-05` |
| `transport:alpha_power_volume_average_mw_m3` | `density_shape_power` | `2.3105926577552259e-03` | `2.3111570000000000e-03` | `5.643422e-07` | `2.441817e-04` |
| `transport:alpha_power_volume_average_mw_m3` | `temperature_shape_power` | `2.7792303880127184e-01` | `2.7792670000000003e-01` | `3.661199e-06` | `1.317325e-05` |

Historical profile FD values found locally:

- `auto_diff.md` contains older profile AD-vs-FD snippets for `n0` and `T0`.
- Those values are from a different historical setup and should not be used as
  the matching FD reference for the current shared-payload run.

### 16-Step Full-Transport FD With Bootstrap

Current FD run:

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

FD values:

Bootstrap-current note:

- The bootstrap row below is the corrected full-transport FD reference using
  realtime NTX momentum-corrected `Upar`.
- For bootstrap, the full FD splits cleanly into explicit geometry plus
  baseline-geometry final-state pieces:
  `-1.656870e+00 + -1.349065e-01 = -1.7917765e+00`, matching the full FD
  `-1.791772e+00` within FD precision.

| Objective | Value | Full FD `d/dRBC:1:0` | Fixed-final-state explicit geometry FD | Baseline-geometry final-state FD |
| --- | ---: | ---: | ---: | ---: |
| `transport:softmax_Er` | `2.0694998194713662e+01` | `-6.2627320000000000e+01` | `0.0000000000000000e+00` | `-6.2627320000000000e+01` |
| `transport:smooth_root_proxy` | `1.9869328464046845e-02` | `-5.5764290000000000e-04` | `0.0000000000000000e+00` | `-5.5764290000000000e-04` |
| `transport:Er_transition_left` | `1.7686989067626193e+01` | `-2.0130240000000000e+01` | `0.0000000000000000e+00` | `-2.0130240000000000e+01` |
| `transport:Er_transition_right` | `1.8352979233941966e+01` | `-2.2349450000000000e+01` | `0.0000000000000000e+00` | `-2.2349450000000000e+01` |
| `transport:Er2_volume_average` | `2.4370331301244494e+02` | `-2.7123420000000000e+02` | `3.5418810000000000e-01` | `-2.7158840000000000e+02` |
| `transport:Er_volume_average` | `-3.4306058803160564e+00` | `-2.3644030000000000e+01` | `-5.3707040000000000e-02` | `-2.3590320000000000e+01` |
| `transport:electron_temperature_volume_average_keV` | `6.4597967427865761e+00` | `-2.2448650000000000e-02` | `-1.2869300000000000e-02` | `-9.5793540000000000e-03` |
| `transport:total_pressure_volume_average` | `3.3559238343437478e+01` | `-7.7521020000000000e-02` | `-7.7538330000000000e-02` | `1.7328650000000000e-05` |
| `transport:alpha_power_volume_average_mw_m3` | `5.7709114846781473e-01` | `1.2073930000000000e-03` | `-1.9243270000000000e-03` | `3.1317200000000000e-03` |
| `transport:bootstrap_current_softmax_abs_scaled` | `1.3615203117088308e+00` | `-1.7917720000000000e+00` | `-1.6568700000000000e+00` | `-1.3490650000000000e-01` |

Comparison against the saved 16-step shared-payload reverse AD rows:

| Objective | Full FD `d/dRBC:1:0` | Shared-payload AD `d/dRBC:1:0` | Abs diff | Rel diff |
| --- | ---: | ---: | ---: | ---: |
| `transport:softmax_Er` | `-6.2627319999999997e+01` | `-6.2626550836918256e+01` | `7.691631e-04` | `1.228159e-05` |
| `transport:smooth_root_proxy` | `-5.5764289999999999e-04` | `-3.7595015565651822e-04` | `1.816927e-04` | `3.258228e-01` |
| `transport:Er_transition_left` | `-2.0130240000000000e+01` | `-2.0130902762944103e+01` | `6.627629e-04` | `3.292375e-05` |
| `transport:Er_transition_right` | `-2.2349450000000000e+01` | `-2.2349701278432580e+01` | `2.512784e-04` | `1.124316e-05` |
| `transport:Er2_volume_average` | `-2.7123419999999999e+02` | `-2.7118433140008011e+02` | `4.986860e-02` | `1.838581e-04` |
| `transport:Er_volume_average` | `-2.3644030000000001e+01` | `-2.3646073978935458e+01` | `2.043979e-03` | `8.644799e-05` |
| `transport:electron_temperature_volume_average_keV` | `-2.2448650000000001e-02` | `-2.2448497735886055e-02` | `1.522641e-07` | `6.782774e-06` |
| `transport:total_pressure_volume_average` | `-7.7521019999999996e-02` | `-7.7520439508832056e-02` | `5.804912e-07` | `7.488178e-06` |
| `transport:alpha_power_volume_average_mw_m3` | `1.2073930000000000e-03` | `1.2073916329946855e-03` | `1.367005e-09` | `1.132196e-06` |
| `transport:bootstrap_current_softmax_abs_scaled` | `-1.7917720000000000e+00` | `-1.7920897096814259e+00` | `3.177097e-04` | `1.773159e-04` |

The transition and bootstrap rows are included in the current mixed
shared-payload AD table below.

### 16-Step Full-Transport Profile FD: `n0`

Current FD run:

```bash
python ./examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --parameter n0 \
  --geometry-fd-lane frozen_linearized \
  --accepted-step-limit 16 \
  --radau-jacobian-reuse-mode legacy \
  --replay-mode accepted \
  --initial-Er-root-ad jax_selected_root
```

Run summary:

- Mode: `transport_realtime_geometry_forward_fd`
- Parameter kind: `profile`
- Baseline value: `4.210000e+00`
- FD step: `1.263000e-06`
- Output JSON: `outputs/autodiff_transport_lagged_ntx/realtime_geometry_fd/n0_forward_fd_summary.json`

| Objective | Value | 16-step FD `d/dn0` |
| --- | ---: | ---: |
| `transport:softmax_Er` | `2.0493040599404146e+01` | `-7.5793390000000000e+00` |
| `transport:smooth_root_proxy` | `9.8039215686309099e-03` | `0.0000000000000000e+00` |
| `transport:Er_transition_left` | `1.7729764197009018e+01` | `-1.4451420000000000e+00` |
| `transport:Er_transition_right` | `1.8376266968588908e+01` | `-1.6704680000000001e+00` |
| `transport:Er2_volume_average` | `2.3720952417865757e+02` | `6.6936770000000000e+02` |
| `transport:Er_volume_average` | `-3.4526135846189607e+00` | `1.6889930000000000e+01` |
| `transport:electron_temperature_volume_average_keV` | `6.5646440779302155e+00` | `-8.6623840000000000e-04` |
| `transport:total_pressure_volume_average` | `3.4213472702876274e+01` | `8.0596720000000002e+00` |
| `transport:alpha_power_volume_average_mw_m3` | `5.8935810845846526e-01` | `3.0339720000000001e-01` |
| `transport:bootstrap_current_softmax_abs_scaled` | `1.4485488845680283e+00` | `-1.0760220000000000e-03` |

Status:

- This is the fresh profile-parameter FD reference for `n0` after the recent
  finite-volume boundary/evaluated-state updates.
- Rerun the shared-payload reverse-AD smoke before making a current
  profile-column AD-vs-FD comparison, because the existing reverse rows below
  were saved before those updates.

## Full Transport: Mixed Shared-Payload AD Update

This is the current mixed optimization-facing smoke after wiring the full
transport path to print both transport and geometry objectives through the
shared-payload machinery.

### 2-Step Mixed Shared-Payload AD

- Command: `benchmark_transport_reverse_ad_only.py ... --accepted-step-limit 2 --reverse-segment-length 1 --full-transport-shared-payload-smoke`
- Residual count: `16`
- Parameter count: `5`
- Elapsed time: `1848.789 s`
- Status: completed without OOM after compact full-transport bootstrap objective rule.

| Objective | Value | `d/dn0` | `d/dT0` | `d/ddensity_shape_power` | `d/dtemperature_shape_power` | `d/dRBC:1:0` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `transport:softmax_Er` | `2.0474480434757560e+01` | `-4.4010821718627282e+00` | `3.6893021316932324e+00` | `-9.7245517683807015e-02` | `2.2784341729715321e+00` | `-5.1317447005281480e+01` |
| `transport:smooth_root_proxy` | `1.9565517403925628e-10` | `0.0000000000000000e+00` | `-4.2240800049036898e-10` | `-1.0873396947539999e-13` | `1.7568065363844018e-08` | `4.1615832435574225e-09` |
| `transport:Er_transition_left` | `1.7657165985271565e+01` | `-1.4241855329757844e+00` | `1.8196098174192941e+00` | `-1.2412582416126678e-02` | `-7.2603261187208350e+00` | `-2.0089861173590997e+01` |
| `transport:Er_transition_right` | `1.8321478503969928e+01` | `-1.6488669584139266e+00` | `1.9854658787281658e+00` | `-1.7786057135425234e-02` | `-6.3370794754853419e+00` | `-2.2291200089653881e+01` |
| `transport:Er2_volume_average` | `2.5789275767390154e+02` | `-1.1130507692496678e+00` | `3.3314765761704457e+01` | `3.9716409094558180e+00` | `-1.6627182341155731e+01` | `-1.8107833510078163e+02` |
| `transport:Er_volume_average` | `-3.5463367280578586e+00` | `-2.0727795330781582e+00` | `9.2860232007956911e-01` | `-1.0172552313027822e-01` | `-8.2119695491996003e-01` | `-2.0555620539844504e+01` |
| `transport:electron_temperature_volume_average_keV` | `6.4471445508615828e+00` | `4.7658840766284793e-04` | `3.4852955165267474e-01` | `9.8558205532291165e-06` | `1.4942008613521125e+00` | `-1.3670544339803787e-02` |
| `transport:total_pressure_volume_average` | `3.3551072628692104e+01` | `7.9013769812319090e+00` | `1.8280094754912617e+00` | `2.3976273858151381e-01` | `7.5981644339917755e+00` | `-7.7574037477612962e-02` |
| `transport:alpha_power_volume_average_mw_m3` | `5.7786228569831921e-01` | `2.7421038425018640e-01` | `8.160464682493223e-02` | `2.3155465380623520e-03` | `2.7875538302773323e-01` | `-1.7350581859371954e-03` |
| `transport:bootstrap_current_softmax_abs_scaled` | `1.3568564560406851e+00` | `-2.2735836563506640e-03` | `2.1636345559494713e-01` | `-1.2305256192806667e-02` | `1.4402456009787310e+00` | `-1.7122866213720369e+00` |
| `geometry:boozer_qi_objective` | `3.2109136309506734e-03` | `0.0` | `0.0` | `0.0` | `0.0` | `8.9989229071147925e-02` |
| `geometry:boozer_maxj_objective` | `1.3389843913753852e-01` | `0.0` | `0.0` | `0.0` | `0.0` | `-1.1593167987082040e+00` |
| `geometry:vmec_aspect_ratio` | `1.0015330918957178e+01` | `0.0` | `0.0` | `0.0` | `0.0` | `-5.4006784187006147e+00` |
| `geometry:vmec_iota_mean` | `-5.9365259966101458e-01` | `0.0` | `0.0` | `0.0` | `0.0` | `2.4405140609263865e-01` |
| `geometry:vmec_magnetic_well` | `-2.7476128749679612e-02` | `0.0` | `0.0` | `0.0` | `0.0` | `-1.1090116065531674e-02` |
| `geometry:vmec_mirror_ratio` | `2.1153803467163693e-01` | `0.0` | `0.0` | `0.0` | `0.0` | `-5.9359094714046601e-01` |

### 16-Step Mixed Shared-Payload AD

- Command: `benchmark_transport_reverse_ad_only.py ... --accepted-step-limit 16 --reverse-segment-length 4 --full-transport-shared-payload-smoke`
- Residual count: `16`
- Parameter count: `5`
- Elapsed time: `3541.502 s`
- Status: completed without OOM after compact full-transport bootstrap objective rule.

| Objective | Value | `d/dn0` | `d/dT0` | `d/ddensity_shape_power` | `d/dtemperature_shape_power` | `d/dRBC:1:0` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `transport:softmax_Er` | `2.0694998241267641e+01` | `-5.0838282426765602e+00` | `4.1306249932125239e+00` | `-1.1591644638113782e-01` | `4.2417414639205937e+00` | `-6.2626550858639305e+01` |
| `transport:smooth_root_proxy` | `1.9868729727533445e-02` | `-5.8593750000000000e-03` | `-7.8661443694676321e-05` | `-4.6968973155259164e-10` | `2.3759369109219077e-03` | `-3.6780535157069163e-04` |
| `transport:Er_transition_left` | `1.7686989084389136e+01` | `-1.4080049846156255e+00` | `1.8189473816018960e+00` | `-1.2527738844939166e-02` | `-7.2434886819567694e+00` | `-2.0130902762944103e+01` |
| `transport:Er_transition_right` | `1.8352979259460643e+01` | `-1.6310887137958741e+00` | `1.9845372872751819e+00` | `-1.7923132068669315e-02` | `-6.3196368732760977e+00` | `-2.2349701278432580e+01` |
| `transport:Er2_volume_average` | `2.4372053202139412e+02` | `-6.8729867601779873e+00` | `3.6065316928902384e+01` | `3.7772605802803998e+00` | `-9.9007834256044234e+00` | `-2.7118433131494726e+02` |
| `transport:Er_volume_average` | `-3.4309746025765144e+00` | `-2.3765427446710738e+00` | `1.0951895590455043e+00` | `-1.0966513920041543e-01` | `-3.1848898508307144e-01` | `-2.3645736589729140e+01` |
| `transport:electron_temperature_volume_average_keV` | `6.4597967687544342e+00` | `3.2044206846031376e-03` | `3.5042845937983902e-01` | `-1.7862516380266411e-04` | `1.5035679924726344e+00` | `-2.2448477914450340e-02` |
| `transport:total_pressure_volume_average` | `3.3559238356010034e+01` | `7.9065460031560715e+00` | `1.8293618202564703e+00` | `2.3978045310409096e-01` | `7.6008795464596064e+00` | `-7.7520438710911577e-02` |
| `transport:alpha_power_volume_average_mw_m3` | `5.7709114832136932e-01` | `2.7390173501999121e-01` | `8.1435926777295040e-02` | `2.3105926842456898e-03` | `2.7792304292856762e-01` | `1.2073914238870094e-03` |
| `transport:bootstrap_current_softmax_abs_scaled` | `1.3615202803949229e+00` | `-4.0512909481635884e-03` | `2.1820601193258365e-01` | `-1.2505757233407980e-02` | `1.4495166463763556e+00` | `-1.7920897096814259e+00` |
| `geometry:boozer_qi_objective` | `3.2109136309506734e-03` | `0.0` | `0.0` | `0.0` | `0.0` | `8.9989229071780308e-02` |
| `geometry:boozer_maxj_objective` | `1.3389843913753852e-01` | `0.0` | `0.0` | `0.0` | `0.0` | `-1.1593167987256550e+00` |
| `geometry:vmec_aspect_ratio` | `1.0015330918957178e+01` | `0.0` | `0.0` | `0.0` | `0.0` | `-5.4006784187006147e+00` |
| `geometry:vmec_iota_mean` | `-5.9365259966101458e-01` | `0.0` | `0.0` | `0.0` | `0.0` | `2.4405140609263865e-01` |
| `geometry:vmec_magnetic_well` | `-2.7476128749679612e-02` | `0.0` | `0.0` | `0.0` | `0.0` | `-1.1090116065531674e-02` |
| `geometry:vmec_mirror_ratio` | `2.1153803467163693e-01` | `0.0` | `0.0` | `0.0` | `0.0` | `-5.9359094714046601e-01` |

Comparison against saved 16-step full-transport FD `d/dRBC:1:0`:

| Objective | FD `d/dRBC:1:0` | Mixed shared AD `d/dRBC:1:0` | Abs diff | Rel diff |
| --- | ---: | ---: | ---: | ---: |
| `transport:softmax_Er` | `-6.2627320000000000e+01` | `-6.2626550858639305e+01` | `7.691414e-04` | `1.228124e-05` |
| `transport:smooth_root_proxy` | `-5.5764290000000003e-04` | `-3.6780535157069163e-04` | `1.898375e-04` | `3.403266e-01` |
| `transport:Er_transition_left` | `-2.0130240000000001e+01` | `-2.0130902762944103e+01` | `6.627629e-04` | `3.292373e-05` |
| `transport:Er_transition_right` | `-2.2349450000000000e+01` | `-2.2349701278432580e+01` | `2.512784e-04` | `1.124317e-05` |
| `transport:Er2_volume_average` | `-2.7123420000000002e+02` | `-2.7118433131494726e+02` | `4.986869e-02` | `1.838583e-04` |
| `transport:Er_volume_average` | `-2.3644030000000001e+01` | `-2.3645736589729140e+01` | `1.706590e-03` | `7.218262e-05` |
| `transport:electron_temperature_volume_average_keV` | `-2.2448650000000001e-02` | `-2.2448477914450340e-02` | `1.720855e-07` | `7.665731e-06` |
| `transport:total_pressure_volume_average` | `-7.7521020000000002e-02` | `-7.7520438710911577e-02` | `5.812891e-07` | `7.498470e-06` |
| `transport:alpha_power_volume_average_mw_m3` | `1.2073930000000000e-03` | `1.2073914238870094e-03` | `1.576113e-09` | `1.305385e-06` |
| `transport:bootstrap_current_softmax_abs_scaled` | `-1.7917720000000000e+00` | `-1.7920897096814259e+00` | `3.177097e-04` | `1.773159e-04` |

Notes:

- The mixed shared-payload run now exercises the optimization-facing full
  transport and geometry rows in one table.
- The compact full-transport bootstrap rule avoids the previous OOM.
- Full-transport rows, including the corrected bootstrap row, are consistent
  with the saved FD references.
- The initial-Er root-only/ambipolarity bootstrap geometry derivative was also
  validated against FD: `d/dRBC:1:0` AD `-1.7061866715282692e+00` vs FD
  `-1.7058819999999999e+00`, relative difference `1.786006e-04`.
