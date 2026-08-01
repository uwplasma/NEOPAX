# Shared Payload FD Comparison

Current comparison target:

- VMEC input: `examples/inputs/input.QI_nfp2_newNT_opt_hires_true`
- Parameter: `RBC:1:0`
- Geometry lane/reference: frozen-linearized VMEC FD plus implicit/JVP AD target
- Shared-payload path: root-only shared payload / optimization-facing internal machinery

## Geometry Objectives

These are the corrected current QI/maxJ values after the Boozer/J-invariant
objective update.

| Objective | Shared-payload value | Shared-payload `d/dRBC:1:0` | Saved frozen-linearized FD | Saved JVP / AD target | `abs(shared - FD)` | `rel(shared - FD)` | `abs(shared - JVP)` | `rel(shared - JVP)` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `geometry:boozer_qi_objective` | `3.2109136309506734e-03` | `8.9989229070800647e-02` | `8.9996041589633161e-02` | `8.9989229074526111e-02` | `6.812519e-06` | `7.569798e-05` | `3.725464e-12` | `4.139899e-11` |
| `geometry:boozer_maxj_objective` | `1.3389843913753852e-01` | `-1.1593167987379616e+00` | `-1.1591364898936340e+00` | `-1.1593167988198267e+00` | `1.803088e-04` | `1.555366e-04` | `8.186518e-11` | `7.061503e-11` |

Conclusion:

- The shared-payload geometry rows match the implicit/JVP AD targets to around
  `1e-10` relative or better.
- The FD differences are at the expected frozen-linearized FD level:
  about `7.6e-05` relative for QI and `1.6e-04` relative for maxJ.

## Geometry Rows Without Saved FD Reference

The shared-payload root-only smoke also printed these geometry rows in an older
saved table. I do not currently have standalone frozen-linearized FD references
saved locally for these rows.

| Objective | Shared-payload value | Shared-payload `d/dRBC:1:0` | Saved FD status |
| --- | ---: | ---: | --- |
| `geometry:vmec_aspect_ratio` | `1.0015330918957178e+01` | `-5.4006784187006147e+00` | no saved FD found locally |
| `geometry:vmec_iota_mean` | `-5.9365259966101458e-01` | `2.4405140609263865e-01` | no saved FD found locally |
| `geometry:vmec_magnetic_well` | `-2.7476128749679612e-02` | `-1.1090116065531674e-02` | no saved FD found locally |
| `geometry:vmec_mirror_ratio` | `2.1100247521308457e-01` | `-5.9979060848576748e-01` | no saved FD found locally |

Important note:

- The QI/maxJ shared rows above are current corrected values.
- The aspect/iota/well/mirror rows are the saved shared-payload rows currently
  recorded in `optimization_path.md`. They still need standalone FD rows if we
  want a complete geometry FD comparison table.

## Root-Only Transport / Ambipolarity Objectives

The current saved shared-payload root-only smoke values now have matching
root-only FD references from `benchmark_transport_realtime_geometry_forward_fd.py`
with `--initial-Er-root-only-fd`.

| Objective | Shared-payload value | Shared-payload `d/dRBC:1:0` | Root-only FD value | Root-only FD `d/dRBC:1:0` | Abs diff | Rel diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `transport:softmax_Er` | `2.0479476664720302e+01` | `-5.1293330925305824e+01` | `2.0479476574395946e+01` | `-5.1295189999999998e+01` | `1.859075e-03` | `3.624267e-05` |
| `transport:smooth_root_proxy` | `8.0591612259350128e-11` | `2.2505242258459596e-09` | `8.0591615122458892e-11` | `2.1395930000000002e-09` | `1.109312e-10` | `5.184688e-02` |
| `transport:Er_transition_left` | `1.7657228878480385e+01` | `-2.0079770135329596e+01` | `1.7657228840367615e+01` | `-2.0054269999999999e+01` | `2.550014e-02` | `1.271556e-03` |
| `transport:Er_transition_right` | `1.8321801837429653e+01` | `-2.2278454432960640e+01` | `1.8321801790781215e+01` | `-2.2277920000000002e+01` | `5.344330e-04` | `2.398936e-05` |
| `transport:Er2_volume_average` | `2.5947838715347029e+02` | `-1.8476398068474273e+02` | `2.5946722947647578e+02` | `-1.8458140000000000e+02` | `1.825807e-01` | `9.891608e-04` |
| `transport:Er_volume_average` | `-3.5568787373760746e+00` | `-2.0622727802587601e+01` | `-3.5566293393826456e+00` | `-2.0614270000000001e+01` | `8.457803e-03` | `4.102887e-04` |
| `transport:bootstrap_current_softmax_abs_scaled` | `1.3565451497857757e+00` | `-1.7061866715282692e+00` | `1.3565451791686551e+00` | `-1.7058819999999999e+00` | `3.046715e-04` | `1.786006e-04` |

The bootstrap current objective is included in this root-only table. The
`smooth_root_proxy` row has a larger relative difference because the derivative
is very small; the absolute difference is `1.1e-10`.

Saved shared-payload AD derivative matrix for profile and geometry columns:

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

The `temperature_shape_power` frozen-linearized root FD check validates all
rows. The earlier selected-root FD lane reselected roots at `p +/- h` and
produced large mismatches for the two Er volume-average rows; that lane is a
branch-sensitivity diagnostic, not the AD-branch reference.

The `n0` frozen-linearized root FD check also validates all rows. The earlier
selected-root `n0` FD check had the same branch-reselection contamination in
the Er volume-average rows and should be kept only as a branch-sensitivity
diagnostic.

## Existing Full-Transport FD Values

### 2-Step Shared-Payload AD

This is the current full-transport shared-payload AD smoke with `2` accepted
steps. A matching full realtime-geometry 2-step FD table for `d/dRBC:1:0` was
not found in the saved local docs/outputs; the saved 2-step references I found
are profile-only `softmax_Er` forward-AD checks, so they should not be used as
the FD reference for this table.

| Objective | Value | `d/dn0` | `d/dT0` | `d/ddensity_shape_power` | `d/dtemperature_shape_power` | `d/dRBC:1:0` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `transport:softmax_Er` | `2.0474480434757560e+01` | `-4.4010821718627282e+00` | `3.6893021316932324e+00` | `-9.7245517683807015e-02` | `2.2784341729715321e+00` | `-5.1317447005281466e+01` |
| `transport:smooth_root_proxy` | `1.9565517403925628e-10` | `0.0000000000000000e+00` | `-4.2240800049036898e-10` | `-1.0873396947539999e-13` | `1.7568065363844018e-08` | `4.1615832435909986e-09` |
| `transport:Er2_volume_average` | `2.5789275767390154e+02` | `-1.1130507692496927e+00` | `3.3314765761704457e+01` | `3.9716409094558180e+00` | `-1.6627182341155759e+01` | `-1.8107833510078140e+02` |
| `transport:Er_volume_average` | `-3.5463367280578586e+00` | `-2.0727795330781582e+00` | `9.2860232007956911e-01` | `-1.0172552313027816e-01` | `-8.2119695491996003e-01` | `-2.0555620539844501e+01` |
| `transport:electron_temperature_volume_average_keV` | `6.4471445508615828e+00` | `4.7658840766284793e-04` | `3.4852955165267474e-01` | `9.8558205532291165e-06` | `1.4942008613521125e+00` | `-1.3670544339803796e-02` |
| `transport:total_pressure_volume_average` | `3.3551072628692104e+01` | `7.9013769812319090e+00` | `1.8280094754912617e+00` | `2.3976273858151381e-01` | `7.5981644339917755e+00` | `-7.7574037477612573e-02` |
| `transport:alpha_power_volume_average_mw_m3` | `5.7786228569831921e-01` | `2.7421038425018640e-01` | `8.1604646824932223e-02` | `2.3155465380623520e-03` | `2.7875538302773323e-01` | `-1.7350581859371954e-03` |

Matching FD command to generate the missing 2-step geometry reference:

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

### 16-Step FD Table

These are not root-only ambipolarity FD values, but they are the saved
16-step realtime-geometry FD references for the full transport run with
`--initial-Er-root-ad jax_selected_root`.

| Objective | 16-step FD `d/dRBC:1:0` | 16-step reverse AD `d/dRBC:1:0` | `abs diff` | `rel diff` |
| --- | ---: | ---: | ---: | ---: |
| `transport:softmax_Er` | `-6.2627250000000000e+01` | `-6.2626550584360913e+01` | `6.994156e-04` | `1.116791e-05` |
| `transport:smooth_root_proxy` | `-5.5764050000000004e-04` | `-3.7595015613984747e-04` | `1.816903e-04` | `3.258198e-01` |
| `transport:Er2_volume_average` | `-2.7123439999999999e+02` | `-2.7118432937475063e+02` | `5.007063e-02` | `1.846028e-04` |
| `transport:Er_volume_average` | `-2.3644030000000001e+01` | `-2.3646073953442198e+01` | `2.043953e-03` | `8.644691e-05` |
| `transport:electron_temperature_volume_average_keV` | `-2.2448650000000001e-02` | `-2.2448497948865054e-02` | `1.520511e-07` | `6.773286e-06` |
| `transport:total_pressure_volume_average` | `-7.7521010000000001e-02` | `-7.7520440608481067e-02` | `5.693915e-07` | `7.344996e-06` |
| `transport:alpha_power_volume_average_mw_m3` | `1.2073920000000001e-03` | `1.2073916077413038e-03` | `3.922587e-10` | `3.248810e-07` |

This full-transport table should not be used as the FD reference for the
root-only ambipolarity smoke, because the root-only path does not include the
Radau time evolution.

### 16-Step Shared-Payload Smoke Update

Current run:

- `mode = transport_reverse_ad_only_full_transport_shared_payload_smoke`
- `accepted_step_limit = 16`
- `elapsed_s = 3531.475`
- output JSON: `outputs/autodiff_transport_lagged_ntx/reverse_ad/transport_reverse_ad_only_full_transport_shared_payload_smoke.json`

Full shared-payload AD table:

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

Profile-column comparison:

- The full profile-column table for this run is saved in
  `optimization_ad_vs_fd.md` under `16-Step Full-Transport Shared-Payload
  Smoke`.
- Existing saved 16-step profile references are profile-only references, not a
  matching `profiles_plus_realtime_geometry` frozen-FD table.
- Pressure/temperature/alpha profile derivatives agree tightly with those
  references; Er/root-sensitive profile derivatives differ because the current
  run includes realtime geometry/root coupling.

Bootstrap FD update:

- The 16-step full-transport FD run with `jax_selected_root` now includes
  `bootstrap_current_softmax_abs_scaled`.
- The full saved table is in `optimization_ad_vs_fd.md` under
  `16-Step Full-Transport FD With Bootstrap`.
- New FD-only rows awaiting matching full-reverse AD rows:

| Objective | FD `d/dRBC:1:0` |
| --- | ---: |
| `transport:Er_transition_left` | `-2.0130240000000000e+01` |
| `transport:Er_transition_right` | `-2.2349450000000000e+01` |
| `transport:bootstrap_current_softmax_abs_scaled` | `-1.7311170000000000e+00` |
