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

The current saved shared-payload root-only smoke values are:

| Objective | Shared-payload value | Shared-payload `d/dRBC:1:0` | Root-only FD status |
| --- | ---: | ---: | --- |
| `transport:softmax_Er` | `2.0479476664720302e+01` | `-5.1293330714789889e+01` | no root-only FD found locally |
| `transport:smooth_root_proxy` | `8.0591612259350128e-11` | `2.2505242252037452e-09` | no root-only FD found locally |
| `transport:Er_transition_left` | `1.7657228878480385e+01` | `-2.0079770095990593e+01` | no root-only FD found locally |
| `transport:Er_transition_right` | `1.8321801837429653e+01` | `-2.2278454387928971e+01` | no root-only FD found locally |
| `transport:Er2_volume_average` | `2.5947838715347029e+02` | `-1.8476397875085149e+02` | no root-only FD found locally |
| `transport:Er_volume_average` | `-3.5568787373760746e+00` | `-2.0622727790532899e+01` | no root-only FD found locally |

I did not find a saved finite-difference table for the root-only ambipolarity
path in the local repo docs/outputs. The FD values currently saved in
`realtime_transport_geometry_reverse_current_state.md` are for the full
16-accepted-step transport evolution, not the root-only ambipolarity objective.

## Existing Full-Transport FD Values

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
