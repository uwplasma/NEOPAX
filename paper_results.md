# Reverse-AD benchmark results

## Realtime VMEC + lagged NTX, full shared payload

Recorded 2026-08-30 from the reverse benchmark with the direct-directional
VMEC coefficient rebuild mode, scalar final cotangents, and the joint-local
bootstrap mode.

- Runtime build: 309.299 s.
- Solver components: 36.113 s.
- Realized-schedule reverse forward: 188.777 s.
- Final-objective cotangents: 54.483 s.
- Reverse-segment GPU compilation alarm: 190.220 s.
- Segmented cotangent sweep: 796.711 s (`support_reuse=7`, `support_rebuild=9`).
- Initial-cache support pullback: 134.653 s.
- Initial-state pullback: 136.643 s.
- Initial-Er root-boundary pullback: 54.016 s.
- Total benchmark elapsed time: 2176.856 s.

## Complete reverse-AD objective table

The parameter columns are the six profile scalars followed by the two VMEC
boundary coefficients. Geometry objectives have an exactly-zero derivative
with respect to every profile scalar.

### Transport objectives

| objective | value | d/n0 | d/T0 | d/density shape power | d/temperature shape power | d/density shape alpha | d/temperature shape alpha | d/RBC:1:0 | d/ZBS:1:0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `transport:softmax_Er` | 2.0509414483715211e+01 | -4.2545379617068706e+00 | 3.6091764765955014e+00 | -9.8361182246815421e-02 | 2.3583918877598946e+00 | 1.8098220510597274e-01 | 1.3494737456808437e+01 | -5.1548105229271457e+01 | 1.4343124491042161e+01 |
| `transport:net_total_power_volume_average_mw_m3` | 5.0358800238037182e-01 | 2.3806149032853224e-01 | 8.1640670066313328e-02 | 9.0869773379130797e-04 | 2.7589452502078771e-01 | 2.0350057071476713e-04 | -3.9304249453946510e-01 | -1.1394217595813690e-02 | 1.8990845726998083e-03 |
| `transport:Er_transition_left` | 1.7731259538315076e+01 | -1.4462240960481969e+00 | 1.8373970682514851e+00 | -1.2779246490158686e-02 | -7.1805547591007581e+00 | 1.5764994644736911e-02 | 1.6102251226120909e+01 | -2.0312089728954220e+01 | 1.5368566615827994e-01 |
| `transport:Er_transition_right` | 1.8377781334972461e+01 | -1.6701201873140783e+00 | 2.0018574455833336e+00 | -1.8180325645076521e-02 | -6.2629652796043800e+00 | 2.3795172597101875e-02 | 1.6294693374129025e+01 | -2.2505721135089395e+01 | 1.0204156159572406e+00 |
| `transport:Er2_volume_average` | 2.3628446892755093e+02 | -7.3275130620973428e+01 | 7.2063394397715996e+01 | 2.0106140455113719e+00 | 7.1797546606802086e+01 | -7.5853909268514528e+00 | 1.4721092513807679e+02 | -7.3686037134677019e+02 | 3.0894251816827875e+02 |
| `transport:Er_volume_average` | -3.4335525053006659e+00 | -5.2104998227401556e-02 | -2.2367460848753223e-01 | -5.2865563517223185e-02 | -3.5678855707251120e+00 | -9.7748389456804308e-03 | 2.6666395545549562e+00 | -4.0596812950350758e+00 | -9.3264399639932272e-02 |
| `transport:electron_temperature_volume_average_keV` | 6.5649815320451319e+00 | 5.1997751512444790e-04 | 3.5541916397162049e-01 | -3.2982545890333781e-05 | 1.5233921499482435e+00 | 6.2113760748155888e-04 | -3.0437674606912695e+00 | -1.3254618319760575e-02 | -4.0333600271927436e-02 |
| `transport:total_pressure_volume_average` | 3.4214058267522695e+01 | 8.0604918426085881e+00 | 1.8648494493034202e+00 | 2.4426191336103756e-01 | 7.7511118436220290e+00 | -1.3265820510914987e+00 | -1.4519173981199284e+01 | -7.2635924492936238e-02 | -2.3766611274561567e-01 |
| `transport:alpha_power_volume_average_mw_m3` | 5.8931817387931074e-01 | 2.7840069077503449e-01 | 8.3968123605256809e-02 | 2.3453720601787327e-03 | 2.8614146039497745e-01 | -7.6099657781782334e-03 | -4.1311478992815931e-01 | -1.1559358488979699e-02 | 1.3974197359855141e-03 |
| `transport:bootstrap_current_softmax_abs_scaled` | 1.4489740594414575e+00 | -1.1599888255404145e-03 | 2.3110056010339713e-01 | -1.3236579671547351e-02 | 1.5251864856735118e+00 | 1.1411504697144575e-01 | -3.5246002196092232e+00 | -1.7723476056699499e+00 | -6.2437599466515081e+00 |

### Geometry objectives

| objective | value | d/dRBC:1:0 | d/dZBS:1:0 |
| --- | ---: | ---: | ---: |
| `geometry:boozer_qi_objective` | 0.21192029964274445 | 5.9392891187805503 | -0.12365550091203659 |
| `geometry:boozer_maxj_objective` | 443.87332574094023 | -3843.1351877037669 | -1920.5082803588884 |
| `geometry:vmec_aspect_ratio` | 10.015330918957178 | -5.4006784187006147 | -5.5226885751318529 |
| `geometry:vmec_iota_mean` | -0.59365259966101458 | 0.24405140609263865 | 0.14567526019820762 |
| `geometry:vmec_magnetic_well` | -0.027476128749679612 | -0.011090116065531674 | -0.04168202748223848 |
| `geometry:vmec_mirror_ratio` | 0.21153803467163693 | -0.59359094714046601 | 0.41437006125401626 |
| `geometry:vmec_dmerc_stability_softmax` | 3.2329612235882421 | -7.2859943466255244 | -1.5178996489713270 |

These are reverse-AD values. The frozen-VMEC finite-difference script now
writes `vmec_dmerc_stability_softmax_value` and
`vmec_dmerc_stability_softmax_gradient_fd` into its JSON report.

## Transport-profile frozen-root FD validation

Both FD runs used the same accepted 16-step replay and
`initial-Er-root-fd-root-lane=frozen_linearized`, matching the reverse rule's
local implicit derivative of the already selected Er root. FD values below
are the values printed by the benchmark (six decimal places); the errors are
therefore conservative at that displayed precision.

| objective | n0 reverse | n0 FD | n0 abs. error | n0 rel. error | temperature-shape-alpha reverse | temperature-shape-alpha FD | alpha abs. error | alpha rel. error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `softmax_Er` | -4.254537962e+00 | -4.254502000e+00 | 3.596e-05 | 8.453e-06 | 1.349473746e+01 | 1.349478000e+01 | 4.254e-05 | 3.153e-06 |
| `net_total_power_volume_average_mw_m3` | 2.380614903e-01 | 2.380615000e-01 | 9.671e-09 | 4.063e-08 | -3.930424945e-01 | -3.930425000e-01 | 5.461e-09 | 1.389e-08 |
| `Er_transition_left` | -1.446224096e+00 | -1.446190000e+00 | 3.410e-05 | 2.358e-05 | 1.610225123e+01 | 1.610224000e+01 | 1.123e-05 | 6.972e-07 |
| `Er_transition_right` | -1.670120187e+00 | -1.670138000e+00 | 1.781e-05 | 1.067e-05 | 1.629469337e+01 | 1.629478000e+01 | 8.663e-05 | 5.316e-06 |
| `Er2_volume_average` | -7.327513062e+01 | -7.327482000e+01 | 3.106e-04 | 4.239e-06 | 1.472109251e+02 | 1.472106000e+02 | 3.251e-04 | 2.209e-06 |
| `Er_volume_average` | -5.210499823e-02 | -5.210322000e-02 | 1.778e-06 | 3.413e-05 | 2.666639555e+00 | 2.666646000e+00 | 6.445e-06 | 2.417e-06 |
| `electron_temperature_volume_average_keV` | 5.199775151e-04 | 5.199775000e-04 | 1.512e-11 | 2.909e-08 | -3.043767461e+00 | -3.043767000e+00 | 4.607e-07 | 1.514e-07 |
| `total_pressure_volume_average` | 8.060491843e+00 | 8.060492000e+00 | 1.574e-07 | 1.953e-08 | -1.451917398e+01 | -1.451917000e+01 | 3.981e-06 | 2.742e-07 |
| `alpha_power_volume_average_mw_m3` | 2.784006908e-01 | 2.784007000e-01 | 9.225e-09 | 3.314e-08 | -4.131147899e-01 | -4.131148000e-01 | 1.007e-08 | 2.438e-08 |
| `bootstrap_current_softmax_abs_scaled` | -1.159988826e-03 | -1.159535000e-03 | 4.538e-07 | 3.912e-04 | -3.524600220e+00 | -3.524605000e+00 | 4.780e-06 | 1.356e-06 |

## RBC:1:0 frozen-geometry FD validation

The earlier frozen-geometry FD implementation used an implicit-VMEC primal
state while reverse AD used the configured forward-VMEC state. Its historical
results are retained below only as an audit record. The corrected lane now
uses the forward state as its zero point and obtains only the raw-block
tangent/structural mask from the implicit system.

### Historical, misaligned baseline (superseded)

| objective | primal abs. mismatch | primal rel. mismatch | reverse AD | frozen geometry FD | abs. error | rel. error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `softmax_Er` | 1.326e-02 | 6.465e-04 | -5.154810523e+01 | -5.102169000e+01 | 5.264e-01 | 1.021e-02 |
| `net_total_power_volume_average_mw_m3` | 1.825e-05 | 3.625e-05 | -1.139421760e-02 | -1.243262000e-02 | 1.038e-03 | 9.113e-02 |
| `Er_transition_left` | 1.094e-03 | 6.171e-05 | -2.031208973e+01 | -2.030807000e+01 | 4.020e-03 | 1.979e-04 |
| `Er_transition_right` | 1.111e-03 | 6.047e-05 | -2.250572114e+01 | -2.250010000e+01 | 5.621e-03 | 2.498e-04 |
| `Er2_volume_average` | 5.831e-01 | 2.468e-03 | -7.368603713e+02 | -7.515035000e+02 | 1.464e+01 | 1.987e-02 |
| `Er_volume_average` | 1.691e-04 | 4.925e-05 | -4.059681295e+00 | -2.216080000e+01 | 1.810e+01 | 4.459e+00 |
| `electron_temperature_volume_average_keV` | 2.422e-04 | 3.689e-05 | -1.325461832e-02 | -1.311081000e-02 | 1.438e-04 | 1.085e-02 |
| `total_pressure_volume_average` | 4.208e-04 | 1.230e-05 | -7.263592449e-02 | -7.309052000e-02 | 4.546e-04 | 6.259e-03 |
| `alpha_power_volume_average_mw_m3` | 1.707e-05 | 2.897e-05 | -1.155935849e-02 | -1.259516000e-02 | 1.036e-03 | 8.961e-02 |
| `bootstrap_current_softmax_abs_scaled` | 3.043e-04 | 2.100e-04 | -1.772347606e+00 | -1.768463000e+00 | 3.885e-03 | 2.192e-03 |

### Corrected forward-state baseline

The corrected FD primal values agree with the reverse benchmark baseline to
at worst `1.1e-08` relative (`Er2_volume_average`); all other primal relative
mismatches are below `3.6e-09`. The FD values below are printed to six decimal
places, so the listed errors are conservative at that precision.

| objective | reverse AD d/RBC:1:0 | corrected frozen FD | abs. error | rel. error |
| --- | ---: | ---: | ---: | ---: |
| `softmax_Er` | -5.154810523e+01 | -5.154823000e+01 | 1.248e-04 | 2.420e-06 |
| `net_total_power_volume_average_mw_m3` | -1.139421760e-02 | -1.139414000e-02 | 7.760e-08 | 6.810e-06 |
| `Er_transition_left` | -2.031208973e+01 | -2.031183000e+01 | 2.597e-04 | 1.279e-05 |
| `Er_transition_right` | -2.250572114e+01 | -2.250562000e+01 | 1.011e-04 | 4.494e-06 |
| `Er2_volume_average` | -7.368603713e+02 | -7.404802000e+02 | 3.620e+00 | 4.913e-03 |
| `Er_volume_average` | -4.059681295e+00 | -3.995130000e+00 | 6.455e-02 | 1.590e-02 |
| `electron_temperature_volume_average_keV` | -1.325461832e-02 | -1.353714000e-02 | 2.825e-04 | 2.132e-02 |
| `total_pressure_volume_average` | -7.263592449e-02 | -7.300137000e-02 | 3.654e-04 | 5.031e-03 |
| `alpha_power_volume_average_mw_m3` | -1.155935849e-02 | -1.155964000e-02 | 2.815e-07 | 2.435e-05 |
| `bootstrap_current_softmax_abs_scaled` | -1.772347606e+00 | -1.772034000e+00 | 3.136e-04 | 1.769e-04 |

## wHe realtime-geometry frozen-linearized FD results

Recorded from `Solve_Transport_equations_wHe_radau_ntx_exact_lagged_runtime_vmec_realtime_geometry_benchmark.toml`
with the accepted 16-step replay, frozen-linearized geometry lane, and
frozen-linearized selected-root lane. The four-species momentum-correction
selector fix is included. The pasted run completed the first three profile
parameters; the remaining loop entries are intentionally not inferred here.

Baseline objective values:

| objective | value |
| --- | ---: |
| `softmax_Er` | 2.0480070668676280e+01 |
| `net_total_power_volume_average_mw_m3` | 5.0826535626276426e-01 |
| `Er_transition_left` | 1.7728042470417037e+01 |
| `Er_transition_right` | 1.8374574669825218e+01 |
| `Er2_volume_average` | 2.3861229649415688e+02 |
| `Er_volume_average` | -3.4490162721552524e+00 |
| `electron_temperature_volume_average_keV` | 6.5641845786291553e+00 |
| `total_pressure_volume_average` | 3.4212729678180352e+01 |
| `alpha_power_volume_average_mw_m3` | 5.8937225590716658e-01 |
| `bootstrap_current_softmax_abs_scaled` | 1.4479788427078408e+00 |

| objective | d/n0 FD | d/T0 FD | d/density-shape-power FD |
| --- | ---: | ---: | ---: |
| `softmax_Er` | -4.405941e+00 | 3.693007e+00 | -9.766809e-02 |
| `net_total_power_volume_average_mw_m3` | 2.408879e-01 | 8.140385e-02 | 9.949851e-04 |
| `Er_transition_left` | -1.447162e+00 | 1.837081e+00 | -1.280441e-02 |
| `Er_transition_right` | -1.670833e+00 | 2.001425e+00 | -1.821576e-02 |
| `Er2_volume_average` | -1.136820e-01 | 2.953277e+01 | 2.908472e+00 |
| `Er_volume_average` | -4.691384e+00 | 2.484746e+00 | -1.133473e-01 |
| `electron_temperature_volume_average_keV` | 2.627422e-04 | 3.553105e-01 | -1.731593e-05 |
| `total_pressure_volume_average` | 8.059606e+00 | 1.864677e+00 | 2.442594e-01 |
| `alpha_power_volume_average_mw_m3` | 2.790502e-01 | 8.360589e-02 | 2.354167e-03 |
| `bootstrap_current_softmax_abs_scaled` | -9.391406e-04 | 2.307464e-01 | -1.320395e-02 |
