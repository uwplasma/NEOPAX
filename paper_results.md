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

### wHe reverse AD versus frozen-linearized FD

The reverse run used the selected-root compact pullback after aligning its
state preparation with the forward local particle-flux evaluator (configured
species floors, fixed-species projection, and center-gradient construction).
The FD values above are printed to six decimal places, so the reported errors
are conservative with respect to the unprinted FD precision.

#### Reverse-AD transport derivatives

| objective | d/n0 | d/T0 | d/density-shape-power | d/temperature-shape-power | d/density-shape-alpha | d/temperature-shape-alpha | d/RBC:1:0 | d/ZBS:1:0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `softmax_Er` | -4.405970182e+00 | 3.693005393e+00 | -9.764981822e-02 | 2.326826072e+00 | 1.781345735e-01 | 1.381981115e+01 | -5.161877072e+01 | 1.544093574e+01 |
| `net_total_power_volume_average_mw_m3` | 2.408879193e-01 | 8.140384894e-02 | 9.949818099e-04 | 2.756007771e-01 | -2.352270801e-04 | -3.942156986e-01 | -6.528865665e-03 | -2.280228812e-03 |
| `Er_transition_left` | -1.447091223e+00 | 1.837078552e+00 | -1.280427940e-02 | -7.180969420e+00 | 1.581278314e-02 | 1.610114938e+01 | -2.029714231e+01 | 1.538368183e-01 |
| `Er_transition_right` | -1.670881677e+00 | 2.001436550e+00 | -1.820510244e-02 | -6.263604560e+00 | 2.384770614e-02 | 1.629422742e+01 | -2.248753599e+01 | 1.020231456e+00 |
| `Er2_volume_average` | -1.137440200e-01 | 2.953282497e+01 | 2.908458025e+00 | -3.304416001e+01 | -7.914954885e+00 | 1.501751046e+02 | -1.370294385e+02 | -1.710580936e+02 |
| `Er_volume_average` | -4.691408779e+00 | 2.484760954e+00 | -1.133619545e-01 | 3.080282794e+00 | 8.113053755e-02 | 2.623662925e+00 | -4.236791466e+01 | 3.015190408e+01 |
| `electron_temperature_volume_average_keV` | 2.627427320e-04 | 3.553105415e-01 | -1.731549853e-05 | 1.522883654e+00 | 3.187910548e-04 | -3.043178743e+00 | -1.297160075e-02 | -3.997608597e-02 |
| `total_pressure_volume_average` | 8.059606042e+00 | 1.864676793e+00 | 2.442593657e-01 | 7.750766053e+00 | -1.326611062e+00 | -1.452040307e+01 | -7.360244916e-02 | -2.373430136e-01 |
| `alpha_power_volume_average_mw_m3` | 2.790502063e-01 | 8.360589676e-02 | 2.354163306e-03 | 2.852953771e-01 | -7.627225877e-03 | -4.132077181e-01 | -6.686679722e-03 | -2.750729344e-03 |
| `bootstrap_current_softmax_abs_scaled` | -9.388484465e-04 | 2.307460298e-01 | -1.320397873e-02 | 1.523316057e+00 | 1.139955494e-01 | -3.521755735e+00 | -1.760376607e+00 | -6.237748905e+00 |

| objective | n0 reverse | n0 FD | n0 abs. error | n0 rel. error | T0 reverse | T0 FD | T0 abs. error | T0 rel. error | density-power reverse | density-power FD | density-power abs. error | density-power rel. error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `softmax_Er` | -4.405970182e+00 | -4.405941e+00 | 2.918e-05 | 6.623e-06 | 3.693005393e+00 | 3.693007e+00 | 1.607e-06 | 4.351e-07 | -9.764981822e-02 | -9.766809e-02 | 1.827e-05 | 1.871e-04 |
| `net_total_power_volume_average_mw_m3` | 2.408879193e-01 | 2.408879e-01 | 1.931e-08 | 8.018e-08 | 8.140384894e-02 | 8.140385e-02 | 1.063e-09 | 1.306e-08 | 9.949818099e-04 | 9.949851e-04 | 3.290e-09 | 3.307e-06 |
| `Er_transition_left` | -1.447091223e+00 | -1.447162e+00 | 7.078e-05 | 4.891e-05 | 1.837078552e+00 | 1.837081e+00 | 2.448e-06 | 1.333e-06 | -1.280427940e-02 | -1.280441e-02 | 1.306e-07 | 1.020e-05 |
| `Er_transition_right` | -1.670881677e+00 | -1.670833e+00 | 4.868e-05 | 2.913e-05 | 2.001436550e+00 | 2.001425e+00 | 1.155e-05 | 5.771e-06 | -1.820510244e-02 | -1.821576e-02 | 1.066e-05 | 5.851e-04 |
| `Er2_volume_average` | -1.137440200e-01 | -1.136820e-01 | 6.202e-05 | 5.456e-04 | 2.953282497e+01 | 2.953277e+01 | 5.497e-05 | 1.861e-06 | 2.908458025e+00 | 2.908472e+00 | 1.398e-05 | 4.805e-06 |
| `Er_volume_average` | -4.691408779e+00 | -4.691384e+00 | 2.478e-05 | 5.282e-06 | 2.484760954e+00 | 2.484746e+00 | 1.495e-05 | 6.018e-06 | -1.133619545e-01 | -1.133473e-01 | 1.465e-05 | 1.293e-04 |
| `electron_temperature_volume_average_keV` | 2.627427320e-04 | 2.627422e-04 | 5.320e-10 | 2.025e-06 | 3.553105415e-01 | 3.553105e-01 | 4.149e-08 | 1.168e-07 | -1.731549853e-05 | -1.731593e-05 | 4.315e-10 | 2.492e-05 |
| `total_pressure_volume_average` | 8.059606042e+00 | 8.059606e+00 | 4.173e-08 | 5.178e-09 | 1.864676793e+00 | 1.864677e+00 | 2.069e-07 | 1.109e-07 | 2.442593657e-01 | 2.442594e-01 | 3.428e-08 | 1.403e-07 |
| `alpha_power_volume_average_mw_m3` | 2.790502063e-01 | 2.790502e-01 | 6.276e-09 | 2.249e-08 | 8.360589676e-02 | 8.360589e-02 | 6.758e-09 | 8.083e-08 | 2.354163306e-03 | 2.354167e-03 | 3.694e-09 | 1.569e-06 |
| `bootstrap_current_softmax_abs_scaled` | -9.388484465e-04 | -9.391406e-04 | 2.922e-07 | 3.111e-04 | 2.307460298e-01 | 2.307464e-01 | 3.702e-07 | 1.604e-06 | -1.320397873e-02 | -1.320395e-02 | 2.873e-08 | 2.176e-06 |

#### Remaining profile and geometry parameter comparisons

The following FD results were produced by the same wHe accepted-schedule,
frozen-linearized geometry/root lane as the preceding table.  `rel. error` is
`|reverse - FD| / |FD|`; it is quoted from the printed six-decimal FD values.

##### `temperature_shape_power`

| objective | reverse AD | FD | abs. error | rel. error |
| --- | ---: | ---: | ---: | ---: |
| `softmax_Er` | 2.326826072e+00 | 2.326775e+00 | 5.107e-05 | 2.195e-05 |
| `net_total_power_volume_average_mw_m3` | 2.756007771e-01 | 2.756008e-01 | 2.286e-08 | 8.294e-08 |
| `Er_transition_left` | -7.180969420e+00 | -7.180946e+00 | 2.342e-05 | 3.261e-06 |
| `Er_transition_right` | -6.263604560e+00 | -6.263527e+00 | 7.756e-05 | 1.238e-05 |
| `Er2_volume_average` | -3.304416001e+01 | -3.304452e+01 | 3.600e-04 | 1.089e-05 |
| `Er_volume_average` | 3.080282794e+00 | 3.080242e+00 | 4.079e-05 | 1.324e-05 |
| `electron_temperature_volume_average_keV` | 1.522883654e+00 | 1.522884e+00 | 3.465e-07 | 2.275e-07 |
| `total_pressure_volume_average` | 7.750766053e+00 | 7.750766e+00 | 5.367e-08 | 6.924e-09 |
| `alpha_power_volume_average_mw_m3` | 2.852953771e-01 | 2.852954e-01 | 2.294e-08 | 8.041e-08 |
| `bootstrap_current_softmax_abs_scaled` | 1.523316057e+00 | 1.523310e+00 | 6.057e-06 | 3.976e-06 |

##### `density_shape_alpha`

| objective | reverse AD | FD | abs. error | rel. error |
| --- | ---: | ---: | ---: | ---: |
| `softmax_Er` | 1.781345735e-01 | 1.780592e-01 | 7.537e-05 | 4.233e-04 |
| `net_total_power_volume_average_mw_m3` | -2.352270801e-04 | -2.352239e-04 | 3.180e-09 | 1.352e-05 |
| `Er_transition_left` | 1.581278314e-02 | 1.578012e-02 | 3.266e-05 | 2.070e-03 |
| `Er_transition_right` | 2.384770614e-02 | 2.393388e-02 | 8.617e-05 | 3.600e-03 |
| `Er2_volume_average` | -7.914954885e+00 | -7.915027e+00 | 7.211e-05 | 9.111e-06 |
| `Er_volume_average` | 8.113053755e-02 | 8.114724e-02 | 1.670e-05 | 2.058e-04 |
| `electron_temperature_volume_average_keV` | 3.187910548e-04 | 3.187894e-04 | 1.655e-09 | 5.191e-06 |
| `total_pressure_volume_average` | -1.326611062e+00 | -1.326611e+00 | 6.244e-08 | 4.706e-08 |
| `alpha_power_volume_average_mw_m3` | -7.627225877e-03 | -7.627223e-03 | 2.877e-09 | 3.772e-07 |
| `bootstrap_current_softmax_abs_scaled` | 1.139955494e-01 | 1.139911e-01 | 4.449e-06 | 3.903e-05 |

##### `temperature_shape_alpha`

| objective | reverse AD | FD | abs. error | rel. error |
| --- | ---: | ---: | ---: | ---: |
| `softmax_Er` | 1.381981115e+01 | 1.381965e+01 | 1.611e-04 | 1.166e-05 |
| `net_total_power_volume_average_mw_m3` | -3.942156986e-01 | -3.942157e-01 | 1.358e-09 | 3.446e-09 |
| `Er_transition_left` | 1.610114938e+01 | 1.610097e+01 | 1.794e-04 | 1.114e-05 |
| `Er_transition_right` | 1.629422742e+01 | 1.629420e+01 | 2.742e-05 | 1.683e-06 |
| `Er2_volume_average` | 1.501751046e+02 | 1.501754e+02 | 2.954e-04 | 1.967e-06 |
| `Er_volume_average` | 2.623662925e+00 | 2.623741e+00 | 7.807e-05 | 2.976e-05 |
| `electron_temperature_volume_average_keV` | -3.043178743e+00 | -3.043179e+00 | 2.569e-07 | 8.441e-08 |
| `total_pressure_volume_average` | -1.452040307e+01 | -1.452040e+01 | 3.066e-06 | 2.111e-07 |
| `alpha_power_volume_average_mw_m3` | -4.132077181e-01 | -4.132077e-01 | 1.809e-08 | 4.377e-08 |
| `bootstrap_current_softmax_abs_scaled` | -3.521755735e+00 | -3.521756e+00 | 2.645e-07 | 7.511e-08 |

##### `RBC:1:0`

| objective | reverse AD | FD | abs. error | rel. error |
| --- | ---: | ---: | ---: | ---: |
| `softmax_Er` | -5.161877072e+01 | -5.161912e+01 | 3.493e-04 | 6.767e-06 |
| `net_total_power_volume_average_mw_m3` | -6.528865665e-03 | -5.767493e-03 | 7.614e-04 | 1.320e-01 |
| `Er_transition_left` | -2.029714231e+01 | -2.029573e+01 | 1.412e-03 | 6.959e-05 |
| `Er_transition_right` | -2.248753599e+01 | -2.248608e+01 | 1.456e-03 | 6.475e-05 |
| `Er2_volume_average` | -1.370294385e+02 | -9.018049e+01 | 4.685e+01 | 5.195e-01 |
| `Er_volume_average` | -4.236791466e+01 | -4.252461e+01 | 1.567e-01 | 3.685e-03 |
| `electron_temperature_volume_average_keV` | -1.297160075e-02 | -1.312590e-02 | 1.543e-04 | 1.176e-02 |
| `total_pressure_volume_average` | -7.360244916e-02 | -7.383215e-02 | 2.297e-04 | 3.111e-03 |
| `alpha_power_volume_average_mw_m3` | -6.686679722e-03 | -5.925965e-03 | 7.607e-04 | 1.284e-01 |
| `bootstrap_current_softmax_abs_scaled` | -1.760376607e+00 | -1.759171e+00 | 1.206e-03 | 6.853e-04 |

##### `ZBS:1:0`

| objective | reverse AD | FD | abs. error | rel. error |
| --- | ---: | ---: | ---: | ---: |
| `softmax_Er` | 1.544093574e+01 | 1.547149e+01 | 3.055e-02 | 1.975e-03 |
| `net_total_power_volume_average_mw_m3` | -2.280228812e-03 | -2.903250e-03 | 6.230e-04 | 2.146e-01 |
| `Er_transition_left` | 1.538368183e-01 | 1.537210e-01 | 1.158e-04 | 7.534e-04 |
| `Er_transition_right` | 1.020231456e+00 | 1.020293e+00 | 6.154e-05 | 6.032e-05 |
| `Er2_volume_average` | -1.710580936e+02 | -2.092590e+02 | 3.820e+01 | 1.826e-01 |
| `Er_volume_average` | 3.015190408e+01 | 3.030914e+01 | 1.572e-01 | 5.188e-03 |
| `electron_temperature_volume_average_keV` | -3.997608597e-02 | -4.003870e-02 | 6.261e-05 | 1.564e-03 |
| `total_pressure_volume_average` | -2.373430136e-01 | -2.374806e-01 | 1.376e-04 | 5.794e-04 |
| `alpha_power_volume_average_mw_m3` | -2.750729344e-03 | -3.373287e-03 | 6.226e-04 | 1.846e-01 |
| `bootstrap_current_softmax_abs_scaled` | -6.237748905e+00 | -6.237500e+00 | 2.489e-04 | 3.990e-05 |

### Superseding wHe geometry comparison after initial-root He-floor correction

The preceding wHe `RBC:1:0` and `ZBS:1:0` geometry tables were obtained
before the initial ambipolar-root reconstruction preserved the configured He
temperature at the density floor.  They are superseded by this comparison.
The normal forward state has always used
`pressure = configured_temperature * max(density, density_floor)`; the
reverse and frozen-root FD reconstruction now use that same state.

The values below are from the corrected reverse shared-payload run and the
corrected accepted-schedule frozen-linearized FD runs.  FD is printed to six
decimal digits by the benchmark, so the small reported errors are conservative
at that output precision.

| objective | RBC reverse AD | RBC FD | RBC rel. error | ZBS reverse AD | ZBS FD | ZBS rel. error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `softmax_Er` | -5.161877696e+01 | -5.161888e+01 | 1.996e-06 | 1.544093365e+01 | 1.544048e+01 | 2.938e-05 |
| `net_total_power_volume_average_mw_m3` | -6.527003888e-03 | -6.527018e-03 | 2.162e-06 | -2.281685584e-03 | -2.281675e-03 | 4.639e-06 |
| `Er_transition_left` | -2.029715886e+01 | -2.029704e+01 | 5.856e-06 | 1.538496870e-01 | 1.539699e-01 | 7.808e-04 |
| `Er_transition_right` | -2.248755529e+01 | -2.248736e+01 | 8.685e-06 | 1.020246312e+00 | 1.019976e+00 | 2.650e-04 |
| `Er2_volume_average` | -1.370697603e+02 | -1.370601e+02 | 7.048e-05 | -1.710248368e+02 | -1.710214e+02 | 2.010e-05 |
| `Er_volume_average` | -4.236329294e+01 | -4.236430e+01 | 2.377e-05 | 3.014820897e+01 | 3.014751e+01 | 2.318e-05 |
| `electron_temperature_volume_average_keV` | -1.297161609e-02 | -1.311549e-02 | 1.097e-02 | -3.997609997e-02 | -4.010611e-02 | 3.242e-03 |
| `total_pressure_volume_average` | -7.360239022e-02 | -7.378432e-02 | 2.466e-03 | -2.373430410e-01 | -2.375081e-01 | 6.950e-04 |
| `alpha_power_volume_average_mw_m3` | -6.684817932e-03 | -6.684999e-03 | 2.709e-05 | -2.752186254e-03 | -2.752328e-03 | 5.150e-05 |
| `bootstrap_current_softmax_abs_scaled` | -1.760377166e+00 | -1.760067e+00 | 1.762e-04 | -6.237749176e+00 | -6.237928e+00 | 2.867e-05 |
