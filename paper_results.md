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
