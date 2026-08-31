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

Geometry rows from the shared-payload table (all profile derivatives were
exactly zero):

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
