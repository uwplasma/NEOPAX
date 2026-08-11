# Shared-Payload 8-Parameter Benchmark Snapshot

Date saved: 2026-08-06

This file records the completed full-transport shared-payload reverse-AD run
with the expanded optimization parameter set.

This is a new benchmark snapshot, not a replacement for the older validated
5-parameter table in `optimization_ad_vs_fd.md`. The current run uses two
additional profile shape-alpha parameters and two VMEC harmonics.

## Run

Command shape:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --reverse-parameter-mode profiles_plus_realtime_geometry \
  --reverse-geometry-parameter RBC:1:0,ZBS:1:0 \
  --realtime-geometry-gradient-path reverse_payload \
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

Run summary:

- Mode: `transport_reverse_ad_only_full_transport_shared_payload_smoke`
- Residual count: `16`
- Parameter count: `8`
- Elapsed time: `8608.422 s`
- Output: `outputs/autodiff_transport_lagged_ntx/reverse_ad/transport_reverse_ad_only_full_transport_shared_payload_smoke.json`
- VMEC input from TOML: `examples/inputs/input.QI_nfp2_newNT_opt_hires_true`

Parameter order:

```text
n0, T0, density_shape_power, temperature_shape_power,
density_shape_alpha, temperature_shape_alpha,
vmec:RBC:1:0, vmec:ZBS:1:0
```

## Reverse-AD Rows

| Objective | Value | `d/dn0` | `d/dT0` | `d/ddensity_shape_power` | `d/dtemperature_shape_power` | `d/ddensity_shape_alpha` | `d/dtemperature_shape_alpha` | `d/dRBC:1:0` | `d/dZBS:1:0` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `transport:softmax_Er` | `2.1901984427966120e+01` | `-4.6212827650037696e+00` | `3.9899792308177973e+00` | `-1.1665473683809535e-01` | `4.8121590689444522e+00` | `2.2525392047893905e-01` | `1.0137748536682931e+01` | `-7.1867199080799821e+01` | `1.6172173041767433e+01` |
| `transport:smooth_root_proxy` | `2.0901587849165550e-02` | `-6.0823713405263113e-01` | `-4.9871959787244902e-01` | `-6.1437030346753012e-06` | `8.5099566900273249e+00` | `-5.6509938477803398e-04` | `-2.6510247645032159e+00` | `9.9663647868526617e+00` | `-6.6378940377612192e+00` |
| `transport:Er_transition_left` | `1.7820463773802203e+01` | `-1.3817755279481831e+00` | `1.8199455732321661e+00` | `-1.2386749994381562e-02` | `-7.1879403780524322e+00` | `1.4823877035899186e-02` | `1.6072613089458446e+01` | `-2.0026802627011239e+01` | `-1.2430485734500962e-01` |
| `transport:Er_transition_right` | `1.8476964719552548e+01` | `-1.5882154154317263e+00` | `1.9833743851540413e+00` | `-1.8253773594419265e-02` | `-6.2660532892013423e+00` | `2.4051119544514693e-02` | `1.6240608263891964e+01` | `-2.2230577867179537e+01` | `7.2173907384926217e-01` |
| `transport:Er2_volume_average` | `2.3828852492934604e+02` | `1.0298356484267750e+01` | `3.8944243985017621e+01` | `1.4333172211656180e+00` | `-4.8582324521614524e+00` | `-1.9807052926495658e+01` | `1.4493497364592423e+02` | `-2.6158459767312769e+02` | `-3.3170453206070420e+01` |
| `transport:Er_volume_average` | `-3.4814658282471411e+00` | `-4.2305636653172307e+00` | `2.2316726986613764e+00` | `-1.1797200287468337e-01` | `2.7398931073105182e+00` | `3.4434359939826731e-01` | `1.8554656238649001e+00` | `-4.4192055552946904e+01` | `2.7639338929538802e+01` |
| `transport:electron_temperature_volume_average_keV` | `6.5783684429785927e+00` | `3.9771962240737757e-03` | `3.5770053782616734e-01` | `-5.0721363274344386e-04` | `1.5326114138430027e+00` | `9.1662248023662574e-03` | `-3.0487982929834874e+00` | `-2.6868973427082284e-02` | `-4.1892082915762047e-02` |
| `transport:total_pressure_volume_average` | `3.4221494926657719e+01` | `8.0687017688186558e+00` | `1.8660888776355311e+00` | `2.4425163061691829e-01` | `7.7476635761315009e+00` | `-1.3250470862147909e+00` | `-1.4471598970710694e+01` | `-6.7796644212052712e-02` | `-2.3527654997475150e-01` |
| `transport:alpha_power_volume_average_mw_m3` | `5.8749986137419818e-01` | `2.7779032056142522e-01` | `8.3492837636644834e-02` | `2.3339725743700859e-03` | `2.8410103130252184e-01` | `-7.5735802753544501e-03` | `-4.1078883280670081e-01` | `-5.4720139220102168e-03` | `6.5960088901149787e-04` |
| `transport:bootstrap_current_softmax_abs_scaled` | `1.4592523038897500e+00` | `-4.8170940991476008e-03` | `2.3506257897750871e-01` | `-1.3685149619976383e-02` | `1.5443016164842926e+00` | `1.1530973728989169e-01` | `-3.5542122055113707e+00` | `-1.9484495925060890e+00` | `-6.3031611594827819e+00` |
| `geometry:boozer_qi_objective` | `2.1192029797323711e-01` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `5.9392927523725234e+00` | `-1.2365499862085017e-01` |
| `geometry:boozer_maxj_objective` | `4.4387332711513375e+02` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `-3.8431400490597589e+03` | `-1.9205081921396923e+03` |
| `geometry:vmec_aspect_ratio` | `1.0015330918957178e+01` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `-5.4006784187006147e+00` | `-5.5226885751318529e+00` |
| `geometry:vmec_iota_mean` | `-5.9365259946730364e-01` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `2.4405140677736042e-01` | `1.4567526055774602e-01` |
| `geometry:vmec_magnetic_well` | `-2.7476128719881053e-02` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `-1.1090112492865956e-02` | `-4.1682026469118227e-02` |
| `geometry:vmec_mirror_ratio` | `2.1153803467611412e-01` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `-5.9359094689600833e-01` | `4.1437006136733534e-01` |

## Notes For Next Validation

- The older 5-parameter 16-step shared-payload table should not be used as a
  direct numeric reference for this run.
- This snapshot includes `density_shape_alpha` and
  `temperature_shape_alpha`, and also includes `ZBS:1:0`.
- The geometry objectives `boozer_qi_objective` and `boozer_maxj_objective`
  are recorded here as least-squares residual rows from this benchmark output.
  Their raw frozen-linearized geometry FD should be rerun before using them as
  a physics/objective validation reference.
- Pure VMEC rows (`aspect_ratio`, `iota_mean`, `magnetic_well`,
  `mirror_ratio`) remain consistent with the older `RBC:1:0` derivatives, but
  `ZBS:1:0` still needs matching frozen-linearized FD rows if we want an FD
  comparison table.

## Geometry Frozen-FD Validation

### `boozer_qi_objective`, `RBC:1:0`

Current effective geometry benchmark result:

```bash
python ./examples/benchmarks/compare_geometry_qi_frozen_linearized_fd.py \
  --vmec-input ./examples/inputs/input.QI_nfp2_newNT_opt_hires_true \
  --parameter RBC:1:0 \
  --objective boozer_qi_objective \
  --multigrid \
  --forward-linear-solve-mode raw_block \
  --forward-linear-maxiter 300 \
  --adjoint-maxiter 300
```

| Quantity | Value |
| --- | ---: |
| Baseline objective value | `2.1192029964274489e-01` |
| Frozen-linearized FD `d/dRBC:1:0` | `5.9397387449796542e+00` |
| Forward JVP `d/dRBC:1:0` | `5.9392891189187216e+00` |
| Optimization-internal raw-block reverse `d/dRBC:1:0` | `5.9392891187135319e+00` |
| Shared-payload reverse row `d/dRBC:1:0` | `5.9392927523725234e+00` |

| Comparison | Abs diff | Rel diff |
| --- | ---: | ---: |
| Shared-payload reverse vs frozen-linearized FD | `4.4599260713074074e-04` | `7.5086232960616260e-05` |
| Shared-payload reverse vs forward JVP | `3.6334538018323315e-06` | `6.1176577349274090e-07` |
| Shared-payload reverse vs optimization-internal raw-block reverse | `3.6336589914753860e-06` | `6.1180032135941000e-07` |
| Optimization-internal raw-block reverse vs forward JVP | `2.0518964305438203e-10` | `3.4547845532688575e-11` |

Status:

- The FD value is now updated for the current QI objective.
- The standalone optimization-internal raw-block reverse matches the forward
  JVP target to `3.5e-11` relative.
- The full shared-payload reverse row is within `6.2e-07` relative of the
  standalone raw-block/JVP value and within the expected frozen-linearized FD
  level compared with FD.

### `boozer_qi_objective`, `ZBS:1:0`

Current effective geometry benchmark result:

```bash
python ./examples/benchmarks/compare_geometry_qi_frozen_linearized_fd.py \
  --vmec-input ./examples/inputs/input.QI_nfp2_newNT_opt_hires_true \
  --parameter ZBS:1:0 \
  --objective boozer_qi_objective \
  --multigrid \
  --forward-linear-solve-mode raw_block \
  --forward-linear-maxiter 300 \
  --adjoint-maxiter 300
```

| Quantity | Value |
| --- | ---: |
| Baseline objective value | `2.1192029964274489e-01` |
| Frozen-linearized FD `d/dZBS:1:0` | `-1.2367491295362752e-01` |
| Forward JVP `d/dZBS:1:0` | `-1.2365550023172692e-01` |
| Optimization-internal raw-block reverse `d/dZBS:1:0` | `-1.2365550092181365e-01` |
| Shared-payload reverse row `d/dZBS:1:0` | `-1.2365499862085017e-01` |

| Comparison | Abs diff | Rel diff |
| --- | ---: | ---: |
| Shared-payload reverse vs frozen-linearized FD | `1.9914332777357102e-05` | `1.6102160334508642e-04` |
| Shared-payload reverse vs forward JVP | `5.0161087675193450e-07` | `4.0565189240424390e-06` |
| Shared-payload reverse vs optimization-internal raw-block reverse | `5.0230096348968800e-07` | `4.0620996214902625e-06` |
| Optimization-internal raw-block reverse vs forward JVP | `6.9008673775350360e-10` | `5.5807201172636930e-09` |

Status:

- The FD value is now saved for the current QI objective with `ZBS:1:0`.
- The standalone optimization-internal raw-block reverse matches the forward
  JVP target to `5.6e-09` relative.
- The full shared-payload reverse row is within `4.1e-06` relative of the
  forward JVP/internal raw-block value and within the expected
  frozen-linearized FD level compared with FD.

### `boozer_maxj_objective`, `RBC:1:0`

Current effective geometry benchmark result:

```bash
python ./examples/benchmarks/compare_geometry_qi_frozen_linearized_fd.py \
  --vmec-input ./examples/inputs/input.QI_nfp2_newNT_opt_hires_true \
  --parameter RBC:1:0 \
  --objective boozer_maxj_objective \
  --multigrid \
  --forward-linear-solve-mode raw_block \
  --forward-linear-maxiter 300 \
  --adjoint-maxiter 300
```

| Quantity | Value |
| --- | ---: |
| Baseline objective value | `4.4387332574093733e+02` |
| Frozen-linearized FD `d/dRBC:1:0` | `-3.8425374640355458e+03` |
| Forward JVP `d/dRBC:1:0` | `-3.8431351880877260e+03` |
| Optimization-internal raw-block reverse `d/dRBC:1:0` | `-3.8431351877333364e+03` |
| Shared-payload reverse row `d/dRBC:1:0` | `-3.8431400490597589e+03` |

| Comparison | Abs diff | Rel diff |
| --- | ---: | ---: |
| Shared-payload reverse vs frozen-linearized FD | `6.0258502421311280e-01` | `1.5681955735058996e-04` |
| Shared-payload reverse vs forward JVP | `4.8609720329295670e-03` | `1.2648454438961066e-06` |
| Shared-payload reverse vs optimization-internal raw-block reverse | `4.8613264225423340e-03` | `1.2649376576860733e-06` |
| Optimization-internal raw-block reverse vs forward JVP | `3.5438961276668124e-07` | `9.2213673322019940e-11` |

Status:

- The FD value is now updated for the current maxJ objective.
- The standalone optimization-internal raw-block reverse matches the forward
  JVP target to `9.2e-11` relative.
- The full shared-payload reverse row is within `1.3e-06` relative of the
  standalone raw-block/JVP value and within the expected frozen-linearized FD
  level compared with FD.

### `boozer_maxj_objective`, `ZBS:1:0`

Current effective geometry benchmark result:

```bash
python ./examples/benchmarks/compare_geometry_qi_frozen_linearized_fd.py \
  --vmec-input ./examples/inputs/input.QI_nfp2_newNT_opt_hires_true \
  --parameter ZBS:1:0 \
  --objective boozer_maxj_objective \
  --multigrid \
  --forward-linear-solve-mode raw_block \
  --forward-linear-maxiter 300 \
  --adjoint-maxiter 300
```

## Previous Timing Reference: 8 Accepted Steps

Saved from the pasted terminal run on 2026-08-07. This is the full-transport
shared-payload smoke benchmark with 8 accepted Radau steps, 2-step reverse
segments, profile parameters plus `RBC:1:0` and `ZBS:1:0`.

```bash
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --reverse-parameter-mode profiles_plus_realtime_geometry \
  --reverse-geometry-parameter RBC:1:0,ZBS:1:0 \
  --realtime-geometry-gradient-path reverse_payload \
  --objective all \
  --accepted-step-limit 8 \
  --radau-jacobian-reuse-mode legacy \
  --timing-mode jit-warm \
  --reverse-segment-length 2 \
  --reverse-stage-adjoint-solve-mode bicgstab \
  --reverse-rhs-transpose-mode explicit_ntx_interpolated \
  --reverse-step-bwd-mode reduced_cotangent \
  --initial-Er-root-ad jax_selected_root \
  --full-transport-shared-payload-smoke
```

| Phase | Elapsed [s] |
| --- | ---: |
| Realtime geometry runtime build | `221.206` |
| Realtime geometry solver components | `31.127` |
| Support reverse profile-state VJP | `32.441` |
| Support reverse initial-carry VJP | `21.591` |
| Support reverse realized-schedule VJP forward | `170.647` |
| Support reverse segmented cotangent sweep | `623.629` |
| Initial-Er root boundary compact pullback | `50.171` |
| Geometry objective VMEC implicit/raw-block aux | `0.009` |
| Geometry objective Boozer input tables | `0.511` |
| Geometry objective Boozer VJP | `1.981` |
| Geometry objective VMEC cotangents | `8.063` |
| Geometry objective Boozer light cotangents | `0.576` |
| Geometry objective aspect proxy cotangents | `0.059` |
| Geometry objective J-QI/maxJ Boozer cotangents | `7.780` |
| Geometry objective Boozer cotangents to state | `6.727` |
| Geometry objective final VMEC parameter pullback | `14.088` |
| Total reported mode elapsed | `1895.859` |

Segmented cotangent sweep details:

| Segment | Active steps | Support reuse | Support rebuild | Elapsed [s] |
| --- | ---: | ---: | ---: | ---: |
| 4/4 | `2` | `0` | `2` | `402.010` |
| 3/4 | `2` | `1` | `1` | `110.310` |
| 2/4 | `2` | `0` | `2` | `110.496` |
| 1/4 | `2` | `2` | `0` | `0.810` |
| Total | `8` | `3` | `5` | `623.629` |

Payload diagnostics:

| Diagnostic | Value |
| --- | ---: |
| `residual_count` | `16` |
| `parameter_count` | `8` |
| `geometry_active_float_leaves` | `9` |
| `ntx_support_active_float_leaves` | `19` |
| `ntx_surface_backend` | `vmec` |
| `ntx_surface_branch` | `vmec_traceable` |
| `compact_payload_tangent_contract` | `True` |
| `raw_block_param_bar_l2` | `5.148705e+02` |
| `raw_block_param_bar_all_finite` | `True` |
| `raw_block_param_bar_first_nonfinite` | `None` |

Output JSON:

```text
outputs/autodiff_transport_lagged_ntx/reverse_ad/transport_reverse_ad_only_full_transport_shared_payload_smoke.json
```

| Quantity | Value |
| --- | ---: |
| Baseline objective value | `4.4387332574093733e+02` |
| Frozen-linearized FD `d/dZBS:1:0` | `-1.9205101180950553e+03` |
| Forward JVP `d/dZBS:1:0` | `-1.9205082810004324e+03` |
| Optimization-internal raw-block reverse `d/dZBS:1:0` | `-1.9205082803893893e+03` |
| Shared-payload reverse row `d/dZBS:1:0` | `-1.9205081921396923e+03` |

| Comparison | Abs diff | Rel diff |
| --- | ---: | ---: |
| Shared-payload reverse vs frozen-linearized FD | `1.9259553630490700e-03` | `1.0028353117761315e-06` |
| Shared-payload reverse vs forward JVP | `8.8860740106611050e-05` | `4.6269386591929540e-08` |
| Shared-payload reverse vs optimization-internal raw-block reverse | `8.8249697000719610e-05` | `4.5951219217251536e-08` |
| Optimization-internal raw-block reverse vs forward JVP | `6.1104310589144010e-07` | `3.1816738929817846e-10` |

Status:

- The FD value is now saved for the current maxJ objective with `ZBS:1:0`.
- The standalone optimization-internal raw-block reverse matches the forward
  JVP target to `3.2e-10` relative.
- The full shared-payload reverse row is within `4.7e-08` relative of the
  forward JVP value and `1.1e-06` relative of frozen-linearized FD.

## Geometry Frozen-FD Commands To Rerun

Use the same VMEC input from the transport TOML:

```bash
python ./examples/benchmarks/compare_geometry_qi_frozen_linearized_fd.py \
  --vmec-input ./examples/inputs/input.QI_nfp2_newNT_opt_hires_true \
  --parameter RBC:1:0 \
  --objective boozer_qi_objective \
  --multigrid \
  --forward-linear-solve-mode raw_block \
  --forward-linear-maxiter 300 \
  --adjoint-maxiter 300
```

```bash
python ./examples/benchmarks/compare_geometry_qi_frozen_linearized_fd.py \
  --vmec-input ./examples/inputs/input.QI_nfp2_newNT_opt_hires_true \
  --parameter RBC:1:0 \
  --objective boozer_maxj_objective \
  --multigrid \
  --forward-linear-solve-mode raw_block \
  --forward-linear-maxiter 300 \
  --adjoint-maxiter 300
```

Then repeat for `ZBS:1:0`:

```bash
python ./examples/benchmarks/compare_geometry_qi_frozen_linearized_fd.py \
  --vmec-input ./examples/inputs/input.QI_nfp2_newNT_opt_hires_true \
  --parameter ZBS:1:0 \
  --objective boozer_qi_objective \
  --multigrid \
  --forward-linear-solve-mode raw_block \
  --forward-linear-maxiter 300 \
  --adjoint-maxiter 300
```

```bash
python ./examples/benchmarks/compare_geometry_qi_frozen_linearized_fd.py \
  --vmec-input ./examples/inputs/input.QI_nfp2_newNT_opt_hires_true \
  --parameter ZBS:1:0 \
  --objective boozer_maxj_objective \
  --multigrid \
  --forward-linear-solve-mode raw_block \
  --forward-linear-maxiter 300 \
  --adjoint-maxiter 300
```
