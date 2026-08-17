# Optimization AD vs FD References

Reference parameter:

- VMEC harmonic: `RBC:1:0`

This file collects the saved optimization-facing AD rows and the available FD
references. Root-only ambipolarity and full time-evolution transport are kept
separate because they are different maps.

## 2026-08-17 Current Runtime: 16-Step Reverse AD vs Frozen-Root FD

This is a new, separate table for the current clean runtime, not a replacement
for the historical table below. It compares the reverse run in attachment
`52544449-e29e-4f6c-bc2a-d233140cac7b/pasted-text.txt` with the frozen-root FD
runs in attachment `68ec4a34-76b4-4b04-b8e2-25043283a041/pasted-text.txt`.

- Reverse AD: `f5d22b9`, 16 accepted steps, `bicgstab`, segment length 4.
- FD: accepted-step replay, frozen-linearized geometry and initial-`Er` root.
- Parameters available in the pasted FD output: `n0`, `vmec:RBC:1:0`, and
  `vmec:ZBS:1:0`.
- `smooth_root_proxy` FD values are near zero, so relative errors are omitted.

| Objective | Parameter | Reverse AD | FD | Abs diff | Rel diff |
| --- | --- | ---: | ---: | ---: | ---: |
| `transport:softmax_Er` | `n0` | `-4.7372059445518024e+00` | `-4.737153e+00` | `5.294455e-05` | `1.117645e-05` |
| `transport:softmax_Er` | `vmec:RBC:1:0` | `-5.9294171360644114e+01` | `-5.929451e+01` | `3.386394e-04` | `5.711142e-06` |
| `transport:softmax_Er` | `vmec:ZBS:1:0` | `1.7363563921683088e+01` | `1.736354e+01` | `2.392168e-05` | `1.377696e-06` |
| `transport:smooth_root_proxy` | `n0` | `1.3140207886077174e-12` | `2.060242e-12` | `7.462212e-13` | not meaningful |
| `transport:smooth_root_proxy` | `vmec:RBC:1:0` | `-1.5181833103128324e-11` | `-1.570778e-11` | `5.259469e-13` | not meaningful |
| `transport:smooth_root_proxy` | `vmec:ZBS:1:0` | `6.4341188451947432e-12` | `1.022043e-11` | `3.786311e-12` | not meaningful |
| `transport:Er_transition_left` | `n0` | `-1.4431430225336488e+00` | `-1.443123e+00` | `2.002253e-05` | `1.387445e-05` |
| `transport:Er_transition_left` | `vmec:RBC:1:0` | `-2.0323526025367933e+01` | `-2.032336e+01` | `1.660254e-04` | `8.169189e-06` |
| `transport:Er_transition_left` | `vmec:ZBS:1:0` | `1.4570485947972855e-01` | `1.455537e-01` | `1.511595e-04` | `1.038513e-03` |
| `transport:Er_transition_right` | `n0` | `-1.6668910892926467e+00` | `-1.666926e+00` | `3.491071e-05` | `2.094317e-05` |
| `transport:Er_transition_right` | `vmec:RBC:1:0` | `-2.2520000095719766e+01` | `-2.251924e+01` | `7.600957e-04` | `3.375317e-05` |
| `transport:Er_transition_right` | `vmec:ZBS:1:0` | `1.0112381885185524e+00` | `1.011268e+00` | `2.981148e-05` | `2.947931e-05` |
| `transport:Er2_volume_average` | `n0` | `-6.1442936636985735e+01` | `-6.144275e+01` | `1.866370e-04` | `3.037575e-06` |
| `transport:Er2_volume_average` | `vmec:RBC:1:0` | `-6.4027857277260023e+02` | `-6.439559e+02` | `3.677327e+00` | `5.710526e-03` |
| `transport:Er2_volume_average` | `vmec:ZBS:1:0` | `2.3683438177132660e+02` | `2.405000e+02` | `3.665618e+00` | `1.524166e-02` |
| `transport:Er_volume_average` | `n0` | `-2.0463404184736076e+00` | `-2.046336e+00` | `4.418474e-06` | `2.159212e-06` |
| `transport:Er_volume_average` | `vmec:RBC:1:0` | `-2.0502251533818249e+01` | `-2.043617e+01` | `6.608153e-02` | `3.233558e-03` |
| `transport:Er_volume_average` | `vmec:ZBS:1:0` | `1.3012056495750357e+01` | `1.294637e+01` | `6.568650e-02` | `5.073738e-03` |
| `transport:electron_temperature_volume_average_keV` | `n0` | `8.1424000230967408e-04` | `8.142367e-04` | `3.302310e-09` | `4.055712e-06` |
| `transport:electron_temperature_volume_average_keV` | `vmec:RBC:1:0` | `-1.4246699858583421e-02` | `-1.473905e-02` | `4.923501e-04` | `3.340447e-02` |
| `transport:electron_temperature_volume_average_keV` | `vmec:ZBS:1:0` | `-4.0362002587967301e-02` | `-4.080381e-02` | `4.418074e-04` | `1.082760e-02` |
| `transport:total_pressure_volume_average` | `n0` | `8.0616879274542512e+00` | `8.061688e+00` | `7.254575e-08` | `8.998829e-09` |
| `transport:total_pressure_volume_average` | `vmec:RBC:1:0` | `-7.2256082317058004e-02` | `-7.291633e-02` | `6.602477e-04` | `9.054867e-03` |
| `transport:total_pressure_volume_average` | `vmec:ZBS:1:0` | `-2.3734153978905831e-01` | `-2.379263e-01` | `5.847602e-04` | `2.457737e-03` |
| `transport:alpha_power_volume_average_mw_m3` | `n0` | `2.7876640348138404e-01` | `2.787664e-01` | `3.481384e-09` | `1.248854e-08` |
| `transport:alpha_power_volume_average_mw_m3` | `vmec:RBC:1:0` | `-7.7627036193996983e-03` | `-7.763178e-03` | `4.743806e-07` | `6.110650e-05` |
| `transport:alpha_power_volume_average_mw_m3` | `vmec:ZBS:1:0` | `-1.1437598028239352e-03` | `-1.143404e-03` | `3.558028e-07` | `3.111786e-04` |
| `transport:bootstrap_current_softmax_abs_scaled` | `n0` | `-1.4697163840828864e-03` | `-1.468507e-03` | `1.209384e-06` | `8.235467e-04` |
| `transport:bootstrap_current_softmax_abs_scaled` | `vmec:RBC:1:0` | `-1.7890068335937415e+00` | `-1.788703e+00` | `3.038336e-04` | `1.698625e-04` |
| `transport:bootstrap_current_softmax_abs_scaled` | `vmec:ZBS:1:0` | `-6.2524365315469987e+00` | `-6.253120e+00` | `6.834685e-04` | `1.093004e-04` |

### 2026-08-17 Current Runtime: `T0` Reverse AD vs Frozen-Root FD

The FD values below are from the 16-step `T0` command recorded in the terminal
output. Reverse AD values are the matching current run in attachment
`52544449-e29e-4f6c-bc2a-d233140cac7b/pasted-text.txt`.

| Objective | Reverse AD `d/dT0` | Frozen-root FD `d/dT0` | Abs diff | Rel diff |
| --- | ---: | ---: | ---: | ---: |
| `transport:softmax_Er` | `3.9104300296804584e+00` | `3.910435e+00` | `4.970320e-06` | `1.271040e-06` |
| `transport:smooth_root_proxy` | `1.8756469174381453e-13` | `1.624273e-13` | `2.513739e-14` | not meaningful |
| `transport:Er_transition_left` | `1.8372816062999480e+00` | `1.837278e+00` | `3.606300e-06` | `1.962849e-06` |
| `transport:Er_transition_right` | `2.0017656379975195e+00` | `2.001785e+00` | `1.936200e-05` | `9.672369e-06` |
| `transport:Er2_volume_average` | `6.5043986175639475e+01` | `6.504393e+01` | `5.617564e-05` | `8.636569e-07` |
| `transport:Er_volume_average` | `9.3973858121025389e-01` | `9.397335e-01` | `5.081210e-06` | `5.407076e-06` |
| `transport:electron_temperature_volume_average_keV` | `3.5561174373341170e-01` | `3.556117e-01` | `4.373341e-08` | `1.229808e-07` |
| `transport:total_pressure_volume_average` | `1.8651780767713835e+00` | `1.865178e+00` | `7.677138e-08` | `4.116035e-08` |
| `transport:alpha_power_volume_average_mw_m3` | `8.3728491250583875e-02` | `8.372849e-02` | `1.250584e-09` | `1.493618e-08` |
| `transport:bootstrap_current_softmax_abs_scaled` | `2.3160616144396828e-01` | `2.316064e-01` | `2.385560e-07` | `1.030006e-06` |

## Current 16-Step Full-Transport Shared-Payload AD vs FD

Current reverse AD source:

- Attachment: `ddf26dbf-0458-434f-a624-12b2faef2a72/pasted-text.txt`
- Command: `benchmark_transport_reverse_ad_only.py ... --accepted-step-limit 16 --reverse-segment-length 4 --full-transport-shared-payload-smoke`
- Mode: `transport_reverse_ad_only_full_transport_shared_payload_smoke`
- Residual count: `16`
- Parameter count: `8`
- Runtime status: completed; no OOM
- Elapsed time: `7960.344 s`
- Segmented reverse sweep: `4271.987 s`, `support_reuse=6`, `support_rebuild=10`

Status:

- The complete stdout from attachment
  `384266e9-149e-446a-b2fe-d985754c3fd4/pasted-text.txt` is saved below as the
  current completed 8-parameter shared-payload reverse AD reference.
- The older partial comparison table immediately below is kept as an audit
  trail for the first transport rows; the compact matrix and worst-case summary
  below it supersede the earlier "truncated stdout" note.

FD references:

- Profile FD JSONs under
  `outputs/autodiff_transport_lagged_ntx/realtime_geometry_fd/*_forward_fd_summary.json`.
- Refreshed `RBC:1:0` FD values from the saved 2026-08-09 full-transport
  16-step FD run.
- Refreshed `ZBS:1:0` FD values from the saved frozen-root full-transport
  16-step FD run.

| Objective | Parameter | Reverse AD | FD | Abs diff | Rel diff |
| --- | --- | ---: | ---: | ---: | ---: |
| `transport:softmax_Er` | `n0` | `-7.5788861196501287e+00` | `-7.5788280000000000e+00` | `5.811965e-05` | `7.668686e-06` |
| `transport:softmax_Er` | `T0` | `5.5481041431530871e+00` | `5.5481180000000000e+00` | `1.385685e-05` | `2.497576e-06` |
| `transport:softmax_Er` | `density_shape_power` | `-1.4065516873746525e-01` | `-1.4622072856222226e-01` | `5.565560e-03` | `3.806273e-02` |
| `transport:softmax_Er` | `temperature_shape_power` | `7.0011320421952696e+00` | `7.0010103373855754e+00` | `1.217048e-04` | `1.738389e-05` |
| `transport:softmax_Er` | `density_shape_alpha` | `2.6082258582205270e-01` | `2.6128750000000000e-01` | `4.649142e-04` | `1.779320e-03` |
| `transport:softmax_Er` | `temperature_shape_alpha` | `1.3627446982536663e+01` | `1.3627480000000000e+01` | `3.301746e-05` | `2.422859e-06` |
| `transport:softmax_Er` | `vmec:RBC:1:0` | `-7.8516612048142349e+01` | `-7.8543180000000007e+01` | `2.656795e-02` | `3.382592e-04` |
| `transport:softmax_Er` | `vmec:ZBS:1:0` | `3.6119357588642728e+01` | `3.6140230000000000e+01` | `2.087241e-02` | `5.775395e-04` |
| `transport:smooth_root_proxy` | `n0` | `1.8468993787754641e-13` | `0.0000000000000000e+00` | `1.846899e-13` | not meaningful |
| `transport:smooth_root_proxy` | `T0` | `1.0913831787141299e-13` | `1.6242730000000000e-13` | `5.328898e-14` | not meaningful |
| `transport:smooth_root_proxy` | `density_shape_power` | `6.5876084371438111e-18` | `0.0000000000000000e+00` | `6.587608e-18` | not meaningful |
| `transport:smooth_root_proxy` | `temperature_shape_power` | `-1.3072699977751167e-12` | `0.0000000000000000e+00` | `1.307270e-12` | not meaningful |
| `transport:smooth_root_proxy` | `density_shape_alpha` | `-1.1970961861304089e-17` | `0.0000000000000000e+00` | `1.197096e-17` | not meaningful |
| `transport:smooth_root_proxy` | `temperature_shape_alpha` | `3.6847635980041285e-13` | `0.0000000000000000e+00` | `3.684764e-13` | not meaningful |
| `transport:smooth_root_proxy` | `vmec:RBC:1:0` | `-2.5108173983191352e-12` | `-2.6179630000000000e-12` | `1.071456e-13` | not meaningful |
| `transport:smooth_root_proxy` | `vmec:ZBS:1:0` | `1.3915732358342333e-12` | `0.0000000000000000e+00` | `1.391573e-12` | not meaningful |
| `transport:Er_transition_left` | `n0` | `-1.4467383415531603e+00` | `-1.4467480000000000e+00` | `9.658447e-06` | `6.675970e-06` |
| `transport:Er_transition_left` | `T0` | `1.8373054811266103e+00` | `1.8373090000000000e+00` | `3.518873e-06` | `1.915232e-06` |
| `transport:Er_transition_left` | `density_shape_power` | `-1.2789965825941793e-02` | `-1.4053419169120691e-02` | `1.263453e-03` | `8.990363e-02` |
| `transport:Er_transition_left` | `temperature_shape_power` | `-7.1806451200643551e+00` | `-7.1740150620058785e+00` | `6.630058e-03` | `9.241768e-04` |
| `transport:Er_transition_left` | `density_shape_alpha` | `1.5785389591463254e-02` | `1.5976850000000000e-02` | `1.914604e-04` | `1.198361e-02` |
| `transport:Er_transition_left` | `temperature_shape_alpha` | `1.6101823573643419e+01` | `1.6101590000000000e+01` | `2.335736e-04` | `1.450625e-05` |
| `transport:Er_transition_left` | `vmec:RBC:1:0` | `-2.0306396145716139e+01` | `-2.0306530000000000e+01` | `1.338543e-04` | `6.591687e-06` |
| `transport:Er_transition_left` | `vmec:ZBS:1:0` | `1.5433021332202301e-01` | `1.5425820000000000e-01` | `7.201332e-05` | `4.668363e-04` |
| `transport:Er_transition_right` | `n0` | `-1.6706147148780452e+00` | `-1.6705960000000000e+00` | `1.871488e-05` | `1.120252e-05` |
| `transport:Er_transition_right` | `T0` | `2.0017329575732847e+00` | `2.0017310000000000e+00` | `1.957573e-06` | `9.779402e-07` |
| `transport:Er_transition_right` | `density_shape_power` | `-1.8191178260101894e-02` | `-8.8057857302222442e-04` | `1.731060e-02` | `1.965821e+01` |
| `transport:Er_transition_right` | `temperature_shape_power` | `-6.2631084303152251e+00` | `-6.3494936271742599e+00` | `8.638520e-02` | `1.360505e-02` |
| `transport:Er_transition_right` | `density_shape_alpha` | `2.3817948337074714e-02` | `2.3610870000000000e-02` | `2.070783e-04` | `8.770466e-03` |
| `transport:Er_transition_right` | `temperature_shape_alpha` | `1.6294532234336653e+01` | `1.6294360000000000e+01` | `1.722343e-04` | `1.057018e-05` |
| `transport:Er_transition_right` | `vmec:RBC:1:0` | `-2.2498789234739608e+01` | `-2.2497940000000000e+01` | `8.492347e-04` | `3.774722e-05` |
| `transport:Er_transition_right` | `vmec:ZBS:1:0` | `1.0210612055125150e+00` | `1.0207530000000000e+00` | `3.082055e-04` | `3.019394e-04` |
| `transport:Er2_volume_average` | `n0` | `6.6923799716616873e+02` | `6.6924850000000000e+02` | `1.050283e-02` | `1.569347e-05` |
| `transport:Er2_volume_average` | `T0` | `-3.6150089065989789e+02` | `-3.6157180000000000e+02` | `7.090934e-02` | `1.961141e-04` |
| `transport:Er2_volume_average` | `density_shape_power` | `1.1723514237983892e+01` | `1.2970932535457298e+01` | `1.247418e+00` | `9.617029e-02` |
| `transport:Er2_volume_average` | `temperature_shape_power` | `-9.9263140512588939e+02` | `-9.9282705091259993e+02` | `1.956458e-01` | `1.970593e-04` |
| `transport:Er2_volume_average` | `density_shape_alpha` | `-2.5500745056169166e+01` | `-2.5533480000000000e+01` | `3.273494e-02` | `1.282040e-03` |
| `transport:Er2_volume_average` | `temperature_shape_alpha` | `1.5501576948237096e+02` | `1.5504050000000000e+02` | `2.473052e-02` | `1.595100e-04` |
| `transport:Er2_volume_average` | `vmec:RBC:1:0` | `5.4255042598612426e+03` | `5.4307290000000003e+03` | `5.224740e+00` | `9.620698e-04` |
| `transport:Er2_volume_average` | `vmec:ZBS:1:0` | `-4.5451072448987597e+03` | `-4.5484850000000000e+03` | `3.377755e+00` | `7.426110e-04` |
| `transport:Er_volume_average` | `n0` | `1.6894704215412517e+01` | `1.6893630000000000e+01` | `1.074215e-03` | `6.358701e-05` |
| `transport:Er_volume_average` | `T0` | `-1.0123399603364410e+01` | `-1.0115900000000000e+01` | `7.499603e-03` | `7.413679e-04` |
| `transport:Er_volume_average` | `density_shape_power` | `1.6963617698293820e-01` | `2.0482617935080327e-01` | `3.519000e-02` | `1.718042e-01` |
| `transport:Er_volume_average` | `temperature_shape_power` | `-2.7865259395311515e+01` | `-2.7863388379530583e+01` | `1.871016e-03` | `6.714961e-05` |
| `transport:Er_volume_average` | `density_shape_alpha` | `-4.4400532599246079e-01` | `-4.4494800000000000e-01` | `9.426740e-04` | `2.118616e-03` |
| `transport:Er_volume_average` | `temperature_shape_alpha` | `2.8276657155341169e+00` | `2.8288200000000001e+00` | `1.154284e-03` | `4.080445e-04` |
| `transport:Er_volume_average` | `vmec:RBC:1:0` | `1.3669898081845315e+02` | `1.3627310000000000e+02` | `4.258808e-01` | `3.125201e-03` |
| `transport:Er_volume_average` | `vmec:ZBS:1:0` | `-1.1085217069468665e+02` | `-1.1056010000000000e+02` | `2.920707e-01` | `2.641737e-03` |

Current frozen-root `ZBS:1:0` FD run:

```bash
python ./examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --parameter ZBS:1:0 \
  --geometry-fd-lane frozen_linearized \
  --accepted-step-limit 16 \
  --radau-jacobian-reuse-mode legacy \
  --replay-mode accepted \
  --initial-Er-root-ad jax_selected_root \
  --initial-Er-root-fd-root-lane frozen_linearized
```

- Parameter: `ZBS:1:0`
- Geometry FD lane: `frozen_linearized`
- Root FD lane: `frozen_linearized`
- Output JSON: `outputs/autodiff_transport_lagged_ntx/realtime_geometry_fd/ZBS_1_0_forward_fd_frozen_linearized_summary.json`

| Objective | Value | FD `d/dvmec:ZBS:1:0` | Explicit-geometry FD | Final-state FD |
| --- | ---: | ---: | ---: | ---: |
| `transport:softmax_Er` | `2.0493049813391263e+01` | `3.6140230000000000e+01` | `0.0000000000000000e+00` | `3.6140230000000000e+01` |
| `transport:smooth_root_proxy` | `9.8039215686309116e-03` | `0.0000000000000000e+00` | `0.0000000000000000e+00` | `0.0000000000000000e+00` |
| `transport:Er_transition_left` | `1.7729764966344089e+01` | `1.5425820000000000e-01` | `0.0000000000000000e+00` | `1.5425820000000000e-01` |
| `transport:Er_transition_right` | `1.8376267791735049e+01` | `1.0207530000000000e+00` | `0.0000000000000000e+00` | `1.0207530000000000e+00` |
| `transport:Er2_volume_average` | `2.3720890299502904e+02` | `-4.5484850000000000e+03` | `8.4861320000000003e-01` | `-4.5493340000000003e+03` |
| `transport:Er_volume_average` | `-3.4526122553412755e+00` | `-1.1056010000000000e+02` | `-1.5971980000000000e-01` | `-1.1040040000000000e+02` |
| `transport:electron_temperature_volume_average_keV` | `6.5646442870649961e+00` | `-3.2035020000000000e-02` | `-3.9826940000000003e-02` | `7.7919190000000003e-03` |
| `transport:total_pressure_volume_average` | `3.4213473064475188e+01` | `-2.3491310000000000e-01` | `-2.3763150000000000e-01` | `2.7184300000000001e-03` |
| `transport:alpha_power_volume_average_mw_m3` | `5.8935809218241009e-01` | `-1.6199040000000000e-01` | `-6.8671580000000002e-03` | `-1.5512330000000000e-01` |
| `transport:bootstrap_current_softmax_abs_scaled` | `1.4485504661986399e+00` | `-6.2410010000000000e+00` | `-6.3191880000000000e+00` | `7.8187120000000004e-02` |

Current shared-payload reverse AD vs frozen-root `ZBS:1:0` FD:

| Objective | Reverse AD | FD | Abs diff | Rel diff |
| --- | ---: | ---: | ---: | ---: |
| `transport:softmax_Er` | `3.6119357588642728e+01` | `3.6140230000000000e+01` | `2.087241e-02` | `5.775395e-04` |
| `transport:smooth_root_proxy` | `1.3915732358342333e-12` | `0.0000000000000000e+00` | `1.391573e-12` | not meaningful |
| `transport:Er_transition_left` | `1.5433021332202301e-01` | `1.5425820000000000e-01` | `7.201332e-05` | `4.668363e-04` |
| `transport:Er_transition_right` | `1.0210612055125150e+00` | `1.0207530000000000e+00` | `3.082055e-04` | `3.019394e-04` |
| `transport:Er2_volume_average` | `-4.5451072448987597e+03` | `-4.5484850000000000e+03` | `3.377755e+00` | `7.426110e-04` |
| `transport:Er_volume_average` | `-1.1085217069468665e+02` | `-1.1056010000000000e+02` | `2.920707e-01` | `2.641737e-03` |
| `transport:electron_temperature_volume_average_keV` | `-3.1844772862062630e-02` | `-3.2035020000000000e-02` | `1.902471e-04` | `5.938724e-03` |
| `transport:total_pressure_volume_average` | `-2.3469849316053068e-01` | `-2.3491310000000000e-01` | `2.146068e-04` | `9.135584e-04` |
| `transport:alpha_power_volume_average_mw_m3` | `-1.6182403125293193e-01` | `-1.6199040000000000e-01` | `1.663687e-04` | `1.027028e-03` |
| `transport:bootstrap_current_softmax_abs_scaled` | `-6.2411213048264766e+00` | `-6.2410010000000000e+00` | `1.203048e-04` | `1.927653e-05` |

Summary:

- Most rows shown here are consistent at `O(1e-3)` relative or better.
- The largest mismatches are concentrated in `density_shape_power`,
  `density_shape_alpha`, and `temperature_shape_alpha` for the root/Er
  transition or Er-volume rows. These are likely the most branch-sensitive FD
  columns and should be checked with a frozen-root/branch diagnostic if we need
  strict agreement.
- `smooth_root_proxy` derivatives are effectively zero in both AD and FD; the
  relative error is not meaningful there.
- Full current comparison for pressure, alpha power, bootstrap, and geometry
  rows is now saved in the complete matrix below.

### Current Complete 16-Step Shared-Payload Reverse AD Matrix

Source:

- Attachment: `384266e9-149e-446a-b2fe-d985754c3fd4/pasted-text.txt`
- Mode: `transport_reverse_ad_only_full_transport_shared_payload_smoke`
- Residual count: `16`
- Parameter count: `8`
- Elapsed time: `7960.344 s`

| Objective | Value | `d/dn0` | `d/dT0` | `d/ddensity_shape_power` | `d/dtemperature_shape_power` | `d/ddensity_shape_alpha` | `d/dtemperature_shape_alpha` | `d/dRBC:1:0` | `d/dZBS:1:0` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `transport:softmax_Er` | `2.0493040653152047e+01` | `-7.5788861196501287e+00` | `5.5481041431530871e+00` | `-1.4065516873746525e-01` | `7.0011320421952696e+00` | `2.6082258582205270e-01` | `1.3627446982536663e+01` | `-7.8516612048142349e+01` | `3.6119357588642728e+01` |
| `transport:smooth_root_proxy` | `9.8039215686309099e-03` | `1.8468993787754641e-13` | `1.0913831787141299e-13` | `6.5876084371438111e-18` | `-1.3072699977751167e-12` | `-1.1970961861304089e-17` | `3.6847635980041285e-13` | `-2.5108173983191352e-12` | `1.3915732358342333e-12` |
| `transport:Er_transition_left` | `1.7729764216611777e+01` | `-1.4467383415531603e+00` | `1.8373054811266103e+00` | `-1.2789965825941793e-02` | `-7.1806451200643551e+00` | `1.5785389591463254e-02` | `1.6101823573643419e+01` | `-2.0306396145716139e+01` | `1.5433021332202301e-01` |
| `transport:Er_transition_right` | `1.8376267082555085e+01` | `-1.6706147148780452e+00` | `2.0017329575732847e+00` | `-1.8191178260101894e-02` | `-6.2631084303152251e+00` | `2.3817948337074714e-02` | `1.6294532234336653e+01` | `-2.2498789234739608e+01` | `1.0210612055125150e+00` |
| `transport:Er2_volume_average` | `2.3720951871161546e+02` | `6.6923799716616873e+02` | `-3.6150089065989789e+02` | `1.1723514237983892e+01` | `-9.9263140512588939e+02` | `-2.5500745056169166e+01` | `1.5501576948237096e+02` | `5.4255042598612426e+03` | `-4.5451072448987597e+03` |
| `transport:Er_volume_average` | `-3.4526136020627689e+00` | `1.6894704215412517e+01` | `-1.0123399603364410e+01` | `1.6963617698293820e-01` | `-2.7865259395311515e+01` | `-4.4400532599246079e-01` | `2.8276657155341169e+00` | `1.3669898081845315e+02` | `-1.1085217069468665e+02` |
| `transport:electron_temperature_volume_average_keV` | `6.5646440796139922e+00` | `-8.6619872749393867e-04` | `3.5611894122635163e-01` | `-4.3211954898123711e-05` | `1.5250078231079183e+00` | `5.3339610943692328e-04` | `-3.0435342768367408e+00` | `-2.3726977964542796e-02` | `-3.1844772862062630e-02` |
| `transport:total_pressure_volume_average` | `3.4213472705740244e+01` | `8.0596720085197759e+00` | `1.8650270327188381e+00` | `2.4425502141668159e-01` | `7.7515855901247424e+00` | `-1.3265837825838147e+00` | `-1.4519696704079431e+01` | `-7.6644633630893411e-02` | `-2.3469849316053068e-01` |
| `transport:alpha_power_volume_average_mw_m3` | `5.8935810831572644e-01` | `3.0339361742164600e-01` | `6.9385082611546092e-02` | `2.6728550609761139e-03` | `2.5036297131320684e-01` | `-8.2338695461871159e-03` | `-4.1292076345191370e-01` | `1.9544849790848531e-01` | `-1.6182403125293193e-01` |
| `transport:bootstrap_current_softmax_abs_scaled` | `1.4485486870521171e+00` | `-1.0774573904620000e-03` | `2.3095562735965830e-01` | `-1.3222637711917493e-02` | `1.5244064570671323e+00` | `1.1406429742135712e-01` | `-3.5233961360644237e+00` | `-1.7673950710878303e+00` | `-6.2411213048264766e+00` |
| `geometry:boozer_qi_objective` | `2.1192029797323711e-01` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `5.9392927523334720e+00` | `-1.2365499862721663e-01` |
| `geometry:boozer_maxj_objective` | `4.4387332711513375e+02` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `-3.8431400489698863e+03` | `-1.9205081920611992e+03` |
| `geometry:vmec_aspect_ratio` | `1.0015330918957178e+01` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `-5.4006784187006147e+00` | `-5.5226885751318529e+00` |
| `geometry:vmec_iota_mean` | `-5.9365259946730364e-01` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `2.4405140677736042e-01` | `1.4567526055774602e-01` |
| `geometry:vmec_magnetic_well` | `-2.7476128719881053e-02` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `-1.1090112492865956e-02` | `-4.1682026469118227e-02` |
| `geometry:vmec_mirror_ratio` | `2.1153803467611412e-01` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `-5.9359094689600833e-01` | `4.1437006136733534e-01` |

### Current 16-Step Transport: Full AD vs Saved FD Table

This table contains every transport objective and every parameter for which a
saved FD reference exists locally.

| Objective | Parameter | Reverse AD | FD | Abs diff | Rel diff |
| --- | --- | ---: | ---: | ---: | ---: |
| `transport:softmax_Er` | `n0` | `-7.5788861196501287e+00` | `-7.5788280000000000e+00` | `5.811965e-05` | `7.668686e-06` |
| `transport:softmax_Er` | `T0` | `5.5481041431530871e+00` | `5.5481180000000000e+00` | `1.385685e-05` | `2.497576e-06` |
| `transport:softmax_Er` | `density_shape_power` | `-1.4065516873746525e-01` | `-1.4622072856222226e-01` | `5.565560e-03` | `3.806273e-02` |
| `transport:softmax_Er` | `temperature_shape_power` | `7.0011320421952696e+00` | `7.0010103373855754e+00` | `1.217048e-04` | `1.738389e-05` |
| `transport:softmax_Er` | `density_shape_alpha` | `2.6082258582205270e-01` | `2.6128750000000000e-01` | `4.649142e-04` | `1.779320e-03` |
| `transport:softmax_Er` | `temperature_shape_alpha` | `1.3627446982536663e+01` | `1.3627480000000000e+01` | `3.301746e-05` | `2.422859e-06` |
| `transport:softmax_Er` | `vmec:RBC:1:0` | `-7.8516612048142349e+01` | `-7.8543180000000007e+01` | `2.656795e-02` | `3.382592e-04` |
| `transport:softmax_Er` | `vmec:ZBS:1:0` | `3.6119357588642728e+01` | `3.6140230000000000e+01` | `2.087241e-02` | `5.775395e-04` |
| `transport:smooth_root_proxy` | `n0` | `1.8468993787754641e-13` | `0.0000000000000000e+00` | `1.846899e-13` | not meaningful |
| `transport:smooth_root_proxy` | `T0` | `1.0913831787141299e-13` | `1.6242730000000000e-13` | `5.328898e-14` | not meaningful |
| `transport:smooth_root_proxy` | `density_shape_power` | `6.5876084371438111e-18` | `0.0000000000000000e+00` | `6.587608e-18` | not meaningful |
| `transport:smooth_root_proxy` | `temperature_shape_power` | `-1.3072699977751167e-12` | `0.0000000000000000e+00` | `1.307270e-12` | not meaningful |
| `transport:smooth_root_proxy` | `density_shape_alpha` | `-1.1970961861304089e-17` | `0.0000000000000000e+00` | `1.197096e-17` | not meaningful |
| `transport:smooth_root_proxy` | `temperature_shape_alpha` | `3.6847635980041285e-13` | `0.0000000000000000e+00` | `3.684764e-13` | not meaningful |
| `transport:smooth_root_proxy` | `vmec:RBC:1:0` | `-2.5108173983191352e-12` | `-2.6179630000000000e-12` | `1.071456e-13` | not meaningful |
| `transport:smooth_root_proxy` | `vmec:ZBS:1:0` | `1.3915732358342333e-12` | `0.0000000000000000e+00` | `1.391573e-12` | not meaningful |
| `transport:Er_transition_left` | `n0` | `-1.4467383415531603e+00` | `-1.4467480000000000e+00` | `9.658447e-06` | `6.675970e-06` |
| `transport:Er_transition_left` | `T0` | `1.8373054811266103e+00` | `1.8373090000000000e+00` | `3.518873e-06` | `1.915232e-06` |
| `transport:Er_transition_left` | `density_shape_power` | `-1.2789965825941793e-02` | `-1.4053419169120691e-02` | `1.263453e-03` | `8.990363e-02` |
| `transport:Er_transition_left` | `temperature_shape_power` | `-7.1806451200643551e+00` | `-7.1740150620058785e+00` | `6.630058e-03` | `9.241768e-04` |
| `transport:Er_transition_left` | `density_shape_alpha` | `1.5785389591463254e-02` | `1.5976850000000000e-02` | `1.914604e-04` | `1.198361e-02` |
| `transport:Er_transition_left` | `temperature_shape_alpha` | `1.6101823573643419e+01` | `1.6101590000000000e+01` | `2.335736e-04` | `1.450625e-05` |
| `transport:Er_transition_left` | `vmec:RBC:1:0` | `-2.0306396145716139e+01` | `-2.0306530000000000e+01` | `1.338543e-04` | `6.591687e-06` |
| `transport:Er_transition_left` | `vmec:ZBS:1:0` | `1.5433021332202301e-01` | `1.5425820000000000e-01` | `7.201332e-05` | `4.668363e-04` |
| `transport:Er_transition_right` | `n0` | `-1.6706147148780452e+00` | `-1.6705960000000000e+00` | `1.871488e-05` | `1.120252e-05` |
| `transport:Er_transition_right` | `T0` | `2.0017329575732847e+00` | `2.0017310000000000e+00` | `1.957573e-06` | `9.779402e-07` |
| `transport:Er_transition_right` | `density_shape_power` | `-1.8191178260101894e-02` | `-8.8057857302222442e-04` | `1.731060e-02` | `1.965821e+01` |
| `transport:Er_transition_right` | `temperature_shape_power` | `-6.2631084303152251e+00` | `-6.3494936271742599e+00` | `8.638520e-02` | `1.360505e-02` |
| `transport:Er_transition_right` | `density_shape_alpha` | `2.3817948337074714e-02` | `2.3610870000000000e-02` | `2.070783e-04` | `8.770466e-03` |
| `transport:Er_transition_right` | `temperature_shape_alpha` | `1.6294532234336653e+01` | `1.6294360000000000e+01` | `1.722343e-04` | `1.057018e-05` |
| `transport:Er_transition_right` | `vmec:RBC:1:0` | `-2.2498789234739608e+01` | `-2.2497940000000000e+01` | `8.492347e-04` | `3.774722e-05` |
| `transport:Er_transition_right` | `vmec:ZBS:1:0` | `1.0210612055125150e+00` | `1.0207530000000000e+00` | `3.082055e-04` | `3.019394e-04` |
| `transport:Er2_volume_average` | `n0` | `6.6923799716616873e+02` | `6.6924850000000000e+02` | `1.050283e-02` | `1.569347e-05` |
| `transport:Er2_volume_average` | `T0` | `-3.6150089065989789e+02` | `-3.6157180000000000e+02` | `7.090934e-02` | `1.961141e-04` |
| `transport:Er2_volume_average` | `density_shape_power` | `1.1723514237983892e+01` | `1.2970932535457298e+01` | `1.247418e+00` | `9.617029e-02` |
| `transport:Er2_volume_average` | `temperature_shape_power` | `-9.9263140512588939e+02` | `-9.9282705091259993e+02` | `1.956458e-01` | `1.970593e-04` |
| `transport:Er2_volume_average` | `density_shape_alpha` | `-2.5500745056169166e+01` | `-2.5533480000000000e+01` | `3.273494e-02` | `1.282040e-03` |
| `transport:Er2_volume_average` | `temperature_shape_alpha` | `1.5501576948237096e+02` | `1.5504050000000000e+02` | `2.473052e-02` | `1.595100e-04` |
| `transport:Er2_volume_average` | `vmec:RBC:1:0` | `5.4255042598612426e+03` | `5.4307290000000003e+03` | `5.224740e+00` | `9.620698e-04` |
| `transport:Er2_volume_average` | `vmec:ZBS:1:0` | `-4.5451072448987597e+03` | `-4.5484850000000000e+03` | `3.377755e+00` | `7.426110e-04` |
| `transport:Er_volume_average` | `n0` | `1.6894704215412517e+01` | `1.6893630000000000e+01` | `1.074215e-03` | `6.358701e-05` |
| `transport:Er_volume_average` | `T0` | `-1.0123399603364410e+01` | `-1.0115900000000000e+01` | `7.499603e-03` | `7.413679e-04` |
| `transport:Er_volume_average` | `density_shape_power` | `1.6963617698293820e-01` | `2.0482617935080327e-01` | `3.519000e-02` | `1.718042e-01` |
| `transport:Er_volume_average` | `temperature_shape_power` | `-2.7865259395311515e+01` | `-2.7863388379530583e+01` | `1.871016e-03` | `6.714961e-05` |
| `transport:Er_volume_average` | `density_shape_alpha` | `-4.4400532599246079e-01` | `-4.4494800000000000e-01` | `9.426740e-04` | `2.118616e-03` |
| `transport:Er_volume_average` | `temperature_shape_alpha` | `2.8276657155341169e+00` | `2.8288200000000001e+00` | `1.154284e-03` | `4.080445e-04` |
| `transport:Er_volume_average` | `vmec:RBC:1:0` | `1.3669898081845315e+02` | `1.3627310000000000e+02` | `4.258808e-01` | `3.125201e-03` |
| `transport:Er_volume_average` | `vmec:ZBS:1:0` | `-1.1085217069468665e+02` | `-1.1056010000000000e+02` | `2.920707e-01` | `2.641737e-03` |
| `transport:electron_temperature_volume_average_keV` | `n0` | `-8.6619872749393867e-04` | `-8.6577110000000000e-04` | `4.276275e-07` | `4.939267e-04` |
| `transport:electron_temperature_volume_average_keV` | `T0` | `3.5611894122635163e-01` | `3.5611920000000000e-01` | `2.587736e-07` | `7.266490e-07` |
| `transport:electron_temperature_volume_average_keV` | `density_shape_power` | `-4.3211954898123711e-05` | `-4.5459636055511510e-05` | `2.247681e-06` | `4.944345e-02` |
| `transport:electron_temperature_volume_average_keV` | `temperature_shape_power` | `1.5250078231079183e+00` | `1.5250080035814999e+00` | `1.804736e-07` | `1.183427e-07` |
| `transport:electron_temperature_volume_average_keV` | `density_shape_alpha` | `5.3339610943692328e-04` | `5.3346070000000000e-04` | `6.459056e-08` | `1.210784e-04` |
| `transport:electron_temperature_volume_average_keV` | `temperature_shape_alpha` | `-3.0435342768367408e+00` | `-3.0435330000000000e+00` | `1.276837e-06` | `4.195245e-07` |
| `transport:electron_temperature_volume_average_keV` | `vmec:RBC:1:0` | `-2.3726977964542796e-02` | `-2.3962400000000000e-02` | `2.354220e-04` | `9.824643e-03` |
| `transport:electron_temperature_volume_average_keV` | `vmec:ZBS:1:0` | `-3.1844772862062630e-02` | `-3.2035020000000000e-02` | `1.902471e-04` | `5.938724e-03` |
| `transport:total_pressure_volume_average` | `n0` | `8.0596720085197759e+00` | `8.0596760000000000e+00` | `3.991480e-06` | `4.952408e-07` |
| `transport:total_pressure_volume_average` | `T0` | `1.8650270327188381e+00` | `1.8650300000000000e+00` | `2.967281e-06` | `1.591010e-06` |
| `transport:total_pressure_volume_average` | `density_shape_power` | `2.4425502141668159e-01` | `2.4425428316969069e-01` | `7.382470e-07` | `3.022453e-06` |
| `transport:total_pressure_volume_average` | `temperature_shape_power` | `7.7515855901247424e+00` | `7.7515860693229879e+00` | `4.791982e-07` | `6.181938e-08` |
| `transport:total_pressure_volume_average` | `density_shape_alpha` | `-1.3265837825838147e+00` | `-1.3265840000000000e+00` | `2.174162e-07` | `1.638918e-07` |
| `transport:total_pressure_volume_average` | `temperature_shape_alpha` | `-1.4519696704079431e+01` | `-1.4519690000000000e+01` | `6.704079e-06` | `4.617233e-07` |
| `transport:total_pressure_volume_average` | `vmec:RBC:1:0` | `-7.6644633630893411e-02` | `-7.6998129999999998e-02` | `3.534964e-04` | `4.590973e-03` |
| `transport:total_pressure_volume_average` | `vmec:ZBS:1:0` | `-2.3469849316053068e-01` | `-2.3491310000000000e-01` | `2.146068e-04` | `9.135584e-04` |
| `transport:alpha_power_volume_average_mw_m3` | `n0` | `3.0339361742164600e-01` | `3.0339440000000000e-01` | `7.825784e-07` | `2.579409e-06` |
| `transport:alpha_power_volume_average_mw_m3` | `T0` | `6.9385082611546092e-02` | `6.9382040000000000e-02` | `3.042612e-06` | `4.385301e-05` |
| `transport:alpha_power_volume_average_mw_m3` | `density_shape_power` | `2.6728550609761139e-03` | `2.7169120101847946e-03` | `4.405695e-05` | `1.621582e-02` |
| `transport:alpha_power_volume_average_mw_m3` | `temperature_shape_power` | `2.5036297131320684e-01` | `2.5035853991791279e-01` | `4.431395e-06` | `1.770020e-05` |
| `transport:alpha_power_volume_average_mw_m3` | `density_shape_alpha` | `-8.2338695461871159e-03` | `-8.2351060000000000e-03` | `1.236454e-06` | `1.501442e-04` |
| `transport:alpha_power_volume_average_mw_m3` | `temperature_shape_alpha` | `-4.1292076345191370e-01` | `-4.1291900000000001e-01` | `1.763452e-06` | `4.270697e-06` |
| `transport:alpha_power_volume_average_mw_m3` | `vmec:RBC:1:0` | `1.9544849790848531e-01` | `1.9566950000000000e-01` | `2.210021e-04` | `1.129466e-03` |
| `transport:alpha_power_volume_average_mw_m3` | `vmec:ZBS:1:0` | `-1.6182403125293193e-01` | `-1.6199040000000000e-01` | `1.663687e-04` | `1.027028e-03` |
| `transport:bootstrap_current_softmax_abs_scaled` | `n0` | `-1.0774573904620000e-03` | `-1.0791150000000000e-03` | `1.657610e-06` | `1.536082e-03` |
| `transport:bootstrap_current_softmax_abs_scaled` | `T0` | `2.3095562735965830e-01` | `2.3095560000000000e-01` | `2.735966e-08` | `1.184628e-07` |
| `transport:bootstrap_current_softmax_abs_scaled` | `density_shape_power` | `-1.3222637711917493e-02` | `-1.3221883476965957e-02` | `7.542350e-07` | `5.704444e-05` |
| `transport:bootstrap_current_softmax_abs_scaled` | `temperature_shape_power` | `1.5244064570671323e+00` | `1.5244087309129739e+00` | `2.273846e-06` | `1.491625e-06` |
| `transport:bootstrap_current_softmax_abs_scaled` | `density_shape_alpha` | `1.1406429742135712e-01` | `1.1406510000000000e-01` | `8.025786e-07` | `7.036146e-06` |
| `transport:bootstrap_current_softmax_abs_scaled` | `temperature_shape_alpha` | `-3.5233961360644237e+00` | `-3.5233890000000001e+00` | `7.136064e-06` | `2.025341e-06` |
| `transport:bootstrap_current_softmax_abs_scaled` | `vmec:RBC:1:0` | `-1.7673950710878303e+00` | `-1.7671640000000000e+00` | `2.310711e-04` | `1.307581e-04` |
| `transport:bootstrap_current_softmax_abs_scaled` | `vmec:ZBS:1:0` | `-6.2411213048264766e+00` | `-6.2410010000000000e+00` | `1.203048e-04` | `1.927653e-05` |

### Current 16-Step Transport: Worst AD vs Saved FD Rows

Saved FD references currently exist for `n0`, `T0`, `density_shape_power`,
`temperature_shape_power`, `density_shape_alpha`, `temperature_shape_alpha`,
`RBC:1:0`, and `ZBS:1:0`.

| Objective | Worst FD parameter | Reverse AD | FD | Rel diff |
| --- | --- | ---: | ---: | ---: |
| `transport:softmax_Er` | `density_shape_power` | `-1.4065516873746525e-01` | `-1.4622072856222226e-01` | `3.806273e-02` |
| `transport:smooth_root_proxy` | `vmec:RBC:1:0` | `-2.5108173983191352e-12` | `0.0000000000000000e+00` | not meaningful |
| `transport:Er_transition_left` | `density_shape_alpha` | `1.5785389591463254e-02` | `1.5976850000000000e-02` | `1.198361e-02` |
| `transport:Er_transition_right` | `density_shape_power` | `-1.8191178260101894e-02` | `-8.8057857302222442e-04` | `1.965821e+01` |
| `transport:Er2_volume_average` | `density_shape_power` | `1.1723514237983892e+01` | `1.2970932535457298e+01` | `9.617029e-02` |
| `transport:Er_volume_average` | `density_shape_power` | `1.6963617698293820e-01` | `2.0482617935080327e-01` | `1.718042e-01` |
| `transport:electron_temperature_volume_average_keV` | `density_shape_power` | `-4.3211954898123711e-05` | `-4.5459636055511510e-05` | `4.944345e-02` |
| `transport:total_pressure_volume_average` | `vmec:RBC:1:0` | `-7.6644633630893411e-02` | `-7.6944139999999994e-02` | `3.892517e-03` |
| `transport:alpha_power_volume_average_mw_m3` | `density_shape_power` | `2.6728550609761139e-03` | `2.7169120101847946e-03` | `1.621582e-02` |
| `transport:bootstrap_current_softmax_abs_scaled` | `n0` | `-1.0774573904620000e-03` | `-1.0791150000000000e-03` | `1.536082e-03` |

Interpretation:

- `bootstrap_current_softmax_abs_scaled` is now in good agreement with the
  saved 16-step FD references; the worst available relative difference is
  `1.536082e-03`.
- The new frozen-root `ZBS:1:0` FD rows are all within `5.94e-03` relative
  difference or better, excluding the near-zero `smooth_root_proxy` row.
- `total_pressure_volume_average`, `alpha_power_volume_average_mw_m3`, and the
  main profile columns are also consistent at roughly `O(1e-2)` or better in
  the worst row, usually much better.
- The frozen-root `density_shape_alpha` run removes the previous very large
  transition mismatch. The largest remaining discrepancies are now concentrated
  in still-selected-root `density_shape_power` rows and the sharp/root-sensitive
  Er transition diagnostics.

### Current Geometry QI/MaxJ Rows vs Frozen Linearized FD/JVP

These are the current geometry rows from the same 8-parameter shared-payload
run, compared against the saved frozen-linearized geometry FD checks.

| Objective | Parameter | Shared reverse AD | Frozen FD | JVP/raw-block target | `rel(AD, FD)` | `rel(AD, JVP)` |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `geometry:boozer_qi_objective` | `RBC:1:0` | `5.9392927523334720e+00` | `5.9397387449796542e+00` | `5.9392891189187216e+00` | `7.508605e-05` | `6.117593e-07` |
| `geometry:boozer_qi_objective` | `ZBS:1:0` | `-1.2365499862721663e-01` | `-1.2367491295362752e-01` | `-1.2365550023172692e-01` | `1.610213e-04` | `4.056470e-06` |
| `geometry:boozer_maxj_objective` | `RBC:1:0` | `-3.8431400489698863e+03` | `-3.8425374640355458e+03` | `-3.8431351880877260e+03` | `1.568195e-04` | `1.265863e-06` |
| `geometry:boozer_maxj_objective` | `ZBS:1:0` | `-1.9205081920611992e+03` | `-1.9205101180950553e+03` | `-1.9205082810004324e+03` | `1.002876e-06` | `4.630957e-08` |

Conclusion:

- The current shared-payload geometry rows match the raw-block/JVP target very
  tightly for both `RBC:1:0` and `ZBS:1:0`.
- The frozen FD mismatch remains at the expected finite-difference level for
  these geometry rows.

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

### 16-Step Full-Transport Geometry FD: `RBC:1:0`

Current frozen-root FD run:

```bash
python ./examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --parameter RBC:1:0 \
  --geometry-fd-lane frozen_linearized \
  --accepted-step-limit 16 \
  --radau-jacobian-reuse-mode legacy \
  --replay-mode accepted \
  --initial-Er-root-ad jax_selected_root \
  --initial-Er-root-fd-root-lane frozen_linearized
```

Run summary:

- Mode: `transport_realtime_geometry_forward_fd`
- Parameter kind: `realtime_geometry`
- Baseline value: `0.000000e+00`
- FD step: `3.313117e-07`
- Geometry FD lane: `frozen_linearized`
- Root FD lane: `frozen_linearized`
- Output JSON: `outputs/autodiff_transport_lagged_ntx/realtime_geometry_fd/RBC_1_0_forward_fd_frozen_linearized_summary.json`

| Objective | Value | Full FD `d/dRBC:1:0` | Fixed-final-state explicit geometry FD | Baseline-geometry final-state FD |
| --- | ---: | ---: | ---: | ---: |
| `transport:softmax_Er` | `2.0493049813391263e+01` | `-7.8543180000000007e+01` | `0.0000000000000000e+00` | `-7.8543180000000007e+01` |
| `transport:smooth_root_proxy` | `9.8039215686309116e-03` | `-2.6179630000000000e-12` | `0.0000000000000000e+00` | `-2.6179630000000000e-12` |
| `transport:Er_transition_left` | `1.7729764966344089e+01` | `-2.0306530000000000e+01` | `0.0000000000000000e+00` | `-2.0306530000000000e+01` |
| `transport:Er_transition_right` | `1.8376267791735049e+01` | `-2.2497940000000000e+01` | `0.0000000000000000e+00` | `-2.2497940000000000e+01` |
| `transport:Er2_volume_average` | `2.3720890299502904e+02` | `5.4307290000000003e+03` | `3.4856740000000000e-01` | `5.4303800000000001e+03` |
| `transport:Er_volume_average` | `-3.4526122553412755e+00` | `1.3627310000000000e+02` | `-5.0665630000000000e-02` | `1.3632380000000001e+02` |
| `transport:electron_temperature_volume_average_keV` | `6.5646442870649961e+00` | `-2.3962400000000000e-02` | `-1.2263670000000000e-02` | `-1.1698730000000000e-02` |
| `transport:total_pressure_volume_average` | `3.4213473064475188e+01` | `-7.6998129999999998e-02` | `-7.3820770000000001e-02` | `-3.1773630000000001e-03` |
| `transport:alpha_power_volume_average_mw_m3` | `5.8935809218241009e-01` | `1.9566950000000000e-01` | `-1.8709600000000000e-03` | `1.9754040000000000e-01` |
| `transport:bootstrap_current_softmax_abs_scaled` | `1.4485504661986399e+00` | `-1.7671640000000000e+00` | `-1.6742240000000000e+00` | `-9.2914000000000002e-02` |

Current shared-payload reverse AD vs frozen-root FD:

| Objective | Reverse AD `d/dRBC:1:0` | Frozen-root FD `d/dRBC:1:0` | Abs diff | Rel diff |
| --- | ---: | ---: | ---: | ---: |
| `transport:softmax_Er` | `-7.8516612048142349e+01` | `-7.8543180000000007e+01` | `2.656795e-02` | `3.382592e-04` |
| `transport:smooth_root_proxy` | `-2.5108173983191352e-12` | `-2.6179630000000000e-12` | `1.071456e-13` | not meaningful |
| `transport:Er_transition_left` | `-2.0306396145716139e+01` | `-2.0306530000000000e+01` | `1.338543e-04` | `6.591687e-06` |
| `transport:Er_transition_right` | `-2.2498789234739608e+01` | `-2.2497940000000000e+01` | `8.492347e-04` | `3.774722e-05` |
| `transport:Er2_volume_average` | `5.4255042598612426e+03` | `5.4307290000000003e+03` | `5.224740e+00` | `9.620698e-04` |
| `transport:Er_volume_average` | `1.3669898081845315e+02` | `1.3627310000000000e+02` | `4.258808e-01` | `3.125201e-03` |
| `transport:electron_temperature_volume_average_keV` | `-2.3726977964542796e-02` | `-2.3962400000000000e-02` | `2.354220e-04` | `9.824643e-03` |
| `transport:total_pressure_volume_average` | `-7.6644633630893411e-02` | `-7.6998129999999998e-02` | `3.534964e-04` | `4.590973e-03` |
| `transport:alpha_power_volume_average_mw_m3` | `1.9544849790848531e-01` | `1.9566950000000000e-01` | `2.210021e-04` | `1.129466e-03` |
| `transport:bootstrap_current_softmax_abs_scaled` | `-1.7673950710878303e+00` | `-1.7671640000000000e+00` | `2.310711e-04` | `1.307581e-04` |

Relative-error change versus the previous `RBC:1:0` selected-root/full-FD row:

| Objective | Previous rel diff | Frozen-root rel diff |
| --- | ---: | ---: |
| `transport:softmax_Er` | `9.929168e-04` | `3.382592e-04` |
| `transport:Er_transition_left` | `2.847481e-04` | `6.591687e-06` |
| `transport:Er_transition_right` | `3.330222e-05` | `3.774722e-05` |
| `transport:Er2_volume_average` | `3.090910e-03` | `9.620698e-04` |
| `transport:Er_volume_average` | `1.007465e-03` | `3.125201e-03` |
| `transport:electron_temperature_volume_average_keV` | `1.039704e-02` | `9.824643e-03` |
| `transport:total_pressure_volume_average` | `3.892517e-03` | `4.590973e-03` |
| `transport:alpha_power_volume_average_mw_m3` | `3.249610e-03` | `1.129466e-03` |
| `transport:bootstrap_current_softmax_abs_scaled` | `1.375496e-04` | `1.307581e-04` |

### 16-Step Full-Transport Profile FD: `n0`

Current frozen-root FD run:

```bash
python ./examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --parameter n0 \
  --geometry-fd-lane frozen_linearized \
  --accepted-step-limit 16 \
  --radau-jacobian-reuse-mode legacy \
  --replay-mode accepted \
  --initial-Er-root-ad jax_selected_root \
  --initial-Er-root-fd-root-lane frozen_linearized
```

Run summary:

- Mode: `transport_realtime_geometry_forward_fd`
- Parameter kind: `profile`
- Baseline value: `4.210000e+00`
- FD step: `1.263000e-06`
- Root FD lane: `frozen_linearized`
- Output JSON: `outputs/autodiff_transport_lagged_ntx/realtime_geometry_fd/n0_forward_fd_frozen_linearized_summary.json`

| Objective | Value | Frozen-root 16-step FD `d/dn0` |
| --- | ---: | ---: |
| `transport:softmax_Er` | `2.0493040599404146e+01` | `-7.5788280000000000e+00` |
| `transport:smooth_root_proxy` | `9.8039215686309099e-03` | `0.0000000000000000e+00` |
| `transport:Er_transition_left` | `1.7729764197009018e+01` | `-1.4467480000000000e+00` |
| `transport:Er_transition_right` | `1.8376266968588908e+01` | `-1.6705960000000000e+00` |
| `transport:Er2_volume_average` | `2.3720952417865757e+02` | `6.6924850000000000e+02` |
| `transport:Er_volume_average` | `-3.4526135846189607e+00` | `1.6893630000000000e+01` |
| `transport:electron_temperature_volume_average_keV` | `6.5646440779302155e+00` | `-8.6577110000000000e-04` |
| `transport:total_pressure_volume_average` | `3.4213472702876274e+01` | `8.0596760000000000e+00` |
| `transport:alpha_power_volume_average_mw_m3` | `5.8935810845846526e-01` | `3.0339440000000000e-01` |
| `transport:bootstrap_current_softmax_abs_scaled` | `1.4485488845680283e+00` | `-1.0791150000000000e-03` |

Current shared-payload reverse AD vs frozen-root FD:

| Objective | Reverse AD | Frozen-root FD | Abs diff | Rel diff |
| --- | ---: | ---: | ---: | ---: |
| `transport:softmax_Er` | `-7.5788861196501287e+00` | `-7.5788280000000000e+00` | `5.811965e-05` | `7.668686e-06` |
| `transport:smooth_root_proxy` | `1.8468993787754641e-13` | `0.0000000000000000e+00` | `1.846899e-13` | not meaningful |
| `transport:Er_transition_left` | `-1.4467383415531603e+00` | `-1.4467480000000000e+00` | `9.658447e-06` | `6.675970e-06` |
| `transport:Er_transition_right` | `-1.6706147148780452e+00` | `-1.6705960000000000e+00` | `1.871488e-05` | `1.120252e-05` |
| `transport:Er2_volume_average` | `6.6923799716616873e+02` | `6.6924850000000000e+02` | `1.050283e-02` | `1.569347e-05` |
| `transport:Er_volume_average` | `1.6894704215412517e+01` | `1.6893630000000000e+01` | `1.074215e-03` | `6.358701e-05` |
| `transport:electron_temperature_volume_average_keV` | `-8.6619872749393867e-04` | `-8.6577110000000000e-04` | `4.276275e-07` | `4.939267e-04` |
| `transport:total_pressure_volume_average` | `8.0596720085197759e+00` | `8.0596760000000000e+00` | `3.991480e-06` | `4.952408e-07` |
| `transport:alpha_power_volume_average_mw_m3` | `3.0339361742164600e-01` | `3.0339440000000000e-01` | `7.825784e-07` | `2.579409e-06` |
| `transport:bootstrap_current_softmax_abs_scaled` | `-1.0774573904620000e-03` | `-1.0791150000000000e-03` | `1.657610e-06` | `1.536082e-03` |

Status:

- This supersedes the selected-root `n0_forward_fd_summary.json` values for
  AD-vs-FD comparison against the shared-payload reverse path.
- The frozen-root lane matches the reverse AD selected initial-Er branch. All
  nonzero rows are below `1.6e-03` relative error, with most rows at
  `O(1e-05)` or better.
- Compared with the earlier selected-root FD row, the Er/root-sensitive
  objectives improved substantially: for example `Er_transition_left` improved
  from `1.104426e-03` to `6.675970e-06` relative error. Two non-Er rows moved
  slightly the other way but remain small: `electron_temperature_volume_average_keV`
  changed from `4.576786e-05` to `4.939267e-04`, and
  `bootstrap_current_softmax_abs_scaled` changed from `1.334304e-03` to
  `1.536082e-03`.

### 16-Step Full-Transport Profile FD: `T0`

Current frozen-root FD run:

```bash
python ./examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --parameter T0 \
  --geometry-fd-lane frozen_linearized \
  --accepted-step-limit 16 \
  --radau-jacobian-reuse-mode legacy \
  --replay-mode accepted \
  --initial-Er-root-ad jax_selected_root \
  --initial-Er-root-fd-root-lane frozen_linearized
```

Run summary:

- Mode: `transport_realtime_geometry_forward_fd`
- Parameter kind: `profile`
- Baseline value: `1.780000e+01`
- FD step: `5.340000e-06`
- Root FD lane: `frozen_linearized`
- Output JSON: `outputs/autodiff_transport_lagged_ntx/realtime_geometry_fd/T0_forward_fd_frozen_linearized_summary.json`

| Objective | Value | Frozen-root 16-step FD `d/dT0` |
| --- | ---: | ---: |
| `transport:softmax_Er` | `2.0493040599404146e+01` | `5.5481180000000000e+00` |
| `transport:smooth_root_proxy` | `9.8039215686309099e-03` | `1.6242730000000000e-13` |
| `transport:Er_transition_left` | `1.7729764197009018e+01` | `1.8373090000000000e+00` |
| `transport:Er_transition_right` | `1.8376266968588908e+01` | `2.0017310000000000e+00` |
| `transport:Er2_volume_average` | `2.3720952417865757e+02` | `-3.6157180000000000e+02` |
| `transport:Er_volume_average` | `-3.4526135846189607e+00` | `-1.0115900000000000e+01` |
| `transport:electron_temperature_volume_average_keV` | `6.5646440779302155e+00` | `3.5611920000000000e-01` |
| `transport:total_pressure_volume_average` | `3.4213472702876274e+01` | `1.8650300000000000e+00` |
| `transport:alpha_power_volume_average_mw_m3` | `5.8935810845846526e-01` | `6.9382040000000000e-02` |
| `transport:bootstrap_current_softmax_abs_scaled` | `1.4485488845680283e+00` | `2.3095560000000000e-01` |

Current shared-payload reverse AD vs frozen-root FD:

| Objective | Reverse AD | Frozen-root FD | Abs diff | Rel diff |
| --- | ---: | ---: | ---: | ---: |
| `transport:softmax_Er` | `5.5481041431530871e+00` | `5.5481180000000000e+00` | `1.385685e-05` | `2.497576e-06` |
| `transport:smooth_root_proxy` | `1.0913831787141299e-13` | `1.6242730000000000e-13` | `5.328898e-14` | not meaningful |
| `transport:Er_transition_left` | `1.8373054811266103e+00` | `1.8373090000000000e+00` | `3.518873e-06` | `1.915232e-06` |
| `transport:Er_transition_right` | `2.0017329575732847e+00` | `2.0017310000000000e+00` | `1.957573e-06` | `9.779402e-07` |
| `transport:Er2_volume_average` | `-3.6150089065989789e+02` | `-3.6157180000000000e+02` | `7.090934e-02` | `1.961141e-04` |
| `transport:Er_volume_average` | `-1.0123399603364410e+01` | `-1.0115900000000000e+01` | `7.499603e-03` | `7.413679e-04` |
| `transport:electron_temperature_volume_average_keV` | `3.5611894122635163e-01` | `3.5611920000000000e-01` | `2.587736e-07` | `7.266490e-07` |
| `transport:total_pressure_volume_average` | `1.8650270327188381e+00` | `1.8650300000000000e+00` | `2.967281e-06` | `1.591010e-06` |
| `transport:alpha_power_volume_average_mw_m3` | `6.9385082611546092e-02` | `6.9382040000000000e-02` | `3.042612e-06` | `4.385301e-05` |
| `transport:bootstrap_current_softmax_abs_scaled` | `2.3095562735965830e-01` | `2.3095560000000000e-01` | `2.735966e-08` | `1.184628e-07` |

Status:

- This supersedes the selected-root `T0_forward_fd_summary.json` values for
  AD-vs-FD comparison against the shared-payload reverse path.
- The frozen-root lane removes the earlier large `Er_transition_right`
  branch-sensitive mismatch: the relative error changed from `4.819827e-03`
  to `9.779402e-07`.

### 16-Step Full-Transport Profile FD: `density_shape_power`

Current FD run:

```bash
python ./examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --parameter density_shape_power \
  --geometry-fd-lane frozen_linearized \
  --accepted-step-limit 16 \
  --radau-jacobian-reuse-mode legacy \
  --replay-mode accepted \
  --initial-Er-root-ad jax_selected_root
```

Run summary:

- Mode: `transport_realtime_geometry_forward_fd`
- Parameter kind: `profile`
- Baseline value: `1.000000e+01`
- FD step: `3.000000e-06`
- Output JSON: `outputs/autodiff_transport_lagged_ntx/realtime_geometry_fd/density_shape_power_forward_fd_summary.json`

| Objective | Value | 16-step FD `d/ddensity_shape_power` |
| --- | ---: | ---: |
| `transport:softmax_Er` | `2.0493040599404146e+01` | `-1.4622070000000000e-01` |
| `transport:smooth_root_proxy` | `9.8039215686309099e-03` | `0.0000000000000000e+00` |
| `transport:Er_transition_left` | `1.7729764197009018e+01` | `-1.4053420000000000e-02` |
| `transport:Er_transition_right` | `1.8376266968588908e+01` | `-8.8057860000000005e-04` |
| `transport:Er2_volume_average` | `2.3720952417865757e+02` | `1.2970930000000000e+01` |
| `transport:Er_volume_average` | `-3.4526135846189607e+00` | `2.0482620000000000e-01` |
| `transport:electron_temperature_volume_average_keV` | `6.5646440779302155e+00` | `-4.5459640000000003e-05` |
| `transport:total_pressure_volume_average` | `3.4213472702876274e+01` | `2.4425430000000000e-01` |
| `transport:alpha_power_volume_average_mw_m3` | `5.8935810845846526e-01` | `2.7169120000000001e-03` |
| `transport:bootstrap_current_softmax_abs_scaled` | `1.4485488845680283e+00` | `-1.3221880000000000e-02` |

Status:

- This is the fresh profile-parameter FD reference for
  `density_shape_power` after the recent finite-volume boundary/evaluated-state
  updates.
- This selected-root FD lane is branch-sensitive for the initial ambipolar
  root and should be kept as a diagnostic, not as the final branch-compatible
  AD comparison.

Frozen-linearized initial-Er root FD run:

```bash
python ./examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --parameter density_shape_power \
  --geometry-fd-lane frozen_linearized \
  --accepted-step-limit 16 \
  --radau-jacobian-reuse-mode legacy \
  --replay-mode accepted \
  --initial-Er-root-ad jax_selected_root \
  --initial-Er-root-fd-root-lane frozen_linearized
```

Frozen-root run summary:

- Mode: `transport_realtime_geometry_forward_fd`
- Parameter kind: `profile`
- Baseline value: `1.000000e+01`
- FD step: `3.000000e-06`
- Root FD lane: `frozen_linearized`
- Output JSON: `outputs/autodiff_transport_lagged_ntx/realtime_geometry_fd/density_shape_power_forward_fd_frozen_linearized_summary.json`

| Objective | Value | Frozen-root 16-step FD `d/ddensity_shape_power` |
| --- | ---: | ---: |
| `transport:softmax_Er` | `2.0493040653150882e+01` | `-1.4061170000000000e-01` |
| `transport:smooth_root_proxy` | `9.8039215686309099e-03` | `-2.8912060000000000e-13` |
| `transport:Er_transition_left` | `1.7729764216611777e+01` | `-1.2781400000000000e-02` |
| `transport:Er_transition_right` | `1.8376267082555081e+01` | `-1.8178450000000000e-02` |
| `transport:Er2_volume_average` | `2.3720951871159608e+02` | `1.1721080000000000e+01` |
| `transport:Er_volume_average` | `-3.4526136020634448e+00` | `1.6956640000000000e-01` |
| `transport:electron_temperature_volume_average_keV` | `6.5646440796139922e+00` | `-4.3206920000000000e-05` |
| `transport:total_pressure_volume_average` | `3.4213472705740244e+01` | `2.4425500000000000e-01` |
| `transport:alpha_power_volume_average_mw_m3` | `5.8935810831572644e-01` | `2.6727670000000000e-03` |
| `transport:bootstrap_current_softmax_abs_scaled` | `1.4485486870521171e+00` | `-1.3221700000000000e-02` |

Current shared-payload reverse AD vs frozen-root FD:

| Objective | Reverse AD | Frozen-root FD | Abs diff | Rel diff |
| --- | ---: | ---: | ---: | ---: |
| `transport:softmax_Er` | `-1.4065516873156680e-01` | `-1.4061170000000000e-01` | `4.346873e-05` | `3.091402e-04` |
| `transport:smooth_root_proxy` | `6.5876403693064662e-18` | `-2.8912060000000000e-13` | `2.891272e-13` | not meaningful |
| `transport:Er_transition_left` | `-1.2789965811649196e-02` | `-1.2781400000000000e-02` | `8.565812e-06` | `6.701779e-04` |
| `transport:Er_transition_right` | `-1.8191168425367180e-02` | `-1.8178450000000000e-02` | `1.271843e-05` | `6.996430e-04` |
| `transport:Er2_volume_average` | `1.1723514222734334e+01` | `1.1721080000000000e+01` | `2.434223e-03` | `2.076790e-04` |
| `transport:Er_volume_average` | `1.6963617690233440e-01` | `1.6956640000000000e-01` | `6.977690e-05` | `4.115019e-04` |
| `transport:electron_temperature_volume_average_keV` | `-4.3211955291746346e-05` | `-4.3206920000000000e-05` | `5.035292e-09` | `1.165390e-04` |
| `transport:total_pressure_volume_average` | `2.4425502141708200e-01` | `2.4425500000000000e-01` | `2.141708e-08` | `8.768329e-08` |
| `transport:alpha_power_volume_average_mw_m3` | `2.6728549616084540e-03` | `2.6727670000000000e-03` | `8.796161e-08` | `3.291032e-05` |
| `transport:bootstrap_current_softmax_abs_scaled` | `-1.3222637714283003e-02` | `-1.3221700000000000e-02` | `9.377143e-07` | `7.092237e-05` |

Interpretation:

- The previous large `density_shape_power` mismatches were caused by the FD
  lane reselecting/changing the initial ambipolar root branch.
- The frozen-root full-transport FD lane differentiates the same baseline
  selected branch as the reverse AD rule and brings all nonzero rows to
  approximately `O(1e-3)` relative error or better.

### 16-Step Full-Transport Profile FD: `density_shape_alpha`

Current frozen-root FD run:

```bash
python ./examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --parameter density_shape_alpha \
  --geometry-fd-lane frozen_linearized \
  --accepted-step-limit 16 \
  --radau-jacobian-reuse-mode legacy \
  --replay-mode accepted \
  --initial-Er-root-ad jax_selected_root \
  --initial-Er-root-fd-root-lane frozen_linearized
```

Run summary:

- Mode: `transport_realtime_geometry_forward_fd`
- Parameter kind: `profile`
- Baseline value: `1.000000e+00`
- FD step: `3.000000e-07`
- Root FD lane: `frozen_linearized`
- Output JSON: `outputs/autodiff_transport_lagged_ntx/realtime_geometry_fd/density_shape_alpha_forward_fd_frozen_linearized_summary.json`

| Objective | Value | Frozen-root 16-step FD `d/ddensity_shape_alpha` |
| --- | ---: | ---: |
| `transport:softmax_Er` | `2.0493040599404146e+01` | `2.6128750000000000e-01` |
| `transport:smooth_root_proxy` | `9.8039215686309099e-03` | `0.0000000000000000e+00` |
| `transport:Er_transition_left` | `1.7729764197009018e+01` | `1.5976850000000000e-02` |
| `transport:Er_transition_right` | `1.8376266968588908e+01` | `2.3610870000000000e-02` |
| `transport:Er2_volume_average` | `2.3720952417865757e+02` | `-2.5533480000000000e+01` |
| `transport:Er_volume_average` | `-3.4526135846189607e+00` | `-4.4494800000000000e-01` |
| `transport:electron_temperature_volume_average_keV` | `6.5646440779302155e+00` | `5.3346070000000000e-04` |
| `transport:total_pressure_volume_average` | `3.4213472702876274e+01` | `-1.3265840000000000e+00` |
| `transport:alpha_power_volume_average_mw_m3` | `5.8935810845846526e-01` | `-8.2351060000000000e-03` |
| `transport:bootstrap_current_softmax_abs_scaled` | `1.4485488845680283e+00` | `1.1406510000000000e-01` |

Current shared-payload reverse AD vs frozen-root FD:

| Objective | Reverse AD | Frozen-root FD | Abs diff | Rel diff |
| --- | ---: | ---: | ---: | ---: |
| `transport:softmax_Er` | `2.6082258582205270e-01` | `2.6128750000000000e-01` | `4.649142e-04` | `1.779320e-03` |
| `transport:smooth_root_proxy` | `-1.1970961861304089e-17` | `0.0000000000000000e+00` | `1.197096e-17` | not meaningful |
| `transport:Er_transition_left` | `1.5785389591463254e-02` | `1.5976850000000000e-02` | `1.914604e-04` | `1.198361e-02` |
| `transport:Er_transition_right` | `2.3817948337074714e-02` | `2.3610870000000000e-02` | `2.070783e-04` | `8.770466e-03` |
| `transport:Er2_volume_average` | `-2.5500745056169166e+01` | `-2.5533480000000000e+01` | `3.273494e-02` | `1.282040e-03` |
| `transport:Er_volume_average` | `-4.4400532599246079e-01` | `-4.4494800000000000e-01` | `9.426740e-04` | `2.118616e-03` |
| `transport:electron_temperature_volume_average_keV` | `5.3339610943692328e-04` | `5.3346070000000000e-04` | `6.459056e-08` | `1.210784e-04` |
| `transport:total_pressure_volume_average` | `-1.3265837825838147e+00` | `-1.3265840000000000e+00` | `2.174162e-07` | `1.638918e-07` |
| `transport:alpha_power_volume_average_mw_m3` | `-8.2338695461871159e-03` | `-8.2351060000000000e-03` | `1.236454e-06` | `1.501442e-04` |
| `transport:bootstrap_current_softmax_abs_scaled` | `1.1406429742135712e-01` | `1.1406510000000000e-01` | `8.025786e-07` | `7.036146e-06` |

Status:

- This supersedes the selected-root `density_shape_alpha_forward_fd_summary.json`
  values for AD-vs-FD comparison against the shared-payload reverse path.
- The branch-sensitive transition rows improved substantially compared with the
  selected-root FD row: `Er_transition_left` changed from `4.076696e+00` to
  `1.198361e-02` relative error, and `Er_transition_right` changed from
  `8.790283e-01` to `8.770466e-03`.

### 16-Step Full-Transport Profile FD: `temperature_shape_alpha`

Current frozen-root FD run:

```bash
python ./examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --parameter temperature_shape_alpha \
  --geometry-fd-lane frozen_linearized \
  --accepted-step-limit 16 \
  --radau-jacobian-reuse-mode legacy \
  --replay-mode accepted \
  --initial-Er-root-ad jax_selected_root \
  --initial-Er-root-fd-root-lane frozen_linearized
```

Run summary:

- Mode: `transport_realtime_geometry_forward_fd`
- Parameter kind: `profile`
- Baseline value: `1.000000e+00`
- FD step: `3.000000e-07`
- Root FD lane: `frozen_linearized`
- Output JSON: `outputs/autodiff_transport_lagged_ntx/realtime_geometry_fd/temperature_shape_alpha_forward_fd_frozen_linearized_summary.json`

| Objective | Value | 16-step FD `d/dtemperature_shape_alpha` |
| --- | ---: | ---: |
| `transport:softmax_Er` | `2.0493040599404146e+01` | `1.3627480000000000e+01` |
| `transport:smooth_root_proxy` | `9.8039215686309099e-03` | `0.0000000000000000e+00` |
| `transport:Er_transition_left` | `1.7729764197009018e+01` | `1.6101590000000000e+01` |
| `transport:Er_transition_right` | `1.8376266968588908e+01` | `1.6294360000000000e+01` |
| `transport:Er2_volume_average` | `2.3720952417865757e+02` | `1.5504050000000000e+02` |
| `transport:Er_volume_average` | `-3.4526135846189607e+00` | `2.8288200000000001e+00` |
| `transport:electron_temperature_volume_average_keV` | `6.5646440779302155e+00` | `-3.0435330000000000e+00` |
| `transport:total_pressure_volume_average` | `3.4213472702876274e+01` | `-1.4519690000000000e+01` |
| `transport:alpha_power_volume_average_mw_m3` | `5.8935810845846526e-01` | `-4.1291900000000001e-01` |
| `transport:bootstrap_current_softmax_abs_scaled` | `1.4485488845680283e+00` | `-3.5233890000000001e+00` |

Status:

- This supersedes the selected-root `temperature_shape_alpha_forward_fd_summary.json`
  values for AD-vs-FD comparison against the shared-payload reverse path.
- Compared with the earlier selected-root FD row, the most branch-sensitive
  relative errors improved sharply: `softmax_Er` changed from `4.101955e-03`
  to `2.422859e-06`, `Er2_volume_average` from `7.332884e-02` to
  `1.595100e-04`, `Er_volume_average` from `1.091484e-01` to
  `4.080445e-04`, and `alpha_power_volume_average_mw_m3` from
  `1.067249e-03` to `4.270697e-06`.

### 16-Step Full-Transport Profile FD: `temperature_shape_power`

Current FD run:

```bash
python ./examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --parameter temperature_shape_power \
  --geometry-fd-lane frozen_linearized \
  --accepted-step-limit 16 \
  --radau-jacobian-reuse-mode legacy \
  --replay-mode accepted \
  --initial-Er-root-ad jax_selected_root
```

Run summary:

- Mode: `transport_realtime_geometry_forward_fd`
- Parameter kind: `profile`
- Baseline value: `2.000000e+00`
- FD step: `6.000000e-07`
- Output JSON: `outputs/autodiff_transport_lagged_ntx/realtime_geometry_fd/temperature_shape_power_forward_fd_summary.json`

| Objective | Value | 16-step FD `d/dtemperature_shape_power` |
| --- | ---: | ---: |
| `transport:softmax_Er` | `2.0493040599404146e+01` | `7.0010100000000000e+00` |
| `transport:smooth_root_proxy` | `9.8039215686309099e-03` | `0.0000000000000000e+00` |
| `transport:Er_transition_left` | `1.7729764197009018e+01` | `-7.1740150000000003e+00` |
| `transport:Er_transition_right` | `1.8376266968588908e+01` | `-6.3494940000000000e+00` |
| `transport:Er2_volume_average` | `2.3720952417865757e+02` | `-9.9282710000000002e+02` |
| `transport:Er_volume_average` | `-3.4526135846189607e+00` | `-2.7863390000000000e+01` |
| `transport:electron_temperature_volume_average_keV` | `6.5646440779302155e+00` | `1.5250080000000001e+00` |
| `transport:total_pressure_volume_average` | `3.4213472702876274e+01` | `7.7515860000000001e+00` |
| `transport:alpha_power_volume_average_mw_m3` | `5.8935810845846526e-01` | `2.5035850000000002e-01` |
| `transport:bootstrap_current_softmax_abs_scaled` | `1.4485488845680283e+00` | `1.5244090000000001e+00` |

Status:

- This is the fresh profile-parameter FD reference for
  `temperature_shape_power` after the recent finite-volume
  boundary/evaluated-state updates.
- This selected-root FD lane is branch-sensitive for the initial ambipolar
  root and should be kept as a diagnostic.

Frozen-linearized initial-Er root FD run:

```bash
python ./examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --parameter temperature_shape_power \
  --geometry-fd-lane frozen_linearized \
  --accepted-step-limit 16 \
  --radau-jacobian-reuse-mode legacy \
  --replay-mode accepted \
  --initial-Er-root-ad jax_selected_root \
  --initial-Er-root-fd-root-lane frozen_linearized
```

Frozen-root run summary:

- Mode: `transport_realtime_geometry_forward_fd`
- Parameter kind: `profile`
- Baseline value: `2.000000e+00`
- FD step: `6.000000e-07`
- Root FD lane: `frozen_linearized`
- Output JSON: `outputs/autodiff_transport_lagged_ntx/realtime_geometry_fd/temperature_shape_power_forward_fd_frozen_linearized_summary.json`

| Objective | Value | Frozen-root 16-step FD `d/dtemperature_shape_power` |
| --- | ---: | ---: |
| `transport:softmax_Er` | `2.0493040653150882e+01` | `7.0011720000000004e+00` |
| `transport:smooth_root_proxy` | `9.8039215686309099e-03` | `0.0000000000000000e+00` |
| `transport:Er_transition_left` | `1.7729764216611777e+01` | `-7.1805420000000002e+00` |
| `transport:Er_transition_right` | `1.8376267082555081e+01` | `-6.2632549999999998e+00` |
| `transport:Er2_volume_average` | `2.3720951871159608e+02` | `-9.9263349999999991e+02` |
| `transport:Er_volume_average` | `-3.4526136020634448e+00` | `-2.7865130000000001e+01` |
| `transport:electron_temperature_volume_average_keV` | `6.5646440796139922e+00` | `1.5250040000000000e+00` |
| `transport:total_pressure_volume_average` | `3.4213472705740244e+01` | `7.7515490000000000e+00` |
| `transport:alpha_power_volume_average_mw_m3` | `5.8935810831572644e-01` | `2.5035980000000000e-01` |
| `transport:bootstrap_current_softmax_abs_scaled` | `1.4485486870521171e+00` | `1.5244080000000000e+00` |

Current shared-payload reverse AD vs frozen-root FD:

| Objective | Reverse AD | Frozen-root FD | Abs diff | Rel diff |
| --- | ---: | ---: | ---: | ---: |
| `transport:softmax_Er` | `7.0011320415517044e+00` | `7.0011720000000004e+00` | `3.995845e-05` | `5.707394e-06` |
| `transport:smooth_root_proxy` | `-1.3072700012652828e-12` | `0.0000000000000000e+00` | `1.307270e-12` | not meaningful |
| `transport:Er_transition_left` | `-7.1806451216258500e+00` | `-7.1805420000000002e+00` | `1.031216e-04` | `1.436126e-05` |
| `transport:Er_transition_right` | `-6.2631095050085079e+00` | `-6.2632549999999998e+00` | `1.454950e-04` | `2.322993e-05` |
| `transport:Er2_volume_average` | `-9.9263140345944248e+02` | `-9.9263349999999991e+02` | `2.096541e-03` | `2.112099e-06` |
| `transport:Er_volume_average` | `-2.7865259386504583e+01` | `-2.7865130000000001e+01` | `1.293865e-04` | `4.643312e-06` |
| `transport:electron_temperature_volume_average_keV` | `1.5250078231509216e+00` | `1.5250040000000000e+00` | `3.823151e-06` | `2.506978e-06` |
| `transport:total_pressure_volume_average` | `7.7515855900810005e+00` | `7.7515490000000000e+00` | `3.659008e-05` | `4.720357e-06` |
| `transport:alpha_power_volume_average_mw_m3` | `2.5036298217189984e-01` | `2.5035980000000000e-01` | `3.182172e-06` | `1.271039e-05` |
| `transport:bootstrap_current_softmax_abs_scaled` | `1.5244064573255467e+00` | `1.5244080000000000e+00` | `1.542674e-06` | `1.011983e-06` |

Interpretation:

- The frozen-root full-transport FD lane agrees very tightly with the current
  shared-payload reverse AD for `temperature_shape_power`; all nonzero rows are
  below `2.4e-05` relative error.

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

Historical comparison against saved 16-step full-transport FD `d/dRBC:1:0`:

This table predates the refreshed 2026-08-09 `RBC:1:0` FD reference above.
Keep it as a record of the earlier match, but rerun reverse AD before using it
as the active comparison.

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
- Full-transport rows, including the corrected bootstrap row, were consistent
  with the older saved FD references shown in the historical table above.
- The initial-Er root-only/ambipolarity bootstrap geometry derivative was also
  validated against FD: `d/dRBC:1:0` AD `-1.7061866715282692e+00` vs FD
  `-1.7058819999999999e+00`, relative difference `1.786006e-04`.
