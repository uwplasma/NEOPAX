# Generic AD16/FD16 after the fixed VMEC Fourier projection

Run: `outputs/fixed_projection_ad_fd16_gu6ZO2/black_box`, NEOPAX commit
`6cbe394`, JAX 0.10.0, CUDA, FP64, `raw_block` VMEX response and the
existing fixed solver-mean benchmark control. All eight DOFs and thirteen
transport objectives completed. All 104 full-precision gradient pairs in
this report were checked against the subsequently received output archive.

All eight unperturbed baselines have exactly matching initial state, final
state, final time, transport geometry and coefficient-table arrays. The
separate current-geometry trace also passed exact AD/FD comparison. These
checks validate baseline consistency, not equality of the gradients.

The CUDA projection tests passed value, JVP, VJP, second derivatives and
bit equality across the tested JIT/layout/batched contexts. Isolated timing
and compiled buffers were:

| Measurement | Previous native projection | Fixed product |
| --- | ---: | ---: |
| Median synchronized call | `1.459950e-4 s` | `1.291130e-4 s` |
| Temporary device buffers | `5,013,520 bytes` | `0 bytes` |
| Generated code | `2,984 bytes` | `5,024 bytes` |

These are isolated compiled-kernel measurements, not full-run GPU memory
peaks or a controlled full-solver performance comparison.

Each table entry below is `abs(AD-FD)/abs(FD)`. The [CSV](fixed_projection_generic_ad_fd16_2026-11-25.csv)
also retains both gradients and absolute differences for all 104 pairs.

| Objective | n0 | T0 | density_shape_power | temperature_shape_power | density_shape_alpha | temperature_shape_alpha | RBC:1:0 | ZBS:1:0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| softmax_Er | `1.794e-09` | `2.372e-09` | `2.456e-08` | `1.607e-08` | `2.050e-07` | `1.703e-08` | `8.382e-06` | `1.651e-05` |
| net_total_power_volume_average_mw_m3 | `7.606e-11` | `9.976e-11` | `1.888e-09` | `1.821e-10` | `6.804e-07` | `8.236e-10` | `5.892e-06` | `3.662e-06` |
| Er_transition_left | `8.965e-09` | `1.669e-09` | `1.349e-06` | `6.866e-09` | `1.002e-06` | `1.927e-08` | `6.329e-06` | `5.937e-05` |
| Er_transition_right | `2.857e-08` | `3.711e-09` | `2.601e-07` | `1.259e-09` | `2.022e-06` | `1.229e-08` | `5.661e-06` | `1.218e-04` |
| Er2_volume_average | `5.693e-09` | `7.174e-10` | `4.939e-09` | `1.003e-08` | `5.672e-08` | `3.196e-09` | `1.046e-05` | `1.766e-06` |
| Er_volume_average | `6.238e-10` | `4.205e-10` | `2.573e-09` | `5.660e-09` | `3.943e-08` | `2.469e-09` | `6.802e-06` | `4.440e-06` |
| electron_temperature_volume_average_keV | `8.855e-07` | `1.654e-10` | `4.554e-07` | `6.567e-10` | `1.967e-06` | `3.392e-10` | `3.390e-08` | `2.108e-06` |
| total_pressure_volume_average | `2.342e-10` | `1.571e-10` | `9.479e-10` | `4.274e-10` | `1.326e-09` | `7.444e-10` | `1.535e-07` | `2.170e-06` |
| alpha_power_volume_average_mw_m3 | `3.522e-10` | `5.076e-11` | `5.520e-09` | `5.411e-11` | `2.754e-09` | `3.025e-11` | `4.499e-06` | `3.588e-06` |
| bootstrap_current_softmax_abs_scaled | `5.095e-10` | `8.195e-10` | `1.771e-09` | `4.312e-09` | `3.765e-08` | `4.896e-09` | `4.648e-03` | `5.635e-04` |
| Er_transition_rho | `1.020e-08` | `1.097e-09` | `5.426e-08` | `2.817e-08` | `2.776e-06` | `8.830e-09` | `1.763e-05` | `1.029e-05` |
| Er_transition_location_moment | `1.020e-08` | `1.097e-09` | `5.426e-08` | `2.817e-08` | `2.776e-06` | `8.830e-09` | `1.763e-05` | `1.029e-05` |
| Er_transition_strength | `7.008e-09` | `9.496e-09` | `3.980e-08` | `4.310e-09` | `6.937e-08` | `1.384e-09` | `1.094e-04` | `1.136e-06` |

The largest remaining geometry-gradient differences are bootstrap current:
`4.647558915e-3` for RBC and `5.635140498e-4` for ZBS. The softmax-Er
differences are `8.381627438e-6` and `1.650856705e-5`, respectively. The
largest profile-DOF difference is `2.776245106e-6`, for transition radius
and its location moment versus density shape alpha. The cause of the
remaining geometry-gradient differences is unresolved; matching baselines
do not justify attributing them to FD noise or derivative rules.

## Saved support-chain localization

The received archive also contains the terminal-state and pre-scan-fold
support cotangents and both geometry FD endpoints. NumPy-only contractions
of those cotangents with the saved FD support secants give:

| Parameter and objective | Full production AD versus FD | Support-chain contraction versus FD |
| --- | ---: | ---: |
| RBC bootstrap current | `4.647558915e-3` | `1.455171e-8` |
| ZBS bootstrap current | `5.635140498e-4` | `3.970949e-10` |
| RBC softmax Er | `8.381627438e-6` | `2.049451e-9` |
| ZBS softmax Er | `1.650856705e-5` | `1.019911e-8` |

Both columns use `abs(value-FD)/abs(FD)`. All thirteen support-chain
comparisons are retained in the [26-row CSV](fixed_projection_support_chain_2026-11-25.csv).
Their maximum relative differences are `3.467395e-7` for RBC and
`1.361513e-7` for ZBS. Initial state, final state and time match exactly;
saved ZBS objective values differ by at most `8.881784e-16`.

Thus the transport adjoint, when applied to the observed FD support direction,
reproduces the full FD derivative much more closely than the final production
parameter derivative. The large observed gap enters in the subsequent
geometry/NTX parameter-to-support mapping, or in the FD secant used to check
that mapping. This does not distinguish an AD-rule error from forward
rounding in the NTX solve, and does not establish every transport derivative
in every direction. Prepared NTX matrices and endpoint native scan inputs
were not retained, so that boundary cannot be diagnosed from this archive alone.

Generic AD wall time was `47:01.90`, with `9,647,056 KiB` maximum host RSS.
RBC and ZBS FD wall times were `57:23.85` and `55:55.75`, with maximum host
RSS `6,453,540 KiB` and `6,464,408 KiB`. These are host measurements and do
not establish GPU peaks or a speed change caused by the correction.

The NTSS1D kernel checks passed all nine tests. A subsequent log shows that
NTSS1D AD and four profile FD columns completed, followed by a coefficient
database mismatch in the density-shape-alpha baseline build. Its results and
failure boundary are recorded in the
[partial NTSS1D report](fixed_projection_ntss1d_partial_ad_fd16_2026-11-25.md).
The full comparison is incomplete. No numerical settings or methods were
changed while analysing these logs.
