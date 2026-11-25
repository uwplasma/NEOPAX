# Partial NTSS1D AD16/FD16 after the fixed VMEC Fourier projection

Run: `outputs/fixed_projection_ad_fd16_gu6ZO2/ntss1d`, NEOPAX `6cbe394`,
JAX 0.10.0, CUDA, FP64 and the existing fixed solver-mean control. AD16
completed. Four FD columns completed with exact baseline state, final time,
geometry and database equality: n0, T0, density shape power and temperature
shape power. Their 52 comparisons below use the full-precision printed AD
and FD gradients. Each entry is `abs(AD-FD)/abs(FD)`. The [CSV](fixed_projection_ntss1d_partial_ad_fd16_2026-11-25.csv)
retains both gradients and absolute differences.

| Objective | n0 | T0 | density_shape_power | temperature_shape_power |
| --- | ---: | ---: | ---: | ---: |
| softmax_Er | `1.796e-09` | `1.102e-09` | `9.270e-08` | `1.090e-08` |
| net_total_power_volume_average_mw_m3 | `3.390e-10` | `5.714e-10` | `3.519e-08` | `1.926e-10` |
| Er_transition_left | `1.330e-08` | `8.423e-09` | `2.250e-07` | `9.027e-09` |
| Er_transition_right | `2.316e-08` | `3.662e-09` | `1.035e-06` | `1.020e-08` |
| Er2_volume_average | `7.534e-09` | `4.007e-10` | `3.706e-09` | `4.220e-09` |
| Er_volume_average | `1.616e-10` | `4.849e-10` | `4.346e-09` | `2.874e-09` |
| electron_temperature_volume_average_keV | `1.988e-07` | `1.792e-10` | `1.638e-06` | `6.321e-10` |
| total_pressure_volume_average | `2.842e-10` | `1.907e-10` | `4.506e-09` | `4.487e-10` |
| alpha_power_volume_average_mw_m3 | `4.110e-10` | `3.899e-10` | `1.677e-08` | `2.923e-10` |
| bootstrap_current_softmax_abs_scaled | `2.801e-09` | `1.243e-09` | `1.292e-07` | `2.869e-09` |
| Er_transition_rho | `1.218e-08` | `4.973e-09` | `4.097e-07` | `6.741e-09` |
| Er_transition_location_moment | `1.218e-08` | `4.973e-09` | `4.097e-07` | `6.741e-09` |
| Er_transition_strength | `1.264e-10` | `2.457e-08` | `3.671e-08` | `7.794e-09` |

The next FD worker, density shape alpha, stopped in the unperturbed
geometry/database reference guard, before transport stepping, root freezing
for endpoints or any FD perturbation. Only the coefficient tables were
reported as different:

| Field | Differing entries | Maximum absolute difference |
| --- | ---: | ---: |
| D11_log | 978 | `1.2163237483875378e-6` |
| D13 | 1232 | `2.230284553617068e-5` |
| D33 | 929 | `1.160430045499261e-6` |

No NTSS1D FD result is available for density shape alpha, temperature shape
alpha, RBC or ZBS. The passed NTSS1D interpolation kernel tests do not
establish reproducibility of this coefficient build. The exact guard remains
active. No tolerances, solver settings or numerical methods were changed.

## Received archive: first differing saved boundary

The failed density-alpha reference capture was compared with the successful
NTSS1D n0 reference capture. Shape, dtype and bytes match for all 25 saved
transport-geometry fields, 19 scan-channel fields and 147 native-surface
fields. Only the three database coefficient tables differ among the 19
saved database fields. The downstream initial Er differs at all 51 radii,
with maximum absolute difference `5.405316169834862e-3`.

The successful generic n0 capture and successful NTSS1D n0 capture also have
identical saved upstream fields and identical D11_log, D13 and D33 tables.
This compares the common coefficient build only; it does not compare gradients
between different interpolators. All 52 completed NTSS1D gradient pairs in
the report were checked against the received CSV.

The coefficient differences are strongest at the lowest collision frequencies:
maximum D13 differences over radius/field for the first three frequency
indices are `2.230285e-5`, `5.4679e-6` and `6.8698e-9`, falling toward
roundoff at higher frequencies. This pattern suggests numerical sensitivity,
but does not identify the first differing operation.

The archive has neither actual prepared NTX arrays nor raw coefficient-scan
results. Its first differing saved boundary therefore lies between the
identical numeric native inputs and the normalized coefficient tables.
Preparation, solve and normalization must remain distinct possible causes
until their actual arrays/executables are captured. The selected profile
parameter has not been perturbed here; neither time stepping nor FD endpoint
freezing can cause this particular failure.
