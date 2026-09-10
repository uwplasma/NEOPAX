# Database reverse AD versus FD status

## Completed one-step database reverse run

The black-box `ntx_scan_runtime` database benchmark completed its one accepted
step / one reverse segment validation run with `RBC:1:0,ZBS:1:0` and all
objectives.  The run used the fixed-table segment transpose, one recorded scan
fold, and compact VMEC JVP/bar contraction.

Observed successful boundary diagnostics:

```text
database final recorded-scan fold ready ... contract=one_batched_scan_transpose
compact_payload_tangent_contract=True
raw_block_param_bar_all_finite=True
raw_block_param_bar_first_nonfinite=None
```

The run completed and wrote:

```text
outputs/autodiff_transport_lagged_ntx/reverse_ad/
transport_reverse_ad_only_full_transport_shared_payload_smoke.json
```

This validates the database reverse plumbing for the one-step case.  It is not
yet a 16-step FD validation.

## Pure geometry objective comparison

The pure geometry rows should be independent of whether transport uses the
Lij flux model or the fixed scan database.  They are compared below with the
existing realtime eight-parameter snapshot in
`shared_payload_8param_benchmark_snapshot.md`.  That snapshot uses the same
VMEC geometry parameters, `RBC:1:0` and `ZBS:1:0`.

| Geometry objective | Database `d/dRBC:1:0` | Existing realtime | Relative difference | Database `d/dZBS:1:0` | Existing realtime | Relative difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `boozer_qi_objective` | 5.939289118755e+00 | 5.939292752373e+00 | 6.118e-07 | -1.236555009546e-01 | -1.236549986209e-01 | 4.062e-06 |
| `boozer_maxj_objective` | -3.843135187617e+03 | -3.843140049060e+03 | 1.265e-06 | -1.920508280424e+03 | -1.920508192140e+03 | 4.597e-08 |
| `vmec_aspect_ratio` | -5.400678418701e+00 | -5.400678418701e+00 | 0 | -5.522688575132e+00 | -5.522688575132e+00 | 0 |
| `vmec_iota_mean` | 2.440514060926e-01 | 2.440514067774e-01 | 2.806e-09 | 1.456752601982e-01 | 1.456752605577e-01 | 2.468e-09 |
| `vmec_magnetic_well` | -1.109011606553e-02 | -1.109011249287e-02 | 3.221e-07 | -4.168202748224e-02 | -4.168202646912e-02 | 2.431e-08 |
| `vmec_mirror_ratio` | -5.935909471405e-01 | -5.935909468960e-01 | 4.118e-10 | 4.143700612540e-01 | 4.143700613673e-01 | 2.735e-10 |

All six geometry rows agree with the existing realtime values to within
`4.1e-06` relative difference.  The differences are numerical-level changes
from the payload-transpose route; the pure VMEC rows are effectively identical.

## Completed 16-step / four-segment database reverse AD

This is the completed black-box database run with 16 accepted steps, four
reverse segments of length four, `block` stage-adjoint solves, and
`RBC:1:0,ZBS:1:0`.  Every database reverse trace checkpoint was finite.  The
recorded scan fold and compact VMEC payload tangent contraction both completed:

```text
database final recorded-scan fold ... objective_rows=10 groups=1
compact_payload_tangent_contract=True
raw_block_param_bar_all_finite=True
```

Elapsed time reported by the benchmark: `2488.672 s`.

| Objective | Residual | `d/dRBC:1:0` reverse AD | `d/dZBS:1:0` reverse AD |
| --- | ---: | ---: | ---: |
| `transport:softmax_Er` | 2.134759724591e+01 | -1.818407580994e+01 | 1.466436166849e+01 |
| `transport:net_total_power_volume_average_mw_m3` | 5.080753004292e-01 | -1.212541090351e-03 | -6.625277989554e-03 |
| `transport:Er_transition_left` | 1.790225730111e+01 | -4.443282305723e+00 | 7.495554214422e+00 |
| `transport:Er_transition_right` | 1.866676994943e+01 | -5.881563655084e+00 | 8.201217022228e+00 |
| `transport:Er2_volume_average` | 2.697044451329e+02 | 4.138176591720e+01 | -1.956390243890e+01 |
| `transport:Er_volume_average` | -2.851164440603e+00 | -1.181510702272e+01 | 9.267949876401e+00 |
| `transport:electron_temperature_volume_average_keV` | 6.566732168571e+00 | -1.265699871516e-02 | -3.965621786972e-02 |
| `transport:total_pressure_volume_average` | 3.421739241364e+01 | -7.370940494543e-02 | -2.376398114501e-01 |
| `transport:alpha_power_volume_average_mw_m3` | 5.891952804440e-01 | -1.368913151274e-03 | -7.094775489872e-03 |
| `transport:bootstrap_current_softmax_abs_scaled` | 1.371734270888e+00 | -1.321165949153e+00 | -4.029156666342e-01 |
| `geometry:boozer_qi_objective` | 2.119202996427e-01 | 5.939289118713e+00 | -1.236555009430e-01 |
| `geometry:boozer_maxj_objective` | 4.438733257409e+02 | -3.843135187597e+03 | -1.920508280381e+03 |
| `geometry:vmec_aspect_ratio` | 1.001533091896e+01 | -5.400678418701e+00 | -5.522688575132e+00 |
| `geometry:vmec_iota_mean` | -5.936525996610e-01 | 2.440514060926e-01 | 1.456752601982e-01 |
| `geometry:vmec_magnetic_well` | -2.747612874968e-02 | -1.109011606553e-02 | -4.168202748224e-02 |
| `geometry:vmec_mirror_ratio` | 2.115380346716e-01 | -5.935909471405e-01 | 4.143700612540e-01 |
| `geometry:vmec_dmerc_stability_softmax` | 3.232961223588e+00 | -7.285994346626e+00 | -1.517899648971e+00 |

### Database transport-profile reverse-AD derivatives

The same completed 16-step / four-segment run printed the following six
profile columns.  They are preserved here rather than inferred from a
geometry-only table.

| Objective | `d/dn0` | `d/dT0` | `d/density_shape_power` | `d/dtemperature_shape_power` | `d/ddensity_shape_alpha` | `d/dtemperature_shape_alpha` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `transport:softmax_Er` | -2.870229283213e+00 | 2.774011882695e+00 | -8.097000909236e-02 | 1.871854895715e+00 | 1.531295492132e-01 | 1.076840729740e+01 |
| `transport:net_total_power_volume_average_mw_m3` | 2.414499437540e-01 | 8.101211771277e-02 | 1.001525151170e-03 | 2.745038988131e-01 | -2.450476062839e-04 | -3.939093535316e-01 |
| `transport:Er_transition_left` | -9.854028863394e-01 | 1.588983182064e+00 | -1.226591102776e-02 | -7.168755063746e+00 | 1.526013687084e-02 | 1.556805294983e+01 |
| `transport:Er_transition_right` | -1.099895549700e+00 | 1.691440444960e+00 | -1.708680519920e-02 | -6.281482073597e+00 | 2.255973257002e-02 | 1.557705491452e+01 |
| `transport:Er2_volume_average` | 2.584258925801e+00 | 3.132042405422e+01 | 2.282433626328e+00 | -1.193829755550e+01 | 3.086192380921e+00 | 1.069449215483e+02 |
| `transport:Er_volume_average` | -1.902403161953e+00 | 8.660546286795e-01 | -6.868899394956e-02 | -6.940587898690e-01 | -1.683207994445e-01 | 3.016580559690e+00 |
| `transport:electron_temperature_volume_average_keV` | 8.635230226620e-04 | 3.557791789982e-01 | -7.134485771230e-05 | 1.524801586181e+00 | 1.221864639035e-03 | -3.044927257787e+00 |
| `transport:total_pressure_volume_average` | 8.062428661443e+00 | 1.865417330779e+00 | 2.442669964820e-01 | 7.752335956477e+00 | -1.326505315369e+00 | -1.451660435636e+01 |
| `transport:alpha_power_volume_average_mw_m3` | 2.796210021162e-01 | 8.321610405521e-02 | 2.360917883672e-03 | 2.842041456067e-01 | -7.638350629771e-03 | -4.128986884945e-01 |
| `transport:bootstrap_current_softmax_abs_scaled` | 1.069765860401e-01 | 1.747940465558e-01 | -1.199930929329e-02 | 7.078898335030e-01 | 4.157598581050e-02 | -6.196723827141e-01 |

### Comparison with the wHe Lij realtime reverse-AD reference

The Lij reference is the full shared-payload wHe table in
`paper_results.md`.  The numbers below are the largest absolute relative
difference among the six profile columns for each objective, using the Lij
value as the denominator.  This is a flux-model comparison, not an AD-versus-
FD correctness result: `ntx_scan_runtime` and `ntx_exact_lij_runtime` have
different transport fluxes and therefore need not have identical transport
profile derivatives.

| Objective | largest profile difference versus Lij | profile column | profile sign change versus Lij |
| --- | ---: | --- | --- |
| `transport:softmax_Er` | 32.54% | `n0` | none |
| `transport:net_total_power_volume_average_mw_m3` | 220.42% | `density_shape_alpha` | `density_shape_alpha` |
| `transport:Er_transition_left` | 31.86% | `n0` | none |
| `transport:Er_transition_right` | 34.14% | `n0` | none |
| `transport:Er2_volume_average` | 140.69% | `density_shape_alpha` | `n0`, `temperature_shape_power`, `density_shape_alpha` |
| `transport:Er_volume_average` | 3551.10% | `n0` | `T0` |
| `transport:electron_temperature_volume_average_keV` | 116.31% | `density_shape_power` | none |
| `transport:total_pressure_volume_average` | 0.03% | `T0` | none |
| `transport:alpha_power_volume_average_mw_m3` | 0.90% | `T0` | none |
| `transport:bootstrap_current_softmax_abs_scaled` | 9322.21% | `n0` | `n0` |

The near agreement for total pressure and alpha power is useful evidence that
the profile-state/transport objective pieces remain consistent.  The large
differences are concentrated in Er and bootstrap-sensitive responses, where
the fixed database flux model changes the nonlinear transport solution.

The first ten rows are the database transport derivatives to compare with the
matching frozen-linearized database FD loop.  The last seven rows are pure
geometry derivatives and are independent of transport segmentation/database
use; the first six have already been compared above with the existing realtime
reference table.

## Completed frozen-linearized database profile FD comparison

These FD runs use the same database TOML, 16 accepted steps, accepted-schedule
replay, and frozen-linearized geometry and initial-Er-root lanes.  Thus the
comparison below isolates the same profile derivative represented by the
reverse-AD profile columns above.  FD values are emitted by the benchmark at
six significant digits, so the reported relative errors are bounded by that
printing precision.  Relative error is
`abs(AD - FD) / max(abs(AD), abs(FD))`.

### `n0`

| Objective | Reverse AD | FD | Relative error |
| --- | ---: | ---: | ---: |
| `softmax_Er` | -2.870229283213e+00 | -2.870229e+00 | 9.867e-08 |
| `net_total_power_volume_average_mw_m3` | 2.414499437540e-01 | 2.414499e-01 | 1.812e-07 |
| `Er_transition_left` | -9.854028863394e-01 | -9.854028e-01 | 8.762e-08 |
| `Er_transition_right` | -1.099895549700e+00 | -1.099896e+00 | 4.094e-07 |
| `Er2_volume_average` | 2.584258925801e+00 | 2.584259e+00 | 2.871e-08 |
| `Er_volume_average` | -1.902403161953e+00 | -1.902403e+00 | 8.513e-08 |
| `electron_temperature_volume_average_keV` | 8.635230226620e-04 | 8.635236e-04 | 6.686e-07 |
| `total_pressure_volume_average` | 8.062428661443e+00 | 8.062429e+00 | 4.199e-08 |
| `alpha_power_volume_average_mw_m3` | 2.796210021162e-01 | 2.796210e-01 | 7.568e-09 |
| `bootstrap_current_softmax_abs_scaled` | 1.069765860401e-01 | 1.069766e-01 | 1.305e-07 |

### `T0`

| Objective | Reverse AD | FD | Relative error |
| --- | ---: | ---: | ---: |
| `softmax_Er` | 2.774011882695e+00 | 2.774012e+00 | 4.229e-08 |
| `net_total_power_volume_average_mw_m3` | 8.101211771277e-02 | 8.101212e-02 | 2.823e-08 |
| `Er_transition_left` | 1.588983182064e+00 | 1.588983e+00 | 1.146e-07 |
| `Er_transition_right` | 1.691440444960e+00 | 1.691440e+00 | 2.631e-07 |
| `Er2_volume_average` | 3.132042405422e+01 | 3.132042e+01 | 1.294e-07 |
| `Er_volume_average` | 8.660546286795e-01 | 8.660546e-01 | 3.312e-08 |
| `electron_temperature_volume_average_keV` | 3.557791789982e-01 | 3.557792e-01 | 5.903e-08 |
| `total_pressure_volume_average` | 1.865417330779e+00 | 1.865417e+00 | 1.773e-07 |
| `alpha_power_volume_average_mw_m3` | 8.321610405521e-02 | 8.321610e-02 | 4.873e-08 |
| `bootstrap_current_softmax_abs_scaled` | 1.747940465558e-01 | 1.747940e-01 | 2.663e-07 |

### `density_shape_power`

| Objective | Reverse AD | FD | Relative error |
| --- | ---: | ---: | ---: |
| `softmax_Er` | -8.097000909236e-02 | -8.097001e-02 | 1.121e-08 |
| `net_total_power_volume_average_mw_m3` | 1.001525151170e-03 | 1.001525e-03 | 1.509e-07 |
| `Er_transition_left` | -1.226591102776e-02 | -1.226589e-02 | 1.714e-06 |
| `Er_transition_right` | -1.708680519920e-02 | -1.708679e-02 | 8.895e-07 |
| `Er2_volume_average` | 2.282433626328e+00 | 2.282434e+00 | 1.637e-07 |
| `Er_volume_average` | -6.868899394956e-02 | -6.868899e-02 | 5.750e-08 |
| `electron_temperature_volume_average_keV` | -7.134485771230e-05 | -7.134500e-05 | 1.994e-06 |
| `total_pressure_volume_average` | 2.442669964820e-01 | 2.442670e-01 | 1.440e-08 |
| `alpha_power_volume_average_mw_m3` | 2.360917883672e-03 | 2.360918e-03 | 4.927e-08 |
| `bootstrap_current_softmax_abs_scaled` | -1.199930929329e-02 | -1.199931e-02 | 5.890e-08 |

### `temperature_shape_power`

| Objective | Reverse AD | FD | Relative error |
| --- | ---: | ---: | ---: |
| `softmax_Er` | 1.871854895715e+00 | 1.871855e+00 | 5.571e-08 |
| `net_total_power_volume_average_mw_m3` | 2.745038988131e-01 | 2.745039e-01 | 4.324e-09 |
| `Er_transition_left` | -7.168755063746e+00 | -7.168755e+00 | 8.892e-09 |
| `Er_transition_right` | -6.281482073597e+00 | -6.281482e+00 | 1.172e-08 |
| `Er2_volume_average` | -1.193829755550e+01 | -1.193830e+01 | 2.048e-07 |
| `Er_volume_average` | -6.940587898690e-01 | -6.940588e-01 | 1.460e-08 |
| `electron_temperature_volume_average_keV` | 1.524801586181e+00 | 1.524802e+00 | 2.714e-07 |
| `total_pressure_volume_average` | 7.752335956477e+00 | 7.752336e+00 | 5.614e-09 |
| `alpha_power_volume_average_mw_m3` | 2.842041456067e-01 | 2.842041e-01 | 1.605e-07 |
| `bootstrap_current_softmax_abs_scaled` | 7.078898335030e-01 | 7.078898e-01 | 4.733e-08 |

### `density_shape_alpha`

| Objective | Reverse AD | FD | Relative error |
| --- | ---: | ---: | ---: |
| `softmax_Er` | 1.531295492132e-01 | 1.531296e-01 | 3.317e-07 |
| `net_total_power_volume_average_mw_m3` | -2.450476062839e-04 | -2.450479e-04 | 1.199e-06 |
| `Er_transition_left` | 1.526013687084e-02 | 1.526044e-02 | 1.986e-05 |
| `Er_transition_right` | 2.255973257002e-02 | 2.255976e-02 | 1.216e-06 |
| `Er2_volume_average` | 3.086192380921e+00 | 3.086193e+00 | 2.006e-07 |
| `Er_volume_average` | -1.683207994445e-01 | -1.683208e-01 | 3.300e-09 |
| `electron_temperature_volume_average_keV` | 1.221864639035e-03 | 1.221864e-03 | 5.230e-07 |
| `total_pressure_volume_average` | -1.326505315369e+00 | -1.326505e+00 | 2.377e-07 |
| `alpha_power_volume_average_mw_m3` | -7.638350629771e-03 | -7.638351e-03 | 4.847e-08 |
| `bootstrap_current_softmax_abs_scaled` | 4.157598581050e-02 | 4.157599e-02 | 1.008e-07 |

### `temperature_shape_alpha`

| Objective | Reverse AD | FD | Relative error |
| --- | ---: | ---: | ---: |
| `softmax_Er` | 1.076840729740e+01 | 1.076841e+01 | 2.510e-07 |
| `net_total_power_volume_average_mw_m3` | -3.939093535316e-01 | -3.939094e-01 | 1.180e-07 |
| `Er_transition_left` | 1.556805294983e+01 | 1.556805e+01 | 1.895e-07 |
| `Er_transition_right` | 1.557705491452e+01 | 1.557706e+01 | 3.265e-07 |
| `Er2_volume_average` | 1.069449215483e+02 | 1.069449e+02 | 2.015e-07 |
| `Er_volume_average` | 3.016580559690e+00 | 3.016581e+00 | 1.460e-07 |
| `electron_temperature_volume_average_keV` | -3.044927257787e+00 | -3.044927e+00 | 8.466e-08 |
| `total_pressure_volume_average` | -1.451660435636e+01 | -1.451660e+01 | 3.001e-07 |
| `alpha_power_volume_average_mw_m3` | -4.128986884945e-01 | -4.128987e-01 | 2.787e-08 |
| `bootstrap_current_softmax_abs_scaled` | -6.196723827141e-01 | -6.196724e-01 | 2.790e-08 |

All six completed profile FD columns agree with reverse AD.  The largest
displayed relative error is `1.986e-05` for the small
`Er_transition_left` derivative with respect to `density_shape_alpha`; this
is compatible with a six-significant-digit centered finite difference.  The
next largest is `1.994e-06` on the small
`electron_temperature_volume_average_keV` derivative with respect to
`density_shape_power`.  For `temperature_shape_alpha`, the largest displayed
relative error is `3.265e-07` (`Er_transition_right`).

## Initial ambipolar-Er root-only geometry FD

Configuration: database black-box realtime-VMEC benchmark; `RBC:1:0` and
`ZBS:1:0`; `geometry_fd_lane=frozen_linearized`;
`root_fd_lane=frozen_linearized`.
This is a no-Radau diagnostic of the selected initial ambipolar root and its
seven root-only objectives.  It is the matching FD lane for the new
database-specific root-only reverse boundary.

Baseline objective values:

| Objective | Value |
| --- | ---: |
| `softmax_Er` | 2.1328614487713111e+01 |
| `net_total_power_volume_average_mw_m3` | 5.0833886344044954e-01 |
| `Er_transition_left` | 1.7886676618640148e+01 |
| `Er_transition_right` | 1.8650083623318860e+01 |
| `Er2_volume_average` | 2.7635136665314184e+02 |
| `Er_volume_average` | -2.8566207362063278e+00 |
| `bootstrap_current_softmax_abs_scaled` | 1.3756556924177652e+00 |

The corrected database root-only reverse AD run completed with one recorded
scan transpose (`objective_rows=7`, all raw VMEC parameter bars finite).  It
includes the scan-owned `a_b` and `Er_list` coordinate bars for both the
selected-root flux and corrected-bootstrap Upar paths.

| Objective | RBC FD | RBC AD | RBC relative error | ZBS FD | ZBS AD | ZBS relative error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `softmax_Er` | -2.664020e+01 | -2.664034988650e+01 | 5.626e-06 | 5.781550e+00 | 5.782007566251e+00 | 7.915e-05 |
| `net_total_power_volume_average_mw_m3` | -1.716389e-03 | -1.716403818524e-03 | 8.634e-06 | -6.398319e-03 | -6.398310018111e-03 | 1.404e-06 |
| `Er_transition_left` | -1.338278e+01 | -1.338265092869e+01 | 9.645e-06 | -8.848652e-01 | -8.850610698362e-01 | 2.214e-04 |
| `Er_transition_right` | -1.515964e+01 | -1.515906028260e+01 | 3.824e-05 | -5.230563e-01 | -5.238769121529e-01 | 1.569e-03 |
| `Er2_volume_average` | -2.831677e+02 | -2.831596471801e+02 | 2.844e-05 | -3.579285e+02 | -3.578981342375e+02 | 8.484e-05 |
| `Er_volume_average` | -6.422814e+00 | -6.423623674892e+00 | 1.261e-04 | 1.387330e+01 | 1.387366143587e+01 | 2.605e-05 |
| `bootstrap_current_softmax_abs_scaled` | -1.989528e+00 | -1.986205686675e+00 | 1.670e-03 | -1.611476e+00 | -1.607858441965e+00 | 2.245e-03 |

All seven root-only objectives now agree with frozen-linearized FD for both
geometry parameters.  The largest discrepancy is 0.2245% on the ZBS
bootstrap derivative; every Er objective is within 0.157%.

### Compact LaTex table: database ambipolar root AD versus FD

The following is the publication-oriented subset of the preceding table.
Objectives are columns, parameter DOFs are rows, and the power column is the
**net** total-power objective (not alpha power).  Entries use
`abs(AD - FD) / max(abs(AD), abs(FD))`.

```latex
\begin{table}[t]
\centering
\def\arraystretch{1.5}
\scriptsize
\begin{tabular}{ |l||c|c|c|c|c| }
    \hline
    \noalign{\vskip -0.085in}
    DOF
      & $E_r^{\max}$ (softmax)
      & $E_{r,\mathrm{left}}$
      & $E_{r,\mathrm{right}}$
      & $P_{\mathrm{net}}$
      & $J_{\mathrm{boots}}$ \\[-1.5ex]
    \hline
    \noalign{\vskip -0.085in}
    $n_0$
      & $1.599\times10^{-7}$
      & $2.111\times10^{-8}$
      & $1.256\times10^{-7}$
      & $1.988\times10^{-7}$
      & $2.681\times10^{-7}$ \\
    $T_0$
      & $1.066\times10^{-7}$
      & $1.132\times10^{-7}$
      & $7.723\times10^{-8}$
      & $6.872\times10^{-9}$
      & $9.498\times10^{-8}$ \\
    $\alpha_n$
      & $2.353\times10^{-8}$
      & $5.039\times10^{-7}$
      & $4.651\times10^{-7}$
      & $4.412\times10^{-7}$
      & $1.282\times10^{-7}$ \\
    $\alpha_T$
      & $2.507\times10^{-8}$
      & $4.765\times10^{-8}$
      & $3.090\times10^{-8}$
      & $1.381\times10^{-7}$
      & $8.736\times10^{-9}$ \\
    $\beta_n$
      & $7.138\times10^{-7}$
      & $6.675\times10^{-6}$
      & $5.885\times10^{-6}$
      & $7.169\times10^{-7}$
      & $5.970\times10^{-9}$ \\
    $\beta_T$
      & $1.129\times10^{-7}$
      & $2.529\times10^{-7}$
      & $9.510\times10^{-8}$
      & $6.279\times10^{-8}$
      & $2.892\times10^{-8}$ \\
    $\mathrm{RBC}(1,0)$
      & $5.626\times10^{-6}$
      & $9.645\times10^{-6}$
      & $3.824\times10^{-5}$
      & $8.634\times10^{-6}$
      & $1.670\times10^{-3}$ \\
    $\mathrm{ZBS}(1,0)$
      & $7.915\times10^{-5}$
      & $2.214\times10^{-4}$
      & $1.569\times10^{-3}$
      & $1.404\times10^{-6}$
      & $2.245\times10^{-3}$ \\
    \\[-1.5ex]\hline
\end{tabular}
\caption{Relative errors between database ambipolar-root reverse AD and
frozen-linearized finite differences.}
\label{tab:database-ambipolar-ad-fd-relative-error}
\end{table}
```

Here $\alpha_n$, $\alpha_T$, $\beta_n$, and $\beta_T$ denote
`density_shape_power`, `temperature_shape_power`, `density_shape_alpha`, and
`temperature_shape_alpha`, respectively.  The profile FD values were recovered
from the saved root-only FD run record and are now included.

### Root-only reverse AD: complete parameter table

The same root-only reverse run included all six profile columns as well as
the two VMEC boundary columns (eight parameters total).  These are the AD
values from the run whose geometry columns are compared above; the profile
columns agree with their frozen-linearized FD summaries to the printed FD
precision.

| Objective | `n0` | `T0` | `density_shape_power` | `temperature_shape_power` | `density_shape_alpha` | `temperature_shape_alpha` | `RBC:1:0` | `ZBS:1:0` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `softmax_Er` | -2.760400441514e+00 | 2.705100711689e+00 | -7.784405183152e-02 | 1.490855037375e+00 | 1.451734036199e-01 | 1.116894873924e+01 | -2.664034988650e+01 | 5.782007566251e+00 |
| `net_total_power_volume_average_mw_m3` | 2.415416519867e-01 | 8.104652944301e-02 | 1.003726557152e-03 | 2.747697620410e-01 | -2.528779187142e-04 | -3.943101752416e-01 | -1.716403818524e-03 | -6.398310018111e-03 |
| `Er_transition_left` | -9.896305791113e-01 | 1.586751820313e+00 | -1.208521391007e-02 | -7.173957341868e+00 | 1.496282012102e-02 | 1.556026393455e+01 | -1.338265092869e+01 | -8.850610698362e-01 |
| `Er_transition_right` | -1.103871861332e+00 | 1.688983130444e+00 | -1.686008784223e-02 | -6.297007194606e+00 | 2.214694965363e-02 | 1.558583851783e+01 | -1.515906028260e+01 | -5.238769121529e-01 |
| `Er2_volume_average` | 4.020105783137e+00 | 3.110163409143e+01 | 2.611484575702e+00 | -1.258956845674e+01 | -1.927877202344e+00 | 1.051952483229e+02 | -2.831596471801e+02 | -3.578981342375e+02 |
| `Er_volume_average` | -1.830227971207e+00 | 8.218177549341e-01 | -7.154002818231e-02 | -8.502304591933e-01 | -1.092356862620e-01 | 3.151494487294e+00 | -6.423623674892e+00 | 1.387366143587e+01 |
| `bootstrap_current_softmax_abs_scaled` | 1.085312709009e-01 | 1.746717834088e-01 | -1.154277852071e-02 | 6.761305940931e-01 | 3.780884022570e-02 | -5.335135845721e-01 | -1.986205686675e+00 | -1.607858441965e+00 |

### Remaining geometry-FD discrepancy: identified linearization mismatch

This is not evidence of a remaining omitted database-table coordinate term.
The FD lane forms `state_star` with the configured **forward** VMEC solve and
uses `state_star +/- h * state_tangent`.  The root-only AD table instead
starts from that forward-built runtime for its objective and selected-root
values, but its final geometry payload transpose calls
`geometry_raw_block_solve_from_param_vector`, which performs a second
`implicit.solve_implicit_with_aux(...)` and rebuilds the geometry/scan payload
from that second VMEC state.  Consequently the objective/root cotangent and
the VMEC payload Jacobian can be evaluated at different converged numerical
states.

The reported baseline values show this directly: the FD and AD root-only
`Er_transition_right` values differ by about `1.78e-4`, `Er2` by about
`6.88e-4`, and `Er_volume_average` by about `1.03e-4`, despite having the
same configured physical parameter point.  The next correction is therefore
to retain and reuse the original forward VMEC state/mask in the raw-block
payload transpose.  It must not change the database-table boundary or the
Lij path.

## Remaining validation

1. Run the same database configuration at 16 accepted steps / four segments.
2. Compare database transport-objective geometry derivatives with a matching
   frozen-schedule FD run.  The table above validates only the transport-model
   independent geometry rows.

## 2026-09-10 post compact-face-coordinate 16-step status

The complete black-box database reverse run now finishes after the compact
native face-coordinate transpose correction.  All direct-flux, compact-face,
equation-assembly, recorded-scan-fold, and final VMEC parameter bars were
finite.  The completed run used 16 accepted steps, four reverse segments,
`block` stage adjoints, `explicit_database` RHS transpose, and both segment
diagnostic switches.

### Saved 16-step database FD comparison: `RBC:1:0`

The saved frozen-linearized full-transport FD output is available for RBC.
It was produced before the most recent forward/reverse corrections, so its
baseline objective differs from the current run (`softmax_Er`: FD
`2.136291598654e+01`; current AD `2.134759724591e+01`).  It is therefore a
useful discrepancy diagnostic, but not yet the final same-revision reference.

| Objective | Current reverse AD | Saved FD | Relative difference |
| --- | ---: | ---: | ---: |
| `softmax_Er` | -2.768402478133e+01 | -2.823097e+01 | 1.937e-02 |
| `net_total_power_volume_average_mw_m3` | 1.264105349756e-04 | 2.917166e-04 | 5.667e-01 |
| `Er_transition_left` | -1.344576040349e+01 | -1.345082e+01 | 3.762e-04 |
| `Er_transition_right` | -1.527442997592e+01 | -1.529050e+01 | 1.051e-03 |
| `Er2_volume_average` | -2.960542023652e+02 | -3.004361e+02 | 1.459e-02 |
| `Er_volume_average` | -6.834827458946e+00 | -6.979576e+00 | 2.074e-02 |
| `electron_temperature_volume_average_keV` | -1.500766890758e-02 | -1.661841e-02 | 9.693e-02 |
| `total_pressure_volume_average` | -7.022608365525e-02 | -7.324379e-02 | 4.120e-02 |
| `alpha_power_volume_average_mw_m3` | -4.466139933500e-05 | 1.144429e-04 | 1.390e+00 (sign differs) |
| `bootstrap_current_softmax_abs_scaled` | -1.500156216981e+00 | -2.129220e+00 | 2.954e-01 |

The prior full 16-step ZBS FD command was issued, but the completed ZBS
full-transport output is not in the saved pasted records.  Do not substitute
the root-only ZBS FD result here: it differentiates a different map.

### Memory/timing observation

The completed diagnostic run remained resident at roughly 58% of the 30 GB
host-RAM allocation (about 17 GB) through most of the reverse sweep, only
dropping after completion.  This is still unacceptably high for the intended
compact database boundary.  The run also reported approximately 1098 s in
the segmented cotangent sweep and 1055 s in the final recorded-scan fold.
Future optimization must reduce retained host payload/tape memory without
moving the recorded NTX scan back inside transport segments.

## 2026-09-10 refreshed same-revision 16-step geometry FD comparison

Fresh frozen-linearized accepted-replay FD was run for both `RBC:1:0` and
`ZBS:1:0` after the compact native face-coordinate transpose correction.  Its
baseline objectives agree with the current AD baseline (for example,
`softmax_Er` differs by about `2e-7` relatively), so this supersedes the stale
RBC comparison above as the active full-transport validation.

Entries below are `abs(AD - FD) / max(abs(AD), abs(FD))`.

| Objective | RBC relative error | ZBS relative error | Status |
| --- | ---: | ---: | --- |
| `softmax_Er` | 1.644e-05 | 1.471e-04 | matches |
| `net_total_power_volume_average_mw_m3` | 5.443e-01 | 1.046e-02 | RBC missing/wrong contribution |
| `Er_transition_left` | 2.303e-05 | 1.369e-04 | matches |
| `Er_transition_right` | 4.125e-05 | 5.292e-04 | matches |
| `Er2_volume_average` | 9.729e-05 | 9.701e-06 | matches |
| `Er_volume_average` | 9.854e-05 | 9.107e-05 | matches |
| `electron_temperature_volume_average_keV` | 6.877e-02 | 2.599e-02 | incomplete explicit geometry contribution |
| `total_pressure_volume_average` | 4.201e-02 | 1.239e-02 | incomplete explicit geometry contribution |
| `alpha_power_volume_average_mw_m3` | 6.186e-01 | 1.026e-02 | RBC missing/wrong contribution |
| `bootstrap_current_softmax_abs_scaled` | 2.995e-01 | 4.905e-01 | missing/wrong terminal bootstrap geometry contribution |

Conclusion: the scan-coordinate terms now repair the geometry sensitivity of
the ambipolar/Er channels, but the full transport reverse still lacks or
misroutes explicit geometry terms in power, thermodynamic averages, and the
terminal bootstrap objective.  The next reverse-AD audit should isolate those
three boundary classes; it should not change the now-matching Er-coordinate
path.

### Geometry-dependent-source caveat

The present benchmark source-model interface is state-only:
`source_model(state)`.  Therefore source terms have no direct geometry
derivative beyond their dependence on the evolved state and the explicit
geometry factors in the equation/volume-average algebra.  Both the database
and exact-Lij paths preserve the same source-model object when rebuilding
equations at a perturbed geometry.

If a future source model closes over geometry (or is redesigned to accept it
explicitly), this will be insufficient: the geometry reverse must then rebuild
or differentiate that source model at the perturbed geometry.  A regression
test with an intentionally geometry-dependent source is required before such
a model is enabled.
