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

## Remaining validation

1. Run the same database configuration at 16 accepted steps / four segments.
2. Compare database transport-objective geometry derivatives with a matching
   frozen-schedule FD run.  The table above validates only the transport-model
   independent geometry rows.
