# Full-time AD with restored Er-left/right rows (archived 2026-10-22)

The user supplied the completed objective/Jacobian output from the full-time
AD benchmark after restoring the fixed-cell Er rows. The printed command
uses the database black-box TOML, no accepted-step prefix limit, a
1,000-accepted-step guard, 50-step reverse segments, forward Jacobian reuse
`legacy`, and accepted replay reuse `recorded_forward`. This is the same
configured 20-second case as the previous reference.

## Saved evidence

- [Original supplied output](database_transition_objectives_ad_recorded_reuse_full_t20_left_right_2026-10-22.txt), copied byte for byte from attachment
  `7f856dd9-0aa6-4f9d-ad2c-f62671a0d671/Pasted text.txt`.
- [Full 20-by-8 residual/Jacobian CSV](database_transition_objectives_ad_recorded_reuse_full_t20_left_right_2026-10-22.csv), retaining every printed digit.
- [All 144 common old/new derivative pairs](database_transition_objectives_ad_recorded_reuse_full_t20_left_right_2026-10-22_vs_previous_ad.csv).
- [Previous full-time 18-row reference](database_transition_objectives_ad_recorded_reuse_full_t20_2026-10-19.md).

Attachment SHA-256:
`936710f010b1e1b2270f96428368f76ec1446a0d801097c770c632e3ea5bb556`.

## Observed outcome

| Quantity | Supplied result |
| --- | ---: |
| Replay Jacobian reuse mode | recorded_forward |
| Residual/Jacobian shape | 20 x 8 |
| Finite residuals | 20 / 20 |
| Finite derivatives | 160 / 160 |
| Benchmark banner | all_finite=True |
| Internal benchmark elapsed time | 2822.780 s |
| Whole-process wall time | 1:02:18 |
| User/system time | 4410.46 / 237.09 s |
| CPU utilization | 124% |
| JSON report | written |

The source excerpt begins at the final geometry pullback and ends partway
through the timing footer. It does not include this run's accepted-step
count, segment endpoint checks, peak RSS, process exit status, Git commit or
dependency versions. The earlier run's 290 accepted steps and zero replay
endpoint differences must not be assigned to this new run as observations.

## Comparison with the previous full-time reference

The 18 common residual values match exactly at the supplied printed precision.
All 108 common profile derivatives also match exactly, including the
root-position, strength, bootstrap and net-power rows. The 36 common geometry
derivatives differ only slightly:

- Maximum absolute difference: 7.9744495451450348e-8
  (`boozer_maxj_objective` / `vmec:RBC:1:0`).
- Maximum relative difference using the larger old/new magnitude:
  7.7167180643104206e-10
  (`Er_transition_rho` / `vmec:RBC:1:0`).

These tiny differences are consistent with floating-point evaluation changes.
Adding the two rows has not materially changed the existing reported values
or derivatives. This old/new AD comparison does not replace full-time FD
validation, which remains unresolved.

## Root-related observables

| Observable | Full-time value |
| --- | ---: |
| Er-left: final-state Er[20], kV/m | 3.9311269029818902e+01 |
| Er-right: final-state Er[21], kV/m | 3.9894759548722909e+01 |
| Smooth transition radius | 5.1703593282774429e-01 |
| Location moment (benchmark target 0.5) | 1.7035932827744293e-02 |
| Transition strength | 2.0149622433190042e+00 |
| Bootstrap-current softmax | 2.0426059244509895e+00 |
| Net-power average, MW/m^3 | 9.5305334313012002e-01 |

The benchmark uses unit weights and zero targets, so these are raw
observables and derivatives with respect to physical parameters, not the
weighted optimization cost gradient.

| Parameter | Er-left AD | Er-right AD | Location-moment AD | Strength AD |
| --- | ---: | ---: | ---: | ---: |
| n0 | 6.1285920311442768e+00 | 6.4179006328086148e+00 | 5.6699915363382471e-05 | 4.7338579231761335e-01 |
| T0 | -6.5074177364127178e-03 | -8.5180627791456628e-03 | -1.1307263101707243e-06 | -9.3901900090527621e-04 |
| density_shape_power | -8.9514396017198933e-02 | -9.6669550250637032e-02 | -4.0333156219529525e-06 | -8.1676945636763327e-03 |
| temperature_shape_power | -1.5316294406409560e-02 | -2.0056397619912177e-02 | -2.6650224234697437e-06 | -2.2130059927788328e-03 |
| density_shape_alpha | 2.9360584924541111e-01 | 3.2260389747777424e-01 | 2.4180930096082660e-05 | 2.3347710583286281e-02 |
| temperature_shape_alpha | 1.5139943429720935e-02 | 1.9835162250558657e-02 | 2.6389471029123737e-06 | 2.1911294597008061e-03 |
| vmec:RBC:1:0 | -2.6204250877274113e+01 | -2.4890057582657356e+01 | -1.4395585786521682e-04 | 1.6089073904610214e+00 |
| vmec:ZBS:1:0 | 1.1900617440086096e+01 | 1.3393020899226411e+01 | 1.8381717423638551e-03 | 2.5790881580335594e-02 |

Both fixed-cell Er values are positive. These samples do not straddle the
positive-to-negative transition in this run. Their larger sensitivities
therefore show that the sampled Er amplitudes respond more strongly than
the selected crossing radius. They do not prove a missing derivative in
the smooth location objective. Moving the crossing and changing the Er
amplitude are different responses; weak local position sensitivity remains
consistent with the observed difficulty moving the transition through
profile changes.

The standalone optimization examples use a location target/prior of 0.514,
while this benchmark uses 0.5. Because the prior enters the smooth selection,
the benchmark Jacobian must not be treated as the exact optimizer Jacobian
at its different settings. Cost comparisons also require residual weights,
target errors, and parameter scaling.

## Validation scope

Local work parsed the supplied text, checked unique row/column labels,
20 complete finite residuals and 160 finite derivatives, preserved numeric
strings in the CSV, checked 144 common derivative comparisons, and verified
the attachment-copy SHA-256. No VMEC, NTX, root, transport, AD benchmark,
FD benchmark or optimization solve was run locally.
