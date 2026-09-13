# Database reverse AD versus FD status

## 2026-09-13 completed batched-modes timing result

The completed `shared_multi_rhs + batched_split` 16/4 run is recorded in
[the batched-modes run note](docs/benchmarks/database_reverse_16x4_2026-09-13_batched_modes_run.md).
It preserved every transport derivative at printed precision and reduced peak
host RSS from 13.815 GiB to 12.792 GiB. It did **not** improve the steady warm
segment: the mean changed from 22.264 s to 23.779 s. Thus the objective-row
batching hypothesis is rejected as the explanation for the approximately
22-second warm segment. The initial direct-RHS support phase also increased
from 568.883 s to 617.280 s. No AD-versus-FD conclusion changed.

The next isolated candidate removes the database support transpose from the
sequential four-step state recurrence, retains the exact solved stage
cotangents, and batches the independent per-step support contractions after
the recurrence. It is not part of the completed measurement above and must not
be described as faster until a full GPU timing is available.

## Implementation resumed after the completed-run record

The completed timings and derivative values below remain the measured
reference. Further performance changes are now independently selectable;
defaults preserve that run. See
[database reverse performance modes](docs/benchmarks/database_reverse_performance_modes.md)
for current/earlier selectors, the opt-in structural-zero initial-support
candidate, validation plan, and no-diagnostics 16/4 commands. No new full GPU
timing or derivative comparison has replaced the recorded reference.

## 2026-09-12 completed no-diagnostics performance comparison (latest)

This completed record supersedes the earlier partial-run timing conclusions
and the pending-performance statements below. Implementation is paused at the
user's request while these results are recorded. No new CLI restoration or
performance implementation has been applied as part of this record.

Sources (preserved verbatim, including all printed values and resource counters):

- [Pre-change no-diagnostics baseline](docs/benchmarks/database_reverse_16x4_2026-09-12_baseline.md):
  user attachment `79cfff38-9c8b-4b18-a1c4-26166b9c44e8`.
- [Completed performance-change run](docs/benchmarks/database_reverse_16x4_2026-09-12_performance_run.md):
  user attachment `ad2267de-488a-4836-bc6d-8aa7a928b6a1`.

Both use the same database wHe TOML, 16 accepted steps / 4-step segments,
`block`, `explicit_database`, `grouped_vjp`,
`joint_local_vjp_upar_only`, `jax_selected_root`, all objectives, six profile
DOFs and two geometry DOFs. Diagnostics and persistent compilation cache are off.
The executed commands in the `/usr/bin/time` footers match except that the new
one explicitly states `--optimization-api-profile-dofs include`, already the
default. The mangled command at the top is not the executed command.
Remote commit hashes and identical machine conditions are not printed in these
logs; do not infer either from `git pull` saying up to date.

### Completed timing and host-memory result

| Metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| Whole-process wall time | 1:19:27 (4767 s) | 1:18:20 (4700 s) | -67 s (-1.405%) |
| Benchmark-reported internal elapsed time | 3805.329 s | 3776.534 s | -28.795 s (-0.757%) |
| Peak process host RSS | 15051068 KiB (14.353817 GiB) | 14486496 KiB (13.815399 GiB) | -564572 KiB / -551.340 MiB (-3.751%) |
| User CPU time | 5523.60 s | 5387.23 s | -136.37 s |
| System CPU time | 274.81 s | 279.79 s | +4.98 s |
| CPU percentage | 121% | 120% | -1 percentage point |
| Mean of the three warm four-step segments | 22.813 s | 22.264 s | -0.549 s (-2.407%) |
| Major page faults | 0 | 1794 | +1794 |
| Swaps | 0 | 0 | unchanged |
| Exit status | 0 | 0 | both successful |

The completed run is **slightly faster and uses slightly less peak host memory**,
not slower overall. This is only a 1.4% wall-time and 3.75% RSS improvement in
one comparison, not a demonstrated large or repeatable performance gain.
The initial direct-RHS support phase still regresses from 296.198 to 568.883 s
(+272.685 s, +92.06%). The user target of approximately 6 s warm reverse
against approximately 2 s forward is **not achieved**.

The earlier +112.525 s result was the subtotal of 13 matching printed phases
through initial-profile support, not whole-process wall time. The final fold
and later geometry phases were then absent. Do not treat printed phase sums
as exhaustive/disjoint wall-time accounting, or add nested compiler alarms,
individual segment times and the sweep total together.

### Every reported progress-phase timing

All times are seconds. Individual segment rows are included in the sweep total.
The first segment includes first-call overhead and is not a warm execution time.
The `realized-schedule vjp forward` marker measures schedule-artifact reuse
in this mode, not an independently timed complete forward evolution.

| Reported phase | Before | After | After - before |
| --- | ---: | ---: | ---: |
| realtime geometry runtime build ready | 646.782 | 628.126 | -18.656 |
| realtime geometry solver components ready | 0.566 | 0.654 | +0.088 |
| support reverse profile-state vjp ready | 116.223 | 108.166 | -8.057 |
| support reverse initial carry vjp ready | 0.918 | 0.931 | +0.013 |
| support reverse realized-schedule vjp forward ready | 2.102 | 1.919 | -0.183 |
| support reverse final-objective cotangents ready | 320.216 | 286.341 | -33.875 |
| support reverse segment 4/4 ready | 1067.933 | 1009.543 | -58.390 |
| support reverse segment 3/4 ready | 22.907 | 22.364 | -0.543 |
| support reverse segment 2/4 ready | 23.099 | 22.391 | -0.708 |
| support reverse segment 1/4 ready | 22.433 | 22.037 | -0.396 |
| support reverse segmented cotangent sweep ready | 1136.374 | 1076.337 | -60.037 |
| support reverse reduced carry bars expanded ready | 0.704 | 0.693 | -0.011 |
| support reverse initial direct-RHS support pullback ready | 296.198 | 568.883 | +272.685 |
| support reverse initial state pullback ready | 182.195 | 156.957 | -25.238 |
| initial-Er root boundary compact pullback ready | 280.169 | 265.768 | -14.401 |
| support reverse profile parameter pullback ready | 0.960 | 1.063 | +0.103 |
| support reverse initial-profile scan payload pullback ready | 1.468 | 1.562 | +0.094 |
| database final recorded-scan fold ready | 770.232 | 707.702 | -62.530 |
| objective_table vmec implicit state/raw-block aux ready | 0.012 | 0.009 | -0.003 |
| objective_table booz input tables ready | 2.348 | 1.767 | -0.581 |
| objective_table booz_xform vjp ready | 8.907 | 7.520 | -1.387 |
| objective_table vmec objective cotangents ready | 18.757 | 16.327 | -2.430 |
| objective_table DMerc softmax cotangent ready | 12.440 | 11.926 | -0.514 |
| objective_table boozer light cotangents ready | 3.271 | 2.502 | -0.769 |
| objective_table aspect proxy cotangents ready | 0.103 | 0.089 | -0.014 |
| objective_table j-qi/maxj Boozer cotangents ready | 34.504 | 31.715 | -2.789 |
| objective_table booz cotangents pulled to state | 9.721 | 6.601 | -3.120 |
| objective_table final vmec parameter pullback ready | 51.323 | 41.566 | -9.757 |

Compiler events, reported separately (not added to the phase or wall totals):

| Reported compilation | Before | After |
| --- | ---: | ---: |
| Early `jit_scan` alarm | 151.718771361 s | 137.964156020 s |
| Post-sweep `jit__pullback` alarm | 187.536269204 s | 370.032824865 s |
| Appended `jit__direct_cotangents` alarm | not present | 144.512548761 s |

The last compiler label is associated with an initial-root experiment kernel
in the inspected source, whereas this full-transport continuation should fold
the recorded scan. Its ownership remains unverified in the pasted output; retain
the raw line but do not assign it to a transport subphase or add it to total time.

### Derivatives before versus after

All 17 residuals match exactly at printed precision. All 136 Jacobian entries
are finite. Of these, 122 printed entries are identical; the other 14 differ
only at small floating-point levels:

- All 102 entries in the six profile columns are unchanged, including all
  60 transport-profile derivatives and the 42 zero geometry-profile entries.
- Maximum relative change in the 80 transport-Jacobian entries:
  `2.806420e-14` (Er transition right versus ZBS).
- Maximum relative change over all 136 entries:
  `9.929343e-11` (Boozer QI versus ZBS).
- Maximum absolute change: `9.906944e-8` (Boozer max-J versus RBC,
  whose derivative magnitude is approximately 3843).

Here relative change is `abs(after-before)/max(abs(after),abs(before))`;
equal zero pairs are assigned zero. These changes do not indicate a material
derivative regression. They also do not improve the previously recorded AD-FD
mismatches.

#### All residuals and new geometry derivatives (full printed precision)

| Objective | Residual | d/dRBC:1:0 | d/dZBS:1:0 |
| --- | ---: | ---: | ---: |
| `transport:softmax_Er` | 2.1347597245906741e+01 | -2.7683707281136634e+01 | 6.3556954390710576e+00 |
| `transport:net_total_power_volume_average_mw_m3` | 5.0807530042924676e-01 | 5.7487580005064010e-05 | -6.4216000806514321e-03 |
| `transport:Er_transition_left` | 1.7902257301111909e+01 | -1.3445891807357173e+01 | -8.6697187673723708e-01 |
| `transport:Er_transition_right` | 1.8666769949428140e+01 | -1.5274580237777837e+01 | -5.0636952455206419e-01 |
| `transport:Er2_volume_average` | 2.6970444513292227e+02 | -2.9602462298898990e+02 | -3.3046442994759883e+02 |
| `transport:Er_volume_average` | -2.8511644406025951e+00 | -6.8360480192986506e+00 | 1.4302240534246733e+01 |
| `transport:electron_temperature_volume_average_keV` | 6.5667321685714866e+00 | -1.6115942109770211e-02 | -3.9990631490992204e-02 |
| `transport:total_pressure_volume_average` | 3.4217392413639516e+01 | -7.3305802256671312e-02 | -2.3761691183351247e-01 |
| `transport:alpha_power_volume_average_mw_m3` | 5.8919528044396130e-01 | -1.1720032048136180e-04 | -6.8872114596785893e-03 |
| `transport:bootstrap_current_softmax_abs_scaled` | 1.3717342708882792e+00 | -2.1363564988892496e+00 | -1.2135493890455435e+00 |
| `geometry:boozer_qi_objective` | 2.1192029964274445e-01 | 5.9392891187040391e+00 | -1.2365550093568345e-01 |
| `geometry:boozer_maxj_objective` | 4.4387332574094023e+02 | -3.8431351876830449e+03 | -1.9205082804617996e+03 |
| `geometry:vmec_aspect_ratio` | 1.0015330918957178e+01 | -5.4006784187006147e+00 | -5.5226885751318529e+00 |
| `geometry:vmec_iota_mean` | -5.9365259966101458e-01 | 2.4405140609263865e-01 | 1.4567526019820762e-01 |
| `geometry:vmec_magnetic_well` | -2.7476128749679612e-02 | -1.1090116065531674e-02 | -4.1682027482238482e-02 |
| `geometry:vmec_mirror_ratio` | 2.1153803467163693e-01 | -5.9359094714046601e-01 | 4.1437006125401626e-01 |
| `geometry:vmec_dmerc_stability_softmax` | 3.2329612235882421e+00 | -7.2859943466255244e+00 | -1.5178996489713270e+00 |

#### All six profile columns (full printed precision; identical to baseline)

| Objective | d/dn0 | d/dT0 | d/density_shape_power | d/dtemperature_shape_power | d/ddensity_shape_alpha | d/dtemperature_shape_alpha |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `transport:softmax_Er` | -2.8702292832128968e+00 | 2.7740118826951718e+00 | -8.0970009092364181e-02 | 1.8718548957145615e+00 | 1.5312954921315222e-01 | 1.0768407297399117e+01 |
| `transport:net_total_power_volume_average_mw_m3` | 2.4144994375404702e-01 | 8.1012117712766646e-02 | 1.0015251511699052e-03 | 2.7450389881309428e-01 | -2.4504760628385079e-04 | -3.9390935353161921e-01 |
| `transport:Er_transition_left` | -9.8540288633937534e-01 | 1.5889831820635036e+00 | -1.2265911027762052e-02 | -7.1687550637460378e+00 | 1.5260136870844827e-02 | 1.5568052949830751e+01 |
| `transport:Er_transition_right` | -1.0998955496995126e+00 | 1.6914404449598357e+00 | -1.7086805199196833e-02 | -6.2814820735972869e+00 | 2.2559732570020176e-02 | 1.5577054914522920e+01 |
| `transport:Er2_volume_average` | 2.5842589258007074e+00 | 3.1320424054222428e+01 | 2.2824336263282099e+00 | -1.1938297555500746e+01 | 3.0861923809212257e+00 | 1.0694492154833915e+02 |
| `transport:Er_volume_average` | -1.9024031619531510e+00 | 8.6605462867946792e-01 | -6.8688993949557423e-02 | -6.9405878986901259e-01 | -1.6832079944450518e-01 | 3.0165805596904787e+00 |
| `transport:electron_temperature_volume_average_keV` | 8.6352302266201608e-04 | 3.5577917899824130e-01 | -7.1344857712301452e-05 | 1.5248015861809205e+00 | 1.2218646390351195e-03 | -3.0449272577867226e+00 |
| `transport:total_pressure_volume_average` | 8.0624286614427199e+00 | 1.8654173307789694e+00 | 2.4426699648199540e-01 | 7.7523359564773786e+00 | -1.3265053153685458e+00 | -1.4516604356364713e+01 |
| `transport:alpha_power_volume_average_mw_m3` | 2.7962100211621660e-01 | 8.3216104055209142e-02 | 2.3609178836717423e-03 | 2.8420414560668772e-01 | -7.6383506297708842e-03 | -4.1289868849445077e-01 |
| `transport:bootstrap_current_softmax_abs_scaled` | 1.0697658604014659e-01 | 1.7479404655582315e-01 | -1.1999309293285526e-02 | 7.0788983350295842e-01 | 4.1575985810500586e-02 | -6.1967238271409819e-01 |
| `geometry:boozer_qi_objective` | 0.0000000000000000e+00 | 0.0000000000000000e+00 | 0.0000000000000000e+00 | 0.0000000000000000e+00 | 0.0000000000000000e+00 | 0.0000000000000000e+00 |
| `geometry:boozer_maxj_objective` | 0.0000000000000000e+00 | 0.0000000000000000e+00 | 0.0000000000000000e+00 | 0.0000000000000000e+00 | 0.0000000000000000e+00 | 0.0000000000000000e+00 |
| `geometry:vmec_aspect_ratio` | 0.0000000000000000e+00 | 0.0000000000000000e+00 | 0.0000000000000000e+00 | 0.0000000000000000e+00 | 0.0000000000000000e+00 | 0.0000000000000000e+00 |
| `geometry:vmec_iota_mean` | 0.0000000000000000e+00 | 0.0000000000000000e+00 | 0.0000000000000000e+00 | 0.0000000000000000e+00 | 0.0000000000000000e+00 | 0.0000000000000000e+00 |
| `geometry:vmec_magnetic_well` | 0.0000000000000000e+00 | 0.0000000000000000e+00 | 0.0000000000000000e+00 | 0.0000000000000000e+00 | 0.0000000000000000e+00 | 0.0000000000000000e+00 |
| `geometry:vmec_mirror_ratio` | 0.0000000000000000e+00 | 0.0000000000000000e+00 | 0.0000000000000000e+00 | 0.0000000000000000e+00 | 0.0000000000000000e+00 | 0.0000000000000000e+00 |
| `geometry:vmec_dmerc_stability_softmax` | 0.0000000000000000e+00 | 0.0000000000000000e+00 | 0.0000000000000000e+00 | 0.0000000000000000e+00 | 0.0000000000000000e+00 | 0.0000000000000000e+00 |

#### Geometry-column comparison against the pre-change AD run

| Objective | RBC before | RBC after | RBC relative change | ZBS before | ZBS after | ZBS relative change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `transport:softmax_Er` | -2.7683707281136662e+01 | -2.7683707281136634e+01 | 1.027e-15 | 6.3556954390710434e+00 | 6.3556954390710576e+00 | 2.236e-15 |
| `transport:net_total_power_volume_average_mw_m3` | 5.7487580005064010e-05 | 5.7487580005064010e-05 | 0.000e+0 | -6.4216000806514321e-03 | -6.4216000806514321e-03 | 0.000e+0 |
| `transport:Er_transition_left` | -1.3445891807357155e+01 | -1.3445891807357173e+01 | 1.321e-15 | -8.6697187673722109e-01 | -8.6697187673723708e-01 | 1.844e-14 |
| `transport:Er_transition_right` | -1.5274580237777851e+01 | -1.5274580237777837e+01 | 9.304e-16 | -5.0636952455207840e-01 | -5.0636952455206419e-01 | 2.806e-14 |
| `transport:Er2_volume_average` | -2.9602462298898990e+02 | -2.9602462298898990e+02 | 0.000e+0 | -3.3046442994759906e+02 | -3.3046442994759883e+02 | 6.880e-16 |
| `transport:Er_volume_average` | -6.8360480192986559e+00 | -6.8360480192986506e+00 | 7.796e-16 | 1.4302240534246726e+01 | 1.4302240534246733e+01 | 4.968e-16 |
| `transport:electron_temperature_volume_average_keV` | -1.6115942109770211e-02 | -1.6115942109770211e-02 | 0.000e+0 | -3.9990631490992204e-02 | -3.9990631490992204e-02 | 0.000e+0 |
| `transport:total_pressure_volume_average` | -7.3305802256671312e-02 | -7.3305802256671312e-02 | 0.000e+0 | -2.3761691183351247e-01 | -2.3761691183351247e-01 | 0.000e+0 |
| `transport:alpha_power_volume_average_mw_m3` | -1.1720032048136180e-04 | -1.1720032048136180e-04 | 0.000e+0 | -6.8872114596785893e-03 | -6.8872114596785893e-03 | 0.000e+0 |
| `transport:bootstrap_current_softmax_abs_scaled` | -2.1363564988892496e+00 | -2.1363564988892496e+00 | 0.000e+0 | -1.2135493890455427e+00 | -1.2135493890455435e+00 | 7.319e-16 |
| `geometry:boozer_qi_objective` | 5.9392891187203531e+00 | 5.9392891187040391e+00 | 2.747e-12 | -1.2365550092340527e-01 | -1.2365550093568345e-01 | 9.929e-11 |
| `geometry:boozer_maxj_objective` | -3.8431351877821144e+03 | -3.8431351876830449e+03 | 2.578e-11 | -1.9205082804683188e+03 | -1.9205082804617996e+03 | 3.395e-12 |
| `geometry:vmec_aspect_ratio` | -5.4006784187006147e+00 | -5.4006784187006147e+00 | 0.000e+0 | -5.5226885751318529e+00 | -5.5226885751318529e+00 | 0.000e+0 |
| `geometry:vmec_iota_mean` | 2.4405140609263865e-01 | 2.4405140609263865e-01 | 0.000e+0 | 1.4567526019820762e-01 | 1.4567526019820762e-01 | 0.000e+0 |
| `geometry:vmec_magnetic_well` | -1.1090116065531674e-02 | -1.1090116065531674e-02 | 0.000e+0 | -4.1682027482238482e-02 | -4.1682027482238482e-02 | 0.000e+0 |
| `geometry:vmec_mirror_ratio` | -5.9359094714046601e-01 | -5.9359094714046601e-01 | 0.000e+0 | 4.1437006125401626e-01 | 4.1437006125401626e-01 | 0.000e+0 |
| `geometry:vmec_dmerc_stability_softmax` | -7.2859943466255244e+00 | -7.2859943466255244e+00 | 0.000e+0 | -1.5178996489713270e+00 | -1.5178996489713270e+00 | 0.000e+0 |

#### Comparison with the existing 16-step geometry FD values

No new FD run was performed. The FD references are the existing
same-revision frozen-linearized accepted-replay values in the composite
face-geometry correction section below, printed to six significant digits.
The formula is `abs(AD-FD)/max(abs(AD),abs(FD))`.

| Objective | New RBC AD | Saved RBC FD | RBC relative error | New ZBS AD | Saved ZBS FD | ZBS relative error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `transport:softmax_Er` | -2.7683707281136634e+01 | -2.768448e+01 | 2.791e-5 | 6.3556954390710576e+00 | 6.356321e+00 | 9.842e-5 |
| `transport:net_total_power_volume_average_mw_m3` | 5.7487580005064010e-05 | 5.760105e-05 | 1.970e-3 | -6.4216000806514321e-03 | -6.421616e-03 | 2.479e-6 |
| `transport:Er_transition_left` | -1.3445891807357173e+01 | -1.344607e+01 | 1.325e-5 | -8.6697187673723708e-01 | -8.667252e-01 | 2.845e-4 |
| `transport:Er_transition_right` | -1.5274580237777837e+01 | -1.527506e+01 | 3.141e-5 | -5.0636952455206419e-01 | -5.059553e-01 | 8.180e-4 |
| `transport:Er2_volume_average` | -2.9602462298898990e+02 | -2.960254e+02 | 2.625e-6 | -3.3046442994759883e+02 | -3.304945e+02 | 9.099e-5 |
| `transport:Er_volume_average` | -6.8360480192986506e+00 | -6.835501e+00 | 8.002e-5 | 1.4302240534246733e+01 | 1.430209e+01 | 1.053e-5 |
| `transport:electron_temperature_volume_average_keV` | -1.6115942109770211e-02 | -1.611605e-02 | 6.695e-6 | -3.9990631490992204e-02 | -3.999069e-02 | 1.463e-6 |
| `transport:total_pressure_volume_average` | -7.3305802256671312e-02 | -7.330528e-02 | 7.124e-6 | -2.3761691183351247e-01 | -2.376173e-01 | 1.634e-6 |
| `transport:alpha_power_volume_average_mw_m3` | -1.1720032048136180e-04 | -1.170870e-04 | 9.669e-4 | -6.8872114596785893e-03 | -6.887228e-03 | 2.402e-6 |
| `transport:bootstrap_current_softmax_abs_scaled` | -2.1363564988892496e+00 | -2.141504e+00 | 2.404e-3 | -1.2135493890455435e+00 | -1.217705e+00 | 3.413e-3 |

The saved profile-FD comparisons also remain unchanged because all profile
derivatives match the prior AD values exactly. The largest saved transport-profile
relative error remains approximately `1.986e-5` (Er transition left versus
density_shape_alpha). The bootstrap geometry errors remain approximately
`2.404e-3` (RBC) and `3.413e-3` (ZBS), while the small RBC net-power derivative
retains approximately `1.970e-3` relative error. There is no new FD validation.

### Executed command (from the time footer, not the mangled pasted header)

```bash
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py --config ./examples/benchmarks/Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box.toml --reverse-parameter-mode profiles_plus_realtime_geometry --reverse-geometry-parameter RBC:1:0,ZBS:1:0 --realtime-geometry-gradient-path reverse_payload --optimization-api-profile-dofs include --objective all --accepted-step-limit 16 --radau-jacobian-reuse-mode legacy --timing-mode jit-warm --reverse-segment-length 4 --reverse-stage-adjoint-solve-mode block --reverse-rhs-transpose-mode explicit_database --reverse-step-bwd-mode reduced_cotangent_call_boundary --reverse-initial-cache-support-pullback-mode scalar --reverse-rebuild-support-pullback-mode separate --reverse-final-objective-cotangent-mode grouped_vjp --reverse-bootstrap-cotangent-mode joint_local_vjp_upar_only --initial-Er-root-ad jax_selected_root --full-transport-shared-payload-smoke --reverse-schedule-artifact-mode reuse_static_probe
```

### Resume point after recording

The user requested restoring the previous performance implementation as a
CLI-selectable baseline, keeping current experiments separately selectable,
then measuring database warm reverse costs against their approximately
2 s forward / 6 s reverse target. None of the three automatically applied
performance changes currently has such a selector. No whole-commit rollback,
root optimization JIT change, derivative-contract change, or cache eviction
is authorized by this recording step. Resume that implementation only after
this recording/comparison handoff.



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

## 2026-09-12 restored finite 16-step database reverse

The 16-step / four-segment run after restoring fixed-table scale-cotangent
ownership completed with every segment, recorded-scan bar, and final VMEC bar
finite.  The profile derivatives are unchanged and retain their existing FD
agreement.  The transport geometry derivatives returned to the earlier
near-matching Er values, while the corrected terminal bootstrap coordinate
bar remains active.

The matching same-revision frozen-linearized FD values are those recorded in
the preceding section.  Relative error is
`abs(AD - FD) / max(abs(AD), abs(FD))`.

| Objective | RBC reverse AD | RBC relative error | ZBS reverse AD | ZBS relative error |
| --- | ---: | ---: | ---: | ---: |
| `softmax_Er` | -2.768402478133e+01 | 1.644e-05 | 6.355386240655e+00 | 1.471e-04 |
| `net_total_power_volume_average_mw_m3` | 1.264105349756e-04 | 5.443e-01 | -6.354428922984e-03 | 1.046e-02 |
| `Er_transition_left` | -1.344576040349e+01 | 2.303e-05 | -8.668439084257e-01 | 1.369e-04 |
| `Er_transition_right` | -1.527442997592e+01 | 4.125e-05 | -5.062231912903e-01 | 5.292e-04 |
| `Er2_volume_average` | -2.960542023652e+02 | 9.729e-05 | -3.304912938498e+02 | 9.701e-06 |
| `Er_volume_average` | -6.834827458946e+00 | 9.854e-05 | 1.430339257577e+01 | 9.107e-05 |
| `electron_temperature_volume_average_keV` | -1.500766890758e-02 | 6.877e-02 | -3.895125052489e-02 | 2.599e-02 |
| `total_pressure_volume_average` | -7.022608365525e-02 | 4.201e-02 | -2.346720852021e-01 | 1.239e-02 |
| `alpha_power_volume_average_mw_m3` | -4.466139933500e-05 | 6.186e-01 | -6.816570577335e-03 | 1.026e-02 |
| `bootstrap_current_softmax_abs_scaled` | -2.136081447591e+00 | 2.532e-03 | -1.213281529367e+00 | 3.633e-03 |

The state-adjoint and scale-ownership corrections therefore solve the
nonfinite regression without removing the bootstrap correction.  The Er rows
are again within `5.3e-4`, and bootstrap is within `3.7e-3`.  Work should now
focus only on the RBC power term and the explicit geometry terms in the
temperature and pressure volume averages; the Radau state transpose and
recorded-scan coordinate ownership should not be changed again for those
remaining discrepancies.

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

## 2026-09-12 composite face-geometry correction: 16-step result

The subsequent 16-step / four-segment database reverse run includes the
previously omitted turbulent/classical contribution to the composite face-flux
geometry pullback.  All database geometry, state, recorded-scan, and final VMEC
cotangents remained finite.  The values below are the complete transport
geometry derivatives from that run, compared with the matching same-revision
frozen-linearized accepted-replay FD run.  Relative error is
`abs(AD - FD) / max(abs(AD), abs(FD))`.

| Objective | RBC reverse AD | RBC FD | RBC relative error | ZBS reverse AD | ZBS FD | ZBS relative error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `softmax_Er` | -2.768370728114e+01 | -2.768448e+01 | 2.791e-05 | 6.355695439071e+00 | 6.356321e+00 | 9.842e-05 |
| `net_total_power_volume_average_mw_m3` | 5.748758000506e-05 | 5.760105e-05 | 1.970e-03 | -6.421600080651e-03 | -6.421616e-03 | 2.479e-06 |
| `Er_transition_left` | -1.344589180736e+01 | -1.344607e+01 | 1.325e-05 | -8.669718767372e-01 | -8.667252e-01 | 2.845e-04 |
| `Er_transition_right` | -1.527458023778e+01 | -1.527506e+01 | 3.141e-05 | -5.063695245521e-01 | -5.059553e-01 | 8.180e-04 |
| `Er2_volume_average` | -2.960246229890e+02 | -2.960254e+02 | 2.625e-06 | -3.304644299476e+02 | -3.304945e+02 | 9.099e-05 |
| `Er_volume_average` | -6.836048019299e+00 | -6.835501e+00 | 8.001e-05 | 1.430224053425e+01 | 1.430209e+01 | 1.052e-05 |
| `electron_temperature_volume_average_keV` | -1.611594210977e-02 | -1.611605e-02 | 6.695e-06 | -3.999063149099e-02 | -3.999069e-02 | 1.463e-06 |
| `total_pressure_volume_average` | -7.330580225667e-02 | -7.330528e-02 | 7.124e-06 | -2.376169118335e-01 | -2.376173e-01 | 1.634e-06 |
| `alpha_power_volume_average_mw_m3` | -1.172003204814e-04 | -1.170870e-04 | 9.668e-04 | -6.887211459679e-03 | -6.887228e-03 | 2.402e-06 |
| `bootstrap_current_softmax_abs_scaled` | -2.136356498889e+00 | -2.141504e+00 | 2.404e-03 | -1.213549389046e+00 | -1.217705e+00 | 3.413e-03 |

The face-geometry correction closes the earlier temperature and pressure gaps:
both now agree with FD at `O(1e-6)` to `O(1e-5)`.  The RBC net-power relative
error is `1.970e-3`, but its absolute AD--FD difference is only `1.135e-7`
because the derivative is a small cancellation residual.  Bootstrap remains
the largest material discrepancy, at `2.404e-3` for RBC and `3.413e-3` for
ZBS.

### Compact LaTeX table: full-transport AD versus FD for all DOFs

Objectives are columns and all six plasma-profile DOFs plus both VMEC boundary
DOFs are rows.  The power entry is the net total-power objective.  Entries are
relative errors using the same normalization as above.  Here $\alpha_n$,
$\alpha_T$, $\beta_n$, and $\beta_T$ denote `density_shape_power`,
`temperature_shape_power`, `density_shape_alpha`, and
`temperature_shape_alpha`, respectively.

```latex
\begin{table}[t]
\centering
\def\arraystretch{1.5}
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
    & $9.867\times10^{-8}$
    & $8.762\times10^{-8}$
    & $4.094\times10^{-7}$
    & $1.812\times10^{-7}$
    & $1.305\times10^{-7}$ \\
  $T_0$
    & $4.229\times10^{-8}$
    & $1.146\times10^{-7}$
    & $2.631\times10^{-7}$
    & $2.823\times10^{-8}$
    & $2.663\times10^{-7}$ \\
  $\alpha_n$
    & $1.121\times10^{-8}$
    & $1.714\times10^{-6}$
    & $8.895\times10^{-7}$
    & $1.509\times10^{-7}$
    & $5.890\times10^{-8}$ \\
  $\alpha_T$
    & $5.571\times10^{-8}$
    & $8.892\times10^{-9}$
    & $1.172\times10^{-8}$
    & $4.324\times10^{-9}$
    & $4.733\times10^{-8}$ \\
  $\beta_n$
    & $3.317\times10^{-7}$
    & $1.986\times10^{-5}$
    & $1.216\times10^{-6}$
    & $1.199\times10^{-6}$
    & $1.008\times10^{-7}$ \\
  $\beta_T$
    & $2.510\times10^{-7}$
    & $1.895\times10^{-7}$
    & $3.265\times10^{-7}$
    & $1.180\times10^{-7}$
    & $2.790\times10^{-8}$ \\
  $\mathrm{RBC}(1,0)$
    & $2.791\times10^{-5}$
    & $1.325\times10^{-5}$
    & $3.141\times10^{-5}$
    & $1.970\times10^{-3}$
    & $2.404\times10^{-3}$ \\
  $\mathrm{ZBS}(1,0)$
    & $9.842\times10^{-5}$
    & $2.845\times10^{-4}$
    & $8.180\times10^{-4}$
    & $2.479\times10^{-6}$
    & $3.413\times10^{-3}$ \\
  \\[-1.5ex]\hline
\end{tabular}
\caption{Relative errors between 16-step database full-transport reverse AD
and frozen-linearized finite differences for the plasma-profile and VMEC
boundary DOFs, after restoring the complete composite face-flux geometry
pullback.}
\label{tab:database-full-transport-geometry-ad-fd-relative-error}
\end{table}
```

### Non-colored speed/memory audit — 2026-09-12

The clean no-diagnostics baseline used the established exact `block`
stage-adjoint solve. It reported 320.216 s for terminal-objective cotangents,
1136.374 s for the four reverse segments (1067.933 s for the first
compile-plus-execute, then 22.907, 23.099 and 22.433 s), 296.198 s for the
initial direct-RHS support pullback, and 770.232 s for the final batched
recorded-database scan fold. Peak host RSS was 15051068 KiB (14.354 GiB).
The first segment timer is not a warm execution timer; its excess cannot be
assigned entirely to compilation without measuring the compilation separately.

The user reported four selected tests passing in 13.50 s for the new solve
layout, joint ordinary-objective VJP, and initial-support reuse. These are
algebra/dispatch checks, not evidence of reduced memory or execution time.

Follow-up audit caught a selector integration error: the new
`block_database_multi_rhs` solve used the established generic forward-mode
Jacobian, but the subsequent input pullback exempted only `block` from the
compact database state VJP. The exemption now covers both exact layouts.
Tests cover both selectors' matrix and carry contracts, plus a JIT-compiled,
three-stage coupled nonlinear solve and input pullback against an independent
residual VJP. The default `block`, root optimization lane, and caches are
unchanged.

Local verification used the installed CPU JAX 0.5.0 and the exact function/test
definitions in an isolated harness: seven checks passed, plus one check
reproducing the pre-fix selector failure. Full repository pytest collection
was blocked by missing physics dependencies (`h5py` first); this is not a full
repository or GPU benchmark pass.

Subsequent user verification ran the actual repository regression selection
on CPU and passed: **7 passed, 122 deselected in 10.60 s**. This supersedes
the isolated-harness limitation for those seven targeted checks, but is not
a complete test-suite or GPU benchmark result:

```bash
JAX_PLATFORMS=cpu PYTHONPATH="$HOME/VMEX:$HOME/NTX/src" \
python -m pytest tests/test_solvers.py -q -p no:cacheprovider \
-k 'database_block_multi_rhs or database_exact_block_solve_and_carry_pullback or database_plain_block' \
--maxfail=1
```

**Correction to the proposed factorization speedup:** installed JAX 0.5.0
already shares the LU across objective rows in the original mapped `block`
solve. Small compiled CPU examples of mapped-vector and explicit-column RHS
layouts each have one unbatched LU and two matrix-RHS triangular solves, with
identical outputs, including when `jacfwd` builds the matrix inside the map.
The optional new selector therefore does not establish a factorization-count
saving. Keep `block` as the validated baseline; do not request a long run on
the premise of avoiding ten factorizations.

Remaining non-colored candidates and evidence limits:

- `grouped_joint_vjp` shares the ordinary terminal state/geometry objective
  trace. Bootstrap is unchanged; timing and peak memory still require
  measurement. The established `grouped_vjp` remains available.
- Matrix construction and outgoing state pullback both express stage
  `jacfwd` work in source, but run inside the same step JIT in the default
  memory mode. Compiler common-subexpression elimination may already share
  this work; do not infer two runtime Jacobian builds from source alone.
- The fixed-table support pullback has a separate non-inline JIT inside a
  per-stage scan. Sharing applicable primal work across that boundary is a
  concrete next investigation. Within the support split, table and local
  flux-geometry paths also prepare identical flux payloads/cotangents; check
  compiled reuse before claiming an execution saving.
- A four-step forward time does not isolate the reverse's exact stage
  Jacobian construction, objective cotangents, and geometry/table support
  transposes. Measure those components while keeping the same equations,
  finite Jacobian contract, full nonlocal coupling and cached executables.

Colored modes remain isolated experiments and are not this optimization plan.

### Shared database support preparation implemented - 2026-09-12

`ComposedEquationSystem` now prepares the common primal inputs and flux
cotangents once inside its built-in split-support call. The three existing
compact contractions consume the same call-local record via a private
keyword; their standalone behavior is retained. Public-hook overrides and
incomplete fixtures bypass sharing. Density and temperature native-face
payloads remain distinct. No change to the root-only lane, Radau state/matrix
contract, caches, table-coordinate ownership or final recorded scan fold.

Four new cases in `tests/test_database_support_reuse.py`, plus five existing
boundary regressions, passed through an exact-source CPU harness:
**9 passed in 5.26 s**. Tests exercise actual fixed-flux capture/assembly and
equation-to-flux VJPs with small polynomial local flux laws (no NTX or VMEC
solve), JIT/vmap, dynamic inputs and public overrides. Full project import
and production GPU tests remain unverified locally.

The small JAX 0.5.0 CPU comparison measured:

| Quantity | Independent partials | Shared preparation |
| --- | ---: | ---: |
| Primal preparation/capture calls during tracing | 3 | 1 |
| Equation-to-flux VJP constructions during tracing | 2 | 1 |
| JAXPR equation count | 557 | 401 |
| Compiled temporary buffers (bytes) | 856 | 856 |

Optimized HLO sizes were essentially identical. These measurements support
less tracing/lowering work without extra temporary buffers in the fixture;
they do **not** establish reduced production warm time or host RSS. In
particular the ~23 s warm reverse segments are not yet explained or fixed by
this result. Existing AD/FD tables above are unchanged, not new benchmark
measurements of this patch.

Short full-environment gate (six selected cases):

```bash
JAX_PLATFORMS=cpu PYTHONPATH="$HOME/VMEX:$HOME/NTX/src" \
python -m pytest \
  tests/test_database_support_reuse.py \
  tests/test_ntx_geometry_implicit_pullback.py \
  -q -p no:cacheprovider \
  -k 'database_split_support_shared_primal or database_split_support_payload_uses_explicit_database_boundaries or database_fixed_payload_split_geometry_matches_generic_vjp' \
  --maxfail=1
```

User subsequently ran that gate successfully: **6 passed, 100 deselected in
14.06 s**. This is full-project confirmation of those six selected checks,
not a new AD/FD or timing measurement.

### Centre physical-mesh derivative optimization - 2026-09-12

The direct-centre fixed-table geometry partial now has a built-in physical-mesh
path using one scalar `a_b` JVP and contraction with all flux objective bars.
This is the same partial derivative as the retained per-radius VJP: normalized
coordinates are fixed, the physical radii and spacing move, and the entire
`Monoenergetic` table (including its query-coordinate scale) stays fixed.
Table/query-coordinate, native-face, equation, source and root contributions
are not removed or reassigned. The default exact `block` solve is untouched.
No caches are cleared and no new table/VMEC solve is introduced.

The new fast path is restricted to the exact built-in database model with a
complete physical mesh; custom/legacy models retain the original method body.
`tests/test_database_center_geometry.py` covers a real table and nonconstant
profiles, scalar and ten-objective bars, heat-only batches, integer geometry
leaf shapes and custom nonradial geometry dependencies. An isolated harness
executing the actual source kernels passed **4 tests in 175.42 s** on CPU
(JAX 0.5.0, interpax 0.3.7, equinox 0.11.12). Both the previous per-radius VJP
and the direct forward-flux VJP are references. This is not full repository
import validation and does not update any of the AD/FD tables above.

Performance comparison, actual source kernels on CPU, with a synthetic
four-species/51-radius/four-energy case and a 7x16x11 table, ten objective
rows. The database is passed dynamically. Host RSS was measured in separate
fresh processes; warm calls were synchronized (median of nine). No caches
were cleared.

| Measurement | Old radius VJP | New scalar mesh JVP |
| --- | ---: | ---: |
| Trace/lower | 4.088 s | 2.813 s |
| Compile | 45.288 s | 4.095 s |
| Warm execution | 35.039 ms | 3.720 ms |
| Peak process RSS | 2,158,880 KiB | 865,552 KiB |
| Compiled temporary buffers | 780,576 B | 1,566,632 B |
| Output buffers | 20,864 B | 20,864 B |

The retained all-radii JVP adds about 0.75 MiB of temporary arrays but reduces
the measured **host process peak** from 2.06 to 0.83 GiB, compilation time and
warm execution. Temporary-buffer-only optimization was not a sufficient
criterion for the user's host-RAM concern: bounded radius-map alternatives
used less scratch but kept 45-54 s compilation and slower execution, so they
were not retained. Real-kernel comparisons at both scalar and ten-objective
representative shapes were finite and agreed with the original VJP.

Do not extrapolate these component results to the entire 16/4 GPU benchmark:
its new segment timings, host peak and AD/FD output comparison remain pending.
There is no new CLI option and no change to root, Radau, cache or derivative
ownership contracts. The previous full-run derivative tables remain the
recorded reference values, not results measured after this performance edit.

Subsequent user full-project validation: **8 passed in 185.95 s (3:05)**,
running all cases in `tests/test_database_center_geometry.py` and
`tests/test_database_support_reuse.py` with `JAX_PLATFORMS=cpu`,
`-q -p no:cacheprovider --maxfail=1`. These eight checks now have confirmation
in the actual project environment, beyond the isolated source harness.
This is not a complete-suite result and does not establish new full 16/4
GPU timings, peak RSS or AD/FD values. Those remain pending; no production
code changes were made while recording this test result.
