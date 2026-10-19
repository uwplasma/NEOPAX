# Full-time database reverse AD reference: 20 seconds / 290 accepted steps

The user supplied a completed GPU run on `neopax-23`, with `jax_env` active,
using `benchmark_transport_reverse_ad_only.py` and the same database benchmark
TOML as the archived 16-step checks. Its configured final time is 20 seconds.
The actual executed command in the timing footer has no accepted-step prefix
limit, uses `--max-reverse-accepted-steps 1000`, 50-step reverse segments,
forward `--radau-jacobian-reuse-mode legacy`, and
`--reverse-replay-jacobian-reuse-mode recorded_forward`.

## Saved evidence

- [Full 18-by-8 residual/Jacobian CSV](database_transition_objectives_ad_recorded_reuse_full_t20_2026-10-19.csv), retaining every printed numeric digit.
- [Attachment containing the remaining rows and successful timing footer](database_transition_objectives_ad_recorded_reuse_full_t20_2026-10-19.txt), copied byte for byte from `a69ff3f6-970f-4d0a-b36f-1cf8dcfea35f/Pasted text.txt`.
- [Prefix pasted directly in the same user message](database_transition_objectives_ad_recorded_reuse_full_t20_2026-10-19_prefix.txt), containing the AD banner, softmax row and the six profile entries in the net-power row. The attachment begins with the two remaining net-power geometry entries.
- [Earlier schedule/segment output](database_transition_objectives_ad_recorded_reuse_full_t20_2026-10-19_schedule.txt), copied byte for byte from `075c5807-8a3a-4b7a-a09b-e7b3b3c41e0e/Pasted text.txt`.
- [All six replay endpoint checks](database_transition_objectives_ad_recorded_reuse_full_t20_2026-10-19_segments.csv).

SHA-256 of the final attachment:
`803401b43a9f37902fa2724b9651fd84495cc6f8b518dce4629f532c72d76581`.
SHA-256 of the earlier schedule attachment:
`b6ebfb888dae800c16e5846e28b2c347f64c029c9b55232d24a07a6abcbe9381`.
The pasted prefix is transcribed text, rather than a separately attached file.
The remote Git commit and dependency versions are not printed in this evidence.

## Outcome

| Quantity | Observed result |
| --- | ---: |
| Exit status | 0 |
| Accepted steps | 290 |
| Reverse segments | 6: five of 50 steps, one of 40 |
| Residual/Jacobian shape | 18 x 8 |
| Finite residuals | 18 / 18 |
| Finite derivatives | 144 / 144 |
| Replay endpoint absolute/scaled differences | 0 / 0 for all six segments |
| Internal benchmark elapsed time | 2915.646 s |
| Whole-process wall time | 1:03:09 |
| Maximum host RSS | 12,389,024 KiB |
| Swaps | 0 |

The final recorded-scan transpose reports `objective_rows=11`, `groups=1`,
`contract=one_batched_scan_transpose`. Its parameter bars are finite. The
1,000-accepted-step guard did not bind. This run passed the earlier schedule
allocation OOM. Peak host RSS does not measure GPU memory or establish memory
stability over multiple optimizer evaluations.

## Raw transport observables and selected derivatives

The benchmark uses unit weights and zero targets for these transport rows, so
the saved residual values are the raw observables and the saved Jacobian entries
are their derivatives with respect to physical profile/boundary parameters.

| Observable | Full-time value |
| --- | ---: |
| Transition radius | 5.1703593282774429e-01 |
| Transition location moment | 1.7035932827744293e-02 |
| Transition strength | 2.0149622433190042e+00 |
| Bootstrap current softmax | 2.0426059244509895e+00 |
| Net power volume average (MW/m^3) | 9.5305334313012002e-01 |

| Parameter | Location moment AD | Strength AD | Bootstrap AD | Net-power average AD |
| --- | ---: | ---: | ---: | ---: |
| n0 | 5.6699915363382471e-05 | 4.7338579231761335e-01 | 1.5371077389306280e+00 | 1.0159391736863046e+00 |
| T0 | -1.1307263101707243e-06 | -9.3901900090527621e-04 | 1.5828898495447025e-03 | 5.0817114351032827e-04 |
| density_shape_power | -4.0333156219529525e-06 | -8.1676945636763327e-03 | -2.6017594138308862e-02 | 1.6272728525997065e-03 |
| temperature_shape_power | -2.6650224234697437e-06 | -2.2130059927788328e-03 | 3.7345851772254595e-03 | 1.1995750708314783e-03 |
| density_shape_alpha | 2.4180930096082660e-05 | 2.3347710583286281e-02 | 2.1973267528231283e-01 | -1.2002451549893472e-02 |
| temperature_shape_alpha | 2.6389471029123737e-06 | 2.1911294597008061e-03 | -3.7028769162505046e-03 | -1.1901715575170437e-03 |
| vmec:RBC:1:0 | -1.4395585793338666e-04 | 1.6089073904951214e+00 | 2.2705980419405488e+01 | 5.8907293269565768e+00 |
| vmec:ZBS:1:0 | 1.8381717423648172e-03 | 2.5790881567778305e-02 | -5.7627794601880700e+00 | 1.1512451530547568e-01 |

The radius row has the same derivatives as the location-moment row up to
floating-point precision. Full-time FD errors are pending: the old 16-step
FD results evaluate a different endpoint and cannot validate this table.

## Matching full-time FD run

With `jax_env` active on the GPU machine, the saved
[eight-parameter loop](run_database_transition_fd_full_t20_2026-10-19.sh) runs:

```bash
cd ~/NEOPAX
bash docs/benchmarks/run_database_transition_fd_full_t20_2026-10-19.sh
```

The loop preserves the completed AD JSON as `ad_reference.json`, uses a fresh
Python process for each of the six profile parameters and two geometry
parameters, and saves logs and full-precision FD JSON copies in
`outputs/database_transition_fd_full_t20_2026-10-19/`. A failed process stops
the loop. Each FD result contains the eleven transport objectives.

Each invocation uses the same benchmark TOML and these existing flags:

```bash
python examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py \
  --config examples/benchmarks/Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box.toml \
  --parameter n0 \
  --radau-jacobian-reuse-mode legacy \
  --geometry-fd-lane frozen_linearized \
  --replay-mode accepted \
  --fd-accepted-replay-contract reverse_frozen_metadata \
  --initial-Er-root-ad jax_selected_root \
  --initial-Er-root-fd-root-lane frozen_linearized \
  --diagnose-fd-endpoint
```

Omitting `--accepted-step-limit` requests the full 20-second baseline. The
TOML retains its 20,000-attempt budget. FD schedule discovery saves scalar
metadata without allocating the reverse probe's segment-start carry bank;
the AD accepted-step guard is not a flag of this FD runner.

The FD oracle freezes accepted timesteps, solver metadata and the selected
initial-root branch, and uses the aligned VMEC geometry linearization for
geometry columns. Its derivatives come from endpoint central differences.
The `recorded_forward` selector belongs to the AD runner; the FD runner uses
the established `reverse_frozen_metadata` contract rather than calling the
reverse sweep. This is an implicit accepted-step derivative comparison, not
a requirement for identical Newton-cache history or iteration counts.

Before judging gradient errors, compare all eleven FD baseline observables
with this saved AD reference and verify the baseline reaches 20 seconds with
290 accepted steps. Use the JSON derivatives at full precision. The earlier
16-step FD numbers and the seven additional geometry-only AD rows are outside
this full-time transport FD comparison.

Local validation only parsed supplied text, checked 18 complete rows / 144
finite derivatives and all six zero endpoint differences, and checked the
saved shell script syntax. No VMEC, NTX, transport, FD or optimization solve
was run locally.
