# Completed database reverse 16/4 batched-modes run — 2026-09-13

Source: user attachment
`876b5e83-765d-4bde-b7a8-f8d994ad79ee/pasted-text.txt`.
The pasted header is malformed; the `/usr/bin/time -v` footer records the
executed command. The run completed successfully with
`--reverse-database-stage-jacobian-mode shared_multi_rhs` and
`--reverse-database-support-objective-mode batched_split`. It did **not** use
the later `deferred_segment_batch` candidate.

## Result relative to the preceding completed run

| Metric | Preceding run | Batched-modes run | Change |
| --- | ---: | ---: | ---: |
| Whole-process wall time | 1:18:20 (4700 s) | 1:13:40 (4420 s) | -280 s (-5.96%) |
| Benchmark internal elapsed | 3776.534 s | 3556.578 s | -219.956 s (-5.82%) |
| Peak host RSS | 13.815 GiB | 12.792 GiB | -1.023 GiB (-7.41%) |
| Cold segment 4/4 | 1009.543 s | 823.761 s | -185.782 s (-18.40%) |
| Mean warm four-step segment | 22.264 s | 23.779 s | +1.515 s (+6.81%) |
| Initial direct-RHS support pullback | 568.883 s | 617.280 s | +48.397 s (+8.51%) |
| Final recorded-scan fold | 707.702 s | 692.644 s | -15.058 s (-2.13%) |

The candidate reduced compilation-dominated and whole-process measurements,
but it failed the principal warm-segment target. The three warm segments were
22.240, 22.938 and 26.160 seconds; the target remains approximately six
seconds for a four-step segment.

## Other reported phases

| Phase | Seconds |
| --- | ---: |
| Runtime database/geometry build | 608.101 |
| Solver components | 0.487 |
| Profile-state VJP | 92.684 |
| Initial carry VJP | 0.795 |
| Realized-schedule forward | 1.516 |
| Final-objective cotangents | 266.166 |
| Complete segmented sweep | 895.101 |
| Reduced carry expansion | 0.974 |
| Initial state pullback | 156.497 |
| Initial-Er root compact pullback | 249.745 |
| Profile parameter pullback | 0.961 |
| Initial-profile scan payload pullback | 1.383 |

The process used 4978.42 user seconds and 256.09 system seconds, reported 118%
CPU utilization, wrote 592312 filesystem blocks, had no major page faults or
swaps, and exited with status zero.

## Derivative preservation

All ten transport objective values and all 80 transport Jacobian entries are
identical to the preceding completed run at printed precision. The only
printed changes are four geometry-only VMEC derivatives, with maximum relative
change below `8e-12`; these are ordinary run-to-run floating-point differences.
Consequently the saved 16-step AD-versus-FD conclusions are unchanged,
including the approximately `1.970e-3` RBC net-power error and bootstrap errors
of approximately `2.404e-3` (RBC) and `3.413e-3` (ZBS).

This result therefore validates derivative preservation and modest peak-RSS
reduction, but **does not validate a warm reverse speedup**.
