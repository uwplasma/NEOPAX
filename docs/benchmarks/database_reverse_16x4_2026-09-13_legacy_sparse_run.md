# Database reverse 16x4 legacy-sparse run (2026-09-13)

This run measures the isolated legacy `Monoenergetic` sparse interpolation
transpose selected by
`--reverse-database-interpolation-transpose-mode legacy_sparse`.  The stable
database reverse configuration remains otherwise unchanged: scalar objective
support, inline segment support, independent exact block stage Jacobians, and
one final recorded-scan transpose.  Diagnostics, profiler traces, compilation
cache, and deferred segment batching were disabled.

## Outcome

The process completed with exit status zero.  Every raw block parameter bar was
finite, there was no swap activity, and all 136 residual-Jacobian entries
matched the established 2026-09-12 performance run.

Following this validation, the reverse benchmark CLI now selects
`legacy_sparse` for the database interpolation transpose and `grouped_vjp` for
the ordinary terminal-objective cotangents by default.  Its lane-local
`config` defaults also resolve initial-cache and rebuild support pullbacks to
the validated `scalar` and `separate` routes for database full transport.
Realtime/Lij keeps its existing NTX selections, and explicit user selections
are not overwritten.  The reference interpolation/objective modes remain
available explicitly as `established` and `scalar`.  This promotion is limited
to the benchmark CLI; the generic internal callback retains its prior defaults
so initial-Er root-only and unrelated programmatic callers are not rerouted.

| Quantity | Established run | Legacy sparse | Change |
|---|---:|---:|---:|
| Evaluation time | 3776.534 s | 2695.506 s | -1081.028 s (-28.63%) |
| Wall time | 1:18:20 | 59:04.86 | -19:15.14 (-24.58%) |
| Peak RSS | 14,486,496 KiB | 12,289,244 KiB | -2,197,252 KiB (-15.17%) |
| Cold segment 4/4 | 1009.543 s | 476.076 s | -533.467 s (-52.84%) |
| Warm segment 3/4 | 22.364 s | 0.844 s | 26.50x faster |
| Warm segment 2/4 | 22.391 s | 0.821 s | 27.27x faster |
| Warm segment 1/4 | 22.037 s | 0.745 s | 29.58x faster |
| Segmented sweep | 1076.337 s | 478.486 s | -597.851 s (-55.55%) |
| Initial direct-RHS support | 568.883 s | 183.143 s | -385.740 s (-67.81%) |
| Final recorded-scan fold | 707.702 s | 690.580 s | -17.122 s (-2.42%) |

The three warm four-step segments total 2.410 s.  Their mean is 0.803 s,
corresponding to about 3.21 s for sixteen reverse steps once the segment kernel
is compiled.  This meets the target of keeping the database reverse execution
within a few times the roughly two-second forward solve.  The remaining
476.076 s first-segment cost is dominated by cold compilation.

## Numerical parity

An automated comparison parsed the complete Jacobian tables from this run and
`database_reverse_16x4_2026-09-12_performance_run.md`.

- matched rows: 136 / 136;
- profile derivative rows: 60 / 60 bit-for-bit identical;
- transport geometry rows: maximum absolute difference `9.7811e-10`, maximum
  relative difference `9.6124e-11`;
- all rows, including geometry-only objectives: maximum absolute difference
  `9.0018e-08`, maximum relative difference `3.0340e-10`.

The larger absolute value in the all-row comparison belongs to a Boozer Max-J
derivative of magnitude approximately 1920; its relative difference remains
below `5e-11`.

## Phase timings

| Phase | Elapsed |
|---|---:|
| Runtime database build | 602.463 s |
| Solver components | 0.471 s |
| Profile-state VJP | 96.025 s |
| Initial carry VJP | 0.800 s |
| Realized-schedule forward replay | 1.563 s |
| Final-objective cotangents | 268.380 s |
| Segmented cotangent sweep | 478.486 s |
| Reduced carry expansion | 0.726 s |
| Initial direct-RHS support pullback | 183.143 s |
| Initial state pullback | 151.444 s |
| Initial-Er root compact pullback | 240.367 s |
| Profile parameter pullback | 0.897 s |
| Initial-profile scan payload pullback | 1.353 s |
| Final recorded-scan fold | 690.580 s |

The next performance work should preserve the verified sparse segment runtime
and target cold compilation and the independent terminal/root/final-scan
families.  Those changes require their own CLI modes and parity checks.

Two isolated initial-boundary modes were subsequently added for the first
follow-up timing experiment:

- `--reverse-database-initial-support-mode reduced_zero` uses the existing
  structural proof that the reduced accepted-step carry has zero
  `prev_stages` cotangents and therefore returns an exact zero initial-RHS
  support bar without tracing a support transpose;
- `--reverse-database-initial-state-mode reduced_zero_rhs` preserves the
  initial `y` and `lagged_reference_y` identities and the complete state
  packing/projection pullback, but does not trace the direct-RHS state branch.

The second mode requires the first and is restricted to the black-box Radau
database boundary.  Both remain opt-in until their full 16x4 timing and
Jacobian parity are measured.

## Resource report

```text
User time (seconds): 4036.65
System time (seconds): 230.69
Percent of CPU this job got: 120%
Elapsed (wall clock) time: 59:04.86
Maximum resident set size (kbytes): 12289244
Major page faults: 0
Minor page faults: 16975748
Swaps: 0
File system outputs: 510528
Exit status: 0
```
