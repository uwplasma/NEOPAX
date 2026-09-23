# Database reverse performance modes

## Scope and reference

These controls apply only to the full-transport runtime-database reverse
benchmark. They do not change the root-only optimization lane, forward RHS,
Radau matrix/state derivative contract, geometry/coordinate derivative split,
accepted schedule, or compilation-cache policy. Defaults retain the completed
2026-09-12 performance run (`84e75e9` workspace baseline).

The measured reference is 22.264 s per warm four-step segment and 13.815 GiB
peak process RSS. The user target is roughly 3x the corresponding forward
solver time, with no increase in memory. No new full-run speedup is established
by the selector implementation or its small regression tests.

| Full-transport CLI option | Earlier implementation | Completed-run default | New candidate |
| --- | --- | --- | --- |
| `--reverse-database-initial-support-mode` | `generic` | `split` | `reduced_zero` |
| `--reverse-database-support-preparation-mode` | `separate` | `shared` | unchanged |
| `--reverse-database-center-geometry-mode` | `radial_vjp` | `scalar_jvp` | unchanged |
| `--reverse-database-stage-jacobian-mode` | `independent` | `independent` | `shared_multi_rhs` |
| `--reverse-database-support-objective-mode` | `scalar` | `scalar` | `batched_split` |
| `--reverse-database-segment-support-mode` | `inline` | `inline` | `deferred_segment_batch` |

Selecting `generic / separate / radial_vjp` restores the earlier performance
implementations for these three changed boundaries without reverting commits.
It does not revert derivative fixes or unrelated optimization work. Other
existing options remain independent: keep `block`, `grouped_vjp`, and
`joint_local_vjp_upar_only` when comparing these controls alone.

Explicit nondefault controls are rejected outside the database full-transport
shared-payload benchmark, including root-only smoke runs. Default configuration
returns the original physics context with identical callable identities.
Overridden support kernels remain `jit(inline=False)` boundaries with dynamic
state, geometry, tables, and cotangents.

## Initial-support candidate: `reduced_zero`

The grouped reduced reverse driver expands its carry cotangent by initializing
all fields to zero and restoring only `y`, `lagged_response_cache`, and
`lagged_reference_y`. The initial predictor-stage cotangents therefore remain
exactly zero. The initial direct-RHS support pullback receives only the sum of
these predictor-stage cotangents; its contribution in this driver is zero.

`reduced_zero` avoids tracing/compiling that support transpose. It requires an
explicit structural-zero contract from the caller, not a numerical threshold,
NaN replacement, or `stop_gradient`. General initial carries with nonzero
predictor-stage cotangents retain the `generic` and `split` implementations.

The initial state VJP, selected-root implicit state/support pullbacks, profile
parameter VJP, initial-profile support pullback, and recorded NTX scan fold
remain in place. This targets the 568.883 s initial-support phase, not the
22.264 s warm segment. That phase's elapsed time is not a measured saving for
the candidate; a full GPU run is still required.

## Validation order

1. Selectors/default identity and legacy propagation; nonzero/zero support
   cotangents; composed initial-state/root/profile derivative preservation.
2. Real-table centre geometry and shared/separate support equivalence.
3. Bounded stage-cost probe using production equations, exact finite `jacfwd`
   and coupled block solve. Separate Jacobian construction, matrix assembly,
   solve, and outgoing-state contraction. Inspect optimized HLO before claiming
   repeated source code causes repeated execution.
4. Add any demonstrated warm-segment candidate behind a new explicit selector.
   Do not change the current default or substitute a different state derivative.
5. Compare full 16/4 derivatives, synchronized warm timings, whole-process
   elapsed time and peak RSS. Only then promote a candidate.
6. Subsequently target startup/profile-root and terminal/bootstrap phases,
   preserving the optimization lane's root JIT boundaries and reusable caches.

The standalone probe is `tests/benchmark_database_reverse_stage_cost.py`. Its
stages are constructed from stored geometry and a fixed database, not recorded
converged stages from the long benchmark. Its timings are component evidence,
not a measurement of the full 16-step reverse/forward ratio.

The probe now selects **one** kernel per fresh process by default:
`--component independent` or `--component shared`. Support choices are
`support_table`, `support_flux_geometry`, `support_equation_geometry`, and
`support_table_and_geometry`. Use `--lower-only --dump-dir DIRECTORY` for
graph export without compiling the selected kernel. Fixture preparation still
runs. See [the support-cost audit and larger-machine command](database_reverse_support_cost_audit.md)
for the loaded-table size distinction and measurement limitations. The
probe control/metadata/comparison tests passed 22 cases without JAX compilation.

## Warm-stage candidate: `shared`

The stage matrix and outgoing state transpose use the same finite forward-mode
Jacobian evaluated at each converged stage. The candidate constructs those
Jacobians once and uses them for both operations. It retains the full coupled
Radau matrix, objective-vmapped dense solve, signs and contraction order. It
does not use coloring, Woodbury, a frozen/reference Jacobian, or the previously
problematic compact reverse-mode state VJP. Jacobians are local step
temporaries, not new scan-carry fields or trajectory records.

The candidate requires `block`, `explicit_database`, `full` stage cotangents,
`separate` RHS pullback and `default` stage-adjoint memory mode. Other
combinations are rejected. The CLI additionally restricts this candidate to
the tested `reduced_cotangent_call_boundary` step route.
`independent` retains the original solve/state
construction, including the default route.

In an initial local GPU probe (JAX 0.5.0, RTX 3060 Laptop, four species, five
radii, seven stages, ten objective rows), residuals and outgoing state bars
matched exactly elementwise. Optimized HLO was 17.89 MB versus 7.65 MB; reported FLOPs were
136.13 M versus 68.24 M; scratch storage was 43.10 MB versus 34.73 MB. This
confirms substantial duplicate derivative work survives compiler CSE in that
production-equation fixture. The initial two-sample warm medians were 38.59 ms
versus 27.65 ms, but were noisy/concurrent with CPU testing and are not a stable
speedup measurement. Larger-shape and full-segment verification remain needed.

A subsequent 51-radius probe (408 state variables; same species, stages and
objective count) also gave **exactly equal** residuals and outgoing state
bars. Exact measurements and raw warm samples are saved in
[`database_stage_cost_2026-09-12.json`](database_stage_cost_2026-09-12.json).

| Stage solve + outgoing-state component | Independent | Shared |
| --- | ---: | ---: |
| Lowering | 14.437 s | 2.639 s |
| Compilation | 95.995 s | 58.246 s |
| Warm median (7 calls) | 0.681442 s | 0.415849 s |
| Temporary device storage | 2.429 GB | 1.972 GB |
| Optimized HLO text size | 16.466 MB | 8.593 MB |

Samples drifted downward in both groups; preserve the raw samples rather than
treating this sequential local comparison as a stable full-run speedup.
Standalone medians were 0.2960 s for seven stage Jacobians, 0.00148 s for block
assembly, 0.2250 s for the dense solve and 0.00157 s for the saved-J contraction.
These component measurements are not additive timings of a fused kernel.

The subsequent support-table compilation in the same multi-kernel local probe
hit the WSL **host** memory limit: the kernel journal reported an OOM kill with
7,018,620 KiB anonymous RSS. No support timing was obtained. This was not a GPU
allocation failure or a failed 16/4 production run. Avoid retaining all probe
variants when testing support on a small host: use a fresh process and the
probe's `--support-only` option with a smaller radius count. This is measurement
isolation, not cache clearing in the production benchmark. Full production host
RSS remains unmeasured for the new selectors.

The fresh five-radius support-only fallback also failed to reach a timing
result: it was deliberately stopped when anonymous/process RSS was about
7,015,648 KiB and local memory/swap were almost exhausted. This second attempt
was a safety stop, not another observed OOM kill. Thus radius reduction alone
did not make the support compilation fit this local host. Do not treat the
support probe as a lightweight test. It needs a separate compiler-graph audit
and an adequately sized environment before further timing runs.

The literal new production shared-J helper subsequently passed a separate
five-radius GPU real-table parity check against the independently validated
shared-J algebra: all outputs finite, maximum absolute difference zero.
That check evaluated both implementations in one kernel; its timing is not
an isolated performance measurement of the new helper.

Verification: 67 focused mode/shared-J/support/root-preservation tests passed;
19 real-table/Radau/root-JIT compatibility cases passed separately. After the
final CLI restriction was added, 23 CLI/Radau/root-JIT cases passed again.
The actual benchmark `--help` also exposes all four controls successfully.

## Completed `shared_multi_rhs + batched_split` result

The completed 2026-09-13 16/4 run preserved every transport derivative at
printed precision and reduced peak host RSS from 13.815 GiB to 12.792 GiB.
It reduced the cold segment from 1009.543 s to 823.761 s, but the mean warm
segment increased from 22.264 s to 23.779 s. It therefore did not meet the
warm-runtime target. See
[the completed batched-modes record](database_reverse_16x4_2026-09-13_batched_modes_run.md).

## Deferred segment-support candidate

For the direct database RHS, a step's support cotangent does not feed the
state-adjoint recurrence. The exact residual-stage cotangents can therefore be
retained as a bounded numeric record while the four state steps run
sequentially, after which the four independent fixed-table/local-geometry
support contractions can be transformed together. The final recorded NTX
scan is still transposed exactly once after the complete segmented sweep.

`deferred_segment_batch` implements that schedule without changing the seven
converged-stage Jacobians, dense block solve, signs, state recurrence,
coordinate/table ownership or derivative formulas. It requires
`shared_multi_rhs`, `batched_split`, shared support preparation, `block`,
`explicit_database`, full cotangents, separate RHS pullback, default stage
memory, and the reduced-cotangent call boundary. All other combinations are
rejected, and the default remains `inline`.

The exact eager/JIT step result, padding mask, complete two-step segment result,
CLI isolation and mode plumbing passed 82 focused CPU tests. This establishes
mathematical and routing parity only. Warm GPU speed and peak RSS remain
unmeasured.

## Reproducible no-diagnostics 16/4 command

Run from the NEOPAX repository. This explicitly selects the completed-run
reference. For the initial-support candidate change only `split` to
`reduced_zero`. For the earlier implementation change the three values to
`generic`, `separate`, `radial_vjp` respectively.
For the completed batched candidate select `shared_multi_rhs` and
`batched_split`. For the deferred candidate additionally select
`deferred_segment_batch`. These options do not change the default.

```bash
env -u JAX_COMPILATION_CACHE_DIR \
  -u NEOPAX_DATABASE_GEOMETRY_VJP_DIAGNOSTICS \
  -u NEOPAX_DATABASE_STATE_VJP_DIAGNOSTICS \
  -u NEOPAX_DATABASE_GEOMETRY_VJP_DIAGNOSTIC_FACE_INDEX \
  JAX_ENABLE_COMPILATION_CACHE=0 \
  PYTHONPATH="$HOME/VMEX:$HOME/NTX/src" \
  /usr/bin/time -v \
  python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --config ./examples/benchmarks/Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box.toml \
  --reverse-parameter-mode profiles_plus_realtime_geometry \
  --reverse-geometry-parameter RBC:1:0,ZBS:1:0 \
  --realtime-geometry-gradient-path reverse_payload \
  --optimization-api-profile-dofs include \
  --objective all \
  --accepted-step-limit 16 \
  --radau-jacobian-reuse-mode legacy \
  --timing-mode jit-warm \
  --reverse-segment-length 4 \
  --reverse-stage-adjoint-solve-mode block \
  --reverse-rhs-transpose-mode explicit_database \
  --reverse-step-bwd-mode reduced_cotangent_call_boundary \
  --reverse-initial-cache-support-pullback-mode scalar \
  --reverse-rebuild-support-pullback-mode separate \
  --reverse-final-objective-cotangent-mode grouped_vjp \
  --reverse-bootstrap-cotangent-mode joint_local_vjp_upar_only \
  --initial-Er-root-ad jax_selected_root \
  --full-transport-shared-payload-smoke \
  --reverse-schedule-artifact-mode reuse_static_probe \
  --reverse-database-initial-support-mode split \
  --reverse-database-support-preparation-mode shared \
  --reverse-database-center-geometry-mode scalar_jvp \
  --reverse-database-stage-jacobian-mode shared_multi_rhs \
  --reverse-database-support-objective-mode batched_split \
  --reverse-database-segment-support-mode deferred_segment_batch
```

All selectors are printed in the full-transport progress banner and
stored in its JSON result. Keep the executed command and `/usr/bin/time -v`
footer alongside every comparison; do not infer the settings from a pasted,
possibly mangled shell header.
