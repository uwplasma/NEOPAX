# Combined geometry/profile optimization: open issues and derivative audit

Recorded 2026-10-14, branch `en/v0_new`. Remote run used commit `7c1b2c6`.

## Context and intended experiment

The user reports that single-stage combined geometry/profile optimization did
not recover the quality of the established geometry-only net-power/root/bootstrap
result. Profile motion mostly corrected net power, while root and bootstrap
targets remained insufficiently improved. The two-stage experiment was intended
to test whether first obtaining the geometry-only result would solve this: the
combined phase should begin at that accepted geometry with nominal profiles.

An actual handoff defect was found and fixed: the old least-squares adapter
always used `problem.x0`, even when the supplied initial evaluation came from
the accepted geometry coordinates. The adapter now accepts an explicit `x0`,
and the warm-start caller passes it. This fix establishes consistent starting
coordinates and residuals; it does not prove that the full transport Jacobian
is correct or that a multi-objective optimization will improve every term.

The current isolated reproduction is
`examples/optimization/optimize_geometry_profiles_from_geometry_result.py`.
It restores saved geometry-only coordinates against their original baseline,
adds six analytical profile DoFs, and runs only the combined solve. CUDA command
buffers are disabled before JAX initialization by default. The original
two-stage script does not automatically apply that backend flag. The updated
two-stage handoff also keeps host-side coordinates/scales instead of retaining
the entire geometry-only problem during phase 2.

## 1. Host-memory growth remains open

The combined-only run avoids the previous CUDA graph-instantiation OOM, but
process RSS still grows despite stable measured stage-cache sizes and stable
successful-evaluation live GPU allocation.

| Evaluation | RSS (MiB) | Live GPU bytes | Live JAX arrays |
| --- | ---: | ---: | ---: |
| 1 | 10715.2 | 48666368 | 689 |
| 13 | 10910.5 | 48679680 | 743 |
| 19 | 11096.4 | 48673024 | 717 |
| 28 | 11311.1 | 48679680 | 743 |
| 29 | 11423.7 | 48679680 | 743 |
| 30 | 11582.4 | 48679680 | 743 |

RSS increased 867.2 MiB between evaluations 1 and 30; the final increments
were 112.6 and 158.7 MiB. This is not an established flat-memory result.
Live GPU allocation after successful evaluations is about 46.4 MiB; peak GPU
allocation remains about 5.54 GiB. The five measured persistent stage caches
remain at one entry after warmup. These probes do not cover every JAX dispatch
cache, native compiler allocation, or allocator-retained buffer.

Failed-evaluation samples temporarily show roughly 5600 live arrays and
427 MiB of live GPU allocations. The memory wrapper samples in `finally`,
before exception traceback cleanup; subsequent successful evaluations return
to approximately 717-743 arrays and 46.4 MiB. This explains why those samples
cannot alone establish persistent GPU-array retention. Host RSS growth is
still unexplained.

## 2. Reverse-AD trial failures remain open

Some candidates fail with `nonfinite recorded database scan cotangent at input
fixed-table bars`, including evaluations 2, 7, 10, 16, 17, and 18 in the supplied
excerpts. Nonfinite leaves include `a_b`, `Er_list`, `D11_log`, `D13`, and
sometimes `D33`. The check in
`NEOPAX/_reverse_ad_initial_er.py:fold_recorded_ntx_scan_database_bar_groups_into_support`
rejects these bars before the final recorded NTX scan transpose runs. The
precise upstream producer within transport/root reverse is not yet identified.

The user notes similarity to a previously fixed helium density-floor failure.
Treat this as a hypothesis, not an established common cause. The existing fix
in `initial_state_for_parameter_vector_compact` constructs pressure from
configured temperature and safe density, preserving finite temperature for
zero-concentration helium. Its regression is
`tests/test_main_helpers.py:test_reverse_initial_state_preserves_fixed_temperature_at_density_floor`.
Remaining floor handling in later transport/reverse boundaries still warrants
inspection. Do not fix this by silently zeroing nonfinite physical derivatives.

OOM/resource-exhaustion errors now propagate instead of being retried as
ordinary failed physical trials. Ordinary nonfinite candidate trials still
receive the established optimizer penalty.

## 3. Root/bootstrap progress and derivative validation

The latest combined-only solve exhausted 30 function evaluations, with status
0 and first-order optimality about 127. It did not establish convergence.
The final accepted point was evaluation 29; evaluation 30 had a slightly
higher trial cost.

| Term | Saved geometry start | Final accepted evaluation 29 |
| --- | ---: | ---: |
| Total cost | 3.58735782 | 1.58009824 |
| Transition radius | 0.556547106 | 0.556570729 |
| Transition-location cost | 0.905128115 | 0.906133480 |
| Crossing-strength cost | 4.23051312e-24 | 4.28071688e-24 |
| Bootstrap penalty value | 0.464691425 | 0.402322865 |
| Bootstrap cost | 0.431876241 | 0.323727376 |
| Net power (MW) | 293.597137 | 300.102792 |
| Net-power cost | 1.87073863 | 0.000482147657 |

The root-location term is essentially stalled and slightly worse. Bootstrap
cost improves by about 25%, but remains active. Crossing strength already
exceeds its 0.9 target, so its near-zero penalty is expected. Net-power
improvement supplies most of the total cost reduction.

Current default objectives use transition location/existence, bootstrap, and
net power. Maximum-Er and the old fixed-index left/right Er targets are off.
The location target is 0.514, with weight 1000; strength-deficit weight is
100; bootstrap weight is 4; net-power weight is 10000. Weights alone do not
establish which term dominates: residual and Jacobian magnitudes both matter.

The new location and strength functions live in
`NEOPAX/_reverse_ad_transport.py:smooth_positive_to_negative_er_transition_metrics`.
Candidate locations use a smoothed chord-interpolated zero, not just a radial
face midpoint. The location residual is selected location minus target radius.
The strength residual applies a smooth positive-part transform to target minus
strength. Bootstrap applies a hard positive-part transform above its limit.
The least-squares assembly must include each transform's chain rule and the
square root of its objective weight exactly once, followed by the parameter
coordinate scaling exactly once.

Existing validation found in source:

- `test_full_transport_smooth_transition_location_ignores_near_axis_zero`
  checks values, finite gradients, and selected gradient signs. It is not an
  independent numerical derivative comparison for the new residuals.
- Existing transport/reverse parity tests compare implementations sharing
  physics and may not validate the new optional objective derivatives against
  independent perturbations of optimizer parameters.
- Prior lightweight solver tests established starting-coordinate and setup
  plumbing, not end-to-end plasma-physics Jacobian accuracy.

The historical 16-step reverse benchmark selected the original ten transport
rows, including fixed left/right Er values, and omitted all three new transition
rows. The current benchmark's `--full-transport-shared-payload-smoke --objective all`
now replaces those two fixed-index rows with `Er_transition_rho`,
`Er_transition_location_moment`, and `Er_transition_strength` (eleven transport
rows). The same label order is passed to the reverse builder and reported
Jacobian. Individual new rows are also selectable with `--objective` in that
lane. This is a reverse-Jacobian benchmark, not an independent finite-difference
validation of the complete transport/geometry chain.

The matching full-transport FD runner now uses the same eleven objective labels
and production scalar value functions. Baseline, perturbed endpoints, split
diagnostics, and JSON labels use that order; derivatives are still obtained
from central endpoint differences. The independent root-only FD mode retains
its existing objectives. The paired GPU command and comparison scope are in
[the AD/FD benchmark note](benchmarks/database_transition_objectives_ad_fd.md).
Only local synthetic objective/routing checks have been run for this update;
the complete 16-step GPU comparison remains pending.

The user subsequently supplied a successful updated 16-step AD run. It contains
all eleven transport rows plus seven geometry rows, with every entry of the
18-by-8 Jacobian finite. The new root rows have nonzero derivatives in every
parameter column. The
[archived AD reference](benchmarks/database_transition_objectives_ad_16x4_2026-10-14.md)
preserves the full output and CSV table. This confirms participation of the
new rows, while their full-chain numerical FD comparison remains pending.

### Independent derivative audit

Added `tests/test_root_bootstrap_objective_derivatives.py`, executable with:

```bash
JAX_PLATFORMS=cpu python -m unittest discover -s tests -p test_root_bootstrap_objective_derivatives.py -v
```

The checks use independent NumPy formulas and central finite differences.
They exercise the production terminal-objective dispatch, the compact
bootstrap Upar cotangent, and the actual mixed residual assembly/scaling.
The residual-assembly fixture supplies a synthetic, explicitly known raw
parameter Jacobian: it does not substitute for a transport Jacobian test.

Final local audit: all five test methods passed in 2.127 seconds of test
execution (excluding imports). These methods include the cases below.

| Check | Maximum absolute derivative discrepancy |
| --- | ---: |
| Transition metrics, resolved crossing with axis zero | 1.896e-10 |
| Transition metrics, sharp crossing | 5.352e-10 |
| Transition metrics, one-sign profiles | 2.489e-09 |
| Transition metrics, zero profile | 4.904e-08 |
| Grouped terminal VJP, Er and radial-grid cotangent rows | 1.896e-10 |
| Compact bootstrap Upar bar, both signs and array orientations | 1.927e-14 |
| Weighted residual Jacobian and coordinate scaling | below 3e-10 |

The transition comparison includes derivatives with respect to both Er and
the radial grid. The weighted residual checks cover active/inactive strength
and bootstrap penalties, zero-weight diagnostic rows, the scalar-transform
chain rule, and nominal-profile/geometry column scaling. A strong isolated
crossing has a nonzero location derivative within its selected face: the
metric is not frozen to a face midpoint. Uniformly rescaling that example's
Er amplitude leaves its chord location unchanged, illustrating a genuine
insensitive direction rather than a missing derivative. This synthetic
example does not establish what directions the remote run actually took.
The bootstrap hinge is nondifferentiable exactly at its limit; numerical
checks cover both sides away from that point. The reported remote bootstrap
value is well above the limit, so a satisfied/inactive bootstrap hinge does
not explain the remaining penalty there.

The local formulas and residual assembly agree with numerical derivatives in
these cases; no corresponding physics-formula change is justified by this
audit. End-to-end geometry/profile Jacobian validation at the saved remote
point has not been established. In particular, correctness of local terminal
cost derivatives does not certify the segmented transport reverse, selected
initial-root pullback, momentum-corrected flux pullback, or VMEC transpose.
No VMEC, NTX database, or full transport solves were run locally.

### Remaining validation needed to explain the plateau

1. At the saved combined starting point, inspect each active residual's
   geometry/profile Jacobian blocks and contributions to `J.T @ r`, rather
   than only the total cost or nominal objective weights.
2. Compare the complete weighted residual Jacobian against central
   differences in separate geometry-only and profile-only directions at
   that same point, using more than one perturbation size. Check selected-Er
   branches and realized transport schedules when interpreting differences.
3. Locate the first nonfinite reverse contribution upstream of the recorded
   database fold, including density/temperature floor handling. The previous
   helium initialization fix is not proof that this later failure has the
   same cause.
4. Inspect host-side compilation/dispatch retention outside the five measured
   stage caches. Stable live GPU arrays do not establish bounded host RSS.

These expensive end-to-end checks have not been run on the local CPU machine.

## 4. Original geometry-stage reproducibility report

The user also reported that phase 1 of the original two-stage script did not
match the standalone geometry-only optimization result. Mocked setup checks
found matching configuration, objective definitions, builder arguments,
starting coordinates, and least-squares options. That is a source/plumbing
check, not numerical trajectory parity. This original report remains open.

## Evidence

Remote output directory:
`/home/exouser/NEOPAX/outputs/combined_from_geometry_result`.
Saved geometry-only directory:
`/home/exouser/NEOPAX/outputs/geometry_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_optimization`.

User-provided log attachments in this workspace session:

- `e190d9ba-db31-44b4-a8fd-817e52715b27/Pasted text.txt`: initial combined-only
  evaluations and nonfinite reverse bars.
- `9e5cf0c6-cd1b-4a44-bbf3-914ae73377da/Pasted text.txt`: evaluations 13-19.
- `2b353984-e5a8-426b-b327-8fc0e9ee8ea7/Pasted text.txt`: evaluations 28-30
  and final optimizer summary.

These are under the user's `.codex/attachments` directory. They are evidence
for the observations above, not numerical derivative validation artifacts.
