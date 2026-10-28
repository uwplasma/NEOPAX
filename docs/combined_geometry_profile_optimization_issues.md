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

### Stage-2 profile-only control and initial-profile plots (2026-10-16)

The saved-result runner now accepts `--profile-only`. SciPy then receives only
the six profile coordinates and six Jacobian columns; every evaluation and
saved VMEC input uses the accepted geometry coordinates from stage 1. All
existing objective rows and weights are retained. The same combined evaluator
is used internally, including its geometry reverse work, so this is a control
of optimizer variables rather than a separate, faster profile-AD backend.
The existing reverse failures and memory issue are still under investigation.

Run on the GPU machine, with the usual environment active:

```bash
cd ~/NEOPAX
PYTHONPATH="$HOME/VMEX:$HOME/NTX/src" \
python examples/optimization/optimize_geometry_profiles_from_geometry_result.py \
  --geometry-result-dir outputs/geometry_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_optimization \
  --profile-only \
  --out-dir outputs/profiles_only_from_geometry_result_diagnostics \
  --diagnose-optimization
```

Omitting `--profile-only` runs the existing geometry-plus-profiles case. The
default output directories are separate (`profiles_from_geometry_result` and
`combined_from_geometry_result`); an explicit `--out-dir` takes precedence.

Both modes now save `profile_parameters_initial.json` as well as the optimized
parameters, and by default write:

- `initial_profiles_initial_vs_optimized.png`: density and temperature at
  **t = 0** for the starting and optimized cases, plus their differences.
  All configured species, including zero-density helium, are included.
- `initial_optimized_initial_profiles.csv`: the plotted radial values in
  units of `1e20 m^-3` and `keV`.

The plots reconstruct the existing analytical initialization map on the
normalized radial grid, preserving configured temperatures and density-floor
handling. They do not solve VMEC, an ambipolar root, or transport again. Use
`--no-plot-initial-profiles` to omit this postprocessing. The summary records
both the six-variable optimizer vector and the full vector with fixed geometry
in profile-only mode.

To plot an already completed combined run, including one made before this
plotting addition, use its existing output directory:

```bash
PYTHONPATH="$HOME/VMEX:$HOME/NTX/src" \
python examples/optimization/optimize_geometry_profiles_from_geometry_result.py \
  --geometry-result-dir outputs/geometry_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_optimization \
  --out-dir outputs/combined_from_geometry_result_diagnostics \
  --plot-profiles-only
```

This mode reads the stored optimized parameters, uses stored initial parameters
when available (otherwise the saved geometry run's nominal profile config),
and exits without constructing a physics problem or running an optimizer. New
stage-2 summaries also store their config for reproducible plot regeneration.

Validation: 25 lightweight diagnostics/lifecycle/continuation checks pass,
including an actual small SciPy solve that never changes the saved geometry,
six-column diagnostic output, initial-profile CSV values and finite helium
temperature, and plotting a legacy completed run with physics builders and
the optimizer forbidden. No VMEC, NTX, transport, or FD solve was run locally.

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
The local checks for this update use synthetic objectives and mocked routing.
Remote validation results are recorded below; all eight columns now have
FD results, with the geometry comparisons still unresolved.

The user subsequently supplied a successful updated 16-step AD run. It contains
all eleven transport rows plus seven geometry rows, with every entry of the
18-by-8 Jacobian finite. The new root rows have nonzero derivatives in every
parameter column. The
[archived AD reference](benchmarks/database_transition_objectives_ad_16x4_2026-10-14.md)
preserves the full output and CSV table. This confirms participation of the
new rows.

A subsequently supplied sixteen-step `n0` FD run matches all eleven baseline
values exactly at printed precision and agrees with the AD column to a maximum
relative error of 3.367259e-07. The new radius/location errors are 2.236e-08 and
the strength error is 1.312e-07, computed from the rounded `.6e` console
gradients. The [updated comparison table](../reverse_ad_vs_fd_database.md#2026-10-15-transition-objective-adfd-comparison-current-eleven-row-case)
records the values. The remaining seven FD runs have subsequently completed:
all six profile columns share the AD baseline and agree to a maximum relative
error of 1.948462e-06 across all eleven objectives. The largest profile error
for strength is 2.213691e-07, and for net power it is 3.666790e-07.

The first pair of geometry FD runs used a baseline whose root radius is
0.61290799115125838 instead of AD's 0.61293923577522791, strength is
2.1958860308890658 instead of 2.2261594486928726, and volume-averaged net power
is 0.50804023882399219 instead of 0.50807530056153805. Their apparent root
location errors are 7.25%/6.30%, strength errors 24.21%/17.12%, and net-power
errors 80.29%/0.0562% for RBC/ZBS. These are baseline-mismatched comparisons;
the source of the primal difference must be resolved before attributing them
to incorrect geometry derivatives. The profile checks apply at the benchmark
point, not the saved combined-optimization point. Memory growth and failed
reverse trials remain open.

A subsequent [historical-command and source audit](benchmarks/database_geometry_fd_context_mismatch_2026-10-15.md)
found a concrete geometry configuration mismatch: the benchmark helper
hard-coded Boozer resolution `(18, 18)` when the TOML omitted it, while the
production runtime selected automatic `(36, 11)` for the same VMEC input.
The helper now follows the runtime's optional-value handling, and geometry FD
prints its resolved resolution. The same helper is used by AD geometry, so
new geometry checks need a refreshed AD reference and matching baselines.
The supplied command's replay/root flags were appropriate. At the time of
that correction, only mocked/synthetic checks had verified the change.

The user has subsequently supplied the full eight-parameter FD rerun with
geometry banners reporting `(36, 11)`. All runs succeed and all 88 baseline
objective values match the archived AD reference at printed precision.
The [new comparison](../reverse_ad_vs_fd_database.md#2026-10-15-boozer-aligned-fd-rerun-all-eight-dofs)
confirms that the observed geometry baseline mismatch is resolved. Against
the same archived AD columns, RBC/ZBS relative errors are 1.520258e-04/
2.319614e-05 for radius/location, 5.968463e-04/3.284039e-05 for strength,
and 4.998444e-04/5.595834e-06 for net power. Bootstrap remains at
3.220081e-03/7.550967e-03. Profile gradients are unchanged at printed
precision. These are comparisons with the older AD run; a refreshed AD
reference using the corrected shared context remains outstanding. No new
evidence resolves host-memory growth, failed trials, or behavior at the
saved combined-optimization point.

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

### Interpretation after the benchmark gradient comparison (2026-10-15)

The improved AD/FD agreement makes an omitted or identically zero transition
derivative less likely at the benchmark point. It does not establish the
full-final-time Jacobian at the saved optimized geometry, or explain why the
optimizer stops making progress there.

Re-reading the supplied combined-only logs gives a concrete rejected trial:

| Quantity | Initial evaluation 1 | Trial evaluation 5 | Final accepted evaluation 29 |
| --- | ---: | ---: | ---: |
| Transition radius | 0.556547106 | 0.536939071 | 0.556570729 |
| Transition-location cost | 0.905128115 | 0.263100495 | 0.906133480 |
| Bootstrap cost | 0.431876241 | 0.227883226 | 0.323727376 |
| Net power (MW) | 293.597137 | 274.038798 | 300.102792 |
| Net-power cost | 1.87073863 | 30.7548979 | 0.000482147657 |
| Total cost | 3.58735782 | 31.5602133 | 1.58009824 |

Trial 5 improves both location and bootstrap, but its power error raises the
total cost above the current point, so ordinary TRF rejects it. This is direct
evidence that the evaluated trial points can move the root; it does not prove
that this trial direction is the best available tradeoff. The subsequent
accepted points favor correcting power while keeping the root near 0.55657.
The geometry variables remain active in the combined problem; warm starting
does not constrain the individual phase-1 objective values to improve.

However, power accounts for only 0.030514 percent of the final accepted cost.
The nominal power weight alone therefore does not establish why the later
root plateau persists. Small objective cost also does not imply a small
gradient or curvature: those depend on the weighted Jacobian. Bootstrap cost
falls by 25.0416 percent over the run, while root location moves only
2.3623e-05 and slightly worsens its target error. Strength is already above
its target, so its inactive penalty is expected.

At least six evaluations in the supplied excerpts (2, 7, 10, 16, 17, 18) fail
in reverse AD. The least-squares adapter evaluates residuals and Jacobian
together; an exception in that reverse computation is returned to SciPy as
an all-1e6 residual vector. Thus derivative failures can cause rejection of
trial points regardless of their physical objective values, restricting the
search. The logs do not establish the first upstream nonfinite producer or
prove that these failures alone cause the plateau.

The solve terminates on the 30-function-evaluation budget, with first-order
optimality about 127, rather than convergence. Function evaluations include
rejected and failed trials; they are not 30 accepted iterations. Adding
profile freedom changes the local step allocation and trajectory, even
though holding profiles fixed still permits the geometry-only directions.

The unresolved diagnosis is the balance of weighted Jacobian rows, parameter
scales, bounds, and realized versus predicted reductions at the actual warm
point, together with the failed reverse trials. The successful benchmark
does not provide those quantities. Recording the already-computed residuals,
Jacobians, and trial coordinates would permit that analysis without adding
physics solves; no expensive local test was run for this log/source audit.

### Diagnostics added for the next combined run

The two-stage runner's combined phase, the simple combined runner, and the
saved-geometry combined-only runner now default to `--diagnose-optimization`.
Use `--no-diagnose-optimization` to disable it. The geometry-only phase does
not enable this recorder. The core `optimization.least_squares` adapter
accepts an optional `diagnostics_dir`; its ordinary default remains disabled.

Each invocation creates a unique `run_*` subdirectory. For the saved-geometry
runner this is under `OUT_DIR/optimization_diagnostics`; for the two-stage
runner it is under `OUT_DIR/combined_diagnostics_mMODE`. Files contain:

- `events.jsonl`: initial/accepted points distinguished from evaluated trials,
  residual costs, geometry/profile Jacobian norms and `J.T @ r` contributions,
  row-sensitivity correlations, singular values, physical coordinate scales,
  trust scales, bounds, and the final SciPy status/optimality.
- `trial_NNNN.npz`: each successful evaluation's unmodified residual vector,
  full weighted Jacobian, optimizer coordinates, parameter/residual labels,
  and physical coordinate scales. Nonfinite entries are recorded before the
  adapter's existing residual/Jacobian replacement policy.
- Trial records compare each step from the last accepted point against that
  point's linear residual model, separating geometry/profile contributions
  and reporting predicted versus actual changes. This is the unregularized
  weighted residual model, not SciPy's internal augmented bounded-TRF ratio.
- Failed-trial events retain coordinates, exception type, traceback filename,
  the preceding linear model's prediction, and any terminal transport
  objective values computed before the reverse failure. They identify the
  first observed nonfinite reverse boundary when available.
- Reverse checkpoint events inspect the terminal objective bars, each
  segment's state/support bars and accumulated support, initial direct-RHS
  and state pullbacks, selected-root contributions, initial-profile support,
  and recorded scan boundaries. Entries identify affected objectives and
  leaves, shapes, first nonfinite indices, and counts. Initial/final state
  summaries include density/pressure/temperature ranges by species; selected
  root masks include the number of selected roots and unselected indices.

Acceptance is observed at SciPy's Jacobian request at its initial/accepted
TRF points. A successful residual evaluation alone is not labeled accepted.
The recorder supports the examples' linear-loss TRF path. A precomputed
initial evaluation is saved without repeating its physics or reverse work;
reverse checkpoints are active during subsequent delegated evaluations.

The recorder keeps only small host arrays for the last trial and accepted
linearization, and streams records to disk. It adds metadata reductions,
synchronizations, small host linear algebra, and file I/O, but no forward
solve, VJP, reverse-segment replay, or retained device-array history. It does
not enable the older opt-in segment replay diagnostics. A first bad boundary
locates a phase or segment; it does not identify an individual operation
inside a compiled segment.

GPU-only command continuing from the existing geometry result:

```bash
PYTHONPATH="$HOME/VMEX:$HOME/NTX/src" \
python examples/optimization/optimize_geometry_profiles_from_geometry_result.py \
  --geometry-result-dir outputs/geometry_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_optimization \
  --out-dir outputs/combined_from_geometry_result_diagnostics \
  --diagnose-optimization
```

Local verification ran only synthetic/mocked tests: 17 diagnostics/solver
tests and 3 saved-result runner tests passed. A bounded scalar SciPy problem
follows the same evaluation coordinates and count with diagnostics on/off;
other checks cover rejected/failed versus accepted points, exact analytic
step predictions, raw nonfinite Jacobian preservation, scan-guard failure
provenance, root-mask/species-floor summaries, and fatal OOM propagation.
No VMEC, NTX, transport, or GPU optimization was run locally.

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

## 4. Instrumented remote run supplied 2026-10-16

The user supplied the complete console output of the saved-geometry
combined-only runner with `--diagnose-optimization`. The output is archived
byte for byte in [the console record](benchmarks/combined_from_geometry_result_diagnostics_2026-10-16.txt)
and parsed into [a 30-evaluation CSV](benchmarks/combined_from_geometry_result_diagnostics_2026-10-16.csv).
Source attachment: `dcabc25e-abdf-490f-bcd4-10d625f77996/Pasted text.txt`.
Raw SHA-256: `65307552afb53fb46a8d0efb476ca37b0d459a51f298bf231fe42caef04ed067`.
The remote Git commit/dependency versions are not printed.

The runner confirms that geometry phase 1 was skipped, 30 combined parameters
were active, the database grid was `(25,25,33)`, and CUDA command buffers were
disabled. The initial objective values match the previous saved-geometry
start at printed precision. The diagnostic directory is:

```text
/home/exouser/NEOPAX/outputs/combined_from_geometry_result_diagnostics/optimization_diagnostics/run_jumo5hl4
```

### Accepted solution still exhibits the root plateau

There are 23 evaluations returning residuals/Jacobians, including the initial
point, 17 accepted updates, and 5 rejected successful evaluations. Seven
evaluations fail in reverse AD. The final accepted point is trial 29; trial
30 has higher cost and is rejected. The 30-evaluation limit stops the solve
with status 0 and first-order optimality `3.10e+05`. This stopping condition
does not explain why the root fails to improve.

| Quantity | Initial point | Final accepted trial 29 |
| --- | ---: | ---: |
| Total cost | 3.58735782 | 1.61019709 |
| Transition radius (target 0.514) | 0.556547106 | 0.556570780 |
| Location cost | 0.905128115 | 0.906135666 |
| Transition strength | 1.75939014 | 1.75650827 |
| Bootstrap penalty | 0.464691425 | 0.417396899 |
| Bootstrap cost | 0.431876241 | 0.348440343 |
| Net power (MW) | 293.597137 | 300.134477 |
| Net-power cost | 1.87073863 | 0.000825206979 |

The radius changes by only `+2.3674e-05`, slightly worsening its target
error. Bootstrap cost improves by 19.3194 percent. Strength already exceeds
its 0.9 target and its deficit cost remains negligible. Power contributes
only 0.051249 percent of the final accepted cost, so continued power-cost
dominance is not established by the cost breakdown. Row sensitivities and
curvature at this point are still needed to assess the actual step choice.

The successful and failed sequence differs from the previous run (for
example trial 5 now fails), and the final cost is 1.61019709 rather than
1.58009824. Trial numbers alone do not establish identical coordinates
between runs; the console does not establish why their trajectories differ.

### First nonfinite checkpoint is within segmented transport reverse

Every failed trial first reports the reduced state cotangent leaf `.y`
with shape `(13,408)`, and all 5304 entries are nonfinite. All thirteen
objective rows are affected, including pressure, temperature, power,
bootstrap, and the transition observables.

| Failed trial | First bad checkpoint | Total segments |
| --- | --- | ---: |
| 2 | `segment_2_state_bars` | 4 |
| 5 | `segment_4_state_bars` | 5 |
| 7 | `segment_3_state_bars` | 5 |
| 9 | `segment_4_state_bars` | 5 |
| 10 | `segment_4_state_bars` | 5 |
| 23 | `segment_3_state_bars` | 5 |
| 24 | `segment_3_state_bars` | 5 |

The reverse segments execute in descending order. Thus these checkpoints
follow already-completed later-time segments and precede the initial-state
and selected-initial-root pullbacks and the final recorded scan transpose.
The eventual `input fixed-table bars` exception detects downstream
nonfinite bars; it is not the first observed failure boundary.

Source tracing confirms that the database segment executes a reverse scan
over its recorded accepted-step primal data, using the common transport
state adjoint and rebuild pullback for the objective batch. The exact
`block` lane solves the transposed stage-residual matrix before assembling
the state/support pullbacks. A nonfinite or singular shared stage system,
bad RHS derivative, a bad rebuild pullback, or adjoint overflow are possible
producers inside the first bad segment. The console does not distinguish
them. In particular, all-objective contamination does not establish a root
metric derivative error or prove a helium density-floor cause. No physics
formula or derivative was changed on this evidence alone.

### Host-memory growth persists, including successful trials

| Evaluation | RSS (MiB) | Live JAX arrays | Live device bytes |
| --- | ---: | ---: | ---: |
| 1 | 6318.4 | 689 | 48504576 |
| 13 | 8525.0 | 744 | 48517888 |
| 26 | 10038.3 | 744 | 48517888 |
| 29 | 10506.0 | 744 | 48517888 |
| 30 | 10662.8 | 744 | 48517888 |

RSS increases 4344.4 MiB (4.24 GiB) across the run. Evaluations 26-30 alone
add 624.5 MiB despite identical sampled live-array counts/device bytes and
all five measured stage caches staying at one entry. The five evaluations
all return residuals and Jacobians, though trial 30 is rejected by TRF.
Therefore failed-trial traceback lifetimes alone do not explain the late
growth. Successful live GPU allocation is about 46 MiB and sampled peak GPU
allocation plateaus at 5882395648 bytes. Failed-evaluation samples temporarily
rise to about 449 MB, returning to the successful baseline afterward.
This is persistent host-RSS growth, not a demonstrated persistent live
GPU-array increase. It does not identify the responsible host allocation or
establish whether the new metadata checks contribute to that growth.

### Initial request for the saved event artifact

The attachment is the console output only. The existing remote
`events.jsonl`, NPZ snapshots, and failed-trial traceback files were not
attached and are not present in the local checkout. Weighted row/column
gradients, model predictions, bounds, and species/root-mask summaries are
saved in that JSONL; they are not printed in the console excerpt.

The next input is `events.jsonl` from the completed `run_jumo5hl4` directory.
It also contains terminal transport objective values for failed candidates
when computed before failure, which can reveal whether a promising physical
trial was discarded due to reverse AD. Reading that existing file requires
no new optimization run. No expensive local test or physics solve was run
to parse, archive, or inspect this evidence.

### Event excerpts supplied subsequently: mechanism of the root plateau

The user subsequently supplied seven event-file fragments. They are archived
with byte hashes and line coverage in
[the excerpt manifest](benchmarks/combined_diagnostics_events_2026-10-16/manifest.json).
Raw parts are preserved separately. The normalized excerpt contains 784
unique complete JSON events after removing five exact duplicate records;
five cut JSON lines and one shell prompt are omitted only from that derived
excerpt. In particular, the setup and early trial records are absent, and
trial 18's Jacobian record is cut across an unrecoverable gap. No missing
values or JSON records were reconstructed.

There are 17 complete trial/Jacobian records (11-17, 19-22, 25-30), three
failed-trial records (10, 23, 24), and the final solver record. Their values
agree with the archived console run. The recorder stores each weighted
row's contribution `g_i = r_i * J_i`, so its nonzero-residual Jacobian rows
can be recovered from these records without the NPZ files. Zero-residual
rows here also have zero logged Jacobian norms. The host-only
[analysis script](benchmarks/combined_diagnostics_events_2026-10-16/analyze_excerpt.py)
verifies reconstructed row norms, `J.T @ r`, costs, and available model
predictions against the saved data. It imports JSON/CSV/NumPy only and runs
no physics or finite differences.

Derived records:

- [Weighted objective gradients](benchmarks/combined_diagnostics_events_2026-10-16/objective_rows.csv).
- [Point sensitivities, correlations, and conditioning](benchmarks/combined_diagnostics_events_2026-10-16/point_analysis.csv).
- [Predicted and actual changes](benchmarks/combined_diagnostics_events_2026-10-16/step_predictions.csv).
- [Transport values at failed trials](benchmarks/combined_diagnostics_events_2026-10-16/failed_transport_values.csv).
- [Selected numerical analysis](benchmarks/combined_diagnostics_events_2026-10-16/analysis.json).

**The location objective is active and its gradient is nonzero.** The row
`transport:Er_transition_rho` is deliberately a zero-weight diagnostic;
`transport:Er_transition_location_moment` is the active weight-1000 row.
The zero diagnostic row must not be mistaken for an omitted root objective.

**A small power cost still creates a large gradient.** At accepted trial 28,
location cost is 0.906137 and power cost is only 0.000944268, but their
geometry-block cost-gradient norms are 0.766054 and 65.3964 respectively.
Bootstrap's corresponding norm is 32.9428. The root/power weighted Jacobian
row cosine is 0.973843: most location sensitivity lies along the very steep
power-sensitive direction. Projection of the location row onto directions
preserving power reduces its norm from 0.573467 to 0.130304; preserving all
other active rows reduces it to 0.100754. These are local linear quantities,
not feasibility guarantees for a large nonlinear step.

**Step contractions make meaningful root motion impossible at the observed
scale.** The parameter-step norm falls from `1.97816e-05` at trial 22 to
`4.93623e-06` and `1.23422e-06` at failed trials 23 and 24, then to
`3.08565e-07` at accepted trial 25. These are recorded parameter-step norms,
not direct measurements of SciPy's internal trust radius. They exhibit the
contractions expected after the poor model agreement and rejected reverse
failures. Trial 25 predicts a radius change of only `-6.17e-09`; its actual
change is about `-4.55e-09`. The root is not frozen, but changes at this scale
cannot appreciably reduce the remaining radius error of about 0.04257.

**Finite derivatives also show a severe shared spike.** The following norms
are the geometry block of each weighted cost-gradient contribution:

| Row | Trial 28 | Trial 29 | Trial 30 |
| --- | ---: | ---: | ---: |
| Location | 0.766054 | 3887.79 | 0.769934 |
| Bootstrap | 32.9428 | 62527.6 | 31.7596 |
| Net power | 65.3964 | 355818 | 69.3672 |

At trial 29 the root/power row cosine is 0.999999998978 and root/bootstrap is
0.999999864596. The largest singular value rises from about 1565 to
8.93225e+06 and returns to 1580 at trial 30; the other six active singular
values stay much smaller. The active seven-row Jacobian condition number
rises from about 1.55e+04 to 9.80e+07. The independent location sensitivity
after preserving the other rows stays near 0.09-0.11 even during this spike:
the large new component is overwhelmingly shared with the other transport
rows rather than providing a useful independent root direction.

Trial 29 was reached using the preceding, relatively regular Jacobian and
had good model agreement (ratio 0.97445). Its newly computed, spiky Jacobian
then drives trial 30. That model predicts a cost reduction of 0.00132997;
the actual cost increases by 0.0000892746. It predicts a weighted power
residual change of `-0.0482044`, whereas the actual change is `+0.00497337`.
This is direct evidence of poor local model reliability at the realized
step size, not an independent FD proof that the infinitesimal derivative is
incorrect. It is consistent with strong ill-conditioning or an unstable
shared full-trajectory reverse contribution. The precise producing
operation has not yet been localized inside the compiled segment.

**The observations do not support profiles consuming all the motion.** In
the available successful trials, roughly 90-99.99 percent of the squared
parameter-step norm is in ESS geometry coordinates. The final profile
coordinates are nominal relative deltas; their largest absolute change is
only 0.02205 percent from nominal. The final active-bound mask is zero in
every coordinate. Thus neither frozen geometry nor an active profile bound
explains this plateau in the recorded run.

**Bad reverse trials discard improving transport candidates.** Failed
trial 10 has location/bootstrap/power cost changes of `-2.35e-05`,
`-0.00666482`, and `-0.00195537` relative to accepted trial 8. All three
transport terms improve, but reverse failure still produces the adapter's
huge artificial penalty. Failed trials 23 and 24 also reduce the sum of
these three transport costs by about 0.000254152 and 0.0000658407 respectively,
although bootstrap alone slightly worsens there. Their total physical costs
are unknown because the remaining geometry objective values were not
reported before the reverse failure. These observations establish useful
transport progress being excluded, not that every failed candidate would
have improved total cost.

Initial and final states and terminal cotangents in the supplied failed
trials are finite. In trials 23/24, the helium density starts at zero but its
recorded temperature is finite (roughly 0.72-12.46 keV), and terminal helium
density is positive. This supports the previous initialization-floor fix
being active; it does not rule out a floor-related derivative issue at an
intermediate Radau stage. The first observed nonfinite output remains the
segmented transport state reverse, preceding initial-root and scan folds.

Together these records explain the observed step mechanism: a weak
independent root direction competes with much steeper, correlated transport
directions while failed reverse evaluations and unreliable models contract
the available steps. The outstanding source-level diagnosis is the shared
stage-adjoint matrix/solve and subsequent state/rebuild pullbacks, including
their finite-but-large amplification at trial 29. The 16-step benchmark
agreement does not establish the conditioning or accuracy of this much
longer warm-geometry trajectory. No objective weights, solver equations, or
physical derivatives were changed on the basis of this excerpt analysis.

### Why the nonzero profile gradient does not produce visible root motion

The profile-only calculation sharpens the explanation above: objective
competition is insufficient to explain the stall, and the available data do
not establish a stationary trade-off between root location and power.

The host-only analysis now also saves
[per-profile sensitivities](benchmarks/combined_diagnostics_events_2026-10-16/profile_sensitivities.csv),
[hypothetical profile directions](benchmarks/combined_diagnostics_events_2026-10-16/profile_direction_models.csv), and
[the actual profile-step projections](benchmarks/combined_diagnostics_events_2026-10-16/chosen_profile_steps.csv).
All numbers below use the stored weighted Jacobian at accepted trial 28,
before the shared derivative spike at trial 29. Profile coordinates are
nominal relative deltas, so a coordinate change of 0.01 means a one-percent
change in that profile parameter.

- The actual profile step from trial 28 to 29 has norm `7.42888e-07` and
  cosine `0.804944` with the negative root profile gradient. It is already
  directed toward improving the root. Its predicted contribution to radius
  change is only `-1.34387e-09`; geometry contributes `-3.14210e-08`.
  The remaining radius error is about `0.04257`.
- Even reallocating that entire joint step's norm (`2.47667e-06`) to the
  steepest root-improving profile direction would reduce radius by at most
  `5.56591e-09` in this linear model. A change of allocation alone cannot
  make this particular tiny step useful for the root.
- The radius derivative with respect to the relative `T0` coordinate is
  `0.00167190`. A one-percent decrease in `T0` therefore predicts a radius
  decrease of only `1.67190e-05` locally. Such a step has not been evaluated;
  it illustrates the small sensitivity, not a validated finite-step result.
- A feasible infinitesimal profile direction improving the root while
  preserving both power and bootstrap to first order exists. Its unit
  vector, in the six documented profile coordinates, is approximately
  `(0.264898, -0.571459, -0.698778, -0.222239, 0.199328, 0.160785)`.
  Multiplying it by `1e-04` predicts radius change `-6.43278e-08` and total
  cost decrease `2.73849e-06`, with negligible first-order power/bootstrap
  changes. These are saved-linear-model quantities, not a physics trial.
  This specifically rules out a necessary local conflict with these two
  objectives as the complete explanation.

The recorded failures and solver adapter explain an observed route to these
tiny steps. `optimization.least_squares._fun` requests a joint residual and
Jacobian evaluation. A non-resource exception during the reverse part of
that evaluation returns all-`1e6` residuals to SciPy, even when the forward
transport objectives were already finite and improving. SciPy consequently
rejects the trial and contracts its common trust region; there is no
independent profile trust region in this run. The logged step contractions
after failed trials 23 and 24 are consistent with that mechanism. Poor
predictions from the finite Jacobian spikes also cause rejections. The run
then reaches its 30-evaluation budget with profile changes below 0.023
percent, far too small to appreciably affect the location at the ordinary
saved sensitivities.

Thus the root profile gradient is present and is used by successful steps.
The unresolved problem is obtaining reliable full-trajectory derivatives
and useful accepted step sizes. The exact arithmetic operation responsible
for nonfinite and spiking reverse contributions remains unlocalized; the
excerpt analysis does not prove a global profile limitation or justify
calling the result a converged compromise. No physics solve or FD test was
run for this additional analysis.

## 5. Profile-only run and remaining temperature-floor cancellation (2026-10-16)

The new profile-only console excerpt is archived byte-for-byte as
[raw output](benchmarks/profiles_only_from_geometry_result_diagnostics_2026-10-16.txt),
with a [nine-evaluation table](benchmarks/profiles_only_from_geometry_result_diagnostics_2026-10-16.csv).
Its SHA256 is `e633c87a1a14ddbb741708fa12c57bc5bf3fa8e2f3fbb1627d2cc5a41a0809e2`.
The source attachment is `1ba95249-e402-41cc-a64f-00dbbdd63c20/Pasted text.txt`.
The excerpt ends after failed trial 9 and writing trial-10 inputs; it contains
no final solver result. The remote diagnostic directory is
`outputs/profiles_only_from_geometry_result_diagnostics/optimization_diagnostics/run_l_h39qze`.

The optimizer has six parameters and an `(11,6)` Jacobian, and its geometry
observables/costs stay fixed. Only trial 5 is marked accepted after the initial
point. Trial 6 returns a larger cost and is not accepted. Trials 2, 3, 4, 7, 8,
and 9 fail with all 5304 entries of the `(13,408)` transport state cotangent
nonfinite. Their first bad boundaries are segments 4, 4, 3, 4, 5, and 5
respectively. The failure therefore persists when geometry coordinates are
fixed, before the initial-root and final geometry pullbacks.

| Quantity | Initial | Accepted trial 5 |
| --- | ---: | ---: |
| Total cost | 3.58735782 | 1.73138136 |
| Root radius, target 0.514 | 0.556547106 | 0.556556881 |
| Location cost | 0.905128115 | 0.905544066 |
| Bootstrap cost | 0.431876241 | 0.446222432 |
| Net power, MW | 293.597137 | 299.999246 |
| Power cost | 1.87073863 | 2.59642e-08 |

Power-cost reduction is 1.87073860; location and bootstrap costs worsen by
0.000415951 and 0.014346191 (bootstrap +3.32 percent). This explains the total
cost reduction but does not establish a converged profile optimum. The six
failed derivatives exclude most attempted candidates in this partial run.
RSS rises from 6321.3 to 8003.9 MiB; the sampled GPU peak stays at 5879575296
bytes. The console lacks the profile coordinates and per-row Jacobians needed
to assess predicted versus actual changes for this particular run.

### Confirmed numerical bug still present after the earlier packed-floor fix

The historical fix is commit `dbb5c76`, "Stabilize transport temperature-floor
pullback". It replaced a divide/multiply temperature-floor calculation only
in `_transport_solvers._project_packed_transport_state_arrays`.
The general helper `_state.apply_transport_temperature_floor` still used
`safe_n * maximum(pressure / safe_n, T_floor)`.

This remaining helper is on the current database reverse path:

1. `_flat_rhs_direct_black_box_state_pullback_factory` calls the equation
   system's `pullback_direct_rhs_state`.
2. `ComposedEquationSystem.pullback_direct_rhs_state` accumulates direct and
   flux state bars and calls `_prepare_working_state_pullback`.
3. That method differentiates `_prepare_working_state`, which applies density
   flooring, the fixed-temperature projection, and the general temperature
   floor. The general floor is also called by the face-state/flux builders.

Above the temperature floor, the pressure map is mathematically the identity.
Its density derivative should therefore be zero. Differentiating the quotient
and product separately instead creates opposing density bars. Roundoff leaves
a spurious density contribution, and sufficiently large finite incoming bars
produce overflowing intermediate terms and `inf - inf`.

A small CPU-only reproducer using the original helper, at densities including
`1e-6` and temperature 12.5, observed:

- Unit incoming pressure bars: spurious density bars of order `1e-15`.
- Incoming pressure bars `1e150`: spurious density bars of order `1e135`.
- Incoming pressure bars `3e307`: NaN density bars in eager mode and infinite
  density bars under JIT, although the true density bars are zero and the true
  pressure bars are finite `3e307`.

The helper now uses `maximum(pressure, safe_n * T_floor)`, matching the earlier
packed projection fix. For positive floored density this is the same physical
floor, including the half derivative selected at equality. It removes the
cancelling quotient/product without clipping or replacing any cotangent.
Floating-point round trips in the old inactive branch can differ by an ulp;
bitwise equality of long adaptive trajectories is not claimed.

`tests/test_temperature_floor_pullback_stability.py` checks inactive-floor
identity derivatives through the large-bar range, scalar/species floors,
equality derivatives, the disabled-floor no-op, and the actual composed RHS
preparation transpose with a fixed helium temperature. The new inactive-floor
test failed on the previous implementation before the patch. The patched
floor/replay/diagnostics/lifecycle suite passes 31 small tests.

An additional existing state/equation-helper selection returned 24 passes and
two unrelated baseline failures: an obsolete assertion that the deliberately
no-op `apply_er_dirichlet_boundary_state` should clamp cell centers, and a
missing `jax` import in the floating-edge-node test. Those tests and the
boundary helper were not changed. The prior packed-floor regression and the
initial zero-density/fixed-temperature regression both pass in that selection.

This establishes a numerical defect in a live reverse path. It does **not**
yet prove that it is the only cause of the remote trial failures: the console
does not record the arithmetic operation or incoming-bar magnitude inside the
first failing compiled segment. The GPU reevaluation below now passes at saved
trial 2; the other failing points remain unverified. Segment-state checkpoints now include finite magnitudes per
objective, using the already-computed bars and no extra RHS/VJP evaluation.

### Reevaluate one saved failing point instead of rerunning optimization

The saved-result runner now accepts `--replay-diagnostics` and `--replay-trial`.
It reads the coordinates directly from `trial` or `failed_trial` events,
checks labels, physical scales, baseline profiles, and the coordinate mode,
and runs one fresh full objective/AD evaluation after ordinary problem setup.
There is no nominal objective evaluation, optimizer, plotting, or penalty
fallback. This is a fresh adaptive trajectory at the saved coordinates, not
a replay of the original recorded Radau stages. Source diagnostics are read
only; new inputs and diagnostics are written under the replay output.

After syncing this patch to the GPU machine:

```bash
cd ~/NEOPAX
PYTHONPATH="$HOME/VMEX:$HOME/NTX/src" \
python examples/optimization/optimize_geometry_profiles_from_geometry_result.py \
  --geometry-result-dir outputs/geometry_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_optimization \
  --profile-only \
  --replay-diagnostics outputs/profiles_only_from_geometry_result_diagnostics/optimization_diagnostics/run_l_h39qze \
  --replay-trial 2 \
  --out-dir outputs/profiles_only_reverse_replay_trial_0002
```

`replay_summary.json` records success or the original error, and the new
`replay_diagnostics/run_*/events.jsonl` records reverse checkpoints and the
small residual/Jacobian snapshot when available. Nonfinite returned arrays
are preserved separately, and exceptions propagate instead of becoming an
artificial optimizer penalty. No such GPU evaluation has been run locally.

### GPU reevaluation of saved failing trial 2 now succeeds

The user ran the one-trial command on the GPU machine after syncing the patch.
The complete console attachment is archived as
[trial-2 replay output](benchmarks/profiles_only_reverse_replay_trial_0002_2026-10-16.txt),
with a [parsed result and comparisons](benchmarks/profiles_only_reverse_replay_trial_0002_2026-10-16.csv).
The source attachment is `9d3201bc-814f-46e8-8abf-4a0b1ecb7d78/Pasted text.txt`,
6637 bytes, SHA256
`8d16f4f18717ef2a84b05394021878a7ab12375b3465b86426ffcef981e72280`.

The runner identifies `source_trial=2 previous_event=failed_trial`, uses six
profile coordinates with fixed saved geometry, and prints `optimizer=skipped`.
All six reverse segments complete with no first-bad checkpoint printed. The
initial-state, selected-root, profile, recorded-scan, and final VMEC pullbacks
all complete. The final line is `finite=True cost=1.71015613e+00`; this flag is
set only after checking every returned residual and Jacobian entry, without
the optimizer's penalty or nonfinite-replacement fallback.

| Quantity | Stage-2 starting point | Saved trial 2 after patch |
| --- | ---: | ---: |
| Total cost | 3.58735782 | 1.71015613 |
| Root radius, target 0.514 | 0.556547106 | 0.556540936 |
| Root location cost | 0.905128115 | 0.904865627 |
| Bootstrap penalty | 0.464691425 | 0.460275840 |
| Bootstrap cost | 0.431876241 | 0.423707698 |
| Net power, MW | 293.597137 | 299.792329 |
| Power cost | 1.87073863 | 0.00196796905 |

Compared with the starting point, this candidate improves all three active
transport penalties. Location cost falls 0.0290 percent and bootstrap cost
falls 1.8914 percent; total cost falls 52.3283 percent. It also has lower cost
than the old run's only accepted update shown (trial 5, cost 1.73138136), by
0.02122523. Thus the old reverse failure excluded coordinates that now produce
a useful physical candidate, including a small improvement in root location.
The root-radius change is still only `-6.17e-06`; one evaluation does not
establish meaningful root convergence or reliable full-trajectory derivatives
throughout an optimization.

This is positive GPU evidence for the remaining floor fix at this saved point.
It is not a replay of identical Radau stages: the fresh trajectory takes 268
accepted steps, versus 269 in the original failed trial. The algebraic change
can remove floating-point round trips and change the adaptive schedule. The
original first-bad arithmetic operation was not recorded, so this evidence
does not prove that every failure had this sole cause. The remaining failed
profile-only trials are 3, 4, 7, 8, and 9.

The new diagnostic directory is
`outputs/profiles_only_reverse_replay_trial_0002/replay_diagnostics/run_3zq4cnev`.
One memory sample reports RSS 10371.0 MiB, 690 live JAX arrays, 48481024 live
device bytes, and a device peak of 5882109696 bytes. A single evaluation cannot
establish iteration-to-iteration memory stability; that issue remains open.
The long first reverse-segment time (586.063 seconds) is followed by segments
of about 11-12 seconds. Its precise compilation/execution split is not reported.
Printed NaNs for disabled diagnostics such as the old left/right Er terms are
not nonfinite returned optimizer residuals; the finite check above covers the
actual residual and Jacobian arrays. No physics test was run locally to
archive or analyze this result.

### Restarted profile-only optimization still encounters nonfinite adjoints

The subsequent partial GPU run is archived as
[raw output](benchmarks/profiles_only_from_geometry_result_floor_fix_2026-10-16.txt)
and a [six-evaluation table](benchmarks/profiles_only_from_geometry_result_floor_fix_2026-10-16.csv).
Source attachment: `22b8a05a-a155-4a0d-bc44-6b0275e53624/Pasted text.txt`,
26893 bytes, SHA256
`c7533109a6ddd6d43aa204ac6de07ad58b4d74004d2e6eaea8bac65caaeaafbb`.
The console shows the update from `85527a5` to `4925ed5`; that update archives
the successful saved-point evaluation. Diagnostics are under
`outputs/profiles_only_from_geometry_result_floor_fix/optimization_diagnostics/run_q_i_mrcl`.
The excerpt ends after trial 6 and contains no final optimizer result.

| Trial | Outcome | Cost | Accepted transport steps | First failure |
| --- | --- | ---: | ---: | --- |
| 1 | Initial | 3.58735728 | 262 | None |
| 2 | Failed reverse | -- | 260 | Segment 5/6 state bars |
| 3 | Accepted | 1.72184763 | 272 | None |
| 4 | Failed forward guard | -- | 500 | Time 0.01926165, target 2.0 |
| 5 | Returned, not accepted | 455.079736 | 211 | None |
| 6 | Failed reverse | -- | 214 | Segment 3/5 state bars |

Both reverse failures have all 5304 entries of the `(13,408)` state cotangent
nonfinite at the first bad segment boundary, affecting all 13 objective rows.
The initial-root and recorded-scan failures occur downstream of that boundary.
This locates the failure within the transport reverse sweep, but does not yet
identify the first arithmetic operation or distinguish growing finite bars
from a local singular/nonfinite derivative.

Trial 3 achieves 299.955995 MW, while root radius moves from 0.556547106 to
0.556552745 and bootstrap cost grows from 0.431876249 to 0.436776359. Trial 5
does move the root substantially, to 0.595853596, with geometry fixed; its
399.374399 MW raises power cost to 450.62338 and it is rejected. Root dependence
on profiles is therefore present, but this partial run does not establish a
usable root-improving optimization trajectory.

Memory samples return to approximately 48.49 MB in live device arrays after
successful evaluations. Failed reverse evaluations retain approximately
450.3 MB at the sample boundary, and RSS rises from 6333.2 to about 9588 MiB.
The sampled device peak stays near 5.882 GB. These six evaluations do not
establish that the earlier memory-growth issue is resolved, nor show the old
CUDA-graph OOM failure recurring.

### Saved and newly proposed trial 2 have different profile coordinates

The user compared `replay_summary.json` from the passing saved-point evaluation
with the new run's `failed_trial` event. The result is
`Identical coordinates: False`. All six supplied coordinates and their
physical conversions are saved in the
[coordinate comparison](benchmarks/profiles_only_trial_0002_coordinate_comparison_2026-10-16.csv).
The profile map is `physical = baseline * (1 + delta)`.

| Parameter | Passing saved trial 2 | Failing new trial 2 |
| --- | ---: | ---: |
| Density shape power, delta | 0.13597657278247482 | 0.17562346680591331 |
| Density shape power, physical | 11.35976572782475 | 11.756234668059133 |
| Density shape alpha, delta | -0.28967563749252079 | -0.20675781451513051 |
| Density shape alpha, physical | 0.7103243625074792 | 0.7932421854848695 |

The ordinal trial number is not a shared parameter point. The passing saved
point takes 268 transport steps after the floor patch, while the newly proposed
failing point takes 260. Thus the new failure does not establish that the
previously passing coordinates now fail. Equally, different coordinates alone
do not prove that evaluation history is irrelevant: the new failed coordinates
have not yet been reevaluated in a fresh process.

The floor fix removes a confirmed cancellation defect and the saved-point
evaluation now passes. It has not resolved nonfinite AD throughout profile
optimization. Why the optimizer chooses a different first step is a separate
question. The initial cost changes only by about `5.4e-7`, but the initial
adaptive trajectory changes from 264 to 262 accepted steps. Comparing the
existing initial-point residual/Jacobian snapshots can measure the derivative
change without running transport:

- Old: `outputs/profiles_only_from_geometry_result_diagnostics/optimization_diagnostics/run_l_h39qze/trial_0001.npz`.
- New: `outputs/profiles_only_from_geometry_result_floor_fix/optimization_diagnostics/run_q_i_mrcl/trial_0001.npz`.

Before attributing the step change to derivatives, compare the snapshots' `x`,
parameter labels, residual labels and physical scales, plus the setup events'
bounds and trust scales. The existing per-segment summaries in the new
`events.jsonl` can also show whether finite cotangent magnitudes grow before
the failing boundary. Neither analysis requires a new physics evaluation.

### Starting Jacobians explain the changed optimizer proposal

The user compared the two `trial_0001.npz` files. Assertions confirm identical
starting coordinates, parameter labels, residual labels, and physical scales.
The transcribed active-row residuals and printed Jacobian entries are saved in
[the Jacobian comparison](benchmarks/profiles_only_initial_jacobian_comparison_2026-10-16.csv),
and all six reported singular values in
[the spectrum comparison](benchmarks/profiles_only_initial_jacobian_spectrum_2026-10-16.csv).
These files preserve the precision printed by the user; they are not copies of
the full-precision NPZ arrays.

| Active weighted residual row | Old residual | New residual | Relative change in Jacobian row, 2-norm | New cost-gradient norm |
| --- | ---: | ---: | ---: | ---: |
| Root location moment | 1.345457628517308 | 1.345457627189546 | 1.8520% | 0.1012507 |
| Bootstrap penalty | 0.929382849866232 | 0.9293828586068804 | 1.1875% | 3.2303542 |
| Net power | -1.934289856848892 | -1.9342895731232934 | 1.0027% | 821.3169044 |

Here the relative row change is `norm(J_new - J_old) / norm(J_old)`, and each
cost-gradient norm is `abs(residual) * norm(J_row)` in the six relative profile
coordinates. These are differences between two AD calculations, not FD error
estimates or proof of long-trajectory derivative accuracy. The power row's
density-shape entries change proportionally more: power derivative with respect
to relative density-shape power changes from -0.635516939 to -0.5099569165
(-19.76% in magnitude), and density-shape alpha from -0.05064997072 to
-0.08083370919 (+59.59% in magnitude). Both entries are small compared with
the power row's n0 and T0 entries.

The reported full `(11,6)` Jacobian has only three effective singular values:
about 424.6, 0.3160, and 0.01739 in the new run; the remaining values are at
`3.3e-14` or below. Geometry is fixed, the raw root/strength diagnostic rows
have zero weight, and the strength-deficit penalty is already satisfied. The
three active transport rows therefore constrain only three combinations of
six profile variables at first order. The largest-to-third singular-value
ratio is approximately 24413. This expected local rank deficiency is not
evidence of a broken AD calculation.

A bounded linear TRF calculation reproduces both actual optimizer proposals
using the printed active-row Jacobians, their residuals, the nominal profile
bounds from the source, unit trust scaling, and 11 residual rows with active
rows at positions 6, 9, and 10. Maximum absolute coordinate discrepancies are
`2.19e-9` for the old step and `3.11e-9` for the new step. The nontransport
constant costs and negligible inactive derivatives are omitted from this
small model. This establishes that the supplied changes in the starting
residual/Jacobian are sufficient to explain the different first steps; an
optimizer-setting change is not needed to reproduce them. It does not explain
the numerical origin of the derivative differences.

Both steps target almost the same linearized objective change:

| Model prediction using each run's own Jacobian | Old proposal | New proposal |
| --- | ---: | ---: |
| Root-radius change | -6.61325e-6 | -6.68047e-6 |
| Net-power change, MW | +6.36183473 | +6.36181342 |

Root-radius changes divide the weighted location residual change by
`sqrt(1000)`. Power changes multiply the weighted power residual change by
`331.0187969899648 / 100`. Approximately 97.47% of the squared difference
between the two coordinate steps lies in the null space of either three-row
linear model. Substantially different density shapes can therefore serve
nearly the same predicted improvement. The new initial power cost-gradient
norm is about 8112 times the root-location cost-gradient norm, and bootstrap
and power sensitivity rows have cosine 0.995899. These are local properties;
they do not establish behavior after power is matched or convergence of the
full nonlinear optimization.

The analysis is reproducible without importing JAX, NEOPAX, VMEC, or NTX:

```bash
python docs/benchmarks/analyze_profiles_only_initial_jacobians_2026-10-16.py --linear-trf
```

The tiny NumPy/SciPy analysis completed locally. It performs only matrix
operations and two bounded linear least-squares calculations with six
variables; no transport, geometry, or FD physics test was run. In this
reconstruction it matters to keep 11 residual rows: SciPy's dense trust-region
helper uses the original row count in its full-rank test. A three-row
underdetermined toy problem follows a different numerical decision and is
not used as the reconstruction above.

The remaining nonfinite transport state adjoint in new trials 2 and 6 remains
open. The small optimizer Jacobian's rank deficiency does not identify the
failure mechanism in the large per-step Radau adjoint solve. Existing segment
magnitude summaries can distinguish modest incoming bars from accumulation
toward overflow before deciding whether further one-point instrumentation is
needed.

### Moderate segment inputs precede the remaining reverse failures

The user extracted existing state-bar magnitude summaries from
`run_q_i_mrcl/events.jsonl` for failed trials 2 and 6. All eleven supplied
segment summaries are archived in
[the segment-state table](benchmarks/profiles_only_floor_fix_segment_state_bars_2026-10-16.csv).

| Trial | Last good reverse boundary | Minimum state bar | Maximum state bar | First bad boundary |
| --- | --- | ---: | ---: | --- |
| 2 | Segment 6 | -30.949954685288905 | 24.14319394675477 | Segment 5 |
| 6 | Segment 4 | -29.595924348132215 | 24.709024827054503 | Segment 3 |

Trial 6's earlier segment 5 is also finite, ranging from -35.68151748227225
to 24.56835936834783. At every bad boundary the finite min/max fields are
`null`, consistent with the previously reported all-5304-entry contamination.
All earlier reverse boundaries remain bad after the first failure.

The incoming state bars at the last good boundaries are modest. Already huge
state bars arriving from a later segment are therefore not the observed
starting condition. Each failing segment still contains 50 accepted steps;
the summaries do not exclude rapid amplification within those steps or a
single local nonfinite derivative/solve. They do not identify helium flooring
as the cause of these new failures.

The current example selects `reverse_stage_adjoint_solve_mode="block"`,
`reverse_stage_adjoint_memory_mode="default"`, and
`reverse_rhs_transpose_mode="explicit_database"`. In that lane the stage
matrix uses forward-mode RHS Jacobians. The subsequent state transpose also
uses forward-mode Jacobians; it does not use the compact direct-state VJP
selected by the explicit-database-Jacobian alternatives. The earlier general
floor fix changes live RHS preparation, but the current failing component
must be identified rather than assumed to be that previous VJP defect.

### Opt-in component checks for one saved failing point

`NEOPAX_REVERSE_STEP_DIAGNOSTICS=1` enables finite-input/nonfinite-output
checks in the existing objective-batched reverse step:

- Final-state projection transpose.
- Stage-adjoint RHS scaling.
- Dense stage-adjoint solve and its already-built matrix summary.
- RHS state pullback.
- Addition of the direct and RHS state bars.

Messages have prefix `[reverse-step-first-nonfinite]` and report the component,
transport time, timestep, incoming absolute maximum, surviving finite-output
absolute maximum and nonfinite count. The current dense block lane also
reports matrix finiteness, nonfinite count/first row and column, finite
absolute maximum, and minimum absolute diagonal. A finite matrix with a bad
solve output is distinguished from an already nonfinite matrix; the summary
does not compute or claim a condition number or identify a scalar operation
inside matrix assembly.

The scalar print condition is outside objective vmaps to avoid their batching
rule executing both print branches. Already nonfinite inputs do not generate
more transition messages. The stage solve can produce both its detailed
matrix message and the enclosing step-component message at the same time.
No full matrix, per-step history, or extra RHS, Jacobian, factorization, VJP,
or frozen-adjoint comparison is recorded or evaluated. The flag adds scalar
checks and debug effects and can change compilation; it is not an identical
GPU executable. It must be set before starting a fresh process.

After writing the first bad segment checkpoint, an instrumented run raises
immediately rather than propagating its bad state bars through the remaining
segments, initial-root pullback and recorded-scan fold. With the environment
flag unset, the existing mathematics, solve layout and failure policy remain
unchanged. This is diagnostic instrumentation, not a fix for the remaining AD
failure.

After syncing the patch to the GPU machine, reevaluate the **new** failed trial
2, rather than the old trial 2 that already passed:

```bash
cd ~/NEOPAX
set -o pipefail
NEOPAX_REVERSE_STEP_DIAGNOSTICS=1 \
PYTHONPATH="$HOME/VMEX:$HOME/NTX/src" \
python examples/optimization/optimize_geometry_profiles_from_geometry_result.py \
  --geometry-result-dir outputs/geometry_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_optimization \
  --profile-only \
  --replay-diagnostics outputs/profiles_only_from_geometry_result_floor_fix/optimization_diagnostics/run_q_i_mrcl \
  --replay-trial 2 \
  --out-dir outputs/profiles_only_floor_fix_trial_0002_step_diagnostics \
  2>&1 | tee outputs/profiles_only_floor_fix_trial_0002_step_diagnostics.log
```

This is one fresh adaptive forward/reverse evaluation at the stored six
coordinates, with fixed saved geometry and no optimizer or nominal objective
evaluation. A fresh passing result would be evidence to investigate evaluation
history or compilation/trajectory sensitivity, rather than a reproduced
failure. The GPU result below now reproduces the failure at these new saved
coordinates in a fresh process.

Local validation used only small arrays, synthetic four-state analytic RHS
functions and mocked optimization/replay setup: 32 unittest checks and 20
existing analytic Radau checks pass. The new checks cover a finite-to-NaN
scan transition emitting once, finite-operand overflow, finite/singular/
nonfinite 2x2 stage matrices with one matrix assembly, and the opt-in host
segment guard. No expensive transport, VMEC, NTX or FD test was run locally.

### GPU replay locates an already nonfinite stage matrix

The user ran the instrumented single-point replay on the GPU machine. Its
complete console output is archived as
[raw replay output](benchmarks/profiles_only_floor_fix_trial_0002_step_diagnostics_2026-10-16.txt),
with the [first-failure and memory fields](benchmarks/profiles_only_floor_fix_trial_0002_step_diagnostics_2026-10-16.csv).
The source attachment is `f309d43d-8d05-4a19-be89-06a52e64248e/Pasted text.txt`,
8520 bytes, SHA256
`0ec772b3d506f239a7c0fca93a9579dae3f3c5bdf81152cfc54be50be610d50a`.
New diagnostics are in
`outputs/profiles_only_floor_fix_trial_0002_step_diagnostics/replay_diagnostics/run_a2n6soqo`.

The source is the new failed trial 2 in `run_q_i_mrcl`, not the old saved trial
2 that passed. Its local replay trial number is 1 because the optimizer and
nominal objective evaluation are skipped. Segment 6 completes with ten active
steps; the first bad segment is again segment 5.

| First failing dense stage-adjoint solve | Value |
| --- | ---: |
| Transport time | 0.23283061626736629 |
| Timestep | 0.030139674511190269 |
| Maximum absolute incoming stage-adjoint RHS | 1.7897597282504262e-6 |
| Stage matrix finite | False |
| Nonfinite matrix entries | 8156736 |
| First nonfinite matrix row/column | 0 / 0 |
| Nonfinite stage-adjoint solution entries | 37128 |

The seven-stage, 408-state system has dimension `7*408 = 2856`, and
`2856**2 = 8156736`: every matrix entry is nonfinite before the solve.
All `13*2856 = 37128` solution entries then become nonfinite. The reported
`matrix_finite_abs_max=0` is the reduction's default for no surviving finite
entries; it does not mean the matrix is zero. Its diagonal minimum is NaN.

This establishes a local nonfinite linearization before the dense solve, with
a small finite RHS. Overflow caused by large incoming adjoints is not the
observed cause at this step, and a merely ill-conditioned finite matrix does
not describe the captured matrix. All objective rows fail together because
they share it. This result does not identify the first bad scalar operation
or establish that helium flooring caused the nonfinite matrix.

The matrix builder computes stage states as
`Y_i = y_start + h * sum_j A_ij * z_j`, obtains forward-mode RHS Jacobians at
those states, and assembles `M_ij = delta_ij*I - h*A_ij*J_i`. The console does
not contain the replayed `y_start`, `z_j`, stage states or primal RHS values.
It therefore cannot yet distinguish invalid replayed primal stages from
nonfinite RHS values/derivatives at finite stage inputs. The source's minimal
accepted-step replay strips the ordinary convergence/error diagnostics and
advances its prescribed accepted schedule, so replay validity must be checked
rather than inferred solely from a successful adaptive baseline.

This fresh-process reproduction shows that an earlier optimizer evaluation is
not necessary for the failure at these coordinates with instrumentation. It
does not prove bitwise equality of the original and instrumented trajectories.
The first reverse segment takes 402.953 seconds, including a reported 259.150
second GPU compile; this is a single-point diagnostic, not an inexpensive
runtime benchmark. The one memory sample (RSS 9993.3 MiB, live device bytes
2377989376, peak 5321471488) cannot establish iteration-to-iteration growth.

### Capture primal stage values from the same differentiation call

The opt-in dense-block diagnostic now also reports finiteness of the step
input, trial output, each stage's slopes/states, primal RHS and RHS Jacobian.
The primal RHS is retained with `jax.jacfwd(..., has_aux=True)` from the same
RHS call that builds the Jacobian. No separate RHS evaluation is added. Other
matrix lanes report `stage_rhs_captured=False` where no primal RHS was retained.
Available Newton iteration counts and cached-Jacobian finiteness are included.

Setting `NEOPAX_REVERSE_STEP_SNAPSHOT_DIR` additionally saves a compressed NPZ
only when a finite stage-adjoint RHS produces a nonfinite solve output. The
file contains the step input/trial state, stage times/slopes/states, captured
primal RHS, first objective's incoming stage bar, finite flags and available
Newton metadata. It contains no full stage matrix or per-iteration history.
The callback writes and releases these small host arrays; no device history is
retained. Existing diagnostics and solve arithmetic remain unchanged when the
diagnostic flag is unset. Diagnostic compilation can differ, as before.

After syncing this extension, one saved-point GPU replay captures the missing
data without rerunning optimization:

```bash
cd ~/NEOPAX
set -o pipefail
NEOPAX_REVERSE_STEP_DIAGNOSTICS=1 \
NEOPAX_REVERSE_STEP_SNAPSHOT_DIR=outputs/profiles_only_floor_fix_trial_0002_stage_inputs/bad_steps \
PYTHONPATH="$HOME/VMEX:$HOME/NTX/src" \
python examples/optimization/optimize_geometry_profiles_from_geometry_result.py \
  --geometry-result-dir outputs/geometry_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_optimization \
  --profile-only \
  --replay-diagnostics outputs/profiles_only_from_geometry_result_floor_fix/optimization_diagnostics/run_q_i_mrcl \
  --replay-trial 2 \
  --out-dir outputs/profiles_only_floor_fix_trial_0002_stage_inputs \
  2>&1 | tee outputs/profiles_only_floor_fix_trial_0002_stage_inputs.log
```

The first-failure line now distinguishes nonfinite primal values from bad
derivatives of a finite RHS. `[reverse-step-snapshot] wrote ...npz` identifies
the saved step. This extends diagnostic evidence; it is not a fix for the
underlying physics/replay derivative failure. No new GPU result is claimed for
this extension yet.

Validation: 34 small unittest checks and 20 existing four-state analytic Radau
checks pass. The new cases explicitly distinguish a finite primal RHS with a
deliberately nonfinite JVP from nonfinite replay stages, check that capturing
primal RHS values makes no second RHS call, and verify a snapshot is written
only at a finite-input/nonfinite-output transition. No expensive physics test
was run locally.

### Latest GPU evidence: the reconstructed primal state is already nonfinite

The next instrumented saved-point run is archived as
[complete output](benchmarks/profiles_only_floor_fix_trial_0002_stage_inputs_2026-10-16.txt)
and [extracted fields](benchmarks/profiles_only_floor_fix_trial_0002_stage_inputs_2026-10-16.csv).
Source attachment: `6fb7591e-3efd-4bed-a20d-e3586b439872/Pasted text.txt`,
9358 bytes, SHA256
`f5b3692c0f4393f20e5d64f8231a0dcc735a4908207ae964b196dfad4a929162`.
The new run is
`outputs/profiles_only_floor_fix_trial_0002_stage_inputs/replay_diagnostics/run_mhm7s1k1`.

The failure repeats in segment 5 at `t=0.23283061626736629`,
`dt=0.030139674511190269`. The incoming adjoint RHS remains finite with
maximum magnitude `1.7897597282504262e-6`. Crucially, the captured step input
and trial state are already nonfinite. All seven stage slope/state/RHS/
Jacobian blocks contain nonfinite values, as does the cached Jacobian. The
reported Newton iteration count is 1, and every entry of the 2856-square
stage-adjoint matrix is nonfinite.

This distinguishes invalid reconstructed primal data from a derivative
failure at finite primal inputs. The reported reverse step is not the first
primal failure: its input is already bad, so an earlier reconstruction inside
that segment must be examined. The original adaptive forward result is
finite; its accepted trajectory and its reverse reconstruction should agree.
The exact first divergence in this GPU run is not yet identified.

The remote snapshot is
`outputs/profiles_only_floor_fix_trial_0002_stage_inputs/bad_steps/bad_reverse_step_brf2omsa.npz`.
Its contents have not been supplied locally. One memory sample (RSS 10028.8
MiB, device live 2377989632 bytes and peak 5321470976 bytes) cannot establish
iteration-to-iteration memory growth.

### Verified discrepancy in accepted-step Newton cache reconstruction

Source inspection found that the configured `radau_jacobian_reuse_mode="legacy"`
is an alias for `dt_close`. A checkpoint restores the complete carry at its
segment start, but subsequent accepted-only replay slots recompute Jacobian
reuse decisions from the replay's evolving cache. Those decisions were not
recorded from the forward solve.

A rejected forward attempt can refresh a Jacobian without changing physical
`y` or `t`. A later accepted timestep can be close to the previous accepted
timestep but far from that rejected attempt's timestep. The forward accepted
step then rebuilds its Jacobian, while accepted-only replay can reuse an older
Jacobian. Thus equal physical state and timestep did not imply equal Newton
setup in this implementation. This does not require differentiating or
replaying rejected attempts to correct the accepted-step reconstruction.

A tiny scalar ODE probe demonstrates this discrepancy with actual Radau
attempts. It prescribes a small accepted step, a large rejected attempt and
another small accepted step. For `y'=-100*y**3`, the compared final accepted
step starts at the same `y=0.91287092`, time and timestep in both lanes:

| Lane | Jacobian used | Newton iterations | Endpoint difference from forward |
| --- | ---: | ---: | ---: |
| Forward accepted step | -249.99999339 | 6 | 0 |
| Existing accepted-only replay | -300 | 7 | 3.77475828e-15 |
| Replay with recorded forward rebuild decisions | -249.99999339 | 6 | 0 |

Both original scalar solves converge and stay finite. This establishes a
numerical-setup discrepancy, not a reproduction of the GPU NaNs. The scalar
probe uses caller-prescribed attempt sizes to isolate the cache behavior;
it is not a full adaptive transport validation. Its
[script](benchmarks/probe_accepted_replay_newton_cache_2026-10-16.py) and
[output](benchmarks/accepted_replay_newton_cache_probe_2026-10-16.txt) are saved.

The optimization correction records one boolean Jacobian-reuse decision per
forward attempt alongside the existing scalar schedule. Only the accepted
positions are used during reconstruction. A recorded forward rebuild
invalidates the replay cache for that slot, ensuring Newton rebuilds there.
Rejected attempts are not reconstructed or differentiated. No per-step
Jacobian/LU matrices, stage solutions or full carries are added to the
forward tape. The boolean mask is a numerical argument to the persistent
optimization replay executable, so changing its values does not create new
static JIT identities per iteration. Existing benchmark replay defaults keep
their established numerical path.

Separately, the minimal replay previously discarded its existing Newton
convergence flag and advanced even an unsuccessful reconstruction. Checked
optimization replay now retains that scalar flag and residual norm, freezes
at the first unconverged/nonfinite reconstructed step, and stops before
differentiating that segment. An explicitly enabled snapshot captures its
state, stages, previous stages, cached Jacobian and Newton/cache metadata.
It also checks each reconstructed segment's endpoint against the next
adaptive checkpoint (or the adaptive final state), using the configured state
tolerances. Agreement at endpoints does not prove every internal stage is
bitwise identical. No automatic fresh-cache retry or timestep adjustment is
performed to hide a failed reconstruction.

Validation: 36 small scalar, four-state analytic and mocked replay checks,
plus 6 subtests, pass. They include the actual accepted/rejected/accepted
cache scenario, preservation of healthy minimal replay values, retention of
the scalar forward decisions, rejection of a finite but unconverged solve,
stopping later slots, snapshot creation and endpoint mismatch detection.
No expensive transport, VMEC, NTX or FD calculation was run locally.

**Status:** the cache discrepancy is corrected in the optimization path and
the scalar comparison passes. The GPU verification recorded below now also
passes the previously failing saved trial. The partial stage-2 run below also
shows stable sampled GPU memory and no failures through 19 completed
evaluations; accepted root progress remains limited. A final optimization
result is not present in the supplied excerpt.

The existing saved-point replay can verify the correction without rerunning
optimization. Preserve the previous evidence by using a new output directory:

```bash
cd ~/NEOPAX
set -o pipefail
NEOPAX_REVERSE_STEP_DIAGNOSTICS=1 \
NEOPAX_REVERSE_STEP_SNAPSHOT_DIR=outputs/profiles_only_trial_0002_checked_replay/bad_steps \
PYTHONPATH="$HOME/VMEX:$HOME/NTX/src" \
python examples/optimization/optimize_geometry_profiles_from_geometry_result.py \
  --geometry-result-dir outputs/geometry_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_optimization \
  --profile-only \
  --replay-diagnostics outputs/profiles_only_from_geometry_result_floor_fix/optimization_diagnostics/run_q_i_mrcl \
  --replay-trial 2 \
  --out-dir outputs/profiles_only_trial_0002_checked_replay \
  2>&1 | tee outputs/profiles_only_trial_0002_checked_replay.log
```

### GPU verification: the previously failing saved trial now passes

The user ran the checked replay command above. The complete output is saved
as [raw output](benchmarks/profiles_only_trial_0002_checked_replay_2026-10-16.txt),
with an [evaluation summary](benchmarks/profiles_only_trial_0002_checked_replay_2026-10-16.csv)
and [per-segment comparisons](benchmarks/profiles_only_trial_0002_checked_replay_2026-10-16_segments.csv).
Source attachment: `0f16712d-848f-4f31-afc9-ff575de21141/Pasted text.txt`,
7524 bytes, SHA256
`6c9d2002b71bb1eb4f13c49dfc4a52902925a1d75317ed4fe042129471e849ec`.
The local checkout at archival time is commit
`e83a143c908248d39541331e69f6ec96e3452cc5`,
`Preserve forward Newton rebuild decisions in accepted-step replay`.
The remote log shows an up-to-date pull but does not print its HEAD hash.
New diagnostics are in
`outputs/profiles_only_trial_0002_checked_replay/replay_diagnostics/run_3953akd7`.

This evaluates the same new failed trial 2 from `run_q_i_mrcl`, with six
profile coordinates, fixed saved geometry and no optimizer. All 260 accepted
steps reconstruct successfully. Every one of the six segment endpoints has
zero maximum absolute difference and zero scaled difference from its adaptive
forward checkpoint, including segment 5 where the previous run failed.
The segment sizes are 50, 50, 50, 50, 50 and 10 accepted steps. Endpoint
agreement does not separately compare every internal stage with the forward
stage values, which are not retained.

The full reverse sweep, initial selected-root boundary, profile pullback,
recorded database scan fold and geometry derivative calculations complete.
The runner prints `finite=True` only after checking that every entry of the
returned residual vector and Jacobian is finite. No failed-trial penalty
fallback is used by this replay mode.

| Returned metric at the saved trial | Value |
| --- | ---: |
| Total cost | 1.70831691 |
| Transition radius | 0.556541236 |
| Transition location cost | 0.904878400 |
| Transition strength | 1.75667244 |
| Bootstrap penalty | 0.459775064 |
| Bootstrap cost | 0.422786219 |
| Net power | 299.849217 MW |
| Net-power cost | 0.00103745104 |

The printed NaNs for unused `softmax_Er` and left/right Er report fields are
not entries of the active residual/Jacobian checked above. The transition
location and strength functions are the active root objectives here.

This is GPU evidence that the correction resolves the reproduced nonfinite
replay failure at these coordinates. It does not establish that every prior
nonfinite trial is fixed. The root/bootstrap optimization question also remains:
this single proposed point moves root radius by only about `-5.87e-6` from the
starting radius 0.556547106, lowers bootstrap cost by about 2.10%, and brings
net power close to its 300 MW target. It is not an optimization trajectory.

After the evaluation, live device allocation is 48497408 bytes (46.25 MiB),
with a peak of 5879726592 bytes (5.48 GiB), 692 live JAX arrays and host RSS
10491.5 MiB. Runtime-scan, replay, reverse and bootstrap stage caches each
have one entry; the persistent schedule-probe cache has zero entries. These
are one-evaluation measurements, not evidence of a flat memory trend across
iterations. The reverse sweep takes 703.421 seconds; its first processed
segment takes 644.283 seconds including startup/compilation effects, whereas
the remaining 50-step segments take 11.706-11.892 seconds each.

The next comparison is the profile-only stage-2 optimization from the same
saved geometry, with a new output directory and the existing optimization/
memory diagnostics enabled by default:

```bash
cd ~/NEOPAX
set -o pipefail
NEOPAX_REVERSE_STEP_DIAGNOSTICS=1 \
NEOPAX_REVERSE_STEP_SNAPSHOT_DIR=outputs/profiles_only_from_geometry_result_replay_fix/bad_steps \
PYTHONPATH="$HOME/VMEX:$HOME/NTX/src" \
python examples/optimization/optimize_geometry_profiles_from_geometry_result.py \
  --geometry-result-dir outputs/geometry_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_optimization \
  --profile-only \
  --max-nfev 30 \
  --out-dir outputs/profiles_only_from_geometry_result_replay_fix \
  2>&1 | tee outputs/profiles_only_from_geometry_result_replay_fix.log
```

This starts profiles at the stage-2 initial point defined by the saved geometry
result and current runner. It does not initialize the optimizer from the
single replayed trial. The purpose is to assess accepted updates, root/
bootstrap progress, remaining failures and live-memory/cache trends together.

### Partial profile-only stage 2: stable memory, finite AD, limited accepted root progress

The next attachment contains the profile-only optimization after the replay
correction, with `--max-nfev 30`. It is archived as
[raw output](benchmarks/profiles_only_from_geometry_result_replay_fix_partial_2026-10-16.txt),
[evaluation/objective table](benchmarks/profiles_only_from_geometry_result_replay_fix_partial_2026-10-16.csv),
[memory samples](benchmarks/profiles_only_from_geometry_result_replay_fix_partial_2026-10-16_memory.csv)
and [segment comparisons](benchmarks/profiles_only_from_geometry_result_replay_fix_partial_2026-10-16_segments.csv).
Source attachment: `57b38d68-7b26-4a1c-bdb1-cab9fd8029b2/Pasted text.txt`,
110714 bytes, SHA256
`afc8f3c1c3373fb77ae9cf4dfd8e74e25e227b0803ebb8419ea4bb1057e899be`.
Diagnostics are in
`outputs/profiles_only_from_geometry_result_replay_fix/optimization_diagnostics/run_s2w3mheh`.

**This is a partial run.** There are 19 completed evaluations: the initial
point, seven accepted updates (2, 6, 9, 12, 13, 16, 18), and eleven rejected
returned trials. The excerpt ends during evaluation 20, after its reverse
transport sweep and initial-root/profile pullbacks, before the final database
scan fold and cost report. There is no terminal optimizer status or final
optimized profile report. A rejected trial here returned residuals and a
Jacobian; it did not fail reverse AD.

All 110 reported segment endpoint comparisons have zero absolute and scaled
differences: 105 belong to the 19 completed evaluations and five to the
in-progress evaluation 20. The completed trajectories use 217-278 accepted
transport steps. No nonfinite-AD exception, invalid primal replay, checkpoint
mismatch, transport step-limit failure or CUDA OOM is reported in this excerpt.

#### Memory after warmup

| Sample | Host RSS (MiB) | Live JAX arrays | Live device bytes | Peak device bytes |
| --- | ---: | ---: | ---: | ---: |
| Evaluation 1 | 10232.2 | 691 | 48844288 | 5879747584 |
| Evaluation 2 | 10294.8 | 720 | 48848384 | 5909524480 |
| Evaluation 3 | 10296.1 | 746 | 48852224 | 5909524480 |
| Evaluation 19 | 10308.0 | 746 | 48852736 | 5909524480 |

Across evaluations 3-19, live GPU allocation stays between 48852224 and
48852992 bytes: 46.5891-46.5898 MiB, a 768-byte range. The device peak stays
at 5.50 GiB from evaluation 2 onward. Live JAX array count stays at 746 from
evaluation 3. All five reported stage caches stay at one entry after the
persistent schedule probe is first used in evaluation 2.

Host RSS fluctuates between 10296.1 and 10308.0 MiB over evaluations 3-19,
including decreases; it is 75.8 MiB higher than the first sample, with most
of that rise in the first two evaluations. The earlier multi-gigabyte host
growth and CUDA-graph OOM pattern are not reproduced in this observed interval.
This does not establish a longer-term bound beyond the supplied samples.

#### Why root-improving trials are rejected

| Evaluation | Status | Root radius (target 0.514) | Location cost | Bootstrap cost | Net power (MW) | Power cost | Total cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Initial | 0.556547106 | 0.905128113 | 0.431876249 | 293.597138 | 1.87073808 | 3.58735728 |
| 2 | Accepted | 0.556541245 | 0.904878742 | 0.422797469 | 299.848913 | 0.001041643 | 1.70833269 |
| 6 | Accepted | 0.556532140 | 0.904491456 | 0.422073708 | 300.055541 | 0.000140767 | 1.70632077 |
| 7 | Rejected | 0.517263386 | 0.005324846 | 0.267194998 | 250.417504 | 112.181557 | 112.833691 |
| 17 | Rejected | 0.536908480 | 0.262399231 | 0.337762235 | 275.651730 | 27.0520952 | 28.0318715 |
| 18 | Accepted | 0.556530235 | 0.904410441 | 0.421609526 | 299.974662 | 0.000029296 | 1.70566410 |
| 19 | Rejected | 0.536909026 | 0.262411733 | 0.338176914 | 275.818056 | 26.6837657 | 27.6639691 |

The root objective responds substantially to profile changes. Trial 7 nearly
reaches its target and also lowers bootstrap cost, but its power cost alone
is 112.18, compared with total cost 1.7063 at its preceding accepted trial 6.
The later proposals near root radius 0.5369 are also rejected because their
power cost is roughly 27-36, outweighing their location/bootstrap improvements.
These returned points demonstrate physical objective changes; they do not
independently validate the local Jacobian entries.

In the accepted sequence, total cost falls by 52.45%, mainly by satisfying
power. Between initial evaluation 1 and accepted evaluation 18, root radius
changes by only `-1.6871e-5`, location cost falls by 0.0793%, and bootstrap
cost falls by 2.377%. This confirms that accepted root progress remains small
even after the reproduced AD/replay failure has been removed in this excerpt.

The late trial points form two recurring groups. Accepted evaluations 13, 16
and 18 remain near radius 0.556530 and power 300 MW, while rejected evaluations
14, 15, 17 and 19 remain near radius 0.53691 and power 275-276 MW. The accepted
root moves by only `-8.4e-8` between evaluations 13 and 18. The radius gap
between evaluations 18 and 19 is 0.019621209, and their power gap is 24.156606
MW. These values show repeated proposals into the second region, not two
demonstrated stationary minima or accepted cycling between them.

If recorded parameter-step norms shrink toward zero while this finite jump
persists, a branch boundary, sharp transition or numerical inconsistency is
a stronger concern than ordinary overshooting. That conclusion requires the
coordinate/model records; the console alone does not establish shrinking
parameter steps or a discontinuity. The least-squares wrapper keys cached
evaluations by the exact coordinate tuple, keeps only one successful point,
and uses that same-point evaluation for residual/Jacobian requests. The
profile-only wrapper copies its fixed full vector before inserting profile
coordinates. This source inspection finds no obvious cache-key collision or
in-place mutation there; it is not a same-coordinate repeatability test of
the entire transport evaluator.

The console proves why these particular trials fail the total-cost acceptance
test. It does not show whether the saved local model predicted their power
loss, whether bounds prevented compensation, or whether a sufficiently small
feasible step can improve root/bootstrap while retaining power. It does not
establish that the three physical targets are incompatible. Those questions
require the already-saved `model` and bound records, not another transport or
optimization run.

The following host-only command reads those existing records for trials 7,
17 and 19 and their accepted anchors. It imports only the Python standard library
and can run while optimization continues. An unfinished final JSONL line is
ignored; completed malformed records are reported as an error.

```bash
cd ~/NEOPAX
python - <<'PY'
import json
from pathlib import Path

path = Path('outputs/profiles_only_from_geometry_result_replay_fix/'
            'optimization_diagnostics/run_s2w3mheh/events.jsonl')
labels = []
with path.open() as stream:
    for line in stream:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            if not line.endswith('\n'):
                break
            raise
        trial = event.get('trial')
        if event['event'] == 'setup':
            labels = event['parameter_labels']
        if event['event'] == 'accepted' and trial in (6, 16, 18):
            print('Accepted anchor', trial, 'near lower bounds:',
                  [labels[i] for i in event['lower_bound_near']], 'near upper bounds:',
                  [labels[i] for i in event['upper_bound_near']])
        if event['event'] != 'trial' or trial not in (7, 17, 19):
            continue
        model = event['model']
        print('\nTrial', trial, 'from accepted', model['from_accepted_trial'])
        print('Predicted / actual total reduction:', model['predicted_reduction'],
              model['actual_reduction'], 'ratio:', model['actual_over_predicted'])
        print('Coordinate step:', dict(zip(labels, model['step'], strict=True)))
        print('Profile / trust step norm:', model['profile_step_norm'], model['trust_step_norm'])
        actual = {row['label']: row['cost'] for row in event['rows']}
        for i, row in enumerate(model['rows']):
            label = row['label']
            if any(key in label for key in ('transition_location', 'bootstrap', 'net_total_power')):
                print(label, 'old / predicted / actual cost:', model['old_cost'][i],
                      model['predicted_cost'][i], actual[label])
PY
```

Only file parsing, table extraction and consistency checks were performed
locally for this update. No physics solve, derivative replay or FD test was run.

### Completed profile-only stage 2: plotting failure and late host-memory growth

The completion tail is archived as
[raw output](benchmarks/profiles_only_from_geometry_result_replay_fix_completion_2026-10-16.txt),
[objective table](benchmarks/profiles_only_from_geometry_result_replay_fix_completion_2026-10-16.csv),
[memory samples](benchmarks/profiles_only_from_geometry_result_replay_fix_completion_2026-10-16_memory.csv)
and [segment comparisons](benchmarks/profiles_only_from_geometry_result_replay_fix_completion_2026-10-16_segments.csv).
Source attachment: `0ef15958-2731-4ddb-af33-203d4c3c5f7f/Pasted text.txt`,
8996 bytes, SHA256
`99bee4ebffcc709fc1888e4fe4cf15d5cb71e76b432601da5ece06bf59d43e96`.
This tail contains completed evaluations 29 and 30; evaluations 20-28 are
not supplied as completed console records. Its five replay endpoint checks
belong to evaluation 30, with 227 accepted transport steps; all five report
zero absolute and scaled differences.

The optimizer returned after 30 function evaluations with `ftol` termination,
initial cost 3.58735728, final cost 1.70563781 and first-order optimality 2.90.
The root radius is still 0.556530127 against target 0.514, versus initial
0.556547106: a change of only -0.000016979. Final location cost is 0.904405851,
bootstrap cost 0.421589492 and power cost 0.0000276351994, at 299.975391 MW.
Small total-cost changes triggered termination; this does not demonstrate
that the root target was reached or the first-order optimality was small.
The missing coordinate/model records discussed above are still needed to
explain the repeated unsuccessful root-improving proposals.

| Sample | Host RSS (MiB) | Live JAX arrays | Live device bytes | Peak device bytes |
| --- | ---: | ---: | ---: | ---: |
| Evaluation 19 (earlier excerpt) | 10308.0 | 746 | 48852736 | 5909524480 |
| Evaluation 29 | 10760.9 | 746 | 48852224 | 5909524480 |
| Evaluation 30 | 10912.7 | 746 | 48852224 | 5909524480 |

Host RSS rises 151.8 MiB between evaluations 29 and 30 and 680.5 MiB from
evaluation 1 to 30. Live device bytes, device peak, live array count and all
five reported one-entry stage caches are unchanged between 29 and 30.
**Late host-memory growth remains open.** The earlier stable-memory finding
applies only to evaluations 3-19; these counters do not identify the later
host allocation source or establish that all memory growth was fixed.

The subsequent traceback occurs in post-optimization plotting:
`initial_profile_snapshot` reads `problem.profile_cfg`, but the actual
`GeometryFullTransportLeastSquaresProblem` exposes profiles through
`problem.config["profiles"]`. The plot helper now reads
`problem.config.get("profiles", {})`, consistent with the problem interface.
The analytical-profile regression fixture no longer adds the nonexistent
attribute that had hidden this failure. The saved-result plotting fixture
also uses configuration alone. All 10 tests in
`tests/test_combined_from_geometry_result.py` pass; these are mocked setup
checks and small analytical profile plots, with no transport/VMEC/NTX solve.

Before calling the plot helper, the runner already writes
`input.QI_combined_optimized`, `profile_parameters_optimized.json` and
`optimization_summary.json`. Therefore the optimized result survives this
plotting exception. After updating the script, recover only the plots from
the existing output; no optimization or derivative replay is needed:

```bash
cd ~/NEOPAX
JAX_PLATFORMS=cpu \
PYTHONPATH="$HOME/VMEX:$HOME/NTX/src" \
python examples/optimization/optimize_geometry_profiles_from_geometry_result.py \
  --geometry-result-dir outputs/geometry_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_optimization \
  --profile-only \
  --plot-profiles-only \
  --out-dir outputs/profiles_only_from_geometry_result_replay_fix
```

This produces `initial_profiles_initial_vs_optimized.png` and
`initial_optimized_initial_profiles.csv` from the saved physical profile
parameters. The CPU work is only analytical profile construction and plotting.

## 6. Original geometry-stage reproducibility report

The user also reported that phase 1 of the original two-stage script did not
match the standalone geometry-only optimization result. Mocked setup checks
found matching configuration, objective definitions, builder arguments,
starting coordinates, and least-squares options. That is a source/plumbing
check, not numerical trajectory parity. This original report remains open.

### Geometry-only optimization after the replay correction (2026-10-17)

Subsequent steering: the user reports that the same script invoked without
arguments now produces the intended behavior. The [exact-command audit](benchmarks/geometry_only_command_audit_2026-10-17.md)
recovers the earlier assistant command and verifies that all optimization
arguments match the script defaults; only the output directory differs.
The earlier command also enabled reverse diagnostics, disabled CUDA command
buffers, and overrode VMEX/NTX import paths. Their effect on this remote
comparison is not isolated. The new report therefore weakens attribution of
the earlier bad result to replay itself; neither a replay regression nor a
specific environment-induced defect is established. No new successful-run
optimization evaluations or internal-stage arrays are available in this
audit. The subsequently supplied [successful postprocessing tail](benchmarks/geometry_only_bare_run_postprocessing_2026-10-17.txt)
confirms the optimized geometry's transport report reaches `t=2`, reports
`done=True` and `failed=False`, writes the bootstrap reports and returns to
the shell. It contains no initial/final optimizer cost or evaluation history,
so it does not isolate why the earlier diagnostic command diverged. The
[command audit](benchmarks/geometry_only_command_audit_2026-10-17.md#supplied-successful-run-excerpt-and-combined-continuation)
records the geometry-plus-profile continuation from this saved result,
without a repeated geometry stage or expensive local tests.

The subsequent audit reproduced an actual optimizer/report configuration
mismatch: optimization forces Radau `legacy` (`dt_close`), while the original
postprocessing config omitted the key and the forward solver defaulted to
`retry_only`. The example configuration helper now explicitly sets `legacy`
for both. This preserves the optimizer's existing mode and AD equations while
aligning fresh reports. A mocked report/configuration regression test fails
before the fix and passes after it; all 17 small checks in the simple-combined
and warm-start test files pass. This setup mismatch is confirmed, but its
effect on electron/ion branch selection and the difference between shell
commands remain numerically unisolated. See the [updated command audit](benchmarks/geometry_only_command_audit_2026-10-17.md#optimizer-versus-plotted-forward-solver-confirmed-mismatch).

The user also clarified that the next combined optimization must start from
the usual initial geometry and nominal profiles, with all 30 DoFs active
immediately. The single-stage simple-least-squares command in that audit is
the requested case; saved-geometry stage-2 commands are not substitutes.

On 2026-10-18 the user clarified that the requested larger step limit concerns
the reverse sweep. The geometry/shared CLI now accepts
`--max-reverse-accepted-steps` (positive integer, default 500). The single-stage
combined runner, saved-geometry/profile-only runner and warm-start runner
inherit it; the geometry and combined problem builders pass the supplied cap
through to the reverse setup. Setup/summary output records the selected cap.
This raises the guard on the accepted trajectory being differentiated; it
does not add rejected steps to differentiation, change `t_final=2`, change the
forward attempted-step budget of 1000, or replace the full-time solve with a
fixed accepted-step prefix. The checkpoint segment length stays at 50.
All 31 checks in the simple-combined, warm-start and saved-geometry test files
pass, using configuration/mocks and a small analytical-profile plot. No
transport, ambipolar-root, VMEX or NTX solve was run locally.

The user then clarified that the total trace budget must also increase.
The shared CLI now additionally exposes `--transport-max-steps` (alias
`--max-steps`, positive integer, default 1000). `transport_config` passes it
to the actual solver configuration, so it affects the adaptive trajectory
recorded for reverse AD as well as fresh forward reports. Setup/summary
output records both budgets. For example, use
`--max-reverse-accepted-steps 1000 --transport-max-steps 2000`; the larger
total includes accepted and rejected adaptive attempts, while the adjoint
continues to differentiate accepted steps only. The physical final time and
checkpoint segment length are unchanged. Saved-result continuations inherit
their stored total budget unless the CLI explicitly overrides it. A fallback
preserves older argument namespaces used by the legacy combined example.
All 36 lightweight tests and two subtests in the three affected runner test
files pass, with no expensive local physics evaluation.

The user supplied the standalone geometry-only run at `c6a1cca`, using the
corrected reverse implementation from `e83a143`. The complete supplied excerpt
is archived as [raw output](benchmarks/geometry_only_net_power_root_bootstrap_replay_fix_2026-10-17.txt),
with an [evaluation table](benchmarks/geometry_only_net_power_root_bootstrap_replay_fix_2026-10-17.csv)
and [replay endpoint table](benchmarks/geometry_only_net_power_root_bootstrap_replay_fix_2026-10-17_segments.csv).
Source attachment: `a1bed547-bce0-4e92-ba61-57c1a3ee3c04/Pasted text.txt`,
134380 bytes, SHA256
`b69e29806135e0b3e191aeeabf6acb7cf5b243dcfbbfb88710f13a5b3071f4b1`.

The output directory is
`outputs/geometry_only_net_power_root_bootstrap_replay_fix`.
There are 24 RBC/ZBS variables, no profile DoFs, max mode 2, database grid
`(25,25,33)`, and the surrogate QI backend. The new location/strength objectives
are enabled with target radius 0.514 and strength 0.9. Old left/right targets
and max-Er are disabled; bootstrap and net power are enabled. The log confirms
the intended geometry-only setup. This is not an accidental profile-only run.

| Quantity | Initial | Best/final report | Target |
| --- | ---: | ---: | ---: |
| Total cost | 17335.8986 | 2461.26980 | 0 |
| Reported transition location | 0.117510582 | 0.371065463 | 0.514 |
| Crossing strength | -3.26833863 | -1.18145509 | at least 0.9 |
| Location cost | 78.6019292 | 10.2151409 | 0 |
| Strength deficit cost | 868.752348 | 216.622766 | 0 |
| Bootstrap penalty cost | 38.4796343 | 0.0919853288 | 0 |
| Net power (reference-volume MW) | 898.570402 | 78.7782990 | 300 |
| Net-power cost | 16349.1497 | 2233.16158 | 0 |

The optimizer exhausts 30 function evaluations with first-order optimality
`2.84e5`. This is a stalled, unsatisfactory result, despite its initial cost
reduction. The crossing-strength condition is still violated; the reported
soft transition location should not be interpreted as proof of an existing
physical crossing. A much better saved geometry is already known: its
nominal-profile evaluation in the stage-2 evidence gives cost about 3.58736,
power 293.597 MW and positive strength 1.75939 with the current objective
definitions. This run does not establish that the targets are incompatible.

Two trials fail: evaluation 2 has a typed VMEC equilibrium/Jacobian failure;
evaluation 4 fails to reach transport final time, stopping at
`0.0018070356365332645` instead of 2. They must not both be described as VMEC
failures. The CSV has an initial report, 30 optimizer evaluations (including
those two failed trials), and the final report: 32 rows in total.

All 44 reported replay endpoints have exactly zero absolute/scaled
differences. The final report reproduces best evaluation 27's cost,
transition location/strength, bootstrap cost and power at printed precision.
There is no reported CUDA OOM or replay-endpoint mismatch. These observations
support primal replay consistency at the checked boundaries; they do not
independently validate the geometry objective Jacobian.

Source/history inspection confirms that geometry-only coordinates still use
the dedicated ESS parameterization, and unit SciPy trust scaling was introduced
in `540ef7c8` on 2026-09-16, before the replay correction. The replay correction
did not introduce that scaling convention. No causal attribution to scaling
or to the replay correction is established by this log.

There is a separate validation gap: `optimization.least_squares._jac` still
uses `np.where(np.isfinite(jacobian), jacobian, 0.0)`. The standalone geometry
runner does not enable `diagnostics_dir` or save raw Jacobian snapshots.
Consequently finite costs and a lack of printed AD exceptions cannot prove
that all Jacobian entries were finite. The log does not establish that this
replacement actually occurred. The existing 16-step AD/FD checks concern a
different trajectory and setup; full-time geometry derivatives at this
stalled point remain unverified. Do not label the optimizer regression solved
or prescribe more evaluations/changed weights as an established fix.

This update only parses supplied logs and reviews source/history; no local
VMEC, NTX, transport or finite-difference physics test was run. The log ends
after the final objective report, before stage-input/summary/postprocessing
completion messages, so those later artifacts are not confirmed by this
excerpt.

### Opt-in 16-step benchmark using the corrected replay

The AD benchmark now has `--reverse-replay-jacobian-reuse-mode` with default
`legacy` and opt-in `recorded_forward`. Only the accepted-step replay's Newton
reuse/rebuild decisions change, with replay validity and endpoint checks.
The root, bootstrap, geometry and adjoint implementations remain on the
existing benchmark path; no optimization stage factories are substituted.
Benchmark objectives and physical parameter columns remain the same, allowing
comparison with the earlier 18-by-8 AD/FD case. Reports
have separate filenames and record the mode and an all-entry finiteness check;
nonfinite residuals/Jacobians are preserved in JSON and cause a failed exit.
This does not change the optimizer adapter's nonfinite handling discussed
above. See the [16-step GPU command and reuse selector](benchmarks/database_transition_objectives_ad_fd.md#replay-jacobian-reuse-comparison-2026-10-17).
Local checks use mocked benchmark wiring/reporting plus tiny synthetic replay
equations: 17 tests and 33 subtests, plus four existing mocked geometry/stage
and tiny replay-cache checks pass. No local physics evaluation was run.

The subsequent GPU benchmark succeeds with all 18 residuals/144 Jacobian
entries finite and four exactly matching replay endpoints. The largest
relative difference from the archived AD over all 88 transport derivatives
is `4.217200e-7` (net power versus RBC:1:0); root/bootstrap/strength rows are
also very close. QI/MaxJ geometry-only baselines and derivatives differ after
the intervening Boozer-context correction, so those changes cannot be
assigned to replay reuse. The [full comparison and artifacts](../reverse_ad_vs_fd_database.md#2026-10-17-recorded-forward-replay-ad-compared-with-the-archived-ad-and-fd)
retain all pairs. This confirms no large transport-gradient change at the
16-step benchmark point; it does not explain the stalled full-time optimizer
or establish that the late-time replay-policy difference was exercised.

### Replay invariance and incomplete cache reconstruction audit (2026-10-17)

The user reports that the geometry-only optimizer satisfied the objectives
before the replay change. This is an unresolved regression report. Different
benchmark inputs do not explain that before/after result. At identical inputs,
an accepted-step reconstruction change should preserve physical objective
values and accurate derivatives within numerical tolerance. The successful
16-step comparison only validates that tested prefix; it does not establish
full-time derivative parity or resolve the optimization regression.

Source inspection of `e83a143` confirms that the new metadata is an accepted
step's Jacobian-reuse boolean. The checked replay clears `cache_valid` when
the forward rebuilt, but otherwise still decides reuse from its own cache
and `cache_dt`. Forward rejected attempts can update the Jacobian, LU factors,
and cache time without advancing the physical state. Their matrix provenance
is not fully represented by that single boolean. Consequently
`recorded_forward` must not be described as reproducing every forward Newton
setup exactly.

A [scalar reproducer](benchmarks/probe_recorded_replay_lu_provenance_2026-10-17.py)
uses `y'=-100*y`, one accepted step at `dt=0.001`, a rejected attempt at
`dt=0.002`, and an accepted retry at `dt=0.001802`, with the existing legacy
forward reuse policy. The actual checked segment replay receives the recorded
rebuild mask and only the two accepted slots. Forward reuses the rejected
attempt's LU, whereas replay refactors at the accepted timestep. Their Newton
iteration counts differ, even though both converge and the physical endpoints
agree to roundoff. See the [probe output](benchmarks/recorded_replay_lu_provenance_2026-10-17.txt).

This demonstrates incomplete numerical-setup reconstruction. It does **not**
demonstrate an incorrect objective derivative or explain the large optimization
cost regression: for this scalar case the endpoint discrepancy is tiny, and
the optimizer's `block` adjoint builds stage RHS Jacobians independently of
the cached Newton factorization. A full-time, identical-input comparison is
still missing. No production replay behavior was changed in this audit, and
no local plasma-physics evaluation was run.

Clarification: reproducing Newton iteration counts or the exact cached
factorization is **not** required for a correct implicit accepted-step
derivative. The geometry optimizer selects the `block` stage-adjoint mode,
which forms the stage-residual derivative from RHS Jacobians at
the reconstructed Radau stages, independently of the forward Newton cache.
If both reconstructions reach the same stage solution and the adjoint solve
is accurate, different Newton iteration counts should leave derivatives
unchanged within numerical accuracy. The scalar LU audit is a numerical-setup
fidelity observation, not a demonstrated defect requiring identical cache
history. Matching step endpoints alone does not establish matching internal
stages. Further regression diagnosis should compare stage solutions and their
derivatives, rather than require identical Newton iteration counts.

The scalar probe now checks this directly with the production block-transpose
function. Despite 5 versus 2 Newton iterations, forward and replay internal
stage states differ by at most `8.770761894538737e-15`. Both yield the implicit
accepted-step derivative `0.8351031779295103`, equal to the independent
three-stage Radau IIA stability-function reference. Altering only the cached
Newton Jacobian and LU factors also leaves that derivative exactly unchanged.
The executable assertions pass on the scalar CPU problem, with no plasma
physics evaluation. This supports the user's point: different reconstruction
iteration counts are not by themselves a gradient defect.

Reinspection of the supplied GPU table finds 66 profile/transport AD-FD
pairs, with maximum relative error `1.9358346865813661e-6`. The four 16-step
benchmark replay endpoints match exactly. The benchmark and optimization
wrapper call the same database segment reverse body and the exact `block`
stage equations. The wrapper rebuilds the RHS from each trial's live support
payload; `git blame` dates that behavior to `7e6fb463` on 2026-09-14, before
the October replay correction. No new stale-first-trial RHS defect was found
in this inspection. The supplied GPU log does not store internal stage arrays,
so direct GPU stage-by-stage equality cannot be reconstructed from it. The
16-step agreement remains strong positive evidence for that trajectory, not
evidence of a cache-induced gradient regression. The full-time optimization
regression is still unresolved.

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
