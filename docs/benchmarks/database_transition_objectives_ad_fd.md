# Database transport transition-objective AD/FD comparison

## Full-time comparison (2026-10-19)

The full 20-second recorded-forward AD run succeeds: 290 accepted steps, six
50-step segments (40 active steps in the final segment), all 18 residuals / 144
Jacobian entries finite, and zero replay endpoint differences in all segments.
The [full-time reference](database_transition_objectives_ad_recorded_reuse_full_t20_2026-10-19.md)
archives the original output, complete value/gradient CSV, timing and the
matching FD setup. The [eight-DoF full-time FD loop](run_database_transition_fd_full_t20_2026-10-19.sh)
uses the existing accepted-step frozen-metadata and frozen-root contracts with
no accepted-step prefix limit. Full-time FD results remain pending; keep them
separate from the sixteen-step comparisons below.

The full-transport shared-payload AD benchmark and realtime-geometry FD runner
now use the same eleven transport objective rows. `Er_transition_left` and
`Er_transition_right` are replaced by `Er_transition_rho`,
`Er_transition_location_moment`, and `Er_transition_strength`. Bootstrap and
net power remain included.

The FD runner imports `FULL_TRANSPORT_OBJECTIVE_LABELS` from the AD benchmark
and evaluates the production scalar value functions. It computes derivatives
by central differences of perturbed endpoint solves, without using the
transport reverse sweep. This checks derivatives through transport and its
boundaries; sharing the primal objective formulas does not independently
validate their physical definitions.

Both benchmarks use the scalar objective defaults: transition radial window
0.25 to 0.75, target/prior 0.5, Er temperature 2.0 kV/m, smoothing epsilon
1e-6, radial softness 0.05, and softmax beta 16. These are benchmark defaults;
they differ from some optimization-script target/window settings.

## Replay Jacobian reuse comparison (2026-10-17)

The AD benchmark now accepts
`--reverse-replay-jacobian-reuse-mode legacy|recorded_forward`. This changes
only the Newton-cache reuse decisions in accepted-step replay. The default
`legacy` recomputes reuse from the replay cache, as before. `recorded_forward`
applies the original forward rebuild decisions, checks reconstructed Newton
solves and compares segment endpoints with adaptive checkpoints. Rejected
transport attempts are not replayed or differentiated.

The recorded metadata forces forward rebuilds; it does not retain complete
Jacobian/LU provenance across skipped rejected attempts. A later
[scalar audit](probe_recorded_replay_lu_provenance_2026-10-17.py) demonstrates
different Newton iteration counts and LU factors despite converged endpoints
agreeing to roundoff. This mode is therefore not proof of exact forward
Newton-setup reconstruction, nor does that scalar finding establish a wrong
transport derivative or the cause of the optimizer regression.

The scalar probe also checks the production `block` adjoint: internal stage
states agree within `8.77e-15`, and forward/replay implicit derivatives are
both `0.8351031779295103`, equal to the Radau IIA stability-function reference.
Changing only the cached Newton matrices leaves this derivative unchanged.
Different reconstruction iteration counts therefore do not invalidate an
implicit accepted-step gradient when the same stage solution is reached.

Both choices retain the same benchmark root, bootstrap, geometry and reverse
implementations. No production optimization stage factories are substituted.
The forward solver's separate `--radau-jacobian-reuse-mode legacy` setting is
unchanged in this comparison; it still controls the actual adaptive solve.

The recorded replay retains the existing benchmark configuration, eleven transport
objectives (including the three new transition rows, bootstrap and net power),
seven geometry rows, and physical parameter columns. With profiles included
and `RBC:1:0,ZBS:1:0`, this gives the same 18-by-8 Jacobian used in the archived
comparison. It does not apply the optimizer's ESS scaling, weights or root
target/window overrides. The numerical comparison therefore concerns the
16-step benchmark point, not the stalled full-time optimization point.

After syncing this change to the GPU machine, use the existing `jax_env`:

```bash
cd ~/NEOPAX
mkdir -p outputs
set -o pipefail

env -u JAX_COMPILATION_CACHE_DIR \
JAX_ENABLE_COMPILATION_CACHE=0 \
XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer=" \
NEOPAX_REVERSE_STEP_DIAGNOSTICS=1 \
PYTHONPATH="$HOME/VMEX:$HOME/NTX/src" \
/usr/bin/time -v \
python examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --config examples/benchmarks/Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box.toml \
  --reverse-parameter-mode profiles_plus_realtime_geometry \
  --reverse-geometry-parameter RBC:1:0,ZBS:1:0 \
  --realtime-geometry-gradient-path reverse_payload \
  --optimization-api-profile-dofs include \
  --objective all \
  --accepted-step-limit 16 \
  --reverse-segment-length 4 \
  --radau-jacobian-reuse-mode legacy \
  --timing-mode jit-warm \
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
  --reverse-replay-jacobian-reuse-mode recorded_forward \
  2>&1 | tee outputs/database_transition_ad_16x4_recorded_reuse.log
```

The final banner and JSON record `reverse_replay_jacobian_reuse_mode` and `all_finite`.
All 18 residuals and 144 Jacobian entries are checked. If any are nonfinite,
the raw report is written and the benchmark raises instead of returning a
successful exit status. The selected mode requires the database Radau shared
payload lane, positive segment length, minimal segment-start replay,
`reuse_segment_primal_record`, `reuse_static_probe` and inline segment support.

The new mode writes
`outputs/autodiff_transport_lagged_ntx/reverse_ad/transport_reverse_ad_only_full_transport_shared_payload_smoke_replay_jacobian_recorded_forward.json`.
The legacy report retains its original filename. Compare
objective values and derivative columns by label against that reference and
the archived FD table before judging agreement. Local validation uses mocked
benchmark routing, report checks and tiny synthetic replay equations; no local
VMEC, NTX or full transport evaluation is required. The two lightweight test
files `test_reverse_benchmark_root_objectives.py` and
`test_checked_primal_replay.py` cover the selector, unchanged stage wiring,
filtering accepted decisions, and invalid reconstruction.

Validation: those two files pass 17 tests and 33 subtests; four existing
mocked geometry-scaling/stage and tiny replay-cache checks also pass. The
broader stage-switching proposal was removed before any remote run: the
final option only changes replay Jacobian reuse and its validity checks.

The user has now supplied a successful recorded-forward GPU run: all 144 AD
entries are finite, and all four replay endpoints match exactly. Across 88
transport derivatives, the largest change from the archived AD is
`4.217200e-7` relative. QI/MaxJ geometry-only rows differ with their baseline
values after the intervening Boozer-context correction; they are not a
replay-only comparison against the older archive. See the
[new AD/AD and AD/FD tables](../../reverse_ad_vs_fd_database.md#2026-10-17-recorded-forward-replay-ad-compared-with-the-archived-ad-and-fd)
for all saved values and the limits of this 16-step check.

## Matching 16-step FD command

On the GPU machine, from NEOPAX with `jax_env` active, use the same database
TOML as the AD run:

```bash
env -u JAX_COMPILATION_CACHE_DIR \
JAX_ENABLE_COMPILATION_CACHE=0 \
PYTHONPATH="$HOME/VMEX:$HOME/NTX/src" \
/usr/bin/time -v \
python examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py \
  --config examples/benchmarks/Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box.toml \
  --parameter n0 \
  --accepted-step-limit 16 \
  --radau-jacobian-reuse-mode legacy \
  --geometry-fd-lane frozen_linearized \
  --replay-mode accepted \
  --fd-accepted-replay-contract reverse_frozen_metadata \
  --initial-Er-root-ad jax_selected_root \
  --initial-Er-root-fd-root-lane frozen_linearized
```

The selected-root linearization freezes the baseline root branch, matching the
local implicit derivative used by AD. The accepted-replay contract freezes
solver predictor/controller/Jacobian/factorization metadata to the unperturbed
replay, matching the metadata held nondifferentiable by reverse AD. These
existing FD modes are selected explicitly by the command; their defaults have
not changed.

For all eight columns of the AD case, repeat the command with `--parameter`
set to each of:

```text
n0
T0
density_shape_power
temperature_shape_power
density_shape_alpha
temperature_shape_alpha
RBC:1:0
ZBS:1:0
```

Compare FD rows by name with the AD `transport:<name>` residual rows and the
matching parameter column. The AD benchmark uses unit weights and zero targets
for these raw observables, so its transport residual Jacobian equals their
objective Jacobian. The AD run also reports seven geometry-only residual rows;
this FD runner covers the eleven transport rows, not those additional rows.

The FD console and JSON contain eleven objective values and eleven derivative
entries per parameter. JSON reports are written under
`outputs/autodiff_transport_lagged_ntx/realtime_geometry_fd/`; the selected
command uses the `_frozen_linearized_summary.json` filename suffix. Keep both
runs' configuration, baseline values, step count, replay mode, selected-root
mode, FD step size, and commit when comparing results.

## Local checks and remaining validation

`tests/test_reverse_benchmark_root_objectives.py` checks the shared row order,
the AD reverse-builder request, FD replay endpoint routing, and synthetic
central differences of the new terminal root rows against local AD. It runs
without VMEC, NTX, or transport solves. It does not validate the complete GPU
Jacobian. Remote comparison results are recorded below.

The updated 16-step AD run has now completed successfully: 18 residual rows,
eight parameter columns, and all 144 Jacobian entries finite. Its
[run note, original output, and CSV reference](database_transition_objectives_ad_16x4_2026-10-14.md)
are archived for comparison.

The matching sixteen-step `n0` FD run has now completed successfully. All
eleven baseline values match the AD reference at printed precision. Across
the eleven derivatives, the largest relative error is 3.367259e-07; the new
radius and location-moment rows each give 2.236e-08, and strength gives
1.312e-07. These comparisons use the rounded `.6e` console gradients (seven
significant digits). The
[current AD/FD tables](../../reverse_ad_vs_fd_database.md#2026-10-15-boozer-aligned-fd-rerun-all-eight-dofs)
contain the detailed comparisons and the eight-column matrix.

The first remaining-seven-DoF FD batch completed. All six profile columns
match the AD baseline, with a largest relative derivative error of 1.948462e-06
across all eleven objectives. Both geometry FD runs in that batch had different
baseline values from AD, so those discrepancies did not establish a successful
validation. For RBC/ZBS respectively, the apparent relative differences are
7.249889%/6.297522% for radius and location, 24.208012%/17.115679% for strength,
and 80.292974%/0.056215% for net power. The
[combined CSV](database_transition_objectives_ad_fd_16_2026-10-15.csv) retains
all 88 pairs and flags the mismatched baselines. The next geometry check must
first reproduce the AD primal baseline; no extra local physics run was used
to record these results.

The [geometry context audit](database_geometry_fd_context_mismatch_2026-10-15.md)
subsequently found and corrected a default-resolution mismatch: geometry FD
used `(mboz, nboz) = (18, 18)` while the runtime used automatic `(36, 11)`.
The FD command's mode flags remain appropriate. After updating the code, its
geometry startup banner should print `(36, 11)`. Matching baselines and a
refreshed AD geometry reference are still required before judging the
corrected geometry derivatives. Only mocked/synthetic local checks were used
for that source correction.

## Boozer-aligned remote FD rerun

The user has now supplied the full eight-DoF FD rerun after the correction.
All runs exited successfully. Both geometry banners report `(36, 11)`, and
all 88 baseline objective values match the archived AD reference exactly at
printed precision. The profile FD gradients are unchanged at printed
precision. The observed geometry baseline mismatch is resolved in this rerun.

Against the archived AD geometry columns, RBC/ZBS relative errors are now
1.520258e-04/2.319614e-05 for radius and location moment,
5.968463e-04/3.284039e-05 for strength, and
4.998444e-04/5.595834e-06 for net power. Bootstrap has the largest remaining
geometry differences, 3.220081e-03/7.550967e-03. The
[updated tables](../../reverse_ad_vs_fd_database.md#2026-10-15-boozer-aligned-fd-rerun-all-eight-dofs)
and [comparison CSV](database_transition_objectives_fd_boozer_aligned_16_2026-10-15.csv)
record every pair and retain the before-fix discrepancies. A refreshed AD run
after the shared-context correction is not included in the supplied evidence,
so these comparisons remain explicitly against the older archived AD run.
