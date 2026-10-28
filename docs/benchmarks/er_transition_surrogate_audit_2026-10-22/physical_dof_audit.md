# Test the new root costs through geometry and profile parameters

The purpose is to determine whether the NEW costs guide physical parameter
changes toward the requested Er downcrossing, or merely reduce amplitude.
The first full-time physical AD and forward descent runs are saved below.
The forward profile test exposed a test-harness mismatch. Corrected small
profile steps now reduce the cost and move the terminal root toward the
target, with directional changes approaching the AD prediction. Efficient
root placement through optimization remains unverified.

## Candidates and observations

The existing AD benchmark's `--root-loss-audit` now evaluates only:

1. Interpolated Er at the target (diagnostic).
2. Target-zero plus local sign-margin cost (new candidate).
3. Target-zero plus annular sign-margin cost (new candidate).
4. Radius of the nearest actual downcrossing in the annulus (diagnostic).

It does not request the old position/strength, left/right, net-power,
bootstrap, or explicit geometry-objective rows. Geometry PARAMETERS remain
active: the standard test has four rows and eight columns, with six profile
parameters and RBC:1:0/ZBS:1:0. The rows use the established full-transport
reverse path, including initial-root, live database and VMEC sensitivities.
The ordinary benchmark retains its existing objective list.

Both candidates interpolate an off-grid target. The annular sign penalty
excludes the core and tapers its margin near the requested zero. Defaults
are annulus 0.30--0.90 and target 0.514; the inner cutoff is an explicit
experiment setting, not a validated universal core boundary. Unresolved
bands are rejected.

Each candidate is a residual block with one overall root weight. Its audit
row is `C = 0.5 * sum(residuals**2)` at weight one. A future least-squares
optimizer must use the block, not square this cost row again.

The report saves the physical Jacobian, concrete terminal Er/rho, zero/sign
cost contributions, real downcrossings, amplitude RMS and wrong-sign
fraction. It predicts root movement along normalized cost-descent directions
separately for profiles and geometry. Missing roots are marked unavailable,
never treated as a successful target-radius solution.

Profile coordinates use nominal scales times the simple optimizer's profile
trust multiplier (default 3.7). Geometry coordinates use the shared ESS
scaling implementation (alpha 1.2). These settings and every physical
direction are saved. The geometric benchmark covers the selected two
harmonics; it does not establish controllability with every optimizer
harmonic or at every physical state.

## Run on the GPU machine

With the updated checkout and jax_env active:

```bash
cd ~/NEOPAX
bash examples/benchmarks/run_er_root_candidates.sh
```

The shell script contains the full reproducible commands. It runs:

- One full-time candidate-only AD benchmark with recorded_forward Newton
  setup reuse, 50-step segments, a 1000-accepted-step discovery guard, and
  no 16-step prefix limit. The existing TOML has final simulation time 20
  seconds and a 20000-attempt budget, using
  input.QI_nfp2_newNT_opt_hires_true as its geometry seed.
- One independent adaptive forward baseline, followed by four downhill
  forward endpoints: each new candidate with profiles only, then geometry
  only. Default step length is 0.01 in the recorded trust coordinates.

These are substantial GPU calculations. No geometry warm-start optimization
is needed. Each forward endpoint gets a fresh process so its compiled
executables are released on exit. CUDA command buffers are disabled as in
the previous memory workaround.

The forward checker first requires its baseline to reproduce AD terminal
Er, rho and time. On mismatch it saves differences and stops before running
downhill endpoints. Geometry changes rebuild physical geometry and the
database. Profile changes follow the AD parameter map: initial profile
values change under a fixed configuration, and the requested JAX selected
root is recomputed. Prescribed frozen-species temperatures remain those in
the baseline config, matching the function differentiated by AD. Accepted
timesteps adapt independently. These finite moves are not the frozen-schedule
FD oracle, which has a separate unresolved replay issue.

Default outputs are in outputs/new_root_candidates:

- ad.log and ad_summary.json: new cost values, physical gradients,
  predicted movement and saved inputs.
- descent/summary.json: actual cost, zero/sign, root-position and amplitude
  changes for each candidate and parameter group.
- descent/terminal_Er_comparison.svg: baseline and endpoint field profiles.
- Separate task, endpoint JSON and log files for the forward solves.

Interpret cost_decreased_without_position_improvement alongside
annular_Er_rms_change. A cheaper cost with an unchanged/misplaced root and a
weaker field exposes the amplitude failure we want to detect. Root
creation/loss and bracket changes are recorded separately. A finite move
need not agree with the conditional local derivative when branches change.

To sample two step sizes:

```bash
NEOPAX_ROOT_DESCENT_STEPS=0.01,0.003 \
  bash examples/benchmarks/run_er_root_candidates.sh outputs/new_root_candidates_two_steps
```

To run only the forward check using the saved AD result:

```bash
cd ~/NEOPAX
env -u JAX_COMPILATION_CACHE_DIR \
JAX_ENABLE_COMPILATION_CACHE=0 \
XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer=" \
PYTHONPATH="$HOME/VMEX:$HOME/NTX/src${PYTHONPATH:+:$PYTHONPATH}" \
python examples/benchmarks/check_er_root_candidate_descent.py \
  --ad-report outputs/new_root_candidates/ad_summary.json \
  --out-dir outputs/new_root_candidates/descent \
  --step-sizes 0.01,0.003
```

Completed endpoints are reused only when their task digest matches. The
digest includes parameters, resolved configuration, seed contents and local
model sources. External VMEX/NTX/environment changes require a new output
directory; their installed code is not included in that digest.

## First physical AD result: 2026-10-23

The GPU run on neopax-24 completed successfully with four rows, eight
parameters, recorded_forward reuse, 290 accepted steps across six reverse
segments, and all finite values/derivatives. Wall time was 55:43.78 and peak
host RSS was 11793504 KiB. This is the configured full 20-second benchmark;
the excerpt does not print the terminal simulation time or replay endpoint
differences separately.

Evidence is retained in [the original pasted log](physical_ad_2026-10-23.txt)
and [the 32 physical derivatives](physical_ad_2026-10-23.csv), with numeric
strings preserved as printed. Raw-log SHA256:
`4ffde017decc5217972f7be63f32a051468b2a17b4986ff675defd5bd8d50b59`.

The nearest actual crossing is 0.51703593282774452, versus target 0.514,
giving an outward position error of 0.003035932827744503. Interpolated Er at
the target is 5.0382040564010531 kV/m. Both candidate costs and all their
gradients are identical at the printed precision. In particular,

```
C_local = C_annular = 3.1729375142420033 = Er(target)^2 / 8
dC/dp = Er(target)/4 * dEr(target)/dp
```

The gradient identity matches the measured table to a maximum absolute
discrepancy of 7.61e-11 across the eight physical parameter derivatives.

Thus the sign margins make no measurable contribution at this state;
the shared target-zero term supplies the observed cost and gradient. This
does not establish equivalence when sign constraints become active later.

For the default trust-coordinate step length of 0.01, first-order AD predicts:

| Changed group | Root shift | Resulting root | Root-error reduction | Candidate-cost reduction |
| --- | ---: | ---: | ---: | ---: |
| Profiles | -8.9408553e-6 | 0.51702699197 | 0.2945% | 8.1979% |
| Geometry | -1.5887882e-5 | 0.51702004495 | 0.5233% | 2.9645% |

These are predictions, not observed finite changes. Both directions locally
improve root position. The profile direction achieves 98.9543% of the
maximum possible local root-displacement rate for a unit step in the saved
profile metric; geometry achieves 86.1692%. The profile direction therefore
already points almost along the best local root-moving direction available
in this metric, even though the root's parameter sensitivity is small.
Larger cost gradients alone do not imply faster root displacement.

The identical physical directions also mean that separate local and annular
forward runs at this baseline would duplicate the physical solves. For the
first forward test, `--candidates annular --groups profiles,geometry` is
sufficient: baseline plus two changed endpoints. Endpoint files retain both
candidate costs, root observations and field amplitudes. The two formulas
can be distinguished if their sign contributions become active.

## First forward descent result and test correction: 2026-10-23

The supplied run completed successfully in 46:53.67. Its baseline passed
the terminal Er/rho/time agreement check. The observed endpoint values are
saved in [physical_descent_2026-10-23.csv](physical_descent_2026-10-23.csv).
These observations came from the original checker, before the correction
described below; preserve them as that experiment rather than relabelling
them as a corrected AD-matched comparison.

| Group | Predicted root shift | Observed root shift | Predicted cost change | Observed cost change |
| --- | ---: | ---: | ---: | ---: |
| Profiles | -8.9408553e-6 | +0.01960100135 | -0.26011520 | +121.29227260 |
| Geometry | -1.5887882e-5 | -1.6495913e-5 | -0.09406100 | -0.10394860 |

The geometry move improves the actual root to 0.5170194369139547 with the
same final crossing bracket. The profile move worsens it to
0.5366369341726912, changing final bracket 25 to 26; target Er increases to
31.444456007850473 kV/m. This is not the amplitude-collapse scenario:
annular field RMS increases by 1.2584116840933817 kV/m and the annular cost
rises from about 3.17294 to 124.46521. A changed final grid bracket alone
does not establish a discontinuous initial-root branch switch.

Review found two differences between the checker and the AD parameter map:

1. The checker wrote changed parameter values into config["profiles"],
   whereas the AD benchmark changes its initial profile state under a fixed
   config. Solver preparation builds prescribed frozen-species temperature
   profiles from that config; helium temperature is frozen in this case.
2. It reused the general initializer's Er profile with
   initial_er_root_ad="off". The benchmark explicitly requests the JAX
   selected-root initializer, which has a separate entrypoint and scan
   batching. Baseline agreement does not establish their equivalence at a
   changed parameter point.

The corrected checker preserves the profile config, passes changed values
to the production initial-state builder, and invokes its requested
jax_selected_root mode. It saves the actual initial Er/density/pressure,
the general initializer's Er, and initial sign-change observations so later
differences can be located before or during transport. A mocked test verifies
that the selected-root result is passed to the solver without replacement.

Neither mismatch has yet been demonstrated to explain the complete profile
failure. A finite-step change of branch or accepted schedule can still be
relevant after the test contract is corrected. Step 0.01 is not a 1% change
in every physical profile parameter: with the saved direction, n0 decreases
by 0.15539654133, about 3.69% of its baseline value.

The subsequent check reused the existing AD report and tested only profile
steps 0.001 and 0.0001 with the corrected checker in
outputs/new_root_candidates/descent_ad_contract. The code change invalidated
the old endpoint digests, so its baseline was checked again. No AD rerun or
repeated geometry endpoint was required. Results are recorded below.

## Isolating the initial-root boundary before another transport run

The checker also has an initialization-only diagnostic. It builds the
baseline geometry/database once, reuses it for every profile point, and
does not prepare or execute a transport rollout. It compares:

- The JAX selected-root and general initialization entrypoints on identical
  pre-root states and the same runtime.
- Actual selected-root changes against local implicit continuation. The
  latter uses the fixed-root charge-flux residual's directional finite
  difference and its AD derivative with respect to Er:
  `dEr/ds = -(dF/ds)/(dF/dEr)`. Its residual FD step is recorded explicitly.
- The extra frozen-temperature changes that the old configuration-mutating
  checker would have introduced, computed with the actual profile builder.

Run this after syncing the corrected checker, with the existing AD report:

```bash
cd ~/NEOPAX
neopax_root_ad_report=outputs/autodiff_transport_lagged_ntx/reverse_ad/transport_reverse_ad_only_full_transport_shared_payload_smoke_root_loss_audit_replay_jacobian_recorded_forward.json
env -u JAX_COMPILATION_CACHE_DIR \
JAX_ENABLE_COMPILATION_CACHE=0 \
XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer=" \
PYTHONPATH="$HOME/VMEX:$HOME/NTX/src${PYTHONPATH:+:$PYTHONPATH}" \
python examples/benchmarks/check_er_root_candidate_descent.py \
  --ad-report "$neopax_root_ad_report" \
  --diagnose-profile-initialization \
  --candidates annular \
  --step-sizes 0.01,0.001,0.0001 \
  --out-dir outputs/new_root_candidates/profile_initialization_diagnostic
```

It saves initialization_diagnostic.json with per-radius selected/general
roots, baseline charge-flux residual and Er derivative, implicit response,
finite-step prediction errors, and initial sign changes within the annulus.
The charge-flux residual and derivative help identify an inaccurate root or
an ill-conditioned implicit boundary; a finite Er alone does not establish
either root accuracy or derivative conditioning.

A disagreement between initialization entrypoints at the same state
identifies an initializer discrepancy. If selected-root changes fail to
approach the implicit prediction as steps shrink, the initial-root boundary
needs investigation before judging the full transport adjoint. If they
agree, a subsequent corrected full-time check can focus on transport,
adaptive scheduling and response-cache reuse. A sign change is an
observation, not by itself proof of discontinuous branch selection.

Full forward checks now retain compact adaptive schedule metadata (accepted
masks, timesteps and error norms, without per-step state histories), as well
as measured and predicted directional slopes. This makes small-step
convergence and changed solver decisions reviewable from the saved results.

## Initial-root diagnostic result: 2026-10-23

The initialization-only run completed with one baseline runtime/database
build and no transport solve. The printed observations are preserved in
[profile_initialization_2026-10-23.csv](profile_initialization_2026-10-23.csv).
The full per-radius report remains on the GPU machine at
outputs/new_root_candidates/profile_initialization_diagnostic/initialization_diagnostic.json;
only its console summary was supplied here.

| Trust-coordinate step | Initial-response relative discrepancy | Maximum selected/general Er difference (kV/m) | Initial sign-change indices in annulus | Extra legacy frozen-temperature change (keV) |
| --- | ---: | ---: | --- | ---: |
| 0.01 | 99.0607% | 0.3445498685 | [31] | 0.00930056006 |
| 0.001 | 2.82551% | 0.00159466513 | [] | 0.000930052435 |
| 0.0001 | 0.837100% | 0.000931644131 | [] | 0.0000930052077 |

The relative discrepancy compares the vector (Er_selected(p+h*d)-Er_0)/h
with local implicit continuation, using the larger of the two annular L2
norms as denominator. It is an aggregate measure over radii with finite
implicit rates, not a per-radius error bound. Continuation uses a central
FD of the fixed-root charge-flux residual and the production derivative
with respect to Er; this is not a full transport AD/FD comparison.

The small-step response approaches the implicit prediction. This supports
the local initial-root response along the tested profile direction, without
validating the complete reverse chain or excluding errors at individual
radii. The 0.01 step is poorly represented by the baseline linearization
and changes the initial Er sign at annular grid index 31. This alone does
not distinguish smooth nonlinear motion from a change in selected branch.
These initial sign observations are separate from the previously observed
terminal crossing-bracket change.

The general and JAX selected-root entrypoints return different Er values
on the same pre-root state/runtime. Their differences remain about
9.32e-4 kV/m at the smallest step. The console summary does not locate
these discrepancies or give the baseline entrypoint difference, so it does
not establish their relative size or effect on final transport. It does
confirm that substituting the general initializer was not an equivalent
test of the AD parameter map. The extra frozen-temperature dependency of
the original checker is also measured and nonzero; its causal contribution
to the old terminal failure is still unisolated.

The subsequent comparison used the corrected full-time checker, profiles
only and steps 0.001 and 0.0001, reusing the existing AD report. The corrected
contract invokes the selected-root initializer and preserves the baseline
configuration. No new AD benchmark or repeated geometry endpoint was needed.
For those steps, AD predicts terminal root shifts of approximately
-8.9408553e-7 and -8.9408553e-8, and cost changes of -0.02601152 and
-0.002601152. Its measured directional slopes now agree closely, as recorded
below.

## Corrected full-time profile descent: 2026-10-23

The corrected checker completed its baseline and both independent adaptive
profile endpoints on neopax-24. The baseline passed its terminal Er/rho/time
comparison before the endpoints ran. The pasted console output is retained
in [physical_descent_ad_contract_2026-10-23.txt](physical_descent_ad_contract_2026-10-23.txt),
with a comparison table in
[physical_descent_ad_contract_2026-10-23.csv](physical_descent_ad_contract_2026-10-23.csv).
The full endpoint and schedule JSON files remain on the GPU machine under
outputs/new_root_candidates/descent_ad_contract.

| Profile step | AD cost change | Observed cost change | Relative discrepancy | AD root shift | Observed root shift | Relative discrepancy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.001 | -0.02601151997 | -0.02620703396 | 0.75164% | -8.94085527e-7 | -9.20579616e-7 | 2.96326% |
| 0.0001 | -0.00260115200 | -0.00260181830 | 0.025616% | -8.94085527e-8 | -8.94909506e-8 | 0.092159% |

Relative discrepancies use the absolute AD prediction as denominator.
Equivalent directional slopes are saved without rounding in the CSV.
Both steps reduce the candidate cost and improve the actual terminal root,
which moves from 0.51703593282774452 to 0.5170350122481288 and
0.5170358433367939, respectively, toward target 0.514. No initial Er sign
changes or terminal crossing-bracket changes are reported. Both candidates
remain numerically equal at these endpoints: the shared target-zero term
accounts for the cost change and sign penalties are negligible.

Accepted-step counts change by -2 and +1, respectively. Thus the close
small-step agreement survives these specific changes in adaptive schedule;
it does not require identical accepted-step counts. This supports the cost
and actual-root directional derivatives through the full simulation along
the tested profile direction. It does not independently validate every
profile column, every objective, or behavior near selection boundaries.

The previous assertion that profiles cannot move the root is not supported
by these corrected observations. They do move it in the predicted direction.
The old failed run used both a different initial-state map and a larger
0.01 move, so these results do not isolate which change contributed how much
to that failure. The subsequent corrected 0.01 endpoint is recorded below.

Effective root placement remains a separate question. At step 0.001 the
cost drops 0.82595%, but the root-position error drops only 0.030323%; at
step 0.0001 the corresponding changes are 0.082000% and 0.0029477%.
The root response is locally weak even though it is correctly represented
by AD along this direction. Annular Er RMS also decreases, so the experiment
shows both amplitude and root changes, not root motion alone. Together with
the earlier near-optimal root-direction alignment under the chosen profile
metric, this means a steeper scalar cost by itself cannot be assumed to
provide much faster root motion. These two small successful steps do not
establish that the surrogate will position the root efficiently in an
optimization.

## Corrected large profile step: 2026-10-23

The follow-up reused the baseline and both small endpoints, then ran only
the missing 0.01 profile endpoint with the corrected initial-state map.
The complete supplied command/output is archived byte-for-byte in
[physical_descent_ad_contract_larger_step_2026-10-23.txt](physical_descent_ad_contract_larger_step_2026-10-23.txt).
The same comparison CSV now includes this endpoint; its relative errors
are dimensionless fractions, not percentages.

| Profile step | AD cost change | Observed cost change | AD root shift | Observed root shift | Maximum initial Er change (kV/m) | Initial sign-change indices |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0.0001 | -0.002601152 | -0.002601818 | -8.9408553e-8 | -8.9490951e-8 | 0.0160864 | [] |
| 0.001 | -0.02601152 | -0.02620703 | -8.9408553e-7 | -9.2057962e-7 | 0.156093 | [] |
| 0.01 | -0.2601152 | +121.2544677 | -8.9408553e-6 | +0.0196007514 | 24.3314954 | [31] |

The large endpoint raises the annular cost from 3.1729375142420033 to
124.42740523205913. The terminal crossing moves outward from
0.51703593282774452 to 0.5366366842306032, away from target 0.514, and
changes bracket 25 to 26. Initial Er changes sign at index 31. Accepted
steps decrease by four. The added initial-field and terminal observations
show a failure of the local model for this finite move; it is not a
nonfinite reverse-AD failure.

This outcome is essentially the same as the original checker's 0.01
endpoint: the corrected and original root shifts differ by only
2.4994326e-7 (about 0.0012752% of the original outward shift), and the
observed cost increases differ by about 0.03117%. Therefore, the original
checker mismatches were real but were not the main cause of this large-step
failure. The successful small-step comparisons and failed large step are
now obtained under the same AD parameter map.

The large initial-field change and index-31 sign change are consistent
with a change of the selected initial ambipolar root. The initializer
selects a minimum-entropy root discretely, while the tested implicit
derivative follows the local selected solution. However, these outputs do
not include all candidate roots, entropy gaps, and convergence residuals
at the changed point. They cannot distinguish a branch-selection change,
loss of a root, inaccurate root discovery, and strong smooth nonlinearity
conclusively. A single endpoint sign change is not proof of a discontinuity.

The next useful experiment is a root-only optimization with an adaptive
trust region or backtracking. For profiles, 0.001 is a demonstrated useful
initial step in the saved trust coordinates, not a global safe bound.
Trial acceptance must use the actual candidate cost; the 0.01 endpoint
must be rejected. Recompute gradients after accepted parameter changes,
adapt step lengths from actual versus predicted reduction, and monitor
the actual crossing alongside the cost. There is no reason to repeat the
same baseline AD benchmark or sweep more fixed downhill endpoints first.
This approach handles the observed bad finite trial, but convergence or
progress across a root-selection boundary is not established by these data.

## Profiles-only optimization pilot

The executable optimization pilot is now available as
examples/optimization/optimize_profiles_er_root_candidate.py. It varies the
six initial-profile parameters at fixed geometry/configuration and uses the
annular candidate cost directly as a scalar objective, with weight one.
It does not square the scalar cost again. Every trial integrates the full
configured transport interval and recomputes the selected initial Er.
Frozen helium temperature continues to come from the fixed baseline config.

The first gradient is reused from the supplied full-time AD report after
checking its independent forward endpoint. Matching saved descent
endpoints can be reused with --descent-dir. Cache matching includes the
resolved configuration, physical point, root settings/mode and local model
source digest. External VMEX/NTX/environment changes require fresh forward
checks rather than trusting this cache; installed external sources are not
included in its digest.

New forward and AD evaluations run in separate subprocesses, releasing their
compiled state when they exit. A loss-increasing trial is backtracked
without computing its AD. A proposed accepted point gets a fresh AD table;
it must be finite and reproduce the independent forward endpoint before
acceptance. Root loss, an incomplete transport endpoint, or failed/mismatched
AD also causes rejection and step reduction. This is an adaptive scalar
descent pilot, without a change to transport or replay algorithms.

For parity, the AD worker retains the established four-row/eight-column
candidate audit. The optimizer uses only the annular cost gradient's six
profile columns. Geometry parameters remain fixed and other rows do not
enter the scalar objective. Root creation from a missing initial benchmark
crossing is outside this pilot: the saved baseline must already have an
annular downcrossing.

After syncing the driver and its benchmark helper changes on neopax-24:

```bash
cd ~/NEOPAX
mkdir -p outputs/new_root_candidates
set -o pipefail
neopax_root_ad_report=outputs/autodiff_transport_lagged_ntx/reverse_ad/transport_reverse_ad_only_full_transport_shared_payload_smoke_root_loss_audit_replay_jacobian_recorded_forward.json
env -u JAX_COMPILATION_CACHE_DIR \
JAX_ENABLE_COMPILATION_CACHE=0 \
XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer=" \
PYTHONPATH="$HOME/VMEX:$HOME/NTX/src${PYTHONPATH:+:$PYTHONPATH}" \
python examples/optimization/optimize_profiles_er_root_candidate.py \
  --ad-report "$neopax_root_ad_report" \
  --descent-dir outputs/new_root_candidates/descent_ad_contract \
  --out-dir outputs/new_root_candidates/profiles_annular_optimization \
  --max-iterations 5 \
  --initial-step 0.001 \
  --max-step 0.01 \
  --root-tolerance 0.0001 \
  2>&1 | tee outputs/new_root_candidates/profiles_annular_optimization.log
```

The iteration budget counts accepted moves; each accepted move requires a
new full AD evaluation. The default limit is eight trial attempts per
iteration. Step lengths shrink on rejection and adapt from actual versus
predicted loss reduction after acceptance. The initial 0.001 is the tested
profile trust-coordinate step, not a uniform fractional profile change.

Outputs include summary.json, history.csv, events.jsonl, checkpoint.json,
and per-evaluation task/result/log files. Actual crossing error, loss,
parameters and step lengths are saved for every accepted point. A cheaper
cost alone does not set converged=true: that requires an actual crossing
within --root-tolerance and no wrong-sign annular grid points. An iteration
limit or exhausted backtracking is reported separately. This does not
establish continuous sign correctness between grid points.

Resume with the same command plus --resume; --max-iterations can be increased.
The remaining options, saved seed and local source digests must match the
checkpoint. If interrupted during an unaccepted trial, the previous
accepted state remains available, although the interrupted trial may be
repeated. Resuming after external VMEX/NTX/environment changes is not covered
by the local source digest. The five-move pilot is a progress experiment,
not a promise to reach the target within that budget.

## First optimization progress: 2026-10-23 (partial)

The user started the five-accepted-move profiles-only pilot on the GPU
machine. The supplied excerpt shows two matching forward endpoints reused,
three accepted moves after fresh AD evaluations, and the next forward trial
undergoing its AD check. It does not yet show the end of iteration 4 or the
final optimization status. The partial output is saved in
[profiles_annular_optimization_progress_2026-10-23.txt](profiles_annular_optimization_progress_2026-10-23.txt),
with reported and derived scalar values in
[profiles_annular_optimization_progress_2026-10-23.csv](profiles_annular_optimization_progress_2026-10-23.csv).

| Iteration | Reported state | Step | Candidate cost | Actual terminal crossing | Cumulative root-error reduction |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | Baseline | 0 | 3.1729375142 | 0.517035932828 | 0% |
| 1 | Accepted | 0.001 | 3.1467304803 | 0.517035012248 | 0.030323% |
| 2 | Accepted | 0.0015 | 3.1066267181 | 0.517033524101 | 0.079341% |
| 3 | Accepted | 0.00225 | 3.0446550495 | 0.517031045629 | 0.160978% |
| 4 | Forward trial; AD pending | 0.003375 | 2.9474599970 | 0.517026755396 | 0.302294% |

The actual/predicted cost-reduction ratios of the accepted moves are
1.0075164385680253, 1.0127018743975937, and 1.0179963582227247. Their full-transport AD tables
passed the driver's finite-gradient, exact-parameter and forward-primal
checks. All three ratios exceed 0.75, so the intended 1.5 growth factor
raises steps from 0.001 to 0.0015, 0.00225, then 0.003375. No accepted
move needed backtracking. The fourth forward trial has a derived reduction
ratio of about 1.027146, but remains unaccepted until its AD checks finish.

After three accepted moves, cost is down 4.04302% and the actual root has
moved inward by 4.8871991e-6 toward target 0.514. This is measurable,
consistent progress with recomputed gradients, although the root error is
still 0.0030310456, above the 0.0001 success tolerance. The pending fourth
trial would bring cumulative cost reduction to 7.10627% and root-error
reduction to 0.302294%; those are forward observations, not a completed
optimization result. Comparing the loss and position percentages alone
does not diagnose a faulty surrogate.

Let the existing five-move pilot finish before changing settings or
launching another run. Its endpoint status and accepted history will show
whether the increasing steps continue to give reliable progress or require
backtracking near the previously observed bad finite move. This excerpt
contains no memory measurements and establishes neither bounded memory nor
final convergence.

### What this establishes about candidate quality

The pilot demonstrates that the annular candidate has usable local profile
derivatives and repeatedly produces small actual root improvements from
this particular baseline. It has not demonstrated efficient positioning to
the target, performance from a missing root, or superiority over the other
candidate or the old losses. Only the annular candidate is optimized here.
The relative percentage changes in cost and root error are not directly
comparable measures of candidate quality, but the absolute crossing motion
is still small and target attainment remains unobserved.

At the original baseline, local and annular candidates had identical
values/gradients because their shared target-zero term dominated while
their sign penalties were negligible. A trajectory confined to that regime
does not discriminate between their sign-constraint designs. Later AD
reports already contain both candidate rows, their zero/sign contributions,
and projected root-progress rates separately for profiles and geometry.
An offline inspection of those existing JSON files can establish whether
the rows have separated along the trajectory without rerunning transport
or AD. The later equality of the candidates cannot be inferred from these
annular-only console lines.

The initial annular profile direction was already close to the steepest
root-moving direction in the chosen trust metric. Therefore slow root
motion here does not by itself prove an incorrectly oriented loss: weak
local physical controllability and limited useful finite steps remain
possible. These findings support retaining the candidate for evaluation,
not declaring it a validated replacement root-position objective.

## Inspecting existing profiles: 2026-10-24

The full terminal Er/rho arrays and later AD reports remain on the GPU
machine. The local workspace contains the supplied scalar logs, not the
remote evaluation JSON files. No full-profile conclusion has been inferred
from unavailable arrays.

The supplied small-step results nevertheless allow an exact scalar
decomposition. Let q be the actual interpolated downcrossing, t the target,
E the interpolated Er(t), and define A=E/(q-t). This A is the effective
secant slope magnitude between target and crossing, not an independently
measured differential slope. Then E=A*(q-t), and between two saved states:

```
Delta E = mean(A) * Delta q + mean(q-t) * Delta A
```

This symmetric split is exact for the two endpoints; it is not a separation
of independent physical controls. If target and crossing share the same
piecewise-linear cell, A equals the magnitude of that cell's Er slope.
Otherwise it remains the secant defined above.

For the first accepted move (the previously saved 0.001 endpoint), Er(t)
drops from 5.0382040564010531 to 5.017354267167761 kV/m. The split gives:

| Part of target-field change | Change (kV/m) | Fraction of the decrease |
| --- | ---: | ---: |
| Crossing displacement | -0.001524793785 | 7.3132% |
| Effective secant-slope change | -0.019324995448 | 92.6868% |
| Total | -0.020849789233 | 100% |

The effective slope decreases from 1659.52421949 to 1653.15783166 kV/m per
unit rho. The smaller 0.0001 endpoint gives a similar split: 7.1867% versus
92.8133%. These calculations are saved in
[target_field_change_decomposition_2026-10-24.csv](target_field_change_decomposition_2026-10-24.csv).
They sharpen the observed behavior: the target-zero term mostly decreases
with a weaker effective slope while the actual crossing moves little. They
do not establish whether transport physics, the profile basis/scales, or
spatial discretization causes this response. The nearly optimal local
root-moving direction also prevents treating this result alone as proof
that a differently oriented loss would solve the problem.

The read-only tool examples/benchmarks/analyze_er_root_saved_profiles.py
now inspects the already computed remote JSON reports. It imports no JAX
or NEOPAX solver modules and changes no optimizer source/checkpoints. It:

- Reconstructs off-grid downcrossings from the actual arrays, excluding the
  core according to the saved annulus, and saves nearby cell values.
- Reports the two crossing-cell Er values, their contrast, root fraction
  inside the cell, local slope and target Er across saved points.
- Computes the exact motion/slope split when its same-cell conditions hold,
  plus a local common-rescaling fit; it does not call a missing or changed
  bracket a successful zero-motion result.
- Compares local and annular cost/sign contributions and their predicted
  root movement for profile and geometry groups using existing AD rows.
- Uses cached descent endpoints, when available, to compare initial Er
  profiles and mark initial sign changes. Accepted and pending trials are
  kept distinct using events.jsonl. A partial final event line is ignored
  while a running optimizer completes its write.

After syncing the reader to the GPU machine, this command needs no new
physics calculation or compilation:

```bash
cd ~/NEOPAX
python examples/benchmarks/analyze_er_root_saved_profiles.py \
  --ad-report outputs/autodiff_transport_lagged_ntx/reverse_ad/transport_reverse_ad_only_full_transport_shared_payload_smoke_root_loss_audit_replay_jacobian_recorded_forward.json \
  --run-dir outputs/new_root_candidates/profiles_annular_optimization \
  --descent-dir outputs/new_root_candidates/descent_ad_contract \
  --out-dir outputs/new_root_candidates/saved_profile_analysis \
  --plot
```

It writes summary.json, crossing_summary.csv, crossing_cells.csv,
candidate_comparison.csv, and optionally terminal_profiles.svg. A missing
crossing is explicitly unavailable for root-direction interpretation.
Staying inside one cell does not prove numerical pinning; separating a
grid effect from a physical response requires evidence beyond this
single-grid trajectory.

## Root creation from the usual optimization start: 2026-10-24

The user requested testing candidates from the usual unoptimized initial
geometry/profiles, whose final transport state has no electron-root region.
This is more informative about root creation than the existing benchmark
near a resolved crossing. The absence of an electron region must still be
verified for the actual selected configuration, rather than assumed from
the input filename.

examples/benchmarks/benchmark_er_root_candidates_optimization_start.py now
resolves the real geometry-only example defaults, also shared by the
combined geometry/profile example. It uses:

- examples/inputs/input.QI_nfp2_initial, with zero geometry deltas and no
  geometry warm start.
- The unchanged analytical profiles from full_transport_database_vacuum_t2.toml:
  n0=4.21, T0=17.8, density_shape_power=10, temperature_shape_power=2,
  and the production defaults of 1 for both shape alpha parameters. Species
  multipliers, floors, sources, boundary conditions and equation toggles are
  preserved through the actual optimization config preparation helper.
- The optimizer's effective NTX mesh (25,25,33), not the unoverridden mesh
  stored in the TOML or the old benchmark's (25,31,64).
- Full final time 2 seconds, 1000 attempted-step budget, 500 reverse accepted
  step guard and recorded_forward replay. There is no accepted-step prefix.
- The initial max-mode=2 geometry parameterization, ESS alpha=1.2, the
  optimizer's derivative context (mboz/nboz=18/18 and its surfaces), and the
  six profile columns. An explicit --geometry-parameters subset is optional.
- Root target 0.514 and the optimizer's sign-constraint annulus 0.05--0.9;
  the inner cutoff can be overridden explicitly. Each candidate retains one
  overall root weight of one and the profile trust multiplier remains 3.7.

Before the reverse sweep, a terminal observer saves the full baseline Er
and checks the final time, the absence of a downcrossing, and nonpositive
Er throughout the chosen annulus. An all-positive field is not mistaken
for an absent electron region merely because it has no downcrossing. An
unexpected positive region or crossing stops the experiment and retains
baseline_forward_observation.json for inspection.

The same AD computation evaluates both candidate costs and their shared
target-field/chord-root diagnostics. It also evaluates a new diagnostic
row, audit_root_inner_mean_Er, to assess whether descent raises the mean
inner field before any crossing exists. This row is not an extra objective
or a second weight. A missing actual crossing keeps projected root-position
progress unavailable; the raw chord diagnostic's finite placeholder is
not interpreted as root attainment. The ordinary benchmark keeps its
original four audit rows by default.

By default, the Python entrypoint runs that shared AD, then the existing
independent forward checker for local/annular candidates with profiles and
geometry separately. It checks baseline forward/AD agreement and reports
inner-mean Er change, positive inner fraction, wrong-sign fraction, finite
field margins and actual downcrossing creation, alongside loss and RMS.
A cost decrease alone does not identify the winner. These independent
diagnostic forward evaluations still use fresh processes; this experiment
does not fix the persistent-compilation issue in the separate optimizer.

After syncing the new script and helper changes to the GPU machine:

```bash
cd ~/NEOPAX
env -u JAX_COMPILATION_CACHE_DIR \
JAX_ENABLE_COMPILATION_CACHE=0 \
XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer=" \
PYTHONPATH="$HOME/VMEX:$HOME/NTX/src${PYTHONPATH:+:$PYTHONPATH}" \
python examples/benchmarks/benchmark_er_root_candidates_optimization_start.py \
  --out-dir outputs/new_root_candidates/optimization_start \
  --step-sizes 0.001
```

This is a candidate-creation comparison, not a new optimization. For a
smaller first job, add --ad-only to compute/inspect the shared baseline AD
before deciding to run its forward endpoints. Then reuse that result with
--ad-report outputs/new_root_candidates/optimization_start/ad_summary.json;
AD is not recomputed. Physics/config overrides are rejected when reusing a
report to avoid silently comparing a different case. More steps can be
supplied to --step-sizes; existing matching forward endpoints are reused.

The output directory contains case_manifest.json (resolved config, original
seed SHA256, profile values and geometry parameter list), the baseline
forward observation, ad.log, ad_summary.json, and the descent comparison.
The AD report uses a separate filename rather than overwriting the previous
near-root benchmark report. No physical result from this new start has
been generated locally.

## Local verification and limits

Only short-array and mocked tests run locally. They check new-cost
derivatives through Er and rho against independent NumPy differences,
off-grid/core behavior, optimizer scaling, amplitude-only directions with
zero root movement, candidate-only routing and forward-worker inputs.
They do not execute VMEC, NTX, transport or optimization.

The initialization/descent suite passed 18 tests and 5 subtests. This includes an
actual short-array profile-construction regression proving the extra frozen
helium temperature dependence, and a mocked initialization probe verifying
one runtime build and no transport invocation.

With the optimization driver and benchmark integration, the targeted suite
passes 40 tests and 59 subtests. Analytic/mocked workers verify backtracking,
gradient recomputation, fixed geometry/configuration, nonfinite-AD rejection,
root-loss and incomplete-forward rejection, cache matching, actual-crossing
convergence reporting, checkpoint resume, and the explicit profile point
reaching the existing reverse evaluator. The physical pilot has now started
on the user's GPU machine; only the partial progress above has been supplied.

The GPU AD, original finite forward experiment, initialization-only
diagnostic, and corrected small- and large-step profile transport checks
are complete.
These assess the new formulas at the existing full-time benchmark state;
they do not establish optimization convergence. This update only archives
the supplied GPU results and computes scalar comparisons locally; it runs
no local physics and changes no solver or objective implementation.

The new optimization-start setup passed the targeted 45-test suite with
59 subtests. Its additional tests check effective config parity against
the actual example's source/defaults, reject all-positive baselines without
calling them root-free, check the pre-crossing mean-field derivative on
short arrays, and verify the original geometry context and terminal gate
through mocked workers. The tests run no VMEC, NTX database, transport or
physical optimization.

## Optimization-start forward results received 2026-10-25

The supplied output reuses optimization_start/ad_summary.json and completes
all four local/annular, profiles/geometry forward checks at trust step 0.001.
The checker proceeds past its forward-versus-AD baseline gate and reports no
failed endpoints. The raw output, including the physical gradients, is saved
in optimization_start_descent_2026-10-25.txt; derived scalar comparisons are
in optimization_start_descent_2026-10-25.csv. These files archive user-run
GPU results. No local physics or new derivative test was run to analyze them.

The starting target field reconstructed from the common target-zero term is
-15.17613522331 kV/m; the starting inner mean is -6.40029809590 kV/m. Neither
candidate creates a positive-Er region or a downcrossing at this step. Every
endpoint has zero positive inner fraction and a negative annular maximum.
Root-position changes and derivatives are correctly unavailable, rather
than evidence of a zero physical root sensitivity before a root exists.

| Candidate/group | Predicted cost change | Observed cost change | Inner mean Er change (kV/m) | Electron region created |
| --- | ---: | ---: | ---: | --- |
| Local/profiles | -0.139034329 | -0.138399310 | +0.007885898 | No |
| Annular/profiles | -0.116052286 | -0.115580225 | +0.007752955 | No |
| Local/geometry | -4.600034696 | -12.384914379 | +0.156838233 | No |
| Annular/geometry | -3.286547120 | -10.834534241 | +0.161021861 | No |

The profile cost slopes agree with AD to 0.4567% (local) and 0.4068%
(annular), relative to the AD prediction. Their target-field increases,
+0.02802652 and +0.02807675 kV/m, also agree with the predicted +0.02810166
and +0.02815128. The profile descent directions have cosine 0.999814 in
the saved trust coordinates (angle 1.1049 degrees). These candidates
therefore provide almost the same local profile control at this state.
The shared target-zero term accounts for 76.76% of the local profile cost
reduction and 92.08% of the annular profile reduction. The annular choice
has not demonstrated better electron-region creation in this experiment.

Geometry lowers both costs, but its reductions are 2.69235 and 3.29663
times their linear AD predictions. The observed target-field increases
(+3.11540 and +3.11344 kV/m) substantially exceed the predicted +0.79699
and +0.79716. In contrast, the observed inner-mean increases are within
about 2% of their predictions (+0.15397 and +0.15821). Thus the geometry
discrepancy is particularly strong at the target, rather than uniform
across the field observables. Initial Er changes by up to 4.165/4.172 kV/m,
without initial sign changes. This single finite step cannot distinguish
curvature, branch selection, and a derivative/forward mismatch; it does
not establish an incorrect geometry derivative or a branch switch.

The next focused check can reuse this AD report and the saved baseline:
take the annular geometry direction at steps 0.0001 and 0.00001 and check
whether its observed slope approaches the AD value. There is no need to
repeat the already matching profile test or recompute AD for that check.
Successful infinitesimal descent alone would still leave sustained
electron-region creation untested. A later optimization comparison should
judge actual positive inner-field formation and resolved downcrossings,
alongside preservation of the negative outer field, rather than cost alone.

### Smaller geometry steps received 2026-10-25

The user completed the focused annular geometry check, reusing the saved
AD report and matching cached baseline/0.001 endpoint. The additional
0.0001 and 0.00001 endpoints both completed finite full adaptive solves.
The supplied output is archived in optimization_start_geometry_steps_2026-10-25.txt;
the derived comparisons are in optimization_start_geometry_steps_2026-10-25.csv.

| Trust step | Observed cost directional slope | AD prediction | Relative difference |
| ---: | ---: | ---: | ---: |
| 0.001 | -10834.534241 | -3286.547120 | 229.6631% |
| 0.0001 | -2938.245504 | -3286.547120 | 10.5978% |
| 0.00001 | -3234.821593 | -3286.547120 | 1.5739% |

Relative difference here is abs(observed - AD) / abs(AD). These are
one-sided finite moves with fresh adaptive schedules, not a central or
frozen-schedule FD Jacobian check. The shrinking-step agreement supports
the local annular geometry directional derivative; it does not establish
all geometry Jacobian entries or eliminate the remaining 1.57% difference.
The much larger response at 0.001 is consistent with strong finite-step
nonlinearity. It is not evidence, by itself, of an incorrect AD derivative.
No specific initial-root branch switch has been demonstrated.

The inner-mean Er changes likewise approach the predictions: at 0.0001,
observed +0.014006221 versus predicted +0.015820729 kV/m; at 0.00001,
observed +0.001561299 versus predicted +0.001582073 kV/m. The target-field
increases are +0.071391768 and +0.007845200 kV/m, versus predictions of
+0.079716110 and +0.007971611. Initial Er max changes are 0.030141836
and 0.006580411 kV/m, with no initial sign changes. Accepted step counts
match the baseline for these two endpoints; matching counts do not prove
identical adaptive schedules or Newton histories.

Every endpoint still has negative Er throughout the chosen annulus, zero
positive inner fraction, no downcrossing, and an unchanged wrong-sign
fraction. Infinitesimal cost descent has now been checked for profiles
and for this geometry direction. No winner for electron-region creation
has been demonstrated. The next useful comparison is sustained descent
from the no-root start, with gradients recomputed at accepted points,
backtracking, and actual sign/root creation recorded. The older annular
profile pilot explicitly requires an existing crossing and must not be
used unchanged for this no-root experiment. Further smaller-step checks
alone would not answer which loss creates an electron-root region.

Only scalar analysis and documentation updates were performed locally.
There were no local VMEC, NTX, transport, AD or optimization runs, and no
solver or cost implementation changes in this archival update.

## Root-creation optimization comparison prepared 2026-10-25

examples/optimization/optimize_er_root_candidates.py now compares sustained
optimization from the verified all-negative optimization-start AD report.
It reuses the existing pilot's host descent/backtracking loop. Each selected
candidate and DOF group starts independently from the exact same original
point. The root weight is one, each scalar cost is minimized directly, and
coordinate scales stay fixed to the saved report. There is no geometry warm
start, accepted-step prefix, alternate solver or extra creation objective.

The pilot now accepts an explicit candidate/group and --allow-root-creation.
This experiment enables that option: finite, cost-decreasing full-time trials
are eligible before a downcrossing exists. Crossings may also disappear at a
later lower-cost point; their appearance/disappearance is observed rather
than enforced as an extra acceptance constraint. The older annular pilot's
default existing-crossing behavior is retained for its original command.

Every eligible trial gets a fresh AD evaluation at its actual parameter
vector before acceptance. Nonfinite or wrong-point AD, incomplete forward
solves, inconsistent AD/independent-forward terminal fields, and insufficient
cost reduction trigger backtracking. The original seed AD is reused.
Matching forward endpoints from the descent checker and completed forward
workers from the same optimization directory are reused, including after
resume. Changed source/input digests invalidate reuse.

For repeated AD, the worker preserves the initial experiment's exact geometry
context (18/18 Boozer resolution and its original surfaces), geometry column
order, five creation-audit rows, full-time transport configuration, fixed
profile configuration and 500 accepted-step reverse guard. Geometry trials
explicitly permit nonzero boundary parameters in the existing production
table-result builder; the ordinary benchmark override still preserves
geometry by default. No reverse-kernel/replay implementation was changed.

Accepted history, events and summary report both cost components, actual
off-grid downcrossings, root error when available, target Er, inner mean Er,
positive inner fraction and negative outer fraction. Crossings are measured
from the concrete terminal field, never from the missing-root diagnostic's
placeholder. Target success requires a downcrossing within tolerance and
strictly positive inner / negative outer signs. Zero or all-positive fields
do not meet that success condition. Budget exhaustion and a stationary point
without the requested root remain distinct from convergence.

The comparison writes independent local_profiles/annular_profiles and/or
local_geometry/annular_geometry subdirectories, each with checkpoints,
history.csv, events.jsonl, summary.json and worker logs. comparison.csv/json
record the first accepted iteration with a positive inner region and the
first with a downcrossing, as well as final signs/root/status. The SVG compares
initial/latest terminal Er and sign-fraction history. Root creation and final
root/sign quality are the comparison criteria; raw costs have different
numerical scales across the two formulas.

After syncing the implementation to the GPU machine, run all four cases:

```bash
cd ~/NEOPAX
mkdir -p outputs/new_root_candidates
set -o pipefail

env -u JAX_COMPILATION_CACHE_DIR \
JAX_ENABLE_COMPILATION_CACHE=0 \
XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer=" \
PYTHONPATH="$HOME/VMEX:$HOME/NTX/src${PYTHONPATH:+:$PYTHONPATH}" \
python examples/optimization/optimize_er_root_candidates.py \
  --ad-report outputs/new_root_candidates/optimization_start/ad_summary.json \
  --descent-dir outputs/new_root_candidates/optimization_start/descent \
  --out-dir outputs/new_root_candidates/root_creation_optimization \
  --groups profiles,geometry \
  --max-iterations 10 \
  --initial-step 0.001 \
  --max-step 0.05 \
  --growth-factor 2 \
  2>&1 | tee outputs/new_root_candidates/root_creation_optimization.log
```

For the profile issue alone, select --groups profiles (the script default).
Use --candidates annular or local to select one case. The accepted-iteration
budget is a pilot limit, not an inference that a candidate cannot create a
root. Add --resume with otherwise matching settings and a larger
--max-iterations value to extend existing runs without restarting their
accepted trajectory. Each case always starts from the shared seed unless
explicitly resuming that case's checkpoint.

This preparation does not fix compilation overhead: physics evaluations
still use isolated workers and an accepted trial normally needs independent
forward and AD solves. The baseline AD is not recomputed. At preparation
time, no root-creation optimization result had been supplied; partial
progress is recorded below.

The final targeted local suite passed 56 tests and 59 subtests in 14.87 s.
Tests use analytic fields, short arrays and mocked physics workers. New
checks cover root-free accepted trajectories, recomputed gradients, actual
root formation, the two independent candidate seeds, geometry-only moves,
correct stationary/target statuses, zero/all-positive false-success cases,
preserved AD geometry context, nonfinite diagnostic serialization, cached
forward reuse and checkpoint resume. No local VMEC, NTX database, transport,
physical reverse AD or physical optimization was run.

### First root-creation optimization progress received 2026-10-25

The GPU run has started local/profiles, the first of the four independent
cases. Both baseline and first trial forward endpoints were reused from
the checker. Iteration 1 completed finite matching AD and was accepted:
cost 41.43344461072115 -> 41.295045300924556, step 0.001, reduction ratio
0.9954326449558978. The inner mean Er increased from -6.400298095898477 to
-6.392412198060221 kV/m. There is no positive inner region or downcrossing.

Iteration 2's independent forward trial at step 0.002 reached cost
41.0220243009842, inner mean Er -6.376993919691591 and target field
-15.09253029132828 kV/m. Its measured forward cost reduction is 0.2730210,
versus predicted 0.2755325 (ratio 0.9908849). The next supplied excerpt
confirms finite matching AD and acceptance of this second update. It is
marked accepted in root_creation_local_profiles_progress_2026-10-25.csv.
The next forward worker, evaluation_0004_forward.log, then completed the
third trial at step 0.004. Its cost is 40.49079470185608, with target field
-14.983337262135926 and inner mean -6.347421931015971 kV/m. The actual
cost reduction from accepted iteration 2 is 0.5312295991 versus predicted
0.5410593936 (ratio 0.9818323). The next supplied event confirms finite
matching AD and acceptance of iteration 3, now marked accepted in the CSV.
The evaluation_0006_forward.log worker has started the fourth trial; no
fourth-point field metrics or acceptance have been supplied. The positive inner fraction remains zero,
outer negative fraction one, and there is still no downcrossing.

The third forward trial's cumulative cost reduction is about 2.28%, with
a target Er increase of 0.1927980 kV/m and inner mean increase of 0.0528762
kV/m from the original state. T0 has increased by about 2.33%. All three
updates now have supplied acceptance events with finite matching AD.
The configured fourth trial step is 0.008. No rerun with a larger initial
step is needed to reach it. The run remains incomplete, with no observed
positive inner region or downcrossing at the last accepted point.

Across the first two accepted moves the mean inner field rises by 0.0233042
kV/m and the target field by 0.0836049 kV/m. Positive inner fraction is
still zero, outer negative fraction is one and wrong-sign fraction is
unchanged at 0.5348837209302325. This is early cost descent toward less
negative inner Er, not demonstrated electron-region creation. A null
root/error correctly records that no crossing exists.

The initial step has grown from 0.001 to 0.002 as configured. These are
successive optimized points with a recomputed gradient, not repetitions
of the original fixed-direction descent check. Every geometry parameter
remains zero in this profile-only case; subsequent geometry cases start
independently from the shared original seed. No other case or final
convergence result has been supplied. Only scalar analysis and archival
updates were performed locally.

### Larger profile updates received 2026-10-25

The next excerpt confirms acceptance of local/profiles iterations 4 and 5
at steps 0.008 and 0.016, with reduction ratios 0.9639724260 and
0.9315217209. No trial rejection or nonfinite AD failure is shown. The
sixth forward trial at step 0.032 completed its cost check and started
evaluation_0011_ad.log; acceptance of iteration 6 is not yet supplied.
The CSV has five accepted updates and one explicitly pending forward
trial. The supplied excerpt is saved in
root_creation_local_profiles_progress_2026-10-25.txt.

| Quantity | Initial | Accepted iteration 5 | Iteration 6, AD pending |
| --- | ---: | ---: | ---: |
| Local cost | 41.43344461 | 37.67956655 | 34.65432407 |
| Target Er (kV/m) | -15.17613522 | -14.38769327 | -13.73685031 |
| Inner mean Er (kV/m) | -6.40029810 | -6.18974958 | -5.97333194 |
| T0 (keV) | 17.8 | 19.61952478 | 21.40402179 |
| n0 | 4.21 | 4.08658155 | 3.89877786 |
| Positive inner fraction | 0 | 0 | 0 |
| Negative outer fraction | 1 | 1 | 1 |

The accepted iteration-5 cost is 9.06% below the initial cost; T0 has risen
10.22% and n0 fallen 2.93%. Trial 6 has a 16.36% cumulative cost decrease,
T0 +20.25% and n0 -7.39%. Its actual forward cost reduction is 3.02524248
versus predicted 3.37750533 (ratio 0.89570324). Both zero and sign costs
fall: at trial 6 they are 23.58763206 and 11.06669202, respectively.

The profile changes are now substantial, while the observed inner field
remains negative everywhere in the test annulus. Small initial updates
alone therefore no longer explain the distance from root creation. This
trajectory shows continuous field/cost improvement with no demonstrated
positive region, not evidence that AD has failed or that profile control
is universally ineffective. The current candidate is still local/profiles;
there are no supplied results from the three other cases. A winner and
final root-creation outcome remain unresolved. If iteration 6 is accepted,
its reported ratio schedules the next trial at the configured cap 0.05.
No extra physics or derivative tests were run locally.

## Profile comparison on the geometry-only optimized boundary: 2026-10-25

The user proposed fixing the boundary to the accepted geometry-only result
instead of the original unoptimized geometry. This is a useful control for
the profile-sensitivity question. The incomplete original-geometry profile
trajectory does not establish that its geometry can never support the desired
root; it only records the tested loss direction and profile path. A root
already present on the optimized geometry makes the new experiment a
refinement/control test, not evidence of root creation from an absent region.

The comparison runner now accepts --geometry-result-dir as an alternative
to --ad-report. It reads geometry_phase_summary.json, or the standalone
optimization_summary.json, and uses the exported accepted geometry input:
accepted_geometry_input for a saved warm geometry phase, or
input.QI_neopax_database_full_transport_net_power_optimized (with the final
stage export as a fallback) for the standalone example. It does not use
the original seed or reapply saved scaled coordinates to an already updated
boundary. A missing accepted export or a result containing profile DOFs is
an error. Relocated warm-phase exports can be resolved inside the result
directory by their recorded basename.

Saved config snapshots are preserved when available. The standalone geometry
summary does not store a complete config snapshot; in that format the current
geometry example supplies nominal profiles/config and the saved mesh, final
time, attempt budget and reverse guard override its defaults. No profiles
from the in-progress original-geometry optimization are loaded. The effective
config, accepted input and summary hashes are recorded in the baseline case
manifest. There is no geometry-only rerun, and geometry stays fixed throughout
the recommended profile-only comparison.

The runner prepares a new baseline full-time AD report for this different
physical boundary, saves it under baseline/ad_summary.json and reuses it on
later invocations only when the geometry input and summary still match.
It observes the actual baseline signs and crossing instead of applying the
original all-negative-field gate. Initial root presence/position/error are
included in comparison.csv; a crossing present at iteration zero is not
marked as newly created. If the baseline already meets the root/sign target,
the pilot correctly reports that instead of forcing a profile change.

After syncing these changes, run the two profile-only loss comparisons:

```bash
cd ~/NEOPAX
mkdir -p outputs/new_root_candidates
set -o pipefail

env -u JAX_COMPILATION_CACHE_DIR \
JAX_ENABLE_COMPILATION_CACHE=0 \
XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer=" \
PYTHONPATH="$HOME/VMEX:$HOME/NTX/src${PYTHONPATH:+:$PYTHONPATH}" \
python examples/optimization/optimize_er_root_candidates.py \
  --geometry-result-dir outputs/geometry_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_optimization \
  --out-dir outputs/new_root_candidates/profiles_on_optimized_geometry \
  --groups profiles \
  --max-iterations 10 \
  --initial-step 0.001 \
  --max-step 0.05 \
  --growth-factor 2 \
  2>&1 | tee outputs/new_root_candidates/profiles_on_optimized_geometry.log
```

The two candidates start independently from the same accepted boundary and
nominal profiles. Matching independent forward endpoints can be shared across
the candidate runs. Each accepted-point AD retains the baseline geometry
context and fixed profile configuration. Existing root-free reports and
commands retain their all-negative baseline validation. The saved-geometry
comparison uses a separate directory; its checkpoints cannot be confused
with the original-geometry run.

The local targeted suite passed 64 tests and 59 subtests in 15.51 s. New
tests check accepted-input selection without double-applying geometry deltas,
saved time/grid/budget/config parity, relocated phase exports, rejection of
missing/wrong exports, observing a root already present, honest initial
versus created crossing flags, one-time baseline AD preparation through a
mocked subprocess, and invalidation after the accepted input changes. There
were no local plasma, VMEC, NTX database, transport, physical AD or physical
optimization runs. At preparation time the physical control was pending;
the subsequent baseline discrepancy is recorded below.

## Saved optimized boundary did not reproduce the electron-root baseline

The next supplied GPU output selects the intended exported input under
geometry_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_optimization.
Its SHA256 is c2b492828478b49431d5cd1c87a32281c172f380ae73bd04ebafdbfa7cabca85;
the summary SHA256 is 91acd6ec9fbddb8c3bbdacf470fb445f20354bb6723cd16fd14183f9bc4d7344.
It reports the nominal six profile values, 2 seconds, mesh 25/25/33, 1000
attempts and a 500-step reverse guard. Nevertheless the baseline is all
negative in the test annulus: target Er -6.40327643587 kV/m, inner mean
-3.07654876829 kV/m, annular maximum -0.697229956770 kV/m and no downcrossing.
Local/annular costs are 9.00701668425/6.77754662981. The excerpt contains
baseline AD progress, not an accepted optimization update. Raw output and
extracted setup/metrics are archived as saved_geometry_baseline_mismatch_2026-10-25.txt/json.

The user correctly rejected this as an unverified reproduction of the desired
geometry-only result. Selecting the exported boundary filename did not prove
that the complete numerical configuration was reproduced. Earlier unit tests
verified routing/config metadata and mocked workers, not physical parity
with the saved optimized transport field.

### Concrete setup mismatch and correction

The saved-geometry loader rebased the accepted boundary to zero geometry
deltas. evaluate_geometry_transport_realtime_geometry_least_squares uses
use_runtime_payload=True for an all-zero geometry vector. It therefore reuses
the baseline runtime made by build_runtime_context rather than the trial
runtime made from the optimization's geometry context. Nonzero geometry
updates in the geometry-only optimizer use that trial context, with 18/18
Boozer resolution.

The loader previously left geometry.mboz/nboz unset, even though its manifest
declared the separate derivative context's 18/18 values. The orchestrator's
runtime constructor instead applies the input-dependent defaults: 6*mpol and
2*ntor-1 (36/11 for mpol=ntor=6). Thus the manifest did not guarantee 18/18
for the actually reused transport runtime. The loader now explicitly writes
18/18 into the saved-geometry runtime configuration, and a regression checks
the arguments passed through the real orchestrator runtime constructor with
the expensive builders mocked. The original unoptimized-start path remains
unchanged. Old saved-geometry AD caches lack the new runtime-contract marker
and are rejected rather than silently reused.

The setup also used generic/split initial-state/support selectors and
established root/bootstrap transpose selectors, while the actual geometry
factory uses reduced_zero_rhs/reduced_zero and legacy_sparse. The saved-case
AD argument builder now matches the existing factory bundle, verified by
extracting its actual options from source. These reverse-selector differences
are not, by themselves, evidence explaining the terminal forward field:
reduced_zero_rhs calls the same initial-carry forward function. The changed
runtime resolution is a concrete forward setup discrepancy; its responsibility
for the observed missing electron region has not been physically verified.
Persistent versus benchmark runtime construction and the historical saved
report's solver/config provenance also remain relevant until parity is read.

### Read existing outputs before another solve

audit_saved_geometry_root_baseline.py reads case_manifest.json,
baseline_forward_observation.json and the geometry result's existing
optimized/transport/transport_solution.h5. It prints actual configured and
effective runtime resolution versus the declared context, both terminal
field summaries, crossing locations and max absolute Er difference. It uses
NumPy/stdlib plus h5py to read that file; it never imports NEOPAX, JAX, VMEX
or NTX and performs no transport/geometry/AD solve. It works even if the
baseline AD report is not yet complete.

After syncing, the read-only physics audit command on the GPU host is:

```bash
cd /home/exouser/NEOPAX &&
/home/exouser/jax_env/bin/python examples/benchmarks/audit_saved_geometry_root_baseline.py \
  --geometry-result-dir outputs/geometry_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_optimization \
  --baseline-dir outputs/new_root_candidates/profiles_on_optimized_geometry/baseline
```

The new baseline observer also checks that saved HDF5 reference when it is
available. It writes baseline_saved_forward_parity.json and stops before
the reverse sweep and candidate optimization if final time/grid/Er differ
(Er rtol=1e-5, atol=1e-6). This is a reproduction check, not proof that the
historical saved report used the same solver/dependency versions. Missing
reference files are explicitly reported as unavailable; they do not establish
physical parity. A corrected physical attempt must use a new output directory
instead of resuming the old baseline/checkpoints. Do not rank losses using
the unverified ion-root baseline or claim this geometry cannot support an
electron root from this result.

The focused suite passed 73 tests and 77 subtests in 17.63 s. New tests cover
the real runtime-constructor arguments without a solve, selector parity with
the geometry factory, early baseline mismatch rejection, old-cache rejection,
stored HDF5 time/field selection, actual crossings, input-dependent Boozer
defaults, and an isolated audit import proving no physics packages are loaded.
Only short arrays, small stored files and mocked workers ran locally. The
physical baseline discrepancy was still awaiting the stored-output audit at
that point; these tests do not establish that the corrected setup reproduces
the root. The subsequent remote audit follows.

### Stored-output audit received on 2026-10-26

The supplied audit is archived in saved_geometry_output_audit_2026-10-26.txt.
Both fields reach t=2 on the same grid. The existing geometry-only HDF5 has
positive Er throughout the tested inner region, negative Er throughout the
outer region, and a downcrossing at rho=0.5174235896040508. Its target Er is
+5.457708927447452 kV/m and inner mean is +23.630288155087445 kV/m. The
candidate baseline has no downcrossing, target Er -6.4032764358708425 kV/m,
and inner mean -3.07654876829172 kV/m. The maximum absolute field difference
is 43.9112910165388 kV/m. Thus the candidate experiment did not reproduce
the stored electron-root result. These candidate losses cannot be ranked on
the assumption that it did.

The saved manifest leaves runtime mboz/nboz unset while declaring the separate
context's 18/18 resolution. Applying the defaults to the current input gives
36/11. Explicit 18/18 configuration is already included in the loader fix;
this audit reads the old baseline, so it does not test that corrected forward
setup.

There is also an input-provenance discrepancy. The earlier baseline recorded
SHA256 c2b492828478b49431d5cd1c87a32281c172f380ae73bd04ebafdbfa7cabca85;
the audit reads 189dc88618cac19e9ec9bf91c312c27ef989f5e14951359d9ccb9ad618db3bf9
from the same path. The file bytes changed. This does not by itself prove a
boundary change, since formatting/comments also affect a hash. Neither the
historical effective resolution nor causal responsibility for the field
difference is established from the changed input alone.

The offline audit now prints source_provenance, comparing the baseline's
recorded input and summary hashes with the current files. It also reports
the input copy beside the saved HDF5 at
optimized/transport/input.QI_neopax_geometry_full_transport_optimized and
whether its bytes match the current baseline input. Missing hashes or files
are explicitly unknown or unavailable. The audit still imports no physics
packages and starts no solver. Source identity should be checked before a
fresh baseline run, using a new output directory. The existing pre-reverse
gate then requires time/grid/Er parity before any candidate optimization.

The user then clarified that the geometry-only optimization had been rerun
and its newly optimized result again has an electron root. That explains
why the input at the same output path changed. The old candidate baseline
must not be resumed or interpreted as a reproduction of this newer result.
The requested next experiment uses the current exported optimized boundary
from that directory and a fresh baseline/output directory. No further audit
of the old baseline is required to start that experiment. Both local and
annular losses, with profiles and geometry varied in separate runs, start
from this same current result after the existing forward-parity check.

### Fresh saved-geometry baseline reproduced the electron root

The next supplied GPU excerpt is archived verbatim as
optimized_geometry_rerun_baseline_2026-10-26.txt. It uses the fresh directory
outputs/new_root_candidates/optimized_geometry_rerun_20261026, input hash
189dc88618cac19e9ec9bf91c312c27ef989f5e14951359d9ccb9ad618db3bf9 and summary
hash e358b1a9f7ff630516e5b60d529cf3c987f526cb5294939b0b9a04c8b671f4da.
The new saved-geometry runtime contract is present, with explicit 18/18
resolution and the geometry factory's reduced_zero/reduced_zero_rhs,
legacy_sparse and grouped_joint_vjp selectors.

The baseline has positive Er throughout the inner test region and negative
Er throughout the outer test region. Its downcrossing is at
rho=0.5174235897859059 versus 0.5174235896040508 in the saved HDF5, a difference
of about 1.82e-10. Target Er is +5.457709756549644 versus +5.457708927447452
kV/m, an absolute difference of about 8.29e-7 kV/m. Inner mean Er is
+23.63028927047991 kV/m. The run continues through final-objective cotangents
and the segmented reverse sweep after the terminal observer, so the saved
forward-parity gate did not reject this baseline. The old all-negative
baseline is no longer the starting point for this experiment.

Both candidate costs are initially 3.723324473342151, entirely dominated by
the shared target-zero term. The local/annular sign contributions are only
2.22e-26/2.82e-26 because the requested signs already hold. Equal starting
costs are expected here; this observation does not rank the two losses.
The existing root's distance from the target 0.514 is about 0.00342359.

The initial reverse sweep processes 179 accepted steps in four segments
(50+50+50+29). The first reverse segment takes 521.285 seconds, with later
segments taking approximately 10 seconds each, consistent with substantial
first-use compilation/setup overhead. The supplied excerpt contains no
traceback, nonfinite report, failed comparison or accepted optimizer update.
It ends during initial-state/support pullbacks. Full AD completion and the
subsequent four candidate/group optimizations remain unobserved. No local
physics run was performed to interpret this output. Since both the saved
geometry and runtime setup changed relative to the old baseline, this is a
successful reproduction control, not an isolated measurement of the effect
of the Boozer-resolution fix alone.

### Local loss decreases while the root moves away from its target

The longer supplied output is archived verbatim as
optimized_geometry_local_profiles_progress_2026-10-26.txt. Extracted endpoints
are in optimized_geometry_local_profiles_progress_2026-10-26.csv, including
the baseline, five accepted iterations, four rejected iteration-6 trials,
and one iteration-6 endpoint whose AD is still pending. All 150 baseline
Jacobian entries (five objectives by 30 physical parameters) are preserved
in optimized_geometry_baseline_gradients_2026-10-26.csv.

The baseline AD completes with all_finite=True, five rows and 30 columns in
1566.205 seconds. Both losses initially have the same six profile gradients:
n0 +1.6206550680707883, T0 +1.4168639203948246,
density_shape_power -0.09605067300591752,
temperature_shape_power +2.4287410777289704,
density_shape_alpha +0.2595389353382458,
temperature_shape_alpha +0.17410176205436656. The scaled cost-gradient norms
are 98.39673308242583 (profiles) and 251.56117203171567 (geometry, local
candidate). Finite AD is not itself a finite-difference validation.

Only the local/profiles optimization is observed in this excerpt. Its target
is rho=0.514. The accepted results are:

| Iteration | Step | Cost | Root rho | Absolute root error | Target Er [kV/m] |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 3.72332447334 | 0.517423589786 | 0.003423589786 | +5.45770975655 |
| 1 | 0.001 | 2.54132162185 | 0.497803997032 | 0.016196002968 | -2.69784772993 |
| 2 | 0.002 | 2.52376773032 | 0.497792273779 | 0.016207726221 | -2.68747295184 |
| 3 | 0.004 | 2.48751990365 | 0.497768501968 | 0.016231498032 | -2.66600395627 |
| 4 | 0.008 | 2.40923105292 | 0.497715089857 | 0.016284910143 | -2.62033072047 |
| 5 | 0.016 | 1.52073710218 | 0.477939962401 | 0.036060037599 | -2.07210007171 |

Thus cost falls 59.1565 percent but root error grows by a factor of 10.5328.
The root moves substantially under profile DOFs in this saved geometry;
weak or absent profile response does not describe these accepted steps.
At accepted iteration 5 n0 is 10.4661 percent below its nominal value and
T0 is 4.0320 percent lower. Both local zero and sign contributions decrease
from accepted iteration 1 to 5 while the crossing error increases. The local
candidate therefore does not reliably order these profiles by root-position
error. The experiment optimizes only the candidate root cost, so the loss's
finite-step behavior is isolated from competition with other objectives.

At the first step AD predicts delta-rho=-1.4213463127761531e-5. The observed
change is -0.01961959275424716: about 1380 times larger and 1.000599 times a
1/51 radial spacing. Iteration 5 moves inward another 1.008532 spacings.
These near-cell-sized moves warrant examining the stored initial and final
Er arrays. They do not alone prove grid pinning, an initial selected-root
branch switch, a solver error, or an AD error. The initial branch-conditioned
derivative cannot predict a jump between branches from this evidence alone.

Iteration 6 rejects step sizes 0.032, 0.016, 0.008 and 0.004 because costs
1.70364702440, 1.97302823442, 2.09594465805 and 2.15347144369 exceed the current
1.52073710218. All four rejected roots are around 0.4973--0.4975, closer to the
target than the accepted 0.47794 root. This directly shows disagreement
between scalar-cost acceptance and actual root-position improvement.
The reduced step 0.002 has cost 1.50974982370 and root 0.477929045904; its AD
has started, but acceptance is not shown. The rejection messages are ordinary
cost backtracking, with no nonfinite-AD failure reported. Subsequent local
iterations and all geometry/annular optimization results remain unobserved.

The existing analyze_er_root_saved_profiles.py can read these saved worker
JSON files without a transport or AD solve. Its output includes crossing
cells, field contrasts/slopes, stored initial-Er sign changes, both cost and
actual-crossing gradient directions, and independent forward/AD endpoint
differences. A same-cell decomposition is only applied when the target and
crossing share that cell; it cannot be used across the observed jumps.
This read-only analysis can run alongside the ongoing GPU job:

```bash
cd /home/exouser/NEOPAX || exit 1
neopax_case=outputs/new_root_candidates/optimized_geometry_rerun_20261026
/home/exouser/jax_env/bin/python examples/benchmarks/analyze_er_root_saved_profiles.py \
  --ad-report "$neopax_case/baseline/ad_summary.json" \
  --run-dir "$neopax_case/local_profiles" \
  --out-dir "$neopax_case/local_profiles/saved_profile_analysis" \
  --plot
```

The acceptance rule is intentionally left unchanged for this diagnostic
comparison. Adding a requirement that actual root error decrease would
change the experiment and would not validate the candidate loss itself.
No new expensive physics run was performed locally.

### Stored gradient directions confirm local position misalignment

The user ran the saved-profile analyzer on the remote machine. Its printed
output is archived verbatim as
optimized_geometry_saved_profile_analysis_2026-10-26.txt. Eleven concrete
crossings are tabulated in optimized_geometry_saved_crossings_2026-10-26.csv;
24 candidate/group directions at six AD points are preserved in
optimized_geometry_saved_candidate_directions_2026-10-26.csv. These are
evaluations along the local/profiles trajectory, not independent results
from the annular or geometry-only optimizations.

The interpolated crossing changes from bracket 25, spanning rho=0.5 to
0.5196078431372548, to bracket 24 (0.48039215686274506 to 0.5) at the first
accepted point, and to bracket 23 at accepted iteration 5. The fractional
position within the crossing cell stays near 0.889, 0.888 and 0.875,
respectively. Each observed endpoint has one downcrossing in the annulus.
Thus the large reported shifts reflect changes in the terminal front's
crossing cell, not selection between multiple simultaneous terminal roots.
The supplied plot corroborates the lower positive-region amplitudes and
the inward movement of the front. Its baseline-centered zoom omits the
iteration-5 crossing near 0.478; that crossing is present in the full-domain
plot and saved crossing table.

At the first accepted profile point rho_root=0.497803997 is below the target
0.514, so a locally improving direction must increase rho_root. Instead,
the saved AD predicts these root rates for cost descent:

| Candidate | DOF group | d rho_root / d trust-step | Alignment with decreasing root error |
| --- | --- | --- | --- |
| local | profiles | -0.00581170040823 | -0.854809815 |
| annular | profiles | -0.00522191576482 | -0.768061761 |
| local | geometry | -0.0191084400029 | -0.711614059 |
| annular | geometry | -0.0111001303222 | -0.413378004 |

Both candidate descent directions therefore worsen actual root error
locally at this particular saved profile. This is stronger evidence of loss
misalignment than a finite-step overshoot alone. Annular behavior is not
uniformly adverse at every saved point: at accepted iteration 5 its profile
direction has a small positive alignment of 0.0131673446, while the local
profile direction remains adverse (-0.3471845124). No claim is made here
about the completed independent annular optimization.

The local/profiles gradient at iteration 1 predicts delta-rho=-1.16234008165e-5
for the next step of 0.002. The observed delta is -1.17232522581e-5, within
0.8591 percent. At iteration 2, the next 0.004 step predicts
-2.37878557248e-5 versus observed -2.37718112108e-5, within 0.06745 percent.
These are finite directional comparisons on adaptive forward endpoints,
not symmetric FD validation of the complete Jacobian. They support the
interpretation that the local AD is describing the same adverse motion,
rather than a nonfinite reverse failure causing these two accepted moves.

At the baseline, an exact same-cell differential decomposition attributes
31.4194 percent of the profile direction's target-Er change to crossing
motion and 68.5806 percent to a change in the local field slope. For geometry
the corresponding fractions are 33.7013 and 66.2987 percent. After the front
has moved inside the target radius, the target lies on the negative branch.
Making that branch less negative can reduce the shared target-zero cost
while the crossing moves farther inward. The crossing-cell slope decreases
from 1594.1483 to 1138.7065 kV/m per unit rho between the baseline and
accepted iteration 5. A same-cell target/crossing decomposition is not
valid at those later points because the target is outside their crossing
cell. The plotted shapes and the saved candidate directions support the
amplitude-versus-position interpretation without extending that identity
outside its domain.

Increasing the overall weight cannot change a candidate's descent direction.
A suitable next loss needs a position term insensitive to uniform field
amplitude scaling when a crossing exists, while retaining a creation signal
when no electron region exists. The already-recorded actual-crossing
Jacobian can assess such a position term locally before another physics
benchmark is requested. Whether profiles can continuously attain the exact
target on this radial grid remains unverified; a position loss would order
the observed jumps correctly but cannot guarantee physical attainability.
The origin of the first front-cell jump is also unresolved: the printed
output does not include the stored initial-Er sign-change diagnostics.
No solver, replay, optimization acceptance rule or candidate formula was
changed to produce this analysis.

### Radial-resolution hypothesis raised by the user

The user proposed that the observed behavior may arise from radial-grid
resolution. This is a credible and unresolved contributor, and the candidate
assessment above is conditional on the current discretization. It does not
show that the same adverse directions or trajectory persist after radial
convergence. A grid check should precede any general rejection of the losses
or claim that the requested position cannot be attained.

Source inspection confirms geometry.n_radial=51 in
full_transport_database_vacuum_t2.toml. The VMEX-to-NEOPAX geometry constructor
builds n_r+1 uniform faces on [0,1] and midpoint cell centers, so the spacing
is 1/51=0.0196078431. The first root shift is -0.0196195928, approximately one
cell. The steep positive-to-negative transition spans adjacent centers in
the supplied plot, and its fractional crossing location changes much less
than its crossing-cell index. These facts support investigating an
under-resolved front, but do not demonstrate its continuum width or prove
that radial discretization causes the branch changes.

The radius diagnostic linearly interpolates the zero between adjacent Er
values. It does not round it to a grid point, and the spacing is not a hard
lower bound on possible interpolated motion. The initializer chooses
ambipolar roots at radial sample locations. The subsequent Er evolution is
not a collection of independent radius-only solves: ElectricFieldEquation
includes finite-volume radial diffusion as well as the ambipolar source,
with DEr=2.0 in this configuration. Its transition therefore needs a spatial
resolution check. Neither the initializer nor the evolution has yet been
identified as the origin of the first front-cell jump.

The smallest useful physical follow-up is forward-only: evaluate the same
saved optimized boundary and two fixed profile vectors (the baseline and
first accepted local/profiles point) at 51, 101 and 201 radial cells. Reuse
the completed 51-cell endpoints where their complete inputs match. Hold
the physical profile parameters, diffusion coefficient, final time, solver
tolerances, Boozer resolution, NTX database radii and angular/velocity mesh
fixed. Rebuild the analytical initial profiles and selected-root initializer
on each grid; interpolating the old Er output onto a denser grid would not
test the PDE discretization. No optimization or reverse AD is needed for
this first check. Compare the initial and final crossing cells/radii,
physical transition width, target Er and changes between the two profile
vectors. Extend refinement only if those values fail to converge.

The local candidate's two sign probes are the grid nodes immediately beside
the target, so changing n_radial also changes their physical positions.
Its raw costs across resolutions alone cannot isolate spatial-solver error.
Compare actual crossings and Er at the same physical target first, or keep
sign probes at fixed physical radii for a cross-grid loss comparison.
The field-amplitude dependence of the candidate formula remains a separate
mathematical property, but the observed optimization consequences may depend
strongly on resolution. No new physics solve was launched locally, and the
ongoing four-case remote optimization was not modified or restarted.

### Local/profiles progress through accepted iteration 6

The next supplied log, archived as
optimized_geometry_local_profiles_update_2026-10-26.txt, is a continuation of
the original 51-cell optimization rather than output from the new radial
convergence script. The existing progress CSV is updated to include six
accepted iterations, six rejected trials and one iteration-7 endpoint with
AD pending, in addition to the original baseline. The new raw archive
preserves the earlier excerpt separately.

Iteration 6 accepts its fifth trial at step 0.002: cost=1.509749823695421,
root=0.47792904590370633, absolute root error=0.036070954096293684 and cost
reduction ratio=0.9980227370340182. Thus that accepted-point reverse check
completed without a reported nonfinite failure.

Iteration 7 rejects step sizes 0.004 and 0.002, with costs 2.122336000801336
and 2.15204698998079. Their roots 0.4975268217866157 and 0.4975426632934679
are closer to the target than the current accepted root, while their costs
are higher. The next smaller step, 0.001, has cost=1.5042234423010103,
root=0.4779228032365286 and error=0.03607719676347143. Its AD has started but
acceptance is not shown. All shown rejections are cost backtracking.

There are no 101/201-cell endpoints or completed geometry/annular case
results in this supplied excerpt. It reinforces the cost/position mismatch
on the 51-cell trajectory but does not determine its resolution dependence.

### Original root/bootstrap control baseline and first rejected profile moves

The supplied original-control run is archived in
original_root_bootstrap_baseline_2026-10-26.txt. Its baseline input/summary
hashes match the verified optimized-geometry case (189dc886.../e358b1a9...),
and it reproduces the same terminal crossing rho=0.5174235897859059 and
target Er=5.457709756549644 kV/m. The full 179-accepted-step, eight-objective,
30-parameter reverse pass completes with all_finite=True in 1781.243 seconds.
The native momentum-corrected bootstrap row uses joint_local_vjp_upar_only
with the legacy_sparse interpolation transpose. No nonfinite reverse error
is reported.

All 240 raw Jacobian entries are saved in
original_root_bootstrap_baseline_jacobian_2026-10-26.csv; all 150 scalar-cost
gradients in original_root_bootstrap_baseline_cost_gradients_2026-10-26.csv;
ten candidate/control block directions in
original_root_bootstrap_baseline_directions_2026-10-26.csv. Joint-cost rows
are baseline diagnostics, not an additional selected optimization case.

At the baseline:

- Original soft position moment: 0.003423589785905512.
- Original strength: 1.6247977735119645, above target 0.9.
- Original root cost: 0.005860483511078274, dominated by position.
- Bootstrap observable: 0.5380608878578748, above limit 0.1.
- Bootstrap cost: 0.3837946829416591.

The original soft location equals the actual selected chord crossing to
the printed precision. With strength already satisfied, its position cost
is approximately 0.5*1000*(rho_root-0.514)^2 at this baseline and the two
reported trial points. Its block-normalized descent aligns with the
actual-crossing gradient: predicted root rates are -0.014296418496599439
for profiles and -0.03905190803700927 for geometry, equal to the negatives
of the respective scaled root-gradient norms. Thus the original objective
already provides the direct crossing-distance behavior proposed as a control
for this particular branch. No replacement position term is necessary to
obtain that local behavior here; this observation does not establish it for
root-free profiles or multiple competing crossings.

The profile block cost-gradient norm is 0.04894507233977527 for the original
root pair and 30.90231911962548 for bootstrap. This is a weighted-cost scale
difference, not proof that the original root cannot respond to profile DOFs.
The test normalizes each objective's block descent direction. For example,
the original root's first step changes T0 by -0.0638738967252485 keV, similar
to the new local candidate's -0.0624584086280142 keV at the same trust step.
The bootstrap norm is approximately 631.37 times larger, relevant if these
weighted objectives are combined later, but the requested runs are separate.

The original/profiles run correctly rejects its first two finite proposals:

| Status | Step | Original root cost | Actual root | Absolute error |
| --- | --- | --- | --- | --- |
| Baseline | 0 | 0.00586048351108 | 0.517423589786 | 0.003423589786 |
| Rejected trial 1 | 0.001 | 0.131148401396 | 0.497804420270 | 0.016195579730 |
| Rejected trial 2 | 0.0005 | 0.131104484888 | 0.497807132132 | 0.016192867868 |

The trial strengths remain above target at 1.53488696738/1.53581332096, so
the strength costs are only 7.75e-24/7.73e-24. Their bootstrap values improve
to 0.500214398328/0.502613046911, but bootstrap is monitored rather than
included in this original-root run's acceptance cost. The root-pair objective
penalizes the wrong-direction front jumps instead of rewarding them as the
new local/annular losses did on their earlier trajectories.

original_root_profiles_progress_2026-10-26.csv records the baseline and both
rejected endpoints. The excerpt ends as the next forward evaluation begins;
no accepted original-root iteration, geometry-only control or bootstrap-only
optimization is yet observed. Ordinary backtracking should be allowed to
test smaller moves. Whether those remain on the baseline branch and improve
the target error is unresolved, as is the role of radial resolution in the
front-cell jumps. These data support good cost ordering on the observed
original-root trials, not a claim of completed convergence or full AD/FD
validation.

### Original controls: exhausted backtracking and improving bootstrap profiles

Two subsequent excerpts are preserved verbatim as
original_root_profiles_backtracking_2026-10-26.txt and
original_root_geometry_bootstrap_progress_2026-10-26.txt. The profile progress
CSV now contains all eight rejected original-root proposals. A matching
original_root_geometry_progress_2026-10-26.csv records the eight geometry
proposals; bootstrap_profiles_progress_2026-10-26.csv records three accepted
bootstrap iterations and the fourth forward endpoint with AD still pending.

Both original position+strength runs terminate with backtracking_exhausted,
converged=False and zero accepted iterations. Each tests steps from 0.001 to
0.0000078125, a factor of 128. The smallest profile step gives
root=0.49781068525865185, cost=0.13104695589722182; the smallest geometry step
gives root=0.4978104489927187, cost=0.13105078090868794. Both remain about one
51-cell spacing inward of baseline root=0.5174235897859059, whose cost was
0.005860483511078274. The root errors are approximately 0.01619 versus the
baseline 0.00342359. The original soft-selected location continues to match
the actual interpolated crossing to printed precision, and strength remains
above its target. Rejecting these proposals is correct for the requested
root-position objective.

The baseline AD predicts root shifts of only -1.11690769505e-7 for the smallest
profile move and -3.05093031539e-7 for the smallest geometry move. Observed
shifts are approximately -0.0196129045 and -0.0196131408. Shrinking the tested
steps has not recovered the baseline local response. These finite moves do
not validate the root Jacobian at baseline; finite AD alone is not proof of
agreement. It is premature to attribute the mismatch exclusively to radial
resolution, initial-root selection, replay, or an incorrect derivative.

Source inspection confirms that the first original/profiles evaluation is
an independent zero-move forward solve. optimize() compares its terminal
field, grid, final time and selected cost against the AD primal before
testing proposals. That comparison passed; otherwise this log could not
reach backtracking. Geometry and bootstrap reuse this same input-matched
forward baseline. Thus there is no observed zero-move forward/AD mismatch
in this run. A common baseline/worker mismatch should not be claimed from
the nonzero-step jumps alone. In particular, profile-only proposals retain
zero geometry deltas and use the same runtime-construction path as baseline.
The logs do not contain the initial Er/density/pressure arrays or solver
counts for each trial, so they cannot locate where the jump first develops.

Bootstrap profiles behave differently:

| Point | Bootstrap observable | Bootstrap cost | Actual root | Reduction ratio |
| --- | --- | --- | --- | --- |
| Baseline | 0.538060887858 | 0.383794682942 | 0.517423589786 | — |
| Accepted 1 | 0.499613663931 | 0.319382160801 | 0.497803920457 | 2.08439120 |
| Accepted 2 | 0.486106729205 | 0.298156812675 | 0.497790357030 | 0.98479762 |
| Accepted 3 | 0.459824297453 | 0.258947050076 | 0.497761781845 | 0.95642721 |
| Iteration 4, AD pending | 0.397584978982 | 0.177113639432 | 0.478063209937 | Not yet accepted |

The first three accepted steps reduce bootstrap cost by 32.5298% and the
raw observable by 14.5405%; the limit 0.1 is not met. Each acceptance requires
finite accepted-point AD and agreement between its primal and the separate
forward endpoint. The second and third actual cost reductions differ from
their AD linear predictions by approximately 1.52% and 4.36%. These are useful
directional checks at the new bootstrap points, not full Jacobian checks or
validation of the baseline root gradient. The first step crosses the same
front-cell change and its reduction ratio is 2.084. Bootstrap alone does not
penalize the increased root-position error, so acceptance is consistent with
that separate objective. The geometry-only bootstrap run has not started in
these excerpts. No nonfinite AD error is shown.

The next diagnosis uses saved files rather than a new physical experiment.
analyze_er_root_saved_profiles.py now includes initial crossing/bracket,
initial Er/density/pressure changes, accepted/attempt counts and final time
in initial_state_summary.csv. Its console output includes saved-initial-state
records, and it can read the optimizer's wrapped forward-task cache so later
independent groups recover the initial baseline arrays from original_profiles.
Initial sign changes would locate an initializer contribution; a terminal
front change without an initial sign/cell change instead requires examining
the evolution. Neither observation by itself proves that the derivative or
the continuum physics is wrong. No physics settings or production loss
definition were changed. All ten saved-array analyzer tests passed using
NumPy and temporary JSON files only; no local physics solve was run.

### Bootstrap profiles: five accepted steps and forward-worker failures

The next excerpts are archived as bootstrap_profiles_update_2026-10-26.txt
and original_controls_cumulative_update_2026-10-26.txt. The latter contains
the same endpoint/status progression with more original-control history;
neither reports a result after starting evaluation_0014_forward.log.
bootstrap_profiles_progress_2026-10-26.csv is updated without double-counting
overlapping excerpts or gradient_started/accepted records of the same trial.
It now has one baseline, five accepted steps and three failed forward trials.
Unknown costs for failed workers are left empty rather than assigned zero.

Iterations 4 and 5 both complete accepted-point AD and are accepted:

| Accepted iteration | Step | Bootstrap value | Bootstrap cost | Root | Cost reduction ratio |
| --- | --- | --- | --- | --- | --- |
| 4 | 0.008 | 0.397584978982 | 0.177113639432 | 0.478063209937 | 1.09887375243 |
| 5 | 0.008 | 0.345857588616 | 0.120891907760 | 0.458316082559 | 1.05605171199 |

Relative to baseline, bootstrap cost has fallen by 68.5009% and its raw
observable by 35.7215%. The limit 0.1 is still unmet. The positive core
remains present: inner_positive_fraction=0.8695652173913043, with all sampled
outer Er values negative. Strength=1.187908560835505 still exceeds 0.9. The
root has moved inward by 0.0591075072272, about three 51-cell spacings, while
its target error has grown to 0.0556839174413 (16.2648 times the original
error). The weighted position cost alone would be about 1.55035 instead of
the original 0.00586048. Bootstrap-only acceptance does not constrain that
position cost, so this tradeoff is consistent with the selected objective.
These results show substantial profile control of bootstrap on this
trajectory; they do not resolve the baseline root derivative/forward-jump
mismatch or demonstrate convergence of any original-root run.

Three trial evaluations fail inside the forward worker:

- Iteration 5, trial 1, step 0.016: evaluation_0009_forward.log.
- Iteration 6, trial 1, step 0.016: evaluation_0012_forward.log.
- Iteration 6, trial 2, step 0.008: evaluation_0013_forward.log.

At iteration 5, backtracking to 0.008 succeeds. At iteration 6, the excerpt
ends as evaluation_0014_forward.log starts, before a third trial result.
No reverse worker is launched for the three failed forward trials. Their
only reported reason is "forward worker exited 1"; the actual exception is
not in either attachment. It could be in runtime/initialization, transport,
or the terminal observable calculation. It should not be labeled as another
nonfinite reverse-AD failure, an attempt-budget failure, or an OOM without
reading those saved logs. The five accepted steps have passed the existing
finite-AD and independent forward/primal checks.

The requested follow-up reads the last 60 lines of these three logs on the
remote machine; it runs no simulation. Saved initial-state analysis from the
preceding section remains the next check for the separate root-jump issue.
No source code or physics setting was changed for this progress update and
no local physics test was run.

### Profile-only joint-objective continuation, 2026-10-27

The requested existing SciPy stage-2 run from the saved optimized geometry
has completed. Full evidence and a read-only diagnostic command are in
[the joint-objective continuation report](profiles_only_root_bootstrap_net_power_2026-10-27.md).
Total cost decreases by only 0.04893%; the accepted final root is effectively
unchanged at 0.517423569, bootstrap slightly worsens, and net power moves
from 299.890707 to 299.930314 MW. The ftol termination leaves first-order
optimality 18.5. Many completed proposals still jump inward to root near
0.497811 and power near 275.9 MW. The one reported failed solve hits the
500-accepted-step guard before final time, with no reported nonfinite AD.
Device counters/stage-cache sizes stabilize, but host RSS grows by 620.9 MiB;
this finite run does not establish a complete memory fix. The next diagnosis
uses already-saved step models and Jacobians, without a new physics run.

### Independent-species stage-2 partial run, 2026-10-27

[The partial independent-species report](species_profiles_root_bootstrap_net_power_partial_2026-10-27.md)
archives evaluations 4-9 of the expanded-DOF run. Those proposals still show
the same root near 0.49781 and power near 275.8 MW; at evaluation 9 net-power
cost contributes about 98.31% of total cost. No accepted-step notification
or final outcome is included, so these are not reported as optimized results.
The reverse sweeps shown complete without a reported exception. Host RSS
grows by only 11 MiB in this short window, device counters/cache sizes are
nearly flat, and no OOM is reported. A read-only command in the report prints
the already-saved independent parameter labels and trial prediction/actual
reductions while the optimizer continues. No local physics run was made.
