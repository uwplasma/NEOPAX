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
forward and AD solves. The baseline AD is not recomputed. No root-creation
optimization result from the physical machine has yet been supplied.

The final targeted local suite passed 56 tests and 59 subtests in 14.87 s.
Tests use analytic fields, short arrays and mocked physics workers. New
checks cover root-free accepted trajectories, recomputed gradients, actual
root formation, the two independent candidate seeds, geometry-only moves,
correct stationary/target statuses, zero/all-positive false-success cases,
preserved AD geometry context, nonfinite diagnostic serialization, cached
forward reuse and checkpoint resume. No local VMEC, NTX database, transport,
physical reverse AD or physical optimization was run.
