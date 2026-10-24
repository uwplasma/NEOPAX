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
reaching the existing reverse evaluator. No real optimization has been run.

The GPU AD, original finite forward experiment, initialization-only
diagnostic, and corrected small- and large-step profile transport checks
are complete.
These assess the new formulas at the existing full-time benchmark state;
they do not establish optimization convergence. This update only archives
the supplied GPU results and computes scalar comparisons locally; it runs
no local physics and changes no solver or objective implementation.
