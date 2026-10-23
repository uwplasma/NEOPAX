# Test the new root costs through geometry and profile parameters

The purpose is to determine whether the NEW costs guide physical parameter
changes toward the requested Er downcrossing, or merely reduce amplitude.
The first full-time physical AD and forward descent runs are saved below.
The forward profile test exposed a test-harness mismatch; corrected checks
are pending, so optimization effectiveness remains unverified.

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

The next targeted check should reuse the existing AD report and test only
profile steps 0.001 and 0.0001 with the corrected checker, in a new output
directory such as outputs/new_root_candidates/descent_ad_contract. The code
change invalidates the old endpoint digests; its baseline must be checked
again. No AD rerun or repeated geometry endpoint is required for this check.

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

## Local verification and limits

Only short-array and mocked tests run locally. They check new-cost
derivatives through Er and rho against independent NumPy differences,
off-grid/core behavior, optimizer scaling, amplitude-only directions with
zero root movement, candidate-only routing and forward-worker inputs.
They do not execute VMEC, NTX, transport or optimization.

The latest targeted suite passed 18 tests and 5 subtests. This includes an
actual short-array profile-construction regression proving the extra frozen
helium temperature dependence, and a mocked initialization probe verifying
one runtime build and no transport invocation.

The GPU AD and original finite forward experiment are complete. The
corrected profile-contract checks remain to be run. These assess the new
formulas at the existing full-time benchmark state; they do not establish
optimization convergence.
