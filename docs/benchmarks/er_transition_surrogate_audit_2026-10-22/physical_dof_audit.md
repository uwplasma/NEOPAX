# Test the new root costs through geometry and profile parameters

The purpose is to determine whether the NEW costs guide physical parameter
changes toward the requested Er downcrossing, or merely reduce amplitude.
No new physical run has yet established that either candidate succeeds.

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
downhill endpoints. Finite moves recompute physical geometry, the database
and selected initial root and use adaptive accepted timesteps. They assess
actual finite physical improvement; they are not the frozen-schedule FD
oracle, which has a separate unresolved replay issue.

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

## Local verification and limits

Only short-array and mocked tests run locally. They check new-cost
derivatives through Er and rho against independent NumPy differences,
off-grid/core behavior, optimizer scaling, amplitude-only directions with
zero root movement, candidate-only routing and forward-worker inputs.
They do not execute VMEC, NTX, transport or optimization.

The GPU experiment remains to be run. It assesses the new formulas at the
existing full-time benchmark state. Useful directions there are a
prerequisite for choosing an optimization experiment, not a convergence
guarantee.
