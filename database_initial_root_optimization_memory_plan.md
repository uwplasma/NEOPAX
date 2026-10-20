# Live NTX-database initial-root optimization and memory plan

## Rerunning the smooth electron-root examples (2026-10-20)

The root-only API now accepts `Er_transition_rho`,
`Er_transition_location_moment`, and `Er_transition_strength`. It calls the
same local objective functions as full transport, on the **selected initial
ambipolar roots**. Both Er cotangents and direct radial-grid cotangents remain
in the reverse calculation. The established exact-Lij and database reverse
stages are retained.

The nine transition examples (vacuum/finite-beta, exact-Lij/database,
with/without bootstrap, plus the profile-only example) replace the two
fixed-cell Er targets with:

- `Er_transition_location_moment`, target zero, weight 1000: move the smooth
  positive-to-negative transition toward `rho = 0.514`.
- `Er_transition_strength_deficit`, target zero, weight 100: smooth positive
  part of `0.90 - Er_transition_strength`. This penalizes a weak or absent
  transition without imposing an upper strength target.
- Zero-weight reporting rows for the selected transition radius and strength.

Their options use the radial window `[0.05, 0.9]`, Er scale 2 kV/m, radial
softness 0.05, softmax beta 16, and smoothing epsilon `1e-6`. Settings are near
the top of each script and saved in its optimization summary. Iteration
reports include the new observables and separate location/strength costs.
Output directories now end in `_electron_root_optimization`, including the
DMerc wrapper that inherits the updated exact-Lij bootstrap example.

The QI, geometry, bootstrap, profile bounds, seed inputs and resolution
settings retain each example's previous defaults. In particular, the
profile-only example still optimizes its maximum-Er target and uses
`input.QI_nfp2_newNT_opt_hires`; the vacuum geometry examples start from
`input.QI_nfp2_initial`. The separate maximum-Er-only examples retain their
maximum-Er objective.

These runs do **not** integrate transport in time, so Radau Jacobian replay,
accepted-step limits and total attempted-step limits do not apply. Pulling
the new source includes current shared flux/root implementations, but the
recent full-transport replay corrections are not part of this root-only
calculation.

After syncing the changes, run on the remote GPU machine in `jax_env`:

```bash
cd ~/NEOPAX
mkdir -p outputs
set -o pipefail

# Geometry + QI + initial electron-root transition.
env XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer=" \
PYTHONPATH="$HOME/VMEX:$HOME/NTX/src" \
python examples/optimization/optimize_geometry_qi_max_er_transition_initial_root_database.py \
  2>&1 | tee outputs/ambipolarity_electron_root_database.log

# Same root-only calculation with the bootstrap-current penalty.
env XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer=" \
PYTHONPATH="$HOME/VMEX:$HOME/NTX/src" \
python examples/optimization/optimize_geometry_qi_max_er_transition_bootstrap_initial_root_database.py \
  2>&1 | tee outputs/ambipolarity_electron_root_bootstrap_database.log
```

For the finite-beta database counterparts, use the corresponding filenames
ending in `_initial_root_database_finitebeta.py`. They retain their existing
finite-beta seed, mode schedule and beta/DMerc switches. For exact-Lij, use
the corresponding filenames without `_database`.

To optimize only the six profile parameters on the profile example's fixed
geometry:

```bash
env XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer=" \
PYTHONPATH="$HOME/VMEX:$HOME/NTX/src" \
python examples/optimization/optimize_profiles_max_er_transition_bootstrap_initial_root.py \
  2>&1 | tee outputs/ambipolarity_profiles_electron_root_bootstrap.log
```

Validation: 37 lightweight tests plus 48 subtests passed. They check root-only
objective values and Er/rho-grid derivatives against independent NumPy finite
differences, non-default option forwarding, transformed residual/Jacobian
assembly and example builders. No VMEC, NTX, ambipolar root, transport or
optimization solve was run locally. Full optimization convergence and memory
behavior with the new objective combination still need remote validation.

### First GPU rerun: repeated-observable stage labels

The first database/bootstrap rerun reached the root-to-database scan fold
(`objective_rows=4`), then failed in the compiled-stage result adapter before
the initial optimizer report. Its metadata contained five labels:
`Er_transition_rho`, `Er_transition_location_moment`,
`Er_transition_strength`, `Er_transition_strength`, and
`bootstrap_current_softmax_abs_scaled`. The result requires unique labels.

The reporting strength and transformed deficit intentionally share one raw
observable. The evaluator already requested four unique backend rows, while
root-stage setup and payload-result setup incorrectly retained both strength
entries. Both setup lists now deduplicate names in first-occurrence order.
The five transport residual rows, their targets, weights and chain rules
remain separate.

Four new lightweight regression cases exercise the real public problem
constructor and result adapter for the three database root-stage variants
and the staged exact-Lij variant. Physics kernels are stubbed; unique table
labels, array shapes, reuse of the strength row and expansion to reporting
plus deficit residuals are checked with production assembly code. The
earlier tests stopped above this adapter and missed the mismatch. The same
remote command can be rerun after syncing this fix; no GPU rerun has been
performed locally.

## Scope

Make the geometry + initial-ambipolar-root optimization path support the
existing live NTX database model, `neoclassical.flux_model = "ntx_scan_runtime"`.
The exact-Lij optimization lane and the benchmark/reference path must remain
unchanged.

The shared reverse infrastructure already provides the required database
mathematics: VMEC state to live scan inputs, database construction, database
cotangents, and one retained database-to-scan transpose.  The missing bridge
is using that current live database in every initial-root optimization
evaluation.

## Implementation

1. Detect `ntx_scan_runtime` in the initial-root optimization evaluator.
   Keep the present exact-Lij support construction selected exclusively for
   `ntx_exact_lij_runtime`.

2. For each optimizer vector, use its current VMEC raw-block state to build
   current geometry, scan channels, and scan surfaces.  Rebuild the matching
   runtime database and use this runtime for selected-root and transport
   objective *values*, not only for their cotangents.

3. Reuse the established database reverse boundary:
   - split the current scan runtime into a fixed database segment and retained
     scan owner;
   - accumulate root/objective bars on `{geometry, database}`;
   - perform exactly one batched database-to-scan transpose;
   - use the existing live scan payload-to-VMEC transpose.

4. Do not route database cases through the exact-Lij staged/JIT payload
   adapter.  A database-specific staged boundary may be introduced only after
   the non-staged current-runtime path is correct.

## Correctness acceptance tests

1. At `x0`, compare database optimization-path residuals and Jacobian against
   the established database benchmark path.

2. Repeat the comparison at a nonzero VMEC boundary perturbation.  This is
   mandatory: it detects accidentally evaluating transport root values on a
   baseline database while reporting derivatives for a perturbed geometry.

3. Cover the active root objectives individually and together:
   `softmax_Er`, `Er_transition_left`, `Er_transition_right`, and
   `bootstrap_current_softmax_abs_scaled` when supported by the database
   runtime.

## Memory and JIT acceptance tests

1. Add a database initial-root repeated-evaluation harness.  Warm once, then
   evaluate one fixed parameter vector repeatedly without SciPy.

2. Report RSS delta, live JAX-array count, JAX dispatch-cache count, and the
   number of database scan folds.

3. Add checkpoints at: raw VMEC solve, live scan/database build, selected
   root, database-to-scan fold, and VMEC payload transpose.

4. Run the harness twice: at fixed `x0` and at one fixed nonzero boundary
   perturbation.  After warmup, neither run may show a continuing RSS slope or
   unbounded dispatch-cache growth.  A one-time compilation/allocation at the
   first evaluation of each fixed shape is acceptable; repeated evaluations
   must settle.

## Deliverables after validation

Create separate, database-specific geometry optimization examples for:

- QI + max-Er;
- QI + max-Er + transition roots;
- QI + max-Er + transition roots + bootstrap penalty.

They must explicitly use the live database configuration.  Existing Lij
examples retain their exact-Lij staged lane unchanged.
