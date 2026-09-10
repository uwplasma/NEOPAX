# Live NTX-database initial-root optimization and memory plan

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
