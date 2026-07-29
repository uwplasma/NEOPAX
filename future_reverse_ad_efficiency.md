# Future Reverse-AD Efficiency

## Current bottleneck

The expensive part of full realtime-geometry transport reverse AD is the Radau
time-evolution reverse sweep, especially the phase reported as:

```text
support reverse segmented cotangent sweep ready
```

This phase walks the realized accepted-step schedule backward, replays the
Radau step data, solves stage adjoints, and accumulates cotangents for the
transport state and geometry/NTX support payload.

The current priority is to preserve the validated benchmark behavior and
numerics. Efficiency work should therefore be benchmark-gated against the
existing FD and reverse-AD comparisons before becoming the optimization default.

## Safe Optimization Directions

1. Add timing diagnostics inside the segmented reverse sweep.
   Measure per-segment and per-step time, separating stage-adjoint solve,
   RHS transpose work, NTX/support payload pullback, and carry propagation.

2. Avoid support payload cotangents when geometry is not active.
   Profile-only optimization should not build geometry/NTX support bars.

3. Restrict objective basis when fewer objectives are requested.
   The all-objective table path is useful for validation and multi-objective
   optimization, but single-objective runs should avoid reversing unused rows.

4. Audit batched objective fusion inside each accepted-step reverse.
   Confirm all objectives share primal stage data, lagged response data, and
   linearization work instead of repeating expensive kernels.

5. Inspect replay payload size.
   Remove fields from segmented replay arrays if the reduced-cotangent backward
   pass does not actually use them.

6. Benchmark segment length choices.
   Compare segment lengths such as 1, 4, 8, and 16 for the 16-accepted-step
   case. Larger segments can reduce overhead but may increase memory.

7. Evaluate alternative Radau stage adjoint solvers.
   Test any structured or preconditioned options against the validated FD
   comparisons before enabling them by default.

## Guardrails

- Do not change forward solver behavior.
- Do not change accepted-step schedule semantics.
- Do not replace the validated grouped reverse path with a direct builder
  unless the direct builder matches values and memory behavior.
- Do not introduce Python loops over objectives as the production strategy.
- Keep profile-only reverse AD as the frozen-geometry limit of the
  realtime-geometry path.
- Every efficiency change should be checked against the saved FD/reverse
  benchmark values for both profile and geometry parameters.

## Suggested First Benchmark

Use the validated internal optimization smoke with the 16-step realtime
geometry case:

```bash
python ./examples/optimization/transport_realtime_geometry_reverse_smoke.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --reverse-geometry-parameter RBC:1:0 \
  --objective all \
  --accepted-step-limit 16 \
  --reverse-segment-length 4 \
  --initial-Er-root-ad jax_selected_root \
  --ntx-exact-derivative-mode direct \
  --ntx-exact-derivative-field-pullback-mode generic_jvp \
  --radau-jacobian-reuse-mode legacy \
  --reverse-stage-adjoint-solve-mode bicgstab \
  --reverse-rhs-transpose-mode explicit_ntx_interpolated \
  --reverse-step-bwd-mode reduced_cotangent \
  --hide-solver-iterations
```

The first efficiency milestone is better timing attribution, not a numerical
change.
