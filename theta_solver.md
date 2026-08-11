# Theta Solver Current Situation

## Goal

Make the theta/Newton solver follow the same transport physics semantics as the updated Radau path. The intended difference is the time integrator residual:

- Radau: multi-stage implicit residual.
- Theta: one-state implicit residual.

The RHS construction, lagged transport response behavior, profile/source physics, and solver configuration should otherwise be TOML-driven and comparable.

## What Was Fixed

- Threaded missing theta flags through the saved-loop and scan paths:
  - `freeze_attempt_linearization`
  - `use_transport_lagged_response`

- Fixed theta controller logic that accidentally called `jnp.logical_or` with three arguments.

- Fixed `theta_jacobian_reuse_mode` semantics:
  - `refresh_each_iteration` now recomputes the Newton linearization.
  - `freeze_attempt` enables frozen/reused attempt linearization.

- Reworked theta predictor/Newton RHS linearization so solver-mode choices are static Python branches instead of `jax.lax.cond` branches that trace unused modes:
  - `theta_rhs_mode = "lagged_response"` differentiates/evaluates `evaluate_with_lagged_response(...)`.
  - `theta_rhs_mode = "lagged_linear_state"` uses the linearized state response.
  - `theta_rhs_mode = "black_box"` uses the raw RHS.

- Added transport summary printing through `[transport_output] transport_print_summary = true`, so silent theta runs report output path, final time, step count, and failure status.

- Fixed the summary printer scoping bug where `run_transport()` local `import jax` branches made `jax.device_get(...)` unavailable to the summary helper.

- Added a wHe theta TOML counterpart:
  - `examples/Solve_Transport_Equations/Solve_Transport_equations_wHe_theta.toml`
  - It is physics-identical to `Solve_Transport_equations_wHe_radau.toml` outside `[transport_solver]` and `[transport_output]`.

## Current Forward Status

- Static checks pass.
- The wHe theta TOML parses and uses:
  - `transport_solver_backend = "theta_newton"`
  - `theta_rhs_mode = "lagged_response"`
  - `theta_predictor_mode = "linearized"`
  - `theta_controller_mode = "current"`
  - `theta_jacobian_reuse_mode = "refresh_each_iteration"`
  - `theta_lagged_response_reuse_mode = "retry_only"`

- The W7X ReLU theta example has summary printing enabled, but it still uses `theta_rhs_mode = "black_box"`. That is a separate ReLU example, not the wHe Radau-parity lagged-response case.

## Current Reverse/Shared-Payload Status

- Radau shared-payload reverse remains the validated production path.

- Theta reverse/shared-payload scaffolding now exists:
  - prepared theta rollout
  - theta execution context
  - theta adaptive schedule trace
  - theta reduced cotangent structures
  - one-step theta implicit residual transpose path

- For theta solvers, `reverse_stage_cotangent_mode = "full"` dispatches internally to the theta implicit transpose path. This keeps the benchmark/API solver-driven instead of requiring a special user-facing theta mode.

- This theta reverse path has passed static checks, but still needs runtime validation before it should be called production-complete.

## Remaining Risks

- Theta Newton currently builds dense one-state residual Jacobians. This should be cheaper than Radau stage Jacobians, but it can still be memory-heavy for large transport states.

- If the wHe theta run fails now, the likely remaining issue is in theta Newton linearization/runtime behavior, not printing and not physics TOML mismatch.

- The theta reverse/shared-payload path must still be checked numerically against forward AD/FD before using it as a benchmark-quality derivative path.

## Commands To Run

Forward wHe theta parity run:

```bash
python -m NEOPAX ./examples/Solve_Transport_Equations/Solve_Transport_equations_wHe_theta.toml
```

W7X ReLU theta run:

```bash
python -m NEOPAX ./examples/inputs/W7X_trinity3d_initial_profiles_relu_theta.toml
```

Known Radau shared-payload reference shape:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --reverse-parameter-mode profiles_plus_realtime_geometry \
  --reverse-geometry-parameter RBC:1:0 \
  --realtime-geometry-gradient-path reverse_payload \
  --objective all \
  --accepted-step-limit 16 \
  --radau-jacobian-reuse-mode legacy \
  --timing-mode jit-warm \
  --reverse-segment-length 4 \
  --reverse-stage-adjoint-solve-mode bicgstab \
  --reverse-rhs-transpose-mode explicit_ntx_interpolated \
  --reverse-step-bwd-mode reduced_cotangent \
  --initial-Er-root-ad jax_selected_root \
  --full-transport-shared-payload-smoke
```

## Validation Run In This Pass

- `python -m py_compile NEOPAX/_transport_solvers.py NEOPAX/_reverse_ad_transport.py NEOPAX/_orchestrator.py`
- TOML comparison confirmed no physics-section differences between wHe Radau and wHe theta.
- `git diff --check` passed with only line-ending warnings.
