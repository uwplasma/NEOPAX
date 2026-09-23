# Reverse AD Transport Geometry Extension Plan

Date: 2026-07-21

Goal: extend the existing transport reverse-AD profile benchmark so realtime
VMEC geometry harmonics can be differentiated together with profile parameters.
The target outputs are transport-objective derivatives such as
`d objective / d profiles` and `d objective / d VMEC harmonics`, for example
`d softmax_Er / d n0`, `d total_pressure / d T0`, and
`d objective / d RBC:1:0`.

## Hard Guidelines

- Do not contaminate the working forward solver paths.
- Frozen-geometry transport and realtime-VMEC transport must keep their current
  primal behavior and output agreement.
- Build realtime-geometry reverse AD by extending the already-correct profiles
  reverse AD lane, not by creating a separate post-hoc geometry-only transport
  lane.
- In frozen-geometry mode, the new machinery must reduce to the existing
  benchmarked profile-only reverse AD behavior.
- In realtime-geometry mode, the primal must follow the same step map and the
  same numerical geometry construction as the working realtime forward solver.
- Geometry parameters must be inserted into the same reverse parameter machinery
  as profile parameters, with geometry contributions enabled only for realtime
  geometry.
- Do not differentiate "support structures" as objectives. Support payloads are
  cached data needed to evaluate fluxes; the differentiated objects are
  transport objectives with respect to profiles and VMEC harmonics.
- Do not use generic reverse through the full VMEC/Boozer/NTX support-building
  graph. Use compact, operator-paired pullbacks.
- Do not replace reverse values with finite differences or forward JVPs.
  FD/JVP are validation oracles only.
- Do not hide missing cotangents by zeroing them. Missing geometry cotangents
  must be fixed by routing the correct compact pullback.

## Geometry Source Rules

- The realtime reverse primal must use the same geometry sources as the
  realtime forward solver.
- Boozer/file-derived quantities must continue to come from the same Boozer/file
  path used by the forward solver.
- VMEC/NTX quantities that are realtime in the forward solver must come from the
  realtime VMEC state.
- `a_b`, `R0`, `r_grid`, `Vprime`, `Vprime_half`, `overVprime`, and
  `integrated_volume` must be finite before any reverse benchmark result is
  trusted.
- Realtime geometry construction must not overwrite or simplify frozen geometry
  construction.

## Ambipolar Initial Er

- For the first working realtime-geometry reverse benchmark, keep the initial
  ambipolar `Er` behavior the same as in the frozen/profile reverse benchmark.
- The baseline initial `Er` is computed by the normal ambipolar solve using the
  active profiles and geometry, then treated as fixed during the transport
  rollout derivative.
- Differentiating through the ambipolar root solve is a later phase, documented
  separately in `ambipolarity_ad_plan.md`.

## Desired Reverse Outputs

- For each transport objective, print profile derivatives and geometry harmonic
  derivatives from the same reverse pass or from an explicitly equivalent
  compact multi-RHS rule.
- Example profile outputs:
  - `d objective / d n0`
  - `d objective / d T0`
  - `d objective / d density_shape_power`
  - `d objective / d temperature_shape_power`
- Example geometry outputs:
  - `d objective / d RBC:1:0`
  - `d objective / d ZBS:1:0`
- Geometry-only objectives such as QI may be printed as diagnostics. They do not
  depend on transport evolution and should match the validated geometry-only
  operator-paired benchmark.

## Validated Geometry-Only Operator Pair

The scalar QI diagnostic showed that the raw-block forward and raw-block
transpose reverse pair is internally consistent.

Command:

```bash
python ./examples/benchmarks/compare_geometry_qi_frozen_linearized_fd.py \
  --vmec-input ./examples/inputs/input.QI_nfp2_newNT_opt_hires_true \
  --parameter RBC:1:0 \
  --objective boozer_qi_objective \
  --reference-fd=5.909044e-01 \
  --multigrid \
  --forward-linear-solve-mode raw_block \
  --forward-linear-maxiter 300 \
  --adjoint-maxiter 300
```

Observed:

- `frozen_linearized_fd=-3.0868613803103507e-03`
- `forward_jvp=-3.0867443678659922e-03`
- Earlier matching reverse:
  `raw_block_transpose_reverse_param_grad=-3.0867443806652517e-03`

The optimization-style forward diagnostic also matched the same value for this
case:

```bash
python ./examples/benchmarks/compare_geometry_qi_frozen_linearized_fd.py \
  --vmec-input ./examples/inputs/input.QI_nfp2_newNT_opt_hires_true \
  --parameter RBC:1:0 \
  --objective boozer_qi_objective \
  --reference-fd=5.909044e-01 \
  --multigrid \
  --forward-linear-solve-mode block_corrected \
  --forward-linear-maxiter 300 \
  --adjoint-maxiter 300
```

Observed:

- `frozen_linearized_fd=-3.0867226640350623e-03`
- `forward_jvp=-3.0867443613196649e-03`
- `reverse_state_dot_tangent=-3.0867443613288997e-03`
- `raw_block_transpose_reverse_param_grad=-3.0867443792406135e-03`

Interpretation:

- For this case, the VMEX optimization-style block correction did not change
  the raw-block tangent materially.
- The matching reverse equivalent is the raw-block transpose rule, not the
  non-converged preconditioned transpose GMRES helper.
- Do not compare raw-block reverse against a different preconditioned
  GMRES-only forward target.

Default reverse convention:

- Geometry-objective tables should use the raw-block transpose VMEC
  state-to-harmonic pullback by default.
- Transport realtime-geometry reverse payloads should use the same split:
  first pull geometry/Boozer/NTX payload cotangents back to the converged VMEC
  state, then apply the raw-block transpose pullback to VMEC harmonics.
- Generic VMEC solve VJPs, preconditioned multi-RHS helpers, and full
  `geometry_delta -> payload` VJPs are diagnostics only until they are shown to
  match this operator-paired convention.

## Relation To VMEX QI Optimization

VMEX optimization examples call:

```python
opt.least_squares(..., jac="implicit")
```

Inside that implicit path, the default is:

```python
jac_solver="block"
```

That optimization path is:

```text
A_raw dz0 = -F_raw,p dp
dz = GMRES(A_pre, -F_pre,p dp, x0=dz0, max_restarts=min(3, cfg.adjoint_maxiter))
dJ = J_z dz + J_p dp
```

The current diagnostics expose:

- `--forward-linear-solve-mode raw_block`: raw block solve only.
- `--forward-linear-solve-mode block_corrected`: raw block solve plus the short
  preconditioned GMRES correction, matching the optimization pattern for one
  direction.

These are diagnostic-only helpers and must not alter VMEX optimization defaults.

## Transport Reverse Architecture

1. Start from the known-good profile reverse AD lane.
2. Add geometry harmonics to the same parameter vector/cotangent structure.
3. Ensure frozen geometry sets geometry tangents/cotangents inactive while
   preserving profile reverse behavior.
4. Ensure realtime geometry activates VMEC harmonic tangents/cotangents only in
   the same places where the realtime forward solver uses VMEC state.
5. Route flux-evaluation geometry effects through the same compact reverse
   machinery as the profile effects.
6. Use compact geometry/NTX pullbacks for geometry support or flux payloads.
7. Print profile and geometry derivatives together for each objective.
8. Keep geometry-only objectives such as QI as separate diagnostics or sidecar
   rows, with the validated raw-block/operator-paired geometry rule.

## Forward FD Validation Plan

Add a realtime-geometry forward FD benchmark analogous to the frozen
profile/geometry FD checks.

Requirements:

- Use the same realtime VMEC forward solver path as the production TOML.
- Freeze the accepted transport step schedule for reverse comparison.
- Freeze the baseline VMEC solve logic consistently with the implicit
  linearized geometry diagnostic.
- Compare reverse AD against FD for:
  - profile-only perturbations,
  - geometry-only perturbations,
  - mixed profile plus geometry perturbations.
- Confirm that in frozen geometry mode, profile derivatives match the existing
  profile-only reverse benchmark.

## Current Commands Of Interest

Profile forward AD baseline:

```bash
python ./examples/benchmarks/benchmark_transport_forward_ad_only.py \
  --ntx-exact-derivative-mode direct \
  --ntx-exact-derivative-field-pullback-mode compact_vjp \
  --ntx-exact-derivative-pullback-algebra scalar_contract_lowdot_ntx \
  --parameter n0 \
  --accepted-step-limit 16 \
  --radau-jacobian-reuse-mode legacy \
  --forward-ad-fusion-mode accepted_replay
```

Realtime geometry reverse target command shape:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --reverse-parameter-mode profiles_plus_realtime_geometry \
  --reverse-geometry-parameter RBC:1:0 \
  --realtime-geometry-gradient-path reverse_payload \
  --ntx-exact-derivative-mode direct \
  --ntx-exact-derivative-field-pullback-mode generic_jvp \
  --objective all \
  --accepted-step-limit 2 \
  --radau-jacobian-reuse-mode legacy \
  --timing-mode jit-warm \
  --reverse-segment-length 1 \
  --reverse-stage-adjoint-solve-mode bicgstab \
  --reverse-rhs-transpose-mode explicit_ntx_interpolated \
  --reverse-step-bwd-mode reduced_cotangent
```

This command should only be trusted after the realtime geometry primal
diagnostics are finite and match the forward solver geometry construction.

## Things To Avoid

- Do not use `support_segment_probe` as the final derivative path. It is a
  support-only diagnostic.
- Do not print geometry derivatives if the primal geometry diagnostics are
  non-finite.
- Do not route geometry gradients through a Python loop over objectives as the
  final implementation.
- Do not make full VMEC/Boozer/NTX generic VJPs part of the transport reverse
  benchmark.
- Do not change the working forward solver TOML path to make reverse easier.
- Do not conflate the full nonlinear FD reference with the frozen-linearized
  implicit reference for sensitive VMEC/Boozer/QI objectives.
