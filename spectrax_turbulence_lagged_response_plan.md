# SPECTRAX Turbulence Lagged Response Plan

## Goal

Add SPECTRAX-GK turbulent flux modes to NEOPAX that use the existing lagged-response transport solver machinery, analogous in solver interface to the NTX exact-runtime neoclassical lagged response, but without changing or contaminating the working NTX lagged path.

The target use case is:

- NTX remains the neoclassical flux model.
- SPECTRAX-GK becomes an optional turbulence flux model.
- The transport solver can rebuild or reuse a turbulent lagged response using the same solver-level cache and drift machinery already available for lagged responses.
- Future realtime-geometry optimization can differentiate transport objectives with respect to profiles and geometry parameters, but the first SPECTRAX turbulence plan should focus on profile-state turbulent response correctness.

This file is a plan only. Implementation should happen later and should be validated in small steps.

## Non-contamination requirement

Do not modify the working NTX lagged-response semantics while implementing this plan.

Concretely:

- Do not change `NTXExactLijRuntimeTransportModel` behavior to support SPECTRAX.
- Do not change existing NTX benchmark defaults.
- Do not change solver-level `lagged_response` behavior unless the change is model-agnostic and separately validated on the existing NTX benchmarks.
- Implement SPECTRAX turbulence as a separate `TransportFluxModelBase` subclass or as an extension of the existing SPECTRAX turbulence model class.
- Keep SPECTRAX-specific response pytrees separate from `NTXExactLijLaggedResponse`.
- Add tests/benchmarks that can run SPECTRAX turbulence alone before combining it with NTX.

The intended shared contract is only the existing public model interface:

```python
build_lagged_response(state, **kwargs)
evaluate_with_lagged_response(state, lagged_response, **kwargs)
pullback_build_lagged_response(...)
pullback_evaluate_with_lagged_response(...)
```

## Current starting point

NEOPAX already has a scaffolded SPECTRAX turbulence model:

```text
SpectraXQuasilinearRuntimeTransportModel
```

registered under:

```text
spectrax_quasilinear_runtime
spectrax_quasilinear_runtime_lagged
```

The current implementation uses a smooth in-repo proxy evaluator, not the full SPECTRAX-GK runtime. Its lagged response already has the right shape:

```text
reference_state
reference_flux
linearized flux update
```

The new work should replace or extend this proxy lane with real SPECTRAX-GK nonlinear and quasilinear evaluators while preserving the same transport-facing interface.

## Mode A: nonlinear SPECTRAX-GK lagged response

### Intent

Use SPECTRAX-GK nonlinear turbulent fluxes directly as the expensive turbulence response, then linearize those fluxes around the reference transport state.

This is the closest analog to how T3D/GX uses nonlinear or expensive gyrokinetic flux evaluations:

1. Evaluate the base turbulent flux at the current transport state.
2. Perturb profile inputs.
3. Re-evaluate turbulent fluxes.
4. Build a local finite-difference response.
5. Reuse that response inside the transport nonlinear/Radau solve until the solver decides to rebuild it.

### First implementation derivative mode

Use finite differences first.

Do not require SPECTRAX nonlinear AD for the first nonlinear turbulent response.

Reason:

- Nonlinear turbulent windows are expensive and may include diagnostic/postprocessing choices.
- Saturated nonlinear fluxes are less smooth than reduced linear/quasilinear features.
- A T3D-like finite-difference response is easier to validate against existing transport intuition.
- FD lets us isolate the coupling and lagged-response semantics before debugging nonlinear AD.

### Response contents

The nonlinear lagged response should store:

```text
reference_state
reference_fluxes
profile_response_basis
finite_difference_steps
dGamma/dprofile_basis
dQ/dprofile_basis
dUpar/dprofile_basis, optional or zero initially
diagnostics
```

The first profile basis should be local and small:

- density-gradient perturbations,
- temperature-gradient perturbations,
- optionally profile-value perturbations if the SPECTRAX nonlinear input depends directly on profile values rather than only gradients,
- no geometry perturbations in the first version.

### Evaluation

For a current state near the reference state:

```text
delta = current_state - reference_state
flux(current_state) ~= reference_flux + response_matrix @ delta_features
```

The `evaluate_with_lagged_response` implementation should be pure JAX algebra after the response is built. It should not call SPECTRAX-GK again.

### Rebuild and reuse criterion

Use the existing NEOPAX solver-level lagged-response controls:

```toml
radau_rhs_mode = "lagged_response"
lagged_response_reuse_mode = "global_state_drift"
lagged_response_reuse_rtol = ...
lagged_response_reuse_atol = ...
```

The model should not implement its own competing accepted-step/rejected-step policy unless later evidence shows the global state drift metric is insufficient for turbulence.

### Practical nonlinear questions to settle before implementation

1. What exact nonlinear SPECTRAX output is the turbulent flux?
2. What time window or diagnostic average defines the response?
3. Is the nonlinear flux deterministic enough for finite-difference columns?
4. Which quantities should perturb: profile values, gradients, normalized gradients, collisionality, beta, or geometry inputs?
5. How expensive is one full response build per radial surface?
6. Should nonlinear SPECTRAX be run at all radii in one response build, or only at selected anchor radii with interpolation?

## Mode B: quasilinear SPECTRAX-GK lagged response

### Intent

Use SPECTRAX-GK quasilinear turbulent fluxes as a differentiable turbulence closure.

Unlike the nonlinear mode, the quasilinear mode should use AD, not finite differences.

### Derivative mode

Quasilinear mode is AD-only.

Do not add a finite-difference quasilinear response mode unless later evidence shows an AD path is impossible for a required quasilinear output.

Reason:

- The quasilinear lane is intended to be a differentiable reduced model.
- NEOPAX already has a JVP-style lagged response scaffold for SPECTRAX-like turbulence.
- The quasilinear closure is expected to be smoother and cheaper than nonlinear turbulent windows.
- AD avoids storing explicit dense Jacobians and keeps the model closer to the current compact lagged-response strategy.

### AD response form

The preferred quasilinear response is compact JVP/VJP style:

```text
reference_state
reference_flux
linearized evaluator around reference_state
```

For forward/evaluate:

```python
_, tangent_flux = jax.jvp(
    spectrax_quasilinear_flux_fn,
    (reference_state,),
    (current_state - reference_state,),
)
flux = reference_flux + tangent_flux
```

For reverse-mode transport:

- Prefer model-provided compact pullbacks.
- Avoid materializing full dense flux Jacobians when possible.
- If the quasilinear response is a vector of species/radius fluxes, propagate cotangents through the same reduced feature/JAX path.

### Quasilinear output choices

The first quasilinear implementation should define one of these explicitly:

1. Linear growth-rate based proxy.
2. Mixing-length heat-flux proxy.
3. Shape-aware quasilinear heat-flux proxy.
4. Existing SPECTRAX quasilinear transport weights plus smooth saturation rule.

For AD-only mode, the chosen output must avoid non-JAX reporting paths such as Python `float(...)` conversion or JSON dataclass construction inside the differentiated function.

Use lower-level JAX arrays/features as the differentiable interface, then build reporting diagnostics outside the AD path.

### Important quasilinear caution

Current SPECTRAX-GK examples note that some quasilinear optimization lanes use finite-difference outer Jacobians because they depend on a nonsymmetric eigenvector selection.

That does not block the plan, but it means the first AD-only quasilinear mode should choose an output whose AD path is already validated, or should first add an AD validation benchmark.

Recommended first target:

- a reduced smooth feature or growth-rate based objective with confirmed AD-vs-FD behavior,
- then move to heat-flux weights or mixing-length saturated flux after branch/eigenvector continuity is validated.

## Shared NEOPAX integration design

### Model keys

Suggested model keys:

```text
spectrax_nonlinear_runtime_lagged_fd
spectrax_quasilinear_runtime_lagged_ad
```

or, if we keep one family name:

```toml
[turbulence]
flux_model = "spectrax_runtime_lagged"

[turbulence.spectrax]
mode = "nonlinear_fd"       # nonlinear finite-difference response
# or
mode = "quasilinear_ad"     # quasilinear AD response
```

The exact names can be chosen later, but nonlinear FD and quasilinear AD should remain explicit in the TOML and benchmark output.

### Response dataclasses

Add separate response pytrees, for example:

```python
SpectraXNonlinearFDLaggedResponse
SpectraXQuasilinearADLaggedResponse
```

Do not reuse NTX response classes.

### Combined flux behavior

The existing `CombinedTransportFluxModel` can already combine:

```text
neoclassical_response
turbulent_response
classical_response
```

So the SPECTRAX response should slot into `turbulent_response`.

This means the solver-level lagged-response cache can remain model-agnostic.

### Face fluxes and channels

The SPECTRAX turbulent model must return the same flux dictionary shape as other NEOPAX flux models:

```python
{
    "Gamma": ...,
    "Q": ...,
    "Upar": ...,
}
```

First version:

- support `Q_turb`,
- support optional `Gamma_turb`,
- set `Upar_turb = 0` unless a validated turbulent parallel momentum channel exists.

### Geometry

First implementation should keep geometry fixed during the response build/evaluation.

Later extension:

- include realtime geometry in the SPECTRAX input,
- expose geometry-response pullbacks,
- compare against geometry-objective AD benchmarks and forward finite differences.

Do not start by differentiating nonlinear SPECTRAX fluxes with respect to geometry unless the profile-only response is already correct.

## Benchmark plan

### Phase 1: isolated model-call benchmarks

Create small tests that do not involve Radau:

1. Evaluate nonlinear SPECTRAX flux for one profile state.
2. Build nonlinear FD lagged response.
3. Evaluate nonlinear FD lagged response at perturbed states.
4. Compare against direct nonlinear SPECTRAX flux at the perturbed states.

For quasilinear:

1. Evaluate quasilinear SPECTRAX flux for one profile state.
2. Build quasilinear AD lagged response.
3. Compare JVP-linearized response against direct quasilinear flux for small perturbations.
4. Compare reverse pullbacks against finite differences only as a validation test, not as the production derivative mode.

### Phase 2: transport one-step benchmarks

Use a one-accepted-step or short accepted-step-limit transport run:

```toml
radau_rhs_mode = "lagged_response"
lagged_response_reuse_mode = "retry_only"
stop_after_accepted_steps = 1
```

Then repeat with:

```toml
lagged_response_reuse_mode = "global_state_drift"
```

Record:

- number of turbulent response rebuilds,
- number of turbulent response reuses,
- objective values,
- flux finiteness,
- solver accepted/rejected attempts.

### Phase 3: reverse/AD benchmarks

For quasilinear AD:

- run profile-gradient reverse checks,
- compare against finite differences,
- verify no dense Jacobian materialization,
- verify NTX-only reverse benchmarks still match previous values.

For nonlinear FD:

- the lagged response itself may be FD-built,
- transport reverse through the cheap linearized response should be possible,
- gradients should be interpreted as gradients of the lagged FD surrogate, not full nonlinear SPECTRAX AD.

### Phase 4: combined NTX plus SPECTRAX turbulence

Only after isolated turbulence benchmarks pass:

- combine `ntx_exact_lij_runtime` neoclassical with SPECTRAX turbulent lagged response,
- verify the combined response rebuild/reuse counts,
- verify that NTX response values are unchanged relative to NTX-only benchmarks when SPECTRAX is disabled,
- verify that SPECTRAX response values are unchanged relative to SPECTRAX-only benchmarks when NTX is disabled.

## Validation criteria

### Nonlinear FD mode

Accept when:

- direct nonlinear SPECTRAX fluxes are finite,
- FD response columns are finite,
- lagged linearized flux matches direct perturbed flux to expected first-order accuracy for small perturbations,
- transport run accepts at least one step,
- rebuild/reuse counts are reported,
- disabling SPECTRAX returns existing NTX behavior.

### Quasilinear AD mode

Accept when:

- quasilinear flux is finite,
- JVP-linearized lagged response matches direct quasilinear perturbation at small amplitude,
- reverse pullback agrees with finite-difference validation,
- no Python `float(...)` or report dataclass conversion appears inside the differentiated path,
- memory is consistent with compact JVP/VJP expectations,
- disabling SPECTRAX returns existing NTX behavior.

## Risks

### Nonlinear mode risks

- Full nonlinear SPECTRAX windows may be too expensive to rebuild frequently.
- Flux averages may be noisy, making FD columns unstable.
- Long-window diagnostics may not be suitable inside a Radau inner-loop response.
- Anchor/interpolation may be needed sooner than expected.

### Quasilinear mode risks

- Eigenvector or branch selection can make AD fragile.
- Some public SPECTRAX diagnostics are report-oriented and convert to Python floats.
- Differentiable lower-level feature extraction may need to be exposed before NEOPAX can use it safely.

### Shared risks

- A turbulent response has different natural variables than NTX.
- Copying NTX derivative internals directly would be the wrong abstraction.
- Solver-level lagged-response logic should stay generic; model-specific response math belongs inside the SPECTRAX turbulent model.

## First implementation checklist for later

1. Add a tiny direct SPECTRAX nonlinear flux evaluator wrapper that returns JAX/NumPy arrays in NEOPAX flux dictionary format.
2. Add `SpectraXNonlinearFDLaggedResponse`.
3. Add nonlinear FD `build_lagged_response` and cheap algebraic `evaluate_with_lagged_response`.
4. Add a small no-Radau benchmark comparing direct nonlinear flux perturbations against the FD lagged response.
5. Add a quasilinear AD evaluator that avoids report-time Python conversions.
6. Add `SpectraXQuasilinearADLaggedResponse` or reuse the existing compact JVP response only if the type/name makes the AD-only contract explicit.
7. Add quasilinear AD-vs-FD validation benchmark.
8. Add one-step transport benchmarks.
9. Add combined NTX plus SPECTRAX turbulence benchmarks.
10. Re-run existing NTX lagged benchmarks to confirm no regression.

## Summary

The plan is to add SPECTRAX-GK as a turbulence lagged-response family, not to alter NTX.

Use:

- nonlinear SPECTRAX-GK with finite-difference lagged response first,
- quasilinear SPECTRAX-GK with AD-only lagged response,
- existing NEOPAX solver-level lagged-response rebuild/reuse machinery,
- isolated benchmarks before combined NTX plus turbulence runs.

The intended end state is a NEOPAX transport configuration where:

```text
neoclassical fluxes: NTX exact-runtime lagged response
turbulent fluxes: SPECTRAX-GK lagged response
solver reuse: existing Radau lagged-response cache and drift criteria
```

without changing the validated NTX path.
