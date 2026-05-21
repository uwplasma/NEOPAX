# SPECTRAX Quasilinear Runtime Method

## Goal

Define a practical NEOPAX turbulence flux model based on a local SPECTRAX-GK linear scan and a fixed-coefficient spectral-envelope closure,

```text
log Q_turb = b0 + b1 f1 + b2 f2
```

with:

- `f1 = log_positive_ky_centroid`
- `f2 = ky_weighted_std`

The initial target is:

- electrostatic turbulence,
- adiabatic electrons,
- ion-driven heat transport,
- lagged-response support analogous to the current NTX exact-runtime path,
- derivatives tailored to this quasilinear model rather than copied from the NTX neoclassical implementation.

This note is intentionally about the first runtime-capable reduced model, not the full future turbulence coupling roadmap.

## Why this model first

Relative to the current SPECTRAX-GK options discussed so far:

- it performs much better than plain `mixing_length` on the current stored HSX and W7-X holdouts,
- it is still based on a local linear scan rather than a full nonlinear turbulent solve,
- if `b0`, `b1`, and `b2` are treated as fixed runtime inputs, the closure is just a smooth map on top of differentiable linear outputs,
- it is a better candidate than direct nonlinear turbulent coupling for a first in-transport NEOPAX implementation.

Relative to a T3D-like power-law closure:

- it is more surrogate-like and less physics-closure-like,
- but it is compact,
- already has evidence in the SPECTRAX-GK repo,
- and can be implemented without introducing an extra fitted metric layer first.

## Current evidence from SPECTRAX-GK

The current SPECTRAX-GK repo already contains:

- a differentiable reduced linear/quasilinear lane,
- AD-vs-FD gates for linear growth, frequency, quasilinear weights, and mixing-length heat-flux proxies,
- leave-one-out `spectral_envelope_ridge` fits across multiple cases,
- encouraging holdout behavior for HSX and W7-X in the electrostatic adiabatic-electron lane.

Important scope restriction:

- the current positive evidence is mainly in the reduced electrostatic, adiabatic-electron, ion-heat-flux-oriented regime.

So this NEOPAX implementation should explicitly start in that regime rather than over-claiming generality.

## Proposed model form

### 1. Local linear scan

At each transport radial location `r_i`, run a local SPECTRAX-GK linear scan over a chosen set of `ky` values:

```text
ky_1, ky_2, ..., ky_N
```

For each `ky_j`, obtain the linear result needed to construct spectral features.

The first implementation should keep the scan narrowly scoped:

- one local geometry per radius,
- one local profile state per radius,
- one fixed set of runtime scan controls,
- one electrostatic adiabatic-electron configuration.

### 2. Spectrum features

Construct the same reduced features used in the SPECTRAX-GK `spectral_envelope_ridge` candidate.

For the first model:

```text
f1 = log_positive_ky_centroid
f2 = ky_weighted_std
```

Conceptually:

- `log_positive_ky_centroid` captures where the positive or effective spectral weight is centered in `ky`,
- `ky_weighted_std` captures the spread of that spectral envelope.

The exact feature implementation should be lifted from the SPECTRAX-GK reduced candidate logic as faithfully as possible, but with one extra requirement for NEOPAX runtime use:

- use a smooth implementation whenever a hard threshold or clipping step would otherwise appear.

### 3. Fixed-coefficient closure

Treat `b0`, `b1`, and `b2` as user-supplied fixed model parameters:

```text
log Q_total = b0 + b1 f1 + b2 f2
Q_total = exp(b0 + b1 f1 + b2 f2)
```

This means:

- no fitting is done inside NEOPAX runtime,
- no regression coefficients are updated during the transport solve,
- the closure is simply a differentiable algebraic map from spectral features to total turbulent heat flux.

### 4. Species/channel reconstruction

The spectral-envelope model currently gives a total heat-flux level, not a full multi-channel turbulent closure by itself.

So a first NEOPAX runtime version should reconstruct channel-level fluxes using linear or quasilinear weights from the same scan.

Recommended first version:

```text
Q_s = WQ_s * Q_total
Gamma_s = WG_s * Q_total
```

where:

- `WQ_s` is a species heat-flux partition weight,
- `WG_s` is a species particle-flux partition weight.

The first implementation should keep this simple:

- use smooth normalized weights,
- ensure the weights sum sensibly,
- zero-fill channels that are absent by model design, such as adiabatic-electron turbulent channels when appropriate.

## Why this can be differentiable

If `b0`, `b1`, and `b2` are fixed, then the differentiability question becomes:

1. can SPECTRAX-GK provide differentiable local linear outputs?
2. can we compute the spectral features smoothly from those outputs?
3. can we propagate derivatives through the algebraic closure and channel partition?

The SPECTRAX-GK repo already gives strong evidence for item 1:

- differentiable reduced linear/quasilinear objectives,
- implicit-eigenpair sensitivity demos,
- VMEC/Boozer full-chain reduced gradient gates,
- passed AD-vs-FD gates for linear growth, frequency, quasilinear weights, and a quasilinear heat-flux proxy.

So the main implementation caution is item 2:

- avoid hard mode masking,
- avoid hard `max(0, x)` where possible,
- avoid traced Python branching over active modes.

Where positivity or filtering is needed, prefer smooth substitutes:

- softplus-type positive parts,
- soft weighting,
- differentiable normalization floors.

## Comparison with the NTX exact-runtime path

The intended user experience should be similar to the existing NTX exact-runtime turbulence/neoclassical integration pattern:

- local flux model callable inside NEOPAX,
- optional lagged-response mode,
- configurable refresh/reuse behavior,
- compatible with implicit transport backends.

But the derivative model must not simply mimic NTX.

Why not:

- NTX exact runtime is fundamentally a local neoclassical coefficient evaluation problem,
- this SPECTRAX model is a reduced turbulence closure based on a linear `ky` spectrum and spectral features,
- its natural state variables and sensitivities are different.

So the lagged-response machinery should be structurally similar to NTX, but the response variables, Jacobians, and anchor updates should be tailored to this closure.

## Proposed NEOPAX model entry

Add a new turbulence transport model key, for example:

- `"spectrax_quasilinear_runtime"`

And a corresponding lagged-response mode:

- `"spectrax_quasilinear_runtime_lagged"`

The runtime should expose at least:

- `Gamma_turb(r, species)`
- `Q_turb(r, species)`

Optional diagnostics:

- total spectral-envelope heat flux,
- raw features `f1`, `f2`,
- local scan spectra,
- local weight partitions,
- local derivative information,
- refresh/reuse diagnostics.

## Proposed configuration surface

Suggested TOML structure:

```toml
[turbulence]
flux_model = "spectrax_quasilinear_runtime"

[turbulence.spectrax_quasilinear]
enabled = true
mode = "spectral_envelope_fixed"
spectrax_root = "/path/to/SPECTRAX-GK"
template = "/path/to/reference_runtime.toml"

# Fixed closure coefficients
b0 = 2.29
b1 = 0.80
b2 = -0.57

# Scan definition
ky_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
Nl = 4
Nm = 8

# Runtime scope
electrostatic_only = true
adiabatic_electrons_only = true

# Channel reconstruction
channel_partition = "linear_weights"

# Lagged-response controls
response_mode = "lagged"
anchor_count = 3
recompute_tolerance = 0.05
refresh_every_n_accepted_steps = 1
```

The exact field names can change, but the capabilities should be explicit.

## Local model evaluation

At each requested radius:

1. extract local geometry and local profile state,
2. build or patch the SPECTRAX runtime input,
3. run the local linear `ky` scan,
4. construct `f1`, `f2`,
5. evaluate

```text
Q_total = exp(b0 + b1 f1 + b2 f2)
```

6. reconstruct species/channel fluxes from weights,
7. return fluxes and derivative data.

This local evaluator is the core primitive.

Everything else should be built around it:

- direct black-box evaluation,
- lagged-response approximation,
- refresh and cache reuse,
- offline verification scripts,
- future calibration and audit tooling.

## Derivative design

### What should be differentiated

The lagged-response system should differentiate the actual reduced turbulence closure with respect to the local transport-relevant inputs.

At minimum, the local model should expose derivatives with respect to the inputs that define the transported state at one radius, for example:

- local densities,
- local temperatures,
- local gradient drives,
- local `Er` or derived local quantities if they enter the turbulence model,
- possibly selected geometry control variables through the SPECTRAX differentiable geometry path.

The exact variable basis should match the NEOPAX state representation and reconstruction path.

### Chain rule structure

The local Jacobian naturally factorizes as:

```text
dQ_total/dx
  = Q_total * (b1 df1/dx + b2 df2/dx)
```

for any local input `x`.

Then for species channels:

```text
dQ_s/dx = d(WQ_s Q_total)/dx
        = WQ_s dQ_total/dx + Q_total dWQ_s/dx
```

and similarly

```text
dGamma_s/dx = WG_s dQ_total/dx + Q_total dWG_s/dx
```

if `Gamma_s` is reconstructed from the same amplitude level.

This is already qualitatively different from NTX exact-runtime derivatives:

- NTX differentiates local transport coefficients and ambipolar structure,
- this turbulence model differentiates spectral features and channel partitions.

So the response object should store the derivatives that are natural for this closure, not just an NTX-shaped surrogate.

## Lagged-response design

### High-level idea

Use the same broad strategy as NTX lagged response:

- evaluate the expensive local model at anchor states,
- store a local first-order response model,
- reuse that model for nearby transport states,
- refresh it when the state drifts too far or a configured criterion is triggered.

### Local response form

For each radius, store a base state `x0` and base flux outputs:

- `Q0_s`
- `Gamma0_s`

plus first-order response matrices:

```text
Q_s(x)     ≈ Q0_s     + JQ_s (x - x0)
Gamma_s(x) ≈ Gamma0_s + JG_s (x - x0)
```

where:

- `JQ_s` is the Jacobian of the turbulent heat-flux channel with respect to the local input vector,
- `JG_s` is the Jacobian of the turbulent particle-flux channel with respect to the local input vector.

This gives NEOPAX the same practical lagged-response capability it already has for NTX:

- cheap repeated evaluations between refreshes,
- solver-friendly local linearization,
- natural path for implicit integration and benchmarking.

### Refresh criteria

The lagged response should be refreshed when one or more of these occurs:

- local state distance exceeds a configured threshold,
- one accepted transport step has completed and a refresh cadence is due,
- the solver requests a hard update,
- the response prediction fails a sanity check,
- the local linear scan returns a materially different spectral structure than the cached one.

### Anchor strategy

The first version does not need a complicated anchor scheme.

A simple and robust starting point:

- one anchor per active radius,
- optionally a small recent-history bank for reuse,
- no radial interpolation of anchors in version 1.

This is enough to make the lagged mode usable while keeping the implementation understandable.

## Recommended implementation phases

### Phase 1. Offline local evaluator

Build a standalone Python path that:

- extracts one local NEOPAX-like state,
- runs a SPECTRAX linear scan,
- computes `f1`, `f2`,
- evaluates fixed-`b0,b1,b2` `Q_total`,
- reconstructs `Q_s` and `Gamma_s`,
- returns fluxes and local Jacobians.

This phase should include:

- AD-vs-FD checks,
- feature audit output,
- channel partition sanity checks.

### Phase 2. Black-box in-transport runtime model

Integrate the local evaluator into NEOPAX as:

- `flux_model = "spectrax_quasilinear_runtime"`

with direct recomputation at each requested call, no lagged reuse yet.

This gives:

- functional end-to-end coupling,
- initial transport benchmarks,
- validation of normalization and channel mapping.

### Phase 3. Lagged-response model

Add:

- cached local base states,
- Jacobian storage,
- refresh logic,
- state-distance criteria,
- diagnostics for reuse versus refresh.

This becomes the turbulence-side analog of the NTX lagged-response capability.

### Phase 4. Broader closure options

Only after the fixed-`b0,b1,b2` version is stable should we consider:

- alternative feature sets,
- a T3D-like quasilinear metric plus fitted power-law closure,
- T3D-like power-law closures,
- separate particle-flux regressions,
- kinetic-electron or electromagnetic extensions,
- runtime-calibrated or device-specific coefficient sets.

## Future option: T3D-like runtime closure

In addition to the fixed-coefficient spectral-envelope model, a natural future option is a second SPECTRAX-based reduced runtime closure inspired by the same broad idea used in T3D.

The intended structure would be:

1. run a local SPECTRAX linear `ky` scan,
2. build a scalar quasilinear activity metric from the linear spectrum,
3. partition that activity into species channels using linear or quasilinear weights,
4. apply a fitted power-law closure.

Schematically:

```text
M_total = spectral quasilinear activity metric
Q_s = C_Q * WQ_s * M_total^(a-1)
Gamma_s = C_G * WG_s * M_total^(a-1)
```

where:

- `M_total` is built from the local linear scan,
- `WQ_s` and `WG_s` are channel partition weights,
- `a`, `C_Q`, and `C_G` are fixed runtime inputs or calibration constants.

This future option is attractive because:

- it is closer to a classical quasilinear transport closure than the spectral-envelope ridge form,
- it is still compatible with a differentiable reduced linear-scan workflow,
- it may offer a more physics-interpretable alternative to the fixed-`b0,b1,b2` surrogate.

However, it should remain a second-stage option rather than the first implementation because:

- the fixed spectral-envelope model already has better direct evidence in the current SPECTRAX-GK repo,
- the T3D-like closure requires an extra design decision for the exact local metric,
- and we should validate one reduced turbulence runtime path in NEOPAX before supporting multiple closures.

If implemented later, it should reuse the same NEOPAX-side coupling infrastructure:

- the same local SPECTRAX linear scan adapter,
- the same lagged-response cache/update machinery,
- the same derivative plumbing,
- and only change the algebraic closure from spectral features to a fitted power-law metric model.

## Scientific and engineering risks

### 1. Regime limitation

The current evidence is strongest in the:

- electrostatic,
- adiabatic-electron,
- ion-driven heat-flux

lane.

So this model should be explicitly labeled that way.

### 2. Feature smoothness

If feature extraction relies on hard unstable-mode selection, differentiability can become fragile near marginality.

The implementation should prefer smooth weighting and explicit regularization.

### 3. Channel reconstruction uncertainty

The total heat-flux closure is the strongest evidence right now.

The reconstruction of:

- species heat channels,
- particle channels

is a further modeling choice and should be documented as such.

### 4. Fixed coefficients are a user contract

Once `b0`, `b1`, and `b2` are exposed as runtime inputs, users need clarity on what those coefficients represent:

- leave-one-out holdout fit,
- corpus-average fit,
- machine-family fit,
- or user-supplied custom calibration.

The runtime should not silently imply that one set is universally valid.

## Recommended first claim

The first NEOPAX implementation should be described conservatively as:

- a differentiable reduced turbulence runtime model,
- based on a local SPECTRAX-GK linear `ky` scan,
- using a fixed-coefficient spectral-envelope closure,
- with optional lagged-response reuse inside NEOPAX,
- validated first in the electrostatic adiabatic-electron lane.

That is strong enough to be useful and honest enough to remain scientifically defensible.

## Immediate next steps

1. Identify the exact SPECTRAX-GK feature-construction code path for `log_positive_ky_centroid` and `ky_weighted_std`.
2. Reimplement or expose that path in a runtime-friendly, smooth, testable form.
3. Build a standalone local evaluator returning:
   - `Q_total`
   - `Q_s`
   - `Gamma_s`
   - feature values
   - Jacobians
4. Add AD-vs-FD tests against local profile perturbations.
5. Integrate as a NEOPAX black-box turbulence model.
6. Add lagged-response cache and refresh logic.

## Current NEOPAX scaffold status

The first in-repo scaffold is now started in NEOPAX with a deliberately safe scope:

- new model keys:
  - `spectrax_quasilinear_runtime`
  - `spectrax_quasilinear_runtime_lagged`
- new model-local runtime helper:
  - `NEOPAX/_spectrax_quasilinear_runtime.py`
- current backend:
  - `backend_mode = "smooth_proxy"`

This means the present implementation is:

- flux-model-layer only,
- fully local to the turbulence model interface,
- differentiable and cheap enough for smoke tests,
- not yet connected to an external SPECTRAX-GK runtime scan.

The current scaffold already provides:

- a smooth local feature map,
- a fixed-coefficient closure

```text
log Q_total = b0 + b1 f1 + b2 f2
```

- simple channel reconstruction,
- lagged-response-compatible `build_lagged_response(...)`,
- lagged-response-compatible `evaluate_with_lagged_response(...)`,
- orchestrator wiring for normal NEOPAX model selection.

Two benchmark-style example configs now exist for smoke testing:

- `examples/benchmarks/Calculate_Fluxes_noHe_spectrax_quasilinear_runtime_benchmark.toml`
- `examples/benchmarks/Solve_Transport_equations_noHe_radau_spectrax_quasilinear_runtime_benchmark.toml`

These are intentionally not the final scientific model. They are the first safe integration scaffold so that:

- the configuration surface is exercised,
- the flux-model hooks are validated,
- lagged-response plumbing is tested at the model layer,
- future replacement of the smooth proxy with a real SPECTRAX adapter can happen behind the same interface.
