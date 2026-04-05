# GX and T3D Context for SPECTRAX-GK Turbulent Flux Model

## Purpose
This note captures how T3D couples to GX for turbulent flux evaluation, and translates those ideas into a practical implementation plan for a future SPECTRAX-GK turbulent flux model in NEOPAX.

## What T3D does with GX

### 1) Radial sampling strategy
- T3D evaluates turbulent fluxes on flux-tube points defined by the transport radial grid midpoints.
- Number of flux tubes is effectively:
  - N_fluxtubes = len(grid.midpoints)
- One GX run is launched per radial midpoint (and per optional field-line center).

### 2) Finite-difference Jacobian strategy
For each nonlinear transport iteration, T3D computes:
- Base flux evaluation (unperturbed gradients).
- Perturbed density-gradient evaluations for each evolved density channel.
- Perturbed temperature-gradient evaluations for each evolved temperature channel.

This produces both:
- Fluxes: particle and heat flux.
- Jacobians: dFlux/d(grad n), dFlux/d(grad T), used by the transport solver.

Total turbulent-code calls per iteration are approximately:
- (1 + N_density_evolved + N_temperature_evolved)
  x N_fluxtubes
  x N_zeta_centers

### 3) Angular/field-line handling
- T3D supports multiple field-line centers via zeta_center list.
- Default behavior is averaging over the selected field lines.

### 4) Optional interpolation mode
- With theta_zeta_fluxes enabled, T3D uses a theta-zeta interpolation/integration path.
- It builds an interpolant (scipy griddata, cubic) on a fixed theta-zeta mesh and integrates to recover flux-surface-averaged quantities.
- If disabled, it simply averages over zeta_center realizations.

### 5) Practical GX defaults often seen in templates
A common test/regression template uses values like:
- ntheta = 64
- nx = 64
- ny = 64
- nhermite = 12
- nlaguerre = 4

These are not universal defaults of all runs, but are representative settings used in the repository templates.

### 6) Recompute-versus-reuse behavior in real runs
- T3D requests GX flux updates at each transport iteration (time step and Newton sub-iteration).
- This does not always imply a fresh GX launch for every request.
- GX checks for existing outputs for the current run key and can reuse them when overwrite policy allows.
- In practice, new `(step_idx, iter_idx, perturbation, radius, zeta_center)` combinations often trigger fresh GX runs unless replay/restart reuse is enabled.

A useful mental model is:
- engine-level policy: always ask for fluxes each iteration,
- model-level policy: decide rerun versus cache reuse.

### 7) GX ReLU surrogate path
T3D's GX model includes an optional local surrogate mode that reduces expensive GX relaunches between refresh points.

How it works:
- Full GX pass:
  - run base plus perturbation calculations,
  - extract base fluxes and local finite-difference derivatives.
- Build local surrogate:
  - store base state gradients and derivative tensors.
- ReLU mode steps:
  - predict fluxes from local linearized updates around the stored base state,
  - avoid launching full GX for those steps.
- Periodic refresh:
  - after `nsteps_ReLU` transport steps (at iteration 0), disable surrogate mode and rebuild with fresh full GX runs.

This path is controlled by runtime knobs such as `build_ReLU` and `nsteps_ReLU`.
If `build_ReLU = false`, GX always uses the full external-run path.

## Why this matters for SPECTRAX-GK in NEOPAX
A T3D-like coupling pattern is robust for integrating an external turbulence engine into a transport solver:
- Clean separation of transport and turbulence responsibilities.
- Deterministic call pattern for Jacobian construction.
- Compatible with future surrogate modes (reuse previously computed derivatives between expensive GK evaluations).

## Proposed NEOPAX interface for SPECTRAX-GK

### A) New turbulent model entry in transport flux registry
Add a transport model key, for example:
- "spectrax_gk"

The model callable should return at minimum:
- Gamma_turb (n_species x n_radial)
- Q_turb (n_species x n_radial)
Optional:
- Upar_turb (n_species x n_radial)
- Derivative tensors (if solver path can consume them now or later)

### B) Minimal runtime parameter block
Add a [turbulence] TOML section supporting:
- transport_model = "spectrax_gk"
- spectrax_template = path
- spectrax_outputs = path
- overwrite = bool
- run_command = string or null
- zeta_center = [list]
- theta_zeta_fluxes = bool
- dkap_n = float
- dkap_T = float
- gpus_per_job (or similar resource knob)

### C) Core call workflow (T3D-like)
For each transport iteration:
1. Build base gradients at each radial point.
2. Query SPECTRAX-GK for base fluxes.
3. Apply density-gradient perturbations one species at a time and re-query.
4. Apply temperature-gradient perturbations one species at a time and re-query.
5. Build Jacobians from finite differences.
6. Return fluxes (and optionally Jacobians) to transport equations.

### D) Interpolation strategy
Support two modes:
- Fast mode: direct average over zeta centers.
- Detailed mode: theta-zeta interpolation and weighted integration to flux-surface averages.

Recommendation for first implementation:
- Start with direct zeta averaging only.
- Add theta-zeta interpolation in a second pass when diagnostics are stable.

## Numerical and software considerations

### 1) Determinism and caching
- Cache outputs by a stable key:
  - timestep, nonlinear iteration, radial index, perturbation id, zeta center.
- Reuse prior outputs when overwrite = false.

Recommended explicit cache key for this integration:
- `(step_idx, iter_idx, perturbation_id, radial_index, zeta_center, model_config_hash)`

Including a light config hash prevents accidental reuse when template resolution, physics toggles, or launch settings change.

### 2) Failure handling
- Detect missing/partial output files.
- Retry policy for stalled jobs.
- Fallback option:
  - Use previous valid turbulent fluxes.
  - Or switch temporarily to analytical turbulent model.

### 3) Solver coupling philosophy
- External turbulence execution should remain outside jitted kernels.
- JAX-side transport rhs should consume arrays already assembled from external results.
- Keep model selection and process orchestration in Python-level setup/orchestration code.

## Suggested staged implementation plan

### Stage 1: MVP
- Add "spectrax_gk" model entry.
- Base flux evaluation only (no Jacobians), optional fixed coefficients fallback.
- Single zeta center, no interpolation.

### Stage 2: Practical transport coupling
- Add finite-difference perturbation loop for Jacobians.
- Add file-based cache and restart behavior.
- Add multiple zeta centers with averaging.

### Stage 3: Advanced diagnostics and robustness
- Add theta-zeta interpolation mode.
- Add stall detection and automatic relaunch policy.
- Add performance statistics and per-radius timing logs.

## Mapping to current NEOPAX architecture
- _transport_flux_models.py:
  - host registry entry and wrapper model class.
- _parameters.py + run_from_toml.py:
  - parse and carry SPECTRAX-GK runtime knobs.
- External runner helper module (recommended):
  - isolated process launch, I/O parsing, retry logic.
- _turbulence.py:
  - can host turbulence helper math and post-processing utilities.

## Bottom line
A T3D-like GX coupling pattern is a strong template for SPECTRAX-GK in NEOPAX:
- radial-point turbulence evaluations,
- optional multi-zeta treatment,
- finite-difference Jacobians,
- and transport-side composition through the turbulent flux model registry.
