# SFINCS and T3D Context for sfincs_jax Neoclassical Flux Model

## Purpose
This note captures how T3D couples to SFINCS for neoclassical flux evaluation, and translates those ideas into a practical implementation plan for a future sfincs_jax neoclassical flux model in NEOPAX.

## What T3D does with SFINCS

### 1) Radial sampling strategy
- T3D evaluates neoclassical fluxes on flux-tube points defined by the transport radial grid midpoints.
- Number of neoclassical surfaces is effectively:
  - N_fluxtubes = len(grid.midpoints)
- One SFINCS run is launched per radial midpoint.

### 2) Finite-difference Jacobian strategy
For each nonlinear transport iteration, T3D computes:
- Base flux evaluation (unperturbed gradients).
- Perturbed density-gradient evaluations for each evolved density channel.
- Perturbed temperature-gradient evaluations for each evolved temperature channel.

This produces both:
- Fluxes: particle and heat flux.
- Jacobians: dFlux/d(grad n), dFlux/d(grad T), used by the transport solver.

Total neoclassical-code calls per iteration are approximately:
- (1 + N_density_evolved + N_temperature_evolved)
  x N_fluxtubes

### 3) Surface handling
- T3D does not use the GX-style theta-zeta postprocessing interpolation path for SFINCS.
- Instead, it runs one neoclassical solve per requested radial surface and reads the resulting fluxes directly from each SFINCS output.
- For VMEC geometry, T3D sets SFINCS to use VMEC-compatible radial-coordinate options and passes each target surface through `rn_wish`.

### 4) Resolution handling
- Resolution is controlled primarily by the SFINCS template namelist, not by T3D itself.
- Representative template parameters seen in the repository are:
  - Ntheta = 19
  - Nzeta = 59
  - Nxi = 60
  - Nx = 5
- These govern angular and velocity-space resolution inside SFINCS.

### 5) Practical execution model
- T3D writes one input namelist per radial surface and perturbation id.
- It launches SFINCS externally, collects `sfincsOutput.h5`, and converts returned fluxes into T3D normalization.
- The perturbed runs are used to build flux derivatives for Newton-based transport iterations.

### 6) Recompute-versus-reuse behavior in real runs
- T3D requests SFINCS flux updates at each transport iteration (time step and Newton sub-iteration).
- This does not always imply a fresh SFINCS executable launch for every request.
- The SFINCS model checks whether the expected output file already exists for the current run key; if present and `overwrite = false`, it reuses that output instead of relaunching.
- In practice, new `(step_idx, iter_idx, perturbation, radius)` combinations usually create new paths, so recomputation is common unless replay/restart reuse is explicitly enabled.

A useful mental model is:
- engine-level policy: always ask for fluxes each iteration,
- model-level policy: decide rerun versus cache reuse.

## Why this matters for sfincs_jax in NEOPAX
A T3D-like SFINCS coupling pattern is a strong template for integrating sfincs_jax as an external or semi-external neoclassical engine into a transport solver:
- Clean separation of transport and neoclassical responsibilities.
- Deterministic call pattern for Jacobian construction.
- Easier parity tracking against SFINCS/T3D workflows.
- Natural path for later upgrades from file-driven execution toward direct in-memory Python calls.

Compared with the SPECTRAX-GK/GX turbulent path, the sfincs_jax path is simpler in one important way:
- no theta-zeta interpolation layer is required in the first implementation.

## Proposed NEOPAX interface for sfincs_jax

### A) New neoclassical model entry in transport flux registry
Add a transport model key, for example:
- "sfincs_jax"

The model callable should return at minimum:
- Gamma_neo (n_species x n_radial)
- Q_neo (n_species x n_radial)
Optional:
- Upar_neo (n_species x n_radial)
- Derivative tensors (if solver path can consume them now or later)
- Ambipolar diagnostics / Er diagnostics if the chosen sfincs_jax mode exposes them cleanly

### B) Minimal runtime parameter block
Add a `[neoclassical]` TOML section supporting:
- transport_model = "sfincs_jax"
- sfincs_jax_input_template = path
- sfincs_jax_outputs = path
- overwrite = bool
- run_command = string or null
- use_python_api = bool
- differentiable = bool
- dkap_n = float
- dkap_T = float
- cpus_per_job = int
- mpis_per_job = int
- device = "cpu" or "gpu"

### C) Core call workflow (T3D-like)
For each transport iteration:
1. Build base gradients at each radial point.
2. Query sfincs_jax for base fluxes.
3. Apply density-gradient perturbations one species at a time and re-query.
4. Apply temperature-gradient perturbations one species at a time and re-query.
5. Build Jacobians from finite differences.
6. Return fluxes (and optionally Jacobians) to transport equations.

### D) Two execution modes
Support two modes from the beginning:
- External-style mode:
  - write input, run `sfincs_jax`, read `sfincsOutput.h5`
- Direct Python mode:
  - call sfincs_jax library functions in memory and collect result arrays directly

Recommendation for first implementation:
- Start with the external-style mode because it mirrors T3D/SFINCS closely and is easier to validate.
- Add direct Python mode in a second pass when parity and API stability are established.

## Numerical and software considerations

### 1) Determinism and caching
- Cache outputs by a stable key:
  - timestep, nonlinear iteration, radial index, perturbation id.
- Reuse prior outputs when overwrite = false.
- Keep cache semantics aligned with current transport restart behavior.

Recommended explicit cache key for this integration:
- `(step_idx, iter_idx, perturbation_id, radial_index, model_config_hash)`

Including a light config hash prevents accidental reuse when runtime knobs or template resolution parameters change between runs.

### 2) Failure handling
- Detect missing/partial output files.
- Retry policy for failed or incomplete runs.
- Fallback option:
  - use previous valid neoclassical fluxes,
  - or switch temporarily to an analytical/legacy neoclassical model,
  - or disable Jacobian refresh and reuse previous derivatives for one step.

### 3) Solver coupling philosophy
- External process execution should remain outside jitted kernels.
- JAX-side transport rhs should consume arrays already assembled from sfincs_jax results.
- Keep registry/model selection in Python-level setup code.
- If direct Python mode is enabled later, keep the transport/neoclassical boundary explicit rather than mixing sfincs_jax internals into NEOPAX core equations.

### 4) Differentiability strategy
- Do not require end-to-end differentiability in the first NEOPAX integration.
- Treat finite-difference Jacobians as the default coupling mechanism, consistent with T3D.
- If sfincs_jax direct differentiable modes become useful later, add them as an opt-in advanced path rather than the default.

## Suggested staged implementation plan

### Stage 1: MVP
- Add "sfincs_jax" model entry.
- Base flux evaluation only.
- Single radial call loop.
- External-style input/output execution.

### Stage 2: Practical transport coupling
- Add finite-difference perturbation loop for Jacobians.
- Add file-based cache and restart behavior.
- Add robust failure detection and reuse policy.

### Stage 3: Direct-library integration
- Add optional in-memory Python API path.
- Return structured diagnostics without mandatory H5 round-trip.
- Expose optional differentiable solve path where useful.

### Stage 4: Advanced coupling and performance
- Add smarter reuse of previous Jacobians between expensive solves.
- Add resource-aware batching across radial surfaces.
- Add per-radius timing, memory, and convergence diagnostics.
- Evaluate whether ambipolar-Er workflows should be integrated into the same interface or split into a separate helper path.

## Mapping to current NEOPAX architecture
- `_transport_flux_models.py`:
  - host registry entry and wrapper model class.
- `_parameters.py` + TOML runner:
  - parse and carry sfincs_jax runtime knobs.
- External runner helper module (recommended):
  - isolated process launch, I/O parsing, retry logic, cache-key management.
- `_turbulence.py` is not the right home for this path:
  - sfincs_jax should live in the neoclassical transport layer, not the turbulent one.
- Optional future helper module:
  - direct Python API adapter for sfincs_jax result extraction.

## Mapping to current sfincs_jax architecture
Relevant current sfincs_jax modules include:
- `sfincs_jax/cli.py`:
  - executable entrypoint.
- `sfincs_jax/io.py`:
  - output writing and result packaging.
- `sfincs_jax/v3.py`, `sfincs_jax/v3_driver.py`, `sfincs_jax/v3_system.py`:
  - main SFINCS-v3-style solve path.
- `sfincs_jax/residual.py`, `sfincs_jax/solver.py`, `sfincs_jax/implicit_solve.py`:
  - nonlinear/linear solve internals.
- `sfincs_jax/transport_matrix.py`, `sfincs_jax/diagnostics.py`:
  - transport-related outputs and diagnostic extraction.
- `sfincs_jax/geometry.py`, `sfincs_jax/vmec_geometry.py`, `sfincs_jax/vmec_wout.py`:
  - geometry ingestion and processing.

This means a future NEOPAX adapter can start file-based, then later migrate toward a direct-library integration without changing the transport-side abstraction.

## Bottom line
A T3D-like SFINCS coupling pattern is a strong template for sfincs_jax in NEOPAX:
- radial-point neoclassical evaluations,
- base-plus-perturb finite-difference Jacobians,
- no required theta-zeta interpolation layer,
- and transport-side composition through the neoclassical flux model registry.

The practical first target should be parity and robustness, not full differentiable coupling on day one.