# SFINCS_JAX Connection Plan

## Goal
Connect `sfincs_jax` to NEOPAX in two stages:

1. Short term:
   Build a standalone postprocessing workflow that takes a completed NEOPAX run, evaluates new `sfincs_jax` particle fluxes, heat fluxes, and `Uparallel` at selected radial locations, and runs those surface solves in parallel.
2. Long term:
   Reuse the same connection layer from inside NEOPAX during transport evolution, with controllable refresh criteria so `sfincs_jax` is called only when scientifically useful and computationally affordable.

This plan is intentionally centered on one shared adapter so we do not build one path for offline analysis and a separate incompatible path for online transport coupling.

## Design Principles

- Keep the `sfincs_jax` boundary explicit and Python-level.
- Make one local-surface adapter the core primitive.
- Use subprocess-based parallelism first for robustness.
- Prefer a file-compatible output format that NEOPAX can already read.
- Add live in-transport coupling only after the offline workflow is validated.
- Make refresh/reuse criteria configurable from TOML.

## Existing Relevant Pieces

### In `sfincs_jax`
- `write_sfincs_jax_output_h5(...)` already provides a clean single-surface execution path and can return in-memory results.
- `sfincs_jax` already supports process-based parallel execution patterns and GPU worker assignment.
- One namelist corresponds naturally to one local flux-surface solve.

### In NEOPAX
- The current transport model interface already expects `Gamma`, `Q`, and `Upar`.
- `FluxesRFileTransportModel` already reads `r`, `Gamma`, `Q`, and `Upar` from HDF5 profiles.
- The transport model registry is already the natural place for a future `sfincs_jax` model.
- Existing state and geometry objects already provide the data needed to build local profile inputs.

## Target Architecture

### Shared Core Adapter
Create one small `sfincs_jax` connection layer with four responsibilities:

1. Extract local inputs from a NEOPAX state at one radius.
2. Build or patch a `sfincs_jax` namelist for that local surface.
3. Execute `sfincs_jax` and read back `Gamma`, `Q`, and `Upar`.
4. Return results in a normalized NEOPAX-friendly structure.

This adapter should be usable both:
- offline from a standalone script,
- and online from a NEOPAX transport flux model.

### Proposed Modules

- `NEOPAX/NEOPAX/_sfincs_jax_adapter.py`
  Core single-surface adapter, input preparation, execution, result parsing.
- `NEOPAX/examples/Solve_Er_General/postprocess_sfincs_jax_fluxes.py`
  Standalone postprocessing driver for completed NEOPAX runs.
- `NEOPAX/NEOPAX/_transport_flux_models.py`
  Future registry integration via a new `sfincs_jax_external` transport model.

## Phase 1: Standalone Postprocessing Workflow

### Scope
Given a completed NEOPAX run, evaluate updated:
- particle flux `Gamma`,
- heat flux `Q`,
- parallel flow `Upar`,

at selected radial points using `sfincs_jax`.

### Inputs
The standalone workflow should accept:
- a NEOPAX output state file or a clearly defined NEOPAX snapshot source,
- a `sfincs_jax` template namelist,
- geometry input path or equilibrium override,
- a list of radial points or a radial selection rule,
- execution settings for parallel workers and device placement.

### Data Extraction
For each selected radial point, extract:
- species densities,
- species temperatures,
- local `Er`,
- local density gradients,
- local temperature gradients,
- local radius selector such as `rN_wish`.

Where NEOPAX stores pressure instead of temperature, reconstruct temperature consistently from pressure and density using the same conventions already used in the codebase.

### Output
Write a compact HDF5 file with datasets:
- `r`
- `Gamma`
- `Q`
- `Upar`

Optional metadata:
- selected radii,
- source NEOPAX file,
- source `sfincs_jax` template,
- species labels,
- runtime settings,
- timing and convergence diagnostics if available.

This format should remain compatible with the existing NEOPAX `read_flux_profile_file(...)` path so the output can immediately be reused as a profile-based transport input if needed.

### Parallel Execution Strategy
Use outer parallelism across radial surfaces:

- CPU mode:
  One subprocess per surface up to a configured worker limit.
- GPU mode:
  One subprocess per visible GPU with `CUDA_VISIBLE_DEVICES` pinned per worker.

Reasoning:
- each surface solve is independent,
- subprocesses isolate JAX device state cleanly,
- this avoids fragile in-process device scheduling early on.

### Deliverables

1. Single-surface adapter API.
2. Standalone postprocessing script.
3. HDF5 output writer for `r/Gamma/Q/Upar`.
4. Parallel execution support.
5. Minimal documentation and example invocation.

## Phase 2: Validation of the Offline Path

### Objectives
Verify that the standalone workflow is scientifically and numerically trustworthy before coupling it into transport.

### Checks

1. Single-surface parity:
   Compare adapter outputs against direct `sfincs_jax` reference runs for the same local inputs.
2. Multi-surface consistency:
   Check that assembled radial profiles are ordered and mapped correctly.
3. Parallel reproducibility:
   Confirm parallel surface runs match sequential results within tolerances.
4. File compatibility:
   Confirm the generated HDF5 can be read by `FluxesRFileTransportModel`.

### Success Criteria
- no shape or normalization mismatches,
- acceptable numerical agreement between sequential and parallel execution,
- output file works as a drop-in NEOPAX flux profile source.

## Phase 3: NEOPAX Integration

### Scope
Add a new transport flux model that calls the shared adapter during transport evolution.

### Proposed Model Name
- `sfincs_jax_external`

This model should implement the same transport model contract as existing models and return:
- `Gamma`
- `Q`
- `Upar`

with the same array layout expected by NEOPAX transport equations.

### Integration Point
Integrate through the existing transport model registry in `NEOPAX/_transport_flux_models.py`.

The model should:
- accept the current NEOPAX state,
- choose the radial points to evaluate,
- call the adapter,
- assemble returned profiles onto the transport grid,
- optionally reuse cached values when no refresh is needed.

### First Integration Target
Do not attempt full Jacobian-aware tightly coupled `sfincs_jax` transport on the first integration.

Instead:
- start with flux refresh at selected iterations,
- return updated profiles,
- let the existing solver consume them through the standard flux path.

## Refresh and Reuse Policy

### Why This Matters
Calling `sfincs_jax` every single transport sub-iteration may be too expensive. We need configurable refresh criteria.

### Proposed Control Knobs
Add TOML options such as:

- `sfincs_jax_enable = true`
- `sfincs_jax_template = "..."`
- `sfincs_jax_mode = "postprocess" | "live"`
- `sfincs_jax_parallel_backend = "cpu" | "gpu"`
- `sfincs_jax_parallel_workers = N`
- `sfincs_jax_radial_indices = [...]`
- `sfincs_jax_radial_stride = N`
- `sfincs_jax_update_every_timestep = N`
- `sfincs_jax_update_every_nonlinear_iter = N`
- `sfincs_jax_min_profile_change = 0.0`
- `sfincs_jax_max_stale_iterations = N`
- `sfincs_jax_reuse_last_valid = true`
- `sfincs_jax_write_debug_outputs = false`

### Recommended Initial Refresh Logic
Refresh if any of the following are true:

1. It is the first transport iteration.
2. `step_index % update_every_timestep == 0`.
3. `nonlinear_iter % update_every_nonlinear_iter == 0`.
4. The relative change in local `n`, `T`, or `Er` exceeds a configured threshold.
5. The current `sfincs_jax` fluxes have been reused for more than `max_stale_iterations`.

This gives a controlled balance between cost and fidelity.

## Caching and Warm Starts

### Short-Term Requirement
Cache by a stable key that includes:
- radius index,
- local profile state,
- geometry identifier,
- template hash,
- execution mode.

### Long-Term Requirement
Support optional warm starts using prior `sfincs_jax` state if that path proves stable and useful.

### Fallback Behavior
If a refresh fails:
- reuse the last valid fluxes if configured,
- log the failure clearly,
- optionally mark the timestep as degraded in diagnostics.

## Radial Sampling Strategy

### Short Term
Allow user-selected radial points:
- explicit list,
- evenly spaced subset,
- all points,
- face grid or center grid, depending on how the final coupling is defined.

### Long Term
Match the transport discretization chosen by NEOPAX:
- likely face-centered for closure-consistent flux evaluation,
- or a clearly documented interpolation from selected evaluation points to the solver grid.

The first implementation should keep the choice simple and explicit rather than trying to be overly clever.

## Iteration Criteria for the Live Coupling

### Minimum Viable Policy
Use a fixed refresh cadence:
- every `N` time steps,
- optionally every `M` nonlinear iterations inside a time step.

### Better Policy
Use an adaptive trigger:
- refresh only if `n`, `T`, or `Er` changed enough since the last `sfincs_jax` call.

### Suggested First Live Default
- refresh at the start of each transport time step,
- do not refresh on every nonlinear iteration,
- allow optional extra refresh if `Er` changes above a threshold.

This is a conservative first live strategy that should keep cost manageable while still reacting to important profile changes.

## Staged Implementation Plan

### Stage 1: Adapter MVP
- Create `_sfincs_jax_adapter.py`.
- Implement one-surface local input extraction.
- Implement one-surface `sfincs_jax` run and result parsing.
- Return normalized `Gamma/Q/Upar`.

### Stage 2: Standalone Profile Tool
- Add postprocessing script for completed NEOPAX runs.
- Add radial loop over selected surfaces.
- Add HDF5 output writer.
- Add CPU and GPU subprocess parallelism.

### Stage 3: Offline Validation
- Validate against direct `sfincs_jax` runs.
- Validate sequential versus parallel.
- Validate HDF5 compatibility with NEOPAX profile readers.

### Stage 4: NEOPAX Live Model
- Add `sfincs_jax_external` transport model entry.
- Add TOML parsing for runtime controls.
- Add refresh/reuse policy.
- Return assembled `Gamma/Q/Upar` to the transport stack.

### Stage 5: Live-Coupling Optimization
- Add caching.
- Add warm starts if useful.
- Add selective radial subsampling.
- Add richer diagnostics and failure recovery.

## Risks and Open Questions

### Scientific
- exact normalization and coordinate mapping must be verified carefully,
- choice of center-grid versus face-grid evaluation must be made explicit,
- consistency of gradients passed from NEOPAX to `sfincs_jax` must be documented.

### Numerical
- parallel process runs may expose small reproducibility differences,
- over-refreshing may make transport too expensive,
- under-refreshing may make transport stale or inconsistent.

### Software
- JAX device management is easier in subprocesses than in a shared process,
- file naming and cache-key design need to avoid accidental reuse,
- live integration should not contaminate jitted solver internals with process-launch logic.

## Recommended Immediate Next Step
Build the standalone postprocessing workflow first.

That gives:
- direct scientific value on completed NEOPAX runs,
- a reusable single-surface adapter,
- a validated output format,
- and a low-risk foundation for the later live NEOPAX coupling.

Only after that is working cleanly should the `sfincs_jax_external` live transport model be added.
