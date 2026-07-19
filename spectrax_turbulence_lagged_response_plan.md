# SPECTRAX Turbulence Lagged Response Plan

## Goal

Add a turbulence lagged-response mode to NEOPAX that can consume SPECTRAX-GK
flux information through the existing transport-solver lagged-response
machinery.

This should enter NEOPAX as SPECTRAX turbulence support analogous in role to how
`ntx` provides neoclassical fluxes, but without changing the behavior of
existing turbulence or neoclassical models.

For the file-backed path, do not create a second file-reader model just for the
lagged response. Reuse the existing turbulence file-reader model and add an
opt-in lagged-response submode inside it. The non-lagged file-reader behavior
must remain exactly as it is today unless that submode is explicitly selected.

The first target is not a live SPECTRAX runtime call. The first target is a
file-backed finite-difference submode of that new turbulence model in which the
transport file contains:

- a reference turbulent flux `Q_ref`,
- perturbed turbulent flux evaluations `Q_perturb`,
- perturbation metadata sufficient to reconstruct a local Jacobian.

After that works, add a separate live SPECTRAX-GK turbulence model path that
builds the same lagged-response object from live evaluations. That live model
may later support AD-backed construction from a single differentiable
SPECTRAX-GK evaluation path.

The lagged response should represent a local affine model:

```text
Q(x) ~= Q_ref + J_ref (x - x_ref)
```

where:

- `x_ref` is the reference transport state used to build the response,
- `Q_ref` is the reference turbulent flux vector,
- `J_ref` is the local flux Jacobian with respect to the chosen transport-state
  basis.

This is a response-model question, not an integrator question. The solver may
be theta-based or Radau-based; the lagged turbulence response should be usable
through the same solver-level `build_lagged_response(...)` and
`evaluate_with_lagged_response(...)` contract.

## Scope

In scope:

- a new turbulence flux model dedicated to SPECTRAX-GK lagged responses,
- a new turbulence lagged-response object,
- a file-backed finite-difference submode of that turbulence model,
- a future live SPECTRAX-GK build path,
- an AD build path for the live SPECTRAX-GK model only after the file-backed
  path is stable,
- a transport-file extension for FD payloads,
- solver-side reuse and rebuild using the existing lagged-response controls,
- validation diagnostics comparing lagged predictions to fresh turbulence
  evaluations.

Out of scope for the first rollout:

- geometry-parameter perturbations,
- long-window production nonlinear transport claims,
- replacing the existing NTX lagged-response semantics,
- requiring live SPECTRAX-GK AD before the file-backed FD path works.

## Non-contamination requirements

Do not disturb the current NTX lagged-response path while implementing this.

Concretely:

- Do not change the semantics of the working NTX lagged-response modes.
- Do not change existing NTX benchmark defaults.
- Do not change existing turbulence-model defaults or benchmark lanes.
- Do not change the existing non-lagged file-reader turbulence behavior.
- Do not activate any of this machinery unless the SPECTRAX lagged-response
  option is explicitly selected.
- Do not reuse NTX-specific lagged-response dataclasses for turbulence payloads.
- Only share model-agnostic solver plumbing where the behavior is identical and
  separately validated.

The intended shared interface is:

```python
build_lagged_response(state, **kwargs)
evaluate_with_lagged_response(state, lagged_response, **kwargs)
pullback_build_lagged_response(...)
pullback_evaluate_with_lagged_response(...)
```

## Core design

### State basis

The lagged turbulence response should be built with respect to an explicit
transport-state basis. The first version should keep this basis small and
transparent.

Recommended initial basis:

- ion density-gradient channels,
- ion/electron temperature-gradient channels,
- optionally profile-value channels only if the turbulence closure depends
  directly on values and not only normalized gradients.

Do not include geometry channels in the first version.

### Response object

The turbulence lagged-response object should store:

```text
reference_state x_ref
reference_flux Q_ref
state_basis metadata
perturbation_steps
either:
  perturbed_flux_payloads
or:
  local_jacobian J_ref
diagnostics
```

For FD mode, `J_ref` can be reconstructed from:

```text
J_ref[:, i] ~= (Q_perturb_i - Q_ref) / delta_i
```

For AD mode, `J_ref` may be stored explicitly or represented implicitly through
JVP/VJP-compatible evaluator state.

### Solver-side evaluation

Once the response is built, evaluation inside the transport solver should be
pure algebra:

```text
delta_x = x - x_ref
Q_lagged = Q_ref + J_ref @ delta_x
```

The first implementation should materialize `J_ref` explicitly for simplicity.
Later compact tangent-only representations are allowed if they preserve the same
transport-facing behavior.

## Mode 1: file-backed FD submode of the existing turbulence file-reader model

### Purpose

This is the first rollout target.

The transport file will not just store baseline turbulent fluxes. It will also
store the perturbed turbulent flux payloads needed to reconstruct a local
finite-difference turbulence response.

This submode gives:

- a stable schema first,
- decoupling of response consumption from response generation,
- a validation target before live SPECTRAX-GK integration,
- a Trinity-like local finite-difference lagged response without requiring AD.

### Required transport-file contents

Extend the transport file schema with a versioned turbulence-response section
containing at least:

```text
response_kind = "spectrax_turbulence_fd"
reference_state_basis_name
reference_state_vector
reference_flux_vector
perturbation_labels
perturbation_steps
perturbed_flux_vectors
optional_precomputed_jacobian
metadata
```

Recommended metadata:

- state-basis ordering,
- units/normalization conventions,
- surface/radius indexing,
- species indexing,
- turbulence observable definition,
- generation timestamp or provenance tag,
- schema version.

### FD build semantics

At one rebuild point with `N` active state directions:

- one baseline flux evaluation produces `Q_ref`,
- `N` one-sided perturbed evaluations produce `Q_perturb_i`,
- optionally use centered differences later, but one-sided FD is enough to
  start.

So the first FD mode requires:

```text
1 + N
```

turbulent flux evaluations per response build.

### NEOPAX runtime behavior

At runtime:

1. Load `Q_ref`, perturbation metadata, and `Q_perturb_i` from the transport
   file.
2. Build `J_ref` in memory.
3. Construct a turbulence lagged-response object inside the selected file-reader
   turbulence model submode.
4. Use the existing solver lagged-response machinery to reuse or rebuild that
   object according to drift settings.

If the lagged-response submode is not selected, the existing file-reader model
must keep using only the baseline file fluxes exactly as it does today.

The first implementation can assume the file already exists and is correct.
Generating the file can remain an external preprocessing step.

### Validation for file-backed FD mode

At minimum, validate:

- file round-trip for all new fields,
- Jacobian reconstruction from stored perturbations,
- lagged prediction against direct baseline at `x_ref`,
- lagged prediction against one or two nearby fresh turbulence evaluations.

## Mode 2: live SPECTRAX-GK FD builder

### Purpose

After the file-backed FD submode works, allow a separate live SPECTRAX-GK
turbulence model in NEOPAX to build the same FD response live by calling a
turbulence evaluator repeatedly.

This mode uses the same response object and the same solver-side evaluation
path. Only the response-construction source changes.

### Build semantics

Given a live turbulence flux function `Q(x)`:

1. evaluate `Q_ref = Q(x_ref)`,
2. for each active state direction `i`, evaluate
   `Q_perturb_i = Q(x_ref + delta_i e_i)`,
3. form `J_ref`.

The resulting response object should be identical in meaning to the one loaded
from file.

### Benefit

This mode proves that the lagged-response API is not tied to file I/O and will
make later AD integration cleaner.

## Mode 3: live SPECTRAX-GK AD builder

### Purpose

Allow a differentiable SPECTRAX-GK flux lane to build the same turbulence
lagged-response object from a single live evaluation path plus tangent
information.

### AD semantics

If the turbulence flux function is differentiable enough:

```text
Q_ref = Q(x_ref)
J_ref = dQ/dx evaluated at x_ref
```

The first AD implementation may materialize `J_ref` explicitly even if that is
not the final desired representation.

Later compact representations may use JVP/VJP logic instead, but the transport
interface should remain the same.

### Important caution

AD mode should not be the first milestone unless the selected SPECTRAX-GK
observable is already known to be stable and differentiable.

If the target flux depends on:

- noisy nonlinear windows,
- non-smooth postprocessing,
- Python reporting code,
- branchy eigenvector selection,

then the first AD target should be a reduced or quasilinear turbulence
observable whose tangent path is already validated.

### Benefit

AD mode can reduce response-build cost from:

```text
1 + N evaluations
```

to:

```text
1 evaluation path plus tangent extraction
```

when the observable and implementation are suitable.

## Reuse and rebuild policy

Use the existing solver-level lagged-response controls first.

Recommended first settings:

```toml
radau_rhs_mode = "lagged_response"
lagged_response_reuse_mode = "global_state_drift"
lagged_response_reuse_rtol = ...
lagged_response_reuse_atol = ...
```

The selected SPECTRAX turbulence model should not invent its own competing
accepted-step or rejected-step policy unless later evidence shows that the
generic global drift criterion is insufficient.

Possible later addition:

- a prediction-mismatch rebuild trigger comparing lagged `Q` to a fresh
  turbulence evaluation at occasional validation points.

## Recommended rollout order

### Phase 1: schema and file-backed FD submode

1. Define the turbulence lagged-response dataclass.
2. Extend the transport file schema for baseline and perturbed flux payloads.
3. Add a file loader that reconstructs `J_ref`.
4. Extend the existing turbulence file-reader model with an opt-in FD
   lagged-response submode while preserving the current non-lagged behavior.
5. Add unit tests and small synthetic integration tests.

### Phase 2: live SPECTRAX-GK FD builder

1. Define a live turbulence flux callback API.
2. Build `Q_ref` and `Q_perturb_i` from live calls.
3. Implement that as a separate live SPECTRAX-GK turbulence model path.
4. Reuse the exact same response object and evaluator from Phase 1.
5. Compare live-built FD responses against file-backed FD responses.

### Phase 3: live SPECTRAX-GK AD builder

1. Define a differentiable turbulence flux API.
2. Add AD-based Jacobian extraction.
3. Build the same response object from AD.
4. Compare AD Jacobian columns against FD Jacobian columns.
5. Gate rollout on AD-vs-FD agreement.

## Validation matrix

### FD file mode

- schema round-trip,
- Jacobian reconstruction sanity,
- exact recovery at `x_ref`,
- nearby-state lagged-prediction checks.

### Live FD mode

- equality with file-backed FD on the same generated payload,
- rebuild and reuse behavior across a small transport solve.

### AD mode

- columnwise AD-vs-FD Jacobian comparison,
- lagged-prediction agreement with FD mode,
- reverse-mode smoke test if transport objectives consume the turbulence lagged
  response in differentiated solves.

## Open questions

1. What exact SPECTRAX-GK turbulent observable should define `Q` first?
2. What is the first active-state basis: gradients only, or gradients plus
   profile values?
3. Is the first target quasilinear, reduced nonlinear, or true nonlinear
   window-averaged flux?
4. Should the transport file store only `Q_perturb` or also a precomputed
   `J_ref` for convenience?
5. How large can the active state basis be before explicit FD becomes too
   expensive?
6. Should the first response be per-radius local, or a coupled multi-radius
   vector response?
7. What minimum diagnostic payload is needed to audit stale or mismatched
   response files?

## Immediate next step

Implement Phase 1 only:

- transport-file schema extension for turbulence FD payloads,
- NEOPAX loader for `Q_ref` plus `Q_perturb`,
- in-memory Jacobian reconstruction,
- lagged turbulence evaluation through the existing solver machinery.

Do not block Phase 1 on live SPECTRAX-GK AD support.
