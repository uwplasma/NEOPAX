# Automatic Differentiation Strategy for Transport Optimization

This note describes the recommended automatic-differentiation architecture for NEOPAX when the end goal is profile optimization through the transport evolution, i.e. differentiating a final transport state with respect to the initial state and selected model or profile parameters.

The main conclusion is:

- `lagged_response` should remain a **forward-solve acceleration strategy**
- gradients for optimization should be built around **custom differentiation at the transport step or rollout level**
- we should avoid relying on naive reverse-mode differentiation through the full implicit solve trace

## Optimization Goal

The target use case is:

- optimize initial density / temperature / electric-field profiles
- or profile-parameterized controls
- by differentiating an objective depending on the final transport state

Conceptually:

```text
Y_final = RunTransport(Y0, theta)
J = Objective(Y_final, theta)
```

We want stable and memory-efficient access to:

```text
dJ/dY0, dJ/dtheta
```

where:

- `Y0` is the initial transport state
- `theta` denotes optimization or model parameters

## Why the Naive AD Path Is Not Enough

A direct reverse-mode trace through the current transport solver is expensive because it combines:

1. expensive local transport physics
   - exact-runtime NTX neoclassical solves
   - turbulent flux models

2. implicit timestepping
   - Radau stages
   - Newton iterations
   - repeated residual evaluations

3. rollout history
   - multiple accepted steps
   - possible rejected steps

If treated naively, reverse-mode AD attempts to propagate through:

- the full NTX solve trace
- every stage residual evaluation
- every Newton iteration
- the entire saved rollout

This can become prohibitive in:

- peak memory
- compile time
- total gradient wall time

## Role of `lagged_response`

The current `lagged_response` framework is still the right forward-model abstraction.

Its purpose is:

- reduce repeated expensive flux evaluations within one implicit step attempt
- keep the surrounding transport assembly live
- provide a model-agnostic interface for expensive flux-response approximations

It should **not** be treated as the full AD solution by itself.

The intended separation of concerns is:

- `lagged_response`
  - accelerate the forward implicit solve

- custom differentiation rules
  - make optimization gradients practical

## Recommended Architecture

The recommended architecture has three levels.

### Level 1: Local flux-response model

At the flux-model level:

- use reduced local responses
- keep exact-runtime NTX differentiation close to the prepared solve boundary
- avoid dense full-state Jacobians

This is already the direction of the current D1 implementation:

- local NTX response uses reduced transport moments
- on-demand `jvp(...)` is preferred over explicit dense Jacobians
- optional batching controls handle radial and scan dimensions

This layer is the correct place to:

- reduce local memory
- reduce local compile size
- encode model-specific response approximations

### Level 2: Transport step map

The next key abstraction is one accepted transport step:

```text
Y_{n+1} = Phi(Y_n, theta)
```

This map should eventually expose a custom differentiation rule.

The forward pass:

- runs the chosen implicit solver step
- can use `black_box`, `lagged_response`, or other step-local strategies

The backward pass:

- should **not** differentiate through every Newton iteration and stage trace
- should instead use an implicit/linearized view of the accepted step map

This is the most important architectural shift for optimization.

### Level 3: Rollout map

The full transport run is the composition of step maps:

```text
Y_final = Phi_{N-1} o ... o Phi_1 o Phi_0 (Y0, theta)
```

The long-term optimization-oriented target is:

- a custom VJP for the full rollout
- or a backward adjoint-style integration over accepted steps

This avoids storing the full reverse-mode trace of:

- all implicit iterations
- all internal stage evaluations
- all local physics intermediates

## Recommended Differentiation Strategy

### Preferred approach

Use:

- forward acceleration with `lagged_response`
- custom differentiation at the **transport-step** or **rollout** level

Do **not** depend on:

- naive reverse-mode through the entire solver implementation

### Why this is preferable

This gives a better tradeoff between:

- physical fidelity
- optimization gradients
- memory usage
- compile size

In particular:

- local NTX differentiation remains available where needed
- but the global solver does not need to retain the entire internal trace

## Interaction with NTX

NTX already exposes the right low-level ingredients:

- prepared monoenergetic solve paths
- explicit custom-VJP solve entry points

This suggests the correct division of responsibilities:

- NTX:
  - solve-level derivative contract
  - local monoenergetic sensitivities

- NEOPAX:
  - reduced transport response
  - implicit step map
  - rollout-level custom differentiation

This is better than pushing all AD responsibility into:

- the full transport solver trace
- or a single huge JAX graph

## Recommended State Representation for AD

For optimization, the AD path should use the smallest physically meaningful saved information.

### At the local flux-model level

Prefer saving:

- reduced transport moments
- reference local inputs
- compact lagged-response state

Avoid saving:

- full coefficient scans if reduced moments suffice
- dense Jacobians if directional derivatives or implicit rules suffice

### At the step level

Prefer saving:

- accepted state `Y_n`
- accepted next state `Y_{n+1}`
- timestep size
- response objects or linearization data needed by the accepted step map

Avoid saving:

- every Newton iterate
- every internal stage state unless the custom backward rule genuinely requires them

### At the rollout level

Prefer:

- accepted-step checkpoints
- checkpoint/replay or adjoint strategies

Avoid:

- full naive reverse-mode history through the entire rollout

## State-of-the-Art JAX Practices for This Problem

For this transport-optimization setting, the most appropriate JAX practices are:

### 1. Differentiate the smallest useful object

Do not differentiate unnecessarily large outputs.

Prefer:

- reduced transport moments
- local response variables

instead of:

- full coefficient histories
- full assembled solver traces

### 2. Prefer JVPs for local response models

For step-local lagged responses, prefer:

- `jax.jvp(...)`

over:

- explicit dense `jacfwd(...)` Jacobian materialization

This is especially appropriate when:

- only the actual local perturbation is needed
- memory matters

### 3. Use `custom_vjp` at real solve boundaries

Use `custom_vjp` where there is a clear mathematical solver map, especially:

- prepared NTX solve boundaries
- accepted transport step maps
- full rollout map

Do not scatter custom rules across arbitrary small helper functions unless there is a strong reason.

### 4. Use batching hierarchies rather than global full-axis batching

Prefer:

- chunked `lax.map` over heavy outer dimensions
- `vmap` inside manageable chunks

over:

- single giant global `vmap` across all heavy axes

This is already relevant for:

- radial batching
- monoenergetic scan batching

and may also become relevant for:

- implicit stage batching

### 5. Use checkpointing only as a memory rescue mechanism

`jax.checkpoint` / rematerialization is appropriate when:

- peak memory is the blocker

It should not be the primary speed strategy because it trades:

- lower memory

for:

- more recomputation

For this application, checkpointing should be targeted at:

- expensive local solve boundaries

not broad regions of cheap algebra.

## What Should Be Prioritized Next

### Priority 1: Step-level custom differentiation

Implement a custom VJP for one accepted transport step:

```text
Y_{n+1} = Phi(Y_n, theta)
```

This should become the main gradient interface for optimization-oriented transport stepping.

Benefits:

- large memory reduction
- cleaner treatment of implicit solvers
- avoids backpropagating through all Newton and stage internals

### Priority 2: Rollout-level custom differentiation

Once step-level differentiation is stable, implement:

```text
Y_final = RunTransport(Y0, theta)
```

with:

- checkpointed replay
- or adjoint-style backward propagation over accepted steps

Benefits:

- scalable final-state optimization
- stable memory use across long runs

### Priority 3: Selective lagging by transport component

Allow:

- lag neoclassical NTX response
- keep analytical turbulence live

This is useful because:

- it keeps the lagged response targeted at the expensive physics
- it reduces lagged payload size
- it simplifies attribution of accuracy and performance effects

### Priority 4: Solver-side memory reduction

If still needed after the above:

- reduce stage batching memory in Radau
- reduce unnecessary saved-loop carry state in specialized benchmarking or optimization modes

These are valuable, but they are secondary to the step/rollout differentiation architecture.

## Suggested End-State Workflow

The desired long-term workflow is:

1. forward transport solve
   - use `lagged_response` to accelerate expensive flux evaluation

2. accepted-step map
   - expose custom backward rule

3. rollout objective differentiation
   - use accepted-step checkpoints or adjoint replay

4. optimization
   - differentiate final-state objectives with practical memory use

This gives a coherent architecture where:

- `lagged_response` improves the forward solve
- custom step/rollout differentiation makes optimization feasible

## Short Summary

The best next-generation AD architecture for NEOPAX profile optimization is:

- keep `lagged_response` as the forward acceleration mechanism
- do not rely on naive reverse-mode through the full implicit solver
- introduce custom differentiation at the transport-step level
- extend that to the rollout level for final-state optimization
- continue using reduced local NTX responses and prepared-solve derivative contracts

This is the most promising route to simultaneously improve:

- memory efficiency
- optimization gradient quality
- scalability to expensive neoclassical and turbulent models

