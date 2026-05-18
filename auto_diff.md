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

## Magnetic-Configuration Differentiability Test Plan

This section focuses on a more specific target than profile optimization:

- differentiate a final transport-state diagnostic
- with respect to magnetic-configuration parameters
- starting from the lagged exact-runtime NTX response path currently being used

The motivating example is a VMEC shape coefficient such as `RBC`.

The intended comparison is:

- automatic differentiation
- finite differences

for final-state diagnostics such as:

- the maximum of `Er`
- the radial position of an `Er` root / sign change

### First important caveat: use smooth diagnostics first

The first AD-vs-FD checks should use **smooth scalar objectives**.

While the final goal may include:

- `max(Er)`
- root position of `Er`

these are not ideal first diagnostics because they are not fully smooth:

- `max(Er)` can switch active index
- root location can jump if the profile flattens or multiple crossings compete

So the recommended first objectives are:

- soft maximum of final `Er`
- integrated `Er^2`
- smooth weighted radial center of positive `Er`
- a smoothed zero-crossing locator
- volume-averaged alpha power

Only after these pass should we move to sharper diagnostics.

### Feasibility

This is feasible, but in stages.

There are really two different questions:

1. can we differentiate through the lagged exact-runtime NTX transport path?
2. can we differentiate all the way back to a VMEC parameter like `RBC`?

The answer is:

- question 1: likely yes, and this should be tested first
- question 2: probably feasible, but it will likely require a new magnetic-input mode

The main reason is that the current setup is still largely file-driven:

- transport geometry is built from `vmec_file` + `boozer_file`
- exact-runtime NTX support is prebuilt from those files

That is good for forward solves, but not the right interface for
parameter-differentiable magnetic design.

### Recommended staged plan

#### Stage A: Local lagged NTX differentiability

Before touching magnetic parameters directly, verify that the lagged
exact-runtime NTX path is differentiable with respect to already-live local
inputs.

Suggested first parameters:

- local `Er`
- local density
- local temperature

Suggested outputs:

- one transport moment
- one `Lij` entry
- one flux component

Test:

- compare JAX `grad` / `jvp` against finite differences
- do this first for a single radius or reduced local problem

This checks the local lagged-response differentiability in isolation.

#### Stage B: Full transport differentiability with frozen geometry

Keep the magnetic files fixed and test AD through the full transport solve.

Suggested parameters:

- `n0`
- `T0`
- `density_shape_power`
- `temperature_shape_power`

Suggested objectives:

- softmax(final `Er`)
- integrated final `Er^2`
- smooth `Er` root-location proxy
- volume-averaged alpha power

Success criterion:

- AD and FD agree for final-state diagnostics
- lagged exact-runtime NTX remains stable under differentiation

This validates the whole transport rollout before adding magnetic design
variables.

#### Stage C: Geometry-channel sensitivity bridge

Before going all the way to `RBC`, test sensitivity to geometry quantities that
the runtime support already depends on.

Examples include:

- `dr/ds`
- `iota`
- `B00`

or other smooth geometry channels, if exposed conveniently.

This acts as a bridge between:

- transport-state differentiation
- and magnetic-parameter differentiation

If this stage is problematic, then going directly to `RBC` is premature.

#### Stage D: New magnetic-input mode

To differentiate with respect to `RBC` itself, add a new magnetic-input mode
that is not centered on static files.

Conceptually, the new path should look like:

```text
theta_mag -> vmec_jax equilibrium -> Boozer/surface representation -> NTX prepared support -> transport rollout
```

Instead of:

```text
vmec_file + boozer_file -> static support -> transport rollout
```

The new mode should accept an in-memory magnetic configuration or equilibrium
object as the true differentiable input.

#### Stage E: Magnetic-parameter AD-vs-FD test

Once the new magnetic-input mode exists:

- choose one small `RBC` perturbation direction
- compute a transport objective from the final state
- compare AD against finite differences

Suggested initial objectives:

- softmax(final `Er`)
- smooth weighted center of positive `Er`
- smooth `Er` root-location proxy
- volume-averaged alpha power

Only later:

- `max(final Er)`
- root-position diagnostics

### Suggested smooth root-position proxy

For a first differentiable proxy of the `Er` root location, avoid a hard
sign-change detector and instead use a smooth weight that concentrates near
small `|Er|`.

One practical form is:

```text
w_i = exp(-beta * |Er_i|)
r_root_proxy = sum(r_i * w_i) / sum(w_i)
```

where:

- `r_i` are the radial coordinates
- `beta` controls how tightly the weight concentrates near the zero-crossing

This is not identical to a hard root finder, but it is a much better first
diagnostic for AD-vs-FD agreement.

If needed, this can later be refined so that it also prefers locations where
the profile changes sign, but the simple near-zero weighted centroid is a good
first test metric.

### Suggested volume-averaged alpha-power objective

To include a physics scalar depending on other transport-state components, add
an objective based on volume-averaged alpha power.

Conceptually:

```text
J_alpha = <P_alpha(state_final)>
```

This is useful because it brings in sensitivity to:

- density
- temperature
- and any magnetic-configuration influence acting through the evolved final state

So it complements the `Er`-focused diagnostics and helps test whether the AD
path is behaving sensibly for broader state couplings as well.

### Why a new magnetic-input mode is likely needed

The current exact-runtime support build is centered on:

- `vmec_file`
- `boozer_file`
- precomputed support from those files

For real magnetic-parameter differentiation, the natural interface should be:

- parameterized VMEC state
- parameterized Boozer/surface representation

not file paths.

So yes, a new mode is likely needed if the target is genuine `RBC`
differentiation.

### Key technical uncertainty

The largest uncertainty is not the lagged response itself.

It is whether the full chain:

- VMEC parameter update
- equilibrium update
- Boozer / surface conversion
- NTX prepared support construction

is available in a form that is both:

- differentiable
- practical enough for repeated AD-vs-FD comparisons

That should be treated as a separate milestone.

### Concrete testing ladder

1. local lagged NTX AD vs FD
2. full transport AD vs FD with frozen geometry
3. geometry-channel AD vs FD
4. add differentiable magnetic-input mode
5. `RBC` AD vs FD on final-state transport diagnostics

### Practical recommendation

The first implementation should **not** start by differentiating with respect
to `RBC` directly.

The safer order is:

1. prove the lagged NTX transport stack differentiates correctly with frozen geometry
2. start with physically meaningful initial-profile parameters such as `n0`, `T0`, `density_shape_power`, and `temperature_shape_power`
3. then introduce the new magnetic-input mode
4. only then test `RBC`

### Bottom line

Yes, the idea makes sense.

Yes, it is likely feasible.

But the right plan is:

- first validate AD through the lagged exact-runtime NTX transport path itself
- then add a differentiable in-memory magnetic-input mode
- then test VMEC-parameter sensitivities such as `RBC` against finite differences

## Plan: Full-Rollout Gradients with Adaptive Solver Logic Preserved

This section records the current planning direction after the one-step
diagnostic succeeded but the full adaptive rollout failed AD-vs-FD parity.

### Planning goal

The goal is:

- keep the normal full transport solve as the production forward path
- obtain reliable derivatives for final-state objectives
- keep memory and wall time as low as possible
- avoid blindly inheriting a generic adjoint strategy if a more
  physics-informed solver-aware strategy is available

In particular, the target is:

- full-rollout derivatives as trustworthy as the one-step derivatives
- without giving up the current adaptive acceptance / retry logic in the
  forward solve

### What we know now

From the current diagnostics:

- local lagged NTX differentiation is good
- one accepted transport step differentiates very well
- the full adaptive rollout is where AD-vs-FD fails
- the full-rollout mismatch is strongly correlated with adaptive path changes
  across nearby parameter values

So the main problem is not:

- local NTX differentiability
- or the accepted-step physics map itself

The main problem is:

- differentiating the full adaptive retry/accept/reject trace naively

### Design principle

We should distinguish two maps:

1. the raw implemented solver trace
   - includes every rejected trial and controller branch
2. the accepted transport evolution map
   - the physically relevant map that takes one accepted state to the next

The planning direction is:

- preserve the raw adaptive logic in the forward solve
- but build differentiation around the accepted-step / accepted-rollout map

This is the key way to keep the current forward behavior while avoiding the
worst branch-sensitivity in gradients.

### Efficiency principle

The default assumption should be:

- do not use a generic library adjoint blindly if it is more expensive than
  necessary for this known physics system

We should prefer:

- reduced local responses
- accepted-step replay rather than full trace storage
- custom VJP/JVP rules at solver-relevant boundaries
- small saved state
- physics-informed structure whenever it lowers cost

Diffrax-like ideas are useful as references, but the target is not “copy
Diffrax”; the target is:

- a NEOPAX-specific gradient path that is at least as reliable
- and ideally more efficient for this transport problem

### Staged plan

#### Stage 0: Keep the current forward benchmark as the baseline

The current lagged exact-runtime NTX transport benchmark remains the forward
reference case.

We continue to measure:

- wall time
- accepted-step count
- rejected-step behavior
- gradient parity diagnostics

This ensures all later AD work is judged against the real production solve.

#### Stage 1: Strengthen full-rollout diagnostics

Before changing the differentiation design, improve the diagnosis of the full
rollout mismatch.

Add or keep:

- FD step-size sweep for the full solve
- accepted-step count comparison
- accepted-step size sequence comparison
- rejected-attempt count comparison
- possibly Newton-iteration summaries

Purpose:

- determine how much of the mismatch is pure FD instability
- determine how much is branch/path divergence

#### Stage 2: Freeze accepted-step sequence as a diagnostic reference

Add a diagnostic mode that:

- runs the baseline full solve once
- records the accepted timestep sequence
- reruns nearby parameter values on that fixed accepted sequence

Purpose:

- remove controller/retry path drift from the FD comparison
- test whether the accepted-rollout map itself has good parity

Interpretation:

- if parity becomes good, then the main issue is adaptive controller path
  sensitivity
- if parity is still bad, then the issue is deeper in the accepted rollout map

This is a diagnostic tool, not necessarily the final production gradient path.

#### Stage 3: Accepted-step custom differentiation

Promote the accepted transport step to the main differentiated object:

```text
Y_{n+1} = Phi(Y_n, theta)
```

Forward:

- keep the current adaptive nonlinear solve
- accept the converged step as usual

Backward:

- differentiate the accepted-step map
- do not explicitly backpropagate through every rejected trial

This is the most important architectural shift.

#### Stage 4: Accepted-rollout replay differentiation

For the full run:

- record the accepted-step sequence or enough information to replay it
- define the backward pass on the accepted sequence

This means:

- rejected steps remain forward-only implementation details
- gradients are built from the realized accepted rollout

This is likely the closest efficient analogue to a Diffrax-style
solver-aware differentiation strategy, while still staying custom to NEOPAX.

#### Stage 5: Controller-gradient policy

Make an explicit policy decision about the controller logic.

Likely best choices are:

- controller decisions remain active in the forward solve
- controller branch logic is not treated as a primary smooth object in the
  backward pass

Possible implementations:

- stop-gradient through retry/controller branch updates
- or differentiate only the accepted-step replay map

This is likely necessary to keep gradients stable without distorting the
forward adaptive solver.

#### Stage 6: Compare against a Diffrax reference solver only as a benchmark

If needed, compare against a mature differentiable implicit solver such as
Kvaerno5 in Diffrax.

Use this comparison only to answer:

- does a generic solver-aware AD path produce better full-rollout parity?
- what are the memory and wall-time costs?

Do not adopt it blindly.

The decision criterion should be:

- reliability
- memory
- speed
- ability to exploit the known transport/NTX structure

If NEOPAX-specific accepted-step differentiation is cheaper and equally
reliable, that should be preferred.

### Concrete benchmarking targets

Each stage should be judged on:

1. gradient parity
   - AD vs FD for full-rollout objectives
2. forward cost
   - does the normal transport solve stay unchanged?
3. gradient cost
   - memory and time for derivative evaluation
4. robustness
   - does parity hold across `n0`, `T0`, `density_shape_power`,
     `temperature_shape_power`?

### Priority order

The recommended order is:

1. improve full-rollout diagnostics
2. add fixed-accepted-sequence diagnostic mode
3. design accepted-step custom differentiation
4. design accepted-rollout replay differentiation
5. only then compare against Diffrax/Kvaerno5 as a reference point

### Short summary of the plan

The plan is not to remove adaptive logic.

The plan is:

- keep the current adaptive forward solver
- diagnose full-rollout path divergence carefully
- move differentiation toward accepted-step / accepted-rollout replay rules
- use Diffrax only as a benchmark reference, not as the default solution
- aim for a NEOPAX-specific gradient path that is both reliable and efficient
