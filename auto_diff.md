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
- naive full-rollout central FD is unstable across FD step size
- full-rollout accepted times and accepted `dt` sequences drift even when the
  saved accepted-mask pattern still looks the same

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

Important clarification:

- identical accepted history is **not** the final gradient goal
- replaying or freezing accepted history is only a diagnostic and
  construction tool
- the final target is an accepted-trajectory-based backward pass for the full
  adaptive solve

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

#### Stage 1: Finish the essential full-rollout diagnostics

At this point the main architectural conclusion is already clear:

- local/one-step differentiation is sound
- naive full adaptive trace differentiation is not the right object

So only the minimum remaining diagnostics should be kept.

Keep:

- accepted-step count comparison
- accepted-step size sequence comparison
- final accepted-step nonlinear summary

Purpose:

- confirm where the accepted trajectory begins to diverge
- avoid spending more time trying to rescue naive full-rollout FD as the main
  truth reference

Current status:

- this stage is effectively complete for `n0`
- the evidence already supports changing the autodiff path

#### Stage 2: Accepted-history replay as a diagnostic only

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

This is **not** the final production gradient definition.

It is only used to answer:

- whether the accepted trajectory is the right object to build the backward
  pass around
- and whether controller/retry branching is the dominant source of parity loss

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

- record the accepted-step sequence or enough information to reconstruct the
  realized accepted trajectory
- define the backward pass on that realized accepted trajectory

This means:

- rejected steps remain forward-only implementation details
- gradients are built from the realized accepted rollout

This should be understood as:

- a solver-aware backward pass for the adaptive solve
- not as a requirement that nearby parameter values always share identical
  accepted histories

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

1. finish the minimum accepted-history diagnostic
2. design accepted-step custom differentiation
3. design accepted-rollout accepted-trajectory differentiation
4. make the controller-gradient policy explicit
5. only then compare against Diffrax/Kvaerno5 as a reference point

### Short summary of the plan

The plan is not to remove adaptive logic, and not to insist that gradients are
only meaningful when the same accepted history repeats exactly.

The plan is:

- keep the current adaptive forward solver
- use only the minimum diagnostics needed to confirm the accepted-trajectory
  picture
- move differentiation toward accepted-step / accepted-trajectory backward
  rules
- use Diffrax only as a benchmark reference, not as the default solution
- aim for a NEOPAX-specific gradient path that is both reliable and efficient

## Updated design decision after diagnostics

This section supersedes the exploratory replay-diagnostic direction above.

### What the benchmark established

The current benchmark evidence is already sufficient to change direction:

- the one-step lagged exact-runtime NTX transport map differentiates very well
- the full adaptive rollout does not agree with naive AD-vs-FD checks
- the full-rollout FD reference is unstable across FD step size
- nearby parameter values drift onto different accepted-time / accepted-`dt`
  histories

So the main issue is not local differentiability. The main issue is that the
full adaptive trace is not the right object to differentiate naively.

### Revised implementation target

The target should now be stated more precisely:

- keep the current production forward solve and its adaptive retry/accept logic
- build sensitivities around the accepted transport evolution
- do not make rejected trial steps the primary differentiated object

In other words, the design goal is not "identical accepted history only". The
design goal is:

- an accepted-step / accepted-rollout backward path for the production solve

### Recommended NEOPAX implementation path

The next implementation work should be:

1. Define one accepted transport step as the principal differentiated object.
   This step should include the solver-side carry that is genuinely needed by
   the realized accepted update, rather than just the physical `TransportState`.

2. Add a custom JVP or VJP for that accepted step map.
   The backward rule should be expressed in terms of the accepted step map, not
   in terms of every rejected nonlinear sub-attempt.

3. Compose accepted-step rules into a rollout-level backward path.
   This can use checkpointing or replay over accepted steps, but the replay
   should be based on solver-relevant carry, not just saved physical state.

4. Make controller policy explicit.
   The adaptive controller remains active in the forward solve, but its
   accept/reject branch logic should not be treated as the central smooth
   object in the backward pass.

### Diffrax / Kvaerno5 comparison

Diffrax is a useful reference because it separates four concerns that NEOPAX
should also separate:

- solver step definition
- nonlinear/root-finding method inside the implicit step
- adaptive step-size controller
- adjoint / autodiff strategy

From the Diffrax documentation:

- `RecursiveCheckpointAdjoint` is the default differentiation strategy and is
  described as differentiating the numerical solution directly while using
  online checkpointing to control memory
- `ForwardMode` and `DirectAdjoint` are available when different AD behavior is
  required
- implicit solvers such as `Kvaerno5` take an explicit root finder
- Diffrax provides `VeryChord` plus tolerance plumbing via
  `with_stepsize_controller_tols(...)`
- the abstract solver API has an explicit evolving solver state passed through
  `init(...)` and `step(...)`

This suggests a good comparison point:

- Diffrax does not win by pretending rejected steps do not exist
- Diffrax wins by making the solve, solver state, controller, and adjoint
  policy explicit and solver-aware

That is the useful lesson for NEOPAX.

### What NEOPAX should copy from Diffrax

NEOPAX should copy the separation of concepts:

- accepted step map
- solver carry/state
- nonlinear solve policy
- controller policy
- sensitivity policy

NEOPAX should not blindly copy the exact implementation, because our problem is
more structured:

- transport state has known physics structure
- NTX already exposes solve-boundary derivative contracts
- we can likely save less state than a general-purpose ODE framework
- we can choose transport-specific linearizations instead of generic ones

### Practical next step

The next real engineering task should be:

- identify the smallest solver-carry object needed to define a faithful
  accepted transport step in NEOPAX
- then design the custom accepted-step derivative rule around that object

Only after that should we benchmark against Diffrax/Kvaerno5 to compare:

- gradient quality
- memory
- wall time

The purpose of the Diffrax comparison should be benchmarking and design
calibration, not replacement-by-default.

### Candidate minimal accepted-step carry for current Radau path

Looking at the current custom Radau implementation, the accepted-step object is
not just `(t_n, Y_n, dt_n)`. The current forward step also depends on a small
amount of solver carry that influences:

- predictor quality
- Jacobian reuse
- lagged-response reuse
- adaptive controller evolution

For the current implementation, the smallest *faithful* accepted-step carry
appears to split into two groups.

#### Group A: forward-essential carry

These fields affect the realized accepted step or the next proposed step size,
and so are the current best candidates for the accepted-step state that should
be formalized:

- physical step state:
  - `t`
  - `y`
  - `dt`

- controller memory:
  - `prev_error`
  - `recent_reject_count`
  - `regrowth_cooldown`
  - `easy_growth_streak`

- predictor / Newton warm-start memory:
  - `prev_stages`
  - `prev_dt`
  - `prev_theta_final`
  - `prev_newton_iter_count`

- lagged-response reuse state:
  - `lagged_response_cache`
  - `lagged_response_valid`
  - `lagged_reference_y`

- Jacobian / factorization reuse state:
  - `jacobian`
  - `cache_valid`
  - `cache_dt`
  - `cache_age`
  - `real_lu`
  - `real_piv`
  - `complex_lu`
  - `complex_piv`

This is the current practical definition of the "accepted-step carry" in the
production solver.

#### Group B: diagnostic / reporting outputs

These are useful for analysis and benchmarking, but they should not be treated
as part of the core accepted-step state:

- accepted / failed flags
- fail code
- error norm
- Newton iteration count
- final residual norm
- final delta norm
- `theta_final`
- slow-contraction / blowup / nonfinite diagnostics

They are valuable benchmark outputs, but not primary candidates for the state
that needs to be carried through the differentiated rollout.

### Immediate design implication

The next design step should not be "wrap the whole solver trace in a custom
VJP". It should be:

1. formalize a first-class accepted-step carry object using Group A
2. define the accepted step map on that carry object
3. decide which parts of Group A are truly required by the backward rule, and
   which can remain forward-only acceleration state

That gives us a concrete path to reduce the problem further:

- first define the faithful accepted-step boundary
- then shrink the backward state from there, instead of guessing too early

### First candidate backward payload

A useful first cut is to distinguish between:

- exact accepted-step replay state
- recomputable linearization state
- forward-only controller state

For the current Radau implementation, the first conservative candidate for a
one-step backward payload is:

- `t_n`
- `y_n`
- accepted `dt_n`
- `prev_stages`
- `prev_dt`
- `prev_theta_final`
- `prev_newton_iter_count`
- `lagged_response_cache`
- `lagged_response_valid`
- `lagged_reference_y`
- accepted output state `y_{n+1}`

Rationale:

- this is enough to replay the accepted-step attempt with the same predictor
  and lagged-response context
- it avoids storing controller bookkeeping that mainly affects future step-size
  proposals
- it avoids storing Jacobian/factorization caches that can be recomputed

#### Candidate recompute set

The following fields should be treated as recomputable by default:

- `jacobian`
- `cache_valid`
- `cache_dt`
- `cache_age`
- `real_lu`
- `real_piv`
- `complex_lu`
- `complex_piv`
- stage-level diagnostics
- error norms
- Newton residual summaries

These are useful for speed in the forward solve, but they are poor first
choices for minimal backward storage.

#### Candidate forward-only controller state

The following fields should initially be treated as forward-only, unless a
later rollout-level backward design proves otherwise:

- `prev_error`
- `recent_reject_count`
- `regrowth_cooldown`
- `easy_growth_streak`

These primarily control future timestep proposals and regrowth behavior. They
matter for the production adaptive rollout, but they are not obvious
requirements for the local accepted-step derivative.

### Recommended next implementation move

The next implementation step should be:

1. introduce a candidate accepted-step backward payload object in code
2. populate it from the current accepted-step carry/result boundary
3. keep it unused by the solver for now

That gives us a concrete object to optimize before we commit to a custom
JVP/VJP rule.

### Accepted-step JVP contract

The next AD implementation should target the accepted-step map directly, and
its contract should be written down explicitly before code is added.

#### Proposed primal map

For the current Radau path, define the primal accepted-step map as:

- inputs:
  - solver context
  - physics context
  - accepted-step carry at `n`
  - proposed step size `dt_n`

- outputs:
  - accepted-step attempt result
  - updated accepted-step carry after the attempt

In symbols, the intended boundary is:

- `(solver_ctx, physics_ctx, carry_n, dt_n) -> attempt_result_n`

where `attempt_result_n` contains:

- `y_trial`
- acceptance error estimate
- nonlinear convergence summary
- updated lagged-response reuse state
- updated linearization caches
- accepted-step diagnostics

The important point is that the custom derivative should attach to this map,
not to the full outer adaptive loop.

#### Tangent semantics

The first custom rule should be a **JVP** for this accepted-step map, because:

- the current benchmark uses `jax.jacfwd`
- the one-step diagnostic is already the strongest validation target
- JVP is the most direct first match to the current validation harness

The intended tangent meaning is:

- differentiate the numerical accepted-step update with respect to the
  physically meaningful step inputs
- do not differentiate with respect to the Python identity of solver/physics
  callables
- do not make controller bookkeeping or reuse caches primary tangent objects

#### First tangent-active fields

The first JVP should treat these carry fields as tangent-active:

- `t`
- `y`
- `dt`
- `prev_stages`
- `prev_dt`
- `prev_theta_final`
- `prev_newton_iter_count`
- `lagged_response_cache`
- `lagged_response_valid`
- `lagged_reference_y`

These are the best first candidates because they influence:

- predictor quality
- local implicit solve behavior
- lagged-response reconstruction
- the realized accepted-step state update

#### First forward-only fields

The first JVP should treat these as forward-only by default:

- `prev_error`
- `recent_reject_count`
- `regrowth_cooldown`
- `easy_growth_streak`
- `jacobian`
- `cache_valid`
- `cache_dt`
- `cache_age`
- `real_lu`
- `real_piv`
- `complex_lu`
- `complex_piv`

Rationale:

- controller bookkeeping mostly affects future step proposals, not the local
  accepted-step physics map
- Jacobian/LU caches are acceleration state and can be recomputed

This is a design choice, not an algebraic identity, and must be validated by
the one-step benchmark first.

#### Static context semantics

The following should be passed as explicit **static context**, not as tangent
data:

- solver configuration / kernel context
- physics callable bundle

This does **not** mean the transport physics is excluded from
differentiation. It only means:

- differentiate the numerical map produced by those callables
- do not differentiate with respect to the Python callable objects

In other words:

- `f(t, y, p)` remains differentiable
- the Python object that implements `f` is treated as static

#### First validation criterion

The first custom JVP implementation should be considered acceptable only if:

1. the one-step benchmark still matches FD at the current excellent level
2. the forward production solve remains unchanged
3. the new JVP boundary compiles reliably under the existing `jax.jacfwd`
   benchmark path

Only after that should rollout-level behavior be reassessed.

#### Implementation consequence

The next code change should therefore be:

1. keep the current accepted-step primal boundary
2. attach a top-level custom JVP to that boundary
3. use static solver/physics context
4. implement tangent flow only for the first tangent-active carry fields

That is the cleanest next experimental AD step for NEOPAX.

### Current custom-JVP status

The first two custom-JVP attempts at the accepted-step boundary both failed
under the existing `jax.jacfwd` benchmark path, and those failures should now
be treated as concrete design information rather than incidental bugs.

#### Failure 1: closure-local custom JVP

The first attempt attached `custom_jvp` while the accepted-step implementation
still depended on solver-local closures. Under the one-step benchmark this
failed with:

- `TypeError: No constant handler for type: <class 'jax._src.interpreters.ad.JVPTracer'>`

Interpretation:

- the AD boundary was still too deep inside closure-captured traced state
- this was not a valid JAX custom-derivative boundary

#### Failure 2: module-scope wrapper around accepted-step attempt

After hoisting the accepted-step primal logic and making solver/physics context
explicit, a second module-scope `custom_jvp` attempt was tested. The one-step
benchmark still failed, now with:

- `TypeError: No constant handler for type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>`

This happened even after:

- hoisting the accepted-step primal function to module scope
- making solver context explicit
- making physics callable context explicit
- removing the solver-local function-handle argument from the custom-JVP
  boundary

Interpretation:

- the accepted-step attempt wrapper is still not a sufficiently clean custom
  JVP boundary for the current `jacfwd` + `jit` + loop structure
- the remaining issue is not just "too many hidden locals"
- the current benchmark path is still forcing JAX to treat some traced value as
  a static constant at the custom-JVP boundary

#### Immediate conclusion

At present, the accepted-step custom-JVP path should be considered:

- structurally informative
- but not yet a valid production AD implementation

The important result is not just that the code failed. The important result is:

- simply hoisting the primal step to module scope is **not sufficient by
  itself** to make the current accepted-step custom-JVP valid under the
  existing benchmark execution path

This should guide the next design step. The next iteration should begin from
the recorded failure mode instead of retrying the same boundary with minor
variations.

### SPECTRAX-GK comparison

The local `SPECTRAX-GK` codebase suggests a more useful analogy than Diffrax
for the current NEOPAX Radau work.

What SPECTRAX-GK appears to do:

- it keeps most time integration as ordinary JAX code (`jit`, `scan`,
  checkpointing)
- it does **not** appear to expose one big public "custom backward rule for the
  whole solver" API for its native integrators
- instead, it uses a `custom_vjp` on a mathematically meaningful **subsolve**
  inside the RHS assembly:
  - `spectraxgk/terms/fields.py: solve_fields`

So the closest lesson is:

- separate forward and backward rules at the level of the important solver
  subproblem
- not necessarily at the level of the entire time-integration loop

For NEOPAX, the analogous subproblem is not the whole adaptive Radau loop.
It is the **accepted implicit Radau stage solve**.

This strengthens the current plan:

- keep the adaptive production loop as ordinary forward solver logic
- isolate the converged implicit Radau stage solve as its own mathematical
  object
- attach the custom derivative rule there

This is more aligned with the SPECTRAX-GK pattern than trying to insert
`custom_jvp` directly into the full adaptive loop machinery.

### Radau-native implicit differentiation plan

The next iteration should be based on the **converged accepted Radau step as
an implicit nonlinear solve**, not on:

- tracing raw Newton iterations
- tracing rejected-step controller history
- fixed accepted-step replay as a final solution
- more `custom_jvp` boundary experiments around the current loop structure

#### Why this is the right Radau-native object

For one accepted Radau step, the solver computes stage values `Z` by solving a
nonlinear collocation system. In the current implementation this appears in
`_radau_single_step_primal(...)` as:

- stage predictor: `z0`
- stage residual map: `residual(z_flat)`
- Newton loop over `z`
- converged stage stack: `stages_final = z_final.reshape((num_stages, state_dim))`
- accepted state update:
  - `flat_next = flat_y + h_value * (b @ stages_final)`

So the mathematically meaningful accepted-step map is:

- input:
  - current state `y_n`
  - step size `h`
  - solver/physics context
  - selected carry fields that actually affect the implicit equations
- implicit stage solve:
  - find `Z` such that `R(Z; y_n, h, p) = 0`
- output:
  - `y_{n+1} = y_n + h * (b^T Z)`

This is the object to differentiate. The relevant derivative is therefore the
derivative of the **solution of the stage equations**, not the derivative of
the particular Newton history used to reach that solution.

#### Accepted-step implicit system in the current code

The current stage residual is implemented as:

- `stages = z_flat.reshape((s, n))`
- `stage_states = y_n + h * (A @ stages)`
- `evals = f(t_n + c_i h, stage_states_i, ...)`
- `R(Z) = Z - F(stage_states)`

in flattened form:

- `residual(z_flat) = (stages - evals).reshape((-1,))`

where:

- `s = num_stages`
- `n = state_dim`
- `Z in R^{s x n}`

The accepted step then uses:

- `y_{n+1} = y_n + h * (b^T Z)`

This means the local tangent equation should come from:

- `dR = R_Z dZ + R_y dy_n + R_h dh + R_p dp = 0`

and therefore:

- `dZ = - R_Z^{-1} (R_y dy_n + R_h dh + R_p dp)`

followed by:

- `dy_{n+1} = dy_n + d[h * (b^T Z)]`

This is the core Radau-native JVP object.

#### What the current solver already computes

The current implementation already builds the linearization used by the stage
Newton solver:

- `jacobian_ref = jacfwd(_rhs_eval_at_current_time)(flat_y)`
- transformed real block solve:
  - `real_matrix = I - lambda_real * h * jacobian_ref`
  - `real_lu_out`, `real_piv_out`
- transformed complex block solves:
  - `complex_lu_out`, `complex_piv_out`
- transformed stage linear solver:
  - `stage_solver(rhs)`

This means we already have:

- a stage-space linear solve mechanism
- LU factorizations for the Radau eigenbasis blocks
- the converged stage vector `z_final`
- the accepted output state `flat_next`

So the new path should reuse this solver mathematics rather than trying to
differentiate through the explicit Newton loop.

#### Important subtlety

The current `stage_solver(...)` is the Newton linear solve built from
`jacobian_ref = df/dy |_(t_n, y_n)` (or the lagged linear-response
approximation), not yet the full exact Jacobian of the nonlinear collocation
residual `R_Z` evaluated at the converged `Z`.

So there are two Radau-native implementation levels:

1. **Approximate implicit-diff JVP**
- reuse the existing Newton linearization directly
- cheapest path to a first Radau-native derivative rule
- may already be much better than tracing Newton iterations

2. **Full implicit-diff JVP**
- build/apply the exact linearized collocation Jacobian `R_Z` at the converged
  accepted step
- solve the tangent system with that operator
- this is the most principled final target

The recommended order is:

1. write the accepted-step implicit system explicitly
2. implement the approximate implicit-diff JVP first
3. validate one-step parity and full-rollout behavior
4. then decide whether the exact `R_Z` linearization is necessary

#### What should be reverted or ignored for this plan

The following are not part of the long-term Radau-native AD solution:

- fixed-step accepted-rollout replay as a final gradient method
- `custom_jvp` boundary hacks around the current step loop
- stop-gradient masking as a primary AD fix
- differentiating through raw Newton iterate history

The useful structural refactor that should remain is:

- accepted-step carry/result/context dataclasses
- module-scope helper math and accepted-step primal decomposition
- clearer separation between accepted-step primal state and controller logic

#### Immediate next coding step

The next code change should be small and solver-mathematical:

1. extract/document the accepted-step residual map `R(Z; y_n, h, p)`
2. expose a module-scope helper that evaluates this residual at arbitrary
   `z_flat`
3. expose a module-scope helper that applies the current Newton
   linearization-based inverse to a stage-space right-hand side
4. use those two helpers to implement the **first implicit-diff accepted-step
   JVP**

That is the first step in the new plan that is genuinely on the Radau path and
changes the derivative object for the right reason.

#### Current implementation progress

The following forward-neutral helpers now exist in `_transport_solvers.py`:

- `_radau_stage_residual(...)`
- `_radau_apply_stage_linear_solve(...)`
- `_radau_approximate_accepted_step_tangent(...)`

The new tangent helper currently uses the approximate linearized stage system:

- `(I - h A ⊗ J_ref) dZ = J_ref (dy + dh * A Z)`

followed by:

- `dy_next = dy + dh * (b^T Z) + h * (b^T dZ)`

This is **not yet** the full exact collocation implicit derivative, but it is
the first solver-mathematical tangent object that is:

- Radau-native
- based on the converged accepted step
- independent of raw Newton iteration tracing

The current code now also contains a pure helper that returns:

- the primal accepted-step attempt result
- plus the approximate accepted-step tangent result in the same output shape

via:

- `_execute_radau_accepted_step_attempt_with_approx_tangent(...)`

That helper should be treated as the code-level target contract for the first
accepted-step JVP implementation.

In addition, the actual Radau attempt/control wrapper has now been pulled much
closer to module scope:

- `_radau_attempt_step_lean(...)`
- `_radau_step_fn(...)`
- `_RadauSolveExecutionContext`

So the next AD boundary should target these module-scope step objects rather
than the older deeply nested `solve`-local wrappers.

#### What the latest failures clarified

The repeated `custom_jvp` failures are now informative enough to count as a
design result.

What we tried:

- attach `custom_jvp` around accepted-step attempt helpers
- move those helpers to module scope
- move the step/control wrapper to module scope
- remove the old `solve`-local `step_fn` shim

What still happened under `jax.jacfwd(...)`:

- `DynamicJaxprTracer` constant-handler failures inside the compiled adaptive
  step path

So the current conclusion is:

- the forward-neutral refactor was still worth doing
- but the live custom derivative rule should **not** be attached from inside
  the current `solve -> jit(step_fn) -> lax.cond(...)` machinery

This does **not** invalidate the Radau-native implicit-diff plan.
It narrows it:

- the right mathematical object is still the converged implicit Radau stage
  solve
- but the future AD hook must sit at the **stage subsolve boundary**, not at
  the surrounding adaptive loop wrapper

#### Updated immediate implementation target

Following the SPECTRAX-GK analogy more closely, the next concrete code target
should be a first-class Radau stage subsolve object:

- primal:
  - run the Newton solve for the collocation stage system
- derivative:
  - use the implicit linearized stage solve

This is now the clearest NEOPAX-native analogue to:

- `solve_fields` in `SPECTRAX-GK`

rather than trying to make the whole adaptive loop itself the first custom-AD
boundary.

The current code now exposes this direction more explicitly by isolating the
Newton stage solve into a standalone helper:

- `_radau_run_stage_subsolve(...)`

That helper is still forward-only for now, but it is a much better future AD
attachment point than the full adaptive step wrapper.

The next useful structural refinement is now also in place:

- `_RadauStageSubsolveInputs`
- `_radau_build_stage_subsolve_inputs(...)`
- `_radau_stage_subsolve_residual(...)`
- `_radau_stage_subsolve_linear_solve(...)`
- `_radau_run_stage_subsolve_from_inputs(...)`

This matters because the future derivative rule should attach to a **primitive
with explicit Radau data**, not to a large wrapper whose important solver
inputs are still implicit in local variables.

So the immediate AD target has now been narrowed further:

- not the adaptive loop
- not the whole accepted-step wrapper
- but the explicit stage-subsolve primitive defined by:
  - a bundled subsolve input object
  - a residual map
  - a stage linear solve

The next solver-mathematical refinement is also now explicit in code:

- `_RadauStageSubsolveTangentInputs`
- `_RadauStageSubsolveApproximateTangentResult`
- `_radau_extract_stage_subsolve_tangent_inputs(...)`
- `_radau_compute_stage_subsolve_approximate_tangent(...)`
- `_radau_run_stage_subsolve_with_approx_tangent(...)`

This means the future AD hook no longer needs to invent the stage-subsolve
contract on the fly. We now have a dedicated primitive-level pairing of:

- primal:
  - explicit stage-subsolve inputs
  - explicit stage-subsolve result

- tangent:
  - explicit tangent-active subsolve inputs
  - explicit approximate tangent result

That is a much better starting point for a Radau-native custom derivative than
the earlier attempts to attach `custom_jvp` around the adaptive loop wrappers.

The stage-subsolve path has now also been cleaned up one more step:

- `_radau_run_stage_subsolve(...)` now works directly from:
  - `kernel_context`
  - `physics_context`
  - `_RadauStageSubsolveInputs`

So the current primitive is no longer "explicit inputs plus lambda adapters".
It is now an explicit module-scope stage-subsolve path end to end. That is the
right shape for the next real AD experiment.

We also tested the next live custom-AD experiment exactly there:

- `_radau_run_stage_subsolve_autodiff(...)`

with its first tangent lift via:

- `_radau_build_stage_subsolve_tangent_result(...)`

That experiment still failed under the real benchmark execution path with the
same `DynamicJaxprTracer` constant-handler failure.

So the updated conclusion is:

- even the explicit stage-subsolve primitive is **not** currently a legal live
  `custom_jvp` attachment when invoked from inside the present
  `solve -> jit(step_fn) -> lax.cond(...)` machinery

This is an important negative result. It means:

- the stage-subsolve remains the right mathematical object
- but the custom derivative still cannot be attached from *inside* the current
  compiled adaptive solve path

So the next Radau-native experiment should be split in two:

1. **standalone AD-facing subsolve validation**
- attach the custom derivative to the explicit stage-subsolve primitive
- validate it outside the production adaptive loop

2. **production-path reintegration**
- only after the standalone subsolve derivative is validated
- decide how to reintroduce it into the larger Radau solve architecture

The current code now includes that standalone validation entrypoint:

- `_radau_run_stage_subsolve_standalone_autodiff(...)`

This is intentionally different from the earlier failed attempts because it is
not invoked from inside the present `solve -> jit(step_fn) -> lax.cond(...)`
machinery.

### Updated composition strategy

The current evidence strongly suggests we should **not** replace the existing
local one-step AD path:

- the production one-step diagnostic is already excellent
- the first standalone custom stage-subsolve tangent is currently less accurate

So the next layer to redesign is not the local one-step derivative itself.
It is the **multi-step composition boundary**.

The corresponding code direction is now explicit:

- keep the existing accepted-step primal logic
- expose accepted-step composition as its own object
- leave the production adaptive loop untouched

The current code now contains that accepted-step composition boundary as:

- `_RadauAcceptedStepMapResult`
- `_radau_apply_accepted_step_map(...)`

This is the right architectural seam for future AD-facing multi-step
composition experiments:

- preserve the good local one-step AD
- redesign only the broken rollout/composition layer
