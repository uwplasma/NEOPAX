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

### Main conclusion now

The controller investigation is now the main result.

What we tested:

- accepted-step composition without controller `dt` updates
- accepted-step composition with the real Radau controller `dt` updates

What we found:

- accepted-step composition alone stays excellent out to at least `40` steps
- once the real controller updates `dt`, AD-vs-FD begins to drift
- the first mismatch appears in `next_dt`, before any accepted/rejected path
  split
- later, the changed `next_dt` values become changed attempted `dt` values
- only after that does the accepted-mask / path itself diverge

So the first real AD failure is **controller-driven timestep drift**, not the
implicit Radau step and not the accepted-step composition map by itself.

The controller-only diagnostic showed the progression clearly:

- `2` steps:
  - accepted mask equal
  - attempted `dt` equal
  - next `dt` equal
- `3` steps:
  - accepted mask equal
  - attempted `dt` equal
  - next `dt` already slightly different
- `5` to `20` steps:
  - accepted mask still equal
  - attempted `dt` and next `dt` now different
  - AD-vs-FD error grows materially
- `40` steps:
  - accepted mask finally diverges too
  - rollout AD-vs-FD can blow up completely

This means:

- rejection/path bifurcation is a **later consequence**
- the controller update is the **first broken layer**

So the main Radau-native solution effort should now target the adaptive
controller handling, not the implicit step derivative:

- either treat controller `dt` evolution as forward-only / nondifferentiated
- or define a custom AD rule for the adaptive rollout that prevents controller
  drift from dominating the derivative

This is now the main approach and should be treated as the working diagnosis
for full-solve AD failure in the current Radau path.

### First concrete solution idea

The first concrete solution idea is:

- keep the current differentiated accepted-step map
- keep the real Radau controller for forward execution
- but treat the controller-driven `dt` update as **forward-only**

In other words:

1. run the accepted Radau step
2. compute the next `dt` with the real controller
3. use that `dt` for the next forward step
4. block AD through the controller state update and `dt` evolution

This means the differentiated object becomes:

- the rollout of accepted Radau steps
- **conditioned on the realized forward timestep schedule**

not:

- the full derivative of the adaptive controller itself

Important interpretation:

- the forward solve still decides the actual times / `dt`s
- AD then differentiates the state evolution along that realized schedule
- so the derivative answers:
  - "how does the solution change along this forward-generated adaptive
    schedule?"

rather than:

  - "how does the solution change including sensitivity of the controller's
    timestep choices?"

This is a deliberate approximation, but it is motivated by the evidence:

- local step AD is good
- accepted-step composition AD is good

## Realized-schedule replay investigation update

This section records the current status of the solve-level AD work around the
realized adaptive Radau replay.

### What was established before the latest replay fix attempt

In the realized-schedule rollout check:

- the first nonfinite tangent consistently appeared at accepted attempt
  `index=144`
- that step had benign primal diagnostics:
  - accepted step
  - `dt ~= 9.304070e-06`
  - moderate `err_norm`
  - small `theta_final`
  - small Newton iteration count

Several localized A/B tests were run by zeroing tangent channels in the replay
carry:

- `lagged_response_cache`
- `prev_stages`
- `lagged_response_cache + prev_stages`
- `y`
- `prev_error`
- `y + prev_error`

None of those moved the first bad step.

The decisive diagnosis was:

- even with **all incoming replay tangents zeroed**, the replay still produced
  a first nonfinite tangent at accepted attempt `144`

That means the failure was not just a large physical sensitivity or simple
accumulation effect. It was structural: the replay JVP machinery itself was
manufacturing bad tangents.

### Structural replay fix attempt

A real fix attempt was then made:

- the realized-schedule replay was changed so that accepted attempts no longer
  use raw `jax.jvp(...)` through the full primal accepted-step attempt
- instead, the replay now calls an AD-facing accepted-step wrapper with a
  `custom_jvp`
- that custom JVP uses the existing Radau-native approximate implicit-diff
  tangent, rather than differentiating through the raw Jacobian/LU/Newton
  internals

This was not just another diagnostic print. It changed the actual AD path used
by the replay.

### What changed after that fix attempt

After switching the replay to the accepted-step custom JVP:

- the replay still first failed at accepted attempt `144`
- but several previously contaminated tangent channels were cleaned up:
  - `prev_error_dot_abs` became `0`
  - `jacobian_dot_max_abs` became `0`
  - `real_lu_dot_max_abs` became `0`
  - `complex_lu_dot_max_abs` became `0`

So the structural replay fix partially worked:

- it removed the bad AD path through raw Jacobian/LU/Newton internals
- but it did **not** fully remove the `NaN`

### Current remaining failure mode

With the accepted-step custom JVP active, the first nonfinite quantities at
accepted attempt `144` are now:

- `y_dot_max_abs -> NaN`
- `prev_stages_dot_max_abs -> NaN`

while:

- `prev_error_dot_abs = 0`
- Jacobian/LU tangent summaries remain `0`

This strongly suggests the remaining bug is now inside the **approximate
accepted-step tangent formula itself**, especially the part that builds:

- the accepted state tangent `dy_next`
- the stage-history tangent `dz_stages`

not in the raw replay-through-primal path anymore.

### Practical conclusion

The current state should be interpreted as:

- the original realized-schedule replay JVP was structurally wrong
- replacing it with the accepted-step custom JVP was a meaningful fix attempt
  and improved the AD path
- the remaining `NaN` is now localized much more narrowly to the current
  approximate accepted-step tangent implementation

So the next debugging/fix work should target:

- `_radau_approximate_accepted_step_tangent(...)`
- the stage linear solve used inside that tangent construction
- any hidden dependence of `dz_stages` / `dy_next` on replay state that should
  remain forward-only

### Current solve plan

At this point, the problem should be treated as an **accepted-step tangent
construction bug**, not as a generic replay-AD or controller-history problem.

The practical plan is:

1. debug `_radau_approximate_accepted_step_tangent(...)` directly
   - especially on the known problematic accepted attempt `144`

2. inspect the tangent pipeline in order:
   - stage RHS assembly
   - transformed stage RHS
   - real block linear solve
   - complex block linear solve
   - reconstructed `dz_stages`
   - final `dy_next`

3. enforce the basic correctness property:
   - zero tangent input must give zero tangent output

4. keep the first stable accepted-step custom JVP intentionally narrow:
   - differentiate primarily with respect to accepted-step state `y_n`
   - and trial step size `h`
   - keep replay/controller/cache internals forward-only until the core tangent
     path is stable

The current leading hypothesis is:

- the remaining `NaN` is produced inside the current approximate implicit-diff
  accepted-step tangent formula itself
- most likely in the `dZ` stage solve or in the subsequent reconstruction of
  `dy_next`
- controller `dt` sensitivity is the first thing that destabilizes the full
  derivative

So this should be treated as the **first implementation path** for obtaining a
useful full-solve AD signal in the Radau solver.

### Follow-up after zero-input safeguards

The accepted-step approximate tangent was then tightened in two conservative
ways:

- `_radau_apply_stage_linear_solve(...)` now short-circuits an exactly zero RHS
  to a zero solution instead of sending it through the LU solve
- `_radau_approximate_accepted_step_tangent(...)` now explicitly enforces
  zero-in -> zero-out for the narrow accepted-step tangent inputs
  (`dy_source == 0` and `dh_source == 0`)

These were real implementation changes aimed at the remaining tangent
construction bug, not just more prints.

### What the new run showed

After those safeguards, the realized-schedule replay still failed at the same
accepted attempt:

- `first_bad_index = 144`
- `first_bad_was_accepted = True`
- `first_bad_dt ~= 9.304070e-06`

But the local window became more informative:

- at attempts `142` and `143`
  - `prev_error_dot_abs = 0`
  - `jacobian_dot_max_abs = 0`
  - `real_lu_dot_max_abs = 0`
  - `complex_lu_dot_max_abs = 0`
  - `y_dot_max_abs` and `prev_stages_dot_max_abs` were still finite
- at accepted attempt `144`
  - `y_dot_max_abs -> NaN`
  - `prev_stages_dot_max_abs -> NaN`
  - while Jacobian/LU tangent summaries stayed `0`

So the remaining failure is now even more clearly concentrated in the
accepted-step tangent outputs themselves, especially:

- accepted-state tangent assembly `dy_next`
- stage-history tangent assembly `dz_stages`

not in the raw Newton/LU differentiation path.

### New local comparison hook

To answer the key question "is the custom one-step tangent itself already
wrong?", a new very local diagnostic hook was added.

The benchmark NaN-debug path now also:

1. replays the primal carry forward to the first bad attempt
2. isolates that exact accepted-step input
3. compares, on that single step with **zero tangent input**:
   - the custom accepted-step JVP
   - a direct one-step `jax.jvp` of the raw accepted-step primal

This comparison is reported as:

- `one-step zero-tangent compare: ...`

and records:

- target attempt index
- whether that target was accepted
- trial `dt`
- custom one-step tangent finiteness and max norms
- direct one-step tangent finiteness and max norms

### Why this comparison matters

This is the most direct test of the current hypothesis:

- if the **custom one-step zero-tangent** output is already nonzero or
  nonfinite while the **direct one-step zero-tangent** output is clean,
  then the bug is inside the custom accepted-step tangent itself
- if both are clean, then the remaining bug lies in how the one-step tangent is
  composed or propagated through the realized-schedule replay

This check was motivated by the earlier important observation that:

- isolated one-step AD had behaved much better than solver-level replay AD

So the current next-step diagnosis is no longer broad tangent-channel A/B
testing. It is:

- compare custom vs direct one-step tangent on the exact failing accepted step
- then inspect whichever side first violates zero-in -> zero-out

### Latest update: local derivatives are good, long-composition parity was the bug

The subsequent focused tests changed the diagnosis materially.

#### 1. One-step diagnostic passed very strongly

Running the cheap one-step diagnostic for parameter `n0` showed excellent
AD-vs-FD agreement for all tracked objectives.

Representative result:

- max relative error was on the order of `1e-6`

Interpretation:

- the local accepted Radau step derivative is good
- the accepted-step custom/solver-local math is not the main remaining bug

So the previously suspected "one-step tangent is wrong" hypothesis is no
longer the leading one.

#### 2. Short accepted-step composition also passed

The short accepted-step composition check with:

- `step_count = 2`
- `step_count = 3`

also passed with similarly tiny relative errors.

Interpretation:

- one accepted step is good
- short composition of accepted steps is also good

So the failure is not caused by immediate local composition of the accepted-step
map.

#### 3. The real bug was in long-horizon replay/carry parity

Inspecting the production adaptive step path against the replay/debug
composition path revealed an important implementation mismatch.

Production path:

- `_radau_attempt_step_lean(...)`
- uses `attempt_result.carry_after_attempt`
- then applies the timestep controller on top of that full post-attempt carry

Replay/debug long-composition paths:

- `_radau_replay_realized_attempt_rollout(...)`
- `_radau_debug_realized_attempt_replay(...)`

were manually reconstructing `next_carry` field-by-field instead of starting
from `attempt_result.carry_after_attempt`.

That is dangerous because it can silently drop or desynchronize post-attempt
state that only matters after many attempts, especially:

- `lagged_response_cache`
- `lagged_reference_y`
- lagged-response validity / reuse-related carry state

This matches the observed symptom pattern exactly:

- one-step good
- 2-step / 3-step composition good
- long realized schedule fails later (around attempt `144`)

#### 4. Replay/carry parity fix

The replay/composition path was patched so that:

- accepted attempts now start from `attempt_result.carry_after_attempt`
- rejected attempts explicitly carry forward the lagged-response cache and
  lagged-reference state from the post-attempt result in forward-only form

This brings the replay/debug long-composition path much closer to the actual
production forward step logic.

#### 5. Fast realized-schedule AD debug is now finite

After that parity fix, the fast realized-schedule AD debug run became fully
finite for `n0`.

Observed result:

- baseline realized schedule:
  - `attempt_count = 184`
  - `accepted_count = 115`
  - `completed = True`
  - `failed = False`
- all reported AD objective derivatives were finite
- NaN localization did not trigger at all

Example finite AD values from the fast check:

- `softmax_Er = -4.840568e+00`
- `smooth_root_proxy = -7.633311e-05`
- `Er2_volume_average = 6.354004e+00`
- `Er_volume_average = -1.389743e+00`
- `electron_temperature_volume_average_keV = 1.204892e-02`
- `total_pressure_volume_average = 7.942981e+00`
- `alpha_power_volume_average_mw_m3 = 2.703767e-01`

Interpretation:

- the old long-horizon NaN was not an unavoidable property of the accepted-step
  tangent
- it was strongly tied to replay/composition carry mismatch

#### Updated leading conclusion

The current best interpretation is now:

- one-step derivative is good
- short accepted-step composition is good
- the major long-horizon failure was a replay/composition carry-parity bug
- the carry-parity fix removed the nonfinite behavior in the fast
  realized-schedule AD path

So the work is no longer "find the first NaN".

The next job is:

- validate the now-finite long-horizon AD path against FD
- confirm that the full realized-schedule rollout check gives acceptable
  AD-vs-FD agreement

#### Updated practical next step

The next validation run should be:

```bash
python examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py --parameter n0 --realized-schedule-rollout-check
```

If this passes, then the core Radau AD path is no longer blocked by the
long-horizon NaN issue.

---

## 2026-05-22: Post-NaN status update

### 1. Full adaptive FD is not a fair reference

After the long-horizon NaN was fixed, the full realized-schedule rollout check
was rerun:

```bash
python examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py --parameter n0 --realized-schedule-rollout-check
```

The result was:

- AD stayed finite
- but `fd_minus` and `fd_plus` followed very different adaptive paths

Observed path counts:

- baseline:
  - `attempt_count = 184`
  - `accepted_count = 115`
- `fd_minus`:
  - `attempt_count = 137`
  - `accepted_count = 87`
- `fd_plus`:
  - `attempt_count = 179`
  - `accepted_count = 113`

Interpretation:

- this is no longer a NaN problem in the AD path
- the remaining AD-vs-FD mismatch is largely because the central FD reference
  is comparing different discrete adaptive histories

So the next benchmark target became:

- compare AD against FD on a **frozen baseline realized schedule**

### 2. Added reusable frozen-schedule helpers

To support fairer FD checks and future tooling, reusable Radau replay helpers
were added.

Solver-level helpers in `_transport_solvers.py`:

- `_radau_dt_sequence_from_time_list(...)`
- `_radau_run_prepared_on_time_list(...)`
- `_radau_run_prepared_on_realized_trace(...)`

These provide:

- replay on a caller-provided absolute time list
- replay on a frozen realized adaptive trace

Orchestrator-level helper in `_orchestrator.py`:

- `run_transport_on_time_list(...)`

Public export in `__init__.py`:

- `NEOPAX.run_transport_on_time_list(...)`

Important scope note:

- `run_transport_on_time_list(...)` currently supports only the custom
  `RADAUSolver`

### 3. Added a benchmark mode for frozen-FD comparison

The benchmark script now has:

```bash
--realized-schedule-frozen-fd-check
--realized-schedule-frozen-replay-mode attempt|accepted
--realized-schedule-frozen-accepted-steps N
```

Goal:

- run one adaptive baseline rollout
- compute AD
- compute `fd_minus` / `fd_plus` by replaying the perturbed states on the
  frozen baseline schedule

This avoids paying for two extra adaptive FD solves and avoids comparing
against unrelated adaptive paths.

### 4. Important correction: frozen-FD failure is a frozen primal replay issue, not AD

Running the first full frozen-FD attempt showed:

- AD stayed finite
- `fd_minus` / `fd_plus` frozen replay became nonfinite
- therefore the printed frozen FD gradient became `nan`

That means:

- the old AD NaN bug appears fixed
- the new problem is that the **forced frozen primal replay** can itself become
  nonfinite when driven too far away from the baseline schedule

So the next debugging target changed again:

- not "why is AD nonfinite?"
- but "how far can the frozen primal replay be pushed before it becomes
  nonfinite?"

### 5. Prefix frozen replay mode was added

To localize that new failure efficiently, the frozen-FD benchmark now supports
replaying only the first `N` accepted steps from the baseline realized
schedule.

Intended progression:

```bash
python examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py --parameter n0 --realized-schedule-frozen-fd-check --realized-schedule-frozen-replay-mode attempt --realized-schedule-frozen-accepted-steps 1
python examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py --parameter n0 --realized-schedule-frozen-fd-check --realized-schedule-frozen-replay-mode attempt --realized-schedule-frozen-accepted-steps 2
python examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py --parameter n0 --realized-schedule-frozen-fd-check --realized-schedule-frozen-replay-mode attempt --realized-schedule-frozen-accepted-steps 3
```

The benchmark also prints:

- `state_finite`
- `objectives_finite`
- `all_finite`

for both:

- `frozen_fd_minus`
- `frozen_fd_plus`

### 6. One-step frozen prefix result: finite replay, but initial comparison logic was wrong

The first prefix test with `accepted_steps = 1` showed:

- both frozen replays were finite
- but the objective errors looked enormous

That turned out to be a benchmark bug:

- AD was still being computed for the full realized-schedule objective
- while FD was being computed for the truncated 1-step frozen replay objective

So those errors were not meaningful.

This was patched so that:

- when `--realized-schedule-frozen-accepted-steps N` is used
- AD and FD are both computed for the **same truncated frozen objective**

The benchmark now prints:

- `ad_mode = realized_schedule_jvp`
  when using the full realized-schedule AD boundary
- `ad_mode = frozen_trace_direct`
  when using a truncated frozen prefix comparison

### Current best next step

The correct immediate rerun is now:

```bash
python examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py --parameter n0 --realized-schedule-frozen-fd-check --realized-schedule-frozen-replay-mode attempt --realized-schedule-frozen-accepted-steps 1
```

Then advance to:

- `accepted_steps = 2`
- `accepted_steps = 3`

This should tell us whether the frozen replay comparison itself is sound
step-by-step, and where the frozen primal replay first becomes unreliable if
it still fails at larger prefixes.

---

## Next Session Checkpoint

### What is currently true

- the long-horizon AD NaN on the custom realized-schedule path appears fixed
- the main remaining problem is **not** the adaptive AD itself
- the current failure is that the **frozen primal replay used for FD**
  becomes nonfinite on the full baseline realized schedule

Most recent full frozen-FD run:

```bash
python examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py --parameter n0 --realized-schedule-frozen-fd-check --realized-schedule-frozen-replay-mode attempt
```

Observed result:

- AD remained finite
- `frozen_fd_minus`:
  - `state_finite = False`
  - `objectives_finite = False`
  - `all_finite = False`
- `frozen_fd_plus`:
  - `state_finite = False`
  - `objectives_finite = False`
  - `all_finite = False`

Interpretation:

- the benchmark target is still correct:
  - adaptive AD on the custom realized-schedule path
  - FD computed from `fd_minus` / `fd_plus` on the **same frozen baseline path**
- but the frozen primal replay itself is not yet robust enough over the full
  schedule for those perturbed inputs

### Important benchmark status

The script currently distinguishes two different uses of the frozen-path mode:

1. **Main benchmark**

- no prefix truncation
- compares:
  - adaptive AD via the custom realized-schedule JVP
  - against frozen-path FD

2. **Prefix diagnostic**

- uses `--realized-schedule-frozen-accepted-steps N`
- this is now a **forward-only frozen replay stability diagnostic**
- it does **not** try to push AD through the raw frozen replay helper

Reason:

- trying to run `jax.jacfwd` through the raw truncated frozen replay path hit a
  tracer error:
  - `TypeError: No constant handler for type: DynamicJaxprTracer`
- that raw frozen replay helper is not intended to be the production AD
  boundary anyway

So:

- AD should stay on the custom realized-schedule path
- frozen replay is only for primal FD reference and replay-stability diagnosis

### Exact next things to run

The next session should localize where the frozen primal replay first stops
being finite.

Run in this order:

```bash
python examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py --parameter n0 --realized-schedule-frozen-fd-check --realized-schedule-frozen-replay-mode attempt --realized-schedule-frozen-accepted-steps 1
python examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py --parameter n0 --realized-schedule-frozen-fd-check --realized-schedule-frozen-replay-mode attempt --realized-schedule-frozen-accepted-steps 2
python examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py --parameter n0 --realized-schedule-frozen-fd-check --realized-schedule-frozen-replay-mode attempt --realized-schedule-frozen-accepted-steps 3
```

Then, if still finite:

- `accepted_steps = 5`
- `accepted_steps = 10`

### Additional discriminator

If `attempt` mode fails early, compare against:

```bash
python examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py --parameter n0 --realized-schedule-frozen-fd-check --realized-schedule-frozen-replay-mode accepted --realized-schedule-frozen-accepted-steps 1
```

Then similarly for `2`, `3`, etc.

Why:

- if `accepted` mode stays finite while `attempt` mode fails
- then the destabilizing piece is likely in the rejected-attempt replay path,
  not the accepted-step map itself

### What to focus on next

Do **not** restart broad NaN debugging of the adaptive AD path.

The focus is now:

- frozen-path primal replay stability
- especially the difference between:
  - full-attempt replay
  - accepted-step-only replay

That is the cleanest way to isolate the remaining mismatch between:

- adaptive AD on the custom realized-schedule boundary
- and a valid frozen-path FD reference

## 2026-05-22: safe-path AD/FD and adaptive-vs-direct AD status

We now have a much cleaner comparison path than the earlier full adaptive FD or
full frozen-path FD tests.

### What is established now

- The long-horizon adaptive AD NaN issue is no longer the main blocker for this
  benchmark path.
- For the current benchmark helper, the initial ambipolar `Er` solve is
  effectively frozen out of the differentiated parameter path:
  - baseline `Er_init`, `fd_minus Er_init`, `fd_plus Er_init`, and AD all start
    from the same baseline-initialized `Er`
  - only density/pressure are re-parameterized with `n0`
- The new safe baseline-`dt` path test uses the earliest known frozen-FD
  failure (`fd_plus` first bad attempt `76`) and stops two attempts earlier by
  default (`safe_attempt_index = 74`).

### Important benchmark results

#### 1. Safe baseline-`dt` FD vs adaptive AD

Command:

```bash
python examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py --parameter n0 --baseline-dt-path-safe-fd-check
```

Result:

- baseline / `fd_minus` / `fd_plus` fixed-path replays are all finite
- so this is a valid same-path AD-vs-FD comparison at the safe final time
- pressure-like objectives match well
- `Er`-sensitive objectives do not

Representative output at `safe_final_time = 1.872051e-03`:

- `total_pressure_volume_average`: very small relative error
- `alpha_power_volume_average_mw_m3`: very small relative error
- `electron_temperature_volume_average_keV`: modest relative error
- `Er_volume_average`: large relative error
- `Er2_volume_average`: large relative error

#### 2. Safe baseline-`dt` direct AD vs adaptive AD

Command:

```bash
python examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py --parameter n0 --baseline-dt-path-safe-compose-check
```

Result:

- `fixed_dt_direct_ad` closely matches the fixed-path FD
- `adaptive_ad` shows the same mismatch pattern as in the AD-vs-FD safe test

Conclusion:

- the remaining mismatch is **not mainly an FD artifact**
- the problem is in the **custom adaptive realized-schedule AD path**

### Objective-trajectory debug status

We added an opt-in sampled trajectory objective-compare mode:

```bash
python examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py --parameter n0 --baseline-dt-path-safe-trajectory-compare-check --baseline-dt-path-safe-trajectory-sample-every 5
```

Latest result:

- mismatch starts early
- pressure channel remains very accurate
- `Er` and `Er2` mismatch grow steadily and dominate the error

Observed pattern:

- accepted step `1`: small mismatch already present
- accepted steps `6+`: clear `Er` drift begins
- later steps: `Er` mismatch grows strongly while pressure stays tiny

This strongly suggests:

- the custom adaptive tangent propagation is slightly wrong from the start
- the error accumulates primarily in the `Er` channel
- the bug is not a broad whole-state derivative failure

### New dedicated state-slice debug mode

To distinguish whether only `dEr` is drifting or whether the whole state
tangent is drifting with `Er` amplifying it, a new opt-in state-slice trajectory
mode was added.

Command:

```bash
python examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py --parameter n0 --baseline-dt-path-safe-state-trajectory-compare-check --baseline-dt-path-safe-trajectory-sample-every 5
```

What it is intended to print at sampled accepted steps:

- `full_state_rel_err`
- `density_rel_err`
- `pressure_rel_err`
- `Er_rel_err`

This mode is separate from the standard benchmark path and shares the same
sampling knob:

- `--baseline-dt-path-safe-trajectory-sample-every K`

so it can be made sparse to reduce memory / output volume.

### Current best interpretation

- initial ambipolar `Er` differentiation is **not** the source of the current
  mismatch in this benchmark path
- fixed-path direct AD is the trusted reference
- custom adaptive AD is the path that is wrong
- the mismatch appears early and accumulates primarily in `Er`

### Immediate next thing to run

Run the new state-slice sampled compare:

```bash
python examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py --parameter n0 --baseline-dt-path-safe-state-trajectory-compare-check --baseline-dt-path-safe-trajectory-sample-every 5
```

If that confirms that pressure/density remain tight while `Er` diverges, the
next debugging target should be the custom adaptive propagation of:

- `dEr`
- and any carry/history that feeds the `Er` update, especially `prev_stages`

### 2026-05-23 update: state-slice trajectory result

We ran the state-slice sampled compare at every accepted step:

```bash
python examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py --parameter n0 --baseline-dt-path-safe-state-trajectory-compare-check --baseline-dt-path-safe-trajectory-sample-every 1
```

Key result:

- `density_rel_err` stays essentially zero
- `pressure_rel_err` stays very small
- `Er_rel_err` is already nonzero at accepted step `1`
- `Er_rel_err` then grows monotonically and strongly

Representative values:

- accepted step `1`: `Er_rel_err ~ 2.8e-03`
- accepted step `5`: `Er_rel_err ~ 1.13e-01`
- accepted step `10`: `Er_rel_err ~ 3.90e-01`
- accepted step `36`: `Er_rel_err ~ 1.96e+00`
- accepted step `45`: `Er_rel_err ~ 2.16e+01`

This is strong evidence that:

- the mismatch is not just a long-time accumulation from other channels
- the custom adaptive tangent path is already wrong for `dEr` at the first
  accepted step

### New first-step localizer

A dedicated first-step field compare mode was added:

```bash
python examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py --parameter n0 --baseline-dt-path-first-step-field-compare-check
```

This mode prints, for the first accepted step only:

- `density`: adaptive/direct max abs, absolute error, relative error
- `pressure`: adaptive/direct max abs, absolute error, relative error
- `Er`: adaptive/direct max abs, absolute error, relative error

This is the current best localizer for the next debugging step, because it
removes long-horizon accumulation and focuses directly on the first accepted
step where the `dEr` mismatch already exists.

### New first-step local tangent split

A second first-step diagnostic was added to separate:

- mismatch in the stage tangent solve itself
- mismatch only in the accepted-state reconstruction

Run:

```bash
python examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py --parameter n0 --baseline-dt-path-first-step-local-tangent-compare-check
```

This compares, on the first accepted step only, the custom adaptive JVP
against direct JAX differentiation of the raw one-step attempt for:

- `trial_y`
- `carry_after_attempt.y`
- `stage_history`

and reports:

- `full_rel_err`
- `pressure_rel_err`
- `Er_rel_err`

Interpretation:

- if `stage_history` is already wrong, the bias starts in the local stage
  tangent solve
- if `stage_history` is fine but `trial_y` is wrong, the bias is introduced in
  the accepted-step reconstruction / carry shaping
- if `carry_after_attempt.y` differs much more strongly than `trial_y`, that
  points at the tangent packaging in `_radau_build_approximate_tangent_result`

### Exact one-step residual debug

A second debug-only first-step mode was added to test whether the current
approximation itself is the issue:

```bash
python examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py --parameter n0 --baseline-dt-path-first-step-exact-local-tangent-compare-check
```

This computes an exact local implicit-diff tangent for the first accepted step
by:

- forming the full collocation residual Jacobian with respect to `z`
- differentiating the residual source with respect to `(y_n, h)`
- solving the dense linear system for `dZ`

and then compares:

- `custom_vs_direct`
- `exact_vs_direct`
- `custom_vs_exact`
- `restricted_direct_vs_direct`
- `custom_vs_restricted_direct`

for:

- `trial_y`
- `stage_history`

If `exact_vs_direct` is much better than `custom_vs_direct`, then the custom
accepted-step tangent approximation is the root cause, not the later adaptive
composition machinery.

The mode also now compares against a "restricted direct" JVP where only
`(y, dt)` tangents are active and all other carry tangents are zeroed. That is
the key discriminator for whether the current mismatch is really coming from
ignored carry-field tangents such as:

- `lagged_response_cache`
- `lagged_reference_y`
- `prev_stages`
- other cached/controller carry fields

### 2026-05-23 checkpoint

Current progress:

- the old long-horizon adaptive AD `NaN` issue is no longer the main blocker
- the remaining blocker is a **correctness mismatch** in the custom adaptive AD
- on the safe baseline-`dt` path:
  - fixed-path FD and fixed-path direct AD agree well enough to trust the comparison setup
  - adaptive/custom AD disagrees
- so the bug is in the **custom adaptive derivative path**, not in the FD reference

What is localized now:

- the mismatch starts at the **first accepted step**
- it accumulates over accepted steps
- it is dominated by the **`Er` tangent**
- density tangent is essentially exact in the short safe window
- pressure tangent mismatch exists but is much smaller

Key results already observed:

1. First-step field compare:

```bash
python examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py --parameter n0 --baseline-dt-path-first-step-field-compare-check
```

Observed:

- `density rel err ~ 2.36e-17`
- `pressure rel err ~ 7.20e-07`
- `Er rel err ~ 2.83e-03`

Interpretation:

- the custom tangent is already biased in `dEr` at the first accepted step

2. First-step local tangent split:

```bash
python examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py --parameter n0 --baseline-dt-path-first-step-local-tangent-compare-check
```

Observed:

- `trial_y Er_rel_err ~ 2.826e-03`
- `stage_history Er_rel_err ~ 3.349e-03`
- `carry_after_attempt_y` exactly matches

Interpretation:

- the bug is **not** primarily in later carry packaging
- the first-step bias is already present in the local stage tangent `dZ`

3. Exact residual-based one-step compare:

```bash
python examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py --parameter n0 --baseline-dt-path-first-step-exact-local-tangent-compare-check
```

Observed before the restricted-direct rerun:

- `custom_vs_direct`
  - `trial_y Er_rel_err ~ 2.826e-03`
  - `stage_history Er_rel_err ~ 3.349e-03`
- `exact_vs_direct`
  - `trial_y Er_rel_err ~ 2.686e-03`
  - `stage_history Er_rel_err ~ 3.252e-03`
- `custom_vs_exact`
  - `trial_y Er_rel_err ~ 1.054e-03`
  - `stage_history Er_rel_err ~ 1.346e-03`

Interpretation:

- using the exact collocation residual Jacobian improves the first-step mismatch only slightly
- so the bug is **not explained solely** by the current local stage-linearization approximation

Current leading hypothesis:

- the custom accepted-step JVP is likely dropping derivative dependence through carry fields that full direct JVP still sees
- strongest suspects:
  - `lagged_response_cache`
  - `lagged_reference_y`
  - `prev_stages`
  - possibly other cached/controller carry fields

Current problem:

- the restricted-direct extension to the exact-local helper initially crashed because `restricted_direct_tangent` was not wired in
- that wiring bug has now been fixed and syntax-checked
- the **restricted-direct rerun has not yet been collected**

Immediate next command:

```bash
python examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py --parameter n0 --baseline-dt-path-first-step-exact-local-tangent-compare-check
```

New lines to inspect:

- `restricted_direct_vs_direct`
- `custom_vs_restricted_direct`

Decision rule:

- if `custom_vs_restricted_direct` is much smaller than `custom_vs_direct`, then the missing derivative route is in ignored carry-field tangents
- if it is not much smaller, then the accepted-step custom tangent math itself is still the main bug site

### 2026-05-24 checkpoint

Current status has moved materially beyond the 2026-05-23 first-step picture.

#### What was fixed

The main accepted-step custom JVP bug we found was in the tangent handling for:

- `lagged_response_cache`

Two distinct issues were patched:

1. the lagged-response tangent contribution into the local accepted-step tangent
   solve
2. the tangent of the **output** lagged-response cache carried to the next step
   (instead of incorrectly reusing the old input cache tangent as a placeholder)

That second fix was the important multi-step one.

#### What became exact after the fix

Cheap localized realized-trace custom-vs-direct checks now show:

- step 1: essentially exact
- step 2: essentially exact
- step 3: essentially exact
- step 6: essentially exact

Important commands/results:

```bash
python examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py --parameter n0 --baseline-dt-path-second-step-carry-ablation-check
```

Observed after the cache-output tangent fix:

- `custom_vs_direct Er_rel_err ~ 9.69e-14`

```bash
python examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py --parameter n0 --baseline-dt-path-third-step-carry-ablation-check
```

Observed:

- `custom_vs_direct Er_rel_err ~ 6.24e-11`
- `custom_scan_vs_manual Er_rel_err ~ 3.94e-11`
- `direct_scan_vs_manual Er_rel_err ~ 5.08e-02`

Interpretation:

- the custom accepted-step JVP itself is fine at step 3
- the old trajectory helper/direct reference plumbing was part of the apparent
  mismatch

```bash
python examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py --parameter n0 --realized-trace-sixth-step-carry-ablation-check
```

Observed:

- `custom_vs_direct Er_rel_err ~ 3.37e-11`
- `carry_after_step5_custom_vs_direct Er_rel_err ~ 4.93e-11`

Interpretation:

- the realized-trace custom JVP and realized-trace direct JVP match through at
  least accepted step 6

#### What is no longer the main problem

At this point the evidence no longer supports:

- "the accepted-step custom solver JVP is still broadly wrong"

Instead, the remaining contradictions are now primarily in the **benchmark /
trajectory debug helpers** and in **target mismatch** between:

- realized-trace derivative targets
- fixed-`dt` accepted-path derivative targets

#### Important benchmark interpretation change

The comparison:

- custom realized-trace AD vs fixed-`dt` direct AD

is not the same derivative target once history/carry dependence matters.
So large mismatch there is no longer sufficient evidence that the custom JVP is
wrong.

The cleaner target is:

- realized-trace custom AD vs realized-trace direct AD

and localized checkpoint checks now support that this is matching well at least
through step 6.

#### Practical debugging policy from here

Do **not** keep using the heavy long trajectory diagnostics as the main truth
source. They were:

- RAM-heavy
- slow
- and in some cases comparing mixed targets or buggy helper paths

Instead, use only cheap localized checkpoint tests.

#### Recommended low-cost validation path

Use one accepted-step checkpoint at a time:

```bash
python examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py --parameter n0 --realized-trace-checkpoint-compare-check --realized-trace-checkpoint-index 10
```

Then, if still good:

```bash
python examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py --parameter n0 --realized-trace-checkpoint-compare-check --realized-trace-checkpoint-index 20
```

If later we want a small batch, use sparse checkpoints only:

```bash
python examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py --parameter n0 --realized-trace-sparse-checkpoint-compare-check --realized-trace-sparse-checkpoint-counts 10,20
```

Full-trajectory compare modes are now treated as RAM-heavy diagnostics and should
only be run with:

```bash
--allow-heavy-trajectory-diagnostics
```

Without that override, prefer the checkpoint-localized modes above.

These modes were added specifically to avoid the RAM blow-up from full
trajectory scans.

#### Current concise conclusion

- the key solver-side derivative bug was the `lagged_response_cache` tangent
  propagation
- that bug has been fixed
- localized realized-trace custom-vs-direct checks are now very strong through
  accepted step 6
- the remaining work is mostly validation cleanup and efficient checkpointed
  confirmation, not broad blind solver-derivative debugging

### 2026-05-25 checkpoint

#### Realized-trace custom-vs-direct checkpoint picture is now strong end-to-end

Trusted single-checkpoint custom-vs-direct runs for `parameter=n0` gave:

- accepted step `30`: `Er_rel_err ~ 3.60e-09`
- accepted step `45`: `Er_rel_err ~ 1.62e-09`
- accepted step `90`: `Er_rel_err ~ 1.29e-08`
- accepted step `115`: `Er_rel_err ~ 2.65e-06`

Interpretation:

- the custom realized-trace AD path now matches the direct-AD reference very
  well all the way to the final accepted step
- there is no sign of catastrophic tangent drift
- the main solver-side AD bug really was the `lagged_response_cache`
  tangent-propagation/carry path

At this point, custom-vs-direct mismatch is no longer the main concern.

#### Same-target local AD-vs-FD is now working

A same-target frozen-trace checkpoint FD mode was added:

```bash
python examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py \
  --parameter n0 \
  --realized-trace-checkpoint-frozen-fd-check \
  --realized-trace-checkpoint-index 45
```

Important FD-step rule:

```text
fd_step = max(fd_abs_step, fd_rel_step * max(abs(parameter), 1.0))
```

Because the CLI default still has:

- `--fd-abs-step 1e-4`

changing only `--fd-rel-step` below that floor does **not** change the actual
perturbation.

This explained why:

- `--fd-rel-step 1e-5`
- `--fd-rel-step 1e-6`

initially produced the same FD result: both were clamped to `fd_step = 1e-4`.

#### Best local derivative validation found so far

At checkpoint `45`, this command:

```bash
python examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py \
  --parameter n0 \
  --realized-trace-checkpoint-frozen-fd-check \
  --realized-trace-checkpoint-index 45 \
  --fd-rel-step 1e-6 \
  --fd-abs-step 1e-7
```

used:

- `fd_step = 4.21e-06`

and produced approximately:

- `softmax_Er`: `5.36e-07`
- `smooth_root_proxy`: `3.24e-05`
- `Er2_volume_average`: `2.79e-07`
- `Er_volume_average`: `9.02e-07`
- `electron_temperature_volume_average_keV`: `1.19e-08`
- `total_pressure_volume_average`: `1.44e-10`
- `alpha_power_volume_average_mw_m3`: `1.14e-09`

This is the strongest local AD-vs-FD evidence obtained so far.

#### FD got worse again when made too small

At the same checkpoint, this command:

```bash
python examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py \
  --parameter n0 \
  --realized-trace-checkpoint-frozen-fd-check \
  --realized-trace-checkpoint-index 45 \
  --fd-rel-step 1e-7 \
  --fd-abs-step 1e-8
```

used:

- `fd_step = 4.21e-07`

and the worst relative error increased again to about:

- `smooth_root_proxy`: `6.50e-04`

Interpretation:

- this is normal finite-difference behavior
- too-large `h` gives truncation/nonlinearity error
- too-small `h` gives cancellation / solver-noise / postprocessing-noise
- the local optimum here appears to be around `fd_step ~ 4.21e-06`, not
  `4.21e-07`

So the present conclusion is:

- local AD gradients are very likely correct
- remaining discrepancy is now dominated by FD calibration, not by a broken AD
  path

#### Important conceptual distinction

The frozen-trace checkpoint FD test validates the **local derivative** of the
baseline realized solution map.

It does **not** prove:

- that the same linear gradient remains accurate for large perturbations
- or that adaptive `fd+` / `fd-` with their own evolved schedules will match
  the same frozen-map derivative

So:

- local derivative correctness: strongly supported
- larger-perturbation adaptive behavior: still a separate question

#### New direction: interpolation-based adaptive FD

Because larger perturbations may need their own adaptive schedules, a new mode
was added to compare:

- baseline realized-trace checkpoint AD
- against adaptive `fd_minus` / `fd_plus`
- with their objective trajectories interpolated to the baseline checkpoint time

CLI flag:

```bash
--realized-trace-checkpoint-interpolated-fd-check
```

Recommended first command:

```bash
python examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py \
  --parameter n0 \
  --realized-trace-checkpoint-interpolated-fd-check \
  --realized-trace-checkpoint-index 45 \
  --fd-rel-step 1e-6 \
  --fd-abs-step 1e-7
```

Purpose:

- avoid forcing `fd+` / `fd-` onto the baseline attempted-step map
- let each perturbed solve use its own stable adaptive schedule
- compare on a common physical time via interpolation

#### Resume-next-session priorities

1. Trust the checkpoint custom-vs-direct evidence as the main AD-health signal.
2. Use `fd_rel_step=1e-6`, `fd_abs_step=1e-7` as the current best local-FD
   calibration point.
3. Use the new interpolated adaptive-FD mode for larger-perturbation and
   schedule-sensitive validation.

### 2026-05-26 derivative-target clarification

#### Two different derivative notions must be kept separate

There are two different derivatives in play:

1. **Full adaptive-algorithm derivative**
   - includes sensitivity through:
     - accepted/rejected-step logic
     - timestep-controller evolution
     - solver-history / reuse heuristics
   - this is the derivative of the **numerical algorithm**

2. **Physical/local derivative along the realized solve path**
   - treat adaptive logic as an auxiliary numerical device
   - differentiate the transport evolution along the locally realized path
   - do **not** treat timestep-controller branching as part of the desired
     physical derivative target

For the current transport-optimization use case, the second notion is probably
the more physically meaningful one.

#### Important interpretation for the current custom AD path

The current custom AD path does **not** start from an externally fixed time
grid. Instead:

- the primal adaptive solve runs first
- it realizes an adaptive trace
- the custom JVP then differentiates **along that realized trace**

So:

- the primal path is still chosen adaptively
- but the tangent replay treats the realized trace metadata as fixed

This is different from saying:

- “the solver always runs on a fixed prescribed time path”

which is **not** what is happening.

#### What is artificially frozen

The artificial freezing happens mainly in the **validation comparisons**, not in
the forward primal solve itself.

Examples:

- `--realized-trace-checkpoint-frozen-fd-check`
  - uses the baseline realized trace prefix
  - forces both `fd-` and `fd+` onto that same frozen replay trace
  - this is a same-target local derivative validation tool

- custom-vs-direct checkpoint comparisons
  - also compare derivatives on the same realized/frozen local map

These tools are intended to answer:

- “is the local derivative along the realized path correct?”

not:

- “what is the derivative of every adaptive controller branch decision?”

#### Consequence for interpretation

If the intended scientific derivative is the physically meaningful local
transport sensitivity, then it is reasonable to **not** differentiate through
adaptive controller logic.

In that case:

- differentiating accepted-step state evolution along the realized path is the
  important target
- timestep-controller branching is a solver artifact, not part of the desired
  physical derivative

This means the current custom path may still be conceptually appropriate even if
it is not the exact derivative of the full adaptive algorithm-as-code.

#### Confirmed implementation fact

For `--realized-trace-checkpoint-frozen-fd-check`, `fd-` and `fd+` really are
evaluated on the **same frozen baseline trace prefix**:

- same frozen `accepted_mask`
- same frozen `active_mask`
- same frozen attempted/next `dt` sequence
- same frozen controller-trace metadata

So any mismatch seen in that mode is **not** explained by adaptive path
divergence between `fd-` and `fd+`.

### 2026-05-26 Diffrax adaptive-AD reference

#### Purpose of this note

This note records what Diffrax is actually doing for adaptive AD, especially for
the comparable case:

- `Kvaerno5`
- `PIDController`
- possible rejected steps

The goal was to check both the local installed Diffrax source and the upstream
Diffrax docs/repo, and determine whether Diffrax:

- differentiates through the adaptive solve loop
- freezes the realized adaptive trace
- differentiates through timestep-controller logic
- differentiates through rejected steps

#### High-level conclusion

Diffrax does **not** use a NEOPAX-style:

- run adaptive primal
- freeze the whole realized trace
- replay a custom tangent on that frozen trace

for its main adaptive AD path.

Instead, Diffrax differentiates the **discrete numerical solve loop directly**.

However, Diffrax still makes a deliberate compromise:

- it does **not** fully differentiate all timestep-controller quantities
- it explicitly detaches selected controller-update quantities inside
  `PIDController`

So Diffrax is best described as:

- differentiate the adaptive solver loop directly
- but pragmatically stop gradients through selected controller scalars

#### Main local source facts

##### 1. Default reverse-mode AD differentiates the numerical solver directly

Diffrax's default adjoint is `RecursiveCheckpointAdjoint`, whose docstring says
it differentiates the numerical solution directly:

- local source:
  - `diffrax/_adjoint.py`
  - `RecursiveCheckpointAdjoint`
- upstream docs:
  - <https://docs.kidger.site/diffrax/api/adjoints/>

Relevant local source facts:

- `RecursiveCheckpointAdjoint` is defined at:
  - `diffrax/_adjoint.py:174`
- it uses checkpointed while loops:
  - `diffrax/_adjoint.py:289`
- it calls the ordinary main solve loop rather than a separate replay tangent
  path

This is the discrete-adjoint / discretise-then-optimise route.

##### 2. Forward-mode also goes through the solver internals

`ForwardMode` is defined at:

- `diffrax/_adjoint.py:864`

For Runge-Kutta solvers, it forces:

- `scan_kind = "lax"`

when `scan_kind is None`, so that forward-mode autodiff can pass through the
internal stage loop:

- `diffrax/_adjoint.py:891`

This again indicates that Diffrax is differentiating the solve internals, not a
frozen replayed trace.

##### 3. The adaptive solve loop includes rejected steps inside the AD'd loop

The main solve loop is:

- `diffrax/_integrate.py:273`

Inside each iteration it does:

1. `solver.step(...)`
2. `stepsize_controller.adapt_step_size(...)`
3. conditional keep/reject of the step state

The controller is called at:

- `diffrax/_integrate.py:360`

Accepted/rejected branching is then applied with `jnp.where`-style logic:

- `diffrax/_integrate.py:386`

Rejected steps are counted explicitly at:

- `diffrax/_integrate.py:407`

So rejected steps are part of the actual differentiated adaptive loop. Diffrax
is not removing them from the loop at the architectural level.

##### 4. `Kvaerno5` uses the standard adaptive RK infrastructure

`Kvaerno5` is an `AbstractESDIRK` adaptive RK solver:

- `diffrax/_solver/kvaerno5.py`

Its RK stage loop is implemented via the generic RK machinery:

- `diffrax/_solver/runge_kutta.py:444`

The internal RK stage loop uses `eqxi.while_loop` with checkpointing by
default:

- `diffrax/_solver/runge_kutta.py:1155`

So for `Kvaerno5 + PIDController`, the adaptive solve is still running through
the normal differentiated RK + controller loop.

#### Important nuance: Diffrax still freezes some controller quantities

This is the most important subtlety.

Diffrax does **not** fully differentiate the timestep-controller update logic.

The strongest evidence is in:

- `diffrax/_step_size_controller/adaptive.py`

##### Explicit detachments in `PIDController`

1. Auto-selected initial step size:

- `dt0 = lax.stop_gradient(dt0)`
- location:
  - `diffrax/_step_size_controller/adaptive.py:444`

2. The scaled error proxy used in the PID update:

- `inv_scaled_error = lax.stop_gradient(inv_scaled_error)`
- location:
  - `diffrax/_step_size_controller/adaptive.py:583`

3. The multiplicative PID factor used to update the next timestep:

- `factor = lax.stop_gradient(factor)`
- `factor = eqxi.nondifferentiable(factor)`
- locations:
  - `diffrax/_step_size_controller/adaptive.py:611`
  - `diffrax/_step_size_controller/adaptive.py:612`

So the next-step `dt` update is deliberately treated as nondifferentiable in
the controller update formula, even though the overall solve remains adaptive.

##### Diffrax's own explanation for doing this

The local source comments in `PIDController.init` are unusually explicit. The
author says, in summary:

- this dramatically speeds up gradient computations
- on some training problems it improves training behaviour
- they have not observed it hurting training
- other libraries do something similar without remark
- there is a folk intuition that time discretisation is â€œjust an implementation
  detailâ€ and one â€œdoesn't need to backpropagate through rejected stepsâ€

But the same comment also says the author is **not fully convinced** by this
argument, noting in particular:

- it feels morally wrong from the differentiable-programming viewpoint
- rejected steps really are part of the computational graph
- step-size choices do affect the computed solution
- certain esoteric optimization goals could fail if these gradients are removed

So the source itself presents this as a **pragmatic compromise**, not as a
principled statement that controller logic should never be differentiated.

#### Additional solver-side detachments

There are also some solver-side `stop_gradient` / nondifferentiable choices in
the implicit RK internals, for example around Jacobian/cache handling in:

- `diffrax/_solver/runge_kutta.py`

Examples:

- `lax.stop_gradient(f_pred)`
- `_filter_stop_gradient(...)`
- `eqxi.nondifferentiable(jac_f, name="jac_f")`
- `eqxi.nondifferentiable(jac_k, name="jac_k")`

These appear to be mostly there for custom-VJP / implicit-solver practicality,
not because Diffrax is globally freezing the realized adaptive trace.

#### Side-by-side comparison with NEOPAX

| Question | Diffrax | Current NEOPAX custom adaptive JVP |
|---|---|---|
| Does the primal solve run adaptively? | Yes | Yes |
| Is the main AD path built by differentiating the adaptive solve loop directly? | Yes | Not exactly; primal adaptive solve runs first, then tangent replay follows the realized trace |
| Does the main AD path freeze the whole realized adaptive trace? | No | Yes, the custom JVP reuses frozen realized-trace metadata |
| Are rejected steps part of the differentiated loop execution? | Yes | Not in the same architectural way; the custom tangent follows the realized replay |
| Are timestep-controller quantities fully differentiated? | No | No |
| What controller quantities are explicitly detached? | `dt0`, inverse scaled error, PID factor | realized trace metadata, controller-history fields, cache/Jacobian/LU state, etc. |
| Overall philosophy | Differentiate the adaptive discrete solve, but detach selected controller updates | Differentiate along the realized adaptive path using a solver-native replay approximation |

#### Interpretation for NEOPAX design discussion

Diffrax does **not** support either extreme:

1. **Extreme A:** freeze all adaptive logic and replay a separate tangent path
2. **Extreme B:** differentiate every adaptive-controller scalar exactly

Instead Diffrax sits in the middle:

- it differentiates the adaptive solve loop itself
- but detaches selected controller-update quantities for pragmatic reasons

So if we use Diffrax as a reference point, the most careful conclusion is:

- Diffrax is more adaptive-loop-faithful than the current NEOPAX custom replay
  design
- but Diffrax still does not fully differentiate timestep-controller logic
- therefore Diffrax does **not** provide evidence that every adaptive-control
  quantity should be differentiated

#### Useful references consulted

- local installed source:
  - `diffrax/_adjoint.py`
  - `diffrax/_integrate.py`
  - `diffrax/_step_size_controller/adaptive.py`
  - `diffrax/_solver/runge_kutta.py`
  - `diffrax/_solver/kvaerno5.py`
- upstream docs:
  - <https://docs.kidger.site/diffrax/api/adjoints/>
- upstream repo pages:
  - <https://github.com/patrick-kidger/diffrax/blob/main/diffrax/_adjoint.py>
  - <https://github.com/patrick-kidger/diffrax/blob/main/diffrax/_integrate.py>
  - <https://github.com/patrick-kidger/diffrax/blob/main/diffrax/_step_size_controller/adaptive.py>

### 2026-05-26 frozen-checkpoint FD localization in the low-`dt` region

#### Purpose

Before changing the custom AD architecture, the main question became:

- why does frozen-path AD-vs-FD get much worse at later checkpoints?

We extended the existing frozen-checkpoint report to print:

- objective-level custom/direct/FD comparison
- state-tangent comparison:
  - `custom_vs_fd`
  - `direct_vs_fd`
  - `custom_vs_direct`
  - each with:
    - `full_rel_err`
    - `pressure_rel_err`
    - `Er_rel_err`

This kept the target exactly the same while exposing whether the mismatch was
already present in the checkpoint state tangent.

#### Key result: the mismatch is already in the state tangent

At checkpoint `115`, the report showed:

- `custom_vs_fd` state tangent:
  - `full_rel_err ~ 6.98e-02`
  - `pressure_rel_err ~ 4.60e-02`
  - `Er_rel_err ~ 7.10e-02`
- `direct_vs_fd` is essentially identical
- `custom_vs_direct` remains tiny:
  - around `2.6e-06`

So:

- the late mismatch is **not** being created mainly by objective
  postprocessing
- it is already present in the checkpoint state derivative
- it is **not** custom-specific
- both AD constructions share the same disagreement with the frozen FD

#### Checkpoint scan showed a localized transition, not smooth accumulation

Frozen-checkpoint state-tangent errors were:

- accepted step `90`:
  - `full_rel_err ~ 6.86e-04`
- accepted step `100`:
  - `~1.15e-03`
- accepted step `102` with the original FD step (`4.21e-06`):
  - `~2.14e-01`
- accepted step `103`:
  - `~1.47e-01`
- accepted step `105`:
  - `~1.01e-01`
- accepted step `108`:
  - `~7.49e-02`
- accepted step `110`:
  - `~6.96e-02`
- accepted step `115`:
  - `~6.98e-02`

This means:

- the problem is **not** a smooth long-horizon drift
- the main transition happens sharply between accepted steps `100` and `102`
- the mismatch then relaxes somewhat and plateaus around the `~7e-02` level

#### Interpretation: this aligns with the region where adaptive `dt` becomes very small

The sharp transition was observed right around a region where the primal adaptive
solve drops to significantly smaller `dt`.

This suggests the frozen FD comparison is becoming difficult because the map is
locally much more sensitive there:

- many tiny implicit updates are accumulated
- FD subtracts two very close long-prefix states
- cancellation/noise becomes more important
- the â€œgoodâ€ FD step size becomes smaller and more delicate

So this looks much more like a **low-`dt` / stiff-window FD-calibration
problem** than a generic AD breakdown.

#### Crucial FD-step sweep result at checkpoint `102`

Using the original effective FD step:

- `fd_step = 4.21e-06`
- checkpoint `102`
- state tangent `full_rel_err ~ 2.14e-01`

Then rerunning checkpoint `102` with a smaller FD step:

- `--fd-rel-step 3e-7`
- `--fd-abs-step 1e-8`
- effective `fd_step = 1.263e-06`

produced:

- state tangent `full_rel_err ~ 1.08e-02`
- `pressure_rel_err ~ 1.41e-02`
- `Er_rel_err ~ 1.08e-02`

Objective errors also dropped sharply.

This is very strong evidence that:

- the large checkpoint-102 discrepancy at the original FD step was mostly an
  **FD-step problem**
- not compelling evidence of an AD failure

We then extended the sweep further in both directions.

Larger FD step:

- `--fd-rel-step 3e-6`
- `--fd-abs-step 1e-7`
- effective `fd_step = 1.263e-05`

This produced catastrophic disagreement:

- state tangent `full_rel_err ~ 1.07e+02`
- `pressure_rel_err ~ 2.34e+00`
- `Er_rel_err ~ 1.09e+02`

So that FD step is completely unusable in the low-`dt` region.

Smaller FD steps:

1. `--fd-rel-step 1e-7`, `--fd-abs-step 1e-9`
   - effective `fd_step = 4.21e-07`
   - state tangent:
     - `full_rel_err ~ 1.09e-03`
     - `pressure_rel_err ~ 1.69e-03`
     - `Er_rel_err ~ 1.09e-03`

2. `--fd-rel-step 3e-8`, `--fd-abs-step 1e-10`
   - effective `fd_step = 1.263e-07`
   - state tangent:
     - `full_rel_err ~ 6.27e-05`
     - `pressure_rel_err ~ 1.65e-04`
     - `Er_rel_err ~ 6.27e-05`

The corresponding objective errors also became very small.

This turns the checkpoint-102 story into a textbook FD-calibration result:

- too large `h` gives catastrophic truncation / nonlinear secant error
- modestly smaller `h` gives a large improvement
- much smaller `h` gives excellent AD-vs-FD agreement

So at checkpoint `102`, there is currently **no meaningful evidence of an AD
failure**. The disagreement was overwhelmingly due to a poor FD step choice for
that low-`dt` regime.

#### Important practical conclusion

At least in the low-`dt` window:

- `custom AD` remains stable
- `direct AD` remains stable
- FD can change dramatically with `h`

So when deciding whether the issue is AD or FD:

- if FD moves a lot when `h` changes but AD does not, that points primarily to
  **FD instability**
- this is exactly what the checkpoint-102 sweep started to show

#### Benchmark convenience change

To speed up these FD sweeps, we added:

- `--skip-direct-ad-in-frozen-check`

for:

- `--realized-trace-checkpoint-frozen-fd-check`

This lets the benchmark skip the direct-AD reference path and run only:

- custom AD
- FD

plus the custom-vs-FD state-tangent diagnostics.

This is useful when the goal is FD calibration, not AD-path decomposition.

#### Current best next-step ideas for improving FD precision

The main ideas discussed were:

1. **Calibrate FD step locally**
   - especially at the onset checkpoint (`102`) or nearby
   - the globally good FD step from easier regions may be too large in the
     low-`dt` window

2. **Use a higher-order stencil**
   - e.g. a 5-point central FD instead of the current 2-point central FD
   - this reduces truncation error if the local map is smooth enough

3. **Use a more local comparison window**
   - instead of FD on a long prefix from the start, use a shorter frozen window
     around the troublesome accepted-step region
   - this reduces accumulation of tiny numerical differences

4. **Tighten the forward numerical solves if needed**
   - stricter nonlinear/linear tolerances
   - less approximate reuse in the sensitive region
   - this improves the raw primal solve quality entering the FD subtraction

5. **Keep state-level FD diagnostics, not just objective-level**
   - because the state-tangent report was what made the low-`dt` diagnosis clear

#### Current best interpretation

The most likely picture right now is:

- the custom AD path is not the main issue
- the direct AD path is not the main issue
- the frozen FD comparison becomes very delicate in the small-`dt` region
- with a smaller local FD step, the AD-vs-FD agreement improves a lot
- at checkpoint `102`, the agreement can be made excellent with sufficiently
  small `h`

So the next emphasis should be:

- improve the FD reference locally
- not immediately re-architect the custom derivative again

#### Best next command recorded for resuming later

The most natural follow-up after the checkpoint-102 sweep is to test whether a
later troublesome checkpoint also improves dramatically with a smaller FD step,
for example:

```bash
python examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py \
  --parameter n0 \
  --realized-trace-checkpoint-frozen-fd-check \
  --realized-trace-checkpoint-index 110 \
  --fd-rel-step 3e-8 \
  --fd-abs-step 1e-10 \
  --skip-direct-ad-in-frozen-check
```

This is the current best next test to check whether the apparent late-time
frozen-FD discrepancy was also mostly an FD-calibration artifact.

#### Later checkpoint confirmation: `110` and `115`

That follow-up was run, and it confirmed the same story at later checkpoints.

At checkpoint `110`, using:

```bash
python examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py \
  --parameter n0 \
  --realized-trace-checkpoint-frozen-fd-check \
  --realized-trace-checkpoint-index 110 \
  --fd-rel-step 3e-8 \
  --fd-abs-step 1e-10 \
  --skip-direct-ad-in-frozen-check
```

the result became:

- effective `fd_step = 1.263e-07`
- state tangent:
  - `full_rel_err ~ 1.72e-04`
  - `pressure_rel_err ~ 2.63e-04`
  - `Er_rel_err ~ 1.26e-04`
- worst objective relative error:
  - `~6.62e-04`

So the earlier scary checkpoint-110 discrepancy was also an FD-step artifact.

At checkpoint `115`, using:

```bash
python examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py \
  --parameter n0 \
  --realized-trace-checkpoint-frozen-fd-check \
  --realized-trace-checkpoint-index 115 \
  --fd-rel-step 3e-8 \
  --fd-abs-step 1e-10 \
  --skip-direct-ad-in-frozen-check
```

the result was similarly excellent:

- effective `fd_step = 1.263e-07`
- state tangent:
  - `full_rel_err ~ 9.28e-05`
  - `pressure_rel_err ~ 1.64e-04`
  - `Er_rel_err ~ 8.72e-05`
- worst objective relative error:
  - `~1.50e-04`

So by this point the late-checkpoint picture is very consistent:

- checkpoint `102`: fixed by smaller FD step
- checkpoint `110`: fixed by smaller FD step
- checkpoint `115`: fixed by smaller FD step

This is strong evidence that, for the frozen realized-path derivative target,
the custom AD path is working well and the previously large discrepancies were
primarily FD-calibration artifacts in the low-`dt` region.

#### Center vs 5-point stencil at checkpoint `115`

We also added and ran a dedicated frozen-path stencil comparison mode:

```bash
python examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py \
  --parameter n0 \
  --realized-trace-checkpoint-fd-stencil-check \
  --realized-trace-checkpoint-index 115 \
  --fd-rel-step 3e-8 \
  --fd-abs-step 1e-10
```

This mode compares:

- `custom_ad`
- `fd_center`
- `fd_five_point`

while reusing the already-computed center evaluations for the 5-point stencil.

With the good small FD step:

- center FD was already very good
- 5-point FD was usually a little better
- state tangent improved from:
  - center: `full_rel_err ~ 9.28e-05`
  - five-point: `full_rel_err ~ 4.62e-05`

Specifically for state tangent at checkpoint `115`:

- `custom_vs_fd_center`
  - `full_rel_err ~ 9.28e-05`
  - `pressure_rel_err ~ 1.64e-04`
  - `Er_rel_err ~ 8.72e-05`
- `custom_vs_fd_five_point`
  - `full_rel_err ~ 4.62e-05`
  - `pressure_rel_err ~ 3.22e-05`
  - `Er_rel_err ~ 4.68e-05`

We also checked the same stencil mode with a much larger FD step:

```bash
python examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py \
  --parameter n0 \
  --realized-trace-checkpoint-fd-stencil-check \
  --realized-trace-checkpoint-index 115 \
  --fd-rel-step 1e-6 \
  --fd-abs-step 1e-8
```

and both center and 5-point were poor there.

So the stencil conclusion is:

- 5-point FD can improve the reference once `h` is already in a good regime
- 5-point FD does **not** rescue a badly chosen perturbation
- higher-order stencil helps with truncation error, but does not replace local
  FD-step calibration

#### Updated overall conclusion

The current best overall interpretation is:

- the custom AD path is very likely good for the frozen realized-path
  derivative target
- the direct AD path had already agreed strongly with it
- the remaining large discrepancies were due to FD calibration, especially in
  the low-`dt` region
- center and 5-point FD now both support the same conclusion once the
  perturbation is chosen appropriately

### Updated priorities: NTX AD path and VMEC geometry path

After inspecting the local `NTX`, `vmec_jax`, `booz_xform_jax`, and
`SPECTRAX-GK` source more carefully, the next priorities should be updated.

#### Priority 1: `n0` A/B test for NTX derivative mode

We should explicitly compare the cost and derivative values of the current NTX
path against the NTX `custom_vjp` path in the familiar fixed-geometry `n0`
benchmark.

Reason:

- NTX already provides a dedicated custom-VJP monoenergetic coefficient solve:
  - `ntx.solve_prepared_coefficient_vector_vjp(...)`
- but the current NEOPAX exact-runtime `Lij` path appears to call:
  - `ntx.solve_prepared_coefficient_vector(...)`
  instead
- this means NEOPAX is differentiating through the NTX run, but may not be
  using NTX's more solver-aware reverse-mode path
- this could explain part of the higher AD-side NTX cost in the old `n0` runs

So the first concrete A/B should be:

1. current NTX direct-AD path
2. NTX `custom_vjp` path

and compare:

- derivative values
- wall time / compile time
- memory if practical

This should be kept on the same fixed-geometry `n0` benchmark so we isolate the
NTX derivative-cost question cleanly.

#### Priority 2: geometry benchmark must switch to the NTX-style VMEC path

The current geometry benchmark used the wrong VMEC helper:

- `vmec_jax.solve_fixed_boundary_from_boundary(..., differentiable=True)`

That path goes through the explicit GD solve and is not the normal
convergence-aware VMEC workflow. In practice:

- it is not the same staged path as `vmec_jax input.file`
- it is not the same path used by NTX for differentiable boundary solves
- it does not explain the local `vmec_jax` CLI behavior the user observed

The correct local reference is the NTX-style boundary-param AD path:

- boundary coefficient
- `-> vmec_jax.implicit.solve_fixed_boundary_state_implicit_vmec_residual(...)`
- `-> booz_xform_jax`
- `-> NTX support / NEOPAX transport`

This is the path we should use for geometry-parameter AD in NEOPAX.

Important note:

- the forward geometry map for FD should be the forward version of this same
  AD-capable VMEC/Boozer/NTX path
- then, as before, the actual transport derivative comparison should be done on
  the frozen realized transport path

#### Priority 3: rerun `RBC(1,0)` only after both fixes above

Only after:

1. the NTX derivative-mode comparison is implemented for `n0`
2. the geometry benchmark is rebuilt around the NTX-style implicit VMEC
   residual path

should we rerun the geometry benchmark for:

- `RBC(1,0)`

That rerun should again target the same NEOPAX transport metrics as before, and
the correct comparison should be:

- custom AD on the frozen realized path
- versus frozen-path FD

#### Extra clarification from local code inspection

The local ecosystem now looks like this:

- `NTX`:
  - serious differentiable boundary-parameter path uses the implicit VMEC
    residual solve
- `SPECTRAX-GK`:
  - the strongest local geometry-gradient reports are not using the bad
    explicit VMEC GD helper either
  - they mostly use a solved `vmec_jax` state coefficient
    `-> booz_xform_jax -> downstream objective` bridge
- `booz_xform_jax`:
  - is a JAX transform stage, not the source of the convergence/runtime issue

So the revised implementation order should be:

1. add an NTX derivative-mode switch for the fixed-geometry `n0` benchmark
2. compare NTX direct AD vs NTX `custom_vjp`
3. replace the geometry benchmark VMEC lane with the NTX-style implicit
   residual path
4. rerun `RBC(1,0)` frozen-path AD vs FD

#### Refinement: decouple magnetic AD from NTX and make forward/AD lanes explicit

After the first correction to the geometry benchmark, the next refinement is to
separate the **magnetic differentiation path** from the **transport-through-NTX
path** more cleanly.

The long-term design should **not** depend on NTX wrappers for the magnetic
solve itself. Instead, NEOPAX should own a direct magnetic pipeline based on:

- boundary coefficient parameter
- `-> vmec_jax` implicit residual solve
- `-> booz_xform_jax`
- `-> NEOPAX geometry/support object`

Then the transport layer can consume that geometry, but it should not own the
magnetic solve.

This also means the magnetic path should expose **two explicit lanes**:

1. **Forward geometry lane**
   - used for primal runs and FD evaluations
   - should use the proper forward VMEC/Boozer solve path
   - should define the reference forward map for geometry perturbations

2. **AD geometry lane**
   - same mathematical map as the forward lane
   - but implemented with the AD-capable solver-aware path
   - should not treat the direct `vmec_jax` implicit residual solve API as the
     promoted production geometry derivative lane

Important principle:

- forward/FD geometry runs should use the **forward lane**
- AD should use the **AD lane**
- both lanes must target the same underlying geometry map as closely as
  possible

So the geometry implementation priority is now:

1. remove dependence on NTX wrappers for the magnetic solve itself
2. replace the current implicit-helper benchmark lane with the accepted-point
   exact optimizer callbacks from `vmec_jax`
3. keep `booz_xform_jax` as the next stage
4. expose a NEOPAX-owned forward lane and AD lane for boundary-parameter
   geometry construction
5. only after that, rerun `RBC(1,0)` frozen-path AD vs FD

Immediate implementation note:

- the first geometry implementation should land the **NEOPAX-owned forward lane**
  and the **accepted-point exact forward derivative lane** first
- both lanes should still target the same final NEOPAX transport metric vector
  used in the `n0` tests
- once this forward lane is validated against FD on those same metrics, we
  should be able to add a reverse-mode magnetic path afterward rather than
  trying to start from reverse mode immediately

### Correct vmec_jax geometry differentiation lane

After re-reading the local `vmec_jax` repository, the accepted-point derivative
routes that still follow the usual input-driven VMEC solve are:

- primal solver path:
  - `vmec_jax.api.run_fixed_boundary(...)`
  - documented in:
    - `vmec_jax/docs/quickstart.rst`
    - `vmec_jax/vmec_jax/driver.py::run_fixed_boundary(...)`

- promoted forward derivative path:
  - accepted-point exact tape replay with JVP columns
  - documented in:
    - `vmec_jax/docs/optimization.rst`
    - `vmec_jax/docs/performance.rst`
    - `vmec_jax/vmec_jax/optimization.py::FixedBoundaryExactOptimizer.jacobian_fun`

- promoted reverse derivative path:
  - accepted-point reverse discrete-adjoint replay
  - documented in:
    - `vmec_jax/docs/discrete_adjoint.rst`
    - `vmec_jax/vmec_jax/optimization.py::FixedBoundaryExactOptimizer.objective_and_gradient_fun`
    - `vmec_jax/vmec_jax/optimization.py::FixedBoundaryExactOptimizer.residual_linear_operator`

Important local conclusion:

- the geometry benchmark should **not** use
  `vmec_jax.implicit.solve_fixed_boundary_state_implicit_vmec_residual(...)`
  as the standard AD lane
- that implicit helper is not the main promoted accepted-point optimization
  path that matches the usual VMEC input / iteration workflow

Implementation priority from here:

1. build the NEOPAX geometry derivative helpers around a one-parameter
   accepted-point `FixedBoundaryExactOptimizer`
2. validate the accepted-point **forward** derivative path against central FD
   on scalar VMEC / VMEC->Boozer observables
3. validate the accepted-point **reverse** path against the trusted forward
   derivative values
4. only then reconnect the corrected geometry derivative lane to the full
   NEOPAX transport metric benchmark

#### Raw saved terminal values from the frozen checkpoint runs

These are copied from the terminal outputs pasted during the session so the raw
FD reference values are preserved here as well, not just the relative errors.

Checkpoint `110`, frozen central FD, `fd_step = 1.263000e-07`

- `softmax_Er`
  - `custom_ad = 2.027686e+01`
  - `fd = 2.027908e+01`
- `smooth_root_proxy`
  - `custom_ad = -8.442966e-05`
  - `fd = -8.437382e-05`
- `Er2_volume_average`
  - `custom_ad = 4.344449e+01`
  - `fd = 4.344444e+01`
- `Er_volume_average`
  - `custom_ad = 3.161107e+00`
  - `fd = 3.161492e+00`
- `electron_temperature_volume_average_keV`
  - `custom_ad = 1.237538e-02`
  - `fd = 1.237539e-02`
- `total_pressure_volume_average`
  - `custom_ad = 7.939232e+00`
  - `fd = 7.939232e+00`
- `alpha_power_volume_average_mw_m3`
  - `custom_ad = 2.876068e-01`
  - `fd = 2.876131e-01`

Checkpoint `115`, frozen central FD, `fd_step = 1.263000e-07`

- `softmax_Er`
  - `custom_ad = 4.086843e+01`
  - `fd = 4.087382e+01`
- `smooth_root_proxy`
  - `custom_ad = -8.501898e-05`
  - `fd = -8.500619e-05`
- `Er2_volume_average`
  - `custom_ad = 9.661756e+01`
  - `fd = 9.662527e+01`
- `Er_volume_average`
  - `custom_ad = -4.321700e+00`
  - `fd = -4.321650e+00`
- `electron_temperature_volume_average_keV`
  - `custom_ad = 1.300916e-02`
  - `fd = 1.300917e-02`
- `total_pressure_volume_average`
  - `custom_ad = 7.942294e+00`
  - `fd = 7.942294e+00`
- `alpha_power_volume_average_mw_m3`
  - `custom_ad = 2.845732e-01`
  - `fd = 2.845785e-01`

Checkpoint `115`, frozen stencil check, same `fd_step = 1.263000e-07`

- `softmax_Er`
  - `custom_ad = 4.086843e+01`
  - `fd_center = 4.087382e+01`
  - `fd_five_point = 4.087127e+01`
- `smooth_root_proxy`
  - `custom_ad = -8.501898e-05`
  - `fd_center = -8.500619e-05`
  - `fd_five_point = -8.500043e-05`
- `Er2_volume_average`
  - `custom_ad = 9.661756e+01`
  - `fd_center = 9.662527e+01`
  - `fd_five_point = 9.662307e+01`
- `Er_volume_average`
  - `custom_ad = -4.321700e+00`
  - `fd_center = -4.321650e+00`
  - `fd_five_point = -4.321778e+00`
- `electron_temperature_volume_average_keV`
  - `custom_ad = 1.300916e-02`
  - `fd_center = 1.300917e-02`
  - `fd_five_point = 1.300916e-02`
- `total_pressure_volume_average`
  - `custom_ad = 7.942294e+00`
  - `fd_center = 7.942294e+00`
  - `fd_five_point = 7.942294e+00`
- `alpha_power_volume_average_mw_m3`
  - `custom_ad = 2.845732e-01`
  - `fd_center = 2.845785e-01`
  - `fd_five_point = 2.845740e-01`

Adaptive custom AD vs frozen central FD, parameter `T0`,
`fd_step = 5.340000e-07`, replay mode `attempt`

- `softmax_Er`
  - `ad = -2.160399e+01`
  - `fd = -2.161529e+01`
  - `abs_err = 1.129999e-02`
  - `rel_err = 5.227777e-04`
- `smooth_root_proxy`
  - `ad = 2.070900e-05`
  - `fd = 2.073464e-05`
  - `abs_err = 2.563682e-08`
  - `rel_err = 1.236424e-03`
- `Er2_volume_average`
  - `ad = -2.765750e+01`
  - `fd = -2.767012e+01`
  - `abs_err = 1.262507e-02`
  - `rel_err = 4.562707e-04`
- `Er_volume_average`
  - `ad = 2.291385e+00`
  - `fd = 2.291084e+00`
  - `abs_err = 3.007156e-04`
  - `rel_err = 1.312547e-04`
- `electron_temperature_volume_average_keV`
  - `ad = 3.571291e-01`
  - `fd = 3.571291e-01`
  - `abs_err = 5.455578e-08`
  - `rel_err = 1.527621e-07`
- `total_pressure_volume_average`
  - `ad = 1.835267e+00`
  - `fd = 1.835267e+00`
  - `abs_err = 2.165264e-07`
  - `rel_err = 1.179809e-07`
- `alpha_power_volume_average_mw_m3`
  - `ad = 7.221955e-02`
  - `fd = 7.220444e-02`
  - `abs_err = 1.510863e-05`
  - `rel_err = 2.092480e-04`

VMEC/JAX AD-path notes for next session

- The promoted `vmec_jax` optimization derivative path is the accepted-point
  exact replay / discrete-adjoint path, not the `implicit.py` helper lane.
- The usual primal solve path is still the standard VMEC solve; the AD path
  builds an accepted-point replay tape around the converged solve.
- There are two important exact-tape variants inside that accepted-point path:
  - forward-only `jvp_only` exact tape
  - full forward+reverse exact tape
- `jacobian_fun(...)` can use the lighter forward-only `jvp_only` tape.
- `residual_linear_operator(...)` builds a full exact tape because it must
  support both:
  - `J v` via `matvec`
  - `J^T w` via `rmatvec`
- Therefore matrix-free is not automatically lighter in memory than dense
  forward Jacobian formation. On the hi-res deck, the bidirectional
  `residual_linear_operator(...)` can be worse than `jacobian_fun(...)`.

Likely source of the VMEC OOM

- The small number of output observables is not the cause.
- The main memory driver is the accepted-point exact tape / replay payload.
- A useful mental model is:
  - memory ~ (accepted steps to convergence) x (payload per accepted step)
- Payload per step grows with:
  - hi-res VMEC state size
  - whether basepoint carries are stored
  - whether the tape must support reverse mode
  - replay/preconditioner/control-state payloads
- So yes, more accepted steps to convergence is one important driver, but it is
  not the only one.

What carried information seems necessary vs shrinkable

- Some accepted-step replay information is necessary for the exact replay path.
- The repo already distinguishes:
  - lighter forward-only tapes
  - heavier reverse-capable tapes
- Not every carried field appears mathematically fundamental. The local
  `vmec_jax` notes/code suggest some of the retained payload is a practical
  replay choice and could be reduced further.
- The clearest already-supported shrink is:
  - use the forward-only `jvp_only` exact tape when only forward derivatives are
    needed
- Reverse mode cannot use that lightest tape; it needs a fuller replay payload.

Performance/reuse takeaways

- The clearest improvements already identified in local `vmec_jax` docs/code:
  - use forward-only exact tape for forward validation
  - keep GPU `jvp_only` with basepoint carries enabled
  - reuse one exact optimizer / one exact linearization object inside a
    benchmark run
  - reuse exact residual/state caches when possible
  - reuse cached initial-state tangent information
- The exact `vmec_jax` path already has cache/reuse machinery for:
  - exact state
  - exact residual
  - exact Jacobian
  - initial tangent columns
  - replay scan runners

Comparison against the transport custom replay

- The transport custom rule is conceptually similar:
  - primal adaptive solve first
  - then derivative replay on the realized accepted-step path
- The important difference is scale and payload:
  - transport replay stores a much smaller accepted-step carry/controller/cache
    structure
  - VMEC accepted-point replay stores a much heavier per-step equilibrium /
    preconditioner / replay payload
- So both approaches share the same philosophy:
  - reuse the realized primal path
  - differentiate the replay map rather than naive AD through the whole live
    loop
- But the VMEC tape is much heavier, which is why its OOM/performance issues are
  much sharper than in the transport custom rule.

Follow-up investigation to keep for next session

- Investigate the `iota_mean` mismatch in the VMEC geometry benchmark:
  - forward exact vs FD was poor
  - reverse exact vs forward exact was also poor
- Likely causes to test:
  - forward FD primal solve path and exact accepted-point solve path are not the
    same wrapper / not linearized around the same state
  - forward and reverse exact checks were built from separate optimizer /
    linear-operator instances instead of one shared exact linearization object
  - `iota_mean` is a sensitive postprocessed quantity (`equilibrium_iota_profiles_from_state(...)`
    then averaging), so small equilibrium mismatches may show up there first
- First clean check:
  - build one exact optimizer and one linear operator
  - use the same object for both forward `matvec` and reverse `rmatvec`
  - compare primal basepoint observable values from the exact path vs the FD
    forward-lane solve

Missing profile-AD architecture points to preserve

- The profile-parameter transport AD should not be hard-wired to the current
  benchmark metric vector.
- The correct long-term abstraction is:
  - `state_final = F(parameters)`
  - `objective = g(state_final)`
- So the custom AD rule should expose a general API at the `... -> state_final`
  level, for both:
  - forward mode
  - reverse mode
- Benchmark objectives such as `softmax_Er`, `smooth_root_proxy`, and volume
  averages should sit outside that generic `state_final` AD boundary.

- Initial `Er` construction from ambipolarity should eventually be included in
  the differentiated profile-parameter map.
- In other words, `n0`, `T0`, profile powers, and related parameters should be
  allowed to influence:
  - initial density/pressure
  - and the initial ambipolar `Er`
- Current benchmark Option A intentionally freezes the initial ambipolar `Er`
  for tractable debugging, but this is not the desired final semantic target.

- The `parameters -> carry0` map should become a more general API rather than a
  benchmark-specific helper.
- It should work for:
  - more profile parameters
  - more general parameter families
  - and eventually non-profile controls as well
- The current mixed reverse implementation uses forward-mode JVP columns for
  `d(carry0)/dp` only because the active benchmark currently has very few
  parameters.
- The desired long-term reverse architecture should expose a true reverse-mode
  treatment for `parameters -> carry0` as well, instead of depending on
  forward-mode columns there.

Latest follow-up findings

Transport reverse replay

- The profile-vector reverse benchmark still fails in the custom VJP path, but
  the failure has been localized much better.
- First failure mode:
  - whole-scan reverse transpose through
    `_radau_replay_realized_accepted_rollout(...)`
  - produced absurd PiB-scale inferred sizes / GPU XLA transpose explosion
- First structural fix:
  - reverse replay was reduced from a full replay-result map to a final-state
    only map `carry0 -> final_y`
  - this removed unnecessary `scan_outputs`, but did not fix the core scan
    transpose problem
- Second structural fix:
  - removed `jax.vjp` on the whole replay `lax.scan`
  - replaced it with a manual backward sweep that applies `jax.vjp` only to the
    single accepted-step map at each step
- This removed the catastrophic whole-scan transpose, but then exposed a new
  memory problem:
  - storing the full carry before every accepted step was too heavy
- Third structural fix:
  - switched the reverse sweep to reuse the compact per-step payloads already
    recorded in the primal adaptive rollout trace
  - this avoids saving a full carry history in the backward pass

Very important replay-fidelity conclusion

- Forward and reverse must replay the same accepted-step primal path.
- In the transport solver, the accepted-step primal really does read the cached
  Jacobian/LU reuse state as values:
  - `jacobian`
  - `cache_valid`
  - `cache_dt`
  - `cache_age`
  - `real_lu`, `real_piv`
  - `complex_lu`, `complex_piv`
- Forward JVP does *not* propagate tangents through those fields because
  `_radau_carry_with_forward_only_jvp_fields(...)` applies `stop_gradient(...)`
  to them.
- But the primal step still uses them as frozen replay values.
- Therefore reverse replay must also preserve those values in the replayed
  forward pass, even if reverse does not propagate cotangents through them.
- A temporary reverse reduction that zeroed/disabled those values was *not*
  faithful to the forward replay path and should not be treated as correct.
- The current reverse payload has been updated to carry those reuse values
  again, while still avoiding a full carry-history scan.

Remaining likely transport reverse memory targets

- `prev_stages`
- `lagged_response_cache`
- cached Jacobian/LU reuse blocks
- The main rule for future reductions:
  - do not save memory by changing the replayed primal path
  - save memory by shrinking storage around the same replay path
- Next likely step if reverse still OOMs:
  - inspect which of the compact payload blocks dominates memory
  - especially `prev_stages`, `lagged_response_cache`, and the cached
    Jacobian/LU blocks

VMEC iota mismatch: current localization

- The reverse mismatch is not spread across the whole `iotaf` profile.
- The current probes show:
  - `iotaf_q1`, `iotaf_mid`, `iotaf_q3`, `iotaf_edge`
    have excellent forward/reverse agreement
  - `iotaf_first` is bad
  - `iota_mean` is bad because it includes that first interior contribution
- Deeper probing shows the problem is already below `iotaf`, in the underlying
  half-mesh `iotas` near the axis:
  - `iotas_1` forward/reverse disagree badly
  - `iotas_2` forward/reverse disagree even worse
- Since non-RFP `iotaf_first = 0.5 * (iotas[1] + iotas[2])`, this explains the
  `iotaf_first` mismatch.

What local `vmec_jax` code/docs suggest about the iota reverse issue

- `equilibrium_iota_profiles_from_state(...)` itself is not doing anything
  exotic at `iotaf_first`; the full-mesh smoothing helper `_iotaf_from_iotas`
  uses:
  - `iotaf[0] = 1.5*iotas[1] - 0.5*iotas[2]`
  - `iotaf[1:-1] = 0.5*(iotas[1:-1] + iotas[2:])`
  - `iotaf[-1] = 1.5*iotas[-1] - 0.5*iotas[-2]`
- So the bad point is not an `iotaf`-specific axis-closure branch; it is the
  first interior average of already-bad `iotas`.
- `vmec_jax/docs/performance.rst` explicitly notes that for current-driven iota
  diagnostics, reverse `J.T v` only matches to about `1e-6` relative because
  current-driven iota still needs axis-gauge cotangent cleanup.
- `vmec_jax/optimization.py` already contains special reverse sanitization for
  current-driven iota blocks, with comments stating:
  - current-driven iota has axis/near-axis gauge-null cotangent entries
  - dense JVP columns remain finite there
  - zeroing the null reverse entries gives the matching transpose on the
    boundary-parameter subspace

NEOPAX VMEC scalar benchmark implication

- The generic NEOPAX scalar observable wrapper was originally bypassing that
  current-driven-iota-specific reverse helper path and using only the more
  generic packed-state reverse route.
- A new custom packed-state cotangent helper has been added to the NEOPAX exact
  observable wrapper for:
  - `vmec_scalar_observables`
  - `vmec_iotaf_scalar_observables`
  when `NCURR = 1`
- This helper applies targeted `nan_to_num(...)` sanitization only on the
  iota-related reverse blocks, following the same mechanism used by
  `vmec_jax`'s own workflow-specific residual builders.
- This is the main VMEC iota reverse fix currently under test.

Latest follow-up: transport reverse checkpointed reduced-state path

- The transport reverse custom-VJP path has now been reworked further:
  - the VJP forward pass no longer stores the heavyweight full adaptive rollout
    trace for reverse
  - it now records only the realized schedule/controller metadata
  - it then stores sparse checkpoint carries for reverse replay
- Current default reverse checkpoint interval:
  - `64`
- Environment override:
  - `NEOPAX_TRANSPORT_REVERSE_CHECKPOINT_INTERVAL`

What improved

- The original reverse memory blowups were reduced substantially:
  - from the earlier PiB-scale whole-scan transpose explosion
  - to large but finite OOMs
  - then to a local structural reverse assertion
- So the current transport reverse failure is no longer dominated by global
  replay memory; it is now a local reverse-structure issue.

Current local reverse diagnosis

- The remaining failure now occurs at the local step-replay pullback:
  - `pullback(replay_state_cotangent)`
- This means the reverse path is now failing because the local VJP target still
  did not define a clean cotangent space for JAX.

Reduced replay-state redesign

- To address this, the local reverse no longer tries to VJP through:
  - `full_carry -> full_carry`
- Instead it now attempts to VJP through:
  - `replay_state -> replay_state`
- Full carry reconstruction remains in the closure so the primal accepted-step
  path is unchanged.

Current replay-state cleanup sequence

- `_RadauReplayState` was introduced as the reduced local AD state.
- Fields progressively removed from `_RadauReplayState` because they are poor
  reverse-mode state outputs and should instead live in frozen replay payload:
  - `lagged_response_valid`
  - `prev_newton_iter_count`
  - `lagged_response_cache`
- These fields still affect the primal replay path, but only through frozen
  payload/carry reconstruction, not through the local reverse AD state.

Current likely remaining transport reverse issue

- If the reverse assertion still persists after these reductions, then at least
  one remaining replay-state field is still not a clean reverse-mode state
  output.
- The next debugging step in that case is to instrument the remaining
  `_RadauReplayState` leaves directly and identify which leaf still breaks the
  local pullback contract.

Latest transport reverse status for next session

- The transport reverse benchmark is still failing with:
  - `AssertionError`
  - at the local replay-state pullback:
    - `pullback(replay_state_cotangent)`
  - inside the segmented reverse replay scan

- Important interpretation:
  - this is no longer a memory-scaling failure
  - this is no longer the old full-carry reverse target problem
  - this is no longer obviously due to bool/int leaves in the reduced replay
    state
  - it is now a narrow **local reverse structural mismatch** for the
    `replay_state -> replay_state` VJP target

- Current remaining replay-state fields are approximately:
  - `t`
  - `y`
  - `dt`
  - `prev_stages`
  - `prev_dt`
  - `lagged_reference_y`
  - `prev_theta_final`

- So the likely remaining cause is now one of:
  - `prev_stages` not behaving as a clean local reverse state leaf
  - `lagged_reference_y` not behaving as a clean local reverse state leaf
  - or a remaining mismatch between the local output tangent space and the
    cotangent being fed into `pullback(...)`

- Recommended next debugging step:
  - do **leaf-by-leaf local reverse diagnostics**
  - test the local pullback with only one replay-state leaf active at a time:
    - `y`
    - `t`
    - `dt`
    - `prev_stages`
    - `prev_dt`
    - `lagged_reference_y`
    - `prev_theta_final`
  - identify exactly which leaf first breaks the local VJP contract

- Current conclusion:
  - the transport reverse issue is now highly localized
  - the next move should be instrumentation of the remaining replay-state
    leaves, not another broad structural rewrite

Latest follow-up: Boozer-based QI / Maximum-J gate

- A new geometry benchmark mode has been added:
  - `vmec_booz_qi_maxj_scalar_objectives`
- This mode is intended to test the full AD path that goes through:
  - VMEC solve
  - one shared `booz_xform`
  - Boozer-based QI objective
  - Boozer-based Maximum-J objective

Metric definitions used in this new mode

- QI objective:
  - `vmec_jax.quasi_isodynamic_residual_from_boozer_output(...)`
  - scalar objective = returned `total`
- Maximum-J objective:
  - `balloon_jax.maximum_j_residual_from_boozer_output(...)`
  - scalar objective = `ObjectiveResult.diagnostics["total"]`

Important efficiency note

- Both Boozer-based objectives are computed from one shared Boozer output.
- This is intended to be a cleaner Boozer-path forward/reverse/FD gate than
  the earlier ad hoc VMEC-line QI/Maximum-J implementation.

Current Boozer-based QI / Maximum-J results

- Benchmark command:
  - `python ./examples/benchmarks/benchmark_geometry_vmec_booz_fd_vs_ad.py --mode vmec_booz_qi_maxj_scalar_objectives --param-family RBC --param-m 1 --param-n 0 --exact-solver-device gpu`

- With default FD step (`fd_step=1.561818e-06`):
  - `qi_objective`
    - `ad=1.244365e-04`
    - `fd_center=1.834994e-02`
    - `ad_vs_center_rel_err=9.932187e-01`
    - `reverse=1.244365e-04`
    - `reverse_vs_forward_rel_err=4.933078e-10`
  - `maxj_objective`
    - `ad=5.166964e-03`
    - `fd_center=6.810710e-02`
    - `ad_vs_center_rel_err=9.241347e-01`
    - `reverse=5.166964e-03`
    - `reverse_vs_forward_rel_err=3.022056e-11`

- With tighter FD step (`fd_step=1.561818e-08`):
  - `qi_objective`
    - `fd_center=-3.283983e-04`
    - `ad_vs_center_rel_err=1.378919e+00`
    - `reverse_vs_forward_rel_err=4.420269e-11`
  - `maxj_objective`
    - `fd_center=1.093367e-02`
    - `ad_vs_center_rel_err=5.274263e-01`
    - `reverse_vs_forward_rel_err=3.751068e-11`

- With intermediate FD step (`fd_step=4.685455e-07`):
  - `qi_objective`
    - `fd_center=1.163615e-04`
    - `ad_vs_center_rel_err=6.939618e-02`
    - `reverse_vs_forward_rel_err=3.760461e-10`
  - `maxj_objective`
    - `fd_center=1.284432e-02`
    - `ad_vs_center_rel_err=5.977240e-01`
    - `reverse_vs_forward_rel_err=6.071893e-11`

- With another intermediate FD step (`fd_step=1.561818e-07`):
  - `qi_objective`
    - `fd_center=3.093686e-04`
    - `ad_vs_center_rel_err=5.977726e-01`
    - `reverse_vs_forward_rel_err=2.440547e-11`
  - `maxj_objective`
    - `fd_center=1.352614e-02`
    - `ad_vs_center_rel_err=6.180017e-01`
    - `reverse_vs_forward_rel_err=2.716202e-11`

Current interpretation

- For the Boozer-based QI / Maximum-J objectives:
  - forward exact and reverse exact match extremely well
  - this strongly suggests the exact VMEC + `booz_xform` AD path is internally
    consistent for these objectives
- FD remains step-sensitive:
  - `qi_objective` can get reasonably close at `fd_step=4.685455e-07`
  - `maxj_objective` still shows large FD-vs-AD mismatch across tested steps
- So the current evidence is:
  - forward/reverse exact consistency is good
  - FD is still not a clean truth signal for these Boozer-based objectives,
    especially for Maximum-J

Current end-to-end implication

- The magnetic / geometry side now looks much closer to being usable as the
  upstream AD block for the coupled workflow:
  - VMEC exact forward/reverse is good for the Boozer-based QI / Maximum-J
    objectives
  - so the VMEC + `booz_xform` AD path is no longer the main concern there

- This suggests the natural downstream path is:
  - geometry / magnetics
  - then `NTX`
  - then the full transport solve

- However, the full coupled reverse path is still blocked by the transport
  reverse custom-VJP problem:
  - even if the magnetic / Boozer / VMEC part is good,
  - the outer transport reverse path still has to propagate through
    `parameters -> carry0`, the adaptive accepted-step transport replay, and
    the downstream transport objectives

- Therefore:
  - the geometry / magnetics AD path is no longer the main blocker
  - the transport reverse custom-VJP remains the main blocker for a true
    end-to-end reverse-mode path through magnetics -> NTX -> transport

Latest transport reverse diagnosis update

- The leaf-by-leaf replay-state diagnostics and local output restriction have
  now ruled out the outer replay-state packaging as the primary blocker.
- Concrete result:
  - even with only the local `y` cotangent active
  - and even when the local reverse target is reduced to
    `replay_state -> next_y`
  - the segmented reverse still fails at the local pullback.

- Additional decisive test:
  - with
    `NEOPAX_TRANSPORT_REVERSE_REPLAY_LEAF=y`
    `NEOPAX_TRANSPORT_REVERSE_REPLAY_OUTPUT=y`
    `NEOPAX_TRANSPORT_REVERSE_USE_PRIMAL_STEP=1`
  - the local reverse no longer gave the vague `AssertionError`
  - instead it failed with:
    - `ValueError: Reverse-mode differentiation does not work for lax.while_loop or lax.fori_loop with dynamic start/stop values`

- Interpretation:
  - the real blocker is now clearly below the replay-state boundary
  - the raw accepted-step primal is not reverse-differentiable by generic
    `jax.vjp(...)` because it contains dynamic control flow
  - the accepted-step `custom_jvp` wrapper had been hiding that lower-level
    issue behind a less informative assertion during reverse transposition

- Current conclusion:
  - the next real fix is not another replay-state pruning step
  - it is an explicit accepted-step reverse rule at the accepted-step boundary
  - likely first for the local `carry_in -> accepted_y` or reduced accepted-step
    state map
  - rather than relying on `jax.vjp` through the raw primal step

- Practical next work item:
  - extract / centralize the accepted-step `y` map boundary so a manual
    pullback can attach there
  - then build the accepted-step reverse from the same approximate
    implicit-diff philosophy already used for the forward JVP

Latest lagged-response reverse diagnosis and plan

- Forward mode already treats lagged response explicitly rather than as one
  giant black-box object.
- In `NEOPAX/_transport_solvers.py`, the forward accepted-step tangent splits
  lagged-response handling into:
  - reuse vs rebuild of the cache
  - separate tangent propagation through `build_lagged_response(...)`
  - separate `lagged_eval_tangent` through
    `evaluate_with_lagged_response(...)`

- This means the correct reverse path must mirror the same compressed
  contract.
- The current reverse is still too generic in the lagged-response block of
  `_radau_apply_accepted_step_replay_state_pullback_linearized(...)` because
  it uses broad VJPs like:
  - `jax.vjp(_stage_evals_from_lagged, lagged_response)`
  - `jax.vjp(_build_from_flat, carry_in.y)`

- That generic lagged-response VJP appears to be the main reason the reverse
  path now blows memory back up to very large allocations once the missing
  lagged-response adjoint terms are restored.

- NTX-side inspection confirms that there is already local derivative support
  we should reuse:
  - `NTX/src/ntx/_solver_prepared.py`
    - `solve_prepared_coefficient_vector_vjp(...)`
  - `NTX/src/ntx/_solver_adjoint.py`
    - explicit adjoint helper algebra
  - `NTX/tests/test_solver.py`
    - test that the custom VJP matches direct forward value and gradient

- NEOPAX NTX lagged-response objects are already compressed:
  - `NTXInterpolatedMomentResponse`
  - `NTXPreparedCoefficientResponse`
  - `NTXExactLijLaggedResponse`
  - and `CombinedTransportLaggedResponse` at the model-composition level

- This strongly suggests the reverse should not VJP through the entire
  lagged-response object.
- Instead it should use model-aware pullbacks, especially for NTX.

Reverse redesign plan

- 1. Keep the current checkpointed exact discrete outer replay.
- 2. Keep the explicit accepted-step reverse boundary.
- 3. Replace the generic lagged-response reverse with branch-aware handling:
  - cache reused
  - cache rebuilt
- 4. For rebuild:
  - reverse `build_lagged_response(state)` in a model-aware way
  - for NTX exact mode, prefer the NTX `custom_vjp` derivative lane rather
    than raw direct reverse AD
- 5. For reuse:
  - implement reduced pullbacks for the actual cached response types
  - first target:
    - `NTXInterpolatedMomentResponse`
  - then:
    - `NTXPreparedCoefficientResponse`
    - `JVPTransportFluxResponse`
    - `CombinedTransportLaggedResponse`
- 6. Make `CombinedTransportLaggedResponse` reverse recursive by submodel
  rather than one big object VJP.
- 7. Keep validation narrow until the lagged-response reverse is model-aware:
  - one-row benchmark first
  - then compare against forward
  - only then widen to more objectives and remove debug narrowing

Implementation order

- add a lagged-response pullback dispatch helper by response type
- implement `NTXInterpolatedMomentResponse` reverse first
- implement `CombinedTransportLaggedResponse` recursive dispatch
- hook dispatch into
  `_radau_apply_accepted_step_replay_state_pullback_linearized(...)`
- then decide whether the NTX rebuild branch also needs stronger forced
  `custom_vjp` plumbing in the exact-mode setup

Architectural conclusion

- The right architecture is:
  - checkpointed exact discrete outer replay
  - plus model-aware local reverse rules for lagged-response objects
- not:
  - full generic reverse AD through the entire lagged-response cache

- This is also the closest match to how the forward accepted-step tangent
  already treats lagged response.

Latest decisive diagnostics

- Reuse-only narrowed reverse removes the giant OOM:
  - command:
    `NEOPAX_TRANSPORT_REVERSE_REPLAY_OUTPUT=y NEOPAX_TRANSPORT_REVERSE_REUSE_ONLY=1 python ./examples/benchmarks/benchmark_transport_profile_vector_ad_compare.py --ntx-exact-derivative-mode direct --ad-mode reverse --objective-indices 0`
  - result:
    - memory dropped from about `145 GiB` to about `4.23 GiB`
    - run completed
    - reverse values still absurd:
      - `n0: -8.216707e+31`
      - `T0: 5.389379e+31`
      - `density_shape_power: -9.917097e+29`
      - `temperature_shape_power: 1.692342e+32`

- Conclusion from that run:
  - the dominant memory blocker is the rebuild-branch reverse
  - the reuse-only path is not the source of the giant OOM
  - correctness is still wrong even when the reuse-only path runs

- Initial-carry leaf filter diagnostics were added in
  `examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py`:
  - env:
    `NEOPAX_TRANSPORT_REVERSE_INITIAL_CARRY_LEAF`
  - purpose:
    - filter the final benchmark contraction
      `carry0_tangent • carry0_bar`
      to one initial-carry leaf at a time

- `y`-only initial-carry contraction:
  - command:
    `NEOPAX_TRANSPORT_REVERSE_REPLAY_OUTPUT=y NEOPAX_TRANSPORT_REVERSE_REUSE_ONLY=1 NEOPAX_TRANSPORT_REVERSE_INITIAL_CARRY_LEAF=y python ./examples/benchmarks/benchmark_transport_profile_vector_ad_compare.py --ntx-exact-derivative-mode direct --ad-mode reverse --objective-indices 0`
  - result:
    - same absurd values as the full reuse-only contraction
  - conclusion:
    - the bad signal is already present in the `y` contribution

- `lagged_response_cache`-only initial-carry contraction:
  - command:
    `NEOPAX_TRANSPORT_REVERSE_REPLAY_OUTPUT=y NEOPAX_TRANSPORT_REVERSE_REUSE_ONLY=1 NEOPAX_TRANSPORT_REVERSE_INITIAL_CARRY_LEAF=lagged_response_cache python ./examples/benchmarks/benchmark_transport_profile_vector_ad_compare.py --ntx-exact-derivative-mode direct --ad-mode reverse --objective-indices 0`
  - result:
    - all reported sensitivities were exactly zero
  - conclusion:
    - the absurd reverse values are not coming from the final
      `lagged_response_cache` carry contraction

Local adjoint consistency diagnostics

- Added env:
  `NEOPAX_TRANSPORT_REVERSE_LOCAL_ADJOINT_CHECK=1`
  in `examples/benchmarks/benchmark_transport_autodiff_lagged_ntx.py`

- This diagnostic checks the local reuse-only accepted-step adjoint consistency
  against the forward approximate tangent map:
  - compares
    `⟨J v, w⟩`
    vs
    `⟨v, J^T w⟩`
  - at the local accepted-step replay-state boundary

- Run:
  - command:
    `NEOPAX_TRANSPORT_REVERSE_REPLAY_OUTPUT=y NEOPAX_TRANSPORT_REVERSE_REUSE_ONLY=1 NEOPAX_TRANSPORT_REVERSE_LOCAL_ADJOINT_CHECK=1 python ./examples/benchmarks/benchmark_transport_profile_vector_ad_compare.py --ntx-exact-derivative-mode direct --ad-mode reverse --objective-indices 0`
  - result:
    - `[autodiff-gate] local-adjoint-check lhs=1.074419e-01 rhs=1.074419e-01 abs_err=1.537659e-14`
  - conclusion:
    - the local reuse-only accepted-step `y -> accepted_y` pullback is
      internally consistent
    - the core one-step local reverse is probably not the source of the
      `1e31` scale explosion

Rollout-level adjoint diagnostic attempt

- Added env:
  - `NEOPAX_TRANSPORT_REVERSE_ROLLOUT_ADJOINT_CHECK=1`
  - optional:
    `NEOPAX_TRANSPORT_REVERSE_ROLLOUT_ADJOINT_BASIS=<int>`

- Purpose:
  - check adjoint consistency one layer outward at the full
    `carry -> final_y` realized-schedule boundary

- Result:
  - this diagnostic is currently too expensive
  - it uses `jax.jvp(_final_y_from_carry, ...)`
  - the run OOMed at about `2.84 GiB` extra allocation while building
    checkpoint carries in `_radau_replay_realized_checkpoint_carries(...)`

- Important interpretation:
  - this OOM is from the diagnostic itself, not a new solver regression
  - it means rollout-level forward-mode probing through the replayed rollout is
    too expensive for routine use here

Cheaper parameter/carry diagnostic

- Added env:
  `NEOPAX_TRANSPORT_REVERSE_PARAMETER_CARRY_DIAGNOSTIC=1`

- Purpose:
  - print cheap quantities already available in the backward pass:
    - `||carry0_bar.y||`
    - `max(abs(carry0_bar.y))`
    - `||carry0_tangent.y||` for each parameter basis
    - `max(abs(carry0_tangent.y))`
    - `vdot(carry0_tangent.y, carry0_bar.y)`
  - this should help distinguish:
    - huge rollout cotangent `carry0_bar.y`
    - vs a bad parameter-to-initial-carry tangent

Current best diagnosis

- Giant memory OOM:
  - still due to the rebuild-branch reverse being traced in the normal narrowed
    reverse
- Wrong huge gradients:
  - not due to the final `lagged_response_cache` contraction
  - not obviously due to the local accepted-step reuse-only pullback, since the
    local adjoint check passes to `1e-14`
  - most likely live one layer outward:
    - in the outer realized-schedule reverse accumulation
    - or in the parameter-to-initial-carry tangent / contraction layer

Current next recommended test

- command:
  `NEOPAX_TRANSPORT_REVERSE_REPLAY_OUTPUT=y NEOPAX_TRANSPORT_REVERSE_REUSE_ONLY=1 NEOPAX_TRANSPORT_REVERSE_LOCAL_ADJOINT_CHECK=1 NEOPAX_TRANSPORT_REVERSE_PARAMETER_CARRY_DIAGNOSTIC=1 python ./examples/benchmarks/benchmark_transport_profile_vector_ad_compare.py --ntx-exact-derivative-mode direct --ad-mode reverse --objective-indices 0`

- what it should tell us:
  - whether `carry0_bar.y` itself is already huge
  - or whether the parameter-to-initial-carry `y` tangents are what make the
    final scalar contractions blow up

Latest reverse-segment localization

- Parameter/carry diagnostic result:
  - `carry0_tangent.y` is ordinary size for all four basis parameters
  - `carry0_bar.y` is already enormous:
    - `carry0_bar_y_l2=1.463782e+33`
    - `carry0_bar_y_max=7.577226e+32`
  - conclusion:
    - the blow-up is in the rollout cotangent itself
    - not in the parameter-to-initial-carry tangent map

- Added env in `NEOPAX/_transport_solvers.py`:
  - `NEOPAX_TRANSPORT_REVERSE_SEGMENT_DIAGNOSTIC=1`
  - purpose:
    - print `||replay_state_bar.y||` and `max(abs(...))` before and after each
      reverse segment in `_radau_adaptive_final_y_realized_schedule_vjp_bwd`

- Run:
  - command:
    `NEOPAX_TRANSPORT_REVERSE_REPLAY_OUTPUT=y NEOPAX_TRANSPORT_REVERSE_REUSE_ONLY=1 NEOPAX_TRANSPORT_REVERSE_SEGMENT_DIAGNOSTIC=1 python ./examples/benchmarks/benchmark_transport_profile_vector_ad_compare.py --ntx-exact-derivative-mode direct --ad-mode reverse --objective-indices 0`

- Decisive result:
  - for segments `312` down through `3`, the replay-state `y` cotangent stays
    exactly at:
    - `l2 = 1.0`
    - `max = 1.0`
  - first blow-up occurs only at the initial segments:
    - `seg=2`:
      - after: `l2=5.638296e+21`, `max=2.882411e+21`
    - `seg=1`:
      - after: `l2=4.333196e+26`, `max=1.972291e+26`
    - `seg=0`:
      - after: `l2=1.463782e+33`, `max=7.577226e+32`

- Strong updated conclusion:
  - the instability is highly localized
  - it is not a gradual accumulation across the whole replay
  - it enters in the first few reverse segments near the initial carry
  - next debugging should focus specifically on segment `0-2`, especially:
    - payload reconstruction there
    - `_radau_replay_realized_accepted_carry_pullback(...)`
    - the way replay-state bars are threaded across those earliest segments

Most important current result

- Run:

  `NEOPAX_TRANSPORT_REVERSE_REPLAY_OUTPUT=y NEOPAX_TRANSPORT_REVERSE_REUSE_ONLY=1 NEOPAX_TRANSPORT_REVERSE_STEP_PULLBACK_CHECK=1 python ./examples/benchmarks/benchmark_transport_profile_vector_ad_compare.py --ntx-exact-derivative-mode direct --ad-mode reverse --objective-indices 0`

- Decisive result at the targeted problematic replay step:
  - `step-pullback-check seg=1 step=10 lhs=1.843063e+23 rhs=1.577166e+22 abs_err=1.685347e+23`

- Interpretation:
  - the remaining bug is in the local replay-step pullback itself at later carries
  - it is not just outer replay composition
  - the earlier initial-carry local adjoint check was too narrow and does not validate the problematic early replay steps
  - the stage-solve transpose fix was real and should stay, but it was not the final bug

Most important next steps

1. Treat the issue as a local replay-step pullback mismatch.
   - Main target:
     - `_radau_apply_accepted_step_replay_state_pullback_linearized(...)`
   - Proven failing site:
     - `seg=1 step=10`

2. Stop adding broad diagnostics unless they directly validate a patch.
   - The current step-level adjoint check is enough.
   - Use it as the main correctness probe.

3. Refactor the local replay-step reverse to mirror the forward tangent helper more literally.
   - Forward source of truth:
     - `_radau_accepted_step_y_tangent_from_primal_linearized(...)`
   - Reverse should be derived from that same reduced map, not maintained as separate handwritten algebra that can drift.

4. Split the local reverse into explicit submaps.
   - output projection pullback
   - accepted-`y` accumulation pullback
   - stage linear solve transpose
   - Jacobian/time-source accumulation pullback
   - lagged-response contribution pullback

5. Replace the current handwritten `dy_bar` / `dh_bar` assembly with the transpose of the same reduced forward tangent map.
   - The now-proven mismatch site is the local algebra after `stage_rhs_bar`.

6. Keep the reuse-only narrowed test while fixing correctness.
   - Continue using:
     - `NEOPAX_TRANSPORT_REVERSE_REPLAY_OUTPUT=y`
     - `NEOPAX_TRANSPORT_REVERSE_REUSE_ONLY=1`
     - `--objective-indices 0`

7. Primary validation after each patch:
   - rerun the same `STEP_PULLBACK_CHECK=1` command
   - success criterion:
     - `lhs` and `rhs` agree closely for `seg=1 step=10`

8. After the local replay-step pullback matches, re-check segment growth.
   - rerun:
     - `NEOPAX_TRANSPORT_REVERSE_SEGMENT_DIAGNOSTIC=1`
   - expectation:
     - the early-segment `y_bar` explosion should disappear or drop sharply

9. Only after reuse-only correctness is fixed, return to rebuild OOM work.
   - rebuild branch is still the memory blocker for the full reverse path
   - but it is not the current correctness blocker
