# Theta Solver Update

This note describes a practical update path for the `theta` / `theta_newton`
solver in NEOPAX based on a simple idea:

- reuse the **well-behaved structural components** of the custom Radau solver
- do **not** blindly copy Radau-specific stage machinery
- keep `lagged_response` as a forward-solve acceleration mechanism
- build a cleaner, more robust implicit theta solver around the same
  controller-quality and Newton-quality lessons

The goal is not to turn theta into Radau. The goal is to transfer the parts of
the Radau implementation that are conceptually solver-agnostic and have
already shown themselves to be useful.

## Main idea

The custom Radau solver has accumulated several useful ingredients:

- robust Newton diagnostics
- better timestep-controller structure
- cache / reuse logic
- lagged-response construction and reuse hooks
- cleaner separation between:
  - primal solve state
  - controller state
  - forward-only reuse state

Many of these ideas are not inherently Radau-specific.

The parts that **are** Radau-specific are:

- multi-stage collocation structure
- stage predictors based on `prev_stages`
- transformed stage-space linear solves
- embedded Radau error estimators

So the right update strategy for theta is:

- reuse the generic good ideas
- replace the Radau-specific internals with theta-specific analogues

## What should be reused from Radau

### 1. Timestep controller structure

The outer adaptive-step controller design can largely be reused.

Useful parts:

- accepted / rejected step logic
- safety-factor based growth / shrink rules
- min / max step-growth caps
- cooldown after recent rejection
- scalar-history logic
- explicit controller modes

This is valuable because theta also needs:

- consistent accepted-step adaptation
- predictable recovery after difficult regions
- less heuristic and less ad hoc step-size evolution

The exact growth law may differ from Radau, but the **structure** is reusable.

### 2. Newton-quality diagnostics

The theta solver should reuse most of the Newton-quality diagnostics that were
added for Radau.

These include:

- contraction monitoring
- residual blow-up detection
- nonfinite detection
- Newton-iteration count tracking
- scaled convergence metrics
- shrink suggestions when Newton quality is poor

These ideas are highly transferable because theta also solves an implicit
nonlinear system every accepted step.

In practice this means theta should expose and use diagnostics like:

- final residual norm
- final correction norm
- Newton iteration count
- nonfinite / divergence flags
- slow-contraction flags

Those metrics can then feed both:

- debugging
- adaptive controller decisions

### 3. Jacobian / factorization reuse policy

The Radau solver already contains useful logic around:

- Jacobian reuse
- reuse validity checks
- cache age / cache dt tracking

Theta should adopt the same general pattern.

For theta, the natural reusable objects are:

- Jacobian of the implicit residual
- LU factorization of the theta step linearization
- metadata for when reuse is still acceptable

This helps:

- runtime
- stability
- repeated attempts on nearby `dt`

### 4. Forward-only reuse state design

One of the most important lessons from the AD work is that not every piece of
solver state should be treated as a primary tangent object.

That design principle should be built into theta from the start.

In theta, the following should likely be treated as forward-useful but not
necessarily first-class differentiated state:

- Jacobian cache
- LU cache
- retry bookkeeping
- controller-history scalars
- lagged-response reuse metadata

This separation is useful even before any custom AD is implemented because it
gives a cleaner solver architecture.

### 5. Diagnostic structure

The Radau work showed the value of compact but informative diagnostics.

Theta should adopt the same philosophy:

- keep per-step summaries
- avoid giant dumps by default
- make localized debugging easy

Useful fields include:

- accepted / rejected
- attempted `dt`
- next `dt`
- residual norm
- correction norm
- Newton iteration count
- reuse flags
- controller flags

This will make future tuning much faster.

## What should not be copied directly from Radau

### 1. Stage predictor machinery

Theta is not a collocation method with multiple internal stages.

So Radau-specific items like:

- `prev_stages`
- stage-history replay
- collocation-stage predictors
- transformed stage-space solves

should not be copied into theta.

Instead, theta should have its own simpler predictor concept.

### 2. Radau embedded estimator formulas

Radau has method-specific error-estimation structure tied to:

- stage combinations
- embedded formulas
- collocation residual structure

Theta needs its own error estimator.

That estimator should be chosen because it makes sense for theta, not because
it resembles Radau.

### 3. Radau-specific complex block linear algebra

The real/complex block factorization machinery is specific to the transformed
Radau stage system.

Theta should remain much simpler:

- one implicit step residual
- one linearization per Newton iteration or per reuse window
- one factorization path

## What theta should use instead

### 1. A theta-specific predictor

Theta should use a simpler predictor for the next nonlinear solve, such as:

- explicit Euler predictor
- extrapolation from previous accepted increment
- previous Newton correction based predictor
- previous accepted-step slope predictor

These are the natural theta analogues of “good initial guess” logic.

The predictor should be:

- cheap
- stable
- easy to reason about

### 2. A theta-specific error estimator

Good candidates include:

- step-doubling
- defect / residual-based estimator
- embedded comparison against a cheaper predictor
- comparison between one full theta step and two half theta steps

This is one of the main places where theta-specific design is required.

The most important architectural point is:

- the **controller shell** can be shared in spirit
- the **local error signal** should be theta-native

### 3. A theta-specific accepted-step map

Conceptually, theta should also be organized around an accepted-step map:

```text
Y_{n+1} = Phi_theta(Y_n, h, params)
```

This is useful even if we are not immediately implementing custom AD.

It helps because it separates:

- implicit step physics
- Newton solve
- controller evolution
- retry logic

That same accepted-step boundary later becomes the natural place for:

- custom JVP / VJP
- replay diagnostics
- checkpointing

## Role of lagged response in theta

Yes, the same general `lagged_response` idea makes sense for theta.

The principle is unchanged:

- `lagged_response` is a **forward-solve acceleration mechanism**
- it is not the entire differentiation strategy by itself

For theta, the lagged-response path should reuse the same conceptual interface:

- `build_lagged_response(state)`
- `evaluate_with_lagged_response(state, lagged_response)`

That means theta can benefit from:

- freezing expensive local transport-response components over one attempt
- reducing repeated expensive transport evaluations during Newton iterations
- retry-aware or drift-aware reuse of lagged response objects

### Recommended lagged-response policy for theta

Theta should reuse the same high-level options already explored around Radau:

- `black_box`
- `lagged_response`

and likely the same reuse-policy ideas:

- retry-only reuse
- global-state-drift reuse
- tolerances for deciding when a lagged response is still close enough

This is valuable because the expensive transport physics is not method-specific.
If lagging helps the forward Radau solve, it can also help the forward theta
solve.

### What should remain forward-only

Even if theta later gets custom AD, the following should initially be treated
as forward-only or low-priority tangent channels:

- lagged-response reuse flags
- controller-history bookkeeping
- Jacobian/LU reuse metadata
- retry counters

This mirrors the main lesson from the Radau AD work:

- the differentiated mathematical object should stay small and meaningful
- the forward-acceleration bookkeeping should not dominate the AD path

## Suggested architecture for theta update

The updated theta solver should be organized around the following layers.

### Layer 1: Theta implicit residual

Define a clean step residual:

```text
R_theta(Y_{n+1}; Y_n, h, params) = 0
```

This is the core mathematical object.

### Layer 2: Newton solve wrapper

Build a clean Newton solve around that residual with:

- reusable Jacobian / LU policy
- robust diagnostics
- divergence detection
- final-quality summaries

### Layer 3: Accepted-step map

Package one successful theta step as:

- primal next state
- error estimate
- Newton diagnostics
- reuse outputs

This should be the theta equivalent of the accepted-step boundary that has been
so useful in the Radau work.

### Layer 4: Controller wrapper

Use a controller shell inspired by the better-behaved Radau controller logic:

- accepted / rejected update
- growth / shrink with caps
- optional history-aware modes
- Newton-quality-informed shrink

### Layer 5: Lagged-response integration

Integrate:

- lagged construction
- reuse checks
- drift/retry policy

without coupling that too tightly to the differentiated mathematical object.

## Proposed implementation priorities

### Priority 1. Clean theta accepted-step structure

Refactor theta around a first-class accepted-step map:

- input carry
- one attempted theta solve
- accepted-step result
- updated carry

This matters more than clever controller tuning at first.

### Priority 2. Reuse Newton diagnostics and reuse policy

Bring over the good Radau-side ideas for:

- residual/correction metrics
- divergence flags
- cache reuse checks

This should improve both robustness and observability quickly.

### Priority 3. Add lagged-response support in the same style

Use the existing lagged-response constructor/reuse philosophy for theta:

- build lagged object from current state
- evaluate expensive pieces through that object during Newton iterations
- control reuse with explicit policy/tolerances

### Priority 4. Add a theta-native error estimator

Do not borrow the Radau estimator formula.
Choose a theta-appropriate estimator and connect it to the shared controller
shell.

### Priority 5. Only later consider theta custom AD

If optimization through theta becomes important, then the natural future step
is:

- custom AD at the theta accepted-step boundary

But this should come **after** the forward solver structure is cleaned up.

## Does this make sense?

Yes, it makes good sense.

In fact, this is probably the right design attitude:

- reuse the mature, solver-agnostic parts of the Radau work
- keep theta simpler where theta should be simpler
- share forward-acceleration ideas like lagged construction and reuse
- do not inherit Radau-specific stage complexity unnecessarily

So the central principle is:

- **share architecture and tactics**
- **do not force method-specific internals across methods**

## Short practical summary

For theta, we should:

- reuse the controller framework style
- reuse Newton robustness tactics
- reuse Jacobian/LU reuse logic
- reuse lagged-response construction and reuse ideas
- keep controller/cache/reuse state conceptually separate from the main
  differentiated mathematical object

But we should **not**:

- copy Radau stage predictors
- copy Radau stage-history state
- copy Radau embedded estimators
- copy Radau transformed stage linear algebra

Theta should get its own:

- predictor
- error estimator
- accepted-step residual formulation

while still benefiting from the good engineering structure already developed on
the Radau side.
