# Theta Solver Update

This note describes a practical update path for the `theta` / `theta_newton`
solver in NEOPAX based on a simple idea:

- reuse the **well-behaved structural components** of the custom Radau solver
- make the theta solver **look as much like Radau as possible at the solver-architecture level**
- do **not** blindly copy Radau-specific stage machinery
- keep `lagged_response` as a forward-solve acceleration mechanism
- build a cleaner, more robust implicit theta solver around the same
  controller-quality and Newton-quality lessons

The goal is not to turn theta into a fake collocation method. The goal is:

- make theta and Radau use as similar a **solver boundary / controller /
  diagnostics / reuse architecture** as possible
- run the same kinds of NTX lagged-response cases through both solvers
- and later let theta inherit a very similar AD strategy once the Radau AD path
  is correct

So the design target is:

- **maximum architectural similarity**
- **minimum mathematical distortion**

In other words:

- copy the Radau *solver logic shape* as far as it makes sense
- only diverge where theta's one-stage implicit mathematics truly requires it

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
- preserve the same high-level solver object boundaries
- preserve the same controller and diagnostic language
- preserve the same lagged-response and reuse concepts
- replace only the Radau-specific mathematical internals with theta-specific
  analogues

## Compatibility goal

The compatibility goal should be explicit:

- we want theta and Radau to be testable on the same families of transport
  problems
- especially the same NTX lagged-response configurations
- and we want later AD work on theta to follow the same overall pattern as the
  final Radau AD path

That means theta should try to match Radau in:

- accepted-step attempt structure
- accepted / rejected controller flow
- Newton-quality summaries
- reuse-state bookkeeping
- lagged-response construction / reuse policy
- saved diagnostics

and should differ mainly in:

- local implicit residual math
- local linear solve structure
- local error estimator

This is important because it means future comparisons are easier:

- same benchmark cases
- same runtime diagnostics
- same controller interpretation
- same lagged-response modes
- and later, ideally, the same AD boundary design

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

This accepted-step boundary should be made as parallel as possible to the
Radau accepted-step boundary:

- carry in
- one attempted implicit solve
- accepted-step result object
- updated carry out

The internals will differ, but the outer shape should match closely.

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

The stronger design requirement is:

- the theta accepted-step boundary should be intentionally shaped to resemble
  the Radau one, so that later replay / AD / diagnostics work can follow the
  same pattern with minimal conceptual branching

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

## Current implementation progress

The first theta-local implementation slice has now been started in code.

Implemented so far:

- `theta` / `theta_newton` step diagnostics now expose much more of the same
  operational information style used by Radau:
  - convergence flag
  - residual-like error norm
  - Newton iteration count
  - final residual norm
  - final correction norm
  - `theta_final`-style contraction proxy
  - slow-contraction / residual-blowup / nonfinite flags

- `theta_newton` now reuses some expensive per-attempt data across retry
  reductions:
  - `f_old` is built once per attempted step
  - transport `lagged_response` is built once per attempted step and reused
    through reduced-`dt` retries

- `theta_newton` now carries a more Radau-like controller memory state:
  - `prev_error`
  - `prev_dt`
  - `recent_reject_count`
  - `regrowth_cooldown`
  - `easy_growth_streak`
  - `prev_theta_final`
  - `prev_newton_iter_count`

- a theta-local controller helper was added that reuses the same **style** of
  logic as the better-behaved Radau controller:
  - current / Gustafsson / Hairer-lean / NTSS-like growth modes
  - reject-history-aware shrink logic
  - difficulty-aware growth caps using Newton quality and contraction

- theta now has a first-class attempted-step result object,
  `_ThetaAcceptedStepAttemptResult`, so both `theta` and `theta_newton`
  organize one attempted implicit solve in a much more Radau-like way:
  - attempt inputs
  - one attempted solve
  - structured attempt result
  - step-info projection used for saved diagnostics

- both theta backends now flow through a shared
  `_theta_step_info_from_attempt(...)` helper instead of assembling saved
  diagnostics ad hoc from loose tuples

- theta now also has a first-class attempted-step input / reuse wrapper,
  `_ThetaAttemptContext`, so the data needed for one attempted solve is
  packaged explicitly rather than being reconstructed implicitly inside nested
  local functions

- both theta backends now build attempted steps through a shared
  `_theta_make_attempt_context(...)` helper, and `theta_newton` reuses the
  same attempt context across reduced-`dt` retries via
  `_theta_attempt_context_with_dt(...)`

- this makes theta more Radau-like not just at the result boundary, but also
  at the attempt-input boundary:
  - attempt context in
  - one attempted implicit solve
  - structured attempt result out

- theta now also has a shared state-transition helper,
  `_theta_step_transition_from_attempt(...)`, so both `theta` and
  `theta_newton` apply accepted / rejected step outcomes through one common
  carry-update path rather than each backend hand-assembling status and carry
  updates locally

- that means the forward theta architecture is now closer to the Radau shape
  at three distinct boundaries:
  - attempt context construction
  - attempted-step result construction
  - accepted / rejected carry transition

- `theta_newton` now exposes an explicit `theta_controller_mode` setting
  through solver parameters so it can be run with controller families that
  intentionally mirror the Radau-side controller vocabulary

- `theta_newton` now also exposes an explicit
  `theta_jacobian_reuse_mode` setting so forward benchmarks can compare:
  - `refresh_each_iteration`
  - `freeze_attempt`

- `theta_newton` now also exposes Radau-style reuse-control options directly:
  - `theta_jacobian_reuse_rtol`
  - `theta_max_jacobian_age`
  - `theta_lagged_response_reuse_mode`
  - `theta_lagged_response_reuse_rtol`
  - `theta_lagged_response_reuse_atol`

- this gives theta a first real, explicit Jacobian/LU reuse policy knob in the
  same architectural spirit as Radau, while still keeping the local one-stage
  Newton mathematics simple

- the generic custom-solver saved output now also exposes
  `last_attempt_jacobian_reused`, and theta populates it explicitly from the
  chosen linearization-reuse behavior

- the generic custom-solver saved output now also exposes
  `last_attempt_lagged_reused`, and theta populates it explicitly when a
  lagged transport response object was built and reused through the attempted
  solve / retry path

- that means forward benchmarks can now distinguish:
  - whether theta converged
  - how difficult Newton was
  - whether the last attempt ran under reused-vs-refreshed linearization
  - whether the last attempt used lagged transport-response reuse

- theta now also carries a first-class `_ThetaReuseState` inside the step
  state, and that reuse metadata is exposed at the end of a solve through
  `final_reuse_state`

- this is intentionally a lighter-weight forward-only analogue of the more
  mature Radau reuse carry, but it has now become substantially closer:
  - lagged-response cache object
  - lagged-response validity flag
  - lagged-response reference state
  - Jacobian cache
  - cache-validity flag
  - cache `dt`
  - cache age
  - LU factorization cache
  - pivot cache
  - whether lagged response was actually reused on the last attempt
  - whether the last attempt actually reused its Jacobian/LU linearization
  - and the `dt` associated with that most recent linearization choice

- `theta_newton` now also uses Radau-like reuse decision logic in the forward
  path:
  - lagged-response reuse can be controlled with `retry_only` or
    `global_state_drift`
  - global drift reuse uses the same normalized state-drift metric concept
  - Jacobian/LU reuse now respects cache-validity, `dt` closeness, and cache
    age limits in the same overall style as Radau

- the current forward-path architecture is therefore much closer to the Radau
  shape in the places that matter for future benchmarking:
  - accepted-step attempt boundary
  - controller memory
  - Newton-quality summaries
  - retry-aware lagged-response reuse
  - explicit linearization reuse policy
  - saved reuse diagnostics
  - first-class reuse-state carry
  - saved per-step diagnostics

This is intentionally **not** a copy of the Radau stage method.
It is reuse of:

- controller shell ideas
- Newton-quality heuristics
- stateful retry / regrowth behavior

while keeping the theta solver mathematically one-stage and much simpler.

So the implementation direction is already aligned with the intended goal:

- make theta operationally resemble Radau as much as possible
- keep the only real differences at the local implicit-step mathematics level

## Next practical step

The next useful theta-local step is:

- start using the richer theta reuse-state more actively in controller logic,
  rather than only in reuse execution and final diagnostics

- after that, the next strong forward-only target is to make Jacobian / LU
  state itself more visible in saved diagnostics and controller decisions when
  that proves helpful for benchmarking

That would further align:

- debugging
- replayability
- eventual future AD design
- benchmark comparability against the same NTX lagged-response cases

without coupling the theta work to the current Radau AD investigation.

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
