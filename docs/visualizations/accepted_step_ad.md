# Cooling-water animation: accepted-step AD and rejected trials

Open [accepted_step_ad.html](accepted_step_ad.html) in a browser. It is a single
offline HTML file with play/pause, frame stepping, a scrubber, adjustable heat
loss and solver settings, and two AD graphs with the same derivative contract. It needs no installation
or network connection and runs no NEOPAX, VMEC, NTX or transport calculations.

The physical system is Newton cooling: a cup starts at 80 degrees C in a
20-degree room, `T' = -k*(T-20)`. The objective is temperature after 10 minutes;
the parameter is the positive heat-loss rate k in inverse minutes. The exact
reference is `T(10)=20+60*exp(-10*k)` and `dT(10)/dk=-600*exp(-10*k)`.

The educational solver uses implicit Euler, `y_next=y/(1+k*h)` for the excess
temperature `y=T-20`. The absolute difference from two half steps estimates the
local error. A trial is accepted if that error is at most the selected tolerance.
The next proposed timestep is `h*clip(0.9*sqrt(tolerance/error), 0.2, 2)`; the last
trial is clipped to the remaining time. Rejected trials leave physical time and
the accepted temperature unchanged.

Both AD graphs freeze the recorded timesteps, controller and branch decisions.
The accepted-only graph replays only accepted updates. The all-attempts graph
computes each attempted update, then selects
`y_next = where(accepted, y_trial, y_old)`. A rejection's state derivative is the
identity, and the cotangent into its trial computation is zero. With finite
intermediate derivatives, both gradients agree. The comparison is deliberately
not AD through the timestep controller.

Contributions from both graphs are computed by a small reverse-mode AD tape, not
fabricated arrows. The trial table attributes direct parameter cotangents to
operations in each trial. Toy tape-node counts illustrate the larger attempted
graph; they are not measurements of Diffrax's compiled computation or memory.
The page independently displays centered finite differences with the same frozen
accepted schedule. The continuous reference is a separate quantity, not the exact
derivative of a finite-resolution numerical solver.

In the default example there are seven accepted steps and three rejections. Both
graphs give `dT(10)/dk = -80.35145343471771`, with the same numerical final
temperature `30.39372891221125` degrees C. The exact continuous derivative is
`-81.20116994196762`. Each rejected trial contributes exactly zero in both graphs.

## Diffrax connection

Inspected on 2026-10-30: Diffrax's default
[`RecursiveCheckpointAdjoint`](https://docs.kidger.site/diffrax/api/adjoints/)
uses a checkpointed attempt loop. Its
[`_integrate.py`](https://github.com/patrick-kidger/diffrax/blob/main/diffrax/_integrate.py)
selects the old physical state and old solver state when a trial is rejected.
Its [`PIDController`](https://github.com/patrick-kidger/diffrax/blob/main/diffrax/_step_size_controller/pid.py)
uses stop-gradient on the inverse scaled error and timestep adaptation factor.
This describes the default discretize-then-optimize/PID path, not every Diffrax
adjoint or custom controller.

Therefore including rejected attempts in the loop need not alter the physical
gradient. It may still involve additional computations, and nonfinite local
derivatives can cause a numerical zero-times-NaN/inf problem even when the final
forward state is finite. Any auxiliary state that actually survives rejection
and influences accepted steps must also be accounted for. These are distinct
implementation questions, not evidence that rejected steps inherently give bad
gradients.

The historical NEOPAX attempt replay did not satisfy the toy's unchanged-memory
assumption: before commit `0fed61d` it replaced transport/reference caches with
stopped-gradient rejected-attempt outputs. See the
[Kvaerno5 comparison and historical source audit](kvaerno5_vs_neopax_attempt_map.md).
The real Kvaerno5 scalar check has seven accepted steps and two rejections, with
identical adaptive and accepted-time replay gradients. This does not reproduce
the old NEOPAX cache behavior or establish the cause of its reported large
gradients.

This is not a Radau or NEOPAX performance/correctness benchmark. In particular,
the toy method has no Newton iteration, cached Jacobian or database. NEOPAX
replay must reconstruct the correct converged internal stages; identical Newton
iteration counts or cached factors are not required for an accurate implicit
stage derivative. See the scalar audit in
[combined_geometry_profile_optimization_issues.md](../combined_geometry_profile_optimization_issues.md#replay-invariance-and-incomplete-cache-reconstruction-audit-2026-10-17).

The numerical core is in the HTML's `ad-core` script and is DOM independent;
`globalThis.CoolingAD` exposes `simulate`, `frozenValue` and `finiteDifferences`.
The reusable checks are in `tests/check_accepted_step_ad_visualization.js` (run
with Node.js, or evaluate `checkAcceptedStepAD(htmlText)` in a JavaScript runtime).
They check 27 small parameter combinations, independent implicit-Euler gradients,
finite differences, per-trial contribution sums, rejected-state invariance, and
mock-DOM rendering of every default animation frame in both modes. No physical
solver is invoked.
