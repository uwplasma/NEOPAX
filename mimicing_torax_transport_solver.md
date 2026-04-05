# Mimicking Torax Transport Solver in NEOPAX

## Why this note
This file explains how to align NEOPAX transport solver structure with Torax-style timestep residual solving, and clarifies the statement:

"Your solver module currently mixes ODE integrator style and steady-state/root backends in one registry; Torax keeps a tighter distinction around timestep residual solvers."

## Short answer to your question
Yes, NEOPAX already has multiple solver backends beyond Diffrax.

You currently have:
- Time integration backends (Diffrax ODE stepping)
- Root-finding style backends (Newton/Broyden/Anderson)
- Optimizer style backend (Jaxopt residual minimization)

The issue is not "missing solver variety".
The issue is semantic coupling in one generic registry/interface, which makes timestep evolution and steady-state solving look interchangeable even when they have different mathematical meaning and data flow.

## What "mixes ODE and root backends in one registry" means
In `NEOPAX/_transport_solvers.py`, one shared `TransportSolver` interface and a single registration/build path are used for:
- ODE initial value problem stepping: integrates `dy/dt = f(t, y)` over `[t0, t1]`
- Algebraic solve: finds `F(y) = 0` directly

Those are different problem classes.
Keeping them in one flat selection path can blur:
- Expected inputs (`t0,t1,dt` vs `tol,maxiter`)
- Expected outputs (trajectory/solution object vs converged root state)
- How boundary conditions and prescribed profiles at `t+dt` are handled
- How to reason about coupling with implicit theta residuals

## How Torax stays tighter
Torax time-advances by timestep residual solving around a fixed per-step formulation:
1. Build state at `t+dt` with prescribed/boundary info
2. Build residual for evolved unknowns at this step
3. Solve this residual with a chosen nonlinear backend
4. Reconstruct post-step profiles and derived quantities

The nonlinear backend can vary (Newton-like or optimizer-like), but it is still inside the timestep residual framework.

## Current NEOPAX status (good progress)
NEOPAX already has many Torax-like parts:
- Modular equation operators (`density`, `temperature`, `Er`)
- Main orchestration model and solver backend builder
- Multiple nonlinear and ODE-capable backends
- Coupled `Er` in `diffusion` mode in the shared state RHS

Main architectural gap:
- A strict "timestep residual solver" path is not separated as first-class from generic ODE or steady-state solve APIs.

## Suggested refactor (minimal, practical)
1. Split solver families into explicit categories:
   - `TimeStepResidualSolver` (Torax-like primary)
   - `ODEStepper` (explicit/IMEX trajectory stepping)
   - `SteadyStateSolver` (`F(y)=0` style)
2. Add separate registries/builders for each category.
3. Keep one user-facing selector, but route through typed categories.
4. Make the default transport path use `TimeStepResidualSolver` semantics.
5. Keep ODE and steady-state modes available as advanced/alternate modes.

## Why this helps
- Clearer numerics intent per run mode
- Easier debugging and profiling
- Cleaner alignment with Torax mental model
- Less accidental misuse of a solver backend outside its natural formulation

## Direct response
"Dont I already have different solvers than Diffrax ones?"

Yes, you do.

The recommendation is about separating solver semantics and architecture, not adding more solver algorithms.
