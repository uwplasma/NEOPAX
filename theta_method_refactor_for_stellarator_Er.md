# Theta Method Refactor Guidelines for Stellarator Er

## Objective

Make the theta-method path robust for strongly stiff stellarator transport with large Er root variation by treating each step as a nonlinear residual solve and using pseudo-transient continuation.

## Core Residual Formulation

For each accepted pseudo-time step, solve:

R(y_{n+1}) = y_{n+1} - y_n - dt * ((1 - theta) * F(t_n, y_n) + theta * F(t_n + dt, y_{n+1})) = 0

Recommended default:
- theta = 1.0 (Backward Euler for maximum damping)

## Numerical Features to Enable

1. Adaptive pseudo-time continuation (PTC)
- Start from base dt
- Increase dt when Newton converges quickly
- Decrease dt and retry when Newton fails
- Clamp dt between min and max factors of base dt

2. Newton globalization via backtracking line search
- Compute Newton step from linear solve
- Backtrack with alpha *= contraction until residual decreases sufficiently
- If no acceptable alpha is found, reject the step and retry with smaller dt

3. Step retry strategy
- On non-convergence or line-search failure, reduce dt and retry
- Stop after configurable retry count with explicit runtime error

4. Newton-Krylov linear solve option
- Use matrix-free GMRES for Newton directions via Jacobian-vector products
- Keep direct dense linear solve as fallback/default

5. Trust-region step clipping
- Optionally limit Newton step norm to a trust radius before line search
- Helps avoid large overshoot when Er branch sensitivity is severe

6. Homotopy continuation per pseudo-time step
- Solve staged residuals with lambda from 0 -> 1
- Blends old-state forcing with fully implicit forcing to improve robustness

## Recommended Defaults

- theta_ptc_enabled = true
- theta_ptc_dt_min_factor = 1e-4
- theta_ptc_dt_max_factor = 1e3
- theta_ptc_growth = 1.5
- theta_ptc_shrink = 0.5
- theta_line_search_enabled = true
- theta_line_search_contraction = 0.5
- theta_line_search_min_alpha = 1e-4
- theta_line_search_c = 1e-4
- theta_max_step_retries = 8
- theta_linear_solver = "direct" (or "gmres")
- theta_gmres_tol = 1e-8
- theta_gmres_maxiter = 200
- theta_trust_region_enabled = false
- theta_trust_radius = 1.0
- theta_homotopy_steps = 1

## Why this is Appropriate for Stellarator Er

- Er closures can induce strong nonlinearity and branch-sensitive behavior.
- Fully implicit high-order methods can still fail if stage solves are poorly conditioned.
- PTC adds damping far from the solution and recovers larger steps near convergence.
- Line search stabilizes Newton updates against overshoot near Er branch transitions.
- Newton-Krylov reduces linear-solve cost growth for larger state vectors.
- Homotopy stages reduce failure probability when implicit forcing changes abruptly.

## Practical Workflow

1. Begin with theta = 1.0 and PTC enabled.
2. Monitor residual norm and Newton iteration counts.
3. If retries are frequent, lower growth or increase shrink aggressiveness.
4. For smoother regimes, relax damping by increasing max dt factor.
5. Enable `theta_linear_solver = "gmres"` for larger state vectors.
6. Enable trust region and homotopy (`theta_trust_region_enabled`, `theta_homotopy_steps > 1`) for difficult Er branch transitions.

## Relation to Kvaerno5

- Kvaerno5 remains excellent for smooth stiff transients.
- For branch-sensitive Er behavior, theta + Newton + PTC + line search is often more robust.
