# Solver Refactoring Plan (NEOPAX vs. Torax)

## Goal
Modularize the solver interface in NEOPAX to support multiple time integrators and nonlinear solvers, matching and benchmarking against torax (https://github.com/google-deepmind/torax).

## Current NEOPAX Status
- Modular solver interface exists (Diffrax, Newton, jaxopt, some support for Anderson/Broyden).
- Some solvers may still use Python for-loops/lists (not JAX-optimized).
- Predictor-corrector integrators and benchmarking utilities may be missing.
- API consistency, flexible stopping, and registry/factory pattern may need improvement.

## Torax Features (for comparison)
- All solvers use JAX control flow (lax.while_loop/fori_loop) for full JIT compatibility.
- Supports: Diffrax, Newton, jaxopt, Anderson, Broyden, predictor-corrector, and more.
- Consistent solver API: solve(f, x0, ...), returns result object.
- Flexible stopping criteria and runtime solver selection.
- Benchmarking and timing utilities.
- Extensive tests/examples for each solver.

## NEOPAX Refactor Action Plan

1. **Audit and Refactor All Solvers for JAX/JIT**
   - Replace Python for-loops/lists with JAX lax.while_loop or fori_loop.
   - Ensure all solver state/history is managed with JAX arrays.
   - Make all solvers @jax.jit compatible.

2. **Add Missing Solvers**
   - Implement Anderson and Broyden solvers using JAX control flow.
   - Add predictor-corrector integrators (e.g., Heun, trapezoidal) if missing.

3. **Unify Solver API**
   - Standardize all solvers to a common interface: solve(f, x0, ...), returning a result object with status, iterations, etc.
   - Add a registry/factory for runtime solver selection by string/config.

4. **Flexible Stopping and Benchmarking**
   - Allow user-supplied stopping criteria (tolerance, max steps, custom convergence).
   - Add benchmarking/timing utilities for solver comparison.

5. **Testing and Examples**
   - Add/expand tests and example scripts for each solver backend.
   - Validate all solvers on stiff and non-stiff problems.

## Implementation Steps
- [ ] Audit and refactor existing solvers for JAX/JIT.
- [ ] Implement Anderson and Broyden solvers (JAX, lax.while_loop).
- [ ] Add predictor-corrector integrators.
- [ ] Unify solver API and add registry/factory.
- [ ] Add flexible stopping and benchmarking utilities.
- [ ] Expand tests/examples for all solvers.

---

This plan will be updated as features are completed. See torax for reference implementations and best practices.
