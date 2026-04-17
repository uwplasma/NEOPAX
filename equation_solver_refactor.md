# NEOPAX Equation System Refactor Plan (torax-style)

## 1. Equation Registry and Selection
- Keep the registry and equation classes as in _transport_equations.py.
- Add `build_equation_system_from_config(config, species)`:
  - Reads `[equations]` section from config.
  - Returns a list of equation instances to evolve (e.g., density, temperature, Er).

## 2. Equation System Composition
- Add `ComposedEquationSystem` class:
  - Takes a list of equation instances.
  - Implements `__call__`: given state, flux_models, etc., returns a new state with only the selected equations evolved (others zeroed).
  - Handles per-species toggles and quasi-neutrality if needed.

## 3. JIT-Compatible Vector Field
- Add `make_vector_field(equation_system, ...)`:
  - Returns a function `(t, y, args) -> dy/dt` for JAX ODE solvers.
  - Pure, JIT/grad/vmap compatible.

## 4. Solver Integration
- Move all time integrator selection/config (theta, Radau, Diffrax) here or to _transport_solver.py.
- Provide `run_transport_simulation(config, state0, ...)`:
  - Builds equation system.
  - Builds vector field.
  - Selects and runs integrator.
  - Returns solution.

## 5. Orchestrator/Main
- Only parses config, builds state, calls `run_transport_simulation`.
- Handles output.

## 6. Best Practices
- All equation/vector field logic is JAX PyTree-compatible and side-effect free.
- Use dataclasses and @jax.tree_util.register_dataclass for all state/equation objects.
- No Python-side branching/mutation inside vector field.

---

## Implementation Steps
1. Implement `build_equation_system_from_config` in _transport_equations.py.
2. Implement `ComposedEquationSystem` in _transport_equations.py.
3. Implement `make_vector_field` in _transport_equations.py.
4. Move time integrator selection/config to _transport_equations.py or _transport_solver.py.
5. Implement `run_transport_simulation` as the main entrypoint.
6. Refactor main.py to only call `run_transport_simulation`.

---

This plan matches torax's modular, registry-driven, and JIT-compatible architecture for equation selection and time integration.
