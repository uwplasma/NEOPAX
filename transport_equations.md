# transport_equations.md — Refactoring Plan for NEOPAX Transport Equations

## Goal
Separate the definition of the transport equations (PDEs/ODEs) from the source and flux models, enabling modular, composable, and user-extensible equation sets—matching or exceeding torax's flexibility.

## Required Features
- **Equation Definition:** Each equation is defined as: dU/dt = -div(total_flux) + total_source
- **Equation Registry:** Allow registration and retrieval of different equation sets (e.g., density, temperature, Er, user-defined fields).
- **User Extension:** Users can define and register custom equations.
- **JAX Compatibility:** All logic must be JAX-friendly (jit, vmap, grad).
- **Testing:** Add tests for equation assembly and JAX compatibility.
- **Documentation:** Provide usage examples for built-in and user-defined equations.

## Implementation Steps
1. Move all equation logic to a new _transport_equations.py module.
2. Define a standard interface for equations (e.g., EquationBase with __call__(state, flux_models, source_models)).
3. Implement built-in equations as subclasses and register them.
4. Allow runtime composition of equations (e.g., select which fields to evolve).
5. Add registry functions for registration and retrieval.
6. Add tests and documentation.

## Comparison with torax
- torax separates equation logic from flux/source models, allowing modular equation sets and user extension. This plan matches or exceeds torax's state of the art.
