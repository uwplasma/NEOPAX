# Source/Transport Model Refactoring Plan (NEOPAX vs. torax)

## Goal
Make all source and transport models in NEOPAX fully modular, pluggable, and composable at runtime, matching or exceeding torax's flexibility and extensibility.

## Required Features

1. **Standard Interface for Source/Transport Models**
   - All models must implement a standard interface: **they must define a `__call__(state)` method** (callable class or function).
   - Models must be **JAX-compatible**: operate only on JAX arrays, be pure functions (no side effects), and support `jax.jit`, `jax.vmap`, and `jax.grad`.
   - This matches torax, where all source/transport models are callables and JAX-friendly, enabling runtime composition and user extension.

2. **Registry for Source/Transport Models**
   - Implement a registry (dictionary) for source and transport models.
   - Allow registration of built-in and user-defined models at runtime.
   - Provide `register_source(name, class)` and `get_source(name, *args, **kwargs)` functions.

3. **Runtime Composition of Models**
   - Implement a `CombinedSourceModel` (or similar) that takes a list of source models and combines their outputs (e.g., sum, weighted sum, or user-defined composition).
   - Allow users to compose any combination of registered models at runtime.

4. **User-Supplied/Custom Models**
   - Allow users to define and register their own source/transport models without modifying core code.
   - Ensure user models are treated identically to built-in models.

5. **JAX Compatibility**
   - All models and composition logic must be compatible with JAX (jit, vmap, grad, etc.).
   - Avoid Python-side mutation or non-JAX control flow in model logic.

6. **Testing and Validation**
   - Add tests to ensure all registered models and compositions produce correct results.
   - Test JAX compatibility (jit, grad, vmap) for all models.

7. **Documentation and Examples**
   - Document how to register, compose, and use source/transport models.
   - Provide usage examples for built-in and user-defined models.

## Optional Advanced Features
- Support for time/state-dependent source models (e.g., sources that depend on time or the evolving state).
- Support for batch/ensemble application (e.g., vmap over models for parameter scans).
- Support for stochastic or uncertain source models (for UQ studies).

## Implementation Steps
1. Define the standard interface for source/transport models.
2. Implement the registry and registration functions.
3. Refactor built-in models to use the new interface and register them.
4. Implement the composition pattern (CombinedSourceModel).
5. Add tests for registration, composition, and JAX compatibility.
6. Expand documentation and provide usage examples.

---

This plan will ensure NEOPAX source/transport model handling is at least as flexible as torax, and ready for advanced research and user extension.
