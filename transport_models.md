# transport_models.md — Refactoring Plan for NEOPAX Transport Models (Fluxes)

## Goal
Modularize and standardize all transport (flux) models in NEOPAX, matching or exceeding torax's flexibility and extensibility for conservative fluxes (e.g., neoclassical, turbulent, analytic, user-defined).

## Required Features
- **Standard Interface:** All transport models must implement `__call__(state)` and be JAX-compatible (jit, vmap, grad).
- **Registry:** Implement a registry for transport models, with runtime registration and retrieval.
- **Composition:** Allow runtime composition of multiple transport models (e.g., sum of fluxes at faces).
- **User Extension:** Users can define and register custom transport models.
- **JAX Compatibility:** All logic must be JAX-friendly.
- **Testing:** Add tests for registration, composition, and JAX compatibility.
- **Documentation:** Provide usage examples for built-in and user-defined transport models.

## Implementation Steps
1. Move all flux logic (e.g., neoclassical, turbulent) from neoclassical.py, turbulent.py, etc. to a new _transport_models.py module.
2. Define a TransportModelBase class with a standard interface.
3. Implement built-in transport models as subclasses and register them.
4. Implement a CombinedTransportModel for runtime composition (sum of fluxes).
5. Add registry functions for registration and retrieval.
6. Add tests and documentation.

## Comparison with torax
- torax implements transport (flux) models as modular, JAX-compatible callables, with runtime composition and user extension. This plan matches or exceeds torax's state of the art.
