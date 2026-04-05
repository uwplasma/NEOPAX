# sources.md — Refactoring Plan for NEOPAX Source Models

## Goal
Modularize and standardize all non-conservative source models (e.g., heating, particle sources, analytic sources) in NEOPAX, matching or exceeding torax's flexibility and extensibility.

## Required Features
- **Standard Interface:** All source models must implement `__call__(state)` and be JAX-compatible (jit, vmap, grad).
- **Registry:** Implement a registry for source models, with runtime registration and retrieval.
- **Composition:** Allow runtime composition of multiple source models (e.g., sum, weighted sum).
- **User Extension:** Users can define and register custom source models.
- **JAX Compatibility:** All logic must be JAX-friendly.
- **Testing:** Add tests for registration, composition, and JAX compatibility.
- **Documentation:** Provide usage examples for built-in and user-defined sources.

## Implementation Steps
1. Ensure all non-conservative source logic (e.g., heating, analytic sources)  in _sources_models.py can be used with classes in _sources.py, move them to _sources.py
2. Define a SourceModelBase class with a standard interface.
3. Implement built-in source models as subclasses and register them.
4. Implement a CombinedSourceModel for runtime composition.
5. Add registry functions for registration and retrieval.
6. Add tests and documentation.

## Comparison with torax
- torax implements sources as modular, JAX-compatible callables, with runtime composition and user extension. This plan matches or exceeds torax's state of the art.
