# NEOPAX Future Boundary Conditions Improvement Plan

## Goals
- Match or exceed the flexibility and robustness of torax for boundary condition (BC) handling.
- Enable clear, user-friendly, and extensible configuration of BCs for all relevant variables (density, temperature, Er, etc.)
- Support advanced BC types and validation.

## Action Items

### 1. Schema Validation
- Integrate a schema validation library (e.g., pydantic, voluptuous) for TOML/YAML boundary config.
- Validate BC type, required parameters, and value shapes at load time.
- Provide clear error messages for invalid configs.

### 2. TOML Schema & User Experience
- Adopt a more explicit, nested TOML schema for BCs, e.g.:
  ```toml
  [boundary.density.left]
  type = "dirichlet"
  value = 1.0
  [boundary.density.right]
  type = "neumann"
  gradient = 0.0
  [boundary.temperature.left]
  type = "robin"
  value = 0.0
  decay_length = 0.5
  ...
  ```
- Document all supported BC types and parameters for each side/variable.
- Provide example configs for common scenarios.

### 3. Advanced BC Support
- Plan for future support of:
  - Time-dependent BCs (functions of t)
  - Internal/coupled BCs (e.g., interfaces, moving boundaries)
  - User-defined/custom BCs via plugin/registry
- Ensure extensibility in the BC registry and application logic.

### 4. Code Refactoring & Extensibility
- Refactor BC construction to allow easy addition of new BC types.
- Ensure all BC logic is modular and testable.
- Add tests for all supported BC types and config patterns.

### 5. Documentation & Examples
- Write user documentation for BC configuration, including:
  - Supported types (Dirichlet, Neumann, Robin, custom)
  - TOML schema and mapping
  - Example configs for density, temperature, Er
- Add developer documentation for extending BCs.

### 6. Comparison & Migration Guide
- Provide a comparison table with torax BC features.
- Write a migration guide for users familiar with torax.

## Timeline & Milestones
- **Short-term:** Schema validation, improved TOML schema, documentation/examples.
- **Medium-term:** Advanced BC support, extensibility improvements, more tests.
- **Long-term:** Full feature parity with torax, user feedback integration.

---

**Summary:**
This plan will make NEOPAX boundary condition handling robust, user-friendly, and extensible, matching or exceeding the capabilities of torax and supporting future research needs.
