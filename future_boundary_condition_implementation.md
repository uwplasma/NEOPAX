# Future Boundary Condition Implementation Ideas

This document outlines advanced boundary condition (BC) features that go beyond the current state-of-the-art in torax and NEOPAX. These are not yet implemented, but are proposed for future development to make NEOPAX a truly next-generation transport solver framework.

## Advanced BC Features for Future Implementation

1. **Time/State-Dependent BCs**
   - Allow BCs to depend on time or the current state (e.g., time-dependent Dirichlet, feedback BCs).
   - Example: `DirichletBC(lambda t: T_edge(t))` or `NeumannBC(lambda state: grad(state))`.

2. **Event/Condition-Based BC Switching**
   - Allow BCs to change type or value based on simulation events or state (e.g., switch from Dirichlet to Neumann if a threshold is crossed).

3. **Vectorized/Batch BC Application**
   - Support batch application of BCs for multiple simulations (for parameter scans or ensemble runs).

4. **Physics-Informed/Nonlocal BCs**
   - Allow BCs that depend on integrals or nonlocal properties of the state (e.g., global constraints, flux-matching).

5. **BCs for Coupled/Multiphysics Problems**
   - Support BCs that couple multiple fields (e.g., density and temperature at the edge must satisfy a joint constraint).

6. **BCs with Automatic Differentiation Support**
   - Ensure all BC logic is compatible with JAX autodiff for gradient-based optimization and control (already true for current modular BCs, but must be preserved for advanced BCs).

7. **BCs with Uncertainty/Noise**
   - Allow stochastic or uncertain BCs for uncertainty quantification studies.

8. **BC Documentation and Examples**
   - Provide extensive documentation and usage examples for all BC types, including custom/user-supplied and advanced BCs.

---

These features are not present in torax as of March 2026, and would represent a significant advance in flexibility, research capability, and user experience for NEOPAX and the broader plasma modeling community.
