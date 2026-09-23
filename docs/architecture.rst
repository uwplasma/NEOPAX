Architecture
============

NEOPAX is organized around a small number of composable runtime objects:

- geometry models provide radial grids and volume factors
- state objects carry density, pressure, and electric field profiles
- flux models produce particle, heat, and momentum-related fluxes
- source models produce density and pressure source components
- equation objects assemble finite-volume RHS terms
- solver backends advance the state in time

The central design rule is that physics models should not own the global
transport equation.  They provide local or face flux information; the equation
assembly applies finite-volume divergence, sources, boundary handling, and
state projection.

Lagged response architecture
----------------------------

Expensive flux models can expose a lagged response object.  The solver then
uses this object to approximate or linearize the expensive flux response while
still assembling the transport equations live.  This is especially important
for runtime NTX and differentiable reverse-AD workflows.

For details, see :doc:`expensive_response_methods`.

