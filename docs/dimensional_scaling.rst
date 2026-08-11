Dimensional Scaling
===================

NEOPAX separates physical transport quantities from the scaled state used by
the numerical solver.

The evolved state is stored as:

- density-like state variables for :math:`n_a`
- pressure-like state variables for :math:`p_a = n_a T_a`
- optionally the radial electric field :math:`E_r`

Flux models return physical particle and heat fluxes.  The equation assembly
then applies the internal conversion factors used by the state representation.
This keeps the flux models readable while allowing the solver state to remain
well scaled.

Common physical quantities
--------------------------

- :math:`n_a`: species density
- :math:`T_a`: species temperature
- :math:`p_a = n_a T_a`: species pressure
- :math:`\Gamma_a`: radial particle flux
- :math:`Q_a`: conductive heat flux
- :math:`U_{\parallel,a}`: parallel flow / momentum-related output
- :math:`E_r`: radial electric field
- :math:`V'`: differential volume factor

Radial coordinate
-----------------

The transport grid is a 1D radial grid.  Many examples label the normalized
radial coordinate as :math:`\rho`; internally the geometry object carries both
cell centers and cell faces.

When comparing with NTSS/T3D-style inputs, check whether a model parameter is
defined using physical minor-radius gradients or normalized
:math:`\rho` gradients.  The ReLU turbulence model intentionally follows the
T3D convention and uses normalized logarithmic gradient thresholds.

