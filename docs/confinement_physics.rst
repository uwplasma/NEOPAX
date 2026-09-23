Confinement Physics
===================

NEOPAX separates transport physics into modular flux channels:

- neoclassical transport
- turbulent transport
- classical transport
- source and sink models

The neoclassical channel can use database-backed NTX coefficients or runtime
NTX evaluations.  Turbulent transport can be prescribed analytically, loaded
from files, or represented with simplified threshold models such as the ReLU
closure.

The main physics quantities exposed by the transport equations are:

- particle confinement through :math:`\Gamma_a`
- heat confinement through :math:`Q_a`
- ambipolar electric-field evolution through :math:`\sum_a Z_a\Gamma_a`
- bootstrap-current-related diagnostics through momentum-corrected
  neoclassical flow outputs

For model-specific formulas, see :doc:`transport_physics_and_flux_models`.

