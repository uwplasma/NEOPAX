Theory and Conventions
======================

NEOPAX solves flux-surface-averaged 1D radial transport equations.  The
primary radial coordinate is denoted :math:`\rho`, and the geometry supplies
the differential volume factor :math:`dV/d\rho`.

The conservative radial divergence used throughout the transport equations is

.. math::

   \mathcal{D}[F]_i =
   -\frac{
      V'_{i+1/2} F_{i+1/2} - V'_{i-1/2} F_{i-1/2}
   }{
      V'_i \Delta \rho_i
   },

with an axis-safe finite-volume fallback when the point value :math:`V'_i`
vanishes at the magnetic axis.

Sign convention
---------------

The equation code uses ``conservative_update`` for the operator above.  Thus a
positive outward face flux decreases the cell content through the negative
flux divergence.

Flux decomposition
------------------

Particle and heat fluxes are split into neoclassical, turbulent, and classical
channels:

.. math::

   \Gamma_a =
   \Gamma_a^{\mathrm{neo}} +
   \Gamma_a^{\mathrm{turb}} +
   \Gamma_a^{\mathrm{class}},

.. math::

   Q_a =
   Q_a^{\mathrm{neo}} +
   Q_a^{\mathrm{turb}} +
   Q_a^{\mathrm{class}}.

NEOPAX keeps these components available separately because the temperature
equation assembles conductive, convective, and work terms differently.

