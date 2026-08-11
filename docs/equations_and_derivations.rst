Equations and Derivations
=========================

This page states the transport equations solved by NEOPAX in the notation used
by the finite-volume implementation.

Density equation
----------------

For each independently evolved species :math:`a`,

.. math::

   \frac{\partial n_a}{\partial t}
   =
   -\left(\frac{dV}{d\rho}\right)^{-1}
   \frac{\partial}{\partial \rho}
   \left[
      \frac{dV}{d\rho}\,\Gamma_a
   \right]
   + S_a^{(n)}.

Equivalently, using the NEOPAX operator :math:`\mathcal{D}`,

.. math::

   \partial_t n_a = \mathcal{D}[\Gamma_a] + S_a^{(n)}.

When quasi-neutrality is enabled, electron density is reconstructed
algebraically and is not independently evolved.

Pressure / temperature equation
-------------------------------

NEOPAX evolves pressure :math:`p_a = n_a T_a`.  The implemented pressure
equation is

.. math::

   \frac{\partial p_a}{\partial t}
   =
   \frac{2}{3}
   \left\{
      -\left(\frac{dV}{d\rho}\right)^{-1}
      \frac{\partial}{\partial \rho}
      \left[
         \frac{dV}{d\rho}
         \left(
            Q_a
            +
            T_a \Gamma_a^{\mathrm{neo}}
            +
            T_a \Gamma_a^{\mathrm{turb}}
            +
            T_a \Gamma_a^{\mathrm{class}}
         \right)
      \right]
      + S_a^{(p)}
      + q_a \Gamma_a E_r
   \right\}.

Equivalently,

.. math::

   \frac{3}{2}\frac{\partial n_a T_a}{\partial t}
   -
   \left(\frac{dV}{d\rho}\right)^{-1}
   \frac{\partial}{\partial \rho}
   \left[
      \frac{dV}{d\rho}
      \left(
         Q_a + T_a\Gamma_a
      \right)
   \right]
   =
   S_a^{(p)} + q_a\Gamma_a E_r,

where the implementation keeps the convective pieces split by physics channel
before summing them.

Radial electric-field equation
------------------------------

When :math:`E_r` is evolved, NEOPAX uses a relaxation-diffusion equation driven
by ambipolar charge balance:

.. math::

   \frac{\partial E_r}{\partial t}
   =
   \tau_E
   \left[
      D_E\,\mathcal{D}\!\left(F_E\right)
      -
      \mathcal{A}
   \right],

with diffusive field flux

.. math::

   F_E = -\frac{\partial E_r}{\partial \rho},

and local ambipolar source

.. math::

   \mathcal{A}
   =
   \frac{e\,10^{-3}}{\epsilon_{\mathrm{eff}}}
   \sum_a Z_a \Gamma_a.

The effective permittivity factor used by the default NEOPAX mode is

.. math::

   \epsilon_{\mathrm{eff}}
   =
   \frac{1 + (\epsilon_{\mathrm{geo}}\iota^2)^{-1}}{B_0^2}
   \sum_a \frac{n_a m_a}{Z_a^2},

up to the unit conversions used internally by the state representation.

The source mode controls how :math:`\Gamma_a` is interpreted for the
ambipolar term:

- ``ambipolar_local`` uses center fluxes directly.
- transport-centered modes reconstruct center values from face-primary
  transport fluxes.

Initial ambipolar-root solves are local center solves.  During transport,
the density and temperature equations remain face-primary finite-volume
updates.

