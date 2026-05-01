Transport Physics And Flux Models
=================================

This page summarizes the transport equations solved by NEOPAX and the
mathematical structure of the built-in flux models. The goal is to make the
notation explicit and to separate clearly:

- the conservation-law structure of the transport system
- the flux decomposition into neoclassical, turbulent, and classical channels
- the way NTX/NTSS-inspired neoclassical models are assembled


Flux-Surface-Averaged Transport Form
------------------------------------

The underlying transport equations are conservation laws of the form

.. math::

   \partial_t u + \nabla \cdot \mathbf{F} = S.

NEOPAX solves a 1D radial, flux-surface-averaged version of these equations on
the radial coordinate :math:`\rho`. In this reduced form, the divergence
operator is represented as

.. math::

   \mathcal{V}_{\rho}[F]
   :=
   -\frac{1}{V'(\rho)}
   \frac{\partial}{\partial \rho}
   \left( V'(\rho)\,F(\rho) \right),

where :math:`V'(\rho)` is the differential volume factor carried by the
geometry model. In the code, this is the same conservative volume-weighted
operator used in ``conservative_update(...)``.

With that convention, NEOPAX evolves:

- densities :math:`n_s`
- pressures :math:`p_s = n_s T_s`
- optionally the radial electric field :math:`E_r`


Density Equation
----------------

For each independently evolved species,

.. math::

   \partial_t n_s
   =
   \mathcal{V}_{\rho}\!\left(\Gamma_s\right)
   +
   S^{(n)}_s.

Here:

- :math:`\Gamma_s` is the radial particle flux
- :math:`S^{(n)}_s` is the density-source contribution

NEOPAX can evolve only the independent ion/impurity densities and reconstruct
the electron density algebraically through quasi-neutrality.


Pressure / Temperature Equation
-------------------------------

NEOPAX evolves pressure rather than temperature directly. For each active
species,

.. math::

   \partial_t p_s
   =
   \frac{2}{3}
   \left[
      \mathcal{V}_{\rho}
      \left(
         Q_s
         +
         T_s \Gamma_s^{\mathrm{neo}}
         +
         T_s \Gamma_s^{\mathrm{turb}}
         +
         T_s \Gamma_s^{\mathrm{class}}
      \right)
      +
      S^{(p)}_s
      +
      q_s \Gamma_s E_r
   \right].

The fluxes are split as

.. math::

   \Gamma_s
   =
   \Gamma_s^{\mathrm{neo}}
   +
   \Gamma_s^{\mathrm{turb}}
   +
   \Gamma_s^{\mathrm{class}},

.. math::

   Q_s
   =
   Q_s^{\mathrm{neo}}
   +
   Q_s^{\mathrm{turb}}
   +
   Q_s^{\mathrm{class}}.

So the pressure equation contains:

- conductive heat flux :math:`Q_s`
- convective energy transport :math:`T_s \Gamma_s`
- pressure-source terms :math:`S^{(p)}_s`
- the optional work term :math:`q_s \Gamma_s E_r`


Radial Electric Field Equation
------------------------------

The :math:`E_r` equation is written as a relaxation-diffusion equation driven
by ambipolar charge balance:

.. math::

   \partial_t E_r
   =
   \tau_{E_r}
   \left[
      D_{E_r}\,\mathcal{V}_{\rho}(F_{E_r})
      -
      \mathcal{A}
   \right],

with

.. math::

   \mathcal{A}
   =
   \sum_s Z_s \Gamma_s

for the local ambipolar source mode. In the transport-centered mode, NEOPAX
builds the charge balance from face-reconstructed particle fluxes before
mapping them back to cell centers.


Flux-Model Composition
----------------------

The runtime transport model is built as a composition of up to three channels:

- neoclassical
- turbulent
- classical

At the combined-model level, NEOPAX keeps the components separate as long as
possible and exposes both:

- total outputs: ``Gamma``, ``Q``, ``Upar``
- split outputs:
  - ``Gamma_neo``, ``Q_neo``, ``Upar_neo``
  - ``Gamma_turb``, ``Q_turb``, ``Upar_turb``
  - ``Gamma_classical``, ``Q_classical``, ``Upar_classical``

This split is important because the pressure equation assembles the convective
terms channel-by-channel, even though the final conservative update uses the
total radial energy flux.


NTX / NTSS-Inspired Neoclassical Models
---------------------------------------

The built-in neoclassical models are based on the standard transport-matrix
representation

.. math::

   \Gamma_a
   =
   -n_a
   \left(
      L_{11}^{(a)} A_{1,a}
      +
      L_{12}^{(a)} A_{2,a}
      +
      L_{13}^{(a)} A_{3}
   \right),

.. math::

   Q_a
   =
   -n_a T_a
   \left(
      L_{21}^{(a)} A_{1,a}
      +
      L_{22}^{(a)} A_{2,a}
      +
      L_{23}^{(a)} A_{3}
   \right),

.. math::

   U_{\parallel,a}
   =
   -n_a
   \left(
      L_{31}^{(a)} A_{1,a}
      +
      L_{32}^{(a)} A_{2,a}
      +
      L_{33}^{(a)} A_{3}
   \right),

where:

- :math:`a` labels the species
- :math:`A_1`, :math:`A_2`, :math:`A_3` are the thermodynamic-force terms used
  internally by NEOPAX
- :math:`L_{ij}^{(a)}` are the species transport coefficients after energy
  convolution

In the database-driven NTX pathway, the monoenergetic inputs are

- :math:`D_{11}`
- :math:`D_{13}`
- :math:`D_{33}`

tabulated as functions of radial position, collisionality, and
:math:`E_r / v`. NEOPAX interpolates these coefficients and performs the
energy convolution on the active Laguerre velocity grid:

.. math::

   L_{ij}^{(a)}
   \sim
   \sum_k w_{ij}(x_k)\,D_{\alpha\beta}(x_k),

with the appropriate geometry- and species-dependent prefactors. In the
current implementation:

- :math:`D_{11}` is stored in logarithmic form in the database path
- :math:`D_{13}` and :math:`D_{33}` are used directly after interpolation
- :math:`D_{33}` is divided by :math:`\nu / v` in the final assembly, matching
  the code path used to build :math:`L_{33}`

This is the same broad NTX/NTSS-style strategy: monoenergetic transport data
first, thermodynamic-force assembly second, fluxes last.


Built-In Neoclassical Models
----------------------------

``ntx_database``
^^^^^^^^^^^^^^^^

This is the standard database-driven neoclassical model.

Workflow:

1. read a precomputed monoenergetic NTX database
2. interpolate :math:`D_{11}`, :math:`D_{13}`, :math:`D_{33}` at the local
   :math:`(\rho,\nu/v,E_r/v)` state
3. convolve over the active energy grid
4. assemble :math:`L_{ij}`
5. evaluate :math:`\Gamma`, :math:`Q`, and :math:`U_{\parallel}`

This is the main NTX-backed model used in the stock transport examples.


``ntx_scan_runtime``
^^^^^^^^^^^^^^^^^^^^

This model uses the same final transport law as ``ntx_database``, but the
monoenergetic database is built on the fly from NTX scan grids supplied in the
TOML file:

- ``ntx_scan_rho``
- ``ntx_scan_nu_v``
- ``ntx_scan_er_tilde``

So the distinction is:

- ``ntx_database`` reads a prebuilt database from file
- ``ntx_scan_runtime`` builds the database at runtime and then uses the same
  interpolation-and-convolution structure


``ntx_exact_lij_runtime``
^^^^^^^^^^^^^^^^^^^^^^^^^

This model bypasses the intermediate interpolated monoenergetic database.
Instead, it solves NTX directly on the active NEOPAX energy grid at the local
state and assembles :math:`L_{ij}` in real time.

Conceptually:

1. evaluate the local :math:`\nu / v` and :math:`E_r / v` arrays on the active
   energy grid
2. call NTX directly for those energy-grid points
3. assemble :math:`L_{ij}` immediately
4. evaluate :math:`\Gamma`, :math:`Q`, and :math:`U_{\parallel}`

This is the most direct NTX-backed path currently available in NEOPAX and is
the main benchmark target for the lagged-response work.


``ntx_database_with_momentum``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is the momentum-correction branch built on top of the same NTX-style
transport-matrix framework. It extends the coefficient assembly with the extra
matrix structure required by the momentum-correction closure.


Built-In Turbulent Models
-------------------------

``turbulent_analytical``
^^^^^^^^^^^^^^^^^^^^^^^^

This model uses simple diffusive closures:

.. math::

   \Gamma_s^{\mathrm{turb}}
   =
   -\chi_{n,s}\,\frac{\partial n_s}{\partial \rho},

.. math::

   Q_s^{\mathrm{turb}}
   =
   -n_s\,\chi_{T,s}\,\frac{\partial T_s}{\partial \rho}.

It is a lightweight analytical model useful for controlled transport tests.


``turbulent_power_analytical`` and ``ntss_power_over_n``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These models use the same diffusive structure but normalize the transport
coefficients with a total-power scaling consistent with the NTSS
``power-over-n`` style:

.. math::

   \chi_{n,s}^{\mathrm{eff}}
   \propto
   \chi_{n,s}\,\frac{P^{3/4}}{n_e},

.. math::

   \chi_{T,s}^{\mathrm{eff}}
   \propto
   \chi_{T,s}\,\frac{P^{3/4}}{n_e}.

Then

.. math::

   \Gamma_s^{\mathrm{turb}}
   =
   -\chi_{n,s}^{\mathrm{eff}}
   \frac{\partial n_s}{\partial \rho},

.. math::

   Q_s^{\mathrm{turb}}
   =
   -n_s\,\chi_{T,s}^{\mathrm{eff}}
   \frac{\partial T_s}{\partial \rho}.

In the code, the total-power factor is either provided explicitly or inferred
from the active pressure-source model.


File-Driven Flux Model
----------------------

``fluxes_r_file``
^^^^^^^^^^^^^^^^^

This model reads radial profiles of:

- :math:`\Gamma_s(\rho)`
- :math:`Q_s(\rho)`
- :math:`U_{\parallel,s}(\rho)`

from file and interpolates them onto the active NEOPAX radial grid.

This is mainly useful for:

- benchmark playback
- isolated transport-equation tests
- feeding externally generated fluxes into the transport solver


Classical Channel
-----------------

The combined transport interface reserves a classical channel in the same way
as the neoclassical and turbulent channels. In the stock examples this is
usually set to ``none``, but the split remains explicit in the transport
assembly so that classical contributions can be added without changing the
equation structure.
