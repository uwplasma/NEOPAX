Algorithms
==========

This page describes the numerical layout used by the transport equations.

Center and face grids
---------------------

NEOPAX uses a cell-centered finite-volume layout.

Cell centers carry:

- evolved density state :math:`n_a`
- evolved pressure state :math:`p_a = n_aT_a`
- evolved or prescribed :math:`E_r`
- source terms
- local ambipolarity terms

Cell faces carry:

- particle fluxes :math:`\Gamma_{a,i+1/2}`
- heat fluxes :math:`Q_{a,i+1/2}`
- convective energy fluxes :math:`T_a\Gamma_a`
- face values and face gradients reconstructed from the center state
- boundary values imposed by boundary conditions

The conservative update is always assembled from face fluxes:

.. math::

   \mathcal{D}[F]_i =
   -\frac{
      V'_{i+1/2}F_{i+1/2}
      -
      V'_{i-1/2}F_{i-1/2}
   }{
      V'_i\Delta\rho_i
   }.

Boundary conditions
-------------------

Boundary conditions are applied as face constraints for a cell-centered state.
Let :math:`u_1` and :math:`u_2` be the first two cell centers, and
:math:`u_{N}` and :math:`u_{N-1}` the last two cell centers.  Let
:math:`\Delta` be the spacing between the first two boundary faces at the side
being constrained.

Left Dirichlet, :math:`u_L` specified:

.. math::

   \left.\frac{\partial u}{\partial \rho}\right|_L
   =
   \frac{-8u_L + 9u_1 - u_2}{3\Delta}.

Left Neumann, :math:`g_L = \partial_\rho u|_L` specified:

.. math::

   u_L = \frac{-3\Delta g_L + 9u_1 - u_2}{8}.

Left Robin / exponential decay:

.. math::

   \left.\frac{\partial u}{\partial \rho}\right|_L
   =
   \frac{u_L}{L_L},
   \qquad
   u_L =
   \frac{9u_1 - u_2}{8 + 3\Delta/L_L}.

Right Dirichlet, :math:`u_R` specified:

.. math::

   \left.\frac{\partial u}{\partial \rho}\right|_R
   =
   \frac{8u_R - 9u_N + u_{N-1}}{3\Delta}.

Right Neumann, :math:`g_R = \partial_\rho u|_R` specified:

.. math::

   u_R = \frac{3\Delta g_R + 9u_N - u_{N-1}}{8}.

Right Robin / exponential decay:

.. math::

   \left.\frac{\partial u}{\partial \rho}\right|_R
   =
   -\frac{u_R}{L_R},
   \qquad
   u_R =
   \frac{9u_N - u_{N-1}}{8 + 3\Delta/L_R}.

These formulas are the cell-centered finite-volume boundary formulas.  They
are not node-centered formulas copied from a boundary-node discretization.

Canonical evaluated state
-------------------------

Flux models receive a canonical evaluated transport state containing:

- center values
- face values
- center gradients
- face gradients

This prevents different flux models from silently applying different boundary
or face-reconstruction conventions.

Time solvers
------------

NEOPAX currently supports custom Radau and theta-family implicit solvers.

Radau
^^^^^

The Radau backend solves a multi-stage Radau IIA collocation system.  It is
the preferred backend for accurate timing-dependent evolution because it has:

- high-order implicit collocation
- embedded error estimation
- robust adaptive timestep control
- stage-level Newton solves
- structured lagged-response support for expensive flux models

Use Radau when the transient time history matters.

Theta
^^^^^

The theta backend solves a one-state implicit update

.. math::

   R(y_{n+1})
   =
   y_{n+1}
   -
   y_n
   -
   \Delta t
   \left[
      (1-\theta)f(t_n,y_n)
      +
      \theta f(t_{n+1},y_{n+1})
   \right].

Theta is simpler and can be useful for relaxed convergence toward steady
states, especially at looser tolerances.  It is generally not the first choice
when exact transient timing is the main quantity of interest.

