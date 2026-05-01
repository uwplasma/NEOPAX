Solver Backends
===============

NEOPAX supports several time-integration backends for the transport problem.
The main distinction is between:

- Diffrax ODE solvers
- theta-based implicit solvers
- the custom implicit Radau solver

This page focuses especially on the custom Radau implementation, since it is
the most solver-specific part of the codebase.


Available Backends
------------------

The active backend is selected with
``transport_solver.transport_solver_backend``.

Common values are:

- ``diffrax_kvaerno5``
- ``diffrax_tsit5``
- ``diffrax_dopri5``
- ``theta``
- ``theta_newton``
- ``radau``


Diffrax Backends
----------------

The Diffrax backends are used through the normal Diffrax solver interface with
PID-style adaptive stepsize control.

These are the easiest way to use mature general-purpose time integrators while
keeping the NEOPAX transport RHS unchanged.


Theta Backends
--------------

``theta``
^^^^^^^^^

This backend applies a fixed-form implicit theta update. It supports predictor
logic and can also use the shared ``rhs_mode`` options described below.

``theta_newton``
^^^^^^^^^^^^^^^^

This is the fully Newton-based implicit theta backend. It is often a useful
proving ground for new implicit-RHS strategies because its structure is simpler
than Radau while still exercising the same transport-state machinery.


Custom Radau Solver
-------------------

NEOPAX implements a custom fixed-stage Radau IIA solver. For a stage count
:math:`s`, the collocation equations are

.. math::

   Y_i
   =
   y_n
   +
   h \sum_{j=1}^{s} a_{ij}\,
   f(t_n + c_j h, Y_j),
   \qquad i = 1,\dots,s,

with final update

.. math::

   y_{n+1}
   =
   y_n
   +
   h \sum_{i=1}^{s} b_i\,
   f(t_n + c_i h, Y_i).

For Radau IIA, the formal order is

.. math::

   p = 2s - 1.

NEOPAX currently supports the fixed stage counts:

- ``3``
- ``5``
- ``7``
- ``9``
- ``11``

through the ``radau_num_stages`` setting.


How The NEOPAX Radau Implementation Works
-----------------------------------------

The custom Radau solver in ``_transport_solvers.py`` has four main pieces.

Stage configuration
^^^^^^^^^^^^^^^^^^^

The Radau tableau, embedded estimator data, and stage-transform information
are built once for each supported fixed stage count.

Nonlinear stage solve
^^^^^^^^^^^^^^^^^^^^^

Each timestep solves the coupled Radau stage equations with Newton iteration.

Embedded error estimate
^^^^^^^^^^^^^^^^^^^^^^^

The current implementation uses an embedded estimator to measure the local
error and adapt the timestep.

Transformed linear solve
^^^^^^^^^^^^^^^^^^^^^^^^

Instead of treating the stage equations only as one monolithic dense solve, the
solver uses a transformed-stage representation, separating:

- one real block
- additional real blocks corresponding to the complex-conjugate stage pairs

This keeps the implementation closer to the classic Radau literature and to
the NTSS/Hairer-style transformed-stage viewpoint.


Important Radau Settings
------------------------

The main Radau-related settings include:

- ``radau_num_stages``
- ``radau_error_estimator``
- ``rtol``
- ``atol``
- ``nonlinear_solver_tol``
- ``nonlinear_solver_maxiter``
- ``min_step``
- ``max_step``


RHS Modes
---------

The implicit backends share the concept of ``rhs_mode``.

``black_box``
^^^^^^^^^^^^^

This is the reference path. The solver evaluates the full live transport RHS at
each stage / implicit solve point.

Use this as the physics-reference benchmark path.


``lagged_response``
^^^^^^^^^^^^^^^^^^^

This is the current ``Track D1`` mode.

It is designed to keep the expensive transport-flux response lagged while
keeping the surrounding transport assembly live. In other words, the solver
does **not** freeze the whole assembled RHS. Instead, it freezes or linearizes
the flux response and then still assembles live:

- convective terms
- work terms
- ambipolar source terms
- divergence operators
- boundary-condition handling
- source models

This is the intended path for expensive flux kernels such as the NTX runtime
models.

The detailed implementation guidance for building these expensive lagged
responses is documented in :doc:`expensive_response_methods`.


``lagged_linear_state``
^^^^^^^^^^^^^^^^^^^^^^^

This is a broader affine full-RHS approximation:

.. math::

   f(y)
   \approx
   f(y_{\mathrm{ref}})
   +
   J_{\mathrm{ref}}
   (y - y_{\mathrm{ref}}).

It is useful as a comparison mode, but it is conceptually different from the
more transport-structured ``lagged_response`` path.


Why Radau Matters In NEOPAX
---------------------------

Radau is attractive for NEOPAX because the transport problem can be:

- stiff
- strongly implicit
- expensive per RHS evaluation

So a high-order A-stable collocation method can be attractive, especially when
paired with:

- good stage predictors
- effective timestep control
- Jacobian / factorization reuse
- structured lagged transport-response models


Current Practical Guidance
--------------------------

Use ``black_box`` when:

- you want the reference physics path
- you are validating a new flux model
- you are benchmarking another RHS approximation against the live solver

Use ``lagged_response`` when:

- the active flux model is expensive
- you want to test whether flux-only lagging reduces total implicit-solve cost
- you want to compare ``Track D1`` against the fully live benchmark

Use ``lagged_linear_state`` mainly as:

- a numerical comparison mode
- a broader affine surrogate benchmark
- not as the primary structured transport-response path
