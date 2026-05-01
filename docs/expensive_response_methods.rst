Expensive Response Methods
==========================

This page records the recommended method for implementing lagged-response
transport models when the underlying flux kernel is expensive, differentiable,
and already has a lower-level JAX-native solve structure. The immediate target
is the NTX realtime monoenergetic path, but the design is intended to be
reused for later expensive-response implementations.


Design Goal
-----------

For implicit transport backends such as ``radau`` and ``theta_newton``, the
main cost driver can be repeated reevaluation of an expensive flux model
inside stage / Newton iterations. The purpose of the ``lagged_response`` mode
is **not** to freeze the whole transport RHS. Instead, the purpose is:

- keep the flux response cheap over one step attempt
- keep the surrounding transport assembly live
- preserve a JAX-friendly, differentiable path
- avoid turning the expensive model into a giant NEOPAX-side black-box Jacobian

The target structure is:

.. math::

   \text{state}
   \;\to\;
   \text{transport drives}
   \;\to\;
   \text{expensive local flux kernel}
   \;\to\;
   (\Gamma, Q, U_\parallel)
   \;\to\;
   \text{live transport assembly}.


What The NTX Review Suggests
----------------------------

The NTX repository already contains the right lower-level ingredients for this
kind of response model:

- prepared geometry/operator caching through
  ``PreparedMonoenergeticSystem``
- a prepared monoenergetic solve interface through
  ``prepare_monoenergetic_system(...)``
- a differentiated coefficient solve contract point through
  ``solve_prepared_coefficient_vector_vjp(...)``
- a JAX-native scan style based on
  ``jax.vmap(...)`` and ``jax.jit(...)``

This means that NEOPAX should not build its expensive-model response around a
generic full-state Jacobian of the assembled flux model if a better
model-local solve interface already exists.


Recommended Response Hierarchy
------------------------------

The recommended hierarchy for an expensive differentiable transport model is:

1. **Prepared static support**
2. **Reference response point**
3. **On-demand local tangent evaluation**
4. **Live flux-to-RHS assembly**


1. Prepared Static Support
^^^^^^^^^^^^^^^^^^^^^^^^^^

Static geometry/operator data should be prebuilt outside the hot solver loop.
For the NTX prepared monoenergetic solve, this means:

- prebuild one ``PreparedMonoenergeticSystem`` per radial location needed by
  the model
- keep these as pytrees so they remain JAX-compatible as static model-owned
  support
- avoid building surfaces, operators, or derivative blocks inside the traced
  implicit stage loop

This is the analogue of separating:

- static file/geometry setup
- dynamic transport-state dependence

which is necessary for both compile-time control and autodiff clarity.


2. Reference Response Point
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For one lagged-response build at a reference state :math:`y_{\mathrm{ref}}`,
store only the data needed to reapply the local response:

- the reference transport-drive inputs
- the reference monoenergetic coefficient solve output, or the derived
  reference :math:`L_{ij}`
- any small metadata needed to reconstruct the same local drive mapping

The important point is that the lagged object should be compact. It should not
default to storing a full explicit state-to-flux Jacobian for expensive
models.


3. On-Demand Local Tangent Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the actual within-step perturbation
:math:`\delta k = k(y) - k(y_{\mathrm{ref}})`, the preferred method is to
apply one tangent evaluation at the model-local solve level rather than
materializing a full Jacobian.

For the NTX-style path, the most natural linearization variable is the
prepared coefficient solve:

.. math::

   (\nu\_a, \epsilon\_a)
   \;\mapsto\;
   c_a
   \;\mapsto\;
   L_{ij}^{(a)}
   \;\mapsto\;
   (\Gamma_a, Q_a, U_{\parallel,a}).

The response should therefore be built around:

- reference inputs :math:`(\nu_a, \epsilon_a)`
- reference coefficient vector :math:`c_a`
- an on-demand tangent pushforward

and not around a broad NEOPAX-side full-state Jacobian.


Why JVP Is Preferred
--------------------

JAX documents :func:`jax.linearize` as a way to reuse a linearization point for
multiple tangent vectors, but it also notes that the memory usage scales with
the size of the computation much like reverse mode. The same documentation also
notes that when the tangent directions are known at once, it can be more
efficient to use ``vmap(jvp(...))`` and avoid the stored-linearization cost.

Source:
https://docs.jax.dev/en/latest/_autosummary/jax.linearize.html

For NEOPAX ``D1``, the usual situation is:

- we build one lagged response at the start of a step attempt
- we need the pushforward only for the actual state perturbation inside that
  step attempt
- we do **not** usually need a large family of unrelated tangent probes at the
  same point

So the preferred default is:

- store the reference point
- evaluate one on-demand :func:`jax.jvp` for the actual perturbation

rather than:

- build and store a full explicit Jacobian tensor

unless repeated many-direction probes are truly required.


Coefficient-Level Versus Lij-Level Response
-------------------------------------------

For NTX-like neoclassical models, the best linearization point is usually the
coefficient solve, not the final fluxes directly.

Reason:

- NTX naturally solves for monoenergetic coefficients
- the prepared solve and its custom-VJP are already formulated at that level
- mapping coefficients to :math:`L_{ij}` is cheap algebra
- mapping :math:`L_{ij}` to fluxes is also cheap compared with the prepared
  solve itself

So the preferred response chain is:

.. math::

   \delta c
   \;\to\;
   \delta L_{ij}
   \;\to\;
   \delta \Gamma,\ \delta Q,\ \delta U_\parallel.

This keeps the expensive differentiated operation as close as possible to the
actual physics solve primitive.


What Should Stay Live
---------------------

In ``lagged_response`` mode, the following should remain live around the
lagged flux response:

- thermodynamic-force assembly
- convective terms built from current :math:`T` and lagged/current
  :math:`\Gamma`
- work terms built from current :math:`E_r` and lagged/current
  :math:`\Gamma`
- ambipolar charge-balance assembly
- divergence operators
- boundary-condition application
- state floors / regularization
- source-model evaluation, unless a separate future design explicitly chooses
  otherwise

This is the key distinction between:

- a **flux-response lagged model**, and
- a **full assembled-RHS affine surrogate**

The former is the intended ``D1`` design.


Face-Flux Strategy
------------------

Face fluxes deserve separate treatment because the transport discretization
actually consumes face quantities in the conservative update. For expensive
models there are two useful modes:

- ``face_local_response``
  - evaluate the expensive response at the face state directly
  - this is the safer and more faithful choice
- ``interpolate_center_response``
  - reuse already-computed center fluxes and interpolate them radially to the
    faces
  - this is the aggressive reduced-cost option

The second mode is attractive because in the transport equations the center
fluxes are often already available. Reusing them avoids a second expensive
face-local solve in the same RHS evaluation.

However, it is still an additional approximation, because it replaces:

- face-local expensive response evaluation

with:

- center response followed by radial interpolation

So the recommended policy is:

- treat ``face_local_response`` as the safer reference mode
- treat ``interpolate_center_response`` as the performance benchmark mode
- compare them explicitly against the black-box reference on:
  - accepted/rejected steps
  - edge behavior
  - ambipolar response
  - profile drift

This is the right place to save extra expensive NTX evaluations, but only
after verifying that the reduced-cost mode is numerically acceptable for the
target regime.


No-Python-Loop Scan Principle
-----------------------------

For expensive differentiable scans, the active runtime path should avoid Python
loops over scan coordinates. The preferred structure is:

- flatten or broadcast the active scan arrays
- evaluate with ``jax.vmap(...)``
- wrap the vectorized kernel in ``jax.jit(...)`` when appropriate

This is already the pattern used in NTX's prepared scan helpers and should be
the reference pattern in NEOPAX runtime NTX implementations as well.

Python loops are acceptable only in:

- static setup
- model construction
- preprocessing that happens outside the traced hot path

and not in the active differentiated scan/evaluation kernel.


Recommended NEOPAX Template For Future Expensive Models
-------------------------------------------------------

For any future expensive model, the preferred implementation order is:

1. Identify the model-local prepared/static support object.
2. Identify the smallest meaningful transport-drive parameterization.
3. Identify the model-local differentiated solve or response contract point.
4. Store only a compact reference-point lagged-response object.
5. Apply the lagged response with one on-demand tangent pushforward for the
   actual perturbation.
6. Keep the surrounding NEOPAX assembly live.

If the lower-level model already provides a custom derivative rule, NEOPAX
should use that contract point instead of building a broader generic Jacobian
outside the model.


Decision For The NTX Realtime Path
----------------------------------

After reviewing the NTX prepared-solver implementation, the preferred method
for the expensive NTX realtime lagged-response path is:

- prebuild prepared monoenergetic systems per radius
- compute reference monoenergetic transport inputs
- use the NTX prepared coefficient solve as the response kernel
- apply the lagged response through a tangent-on-demand pushforward
- map coefficient tangents into :math:`L_{ij}`
- assemble NEOPAX fluxes and transport terms live

This should be treated as the reference pattern for later expensive-kernel
implementations, not as an NTX-specific special case only.
