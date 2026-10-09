# Geometry/profile block trust-region optimization

## Purpose

The combined full-transport problem has ESS-scaled geometry variables and
nominally scaled analytical-profile variables. At the seed used by the
database full-transport example, adding the profile columns does not increase
the numerical rank of the residual Jacobian. Consequently, an ordinary
single-metric least-squares trust region has no unique physical rule for
dividing a locally redundant step between geometry and profiles.

This experimental optimizer addresses that ambiguity without changing the
established geometry-only optimization path and without running a complete
geometry optimization as a warm start.

## Coordinates

Both parameter blocks are centered at the physical seed:

\[
g = g_0 + S_{\mathrm{ESS}} x_g,
\qquad
p = p_0 + S_{\mathrm{nominal}} x_p.
\]

Therefore, `x_g = 0` and `x_p = 0` at initialization. Geometry retains the
validated ESS map. A profile-coordinate change of `0.1` represents a ten
percent change from that profile's nominal seed value. Physical profile
bounds are transformed into these centered coordinates.

## Nonlinear iteration

At every iteration, one joint local model is formed:

\[
\min_{\Delta x_g,\Delta x_p}
\frac{1}{2}\left\|
r + J_g\Delta x_g + J_p\Delta x_p
\right\|^2.
\]

The model is subject to independent block limits:

- an L2 trust radius in ESS geometry coordinates;
- a maximum fractional change for every profile coordinate, together with an
  L2 profile-block limit and the physical profile bounds.

A small block-normalized proximal term selects a unique step when the two
Jacobian blocks are locally redundant. It is a numerical tie breaker, not a
new physical objective and not a Jacobian-derived reweighting.

The proposed point is evaluated with the actual nonlinear
VMEC/NTX/full-transport problem. The ratio of actual to predicted cost
reduction controls acceptance and trust-limit contraction or expansion. An
accepted candidate's already-computed residual and Jacobian become the next
iteration, so accepted steps require no duplicate derivative evaluation.

## Scope and invariants

- This is one nonlinear optimization from the original seed with both blocks
  active from the first iteration.
- It is not a geometry warm start followed by a combined optimization.
- It does not derive parameter scaling from weighted objective sensitivities.
- The established geometry-only script and `optimization.least_squares`
  behavior are unchanged.
- The ordinary combined optimization example is retained for comparison.

The opt-in example is
`examples/optimization/optimize_geometry_profiles_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_block_trust_region.py`.
