"""Explicit directional second-order response algebra for transport fluxes.

The realtime NTX response is evaluated along a Radau stage displacement
``delta_y``. This module carries the value and the first/second directional
derivatives at the accepted-step anchor. It deliberately does not call generic
JAX differentiation: NTX supplies factorized coefficient derivatives, and
these written rules propagate them through the outer state algebra.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import jax.scipy as jsp


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class DirectionalSecondOrderJet:
    """A value and derivatives along one fixed state displacement.

    ``first`` is ``f'(y0)[delta_y]`` and ``second`` is
    ``f''(y0)[delta_y, delta_y]``. The stage model is consequently
    ``value + first + 0.5 * second``. Keeping only directional quantities
    avoids forming a dense Hessian in the global transport state.
    """

    value: jax.Array
    first: jax.Array
    second: jax.Array


def constant(value) -> DirectionalSecondOrderJet:
    """Return a response-independent value."""
    value = jnp.asarray(value)
    return DirectionalSecondOrderJet(value, jnp.zeros_like(value), jnp.zeros_like(value))


def seed(value, first, second=None) -> DirectionalSecondOrderJet:
    """Seed a value with explicit directional derivatives."""
    value = jnp.asarray(value)
    first = jnp.asarray(first, dtype=value.dtype)
    second = jnp.zeros_like(value) if second is None else jnp.asarray(second, dtype=value.dtype)
    return DirectionalSecondOrderJet(value, first, second)


def _as_jet(value) -> DirectionalSecondOrderJet:
    return value if isinstance(value, DirectionalSecondOrderJet) else constant(value)


def add(left, right) -> DirectionalSecondOrderJet:
    left_jet = _as_jet(left)
    right_jet = _as_jet(right)
    return DirectionalSecondOrderJet(
        left_jet.value + right_jet.value,
        left_jet.first + right_jet.first,
        left_jet.second + right_jet.second,
    )


def negate(value) -> DirectionalSecondOrderJet:
    value = _as_jet(value)
    return DirectionalSecondOrderJet(-value.value, -value.first, -value.second)


def subtract(left, right) -> DirectionalSecondOrderJet:
    return add(left, negate(right))


def select_axis(value, index: int, axis: int = 0) -> DirectionalSecondOrderJet:
    """Apply a static index to every directional field."""
    value = _as_jet(value)
    return DirectionalSecondOrderJet(
        jnp.take(value.value, index, axis=axis),
        jnp.take(value.first, index, axis=axis),
        jnp.take(value.second, index, axis=axis),
    )


def multiply(left, right) -> DirectionalSecondOrderJet:
    """Explicit product rule through second directional order."""
    left_jet = _as_jet(left)
    right_jet = _as_jet(right)
    return DirectionalSecondOrderJet(
        left_jet.value * right_jet.value,
        left_jet.first * right_jet.value + left_jet.value * right_jet.first,
        left_jet.second * right_jet.value
        + 2.0 * left_jet.first * right_jet.first
        + left_jet.value * right_jet.second,
    )


def unary_power(value, exponent: float) -> DirectionalSecondOrderJet:
    """Explicit one-variable chain rule for a constant power."""
    value = _as_jet(value)
    base = value.value
    exponent_value = jnp.asarray(exponent, dtype=base.dtype)
    result = base**exponent_value
    slope = exponent_value * base ** (exponent_value - 1.0)
    curvature = exponent_value * (exponent_value - 1.0) * base ** (exponent_value - 2.0)
    return DirectionalSecondOrderJet(
        result,
        slope * value.first,
        slope * value.second + curvature * value.first * value.first,
    )


def reciprocal(value) -> DirectionalSecondOrderJet:
    return unary_power(value, -1.0)


def divide(left, right) -> DirectionalSecondOrderJet:
    return multiply(left, reciprocal(right))


def sqrt(value) -> DirectionalSecondOrderJet:
    return unary_power(value, 0.5)


def exp(value) -> DirectionalSecondOrderJet:
    value = _as_jet(value)
    result = jnp.exp(value.value)
    return DirectionalSecondOrderJet(
        result,
        result * value.first,
        result * (value.second + value.first * value.first),
    )


def log(value) -> DirectionalSecondOrderJet:
    value = _as_jet(value)
    reciprocal_value = 1.0 / value.value
    return DirectionalSecondOrderJet(
        jnp.log(value.value),
        reciprocal_value * value.first,
        reciprocal_value * value.second - reciprocal_value**2 * value.first * value.first,
    )


def log10(value) -> DirectionalSecondOrderJet:
    result = log(value)
    factor = 1.0 / jnp.log(jnp.asarray(10.0, dtype=result.value.dtype))
    return DirectionalSecondOrderJet(result.value * factor, result.first * factor, result.second * factor)


def erf(value) -> DirectionalSecondOrderJet:
    value = _as_jet(value)
    derivative = 2.0 / jnp.sqrt(jnp.pi) * jnp.exp(-value.value**2)
    curvature = -2.0 * value.value * derivative
    return DirectionalSecondOrderJet(
        jsp.special.erf(value.value),
        derivative * value.first,
        derivative * value.second + curvature * value.first * value.first,
    )


def maximum_with_constant_floor(value, floor) -> DirectionalSecondOrderJet:
    """Apply a floor while retaining the anchor's smooth branch.

    A transport floor is piecewise smooth.  A Taylor response must not
    differentiate through a branch change inside its validity region: entries
    above the anchor floor keep their directional derivatives, while clamped
    entries are constant to second order.
    """
    value = _as_jet(value)
    floor_value = jnp.asarray(floor, dtype=value.value.dtype)
    active = value.value > floor_value
    return DirectionalSecondOrderJet(
        jnp.maximum(value.value, floor_value),
        jnp.where(active, value.first, jnp.zeros_like(value.first)),
        jnp.where(active, value.second, jnp.zeros_like(value.second)),
    )


def absolute_with_fixed_anchor_sign(value) -> DirectionalSecondOrderJet:
    """Absolute value on the anchor's sign branch (zero is assigned + sign)."""
    value = _as_jet(value)
    sign = jnp.where(value.value < 0.0, -1.0, 1.0)
    return DirectionalSecondOrderJet(
        jnp.abs(value.value), sign * value.first, sign * value.second
    )


def sum_axis(value, axis=None, *, keepdims: bool = False) -> DirectionalSecondOrderJet:
    """Apply a linear reduction."""
    value = _as_jet(value)
    return DirectionalSecondOrderJet(
        jnp.sum(value.value, axis=axis, keepdims=keepdims),
        jnp.sum(value.first, axis=axis, keepdims=keepdims),
        jnp.sum(value.second, axis=axis, keepdims=keepdims),
    )


def stack(values, axis: int = 0) -> DirectionalSecondOrderJet:
    values = tuple(_as_jet(value) for value in values)
    return DirectionalSecondOrderJet(
        jnp.stack(tuple(value.value for value in values), axis=axis),
        jnp.stack(tuple(value.first for value in values), axis=axis),
        jnp.stack(tuple(value.second for value in values), axis=axis),
    )


def take(value, indices, axis: int = 0) -> DirectionalSecondOrderJet:
    """Apply a linear gather to each directional field."""
    value = _as_jet(value)
    return DirectionalSecondOrderJet(
        jnp.take(value.value, indices, axis=axis),
        jnp.take(value.first, indices, axis=axis),
        jnp.take(value.second, indices, axis=axis),
    )


def dynamic_index(value, index, axis: int = 0) -> DirectionalSecondOrderJet:
    """JAX dynamic index applied consistently to a directional jet."""
    value = _as_jet(value)
    return DirectionalSecondOrderJet(
        jax.lax.dynamic_index_in_dim(value.value, index, axis=axis, keepdims=False),
        jax.lax.dynamic_index_in_dim(value.first, index, axis=axis, keepdims=False),
        jax.lax.dynamic_index_in_dim(value.second, index, axis=axis, keepdims=False),
    )


def compose_ntx_coefficient_quadratic(
    reference_coefficients,
    dcoefficients_d_nu_hat,
    dcoefficients_d_epsi_hat,
    d2coefficients_d_nu_hat2,
    d2coefficients_d_nu_hat_d_epsi_hat,
    d2coefficients_d_epsi_hat2,
    nu_hat,
    epsi_hat,
) -> DirectionalSecondOrderJet:
    """Compose cached NTX coefficient derivatives with state directions.

    The NTX arrays are anchored derivatives of ``C(nu_hat, epsi_hat)``.
    ``nu_hat`` and ``epsi_hat`` are directional state jets at that same
    anchor. This is the explicit second-order chain rule which must replace
    the current live-coordinate evaluation in the state-quadratic path.
    """
    nu_hat = _as_jet(nu_hat)
    epsi_hat = _as_jet(epsi_hat)
    coefficient0 = jnp.asarray(reference_coefficients)
    coefficient_nu = jnp.asarray(dcoefficients_d_nu_hat, dtype=coefficient0.dtype)
    coefficient_epsi = jnp.asarray(dcoefficients_d_epsi_hat, dtype=coefficient0.dtype)
    coefficient_nunu = jnp.asarray(d2coefficients_d_nu_hat2, dtype=coefficient0.dtype)
    coefficient_nuepsi = jnp.asarray(d2coefficients_d_nu_hat_d_epsi_hat, dtype=coefficient0.dtype)
    coefficient_epsiepsi = jnp.asarray(d2coefficients_d_epsi_hat2, dtype=coefficient0.dtype)
    return DirectionalSecondOrderJet(
        value=coefficient0,
        first=(
            coefficient_nu * nu_hat.first[..., None]
            + coefficient_epsi * epsi_hat.first[..., None]
        ),
        second=(
            coefficient_nu * nu_hat.second[..., None]
            + coefficient_epsi * epsi_hat.second[..., None]
            + coefficient_nunu * nu_hat.first[..., None] ** 2
            + 2.0 * coefficient_nuepsi * nu_hat.first[..., None] * epsi_hat.first[..., None]
            + coefficient_epsiepsi * epsi_hat.first[..., None] ** 2
        ),
    )


def evaluate(value) -> jax.Array:
    """Evaluate the second-order Taylor polynomial at the seeded displacement."""
    value = _as_jet(value)
    return value.value + value.first + 0.5 * value.second
