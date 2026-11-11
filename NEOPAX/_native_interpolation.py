"""Local C1 tensor interpolation and compact scalar-query AD rules.

The endpoint slopes are the adjacent secant; interior slopes are the mean
of the two secants, matching Interpax's default ``cubic`` rule on nonuniform
grids. Values use local 4x4 or 4x4x4 coefficient stencils. Arithmetic stays in JAX so
query, coefficient, knot-coordinate and higher derivatives remain available.
Knots must be sorted, with at least two entries (as required by the database).
"""

from functools import wraps

import jax
import jax.numpy as jnp
from jax.custom_derivatives import SymbolicZero, zero_from_primal


def contract_scalar_queries(function=None, *, argnums=(0, 1, 2), joint=False):
    """Contract wide query tangents after the scalar interpolation derivative.

    Data/coordinate tangents still differentiate the original expression.
    Symbolic zeros avoid constructing those tangents for state-only AD; real
    knot, table and fit derivatives remain available, including higher AD.
    """
    if function is None:
        return lambda fn: contract_scalar_queries(fn, argnums=argnums, joint=joint)

    @jax.custom_jvp
    @wraps(function)
    def contracted(*args):
        return function(*args)

    def rule(primals, tangents):
        def varied(indices, values):
            arguments = list(primals)
            for index, value in zip(indices, values, strict=True):
                arguments[index] = value
            return function(*arguments)

        active = tuple(i for i in argnums if not isinstance(tangents[i], SymbolicZero))
        if joint and active:
            def along_queries(values):
                result = varied(active, values)
                return result, result
            partials, result = jax.jacfwd(along_queries, has_aux=True)(
                jnp.stack(tuple(primals[i] for i in active)))
            directions = jnp.stack(tuple(tangents[i] for i in active))
            result_dot = jax.tree.map(lambda partial: jnp.sum(partial * directions, axis=-1), partials)
        else:
            result = function(*primals)
            result_dot = jax.tree.map(jnp.zeros_like, result)
            for index in active:
                _, partial = jax.jvp(lambda value: varied((index,), (value,)),
                                     (primals[index],), (jnp.ones_like(primals[index]),))
                result_dot = jax.tree.map(
                    lambda accumulated, slope: accumulated + slope * tangents[index],
                    result_dot, partial)
        data_indices = tuple(i for i in range(len(primals)) if i not in argnums)
        data_primals = tuple(primals[i] for i in data_indices)
        data_tangents = tuple(tangents[i] for i in data_indices)
        if any(not isinstance(value, SymbolicZero) for value in jax.tree.leaves(data_tangents)):
            data_dots = jax.tree.map(
                lambda primal, dot: zero_from_primal(primal) if isinstance(dot, SymbolicZero) else dot,
                data_primals, data_tangents)

            def along_data(*values):
                return varied(data_indices, values)

            _, data_result_dot = jax.jvp(along_data, data_primals, data_dots)
            result_dot = jax.tree.map(jnp.add, result_dot, data_result_dot)
        return result, result_dot

    contracted.defjvp(rule, symbolic_zeros=True)
    return contracted


def _cell_upper_index(knots, query):
    """Exact right-sided lookup, with an arithmetic path for near-uniform grids."""
    step = (knots[-1] - knots[0]) / (knots.size - 1)
    offsets = step * jnp.arange(knots.size, dtype=knots.dtype)
    uniform = ((step > 0.) & jnp.all(jnp.diff(knots) > 0.)
               & jnp.all(jnp.abs(knots - (knots[0] + offsets)) < step / 4.))

    def arithmetic():
        scaled = (query - knots[0]) / jnp.where(step == 0., 1., step)
        scaled = jnp.where(jnp.isnan(query), 0., scaled)
        # Clamp before integer conversion, including very large extrapolations.
        index = jnp.clip(jnp.floor(scaled) + 1., 1., knots.size - 1).astype(jnp.int32)
        index += (knots[index] <= query).astype(jnp.int32) - (knots[index - 1] > query).astype(jnp.int32)
        return jnp.where(jnp.isnan(query), knots.size - 1, jnp.clip(index, 1, knots.size - 1))

    def search():
        method = "compare_all" if knots.size <= 64 else "scan_unrolled"
        return jnp.clip(jnp.searchsorted(knots, query, side="right", method=method), 1, knots.size - 1)

    return jax.lax.cond(uniform, arithmetic, search)


def _cubic_geometry(knots, query):
    """Read only the interval and its two neighboring knots."""
    upper = _cell_upper_index(knots, query)
    indices = jnp.clip(upper + jnp.arange(-2, 2), 0, knots.size - 1)
    local_knots = knots[indices]
    spacing = jnp.diff(local_knots)
    # Padded boundary knots repeat. Keep inactive reciprocal derivatives finite.
    inverse = jnp.where(spacing == 0, 0., 1. / jnp.where(spacing == 0, 1., spacing))
    width = spacing[1]
    t = (query - local_knots[1]) * inverse[1]
    return indices, (inverse, width, t, upper == 1, upper == knots.size - 1)


def _cubic_apply(values, geometry):
    """Apply C1 Hermite interpolation along a four-value leading axis.

    Form differences before multiplying by spacing factors. The increment
    form avoids cancellation of the common offset of nearly constant tables.
    """
    inverse, width, t, at_lower, at_upper = geometry
    slopes = jnp.diff(values, axis=0) * inverse.reshape((3,) + (1,) * (values.ndim - 1))
    lower = width * jnp.where(at_lower, slopes[1], .5 * (slopes[0] + slopes[1]))
    upper = width * jnp.where(at_upper, slopes[1], .5 * (slopes[1] + slopes[2]))
    delta = values[2] - values[1]
    return values[1] + t * (lower + t * (3. * delta - 2. * lower - upper
                                       + t * (-2. * delta + lower + upper)))


def cubic1d(knots, values, query, *, extrap=False):
    """Evaluate the database's C1 cubic rule, preserving its boundary slopes."""
    indices, geometry = _cubic_geometry(knots, query)
    value = _cubic_apply(values[indices], geometry)
    if not extrap:
        value = jnp.where((query < knots[0]) | (query > knots[-1]), jnp.nan, value)
    return value


def _bicubic_channels(x, y, tables, xq, yq):
    """Evaluate coefficient channels sharing one local 4x4 Hermite stencil.

    ``tables`` is a tuple of 2D coefficient tables on the same grid. Building
    one interval geometry avoids three full-table derivative constructions and
    the generic 16x16 bicubic coefficient transform per scalar query.
    """
    ix, x_geometry = _cubic_geometry(x, xq)
    iy, y_geometry = _cubic_geometry(y, yq)
    local = jnp.stack(tuple(table[ix[:, None], iy[None, :]] for table in tables), axis=-1)
    result = _cubic_apply(_cubic_apply(local, x_geometry), y_geometry)
    return tuple(result)


@contract_scalar_queries(argnums=(3, 4))
def bicubic_channels(x, y, tables, xq, yq):
    """Local bicubic values with query tangents contracted after interpolation.

    A state Jacobian can have hundreds of directions. Differentiate the two
    scalar queries once before multiplying those directions, so their leading
    dimension does not expand the four-by-four interpolation stencil. Knot
    and coefficient derivatives retain the same local JAX expression.
    """
    return _bicubic_channels(x, y, tables, xq, yq)


@contract_scalar_queries(argnums=(4, 5, 6))
def tricubic_channels(x, y, z, tables, xq, yq, zq):
    """The same tensor C1 cubic rule on a local 4x4x4 coefficient stencil."""
    ix, gx = _cubic_geometry(x, xq)
    iy, gy = _cubic_geometry(y, yq)
    iz, gz = _cubic_geometry(z, zq)
    local = jnp.stack(tuple(table[ix[:, None, None], iy[None, :, None], iz[None, None, :]]
                            for table in tables), axis=-1)
    return tuple(_cubic_apply(_cubic_apply(_cubic_apply(local, gx), gy), gz))
