"""Local C1 cubic interpolation for scalar monoenergetic database queries.

The endpoint slopes are the adjacent secant; interior slopes are the mean
of the two secants, matching Interpax's default ``cubic`` rule on nonuniform
grids. Each value uses a local 4x4 coefficient stencil. All arithmetic stays in JAX so
query, coefficient, knot-coordinate and higher derivatives remain available.
Knots must be sorted, with at least two entries (as required by the database).
"""

import jax
import jax.numpy as jnp
from jax.custom_derivatives import SymbolicZero


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


@jax.custom_jvp
def bicubic_channels(x, y, tables, xq, yq):
    """Local bicubic values with query tangents contracted after interpolation.

    A state Jacobian can have hundreds of directions. Differentiate the two
    scalar queries once before multiplying those directions, so their leading
    dimension does not expand the four-by-four interpolation stencil. Knot
    and coefficient derivatives retain the same local JAX expression.
    """
    return _bicubic_channels(x, y, tables, xq, yq)


def _bicubic_channels_jvp(primals, tangents):
    result = _bicubic_channels(*primals)
    result_dot = jax.tree.map(jnp.zeros_like, result)
    for index in (3, 4):
        direction = tangents[index]
        if isinstance(direction, SymbolicZero):
            continue

        def along_query(value):
            arguments = list(primals)
            arguments[index] = value
            return _bicubic_channels(*arguments)

        _, partial = jax.jvp(
            along_query, (primals[index],), (jnp.ones_like(primals[index]),))
        result_dot = jax.tree.map(
            lambda accumulated, slope: accumulated + slope * direction,
            result_dot, partial)
    if any(not isinstance(value, SymbolicZero)
           for value in jax.tree.leaves(tangents[:3])):
        data_dots = jax.tree.map(
            lambda primal, dot: jnp.zeros_like(primal) if isinstance(dot, SymbolicZero) else dot,
            primals[:3], tangents[:3])
        _, data_result_dot = jax.jvp(
            lambda x, y, tables: _bicubic_channels(x, y, tables, *primals[3:]),
            primals[:3], data_dots)
        result_dot = jax.tree.map(jnp.add, result_dot, data_result_dot)
    return result, result_dot


bicubic_channels.defjvp(_bicubic_channels_jvp, symbolic_zeros=True)
