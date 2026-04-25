import jax
import jax.numpy as jnp
from jax import config
import interpax

# to use higher precision
config.update("jax_enable_x64", True)


def _clamped_interval_index(grid, value):
    last_left = grid.shape[0] - 2
    idx = jnp.searchsorted(grid, value, side="right") - 1
    return jnp.clip(idx, 0, last_left)


def _fraction(grid, idx, value):
    x0 = grid[idx]
    x1 = grid[idx + 1]
    denom = jnp.maximum(x1 - x0, 1.0e-30)
    return jnp.clip((value - x0) / denom, 0.0, 1.0)


def _bilinear(value00, value01, value10, value11, tx, ty):
    v0 = value00 * (1.0 - tx) + value10 * tx
    v1 = value01 * (1.0 - tx) + value11 * tx
    return v0 * (1.0 - ty) + v1 * ty


def _trilinear(values, tx, ty, tz):
    c00 = values[0, 0, 0] * (1.0 - tx) + values[1, 0, 0] * tx
    c01 = values[0, 0, 1] * (1.0 - tx) + values[1, 0, 1] * tx
    c10 = values[0, 1, 0] * (1.0 - tx) + values[1, 1, 0] * tx
    c11 = values[0, 1, 1] * (1.0 - tx) + values[1, 1, 1] * tx
    c0 = c00 * (1.0 - ty) + c10 * ty
    c1 = c01 * (1.0 - ty) + c11 * ty
    return c0 * (1.0 - tz) + c1 * tz


def _corner_cube(table, ir, inu0, inu1, ier0_lo, ier1_lo):
    return jnp.asarray(
        [
            [
                [table[ir, inu0, ier0_lo], table[ir, inu0, ier0_lo + 1]],
                [table[ir, inu1, ier1_lo], table[ir, inu1, ier1_lo + 1]],
            ],
            [
                [table[ir + 1, inu0, ier0_lo], table[ir + 1, inu0, ier0_lo + 1]],
                [table[ir + 1, inu1, ier1_lo], table[ir + 1, inu1, ier1_lo + 1]],
            ],
        ]
    )


def _lagrange3(x, x0, x1, x2, y0, y1, y2):
    h0 = (x - x1) * (x - x2) / ((x0 - x1) * (x0 - x2))
    h1 = (x - x0) * (x - x2) / ((x1 - x0) * (x1 - x2))
    h2 = (x - x0) * (x - x1) / ((x2 - x0) * (x2 - x1))
    return h0 * y0 + h1 * y1 + h2 * y2


def _surface_bilinear(table, er_grid, ir, inu, ty, grid_er_internal):
    ier = _clamped_interval_index(er_grid[ir], grid_er_internal)
    tz = _fraction(er_grid[ir], ier, grid_er_internal)
    return _bilinear(
        table[ir, inu, ier],
        table[ir, inu, ier + 1],
        table[ir, inu + 1, ier],
        table[ir, inu + 1, ier + 1],
        ty,
        tz,
    )


def _surface_monotonic(table, nu_log, er_grid, ir, grid_nu_internal, grid_er_internal):
    xq = jnp.asarray([jnp.clip(grid_nu_internal, nu_log[0], nu_log[-1])])
    yq = jnp.asarray([jnp.clip(grid_er_internal, er_grid[ir, 0], er_grid[ir, -1])])
    return interpax.interp2d(
        xq,
        yq,
        nu_log,
        er_grid[ir],
        table[ir],
        method="monotonic",
        extrap=False,
    )[0]


@jax.jit
def get_Dij_preprocessed_3d(grid_x, grid_nu, grid_Er, database):
    grid_nu_internal = jnp.log10(jnp.maximum(1.0e-12, grid_nu))
    er_ratio = jnp.where(
        grid_x <= database.low_limit_r,
        database.Er_lower_limit,
        jnp.maximum(database.Er_lower_limit, jnp.abs(grid_Er / grid_x)),
    )
    grid_er_internal = jnp.log10(er_ratio)

    ir = _clamped_interval_index(database.r_grid, grid_x)
    inu = _clamped_interval_index(database.nu_log, grid_nu_internal)

    tx = _fraction(database.r_grid, ir, grid_x)
    ty = _fraction(database.nu_log, inu, grid_nu_internal)

    er_grid0 = database.Er_grid[ir]
    er_grid1 = database.Er_grid[ir + 1]
    ier0 = _clamped_interval_index(er_grid0, grid_er_internal)
    ier1 = _clamped_interval_index(er_grid1, grid_er_internal)
    tz0 = _fraction(er_grid0, ier0, grid_er_internal)
    tz1 = _fraction(er_grid1, ier1, grid_er_internal)

    d11_r0 = _bilinear(
        database.D11_log[ir, inu, ier0],
        database.D11_log[ir, inu, ier0 + 1],
        database.D11_log[ir, inu + 1, ier0],
        database.D11_log[ir, inu + 1, ier0 + 1],
        ty,
        tz0,
    )
    d11_r1 = _bilinear(
        database.D11_log[ir + 1, inu, ier1],
        database.D11_log[ir + 1, inu, ier1 + 1],
        database.D11_log[ir + 1, inu + 1, ier1],
        database.D11_log[ir + 1, inu + 1, ier1 + 1],
        ty,
        tz1,
    )
    d11 = d11_r0 * (1.0 - tx) + d11_r1 * tx

    d13 = _trilinear(
        _corner_cube(database.D13, ir, inu, inu + 1, ier0, ier1),
        tx,
        ty,
        0.5 * (tz0 + tz1),
    )
    d33 = _trilinear(
        _corner_cube(database.D33, ir, inu, inu + 1, ier0, ier1),
        tx,
        ty,
        0.5 * (tz0 + tz1),
    )

    return jnp.asarray([d11, d13, d33])


@jax.jit
def get_Dij_preprocessed_3d_er_raw(grid_x, grid_nu, grid_Er, database):
    grid_nu_internal = jnp.log10(jnp.maximum(1.0e-12, grid_nu))
    er_ratio = jnp.maximum(database.Er_lower_limit, jnp.abs(grid_Er))
    grid_er_internal = jnp.log10(er_ratio)

    ir = _clamped_interval_index(database.r_grid, grid_x)
    inu = _clamped_interval_index(database.nu_log, grid_nu_internal)

    tx = _fraction(database.r_grid, ir, grid_x)
    ty = _fraction(database.nu_log, inu, grid_nu_internal)

    er_grid0 = database.Er_grid[ir]
    er_grid1 = database.Er_grid[ir + 1]
    ier0 = _clamped_interval_index(er_grid0, grid_er_internal)
    ier1 = _clamped_interval_index(er_grid1, grid_er_internal)
    tz0 = _fraction(er_grid0, ier0, grid_er_internal)
    tz1 = _fraction(er_grid1, ier1, grid_er_internal)

    d11_r0 = _bilinear(
        database.D11_log[ir, inu, ier0],
        database.D11_log[ir, inu, ier0 + 1],
        database.D11_log[ir, inu + 1, ier0],
        database.D11_log[ir, inu + 1, ier0 + 1],
        ty,
        tz0,
    )
    d11_r1 = _bilinear(
        database.D11_log[ir + 1, inu, ier1],
        database.D11_log[ir + 1, inu, ier1 + 1],
        database.D11_log[ir + 1, inu + 1, ier1],
        database.D11_log[ir + 1, inu + 1, ier1 + 1],
        ty,
        tz1,
    )
    d11 = d11_r0 * (1.0 - tx) + d11_r1 * tx

    d13 = _trilinear(
        _corner_cube(database.D13, ir, inu, inu + 1, ier0, ier1),
        tx,
        ty,
        0.5 * (tz0 + tz1),
    )
    d33 = _trilinear(
        _corner_cube(database.D33, ir, inu, inu + 1, ier0, ier1),
        tx,
        ty,
        0.5 * (tz0 + tz1),
    )

    return jnp.asarray([d11, d13, d33])


@jax.jit
def get_Dij_preprocessed_3d_monotonic(grid_x, grid_nu, grid_Er, database):
    grid_nu_internal = jnp.log10(jnp.maximum(1.0e-12, grid_nu))
    er_ratio = jnp.where(
        grid_x <= database.low_limit_r,
        database.Er_lower_limit,
        jnp.maximum(database.Er_lower_limit, jnp.abs(grid_Er / grid_x)),
    )
    grid_er_internal = jnp.log10(er_ratio)

    ir = _clamped_interval_index(database.r_grid, grid_x)
    tx = _fraction(database.r_grid, ir, grid_x)

    d11_r0 = _surface_monotonic(database.D11_log, database.nu_log, database.Er_grid, ir, grid_nu_internal, grid_er_internal)
    d11_r1 = _surface_monotonic(database.D11_log, database.nu_log, database.Er_grid, ir + 1, grid_nu_internal, grid_er_internal)
    d13_r0 = _surface_monotonic(database.D13, database.nu_log, database.Er_grid, ir, grid_nu_internal, grid_er_internal)
    d13_r1 = _surface_monotonic(database.D13, database.nu_log, database.Er_grid, ir + 1, grid_nu_internal, grid_er_internal)
    d33_r0 = _surface_monotonic(database.D33, database.nu_log, database.Er_grid, ir, grid_nu_internal, grid_er_internal)
    d33_r1 = _surface_monotonic(database.D33, database.nu_log, database.Er_grid, ir + 1, grid_nu_internal, grid_er_internal)

    d11 = d11_r0 * (1.0 - tx) + d11_r1 * tx
    d13 = d13_r0 * (1.0 - tx) + d13_r1 * tx
    d33 = d33_r0 * (1.0 - tx) + d33_r1 * tx
    return jnp.asarray([d11, d13, d33])


@jax.jit
def get_Dij_preprocessed_3d_ntss_radius(grid_x, grid_nu, grid_Er, database):
    arr = database.r_grid
    nr = arr.shape[0]
    xri = jax.lax.cond(nr == 1, lambda: arr[0], lambda: jnp.maximum(1.0e-2 * arr[0], grid_x))
    grid_nu_internal = jnp.log10(jnp.maximum(1.0e-12, grid_nu))
    er_ratio = jnp.where(
        xri <= database.low_limit_r,
        database.Er_lower_limit,
        jnp.maximum(database.Er_lower_limit, jnp.abs(grid_Er / xri)),
    )
    grid_er_internal = jnp.log10(er_ratio)

    inu = _clamped_interval_index(database.nu_log, grid_nu_internal)
    ty = _fraction(database.nu_log, inu, grid_nu_internal)

    exact_mask = jnp.abs(xri - arr) <= database.del_r
    exact_idx = jnp.argmax(exact_mask.astype(jnp.int32))
    is_exact = jnp.any(exact_mask)

    nil = jnp.where(
        xri < arr[1],
        0,
        jnp.where(
            xri >= arr[nr - 2],
            nr - 3,
            jnp.searchsorted(arr[2:nr - 1], xri, side="left"),
        ),
    )
    noi = jnp.where((xri < arr[1]) | (xri >= arr[nr - 2]), 3, 4)
    nil = jnp.where(is_exact, exact_idx, nil)
    noi = jnp.where(is_exact, 1, noi)

    stencil_idx = jnp.minimum(nil + jnp.arange(4, dtype=jnp.int32), nr - 1)

    def eval_surface(ir):
        d11 = _surface_bilinear(database.D11_log, database.Er_grid, ir, inu, ty, grid_er_internal)
        d13 = _surface_bilinear(database.D13, database.Er_grid, ir, inu, ty, grid_er_internal)
        d33 = _surface_bilinear(database.D33, database.Er_grid, ir, inu, ty, grid_er_internal)
        return jnp.asarray([d11, d13, d33])

    atc = jax.vmap(eval_surface)(stencil_idx)

    def exact():
        return atc[0]

    def small_r():
        xr2 = xri * xri
        xr3 = xr2 * xri
        r1 = arr[0]
        r2 = arr[1]
        r3 = arr[2]
        r12 = r1 * r1
        r22 = r2 * r2
        r32 = r3 * r3
        r13 = r1 * r12
        r23 = r2 * r22
        r33 = r3 * r32

        def comp(v0, v1, v2):
            ha = ((v2 - v1) / (r33 - r23) - (v2 - v0) / (r33 - r13)) / (
                (r32 - r22) / (r33 - r23) - (r32 - r12) / (r33 - r13)
            )
            hb = ((v2 - v1) / (r32 - r22) - (v2 - v0) / (r32 - r12)) / (
                (r33 - r23) / (r32 - r22) - (r33 - r13) / (r32 - r12)
            )
            hg = v0 - r12 * ha - r13 * hb
            return hg + xr2 * ha + xr3 * hb

        return jnp.asarray(
            [
                comp(atc[0, 0], atc[1, 0], atc[2, 0]),
                comp(atc[0, 1], atc[1, 1], atc[2, 1]),
                comp(atc[0, 2], atc[1, 2], atc[2, 2]),
            ]
        )

    def edge3():
        x0 = arr[nil]
        x1 = arr[nil + 1]
        x2 = arr[nil + 2]
        return jnp.asarray(
            [
                _lagrange3(xri, x0, x1, x2, atc[0, 0], atc[1, 0], atc[2, 0]),
                _lagrange3(xri, x0, x1, x2, atc[0, 1], atc[1, 1], atc[2, 1]),
                _lagrange3(xri, x0, x1, x2, atc[0, 2], atc[1, 2], atc[2, 2]),
            ]
        )

    def interior4():
        x0 = arr[nil]
        x1 = arr[nil + 1]
        x2 = arr[nil + 2]
        x3 = arr[nil + 3]

        def lagrange4(y0, y1, y2, y3):
            h0 = (xri - x1) * (xri - x2) * (xri - x3) / ((x0 - x1) * (x0 - x2) * (x0 - x3))
            h1 = (xri - x0) * (xri - x2) * (xri - x3) / ((x1 - x0) * (x1 - x2) * (x1 - x3))
            h2 = (xri - x0) * (xri - x1) * (xri - x3) / ((x2 - x0) * (x2 - x1) * (x2 - x3))
            h3 = (xri - x0) * (xri - x1) * (xri - x2) / ((x3 - x0) * (x3 - x1) * (x3 - x2))
            return h0 * y0 + h1 * y1 + h2 * y2 + h3 * y3

        return jnp.asarray(
            [
                lagrange4(atc[0, 0], atc[1, 0], atc[2, 0], atc[3, 0]),
                lagrange4(atc[0, 1], atc[1, 1], atc[2, 1], atc[3, 1]),
                lagrange4(atc[0, 2], atc[1, 2], atc[2, 2], atc[3, 2]),
            ]
        )

    return jax.lax.cond(
        noi == 1,
        exact,
        lambda: jax.lax.cond(
            xri < arr[1],
            small_r,
            lambda: jax.lax.cond((xri >= arr[nr - 2]) | (noi == 3), edge3, interior4),
        ),
    )
