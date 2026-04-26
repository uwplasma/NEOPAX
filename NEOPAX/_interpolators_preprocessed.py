import jax
import jax.numpy as jnp
from jax import config

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


def _clamped_interval_index_fixed(grid, value):
    left = grid[:-1]
    idx = jnp.sum(value >= left).astype(jnp.int32) - 1
    return jnp.clip(idx, 0, grid.shape[0] - 2)


def _fraction_fixed(grid, idx, value):
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


def _inpold_fixed12_er(x, xs, ys, gmix):
    pos = jnp.searchsorted(xs, x, side="left")

    def left_boundary():
        i1, i2, i3 = 0, 1, 2
        a0 = ys[i3] - ys[i2]
        a1 = xs[i3] - xs[i2]
        a2 = 0.5 * a1**2
        a3 = (2.0 / 3.0) * a1 * a2
        b0 = ys[i1] - ys[i2]
        b1 = xs[i1] - xs[i2]
        b2 = 0.5 * b1**2
        b3 = (2.0 / 3.0) * b1 * b2
        c0 = 0.0
        c1 = 1.0
        c2 = b1
        c3 = b1**2
        det = a1 * (b2 * c3 - b3 * c2) - a2 * (b1 * c3 - b3 * c1) + a3 * (b1 * c2 - b2 * c1)
        g1 = jnp.where(jnp.abs(det) <= 1.0e-30, 0.0, (a0 * (b2 * c3 - b3 * c2) - a2 * (b0 * c3 - b3 * c0) + a3 * (b0 * c2 - b2 * c0)) / det)
        g2 = jnp.where(jnp.abs(det) <= 1.0e-30, 0.0, (a1 * (b0 * c3 - b3 * c0) - a0 * (b1 * c3 - b3 * c1) + a3 * (b1 * c0 - b0 * c1)) / det)
        g3 = jnp.where(jnp.abs(det) <= 1.0e-30, 0.0, (a1 * (b2 * c0 - b0 * c2) - a2 * (b1 * c0 - b0 * c1) + a0 * (b1 * c2 - b2 * c1)) / det)

        def x_le_i1():
            gx = xs[i1] - xs[i2]
            dy = g1 + (g2 + g3 * gx) * gx
            y1 = ys[i2] + (g1 + (0.5 * g2 + (1.0 / 3.0) * g3 * gx) * gx) * gx
            return y1 + dy * (x - xs[i1])

        def x_le_i2():
            gx = x - xs[i2]
            return ys[i2] + (g1 + (0.5 * g2 + (1.0 / 3.0) * g3 * gx) * gx) * gx

        return jax.lax.cond(x <= xs[i1], x_le_i1, x_le_i2)

    def right_or_interior():
        i3 = jnp.minimum(jnp.maximum(pos, 2), 11)
        i2 = i3 - 1
        i1 = i2 - 1
        xl = xs[i2]
        yl = ys[i2]
        a0 = ys[i3] - yl
        a1 = xs[i3] - xl
        a2 = 0.5 * a1**2
        b0 = ys[i1] - yl
        b1 = xs[i1] - xl
        b2 = 0.5 * b1**2
        det = a1 * b2 - a2 * b1
        dyl_raw = jnp.where(jnp.abs(det) <= 1.0e-30, 0.0, (a0 * b2 - b0 * a2) / det)
        dyl = jnp.where((yl - ys[i1]) * (yl - ys[i3]) >= 0.0, gmix * dyl_raw, dyl_raw)

        def interior():
            i4 = i3 + 1
            xu = xs[i3]
            yu = ys[i3]
            a0u = ys[i4] - yu
            a1u = xs[i4] - xu
            a2u = 0.5 * a1u**2
            b0u = ys[i2] - yu
            b1u = xs[i2] - xu
            b2u = 0.5 * b1u**2
            detu = a1u * b2u - a2u * b1u
            dyu_raw = jnp.where(jnp.abs(detu) <= 1.0e-30, 0.0, (a0u * b2u - b0u * a2u) / detu)
            dyu = jnp.where((yu - ys[i2]) * (yu - ys[i4]) >= 0.0, gmix * dyu_raw, dyu_raw)
            return xl, yl, dyl, xu, yu, dyu

        def right_boundary():
            i2r = i3
            i1r = i2r + 1
            i3r = i2r - 1
            a0r = ys[i3r] - ys[i2r]
            a1r = xs[i3r] - xs[i2r]
            a2r = 0.5 * a1r**2
            a3r = (2.0 / 3.0) * a1r * a2r
            b0r = ys[i1r] - ys[i2r]
            b1r = xs[i1r] - xs[i2r]
            b2r = 0.5 * b1r**2
            b3r = (2.0 / 3.0) * b1r * b2r
            c0 = 0.0
            c1 = 0.0
            c2 = 1.0
            c3 = 2.0 * b1r
            detr = a1r * (b2r * c3 - b3r * c2) - a2r * (b1r * c3 - b3r * c1) + a3r * (b1r * c2 - b2r * c1)
            g1r = jnp.where(jnp.abs(detr) <= 1.0e-30, 0.0, (a0r * (b2r * c3 - b3r * c2) - a2r * (b0r * c3 - b3r * c0) + a3r * (b0r * c2 - b2r * c0)) / detr)
            xu = xs[i2r]
            yu = ys[i2r]
            return xl, yl, dyl, xu, yu, g1r

        xl, yl, dyl2, xu, yu, dyu = jax.lax.cond(i3 < 11, interior, right_boundary)
        dx = xu - xl
        dx2 = dx**2
        dx3 = dx2 * dx
        a0c = yu - yl - dx * dyl2
        a2c = 0.5 * dx2
        a3c = (1.0 / 3.0) * dx3
        b0c = dyu - dyl2
        b2c = dx
        b3c = dx2
        detc = a2c * b3c - a3c * b2c
        g1c = dyl2
        g2c = jnp.where(jnp.abs(detc) <= 1.0e-30, 0.0, (a0c * b3c - a3c * b0c) / detc)
        g3c = jnp.where(jnp.abs(detc) <= 1.0e-30, 0.0, (a2c * b0c - a0c * b2c) / detc)
        gx = x - xl
        return yl + gx * (g1c + gx * (0.5 * g2c + (1.0 / 3.0) * g3c * gx))

    return jax.lax.cond(x < xs[2], left_boundary, right_or_interior)


def _surface_ntss1d(table, ir, inu, ty, grid_er_internal, database):
    xs = database.Er_grid[ir]
    first = xs[0]
    last = xs[11]

    def interp_row(row):
        ys = table[ir, row]
        return jnp.where(
            grid_er_internal <= first,
            ys[0],
            jnp.where(
                grid_er_internal > last,
                ys[11],
                _inpold_fixed12_er(grid_er_internal, xs, ys, database.gmix_er_ntss1d),
            ),
        )

    v0 = interp_row(inu)
    v1 = interp_row(inu + 1)
    return v0 * (1.0 - ty) + v1 * ty


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


@jax.jit
def get_Dij_preprocessed_3d_ntss_radius_ntss1d(grid_x, grid_nu, grid_Er, database):
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
        d11 = _surface_ntss1d(database.D11_log, ir, inu, ty, grid_er_internal, database)
        d13 = _surface_ntss1d(database.D13, ir, inu, ty, grid_er_internal, database)
        d33 = _surface_ntss1d(database.D33, ir, inu, ty, grid_er_internal, database)
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


@jax.jit
def get_Dij_preprocessed_3d_ntss_radius_ntss1d_fixednu(grid_x, grid_nu, grid_Er, database):
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

    inu = _clamped_interval_index_fixed(database.nu_log, grid_nu_internal)
    ty = _fraction_fixed(database.nu_log, inu, grid_nu_internal)

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
        d11 = _surface_ntss1d(database.D11_log, ir, inu, ty, grid_er_internal, database)
        d13 = _surface_ntss1d(database.D13, ir, inu, ty, grid_er_internal, database)
        d33 = _surface_ntss1d(database.D33, ir, inu, ty, grid_er_internal, database)
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

