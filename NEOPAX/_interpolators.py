import jax
import jax.numpy as jnp
from jax import config
from jax import jit

config.update("jax_enable_x64", True)

import interpax


def _interp1d_linear(x, y, value):
    idx = jnp.clip(jnp.searchsorted(x, value, side="right") - 1, 0, x.shape[0] - 2)
    x0 = x[idx]
    x1 = x[idx + 1]
    y0 = y[idx]
    y1 = y[idx + 1]
    w = (value - x0) / (x1 - x0 + 1e-12)
    return y0 + w * (y1 - y0)


def _er_curve_interp_d11(index, nu_idx, grid_er, database):
    er_grid = database.Er_list[index, :]
    values = database.D11_log[index, nu_idx, :]
    return jax.lax.cond(
        grid_er <= er_grid[0],
        lambda _: values[0],
        lambda _: jax.lax.cond(
            grid_er > er_grid[-1],
            lambda __: database.D11_lower_limit,
            lambda __: _interp1d_linear(er_grid, values, grid_er),
            operand=None,
        ),
        operand=None,
    )


def _er_curve_interp_d13(index, nu_idx, grid_er, database):
    er_grid = database.Er_list[index, :]
    values = database.D13[index, nu_idx, :]
    return jax.lax.cond(
        grid_er <= er_grid[0],
        lambda _: values[0],
        lambda _: jax.lax.cond(
            grid_er > er_grid[-1],
            lambda __: jnp.array(0.0, dtype=values.dtype),
            lambda __: _interp1d_linear(er_grid, values, grid_er),
            operand=None,
        ),
        operand=None,
    )


def _er_curve_interp_d33(index, nu_idx, grid_er, database):
    er_grid = database.Er_list[index, :]
    values = database.D33[index, nu_idx, :]
    return jax.lax.cond(
        grid_er <= er_grid[0],
        lambda _: values[0],
        lambda _: jax.lax.cond(
            grid_er > er_grid[-1],
            lambda __: values[-1],
            lambda __: _interp1d_linear(er_grid, values, grid_er),
            operand=None,
        ),
        operand=None,
    )


def _interp_zero_efield(index, grid_nu, database):
    nu_grid = database.nu_log

    def _low(_):
        return jnp.array(
            [
                database.D11_log[index, 0, 0] + (nu_grid[0] - grid_nu),
                database.D13[index, 0, 0],
                database.D33[index, 0, 0],
            ]
        )

    def _mid(_):
        return jnp.array(
            [
                _interp1d_linear(nu_grid, database.D11_log[index, :, 0], grid_nu),
                _interp1d_linear(nu_grid, database.D13[index, :, 0], grid_nu),
                _interp1d_linear(nu_grid, database.D33[index, :, 0], grid_nu),
            ]
        )

    def _high(_):
        return jnp.array(
            [
                database.D11_log[index, -1, 0] + (grid_nu - nu_grid[-1]),
                jnp.array(0.0, dtype=database.D13.dtype),
                database.D33[index, -1, 0],
            ]
        )

    branch = jnp.where(grid_nu < nu_grid[0], 0, jnp.where(grid_nu > nu_grid[-1], 2, 1))
    return jax.lax.switch(branch, (_low, _mid, _high), operand=None)


def _interp_low_nu_finite(index, grid_nu, grid_er, database):
    er_shift = (database.nu_log[0] - grid_nu) / 3.0 + grid_er
    d11 = jax.lax.cond(
        er_shift > database.Er_list[index, -1],
        lambda _: database.D11_lower_limit,
        lambda _: _interp1d_linear(database.Er_list[index, :], database.D11_log[index, 0, :], er_shift),
        operand=None,
    )
    d13 = _er_curve_interp_d13(index, 0, grid_er, database)
    d33 = _er_curve_interp_d33(index, 0, grid_er, database)
    return jnp.array([d11, d13, d33])


def _interp_high_nu_finite(index, grid_nu, grid_er, database):
    del grid_nu
    nu_last = database.nu_log.shape[0] - 1
    return jnp.array(
        [
            _er_curve_interp_d11(index, nu_last, grid_er, database),
            _er_curve_interp_d13(index, nu_last, grid_er, database),
            _er_curve_interp_d33(index, nu_last, grid_er, database),
        ]
    )


def _interp_finite_row(index, nu_idx, grid_er, database):
    return jnp.array(
        [
            _er_curve_interp_d11(index, nu_idx, grid_er, database),
            _er_curve_interp_d13(index, nu_idx, grid_er, database),
            _er_curve_interp_d33(index, nu_idx, grid_er, database),
        ]
    )


def _lagrange3(xs, ys, x):
    h0 = (x - xs[1]) * (x - xs[2]) / ((xs[0] - xs[1]) * (xs[0] - xs[2]))
    h1 = (x - xs[0]) * (x - xs[2]) / ((xs[1] - xs[0]) * (xs[1] - xs[2]))
    h2 = (x - xs[0]) * (x - xs[1]) / ((xs[2] - xs[0]) * (xs[2] - xs[1]))
    return h0 * ys[0] + h1 * ys[1] + h2 * ys[2]


def _butland4(xs, ys, x, mix):
    dhl0 = (xs[1] - xs[2]) / ((xs[0] - xs[1]) * (xs[0] - xs[2]))
    dhl1 = (2.0 * xs[1] - xs[0] - xs[2]) / ((xs[1] - xs[0]) * (xs[1] - xs[2]))
    dhl2 = (xs[1] - xs[0]) / ((xs[2] - xs[0]) * (xs[2] - xs[1]))
    dhu0 = (xs[2] - xs[3]) / ((xs[1] - xs[2]) * (xs[1] - xs[3]))
    dhu1 = (2.0 * xs[2] - xs[1] - xs[3]) / ((xs[2] - xs[1]) * (xs[2] - xs[3]))
    dhu2 = (xs[2] - xs[1]) / ((xs[3] - xs[1]) * (xs[3] - xs[2]))

    dg_l = dhl0 * ys[0] + dhl1 * ys[1] + dhl2 * ys[2]
    dg_u = dhu0 * ys[1] + dhu1 * ys[2] + dhu2 * ys[3]
    same_sign = dg_l * dg_u > 0.0
    dg_l = jnp.where(same_sign, dg_l, mix * dg_l)
    dg_u = jnp.where(same_sign, dg_u, mix * dg_u)

    dx = xs[2] - xs[1]
    xhat = (x - xs[1]) / (dx + 1e-12)
    ha = 3.0 * (ys[2] - ys[1]) - (2.0 * dg_l + dg_u) * dx
    hb = -2.0 * (ys[2] - ys[1]) + (dg_l + dg_u) * dx
    return ys[1] + xhat * (dg_l * dx + xhat * (ha + xhat * hb))


def _no_data_values(index, nu_idx, database):
    return jnp.array([database.D11_lower_limit, 0.0, database.D33[index, nu_idx, -1]])


def _interp_mid_finite_3(index, grid_nu, grid_er, database, start_idx):
    xs = database.nu_log[start_idx : start_idx + 3]
    rows = jnp.stack([_interp_finite_row(index, start_idx + i, grid_er, database) for i in range(3)], axis=0)
    d11_bad = (rows[2, 0] <= database.D11_lower_limit) | (
        (rows[1, 0] <= database.D11_lower_limit) & (grid_nu <= xs[1])
    ) | ((rows[0, 0] <= database.D11_lower_limit) & (grid_nu <= xs[0]))
    return jax.lax.cond(
        d11_bad,
        lambda _: _no_data_values(index, start_idx, database),
        lambda _: jnp.array(
            [
                _lagrange3(xs, rows[:, 0], grid_nu),
                _lagrange3(xs, rows[:, 1], grid_nu),
                _lagrange3(xs, rows[:, 2], grid_nu),
            ]
        ),
        operand=None,
    )


def _interp_mid_finite_4(index, grid_nu, grid_er, database):
    arr = grid_nu - database.nu_log[1:-1]
    index_nu = jnp.argmax(jnp.where(arr <= 0, arr, -jnp.inf)) + 1
    start_idx = index_nu - 1
    xs = jax.lax.dynamic_slice(database.nu_log, (start_idx,), (4,))
    offsets = jnp.arange(4, dtype=start_idx.dtype)
    rows = jax.vmap(lambda off: _interp_finite_row(index, start_idx + off, grid_er, database))(offsets)
    return jax.lax.cond(
        rows[2, 0] <= database.D11_lower_limit,
        lambda _: _no_data_values(index, start_idx, database),
        lambda _: jnp.array(
            [
                _butland4(xs, rows[:, 0], grid_nu, 0.2),
                _butland4(xs, rows[:, 1], grid_nu, 0.2),
                _butland4(xs, rows[:, 2], grid_nu, 0.2),
            ]
        ),
        operand=None,
    )


def interpolator_nu_Er_general_NTSS(index, grid_nu, grid_er, database):
    nu_grid = database.nu_log
    finite_er = grid_er > database.Er_lower_limit_log
    branch = jnp.where(
        finite_er,
        jnp.where(
            grid_nu < nu_grid[0],
            3,
            jnp.where(
                grid_nu <= nu_grid[1],
                4,
                jnp.where(grid_nu < nu_grid[-2], 6, jnp.where(grid_nu <= nu_grid[-1], 5, 7)),
            ),
        ),
        jnp.where(grid_nu < nu_grid[0], 0, jnp.where(grid_nu <= nu_grid[-1], 1, 2)),
    )
    return jax.lax.switch(
        branch,
        (
            lambda _: _interp_zero_efield(index, grid_nu, database),
            lambda _: _interp_zero_efield(index, grid_nu, database),
            lambda _: _interp_zero_efield(index, grid_nu, database),
            lambda _: _interp_low_nu_finite(index, grid_nu, grid_er, database),
            lambda _: _interp_mid_finite_3(index, grid_nu, grid_er, database, 0),
            lambda _: _interp_mid_finite_3(index, grid_nu, grid_er, database, nu_grid.shape[0] - 3),
            lambda _: _interp_mid_finite_4(index, grid_nu, grid_er, database),
            lambda _: _interp_high_nu_finite(index, grid_nu, grid_er, database),
        ),
        operand=None,
    )


@jit
def interpolator_nu_Er_general(index, grid_nu, grid_er, database):
    x11 = interpax.Interpolator2D(database.nu_log, database.Er_list[index, :], database.D11_log[index, :, :], extrap=True)(grid_nu, grid_er)
    x13 = interpax.Interpolator2D(database.nu_log, database.Er_list[index, :], database.D13[index, :, :], extrap=True)(grid_nu, grid_er)
    x33 = interpax.Interpolator2D(database.nu_log, database.Er_list[index, :], database.D33[index, :, :], extrap=True)(grid_nu, grid_er)
    return jnp.array([x11, x13, x33])


def _make_interpolation_small_r(interpolator_fn):
    @jit
    def interpolation_small_r(grid_x, grid_nu, grid_er, database):
        xr2 = grid_x**2
        xr3 = grid_x**3
        r12 = database.r1**2
        r22 = database.r2**2
        r32 = database.r3**2
        r13 = database.r1**3
        r23 = database.r2**3
        r33 = database.r3**3
        x11_0, x13_0, x33_0 = interpolator_fn(0, grid_nu, grid_er, database)
        x11_1, x13_1, x33_1 = interpolator_fn(1, grid_nu, grid_er, database)
        x11_2, x13_2, x33_2 = interpolator_fn(2, grid_nu, grid_er, database)

        ha1_11 = ((x11_2 - x11_1) / (r33 - r23) - (x11_2 - x11_0) / (r33 - r13)) / (
            (r32 - r22) / (r33 - r23) - (r32 - r12) / (r33 - r13)
        )
        hb1_11 = ((x11_2 - x11_1) / (r32 - r22) - (x11_2 - x11_0) / (r32 - r12)) / (
            (r33 - r23) / (r32 - r22) - (r33 - r13) / (r32 - r12)
        )
        hg1_11 = x11_0 - r12 * ha1_11 - r13 * hb1_11

        ha1_13 = ((x13_2 - x13_1) / (r33 - r23) - (x13_2 - x13_0) / (r33 - r13)) / (
            (r32 - r22) / (r33 - r23) - (r32 - r12) / (r33 - r13)
        )
        hb1_13 = ((x13_2 - x13_1) / (r32 - r22) - (x13_2 - x13_0) / (r32 - r12)) / (
            (r33 - r23) / (r32 - r22) - (r33 - r13) / (r32 - r12)
        )
        hg1_13 = x13_0 - r12 * ha1_13 - r13 * hb1_13

        ha1_33 = ((x33_2 - x33_1) / (r33 - r23) - (x33_2 - x33_0) / (r33 - r13)) / (
            (r32 - r22) / (r33 - r23) - (r32 - r12) / (r33 - r13)
        )
        hb1_33 = ((x33_2 - x33_1) / (r32 - r22) - (x33_2 - x33_0) / (r32 - r12)) / (
            (r33 - r23) / (r32 - r22) - (r33 - r13) / (r32 - r12)
        )
        hg1_33 = x33_0 - r12 * ha1_33 - r13 * hb1_33

        return jnp.array(
            [
                hg1_11 + xr2 * ha1_11 + xr3 * hb1_11,
                hg1_13 + xr2 * ha1_13 + xr3 * hb1_13,
                hg1_33 + xr2 * ha1_33 + xr3 * hb1_33,
            ]
        )

    return interpolation_small_r


def _make_interpolation_large_r(interpolator_fn):
    @jit
    def interpolation_large_r(grid_x, grid_nu, grid_er, database):
        hr0 = (grid_x - database.rnm2) * (grid_x - database.rnm1) / (
            (database.rnm3 - database.rnm2) * (database.rnm3 - database.rnm1)
        )
        hr1 = (grid_x - database.rnm3) * (grid_x - database.rnm1) / (
            (database.rnm2 - database.rnm3) * (database.rnm2 - database.rnm1)
        )
        hr2 = (grid_x - database.rnm3) * (grid_x - database.rnm2) / (
            (database.rnm1 - database.rnm3) * (database.rnm1 - database.rnm2)
        )
        x11_m3, x13_m3, x33_m3 = interpolator_fn(-3, grid_nu, grid_er, database)
        x11_m2, x13_m2, x33_m2 = interpolator_fn(-2, grid_nu, grid_er, database)
        x11_m1, x13_m1, x33_m1 = interpolator_fn(-1, grid_nu, grid_er, database)
        return jnp.array(
            [
                hr0 * x11_m3 + hr1 * x11_m2 + hr2 * x11_m1,
                hr0 * x13_m3 + hr1 * x13_m2 + hr2 * x13_m1,
                hr0 * x33_m3 + hr1 * x33_m2 + hr2 * x33_m1,
            ]
        )

    return interpolation_large_r


def _make_interpolation_mid_r(interpolator_fn):
    @jit
    def interpolation_mid_r(grid_x, grid_nu, grid_er, database):
        arr = grid_x - database.rho[1:-1] * database.a_b
        index = jnp.argmax(jnp.where(arr <= 0, arr, -jnp.inf)) + 1
        idx0 = index - 2
        idx1 = index - 1
        idx2 = index
        idx3 = index + 1
        ridx0 = database.a_b * database.rho.at[idx0].get()
        ridx1 = database.a_b * database.rho.at[idx1].get()
        ridx2 = database.a_b * database.rho.at[idx2].get()
        ridx3 = database.a_b * database.rho.at[idx3].get()
        hr0 = (grid_x - ridx1) * (grid_x - ridx2) * (grid_x - ridx3) / (
            (ridx0 - ridx1) * (ridx0 - ridx2) * (ridx0 - ridx3)
        )
        hr1 = (grid_x - ridx0) * (grid_x - ridx2) * (grid_x - ridx3) / (
            (ridx1 - ridx0) * (ridx1 - ridx2) * (ridx1 - ridx3)
        )
        hr2 = (grid_x - ridx0) * (grid_x - ridx1) * (grid_x - ridx3) / (
            (ridx2 - ridx0) * (ridx2 - ridx1) * (ridx2 - ridx3)
        )
        hr3 = (grid_x - ridx0) * (grid_x - ridx1) * (grid_x - ridx2) / (
            (ridx3 - ridx0) * (ridx3 - ridx1) * (ridx3 - ridx2)
        )
        x11_idx0, x13_idx0, x33_idx0 = interpolator_fn(idx0, grid_nu, grid_er, database)
        x11_idx1, x13_idx1, x33_idx1 = interpolator_fn(idx1, grid_nu, grid_er, database)
        x11_idx2, x13_idx2, x33_idx2 = interpolator_fn(idx2, grid_nu, grid_er, database)
        x11_idx3, x13_idx3, x33_idx3 = interpolator_fn(idx3, grid_nu, grid_er, database)
        return jnp.array(
            [
                hr0 * x11_idx0 + hr1 * x11_idx1 + hr2 * x11_idx2 + hr3 * x11_idx3,
                hr0 * x13_idx0 + hr1 * x13_idx1 + hr2 * x13_idx2 + hr3 * x13_idx3,
                hr0 * x33_idx0 + hr1 * x33_idx1 + hr2 * x33_idx2 + hr3 * x33_idx3,
            ]
        )

    return interpolation_mid_r


interpolation_small_r_generic = _make_interpolation_small_r(interpolator_nu_Er_general)
interpolation_mid_r_generic = _make_interpolation_mid_r(interpolator_nu_Er_general)
interpolation_large_r_generic = _make_interpolation_large_r(interpolator_nu_Er_general)

interpolation_small_r_ntss = _make_interpolation_small_r(interpolator_nu_Er_general_NTSS)
interpolation_mid_r_ntss = _make_interpolation_mid_r(interpolator_nu_Er_general_NTSS)
interpolation_large_r_ntss = _make_interpolation_large_r(interpolator_nu_Er_general_NTSS)


def _make_get_Dij(small_r_fn, mid_r_fn, large_r_fn):
    @jit
    def get_Dij_impl(grid_x, grid_nu, grid_er, database):
        grid_nu_internal = jnp.log10(jnp.maximum(1.0e-12, grid_nu))
        grid_er_internal = jnp.select(
            condlist=[grid_x <= database.low_limit_r, grid_x > database.low_limit_r],
            choicelist=[
                jnp.log10(database.Er_lower_limit),
                jnp.log10(jnp.maximum(database.Er_lower_limit, jnp.abs(grid_er / grid_x))),
            ],
            default=0,
        )
        branch = jnp.where(grid_x < database.r1_lim, 0, jnp.where(grid_x < database.rmn2_lim, 1, 2))
        return jax.lax.switch(
            branch,
            (
                lambda _: small_r_fn(grid_x, grid_nu_internal, grid_er_internal, database),
                lambda _: mid_r_fn(grid_x, grid_nu_internal, grid_er_internal, database),
                lambda _: large_r_fn(grid_x, grid_nu_internal, grid_er_internal, database),
            ),
            operand=None,
        )

    return get_Dij_impl


get_Dij_generic = _make_get_Dij(
    interpolation_small_r_generic,
    interpolation_mid_r_generic,
    interpolation_large_r_generic,
)

get_Dij_ntss = _make_get_Dij(
    interpolation_small_r_ntss,
    interpolation_mid_r_ntss,
    interpolation_large_r_ntss,
)


def get_Dij_alt(grid_x, grid_nu, grid_er, database):
    xg = jnp.zeros(3)
    xg11 = interpax.Interpolator3D(database.rho * database.a_b, database.nu_log, database.Er_list[:], database.D11_log[:, :, :], extrap=True)(
        grid_x, jnp.log10(grid_nu), jnp.abs(grid_er)
    )
    xg13 = interpax.Interpolator3D(database.rho * database.a_b, database.nu_log, database.Er_list[:], database.D13[:, :, :], extrap=True)(
        grid_x, jnp.log10(grid_nu), jnp.abs(grid_er)
    )
    xg33 = interpax.Interpolator3D(database.rho * database.a_b, database.nu_log, database.Er_list[:], database.D33[:, :, :], extrap=True)(
        grid_x, jnp.log10(grid_nu), jnp.abs(grid_er)
    )
    xg = xg.at[0].set(xg11)
    xg = xg.at[1].set(xg13)
    xg = xg.at[2].set(xg33)
    return xg


get_Dij = get_Dij_generic
