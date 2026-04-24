import jax.numpy as jnp
import jax
from jax import config
# to use higher precision
config.update("jax_enable_x64", True)
from jax import jit
import interpax


def _interp1d(x, y, value):
    return interpax.Interpolator1D(x, y, extrap=False)(value)


def _er_curve_interp_d11(index, nu_idx, grid_Er, database):
    er_grid = database.Er_list[index, :]
    values = database.D11_log[index, nu_idx, :]
    return jax.lax.cond(
        grid_Er <= er_grid[0],
        lambda _: values[0],
        lambda _: jax.lax.cond(
            grid_Er > er_grid[-1],
            lambda __: database.D11_lower_limit,
            lambda __: _interp1d(er_grid, values, grid_Er),
            operand=None,
        ),
        operand=None,
    )


def _er_curve_interp_d13(index, nu_idx, grid_Er, database):
    er_grid = database.Er_list[index, :]
    values = database.D13[index, nu_idx, :]
    return jax.lax.cond(
        grid_Er <= er_grid[0],
        lambda _: values[0],
        lambda _: jax.lax.cond(
            grid_Er > er_grid[-1],
            lambda __: jnp.array(0.0, dtype=values.dtype),
            lambda __: _interp1d(er_grid, values, grid_Er),
            operand=None,
        ),
        operand=None,
    )


def _er_curve_interp_d33(index, nu_idx, grid_Er, database):
    er_grid = database.Er_list[index, :]
    values = database.D33[index, nu_idx, :]
    return jax.lax.cond(
        grid_Er <= er_grid[0],
        lambda _: values[0],
        lambda _: jax.lax.cond(
            grid_Er > er_grid[-1],
            lambda __: values[-1],
            lambda __: _interp1d(er_grid, values, grid_Er),
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
                _interp1d(nu_grid, database.D11_log[index, :, 0], grid_nu),
                _interp1d(nu_grid, database.D13[index, :, 0], grid_nu),
                _interp1d(nu_grid, database.D33[index, :, 0], grid_nu),
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


def _interp_low_nu_finite(index, grid_nu, grid_Er, database):
    er_shift = (database.nu_log[0] - grid_nu) / 3.0 + grid_Er
    d11 = jax.lax.cond(
        er_shift > database.Er_list[index, -1],
        lambda _: database.D11_lower_limit,
        lambda _: _interp1d(database.Er_list[index, :], database.D11_log[index, 0, :], er_shift),
        operand=None,
    )
    d13 = _er_curve_interp_d13(index, 0, grid_Er, database)
    d33 = _er_curve_interp_d33(index, 0, grid_Er, database)
    return jnp.array([d11, d13, d33])


def _interp_high_nu_finite(index, grid_nu, grid_Er, database):
    return jnp.array(
        [
            interpax.Interpolator2D(database.nu_log, database.Er_list[index, :], database.D11_log[index, :, :], extrap=True)(grid_nu, grid_Er),
            interpax.Interpolator2D(database.nu_log, database.Er_list[index, :], database.D13[index, :, :], extrap=True)(grid_nu, grid_Er),
            interpax.Interpolator2D(database.nu_log, database.Er_list[index, :], database.D33[index, :, :], extrap=True)(grid_nu, grid_Er),
        ]
    )


def _interp_finite_row(index, nu_idx, grid_Er, database):
    return jnp.array(
        [
            _er_curve_interp_d11(index, nu_idx, grid_Er, database),
            _er_curve_interp_d13(index, nu_idx, grid_Er, database),
            _er_curve_interp_d33(index, nu_idx, grid_Er, database),
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
    xhat = (x - xs[1]) / dx
    ha = 3.0 * (ys[2] - ys[1]) - (2.0 * dg_l + dg_u) * dx
    hb = -2.0 * (ys[2] - ys[1]) + (dg_l + dg_u) * dx
    return ys[1] + xhat * (dg_l * dx + xhat * (ha + xhat * hb))


def _no_data_values(index, nu_idx, database):
    return jnp.array([database.D11_lower_limit, 0.0, database.D33[index, nu_idx, -1]])


def _interp_mid_finite_3(index, grid_nu, grid_Er, database, start_idx):
    xs = database.nu_log[start_idx : start_idx + 3]
    rows = jnp.stack([_interp_finite_row(index, start_idx + i, grid_Er, database) for i in range(3)], axis=0)
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


def _interp_mid_finite_4(index, grid_nu, grid_Er, database):
    arr = grid_nu - database.nu_log[1:-1]
    index_nu = jnp.argmax(jnp.where(arr <= 0, arr, -jnp.inf)) + 1
    start_idx = index_nu - 1
    xs = jax.lax.dynamic_slice(database.nu_log, (start_idx,), (4,))
    offsets = jnp.arange(4, dtype=start_idx.dtype)
    rows = jax.vmap(lambda off: _interp_finite_row(index, start_idx + off, grid_Er, database))(offsets)
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


def interpolator_nu_Er_general_NTSS(index, grid_nu, grid_Er, database):
    nu_grid = database.nu_log
    finite_er = grid_Er > database.Er_lower_limit_log
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
            lambda _: _interp_zero_efield(index, nu_grid[0] - 1.0, database),
            lambda _: _interp_zero_efield(index, grid_nu, database),
            lambda _: _interp_zero_efield(index, nu_grid[-1] + 1.0, database),
            lambda _: _interp_low_nu_finite(index, grid_nu, grid_Er, database),
            lambda _: _interp_mid_finite_3(index, grid_nu, grid_Er, database, 0),
            lambda _: _interp_mid_finite_3(index, grid_nu, grid_Er, database, nu_grid.shape[0] - 3),
            lambda _: _interp_mid_finite_4(index, grid_nu, grid_Er, database),
            lambda _: _interp_high_nu_finite(index, grid_nu, grid_Er, database),
        ),
        operand=None,
    )





@jit
#####Interpolators, which should go to the _interpolators.py
def interpolator_nu_Er_general(index,grid_nu,grid_Er,database):
    x11=interpax.Interpolator2D(database.nu_log,database.Er_list[index,:],database.D11_log[index,:,:],extrap=True)(grid_nu,grid_Er)
    x13=interpax.Interpolator2D(database.nu_log,database.Er_list[index,:],database.D13[index,:,:],extrap=True)(grid_nu,grid_Er)
    x33=interpax.Interpolator2D(database.nu_log,database.Er_list[index,:],database.D33[index,:,:],extrap=True)(grid_nu,grid_Er)
    return jnp.array([x11, x13, x33])


@jit
def interpolator_nu_Er_dispatch(index, grid_nu, grid_Er, database):
    return jax.lax.cond(
        database.interp_mode == 1,
        lambda _: interpolator_nu_Er_general_NTSS(index, grid_nu, grid_Er, database),
        lambda _: interpolator_nu_Er_general(index, grid_nu, grid_Er, database),
        operand=None,
    )

@jit
def interpolation_small_r(grid_x,grid_nu,grid_Er,database):
    xg=jnp.zeros(3)    
    xr2=jnp.power(grid_x,2)
    xr3=jnp.power(grid_x,3)
    r12 = jnp.power(database.r1,2)
    r22 = jnp.power(database.r2,2)
    r32 = jnp.power(database.r3,2)
    r13 = jnp.power(database.r1,3)
    r23 = jnp.power(database.r2,3)
    r33 = jnp.power(database.r3,3)
    x11_0,x13_0,x33_0=interpolator_nu_Er_dispatch(0,grid_nu,grid_Er,database)
    x11_1,x13_1,x33_1=interpolator_nu_Er_dispatch(1,grid_nu,grid_Er,database)
    x11_2,x13_2,x33_2=interpolator_nu_Er_dispatch(2,grid_nu,grid_Er,database)
    ha1_11 = ((x11_2-x11_1)/(r33-r23)-(x11_2-x11_0)/(r33-r13))/((r32-r22)/(r33-r23)-(r32-r12)/(r33-r13))
    hb1_11 = ((x11_2-x11_1)/(r32-r22)-(x11_2-x11_0)/(r32-r12))/((r33-r23)/(r32-r22)-(r33-r13)/(r32-r12))
    hg1_11 = x11_0-r12*ha1_11-r13*hb1_11
    #13
    ha1_13 = ((x13_2-x13_1)/(r33-r23)-(x13_2-x13_0)/(r33-r13))/((r32-r22)/(r33-r23)-(r32-r12)/(r33-r13))
    hb1_13 = ((x13_2-x13_1)/(r32-r22)-(x13_2-x13_0)/(r32-r12))/((r33-r23)/(r32-r22)-(r33-r13)/(r32-r12))
    hg1_13 = x13_0-r12*ha1_13-r13*hb1_13
    #33
    ha1_33 = ((x33_2-x33_1)/(r33-r23)-(x33_2-x33_0)/(r33-r13))/((r32-r22)/(r33-r23)-(r32-r12)/(r33-r13))
    hb1_33 = ((x33_2-x33_1)/(r32-r22)-(x33_2-x33_0)/(r32-r12))/((r33-r23)/(r32-r22)-(r33-r13)/(r32-r12))
    hg1_33 = x33_0-r12*ha1_33-r13*hb1_33
    #Final output
    xg11  = hg1_11+xr2*ha1_11+xr3*hb1_11
    xg13  = hg1_13+xr2*ha1_13+xr3*hb1_13
    xg33  = hg1_33+xr2*ha1_33+xr3*hb1_33   
    xg=xg.at[0].set(xg11)
    xg=xg.at[1].set(xg13)
    xg=xg.at[2].set(xg33)
    return xg#xg11,xg13,xg33

@jit
def interpolation_large_r(grid_x,grid_nu,grid_Er,database):
    xg=jnp.zeros(3)    
    hr0 = (grid_x-database.rnm2)*(grid_x-database.rnm1)/((database.rnm3-database.rnm2)*(database.rnm3-database.rnm1))
    hr1 = (grid_x-database.rnm3)*(grid_x-database.rnm1)/((database.rnm2-database.rnm3)*(database.rnm2-database.rnm1))
    hr2 = (grid_x-database.rnm3)*(grid_x-database.rnm2)/((database.rnm1-database.rnm3)*(database.rnm1-database.rnm2))
    x11_m3,x13_m3,x33_m3=interpolator_nu_Er_dispatch(-3,grid_nu,grid_Er,database)
    x11_m2,x13_m2,x33_m2=interpolator_nu_Er_dispatch(-2,grid_nu,grid_Er,database)
    x11_m1,x13_m1,x33_m1=interpolator_nu_Er_dispatch(-1,grid_nu,grid_Er,database)
    xg11  = hr0*x11_m3+hr1*x11_m2+hr2*x11_m1
    xg13  = hr0*x13_m3+hr1*x13_m2+hr2*x13_m1
    xg33  = hr0*x33_m3+hr1*x33_m2+hr2*x33_m1 
    xg=xg.at[0].set(xg11)
    xg=xg.at[1].set(xg13)
    xg=xg.at[2].set(xg33)
    return xg#xg11,xg13,xg33

@jit
def interpolation_mid_r(grid_x,grid_nu,grid_Er,database):
    xg=jnp.zeros(3)
    arr=grid_x-database.rho[1:-1]*database.a_b
    index = jnp.argmax(jnp.where(arr <= 0, arr, -jnp.inf))+1
    idx0=index-2
    idx1=index-1
    idx2=index
    idx3=index+1
    ridx0=database.a_b*database.rho.at[idx0].get()
    ridx1=database.a_b*database.rho.at[idx1].get()
    ridx2=database.a_b*database.rho.at[idx2].get()
    ridx3=database.a_b*database.rho.at[idx3].get()
    #jax.debug.print("Pe {Pe} ", Pe=ind)
    hr0 = (grid_x-ridx1)*(grid_x-ridx2)*(grid_x-ridx3)/((ridx0-ridx1)*(ridx0-ridx2)*(ridx0-ridx3))
    hr1 = (grid_x-ridx0)*(grid_x-ridx2)*(grid_x-ridx3)/((ridx1-ridx0)*(ridx1-ridx2)*(ridx1-ridx3))
    hr2 = (grid_x-ridx0)*(grid_x-ridx1)*(grid_x-ridx3)/((ridx2-ridx0)*(ridx2-ridx1)*(ridx2-ridx3))
    hr3 = (grid_x-ridx0)*(grid_x-ridx1)*(grid_x-ridx2)/((ridx3-ridx0)*(ridx3-ridx1)*(ridx3-ridx2))
    x11_idx0,x13_idx0,x33_idx0=interpolator_nu_Er_dispatch(idx0,grid_nu,grid_Er,database)
    x11_idx1,x13_idx1,x33_idx1=interpolator_nu_Er_dispatch(idx1,grid_nu,grid_Er,database)
    x11_idx2,x13_idx2,x33_idx2=interpolator_nu_Er_dispatch(idx2,grid_nu,grid_Er,database)
    x11_idx3,x13_idx3,x33_idx3=interpolator_nu_Er_dispatch(idx3,grid_nu,grid_Er,database) 
    xg11  = hr0*x11_idx0+hr1*x11_idx1+hr2*x11_idx2+hr3*x11_idx3
    xg13  = hr0*x13_idx0+hr1*x13_idx1+hr2*x13_idx2+hr3*x13_idx3
    xg33  = hr0*x33_idx0+hr1*x33_idx1+hr2*x33_idx2+hr3*x33_idx3
    xg=xg.at[0].set(xg11)
    xg=xg.at[1].set(xg13)
    xg=xg.at[2].set(xg33)
    return xg#xg11,xg13,xg33


#def get_Dij(grid_x, grid_nu, grid_Er):
#  return jnp.select(condlist=[grid_x < a_b*rho[1], (grid_x >= a_b*rho[1] ) & (grid_x < a_b*rho[-2]), grid_x >= a_b*rho[-2] ],
#    choicelist=[interpolation_small_r(grid_x,grid_nu,grid_Er) ,interpolation_mid_r(grid_x,grid_nu,grid_Er)  , interpolation_large_r(grid_x,grid_nu,grid_Er) ],default=0)


def get_Dij_alt(grid_x, grid_nu, grid_Er,database):
    xg=jnp.zeros(3)
    #grid_nu_internal=jnp.log10(grid_nu)
    #grid_Er_internal=jnp.abs(grid_Er)
    #xg=jnp.select(condlist=[grid_x < r1_lim, (grid_x>=r1_lim) & (grid_x<rmn2_lim), grid_x >= rmn2_lim],
    #                  choicelist=[interpolation_small_r(grid_x,grid_nu_internal,grid_Er_internal) ,
    #                              interpolation_mid_r(grid_x,grid_nu_internal,grid_Er_internal) ,
    #                              interpolation_large_r(grid_x,grid_nu_internal,grid_Er_internal) ],default=0)
    xg11=interpax.Interpolator3D(database.rho*database.a_b,database.nu_log,database.Er_list[:],database.D11_log[:,:,:],extrap=True)(grid_x,jnp.log10(grid_nu),jnp.abs(grid_Er))
    xg13=interpax.Interpolator3D(database.rho*database.a_b,database.nu_log,database.Er_list[:],database.D13[:,:,:],extrap=True)(grid_x,jnp.log10(grid_nu),jnp.abs(grid_Er))
    xg33=interpax.Interpolator3D(database.rho*database.a_b,database.nu_log,database.Er_list[:],database.D33[:,:,:],extrap=True)(grid_x,jnp.log10(grid_nu),jnp.abs(grid_Er))
    xg=xg.at[0].set(xg11)
    xg=xg.at[1].set(xg13)
    xg=xg.at[2].set(xg33)
    return xg
    #return interpax.Interpolator3D(rho*a_b,nu_log,Er_list,D11_log,extrap=True)(grid_x,grid_nu_internal,grid_Er_internal)

@jit
def get_Dij(grid_x, grid_nu, grid_Er,database):
    grid_nu_internal=jnp.log10(jnp.maximum(1.e-12,grid_nu))
    grid_Er_internal=jnp.select(condlist=[grid_x<=database.low_limit_r,grid_x>database.low_limit_r], 
                              choicelist=[jnp.log10(database.Er_lower_limit),jnp.log10(jnp.maximum(database.Er_lower_limit,jnp.abs(grid_Er/grid_x)))],default=0)
    branch = jnp.where(
        grid_x < database.r1_lim,
        0,
        jnp.where(grid_x < database.rmn2_lim, 1, 2),
    )
    return jax.lax.switch(
        branch,
        (
            lambda _: interpolation_small_r(grid_x, grid_nu_internal, grid_Er_internal, database),
            lambda _: interpolation_mid_r(grid_x, grid_nu_internal, grid_Er_internal, database),
            lambda _: interpolation_large_r(grid_x, grid_nu_internal, grid_Er_internal, database),
        ),
        operand=None,
    )

