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
