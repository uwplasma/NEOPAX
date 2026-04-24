import jax
import jax.numpy as jnp
from jax import config

# to use higher precision
config.update("jax_enable_x64", True)


def _lagrange3(x, x0, x1, x2, y0, y1, y2):
    h0 = (x - x1) * (x - x2) / ((x0 - x1) * (x0 - x2))
    h1 = (x - x0) * (x - x2) / ((x1 - x0) * (x1 - x2))
    h2 = (x - x0) * (x - x1) / ((x2 - x0) * (x2 - x1))
    return h0 * y0 + h1 * y1 + h2 * y2


def _butland4(x, x0, x1, x2, x3, y0, y1, y2, y3, gmix):
    dhl0 = (x1 - x2) / ((x0 - x1) * (x0 - x2))
    dhl1 = (2.0 * x1 - x0 - x2) / ((x1 - x0) * (x1 - x2))
    dhl2 = (x1 - x0) / ((x2 - x0) * (x2 - x1))
    dhu0 = (x2 - x3) / ((x1 - x2) * (x1 - x3))
    dhu1 = (2.0 * x2 - x1 - x3) / ((x2 - x1) * (x2 - x3))
    dhu2 = (x2 - x1) / ((x3 - x1) * (x3 - x2))
    dyl = dhl0 * y0 + dhl1 * y1 + dhl2 * y2
    dyu = dhu0 * y1 + dhu1 * y2 + dhu2 * y3
    same_sign = dyl * dyu > 0.0
    dyl = jnp.where(same_sign, dyl, gmix * dyl)
    dyu = jnp.where(same_sign, dyu, gmix * dyu)
    dx = x2 - x1
    xn = (x - x1) / dx
    ha = 3.0 * (y2 - y1) - (2.0 * dyl + dyu) * dx
    hb = -2.0 * (y2 - y1) + (dyl + dyu) * dx
    return y1 + xn * (dyl * dx + xn * (ha + xn * hb))


def _interp_linear(x, xs, ys, n):
    xs = xs[:n]
    ys = ys[:n]
    idx = jnp.clip(jnp.searchsorted(xs, x, side="right") - 1, 0, n - 2)
    x0 = xs[idx]
    x1 = xs[idx + 1]
    y0 = ys[idx]
    y1 = ys[idx + 1]
    w = jnp.where(x1 > x0, (x - x0) / (x1 - x0), 0.0)
    return y0 * (1.0 - w) + y1 * w


def _interp_linear_segment(x, xs, ys, start, count):
    size = xs.shape[0]
    idxs = jnp.arange(size)
    stop = start + count
    valid = (idxs >= start) & (idxs < stop)
    pair_valid = valid[:-1] & valid[1:]
    cand = jnp.where(pair_valid & (x >= xs[:-1]), idxs[:-1], -1)
    idx = jnp.maximum(jnp.max(cand), start)
    idx = jnp.minimum(idx, jnp.maximum(stop - 2, start))
    x0 = xs[idx]
    x1 = xs[idx + 1]
    y0 = ys[idx]
    y1 = ys[idx + 1]
    w = jnp.where(x1 > x0, (x - x0) / (x1 - x0), 0.0)
    return y0 * (1.0 - w) + y1 * w


def _eval_radius_node(ir, xnu, xer, efield, db):
    icu0 = db.nval0[ir]
    icue = db.icnur[ir]
    xs0 = db.axnu0[ir]
    g110 = db.ag110[ir]
    g130 = db.ag130[ir]
    g330 = db.ag330[ir]
    avgs_full = db.axnuar[ir]
    aref_full = db.aref[ir]
    ag11_full = db.ag11[ir]
    ag13_full = db.ag13[ir]
    ag33_full = db.ag33[ir]

    def zero_field():
        last0 = icu0 - 1
        high = jnp.asarray([g110[last0] - xs0[last0] + xnu, db.x13_l, g330[last0]])
        low = jnp.asarray([g110[0] + xs0[0] - xnu, g130[0], g330[0]])
        mid = jnp.asarray([
            _interp_linear_segment(xnu, xs0, g110, 0, icu0),
            _interp_linear_segment(xnu, xs0, g130, 0, icu0),
            _interp_linear_segment(xnu, xs0, g330, 0, icu0),
        ])
        return jnp.where(
            xnu > xs0[last0],
            high,
            jnp.where(xnu < xs0[0], low, mid),
        )

    def group_eval(ic):
        start = db.inulr[ir, ic]
        n = db.inugr[ir, ic] - start + 1
        first = aref_full[start]
        last = aref_full[start + n - 1]
        x11 = jnp.where(
            xer <= first,
            ag11_full[start],
            jnp.where(xer > last, db.x11_l, _interp_linear_segment(xer, aref_full, ag11_full, start, n)),
        )
        x13 = jnp.where(
            xer <= first,
            ag13_full[start],
            jnp.where(xer > last, db.x13_l, _interp_linear_segment(xer, aref_full, ag13_full, start, n)),
        )
        x33 = jnp.where(
            xer <= first,
            ag33_full[start],
            jnp.where(xer > last, ag33_full[start + n - 1], _interp_linear_segment(xer, aref_full, ag33_full, start, n)),
        )
        return jnp.asarray([x11, x13, x33])

    def positive_field():
        def low_branch():
            ic = 0
            start = db.inulr[ir, ic]
            n = db.inugr[ir, ic] - start + 1
            first = aref_full[start]
            last = aref_full[start + n - 1]
            xer_h = (avgs_full[0] - xnu) / 3.0 + xer
            x11 = jnp.where(xer_h > last, db.x11_l, _interp_linear_segment(xer_h, aref_full, ag11_full, start, n))
            x13 = jnp.where(xer > last, db.x13_l, _interp_linear_segment(jnp.maximum(xer, first), aref_full, ag13_full, start, n))
            x33 = jnp.where(xer > last, ag33_full[start + n - 1], _interp_linear_segment(jnp.maximum(xer, first), aref_full, ag33_full, start, n))
            return jnp.asarray([x11, x13, x33])

        def high_branch():
            return group_eval(icue - 1)

        def middle_branch():
            ic_l = jnp.where(
                xnu <= avgs_full[1],
                0,
                jnp.where(
                    xnu >= avgs_full[icue - 2],
                    icue - 3,
                    jnp.sum(((jnp.arange(avgs_full.shape[0]) < icue) & (xnu >= avgs_full)).astype(jnp.int32)) - 2,
                ),
            )
            noic = jnp.where((xnu <= avgs_full[1]) | (xnu >= avgs_full[icue - 2]), 3, 4)
            ic_idx = jnp.minimum(ic_l + jnp.arange(4, dtype=jnp.int32), icue - 1)
            atc = jax.vmap(group_eval)(ic_idx)

            def lag3():
                x0 = avgs_full[ic_l]
                x1 = avgs_full[ic_l + 1]
                x2 = avgs_full[ic_l + 2]
                y11 = _lagrange3(xnu, x0, x1, x2, atc[0, 0], atc[1, 0], atc[2, 0])
                y13 = _lagrange3(xnu, x0, x1, x2, atc[0, 1], atc[1, 1], atc[2, 1])
                y33 = _lagrange3(xnu, x0, x1, x2, atc[0, 2], atc[1, 2], atc[2, 2])
                return jnp.asarray([y11, y13, y33])

            def but4():
                x0 = avgs_full[ic_l]
                x1 = avgs_full[ic_l + 1]
                x2 = avgs_full[ic_l + 2]
                x3 = avgs_full[ic_l + 3]
                y11 = _butland4(xnu, x0, x1, x2, x3, atc[0, 0], atc[1, 0], atc[2, 0], atc[3, 0], db.gmix_nu)
                y13 = _butland4(xnu, x0, x1, x2, x3, atc[0, 1], atc[1, 1], atc[2, 1], atc[3, 1], db.gmix_nu)
                y33 = _butland4(xnu, x0, x1, x2, x3, atc[0, 2], atc[1, 2], atc[2, 2], atc[3, 2], db.gmix_nu)
                return jnp.asarray([y11, y13, y33])

            return jax.lax.cond(noic == 3, lag3, but4)

        return jax.lax.cond(
            xnu < avgs_full[0],
            low_branch,
            lambda: jax.lax.cond(xnu > avgs_full[icue - 1], high_branch, middle_branch),
        )

    return jax.lax.cond(efield <= db.xref_l, zero_field, positive_field)


@jax.jit
def get_Dij_ntss_preprocessed(grid_x, grid_nu, grid_Er, db):
    xri = grid_x
    xnu = jnp.log10(jnp.maximum(1.0e-12, grid_nu))
    efield = jnp.where(xri <= 1.0e-30, 0.0, jnp.abs(grid_Er / xri))
    xer = jnp.log10(jnp.maximum(db.xref_l, efield))
    arr = db.r_grid
    nr = arr.shape[0]

    exact_idx = jnp.argmin(jnp.abs(xri - arr))
    is_exact = jnp.abs(xri - arr[exact_idx]) <= db.del_r

    nil = jnp.where(
        xri < arr[1],
        0,
        jnp.where(
            xri >= arr[nr - 2],
            nr - 3,
            jnp.searchsorted(arr[2:nr - 1], xri, side="left") + 0,
        ),
    )
    noi = jnp.where((xri < arr[1]) | (xri >= arr[nr - 2]), 3, 4)
    nil = jnp.where(is_exact, exact_idx, nil)
    noi = jnp.where(is_exact, 1, noi)

    stencil_idx = jnp.minimum(nil + jnp.arange(4, dtype=jnp.int32), nr - 1)
    atc = jax.vmap(lambda ir: _eval_radius_node(ir, xnu, xer, efield, db))(stencil_idx)

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
            ha = ((v2 - v1) / (r33 - r23) - (v2 - v0) / (r33 - r13)) / ((r32 - r22) / (r33 - r23) - (r32 - r12) / (r33 - r13))
            hb = ((v2 - v1) / (r32 - r22) - (v2 - v0) / (r32 - r12)) / ((r33 - r23) / (r32 - r22) - (r33 - r13) / (r32 - r12))
            hg = v0 - r12 * ha - r13 * hb
            return hg + xr2 * ha + xr3 * hb
        return jnp.asarray([comp(atc[0, 0], atc[1, 0], atc[2, 0]), comp(atc[0, 1], atc[1, 1], atc[2, 1]), comp(atc[0, 2], atc[1, 2], atc[2, 2])])

    def edge3():
        x0 = arr[nil]
        x1 = arr[nil + 1]
        x2 = arr[nil + 2]
        return jnp.asarray([
            _lagrange3(xri, x0, x1, x2, atc[0, 0], atc[1, 0], atc[2, 0]),
            _lagrange3(xri, x0, x1, x2, atc[0, 1], atc[1, 1], atc[2, 1]),
            _lagrange3(xri, x0, x1, x2, atc[0, 2], atc[1, 2], atc[2, 2]),
        ])

    def interior4():
        x0 = arr[nil]
        x1 = arr[nil + 1]
        x2 = arr[nil + 2]
        x3 = arr[nil + 3]
        return jnp.asarray([
            _butland4(xri, x0, x1, x2, x3, atc[0, 0], atc[1, 0], atc[2, 0], atc[3, 0], 1.0),
            _butland4(xri, x0, x1, x2, x3, atc[0, 1], atc[1, 1], atc[2, 1], atc[3, 1], 1.0),
            _butland4(xri, x0, x1, x2, x3, atc[0, 2], atc[1, 2], atc[2, 2], atc[3, 2], 1.0),
        ])

    return jax.lax.cond(
        noi == 1,
        exact,
        lambda: jax.lax.cond(
            xri < arr[1],
            small_r,
            lambda: jax.lax.cond((xri >= arr[nr - 2]) | (noi == 3), edge3, interior4),
        ),
    )
