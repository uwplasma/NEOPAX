import jax
import jax.numpy as jnp
from jax import config

# to use higher precision
config.update("jax_enable_x64", True)

SEGMENT_WIDTH = 16


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


def _segment_arrays(xs_full, ys_full, start):
    max_start = jnp.maximum(xs_full.shape[0] - SEGMENT_WIDTH, 0)
    start = jnp.clip(start, 0, max_start)
    xs = jax.lax.dynamic_slice_in_dim(xs_full, start, SEGMENT_WIDTH, axis=0)
    ys = jax.lax.dynamic_slice_in_dim(ys_full, start, SEGMENT_WIDTH, axis=0)
    return xs, ys


def _inpold_fixed(x, xs, ys, n, icl, fcl, icu, fcu, fmix):
    kcl = jnp.clip(icl, 0, 4)
    kcu = jnp.clip(icu, 0, 4)
    amix = jnp.clip(fmix, 0.0, 1.0)

    def case_n1():
        return ys[0]

    def case_n2():
        dy = (ys[1] - ys[0]) / (xs[1] - xs[0])
        return ys[0] + dy * (x - xs[0])

    def case_n3():
        left_ge3 = kcl >= 3
        right_ge3 = kcu >= 3
        a0 = jnp.where(left_ge3, ys[0], ys[0] - ys[1])
        a1 = jnp.where(left_ge3, 1.0, xs[0] - xs[1])
        a2 = jnp.where(left_ge3, xs[0] - xs[1], 0.5 * (xs[0] - xs[1]) ** 2)
        b0 = jnp.where(right_ge3, ys[2], ys[2] - ys[1])
        b1 = jnp.where(right_ge3, 1.0, xs[2] - xs[1])
        b2 = jnp.where(right_ge3, xs[2] - xs[1], 0.5 * (xs[2] - xs[1]) ** 2)
        det = a1 * b2 - a2 * b1
        g1 = jnp.where(jnp.abs(det) <= 1.0e-30, 0.0, (a0 * b2 - a2 * b0) / det)
        g2 = jnp.where(jnp.abs(det) <= 1.0e-30, 0.0, (a1 * b0 - a0 * b1) / det)

        def left():
            dy = g1 + g2 * (xs[0] - xs[1])
            return ys[0] + dy * (x - xs[0])

        def middle():
            gx = x - xs[1]
            return ys[1] + gx * (g1 + 0.5 * g2 * gx)

        def right():
            dy = g1 + g2 * (xs[2] - xs[1])
            return ys[2] + dy * (x - xs[2])

        return jax.lax.cond(x <= xs[0], left, lambda: jax.lax.cond(x <= xs[2], middle, right))

    def case_ngt3():
        n_eff = jnp.minimum(n, SEGMENT_WIDTH)
        valid_mask = jnp.arange(SEGMENT_WIDTH) < n_eff
        xs_mask = jnp.where(valid_mask, xs, jnp.inf)
        pos = jnp.searchsorted(xs_mask, x, side="left")

        def left_boundary():
            i1, i2, i3 = 0, 1, 2
            a0 = ys[i3] - ys[i2]
            a1 = xs[i3] - xs[i2]
            a2 = 0.5 * a1**2
            a3 = (2.0 / 3.0) * a1 * a2
            use_low_bc = kcl <= 2
            b0 = jnp.where(use_low_bc, ys[i1] - ys[i2], ys[i1])
            b1 = jnp.where(use_low_bc, xs[i1] - xs[i2], 1.0)
            b2 = jnp.where(use_low_bc, 0.5 * b1**2, xs[i1] - xs[i2])
            b3 = jnp.where(use_low_bc, (2.0 / 3.0) * b1 * b2, b2**2)

            def bc_vals():
                c0 = jnp.where(kcl == 0, fcl, jnp.where(kcl <= 2, fcl, fcl))
                c1 = jnp.where(kcl == 0, 1.0, 0.0)
                c2 = jnp.where(kcl == 0, b1, jnp.where(kcl == 1, 1.0, jnp.where(kcl == 2, 0.0, 1.0)))
                c3 = jnp.where(kcl == 0, b1**2, jnp.where(kcl == 1, 2.0 * b1, jnp.where(kcl == 2, 2.0, jnp.where(kcl == 3, 2.0 * b2, 2.0))))
                return c0, c1, c2, c3

            c0, c1, c2, c3 = bc_vals()
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
            i3 = jnp.minimum(jnp.maximum(pos, 2), n_eff - 1)
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
            dyl = jnp.where((yl - ys[i1]) * (yl - ys[i3]) >= 0.0, amix * dyl_raw, dyl_raw)

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
                dyu = jnp.where((yu - ys[i2]) * (yu - ys[i4]) >= 0.0, amix * dyu_raw, dyu_raw)
                return xl, yl, dyl, xu, yu, dyu

            def right_boundary():
                i2r = jnp.minimum(n_eff - 2, i3)
                i1r = i2r + 1
                i3r = i2r - 1
                a0r = ys[i3r] - ys[i2r]
                a1r = xs[i3r] - xs[i2r]
                a2r = 0.5 * a1r**2
                a3r = (2.0 / 3.0) * a1r * a2r
                use_up_bc = kcu <= 2
                b0r = jnp.where(use_up_bc, ys[i1r] - ys[i2r], ys[i1r])
                b1r = jnp.where(use_up_bc, xs[i1r] - xs[i2r], 1.0)
                b2r = jnp.where(use_up_bc, 0.5 * b1r**2, xs[i1r] - xs[i2r])
                b3r = jnp.where(use_up_bc, (2.0 / 3.0) * b1r * b2r, b2r**2)
                c0 = fcu
                c1 = jnp.where(kcu == 0, 1.0, 0.0)
                c2 = jnp.where(kcu == 0, b1r, jnp.where(kcu == 1, 1.0, jnp.where(kcu == 2, 0.0, 1.0)))
                c3 = jnp.where(kcu == 0, b1r**2, jnp.where(kcu == 1, 2.0 * b1r, jnp.where(kcu == 2, 2.0, jnp.where(kcu == 3, 2.0 * b2r, 2.0))))
                detr = a1r * (b2r * c3 - b3r * c2) - a2r * (b1r * c3 - b3r * c1) + a3r * (b1r * c2 - b2r * c1)
                g1r = jnp.where(jnp.abs(detr) <= 1.0e-30, 0.0, (a0r * (b2r * c3 - b3r * c2) - a2r * (b0r * c3 - b3r * c0) + a3r * (b0r * c2 - b2r * c0)) / detr)
                xu = xs[i2r]
                yu = ys[i2r]
                return xl, yl, dyl, xu, yu, g1r

            xl, yl, dyl2, xu, yu, dyu = jax.lax.cond(i3 < n_eff - 1, interior, right_boundary)
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

    return jax.lax.cond(n <= 1, case_n1, lambda: jax.lax.cond(n == 2, case_n2, lambda: jax.lax.cond(n == 3, case_n3, case_ngt3)))


def _inpold_segment(x, xs_full, ys_full, start, count, icl, fcl, icu, fcu, fmix):
    xs, ys = _segment_arrays(xs_full, ys_full, start)
    local_count = jnp.minimum(count, SEGMENT_WIDTH)
    return _inpold_fixed(x, xs, ys, local_count, icl, fcl, icu, fcu, fmix)


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
        fc_l = -1.0
        mid = jnp.asarray([
            _inpold_segment(xnu, xs0, g110, 0, icu0, 0, fc_l, 0, 1.0, db.gmix_nu),
            _inpold_segment(xnu, xs0, g130, 0, icu0, 1, 0.0, 1, 0.0, db.gmix_nu),
            _inpold_segment(xnu, xs0, g330, 0, icu0, 0, 0.0, 0, 0.0, db.gmix_nu),
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
            jnp.where(xer > last, db.x11_l, _inpold_segment(xer, aref_full, ag11_full, start, n, 0, 0.0, 1, 0.0, db.gmix_er)),
        )
        x13 = jnp.where(
            xer <= first,
            ag13_full[start],
            jnp.where(xer > last, db.x13_l, _inpold_segment(xer, aref_full, ag13_full, start, n, 0, 0.0, 1, 0.0, db.gmix_er)),
        )
        x33 = jnp.where(
            xer <= first,
            ag33_full[start],
            jnp.where(xer > last, ag33_full[start + n - 1], _inpold_segment(xer, aref_full, ag33_full, start, n, 0, 0.0, 1, 0.0, db.gmix_er)),
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
            x11 = jnp.where(xer_h > last, db.x11_l, _inpold_segment(xer_h, aref_full, ag11_full, start, n, 0, 0.0, 1, 0.0, db.gmix_er))
            x13 = jnp.where(xer > last, db.x13_l, _inpold_segment(xer, aref_full, ag13_full, start, n, 0, 0.0, 1, 0.0, db.gmix_er))
            x33 = jnp.where(xer > last, ag33_full[start + n - 1], _inpold_segment(xer, aref_full, ag33_full, start, n, 0, 0.0, 1, 0.0, db.gmix_er))
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
        def lagrange4(y0, y1, y2, y3):
            h0 = (xri - x1) * (xri - x2) * (xri - x3) / ((x0 - x1) * (x0 - x2) * (x0 - x3))
            h1 = (xri - x0) * (xri - x2) * (xri - x3) / ((x1 - x0) * (x1 - x2) * (x1 - x3))
            h2 = (xri - x0) * (xri - x1) * (xri - x3) / ((x2 - x0) * (x2 - x1) * (x2 - x3))
            h3 = (xri - x0) * (xri - x1) * (xri - x2) / ((x3 - x0) * (x3 - x1) * (x3 - x2))
            return h0 * y0 + h1 * y1 + h2 * y2 + h3 * y3
        return jnp.asarray([
            lagrange4(atc[0, 0], atc[1, 0], atc[2, 0], atc[3, 0]),
            lagrange4(atc[0, 1], atc[1, 1], atc[2, 1], atc[3, 1]),
            lagrange4(atc[0, 2], atc[1, 2], atc[2, 2], atc[3, 2]),
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
