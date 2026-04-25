import jax
import jax.numpy as jnp
from jax import config

# to use higher precision
config.update("jax_enable_x64", True)

SEGMENT_WIDTH = 16


def _dkfttc(cmul, eps, xkn, xio, xrm, b0):
    a1 = 0.9733
    a2 = 0.3899
    a3 = 0.06778
    a4 = -1.75
    a5 = -1.75
    b2 = 1.0340
    b3 = -0.6689
    b4 = 0.6666667
    b6 = 0.3333333
    c1 = -0.9665
    c2 = 1.75
    d0 = 4.0 / 3.0
    d1 = 3.4229
    d2 = -2.5766
    d3 = -0.6039
    d4 = 2.0 / 3.0
    d5 = -1.1776
    d6 = 0.6756
    d7 = 1.8436

    aio = jnp.maximum(jnp.abs(xio), 1.0e-30)
    amul = jnp.maximum(jnp.abs(cmul), 1.0e-30)
    akn = jnp.maximum(jnp.abs(xkn), 1.0e-30)
    eps = jnp.maximum(eps, 1.0e-30)
    rmul = xrm * amul
    xkn2 = akn**2
    g1 = jnp.sqrt(akn / eps) / aio
    g2 = eps * xkn2
    g3 = eps * xkn2 * aio
    g4 = eps * aio

    db0 = a1 * g1
    gb = 1.0 / (eps**b4 * akn * aio**b6)
    db = db0 * (1.0 + b3 * (akn * eps) ** 2) / (1.0 + b2 * gb * jnp.sqrt(rmul))
    dpl = a2 * g2 / rmul
    dps = a3 * g3 / rmul**2
    dbp = (db**a4 + dpl**a4) ** (1.0 / a4)
    d31 = (dbp**a5 + dps**a5) ** (1.0 / a5)

    d11n_ps = d0 * (akn / aio) ** 2 * (1.0 + d1 * eps**3.6 * (1.0 + d2 * aio**1.6) + d3 * eps**2 * (1.0 - xkn2))
    g33n = 1.0 + d5 * (eps * akn) ** d7 + d6 * eps**3 * aio**2.5
    d33n_ps = d4 * g33n
    d11n_ps = jnp.where(cmul > 0.0, d11n_ps + d0, d11n_ps)

    gc4 = (1.0 + c1 * (eps * akn) ** c2) * g4
    d11 = (d11n_ps + d31 / gc4) * amul
    d33 = (d33n_ps - d31 * gc4) / amul

    d11 = -d11 / (b0**2)
    d31 = d31 * xio / (aio * b0)
    d33 = -d33
    d33 = jnp.where(xkn > 0.0, d33 / g33n, d33)
    return d11, d31, d33


def _dkftte(cmul, efield, eps, xkn, xio, xrm, b0):
    pi = 3.1415927
    pa1 = 3.733333
    pa2 = 1.408160
    pa3 = 0.469388
    pa4 = 0.333333
    pa5 = 2.333333
    pa6 = 1.523808
    pa7 = 1.600000
    pa8 = 0.380952
    pa9 = 0.177777
    pb1 = 1.333333

    d11, d31, d33 = _dkfttc(cmul, eps, xkn, xio, xrm, b0)
    fac = eps * xio
    fac_abs = jnp.maximum(jnp.abs(fac), 1.0e-30)
    a = jnp.abs(efield / fac_abs)

    def corrected():
        amul = jnp.maximum(jnp.abs(cmul), 1.0e-30)
        d11_cl = -pb1 * amul / (b0**2)
        d11_wo = jnp.where(cmul > 0.0, d11 - d11_cl, d11)
        b = jnp.abs(cmul * xrm / jnp.maximum(xio, 1.0e-30))
        a2 = a**2
        b2 = b**2

        val11_a = pa1 * (1.0 - pa3 / b2) / b2
        val31_a = -(pa7 - (pa8 - pa9 / b2) / b2) / b2

        val11_b = pa1 * (1.0 - (b2 - pa2) / a2) / a2
        val31_b = -(pa7 - (pa8 - b2) / a2) / a2

        xln = jnp.log(((1.0 + a) ** 2 + b2) / ((1.0 - a) ** 2 + b2))

        def atg_general():
            return (jnp.arctan((1.0 - a) / b) + jnp.arctan((1.0 + a) / b)) / b

        def atg_large_a():
            return jnp.arctan(2.0 * b / (a2 + b2 - 1.0)) / b

        atg = jax.lax.cond(
            b > 1.0e-6,
            lambda: jax.lax.cond(a > 2.0, atg_large_a, atg_general),
            lambda: jnp.where(a < 1.0, pi / jnp.maximum(b, 1.0e-30), 0.0),
        )

        val11_c = 2.0 * (pa5 + 3.0 * a2 - b2 - a * (1.0 + a2 - b2) * xln) + (
            1.0 + a2 * (2.0 + a2) - b2 * (2.0 - b2 + 6.0 * a2)
        ) * atg
        val31_c = 2.0 * (pa4 + 3.0 * a2 - b2 - a * (a2 - b2) * xln) - (
            1.0 - a2**2 + 6.0 * a2 * b2 - b2**2
        ) * atg

        region_ab = (a2 + b2 >= 16.0) & (b >= 1.0)
        region_a = a >= 4.0
        val11 = jnp.where(region_ab, val11_a, jnp.where(region_a, val11_b, val11_c))
        val31 = jnp.where(region_ab, val31_a, jnp.where(region_a, val31_b, val31_c))

        atg0 = jnp.arctan(1.0 / b) / b
        val110 = jnp.where(
            b > 4.0,
            pa1 * (1.0 - pa3 / b2) / b2,
            jnp.where(
                b < 1.0e-3,
                2.0 * pa5 + pi / jnp.maximum(b, 1.0e-30),
                2.0 * (pa5 - b2 + (1.0 - b2 * (2.0 - b2)) * atg0),
            ),
        )
        val310 = jnp.where(
            b > 4.0,
            -(pa7 - (pa8 - pa9 / b2) / b2) / b2,
            jnp.where(
                b < 1.0e-3,
                2.0 * pa4 - pi / jnp.maximum(b, 1.0e-30),
                2.0 * (pa4 - b2 - (1.0 - b2**2) * atg0),
            ),
        )

        d11_corr = d11_wo * val11 / val110
        d31_corr = d31 * val31 / val310
        eres1 = pa4 * fac_abs / jnp.maximum(b, 1.0e-30)
        d11_corr = d11_corr / (1.0 + (efield / jnp.maximum(eres1, 1.0e-30)) ** 2)
        d11_corr = jnp.where(cmul > 0.0, d11_corr + d11_cl, d11_corr)
        return d11_corr, d31_corr, d33

    return jax.lax.cond(a <= 1.0e-6, lambda: (d11, d31, d33), corrected)


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


def _eval_group(ir, ic, xer, db, aref_full, ag11_full, ag13_full, ag33_full):
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


def _eval_zero_field_node(ir, xnu, db, xs0, g110, g130, g330, icu0, lc_fit, ag11_0_fit):
    last0 = icu0 - 1
    high = jnp.asarray([g110[last0] - xs0[last0] + xnu, db.x13_l, g330[last0]])
    low_g11 = jnp.where(
        lc_fit & (ag11_0_fit < 0.0),
        g110[0] - xs0[0] + xnu,
        g110[0] + xs0[0] - xnu,
    )
    low = jnp.asarray([low_g11, g130[0], g330[0]])
    fc_l = jnp.where(lc_fit & (ag11_0_fit < 0.0), 1.0, -1.0)
    mid = jnp.asarray(
        [
            _inpold_segment(xnu, xs0, g110, 0, icu0, 0, fc_l, 0, 1.0, db.gmix_nu),
            _inpold_segment(xnu, xs0, g130, 0, icu0, 1, 0.0, 1, 0.0, db.gmix_nu),
            _inpold_segment(xnu, xs0, g330, 0, icu0, 0, 0.0, 0, 0.0, db.gmix_nu),
        ]
    )
    return jnp.where(xnu > xs0[last0], high, jnp.where(xnu < xs0[0], low, mid))


def _eval_low_branch(ir, xnu, xer, efield, db, avgs_full, aref_full, ag11_full, ag13_full, ag33_full, lc_fit, ag11_0_fit, ag11_sq_fit, aefld_u_fit, aex_er_fit):
    start = db.inulr[ir, 0]
    n = db.inugr[ir, 0] - start + 1
    last = aref_full[start + n - 1]
    xer_h = (avgs_full[0] - xnu) / 3.0 + xer

    def old_sqrt_nu():
        return jnp.where(
            xer_h > last,
            db.x11_l,
            _inpold_segment(xer_h, aref_full, ag11_full, start, n, 0, 0.0, 1, 0.0, db.gmix_er),
        )

    def fit_branch():
        def tokamak_like():
            return jnp.where(
                xer_h > last,
                db.x11_l,
                _inpold_segment(xer_h, aref_full, ag11_full, start, n, 0, 0.0, 1, 0.0, db.gmix_er) + xnu - avgs_full[0],
            )

        def stellarator_like():
            def large_efield():
                return jnp.where(
                    xer > last,
                    db.x11_l,
                    _inpold_segment(xer, aref_full, ag11_full, start, n, 0, 0.0, 1, 0.0, db.gmix_er) + xnu - avgs_full[0],
                )

            def mixed_fit():
                xer_u = jnp.log10(jnp.maximum(db.xref_l, aefld_u_fit / jnp.maximum(db.r_grid[ir], 1.0e-30)))
                g11_u = _inpold_segment(xer_u, aref_full, ag11_full, start, n, 0, 0.0, 1, 0.0, db.gmix_er)
                d11ad = 10.0 ** (g11_u + xnu - avgs_full[0])
                xmul = 10.0 ** xnu
                arg0 = (xmul / jnp.maximum(jnp.abs(ag11_0_fit), 1.0e-30)) ** aex_er_fit
                argsq = (efield * jnp.sqrt(efield / jnp.maximum(xmul, 1.0e-30)) / jnp.maximum(jnp.abs(ag11_sq_fit), 1.0e-30)) ** aex_er_fit
                d11ft = (arg0 + argsq) ** (-1.0 / jnp.maximum(aex_er_fit, 1.0e-30))
                return jnp.log10(jnp.maximum(1.0e-30, d11ad + d11ft))

            return jax.lax.cond(efield >= aefld_u_fit, large_efield, mixed_fit)

        return jax.lax.cond(ag11_0_fit < 0.0, tokamak_like, stellarator_like)

    x11 = jax.lax.cond(lc_fit, fit_branch, old_sqrt_nu)
    x13 = jnp.where(xer > last, db.x13_l, _inpold_segment(xer, aref_full, ag13_full, start, n, 0, 0.0, 1, 0.0, db.gmix_er))
    x33 = jnp.where(xer > last, ag33_full[start + n - 1], _inpold_segment(xer, aref_full, ag33_full, start, n, 0, 0.0, 1, 0.0, db.gmix_er))
    return jnp.asarray([x11, x13, x33])


@jax.jit
def _eval_high_branch(ir, xnu, efield, db):
    cmul = 10.0 ** xnu
    xrm = db.xrm_fit[ir]
    eps = db.r_grid[ir] / jnp.maximum(xrm, 1.0e-30)
    d11, d13, d33 = _dkftte(-cmul, efield, eps, db.akn_fit[ir], db.air_fit[ir], xrm, 1.0)
    return jnp.asarray([jnp.log10(jnp.maximum(1.0e-30, jnp.abs(d11))), d13, jnp.abs(d33) * cmul])


def _eval_middle_branch(ir, xnu, xer, db, avgs_full, aref_full, ag11_full, ag13_full, ag33_full, icue):
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
    atc = jax.vmap(lambda ic: _eval_group(ir, ic, xer, db, aref_full, ag11_full, ag13_full, ag33_full))(ic_idx)
    no_data = jnp.asarray([db.x11_l, db.x13_l, ag33_full[db.inugr[ir, ic_l]]])

    def lag3():
        x0 = avgs_full[ic_l]
        x1 = avgs_full[ic_l + 1]
        x2 = avgs_full[ic_l + 2]
        y11 = _lagrange3(xnu, x0, x1, x2, atc[0, 0], atc[1, 0], atc[2, 0])
        y13 = _lagrange3(xnu, x0, x1, x2, atc[0, 1], atc[1, 1], atc[2, 1])
        y33 = _lagrange3(xnu, x0, x1, x2, atc[0, 2], atc[1, 2], atc[2, 2])
        interp = jnp.asarray([y11, y13, y33])
        no_data_guard = (
            (atc[2, 0] <= db.x11_l)
            | ((atc[1, 0] <= db.x11_l) & (xnu <= avgs_full[ic_l + 1]))
            | ((atc[0, 0] <= db.x11_l) & (xnu <= avgs_full[ic_l]))
        )
        return jnp.where(no_data_guard, no_data, interp)

    def but4():
        x0 = avgs_full[ic_l]
        x1 = avgs_full[ic_l + 1]
        x2 = avgs_full[ic_l + 2]
        x3 = avgs_full[ic_l + 3]
        y11 = _butland4(xnu, x0, x1, x2, x3, atc[0, 0], atc[1, 0], atc[2, 0], atc[3, 0], db.gmix_nu)
        y13 = _butland4(xnu, x0, x1, x2, x3, atc[0, 1], atc[1, 1], atc[2, 1], atc[3, 1], db.gmix_nu)
        y33 = _butland4(xnu, x0, x1, x2, x3, atc[0, 2], atc[1, 2], atc[2, 2], atc[3, 2], db.gmix_nu)
        interp = jnp.asarray([y11, y13, y33])
        return jnp.where(atc[2, 0] <= db.x11_l, no_data, interp)

    return jax.lax.cond(noic == 3, lag3, but4)


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
    lc_fit = db.lc_fit[ir]
    ag11_0_fit = db.ag11_0_fit[ir]
    ag11_sq_fit = db.ag11_sq_fit[ir]
    aefld_u_fit = db.aefld_u_fit[ir]
    aex_er_fit = db.aex_er_fit[ir]

    def zero_field():
        return _eval_zero_field_node(ir, xnu, db, xs0, g110, g130, g330, icu0, lc_fit, ag11_0_fit)

    def positive_field():
        return jax.lax.cond(
            xnu < avgs_full[0],
            lambda: _eval_low_branch(ir, xnu, xer, efield, db, avgs_full, aref_full, ag11_full, ag13_full, ag33_full, lc_fit, ag11_0_fit, ag11_sq_fit, aefld_u_fit, aex_er_fit),
            lambda: jax.lax.cond(
                xnu > avgs_full[icue - 1],
                lambda: _eval_high_branch(ir, xnu, efield, db),
                lambda: _eval_middle_branch(ir, xnu, xer, db, avgs_full, aref_full, ag11_full, ag13_full, ag33_full, icue),
            ),
        )

    return jax.lax.cond(efield <= db.xref_l, zero_field, positive_field)


@jax.jit
def get_Dij_ntss_preprocessed(grid_x, grid_nu, grid_Er, db):
    arr = db.r_grid
    nr = arr.shape[0]
    xri = jax.lax.cond(nr == 1, lambda: arr[0], lambda: jnp.maximum(1.0e-2 * arr[0], grid_x))
    xnu = jnp.log10(jnp.maximum(1.0e-12, jnp.abs(grid_nu)))
    efield = jnp.where(xri <= 1.0e-30, 0.0, jnp.abs(grid_Er / xri))
    xer = jnp.log10(jnp.maximum(db.xref_l, efield))

    exact_mask = jnp.abs(xri - arr) <= db.del_r
    exact_idx = jnp.argmax(exact_mask.astype(jnp.int32))
    is_exact = jnp.any(exact_mask)

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
