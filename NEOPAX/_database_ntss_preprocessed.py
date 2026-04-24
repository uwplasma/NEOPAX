import dataclasses
import numpy as np
import h5py as h5
import jax
import jax.numpy as jnp
from jax import config
from jaxtyping import Array, Float

# to use higher precision
config.update("jax_enable_x64", True)


DXNU = 0.05
XREF_L = 1.0e-8
DREF = 0.7
GMIX_ER = 0.1
GMIX_NU = 0.2
X11_L = -12.0
X13_L = 0.0
DEL_R = 1.0e-3


def _prepare_ntss_arrays(a_b, rho, nu_v, Er, drds, D11, D13, D33):
    rho = np.asarray(rho, dtype=float)
    nu_v = np.asarray(nu_v, dtype=float)
    Er = np.asarray(Er, dtype=float)
    drds = np.asarray(drds, dtype=float)
    D11 = np.asarray(D11, dtype=float).copy()
    D13 = np.asarray(D13, dtype=float).copy()
    D33 = np.asarray(D33, dtype=float).copy()

    n_r = rho.shape[0]
    n_nu = nu_v.shape[0]
    n_er = Er.shape[-1]
    n_total = n_nu * n_er
    max_groups = n_total

    r_grid = a_b * rho

    axnu0 = np.full((n_r, n_nu), 0.0, dtype=float)
    ag110 = np.full((n_r, n_nu), X11_L, dtype=float)
    ag130 = np.full((n_r, n_nu), X13_L, dtype=float)
    ag330 = np.full((n_r, n_nu), 0.0, dtype=float)
    nval0 = np.zeros(n_r, dtype=np.int32)

    axnu = np.full((n_r, n_total), 0.0, dtype=float)
    aref = np.full((n_r, n_total), np.log10(XREF_L), dtype=float)
    ag11 = np.full((n_r, n_total), X11_L, dtype=float)
    ag13 = np.full((n_r, n_total), X13_L, dtype=float)
    ag33 = np.full((n_r, n_total), 0.0, dtype=float)
    nvale = np.zeros(n_r, dtype=np.int32)

    axnuar = np.full((n_r, max_groups), 0.0, dtype=float)
    inulr = np.zeros((n_r, max_groups), dtype=np.int32)
    inugr = np.zeros((n_r, max_groups), dtype=np.int32)
    icnur = np.zeros(n_r, dtype=np.int32)

    for ir in range(n_r):
        D11[ir, :, :] *= drds[ir] ** 2
        D13[ir, :, :] *= drds[ir]
        D33[ir, :, :] *= nu_v[:, None]

        finite_rows = []
        zero_rows = []
        for inu, nu in enumerate(nu_v):
            xnu = np.log10(max(1.0e-12, nu))
            for ier in range(n_er):
                xref_raw = abs(Er[0, ier]) / max(a_b * rho[ir], 1.0e-30)
                xer = np.log10(max(XREF_L, xref_raw))
                row = (
                    xnu,
                    xer,
                    np.log10(max(1.0e-20, abs(D11[ir, inu, ier]))),
                    D13[ir, inu, ier],
                    abs(D33[ir, inu, ier]),
                )
                if xref_raw < XREF_L:
                    zero_rows.append((xnu, row[2], row[3], row[4]))
                finite_rows.append(row)

        zero_rows.sort(key=lambda row: row[0])
        nv0 = min(len(zero_rows), n_nu)
        nval0[ir] = nv0
        for i, (xnu, g11, g13, g33) in enumerate(zero_rows[:nv0]):
            axnu0[ir, i] = xnu
            ag110[ir, i] = g11
            ag130[ir, i] = g13
            ag330[ir, i] = g33

        finite_rows.sort(key=lambda row: row[0])
        nve = len(finite_rows)
        nvale[ir] = nve
        for i, (xnu, xer, g11, g13, g33) in enumerate(finite_rows):
            axnu[ir, i] = xnu
            aref[ir, i] = xer
            ag11[ir, i] = g11
            ag13[ir, i] = g13
            ag33[ir, i] = g33

        groups = []
        ml = 0
        while ml < max(0, nve - 1):
            mg = ml
            xnuav = axnu[ir, mg]
            while mg + 1 < nve and abs(axnu[ir, mg + 1] - axnu[ir, ml]) < DXNU:
                mg += 1
                xnuav += axnu[ir, mg]
            noe = mg - ml + 1
            next_ml = mg + 1
            if noe > 2:
                idx = np.arange(ml, mg + 1)
                order = np.argsort(aref[ir, idx], kind="mergesort")
                idx_sorted = idx[order]
                aref_block = aref[ir, idx_sorted].copy()
                ag11_block = ag11[ir, idx_sorted].copy()
                ag13_block = ag13[ir, idx_sorted].copy()
                ag33_block = ag33[ir, idx_sorted].copy()
                if aref_block.shape[0] >= 2:
                    aref_block[0] = max(aref_block[0], aref_block[1] - DREF)
                groups.append(
                    (
                        float(xnuav / noe),
                        aref_block,
                        ag11_block,
                        ag13_block,
                        ag33_block,
                    )
                )
            ml = next_ml

        icnu = len(groups)
        icnur[ir] = icnu
        pos = 0
        for ic, (xnu_avg, aref_block, ag11_block, ag13_block, ag33_block) in enumerate(groups):
            start = pos
            end = pos + aref_block.shape[0]
            axnuar[ir, ic] = xnu_avg
            inulr[ir, ic] = start
            inugr[ir, ic] = end - 1
            aref[ir, start:end] = aref_block
            ag11[ir, start:end] = ag11_block
            ag13[ir, start:end] = ag13_block
            ag33[ir, start:end] = ag33_block
            pos = end
        nvale[ir] = pos

    return {
        "a_b": a_b,
        "rho": jnp.asarray(rho),
        "r_grid": jnp.asarray(r_grid),
        "axnu0": jnp.asarray(axnu0),
        "ag110": jnp.asarray(ag110),
        "ag130": jnp.asarray(ag130),
        "ag330": jnp.asarray(ag330),
        "nval0": jnp.asarray(nval0),
        "axnu": jnp.asarray(axnu),
        "aref": jnp.asarray(aref),
        "ag11": jnp.asarray(ag11),
        "ag13": jnp.asarray(ag13),
        "ag33": jnp.asarray(ag33),
        "nvale": jnp.asarray(nvale),
        "axnuar": jnp.asarray(axnuar),
        "inulr": jnp.asarray(inulr),
        "inugr": jnp.asarray(inugr),
        "icnur": jnp.asarray(icnur),
        "x11_l": jnp.asarray(X11_L),
        "x13_l": jnp.asarray(X13_L),
        "xref_l": jnp.asarray(XREF_L),
        "gmix_er": jnp.asarray(GMIX_ER),
        "gmix_nu": jnp.asarray(GMIX_NU),
        "del_r": jnp.asarray(DEL_R),
    }


@jax.tree_util.register_dataclass
@dataclasses.dataclass(eq=False)
class NTSSPreprocessedMonoenergetic:
    a_b: float
    rho: Float[Array, "..."]
    r_grid: Float[Array, "..."]
    axnu0: Float[Array, "..."]
    ag110: Float[Array, "..."]
    ag130: Float[Array, "..."]
    ag330: Float[Array, "..."]
    nval0: Float[Array, "..."]
    axnu: Float[Array, "..."]
    aref: Float[Array, "..."]
    ag11: Float[Array, "..."]
    ag13: Float[Array, "..."]
    ag33: Float[Array, "..."]
    nvale: Float[Array, "..."]
    axnuar: Float[Array, "..."]
    inulr: Float[Array, "..."]
    inugr: Float[Array, "..."]
    icnur: Float[Array, "..."]
    x11_l: float
    x13_l: float
    xref_l: float
    gmix_er: float
    gmix_nu: float
    del_r: float

    @classmethod
    def read_monkes(cls, a_b, monkes_file):
        file = h5.File(monkes_file, "r")
        data = _prepare_ntss_arrays(
            a_b=a_b,
            rho=file["rho"][()],
            nu_v=file["nu_v"][()],
            Er=file["Er"][()],
            drds=file["drds"][()],
            D11=file["D11"][()],
            D13=file["D13"][()],
            D33=file["D33"][()],
        )
        file.close()
        return cls(**data)
