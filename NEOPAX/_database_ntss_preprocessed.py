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


def _prepare_ntss_arrays(
    a_b,
    rho,
    nu_v,
    Er,
    drds,
    D11,
    D13,
    D33,
    *,
    lc_fit_in=None,
    ag11_0_in=None,
    ag11_sq_in=None,
    aefld_u_in=None,
    aex_er_in=None,
    akn_in=None,
    air_in=None,
    xrm_in=None,
):
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
    lc_fit = np.asarray(
        np.zeros(n_r, dtype=bool) if lc_fit_in is None else lc_fit_in,
        dtype=bool,
    )
    ag11_0 = np.asarray(
        np.zeros(n_r, dtype=float) if ag11_0_in is None else ag11_0_in,
        dtype=float,
    )
    ag11_sq = np.asarray(
        np.ones(n_r, dtype=float) if ag11_sq_in is None else ag11_sq_in,
        dtype=float,
    )
    aefld_u = np.asarray(
        np.zeros(n_r, dtype=float) if aefld_u_in is None else aefld_u_in,
        dtype=float,
    )
    aex_er = np.asarray(
        np.ones(n_r, dtype=float) if aex_er_in is None else aex_er_in,
        dtype=float,
    )
    akn = np.asarray(
        np.ones(n_r, dtype=float) if akn_in is None else akn_in,
        dtype=float,
    )
    air = np.asarray(
        np.ones(n_r, dtype=float) if air_in is None else air_in,
        dtype=float,
    )
    xrm = np.asarray(
        np.full(n_r, float(np.max(r_grid) * 2.0) if n_r > 0 else 1.0, dtype=float)
        if xrm_in is None
        else xrm_in,
        dtype=float,
    )

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
                if abs(Er[0, ier]) < XREF_L:
                    zero_rows.append((xnu, row[2], row[3], row[4]))
                finite_rows.append(row)

        nv0 = min(len(zero_rows), n_nu)
        nval0[ir] = nv0
        for i, (xnu, g11, g13, g33) in enumerate(zero_rows[:nv0]):
            axnu0[ir, i] = xnu
            ag110[ir, i] = g11
            ag130[ir, i] = g13
            ag330[ir, i] = g33

        nve = len(finite_rows)

        icnu = 0
        pos = 0
        ml = 0
        while ml < nve:
            mg = ml
            xnuav = finite_rows[mg][0]
            while mg + 1 < nve and abs(finite_rows[mg + 1][0] - finite_rows[ml][0]) < DXNU:
                mg += 1
                xnuav += finite_rows[mg][0]
            noe = mg - ml + 1
            if noe > 2:
                start = pos
                end = pos + noe
                inulr[ir, icnu] = start
                inugr[ir, icnu] = end - 1
                axnuar[ir, icnu] = float(xnuav / noe)
                for local_idx, (xnu, xer, g11, g13, g33) in enumerate(finite_rows[ml : mg + 1]):
                    idx = start + local_idx
                    axnu[ir, idx] = xnu
                    aref[ir, idx] = xer
                    ag11[ir, idx] = g11
                    ag13[ir, idx] = g13
                    ag33[ir, idx] = g33
                if noe >= 2:
                    min_idx = start
                    max_idx = end - 1
                    min_ref = aref[ir, start]
                    max_ref = aref[ir, end - 1]
                    for idx in range(start, end):
                        ref = aref[ir, idx]
                        if ref < min_ref:
                            min_ref = ref
                            min_idx = idx
                        if ref > max_ref:
                            max_ref = ref
                            max_idx = idx

                    if min_idx > start:
                        axnu[ir, start], axnu[ir, min_idx] = axnu[ir, min_idx], axnu[ir, start]
                        aref[ir, start], aref[ir, min_idx] = aref[ir, min_idx], aref[ir, start]
                        ag11[ir, start], ag11[ir, min_idx] = ag11[ir, min_idx], ag11[ir, start]
                        ag13[ir, start], ag13[ir, min_idx] = ag13[ir, min_idx], ag13[ir, start]
                        ag33[ir, start], ag33[ir, min_idx] = ag33[ir, min_idx], ag33[ir, start]
                        if max_idx == start:
                            max_idx = min_idx

                    if max_idx < end - 1:
                        axnu[ir, end - 1], axnu[ir, max_idx] = axnu[ir, max_idx], axnu[ir, end - 1]
                        aref[ir, end - 1], aref[ir, max_idx] = aref[ir, max_idx], aref[ir, end - 1]
                        ag11[ir, end - 1], ag11[ir, max_idx] = ag11[ir, max_idx], ag11[ir, end - 1]
                        ag13[ir, end - 1], ag13[ir, max_idx] = ag13[ir, max_idx], ag13[ir, end - 1]
                        ag33[ir, end - 1], ag33[ir, max_idx] = ag33[ir, max_idx], ag33[ir, end - 1]

                    aref[ir, start] = max(aref[ir, start], aref[ir, start + 1] - DREF)
                pos = end
                icnu += 1
            ml = mg + 1

        icnur[ir] = icnu
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
        "lc_fit": jnp.asarray(lc_fit),
        "ag11_0_fit": jnp.asarray(ag11_0),
        "ag11_sq_fit": jnp.asarray(ag11_sq),
        "aefld_u_fit": jnp.asarray(aefld_u),
        "aex_er_fit": jnp.asarray(aex_er),
        "akn_fit": jnp.asarray(akn),
        "air_fit": jnp.asarray(air),
        "xrm_fit": jnp.asarray(xrm),
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
    lc_fit: Float[Array, "..."]
    ag11_0_fit: Float[Array, "..."]
    ag11_sq_fit: Float[Array, "..."]
    aefld_u_fit: Float[Array, "..."]
    aex_er_fit: Float[Array, "..."]
    akn_fit: Float[Array, "..."]
    air_fit: Float[Array, "..."]
    xrm_fit: Float[Array, "..."]

    @classmethod
    def read_monkes(cls, geometry, monkes_file):
        file = h5.File(monkes_file, "r")
        rho_db = np.asarray(file["rho"][()], dtype=float)
        n_r = rho_db.shape[0]
        geom_rho = np.asarray(geometry.rho_grid, dtype=float)

        def _interp_from_geometry(values):
            values = np.asarray(values, dtype=float)
            return np.interp(rho_db, geom_rho, values)

        optional = {}
        for h5_key, data_key, default in (
            ("lc_fit", "lc_fit_in", np.zeros(n_r, dtype=bool)),
            ("ag11_0", "ag11_0_in", np.zeros(n_r, dtype=float)),
            ("ag11_sq", "ag11_sq_in", np.ones(n_r, dtype=float)),
            ("aefld_u", "aefld_u_in", np.zeros(n_r, dtype=float)),
            ("aex_er", "aex_er_in", np.ones(n_r, dtype=float)),
            ("akn", "akn_in", None),
            ("air", "air_in", None),
        ):
            optional[data_key] = file[h5_key][()] if h5_key in file else default
        if optional["akn_in"] is None:
            optional["akn_in"] = _interp_from_geometry(geometry.curvature)
        if optional["air_in"] is None:
            optional["air_in"] = _interp_from_geometry(geometry.iota)
        optional["xrm_in"] = None
        for h5_key in ("xrm", "Rmajor", "major_radius"):
            if h5_key in file:
                optional["xrm_in"] = file[h5_key][()]
                break
        if optional["xrm_in"] is None:
            r = np.asarray(geometry.r_grid)
            eps = np.asarray(geometry.epsilon_t)
            R0 = float(geometry.R0)
            safe_eps = np.where(np.abs(eps) > 1.0e-12, eps, 1.0)
            local_R = r / safe_eps
            local_R = np.where(np.abs(eps) > 1.0e-12, local_R, R0)
            local_R[0] = R0
            optional["xrm_in"] = _interp_from_geometry(local_R)
        data = _prepare_ntss_arrays(
            a_b=geometry.a_b,
            rho=rho_db,
            nu_v=file["nu_v"][()],
            Er=file["Er"][()],
            drds=file["drds"][()],
            D11=file["D11"][()],
            D13=file["D13"][()],
            D33=file["D33"][()],
            **optional,
        )
        file.close()
        return cls(**data)
