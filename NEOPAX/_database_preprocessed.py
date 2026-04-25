import dataclasses

import h5py as h5
import jax
import jax.numpy as jnp
from jax import config
from jaxtyping import Array, Float

# to use higher precision
config.update("jax_enable_x64", True)

DEL_R = 1.0e-3


def _prepare_monkes_arrays(a_b, rho, nu_v, Er, drds, D11, D13, D33, *, divide_by_radius: bool = True):
    rho = jnp.asarray(rho)
    nu_v = jnp.asarray(nu_v)
    Er = jnp.asarray(Er)
    drds = jnp.asarray(drds)
    D11 = jnp.asarray(D11)
    D13 = jnp.asarray(D13)
    D33 = jnp.asarray(D33)
    n_r = rho.shape[0]
    n_nu = nu_v.shape[0]
    n_er = Er.shape[-1]

    D11_scaled = jnp.array(D11)
    D13_scaled = jnp.array(D13)
    D33_scaled = jnp.array(D33)
    Er_grid = jnp.zeros((n_r, n_er))

    for j in range(n_r):
        D11_scaled = D11_scaled.at[j, :, :].set(D11_scaled[j, :, :] * jnp.square(drds[j]))
        D13_scaled = D13_scaled.at[j, :, :].set(D13_scaled[j, :, :] * drds[j])
        D33_scaled = D33_scaled.at[j, :, :].set(D33_scaled[j, :, :] * nu_v[:, None])
        er_base = jnp.abs(Er[0, :])
        er_row = jax.lax.cond(
            divide_by_radius,
            lambda: jnp.log10(jnp.maximum(1.0e-8, er_base / jnp.maximum(a_b * rho[j], 1.0e-30))),
            lambda: jnp.log10(jnp.maximum(1.0e-8, er_base)),
        )
        Er_grid = Er_grid.at[j, :].set(er_row)

    return {
        "a_b": a_b,
        "rho": rho,
        "r_grid": a_b * rho,
        "nu_log": jnp.log10(nu_v),
        "Er_grid": Er_grid,
        "D11_log": jnp.log10(D11_scaled),
        "D13": D13_scaled,
        "D33": D33_scaled,
        "Er_lower_limit": jnp.array(1.0e-8),
        "low_limit_r": jnp.array(1.0e-3 * a_b),
        "del_r": jnp.array(DEL_R),
    }


def _prepare_monkes_arrays_ertilde(a_b, rho, nu_v, Er_tilde, Er_to_Ertilde, drds, D11, D13, D33):
    rho = jnp.asarray(rho)
    nu_v = jnp.asarray(nu_v)
    Er_tilde = jnp.asarray(Er_tilde)
    Er_to_Ertilde = jnp.asarray(Er_to_Ertilde)
    drds = jnp.asarray(drds)
    D11 = jnp.asarray(D11)
    D13 = jnp.asarray(D13)
    D33 = jnp.asarray(D33)
    n_r = rho.shape[0]
    n_er = Er_tilde.shape[0]

    D11_scaled = jnp.array(D11)
    D13_scaled = jnp.array(D13)
    D33_scaled = jnp.array(D33)
    Er_grid = jnp.zeros((n_r, n_er))

    scale_grid = jax.lax.cond(
        Er_to_Ertilde.ndim == 2,
        lambda: jnp.asarray(Er_to_Ertilde[:, 0]),
        lambda: jnp.asarray(Er_to_Ertilde),
    )

    for j in range(n_r):
        D11_scaled = D11_scaled.at[j, :, :].set(D11_scaled[j, :, :] * jnp.square(drds[j]))
        D13_scaled = D13_scaled.at[j, :, :].set(D13_scaled[j, :, :] * drds[j])
        D33_scaled = D33_scaled.at[j, :, :].set(D33_scaled[j, :, :] * nu_v[:, None])
        er_row = jnp.log10(jnp.maximum(1.0e-8, jnp.abs(Er_tilde) / jnp.maximum(a_b * rho[j], 1.0e-30)))
        Er_grid = Er_grid.at[j, :].set(er_row)

    return {
        "a_b": a_b,
        "rho": rho,
        "r_grid": a_b * rho,
        "nu_log": jnp.log10(nu_v),
        "Er_grid": Er_grid,
        "D11_log": jnp.log10(D11_scaled),
        "D13": D13_scaled,
        "D33": D33_scaled,
        "Er_to_Ertilde_scale": scale_grid,
        "Er_lower_limit": jnp.array(1.0e-8),
        "low_limit_r": jnp.array(1.0e-3 * a_b),
        "del_r": jnp.array(DEL_R),
    }


@jax.tree_util.register_dataclass
@dataclasses.dataclass(eq=False)
class PreprocessedMonoenergetic3D:
    a_b: float
    rho: Float[Array, "..."]
    r_grid: Float[Array, "..."]
    nu_log: Float[Array, "..."]
    Er_grid: Float[Array, "..."]
    D11_log: Float[Array, "..."]
    D13: Float[Array, "..."]
    D33: Float[Array, "..."]
    Er_lower_limit: float
    low_limit_r: float
    del_r: float

    @classmethod
    def read_monkes(cls, a_b, monkes_file):
        file = h5.File(monkes_file, "r")
        data = _prepare_monkes_arrays(
            a_b=a_b,
            rho=file["rho"][()],
            nu_v=file["nu_v"][()],
            Er=file["Er"][()],
            drds=file["drds"][()],
            D11=file["D11"][()],
            D13=file["D13"][()],
            D33=file["D33"][()],
            divide_by_radius=True,
        )
        file.close()
        return cls(**data)

    @classmethod
    def read_data(cls, a_b, rho, nu_v, Er, drds, D11, D13, D33):
        data = _prepare_monkes_arrays(
            a_b=a_b,
            rho=rho,
            nu_v=nu_v,
            Er=Er,
            drds=drds,
            D11=D11,
            D13=D13,
            D33=D33,
            divide_by_radius=True,
        )
        return cls(**data)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(eq=False)
class PreprocessedMonoenergetic3DErRaw:
    a_b: float
    rho: Float[Array, "..."]
    r_grid: Float[Array, "..."]
    nu_log: Float[Array, "..."]
    Er_grid: Float[Array, "..."]
    D11_log: Float[Array, "..."]
    D13: Float[Array, "..."]
    D33: Float[Array, "..."]
    Er_lower_limit: float
    low_limit_r: float
    del_r: float

    @classmethod
    def read_monkes(cls, a_b, monkes_file):
        file = h5.File(monkes_file, "r")
        data = _prepare_monkes_arrays(
            a_b=a_b,
            rho=file["rho"][()],
            nu_v=file["nu_v"][()],
            Er=file["Er"][()],
            drds=file["drds"][()],
            D11=file["D11"][()],
            D13=file["D13"][()],
            D33=file["D33"][()],
            divide_by_radius=False,
        )
        file.close()
        return cls(**data)

    @classmethod
    def read_data(cls, a_b, rho, nu_v, Er, drds, D11, D13, D33):
        data = _prepare_monkes_arrays(
            a_b=a_b,
            rho=rho,
            nu_v=nu_v,
            Er=Er,
            drds=drds,
            D11=D11,
            D13=D13,
            D33=D33,
            divide_by_radius=False,
        )
        return cls(**data)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(eq=False)
class PreprocessedMonoenergetic3DErTilde:
    a_b: float
    rho: Float[Array, "..."]
    r_grid: Float[Array, "..."]
    nu_log: Float[Array, "..."]
    Er_grid: Float[Array, "..."]
    D11_log: Float[Array, "..."]
    D13: Float[Array, "..."]
    D33: Float[Array, "..."]
    Er_to_Ertilde_scale: Float[Array, "..."]
    Er_lower_limit: float
    low_limit_r: float
    del_r: float

    @classmethod
    def read_monkes(cls, a_b, monkes_file):
        file = h5.File(monkes_file, "r")
        data = _prepare_monkes_arrays_ertilde(
            a_b=a_b,
            rho=file["rho"][()],
            nu_v=file["nu_v"][()],
            Er_tilde=file["Er_tilde"][()],
            Er_to_Ertilde=file["Er_to_Ertilde"][()],
            drds=file["drds"][()],
            D11=file["D11"][()],
            D13=file["D13"][()],
            D33=file["D33"][()],
        )
        file.close()
        return cls(**data)

    @classmethod
    def read_data(cls, a_b, rho, nu_v, Er_tilde, Er_to_Ertilde, drds, D11, D13, D33):
        data = _prepare_monkes_arrays_ertilde(
            a_b=a_b,
            rho=rho,
            nu_v=nu_v,
            Er_tilde=Er_tilde,
            Er_to_Ertilde=Er_to_Ertilde,
            drds=drds,
            D11=D11,
            D13=D13,
            D33=D33,
        )
        return cls(**data)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(eq=False)
class PreprocessedMonoenergetic3DMonotonic:
    a_b: float
    rho: Float[Array, "..."]
    r_grid: Float[Array, "..."]
    nu_log: Float[Array, "..."]
    Er_grid: Float[Array, "..."]
    D11_log: Float[Array, "..."]
    D13: Float[Array, "..."]
    D33: Float[Array, "..."]
    Er_lower_limit: float
    low_limit_r: float
    del_r: float

    @classmethod
    def read_monkes(cls, a_b, monkes_file):
        file = h5.File(monkes_file, "r")
        data = _prepare_monkes_arrays(
            a_b=a_b,
            rho=file["rho"][()],
            nu_v=file["nu_v"][()],
            Er=file["Er"][()],
            drds=file["drds"][()],
            D11=file["D11"][()],
            D13=file["D13"][()],
            D33=file["D33"][()],
            divide_by_radius=True,
        )
        file.close()
        return cls(**data)

    @classmethod
    def read_data(cls, a_b, rho, nu_v, Er, drds, D11, D13, D33):
        data = _prepare_monkes_arrays(
            a_b=a_b,
            rho=rho,
            nu_v=nu_v,
            Er=Er,
            drds=drds,
            D11=D11,
            D13=D13,
            D33=D33,
            divide_by_radius=True,
        )
        return cls(**data)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(eq=False)
class PreprocessedMonoenergetic3DNTSSRadius:
    a_b: float
    rho: Float[Array, "..."]
    r_grid: Float[Array, "..."]
    nu_log: Float[Array, "..."]
    Er_grid: Float[Array, "..."]
    D11_log: Float[Array, "..."]
    D13: Float[Array, "..."]
    D33: Float[Array, "..."]
    Er_lower_limit: float
    low_limit_r: float
    del_r: float

    @classmethod
    def read_monkes(cls, a_b, monkes_file):
        file = h5.File(monkes_file, "r")
        data = _prepare_monkes_arrays(
            a_b=a_b,
            rho=file["rho"][()],
            nu_v=file["nu_v"][()],
            Er=file["Er"][()],
            drds=file["drds"][()],
            D11=file["D11"][()],
            D13=file["D13"][()],
            D33=file["D33"][()],
            divide_by_radius=True,
        )
        file.close()
        return cls(**data)

    @classmethod
    def read_data(cls, a_b, rho, nu_v, Er, drds, D11, D13, D33):
        data = _prepare_monkes_arrays(
            a_b=a_b,
            rho=rho,
            nu_v=nu_v,
            Er=Er,
            drds=drds,
            D11=D11,
            D13=D13,
            D33=D33,
            divide_by_radius=True,
        )
        return cls(**data)
