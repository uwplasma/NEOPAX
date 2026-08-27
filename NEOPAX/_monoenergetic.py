from __future__ import annotations

from collections.abc import Callable
import dataclasses
from typing import Any

import jax.numpy as jnp

from ._database import Monoenergetic, Monoenergetic3D, MonoenergeticLogErNoR
from ._database_ntss_preprocessed import NTSSPreprocessedMonoenergetic
from ._database_preprocessed import (
    PreprocessedMonoenergetic3D,
    PreprocessedMonoenergetic3DNTSSRadius,
    PreprocessedMonoenergetic3DNTSSRadiusNTSS1D,
    PreprocessedMonoenergetic3DNTSSRadiusNTSS1DFixedNU,
)


MONOENERGETIC_KIND_GENERIC = "generic"
MONOENERGETIC_KIND_GENERIC_3D = "generic_3d"
MONOENERGETIC_KIND_GENERIC_LOGER_NO_R = "generic_loger_no_r"
MONOENERGETIC_KIND_PREPROCESSED_3D = "preprocessed_3d"
MONOENERGETIC_KIND_PREPROCESSED_3D_RADIAL = "preprocessed_3d_radial"
MONOENERGETIC_KIND_PREPROCESSED_3D_RADIAL_NTSS1D = "preprocessed_3d_radial_ntss1d"
MONOENERGETIC_KIND_PREPROCESSED_3D_NTSS1D_FIXED = "preprocessed_3d_ntss1d_fixed"
MONOENERGETIC_KIND_PREPROCESSED_NTSS = "preprocessed_ntss"


def normalize_interpolation_mode(mode: str | None) -> str:
    return str(mode or MONOENERGETIC_KIND_GENERIC).strip().lower()


def _load_generic(geometry: Any, ntx_file: str):
    return Monoenergetic.read_ntx(geometry.a_b, ntx_file)


def _load_generic_3d(geometry: Any, ntx_file: str):
    return Monoenergetic3D.read_ntx(geometry.a_b, ntx_file)


def _load_generic_loger_no_r(geometry: Any, ntx_file: str):
    return MonoenergeticLogErNoR.read_ntx(geometry.a_b, ntx_file)


def _load_preprocessed_ntss(geometry: Any, ntx_file: str):
    return NTSSPreprocessedMonoenergetic.read_ntx(geometry, ntx_file)


def _load_preprocessed_3d(geometry: Any, ntx_file: str):
    return PreprocessedMonoenergetic3D.read_ntx(geometry.a_b, ntx_file)


def _load_preprocessed_3d_radial(geometry: Any, ntx_file: str):
    return PreprocessedMonoenergetic3DNTSSRadius.read_ntx(geometry.a_b, ntx_file)


def _load_preprocessed_3d_radial_ntss1d(geometry: Any, ntx_file: str):
    return PreprocessedMonoenergetic3DNTSSRadiusNTSS1D.read_ntx(geometry.a_b, ntx_file)


def _load_preprocessed_3d_ntss1d_fixed(geometry: Any, ntx_file: str):
    return PreprocessedMonoenergetic3DNTSSRadiusNTSS1DFixedNU.read_ntx(geometry.a_b, ntx_file)


MONOENERGETIC_LOADERS: dict[str, Callable[[Any, str], Any]] = {
    MONOENERGETIC_KIND_PREPROCESSED_NTSS: _load_preprocessed_ntss,
    MONOENERGETIC_KIND_PREPROCESSED_3D: _load_preprocessed_3d,
    MONOENERGETIC_KIND_PREPROCESSED_3D_RADIAL: _load_preprocessed_3d_radial,
    MONOENERGETIC_KIND_PREPROCESSED_3D_RADIAL_NTSS1D: _load_preprocessed_3d_radial_ntss1d,
    MONOENERGETIC_KIND_PREPROCESSED_3D_NTSS1D_FIXED: _load_preprocessed_3d_ntss1d_fixed,
    MONOENERGETIC_KIND_GENERIC: _load_generic,
    MONOENERGETIC_KIND_GENERIC_3D: _load_generic_3d,
    MONOENERGETIC_KIND_GENERIC_LOGER_NO_R: _load_generic_loger_no_r,
}


MONOENERGETIC_KIND_BY_CLASS: dict[type[Any], str] = {
    PreprocessedMonoenergetic3DNTSSRadiusNTSS1DFixedNU: MONOENERGETIC_KIND_PREPROCESSED_3D_NTSS1D_FIXED,
    PreprocessedMonoenergetic3DNTSSRadiusNTSS1D: MONOENERGETIC_KIND_PREPROCESSED_3D_RADIAL_NTSS1D,
    PreprocessedMonoenergetic3DNTSSRadius: MONOENERGETIC_KIND_PREPROCESSED_3D_RADIAL,
    NTSSPreprocessedMonoenergetic: MONOENERGETIC_KIND_PREPROCESSED_NTSS,
    PreprocessedMonoenergetic3D: MONOENERGETIC_KIND_PREPROCESSED_3D,
    Monoenergetic3D: MONOENERGETIC_KIND_GENERIC_3D,
    MonoenergeticLogErNoR: MONOENERGETIC_KIND_GENERIC_LOGER_NO_R,
    Monoenergetic: MONOENERGETIC_KIND_GENERIC,
}


def load_monoenergetic_database(geometry: Any, ntx_file: str, interpolation_mode: str | None = None):
    mode = normalize_interpolation_mode(interpolation_mode)
    loader = MONOENERGETIC_LOADERS.get(mode)
    if loader is None:
        raise ValueError(
            f"Unknown monoenergetic interpolation_mode '{interpolation_mode}'. "
            f"Expected one of: {', '.join(sorted(MONOENERGETIC_LOADERS))}."
        )
    return loader(geometry, ntx_file)


def monoenergetic_database_kind(database: Any) -> str:
    for cls, kind in MONOENERGETIC_KIND_BY_CLASS.items():
        if isinstance(database, cls):
            return kind
    return MONOENERGETIC_KIND_GENERIC


def database_with_geometry_scale(database: Any, a_b: Any) -> Any:
    """Return a pure database view rebuilt for a new geometry scale.

    The database file contains fixed coefficient values.  Its interpolation
    coordinates and radius limits, however, contain ``a_b``.  A realtime
    geometry reverse must therefore not replace only the model geometry while
    retaining the old database object.

    This deliberately supports only the formats whose complete scale
    dependence is represented by their existing pytree leaves.  The NTSS
    preprocessed format additionally receives geometry-derived fit channels
    at load time and needs its own full reconstruction contract.
    """

    if isinstance(database, NTSSPreprocessedMonoenergetic):
        raise ValueError(
            "Realtime geometry replacement for preprocessed_ntss databases "
            "is not implemented: its fit channels depend on more than a_b."
        )

    old_a_b = jnp.asarray(database.a_b)
    new_a_b = jnp.asarray(a_b, dtype=old_a_b.dtype)
    log_scale_shift = jnp.log10(
        jnp.maximum(old_a_b, 1.0e-30) / jnp.maximum(new_a_b, 1.0e-30)
    )

    if isinstance(database, Monoenergetic):
        return dataclasses.replace(
            database,
            a_b=new_a_b,
            Er_list=database.Er_list + log_scale_shift,
            low_limit_r=jnp.asarray(1.0e-3, dtype=new_a_b.dtype) * new_a_b,
            r1_lim=new_a_b * database.rho[1],
            rmn2_lim=new_a_b * database.rho[-2],
            r1=database.rho[0] * new_a_b,
            r2=database.rho[1] * new_a_b,
            r3=database.rho[2] * new_a_b,
            rnm3=database.rho[-3] * new_a_b,
            rnm2=database.rho[-2] * new_a_b,
            rnm1=database.rho[-1] * new_a_b,
        )

    if isinstance(
        database,
        (
            PreprocessedMonoenergetic3D,
            PreprocessedMonoenergetic3DNTSSRadius,
        ),
    ):
        updates = {
            "a_b": new_a_b,
            "r_grid": new_a_b * database.rho,
            "Er_grid": database.Er_grid + log_scale_shift,
            "low_limit_r": jnp.asarray(1.0e-3, dtype=new_a_b.dtype) * new_a_b,
        }
        if isinstance(database, PreprocessedMonoenergetic3DNTSSRadiusNTSS1D):
            updates["Er_grid_ntss1d"] = jnp.where(
                jnp.isfinite(database.Er_grid_ntss1d),
                database.Er_grid_ntss1d + log_scale_shift,
                database.Er_grid_ntss1d,
            )
        return dataclasses.replace(database, **updates)

    raise TypeError(
        "Unsupported monoenergetic database type for realtime geometry "
        f"replacement: {type(database).__name__}."
    )
