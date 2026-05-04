from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ._database import Monoenergetic, MonoenergeticLogErNoR
from ._database_ntss_preprocessed import NTSSPreprocessedMonoenergetic
from ._database_preprocessed import (
    PreprocessedMonoenergetic3D,
    PreprocessedMonoenergetic3DNTSSRadius,
    PreprocessedMonoenergetic3DNTSSRadiusNTSS1D,
    PreprocessedMonoenergetic3DNTSSRadiusNTSS1DFixedNU,
)


MONOENERGETIC_KIND_GENERIC = "generic"
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
    MONOENERGETIC_KIND_GENERIC_LOGER_NO_R: _load_generic_loger_no_r,
}


MONOENERGETIC_KIND_BY_CLASS: dict[type[Any], str] = {
    PreprocessedMonoenergetic3DNTSSRadiusNTSS1DFixedNU: MONOENERGETIC_KIND_PREPROCESSED_3D_NTSS1D_FIXED,
    PreprocessedMonoenergetic3DNTSSRadiusNTSS1D: MONOENERGETIC_KIND_PREPROCESSED_3D_RADIAL_NTSS1D,
    PreprocessedMonoenergetic3DNTSSRadius: MONOENERGETIC_KIND_PREPROCESSED_3D_RADIAL,
    NTSSPreprocessedMonoenergetic: MONOENERGETIC_KIND_PREPROCESSED_NTSS,
    PreprocessedMonoenergetic3D: MONOENERGETIC_KIND_PREPROCESSED_3D,
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
