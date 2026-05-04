from __future__ import annotations

from typing import Any, Callable

from ._interpolators import get_Dij, get_Dij_3d, get_Dij_loger_no_r
from ._interpolators_ntss_preprocessed import get_Dij_ntss_preprocessed
from ._interpolators_preprocessed import (
    get_Dij_preprocessed_3d,
    get_Dij_preprocessed_3d_ntss_radius,
    get_Dij_preprocessed_3d_ntss_radius_ntss1d,
    get_Dij_preprocessed_3d_ntss_radius_ntss1d_fixednu,
)
from ._monoenergetic import (
    MONOENERGETIC_KIND_GENERIC,
    MONOENERGETIC_KIND_GENERIC_3D,
    MONOENERGETIC_KIND_GENERIC_LOGER_NO_R,
    MONOENERGETIC_KIND_PREPROCESSED_3D,
    MONOENERGETIC_KIND_PREPROCESSED_3D_NTSS1D_FIXED,
    MONOENERGETIC_KIND_PREPROCESSED_3D_RADIAL,
    MONOENERGETIC_KIND_PREPROCESSED_3D_RADIAL_NTSS1D,
    MONOENERGETIC_KIND_PREPROCESSED_NTSS,
    monoenergetic_database_kind,
)


MONOENERGETIC_INTERPOLATION_KERNELS: dict[str, Callable[..., Any]] = {
    MONOENERGETIC_KIND_PREPROCESSED_3D_NTSS1D_FIXED: get_Dij_preprocessed_3d_ntss_radius_ntss1d_fixednu,
    MONOENERGETIC_KIND_PREPROCESSED_3D_RADIAL_NTSS1D: get_Dij_preprocessed_3d_ntss_radius_ntss1d,
    MONOENERGETIC_KIND_PREPROCESSED_NTSS: get_Dij_ntss_preprocessed,
    MONOENERGETIC_KIND_PREPROCESSED_3D_RADIAL: get_Dij_preprocessed_3d_ntss_radius,
    MONOENERGETIC_KIND_PREPROCESSED_3D: get_Dij_preprocessed_3d,
    MONOENERGETIC_KIND_GENERIC: get_Dij,
    MONOENERGETIC_KIND_GENERIC_3D: get_Dij_3d,
    MONOENERGETIC_KIND_GENERIC_LOGER_NO_R: get_Dij_loger_no_r,
}


def monoenergetic_interpolation_kernel(database: Any) -> Callable[..., Any]:
    kind = monoenergetic_database_kind(database)
    return MONOENERGETIC_INTERPOLATION_KERNELS[kind]
