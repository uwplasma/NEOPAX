"""Parameter specifications for the reverse-AD optimization lane.

This module is intentionally small and solver-free. It centralizes the
parameter naming conventions that are currently duplicated across benchmark
scripts, so production reverse-AD code and validation benchmarks can share one
stable dialect.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import jax
import jax.numpy as jnp


PROFILE_PARAMETER_ORDER: tuple[str, ...] = (
    "n0",
    "T0",
    "density_shape_power",
    "temperature_shape_power",
    "density_shape_alpha",
    "temperature_shape_alpha",
)

VMEC_BOUNDARY_FAMILIES: tuple[str, ...] = ("RBC", "ZBS")
VMEX_BOUNDARY_SCALE_MODES: tuple[str, ...] = ("ess", "none", "unit", "identity")


@dataclasses.dataclass(frozen=True, slots=True)
class ProfileParameterSpec:
    """A scalar transport-profile optimization parameter."""

    name: str

    def __post_init__(self) -> None:
        normalized = str(self.name).strip()
        if normalized not in PROFILE_PARAMETER_ORDER:
            allowed = ", ".join(PROFILE_PARAMETER_ORDER)
            raise ValueError(f"Unsupported profile parameter {self.name!r}; choices are: {allowed}.")
        object.__setattr__(self, "name", normalized)

    @property
    def kind(self) -> Literal["profile"]:
        return "profile"

    @property
    def label(self) -> str:
        return self.name


@dataclasses.dataclass(frozen=True, slots=True)
class VmecBoundaryParameterSpec:
    """A scalar VMEC fixed-boundary harmonic parameter."""

    family: str
    m: int
    n: int

    def __post_init__(self) -> None:
        family = str(self.family).strip().upper()
        if family not in VMEC_BOUNDARY_FAMILIES:
            allowed = ", ".join(VMEC_BOUNDARY_FAMILIES)
            raise ValueError(f"Unsupported VMEC boundary family {self.family!r}; choices are: {allowed}.")
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "m", int(self.m))
        object.__setattr__(self, "n", int(self.n))

    @property
    def kind(self) -> Literal["vmec_boundary"]:
        return "vmec_boundary"

    @property
    def label(self) -> str:
        return f"{self.family}:{self.m}:{self.n}"

    @property
    def vmec_label(self) -> str:
        return f"vmec:{self.label}"

    def as_tuple(self) -> tuple[str, int, int]:
        return (self.family, self.m, self.n)


ReverseADParameterSpec = ProfileParameterSpec | VmecBoundaryParameterSpec


@dataclasses.dataclass(frozen=True, slots=True)
class ReverseADParameterSet:
    """Stable mixed parameter vector layout for reverse-AD objectives."""

    profile_specs: tuple[ProfileParameterSpec, ...] = ()
    vmec_boundary_specs: tuple[VmecBoundaryParameterSpec, ...] = ()

    @property
    def specs(self) -> tuple[ReverseADParameterSpec, ...]:
        return (*self.profile_specs, *self.vmec_boundary_specs)

    @property
    def labels(self) -> tuple[str, ...]:
        return parameter_labels(self.specs)

    @property
    def vmec_prefixed_labels(self) -> tuple[str, ...]:
        return parameter_labels(self.specs, vmec_prefix=True)

    @property
    def profile_labels(self) -> tuple[str, ...]:
        return tuple(spec.label for spec in self.profile_specs)

    @property
    def vmec_labels(self) -> tuple[str, ...]:
        return tuple(spec.label for spec in self.vmec_boundary_specs)

    @property
    def vmec_tuples(self) -> tuple[tuple[str, int, int], ...]:
        return vmec_boundary_tuples(self.vmec_boundary_specs)


@dataclasses.dataclass(frozen=True, slots=True)
class VmexBoundaryParameterization:
    """VMEX-style packed VMEC boundary parameter layout and scaling.

    The explicit ``RBC:m:n`` parser remains intentionally permissive for
    diagnostics.  This object is the optimization-facing convention: it mirrors
    VMEX packed boundary DOFs and carries the scale vector used by optimizers.
    """

    specs: tuple[VmecBoundaryParameterSpec, ...]
    scales: tuple[float, ...]
    scale_mode: str = "ess"
    ess_alpha: float = 1.0

    def __post_init__(self) -> None:
        specs = tuple(self.specs)
        scales = tuple(float(value) for value in self.scales)
        if len(specs) != len(scales):
            raise ValueError(
                "VMEX boundary parameterization requires one scale per parameter: "
                f"parameter_count={len(specs)} scale_count={len(scales)}."
            )
        validate_vmex_boundary_parameter_specs(specs)
        scale_mode = str(self.scale_mode).strip().lower()
        if scale_mode not in VMEX_BOUNDARY_SCALE_MODES:
            allowed = ", ".join(VMEX_BOUNDARY_SCALE_MODES)
            raise ValueError(f"Unsupported VMEX boundary scale mode {self.scale_mode!r}; choices are: {allowed}.")
        object.__setattr__(self, "specs", specs)
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "scale_mode", scale_mode)
        object.__setattr__(self, "ess_alpha", float(self.ess_alpha))

    @property
    def parameter_set(self) -> ReverseADParameterSet:
        return ReverseADParameterSet(profile_specs=(), vmec_boundary_specs=self.specs)

    @property
    def labels(self) -> tuple[str, ...]:
        return parameter_labels(self.specs)

    @property
    def vmec_tuples(self) -> tuple[tuple[str, int, int], ...]:
        return vmec_boundary_tuples(self.specs)

    @property
    def x_scale(self):
        """Return the scale vector for optimizers that accept physical DOFs plus ``x_scale``."""

        return jnp.asarray(self.scales, dtype=jnp.float64)

    def scaled_to_physical_delta(self, scaled_values):
        """Map optimization-space deltas to physical VMEC boundary deltas."""

        return jnp.asarray(scaled_values, dtype=jnp.float64) * self.x_scale

    def physical_to_scaled_delta(self, physical_values):
        """Map physical VMEC boundary deltas to scaled optimization coordinates."""

        return jnp.asarray(physical_values, dtype=jnp.float64) / self.x_scale


def is_vmex_independent_boundary_mode(m: int, n: int) -> bool:
    """Return whether ``(m, n)`` is an independent VMEX packed-boundary DOF."""

    m_value = int(m)
    n_value = int(n)
    return m_value > 0 or (m_value == 0 and n_value > 0)


def validate_vmex_boundary_parameter_specs(
    specs: Sequence[VmecBoundaryParameterSpec],
    *,
    allow_fixed: bool = False,
) -> tuple[VmecBoundaryParameterSpec, ...]:
    """Validate specs against the VMEX packed-boundary independent-mode rule."""

    active_specs = tuple(specs)
    if allow_fixed:
        return active_specs
    fixed = tuple(spec.label for spec in active_specs if not is_vmex_independent_boundary_mode(spec.m, spec.n))
    if fixed:
        raise ValueError(
            "VMEX packed boundary optimization excludes fixed/non-independent modes: "
            f"{', '.join(fixed)}. Use explicit benchmark specs for diagnostics if needed."
        )
    return active_specs


def vmex_boundary_mode_level(spec: VmecBoundaryParameterSpec) -> int:
    """Return the VMEX mode level used by ESS-style boundary scaling."""

    return max(abs(int(spec.m)), abs(int(spec.n)))


def vmex_boundary_parameter_scales(
    specs: Sequence[VmecBoundaryParameterSpec],
    *,
    scale_mode: str = "ess",
    ess_alpha: float = 1.0,
) -> tuple[float, ...]:
    """Return VMEX-style optimizer scales for a packed boundary parameter list."""

    active_specs = validate_vmex_boundary_parameter_specs(tuple(specs))
    normalized_mode = str(scale_mode).strip().lower()
    if normalized_mode not in VMEX_BOUNDARY_SCALE_MODES:
        allowed = ", ".join(VMEX_BOUNDARY_SCALE_MODES)
        raise ValueError(f"Unsupported VMEX boundary scale mode {scale_mode!r}; choices are: {allowed}.")
    if normalized_mode in {"none", "unit", "identity"} or float(ess_alpha) <= 0.0:
        return tuple(1.0 for _ in active_specs)
    alpha = float(ess_alpha)
    return tuple(float(np.exp(-alpha * vmex_boundary_mode_level(spec)) / np.exp(-alpha)) for spec in active_specs)


def normalize_profile_parameter_name(name: str) -> str:
    """Return a validated canonical profile parameter name."""

    return ProfileParameterSpec(name).name


def parse_profile_parameter_spec(text: str) -> ProfileParameterSpec:
    """Parse a profile parameter label."""

    return ProfileParameterSpec(str(text).strip())


def parse_profile_parameter_specs(
    values: Sequence[str] | str | None,
    *,
    default_all: bool = True,
) -> tuple[ProfileParameterSpec, ...]:
    """Parse profile parameter labels into canonical specs."""

    if values is None:
        if not default_all:
            return ()
        return tuple(ProfileParameterSpec(name) for name in PROFILE_PARAMETER_ORDER)
    if isinstance(values, str):
        raw_values = tuple(item.strip() for item in values.split(",") if item.strip())
    else:
        raw_values = tuple(str(item).strip() for item in values if str(item).strip())
    return tuple(parse_profile_parameter_spec(item) for item in raw_values)


def parse_vmec_boundary_parameter_spec(text: str) -> VmecBoundaryParameterSpec:
    """Parse `FAMILY:m:n` or `vmec:FAMILY:m:n` into a VMEC harmonic spec."""

    raw = str(text).strip()
    if raw.lower().startswith("vmec:"):
        raw = raw.split(":", 1)[1].strip()
    parts = [part.strip() for part in raw.split(":")]
    if len(parts) != 3:
        raise ValueError(f"VMEC parameter spec {text!r} must be FAMILY:m:n, e.g. RBC:1:0.")
    family, m_text, n_text = parts
    return VmecBoundaryParameterSpec(family, int(m_text), int(n_text))


def parse_reverse_ad_parameter_spec(text: str) -> ReverseADParameterSpec:
    """Parse either a profile parameter or a VMEC boundary parameter."""

    raw = str(text).strip()
    if raw in PROFILE_PARAMETER_ORDER:
        return parse_profile_parameter_spec(raw)
    return parse_vmec_boundary_parameter_spec(raw)


def parse_reverse_ad_parameter_specs(values: Sequence[str] | str) -> tuple[ReverseADParameterSpec, ...]:
    """Parse mixed profile/VMEC parameter labels in vector order."""

    if isinstance(values, str):
        raw_values = tuple(item.strip() for item in values.split(",") if item.strip())
    else:
        raw_values = tuple(str(item).strip() for item in values if str(item).strip())
    if not raw_values:
        raise ValueError("At least one reverse-AD parameter spec is required.")
    return tuple(parse_reverse_ad_parameter_spec(item) for item in raw_values)


def reverse_ad_parameter_set(
    *,
    profiles: Sequence[str] | str | None = None,
    vmec_boundary: Sequence[str | VmecBoundaryParameterSpec] | str | None = None,
    default_profiles: bool = True,
) -> ReverseADParameterSet:
    """Build a stable profile-then-VMEC reverse-AD parameter layout."""

    profile_specs = parse_profile_parameter_specs(profiles, default_all=default_profiles)
    if vmec_boundary is None:
        vmec_specs: tuple[VmecBoundaryParameterSpec, ...] = ()
    elif isinstance(vmec_boundary, str):
        vmec_specs = parse_vmec_boundary_parameter_specs(vmec_boundary)
    else:
        parsed_vmec_specs: list[VmecBoundaryParameterSpec] = []
        for item in vmec_boundary:
            if isinstance(item, VmecBoundaryParameterSpec):
                parsed_vmec_specs.append(item)
            else:
                parsed_vmec_specs.append(parse_vmec_boundary_parameter_spec(str(item)))
        vmec_specs = tuple(parsed_vmec_specs)
    return ReverseADParameterSet(profile_specs=profile_specs, vmec_boundary_specs=vmec_specs)


def reverse_ad_optimization_parameter_set(
    *,
    include_profiles: bool = True,
    profiles: Sequence[str] | str | None = None,
    vmec_boundary: Sequence[str | VmecBoundaryParameterSpec] | str | None = None,
) -> ReverseADParameterSet:
    """Build an optimization parameter layout with optional profile DOFs.

    This is the intent-explicit optimization wrapper around
    ``reverse_ad_parameter_set``.  Set ``include_profiles=False`` for
    geometry-only optimization while still differentiating transport objectives
    with respect to the selected VMEC boundary DOFs.
    """

    if not include_profiles and profiles is not None:
        raise ValueError("profiles must be omitted when include_profiles=False.")
    return reverse_ad_parameter_set(
        profiles=profiles,
        vmec_boundary=vmec_boundary,
        default_profiles=bool(include_profiles),
    )


def parse_vmec_boundary_parameter_specs(
    text: str | None,
    *,
    default: VmecBoundaryParameterSpec | tuple[str, int, int] | None = None,
) -> tuple[VmecBoundaryParameterSpec, ...]:
    """Parse a comma-separated VMEC harmonic list.

    This mirrors the benchmark `--param-specs` syntax while returning typed
    specs for the optimization lane.
    """

    if text is None or not str(text).strip():
        if default is None:
            raise ValueError("At least one VMEC parameter spec is required.")
        if isinstance(default, VmecBoundaryParameterSpec):
            return (default,)
        family, m, n = default
        return (VmecBoundaryParameterSpec(family, m, n),)

    specs: list[VmecBoundaryParameterSpec] = []
    for raw_spec in str(text).split(","):
        spec = raw_spec.strip()
        if spec:
            specs.append(parse_vmec_boundary_parameter_spec(spec))
    if not specs:
        raise ValueError("VMEC parameter spec list did not contain any valid specs.")
    return tuple(specs)


def format_parameter_spec(spec: ReverseADParameterSpec, *, vmec_prefix: bool = False) -> str:
    """Return the canonical label for a reverse-AD parameter spec."""

    if isinstance(spec, ProfileParameterSpec):
        return spec.label
    return spec.vmec_label if vmec_prefix else spec.label


def parameter_labels(
    specs: Sequence[ReverseADParameterSpec],
    *,
    vmec_prefix: bool = False,
) -> tuple[str, ...]:
    """Return canonical labels in vector order."""

    return tuple(format_parameter_spec(spec, vmec_prefix=vmec_prefix) for spec in specs)


def split_parameter_specs(
    specs: Sequence[ReverseADParameterSpec],
) -> tuple[tuple[ProfileParameterSpec, ...], tuple[VmecBoundaryParameterSpec, ...]]:
    """Split mixed reverse-AD parameter specs by parameter kind."""

    profile_specs: list[ProfileParameterSpec] = []
    vmec_specs: list[VmecBoundaryParameterSpec] = []
    for spec in specs:
        if isinstance(spec, ProfileParameterSpec):
            profile_specs.append(spec)
        elif isinstance(spec, VmecBoundaryParameterSpec):
            vmec_specs.append(spec)
        else:
            raise TypeError(f"Unsupported reverse-AD parameter spec type: {type(spec)!r}")
    return tuple(profile_specs), tuple(vmec_specs)


def profile_values_from_config(
    profile_cfg: Mapping[str, object],
    specs: Sequence[ProfileParameterSpec] | None = None,
):
    """Pack profile parameters from a config/profile mapping into a JAX vector."""

    active_specs = (
        tuple(ProfileParameterSpec(name) for name in PROFILE_PARAMETER_ORDER)
        if specs is None
        else tuple(specs)
    )
    return jnp.asarray([float(profile_cfg[spec.name]) for spec in active_specs], dtype=jnp.float64)


def profile_config_with_values(
    profile_cfg: Mapping[str, object],
    specs: Sequence[ProfileParameterSpec],
    values,
) -> dict[str, object]:
    """Return a profile config copy with selected profile parameters replaced."""

    out = dict(profile_cfg)
    for spec, value in zip(specs, values, strict=True):
        out[spec.name] = value
    return out


def vmec_boundary_tuples(
    specs: Sequence[VmecBoundaryParameterSpec],
) -> tuple[tuple[str, int, int], ...]:
    """Return the tuple form expected by existing VMEC autodiff helpers."""

    return tuple(spec.as_tuple() for spec in specs)


def normalize_vmec_boundary_families(families: Sequence[str] | str | None) -> tuple[str, ...]:
    """Return validated VMEC boundary families for harmonic discovery."""

    if families is None:
        return VMEC_BOUNDARY_FAMILIES
    if isinstance(families, str):
        raw_families = tuple(part.strip().upper() for part in families.split(",") if part.strip())
    else:
        raw_families = tuple(str(part).strip().upper() for part in families if str(part).strip())
    if not raw_families:
        raise ValueError("At least one VMEC boundary family is required.")
    unknown = tuple(family for family in raw_families if family not in VMEC_BOUNDARY_FAMILIES)
    if unknown:
        allowed = ", ".join(VMEC_BOUNDARY_FAMILIES)
        raise ValueError(f"Unsupported VMEC boundary families {unknown}; choices are: {allowed}.")
    return raw_families


def _geometry_context_mode_arrays(geometry_context) -> tuple[np.ndarray, np.ndarray]:
    m_arr = np.asarray(jax.device_get(geometry_context.static.modes.m), dtype=int).reshape(-1)
    n_arr = np.asarray(jax.device_get(geometry_context.static.modes.n), dtype=int).reshape(-1)
    if m_arr.shape != n_arr.shape:
        raise ValueError(f"VMEC mode arrays have different shapes: m_shape={m_arr.shape}, n_shape={n_arr.shape}.")
    if m_arr.size == 0:
        raise ValueError("VMEC mode arrays are empty; cannot build boundary parameterization.")
    return m_arr, n_arr


def _geometry_context_boundary_coefficients(geometry_context, family: str) -> np.ndarray:
    boundary_field = "rbc" if family == "RBC" else "zbs"
    return np.asarray(jax.device_get(getattr(geometry_context.boundary, boundary_field)), dtype=float).reshape(-1)


def vmex_packed_boundary_parameter_specs(
    geometry_context,
    *,
    max_mode: int,
    families: Sequence[str] | str | None = None,
    nonzero_only: bool = False,
) -> tuple[VmecBoundaryParameterSpec, ...]:
    """Build VMEX optimizer-style packed fixed-boundary harmonic specs.

    This follows the VMEX mode rule ``m > 0 or (m == 0 and n > 0)`` and uses
    family-major ordering: all selected modes for ``RBC``, then all selected
    modes for ``ZBS``.  Unlike diagnostic discovery, zero coefficients are
    included by default because VMEX optimization can open initially-zero DOFs.
    """

    max_mode_value = int(max_mode)
    if max_mode_value < 0:
        raise ValueError(f"max_mode must be non-negative, got {max_mode!r}.")
    selected_families = normalize_vmec_boundary_families(families)
    m_arr, n_arr = _geometry_context_mode_arrays(geometry_context)
    mode_index_by_pair: dict[tuple[int, int], int] = {}
    for index, (m_value, n_value) in enumerate(zip(m_arr.tolist(), n_arr.tolist(), strict=True)):
        mode_pair = (int(m_value), int(n_value))
        if mode_pair in mode_index_by_pair:
            raise ValueError(f"Duplicate VMEC mode {mode_pair} found in geometry context.")
        mode_index_by_pair[mode_pair] = index

    m_limit = min(max_mode_value, int(np.max(m_arr)))
    n_limit = min(max_mode_value, int(np.max(np.abs(n_arr))))
    specs: list[VmecBoundaryParameterSpec] = []
    for family in selected_families:
        boundary_values = _geometry_context_boundary_coefficients(geometry_context, family) if nonzero_only else None
        if boundary_values is not None and boundary_values.shape != m_arr.shape:
            raise ValueError(
                "VMEC boundary coefficient array shape does not match mode arrays for "
                f"{family}: coefficient_shape={boundary_values.shape}, m_shape={m_arr.shape}."
            )
        for m_value in range(0, m_limit + 1):
            for n_value in range(-n_limit, n_limit + 1):
                if not is_vmex_independent_boundary_mode(m_value, n_value):
                    continue
                mode_index = mode_index_by_pair.get((m_value, n_value))
                if mode_index is None:
                    continue
                if boundary_values is not None:
                    coefficient = float(boundary_values[mode_index])
                    if not np.isfinite(coefficient) or abs(coefficient) == 0.0:
                        continue
                specs.append(VmecBoundaryParameterSpec(family, m_value, n_value))
    if not specs:
        raise ValueError(
            "No VMEX packed boundary harmonics matched the requested selector. "
            "Try a larger max_mode, nonzero_only=False, or a different family list."
        )
    return tuple(specs)


def vmex_boundary_parameterization(
    geometry_context,
    *,
    max_mode: int,
    families: Sequence[str] | str | None = None,
    scale_mode: str = "ess",
    ess_alpha: float = 1.0,
    nonzero_only: bool = False,
) -> VmexBoundaryParameterization:
    """Return VMEX-style packed boundary specs plus optimizer scales."""

    specs = vmex_packed_boundary_parameter_specs(
        geometry_context,
        max_mode=max_mode,
        families=families,
        nonzero_only=nonzero_only,
    )
    scales = vmex_boundary_parameter_scales(specs, scale_mode=scale_mode, ess_alpha=ess_alpha)
    return VmexBoundaryParameterization(
        specs=specs,
        scales=scales,
        scale_mode=scale_mode,
        ess_alpha=ess_alpha,
    )


def discover_vmec_boundary_parameter_specs(
    geometry_context,
    *,
    families: Sequence[str] | str | None = None,
    nonzero_only: bool = True,
) -> tuple[VmecBoundaryParameterSpec, ...]:
    """Discover VMEC boundary harmonics from a geometry autodiff context.

    By default this returns all finite, nonzero selected-family coefficients in
    the order they appear in the VMEC mode arrays.  Set ``nonzero_only=False``
    to include zero-valued harmonics in the same stable order.
    """

    selected_families = normalize_vmec_boundary_families(families)
    m_arr, n_arr = _geometry_context_mode_arrays(geometry_context)
    specs: list[VmecBoundaryParameterSpec] = []
    for family in selected_families:
        boundary_values = _geometry_context_boundary_coefficients(geometry_context, family)
        if boundary_values.shape != m_arr.shape or boundary_values.shape != n_arr.shape:
            raise ValueError(
                "VMEC boundary coefficient array shape does not match mode arrays for "
                f"{family}: coefficient_shape={boundary_values.shape}, "
                f"m_shape={m_arr.shape}, n_shape={n_arr.shape}."
            )
        for m_value, n_value, coefficient in zip(m_arr.tolist(), n_arr.tolist(), boundary_values.tolist()):
            if nonzero_only and not np.isfinite(coefficient):
                continue
            if nonzero_only and abs(float(coefficient)) == 0.0:
                continue
            specs.append(VmecBoundaryParameterSpec(family, int(m_value), int(n_value)))
    if not specs:
        raise ValueError(
            "No VMEC boundary harmonics matched the requested selector. "
            "Try nonzero_only=False or a different family list."
        )
    return tuple(specs)
