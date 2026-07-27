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
)

VMEC_BOUNDARY_FAMILIES: tuple[str, ...] = ("RBC", "ZBS")


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
    m_arr = np.asarray(jax.device_get(geometry_context.static.modes.m), dtype=int).reshape(-1)
    n_arr = np.asarray(jax.device_get(geometry_context.static.modes.n), dtype=int).reshape(-1)
    specs: list[VmecBoundaryParameterSpec] = []
    for family in selected_families:
        boundary_field = "rbc" if family == "RBC" else "zbs"
        boundary_values = np.asarray(
            jax.device_get(getattr(geometry_context.boundary, boundary_field)),
            dtype=float,
        ).reshape(-1)
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
