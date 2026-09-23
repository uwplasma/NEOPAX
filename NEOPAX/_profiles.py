"""Profile initialization models for NEOPAX.

This module mirrors the TORAX idea of separating profile conditions from solver
logic: profile models build initial T, n, and Er consistently, and the solver
just consumes arrays.
"""

from __future__ import annotations

import abc
import dataclasses
from typing import Any

import jax.numpy as jnp


def _right_face_value_from_cells(profile: jnp.ndarray) -> jnp.ndarray:
    """Estimate the outer-face value from cell-centered samples."""
    prof = jnp.asarray(profile)
    if prof.ndim == 1:
        prof = prof[None, :]
        squeeze = True
    else:
        squeeze = False

    if prof.shape[1] >= 2:
        edge = 1.5 * prof[:, -1] - 0.5 * prof[:, -2]
    else:
        edge = prof[:, -1]
    return edge[0] if squeeze else edge


@dataclasses.dataclass(frozen=True, eq=False)
class ProfileSet:
    """Output container for physical profiles and boundary values."""
    temperature: jnp.ndarray
    density: jnp.ndarray
    Er: jnp.ndarray
    T_edge: jnp.ndarray
    n_edge: jnp.ndarray


class ProfileModel(abc.ABC):
    """Abstract base class for profile construction models."""

    @abc.abstractmethod
    def build(self, field, n_species: int) -> ProfileSet:
        """Build profiles and return ProfileSet with physical arrays and boundaries."""
        pass


@dataclasses.dataclass(frozen=True, eq=False)
class AnalyticalProfileModel(ProfileModel):
    n0: float | tuple[float, ...]
    n_edge: float | tuple[float, ...]
    T0: float | tuple[float, ...]
    T_edge: float | tuple[float, ...]
    c_density: tuple[float, ...] | None = None
    c_temperature: tuple[float, ...] | None = None
    density_shape_power: float | tuple[float, ...] = 2.0
    density_shape_alpha: float | tuple[float, ...] = 1.0
    temperature_shape_power: float | tuple[float, ...] = 2.0
    temperature_shape_alpha: float | tuple[float, ...] = 1.0
    n_scale: float | tuple[float, ...] = 1.0
    T_scale: float | tuple[float, ...] = 1.0
    er0_scale: float = 100.0
    er0_peak_rho: float = 0.8
    charge_qp: tuple[float, ...] | None = None

    def build(self, field, n_species: int) -> ProfileSet:
        if n_species < 1:
            raise ValueError("StandardAnalyticalProfileModel requires at least one species.")

        edge_radius = jnp.maximum(field.r_grid_half[-1], jnp.asarray(1.0e-30, dtype=field.r_grid_half.dtype))
        x = field.r_grid / edge_radius

        # User-facing TOML convention:
        #   density inputs in units of 1e20 m^-3
        #   temperature inputs in keV
        # Internally ProfileSet stores physical density [m^-3] and temperature [eV].
        density_scale_physical = 1.0e20
        temperature_scale_physical = 1.0e3

        electron_index = -1
        if self.charge_qp is not None:
            charge_qp_values = tuple(float(value) for value in self.charge_qp)
            if len(charge_qp_values) >= n_species:
                eidx = min(range(n_species), key=lambda idx: charge_qp_values[idx])
                if charge_qp_values[eidx] < 0.0:
                    electron_index = eidx

        def _expand_species_values(raw, default, include_electron):
            if raw is None:
                return jnp.asarray([default] * n_species, dtype=float)
            if isinstance(raw, (int, float)):
                return jnp.asarray([float(raw)] * n_species, dtype=float)
            arr = jnp.asarray(raw, dtype=float)
            if arr.ndim == 0:
                return jnp.broadcast_to(arr, (n_species,))
            vals_len = int(arr.shape[0])
            if vals_len == n_species:
                return arr
            if (not include_electron) and electron_index >= 0 and vals_len == (n_species - 1):
                out = jnp.ones((n_species,), dtype=arr.dtype)
                k = 0
                for i in range(n_species):
                    if i == electron_index:
                        continue
                    out = out.at[i].set(arr[k])
                    k += 1
                return out
            raise ValueError(
                "Species-resolved array length must be n_species, or n_species-1 when electron is omitted. "
                f"Got length {vals_len} for n_species={n_species}."
            )

        n0 = _expand_species_values(self.n0, default=0.0, include_electron=True)
        n_edge = _expand_species_values(self.n_edge, default=0.0, include_electron=True)
        T0 = _expand_species_values(self.T0, default=0.0, include_electron=True)
        T_edge = _expand_species_values(self.T_edge, default=0.0, include_electron=True)
        density_shape_power = _expand_species_values(self.density_shape_power, default=2.0, include_electron=True)
        density_shape_alpha = _expand_species_values(self.density_shape_alpha, default=1.0, include_electron=True)
        temperature_shape_power = _expand_species_values(self.temperature_shape_power, default=2.0, include_electron=True)
        temperature_shape_alpha = _expand_species_values(self.temperature_shape_alpha, default=1.0, include_electron=True)
        c_density = _expand_species_values(self.c_density, default=1.0, include_electron=False)
        c_temperature = _expand_species_values(self.c_temperature, default=1.0, include_electron=True)
        n_scale = _expand_species_values(self.n_scale, default=1.0, include_electron=False)
        T_scale = _expand_species_values(self.T_scale, default=1.0, include_electron=True)

        Er = self.er0_scale * x * (self.er0_peak_rho - x)

        temperature = jnp.zeros((n_species, field.r_grid.shape[0]))
        density = jnp.zeros((n_species, field.r_grid.shape[0]))

        for i in range(n_species):
            density_shape_i = (1.0 - x ** density_shape_power[i]) ** density_shape_alpha[i]
            temperature_shape_i = (1.0 - x ** temperature_shape_power[i]) ** temperature_shape_alpha[i]
            base_density_i = density_scale_physical * ((n0[i] - n_edge[i]) * density_shape_i + n_edge[i])
            base_temperature_i = temperature_scale_physical * ((T0[i] - T_edge[i]) * temperature_shape_i + T_edge[i])
            temperature = temperature.at[i, :].set(c_temperature[i] * T_scale[i] * base_temperature_i)
            density = density.at[i, :].set(c_density[i] * n_scale[i] * base_density_i)

        T_edge = temperature_scale_physical * T_edge
        n_edge = density_scale_physical * n_edge

        return ProfileSet(
            temperature=temperature,
            density=density,
            Er=Er,
            T_edge=T_edge,
            n_edge=n_edge,
        )


@dataclasses.dataclass(frozen=True, eq=False)
class PrescribedProfileModel(ProfileModel):
    """Profile model using user-provided arrays; edges extracted automatically."""
    temperature: list[list[float]]
    density: list[list[float]]
    Er: list[float]

    def build(self, field, n_species: int) -> ProfileSet:
        temp = jnp.asarray(self.temperature)
        dens = jnp.asarray(self.density)
        er = jnp.asarray(self.Er)

        if temp.shape != dens.shape:
            raise ValueError("temperature and density prescribed arrays must have the same shape")
        if temp.shape[0] != n_species:
            raise ValueError("prescribed temperature/density first axis must match n_species")
        if temp.shape[1] != field.r_grid.shape[0]:
            raise ValueError("prescribed temperature/density radial dimension must match field radial grid")
        if er.shape[0] != field.r_grid.shape[0]:
            raise ValueError("prescribed Er radial dimension must match field radial grid")

        # Cell-centered prescribed profiles do not include the boundary face.
        T_edge = _right_face_value_from_cells(temp)
        n_edge = _right_face_value_from_cells(dens)

        return ProfileSet(
            temperature=temp,
            density=dens,
            Er=er,
            T_edge=T_edge,
            n_edge=n_edge,
        )


def build_profiles(profile_cfg: dict[str, Any], field, n_species: int) -> ProfileSet:

    model = str(profile_cfg.get("model", profile_cfg.get("profiles_model", "standard_analytical"))).lower()

    if model in ("standard_analytical", "standard analytical", "analytical"):
        def _coerce_scalar_or_tuple(raw):
            if isinstance(raw, (int, float)):
                return float(raw)
            return tuple(float(v) for v in raw)

        n0 = _coerce_scalar_or_tuple(profile_cfg.get("n0", profile_cfg.get("ni0", profile_cfg.get("ne0", 4.21e20))))
        n_edge = _coerce_scalar_or_tuple(profile_cfg.get("n_edge", profile_cfg.get("nib", profile_cfg.get("neb", 0.6e20))))
        T0 = _coerce_scalar_or_tuple(profile_cfg.get("T0", profile_cfg.get("ti0", profile_cfg.get("te0", 17.8e3))))
        T_edge = _coerce_scalar_or_tuple(profile_cfg.get("T_edge", profile_cfg.get("tib", profile_cfg.get("teb", 0.7e3))))

        c_density = profile_cfg.get("c_density")
        if c_density is None:
            deuterium_ratio = float(profile_cfg.get("deuterium_ratio", 0.5))
            tritium_ratio = float(profile_cfg.get("tritium_ratio", 0.5))
            if n_species == 3:
                c_density = [1.0, deuterium_ratio, tritium_ratio]

        # Build initial profiles using analytical model
        analytical_model = AnalyticalProfileModel(
            n0=n0,
            n_edge=n_edge,
            T0=T0,
            T_edge=T_edge,
            c_density=None if c_density is None else tuple(float(v) for v in c_density),
            c_temperature=None
            if profile_cfg.get("c_temperature") is None
            else tuple(float(v) for v in profile_cfg.get("c_temperature")),
            density_shape_power=_coerce_scalar_or_tuple(profile_cfg.get("density_shape_power", 2.0)),
            density_shape_alpha=_coerce_scalar_or_tuple(profile_cfg.get("density_shape_alpha", 1.0)),
            temperature_shape_power=_coerce_scalar_or_tuple(profile_cfg.get("temperature_shape_power", 2.0)),
            temperature_shape_alpha=_coerce_scalar_or_tuple(profile_cfg.get("temperature_shape_alpha", 1.0)),
            n_scale=(
                float(profile_cfg.get("n_scale", 1.0))
                if isinstance(profile_cfg.get("n_scale", 1.0), (int, float))
                else tuple(float(v) for v in profile_cfg.get("n_scale", (1.0,)))
            ),
            T_scale=(
                float(profile_cfg.get("T_scale", 1.0))
                if isinstance(profile_cfg.get("T_scale", 1.0), (int, float))
                else tuple(float(v) for v in profile_cfg.get("T_scale", (1.0,)))
            ),
            er0_scale=float(profile_cfg.get("er0_scale", 100.0)),
            er0_peak_rho=float(profile_cfg.get("er0_peak_rho", 0.8)),
            charge_qp=None
            if profile_cfg.get("charge_qp") is None
            else tuple(float(v) for v in profile_cfg.get("charge_qp")),
        )
        profile_set = analytical_model.build(field, n_species)

        # Optionally override Er with ambipolar root initialization
        er_init_mode = str(profile_cfg.get("er_initialization_mode", "analytical")).lower()
        if er_init_mode == "ambipolar_root":
            # Import here to avoid circular import
            from ._ambipolarity import find_ambipolar_Er_min_entropy_jit
            # Dummy Gamma/entropy functions: user must provide get_Neoclassical_Fluxes, species, grid, field, database
            # For now, raise NotImplementedError to indicate where user must connect physics
            raise NotImplementedError("Ambipolar root initialization for Er requires access to get_Neoclassical_Fluxes, species, grid, field, database, and initial profiles. Implement this logic in your driver script or extend build_profiles to accept these arguments.")
            # Example (pseudo-code):
            # for i in range(n_radial):
            #     def gamma(er): ...
            #     def entropy(er): ...
            #     profile_set.Er[i] = find_ambipolar_Er_min_entropy_jit(gamma, entropy, ...)[0]

        return profile_set

    if model in ("prescribed", "given"):
        return PrescribedProfileModel(
            temperature=profile_cfg["temperature"],
            density=profile_cfg["density"],
            Er=profile_cfg["Er"],
        ).build(field, n_species)

    raise ValueError(f"Unknown profiles model: {model}")
