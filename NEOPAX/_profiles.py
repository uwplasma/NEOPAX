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
    n0: float
    n_edge: float
    T0: float
    T_edge: float
    c_density: tuple[float, ...] | None = None
    c_temperature: tuple[float, ...] | None = None
    density_shape_power: float = 2.0
    temperature_shape_power: float = 2.0
    n_scale: float = 1.0
    T_scale: float = 1.0
    er0_scale: float = 100.0
    er0_peak_rho: float = 0.8
    charge_qp: tuple[float, ...] | None = None

    def build(self, field, n_species: int) -> ProfileSet:
        if n_species < 1:
            raise ValueError("StandardAnalyticalProfileModel requires at least one species.")

        x = field.r_grid / field.a_b
        density_shape = 1.0 - x ** self.density_shape_power
        temperature_shape = 1.0 - x ** self.temperature_shape_power

        base_density = self.n_scale * ((self.n0-self.n_edge) * density_shape + self.n_edge)
        base_temperature = self.T_scale * ((self.T0-self.T_edge) * temperature_shape + self.T_edge)

        charge_qp = None
        electron_index = -1
        if self.charge_qp is not None:
            charge_qp = jnp.asarray(self.charge_qp, dtype=float)
            if charge_qp.shape[0] >= n_species:
                charge_qp = charge_qp[:n_species]
                eidx = int(jnp.argmin(charge_qp))
                if float(charge_qp[eidx]) < 0.0:
                    electron_index = eidx

        def _expand_concentration(raw, default, include_electron):
            if raw is None:
                return jnp.asarray([default] * n_species, dtype=float)
            vals = [float(v) for v in raw]
            if len(vals) == n_species:
                return jnp.asarray(vals, dtype=float)
            if (not include_electron) and electron_index >= 0 and len(vals) == (n_species - 1):
                out = [1.0] * n_species
                k = 0
                for i in range(n_species):
                    if i == electron_index:
                        continue
                    out[i] = vals[k]
                    k += 1
                return jnp.asarray(out, dtype=float)
            raise ValueError(
                "Concentration array length must be n_species, or n_species-1 when electron is omitted."
            )

        c_density = _expand_concentration(self.c_density, default=1.0, include_electron=False)
        c_temperature = _expand_concentration(self.c_temperature, default=1.0, include_electron=True)

        Er = self.er0_scale * field.rho_grid * (self.er0_peak_rho - field.rho_grid)

        temperature = jnp.zeros((n_species, field.r_grid.shape[0]))
        density = jnp.zeros((n_species, field.r_grid.shape[0]))

        for i in range(n_species):
            temperature = temperature.at[i, :].set(c_temperature[i] * base_temperature)
            density = density.at[i, :].set(c_density[i] * base_density)

        T_edge = temperature[:, -1]
        n_edge = density[:, -1]

        # Center/edge boundary initialization.
        fr = 0.0
        temperature = temperature.at[:, 0].set((4.0 * temperature[:, 1] - temperature[:, 2] - fr * 2.0 * field.dr) / 3.0)
        density = density.at[:, 0].set((4.0 * density[:, 1] - density[:, 2] - fr * 2.0 * field.dr) / 3.0)
        temperature = temperature.at[:, -1].set(T_edge)
        density = density.at[:, -1].set(n_edge)

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

        # Extract edge values directly from profile arrays (not user inputs)
        T_edge = temp[:, -1]
        n_edge = dens[:, -1]

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
        n0 = float(profile_cfg.get("n0", profile_cfg.get("ni0", profile_cfg.get("ne0", 4.21e20))))
        n_edge = float(profile_cfg.get("n_edge", profile_cfg.get("nib", profile_cfg.get("neb", 0.6e20))))
        T0 = float(profile_cfg.get("T0", profile_cfg.get("ti0", profile_cfg.get("te0", 17.8e3))))
        T_edge = float(profile_cfg.get("T_edge", profile_cfg.get("tib", profile_cfg.get("teb", 0.7e3))))

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
            density_shape_power=float(profile_cfg.get("density_shape_power", 2.0)),
            temperature_shape_power=float(profile_cfg.get("temperature_shape_power", 2.0)),
            n_scale=float(profile_cfg.get("n_scale", 1.0)),
            T_scale=float(profile_cfg.get("T_scale", 1.0)),
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
