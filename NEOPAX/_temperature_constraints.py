"""Optional temperature and electric-work closures for the pressure equations."""

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from ._state import safe_density


@dataclasses.dataclass(frozen=True, eq=False)
class MatchedHeliumTemperatureProfile:
    """Keep fixed profiles and the live He <- D constraint in one pytree.

    This travels through the existing fixed-temperature projection metadata,
    including solver packing and checkpoint replay. Species indices are static;
    the fixed profile retains its usual differentiable array leaves.
    """

    profile: Any
    helium_index: int
    deuterium_index: int


jax.tree_util.register_dataclass(
    MatchedHeliumTemperatureProfile,
    data_fields=("profile",),
    meta_fields=("helium_index", "deuterium_index"),
)


def helium_temperature_match_indices(config, species_names):
    """Resolve the opt-in closure, rejecting a competing helium equation."""
    equations = config.get("equations", {})
    enabled = equations.get("match_THe", False)
    if not isinstance(enabled, bool):
        raise ValueError("equations.match_THe must be true or false.")
    if not enabled:
        return None
    names = tuple(species_names)
    if "He" not in names or "D" not in names:
        raise ValueError("equations.match_THe=true requires species named 'He' and 'D'.")
    he, d = names.index("He"), names.index("D")
    active = equations.get("toggle_temperature", [True] * len(names))
    if len(active) != len(names) or active[he]:
        raise ValueError(
            "equations.match_THe=true requires toggle_temperature=false for He; "
            "helium follows deuterium without an independent temperature equation."
        )
    return he, d


def helium_work_match_indices(config, species_names):
    """Resolve NTSS's work closure without affecting the default source map."""
    equations = config.get("equations", {})
    enabled = equations.get("match_workHe", False)
    if not isinstance(enabled, bool):
        raise ValueError("equations.match_workHe must be true or false.")
    if not enabled:
        return None
    names = tuple(species_names)
    if any(name not in names for name in ("e", "D", "T", "He")):
        raise ValueError(
            "equations.match_workHe=true requires species named 'e', 'D', 'T', and 'He'."
        )
    he = names.index("He")
    active = equations.get("toggle_temperature", [True] * len(names))
    if len(active) != len(names) or active[he]:
        raise ValueError(
            "equations.match_workHe=true requires toggle_temperature=false for He; "
            "helium electric work is assigned to the D/T energy equations."
        )
    return he, names.index("D"), names.index("T")


def matched_helium_temperature_profile(profile, indices):
    if indices is None:
        return profile
    return MatchedHeliumTemperatureProfile(profile, *indices)


def fixed_temperature_values(profile):
    return profile.profile if isinstance(profile, MatchedHeliumTemperatureProfile) else profile


def match_helium_temperature_pressure(pressure, density, profile, density_floor):
    """Apply the live closure after floors, supporting saved-state batch axes."""
    if not isinstance(profile, MatchedHeliumTemperatureProfile):
        return pressure
    n = safe_density(density) if density_floor is None else safe_density(density, density_floor)
    he, d = profile.helium_index, profile.deuterium_index
    donor_temperature = pressure[..., d, :] / n[..., d, :]
    return pressure.at[..., he, :].set(n[..., he, :] * donor_temperature)


def match_helium_temperature_state(state, profile, density_floor):
    if not isinstance(profile, MatchedHeliumTemperatureProfile):
        return state
    return dataclasses.replace(
        state,
        pressure=match_helium_temperature_pressure(
            state.pressure, state.density, profile, density_floor
        ),
    )


def match_helium_temperature_row(value, indices):
    """Inherit D's species-specific thermal BC or floor when matching is on."""
    if indices is None or value is None:
        return value
    array = jnp.asarray(value)
    if array.ndim == 0:
        return value
    he, d = indices
    return array.at[he].set(array[d])


def match_helium_temperature_boundary(boundary, indices):
    if indices is None or boundary is None:
        return boundary
    return dataclasses.replace(
        boundary,
        **{
            name: match_helium_temperature_row(getattr(boundary, name), indices)
            for name in (
                "left_value", "right_value", "left_gradient", "right_gradient",
                "left_decay_length", "right_decay_length", "reference_profiles",
            )
        },
    )
