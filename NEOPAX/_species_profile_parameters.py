"""Small differentiable map from independent species DOFs to initial profiles.

The transport/selected-root reverse rules consume this map at their existing
initial-profile boundary. No transport solver or runtime is captured here.
"""

from __future__ import annotations

import jax.numpy as jnp

from ._reverse_ad_parameters import (
    PROFILE_PARAMETER_ORDER, configured_profile_parameter_order,
    split_profile_parameter_name,
)

DEFAULTS = dict(n0=4.21, T0=17.8, density_shape_power=2., temperature_shape_power=2.,
                density_shape_alpha=1., temperature_shape_alpha=1.)


def species_index(token, config):
    species = config.get("species", {})
    names = tuple(species.get("names", ()))
    count = int(species.get("n_species", len(names)))
    if token in names:
        return names.index(token)
    if str(token).isdigit() and 0 <= int(token) < count:
        return int(token)
    raise ValueError(f"Unknown profile species {token!r}; configured names are {names}.")


def species_profile_values(config, profile_cfg=None):
    """Read physical analytical-profile parameters in the configured order."""
    cfg = config.get("profiles", {}) if profile_cfg is None else profile_cfg
    values = []
    for name in configured_profile_parameter_order(config):
        base, token = split_profile_parameter_name(name)
        raw = jnp.asarray(cfg.get(base, DEFAULTS[base]))
        values.append(raw if raw.ndim == 0 else raw[0 if token is None else species_index(token, config)])
    return jnp.asarray(values)


def species_profile_config(parameter_values, *, config, profile_cfg):
    """Replace only the selected species slots, preserving all other inputs."""
    labels = configured_profile_parameter_order(config)
    values = jnp.asarray(parameter_values)
    if values.shape != (len(labels),):
        raise ValueError(f"Species profile vector has shape {values.shape}; expected ({len(labels)},).")
    count = int(config["species"].get("n_species", len(config["species"]["names"])))
    result = dict(profile_cfg)
    for base in PROFILE_PARAMETER_ORDER:
        raw = jnp.asarray(profile_cfg.get(base, DEFAULTS[base]), dtype=values.dtype)
        result[base] = jnp.broadcast_to(raw, (count,))
    for name, value in zip(labels, values, strict=True):
        base, token = split_profile_parameter_name(name)
        if token is None:
            raise ValueError("Species profile coordinates require explicit species indices.")
        result[base] = result[base].at[species_index(token, config)].set(value)
    return result


def quasineutral_initial_density(density, config):
    """Derive electron density from ion DOFs before forming pressure/root inputs."""
    charges = jnp.asarray(config["species"]["charge_qp"], dtype=density.dtype)
    names = tuple(config["species"]["names"])
    electron = names.index("e")
    ions = tuple(i for i, charge in enumerate(config["species"]["charge_qp"]) if charge > 0)
    electron_density = -jnp.sum(charges[jnp.asarray(ions), None] * density[jnp.asarray(ions)], axis=0) / charges[electron]
    return density.at[electron].set(electron_density)
