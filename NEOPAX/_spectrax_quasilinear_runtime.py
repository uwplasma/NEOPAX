from __future__ import annotations

from typing import Any
import dataclasses

import jax
import jax.numpy as jnp

from ._state import safe_density, safe_temperature


def _get_electron_index(species: Any) -> int | None:
    if species is None or not hasattr(species, "species_idx"):
        return None
    species_idx = getattr(species, "species_idx", {})
    if "e" not in species_idx:
        return None
    return int(species_idx["e"])


def _ion_mask(species: Any, n_species: int) -> jax.Array:
    electron_idx = _get_electron_index(species)
    mask = jnp.ones((n_species,), dtype=bool)
    if electron_idx is not None and 0 <= electron_idx < n_species:
        mask = mask.at[electron_idx].set(False)
    return mask


def _select_radial_grid(geometry: Any, radial_size: int):
    if geometry is None:
        return jnp.linspace(0.0, 1.0, radial_size)
    r_grid = jnp.asarray(getattr(geometry, "r_grid", jnp.linspace(0.0, 1.0, radial_size)))
    if int(r_grid.shape[0]) == radial_size:
        return r_grid
    r_grid_half = getattr(geometry, "r_grid_half", None)
    if r_grid_half is not None:
        r_grid_half = jnp.asarray(r_grid_half)
        if int(r_grid_half.shape[0]) == radial_size:
            return r_grid_half
    return jnp.linspace(float(r_grid[0]), float(r_grid[-1]), radial_size)


def _safe_log_gradient(profile: jax.Array, radial_grid: jax.Array) -> jax.Array:
    profile_safe = jnp.maximum(jnp.asarray(profile), jnp.asarray(1.0e-12, dtype=profile.dtype))
    log_profile = jnp.log(profile_safe)
    dlog = jnp.gradient(log_profile, radial_grid, axis=-1)
    return jnp.nan_to_num(dlog, nan=0.0, posinf=0.0, neginf=0.0)


def _smooth_positive(x: jax.Array, width: float = 1.0e-6) -> jax.Array:
    width_arr = jnp.asarray(width, dtype=x.dtype)
    return width_arr * jnp.log1p(jnp.exp(x / width_arr))


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class SpectraXQuasilinearRuntimeDiagnostics:
    feature_f1: jax.Array
    feature_f2: jax.Array
    total_heat_flux: jax.Array
    ion_drive: jax.Array
    density_drive: jax.Array


def evaluate_spectrax_quasilinear_proxy(
    *,
    state: Any,
    species: Any,
    geometry: Any,
    b0: float,
    b1: float,
    b2: float,
    adiabatic_electrons_only: bool = True,
    particle_flux_scale: float = 0.0,
) -> tuple[dict[str, jax.Array], SpectraXQuasilinearRuntimeDiagnostics]:
    """Smooth runtime placeholder for the future SPECTRAX-GK quasilinear lane.

    This is intentionally a model-local proxy backend:

    - it preserves the intended closure structure
    - it stays differentiable and lagged-response-friendly
    - it does not yet call external SPECTRAX-GK runtime machinery
    """
    density = safe_density(state.density)
    temperature = safe_temperature(state.temperature)
    radial_size = int(density.shape[-1])
    radial_grid = _select_radial_grid(geometry, radial_size)

    ion_mask = _ion_mask(species, int(density.shape[0]))
    ion_weight = ion_mask.astype(density.dtype)
    ion_weight_sum = jnp.maximum(jnp.sum(ion_weight), jnp.asarray(1.0, dtype=density.dtype))

    ion_temperature = temperature * ion_weight[:, None]
    ion_density = density * ion_weight[:, None]

    grad_log_t = _safe_log_gradient(ion_temperature, radial_grid)
    grad_log_n = _safe_log_gradient(ion_density, radial_grid)

    ion_drive = jnp.sum(jnp.abs(grad_log_t), axis=0) / ion_weight_sum
    density_drive = jnp.sum(jnp.abs(grad_log_n), axis=0) / ion_weight_sum

    # Proxy feature lane that keeps the intended fixed-coefficient closure shape.
    positive_centroid_proxy = _smooth_positive(ion_drive + 0.5 * density_drive) + jnp.asarray(1.0e-6, dtype=density.dtype)
    feature_f1 = jnp.log(positive_centroid_proxy)
    feature_f2 = jnp.sqrt(jnp.maximum((ion_drive - density_drive) ** 2, jnp.asarray(0.0, dtype=density.dtype)))
    total_heat_flux = jnp.exp(
        jnp.asarray(b0, dtype=density.dtype)
        + jnp.asarray(b1, dtype=density.dtype) * feature_f1
        + jnp.asarray(b2, dtype=density.dtype) * feature_f2
    )

    ion_density_weights = ion_density / jnp.maximum(
        jnp.sum(ion_density, axis=0, keepdims=True),
        jnp.asarray(1.0e-12, dtype=density.dtype),
    )
    q_turb = ion_density_weights * total_heat_flux[None, :]
    gamma_turb = particle_flux_scale * ion_density_weights * total_heat_flux[None, :]

    electron_idx = _get_electron_index(species)
    if adiabatic_electrons_only and electron_idx is not None and 0 <= electron_idx < density.shape[0]:
        q_turb = q_turb.at[electron_idx].set(jnp.zeros_like(q_turb[electron_idx]))
        gamma_turb = gamma_turb.at[electron_idx].set(jnp.zeros_like(gamma_turb[electron_idx]))

    upar = jnp.zeros_like(density)
    diagnostics = SpectraXQuasilinearRuntimeDiagnostics(
        feature_f1=feature_f1,
        feature_f2=feature_f2,
        total_heat_flux=total_heat_flux,
        ion_drive=ion_drive,
        density_drive=density_drive,
    )
    return {"Gamma": gamma_turb, "Q": q_turb, "Upar": upar}, diagnostics
