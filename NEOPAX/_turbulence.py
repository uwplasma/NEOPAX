import jax
import jax.numpy as jnp
from jax import config, jit

from ._cell_variable import get_gradient_density, get_gradient_temperature

DENSITY_STATE_TO_PHYSICAL = 1.0e20
TEMPERATURE_STATE_TO_PHYSICAL = 1.0e3


config.update("jax_enable_x64", True)


@jit
def get_Turbulent_Fluxes_Analytical(
    species,
    grid,
    chi_temperature,
    chi_density,
    temperature,
    density,
    field,
    density_right_constraint=None,
    density_right_grad_constraint=None,
    temperature_right_constraint=None,
    temperature_right_grad_constraint=None,
):
    """Analytical diffusive turbulent flux model.

    Per species:
      Gamma_a = -chi_density[a] * d n_a / dr
      Q_a     = -chi_temperature[a] * d T_a / dr
    """
    temperature = TEMPERATURE_STATE_TO_PHYSICAL * temperature
    density = DENSITY_STATE_TO_PHYSICAL * density
    if density_right_constraint is not None:
        density_right_constraint = DENSITY_STATE_TO_PHYSICAL * density_right_constraint
    if density_right_grad_constraint is not None:
        density_right_grad_constraint = DENSITY_STATE_TO_PHYSICAL * density_right_grad_constraint
    if temperature_right_constraint is not None:
        temperature_right_constraint = TEMPERATURE_STATE_TO_PHYSICAL * temperature_right_constraint
    if temperature_right_grad_constraint is not None:
        temperature_right_grad_constraint = TEMPERATURE_STATE_TO_PHYSICAL * temperature_right_grad_constraint

    n_species = int(temperature.shape[0])

    def _as_species_constraint(arr, fallback):
        if arr is None:
            return fallback
        out = jnp.asarray(arr)
        if out.ndim == 0:
            return jnp.repeat(out[None], n_species, axis=0)
        if out.shape[0] < n_species:
            out = jnp.pad(out, (0, n_species - out.shape[0]), mode="edge")
        return out[:n_species]

    n_right = _as_species_constraint(density_right_constraint, density[:, -1])
    n_right_grad = _as_species_constraint(density_right_grad_constraint, jnp.zeros_like(density[:, -1]))
    t_right = _as_species_constraint(temperature_right_constraint, temperature[:, -1])
    t_right_grad = _as_species_constraint(temperature_right_grad_constraint, jnp.zeros_like(temperature[:, -1]))

    def per_species(a, temperature_a, density_a, n_right_a, n_grad_a, t_right_a, t_grad_a, chi_t_a, chi_n_a):
        del a
        dndr = get_gradient_density(
            density_a,
            field.r_grid,
            field.r_grid_half,
            field.dr,
            right_face_constraint=n_right_a,
            right_face_grad_constraint=n_grad_a,
        )
        dTdr = get_gradient_temperature(
            temperature_a,
            field.r_grid,
            field.r_grid_half,
            field.dr,
            right_face_constraint=t_right_a,
            right_face_grad_constraint=t_grad_a,
        )
        gamma = -chi_n_a * dndr
        q = -chi_t_a * dTdr
        return gamma, q

    gamma_turb, q_turb = jax.vmap(per_species, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0))(
        grid.species_indeces,
        temperature,
        density,
        n_right,
        n_right_grad,
        t_right,
        t_right_grad,
        chi_temperature,
        chi_density,
    )
    return gamma_turb, q_turb
