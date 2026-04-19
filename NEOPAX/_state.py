import jax.numpy as jnp
import dataclasses
import jax
from jaxtyping import Array, Float

from ._constants import Boltzmann


JOULE_PER_EV = 11606 * Boltzmann
JOULE_PER_KEV = 1.0e3 * JOULE_PER_EV


@jax.jit
def get_v_thermal(mass, temperature):
    mass_arr = jnp.asarray(mass)
    temperature_arr = jnp.asarray(temperature)

    # Align mass with temperature leading species axis, e.g.:
    # (n_species,) -> (n_species, 1) when temperature is (n_species, n_r).
    if mass_arr.ndim < temperature_arr.ndim:
        mass_arr = jnp.reshape(
            mass_arr,
            mass_arr.shape + (1,) * (temperature_arr.ndim - mass_arr.ndim),
        )

    return jnp.sqrt(2 * temperature_arr * JOULE_PER_KEV / mass_arr)

@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class TransportState:
    """
    JAX-compatible transport state for arbitrary number of species.
    All fields are JAX arrays for differentiability and vmap support.
    """
    density: Float[Array, "n_species n_radial"]
    pressure: Float[Array, "n_species n_radial"]
    Er: Float[Array, "n_radial"]

    @property
    def temperature(self):
        eps = jnp.asarray(1.0e-20, dtype=self.pressure.dtype)
        return self.pressure / jnp.maximum(self.density, eps)
