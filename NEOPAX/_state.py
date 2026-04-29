import jax.numpy as jnp
import dataclasses
import jax
from jaxtyping import Array, Float

from ._constants import Boltzmann


JOULE_PER_EV = 11606 * Boltzmann
JOULE_PER_KEV = 1.0e3 * JOULE_PER_EV
DEFAULT_TRANSPORT_DENSITY_FLOOR = 1.0e-6
DEFAULT_TRANSPORT_TEMPERATURE_FLOOR = None


def _broadcast_species_floor(reference: Array, floor) -> Array:
    floor_arr = jnp.asarray(floor, dtype=reference.dtype)
    if floor_arr.ndim == 0:
        return floor_arr
    if floor_arr.ndim == 1 and reference.ndim >= 2:
        return floor_arr[:, None]
    return floor_arr


def safe_density(density: Array, floor=DEFAULT_TRANSPORT_DENSITY_FLOOR) -> Array:
    return jnp.maximum(density, _broadcast_species_floor(density, floor))


def safe_temperature(temperature: Array, floor) -> Array:
    if floor is None:
        return temperature
    return jnp.maximum(temperature, _broadcast_species_floor(temperature, floor))


def apply_transport_density_floor(state, density_floor=DEFAULT_TRANSPORT_DENSITY_FLOOR):
    density = getattr(state, "density", None)
    if density is None:
        return state
    return dataclasses.replace(state, density=safe_density(density, density_floor))


def apply_transport_temperature_floor(
    state,
    temperature_floor=DEFAULT_TRANSPORT_TEMPERATURE_FLOOR,
    density_floor=DEFAULT_TRANSPORT_DENSITY_FLOOR,
):
    if temperature_floor is None:
        return state
    density = getattr(state, "density", None)
    pressure = getattr(state, "pressure", None)
    if density is None or pressure is None:
        return state
    safe_n = safe_density(density, density_floor)
    safe_t = safe_temperature(pressure / safe_n, temperature_floor)
    return dataclasses.replace(state, pressure=safe_n * safe_t)


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
        return self.pressure / safe_density(self.density)
