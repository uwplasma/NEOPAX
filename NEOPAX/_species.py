
from functools import partial
from jaxtyping import Array, Float  # https://github.com/google/jaxtyping
import dataclasses
import jax.numpy as jnp
import jax
from jax import config
# to use higher precision
config.update("jax_enable_x64", True)
from jax import jit
import jax.numpy as jnp
from ._constants import Boltzmann, elementary_charge, epsilon_0, hbar, proton_mass
from ._cell_variable import get_gradient_density, get_gradient_temperature
from ._state import get_v_thermal
import interpax
#from _io import grid,field



###This module uses some functions adapted from JAX-MONKES by R. Colin, but class structure was revamped to fit the overall package better#####

JOULE_PER_EV = 11606 * Boltzmann
EV_PER_JOULE = 1 / JOULE_PER_EV


#Get thermodynamical forces
@jit
def get_Thermodynamical_Forces_A1(q,n,T,dndr,dTdr,Er):
    A1=dndr/n-1.5*dTdr/T-1.e+3*Er*q/(T*elementary_charge)    
    return A1

#Get thermodynamical forces
@jit
def get_Thermodynamical_Forces_A2(T,dTdr):
    A2=dTdr/T
    return A2

#Get thermodynamical forces
@jit
def get_Thermodynamical_Forces_A3(Er):
    A3=Er*0.      
    return A3


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class Species:
    """
    JAX-compatible species container for arbitrary number of species.
    All fields are JAX arrays for differentiability and vmap support.
    """
    number_species: int
    species_indices: Array  # shape (n_species,)
    mass_mp: Float[Array, "n_species"]
    charge_qp: Float[Array, "n_species"]
    names: tuple[str, ...] = ()  # Optional: names for each species
    is_frozen: Array = None  # Optional: mask for frozen species (bool or int)

    @property
    def charge(self):
        return self.charge_qp * elementary_charge

    @property
    def mass(self):
        return self.mass_mp * proton_mass
    

def collisionality(species_a: int, species: Species, v: float, r_index: int, density, temperature, v_thermal) -> float:
    """Collisionality of species_a against all species."""
    nu = jnp.sum(
        jax.vmap(nuD_ab, in_axes=(None, None, 0, None, None, None, None, None))(
            species, species_a, species.species_indices, v, r_index, density, temperature, v_thermal
        ),
        axis=0,
    )
    return nu


def nuD_ab(species: Species, species_a: int, species_b: int, v: float, r_index: int,
           density, temperature, v_thermal) -> float:
    """Pairwise pitch-angle scattering frequency for species a against species b."""
    nb = density[species_b, r_index]
    vtb = v_thermal[species_b, r_index]
    prefactor = gamma_ab(species, species_a, species_b, v, r_index, temperature, density) * nb / v**3
    erf_part = jax.scipy.special.erf(v / vtb) - chandrasekhar(v / vtb)
    return prefactor * erf_part


def gamma_ab(species: Species, species_a: int, species_b: int, v: float, r_index: int,
             temperature, density) -> float:
    """Prefactor for pairwise collisionality."""
    lnlambda = coulomb_logarithm(species, species_a, species_b, r_index, temperature, density)
    ea, eb = species.charge[species_a], species.charge[species_b]
    ma = species.mass[species_a]
    return ea**2 * eb**2 * lnlambda / (4 * jnp.pi * epsilon_0**2 * ma**2)


def nupar_ab(species: Species, species_a: int, species_b: int, v: float, r_index: int,
             density, temperature, v_thermal) -> float:
    """Parallel collisionality."""
    nb = density[species_b, r_index]
    vtb = v_thermal[species_b, r_index]
    return (
        2 * gamma_ab(species, species_a, species_b, v, r_index, temperature, density) * nb / v**3
        * chandrasekhar(v / vtb)
    )


def coulomb_logarithm(species: Species, species_a: int, species_b: int, r_index: int,
                      temperature, density) -> float:
    """Coulomb logarithm for collisions between species a and b."""
    lnL = 32.2 + 1.15 * jnp.log10(temperature[0, r_index]**2 / density[0, r_index])
    return lnL


def impact_parameter(species: Species, species_a: int, species_b: int, r_index: int,
                     density, temperature, v_thermal) -> float:
    """Impact parameters for classical Coulomb collision."""
    bmin = jnp.maximum(
        impact_parameter_perp(species, species_a, species_b, r_index, v_thermal),
        debroglie_length(species, species_a, species_b, r_index, v_thermal),
    )
    bmax = debye_length(species, r_index, density, temperature)
    return bmin, bmax


def impact_parameter_perp(species: Species, species_a: int, species_b: int, r_index: int,
                          v_thermal) -> float:
    """Distance of the closest approach for a 90° Coulomb collision."""
    m_reduced = (
        species.mass[species_a]
        * species.mass[species_b]
        / (species.mass[species_a] + species.mass[species_b])
    )
    v_th = jnp.sqrt(v_thermal[species_a, r_index] * v_thermal[species_b, r_index])
    return (
        species.charge[species_a]
        * species.charge[species_a]
        / (4 * jnp.pi * epsilon_0 * m_reduced * v_th**2)
    )


def debroglie_length(species: Species, species_a: int, species_b: int, r_index: int,
                    v_thermal) -> float:
    """Thermal DeBroglie wavelength."""
    m_reduced = (
        species.mass[species_a]
        * species.mass[species_b]
        / (species.mass[species_a] + species.mass[species_b])
    )
    v_th = jnp.sqrt(v_thermal[species_a, r_index] * v_thermal[species_b, r_index])
    return hbar / (2 * m_reduced * v_th)


def debye_length(species: Species, r_index: int, density, temperature) -> float:
    """Scale length for charge screening."""
    den = jnp.sum(
        density[:, r_index] / (temperature[:, r_index] * JOULE_PER_EV) * species.charge**2
    )
    return jnp.sqrt(epsilon_0 / den)


def chandrasekhar(x: jax.Array) -> jax.Array:
    """Chandrasekhar function."""
    return (
        jax.scipy.special.erf(x) - 2 * x / jnp.sqrt(jnp.pi) * jnp.exp(-(x**2))
    ) / (2 * x**2)


def _dchandrasekhar(x):
    return 2 / jnp.sqrt(jnp.pi) * jnp.exp(-(x**2)) - 2 / x * chandrasekhar(x)



def get_species_idx(name: str, names: tuple[str, ...]) -> int:
    """Return the index of a species by name from a tuple of species names."""
    return names.index(name)