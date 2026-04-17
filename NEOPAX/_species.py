
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
JOULE_PER_KEV = 1.0e3 * JOULE_PER_EV
EV_PER_JOULE = 1 / JOULE_PER_EV
STATE_DENSITY_TO_PHYSICAL = 1.0e20
STATE_TEMPERATURE_TO_EV = 1.0e3


#Get thermodynamical forces
@jit
def get_Thermodynamical_Forces_A1(q,n,T,dndr,dTdr,Er):
    A1=dndr/n-1.5*dTdr/T-Er*q/(T*elementary_charge)
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

    @property
    def species_idx(self) -> dict:
        """Return a mapping from species name to index."""
        return {name: i for i, name in enumerate(self.names)}

    @property
    def ion_indices(self) -> tuple:
        """Return a tuple of all species indices except the electron ('e')."""
        if "e" not in self.names:
            raise ValueError("'e' (electron) must be present in species names for ion_indices.")
        eidx = self.names.index("e")
        return tuple(i for i in range(self.number_species) if i != eidx)


# Register Species as a pytree, treating 'names' and 'is_frozen' as static (auxiliary) data
def _species_flatten(s):
    # Only array fields are treated as pytree leaves
    children = (s.number_species, s.species_indices, s.mass_mp, s.charge_qp)
    aux_data = {'names': s.names, 'is_frozen': s.is_frozen}
    return children, aux_data

def _species_unflatten(aux_data, children):
    number_species, species_indices, mass_mp, charge_qp = children
    return Species(
        number_species=number_species,
        species_indices=species_indices,
        mass_mp=mass_mp,
        charge_qp=charge_qp,
        names=aux_data.get('names', ()),
        is_frozen=aux_data.get('is_frozen', None)
    )

jax.tree_util.register_pytree_node(Species, _species_flatten, _species_unflatten)



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
    nb = STATE_DENSITY_TO_PHYSICAL * density[species_b, r_index]
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
    nb = STATE_DENSITY_TO_PHYSICAL * density[species_b, r_index]
    vtb = v_thermal[species_b, r_index]
    return (
        2 * gamma_ab(species, species_a, species_b, v, r_index, temperature, density) * nb / v**3
        * chandrasekhar(v / vtb)
    )


def coulomb_logarithm(species: Species, species_a: int, species_b: int, r_index: int,
                      temperature, density) -> float:
    """Coulomb logarithm for collisions between species a and b."""
    Te_eV = STATE_TEMPERATURE_TO_EV * temperature[0, r_index]
    ne_m3 = STATE_DENSITY_TO_PHYSICAL * density[0, r_index]
    lnL = 32.2 + 1.15 * jnp.log10(Te_eV**2 / ne_m3)
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
        (STATE_DENSITY_TO_PHYSICAL * density[:, r_index])
        / (temperature[:, r_index] * JOULE_PER_KEV)
        * species.charge**2
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
