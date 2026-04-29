
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

COULOMB_LOG_MODEL_DEFAULT = 0
COULOMB_LOG_MODEL_NTSS_LEGACY = 1
COULOMB_LOG_MODEL_FIRST_PRINCIPLES = 2


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



def collisionality(
    species_a: int,
    species: Species,
    v: float,
    r_index: int,
    density,
    temperature,
    v_thermal,
    coulomb_log_model: int = COULOMB_LOG_MODEL_DEFAULT,
) -> float:
    """Collisionality of species_a against all species."""
    nu = jnp.sum(
        jax.vmap(nuD_ab, in_axes=(None, None, 0, None, None, None, None, None, None))(
            species, species_a, species.species_indices, v, r_index, density, temperature, v_thermal, coulomb_log_model
        ),
        axis=0,
    )
    return nu


def collisionality_local(
    species_a: int,
    species: Species,
    v: float,
    density_local,
    temperature_local,
    v_thermal_local,
    coulomb_log_model: int = COULOMB_LOG_MODEL_DEFAULT,
) -> float:
    """Collisionality of species_a against all species using local profiles at one radius."""
    nu = jnp.sum(
        jax.vmap(nuD_ab_local, in_axes=(None, None, 0, None, None, None, None, None))(
            species, species_a, species.species_indices, v, density_local, temperature_local, v_thermal_local, coulomb_log_model
        ),
        axis=0,
    )
    return nu


def collisionality_ntss_like(
    species_a: int,
    species: Species,
    v: float,
    r_index: int,
    density,
    temperature,
    v_thermal,
) -> float:
    """NTSSfusion-like thermal-speed effective collision frequency cnue(vth)."""
    del v
    tau_ab = jax.vmap(
        tau_ab_ntss_like,
        in_axes=(None, None, 0, None, None, None, None),
    )(species, species_a, species.species_indices, r_index, density, temperature, v_thermal)
    kernel = jax.vmap(
        ntss_collision_kernel_thermal,
        in_axes=(None, None, 0, None, None),
    )(species, species_a, species.species_indices, r_index, temperature)
    return 0.75 * jnp.sqrt(jnp.pi) * jnp.sum(kernel / tau_ab, axis=0)


def collisionality_ntss_like_local(
    species_a: int,
    species: Species,
    v: float,
    density_local,
    temperature_local,
    v_thermal_local,
) -> float:
    """Local NTSSfusion-like thermal-speed effective collision frequency cnue(vth)."""
    del v
    tau_ab = jax.vmap(
        tau_ab_ntss_like_local,
        in_axes=(None, None, 0, None, None, None),
    )(species, species_a, species.species_indices, density_local, temperature_local, v_thermal_local)
    kernel = jax.vmap(
        ntss_collision_kernel_thermal_local,
        in_axes=(None, None, 0, None),
    )(species, species_a, species.species_indices, temperature_local)
    return 0.75 * jnp.sqrt(jnp.pi) * jnp.sum(kernel / tau_ab, axis=0)


def _electron_index(species: Species) -> int:
    if species.names and "e" in species.names:
        return species.names.index("e")
    negative = jnp.where(species.charge_qp < 0)[0]
    return int(negative[0]) if negative.size else 0


def zeff(species: Species, density, r_index: int) -> float:
    """Effective charge Zeff = sum_i n_i Z_i^2 / n_e using ion species only."""
    eidx = _electron_index(species)
    ne = jnp.maximum(STATE_DENSITY_TO_PHYSICAL * density[eidx, r_index], 1.0e-30)
    ion_mask = species.charge_qp > 0
    n_phys = STATE_DENSITY_TO_PHYSICAL * density[:, r_index]
    return jnp.sum(jnp.where(ion_mask, n_phys * species.charge_qp**2, 0.0)) / ne


def zeff_local(species: Species, density_local) -> float:
    """Local effective charge Zeff = sum_i n_i Z_i^2 / n_e using ion species only."""
    eidx = _electron_index(species)
    ne = jnp.maximum(STATE_DENSITY_TO_PHYSICAL * density_local[eidx], 1.0e-30)
    ion_mask = species.charge_qp > 0
    n_phys = STATE_DENSITY_TO_PHYSICAL * density_local
    return jnp.sum(jnp.where(ion_mask, n_phys * species.charge_qp**2, 0.0)) / ne


def collisionality_ntss_zeff(
    species_a: int,
    species: Species,
    v: float,
    r_index: int,
    density,
    temperature,
    v_thermal,
) -> float:
    """Legacy NTSSfusion-like simplified electron collisionality using Zeff.

    For electrons this follows the compact electron + Zeff form used in the
    legacy DKES path. For ions we keep the explicit multispecies sum.
    """
    eidx = _electron_index(species)
    def ion_like_branch(_):
        return collisionality(
            species_a,
            species,
            v,
            r_index,
            density,
            temperature,
            v_thermal,
            COULOMB_LOG_MODEL_NTSS_LEGACY,
        )

    def electron_branch(_):
        ne = STATE_DENSITY_TO_PHYSICAL * density[eidx, r_index]
        vte = jnp.maximum(v_thermal[eidx, r_index], 1.0e-30)
        prefactor = gamma_ab(
            species,
            eidx,
            eidx,
            v,
            r_index,
            temperature,
            density,
            v_thermal,
            COULOMB_LOG_MODEL_NTSS_LEGACY,
        ) * ne / jnp.maximum(v, 1.0e-30) ** 3
        kernel_e = ntss_collision_kernel(species, eidx, eidx, v, vte)
        return prefactor * (kernel_e + zeff(species, density, r_index))

    return jax.lax.cond(species_a != eidx, ion_like_branch, electron_branch, operand=None)


def collisionality_ntss_zeff_local(
    species_a: int,
    species: Species,
    v: float,
    density_local,
    temperature_local,
    v_thermal_local,
) -> float:
    """Local legacy NTSSfusion-like simplified electron collisionality using Zeff."""
    eidx = _electron_index(species)
    def ion_like_branch(_):
        return collisionality_local(
            species_a,
            species,
            v,
            density_local,
            temperature_local,
            v_thermal_local,
            COULOMB_LOG_MODEL_NTSS_LEGACY,
        )

    def electron_branch(_):
        ne = STATE_DENSITY_TO_PHYSICAL * density_local[eidx]
        vte = jnp.maximum(v_thermal_local[eidx], 1.0e-30)
        prefactor = gamma_ab_local(
            species,
            eidx,
            eidx,
            temperature_local,
            density_local,
            v_thermal_local,
            COULOMB_LOG_MODEL_NTSS_LEGACY,
        ) * ne / jnp.maximum(v, 1.0e-30) ** 3
        kernel_e = ntss_collision_kernel(species, eidx, eidx, v, vte)
        return prefactor * (kernel_e + zeff_local(species, density_local))

    return jax.lax.cond(species_a != eidx, ion_like_branch, electron_branch, operand=None)


def nuD_ab(species: Species, species_a: int, species_b: int, v: float, r_index: int,
           density, temperature, v_thermal, coulomb_log_model: int = COULOMB_LOG_MODEL_DEFAULT) -> float:
    """Pairwise pitch-angle scattering frequency for species a against species b."""
    nb = STATE_DENSITY_TO_PHYSICAL * density[species_b, r_index]
    vtb = v_thermal[species_b, r_index]
    prefactor = gamma_ab(species, species_a, species_b, v, r_index, temperature, density, v_thermal, coulomb_log_model) * nb / v**3
    erf_part = jax.scipy.special.erf(v / vtb) - chandrasekhar(v / vtb)
    return prefactor * erf_part


def nuD_ab_local(species: Species, species_a: int, species_b: int, v: float,
                 density_local, temperature_local, v_thermal_local, coulomb_log_model: int = COULOMB_LOG_MODEL_DEFAULT) -> float:
    """Pairwise pitch-angle scattering frequency using local profiles at one radius."""
    nb = STATE_DENSITY_TO_PHYSICAL * density_local[species_b]
    vtb = v_thermal_local[species_b]
    prefactor = gamma_ab_local(species, species_a, species_b, temperature_local, density_local, v_thermal_local, coulomb_log_model) * nb / v**3
    erf_part = jax.scipy.special.erf(v / vtb) - chandrasekhar(v / vtb)
    return prefactor * erf_part


def gamma_ab(species: Species, species_a: int, species_b: int, v: float, r_index: int,
             temperature, density, v_thermal, coulomb_log_model: int = COULOMB_LOG_MODEL_DEFAULT) -> float:
    """Prefactor for pairwise collisionality."""
    del v
    lnlambda = coulomb_logarithm(species, species_a, species_b, r_index, temperature, density, v_thermal, coulomb_log_model)
    ea, eb = species.charge[species_a], species.charge[species_b]
    ma = species.mass[species_a]
    return ea**2 * eb**2 * lnlambda / (4 * jnp.pi * epsilon_0**2 * ma**2)


def gamma_ab_local(species: Species, species_a: int, species_b: int,
                   temperature_local, density_local, v_thermal_local, coulomb_log_model: int = COULOMB_LOG_MODEL_DEFAULT) -> float:
    """Prefactor for pairwise collisionality using local profiles at one radius."""
    lnlambda = coulomb_logarithm_local(species, species_a, species_b, temperature_local, density_local, v_thermal_local, coulomb_log_model)
    ea, eb = species.charge[species_a], species.charge[species_b]
    ma = species.mass[species_a]
    return ea**2 * eb**2 * lnlambda / (4 * jnp.pi * epsilon_0**2 * ma**2)


def nupar_ab(species: Species, species_a: int, species_b: int, v: float, r_index: int,
             density, temperature, v_thermal) -> float:
    """Parallel collisionality."""
    nb = STATE_DENSITY_TO_PHYSICAL * density[species_b, r_index]
    vtb = v_thermal[species_b, r_index]
    return (
        2 * gamma_ab(species, species_a, species_b, v, r_index, temperature, density, v_thermal) * nb / v**3
        * chandrasekhar(v / vtb)
    )


def coulomb_logarithm(species: Species, species_a: int, species_b: int, r_index: int,
                      temperature, density, v_thermal, coulomb_log_model: int = COULOMB_LOG_MODEL_DEFAULT) -> float:
    """Coulomb logarithm for collisions between species a and b."""
    return jax.lax.switch(
        coulomb_log_model,
        (
            lambda: _coulomb_logarithm_default(temperature, density, r_index),
            lambda: coulomb_logarithm_ntss_legacy(temperature, density, r_index),
            lambda: coulomb_logarithm_first_principles(species, species_a, species_b, r_index, density, temperature, v_thermal),
        ),
    )


def coulomb_logarithm_local(species: Species, species_a: int, species_b: int,
                            temperature_local, density_local, v_thermal_local, coulomb_log_model: int = COULOMB_LOG_MODEL_DEFAULT) -> float:
    """Coulomb logarithm using local profiles at one radius."""
    return jax.lax.switch(
        coulomb_log_model,
        (
            lambda: _coulomb_logarithm_default_local(temperature_local, density_local),
            lambda: coulomb_logarithm_ntss_legacy_local(temperature_local, density_local),
            lambda: coulomb_logarithm_first_principles_local(species, species_a, species_b, density_local, temperature_local, v_thermal_local),
        ),
    )


def _coulomb_logarithm_default(temperature, density, r_index: int) -> float:
    Te_eV = STATE_TEMPERATURE_TO_EV * temperature[0, r_index]
    ne_m3 = STATE_DENSITY_TO_PHYSICAL * density[0, r_index]
    return 32.2 + 1.15 * jnp.log10(Te_eV**2 / ne_m3)


def _coulomb_logarithm_default_local(temperature_local, density_local) -> float:
    Te_eV = STATE_TEMPERATURE_TO_EV * temperature_local[0]
    ne_m3 = STATE_DENSITY_TO_PHYSICAL * density_local[0]
    return 32.2 + 1.15 * jnp.log10(Te_eV**2 / ne_m3)


def coulomb_logarithm_ntss_legacy(temperature, density, r_index: int) -> float:
    """Legacy NTSS/TC_DKES piecewise Coulomb logarithm."""
    Te_eV = jnp.maximum(STATE_TEMPERATURE_TO_EV * temperature[0, r_index], 1.0e-30)
    ne_cm3 = jnp.maximum(STATE_DENSITY_TO_PHYSICAL * density[0, r_index] / 1.0e6, 1.0e-30)
    low_t = 23.4 - 1.15 * jnp.log10(ne_cm3) + 3.45 * jnp.log10(Te_eV)
    high_t = 25.3 - 1.15 * jnp.log10(ne_cm3) + 2.30 * jnp.log10(Te_eV)
    return jnp.where(Te_eV <= 50.0, low_t, high_t)


def coulomb_logarithm_ntss_legacy_local(temperature_local, density_local) -> float:
    """Local legacy NTSS/TC_DKES piecewise Coulomb logarithm."""
    Te_eV = jnp.maximum(STATE_TEMPERATURE_TO_EV * temperature_local[0], 1.0e-30)
    ne_cm3 = jnp.maximum(STATE_DENSITY_TO_PHYSICAL * density_local[0] / 1.0e6, 1.0e-30)
    low_t = 23.4 - 1.15 * jnp.log10(ne_cm3) + 3.45 * jnp.log10(Te_eV)
    high_t = 25.3 - 1.15 * jnp.log10(ne_cm3) + 2.30 * jnp.log10(Te_eV)
    return jnp.where(Te_eV <= 50.0, low_t, high_t)


def coulomb_logarithm_first_principles(species: Species, species_a: int, species_b: int, r_index: int,
                                       density, temperature, v_thermal) -> float:
    """Coulomb logarithm from impact parameters."""
    bmin, bmax = impact_parameter(species, species_a, species_b, r_index, density, temperature, v_thermal)
    return jnp.log(jnp.maximum(bmax, bmin * (1.0 + 1.0e-30)) / jnp.maximum(bmin, 1.0e-30))


def coulomb_logarithm_first_principles_local(species: Species, species_a: int, species_b: int,
                                             density_local, temperature_local, v_thermal_local) -> float:
    """Local Coulomb logarithm from impact parameters."""
    bmin, bmax = impact_parameter_local(species, species_a, species_b, density_local, temperature_local, v_thermal_local)
    return jnp.log(jnp.maximum(bmax, bmin * (1.0 + 1.0e-30)) / jnp.maximum(bmin, 1.0e-30))


def tau_ab_ntss_like(
    species: Species,
    species_a: int,
    species_b: int,
    r_index: int,
    density,
    temperature,
    v_thermal,
) -> float:
    """NTSSfusion-like test-particle collision time tau(a,b)."""
    lnL = coulomb_logarithm(
        species,
        species_a,
        species_b,
        r_index,
        temperature,
        density,
        v_thermal,
        COULOMB_LOG_MODEL_DEFAULT,
    )
    charge_prod = jnp.maximum(jnp.abs(species.charge_qp[species_a] * species.charge_qp[species_b]), 1.0e-30)
    mass_a = species.mass[species_a]
    temperature_a_eV = STATE_TEMPERATURE_TO_EV * temperature[species_a, r_index]
    density_b_m3 = STATE_DENSITY_TO_PHYSICAL * density[species_b, r_index]
    return (
        3.0
        * jnp.power(epsilon_0 / charge_prod, 2)
        / elementary_charge
        * jnp.sqrt(mass_a / elementary_charge * jnp.power(2.0 * jnp.pi * temperature_a_eV, 3))
        / (elementary_charge * jnp.maximum(density_b_m3, 1.0e-30) * lnL)
    )


def tau_ab_ntss_like_local(
    species: Species,
    species_a: int,
    species_b: int,
    density_local,
    temperature_local,
    v_thermal_local,
) -> float:
    """Local NTSSfusion-like test-particle collision time tau(a,b)."""
    lnL = coulomb_logarithm_local(
        species,
        species_a,
        species_b,
        temperature_local,
        density_local,
        v_thermal_local,
        COULOMB_LOG_MODEL_DEFAULT,
    )
    charge_prod = jnp.maximum(jnp.abs(species.charge_qp[species_a] * species.charge_qp[species_b]), 1.0e-30)
    mass_a = species.mass[species_a]
    temperature_a_eV = STATE_TEMPERATURE_TO_EV * temperature_local[species_a]
    density_b_m3 = STATE_DENSITY_TO_PHYSICAL * density_local[species_b]
    return (
        3.0
        * jnp.power(epsilon_0 / charge_prod, 2)
        / elementary_charge
        * jnp.sqrt(mass_a / elementary_charge * jnp.power(2.0 * jnp.pi * temperature_a_eV, 3))
        / (elementary_charge * jnp.maximum(density_b_m3, 1.0e-30) * lnL)
    )


def ntss_collision_kernel(
    species: Species,
    species_a: int,
    species_b: int,
    v: float,
    v_thermal_b,
) -> float:
    """Velocity kernel used by NTSSfusion for effective collisionality."""
    del species, species_a, species_b
    x = jnp.maximum((v / jnp.maximum(v_thermal_b, 1.0e-30)) ** 2, 1.0e-30)
    sqx = jnp.sqrt(x)
    return jax.scipy.special.erf(sqx) * (1.0 - 0.5 / x) + jnp.exp(-x) / (sqx * jnp.sqrt(jnp.pi))


def ntss_collision_kernel_thermal(
    species: Species,
    species_a: int,
    species_b: int,
    r_index: int,
    temperature,
) -> float:
    """NTSSfusion kernel evaluated at vn=1 (thermal speed of test species)."""
    x = (
        jnp.maximum(temperature[species_a, r_index], 1.0e-30)
        / jnp.maximum(temperature[species_b, r_index], 1.0e-30)
        * species.mass_mp[species_b]
        / jnp.maximum(species.mass_mp[species_a], 1.0e-30)
    )
    x = jnp.maximum(x, 1.0e-30)
    sqx = jnp.sqrt(x)
    return jax.scipy.special.erf(sqx) * (1.0 - 0.5 / x) + jnp.exp(-x) / (sqx * jnp.sqrt(jnp.pi))


def ntss_collision_kernel_thermal_local(
    species: Species,
    species_a: int,
    species_b: int,
    temperature_local,
) -> float:
    """Local NTSSfusion kernel evaluated at vn=1 (thermal speed of test species)."""
    x = (
        jnp.maximum(temperature_local[species_a], 1.0e-30)
        / jnp.maximum(temperature_local[species_b], 1.0e-30)
        * species.mass_mp[species_b]
        / jnp.maximum(species.mass_mp[species_a], 1.0e-30)
    )
    x = jnp.maximum(x, 1.0e-30)
    sqx = jnp.sqrt(x)
    return jax.scipy.special.erf(sqx) * (1.0 - 0.5 / x) + jnp.exp(-x) / (sqx * jnp.sqrt(jnp.pi))


def impact_parameter(species: Species, species_a: int, species_b: int, r_index: int,
                     density, temperature, v_thermal) -> float:
    """Impact parameters for classical Coulomb collision."""
    bmin = jnp.maximum(
        impact_parameter_perp(species, species_a, species_b, r_index, v_thermal),
        debroglie_length(species, species_a, species_b, r_index, v_thermal),
    )
    bmax = debye_length(species, r_index, density, temperature)
    return bmin, bmax


def impact_parameter_local(species: Species, species_a: int, species_b: int,
                           density_local, temperature_local, v_thermal_local) -> float:
    """Local impact parameters for classical Coulomb collision."""
    bmin = jnp.maximum(
        impact_parameter_perp_local(species, species_a, species_b, v_thermal_local),
        debroglie_length_local(species, species_a, species_b, v_thermal_local),
    )
    bmax = debye_length_local(species, density_local, temperature_local)
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
        jnp.abs(species.charge[species_a] * species.charge[species_b])
        / (4 * jnp.pi * epsilon_0 * m_reduced * v_th**2)
    )


def impact_parameter_perp_local(species: Species, species_a: int, species_b: int,
                                v_thermal_local) -> float:
    """Local distance of the closest approach for a 90 degree Coulomb collision."""
    m_reduced = (
        species.mass[species_a]
        * species.mass[species_b]
        / (species.mass[species_a] + species.mass[species_b])
    )
    v_th = jnp.sqrt(v_thermal_local[species_a] * v_thermal_local[species_b])
    return jnp.abs(species.charge[species_a] * species.charge[species_b]) / (
        4 * jnp.pi * epsilon_0 * m_reduced * v_th**2
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


def debroglie_length_local(species: Species, species_a: int, species_b: int,
                           v_thermal_local) -> float:
    """Local thermal DeBroglie wavelength."""
    m_reduced = (
        species.mass[species_a]
        * species.mass[species_b]
        / (species.mass[species_a] + species.mass[species_b])
    )
    v_th = jnp.sqrt(v_thermal_local[species_a] * v_thermal_local[species_b])
    return hbar / (2 * m_reduced * v_th)


def debye_length(species: Species, r_index: int, density, temperature) -> float:
    """Scale length for charge screening."""
    den = jnp.sum(
        (STATE_DENSITY_TO_PHYSICAL * density[:, r_index])
        / (temperature[:, r_index] * JOULE_PER_KEV)
        * species.charge**2
    )
    return jnp.sqrt(epsilon_0 / den)


def debye_length_local(species: Species, density_local, temperature_local) -> float:
    """Local scale length for charge screening."""
    den = jnp.sum(
        (STATE_DENSITY_TO_PHYSICAL * density_local)
        / (temperature_local * JOULE_PER_KEV)
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
