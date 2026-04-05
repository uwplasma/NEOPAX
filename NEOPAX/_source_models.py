import jax.numpy as jnp
import jax
from jax import jit
from jax import config
# to use higher precision
config.update("jax_enable_x64", True)
import interpax
from ._constants import proton_mass,elementary_charge
from ._species import coulomb_logarithm

#These should go to the physics_models.py 
#Get FusionPower Fraction to Electrons, using same model as NTSS - update in the future

# Refactored: JAX-compatible, explicit arguments, SourceModelBase subclass
import jax.numpy as jnp
from ._sources import SourceModelBase
from ._state import TransportState

SPECIES_IDX = {
    "e": 0,
    "D": 1,
    "T": 2,
    "He": 3,
}



@jit
def get_plasma_permitivity(state, species_mass, field, grid_x):
    """Return epsilon(r) used in Er diffusion/ambipolar source term."""
    psi_fac = 1.0 + 1.0 / (field.enlogation * jnp.square(field.iota))
    psi_fac = psi_fac.at[0].set(1.0)
    mass_density = jnp.sum(species_mass[:, None] * state.density, axis=0)
    epsilon_r = mass_density * psi_fac / jnp.square(field.B0)
    plasma_permitivity = interpax.Interpolator1D(field.r_grid, epsilon_r, extrap=True)
    return plasma_permitivity(grid_x)



class FusionPowerFractionElectronsSource(SourceModelBase):
    def __call__(self, state: TransportState):
        Te = state.temperature[SPECIES_IDX['e']]
        y2 = 88. / Te
        y = jnp.sqrt(y2)
        part = 2. * (jnp.log((1 - y + y2) / (1 + 2 * y + y2)) / 6 +
                    0.57735026 * jnp.atan(0.57735026 * (2 * y - 1)) +
                    0.30229987) / y2
        return 1. - part


# Refactored: D-T Reaction Source

class DTReactionSource(SourceModelBase):
    def __call__(self, state: TransportState):
        nD = state.density[SPECIES_IDX['D']]
        nT = state.density[SPECIES_IDX['T']]
        TT = state.temperature[SPECIES_IDX['T']]
        t = jnp.power(TT, -1. / 3.)
        wrk = (TT + 1.0134) / (1 + 6.386e-3 * jnp.square(TT + 1.0134)) + 1.877 * jnp.exp(-0.16176 * jnp.sqrt(TT) * TT)
        DTreactionRate = 8.972e-19 * t * t * jnp.exp(-19.94 * t) * wrk
        HeSource = 1e20 * DTreactionRate * nD * nT
        AlphaPower = 3.52e3 * HeSource
        return DTreactionRate, HeSource, AlphaPower


# Refactored: Power Exchange Source


# Stateless/config-driven: all parameters from state/species
class PowerExchangeSource(SourceModelBase):
    def __init__(self, idx_a=None, idx_b=None):
        # Optionally allow config for which species to use, else default to D and T
        self.idx_a = idx_a if idx_a is not None else SPECIES_IDX['D']
        self.idx_b = idx_b if idx_b is not None else SPECIES_IDX['T']
    def __call__(self, state: TransportState, species=None):
        # If species object is provided, use it for mass/charge, else assume standard mapping
        idx_a = self.idx_a
        idx_b = self.idx_b
        nA = state.density[idx_a]
        nB = state.density[idx_b]
        TA = state.temperature[idx_a]
        TB = state.temperature[idx_b]
        # Default values for D and T if species not provided
        if species is not None:
            mA = species.mass[idx_a]
            mB = species.mass[idx_b]
            qA = species.charge[idx_a]
            qB = species.charge[idx_b]
        else:
            # Defaults: D and T
            mA = 2.014 * proton_mass
            mB = 3.016 * proton_mass
            qA = elementary_charge
            qB = elementary_charge
        # Coulomb logarithm: can be a function of state, or use a default
        lnL = 32.2 + 1.15 * jnp.log10(TA**2 / nA)
        Pab = 663. * jnp.sqrt(mA * mB) * jnp.square(qA * qB / (elementary_charge * elementary_charge)) \
            * nA * nB * lnL * (TB - TA) / jnp.power(mA * TB + mB * TA, 1.5)
        return Pab



# Refactored: Bremsstrahlung Radiation Source


# Stateless/config-driven: all parameters from state/species
class BremsstrahlungRadiationSource(SourceModelBase):
    def __init__(self, ZD=None, ZT=None):
        # Optionally allow config for ZD/ZT, else use standard values for D/T
        self.ZD = ZD if ZD is not None else 1.0
        self.ZT = ZT if ZT is not None else 1.0
    def __call__(self, state: TransportState, species=None):
        Te = state.temperature[SPECIES_IDX['e']]
        ne = state.density[SPECIES_IDX['e']]
        nD = state.density[SPECIES_IDX['D']]
        nT = state.density[SPECIES_IDX['T']]
        # If species object is provided, use its Z for D/T
        ZD = self.ZD
        ZT = self.ZT
        if species is not None:
            ZD = species.charge[SPECIES_IDX['D']] / elementary_charge
            ZT = species.charge[SPECIES_IDX['T']] / elementary_charge
        Zeff = (ZD ** 2 * nD + ZT ** 2 * nT) / ne
        PBrems = 3.16e-1 * Zeff * ne * ne * jnp.sqrt(Te)
        return PBrems, Zeff

