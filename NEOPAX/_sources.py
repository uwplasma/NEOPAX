# --- Physical source model functions (JAX-compatible, stateless) ---
import jax.numpy as jnp
from jax import jit
from ._constants import proton_mass, elementary_charge

SPECIES_IDX = {
    "e": 0,
    "D": 1,
    "T": 2,
    "He": 3,
}

@jit
def fusion_power_fraction_electrons(state):
    Te = state.temperature[SPECIES_IDX['e']]
    y2 = 88. / Te
    y = jnp.sqrt(y2)
    part = 2. * (jnp.log((1 - y + y2) / (1 + 2 * y + y2)) / 6 +
                0.57735026 * jnp.atan(0.57735026 * (2 * y - 1)) +
                0.30229987) / y2
    return 1. - part

@jit
def dt_reaction(state):
    nD = state.density[SPECIES_IDX['D']]
    nT = state.density[SPECIES_IDX['T']]
    TT = state.temperature[SPECIES_IDX['T']]
    t = jnp.power(TT, -1. / 3.)
    wrk = (TT + 1.0134) / (1 + 6.386e-3 * jnp.square(TT + 1.0134)) + 1.877 * jnp.exp(-0.16176 * jnp.sqrt(TT) * TT)
    DTreactionRate = 8.972e-19 * t * t * jnp.exp(-19.94 * t) * wrk
    HeSource = 1e20 * DTreactionRate * nD * nT
    AlphaPower = 3.52e3 * HeSource
    return DTreactionRate, HeSource, AlphaPower

@jit
def power_exchange(state, species=None):
    # JAX-jittable, differentiable: vectorized pairwise sum
    n_species = state.temperature.shape[0]
    idx_i, idx_j = jnp.triu_indices(n_species, k=1)
    nA = state.density[idx_i]
    nB = state.density[idx_j]
    TA = state.temperature[idx_i]
    TB = state.temperature[idx_j]
    if species is not None:
        mA = species.mass[idx_i]
        mB = species.mass[idx_j]
        qA = species.charge[idx_i]
        qB = species.charge[idx_j]
    else:
        def get_mass(idx):
            return jnp.where(idx == SPECIES_IDX.get('D', -1), 2.014 * proton_mass,
                    jnp.where(idx == SPECIES_IDX.get('T', -1), 3.016 * proton_mass, proton_mass))
        mA = get_mass(idx_i)
        mB = get_mass(idx_j)
        qA = elementary_charge * jnp.ones_like(idx_i, dtype=state.density.dtype)
        qB = elementary_charge * jnp.ones_like(idx_j, dtype=state.density.dtype)
    lnL = 32.2 + 1.15 * jnp.log10(TA**2 / nA)
    Pab = 663. * jnp.sqrt(mA * mB) * jnp.square(qA * qB / (elementary_charge * elementary_charge)) \
        * nA * nB * lnL * (TB - TA) / jnp.power(mA * TB + mB * TA, 1.5)
    # For each pair (i, j): +Pab to i, -Pab to j
    out = jnp.zeros((n_species,) + Pab.shape[1:], dtype=Pab.dtype)
    out = out.at[idx_i].add(Pab)
    out = out.at[idx_j].add(-Pab)
    return out

@jit
def bremsstrahlung_radiation(state, ZD=None, ZT=None, species=None):
    ZD = ZD if ZD is not None else 1.0
    ZT = ZT if ZT is not None else 1.0
    Te = state.temperature[SPECIES_IDX['e']]
    ne = state.density[SPECIES_IDX['e']]
    nD = state.density[SPECIES_IDX['D']]
    nT = state.density[SPECIES_IDX['T']]
    if species is not None:
        ZD = species.charge[SPECIES_IDX['D']] / elementary_charge
        ZT = species.charge[SPECIES_IDX['T']] / elementary_charge
    Zeff = (ZD ** 2 * nD + ZT ** 2 * nT) / ne
    PBrems = 3.16e-1 * Zeff * ne * ne * jnp.sqrt(Te)
    return PBrems, Zeff


@jit
def bremsstrahlung_radiation_generalized(
    state,
    species=None,
    use_relativistic_correction=False,
    exclude_impurity_bremsstrahlung=False,
    main_ion_indices=(1, 2),  # default: D, T
    electron_index=0,
):
    """
    Generalized bremsstrahlung radiation model (torax-style).
    Args:
        state: object with .density, .temperature arrays (shape: [n_species, ...])
        species: optional, with .charge
        use_relativistic_correction: bool, include Stott correction
        exclude_impurity_bremsstrahlung: bool, only main-ion Zeff
        main_ion_indices: tuple of indices for main ions
        electron_index: index for electrons
    Returns:
        PBrems: bremsstrahlung power density [W/m^3]
        Zeff: effective charge
    """
    ne = state.density[electron_index]
    Te = state.temperature[electron_index]
    n_species = state.density.shape[0]
    if species is not None:
        Z = species.charge / elementary_charge
    else:
        # fallback: e=1, D=1, T=1, He=2
        Z = jnp.array([1.0, 1.0, 1.0, 2.0])[:n_species]
    n = state.density
    # Zeff: sum_s n_s Z_s^2 / n_e
    if exclude_impurity_bremsstrahlung:
        # Only main ions
        n_main = n[jnp.array(main_ion_indices)]
        Z_main = Z[jnp.array(main_ion_indices)]
        Zeff = jnp.sum(n_main * Z_main**2, axis=0) / ne
    else:
        Zeff = jnp.sum(n * Z**2, axis=0) / ne
    # Wesson formula (see torax): 5.35e-37 * Zeff * n_e^2 * sqrt(T_e) [W/m^3]
    PBrems = 5.35e-37 * Zeff * ne**2 * jnp.sqrt(Te)
    if use_relativistic_correction:
        Tm = 511.0 * 1e3  # eV
        Te_eV = Te  # assume Te in eV
        corr = (1.0 + 2.0 * Te_eV / Tm) * (1.0 + (2.0 / Zeff) * (1.0 - 1.0 / (1.0 + Te_eV / Tm)))
        PBrems = PBrems * corr
    return PBrems, Zeff

@jit
def ecrh_source(
    state,
    radius,
    total_power,
    center,
    width,
    electron_index=0,
    profile_shape='gaussian',
):
    """
    Electron Cyclotron Resonance Heating (ECRH) source model.
    Args:
        state: object with .density, .temperature arrays (shape: [n_species, ...])
        radius: 1D array of normalized or physical radius (same shape as profile)
        total_power: total ECRH power to deposit [W]
        center: center of deposition (in same units as radius)
        width: width (stddev) of Gaussian (in same units as radius)
        electron_index: index for electrons (default 0)
        profile_shape: 'gaussian' (default) or 'flat'
    Returns:
        source: array, shape (n_species, ...) with ECRH power density [W/m^3] for each species
    """
    n_species = state.temperature.shape[0]
    shape = radius.shape
    if profile_shape == 'gaussian':
        profile = jnp.exp(-0.5 * ((radius - center) / width) ** 2)
    elif profile_shape == 'flat':
        profile = jnp.ones_like(radius)
    else:
        raise ValueError('Unknown profile_shape: ' + str(profile_shape))
    # Normalize profile to integrate to 1
    norm = jnp.sum(profile)
    profile = profile / norm
    # Power density profile
    power_density = total_power * profile  # [W] * [1] = [W]
    # Assign to electrons only
    out = jnp.zeros((n_species,) + shape, dtype=state.temperature.dtype)
    out = out.at[electron_index].set(power_density)
    return out


@jit
def icrh_source(
    state,
    radius,
    total_power,
    center,
    width,
    ion_indices=(1, 2),  # default: D, T
    profile_shape='gaussian',
):
    """
    Ion Cyclotron Resonance Heating (ICRH) source model.
    Args:
        state: object with .density, .temperature arrays (shape: [n_species, ...])
        radius: 1D array of normalized or physical radius (same shape as profile)
        total_power: total ICRH power to deposit [W]
        center: center of deposition (in same units as radius)
        width: width (stddev) of Gaussian (in same units as radius)
        ion_indices: tuple of indices for ions (default: D, T)
        profile_shape: 'gaussian' (default) or 'flat'
    Returns:
        source: array, shape (n_species, ...) with ICRH power density [W/m^3] for each species
    """
    n_species = state.temperature.shape[0]
    shape = radius.shape
    if profile_shape == 'gaussian':
        profile = jnp.exp(-0.5 * ((radius - center) / width) ** 2)
    elif profile_shape == 'flat':
        profile = jnp.ones_like(radius)
    else:
        raise ValueError('Unknown profile_shape: ' + str(profile_shape))
    norm = jnp.sum(profile)
    profile = profile / norm
    power_density = total_power * profile
    out = jnp.zeros((n_species,) + shape, dtype=state.temperature.dtype)
    for idx in ion_indices:
        out = out.at[idx].add(power_density / len(ion_indices))
    return out

@jit
def pellet_injection_source(
    state,
    radius,
    total_particles,
    center,
    width,
    species_index=1,  # default: D
    profile_shape='gaussian',
):
    """
    Pellet injection source model (density source).
    Args:
        state: object with .density, .temperature arrays (shape: [n_species, ...])
        radius: 1D array of normalized or physical radius (same shape as profile)
        total_particles: total number of particles injected [1/s]
        center: center of deposition (in same units as radius)
        width: width (stddev) of Gaussian (in same units as radius)
        species_index: index for species to receive the source (default: D)
        profile_shape: 'gaussian' (default) or 'flat'
    Returns:
        source: array, shape (n_species, ...) with particle source [1/(m^3 s)] for each species
    """
    n_species = state.density.shape[0]
    shape = radius.shape
    if profile_shape == 'gaussian':
        profile = jnp.exp(-0.5 * ((radius - center) / width) ** 2)
    elif profile_shape == 'flat':
        profile = jnp.ones_like(radius)
    else:
        raise ValueError('Unknown profile_shape: ' + str(profile_shape))
    norm = jnp.sum(profile)
    profile = profile / norm
    particle_density = total_particles * profile
    out = jnp.zeros((n_species,) + shape, dtype=state.density.dtype)
    out = out.at[species_index].set(particle_density)
    return out

