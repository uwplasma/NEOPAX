# --- Physical source model functions (JAX-compatible, stateless) ---
import jax
import jax.numpy as jnp
from jax import jit
from ._constants import proton_mass, elementary_charge


@jit
def fusion_power_fraction_electrons(state, species):
    Te = state.temperature[species.species_idx['e']]
    y2 = 88. / Te
    y = jnp.sqrt(y2)
    part = 2. * (jnp.log((1 - y + y2) / (1 + 2 * y + y2)) / 6 +
                0.57735026 * jnp.atan(0.57735026 * (2 * y - 1)) +
                0.30229987) / y2
    return 1. - part

@jit
def dt_reaction(state, species):
    nD = state.density[species.species_idx['D']]
    nT = state.density[species.species_idx['T']]
    TT = state.temperature[species.species_idx['T']]
    t = jnp.power(TT, -1. / 3.)
    wrk = (TT + 1.0134) / (1 + 6.386e-3 * jnp.square(TT + 1.0134)) + 1.877 * jnp.exp(-0.16176 * jnp.sqrt(TT) * TT)
    DTreactionRate = 8.972e-19 * t * t * jnp.exp(-19.94 * t) * wrk
    HeSource = 1e20 * DTreactionRate * nD * nT
    AlphaPower = 3.52e3 * HeSource
    return DTreactionRate, HeSource, AlphaPower

@jit
def _power_exchange_impl(state, species, pair_active_mask, use_ntssfusion_lnL):
    # JAX-jittable, differentiable: vectorized pairwise sum
    n_species = state.temperature.shape[0]
    idx_i, idx_j = jnp.triu_indices(n_species, k=1)
    nA = state.density[idx_i]
    nB = state.density[idx_j]
    TA = state.temperature[idx_i]
    TB = state.temperature[idx_j]
    # This transport source formula is written for the normalized transport
    # state convention used in NEOPAX:
    #   density in 1e20 m^-3, temperature in keV/eV-style transport units,
    #   masses in proton-mass units, charges in proton-charge units.
    # Using SI masses here blows up the denominator and produces unphysical
    # ~1e15-1e16 source magnitudes.
    mA = species.mass_mp[idx_i]
    mB = species.mass_mp[idx_j]
    qA = species.charge_qp[idx_i]
    qB = species.charge_qp[idx_j]
    qA_col = qA[:, None]
    qB_col = qB[:, None]
    mA_col = mA[:, None]
    mB_col = mB[:, None]
    same_sign = (qA_col * qB_col) > 0.0
    a_is_electron = qA_col < 0.0
    b_is_electron = qB_col < 0.0

    # NTSSfusion-style branch logic from CPlasma_Fluxes.cpp::lnLab.
    lnL_ee = jnp.where(
        TA < 0.01,
        17.24 + 0.5 * jnp.log(TA**3 / nA),
        16.1 + 0.5 * jnp.log(TA**2 / nA),
    )
    lnL_ii = 17.24 - jnp.log(
        jnp.abs(qA_col * qB_col)
        * (mA_col + mB_col)
        / (mA_col * TB + mB_col * TA)
        * jnp.sqrt(nA * qA_col**2 / TA + nB * qB_col**2 / TB)
    )
    cond_ei_1 = (TA < 0.01 * qB_col**2) & (TB * mA_col < TA * mB_col)
    cond_ei_2 = (TA > 0.01 * qB_col**2) & (TB * mA_col < 0.01 * qB_col**2 * mB_col)
    cond_ei_3 = TA * mB_col < TB * jnp.abs(qB_col) * mA_col
    lnL_ei = jnp.where(
        cond_ei_1,
        17.24 + 0.5 * jnp.log(TA**3 / (nA * qB_col**2)),
        jnp.where(
            cond_ei_2,
            16.1 + 0.5 * jnp.log(TA**2 / nA),
            jnp.where(
                cond_ei_3,
                24.24 + 0.5 * jnp.log(TB * (TB * mB_col) ** 2 / (nB * (qB_col**2) ** 2)),
                16.1 + 0.5 * jnp.log(TA**2 / nA),
            ),
        ),
    )
    cond_ie_1 = (TB < 0.01 * qA_col**2) & (TA * mB_col < TB * mA_col)
    cond_ie_2 = (TB > 0.01 * qA_col**2) & (TA * mB_col < 0.01 * qA_col**2 * mA_col)
    cond_ie_3 = TB * mA_col < TA * jnp.abs(qA_col) * mB_col
    lnL_ie = jnp.where(
        cond_ie_1,
        17.24 + 0.5 * jnp.log(TB**3 / (nB * qA_col**2)),
        jnp.where(
            cond_ie_2,
            16.1 + 0.5 * jnp.log(TB**2 / nB),
            jnp.where(
                cond_ie_3,
                24.24 + 0.5 * jnp.log(TA * (TA * mA_col) ** 2 / (nA * (qA_col**2) ** 2)),
                16.1 + 0.5 * jnp.log(TB**2 / nB),
            ),
        ),
    )
    lnL_simple = 32.2 + 1.15 * jnp.log10(TA**2 / nA)
    lnL_ntss = jnp.where(
        same_sign,
        jnp.where(a_is_electron & b_is_electron, lnL_ee, lnL_ii),
        jnp.where(a_is_electron, lnL_ei, lnL_ie),
    )
    lnL = jnp.where(jnp.asarray(use_ntssfusion_lnL), lnL_ntss, lnL_simple)
    mA = mA[:, None]
    mB = mB[:, None]
    qA = qA[:, None]
    qB = qB[:, None]
    pair_prefactor = (
        663.0
        * jnp.sqrt(mA * mB)
        * jnp.square(qA * qB)
    )
    Pab = (
        pair_prefactor
        * nA
        * nB
        * lnL
        * (TB - TA)
        / jnp.power(mA * TB + mB * TA, 1.5)
    )
    pair_mask = jnp.asarray(pair_active_mask, dtype=Pab.dtype)[:, None]
    Pab = Pab * pair_mask
    # For each pair (i, j): +Pab to i, -Pab to j
    out = jnp.zeros((n_species,) + Pab.shape[1:], dtype=Pab.dtype)
    out = out.at[idx_i].add(Pab)
    out = out.at[idx_j].add(-Pab)
    return out


def power_exchange(state, species, pair_active_mask, coulomb_log_mode="neopax_simple"):
    mode = str(coulomb_log_mode).strip().lower()
    if mode in {"neopax_simple", "simple", "default"}:
        use_ntssfusion_lnL = False
    elif mode in {"ntssfusion", "ntss_like", "ntss", "pair_dependent"}:
        use_ntssfusion_lnL = True
    else:
        raise ValueError(
            f"Unknown power_exchange coulomb_log_mode '{coulomb_log_mode}'. "
            "Use 'neopax_simple' or 'ntssfusion'."
        )
    return _power_exchange_impl(state, species, pair_active_mask, jnp.asarray(use_ntssfusion_lnL))


@jit
def _power_exchange_temperature_equilibration_impl(state, species, pair_active_mask):
    # A pairwise temperature-equilibration model in the same general spirit as
    # T3D, but kept in the native NEOPAX conservative pair-exchange form.
    n_species = state.temperature.shape[0]
    idx_i, idx_j = jnp.triu_indices(n_species, k=1)
    nA = state.density[idx_i]
    nB = state.density[idx_j]
    TA = state.temperature[idx_i]
    TB = state.temperature[idx_j]
    mA = species.mass_mp[idx_i]
    mB = species.mass_mp[idx_j]
    qA = species.charge_qp[idx_i]
    qB = species.charge_qp[idx_j]

    nA_cgs = jnp.maximum(nA * 1.0e14, 1.0e-30)
    nB_cgs = jnp.maximum(nB * 1.0e14, 1.0e-30)
    TA_eV = jnp.maximum(TA * 1.0e3, 1.0e-30)
    TB_eV = jnp.maximum(TB * 1.0e3, 1.0e-30)
    qA_abs = jnp.maximum(jnp.abs(qA), 1.0e-30)
    qB_abs = jnp.maximum(jnp.abs(qB), 1.0e-30)

    a_is_electron = qA < 0.0
    b_is_electron = qB < 0.0
    both_electrons = a_is_electron & b_is_electron
    electron_ion = a_is_electron ^ b_is_electron

    lnL_ee = (
        23.5
        - jnp.log(jnp.sqrt(nA_cgs) / jnp.power(TA_eV, 1.25))
        - jnp.sqrt(1.0e-5 + jnp.square(jnp.log(TA_eV) - 2.0) / 16.0)
    )
    lnL_ei = jnp.where(
        a_is_electron,
        24.0 - jnp.log(jnp.sqrt(nA_cgs) / TA_eV),
        24.0 - jnp.log(jnp.sqrt(nB_cgs) / TB_eV),
    )
    lnL_ii = 23.0 - jnp.log(
        (qA_abs * qB_abs)
        * (mA + mB)
        / jnp.maximum(mA * TB_eV + mB * TA_eV, 1.0e-30)
        * jnp.sqrt(
            nA_cgs * qA_abs**2 / TA_eV
            + nB_cgs * qB_abs**2 / TB_eV
        )
    )
    lnL = jnp.where(both_electrons, lnL_ee, jnp.where(electron_ion, lnL_ei, lnL_ii))

    pair_prefactor = (
        (4.0 * 2.85e2 / jnp.sqrt(jnp.pi))
        * jnp.sqrt(mA[:, None] * mB[:, None])
        * jnp.square((qA * qB)[:, None])
    )
    Pab = (
        pair_prefactor
        * nA
        * nB
        * lnL
        * (TB - TA)
        / jnp.power(mA[:, None] * TB + mB[:, None] * TA, 1.5)
    )
    pair_mask = jnp.asarray(pair_active_mask, dtype=Pab.dtype)[:, None]
    Pab = Pab * pair_mask

    out = jnp.zeros((n_species,) + Pab.shape[1:], dtype=Pab.dtype)
    out = out.at[idx_i].add(Pab)
    out = out.at[idx_j].add(-Pab)
    return out


def power_exchange_temperature_equilibration(state, species, pair_active_mask):
    return _power_exchange_temperature_equilibration_impl(state, species, pair_active_mask)


@jit
def _t3d_loglambda_pair(nA, TA, mA, qA, nB, TB, mB, qB):
    nA_cgs = jnp.maximum(nA * 1.0e14, 1.0e-30)
    nB_cgs = jnp.maximum(nB * 1.0e14, 1.0e-30)
    TA_eV = jnp.maximum(TA * 1.0e3, 1.0e-30)
    TB_eV = jnp.maximum(TB * 1.0e3, 1.0e-30)
    qA_abs = jnp.maximum(jnp.abs(qA), 1.0e-30)
    qB_abs = jnp.maximum(jnp.abs(qB), 1.0e-30)

    a_is_electron = qA < 0.0
    b_is_electron = qB < 0.0
    both_electrons = jnp.logical_and(a_is_electron, b_is_electron)
    a_electron_b_ion = jnp.logical_and(a_is_electron, jnp.logical_not(b_is_electron))
    b_electron_a_ion = jnp.logical_and(b_is_electron, jnp.logical_not(a_is_electron))

    lnL_ee = (
        23.5
        - jnp.log(jnp.sqrt(nA_cgs) / jnp.power(TA_eV, 1.25))
        - jnp.sqrt(1.0e-5 + jnp.square(jnp.log(TA_eV) - 2.0) / 16.0)
    )
    lnL_ei = 24.0 - jnp.log(jnp.sqrt(nA_cgs) / TA_eV)
    lnL_ie = 24.0 - jnp.log(jnp.sqrt(nB_cgs) / TB_eV)
    lnL_ii = 23.0 - jnp.log(
        (qA_abs * qB_abs)
        * (mA + mB)
        / jnp.maximum(mA * TB_eV + mB * TA_eV, 1.0e-30)
        * jnp.sqrt(
            nA_cgs * qA_abs**2 / TA_eV
            + nB_cgs * qB_abs**2 / TB_eV
        )
    )
    return jnp.where(
        both_electrons,
        lnL_ee,
        jnp.where(a_electron_b_ion, lnL_ei, jnp.where(b_electron_a_ion, lnL_ie, lnL_ii)),
    )


@jit
def _power_exchange_t3d_exact_impl(state, species, active_matrix, t_ref):
    density = state.density
    temperature = state.temperature
    pressure = state.pressure
    q = species.charge_qp
    m = species.mass_mp
    dtype = pressure.dtype

    n_s = density[:, None, :]
    n_u = density[None, :, :]
    T_s = temperature[:, None, :]
    T_u = temperature[None, :, :]
    p_s = pressure[:, None, :]
    p_u = pressure[None, :, :]
    q_s = q[:, None]
    q_u = q[None, :]
    m_s = m[:, None]
    m_u = m[None, :]

    n_s_cgs = jnp.maximum(n_s * 1.0e14, 1.0e-30)
    n_u_cgs = jnp.maximum(n_u * 1.0e14, 1.0e-30)
    T_s_eV = jnp.maximum(T_s * 1.0e3, 1.0e-30)
    T_u_eV = jnp.maximum(T_u * 1.0e3, 1.0e-30)
    q_s_abs = jnp.maximum(jnp.abs(q_s), 1.0e-30)
    q_u_abs = jnp.maximum(jnp.abs(q_u), 1.0e-30)
    m_s_3d = m_s[:, :, None]
    m_u_3d = m_u[:, :, None]
    q_s_abs_3d = q_s_abs[:, :, None]
    q_u_abs_3d = q_u_abs[:, :, None]

    s_is_electron = q_s < 0.0
    u_is_electron = q_u < 0.0
    both_electrons = jnp.logical_and(s_is_electron, u_is_electron)
    s_electron_u_ion = jnp.logical_and(s_is_electron, jnp.logical_not(u_is_electron))
    u_electron_s_ion = jnp.logical_and(u_is_electron, jnp.logical_not(s_is_electron))

    log_lambda_ee = (
        23.5
        - jnp.log(jnp.sqrt(n_s_cgs) / jnp.power(T_s_eV, 1.25))
        - jnp.sqrt(1.0e-5 + jnp.square(jnp.log(T_s_eV) - 2.0) / 16.0)
    )
    log_lambda_ei = 24.0 - jnp.log(jnp.sqrt(n_s_cgs) / T_s_eV)
    log_lambda_ie = 24.0 - jnp.log(jnp.sqrt(n_u_cgs) / T_u_eV)
    log_lambda_ii = 23.0 - jnp.log(
        (q_s_abs_3d * q_u_abs_3d)
        * (m_s_3d + m_u_3d)
        / jnp.maximum(m_s_3d * T_u_eV + m_u_3d * T_s_eV, 1.0e-30)
        * jnp.sqrt(
            n_s_cgs * q_s_abs_3d**2 / T_s_eV
            + n_u_cgs * q_u_abs_3d**2 / T_u_eV
        )
    )
    log_lambda = jnp.where(
        both_electrons[:, :, None],
        log_lambda_ee,
        jnp.where(
            s_electron_u_ion[:, :, None],
            log_lambda_ei,
            jnp.where(u_electron_s_ion[:, :, None], log_lambda_ie, log_lambda_ii),
        ),
    )

    nu_su = (
        2.85e2
        * jnp.square(q_s * q_u)[:, :, None]
        / jnp.sqrt(jnp.maximum(m_s, 1.0e-30))[:, :, None]
        * log_lambda
        * n_u
        / jnp.power(jnp.maximum(T_s, 1.0e-30), 1.5)
    )
    nu_equil = (
        jnp.asarray(t_ref, dtype=dtype)
        * (8.0 / (3.0 * jnp.sqrt(jnp.pi)))
        * nu_su
        * jnp.sqrt(jnp.maximum(m_u / jnp.maximum(m_s, 1.0e-30), 1.0e-30))[:, :, None]
        * jnp.power(
            jnp.maximum(m_u / jnp.maximum(m_s, 1.0e-30), 1.0e-30)[:, :, None]
            + T_u / jnp.maximum(T_s, 1.0e-30),
            -1.5,
        )
    )
    equil_factor = (
        n_s
        * p_u
        / jnp.maximum(n_u * p_s, 1.0e-30)
        - 1.0
    )
    coll_eq_pairs = p_s * nu_equil * equil_factor
    coll_eq_pairs = coll_eq_pairs * jnp.asarray(active_matrix, dtype=dtype)[:, :, None]
    coll_eq = jnp.sum(coll_eq_pairs, axis=1)
    return 1.5 * coll_eq


def power_exchange_t3d_exact(state, species, active_matrix, t_ref):
    return _power_exchange_t3d_exact_impl(state, species, active_matrix, t_ref)

@jit
def bremsstrahlung_radiation_generalized(
    state,
    species=None,
    use_relativistic_correction=False,
    exclude_impurity_bremsstrahlung=False,
    main_ion_indices=(1, 2),  # default: D, T
    electron_index=0,
    delta_zeff=0.0,
    brems_coefficient=3.16e-1,
):
    """
    Generalized bremsstrahlung radiation model.
    Args:
        state: object with .density, .temperature arrays (shape: [n_species, ...])
        species: optional, with .charge
        use_relativistic_correction: bool, include Stott correction
        exclude_impurity_bremsstrahlung: bool, only main-ion Zeff
        main_ion_indices: tuple of indices for main ions
        electron_index: index for electrons
        delta_zeff: additive Zeff offset, analogous to NTSSfusion DeltaZeff
        brems_coefficient: prefactor in `PBrems = C * Zeff * ne^2 * sqrt(Te)`
    Returns:
        PBrems: bremsstrahlung power density in the normalized transport units
            used elsewhere in NEOPAX, matching the NTSS-like
            `C * Zeff * ne^2 * sqrt(Te)` form.
        Zeff: effective charge
    """
    ne = state.density[electron_index]
    Te = state.temperature[electron_index]
    n_species = state.density.shape[0]
    if species is not None:
        Z = species.charge / elementary_charge
    else:
        # Signed fallback charges so electron can be excluded from Zeff.
        Z = jnp.array([-1.0, 1.0, 1.0, 2.0])[:n_species]
    Z = jnp.asarray(Z, dtype=ne.dtype)
    if Z.ndim == 1:
        Z = Z[:, None]
    n = state.density
    # Zeff should only include ion species. Including the electron row would
    # add an unphysical +1 offset and double the pure D/T Zeff from 1 to 2.
    ion_mask = Z > 0.0
    n_main = n[jnp.array(main_ion_indices)]
    Z_main = Z[jnp.array(main_ion_indices)]
    full_zeff = jnp.sum(jnp.where(ion_mask, n * Z**2, 0.0), axis=0) / ne
    main_ion_zeff = jnp.sum(n_main * Z_main**2, axis=0) / ne
    Zeff = jax.lax.cond(
        jnp.asarray(exclude_impurity_bremsstrahlung),
        lambda _: main_ion_zeff,
        lambda _: full_zeff,
        operand=None,
    )
    Zeff = Zeff + jnp.asarray(delta_zeff, dtype=ne.dtype)
    # Keep the NTSS-like transport normalization used by the original default
    # source, while upgrading Zeff to the full multispecies expression.
    PBrems = jnp.asarray(brems_coefficient, dtype=ne.dtype) * Zeff * ne**2 * jnp.sqrt(Te)
    Tm = 511.0 * 1e3  # eV
    Te_eV = Te  # assume Te in eV
    corr = (1.0 + 2.0 * Te_eV / Tm) * (1.0 + (2.0 / Zeff) * (1.0 - 1.0 / (1.0 + Te_eV / Tm)))
    PBrems = jax.lax.cond(
        jnp.asarray(use_relativistic_correction),
        lambda val: val * corr,
        lambda val: val,
        PBrems,
    )
    return PBrems, Zeff


def bremsstrahlung_radiation(
    state,
    species,
    delta_zeff=0.0,
    coefficient_mode="astra",
    brems_coefficient=None,
):
    mode = str(coefficient_mode).strip().lower()
    if brems_coefficient is not None:
        coefficient = float(brems_coefficient)
    elif mode in {"astra", "default", "neopax"}:
        coefficient = 3.16e-1
    elif mode in {"ntssfusion", "ntss", "wobig", "nrl"}:
        coefficient = 3.34e-1
    else:
        raise ValueError(
            f"Unknown bremsstrahlung coefficient_mode '{coefficient_mode}'. "
            "Use 'astra' or 'ntssfusion', or provide brems_coefficient explicitly."
        )
    return bremsstrahlung_radiation_generalized(
        state,
        species=species,
        exclude_impurity_bremsstrahlung=False,
        delta_zeff=delta_zeff,
        brems_coefficient=coefficient,
    )

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

