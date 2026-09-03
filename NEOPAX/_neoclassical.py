import h5py as h5
import jax.numpy as jnp
import jax
from jax import config
# to use higher precision
config.update("jax_enable_x64", True)
from jax import jit
import jax.numpy as jnp
import lineax
import interpax
from typing import Any, Callable
from ._constants import elementary_charge, epsilon_0
from ._species import (
    COULOMB_LOG_MODEL_DEFAULT,
    COULOMB_LOG_MODEL_FIRST_PRINCIPLES,
    COULOMB_LOG_MODEL_NTSS_LEGACY,
    collisionality,
    collisionality_local,
    collisionality_ntss_like,
    collisionality_ntss_like_local,
    collisionality_ntss_zeff,
    collisionality_ntss_zeff_local,
)
from ._monoenergetic import (
    MONOENERGETIC_KIND_GENERIC,
    MONOENERGETIC_KIND_PREPROCESSED_3D,
    MONOENERGETIC_KIND_PREPROCESSED_3D_NTSS1D_FIXED,
    MONOENERGETIC_KIND_PREPROCESSED_3D_RADIAL,
    MONOENERGETIC_KIND_PREPROCESSED_3D_RADIAL_NTSS1D,
    MONOENERGETIC_KIND_PREPROCESSED_NTSS,
    monoenergetic_database_kind,
)
from ._monoenergetic_interpolators import monoenergetic_interpolation_kernel
from ._species import get_Thermodynamical_Forces_A1, get_Thermodynamical_Forces_A2, get_Thermodynamical_Forces_A3
from ._cell_variable import get_gradient_density, get_gradient_temperature
from ._state import JOULE_PER_KEV, get_v_thermal

DENSITY_STATE_TO_PHYSICAL = 1.0e20
TEMPERATURE_STATE_TO_PHYSICAL = 1.0e3

COLLISIONALITY_MODEL_DEFAULT = 0
COLLISIONALITY_MODEL_NTSS_LIKE = 1
COLLISIONALITY_MODEL_NTSS_LEGACY = 2
COLLISIONALITY_MODEL_FIRST_PRINCIPLES = 3
COLLISIONALITY_MODEL_NTSS_ZEFF = 4



@jit
def get_plasma_permitivity(state, species_mass, geometry, grid_x):
    """Return epsilon(r) used in Er diffusion/ambipolar source term."""
    psi_den = geometry.enlogation * jnp.square(geometry.iota)
    psi_den_active = jnp.abs(psi_den) > 0.0
    psi_den_safe = jnp.where(psi_den_active, psi_den, 1.0)
    psi_fac = 1.0 + jnp.where(psi_den_active, 1.0 / psi_den_safe, 0.0)
    psi_fac = psi_fac.at[0].set(1.0)
    mass_density = DENSITY_STATE_TO_PHYSICAL * jnp.sum(species_mass[:, None] * state.density, axis=0)
    epsilon_r = mass_density * psi_fac / jnp.square(geometry.B0)
    plasma_permitivity = interpax.Interpolator1D(geometry.r_grid, epsilon_r, extrap=True)
    return plasma_permitivity(grid_x)


def _as_species_constraint(arr, n_species):
    if arr is None:
        return None
    out = jnp.asarray(arr)
    if out.ndim == 0:
        return jnp.repeat(out[None], n_species, axis=0)
    if out.shape[0] < n_species:
        out = jnp.pad(out, (0, n_species - out.shape[0]), mode="edge")
    return out[:n_species]


def _collisionality_kind(collisionality_model: str | None) -> int:
    key = str(collisionality_model or "default").strip().lower()
    if key in {"ntss_like", "ntss", "ntssfusion"}:
        return COLLISIONALITY_MODEL_NTSS_LIKE
    if key in {"ntss_legacy", "ntssfusion_legacy", "legacy_ntss", "legacy"}:
        return COLLISIONALITY_MODEL_NTSS_LEGACY
    if key in {"first_principles", "first_principles_log", "impact_parameter", "impact_parameters"}:
        return COLLISIONALITY_MODEL_FIRST_PRINCIPLES
    if key in {"ntss_zeff", "zeff", "ntss_simplified", "ntssfusion_zeff"}:
        return COLLISIONALITY_MODEL_NTSS_ZEFF
    return COLLISIONALITY_MODEL_DEFAULT


@jit
def _nu_over_vnew(
    species,
    index_species,
    v_new_a,
    r_index,
    density,
    temperature,
    v_thermal,
    collisionality_kind,
):
    use_ntss_like = collisionality_kind == COLLISIONALITY_MODEL_NTSS_LIKE
    use_ntss_zeff = collisionality_kind == COLLISIONALITY_MODEL_NTSS_ZEFF
    coulomb_log_model = jax.lax.switch(
        collisionality_kind,
        (
            lambda: COULOMB_LOG_MODEL_DEFAULT,
            lambda: COULOMB_LOG_MODEL_DEFAULT,
            lambda: COULOMB_LOG_MODEL_NTSS_LEGACY,
            lambda: COULOMB_LOG_MODEL_FIRST_PRINCIPLES,
            lambda: COULOMB_LOG_MODEL_NTSS_LEGACY,
        ),
    )
    return jax.lax.cond(
        use_ntss_like,
        lambda _: jnp.full_like(
            v_new_a,
            collisionality_ntss_like(index_species, species, v_new_a, r_index, density, temperature, v_thermal)
            / jnp.maximum(v_thermal[index_species, r_index], 1.0e-30),
        ),
        lambda _: jax.lax.cond(
            use_ntss_zeff,
            lambda __: collisionality_ntss_zeff(
                index_species,
                species,
                v_new_a,
                r_index,
                density,
                temperature,
                v_thermal,
            ) / v_new_a,
            lambda __: collisionality(
                index_species,
                species,
                v_new_a,
                r_index,
                density,
                temperature,
                v_thermal,
                coulomb_log_model,
            ) / v_new_a,
            operand=None,
        ),
        operand=None,
    )


@jit
def _nu_over_vnew_local(
    species,
    index_species,
    v_new_a,
    density_local,
    temperature_local,
    v_thermal_local,
    collisionality_kind,
):
    use_ntss_like = collisionality_kind == COLLISIONALITY_MODEL_NTSS_LIKE
    use_ntss_zeff = collisionality_kind == COLLISIONALITY_MODEL_NTSS_ZEFF
    coulomb_log_model = jax.lax.switch(
        collisionality_kind,
        (
            lambda: COULOMB_LOG_MODEL_DEFAULT,
            lambda: COULOMB_LOG_MODEL_DEFAULT,
            lambda: COULOMB_LOG_MODEL_NTSS_LEGACY,
            lambda: COULOMB_LOG_MODEL_FIRST_PRINCIPLES,
            lambda: COULOMB_LOG_MODEL_NTSS_LEGACY,
        ),
    )
    return jax.lax.cond(
        use_ntss_like,
        lambda _: jnp.full_like(
            v_new_a,
            collisionality_ntss_like_local(index_species, species, v_new_a, density_local, temperature_local, v_thermal_local)
            / jnp.maximum(v_thermal_local[index_species], 1.0e-30),
        ),
        lambda _: jax.lax.cond(
            use_ntss_zeff,
            lambda __: collisionality_ntss_zeff_local(
                index_species,
                species,
                v_new_a,
                density_local,
                temperature_local,
                v_thermal_local,
            ) / v_new_a,
            lambda __: collisionality_local(
                index_species,
                species,
                v_new_a,
                density_local,
                temperature_local,
                v_thermal_local,
                coulomb_log_model,
            ) / v_new_a,
            operand=None,
        ),
        operand=None,
    )


def _assemble_lij_matrix(species, energy_grid, index_species, vth_a, nu_vnew_a, Dij):
    Lij = jnp.zeros((3, 3))
    L11_fac_a = -1.0 / jnp.sqrt(jnp.pi) * (species.mass[index_species] / species.charge[index_species]) ** 2 * vth_a**3
    L13_fac_a = -1.0 / jnp.sqrt(jnp.pi) * (species.mass[index_species] / species.charge[index_species]) * vth_a**2
    L33_fac_a = -1.0 / jnp.sqrt(jnp.pi) * vth_a
    D11_a = -(10**Dij.at[:, 0].get())
    D13_a = -Dij.at[:, 1].get()
    D33_a = -jnp.true_divide(Dij.at[:, 2].get(), nu_vnew_a)
    Lij = Lij.at[0, 0].set(L11_fac_a * jnp.sum(energy_grid.L11_weight * energy_grid.xWeights * D11_a))
    Lij = Lij.at[0, 1].set(L11_fac_a * jnp.sum(energy_grid.L12_weight * energy_grid.xWeights * D11_a))
    Lij = Lij.at[1, 0].set(Lij.at[0, 1].get())
    Lij = Lij.at[1, 1].set(L11_fac_a * jnp.sum(energy_grid.L22_weight * energy_grid.xWeights * D11_a))
    Lij = Lij.at[0, 2].set(L13_fac_a * jnp.sum(energy_grid.L13_weight * energy_grid.xWeights * D13_a))
    Lij = Lij.at[1, 2].set(L13_fac_a * jnp.sum(energy_grid.L23_weight * energy_grid.xWeights * D13_a))
    Lij = Lij.at[2, 0].set(-Lij.at[0, 2].get())
    Lij = Lij.at[2, 1].set(-Lij.at[1, 2].get())
    Lij = Lij.at[2, 2].set(L33_fac_a * jnp.sum(energy_grid.L33_weight * energy_grid.xWeights * D33_a))
    return Lij


def _interpolated_lij_matrix(
    species,
    energy_grid,
    database,
    index_species,
    radius_value,
    Er_value,
    vth_a,
    nu_vnew_a,
):
    v_new_a = energy_grid.v_norm * vth_a
    Er_vnew_a = Er_value * 1.0e3 / v_new_a
    kernel = monoenergetic_interpolation_kernel(database)
    Dij = jax.vmap(kernel, in_axes=(None, 0, 0, None))(radius_value, nu_vnew_a, Er_vnew_a, database)
    return _assemble_lij_matrix(species, energy_grid, index_species, vth_a, nu_vnew_a, Dij)

@jit
def get_Lij_matrix(species, energy_grid, geometry, database, index_species, r_index, Er, temperature, density, v_thermal, collisionality_kind=COLLISIONALITY_MODEL_DEFAULT):
    vth_a = v_thermal[index_species, r_index]
    v_new_a = energy_grid.v_norm * vth_a
    nu_vnew_a = _nu_over_vnew(species, index_species, v_new_a, r_index, density, temperature, v_thermal, collisionality_kind)
    return _interpolated_lij_matrix(
        species,
        energy_grid,
        database,
        index_species,
        geometry.r_grid[r_index],
        Er[r_index],
        vth_a,
        nu_vnew_a,
    )


@jit
def get_Lij_matrix_local(species, energy_grid, geometry, database, index_species, r_index, Er_value, temperature, density, v_thermal, collisionality_kind=COLLISIONALITY_MODEL_DEFAULT):
    vth_a = v_thermal[index_species, r_index]
    v_new_a = energy_grid.v_norm * vth_a
    nu_vnew_a = _nu_over_vnew(species, index_species, v_new_a, r_index, density, temperature, v_thermal, collisionality_kind)
    return _interpolated_lij_matrix(
        species,
        energy_grid,
        database,
        index_species,
        geometry.r_grid[r_index],
        Er_value,
        vth_a,
        nu_vnew_a,
    )


@jit
def get_Lij_matrix_at_radius(species, energy_grid, geometry, database, index_species, radius_value, Er_value, temperature_local, density_local, v_thermal_local, collisionality_kind=COLLISIONALITY_MODEL_DEFAULT):
    vth_a = v_thermal_local[index_species]
    v_new_a = energy_grid.v_norm * vth_a
    nu_vnew_a = _nu_over_vnew_local(species, index_species, v_new_a, density_local, temperature_local, v_thermal_local, collisionality_kind)
    return _interpolated_lij_matrix(
        species,
        energy_grid,
        database,
        index_species,
        radius_value,
        Er_value,
        vth_a,
        nu_vnew_a,
    )


@jit
def _get_Neoclassical_Fluxes_generic(
    species,
    energy_grid,
    geometry,
    database,
    Er,
    temperature,
    density,
    density_right_constraint=None,
    density_right_grad_constraint=None,
    temperature_right_constraint=None,
    temperature_right_grad_constraint=None,
    collisionality_kind=COLLISIONALITY_MODEL_DEFAULT,
):
    v_thermal = get_v_thermal(species.mass, temperature)
    r_grid = geometry.r_grid
    r_grid_half = geometry.r_grid_half
    dr = geometry.dr

    n_species = int(temperature.shape[0])
    n_right = _as_species_constraint(density_right_constraint, n_species)
    if n_right is None:
        n_right = density[:, -1]
    n_right_grad = _as_species_constraint(density_right_grad_constraint, n_species)
    if n_right_grad is None:
        n_right_grad = jnp.zeros_like(n_right)
    t_right = _as_species_constraint(temperature_right_constraint, n_species)
    if t_right is None:
        t_right = temperature[:, -1]
    t_right_grad = _as_species_constraint(temperature_right_grad_constraint, n_species)
    if t_right_grad is None:
        t_right_grad = jnp.zeros_like(t_right)

    def get_Neoclassical_Fluxes_internal(a, Lij, temperature, density, Er, n_rc, n_rg, t_rc, t_rg):
        dndr = get_gradient_density(
            density,
            r_grid,
            r_grid_half,
            dr,
            right_face_constraint=n_rc,
            right_face_grad_constraint=n_rg,
        )
        dTdr = get_gradient_temperature(
            temperature,
            r_grid,
            r_grid_half,
            dr,
            right_face_constraint=t_rc,
            right_face_grad_constraint=t_rg,
        )
        A1 = get_Thermodynamical_Forces_A1(species.charge[a], density, temperature, dndr, dTdr, Er)
        A2 = get_Thermodynamical_Forces_A2(temperature, dTdr)
        A3 = get_Thermodynamical_Forces_A3(Er)
        density_phys = DENSITY_STATE_TO_PHYSICAL * density
        temperature_phys = TEMPERATURE_STATE_TO_PHYSICAL * temperature
        Gamma = -density_phys * (Lij[:, 0, 0] * A1 + Lij[:, 0, 1] * A2 + Lij[:, 0, 2] * A3)
        Q = -temperature_phys * density_phys * (Lij[:, 1, 0] * A1 + Lij[:, 1, 1] * A2 + Lij[:, 1, 2] * A3)
        Upar = -density_phys * (Lij[:, 2, 0] * A1 + Lij[:, 2, 1] * A2 + Lij[:, 2, 2] * A3)
        return Gamma, Q, Upar

    Lij = jax.vmap(
        lambda a: jax.vmap(
            lambda r: get_Lij_matrix(species, energy_grid, geometry, database, a, r, Er, temperature, density, v_thermal, collisionality_kind),
            in_axes=(0),
        )(geometry.full_grid_indices),
        in_axes=(0),
    )(species.species_indices)
    results = jax.vmap(get_Neoclassical_Fluxes_internal, in_axes=(0, 0, 0, 0, None, 0, 0, 0, 0))(
        species.species_indices,
        Lij,
        temperature,
        density,
        Er,
        n_right,
        n_right_grad,
        t_right,
        t_right_grad,
    )
    Gamma, Q, Upar = results
    return Lij, Gamma, Q, Upar


@jit
def _get_Neoclassical_Fluxes_Faces_generic(
    species,
    energy_grid,
    geometry,
    database,
    Er_faces,
    temperature_faces,
    density_faces,
    dndr_faces,
    dTdr_faces,
    collisionality_kind=COLLISIONALITY_MODEL_DEFAULT,
):
    v_thermal_faces = get_v_thermal(species.mass, temperature_faces)
    radius_values = geometry.r_grid_half

    Lij_faces = jax.vmap(
        lambda a: jax.vmap(
            lambda radius_value, er_value, temperature_local, density_local, vthermal_local: get_Lij_matrix_at_radius(
                species,
                energy_grid,
                geometry,
                database,
                a,
                radius_value,
                er_value,
                temperature_local,
                density_local,
                vthermal_local,
                collisionality_kind,
            ),
            in_axes=(0, 0, 1, 1, 1),
        )(radius_values, Er_faces, temperature_faces, density_faces, v_thermal_faces),
        in_axes=(0,),
    )(species.species_indices)

    A1 = jax.vmap(
        lambda charge, density_a, temperature_a, dndr_a, dTdr_a: get_Thermodynamical_Forces_A1(
            charge, density_a, temperature_a, dndr_a, dTdr_a, Er_faces
        ),
        in_axes=(0, 0, 0, 0, 0),
    )(species.charge, density_faces, temperature_faces, dndr_faces, dTdr_faces)
    A2 = jax.vmap(get_Thermodynamical_Forces_A2, in_axes=(0, 0))(temperature_faces, dTdr_faces)
    A3 = get_Thermodynamical_Forces_A3(Er_faces)

    density_phys = DENSITY_STATE_TO_PHYSICAL * density_faces
    temperature_phys = TEMPERATURE_STATE_TO_PHYSICAL * temperature_faces

    Gamma = -density_phys * (
        Lij_faces[:, :, 0, 0] * A1
        + Lij_faces[:, :, 0, 1] * A2
        + Lij_faces[:, :, 0, 2] * A3[None, :]
    )
    Q = -temperature_phys * density_phys * (
        Lij_faces[:, :, 1, 0] * A1
        + Lij_faces[:, :, 1, 1] * A2
        + Lij_faces[:, :, 1, 2] * A3[None, :]
    )
    Upar = -density_phys * (
        Lij_faces[:, :, 2, 0] * A1
        + Lij_faces[:, :, 2, 1] * A2
        + Lij_faces[:, :, 2, 2] * A3[None, :]
    )
    return Lij_faces, Gamma, Q, Upar

_CENTER_FLUX_DISPATCH: dict[str, Callable[..., Any]] = {
    MONOENERGETIC_KIND_PREPROCESSED_3D_NTSS1D_FIXED: _get_Neoclassical_Fluxes_generic,
    MONOENERGETIC_KIND_PREPROCESSED_3D_RADIAL_NTSS1D: _get_Neoclassical_Fluxes_generic,
    MONOENERGETIC_KIND_PREPROCESSED_NTSS: _get_Neoclassical_Fluxes_generic,
    MONOENERGETIC_KIND_PREPROCESSED_3D_RADIAL: _get_Neoclassical_Fluxes_generic,
    MONOENERGETIC_KIND_PREPROCESSED_3D: _get_Neoclassical_Fluxes_generic,
    MONOENERGETIC_KIND_GENERIC: _get_Neoclassical_Fluxes_generic,
}


_FACE_FLUX_DISPATCH: dict[str, Callable[..., Any]] = {
    MONOENERGETIC_KIND_PREPROCESSED_3D_NTSS1D_FIXED: _get_Neoclassical_Fluxes_Faces_generic,
    MONOENERGETIC_KIND_PREPROCESSED_3D_RADIAL_NTSS1D: _get_Neoclassical_Fluxes_Faces_generic,
    MONOENERGETIC_KIND_PREPROCESSED_NTSS: _get_Neoclassical_Fluxes_Faces_generic,
    MONOENERGETIC_KIND_PREPROCESSED_3D_RADIAL: _get_Neoclassical_Fluxes_Faces_generic,
    MONOENERGETIC_KIND_PREPROCESSED_3D: _get_Neoclassical_Fluxes_Faces_generic,
    MONOENERGETIC_KIND_GENERIC: _get_Neoclassical_Fluxes_Faces_generic,
}


def get_Neoclassical_Fluxes(
    species,
    energy_grid,
    geometry,
    database,
    Er,
    temperature,
    density,
    density_right_constraint=None,
    density_right_grad_constraint=None,
    temperature_right_constraint=None,
    temperature_right_grad_constraint=None,
    collisionality_model="default",
):
    collisionality_kind = _collisionality_kind(collisionality_model)
    flux_fn = _CENTER_FLUX_DISPATCH[monoenergetic_database_kind(database)]
    return flux_fn(
        species,
        energy_grid,
        geometry,
        database,
        Er,
        temperature,
        density,
        density_right_constraint=density_right_constraint,
        density_right_grad_constraint=density_right_grad_constraint,
        temperature_right_constraint=temperature_right_constraint,
        temperature_right_grad_constraint=temperature_right_grad_constraint,
        collisionality_kind=collisionality_kind,
    )


def get_Neoclassical_Fluxes_Faces(
    species,
    energy_grid,
    geometry,
    database,
    Er_faces,
    temperature_faces,
    density_faces,
    dndr_faces,
    dTdr_faces,
    collisionality_model="default",
):
    collisionality_kind = _collisionality_kind(collisionality_model)
    flux_fn = _FACE_FLUX_DISPATCH[monoenergetic_database_kind(database)]
    return flux_fn(
        species,
        energy_grid,
        geometry,
        database,
        Er_faces,
        temperature_faces,
        density_faces,
        dndr_faces,
        dTdr_faces,
        collisionality_kind=collisionality_kind,
    )


#####FOR MOMEMTUM CORRECTION
@jit
def get_Lij_matrix_with_momentum_correction(species, energy_grid, geometry, database, index_species, r_index, Er, temperature, density, v_thermal):
    #For no momentum correction, Lij is just a 3 x 3 matrix for each species at each radial position
    Lij=jnp.zeros((5,5))
    Eij=jnp.zeros((5,5))
    nu_weighted_average=jnp.zeros(3)
    #Thermal velocities
    vth_a = v_thermal[index_species, r_index]
    #velocities for convolution
    v_new_a = energy_grid.v_norm * vth_a
    #Er's for convolution
    Er_vnew_a = Er[r_index] * 1.e+3 / v_new_a
    n = density[index_species, r_index]
    T = temperature[index_species, r_index]
    #same species collisionalities
    ##nu_vnew_T=collisionality(species_a_loc, v_new_a, *global_species_loc)/v_new_a 
    nu_a = collisionality(
        index_species,
        species,
        v_new_a,
        r_index,
        density,
        temperature,
        v_thermal,
    )
    nu_vnew_a=nu_a/v_new_a
    #L11_fac_T=nT*2./jnp.sqrt(jnp.pi)*(tritium_loc.species.mass/tritium_loc.species.charge)**2*vth_T**3
    #L11_fac_a=n/jnp.sqrt(jnp.pi)*(species.mass[index_species]/species.charge[index_species])**2*vth_a**3
    #L13_fac_a=n/jnp.sqrt(jnp.pi)*(species.mass[index_species]/species.charge[index_species])*vth_a**2#*B00(r_grid[r_index])
    #L33_fac_a=n/jnp.sqrt(jnp.pi)*vth_a#*B00(r_grid[r_index])
    L11_fac_a=-1./jnp.sqrt(jnp.pi)*(species.mass[index_species]/species.charge[index_species])**2*vth_a**3
    L13_fac_a=-1./jnp.sqrt(jnp.pi)*(species.mass[index_species]/species.charge[index_species])*vth_a**2#*B00(r_grid[r_index])
    L33_fac_a=-1./jnp.sqrt(jnp.pi)*vth_a#*B00(r_grid[r_index])    
    #Interpolate D11's, D13's and D33's 
    kernel = monoenergetic_interpolation_kernel(database)
    Dij = jax.vmap(kernel, in_axes=(None, 0, 0, None))(geometry.r_grid[r_index], nu_vnew_a, Er_vnew_a, database)
    D11_a=-10**Dij.at[:,0].get()###+4.*nu_vnew_a/3.
    D13_a=-Dij.at[:,1].get()
    D33_a=-jnp.true_divide(Dij.at[:,2].get(),nu_vnew_a)
    Lij=Lij.at[0,0].set(L11_fac_a*jnp.sum(energy_grid.L11_weight*energy_grid.xWeights*(D11_a)))
    Lij=Lij.at[0,1].set(L11_fac_a*jnp.sum(energy_grid.L12_weight*energy_grid.xWeights*(D11_a)))
    Lij=Lij.at[1,0].set(Lij.at[0,1].get())
    Lij=Lij.at[1,1].set(L11_fac_a*jnp.sum(energy_grid.L22_weight*energy_grid.xWeights*(D11_a)))
    Lij=Lij.at[0,2].set(L13_fac_a*jnp.sum(energy_grid.L13_weight*energy_grid.xWeights*(D13_a)))
    Lij=Lij.at[1,2].set(L13_fac_a*jnp.sum(energy_grid.L23_weight*energy_grid.xWeights*(D13_a)))
    Lij=Lij.at[2,0].set(-Lij.at[0,2].get())
    Lij=Lij.at[2,1].set(-Lij.at[1,2].get())
    Lij=Lij.at[2,2].set(L33_fac_a*jnp.sum(energy_grid.L33_weight*energy_grid.xWeights*(D33_a)))
    #Entries of the Lij matrix related with momentum correction
    Lij=Lij.at[0,3].set(Lij.at[1,2].get())
    Lij=Lij.at[1,3].set(L13_fac_a*jnp.sum(energy_grid.L24_weight*energy_grid.xWeights*(D13_a)))       
    Lij=Lij.at[0,4].set(Lij.at[1,3].get()) 
    Lij=Lij.at[1,4].set(L13_fac_a*jnp.sum(energy_grid.L25_weight*energy_grid.xWeights*(D13_a))) 
    Lij=Lij.at[3,0].set(-Lij.at[0,3].get())
    Lij=Lij.at[4,0].set(-Lij.at[0,4].get())    
    Lij=Lij.at[3,1].set(-Lij.at[1,3].get())    
    Lij=Lij.at[4,1].set(-Lij.at[1,4].get())    
    Lij=Lij.at[3,2].set(L33_fac_a*jnp.sum(energy_grid.L43_weight*energy_grid.xWeights*(D33_a)))
    Lij=Lij.at[2,3].set(Lij.at[3,2].get())    
    Lij=Lij.at[3,3].set(L33_fac_a*jnp.sum(energy_grid.L44_weight*energy_grid.xWeights*(D33_a)))
    Lij=Lij.at[2,4].set(Lij.at[3,3].get())    
    Lij=Lij.at[4,2].set(Lij.at[3,3].get())        
    Lij=Lij.at[3,4].set(L33_fac_a*jnp.sum(energy_grid.L45_weight*energy_grid.xWeights*(D33_a))) 
    Lij=Lij.at[4,3].set(Lij.at[3,4].get())          
    Lij=Lij.at[4,4].set(L33_fac_a*jnp.sum(energy_grid.L55_weight*energy_grid.xWeights*(D33_a))) 
    #collisionality weighted velocity integrals matrix Eij 
    Eij=Eij.at[0,2].set(L13_fac_a*jnp.sum(energy_grid.L13_weight*nu_a*energy_grid.xWeights*(D13_a)))
    Eij=Eij.at[1,2].set(L13_fac_a*jnp.sum(energy_grid.L23_weight*nu_a*energy_grid.xWeights*(D13_a)))
    Eij=Eij.at[2,0].set(-Eij.at[0,2].get())
    Eij=Eij.at[2,1].set(-Eij.at[1,2].get())   
    Eij=Eij.at[2,2].set(L33_fac_a*jnp.sum(energy_grid.L33_weight*nu_a*energy_grid.xWeights*(D33_a)))    
    ##
    Eij=Eij.at[0,3].set(Eij.at[1,2].get())
    Eij=Eij.at[1,3].set(L13_fac_a*jnp.sum(energy_grid.L24_weight*nu_a*energy_grid.xWeights*(D13_a)))  
    Eij=Eij.at[0,4].set(Eij.at[1,3].get())  
    Eij=Eij.at[1,4].set(L13_fac_a*jnp.sum(energy_grid.L25_weight*nu_a*energy_grid.xWeights*(D13_a))) 
    Eij=Eij.at[3,0].set(-Eij.at[0,3].get())
    Eij=Eij.at[4,0].set(-Eij.at[0,4].get())    
    Eij=Eij.at[3,1].set(-Eij.at[1,3].get())    
    Eij=Eij.at[4,1].set(-Eij.at[1,4].get())    
    Eij=Eij.at[3,2].set(L33_fac_a*jnp.sum(energy_grid.L43_weight*nu_a*energy_grid.xWeights*(D33_a)))
    Eij=Eij.at[2,3].set(Eij.at[3,2].get())    
    Eij=Eij.at[3,3].set(L33_fac_a*jnp.sum(energy_grid.L44_weight*nu_a*energy_grid.xWeights*(D33_a)))
    Eij=Eij.at[2,4].set(Eij.at[3,3].get())    
    Eij=Eij.at[4,2].set(Eij.at[3,3].get())        
    Eij=Eij.at[3,4].set(L33_fac_a*jnp.sum(energy_grid.L45_weight*nu_a*energy_grid.xWeights*(D33_a))) 
    Eij=Eij.at[4,3].set(Eij.at[3,4].get())          
    Eij=Eij.at[4,4].set(L33_fac_a*jnp.sum(energy_grid.L55_weight*nu_a*energy_grid.xWeights*(D33_a)))     
    #Velocity average of collisionalities
    nu_weighted_average=nu_weighted_average.at[0].set(jnp.sum(nu_a*energy_grid.L13_weight*energy_grid.xWeights))
    nu_weighted_average=nu_weighted_average.at[1].set(jnp.sum(nu_a*energy_grid.L23_weight*energy_grid.xWeights))
    nu_weighted_average=nu_weighted_average.at[2].set(jnp.sum(nu_a*energy_grid.L24_weight*energy_grid.xWeights))    
    return Lij,Eij,nu_weighted_average


#Calculate Cm and Cn, which represent the collision operator terms for each pair of species~
#This uses Hinton/Braginskii,... maybe in the future would be worth to try and see if this can be upgraded to a better model
@jit
def get_Collision_Operator_terms(species, a, b, r_index, temperature, density, v_thermal):
    CM_ab = jnp.zeros((3, 3))
    CN_ab = jnp.zeros((3, 3))
    T_e = temperature[0, r_index]
    n_e = density[0, r_index]
    mass_a = species.mass[a]
    temperature_a = temperature[a, r_index]
    density_b = density[b, r_index]
    v_ratio_squared_ba = jnp.power(v_thermal[b, r_index], 2) / jnp.power(v_thermal[a, r_index], 2)
    mass_ratio_ab = mass_a / species.mass[b]
    charge_ratio_ab = species.charge[a] / species.charge[b]
    term = jnp.sqrt(1.0 + v_ratio_squared_ba)
    CM_ab = CM_ab.at[0, 0].set(-(1.0 + mass_ratio_ab) / jnp.power(term, 3))
    CM_ab = CM_ab.at[0, 1].set(1.5 * CM_ab.at[0, 0].get() / (1.0 + v_ratio_squared_ba))
    CM_ab = CM_ab.at[0, 2].set(1.25 * CM_ab.at[0, 1].get() / (1.0 + v_ratio_squared_ba))
    CM_ab = CM_ab.at[1, 0].set(CM_ab.at[0, 1].get())
    CM_ab = CM_ab.at[1, 1].set(-(3.25 + 4.0 * v_ratio_squared_ba + 7.5 * jnp.power(v_ratio_squared_ba, 2)) / jnp.power(term, 5))
    CM_ab = CM_ab.at[1, 2].set(-(4.3125 + 6.0 * v_ratio_squared_ba + 15.75 * jnp.power(v_ratio_squared_ba, 2)) / jnp.power(term, 7))
    CM_ab = CM_ab.at[2, 0].set(CM_ab.at[0, 2].get())
    CM_ab = CM_ab.at[2, 1].set(CM_ab.at[1, 2].get())
    CM_ab = CM_ab.at[2, 2].set(-(6.765625 + 17.0 * v_ratio_squared_ba + 57.375 * jnp.power(v_ratio_squared_ba, 2) + 28.0 * jnp.power(v_ratio_squared_ba, 3) + 21.875 * jnp.power(v_ratio_squared_ba, 4)) / jnp.power(term, 9))
    CN_ab = CN_ab.at[0, 0].set(-CM_ab.at[0, 0].get())
    CN_ab = CN_ab.at[0, 1].set(-v_ratio_squared_ba * CM_ab.at[0, 1].get())
    CN_ab = CN_ab.at[0, 2].set(-v_ratio_squared_ba * v_ratio_squared_ba * CM_ab.at[0, 2].get())
    CN_ab = CN_ab.at[1, 0].set(-CM_ab.at[1, 0].get())
    CN_ab = CN_ab.at[1, 1].set(6.75 * mass_ratio_ab / jnp.power(term, 5))
    CN_ab = CN_ab.at[1, 2].set(14.0625 * mass_ratio_ab * v_ratio_squared_ba / jnp.power(term, 7))
    CN_ab = CN_ab.at[2, 0].set(-CM_ab.at[2, 0].get())
    CN_ab = CN_ab.at[2, 1].set(CN_ab.at[1, 2].get() * v_ratio_squared_ba / mass_ratio_ab)
    CN_ab = CN_ab.at[2, 2].set(2625.0 / 64.0 * mass_ratio_ab * v_ratio_squared_ba / jnp.power(term, 9))
    lnlambda = 32.2 + 1.15 * jnp.log10(jnp.power(T_e, 2) / n_e)
    tau_ab = 3.0 * jnp.power(epsilon_0 / charge_ratio_ab, 2) / elementary_charge * jnp.sqrt(mass_a / elementary_charge * jnp.power(2.0 * jnp.pi * temperature_a, 3)) / (elementary_charge * density_b * lnlambda)
    return CM_ab, CN_ab, tau_ab


@jit 
def get_rhs(a, r_index, Lij, A1, A2, A3):
    rhs = jnp.zeros(3)
    rhs = rhs.at[0].set(-Lij[2, 0] * A1[a, r_index] - Lij[2, 1] * A2[a, r_index] - Lij[2, 2] * A3[r_index])
    rhs = rhs.at[1].set(
        -(2.5 * Lij[2, 0] - Lij[3, 0]) * A1[a, r_index]
        - (2.5 * Lij[2, 1] - Lij[3, 1]) * A2[a, r_index]
        - (2.5 * Lij[2, 2] - Lij[3, 2]) * A3[r_index]
    )
    rhs = rhs.at[2].set(
        -(4.375 * Lij[2, 0] - 3.5 * Lij[3, 0] + 0.5 * Lij[4, 0]) * A1[a, r_index]
        - (4.375 * Lij[2, 1] - 3.5 * Lij[3, 1] + 0.5 * Lij[4, 1]) * A2[a, r_index]
        - (4.375 * Lij[2, 2] - 3.5 * Lij[3, 2] + 0.5 * Lij[4, 2]) * A3[r_index]
    )
    return rhs




@jit
#Auxiliar ffunction to get sum matrix for one species a
def get_sum(a,j,k,CM,CN,tau):
    sum_kn=jnp.sum(jnp.true_divide(CM.at[a,:,j,k].get(),tau.at[a,:].get()))+CN.at[a,a,j,k].get()/tau.at[a,a].get()
    return sum_kn

@jit
#Auxiliar ffunction to get sum matrix for one species a
def get_sum_total(a,CM,CN,tau):
    sum_kn=jnp.zeros((3,3))
    sum_kn=sum_kn.at[0,0].set(jnp.sum(jnp.true_divide(CM.at[a,:,0,0].get(),tau.at[a,:].get()))+CN.at[a,a,0,0].get()/tau.at[a,a].get())
    sum_kn=sum_kn.at[0,1].set(jnp.sum(jnp.true_divide(CM.at[a,:,0,1].get(),tau.at[a,:].get()))+CN.at[a,a,0,1].get()/tau.at[a,a].get())
    sum_kn=sum_kn.at[0,2].set(jnp.sum(jnp.true_divide(CM.at[a,:,0,2].get(),tau.at[a,:].get()))+CN.at[a,a,0,2].get()/tau.at[a,a].get())
    sum_kn=sum_kn.at[1,0].set(jnp.sum(jnp.true_divide(CM.at[a,:,1,0].get(),tau.at[a,:].get()))+CN.at[a,a,1,0].get()/tau.at[a,a].get())
    sum_kn=sum_kn.at[1,1].set(jnp.sum(jnp.true_divide(CM.at[a,:,1,1].get(),tau.at[a,:].get()))+CN.at[a,a,1,1].get()/tau.at[a,a].get())
    sum_kn=sum_kn.at[1,2].set(jnp.sum(jnp.true_divide(CM.at[a,:,1,2].get(),tau.at[a,:].get()))+CN.at[a,a,1,2].get()/tau.at[a,a].get())
    sum_kn=sum_kn.at[2,0].set(jnp.sum(jnp.true_divide(CM.at[a,:,2,0].get(),tau.at[a,:].get()))+CN.at[a,a,2,0].get()/tau.at[a,a].get())     
    sum_kn=sum_kn.at[2,1].set(jnp.sum(jnp.true_divide(CM.at[a,:,2,1].get(),tau.at[a,:].get()))+CN.at[a,a,2,1].get()/tau.at[a,a].get())     
    sum_kn=sum_kn.at[2,2].set(jnp.sum(jnp.true_divide(CM.at[a,:,2,2].get(),tau.at[a,:].get()))+CN.at[a,a,2,2].get()/tau.at[a,a].get())     
    return sum_kn


@jit
#auxilir matrix to construct matrix for species a
def get_A_matrix(grid, a, b, coeff, nucoeff, CN, sum, tau, v_thermal, field, r_index):
    I=jnp.identity(3)
    A=jnp.zeros((3,3))
    factor = 2. / jnp.power(v_thermal[a, r_index], 2) / field.Bsqav[r_index]
    # ``I`` lives in Sonine-moment space.  It must not be used to decide
    # whether two *species* are equal: doing so aliases species index 3 to
    # the final Sonine index and corrupts the four-species (wHe) system.
    same_species = jnp.equal(a, b).astype(coeff.dtype)
    A = (I - factor * jnp.multiply(jnp.matmul(jnp.transpose(coeff), sum) + nucoeff, grid.Sonine_expansion)) * same_species \
        - factor * jnp.multiply(jnp.matmul(jnp.transpose(coeff), CN.at[a, b, :, :].get() / tau.at[a, b].get()), grid.Sonine_expansion) * (1. - same_species)
    #A=A.at[0,0].set(I.at[a,b].get()*(1.-factor*(jnp.sum(coeff.at[:,0].get()*sum.at[:,0].get())+nucoeff.at[0,0].get())*Sonine_expansion.at[0].get())-factor*jnp.sum(coeff.at[:,0].get()*CN.at[a,b,:,0].get())/tau.at[a,b].get()*Sonine_expansion.at[0].get()*(1.-I.at[a,b].get()))
    #A=A.at[0,1].set(I.at[a,b].get()*(-factor*(jnp.sum(coeff.at[:,0].get()*sum.at[:,1].get())+nucoeff.at[1,0].get())*Sonine_expansion.at[1].get())-factor*jnp.sum(coeff.at[:,0].get()*CN.at[a,b,:,1].get())/tau.at[a,b].get()*Sonine_expansion.at[1].get()*(1.-I.at[a,b].get()))
    #A=A.at[0,2].set(I.at[a,b].get()*(-factor*(jnp.sum(coeff.at[:,0].get()*sum.at[:,2].get())+nucoeff.at[2,0].get())*Sonine_expansion.at[2].get())-factor*jnp.sum(coeff.at[:,0].get()*CN.at[a,b,:,2].get())/tau.at[a,b].get()*Sonine_expansion.at[2].get()*(1.-I.at[a,b].get()))
    #A=A.at[1,0].set(I.at[a,b].get()*(-factor*(jnp.sum(coeff.at[:,1].get()*sum.at[:,0].get())+nucoeff.at[0,1].get())*Sonine_expansion.at[0].get())-factor*jnp.sum(coeff.at[:,1].get()*CN.at[a,b,:,0].get())/tau.at[a,b].get()*Sonine_expansion.at[0].get()*(1.-I.at[a,b].get()))
    #A=A.at[1,1].set(I.at[a,b].get()*(1.-factor*(jnp.sum(coeff.at[:,1].get()*sum.at[:,1].get())+nucoeff.at[1,1].get())*Sonine_expansion.at[1].get())-factor*jnp.sum(coeff.at[:,1].get()*CN.at[a,b,:,1].get())/tau.at[a,b].get()*Sonine_expansion.at[1].get()*(1.-I.at[a,b].get()))
    #A=A.at[1,2].set(I.at[a,b].get()*(-factor*(jnp.sum(coeff.at[:,1].get()*sum.at[:,2].get())+nucoeff.at[2,1].get())*Sonine_expansion.at[2].get())-factor*jnp.sum(coeff.at[:,1].get()*CN.at[a,b,:,2].get())/tau.at[a,b].get()*Sonine_expansion.at[2].get()*(1.-I.at[a,b].get()))
    #A=A.at[2,0].set(I.at[a,b].get()*(-factor*(jnp.sum(coeff.at[:,2].get()*sum.at[:,0].get())+nucoeff.at[0,2].get())*Sonine_expansion.at[0].get())-factor*jnp.sum(coeff.at[:,2].get()*CN.at[a,b,:,0].get())/tau.at[a,b].get()*Sonine_expansion.at[0].get()*(1.-I.at[a,b].get()))
    #A=A.at[2,1].set(I.at[a,b].get()*(-factor*(jnp.sum(coeff.at[:,2].get()*sum.at[:,1].get())+nucoeff.at[1,2].get())*Sonine_expansion.at[1].get())-factor*jnp.sum(coeff.at[:,2].get()*CN.at[a,b,:,1].get())/tau.at[a,b].get()*Sonine_expansion.at[1].get()*(1.-I.at[a,b].get()))
    #A=A.at[2,2].set(I.at[a,b].get()*(1.-factor*(jnp.sum(coeff.at[:,2].get()*sum.at[:,2].get())+nucoeff.at[2,2].get())*Sonine_expansion.at[2].get())-factor*jnp.sum(coeff.at[:,2].get()*CN.at[a,b,:,2].get())/tau.at[a,b].get()*Sonine_expansion.at[2].get()*(1.-I.at[a,b].get()))
    return A

@jit
#auxilir matrix to construct matrix for species a
def get_correction_matrix(grid, a, b, coeff, nucoeff, CM, CN, sum, tau, factor, correction, r_index,
                         dndr, dTdr, temperature, density, charge):
    # The Sonine identity has no role in the species-diagonal predicate.
    # Keep that predicate explicit, as in NTSSfusion's ``ia == ib`` branch.
    same_species = jnp.equal(a, b).astype(coeff.dtype)
    A = -factor * jnp.matmul(jnp.matmul(jnp.transpose(coeff), sum) + nucoeff,
                            jnp.multiply(grid.Sonine_expansion, correction.at[b].get())) * same_species \
        - factor * jnp.matmul(jnp.matmul(jnp.transpose(coeff), CN.at[a, b, :, :].get() / tau.at[a, b].get()),
                            jnp.multiply(grid.Sonine_expansion, correction.at[b].get())) * (1. - same_species)
    add1 = (1. - same_species) * (
        dndr[a, r_index] / density[a, r_index] + dTdr[a, r_index] / temperature[a, r_index]
        - charge[a] / charge[b] * temperature[b, r_index] / temperature[a, r_index]
        * (dndr[b, r_index] / density[b, r_index] + dTdr[b, r_index] / temperature[b, r_index])
    ) * CM.at[a, b, 0, 0].get() / tau.at[a, b].get()
    add2 = (1. - same_species) * (
        dndr[a, r_index] / density[a, r_index] + dTdr[a, r_index] / temperature[a, r_index]
        - charge[a] / charge[b] * temperature[b, r_index] / temperature[a, r_index]
        * (dndr[b, r_index] / density[b, r_index] + dTdr[b, r_index] / temperature[b, r_index])
    ) * (2.5 * CM.at[a, b, 0, 0].get() - CM.at[a, b, 1, 0].get()) / tau.at[a, b].get()
    add3 = (1. - same_species) * charge[a] / charge[b] * temperature[b, r_index] / temperature[a, r_index] \
        * (dTdr[b, r_index] / temperature[b, r_index]) * CN.at[a, b, 0, 1].get() / tau.at[a, b].get()
    add4 = (1. - same_species) * charge[a] / charge[b] * temperature[b, r_index] / temperature[a, r_index] \
        * (dTdr[b, r_index] / temperature[b, r_index]) * (2.5 * CN.at[a, b, 0, 1].get() - CN.at[a, b, 1, 1].get()) / tau.at[a, b].get()
    return A, add1, add2, add3, add4


@jit 
def get_Matrix(grid, field, a, r_index, Lij, Eij, CM_ab, CN_ab, tau, v_thermal):
    coeff=jnp.zeros((3,3))
    nucoeff=jnp.zeros((3,3))
    #Get coeff matrices for species a
    coeff=coeff.at[0,0].set(Lij.at[2,2].get())
    coeff=coeff.at[0,1].set(2.5*Lij.at[2,2].get()-Lij.at[3,2].get())
    coeff=coeff.at[0,2].set(4.375*Lij.at[2,2].get()-3.5*Lij.at[3,2].get()+0.5*Lij.at[3,3].get())
    coeff=coeff.at[1,0].set(Lij.at[2,2].get()-0.4*Lij.at[3,2].get())
    coeff=coeff.at[1,1].set(2.5*Lij.at[2,2].get()-2.0*Lij.at[3,2].get()+0.4*Lij.at[3,3].get())
    coeff=coeff.at[1,2].set(4.375*Lij.at[2,2].get()-5.25*Lij.at[3,2].get()+1.9*Lij.at[3,3].get()-0.2*Lij.at[3,4].get())
    coeff=coeff.at[2,0].set(Lij.at[2,2].get()-0.8*Lij.at[3,2].get()+4.0*Lij.at[3,3].get()/35.0)
    coeff=coeff.at[2,1].set(2.5*Lij.at[2,2].get()-3.0*Lij.at[3,2].get()+38.0*Lij.at[3,3].get()/35.0-4.0*Lij.at[3,4].get()/35.0)
    coeff=coeff.at[2,2].set(4.375*Lij.at[2,2].get()-7.0*Lij.at[3,2].get()+3.8*Lij.at[3,3].get()-0.8*Lij.at[3,4].get()+2.0*Lij.at[4,4].get()/35.0)
    nucoeff=nucoeff.at[0,0].set(Eij.at[2,2].get())
    nucoeff=nucoeff.at[0,1].set(2.5*Eij.at[2,2].get()-Eij.at[3,2].get())
    nucoeff=nucoeff.at[0,2].set(4.375*Eij.at[2,2].get()-3.5*Eij.at[3,2].get()+0.5*Eij.at[3,3].get())
    nucoeff=nucoeff.at[1,0].set(nucoeff.at[0,1].get())
    nucoeff=nucoeff.at[1,1].set(6.25*Eij.at[2,2].get()-5.0*Eij.at[3,2].get()+Eij.at[3,3].get())
    nucoeff=nucoeff.at[1,2].set(10.9375*Eij.at[2,2].get()-13.125*Eij.at[3,2].get()+4.75*Eij.at[3,3].get()-0.5*Eij.at[3,4].get())
    nucoeff=nucoeff.at[2,0].set(nucoeff.at[0,2].get())
    nucoeff=nucoeff.at[2,1].set(nucoeff.at[1,2].get())
    nucoeff=nucoeff.at[2,2].set(19.140625*Eij.at[2,2].get()-30.625*Eij.at[3,2].get()+16.625*Eij.at[3,3].get()-3.5*Eij.at[3,4].get()+0.25*Eij.at[4,4].get())
    #Get sum matrix to be used for constructing A in A*x=rhs system, its norder+1 x norder+1 matrix, thus 3x3 in this case
    sum = jax.vmap(jax.vmap(get_sum, in_axes=(None, None, 0, None, None, None)), in_axes=(None, 0, None, None, None, None))(a, grid.sonine_indeces, grid.sonine_indeces, CM_ab, CN_ab, tau)
    #Get a 3x3 for each species
    M = jax.vmap(get_A_matrix, in_axes=(None, None, 0, None, None, None, None, None, None, None, None))(
        grid, a, jnp.arange(v_thermal.shape[0]), coeff, nucoeff, CN_ab, sum, tau, v_thermal, field, r_index)
    # ``M`` is indexed as (collision species, Sonine row, Sonine column).
    # Keep the Sonine row as the output row and concatenate the collision
    # species/Sonine-column block into the output column.  The previous target
    # shape used ``M.shape[0]`` for the row count, which happened to be correct
    # only for three kinetic species; four-species (wHe) runs then produced a
    # 16x12 operator for a length-12 RHS.
    M = jnp.reshape(
        jnp.transpose(M, (1, 0, 2)),
        (M.shape[1], M.shape[0] * M.shape[2]),
    )
    return M


@jit
def _ntss_radial_flux_correction_terms(
    field,
    a,
    r_index,
    sum,
    add1,
    add2,
    add3,
    add4,
    nu_av,
    A1,
    A2,
    dTdr,
    temperature,
    mass,
    charge,
):
    """Return NTSSfusion's added particle/heat flux velocities.

    NTSSfusion stores temperature in eV and charge as ``Z``.  NEOPAX stores
    temperature in keV and charge in Coulomb, so the common prefactor is
    written directly in SI here.  The two returned quantities are ``Gamma/n``
    and ``Q/(n T)`` respectively; their outer state-unit factors belong to
    :func:`get_corrected_fluxes`.
    """

    temperature_joule = temperature[a, r_index] * JOULE_PER_KEV
    prefactor = (
        mass[a]
        * temperature_joule
        * field.G_PS[r_index]
        / jnp.square(charge[a] * field.B0[r_index])
    )
    dln_temperature = dTdr[a, r_index] / temperature[a, r_index]
    particle = prefactor * (
        add1
        - dln_temperature * sum[0, 1]
        - add3
        + A1[a, r_index] * nu_av[0] / 1.5
        + A2[a, r_index] * nu_av[1] / 1.5
    )
    heat = prefactor * (
        add2
        - dln_temperature * (2.5 * sum[0, 1] - sum[1, 1])
        - add4
        + A1[a, r_index] * nu_av[1] / 1.5
        + A2[a, r_index] * nu_av[2] / 1.5
    )
    return particle, heat


@jit
def get_corrected_fluxes(grid, field, a, r_index, Lij, Eij, nu_av, CM_ab, CN_ab, tau, correction,
                         v_thermal, density, temperature, A1, A2, A3, mass, charge, dndr, dTdr):
    coeff=jnp.zeros((3,3))
    nucoeff=jnp.zeros((3,3))
    #Get coeff matrices for the correction
    coeff=coeff.at[0,0].set(Lij.at[2,0].get())
    coeff=coeff.at[0,1].set(Lij.at[3,0].get())
    coeff=coeff.at[1,0].set(Lij.at[2,0].get()-0.4*Lij.at[2,1].get())
    coeff=coeff.at[1,1].set(Lij.at[3,0].get()-0.4*Lij.at[3,1].get())
    coeff=coeff.at[2,0].set(Lij.at[2,0].get()-0.8*Lij.at[2,1].get()+4.0/35.0*Lij.at[3,1].get())
    coeff=coeff.at[2,1].set(Lij.at[3,0].get()-0.8*Lij.at[3,1].get()+4.0/35.0*Lij.at[4,1].get())
    nucoeff=nucoeff.at[0,0].set(Eij.at[2,0].get())
    nucoeff=nucoeff.at[0,1].set(Eij.at[3,0].get())
    nucoeff=nucoeff.at[1,0].set(2.5*Eij.at[2,0].get()-Eij.at[2,1].get())
    nucoeff=nucoeff.at[1,1].set(2.5*Eij.at[3,0].get()-Eij.at[3,1].get())
    nucoeff=nucoeff.at[2,0].set(4.375*Eij.at[2,0].get()-3.5*Eij.at[2,1].get()+0.5*Eij.at[3,1].get())
    nucoeff=nucoeff.at[2,1].set(4.375*Eij.at[3,0].get()-3.5*Eij.at[3,1].get()+0.5*Eij.at[4,1].get())
    #Get sum matrix yet again
    sum=jax.vmap(jax.vmap(get_sum,in_axes=(None,None,0,None,None,None)),in_axes=(None,0,None,None,None,None))(a,grid.sonine_indeces,grid.sonine_indeces,CM_ab,CN_ab,tau)
    #Get a 3x3 for each species
    factor = 2. / jnp.power(v_thermal[a, r_index], 2) / field.Bsqav[r_index]  # Define bsqav
    # get vector for correction
    M, add1, add2, add3, add4 = jax.vmap(
        get_correction_matrix,
        # Vectorize over collision species ``b``.  Mapping ``coeff`` instead
        # happened to broadcast for three species, but passed a length-four
        # correction vector into the 3-Sonine matvec in the wHe case.
        in_axes=(None, None, 0, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
    )(
        grid, a, jnp.arange(density.shape[0]), coeff, nucoeff, CM_ab, CN_ab, sum, tau, factor, correction, r_index,
        dndr, dTdr, temperature, density, charge
    )
    C = jnp.sum(M, axis=0)  # should be a (n_species,3) and we sum on all species
    ADD1 = jnp.sum(add1)
    ADD2 = jnp.sum(add2)
    ADD3 = jnp.sum(add3)
    ADD4 = jnp.sum(add4)
    particle_ps, heat_ps = _ntss_radial_flux_correction_terms(
        field,
        a,
        r_index,
        sum,
        ADD1,
        ADD2,
        ADD3,
        ADD4,
        nu_av,
        A1,
        A2,
        dTdr,
        temperature,
        mass,
        charge,
    )
    # calculate corrected fluxes for species a
    Gamma = density[a, r_index] * (-(Lij.at[0, 0].get() * A1[a, r_index] + Lij.at[0, 1].get() * A2[a, r_index] + Lij.at[0, 2].get() * A3[r_index])
        + C.at[0].get() + particle_ps)
    Q = temperature[a, r_index] * density[a, r_index] * (-(Lij.at[1, 0].get() * A1[a, r_index] + Lij.at[1, 1].get() * A2[a, r_index] + Lij.at[1, 2].get() * A3[r_index])
    +C.at[1].get() + heat_ps)
    Upar = correction.at[a, 0].get() * density[a, r_index]
    qpar = correction.at[a, 1].get()
    Upar2 = correction.at[a, 2].get()
    return Gamma, Q, Upar, qpar, Upar2



@jit
#Get momentum correction at one radial position
def get_momentum_Correction(species, energy_grid, geometry, r_index, Lij, Eij, nu_av,
                            v_thermal, density, temperature, A1, A2, A3, mass, charge, dndr, dTdr):
    #Get collisional operator expansion matrix for a radial position
    n_species = density.shape[0]
    species_indices = jnp.arange(n_species)
    CM_ab, CN_ab, tau = jax.vmap(
        lambda species_a: jax.vmap(
            lambda species_b: get_Collision_Operator_terms(
                species,
                species_a,
                species_b,
                r_index,
                temperature,
                density,
                v_thermal,
            )
        )(species_indices)
    )(species_indices)
    #construct the linear system M*solution = rhs to be solved
    #Construct rhs vector
    rhs = jax.vmap(
        lambda species_a, lij_a: get_rhs(species_a, r_index, lij_a, A1, A2, A3)
    )(species_indices, Lij)
    rhs = jnp.reshape(rhs, rhs.shape[0] * rhs.shape[1])
    #Construct matrix M=
    M = jax.vmap(
        lambda species_a, lij_a, eij_a, cm_a, cn_a, tau_a: get_Matrix(
            energy_grid,
            geometry,
            species_a,
            r_index,
            lij_a,
            eij_a,
            cm_a,
            cn_a,
            tau_a,
            v_thermal,
        )
    )(species_indices, Lij, Eij, CM_ab, CN_ab, tau)
    S = lineax.MatrixLinearOperator(jnp.reshape(M, (M.shape[0] * M.shape[1], M.shape[2])))
    #Solve linear system using lineax to get the correction 
    solution = lineax.linear_solve(S, rhs)
    corr = jnp.reshape(solution.value, (CM_ab.shape[0], CM_ab.shape[-1]))
    #Now we need to get corrected fluxes
    #Then we apply correction to fluxes for each species in a function similar to the one for getting matrix M 
    Gamma, Q, Upar, qpar, Upar2 = jax.vmap(
        lambda species_a, lij_a, eij_a, nu_a: get_corrected_fluxes(
            energy_grid,
            geometry,
            species_a,
            r_index,
            lij_a,
            eij_a,
            nu_a,
            CM_ab,
            CN_ab,
            tau,
            corr,
            v_thermal,
            density,
            temperature,
            A1,
            A2,
            A3,
            mass,
            charge,
            dndr,
            dTdr,
        )
    )(species_indices, Lij, Eij, nu_av)
    return Gamma, Q, Upar, qpar, Upar2






@jit
#Get_fluxes with momentum correction, unified interface
def get_Neoclassical_Fluxes_With_Momentum_Correction(
    species,
    energy_grid,
    geometry,
    database,
    Er,
    temperature,
    density,
    density_right_constraint=None,
    density_right_grad_constraint=None,
    temperature_right_constraint=None,
    temperature_right_grad_constraint=None,
):
    # TransportState stores normalized units (density in 1e20 m^-3,
    # temperature in keV). The low-level species/collision helpers convert
    # those to physical units internally where needed, so do not rescale here.
    v_thermal = get_v_thermal(species.mass, temperature)
    r_grid = geometry.r_grid
    r_grid_half = geometry.r_grid_half
    dr = geometry.dr

    n_species = int(temperature.shape[0])
    n_right = _as_species_constraint(density_right_constraint, n_species)
    if n_right is None:
        n_right = density[:, -1]
    n_right_grad = _as_species_constraint(density_right_grad_constraint, n_species)
    if n_right_grad is None:
        n_right_grad = jnp.zeros_like(n_right)
    t_right = _as_species_constraint(temperature_right_constraint, n_species)
    if t_right is None:
        t_right = temperature[:, -1]
    t_right_grad = _as_species_constraint(temperature_right_grad_constraint, n_species)
    if t_right_grad is None:
        t_right_grad = jnp.zeros_like(t_right)

    # Compute gradients and thermodynamic forces for each species
    def get_gradients_and_forces(density, temperature, n_rc, n_rg, t_rc, t_rg, Er, a):
        dndr = get_gradient_density(
            density,
            r_grid,
            r_grid_half,
            dr,
            right_face_constraint=n_rc,
            right_face_grad_constraint=n_rg,
        )
        dTdr = get_gradient_temperature(
            temperature,
            r_grid,
            r_grid_half,
            dr,
            right_face_constraint=t_rc,
            right_face_grad_constraint=t_rg,
        )
        A1 = get_Thermodynamical_Forces_A1(species.charge[a], density, temperature, dndr, dTdr, Er)
        A2 = get_Thermodynamical_Forces_A2(temperature, dTdr)
        A3 = get_Thermodynamical_Forces_A3(Er)
        return dndr, dTdr, A1, A2, A3

    # Vectorize over species
    grads_forces = jax.vmap(get_gradients_and_forces, in_axes=(0,0,0,0,0,0,None,0))(
        density,
        temperature,
        n_right,
        n_right_grad,
        t_right,
        t_right_grad,
        Er,
        jnp.arange(n_species),
    )
    dndr, dTdr, A1, A2, A3 = grads_forces

    species_indices = jnp.arange(n_species)
    # ``energy_grid`` is the velocity/Sonine grid while ``geometry`` is the
    # transport geometry.  The radial loop must therefore come from the
    # latter; StandardLaguerreEnergyGrid deliberately has no radial indices.
    radial_indices = geometry.full_grid_indices
    # Compute Lij, Eij, nu_weighted_average for all species and radial points
    # The coefficient kernel consumes full profile arrays and extracts both
    # the species and radius internally.  Map only those scalar indices: the
    # former nested ``in_axes`` also mapped temperature/density over axis zero
    # during the radial loop, which is invalid when n_radius != n_species.
    def _one_species(species_index):
        return jax.vmap(
            lambda radial_index: get_Lij_matrix_with_momentum_correction(
                species,
                energy_grid,
                geometry,
                database,
                species_index,
                radial_index,
                Er,
                temperature,
                density,
                v_thermal,
            )
        )(radial_indices)

    Lij, Eij, nu_weighted_average = jax.vmap(_one_species)(species_indices)
    # Adjust Lij and Eij as before
    Lij = Lij.at[:, 0, :, :].set(Lij.at[:, 1, :, :].get())
    Eij = Eij.at[:, 0, :, :].set(Eij.at[:, 1, :, :].get())
    # Compute momentum correction for all radial points
    correction = jax.vmap(
        lambda radial_index, lij_at_radius, eij_at_radius, nu_at_radius: (
            get_momentum_Correction(
                species,
                energy_grid,
                geometry,
                radial_index,
                lij_at_radius,
                eij_at_radius,
                nu_at_radius,
                v_thermal,
                density,
                temperature,
                A1,
                A2,
                A3,
                species.mass,
                species.charge,
                dndr,
                dTdr,
            )
        )
    )(
        radial_indices,
        jnp.moveaxis(Lij, 1, 0),
        jnp.moveaxis(Eij, 1, 0),
        jnp.moveaxis(nu_weighted_average, 1, 0),
    )
    # The radial map returns (radius, species); NEOPAX flux-model consumers
    # uniformly use (species, radius).
    return tuple(jnp.swapaxes(component, 0, 1) for component in correction)






