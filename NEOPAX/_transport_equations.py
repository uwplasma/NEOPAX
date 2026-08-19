from typing import Dict, Type
import dataclasses
import jax
import jax.numpy as jnp
from jax import jit
from ._fem import cell_centered_from_faces, conservative_update, faces_from_cell_centered
from ._cell_variable import make_profile_cell_variable
from ._boundary_conditions import left_constraints_from_bc_model, right_constraints_from_bc_model
from ._constants import elementary_charge
from ._source_models import (
    assemble_density_source_components,
    assemble_pressure_source_components,
    sum_source_components,
)
from ._transport_flux_models import (
    _add_float_delta_tree,
    _float_delta_tree_like,
    _sanitize_float_delta_bar_tree,
    build_evaluated_transport_state,
    build_face_transport_state,
    build_ntss_like_face_transport_state,
)
from ._transport_debug import lagged_timing_enabled, lagged_timing_start, lagged_timing_end
from ._state import (
    DEFAULT_TRANSPORT_DENSITY_FLOOR,
    DEFAULT_TRANSPORT_TEMPERATURE_FLOOR,
    _broadcast_species_floor,
    apply_transport_density_floor,
    apply_transport_temperature_floor,
    safe_density,
    safe_temperature,
)

DENSITY_STATE_TO_PHYSICAL = 1.0e20
PARTICLE_FLUX_PHYSICAL_TO_STATE = 1.0e-20
HEAT_FLUX_PHYSICAL_TO_STATE = 1.0e-23


def _minmod_pair(a, b):
    same_sign = (a * b) > 0.0
    return jnp.where(same_sign, jnp.sign(a) * jnp.minimum(jnp.abs(a), jnp.abs(b)), 0.0)


def _minmod3(a, b, c):
    return _minmod_pair(a, _minmod_pair(b, c))


def _mc_limited_face_states(profile_ghost):
    um = profile_ghost[:, :-2]
    u0 = profile_ghost[:, 1:-1]
    up = profile_ghost[:, 2:]
    slope = _minmod3(
        0.5 * (up - um),
        2.0 * (u0 - um),
        2.0 * (up - u0),
    )
    left_states = jnp.concatenate([profile_ghost[:, :1], u0 + 0.5 * slope], axis=1)
    right_states = jnp.concatenate([u0 - 0.5 * slope, profile_ghost[:, -1:]], axis=1)
    return left_states, right_states


def _temperature_face_states(temperature_ghost, reconstruction_mode):
    mode = str(reconstruction_mode).strip().lower()
    if mode in {"tvd_mc", "mc", "muscl", "muscl_tvd"}:
        return _mc_limited_face_states(temperature_ghost)
    return temperature_ghost[:, :-1], temperature_ghost[:, 1:]


def _cell_centered_flux_faces(flux, reconstruction_mode):
    mode = str(reconstruction_mode).strip().lower()
    if flux.ndim == 1:
        flux = flux[None, :]
        squeeze = True
    else:
        squeeze = False

    flux_ghost = jnp.concatenate([flux[:, :1], flux, flux[:, -1:]], axis=1)
    if mode in {"tvd_mc", "mc", "muscl", "muscl_tvd"}:
        left_states, right_states = _mc_limited_face_states(flux_ghost)
        faces = 0.5 * (left_states + right_states)
    else:
        faces = faces_from_cell_centered(flux) if flux.shape[0] == 1 else jax.vmap(faces_from_cell_centered)(flux)

    if squeeze:
        return faces[0]
    return faces


def enforce_quasi_neutrality(state, species):
    """
    Reconstruct electron density from ion densities and species charges.
    Returns a new TransportState with quasi-neutral electron density.
    """
    from ._species import get_species_idx
    charge_qp = jnp.asarray(species.charge_qp)
    eidx = get_species_idx("e", species.names)
    ion_indices = jnp.array(species.ion_indices)
    Z_i = jnp.take(charge_qp, ion_indices, axis=0)
    n_i = jnp.take(state.density, ion_indices, axis=0)
    Z_e = charge_qp[eidx]
    n_e = -jnp.sum(Z_i[:, None] * n_i, axis=0) / Z_e
    density = state.density.at[eidx, :].set(n_e)
    return dataclasses.replace(state, density=density)


def project_fixed_temperature_species(
    state,
    temperature_active_mask=None,
    fixed_temperature_profile=None,
    density_floor=DEFAULT_TRANSPORT_DENSITY_FLOOR,
):
    """
    Keep the closure temperature fixed for species whose temperature equation
    is disabled by projecting pressure = n * T_fixed on the working state.
    """
    if temperature_active_mask is None or fixed_temperature_profile is None:
        return state

    active_mask = jnp.asarray(temperature_active_mask, dtype=bool)
    if active_mask.ndim == 0:
        active_mask = active_mask[None]
    active_mask = active_mask[:, None]
    fixed_temperature = jnp.asarray(fixed_temperature_profile, dtype=state.pressure.dtype)
    fixed_pressure = safe_density(state.density, density_floor) * fixed_temperature
    pressure = jnp.where(active_mask, state.pressure, fixed_pressure)
    return dataclasses.replace(state, pressure=pressure)


def apply_er_dirichlet_boundary_state(state, er_bc_model):
    """Legacy helper kept for compatibility.

    In the cell-centered FV layout, Dirichlet values belong on faces and should
    influence the solution through face constraints/flux closure, not by
    overwriting the first or last cell-centered Er state directly.
    """
    del er_bc_model
    return state


def _expand_density_rhs_to_full_shape(density_rhs, template_density, species):
    """Expand a reduced density RHS back to full physical species ordering."""
    density_rhs = jnp.asarray(density_rhs)
    template_density = jnp.asarray(template_density)

    if density_rhs.shape == template_density.shape:
        return density_rhs

    if density_rhs.ndim != template_density.ndim or density_rhs.ndim != 2:
        return jnp.zeros_like(template_density)

    n_species = template_density.shape[0]
    if density_rhs.shape[0] == n_species - 1 and species is not None and hasattr(species, "names"):
        names = tuple(getattr(species, "names", ()))
        if "e" in names:
            eidx = names.index("e")
            out = jnp.zeros_like(template_density)
            left_width = eidx
            right_width = n_species - eidx - 1
            if left_width > 0:
                out = out.at[:left_width, :].set(density_rhs[:left_width, :])
            if right_width > 0:
                out = out.at[eidx + 1 :, :].set(density_rhs[left_width:, :])
            return out

    return jnp.zeros_like(template_density)

@jit
def _plasma_permitivity_from_prefactor(state, species_mass, permitivity_prefactor):
    """Plasma permittivity on the transport grid using a precomputed geometry prefactor."""
    mass_density = DENSITY_STATE_TO_PHYSICAL * jnp.sum(species_mass[:, None] * state.density, axis=0)
    return mass_density * permitivity_prefactor


# --- Modular Equation Registry and Base ---
__equation_registry: Dict[str, Type] = {}

def register_equation(name: str):
    """Decorator to register equation classes in the registry."""
    def decorator(cls):
        __equation_registry[name] = cls
        return cls
    return decorator


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class EquationBase:
    """
    Base class for transport equations. Subclasses must implement __call__.
    """

    def __call__(self, state, models, field, species, energy_grid, database, solver_parameters, bc=None, **kwargs):
        raise NotImplementedError

def get_equation(name: str) -> Type:
    return __equation_registry[name]

def list_equations():
    return list(__equation_registry.keys())

# --- Example built-in equation: Density evolution ---

# --- JAX-friendly, torax-style DensityEquation ---
@register_equation("density")
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class TransportLaggedResponse:
    flux_response: object = dataclasses.field(repr=False, default=None)


def _flux_has_key(fluxes, key):
    return isinstance(fluxes, dict) and key in fluxes and fluxes.get(key, None) is not None


def _get_face_flux(fluxes, key):
    face_key = f"{key}_faces"
    if _flux_has_key(fluxes, face_key):
        return fluxes[face_key]
    if _flux_has_key(fluxes, key):
        return fluxes[key]
    return None


def _get_center_flux(fluxes, key):
    if _flux_has_key(fluxes, key):
        return fluxes[key]
    face_value = _get_face_flux(fluxes, key)
    if face_value is None:
        return None
    return jax.vmap(cell_centered_from_faces)(face_value)


def _with_center_fluxes_from_faces(fluxes, keys=("Gamma", "Q", "Upar")):
    """Return a center-view of face-primary fluxes without changing lagged storage."""
    if not isinstance(fluxes, dict):
        return fluxes
    out = dict(fluxes)
    for key in keys:
        if not _flux_has_key(out, key):
            center_value = _get_center_flux(out, key)
            if center_value is not None:
                out[key] = center_value
    return out


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class DensityEquation(EquationBase):
    dr_cells: jax.Array = dataclasses.field(repr=False)
    Vprime: jax.Array = dataclasses.field(repr=False)
    Vprime_half: jax.Array = dataclasses.field(repr=False)
    flux_model: callable = dataclasses.field(repr=False)
    flux_faces_builder: callable = dataclasses.field(repr=False)
    active_species_mask: jax.Array = dataclasses.field(repr=False)
    independent_density_mask: jax.Array = dataclasses.field(repr=False)
    face_flux_builder: callable = dataclasses.field(repr=False, default=None)
    density_bc_model: object = dataclasses.field(repr=False, default=None)
    particle_flux_reconstruction: str = "closure_face_flux"
    particle_face_closure_mode: str = "reconstructed"
    source_model: callable = dataclasses.field(repr=False, default=None)
    species: object = dataclasses.field(repr=False, default=None)
    name: str = "density"

    def _mode_requests_face_fluxes(self, mode_value):
        mode = str(mode_value).strip().lower()
        return mode in {"closure_face_flux", "model_face_flux", "face_closure"}

    def _use_model_face_particle_fluxes(self):
        return self._mode_requests_face_fluxes(self.particle_flux_reconstruction)

    def enforce_dirichlet_boundary_rhs(self, state, density_rhs):
        del state
        return density_rhs

    def debug_components(self, state, fluxes=None, source_outputs=None):
        if fluxes is None:
            fluxes = self.flux_model(state)
        use_face_gamma = self._use_model_face_particle_fluxes()
        need_face_fluxes = use_face_gamma
        face_fluxes = (
            fluxes
            if (need_face_fluxes and _flux_has_key(fluxes, "Gamma_faces"))
            else (
                self.face_flux_builder(state, center_fluxes=fluxes)
                if (self.face_flux_builder is not None and need_face_fluxes)
                else None
            )
        )
        gamma_center = _get_center_flux(fluxes, "Gamma")
        Gamma = (
            PARTICLE_FLUX_PHYSICAL_TO_STATE * gamma_center
            if gamma_center is not None
            else None
        )
        Gamma_faces_raw = (
            PARTICLE_FLUX_PHYSICAL_TO_STATE * _get_face_flux(face_fluxes, "Gamma")
            if (face_fluxes is not None and _get_face_flux(face_fluxes, "Gamma") is not None and use_face_gamma)
            else self.flux_faces_builder(Gamma, self.particle_flux_reconstruction)
        )
        gamma_divergence_raw = jax.vmap(
            lambda flux: conservative_update(flux, self.dr_cells, self.Vprime, self.Vprime_half)
        )(Gamma_faces_raw)
        source_components = assemble_density_source_components(
            None if self.source_model is None else (self.source_model(state) if source_outputs is None else source_outputs),
            state,
            self.species,
        )
        source_rhs = sum_source_components(source_components, state.density)
        Gamma_faces = Gamma_faces_raw * self.independent_density_mask[:, None]
        gamma_divergence = gamma_divergence_raw * self.independent_density_mask[:, None]
        density_rhs_raw = gamma_divergence_raw + source_rhs
        density_rhs = density_rhs_raw * self.independent_density_mask[:, None]
        return {
            "Gamma_center": Gamma,
            "Gamma_faces_raw": Gamma_faces_raw,
            "Gamma_faces": Gamma_faces,
            "gamma_divergence_raw": gamma_divergence_raw,
            "gamma_divergence": gamma_divergence,
            "gamma_divergence_active": gamma_divergence,
            **{f"source_{key}": value for key, value in source_components.items()},
            "source_rhs": source_rhs,
            "density_rhs_raw": density_rhs_raw,
            "density_rhs": density_rhs,
        }

    def __call__(self, state, fluxes=None, source_outputs=None):
        if fluxes is None:
            fluxes = self.flux_model(state)
        use_face_gamma = self._use_model_face_particle_fluxes()
        need_face_fluxes = use_face_gamma
        face_fluxes = (
            fluxes
            if (need_face_fluxes and _flux_has_key(fluxes, "Gamma_faces"))
            else (
                self.face_flux_builder(state, center_fluxes=fluxes)
                if (self.face_flux_builder is not None and need_face_fluxes)
                else None
            )
        )
        gamma_center = _get_center_flux(fluxes, "Gamma")
        Gamma = (
            PARTICLE_FLUX_PHYSICAL_TO_STATE * gamma_center
            if gamma_center is not None
            else None
        )
        Gamma_faces = (
            PARTICLE_FLUX_PHYSICAL_TO_STATE * (
                _get_face_flux(face_fluxes, "Gamma")
            )
            if (face_fluxes is not None and use_face_gamma)
            else self.flux_faces_builder(Gamma, self.particle_flux_reconstruction)
        )
        gamma_divergence = jax.vmap(
            lambda flux: conservative_update(flux, self.dr_cells, self.Vprime, self.Vprime_half)
        )(Gamma_faces)
        source_rhs = jnp.zeros_like(gamma_divergence)
        if self.source_model is not None:
            source_components = assemble_density_source_components(
                self.source_model(state) if source_outputs is None else source_outputs,
                state,
                self.species,
            )
            source_rhs = sum_source_components(source_components, state.density)
        density_rhs = gamma_divergence + source_rhs
        return density_rhs * self.independent_density_mask[:, None]

# --- Factory function to build DensityEquation up front ---
def build_density_equation(
    field,
    flux_model,
    source_model,
    bc_density,
    species,
    bc_temperature=None,
    bc_er=None,
    reconstruction="linear",
    active_species_mask=None,
    particle_flux_reconstruction="closure_face_flux",
    particle_face_closure_mode="reconstructed",
    density_floor=DEFAULT_TRANSPORT_DENSITY_FLOOR,
    temperature_floor=DEFAULT_TRANSPORT_TEMPERATURE_FLOOR,
):
    dr_cells = jnp.diff(field.r_grid_half)
    Vprime = field.Vprime
    Vprime_half = field.Vprime_half
    def flux_faces_builder(flux, face_reconstruction="centered"):
        return _cell_centered_flux_faces(flux, face_reconstruction)
    def face_flux_builder(state, center_fluxes=None):
        state = apply_transport_density_floor(state, density_floor)
        state = apply_transport_temperature_floor(state, temperature_floor, density_floor)
        evaluated_state = build_evaluated_transport_state(
            state,
            field,
            bc_density=bc_density,
            bc_temperature=bc_temperature,
            bc_er=bc_er,
            reconstruction=reconstruction,
            density_floor=density_floor,
            temperature_floor=temperature_floor,
        )
        face_mode = str(particle_face_closure_mode).strip().lower()
        if face_mode in {"ntss_like", "ntss", "half_point"}:
            face_state = build_ntss_like_face_transport_state(
                state,
                field,
                bc_density=bc_density,
                bc_temperature=bc_temperature,
                bc_er=bc_er,
                density_floor=density_floor,
                temperature_floor=temperature_floor,
            )
        else:
            face_state = build_face_transport_state(
                state,
                field,
                bc_density=bc_density,
                bc_temperature=bc_temperature,
                bc_er=bc_er,
                reconstruction=reconstruction,
                density_floor=density_floor,
                temperature_floor=temperature_floor,
            )
        return flux_model.evaluate_face_fluxes(
            state,
            face_state,
            bc_density=bc_density,
            bc_temperature=bc_temperature,
            bc_er=bc_er,
            particle_face_closure_mode=face_mode,
            center_fluxes=center_fluxes,
            evaluated_state=evaluated_state,
        )
    if active_species_mask is None:
        active_species_mask = jnp.ones(species.number_species, dtype=bool)
    active_species_mask = jnp.asarray(active_species_mask, dtype=bool)
    independent_density_mask = active_species_mask
    if hasattr(species, "names") and "e" in tuple(species.names):
        eidx = tuple(species.names).index("e")
        independent_density_mask = independent_density_mask.at[eidx].set(False)
    return DensityEquation(
        dr_cells=dr_cells,
        Vprime=Vprime,
        Vprime_half=Vprime_half,
        flux_model=flux_model,
        source_model=source_model,
        flux_faces_builder=flux_faces_builder,
        active_species_mask=active_species_mask,
        independent_density_mask=independent_density_mask,
        face_flux_builder=face_flux_builder,
        density_bc_model=bc_density,
        particle_flux_reconstruction=str(particle_flux_reconstruction),
        particle_face_closure_mode=str(particle_face_closure_mode),
        species=species,
    )

# --- Example built-in equation: Pressure evolution ---

# --- JAX-friendly, torax-style PressureEquation ---
@register_equation("temperature")
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class TemperatureEquation(EquationBase):
    dr_cells: jax.Array = dataclasses.field(repr=False)
    Vprime: jax.Array = dataclasses.field(repr=False)
    Vprime_half: jax.Array = dataclasses.field(repr=False)
    flux_model: callable = dataclasses.field(repr=False)
    flux_faces_builder: callable = dataclasses.field(repr=False)
    temperature_ghost_builder: callable = dataclasses.field(repr=False)
    charge_qp: jax.Array = dataclasses.field(repr=False)
    active_species_mask: jax.Array = dataclasses.field(repr=False)
    face_flux_builder: callable = dataclasses.field(repr=False, default=None)
    temperature_bc_model: object = dataclasses.field(repr=False, default=None)
    convection_reconstruction: str = "tvd_mc"
    heat_flux_reconstruction: str = "tvd_mc"
    include_neo_convection: bool = True
    include_turbulent_convection: bool = True
    include_classical_convection: bool = True
    include_work_term: bool = True
    source_model: callable = dataclasses.field(repr=False, default=None)
    species: object = dataclasses.field(repr=False, default=None)
    name: str = "temperature"

    def _mode_requests_face_fluxes(self, mode_value):
        mode = str(mode_value).strip().lower()
        return mode in {"closure_face_flux", "model_face_flux", "face_closure"}

    def _use_model_face_heat_fluxes(self):
        return self._mode_requests_face_fluxes(self.heat_flux_reconstruction)

    def _use_model_face_particle_fluxes(self):
        return self._mode_requests_face_fluxes(self.convection_reconstruction)

    def enforce_dirichlet_boundary_rhs(self, state, density_rhs, pressure_rhs):
        del state, density_rhs
        return pressure_rhs

    def debug_components(self, state, fluxes=None, source_outputs=None):
        if fluxes is None:
            fluxes = self.flux_model(state)
        use_face_q = self._use_model_face_heat_fluxes()
        use_face_gamma = self._use_model_face_particle_fluxes()
        need_face_fluxes = use_face_q or use_face_gamma
        face_fluxes = (
            fluxes
            if (
                need_face_fluxes
                and (_flux_has_key(fluxes, "Q_faces") or _flux_has_key(fluxes, "Gamma_faces"))
            )
            else (
                self.face_flux_builder(state, center_fluxes=fluxes)
                if (self.face_flux_builder is not None and need_face_fluxes)
                else None
            )
        )
        q_center = _get_center_flux(fluxes, "Q")
        Q = HEAT_FLUX_PHYSICAL_TO_STATE * q_center if q_center is not None else None
        temperature_ghost = self.temperature_ghost_builder(state.temperature)
        Q_faces = (
            HEAT_FLUX_PHYSICAL_TO_STATE * (
                _get_face_flux(face_fluxes, "Q")
            )
            if (face_fluxes is not None and use_face_q)
            else self.flux_faces_builder(Q, self.heat_flux_reconstruction)
        )
        temperature_left, temperature_right = _temperature_face_states(
            temperature_ghost,
            self.convection_reconstruction,
        )

        def _convective_component(gamma_key):
            gamma_face_key = f"{gamma_key}_faces"
            gamma_comp = (
                _get_face_flux(face_fluxes, gamma_key)
                if (face_fluxes is not None and use_face_gamma)
                else _get_center_flux(fluxes, gamma_key)
            )
            if gamma_comp is None:
                gamma_faces = jnp.zeros_like(Q_faces)
            elif face_fluxes is not None and use_face_gamma:
                gamma_faces = PARTICLE_FLUX_PHYSICAL_TO_STATE * gamma_comp
            else:
                gamma_faces = self.flux_faces_builder(PARTICLE_FLUX_PHYSICAL_TO_STATE * gamma_comp)
            temperature_upwind = jnp.where(gamma_faces >= 0.0, temperature_left, temperature_right)
            return gamma_faces, temperature_upwind * gamma_faces

        gamma_neo_faces, convective_neo_faces = (
            _convective_component("Gamma_neo")
            if self.include_neo_convection
            else (jnp.zeros_like(Q_faces), jnp.zeros_like(Q_faces))
        )
        gamma_turb_faces, convective_turb_faces = (
            _convective_component("Gamma_turb")
            if self.include_turbulent_convection
            else (jnp.zeros_like(Q_faces), jnp.zeros_like(Q_faces))
        )
        gamma_classical_faces, convective_classical_faces = (
            _convective_component("Gamma_classical")
            if self.include_classical_convection
            else (jnp.zeros_like(Q_faces), jnp.zeros_like(Q_faces))
        )
        total_energy_flux_faces = Q_faces + convective_neo_faces + convective_turb_faces + convective_classical_faces

        q_divergence = jax.vmap(
            lambda flux: conservative_update(flux, self.dr_cells, self.Vprime, self.Vprime_half)
        )(Q_faces)
        convective_neo_divergence = jax.vmap(
            lambda flux: conservative_update(flux, self.dr_cells, self.Vprime, self.Vprime_half)
        )(convective_neo_faces)
        convective_turb_divergence = jax.vmap(
            lambda flux: conservative_update(flux, self.dr_cells, self.Vprime, self.Vprime_half)
        )(convective_turb_faces)
        convective_classical_divergence = jax.vmap(
            lambda flux: conservative_update(flux, self.dr_cells, self.Vprime, self.Vprime_half)
        )(convective_classical_faces)
        thermal_flux_rhs = jax.vmap(
            lambda flux: conservative_update(flux, self.dr_cells, self.Vprime, self.Vprime_half)
        )(total_energy_flux_faces)
        source_components = assemble_pressure_source_components(
            None if self.source_model is None else (self.source_model(state) if source_outputs is None else source_outputs),
            state,
            self.species,
        )
        source_rhs = sum_source_components(source_components, state.pressure)
        work_rhs = (
            self.charge_qp[:, None]
            * PARTICLE_FLUX_PHYSICAL_TO_STATE
            * _get_center_flux(fluxes, "Gamma")
            * state.Er[None, :]
            if self.include_work_term
            else jnp.zeros_like(state.pressure)
        )
        total_rhs = (2.0 / 3.0) * (thermal_flux_rhs + source_rhs + work_rhs)
        return {
            "Q_faces": Q_faces,
            "Gamma_neo_faces": gamma_neo_faces,
            "Gamma_turb_faces": gamma_turb_faces,
            "Gamma_classical_faces": gamma_classical_faces,
            "convective_neo_faces": convective_neo_faces,
            "convective_turb_faces": convective_turb_faces,
            "convective_classical_faces": convective_classical_faces,
            "q_divergence": q_divergence,
            "convective_neo_divergence": convective_neo_divergence,
            "convective_turb_divergence": convective_turb_divergence,
            "convective_classical_divergence": convective_classical_divergence,
            "thermal_flux_rhs": thermal_flux_rhs,
            **{f"source_{key}": value for key, value in source_components.items()},
            "source_rhs": source_rhs,
            "work_rhs": work_rhs,
            "pressure_rhs": total_rhs * self.active_species_mask[:, None],
        }

    def __call__(self, state, fluxes=None, source_outputs=None):
        if fluxes is None:
            fluxes = self.flux_model(state)
        use_face_q = self._use_model_face_heat_fluxes()
        use_face_gamma = self._use_model_face_particle_fluxes()
        need_face_fluxes = use_face_q or use_face_gamma
        face_fluxes = (
            fluxes
            if (
                need_face_fluxes
                and (_flux_has_key(fluxes, "Q_faces") or _flux_has_key(fluxes, "Gamma_faces"))
            )
            else (
                self.face_flux_builder(state, center_fluxes=fluxes)
                if (self.face_flux_builder is not None and need_face_fluxes)
                else None
            )
        )
        q_center = _get_center_flux(fluxes, "Q")
        Q = HEAT_FLUX_PHYSICAL_TO_STATE * q_center if q_center is not None else None
        temperature_ghost = self.temperature_ghost_builder(state.temperature)
        Q_faces = (
            HEAT_FLUX_PHYSICAL_TO_STATE * _get_face_flux(face_fluxes, "Q")
            if (face_fluxes is not None and use_face_q)
            else self.flux_faces_builder(Q, self.heat_flux_reconstruction)
        )
        temperature_left, temperature_right = _temperature_face_states(
            temperature_ghost,
            self.convection_reconstruction,
        )

        def _convective_component(gamma_key):
            gamma_comp = (
                _get_face_flux(face_fluxes, gamma_key)
                if (face_fluxes is not None and use_face_gamma)
                else _get_center_flux(fluxes, gamma_key)
            )
            if gamma_comp is None:
                gamma_faces = jnp.zeros_like(Q_faces)
            elif face_fluxes is not None and use_face_gamma:
                gamma_faces = PARTICLE_FLUX_PHYSICAL_TO_STATE * gamma_comp
            else:
                gamma_faces = self.flux_faces_builder(PARTICLE_FLUX_PHYSICAL_TO_STATE * gamma_comp)
            temperature_upwind = jnp.where(gamma_faces >= 0.0, temperature_left, temperature_right)
            return temperature_upwind * gamma_faces

        convective_flux_faces = jnp.zeros_like(Q_faces)
        if self.include_neo_convection:
            convective_flux_faces = convective_flux_faces + _convective_component("Gamma_neo")
        if self.include_turbulent_convection:
            convective_flux_faces = convective_flux_faces + _convective_component("Gamma_turb")
        if self.include_classical_convection:
            convective_flux_faces = convective_flux_faces + _convective_component("Gamma_classical")

        total_energy_flux_faces = Q_faces + convective_flux_faces
        thermal_flux_rhs = jax.vmap(
            lambda flux: conservative_update(flux, self.dr_cells, self.Vprime, self.Vprime_half)
        )(total_energy_flux_faces)
        source_components = assemble_pressure_source_components(
            None if self.source_model is None else (self.source_model(state) if source_outputs is None else source_outputs),
            state,
            self.species,
        )
        source_rhs = sum_source_components(source_components, state.pressure)
        work_rhs = (
            self.charge_qp[:, None]
            * PARTICLE_FLUX_PHYSICAL_TO_STATE
            * _get_center_flux(fluxes, "Gamma")
            * state.Er[None, :]
            if self.include_work_term
            else jnp.zeros_like(state.pressure)
        )
        return (2.0 / 3.0) * (thermal_flux_rhs + source_rhs + work_rhs) * self.active_species_mask[:, None]

def _build_species_faces_builder(field, bc_model, reconstruction="linear"):
    if bc_model is not None and hasattr(bc_model, "right_type"):
        def faces_builder(profile):
            lv, lg = left_constraints_from_bc_model(
                bc_model,
                profile[:, 0],
                profile=profile,
                face_centers=field.r_grid_half,
            )
            rv, rg = right_constraints_from_bc_model(
                bc_model,
                profile[:, -1],
                profile=profile,
                face_centers=field.r_grid_half,
            )
            if rv is not None:
                return jax.vmap(
                    lambda prof, left_val, left_grad, right_val: make_profile_cell_variable(
                        prof,
                        field.r_grid_half,
                        left_face_constraint=left_val,
                        left_face_grad_constraint=left_grad,
                        right_face_constraint=right_val,
                    ).face_value(reconstruction=reconstruction)
                )(profile, lv, lg, jnp.asarray(rv))
            return jax.vmap(
                lambda prof, left_val, left_grad, right_grad: make_profile_cell_variable(
                    prof,
                    field.r_grid_half,
                    left_face_constraint=left_val,
                    left_face_grad_constraint=left_grad,
                    right_face_grad_constraint=right_grad,
                ).face_value(reconstruction=reconstruction)
            )(profile, lv, lg, jnp.asarray(rg))
    elif bc_model is not None and hasattr(bc_model, "apply_ghost"):
        def faces_builder(profile):
            if hasattr(bc_model, "apply_ghost_all"):
                ghost = bc_model.apply_ghost_all(profile)
            else:
                ghost = jax.vmap(lambda prof: bc_model.apply_ghost(prof))(profile)
            return jax.vmap(faces_from_cell_centered)(ghost)
    else:
        def faces_builder(profile):
            return jax.vmap(
                lambda prof: make_profile_cell_variable(
                    prof,
                    field.r_grid_half,
                    left_face_grad_constraint=jnp.asarray(0.0, dtype=prof.dtype),
                    right_face_constraint=(
                        1.5 * prof[-1] - 0.5 * prof[-2]
                        if prof.shape[0] >= 2
                        else prof[-1]
                    ),
                ).face_value(reconstruction=reconstruction)
            )(profile)
    return faces_builder


def _build_species_ghost_builder(field, bc_model):
    if bc_model is not None and hasattr(bc_model, "apply_ghost_all"):
        def ghost_builder(profile):
            return bc_model.apply_ghost_all(profile)
    elif bc_model is not None and hasattr(bc_model, "apply_ghost"):
        def ghost_builder(profile):
            return jax.vmap(lambda prof: bc_model.apply_ghost(prof))(profile)
    elif bc_model is not None and hasattr(bc_model, "right_type"):
        def ghost_builder(profile):
            left_value, left_grad = left_constraints_from_bc_model(
                bc_model,
                profile[:, 0],
                profile=profile,
                face_centers=field.r_grid_half,
            )
            right_value, right_grad = right_constraints_from_bc_model(
                bc_model,
                profile[:, -1],
                profile=profile,
                face_centers=field.r_grid_half,
            )
            dx_left = field.r_grid_half[1] - field.r_grid_half[0]
            dx_right = field.r_grid_half[-1] - field.r_grid_half[-2]

            if left_value is None:
                left_face = profile[:, 0] - 0.5 * dx_left * jnp.asarray(left_grad)
            else:
                left_face = jnp.asarray(left_value)
            if right_value is None:
                right_face = profile[:, -1] + 0.5 * dx_right * jnp.asarray(right_grad)
            else:
                right_face = jnp.asarray(right_value)

            left_ghost = 2.0 * left_face - profile[:, 0]
            right_ghost = 2.0 * right_face - profile[:, -1]
            return jnp.concatenate([left_ghost[:, None], profile, right_ghost[:, None]], axis=1)
    else:
        def ghost_builder(profile):
            left_ghost = profile[:, :1]
            if profile.shape[1] >= 2:
                right_face = 1.5 * profile[:, -1] - 0.5 * profile[:, -2]
            else:
                right_face = profile[:, -1]
            right_ghost = 2.0 * right_face[:, None] - profile[:, -1:]
            return jnp.concatenate([left_ghost, profile, right_ghost], axis=1)
    return ghost_builder


# --- Factory function to build PressureEquation up front ---
def build_temperature_equation(
    field,
        flux_model,
        source_model,
        species,
        bc_temperature,
    bc_density=None,
    bc_gamma=None,
    bc_er=None,
    active_species_mask=None,
    charge_qp=None,
    include_neo_convection=True,
    include_turbulent_convection=True,
    include_classical_convection=True,
    include_work_term=True,
    convection_reconstruction="tvd_mc",
    heat_flux_reconstruction="tvd_mc",
    reconstruction="linear",
    density_floor=DEFAULT_TRANSPORT_DENSITY_FLOOR,
    temperature_floor=DEFAULT_TRANSPORT_TEMPERATURE_FLOOR,
):
    dr_cells = jnp.diff(field.r_grid_half)
    Vprime = field.Vprime
    Vprime_half = field.Vprime_half
    def flux_faces_builder(flux, face_reconstruction="centered"):
        return _cell_centered_flux_faces(flux, face_reconstruction)
    def face_flux_builder(state, center_fluxes=None):
        state = apply_transport_density_floor(state, density_floor)
        state = apply_transport_temperature_floor(state, temperature_floor, density_floor)
        evaluated_state = build_evaluated_transport_state(
            state,
            field,
            bc_density=bc_density,
            bc_temperature=bc_temperature,
            bc_er=bc_er,
            reconstruction=reconstruction,
            density_floor=density_floor,
            temperature_floor=temperature_floor,
        )
        face_state = build_face_transport_state(
            state,
            field,
            bc_density=bc_density,
            bc_temperature=bc_temperature,
            bc_er=bc_er,
            reconstruction=reconstruction,
            density_floor=density_floor,
            temperature_floor=temperature_floor,
        )
        return flux_model.evaluate_face_fluxes(
            state,
            face_state,
            bc_density=bc_density,
            bc_temperature=bc_temperature,
            bc_er=bc_er,
            center_fluxes=center_fluxes,
            evaluated_state=evaluated_state,
        )
    temperature_ghost_builder = _build_species_ghost_builder(field, bc_temperature)
    if active_species_mask is None:
        active_species_mask = jnp.ones(species.number_species, dtype=bool)
    return TemperatureEquation(
        dr_cells=dr_cells,
        Vprime=Vprime,
        Vprime_half=Vprime_half,
        flux_model=flux_model,
        source_model=source_model,
        species=species,
        flux_faces_builder=flux_faces_builder,
        face_flux_builder=face_flux_builder,
        temperature_ghost_builder=temperature_ghost_builder,
        temperature_bc_model=bc_temperature,
        charge_qp=jnp.asarray(charge_qp),
        active_species_mask=jnp.asarray(active_species_mask, dtype=bool),
        include_neo_convection=bool(include_neo_convection),
        include_turbulent_convection=bool(include_turbulent_convection),
        include_classical_convection=bool(include_classical_convection),
        include_work_term=bool(include_work_term),
        convection_reconstruction=str(convection_reconstruction),
        heat_flux_reconstruction=str(heat_flux_reconstruction),
    )

# --- Example built-in equation: Electric field (Er) evolution ---

# --- JAX-friendly, torax-style ElectricFieldEquation ---
@register_equation("Er")
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class ElectricFieldEquation(EquationBase):
    dr_cells: jax.Array = dataclasses.field(repr=False)
    Vprime: jax.Array = dataclasses.field(repr=False)
    Vprime_half: jax.Array = dataclasses.field(repr=False)
    flux_model: callable = dataclasses.field(repr=False)
    species_mass: jax.Array = dataclasses.field(repr=False)
    charge_qp: jax.Array = dataclasses.field(repr=False)
    permitivity_prefactor: jax.Array = dataclasses.field(repr=False)
    gamma_faces_builder: callable = dataclasses.field(repr=False)
    er_diffusive_flux_builder: callable = dataclasses.field(repr=False)
    er_bc_model: object = dataclasses.field(repr=False, default=None)
    source_mode: str = "ambipolar_local"
    permitivity_mode: str = "neopax_local"
    Er_relax: float = 1.0
    DEr: float = 1.0
    boundary_mode: str = "standard"
    ntss_B0_mid: float = 0.0
    ntss_psfactor_mid: float = 1.0
    ntss_density_indices: jax.Array = dataclasses.field(repr=False, default=None)
    name: str = "Er"

    def _charge_flux_from_gamma(self, Gamma):
        mode = str(self.source_mode).strip().lower()
        if mode in {"ambipolar_local", "transport_local", "local"}:
            return jnp.sum(self.charge_qp[:, None] * Gamma, axis=0)

        Gamma_faces = self.gamma_faces_builder(Gamma)
        ambipolar_flux_center = 0.5 * (Gamma_faces[:, :-1] + Gamma_faces[:, 1:])
        return jnp.sum(self.charge_qp[:, None] * ambipolar_flux_center, axis=0)

    def _charge_flux_faces_from_gamma(self, Gamma):
        Gamma_faces = self.gamma_faces_builder(Gamma)
        return jnp.sum(self.charge_qp[:, None] * Gamma_faces, axis=0)

    def _er_diffusion(self, Er):
        # When DEr == 0 we want a true pure-ambipolar RHS, not 0 * NaN.
        if float(self.DEr) == 0.0:
            er_diffusive_flux = jnp.zeros(Er.shape[0] + 1, dtype=Er.dtype)
            er_diffusion = jnp.zeros_like(Er)
        else:
            er_diffusive_flux = self.er_diffusive_flux_builder(Er)
            er_diffusion = conservative_update(
                er_diffusive_flux, self.dr_cells, self.Vprime, self.Vprime_half
            )
        return er_diffusive_flux, er_diffusion

    def _charge_flux_and_ambi_term(self, state, Gamma, plasma_permitivity):
        charge_flux = self._charge_flux_from_gamma(Gamma)
        mode = str(self.permitivity_mode).strip().lower()
        if mode in {"ntss_like_midpoint", "ntss_like", "ntssfusion_midpoint"}:
            density_indices = self.ntss_density_indices
            if density_indices is None:
                ni_mid = jnp.asarray(1.0, dtype=charge_flux.dtype)
            else:
                ni_mid = jnp.sum(state.density[density_indices, state.density.shape[1] // 2])
            ni_mid = jnp.maximum(ni_mid, jnp.asarray(1.0e-30, dtype=charge_flux.dtype))
            coeffG = (
                jnp.asarray(95780.0, dtype=charge_flux.dtype)
                * jnp.asarray(self.ntss_B0_mid, dtype=charge_flux.dtype) ** 2
                / (ni_mid * jnp.asarray(self.ntss_psfactor_mid, dtype=charge_flux.dtype))
            )
            ambi_term = coeffG * (charge_flux * jnp.asarray(1.0e-20, dtype=charge_flux.dtype))
            return charge_flux, ambi_term

        ambi_term = charge_flux * elementary_charge * 1.0e-3 / plasma_permitivity
        return charge_flux, ambi_term

    def _outer_face_ambi_term(self, state, Gamma, plasma_permitivity):
        charge_flux_faces = self._charge_flux_faces_from_gamma(Gamma)
        charge_flux_edge = charge_flux_faces[-1]
        mode = str(self.permitivity_mode).strip().lower()
        if mode in {"ntss_like_midpoint", "ntss_like", "ntssfusion_midpoint"}:
            density_indices = self.ntss_density_indices
            if density_indices is None:
                ni_mid = jnp.asarray(1.0, dtype=charge_flux_edge.dtype)
            else:
                ni_mid = jnp.sum(state.density[density_indices, state.density.shape[1] // 2])
            ni_mid = jnp.maximum(ni_mid, jnp.asarray(1.0e-30, dtype=charge_flux_edge.dtype))
            coeffG = (
                jnp.asarray(95780.0, dtype=charge_flux_edge.dtype)
                * jnp.asarray(self.ntss_B0_mid, dtype=charge_flux_edge.dtype) ** 2
                / (ni_mid * jnp.asarray(self.ntss_psfactor_mid, dtype=charge_flux_edge.dtype))
            )
            return coeffG * (charge_flux_edge * jnp.asarray(1.0e-20, dtype=charge_flux_edge.dtype))

        plasma_permitivity_edge = (
            1.5 * plasma_permitivity[-1] - 0.5 * plasma_permitivity[-2]
            if plasma_permitivity.shape[0] >= 2
            else plasma_permitivity[-1]
        )
        plasma_permitivity_edge = jnp.maximum(
            plasma_permitivity_edge,
            jnp.asarray(1.0e-30, dtype=charge_flux_edge.dtype),
        )
        return charge_flux_edge * elementary_charge * 1.0e-3 / plasma_permitivity_edge

    def debug_components(self, state, fluxes=None):
        if fluxes is None:
            fluxes = self.flux_model(state)
        Er = state.Er
        plasma_permitivity = _plasma_permitivity_from_prefactor(
            state,
            self.species_mass,
            self.permitivity_prefactor,
        )
        Gamma = _get_center_flux(fluxes, "Gamma")
        charge_flux, ambi_term = self._charge_flux_and_ambi_term(state, Gamma, plasma_permitivity)
        ambi_term_edge = self._outer_face_ambi_term(state, Gamma, plasma_permitivity)
        er_diffusive_flux, er_diffusion = self._er_diffusion(Er)
        return {
            "charge_flux": charge_flux,
            "plasma_permitivity": plasma_permitivity,
            "ambi_term": ambi_term,
            "ambi_term_edge": ambi_term_edge,
            "er_diffusive_flux": er_diffusive_flux,
            "er_diffusion": er_diffusion,
        }

    def __call__(self, state, fluxes=None):
        if fluxes is None:
            fluxes = self.flux_model(state)
        Er = state.Er
        plasma_permitivity = _plasma_permitivity_from_prefactor(
            state,
            self.species_mass,
            self.permitivity_prefactor,
        )
        Gamma = _get_center_flux(fluxes, "Gamma")
        _, ambi_term = self._charge_flux_and_ambi_term(state, Gamma, plasma_permitivity)
        _, Er_diffusion = self._er_diffusion(Er)
        SourceEr = self.Er_relax * (self.DEr * Er_diffusion - ambi_term)
        if self.boundary_mode == "floating_ambipolar_edge":
            SourceEr = SourceEr.at[-1].set(-self.Er_relax * self._outer_face_ambi_term(state, Gamma, plasma_permitivity))
        SourceEr = self.enforce_dirichlet_boundary_rhs(state, SourceEr)
        return SourceEr

    def enforce_dirichlet_boundary_rhs(self, state, er_rhs):
        del state
        return er_rhs

    def ap_linear_split(self, state):
        """
        Return diagonal linearization and explicit source for optional AP preconditioning.
        Uses only attributes set at construction and the current state.
        """
        def diffusion_part(er_vec):
            er_diffusive_flux = self.er_diffusive_flux_builder(er_vec)
            er_diff = conservative_update(er_diffusive_flux, self.dr_cells, self.Vprime, self.Vprime_half)
            return self.Er_relax * self.DEr * er_diff

        diag_linear = jnp.diag(jax.jacfwd(diffusion_part)(state.Er))
        full_rhs = self(state)
        explicit_source = full_rhs - diag_linear * state.Er
        return diag_linear, explicit_source

# --- Factory function to build ElectricFieldEquation up front ---
def build_electric_field_equation(
    field,
    species_names,
    flux_model,
    species_mass,
    charge_qp,
    bc_gamma,
    bc_er,
    Er_relax=1.0,
    DEr=1.0,
    source_mode="ambipolar_local",
    permitivity_mode="neopax_local",
    reconstruction="linear",
    boundary_mode="standard",
):
    dr_cells = jnp.diff(field.r_grid_half)
    Vprime = field.Vprime
    Vprime_half = field.Vprime_half
    psi_den = field.enlogation * jnp.square(field.iota)
    psi_den_active = jnp.abs(psi_den) > 0.0
    psi_den_safe = jnp.where(psi_den_active, psi_den, 1.0)
    psi_fac = 1.0 + jnp.where(psi_den_active, 1.0 / psi_den_safe, 0.0)
    psi_fac = psi_fac.at[0].set(1.0)
    permitivity_prefactor = psi_fac / jnp.square(field.B0)
    mid_idx = int(field.r_grid.shape[0] // 2)
    ntss_B0_mid = jnp.asarray(field.B0[mid_idx])
    ntss_psfactor_mid = jnp.asarray(psi_fac[mid_idx])
    names = tuple(species_names) if species_names is not None else ()
    dt_indices = [i for i, name in enumerate(names) if str(name) in {"D", "T"}]
    if dt_indices:
        ntss_density_indices = jnp.asarray(dt_indices, dtype=jnp.int32)
    else:
        ion_indices = [i for i, q in enumerate(jnp.asarray(charge_qp)) if float(q) > 0.0]
        ntss_density_indices = jnp.asarray(ion_indices, dtype=jnp.int32)
    # Pre-build the gamma_faces_builder function for BC handling (density/Er)
    if bc_gamma is not None and hasattr(bc_gamma, "right_type"):
        def gamma_faces_builder(Gamma):
            lv, lg = left_constraints_from_bc_model(
                bc_gamma,
                Gamma[:, 0],
                profile=Gamma,
                face_centers=field.r_grid_half,
            )
            rv, rg = right_constraints_from_bc_model(
                bc_gamma,
                Gamma[:, -1],
                profile=Gamma,
                face_centers=field.r_grid_half,
            )
            if rv is not None:
                return jax.vmap(
                    lambda G, left_val, left_grad, right_val: make_profile_cell_variable(
                        G,
                        field.r_grid_half,
                        left_face_constraint=left_val,
                        left_face_grad_constraint=left_grad,
                        right_face_constraint=right_val,
                    ).face_value(reconstruction=reconstruction)
                )(Gamma, lv, lg, jnp.asarray(rv))
            else:
                return jax.vmap(
                    lambda G, left_val, left_grad, right_grad: make_profile_cell_variable(
                        G,
                        field.r_grid_half,
                        left_face_constraint=left_val,
                        left_face_grad_constraint=left_grad,
                        right_face_grad_constraint=right_grad,
                    ).face_value(reconstruction=reconstruction)
                )(Gamma, lv, lg, jnp.asarray(rg))
    elif bc_gamma is not None and hasattr(bc_gamma, "apply_ghost"):
        def gamma_faces_builder(Gamma):
            if hasattr(bc_gamma, "apply_ghost_all"):
                Gamma_ghost = bc_gamma.apply_ghost_all(Gamma)
            else:
                Gamma_ghost = jax.vmap(lambda G: bc_gamma.apply_ghost(G))(Gamma)
            return jax.vmap(faces_from_cell_centered)(Gamma_ghost)
    else:
        def gamma_faces_builder(Gamma):
            return jax.vmap(
                lambda G: make_profile_cell_variable(
                    G,
                    field.r_grid_half,
                    left_face_grad_constraint=jnp.asarray(0.0, dtype=G.dtype),
                    right_face_constraint=(
                        1.5 * G[-1] - 0.5 * G[-2]
                        if G.shape[0] >= 2
                        else G[-1]
                    ),
                ).face_value(reconstruction=reconstruction)
            )(Gamma)
    # Pre-build the diffusive Er face-flux builder for BC handling.
    if bc_er is not None and hasattr(bc_er, "right_type"):
        def er_diffusive_flux_builder(er_profile):
            lv_er, lg_er = left_constraints_from_bc_model(
                bc_er,
                er_profile[0],
                profile=er_profile,
                face_centers=field.r_grid_half,
            )
            rv_er, rg_er = right_constraints_from_bc_model(
                bc_er,
                er_profile[-1],
                profile=er_profile,
                face_centers=field.r_grid_half,
            )
            if rv_er is not None:
                er_cell_var = make_profile_cell_variable(
                    er_profile,
                    field.r_grid_half,
                    left_face_constraint=None if lv_er is None else jnp.asarray(lv_er).reshape(-1)[0],
                    left_face_grad_constraint=None if lg_er is None else jnp.asarray(lg_er).reshape(-1)[0],
                    right_face_constraint=jnp.asarray(rv_er).reshape(-1)[0],
                )
            else:
                er_cell_var = make_profile_cell_variable(
                er_profile,
                field.r_grid_half,
                    left_face_constraint=None if lv_er is None else jnp.asarray(lv_er).reshape(-1)[0],
                    left_face_grad_constraint=None if lg_er is None else jnp.asarray(lg_er).reshape(-1)[0],
                    right_face_grad_constraint=jnp.asarray(rg_er).reshape(-1)[0],
                )
            return -er_cell_var.face_grad()
    elif bc_er is not None and hasattr(bc_er, "apply_ghost"):
        def er_diffusive_flux_builder(er_profile):
            er_ghost = bc_er.apply_ghost(er_profile)
            return -jnp.diff(er_ghost) / jnp.diff(field.r_grid_half)
    else:
        def er_diffusive_flux_builder(er_profile):
            er_cell_var = make_profile_cell_variable(
                er_profile,
                field.r_grid_half,
                left_face_grad_constraint=jnp.asarray(0.0, dtype=er_profile.dtype),
                right_face_constraint=(
                    1.5 * er_profile[-1] - 0.5 * er_profile[-2]
                    if er_profile.shape[0] >= 2
                    else er_profile[-1]
                ),
            )
            return -er_cell_var.face_grad()
    return ElectricFieldEquation(
        dr_cells=dr_cells,
        Vprime=Vprime,
        Vprime_half=Vprime_half,
        flux_model=flux_model,
        species_mass=species_mass,
        charge_qp=charge_qp,
        permitivity_prefactor=permitivity_prefactor,
        gamma_faces_builder=gamma_faces_builder,
        er_diffusive_flux_builder=er_diffusive_flux_builder,
        er_bc_model=bc_er,
        source_mode=str(source_mode).strip().lower(),
        permitivity_mode=str(permitivity_mode).strip().lower(),
        Er_relax=Er_relax,
        DEr=DEr,
        boundary_mode=str(boundary_mode).strip().lower(),
        ntss_B0_mid=ntss_B0_mid,
        ntss_psfactor_mid=ntss_psfactor_mid,
        ntss_density_indices=ntss_density_indices,
    )


def _resolve_er_boundary_mode(config, solver_cfg):
    er_right_cfg = config.get("boundary", {}).get("Er", {}).get("right", {})
    if isinstance(er_right_cfg, dict):
        right_type = er_right_cfg.get("type")
        if str(right_type).strip().lower() in {"floating_ambipolar_edge", "ambipolar_edge_root"}:
            return str(right_type).strip().lower()
    return str(solver_cfg.get("Er_right_boundary_mode", solver_cfg.get("Er_boundary_mode", "standard"))).strip().lower()


# --- Equation System Builder (torax-style) ---
def build_equation_system(
    config,
    species,
    field,
    flux_model,
    source_models=None,
    solver_cfg=None,
    boundary_models=None,
):
    """
    Build the list of equation instances to evolve using prebuilt runtime
    objects. This avoids rebuilding geometry, databases, and flux models inside
    the equation builder and keeps compile closures smaller.
    """
    equations_cfg = config.get("equations", {})
    eqn_flags = {
        "density": equations_cfg.get("toggle_density", [True]*getattr(species, 'number_species', 3)),
        "temperature": equations_cfg.get("toggle_temperature", [True]*getattr(species, 'number_species', 3)),
        "Er": equations_cfg.get("toggle_Er", True),
    }
    equations_to_evolve = []
    species_mass = getattr(species, "mass", None)
    charge_qp = getattr(species, "charge_qp", None)
    solver_cfg = {} if solver_cfg is None else solver_cfg
    boundary_models = {} if boundary_models is None else boundary_models
    source_models = {} if source_models is None else source_models
    bc_density = boundary_models.get("density")
    bc_temperature = boundary_models.get("temperature")
    bc_gamma = boundary_models.get("gamma")
    bc_er = boundary_models.get("Er")
    density_source_model = source_models.get("density")
    temperature_source_model = source_models.get("temperature")
    Er_relax = solver_cfg.get("Er_relax", 1.0)
    DEr = solver_cfg.get("DEr", 1.0)
    Er_source_mode = solver_cfg.get(
        "Er_source_mode",
        config.get("ambipolarity", {}).get("er_ambipolar_flux_mode", "ambipolar_local"),
    )
    Er_permitivity_mode = solver_cfg.get(
        "Er_permittivity_mode",
        solver_cfg.get("Er_permitivity_mode", "neopax_local"),
    )
    Er_boundary_mode = _resolve_er_boundary_mode(config, solver_cfg)
    density_flux_reconstruction = solver_cfg.get("density_flux_reconstruction", "closure_face_flux")
    density_particle_face_closure_mode = solver_cfg.get("density_particle_face_closure_mode", "reconstructed")
    include_neo_convection = solver_cfg.get("temperature_include_neo_convection", True)
    include_turbulent_convection = solver_cfg.get("temperature_include_turbulent_convection", True)
    include_classical_convection = solver_cfg.get("temperature_include_classical_convection", True)
    include_work_term = solver_cfg.get(
        "temperature_include_work_term",
        solver_cfg.get("temperature_include_work_source_term", True),
    )
    convection_reconstruction = solver_cfg.get("temperature_convection_reconstruction", "closure_face_flux")
    heat_flux_reconstruction = solver_cfg.get("temperature_heat_flux_reconstruction", "closure_face_flux")
    density_floor = solver_cfg.get("density_floor", DEFAULT_TRANSPORT_DENSITY_FLOOR)
    temperature_floor = solver_cfg.get("temperature_floor", DEFAULT_TRANSPORT_TEMPERATURE_FLOOR)

    if any(eqn_flags["density"]):
        equations_to_evolve.append(build_density_equation(
            field,
            flux_model,
            density_source_model,
            bc_density,
            species,
            bc_temperature=bc_temperature,
            bc_er=bc_er,
            active_species_mask=eqn_flags["density"],
            particle_flux_reconstruction=density_flux_reconstruction,
            particle_face_closure_mode=density_particle_face_closure_mode,
            density_floor=density_floor,
            temperature_floor=temperature_floor,
        ))
    if any(eqn_flags["temperature"]):
        equations_to_evolve.append(build_temperature_equation(
            field,
            flux_model,
            temperature_source_model,
            species,
            bc_temperature,
            bc_density=bc_density,
            bc_gamma=bc_gamma,
            bc_er=bc_er,
            active_species_mask=eqn_flags["temperature"],
            charge_qp=charge_qp,
            include_neo_convection=include_neo_convection,
            include_turbulent_convection=include_turbulent_convection,
            include_classical_convection=include_classical_convection,
            include_work_term=include_work_term,
            convection_reconstruction=convection_reconstruction,
            heat_flux_reconstruction=heat_flux_reconstruction,
            density_floor=density_floor,
            temperature_floor=temperature_floor,
        ))
    if eqn_flags["Er"]:
        equations_to_evolve.append(build_electric_field_equation(
            field,
            getattr(species, "names", ()),
            flux_model,
            species_mass,
            charge_qp,
            bc_gamma,
            bc_er,
            Er_relax=Er_relax,
            DEr=DEr,
            source_mode=Er_source_mode,
            permitivity_mode=Er_permitivity_mode,
            boundary_mode=Er_boundary_mode,
        ))
    return equations_to_evolve


def build_equation_system_from_config(config, species):
    """
    Backward-compatible wrapper that builds the required runtime objects from
    config before delegating to ``build_equation_system``.
    """
    from ._boundary_conditions import build_boundary_condition_model
    from ._database import Monoenergetic
    from ._energy_grid_models import get_energy_grid_model
    from ._geometry_models import get_geometry_model
    from ._source_models import build_source_models_from_config
    from ._transport_flux_models import ZeroTransportModel, build_transport_flux_model, get_transport_flux_model

    geom_cfg = config.get("geometry", {})
    n_radial = int(geom_cfg.get("n_radial", 51))
    rho_edge = float(geom_cfg.get("rho_edge", 1.0))
    vmec_file = geom_cfg.get("vmec_file")
    boozer_file = geom_cfg.get("boozer_file")
    field = None
    if vmec_file is not None and boozer_file is not None:
        field = get_geometry_model("vmec_booz", n_r=n_radial, vmec=vmec_file, booz=boozer_file, rho_edge=rho_edge)

    energy_grid_cfg = config.get("energy_grid", {})
    n_x = int(energy_grid_cfg.get("n_x", 4))
    energy_grid = get_energy_grid_model("standard_laguerre", n_x=n_x, n_order=3)
    neoclassical_cfg = config.get("neoclassical", {})
    database = None
    neoclassical_file = neoclassical_cfg.get("neoclassical_file")
    if neoclassical_file and field is not None:
        database = Monoenergetic.read_ntx(field.a_b, neoclassical_file)

    neoclassical_factory = get_transport_flux_model(neoclassical_cfg.get("flux_model", "ntx_database"))
    turbulence_factory = get_transport_flux_model(config.get("turbulence", {}).get("flux_model", "none"))
    classical_factory = get_transport_flux_model(config.get("classical", {}).get("flux_model", "none")) if "classical" in config else None
    neoclassical_model = neoclassical_factory(species, energy_grid, field, database)
    turbulence_model = turbulence_factory(species, energy_grid, field, database) if turbulence_factory is not None else ZeroTransportModel()
    classical_model = classical_factory(species, energy_grid, field, database) if classical_factory is not None else ZeroTransportModel()
    flux_model = build_transport_flux_model(neoclassical_model, turbulence_model, classical_model)

    boundary_cfg = dict(config.get("boundary", {}))
    er_cfg = boundary_cfg.get("Er")
    if isinstance(er_cfg, dict):
        er_cfg = dict(er_cfg)
        right_cfg = er_cfg.get("right")
        if isinstance(right_cfg, dict):
            right_cfg = dict(right_cfg)
            right_type = str(right_cfg.get("type", "")).strip().lower()
            if right_type in {"floating_ambipolar_edge", "ambipolar_edge_root"}:
                right_cfg["type"] = "neumann"
                right_cfg.setdefault("gradient", 0.0)
            er_cfg["right"] = right_cfg
        boundary_cfg["Er"] = er_cfg
    dr = getattr(field, "dr", 1.0)
    boundary_models = {
        key: build_boundary_condition_model(
            boundary_cfg[key],
            dr,
            species_names=species.names if key in {"density", "temperature", "gamma"} else None,
        )
        for key in ("density", "temperature", "Er", "gamma")
        if key in boundary_cfg
    }
    solver_cfg = config.get("transport_solver", {})
    if not solver_cfg:
        solver_cfg = config.get("solver", config.get("transport", {}))
    source_models = build_source_models_from_config(config, species)

    return build_equation_system(
        config=config,
        species=species,
        field=field,
        flux_model=flux_model,
        source_models=source_models,
        solver_cfg=solver_cfg,
        boundary_models=boundary_models,
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class ComposedEquationSystem:
    equations: tuple
    density_equation: object | None = None
    temperature_equation: object | None = None
    er_equation: object | None = None
    species: object | None = None
    shared_flux_model: object | None = None
    density_floor: object = DEFAULT_TRANSPORT_DENSITY_FLOOR
    temperature_floor: object = DEFAULT_TRANSPORT_TEMPERATURE_FLOOR
    temperature_active_mask: object | None = None
    fixed_temperature_profile: object | None = None
    er_bc_model: object | None = None
    config: object | None = None
    source_models: object | None = None
    solver_cfg: object | None = None
    boundary_models: object | None = None
    debug_nonfinite_rhs_components: bool = False

    @staticmethod
    def _split_realtime_geometry_payload(payload):
        if isinstance(payload, dict) and "ntx_support" in payload and "geometry" in payload:
            return payload["ntx_support"], payload["geometry"]
        return payload, None

    @staticmethod
    def _realtime_geometry_payload_bar(payload, ntx_support_bar, geometry_bar):
        if isinstance(payload, dict) and "ntx_support" in payload and "geometry" in payload:
            return {
                "ntx_support": _sanitize_float_delta_bar_tree(payload["ntx_support"], ntx_support_bar),
                "geometry": _sanitize_float_delta_bar_tree(payload["geometry"], geometry_bar),
            }
        return _sanitize_float_delta_bar_tree(payload, ntx_support_bar)

    @staticmethod
    def _flux_model_with_geometry_payload(model, geometry):
        if model is None or not dataclasses.is_dataclass(model):
            return model
        updates = {}
        for field in dataclasses.fields(model):
            value = getattr(model, field.name)
            if field.name in {"geometry", "field"}:
                if value is not geometry:
                    updates[field.name] = geometry
                continue
            if dataclasses.is_dataclass(value):
                new_value = ComposedEquationSystem._flux_model_with_geometry_payload(value, geometry)
                if new_value is not value:
                    updates[field.name] = new_value
        if not updates:
            return model
        return dataclasses.replace(model, **updates)

    @staticmethod
    def _flux_model_geometry(model):
        if model is None or not dataclasses.is_dataclass(model):
            return None
        for name in ("geometry", "field"):
            if hasattr(model, name):
                value = getattr(model, name)
                if hasattr(value, "r_grid_half"):
                    return value
        for field in dataclasses.fields(model):
            value = getattr(model, field.name)
            if dataclasses.is_dataclass(value):
                geometry = ComposedEquationSystem._flux_model_geometry(value)
                if geometry is not None:
                    return geometry
        return None

    @staticmethod
    def _direct_geometry_build_lagged_response_bar(model, geometry, state, response_bar):
        """Geometry cotangent for non-NTX lagged-response rebuilds.

        The NTX exact model exposes its realtime geometry through an explicit
        support payload and is handled by the ntx_support branch.  This helper
        only covers transport submodels that store the transport geometry
        directly as ``field``/``geometry`` and build a lagged response from it
        (for example the turbulent power-over-n model).
        """
        if model is None or response_bar is None:
            return _float_delta_tree_like(geometry)

        if (
            dataclasses.is_dataclass(model)
            and hasattr(model, "neoclassical_model")
            and hasattr(model, "turbulent_model")
            and hasattr(model, "classical_model")
        ):
            bars = []
            for model_name, response_name in (
                ("turbulent_model", "turbulent_response"),
                ("classical_model", "classical_response"),
            ):
                submodel = getattr(model, model_name, None)
                subresponse_bar = getattr(response_bar, response_name, None)
                bars.append(
                    ComposedEquationSystem._direct_geometry_build_lagged_response_bar(
                        submodel,
                        geometry,
                        state,
                        subresponse_bar,
                    )
                )
            return jax.tree_util.tree_map(lambda *values: sum(values), *bars)

        if not dataclasses.is_dataclass(model) or not callable(getattr(model, "build_lagged_response", None)):
            return _float_delta_tree_like(geometry)

        field_names = {field.name for field in dataclasses.fields(model)}
        geometry_field_names = tuple(name for name in ("geometry", "field") if name in field_names)
        if not geometry_field_names:
            return _float_delta_tree_like(geometry)

        geometry_delta0 = _float_delta_tree_like(geometry)

        def _response_from_geometry_delta(geometry_delta):
            geometry_value = _add_float_delta_tree(geometry, geometry_delta)
            updates = {name: geometry_value for name in geometry_field_names}
            return dataclasses.replace(model, **updates).build_lagged_response(state)

        _, geometry_pullback = jax.vjp(_response_from_geometry_delta, geometry_delta0)
        (geometry_bar,) = geometry_pullback(response_bar)
        return _sanitize_float_delta_bar_tree(geometry, geometry_bar)

    def with_geometry_payload(self, geometry):
        if self.config is None or self.solver_cfg is None or self.boundary_models is None:
            raise ValueError("Geometry-payload pullback requires equation-system construction metadata.")
        flux_model_source = self.shared_flux_model
        if flux_model_source is None:
            flux_model_source = next(
                (
                    getattr(eq, "flux_model", None)
                    for eq in self.equations
                    if getattr(eq, "flux_model", None) is not None
                ),
                None,
            )
        flux_model = self._flux_model_with_geometry_payload(flux_model_source, geometry)
        equations = build_equation_system(
            config=self.config,
            species=self.species,
            field=geometry,
            flux_model=flux_model,
            source_models=self.source_models,
            solver_cfg=self.solver_cfg,
            boundary_models=self.boundary_models,
        )
        return dataclasses.replace(
            self,
            equations=tuple(equations),
            density_equation=next((eq for eq in equations if getattr(eq, "name", None) == "density"), None),
            temperature_equation=next((eq for eq in equations if getattr(eq, "name", None) == "temperature"), None),
            er_equation=next((eq for eq in equations if getattr(eq, "name", None) == "Er"), None),
            shared_flux_model=flux_model if len(equations) >= 1 else None,
        )

    def _prepare_working_state(self, state):
        working_state = state
        eidx = None
        if self.species is not None and hasattr(self.species, "names") and "e" in tuple(getattr(self.species, "names", ())):
            try:
                working_state = enforce_quasi_neutrality(state, self.species)
                eidx = int(tuple(self.species.names).index("e"))
            except Exception:
                working_state = state
        working_state = apply_transport_density_floor(working_state, self.density_floor)
        working_state = apply_er_dirichlet_boundary_state(working_state, self.er_bc_model)
        working_state = project_fixed_temperature_species(
            working_state,
            self.temperature_active_mask,
            self.fixed_temperature_profile,
            density_floor=self.density_floor,
        )
        working_state = apply_transport_temperature_floor(
            working_state,
            self.temperature_floor,
            self.density_floor,
        )
        return working_state, eidx

    def _resolve_equations(self):
        density_eq = self.density_equation
        temperature_eq = self.temperature_equation
        er_eq = self.er_equation
        if density_eq is None:
            density_eq = next((eq for eq in self.equations if getattr(eq, "name", None) == "density"), None)
        if temperature_eq is None:
            temperature_eq = next((eq for eq in self.equations if getattr(eq, "name", None) == "temperature"), None)
        if er_eq is None:
            er_eq = next((eq for eq in self.equations if getattr(eq, "name", None) == "Er"), None)
        return density_eq, temperature_eq, er_eq

    def _shared_flux_bc_kwargs(self):
        density_eq, temperature_eq, er_eq = self._resolve_equations()
        return {
            "bc_density": getattr(density_eq, "density_bc_model", None),
            "bc_temperature": getattr(temperature_eq, "temperature_bc_model", None),
            "bc_er": getattr(er_eq, "er_bc_model", self.er_bc_model),
        }

    def _shared_flux_call_kwargs(self, extra_kwargs=None):
        call_kwargs = dict(self._shared_flux_bc_kwargs())
        if extra_kwargs:
            call_kwargs.update(extra_kwargs)
        return call_kwargs

    @staticmethod
    def _shared_fluxes_zero_like(shared_fluxes):
        return jax.tree_util.tree_map(jnp.zeros_like, shared_fluxes)

    @staticmethod
    def _shared_fluxes_add(lhs, rhs):
        def _add_leaf(a, b):
            a_arr = jnp.asarray(a)
            b_arr = jnp.asarray(b)
            if a_arr.dtype == jax.dtypes.float0:
                if b_arr.dtype == jax.dtypes.float0:
                    return jnp.zeros(b_arr.shape, dtype=jnp.float64)
                return b_arr
            if b_arr.dtype == jax.dtypes.float0:
                return a_arr
            return a_arr + b_arr

        return jax.tree_util.tree_map(_add_leaf, lhs, rhs)

    def _debug_nonfinite_rhs_components(
        self,
        working_state,
        shared_fluxes,
        density_rhs,
        pressure_rhs,
        Er_rhs,
    ):
        if not self.debug_nonfinite_rhs_components:
            return

        rhs_finite = jnp.logical_and(
            jnp.all(jnp.isfinite(density_rhs)),
            jnp.logical_and(
                jnp.all(jnp.isfinite(pressure_rhs)),
                jnp.all(jnp.isfinite(Er_rhs)),
            ),
        )

        def _print_array_stats(label, value):
            value = jnp.asarray(value)
            finite_mask = jnp.isfinite(value)
            flat_bad = jnp.ravel(jnp.logical_not(finite_mask))
            first_bad = jnp.argmax(flat_bad)
            flat_value = jnp.ravel(value)
            first_bad_value = flat_value[jnp.minimum(first_bad, flat_value.shape[0] - 1)]
            jax.debug.print(
                f"[nonfinite-rhs] {label}: finite={{finite}} min={{min:.6e}} max={{max:.6e}} "
                f"first_bad_flat={{first_bad}} first_bad_value={{first_bad_value:.6e}}",
                finite=jnp.all(finite_mask),
                min=jnp.nanmin(value),
                max=jnp.nanmax(value),
                first_bad=first_bad,
                first_bad_value=first_bad_value,
            )

        def _print_edge_stats(label, value):
            value = jnp.asarray(value)
            if value.ndim < 1:
                return
            _print_array_stats(f"{label}.left_edge", value[..., 0])
            _print_array_stats(f"{label}.right_edge", value[..., -1])
            if int(value.shape[-1]) > 1:
                _print_array_stats(f"{label}.left_edge_1", value[..., 1])
                _print_array_stats(f"{label}.right_edge_1", value[..., -2])

        def _print_center_from_faces_stats(fluxes, key):
            center_key = key
            face_key = f"{key}_faces"
            if _flux_has_key(fluxes, center_key) or not _flux_has_key(fluxes, face_key):
                return
            _print_array_stats(
                f"flux.{key}_center_from_faces",
                jax.vmap(cell_centered_from_faces)(fluxes[face_key]),
            )

        diagnostic_geometry = self._flux_model_geometry(self.shared_flux_model)
        if diagnostic_geometry is None:
            for equation in self.equations:
                diagnostic_geometry = self._flux_model_geometry(getattr(equation, "flux_model", None))
                if diagnostic_geometry is not None:
                    break
        evaluated_state = None
        if diagnostic_geometry is not None:
            evaluated_state = build_evaluated_transport_state(
                working_state,
                diagnostic_geometry,
                **self._shared_flux_bc_kwargs(),
                density_floor=self.density_floor,
                temperature_floor=self.temperature_floor,
            )

        def _print(_):
            _print_array_stats("state.density", working_state.density)
            _print_array_stats("state.temperature", working_state.temperature)
            _print_array_stats("state.pressure", working_state.pressure)
            _print_array_stats("state.Er", working_state.Er)
            _print_edge_stats("state.density", working_state.density)
            _print_edge_stats("state.temperature", working_state.temperature)
            _print_edge_stats("state.pressure", working_state.pressure)
            _print_edge_stats("state.Er", working_state.Er)
            if evaluated_state is not None:
                _print_array_stats("evaluated.center.density", evaluated_state.center.density)
                _print_array_stats("evaluated.center.temperature", evaluated_state.center.temperature)
                _print_array_stats("evaluated.center.pressure", evaluated_state.center.pressure)
                _print_array_stats("evaluated.center.Er", evaluated_state.center.Er)
                _print_array_stats("evaluated.face.density", evaluated_state.face.density)
                _print_array_stats("evaluated.face.temperature", evaluated_state.face.temperature)
                _print_array_stats("evaluated.face.pressure", evaluated_state.face.pressure)
                _print_array_stats("evaluated.face.Er", evaluated_state.face.Er)
                _print_array_stats("evaluated.grad_center.density", evaluated_state.density_grad_center)
                _print_array_stats("evaluated.grad_center.temperature", evaluated_state.temperature_grad_center)
                _print_array_stats("evaluated.grad_center.Er", evaluated_state.Er_grad_center)
                _print_array_stats("evaluated.grad_face.density", evaluated_state.density_grad_face)
                _print_array_stats("evaluated.grad_face.temperature", evaluated_state.temperature_grad_face)
                _print_array_stats("evaluated.grad_face.Er", evaluated_state.Er_grad_face)
                _print_edge_stats("evaluated.face.density", evaluated_state.face.density)
                _print_edge_stats("evaluated.face.temperature", evaluated_state.face.temperature)
                _print_edge_stats("evaluated.face.pressure", evaluated_state.face.pressure)
                _print_edge_stats("evaluated.face.Er", evaluated_state.face.Er)
                _print_edge_stats("evaluated.grad_face.density", evaluated_state.density_grad_face)
                _print_edge_stats("evaluated.grad_face.temperature", evaluated_state.temperature_grad_face)
                _print_edge_stats("evaluated.grad_face.Er", evaluated_state.Er_grad_face)
            _print_array_stats("rhs.density", density_rhs)
            _print_array_stats("rhs.pressure", pressure_rhs)
            _print_array_stats("rhs.Er", Er_rhs)
            _print_edge_stats("rhs.density", density_rhs)
            _print_edge_stats("rhs.pressure", pressure_rhs)
            _print_edge_stats("rhs.Er", Er_rhs)
            if isinstance(shared_fluxes, dict):
                for key in sorted(shared_fluxes):
                    value = shared_fluxes.get(key)
                    if value is not None:
                        _print_array_stats(f"flux.{key}", value)
                        _print_edge_stats(f"flux.{key}", value)
                for key in ("Gamma", "Q", "Upar", "Gamma_neo", "Q_neo", "Upar_neo"):
                    _print_center_from_faces_stats(shared_fluxes, key)
            return jnp.asarray(0, dtype=jnp.int32)

        def _skip(_):
            return jnp.asarray(0, dtype=jnp.int32)

        jax.lax.cond(jnp.logical_not(rhs_finite), _print, _skip, operand=None)

    def _prepare_working_state_pullback(self, state, working_state_bar):
        def _prepared_state(state_value):
            return self._prepare_working_state(state_value)[0]

        _, prepare_pullback = jax.vjp(_prepared_state, state)
        (state_bar,) = prepare_pullback(working_state_bar)
        return state_bar

    def build_lagged_response(self, state):
        working_state, eidx = self._prepare_working_state(state)
        if lagged_timing_enabled():
            jax.debug.callback(lambda: lagged_timing_start("equations.build_lagged_response"), ordered=True)
        flux_response = None
        if self.shared_flux_model is not None:
            flux_response = self.shared_flux_model.build_lagged_response(
                working_state,
                **self._shared_flux_bc_kwargs(),
            )
        if lagged_timing_enabled():
            jax.debug.callback(lambda: lagged_timing_end("equations.build_lagged_response"), ordered=True)
        return TransportLaggedResponse(
            flux_response=flux_response,
        )

    def pullback_build_lagged_response(self, state, lagged_response_bar, **kwargs):
        working_state, eidx = self._prepare_working_state(state)
        flux_response_bar = None if lagged_response_bar is None else lagged_response_bar.flux_response
        if self.shared_flux_model is None or flux_response_bar is None:
            working_state_bar = jax.tree_util.tree_map(jnp.zeros_like, working_state)
        else:
            pullback_fn = getattr(self.shared_flux_model, "pullback_build_lagged_response", None)
            if callable(pullback_fn):
                working_state_bar = pullback_fn(
                    working_state,
                    flux_response_bar,
                    **self._shared_flux_call_kwargs(kwargs),
                )
            else:
                _, flux_pullback = jax.vjp(
                    lambda working_state_value: self.shared_flux_model.build_lagged_response(
                        working_state_value,
                        **self._shared_flux_call_kwargs(kwargs),
                    ),
                    working_state,
                )
                (working_state_bar,) = flux_pullback(flux_response_bar)
        return self._prepare_working_state_pullback(state, working_state_bar)

    def pullback_build_lagged_response_support_payload(self, state, lagged_response_bar, support, **kwargs):
        support, geometry = self._split_realtime_geometry_payload(support)
        if geometry is not None:
            support_bar = self.pullback_build_lagged_response_support_payload(
                state,
                lagged_response_bar,
                support,
                **kwargs,
            )
            working_state, _eidx = self._prepare_working_state(state)
            flux_response_bar = None if lagged_response_bar is None else lagged_response_bar.flux_response
            geometry_bar = self._direct_geometry_build_lagged_response_bar(
                self.shared_flux_model,
                geometry,
                working_state,
                flux_response_bar,
            )
            return self._realtime_geometry_payload_bar(
                {"ntx_support": support, "geometry": geometry},
                support_bar,
                geometry_bar,
            )
        working_state, _eidx = self._prepare_working_state(state)
        flux_response_bar = None if lagged_response_bar is None else lagged_response_bar.flux_response
        if self.shared_flux_model is None or flux_response_bar is None:
            return jax.tree_util.tree_map(jnp.zeros_like, support)
        pullback_fn = getattr(self.shared_flux_model, "pullback_build_lagged_response_support_payload", None)
        if callable(pullback_fn):
            return _sanitize_float_delta_bar_tree(
                support,
                pullback_fn(
                    working_state,
                    flux_response_bar,
                    support,
                    **self._shared_flux_call_kwargs(kwargs),
                ),
            )
        support_delta0 = _float_delta_tree_like(support)
        _, support_delta_pullback = jax.vjp(
            lambda support_delta: self.shared_flux_model.with_support_payload(
                _add_float_delta_tree(support, support_delta)
            ).build_lagged_response(
                working_state,
                **self._shared_flux_call_kwargs(kwargs),
            ),
            support_delta0,
        )
        (support_bar,) = support_delta_pullback(flux_response_bar)
        return _sanitize_float_delta_bar_tree(support, support_bar)

    def pullback_build_lagged_response_support_payload_batched_interpolated_faces(
        self,
        state,
        lagged_response_bars,
        support,
        **kwargs,
    ):
        """Batched counterpart for the exact NTX interpolated-face support lane.

        The geometry branch remains a device VJP over its already-batched
        cotangent leaves. The NTX branch is delegated to the dedicated local
        multi-RHS implementation rather than mapped through the scalar rule.
        """
        support, geometry = self._split_realtime_geometry_payload(support)
        if geometry is not None:
            ntx_support_bar = self.pullback_build_lagged_response_support_payload_batched_interpolated_faces(
                state,
                lagged_response_bars,
                support,
                **kwargs,
            )
            working_state, _eidx = self._prepare_working_state(state)
            flux_response_bars = (
                None if lagged_response_bars is None else lagged_response_bars.flux_response
            )
            if flux_response_bars is None:
                geometry_bar = jax.tree_util.tree_map(
                    lambda leaf: jnp.broadcast_to(
                        jnp.zeros_like(jnp.asarray(leaf)),
                        (0,) + jnp.asarray(leaf).shape,
                    ),
                    geometry,
                )
            else:
                geometry_bar = jax.vmap(
                    lambda flux_response_bar: self._direct_geometry_build_lagged_response_bar(
                        self.shared_flux_model,
                        geometry,
                        working_state,
                        flux_response_bar,
                    )
                )(flux_response_bars)
            # Both bars already carry the leading objective axis. The scalar
            # payload sanitizer deliberately restores primal-shaped non-float
            # leaves, which would discard that axis here.
            return {"ntx_support": ntx_support_bar, "geometry": geometry_bar}

        working_state, _eidx = self._prepare_working_state(state)
        flux_response_bars = None if lagged_response_bars is None else lagged_response_bars.flux_response
        if self.shared_flux_model is None or flux_response_bars is None:
            raise NotImplementedError(
                "batched interpolated-face support pullback requires an active shared flux response."
            )
        pullback_fn = getattr(
            self.shared_flux_model,
            "pullback_build_lagged_response_support_payload_batched_interpolated_faces",
            None,
        )
        if not callable(pullback_fn):
            raise NotImplementedError(
                "The active shared flux model does not expose the batched interpolated-face "
                "support pullback."
            )
        return pullback_fn(
            working_state,
            flux_response_bars,
            support,
            **self._shared_flux_call_kwargs(kwargs),
        )

    def pullback_build_lagged_response_state_and_support_payload_batched_interpolated_faces(
        self,
        state,
        lagged_response_bars,
        support,
        **kwargs,
    ):
        """Joint objective-batched state/support transpose for the NTX face lane.

        This is intentionally separate from the established scalar and
        support-only APIs.  The model returns both cotangent trees from one
        local NTX implicit-adjoint construction; this wrapper only translates
        the working-state and realtime-geometry payload boundaries.
        """
        support, geometry = self._split_realtime_geometry_payload(support)
        working_state, _eidx = self._prepare_working_state(state)
        flux_response_bars = None if lagged_response_bars is None else lagged_response_bars.flux_response
        if self.shared_flux_model is None or flux_response_bars is None:
            raise NotImplementedError(
                "batched joint interpolated-face pullback requires an active shared flux response."
            )
        pullback_fn = getattr(
            self.shared_flux_model,
            "pullback_build_lagged_response_state_and_support_payload_batched_interpolated_faces",
            None,
        )
        if not callable(pullback_fn):
            raise NotImplementedError(
                "The active shared flux model does not expose the batched joint "
                "interpolated-face state/support pullback."
            )
        working_state_bars, ntx_support_bar = pullback_fn(
            working_state,
            flux_response_bars,
            support,
            **self._shared_flux_call_kwargs(kwargs),
        )
        state_bars = jax.vmap(
            lambda working_state_bar: self._prepare_working_state_pullback(state, working_state_bar)
        )(working_state_bars)
        if geometry is None:
            return state_bars, ntx_support_bar
        geometry_bar = jax.vmap(
            lambda flux_response_bar: self._direct_geometry_build_lagged_response_bar(
                self.shared_flux_model,
                geometry,
                working_state,
                flux_response_bar,
            )
        )(flux_response_bars)
        return state_bars, {"ntx_support": ntx_support_bar, "geometry": geometry_bar}

    def evaluate_with_lagged_response(self, t, state, runtime, lagged_response):
        del t, runtime
        return self._evaluate_state(state, lagged_response=lagged_response)

    def _evaluate_with_shared_fluxes_from_working_state(self, working_state, eidx, state_reference, shared_fluxes):
        from ._state import TransportState
        density_eq, temperature_eq, er_eq = self._resolve_equations()

        density_rhs = (
            density_eq(working_state, fluxes=shared_fluxes)
            if density_eq is not None
            else jnp.zeros_like(state_reference.density)
        )
        pressure_rhs = (
            temperature_eq(working_state, fluxes=shared_fluxes)
            if temperature_eq is not None
            else jnp.zeros_like(state_reference.pressure)
        )
        Er_rhs = (
            er_eq(working_state, fluxes=shared_fluxes)
            if er_eq is not None
            else jnp.zeros_like(state_reference.Er)
        )

        density_rhs = _expand_density_rhs_to_full_shape(density_rhs, state_reference.density, self.species)

        if eidx is not None:
            density_rhs = density_rhs.at[int(eidx), :].set(jnp.zeros_like(density_rhs[int(eidx), :]))

        if density_eq is not None and hasattr(density_eq, "enforce_dirichlet_boundary_rhs"):
            density_rhs = density_eq.enforce_dirichlet_boundary_rhs(working_state, density_rhs)

        if temperature_eq is not None and hasattr(temperature_eq, "enforce_dirichlet_boundary_rhs"):
            pressure_rhs = temperature_eq.enforce_dirichlet_boundary_rhs(working_state, density_rhs, pressure_rhs)

        if er_eq is not None and hasattr(er_eq, "enforce_dirichlet_boundary_rhs"):
            Er_rhs = er_eq.enforce_dirichlet_boundary_rhs(working_state, Er_rhs)

        self._debug_nonfinite_rhs_components(
            working_state,
            shared_fluxes,
            density_rhs,
            pressure_rhs,
            Er_rhs,
        )

        return TransportState(
            density=density_rhs,
            pressure=pressure_rhs,
            Er=Er_rhs,
        )

    def evaluate_with_shared_fluxes(self, t, state, runtime, shared_fluxes):
        del t, runtime
        working_state, eidx = self._prepare_working_state(state)
        return self._evaluate_with_shared_fluxes_from_working_state(
            working_state,
            eidx,
            state,
            shared_fluxes,
        )

    def pullback_shared_fluxes(self, state, shared_fluxes, rhs_bar):
        """Reverse-only pullback for the shared-flux -> RHS assembly.

        This keeps the primal map unchanged, but lets reverse split the shared
        transport flux assembly by equation instead of VJP-ing the whole
        composed RHS map as one giant object.
        """
        working_state, eidx = self._prepare_working_state(state)
        density_eq, temperature_eq, er_eq = self._resolve_equations()
        zero_flux_bar = self._shared_fluxes_zero_like(shared_fluxes)

        density_bar = rhs_bar.density
        pressure_bar = rhs_bar.pressure
        er_bar = rhs_bar.Er

        def _density_map(fluxes_value):
            density_rhs = (
                density_eq(working_state, fluxes=fluxes_value)
                if density_eq is not None
                else jnp.zeros_like(state.density)
            )
            density_rhs = _expand_density_rhs_to_full_shape(density_rhs, state.density, self.species)
            if eidx is not None:
                density_rhs = density_rhs.at[int(eidx), :].set(jnp.zeros_like(density_rhs[int(eidx), :]))
            if density_eq is not None and hasattr(density_eq, "enforce_dirichlet_boundary_rhs"):
                density_rhs = density_eq.enforce_dirichlet_boundary_rhs(working_state, density_rhs)
            return density_rhs

        def _pressure_map(fluxes_value):
            density_rhs = (
                density_eq(working_state, fluxes=fluxes_value)
                if density_eq is not None
                else jnp.zeros_like(state.density)
            )
            density_rhs = _expand_density_rhs_to_full_shape(density_rhs, state.density, self.species)
            if eidx is not None:
                density_rhs = density_rhs.at[int(eidx), :].set(jnp.zeros_like(density_rhs[int(eidx), :]))
            pressure_rhs = (
                temperature_eq(working_state, fluxes=fluxes_value)
                if temperature_eq is not None
                else jnp.zeros_like(state.pressure)
            )
            if temperature_eq is not None and hasattr(temperature_eq, "enforce_dirichlet_boundary_rhs"):
                pressure_rhs = temperature_eq.enforce_dirichlet_boundary_rhs(working_state, density_rhs, pressure_rhs)
            return pressure_rhs

        def _er_map(fluxes_value):
            er_rhs = (
                er_eq(working_state, fluxes=fluxes_value)
                if er_eq is not None
                else jnp.zeros_like(state.Er)
            )
            if er_eq is not None and hasattr(er_eq, "enforce_dirichlet_boundary_rhs"):
                er_rhs = er_eq.enforce_dirichlet_boundary_rhs(working_state, er_rhs)
            return er_rhs

        flux_bar = zero_flux_bar
        if density_eq is not None:
            _, density_pullback = jax.vjp(_density_map, shared_fluxes)
            (density_flux_bar,) = density_pullback(density_bar)
            flux_bar = self._shared_fluxes_add(flux_bar, density_flux_bar)
        if temperature_eq is not None:
            _, pressure_pullback = jax.vjp(_pressure_map, shared_fluxes)
            (pressure_flux_bar,) = pressure_pullback(pressure_bar)
            flux_bar = self._shared_fluxes_add(flux_bar, pressure_flux_bar)
        if er_eq is not None:
            _, er_pullback = jax.vjp(_er_map, shared_fluxes)
            (er_flux_bar,) = er_pullback(er_bar)
            flux_bar = self._shared_fluxes_add(flux_bar, er_flux_bar)
        return flux_bar

    def _pullback_shared_flux_rhs_state_and_fluxes(self, state, working_state, eidx, shared_fluxes, rhs_bar):
        """Joint assembly pullback with respect to working state and shared fluxes."""

        def _assembly_map(working_state_value, fluxes_value):
            return self._evaluate_with_shared_fluxes_from_working_state(
                working_state_value,
                eidx,
                state,
                fluxes_value,
            )

        _, assembly_pullback = jax.vjp(_assembly_map, working_state, shared_fluxes)
        return assembly_pullback(rhs_bar)

    def _pullback_shared_flux_rhs_state(self, state, working_state, eidx, shared_fluxes, rhs_bar):
        """Reverse-only split pullback for fixed-shared-flux RHS state dependence."""
        density_eq, temperature_eq, er_eq = self._resolve_equations()
        zero_working_state_bar = jax.tree_util.tree_map(jnp.zeros_like, working_state)

        density_bar = rhs_bar.density
        pressure_bar = rhs_bar.pressure
        er_bar = rhs_bar.Er

        def _add_state_bars(lhs, rhs):
            return jax.tree_util.tree_map(lambda a, b: a + b, lhs, rhs)

        def _density_map(working_state_value):
            density_rhs = (
                density_eq(working_state_value, fluxes=shared_fluxes)
                if density_eq is not None
                else jnp.zeros_like(state.density)
            )
            density_rhs = _expand_density_rhs_to_full_shape(density_rhs, state.density, self.species)
            if eidx is not None:
                density_rhs = density_rhs.at[int(eidx), :].set(jnp.zeros_like(density_rhs[int(eidx), :]))
            if density_eq is not None and hasattr(density_eq, "enforce_dirichlet_boundary_rhs"):
                density_rhs = density_eq.enforce_dirichlet_boundary_rhs(working_state_value, density_rhs)
            return density_rhs

        def _pressure_map(working_state_value):
            density_rhs = (
                density_eq(working_state_value, fluxes=shared_fluxes)
                if density_eq is not None
                else jnp.zeros_like(state.density)
            )
            density_rhs = _expand_density_rhs_to_full_shape(density_rhs, state.density, self.species)
            if eidx is not None:
                density_rhs = density_rhs.at[int(eidx), :].set(jnp.zeros_like(density_rhs[int(eidx), :]))
            pressure_rhs = (
                temperature_eq(working_state_value, fluxes=shared_fluxes)
                if temperature_eq is not None
                else jnp.zeros_like(state.pressure)
            )
            if temperature_eq is not None and hasattr(temperature_eq, "enforce_dirichlet_boundary_rhs"):
                pressure_rhs = temperature_eq.enforce_dirichlet_boundary_rhs(
                    working_state_value,
                    density_rhs,
                    pressure_rhs,
                )
            return pressure_rhs

        def _er_map(working_state_value):
            er_rhs = (
                er_eq(working_state_value, fluxes=shared_fluxes)
                if er_eq is not None
                else jnp.zeros_like(state.Er)
            )
            if er_eq is not None and hasattr(er_eq, "enforce_dirichlet_boundary_rhs"):
                er_rhs = er_eq.enforce_dirichlet_boundary_rhs(working_state_value, er_rhs)
            return er_rhs

        working_state_bar = zero_working_state_bar
        if density_eq is not None:
            _, density_pullback = jax.vjp(_density_map, working_state)
            (density_state_bar,) = density_pullback(density_bar)
            working_state_bar = _add_state_bars(working_state_bar, density_state_bar)
        if temperature_eq is not None:
            _, pressure_pullback = jax.vjp(_pressure_map, working_state)
            (pressure_state_bar,) = pressure_pullback(pressure_bar)
            working_state_bar = _add_state_bars(working_state_bar, pressure_state_bar)
        if er_eq is not None:
            _, er_pullback = jax.vjp(_er_map, working_state)
            (er_state_bar,) = er_pullback(er_bar)
            working_state_bar = _add_state_bars(working_state_bar, er_state_bar)
        return working_state_bar

    def _pullback_shared_flux_rhs_state_component(
        self,
        state,
        working_state,
        eidx,
        shared_fluxes,
        rhs_bar,
        *,
        component: str,
    ):
        """Diagnostic fixed-shared-flux state pullback for one equation block."""
        density_eq, temperature_eq, er_eq = self._resolve_equations()
        zero_working_state_bar = jax.tree_util.tree_map(jnp.zeros_like, working_state)
        component_name = str(component).strip().lower()

        def _er_direct_subterm_map(working_state_value, subterm_name):
            if er_eq is None:
                return jnp.zeros_like(state.Er)
            Er = working_state_value.Er
            plasma_permitivity = _plasma_permitivity_from_prefactor(
                working_state_value,
                er_eq.species_mass,
                er_eq.permitivity_prefactor,
            )
            Gamma = _get_center_flux(shared_fluxes, "Gamma")
            _, ambi_term = er_eq._charge_flux_and_ambi_term(
                working_state_value,
                Gamma,
                plasma_permitivity,
            )
            _, er_diffusion = er_eq._er_diffusion(Er)
            if subterm_name == "er_diffusion":
                er_rhs = er_eq.Er_relax * er_eq.DEr * er_diffusion
            elif subterm_name in {"er_ambipolar", "er_ambi_coeff"}:
                er_rhs = -er_eq.Er_relax * ambi_term
                if er_eq.boundary_mode == "floating_ambipolar_edge":
                    er_rhs = er_rhs.at[-1].set(
                        -er_eq.Er_relax
                        * er_eq._outer_face_ambi_term(
                            working_state_value,
                            Gamma,
                            plasma_permitivity,
                        )
                    )
            elif subterm_name == "er_ambi_charge_flux":
                # In this fixed-shared-flux diagnostic, charge_flux is held
                # fixed; any nonzero state pullback here would indicate we
                # accidentally differentiated through the flux model again.
                er_rhs = jnp.zeros_like(state.Er)
            else:
                raise ValueError(f"Unknown Er direct RHS subterm {subterm_name!r}.")
            if hasattr(er_eq, "enforce_dirichlet_boundary_rhs"):
                er_rhs = er_eq.enforce_dirichlet_boundary_rhs(working_state_value, er_rhs)
            return er_rhs

        def _density_map(working_state_value):
            density_rhs = (
                density_eq(working_state_value, fluxes=shared_fluxes)
                if density_eq is not None
                else jnp.zeros_like(state.density)
            )
            density_rhs = _expand_density_rhs_to_full_shape(density_rhs, state.density, self.species)
            if eidx is not None:
                density_rhs = density_rhs.at[int(eidx), :].set(jnp.zeros_like(density_rhs[int(eidx), :]))
            if density_eq is not None and hasattr(density_eq, "enforce_dirichlet_boundary_rhs"):
                density_rhs = density_eq.enforce_dirichlet_boundary_rhs(working_state_value, density_rhs)
            return density_rhs

        def _pressure_map(working_state_value):
            density_rhs = (
                density_eq(working_state_value, fluxes=shared_fluxes)
                if density_eq is not None
                else jnp.zeros_like(state.density)
            )
            density_rhs = _expand_density_rhs_to_full_shape(density_rhs, state.density, self.species)
            if eidx is not None:
                density_rhs = density_rhs.at[int(eidx), :].set(jnp.zeros_like(density_rhs[int(eidx), :]))
            pressure_rhs = (
                temperature_eq(working_state_value, fluxes=shared_fluxes)
                if temperature_eq is not None
                else jnp.zeros_like(state.pressure)
            )
            if temperature_eq is not None and hasattr(temperature_eq, "enforce_dirichlet_boundary_rhs"):
                pressure_rhs = temperature_eq.enforce_dirichlet_boundary_rhs(
                    working_state_value,
                    density_rhs,
                    pressure_rhs,
                )
            return pressure_rhs

        def _er_map(working_state_value):
            er_rhs = (
                er_eq(working_state_value, fluxes=shared_fluxes)
                if er_eq is not None
                else jnp.zeros_like(state.Er)
            )
            if er_eq is not None and hasattr(er_eq, "enforce_dirichlet_boundary_rhs"):
                er_rhs = er_eq.enforce_dirichlet_boundary_rhs(working_state_value, er_rhs)
            return er_rhs

        if component_name == "density":
            if density_eq is None:
                return zero_working_state_bar
            _, density_pullback = jax.vjp(_density_map, working_state)
            (density_state_bar,) = density_pullback(rhs_bar.density)
            return density_state_bar
        if component_name == "pressure":
            if temperature_eq is None:
                return zero_working_state_bar
            _, pressure_pullback = jax.vjp(_pressure_map, working_state)
            (pressure_state_bar,) = pressure_pullback(rhs_bar.pressure)
            return pressure_state_bar
        if component_name == "er":
            if er_eq is None:
                return zero_working_state_bar
            _, er_pullback = jax.vjp(_er_map, working_state)
            (er_state_bar,) = er_pullback(rhs_bar.Er)
            return er_state_bar
        if component_name in {"er_diffusion", "er_ambipolar", "er_ambi_coeff", "er_ambi_charge_flux"}:
            if er_eq is None:
                return zero_working_state_bar
            _, er_pullback = jax.vjp(
                lambda working_state_value: _er_direct_subterm_map(working_state_value, component_name),
                working_state,
            )
            (er_state_bar,) = er_pullback(rhs_bar.Er)
            return er_state_bar
        raise ValueError(f"Unknown direct RHS state component {component!r}.")

    def pullback_evaluate_with_lagged_response(self, t, state, runtime, lagged_response, rhs_bar):
        """Reverse-only pullback for lagged-response dependence of the RHS."""
        del t, runtime
        if self.shared_flux_model is None or lagged_response is None or lagged_response.flux_response is None:
            return TransportLaggedResponse(flux_response=None)

        working_state, eidx = self._prepare_working_state(state)
        shared_fluxes = self.shared_flux_model.evaluate_with_lagged_response(
            working_state,
            lagged_response.flux_response,
            **self._shared_flux_bc_kwargs(),
        )
        flux_bar = self.pullback_shared_fluxes(state, shared_fluxes, rhs_bar)
        pullback_fn = getattr(self.shared_flux_model, "pullback_evaluate_with_lagged_response", None)
        if callable(pullback_fn):
            flux_response_bar = pullback_fn(
                working_state,
                lagged_response.flux_response,
                flux_bar,
                **self._shared_flux_bc_kwargs(),
            )
        else:
            _, flux_pullback = jax.vjp(
                lambda response_value: self.shared_flux_model.evaluate_with_lagged_response(
                    working_state,
                    response_value,
                    **self._shared_flux_bc_kwargs(),
                ),
                lagged_response.flux_response,
            )
            (flux_response_bar,) = flux_pullback(flux_bar)
        return TransportLaggedResponse(flux_response=flux_response_bar)

    def pullback_evaluate_with_lagged_response_support_payload(
        self,
        t,
        state,
        runtime,
        lagged_response,
        rhs_bar,
        support,
    ):
        support, geometry = self._split_realtime_geometry_payload(support)
        if geometry is not None:
            support_bar = self.pullback_evaluate_with_lagged_response_support_payload(
                t,
                state,
                runtime,
                lagged_response,
                rhs_bar,
                support,
            )
            geometry_delta0 = _float_delta_tree_like(geometry)
            _, geometry_pullback = jax.vjp(
                lambda geometry_delta: self.with_geometry_payload(
                    _add_float_delta_tree(geometry, geometry_delta)
                ).evaluate_with_lagged_response(
                    t,
                    state,
                    runtime,
                    lagged_response,
                ),
                geometry_delta0,
            )
            (geometry_bar,) = geometry_pullback(rhs_bar)
            return self._realtime_geometry_payload_bar(
                {"ntx_support": support, "geometry": geometry},
                support_bar,
                geometry_bar,
            )
        if self.shared_flux_model is None or lagged_response is None or lagged_response.flux_response is None:
            return jax.tree_util.tree_map(jnp.zeros_like, support)

        working_state, _eidx = self._prepare_working_state(state)
        shared_fluxes = self.shared_flux_model.evaluate_with_lagged_response(
            working_state,
            lagged_response.flux_response,
            **self._shared_flux_bc_kwargs(),
        )
        flux_bar = self.pullback_shared_fluxes(state, shared_fluxes, rhs_bar)
        pullback_fn = getattr(
            self.shared_flux_model,
            "pullback_evaluate_with_lagged_response_support_payload",
            None,
        )
        if callable(pullback_fn):
            return _sanitize_float_delta_bar_tree(
                support,
                pullback_fn(
                    working_state,
                    lagged_response.flux_response,
                    flux_bar,
                    support,
                    **self._shared_flux_bc_kwargs(),
                ),
            )
        support_delta0 = _float_delta_tree_like(support)
        _, support_delta_pullback = jax.vjp(
            lambda support_delta: self.shared_flux_model.with_support_payload(
                _add_float_delta_tree(support, support_delta)
            ).evaluate_with_lagged_response(
                working_state,
                lagged_response.flux_response,
                **self._shared_flux_bc_kwargs(),
            ),
            support_delta0,
        )
        (support_bar,) = support_delta_pullback(flux_bar)
        return _sanitize_float_delta_bar_tree(support, support_bar)

    def pullback_evaluate_with_lagged_response_state(self, t, state, runtime, lagged_response, rhs_bar):
        """Reverse-only split pullback for state dependence of the lagged RHS."""
        del t, runtime
        if self.shared_flux_model is None or lagged_response is None or lagged_response.flux_response is None:
            _, rhs_pullback = jax.vjp(
                lambda state_value: self._evaluate_state(state_value, lagged_response=lagged_response),
                state,
            )
            (state_bar,) = rhs_pullback(rhs_bar)
            return state_bar

        working_state, eidx = self._prepare_working_state(state)
        shared_fluxes = self.shared_flux_model.evaluate_with_lagged_response(
            working_state,
            lagged_response.flux_response,
            **self._shared_flux_bc_kwargs(),
        )

        direct_working_state_bar, flux_bar = self._pullback_shared_flux_rhs_state_and_fluxes(
            state,
            working_state,
            eidx,
            shared_fluxes,
            rhs_bar,
        )
        flux_state_pullback_fn = getattr(self.shared_flux_model, "pullback_evaluate_with_lagged_response_state", None)
        if callable(flux_state_pullback_fn):
            working_state_bar = flux_state_pullback_fn(
                working_state,
                lagged_response.flux_response,
                flux_bar,
                **self._shared_flux_bc_kwargs(),
            )
        else:
            _, flux_state_pullback = jax.vjp(
                lambda working_state_value: self.shared_flux_model.evaluate_with_lagged_response(
                    working_state_value,
                    lagged_response.flux_response,
                    **self._shared_flux_bc_kwargs(),
                ),
                working_state,
            )
            (working_state_bar,) = flux_state_pullback(flux_bar)

        total_working_state_bar = jax.tree_util.tree_map(
            lambda a, b: a + b,
            direct_working_state_bar,
            working_state_bar,
        )
        return self._prepare_working_state_pullback(state, total_working_state_bar)

    def pullback_evaluate_with_lagged_response_all(
        self,
        t,
        state,
        runtime,
        lagged_response,
        rhs_bar,
        support,
    ):
        """Joint fixed-lagged RHS pullback for state, response, and NTX support.

        This is an exact reverse-only convenience hook.  When the support
        payload contains a realtime-geometry component, retain the existing
        geometry pullback path: it differentiates the VMEC payload separately
        and must not be replaced by a generic transport VJP.  For the normal
        NTX-support-only path, evaluate shared fluxes and the RHS assembly
        pullback once, then fan its common flux cotangent out to the three
        existing exact flux-model pullbacks.
        """
        del t, runtime
        support_ntx, geometry = self._split_realtime_geometry_payload(support)
        if (
            self.shared_flux_model is None
            or lagged_response is None
            or lagged_response.flux_response is None
            or support_ntx is None
        ):
            return (
                self.pullback_evaluate_with_lagged_response_state(
                    0.0, state, None, lagged_response, rhs_bar
                ),
                self.pullback_evaluate_with_lagged_response(
                    0.0, state, None, lagged_response, rhs_bar
                ),
                self.pullback_evaluate_with_lagged_response_support_payload(
                    0.0, state, None, lagged_response, rhs_bar, support
                ),
            )

        working_state, eidx = self._prepare_working_state(state)
        shared_fluxes = self.shared_flux_model.evaluate_with_lagged_response(
            working_state,
            lagged_response.flux_response,
            **self._shared_flux_bc_kwargs(),
        )
        direct_working_state_bar, flux_bar = self._pullback_shared_flux_rhs_state_and_fluxes(
            state,
            working_state,
            eidx,
            shared_fluxes,
            rhs_bar,
        )

        flux_state_pullback_fn = getattr(
            self.shared_flux_model,
            "pullback_evaluate_with_lagged_response_state",
            None,
        )
        if callable(flux_state_pullback_fn):
            working_state_bar = flux_state_pullback_fn(
                working_state,
                lagged_response.flux_response,
                flux_bar,
                **self._shared_flux_bc_kwargs(),
            )
        else:
            _, flux_state_pullback = jax.vjp(
                lambda working_state_value: self.shared_flux_model.evaluate_with_lagged_response(
                    working_state_value,
                    lagged_response.flux_response,
                    **self._shared_flux_bc_kwargs(),
                ),
                working_state,
            )
            (working_state_bar,) = flux_state_pullback(flux_bar)

        response_pullback_fn = getattr(
            self.shared_flux_model,
            "pullback_evaluate_with_lagged_response",
            None,
        )
        if callable(response_pullback_fn):
            flux_response_bar = response_pullback_fn(
                working_state,
                lagged_response.flux_response,
                flux_bar,
                **self._shared_flux_bc_kwargs(),
            )
        else:
            _, response_pullback = jax.vjp(
                lambda response_value: self.shared_flux_model.evaluate_with_lagged_response(
                    working_state,
                    response_value,
                    **self._shared_flux_bc_kwargs(),
                ),
                lagged_response.flux_response,
            )
            (flux_response_bar,) = response_pullback(flux_bar)

        if geometry is not None:
            # Geometry remains on the established VMEC payload reverse path.
            # State and response above still shared their assembly traversal.
            support_bar = self.pullback_evaluate_with_lagged_response_support_payload(
                0.0,
                state,
                None,
                lagged_response,
                rhs_bar,
                support,
            )
        else:
            support_pullback_fn = getattr(
                self.shared_flux_model,
                "pullback_evaluate_with_lagged_response_support_payload",
                None,
            )
            if callable(support_pullback_fn):
                support_bar = support_pullback_fn(
                    working_state,
                    lagged_response.flux_response,
                    flux_bar,
                    support_ntx,
                    **self._shared_flux_bc_kwargs(),
                )
            else:
                support_delta0 = _float_delta_tree_like(support_ntx)
                _, support_pullback = jax.vjp(
                    lambda support_delta: self.shared_flux_model.with_support_payload(
                        _add_float_delta_tree(support_ntx, support_delta)
                    ).evaluate_with_lagged_response(
                        working_state,
                        lagged_response.flux_response,
                        **self._shared_flux_bc_kwargs(),
                    ),
                    support_delta0,
                )
                (support_bar,) = support_pullback(flux_bar)

        state_bar = self._prepare_working_state_pullback(
            state,
            jax.tree_util.tree_map(lambda a, b: a + b, direct_working_state_bar, working_state_bar),
        )
        return (
            state_bar,
            TransportLaggedResponse(flux_response=flux_response_bar),
            (
                support_bar
                if geometry is not None
                else _sanitize_float_delta_bar_tree(support_ntx, support_bar)
            ),
        )

    def pullback_evaluate_with_lagged_response_state_direct(self, t, state, runtime, lagged_response, rhs_bar):
        """State pullback through equation assembly with shared fluxes held fixed."""
        del t, runtime
        if self.shared_flux_model is None or lagged_response is None or lagged_response.flux_response is None:
            return jax.tree_util.tree_map(jnp.zeros_like, state)

        working_state, eidx = self._prepare_working_state(state)
        shared_fluxes = self.shared_flux_model.evaluate_with_lagged_response(
            working_state,
            lagged_response.flux_response,
            **self._shared_flux_bc_kwargs(),
        )
        direct_working_state_bar = self._pullback_shared_flux_rhs_state(
            state,
            working_state,
            eidx,
            shared_fluxes,
            rhs_bar,
        )
        return self._prepare_working_state_pullback(state, direct_working_state_bar)

    def _pullback_evaluate_with_lagged_response_state_direct_component(
        self,
        t,
        state,
        runtime,
        lagged_response,
        rhs_bar,
        *,
        component: str,
    ):
        del t, runtime
        if self.shared_flux_model is None or lagged_response is None or lagged_response.flux_response is None:
            return jax.tree_util.tree_map(jnp.zeros_like, state)

        working_state, eidx = self._prepare_working_state(state)
        shared_fluxes = self.shared_flux_model.evaluate_with_lagged_response(
            working_state,
            lagged_response.flux_response,
            **self._shared_flux_bc_kwargs(),
        )
        direct_working_state_bar = self._pullback_shared_flux_rhs_state_component(
            state,
            working_state,
            eidx,
            shared_fluxes,
            rhs_bar,
            component=component,
        )
        return self._prepare_working_state_pullback(state, direct_working_state_bar)

    def pullback_evaluate_with_lagged_response_state_direct_density(self, t, state, runtime, lagged_response, rhs_bar):
        return self._pullback_evaluate_with_lagged_response_state_direct_component(
            t,
            state,
            runtime,
            lagged_response,
            rhs_bar,
            component="density",
        )

    def pullback_evaluate_with_lagged_response_state_direct_pressure(self, t, state, runtime, lagged_response, rhs_bar):
        return self._pullback_evaluate_with_lagged_response_state_direct_component(
            t,
            state,
            runtime,
            lagged_response,
            rhs_bar,
            component="pressure",
        )

    def pullback_evaluate_with_lagged_response_state_direct_er(self, t, state, runtime, lagged_response, rhs_bar):
        return self._pullback_evaluate_with_lagged_response_state_direct_component(
            t,
            state,
            runtime,
            lagged_response,
            rhs_bar,
            component="er",
        )

    def pullback_evaluate_with_lagged_response_state_direct_er_diffusion(self, t, state, runtime, lagged_response, rhs_bar):
        return self._pullback_evaluate_with_lagged_response_state_direct_component(
            t,
            state,
            runtime,
            lagged_response,
            rhs_bar,
            component="er_diffusion",
        )

    def pullback_evaluate_with_lagged_response_state_direct_er_ambipolar(self, t, state, runtime, lagged_response, rhs_bar):
        return self._pullback_evaluate_with_lagged_response_state_direct_component(
            t,
            state,
            runtime,
            lagged_response,
            rhs_bar,
            component="er_ambipolar",
        )

    def pullback_evaluate_with_lagged_response_state_direct_er_ambi_coeff(self, t, state, runtime, lagged_response, rhs_bar):
        return self._pullback_evaluate_with_lagged_response_state_direct_component(
            t,
            state,
            runtime,
            lagged_response,
            rhs_bar,
            component="er_ambi_coeff",
        )

    def pullback_evaluate_with_lagged_response_state_direct_er_ambi_charge_flux(self, t, state, runtime, lagged_response, rhs_bar):
        return self._pullback_evaluate_with_lagged_response_state_direct_component(
            t,
            state,
            runtime,
            lagged_response,
            rhs_bar,
            component="er_ambi_charge_flux",
        )

    def pullback_evaluate_with_lagged_response_state_direct_generic(self, t, state, runtime, lagged_response, rhs_bar):
        """Diagnostic generic VJP for fixed-shared-flux RHS state dependence."""
        del t, runtime
        if self.shared_flux_model is None or lagged_response is None or lagged_response.flux_response is None:
            return jax.tree_util.tree_map(jnp.zeros_like, state)

        working_state, eidx = self._prepare_working_state(state)
        shared_fluxes = self.shared_flux_model.evaluate_with_lagged_response(
            working_state,
            lagged_response.flux_response,
            **self._shared_flux_bc_kwargs(),
        )

        def _rhs_from_working_state(working_state_value):
            return self._evaluate_with_shared_fluxes_from_working_state(
                working_state_value,
                eidx,
                state,
                shared_fluxes,
            )

        _, direct_pullback = jax.vjp(_rhs_from_working_state, working_state)
        (direct_working_state_bar,) = direct_pullback(rhs_bar)
        return self._prepare_working_state_pullback(state, direct_working_state_bar)

    def pullback_evaluate_with_lagged_response_state_flux(self, t, state, runtime, lagged_response, rhs_bar):
        """State pullback through the shared-flux model, with equation assembly transposed separately."""
        del t, runtime
        if self.shared_flux_model is None or lagged_response is None or lagged_response.flux_response is None:
            return jax.tree_util.tree_map(jnp.zeros_like, state)

        working_state, _eidx = self._prepare_working_state(state)
        shared_fluxes = self.shared_flux_model.evaluate_with_lagged_response(
            working_state,
            lagged_response.flux_response,
            **self._shared_flux_bc_kwargs(),
        )
        flux_bar = self.pullback_shared_fluxes(state, shared_fluxes, rhs_bar)
        flux_state_pullback_fn = getattr(self.shared_flux_model, "pullback_evaluate_with_lagged_response_state", None)
        if callable(flux_state_pullback_fn):
            working_state_bar = flux_state_pullback_fn(
                working_state,
                lagged_response.flux_response,
                flux_bar,
                **self._shared_flux_bc_kwargs(),
            )
        else:
            _, flux_state_pullback = jax.vjp(
                lambda working_state_value: self.shared_flux_model.evaluate_with_lagged_response(
                    working_state_value,
                    lagged_response.flux_response,
                    **self._shared_flux_bc_kwargs(),
                ),
                working_state,
            )
            (working_state_bar,) = flux_state_pullback(flux_bar)
        return self._prepare_working_state_pullback(state, working_state_bar)

    def pullback_evaluate_with_lagged_response_state_flux_generic(self, t, state, runtime, lagged_response, rhs_bar):
        """Diagnostic generic VJP for flux-model state dependence."""
        del t, runtime
        if self.shared_flux_model is None or lagged_response is None or lagged_response.flux_response is None:
            return jax.tree_util.tree_map(jnp.zeros_like, state)

        working_state, _eidx = self._prepare_working_state(state)
        shared_fluxes = self.shared_flux_model.evaluate_with_lagged_response(
            working_state,
            lagged_response.flux_response,
            **self._shared_flux_bc_kwargs(),
        )
        flux_bar = self.pullback_shared_fluxes(state, shared_fluxes, rhs_bar)

        def _complete_bar_like(output, bar):
            if not isinstance(output, dict) or not isinstance(bar, dict):
                return bar

            def _bar_or_zero(key, template):
                value = bar.get(key, None)
                if value is None:
                    return jnp.zeros_like(template)
                arr = jnp.asarray(value)
                if arr.dtype == jax.dtypes.float0:
                    return jnp.zeros_like(template)
                return jnp.asarray(value, dtype=jnp.asarray(template).dtype)

            return {key: _bar_or_zero(key, value) for key, value in output.items()}

        def _fluxes_from_working_state(working_state_value):
            fluxes = self.shared_flux_model.evaluate_with_lagged_response(
                working_state_value,
                lagged_response.flux_response,
                **self._shared_flux_bc_kwargs(),
            )
            return _with_center_fluxes_from_faces(fluxes)

        flux_output, flux_state_pullback = jax.vjp(_fluxes_from_working_state, working_state)
        flux_bar = _complete_bar_like(flux_output, flux_bar)
        (working_state_bar,) = flux_state_pullback(flux_bar)
        return self._prepare_working_state_pullback(state, working_state_bar)

    def pullback_evaluate_with_lagged_response_state_joint_generic(self, t, state, runtime, lagged_response, rhs_bar):
        """Diagnostic joint assembly pullback with generic flux-model state VJP."""
        del t, runtime
        if self.shared_flux_model is None or lagged_response is None or lagged_response.flux_response is None:
            return jax.tree_util.tree_map(jnp.zeros_like, state)

        working_state, eidx = self._prepare_working_state(state)
        shared_fluxes = self.shared_flux_model.evaluate_with_lagged_response(
            working_state,
            lagged_response.flux_response,
            **self._shared_flux_bc_kwargs(),
        )
        direct_working_state_bar, flux_bar = self._pullback_shared_flux_rhs_state_and_fluxes(
            state,
            working_state,
            eidx,
            shared_fluxes,
            rhs_bar,
        )

        def _complete_bar_like(output, bar):
            if not isinstance(output, dict) or not isinstance(bar, dict):
                return bar

            def _bar_or_zero(key, template):
                value = bar.get(key, None)
                if value is None:
                    return jnp.zeros_like(template)
                arr = jnp.asarray(value)
                if arr.dtype == jax.dtypes.float0:
                    return jnp.zeros_like(template)
                return jnp.asarray(value, dtype=jnp.asarray(template).dtype)

            return {key: _bar_or_zero(key, value) for key, value in output.items()}

        def _fluxes_from_working_state(working_state_value):
            fluxes = self.shared_flux_model.evaluate_with_lagged_response(
                working_state_value,
                lagged_response.flux_response,
                **self._shared_flux_bc_kwargs(),
            )
            return _with_center_fluxes_from_faces(fluxes)

        flux_output, flux_state_pullback = jax.vjp(_fluxes_from_working_state, working_state)
        flux_bar = _complete_bar_like(flux_output, flux_bar)
        (flux_working_state_bar,) = flux_state_pullback(flux_bar)
        total_working_state_bar = jax.tree_util.tree_map(
            lambda a, b: a + b,
            direct_working_state_bar,
            flux_working_state_bar,
        )
        return self._prepare_working_state_pullback(state, total_working_state_bar)

    def _evaluate_state(self, state, lagged_response=None):
        import jax.numpy as jnp
        from ._state import TransportState
        working_state, eidx = self._prepare_working_state(state)
        density_eq, temperature_eq, er_eq = self._resolve_equations()

        shared_fluxes = None
        if lagged_response is None:
            if self.shared_flux_model is not None:
                shared_fluxes = self.shared_flux_model(working_state)
        else:
            if self.shared_flux_model is not None:
                shared_fluxes = self.shared_flux_model.evaluate_with_lagged_response(
                    working_state,
                    lagged_response.flux_response,
                    **self._shared_flux_bc_kwargs(),
                )

        if shared_fluxes is not None:
            return self.evaluate_with_shared_fluxes(0.0, state, None, shared_fluxes)

        density_rhs = (
            density_eq(working_state, fluxes=shared_fluxes)
            if density_eq is not None
            else jnp.zeros_like(state.density)
        )
        pressure_rhs = (
            temperature_eq(working_state, fluxes=shared_fluxes)
            if temperature_eq is not None
            else jnp.zeros_like(state.pressure)
        )
        Er_rhs = (
            er_eq(working_state, fluxes=shared_fluxes)
            if er_eq is not None
            else jnp.zeros_like(state.Er)
        )

        density_rhs = _expand_density_rhs_to_full_shape(density_rhs, state.density, self.species)

        if eidx is not None:
            density_rhs = density_rhs.at[int(eidx), :].set(jnp.zeros_like(density_rhs[int(eidx), :]))

        if density_eq is not None and hasattr(density_eq, "enforce_dirichlet_boundary_rhs"):
            density_rhs = density_eq.enforce_dirichlet_boundary_rhs(working_state, density_rhs)

        if temperature_eq is not None and hasattr(temperature_eq, "enforce_dirichlet_boundary_rhs"):
            pressure_rhs = temperature_eq.enforce_dirichlet_boundary_rhs(working_state, density_rhs, pressure_rhs)

        if er_eq is not None and hasattr(er_eq, "enforce_dirichlet_boundary_rhs"):
            Er_rhs = er_eq.enforce_dirichlet_boundary_rhs(working_state, Er_rhs)

        return TransportState(
            density=density_rhs,
            pressure=pressure_rhs,
            Er=Er_rhs,
        )

    def __call__(self, t, state, runtime):
        """
        Call all equations with state, return a TransportState matching the state structure.
        Always output all three fields, setting missing ones to zero arrays of the correct shape.
        When electrons are present, evaluate the RHS on a quasi-neutral working
        state, but keep electron density out of the solved density subsystem.
        This matches the NTSS-style pattern: evolve independent ion/impurity
        density rows, reconstruct electron density algebraically for the working
        state and accepted/output states.
        """
        del t, runtime
        return self._evaluate_state(state)

    def vector_field(self, t, y, args):
        """
        Torax-style vector field for JAX ODE solvers: (t, y, args) -> dy/dt
        y is the state, args[0] is the runtime dict.
        """
        return self(t,y, args)


