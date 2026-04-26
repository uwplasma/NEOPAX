from typing import Dict, Type
import dataclasses
import jax
import jax.numpy as jnp
from jax import jit
from ._fem import conservative_update, faces_from_cell_centered
from ._cell_variable import make_profile_cell_variable
from ._boundary_conditions import left_constraints_from_bc_model, right_constraints_from_bc_model
from ._constants import elementary_charge
from ._source_models import (
    assemble_density_source_components,
    assemble_pressure_source_components,
    sum_source_components,
)
from ._transport_flux_models import build_face_transport_state, build_ntss_like_face_transport_state
from ._state import (
    DEFAULT_TRANSPORT_DENSITY_FLOOR,
    DEFAULT_TRANSPORT_TEMPERATURE_FLOOR,
    apply_transport_density_floor,
    apply_transport_temperature_floor,
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


def project_fixed_temperature_species(state, temperature_active_mask=None, fixed_temperature_profile=None):
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
    fixed_pressure = state.density * fixed_temperature
    pressure = jnp.where(active_mask, state.pressure, fixed_pressure)
    return dataclasses.replace(state, pressure=pressure)


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
class DensityEquation(EquationBase):
    dr_cells: jax.Array = dataclasses.field(repr=False)
    Vprime: jax.Array = dataclasses.field(repr=False)
    Vprime_half: jax.Array = dataclasses.field(repr=False)
    flux_model: callable = dataclasses.field(repr=False)
    flux_faces_builder: callable = dataclasses.field(repr=False)
    active_species_mask: jax.Array = dataclasses.field(repr=False)
    independent_density_mask: jax.Array = dataclasses.field(repr=False)
    face_flux_builder: callable = dataclasses.field(repr=False, default=None)
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

    def debug_components(self, state, fluxes=None):
        if fluxes is None:
            fluxes = self.flux_model(state)
        use_face_gamma = self._use_model_face_particle_fluxes()
        need_face_fluxes = use_face_gamma
        face_fluxes = self.face_flux_builder(state) if (self.face_flux_builder is not None and need_face_fluxes) else None
        Gamma = PARTICLE_FLUX_PHYSICAL_TO_STATE * fluxes["Gamma"]
        Gamma_faces_raw = (
            PARTICLE_FLUX_PHYSICAL_TO_STATE * face_fluxes["Gamma"]
            if (face_fluxes is not None and face_fluxes.get("Gamma", None) is not None and use_face_gamma)
            else self.flux_faces_builder(Gamma, self.particle_flux_reconstruction)
        )
        gamma_divergence_raw = jax.vmap(
            lambda flux: conservative_update(flux, self.dr_cells, self.Vprime, self.Vprime_half)
        )(Gamma_faces_raw)
        source_components = assemble_density_source_components(
            None if self.source_model is None else self.source_model(state),
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

    def __call__(self, state, fluxes=None):
        if fluxes is None:
            fluxes = self.flux_model(state)
        use_face_gamma = self._use_model_face_particle_fluxes()
        need_face_fluxes = use_face_gamma
        face_fluxes = self.face_flux_builder(state) if (self.face_flux_builder is not None and need_face_fluxes) else None
        Gamma = PARTICLE_FLUX_PHYSICAL_TO_STATE * fluxes["Gamma"]
        Gamma_faces = (
            PARTICLE_FLUX_PHYSICAL_TO_STATE * face_fluxes["Gamma"]
            if (face_fluxes is not None and face_fluxes.get("Gamma", None) is not None and use_face_gamma)
            else self.flux_faces_builder(Gamma, self.particle_flux_reconstruction)
        )
        gamma_divergence = jax.vmap(
            lambda flux: conservative_update(flux, self.dr_cells, self.Vprime, self.Vprime_half)
        )(Gamma_faces)
        source_rhs = jnp.zeros_like(gamma_divergence)
        if self.source_model is not None:
            source_components = assemble_density_source_components(
                self.source_model(state),
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
    def face_flux_builder(state):
        state = apply_transport_density_floor(state, density_floor)
        state = apply_transport_temperature_floor(state, temperature_floor, density_floor)
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
        bc = self.temperature_bc_model
        if bc is None:
            return pressure_rhs

        out = pressure_rhs

        left_type = str(getattr(bc, "left_type", "")).strip().lower()
        if left_type == "dirichlet":
            left_value = getattr(bc, "left_value", None)
            if left_value is None:
                t_left = state.temperature[:, 0]
            else:
                t_left = jnp.asarray(left_value, dtype=pressure_rhs.dtype)
                if t_left.ndim == 0:
                    t_left = jnp.broadcast_to(t_left, (pressure_rhs.shape[0],))
            out = out.at[:, 0].set(t_left * density_rhs[:, 0])

        right_type = str(getattr(bc, "right_type", "")).strip().lower()
        if right_type == "dirichlet":
            right_value = getattr(bc, "right_value", None)
            if right_value is None:
                t_right = state.temperature[:, -1]
            else:
                t_right = jnp.asarray(right_value, dtype=pressure_rhs.dtype)
                if t_right.ndim == 0:
                    t_right = jnp.broadcast_to(t_right, (pressure_rhs.shape[0],))
            out = out.at[:, -1].set(t_right * density_rhs[:, -1])

        return out

    def debug_components(self, state, fluxes=None):
        if fluxes is None:
            fluxes = self.flux_model(state)
        use_face_q = self._use_model_face_heat_fluxes()
        use_face_gamma = self._use_model_face_particle_fluxes()
        need_face_fluxes = use_face_q or use_face_gamma
        face_fluxes = self.face_flux_builder(state) if (self.face_flux_builder is not None and need_face_fluxes) else None
        Q = HEAT_FLUX_PHYSICAL_TO_STATE * fluxes["Q"]
        temperature_ghost = self.temperature_ghost_builder(state.temperature)
        Q_faces = (
            HEAT_FLUX_PHYSICAL_TO_STATE * face_fluxes["Q"]
            if (face_fluxes is not None and use_face_q)
            else self.flux_faces_builder(Q, self.heat_flux_reconstruction)
        )
        temperature_left, temperature_right = _temperature_face_states(
            temperature_ghost,
            self.convection_reconstruction,
        )

        def _convective_component(gamma_key):
            gamma_comp = (face_fluxes.get(gamma_key, None) if (face_fluxes is not None and use_face_gamma) else fluxes.get(gamma_key, None))
            if gamma_comp is None:
                gamma_faces = jnp.zeros_like(Q_faces)
            elif face_fluxes is not None and use_face_gamma:
                gamma_faces = PARTICLE_FLUX_PHYSICAL_TO_STATE * gamma_comp
            else:
                gamma_faces = self.flux_faces_builder(PARTICLE_FLUX_PHYSICAL_TO_STATE * gamma_comp)
            temperature_upwind = jnp.where(gamma_faces >= 0.0, temperature_left, temperature_right)
            return temperature_upwind * gamma_faces

        convective_neo_faces = (
            _convective_component("Gamma_neo") if self.include_neo_convection else jnp.zeros_like(Q_faces)
        )
        convective_turb_faces = (
            _convective_component("Gamma_turb") if self.include_turbulent_convection else jnp.zeros_like(Q_faces)
        )
        convective_classical_faces = (
            _convective_component("Gamma_classical") if self.include_classical_convection else jnp.zeros_like(Q_faces)
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
            None if self.source_model is None else self.source_model(state),
            state,
            self.species,
        )
        source_rhs = sum_source_components(source_components, state.pressure)
        work_rhs = (
            self.charge_qp[:, None]
            * PARTICLE_FLUX_PHYSICAL_TO_STATE
            * fluxes["Gamma"]
            * state.Er[None, :]
            if self.include_work_term
            else jnp.zeros_like(state.pressure)
        )
        total_rhs = (2.0 / 3.0) * (thermal_flux_rhs + source_rhs + work_rhs)
        return {
            "Q_faces": Q_faces,
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

    def __call__(self, state, fluxes=None):
        if fluxes is None:
            fluxes = self.flux_model(state)
        use_face_q = self._use_model_face_heat_fluxes()
        use_face_gamma = self._use_model_face_particle_fluxes()
        need_face_fluxes = use_face_q or use_face_gamma
        face_fluxes = self.face_flux_builder(state) if (self.face_flux_builder is not None and need_face_fluxes) else None
        Q = HEAT_FLUX_PHYSICAL_TO_STATE * fluxes["Q"]
        temperature_ghost = self.temperature_ghost_builder(state.temperature)
        Q_faces = (
            HEAT_FLUX_PHYSICAL_TO_STATE * face_fluxes["Q"]
            if (face_fluxes is not None and use_face_q)
            else self.flux_faces_builder(Q, self.heat_flux_reconstruction)
        )
        temperature_left, temperature_right = _temperature_face_states(
            temperature_ghost,
            self.convection_reconstruction,
        )

        def _convective_component(gamma_key):
            gamma_comp = (face_fluxes.get(gamma_key, None) if (face_fluxes is not None and use_face_gamma) else fluxes.get(gamma_key, None))
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
            None if self.source_model is None else self.source_model(state),
            state,
            self.species,
        )
        source_rhs = sum_source_components(source_components, state.pressure)
        work_rhs = (
            self.charge_qp[:, None]
            * PARTICLE_FLUX_PHYSICAL_TO_STATE
            * fluxes["Gamma"]
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
                    right_face_grad_constraint=jnp.asarray(0.0, dtype=prof.dtype),
                ).face_value(reconstruction=reconstruction)
            )(profile)
    return faces_builder


def _build_species_ghost_builder(bc_model):
    if bc_model is not None and hasattr(bc_model, "apply_ghost_all"):
        def ghost_builder(profile):
            return bc_model.apply_ghost_all(profile)
    elif bc_model is not None and hasattr(bc_model, "apply_ghost"):
        def ghost_builder(profile):
            return jax.vmap(lambda prof: bc_model.apply_ghost(prof))(profile)
    else:
        def ghost_builder(profile):
            return jnp.concatenate([profile[:, :1], profile, profile[:, -1:]], axis=1)
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
    def face_flux_builder(state):
        state = apply_transport_density_floor(state, density_floor)
        state = apply_transport_temperature_floor(state, temperature_floor, density_floor)
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
        )
    temperature_ghost_builder = _build_species_ghost_builder(bc_temperature)
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
    source_mode: str = "ambipolar_local"
    Er_relax: float = 1.0
    DEr: float = 1.0
    boundary_mode: str = "standard"
    Er_edge_relax: float = 1.0
    name: str = "Er"

    def _charge_flux_from_gamma(self, Gamma):
        mode = str(self.source_mode).strip().lower()
        if mode in {"ambipolar_local", "transport_local", "local"}:
            return jnp.sum(self.charge_qp[:, None] * Gamma, axis=0)

        Gamma_faces = self.gamma_faces_builder(Gamma)
        ambipolar_flux_center = 0.5 * (Gamma_faces[:, :-1] + Gamma_faces[:, 1:])
        return jnp.sum(self.charge_qp[:, None] * ambipolar_flux_center, axis=0)

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

    def debug_components(self, state, fluxes=None):
        if fluxes is None:
            fluxes = self.flux_model(state)
        Er = state.Er
        plasma_permitivity = _plasma_permitivity_from_prefactor(
            state,
            self.species_mass,
            self.permitivity_prefactor,
        )
        Gamma = fluxes["Gamma"]
        charge_flux = self._charge_flux_from_gamma(Gamma)
        ambi_term = charge_flux * elementary_charge * 1.e-3 / plasma_permitivity
        er_diffusive_flux, er_diffusion = self._er_diffusion(Er)
        return {
            "charge_flux": charge_flux,
            "plasma_permitivity": plasma_permitivity,
            "ambi_term": ambi_term,
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
        Gamma = fluxes["Gamma"]
        charge_flux = self._charge_flux_from_gamma(Gamma)
        ambi_term = charge_flux * elementary_charge * 1.e-3 / plasma_permitivity
        _, Er_diffusion = self._er_diffusion(Er)
        SourceEr = self.Er_relax * (self.DEr * Er_diffusion - ambi_term)
        SourceEr = SourceEr.at[0].set(0.)
        if self.boundary_mode == "floating_ambipolar_edge":
            SourceEr = SourceEr.at[-1].set(-self.Er_edge_relax * ambi_term[-1])
        return SourceEr

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
    flux_model,
    species_mass,
    charge_qp,
    bc_gamma,
    bc_er,
    Er_relax=1.0,
    DEr=1.0,
    source_mode="ambipolar_local",
    reconstruction="linear",
    boundary_mode="standard",
    Er_edge_relax=1.0,
):
    dr_cells = jnp.diff(field.r_grid_half)
    Vprime = field.Vprime
    Vprime_half = field.Vprime_half
    psi_fac = 1.0 + 1.0 / (field.enlogation * jnp.square(field.iota))
    psi_fac = psi_fac.at[0].set(1.0)
    permitivity_prefactor = psi_fac / jnp.square(field.B0)
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
                    right_face_grad_constraint=jnp.asarray(0.0, dtype=G.dtype),
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
                right_face_constraint=er_profile[-1],
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
        source_mode=str(source_mode).strip().lower(),
        Er_relax=Er_relax,
        DEr=DEr,
        boundary_mode=str(boundary_mode).strip().lower(),
        Er_edge_relax=Er_edge_relax,
    )


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
    Er_source_mode = solver_cfg.get("Er_source_mode", "transport_centered")
    Er_boundary_mode = solver_cfg.get("Er_right_boundary_mode", solver_cfg.get("Er_boundary_mode", "standard"))
    Er_edge_relax = solver_cfg.get("Er_edge_relax", Er_relax)
    density_flux_reconstruction = solver_cfg.get("density_flux_reconstruction", "closure_face_flux")
    density_particle_face_closure_mode = solver_cfg.get("density_particle_face_closure_mode", "reconstructed")
    include_neo_convection = solver_cfg.get("temperature_include_neo_convection", True)
    include_turbulent_convection = solver_cfg.get("temperature_include_turbulent_convection", True)
    include_classical_convection = solver_cfg.get("temperature_include_classical_convection", True)
    include_work_term = solver_cfg.get("temperature_include_work_term", True)
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
            flux_model,
            species_mass,
            charge_qp,
            bc_gamma,
            bc_er,
            Er_relax=Er_relax,
            DEr=DEr,
            source_mode=Er_source_mode,
            boundary_mode=Er_boundary_mode,
            Er_edge_relax=Er_edge_relax,
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
    vmec_file = geom_cfg.get("vmec_file")
    boozer_file = geom_cfg.get("boozer_file")
    field = None
    if vmec_file is not None and boozer_file is not None:
        field = get_geometry_model("vmec_booz", n_r=n_radial, vmec=vmec_file, booz=boozer_file)

    energy_grid_cfg = config.get("energy_grid", {})
    n_x = int(energy_grid_cfg.get("n_x", 4))
    energy_grid = get_energy_grid_model("standard_laguerre", n_x=n_x, n_order=3)
    neoclassical_cfg = config.get("neoclassical", {})
    database = None
    neoclassical_file = neoclassical_cfg.get("neoclassical_file")
    if neoclassical_file and field is not None:
        database = Monoenergetic.read_monkes(field.a_b, neoclassical_file)

    neoclassical_factory = get_transport_flux_model(neoclassical_cfg.get("flux_model", "monkes_database"))
    turbulence_factory = get_transport_flux_model(config.get("turbulence", {}).get("flux_model", "none"))
    classical_factory = get_transport_flux_model(config.get("classical", {}).get("flux_model", "none")) if "classical" in config else None
    neoclassical_model = neoclassical_factory(species, energy_grid, field, database)
    turbulence_model = turbulence_factory(species, energy_grid, field, database) if turbulence_factory is not None else ZeroTransportModel()
    classical_model = classical_factory(species, energy_grid, field, database) if classical_factory is not None else ZeroTransportModel()
    flux_model = build_transport_flux_model(neoclassical_model, turbulence_model, classical_model)

    boundary_cfg = config.get("boundary", {})
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
        working_state = project_fixed_temperature_species(
            working_state,
            self.temperature_active_mask,
            self.fixed_temperature_profile,
        )
        working_state = apply_transport_temperature_floor(
            working_state,
            self.temperature_floor,
            self.density_floor,
        )
        return working_state, eidx

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
        import jax.numpy as jnp
        from ._state import TransportState
        working_state, eidx = self._prepare_working_state(state)

        shared_fluxes = None
        if self.shared_flux_model is not None:
            shared_fluxes = self.shared_flux_model(working_state)

        density_eq = self.density_equation
        temperature_eq = self.temperature_equation
        er_eq = self.er_equation
        if density_eq is None:
            density_eq = next((eq for eq in self.equations if getattr(eq, "name", None) == "density"), None)
        if temperature_eq is None:
            temperature_eq = next((eq for eq in self.equations if getattr(eq, "name", None) == "temperature"), None)
        if er_eq is None:
            er_eq = next((eq for eq in self.equations if getattr(eq, "name", None) == "Er"), None)

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

        # Keep the returned full density RHS aligned with the reduced solved
        # subsystem: electrons are reconstructed in the working/output state,
        # but their transport RHS row is not evolved independently.
        if eidx is not None:
            density_rhs = density_rhs.at[int(eidx), :].set(jnp.zeros_like(density_rhs[int(eidx), :]))

        if temperature_eq is not None and hasattr(temperature_eq, "enforce_dirichlet_boundary_rhs"):
            pressure_rhs = temperature_eq.enforce_dirichlet_boundary_rhs(working_state, density_rhs, pressure_rhs)

        return TransportState(
            density=density_rhs,
            pressure=pressure_rhs,
            Er=Er_rhs,
        )

    def vector_field(self, t, y, args):
        """
        Torax-style vector field for JAX ODE solvers: (t, y, args) -> dy/dt
        y is the state, args[0] is the runtime dict.
        """
        return self(t,y, args)


