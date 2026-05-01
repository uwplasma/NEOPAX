from __future__ import annotations


from typing import Any, Callable
import abc
import dataclasses
import h5py
import jax
import jax.numpy as jnp
import interpax
import sys
from pathlib import Path
from ._cell_variable import (
    get_gradient_density,
    get_gradient_temperature,
    make_profile_cell_variable,
)
from ._fem import cell_centered_from_faces, faces_from_cell_centered
from ._boundary_conditions import (
    left_constraints_from_bc_model,
    right_constraints_from_bc_model,
)
from ._neoclassical import (
    _as_species_constraint,
    _collisionality_kind,
    _nu_over_vnew_local,
    get_Lij_matrix_local,
    get_Neoclassical_Fluxes,
    get_Neoclassical_Fluxes_Faces,
    get_Neoclassical_Fluxes_With_Momentum_Correction,
)
from ._species import get_Thermodynamical_Forces_A1, get_Thermodynamical_Forces_A2, get_Thermodynamical_Forces_A3
from ._state import (
    DEFAULT_TRANSPORT_DENSITY_FLOOR,
    DEFAULT_TRANSPORT_TEMPERATURE_FLOOR,
    TransportState,
    apply_transport_density_floor,
    apply_transport_temperature_floor,
    get_v_thermal,
    safe_density,
    safe_temperature,
)
from ._source_models import assemble_pressure_source_components, sum_source_components
from ._model_api import (
    ModelCapabilities,
    ModelValidationContext,
    transport_model as transport_model_decorator,
    validate_transport_flux_builder,
)

DENSITY_STATE_TO_PHYSICAL = 1.0e20
TEMPERATURE_STATE_TO_PHYSICAL = 1.0e3
PRESSURE_SOURCE_STATE_TO_MW_M3 = 1.0 / 62.422
from ._turbulence import get_Turbulent_Fluxes_Analytical, get_Turbulent_Fluxes_PowerOverN


def compute_total_power_mw(state, species, pressure_source_model, geometry, fallback_mw=3.0):
    fallback = jnp.asarray(fallback_mw, dtype=state.density.dtype)
    if pressure_source_model is None or geometry is None:
        return fallback
    raw_sources = pressure_source_model(state)
    if not isinstance(raw_sources, dict):
        return fallback

    net_power_density = None
    alpha_power = raw_sources.get("AlphaPower")
    if alpha_power is not None:
        net_power_density = jnp.asarray(alpha_power, dtype=state.density.dtype)

    pbrems = raw_sources.get("PBrems")
    if pbrems is not None:
        pbrems_arr = jnp.asarray(pbrems, dtype=state.density.dtype)
        net_power_density = -pbrems_arr if net_power_density is None else net_power_density - pbrems_arr

    for key in ("heating", "external_heating", "ecrh", "icrh", "nbi", "ohmic_heating"):
        value = raw_sources.get(key)
        if value is None:
            continue
        arr = jnp.asarray(value, dtype=state.density.dtype)
        net_power_density = arr if net_power_density is None else net_power_density + arr

    if net_power_density is None:
        components = assemble_pressure_source_components(raw_sources, state, species)
        if not components:
            return fallback
        power_density_state = jnp.sum(sum_source_components(components, state.pressure), axis=0)
        power_density_mw_m3 = PRESSURE_SOURCE_STATE_TO_MW_M3 * power_density_state
        total_power = jnp.trapezoid(power_density_mw_m3 * geometry.Vprime, x=geometry.r_grid)
        return jnp.where(total_power < 0.0, fallback, total_power)

    power_density_mw_m3 = PRESSURE_SOURCE_STATE_TO_MW_M3 * net_power_density
    total_power = jnp.trapezoid(power_density_mw_m3 * geometry.Vprime, x=geometry.r_grid)
    return jnp.where(total_power < 0.0, fallback, total_power)


def compute_total_power_breakdown_mw(state, pressure_source_model, geometry):
    if pressure_source_model is None or geometry is None:
        return {}
    raw_sources = pressure_source_model(state)
    if not isinstance(raw_sources, dict):
        return {}

    breakdown: dict[str, jax.Array] = {}
    dtype = state.density.dtype

    def _integrate_state_power(name, value, sign=1.0):
        if value is None:
            return
        arr = jnp.asarray(value, dtype=dtype)
        power_density_mw_m3 = PRESSURE_SOURCE_STATE_TO_MW_M3 * (jnp.asarray(sign, dtype=dtype) * arr)
        breakdown[name] = jnp.trapezoid(power_density_mw_m3 * geometry.Vprime, x=geometry.r_grid)

    _integrate_state_power("alpha_power_mw", raw_sources.get("AlphaPower"), sign=1.0)
    _integrate_state_power("bremsstrahlung_mw", raw_sources.get("PBrems"), sign=-1.0)

    for key in ("heating", "external_heating", "ecrh", "icrh", "nbi", "ohmic_heating"):
        value = raw_sources.get(key)
        if value is None:
            continue
        _integrate_state_power(f"{key}_mw", value, sign=1.0)

    if breakdown:
        total = jnp.asarray(0.0, dtype=dtype)
        for value in breakdown.values():
            total = total + jnp.asarray(value, dtype=dtype)
        breakdown["net_total_mw"] = total
    return breakdown



# Registry for modular selection
TRANSPORT_FLUX_MODEL_REGISTRY: dict[str, Callable[[], "TransportFluxModelBase"]] = {}
TRANSPORT_FLUX_MODEL_CAPABILITIES: dict[str, ModelCapabilities] = {}

def register_transport_flux_model(
    name: str,
    builder: Callable[..., "TransportFluxModelBase"],
    *,
    capabilities: ModelCapabilities | None = None,
    validate: bool = False,
    validation_context: ModelValidationContext | None = None,
) -> None:
    key = str(name).strip().lower()
    if validate:
        if validation_context is None:
            raise ValueError("validation_context is required when validate=True for a transport flux model.")
        validate_transport_flux_builder(
            builder,
            validation_context,
            capabilities=capabilities,
            name=f"transport flux model '{name}'",
        )
    TRANSPORT_FLUX_MODEL_REGISTRY[key] = builder
    TRANSPORT_FLUX_MODEL_CAPABILITIES[key] = capabilities or ModelCapabilities()

def get_transport_flux_model(name: str) -> Callable[..., "TransportFluxModelBase"]:
    key = str(name).strip().lower()
    if key not in TRANSPORT_FLUX_MODEL_REGISTRY:
        raise ValueError(f"Unknown transport flux model '{name}'.")
    return TRANSPORT_FLUX_MODEL_REGISTRY[key]


def get_transport_flux_model_capabilities(name: str) -> ModelCapabilities:
    key = str(name).strip().lower()
    if key not in TRANSPORT_FLUX_MODEL_CAPABILITIES:
        raise ValueError(f"Unknown transport flux model '{name}'.")
    return TRANSPORT_FLUX_MODEL_CAPABILITIES[key]


def transport_flux_model(name: str, **register_kwargs):
    return transport_model_decorator(name, register_transport_flux_model, **register_kwargs)


@dataclasses.dataclass(frozen=True, eq=False)
class TransportFluxModelBase(abc.ABC):
        """
        Abstract base class for transport flux models.
        Output dict keys:
            - Gamma: particle flux
            - Q: heat flux
            - Upar: parallel flow
        """
        @abc.abstractmethod
        def __call__(self, state, geometry=None, params=None) -> dict:
                pass

        def build_local_particle_flux_evaluator(self, state):
                del state
                return None

        def evaluate_face_fluxes(self, state, face_state, **kwargs):
                del state, face_state, kwargs
                return None

        def build_lagged_response(self, state, **kwargs):
                del kwargs
                return JVPTransportFluxResponse(
                        reference_state=state,
                        reference_flux=self(state),
                )

        def evaluate_with_lagged_response(self, state, lagged_response, **kwargs):
                del kwargs
                delta_state = jax.tree_util.tree_map(
                        lambda current, reference: current - reference,
                        state,
                        lagged_response.reference_state,
                )
                tangent_flux = jax.jvp(
                        self.__call__,
                        (lagged_response.reference_state,),
                        (delta_state,),
                )[1]
                return jax.tree_util.tree_map(
                        lambda reference, tangent: reference + tangent,
                        lagged_response.reference_flux,
                        tangent_flux,
                )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class JVPTransportFluxResponse:
        reference_state: Any
        reference_flux: dict


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class NTXPreparedCoefficientResponse:
    reference_transport_moments: jax.Array
    reference_nu_hat: jax.Array
    reference_epsi_hat: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class NTXExactLijLaggedResponse:
        center_response: NTXPreparedCoefficientResponse


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class FaceTransportState:
    density: jax.Array
    pressure: jax.Array
    Er: jax.Array

    @property
    def temperature(self):
        return self.pressure / safe_density(self.density)


def _flatten_flux_dict(fluxes: dict) -> tuple[jax.Array, tuple[str, ...]]:
    ordered_keys = tuple(sorted(str(key) for key in fluxes.keys()))
    flat_parts = []
    dtype = None
    for key in ordered_keys:
        arr = jnp.asarray(fluxes[key])
        dtype = arr.dtype if dtype is None else dtype
        flat_parts.append(arr.reshape((-1,)))
    if not flat_parts:
        return jnp.zeros((0,), dtype=jnp.float64), ordered_keys
    return jnp.concatenate(flat_parts, axis=0).astype(dtype), ordered_keys


def _unflatten_flux_dict(flat_flux: jax.Array, reference_flux: dict) -> dict:
    ordered_keys = tuple(sorted(str(key) for key in reference_flux.keys()))
    out: dict[str, jax.Array] = {}
    offset = 0
    for key in ordered_keys:
        reference_arr = jnp.asarray(reference_flux[key])
        size = int(reference_arr.size)
        out[key] = jnp.asarray(flat_flux[offset:offset + size], dtype=reference_arr.dtype).reshape(reference_arr.shape)
        offset += size
    return out


def _extract_right_constraints(bc_model: Any, state_arr: jax.Array) -> tuple[jax.Array, jax.Array]:
    n_species = state_arr.shape[0]
    default_value = state_arr[:, -1]
    default_grad = jnp.zeros_like(default_value)
    if bc_model is None:
        return default_value, default_grad

    right_type = str(getattr(bc_model, "right_type", "dirichlet")).strip().lower()

    def _as_species(arr, fallback):
        if arr is None:
            return fallback
        out = jnp.asarray(arr)
        if out.ndim == 0:
            out = jnp.repeat(out[None], n_species, axis=0)
        if out.shape[0] < n_species:
            out = jnp.pad(out, (0, n_species - out.shape[0]), mode="edge")
        return out[:n_species]

    right_value = _as_species(getattr(bc_model, "right_value", None), default_value)
    right_grad = _as_species(getattr(bc_model, "right_gradient", None), default_grad)
    right_decay = _as_species(getattr(bc_model, "right_decay_length", None), jnp.ones_like(default_value))

    if right_type == "dirichlet":
        return right_value, jnp.zeros_like(right_value)
    if right_type == "neumann":
        return default_value, right_grad
    if right_type == "robin":
        robin_grad = -default_value / (right_decay + 1e-12)
        return default_value, robin_grad
    return default_value, default_grad


def _extract_face_constraints(
    bc_model: Any,
    state_arr: jax.Array,
    face_centers: jax.Array,
) -> tuple[jax.Array | None, jax.Array | None, jax.Array | None, jax.Array | None]:
    default_left = state_arr[:, 0]
    default_right = state_arr[:, -1]
    left_value, left_grad = left_constraints_from_bc_model(
        bc_model,
        default_left,
        profile=state_arr,
        face_centers=face_centers,
    )
    right_value, right_grad = right_constraints_from_bc_model(
        bc_model,
        default_right,
        profile=state_arr,
        face_centers=face_centers,
    )
    return left_value, left_grad, right_value, right_grad


def _face_profile(profile, face_centers, bc_model=None, reconstruction="linear"):
    if profile.ndim == 1:
        profile_2d = profile[None, :]
        squeeze = True
    else:
        profile_2d = profile
        squeeze = False
    left_value, left_grad, right_value, right_grad = _extract_face_constraints(
        bc_model,
        profile_2d,
        face_centers,
    )
    faces = jax.vmap(
        lambda prof, lv, lg, rv, rg: make_profile_cell_variable(
            prof,
            face_centers,
            left_face_constraint=lv,
            left_face_grad_constraint=lg,
            right_face_constraint=rv,
            right_face_grad_constraint=rg,
        ).face_value(reconstruction=reconstruction)
    )(profile_2d, left_value, left_grad, right_value, right_grad)
    if squeeze:
        return faces[0]
    return faces


def _face_profile_gradient(profile, face_centers, bc_model=None):
    if profile.ndim == 1:
        profile_2d = profile[None, :]
        squeeze = True
    else:
        profile_2d = profile
        squeeze = False
    left_value, left_grad, right_value, right_grad = _extract_face_constraints(
        bc_model,
        profile_2d,
        face_centers,
    )

    grads = jax.vmap(
        lambda prof, lv, lg, rv, rg: make_profile_cell_variable(
            prof,
            face_centers,
            left_face_constraint=lv,
            left_face_grad_constraint=lg,
            right_face_constraint=rv,
            right_face_grad_constraint=rg,
        ).face_grad()
    )(profile_2d, left_value, left_grad, right_value, right_grad)
    if squeeze:
        return grads[0]
    return grads


def build_face_transport_state(
    state: TransportState,
    geometry: Any,
    *,
    bc_density: Any = None,
    bc_temperature: Any = None,
    bc_er: Any = None,
    reconstruction: str = "linear",
    density_floor: Any = DEFAULT_TRANSPORT_DENSITY_FLOOR,
    temperature_floor: Any = DEFAULT_TRANSPORT_TEMPERATURE_FLOOR,
) -> FaceTransportState:
    state = apply_transport_density_floor(state, density_floor)
    state = apply_transport_temperature_floor(state, temperature_floor, density_floor)
    density_faces = _face_profile(
        state.density,
        geometry.r_grid_half,
        bc_model=bc_density,
        reconstruction=reconstruction,
    )
    density_faces = safe_density(density_faces, density_floor)
    temperature_faces = _face_profile(
        state.temperature,
        geometry.r_grid_half,
        bc_model=bc_temperature,
        reconstruction=reconstruction,
    )
    temperature_faces = safe_temperature(temperature_faces, temperature_floor)
    pressure_faces = density_faces * temperature_faces
    er_faces = _face_profile(
        state.Er,
        geometry.r_grid_half,
        bc_model=bc_er,
        reconstruction=reconstruction,
    )
    return FaceTransportState(
        density=density_faces,
        pressure=pressure_faces,
        Er=er_faces,
    )


def _ntss_like_face_profile(profile, face_centers, bc_model=None, density_floor=None):
    if profile.ndim == 1:
        profile_2d = profile[None, :]
        squeeze = True
    else:
        profile_2d = profile
        squeeze = False
    left_value, _left_grad, right_value, _right_grad = _extract_face_constraints(
        bc_model,
        profile_2d,
        face_centers,
    )
    if left_value is None:
        left_value = profile_2d[:, 0]
    if right_value is None:
        right_value = profile_2d[:, -1]
    cell_centers = 0.5 * (face_centers[1:] + face_centers[:-1])
    inner = 0.5 * (profile_2d[:, :-1] + profile_2d[:, 1:])
    faces = jnp.concatenate([left_value[..., None], inner, right_value[..., None]], axis=-1)
    if density_floor is not None:
        faces = safe_density(faces, density_floor)
    if squeeze:
        return faces[0]
    return faces


def _ntss_like_face_gradient(profile, face_centers, bc_model=None):
    if profile.ndim == 1:
        profile_2d = profile[None, :]
        squeeze = True
    else:
        profile_2d = profile
        squeeze = False

    reference = _face_profile_gradient(profile_2d, face_centers, bc_model=bc_model)
    cell_centers = 0.5 * (face_centers[1:] + face_centers[:-1])
    inner = (profile_2d[:, 1:] - profile_2d[:, :-1]) / (cell_centers[1:] - cell_centers[:-1])
    grads = reference.at[:, 1:-1].set(inner)
    if squeeze:
        return grads[0]
    return grads


def build_ntss_like_face_transport_state(
    state: TransportState,
    geometry: Any,
    *,
    bc_density: Any = None,
    bc_temperature: Any = None,
    bc_er: Any = None,
    density_floor: Any = DEFAULT_TRANSPORT_DENSITY_FLOOR,
    temperature_floor: Any = DEFAULT_TRANSPORT_TEMPERATURE_FLOOR,
) -> FaceTransportState:
    state = apply_transport_density_floor(state, density_floor)
    state = apply_transport_temperature_floor(state, temperature_floor, density_floor)
    density_faces = _ntss_like_face_profile(
        state.density,
        geometry.r_grid_half,
        bc_model=bc_density,
        density_floor=density_floor,
    )
    temperature_faces = _ntss_like_face_profile(
        state.temperature,
        geometry.r_grid_half,
        bc_model=bc_temperature,
    )
    temperature_faces = safe_temperature(temperature_faces, temperature_floor)
    pressure_faces = density_faces * temperature_faces
    er_faces = _ntss_like_face_profile(
        state.Er,
        geometry.r_grid_half,
        bc_model=bc_er,
    )
    return FaceTransportState(
        density=density_faces,
        pressure=pressure_faces,
        Er=er_faces,
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class CombinedTransportLaggedResponse:
    neoclassical_response: object = dataclasses.field(repr=False)
    turbulent_response: object = dataclasses.field(repr=False)
    classical_response: object = dataclasses.field(repr=False)


@dataclasses.dataclass(frozen=True, eq=False)
class CombinedTransportFluxModel(TransportFluxModelBase):
    neoclassical_model: TransportFluxModelBase
    turbulent_model: TransportFluxModelBase
    classical_model: TransportFluxModelBase
    include_turbulent_particle_flux: bool = True

    @staticmethod
    def _zero_like_flux(reference, fallback=0):
        if reference is not None:
            return jnp.zeros_like(jnp.asarray(reference))
        return fallback

    def __call__(self, state, *args, **kwargs) -> dict:
        # Only pass 'state' to the model instances, as expected by their __call__
        neo = self.neoclassical_model(state)
        turb = self.turbulent_model(state)
        classical = self.classical_model(state)
        gamma_turb = (
            turb.get("Gamma", 0)
            if self.include_turbulent_particle_flux
            else self._zero_like_flux(
                turb.get("Gamma", None),
                self._zero_like_flux(neo.get("Gamma", None), self._zero_like_flux(classical.get("Gamma", None), 0)),
            )
        )
        out = {
            "Gamma": neo.get("Gamma", 0) + gamma_turb + classical.get("Gamma", 0),
            "Q":     neo.get("Q", 0)     + turb.get("Q", 0)     + classical.get("Q", 0),
            "Upar":  neo.get("Upar", 0)  + turb.get("Upar", 0)  + classical.get("Upar", 0),
            "Gamma_neo": neo.get("Gamma", 0),
            "Q_neo":     neo.get("Q", 0),
            "Upar_neo":  neo.get("Upar", 0),
            "Gamma_turb": gamma_turb,
            "Q_turb":     turb.get("Q", 0),
            "Upar_turb":  turb.get("Upar", 0),
            "Gamma_classical": classical.get("Gamma", 0),
            "Q_classical":     classical.get("Q", 0),
            "Upar_classical":  classical.get("Upar", 0),
        }
        return out

    def build_local_particle_flux_evaluator(self, state):
        neo_eval = self.neoclassical_model.build_local_particle_flux_evaluator(state)
        turb_eval = self.turbulent_model.build_local_particle_flux_evaluator(state)
        classical_eval = self.classical_model.build_local_particle_flux_evaluator(state)
        if neo_eval is None or turb_eval is None or classical_eval is None:
            return None

        def evaluator(radius_index, er_value):
            gamma_turb = (
                turb_eval(radius_index, er_value)
                if self.include_turbulent_particle_flux
                else jnp.zeros_like(jnp.asarray(neo_eval(radius_index, er_value)))
            )
            return neo_eval(radius_index, er_value) + gamma_turb + classical_eval(radius_index, er_value)

        return evaluator

    def evaluate_face_fluxes(self, state, face_state, **kwargs):
        neo = self.neoclassical_model.evaluate_face_fluxes(state, face_state, **kwargs)
        turb = self.turbulent_model.evaluate_face_fluxes(state, face_state, **kwargs)
        classical = self.classical_model.evaluate_face_fluxes(state, face_state, **kwargs)
        if neo is None or turb is None or classical is None:
            return None
        gamma_turb = (
            turb.get("Gamma", 0)
            if self.include_turbulent_particle_flux
            else self._zero_like_flux(
                turb.get("Gamma", None),
                self._zero_like_flux(neo.get("Gamma", None), self._zero_like_flux(classical.get("Gamma", None), 0)),
            )
        )
        return {
            "Gamma": neo.get("Gamma", 0) + gamma_turb + classical.get("Gamma", 0),
            "Q": neo.get("Q", 0) + turb.get("Q", 0) + classical.get("Q", 0),
            "Upar": neo.get("Upar", 0) + turb.get("Upar", 0) + classical.get("Upar", 0),
            "Gamma_neo": neo.get("Gamma", 0),
            "Q_neo": neo.get("Q", 0),
            "Upar_neo": neo.get("Upar", 0),
            "Gamma_turb": gamma_turb,
            "Q_turb": turb.get("Q", 0),
            "Upar_turb": turb.get("Upar", 0),
            "Gamma_classical": classical.get("Gamma", 0),
            "Q_classical": classical.get("Q", 0),
            "Upar_classical": classical.get("Upar", 0),
        }

    def build_lagged_response(self, state, **kwargs):
        return CombinedTransportLaggedResponse(
            neoclassical_response=self.neoclassical_model.build_lagged_response(state, **kwargs),
            turbulent_response=self.turbulent_model.build_lagged_response(state, **kwargs),
            classical_response=self.classical_model.build_lagged_response(state, **kwargs),
        )

    def evaluate_with_lagged_response(self, state, lagged_response, **kwargs):
        neo = (
            self.neoclassical_model(state)
            if lagged_response.neoclassical_response is None
            else self.neoclassical_model.evaluate_with_lagged_response(
                state,
                lagged_response.neoclassical_response,
                **kwargs,
            )
        )
        turb = (
            self.turbulent_model(state)
            if lagged_response.turbulent_response is None
            else self.turbulent_model.evaluate_with_lagged_response(
                state,
                lagged_response.turbulent_response,
                **kwargs,
            )
        )
        classical = (
            self.classical_model(state)
            if lagged_response.classical_response is None
            else self.classical_model.evaluate_with_lagged_response(
                state,
                lagged_response.classical_response,
                **kwargs,
            )
        )
        gamma_turb = (
            turb.get("Gamma", 0)
            if self.include_turbulent_particle_flux
            else self._zero_like_flux(
                turb.get("Gamma", None),
                self._zero_like_flux(neo.get("Gamma", None), self._zero_like_flux(classical.get("Gamma", None), 0)),
            )
        )
        return {
            "Gamma": neo.get("Gamma", 0) + gamma_turb + classical.get("Gamma", 0),
            "Q": neo.get("Q", 0) + turb.get("Q", 0) + classical.get("Q", 0),
            "Upar": neo.get("Upar", 0) + turb.get("Upar", 0) + classical.get("Upar", 0),
            "Gamma_neo": neo.get("Gamma", 0),
            "Q_neo": neo.get("Q", 0),
            "Upar_neo": neo.get("Upar", 0),
            "Gamma_turb": gamma_turb,
            "Q_turb": turb.get("Q", 0),
            "Upar_turb": turb.get("Upar", 0),
            "Gamma_classical": classical.get("Gamma", 0),
            "Q_classical": classical.get("Q", 0),
            "Upar_classical": classical.get("Upar", 0),
        }




@dataclasses.dataclass(frozen=True, eq=False)
class NTXDatabaseTransportModel(TransportFluxModelBase):
    species: Any
    energy_grid: Any
    geometry: Any
    database: Any
    collisionality_model: str = "default"
    bc_density: Any = None
    bc_temperature: Any = None

    def __call__(self, state) -> dict:
        density = safe_density(state.density)
        _, gamma_neo, q_neo, upar_neo = get_Neoclassical_Fluxes(
            self.species,
            self.energy_grid,
            self.geometry,
            self.database,
            state.Er,
            state.temperature,
            density,
            collisionality_model=self.collisionality_model,
        )
        return {
            "Gamma": gamma_neo,
            "Q": q_neo,
            "Upar": upar_neo,
        }

    def build_local_particle_flux_evaluator(self, state):
        species = self.species
        energy_grid = self.energy_grid
        geometry = self.geometry
        database = self.database
        density = safe_density(state.density)
        temperature = state.temperature
        density_right_constraint, density_right_grad_constraint = _extract_right_constraints(
            self.bc_density,
            density,
        )
        temperature_right_constraint, temperature_right_grad_constraint = _extract_right_constraints(
            self.bc_temperature,
            temperature,
        )

        def evaluator(radius_index, er_value):
            er_scalar = jnp.asarray(er_value, dtype=state.Er.dtype)
            er_profile = state.Er.at[radius_index].set(er_scalar)
            _, gamma_neo, _, _ = get_Neoclassical_Fluxes(
                species,
                energy_grid,
                geometry,
                database,
                er_profile,
                temperature,
                density,
                density_right_constraint=density_right_constraint,
                density_right_grad_constraint=density_right_grad_constraint,
                temperature_right_constraint=temperature_right_constraint,
                temperature_right_grad_constraint=temperature_right_grad_constraint,
                collisionality_model=self.collisionality_model,
            )
            return gamma_neo[:, radius_index]

        return evaluator

    def evaluate_face_fluxes(self, state, face_state, **kwargs):
        density = safe_density(state.density)
        face_density = safe_density(face_state.density)
        bc_density = kwargs.get("bc_density")
        bc_temperature = kwargs.get("bc_temperature")
        particle_face_closure_mode = str(kwargs.get("particle_face_closure_mode", "reconstructed")).strip().lower()
        if particle_face_closure_mode in {"ntss_like", "ntss", "half_point"}:
            dndr_faces = _ntss_like_face_gradient(
                density,
                self.geometry.r_grid_half,
                bc_model=bc_density,
            )
            dTdr_faces = _ntss_like_face_gradient(
                state.temperature,
                self.geometry.r_grid_half,
                bc_model=bc_temperature,
            )
        else:
            dndr_faces = _face_profile_gradient(
                density,
                self.geometry.r_grid_half,
                bc_model=bc_density,
            )
            dTdr_faces = _face_profile_gradient(
                state.temperature,
                self.geometry.r_grid_half,
                bc_model=bc_temperature,
            )
        _, gamma_neo, q_neo, upar_neo = get_Neoclassical_Fluxes_Faces(
            self.species,
            self.energy_grid,
            self.geometry,
            self.database,
            face_state.Er,
            face_state.temperature,
            face_density,
            dndr_faces,
            dTdr_faces,
            collisionality_model=self.collisionality_model,
        )
        return {
            "Gamma": gamma_neo,
            "Q": q_neo,
            "Upar": upar_neo,
        }


def _as_float_array(value, *, name: str, positive: bool = False) -> jax.Array:
    arr = jnp.asarray(value, dtype=jnp.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional list/array.")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one value.")
    if not bool(jnp.all(jnp.isfinite(arr))):
        raise ValueError(f"{name} contains non-finite values.")
    if positive and not bool(jnp.all(arr > 0.0)):
        raise ValueError(f"{name} values must be positive.")
    return arr


def _import_ntx():
    try:
        import ntx

        return ntx
    except ImportError:
        repo_root = Path(__file__).resolve().parents[2]
        ntx_src = repo_root / "NTX" / "src"
        if ntx_src.is_dir() and str(ntx_src) not in sys.path:
            sys.path.insert(0, str(ntx_src))
        import ntx

        return ntx


def _load_ntx_vmec_boozer_channels(wout_path: Path, boozmn_path: Path, rho: jax.Array) -> dict[str, jax.Array | float]:
    from netCDF4 import Dataset
    import numpy as np
    import interpax

    rho = jnp.asarray(rho, dtype=jnp.float64)
    if rho.ndim != 1 or rho.shape[0] == 0:
        raise ValueError("ntx_scan_rho must be a non-empty one-dimensional array.")

    with Dataset(wout_path, mode="r") as vfile:
        ns = int(np.asarray(vfile.variables["ns"][:]).reshape(-1)[0])
        s_full = jnp.linspace(0.0, 1.0, ns)
        s_half = jnp.asarray([(i - 0.5) / (ns - 1) for i in range(ns)], dtype=jnp.float64)
        rho_half = jnp.sqrt(s_half)
        rho_full = jnp.sqrt(s_full)

        volume_p = float(np.asarray(vfile.variables["volume_p"][:]).reshape(-1)[-1])
        phi = np.asarray(vfile.variables["phi"][:], dtype=float)
        iotaf = np.asarray(vfile.variables["iotaf"][:], dtype=float)
        psia = float(jnp.abs(phi[-1]) / (2.0 * jnp.pi))

    with Dataset(boozmn_path, mode="r") as bfile:
        bmnc_b = np.asarray(bfile.variables["bmnc_b"][:], dtype=float)
        rmnc_b = np.asarray(bfile.variables["rmnc_b"][:], dtype=float)
        xm_b = np.asarray(bfile.variables["ixm_b"][:], dtype=float)
        xn_b = np.asarray(bfile.variables["ixn_b"][:], dtype=float)
        buco = np.asarray(bfile.variables["buco_b"][:], dtype=float)
        bvco = np.asarray(bfile.variables["bvco_b"][:], dtype=float)

    zero_mode = np.where((xm_b == 0) & (xn_b == 0))[0]
    if zero_mode.size == 0:
        raise ValueError("Could not find Boozer (m,n)=(0,0) mode in the boozmn file.")
    mode00 = int(zero_mode[0])

    r0_b = float(rmnc_b[-1, mode00])
    a_b = float(np.sqrt(volume_p / (2.0 * np.pi**2 * r0_b)))

    b00 = interpax.Interpolator1D(rho_half[1:], bmnc_b[:, mode00], extrap=True)
    r00 = interpax.Interpolator1D(rho_full[1:], rmnc_b[:, mode00], extrap=True)
    boozer_i = interpax.Interpolator1D(rho_half[1:], buco[1:], extrap=True)
    boozer_g = interpax.Interpolator1D(rho_half[1:], bvco[1:], extrap=True)
    iota = interpax.Interpolator1D(rho_full, iotaf, extrap=True)

    b00_rho = b00(rho)
    r00_rho = r00(rho)
    i_rho = boozer_i(rho)
    g_rho = boozer_g(rho)
    iota_rho = iota(rho)

    dpsidrtilde = rho * a_b * b00_rho
    drds = a_b / (2.0 * rho)
    dr_tildedr = 2.0 * psia / (a_b**2 * b00_rho)
    dr_tildeds = dr_tildedr * drds

    boozer_jacobian = g_rho + iota_rho * i_rho
    sqrt_pi = jnp.sqrt(jnp.pi)
    fac_reference_to_sfincs_11 = 8.0 * boozer_jacobian * b00_rho * psia**2 / (sqrt_pi * g_rho**2)
    fac_reference_to_sfincs_31 = 4.0 * b00_rho * psia / (sqrt_pi * g_rho)
    fac_reference_to_sfincs_33 = -2.0 * b00_rho / (boozer_jacobian * sqrt_pi)
    fac_sfincs_to_dkes_11 = 1.0 / (
        8.0 * boozer_jacobian * dpsidrtilde**2 / (g_rho**2 * b00_rho * sqrt_pi)
    )
    fac_sfincs_to_dkes_31 = 1.0 / (4.0 * dpsidrtilde / (g_rho * sqrt_pi))
    fac_sfincs_to_dkes_33 = 1.0 / (-2.0 * b00_rho / (boozer_jacobian * sqrt_pi))

    epsilon_t = rho * a_b / r00_rho
    fac_dkes_to_d11star = -(8.0 / jnp.pi) * iota_rho * r00_rho
    fac_dkes_to_d31star = -(3.0 / 1.46) * iota_rho * jnp.sqrt(epsilon_t) / 2.0
    fac_dkes_to_d33star = jnp.asarray(1.0, dtype=jnp.float64)

    return {
        "a_b": a_b,
        "psia": psia,
        "b00": b00_rho,
        "r00": r00_rho,
        "boozer_i": i_rho,
        "boozer_g": g_rho,
        "iota": iota_rho,
        "drds": drds,
        "dr_tildedr": dr_tildedr,
        "dr_tildeds": dr_tildeds,
        "fac_reference_to_sfincs_11": fac_reference_to_sfincs_11,
        "fac_reference_to_sfincs_31": fac_reference_to_sfincs_31,
        "fac_reference_to_sfincs_33": fac_reference_to_sfincs_33,
        "fac_sfincs_to_dkes_11": fac_sfincs_to_dkes_11,
        "fac_sfincs_to_dkes_31": fac_sfincs_to_dkes_31,
        "fac_sfincs_to_dkes_33": fac_sfincs_to_dkes_33,
        "fac_dkes_to_d11star": fac_dkes_to_d11star,
        "fac_dkes_to_d31star": fac_dkes_to_d31star,
        "fac_dkes_to_d33star": fac_dkes_to_d33star,
    }


def _build_ntx_field_channels(rho: jax.Array, er_tilde: jax.Array, channels: dict[str, jax.Array | float]) -> tuple[jax.Array, jax.Array, jax.Array]:
    b00 = jnp.asarray(channels["b00"], dtype=jnp.float64)
    dr_tildedr = jnp.asarray(channels["dr_tildedr"], dtype=jnp.float64)
    dr_tildeds = jnp.asarray(channels["dr_tildeds"], dtype=jnp.float64)
    er = er_tilde[None, :] * dr_tildedr[:, None] * b00[:, None]
    es = er_tilde[None, :] * dr_tildeds[:, None] * b00[:, None]
    er_to_ertilde = jnp.broadcast_to(1.0 / dr_tildedr[:, None], er.shape)
    return er, es, er_to_ertilde


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class NTXRuntimeScanChannels:
    rho: Any
    a_b: float
    psia: float
    b00: Any
    r00: Any
    boozer_i: Any
    boozer_g: Any
    iota: Any
    drds: Any
    dr_tildedr: Any
    dr_tildeds: Any
    fac_reference_to_sfincs_11: Any
    fac_reference_to_sfincs_31: Any
    fac_reference_to_sfincs_33: Any
    fac_sfincs_to_dkes_11: Any
    fac_sfincs_to_dkes_31: Any
    fac_sfincs_to_dkes_33: Any
    fac_dkes_to_d11star: Any
    fac_dkes_to_d31star: Any
    fac_dkes_to_d33star: Any

    @classmethod
    def from_mapping(cls, rho, channels: dict[str, jax.Array | float]) -> "NTXRuntimeScanChannels":
        rho = _as_float_array(rho, name="rho_scan")
        return cls(
            rho=rho,
            a_b=float(channels["a_b"]),
            psia=float(channels["psia"]),
            b00=jnp.asarray(channels["b00"], dtype=jnp.float64),
            r00=jnp.asarray(channels["r00"], dtype=jnp.float64),
            boozer_i=jnp.asarray(channels["boozer_i"], dtype=jnp.float64),
            boozer_g=jnp.asarray(channels["boozer_g"], dtype=jnp.float64),
            iota=jnp.asarray(channels["iota"], dtype=jnp.float64),
            drds=jnp.asarray(channels["drds"], dtype=jnp.float64),
            dr_tildedr=jnp.asarray(channels["dr_tildedr"], dtype=jnp.float64),
            dr_tildeds=jnp.asarray(channels["dr_tildeds"], dtype=jnp.float64),
            fac_reference_to_sfincs_11=jnp.asarray(channels["fac_reference_to_sfincs_11"], dtype=jnp.float64),
            fac_reference_to_sfincs_31=jnp.asarray(channels["fac_reference_to_sfincs_31"], dtype=jnp.float64),
            fac_reference_to_sfincs_33=jnp.asarray(channels["fac_reference_to_sfincs_33"], dtype=jnp.float64),
            fac_sfincs_to_dkes_11=jnp.asarray(channels["fac_sfincs_to_dkes_11"], dtype=jnp.float64),
            fac_sfincs_to_dkes_31=jnp.asarray(channels["fac_sfincs_to_dkes_31"], dtype=jnp.float64),
            fac_sfincs_to_dkes_33=jnp.asarray(channels["fac_sfincs_to_dkes_33"], dtype=jnp.float64),
            fac_dkes_to_d11star=jnp.asarray(channels["fac_dkes_to_d11star"], dtype=jnp.float64),
            fac_dkes_to_d31star=jnp.asarray(channels["fac_dkes_to_d31star"], dtype=jnp.float64),
            fac_dkes_to_d33star=jnp.asarray(channels["fac_dkes_to_d33star"], dtype=jnp.float64),
        )

    def as_mapping(self) -> dict[str, jax.Array | float]:
        return {
            "a_b": self.a_b,
            "psia": self.psia,
            "b00": self.b00,
            "r00": self.r00,
            "boozer_i": self.boozer_i,
            "boozer_g": self.boozer_g,
            "iota": self.iota,
            "drds": self.drds,
            "dr_tildedr": self.dr_tildedr,
            "dr_tildeds": self.dr_tildeds,
            "fac_reference_to_sfincs_11": self.fac_reference_to_sfincs_11,
            "fac_reference_to_sfincs_31": self.fac_reference_to_sfincs_31,
            "fac_reference_to_sfincs_33": self.fac_reference_to_sfincs_33,
            "fac_sfincs_to_dkes_11": self.fac_sfincs_to_dkes_11,
            "fac_sfincs_to_dkes_31": self.fac_sfincs_to_dkes_31,
            "fac_sfincs_to_dkes_33": self.fac_sfincs_to_dkes_33,
            "fac_dkes_to_d11star": self.fac_dkes_to_d11star,
            "fac_dkes_to_d31star": self.fac_dkes_to_d31star,
            "fac_dkes_to_d33star": self.fac_dkes_to_d33star,
        }


def build_ntx_runtime_scan_channels(vmec_file, boozer_file, rho_scan) -> NTXRuntimeScanChannels:
    rho = _as_float_array(rho_scan, name="rho_scan")
    channels = _load_ntx_vmec_boozer_channels(Path(vmec_file), Path(boozer_file), rho)
    return NTXRuntimeScanChannels.from_mapping(rho, channels)


def _build_ntx_surface_loader(vmec_file, boozer_file, surface_backend="auto"):
    ntx = _import_ntx()
    backend = str(surface_backend).strip().lower()
    vmec_file = str(vmec_file)
    boozer_file = str(boozer_file)

    def load(rho_value: float):
        if backend == "vmec":
            return ntx.surface_from_vmec_jax_vmec_wout_file(vmec_file, s=float(rho_value**2))
        if backend == "boozmn":
            return ntx.load_boozmn_surface(boozer_file, rho=float(rho_value)).surface
        if backend == "auto":
            try:
                return ntx.load_boozmn_surface(boozer_file, rho=float(rho_value)).surface
            except Exception:
                return ntx.surface_from_vmec_jax_vmec_wout_file(vmec_file, s=float(rho_value**2))
        raise ValueError("surface_backend must be one of: auto, boozmn, vmec")

    return ntx, load


def build_ntx_runtime_surfaces(vmec_file, boozer_file, rho_values, *, surface_backend="auto") -> tuple[Any, ...]:
    rho_arr = _as_float_array(rho_values, name="rho_values")
    _, loader = _build_ntx_surface_loader(vmec_file, boozer_file, surface_backend=surface_backend)
    return tuple(loader(float(rho_value)) for rho_value in rho_arr)


@dataclasses.dataclass(frozen=True, eq=False)
class NTXExactLijRuntimeSupport:
    center_channels: NTXRuntimeScanChannels
    face_channels: NTXRuntimeScanChannels
    center_prepared: Any
    face_prepared: Any
    grid: Any


@dataclasses.dataclass(frozen=True, eq=False)
class NTXRuntimeScanTransportModel(TransportFluxModelBase):
    species: Any
    energy_grid: Any
    geometry: Any
    vmec_file: str | None
    boozer_file: str | None
    rho_scan: Any
    nu_v_scan: Any
    er_tilde_scan: Any
    n_theta: int = 25
    n_zeta: int = 25
    n_xi: int = 64
    surface_backend: str = "auto"
    source_name: str = "ntx_scan_runtime"
    collisionality_model: str = "default"
    bc_density: Any = None
    bc_temperature: Any = None
    channels: NTXRuntimeScanChannels | None = None
    database: Any = None

    def _scan_axes(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        rho = _as_float_array(self.rho_scan, name="rho_scan")
        nu_v = _as_float_array(self.nu_v_scan, name="nu_v_scan", positive=True)
        er_tilde = _as_float_array(self.er_tilde_scan, name="er_tilde_scan")
        if not bool(jnp.all((rho > 0.0) & (rho <= 1.0))):
            raise ValueError("rho_scan values must satisfy 0 < rho <= 1.")
        return rho, nu_v, er_tilde

    def _static_channels(self) -> NTXRuntimeScanChannels:
        rho, _, _ = self._scan_axes()
        if self.channels is not None:
            if self.channels.rho.shape != rho.shape or not bool(jnp.allclose(self.channels.rho, rho)):
                raise ValueError("Provided ntx_scan_channels rho grid does not match rho_scan.")
            return self.channels
        if self.vmec_file is None or self.boozer_file is None:
            raise ValueError("vmec_file and boozer_file are required when ntx_scan_channels are not provided.")
        return build_ntx_runtime_scan_channels(self.vmec_file, self.boozer_file, rho)

    def _surface_loader(self, ntx):
        del ntx
        if self.vmec_file is None or self.boozer_file is None:
            raise ValueError("vmec_file and boozer_file are required to build NTX runtime scan surfaces.")
        _, loader = _build_ntx_surface_loader(self.vmec_file, self.boozer_file, surface_backend=self.surface_backend)
        return loader

    def _build_runtime_database(self):
        if self.database is not None:
            return self.database

        ntx = _import_ntx()
        rho, nu_v, er_tilde = self._scan_axes()
        static_channels = self._static_channels()
        channels = static_channels.as_mapping()
        er, es, er_to_ertilde = _build_ntx_field_channels(rho, er_tilde, channels)
        grid = ntx.GridSpec(
            n_theta=int(self.n_theta),
            n_zeta=int(self.n_zeta),
            n_xi=int(self.n_xi),
        )
        scan = ntx.build_ntx_neopax_scan(
            self._surface_loader(ntx),
            rho=rho,
            nu_v=nu_v,
            Es=es,
            Er=er,
            drds=jnp.asarray(channels["drds"], dtype=jnp.float64),
            grid=grid,
            source_name=self.source_name,
        )
        scan = dataclasses.replace(
            scan,
            Er_tilde=er_tilde,
            Er_to_Ertilde=er_to_ertilde,
            dr_tildedr=jnp.asarray(channels["dr_tildedr"], dtype=jnp.float64),
            dr_tildeds=jnp.asarray(channels["dr_tildeds"], dtype=jnp.float64),
            a_b=float(channels["a_b"]),
            psia=float(channels["psia"]),
            b00=jnp.asarray(channels["b00"], dtype=jnp.float64),
            r00=jnp.asarray(channels["r00"], dtype=jnp.float64),
            boozer_i=jnp.asarray(channels["boozer_i"], dtype=jnp.float64),
            boozer_g=jnp.asarray(channels["boozer_g"], dtype=jnp.float64),
            iota=jnp.asarray(channels["iota"], dtype=jnp.float64),
            fac_reference_to_sfincs_11=jnp.asarray(channels["fac_reference_to_sfincs_11"], dtype=jnp.float64),
            fac_reference_to_sfincs_31=jnp.asarray(channels["fac_reference_to_sfincs_31"], dtype=jnp.float64),
            fac_reference_to_sfincs_33=jnp.asarray(channels["fac_reference_to_sfincs_33"], dtype=jnp.float64),
            fac_monkes_to_sfincs_11=jnp.asarray(channels["fac_reference_to_sfincs_11"], dtype=jnp.float64),
            fac_monkes_to_sfincs_31=jnp.asarray(channels["fac_reference_to_sfincs_31"], dtype=jnp.float64),
            fac_monkes_to_sfincs_33=jnp.asarray(channels["fac_reference_to_sfincs_33"], dtype=jnp.float64),
            fac_sfincs_to_dkes_11=jnp.asarray(channels["fac_sfincs_to_dkes_11"], dtype=jnp.float64),
            fac_sfincs_to_dkes_31=jnp.asarray(channels["fac_sfincs_to_dkes_31"], dtype=jnp.float64),
            fac_sfincs_to_dkes_33=jnp.asarray(channels["fac_sfincs_to_dkes_33"], dtype=jnp.float64),
            fac_dkes_to_d11star=jnp.asarray(channels["fac_dkes_to_d11star"], dtype=jnp.float64),
            fac_dkes_to_d31star=jnp.asarray(channels["fac_dkes_to_d31star"], dtype=jnp.float64),
            fac_dkes_to_d33star=jnp.asarray(channels["fac_dkes_to_d33star"], dtype=jnp.float64),
        )
        print(
            "[NEOPAX] built runtime NTX scan database: "
            f"rho={int(rho.shape[0])} nu_v={int(nu_v.shape[0])} "
            f"Er_tilde={int(er_tilde.shape[0])} "
            f"grid=({grid.n_theta},{grid.n_zeta},{grid.n_xi}) backend={str(self.surface_backend).strip().lower()}"
        )
        return ntx.to_neopax_monoenergetic(scan, a_b=float(channels["a_b"]))

    def with_static_channels(self) -> "NTXRuntimeScanTransportModel":
        if self.channels is not None:
            return self
        return dataclasses.replace(self, channels=self._static_channels())

    def with_scan_inputs(
        self,
        *,
        rho_scan=None,
        nu_v_scan=None,
        er_tilde_scan=None,
        clear_database: bool = True,
    ) -> "NTXRuntimeScanTransportModel":
        new_rho = self.rho_scan if rho_scan is None else rho_scan
        new_nu_v = self.nu_v_scan if nu_v_scan is None else nu_v_scan
        new_er_tilde = self.er_tilde_scan if er_tilde_scan is None else er_tilde_scan

        new_channels = self.channels
        if self.channels is not None and rho_scan is not None:
            old_rho = _as_float_array(self.rho_scan, name="rho_scan")
            candidate_rho = _as_float_array(new_rho, name="rho_scan")
            same_rho = old_rho.shape == candidate_rho.shape and bool(jnp.allclose(old_rho, candidate_rho))
            if not same_rho:
                new_channels = None

        return dataclasses.replace(
            self,
            rho_scan=new_rho,
            nu_v_scan=new_nu_v,
            er_tilde_scan=new_er_tilde,
            channels=new_channels,
            database=None if clear_database else self.database,
        )

    def _database_model(self) -> NTXDatabaseTransportModel:
        return NTXDatabaseTransportModel(
            species=self.species,
            energy_grid=self.energy_grid,
            geometry=self.geometry,
            database=self._build_runtime_database(),
            collisionality_model=self.collisionality_model,
            bc_density=self.bc_density,
            bc_temperature=self.bc_temperature,
        )

    def __call__(self, state) -> dict:
        return self._database_model()(state)

    def build_local_particle_flux_evaluator(self, state):
        return self._database_model().build_local_particle_flux_evaluator(state)

    def evaluate_face_fluxes(self, state, face_state, **kwargs):
        return self._database_model().evaluate_face_fluxes(state, face_state, **kwargs)

    def with_runtime_database(self) -> "NTXRuntimeScanTransportModel":
        if self.database is not None:
            return self
        model = self.with_static_channels()
        return dataclasses.replace(model, database=model._build_runtime_database())


def build_ntx_exact_lij_runtime_support(
    vmec_file,
    boozer_file,
    rho_center,
    rho_face,
    *,
    surface_backend="auto",
    n_theta=25,
    n_zeta=25,
    n_xi=64,
) -> NTXExactLijRuntimeSupport:
    ntx = _import_ntx()
    grid_spec = ntx.GridSpec(n_theta=int(n_theta), n_zeta=int(n_zeta), n_xi=int(n_xi))
    center_channels = build_ntx_runtime_scan_channels(vmec_file, boozer_file, rho_center)
    face_channels = build_ntx_runtime_scan_channels(vmec_file, boozer_file, rho_face)
    center_surfaces = build_ntx_runtime_surfaces(
        vmec_file,
        boozer_file,
        center_channels.rho,
        surface_backend=surface_backend,
    )
    face_surfaces = build_ntx_runtime_surfaces(
        vmec_file,
        boozer_file,
        face_channels.rho,
        surface_backend=surface_backend,
    )

    def _stack_optional(*values):
        first = values[0]
        if first is None:
            return None
        return jnp.stack([jnp.asarray(value) for value in values], axis=0)

    center_prepared_tuple = tuple(ntx.prepare_monoenergetic_system(surface, grid_spec) for surface in center_surfaces)
    face_prepared_tuple = tuple(ntx.prepare_monoenergetic_system(surface, grid_spec) for surface in face_surfaces)
    center_prepared = jax.tree_util.tree_map(_stack_optional, *center_prepared_tuple)
    face_prepared = jax.tree_util.tree_map(_stack_optional, *face_prepared_tuple)
    return NTXExactLijRuntimeSupport(
        center_channels=center_channels,
        face_channels=face_channels,
        center_prepared=center_prepared,
        face_prepared=face_prepared,
        grid=grid_spec,
    )


@dataclasses.dataclass(frozen=True, eq=False)
class NTXExactLijRuntimeTransportModel(TransportFluxModelBase):
    species: Any
    energy_grid: Any
    geometry: Any
    vmec_file: str | None
    boozer_file: str | None
    n_theta: int = 25
    n_zeta: int = 25
    n_xi: int = 64
    surface_backend: str = "auto"
    face_response_mode: str = "face_local_response"
    radial_batch_size: int | None = None
    scan_batch_size: int | None = None
    use_remat: bool = False
    collisionality_model: str = "default"
    bc_density: Any = None
    bc_temperature: Any = None
    support: NTXExactLijRuntimeSupport | None = None

    def _rho_center_face(self):
        a_b = jnp.asarray(self.geometry.a_b, dtype=jnp.float64)
        rho_center = jnp.asarray(self.geometry.r_grid, dtype=jnp.float64) / a_b
        rho_face = jnp.asarray(self.geometry.r_grid_half, dtype=jnp.float64) / a_b
        return rho_center, rho_face

    def _static_support(self) -> NTXExactLijRuntimeSupport:
        if self.support is not None:
            return self.support
        if self.vmec_file is None or self.boozer_file is None:
            raise ValueError("vmec_file and boozer_file are required when ntx_exact_lij_support is not provided.")
        rho_center, rho_face = self._rho_center_face()
        return build_ntx_exact_lij_runtime_support(
            self.vmec_file,
            self.boozer_file,
            rho_center,
            rho_face,
            surface_backend=self.surface_backend,
            n_theta=self.n_theta,
            n_zeta=self.n_zeta,
            n_xi=self.n_xi,
        )

    def with_static_support(self) -> "NTXExactLijRuntimeTransportModel":
        if self.support is not None:
            return self
        return dataclasses.replace(self, support=self._static_support())

    def with_transport_resolution(self, *, n_theta=None, n_zeta=None, n_xi=None) -> "NTXExactLijRuntimeTransportModel":
        return dataclasses.replace(
            self,
            n_theta=self.n_theta if n_theta is None else int(n_theta),
            n_zeta=self.n_zeta if n_zeta is None else int(n_zeta),
            n_xi=self.n_xi if n_xi is None else int(n_xi),
            support=None,
        )

    def with_face_response_mode(self, face_response_mode: str) -> "NTXExactLijRuntimeTransportModel":
        return dataclasses.replace(self, face_response_mode=str(face_response_mode))

    def with_radial_batch_size(self, radial_batch_size: int | None) -> "NTXExactLijRuntimeTransportModel":
        normalized = None if radial_batch_size in (None, 0) else int(radial_batch_size)
        return dataclasses.replace(self, radial_batch_size=normalized)

    def with_scan_batch_size(self, scan_batch_size: int | None) -> "NTXExactLijRuntimeTransportModel":
        normalized = None if scan_batch_size in (None, 0) else int(scan_batch_size)
        return dataclasses.replace(self, scan_batch_size=normalized)

    def with_use_remat(self, use_remat: bool) -> "NTXExactLijRuntimeTransportModel":
        return dataclasses.replace(self, use_remat=bool(use_remat))

    def _map_radius_axis(self, fn, radius_indices):
        batch_size = self.radial_batch_size
        if batch_size is None or int(batch_size) <= 1:
            return jax.lax.map(fn, radius_indices)

        n_radius = int(radius_indices.shape[0])
        batch_size = int(batch_size)
        if batch_size >= n_radius:
            return jax.vmap(fn)(radius_indices)

        n_full = n_radius // batch_size
        remainder = n_radius % batch_size
        outputs = []

        if n_full > 0:
            chunked = radius_indices[: n_full * batch_size].reshape((n_full, batch_size))
            mapped = jax.lax.map(lambda chunk: jax.vmap(fn)(chunk), chunked)
            mapped = jax.tree_util.tree_map(
                lambda arr: arr.reshape((n_full * batch_size,) + arr.shape[2:]),
                mapped,
            )
            outputs.append(mapped)

        if remainder > 0:
            tail = radius_indices[n_full * batch_size :]
            outputs.append(jax.vmap(fn)(tail))

        if len(outputs) == 1:
            return outputs[0]
        return jax.tree_util.tree_map(lambda *parts: jnp.concatenate(parts, axis=0), *outputs)

    def _local_scan_inputs(
        self,
        *,
        drds_value,
        species_index: int,
        er_value,
        temperature_local,
        density_local,
        vthermal_local,
        collisionality_kind,
    ):
        vth_a = vthermal_local[species_index]
        v_new_a = self.energy_grid.v_norm * vth_a
        epsi_hat_a = (er_value * 1.0e3 / v_new_a) * drds_value
        nu_hat_a = _nu_over_vnew_local(
            self.species,
            species_index,
            v_new_a,
            density_local,
            temperature_local,
            vthermal_local,
            collisionality_kind,
        )
        return nu_hat_a, epsi_hat_a, vth_a

    def _lij_from_coefficient_scan(
        self,
        coeff_scan,
        *,
        drds_value,
        species_index: int,
        vth_a,
    ):
        transport_moments = self._transport_moments_from_coefficient_scan(
            coeff_scan,
            drds_value=drds_value,
        )
        return self._lij_from_transport_moments(
            transport_moments,
            species_index=species_index,
            vth_a=vth_a,
        )

    def _transport_moments_from_coefficient_scan(
        self,
        coeff_scan,
        *,
        drds_value,
    ):
        d11_a = -(jnp.asarray(coeff_scan[:, 0], dtype=jnp.float64) * drds_value**2)
        d13_a = -(jnp.asarray(coeff_scan[:, 2], dtype=jnp.float64) * drds_value)
        d33_a = -jnp.asarray(coeff_scan[:, 3], dtype=jnp.float64)
        weighted_l11 = self.energy_grid.L11_weight * self.energy_grid.xWeights
        weighted_l12 = self.energy_grid.L12_weight * self.energy_grid.xWeights
        weighted_l22 = self.energy_grid.L22_weight * self.energy_grid.xWeights
        weighted_l13 = self.energy_grid.L13_weight * self.energy_grid.xWeights
        weighted_l23 = self.energy_grid.L23_weight * self.energy_grid.xWeights
        weighted_l33 = self.energy_grid.L33_weight * self.energy_grid.xWeights
        return jnp.stack(
            [
                jnp.sum(weighted_l11 * d11_a),
                jnp.sum(weighted_l12 * d11_a),
                jnp.sum(weighted_l22 * d11_a),
                jnp.sum(weighted_l13 * d13_a),
                jnp.sum(weighted_l23 * d13_a),
                jnp.sum(weighted_l33 * d33_a),
            ],
            axis=0,
        )

    def _transport_moments_from_inputs_impl(self, prepared, nu_hat_a, epsi_hat_a, *, drds_value):
        coeff_scan = self._coefficient_scan_from_inputs(prepared, nu_hat_a, epsi_hat_a)
        return self._transport_moments_from_coefficient_scan(
            coeff_scan,
            drds_value=drds_value,
        )

    def _transport_moments_from_inputs(self, prepared, nu_hat_a, epsi_hat_a, *, drds_value):
        return self._transport_moments_from_inputs_impl(
            prepared,
            nu_hat_a,
            epsi_hat_a,
            drds_value=drds_value,
        )

    def _lij_from_transport_moments(
        self,
        transport_moments,
        *,
        species_index: int,
        vth_a,
    ):
        charge = self.species.charge[species_index]
        mass = self.species.mass[species_index]
        l11_fac = -1.0 / jnp.sqrt(jnp.pi) * (mass / charge) ** 2 * vth_a**3
        l13_fac = -1.0 / jnp.sqrt(jnp.pi) * (mass / charge) * vth_a**2
        l33_fac = -1.0 / jnp.sqrt(jnp.pi) * vth_a

        lij = jnp.zeros((3, 3), dtype=jnp.float64)
        lij = lij.at[0, 0].set(l11_fac * transport_moments[0])
        lij = lij.at[0, 1].set(l11_fac * transport_moments[1])
        lij = lij.at[1, 0].set(lij[0, 1])
        lij = lij.at[1, 1].set(l11_fac * transport_moments[2])
        lij = lij.at[0, 2].set(l13_fac * transport_moments[3])
        lij = lij.at[1, 2].set(l13_fac * transport_moments[4])
        lij = lij.at[2, 0].set(-lij[0, 2])
        lij = lij.at[2, 1].set(-lij[1, 2])
        lij = lij.at[2, 2].set(l33_fac * transport_moments[5])
        return lij

    def _batched_lij_from_transport_moments(self, transport_moments, v_thermal):
        charge = jnp.asarray(self.species.charge, dtype=jnp.float64)[:, None]
        mass = jnp.asarray(self.species.mass, dtype=jnp.float64)[:, None]
        inv_sqrt_pi = 1.0 / jnp.sqrt(jnp.pi)
        l11_fac = -inv_sqrt_pi * (mass / charge) ** 2 * v_thermal**3
        l13_fac = -inv_sqrt_pi * (mass / charge) * v_thermal**2
        l33_fac = -inv_sqrt_pi * v_thermal

        l00 = l11_fac * transport_moments[:, :, 0]
        l01 = l11_fac * transport_moments[:, :, 1]
        l11 = l11_fac * transport_moments[:, :, 2]
        l02 = l13_fac * transport_moments[:, :, 3]
        l12 = l13_fac * transport_moments[:, :, 4]
        l22 = l33_fac * transport_moments[:, :, 5]

        row0 = jnp.stack((l00, l01, l02), axis=-1)
        row1 = jnp.stack((l01, l11, l12), axis=-1)
        row2 = jnp.stack((-l02, -l12, l22), axis=-1)
        return jnp.stack((row0, row1, row2), axis=-2)

    def _solve_coefficient_scan_prepared_impl(self, prepared, nu_hat_a, epsi_hat_a):
        ntx = _import_ntx()

        def _solve_one(nu_hat_value, epsi_hat_value):
            case = ntx.MonoenergeticCase(nu_hat=nu_hat_value, epsi_hat=epsi_hat_value)
            return ntx.solve_prepared_coefficient_vector(prepared, case)
        batch_size = self.scan_batch_size
        case_count = int(nu_hat_a.shape[0])
        if batch_size is None or int(batch_size) <= 0 or int(batch_size) >= case_count:
            return jax.vmap(_solve_one)(nu_hat_a, epsi_hat_a)

        batch_size = int(batch_size)
        n_full = case_count // batch_size
        remainder = case_count % batch_size
        outputs = []

        if n_full > 0:
            nu_full = nu_hat_a[: n_full * batch_size].reshape((n_full, batch_size))
            epsi_full = epsi_hat_a[: n_full * batch_size].reshape((n_full, batch_size))
            full = jax.lax.map(
                lambda chunk: jax.vmap(_solve_one)(chunk[0], chunk[1]),
                (nu_full, epsi_full),
            )
            outputs.append(full.reshape((n_full * batch_size, -1)))

        if remainder > 0:
            outputs.append(
                jax.vmap(_solve_one)(
                    nu_hat_a[n_full * batch_size :],
                    epsi_hat_a[n_full * batch_size :],
                )
            )

        if len(outputs) == 1:
            return outputs[0]
        return jnp.concatenate(outputs, axis=0)

    def _solve_coefficient_scan_prepared(self, prepared, nu_hat_a, epsi_hat_a):
        evaluator = self._solve_coefficient_scan_prepared_impl
        if self.use_remat:
            evaluator = jax.checkpoint(evaluator)
        return evaluator(prepared, nu_hat_a, epsi_hat_a)

    def _coefficient_scan_from_inputs(self, prepared, nu_hat_a, epsi_hat_a):
        return self._solve_coefficient_scan_prepared(prepared, nu_hat_a, epsi_hat_a)

    def _solve_lij_prepared_local_impl(
        self,
        prepared,
        *,
        drds_value,
        species_index: int,
        er_value,
        temperature_local,
        density_local,
        vthermal_local,
        collisionality_kind,
    ):
        nu_hat_a, epsi_hat_a, vth_a = self._local_scan_inputs(
            drds_value=drds_value,
            species_index=species_index,
            er_value=er_value,
            temperature_local=temperature_local,
            density_local=density_local,
            vthermal_local=vthermal_local,
            collisionality_kind=collisionality_kind,
        )
        coeff_scan = self._solve_coefficient_scan_prepared(prepared, nu_hat_a, epsi_hat_a)
        return self._lij_from_coefficient_scan(
            coeff_scan,
            drds_value=drds_value,
            species_index=species_index,
            vth_a=vth_a,
        )

    def _solve_lij_prepared_local(
        self,
        prepared,
        *,
        drds_value,
        species_index: int,
        er_value,
        temperature_local,
        density_local,
        vthermal_local,
        collisionality_kind,
    ):
        return self._solve_lij_prepared_local_impl(
            prepared,
            drds_value=drds_value,
            species_index=species_index,
            er_value=er_value,
            temperature_local=temperature_local,
            density_local=density_local,
            vthermal_local=vthermal_local,
            collisionality_kind=collisionality_kind,
        )

    def _build_coefficient_response_local(
        self,
        prepared,
        *,
        drds_value,
        species_index: int,
        er_value,
        temperature_local,
        density_local,
        vthermal_local,
        collisionality_kind,
    ):
        ref_nu_hat, ref_epsi_hat, _ = self._local_scan_inputs(
            drds_value=drds_value,
            species_index=species_index,
            er_value=er_value,
            temperature_local=temperature_local,
            density_local=density_local,
            vthermal_local=vthermal_local,
            collisionality_kind=collisionality_kind,
        )
        reference_transport_moments = self._transport_moments_from_inputs(
            prepared,
            ref_nu_hat,
            ref_epsi_hat,
            drds_value=drds_value,
        )
        return NTXPreparedCoefficientResponse(
            reference_transport_moments=reference_transport_moments,
            reference_nu_hat=ref_nu_hat,
            reference_epsi_hat=ref_epsi_hat,
        )

    def _lij_center(self, Er, temperature, density):
        support = self._static_support()
        collisionality_kind = _collisionality_kind(self.collisionality_model)
        v_thermal = get_v_thermal(self.species.mass, temperature)
        species_indices = jnp.arange(int(self.species.number_species), dtype=jnp.int32)
        radius_indices = jnp.arange(Er.shape[0], dtype=jnp.int32)

        def _per_radius(radius_index):
            prepared = jax.tree_util.tree_map(
                lambda arr: jax.lax.dynamic_index_in_dim(arr, radius_index, axis=0, keepdims=False),
                support.center_prepared,
            )
            drds_value = jax.lax.dynamic_index_in_dim(support.center_channels.drds, radius_index, axis=0, keepdims=False)
            er_value = jax.lax.dynamic_index_in_dim(Er, radius_index, axis=0, keepdims=False)
            temperature_local = jax.lax.dynamic_index_in_dim(temperature, radius_index, axis=1, keepdims=False)
            density_local = jax.lax.dynamic_index_in_dim(density, radius_index, axis=1, keepdims=False)
            vthermal_local = jax.lax.dynamic_index_in_dim(v_thermal, radius_index, axis=1, keepdims=False)
            return jax.vmap(
                lambda species_index: self._solve_lij_prepared_local(
                    prepared,
                    drds_value=drds_value,
                    species_index=species_index,
                    er_value=er_value,
                    temperature_local=temperature_local,
                    density_local=density_local,
                    vthermal_local=vthermal_local,
                    collisionality_kind=collisionality_kind,
                )
            )(species_indices)

        lij_by_radius = self._map_radius_axis(_per_radius, radius_indices)
        return jnp.swapaxes(lij_by_radius, 0, 1)

    def _lij_faces(self, Er_faces, temperature_faces, density_faces):
        support = self._static_support()
        collisionality_kind = _collisionality_kind(self.collisionality_model)
        v_thermal_faces = get_v_thermal(self.species.mass, temperature_faces)
        species_indices = jnp.arange(int(self.species.number_species), dtype=jnp.int32)
        radius_indices = jnp.arange(Er_faces.shape[0], dtype=jnp.int32)

        def _per_radius(radius_index):
            prepared = jax.tree_util.tree_map(
                lambda arr: jax.lax.dynamic_index_in_dim(arr, radius_index, axis=0, keepdims=False),
                support.face_prepared,
            )
            drds_value = jax.lax.dynamic_index_in_dim(support.face_channels.drds, radius_index, axis=0, keepdims=False)
            er_value = jax.lax.dynamic_index_in_dim(Er_faces, radius_index, axis=0, keepdims=False)
            temperature_local = jax.lax.dynamic_index_in_dim(temperature_faces, radius_index, axis=1, keepdims=False)
            density_local = jax.lax.dynamic_index_in_dim(density_faces, radius_index, axis=1, keepdims=False)
            vthermal_local = jax.lax.dynamic_index_in_dim(v_thermal_faces, radius_index, axis=1, keepdims=False)
            return jax.vmap(
                lambda species_index: self._solve_lij_prepared_local(
                    prepared,
                    drds_value=drds_value,
                    species_index=species_index,
                    er_value=er_value,
                    temperature_local=temperature_local,
                    density_local=density_local,
                    vthermal_local=vthermal_local,
                    collisionality_kind=collisionality_kind,
                )
            )(species_indices)

        lij_by_radius = self._map_radius_axis(_per_radius, radius_indices)
        return jnp.swapaxes(lij_by_radius, 0, 1)

    def _assemble_center_fluxes(self, Er, temperature, density, lij, n_right, n_right_grad, t_right, t_right_grad):
        dndr_all = jax.vmap(
            lambda density_a, right_value, right_grad: get_gradient_density(
                density_a,
                self.geometry.r_grid,
                self.geometry.r_grid_half,
                self.geometry.dr,
                right_face_constraint=right_value,
                right_face_grad_constraint=right_grad,
            )
        )(density, n_right, n_right_grad)
        dTdr_all = jax.vmap(
            lambda temperature_a, right_value, right_grad: get_gradient_temperature(
                temperature_a,
                self.geometry.r_grid,
                self.geometry.r_grid_half,
                self.geometry.dr,
                right_face_constraint=right_value,
                right_face_grad_constraint=right_grad,
            )
        )(temperature, t_right, t_right_grad)
        a1 = jax.vmap(
            lambda charge, density_a, temperature_a, dndr_a, dTdr_a: get_Thermodynamical_Forces_A1(
                charge,
                density_a,
                temperature_a,
                dndr_a,
                dTdr_a,
                Er,
            )
        )(self.species.charge, density, temperature, dndr_all, dTdr_all)
        a2 = jax.vmap(get_Thermodynamical_Forces_A2)(temperature, dTdr_all)
        a3 = get_Thermodynamical_Forces_A3(Er)
        density_phys = DENSITY_STATE_TO_PHYSICAL * density
        temperature_phys = TEMPERATURE_STATE_TO_PHYSICAL * temperature
        gamma = -density_phys * (
            lij[:, :, 0, 0] * a1
            + lij[:, :, 0, 1] * a2
            + lij[:, :, 0, 2] * a3[None, :]
        )
        q = -temperature_phys * density_phys * (
            lij[:, :, 1, 0] * a1
            + lij[:, :, 1, 1] * a2
            + lij[:, :, 1, 2] * a3[None, :]
        )
        upar = -density_phys * (
            lij[:, :, 2, 0] * a1
            + lij[:, :, 2, 1] * a2
            + lij[:, :, 2, 2] * a3[None, :]
        )
        return gamma, q, upar

    def _cell_centered_flux_to_faces_centered(self, flux):
        if flux.ndim == 1:
            return faces_from_cell_centered(flux)
        return jax.vmap(faces_from_cell_centered)(flux)

    def __call__(self, state) -> dict:
        density = safe_density(state.density)
        temperature = state.temperature
        n_species = int(temperature.shape[0])
        n_right = _as_species_constraint(None if self.bc_density is None else getattr(self.bc_density, "right_value", None), n_species)
        if n_right is None:
            n_right = density[:, -1]
        n_right_grad = _as_species_constraint(None if self.bc_density is None else getattr(self.bc_density, "right_gradient", None), n_species)
        if n_right_grad is None:
            n_right_grad = jnp.zeros_like(n_right)
        t_right = _as_species_constraint(None if self.bc_temperature is None else getattr(self.bc_temperature, "right_value", None), n_species)
        if t_right is None:
            t_right = temperature[:, -1]
        t_right_grad = _as_species_constraint(None if self.bc_temperature is None else getattr(self.bc_temperature, "right_gradient", None), n_species)
        if t_right_grad is None:
            t_right_grad = jnp.zeros_like(t_right)

        lij = self._lij_center(state.Er, temperature, density)
        gamma, q, upar = self._assemble_center_fluxes(
            state.Er,
            temperature,
            density,
            lij,
            n_right,
            n_right_grad,
            t_right,
            t_right_grad,
        )
        return {"Gamma": gamma, "Q": q, "Upar": upar}

    def build_lagged_response(self, state, **kwargs):
        del kwargs
        density = safe_density(state.density)
        temperature = state.temperature
        support = self._static_support()
        collisionality_kind = _collisionality_kind(self.collisionality_model)
        v_thermal = get_v_thermal(self.species.mass, temperature)
        species_indices = jnp.arange(int(self.species.number_species), dtype=jnp.int32)
        radius_indices = jnp.arange(state.Er.shape[0], dtype=jnp.int32)

        def _per_radius(radius_index):
            prepared = jax.tree_util.tree_map(
                lambda arr: jax.lax.dynamic_index_in_dim(arr, radius_index, axis=0, keepdims=False),
                support.center_prepared,
            )
            drds_value = jax.lax.dynamic_index_in_dim(support.center_channels.drds, radius_index, axis=0, keepdims=False)
            er_value = jax.lax.dynamic_index_in_dim(state.Er, radius_index, axis=0, keepdims=False)
            temperature_local = jax.lax.dynamic_index_in_dim(temperature, radius_index, axis=1, keepdims=False)
            density_local = jax.lax.dynamic_index_in_dim(density, radius_index, axis=1, keepdims=False)
            vthermal_local = jax.lax.dynamic_index_in_dim(v_thermal, radius_index, axis=1, keepdims=False)
            return jax.vmap(
                lambda species_index: self._build_coefficient_response_local(
                    prepared,
                    drds_value=drds_value,
                    species_index=species_index,
                    er_value=er_value,
                    temperature_local=temperature_local,
                    density_local=density_local,
                    vthermal_local=vthermal_local,
                    collisionality_kind=collisionality_kind,
                )
            )(species_indices)

        response_by_radius = self._map_radius_axis(_per_radius, radius_indices)
        return NTXExactLijLaggedResponse(center_response=response_by_radius)

    def evaluate_with_lagged_response(self, state, lagged_response, **kwargs):
        del kwargs
        density = safe_density(state.density)
        temperature = state.temperature
        n_species = int(temperature.shape[0])
        n_right = _as_species_constraint(None if self.bc_density is None else getattr(self.bc_density, "right_value", None), n_species)
        if n_right is None:
            n_right = density[:, -1]
        n_right_grad = _as_species_constraint(None if self.bc_density is None else getattr(self.bc_density, "right_gradient", None), n_species)
        if n_right_grad is None:
            n_right_grad = jnp.zeros_like(n_right)
        t_right = _as_species_constraint(None if self.bc_temperature is None else getattr(self.bc_temperature, "right_value", None), n_species)
        if t_right is None:
            t_right = temperature[:, -1]
        t_right_grad = _as_species_constraint(None if self.bc_temperature is None else getattr(self.bc_temperature, "right_gradient", None), n_species)
        if t_right_grad is None:
            t_right_grad = jnp.zeros_like(t_right)

        support = self._static_support()
        collisionality_kind = _collisionality_kind(self.collisionality_model)
        v_thermal = get_v_thermal(self.species.mass, temperature)
        species_indices = jnp.arange(int(self.species.number_species), dtype=jnp.int32)
        center_response = lagged_response.center_response

        radius_indices = jnp.arange(state.Er.shape[0], dtype=jnp.int32)

        def _transport_moment_tangent_per_radius(radius_index):
            prepared = jax.tree_util.tree_map(
                lambda arr: jax.lax.dynamic_index_in_dim(arr, radius_index, axis=0, keepdims=False),
                support.center_prepared,
            )
            drds_value = jax.lax.dynamic_index_in_dim(support.center_channels.drds, radius_index, axis=0, keepdims=False)
            er_value = jax.lax.dynamic_index_in_dim(state.Er, radius_index, axis=0, keepdims=False)
            temperature_local = jax.lax.dynamic_index_in_dim(temperature, radius_index, axis=1, keepdims=False)
            density_local = jax.lax.dynamic_index_in_dim(density, radius_index, axis=1, keepdims=False)
            vthermal_local = jax.lax.dynamic_index_in_dim(v_thermal, radius_index, axis=1, keepdims=False)
            ref_nu_radius = jax.lax.dynamic_index_in_dim(center_response.reference_nu_hat, radius_index, axis=0, keepdims=False)
            ref_epsi_radius = jax.lax.dynamic_index_in_dim(center_response.reference_epsi_hat, radius_index, axis=0, keepdims=False)
            return jax.vmap(
                lambda species_index, ref_nu_species, ref_epsi_species: jax.jvp(
                    lambda nu_hat_a, epsi_hat_a: self._transport_moments_from_inputs(
                        prepared,
                        nu_hat_a,
                        epsi_hat_a,
                        drds_value=drds_value,
                    ),
                    (ref_nu_species, ref_epsi_species),
                    tuple(
                        current_value - reference_value
                        for current_value, reference_value in zip(
                            self._local_scan_inputs(
                                drds_value=drds_value,
                                species_index=species_index,
                                er_value=er_value,
                                temperature_local=temperature_local,
                                density_local=density_local,
                                vthermal_local=vthermal_local,
                                collisionality_kind=collisionality_kind,
                            )[:2],
                            (ref_nu_species, ref_epsi_species),
                        )
                    ),
                )[1]
            )(species_indices, ref_nu_radius, ref_epsi_radius)

        transport_moment_tangent_by_radius = self._map_radius_axis(_transport_moment_tangent_per_radius, radius_indices)
        transport_moments = (
            center_response.reference_transport_moments
            + transport_moment_tangent_by_radius
        )
        transport_moments = jnp.swapaxes(transport_moments, 0, 1)
        lij = self._batched_lij_from_transport_moments(transport_moments, v_thermal)
        gamma, q, upar = self._assemble_center_fluxes(
            state.Er,
            temperature,
            density,
            lij,
            n_right,
            n_right_grad,
            t_right,
            t_right_grad,
        )
        return {"Gamma": gamma, "Q": q, "Upar": upar}

    def build_local_particle_flux_evaluator(self, state):
        density = safe_density(state.density)
        temperature = state.temperature
        support = self._static_support()
        collisionality_kind = _collisionality_kind(self.collisionality_model)
        v_thermal = get_v_thermal(self.species.mass, temperature)
        species_indices = jnp.arange(int(self.species.number_species), dtype=jnp.int32)
        dndr_all = jax.vmap(
            lambda density_a: get_gradient_density(
                density_a,
                self.geometry.r_grid,
                self.geometry.r_grid_half,
                self.geometry.dr,
            )
        )(density)
        dTdr_all = jax.vmap(
            lambda temperature_a: get_gradient_temperature(
                temperature_a,
                self.geometry.r_grid,
                self.geometry.r_grid_half,
                self.geometry.dr,
            )
        )(temperature)

        def evaluator(radius_index, er_value):
            radius_index = jnp.asarray(radius_index, dtype=jnp.int32)
            er_scalar = jnp.asarray(er_value, dtype=state.Er.dtype)
            prepared = jax.tree_util.tree_map(
                lambda arr: jax.lax.dynamic_index_in_dim(arr, radius_index, axis=0, keepdims=False),
                support.center_prepared,
            )
            drds_value = jax.lax.dynamic_index_in_dim(support.center_channels.drds, radius_index, axis=0, keepdims=False)
            temperature_local = jax.lax.dynamic_index_in_dim(temperature, radius_index, axis=1, keepdims=False)
            density_local = jax.lax.dynamic_index_in_dim(density, radius_index, axis=1, keepdims=False)
            vthermal_local = jax.lax.dynamic_index_in_dim(v_thermal, radius_index, axis=1, keepdims=False)
            er_profile = state.Er.at[radius_index].set(er_scalar)
            lij = jax.vmap(
                lambda species_index: self._solve_lij_prepared_local(
                    prepared,
                    drds_value=drds_value,
                    species_index=species_index,
                    er_value=er_scalar,
                    temperature_local=temperature_local,
                    density_local=density_local,
                    vthermal_local=vthermal_local,
                    collisionality_kind=collisionality_kind,
                )
            )(species_indices)
            a1 = jax.vmap(
                lambda charge, density_a, temperature_a, dndr_a, dTdr_a: get_Thermodynamical_Forces_A1(
                    charge,
                    density_a,
                    temperature_a,
                    dndr_a,
                    dTdr_a,
                    er_profile,
                )
            )(self.species.charge, density, temperature, dndr_all, dTdr_all)
            a2 = jax.vmap(get_Thermodynamical_Forces_A2)(temperature, dTdr_all)
            a3 = get_Thermodynamical_Forces_A3(er_profile)
            density_phys = DENSITY_STATE_TO_PHYSICAL * density_local
            return -density_phys * (
                lij[:, 0, 0] * jax.lax.dynamic_index_in_dim(a1, radius_index, axis=1, keepdims=False)
                + lij[:, 0, 1] * jax.lax.dynamic_index_in_dim(a2, radius_index, axis=1, keepdims=False)
                + lij[:, 0, 2] * jax.lax.dynamic_index_in_dim(a3, radius_index, axis=0, keepdims=False)
            )

        return evaluator

    def evaluate_face_fluxes(self, state, face_state, **kwargs):
        face_response_mode = str(kwargs.get("face_response_mode", self.face_response_mode)).strip().lower()
        center_fluxes = kwargs.get("center_fluxes")
        if face_response_mode in {"interpolate_center_response", "interpolate_center_fluxes", "center_interpolation"}:
            if center_fluxes is None:
                center_fluxes = self(state)
            return {
                "Gamma": self._cell_centered_flux_to_faces_centered(center_fluxes["Gamma"]),
                "Q": self._cell_centered_flux_to_faces_centered(center_fluxes["Q"]),
                "Upar": self._cell_centered_flux_to_faces_centered(center_fluxes["Upar"]),
            }

        density = safe_density(state.density)
        face_density = safe_density(face_state.density)
        bc_density = kwargs.get("bc_density")
        bc_temperature = kwargs.get("bc_temperature")
        particle_face_closure_mode = str(kwargs.get("particle_face_closure_mode", "reconstructed")).strip().lower()
        if particle_face_closure_mode in {"ntss_like", "ntss", "half_point"}:
            dndr_faces = _ntss_like_face_gradient(
                density,
                self.geometry.r_grid_half,
                bc_model=bc_density,
            )
            dTdr_faces = _ntss_like_face_gradient(
                state.temperature,
                self.geometry.r_grid_half,
                bc_model=bc_temperature,
            )
        else:
            dndr_faces = _face_profile_gradient(
                density,
                self.geometry.r_grid_half,
                bc_model=bc_density,
            )
            dTdr_faces = _face_profile_gradient(
                state.temperature,
                self.geometry.r_grid_half,
                bc_model=bc_temperature,
            )
        lij_faces = self._lij_faces(face_state.Er, face_state.temperature, face_density)
        a1 = jax.vmap(
            lambda charge, density_a, temperature_a, dndr_a, dTdr_a: get_Thermodynamical_Forces_A1(
                charge, density_a, temperature_a, dndr_a, dTdr_a, face_state.Er
            ),
            in_axes=(0, 0, 0, 0, 0),
        )(self.species.charge, face_density, face_state.temperature, dndr_faces, dTdr_faces)
        a2 = jax.vmap(get_Thermodynamical_Forces_A2, in_axes=(0, 0))(face_state.temperature, dTdr_faces)
        a3 = get_Thermodynamical_Forces_A3(face_state.Er)
        density_phys = DENSITY_STATE_TO_PHYSICAL * face_density
        temperature_phys = TEMPERATURE_STATE_TO_PHYSICAL * face_state.temperature
        gamma = -density_phys * (
            lij_faces[:, :, 0, 0] * a1
            + lij_faces[:, :, 0, 1] * a2
            + lij_faces[:, :, 0, 2] * a3[None, :]
        )
        q = -temperature_phys * density_phys * (
            lij_faces[:, :, 1, 0] * a1
            + lij_faces[:, :, 1, 1] * a2
            + lij_faces[:, :, 1, 2] * a3[None, :]
        )
        upar = -density_phys * (
            lij_faces[:, :, 2, 0] * a1
            + lij_faces[:, :, 2, 1] * a2
            + lij_faces[:, :, 2, 2] * a3[None, :]
        )
        return {"Gamma": gamma, "Q": q, "Upar": upar}


def build_ntx_exact_lij_runtime_transport_model(
    species,
    energy_grid,
    geometry,
    *,
    vmec_file,
    boozer_file,
    ntx_exact_n_theta=25,
    ntx_exact_n_zeta=25,
    ntx_exact_n_xi=64,
    ntx_exact_surface_backend="auto",
    ntx_exact_face_response_mode="face_local_response",
    ntx_exact_radial_batch_size=None,
    ntx_exact_scan_batch_size=None,
    ntx_exact_use_remat=False,
    ntx_exact_lij_support=None,
    preload_support=False,
    collisionality_model="default",
    bc_density=None,
    bc_temperature=None,
    **kwargs,
):
    del kwargs
    model = NTXExactLijRuntimeTransportModel(
        species=species,
        energy_grid=energy_grid,
        geometry=geometry,
        vmec_file=str(vmec_file) if vmec_file is not None else None,
        boozer_file=str(boozer_file) if boozer_file is not None else None,
        n_theta=int(ntx_exact_n_theta),
        n_zeta=int(ntx_exact_n_zeta),
        n_xi=int(ntx_exact_n_xi),
        surface_backend=str(ntx_exact_surface_backend),
        face_response_mode=str(ntx_exact_face_response_mode),
        radial_batch_size=(
            None
            if ntx_exact_radial_batch_size in (None, "", 0, "0")
            else int(ntx_exact_radial_batch_size)
        ),
        scan_batch_size=(
            None
            if ntx_exact_scan_batch_size in (None, "", 0, "0")
            else int(ntx_exact_scan_batch_size)
        ),
        use_remat=bool(ntx_exact_use_remat),
        collisionality_model=str(collisionality_model),
        bc_density=bc_density,
        bc_temperature=bc_temperature,
        support=ntx_exact_lij_support,
    )
    if preload_support:
        return model.with_static_support()
    return model


def build_ntx_runtime_scan_transport_model(
    species,
    energy_grid,
    geometry,
    *,
    vmec_file,
    boozer_file,
    ntx_scan_rho,
    ntx_scan_nu_v,
    ntx_scan_er_tilde,
    ntx_scan_n_theta=25,
    ntx_scan_n_zeta=25,
    ntx_scan_n_xi=64,
    ntx_scan_surface_backend="auto",
    ntx_scan_source_name="ntx_scan_runtime",
    collisionality_model="default",
    bc_density=None,
    bc_temperature=None,
    ntx_scan_channels=None,
    preload_channels=False,
    prebuild_database=True,
    **kwargs,
):
    del kwargs
    model = NTXRuntimeScanTransportModel(
        species=species,
        energy_grid=energy_grid,
        geometry=geometry,
        vmec_file=str(vmec_file),
        boozer_file=str(boozer_file),
        rho_scan=ntx_scan_rho,
        nu_v_scan=ntx_scan_nu_v,
        er_tilde_scan=ntx_scan_er_tilde,
        n_theta=int(ntx_scan_n_theta),
        n_zeta=int(ntx_scan_n_zeta),
        n_xi=int(ntx_scan_n_xi),
        surface_backend=str(ntx_scan_surface_backend),
        source_name=str(ntx_scan_source_name),
        collisionality_model=str(collisionality_model),
        bc_density=bc_density,
        bc_temperature=bc_temperature,
        channels=ntx_scan_channels,
        database=None,
    )
    if preload_channels:
        model = model.with_static_channels()
    if not prebuild_database:
        return model
    return model.with_runtime_database()
    


# --- Torax-style, JAX-friendly ZeroTransportModel ---
@dataclasses.dataclass(frozen=True, eq=False)
class ZeroTransportModel(TransportFluxModelBase):
    shape: Any = None

    def __call__(self, state) -> dict:
        arr_shape = self.shape if self.shape is not None else state.density.shape
        gamma = jnp.zeros(arr_shape)
        q = jnp.zeros(arr_shape)
        upar = jnp.zeros(arr_shape)
        return {"Gamma": gamma, "Q": q, "Upar": upar}

    def build_local_particle_flux_evaluator(self, state):
        zeros = jnp.zeros(state.density.shape[0], dtype=state.density.dtype)

        def evaluator(radius_index, er_value):
            del radius_index, er_value
            return zeros

        return evaluator

    def evaluate_face_fluxes(self, state, face_state, **kwargs):
        del state, kwargs
        arr_shape = self.shape if self.shape is not None else face_state.density.shape
        gamma = jnp.zeros(arr_shape)
        q = jnp.zeros(arr_shape)
        upar = jnp.zeros(arr_shape)
        return {"Gamma": gamma, "Q": q, "Upar": upar}

    def build_lagged_response(self, state, **kwargs):
        del state, kwargs
        return None


def _normalize_flux_dataset(arr, n_species):
    out = jnp.asarray(arr, dtype=float)
    if out.ndim == 1:
        out = out[None, :]
    elif out.ndim != 2:
        raise ValueError(f"Flux dataset must be 1D or 2D, got shape {out.shape}.")

    if out.shape[0] == n_species:
        return out
    if out.shape[1] == n_species:
        return jnp.swapaxes(out, 0, 1)
    if out.shape[0] == 1:
        return jnp.repeat(out, n_species, axis=0)
    if out.shape[1] == 1:
        return jnp.repeat(jnp.swapaxes(out, 0, 1), n_species, axis=0)
    raise ValueError(f"Flux dataset shape {out.shape} is not compatible with n_species={n_species}.")


def read_flux_profile_file(path, n_species):
    with h5py.File(path, "r") as f:
        keys = set(f.keys())

        def _first(*names):
            for name in names:
                if name in keys:
                    return f[name][...]
            return None

        r = _first("r", "rho", "r_grid", "radius")
        if r is None:
            raise ValueError(f"Flux file '{path}' must contain one of datasets: r, rho, r_grid, radius.")
        gamma = _first("Gamma", "gamma")
        q = _first("Q", "q")
        upar = _first("Upar", "upar", "u_par")

    if gamma is None and q is None and upar is None:
        raise ValueError(f"Flux file '{path}' must contain at least one of Gamma, Q, or Upar.")

    r_arr = jnp.ravel(jnp.asarray(r, dtype=float))
    gamma_arr = None if gamma is None else _normalize_flux_dataset(gamma, n_species)
    q_arr = None if q is None else _normalize_flux_dataset(q, n_species)
    upar_arr = None if upar is None else _normalize_flux_dataset(upar, n_species)
    return r_arr, gamma_arr, q_arr, upar_arr


def _flux_profile_debug_summary(name, arr):
    if arr is None:
        return f"{name}=missing"

    arr_np = jnp.asarray(arr)
    pieces = [f"{name}.shape={tuple(arr_np.shape)}"]
    if arr_np.ndim == 1:
        finite = jnp.isfinite(arr_np)
        nfinite = int(jnp.sum(finite))
        if nfinite > 0:
            pieces.append(
                "finite={}/{} min={:.6e} max={:.6e}".format(
                    nfinite,
                    arr_np.shape[0],
                    float(jnp.min(arr_np[finite])),
                    float(jnp.max(arr_np[finite])),
                )
            )
        else:
            pieces.append(f"finite=0/{arr_np.shape[0]}")
        return " ".join(pieces)

    for idx in range(arr_np.shape[0]):
        prof = arr_np[idx]
        finite = jnp.isfinite(prof)
        nfinite = int(jnp.sum(finite))
        if nfinite > 0:
            pieces.append(
                "s{}:finite={}/{} min={:.6e} max={:.6e}".format(
                    idx,
                    nfinite,
                    prof.shape[0],
                    float(jnp.min(prof[finite])),
                    float(jnp.max(prof[finite])),
                )
            )
        else:
            pieces.append(f"s{idx}:finite=0/{prof.shape[0]}")
    return " ".join(pieces)


def build_fluxes_r_file_transport_model(
    species,
    geometry,
    *,
    fluxes_file=None,
    file=None,
    flux_file=None,
    neoclassical_file=None,
    turbulence_file=None,
    classical_file=None,
    grid_location="cell_centered",
    profile_location=None,
    **kwargs,
):
    q_scale = float(
        kwargs.pop(
            "debug_heat_flux_scale",
            kwargs.pop(
                "heat_flux_scale",
                kwargs.pop("q_scale", 1.0),
            ),
        )
    )
    path = (
        fluxes_file
        or file
        or flux_file
        or neoclassical_file
        or turbulence_file
        or classical_file
    )
    if path is None:
        raise ValueError(
            "fluxes_r_file requires a flux file. "
            "Provide one of: fluxes_file, file, flux_file, neoclassical_file, turbulence_file, or classical_file."
    )
    location = profile_location if profile_location is not None else grid_location
    r_data, gamma_data, q_data, upar_data = read_flux_profile_file(path, species.number_species)
    r_finite = jnp.isfinite(r_data)
    r_nfinite = int(jnp.sum(r_finite))
    if r_nfinite > 0:
        r_summary = "finite={}/{} min={:.6e} max={:.6e}".format(
            r_nfinite,
            r_data.shape[0],
            float(jnp.min(r_data[r_finite])),
            float(jnp.max(r_data[r_finite])),
        )
    else:
        r_summary = f"finite=0/{r_data.shape[0]}"
    print(
        "[NEOPAX] fluxes_r_file loaded: "
        f"path={path} profile_location={str(location).strip().lower()} "
        f"r.shape={tuple(r_data.shape)} q_scale={q_scale:.6e} {r_summary}"
    )
    print(f"[NEOPAX] fluxes_r_file dataset: {_flux_profile_debug_summary('Gamma', gamma_data)}")
    print(f"[NEOPAX] fluxes_r_file dataset: {_flux_profile_debug_summary('Q', q_data)}")
    print(f"[NEOPAX] fluxes_r_file dataset: {_flux_profile_debug_summary('Upar', upar_data)}")
    return FluxesRFileTransportModel(
        species=species,
        geometry=geometry,
        r_data=r_data,
        gamma_data=gamma_data,
        q_data=q_data,
        upar_data=upar_data,
        profile_location=str(location).strip().lower(),
        q_scale=q_scale,
    )


@dataclasses.dataclass(frozen=True, eq=False)
class FluxesRFileTransportModel(TransportFluxModelBase):
    species: Any
    geometry: Any
    r_data: Any
    gamma_data: Any = None
    q_data: Any = None
    upar_data: Any = None
    profile_location: str = "cell_centered"
    q_scale: float = 1.0

    def with_q_scale(self, q_scale: float) -> "FluxesRFileTransportModel":
        return dataclasses.replace(self, q_scale=float(q_scale))

    def _interp_species_profile(self, data, target_r):
        if data is None:
            return jnp.zeros((self.species.number_species, target_r.shape[0]), dtype=target_r.dtype)
        return jax.vmap(lambda prof: interpax.interp1d(target_r, self.r_data, prof))(data)

    def _normalize_profile_location(self):
        location = str(self.profile_location).strip().lower()
        aliases = {
            "cell": "cell_centered",
            "cells": "cell_centered",
            "center": "cell_centered",
            "centers": "cell_centered",
            "cell_centered": "cell_centered",
            "cell-centred": "cell_centered",
            "cell_centred": "cell_centered",
            "face": "face_centered",
            "faces": "face_centered",
            "face_centered": "face_centered",
            "face-centred": "face_centered",
            "face_centred": "face_centered",
        }
        if location not in aliases:
            raise ValueError(
                f"Unsupported fluxes_r_file profile_location '{self.profile_location}'. "
                "Expected one of: cell_centered, face_centered."
            )
        return aliases[location]

    def _data_on_cell_grid(self, data):
        location = self._normalize_profile_location()
        if location == "cell_centered":
            return self._interp_species_profile(data, self.geometry.r_grid)
        face_values = self._interp_species_profile(data, self.geometry.r_grid_half)
        return jax.vmap(cell_centered_from_faces)(face_values)

    def _data_on_face_grid(self, data):
        location = self._normalize_profile_location()
        if location == "face_centered":
            return self._interp_species_profile(data, self.geometry.r_grid_half)
        cell_values = self._interp_species_profile(data, self.geometry.r_grid)
        return jax.vmap(faces_from_cell_centered)(cell_values)

    def __call__(self, state) -> dict:
        del state
        gamma = self._data_on_cell_grid(self.gamma_data)
        q = self.q_scale * self._data_on_cell_grid(self.q_data)
        upar = self._data_on_cell_grid(self.upar_data)
        return {"Gamma": gamma, "Q": q, "Upar": upar}

    def build_local_particle_flux_evaluator(self, state):
        del state
        gamma = self._data_on_cell_grid(self.gamma_data)

        def evaluator(radius_index, er_value):
            del er_value
            return gamma[:, radius_index]

        return evaluator

    def evaluate_face_fluxes(self, state, face_state, **kwargs):
        del state, face_state, kwargs
        gamma = self._data_on_face_grid(self.gamma_data)
        q = self.q_scale * self._data_on_face_grid(self.q_data)
        upar = self._data_on_face_grid(self.upar_data)
        return {"Gamma": gamma, "Q": q, "Upar": upar}






# --- Torax-style, JAX-friendly AnalyticalTurbulentTransportModel ---
@dataclasses.dataclass(frozen=True, eq=False)
class AnalyticalTurbulentTransportModel(TransportFluxModelBase):
    species: Any
    grid: Any
    chi_t: Any
    chi_n: Any
    field: Any

    def with_transport_coeffs(self, *, chi_t=None, chi_n=None) -> "AnalyticalTurbulentTransportModel":
        return dataclasses.replace(
            self,
            chi_t=self.chi_t if chi_t is None else chi_t,
            chi_n=self.chi_n if chi_n is None else chi_n,
        )

    def __call__(self, state) -> dict:
        gamma_turb, q_turb = get_Turbulent_Fluxes_Analytical(
            self.species,
            self.grid,
            self.chi_t,
            self.chi_n,
            state.temperature,
            state.density,
            self.field,
        )
        upar = jnp.zeros_like(state.density)
        return {"Gamma": gamma_turb, "Q": q_turb, "Upar": upar}

    def build_local_particle_flux_evaluator(self, state):
        gamma_turb, _ = get_Turbulent_Fluxes_Analytical(
            self.species,
            self.grid,
            self.chi_t,
            self.chi_n,
            state.temperature,
            state.density,
            self.field,
        )

        def evaluator(radius_index, er_value):
            del er_value
            return gamma_turb[:, radius_index]

        return evaluator

    def evaluate_face_fluxes(self, state, face_state, **kwargs):
        bc_density = kwargs.get("bc_density")
        bc_temperature = kwargs.get("bc_temperature")
        dndr_faces = _face_profile_gradient(
            DENSITY_STATE_TO_PHYSICAL * state.density,
            self.field.r_grid_half,
            bc_model=bc_density,
        )
        dTdr_faces = _face_profile_gradient(
            TEMPERATURE_STATE_TO_PHYSICAL * state.temperature,
            self.field.r_grid_half,
            bc_model=bc_temperature,
        )
        gamma = -self.chi_n[:, None] * dndr_faces
        q = -(DENSITY_STATE_TO_PHYSICAL * face_state.density) * self.chi_t[:, None] * dTdr_faces
        upar = jnp.zeros_like(gamma)
        return {"Gamma": gamma, "Q": q, "Upar": upar}

    def build_lagged_response(self, state, **kwargs):
        del kwargs
        return JVPTransportFluxResponse(
            reference_state=state,
            reference_flux=self(state),
        )

    def evaluate_with_lagged_response(self, state, lagged_response, **kwargs):
        del kwargs
        delta_state = jax.tree_util.tree_map(
            lambda current, reference: current - reference,
            state,
            lagged_response.reference_state,
        )
        tangent_flux = jax.jvp(
            self.__call__,
            (lagged_response.reference_state,),
            (delta_state,),
        )[1]
        return jax.tree_util.tree_map(
            lambda reference, tangent: reference + tangent,
            lagged_response.reference_flux,
            tangent_flux,
        )


@dataclasses.dataclass(frozen=True, eq=False)
class PowerAnalyticalTurbulentTransportModel(TransportFluxModelBase):
    species: Any
    field: Any
    chi_t: Any
    chi_n: Any
    pressure_source_model: Any = None
    total_power_mw: Any = None

    def with_transport_coeffs(
        self,
        *,
        chi_t=None,
        chi_n=None,
        pressure_source_model=None,
        total_power_mw=None,
    ) -> "PowerAnalyticalTurbulentTransportModel":
        return dataclasses.replace(
            self,
            chi_t=self.chi_t if chi_t is None else chi_t,
            chi_n=self.chi_n if chi_n is None else chi_n,
            pressure_source_model=self.pressure_source_model if pressure_source_model is None else pressure_source_model,
            total_power_mw=self.total_power_mw if total_power_mw is None else total_power_mw,
        )

    def _effective_total_power_mw(self, state):
        if self.total_power_mw is not None:
            return jnp.asarray(self.total_power_mw, dtype=state.density.dtype)
        return compute_total_power_mw(
            state,
            self.species,
            self.pressure_source_model,
            self.field,
        )

    def __call__(self, state) -> dict:
        total_power_mw = self._effective_total_power_mw(state)
        gamma_turb, q_turb = get_Turbulent_Fluxes_PowerOverN(
            self.species,
            self.chi_t,
            self.chi_n,
            total_power_mw,
            state.temperature,
            state.density,
            self.field,
        )
        upar = jnp.zeros_like(state.density)
        return {"Gamma": gamma_turb, "Q": q_turb, "Upar": upar}

    def build_local_particle_flux_evaluator(self, state):
        total_power_mw = self._effective_total_power_mw(state)
        gamma_turb, _ = get_Turbulent_Fluxes_PowerOverN(
            self.species,
            self.chi_t,
            self.chi_n,
            total_power_mw,
            state.temperature,
            state.density,
            self.field,
        )

        def evaluator(radius_index, er_value):
            del er_value
            return gamma_turb[:, radius_index]

        return evaluator

    def evaluate_face_fluxes(self, state, face_state, **kwargs):
        bc_density = kwargs.get("bc_density")
        bc_temperature = kwargs.get("bc_temperature")
        total_power_mw = self._effective_total_power_mw(state)
        dndr_faces = _face_profile_gradient(
            DENSITY_STATE_TO_PHYSICAL * state.density,
            self.field.r_grid_half,
            bc_model=bc_density,
        )
        dTdr_faces = _face_profile_gradient(
            TEMPERATURE_STATE_TO_PHYSICAL * state.temperature,
            self.field.r_grid_half,
            bc_model=bc_temperature,
        )
        electron_idx = int(self.species.species_idx["e"])
        ne_face = jnp.maximum(jnp.asarray(face_state.density[electron_idx], dtype=state.density.dtype), 1.0e-12)
        p075 = jnp.where(total_power_mw < 0.0, jnp.asarray(3.0, dtype=state.density.dtype), jnp.power(total_power_mw, 0.75))
        density_coeff = jnp.asarray(self.chi_n, dtype=state.density.dtype)[:, None] * p075 / ne_face[None, :]
        heat_coeff = jnp.asarray(self.chi_t, dtype=state.density.dtype)[:, None] * p075 / ne_face[None, :]
        gamma = -density_coeff * dndr_faces
        q = -(DENSITY_STATE_TO_PHYSICAL * face_state.density) * heat_coeff * dTdr_faces
        upar = jnp.zeros_like(gamma)
        return {"Gamma": gamma, "Q": q, "Upar": upar}

    def build_lagged_response(self, state, **kwargs):
        del kwargs
        return JVPTransportFluxResponse(
            reference_state=state,
            reference_flux=self(state),
        )

    def evaluate_with_lagged_response(self, state, lagged_response, **kwargs):
        del kwargs
        delta_state = jax.tree_util.tree_map(
            lambda current, reference: current - reference,
            state,
            lagged_response.reference_state,
        )
        tangent_flux = jax.jvp(
            self.__call__,
            (lagged_response.reference_state,),
            (delta_state,),
        )[1]
        return jax.tree_util.tree_map(
            lambda reference, tangent: reference + tangent,
            lagged_response.reference_flux,
            tangent_flux,
        )


# --- PATCH: Accept [neoclassical]/flux_model and [turbulence]/model as defaults ---

# --- Refactored: Only the orchestrator builds models; this function is now a pure factory ---
def build_transport_flux_model(neo_model: TransportFluxModelBase,
                              turb_model: TransportFluxModelBase,
                              classical_model: TransportFluxModelBase = None,
                              *,
                              include_turbulent_particle_flux: bool = True) -> CombinedTransportFluxModel:
    """
    Build the composed transport model from explicit model instances.
    All models must be constructed up front by the orchestrator.
    """
    if classical_model is None:
        classical_model = ZeroTransportModel()
    return CombinedTransportFluxModel(
        neo_model,
        turb_model,
        classical_model,
        include_turbulent_particle_flux=bool(include_turbulent_particle_flux),
    )

register_transport_flux_model(
    "ntx_database",
    lambda species, energy_grid, geometry, database, collisionality_model="default", bc_density=None, bc_temperature=None: NTXDatabaseTransportModel(
        species=species,
        energy_grid=energy_grid,
        geometry=geometry,
        database=database,
        collisionality_model=collisionality_model,
        bc_density=bc_density,
        bc_temperature=bc_temperature,
    ),
)

register_transport_flux_model(
    "ntx_scan_runtime",
    lambda species, energy_grid, geometry, database=None, **kwargs: build_ntx_runtime_scan_transport_model(
        species,
        energy_grid,
        geometry,
        **kwargs,
    ),
)

register_transport_flux_model(
    "ntx_exact_lij_runtime",
    lambda species, energy_grid, geometry, database=None, **kwargs: build_ntx_exact_lij_runtime_transport_model(
        species,
        energy_grid,
        geometry,
        **kwargs,
    ),
)

register_transport_flux_model(
    "ntx_database_with_momentum",
    lambda species, energy_grid, geometry, database,
           density_right_constraint=None, density_right_grad_constraint=None,
           temperature_right_constraint=None, temperature_right_grad_constraint=None: NTXDatabaseTransportModel(
        species=species,
        energy_grid=energy_grid,
        geometry=geometry,
        database=database,
        bc_density=None,
        bc_temperature=None,
    ),
)

register_transport_flux_model(
    "turbulent_analytical",
    lambda species, grid, chi_t, chi_n, field: AnalyticalTurbulentTransportModel(
        species=species,
        grid=grid,
        chi_t=chi_t,
        chi_n=chi_n,
        field=field,
    ),
)

register_transport_flux_model(
    "turbulent_power_analytical",
    lambda species, grid, field, chi_t, chi_n, pressure_source_model=None, total_power_mw=None: PowerAnalyticalTurbulentTransportModel(
        species=species,
        field=field,
        chi_t=chi_t,
        chi_n=chi_n,
        pressure_source_model=pressure_source_model,
        total_power_mw=total_power_mw,
    ),
)

register_transport_flux_model(
    "ntss_power_over_n",
    lambda species, grid, field, chi_t, chi_n, pressure_source_model=None, total_power_mw=None: PowerAnalyticalTurbulentTransportModel(
        species=species,
        field=field,
        chi_t=chi_t,
        chi_n=chi_n,
        pressure_source_model=pressure_source_model,
        total_power_mw=total_power_mw,
    ),
)

register_transport_flux_model(
    "fluxes_r_file",
    lambda species, energy_grid, geometry, database, **kwargs: build_fluxes_r_file_transport_model(
        species=species,
        geometry=geometry,
        **kwargs,
    ),
)

register_transport_flux_model(
    "none",
    lambda *args, **kwargs: ZeroTransportModel(),
)
