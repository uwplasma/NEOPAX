from __future__ import annotations


import os
import functools
from typing import Any, Callable
import abc
import dataclasses
import h5py
import jax
import jax.numpy as jnp
import interpax
import numpy as np
import sys
import types
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
    _broadcast_species_floor,
    apply_transport_density_floor,
    apply_transport_temperature_floor,
    get_v_thermal,
    safe_density,
    safe_temperature,
)
from ._database import D11_POSITIVE_FLOOR
from ._source_models import assemble_pressure_source_components, sum_source_components
from ._model_api import (
    ModelCapabilities,
    ModelValidationContext,
    transport_model as transport_model_decorator,
    validate_transport_flux_builder,
)
from ._transport_debug import lagged_timing_enabled, lagged_timing_start, lagged_timing_end
from ._constants import elementary_charge
from ._spectrax_quasilinear_runtime import (
    SpectraXQuasilinearRuntimeDiagnostics,
    evaluate_spectrax_quasilinear_proxy,
)

DENSITY_STATE_TO_PHYSICAL = 1.0e20
TEMPERATURE_STATE_TO_PHYSICAL = 1.0e3
PRESSURE_SOURCE_STATE_TO_MW_M3 = 1.0 / 62.422
_INTERPOLATED_RESPONSE_FIELD_NAMES = (
    "reference_log_nu_star",
    "reference_transport_moments",
    "dtransport_moments_d_er",
    "dtransport_moments_d_log_nu_star",
)
from ._turbulence import get_Turbulent_Fluxes_Analytical, get_Turbulent_Fluxes_PowerOverN


def _ntx_local_pullback_finite_debug_enabled() -> bool:
    raw = str(os.environ.get("NEOPAX_TRANSPORT_NTX_LOCAL_PULLBACK_FINITE_DEBUG", "")).strip().lower()
    return raw not in {"", "0", "false", "no", "off"}


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

        def pullback_build_lagged_response(self, state, lagged_response_bar, **kwargs):
                _, pullback = jax.vjp(
                        lambda state_value: self.build_lagged_response(state_value, **kwargs),
                        state,
                )
                (state_bar,) = pullback(lagged_response_bar)
                return state_bar


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
class NTXInterpolatedMomentResponse:
    reference_er: jax.Array
    reference_log_nu_star: jax.Array
    reference_transport_moments: jax.Array
    dtransport_moments_d_er: jax.Array
    dtransport_moments_d_log_nu_star: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class _NTXInterpolatedMomentResponseFieldBars:
    reference_log_nu_star: jax.Array
    reference_transport_moments: jax.Array
    dtransport_moments_d_er: jax.Array
    dtransport_moments_d_log_nu_star: jax.Array


def _interpolated_response_field_bar_tuple(
    field_bars: _NTXInterpolatedMomentResponseFieldBars | tuple[jax.Array, jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    if isinstance(field_bars, _NTXInterpolatedMomentResponseFieldBars):
        return dataclasses.astuple(field_bars)
    return tuple(field_bars)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class NTXExactLijLaggedResponse:
        center_response: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class SpectraXTurbulenceFDLaggedResponse:
    reference_state: Any
    reference_flux: dict
    reference_basis: Any = None
    perturb_kind_codes: Any = None
    perturb_species_indices: Any = None
    perturb_delta: Any = None
    perturb_present: Any = None
    gamma_perturb: Any = None
    q_perturb: Any = None


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

    def pullback_build_lagged_response(self, state, lagged_response_bar, **kwargs):
        def _submodel_pullback(model, response_bar):
            if response_bar is None:
                return None
            pullback_fn = getattr(model, "pullback_build_lagged_response", None)
            if callable(pullback_fn):
                return pullback_fn(state, response_bar, **kwargs)
            _, pb = jax.vjp(
                lambda state_value: model.build_lagged_response(state_value, **kwargs),
                state,
            )
            (state_bar,) = pb(response_bar)
            return state_bar

        neo_bar = _submodel_pullback(
            self.neoclassical_model,
            lagged_response_bar.neoclassical_response,
        )
        turb_bar = _submodel_pullback(
            self.turbulent_model,
            lagged_response_bar.turbulent_response,
        )
        classical_bar = _submodel_pullback(
            self.classical_model,
            lagged_response_bar.classical_response,
        )

        def _zero_like_state():
            return jax.tree_util.tree_map(jnp.zeros_like, state)

        state_bar = _zero_like_state()
        for part in (neo_bar, turb_bar, classical_bar):
            if part is None:
                continue
            state_bar = jax.tree_util.tree_map(lambda a, b: a + b, state_bar, part)
        return state_bar

    def pullback_build_lagged_response_support_payload(self, state, lagged_response_bar, support, **kwargs):
        pullback_fn = getattr(
            self.neoclassical_model,
            "pullback_build_lagged_response_support_payload",
            None,
        )
        if callable(pullback_fn):
            return _sanitize_float_delta_bar_tree(
                support,
                pullback_fn(
                    state,
                    lagged_response_bar.neoclassical_response,
                    support,
                    **kwargs,
                ),
            )
        support_delta0 = _float_delta_tree_like(support)
        _, support_delta_pullback = jax.vjp(
            lambda support_delta: self.neoclassical_model.with_support_payload(
                _add_float_delta_tree(support, support_delta)
            ).build_lagged_response(state, **kwargs),
            support_delta0,
        )
        (support_bar,) = support_delta_pullback(lagged_response_bar.neoclassical_response)
        return _sanitize_float_delta_bar_tree(support, support_bar)

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

    def pullback_evaluate_with_lagged_response(self, state, lagged_response, flux_bar, **kwargs):
        def _submodel_pullback(model, subresponse, subflux_bar):
            if subresponse is None:
                return None
            pullback_fn = getattr(model, "pullback_evaluate_with_lagged_response", None)
            if callable(pullback_fn):
                return pullback_fn(state, subresponse, subflux_bar, **kwargs)
            _, pb = jax.vjp(
                lambda response_value: model.evaluate_with_lagged_response(
                    state,
                    response_value,
                    **kwargs,
                ),
                subresponse,
            )
            (subresponse_bar,) = pb(subflux_bar)
            return subresponse_bar

        gamma_total_bar = flux_bar.get("Gamma", 0)
        q_total_bar = flux_bar.get("Q", 0)
        upar_total_bar = flux_bar.get("Upar", 0)

        zero_gamma = self._zero_like_flux(flux_bar.get("Gamma", None), 0)
        neo_flux_bar = {
            "Gamma": gamma_total_bar + flux_bar.get("Gamma_neo", zero_gamma),
            "Q": q_total_bar + flux_bar.get("Q_neo", 0),
            "Upar": upar_total_bar + flux_bar.get("Upar_neo", 0),
        }
        turb_flux_bar = {
            "Gamma": (
                gamma_total_bar + flux_bar.get("Gamma_turb", zero_gamma)
                if self.include_turbulent_particle_flux
                else flux_bar.get("Gamma_turb", zero_gamma)
            ),
            "Q": q_total_bar + flux_bar.get("Q_turb", 0),
            "Upar": upar_total_bar + flux_bar.get("Upar_turb", 0),
        }
        classical_flux_bar = {
            "Gamma": gamma_total_bar + flux_bar.get("Gamma_classical", zero_gamma),
            "Q": q_total_bar + flux_bar.get("Q_classical", 0),
            "Upar": upar_total_bar + flux_bar.get("Upar_classical", 0),
        }

        return CombinedTransportLaggedResponse(
            neoclassical_response=_submodel_pullback(
                self.neoclassical_model,
                lagged_response.neoclassical_response,
                neo_flux_bar,
            ),
            turbulent_response=_submodel_pullback(
                self.turbulent_model,
                lagged_response.turbulent_response,
                turb_flux_bar,
            ),
            classical_response=_submodel_pullback(
                self.classical_model,
                lagged_response.classical_response,
                classical_flux_bar,
            ),
        )

    def pullback_evaluate_with_lagged_response_support_payload(
        self,
        state,
        lagged_response,
        flux_bar,
        support,
        **kwargs,
    ):
        gamma_total_bar = flux_bar.get("Gamma", 0)
        q_total_bar = flux_bar.get("Q", 0)
        upar_total_bar = flux_bar.get("Upar", 0)

        zero_gamma = self._zero_like_flux(flux_bar.get("Gamma", None), 0)
        neo_flux_bar = {
            "Gamma": gamma_total_bar + flux_bar.get("Gamma_neo", zero_gamma),
            "Q": q_total_bar + flux_bar.get("Q_neo", 0),
            "Upar": upar_total_bar + flux_bar.get("Upar_neo", 0),
        }
        pullback_fn = getattr(
            self.neoclassical_model,
            "pullback_evaluate_with_lagged_response_support_payload",
            None,
        )
        if callable(pullback_fn):
            return _sanitize_float_delta_bar_tree(
                support,
                pullback_fn(
                    state,
                    lagged_response.neoclassical_response,
                    neo_flux_bar,
                    support,
                    **kwargs,
                ),
            )
        support_delta0 = _float_delta_tree_like(support)
        _, support_delta_pullback = jax.vjp(
            lambda support_delta: self.neoclassical_model.with_support_payload(
                _add_float_delta_tree(support, support_delta)
            ).evaluate_with_lagged_response(
                state,
                lagged_response.neoclassical_response,
                **kwargs,
            ),
            support_delta0,
        )
        (support_bar,) = support_delta_pullback(neo_flux_bar)
        return _sanitize_float_delta_bar_tree(support, support_bar)

    def pullback_evaluate_with_lagged_response_state(self, state, lagged_response, flux_bar, **kwargs):
        def _zero_state_bar():
            return jax.tree_util.tree_map(jnp.zeros_like, state)

        def _submodel_state_pullback(model, subresponse, subflux_bar):
            if subresponse is None:
                return _zero_state_bar()
            pullback_fn = getattr(model, "pullback_evaluate_with_lagged_response_state", None)
            if callable(pullback_fn):
                return pullback_fn(state, subresponse, subflux_bar, **kwargs)
            _, pb = jax.vjp(
                lambda state_value: model.evaluate_with_lagged_response(
                    state_value,
                    subresponse,
                    **kwargs,
                ),
                state,
            )
            (state_bar,) = pb(subflux_bar)
            return state_bar

        gamma_total_bar = flux_bar.get("Gamma", 0)
        q_total_bar = flux_bar.get("Q", 0)
        upar_total_bar = flux_bar.get("Upar", 0)

        zero_gamma = self._zero_like_flux(flux_bar.get("Gamma", None), 0)
        neo_flux_bar = {
            "Gamma": gamma_total_bar + flux_bar.get("Gamma_neo", zero_gamma),
            "Q": q_total_bar + flux_bar.get("Q_neo", 0),
            "Upar": upar_total_bar + flux_bar.get("Upar_neo", 0),
        }
        turb_flux_bar = {
            "Gamma": (
                gamma_total_bar + flux_bar.get("Gamma_turb", zero_gamma)
                if self.include_turbulent_particle_flux
                else flux_bar.get("Gamma_turb", zero_gamma)
            ),
            "Q": q_total_bar + flux_bar.get("Q_turb", 0),
            "Upar": upar_total_bar + flux_bar.get("Upar_turb", 0),
        }
        classical_flux_bar = {
            "Gamma": gamma_total_bar + flux_bar.get("Gamma_classical", zero_gamma),
            "Q": q_total_bar + flux_bar.get("Q_classical", 0),
            "Upar": upar_total_bar + flux_bar.get("Upar_classical", 0),
        }

        neo_state_bar = _submodel_state_pullback(
            self.neoclassical_model,
            lagged_response.neoclassical_response,
            neo_flux_bar,
        )
        turb_state_bar = _submodel_state_pullback(
            self.turbulent_model,
            lagged_response.turbulent_response,
            turb_flux_bar,
        )
        classical_state_bar = _submodel_state_pullback(
            self.classical_model,
            lagged_response.classical_response,
            classical_flux_bar,
        )
        return jax.tree_util.tree_map(
            lambda a, b, c: a + b + c,
            neo_state_bar,
            turb_state_bar,
            classical_state_bar,
        )




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
    _install_vmec_jax_api_compat_for_ntx()
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


def _install_vmec_jax_api_compat_for_ntx() -> None:
    """Provide the legacy `vmec_jax.api.read_wout` import expected by NTX.

    Newer vmec_jax versions export `read_wout` at the package top level instead
    of through `vmec_jax.api`.  Keep the compatibility local to the NEOPAX
    process so the NTX checkout itself remains untouched.
    """

    if "vmec_jax.api" in sys.modules:
        return
    try:
        import vmec_jax
    except ImportError:
        return

    read_wout = getattr(vmec_jax, "read_wout", None)
    if read_wout is None:
        try:
            from vmec_jax.core.wout import read_wout as read_wout
        except ImportError:
            return

    api_module = types.ModuleType("vmec_jax.api")
    api_module.read_wout = read_wout
    sys.modules["vmec_jax.api"] = api_module
    try:
        setattr(vmec_jax, "api", api_module)
    except Exception:
        pass


def _ntx_prepared_coefficient_vector_solver(ntx, derivative_mode: str):
    if derivative_mode == "custom_jvp":
        solver = getattr(ntx, "solve_prepared_coefficient_vector_jvp", None)
        if solver is not None:
            return solver
        try:
            from ntx._solver_prepared import solve_prepared_coefficient_vector_jvp

            return solve_prepared_coefficient_vector_jvp
        except ImportError as exc:
            raise AttributeError(
                "ntx_exact_derivative_mode='custom_jvp' requires NTX to provide "
                "solve_prepared_coefficient_vector_jvp."
            ) from exc
    if derivative_mode == "custom_vjp":
        solver = getattr(ntx, "solve_prepared_coefficient_vector_vjp", None)
        if solver is not None:
            return solver
        try:
            from ntx._solver_prepared import solve_prepared_coefficient_vector_vjp

            return solve_prepared_coefficient_vector_vjp
        except ImportError as exc:
            raise AttributeError(
                "ntx_exact_derivative_mode='custom_vjp' requires NTX to provide "
                "solve_prepared_coefficient_vector_vjp."
            ) from exc
    if derivative_mode == "recompute_vjp":
        solver = getattr(ntx, "solve_prepared_coefficient_vector_recompute_vjp", None)
        if solver is not None:
            return solver
        try:
            from ntx._solver_prepared import solve_prepared_coefficient_vector_recompute_vjp

            return solve_prepared_coefficient_vector_recompute_vjp
        except ImportError as exc:
            raise AttributeError(
                "ntx_exact_derivative_mode='recompute_vjp' requires NTX to provide "
                "solve_prepared_coefficient_vector_recompute_vjp."
            ) from exc
    if derivative_mode == "iterative_vjp":
        solver = getattr(ntx, "solve_prepared_coefficient_vector_iterative_vjp", None)
        if solver is not None:
            return solver
        try:
            from ntx._solver_prepared import solve_prepared_coefficient_vector_iterative_vjp

            return solve_prepared_coefficient_vector_iterative_vjp
        except ImportError as exc:
            raise AttributeError(
                "ntx_exact_derivative_mode='iterative_vjp' requires NTX to provide "
                "solve_prepared_coefficient_vector_iterative_vjp."
            ) from exc
    if derivative_mode == "iterative_jvp":
        solver = getattr(ntx, "solve_prepared_coefficient_vector_iterative_jvp", None)
        if solver is not None:
            return solver
        try:
            from ntx._solver_prepared import solve_prepared_coefficient_vector_iterative_jvp

            return solve_prepared_coefficient_vector_iterative_jvp
        except ImportError as exc:
            raise AttributeError(
                "ntx_exact_derivative_mode='iterative_jvp' requires NTX to provide "
                "solve_prepared_coefficient_vector_iterative_jvp."
            ) from exc
    return ntx.solve_prepared_coefficient_vector


def _ntx_prepared_coefficient_vector_derivative_pullback(ntx):
    solver = getattr(ntx, "solve_prepared_coefficient_vector_derivative_vjp", None)
    if solver is not None:
        return solver
    try:
        from ntx._solver_prepared import solve_prepared_coefficient_vector_derivative_vjp

        return solve_prepared_coefficient_vector_derivative_vjp
    except ImportError as exc:
        raise AttributeError(
            "ntx_exact_derivative_field_pullback_mode='compact_vjp' requires NTX "
            "to provide solve_prepared_coefficient_vector_derivative_vjp."
        ) from exc


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


def _float_delta_tree_like(tree):
    def _zero_leaf(leaf):
        arr = jnp.asarray(leaf)
        if jnp.issubdtype(arr.dtype, jnp.inexact):
            return jnp.zeros_like(arr)
        return jnp.zeros(arr.shape, dtype=jnp.float64)

    return jax.tree_util.tree_map(_zero_leaf, tree)


def _add_float_delta_tree(primal_tree, delta_tree):
    def _add_leaf(primal_leaf, delta_leaf):
        arr = jnp.asarray(primal_leaf)
        if jnp.issubdtype(arr.dtype, jnp.inexact):
            return arr + jnp.asarray(delta_leaf, dtype=arr.dtype)
        return primal_leaf

    return jax.tree_util.tree_map(_add_leaf, primal_tree, delta_tree)


def _sanitize_float_delta_bar_tree(primal_tree, bar_tree):
    def _sanitize_leaf(primal_leaf, bar_leaf):
        arr = jnp.asarray(primal_leaf)
        if jnp.issubdtype(arr.dtype, jnp.inexact):
            return jnp.asarray(bar_leaf, dtype=arr.dtype)
        return jnp.zeros(arr.shape, dtype=jnp.float64)

    return jax.tree_util.tree_map(_sanitize_leaf, primal_tree, bar_tree)


def _support_with_channel_delta(support, center_delta, face_delta):
    return dataclasses.replace(
        support,
        center_channels=_add_float_delta_tree(support.center_channels, center_delta),
        face_channels=_add_float_delta_tree(support.face_channels, face_delta),
    )


def _support_with_center_delta(support, center_channels_delta, center_prepared_delta):
    return dataclasses.replace(
        support,
        center_channels=_add_float_delta_tree(support.center_channels, center_channels_delta),
        center_prepared=_add_float_delta_tree(support.center_prepared, center_prepared_delta),
    )


def _support_bar_from_channel_bars(support, center_bar, face_bar):
    return dataclasses.replace(
        _float_delta_tree_like(support),
        center_channels=_sanitize_float_delta_bar_tree(support.center_channels, center_bar),
        face_channels=_sanitize_float_delta_bar_tree(support.face_channels, face_bar),
    )


def _support_bar_from_center_bars(support, center_channels_bar, center_prepared_bar):
    return dataclasses.replace(
        _float_delta_tree_like(support),
        center_channels=_sanitize_float_delta_bar_tree(support.center_channels, center_channels_bar),
        center_prepared=_sanitize_float_delta_bar_tree(support.center_prepared, center_prepared_bar),
    )


def build_ntx_runtime_scan_channels(vmec_file, boozer_file, rho_scan) -> NTXRuntimeScanChannels:
    rho = _as_float_array(rho_scan, name="rho_scan")
    channels = _load_ntx_vmec_boozer_channels(Path(vmec_file), Path(boozer_file), rho)
    return NTXRuntimeScanChannels.from_mapping(rho, channels)


def _build_ntx_surface_loader(vmec_file, boozer_file, surface_backend="auto"):
    ntx = _import_ntx()
    backend = str(surface_backend).strip().lower()
    vmec_file = str(vmec_file)
    boozer_file = str(boozer_file)
    vmec_file_loaders = []
    for loader_name in ("surface_from_vmex_vmec_wout_file", "surface_from_vmec_jax_vmec_wout_file"):
        loader = getattr(ntx, loader_name, None)
        if loader is not None:
            vmec_file_loaders.append(loader)
    if not vmec_file_loaders:
        raise AttributeError(
            "NTX does not expose a VMEC surface loader. Expected either "
            "'surface_from_vmex_vmec_wout_file' or "
            "'surface_from_vmec_jax_vmec_wout_file'."
        )

    def load_vmec(rho_value: float):
        s_value = float(rho_value**2)
        errors = []
        for loader in vmec_file_loaders:
            try:
                return loader(vmec_file, s=s_value)
            except Exception as exc:
                errors.append(exc)
        raise errors[-1]

    def load(rho_value: float):
        if backend == "vmec":
            return load_vmec(rho_value)
        if backend == "boozmn":
            return ntx.load_boozmn_surface(boozer_file, rho=float(rho_value)).surface
        if backend == "auto":
            try:
                return ntx.load_boozmn_surface(boozer_file, rho=float(rho_value)).surface
            except Exception:
                return load_vmec(rho_value)
        raise ValueError("surface_backend must be one of: auto, boozmn, vmec")

    return ntx, load


def build_ntx_runtime_surfaces(vmec_file, boozer_file, rho_values, *, surface_backend="auto") -> tuple[Any, ...]:
    rho_arr = _as_float_array(rho_values, name="rho_values")
    _, loader = _build_ntx_surface_loader(vmec_file, boozer_file, surface_backend=surface_backend)
    return tuple(loader(float(rho_value)) for rho_value in rho_arr)


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=("center_channels", "face_channels", "center_prepared", "face_prepared"),
    meta_fields=("grid",),
)
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
    surface_backend: str = "vmec"
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
    surface_backend: str = "vmec"
    face_response_mode: str = "face_local_response"
    radial_batch_size: int | None = None
    radial_batch_mode: str = "simple"
    scan_batch_size: int | None = None
    response_anchor_count: int | None = None
    use_remat: bool = False
    derivative_mode: str = "direct"
    derivative_field_pullback_mode: str = "generic_jvp"
    derivative_pullback_boundary: str = "inline"
    derivative_pullback_algebra: str = "ntx_helper"
    er_v_floor: float | None = None
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

    def with_support_payload(
        self,
        support: NTXExactLijRuntimeSupport,
    ) -> "NTXExactLijRuntimeTransportModel":
        """Return a copy with an explicit NTX support payload.

        This is the realtime-geometry AD boundary: the model/scaffolding stays a
        static Python object, while the support pytree carries differentiable
        arrays produced by the VMEC/NTX geometry path.
        """

        return dataclasses.replace(self, support=support)

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

    def with_radial_batch_mode(self, radial_batch_mode: str | None) -> "NTXExactLijRuntimeTransportModel":
        mode = self._normalize_radial_batch_mode(radial_batch_mode)
        return dataclasses.replace(self, radial_batch_mode=mode)

    def with_scan_batch_size(self, scan_batch_size: int | None) -> "NTXExactLijRuntimeTransportModel":
        normalized = None if scan_batch_size in (None, 0) else int(scan_batch_size)
        return dataclasses.replace(self, scan_batch_size=normalized)

    def with_response_anchor_count(self, response_anchor_count: int | None) -> "NTXExactLijRuntimeTransportModel":
        normalized = None if response_anchor_count in (None, 0) else int(response_anchor_count)
        return dataclasses.replace(self, response_anchor_count=normalized)

    def with_use_remat(self, use_remat: bool) -> "NTXExactLijRuntimeTransportModel":
        return dataclasses.replace(self, use_remat=bool(use_remat))

    def with_derivative_mode(self, derivative_mode: str) -> "NTXExactLijRuntimeTransportModel":
        return dataclasses.replace(self, derivative_mode=self._normalize_derivative_mode(derivative_mode))

    def with_derivative_field_pullback_mode(
        self,
        derivative_field_pullback_mode: str,
    ) -> "NTXExactLijRuntimeTransportModel":
        return dataclasses.replace(
            self,
            derivative_field_pullback_mode=self._normalize_derivative_field_pullback_mode(
                derivative_field_pullback_mode
            ),
        )

    def with_derivative_pullback_boundary(
        self,
        derivative_pullback_boundary: str,
    ) -> "NTXExactLijRuntimeTransportModel":
        return dataclasses.replace(
            self,
            derivative_pullback_boundary=self._normalize_derivative_pullback_boundary(
                derivative_pullback_boundary
            ),
        )

    def with_derivative_pullback_algebra(
        self,
        derivative_pullback_algebra: str,
    ) -> "NTXExactLijRuntimeTransportModel":
        return dataclasses.replace(
            self,
            derivative_pullback_algebra=self._normalize_derivative_pullback_algebra(
                derivative_pullback_algebra
            ),
        )

    def with_er_v_floor(self, er_v_floor: float | None) -> "NTXExactLijRuntimeTransportModel":
        normalized = None if er_v_floor in (None, "", 0, "0") else float(er_v_floor)
        return dataclasses.replace(self, er_v_floor=normalized)

    @staticmethod
    def _normalize_radial_batch_mode(radial_batch_mode: str | None) -> str:
        mode = "simple" if radial_batch_mode in (None, "") else str(radial_batch_mode).strip().lower()
        aliases = {
            "default": "simple",
            "auto": "simple",
            "lax.map": "lax_map",
            "laxmap": "lax_map",
            "map": "lax_map",
            "scan": "lax_map",
            "vmapped": "vmap",
            "chunked": "hybrid",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"simple", "lax_map", "vmap", "hybrid"}:
            raise ValueError(
                "ntx_exact_radial_batch_mode must be one of: simple, lax_map, vmap, hybrid"
            )
        return mode

    @staticmethod
    def _normalize_derivative_mode(derivative_mode: str | None) -> str:
        mode = "direct" if derivative_mode in (None, "") else str(derivative_mode).strip().lower()
        aliases = {
            "plain": "direct",
            "jax": "direct",
            "custom": "custom_vjp",
            "custom-vjp": "custom_vjp",
            "customvjp": "custom_vjp",
            "custom-jvp": "custom_jvp",
            "customjvp": "custom_jvp",
            "implicit-jvp": "custom_jvp",
            "implicit_jvp": "custom_jvp",
            "iterative": "iterative_vjp",
            "iterative-vjp": "iterative_vjp",
            "bicgstab": "iterative_vjp",
            "matrix-free": "iterative_vjp",
            "matrix_free": "iterative_vjp",
            "iterative-jvp": "iterative_jvp",
            "matrix-free-jvp": "iterative_jvp",
            "matrix_free_jvp": "iterative_jvp",
        }
        mode = aliases.get(mode, mode)
        if mode not in {
            "direct",
            "custom_jvp",
            "custom_vjp",
            "recompute_vjp",
            "iterative_vjp",
            "iterative_jvp",
        }:
            raise ValueError(
                "ntx_exact_derivative_mode must be one of: direct, custom_jvp, "
                "custom_vjp, recompute_vjp, iterative_vjp, iterative_jvp"
            )
        return mode

    @staticmethod
    def _normalize_derivative_field_pullback_mode(mode: str | None) -> str:
        normalized = "generic_jvp" if mode in (None, "") else str(mode).strip().lower()
        aliases = {
            "default": "generic_jvp",
            "generic": "generic_jvp",
            "jvp": "generic_jvp",
            "compact": "compact_vjp",
            "compact-vjp": "compact_vjp",
            "custom_vjp": "compact_vjp",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"generic_jvp", "compact_vjp"}:
            raise ValueError(
                "ntx_exact_derivative_field_pullback_mode must be one of: "
                "generic_jvp, compact_vjp"
            )
        return normalized

    @staticmethod
    def _normalize_derivative_pullback_boundary(mode: str | None) -> str:
        normalized = "inline" if mode in (None, "") else str(mode).strip().lower()
        aliases = {
            "default": "inline",
            "none": "inline",
            "off": "inline",
            "per-energy": "per_energy_jit",
            "per_energy": "per_energy_jit",
            "energy_jit": "per_energy_jit",
            "case_jit": "per_energy_jit",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"inline", "per_energy_jit"}:
            raise ValueError(
                "ntx_exact_derivative_pullback_boundary must be one of: "
                "inline, per_energy_jit"
            )
        return normalized

    @staticmethod
    def _normalize_derivative_pullback_algebra(mode: str | None) -> str:
        normalized = "ntx_helper" if mode in (None, "") else str(mode).strip().lower()
        aliases = {
            "default": "ntx_helper",
            "ntx": "ntx_helper",
            "compact": "ntx_helper",
            "compact_vjp": "ntx_helper",
            "scalar": "scalar_contract",
            "scalar-contract": "scalar_contract",
            "contract": "scalar_contract",
            "lowdot": "scalar_contract_lowdot",
            "scalar-lowdot": "scalar_contract_lowdot",
            "scalar_contract_no_f_dot": "scalar_contract_lowdot",
            "lowdot_sequential": "scalar_contract_lowdot_sequential",
            "scalar-lowdot-sequential": "scalar_contract_lowdot_sequential",
            "scalar_contract_lowdot_while": "scalar_contract_lowdot_sequential",
            "lowdot_ntx": "scalar_contract_lowdot_ntx",
            "scalar-lowdot-ntx": "scalar_contract_lowdot_ntx",
            "scalar_contract_lowdot_helper": "scalar_contract_lowdot_ntx",
            "lowdot_recompute": "scalar_contract_lowdot_recompute",
            "scalar-lowdot-recompute": "scalar_contract_lowdot_recompute",
            "scalar_contract_lowdot_replay": "scalar_contract_lowdot_recompute",
            "matrix_free": "scalar_contract_matrix_free",
            "matrix-free": "scalar_contract_matrix_free",
            "krylov": "scalar_contract_matrix_free",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in {
            "ntx_helper",
            "scalar_contract",
            "scalar_contract_lowdot",
            "scalar_contract_lowdot_sequential",
            "scalar_contract_lowdot_ntx",
            "scalar_contract_lowdot_recompute",
            "scalar_contract_matrix_free",
        }:
            raise ValueError(
                "ntx_exact_derivative_pullback_algebra must be one of: "
                "ntx_helper, scalar_contract, scalar_contract_lowdot, "
                "scalar_contract_lowdot_sequential, scalar_contract_lowdot_ntx, "
                "scalar_contract_lowdot_recompute, scalar_contract_matrix_free"
            )
        return normalized

    def _map_radius_axis_hybrid(self, fn, radius_indices):
        batch_size = self.radial_batch_size
        if batch_size is None or int(batch_size) <= 1:
            return jax.lax.map(fn, radius_indices)

        batch_size = int(batch_size)
        n_radius = int(radius_indices.shape[0])
        if n_radius <= batch_size:
            return jax.vmap(fn)(radius_indices)

        n_full = (n_radius // batch_size) * batch_size
        full_indices = radius_indices[:n_full]
        full_chunks = full_indices.reshape((n_full // batch_size, batch_size))

        chunk_outputs = jax.lax.map(lambda chunk: jax.vmap(fn)(chunk), full_chunks)
        flat_outputs = jax.tree_util.tree_map(
            lambda arr: arr.reshape((n_full,) + arr.shape[2:]),
            chunk_outputs,
        )

        if n_full == n_radius:
            return flat_outputs

        tail_indices = radius_indices[n_full:]
        tail_outputs = jax.vmap(fn)(tail_indices)
        return jax.tree_util.tree_map(
            lambda full_arr, tail_arr: jnp.concatenate([full_arr, tail_arr], axis=0),
            flat_outputs,
            tail_outputs,
        )

    def _map_radius_axis(self, fn, radius_indices):
        mode = self._normalize_radial_batch_mode(self.radial_batch_mode)
        batch_size = self.radial_batch_size
        if mode == "lax_map":
            return jax.lax.map(fn, radius_indices)
        if mode == "vmap":
            return jax.vmap(fn)(radius_indices)
        if mode == "hybrid":
            return self._map_radius_axis_hybrid(fn, radius_indices)
        if batch_size is None or int(batch_size) <= 1:
            return jax.lax.map(fn, radius_indices)
        return jax.vmap(fn)(radius_indices)

    def _map_radius_axis_unbatched(self, fn, radius_indices):
        return jax.lax.map(fn, radius_indices)

    def _response_anchor_indices(self, n_radius: int) -> jax.Array:
        anchor_count = self.response_anchor_count
        if anchor_count is None or int(anchor_count) >= n_radius or int(anchor_count) < 2:
            return jnp.arange(n_radius, dtype=jnp.int32)
        anchor_count = int(anchor_count)
        raw = [round(i * (n_radius - 1) / (anchor_count - 1)) for i in range(anchor_count)]
        anchor_indices = []
        for idx in raw:
            idx_i = int(idx)
            if not anchor_indices or idx_i != anchor_indices[-1]:
                anchor_indices.append(idx_i)
        if anchor_indices[-1] != n_radius - 1:
            anchor_indices[-1] = n_radius - 1
        return jnp.asarray(anchor_indices, dtype=jnp.int32)

    def _interpolate_anchor_values(self, anchor_indices, anchor_values, target_rho):
        anchor_rho = target_rho[anchor_indices]
        n_target = int(target_rho.shape[0])
        flat_anchor = anchor_values.reshape((anchor_values.shape[0], -1)).T
        flat_full = jax.vmap(lambda values: jnp.interp(target_rho, anchor_rho, values))(flat_anchor)
        return flat_full.T.reshape((n_target,) + anchor_values.shape[1:])

    def _regularize_axis_radius0(self, values_by_radius, radius_coordinates):
        radius_coordinates = jnp.asarray(radius_coordinates, dtype=jnp.float64)
        if int(radius_coordinates.shape[0]) < 4:
            return values_by_radius

        def _regularize_leaf(leaf):
            arr = jnp.asarray(leaf)
            if arr.ndim == 0 or int(arr.shape[0]) < 4:
                return arr

            xr = jnp.asarray(radius_coordinates[0], dtype=arr.dtype)
            r1 = jnp.asarray(radius_coordinates[1], dtype=arr.dtype)
            r2 = jnp.asarray(radius_coordinates[2], dtype=arr.dtype)
            r3 = jnp.asarray(radius_coordinates[3], dtype=arr.dtype)
            xr2 = xr * xr
            xr3 = xr2 * xr
            r12 = r1 * r1
            r22 = r2 * r2
            r32 = r3 * r3
            r13 = r1 * r12
            r23 = r2 * r22
            r33 = r3 * r32
            v1 = arr[1]
            v2 = arr[2]
            v3 = arr[3]

            ha = ((v3 - v2) / (r33 - r23) - (v3 - v1) / (r33 - r13)) / (
                (r32 - r22) / (r33 - r23) - (r32 - r12) / (r33 - r13)
            )
            hb = ((v3 - v2) / (r32 - r22) - (v3 - v1) / (r32 - r12)) / (
                (r33 - r23) / (r32 - r22) - (r33 - r13) / (r32 - r12)
            )
            hg = v1 - r12 * ha - r13 * hb
            axis_value = hg + xr2 * ha + xr3 * hb
            return arr.at[0].set(axis_value)

        return jax.tree_util.tree_map(_regularize_leaf, values_by_radius)

    def _map_radius_axis_regularized_at_axis0(self, fn, radius_indices, radius_coordinates, *, unbatched: bool = False):
        radius_coordinates = jnp.asarray(radius_coordinates, dtype=jnp.float64)
        n_radius = int(radius_indices.shape[0])
        map_fn = self._map_radius_axis_unbatched if unbatched else self._map_radius_axis
        if n_radius == 0:
            return map_fn(fn, radius_indices)

        if n_radius < 4:
            return map_fn(fn, radius_indices)

        def _regularized_skip_axis(_):
            mapped_non_axis = map_fn(fn, radius_indices[1:])
            mapped_with_placeholder = jax.tree_util.tree_map(
                lambda arr: jnp.concatenate([arr[:1], arr], axis=0),
                mapped_non_axis,
            )
            return self._regularize_axis_radius0(mapped_with_placeholder, radius_coordinates)

        def _direct_map(_):
            return map_fn(fn, radius_indices)

        return jax.lax.cond(
            jnp.isclose(radius_coordinates[0], 0.0),
            _regularized_skip_axis,
            _direct_map,
            operand=None,
        )

    def _regularize_center_fluxes_axis0(self, gamma, q, upar):
        regularized_fluxes = tuple(
            jnp.swapaxes(
                self._regularize_axis_radius0(jnp.swapaxes(flux, 0, 1), self.geometry.r_grid),
                0,
                1,
            )
            for flux in (gamma, q, upar)
        )
        return regularized_fluxes

    def _log_nu_star_from_nu_hat(self, nu_hat_a):
        weights = jnp.asarray(self.energy_grid.xWeights, dtype=jnp.float64)
        weights = weights / jnp.maximum(jnp.sum(weights), 1.0e-30)
        safe_nu_hat = jnp.maximum(jnp.asarray(nu_hat_a, dtype=jnp.float64), 1.0e-30)
        return jnp.sum(weights * jnp.log(safe_nu_hat))

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
        # Feed the exact NTX runtime with the same field convention used by the
        # file/database benchmarks: epsi_hat = Es / v_new, with Es = Er * dr/ds.
        # Preserve the forward regularized-axis philosophy here as well:
        # raw centerline `drds` entries may be singular (`inf`), but forward
        # regularizes the axis from neighboring radii instead of treating this
        # singular local primitive as a physical NTX input. Reverse/replay can
        # still touch the local primitive transiently, so force the raw
        # singular lane to contribute zero Es instead of injecting `nan/inf`
        # into `epsi_hat`.
        drds_is_finite = jnp.isfinite(drds_value)
        er_times_drds = jnp.where(
            drds_is_finite,
            jnp.asarray(er_value * drds_value, dtype=jnp.result_type(er_value, drds_value, jnp.float64)),
            jnp.asarray(0.0, dtype=jnp.result_type(er_value, drds_value, jnp.float64)),
        )
        epsi_hat_a = er_times_drds * 1.0e3 / v_new_a
        if self.er_v_floor is not None:
            er_v_floor = jnp.asarray(self.er_v_floor, dtype=jnp.float64)
            sign = jnp.where(epsi_hat_a < 0.0, -1.0, 1.0)
            epsi_hat_a = jnp.where(
                drds_is_finite,
                sign * jnp.maximum(jnp.abs(jnp.asarray(epsi_hat_a, dtype=jnp.float64)), er_v_floor),
                jnp.asarray(0.0, dtype=jnp.float64),
            )
        if _ntx_local_pullback_finite_debug_enabled():
            def _local_scan_debug_callback(
                species_index_value,
                drds_value_in,
                er_value_in,
                temperature_local_in,
                density_local_in,
                vthermal_local_in,
                vth_a_in,
                v_new_a_in,
                epsi_hat_a_in,
            ):
                epsi_arr = np.asarray(epsi_hat_a_in)
                if np.issubdtype(epsi_arr.dtype, np.inexact) and not np.all(np.isfinite(epsi_arr)):
                    print(
                        "[autodiff-gate] ntx-local-scan-nonfinite "
                        f"species_index={int(np.asarray(species_index_value))} "
                        f"drds={np.asarray(drds_value_in)} "
                        f"er={np.asarray(er_value_in)} "
                        f"temperature_local={np.asarray(temperature_local_in)} "
                        f"density_local={np.asarray(density_local_in)} "
                        f"vthermal_local={np.asarray(vthermal_local_in)} "
                        f"vth_a={np.asarray(vth_a_in)} "
                        f"v_new_a={np.asarray(v_new_a_in)} "
                        f"epsi_hat={epsi_arr}"
                    )

            jax.debug.callback(
                _local_scan_debug_callback,
                jnp.asarray(species_index, dtype=jnp.int32),
                drds_value,
                er_value,
                temperature_local,
                density_local,
                vthermal_local,
                vth_a,
                v_new_a,
                epsi_hat_a,
                ordered=True,
            )
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
        # Match the database convention, which floors physical D11 before taking log10.
        d11_physical = jnp.asarray(coeff_scan[:, 0], dtype=jnp.float64) * drds_value**2
        d11_physical = jnp.maximum(d11_physical, jnp.asarray(D11_POSITIVE_FLOOR, dtype=jnp.float64))
        d11_a = -d11_physical
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

    def _transport_moments_from_single_coefficient_vector(
        self,
        coefficient_vector,
        *,
        drds_value,
        energy_index,
    ):
        d11_physical = jnp.asarray(coefficient_vector[0], dtype=jnp.float64) * drds_value**2
        d11_physical = jnp.maximum(d11_physical, jnp.asarray(D11_POSITIVE_FLOOR, dtype=jnp.float64))
        d11_a = -d11_physical
        d13_a = -(jnp.asarray(coefficient_vector[2], dtype=jnp.float64) * drds_value)
        d33_a = -jnp.asarray(coefficient_vector[3], dtype=jnp.float64)
        idx = jnp.asarray(energy_index, dtype=jnp.int32)
        weighted_l11 = self.energy_grid.L11_weight[idx] * self.energy_grid.xWeights[idx]
        weighted_l12 = self.energy_grid.L12_weight[idx] * self.energy_grid.xWeights[idx]
        weighted_l22 = self.energy_grid.L22_weight[idx] * self.energy_grid.xWeights[idx]
        weighted_l13 = self.energy_grid.L13_weight[idx] * self.energy_grid.xWeights[idx]
        weighted_l23 = self.energy_grid.L23_weight[idx] * self.energy_grid.xWeights[idx]
        weighted_l33 = self.energy_grid.L33_weight[idx] * self.energy_grid.xWeights[idx]
        return jnp.stack(
            [
                weighted_l11 * d11_a,
                weighted_l12 * d11_a,
                weighted_l22 * d11_a,
                weighted_l13 * d13_a,
                weighted_l23 * d13_a,
                weighted_l33 * d33_a,
            ],
            axis=0,
        )

    def _pullback_transport_moments_from_single_coefficient_vector(
        self,
        coefficient_vector,
        *,
        drds_value,
        energy_index,
        transport_moments_bar,
    ):
        coeff_bar = jnp.zeros_like(coefficient_vector)
        idx = jnp.asarray(energy_index, dtype=jnp.int32)
        weighted_l11 = self.energy_grid.L11_weight[idx] * self.energy_grid.xWeights[idx]
        weighted_l12 = self.energy_grid.L12_weight[idx] * self.energy_grid.xWeights[idx]
        weighted_l22 = self.energy_grid.L22_weight[idx] * self.energy_grid.xWeights[idx]
        weighted_l13 = self.energy_grid.L13_weight[idx] * self.energy_grid.xWeights[idx]
        weighted_l23 = self.energy_grid.L23_weight[idx] * self.energy_grid.xWeights[idx]
        weighted_l33 = self.energy_grid.L33_weight[idx] * self.energy_grid.xWeights[idx]

        d11_a_bar = (
            weighted_l11 * transport_moments_bar[0]
            + weighted_l12 * transport_moments_bar[1]
            + weighted_l22 * transport_moments_bar[2]
        )
        d13_a_bar = (
            weighted_l13 * transport_moments_bar[3]
            + weighted_l23 * transport_moments_bar[4]
        )
        d33_a_bar = weighted_l33 * transport_moments_bar[5]

        drds_is_finite = jnp.isfinite(drds_value)
        safe_drds = jnp.where(
            drds_is_finite,
            jnp.asarray(drds_value, dtype=jnp.float64),
            jnp.asarray(0.0, dtype=jnp.float64),
        )
        safe_drds_sq = safe_drds**2

        d11_physical = jnp.asarray(coefficient_vector[0], dtype=jnp.float64) * safe_drds_sq
        d11_active = jnp.asarray(
            d11_physical >= jnp.asarray(D11_POSITIVE_FLOOR, dtype=jnp.float64),
            dtype=jnp.float64,
        )
        coeff_bar = coeff_bar.at[0].add(
            jnp.asarray(-safe_drds_sq * d11_active * d11_a_bar, dtype=coefficient_vector.dtype)
        )
        coeff_bar = coeff_bar.at[2].add(
            jnp.asarray(-safe_drds * d13_a_bar, dtype=coefficient_vector.dtype)
        )
        coeff_bar = coeff_bar.at[3].add(
            jnp.asarray(-d33_a_bar, dtype=coefficient_vector.dtype)
        )
        return coeff_bar

    def _single_coefficient_vector_from_inputs(
        self,
        prepared,
        nu_hat_value,
        epsi_hat_value,
        *,
        derivative_mode_override=None,
    ):
        ntx = _import_ntx()
        derivative_mode = (
            self._normalize_derivative_mode(self.derivative_mode)
            if derivative_mode_override is None
            else self._normalize_derivative_mode(derivative_mode_override)
        )
        case = ntx.MonoenergeticCase(nu_hat=nu_hat_value, epsi_hat=epsi_hat_value)
        return _ntx_prepared_coefficient_vector_solver(ntx, derivative_mode)(prepared, case)

    def _single_energy_transport_moment_from_inputs(
        self,
        prepared,
        nu_hat_value,
        epsi_hat_value,
        *,
        drds_value,
        energy_index,
        derivative_mode_override=None,
    ):
        return self._transport_moments_from_single_coefficient_vector(
            self._single_coefficient_vector_from_inputs(
                prepared,
                nu_hat_value,
                epsi_hat_value,
                derivative_mode_override=derivative_mode_override,
            ),
            drds_value=drds_value,
            energy_index=energy_index,
        )

    def _transport_moments_from_inputs_impl(
        self,
        prepared,
        nu_hat_a,
        epsi_hat_a,
        *,
        drds_value,
        derivative_mode_override=None,
    ):
        coeff_scan = self._coefficient_scan_from_inputs(
            prepared,
            nu_hat_a,
            epsi_hat_a,
            derivative_mode_override=derivative_mode_override,
        )
        return self._transport_moments_from_coefficient_scan(
            coeff_scan,
            drds_value=drds_value,
        )

    def _transport_moments_from_inputs(
        self,
        prepared,
        nu_hat_a,
        epsi_hat_a,
        *,
        drds_value,
        derivative_mode_override=None,
    ):
        return self._transport_moments_from_inputs_impl(
            prepared,
            nu_hat_a,
            epsi_hat_a,
            drds_value=drds_value,
            derivative_mode_override=derivative_mode_override,
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

    def _solve_coefficient_scan_prepared_impl(self, prepared, nu_hat_a, epsi_hat_a, *, derivative_mode_override=None):
        ntx = _import_ntx()
        derivative_mode = (
            self._normalize_derivative_mode(self.derivative_mode)
            if derivative_mode_override is None
            else self._normalize_derivative_mode(derivative_mode_override)
        )
        solve_one_coefficient_vector = _ntx_prepared_coefficient_vector_solver(
            ntx,
            derivative_mode,
        )

        def _solve_one(nu_hat_value, epsi_hat_value):
            case = ntx.MonoenergeticCase(nu_hat=nu_hat_value, epsi_hat=epsi_hat_value)
            return solve_one_coefficient_vector(prepared, case)
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

    def _solve_coefficient_scan_prepared(self, prepared, nu_hat_a, epsi_hat_a, *, derivative_mode_override=None):
        evaluator = self._solve_coefficient_scan_prepared_impl
        if self.use_remat:
            evaluator = jax.checkpoint(evaluator)
        return evaluator(prepared, nu_hat_a, epsi_hat_a, derivative_mode_override=derivative_mode_override)

    def _coefficient_scan_from_inputs(self, prepared, nu_hat_a, epsi_hat_a, *, derivative_mode_override=None):
        return self._solve_coefficient_scan_prepared(
            prepared,
            nu_hat_a,
            epsi_hat_a,
            derivative_mode_override=derivative_mode_override,
        )

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
        derivative_mode_override=None,
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
        coeff_scan = self._solve_coefficient_scan_prepared(
            prepared,
            nu_hat_a,
            epsi_hat_a,
            derivative_mode_override=derivative_mode_override,
        )
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
        derivative_mode_override=None,
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
            derivative_mode_override=derivative_mode_override,
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

    def _interpolated_moment_local_scan_primitives(
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
        return self._local_scan_inputs(
            drds_value=drds_value,
            species_index=species_index,
            er_value=er_value,
            temperature_local=temperature_local,
            density_local=density_local,
            vthermal_local=vthermal_local,
            collisionality_kind=collisionality_kind,
        )

    def _build_interpolated_moment_response_local(
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
        reference_nu_hat, reference_epsi_hat, vth_a = self._interpolated_moment_local_scan_primitives(
            drds_value=drds_value,
            species_index=species_index,
            er_value=er_value,
            temperature_local=temperature_local,
            density_local=density_local,
            vthermal_local=vthermal_local,
            collisionality_kind=collisionality_kind,
        )
        return self._interpolated_moment_reduced_local_outputs_from_primitives(
            prepared,
            drds_value=drds_value,
            nu_hat_a=reference_nu_hat,
            epsi_hat_a=reference_epsi_hat,
            vth_a=vth_a,
        )

    def _interpolated_moment_reduced_local_outputs_from_primitives(
        self,
        prepared,
        *,
        drds_value,
        nu_hat_a,
        epsi_hat_a,
        vth_a,
    ):
        return (
            self._log_nu_star_from_nu_hat(nu_hat_a),
            self._transport_moments_from_inputs(
                prepared,
                nu_hat_a,
                epsi_hat_a,
                drds_value=drds_value,
            ),
            self._dtransport_moments_d_er_from_scan_primitives(
                prepared,
                drds_value=drds_value,
                nu_hat_a=nu_hat_a,
                epsi_hat_a=epsi_hat_a,
                vth_a=vth_a,
            ),
            self._dtransport_moments_d_log_nu_star_from_scan_primitives(
                prepared,
                drds_value=drds_value,
                nu_hat_a=nu_hat_a,
                epsi_hat_a=epsi_hat_a,
            ),
        )

    def _pullback_transport_moments_from_coefficient_scan(
        self,
        coeff_scan,
        *,
        drds_value,
        transport_moments_bar,
    ):
        coeff_scan_bar = jnp.zeros_like(coeff_scan)
        weighted_l11 = self.energy_grid.L11_weight * self.energy_grid.xWeights
        weighted_l12 = self.energy_grid.L12_weight * self.energy_grid.xWeights
        weighted_l22 = self.energy_grid.L22_weight * self.energy_grid.xWeights
        weighted_l13 = self.energy_grid.L13_weight * self.energy_grid.xWeights
        weighted_l23 = self.energy_grid.L23_weight * self.energy_grid.xWeights
        weighted_l33 = self.energy_grid.L33_weight * self.energy_grid.xWeights

        d11_a_bar = (
            weighted_l11 * transport_moments_bar[0]
            + weighted_l12 * transport_moments_bar[1]
            + weighted_l22 * transport_moments_bar[2]
        )
        d13_a_bar = (
            weighted_l13 * transport_moments_bar[3]
            + weighted_l23 * transport_moments_bar[4]
        )
        d33_a_bar = weighted_l33 * transport_moments_bar[5]

        drds_is_finite = jnp.isfinite(drds_value)
        safe_drds = jnp.where(
            drds_is_finite,
            jnp.asarray(drds_value, dtype=jnp.float64),
            jnp.asarray(0.0, dtype=jnp.float64),
        )
        safe_drds_sq = safe_drds**2

        d11_physical = jnp.asarray(coeff_scan[:, 0], dtype=jnp.float64) * safe_drds_sq
        d11_active = jnp.asarray(
            d11_physical >= jnp.asarray(D11_POSITIVE_FLOOR, dtype=jnp.float64),
            dtype=jnp.float64,
        )
        coeff_scan_bar = coeff_scan_bar.at[:, 0].add(
            jnp.asarray(-safe_drds_sq * d11_active * d11_a_bar, dtype=coeff_scan.dtype)
        )
        coeff_scan_bar = coeff_scan_bar.at[:, 2].add(
            jnp.asarray(-safe_drds * d13_a_bar, dtype=coeff_scan.dtype)
        )
        coeff_scan_bar = coeff_scan_bar.at[:, 3].add(
            jnp.asarray(-d33_a_bar, dtype=coeff_scan.dtype)
        )
        return coeff_scan_bar

    def _pullback_transport_moments_from_scan_primitives(
        self,
        prepared,
        *,
        drds_value,
        reference_nu_hat,
        reference_epsi_hat,
        reference_transport_moments_bar,
    ):
        # Reuse NTX's lower-level adjoint algebra directly so this reverse lane
        # stays on the NEOPAX side and never pushes a traced `prepared` through
        # NTX's `custom_vjp(..., nondiff_argnums=(0,))` wrapper.
        from ntx._solver_adjoint import (
            _coefficient_mode_pullback,
            _parameter_gradient_from_adjoint,
            _prepared_implicit_vjp_primal,
        )
        from ntx._solver_context import _operator_context
        from ntx._solver_factorization import _solve_factorized_adjoint

        energy_indices = jnp.arange(reference_nu_hat.shape[0], dtype=jnp.int32)

        def _one_case_pullback(energy_index):
            nu_hat_value = reference_nu_hat[energy_index]
            epsi_hat_value = reference_epsi_hat[energy_index]
            (
                coefficients,
                f1_full_value,
                f3_full_value,
                saved_lu_value,
                saved_piv_value,
                saved_lower_value,
                saved_upper_value,
            ) = _prepared_implicit_vjp_primal(
                prepared,
                nu_hat_value,
                epsi_hat_value,
            )
            coefficient_bar = self._pullback_transport_moments_from_single_coefficient_vector(
                coefficients,
                drds_value=drds_value,
                energy_index=energy_index,
                transport_moments_bar=reference_transport_moments_bar,
            )
            ctx = _operator_context(
                prepared.surface,
                prepared.geometry,
                prepared.grid,
                nu_hat_value,
                epsi_hat_value,
            )
            f1_bar_low, f3_bar_low, nu_bar_direct = _coefficient_mode_pullback(
                prepared.geometry,
                f1_full_value[:3],
                f3_full_value[:3],
                ctx.nu_hat,
                coefficient_bar,
            )
            g1 = jnp.zeros_like(f1_full_value).at[:3].set(f1_bar_low)
            g3 = jnp.zeros_like(f3_full_value).at[:3].set(f3_bar_low)
            lambda1 = _solve_factorized_adjoint(
                saved_lu_value,
                saved_piv_value,
                saved_lower_value,
                saved_upper_value,
                g1,
            )
            lambda3 = _solve_factorized_adjoint(
                saved_lu_value,
                saved_piv_value,
                saved_lower_value,
                saved_upper_value,
                g3,
            )
            nu_bar_implicit, epsi_bar = _parameter_gradient_from_adjoint(
                prepared,
                ctx,
                f1_full_value,
                f3_full_value,
                lambda1,
                lambda3,
            )
            nu_bar_total = nu_bar_direct + nu_bar_implicit
            if _ntx_local_pullback_finite_debug_enabled():
                def _tm_case_debug_callback(
                    energy_idx,
                    nu_value,
                    epsi_value,
                    coeff_bar_value,
                    nu_direct_value,
                    nu_implicit_value,
                    nu_total_value,
                    epsi_bar_value,
                ):
                    entries = [
                        ("nu_hat_value", nu_value),
                        ("epsi_hat_value", epsi_value),
                        ("coefficient_bar", coeff_bar_value),
                        ("nu_bar_direct", nu_direct_value),
                        ("nu_bar_implicit", nu_implicit_value),
                        ("nu_bar_total", nu_total_value),
                        ("epsi_bar", epsi_bar_value),
                    ]
                    for name, value in entries:
                        arr = np.asarray(value)
                        if not np.issubdtype(arr.dtype, np.inexact):
                            continue
                        if not np.all(np.isfinite(arr)):
                            finite_mask = np.isfinite(arr)
                            if arr.ndim == 0:
                                value_summary = f"value={arr!r}"
                            else:
                                value_summary = (
                                    f"value={arr} "
                                    f"finite_mask={finite_mask}"
                                )
                            print(
                                "[autodiff-gate] ntx-transport-pullback-nonfinite "
                                f"energy_index={int(np.asarray(energy_idx))} "
                                f"name={name} shape={arr.shape} {value_summary}"
                            )
                            break

                jax.debug.callback(
                    _tm_case_debug_callback,
                    energy_index,
                    nu_hat_value,
                    epsi_hat_value,
                    coefficient_bar,
                    nu_bar_direct,
                    nu_bar_implicit,
                    nu_bar_total,
                    epsi_bar,
                    ordered=True,
                )
            return nu_bar_total, epsi_bar

        nu_hat_bar, epsi_hat_bar = jax.lax.map(
            _one_case_pullback,
            energy_indices,
        )
        return nu_hat_bar, epsi_hat_bar

    def _compact_coefficient_derivative_pullback_from_scan_primitives(
        self,
        prepared,
        *,
        drds_value,
        reference_nu_hat,
        reference_epsi_hat,
        nu_hat_tangent,
        epsi_hat_tangent,
        reference_transport_moments_bar,
    ):
        ntx = _import_ntx()
        derivative_pullback = _ntx_prepared_coefficient_vector_derivative_pullback(ntx)
        energy_indices = jnp.arange(reference_nu_hat.shape[0], dtype=jnp.int32)

        def _one_case_pullback(args):
            energy_index, nu_hat_value, epsi_hat_value, nu_hat_dot, epsi_hat_dot = args
            case = ntx.MonoenergeticCase(nu_hat=nu_hat_value, epsi_hat=epsi_hat_value)
            case_dot = ntx.MonoenergeticCase(nu_hat=nu_hat_dot, epsi_hat=epsi_hat_dot)
            coefficient_vector = self._single_coefficient_vector_from_inputs(
                prepared,
                nu_hat_value,
                epsi_hat_value,
                derivative_mode_override="direct",
            )
            coefficient_bar = self._pullback_transport_moments_from_single_coefficient_vector(
                coefficient_vector,
                drds_value=drds_value,
                energy_index=energy_index,
                transport_moments_bar=reference_transport_moments_bar,
            )
            base_case_bar, tangent_case_bar = derivative_pullback(
                prepared,
                case,
                case_dot,
                coefficient_bar,
            )
            return (
                base_case_bar.nu_hat,
                jnp.zeros_like(epsi_hat_value)
                if base_case_bar.epsi_hat is None
                else base_case_bar.epsi_hat,
                tangent_case_bar.nu_hat,
                jnp.zeros_like(epsi_hat_value)
                if tangent_case_bar.epsi_hat is None
                else tangent_case_bar.epsi_hat,
            )

        derivative_pullback_boundary = self._normalize_derivative_pullback_boundary(
            self.derivative_pullback_boundary
        )
        per_case_pullback = (
            jax.jit(_one_case_pullback, inline=False)
            if derivative_pullback_boundary == "per_energy_jit"
            else _one_case_pullback
        )
        return jax.lax.map(
            per_case_pullback,
            (
                energy_indices,
                reference_nu_hat,
                reference_epsi_hat,
                nu_hat_tangent,
                epsi_hat_tangent,
            ),
        )

    def _scalar_contract_coefficient_derivative_pullback_from_scan_primitives(
        self,
        prepared,
        *,
        drds_value,
        reference_nu_hat,
        reference_epsi_hat,
        nu_hat_tangent,
        epsi_hat_tangent,
        reference_transport_moments_bar,
    ):
        from ntx._solver_adjoint import (
            _coefficient_mode_pullback,
            _prepared_implicit_vjp_primal,
        )
        from ntx._solver_context import _operator_context
        from ntx.operators import (
            apply_nullspace_condition,
            operator_blocks,
            parameter_derivative_blocks,
            source_modes,
        )
        from ntx.transport import coefficients_from_modes
        from jax.scipy.linalg import lu_factor, lu_solve

        energy_indices = jnp.arange(reference_nu_hat.shape[0], dtype=jnp.int32)

        def _zero_first_row_if_needed(block, k):
            zeroed = block.at[0, :].set(jnp.zeros((block.shape[1],), dtype=block.dtype))
            return jnp.where(jnp.asarray(k) == 0, zeroed, block)

        def _take_mode(values, k):
            return jax.lax.dynamic_index_in_dim(values, k, axis=0, keepdims=False)

        def _solve_factorized_modes_scan(
            saved_lu,
            saved_piv,
            saved_lower,
            saved_upper,
            source,
        ):
            mode_count = source.shape[0]
            last_index = mode_count - 1
            y_last = lu_solve(
                (_take_mode(saved_lu, last_index), _take_mode(saved_piv, last_index)),
                _take_mode(source, last_index),
            )

            def _backward_y(y_next, k):
                rhs = _take_mode(source, k) - _take_mode(saved_upper, k) @ y_next
                y_k = lu_solve(
                    (_take_mode(saved_lu, k), _take_mode(saved_piv, k)),
                    rhs,
                )
                return y_k, y_k

            _, y_tail = jax.lax.scan(
                _backward_y,
                y_last,
                jnp.arange(last_index, dtype=jnp.int32),
                reverse=True,
            )
            y = jnp.concatenate([y_tail, y_last[None, ...]], axis=0)
            mode0 = _take_mode(y, 0)

            def _forward_mode(mode_prev, k):
                propagated = lu_solve(
                    (_take_mode(saved_lu, k), _take_mode(saved_piv, k)),
                    _take_mode(saved_lower, k) @ mode_prev,
                )
                mode_k = _take_mode(y, k) - propagated
                return mode_k, mode_k

            _, mode_tail = jax.lax.scan(
                _forward_mode,
                mode0,
                jnp.arange(1, mode_count, dtype=jnp.int32),
            )
            return jnp.concatenate([mode0[None, ...], mode_tail], axis=0)

        def _solve_factorized_adjoint_scan(
            saved_lu,
            saved_piv,
            saved_lower,
            saved_upper,
            source_bar,
        ):
            mode_count = source_bar.shape[0]
            last_index = mode_count - 1
            mu_last = _take_mode(source_bar, last_index)

            def _backward_mu(mu_next, k):
                propagated = lu_solve(
                    (
                        _take_mode(saved_lu, k + 1),
                        _take_mode(saved_piv, k + 1),
                    ),
                    mu_next,
                    trans=1,
                )
                mu_k = _take_mode(source_bar, k) - _take_mode(saved_lower, k + 1).T @ propagated
                return mu_k, mu_k

            _, mu_tail = jax.lax.scan(
                _backward_mu,
                mu_last,
                jnp.arange(last_index, dtype=jnp.int32),
                reverse=True,
            )
            mu = jnp.concatenate([mu_tail, mu_last[None, ...]], axis=0)
            adjoint0 = lu_solve(
                (_take_mode(saved_lu, 0), _take_mode(saved_piv, 0)),
                _take_mode(mu, 0),
                trans=1,
            )

            def _forward_adjoint(adjoint_prev, k):
                rhs = _take_mode(mu, k) - _take_mode(saved_upper, k - 1).T @ adjoint_prev
                adjoint_k = lu_solve(
                    (_take_mode(saved_lu, k), _take_mode(saved_piv, k)),
                    rhs,
                    trans=1,
                )
                return adjoint_k, adjoint_k

            _, adjoint_tail = jax.lax.scan(
                _forward_adjoint,
                adjoint0,
                jnp.arange(1, mode_count, dtype=jnp.int32),
            )
            return jnp.concatenate([adjoint0[None, ...], adjoint_tail], axis=0)

        def _solve_factorized_low_modes_scan(
            saved_lu,
            saved_piv,
            saved_lower,
            saved_upper,
            source_for_mode,
        ):
            mode_count = saved_lu.shape[0]
            last_index = mode_count - 1
            source0 = source_for_mode(jnp.asarray(0, dtype=jnp.int32))
            zero = jnp.zeros_like(source0)
            y_last = lu_solve(
                (_take_mode(saved_lu, last_index), _take_mode(saved_piv, last_index)),
                source_for_mode(jnp.asarray(last_index, dtype=jnp.int32)),
            )
            y0 = jnp.where(last_index == 0, y_last, zero)
            y1 = jnp.where(last_index == 1, y_last, zero)
            y2 = jnp.where(last_index == 2, y_last, zero)

            def _backward_y(carry, k):
                y_next, y0_value, y1_value, y2_value = carry
                rhs = source_for_mode(k) - _take_mode(saved_upper, k) @ y_next
                y_k = lu_solve(
                    (_take_mode(saved_lu, k), _take_mode(saved_piv, k)),
                    rhs,
                )
                y0_value = jnp.where(k == 0, y_k, y0_value)
                y1_value = jnp.where(k == 1, y_k, y1_value)
                y2_value = jnp.where(k == 2, y_k, y2_value)
                return (y_k, y0_value, y1_value, y2_value), None

            (_, y0, y1, y2), _ = jax.lax.scan(
                _backward_y,
                (y_last, y0, y1, y2),
                jnp.arange(last_index, dtype=jnp.int32),
                reverse=True,
            )
            mode0 = y0
            mode1 = y1 - lu_solve(
                (_take_mode(saved_lu, 1), _take_mode(saved_piv, 1)),
                _take_mode(saved_lower, 1) @ mode0,
            )
            mode2 = y2 - lu_solve(
                (_take_mode(saved_lu, 2), _take_mode(saved_piv, 2)),
                _take_mode(saved_lower, 2) @ mode1,
            )
            return jnp.stack([mode0, mode1, mode2], axis=0)

        def _contract_factorized_source_bar_pair_scan(
            saved_lu,
            saved_piv,
            saved_lower,
            saved_upper,
            source_for_mode,
            source_bar_pair_for_mode,
        ):
            mode_count = saved_lu.shape[0]
            last_index = mode_count - 1
            mu_last = source_bar_pair_for_mode(jnp.asarray(last_index, dtype=jnp.int32))

            def _backward_mu(mu_next, k):
                propagated = lu_solve(
                    (
                        _take_mode(saved_lu, k + 1),
                        _take_mode(saved_piv, k + 1),
                    ),
                    mu_next,
                    trans=1,
                )
                mu_k = source_bar_pair_for_mode(k) - _take_mode(saved_lower, k + 1).T @ propagated
                return mu_k, mu_k

            _, mu_tail = jax.lax.scan(
                _backward_mu,
                mu_last,
                jnp.arange(last_index, dtype=jnp.int32),
                reverse=True,
            )
            mu = jnp.concatenate([mu_tail, mu_last[None, ...]], axis=0)
            adjoint0 = lu_solve(
                (_take_mode(saved_lu, 0), _take_mode(saved_piv, 0)),
                _take_mode(mu, 0),
                trans=1,
            )
            source0 = source_for_mode(jnp.asarray(0, dtype=jnp.int32))
            contract0 = jnp.sum(adjoint0 * source0[:, None], axis=0)

            def _forward_adjoint(carry, k):
                adjoint_prev, contract = carry
                rhs = _take_mode(mu, k) - _take_mode(saved_upper, k - 1).T @ adjoint_prev
                adjoint_k = lu_solve(
                    (_take_mode(saved_lu, k), _take_mode(saved_piv, k)),
                    rhs,
                    trans=1,
                )
                source_k = source_for_mode(k)
                contract = contract + jnp.sum(adjoint_k * source_k[:, None], axis=0)
                return (adjoint_k, contract), None

            (_, contracted), _ = jax.lax.scan(
                _forward_adjoint,
                (adjoint0, contract0),
                jnp.arange(1, mode_count, dtype=jnp.int32),
            )
            return contracted

        use_lowdot = (
            self._normalize_derivative_pullback_algebra(self.derivative_pullback_algebra)
            == "scalar_contract_lowdot"
        )
        use_matrix_free = (
            self._normalize_derivative_pullback_algebra(self.derivative_pullback_algebra)
            == "scalar_contract_matrix_free"
        )
        use_operator_solve = use_matrix_free

        def _safe_divide(numerator, denominator, dtype):
            safe_denominator = jnp.where(
                jnp.abs(denominator) > jnp.asarray(0.0, dtype=dtype),
                denominator,
                jnp.asarray(1.0, dtype=dtype),
            )
            return numerator / safe_denominator

        def _bicgstab_fixed_iterations(matvec, b, *, tol=1.0e-10, maxiter=40):
            dtype = b.dtype
            b = jnp.asarray(b, dtype=dtype)
            x0 = jnp.zeros_like(b)
            r0 = b - matvec(x0)
            r_hat = r0
            zeros = jnp.zeros_like(b)
            one = jnp.asarray(1.0, dtype=dtype)
            zero = jnp.asarray(0.0, dtype=dtype)
            norm_b = jnp.sqrt(jnp.maximum(jnp.vdot(b, b), zero))
            threshold = jnp.asarray(tol, dtype=dtype) * jnp.maximum(norm_b, one)
            initial_residual_norm = jnp.sqrt(jnp.maximum(jnp.vdot(r0, r0), zero))
            initial_state = (
                x0,
                r0,
                r_hat,
                zeros,
                zeros,
                one,
                one,
                one,
                initial_residual_norm <= threshold,
            )

            def _body(_idx, state):
                x, r, r_shadow, p, v, rho_prev, alpha_prev, omega_prev, converged = state
                rho = jnp.vdot(r_shadow, r)
                beta = _safe_divide(rho, rho_prev, dtype) * _safe_divide(
                    alpha_prev,
                    omega_prev,
                    dtype,
                )
                p_candidate = r + beta * (p - omega_prev * v)
                v_candidate = matvec(p_candidate)
                alpha = _safe_divide(rho, jnp.vdot(r_shadow, v_candidate), dtype)
                s = r - alpha * v_candidate
                t = matvec(s)
                omega = _safe_divide(jnp.vdot(t, s), jnp.vdot(t, t), dtype)
                x_candidate = x + alpha * p_candidate + omega * s
                r_candidate = s - omega * t
                residual_norm = jnp.sqrt(jnp.maximum(jnp.vdot(r_candidate, r_candidate), zero))
                converged_candidate = residual_norm <= threshold
                use_candidate = jnp.logical_not(converged)
                return (
                    jnp.where(use_candidate, x_candidate, x),
                    jnp.where(use_candidate, r_candidate, r),
                    r_shadow,
                    jnp.where(use_candidate, p_candidate, p),
                    jnp.where(use_candidate, v_candidate, v),
                    jnp.where(use_candidate, rho, rho_prev),
                    jnp.where(use_candidate, alpha, alpha_prev),
                    jnp.where(use_candidate, omega, omega_prev),
                    jnp.logical_or(converged, converged_candidate),
                )

            final_state = jax.lax.fori_loop(0, int(maxiter), _body, initial_state)
            return final_state[0]

        def _matrix_free_block_operator_solve(ctx, source, *, transpose=False):
            mode_count = source.shape[0]
            last_index = mode_count - 1

            def _fixed_blocks(k):
                lower, diagonal, upper = operator_blocks(
                    ctx,
                    k,
                    prepared.d_theta,
                    prepared.d_zeta,
                )

                def _fix_nullspace(blocks):
                    lower_in, diagonal_in, upper_in = blocks
                    diagonal_fixed, upper_fixed = apply_nullspace_condition(
                        diagonal_in,
                        upper_in,
                    )
                    return lower_in, diagonal_fixed, upper_fixed

                return jax.lax.cond(
                    k == 0,
                    _fix_nullspace,
                    lambda blocks: blocks,
                    (lower, diagonal, upper),
                )

            def _apply_primal(modes):

                def _row(k):
                    lower, diagonal, upper = _fixed_blocks(k)
                    value = diagonal @ _take_mode(modes, k)

                    def _add_lower(current):
                        return current + lower @ _take_mode(modes, k - 1)

                    def _add_upper(current):
                        return current + upper @ _take_mode(modes, k + 1)

                    value = jax.lax.cond(k > 0, _add_lower, lambda current: current, value)
                    value = jax.lax.cond(
                        k < last_index,
                        _add_upper,
                        lambda current: current,
                        value,
                    )
                    return value

                return jax.lax.map(_row, jnp.arange(mode_count, dtype=jnp.int32))

            def _apply_transpose(modes):

                def _row(k):
                    _, diagonal, _ = _fixed_blocks(k)
                    value = diagonal.T @ _take_mode(modes, k)

                    def _add_previous_upper(current):
                        _, _, upper_prev = _fixed_blocks(k - 1)
                        return current + upper_prev.T @ _take_mode(modes, k - 1)

                    def _add_next_lower(current):
                        lower_next, _, _ = _fixed_blocks(k + 1)
                        return current + lower_next.T @ _take_mode(modes, k + 1)

                    value = jax.lax.cond(
                        k > 0,
                        _add_previous_upper,
                        lambda current: current,
                        value,
                    )
                    value = jax.lax.cond(
                        k < last_index,
                        _add_next_lower,
                        lambda current: current,
                        value,
                    )
                    return value

                return jax.lax.map(_row, jnp.arange(mode_count, dtype=jnp.int32))

            matvec = _apply_transpose if transpose else _apply_primal
            solution = _bicgstab_fixed_iterations(
                matvec,
                source,
                tol=1.0e-10,
                maxiter=40,
            )
            return solution

        def _exact_recompute_block_operator_solve(ctx, source, *, transpose=False):
            """Exact block-tridiagonal solve without saved factor stacks.

            This keeps the dense per-mode block algebra from NTX's factorized
            solve, but recomputes the Schur complement factor for a requested
            mode instead of carrying the whole LU/lower/upper stack as live
            arrays in the reverse graph.
            """

            source = jnp.asarray(source, dtype=prepared.grid.jax_dtype)
            mode_count = source.shape[0]
            last_index = mode_count - 1

            def _fixed_blocks(k):
                lower, diagonal, upper = operator_blocks(
                    ctx,
                    k,
                    prepared.d_theta,
                    prepared.d_zeta,
                )

                def _fix_nullspace(blocks):
                    lower_in, diagonal_in, upper_in = blocks
                    diagonal_fixed, upper_fixed = apply_nullspace_condition(
                        diagonal_in,
                        upper_in,
                    )
                    return lower_in, diagonal_fixed, upper_fixed

                return jax.lax.cond(
                    k == 0,
                    _fix_nullspace,
                    lambda blocks: blocks,
                    (lower, diagonal, upper),
                )

            def _terminal_blocks():
                lower, diagonal, _ = operator_blocks(
                    ctx,
                    jnp.asarray(last_index, dtype=jnp.int32),
                    prepared.d_theta,
                    prepared.d_zeta,
                )
                return lower, diagonal, lower

            lower_terminal, delta_terminal, lower_next = _terminal_blocks()
            lu_terminal, piv_terminal = lu_factor(delta_terminal)
            x_terminal = lu_solve((lu_terminal, piv_terminal), lower_next)
            zeros_block = jnp.zeros_like(delta_terminal)
            zeros_piv = jnp.zeros((delta_terminal.shape[0],), dtype=jnp.int32)

            def _factor_at(target_k):
                target_k = jnp.asarray(target_k, dtype=jnp.int32)
                terminal_selected = target_k == last_index
                selected_lower = jnp.where(terminal_selected, lower_terminal, zeros_block)
                selected_upper = zeros_block
                selected_lu = jnp.where(terminal_selected, lu_terminal, zeros_block)
                selected_piv = jnp.where(terminal_selected, piv_terminal, zeros_piv)

                def _scan_factor(carry, k):
                    x_prev, lower_sel, upper_sel, lu_sel, piv_sel = carry
                    lower, diagonal, upper = _fixed_blocks(k)
                    delta_k = diagonal - upper @ x_prev
                    lu_k, piv_k = lu_factor(delta_k)
                    is_target = k == target_k
                    lower_sel = jnp.where(is_target, lower, lower_sel)
                    upper_sel = jnp.where(is_target, upper, upper_sel)
                    lu_sel = jnp.where(is_target, lu_k, lu_sel)
                    piv_sel = jnp.where(is_target, piv_k, piv_sel)
                    x_next = jax.lax.cond(
                        k > 0,
                        lambda _: lu_solve((lu_k, piv_k), lower),
                        lambda _: x_prev,
                        operand=None,
                    )
                    return (x_next, lower_sel, upper_sel, lu_sel, piv_sel), None

                (_, selected_lower, selected_upper, selected_lu, selected_piv), _ = jax.lax.scan(
                    _scan_factor,
                    (x_terminal, selected_lower, selected_upper, selected_lu, selected_piv),
                    jnp.arange(last_index - 1, -1, -1, dtype=jnp.int32),
                )
                return selected_lower, selected_lu, selected_piv, selected_upper

            def _solve_primal():
                y_terminal = lu_solve(
                    (lu_terminal, piv_terminal),
                    _take_mode(source, last_index),
                )

                def _backward_solve(carry, k):
                    x_prev, y_next = carry
                    lower, diagonal, upper = _fixed_blocks(k)
                    delta_k = diagonal - upper @ x_prev
                    lu_k, piv_k = lu_factor(delta_k)
                    rhs = _take_mode(source, k) - upper @ y_next
                    y_k = lu_solve((lu_k, piv_k), rhs)
                    x_next = jax.lax.cond(
                        k > 0,
                        lambda _: lu_solve((lu_k, piv_k), lower),
                        lambda _: x_prev,
                        operand=None,
                    )
                    return (x_next, y_k), y_k

                (_, _), y_desc = jax.lax.scan(
                    _backward_solve,
                    (x_terminal, y_terminal),
                    jnp.arange(last_index - 1, -1, -1, dtype=jnp.int32),
                )
                y = jnp.concatenate([jnp.flip(y_desc, axis=0), y_terminal[None, ...]], axis=0)
                mode0 = _take_mode(y, jnp.asarray(0, dtype=jnp.int32))

                def _forward_mode(mode_prev, k):
                    lower, lu_k, piv_k, _upper = _factor_at(k)
                    mode_k = _take_mode(y, k) - lu_solve(
                        (lu_k, piv_k),
                        lower @ mode_prev,
                    )
                    return mode_k, mode_k

                _, mode_tail = jax.lax.scan(
                    _forward_mode,
                    mode0,
                    jnp.arange(1, mode_count, dtype=jnp.int32),
                )
                return jnp.concatenate([mode0[None, ...], mode_tail], axis=0)

            def _solve_transpose():
                mu_terminal = _take_mode(source, last_index)

                def _backward_mu(mu_next, k):
                    lower_next_k, lu_next, piv_next, _upper_next = _factor_at(k + 1)
                    propagated = lu_solve((lu_next, piv_next), mu_next, trans=1)
                    mu_k = _take_mode(source, k) - lower_next_k.T @ propagated
                    return mu_k, mu_k

                _, mu_desc = jax.lax.scan(
                    _backward_mu,
                    mu_terminal,
                    jnp.arange(last_index - 1, -1, -1, dtype=jnp.int32),
                )
                mu = jnp.concatenate([jnp.flip(mu_desc, axis=0), mu_terminal[None, ...]], axis=0)
                _lower0, lu0, piv0, _upper0 = _factor_at(jnp.asarray(0, dtype=jnp.int32))
                adjoint0 = lu_solve((lu0, piv0), _take_mode(mu, 0), trans=1)

                def _forward_adjoint(adjoint_prev, k):
                    _lower, lu_k, piv_k, _upper = _factor_at(k)
                    _lower_prev, _lu_prev, _piv_prev, upper_prev = _factor_at(k - 1)
                    rhs = _take_mode(mu, k) - upper_prev.T @ adjoint_prev
                    adjoint_k = lu_solve((lu_k, piv_k), rhs, trans=1)
                    return adjoint_k, adjoint_k

                _, adjoint_tail = jax.lax.scan(
                    _forward_adjoint,
                    adjoint0,
                    jnp.arange(1, mode_count, dtype=jnp.int32),
                )
                return jnp.concatenate([adjoint0[None, ...], adjoint_tail], axis=0)

            if transpose:
                return _solve_transpose()
            return _solve_primal()

        def _block_operator_solve(ctx, source, *, transpose=False):
            return _matrix_free_block_operator_solve(ctx, source, transpose=transpose)

        def _one_case_pullback(args):
            energy_index, nu_hat_value, epsi_hat_value, nu_hat_dot, epsi_hat_dot = args
            ctx = _operator_context(
                prepared.surface,
                prepared.geometry,
                prepared.grid,
                nu_hat_value,
                epsi_hat_value,
            )
            mode_indices = jnp.arange(prepared.grid.n_xi + 1, dtype=jnp.int32)
            if use_operator_solve:
                source1, source3 = source_modes(ctx, prepared.grid.n_xi)
                f_matrix = _block_operator_solve(
                    ctx,
                    jnp.stack([source1, source3], axis=-1),
                    transpose=False,
                )
                f1_full = f_matrix[..., 0]
                f3_full = f_matrix[..., 1]
                coefficients = jnp.stack(
                    coefficients_from_modes(
                        prepared.geometry,
                        f1_full[:3],
                        f3_full[:3],
                        ctx.nu_hat,
                    )
                )
            else:
                (
                    coefficients,
                    f1_full,
                    f3_full,
                    saved_lu,
                    saved_piv,
                    saved_lower,
                    saved_upper,
                ) = _prepared_implicit_vjp_primal(
                    prepared,
                    nu_hat_value,
                    epsi_hat_value,
                )
            coefficient_bar = self._pullback_transport_moments_from_single_coefficient_vector(
                coefficients,
                drds_value=drds_value,
                energy_index=energy_index,
                transport_moments_bar=reference_transport_moments_bar,
            )

            def _source_dot_pair_for_mode(k):
                diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
                    ctx,
                    k,
                    prepared.d_theta,
                    prepared.d_zeta,
                )
                diagonal_nu = _zero_first_row_if_needed(diagonal_nu, k)
                diagonal_epsi = _zero_first_row_if_needed(diagonal_epsi, k)
                diagonal_dot = nu_hat_dot * diagonal_nu + epsi_hat_dot * diagonal_epsi
                return (
                    -(diagonal_dot @ _take_mode(f1_full, k)),
                    -(diagonal_dot @ _take_mode(f3_full, k)),
                )

            def _source1_dot_for_mode(k):
                source1_dot_k, _ = _source_dot_pair_for_mode(k)
                return source1_dot_k

            def _source3_dot_for_mode(k):
                _, source3_dot_k = _source_dot_pair_for_mode(k)
                return source3_dot_k

            def _source_dot_matrix_for_mode(k):
                source1_dot_k, source3_dot_k = _source_dot_pair_for_mode(k)
                return jnp.stack([source1_dot_k, source3_dot_k], axis=-1)

            if use_operator_solve:
                source_dot_matrix = jax.lax.map(
                    _source_dot_matrix_for_mode,
                    mode_indices,
                )
                f_dot_matrix = _block_operator_solve(
                    ctx,
                    source_dot_matrix,
                    transpose=False,
                )
                f1_dot = f_dot_matrix[..., 0]
                f3_dot = f_dot_matrix[..., 1]
                f1_dot_low = f1_dot[:3]
                f3_dot_low = f3_dot[:3]
            elif use_lowdot:
                f_dot_low_matrix = _solve_factorized_low_modes_scan(
                    saved_lu,
                    saved_piv,
                    saved_lower,
                    saved_upper,
                    _source_dot_matrix_for_mode,
                )
                f1_dot_low = f_dot_low_matrix[..., 0]
                f3_dot_low = f_dot_low_matrix[..., 1]
            else:
                source1_dot, source3_dot = jax.lax.map(_source_dot_pair_for_mode, mode_indices)
                f1_dot = _solve_factorized_modes_scan(
                    saved_lu,
                    saved_piv,
                    saved_lower,
                    saved_upper,
                    source1_dot,
                )
                f3_dot = _solve_factorized_modes_scan(
                    saved_lu,
                    saved_piv,
                    saved_lower,
                    saved_upper,
                    source3_dot,
                )
                f1_dot_low = f1_dot[:3]
                f3_dot_low = f3_dot[:3]

            def _coefficient_pullback(modes1, modes3, nu_value):
                return _coefficient_mode_pullback(
                    prepared.geometry,
                    modes1,
                    modes3,
                    nu_value,
                    coefficient_bar,
                )

            (
                f1_bar_low,
                f3_bar_low,
                nu_bar_direct,
            ), (
                f1_bar_low_dot,
                f3_bar_low_dot,
                nu_bar_direct_dot,
            ) = jax.jvp(
                _coefficient_pullback,
                (f1_full[:3], f3_full[:3], ctx.nu_hat),
                (f1_dot_low, f3_dot_low, nu_hat_dot),
            )

            g1 = jnp.zeros_like(f1_full).at[:3].set(f1_bar_low)
            g3 = jnp.zeros_like(f3_full).at[:3].set(f3_bar_low)
            g1_dot = jnp.zeros_like(f1_full).at[:3].set(f1_bar_low_dot)
            g3_dot = jnp.zeros_like(f3_full).at[:3].set(f3_bar_low_dot)

            if use_operator_solve:
                lambda_matrix = _block_operator_solve(
                    ctx,
                    jnp.stack([g1, g3], axis=-1),
                    transpose=True,
                )
                lambda1 = lambda_matrix[..., 0]
                lambda3 = lambda_matrix[..., 1]
            elif use_lowdot:
                lambda_matrix = _solve_factorized_adjoint_scan(
                    saved_lu,
                    saved_piv,
                    saved_lower,
                    saved_upper,
                    jnp.stack([g1, g3], axis=-1),
                )
                lambda1 = lambda_matrix[..., 0]
                lambda3 = lambda_matrix[..., 1]
            else:
                lambda1 = _solve_factorized_adjoint_scan(
                    saved_lu,
                    saved_piv,
                    saved_lower,
                    saved_upper,
                    g1,
                )
                lambda3 = _solve_factorized_adjoint_scan(
                    saved_lu,
                    saved_piv,
                    saved_lower,
                    saved_upper,
                    g3,
                )

            def _adjoint_rhs_dot_for_mode(k):
                diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
                    ctx,
                    k,
                    prepared.d_theta,
                    prepared.d_zeta,
                )
                diagonal_nu = _zero_first_row_if_needed(diagonal_nu, k)
                diagonal_epsi = _zero_first_row_if_needed(diagonal_epsi, k)
                diagonal_dot = nu_hat_dot * diagonal_nu + epsi_hat_dot * diagonal_epsi
                return (
                    _take_mode(g1_dot, k) - diagonal_dot.T @ _take_mode(lambda1, k),
                    _take_mode(g3_dot, k) - diagonal_dot.T @ _take_mode(lambda3, k),
                )

            if use_operator_solve:

                def _adjoint_rhs_dot_matrix_for_mode(k):
                    adjoint_rhs1_dot_k, adjoint_rhs3_dot_k = _adjoint_rhs_dot_for_mode(k)
                    return jnp.stack([adjoint_rhs1_dot_k, adjoint_rhs3_dot_k], axis=-1)

                adjoint_rhs_dot_matrix = jax.lax.map(
                    _adjoint_rhs_dot_matrix_for_mode,
                    mode_indices,
                )
                lambda_dot_matrix = _block_operator_solve(
                    ctx,
                    adjoint_rhs_dot_matrix,
                    transpose=True,
                )
                lambda1_dot = lambda_dot_matrix[..., 0]
                lambda3_dot = lambda_dot_matrix[..., 1]
            elif use_lowdot:

                def _adjoint_rhs_dot_matrix_for_mode(k):
                    adjoint_rhs1_dot_k, adjoint_rhs3_dot_k = _adjoint_rhs_dot_for_mode(k)
                    return jnp.stack([adjoint_rhs1_dot_k, adjoint_rhs3_dot_k], axis=-1)

                adjoint_rhs_dot_matrix = jax.lax.map(
                    _adjoint_rhs_dot_matrix_for_mode,
                    mode_indices,
                )
                lambda_dot_matrix = _solve_factorized_adjoint_scan(
                    saved_lu,
                    saved_piv,
                    saved_lower,
                    saved_upper,
                    adjoint_rhs_dot_matrix,
                )
                lambda1_dot = lambda_dot_matrix[..., 0]
                lambda3_dot = lambda_dot_matrix[..., 1]
            else:
                adjoint_rhs1_dot, adjoint_rhs3_dot = jax.lax.map(
                    _adjoint_rhs_dot_for_mode,
                    mode_indices,
                )
                lambda1_dot = _solve_factorized_adjoint_scan(
                    saved_lu,
                    saved_piv,
                    saved_lower,
                    saved_upper,
                    adjoint_rhs1_dot,
                )
                lambda3_dot = _solve_factorized_adjoint_scan(
                    saved_lu,
                    saved_piv,
                    saved_lower,
                    saved_upper,
                    adjoint_rhs3_dot,
                )

            def _accumulate_base_bars(carry, k):
                nu_bar, epsi_bar = carry
                diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
                    ctx,
                    k,
                    prepared.d_theta,
                    prepared.d_zeta,
                )
                diagonal_nu = _zero_first_row_if_needed(diagonal_nu, k)
                diagonal_epsi = _zero_first_row_if_needed(diagonal_epsi, k)
                f1_k = _take_mode(f1_full, k)
                f3_k = _take_mode(f3_full, k)
                lambda1_k = _take_mode(lambda1, k)
                lambda3_k = _take_mode(lambda3, k)
                nu_bar = nu_bar - (
                    jnp.vdot(lambda1_k, diagonal_nu @ f1_k)
                    + jnp.vdot(lambda3_k, diagonal_nu @ f3_k)
                )
                epsi_bar = epsi_bar - (
                    jnp.vdot(lambda1_k, diagonal_epsi @ f1_k)
                    + jnp.vdot(lambda3_k, diagonal_epsi @ f3_k)
                )
                return (nu_bar, epsi_bar), None

            (
                nu_bar_implicit,
                epsi_bar,
            ), _ = jax.lax.scan(
                _accumulate_base_bars,
                (
                    jnp.asarray(0.0, dtype=prepared.grid.jax_dtype),
                    jnp.asarray(0.0, dtype=prepared.grid.jax_dtype),
                ),
                mode_indices,
            )

            def _accumulate_lambda_dot_directional_bars(carry, k):
                nu_bar_dot, epsi_bar_dot = carry
                diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
                    ctx,
                    k,
                    prepared.d_theta,
                    prepared.d_zeta,
                )
                diagonal_nu = _zero_first_row_if_needed(diagonal_nu, k)
                diagonal_epsi = _zero_first_row_if_needed(diagonal_epsi, k)
                f1_k = _take_mode(f1_full, k)
                f3_k = _take_mode(f3_full, k)
                lambda1_dot_k = _take_mode(lambda1_dot, k)
                lambda3_dot_k = _take_mode(lambda3_dot, k)
                nu_bar_dot = nu_bar_dot - (
                    jnp.vdot(lambda1_dot_k, diagonal_nu @ f1_k)
                    + jnp.vdot(lambda3_dot_k, diagonal_nu @ f3_k)
                )
                epsi_bar_dot = epsi_bar_dot - (
                    jnp.vdot(lambda1_dot_k, diagonal_epsi @ f1_k)
                    + jnp.vdot(lambda3_dot_k, diagonal_epsi @ f3_k)
                )
                return (nu_bar_dot, epsi_bar_dot), None

            (
                nu_bar_implicit_dot,
                epsi_bar_dot,
            ), _ = jax.lax.scan(
                _accumulate_lambda_dot_directional_bars,
                (
                    jnp.asarray(0.0, dtype=prepared.grid.jax_dtype),
                    jnp.asarray(0.0, dtype=prepared.grid.jax_dtype),
                ),
                mode_indices,
            )

            def _source_bar_pair_for_mode(lambdas, k):
                diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
                    ctx,
                    k,
                    prepared.d_theta,
                    prepared.d_zeta,
                )
                diagonal_nu = _zero_first_row_if_needed(diagonal_nu, k)
                diagonal_epsi = _zero_first_row_if_needed(diagonal_epsi, k)
                lambda_k = _take_mode(lambdas, k)
                return jnp.stack(
                    [diagonal_nu.T @ lambda_k, diagonal_epsi.T @ lambda_k],
                    axis=-1,
                )

            if use_lowdot:
                f1_field_dot = _contract_factorized_source_bar_pair_scan(
                    saved_lu,
                    saved_piv,
                    saved_lower,
                    saved_upper,
                    _source1_dot_for_mode,
                    lambda k: _source_bar_pair_for_mode(lambda1, k),
                )
                f3_field_dot = _contract_factorized_source_bar_pair_scan(
                    saved_lu,
                    saved_piv,
                    saved_lower,
                    saved_upper,
                    _source3_dot_for_mode,
                    lambda k: _source_bar_pair_for_mode(lambda3, k),
                )
                nu_bar_implicit_dot = nu_bar_implicit_dot - f1_field_dot[0] - f3_field_dot[0]
                epsi_bar_dot = epsi_bar_dot - f1_field_dot[1] - f3_field_dot[1]
            else:

                def _accumulate_full_dot_directional_bars(carry, k):
                    nu_bar_dot, epsi_bar_dot = carry
                    diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
                        ctx,
                        k,
                        prepared.d_theta,
                        prepared.d_zeta,
                    )
                    diagonal_nu = _zero_first_row_if_needed(diagonal_nu, k)
                    diagonal_epsi = _zero_first_row_if_needed(diagonal_epsi, k)
                    f1_dot_k = _take_mode(f1_dot, k)
                    f3_dot_k = _take_mode(f3_dot, k)
                    lambda1_k = _take_mode(lambda1, k)
                    lambda3_k = _take_mode(lambda3, k)
                    nu_bar_dot = nu_bar_dot - (
                        jnp.vdot(lambda1_k, diagonal_nu @ f1_dot_k)
                        + jnp.vdot(lambda3_k, diagonal_nu @ f3_dot_k)
                    )
                    epsi_bar_dot = epsi_bar_dot - (
                        jnp.vdot(lambda1_k, diagonal_epsi @ f1_dot_k)
                        + jnp.vdot(lambda3_k, diagonal_epsi @ f3_dot_k)
                    )
                    return (nu_bar_dot, epsi_bar_dot), None

                (
                    nu_bar_implicit_dot,
                    epsi_bar_dot,
                ), _ = jax.lax.scan(
                    _accumulate_full_dot_directional_bars,
                    (nu_bar_implicit_dot, epsi_bar_dot),
                    mode_indices,
                )
            return (
                nu_bar_direct + nu_bar_implicit,
                epsi_bar,
                nu_bar_direct_dot + nu_bar_implicit_dot,
                epsi_bar_dot,
            )

        return jax.lax.map(
            _one_case_pullback,
            (
                energy_indices,
                reference_nu_hat,
                reference_epsi_hat,
                nu_hat_tangent,
                epsi_hat_tangent,
            ),
        )

    def _dtransport_moments_d_er_from_scan_primitives(
        self,
        prepared,
        *,
        drds_value,
        nu_hat_a,
        epsi_hat_a,
        vth_a,
    ):
        epsi_hat_tangent = jnp.asarray(1.0e3, dtype=epsi_hat_a.dtype) / (self.energy_grid.v_norm * vth_a)
        energy_indices = jnp.arange(nu_hat_a.shape[0], dtype=jnp.int32)
        normalized_derivative_mode = self._normalize_derivative_mode(self.derivative_mode)
        derivative_mode_override = (
            "iterative_jvp"
            if normalized_derivative_mode == "iterative_vjp"
            else "direct"
            if normalized_derivative_mode == "recompute_vjp"
            else None
        )

        def _per_energy(args):
            energy_index, nu_hat_value, epsi_hat_value, epsi_hat_tangent_value = args
            return jax.jvp(
                lambda nu_value, epsi_value: self._single_energy_transport_moment_from_inputs(
                    prepared,
                    nu_value,
                    epsi_value,
                    drds_value=drds_value,
                    energy_index=energy_index,
                    derivative_mode_override=derivative_mode_override,
                ),
                (nu_hat_value, epsi_hat_value),
                (jnp.asarray(0.0, dtype=nu_hat_value.dtype), epsi_hat_tangent_value),
            )[1]

        return jnp.sum(
            jax.lax.map(
                _per_energy,
                (energy_indices, nu_hat_a, epsi_hat_a, epsi_hat_tangent),
            ),
            axis=0,
        )

    def _dtransport_moments_d_log_nu_star_from_scan_primitives(
        self,
        prepared,
        *,
        drds_value,
        nu_hat_a,
        epsi_hat_a,
    ):
        energy_indices = jnp.arange(nu_hat_a.shape[0], dtype=jnp.int32)
        normalized_derivative_mode = self._normalize_derivative_mode(self.derivative_mode)
        derivative_mode_override = (
            "iterative_jvp"
            if normalized_derivative_mode == "iterative_vjp"
            else "direct"
            if normalized_derivative_mode == "recompute_vjp"
            else None
        )

        def _per_energy(args):
            energy_index, nu_hat_value, epsi_hat_value = args
            return jax.jvp(
                lambda nu_value, epsi_value: self._single_energy_transport_moment_from_inputs(
                    prepared,
                    nu_value,
                    epsi_value,
                    drds_value=drds_value,
                    energy_index=energy_index,
                    derivative_mode_override=derivative_mode_override,
                ),
                (nu_hat_value, epsi_hat_value),
                (nu_hat_value, jnp.asarray(0.0, dtype=epsi_hat_value.dtype)),
            )[1]

        return jnp.sum(
            jax.lax.map(
                _per_energy,
                (energy_indices, nu_hat_a, epsi_hat_a),
            ),
            axis=0,
        )

    def _pullback_dtransport_moments_d_er_from_scan_primitives(
        self,
        prepared,
        *,
        drds_value,
        reference_nu_hat,
        reference_epsi_hat,
        vth_a,
        dtransport_moments_d_er_bar,
    ):
        epsi_hat_tangent = jnp.asarray(1.0e3, dtype=reference_epsi_hat.dtype) / (
            self.energy_grid.v_norm * vth_a
        )
        derivative_field_pullback_mode = self._normalize_derivative_field_pullback_mode(
            self.derivative_field_pullback_mode
        )
        if derivative_field_pullback_mode == "compact_vjp":
            derivative_pullback = (
                self._scalar_contract_coefficient_derivative_pullback_from_scan_primitives
                if self._normalize_derivative_pullback_algebra(
                    self.derivative_pullback_algebra
                )
                in {
                    "scalar_contract",
                    "scalar_contract_lowdot",
                    "scalar_contract_lowdot_sequential",
                    "scalar_contract_lowdot_ntx",
                    "scalar_contract_matrix_free",
                }
                else self._compact_coefficient_derivative_pullback_from_scan_primitives
            )
            (
                base_nu_bar,
                base_epsi_bar,
                nu_hat_bar,
                epsi_hat_bar,
            ) = derivative_pullback(
                prepared,
                drds_value=drds_value,
                reference_nu_hat=reference_nu_hat,
                reference_epsi_hat=reference_epsi_hat,
                nu_hat_tangent=jnp.zeros_like(reference_nu_hat),
                epsi_hat_tangent=epsi_hat_tangent,
                reference_transport_moments_bar=dtransport_moments_d_er_bar,
            )
            vth_a_bar = jnp.sum(base_epsi_bar * (-epsi_hat_tangent / vth_a), axis=0)
            return nu_hat_bar, epsi_hat_bar, vth_a_bar

        def _transport_pullback_fn(nu_hat_value, epsi_hat_value):
            return self._pullback_transport_moments_from_scan_primitives(
                prepared,
                drds_value=drds_value,
                reference_nu_hat=nu_hat_value,
                reference_epsi_hat=epsi_hat_value,
                reference_transport_moments_bar=dtransport_moments_d_er_bar,
            )

        (base_nu_bar, base_epsi_bar), (nu_hat_bar, epsi_hat_bar) = jax.jvp(
            _transport_pullback_fn,
            (reference_nu_hat, reference_epsi_hat),
            (jnp.zeros_like(reference_nu_hat), epsi_hat_tangent),
        )
        vth_a_bar = jnp.sum(base_epsi_bar * (-epsi_hat_tangent / vth_a), axis=0)
        return nu_hat_bar, epsi_hat_bar, vth_a_bar

    def _pullback_dtransport_moments_d_log_nu_star_from_scan_primitives(
        self,
        prepared,
        *,
        drds_value,
        reference_nu_hat,
        reference_epsi_hat,
        dtransport_moments_d_log_nu_star_bar,
    ):
        derivative_field_pullback_mode = self._normalize_derivative_field_pullback_mode(
            self.derivative_field_pullback_mode
        )
        if derivative_field_pullback_mode == "compact_vjp":
            derivative_pullback = (
                self._scalar_contract_coefficient_derivative_pullback_from_scan_primitives
                if self._normalize_derivative_pullback_algebra(
                    self.derivative_pullback_algebra
                )
                in {
                    "scalar_contract",
                    "scalar_contract_lowdot",
                    "scalar_contract_lowdot_sequential",
                    "scalar_contract_lowdot_ntx",
                    "scalar_contract_matrix_free",
                }
                else self._compact_coefficient_derivative_pullback_from_scan_primitives
            )
            (
                base_nu_bar,
                _base_epsi_bar,
                nu_hat_bar,
                epsi_hat_bar,
            ) = derivative_pullback(
                prepared,
                drds_value=drds_value,
                reference_nu_hat=reference_nu_hat,
                reference_epsi_hat=reference_epsi_hat,
                nu_hat_tangent=reference_nu_hat,
                epsi_hat_tangent=jnp.zeros_like(reference_epsi_hat),
                reference_transport_moments_bar=dtransport_moments_d_log_nu_star_bar,
            )
            return nu_hat_bar + base_nu_bar, epsi_hat_bar

        def _transport_pullback_fn(nu_hat_value, epsi_hat_value):
            return self._pullback_transport_moments_from_scan_primitives(
                prepared,
                drds_value=drds_value,
                reference_nu_hat=nu_hat_value,
                reference_epsi_hat=epsi_hat_value,
                reference_transport_moments_bar=dtransport_moments_d_log_nu_star_bar,
            )

        (base_nu_bar, _base_epsi_bar), (nu_hat_bar, epsi_hat_bar) = jax.jvp(
            _transport_pullback_fn,
            (reference_nu_hat, reference_epsi_hat),
            (reference_nu_hat, jnp.zeros_like(reference_epsi_hat)),
        )
        return nu_hat_bar + base_nu_bar, epsi_hat_bar

    def _scalar_contract_lowdot_two_derivative_pullbacks_from_scan_primitives(
        self,
        prepared,
        *,
        drds_value,
        reference_nu_hat,
        reference_epsi_hat,
        base_transport_moments_bar,
        first_nu_hat_tangent,
        first_epsi_hat_tangent,
        first_transport_moments_bar,
        second_nu_hat_tangent,
        second_epsi_hat_tangent,
        second_transport_moments_bar,
    ):
        energy_indices = jnp.arange(reference_nu_hat.shape[0], dtype=jnp.int32)
        normalized_pullback_algebra = self._normalize_derivative_pullback_algebra(
            self.derivative_pullback_algebra
        )
        use_ntx_lowdot = normalized_pullback_algebra == "scalar_contract_lowdot_ntx"
        use_sequential_lowdot = (
            normalized_pullback_algebra == "scalar_contract_lowdot_sequential"
        )
        if use_ntx_lowdot:
            ntx_module = _import_ntx()

            def _one_case_pullback_ntx(args):
                (
                    energy_index,
                    nu_hat_value,
                    epsi_hat_value,
                    first_nu_dot,
                    first_epsi_dot,
                    second_nu_dot,
                    second_epsi_dot,
                ) = args

                def _coefficient_bars_from_coefficients(coefficients):
                    base_coefficient_bar = self._pullback_transport_moments_from_single_coefficient_vector(
                        coefficients,
                        drds_value=drds_value,
                        energy_index=energy_index,
                        transport_moments_bar=base_transport_moments_bar,
                    )
                    first_coefficient_bar = self._pullback_transport_moments_from_single_coefficient_vector(
                        coefficients,
                        drds_value=drds_value,
                        energy_index=energy_index,
                        transport_moments_bar=first_transport_moments_bar,
                    )
                    second_coefficient_bar = self._pullback_transport_moments_from_single_coefficient_vector(
                        coefficients,
                        drds_value=drds_value,
                        energy_index=energy_index,
                        transport_moments_bar=second_transport_moments_bar,
                    )
                    return base_coefficient_bar, first_coefficient_bar, second_coefficient_bar

                return ntx_module.solve_prepared_coefficient_vector_lowdot_two_pullbacks(
                    prepared,
                    ntx_module.MonoenergeticCase(
                        nu_hat=nu_hat_value,
                        epsi_hat=epsi_hat_value,
                    ),
                    ntx_module.MonoenergeticCase(
                        nu_hat=first_nu_dot,
                        epsi_hat=first_epsi_dot,
                    ),
                    ntx_module.MonoenergeticCase(
                        nu_hat=second_nu_dot,
                        epsi_hat=second_epsi_dot,
                    ),
                    _coefficient_bars_from_coefficients,
                )

            return jax.lax.map(
                _one_case_pullback_ntx,
                (
                    energy_indices,
                    reference_nu_hat,
                    reference_epsi_hat,
                    first_nu_hat_tangent,
                    first_epsi_hat_tangent,
                    second_nu_hat_tangent,
                    second_epsi_hat_tangent,
                ),
            )

        from ntx._solver_adjoint import (
            _coefficient_mode_pullback,
            _prepared_implicit_vjp_primal,
        )
        from ntx._solver_context import _operator_context
        from ntx.operators import (
            parameter_derivative_blocks,
        )
        from jax.scipy.linalg import lu_solve

        use_recompute_lowdot = normalized_pullback_algebra == "scalar_contract_lowdot_recompute"

        def _zero_first_row_if_needed(block, k):
            zeroed = block.at[0, :].set(jnp.zeros((block.shape[1],), dtype=block.dtype))
            return jnp.where(jnp.asarray(k) == 0, zeroed, block)

        def _take_mode(values, k):
            return jax.lax.dynamic_index_in_dim(values, k, axis=0, keepdims=False)

        def _solve_factorized_low_modes_scan(
            saved_lu,
            saved_piv,
            saved_lower,
            saved_upper,
            source_for_mode,
        ):
            mode_count = saved_lu.shape[0]
            last_index = mode_count - 1
            source0 = source_for_mode(jnp.asarray(0, dtype=jnp.int32))
            zero = jnp.zeros_like(source0)
            y_last = lu_solve(
                (_take_mode(saved_lu, last_index), _take_mode(saved_piv, last_index)),
                source_for_mode(jnp.asarray(last_index, dtype=jnp.int32)),
            )
            y0 = jnp.where(last_index == 0, y_last, zero)
            y1 = jnp.where(last_index == 1, y_last, zero)
            y2 = jnp.where(last_index == 2, y_last, zero)

            def _backward_y(carry, k):
                y_next, y0_value, y1_value, y2_value = carry
                rhs = source_for_mode(k) - _take_mode(saved_upper, k) @ y_next
                y_k = lu_solve(
                    (_take_mode(saved_lu, k), _take_mode(saved_piv, k)),
                    rhs,
                )
                y0_value = jnp.where(k == 0, y_k, y0_value)
                y1_value = jnp.where(k == 1, y_k, y1_value)
                y2_value = jnp.where(k == 2, y_k, y2_value)
                return (y_k, y0_value, y1_value, y2_value), None

            (_, y0, y1, y2), _ = jax.lax.scan(
                _backward_y,
                (y_last, y0, y1, y2),
                jnp.arange(last_index, dtype=jnp.int32),
                reverse=True,
            )
            mode0 = y0
            mode1 = y1 - lu_solve(
                (_take_mode(saved_lu, 1), _take_mode(saved_piv, 1)),
                _take_mode(saved_lower, 1) @ mode0,
            )
            mode2 = y2 - lu_solve(
                (_take_mode(saved_lu, 2), _take_mode(saved_piv, 2)),
                _take_mode(saved_lower, 2) @ mode1,
            )
            return jnp.stack([mode0, mode1, mode2], axis=0)

        def _solve_factorized_adjoint_scan(
            saved_lu,
            saved_piv,
            saved_lower,
            saved_upper,
            source_bar,
        ):
            mode_count = source_bar.shape[0]
            last_index = mode_count - 1
            mu_last = _take_mode(source_bar, last_index)

            def _backward_mu(mu_next, k):
                propagated = lu_solve(
                    (
                        _take_mode(saved_lu, k + 1),
                        _take_mode(saved_piv, k + 1),
                    ),
                    mu_next,
                    trans=1,
                )
                mu_k = _take_mode(source_bar, k) - _take_mode(saved_lower, k + 1).T @ propagated
                return mu_k, mu_k

            _, mu_tail = jax.lax.scan(
                _backward_mu,
                mu_last,
                jnp.arange(last_index, dtype=jnp.int32),
                reverse=True,
            )
            mu = jnp.concatenate([mu_tail, mu_last[None, ...]], axis=0)
            adjoint0 = lu_solve(
                (_take_mode(saved_lu, 0), _take_mode(saved_piv, 0)),
                _take_mode(mu, 0),
                trans=1,
            )

            def _forward_adjoint(adjoint_prev, k):
                rhs = _take_mode(mu, k) - _take_mode(saved_upper, k - 1).T @ adjoint_prev
                adjoint_k = lu_solve(
                    (_take_mode(saved_lu, k), _take_mode(saved_piv, k)),
                    rhs,
                    trans=1,
                )
                return adjoint_k, adjoint_k

            _, adjoint_tail = jax.lax.scan(
                _forward_adjoint,
                adjoint0,
                jnp.arange(1, mode_count, dtype=jnp.int32),
            )
            return jnp.concatenate([adjoint0[None, ...], adjoint_tail], axis=0)

        def _contract_factorized_source_bar_pair_scan(
            saved_lu,
            saved_piv,
            saved_lower,
            saved_upper,
            source_for_mode,
            source_bar_pair_for_mode,
        ):
            mode_count = saved_lu.shape[0]
            last_index = mode_count - 1
            mu_last = source_bar_pair_for_mode(jnp.asarray(last_index, dtype=jnp.int32))

            def _backward_mu(mu_next, k):
                propagated = lu_solve(
                    (
                        _take_mode(saved_lu, k + 1),
                        _take_mode(saved_piv, k + 1),
                    ),
                    mu_next,
                    trans=1,
                )
                mu_k = source_bar_pair_for_mode(k) - _take_mode(
                    saved_lower, k + 1
                ).T @ propagated
                return mu_k, mu_k

            _, mu_tail = jax.lax.scan(
                _backward_mu,
                mu_last,
                jnp.arange(last_index, dtype=jnp.int32),
                reverse=True,
            )
            mu = jnp.concatenate([mu_tail, mu_last[None, ...]], axis=0)
            adjoint0 = lu_solve(
                (_take_mode(saved_lu, 0), _take_mode(saved_piv, 0)),
                _take_mode(mu, 0),
                trans=1,
            )
            source0 = source_for_mode(jnp.asarray(0, dtype=jnp.int32))
            contract0 = jnp.sum(adjoint0 * source0[:, None], axis=0)

            def _forward_adjoint(carry, k):
                adjoint_prev, contract = carry
                rhs = _take_mode(mu, k) - _take_mode(saved_upper, k - 1).T @ adjoint_prev
                adjoint_k = lu_solve(
                    (_take_mode(saved_lu, k), _take_mode(saved_piv, k)),
                    rhs,
                    trans=1,
                )
                source_k = source_for_mode(k)
                contract = contract + jnp.sum(adjoint_k * source_k[:, None], axis=0)
                return (adjoint_k, contract), None

            (_, contracted), _ = jax.lax.scan(
                _forward_adjoint,
                (adjoint0, contract0),
                jnp.arange(1, mode_count, dtype=jnp.int32),
            )
            return contracted

        def _base_pullback(
            ctx,
            mode_indices,
            energy_index,
            coefficients,
            f1_full,
            f3_full,
            saved_lu,
            saved_piv,
            saved_lower,
            saved_upper,
            transport_moments_bar,
        ):
            coefficient_bar = self._pullback_transport_moments_from_single_coefficient_vector(
                coefficients,
                drds_value=drds_value,
                energy_index=energy_index,
                transport_moments_bar=transport_moments_bar,
            )
            f1_bar_low, f3_bar_low, nu_bar_direct = _coefficient_mode_pullback(
                prepared.geometry,
                f1_full[:3],
                f3_full[:3],
                ctx.nu_hat,
                coefficient_bar,
            )
            g1 = jnp.zeros_like(f1_full).at[:3].set(f1_bar_low)
            g3 = jnp.zeros_like(f3_full).at[:3].set(f3_bar_low)

            def _source_bar_matrix_for_mode(k):
                return jnp.stack([_take_mode(g1, k), _take_mode(g3, k)], axis=-1)

            def _parameter_source_matrix_for_mode(k):
                diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
                    ctx,
                    k,
                    prepared.d_theta,
                    prepared.d_zeta,
                )
                diagonal_nu = _zero_first_row_if_needed(diagonal_nu, k)
                diagonal_epsi = _zero_first_row_if_needed(diagonal_epsi, k)
                f1_k = _take_mode(f1_full, k)
                f3_k = _take_mode(f3_full, k)
                return jnp.stack(
                    [
                        jnp.stack([diagonal_nu @ f1_k, diagonal_nu @ f3_k], axis=-1),
                        jnp.stack([diagonal_epsi @ f1_k, diagonal_epsi @ f3_k], axis=-1),
                    ],
                    axis=-1,
                )

            def _contract_factorized_parameter_sources_scan():
                mode_count = saved_lu.shape[0]
                last_index = mode_count - 1
                mu_last = _source_bar_matrix_for_mode(jnp.asarray(last_index, dtype=jnp.int32))

                def _backward_mu(mu_next, k):
                    propagated = lu_solve(
                        (
                            _take_mode(saved_lu, k + 1),
                            _take_mode(saved_piv, k + 1),
                        ),
                        mu_next,
                        trans=1,
                    )
                    mu_k = _source_bar_matrix_for_mode(k) - _take_mode(
                        saved_lower, k + 1
                    ).T @ propagated
                    return mu_k, mu_k

                _, mu_tail = jax.lax.scan(
                    _backward_mu,
                    mu_last,
                    jnp.arange(last_index, dtype=jnp.int32),
                    reverse=True,
                )
                mu = jnp.concatenate([mu_tail, mu_last[None, ...]], axis=0)
                adjoint0 = lu_solve(
                    (_take_mode(saved_lu, 0), _take_mode(saved_piv, 0)),
                    _take_mode(mu, 0),
                    trans=1,
                )
                source0 = _parameter_source_matrix_for_mode(jnp.asarray(0, dtype=jnp.int32))
                contract0 = jnp.sum(adjoint0[..., None] * source0, axis=(0, 1))

                def _forward_adjoint(carry, k):
                    adjoint_prev, contract = carry
                    rhs = _take_mode(mu, k) - _take_mode(saved_upper, k - 1).T @ adjoint_prev
                    adjoint_k = lu_solve(
                        (_take_mode(saved_lu, k), _take_mode(saved_piv, k)),
                        rhs,
                        trans=1,
                    )
                    source_k = _parameter_source_matrix_for_mode(k)
                    contract = contract + jnp.sum(adjoint_k[..., None] * source_k, axis=(0, 1))
                    return (adjoint_k, contract), None

                (_, contracted), _ = jax.lax.scan(
                    _forward_adjoint,
                    (adjoint0, contract0),
                    jnp.arange(1, mode_count, dtype=jnp.int32),
                )
                return contracted

            nu_bar_implicit, epsi_bar = -_contract_factorized_parameter_sources_scan()
            return nu_bar_direct + nu_bar_implicit, epsi_bar

        def _one_direction_pullback(
            ctx,
            mode_indices,
            energy_index,
            coefficients,
            f1_full,
            f3_full,
            saved_lu,
            saved_piv,
            saved_lower,
            saved_upper,
            nu_hat_dot,
            epsi_hat_dot,
            transport_moments_bar,
            f1_dot_low,
            f3_dot_low,
        ):
            coefficient_bar = self._pullback_transport_moments_from_single_coefficient_vector(
                coefficients,
                drds_value=drds_value,
                energy_index=energy_index,
                transport_moments_bar=transport_moments_bar,
            )

            def _source_dot_pair_for_mode(k):
                diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
                    ctx,
                    k,
                    prepared.d_theta,
                    prepared.d_zeta,
                )
                diagonal_nu = _zero_first_row_if_needed(diagonal_nu, k)
                diagonal_epsi = _zero_first_row_if_needed(diagonal_epsi, k)
                diagonal_dot = nu_hat_dot * diagonal_nu + epsi_hat_dot * diagonal_epsi
                return (
                    -(diagonal_dot @ _take_mode(f1_full, k)),
                    -(diagonal_dot @ _take_mode(f3_full, k)),
                )

            def _source1_dot_for_mode(k):
                source1_dot_k, _ = _source_dot_pair_for_mode(k)
                return source1_dot_k

            def _source3_dot_for_mode(k):
                _, source3_dot_k = _source_dot_pair_for_mode(k)
                return source3_dot_k

            def _coefficient_pullback(modes1, modes3, nu_value):
                return _coefficient_mode_pullback(
                    prepared.geometry,
                    modes1,
                    modes3,
                    nu_value,
                    coefficient_bar,
                )

            (
                f1_bar_low,
                f3_bar_low,
                nu_bar_direct,
            ), (
                f1_bar_low_dot,
                f3_bar_low_dot,
                nu_bar_direct_dot,
            ) = jax.jvp(
                _coefficient_pullback,
                (f1_full[:3], f3_full[:3], ctx.nu_hat),
                (f1_dot_low, f3_dot_low, nu_hat_dot),
            )

            g1 = jnp.zeros_like(f1_full).at[:3].set(f1_bar_low)
            g3 = jnp.zeros_like(f3_full).at[:3].set(f3_bar_low)
            g1_dot = jnp.zeros_like(f1_full).at[:3].set(f1_bar_low_dot)
            g3_dot = jnp.zeros_like(f3_full).at[:3].set(f3_bar_low_dot)

            def _solve_lambda_matrix():
                return _solve_factorized_adjoint_scan(
                    saved_lu,
                    saved_piv,
                    saved_lower,
                    saved_upper,
                    jnp.stack([g1, g3], axis=-1),
                )

            lambda_matrix = _solve_lambda_matrix()
            lambda1 = lambda_matrix[..., 0]
            lambda3 = lambda_matrix[..., 1]

            def _adjoint_rhs_dot_for_mode(k):
                diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
                    ctx,
                    k,
                    prepared.d_theta,
                    prepared.d_zeta,
                )
                diagonal_nu = _zero_first_row_if_needed(diagonal_nu, k)
                diagonal_epsi = _zero_first_row_if_needed(diagonal_epsi, k)
                diagonal_dot = nu_hat_dot * diagonal_nu + epsi_hat_dot * diagonal_epsi
                return (
                    _take_mode(g1_dot, k) - diagonal_dot.T @ _take_mode(lambda1, k),
                    _take_mode(g3_dot, k) - diagonal_dot.T @ _take_mode(lambda3, k),
                )

            def _adjoint_rhs_dot_matrix_for_mode(k):
                adjoint_rhs1_dot_k, adjoint_rhs3_dot_k = _adjoint_rhs_dot_for_mode(k)
                return jnp.stack([adjoint_rhs1_dot_k, adjoint_rhs3_dot_k], axis=-1)

            def _accumulate_base_bars(carry, k):
                nu_bar, epsi_bar = carry
                diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
                    ctx,
                    k,
                    prepared.d_theta,
                    prepared.d_zeta,
                )
                diagonal_nu = _zero_first_row_if_needed(diagonal_nu, k)
                diagonal_epsi = _zero_first_row_if_needed(diagonal_epsi, k)
                f1_k = _take_mode(f1_full, k)
                f3_k = _take_mode(f3_full, k)
                lambda1_k = _take_mode(lambda1, k)
                lambda3_k = _take_mode(lambda3, k)
                nu_bar = nu_bar - (
                    jnp.vdot(lambda1_k, diagonal_nu @ f1_k)
                    + jnp.vdot(lambda3_k, diagonal_nu @ f3_k)
                )
                epsi_bar = epsi_bar - (
                    jnp.vdot(lambda1_k, diagonal_epsi @ f1_k)
                    + jnp.vdot(lambda3_k, diagonal_epsi @ f3_k)
                )
                return (nu_bar, epsi_bar), None

            (
                nu_bar_implicit,
                epsi_bar,
            ), _ = jax.lax.scan(
                _accumulate_base_bars,
                (
                    jnp.asarray(0.0, dtype=prepared.grid.jax_dtype),
                    jnp.asarray(0.0, dtype=prepared.grid.jax_dtype),
                ),
                mode_indices,
            )

            def _parameter_source_matrix_for_mode(k):
                diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
                    ctx,
                    k,
                    prepared.d_theta,
                    prepared.d_zeta,
                )
                diagonal_nu = _zero_first_row_if_needed(diagonal_nu, k)
                diagonal_epsi = _zero_first_row_if_needed(diagonal_epsi, k)
                f1_k = _take_mode(f1_full, k)
                f3_k = _take_mode(f3_full, k)
                return jnp.stack(
                    [
                        jnp.stack([diagonal_nu @ f1_k, diagonal_nu @ f3_k], axis=-1),
                        jnp.stack([diagonal_epsi @ f1_k, diagonal_epsi @ f3_k], axis=-1),
                    ],
                    axis=-1,
                )

            def _contract_adjoint_dot_parameter_sources_scan():
                mode_count = saved_lu.shape[0]
                last_index = mode_count - 1
                mu_last = _adjoint_rhs_dot_matrix_for_mode(
                    jnp.asarray(last_index, dtype=jnp.int32)
                )

                def _backward_mu(mu_next, k):
                    propagated = lu_solve(
                        (
                            _take_mode(saved_lu, k + 1),
                            _take_mode(saved_piv, k + 1),
                        ),
                        mu_next,
                        trans=1,
                    )
                    mu_k = _adjoint_rhs_dot_matrix_for_mode(k) - _take_mode(
                        saved_lower, k + 1
                    ).T @ propagated
                    return mu_k, mu_k

                _, mu_tail = jax.lax.scan(
                    _backward_mu,
                    mu_last,
                    jnp.arange(last_index, dtype=jnp.int32),
                    reverse=True,
                )
                mu = jnp.concatenate([mu_tail, mu_last[None, ...]], axis=0)
                adjoint0 = lu_solve(
                    (_take_mode(saved_lu, 0), _take_mode(saved_piv, 0)),
                    _take_mode(mu, 0),
                    trans=1,
                )
                source0 = _parameter_source_matrix_for_mode(jnp.asarray(0, dtype=jnp.int32))
                contract0 = jnp.sum(adjoint0[..., None] * source0, axis=(0, 1))

                def _forward_adjoint(carry, k):
                    adjoint_prev, contract = carry
                    rhs = _take_mode(mu, k) - _take_mode(saved_upper, k - 1).T @ adjoint_prev
                    adjoint_k = lu_solve(
                        (_take_mode(saved_lu, k), _take_mode(saved_piv, k)),
                        rhs,
                        trans=1,
                    )
                    source_k = _parameter_source_matrix_for_mode(k)
                    contract = contract + jnp.sum(adjoint_k[..., None] * source_k, axis=(0, 1))
                    return (adjoint_k, contract), None

                (_, contracted), _ = jax.lax.scan(
                    _forward_adjoint,
                    (adjoint0, contract0),
                    jnp.arange(1, mode_count, dtype=jnp.int32),
                )
                return contracted

            nu_bar_implicit_dot, epsi_bar_dot = -_contract_adjoint_dot_parameter_sources_scan()

            if use_recompute_lowdot:
                lambda_matrix = _solve_lambda_matrix()
                lambda1 = lambda_matrix[..., 0]
                lambda3 = lambda_matrix[..., 1]

            def _source_bar_pair_for_mode(lambdas, k):
                diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
                    ctx,
                    k,
                    prepared.d_theta,
                    prepared.d_zeta,
                )
                diagonal_nu = _zero_first_row_if_needed(diagonal_nu, k)
                diagonal_epsi = _zero_first_row_if_needed(diagonal_epsi, k)
                lambda_k = _take_mode(lambdas, k)
                return jnp.stack(
                    [diagonal_nu.T @ lambda_k, diagonal_epsi.T @ lambda_k],
                    axis=-1,
                )

            f1_field_dot = _contract_factorized_source_bar_pair_scan(
                saved_lu,
                saved_piv,
                saved_lower,
                saved_upper,
                _source1_dot_for_mode,
                lambda k: _source_bar_pair_for_mode(lambda1, k),
            )
            f3_field_dot = _contract_factorized_source_bar_pair_scan(
                saved_lu,
                saved_piv,
                saved_lower,
                saved_upper,
                _source3_dot_for_mode,
                lambda k: _source_bar_pair_for_mode(lambda3, k),
            )
            nu_bar_implicit_dot = nu_bar_implicit_dot - f1_field_dot[0] - f3_field_dot[0]
            epsi_bar_dot = epsi_bar_dot - f1_field_dot[1] - f3_field_dot[1]
            return (
                nu_bar_direct + nu_bar_implicit,
                epsi_bar,
                nu_bar_direct_dot + nu_bar_implicit_dot,
                epsi_bar_dot,
            )

        def _one_case_pullback(args):
            (
                energy_index,
                nu_hat_value,
                epsi_hat_value,
                first_nu_dot,
                first_epsi_dot,
                second_nu_dot,
                second_epsi_dot,
            ) = args
            ctx = _operator_context(
                prepared.surface,
                prepared.geometry,
                prepared.grid,
                nu_hat_value,
                epsi_hat_value,
            )
            mode_indices = jnp.arange(prepared.grid.n_xi + 1, dtype=jnp.int32)
            (
                coefficients,
                f1_full,
                f3_full,
                saved_lu,
                saved_piv,
                saved_lower,
                saved_upper,
            ) = _prepared_implicit_vjp_primal(
                prepared,
                nu_hat_value,
                epsi_hat_value,
            )
            base = _base_pullback(
                ctx,
                mode_indices,
                energy_index,
                coefficients,
                f1_full,
                f3_full,
                saved_lu,
                saved_piv,
                saved_lower,
                saved_upper,
                base_transport_moments_bar,
            )

            def _source_dot_pair_for_direction(k, nu_hat_dot, epsi_hat_dot):
                diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
                    ctx,
                    k,
                    prepared.d_theta,
                    prepared.d_zeta,
                )
                diagonal_nu = _zero_first_row_if_needed(diagonal_nu, k)
                diagonal_epsi = _zero_first_row_if_needed(diagonal_epsi, k)
                diagonal_dot = nu_hat_dot * diagonal_nu + epsi_hat_dot * diagonal_epsi
                return (
                    -(diagonal_dot @ _take_mode(f1_full, k)),
                    -(diagonal_dot @ _take_mode(f3_full, k)),
                )

            def _packed_source_dot_matrix_for_mode(k):
                first_source1, first_source3 = _source_dot_pair_for_direction(
                    k,
                    first_nu_dot,
                    first_epsi_dot,
                )
                second_source1, second_source3 = _source_dot_pair_for_direction(
                    k,
                    second_nu_dot,
                    second_epsi_dot,
                )
                return jnp.stack(
                    [first_source1, first_source3, second_source1, second_source3],
                    axis=-1,
                )

            packed_f_dot_low_matrix = _solve_factorized_low_modes_scan(
                saved_lu,
                saved_piv,
                saved_lower,
                saved_upper,
                _packed_source_dot_matrix_for_mode,
            )
            first_f1_dot_low = packed_f_dot_low_matrix[..., 0]
            first_f3_dot_low = packed_f_dot_low_matrix[..., 1]
            second_f1_dot_low = packed_f_dot_low_matrix[..., 2]
            second_f3_dot_low = packed_f_dot_low_matrix[..., 3]
            first = _one_direction_pullback(
                ctx,
                mode_indices,
                energy_index,
                coefficients,
                f1_full,
                f3_full,
                saved_lu,
                saved_piv,
                saved_lower,
                saved_upper,
                first_nu_dot,
                first_epsi_dot,
                first_transport_moments_bar,
                first_f1_dot_low,
                first_f3_dot_low,
            )
            second = _one_direction_pullback(
                ctx,
                mode_indices,
                energy_index,
                coefficients,
                f1_full,
                f3_full,
                saved_lu,
                saved_piv,
                saved_lower,
                saved_upper,
                second_nu_dot,
                second_epsi_dot,
                second_transport_moments_bar,
                second_f1_dot_low,
                second_f3_dot_low,
            )
            return (*base, *first, *second)

        if use_sequential_lowdot:
            outputs0 = tuple(jnp.zeros_like(reference_nu_hat) for _ in range(10))

            def _body(i, outputs):
                one_output = _one_case_pullback(
                    (
                        energy_indices[i],
                        reference_nu_hat[i],
                        reference_epsi_hat[i],
                        first_nu_hat_tangent[i],
                        first_epsi_hat_tangent[i],
                        second_nu_hat_tangent[i],
                        second_epsi_hat_tangent[i],
                    )
                )
                return tuple(
                    output.at[i].set(value)
                    for output, value in zip(outputs, one_output, strict=True)
                )

            return jax.lax.fori_loop(
                0,
                reference_nu_hat.shape[0],
                _body,
                outputs0,
            )

        return jax.lax.map(
            _one_case_pullback,
            (
                energy_indices,
                reference_nu_hat,
                reference_epsi_hat,
                first_nu_hat_tangent,
                first_epsi_hat_tangent,
                second_nu_hat_tangent,
                second_epsi_hat_tangent,
            ),
        )

    def _pullback_log_nu_star_from_nu_hat(
        self,
        reference_nu_hat,
        reference_log_nu_star_bar,
    ):
        weights = jnp.asarray(self.energy_grid.xWeights, dtype=jnp.float64)
        weights = weights / jnp.maximum(jnp.sum(weights), 1.0e-30)
        nu_hat = jnp.asarray(reference_nu_hat, dtype=jnp.float64)
        safe_nu_hat = jnp.maximum(nu_hat, 1.0e-30)
        # Match the forward primitive exactly: below the floor, the active branch
        # is constant and contributes zero tangent/cotangent.
        active_mask = jnp.asarray(nu_hat >= 1.0e-30, dtype=jnp.float64)
        return jnp.asarray(
            active_mask * weights * reference_log_nu_star_bar / safe_nu_hat,
            dtype=reference_nu_hat.dtype,
        )

    def _pullback_interpolated_moment_reduced_local_outputs(
        self,
        prepared,
        *,
        drds_value,
        reference_nu_hat,
        reference_epsi_hat,
        vth_a,
        field_bars,
    ):
        zero_nu_hat = jnp.zeros_like(reference_nu_hat)
        zero_epsi_hat = jnp.zeros_like(reference_epsi_hat)
        zero_vth_a = jnp.zeros_like(vth_a)
        (
            reference_log_nu_star_bar,
            reference_transport_moments_bar,
            dtransport_moments_d_er_bar,
            dtransport_moments_d_log_nu_star_bar,
        ) = _interpolated_response_field_bar_tuple(field_bars)

        log_nu_star_nu_hat_bar = self._pullback_log_nu_star_from_nu_hat(
            reference_nu_hat,
            reference_log_nu_star_bar,
        )
        use_fused_lowdot_derivative_pullback = (
            self._normalize_derivative_field_pullback_mode(self.derivative_field_pullback_mode)
            == "compact_vjp"
            and self._normalize_derivative_pullback_algebra(self.derivative_pullback_algebra)
            in {
                "scalar_contract_lowdot",
                "scalar_contract_lowdot_sequential",
                "scalar_contract_lowdot_ntx",
                "scalar_contract_lowdot_recompute",
            }
        )
        if use_fused_lowdot_derivative_pullback:
            epsi_hat_tangent = jnp.asarray(1.0e3, dtype=reference_epsi_hat.dtype) / (
                self.energy_grid.v_norm * vth_a
            )
            (
                transport_moments_nu_hat_bar,
                transport_moments_epsi_hat_bar,
                _d_er_base_nu_hat_bar,
                d_er_base_epsi_hat_bar,
                dtransport_d_er_nu_hat_bar,
                dtransport_d_er_epsi_hat_bar,
                d_log_base_nu_hat_bar,
                _d_log_base_epsi_hat_bar,
                d_log_nu_hat_bar,
                dtransport_d_log_nu_star_epsi_hat_bar,
            ) = self._scalar_contract_lowdot_two_derivative_pullbacks_from_scan_primitives(
                prepared,
                drds_value=drds_value,
                reference_nu_hat=reference_nu_hat,
                reference_epsi_hat=reference_epsi_hat,
                base_transport_moments_bar=reference_transport_moments_bar,
                first_nu_hat_tangent=zero_nu_hat,
                first_epsi_hat_tangent=epsi_hat_tangent,
                first_transport_moments_bar=dtransport_moments_d_er_bar,
                second_nu_hat_tangent=reference_nu_hat,
                second_epsi_hat_tangent=zero_epsi_hat,
                second_transport_moments_bar=dtransport_moments_d_log_nu_star_bar,
            )
            dtransport_d_er_vth_a_bar = jnp.sum(
                d_er_base_epsi_hat_bar * (-epsi_hat_tangent / vth_a),
                axis=0,
            )
            dtransport_d_log_nu_star_nu_hat_bar = d_log_nu_hat_bar + d_log_base_nu_hat_bar
        else:
            (
                dtransport_d_er_nu_hat_bar,
                dtransport_d_er_epsi_hat_bar,
                dtransport_d_er_vth_a_bar,
            ) = self._pullback_dtransport_moments_d_er_from_scan_primitives(
                prepared,
                drds_value=drds_value,
                reference_nu_hat=reference_nu_hat,
                reference_epsi_hat=reference_epsi_hat,
                vth_a=vth_a,
                dtransport_moments_d_er_bar=dtransport_moments_d_er_bar,
            )
            (
                dtransport_d_log_nu_star_nu_hat_bar,
                dtransport_d_log_nu_star_epsi_hat_bar,
            ) = self._pullback_dtransport_moments_d_log_nu_star_from_scan_primitives(
                prepared,
                drds_value=drds_value,
                reference_nu_hat=reference_nu_hat,
                reference_epsi_hat=reference_epsi_hat,
                dtransport_moments_d_log_nu_star_bar=dtransport_moments_d_log_nu_star_bar,
            )
            transport_moments_nu_hat_bar, transport_moments_epsi_hat_bar = self._pullback_transport_moments_from_scan_primitives(
                prepared,
                drds_value=drds_value,
                reference_nu_hat=reference_nu_hat,
                reference_epsi_hat=reference_epsi_hat,
                reference_transport_moments_bar=reference_transport_moments_bar,
            )

        nu_hat_bar = (
            log_nu_star_nu_hat_bar
            + transport_moments_nu_hat_bar
            + dtransport_d_er_nu_hat_bar
            + dtransport_d_log_nu_star_nu_hat_bar
        )
        epsi_hat_bar = (
            transport_moments_epsi_hat_bar
            + dtransport_d_er_epsi_hat_bar
            + dtransport_d_log_nu_star_epsi_hat_bar
        )
        vth_a_bar = dtransport_d_er_vth_a_bar

        if _ntx_local_pullback_finite_debug_enabled():
            def _finite_debug_callback(
                log_nu_hat_bar,
                tm_nu_hat_bar,
                tm_epsi_hat_bar,
                der_nu_hat_bar,
                der_epsi_hat_bar,
                der_vth_bar,
                dlog_nu_hat_bar,
                dlog_epsi_hat_bar,
                total_nu_hat_bar,
                total_epsi_hat_bar,
                total_vth_bar,
            ):
                entries = [
                    ("log_nu_star_nu_hat_bar", log_nu_hat_bar),
                    ("transport_moments_nu_hat_bar", tm_nu_hat_bar),
                    ("transport_moments_epsi_hat_bar", tm_epsi_hat_bar),
                    ("dtransport_d_er_nu_hat_bar", der_nu_hat_bar),
                    ("dtransport_d_er_epsi_hat_bar", der_epsi_hat_bar),
                    ("dtransport_d_er_vth_a_bar", der_vth_bar),
                    ("dtransport_d_log_nu_star_nu_hat_bar", dlog_nu_hat_bar),
                    ("dtransport_d_log_nu_star_epsi_hat_bar", dlog_epsi_hat_bar),
                    ("nu_hat_bar_total", total_nu_hat_bar),
                    ("epsi_hat_bar_total", total_epsi_hat_bar),
                    ("vth_a_bar_total", total_vth_bar),
                ]
                for name, value in entries:
                    arr = np.asarray(value)
                    if not np.issubdtype(arr.dtype, np.inexact):
                        continue
                    if not np.all(np.isfinite(arr)):
                        print(
                            "[autodiff-gate] ntx-local-pullback-nonfinite "
                            f"name={name} shape={arr.shape}"
                        )
                        break

            jax.debug.callback(
                _finite_debug_callback,
                log_nu_star_nu_hat_bar,
                transport_moments_nu_hat_bar,
                transport_moments_epsi_hat_bar,
                dtransport_d_er_nu_hat_bar,
                dtransport_d_er_epsi_hat_bar,
                dtransport_d_er_vth_a_bar,
                dtransport_d_log_nu_star_nu_hat_bar,
                dtransport_d_log_nu_star_epsi_hat_bar,
                nu_hat_bar,
                epsi_hat_bar,
                vth_a_bar,
                ordered=True,
            )

        return (
            jnp.asarray(nu_hat_bar, dtype=reference_nu_hat.dtype),
            jnp.asarray(epsi_hat_bar, dtype=reference_epsi_hat.dtype),
            jnp.asarray(vth_a_bar, dtype=vth_a.dtype),
        )

    def _pullback_local_scan_inputs_from_primitives(
        self,
        *,
        drds_value,
        species_index: int,
        er_value,
        temperature_local,
        density_local,
        vthermal_local,
        collisionality_kind,
        reference_nu_hat_bar,
        reference_epsi_hat_bar,
        vth_a_bar,
    ):
        def _forward_linearized_local_scan_inputs(
            der_value,
            dtemperature_local,
            ddensity_local,
        ):
            _, dvthermal = jax.jvp(
                lambda temperature_local_value: get_v_thermal(
                    self.species.mass,
                    temperature_local_value,
                ),
                (temperature_local,),
                (dtemperature_local,),
            )
            _, local_scan_tangent = jax.jvp(
                lambda er_local_value, temperature_local_value, density_local_value, vthermal_local_value: self._local_scan_inputs(
                    drds_value=drds_value,
                    species_index=species_index,
                    er_value=er_local_value,
                    temperature_local=temperature_local_value,
                    density_local=density_local_value,
                    vthermal_local=vthermal_local_value,
                    collisionality_kind=collisionality_kind,
                ),
                (er_value, temperature_local, density_local, vthermal_local),
                (der_value, dtemperature_local, ddensity_local, dvthermal),
            )
            return local_scan_tangent

        return jax.linear_transpose(
            _forward_linearized_local_scan_inputs,
            jnp.zeros_like(er_value),
            jnp.zeros_like(temperature_local),
            jnp.zeros_like(density_local),
        )((reference_nu_hat_bar, reference_epsi_hat_bar, vth_a_bar))

    def _interpolated_response_field_bars(
        self,
        center_response_bar: NTXInterpolatedMomentResponse,
    ) -> _NTXInterpolatedMomentResponseFieldBars:
        """Extract the rebuild-only interpolated field-bar contract."""
        return _NTXInterpolatedMomentResponseFieldBars(
            reference_log_nu_star=jnp.swapaxes(center_response_bar.reference_log_nu_star, 0, 1),
            reference_transport_moments=jnp.swapaxes(center_response_bar.reference_transport_moments, 0, 1),
            dtransport_moments_d_er=jnp.swapaxes(center_response_bar.dtransport_moments_d_er, 0, 1),
            dtransport_moments_d_log_nu_star=jnp.swapaxes(center_response_bar.dtransport_moments_d_log_nu_star, 0, 1),
        )

    def _pullback_interpolated_anchor_response_fields(
        self,
        *,
        anchor_indices,
        anchor_rho,
        target_rho,
        field_bars: _NTXInterpolatedMomentResponseFieldBars,
    ) -> _NTXInterpolatedMomentResponseFieldBars:
        """Transpose interpolation using only the reduced rebuild field bars."""
        response_field_bars = _interpolated_response_field_bar_tuple(field_bars)
        n_anchor = int(anchor_indices.shape[0])
        if n_anchor >= 4:
            non_axis_templates = tuple(
                jnp.zeros(
                    (n_anchor - 1,) + field_bar.shape[1:],
                    dtype=field_bar.dtype,
                )
                for field_bar in response_field_bars
            )

            def _forward_interpolated_fields_from_non_axis_anchor_fields(*non_axis_anchor_response_fields):
                anchor_response_fields = tuple(
                    jnp.concatenate([values[:1], values], axis=0)
                    for values in non_axis_anchor_response_fields
                )
                anchor_response_fields = self._regularize_axis_radius0(anchor_response_fields, anchor_rho)
                return tuple(
                    self._interpolate_anchor_values(anchor_indices, anchor_response_fields[field_index], target_rho)
                    for field_index in range(len(_INTERPOLATED_RESPONSE_FIELD_NAMES))
                )

            def _regularized_case(_):
                non_axis_anchor_response_bar = jax.linear_transpose(
                    _forward_interpolated_fields_from_non_axis_anchor_fields,
                    *non_axis_templates,
                )(response_field_bars)
                raw_anchor_response_bar = tuple(
                    jnp.concatenate(
                        [
                            jnp.zeros_like(values[:1]),
                            values,
                        ],
                        axis=0,
                    )
                    for values in non_axis_anchor_response_bar
                )
                return raw_anchor_response_bar

            anchor_response_templates = tuple(
                jnp.zeros(
                    (n_anchor,) + field_bar.shape[1:],
                    dtype=field_bar.dtype,
                )
                for field_bar in response_field_bars
            )

            def _forward_interpolated_fields_from_anchor_fields(*raw_anchor_response_fields):
                return tuple(
                    self._interpolate_anchor_values(anchor_indices, raw_anchor_response_fields[field_index], target_rho)
                    for field_index in range(len(_INTERPOLATED_RESPONSE_FIELD_NAMES))
                )

            def _direct_case(_):
                return jax.linear_transpose(
                    _forward_interpolated_fields_from_anchor_fields,
                    *anchor_response_templates,
                )(response_field_bars)

            raw_anchor_response_bar = jax.lax.cond(
                jnp.isclose(anchor_rho[0], 0.0),
                _regularized_case,
                _direct_case,
                operand=None,
            )
            return _NTXInterpolatedMomentResponseFieldBars(*raw_anchor_response_bar)

        anchor_response_templates = tuple(
            jnp.zeros(
                (n_anchor,) + field_bar.shape[1:],
                dtype=field_bar.dtype,
            )
            for field_bar in response_field_bars
        )

        def _forward_interpolated_fields_from_anchor_fields(*raw_anchor_response_fields):
            return tuple(
                self._interpolate_anchor_values(anchor_indices, raw_anchor_response_fields[field_index], target_rho)
                for field_index in range(len(_INTERPOLATED_RESPONSE_FIELD_NAMES))
            )

        raw_anchor_response_bar = jax.linear_transpose(
            _forward_interpolated_fields_from_anchor_fields,
            *anchor_response_templates,
        )(response_field_bars)
        return _NTXInterpolatedMomentResponseFieldBars(*raw_anchor_response_bar)

    def _pullback_interpolated_moment_response_local_fields(
        self,
        prepared,
        *,
        drds_value,
        er_value,
        temperature_local,
        density_local,
        collisionality_kind,
        field_bars,
        scan_species: bool = False,
    ):
        vthermal_local = get_v_thermal(self.species.mass, temperature_local)
        species_indices = jnp.arange(int(self.species.number_species), dtype=jnp.int32)

        def _per_species_pullback(species_index, species_field_bars):
            reference_nu_hat, reference_epsi_hat, vth_a = self._interpolated_moment_local_scan_primitives(
                drds_value=drds_value,
                species_index=species_index,
                er_value=er_value,
                temperature_local=temperature_local,
                density_local=density_local,
                vthermal_local=vthermal_local,
                collisionality_kind=collisionality_kind,
            )

            (
                reference_nu_hat_bar,
                reference_epsi_hat_bar,
                vth_a_bar,
            ) = self._pullback_interpolated_moment_reduced_local_outputs(
                prepared,
                drds_value=drds_value,
                reference_nu_hat=reference_nu_hat,
                reference_epsi_hat=reference_epsi_hat,
                vth_a=vth_a,
                field_bars=species_field_bars,
            )

            return self._pullback_local_scan_inputs_from_primitives(
                drds_value=drds_value,
                species_index=species_index,
                er_value=er_value,
                temperature_local=temperature_local,
                density_local=density_local,
                vthermal_local=vthermal_local,
                collisionality_kind=collisionality_kind,
                reference_nu_hat_bar=reference_nu_hat_bar,
                reference_epsi_hat_bar=reference_epsi_hat_bar,
                vth_a_bar=vth_a_bar,
            )

        if scan_species:
            field_bar_tuple = _interpolated_response_field_bar_tuple(field_bars)

            def _accumulate_species(carry, species_index):
                er_bar, temperature_bar, density_bar = carry
                species_field_bars = tuple(
                    jax.lax.dynamic_index_in_dim(
                        field_bar,
                        species_index,
                        axis=0,
                        keepdims=False,
                    )
                    for field_bar in field_bar_tuple
                )
                (
                    er_species_bar,
                    temperature_species_bar,
                    density_species_bar,
                ) = _per_species_pullback(species_index, species_field_bars)
                return (
                    er_bar + er_species_bar,
                    temperature_bar + temperature_species_bar,
                    density_bar + density_species_bar,
                ), None

            (
                er_bar,
                temperature_bar,
                density_bar,
            ), _ = jax.lax.scan(
                _accumulate_species,
                (
                    jnp.zeros_like(er_value),
                    jnp.zeros_like(temperature_local),
                    jnp.zeros_like(density_local),
                ),
                species_indices,
            )
            return er_bar, temperature_bar, density_bar

        er_species_bar, temperature_species_bar, density_species_bar = jax.vmap(
            _per_species_pullback,
            in_axes=(0, 0),
        )(species_indices, field_bars)
        return (
            jnp.sum(er_species_bar, axis=0),
            jnp.sum(temperature_species_bar, axis=0),
            jnp.sum(density_species_bar, axis=0),
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

        lij_by_radius = self._map_radius_axis_regularized_at_axis0(
            _per_radius,
            radius_indices,
            self.geometry.r_grid,
        )
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
        lij_by_radius = self._regularize_axis_radius0(lij_by_radius, self.geometry.r_grid_half)
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
        gamma, q, upar = self._regularize_center_fluxes_axis0(gamma, q, upar)
        return {"Gamma": gamma, "Q": q, "Upar": upar}

    def build_lagged_response(self, state, **kwargs):
        del kwargs
        if lagged_timing_enabled():
            jax.debug.callback(lambda: lagged_timing_start("ntx.build_lagged_response"), ordered=True)
        density = safe_density(state.density)
        temperature = state.temperature
        support = self._static_support()
        collisionality_kind = _collisionality_kind(self.collisionality_model)
        v_thermal = get_v_thermal(self.species.mass, temperature)
        species_indices = jnp.arange(int(self.species.number_species), dtype=jnp.int32)
        n_radius = int(state.Er.shape[0])
        radius_indices = jnp.arange(n_radius, dtype=jnp.int32)
        anchor_indices = self._response_anchor_indices(n_radius)

        if int(anchor_indices.shape[0]) < n_radius:
            target_rho = jnp.asarray(support.center_channels.rho, dtype=jnp.float64)

            def _per_anchor(radius_index):
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
                    lambda species_index: self._build_interpolated_moment_response_local(
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

            anchor_response = self._map_radius_axis_regularized_at_axis0(
                _per_anchor,
                anchor_indices,
                jnp.asarray(self.geometry.r_grid, dtype=jnp.float64)[anchor_indices],
            )
            anchor_reference_log_nu_star = anchor_response[0]
            anchor_reference_transport_moments = anchor_response[1]
            anchor_dtransport_moments_d_er = anchor_response[2]
            anchor_dtransport_moments_d_log_nu_star = anchor_response[3]
            reference_log_nu_star = self._interpolate_anchor_values(
                anchor_indices,
                anchor_reference_log_nu_star,
                target_rho,
            )
            reference_transport_moments = self._interpolate_anchor_values(
                anchor_indices,
                anchor_reference_transport_moments,
                target_rho,
            )
            dtransport_moments_d_er = self._interpolate_anchor_values(
                anchor_indices,
                anchor_dtransport_moments_d_er,
                target_rho,
            )
            dtransport_moments_d_log_nu_star = self._interpolate_anchor_values(
                anchor_indices,
                anchor_dtransport_moments_d_log_nu_star,
                target_rho,
            )
            if lagged_timing_enabled():
                jax.debug.callback(lambda: lagged_timing_end("ntx.build_lagged_response"), ordered=True)
            return NTXExactLijLaggedResponse(
                center_response=NTXInterpolatedMomentResponse(
                    reference_er=state.Er,
                    reference_log_nu_star=jnp.swapaxes(reference_log_nu_star, 0, 1),
                    reference_transport_moments=jnp.swapaxes(reference_transport_moments, 0, 1),
                    dtransport_moments_d_er=jnp.swapaxes(dtransport_moments_d_er, 0, 1),
                    dtransport_moments_d_log_nu_star=jnp.swapaxes(dtransport_moments_d_log_nu_star, 0, 1),
                )
            )

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

        response_by_radius = self._map_radius_axis_regularized_at_axis0(
            _per_radius,
            radius_indices,
            self.geometry.r_grid,
        )
        if lagged_timing_enabled():
            jax.debug.callback(lambda: lagged_timing_end("ntx.build_lagged_response"), ordered=True)
        return NTXExactLijLaggedResponse(center_response=response_by_radius)

    def pullback_build_lagged_response(self, state, lagged_response_bar, **kwargs):
        reverse_stage_cotangent_mode = str(kwargs.pop("reverse_stage_cotangent_mode", "full")).strip().lower()
        del kwargs
        center_response_bar = None if lagged_response_bar is None else lagged_response_bar.center_response
        if center_response_bar is None:
            return jax.tree_util.tree_map(jnp.zeros_like, state)

        def _transpose_primitives_from_builder(builder_fn, response_bar_value):
            density0 = state.density
            pressure0 = state.pressure
            er0 = state.Er

            def _forward_linearized_builder(ddensity, dpressure, der):
                _, dresponse = jax.jvp(
                    builder_fn,
                    (density0, pressure0, er0),
                    (ddensity, dpressure, der),
                )
                return dresponse

            zeros = (
                jnp.zeros_like(density0),
                jnp.zeros_like(pressure0),
                jnp.zeros_like(er0),
            )
            density_bar, pressure_bar, er_bar = jax.linear_transpose(
                _forward_linearized_builder,
                *zeros,
            )(response_bar_value)
            return density_bar, pressure_bar, er_bar

        def _build_coefficient_center_response_from_primitives(density_value, pressure_value, er_value):
            density_safe = safe_density(density_value)
            temperature_value = pressure_value / density_safe
            support = self._static_support()
            collisionality_kind = _collisionality_kind(self.collisionality_model)
            vthermal_value = get_v_thermal(self.species.mass, temperature_value)
            species_indices = jnp.arange(int(self.species.number_species), dtype=jnp.int32)
            radius_indices = jnp.arange(er_value.shape[0], dtype=jnp.int32)

            def _per_radius(radius_index):
                prepared = jax.tree_util.tree_map(
                    lambda arr: jax.lax.dynamic_index_in_dim(arr, radius_index, axis=0, keepdims=False),
                    support.center_prepared,
                )
                drds_value = jax.lax.dynamic_index_in_dim(
                    support.center_channels.drds,
                    radius_index,
                    axis=0,
                    keepdims=False,
                )
                er_local = jax.lax.dynamic_index_in_dim(er_value, radius_index, axis=0, keepdims=False)
                temperature_local = jax.lax.dynamic_index_in_dim(
                    temperature_value,
                    radius_index,
                    axis=1,
                    keepdims=False,
                )
                density_local = jax.lax.dynamic_index_in_dim(
                    density_safe,
                    radius_index,
                    axis=1,
                    keepdims=False,
                )
                vthermal_local = jax.lax.dynamic_index_in_dim(
                    vthermal_value,
                    radius_index,
                    axis=1,
                    keepdims=False,
                )
                return jax.vmap(
                    lambda species_index: self._build_coefficient_response_local(
                        prepared,
                        drds_value=drds_value,
                        species_index=species_index,
                        er_value=er_local,
                        temperature_local=temperature_local,
                        density_local=density_local,
                        vthermal_local=vthermal_local,
                        collisionality_kind=collisionality_kind,
                    )
                )(species_indices)

            return self._map_radius_axis_regularized_at_axis0(
                _per_radius,
                radius_indices,
                self.geometry.r_grid,
            )

        def _build_interpolated_center_response_from_primitives(density_value, pressure_value, er_value):
            density_safe = safe_density(density_value)
            temperature_value = pressure_value / density_safe
            support = self._static_support()
            collisionality_kind = _collisionality_kind(self.collisionality_model)
            vthermal_value = get_v_thermal(self.species.mass, temperature_value)
            species_indices = jnp.arange(int(self.species.number_species), dtype=jnp.int32)
            n_radius = int(er_value.shape[0])
            radius_indices = jnp.arange(n_radius, dtype=jnp.int32)
            anchor_indices = self._response_anchor_indices(n_radius)
            target_rho = jnp.asarray(support.center_channels.rho, dtype=jnp.float64)

            def _per_anchor(radius_index):
                prepared = jax.tree_util.tree_map(
                    lambda arr: jax.lax.dynamic_index_in_dim(arr, radius_index, axis=0, keepdims=False),
                    support.center_prepared,
                )
                drds_value = jax.lax.dynamic_index_in_dim(
                    support.center_channels.drds,
                    radius_index,
                    axis=0,
                    keepdims=False,
                )
                er_local = jax.lax.dynamic_index_in_dim(er_value, radius_index, axis=0, keepdims=False)
                temperature_local = jax.lax.dynamic_index_in_dim(
                    temperature_value,
                    radius_index,
                    axis=1,
                    keepdims=False,
                )
                density_local = jax.lax.dynamic_index_in_dim(
                    density_safe,
                    radius_index,
                    axis=1,
                    keepdims=False,
                )
                vthermal_local = jax.lax.dynamic_index_in_dim(
                    vthermal_value,
                    radius_index,
                    axis=1,
                    keepdims=False,
                )
                return jax.vmap(
                    lambda species_index: self._build_interpolated_moment_response_local(
                        prepared,
                        drds_value=drds_value,
                        species_index=species_index,
                        er_value=er_local,
                        temperature_local=temperature_local,
                        density_local=density_local,
                        vthermal_local=vthermal_local,
                        collisionality_kind=collisionality_kind,
                    )
                )(species_indices)

            anchor_response = self._map_radius_axis_regularized_at_axis0(
                _per_anchor,
                anchor_indices,
                jnp.asarray(self.geometry.r_grid, dtype=jnp.float64)[anchor_indices],
            )
            anchor_reference_log_nu_star = anchor_response[0]
            anchor_reference_transport_moments = anchor_response[1]
            anchor_dtransport_moments_d_er = anchor_response[2]
            anchor_dtransport_moments_d_log_nu_star = anchor_response[3]
            reference_log_nu_star = self._interpolate_anchor_values(
                anchor_indices,
                anchor_reference_log_nu_star,
                target_rho,
            )
            reference_transport_moments = self._interpolate_anchor_values(
                anchor_indices,
                anchor_reference_transport_moments,
                target_rho,
            )
            dtransport_moments_d_er = self._interpolate_anchor_values(
                anchor_indices,
                anchor_dtransport_moments_d_er,
                target_rho,
            )
            dtransport_moments_d_log_nu_star = self._interpolate_anchor_values(
                anchor_indices,
                anchor_dtransport_moments_d_log_nu_star,
                target_rho,
            )
            return NTXInterpolatedMomentResponse(
                reference_er=er_value,
                reference_log_nu_star=jnp.swapaxes(reference_log_nu_star, 0, 1),
                reference_transport_moments=jnp.swapaxes(reference_transport_moments, 0, 1),
                dtransport_moments_d_er=jnp.swapaxes(dtransport_moments_d_er, 0, 1),
                dtransport_moments_d_log_nu_star=jnp.swapaxes(dtransport_moments_d_log_nu_star, 0, 1),
            )

        if isinstance(center_response_bar, NTXInterpolatedMomentResponse):
            support = self._static_support()
            density0 = state.density
            pressure0 = state.pressure
            er0 = state.Er
            n_radius = int(er0.shape[0])
            anchor_indices = self._response_anchor_indices(n_radius)
            target_rho = jnp.asarray(support.center_channels.rho, dtype=jnp.float64)
            anchor_rho = jnp.asarray(self.geometry.r_grid, dtype=jnp.float64)[anchor_indices]
            n_anchor = int(anchor_indices.shape[0])
            collisionality_kind = _collisionality_kind(self.collisionality_model)
            response_field_bars = self._interpolated_response_field_bars(center_response_bar)
            er_bar_direct = jnp.asarray(center_response_bar.reference_er)
            raw_anchor_response_bar = self._pullback_interpolated_anchor_response_fields(
                anchor_indices=anchor_indices,
                anchor_rho=anchor_rho,
                target_rho=target_rho,
                field_bars=response_field_bars,
            )
            raw_anchor_response_fields = _interpolated_response_field_bar_tuple(raw_anchor_response_bar)

            density_bar = jnp.zeros_like(density0)
            pressure_bar = jnp.zeros_like(pressure0)
            er_bar = jnp.asarray(er_bar_direct)

            if reverse_stage_cotangent_mode in {
                "zero_rebuild_anchor_fields",
                "zero_rebuild_interpolated_fields",
                "rebuild_anchor_fields_zero",
            }:
                return dataclasses.replace(
                    state,
                    density=density_bar,
                    pressure=pressure_bar,
                    Er=er_bar,
                )
            zero_local_moment_pullback = reverse_stage_cotangent_mode in {
                "zero_rebuild_local_moment_pullback",
                "zero_rebuild_local_moments",
                "rebuild_local_moment_pullback_zero",
            }
            scan_local_moment_pullback = reverse_stage_cotangent_mode in {
                "scan_rebuild_local_moment_pullback",
                "scan_rebuild_local_moments",
                "rebuild_local_moment_pullback_scan",
                "scan_rebuild_anchor_pullback",
                "scan_rebuild_anchor_local_moment_pullback",
            }
            scan_anchor_pullback = reverse_stage_cotangent_mode in {
                "scan_rebuild_anchor_pullback",
                "scan_rebuild_anchor_local_moment_pullback",
                "rebuild_anchor_pullback_scan",
            }
            anchor_positions = jnp.arange(n_anchor, dtype=jnp.int32)

            def _pullback_one_anchor(anchor_pos):
                radius_index = jax.lax.dynamic_index_in_dim(
                    anchor_indices,
                    anchor_pos,
                    axis=0,
                    keepdims=False,
                )
                is_axis_anchor = jnp.logical_and(
                    jnp.asarray(n_anchor >= 4),
                    jnp.logical_and(
                        jnp.asarray(anchor_pos == 0, dtype=jnp.bool_),
                        jnp.isclose(jax.lax.dynamic_index_in_dim(anchor_rho, 0, axis=0, keepdims=False), 0.0),
                    ),
                )

                def _axis_anchor_zero_pullback(_):
                    density_local0 = jax.lax.dynamic_index_in_dim(
                        density0,
                        radius_index,
                        axis=1,
                        keepdims=False,
                    )
                    pressure_local0 = jax.lax.dynamic_index_in_dim(
                        pressure0,
                        radius_index,
                        axis=1,
                        keepdims=False,
                    )
                    er_local0 = jax.lax.dynamic_index_in_dim(
                        er0,
                        radius_index,
                        axis=0,
                        keepdims=False,
                    )
                    return (
                        radius_index,
                        jnp.zeros_like(density_local0),
                        jnp.zeros_like(pressure_local0),
                        jnp.zeros_like(er_local0),
                    )

                def _non_axis_anchor_pullback(_):
                    local_field_bars = tuple(
                        jax.lax.dynamic_index_in_dim(
                            field_bar,
                            anchor_pos,
                            axis=0,
                            keepdims=False,
                        )
                        for field_bar in raw_anchor_response_fields
                    )
                    density_local0 = jax.lax.dynamic_index_in_dim(
                        density0,
                        radius_index,
                        axis=1,
                        keepdims=False,
                    )
                    pressure_local0 = jax.lax.dynamic_index_in_dim(
                        pressure0,
                        radius_index,
                        axis=1,
                        keepdims=False,
                    )
                    density_safe_local = safe_density(density_local0)
                    temperature_local0 = pressure_local0 / density_safe_local
                    er_local0 = jax.lax.dynamic_index_in_dim(
                        er0,
                        radius_index,
                        axis=0,
                        keepdims=False,
                    )

                    prepared_local = jax.tree_util.tree_map(
                        lambda arr: jax.lax.dynamic_index_in_dim(arr, radius_index, axis=0, keepdims=False),
                        support.center_prepared,
                    )
                    drds_value_local = jax.lax.dynamic_index_in_dim(
                        support.center_channels.drds,
                        radius_index,
                        axis=0,
                        keepdims=False,
                    )

                    if zero_local_moment_pullback:
                        (
                            er_local_bar,
                            temperature_local_bar,
                            density_safe_local_bar,
                        ) = (
                            jnp.zeros_like(er_local0),
                            jnp.zeros_like(temperature_local0),
                            jnp.zeros_like(density_safe_local),
                        )
                    else:
                        (
                            er_local_bar,
                            temperature_local_bar,
                            density_safe_local_bar,
                        ) = self._pullback_interpolated_moment_response_local_fields(
                            prepared_local,
                            drds_value=drds_value_local,
                            er_value=er_local0,
                            temperature_local=temperature_local0,
                            density_local=density_safe_local,
                            collisionality_kind=collisionality_kind,
                            field_bars=local_field_bars,
                            scan_species=scan_local_moment_pullback,
                        )

                    pressure_local_bar = temperature_local_bar / density_safe_local
                    density_safe_total_bar = density_safe_local_bar - (
                        temperature_local_bar * pressure_local0 / (density_safe_local * density_safe_local)
                    )
                    density_floor_local = _broadcast_species_floor(
                        density_local0,
                        DEFAULT_TRANSPORT_DENSITY_FLOOR,
                    )
                    density_local_bar = jnp.where(
                        density_local0 > density_floor_local,
                        density_safe_total_bar,
                        jnp.zeros_like(density_safe_total_bar),
                    )

                    return (
                        radius_index,
                        density_local_bar,
                        pressure_local_bar,
                        er_local_bar,
                    )

                return jax.lax.cond(
                    is_axis_anchor,
                    _axis_anchor_zero_pullback,
                    _non_axis_anchor_pullback,
                    operand=None,
                )

            if scan_anchor_pullback:
                def _accumulate_anchor(carry, anchor_pos):
                    density_carry, pressure_carry, er_carry = carry
                    (
                        radius_index,
                        density_local_bar,
                        pressure_local_bar,
                        er_local_bar,
                    ) = _pullback_one_anchor(anchor_pos)
                    return (
                        density_carry.at[:, radius_index].add(density_local_bar),
                        pressure_carry.at[:, radius_index].add(pressure_local_bar),
                        er_carry.at[radius_index].add(er_local_bar),
                    ), None

                (
                    density_bar,
                    pressure_bar,
                    er_bar,
                ), _ = jax.lax.scan(
                    _accumulate_anchor,
                    (density_bar, pressure_bar, er_bar),
                    anchor_positions,
                )
                return dataclasses.replace(
                    state,
                    density=density_bar,
                    pressure=pressure_bar,
                    Er=er_bar,
                )

            (
                anchor_radius_indices,
                density_anchor_bars,
                pressure_anchor_bars,
                er_anchor_bars,
            ) = jax.vmap(_pullback_one_anchor)(anchor_positions)
            density_bar = density_bar.at[:, anchor_radius_indices].add(
                jnp.swapaxes(density_anchor_bars, 0, 1)
            )
            pressure_bar = pressure_bar.at[:, anchor_radius_indices].add(
                jnp.swapaxes(pressure_anchor_bars, 0, 1)
            )
            er_bar = er_bar.at[anchor_radius_indices].add(er_anchor_bars)
            return dataclasses.replace(
                state,
                density=density_bar,
                pressure=pressure_bar,
                Er=er_bar,
            )

        if isinstance(center_response_bar, NTXPreparedCoefficientResponse):
            density_bar, pressure_bar, er_bar = _transpose_primitives_from_builder(
                _build_coefficient_center_response_from_primitives,
                center_response_bar,
            )
            return dataclasses.replace(
                state,
                density=density_bar,
                pressure=pressure_bar,
                Er=er_bar,
            )

        _, pullback = jax.vjp(
            lambda state_value: self.build_lagged_response(state_value).center_response,
            state,
        )
        (state_bar,) = pullback(center_response_bar)
        return state_bar

    def pullback_build_lagged_response_support_payload(
        self,
        state,
        lagged_response_bar,
        support,
        **kwargs,
    ):
        del kwargs
        center_channels_delta0 = _float_delta_tree_like(support.center_channels)
        center_prepared_delta0 = _float_delta_tree_like(support.center_prepared)
        _, support_delta_pullback = jax.vjp(
            lambda center_channels_delta, center_prepared_delta: self.with_support_payload(
                _support_with_center_delta(
                    support,
                    center_channels_delta,
                    center_prepared_delta,
                )
            ).build_lagged_response(state),
            center_channels_delta0,
            center_prepared_delta0,
        )
        center_channels_bar, center_prepared_bar = support_delta_pullback(lagged_response_bar)
        return _support_bar_from_center_bars(support, center_channels_bar, center_prepared_bar)

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

        if isinstance(center_response, NTXInterpolatedMomentResponse):
            if lagged_timing_enabled():
                jax.debug.callback(lambda: lagged_timing_start("ntx.evaluate_with_lagged_response.coarse"), ordered=True)
            radius_indices = jnp.arange(state.Er.shape[0], dtype=jnp.int32)

            def _current_log_nu_star_per_radius(radius_index):
                drds_value = jax.lax.dynamic_index_in_dim(support.center_channels.drds, radius_index, axis=0, keepdims=False)
                er_value = jax.lax.dynamic_index_in_dim(state.Er, radius_index, axis=0, keepdims=False)
                temperature_local = jax.lax.dynamic_index_in_dim(temperature, radius_index, axis=1, keepdims=False)
                density_local = jax.lax.dynamic_index_in_dim(density, radius_index, axis=1, keepdims=False)
                vthermal_local = jax.lax.dynamic_index_in_dim(v_thermal, radius_index, axis=1, keepdims=False)
                return jax.vmap(
                    lambda species_index: self._log_nu_star_from_nu_hat(
                        self._local_scan_inputs(
                            drds_value=drds_value,
                            species_index=species_index,
                            er_value=er_value,
                            temperature_local=temperature_local,
                            density_local=density_local,
                            vthermal_local=vthermal_local,
                            collisionality_kind=collisionality_kind,
                        )[0]
                    )
                )(species_indices)

            current_log_nu_star = jnp.swapaxes(
                self._map_radius_axis_regularized_at_axis0(
                    _current_log_nu_star_per_radius,
                    radius_indices,
                    self.geometry.r_grid,
                    unbatched=True,
                ),
                0,
                1,
            )
            delta_er = state.Er - center_response.reference_er
            delta_log_nu_star = current_log_nu_star - center_response.reference_log_nu_star
            transport_moments = (
                center_response.reference_transport_moments
                + center_response.dtransport_moments_d_er * delta_er[None, :, None]
                + center_response.dtransport_moments_d_log_nu_star * delta_log_nu_star[:, :, None]
            )
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
            gamma, q, upar = self._regularize_center_fluxes_axis0(gamma, q, upar)
            if lagged_timing_enabled():
                jax.debug.callback(lambda: lagged_timing_end("ntx.evaluate_with_lagged_response.coarse"), ordered=True)
            return {"Gamma": gamma, "Q": q, "Upar": upar}

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
                        derivative_mode_override="direct",
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

        transport_moment_tangent_by_radius = self._map_radius_axis_regularized_at_axis0(
            _transport_moment_tangent_per_radius,
            radius_indices,
            self.geometry.r_grid,
        )
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
        gamma, q, upar = self._regularize_center_fluxes_axis0(gamma, q, upar)
        return {"Gamma": gamma, "Q": q, "Upar": upar}

    def pullback_evaluate_with_lagged_response(self, state, lagged_response, flux_bar, **kwargs):
        del kwargs
        center_response = lagged_response.center_response

        if isinstance(center_response, NTXInterpolatedMomentResponse):
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
            radius_indices = jnp.arange(state.Er.shape[0], dtype=jnp.int32)

            def _current_log_nu_star_per_radius(radius_index):
                drds_value = jax.lax.dynamic_index_in_dim(support.center_channels.drds, radius_index, axis=0, keepdims=False)
                er_value = jax.lax.dynamic_index_in_dim(state.Er, radius_index, axis=0, keepdims=False)
                temperature_local = jax.lax.dynamic_index_in_dim(temperature, radius_index, axis=1, keepdims=False)
                density_local = jax.lax.dynamic_index_in_dim(density, radius_index, axis=1, keepdims=False)
                vthermal_local = jax.lax.dynamic_index_in_dim(v_thermal, radius_index, axis=1, keepdims=False)
                return jax.vmap(
                    lambda species_index: self._log_nu_star_from_nu_hat(
                        self._local_scan_inputs(
                            drds_value=drds_value,
                            species_index=species_index,
                            er_value=er_value,
                            temperature_local=temperature_local,
                            density_local=density_local,
                            vthermal_local=vthermal_local,
                            collisionality_kind=collisionality_kind,
                        )[0]
                    )
                )(species_indices)

            current_log_nu_star = jnp.swapaxes(
                self._map_radius_axis_regularized_at_axis0(
                    _current_log_nu_star_per_radius,
                    radius_indices,
                    self.geometry.r_grid,
                    unbatched=True,
                ),
                0,
                1,
            )
            delta_er = state.Er - center_response.reference_er
            delta_log_nu_star = current_log_nu_star - center_response.reference_log_nu_star
            transport_moments = (
                center_response.reference_transport_moments
                + center_response.dtransport_moments_d_er * delta_er[None, :, None]
                + center_response.dtransport_moments_d_log_nu_star * delta_log_nu_star[:, :, None]
            )
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
            gamma_bar, q_bar, upar_bar = jax.linear_transpose(
                lambda gamma_value, q_value, upar_value: self._regularize_center_fluxes_axis0(
                    gamma_value,
                    q_value,
                    upar_value,
                ),
                gamma,
                q,
                upar,
            )((flux_bar["Gamma"], flux_bar["Q"], flux_bar["Upar"]))

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
                    state.Er,
                )
            )(self.species.charge, density, temperature, dndr_all, dTdr_all)
            a2 = jax.vmap(get_Thermodynamical_Forces_A2)(temperature, dTdr_all)
            a3 = get_Thermodynamical_Forces_A3(state.Er)
            density_phys = DENSITY_STATE_TO_PHYSICAL * density
            temperature_phys = TEMPERATURE_STATE_TO_PHYSICAL * temperature

            lij_bar = jnp.zeros_like(lij)
            lij_bar = lij_bar.at[:, :, 0, 0].add(-density_phys * a1 * gamma_bar)
            lij_bar = lij_bar.at[:, :, 0, 1].add(-density_phys * a2 * gamma_bar)
            lij_bar = lij_bar.at[:, :, 0, 2].add(-density_phys * a3[None, :] * gamma_bar)
            lij_bar = lij_bar.at[:, :, 1, 0].add(-temperature_phys * density_phys * a1 * q_bar)
            lij_bar = lij_bar.at[:, :, 1, 1].add(-temperature_phys * density_phys * a2 * q_bar)
            lij_bar = lij_bar.at[:, :, 1, 2].add(-temperature_phys * density_phys * a3[None, :] * q_bar)
            lij_bar = lij_bar.at[:, :, 2, 0].add(-density_phys * a1 * upar_bar)
            lij_bar = lij_bar.at[:, :, 2, 1].add(-density_phys * a2 * upar_bar)
            lij_bar = lij_bar.at[:, :, 2, 2].add(-density_phys * a3[None, :] * upar_bar)

            charge = jnp.asarray(self.species.charge, dtype=jnp.float64)[:, None]
            mass = jnp.asarray(self.species.mass, dtype=jnp.float64)[:, None]
            inv_sqrt_pi = 1.0 / jnp.sqrt(jnp.pi)
            l11_fac = -inv_sqrt_pi * (mass / charge) ** 2 * v_thermal**3
            l13_fac = -inv_sqrt_pi * (mass / charge) * v_thermal**2
            l33_fac = -inv_sqrt_pi * v_thermal
            transport_moments_bar = jnp.stack(
                (
                    l11_fac * lij_bar[:, :, 0, 0],
                    l11_fac * (lij_bar[:, :, 0, 1] + lij_bar[:, :, 1, 0]),
                    l11_fac * lij_bar[:, :, 1, 1],
                    l13_fac * (lij_bar[:, :, 0, 2] - lij_bar[:, :, 2, 0]),
                    l13_fac * (lij_bar[:, :, 1, 2] - lij_bar[:, :, 2, 1]),
                    l33_fac * lij_bar[:, :, 2, 2],
                ),
                axis=2,
            )

            reference_transport_moments_bar = transport_moments_bar
            dtransport_moments_d_er_bar = transport_moments_bar * delta_er[None, :, None]
            dtransport_moments_d_log_nu_star_bar = transport_moments_bar * delta_log_nu_star[:, :, None]
            reference_er_bar = -jnp.sum(
                transport_moments_bar * center_response.dtransport_moments_d_er,
                axis=(0, 2),
            )
            reference_log_nu_star_bar = -jnp.sum(
                transport_moments_bar * center_response.dtransport_moments_d_log_nu_star,
                axis=2,
            )
            return NTXExactLijLaggedResponse(
                center_response=NTXInterpolatedMomentResponse(
                    reference_er=reference_er_bar,
                    reference_log_nu_star=reference_log_nu_star_bar,
                    reference_transport_moments=reference_transport_moments_bar,
                    dtransport_moments_d_er=dtransport_moments_d_er_bar,
                    dtransport_moments_d_log_nu_star=dtransport_moments_d_log_nu_star_bar,
                )
            )

        if isinstance(center_response, NTXPreparedCoefficientResponse):
            def _reduced_response(
                reference_transport_moments,
                reference_nu_hat,
                reference_epsi_hat,
            ):
                return self.evaluate_with_lagged_response(
                    state,
                    NTXExactLijLaggedResponse(
                        center_response=NTXPreparedCoefficientResponse(
                            reference_transport_moments=reference_transport_moments,
                            reference_nu_hat=reference_nu_hat,
                            reference_epsi_hat=reference_epsi_hat,
                        )
                    ),
                )

            _, pb = jax.vjp(
                _reduced_response,
                center_response.reference_transport_moments,
                center_response.reference_nu_hat,
                center_response.reference_epsi_hat,
            )
            (
                reference_transport_moments_bar,
                reference_nu_hat_bar,
                reference_epsi_hat_bar,
            ) = pb(flux_bar)
            return NTXExactLijLaggedResponse(
                center_response=NTXPreparedCoefficientResponse(
                    reference_transport_moments=reference_transport_moments_bar,
                    reference_nu_hat=reference_nu_hat_bar,
                    reference_epsi_hat=reference_epsi_hat_bar,
                )
            )

        _, pb = jax.vjp(
            lambda response_value: self.evaluate_with_lagged_response(
                state,
                NTXExactLijLaggedResponse(center_response=response_value),
            ),
            center_response,
        )
        (center_response_bar,) = pb(flux_bar)
        return NTXExactLijLaggedResponse(center_response=center_response_bar)

    def pullback_evaluate_with_lagged_response_support_payload(
        self,
        state,
        lagged_response,
        flux_bar,
        support,
        **kwargs,
    ):
        del kwargs
        center_delta0 = _float_delta_tree_like(support.center_channels)
        face_delta0 = _float_delta_tree_like(support.face_channels)
        _, channel_delta_pullback = jax.vjp(
            lambda center_delta, face_delta: self.with_support_payload(
                _support_with_channel_delta(support, center_delta, face_delta)
            ).evaluate_with_lagged_response(state, lagged_response),
            center_delta0,
            face_delta0,
        )
        center_bar, face_bar = channel_delta_pullback(flux_bar)
        return _support_bar_from_channel_bars(support, center_bar, face_bar)

    def pullback_evaluate_with_lagged_response_state(self, state, lagged_response, flux_bar, **kwargs):
        del kwargs
        center_response = lagged_response.center_response

        if isinstance(center_response, NTXInterpolatedMomentResponse):
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
            radius_indices = jnp.arange(state.Er.shape[0], dtype=jnp.int32)

            def _raw_log_nu_star_per_radius(radius_index):
                drds_value = jax.lax.dynamic_index_in_dim(support.center_channels.drds, radius_index, axis=0, keepdims=False)
                er_local = jax.lax.dynamic_index_in_dim(state.Er, radius_index, axis=0, keepdims=False)
                temperature_local = jax.lax.dynamic_index_in_dim(temperature, radius_index, axis=1, keepdims=False)
                density_local = jax.lax.dynamic_index_in_dim(density, radius_index, axis=1, keepdims=False)
                vthermal_local = jax.lax.dynamic_index_in_dim(v_thermal, radius_index, axis=1, keepdims=False)
                return jax.vmap(
                    lambda species_index: self._log_nu_star_from_nu_hat(
                        self._local_scan_inputs(
                            drds_value=drds_value,
                            species_index=species_index,
                            er_value=er_local,
                            temperature_local=temperature_local,
                            density_local=density_local,
                            vthermal_local=vthermal_local,
                            collisionality_kind=collisionality_kind,
                        )[0]
                    )
                )(species_indices)

            raw_current_log_nu_star_by_radius = self._map_radius_axis_unbatched(
                _raw_log_nu_star_per_radius,
                radius_indices,
            )
            current_log_nu_star_by_radius = self._map_radius_axis_regularized_at_axis0(
                _raw_log_nu_star_per_radius,
                radius_indices,
                self.geometry.r_grid,
                unbatched=True,
            )
            current_log_nu_star = jnp.swapaxes(current_log_nu_star_by_radius, 0, 1)
            delta_er = state.Er - center_response.reference_er
            delta_log_nu_star = current_log_nu_star - center_response.reference_log_nu_star
            transport_moments = (
                center_response.reference_transport_moments
                + center_response.dtransport_moments_d_er * delta_er[None, :, None]
                + center_response.dtransport_moments_d_log_nu_star * delta_log_nu_star[:, :, None]
            )
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
            gamma_bar, q_bar, upar_bar = jax.linear_transpose(
                lambda gamma_value, q_value, upar_value: self._regularize_center_fluxes_axis0(
                    gamma_value,
                    q_value,
                    upar_value,
                ),
                gamma,
                q,
                upar,
            )((flux_bar["Gamma"], flux_bar["Q"], flux_bar["Upar"]))

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
            species_charge = jnp.asarray(self.species.charge, dtype=jnp.float64)[:, None]
            a1 = dndr_all / density - 1.5 * dTdr_all / temperature - state.Er[None, :] * species_charge / (
                temperature * elementary_charge
            )
            a2 = dTdr_all / temperature
            a3 = jnp.zeros_like(state.Er)
            density_phys = DENSITY_STATE_TO_PHYSICAL * density
            temperature_phys = TEMPERATURE_STATE_TO_PHYSICAL * temperature

            gamma_sum = lij[:, :, 0, 0] * a1 + lij[:, :, 0, 1] * a2 + lij[:, :, 0, 2] * a3[None, :]
            q_sum = lij[:, :, 1, 0] * a1 + lij[:, :, 1, 1] * a2 + lij[:, :, 1, 2] * a3[None, :]
            upar_sum = lij[:, :, 2, 0] * a1 + lij[:, :, 2, 1] * a2 + lij[:, :, 2, 2] * a3[None, :]

            gamma_sum_bar = -density_phys * gamma_bar
            q_sum_bar = -temperature_phys * density_phys * q_bar
            upar_sum_bar = -density_phys * upar_bar

            density_bar_direct = (
                -DENSITY_STATE_TO_PHYSICAL * gamma_sum * gamma_bar
                -TEMPERATURE_STATE_TO_PHYSICAL * temperature * DENSITY_STATE_TO_PHYSICAL * q_sum * q_bar
                -DENSITY_STATE_TO_PHYSICAL * upar_sum * upar_bar
            )
            temperature_bar_direct = (
                -TEMPERATURE_STATE_TO_PHYSICAL * density_phys * q_sum * q_bar
            )

            lij_bar = jnp.zeros_like(lij)
            lij_bar = lij_bar.at[:, :, 0, 0].add(gamma_sum_bar * a1)
            lij_bar = lij_bar.at[:, :, 0, 1].add(gamma_sum_bar * a2)
            lij_bar = lij_bar.at[:, :, 0, 2].add(gamma_sum_bar * a3[None, :])
            lij_bar = lij_bar.at[:, :, 1, 0].add(q_sum_bar * a1)
            lij_bar = lij_bar.at[:, :, 1, 1].add(q_sum_bar * a2)
            lij_bar = lij_bar.at[:, :, 1, 2].add(q_sum_bar * a3[None, :])
            lij_bar = lij_bar.at[:, :, 2, 0].add(upar_sum_bar * a1)
            lij_bar = lij_bar.at[:, :, 2, 1].add(upar_sum_bar * a2)
            lij_bar = lij_bar.at[:, :, 2, 2].add(upar_sum_bar * a3[None, :])

            a1_bar = (
                lij[:, :, 0, 0] * gamma_sum_bar
                + lij[:, :, 1, 0] * q_sum_bar
                + lij[:, :, 2, 0] * upar_sum_bar
            )
            a2_bar = (
                lij[:, :, 0, 1] * gamma_sum_bar
                + lij[:, :, 1, 1] * q_sum_bar
                + lij[:, :, 2, 1] * upar_sum_bar
            )

            dndr_bar = a1_bar / density
            density_bar_direct = density_bar_direct - a1_bar * dndr_all / (density * density)
            dTdr_bar = -1.5 * a1_bar / temperature + a2_bar / temperature
            temperature_bar_direct = (
                temperature_bar_direct
                + a1_bar * (1.5 * dTdr_all / (temperature * temperature))
                + a1_bar * state.Er[None, :] * species_charge / (elementary_charge * temperature * temperature)
                - a2_bar * dTdr_all / (temperature * temperature)
            )
            er_bar_direct = -jnp.sum(a1_bar * species_charge / (elementary_charge * temperature), axis=0)

            def _density_gradient_map(density_value):
                local_n_right = _as_species_constraint(None if self.bc_density is None else getattr(self.bc_density, "right_value", None), n_species)
                if local_n_right is None:
                    local_n_right = density_value[:, -1]
                local_n_right_grad = _as_species_constraint(None if self.bc_density is None else getattr(self.bc_density, "right_gradient", None), n_species)
                if local_n_right_grad is None:
                    local_n_right_grad = jnp.zeros_like(local_n_right)
                return jax.vmap(
                    lambda density_a, right_value, right_grad: get_gradient_density(
                        density_a,
                        self.geometry.r_grid,
                        self.geometry.r_grid_half,
                        self.geometry.dr,
                        right_face_constraint=right_value,
                        right_face_grad_constraint=right_grad,
                    )
                )(density_value, local_n_right, local_n_right_grad)

            def _temperature_gradient_map(temperature_value):
                local_t_right = _as_species_constraint(None if self.bc_temperature is None else getattr(self.bc_temperature, "right_value", None), n_species)
                if local_t_right is None:
                    local_t_right = temperature_value[:, -1]
                local_t_right_grad = _as_species_constraint(None if self.bc_temperature is None else getattr(self.bc_temperature, "right_gradient", None), n_species)
                if local_t_right_grad is None:
                    local_t_right_grad = jnp.zeros_like(local_t_right)
                return jax.vmap(
                    lambda temperature_a, right_value, right_grad: get_gradient_temperature(
                        temperature_a,
                        self.geometry.r_grid,
                        self.geometry.r_grid_half,
                        self.geometry.dr,
                        right_face_constraint=right_value,
                        right_face_grad_constraint=right_grad,
                    )
                )(temperature_value, local_t_right, local_t_right_grad)

            (density_grad_bar,) = jax.linear_transpose(_density_gradient_map, density)(dndr_bar)
            (temperature_grad_bar,) = jax.linear_transpose(_temperature_gradient_map, temperature)(dTdr_bar)
            density_bar_direct = density_bar_direct + density_grad_bar
            temperature_bar_direct = temperature_bar_direct + temperature_grad_bar

            charge = jnp.asarray(self.species.charge, dtype=jnp.float64)[:, None]
            mass = jnp.asarray(self.species.mass, dtype=jnp.float64)[:, None]
            inv_sqrt_pi = 1.0 / jnp.sqrt(jnp.pi)
            l11_coeff = -inv_sqrt_pi * (mass / charge) ** 2
            l13_coeff = -inv_sqrt_pi * (mass / charge)
            l33_coeff = -inv_sqrt_pi
            l11_fac = l11_coeff * v_thermal**3
            l13_fac = l13_coeff * v_thermal**2
            l33_fac = l33_coeff * v_thermal

            transport_moments_bar = jnp.stack(
                (
                    l11_fac * lij_bar[:, :, 0, 0],
                    l11_fac * (lij_bar[:, :, 0, 1] + lij_bar[:, :, 1, 0]),
                    l11_fac * lij_bar[:, :, 1, 1],
                    l13_fac * (lij_bar[:, :, 0, 2] - lij_bar[:, :, 2, 0]),
                    l13_fac * (lij_bar[:, :, 1, 2] - lij_bar[:, :, 2, 1]),
                    l33_fac * lij_bar[:, :, 2, 2],
                ),
                axis=2,
            )
            l11_fac_bar = (
                transport_moments[:, :, 0] * lij_bar[:, :, 0, 0]
                + transport_moments[:, :, 1] * (lij_bar[:, :, 0, 1] + lij_bar[:, :, 1, 0])
                + transport_moments[:, :, 2] * lij_bar[:, :, 1, 1]
            )
            l13_fac_bar = (
                transport_moments[:, :, 3] * (lij_bar[:, :, 0, 2] - lij_bar[:, :, 2, 0])
                + transport_moments[:, :, 4] * (lij_bar[:, :, 1, 2] - lij_bar[:, :, 2, 1])
            )
            l33_fac_bar = transport_moments[:, :, 5] * lij_bar[:, :, 2, 2]
            v_thermal_bar = (
                3.0 * l11_coeff * v_thermal**2 * l11_fac_bar
                + 2.0 * l13_coeff * v_thermal * l13_fac_bar
                + l33_coeff * l33_fac_bar
            )

            er_bar = er_bar_direct + jnp.sum(
                transport_moments_bar * center_response.dtransport_moments_d_er,
                axis=(0, 2),
            )
            current_log_nu_star_bar = jnp.sum(
                transport_moments_bar * center_response.dtransport_moments_d_log_nu_star,
                axis=2,
            )
            temperature_bar = temperature_bar_direct + 0.5 * v_thermal_bar * v_thermal / temperature

            regularized_log_bar_by_radius = jnp.swapaxes(current_log_nu_star_bar, 0, 1)
            raw_log_templates = jnp.zeros_like(raw_current_log_nu_star_by_radius)

            def _regularize_raw_log_values(raw_values):
                n_radius = int(radius_indices.shape[0])
                if n_radius < 4:
                    return raw_values

                def _regularized_skip_axis(_):
                    mapped_with_placeholder = jnp.concatenate([raw_values[1:2], raw_values[1:]], axis=0)
                    return self._regularize_axis_radius0(mapped_with_placeholder, self.geometry.r_grid)

                def _direct_map(_):
                    return raw_values

                return jax.lax.cond(
                    jnp.isclose(jnp.asarray(self.geometry.r_grid, dtype=jnp.float64)[0], 0.0),
                    _regularized_skip_axis,
                    _direct_map,
                    operand=None,
                )

            (raw_log_bar_by_radius,) = jax.linear_transpose(
                _regularize_raw_log_values,
                raw_log_templates,
            )(regularized_log_bar_by_radius)

            def _local_log_nu_star_pullback(radius_index, log_bar_local):
                drds_value = jax.lax.dynamic_index_in_dim(support.center_channels.drds, radius_index, axis=0, keepdims=False)
                er_local = jax.lax.dynamic_index_in_dim(state.Er, radius_index, axis=0, keepdims=False)
                temperature_local = jax.lax.dynamic_index_in_dim(temperature, radius_index, axis=1, keepdims=False)
                density_local = jax.lax.dynamic_index_in_dim(density, radius_index, axis=1, keepdims=False)
                vthermal_local = jax.lax.dynamic_index_in_dim(v_thermal, radius_index, axis=1, keepdims=False)

                def _per_species(species_index, species_log_bar):
                    nu_hat, epsi_hat, vth_a = self._interpolated_moment_local_scan_primitives(
                        drds_value=drds_value,
                        species_index=species_index,
                        er_value=er_local,
                        temperature_local=temperature_local,
                        density_local=density_local,
                        vthermal_local=vthermal_local,
                        collisionality_kind=collisionality_kind,
                    )
                    del epsi_hat
                    nu_hat_bar = self._pullback_log_nu_star_from_nu_hat(nu_hat, species_log_bar)
                    return self._pullback_local_scan_inputs_from_primitives(
                        drds_value=drds_value,
                        species_index=species_index,
                        er_value=er_local,
                        temperature_local=temperature_local,
                        density_local=density_local,
                        vthermal_local=vthermal_local,
                        collisionality_kind=collisionality_kind,
                        reference_nu_hat_bar=nu_hat_bar,
                        reference_epsi_hat_bar=jnp.zeros_like(nu_hat),
                        vth_a_bar=jnp.zeros_like(vth_a),
                    )

                er_species_bar, temperature_species_bar, density_species_bar = jax.vmap(
                    _per_species,
                    in_axes=(0, 0),
                )(species_indices, log_bar_local)
                return (
                    jnp.sum(er_species_bar, axis=0),
                    jnp.sum(temperature_species_bar, axis=0),
                    jnp.sum(density_species_bar, axis=0),
                )

            er_log_bar_by_radius, temperature_log_bar_by_radius, density_log_bar_by_radius = jax.vmap(
                _local_log_nu_star_pullback,
                in_axes=(0, 0),
            )(radius_indices, raw_log_bar_by_radius)
            er_bar = er_bar + er_log_bar_by_radius
            temperature_bar = temperature_bar + jnp.swapaxes(temperature_log_bar_by_radius, 0, 1)
            density_bar_direct = density_bar_direct + jnp.swapaxes(density_log_bar_by_radius, 0, 1)
            density_floor_arr = _broadcast_species_floor(jnp.asarray(state.density), DEFAULT_TRANSPORT_DENSITY_FLOOR)
            density_active = jnp.asarray(state.density) > density_floor_arr
            density_safe = safe_density(state.density)
            pressure_bar = temperature_bar / density_safe
            density_bar = density_bar_direct - temperature_bar * state.pressure / (density_safe * density_safe)
            density_bar = density_bar * density_active.astype(density_bar.dtype)
            return dataclasses.replace(
                state,
                density=density_bar,
                pressure=pressure_bar,
                Er=er_bar,
            )

        _, pb = jax.vjp(
            lambda state_value: self.evaluate_with_lagged_response(
                state_value,
                lagged_response,
            ),
            state,
        )
        (state_bar,) = pb(flux_bar)
        return state_bar

    def build_local_particle_flux_evaluator(self, state):
        density = safe_density(state.density)
        temperature = state.temperature
        support = self._static_support()
        collisionality_kind = _collisionality_kind(self.collisionality_model)
        v_thermal = get_v_thermal(self.species.mass, temperature)
        species_indices = jnp.arange(int(self.species.number_species), dtype=jnp.int32)
        density_right_constraint, density_right_grad_constraint = _extract_right_constraints(
            self.bc_density,
            density,
        )
        temperature_right_constraint, temperature_right_grad_constraint = _extract_right_constraints(
            self.bc_temperature,
            temperature,
        )
        dndr_all = jax.vmap(
            lambda density_a, right_value, right_grad: get_gradient_density(
                density_a,
                self.geometry.r_grid,
                self.geometry.r_grid_half,
                self.geometry.dr,
                right_face_constraint=right_value,
                right_face_grad_constraint=right_grad,
            )
        )(density, density_right_constraint, density_right_grad_constraint)
        dTdr_all = jax.vmap(
            lambda temperature_a, right_value, right_grad: get_gradient_temperature(
                temperature_a,
                self.geometry.r_grid,
                self.geometry.r_grid_half,
                self.geometry.dr,
                right_face_constraint=right_value,
                right_face_grad_constraint=right_grad,
            )
        )(temperature, temperature_right_constraint, temperature_right_grad_constraint)

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
                    derivative_mode_override="direct",
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
    ntx_exact_surface_backend="vmec",
    ntx_exact_face_response_mode="face_local_response",
    ntx_exact_radial_batch_size=None,
    ntx_exact_radial_batch_mode="simple",
    ntx_exact_scan_batch_size=None,
    ntx_exact_response_anchor_count=None,
    ntx_exact_use_remat=False,
    ntx_exact_derivative_mode="direct",
    ntx_exact_derivative_field_pullback_mode="generic_jvp",
    ntx_exact_derivative_pullback_boundary="inline",
    ntx_exact_derivative_pullback_algebra="ntx_helper",
    ntx_exact_er_v_floor=None,
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
        radial_batch_mode=NTXExactLijRuntimeTransportModel._normalize_radial_batch_mode(
            ntx_exact_radial_batch_mode
        ),
        scan_batch_size=(
            None
            if ntx_exact_scan_batch_size in (None, "", 0, "0")
            else int(ntx_exact_scan_batch_size)
        ),
        response_anchor_count=(
            None
            if ntx_exact_response_anchor_count in (None, "", 0, "0")
            else int(ntx_exact_response_anchor_count)
        ),
        use_remat=bool(ntx_exact_use_remat),
        derivative_mode=NTXExactLijRuntimeTransportModel._normalize_derivative_mode(
            ntx_exact_derivative_mode
        ),
        derivative_field_pullback_mode=NTXExactLijRuntimeTransportModel._normalize_derivative_field_pullback_mode(
            ntx_exact_derivative_field_pullback_mode
        ),
        derivative_pullback_boundary=NTXExactLijRuntimeTransportModel._normalize_derivative_pullback_boundary(
            ntx_exact_derivative_pullback_boundary
        ),
        derivative_pullback_algebra=NTXExactLijRuntimeTransportModel._normalize_derivative_pullback_algebra(
            ntx_exact_derivative_pullback_algebra
        ),
        er_v_floor=(
            None
            if ntx_exact_er_v_floor in (None, "", 0, "0")
            else float(ntx_exact_er_v_floor)
        ),
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


def _normalize_perturb_species_flux_dataset(arr, n_species):
    out = jnp.asarray(arr, dtype=float)
    if out.ndim != 3:
        raise ValueError(f"Perturbation flux dataset must be 3D, got shape {out.shape}.")
    if out.shape[1] == n_species:
        return out
    if out.shape[2] == n_species:
        return jnp.swapaxes(out, 1, 2)
    raise ValueError(f"Perturbation flux dataset shape {out.shape} is not compatible with n_species={n_species}.")


def _spectrax_perturb_kind_codes(labels):
    mapping = {
        "density_gradient": 0,
        "temperature_gradient": 1,
    }
    codes = []
    for raw in labels:
        key = str(raw.decode("utf-8") if isinstance(raw, bytes) else raw).strip().lower()
        if key not in mapping:
            raise ValueError(f"Unsupported SPECTRAX perturb_kind {raw!r}.")
        codes.append(mapping[key])
    return jnp.asarray(codes, dtype=jnp.int32)


def _spectrax_perturb_species_indices(labels, species_names):
    lookup = {str(name).strip().lower(): idx for idx, name in enumerate(species_names)}
    indices = []
    for raw in labels:
        key = str(raw.decode("utf-8") if isinstance(raw, bytes) else raw).strip().lower()
        if key not in lookup:
            raise ValueError(f"Unknown SPECTRAX perturb_species {raw!r} for NEOPAX species {species_names!r}.")
        indices.append(lookup[key])
    return jnp.asarray(indices, dtype=jnp.int32)


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


def read_flux_profile_fd_response_file(path, n_species, species_names):
    with h5py.File(path, "r") as f:
        keys = set(f.keys())
        required = {
            "Gamma_perturb",
            "Q_perturb",
            "perturb_delta",
            "perturb_present",
            "perturb_kind",
            "perturb_species",
        }
        if not required.issubset(keys):
            missing = ", ".join(sorted(required - keys))
            raise ValueError(
                f"Flux file '{path}' is missing SPECTRAX FD lagged-response datasets: {missing}."
            )
        gamma_perturb = _normalize_perturb_species_flux_dataset(f["Gamma_perturb"][...], n_species)
        q_perturb = _normalize_perturb_species_flux_dataset(f["Q_perturb"][...], n_species)
        perturb_delta = jnp.asarray(f["perturb_delta"][...], dtype=float)
        perturb_present = jnp.asarray(f["perturb_present"][...], dtype=bool)
        perturb_kind_codes = _spectrax_perturb_kind_codes(f["perturb_kind"][...])
        perturb_species_indices = _spectrax_perturb_species_indices(f["perturb_species"][...], species_names)
    return (
        gamma_perturb,
        q_perturb,
        perturb_delta,
        perturb_present,
        perturb_kind_codes,
        perturb_species_indices,
    )


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
    lagged_response_mode = str(
        kwargs.pop(
            "lagged_response_mode",
            kwargs.pop("response_mode", "none"),
        )
    ).strip().lower()
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
    gamma_perturb_data = None
    q_perturb_data = None
    perturb_delta_data = None
    perturb_present_data = None
    perturb_kind_codes = None
    perturb_species_indices = None
    if lagged_response_mode == "fd":
        (
            gamma_perturb_data,
            q_perturb_data,
            perturb_delta_data,
            perturb_present_data,
            perturb_kind_codes,
            perturb_species_indices,
        ) = read_flux_profile_fd_response_file(path, species.number_species, species.names)
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
        lagged_response_mode=lagged_response_mode,
        gamma_perturb_data=gamma_perturb_data,
        q_perturb_data=q_perturb_data,
        perturb_delta_data=perturb_delta_data,
        perturb_present_data=perturb_present_data,
        perturb_kind_codes=perturb_kind_codes,
        perturb_species_indices=perturb_species_indices,
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
    lagged_response_mode: str = "none"
    gamma_perturb_data: Any = None
    q_perturb_data: Any = None
    perturb_delta_data: Any = None
    perturb_present_data: Any = None
    perturb_kind_codes: Any = None
    perturb_species_indices: Any = None

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

    def _interp_perturb_species_profile(self, data, target_r):
        if data is None:
            return None
        return jax.vmap(
            lambda perts: jax.vmap(lambda prof: interpax.interp1d(target_r, self.r_data, prof))(perts)
        )(data)

    def _interp_perturb_scalar_profile(self, data, target_r):
        if data is None:
            return None
        return jax.vmap(lambda prof: interpax.interp1d(target_r, self.r_data, prof))(data)

    def _spectrax_fd_basis(self, state):
        density = safe_density(state.density)
        temperature = safe_temperature(state.temperature, 1.0e-12)
        a_minor = jnp.asarray(getattr(self.geometry, "a_b", 1.0), dtype=density.dtype)
        dndr_all = jax.vmap(
            lambda density_a: get_gradient_density(
                density_a,
                self.geometry.r_grid,
                self.geometry.r_grid_half,
                self.geometry.dr,
                right_face_constraint=density_a[-1],
            )
        )(density)
        dTdr_all = jax.vmap(
            lambda temperature_a: get_gradient_temperature(
                temperature_a,
                self.geometry.r_grid,
                self.geometry.r_grid_half,
                self.geometry.dr,
                right_face_constraint=temperature_a[-1],
            )
        )(temperature)
        # Stage 4 currently writes SPECTRAX perturbations in the default
        # gradient_coordinate='rho' convention, so convert NEOPAX's physical-r
        # gradients back to d/d rho using rho = r / a_minor.
        density_basis = -a_minor * dndr_all / density
        temperature_basis = -a_minor * dTdr_all / temperature
        return jax.vmap(
            lambda kind_code, species_index: jax.lax.cond(
                kind_code == 0,
                lambda _: density_basis[species_index],
                lambda _: temperature_basis[species_index],
                operand=None,
            )
        )(self.perturb_kind_codes, self.perturb_species_indices)

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

    def build_lagged_response(self, state, **kwargs):
        del kwargs
        if str(self.lagged_response_mode).strip().lower() != "fd":
            return JVPTransportFluxResponse(reference_state=state, reference_flux=self(state))
        if (
            self.gamma_perturb_data is None
            or self.q_perturb_data is None
            or self.perturb_delta_data is None
            or self.perturb_present_data is None
            or self.perturb_kind_codes is None
            or self.perturb_species_indices is None
        ):
            raise ValueError(
                "fluxes_r_file lagged_response_mode='fd' requires SPECTRAX perturbation datasets in the file."
            )
        return SpectraXTurbulenceFDLaggedResponse(
            reference_state=state,
            reference_flux=self(state),
            reference_basis=self._spectrax_fd_basis(state),
            perturb_kind_codes=self.perturb_kind_codes,
            perturb_species_indices=self.perturb_species_indices,
            perturb_delta=self._interp_perturb_scalar_profile(self.perturb_delta_data, self.geometry.r_grid),
            perturb_present=self._interp_perturb_scalar_profile(
                self.perturb_present_data.astype(float),
                self.geometry.r_grid,
            )
            > 0.5,
            gamma_perturb=self._interp_perturb_species_profile(self.gamma_perturb_data, self.geometry.r_grid),
            q_perturb=self.q_scale * self._interp_perturb_species_profile(self.q_perturb_data, self.geometry.r_grid),
        )

    def evaluate_with_lagged_response(self, state, lagged_response, **kwargs):
        del kwargs
        if (
            str(self.lagged_response_mode).strip().lower() != "fd"
            or not isinstance(lagged_response, SpectraXTurbulenceFDLaggedResponse)
        ):
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

        current_basis = self._spectrax_fd_basis(state)
        delta_basis = current_basis - lagged_response.reference_basis
        perturb_present = lagged_response.perturb_present[:, None, :]
        safe_delta = jnp.where(
            perturb_present[:, 0, :],
            lagged_response.perturb_delta,
            1.0,
        )
        dgamma = jnp.where(
            perturb_present,
            (lagged_response.gamma_perturb - lagged_response.reference_flux["Gamma"][None, :, :]) / safe_delta[:, None, :],
            0.0,
        )
        dq = jnp.where(
            perturb_present,
            (lagged_response.q_perturb - lagged_response.reference_flux["Q"][None, :, :]) / safe_delta[:, None, :],
            0.0,
        )
        gamma = lagged_response.reference_flux["Gamma"] + jnp.sum(dgamma * delta_basis[:, None, :], axis=0)
        q = lagged_response.reference_flux["Q"] + jnp.sum(dq * delta_basis[:, None, :], axis=0)
        return {
            "Gamma": gamma,
            "Q": q,
            "Upar": lagged_response.reference_flux["Upar"],
        }






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


@dataclasses.dataclass(frozen=True, eq=False)
class SpectraXQuasilinearRuntimeTransportModel(TransportFluxModelBase):
    species: Any
    geometry: Any
    backend_mode: str = "smooth_proxy"
    b0: float = 0.0
    b1: float = 1.0
    b2: float = 0.0
    particle_flux_scale: float = 0.0
    adiabatic_electrons_only: bool = True
    electrostatic_only: bool = True
    spectrax_root: str | None = None
    template: str | None = None

    def _evaluate_runtime(self, state) -> tuple[dict[str, jax.Array], SpectraXQuasilinearRuntimeDiagnostics]:
        backend_mode = str(self.backend_mode).strip().lower()
        if backend_mode != "smooth_proxy":
            raise NotImplementedError(
                "SPECTRAX-GK external runtime backend is not implemented yet. "
                "Use backend_mode='smooth_proxy' for the current in-repo scaffold."
            )
        return evaluate_spectrax_quasilinear_proxy(
            state=state,
            species=self.species,
            geometry=self.geometry,
            b0=self.b0,
            b1=self.b1,
            b2=self.b2,
            adiabatic_electrons_only=bool(self.adiabatic_electrons_only),
            particle_flux_scale=float(self.particle_flux_scale),
        )

    def __call__(self, state) -> dict:
        fluxes, _diagnostics = self._evaluate_runtime(state)
        return fluxes

    def build_local_particle_flux_evaluator(self, state):
        fluxes, _diagnostics = self._evaluate_runtime(state)
        gamma_turb = fluxes["Gamma"]

        def evaluator(radius_index, er_value):
            del er_value
            return gamma_turb[:, radius_index]

        return evaluator

    def evaluate_face_fluxes(self, state, face_state, **kwargs):
        del state, kwargs
        fluxes, _diagnostics = self._evaluate_runtime(face_state)
        return fluxes

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
    "spectrax_quasilinear_runtime",
    lambda species, energy_grid, geometry, database=None, **kwargs: SpectraXQuasilinearRuntimeTransportModel(
        species=species,
        geometry=geometry,
        backend_mode=str(kwargs.get("backend_mode", kwargs.get("mode", "smooth_proxy"))),
        b0=float(kwargs.get("b0", 0.0)),
        b1=float(kwargs.get("b1", 1.0)),
        b2=float(kwargs.get("b2", 0.0)),
        particle_flux_scale=float(kwargs.get("particle_flux_scale", 0.0)),
        adiabatic_electrons_only=bool(kwargs.get("adiabatic_electrons_only", True)),
        electrostatic_only=bool(kwargs.get("electrostatic_only", True)),
        spectrax_root=kwargs.get("spectrax_root"),
        template=kwargs.get("template"),
    ),
)

register_transport_flux_model(
    "spectrax_quasilinear_runtime_lagged",
    lambda species, energy_grid, geometry, database=None, **kwargs: SpectraXQuasilinearRuntimeTransportModel(
        species=species,
        geometry=geometry,
        backend_mode=str(kwargs.get("backend_mode", kwargs.get("mode", "smooth_proxy"))),
        b0=float(kwargs.get("b0", 0.0)),
        b1=float(kwargs.get("b1", 1.0)),
        b2=float(kwargs.get("b2", 0.0)),
        particle_flux_scale=float(kwargs.get("particle_flux_scale", 0.0)),
        adiabatic_electrons_only=bool(kwargs.get("adiabatic_electrons_only", True)),
        electrostatic_only=bool(kwargs.get("electrostatic_only", True)),
        spectrax_root=kwargs.get("spectrax_root"),
        template=kwargs.get("template"),
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
