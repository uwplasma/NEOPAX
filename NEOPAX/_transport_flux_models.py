from __future__ import annotations


import functools
import contextlib
import os
from typing import Any, Callable
import abc
import dataclasses
import h5py
import jax
import jax.numpy as jnp
import interpax
import lineax
import numpy as np
import sys
import types
import warnings
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
    _nu_over_vnew,
    assemble_momentum_lij_matrices,
    _nu_over_vnew_local,
    get_Collision_Operator_terms,
    get_Lij_matrix_with_momentum_correction,
    get_Matrix,
    get_momentum_Correction,
    get_corrected_fluxes,
    get_Lij_matrix,
    get_Lij_matrix_local,
    get_Neoclassical_Fluxes,
    get_Neoclassical_Fluxes_Faces,
    get_Neoclassical_Fluxes_With_Momentum_Correction,
    get_Neoclassical_Upar_With_Momentum_Correction,
    pullback_preprocessed_radial_database_fluxes,
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
from ._second_order_response import (
    DirectionalSecondOrderJet,
    add as _jet_add,
    absolute_with_fixed_anchor_sign as _jet_abs,
    divide as _jet_divide,
    compose_ntx_coefficient_quadratic,
    dynamic_index as _jet_dynamic_index,
    erf as _jet_erf,
    evaluate as _jet_evaluate,
    exp as _jet_exp,
    log10 as _jet_log10,
    maximum_with_constant_floor,
    multiply as _jet_multiply,
    negate as _jet_negate,
    select_axis as _jet_select_axis,
    seed as _jet_seed,
    stack as _jet_stack,
    subtract as _jet_subtract,
    sum_axis as _jet_sum_axis,
    take as _jet_take,
    unary_power as _jet_power,
)
from ._database import D11_POSITIVE_FLOOR, Monoenergetic
from ._interpolators_preprocessed import (
    get_Dij_preprocessed_3d_ntss_radius,
    radial_preprocessed_interpolation_stencil,
    radial_preprocessed_interpolation_table_bar,
)
from ._interpolators import monoenergetic_interpolation_table_bar
from ._monoenergetic import MONOENERGETIC_KIND_GENERIC, monoenergetic_database_kind
from ._monoenergetic_interpolators import monoenergetic_interpolation_kernel
from ._source_models import assemble_pressure_source_components, sum_source_components
from ._model_api import (
    ModelCapabilities,
    ModelValidationContext,
    transport_model as transport_model_decorator,
    validate_transport_flux_builder,
)
from ._transport_debug import lagged_timing_enabled, lagged_timing_start, lagged_timing_end
from ._constants import elementary_charge, epsilon_0, proton_mass
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


@contextlib.contextmanager
def _reverse_rebuild_profile_scope(enabled: bool, name: str):
    """Attach an XProf label without changing the compiled calculation."""
    if enabled:
        with jax.named_scope(name):
            yield
    else:
        yield


def _ntx_local_pullback_finite_debug_enabled() -> bool:
    raw = str(os.environ.get("NEOPAX_TRANSPORT_NTX_LOCAL_PULLBACK_FINITE_DEBUG", "")).strip().lower()
    return raw not in {"", "0", "false", "no", "off"}


def _ntx_nonfinite_debug_enabled() -> bool:
    raw = str(os.environ.get("NEOPAX_NTX_NONFINITE_DEBUG", "")).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    return _ntx_local_pullback_finite_debug_enabled()


def _reverse_tree_debug_enabled() -> bool:
    raw = str(os.environ.get("NEOPAX_REVERSE_TREE_DEBUG", "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _debug_first_bad_flat(value):
    value = jnp.asarray(value)
    flat_bad = jnp.ravel(jnp.logical_not(jnp.isfinite(value)))
    return jnp.argmax(flat_bad)


def _debug_array_stats(label, value):
    value = jnp.asarray(value)
    jax.debug.print(
        "[ntx-nonfinite] {label}: finite={finite} min={min:.6e} max={max:.6e} max_abs={max_abs:.6e} first_bad_flat={first_bad}",
        label=label,
        finite=jnp.all(jnp.isfinite(value)),
        min=jnp.nanmin(value),
        max=jnp.nanmax(value),
        max_abs=jnp.nanmax(jnp.abs(value)),
        first_bad=_debug_first_bad_flat(value),
    )


def _debug_arrays_if_any_nonfinite(prefix, labelled_arrays):
    """Print detailed NTX diagnostics only if at least one array is nonfinite."""

    if not _ntx_nonfinite_debug_enabled():
        return

    arrays = tuple((label, value) for label, value in labelled_arrays if value is not None)
    if not arrays:
        return
    trigger = jnp.asarray(False)
    for _, value in arrays:
        trigger = jnp.logical_or(trigger, jnp.logical_not(jnp.all(jnp.isfinite(jnp.asarray(value)))))

    def _print(_):
        jax.debug.print("[ntx-nonfinite] {prefix}: begin", prefix=prefix)
        for label, value in arrays:
            _debug_array_stats(f"{prefix}.{label}", value)
        jax.debug.print("[ntx-nonfinite] {prefix}: end", prefix=prefix)
        return jnp.asarray(0, dtype=jnp.int32)

    def _skip(_):
        return jnp.asarray(0, dtype=jnp.int32)

    jax.lax.cond(trigger, _print, _skip, operand=None)


def _debug_lagged_response_if_nonfinite(prefix, response):
    if isinstance(response, NTXInterpolatedMomentResponse):
        _debug_arrays_if_any_nonfinite(
            prefix,
            (
                ("reference_er", response.reference_er),
                ("reference_log_nu_star", response.reference_log_nu_star),
                ("reference_transport_moments", response.reference_transport_moments),
                ("dtransport_moments_d_er", response.dtransport_moments_d_er),
                ("dtransport_moments_d_log_nu_star", response.dtransport_moments_d_log_nu_star),
            ),
        )
        return
    if isinstance(response, NTXPreparedCoefficientResponse):
        _debug_arrays_if_any_nonfinite(
            prefix,
            (
                ("reference_transport_moments", response.reference_transport_moments),
                ("reference_nu_hat", response.reference_nu_hat),
                ("reference_epsi_hat", response.reference_epsi_hat),
            ),
        )
        return
    if isinstance(response, NTXQuadraticPreparedCoefficientResponse):
        _debug_arrays_if_any_nonfinite(
            prefix,
            (
                ("reference_nu_hat", response.reference_nu_hat),
                ("reference_epsi_hat", response.reference_epsi_hat),
                ("reference_coefficients", response.reference_coefficients),
                ("dcoefficients_d_nu_hat", response.dcoefficients_d_nu_hat),
                ("dcoefficients_d_epsi_hat", response.dcoefficients_d_epsi_hat),
                ("d2coefficients_d_nu_hat2", response.d2coefficients_d_nu_hat2),
                ("d2coefficients_d_nu_hat_d_epsi_hat", response.d2coefficients_d_nu_hat_d_epsi_hat),
                ("d2coefficients_d_epsi_hat2", response.d2coefficients_d_epsi_hat2),
            ),
        )


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


def compute_net_total_power_volume_average_mw_m3(state, pressure_source_model, geometry):
    """Return the signed volume-averaged total source power density.

    This is an optimization observable, not the positive power scale used by
    turbulence closures.  In particular it deliberately preserves a negative
    net value and therefore must not use ``compute_total_power_mw``'s 3-MW
    fallback.
    """
    dtype = state.density.dtype
    zero = jnp.asarray(0.0, dtype=dtype)
    if pressure_source_model is None or geometry is None:
        return zero
    raw_sources = pressure_source_model(state)
    if not isinstance(raw_sources, dict):
        return zero

    net_power_density = None
    alpha_power = raw_sources.get("AlphaPower")
    if alpha_power is not None:
        net_power_density = jnp.asarray(alpha_power, dtype=dtype)

    pbrems = raw_sources.get("PBrems")
    if pbrems is not None:
        pbrems_arr = jnp.asarray(pbrems, dtype=dtype)
        net_power_density = -pbrems_arr if net_power_density is None else net_power_density - pbrems_arr

    for key in ("heating", "external_heating", "ecrh", "icrh", "nbi", "ohmic_heating"):
        value = raw_sources.get(key)
        if value is None:
            continue
        value_arr = jnp.asarray(value, dtype=dtype)
        net_power_density = value_arr if net_power_density is None else net_power_density + value_arr

    if net_power_density is None:
        return zero
    power_density_mw_m3 = PRESSURE_SOURCE_STATE_TO_MW_M3 * net_power_density
    volume = jnp.trapezoid(jnp.asarray(geometry.Vprime, dtype=dtype), x=jnp.asarray(geometry.r_grid, dtype=dtype))
    integral = jnp.trapezoid(
        power_density_mw_m3 * jnp.asarray(geometry.Vprime, dtype=dtype),
        x=jnp.asarray(geometry.r_grid, dtype=dtype),
    )
    return integral / jnp.maximum(volume, jnp.asarray(1.0e-30, dtype=dtype))


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

        def evaluate_with_lagged_response_tangent(
                self, state, state_direction, lagged_response, **kwargs
        ):
                """Optional forward tangent of a cached response.

                Ordinary linear-response models use their established JAX
                forward rule here.  Realtime quadratic NTX overrides this
                method with its explicit factorized-Hessian implementation;
                the normal value-only solver never calls this API.
                """
                return jax.jvp(
                        lambda state_value: self.evaluate_with_lagged_response(
                                state_value, lagged_response, **kwargs
                        ),
                        (state,),
                        (state_direction,),
                )[1]

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
class FaceJVPTransportFluxResponse:
        reference_state: Any
        reference_face_flux: dict
        # Some models (notably the NTX database model) have a distinct direct
        # centre primitive in addition to their face primitive.  Keep its
        # anchor so a lagged response can preserve that representation.
        reference_flux: dict | None = None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class NTXPreparedCoefficientResponse:
    reference_transport_moments: jax.Array
    reference_nu_hat: jax.Array
    reference_epsi_hat: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class NTXQuadraticPreparedCoefficientResponse:
    """Energy-resolved realtime NTX coefficient Taylor data.

    ``nu_hat`` and ``epsi_hat`` vary over the energy grid, so their Taylor
    fields must remain energy resolved until the current displacement has
    been applied.  Reducing to transport moments before that contraction
    would incorrectly treat all energy-point displacements as one scalar.
    """

    reference_nu_hat: jax.Array
    reference_epsi_hat: jax.Array
    reference_coefficients: jax.Array
    dcoefficients_d_nu_hat: jax.Array
    dcoefficients_d_epsi_hat: jax.Array
    d2coefficients_d_nu_hat2: jax.Array
    d2coefficients_d_nu_hat_d_epsi_hat: jax.Array
    d2coefficients_d_epsi_hat2: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class NTXFullStateQuadraticPreparedCoefficientResponse:
    """Forward-only full transport-state wrapper around NTX coefficient data.

    The nested coefficient response remains exactly the factorized NTX
    Hessian payload.  ``reference_state`` supplies the missing state anchor
    needed to evaluate its explicit full-state Taylor composition at a Radau
    stage.  This separate type prevents the established coefficient-only
    quadratic response from silently changing meaning.
    """

    reference_state: TransportState
    coefficient_response: NTXQuadraticPreparedCoefficientResponse


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
class _NTXInterpolatedMomentCoefficientRecord:
    """Compact local NTX primitives for an opt-in segment record.

    The three arrays are emitted by the same base/JVP local response
    evaluation that produces :class:`NTXInterpolatedMomentResponse`.  They
    deliberately contain coefficient values only, never an NTX factorisation.
    This private record is not part of the ordinary lagged-response cache.
    """

    coefficient_scan: jax.Array
    dcoefficient_scan_d_er: jax.Array
    dcoefficient_scan_d_log_nu_star: jax.Array


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
        face_response: Any = None
        center_response: Any = None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class _NTXExactLijLaggedResponseCoefficientRecord:
    """Compact anchor coefficients emitted only by the opt-in record builder.

    The leading axis is the response-anchor axis; this object is intentionally
    separate from :class:`NTXExactLijLaggedResponse` so ordinary carry caches
    and all established reverse modes retain their current pytree layout.
    """

    face_anchor_coefficients: _NTXInterpolatedMomentCoefficientRecord


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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class EvaluatedTransportState:
    """Canonical finite-volume values and gradients used by flux models."""

    center: TransportState
    face: FaceTransportState
    density_grad_center: jax.Array
    temperature_grad_center: jax.Array
    Er_grad_center: jax.Array
    density_grad_face: jax.Array
    temperature_grad_face: jax.Array
    Er_grad_face: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class _DirectionalTransportState:
    """Fixed-anchor second-order transport-state response along one stage direction."""

    density: DirectionalSecondOrderJet
    pressure: DirectionalSecondOrderJet
    Er: DirectionalSecondOrderJet

    @property
    def temperature(self):
        return _jet_divide(self.pressure, self.density)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class _DirectionalEvaluatedTransportState:
    """Directional counterpart to :class:`EvaluatedTransportState`."""

    center: _DirectionalTransportState
    face: _DirectionalTransportState
    density_grad_center: DirectionalSecondOrderJet
    temperature_grad_center: DirectionalSecondOrderJet
    Er_grad_center: DirectionalSecondOrderJet
    density_grad_face: DirectionalSecondOrderJet
    temperature_grad_face: DirectionalSecondOrderJet
    Er_grad_face: DirectionalSecondOrderJet


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


def _extrapolated_right_face_value(state_arr: jax.Array) -> jax.Array:
    arr = jnp.asarray(state_arr)
    if arr.ndim == 1:
        if arr.shape[0] >= 2:
            return 1.5 * arr[-1] - 0.5 * arr[-2]
        return arr[-1]
    if arr.shape[-1] >= 2:
        return 1.5 * arr[:, -1] - 0.5 * arr[:, -2]
    return arr[:, -1]


def _extract_right_constraints(
    bc_model: Any,
    state_arr: jax.Array,
    face_centers: jax.Array | None = None,
) -> tuple[jax.Array | None, jax.Array | None]:
    n_species = state_arr.shape[0]
    default_value = _extrapolated_right_face_value(state_arr)
    default_grad = jnp.zeros_like(default_value)
    if bc_model is None:
        return default_value, default_grad

    if face_centers is not None and hasattr(bc_model, "right_type"):
        return right_constraints_from_bc_model(
            bc_model,
            default_value,
            profile=state_arr,
            face_centers=face_centers,
        )

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
    if bc_model is None:
        default_right = _extrapolated_right_face_value(state_arr)
        return None, jnp.zeros_like(state_arr[:, 0]), default_right, None

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


def _homogeneous_directional_bc_model(bc_model):
    """Return the tangent BC model for a fixed-anchor face response.

    Explicit Dirichlet values and Neumann gradients are external constants and
    hence have zero directional response.  A missing Dirichlet value
    deliberately remains ``None``: in that case the ordinary FV rule uses the
    endpoint profile, so it is state dependent. Robin has no additive external
    datum in the implemented FV constraint and remains homogeneous.
    """
    if bc_model is None:
        return None
    changes = {}
    for side in ("left", "right"):
        value_name = f"{side}_value"
        gradient_name = f"{side}_gradient"
        boundary_type = str(getattr(bc_model, f"{side}_type", "dirichlet")).strip().lower()
        if boundary_type == "dirichlet":
            boundary_value = getattr(bc_model, value_name, None)
            if boundary_value is not None:
                changes[value_name] = jnp.zeros_like(jnp.asarray(boundary_value))
        elif boundary_type == "neumann":
            boundary_gradient = getattr(bc_model, gradient_name, None)
            if boundary_gradient is not None:
                changes[gradient_name] = jnp.zeros_like(jnp.asarray(boundary_gradient))
    return dataclasses.replace(bc_model, **changes) if changes else bc_model


def _face_profile_directional(
    profile: DirectionalSecondOrderJet,
    face_centers,
    *,
    bc_model=None,
    reconstruction: str = "linear",
) -> DirectionalSecondOrderJet:
    """Propagate a profile jet through the fixed-branch linear FV face map."""
    if reconstruction != "linear":
        raise NotImplementedError(
            "Full-state quadratic realtime NTX responses currently require "
            "linear face reconstruction; WENO has a state-dependent stencil."
        )
    tangent_bc = _homogeneous_directional_bc_model(bc_model)
    return DirectionalSecondOrderJet(
        _face_profile(profile.value, face_centers, bc_model=bc_model, reconstruction=reconstruction),
        _face_profile(profile.first, face_centers, bc_model=tangent_bc, reconstruction=reconstruction),
        _face_profile(profile.second, face_centers, bc_model=tangent_bc, reconstruction=reconstruction),
    )


def _face_profile_gradient_directional(
    profile: DirectionalSecondOrderJet,
    face_centers,
    *,
    bc_model=None,
) -> DirectionalSecondOrderJet:
    """Propagate a profile jet through the fixed-branch FV face-gradient map."""
    tangent_bc = _homogeneous_directional_bc_model(bc_model)
    return DirectionalSecondOrderJet(
        _face_profile_gradient(profile.value, face_centers, bc_model=bc_model),
        _face_profile_gradient(profile.first, face_centers, bc_model=tangent_bc),
        _face_profile_gradient(profile.second, face_centers, bc_model=tangent_bc),
    )


def _center_profile_gradient(profile, face_centers, bc_model=None):
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
        ).grad()
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


def build_evaluated_transport_state(
    state: TransportState,
    geometry: Any,
    *,
    bc_density: Any = None,
    bc_temperature: Any = None,
    bc_er: Any = None,
    reconstruction: str = "linear",
    density_floor: Any = DEFAULT_TRANSPORT_DENSITY_FLOOR,
    temperature_floor: Any = DEFAULT_TRANSPORT_TEMPERATURE_FLOOR,
) -> EvaluatedTransportState:
    center_state = apply_transport_density_floor(state, density_floor)
    center_state = apply_transport_temperature_floor(center_state, temperature_floor, density_floor)
    face_state = build_face_transport_state(
        center_state,
        geometry,
        bc_density=bc_density,
        bc_temperature=bc_temperature,
        bc_er=bc_er,
        reconstruction=reconstruction,
        density_floor=density_floor,
        temperature_floor=temperature_floor,
    )
    density_center = safe_density(center_state.density, density_floor)
    temperature_center = safe_temperature(center_state.temperature, temperature_floor)
    density_face = safe_density(face_state.density, density_floor)
    temperature_face = safe_temperature(face_state.temperature, temperature_floor)
    face_state = dataclasses.replace(face_state, density=density_face, pressure=density_face * temperature_face)

    return EvaluatedTransportState(
        center=center_state,
        face=face_state,
        density_grad_center=_center_profile_gradient(
            density_center,
            geometry.r_grid_half,
            bc_model=bc_density,
        ),
        temperature_grad_center=_center_profile_gradient(
            temperature_center,
            geometry.r_grid_half,
            bc_model=bc_temperature,
        ),
        Er_grad_center=_center_profile_gradient(
            center_state.Er,
            geometry.r_grid_half,
            bc_model=bc_er,
        ),
        density_grad_face=_face_profile_gradient(
            density_center,
            geometry.r_grid_half,
            bc_model=bc_density,
        ),
        temperature_grad_face=_face_profile_gradient(
            temperature_center,
            geometry.r_grid_half,
            bc_model=bc_temperature,
        ),
        Er_grad_face=_face_profile_gradient(
            center_state.Er,
            geometry.r_grid_half,
            bc_model=bc_er,
        ),
    )


def _center_profile_gradient_directional(
    profile: DirectionalSecondOrderJet,
    face_centers,
    *,
    bc_model=None,
) -> DirectionalSecondOrderJet:
    tangent_bc = _homogeneous_directional_bc_model(bc_model)
    return DirectionalSecondOrderJet(
        _center_profile_gradient(profile.value, face_centers, bc_model=bc_model),
        _center_profile_gradient(profile.first, face_centers, bc_model=tangent_bc),
        _center_profile_gradient(profile.second, face_centers, bc_model=tangent_bc),
    )


def _jet_vthermal_from_temperature(
    reference_vthermal: jax.Array,
    temperature: DirectionalSecondOrderJet,
) -> DirectionalSecondOrderJet:
    """Explicit thermal-speed response, retaining the normal-model anchor."""
    normalized_temperature = _jet_divide(temperature, temperature.value)
    return _jet_multiply(jnp.asarray(reference_vthermal), _jet_power(normalized_temperature, 0.5))


def _nu_over_vnew_local_directional_default(
    species,
    species_index: int,
    v_new: DirectionalSecondOrderJet,
    density_local: DirectionalSecondOrderJet,
    temperature_local: DirectionalSecondOrderJet,
    vthermal_local: DirectionalSecondOrderJet,
) -> DirectionalSecondOrderJet:
    """Default local collision response written through second directional order.

    This mirrors ``collisionality_local(..., COULOMB_LOG_MODEL_DEFAULT) /
    v_new``.  It is deliberately explicit: no generic JVP/VJP traces through
    the collision or NTX code are created.
    """
    electron_temperature = _jet_select_axis(temperature_local, 0)
    electron_density = _jet_select_axis(density_local, 0)
    coulomb_log = _jet_add(
        32.2,
        _jet_multiply(
            1.15,
            _jet_log10(
                _jet_divide(
                    _jet_multiply(1.0e6, _jet_multiply(electron_temperature, electron_temperature)),
                    _jet_multiply(1.0e20, electron_density),
                )
            ),
        ),
    )
    test_charge = species.charge[species_index]
    test_mass = species.mass[species_index]
    collision_sum = None
    for background_index in range(int(species.number_species)):
        background_density = _jet_select_axis(density_local, background_index)
        background_vthermal = _jet_select_axis(vthermal_local, background_index)
        gamma_prefactor = (
            test_charge**2
            * species.charge[background_index] ** 2
            / (4.0 * jnp.pi * epsilon_0**2 * test_mass**2)
        )
        x = _jet_divide(v_new, background_vthermal)
        chandrasekhar = _jet_divide(
            _jet_subtract(
                _jet_erf(x),
                _jet_multiply(
                    2.0 / jnp.sqrt(jnp.pi),
                    _jet_multiply(x, _jet_exp(_jet_multiply(-1.0, _jet_multiply(x, x)))),
                ),
            ),
            _jet_multiply(2.0, _jet_multiply(x, x)),
        )
        pair = _jet_multiply(
            _jet_multiply(gamma_prefactor, coulomb_log),
            _jet_multiply(
                _jet_divide(_jet_multiply(1.0e20, background_density), _jet_power(v_new, 3.0)),
                _jet_subtract(_jet_erf(x), chandrasekhar),
            ),
        )
        collision_sum = pair if collision_sum is None else _jet_add(collision_sum, pair)
    return _jet_divide(collision_sum, v_new)


def _local_scan_inputs_directional_default(
    energy_grid,
    species,
    *,
    drds_value,
    species_index: int,
    er_value: DirectionalSecondOrderJet,
    temperature_local: DirectionalSecondOrderJet,
    density_local: DirectionalSecondOrderJet,
    reference_vthermal_local: jax.Array,
    er_v_floor: float | None,
) -> tuple[DirectionalSecondOrderJet, DirectionalSecondOrderJet, DirectionalSecondOrderJet]:
    """Custom full-state response of the realtime NTX local coordinates."""
    vthermal_local = _jet_vthermal_from_temperature(reference_vthermal_local, temperature_local)
    vth_a = _jet_select_axis(vthermal_local, species_index)
    v_new_a = _jet_multiply(jnp.asarray(energy_grid.v_norm), vth_a)
    finite_drds = jnp.isfinite(drds_value)
    safe_drds = jnp.where(finite_drds, drds_value, jnp.asarray(0.0, dtype=vth_a.value.dtype))
    epsi_hat = _jet_divide(_jet_multiply(1.0e3 * safe_drds, er_value), v_new_a)
    if er_v_floor is not None:
        sign = jnp.where(epsi_hat.value < 0.0, -1.0, 1.0)
        epsi_abs = maximum_with_constant_floor(_jet_abs(epsi_hat), er_v_floor)
        epsi_hat = _jet_multiply(sign, epsi_abs)
    nu_hat = _nu_over_vnew_local_directional_default(
        species, species_index, v_new_a, density_local, temperature_local, vthermal_local
    )
    return nu_hat, epsi_hat, vth_a


def _transport_moments_from_coefficient_scan_directional(
    energy_grid,
    coefficient_scan: DirectionalSecondOrderJet,
    *,
    drds_value,
) -> DirectionalSecondOrderJet:
    """Explicit coefficient-to-moment reduction on the frozen D11 branch."""
    d11 = maximum_with_constant_floor(
        _jet_multiply(_jet_take(coefficient_scan, 0, axis=1), drds_value**2),
        D11_POSITIVE_FLOOR,
    )
    d11 = _jet_multiply(-1.0, d11)
    d13 = _jet_multiply(-drds_value, _jet_take(coefficient_scan, 2, axis=1))
    d33 = _jet_multiply(-1.0, _jet_take(coefficient_scan, 3, axis=1))
    weights = energy_grid.xWeights
    return _jet_stack(
        (
            _jet_sum_axis(_jet_multiply(energy_grid.L11_weight * weights, d11)),
            _jet_sum_axis(_jet_multiply(energy_grid.L12_weight * weights, d11)),
            _jet_sum_axis(_jet_multiply(energy_grid.L22_weight * weights, d11)),
            _jet_sum_axis(_jet_multiply(energy_grid.L13_weight * weights, d13)),
            _jet_sum_axis(_jet_multiply(energy_grid.L23_weight * weights, d13)),
            _jet_sum_axis(_jet_multiply(energy_grid.L33_weight * weights, d33)),
        ),
        axis=0,
    )


def _lij_from_transport_moments_directional(
    species,
    transport_moments: DirectionalSecondOrderJet,
    *,
    species_index: int,
    vth_a: DirectionalSecondOrderJet,
) -> DirectionalSecondOrderJet:
    """Written second-order counterpart of ``_lij_from_transport_moments``."""
    charge, mass = species.charge[species_index], species.mass[species_index]
    inv_sqrt_pi = 1.0 / jnp.sqrt(jnp.pi)
    l11_factor = _jet_multiply(-inv_sqrt_pi * (mass / charge) ** 2, _jet_power(vth_a, 3.0))
    l13_factor = _jet_multiply(-inv_sqrt_pi * (mass / charge), _jet_power(vth_a, 2.0))
    l33_factor = _jet_multiply(-inv_sqrt_pi, vth_a)
    l00, l01, l11 = (_jet_multiply(l11_factor, _jet_take(transport_moments, index)) for index in (0, 1, 2))
    l02, l12 = (_jet_multiply(l13_factor, _jet_take(transport_moments, index)) for index in (3, 4))
    l22 = _jet_multiply(l33_factor, _jet_take(transport_moments, 5))
    return _jet_stack((_jet_stack((l00, l01, l02), axis=-1), _jet_stack((l01, l11, l12), axis=-1), _jet_stack((_jet_negate(l02), _jet_negate(l12), l22), axis=-1)), axis=-2)


def _directional_lij_entry(lij: DirectionalSecondOrderJet, row: int, column: int) -> DirectionalSecondOrderJet:
    return _jet_take(_jet_take(lij, row, axis=0), column, axis=0)


def _assemble_face_fluxes_from_lij_directional_local(
    *,
    charge,
    density: DirectionalSecondOrderJet,
    temperature: DirectionalSecondOrderJet,
    density_gradient: DirectionalSecondOrderJet,
    temperature_gradient: DirectionalSecondOrderJet,
    er: DirectionalSecondOrderJet,
    lij: DirectionalSecondOrderJet,
) -> tuple[DirectionalSecondOrderJet, DirectionalSecondOrderJet, DirectionalSecondOrderJet]:
    """Second-order directional counterpart of one-species face flux assembly."""
    a1 = _jet_subtract(
        _jet_subtract(
            _jet_divide(density_gradient, density),
            _jet_multiply(1.5, _jet_divide(temperature_gradient, temperature)),
        ),
        _jet_divide(_jet_multiply(charge, er), _jet_multiply(elementary_charge, temperature)),
    )
    a2 = _jet_divide(temperature_gradient, temperature)
    gamma = _jet_multiply(
        _jet_multiply(-DENSITY_STATE_TO_PHYSICAL, density),
        _jet_add(
            _jet_multiply(_directional_lij_entry(lij, 0, 0), a1),
            _jet_multiply(_directional_lij_entry(lij, 0, 1), a2),
        ),
    )
    q = _jet_multiply(
        _jet_multiply(-DENSITY_STATE_TO_PHYSICAL * TEMPERATURE_STATE_TO_PHYSICAL, _jet_multiply(temperature, density)),
        _jet_add(
            _jet_multiply(_directional_lij_entry(lij, 1, 0), a1),
            _jet_multiply(_directional_lij_entry(lij, 1, 1), a2),
        ),
    )
    upar = _jet_multiply(
        _jet_multiply(-DENSITY_STATE_TO_PHYSICAL, density),
        _jet_add(
            _jet_multiply(_directional_lij_entry(lij, 2, 0), a1),
            _jet_multiply(_directional_lij_entry(lij, 2, 1), a2),
        ),
    )
    return gamma, q, upar


def _build_evaluated_transport_state_directional(
    state: TransportState,
    state_direction: TransportState,
    geometry: Any,
    *,
    bc_density: Any = None,
    bc_temperature: Any = None,
    bc_er: Any = None,
    reconstruction: str = "linear",
    density_floor: Any = DEFAULT_TRANSPORT_DENSITY_FLOOR,
    temperature_floor: Any = DEFAULT_TRANSPORT_TEMPERATURE_FLOOR,
) -> _DirectionalEvaluatedTransportState:
    """Explicit fixed-anchor state response through the FV evaluation layer."""
    center = _DirectionalTransportState(
        density=maximum_with_constant_floor(_jet_seed(state.density, state_direction.density), density_floor),
        pressure=_jet_seed(state.pressure, state_direction.pressure),
        Er=_jet_seed(state.Er, state_direction.Er),
    )
    if temperature_floor is not None:
        center = dataclasses.replace(
            center,
            pressure=_jet_multiply(
                center.density,
                maximum_with_constant_floor(center.temperature, temperature_floor),
            ),
        )
    density_face = maximum_with_constant_floor(
        _face_profile_directional(center.density, geometry.r_grid_half, bc_model=bc_density, reconstruction=reconstruction),
        density_floor,
    )
    temperature_face = _face_profile_directional(
        center.temperature, geometry.r_grid_half, bc_model=bc_temperature, reconstruction=reconstruction
    )
    if temperature_floor is not None:
        temperature_face = maximum_with_constant_floor(temperature_face, temperature_floor)
    face = _DirectionalTransportState(
        density=density_face,
        pressure=_jet_multiply(density_face, temperature_face),
        Er=_face_profile_directional(center.Er, geometry.r_grid_half, bc_model=bc_er, reconstruction=reconstruction),
    )
    return _DirectionalEvaluatedTransportState(
        center=center,
        face=face,
        density_grad_center=_center_profile_gradient_directional(center.density, geometry.r_grid_half, bc_model=bc_density),
        temperature_grad_center=_center_profile_gradient_directional(center.temperature, geometry.r_grid_half, bc_model=bc_temperature),
        Er_grad_center=_center_profile_gradient_directional(center.Er, geometry.r_grid_half, bc_model=bc_er),
        density_grad_face=_face_profile_gradient_directional(center.density, geometry.r_grid_half, bc_model=bc_density),
        temperature_grad_face=_face_profile_gradient_directional(center.temperature, geometry.r_grid_half, bc_model=bc_temperature),
        Er_grad_face=_face_profile_gradient_directional(center.Er, geometry.r_grid_half, bc_model=bc_er),
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
        right_value = _extrapolated_right_face_value(profile_2d)
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
    geometry: Any = None
    center_flux_mode: str = "direct"

    def __post_init__(self):
        mode = str(self.center_flux_mode).strip().lower()
        aliases = {
            "default": "direct",
            "direct_center": "direct",
            "interpolate": "interpolate_from_faces",
            "interpolate_faces": "interpolate_from_faces",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"direct", "interpolate_from_faces"}:
            raise ValueError(
                "center_flux_mode must be one of: direct, interpolate_from_faces"
            )
        object.__setattr__(self, "center_flux_mode", mode)

    @staticmethod
    def _zero_like_flux(reference, fallback=0):
        if reference is not None:
            return jnp.zeros_like(jnp.asarray(reference))
        return fallback

    @staticmethod
    def _has_complete_face_fluxes(fluxes):
        """Whether one model supplied the complete face representation."""

        return all(
            f"{name}_faces" in fluxes and fluxes[f"{name}_faces"] is not None
            for name in ("Gamma", "Q", "Upar")
        )

    @classmethod
    def _can_contribute_face_fluxes(cls, model, fluxes):
        """A zero model is the sole valid exception to a complete face payload.

        A partial payload must never be interpreted as zero contribution from
        the missing model.  In particular, a centre-only cached neoclassical
        response combined with face-based turbulence used to manufacture
        ``*_faces`` fields with the neoclassical part silently set to zero.
        That bypassed the equation-level face fallback and dropped the
        neoclassical conservative divergence.
        """

        return isinstance(model, ZeroTransportModel) or cls._has_complete_face_fluxes(fluxes)

    def _canonical_face_fluxes(self, state):
        """Evaluate total face fluxes under one stable public key convention."""

        if self.geometry is None:
            raise ValueError(
                "center_flux_mode='interpolate_from_faces' requires the composite "
                "transport flux model to carry its geometry."
            )
        face_state = build_face_transport_state(
            state,
            self.geometry,
            bc_density=getattr(self.neoclassical_model, "bc_density", None),
            bc_temperature=getattr(self.neoclassical_model, "bc_temperature", None),
        )
        raw_face_fluxes = self.evaluate_face_fluxes(state, face_state)
        if raw_face_fluxes is None:
            raise ValueError(
                "center_flux_mode='interpolate_from_faces' requires all active "
                "transport flux models to provide face fluxes."
            )

        def _face_value(name):
            value = raw_face_fluxes.get(f"{name}_faces", raw_face_fluxes.get(name))
            if value is None:
                raise ValueError(
                    "center_flux_mode='interpolate_from_faces' requires face "
                    f"flux '{name}'."
                )
            return value

        return {name: _face_value(name) for name in ("Gamma", "Q", "Upar")}

    @staticmethod
    def _centres_from_faces(face_fluxes):
        return {
            name: jax.vmap(cell_centered_from_faces)(face_fluxes[name])
            for name in ("Gamma", "Q", "Upar")
        }

    def _apply_center_flux_mode(self, fluxes, face_fluxes):
        """Apply the universal centre representation without changing faces."""

        if self.center_flux_mode == "direct":
            return fluxes
        if face_fluxes is None:
            raise ValueError(
                "center_flux_mode='interpolate_from_faces' requires lagged face fluxes."
            )
        centres = self._centres_from_faces(face_fluxes)
        out = dict(fluxes)
        out.update(centres)
        out.update({f"{name}_faces": value for name, value in face_fluxes.items()})
        return out

    @staticmethod
    def _centre_bar_to_face_bar(centre_bar):
        """Transpose ``cell_centered_from_faces`` for a species-profile bar."""

        centre_bar = jnp.asarray(centre_bar)
        face_template = jnp.zeros(
            centre_bar.shape[:-1] + (centre_bar.shape[-1] + 1,),
            dtype=centre_bar.dtype,
        )
        _, pullback = jax.vjp(
            lambda faces: jax.vmap(cell_centered_from_faces)(faces),
            face_template,
        )
        return pullback(centre_bar)[0]

    def _apply_center_flux_mode_pullback(self, flux_bar):
        """Map universal interpolated-centre cotangents to face cotangents."""

        if self.center_flux_mode == "direct":
            return flux_bar
        out = dict(flux_bar)
        for name in ("Gamma", "Q", "Upar"):
            centre_bar = out.get(name, None)
            if centre_bar is None:
                continue
            centre_bar = jnp.asarray(centre_bar)
            if centre_bar.ndim == 0 or centre_bar.dtype == jax.dtypes.float0:
                continue
            face_name = f"{name}_faces"
            face_bar = self._centre_bar_to_face_bar(centre_bar)
            existing_face_bar = out.get(face_name, None)
            if existing_face_bar is not None:
                existing_face_bar = jnp.asarray(existing_face_bar)
                if existing_face_bar.ndim != 0 and existing_face_bar.dtype != jax.dtypes.float0:
                    face_bar = face_bar + existing_face_bar
            out[name] = jnp.zeros_like(centre_bar)
            out[face_name] = face_bar
        return out

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
        if self.center_flux_mode == "direct":
            return out

        # Keep the component face fluxes as well as the total face fluxes.
        # The temperature equation uses (for example) ``Gamma_neo_faces`` to
        # construct the convective heat flux.  Retaining only total
        # ``Gamma_faces`` here made that equation fall back to the 51-point
        # centre ``Gamma_neo`` while its temperature states were on 52 faces.
        # The lagged path already preserves these keys; direct/black-box mode
        # must expose the same representation.
        face_state = build_face_transport_state(
            state,
            self.geometry,
            bc_density=getattr(self.neoclassical_model, "bc_density", None),
            bc_temperature=getattr(self.neoclassical_model, "bc_temperature", None),
        )
        raw_face_fluxes = self.evaluate_face_fluxes(state, face_state)
        if raw_face_fluxes is None:
            raise ValueError(
                "center_flux_mode='interpolate_from_faces' requires all active "
                "transport flux models to provide face fluxes."
            )
        face_fluxes = {
            name: raw_face_fluxes[name]
            for name in ("Gamma", "Q", "Upar")
        }
        out = self._apply_center_flux_mode(out, face_fluxes)
        out.update({
            name: value
            for name, value in raw_face_fluxes.items()
            if name.endswith("_faces")
        })
        # ``evaluate_face_fluxes`` historically returns direct model faces
        # under the unsuffixed component names.  Give those component values
        # an unambiguous face key in the public composite output.
        for name in (
            "Gamma_neo", "Q_neo", "Upar_neo",
            "Gamma_turb", "Q_turb", "Upar_turb",
            "Gamma_classical", "Q_classical", "Upar_classical",
        ):
            face_value = raw_face_fluxes.get(f"{name}_faces", raw_face_fluxes.get(name))
            if face_value is not None:
                out[f"{name}_faces"] = face_value
        return out

    def pullback_direct_rhs_state(self, state, flux_bar):
        """Split direct black-box flux transpose by model ownership.

        Database-backed neoclassical fluxes use their dedicated local state
        transpose. Optional turbulent/classical contributions retain separate
        small flux-model VJPs, never a VJP of the composed transport RHS.
        """
        if self.center_flux_mode != "direct":
            return None
        neo_pullback = getattr(self.neoclassical_model, "pullback_direct_rhs_state", None)
        if not callable(neo_pullback):
            return None
        zero = jnp.zeros_like(jnp.asarray(state.density))

        def _bar(name):
            value = flux_bar.get(name, None)
            if value is None:
                return zero
            value = jnp.asarray(value)
            return zero if value.ndim == 0 or value.dtype == jax.dtypes.float0 else value

        state_bar = neo_pullback(state, {
            "Gamma": _bar("Gamma") + _bar("Gamma_neo"),
            "Q": _bar("Q") + _bar("Q_neo"),
            "Upar": _bar("Upar") + _bar("Upar_neo"),
        })

        def _add(lhs, rhs):
            return jax.tree_util.tree_map(lambda a, b: a + b, lhs, rhs)

        for model, suffix in (
            (self.turbulent_model, "turb"),
            (self.classical_model, "classical"),
        ):
            if isinstance(model, ZeroTransportModel):
                continue
            _, pullback = jax.vjp(lambda state_value: model(state_value), state)
            (model_state_bar,) = pullback({
                "Gamma": _bar("Gamma") + _bar(f"Gamma_{suffix}"),
                "Q": _bar("Q") + _bar(f"Q_{suffix}"),
                "Upar": _bar("Upar") + _bar(f"Upar_{suffix}"),
            })
            state_bar = _add(state_bar, model_state_bar)
        return state_bar

    def pullback_direct_rhs_geometry_by_radius(self, state, flux_bar, geometry):
        """Split direct-flux geometry bars without tracing the database table.

        The neoclassical database branch uses its compact local rule.  Any
        optional non-neoclassical model retains a small model-local VJP so the
        composite contract remains exact when, for example, turbulence owns a
        geometry-dependent coefficient.
        """
        if self.center_flux_mode != "direct":
            return None
        neo_pullback = getattr(
            self.neoclassical_model,
            "pullback_direct_rhs_geometry_by_radius",
            None,
        )
        if not callable(neo_pullback):
            return None
        zero = jnp.zeros_like(jnp.asarray(state.density))

        def _bar(name):
            value = flux_bar.get(name, None)
            if value is None:
                return zero
            value = jnp.asarray(value)
            return zero if value.ndim == 0 or value.dtype == jax.dtypes.float0 else value

        geometry_bar = neo_pullback(
            state,
            {
                "Gamma": _bar("Gamma") + _bar("Gamma_neo"),
                "Q": _bar("Q") + _bar("Q_neo"),
                "Upar": _bar("Upar") + _bar("Upar_neo"),
            },
            geometry,
        )
        geometry_delta0 = _float_delta_tree_like(geometry)
        for model, suffix in (
            (self.turbulent_model, "turb"),
            (self.classical_model, "classical"),
        ):
            if isinstance(model, ZeroTransportModel):
                continue
            field_name = (
                "geometry" if hasattr(model, "geometry") else
                "field" if hasattr(model, "field") else None
            )
            if field_name is None:
                continue

            def _model_fluxes(geometry_delta, model_value=model, name=field_name):
                return dataclasses.replace(
                    model_value,
                    **{name: _add_float_delta_tree(geometry, geometry_delta)},
                )(state)

            _, pullback = jax.vjp(_model_fluxes, geometry_delta0)
            (model_bar,) = pullback(
                {
                    "Gamma": _bar("Gamma") + _bar(f"Gamma_{suffix}"),
                    "Q": _bar("Q") + _bar(f"Q_{suffix}"),
                    "Upar": _bar("Upar") + _bar(f"Upar_{suffix}"),
                }
            )
            geometry_bar = _add_float_delta_tree(geometry_bar, model_bar)
        return geometry_bar

    def pullback_direct_rhs_support_payload(self, state, flux_bar, support):
        """Return the neoclassical direct-support bar for a black-box RHS.

        The recorded scan-database payload is a true black-box path: the
        forward RHS evaluates ``composite(state)`` directly from the database,
        so its transpose must do the same.  In particular, do *not* convert
        the flux bar through a lagged-response object here.  The latter is a
        separate, retained route used only by ``radau_rhs_mode=lagged_*``.

        Older live NTX payloads have no explicit ``database`` leaf.  Preserve
        their established lagged-response bridge below.
        """
        pullback_fn = getattr(self.neoclassical_model, "pullback_direct_rhs_support_payload", None)
        if not callable(pullback_fn):
            return None
        replace_payload = getattr(self.neoclassical_model, "with_support_payload", None)
        if not callable(replace_payload):
            return None
        payload_model = replace_payload(support)
        composite = dataclasses.replace(self, neoclassical_model=payload_model)

        if isinstance(support, dict) and "database" in support:
            # The compact database rule currently covers the direct centre
            # representation.  Face-interpolated composites have an
            # additional face interpolation transpose and retain the generic
            # compatibility route until their equivalent compact rule lands.
            if self.center_flux_mode == "direct":
                zero = jnp.zeros_like(jnp.asarray(state.density))

                def _bar(name):
                    value = flux_bar.get(name, None)
                    if value is None:
                        return zero
                    value = jnp.asarray(value)
                    return zero if value.ndim == 0 or value.dtype == jax.dtypes.float0 else value

                neo_flux_bar = {
                    "Gamma": _bar("Gamma") + _bar("Gamma_neo"),
                    "Q": _bar("Q") + _bar("Q_neo"),
                    "Upar": _bar("Upar") + _bar("Upar_neo"),
                }
                direct_support_bar = pullback_fn(state, neo_flux_bar, support)
                if direct_support_bar is not None:
                    return direct_support_bar

            database = support["database"]
            # Compatibility fallback for non-direct centre representations.
            def _direct_fluxes_from_database(database_value):
                direct_model = dataclasses.replace(payload_model, database=database_value)
                return dataclasses.replace(self, neoclassical_model=direct_model)(state)

            _, database_pullback = jax.vjp(_direct_fluxes_from_database, database)
            (database_bar,) = database_pullback(flux_bar)
            support_bar = dict(_float_delta_tree_like(support))
            support_bar["database"] = _sanitize_float_delta_bar_tree(database, database_bar)
            return support_bar

        response = composite.build_lagged_response(state)
        response_bar = composite.pullback_evaluate_with_lagged_response(
            state, response, flux_bar
        )
        return pullback_fn(state, response_bar.neoclassical_response, support)

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
        has_face_fluxes = all(
            self._can_contribute_face_fluxes(model, fluxes)
            for model, fluxes in (
                (self.neoclassical_model, neo),
                (self.turbulent_model, turb),
                (self.classical_model, classical),
            )
        )
        has_center_fluxes = all(
            "Gamma" in fluxes and "Q" in fluxes and "Upar" in fluxes
            for fluxes in (neo, turb, classical)
        )
        out = {}
        if has_center_fluxes:
            gamma_turb = (
                turb.get("Gamma", 0)
                if self.include_turbulent_particle_flux
                else self._zero_like_flux(
                    turb.get("Gamma", None),
                    self._zero_like_flux(neo.get("Gamma", None), self._zero_like_flux(classical.get("Gamma", None), 0)),
                )
            )
            out.update(
                {
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
            )
        if has_face_fluxes:
            gamma_turb_faces = (
                turb.get("Gamma_faces", 0)
                if self.include_turbulent_particle_flux
                else self._zero_like_flux(
                    turb.get("Gamma_faces", None),
                    self._zero_like_flux(
                        neo.get("Gamma_faces", None),
                        self._zero_like_flux(classical.get("Gamma_faces", None), 0),
                    ),
                )
            )
            out.update(
                {
                    "Gamma_faces": neo.get("Gamma_faces", 0) + gamma_turb_faces + classical.get("Gamma_faces", 0),
                    "Q_faces": neo.get("Q_faces", 0) + turb.get("Q_faces", 0) + classical.get("Q_faces", 0),
                    "Upar_faces": neo.get("Upar_faces", 0) + turb.get("Upar_faces", 0) + classical.get("Upar_faces", 0),
                    "Gamma_neo_faces": neo.get("Gamma_faces", 0),
                    "Q_neo_faces": neo.get("Q_faces", 0),
                    "Upar_neo_faces": neo.get("Upar_faces", 0),
                    "Gamma_turb_faces": gamma_turb_faces,
                    "Q_turb_faces": turb.get("Q_faces", 0),
                    "Upar_turb_faces": turb.get("Upar_faces", 0),
                    "Gamma_classical_faces": classical.get("Gamma_faces", 0),
                    "Q_classical_faces": classical.get("Q_faces", 0),
                    "Upar_classical_faces": classical.get("Upar_faces", 0),
                }
            )
        return out

    def build_lagged_response(self, state, **kwargs):
        return CombinedTransportLaggedResponse(
            neoclassical_response=self.neoclassical_model.build_lagged_response(state, **kwargs),
            turbulent_response=self.turbulent_model.build_lagged_response(state, **kwargs),
            classical_response=self.classical_model.build_lagged_response(state, **kwargs),
        )

    def build_lagged_response_with_compact_coefficient_record(self, state, **kwargs):
        """Opt-in companion that preserves the ordinary combined response."""

        build_with_record = getattr(
            self.neoclassical_model,
            "build_lagged_response_with_compact_coefficient_record",
            None,
        )
        if not callable(build_with_record):
            raise NotImplementedError(
                "The active neoclassical model does not expose compact coefficient records."
            )
        neoclassical_response, coefficient_record = build_with_record(state)
        return (
            CombinedTransportLaggedResponse(
                neoclassical_response=neoclassical_response,
                turbulent_response=self.turbulent_model.build_lagged_response(state, **kwargs),
                classical_response=self.classical_model.build_lagged_response(state, **kwargs),
            ),
            coefficient_record,
        )

    def compact_coefficient_record_zero(self):
        zero_fn = getattr(self.neoclassical_model, "compact_coefficient_record_zero", None)
        if not callable(zero_fn):
            raise NotImplementedError(
                "The active neoclassical model does not expose compact coefficient records."
            )
        return zero_fn()

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

    def pullback_build_lagged_response_support_payload_batched_interpolated_faces(
        self,
        state,
        lagged_response_bars,
        support,
        **kwargs,
    ):
        """Batched exact-support transpose for the interpolated NTX face lane."""
        del kwargs
        pullback_fn = getattr(
            self.neoclassical_model,
            "pullback_build_lagged_response_support_payload_batched_interpolated_faces",
            None,
        )
        if not callable(pullback_fn):
            raise NotImplementedError(
                "The active neoclassical model does not expose the batched interpolated-face "
                "support pullback."
            )
        response_bars = None if lagged_response_bars is None else lagged_response_bars.neoclassical_response
        # The dedicated batched rule already returns every support leaf with
        # its leading objective axis, including zero/non-float leaves.  The
        # scalar sanitizer would drop that axis on non-float leaves.
        return pullback_fn(state, response_bars, support)

    def pullback_build_lagged_response_support_payload_batched_interpolated_faces_reuse_local_vjp_primal(
        self,
        state,
        lagged_response_bars,
        support,
        **kwargs,
    ):
        """Batched interpolated-face transpose that reuses each VJP primal.

        This is intentionally an isolated experimental helper.  Relative to
        :meth:`pullback_build_lagged_response_support_payload_batched_interpolated_faces`,
        each anchor obtains its primal response from the forward half of the
        already-required local ``jax.vjp``.  That primal is then used for the
        interpolation-coordinate transpose, rather than evaluating a separate
        local NTX response solely for that purpose.  The local pullback is
        still applied to the entire objective batch on device.
        """
        # This composite model only delegates the NTX part.  Boundary-condition
        # kwargs belong to the outer transport equation and are not accepted by
        # the local NTX helper, matching the established batched wrapper.
        del kwargs
        # The concrete
        # implementation below is intentionally unreachable here; it is kept
        # temporarily while this helper is moved to the NTX flux model.
        pullback_fn = getattr(
            self.neoclassical_model,
            "pullback_build_lagged_response_support_payload_batched_interpolated_faces_reuse_local_vjp_primal",
            None,
        )
        if not callable(pullback_fn):
            raise NotImplementedError(
                "The active neoclassical model does not expose the batched interpolated-face "
                "primal-reuse support pullback."
            )
        response_bars = None if lagged_response_bars is None else lagged_response_bars.neoclassical_response
        return pullback_fn(state, response_bars, support)

    def pullback_build_lagged_response_support_payload_batched_interpolated_faces_multi_rhs_shared_primal(
        self,
        state,
        lagged_response_bars,
        support,
        **kwargs,
    ):
        """Forward the isolated local-NTX multi-RHS support transpose.

        The Radau vector field belongs to this composite model, while the
        specialised factor-sharing rule belongs to its exact-NTX component.
        Like the existing batched wrappers, outer boundary-condition keywords
        are intentionally not passed to the local NTX helper.
        """
        del kwargs
        pullback_fn = getattr(
            self.neoclassical_model,
            "pullback_build_lagged_response_support_payload_batched_interpolated_faces_multi_rhs_shared_primal",
            None,
        )
        if not callable(pullback_fn):
            raise NotImplementedError(
                "The active neoclassical model does not expose the batched interpolated-face "
                "multi-RHS shared-primal support pullback."
            )
        response_bars = None if lagged_response_bars is None else lagged_response_bars.neoclassical_response
        return pullback_fn(state, response_bars, support)

    def pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_shared_primal(
        self,
        state,
        lagged_response_bars,
        support,
        **kwargs,
    ):
        """Forward the native matrix-RHS NTX support transpose.

        The Radau vector field is bound to this composite model, so every
        opt-in inner NTX pullback needs a matching outer forwarding hook.
        Keeping it separate preserves the prior multi-RHS experiment and all
        established support-pullback selectors.
        """

        del kwargs
        pullback_fn = getattr(
            self.neoclassical_model,
            "pullback_build_lagged_response_support_payload_batched_interpolated_faces_"
            "native_multi_rhs_shared_primal",
            None,
        )
        if not callable(pullback_fn):
            raise NotImplementedError(
                "The active neoclassical model does not expose the native matrix-RHS "
                "interpolated-face support pullback."
            )
        response_bars = (
            None if lagged_response_bars is None else lagged_response_bars.neoclassical_response
        )
        return pullback_fn(state, response_bars, support)

    def pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal(
        self, state, lagged_response_bars, support, **kwargs,
    ):
        """Forward the isolated native matrix-RHS drds-JVP reuse rule."""
        del kwargs
        pullback_fn = getattr(
            self.neoclassical_model,
            "pullback_build_lagged_response_support_payload_batched_interpolated_faces_"
            "native_multi_rhs_reuse_moment_drds_jvp_shared_primal",
            None,
        )
        if not callable(pullback_fn):
            raise NotImplementedError("The active neoclassical model does not expose the native drds-JVP reuse support pullback.")
        response_bars = None if lagged_response_bars is None else lagged_response_bars.neoclassical_response
        return pullback_fn(state, response_bars, support)

    def pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients(
        self, state, lagged_response_bars, support, **kwargs,
    ):
        """Forward the parallel native VMEC coefficient cotangent channel."""

        del kwargs
        pullback_fn = getattr(
            self.neoclassical_model,
            "pullback_build_lagged_response_support_payload_batched_interpolated_faces_"
            "native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients",
            None,
        )
        if not callable(pullback_fn):
            raise NotImplementedError(
                "The active neoclassical model does not expose native VMEC coefficient bars."
            )
        response_bars = None if lagged_response_bars is None else lagged_response_bars.neoclassical_response
        return pullback_fn(state, response_bars, support)

    def pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_directional_product_rule(
        self, state, lagged_response_bars, support, **kwargs,
    ):
        """Forward the opt-in direct directional VMEC-coefficient rule."""

        del kwargs
        pullback_fn = getattr(
            self.neoclassical_model,
            "pullback_build_lagged_response_support_payload_batched_interpolated_faces_"
            "native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_"
            "direct_directional_product_rule",
            None,
        )
        if not callable(pullback_fn):
            raise NotImplementedError(
                "The active neoclassical model does not expose the direct native "
                "VMEC coefficient rule."
            )
        response_bars = (
            None if lagged_response_bars is None else lagged_response_bars.neoclassical_response
        )
        return pullback_fn(state, response_bars, support)

    def pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_coefficient_pullback(
        self, state, lagged_response_bars, support, **kwargs,
    ):
        """Forward the opt-in direct coefficient-transpose rule."""
        del kwargs
        pullback_fn = getattr(
            self.neoclassical_model,
            "pullback_build_lagged_response_support_payload_batched_interpolated_faces_"
            "native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_"
            "direct_coefficient_pullback",
            None,
        )
        if not callable(pullback_fn):
            raise NotImplementedError(
                "The active neoclassical model does not expose the direct coefficient rule."
            )
        response_bars = (
            None if lagged_response_bars is None else lagged_response_bars.neoclassical_response
        )
        return pullback_fn(state, response_bars, support)

    def pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_directional_product_rule_per_energy_call_boundary(
        self, state, lagged_response_bars, support, **kwargs,
    ):
        """Forward the local non-inline call-boundary experiment."""
        del kwargs
        pullback_fn = getattr(
            self.neoclassical_model,
            "pullback_build_lagged_response_support_payload_batched_interpolated_faces_"
            "native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_"
            "direct_directional_product_rule_per_energy_call_boundary",
            None,
        )
        if not callable(pullback_fn):
            raise NotImplementedError(
                "The active neoclassical model does not expose the local call-boundary rule."
            )
        response_bars = (
            None if lagged_response_bars is None else lagged_response_bars.neoclassical_response
        )
        return pullback_fn(state, response_bars, support)

    def pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_compact_shared_primal(
        self,
        state,
        lagged_response_bars,
        support,
        **kwargs,
    ):
        """Forward the compact native matrix-RHS NTX support transpose."""

        del kwargs
        pullback_fn = getattr(
            self.neoclassical_model,
            "pullback_build_lagged_response_support_payload_batched_interpolated_faces_"
            "native_multi_rhs_compact_shared_primal",
            None,
        )
        if not callable(pullback_fn):
            raise NotImplementedError(
                "The active neoclassical model does not expose the compact native "
                "matrix-RHS interpolated-face support pullback."
            )
        response_bars = (
            None if lagged_response_bars is None else lagged_response_bars.neoclassical_response
        )
        return pullback_fn(state, response_bars, support)

    def pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_compact_residual_reuse_moment_drds_jvp_shared_primal(
        self, state, lagged_response_bars, support, **kwargs,
    ):
        """Forward the isolated split-residual native support rule."""
        del kwargs
        pullback_fn = getattr(
            self.neoclassical_model,
            "pullback_build_lagged_response_support_payload_batched_interpolated_faces_"
            "native_multi_rhs_compact_residual_reuse_moment_drds_jvp_shared_primal",
            None,
        )
        if not callable(pullback_fn):
            raise NotImplementedError(
                "The active neoclassical model does not expose the split-residual "
                "native support pullback."
            )
        response_bars = (
            None if lagged_response_bars is None else lagged_response_bars.neoclassical_response
        )
        return pullback_fn(state, response_bars, support)

    def pullback_build_lagged_response_state_and_support_payload_batched_interpolated_faces(
        self,
        state,
        lagged_response_bars,
        support,
        **kwargs,
    ):
        """Joint batched transpose of the combined lagged-response build.

        The NTX neoclassical component supplies the specialised joint
        state/support rule.  Turbulent and classical models do not depend on
        the NTX support payload, but their state cotangents are still part of
        the combined response and must be added here.
        """
        reuse_local_vjp_primal_anchor_response = bool(
            kwargs.pop("reuse_local_vjp_primal_anchor_response", False)
        )
        compact_prepared_support_carry = bool(
            kwargs.pop("compact_prepared_support_carry", False)
        )
        native_multi_rhs_reuse_moment_drds_jvp_shared_primal = bool(
            kwargs.pop("native_multi_rhs_reuse_moment_drds_jvp_shared_primal", False)
        )
        native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_no_prepared_carry = bool(
            kwargs.pop(
                "native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_no_prepared_carry",
                False,
            )
        )
        return_native_vmec_coefficient_bars = bool(
            kwargs.pop("return_native_vmec_coefficient_bars", False)
        )
        pullback_fn = getattr(
            self.neoclassical_model,
            (
                "pullback_build_lagged_response_state_and_support_payload_"
                "batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal"
                "_with_vmec_coefficients_no_prepared_carry"
                if native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_no_prepared_carry
                else "pullback_build_lagged_response_state_and_support_payload_"
                "batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal"
                if native_multi_rhs_reuse_moment_drds_jvp_shared_primal
                else "pullback_build_lagged_response_state_and_support_payload_"
                "batched_interpolated_faces_reuse_local_vjp_primal_compact_prepared_carry"
                if compact_prepared_support_carry
                else (
                "pullback_build_lagged_response_state_and_support_payload_batched_interpolated_faces_reuse_local_vjp_primal"
                if reuse_local_vjp_primal_anchor_response
                else "pullback_build_lagged_response_state_and_support_payload_batched_interpolated_faces"
                )
            ),
            None,
        )
        if not callable(pullback_fn):
            raise NotImplementedError(
                "The active neoclassical model does not expose the batched joint "
                "interpolated-face state/support pullback."
            )
        if lagged_response_bars is None:
            raise ValueError("Combined joint lagged-response pullback requires response cotangents.")

        native_result = pullback_fn(
            state,
            lagged_response_bars.neoclassical_response,
            support,
            packed_support_directional_adjoint=bool(
                kwargs.pop("packed_support_directional_adjoint", False)
            ),
            return_native_vmec_coefficient_bars=return_native_vmec_coefficient_bars,
        )
        if return_native_vmec_coefficient_bars:
            neoclassical_state_bars, support_bars, native_vmec_coefficient_bars = native_result
        else:
            neoclassical_state_bars, support_bars = native_result

        def _batched_state_pullback(model, response_bars):
            if response_bars is None:
                return jax.tree_util.tree_map(jnp.zeros_like, neoclassical_state_bars)
            model_pullback = getattr(model, "pullback_build_lagged_response", None)
            if callable(model_pullback):
                return jax.vmap(
                    lambda response_bar: model_pullback(state, response_bar, **kwargs)
                )(response_bars)
            _, generic_pullback = jax.vjp(
                lambda state_value: model.build_lagged_response(state_value, **kwargs),
                state,
            )
            return jax.vmap(lambda response_bar: generic_pullback(response_bar)[0])(response_bars)

        turbulent_state_bars = _batched_state_pullback(
            self.turbulent_model,
            lagged_response_bars.turbulent_response,
        )
        classical_state_bars = _batched_state_pullback(
            self.classical_model,
            lagged_response_bars.classical_response,
        )
        state_bars = jax.tree_util.tree_map(
            lambda neo_bar, turbulent_bar, classical_bar: neo_bar + turbulent_bar + classical_bar,
            neoclassical_state_bars,
            turbulent_state_bars,
            classical_state_bars,
        )
        if return_native_vmec_coefficient_bars:
            return state_bars, support_bars, native_vmec_coefficient_bars
        return state_bars, support_bars

    def pullback_build_lagged_response_state_and_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal(
        self,
        state,
        lagged_response_bars,
        support,
        **kwargs,
    ):
        """Forward the isolated native matrix-RHS joint pullback.

        This is deliberately a separate entry point for the post-sweep
        initial-carry experiment. Existing rebuild selectors continue to use
        their established generic joint hooks.
        """

        return self.pullback_build_lagged_response_state_and_support_payload_batched_interpolated_faces(
            state,
            lagged_response_bars,
            support,
            native_multi_rhs_reuse_moment_drds_jvp_shared_primal=True,
            **kwargs,
        )

    def pullback_build_lagged_response_state_and_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients(
        self, state, lagged_response_bars, support, **kwargs,
    ):
        """Joint native NTX state/support transpose with coefficient bars."""
        kwargs.pop("return_native_vmec_coefficient_bars", None)
        return self.pullback_build_lagged_response_state_and_support_payload_batched_interpolated_faces(
            state,
            lagged_response_bars,
            support,
            native_multi_rhs_reuse_moment_drds_jvp_shared_primal=True,
            return_native_vmec_coefficient_bars=True,
            **kwargs,
        )

    def pullback_build_lagged_response_state_and_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_no_prepared_carry(
        self, state, lagged_response_bars, support, **kwargs,
    ):
        """Forward the compact native VMEC-coefficient joint contract."""
        kwargs.pop("return_native_vmec_coefficient_bars", None)
        return self.pullback_build_lagged_response_state_and_support_payload_batched_interpolated_faces(
            state,
            lagged_response_bars,
            support,
            native_multi_rhs_reuse_moment_drds_jvp_shared_primal=True,
            native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_no_prepared_carry=True,
            return_native_vmec_coefficient_bars=True,
            **kwargs,
        )

    def pullback_build_lagged_response_state_and_support_payload_batched_interpolated_faces_reuse_local_vjp_primal(
        self,
        state,
        lagged_response_bars,
        support,
        **kwargs,
    ):
        """Opt-in wrapper for the joint local-primal-reuse NTX transpose."""
        return self.pullback_build_lagged_response_state_and_support_payload_batched_interpolated_faces(
            state,
            lagged_response_bars,
            support,
            reuse_local_vjp_primal_anchor_response=True,
            **kwargs,
        )

    def pullback_build_lagged_response_state_and_support_payload_batched_interpolated_faces_reuse_local_vjp_primal_compact_prepared_carry(
        self,
        state,
        lagged_response_bars,
        support,
        **kwargs,
    ):
        """Opt-in joint lowdot wrapper with a compact prepared scan carry."""
        return self.pullback_build_lagged_response_state_and_support_payload_batched_interpolated_faces(
            state,
            lagged_response_bars,
            support,
            reuse_local_vjp_primal_anchor_response=True,
            compact_prepared_support_carry=True,
            **kwargs,
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
        has_face_fluxes = all(
            self._can_contribute_face_fluxes(model, fluxes)
            for model, fluxes in (
                (self.neoclassical_model, neo),
                (self.turbulent_model, turb),
                (self.classical_model, classical),
            )
        )
        has_center_fluxes = all(
            "Gamma" in fluxes and "Q" in fluxes and "Upar" in fluxes
            for fluxes in (neo, turb, classical)
        )
        out = {}
        if has_center_fluxes:
            gamma_turb = (
                turb.get("Gamma", 0)
                if self.include_turbulent_particle_flux
                else self._zero_like_flux(
                    turb.get("Gamma", None),
                    self._zero_like_flux(neo.get("Gamma", None), self._zero_like_flux(classical.get("Gamma", None), 0)),
                )
            )
            out.update(
                {
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
            )
        if has_face_fluxes:
            gamma_turb_faces = (
                turb.get("Gamma_faces", 0)
                if self.include_turbulent_particle_flux
                else self._zero_like_flux(
                    turb.get("Gamma_faces", None),
                    self._zero_like_flux(
                        neo.get("Gamma_faces", None),
                        self._zero_like_flux(classical.get("Gamma_faces", None), 0),
                    ),
                )
            )
            out.update(
                {
                    "Gamma_faces": neo.get("Gamma_faces", 0) + gamma_turb_faces + classical.get("Gamma_faces", 0),
                    "Q_faces": neo.get("Q_faces", 0) + turb.get("Q_faces", 0) + classical.get("Q_faces", 0),
                    "Upar_faces": neo.get("Upar_faces", 0) + turb.get("Upar_faces", 0) + classical.get("Upar_faces", 0),
                    "Gamma_neo_faces": neo.get("Gamma_faces", 0),
                    "Q_neo_faces": neo.get("Q_faces", 0),
                    "Upar_neo_faces": neo.get("Upar_faces", 0),
                    "Gamma_turb_faces": gamma_turb_faces,
                    "Q_turb_faces": turb.get("Q_faces", 0),
                    "Upar_turb_faces": turb.get("Upar_faces", 0),
                    "Gamma_classical_faces": classical.get("Gamma_faces", 0),
                    "Q_classical_faces": classical.get("Q_faces", 0),
                    "Upar_classical_faces": classical.get("Upar_faces", 0),
                }
            )
        lagged_face_fluxes = None
        if all(f"{name}_faces" in out for name in ("Gamma", "Q", "Upar")):
            lagged_face_fluxes = {
                name: out[f"{name}_faces"] for name in ("Gamma", "Q", "Upar")
            }
        return self._apply_center_flux_mode(out, lagged_face_fluxes)

    def evaluate_with_lagged_response_tangent(self, state, state_direction, lagged_response, **kwargs):
        """Directional tangent of the combined cached flux response.

        For the intended quadratic-realtime-NTX route, the neoclassical
        contribution is quadratic and the analytical turbulent/classical
        cached contributions are linear (or absent).  Centered polarization
        is consequently exact for that combined cached response and keeps the
        normal value combiner as the single source of truth.
        """
        plus_state = dataclasses.replace(
            state,
            density=state.density + state_direction.density,
            pressure=state.pressure + state_direction.pressure,
            Er=state.Er + state_direction.Er,
        )
        minus_state = dataclasses.replace(
            state,
            density=state.density - state_direction.density,
            pressure=state.pressure - state_direction.pressure,
            Er=state.Er - state_direction.Er,
        )
        plus = self.evaluate_with_lagged_response(plus_state, lagged_response, **kwargs)
        minus = self.evaluate_with_lagged_response(minus_state, lagged_response, **kwargs)
        return jax.tree_util.tree_map(lambda left, right: 0.5 * (left - right), plus, minus)

    def pullback_evaluate_with_lagged_response(self, state, lagged_response, flux_bar, **kwargs):
        flux_bar = self._apply_center_flux_mode_pullback(flux_bar)

        def _submodel_pullback(model, subresponse, subflux_bar):
            if subresponse is None:
                return None
            if isinstance(subflux_bar, dict) and not subflux_bar:
                return None
            pullback_fn = getattr(model, "pullback_evaluate_with_lagged_response", None)
            if callable(pullback_fn):
                return pullback_fn(state, subresponse, subflux_bar, **kwargs)
            flux_output, pb = jax.vjp(
                lambda response_value: model.evaluate_with_lagged_response(
                    state,
                    response_value,
                    **kwargs,
                ),
                subresponse,
            )
            (subresponse_bar,) = pb(
                _complete_flux_bar_like(
                    flux_output,
                    subflux_bar,
                    context=f"CombinedTransportFluxModel.response.{type(model).__name__}",
                )
            )
            return subresponse_bar

        gamma_total_bar = flux_bar.get("Gamma", 0)
        q_total_bar = flux_bar.get("Q", 0)
        upar_total_bar = flux_bar.get("Upar", 0)
        gamma_faces_total_bar = flux_bar.get("Gamma_faces", 0)
        q_faces_total_bar = flux_bar.get("Q_faces", 0)
        upar_faces_total_bar = flux_bar.get("Upar_faces", 0)

        def _is_missing_bar(value):
            if value is None:
                return True
            arr = jnp.asarray(value)
            return arr.shape == () or arr.dtype == jax.dtypes.float0

        def _first_bar_template(keys, fallback=0):
            for key in keys:
                value = flux_bar.get(key, None)
                if not _is_missing_bar(value):
                    return jnp.asarray(value)
            return fallback

        def _bar_or_zero(key, template):
            value = flux_bar.get(key, None)
            if _is_missing_bar(value):
                return jnp.zeros_like(jnp.asarray(template))
            return jnp.asarray(value, dtype=jnp.asarray(template).dtype)

        gamma_template = _first_bar_template(("Gamma", "Gamma_neo", "Gamma_turb", "Gamma_classical"), 0)
        q_template = _first_bar_template(("Q", "Q_neo", "Q_turb", "Q_classical"), 0)
        upar_template = _first_bar_template(("Upar", "Upar_neo", "Upar_turb", "Upar_classical"), 0)
        gamma_faces_template = _first_bar_template(
            ("Gamma_faces", "Gamma_neo_faces", "Gamma_turb_faces", "Gamma_classical_faces"),
            0,
        )
        q_faces_template = _first_bar_template(("Q_faces", "Q_neo_faces", "Q_turb_faces", "Q_classical_faces"), 0)
        upar_faces_template = _first_bar_template(
            ("Upar_faces", "Upar_neo_faces", "Upar_turb_faces", "Upar_classical_faces"),
            0,
        )
        gamma_total_bar = _bar_or_zero("Gamma", gamma_template)
        q_total_bar = _bar_or_zero("Q", q_template)
        upar_total_bar = _bar_or_zero("Upar", upar_template)
        gamma_faces_total_bar = _bar_or_zero("Gamma_faces", gamma_faces_template)
        q_faces_total_bar = _bar_or_zero("Q_faces", q_faces_template)
        upar_faces_total_bar = _bar_or_zero("Upar_faces", upar_faces_template)
        neo_flux_bar = {}
        turb_flux_bar = {}
        classical_flux_bar = {}
        if any(key in flux_bar for key in ("Gamma", "Q", "Upar", "Gamma_neo", "Q_neo", "Upar_neo")):
            neo_flux_bar.update(
                {
                    "Gamma": gamma_total_bar + _bar_or_zero("Gamma_neo", gamma_template),
                    "Q": q_total_bar + _bar_or_zero("Q_neo", q_template),
                    "Upar": upar_total_bar + _bar_or_zero("Upar_neo", upar_template),
                }
            )
            turb_flux_bar.update(
                {
                    "Gamma": (
                        gamma_total_bar + _bar_or_zero("Gamma_turb", gamma_template)
                        if self.include_turbulent_particle_flux
                        else _bar_or_zero("Gamma_turb", gamma_template)
                    ),
                    "Q": q_total_bar + _bar_or_zero("Q_turb", q_template),
                    "Upar": upar_total_bar + _bar_or_zero("Upar_turb", upar_template),
                }
            )
            classical_flux_bar.update(
                {
                    "Gamma": gamma_total_bar + _bar_or_zero("Gamma_classical", gamma_template),
                    "Q": q_total_bar + _bar_or_zero("Q_classical", q_template),
                    "Upar": upar_total_bar + _bar_or_zero("Upar_classical", upar_template),
                }
            )
        if any(key in flux_bar for key in ("Gamma_faces", "Q_faces", "Upar_faces", "Gamma_neo_faces", "Q_neo_faces", "Upar_neo_faces")):
            neo_flux_bar.update(
                {
                    "Gamma_faces": gamma_faces_total_bar + _bar_or_zero("Gamma_neo_faces", gamma_faces_template),
                    "Q_faces": q_faces_total_bar + _bar_or_zero("Q_neo_faces", q_faces_template),
                    "Upar_faces": upar_faces_total_bar + _bar_or_zero("Upar_neo_faces", upar_faces_template),
                }
            )
            turb_flux_bar.update(
                {
                    "Gamma_faces": (
                        gamma_faces_total_bar + _bar_or_zero("Gamma_turb_faces", gamma_faces_template)
                        if self.include_turbulent_particle_flux
                        else _bar_or_zero("Gamma_turb_faces", gamma_faces_template)
                    ),
                    "Q_faces": q_faces_total_bar + _bar_or_zero("Q_turb_faces", q_faces_template),
                    "Upar_faces": upar_faces_total_bar + _bar_or_zero("Upar_turb_faces", upar_faces_template),
                }
            )
            classical_flux_bar.update(
                {
                    "Gamma_faces": gamma_faces_total_bar + _bar_or_zero("Gamma_classical_faces", gamma_faces_template),
                    "Q_faces": q_faces_total_bar + _bar_or_zero("Q_classical_faces", q_faces_template),
                    "Upar_faces": upar_faces_total_bar + _bar_or_zero("Upar_classical_faces", upar_faces_template),
                }
            )

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
        flux_bar = self._apply_center_flux_mode_pullback(flux_bar)

        def _is_missing_bar(value):
            if value is None:
                return True
            arr = jnp.asarray(value)
            return arr.shape == () or arr.dtype == jax.dtypes.float0

        def _first_bar_template(keys, fallback=0):
            for key in keys:
                value = flux_bar.get(key, None)
                if not _is_missing_bar(value):
                    return jnp.asarray(value)
            return fallback

        def _bar_or_zero(key, template):
            value = flux_bar.get(key, None)
            if _is_missing_bar(value):
                return jnp.zeros_like(jnp.asarray(template))
            return jnp.asarray(value, dtype=jnp.asarray(template).dtype)

        gamma_template = _first_bar_template(("Gamma", "Gamma_neo", "Gamma_turb", "Gamma_classical"), 0)
        q_template = _first_bar_template(("Q", "Q_neo", "Q_turb", "Q_classical"), 0)
        upar_template = _first_bar_template(("Upar", "Upar_neo", "Upar_turb", "Upar_classical"), 0)
        gamma_faces_template = _first_bar_template(
            ("Gamma_faces", "Gamma_neo_faces", "Gamma_turb_faces", "Gamma_classical_faces"),
            0,
        )
        q_faces_template = _first_bar_template(("Q_faces", "Q_neo_faces", "Q_turb_faces", "Q_classical_faces"), 0)
        upar_faces_template = _first_bar_template(
            ("Upar_faces", "Upar_neo_faces", "Upar_turb_faces", "Upar_classical_faces"),
            0,
        )
        gamma_total_bar = _bar_or_zero("Gamma", gamma_template)
        q_total_bar = _bar_or_zero("Q", q_template)
        upar_total_bar = _bar_or_zero("Upar", upar_template)
        gamma_faces_total_bar = _bar_or_zero("Gamma_faces", gamma_faces_template)
        q_faces_total_bar = _bar_or_zero("Q_faces", q_faces_template)
        upar_faces_total_bar = _bar_or_zero("Upar_faces", upar_faces_template)

        neo_flux_bar = {}
        if any(key in flux_bar for key in ("Gamma", "Q", "Upar", "Gamma_neo", "Q_neo", "Upar_neo")):
            neo_flux_bar.update(
                {
                    "Gamma": gamma_total_bar + _bar_or_zero("Gamma_neo", gamma_template),
                    "Q": q_total_bar + _bar_or_zero("Q_neo", q_template),
                    "Upar": upar_total_bar + _bar_or_zero("Upar_neo", upar_template),
                }
            )
        if any(key in flux_bar for key in ("Gamma_faces", "Q_faces", "Upar_faces", "Gamma_neo_faces", "Q_neo_faces", "Upar_neo_faces")):
            neo_flux_bar.update(
                {
                    "Gamma_faces": gamma_faces_total_bar + _bar_or_zero("Gamma_neo_faces", gamma_faces_template),
                    "Q_faces": q_faces_total_bar + _bar_or_zero("Q_neo_faces", q_faces_template),
                    "Upar_faces": upar_faces_total_bar + _bar_or_zero("Upar_neo_faces", upar_faces_template),
                }
            )
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
        flux_output, support_delta_pullback = jax.vjp(
            lambda support_delta: self.neoclassical_model.with_support_payload(
                _add_float_delta_tree(support, support_delta)
            ).evaluate_with_lagged_response(
                state,
                lagged_response.neoclassical_response,
                **kwargs,
            ),
            support_delta0,
        )
        (support_bar,) = support_delta_pullback(
            _complete_flux_bar_like(
                flux_output,
                neo_flux_bar,
                context="CombinedTransportFluxModel.support_payload.neoclassical",
            )
        )
        return _sanitize_float_delta_bar_tree(support, support_bar)

    def pullback_evaluate_with_lagged_response_state(self, state, lagged_response, flux_bar, **kwargs):
        flux_bar = self._apply_center_flux_mode_pullback(flux_bar)

        def _zero_state_bar():
            return jax.tree_util.tree_map(jnp.zeros_like, state)

        def _submodel_state_pullback(model, subresponse, subflux_bar):
            if subresponse is None:
                return _zero_state_bar()
            pullback_fn = getattr(model, "pullback_evaluate_with_lagged_response_state", None)
            if callable(pullback_fn):
                return pullback_fn(state, subresponse, subflux_bar, **kwargs)
            flux_output, pb = jax.vjp(
                lambda state_value: model.evaluate_with_lagged_response(
                    state_value,
                    subresponse,
                    **kwargs,
                ),
                state,
            )
            (state_bar,) = pb(
                _complete_flux_bar_like(
                    flux_output,
                    subflux_bar,
                    context=f"CombinedTransportFluxModel.state.{type(model).__name__}",
                )
            )
            return state_bar

        def _is_missing_bar(value):
            if value is None:
                return True
            arr = jnp.asarray(value)
            return arr.shape == () or arr.dtype == jax.dtypes.float0

        def _first_bar_template(keys, fallback=0):
            for key in keys:
                value = flux_bar.get(key, None)
                if not _is_missing_bar(value):
                    return jnp.asarray(value)
            return fallback

        def _bar_or_zero(key, template):
            value = flux_bar.get(key, None)
            if _is_missing_bar(value):
                return jnp.zeros_like(jnp.asarray(template))
            return jnp.asarray(value, dtype=jnp.asarray(template).dtype)

        gamma_template = _first_bar_template(("Gamma", "Gamma_neo", "Gamma_turb", "Gamma_classical"), 0)
        q_template = _first_bar_template(("Q", "Q_neo", "Q_turb", "Q_classical"), 0)
        upar_template = _first_bar_template(("Upar", "Upar_neo", "Upar_turb", "Upar_classical"), 0)
        gamma_faces_template = _first_bar_template(
            ("Gamma_faces", "Gamma_neo_faces", "Gamma_turb_faces", "Gamma_classical_faces"),
            0,
        )
        q_faces_template = _first_bar_template(("Q_faces", "Q_neo_faces", "Q_turb_faces", "Q_classical_faces"), 0)
        upar_faces_template = _first_bar_template(
            ("Upar_faces", "Upar_neo_faces", "Upar_turb_faces", "Upar_classical_faces"),
            0,
        )
        gamma_total_bar = _bar_or_zero("Gamma", gamma_template)
        q_total_bar = _bar_or_zero("Q", q_template)
        upar_total_bar = _bar_or_zero("Upar", upar_template)
        gamma_faces_total_bar = _bar_or_zero("Gamma_faces", gamma_faces_template)
        q_faces_total_bar = _bar_or_zero("Q_faces", q_faces_template)
        upar_faces_total_bar = _bar_or_zero("Upar_faces", upar_faces_template)

        neo_flux_bar = {}
        turb_flux_bar = {}
        classical_flux_bar = {}
        if any(key in flux_bar for key in ("Gamma", "Q", "Upar", "Gamma_neo", "Q_neo", "Upar_neo")):
            neo_flux_bar.update(
                {
                    "Gamma": gamma_total_bar + _bar_or_zero("Gamma_neo", gamma_template),
                    "Q": q_total_bar + _bar_or_zero("Q_neo", q_template),
                    "Upar": upar_total_bar + _bar_or_zero("Upar_neo", upar_template),
                }
            )
            turb_flux_bar.update(
                {
                    "Gamma": (
                        gamma_total_bar + _bar_or_zero("Gamma_turb", gamma_template)
                        if self.include_turbulent_particle_flux
                        else _bar_or_zero("Gamma_turb", gamma_template)
                    ),
                    "Q": q_total_bar + _bar_or_zero("Q_turb", q_template),
                    "Upar": upar_total_bar + _bar_or_zero("Upar_turb", upar_template),
                }
            )
            classical_flux_bar.update(
                {
                    "Gamma": gamma_total_bar + _bar_or_zero("Gamma_classical", gamma_template),
                    "Q": q_total_bar + _bar_or_zero("Q_classical", q_template),
                    "Upar": upar_total_bar + _bar_or_zero("Upar_classical", upar_template),
                }
            )
        if any(key in flux_bar for key in ("Gamma_faces", "Q_faces", "Upar_faces", "Gamma_neo_faces", "Q_neo_faces", "Upar_neo_faces")):
            neo_flux_bar.update(
                {
                    "Gamma_faces": gamma_faces_total_bar + _bar_or_zero("Gamma_neo_faces", gamma_faces_template),
                    "Q_faces": q_faces_total_bar + _bar_or_zero("Q_neo_faces", q_faces_template),
                    "Upar_faces": upar_faces_total_bar + _bar_or_zero("Upar_neo_faces", upar_faces_template),
                }
            )
            turb_flux_bar.update(
                {
                    "Gamma_faces": (
                        gamma_faces_total_bar + _bar_or_zero("Gamma_turb_faces", gamma_faces_template)
                        if self.include_turbulent_particle_flux
                        else _bar_or_zero("Gamma_turb_faces", gamma_faces_template)
                    ),
                    "Q_faces": q_faces_total_bar + _bar_or_zero("Q_turb_faces", q_faces_template),
                    "Upar_faces": upar_faces_total_bar + _bar_or_zero("Upar_turb_faces", upar_faces_template),
                }
            )
            classical_flux_bar.update(
                {
                    "Gamma_faces": gamma_faces_total_bar + _bar_or_zero("Gamma_classical_faces", gamma_faces_template),
                    "Q_faces": q_faces_total_bar + _bar_or_zero("Q_classical_faces", q_faces_template),
                    "Upar_faces": upar_faces_total_bar + _bar_or_zero("Upar_classical_faces", upar_faces_template),
                }
            )

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
    density_floor: Any = DEFAULT_TRANSPORT_DENSITY_FLOOR
    # This is intentionally an opt-in forward experiment.  The linear
    # response remains the production default and higher-order responses do
    # not yet have exact reverse-AD rules.
    lagged_response_taylor_order: int = 1

    def __post_init__(self):
        order = int(self.lagged_response_taylor_order)
        if order not in {1, 2, 3}:
            raise ValueError(
                "NTX database lagged_response_taylor_order must be 1, 2, or 3."
            )

    def _require_supported_reverse_lagged_response(self):
        if int(self.lagged_response_taylor_order) > 1:
            raise NotImplementedError(
                "Higher-order NTX database lagged responses are forward-only. "
                "Their exact reverse rules require higher-response pullbacks."
            )

    @staticmethod
    def _anchored_taylor_terms(function, anchor_state, delta_state, order):
        """Return first through third directional Taylor terms at one anchor.

        ``delta_state`` is intentionally held fixed in every nested JVP, so
        the outputs are J[delta], H[delta, delta], and D3[delta, delta,
        delta] of ``function`` at ``anchor_state``.
        """
        _, first = jax.jvp(function, (anchor_state,), (delta_state,))
        if order == 1:
            return first, None, None

        def _first_direction(anchor_value):
            return jax.jvp(function, (anchor_value,), (delta_state,))[1]

        _, second = jax.jvp(_first_direction, (anchor_state,), (delta_state,))
        if order == 2:
            return first, second, None

        def _second_direction(anchor_value):
            return jax.jvp(_first_direction, (anchor_value,), (delta_state,))[1]

        _, third = jax.jvp(_second_direction, (anchor_state,), (delta_state,))
        return first, second, third

    @staticmethod
    def _add_taylor_terms(reference, first, second=None, third=None):
        if third is not None:
            return jax.tree_util.tree_map(
                lambda ref, one, two, three: ref + one + 0.5 * two + three / 6.0,
                reference,
                first,
                second,
                third,
            )
        if second is not None:
            return jax.tree_util.tree_map(
                lambda ref, one, two: ref + one + 0.5 * two,
                reference,
                first,
                second,
            )
        return jax.tree_util.tree_map(lambda ref, one: ref + one, reference, first)

    def __call__(self, state) -> dict:
        density = safe_density(state.density, self.density_floor)
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

    def pullback_direct_rhs_state(self, state, flux_bar):
        """Transpose the direct database flux map with respect to its state.

        This is intentionally a *flux-model* boundary.  The Radau reverse no
        longer needs to trace the composed transport RHS (equations, sources,
        quasineutral projection, and the database interpolation) as one VJP.
        The interpolation-coordinate derivative remains the established
        database primitive derivative; table values themselves are handled by
        :meth:`pullback_direct_rhs_support_payload` and folded through the
        recorded scan once after the sweep.
        """
        zero = jnp.zeros_like(jnp.asarray(state.density))

        def _bar(name):
            value = flux_bar.get(name, None)
            if value is None:
                return zero
            value = jnp.asarray(value)
            return zero if value.ndim == 0 or value.dtype == jax.dtypes.float0 else value

        # Split the centre-flux transpose at the same boundaries as the
        # exact-Lij implementation: flux algebra, local Lij construction,
        # finite-volume gradients, then TransportState unpacking.  In
        # particular, do not ask JAX for one VJP through the complete flux
        # model for every Radau state-basis cotangent.
        def _primitive_inputs(state_value):
            density_value = safe_density(state_value.density, self.density_floor)
            return density_value, state_value.temperature, state_value.Er

        density, temperature, er_profile = _primitive_inputs(state)

        def _gradients(density_value, temperature_value):
            # ``__call__`` deliberately uses the ordinary direct-centre RHS
            # contract: no root-specific boundary constraints. Match its
            # ``get_Neoclassical_Fluxes`` default exactly.
            density_right_value = density_value[:, -1]
            density_right_grad_value = jnp.zeros_like(density_right_value)
            temperature_right_value = temperature_value[:, -1]
            temperature_right_grad_value = jnp.zeros_like(temperature_right_value)
            dndr_value = jax.vmap(
                lambda values, right, right_grad: get_gradient_density(
                    values, self.geometry.r_grid, self.geometry.r_grid_half, self.geometry.dr,
                    right_face_constraint=right, right_face_grad_constraint=right_grad,
                )
            )(density_value, density_right_value, density_right_grad_value)
            dtdr_value = jax.vmap(
                lambda values, right, right_grad: get_gradient_temperature(
                    values, self.geometry.r_grid, self.geometry.r_grid_half, self.geometry.dr,
                    right_face_constraint=right, right_face_grad_constraint=right_grad,
                )
            )(temperature_value, temperature_right_value, temperature_right_grad_value)
            return dndr_value, dtdr_value

        dndr, dtdr = _gradients(density, temperature)

        def _lij(density_value, temperature_value, er_value):
            vthermal_value = get_v_thermal(self.species.mass, temperature_value)
            return jax.vmap(
                lambda species_index: jax.vmap(
                    lambda radius_index: get_Lij_matrix(
                        self.species, self.energy_grid, self.geometry, self.database,
                        species_index, radius_index, er_value, temperature_value,
                        density_value, vthermal_value,
                        _collisionality_kind(self.collisionality_model),
                    )
                )(self.geometry.full_grid_indices)
            )(self.species.species_indices)

        lij = _lij(density, temperature, er_profile)

        def _flux_algebra(lij_value, density_value, temperature_value, dndr_value, dtdr_value, er_value):
            a1 = jax.vmap(
                lambda charge, density_a, temperature_a, dndr_a, dtdr_a:
                get_Thermodynamical_Forces_A1(
                    charge, density_a, temperature_a, dndr_a, dtdr_a, er_value
                )
            )(self.species.charge, density_value, temperature_value, dndr_value, dtdr_value)
            a2 = jax.vmap(get_Thermodynamical_Forces_A2)(temperature_value, dtdr_value)
            a3 = get_Thermodynamical_Forces_A3(er_value)
            density_phys = DENSITY_STATE_TO_PHYSICAL * density_value
            temperature_phys = TEMPERATURE_STATE_TO_PHYSICAL * temperature_value
            return {
                "Gamma": -density_phys * (
                    lij_value[:, :, 0, 0] * a1
                    + lij_value[:, :, 0, 1] * a2
                    + lij_value[:, :, 0, 2] * a3[None, :]
                ),
                "Q": -temperature_phys * density_phys * (
                    lij_value[:, :, 1, 0] * a1
                    + lij_value[:, :, 1, 1] * a2
                    + lij_value[:, :, 1, 2] * a3[None, :]
                ),
                "Upar": -density_phys * (
                    lij_value[:, :, 2, 0] * a1
                    + lij_value[:, :, 2, 1] * a2
                    + lij_value[:, :, 2, 2] * a3[None, :]
                ),
            }

        _, algebra_pullback = jax.vjp(
            _flux_algebra, lij, density, temperature, dndr, dtdr, er_profile
        )
        lij_bar, density_bar, temperature_bar, dndr_bar, dtdr_bar, er_bar = algebra_pullback(
            {"Gamma": _bar("Gamma"), "Q": _bar("Q"), "Upar": _bar("Upar")}
        )
        _, lij_pullback = jax.vjp(_lij, density, temperature, er_profile)
        lij_density_bar, lij_temperature_bar, lij_er_bar = lij_pullback(lij_bar)
        _, gradients_pullback = jax.vjp(_gradients, density, temperature)
        gradient_density_bar, gradient_temperature_bar = gradients_pullback((dndr_bar, dtdr_bar))
        _, state_inputs_pullback = jax.vjp(_primitive_inputs, state)
        (state_bar,) = state_inputs_pullback((
            density_bar + lij_density_bar + gradient_density_bar,
            temperature_bar + lij_temperature_bar + gradient_temperature_bar,
            er_bar + lij_er_bar,
        ))
        return state_bar

    def pullback_direct_rhs_support_payload(self, state, flux_bar, support):
        """Compact black-box transpose of the radial NTX database tables.

        This is table-only by design: accumulated table bars are later folded
        exactly once through the retained NTX scan record.  It is the database
        analogue of the low-dot Lij support rule, not a lagged-response VJP.
        """
        if not isinstance(support, dict) or "database" not in support:
            return None
        database = support["database"]
        density = safe_density(state.density, self.density_floor)
        zero = jnp.zeros_like(jnp.asarray(density))

        def _bar(name):
            value = flux_bar.get(name, None)
            if value is None:
                return zero
            value = jnp.asarray(value)
            return zero if value.ndim == 0 or value.dtype == jax.dtypes.float0 else value

        d11_bar, d13_bar, d33_bar = pullback_preprocessed_radial_database_fluxes(
            self.species,
            self.energy_grid,
            self.geometry,
            database,
            state.Er,
            state.temperature,
            density,
            _bar("Gamma"),
            _bar("Q"),
            _bar("Upar"),
            _collisionality_kind(self.collisionality_model),
        )
        database_bar = _float_delta_tree_like(database)
        database_bar = dataclasses.replace(
            database_bar,
            D11_log=d11_bar,
            D13=d13_bar,
            D33=d33_bar,
        )
        support_bar = dict(_float_delta_tree_like(support))
        support_bar["database"] = _sanitize_float_delta_bar_tree(database, database_bar)
        return support_bar

    def pullback_direct_rhs_geometry_by_radius(self, state, flux_bar, geometry):
        """Transpose direct centre fluxes to fixed-database geometry locally.

        The database table is fixed here: its accumulated cotangent is owned
        by :meth:`pullback_direct_rhs_support_payload` and folded through the
        retained scan once after the reverse sweep.  Avoiding a VJP of the
        complete radial flux table keeps that scan record and all unrelated
        equation/source work outside this boundary.
        """
        zero = jnp.zeros_like(jnp.asarray(state.density))

        def _bar(name):
            value = flux_bar.get(name, None)
            if value is None:
                return zero
            value = jnp.asarray(value)
            return zero if value.ndim == 0 or value.dtype == jax.dtypes.float0 else value

        gamma_bar, q_bar, upar_bar = _bar("Gamma"), _bar("Q"), _bar("Upar")
        radius_indices = jnp.arange(gamma_bar.shape[-1], dtype=jnp.int32)
        geometry_delta0 = _float_delta_tree_like(geometry)
        leaves0, treedef = jax.tree_util.tree_flatten(geometry_delta0)
        shapes = tuple(jnp.asarray(leaf).shape for leaf in leaves0)
        sizes = tuple(int(jnp.asarray(leaf).size) for leaf in leaves0)
        flat_delta0 = jnp.concatenate(
            tuple(jnp.ravel(jnp.asarray(leaf)) for leaf in leaves0)
        )

        def _split(flat_delta):
            leaves = []
            offset = 0
            for size, shape in zip(sizes, shapes, strict=True):
                leaves.append(jnp.reshape(flat_delta[offset : offset + size], shape))
                offset += size
            return treedef.unflatten(leaves)

        def _accumulate(carry, radius_index):
            def _local_fluxes(flat_delta):
                model = dataclasses.replace(
                    self,
                    geometry=_add_float_delta_tree(geometry, _split(flat_delta)),
                )
                return model.build_local_direct_flux_evaluator(state)(
                    radius_index, state.Er[radius_index]
                )

            _, pullback = jax.vjp(_local_fluxes, flat_delta0)
            local_bar = {
                "Gamma": jax.lax.dynamic_index_in_dim(
                    gamma_bar, radius_index, axis=1, keepdims=False
                ),
                "Q": jax.lax.dynamic_index_in_dim(
                    q_bar, radius_index, axis=1, keepdims=False
                ),
                "Upar": jax.lax.dynamic_index_in_dim(
                    upar_bar, radius_index, axis=1, keepdims=False
                ),
            }
            (flat_bar,) = pullback(local_bar)
            return carry + flat_bar, None

        flat_bar, _ = jax.lax.scan(
            _accumulate, jnp.zeros_like(flat_delta0), radius_indices
        )
        return _split(flat_bar)

    def pullback_local_particle_flux_support_payload(self, state, flux_bar, support):
        """Compact table transpose of the local direct-centre root flux.

        The selected-root primitive is the local restriction of
        :meth:`__call__`; therefore this uses the identical direct-centre
        last-cell, zero-gradient closure.  It must not use transport face
        boundary constraints, which would transpose a different outer-cell
        flux than the root solver evaluated.
        """
        if not isinstance(support, dict) or "database" not in support:
            return None
        database = support["database"]
        density = safe_density(state.density, self.density_floor)
        density_right_constraint = density[:, -1]
        density_right_grad_constraint = jnp.zeros_like(density_right_constraint)
        temperature_right_constraint = state.temperature[:, -1]
        temperature_right_grad_constraint = jnp.zeros_like(temperature_right_constraint)
        zero = jnp.zeros_like(jnp.asarray(density))

        def _bar(name):
            value = flux_bar.get(name, None)
            if value is None:
                return zero
            value = jnp.asarray(value)
            return zero if value.ndim == 0 or value.dtype == jax.dtypes.float0 else value

        d11_bar, d13_bar, d33_bar = pullback_preprocessed_radial_database_fluxes(
            self.species, self.energy_grid, self.geometry, database,
            state.Er, state.temperature, density,
            _bar("Gamma"), _bar("Q"), _bar("Upar"),
            _collisionality_kind(self.collisionality_model),
            density_right_constraint,
            density_right_grad_constraint,
            temperature_right_constraint,
            temperature_right_grad_constraint,
        )
        database_bar = dataclasses.replace(
            _float_delta_tree_like(database),
            D11_log=d11_bar, D13=d13_bar, D33=d33_bar,
        )
        support_bar = dict(_float_delta_tree_like(support))
        support_bar["database"] = _sanitize_float_delta_bar_tree(database, database_bar)
        return support_bar

    def evaluate_momentum_corrected_fluxes(self, state, *, diagnostics: bool = False) -> dict:
        """Evaluate database-interpolated neoclassical fluxes with momentum correction.

        The runtime database retains the monoenergetic ``D11``, ``D13`` and
        ``D33`` surfaces needed by the established database momentum
        correction.  This is deliberately separate from :meth:`__call__`:
        transport evolution continues to use its configured black-box flux
        path, while the bootstrap objective requests the corrected parallel
        flow explicitly.
        """

        if diagnostics:
            raise NotImplementedError(
                "Momentum-correction diagnostics are not yet exposed for the "
                "interpolated NTX database model."
            )
        density = safe_density(state.density, self.density_floor)
        density_right_constraint, density_right_grad_constraint = _extract_right_constraints(
            self.bc_density,
            density,
            self.geometry.r_grid_half,
        )
        temperature_right_constraint, temperature_right_grad_constraint = _extract_right_constraints(
            self.bc_temperature,
            state.temperature,
            self.geometry.r_grid_half,
        )
        gamma_neo, q_neo, upar_neo, qpar_neo, upar2_neo = (
            get_Neoclassical_Fluxes_With_Momentum_Correction(
                self.species,
                self.energy_grid,
                self.geometry,
                self.database,
                state.Er,
                state.temperature,
                density,
                density_right_constraint=density_right_constraint,
                density_right_grad_constraint=density_right_grad_constraint,
                temperature_right_constraint=temperature_right_constraint,
                temperature_right_grad_constraint=temperature_right_grad_constraint,
            )
        )
        return {
            "Gamma": gamma_neo,
            "Q": q_neo,
            "Upar": upar_neo,
            "Gamma_neo": gamma_neo,
            "Q_neo": q_neo,
            "Upar_neo": upar_neo,
            "qpar_neo": qpar_neo,
            "Upar2_neo": upar2_neo,
        }

    def evaluate_momentum_corrected_upar_only(self, state):
        """Return only the database momentum-corrected parallel flow.

        Keep this separate from the general flux-table evaluator: bootstrap
        objectives require ``Upar`` only, and must not stage the unused
        Gamma/Q/qpar/Upar2 output algebra in their local VJPs.
        """
        density = safe_density(state.density, self.density_floor)
        density_right_constraint, density_right_grad_constraint = _extract_right_constraints(
            self.bc_density, density, self.geometry.r_grid_half
        )
        temperature_right_constraint, temperature_right_grad_constraint = _extract_right_constraints(
            self.bc_temperature, state.temperature, self.geometry.r_grid_half
        )
        return get_Neoclassical_Upar_With_Momentum_Correction(
            self.species,
            self.energy_grid,
            self.geometry,
            self.database,
            state.Er,
            state.temperature,
            density,
            density_right_constraint=density_right_constraint,
            density_right_grad_constraint=density_right_grad_constraint,
            temperature_right_constraint=temperature_right_constraint,
            temperature_right_grad_constraint=temperature_right_grad_constraint,
        )

    def _momentum_corrected_upar_one_radius(self, state, radius_index):
        """Evaluate the corrected database ``U_parallel`` at one radius.

        This is the database analogue of the realtime-Lij local bootstrap
        primitive.  It deliberately exposes a *single* radial momentum solve:
        a state transpose can consequently keep the (large) database tables
        fixed instead of asking JAX to retain the all-radii database graph.
        The axis follows the established full evaluator, whose coefficient
        matrices at index zero are copied from index one before the correction
        solve.
        """
        density = safe_density(state.density, self.density_floor)
        temperature = state.temperature
        density_right, density_right_grad = _extract_right_constraints(
            self.bc_density, density, self.geometry.r_grid_half
        )
        temperature_right, temperature_right_grad = _extract_right_constraints(
            self.bc_temperature, temperature, self.geometry.r_grid_half
        )
        density_right = density[:, -1] if density_right is None else density_right
        density_right_grad = (
            jnp.zeros_like(density_right)
            if density_right_grad is None
            else density_right_grad
        )
        temperature_right = (
            temperature[:, -1] if temperature_right is None else temperature_right
        )
        temperature_right_grad = (
            jnp.zeros_like(temperature_right)
            if temperature_right_grad is None
            else temperature_right_grad
        )
        dndr = jax.vmap(
            lambda values, right, right_grad: get_gradient_density(
                values,
                self.geometry.r_grid,
                self.geometry.r_grid_half,
                self.geometry.dr,
                right_face_constraint=right,
                right_face_grad_constraint=right_grad,
            )
        )(density, density_right, density_right_grad)
        dTdr = jax.vmap(
            lambda values, right, right_grad: get_gradient_temperature(
                values,
                self.geometry.r_grid,
                self.geometry.r_grid_half,
                self.geometry.dr,
                right_face_constraint=right,
                right_face_grad_constraint=right_grad,
            )
        )(temperature, temperature_right, temperature_right_grad)
        A1 = jax.vmap(
            lambda charge, density_a, temperature_a, dndr_a, dTdr_a: get_Thermodynamical_Forces_A1(
                charge, density_a, temperature_a, dndr_a, dTdr_a, state.Er
            )
        )(self.species.charge, density, temperature, dndr, dTdr)
        A2 = jax.vmap(get_Thermodynamical_Forces_A2)(temperature, dTdr)
        A3 = get_Thermodynamical_Forces_A3(state.Er)
        v_thermal = get_v_thermal(self.species.mass, temperature)
        species_indices = jnp.arange(int(self.species.number_species), dtype=jnp.int32)
        coefficient_radius = jnp.maximum(radius_index, jnp.asarray(1, dtype=radius_index.dtype))
        lij, eij, nu_av = jax.vmap(
            lambda species_index: get_Lij_matrix_with_momentum_correction(
                self.species,
                self.energy_grid,
                self.geometry,
                self.database,
                species_index,
                coefficient_radius,
                state.Er,
                temperature,
                density,
                v_thermal,
            )
        )(species_indices)
        _gamma, _q, upar, _qpar, _upar2 = get_momentum_Correction(
            self.species,
            self.energy_grid,
            self.geometry,
            radius_index,
            lij,
            eij,
            nu_av,
            v_thermal,
            density,
            temperature,
            A1,
            A2,
            A3,
            self.species.mass,
            self.species.charge,
            dndr,
            dTdr,
        )
        return upar

    def pullback_momentum_corrected_upar_state_by_radius(self, state, upar_bar):
        """Transpose corrected-Upar to state with database tables held fixed."""
        upar_bar = jnp.asarray(upar_bar, dtype=state.pressure.dtype)
        radius_indices = jnp.arange(upar_bar.shape[-1], dtype=jnp.int32)

        def _zero(leaf):
            array = jnp.asarray(leaf)
            return jnp.zeros_like(array) if jnp.issubdtype(array.dtype, jnp.inexact) else jnp.zeros(array.shape, dtype=jnp.float64)

        state_bar0 = jax.tree_util.tree_map(_zero, state)

        def _accumulate(carry, radius_index):
            _, pullback = jax.vjp(
                lambda state_value: self._momentum_corrected_upar_one_radius(
                    state_value, radius_index
                ),
                state,
            )
            local_bar = jax.lax.dynamic_index_in_dim(
                upar_bar, radius_index, axis=1, keepdims=False
            )
            (state_bar,) = pullback(local_bar)
            return jax.tree_util.tree_map(lambda left, right: left + right, carry, state_bar), None

        state_bar, _ = jax.lax.scan(_accumulate, state_bar0, radius_indices)
        return state_bar

    def pullback_momentum_corrected_upar_state_geometry_by_radius(
        self, state, upar_bar, geometry
    ):
        """Joint compact corrected-Upar transpose to state and geometry.

        The recorded-database bootstrap boundary holds its three interpolation
        tables fixed while differentiating state and geometry.  Keeping those
        two cotangents in separate helpers made two otherwise identical local
        VJP scans.  This is the database counterpart of the exact-Lij joint
        local rule: one single-radius momentum VJP contributes both bars.
        Database-table bars remain the explicit interpolation transpose in
        :meth:`pullback_momentum_corrected_upar_database_by_radius`.
        """
        upar_bar = jnp.asarray(upar_bar, dtype=state.pressure.dtype)
        radius_indices = jnp.arange(upar_bar.shape[-1], dtype=jnp.int32)

        def _zero_state_leaf(leaf):
            array = jnp.asarray(leaf)
            return (
                jnp.zeros_like(array)
                if jnp.issubdtype(array.dtype, jnp.inexact)
                else jnp.zeros(array.shape, dtype=jnp.float64)
            )

        state_bar0 = jax.tree_util.tree_map(_zero_state_leaf, state)
        geometry_delta0 = _float_delta_tree_like(geometry)
        geometry_leaves0, geometry_treedef = jax.tree_util.tree_flatten(
            geometry_delta0
        )
        geometry_shapes = tuple(jnp.asarray(leaf).shape for leaf in geometry_leaves0)
        geometry_sizes = tuple(int(jnp.asarray(leaf).size) for leaf in geometry_leaves0)
        geometry_flat_delta0 = jnp.concatenate(
            tuple(jnp.ravel(jnp.asarray(leaf)) for leaf in geometry_leaves0)
        )

        def _split_geometry(flat_delta):
            leaves = []
            offset = 0
            for size, shape in zip(geometry_sizes, geometry_shapes, strict=True):
                leaves.append(jnp.reshape(flat_delta[offset : offset + size], shape))
                offset += size
            return geometry_treedef.unflatten(leaves)

        def _accumulate(carry, radius_index):
            state_carry, geometry_flat_carry = carry

            def _upar_from_state_and_geometry(state_value, geometry_flat_delta):
                model = dataclasses.replace(
                    self,
                    geometry=_add_float_delta_tree(
                        geometry, _split_geometry(geometry_flat_delta)
                    ),
                )
                return model._momentum_corrected_upar_one_radius(
                    state_value, radius_index
                )

            _, pullback = jax.vjp(
                _upar_from_state_and_geometry, state, geometry_flat_delta0
            )
            local_bar = jax.lax.dynamic_index_in_dim(
                upar_bar, radius_index, axis=1, keepdims=False
            )
            state_bar, geometry_flat_bar = pullback(local_bar)
            return (
                (
                    jax.tree_util.tree_map(
                        lambda left, right: left + right, state_carry, state_bar
                    ),
                    geometry_flat_carry + geometry_flat_bar,
                ),
                None,
            )

        (state_bar, geometry_flat_bar), _ = jax.lax.scan(
            _accumulate,
            (state_bar0, jnp.zeros_like(geometry_flat_delta0)),
            radius_indices,
        )
        geometry_bar = _split_geometry(geometry_flat_bar)
        return state_bar, geometry_bar

    def pullback_momentum_corrected_upar_database_by_radius(self, state, upar_bar):
        """Return compact corrected-Upar bars for the three database tables.

        The only differentiated functions in this rule are the fixed-size
        local momentum solve and the local 5x5 moment assembly.  In
        particular, neither the radial database tables nor the NTX scan are
        differentiated by JAX here: each resulting ``Dij`` bar is scattered
        explicitly through the established interpolation stencil.
        """
        density = safe_density(state.density, self.density_floor)
        temperature = state.temperature
        density_right, density_right_grad = _extract_right_constraints(
            self.bc_density, density, self.geometry.r_grid_half
        )
        temperature_right, temperature_right_grad = _extract_right_constraints(
            self.bc_temperature, temperature, self.geometry.r_grid_half
        )
        density_right = density[:, -1] if density_right is None else density_right
        density_right_grad = (
            jnp.zeros_like(density_right)
            if density_right_grad is None
            else density_right_grad
        )
        temperature_right = (
            temperature[:, -1] if temperature_right is None else temperature_right
        )
        temperature_right_grad = (
            jnp.zeros_like(temperature_right)
            if temperature_right_grad is None
            else temperature_right_grad
        )
        dndr = jax.vmap(
            lambda values, right, right_grad: get_gradient_density(
                values, self.geometry.r_grid, self.geometry.r_grid_half, self.geometry.dr,
                right_face_constraint=right, right_face_grad_constraint=right_grad,
            )
        )(density, density_right, density_right_grad)
        dTdr = jax.vmap(
            lambda values, right, right_grad: get_gradient_temperature(
                values, self.geometry.r_grid, self.geometry.r_grid_half, self.geometry.dr,
                right_face_constraint=right, right_face_grad_constraint=right_grad,
            )
        )(temperature, temperature_right, temperature_right_grad)
        A1 = jax.vmap(
            lambda charge, density_a, temperature_a, dndr_a, dTdr_a: get_Thermodynamical_Forces_A1(
                charge, density_a, temperature_a, dndr_a, dTdr_a, state.Er
            )
        )(self.species.charge, density, temperature, dndr, dTdr)
        A2 = jax.vmap(get_Thermodynamical_Forces_A2)(temperature, dTdr)
        A3 = get_Thermodynamical_Forces_A3(state.Er)
        v_thermal = get_v_thermal(self.species.mass, temperature)
        species_indices = jnp.arange(int(self.species.number_species), dtype=jnp.int32)
        upar_bar = jnp.asarray(upar_bar, dtype=state.pressure.dtype)
        radius_indices = jnp.arange(upar_bar.shape[-1], dtype=jnp.int32)
        database = self.database
        # The established momentum-correction routine always uses its
        # default collisionality construction.  Preserve that convention here
        # rather than silently changing the bootstrap objective's physics.
        default_collisionality_kind = _collisionality_kind("default")

        def _zero_tables():
            return (
                jnp.zeros_like(database.D11_log),
                jnp.zeros_like(database.D13),
                jnp.zeros_like(database.D33),
            )

        def _local_dij_and_moments(radius_index):
            coefficient_radius = jnp.maximum(
                radius_index, jnp.asarray(1, dtype=radius_index.dtype)
            )

            def _one_species(species_index):
                vth = v_thermal[species_index, coefficient_radius]
                v_new = self.energy_grid.v_norm * vth
                nu_over_vnew = _nu_over_vnew(
                    self.species,
                    species_index,
                    v_new,
                    coefficient_radius,
                    density,
                    temperature,
                    v_thermal,
                    default_collisionality_kind,
                )
                er_over_vnew = state.Er[coefficient_radius] * 1.0e3 / v_new
                interpolation_kernel = monoenergetic_interpolation_kernel(database)
                dij = jax.vmap(
                    lambda nu_value, er_value: interpolation_kernel(
                        self.geometry.r_grid[coefficient_radius],
                        nu_value, er_value, database,
                    )
                )(nu_over_vnew, er_over_vnew)
                return dij, vth, nu_over_vnew * v_new, nu_over_vnew, er_over_vnew

            dij, vth, nu, nu_over_vnew, er_over_vnew = jax.vmap(_one_species)(
                species_indices
            )
            moments = jax.vmap(
                lambda species_index, vth_value, nu_value, nu_ratio, dij_value: assemble_momentum_lij_matrices(
                    self.species,
                    self.energy_grid,
                    species_index,
                    vth_value,
                    nu_value,
                    nu_ratio,
                    dij_value,
                )
            )(species_indices, vth, nu, nu_over_vnew, dij)
            return dij, vth, nu, nu_over_vnew, er_over_vnew, moments

        def _accumulate(carry, radius_index):
            dij, vth, nu, nu_over_vnew, er_over_vnew, moments = _local_dij_and_moments(
                radius_index
            )
            lij, eij, nu_av = moments

            def _upar_from_moments(lij_value, eij_value, nu_av_value):
                return get_momentum_Correction(
                    self.species, self.energy_grid, self.geometry, radius_index,
                    lij_value, eij_value, nu_av_value, v_thermal, density,
                    temperature, A1, A2, A3, self.species.mass,
                    self.species.charge, dndr, dTdr,
                )[2]

            _, momentum_pullback = jax.vjp(
                _upar_from_moments, lij, eij, nu_av
            )
            local_upar_bar = jax.lax.dynamic_index_in_dim(
                upar_bar, radius_index, axis=1, keepdims=False
            )
            lij_bar, eij_bar, nu_av_bar = momentum_pullback(local_upar_bar)

            def _moments_from_dij(dij_value):
                return jax.vmap(
                    lambda species_index, vth_value, nu_value, nu_ratio, local_dij: assemble_momentum_lij_matrices(
                        self.species, self.energy_grid, species_index, vth_value,
                        nu_value, nu_ratio, local_dij,
                    )
                )(species_indices, vth, nu, nu_over_vnew, dij_value)

            _, dij_pullback = jax.vjp(_moments_from_dij, dij)
            (dij_bar,) = dij_pullback((lij_bar, eij_bar, nu_av_bar))

            def _scatter_table(table_index, table):
                def _one_species(species_index, nu_ratio, er_ratio, local_bar):
                    def _one_energy(nu_value, er_value, bar_value):
                        if monoenergetic_database_kind(database) == MONOENERGETIC_KIND_GENERIC:
                            return monoenergetic_interpolation_table_bar(
                                self.geometry.r_grid[jnp.maximum(radius_index, 1)],
                                nu_value, er_value, bar_value, table, database,
                            )
                        stencil = radial_preprocessed_interpolation_stencil(
                            self.geometry.r_grid[jnp.maximum(radius_index, 1)],
                            nu_value,
                            er_value,
                            database,
                        )
                        return radial_preprocessed_interpolation_table_bar(
                            stencil, bar_value, table
                        )
                    return jnp.sum(
                        jax.vmap(_one_energy)(nu_ratio, er_ratio, local_bar), axis=0
                    )
                return jnp.sum(
                    jax.vmap(_one_species)(species_indices, nu_over_vnew, er_over_vnew, dij_bar[:, :, table_index]),
                    axis=0,
                )

            d11_bar = _scatter_table(0, database.D11_log)
            d13_bar = _scatter_table(1, database.D13)
            d33_bar = _scatter_table(2, database.D33)
            return tuple(left + right for left, right in zip(carry, (d11_bar, d13_bar, d33_bar), strict=True)), None

        table_bars, _ = jax.lax.scan(_accumulate, _zero_tables(), radius_indices)
        return table_bars

    def pullback_momentum_corrected_upar_geometry_by_radius(
        self, state, upar_bar, geometry
    ):
        """Transpose corrected-Upar to geometry with the database held fixed.

        This is the black-box-database counterpart of the compact realtime-Lij
        bootstrap geometry boundary.  It differentiates one local momentum
        solve per radius, never rebuilds a runtime, regenerates an NTX scan,
        or traces the complete all-radii corrected-flux evaluation.
        """

        upar_bar = jnp.asarray(upar_bar, dtype=state.pressure.dtype)
        radius_indices = jnp.arange(upar_bar.shape[-1], dtype=jnp.int32)
        geometry_delta0 = _float_delta_tree_like(geometry)
        geometry_delta_leaves0, geometry_delta_treedef = jax.tree_util.tree_flatten(
            geometry_delta0
        )
        geometry_delta_shapes = tuple(
            jnp.asarray(leaf).shape for leaf in geometry_delta_leaves0
        )
        geometry_delta_sizes = tuple(
            int(jnp.asarray(leaf).size) for leaf in geometry_delta_leaves0
        )
        flat_delta0 = jnp.concatenate(
            tuple(jnp.ravel(jnp.asarray(leaf)) for leaf in geometry_delta_leaves0)
        )

        def _split_flat_geometry(flat_delta):
            leaves = []
            offset = 0
            for size, shape in zip(
                geometry_delta_sizes, geometry_delta_shapes, strict=True
            ):
                leaves.append(jnp.reshape(flat_delta[offset : offset + size], shape))
                offset += size
            return geometry_delta_treedef.unflatten(leaves)

        def _accumulate(flat_carry, radius_index):
            def _upar_from_geometry_delta(flat_delta):
                geometry_delta = _split_flat_geometry(flat_delta)
                model = dataclasses.replace(
                    self,
                    geometry=_add_float_delta_tree(geometry, geometry_delta),
                )
                return model._momentum_corrected_upar_one_radius(state, radius_index)

            _, pullback = jax.vjp(_upar_from_geometry_delta, flat_delta0)
            local_bar = jax.lax.dynamic_index_in_dim(
                upar_bar, radius_index, axis=1, keepdims=False
            )
            (flat_bar,) = pullback(local_bar)
            return flat_carry + flat_bar, None

        geometry_flat_bar, _ = jax.lax.scan(
            _accumulate, jnp.zeros_like(flat_delta0), radius_indices
        )
        return _split_flat_geometry(geometry_flat_bar)

    def pullback_local_particle_flux_geometry_by_radius(
        self, state, er_profile, residual_bars, geometry
    ):
        """Transpose selected-root charge residuals to fixed-database geometry.

        The selected-root lane needs only the local charge-weighted particle
        flux at each radius.  Keep the interpolation tables fixed and perform
        one local geometry VJP per radius; the NTX scan is deliberately not
        part of this boundary.
        """

        er_profile = jnp.asarray(er_profile, dtype=state.Er.dtype)
        residual_bars = jnp.asarray(residual_bars, dtype=state.Er.dtype)
        if residual_bars.ndim != 2 or residual_bars.shape[1] != er_profile.shape[0]:
            raise ValueError(
                "Local particle-flux geometry pullback expects residual_bars "
                "with shape (objective_count, radial_count)."
            )
        state_with_er = dataclasses.replace(state, Er=er_profile)
        # ``initial_er_charge_flux_residuals`` contracts Gamma with the
        # dimensionless species charges (``charge_qp``), not the physical
        # Coulomb-valued ``charge`` used inside thermodynamic forces.  The
        # local geometry transpose must use that same residual weight.
        charge = jnp.asarray(self.species.charge_qp, dtype=state.Er.dtype)
        radius_indices = jnp.arange(er_profile.shape[0], dtype=jnp.int32)
        geometry_delta0 = _float_delta_tree_like(geometry)
        delta_leaves, delta_treedef = jax.tree_util.tree_flatten(geometry_delta0)
        delta_shapes = tuple(jnp.asarray(leaf).shape for leaf in delta_leaves)
        delta_sizes = tuple(int(jnp.asarray(leaf).size) for leaf in delta_leaves)
        flat_delta0 = jnp.concatenate(
            tuple(jnp.ravel(jnp.asarray(leaf)) for leaf in delta_leaves)
        )

        def _split(flat_delta):
            leaves = []
            offset = 0
            for size, shape in zip(delta_sizes, delta_shapes, strict=True):
                leaves.append(jnp.reshape(flat_delta[offset : offset + size], shape))
                offset += size
            return delta_treedef.unflatten(leaves)

        def _accumulate(flat_carry, radius_index):
            def _local_residual(flat_delta):
                model = dataclasses.replace(
                    self,
                    geometry=_add_float_delta_tree(geometry, _split(flat_delta)),
                )
                gamma = model.build_local_particle_flux_evaluator(state_with_er)(
                    radius_index,
                    er_profile[radius_index],
                )
                return jnp.sum(charge * gamma)

            _, pullback = jax.vjp(_local_residual, flat_delta0)
            local_bars = jax.vmap(lambda bar: pullback(bar)[0])(
                residual_bars[:, radius_index]
            )
            return flat_carry + local_bars, None

        flat_bars, _ = jax.lax.scan(
            _accumulate,
            jnp.zeros((residual_bars.shape[0], flat_delta0.size), dtype=flat_delta0.dtype),
            radius_indices,
        )
        return delta_treedef.unflatten(
            tuple(
                jnp.reshape(
                    flat_bars[:, sum(delta_sizes[:index]) : sum(delta_sizes[: index + 1])],
                    (flat_bars.shape[0],) + shape,
                )
                for index, shape in enumerate(delta_shapes)
            )
        )

    def build_local_direct_flux_evaluator(self, state):
        """Return the direct database flux triplet at one centre radius.

        This is the compact primal shared by selected-root and direct-RHS
        geometry transposes.  It is the local restriction of the direct
        centre database flux evaluation: gradients are shared over the state,
        while interpolation and Lij assembly are only performed at the
        requested radius.
        """
        species = self.species
        energy_grid = self.energy_grid
        geometry = self.geometry
        database = self.database
        density = safe_density(state.density, self.density_floor)
        temperature = state.temperature
        # This primitive must be exactly the local restriction of ``self(state)``.
        # The direct centre database evaluator deliberately uses its established
        # zero-gradient, last-cell right closure; it does *not* use the transport
        # face boundary conditions.  Feeding the face closure here made the last
        # cell differ (by a factor of two in the two-cell regression) from the
        # full direct database flux.  Keep these values in lockstep with
        # ``_get_Neoclassical_Fluxes_generic``.
        density_right_constraint = density[:, -1]
        density_right_grad_constraint = jnp.zeros_like(density_right_constraint)
        temperature_right_constraint = temperature[:, -1]
        temperature_right_grad_constraint = jnp.zeros_like(temperature_right_constraint)
        v_thermal = get_v_thermal(species.mass, temperature)
        collisionality_kind = _collisionality_kind(self.collisionality_model)
        dndr = jax.vmap(
            lambda density_a, n_rc, n_rg: get_gradient_density(
                density_a,
                geometry.r_grid,
                geometry.r_grid_half,
                geometry.dr,
                right_face_constraint=n_rc,
                right_face_grad_constraint=n_rg,
            )
        )(
            density,
            density_right_constraint,
            density_right_grad_constraint,
        )
        dtdr = jax.vmap(
            lambda temperature_a, t_rc, t_rg: get_gradient_temperature(
                temperature_a,
                geometry.r_grid,
                geometry.r_grid_half,
                geometry.dr,
                right_face_constraint=t_rc,
                right_face_grad_constraint=t_rg,
            )
        )(
            temperature,
            temperature_right_constraint,
            temperature_right_grad_constraint,
        )
        species_indices = jnp.asarray(species.species_indices, dtype=jnp.int32)

        def evaluator(radius_index, er_value):
            radius_index = jnp.asarray(radius_index, dtype=jnp.int32)
            er_scalar = jnp.asarray(er_value, dtype=state.Er.dtype)
            lij = jax.vmap(
                lambda species_index: get_Lij_matrix_local(
                    species,
                    energy_grid,
                    geometry,
                    database,
                    species_index,
                    radius_index,
                    er_scalar,
                    temperature,
                    density,
                    v_thermal,
                    collisionality_kind,
                )
            )(species_indices)
            density_local = jax.lax.dynamic_index_in_dim(
                density, radius_index, axis=1, keepdims=False
            )
            temperature_local = jax.lax.dynamic_index_in_dim(
                temperature, radius_index, axis=1, keepdims=False
            )
            dndr_local = jax.lax.dynamic_index_in_dim(
                dndr, radius_index, axis=1, keepdims=False
            )
            dtdr_local = jax.lax.dynamic_index_in_dim(
                dtdr, radius_index, axis=1, keepdims=False
            )
            a1 = jax.vmap(get_Thermodynamical_Forces_A1)(
                species.charge,
                density_local,
                temperature_local,
                dndr_local,
                dtdr_local,
                jnp.broadcast_to(er_scalar, density_local.shape),
            )
            a2 = jax.vmap(get_Thermodynamical_Forces_A2)(
                temperature_local, dtdr_local
            )
            a3 = get_Thermodynamical_Forces_A3(
                jnp.reshape(er_scalar, (1,))
            )[0]
            density_phys = DENSITY_STATE_TO_PHYSICAL * density_local
            temperature_phys = TEMPERATURE_STATE_TO_PHYSICAL * temperature_local
            return {
                "Gamma": -density_phys * (
                    lij[:, 0, 0] * a1 + lij[:, 0, 1] * a2 + lij[:, 0, 2] * a3
                ),
                "Q": -temperature_phys * density_phys * (
                    lij[:, 1, 0] * a1 + lij[:, 1, 1] * a2 + lij[:, 1, 2] * a3
                ),
                "Upar": -density_phys * (
                    lij[:, 2, 0] * a1 + lij[:, 2, 1] * a2 + lij[:, 2, 2] * a3
                ),
            }

        return evaluator

    def build_local_particle_flux_evaluator(self, state):
        """Return the particle component of the direct local flux primitive."""

        local_fluxes = self.build_local_direct_flux_evaluator(state)
        return lambda radius_index, er_value: local_fluxes(radius_index, er_value)["Gamma"]

    def evaluate_face_fluxes(self, state, face_state, **kwargs):
        evaluated = kwargs.get("evaluated_state")
        if evaluated is None:
            evaluated = build_evaluated_transport_state(
                state,
                self.geometry,
                bc_density=kwargs.get("bc_density", self.bc_density),
                bc_temperature=kwargs.get("bc_temperature", self.bc_temperature),
                density_floor=self.density_floor,
            )
        face_density = safe_density(face_state.density, self.density_floor)
        particle_face_closure_mode = str(kwargs.get("particle_face_closure_mode", "reconstructed")).strip().lower()
        if particle_face_closure_mode in {"ntss_like", "ntss", "half_point"}:
            dndr_faces = _ntss_like_face_gradient(
                evaluated.center.density,
                self.geometry.r_grid_half,
                bc_model=kwargs.get("bc_density", self.bc_density),
            )
            dTdr_faces = _ntss_like_face_gradient(
                evaluated.center.temperature,
                self.geometry.r_grid_half,
                bc_model=kwargs.get("bc_temperature", self.bc_temperature),
            )
        else:
            dndr_faces = evaluated.density_grad_face
            dTdr_faces = evaluated.temperature_grad_face
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

    def _fluxes_from_face_fluxes(self, face_fluxes):
        return {
            "Gamma": jax.vmap(cell_centered_from_faces)(face_fluxes["Gamma"]),
            "Q": jax.vmap(cell_centered_from_faces)(face_fluxes["Q"]),
            "Upar": jax.vmap(cell_centered_from_faces)(face_fluxes["Upar"]),
            "Gamma_faces": face_fluxes["Gamma"],
            "Q_faces": face_fluxes["Q"],
            "Upar_faces": face_fluxes["Upar"],
        }

    def build_lagged_response(self, state, **kwargs):
        del kwargs
        face_state = build_face_transport_state(
            state,
            self.geometry,
            bc_density=self.bc_density,
            bc_temperature=self.bc_temperature,
        )
        return FaceJVPTransportFluxResponse(
            reference_state=state,
            reference_face_flux=self.evaluate_face_fluxes(
                state,
                face_state,
                bc_density=self.bc_density,
                bc_temperature=self.bc_temperature,
            ),
            reference_flux=self(state),
        )

    def pullback_build_lagged_response(self, state, lagged_response_bar, **kwargs):
        self._require_supported_reverse_lagged_response()
        return super().pullback_build_lagged_response(state, lagged_response_bar, **kwargs)

    def evaluate_with_lagged_response(self, state, lagged_response, **kwargs):
        del kwargs
        delta_state = jax.tree_util.tree_map(
            lambda current, reference: current - reference,
            state,
            lagged_response.reference_state,
        )

        def _face_fluxes_from_state(state_value):
            face_state_value = build_face_transport_state(
                state_value,
                self.geometry,
                bc_density=self.bc_density,
                bc_temperature=self.bc_temperature,
            )
            return self.evaluate_face_fluxes(
                state_value,
                face_state_value,
                bc_density=self.bc_density,
                bc_temperature=self.bc_temperature,
            )

        order = int(self.lagged_response_taylor_order)
        tangent_face_flux, curvature_face_flux, cubic_face_flux = self._anchored_taylor_terms(
            _face_fluxes_from_state,
            lagged_response.reference_state,
            delta_state,
            order,
        )
        face_fluxes = self._add_taylor_terms(
            lagged_response.reference_face_flux,
            tangent_face_flux,
            curvature_face_flux,
            cubic_face_flux,
        )
        out = self._fluxes_from_face_fluxes(face_fluxes)
        if lagged_response.reference_flux is not None:
            tangent_flux, curvature_flux, cubic_flux = self._anchored_taylor_terms(
                self.__call__,
                lagged_response.reference_state,
                delta_state,
                order,
            )
            out.update(
                self._add_taylor_terms(
                    lagged_response.reference_flux,
                    tangent_flux,
                    curvature_flux,
                    cubic_flux,
                )
            )
        return out


def _as_float_array(value, *, name: str, positive: bool = False) -> jax.Array:
    arr = jnp.asarray(value, dtype=jnp.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional list/array.")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one value.")
    # Axis configuration is validated eagerly at construction time.  The same
    # arrays become tracers when the recorded database is an explicit support
    # leaf inside a JAX VJP, where Python ``bool`` conversion is invalid.
    # Shapes remain statically checked above; defer value validation in traced
    # execution rather than changing the numerical axis conversion.
    if not isinstance(arr, jax.core.Tracer):
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
    if derivative_mode != "direct":
        raise ValueError("ntx_exact_derivative_mode must be 'direct'.")
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


def _ntx_runtime_scan_to_neopax_monoenergetic(scan, *, a_b):
    rho = jnp.asarray(scan.rho, dtype=jnp.float64)
    nu_v = jnp.asarray(scan.nu_v, dtype=jnp.float64)
    er = jnp.asarray(scan.Er, dtype=jnp.float64)
    drds = jnp.asarray(scan.drds, dtype=jnp.float64)
    d11 = jnp.asarray(scan.D11, dtype=jnp.float64) * drds[:, None, None] ** 2
    d13 = jnp.asarray(scan.D13, dtype=jnp.float64) * drds[:, None, None]
    d33 = jnp.asarray(scan.D33, dtype=jnp.float64) * nu_v[None, :, None]
    a_b_value = jnp.asarray(a_b, dtype=jnp.float64)
    radius = a_b_value * rho[:, None]
    er_list = jnp.log10(jnp.maximum(1.0e-8, jnp.abs(er) / radius))
    d11 = jnp.where(d11 > D11_POSITIVE_FLOOR, d11, D11_POSITIVE_FLOOR)
    return Monoenergetic(
        a_b=a_b_value,
        rho=rho,
        nu_log=jnp.log10(nu_v),
        Er_list=er_list,
        D11_log=jnp.log10(d11),
        D13=d13,
        D33=d33,
    )


def _ntx_runtime_scan_with_live_channels(scan, *, channels, er_tilde):
    """Attach live geometry channels to an NTX coefficient scan.

    This deliberately contains no NTX solve.  Keeping it as a pure array
    mapping lets the recorded database transpose recover every channel bar
    after the one retained coefficient pullback.
    """
    rho = jnp.asarray(scan.rho, dtype=jnp.float64)
    _, _, er_to_ertilde = _build_ntx_field_channels(rho, er_tilde, channels)
    return dataclasses.replace(
        scan,
        Er_tilde=er_tilde,
        Er_to_Ertilde=er_to_ertilde,
        dr_tildedr=jnp.asarray(channels["dr_tildedr"], dtype=jnp.float64),
        dr_tildeds=jnp.asarray(channels["dr_tildeds"], dtype=jnp.float64),
        a_b=jnp.asarray(channels["a_b"], dtype=jnp.float64),
        psia=jnp.asarray(channels["psia"], dtype=jnp.float64),
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
            # These are Python floats for the file-backed model but traced
            # scalars for a live VMEC payload.  Do not concretize them here:
            # the scan database must remain connected to geometry.
            a_b=jnp.asarray(channels["a_b"], dtype=jnp.float64),
            psia=jnp.asarray(channels["psia"], dtype=jnp.float64),
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
            bar_arr = jnp.asarray(bar_leaf)
            if bar_arr.dtype == jax.dtypes.float0:
                return jnp.zeros_like(arr)
            return jnp.asarray(bar_leaf, dtype=arr.dtype)
        return jnp.zeros(arr.shape, dtype=jnp.float64)

    return jax.tree_util.tree_map(_sanitize_leaf, primal_tree, bar_tree)


def _sum_float_delta_bar_trees(primal_tree, *bar_trees):
    """Add cotangent trees after converting ``float0`` leaves to zero.

    Integer/static leaves of a prepared NTX system have no tangent space, so
    JAX represents their VJP result as ``float0``.  They must be made regular
    zero leaves before a joint pullback combines several derivative paths.

    ``bar_trees`` here come from an energy ``lax.map`` and therefore have a
    leading energy axis.  In particular, the zero substituted for a static
    leaf must retain the *bar* shape, not the primal leaf shape; otherwise
    rebuilding NTX's validated geometry dataclass mixes batched Fourier
    coefficients with an unbatched mode array.
    """
    def _sanitize_batched_leaf(primal_leaf, bar_leaf):
        primal_arr = jnp.asarray(primal_leaf)
        bar_arr = jnp.asarray(bar_leaf)
        dtype = primal_arr.dtype if jnp.issubdtype(primal_arr.dtype, jnp.inexact) else jnp.float64
        if bar_arr.dtype == jax.dtypes.float0:
            return jnp.zeros(bar_arr.shape, dtype=dtype)
        if jnp.issubdtype(primal_arr.dtype, jnp.inexact):
            return jnp.asarray(bar_leaf, dtype=primal_arr.dtype)
        return jnp.zeros(bar_arr.shape, dtype=dtype)

    def _sanitize_batched_tree(bar_tree):
        return jax.tree_util.tree_map(_sanitize_batched_leaf, primal_tree, bar_tree)

    if not bar_trees:
        return _float_delta_tree_like(primal_tree)
    total = _sanitize_batched_tree(bar_trees[0])
    for bar_tree in bar_trees[1:]:
        total = jax.tree_util.tree_map(
            lambda left, right: left + right,
            total,
            _sanitize_batched_tree(bar_tree),
        )
    return total


def _face_flux_bar_with_interpolated_center_bars(face_output, flux_bar):
    def _is_scalar_bar(value) -> bool:
        arr = jnp.asarray(value)
        return arr.shape == ()

    def _bar_or_zeros_like(value, template):
        if value is None or _is_scalar_bar(value):
            return jnp.zeros_like(template)
        arr = jnp.asarray(value)
        if arr.dtype == jax.dtypes.float0:
            return jnp.zeros_like(template)
        return jnp.asarray(value, dtype=template.dtype)

    def _face_bar_from_center_key(center_key, face_key):
        face_bar = _bar_or_zeros_like(flux_bar.get(face_key, None), face_output[face_key])
        center_bar = flux_bar.get(center_key, None)
        if center_bar is None or _is_scalar_bar(center_bar):
            return face_bar
        center_template = jax.vmap(cell_centered_from_faces)(face_output[face_key])
        if jnp.asarray(center_bar).dtype == jax.dtypes.float0:
            return face_bar
        center_bar = jnp.asarray(
            center_bar,
            dtype=center_template.dtype,
        )
        (center_to_face_bar,) = jax.linear_transpose(
            lambda faces_value: jax.vmap(cell_centered_from_faces)(faces_value),
            face_output[face_key],
        )(center_bar)
        return face_bar + center_to_face_bar

    return {
        "Gamma_faces": _face_bar_from_center_key("Gamma", "Gamma_faces"),
        "Q_faces": _face_bar_from_center_key("Q", "Q_faces"),
        "Upar_faces": _face_bar_from_center_key("Upar", "Upar_faces"),
    }


def _debug_flux_bar_tree_mismatch(context, flux_output, flux_bar):
    if not _reverse_tree_debug_enabled():
        return
    if not isinstance(flux_output, dict) or not isinstance(flux_bar, dict):
        return
    output_keys = tuple(flux_output.keys())
    bar_keys = tuple(flux_bar.keys())
    missing_keys = tuple(key for key in output_keys if key not in flux_bar)
    extra_keys = tuple(key for key in bar_keys if key not in flux_output)
    scalar_or_float0_keys = []
    for key in output_keys:
        value = flux_bar.get(key, None)
        if value is None:
            continue
        arr = jnp.asarray(value)
        if arr.shape == () or arr.dtype == jax.dtypes.float0:
            scalar_or_float0_keys.append(key)
    if missing_keys or extra_keys or scalar_or_float0_keys:
        output_shapes = {
            key: tuple(jnp.asarray(value).shape)
            for key, value in flux_output.items()
        }
        bar_shapes = {
            key: tuple(jnp.asarray(value).shape)
            for key, value in flux_bar.items()
            if value is not None
        }
        print(
            "[reverse-flux-bar-tree] "
            f"context={context} output_keys={output_keys} bar_keys={bar_keys} "
            f"missing_output_bars={missing_keys} extra_bars={extra_keys} "
            f"scalar_or_float0_bars={tuple(scalar_or_float0_keys)} "
            f"output_shapes={output_shapes} bar_shapes={bar_shapes}",
            flush=True,
        )


def _complete_flux_bar_like(flux_output, flux_bar, *, context=None):
    if not isinstance(flux_output, dict) or not isinstance(flux_bar, dict):
        return flux_bar
    if context is not None:
        _debug_flux_bar_tree_mismatch(context, flux_output, flux_bar)

    def _bar_or_zeros(key, template):
        value = flux_bar.get(key, None)
        if value is None:
            return jnp.zeros_like(template)
        arr = jnp.asarray(value)
        if arr.shape == () or arr.dtype == jax.dtypes.float0:
            return jnp.zeros_like(template)
        return jnp.asarray(value, dtype=jnp.asarray(template).dtype)

    return {key: _bar_or_zeros(key, value) for key, value in flux_output.items()}


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


def _support_with_face_delta(support, face_channels_delta, face_prepared_delta):
    return dataclasses.replace(
        support,
        face_channels=_add_float_delta_tree(support.face_channels, face_channels_delta),
        face_prepared=_add_float_delta_tree(support.face_prepared, face_prepared_delta),
    )


def _support_with_face_channel_delta(support, face_channels_delta):
    return dataclasses.replace(
        support,
        face_channels=_add_float_delta_tree(support.face_channels, face_channels_delta),
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


def _support_bar_from_face_bars(support, face_channels_bar, face_prepared_bar):
    return dataclasses.replace(
        _float_delta_tree_like(support),
        face_channels=_sanitize_float_delta_bar_tree(support.face_channels, face_channels_bar),
        face_prepared=_sanitize_float_delta_bar_tree(support.face_prepared, face_prepared_bar),
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
    if rho_arr.shape[0] >= 2 and np.isclose(float(rho_arr[0]), 0.0):
        # NTX/VMEC transport normalization is singular exactly on the magnetic
        # axis. Downstream coefficient maps already regularize axis values from
        # neighboring radii, so use the first non-axis surface as a placeholder.
        axis_placeholder_rho = float(rho_arr[1])
        return tuple(
            loader(axis_placeholder_rho if index == 0 else float(rho_value))
            for index, rho_value in enumerate(rho_arr)
        )
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
    scan_surfaces: tuple[Any, ...] | None = None
    database: Any = None
    lagged_response_taylor_order: int = 1
    coefficient_reverse_mode: str = "generic"
    record_scan_primal: bool = False
    scan_primal_record: Any = None
    scan_primal: Any = None

    def __post_init__(self):
        order = int(self.lagged_response_taylor_order)
        if order not in {1, 2, 3}:
            raise ValueError(
                "ntx_scan_runtime lagged_response_taylor_order must be 1, 2, or 3."
            )
        if self.coefficient_reverse_mode not in {"generic", "structured"}:
            raise ValueError(
                "ntx_scan_runtime coefficient_reverse_mode must be 'generic' or "
                "'structured'."
            )
        if self.record_scan_primal and self.coefficient_reverse_mode != "structured":
            raise ValueError(
                "ntx_scan_runtime record_scan_primal requires coefficient_reverse_mode='structured'."
            )

    def with_runtime_scan_payload(
        self,
        *,
        geometry,
        channels: NTXRuntimeScanChannels,
        scan_surfaces: tuple[Any, ...],
        database=None,
    ) -> "NTXRuntimeScanTransportModel":
        """Replace the complete live NTX-scan input payload.

        Realtime VMEC supplies all three coupled inputs: transport geometry,
        scan channels, and the corresponding traceable NTX surfaces.  A cached
        database is valid only for that same triple.  Passing ``database=None``
        intentionally clears it, so the existing ``with_runtime_database``
        route rebuilds it through the live NTX scan rather than a file-backed
        database loader.
        """

        rho, _, _ = self._scan_axes()
        if channels.rho.shape != rho.shape:
            raise ValueError("Runtime scan payload channel rho grid does not match rho_scan.")
        if len(scan_surfaces) != int(rho.shape[0]):
            raise ValueError("Runtime scan payload surface count does not match rho_scan.")
        return dataclasses.replace(
            self,
            geometry=geometry,
            channels=channels,
            scan_surfaces=tuple(scan_surfaces),
            database=database,
            scan_primal_record=None,
            scan_primal=None,
            vmec_file=None,
            boozer_file=None,
        )

    def with_support_payload(self, support_payload):
        """Rebuild this model from a differentiable live-scan payload.

        Unlike the file-backed database model, the database is intentionally
        *not* an independent support leaf here.  The payload supplies the
        realtime VMEC geometry, live scan channels, and live scan surfaces;
        the existing NTX scan builder regenerates the database from those
        inputs.  This is the same construction used by the forward runtime.
        """

        if not isinstance(support_payload, dict):
            raise TypeError("Live NTX scan support payload must be a mapping.")
        required = ("geometry", "channels", "surfaces")
        missing = tuple(name for name in required if name not in support_payload)
        if missing:
            raise ValueError(f"Live NTX scan support payload is missing {missing!r}.")
        model = self.with_runtime_scan_payload(
            geometry=support_payload["geometry"],
            channels=support_payload["channels"],
            scan_surfaces=support_payload["surfaces"],
            database=support_payload.get("database"),
        )
        # Recorded reverse passes an explicit database leaf.  Its derivative
        # is accumulated by the caller and transposed once through the saved
        # scan primal after the segment sweep.  The established payload has no
        # such leaf and therefore retains its rebuild behaviour unchanged.
        return model if "database" in support_payload else model.with_runtime_database()

    def pullback_build_lagged_response_support_payload(
        self,
        state,
        lagged_response_bar,
        support_payload,
        **kwargs,
    ):
        """VJP through the live NTX scan/database rebuild for one Radau bar."""

        support_delta0 = _float_delta_tree_like(support_payload)
        _, pullback = jax.vjp(
            lambda support_delta: self.with_support_payload(
                _add_float_delta_tree(support_payload, support_delta)
            ).build_lagged_response(state, **kwargs),
            support_delta0,
        )
        (support_bar,) = pullback(lagged_response_bar)
        return _sanitize_float_delta_bar_tree(support_payload, support_bar)

    def _scan_axes(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        rho = _as_float_array(self.rho_scan, name="rho_scan")
        nu_v = _as_float_array(self.nu_v_scan, name="nu_v_scan", positive=True)
        er_tilde = _as_float_array(self.er_tilde_scan, name="er_tilde_scan")
        if not isinstance(rho, jax.core.Tracer):
            if not bool(jnp.all((rho > 0.0) & (rho <= 1.0))):
                raise ValueError("rho_scan values must satisfy 0 < rho <= 1.")
        return rho, nu_v, er_tilde

    def _static_channels(self) -> NTXRuntimeScanChannels:
        rho, _, _ = self._scan_axes()
        if self.channels is not None:
            if self.channels.rho.shape != rho.shape:
                raise ValueError("Provided ntx_scan_channels rho grid does not match rho_scan.")
            if (
                not isinstance(rho, jax.core.Tracer)
                and not isinstance(self.channels.rho, jax.core.Tracer)
                and not bool(jnp.allclose(self.channels.rho, rho))
            ):
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

    def _scan_surfaces(self, ntx, rho):
        """Return supplied live surfaces or the legacy file-backed surfaces.

        The explicit-surface route is deliberately a narrow forward boundary:
        it lets realtime VMEC provide the already-built scan surfaces without
        changing the file-backed runtime-database behaviour.
        """
        if self.scan_surfaces is not None:
            if len(self.scan_surfaces) != int(rho.shape[0]):
                raise ValueError("Provided ntx_scan_surfaces length does not match rho_scan.")
            return self.scan_surfaces
        loader = self._surface_loader(ntx)
        return tuple(loader(float(rho_value)) for rho_value in rho)

    def _build_runtime_database_and_record(self):
        if self.database is not None:
            return self.database, self.scan_primal_record, self.scan_primal

        ntx = _import_ntx()
        rho, nu_v, er_tilde = self._scan_axes()
        static_channels = self._static_channels()
        channels = static_channels.as_mapping()
        er, es, _ = _build_ntx_field_channels(rho, er_tilde, channels)
        grid = ntx.GridSpec(
            n_theta=int(self.n_theta),
            n_zeta=int(self.n_zeta),
            n_xi=int(self.n_xi),
        )
        scan_result = ntx.build_ntx_neopax_scan_from_surfaces(
            self._scan_surfaces(ntx, rho),
            rho=rho,
            nu_v=nu_v,
            Es=es,
            Er=er,
            drds=jnp.asarray(channels["drds"], dtype=jnp.float64),
            grid=grid,
            source_name=self.source_name,
            coefficient_reverse_mode=self.coefficient_reverse_mode,
            return_primal_record=bool(self.record_scan_primal),
        )
        if self.record_scan_primal:
            raw_scan, scan_primal_record = scan_result
        else:
            raw_scan = scan_result
            scan_primal_record = None
        scan = _ntx_runtime_scan_with_live_channels(
            raw_scan, channels=channels, er_tilde=er_tilde,
        )
        print(
            "[NEOPAX] built runtime NTX scan database: "
            f"rho={int(rho.shape[0])} nu_v={int(nu_v.shape[0])} "
            f"Er_tilde={int(er_tilde.shape[0])} "
            f"grid=({grid.n_theta},{grid.n_zeta},{grid.n_xi}) backend={str(self.surface_backend).strip().lower()}"
        )
        return (
            _ntx_runtime_scan_to_neopax_monoenergetic(
                scan,
                a_b=jnp.asarray(channels["a_b"], dtype=jnp.float64),
            ),
            scan_primal_record,
            raw_scan if self.record_scan_primal else None,
        )

    def _build_runtime_database(self):
        return self._build_runtime_database_and_record()[0]

    def pullback_recorded_runtime_database(self, database_bar):
        """Transpose a database cotangent through a retained structured scan primal.

        This deliberately stops at the live scan inputs.  The caller still
        owns the direct transport-geometry/channel bars and combines them
        with the returned scan/surface bars.  Crucially, no scan database is
        rebuilt here.
        """

        if self.scan_primal_record is None or self.scan_primal is None:
            raise ValueError(
                "Recorded runtime scan primal is unavailable; rebuild with "
                "record_scan_primal=True and coefficient_reverse_mode='structured'."
            )
        ntx = _import_ntx()
        channels = self._static_channels()
        rho, _, er_tilde = self._scan_axes()
        del rho
        _, conversion_pullback = jax.vjp(
            lambda raw_scan_value, channel_value: _ntx_runtime_scan_to_neopax_monoenergetic(
                _ntx_runtime_scan_with_live_channels(
                    raw_scan_value,
                    channels=channel_value.as_mapping(),
                    er_tilde=er_tilde,
                ),
                a_b=channel_value.a_b,
            ),
            self.scan_primal,
            channels,
        )
        scan_bar, channels_bar = conversion_pullback(database_bar)
        blocks_bar = ntx.NeopaxScanCoefficientBlocks(
            D11=scan_bar.D11,
            D13=scan_bar.D13,
            D33=scan_bar.D33,
            D33_spitzer=scan_bar.D33_spitzer,
            b00=scan_bar.b00,
            boozer_i=scan_bar.boozer_i,
            boozer_g=scan_bar.boozer_g,
            iota=scan_bar.iota,
            fac_reference_to_sfincs_11=scan_bar.fac_reference_to_sfincs_11,
            fac_reference_to_sfincs_31=scan_bar.fac_reference_to_sfincs_31,
            fac_reference_to_sfincs_33=scan_bar.fac_reference_to_sfincs_33,
            fac_sfincs_to_dkes_11=scan_bar.fac_sfincs_to_dkes_11,
            fac_sfincs_to_dkes_31=scan_bar.fac_sfincs_to_dkes_31,
            fac_sfincs_to_dkes_33=scan_bar.fac_sfincs_to_dkes_33,
        )
        surface_bars, es_bar = (
            ntx.pullback_neopax_scan_coefficient_blocks_from_primal_record(
                self.scan_primal_record,
                coefficient_blocks_bar=blocks_bar,
            )
        )
        # ``Er``, ``Es`` and ``drds`` are direct scan inputs, not coefficient
        # outputs.  Their bars from the conversion must still be returned to
        # the live channels, without replaying the coefficient scan.
        def _raw_scan_inputs_from_channels(channel_value):
            channel_mapping = channel_value.as_mapping()
            er_value, es_value, _ = _build_ntx_field_channels(
                jnp.asarray(self.scan_primal.rho, dtype=jnp.float64),
                er_tilde,
                channel_mapping,
            )
            return dataclasses.replace(
                self.scan_primal,
                Er=er_value,
                Es=es_value,
                drds=jnp.asarray(channel_mapping["drds"], dtype=jnp.float64),
            )

        scan_input_bar = dataclasses.replace(
            scan_bar,
            Es=jnp.asarray(scan_bar.Es) + jnp.asarray(es_bar),
        )
        _, input_channel_pullback = jax.vjp(
            _raw_scan_inputs_from_channels, channels,
        )
        (input_channels_bar,) = input_channel_pullback(scan_input_bar)
        channels_bar = jax.tree_util.tree_map(
            lambda lhs, rhs: jnp.asarray(lhs) + jnp.asarray(rhs),
            channels_bar,
            input_channels_bar,
        )
        return scan_bar, surface_bars, es_bar, channels_bar

    def recorded_runtime_database_support_bar(self, database_bar):
        """Map one accumulated database cotangent to live scan support bars.

        The returned tree intentionally matches the existing realtime scan
        payload.  The caller adds it to the direct per-step channel/surface
        bars before invoking the unchanged VMEC payload transpose.
        """
        _, surface_bars, _, channels_bar = self.pullback_recorded_runtime_database(
            database_bar
        )
        return {
            "channels": channels_bar,
            "surfaces": surface_bars,
        }

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
        new_scan_surfaces = self.scan_surfaces
        if rho_scan is not None:
            old_rho = _as_float_array(self.rho_scan, name="rho_scan")
            candidate_rho = _as_float_array(new_rho, name="rho_scan")
            same_rho = old_rho.shape == candidate_rho.shape and bool(jnp.allclose(old_rho, candidate_rho))
            if not same_rho:
                if self.channels is not None:
                    new_channels = None
                # Explicit live surfaces are paired one-for-one with the scan
                # radii; never silently reuse them on a changed axis.
                new_scan_surfaces = None

        return dataclasses.replace(
            self,
            rho_scan=new_rho,
            nu_v_scan=new_nu_v,
            er_tilde_scan=new_er_tilde,
            channels=new_channels,
            scan_surfaces=new_scan_surfaces,
            database=None if clear_database else self.database,
            scan_primal_record=None if clear_database else self.scan_primal_record,
            scan_primal=None if clear_database else self.scan_primal,
        )

    def _database_model(self) -> NTXDatabaseTransportModel:
        # A recorded/prebuilt runtime owns a concrete interpolation database.
        # Reusing it is essential for black-box Radau: rebuilding here would
        # rerun the complete NTX scan every time a direct database model is
        # requested, including from the compact reverse hooks.
        database = self.database
        if database is None:
            database = self._build_runtime_database()
        return NTXDatabaseTransportModel(
            species=self.species,
            energy_grid=self.energy_grid,
            geometry=self.geometry,
            database=database,
            collisionality_model=self.collisionality_model,
            bc_density=self.bc_density,
            bc_temperature=self.bc_temperature,
            lagged_response_taylor_order=self.lagged_response_taylor_order,
        )

    def __call__(self, state) -> dict:
        return self._database_model()(state)

    def evaluate_momentum_corrected_fluxes(self, state, *, diagnostics: bool = False) -> dict:
        """Delegate corrected parallel-flow evaluation to the rebuilt scan database."""

        return self._database_model().evaluate_momentum_corrected_fluxes(
            state, diagnostics=diagnostics
        )

    def evaluate_momentum_corrected_upar_only(self, state):
        """Return corrected ``U_parallel`` from the rebuilt scan database."""

        return self._database_model().evaluate_momentum_corrected_upar_only(state)

    def pullback_momentum_corrected_upar_state_by_radius(self, state, upar_bar):
        """Delegate the compact database bootstrap state transpose."""

        return self._database_model().pullback_momentum_corrected_upar_state_by_radius(
            state, upar_bar
        )

    def pullback_momentum_corrected_upar_state_geometry_by_radius(
        self, state, upar_bar, geometry
    ):
        """Delegate the joint compact database state/geometry transpose."""

        return self._database_model().pullback_momentum_corrected_upar_state_geometry_by_radius(
            state, upar_bar, geometry
        )

    def pullback_momentum_corrected_upar_database_by_radius(self, state, upar_bar):
        """Delegate compact corrected-bootstrap database table bars."""

        return self._database_model().pullback_momentum_corrected_upar_database_by_radius(
            state, upar_bar
        )

    def pullback_momentum_corrected_upar_geometry_by_radius(
        self, state, upar_bar, geometry
    ):
        """Delegate the fixed-database compact bootstrap geometry transpose."""

        return self._database_model().pullback_momentum_corrected_upar_geometry_by_radius(
            state, upar_bar, geometry
        )

    def pullback_local_particle_flux_geometry_by_radius(
        self, state, er_profile, residual_bars, geometry
    ):
        """Delegate the compact selected-root geometry transpose."""

        return self._database_model().pullback_local_particle_flux_geometry_by_radius(
            state, er_profile, residual_bars, geometry
        )

    def build_local_particle_flux_evaluator(self, state):
        return self._database_model().build_local_particle_flux_evaluator(state)

    def evaluate_face_fluxes(self, state, face_state, **kwargs):
        return self._database_model().evaluate_face_fluxes(state, face_state, **kwargs)

    def build_lagged_response(self, state, **kwargs):
        return self._database_model().build_lagged_response(state, **kwargs)

    def evaluate_with_lagged_response(self, state, lagged_response, **kwargs):
        return self._database_model().evaluate_with_lagged_response(state, lagged_response, **kwargs)

    def pullback_build_lagged_response(self, state, lagged_response_bar, **kwargs):
        if int(self.lagged_response_taylor_order) > 1:
            raise NotImplementedError(
                "Higher-order ntx_scan_runtime lagged responses are forward-only. "
                "Their exact reverse rules require higher-response pullbacks."
            )
        return self._database_model().pullback_build_lagged_response(
            state,
            lagged_response_bar,
            **kwargs,
        )

    def pullback_direct_rhs_support_payload(self, state, flux_bar, support):
        """Transpose direct black-box database interpolation only.

        This is intentionally distinct from ``pullback_build_lagged_response``:
        the black-box Radau RHS called ``self(state)`` in its primal, so the
        local transpose is of that same direct database evaluation.  The
        retained scan primal is consumed later, once, after segment bars have
        been accumulated.
        """

        if not isinstance(support, dict) or "database" not in support:
            return None
        model = self.with_support_payload(support)
        database_model = model._database_model()
        return database_model.pullback_direct_rhs_support_payload(
            state, flux_bar, {"database": support["database"]}
        )

    def pullback_direct_rhs_state(self, state, flux_bar):
        """Delegate the direct database state transpose without rebuilding."""
        return self._database_model().pullback_direct_rhs_state(state, flux_bar)

    def pullback_direct_rhs_geometry_by_radius(self, state, flux_bar, geometry):
        """Delegate the compact fixed-database direct-flux geometry transpose."""

        return self._database_model().pullback_direct_rhs_geometry_by_radius(
            state, flux_bar, geometry
        )

    def pullback_local_particle_flux_support_payload(self, state, flux_bar, support):
        """Delegate the selected-root constrained local-flux transpose."""
        if not isinstance(support, dict) or "database" not in support:
            return None
        model = self.with_support_payload(support)
        return model._database_model().pullback_local_particle_flux_support_payload(
            state, flux_bar, {"database": support["database"]}
        )

    def with_runtime_database(self) -> "NTXRuntimeScanTransportModel":
        if self.database is not None:
            return self
        model = self.with_static_channels()
        database, scan_primal_record, scan_primal = model._build_runtime_database_and_record()
        return dataclasses.replace(
            model,
            database=database,
            scan_primal_record=scan_primal_record,
            scan_primal=scan_primal,
        )


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
    center_response_mode: str = "interpolate_from_faces"
    face_response_mode: str = "face_local_response"
    radial_batch_size: int | None = None
    radial_batch_mode: str = "simple"
    scan_batch_size: int | None = None
    response_anchor_count: int | None = None
    use_remat: bool = False
    derivative_mode: str = "direct"
    derivative_field_pullback_mode: str = "compact_vjp"
    derivative_pullback_boundary: str = "inline"
    derivative_pullback_algebra: str = "ntx_helper"
    er_v_floor: float | None = None
    collisionality_model: str = "default"
    bc_density: Any = None
    bc_temperature: Any = None
    density_floor: Any = DEFAULT_TRANSPORT_DENSITY_FLOOR
    temperature_floor: Any = DEFAULT_TRANSPORT_TEMPERATURE_FLOOR
    support: NTXExactLijRuntimeSupport | None = None
    # Shared with the database/scan response models.  Order two is wired as
    # an explicit forward-only feature gate while the dedicated factorized NTX
    # Hessian primitive is implemented below; order one remains the exact
    # established realtime response path.
    lagged_response_taylor_order: int = 1
    # Opt-in full transport-state composition of the existing coefficient
    # Hessian.  Kept separate from the established coefficient-only order-two
    # mode until its dedicated face evaluator is selected below.
    full_state_quadratic_response: bool = False
    # Explicit diagnostic only: compares the interpolated centre Lij against
    # a live direct-centre NTX evaluation at each response rebuild.
    debug_center_lij_comparison: bool = False

    def __post_init__(self):
        order = int(self.lagged_response_taylor_order)
        if order not in {1, 2}:
            raise ValueError(
                "ntx_exact_lij_runtime lagged_response_taylor_order must be 1 or 2."
            )
        if self.full_state_quadratic_response and order != 2:
            raise ValueError("full_state_quadratic_response requires lagged_response_taylor_order = 2.")
        if self.full_state_quadratic_response and str(self.collisionality_model).strip().lower() not in {"", "default"}:
            raise NotImplementedError("full_state_quadratic_response currently supports collisionality_model = 'default' only.")

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

    def pullback_direct_rhs_support_payload(self, state, flux_bar, support):
        """Black-box exact-Lij support transpose via the established lowdot rule."""
        response = self.with_support_payload(support).build_lagged_response(state)
        return self.with_support_payload(support).pullback_evaluate_with_lagged_response_support_payload(
            state, response, flux_bar, support
        )

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

    def with_center_response_mode(self, center_response_mode: str) -> "NTXExactLijRuntimeTransportModel":
        return dataclasses.replace(
            self,
            center_response_mode=self._normalize_center_response_mode(center_response_mode),
        )

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

    def with_lagged_response_taylor_order(self, lagged_response_taylor_order: int) -> "NTXExactLijRuntimeTransportModel":
        return dataclasses.replace(
            self,
            lagged_response_taylor_order=int(lagged_response_taylor_order),
        )

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
        aliases = {"plain": "direct", "jax": "direct", "default": "direct"}
        mode = aliases.get(mode, mode)
        if mode != "direct":
            raise ValueError("ntx_exact_derivative_mode must be 'direct'.")
        return mode

    @staticmethod
    def _normalize_center_response_mode(center_response_mode: str | None) -> str:
        mode = "interpolate_from_faces" if center_response_mode in (None, "") else str(center_response_mode).strip().lower()
        aliases = {
            "default": "interpolate_from_faces",
            "interpolate": "interpolate_from_faces",
            "interpolate_face_response": "interpolate_from_faces",
            "interpolate_faces": "interpolate_from_faces",
            "interpolate_center_response": "center_local_response",
            "interpolate_center_fluxes": "center_local_response",
            "center_interpolation": "center_local_response",
            "face_local_response": "interpolate_from_faces",
            "local_center_response": "center_local_response",
            "center_response": "center_local_response",
            "center_local": "center_local_response",
            "interpolate_coefficients": "interpolate_face_coefficients",
            "interpolate_face_coefficients": "interpolate_face_coefficients",
            "face_coefficient_interpolation": "interpolate_face_coefficients",
            "interpolate_face_coefficients_cubic": "interpolate_face_coefficients_cubic",
            "interpolate_face_coefficients_four_point": "interpolate_face_coefficients_cubic",
            "face_coefficient_cubic": "interpolate_face_coefficients_cubic",
            "interpolate_face_coefficients_physical_coordinates": "interpolate_face_coefficients_physical_coordinates",
            "interpolate_face_coefficients_er_over_v": "interpolate_face_coefficients_physical_coordinates",
            "face_coefficient_physical_coordinates": "interpolate_face_coefficients_physical_coordinates",
            "interpolate_face_coefficients_native_distance": "interpolate_face_coefficients_native_distance",
            "face_coefficient_native_distance": "interpolate_face_coefficients_native_distance",
            "interpolate_face_coefficients_taylor_reliability": "interpolate_face_coefficients_taylor_reliability",
            "face_coefficient_taylor_reliability": "interpolate_face_coefficients_taylor_reliability",
        }
        mode = aliases.get(mode, mode)
        if mode not in {
            "interpolate_from_faces",
            "center_local_response",
            "interpolate_face_coefficients",
            "interpolate_face_coefficients_cubic",
            "interpolate_face_coefficients_physical_coordinates",
            "interpolate_face_coefficients_native_distance",
            "interpolate_face_coefficients_taylor_reliability",
        }:
            raise ValueError(
                "ntx_exact_center_response_mode must be one of: "
                "interpolate_from_faces, center_local_response, "
                "interpolate_face_coefficients, "
                "interpolate_face_coefficients_cubic, "
                "interpolate_face_coefficients_physical_coordinates, "
                "interpolate_face_coefficients_native_distance, "
                "interpolate_face_coefficients_taylor_reliability"
            )
        return mode

    def _resolved_center_response_mode(self) -> str:
        return self._normalize_center_response_mode(self.center_response_mode)

    def _center_reference_scan_inputs(
        self,
        *,
        channels,
        Er,
        temperature,
        density,
        v_thermal,
        radius_coordinates,
    ):
        """Obtain centre reference NTX coordinates without an NTX solve."""

        collisionality_kind = _collisionality_kind(self.collisionality_model)
        species_indices = jnp.arange(int(self.species.number_species), dtype=jnp.int32)
        radius_indices = jnp.arange(Er.shape[0], dtype=jnp.int32)

        def _per_radius(radius_index):
            drds_value = jax.lax.dynamic_index_in_dim(channels.drds, radius_index, axis=0, keepdims=False)
            er_value = jax.lax.dynamic_index_in_dim(Er, radius_index, axis=0, keepdims=False)
            temperature_local = jax.lax.dynamic_index_in_dim(temperature, radius_index, axis=1, keepdims=False)
            density_local = jax.lax.dynamic_index_in_dim(density, radius_index, axis=1, keepdims=False)
            vthermal_local = jax.lax.dynamic_index_in_dim(v_thermal, radius_index, axis=1, keepdims=False)
            return jax.vmap(
                lambda species_index: self._local_scan_inputs(
                    drds_value=drds_value,
                    species_index=species_index,
                    er_value=er_value,
                    temperature_local=temperature_local,
                    density_local=density_local,
                    vthermal_local=vthermal_local,
                    collisionality_kind=collisionality_kind,
                )[:2]
            )(species_indices)

        nu_hat, epsi_hat = self._map_radius_axis_regularized_at_axis0(
            _per_radius, radius_indices, radius_coordinates
        )
        return nu_hat, epsi_hat

    def _interpolate_face_quadratic_coefficients_to_centres(
        self,
        face_response: NTXQuadraticPreparedCoefficientResponse,
        *,
        center_reference_nu_hat,
        center_reference_epsi_hat,
        weight_mode: str = "radial",
        coordinate_mode: str = "native",
        center_drds=None,
        face_drds=None,
        weight_hi_override=None,
    ) -> NTXQuadraticPreparedCoefficientResponse:
        """Interpolate face NTX Taylor polynomials to cell centres.

        Each face response is anchored at different ``(nu_hat, epsi_hat)``.
        For each centre, first translate face polynomials to that centre's
        reference coordinates. ``weight_mode='radial'`` uses its adjacent
        two-face linear stencil. ``weight_mode='radial_cubic'`` uses a fixed
        four-face Lagrange stencil; this is an opt-in forward experiment,
        intended to test whether a wider common-centre-state coefficient
        reconstruction improves a root layer without additional NTX solves.
        ``weight_mode='native_distance'`` instead weights the two adjacent
        responses by their native ``(nu_hat, epsi_hat)`` distance.
        ``weight_mode='taylor_reliability'`` retains the geometric radial
        weights as a prior, but downweights a side whose quadratic Taylor
        correction is large relative to its local constant-plus-linear
        coefficient scale.  This is a fixed-shape, rebuild-time reliability
        estimate; it does not perform an extra NTX solve.

        ``coordinate_mode='physical_er_over_v'`` keeps the radial
        interpolation query fixed in physical ``(nu/v, Er/v)`` coordinates.
        NTX stores the electric coordinate as ``epsi_hat=drds*Er/v``; hence a
        centre ``epsi_hat`` must be rescaled separately for each face before
        evaluating that face polynomial.  The returned Taylor derivatives are
        transformed back to the centre ``epsi_hat`` by the matching chain
        rule.  This is the realtime analogue of the database's common-query
        coordinate interpolation.
        """

        face_rho = jnp.asarray(self.geometry.r_grid_half, dtype=jnp.float64) / jnp.asarray(
            self.geometry.a_b, dtype=jnp.float64
        )
        center_rho = jnp.asarray(self.geometry.r_grid, dtype=jnp.float64) / jnp.asarray(
            self.geometry.a_b, dtype=jnp.float64
        )
        hi = jnp.searchsorted(face_rho, center_rho, side="right")
        hi = jnp.clip(hi, 1, face_rho.shape[0] - 1)
        lo = hi - 1
        if weight_mode not in {
            "radial",
            "radial_cubic",
            "native_distance",
            "taylor_reliability",
        }:
            raise ValueError(
                "weight_mode must be 'radial', 'radial_cubic', 'native_distance', "
                "or 'taylor_reliability'."
            )
        if coordinate_mode not in {"native", "physical_er_over_v"}:
            raise ValueError(
                "coordinate_mode must be 'native' or 'physical_er_over_v'."
            )
        u_center_native = jnp.asarray(center_reference_nu_hat)
        e_center_native = jnp.asarray(center_reference_epsi_hat)
        if weight_mode == "radial_cubic":
            if weight_hi_override is not None:
                raise ValueError("weight_hi_override is only valid for a two-face stencil.")
            n_faces = int(face_rho.shape[0])
            if n_faces < 4:
                raise ValueError("radial_cubic requires at least four face radii.")
            start = jnp.clip(lo - 1, 0, n_faces - 4)
            face_indices = start[:, None] + jnp.arange(4, dtype=lo.dtype)[None, :]
            nodes = face_rho[face_indices]
            radial_weights = []
            for stencil_index in range(4):
                weight = jnp.ones_like(center_rho)
                for other_index in range(4):
                    if other_index == stencil_index:
                        continue
                    denominator = nodes[:, stencil_index] - nodes[:, other_index]
                    weight = weight * (center_rho - nodes[:, other_index]) / jnp.where(
                        jnp.abs(denominator) > 0.0, denominator, 1.0
                    )
                radial_weights.append(weight)
            weights = jnp.stack(radial_weights, axis=1)
        else:
            face_indices = jnp.stack((lo, hi), axis=1)
            denominator = face_rho[hi] - face_rho[lo]
            u_lo_native = face_response.reference_nu_hat[lo]
            u_hi_native = face_response.reference_nu_hat[hi]
            e_lo_native = face_response.reference_epsi_hat[lo]
            e_hi_native = face_response.reference_epsi_hat[hi]
            radial_weight_hi = (center_rho - face_rho[lo]) / jnp.where(
                jnp.abs(denominator) > 0.0, denominator, 1.0
            )
            radial_weight_hi = jnp.clip(radial_weight_hi, 0.0, 1.0)
            if weight_hi_override is None and weight_mode in {
                "radial",
                "taylor_reliability",
            }:
                weight_hi = radial_weight_hi
            elif weight_hi_override is None:
                # The local left/right span non-dimensionalizes the two
                # native coordinates without assuming that nu_hat and
                # epsi_hat have comparable numerical magnitudes.
                nu_scale = jnp.maximum(jnp.abs(u_hi_native - u_lo_native), 1.0e-30)
                epsi_scale = jnp.maximum(jnp.abs(e_hi_native - e_lo_native), 1.0e-30)
                distance_lo2 = (
                    ((u_center_native - u_lo_native) / nu_scale) ** 2
                    + ((e_center_native - e_lo_native) / epsi_scale) ** 2
                )
                distance_hi2 = (
                    ((u_center_native - u_hi_native) / nu_scale) ** 2
                    + ((e_center_native - e_hi_native) / epsi_scale) ** 2
                )
                inverse_lo = 1.0 / jnp.maximum(distance_lo2, 1.0e-24)
                inverse_hi = 1.0 / jnp.maximum(distance_hi2, 1.0e-24)
                weight_hi = inverse_hi / (inverse_lo + inverse_hi)
            else:
                weight_hi = jnp.asarray(weight_hi_override, dtype=center_rho.dtype)
                if weight_hi.shape != center_rho.shape:
                    raise ValueError("weight_hi_override must have one value per cell centre.")
            weights = jnp.stack((1.0 - weight_hi, weight_hi), axis=1)

        u_center = u_center_native[:, None, ..., None]
        e_center = e_center_native[:, None, ..., None]
        u_face = face_response.reference_nu_hat[face_indices][..., None]
        e_face = face_response.reference_epsi_hat[face_indices][..., None]
        if coordinate_mode == "physical_er_over_v":
            if center_drds is None or face_drds is None:
                raise ValueError(
                    "physical_er_over_v interpolation requires centre and face drds."
                )
            center_drds = jnp.asarray(center_drds, dtype=center_rho.dtype)
            face_drds = jnp.asarray(face_drds, dtype=face_rho.dtype)
            centre_scale = center_drds[:, None]
            face_scale = face_drds[face_indices]
            finite_scale = (
                jnp.isfinite(centre_scale)
                & jnp.isfinite(face_scale)
                & (jnp.abs(centre_scale) > 1.0e-30)
            )
            epsi_scale = jnp.where(finite_scale, face_scale / centre_scale, 1.0)
            epsi_scale = epsi_scale[..., None, None, None]
            e_center = e_center * epsi_scale
        else:
            epsi_scale = jnp.ones(
                face_indices.shape + (1, 1, 1), dtype=center_rho.dtype
            )

        def _translate(c0, cu, ce, cuu, cue, cee):
            du = u_center - u_face
            de = e_center - e_face
            translated_c0 = (
                c0 + cu * du + ce * de
                + 0.5 * cuu * du * du + cue * du * de + 0.5 * cee * de * de
            )
            translated_cu = cu + cuu * du + cue * de
            translated_ce_face = ce + cue * du + cee * de
            # The face expansion is in epsi_face = scale * epsi_center.
            # Convert all electric-coordinate derivatives back to the centre
            # coordinate before the radial blend.
            return (
                translated_c0,
                translated_cu,
                translated_ce_face * epsi_scale,
                cuu,
                cue * epsi_scale,
                cee * epsi_scale * epsi_scale,
            )

        face_fields = (
            face_response.reference_coefficients,
            face_response.dcoefficients_d_nu_hat,
            face_response.dcoefficients_d_epsi_hat,
            face_response.d2coefficients_d_nu_hat2,
            face_response.d2coefficients_d_nu_hat_d_epsi_hat,
            face_response.d2coefficients_d_epsi_hat2,
        )
        translated = _translate(*(field[face_indices] for field in face_fields))
        if weight_mode == "taylor_reliability" and weight_hi_override is None:
            # ``u_center`` / ``e_center`` already carry the face-dependent
            # physical Er/v rescaling when requested above.  Estimate each
            # side's trustworthiness from the retained quadratic correction
            # at this common centre state.  The radial weights remain the
            # tie-breaking prior when both Taylor expansions have equal
            # curvature.
            c0, cu, ce, cuu, cue, cee = (
                field[face_indices] for field in face_fields
            )
            du = u_center - u_face
            de = e_center - e_face
            linear = cu * du + ce * de
            quadratic = (
                0.5 * cuu * du * du
                + cue * du * de
                + 0.5 * cee * de * de
            )
            reduce_axes = tuple(range(3, quadratic.ndim))
            quadratic_scale = jnp.sqrt(jnp.mean(quadratic * quadratic, axis=reduce_axes))
            local_scale = jnp.sqrt(
                jnp.mean(c0 * c0 + linear * linear, axis=reduce_axes)
            )
            curvature = quadratic_scale / jnp.maximum(local_scale, 1.0e-30)
            radial_prior = jnp.stack((1.0 - radial_weight_hi, radial_weight_hi), axis=1)
            score = radial_prior[:, :, None] / jnp.maximum(curvature, 1.0e-12)
            weights = score / jnp.maximum(jnp.sum(score, axis=1, keepdims=True), 1.0e-30)
            weights = weights.reshape(
                weights.shape + (1,) * (translated[0].ndim - weights.ndim)
            )
        else:
            weights = weights.reshape(
                weights.shape + (1,) * (translated[0].ndim - weights.ndim)
            )
        center_c0, center_cu, center_ce, center_cuu, center_cue, center_cee = tuple(
            jnp.sum(weights * value, axis=1)
            for value in translated
        )
        return NTXQuadraticPreparedCoefficientResponse(
            reference_nu_hat=center_reference_nu_hat,
            reference_epsi_hat=center_reference_epsi_hat,
            reference_coefficients=center_c0,
            dcoefficients_d_nu_hat=center_cu,
            dcoefficients_d_epsi_hat=center_ce,
            d2coefficients_d_nu_hat2=center_cuu,
            d2coefficients_d_nu_hat_d_epsi_hat=center_cue,
            d2coefficients_d_epsi_hat2=center_cee,
        )

    @staticmethod
    def _normalize_derivative_field_pullback_mode(mode: str | None) -> str:
        normalized = "compact_vjp" if mode in (None, "") else str(mode).strip().lower()
        aliases = {
            "default": "compact_vjp",
            "compact": "compact_vjp",
            "compact-vjp": "compact_vjp",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized != "compact_vjp":
            raise ValueError("ntx_exact_derivative_field_pullback_mode must be 'compact_vjp'.")
        return normalized

    @staticmethod
    def _normalize_derivative_pullback_boundary(mode: str | None) -> str:
        normalized = "inline" if mode in (None, "") else str(mode).strip().lower()
        aliases = {"default": "inline", "none": "inline", "off": "inline"}
        normalized = aliases.get(normalized, normalized)
        if normalized != "inline":
            raise ValueError("ntx_exact_derivative_pullback_boundary must be 'inline'.")
        return normalized

    @staticmethod
    def _normalize_derivative_pullback_algebra(mode: str | None) -> str:
        normalized = "ntx_helper" if mode in (None, "") else str(mode).strip().lower()
        aliases = {
            "default": "ntx_helper",
            "ntx": "ntx_helper",
            "compact": "ntx_helper",
            "compact_vjp": "ntx_helper",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"ntx_helper", "ntx_helper_lowdot_fused"}:
            raise ValueError(
                "ntx_exact_derivative_pullback_algebra must be 'ntx_helper' or "
                "'ntx_helper_lowdot_fused'."
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

    def _pullback_interpolate_anchor_target_rho(
        self,
        anchor_indices,
        anchor_values,
        target_rho,
        interpolated_values_bar,
    ):
        """Transpose the coordinate part of `_interpolate_anchor_values`.

        JAX does not currently transpose `jnp.interp` with respect to the
        interpolation coordinates in this nested use, so keep this tiny
        piecewise-linear transpose explicit and leave value cotangents to the
        existing anchor-value transpose.
        """
        anchor_rho = target_rho[anchor_indices]
        n_anchor = int(anchor_indices.shape[0])
        flat_anchor = jnp.asarray(anchor_values).reshape((n_anchor, -1))
        flat_bar = jnp.asarray(interpolated_values_bar).reshape((target_rho.shape[0], -1))

        hi = jnp.searchsorted(anchor_rho, target_rho, side="right")
        hi = jnp.clip(hi, 1, n_anchor - 1)
        lo = hi - 1
        rho_lo = anchor_rho[lo]
        rho_hi = anchor_rho[hi]
        den = rho_hi - rho_lo
        den_safe = jnp.where(jnp.abs(den) > 0.0, den, 1.0)
        values_lo = flat_anchor[lo]
        values_hi = flat_anchor[hi]
        values_delta = values_hi - values_lo
        active = jnp.logical_and(target_rho >= anchor_rho[0], target_rho <= anchor_rho[-1])
        active_f = jnp.asarray(active, dtype=flat_bar.dtype)
        slope = values_delta / den_safe[:, None]

        target_bar = active_f * jnp.sum(flat_bar * slope, axis=1)
        lo_anchor_bar = active_f * jnp.sum(
            flat_bar * values_delta * ((target_rho - rho_hi) / (den_safe * den_safe))[:, None],
            axis=1,
        )
        hi_anchor_bar = active_f * jnp.sum(
            flat_bar * values_delta * (-(target_rho - rho_lo) / (den_safe * den_safe))[:, None],
            axis=1,
        )
        return (
            jnp.zeros_like(target_rho)
            .at[jnp.arange(target_rho.shape[0], dtype=jnp.int32)]
            .add(target_bar)
            .at[anchor_indices[lo]]
            .add(lo_anchor_bar)
            .at[anchor_indices[hi]]
            .add(hi_anchor_bar)
        )

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

    def _pullback_transport_moments_from_single_coefficient_vector_and_drds(
        self,
        coefficient_vector,
        *,
        drds_value,
        energy_index,
        transport_moments_bar,
    ):
        """Exact coefficient and direct ``drds`` bars for one energy node.

        This is only the explicit moment prefactor contribution. The implicit
        ``drds`` dependence of the monoenergetic case is returned separately
        by :meth:`_pullback_local_scan_inputs_and_drds_from_primitives`.
        """

        _, pullback = jax.vjp(
            lambda coefficient_value, drds_local: self._transport_moments_from_single_coefficient_vector(
                coefficient_value,
                drds_value=drds_local,
                energy_index=energy_index,
            ),
            coefficient_vector,
            drds_value,
        )
        return pullback(transport_moments_bar)

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

    def _momentum_matrices_from_coefficient_scan(
        self,
        coeff_scan,
        *,
        drds_value,
        species_index: int,
        vth_a,
        nu_hat_a,
    ):
        """Build realtime NTX 5x5 moment matrices for momentum correction."""

        d11_physical = jnp.asarray(coeff_scan[:, 0], dtype=jnp.float64) * drds_value**2
        d11_physical = jnp.maximum(d11_physical, jnp.asarray(D11_POSITIVE_FLOOR, dtype=jnp.float64))
        d11_a = -d11_physical
        d13_a = -(jnp.asarray(coeff_scan[:, 2], dtype=jnp.float64) * drds_value)
        d33_a = -jnp.asarray(coeff_scan[:, 3], dtype=jnp.float64)
        v_new_a = jnp.asarray(self.energy_grid.v_norm, dtype=jnp.float64) * jnp.asarray(vth_a, dtype=jnp.float64)
        nu_a = jnp.asarray(nu_hat_a, dtype=jnp.float64) * v_new_a
        weights = jnp.asarray(self.energy_grid.xWeights, dtype=jnp.float64)

        charge = self.species.charge[species_index]
        mass = self.species.mass[species_index]
        inv_sqrt_pi = 1.0 / jnp.sqrt(jnp.pi)
        l11_fac = -inv_sqrt_pi * (mass / charge) ** 2 * vth_a**3
        l13_fac = -inv_sqrt_pi * (mass / charge) * vth_a**2
        l33_fac = -inv_sqrt_pi * vth_a

        def _weighted(name, values, *, nu_weighted=False):
            weight = jnp.asarray(getattr(self.energy_grid, name), dtype=jnp.float64) * weights
            if nu_weighted:
                weight = weight * nu_a
            return jnp.sum(weight * values)

        lij = jnp.zeros((5, 5), dtype=jnp.float64)
        eij = jnp.zeros((5, 5), dtype=jnp.float64)
        lij = lij.at[0, 0].set(l11_fac * _weighted("L11_weight", d11_a))
        lij = lij.at[0, 1].set(l11_fac * _weighted("L12_weight", d11_a))
        lij = lij.at[1, 0].set(lij[0, 1])
        lij = lij.at[1, 1].set(l11_fac * _weighted("L22_weight", d11_a))
        lij = lij.at[0, 2].set(l13_fac * _weighted("L13_weight", d13_a))
        lij = lij.at[1, 2].set(l13_fac * _weighted("L23_weight", d13_a))
        lij = lij.at[2, 0].set(-lij[0, 2])
        lij = lij.at[2, 1].set(-lij[1, 2])
        lij = lij.at[2, 2].set(l33_fac * _weighted("L33_weight", d33_a))
        lij = lij.at[0, 3].set(lij[1, 2])
        lij = lij.at[1, 3].set(l13_fac * _weighted("L24_weight", d13_a))
        lij = lij.at[0, 4].set(lij[1, 3])
        lij = lij.at[1, 4].set(l13_fac * _weighted("L25_weight", d13_a))
        lij = lij.at[3, 0].set(-lij[0, 3])
        lij = lij.at[4, 0].set(-lij[0, 4])
        lij = lij.at[3, 1].set(-lij[1, 3])
        lij = lij.at[4, 1].set(-lij[1, 4])
        lij = lij.at[3, 2].set(l33_fac * _weighted("L43_weight", d33_a))
        lij = lij.at[2, 3].set(lij[3, 2])
        lij = lij.at[3, 3].set(l33_fac * _weighted("L44_weight", d33_a))
        lij = lij.at[2, 4].set(lij[3, 3])
        lij = lij.at[4, 2].set(lij[3, 3])
        lij = lij.at[3, 4].set(l33_fac * _weighted("L45_weight", d33_a))
        lij = lij.at[4, 3].set(lij[3, 4])
        lij = lij.at[4, 4].set(l33_fac * _weighted("L55_weight", d33_a))

        eij = eij.at[0, 2].set(l13_fac * _weighted("L13_weight", d13_a, nu_weighted=True))
        eij = eij.at[1, 2].set(l13_fac * _weighted("L23_weight", d13_a, nu_weighted=True))
        eij = eij.at[2, 0].set(-eij[0, 2])
        eij = eij.at[2, 1].set(-eij[1, 2])
        eij = eij.at[2, 2].set(l33_fac * _weighted("L33_weight", d33_a, nu_weighted=True))
        eij = eij.at[0, 3].set(eij[1, 2])
        eij = eij.at[1, 3].set(l13_fac * _weighted("L24_weight", d13_a, nu_weighted=True))
        eij = eij.at[0, 4].set(eij[1, 3])
        eij = eij.at[1, 4].set(l13_fac * _weighted("L25_weight", d13_a, nu_weighted=True))
        eij = eij.at[3, 0].set(-eij[0, 3])
        eij = eij.at[4, 0].set(-eij[0, 4])
        eij = eij.at[3, 1].set(-eij[1, 3])
        eij = eij.at[4, 1].set(-eij[1, 4])
        eij = eij.at[3, 2].set(l33_fac * _weighted("L43_weight", d33_a, nu_weighted=True))
        eij = eij.at[2, 3].set(eij[3, 2])
        eij = eij.at[3, 3].set(l33_fac * _weighted("L44_weight", d33_a, nu_weighted=True))
        eij = eij.at[2, 4].set(eij[3, 3])
        eij = eij.at[4, 2].set(eij[3, 3])
        eij = eij.at[3, 4].set(l33_fac * _weighted("L45_weight", d33_a, nu_weighted=True))
        eij = eij.at[4, 3].set(eij[3, 4])
        eij = eij.at[4, 4].set(l33_fac * _weighted("L55_weight", d33_a, nu_weighted=True))

        nu_av = jnp.stack(
            [
                jnp.sum(nu_a * jnp.asarray(self.energy_grid.L13_weight, dtype=jnp.float64) * weights),
                jnp.sum(nu_a * jnp.asarray(self.energy_grid.L23_weight, dtype=jnp.float64) * weights),
                jnp.sum(nu_a * jnp.asarray(self.energy_grid.L24_weight, dtype=jnp.float64) * weights),
            ]
        )
        return lij, eij, nu_av

    def _solve_momentum_matrices_prepared_local(
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
        return self._momentum_matrices_from_coefficient_scan(
            coeff_scan,
            drds_value=drds_value,
            species_index=species_index,
            vth_a=vth_a,
            nu_hat_a=nu_hat_a,
        )

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

    def _build_quadratic_coefficient_response_local(
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
        """Build a full-radius local quadratic moment response.

        The NTX coefficient value/gradient/Hessian is supplied by the
        dedicated factorized primitive.  The small JVPs below differentiate
        only the inexpensive coefficient-to-moment reduction (including its
        D11 floor), never an NTX solve or a custom-VJP rule.
        """
        ref_nu_hat, ref_epsi_hat, _ = self._local_scan_inputs(
            drds_value=drds_value,
            species_index=species_index,
            er_value=er_value,
            temperature_local=temperature_local,
            density_local=density_local,
            vthermal_local=vthermal_local,
            collisionality_kind=collisionality_kind,
        )
        ntx = _import_ntx()

        def _one_energy(nu_hat_value, epsi_hat_value):
            return ntx.solve_prepared_coefficient_vector_hessian_factorized(
                prepared,
                ntx.MonoenergeticCase(
                    nu_hat=nu_hat_value,
                    epsi_hat=epsi_hat_value,
                ),
            )

        (
            coefficient_base,
            coefficient_nu,
            coefficient_epsi,
            coefficient_nunu,
            coefficient_nuepsi,
            coefficient_epsiepsi,
        ) = jax.vmap(_one_energy)(ref_nu_hat, ref_epsi_hat)

        return NTXQuadraticPreparedCoefficientResponse(
            reference_nu_hat=ref_nu_hat,
            reference_epsi_hat=ref_epsi_hat,
            reference_coefficients=coefficient_base,
            dcoefficients_d_nu_hat=coefficient_nu,
            dcoefficients_d_epsi_hat=coefficient_epsi,
            d2coefficients_d_nu_hat2=coefficient_nunu,
            d2coefficients_d_nu_hat_d_epsi_hat=coefficient_nuepsi,
            d2coefficients_d_epsi_hat2=coefficient_epsiepsi,
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
        use_factorized_ntx_two_directional_prepared_vjp: bool = False,
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
            use_factorized_ntx_two_directional_prepared_vjp=(
                use_factorized_ntx_two_directional_prepared_vjp
            ),
        )

    def _interpolated_moment_reduced_local_outputs_from_primitives(
        self,
        prepared,
        *,
        drds_value,
        nu_hat_a,
        epsi_hat_a,
        vth_a,
        use_factorized_ntx_two_directional_prepared_vjp: bool = False,
    ):
        if use_factorized_ntx_two_directional_prepared_vjp:
            return self._interpolated_moment_reduced_local_outputs_from_factorized_ntx_two_directional(
                prepared,
                drds_value=drds_value,
                nu_hat_a=nu_hat_a,
                epsi_hat_a=epsi_hat_a,
                vth_a=vth_a,
            )
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

    def _interpolated_moment_reduced_local_outputs_with_coefficient_record_from_primitives(
        self,
        prepared,
        *,
        drds_value,
        nu_hat_a,
        epsi_hat_a,
        vth_a,
    ):
        """Return the ordinary local response plus compact coefficient primitives.

        This is deliberately separate from the ordinary response routine.  It
        follows its base solve and its two case-direction JVPs, but retains the
        coefficient-vector outputs of those computations rather than only the
        resulting transport moments.  It is intended solely for a future
        segment-bounded replay record; no NTX factorisation is retained.
        """

        coefficient_scan = self._coefficient_scan_from_inputs(
            prepared,
            nu_hat_a,
            epsi_hat_a,
        )
        # The active NTX-exact lane accepts only ``direct`` (including its
        # aliases), so retain the ordinary solver selection exactly.  Keeping
        # this explicit prevents a future unsupported derivative-mode branch
        # from silently giving the record path different primal coefficients.
        self._normalize_derivative_mode(self.derivative_mode)
        derivative_mode_override = None
        energy_indices = jnp.arange(nu_hat_a.shape[0], dtype=jnp.int32)
        epsi_hat_er_tangent = jnp.asarray(1.0e3, dtype=epsi_hat_a.dtype) / (
            self.energy_grid.v_norm * vth_a
        )

        def _directional_coefficient_scans_one_energy(args):
            _energy_index, nu_hat_value, epsi_hat_value, epsi_er_tangent = args

            def _coefficient_value(nu_value, epsi_value):
                return self._single_coefficient_vector_from_inputs(
                    prepared,
                    nu_value,
                    epsi_value,
                    derivative_mode_override=derivative_mode_override,
                )

            _, er_coefficient_dot = jax.jvp(
                _coefficient_value,
                (nu_hat_value, epsi_hat_value),
                (
                    jnp.asarray(0.0, dtype=nu_hat_value.dtype),
                    epsi_er_tangent,
                ),
            )
            _, log_nu_coefficient_dot = jax.jvp(
                _coefficient_value,
                (nu_hat_value, epsi_hat_value),
                (
                    nu_hat_value,
                    jnp.asarray(0.0, dtype=epsi_hat_value.dtype),
                ),
            )
            return er_coefficient_dot, log_nu_coefficient_dot

        er_coefficient_dot_scan, log_nu_coefficient_dot_scan = jax.lax.map(
            _directional_coefficient_scans_one_energy,
            (energy_indices, nu_hat_a, epsi_hat_a, epsi_hat_er_tangent),
        )

        def _directional_moments_one_energy(args):
            coefficient_value, coefficient_dot, energy_index = args
            return jax.jvp(
                lambda coefficients: self._transport_moments_from_single_coefficient_vector(
                    coefficients,
                    drds_value=drds_value,
                    energy_index=energy_index,
                ),
                (coefficient_value,),
                (coefficient_dot,),
            )[1]

        response = (
            self._log_nu_star_from_nu_hat(nu_hat_a),
            self._transport_moments_from_coefficient_scan(
                coefficient_scan,
                drds_value=drds_value,
            ),
            jnp.sum(
                jax.lax.map(
                    _directional_moments_one_energy,
                    (coefficient_scan, er_coefficient_dot_scan, energy_indices),
                ),
                axis=0,
            ),
            jnp.sum(
                jax.lax.map(
                    _directional_moments_one_energy,
                    (coefficient_scan, log_nu_coefficient_dot_scan, energy_indices),
                ),
                axis=0,
            ),
        )
        return response, _NTXInterpolatedMomentCoefficientRecord(
            coefficient_scan=coefficient_scan,
            dcoefficient_scan_d_er=er_coefficient_dot_scan,
            dcoefficient_scan_d_log_nu_star=log_nu_coefficient_dot_scan,
        )

    def _build_interpolated_moment_response_local_with_coefficient_record(
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
        """Opt-in companion of the ordinary local interpolated response."""

        reference_nu_hat, reference_epsi_hat, vth_a = self._interpolated_moment_local_scan_primitives(
            drds_value=drds_value,
            species_index=species_index,
            er_value=er_value,
            temperature_local=temperature_local,
            density_local=density_local,
            vthermal_local=vthermal_local,
            collisionality_kind=collisionality_kind,
        )
        return self._interpolated_moment_reduced_local_outputs_with_coefficient_record_from_primitives(
            prepared,
            drds_value=drds_value,
            nu_hat_a=reference_nu_hat,
            epsi_hat_a=reference_epsi_hat,
            vth_a=vth_a,
        )

    def _interpolated_moment_reduced_local_outputs_from_factorized_ntx_two_directional(
        self,
        prepared,
        *,
        drds_value,
        nu_hat_a,
        epsi_hat_a,
        vth_a,
    ):
        """Return the local interpolated response from one factorization per energy.

        This isolated reverse-rebuild primitive replaces the base NTX solve plus
        two generic JVPs with NTX's factorized base/two-directional helper.  Its
        custom VJP retains those per-energy factors only for this local VJP;
        it neither changes the normal forward path nor adds a timestep tape.
        """
        ntx = _import_ntx()
        solve_three = getattr(
            ntx,
            "solve_prepared_coefficient_vector_two_directional_prepared_vjp",
            None,
        )
        if not callable(solve_three):
            raise RuntimeError(
                "The factorized NTX rebuild-support mode requires "
                "solve_prepared_coefficient_vector_two_directional_prepared_vjp."
            )
        energy_indices = jnp.arange(nu_hat_a.shape[0], dtype=jnp.int32)
        epsi_hat_er_tangent = jnp.asarray(1.0e3, dtype=epsi_hat_a.dtype) / (
            self.energy_grid.v_norm * vth_a
        )

        def _one_energy(args):
            _energy_index, nu_hat_value, epsi_hat_value, epsi_er_tangent = args
            return solve_three(
                prepared,
                ntx.MonoenergeticCase(nu_hat=nu_hat_value, epsi_hat=epsi_hat_value),
                ntx.MonoenergeticCase(
                    nu_hat=jnp.asarray(0.0, dtype=nu_hat_value.dtype),
                    epsi_hat=epsi_er_tangent,
                ),
                ntx.MonoenergeticCase(
                    nu_hat=nu_hat_value,
                    epsi_hat=jnp.asarray(0.0, dtype=epsi_hat_value.dtype),
                ),
            )

        coefficient_scan, er_coefficient_dot_scan, log_nu_coefficient_dot_scan = jax.lax.map(
            _one_energy,
            (energy_indices, nu_hat_a, epsi_hat_a, epsi_hat_er_tangent),
        )

        def _directional_moments_from_one(args):
            coefficient_value, coefficient_dot, energy_index = args
            return jax.jvp(
                lambda coefficients: self._transport_moments_from_single_coefficient_vector(
                    coefficients,
                    drds_value=drds_value,
                    energy_index=energy_index,
                ),
                (coefficient_value,),
                (coefficient_dot,),
            )[1]

        return (
            self._log_nu_star_from_nu_hat(nu_hat_a),
            self._transport_moments_from_coefficient_scan(
                coefficient_scan,
                drds_value=drds_value,
            ),
            jnp.sum(
                jax.lax.map(
                    _directional_moments_from_one,
                    (coefficient_scan, er_coefficient_dot_scan, energy_indices),
                ),
                axis=0,
            ),
            jnp.sum(
                jax.lax.map(
                    _directional_moments_from_one,
                    (coefficient_scan, log_nu_coefficient_dot_scan, energy_indices),
                ),
                axis=0,
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
        use_ntx_lowdot = normalized_pullback_algebra == "ntx_helper_lowdot_fused"
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
                "ntx_helper_lowdot_fused",
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

    def _pullback_interpolated_moment_reduced_local_outputs_with_prepared_support_and_drds(
        self,
        prepared,
        *,
        drds_value,
        reference_nu_hat,
        reference_epsi_hat,
        vth_a,
        field_bars,
        packed_support_directional_adjoint: bool = False,
    ):
        """Joint exact local pullback for state primitives and prepared support.

        This is deliberately an opt-in building block for the joint rebuild
        path. It shares NTX's base, ``d/dEr``, and ``d/dlog(nu)`` factorization
        and returns all support dependence explicitly: the complete prepared
        NTX system and
        both direct and primitive-mediated ``drds`` terms.
        """

        ntx = _import_ntx()
        (
            reference_log_nu_star_bar,
            base_transport_moments_bar,
            dtransport_moments_d_er_bar,
            dtransport_moments_d_log_nu_star_bar,
        ) = _interpolated_response_field_bar_tuple(field_bars)
        zero_nu_hat = jnp.zeros_like(reference_nu_hat)
        zero_epsi_hat = jnp.zeros_like(reference_epsi_hat)
        epsi_hat_tangent = jnp.asarray(1.0e3, dtype=reference_epsi_hat.dtype) / (
            self.energy_grid.v_norm * vth_a
        )
        energy_indices = jnp.arange(reference_nu_hat.shape[0], dtype=jnp.int32)

        def _one_energy(args):
            energy_index, nu_hat_value, epsi_hat_value, epsi_dot = args

            def _bars_and_direct_drds(coefficients, first_coeff_dot, second_coeff_dot):
                def _moment_pullback(coefficients_value, moment_bar):
                    _, pullback = jax.vjp(
                        lambda coefficient_value, drds_local: self._transport_moments_from_single_coefficient_vector(
                            coefficient_value,
                            drds_value=drds_local,
                            energy_index=energy_index,
                        ),
                        coefficients_value,
                        drds_value,
                    )
                    return pullback(moment_bar)

                base_coefficient_bar, base_drds_bar = _moment_pullback(
                    coefficients,
                    base_transport_moments_bar,
                )
                first_coefficient_bar, first_drds_bar = _moment_pullback(
                    coefficients,
                    dtransport_moments_d_er_bar,
                )
                second_coefficient_bar, second_drds_bar = _moment_pullback(
                    coefficients,
                    dtransport_moments_d_log_nu_star_bar,
                )

                def _direct_drds_bar(coefficient_value, moment_bar):
                    _, pullback = jax.vjp(
                        lambda drds_local: self._transport_moments_from_single_coefficient_vector(
                            coefficient_value,
                            drds_value=drds_local,
                            energy_index=energy_index,
                        ),
                        drds_value,
                    )
                    return pullback(moment_bar)[0]

                _, first_drds_bar_dot = jax.jvp(
                    lambda coefficient_value: _direct_drds_bar(
                        coefficient_value,
                        dtransport_moments_d_er_bar,
                    ),
                    (coefficients,),
                    (first_coeff_dot,),
                )
                _, second_drds_bar_dot = jax.jvp(
                    lambda coefficient_value: _direct_drds_bar(
                        coefficient_value,
                        dtransport_moments_d_log_nu_star_bar,
                    ),
                    (coefficients,),
                    (second_coeff_dot,),
                )
                return (
                    base_coefficient_bar,
                    first_coefficient_bar,
                    second_coefficient_bar,
                    (
                        base_drds_bar,
                        first_drds_bar,
                        first_drds_bar_dot,
                        second_drds_bar,
                        second_drds_bar_dot,
                    ),
                )

            lowdot_pullback = (
                ntx.solve_prepared_coefficient_vector_lowdot_two_pullbacks_with_prepared_and_aux_packed_support_adjoint
                if packed_support_directional_adjoint
                else ntx.solve_prepared_coefficient_vector_lowdot_two_pullbacks_with_prepared_and_aux
            )
            return lowdot_pullback(
                prepared,
                ntx.MonoenergeticCase(nu_hat=nu_hat_value, epsi_hat=epsi_hat_value),
                ntx.MonoenergeticCase(nu_hat=jnp.zeros_like(nu_hat_value), epsi_hat=epsi_dot),
                ntx.MonoenergeticCase(nu_hat=nu_hat_value, epsi_hat=jnp.zeros_like(epsi_hat_value)),
                _bars_and_direct_drds,
            )

        (
            base_nu_hat_bar,
            base_epsi_hat_bar,
            base_geometry_bar,
            first_base_nu_hat_bar,
            first_base_epsi_hat_bar,
            first_nu_hat_bar,
            first_epsi_hat_bar,
            first_base_geometry_bar,
            first_geometry_bar,
            second_base_nu_hat_bar,
            second_base_epsi_hat_bar,
            second_nu_hat_bar,
            second_epsi_hat_bar,
            second_base_geometry_bar,
            second_geometry_bar,
            direct_drds_bars,
        ) = jax.lax.map(
            _one_energy,
            (energy_indices, reference_nu_hat, reference_epsi_hat, epsi_hat_tangent),
        )
        base_drds_bar, _first_base_drds_bar, first_drds_bar, second_base_drds_bar, second_drds_bar = (
            direct_drds_bars
        )
        log_nu_star_nu_hat_bar = self._pullback_log_nu_star_from_nu_hat(
            reference_nu_hat,
            reference_log_nu_star_bar,
        )
        nu_hat_bar = (
            log_nu_star_nu_hat_bar
            + base_nu_hat_bar
            + first_nu_hat_bar
            + second_base_nu_hat_bar
            + second_nu_hat_bar
        )
        epsi_hat_bar = (
            base_epsi_hat_bar
            + first_epsi_hat_bar
            + second_base_epsi_hat_bar
            + second_epsi_hat_bar
        )
        vth_a_bar = jnp.sum(
            first_base_epsi_hat_bar * (-epsi_hat_tangent / vth_a),
            axis=0,
        )
        # Every energy evaluation differentiates the same prepared NTX
        # system. Combine its four directional contributions, then reduce the
        # mapped energy axis before returning a cotangent for that one system.
        prepared_bar = jax.tree_util.tree_map(
            lambda values: jnp.sum(values, axis=0),
            _sum_float_delta_bar_trees(
                prepared,
                base_geometry_bar,
                first_geometry_bar,
                second_base_geometry_bar,
                second_geometry_bar,
            ),
        )
        # ``lax.map`` above leaves an energy axis on every direct ``drds``
        # contribution.  ``drds_value`` is one scalar at this anchor, so its
        # cotangent must sum that axis before the species/objective maps and
        # the anchor scatter accumulation.
        direct_drds_bar = jnp.sum(
            base_drds_bar + first_drds_bar + second_base_drds_bar + second_drds_bar,
            axis=0,
        )
        return (
            nu_hat_bar,
            epsi_hat_bar,
            vth_a_bar,
            prepared_bar,
            direct_drds_bar,
        )

    def _pullback_interpolated_moment_prepared_support_and_drds_only(
        self,
        prepared,
        *,
        drds_value,
        reference_nu_hat,
        reference_epsi_hat,
        vth_a,
        field_bars,
        geometry_implicit_ntx_two_directional: bool = False,
    ):
        """Exact prepared/``drds`` pullback without local state primitive bars.

        This is the support-only counterpart of
        ``_pullback_interpolated_moment_reduced_local_outputs_with_prepared_support_and_drds``.
        The rebuild-support lane holds its local NTX case fixed, so requesting
        ``nu_hat``, ``epsi_hat``, and ``vth_a`` cotangents there is unnecessary.
        The NTX helper retains the same implicit adjoint and prepared-support
        algebra while omitting those case/profile contractions.  The optional
        geometry-only branch is a separate experimental representation: it
        requests only ``GeometryOnGrid`` bars from NTX and restores a complete
        prepared-tree bar with fixed leaves zeroed by NEOPAX.
        """

        ntx = _import_ntx()
        (
            _reference_log_nu_star_bar,
            base_transport_moments_bar,
            dtransport_moments_d_er_bar,
            dtransport_moments_d_log_nu_star_bar,
        ) = _interpolated_response_field_bar_tuple(field_bars)
        epsi_hat_tangent = jnp.asarray(1.0e3, dtype=reference_epsi_hat.dtype) / (
            self.energy_grid.v_norm * vth_a
        )
        energy_indices = jnp.arange(reference_nu_hat.shape[0], dtype=jnp.int32)

        def _one_energy(args):
            energy_index, nu_hat_value, epsi_hat_value, epsi_dot = args

            def _bars_and_direct_drds(coefficients, first_coeff_dot, second_coeff_dot):
                def _moment_pullback(coefficients_value, moment_bar):
                    _, pullback = jax.vjp(
                        lambda coefficient_value, drds_local: self._transport_moments_from_single_coefficient_vector(
                            coefficient_value,
                            drds_value=drds_local,
                            energy_index=energy_index,
                        ),
                        coefficients_value,
                        drds_value,
                    )
                    return pullback(moment_bar)

                base_coefficient_bar, base_drds_bar = _moment_pullback(
                    coefficients,
                    base_transport_moments_bar,
                )
                first_coefficient_bar, first_drds_bar = _moment_pullback(
                    coefficients,
                    dtransport_moments_d_er_bar,
                )
                second_coefficient_bar, second_drds_bar = _moment_pullback(
                    coefficients,
                    dtransport_moments_d_log_nu_star_bar,
                )

                def _direct_drds_bar(coefficient_value, moment_bar):
                    _, pullback = jax.vjp(
                        lambda drds_local: self._transport_moments_from_single_coefficient_vector(
                            coefficient_value,
                            drds_value=drds_local,
                            energy_index=energy_index,
                        ),
                        drds_value,
                    )
                    return pullback(moment_bar)[0]

                _, first_drds_bar_dot = jax.jvp(
                    lambda coefficient_value: _direct_drds_bar(
                        coefficient_value,
                        dtransport_moments_d_er_bar,
                    ),
                    (coefficients,),
                    (first_coeff_dot,),
                )
                _, second_drds_bar_dot = jax.jvp(
                    lambda coefficient_value: _direct_drds_bar(
                        coefficient_value,
                        dtransport_moments_d_log_nu_star_bar,
                    ),
                    (coefficients,),
                    (second_coeff_dot,),
                )
                # The auxiliary carries the already-solved local response
                # fields as well as the direct ``drds`` terms.  It avoids a
                # second local forward response solely for anchor-coordinate
                # interpolation.
                return (
                    base_coefficient_bar,
                    first_coefficient_bar,
                    second_coefficient_bar,
                    (
                        (
                            base_drds_bar,
                            first_drds_bar,
                            first_drds_bar_dot,
                            second_drds_bar,
                            second_drds_bar_dot,
                        ),
                        coefficients,
                        first_coeff_dot,
                        second_coeff_dot,
                    ),
                )

            helper = (
                ntx.solve_prepared_coefficient_vector_lowdot_two_pullbacks_geometry_support_only_and_aux
                if geometry_implicit_ntx_two_directional
                else ntx.solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_and_aux
            )
            return helper(
                prepared,
                ntx.MonoenergeticCase(nu_hat=nu_hat_value, epsi_hat=epsi_hat_value),
                ntx.MonoenergeticCase(nu_hat=jnp.zeros_like(nu_hat_value), epsi_hat=epsi_dot),
                ntx.MonoenergeticCase(nu_hat=nu_hat_value, epsi_hat=jnp.zeros_like(epsi_hat_value)),
                _bars_and_direct_drds,
            )

        (
            base_geometry_bar,
            _first_base_geometry_bar,
            first_geometry_bar,
            second_base_geometry_bar,
            second_geometry_bar,
            auxiliary,
        ) = jax.lax.map(
            _one_energy,
            (energy_indices, reference_nu_hat, reference_epsi_hat, epsi_hat_tangent),
        )
        (
            direct_drds_bars,
            coefficient_scan,
            first_coefficient_dot_scan,
            second_coefficient_dot_scan,
        ) = auxiliary
        base_drds_bar, _first_base_drds_bar, first_drds_bar, second_base_drds_bar, second_drds_bar = direct_drds_bars
        if geometry_implicit_ntx_two_directional:
            geometry_bar = jax.tree_util.tree_map(
                lambda values: jnp.sum(values, axis=0),
                _sum_float_delta_bar_trees(
                    prepared.geometry,
                    base_geometry_bar,
                    first_geometry_bar,
                    second_base_geometry_bar,
                    second_geometry_bar,
                ),
            )
            prepared_bar = dataclasses.replace(
                _float_delta_tree_like(prepared),
                geometry=geometry_bar,
            )
        else:
            prepared_bar = jax.tree_util.tree_map(
                lambda values: jnp.sum(values, axis=0),
                _sum_float_delta_bar_trees(
                    prepared,
                    base_geometry_bar,
                    first_geometry_bar,
                    second_base_geometry_bar,
                    second_geometry_bar,
                ),
            )
        direct_drds_bar = jnp.sum(
            base_drds_bar + first_drds_bar + second_base_drds_bar + second_drds_bar,
            axis=0,
        )
        primal_response = (
            self._log_nu_star_from_nu_hat(reference_nu_hat),
            self._transport_moments_from_coefficient_scan(
                coefficient_scan,
                drds_value=drds_value,
            ),
            jax.jvp(
                lambda values: self._transport_moments_from_coefficient_scan(
                    values,
                    drds_value=drds_value,
                ),
                (coefficient_scan,),
                (first_coefficient_dot_scan,),
            )[1],
            jax.jvp(
                lambda values: self._transport_moments_from_coefficient_scan(
                    values,
                    drds_value=drds_value,
                ),
                (coefficient_scan,),
                (second_coefficient_dot_scan,),
            )[1],
        )
        return prepared_bar, direct_drds_bar, primal_response

    def _pullback_interpolated_moment_prepared_support_and_drds_only_multi_rhs(
        self,
        prepared,
        *,
        drds_value,
        reference_nu_hat,
        reference_epsi_hat,
        vth_a,
        field_bars,
        native_factorized_ntx_rhs: bool = False,
        native_compact_ntx_rhs: bool = False,
        native_compact_residual_ntx_rhs: bool = False,
        reuse_joint_moment_drds_jvp: bool = False,
        return_native_vmec_coefficient_bars: bool = False,
        native_vmec_coefficient_bars_only: bool = False,
        native_vmec_direct_directional_product_rule: bool = False,
        native_direct_coefficient_pullback: bool = False,
        native_per_energy_call_boundary: bool = False,
        stream_native_compact_energy: bool = False,
        return_case_bars: bool = False,
        include_second_direction_base_prepared: bool = True,
    ):
        """Batched exact prepared/``drds`` transpose for one local species.

        The leading axis of ``field_bars`` is an arbitrary on-device RHS
        batch.  For every energy, NTX constructs the local primal/factor
        residual once and applies the exact support-only low-dot adjoint to
        that RHS batch.  The factors are consumed before this method returns;
        the unbatched coefficient scans are returned solely to construct the
        already-required anchor response for interpolation.

        This is a private building block.  It is not selected by any current
        transport/reverse mode.
        """

        ntx = _import_ntx()
        if return_native_vmec_coefficient_bars and not native_factorized_ntx_rhs:
            raise ValueError(
                "Native VMEC coefficient bars require the native matrix-RHS NTX helper."
            )
        if native_vmec_coefficient_bars_only and not return_native_vmec_coefficient_bars:
            raise ValueError(
                "native_vmec_coefficient_bars_only requires native VMEC coefficient bars."
            )
        helper_name = (
            "solve_prepared_coefficient_vector_lowdot_two_pullbacks_"
            "prepared_support_only_native_multi_rhs_compact_residual_and_aux"
            if native_compact_residual_ntx_rhs
            else (
                "solve_prepared_coefficient_vector_lowdot_two_pullbacks_"
                "prepared_support_only_native_multi_rhs_compact_and_aux"
                if native_compact_ntx_rhs
                else (
                    "solve_prepared_coefficient_vector_lowdot_two_pullbacks_"
                    "prepared_support_only_native_multi_rhs_and_aux"
                    if native_factorized_ntx_rhs
                    else "solve_prepared_coefficient_vector_lowdot_two_pullbacks_"
                    "prepared_support_only_multi_rhs_and_aux"
                )
            )
        )
        multi_rhs_helper = getattr(
            ntx,
            helper_name,
            None,
        )
        if not callable(multi_rhs_helper):
            raise RuntimeError(
                "The selected multi-RHS prepared-support pullback requires the current NTX helper."
            )
        (
            _reference_log_nu_star_bars,
            base_transport_moments_bars,
            dtransport_moments_d_er_bars,
            dtransport_moments_d_log_nu_star_bars,
        ) = _interpolated_response_field_bar_tuple(field_bars)
        # The local thermal speed is scalar, while the following ``lax.map``
        # iterates over energy.  Materialize the energy axis before mapping.
        epsi_hat_tangent = jnp.broadcast_to(
            jnp.asarray(1.0e3, dtype=reference_epsi_hat.dtype)
            / (self.energy_grid.v_norm * vth_a),
            reference_epsi_hat.shape,
        )
        energy_indices = jnp.arange(reference_nu_hat.shape[0], dtype=jnp.int32)

        def _one_energy(args):
            energy_index, nu_hat_value, epsi_hat_value, epsi_dot = args

            def _bars_and_direct_drds(coefficients, first_coeff_dot, second_coeff_dot):
                def _moment_pullback_batched(coefficients_value, moment_bars):
                    def _one_rhs(moment_bar):
                        _, pullback = jax.vjp(
                            lambda coefficient_value, drds_local: self._transport_moments_from_single_coefficient_vector(
                                coefficient_value,
                                drds_value=drds_local,
                                energy_index=energy_index,
                            ),
                            coefficients_value,
                            drds_value,
                        )
                        return pullback(moment_bar)

                    return jax.vmap(_one_rhs)(moment_bars)

                base_coefficient_bars, base_drds_bars = _moment_pullback_batched(
                    coefficients, base_transport_moments_bars
                )
                (
                    first_coefficient_bars,
                    first_drds_bars,
                ), (
                    first_coefficient_bars_dot,
                    first_drds_bars_dot,
                ) = jax.jvp(
                    lambda coefficient_value: _moment_pullback_batched(
                        coefficient_value, dtransport_moments_d_er_bars
                    ),
                    (coefficients,),
                    (first_coeff_dot,),
                )
                (
                    second_coefficient_bars,
                    second_drds_bars,
                ), (
                    second_coefficient_bars_dot,
                    second_drds_bars_dot,
                ) = jax.jvp(
                    lambda coefficient_value: _moment_pullback_batched(
                        coefficient_value, dtransport_moments_d_log_nu_star_bars
                    ),
                    (coefficients,),
                    (second_coeff_dot,),
                )

                if not reuse_joint_moment_drds_jvp:
                    def _direct_drds_bars(coefficient_value, moment_bars):
                        def _one_rhs(moment_bar):
                            _, pullback = jax.vjp(
                                lambda drds_local: self._transport_moments_from_single_coefficient_vector(
                                    coefficient_value,
                                    drds_value=drds_local,
                                    energy_index=energy_index,
                                ),
                                drds_value,
                            )
                            return pullback(moment_bar)[0]

                        return jax.vmap(_one_rhs)(moment_bars)

                    # Kept as the established path.  The opt-in native mode
                    # below reuses the identical drds tangent returned by the
                    # joint coefficient/drds moment pullback JVP above.
                    _, first_drds_bars_dot = jax.jvp(
                        lambda coefficient_value: _direct_drds_bars(
                            coefficient_value, dtransport_moments_d_er_bars
                        ),
                        (coefficients,),
                        (first_coeff_dot,),
                    )
                    _, second_drds_bars_dot = jax.jvp(
                        lambda coefficient_value: _direct_drds_bars(
                            coefficient_value, dtransport_moments_d_log_nu_star_bars
                        ),
                        (coefficients,),
                        (second_coeff_dot,),
                    )
                return (
                    base_coefficient_bars,
                    first_coefficient_bars,
                    second_coefficient_bars,
                    (
                        base_drds_bars,
                        first_drds_bars,
                        first_drds_bars_dot,
                        second_drds_bars,
                        second_drds_bars_dot,
                    ),
                    (
                        first_coefficient_bars_dot,
                        second_coefficient_bars_dot,
                    ),
                )

            helper_kwargs = dict(return_primal_outputs=True)
            if return_native_vmec_coefficient_bars:
                # NTX's coefficient return uses the compact primal-prepared
                # view.  The grouped primal/factorisation and matrix-RHS
                # adjoint are unchanged; only the returned prepared payload
                # is reduced before crossing this local boundary.
                helper_kwargs["_compact_result"] = True
                helper_kwargs["return_vmec_coefficient_bars"] = True
                helper_kwargs["native_vmec_coefficient_bars_only"] = bool(
                    native_vmec_coefficient_bars_only
                )
                helper_kwargs["native_vmec_direct_directional_product_rule"] = bool(
                    native_vmec_direct_directional_product_rule
                )
                helper_kwargs["native_direct_coefficient_pullback"] = bool(
                    native_direct_coefficient_pullback
                )
            if (
                native_factorized_ntx_rhs
                or native_compact_ntx_rhs
                or native_compact_residual_ntx_rhs
            ) and return_case_bars:
                # The native matrix-RHS helper can return the already-formed
                # case bars.  The normal support-only helper deliberately
                # omits them, so retain its original contract unchanged.
                helper_kwargs["return_case_bars"] = True
            helper_result = multi_rhs_helper(
                prepared,
                ntx.MonoenergeticCase(nu_hat=nu_hat_value, epsi_hat=epsi_hat_value),
                ntx.MonoenergeticCase(
                    nu_hat=jnp.zeros_like(nu_hat_value), epsi_hat=epsi_dot
                ),
                ntx.MonoenergeticCase(
                    nu_hat=nu_hat_value, epsi_hat=jnp.zeros_like(epsi_hat_value)
                ),
                _bars_and_direct_drds,
                **helper_kwargs,
            )
            if (
                native_factorized_ntx_rhs
                or native_compact_ntx_rhs
                or native_compact_residual_ntx_rhs
            ) and return_case_bars:
                if return_native_vmec_coefficient_bars:
                    (
                        primal_outputs,
                        support_result,
                        case_bar_components,
                        native_vmec_coefficient_bars,
                    ) = helper_result
                    return (
                        *support_result[:-1],
                        support_result[-1],
                        primal_outputs,
                        case_bar_components,
                        native_vmec_coefficient_bars,
                    )
                primal_outputs, support_result, case_bar_components = helper_result
                return (*support_result[:-1], support_result[-1], primal_outputs, case_bar_components)
            primal_outputs, support_result = helper_result
            return (*support_result[:-1], support_result[-1], primal_outputs)

        per_energy_pullback = (
            jax.jit(_one_energy, inline=False)
            if native_per_energy_call_boundary
            else _one_energy
        )

        if (
            native_compact_ntx_rhs
            and return_case_bars
            and stream_native_compact_energy
        ):
            # The full native helper returns five objective-batched prepared
            # trees per energy.  The compact helper has already reduced that
            # to one, so accumulate it while scanning energy rather than
            # materialising an energy-by-objective prepared payload and
            # reducing it afterwards.  RHS remain matrix columns throughout.
            first_output = per_energy_pullback(
                (
                    energy_indices[0],
                    reference_nu_hat[0],
                    reference_epsi_hat[0],
                    epsi_hat_tangent[0],
                )
            )
            (
                first_prepared_bars,
                first_direct_drds_bars,
                first_primal_outputs,
                first_case_bar_components,
            ) = first_output

            def _sanitize_rhs_prepared_leaf(primal_leaf, bar_leaf):
                primal_arr = jnp.asarray(primal_leaf)
                bar_arr = jnp.asarray(bar_leaf)
                dtype = (
                    primal_arr.dtype
                    if jnp.issubdtype(primal_arr.dtype, jnp.inexact)
                    else jnp.float64
                )
                if bar_arr.dtype == jax.dtypes.float0:
                    return jnp.zeros(bar_arr.shape, dtype=dtype)
                if jnp.issubdtype(primal_arr.dtype, jnp.inexact):
                    return jnp.asarray(bar_leaf, dtype=primal_arr.dtype)
                # Preserve the RHS axis for static prepared leaves.  The
                # scalar sanitizer intentionally removes it, which is wrong
                # for this matrix-RHS accumulator.
                return jnp.zeros(bar_arr.shape, dtype=dtype)

            first_prepared_bars = jax.tree_util.tree_map(
                _sanitize_rhs_prepared_leaf, prepared, first_prepared_bars
            )

            def _accumulate_energy(carry, args):
                prepared_accum, drds_accum = carry
                prepared_value, drds_value, primal_value, case_value = per_energy_pullback(args)
                prepared_value = jax.tree_util.tree_map(
                    _sanitize_rhs_prepared_leaf, prepared, prepared_value
                )
                return (
                    jax.tree_util.tree_map(
                        lambda total, value: total + value,
                        prepared_accum,
                        prepared_value,
                    ),
                    tuple(
                        total + value
                        for total, value in zip(drds_accum, drds_value, strict=True)
                    ),
                ), (primal_value, case_value)

            (
                compact_prepared_bars,
                direct_drds_bars,
            ), (tail_primal_outputs, tail_case_bar_components) = jax.lax.scan(
                _accumulate_energy,
                (first_prepared_bars, first_direct_drds_bars),
                (
                    energy_indices[1:],
                    reference_nu_hat[1:],
                    reference_epsi_hat[1:],
                    epsi_hat_tangent[1:],
                ),
            )
            primal_outputs = jax.tree_util.tree_map(
                lambda first, tail: jnp.concatenate((first[None, ...], tail), axis=0),
                first_primal_outputs,
                tail_primal_outputs,
            )
            native_case_bar_components = jax.tree_util.tree_map(
                lambda first, tail: jnp.concatenate((first[None, ...], tail), axis=0),
                first_case_bar_components,
                tail_case_bar_components,
            )
            compact_energy_streamed = True
        else:
            mapped_outputs = jax.lax.map(
                per_energy_pullback,
                (energy_indices, reference_nu_hat, reference_epsi_hat, epsi_hat_tangent),
            )
            compact_energy_streamed = False
        if return_native_vmec_coefficient_bars:
            (
                compact_prepared_bars,
                direct_drds_bars,
                primal_outputs,
                native_case_bar_components,
                native_vmec_coefficient_bars,
            ) = mapped_outputs
        elif (
            native_compact_ntx_rhs or native_compact_residual_ntx_rhs
        ) and return_case_bars and not compact_energy_streamed:
            (
                compact_prepared_bars,
                direct_drds_bars,
                primal_outputs,
                native_case_bar_components,
            ) = mapped_outputs
        elif (
            native_factorized_ntx_rhs
            and return_case_bars
            and not native_compact_ntx_rhs
            and not native_compact_residual_ntx_rhs
        ):
            (
                base_prepared_bars,
                first_base_prepared_bars,
                first_prepared_bars,
                second_base_prepared_bars,
                second_prepared_bars,
                direct_drds_bars,
                primal_outputs,
                native_case_bar_components,
            ) = mapped_outputs
        elif not (
            (native_compact_ntx_rhs or native_compact_residual_ntx_rhs)
            and return_case_bars
        ):
            (
                base_prepared_bars,
                first_base_prepared_bars,
                first_prepared_bars,
                second_base_prepared_bars,
                second_prepared_bars,
                direct_drds_bars,
                primal_outputs,
            ) = mapped_outputs
        # ``second_base_prepared_bars`` is the extra base pullback required
        # for the *nu_hat case bar* of d/dlog(nu_hat), because that tangent is
        # itself nu_hat.  It is not a prepared-system derivative: the local
        # prepared payload holds nu_hat fixed.  Keep the historical default
        # for existing private callers, while the native exactness adapter
        # explicitly excludes it.
        prepared_bar_terms = (
            (compact_prepared_bars,)
            if (
                native_compact_ntx_rhs
                or native_compact_residual_ntx_rhs
                or return_native_vmec_coefficient_bars
            )
            and return_case_bars
            else
            (
                base_prepared_bars,
                first_prepared_bars,
                second_prepared_bars,
            )
            if not include_second_direction_base_prepared
            else (
                base_prepared_bars,
                first_prepared_bars,
                second_base_prepared_bars,
                second_prepared_bars,
            )
        )
        if native_compact_ntx_rhs and return_case_bars and compact_energy_streamed:
            prepared_bars = compact_prepared_bars
        else:
            prepared_bars = jax.tree_util.tree_map(
                lambda values: jnp.sum(values, axis=0),
                _sum_float_delta_bar_trees(
                    prepared,
                    *prepared_bar_terms,
                ),
            )
        (
            base_drds_bars,
            _first_base_drds_bars,
            first_drds_bars,
            second_base_drds_bars,
            second_drds_bars,
        ) = direct_drds_bars
        # As for the prepared bar above, the second base term belongs only to
        # the nu_hat case cotangent of the log-nu direction. ``drds`` is not
        # that direction, so the native exact adapter excludes it.
        direct_drds_bars = (
            base_drds_bars + first_drds_bars + second_drds_bars
            if not include_second_direction_base_prepared
            else (
                base_drds_bars
                + first_drds_bars
                + second_base_drds_bars
                + second_drds_bars
            )
        )
        if not (native_compact_ntx_rhs and return_case_bars and compact_energy_streamed):
            direct_drds_bars = jnp.sum(direct_drds_bars, axis=0)
        coefficient_scan, first_coefficient_dot_scan, second_coefficient_dot_scan = primal_outputs
        primal_response = (
            self._log_nu_star_from_nu_hat(reference_nu_hat),
            self._transport_moments_from_coefficient_scan(
                coefficient_scan,
                drds_value=drds_value,
            ),
            jax.jvp(
                lambda values: self._transport_moments_from_coefficient_scan(
                    values,
                    drds_value=drds_value,
                ),
                (coefficient_scan,),
                (first_coefficient_dot_scan,),
            )[1],
            jax.jvp(
                lambda values: self._transport_moments_from_coefficient_scan(
                    values,
                    drds_value=drds_value,
                ),
                (coefficient_scan,),
                (second_coefficient_dot_scan,),
            )[1],
        )
        if (
            native_factorized_ntx_rhs
            or native_compact_ntx_rhs
            or native_compact_residual_ntx_rhs
        ) and return_case_bars:
            (
                base_nu_hat_bars,
                base_epsi_hat_bars,
                _first_base_nu_hat_bars,
                first_base_epsi_hat_bars,
                first_nu_hat_bars,
                first_epsi_hat_bars,
                second_base_nu_hat_bars,
                second_base_epsi_hat_bars,
                second_nu_hat_bars,
                second_epsi_hat_bars,
            ) = native_case_bar_components
            # ``lax.map`` left energy first, whereas the local scan primitive
            # receives one RHS batch of energy values.  This is the same
            # contraction used by the full joint state/support pullback.
            # ``reference_log_nu_star`` is one energy-averaged scalar per
            # objective.  Form its transpose explicitly as (energy, rhs):
            # the scalar helper is intentionally unbatched and would otherwise
            # align an objective axis with the energy axis.
            log_nu_weights = jnp.asarray(
                self.energy_grid.xWeights, dtype=reference_nu_hat.dtype
            )
            log_nu_weights = log_nu_weights / jnp.maximum(
                jnp.sum(log_nu_weights), 1.0e-30
            )
            safe_nu_hat = jnp.maximum(reference_nu_hat, 1.0e-30)
            active_nu_hat = jnp.asarray(
                reference_nu_hat >= 1.0e-30, dtype=reference_nu_hat.dtype
            )
            log_nu_star_nu_hat_bars = (
                active_nu_hat * log_nu_weights / safe_nu_hat
            )[:, None] * _reference_log_nu_star_bars[None, :]
            nu_hat_bars = jnp.swapaxes(
                log_nu_star_nu_hat_bars
                + base_nu_hat_bars
                + first_nu_hat_bars
                + second_base_nu_hat_bars
                + second_nu_hat_bars,
                0,
                1,
            )
            epsi_hat_bars = jnp.swapaxes(
                # The second base bar is for the second physical direction
                # ``(nu_hat, 0)``.  Its epsilon component therefore multiplies
                # that direction's constant zero epsilon slot, rather than
                # the primal epsilon coordinate.  Do not route it through
                # ``drds -> epsi_hat``.
                base_epsi_hat_bars
                + first_epsi_hat_bars
                + second_epsi_hat_bars,
                0,
                1,
            )
            vth_a_energy = jnp.broadcast_to(vth_a, reference_epsi_hat.shape)
            vth_a_bars = jnp.sum(
                first_base_epsi_hat_bars
                * (-epsi_hat_tangent[:, None] / vth_a_energy[:, None]),
                axis=0,
            )
            result = (prepared_bars, direct_drds_bars, primal_response, (
                nu_hat_bars,
                epsi_hat_bars,
                vth_a_bars,
            ))
            if return_native_vmec_coefficient_bars:
                return (
                    *result,
                    jax.tree_util.tree_map(
                        lambda values: jnp.sum(values, axis=0),
                        native_vmec_coefficient_bars,
                    ),
                )
            return result
        return prepared_bars, direct_drds_bars, primal_response

    def _pullback_interpolated_moment_prepared_support_and_drds_only_native_multi_rhs(
        self,
        prepared,
        *,
        drds_value,
        reference_nu_hat,
        reference_epsi_hat,
        vth_a,
        field_bars,
        return_case_bars: bool = False,
    ):
        """Private adapter for NTX's native matrix-RHS support helper.

        This is deliberately not a reverse-dispatch hook.  It exists solely
        for the exact small-model comparison before any experimental selector
        is considered.
        """

        return self._pullback_interpolated_moment_prepared_support_and_drds_only_multi_rhs(
            prepared,
            drds_value=drds_value,
            reference_nu_hat=reference_nu_hat,
            reference_epsi_hat=reference_epsi_hat,
            vth_a=vth_a,
            field_bars=field_bars,
            native_factorized_ntx_rhs=True,
            return_case_bars=return_case_bars,
            include_second_direction_base_prepared=False,
        )

    def _pullback_interpolated_moment_prepared_support_and_drds_only_native_multi_rhs_compact(
        self,
        prepared,
        *,
        drds_value,
        reference_nu_hat,
        reference_epsi_hat,
        vth_a,
        field_bars,
        return_case_bars: bool = False,
    ):
        """Compact-return counterpart of the native matrix-RHS adapter."""

        return self._pullback_interpolated_moment_prepared_support_and_drds_only_multi_rhs(
            prepared,
            drds_value=drds_value,
            reference_nu_hat=reference_nu_hat,
            reference_epsi_hat=reference_epsi_hat,
            vth_a=vth_a,
            field_bars=field_bars,
            native_factorized_ntx_rhs=True,
            native_compact_ntx_rhs=True,
            return_case_bars=return_case_bars,
            include_second_direction_base_prepared=False,
        )

    def _pullback_interpolated_moment_prepared_support_and_drds_only_native_multi_rhs_reuse_moment_drds_jvp(
        self,
        prepared,
        *,
        drds_value,
        reference_nu_hat,
        reference_epsi_hat,
        vth_a,
        field_bars,
        return_case_bars: bool = False,
    ):
        """Native matrix-RHS adapter without duplicate local ``drds`` JVPs.

        The joint moment pullback already produces the directional ``drds``
        cotangents needed by the exact low-dot chain.  This isolated variant
        reuses those values instead of differentiating a second, drds-only
        VJP.  It preserves the native factorization, objective RHS batch, and
        complete return contract of the existing native adapter.
        """

        return self._pullback_interpolated_moment_prepared_support_and_drds_only_multi_rhs(
            prepared,
            drds_value=drds_value,
            reference_nu_hat=reference_nu_hat,
            reference_epsi_hat=reference_epsi_hat,
            vth_a=vth_a,
            field_bars=field_bars,
            native_factorized_ntx_rhs=True,
            reuse_joint_moment_drds_jvp=True,
            return_case_bars=return_case_bars,
            include_second_direction_base_prepared=False,
        )

    def _pullback_interpolated_moment_prepared_support_and_drds_only_native_multi_rhs_reuse_moment_drds_jvp_with_vmec_coefficients(
        self,
        prepared,
        *,
        drds_value,
        reference_nu_hat,
        reference_epsi_hat,
        vth_a,
        field_bars,
        return_case_bars: bool = False,
        native_vmec_coefficient_bars_only: bool = False,
        native_vmec_direct_directional_product_rule: bool = False,
        native_direct_coefficient_pullback: bool = False,
        native_per_energy_call_boundary: bool = False,
    ):
        """Return the validated native support bar plus VMEC coefficient bars.

        This is intentionally private and unselected.  It retains the same
        grouped low-dot NTX adjoint as the current fast ``drds``-reuse mode,
        but exposes the already-available coefficient cotangent in a parallel
        channel instead of asking NEOPAX to differentiate the prepared tree.
        """

        return self._pullback_interpolated_moment_prepared_support_and_drds_only_multi_rhs(
            prepared,
            drds_value=drds_value,
            reference_nu_hat=reference_nu_hat,
            reference_epsi_hat=reference_epsi_hat,
            vth_a=vth_a,
            field_bars=field_bars,
            native_factorized_ntx_rhs=True,
            reuse_joint_moment_drds_jvp=True,
            return_native_vmec_coefficient_bars=True,
            native_vmec_coefficient_bars_only=native_vmec_coefficient_bars_only,
            native_vmec_direct_directional_product_rule=(
                native_vmec_direct_directional_product_rule
            ),
            native_direct_coefficient_pullback=native_direct_coefficient_pullback,
            native_per_energy_call_boundary=native_per_energy_call_boundary,
            return_case_bars=return_case_bars,
            include_second_direction_base_prepared=False,
        )

    def _pullback_interpolated_moment_prepared_support_and_drds_only_native_multi_rhs_compact_residual(
        self,
        prepared,
        *,
        drds_value,
        reference_nu_hat,
        reference_epsi_hat,
        vth_a,
        field_bars,
        return_case_bars: bool = False,
    ):
        """Private split-residual adapter; no reverse selector calls it yet."""
        return self._pullback_interpolated_moment_prepared_support_and_drds_only_multi_rhs(
            prepared,
            drds_value=drds_value,
            reference_nu_hat=reference_nu_hat,
            reference_epsi_hat=reference_epsi_hat,
            vth_a=vth_a,
            field_bars=field_bars,
            native_factorized_ntx_rhs=True,
            native_compact_residual_ntx_rhs=True,
            return_case_bars=return_case_bars,
            include_second_direction_base_prepared=False,
        )

    def _pullback_interpolated_moment_prepared_support_and_drds_only_native_multi_rhs_compact_residual_reuse_moment_drds_jvp(
        self,
        prepared,
        *,
        drds_value,
        reference_nu_hat,
        reference_epsi_hat,
        vth_a,
        field_bars,
        return_case_bars: bool = False,
    ):
        """Split-residual adapter retaining the validated moment-``drds`` reuse."""
        return self._pullback_interpolated_moment_prepared_support_and_drds_only_multi_rhs(
            prepared,
            drds_value=drds_value,
            reference_nu_hat=reference_nu_hat,
            reference_epsi_hat=reference_epsi_hat,
            vth_a=vth_a,
            field_bars=field_bars,
            native_factorized_ntx_rhs=True,
            native_compact_residual_ntx_rhs=True,
            reuse_joint_moment_drds_jvp=True,
            return_case_bars=return_case_bars,
            include_second_direction_base_prepared=False,
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

    def _pullback_local_scan_inputs_and_drds_from_primitives(
        self,
        *,
        drds_value,
        species_index: int,
        er_value,
        temperature_local,
        density_local,
        collisionality_kind,
        reference_nu_hat_bar,
        reference_epsi_hat_bar,
        vth_a_bar,
    ):
        """Transpose local scan primitives including the differentiable drds input.

        The existing state-only helper intentionally holds ``drds`` fixed.
        The joint state/support rebuild path also needs its implicit support
        cotangent, so this companion exposes that fourth input without tracing
        through an NTX coefficient solve.
        """

        def _local_scan_from_inputs(
            drds_local,
            er_local,
            temperature_value,
            density_value,
        ):
            vthermal_value = get_v_thermal(self.species.mass, temperature_value)
            return self._local_scan_inputs(
                drds_value=drds_local,
                species_index=species_index,
                er_value=er_local,
                temperature_local=temperature_value,
                density_local=density_value,
                vthermal_local=vthermal_value,
                collisionality_kind=collisionality_kind,
            )

        _, pullback = jax.vjp(
            _local_scan_from_inputs,
            drds_value,
            er_value,
            temperature_local,
            density_local,
        )
        return pullback((reference_nu_hat_bar, reference_epsi_hat_bar, vth_a_bar))

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

    def _pullback_interpolated_moment_response_local_fields_and_prepared_support_and_drds_flat_prepared(
        self,
        prepared,
        *,
        drds_value,
        er_value,
        temperature_local,
        density_local,
        collisionality_kind,
        field_bars,
        packed_support_directional_adjoint: bool = False,
        return_primal_response: bool = False,
        native_factorized_ntx_rhs: bool = False,
        reuse_joint_moment_drds_jvp: bool = False,
        return_native_vmec_coefficient_bars: bool = False,
        omit_generic_prepared_carry: bool = False,
    ):
        """Joint local pullback, returning prepared-support leaves unflattened.

        A prepared NTX system contains validated geometry dataclasses.  Do not
        return those dataclasses through the nested species/objective ``vmap``
        operations below: JAX batches their leaves one at a time and the
        dataclass validation sees a temporary sentinel in static mode fields.
        The caller rebuilds this static pytree after both maps complete.
        """

        vthermal_local = get_v_thermal(self.species.mass, temperature_local)
        species_indices = jnp.arange(int(self.species.number_species), dtype=jnp.int32)
        if return_native_vmec_coefficient_bars and not native_factorized_ntx_rhs:
            raise ValueError(
                "native VMEC coefficient bars require native_factorized_ntx_rhs=True."
            )
        if return_native_vmec_coefficient_bars and return_primal_response:
            raise ValueError(
                "native VMEC coefficient bars cannot be combined with local-primal reuse."
            )
        if omit_generic_prepared_carry and not return_native_vmec_coefficient_bars:
            raise ValueError(
                "omitting the generic prepared carry requires native VMEC coefficient bars."
            )

        def _per_species_pullback(species_index, species_field_bars):
            reference_nu_hat, reference_epsi_hat, vth_a = (
                self._interpolated_moment_local_scan_primitives(
                    drds_value=drds_value,
                    species_index=species_index,
                    er_value=er_value,
                    temperature_local=temperature_local,
                    density_local=density_local,
                    vthermal_local=vthermal_local,
                    collisionality_kind=collisionality_kind,
                )
            )
            if native_factorized_ntx_rhs:
                native_result = self._pullback_interpolated_moment_prepared_support_and_drds_only_multi_rhs(
                    prepared,
                    drds_value=drds_value,
                    reference_nu_hat=reference_nu_hat,
                    reference_epsi_hat=reference_epsi_hat,
                    vth_a=vth_a,
                    field_bars=species_field_bars,
                    native_factorized_ntx_rhs=True,
                    reuse_joint_moment_drds_jvp=reuse_joint_moment_drds_jvp,
                    return_case_bars=True,
                    return_native_vmec_coefficient_bars=(
                        return_native_vmec_coefficient_bars
                    ),
                    native_vmec_coefficient_bars_only=(
                        return_native_vmec_coefficient_bars
                    ),
                    include_second_direction_base_prepared=False,
                )
                if return_native_vmec_coefficient_bars:
                    (
                        prepared_bar,
                        direct_drds_bar,
                        _primal_response,
                        (
                            reference_nu_hat_bar,
                            reference_epsi_hat_bar,
                            vth_a_bar,
                        ),
                        native_vmec_coefficient_bars,
                    ) = native_result
                else:
                    (
                        prepared_bar,
                        direct_drds_bar,
                        _primal_response,
                        (
                            reference_nu_hat_bar,
                            reference_epsi_hat_bar,
                            vth_a_bar,
                        ),
                    ) = native_result
            else:
                (
                    reference_nu_hat_bar,
                    reference_epsi_hat_bar,
                    vth_a_bar,
                    prepared_bar,
                    direct_drds_bar,
                ) = self._pullback_interpolated_moment_reduced_local_outputs_with_prepared_support_and_drds(
                    prepared,
                    drds_value=drds_value,
                    reference_nu_hat=reference_nu_hat,
                    reference_epsi_hat=reference_epsi_hat,
                    vth_a=vth_a,
                    field_bars=species_field_bars,
                    packed_support_directional_adjoint=packed_support_directional_adjoint,
                )
            def _primitive_pullback(nu_hat_bar, epsi_hat_bar, vth_bar):
                return self._pullback_local_scan_inputs_and_drds_from_primitives(
                    drds_value=drds_value,
                    species_index=species_index,
                    er_value=er_value,
                    temperature_local=temperature_local,
                    density_local=density_local,
                    collisionality_kind=collisionality_kind,
                    reference_nu_hat_bar=nu_hat_bar,
                    reference_epsi_hat_bar=epsi_hat_bar,
                    vth_a_bar=vth_bar,
                )

            if native_factorized_ntx_rhs:
                # NTX returns native case bars with a leading matrix-RHS
                # axis.  This is only the cheap primitive-chain transpose;
                # batching it here preserves the single native NTX adjoint.
                (
                    implicit_drds_bar,
                    er_bar,
                    temperature_bar,
                    density_bar,
                ) = jax.vmap(_primitive_pullback)(
                    reference_nu_hat_bar,
                    reference_epsi_hat_bar,
                    vth_a_bar,
                )
            else:
                (
                    implicit_drds_bar,
                    er_bar,
                    temperature_bar,
                    density_bar,
                ) = _primitive_pullback(
                    reference_nu_hat_bar,
                    reference_epsi_hat_bar,
                    vth_a_bar,
                )
            prepared_leaves = (
                ()
                if omit_generic_prepared_carry
                else tuple(jax.tree_util.tree_leaves(prepared_bar))
            )
            pullback_result = (
                implicit_drds_bar + direct_drds_bar,
                er_bar,
                temperature_bar,
                density_bar,
                prepared_leaves,
            )
            if return_native_vmec_coefficient_bars:
                return (*pullback_result, native_vmec_coefficient_bars)
            if not return_primal_response:
                return pullback_result
            primal_response = self._interpolated_moment_reduced_local_outputs_from_primitives(
                prepared,
                drds_value=drds_value,
                nu_hat_a=reference_nu_hat,
                epsi_hat_a=reference_epsi_hat,
                vth_a=vth_a,
            )
            return (*pullback_result, primal_response)

        mapped_pullback = jax.vmap(_per_species_pullback, in_axes=(0, 0))(
            species_indices, field_bars
        )
        (
            drds_species_bar,
            er_species_bar,
            temperature_species_bar,
            density_species_bar,
            prepared_geometry_species_bar_leaves,
        ) = mapped_pullback[:5]
        prepared_leaves = (
            ()
            if omit_generic_prepared_carry
            else tuple(jnp.sum(values, axis=0) for values in prepared_geometry_species_bar_leaves)
        )
        pullback_result = (
            jnp.sum(drds_species_bar, axis=0),
            jnp.sum(er_species_bar, axis=0),
            jnp.sum(temperature_species_bar, axis=0),
            jnp.sum(density_species_bar, axis=0),
            prepared_leaves,
        )
        if return_native_vmec_coefficient_bars:
            return (
                *pullback_result,
                jax.tree_util.tree_map(
                    lambda values: jnp.sum(values, axis=0), mapped_pullback[5]
                ),
            )
        if return_primal_response:
            return (*pullback_result, mapped_pullback[5])
        return pullback_result

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

    def _lij_from_quadratic_response_at_reference(
        self,
        response: NTXFullStateQuadraticPreparedCoefficientResponse,
        *,
        axis: str,
    ):
        """Return Lij at a quadratic response anchor without another NTX solve."""

        if axis not in {"center", "face"}:
            raise ValueError("axis must be 'center' or 'face'.")
        state = response.reference_state
        evaluated = build_evaluated_transport_state(
            state,
            self.geometry,
            bc_density=self.bc_density,
            bc_temperature=self.bc_temperature,
            density_floor=self.density_floor,
            temperature_floor=self.temperature_floor,
        )
        axis_state = getattr(evaluated, axis)
        support = self._static_support()
        channels = getattr(support, f"{axis}_channels")
        radius_coordinates = self.geometry.r_grid if axis == "center" else self.geometry.r_grid_half
        coefficients = response.coefficient_response.reference_coefficients
        radius_indices = jnp.arange(axis_state.Er.shape[0], dtype=jnp.int32)

        def _per_radius(radius_index):
            drds_value = jax.lax.dynamic_index_in_dim(
                channels.drds, radius_index, axis=0, keepdims=False
            )
            coefficient_scan = jax.lax.dynamic_index_in_dim(
                coefficients, radius_index, axis=0, keepdims=False
            )
            return jax.vmap(
                lambda one_species_scan: self._transport_moments_from_coefficient_scan(
                    one_species_scan, drds_value=drds_value
                )
            )(coefficient_scan)

        moments_by_radius = self._map_radius_axis_regularized_at_axis0(
            _per_radius, radius_indices, radius_coordinates
        )
        moments = jnp.swapaxes(moments_by_radius, 0, 1)
        v_thermal = get_v_thermal(self.species.mass, axis_state.temperature)
        return self._batched_lij_from_transport_moments(moments, v_thermal)

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

    def _assemble_center_fluxes(self, Er, temperature, density, dndr_all, dTdr_all, lij):
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
        evaluated = build_evaluated_transport_state(
            state,
            self.geometry,
            bc_density=self.bc_density,
            bc_temperature=self.bc_temperature,
            density_floor=self.density_floor,
            temperature_floor=self.temperature_floor,
        )
        density = evaluated.center.density
        temperature = evaluated.center.temperature
        lij = self._lij_center(state.Er, temperature, density)
        gamma, q, upar = self._assemble_center_fluxes(
            state.Er,
            temperature,
            density,
            evaluated.density_grad_center,
            evaluated.temperature_grad_center,
            lij,
        )
        gamma, q, upar = self._regularize_center_fluxes_axis0(gamma, q, upar)
        return {
            "Gamma": gamma,
            "Q": q,
            "Upar": upar,
        }

    def evaluate_momentum_corrected_upar_only(self, state):
        """Evaluate only the regularized momentum-corrected Upar field.

        This is intentionally separate from :meth:`evaluate_momentum_corrected_fluxes`.
        The bootstrap objective consumes only Upar, whereas the general flux
        evaluator also constructs Gamma, Q, qpar, and Upar2.  In particular,
        this avoids the per-species ``get_corrected_fluxes`` work that is
        needed exclusively for Gamma/Q after the common momentum correction
        solve is available.  It is not selected by the ordinary flux path.
        """

        evaluated = build_evaluated_transport_state(
            state,
            self.geometry,
            bc_density=self.bc_density,
            bc_temperature=self.bc_temperature,
            density_floor=self.density_floor,
            temperature_floor=self.temperature_floor,
        )
        density = evaluated.center.density
        temperature = evaluated.center.temperature
        dndr = evaluated.density_grad_center
        dTdr = evaluated.temperature_grad_center
        A1 = jax.vmap(
            lambda charge, density_a, temperature_a, dndr_a, dTdr_a: get_Thermodynamical_Forces_A1(
                charge,
                density_a,
                temperature_a,
                dndr_a,
                dTdr_a,
                state.Er,
            )
        )(self.species.charge, density, temperature, dndr, dTdr)
        A2 = jax.vmap(get_Thermodynamical_Forces_A2)(temperature, dTdr)
        A3 = get_Thermodynamical_Forces_A3(state.Er)

        support = self._static_support()
        collisionality_kind = _collisionality_kind(self.collisionality_model)
        v_thermal = get_v_thermal(self.species.mass, temperature)
        n_species = int(self.species.number_species)
        species_indices = jnp.arange(n_species, dtype=jnp.int32)
        radius_indices = jnp.arange(state.Er.shape[0], dtype=jnp.int32)

        def _momentum_matrices_per_radius(radius_index):
            prepared = jax.tree_util.tree_map(
                lambda arr: jax.lax.dynamic_index_in_dim(arr, radius_index, axis=0, keepdims=False),
                support.center_prepared,
            )
            drds_value = jax.lax.dynamic_index_in_dim(
                support.center_channels.drds, radius_index, axis=0, keepdims=False
            )
            er_value = jax.lax.dynamic_index_in_dim(state.Er, radius_index, axis=0, keepdims=False)
            temperature_local = jax.lax.dynamic_index_in_dim(temperature, radius_index, axis=1, keepdims=False)
            density_local = jax.lax.dynamic_index_in_dim(density, radius_index, axis=1, keepdims=False)
            vthermal_local = jax.lax.dynamic_index_in_dim(v_thermal, radius_index, axis=1, keepdims=False)
            return jax.vmap(
                lambda species_index: self._solve_momentum_matrices_prepared_local(
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

        lij_by_radius, eij_by_radius, nu_av_by_radius = self._map_radius_axis_regularized_at_axis0(
            _momentum_matrices_per_radius,
            radius_indices,
            self.geometry.r_grid,
        )

        def _rhs_one(species_index, lij_species, radius_index):
            return jnp.stack(
                [
                    -(
                        lij_species[2, 0] * A1[species_index, radius_index]
                        + lij_species[2, 1] * A2[species_index, radius_index]
                        + lij_species[2, 2] * A3[radius_index]
                    ),
                    -(
                        (2.5 * lij_species[2, 0] - lij_species[3, 0]) * A1[species_index, radius_index]
                        + (2.5 * lij_species[2, 1] - lij_species[3, 1]) * A2[species_index, radius_index]
                        + (2.5 * lij_species[2, 2] - lij_species[3, 2]) * A3[radius_index]
                    ),
                    -(
                        (4.375 * lij_species[2, 0] - 3.5 * lij_species[3, 0] + 0.5 * lij_species[4, 0])
                        * A1[species_index, radius_index]
                        + (4.375 * lij_species[2, 1] - 3.5 * lij_species[3, 1] + 0.5 * lij_species[4, 1])
                        * A2[species_index, radius_index]
                        + (4.375 * lij_species[2, 2] - 3.5 * lij_species[3, 2] + 0.5 * lij_species[4, 2])
                        * A3[radius_index]
                    ),
                ]
            )

        def _upar_per_radius(radius_index, lij_radius, eij_radius, nu_av_radius):
            cm_ab, cn_ab, tau = jax.vmap(
                jax.vmap(
                    get_Collision_Operator_terms,
                    in_axes=(None, None, 0, None, None, None, None),
                ),
                in_axes=(None, 0, None, None, None, None, None),
            )(
                self.species,
                species_indices,
                species_indices,
                radius_index,
                temperature,
                density,
                v_thermal,
            )
            rhs = jax.vmap(_rhs_one, in_axes=(0, 0, None))(
                species_indices, lij_radius, radius_index
            )
            matrix_rows = jax.vmap(
                get_Matrix,
                in_axes=(None, None, 0, None, 0, 0, None, None, None, None),
            )(
                self.energy_grid,
                self.geometry,
                species_indices,
                radius_index,
                lij_radius,
                eij_radius,
                cm_ab,
                cn_ab,
                tau,
                v_thermal,
            )
            operator = lineax.MatrixLinearOperator(
                jnp.reshape(
                    matrix_rows,
                    (matrix_rows.shape[0] * matrix_rows.shape[1], matrix_rows.shape[2]),
                )
            )
            solution = lineax.linear_solve(
                operator, jnp.reshape(rhs, rhs.shape[0] * rhs.shape[1])
            )
            correction = jnp.reshape(solution.value, (n_species, 3))
            # This is exactly the Upar component returned by
            # ``get_corrected_fluxes`` after the common correction solve.
            return correction[:, 0] * density[:, radius_index]

        upar_by_radius = jax.vmap(
            _upar_per_radius,
            in_axes=(0, 0, 0, 0),
        )(radius_indices, lij_by_radius, eij_by_radius, nu_av_by_radius)
        upar = jnp.swapaxes(upar_by_radius, 0, 1)
        return jnp.swapaxes(
            self._regularize_axis_radius0(
                jnp.swapaxes(upar, 0, 1), self.geometry.r_grid
            ),
            0,
            1,
        )

    def evaluate_momentum_corrected_fluxes(self, state, *, diagnostics: bool = False) -> dict:
        """Evaluate realtime NTX fluxes with momentum-corrected parallel flow.

        This uses the same realtime NTX prepared support as the uncorrected
        model, but assembles the extended Sonine moment matrices needed by the
        momentum-correction solve before constructing ``Upar``.
        """

        evaluated = build_evaluated_transport_state(
            state,
            self.geometry,
            bc_density=self.bc_density,
            bc_temperature=self.bc_temperature,
            density_floor=self.density_floor,
            temperature_floor=self.temperature_floor,
        )
        density = evaluated.center.density
        temperature = evaluated.center.temperature
        dndr = evaluated.density_grad_center
        dTdr = evaluated.temperature_grad_center
        A1 = jax.vmap(
            lambda charge, density_a, temperature_a, dndr_a, dTdr_a: get_Thermodynamical_Forces_A1(
                charge,
                density_a,
                temperature_a,
                dndr_a,
                dTdr_a,
                state.Er,
            )
        )(self.species.charge, density, temperature, dndr, dTdr)
        A2 = jax.vmap(get_Thermodynamical_Forces_A2)(temperature, dTdr)
        A3 = get_Thermodynamical_Forces_A3(state.Er)

        support = self._static_support()
        collisionality_kind = _collisionality_kind(self.collisionality_model)
        v_thermal = get_v_thermal(self.species.mass, temperature)
        n_species = int(self.species.number_species)
        species_indices = jnp.arange(n_species, dtype=jnp.int32)
        radius_indices = jnp.arange(state.Er.shape[0], dtype=jnp.int32)

        def _momentum_matrices_per_radius(radius_index):
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
            er_value = jax.lax.dynamic_index_in_dim(state.Er, radius_index, axis=0, keepdims=False)
            temperature_local = jax.lax.dynamic_index_in_dim(temperature, radius_index, axis=1, keepdims=False)
            density_local = jax.lax.dynamic_index_in_dim(density, radius_index, axis=1, keepdims=False)
            vthermal_local = jax.lax.dynamic_index_in_dim(v_thermal, radius_index, axis=1, keepdims=False)
            return jax.vmap(
                lambda species_index: self._solve_momentum_matrices_prepared_local(
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

        lij_by_radius, eij_by_radius, nu_av_by_radius = self._map_radius_axis_regularized_at_axis0(
            _momentum_matrices_per_radius,
            radius_indices,
            self.geometry.r_grid,
        )

        def _rhs_one(species_index, lij_species, radius_index):
            return jnp.stack(
                [
                    -(
                        lij_species[2, 0] * A1[species_index, radius_index]
                        + lij_species[2, 1] * A2[species_index, radius_index]
                        + lij_species[2, 2] * A3[radius_index]
                    ),
                    -(
                        (2.5 * lij_species[2, 0] - lij_species[3, 0]) * A1[species_index, radius_index]
                        + (2.5 * lij_species[2, 1] - lij_species[3, 1]) * A2[species_index, radius_index]
                        + (2.5 * lij_species[2, 2] - lij_species[3, 2]) * A3[radius_index]
                    ),
                    -(
                        (4.375 * lij_species[2, 0] - 3.5 * lij_species[3, 0] + 0.5 * lij_species[4, 0])
                        * A1[species_index, radius_index]
                        + (4.375 * lij_species[2, 1] - 3.5 * lij_species[3, 1] + 0.5 * lij_species[4, 1])
                        * A2[species_index, radius_index]
                        + (4.375 * lij_species[2, 2] - 3.5 * lij_species[3, 2] + 0.5 * lij_species[4, 2])
                        * A3[radius_index]
                    ),
                ]
            )

        def _correction_per_radius(radius_index, lij_radius, eij_radius, nu_av_radius):
            cm_ab, cn_ab, tau = jax.vmap(
                jax.vmap(
                    get_Collision_Operator_terms,
                    in_axes=(None, None, 0, None, None, None, None),
                ),
                in_axes=(None, 0, None, None, None, None, None),
            )(
                self.species,
                species_indices,
                species_indices,
                radius_index,
                temperature,
                density,
                v_thermal,
            )
            rhs = jax.vmap(_rhs_one, in_axes=(0, 0, None))(species_indices, lij_radius, radius_index)
            matrix_rows = jax.vmap(
                get_Matrix,
                in_axes=(None, None, 0, None, 0, 0, None, None, None, None),
            )(
                self.energy_grid,
                self.geometry,
                species_indices,
                radius_index,
                lij_radius,
                eij_radius,
                cm_ab,
                cn_ab,
                tau,
                v_thermal,
            )
            matrix = jnp.reshape(
                matrix_rows,
                (matrix_rows.shape[0] * matrix_rows.shape[1], matrix_rows.shape[2]),
            )
            rhs_flat = jnp.reshape(rhs, rhs.shape[0] * rhs.shape[1])
            operator = lineax.MatrixLinearOperator(matrix)
            solution = lineax.linear_solve(operator, rhs_flat)
            correction = jnp.reshape(solution.value, (n_species, 3))
            corrected = jax.vmap(
                get_corrected_fluxes,
                in_axes=(
                    None,
                    None,
                    0,
                    None,
                    0,
                    0,
                    0,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
            )(
                self.energy_grid,
                self.geometry,
                species_indices,
                radius_index,
                lij_radius,
                eij_radius,
                nu_av_radius,
                cm_ab,
                cn_ab,
                tau,
                correction,
                v_thermal,
                density,
                temperature,
                A1,
                A2,
                A3,
                self.species.mass,
                self.species.charge,
                dndr,
                dTdr,
            )
            if diagnostics:
                return (*corrected, matrix, rhs_flat, correction, matrix @ solution.value - rhs_flat)
            return corrected

        corrected_by_radius = jax.vmap(
            _correction_per_radius, in_axes=(0, 0, 0, 0)
        )(radius_indices, lij_by_radius, eij_by_radius, nu_av_by_radius)
        if diagnostics:
            (
                gamma_by_radius, q_by_radius, upar_by_radius, qpar_by_radius, upar2_by_radius,
                correction_matrix_by_radius, correction_rhs_by_radius,
                correction_solution_by_radius, correction_residual_by_radius,
            ) = corrected_by_radius
        else:
            gamma_by_radius, q_by_radius, upar_by_radius, qpar_by_radius, upar2_by_radius = corrected_by_radius
        gamma = jnp.swapaxes(gamma_by_radius, 0, 1)
        q = jnp.swapaxes(q_by_radius, 0, 1)
        upar = jnp.swapaxes(upar_by_radius, 0, 1)
        qpar = jnp.swapaxes(qpar_by_radius, 0, 1)
        upar2 = jnp.swapaxes(upar2_by_radius, 0, 1)
        gamma, q, upar = self._regularize_center_fluxes_axis0(gamma, q, upar)
        qpar = self._regularize_axis_radius0(jnp.swapaxes(qpar, 0, 1), self.geometry.r_grid)
        upar2 = self._regularize_axis_radius0(jnp.swapaxes(upar2, 0, 1), self.geometry.r_grid)
        result = {
            "Gamma": gamma,
            "Q": q,
            "Upar": upar,
            "Gamma_neo": gamma,
            "Q_neo": q,
            "Upar_neo": upar,
            "qpar_neo": jnp.swapaxes(qpar, 0, 1),
            "Upar2_neo": jnp.swapaxes(upar2, 0, 1),
        }
        if diagnostics:
            result["momentum_correction_diagnostics"] = {
                "matrix_by_radius": correction_matrix_by_radius,
                "rhs_by_radius": correction_rhs_by_radius,
                "solution_by_radius": correction_solution_by_radius,
                "residual_by_radius": correction_residual_by_radius,
                "density": density,
                "temperature": temperature,
            }
        return result

    def _momentum_corrected_upar_one_radius(self, state, radius_index, *, support=None):
        """Local corrected-Upar evaluator used by compact bootstrap pullbacks."""

        evaluated = build_evaluated_transport_state(
            state,
            self.geometry,
            bc_density=self.bc_density,
            bc_temperature=self.bc_temperature,
            density_floor=self.density_floor,
            temperature_floor=self.temperature_floor,
        )
        density = evaluated.center.density
        temperature = evaluated.center.temperature
        dndr = evaluated.density_grad_center
        dTdr = evaluated.temperature_grad_center
        A1 = jax.vmap(
            lambda charge, density_a, temperature_a, dndr_a, dTdr_a: get_Thermodynamical_Forces_A1(
                charge,
                density_a,
                temperature_a,
                dndr_a,
                dTdr_a,
                state.Er,
            )
        )(self.species.charge, density, temperature, dndr, dTdr)
        A2 = jax.vmap(get_Thermodynamical_Forces_A2)(temperature, dTdr)
        A3 = get_Thermodynamical_Forces_A3(state.Er)

        support_value = self._static_support() if support is None else support
        collisionality_kind = _collisionality_kind(self.collisionality_model)
        v_thermal = get_v_thermal(self.species.mass, temperature)
        n_species = int(self.species.number_species)
        species_indices = jnp.arange(n_species, dtype=jnp.int32)

        prepared = jax.tree_util.tree_map(
            lambda arr: jax.lax.dynamic_index_in_dim(arr, radius_index, axis=0, keepdims=False),
            support_value.center_prepared,
        )
        drds_value = jax.lax.dynamic_index_in_dim(
            support_value.center_channels.drds,
            radius_index,
            axis=0,
            keepdims=False,
        )
        er_value = jax.lax.dynamic_index_in_dim(state.Er, radius_index, axis=0, keepdims=False)
        temperature_local = jax.lax.dynamic_index_in_dim(temperature, radius_index, axis=1, keepdims=False)
        density_local = jax.lax.dynamic_index_in_dim(density, radius_index, axis=1, keepdims=False)
        vthermal_local = jax.lax.dynamic_index_in_dim(v_thermal, radius_index, axis=1, keepdims=False)
        lij_radius, eij_radius, nu_av_radius = jax.vmap(
            lambda species_index: self._solve_momentum_matrices_prepared_local(
                prepared,
                drds_value=drds_value,
                species_index=species_index,
                er_value=er_value,
                temperature_local=temperature_local,
                density_local=density_local,
                vthermal_local=vthermal_local,
                collisionality_kind=collisionality_kind,
                derivative_mode_override="direct",
            )
        )(species_indices)

        def _rhs_one(species_index, lij_species):
            return jnp.stack(
                [
                    -(
                        lij_species[2, 0] * A1[species_index, radius_index]
                        + lij_species[2, 1] * A2[species_index, radius_index]
                        + lij_species[2, 2] * A3[radius_index]
                    ),
                    -(
                        (2.5 * lij_species[2, 0] - lij_species[3, 0]) * A1[species_index, radius_index]
                        + (2.5 * lij_species[2, 1] - lij_species[3, 1]) * A2[species_index, radius_index]
                        + (2.5 * lij_species[2, 2] - lij_species[3, 2]) * A3[radius_index]
                    ),
                    -(
                        (4.375 * lij_species[2, 0] - 3.5 * lij_species[3, 0] + 0.5 * lij_species[4, 0])
                        * A1[species_index, radius_index]
                        + (4.375 * lij_species[2, 1] - 3.5 * lij_species[3, 1] + 0.5 * lij_species[4, 1])
                        * A2[species_index, radius_index]
                        + (4.375 * lij_species[2, 2] - 3.5 * lij_species[3, 2] + 0.5 * lij_species[4, 2])
                        * A3[radius_index]
                    ),
                ]
            )

        cm_ab, cn_ab, tau = jax.vmap(
            jax.vmap(
                get_Collision_Operator_terms,
                in_axes=(None, None, 0, None, None, None, None),
            ),
            in_axes=(None, 0, None, None, None, None, None),
        )(
            self.species,
            species_indices,
            species_indices,
            radius_index,
            temperature,
            density,
            v_thermal,
        )
        rhs = jax.vmap(_rhs_one, in_axes=(0, 0))(species_indices, lij_radius)
        matrix_rows = jax.vmap(
            get_Matrix,
            in_axes=(None, None, 0, None, 0, 0, None, None, None, None),
        )(
            self.energy_grid,
            self.geometry,
            species_indices,
            radius_index,
            lij_radius,
            eij_radius,
            cm_ab,
            cn_ab,
            tau,
            v_thermal,
        )
        operator = lineax.MatrixLinearOperator(
            jnp.reshape(matrix_rows, (matrix_rows.shape[0] * matrix_rows.shape[1], matrix_rows.shape[2]))
        )
        solution = lineax.linear_solve(operator, jnp.reshape(rhs, rhs.shape[0] * rhs.shape[1]))
        correction = jnp.reshape(solution.value, (n_species, 3))
        _gamma, _q, upar, _qpar, _upar2 = jax.vmap(
            get_corrected_fluxes,
            in_axes=(
                None,
                None,
                0,
                None,
                0,
                0,
                0,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ),
        )(
            self.energy_grid,
            self.geometry,
            species_indices,
            radius_index,
            lij_radius,
            eij_radius,
            nu_av_radius,
            cm_ab,
            cn_ab,
            tau,
            correction,
            v_thermal,
            density,
            temperature,
            A1,
            A2,
            A3,
            self.species.mass,
            self.species.charge,
            dndr,
            dTdr,
        )
        return upar

    def pullback_momentum_corrected_upar_state_by_radius(self, state, upar_bar):
        """Compact state pullback for sparse corrected-Upar cotangents."""

        upar_bar = jnp.asarray(upar_bar, dtype=state.pressure.dtype)
        radius_count = int(upar_bar.shape[-1])
        radius_indices = jnp.arange(radius_count, dtype=jnp.int32)

        def _zero_like_leaf(leaf):
            arr = jnp.asarray(leaf)
            if jnp.issubdtype(arr.dtype, jnp.inexact):
                return jnp.zeros_like(arr)
            return jnp.zeros(arr.shape, dtype=jnp.float64)

        def _add_trees(left, right):
            return jax.tree_util.tree_map(lambda a, b: a + b, left, right)

        state_bar0 = jax.tree_util.tree_map(_zero_like_leaf, state)

        def _accumulate(carry, radius_index):
            _, pullback = jax.vjp(
                lambda state_value: self._momentum_corrected_upar_one_radius(
                    state_value,
                    radius_index,
                ),
                state,
            )
            local_bar = jax.lax.dynamic_index_in_dim(
                upar_bar,
                radius_index,
                axis=1,
                keepdims=False,
            )
            (state_bar,) = pullback(local_bar)
            return _add_trees(carry, state_bar), None

        state_bar, _ = jax.lax.scan(_accumulate, state_bar0, radius_indices)
        return state_bar

    def pullback_momentum_corrected_upar_support_by_radius(self, state, upar_bar, support):
        """Compact support-payload pullback for sparse corrected-Upar cotangents."""

        upar_bar = jnp.asarray(upar_bar, dtype=state.pressure.dtype)
        radius_count = int(upar_bar.shape[-1])
        radius_indices = jnp.arange(radius_count, dtype=jnp.int32)

        def _batched_zero_tree_leaves(tree):
            return tuple(
                jnp.zeros_like(jnp.asarray(leaf, dtype=jnp.float64))
                if not jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.inexact)
                else jnp.zeros_like(jnp.asarray(leaf))
                for leaf in jax.tree_util.tree_leaves(tree)
            )

        center_channels_bar = _float_delta_tree_like(support.center_channels)
        center_prepared_bar_leaves = _batched_zero_tree_leaves(support.center_prepared)
        face_channels_bar_leaves = _batched_zero_tree_leaves(support.face_channels)
        face_prepared_bar_leaves = _batched_zero_tree_leaves(support.face_prepared)

        def _split_flat_vector(flat, sizes, shapes, treedef):
            leaves = []
            offset = 0
            for size, shape in zip(sizes, shapes, strict=True):
                leaves.append(jnp.reshape(flat[offset : offset + size], shape))
                offset += size
            return treedef.unflatten(leaves), flat[offset]

        def _accumulate(carry, radius_index):
            channels_carry, prepared_leaf_carry = carry
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
            prepared_delta0 = _float_delta_tree_like(prepared)
            prepared_delta_leaves0, prepared_delta_treedef = jax.tree_util.tree_flatten(prepared_delta0)
            prepared_delta_shapes = tuple(jnp.asarray(leaf).shape for leaf in prepared_delta_leaves0)
            prepared_delta_sizes = tuple(int(jnp.asarray(leaf).size) for leaf in prepared_delta_leaves0)
            flat_delta0 = jnp.concatenate(
                [jnp.ravel(jnp.asarray(leaf)) for leaf in prepared_delta_leaves0]
                + [jnp.ravel(jnp.zeros_like(drds_value))]
            )

            def _upar_from_local_support_flat(flat_delta):
                prepared_delta, drds_delta = _split_flat_vector(
                    flat_delta,
                    prepared_delta_sizes,
                    prepared_delta_shapes,
                    prepared_delta_treedef,
                )

                def _add_local_prepared_delta(full, local_delta):
                    full_arr = jnp.asarray(full)
                    if not jnp.issubdtype(full_arr.dtype, jnp.inexact):
                        return full
                    return full_arr.at[radius_index].add(jnp.asarray(local_delta, dtype=full_arr.dtype))

                support_value = dataclasses.replace(
                    support,
                    center_prepared=jax.tree_util.tree_map(
                        _add_local_prepared_delta,
                        support.center_prepared,
                        prepared_delta,
                    ),
                    center_channels=dataclasses.replace(
                        support.center_channels,
                        drds=support.center_channels.drds.at[radius_index].add(drds_delta),
                    ),
                )
                return self.with_support_payload(support_value)._momentum_corrected_upar_one_radius(
                    state,
                    radius_index,
                    support=support_value,
                )

            _, pullback = jax.vjp(_upar_from_local_support_flat, flat_delta0)
            local_bar = jax.lax.dynamic_index_in_dim(
                upar_bar,
                radius_index,
                axis=1,
                keepdims=False,
            )
            (flat_bar,) = pullback(local_bar)
            prepared_flat_size = int(sum(prepared_delta_sizes))
            drds_bar = flat_bar[prepared_flat_size]

            updated_prepared_leaves = []
            offset = 0
            for carry_leaf, size, shape in zip(
                prepared_leaf_carry,
                prepared_delta_sizes,
                prepared_delta_shapes,
                strict=True,
            ):
                local_prepared_bar = jnp.reshape(flat_bar[offset : offset + size], shape)
                updated_prepared_leaves.append(carry_leaf.at[radius_index].add(local_prepared_bar))
                offset += size

            return (
                dataclasses.replace(
                    channels_carry,
                    drds=channels_carry.drds.at[radius_index].add(drds_bar),
                ),
                tuple(updated_prepared_leaves),
            ), None

        (center_channels_bar, center_prepared_bar_leaves), _ = jax.lax.scan(
            _accumulate,
            (center_channels_bar, center_prepared_bar_leaves),
            radius_indices,
        )
        return (
            tuple(jax.tree_util.tree_leaves(center_channels_bar))
            + face_channels_bar_leaves
            + tuple(center_prepared_bar_leaves)
            + face_prepared_bar_leaves
        )

    def pullback_momentum_corrected_upar_geometry_by_radius(self, state, upar_bar, geometry, support):
        """Compact NEOPAX-geometry pullback for sparse corrected-Upar cotangents."""

        upar_bar = jnp.asarray(upar_bar, dtype=state.pressure.dtype)
        radius_count = int(upar_bar.shape[-1])
        radius_indices = jnp.arange(radius_count, dtype=jnp.int32)
        geometry_delta0 = _float_delta_tree_like(geometry)
        geometry_delta_leaves0, geometry_delta_treedef = jax.tree_util.tree_flatten(geometry_delta0)
        geometry_delta_shapes = tuple(jnp.asarray(leaf).shape for leaf in geometry_delta_leaves0)
        geometry_delta_sizes = tuple(int(jnp.asarray(leaf).size) for leaf in geometry_delta_leaves0)
        flat_delta0 = jnp.concatenate([jnp.ravel(jnp.asarray(leaf)) for leaf in geometry_delta_leaves0])

        def _split_flat_geometry(flat):
            leaves = []
            offset = 0
            for size, shape in zip(geometry_delta_sizes, geometry_delta_shapes, strict=True):
                leaves.append(jnp.reshape(flat[offset : offset + size], shape))
                offset += size
            return geometry_delta_treedef.unflatten(leaves)

        def _zero_flat_like():
            return jnp.zeros_like(flat_delta0)

        def _accumulate(flat_carry, radius_index):
            def _upar_from_geometry_flat(flat_delta):
                geometry_delta = _split_flat_geometry(flat_delta)
                geometry_value = _add_float_delta_tree(geometry, geometry_delta)
                model = dataclasses.replace(self, geometry=geometry_value, support=support)
                return model._momentum_corrected_upar_one_radius(
                    state,
                    radius_index,
                    support=support,
                )

            _, pullback = jax.vjp(_upar_from_geometry_flat, flat_delta0)
            local_bar = jax.lax.dynamic_index_in_dim(
                upar_bar,
                radius_index,
                axis=1,
                keepdims=False,
            )
            (flat_bar,) = pullback(local_bar)
            return flat_carry + flat_bar, None

        geometry_flat_bar, _ = jax.lax.scan(_accumulate, _zero_flat_like(), radius_indices)
        return _split_flat_geometry(geometry_flat_bar)

    def pullback_momentum_corrected_upar_state_support_geometry_by_radius(
        self,
        state,
        upar_bar,
        geometry,
        support,
    ):
        """Joint compact Upar pullback with one local VJP per radius.

        This is deliberately an opt-in companion to the three established
        state/support/geometry helpers above.  The support contract is the
        same sparse center-prepared-plus-``drds`` contract used by the
        existing support helper; all other support leaves remain zero.
        """

        upar_bar = jnp.asarray(upar_bar, dtype=state.pressure.dtype)
        radius_count = int(upar_bar.shape[-1])
        radius_indices = jnp.arange(radius_count, dtype=jnp.int32)

        def _zero_like_leaf(leaf):
            arr = jnp.asarray(leaf)
            if jnp.issubdtype(arr.dtype, jnp.inexact):
                return jnp.zeros_like(arr)
            return jnp.zeros(arr.shape, dtype=jnp.float64)

        def _add_trees(left, right):
            return jax.tree_util.tree_map(lambda a, b: a + b, left, right)

        def _zero_tree_leaves(tree):
            return tuple(_zero_like_leaf(leaf) for leaf in jax.tree_util.tree_leaves(tree))

        center_channels_bar0 = _float_delta_tree_like(support.center_channels)
        center_prepared_bar_leaves0 = _zero_tree_leaves(support.center_prepared)
        face_channels_bar_leaves = _zero_tree_leaves(support.face_channels)
        face_prepared_bar_leaves = _zero_tree_leaves(support.face_prepared)
        geometry_delta0 = _float_delta_tree_like(geometry)
        geometry_delta_leaves0, geometry_delta_treedef = jax.tree_util.tree_flatten(
            geometry_delta0
        )
        geometry_delta_shapes = tuple(jnp.asarray(leaf).shape for leaf in geometry_delta_leaves0)
        geometry_delta_sizes = tuple(int(jnp.asarray(leaf).size) for leaf in geometry_delta_leaves0)
        geometry_flat_delta0 = jnp.concatenate(
            [jnp.ravel(jnp.asarray(leaf)) for leaf in geometry_delta_leaves0]
        )

        def _split_flat_vector(flat, sizes, shapes, treedef):
            leaves = []
            offset = 0
            for size, shape in zip(sizes, shapes, strict=True):
                leaves.append(jnp.reshape(flat[offset : offset + size], shape))
                offset += size
            return treedef.unflatten(leaves), flat[offset]

        def _split_flat_geometry(flat):
            leaves = []
            offset = 0
            for size, shape in zip(geometry_delta_sizes, geometry_delta_shapes, strict=True):
                leaves.append(jnp.reshape(flat[offset : offset + size], shape))
                offset += size
            return geometry_delta_treedef.unflatten(leaves)

        def _accumulate(carry, radius_index):
            state_carry, channels_carry, prepared_leaf_carry, geometry_flat_carry = carry
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
            prepared_delta0 = _float_delta_tree_like(prepared)
            prepared_delta_leaves0, prepared_delta_treedef = jax.tree_util.tree_flatten(
                prepared_delta0
            )
            prepared_delta_shapes = tuple(
                jnp.asarray(leaf).shape for leaf in prepared_delta_leaves0
            )
            prepared_delta_sizes = tuple(
                int(jnp.asarray(leaf).size) for leaf in prepared_delta_leaves0
            )
            support_flat_delta0 = jnp.concatenate(
                [jnp.ravel(jnp.asarray(leaf)) for leaf in prepared_delta_leaves0]
                + [jnp.ravel(jnp.zeros_like(drds_value))]
            )

            def _upar_from_local_inputs(
                state_value,
                support_flat_delta,
                geometry_flat_delta,
            ):
                prepared_delta, drds_delta = _split_flat_vector(
                    support_flat_delta,
                    prepared_delta_sizes,
                    prepared_delta_shapes,
                    prepared_delta_treedef,
                )

                def _add_local_prepared_delta(full, local_delta):
                    full_arr = jnp.asarray(full)
                    if not jnp.issubdtype(full_arr.dtype, jnp.inexact):
                        return full
                    return full_arr.at[radius_index].add(
                        jnp.asarray(local_delta, dtype=full_arr.dtype)
                    )

                support_value = dataclasses.replace(
                    support,
                    center_prepared=jax.tree_util.tree_map(
                        _add_local_prepared_delta,
                        support.center_prepared,
                        prepared_delta,
                    ),
                    center_channels=dataclasses.replace(
                        support.center_channels,
                        drds=support.center_channels.drds.at[radius_index].add(drds_delta),
                    ),
                )
                geometry_value = _add_float_delta_tree(
                    geometry,
                    _split_flat_geometry(geometry_flat_delta),
                )
                model = dataclasses.replace(self, geometry=geometry_value, support=support_value)
                return model._momentum_corrected_upar_one_radius(
                    state_value,
                    radius_index,
                    support=support_value,
                )

            _, pullback = jax.vjp(
                _upar_from_local_inputs,
                state,
                support_flat_delta0,
                geometry_flat_delta0,
            )
            local_bar = jax.lax.dynamic_index_in_dim(
                upar_bar,
                radius_index,
                axis=1,
                keepdims=False,
            )
            state_bar, support_flat_bar, geometry_flat_bar = pullback(local_bar)
            prepared_flat_size = int(sum(prepared_delta_sizes))
            drds_bar = support_flat_bar[prepared_flat_size]

            updated_prepared_leaves = []
            offset = 0
            for carry_leaf, size, shape in zip(
                prepared_leaf_carry,
                prepared_delta_sizes,
                prepared_delta_shapes,
                strict=True,
            ):
                local_prepared_bar = jnp.reshape(
                    support_flat_bar[offset : offset + size], shape
                )
                updated_prepared_leaves.append(
                    carry_leaf.at[radius_index].add(local_prepared_bar)
                )
                offset += size

            return (
                _add_trees(state_carry, state_bar),
                dataclasses.replace(
                    channels_carry,
                    drds=channels_carry.drds.at[radius_index].add(drds_bar),
                ),
                tuple(updated_prepared_leaves),
                geometry_flat_carry + geometry_flat_bar,
            ), None

        (
            state_bar,
            center_channels_bar,
            center_prepared_bar_leaves,
            geometry_flat_bar,
        ), _ = jax.lax.scan(
            _accumulate,
            (
                jax.tree_util.tree_map(_zero_like_leaf, state),
                center_channels_bar0,
                center_prepared_bar_leaves0,
                jnp.zeros_like(geometry_flat_delta0),
            ),
            radius_indices,
        )
        support_bar_leaves = (
            tuple(jax.tree_util.tree_leaves(center_channels_bar))
            + face_channels_bar_leaves
            + tuple(center_prepared_bar_leaves)
            + face_prepared_bar_leaves
        )
        return state_bar, support_bar_leaves, _split_flat_geometry(geometry_flat_bar)

    def _build_axis_lagged_response(
        self,
        *,
        channels,
        prepared_all,
        radius_coordinates,
        Er,
        temperature,
        density,
        v_thermal,
    ):
        collisionality_kind = _collisionality_kind(self.collisionality_model)
        species_indices = jnp.arange(int(self.species.number_species), dtype=jnp.int32)
        n_radius = int(Er.shape[0])
        radius_indices = jnp.arange(n_radius, dtype=jnp.int32)
        anchor_indices = self._response_anchor_indices(n_radius)

        if int(anchor_indices.shape[0]) < n_radius:
            target_rho = jnp.asarray(channels.rho, dtype=jnp.float64)

            if int(self.lagged_response_taylor_order) == 2:
                def _per_quadratic_anchor(radius_index):
                    prepared = jax.tree_util.tree_map(
                        lambda arr: jax.lax.dynamic_index_in_dim(arr, radius_index, axis=0, keepdims=False),
                        prepared_all,
                    )
                    drds_value = jax.lax.dynamic_index_in_dim(channels.drds, radius_index, axis=0, keepdims=False)
                    er_value = jax.lax.dynamic_index_in_dim(Er, radius_index, axis=0, keepdims=False)
                    temperature_local = jax.lax.dynamic_index_in_dim(temperature, radius_index, axis=1, keepdims=False)
                    density_local = jax.lax.dynamic_index_in_dim(density, radius_index, axis=1, keepdims=False)
                    vthermal_local = jax.lax.dynamic_index_in_dim(v_thermal, radius_index, axis=1, keepdims=False)
                    return jax.vmap(
                        lambda species_index: self._build_quadratic_coefficient_response_local(
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
                    _per_quadratic_anchor,
                    anchor_indices,
                    jnp.asarray(radius_coordinates, dtype=jnp.float64)[anchor_indices],
                )
                return NTXQuadraticPreparedCoefficientResponse(
                    reference_nu_hat=self._interpolate_anchor_values(
                        anchor_indices, anchor_response.reference_nu_hat, target_rho
                    ),
                    reference_epsi_hat=self._interpolate_anchor_values(
                        anchor_indices, anchor_response.reference_epsi_hat, target_rho
                    ),
                    reference_coefficients=self._interpolate_anchor_values(
                        anchor_indices, anchor_response.reference_coefficients, target_rho
                    ),
                    dcoefficients_d_nu_hat=self._interpolate_anchor_values(
                        anchor_indices, anchor_response.dcoefficients_d_nu_hat, target_rho
                    ),
                    dcoefficients_d_epsi_hat=self._interpolate_anchor_values(
                        anchor_indices, anchor_response.dcoefficients_d_epsi_hat, target_rho
                    ),
                    d2coefficients_d_nu_hat2=self._interpolate_anchor_values(
                        anchor_indices, anchor_response.d2coefficients_d_nu_hat2, target_rho
                    ),
                    d2coefficients_d_nu_hat_d_epsi_hat=self._interpolate_anchor_values(
                        anchor_indices, anchor_response.d2coefficients_d_nu_hat_d_epsi_hat, target_rho
                    ),
                    d2coefficients_d_epsi_hat2=self._interpolate_anchor_values(
                        anchor_indices, anchor_response.d2coefficients_d_epsi_hat2, target_rho
                    ),
                )

            def _per_anchor(radius_index):
                prepared = jax.tree_util.tree_map(
                    lambda arr: jax.lax.dynamic_index_in_dim(arr, radius_index, axis=0, keepdims=False),
                    prepared_all,
                )
                drds_value = jax.lax.dynamic_index_in_dim(channels.drds, radius_index, axis=0, keepdims=False)
                er_value = jax.lax.dynamic_index_in_dim(Er, radius_index, axis=0, keepdims=False)
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
                jnp.asarray(radius_coordinates, dtype=jnp.float64)[anchor_indices],
            )
            reference_log_nu_star = self._interpolate_anchor_values(anchor_indices, anchor_response[0], target_rho)
            reference_transport_moments = self._interpolate_anchor_values(anchor_indices, anchor_response[1], target_rho)
            dtransport_moments_d_er = self._interpolate_anchor_values(anchor_indices, anchor_response[2], target_rho)
            dtransport_moments_d_log_nu_star = self._interpolate_anchor_values(anchor_indices, anchor_response[3], target_rho)
            return NTXInterpolatedMomentResponse(
                reference_er=Er,
                reference_log_nu_star=jnp.swapaxes(reference_log_nu_star, 0, 1),
                reference_transport_moments=jnp.swapaxes(reference_transport_moments, 0, 1),
                dtransport_moments_d_er=jnp.swapaxes(dtransport_moments_d_er, 0, 1),
                dtransport_moments_d_log_nu_star=jnp.swapaxes(dtransport_moments_d_log_nu_star, 0, 1),
            )

        def _per_radius(radius_index):
            prepared = jax.tree_util.tree_map(
                lambda arr: jax.lax.dynamic_index_in_dim(arr, radius_index, axis=0, keepdims=False),
                prepared_all,
            )
            drds_value = jax.lax.dynamic_index_in_dim(channels.drds, radius_index, axis=0, keepdims=False)
            er_value = jax.lax.dynamic_index_in_dim(Er, radius_index, axis=0, keepdims=False)
            temperature_local = jax.lax.dynamic_index_in_dim(temperature, radius_index, axis=1, keepdims=False)
            density_local = jax.lax.dynamic_index_in_dim(density, radius_index, axis=1, keepdims=False)
            vthermal_local = jax.lax.dynamic_index_in_dim(v_thermal, radius_index, axis=1, keepdims=False)
            build_local_response = (
                self._build_quadratic_coefficient_response_local
                if int(self.lagged_response_taylor_order) == 2
                else self._build_coefficient_response_local
            )
            return jax.vmap(
                lambda species_index: build_local_response(
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

        return self._map_radius_axis_regularized_at_axis0(
            _per_radius,
            radius_indices,
            jnp.asarray(radius_coordinates, dtype=jnp.float64),
        )

    def _build_axis_lagged_response_with_coefficient_record(
        self,
        *,
        channels,
        prepared_all,
        radius_coordinates,
        Er,
        temperature,
        density,
        v_thermal,
    ):
        """Build the interpolated face response and compact local coefficients.

        This is deliberately an opt-in companion to
        :meth:`_build_axis_lagged_response`.  It is limited to the established
        interpolated-anchor lane: carrying coefficient scans for the direct
        full-radius coefficient-response lane would neither feed the current
        support transpose nor reduce an NTX solve.
        """

        collisionality_kind = _collisionality_kind(self.collisionality_model)
        species_indices = jnp.arange(int(self.species.number_species), dtype=jnp.int32)
        n_radius = int(Er.shape[0])
        anchor_indices = self._response_anchor_indices(n_radius)
        if int(anchor_indices.shape[0]) >= n_radius:
            raise NotImplementedError(
                "compact coefficient records currently require the interpolated "
                "response-anchor lane."
            )
        target_rho = jnp.asarray(channels.rho, dtype=jnp.float64)

        def _per_anchor(radius_index):
            prepared = jax.tree_util.tree_map(
                lambda arr: jax.lax.dynamic_index_in_dim(arr, radius_index, axis=0, keepdims=False),
                prepared_all,
            )
            drds_value = jax.lax.dynamic_index_in_dim(channels.drds, radius_index, axis=0, keepdims=False)
            er_value = jax.lax.dynamic_index_in_dim(Er, radius_index, axis=0, keepdims=False)
            temperature_local = jax.lax.dynamic_index_in_dim(temperature, radius_index, axis=1, keepdims=False)
            density_local = jax.lax.dynamic_index_in_dim(density, radius_index, axis=1, keepdims=False)
            vthermal_local = jax.lax.dynamic_index_in_dim(v_thermal, radius_index, axis=1, keepdims=False)
            return jax.vmap(
                lambda species_index: self._build_interpolated_moment_response_local_with_coefficient_record(
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

        anchor_rho = jnp.asarray(radius_coordinates, dtype=jnp.float64)[anchor_indices]
        anchor_response, anchor_coefficients = self._map_radius_axis_regularized_at_axis0(
            _per_anchor,
            anchor_indices,
            anchor_rho,
        )
        # At rho=0 the ordinary builder skips the local NTX solve and
        # regularizes only the four response fields from anchors 1--3.  Those
        # extrapolated coefficient leaves are not a physical local primal and
        # must never enter a later support adjoint.  The corresponding
        # interpolation transpose gives the raw axis response a zero bar, so a
        # zero placeholder makes that contract explicit without an extra NTX
        # solve or a record-only axis evaluation.
        if int(anchor_indices.shape[0]) >= 4:
            anchor_coefficients = jax.lax.cond(
                jnp.isclose(anchor_rho[0], 0.0),
                lambda _: jax.tree_util.tree_map(
                    lambda value: value.at[0].set(jnp.zeros_like(value[0])),
                    anchor_coefficients,
                ),
                lambda _: anchor_coefficients,
                operand=None,
            )
        response = NTXInterpolatedMomentResponse(
            reference_er=Er,
            reference_log_nu_star=jnp.swapaxes(
                self._interpolate_anchor_values(anchor_indices, anchor_response[0], target_rho),
                0,
                1,
            ),
            reference_transport_moments=jnp.swapaxes(
                self._interpolate_anchor_values(anchor_indices, anchor_response[1], target_rho),
                0,
                1,
            ),
            dtransport_moments_d_er=jnp.swapaxes(
                self._interpolate_anchor_values(anchor_indices, anchor_response[2], target_rho),
                0,
                1,
            ),
            dtransport_moments_d_log_nu_star=jnp.swapaxes(
                self._interpolate_anchor_values(anchor_indices, anchor_response[3], target_rho),
                0,
                1,
            ),
        )
        return response, anchor_coefficients

    def build_lagged_response_with_compact_coefficient_record(self, state):
        """Return the usual lagged response plus a segment-eligible face record.

        This method is not a replacement for ``build_lagged_response``.  The
        solver will select it only under a future dedicated record mode, so no
        normal rollout or established reverse benchmark gains a new carry leaf.
        """

        if int(self.lagged_response_taylor_order) == 2:
            raise NotImplementedError(
                "Compact coefficient records are a linear-response reverse-replay "
                "optimization and do not support quadratic realtime NTX responses."
            )
        if self._resolved_center_response_mode() != "interpolate_from_faces":
            raise NotImplementedError(
                "compact coefficient records currently require center_response_mode="
                "'interpolate_from_faces'."
            )
        density = safe_density(state.density, self.density_floor)
        temperature = state.temperature
        support = self._static_support()
        face_state = build_face_transport_state(
            state,
            self.geometry,
            bc_density=self.bc_density,
            bc_temperature=self.bc_temperature,
            density_floor=self.density_floor,
            temperature_floor=self.temperature_floor,
        )
        face_density = safe_density(face_state.density, self.density_floor)
        face_temperature = face_state.temperature
        face_v_thermal = get_v_thermal(self.species.mass, face_temperature)
        face_response, face_anchor_coefficients = (
            self._build_axis_lagged_response_with_coefficient_record(
                channels=support.face_channels,
                prepared_all=support.face_prepared,
                radius_coordinates=self.geometry.r_grid_half,
                Er=face_state.Er,
                temperature=face_temperature,
                density=face_density,
                v_thermal=face_v_thermal,
            )
        )
        del density, temperature
        return (
            NTXExactLijLaggedResponse(face_response=face_response, center_response=None),
            _NTXExactLijLaggedResponseCoefficientRecord(
                face_anchor_coefficients=face_anchor_coefficients,
            ),
        )

    def compact_coefficient_record_zero(self):
        """Return a shape-only zero record without an NTX solve.

        Radau uses this solely for slots that reuse an already-cached lagged
        response. It supplies the static branch shape for ``lax.cond`` while
        keeping the record out of the regular carry.
        """

        if int(self.lagged_response_taylor_order) == 2:
            raise NotImplementedError(
                "Compact coefficient records are a linear-response reverse-replay "
                "optimization and do not support quadratic realtime NTX responses."
            )
        if self._resolved_center_response_mode() != "interpolate_from_faces":
            raise NotImplementedError(
                "compact coefficient records currently require center_response_mode="
                "'interpolate_from_faces'."
            )
        support = self._static_support()
        n_radius = int(jnp.asarray(support.face_channels.rho).shape[0])
        n_anchor = int(self._response_anchor_indices(n_radius).shape[0])
        n_species = int(self.species.number_species)
        n_energy = int(jnp.asarray(self.energy_grid.xWeights).shape[0])
        dtype = jnp.asarray(support.face_channels.rho).dtype
        zero_scan = jnp.zeros((n_anchor, n_species, n_energy, 5), dtype=dtype)
        return _NTXExactLijLaggedResponseCoefficientRecord(
            face_anchor_coefficients=_NTXInterpolatedMomentCoefficientRecord(
                coefficient_scan=zero_scan,
                dcoefficient_scan_d_er=jnp.zeros_like(zero_scan),
                dcoefficient_scan_d_log_nu_star=jnp.zeros_like(zero_scan),
            )
        )

    def build_lagged_response(self, state, **kwargs):
        del kwargs
        if lagged_timing_enabled():
            jax.debug.callback(lambda: lagged_timing_start("ntx.build_lagged_response"), ordered=True)
        density = safe_density(state.density, self.density_floor)
        temperature = state.temperature
        support = self._static_support()
        v_thermal = get_v_thermal(self.species.mass, temperature)
        face_state = build_face_transport_state(
            state,
            self.geometry,
            bc_density=self.bc_density,
            bc_temperature=self.bc_temperature,
            density_floor=self.density_floor,
            temperature_floor=self.temperature_floor,
        )
        face_density = safe_density(face_state.density, self.density_floor)
        face_temperature = face_state.temperature
        face_v_thermal = get_v_thermal(self.species.mass, face_temperature)
        center_response_mode = self._resolved_center_response_mode()
        center_local_response = center_response_mode == "center_local_response"
        interpolate_face_coefficients = center_response_mode in {
            "interpolate_face_coefficients",
            "interpolate_face_coefficients_cubic",
            "interpolate_face_coefficients_physical_coordinates",
            "interpolate_face_coefficients_native_distance",
            "interpolate_face_coefficients_taylor_reliability",
        }
        coefficient_weight_mode = (
            "native_distance"
            if center_response_mode == "interpolate_face_coefficients_native_distance"
            else (
                "taylor_reliability"
                if center_response_mode == "interpolate_face_coefficients_taylor_reliability"
                else (
                "radial_cubic"
                if center_response_mode == "interpolate_face_coefficients_cubic"
                else "radial"
                )
            )
        )
        coefficient_coordinate_mode = (
            "physical_er_over_v"
            if center_response_mode == "interpolate_face_coefficients_physical_coordinates"
            else "native"
        )
        if interpolate_face_coefficients and not self.full_state_quadratic_response:
            raise NotImplementedError(
                "interpolate_face_coefficients currently requires the full-state "
                "quadratic NTX response; the linear reverse-capable lane will be "
                "added after this forward representation is validated."
            )
        # Conservative transport divergence always lives on faces.  Therefore
        # a direct-centre response is an *additional* cached axis response for
        # the local Er source, never a replacement for the face response.
        # Both payloads are built at a rebuild and both are evaluated cheaply
        # at stages; no stage may fall back to a live NTX solve.
        face_response = self._build_axis_lagged_response(
            channels=support.face_channels,
            prepared_all=support.face_prepared,
            radius_coordinates=self.geometry.r_grid_half,
            Er=face_state.Er,
            temperature=face_temperature,
            density=face_density,
            v_thermal=face_v_thermal,
        )
        if self.full_state_quadratic_response:
            if not isinstance(face_response, NTXQuadraticPreparedCoefficientResponse):
                raise AssertionError("full-state quadratic response requires quadratic coefficient payload.")
            face_response = NTXFullStateQuadraticPreparedCoefficientResponse(
                reference_state=state,
                coefficient_response=face_response,
            )
        _debug_arrays_if_any_nonfinite(
            "ntx.build_lagged_response.face_state",
            (
                ("density_center", density),
                ("temperature_center", temperature),
                ("Er_center", state.Er),
                ("density_face", face_density),
                ("temperature_face", face_temperature),
                ("Er_face", face_state.Er),
                ("v_thermal_face", face_v_thermal),
            ),
        )
        _debug_lagged_response_if_nonfinite("ntx.build_lagged_response.face_response", face_response)
        center_response = None
        if interpolate_face_coefficients:
            if not isinstance(face_response, NTXFullStateQuadraticPreparedCoefficientResponse):
                raise AssertionError(
                    "interpolate_face_coefficients requires a full-state quadratic face response."
                )
            center_reference_nu_hat, center_reference_epsi_hat = self._center_reference_scan_inputs(
                channels=support.center_channels,
                Er=state.Er,
                temperature=temperature,
                density=density,
                v_thermal=v_thermal,
                radius_coordinates=self.geometry.r_grid,
            )
            center_response = NTXFullStateQuadraticPreparedCoefficientResponse(
                reference_state=state,
                coefficient_response=self._interpolate_face_quadratic_coefficients_to_centres(
                    face_response.coefficient_response,
                    center_reference_nu_hat=center_reference_nu_hat,
                    center_reference_epsi_hat=center_reference_epsi_hat,
                    weight_mode=coefficient_weight_mode,
                    coordinate_mode=coefficient_coordinate_mode,
                    center_drds=support.center_channels.drds,
                    face_drds=support.face_channels.drds,
                ),
            )
            if self.debug_center_lij_comparison:
                # Diagnostic only.  The first operand is recovered from the
                # cached, interpolated coefficient response; the second is a
                # fresh centre NTX solve at exactly the same reference state.
                # It is deliberately outside normal runs because it adds one
                # full direct-centre NTX evaluation at every rebuild.
                interpolated_lij = self._lij_from_quadratic_response_at_reference(
                    center_response, axis="center"
                )
                direct_lij = self._lij_center(state.Er, temperature, density)
                abs_error = jnp.abs(interpolated_lij - direct_lij)
                rel_error = abs_error / jnp.maximum(jnp.abs(direct_lij), 1.0e-30)
                worst_rel = jnp.argmax(rel_error)
                worst_abs = jnp.argmax(abs_error)
                n_radius = direct_lij.shape[1]
                col = worst_rel % 3
                row = (worst_rel // 3) % 3
                radius = (worst_rel // 9) % n_radius
                species = worst_rel // (9 * n_radius)
                abs_col = worst_abs % 3
                abs_row = (worst_abs // 3) % 3
                abs_radius = (worst_abs // 9) % n_radius
                abs_species = worst_abs // (9 * n_radius)
                face_lij = self._lij_from_quadratic_response_at_reference(
                    face_response, axis="face"
                )
                # These two are diagnostic probes only. They expose whether
                # either adjacent face polynomial is locally valid at the
                # centre before the fixed radial 50/50 blend is applied.
                center_lo_response = NTXFullStateQuadraticPreparedCoefficientResponse(
                    reference_state=state,
                    coefficient_response=self._interpolate_face_quadratic_coefficients_to_centres(
                        face_response.coefficient_response,
                        center_reference_nu_hat=center_reference_nu_hat,
                        center_reference_epsi_hat=center_reference_epsi_hat,
                        weight_hi_override=jnp.zeros_like(center_reference_nu_hat[:, 0, 0]),
                        coordinate_mode=coefficient_coordinate_mode,
                        center_drds=support.center_channels.drds,
                        face_drds=support.face_channels.drds,
                    ),
                )
                center_hi_response = NTXFullStateQuadraticPreparedCoefficientResponse(
                    reference_state=state,
                    coefficient_response=self._interpolate_face_quadratic_coefficients_to_centres(
                        face_response.coefficient_response,
                        center_reference_nu_hat=center_reference_nu_hat,
                        center_reference_epsi_hat=center_reference_epsi_hat,
                        weight_hi_override=jnp.ones_like(center_reference_nu_hat[:, 0, 0]),
                        coordinate_mode=coefficient_coordinate_mode,
                        center_drds=support.center_channels.drds,
                        face_drds=support.face_channels.drds,
                    ),
                )
                lij_from_lo = self._lij_from_quadratic_response_at_reference(
                    center_lo_response, axis="center"
                )
                lij_from_hi = self._lij_from_quadratic_response_at_reference(
                    center_hi_response, axis="center"
                )
                face_rho = self.geometry.r_grid_half / self.geometry.a_b
                center_rho = self.geometry.r_grid / self.geometry.a_b
                face_hi = jnp.clip(
                    jnp.searchsorted(face_rho, center_rho[radius], side="right"),
                    1,
                    face_rho.shape[0] - 1,
                )
                face_lo = face_hi - 1
                face_weight_hi = (center_rho[radius] - face_rho[face_lo]) / (
                    face_rho[face_hi] - face_rho[face_lo]
                )
                nu_center = center_reference_nu_hat[radius, species]
                epsi_center = center_reference_epsi_hat[radius, species]
                nu_lo = face_response.coefficient_response.reference_nu_hat[face_lo, species]
                nu_hi = face_response.coefficient_response.reference_nu_hat[face_hi, species]
                epsi_lo = face_response.coefficient_response.reference_epsi_hat[face_lo, species]
                epsi_hi = face_response.coefficient_response.reference_epsi_hat[face_hi, species]
                if coefficient_coordinate_mode == "physical_er_over_v":
                    epsi_query_lo = epsi_center * (
                        support.face_channels.drds[face_lo]
                        / support.center_channels.drds[radius]
                    )
                    epsi_query_hi = epsi_center * (
                        support.face_channels.drds[face_hi]
                        / support.center_channels.drds[radius]
                    )
                else:
                    epsi_query_lo = epsi_center
                    epsi_query_hi = epsi_center
                native_nu_scale = jnp.maximum(jnp.abs(nu_hi - nu_lo), 1.0e-30)
                native_epsi_scale = jnp.maximum(jnp.abs(epsi_hi - epsi_lo), 1.0e-30)
                native_distance_lo2 = (
                    ((nu_center - nu_lo) / native_nu_scale) ** 2
                    + ((epsi_center - epsi_lo) / native_epsi_scale) ** 2
                )
                native_distance_hi2 = (
                    ((nu_center - nu_hi) / native_nu_scale) ** 2
                    + ((epsi_center - epsi_hi) / native_epsi_scale) ** 2
                )
                native_inverse_lo = 1.0 / jnp.maximum(native_distance_lo2, 1.0e-24)
                native_inverse_hi = 1.0 / jnp.maximum(native_distance_hi2, 1.0e-24)
                native_weight_hi = native_inverse_hi / (native_inverse_lo + native_inverse_hi)

                # Inspect the raw NTX coefficient vector as well.  This is
                # only four (n_x) direct solves for the already selected
                # worst cell/species, compared with the full direct-Lij
                # diagnostic above; it separates coefficient error from the
                # later energy-moment/Lij reduction.
                prepared_center = jax.tree_util.tree_map(
                    lambda arr: jax.lax.dynamic_index_in_dim(
                        arr, radius, axis=0, keepdims=False
                    ),
                    support.center_prepared,
                )
                direct_coefficient_scan = self._solve_coefficient_scan_prepared(
                    prepared_center, nu_center, epsi_center
                )

                def _translated_coefficient_scan(face_index):
                    c0 = face_response.coefficient_response.reference_coefficients[
                        face_index, species
                    ]
                    cu = face_response.coefficient_response.dcoefficients_d_nu_hat[
                        face_index, species
                    ]
                    ce = face_response.coefficient_response.dcoefficients_d_epsi_hat[
                        face_index, species
                    ]
                    cuu = face_response.coefficient_response.d2coefficients_d_nu_hat2[
                        face_index, species
                    ]
                    cue = face_response.coefficient_response.d2coefficients_d_nu_hat_d_epsi_hat[
                        face_index, species
                    ]
                    cee = face_response.coefficient_response.d2coefficients_d_epsi_hat2[
                        face_index, species
                    ]
                    du = (nu_center - face_response.coefficient_response.reference_nu_hat[
                        face_index, species
                    ])[:, None]
                    target_epsi = epsi_center
                    if coefficient_coordinate_mode == "physical_er_over_v":
                        target_epsi = target_epsi * (
                            support.face_channels.drds[face_index]
                            / support.center_channels.drds[radius]
                        )
                    de = (target_epsi - face_response.coefficient_response.reference_epsi_hat[
                        face_index, species
                    ])[:, None]
                    return c0 + cu * du + ce * de + 0.5 * cuu * du * du + cue * du * de + 0.5 * cee * de * de

                coefficient_from_lo = _translated_coefficient_scan(face_lo)
                coefficient_from_hi = _translated_coefficient_scan(face_hi)
                coefficient_abs_error = jnp.abs(
                    coefficient_from_hi - direct_coefficient_scan
                )
                coefficient_worst = jnp.argmax(coefficient_abs_error)
                coefficient_count = direct_coefficient_scan.shape[1]
                coefficient_energy = coefficient_worst // coefficient_count
                coefficient_component = coefficient_worst % coefficient_count
                jax.debug.print(
                    "[NEOPAX] centre-Lij face-coefficient diagnostic: "
                    "max_rel={max_rel:.6e} "
                    "species={species} radius={radius} row={row} col={col} "
                    "interpolated={interpolated:.6e} direct={direct:.6e}",
                    max_rel=jnp.max(rel_error),
                    species=species,
                    radius=radius,
                    row=row,
                    col=col,
                    interpolated=interpolated_lij.reshape(-1)[worst_rel],
                    direct=direct_lij.reshape(-1)[worst_rel],
                )
                jax.debug.print(
                    "[NEOPAX] centre-Lij face-coefficient bracket: "
                    "rho_center={rho_center:.6e} lo={lo} rho_lo={rho_lo:.6e} "
                    "hi={hi} rho_hi={rho_hi:.6e} geometric_weight_hi={weight_hi:.6e} "
                    "Lij_lo={lij_lo:.6e} Lij_hi={lij_hi:.6e}",
                    rho_center=center_rho[radius],
                    lo=face_lo,
                    rho_lo=face_rho[face_lo],
                    hi=face_hi,
                    rho_hi=face_rho[face_hi],
                    weight_hi=face_weight_hi,
                    lij_lo=face_lij[species, face_lo, row, col],
                    lij_hi=face_lij[species, face_hi, row, col],
                )
                jax.debug.print(
                    "[NEOPAX] centre-Lij native-distance weight for selected "
                    "species/radius: min_hi={min_hi:.6e} mean_hi={mean_hi:.6e} "
                    "max_hi={max_hi:.6e}",
                    min_hi=jnp.min(native_weight_hi),
                    mean_hi=jnp.mean(native_weight_hi),
                    max_hi=jnp.max(native_weight_hi),
                )
                jax.debug.print(
                    "[NEOPAX] centre-Lij face-coefficient Taylor displacement: "
                    "max_abs_dnu_lo={dnu_lo:.6e} max_abs_dnu_hi={dnu_hi:.6e} "
                    "max_abs_depsi_lo={depsi_lo:.6e} max_abs_depsi_hi={depsi_hi:.6e}",
                    dnu_lo=jnp.max(jnp.abs(nu_center - nu_lo)),
                    dnu_hi=jnp.max(jnp.abs(nu_center - nu_hi)),
                    depsi_lo=jnp.max(jnp.abs(epsi_query_lo - epsi_lo)),
                    depsi_hi=jnp.max(jnp.abs(epsi_query_hi - epsi_hi)),
                )
                jax.debug.print(
                    "[NEOPAX] centre-Lij translated one-sided values: "
                    "from_lo={from_lo:.6e} from_hi={from_hi:.6e} direct={direct:.6e}",
                    from_lo=lij_from_lo[species, radius, row, col],
                    from_hi=lij_from_hi[species, radius, row, col],
                    direct=direct_lij[species, radius, row, col],
                )
                jax.debug.print(
                    "[NEOPAX] centre-coefficient high-face check: "
                    "energy={energy} component={component} direct={direct:.6e} "
                    "from_lo={from_lo:.6e} from_hi={from_hi:.6e} "
                    "max_abs_error_hi={max_abs:.6e}",
                    energy=coefficient_energy,
                    component=coefficient_component,
                    direct=direct_coefficient_scan.reshape(-1)[coefficient_worst],
                    from_lo=coefficient_from_lo.reshape(-1)[coefficient_worst],
                    from_hi=coefficient_from_hi.reshape(-1)[coefficient_worst],
                    max_abs=jnp.max(coefficient_abs_error),
                )
                jax.debug.print(
                    "[NEOPAX] centre-Lij face-coefficient max-abs: "
                    "max_abs={max_abs:.6e} rel_at_max_abs={rel_at_max_abs:.6e} "
                    "species={species} radius={radius} row={row} col={col} "
                    "interpolated={interpolated:.6e} direct={direct:.6e}",
                    max_abs=jnp.max(abs_error),
                    rel_at_max_abs=rel_error.reshape(-1)[worst_abs],
                    species=abs_species,
                    radius=abs_radius,
                    row=abs_row,
                    col=abs_col,
                    interpolated=interpolated_lij.reshape(-1)[worst_abs],
                    direct=direct_lij.reshape(-1)[worst_abs],
                )
                # Rows/columns are the assembled transport Lij indices.  The
                # reduction is over species and radius only, so this shows
                # whether the error is confined to L02/L20 or affects other
                # coefficients as well.
                jax.debug.print(
                    "[NEOPAX] centre-Lij per-entry max-relative error "
                    "(max over species,radius): {matrix}",
                    matrix=jnp.max(rel_error, axis=(0, 1)),
                )
        elif center_local_response:
            center_response = self._build_axis_lagged_response(
                channels=support.center_channels,
                prepared_all=support.center_prepared,
                radius_coordinates=self.geometry.r_grid,
                Er=state.Er,
                temperature=temperature,
                density=density,
                v_thermal=v_thermal,
            )
            if self.full_state_quadratic_response:
                if not isinstance(center_response, NTXQuadraticPreparedCoefficientResponse):
                    raise AssertionError(
                        "full-state quadratic response requires quadratic coefficient payload."
                    )
                center_response = NTXFullStateQuadraticPreparedCoefficientResponse(
                    reference_state=state,
                    coefficient_response=center_response,
                )
        _debug_lagged_response_if_nonfinite(
            "ntx.build_lagged_response.center_response", center_response
        )
        if lagged_timing_enabled():
            jax.debug.callback(lambda: lagged_timing_end("ntx.build_lagged_response"), ordered=True)
        return NTXExactLijLaggedResponse(
            face_response=face_response,
            center_response=center_response,
        )

    def pullback_build_lagged_response(self, state, lagged_response_bar, **kwargs):
        if int(self.lagged_response_taylor_order) == 2:
            raise NotImplementedError(
                "Reverse AD for quadratic realtime NTX Lij responses requires "
                "third local NTX derivatives and is not implemented."
            )
        reverse_stage_cotangent_mode = str(kwargs.pop("reverse_stage_cotangent_mode", "full")).strip().lower()
        reverse_segment_profile_annotations = bool(
            kwargs.pop("reverse_segment_profile_annotations", False)
        )
        del kwargs
        face_response_bar = None if lagged_response_bar is None else lagged_response_bar.face_response
        center_response_bar = None if lagged_response_bar is None else lagged_response_bar.center_response
        state_bar_acc = jax.tree_util.tree_map(jnp.zeros_like, state)
        if face_response_bar is None and center_response_bar is None:
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

        def _build_face_response_from_primitives(density_value, pressure_value, er_value):
            face_state = build_face_transport_state(
                dataclasses.replace(
                    state,
                    density=density_value,
                    pressure=pressure_value,
                    Er=er_value,
                ),
                self.geometry,
                bc_density=self.bc_density,
                bc_temperature=self.bc_temperature,
                density_floor=self.density_floor,
                temperature_floor=self.temperature_floor,
            )
            face_density = safe_density(face_state.density, self.density_floor)
            face_temperature = face_state.temperature
            return self._build_axis_lagged_response(
                channels=self._static_support().face_channels,
                prepared_all=self._static_support().face_prepared,
                radius_coordinates=self.geometry.r_grid_half,
                Er=face_state.Er,
                temperature=face_temperature,
                density=face_density,
                v_thermal=get_v_thermal(self.species.mass, face_temperature),
            )

        if isinstance(face_response_bar, NTXInterpolatedMomentResponse):
            support = self._static_support()
            face_state0 = build_face_transport_state(
                state,
                self.geometry,
                bc_density=self.bc_density,
                bc_temperature=self.bc_temperature,
                density_floor=self.density_floor,
                temperature_floor=self.temperature_floor,
            )
            face_density0 = safe_density(face_state0.density, self.density_floor)
            face_temperature0 = face_state0.temperature
            face_er0 = face_state0.Er
            n_radius = int(face_er0.shape[0])
            anchor_indices = self._response_anchor_indices(n_radius)
            target_rho = jnp.asarray(support.face_channels.rho, dtype=jnp.float64)
            anchor_rho = jnp.asarray(self.geometry.r_grid_half, dtype=jnp.float64)[anchor_indices]
            n_anchor = int(anchor_indices.shape[0])
            collisionality_kind = _collisionality_kind(self.collisionality_model)
            response_field_bars = self._interpolated_response_field_bars(face_response_bar)
            face_er_bar = jnp.asarray(face_response_bar.reference_er)
            with _reverse_rebuild_profile_scope(
                reverse_segment_profile_annotations,
                "reverse_segment/rebuild_state/interpolation_transpose",
            ):
                raw_anchor_response_bar = self._pullback_interpolated_anchor_response_fields(
                    anchor_indices=anchor_indices,
                    anchor_rho=anchor_rho,
                    target_rho=target_rho,
                    field_bars=response_field_bars,
                )
            raw_anchor_response_fields = _interpolated_response_field_bar_tuple(raw_anchor_response_bar)

            face_density_bar = jnp.zeros_like(face_density0)
            face_temperature_bar = jnp.zeros_like(face_temperature0)

            if reverse_stage_cotangent_mode in {
                "zero_rebuild_anchor_fields",
                "zero_rebuild_interpolated_fields",
                "rebuild_anchor_fields_zero",
            }:
                pass
            else:
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

                def _pullback_one_face_anchor(anchor_pos):
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
                            jnp.isclose(
                                jax.lax.dynamic_index_in_dim(anchor_rho, 0, axis=0, keepdims=False),
                                0.0,
                            ),
                        ),
                    )

                    def _axis_anchor_zero_pullback(_):
                        density_local0 = jax.lax.dynamic_index_in_dim(
                            face_density0,
                            radius_index,
                            axis=1,
                            keepdims=False,
                        )
                        temperature_local0 = jax.lax.dynamic_index_in_dim(
                            face_temperature0,
                            radius_index,
                            axis=1,
                            keepdims=False,
                        )
                        er_local0 = jax.lax.dynamic_index_in_dim(
                            face_er0,
                            radius_index,
                            axis=0,
                            keepdims=False,
                        )
                        return (
                            radius_index,
                            jnp.zeros_like(density_local0),
                            jnp.zeros_like(temperature_local0),
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
                            face_density0,
                            radius_index,
                            axis=1,
                            keepdims=False,
                        )
                        temperature_local0 = jax.lax.dynamic_index_in_dim(
                            face_temperature0,
                            radius_index,
                            axis=1,
                            keepdims=False,
                        )
                        er_local0 = jax.lax.dynamic_index_in_dim(
                            face_er0,
                            radius_index,
                            axis=0,
                            keepdims=False,
                        )
                        prepared_local = jax.tree_util.tree_map(
                            lambda arr: jax.lax.dynamic_index_in_dim(arr, radius_index, axis=0, keepdims=False),
                            support.face_prepared,
                        )
                        drds_value_local = jax.lax.dynamic_index_in_dim(
                            support.face_channels.drds,
                            radius_index,
                            axis=0,
                            keepdims=False,
                        )
                        if zero_local_moment_pullback:
                            (
                                er_local_bar,
                                temperature_local_bar,
                                density_local_bar,
                            ) = (
                                jnp.zeros_like(er_local0),
                                jnp.zeros_like(temperature_local0),
                                jnp.zeros_like(density_local0),
                            )
                        else:
                            with _reverse_rebuild_profile_scope(
                                reverse_segment_profile_annotations,
                                "reverse_segment/rebuild_state/local_ntx_pullback",
                            ):
                                (
                                    er_local_bar,
                                    temperature_local_bar,
                                    density_local_bar,
                                ) = self._pullback_interpolated_moment_response_local_fields(
                                    prepared_local,
                                    drds_value=drds_value_local,
                                    er_value=er_local0,
                                    temperature_local=temperature_local0,
                                    density_local=density_local0,
                                    collisionality_kind=collisionality_kind,
                                    field_bars=local_field_bars,
                                    scan_species=scan_local_moment_pullback,
                                )
                        return (
                            radius_index,
                            density_local_bar,
                            temperature_local_bar,
                            er_local_bar,
                        )

                    return jax.lax.cond(
                        is_axis_anchor,
                        _axis_anchor_zero_pullback,
                        _non_axis_anchor_pullback,
                        operand=None,
                    )

                if scan_anchor_pullback:
                    def _accumulate_face_anchor(carry, anchor_pos):
                        density_carry, temperature_carry, er_carry = carry
                        (
                            radius_index,
                            density_local_bar,
                            temperature_local_bar,
                            er_local_bar,
                        ) = _pullback_one_face_anchor(anchor_pos)
                        return (
                            density_carry.at[:, radius_index].add(density_local_bar),
                            temperature_carry.at[:, radius_index].add(temperature_local_bar),
                            er_carry.at[radius_index].add(er_local_bar),
                        ), None

                    (
                        face_density_bar,
                        face_temperature_bar,
                        face_er_bar,
                    ), _ = jax.lax.scan(
                        _accumulate_face_anchor,
                        (face_density_bar, face_temperature_bar, face_er_bar),
                        anchor_positions,
                    )
                else:
                    (
                        anchor_radius_indices,
                        density_anchor_bars,
                        temperature_anchor_bars,
                        er_anchor_bars,
                    ) = jax.vmap(_pullback_one_face_anchor)(anchor_positions)
                    face_density_bar = face_density_bar.at[:, anchor_radius_indices].add(
                        jnp.swapaxes(density_anchor_bars, 0, 1)
                    )
                    face_temperature_bar = face_temperature_bar.at[:, anchor_radius_indices].add(
                        jnp.swapaxes(temperature_anchor_bars, 0, 1)
                    )
                    face_er_bar = face_er_bar.at[anchor_radius_indices].add(er_anchor_bars)

            def _face_state_values(density_value, pressure_value, er_value):
                face_state = build_face_transport_state(
                    dataclasses.replace(
                        state,
                        density=density_value,
                        pressure=pressure_value,
                        Er=er_value,
                    ),
                    self.geometry,
                    bc_density=self.bc_density,
                    bc_temperature=self.bc_temperature,
                    density_floor=self.density_floor,
                    temperature_floor=self.temperature_floor,
                )
                return (
                    safe_density(face_state.density, self.density_floor),
                    face_state.temperature,
                    face_state.Er,
                )

            _, face_state_pullback = jax.vjp(
                _face_state_values,
                state.density,
                state.pressure,
                state.Er,
            )
            density_bar, pressure_bar, er_bar = face_state_pullback(
                (face_density_bar, face_temperature_bar, face_er_bar)
            )
            state_bar_acc = dataclasses.replace(
                state_bar_acc,
                density=state_bar_acc.density + density_bar,
                pressure=state_bar_acc.pressure + pressure_bar,
                Er=state_bar_acc.Er + er_bar,
            )
            if center_response_bar is None:
                return state_bar_acc

        elif face_response_bar is not None:
            density_bar, pressure_bar, er_bar = _transpose_primitives_from_builder(
                _build_face_response_from_primitives,
                face_response_bar,
            )
            state_bar_acc = dataclasses.replace(
                state_bar_acc,
                density=state_bar_acc.density + density_bar,
                pressure=state_bar_acc.pressure + pressure_bar,
                Er=state_bar_acc.Er + er_bar,
            )
            if center_response_bar is None:
                return state_bar_acc

        def _build_coefficient_center_response_from_primitives(density_value, pressure_value, er_value):
            density_safe = safe_density(density_value, self.density_floor)
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
            density_safe = safe_density(density_value, self.density_floor)
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
                    state_bar_acc,
                    density=state_bar_acc.density + density_bar,
                    pressure=state_bar_acc.pressure + pressure_bar,
                    Er=state_bar_acc.Er + er_bar,
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
                    density_safe_local = safe_density(density_local0, self.density_floor)
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
                        self.density_floor,
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
                    state_bar_acc,
                    density=state_bar_acc.density + density_bar,
                    pressure=state_bar_acc.pressure + pressure_bar,
                    Er=state_bar_acc.Er + er_bar,
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
                state_bar_acc,
                density=state_bar_acc.density + density_bar,
                pressure=state_bar_acc.pressure + pressure_bar,
                Er=state_bar_acc.Er + er_bar,
            )

        if isinstance(center_response_bar, NTXPreparedCoefficientResponse):
            density_bar, pressure_bar, er_bar = _transpose_primitives_from_builder(
                _build_coefficient_center_response_from_primitives,
                center_response_bar,
            )
            return dataclasses.replace(
                state_bar_acc,
                density=state_bar_acc.density + density_bar,
                pressure=state_bar_acc.pressure + pressure_bar,
                Er=state_bar_acc.Er + er_bar,
            )

        _, pullback = jax.vjp(
            lambda state_value: self.build_lagged_response(state_value).center_response,
            state,
        )
        (state_bar,) = pullback(center_response_bar)
        return dataclasses.replace(
            state_bar_acc,
            density=state_bar_acc.density + state_bar.density,
            pressure=state_bar_acc.pressure + state_bar.pressure,
            Er=state_bar_acc.Er + state_bar.Er,
        )

    def pullback_build_lagged_response_support_payload(
        self,
        state,
        lagged_response_bar,
        support,
        **kwargs,
    ):
        reverse_segment_profile_annotations = bool(
            kwargs.pop("reverse_segment_profile_annotations", False)
        )
        reuse_local_vjp_primal_anchor_response = bool(
            kwargs.pop("reuse_local_vjp_primal_anchor_response", False)
        )
        support_only_ntx_implicit_pullback = bool(
            kwargs.pop("support_only_ntx_implicit_pullback", False)
        )
        factorized_ntx_two_directional_prepared_vjp = bool(
            kwargs.pop("factorized_ntx_two_directional_prepared_vjp", False)
        )
        geometry_only_prepared_pullback = bool(
            kwargs.pop("geometry_only_prepared_pullback", False)
        )
        geometry_implicit_ntx_two_directional_pullback = bool(
            kwargs.pop("geometry_implicit_ntx_two_directional_pullback", False)
        )
        reverse_rebuild_inner_timing_component = str(
            kwargs.pop("reverse_rebuild_inner_timing_component", "full")
        ).strip().lower()
        del kwargs
        if reverse_rebuild_inner_timing_component not in {
            "full",
            "local_ntx_vjp_and_accumulation",
            "local_ntx_vjp_primal",
            "local_ntx_vjp_transpose",
            "local_ntx_vjp_transport_only",
            "local_ntx_vjp_d_er_only",
            "local_ntx_vjp_d_log_nu_star_only",
            "coordinate_rho_transpose",
        }:
            raise ValueError(
                "reverse_rebuild_inner_timing_component must be one of "
                "{'full', 'local_ntx_vjp_and_accumulation', 'local_ntx_vjp_primal', "
                "'local_ntx_vjp_transpose', 'local_ntx_vjp_transport_only', "
                "'local_ntx_vjp_d_er_only', 'local_ntx_vjp_d_log_nu_star_only', "
                "'coordinate_rho_transpose'}."
            )
        face_response_bar = None if lagged_response_bar is None else lagged_response_bar.face_response
        center_response_bar = None if lagged_response_bar is None else lagged_response_bar.center_response
        if (
            reuse_local_vjp_primal_anchor_response
            and not isinstance(face_response_bar, NTXInterpolatedMomentResponse)
        ):
            raise NotImplementedError(
                "reuse_local_vjp_primal_anchor_response requires an "
                "NTXInterpolatedMomentResponse face cotangent."
            )
        if support_only_ntx_implicit_pullback and not reuse_local_vjp_primal_anchor_response:
            raise ValueError(
                "support_only_ntx_implicit_pullback requires "
                "reuse_local_vjp_primal_anchor_response=True."
            )
        if (
            factorized_ntx_two_directional_prepared_vjp
            and not reuse_local_vjp_primal_anchor_response
        ):
            raise ValueError(
                "factorized_ntx_two_directional_prepared_vjp requires "
                "reuse_local_vjp_primal_anchor_response=True."
            )
        if geometry_only_prepared_pullback:
            if not reuse_local_vjp_primal_anchor_response:
                raise ValueError(
                    "geometry_only_prepared_pullback requires "
                    "reuse_local_vjp_primal_anchor_response=True."
                )
            if support_only_ntx_implicit_pullback or factorized_ntx_two_directional_prepared_vjp:
                raise ValueError(
                    "geometry_only_prepared_pullback is a standalone generic-VJP "
                    "mode and cannot be combined with an NTX custom support rule."
                )
        if geometry_implicit_ntx_two_directional_pullback:
            if not reuse_local_vjp_primal_anchor_response:
                raise ValueError(
                    "geometry_implicit_ntx_two_directional_pullback requires "
                    "reuse_local_vjp_primal_anchor_response=True."
                )
            if (
                support_only_ntx_implicit_pullback
                or factorized_ntx_two_directional_prepared_vjp
                or geometry_only_prepared_pullback
            ):
                raise ValueError(
                    "geometry_implicit_ntx_two_directional_pullback is a standalone "
                    "NTX custom support rule."
                )
        if support_only_ntx_implicit_pullback or geometry_implicit_ntx_two_directional_pullback:
            if not isinstance(face_response_bar, NTXInterpolatedMomentResponse):
                raise NotImplementedError(
                    "geometry/support-only NTX implicit pullback requires an "
                    "interpolated NTX face response."
                )
            if center_response_bar is not None:
                raise NotImplementedError(
                    "geometry/support-only NTX implicit pullback requires center_response=None."
                )
            ntx = _import_ntx()
            if not callable(
                getattr(
                    ntx,
                    (
                        "solve_prepared_coefficient_vector_lowdot_two_pullbacks_geometry_support_only_and_aux"
                        if geometry_implicit_ntx_two_directional_pullback
                        else "solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_and_aux"
                    ),
                    None,
                )
            ):
                raise RuntimeError(
                    "The selected NTX implicit support pullback requires the current NTX helper."
                )
        if face_response_bar is not None:
            face_channels_bar = _float_delta_tree_like(support.face_channels)
            face_prepared_bar = _float_delta_tree_like(support.face_prepared)
            face_state = build_face_transport_state(
                state,
                self.geometry,
                bc_density=self.bc_density,
                bc_temperature=self.bc_temperature,
                density_floor=self.density_floor,
                temperature_floor=self.temperature_floor,
            )
            face_density = safe_density(face_state.density, self.density_floor)
            face_temperature = face_state.temperature
            face_v_thermal = get_v_thermal(self.species.mass, face_temperature)
            collisionality_kind = _collisionality_kind(self.collisionality_model)
            species_indices = jnp.arange(int(self.species.number_species), dtype=jnp.int32)

            def _local_face_support_pullback(
                radius_index,
                local_field_bars,
                *,
                interpolated: bool,
                return_primal_response: bool = False,
            ):
                prepared = jax.tree_util.tree_map(
                    lambda arr: jax.lax.dynamic_index_in_dim(arr, radius_index, axis=0, keepdims=False),
                    support.face_prepared,
                )
                drds_value = jax.lax.dynamic_index_in_dim(
                    support.face_channels.drds,
                    radius_index,
                    axis=0,
                    keepdims=False,
                )
                er_value = jax.lax.dynamic_index_in_dim(face_state.Er, radius_index, axis=0, keepdims=False)
                temperature_local = jax.lax.dynamic_index_in_dim(
                    face_temperature,
                    radius_index,
                    axis=1,
                    keepdims=False,
                )
                density_local = jax.lax.dynamic_index_in_dim(
                    face_density,
                    radius_index,
                    axis=1,
                    keepdims=False,
                )
                vthermal_local = jax.lax.dynamic_index_in_dim(
                    face_v_thermal,
                    radius_index,
                    axis=1,
                    keepdims=False,
                )
                if support_only_ntx_implicit_pullback or geometry_implicit_ntx_two_directional_pullback:
                    if not interpolated:
                        raise NotImplementedError(
                            "geometry/support-only NTX implicit pullback currently requires "
                            "the interpolated NTX face response."
                        )
                    prepared_treedef = jax.tree_util.tree_structure(prepared)

                    def _one_species_support_pullback(species_index, *species_field_bars):
                        reference_nu_hat, reference_epsi_hat, vth_a = (
                            self._interpolated_moment_local_scan_primitives(
                                drds_value=drds_value,
                                species_index=species_index,
                                er_value=er_value,
                                temperature_local=temperature_local,
                                density_local=density_local,
                                vthermal_local=vthermal_local,
                                collisionality_kind=collisionality_kind,
                            )
                        )
                        prepared_bar, drds_bar, primal_response = (
                            self._pullback_interpolated_moment_prepared_support_and_drds_only(
                            prepared,
                            drds_value=drds_value,
                            reference_nu_hat=reference_nu_hat,
                            reference_epsi_hat=reference_epsi_hat,
                            vth_a=vth_a,
                            field_bars=species_field_bars,
                            geometry_implicit_ntx_two_directional=(
                                geometry_implicit_ntx_two_directional_pullback
                            ),
                            )
                        )
                        # This function is the direct output of a species
                        # ``vmap``. Convert static/float0 prepared leaves to
                        # ordinary numeric zero bars *before* batching, as the
                        # established scalar support path does at its outer
                        # support boundary. Otherwise JAX attempts to batch
                        # NTX's static Boozer metadata dataclass leaves.
                        # Do not return the prepared dataclass itself through
                        # this species ``vmap``: its VMEC/Boozer metadata
                        # pytree reconstructors validate Python-level mode
                        # arrays and cannot receive vmap batching sentinels.
                        # The outer support tree needs exactly the same numeric
                        # leaves, so carry those leaves through the map and
                        # rebuild the dataclass only after summing species.
                        return (
                            tuple(
                                jax.tree_util.tree_leaves(
                                    _sanitize_float_delta_bar_tree(prepared, prepared_bar)
                                )
                            ),
                            drds_bar,
                            primal_response,
                        )

                    prepared_species_bar_leaves, drds_species_bars, primal_response = jax.vmap(
                        _one_species_support_pullback
                    )(species_indices, *local_field_bars)
                    prepared_bar = jax.tree_util.tree_unflatten(
                        prepared_treedef,
                        tuple(
                            jnp.sum(values, axis=0)
                            for values in prepared_species_bar_leaves
                        ),
                    )
                    drds_bar = jnp.sum(drds_species_bars, axis=0)
                    if return_primal_response:
                        return primal_response, prepared_bar, drds_bar
                    return prepared_bar, drds_bar

                prepared_delta0 = _float_delta_tree_like(prepared)
                drds_delta0 = jnp.zeros_like(drds_value)

                def _response_from_support_delta(prepared_delta, drds_delta):
                    prepared_value = _add_float_delta_tree(prepared, prepared_delta)
                    drds_local = drds_value + drds_delta
                    if interpolated:
                        return jax.vmap(
                            lambda species_index: self._build_interpolated_moment_response_local(
                                prepared_value,
                                drds_value=drds_local,
                                species_index=species_index,
                                er_value=er_value,
                                temperature_local=temperature_local,
                                density_local=density_local,
                                vthermal_local=vthermal_local,
                                collisionality_kind=collisionality_kind,
                                use_factorized_ntx_two_directional_prepared_vjp=(
                                    factorized_ntx_two_directional_prepared_vjp
                                ),
                            )
                        )(species_indices)
                    return jax.vmap(
                        lambda species_index: self._build_coefficient_response_local(
                            prepared_value,
                            drds_value=drds_local,
                            species_index=species_index,
                            er_value=er_value,
                            temperature_local=temperature_local,
                            density_local=density_local,
                            vthermal_local=vthermal_local,
                            collisionality_kind=collisionality_kind,
                        )
                    )(species_indices)

                if geometry_only_prepared_pullback:
                    # ``d_theta`` and ``d_zeta`` are generated solely from the
                    # fixed GridSpec.  The runtime payload changes only the
                    # sampled GeometryOnGrid fields, so making the full
                    # PreparedMonoenergeticSystem an AD input wastes a dense
                    # operator transpose whose upstream tangent is exactly
                    # zero. Keep those operators (and the source surface) in
                    # the closure and expose only geometry plus ``drds``.
                    geometry_delta0 = _float_delta_tree_like(prepared.geometry)

                    def _response_from_geometry_delta(geometry_delta, drds_delta):
                        prepared_value = dataclasses.replace(
                            prepared,
                            geometry=_add_float_delta_tree(
                                prepared.geometry,
                                geometry_delta,
                            ),
                        )
                        drds_local = drds_value + drds_delta
                        if interpolated:
                            return jax.vmap(
                                lambda species_index: self._build_interpolated_moment_response_local(
                                    prepared_value,
                                    drds_value=drds_local,
                                    species_index=species_index,
                                    er_value=er_value,
                                    temperature_local=temperature_local,
                                    density_local=density_local,
                                    vthermal_local=vthermal_local,
                                    collisionality_kind=collisionality_kind,
                                )
                            )(species_indices)
                        return jax.vmap(
                            lambda species_index: self._build_coefficient_response_local(
                                prepared_value,
                                drds_value=drds_local,
                                species_index=species_index,
                                er_value=er_value,
                                temperature_local=temperature_local,
                                density_local=density_local,
                                vthermal_local=vthermal_local,
                                collisionality_kind=collisionality_kind,
                            )
                        )(species_indices)

                    with _reverse_rebuild_profile_scope(
                        reverse_segment_profile_annotations,
                        "reverse_segment/rebuild_support/local_ntx_vjp_primal",
                    ):
                        primal_response, geometry_pullback = jax.vjp(
                            _response_from_geometry_delta,
                            geometry_delta0,
                            drds_delta0,
                        )

                    def pullback(local_bars):
                        geometry_bar, drds_bar = geometry_pullback(local_bars)
                        return (
                            dataclasses.replace(
                                _float_delta_tree_like(prepared),
                                geometry=geometry_bar,
                            ),
                            drds_bar,
                        )
                else:
                    with _reverse_rebuild_profile_scope(
                        reverse_segment_profile_annotations,
                        "reverse_segment/rebuild_support/local_ntx_vjp_primal",
                    ):
                        primal_response, pullback = jax.vjp(
                            _response_from_support_delta,
                            prepared_delta0,
                            drds_delta0,
                        )
                if reverse_rebuild_inner_timing_component == "local_ntx_vjp_primal":
                    # Diagnostic-only: retain the exact local VJP primal but
                    # skip its transpose and return shape-correct zero support
                    # bars so the outer anchor scan remains valid.
                    zero_prepared_bar = _float_delta_tree_like(prepared)
                    zero_drds_bar = jnp.zeros_like(drds_value)
                    if return_primal_response:
                        return primal_response, zero_prepared_bar, zero_drds_bar
                    return zero_prepared_bar, zero_drds_bar

                # Keep the exact production primal, but selectively zero the
                # response cotangents for the diagnostic-only transpose
                # partitions below.  The tuple order is fixed by
                # ``_interpolated_moment_reduced_local_outputs_from_primitives``:
                # (log_nu_star, transport, dtransport/dEr, dtransport/dlog_nu).
                # These selectors never enter the normal reverse path.
                if reverse_rebuild_inner_timing_component == "local_ntx_vjp_transport_only":
                    local_field_bars = (
                        jnp.zeros_like(local_field_bars[0]),
                        local_field_bars[1],
                        jnp.zeros_like(local_field_bars[2]),
                        jnp.zeros_like(local_field_bars[3]),
                    )
                elif reverse_rebuild_inner_timing_component == "local_ntx_vjp_d_er_only":
                    local_field_bars = (
                        jnp.zeros_like(local_field_bars[0]),
                        jnp.zeros_like(local_field_bars[1]),
                        local_field_bars[2],
                        jnp.zeros_like(local_field_bars[3]),
                    )
                elif reverse_rebuild_inner_timing_component == "local_ntx_vjp_d_log_nu_star_only":
                    local_field_bars = (
                        jnp.zeros_like(local_field_bars[0]),
                        jnp.zeros_like(local_field_bars[1]),
                        jnp.zeros_like(local_field_bars[2]),
                        local_field_bars[3],
                    )
                with _reverse_rebuild_profile_scope(
                    reverse_segment_profile_annotations,
                    "reverse_segment/rebuild_support/local_ntx_vjp_transpose",
                ):
                    prepared_bar, drds_bar = pullback(local_field_bars)
                if return_primal_response:
                    return primal_response, prepared_bar, drds_bar
                return prepared_bar, drds_bar

            def _add_local_face_prepared_bar(prepared_bar, radius_index, local_bar):
                return jax.tree_util.tree_map(
                    lambda arr, local_arr: arr.at[radius_index].add(local_arr),
                    prepared_bar,
                    local_bar,
                )

            if isinstance(face_response_bar, NTXInterpolatedMomentResponse):
                n_radius = int(face_state.Er.shape[0])
                anchor_indices = self._response_anchor_indices(n_radius)
                anchor_rho = jnp.asarray(self.geometry.r_grid_half, dtype=jnp.float64)[anchor_indices]
                target_rho = jnp.asarray(support.face_channels.rho, dtype=jnp.float64)
                n_anchor = int(anchor_indices.shape[0])
                response_field_bars = self._interpolated_response_field_bars(face_response_bar)
                response_field_bar_tuple = _interpolated_response_field_bar_tuple(response_field_bars)

                if reuse_local_vjp_primal_anchor_response:
                    if reverse_rebuild_inner_timing_component == "coordinate_rho_transpose":
                        # Diagnostic-only component boundary.  The coordinate
                        # transpose depends on the anchor-response *shape*, not
                        # on an NTX solve.  Reuse same-shaped, non-constant
                        # slices of the incoming cotangent to keep the GPU
                        # arithmetic live while measuring only the
                        # interpolation-coordinate work.  This branch is
                        # never selected by the normal reverse calculation.
                        anchor_response_fields = tuple(
                            jnp.take(field_bar, anchor_indices, axis=0)
                            for field_bar in response_field_bar_tuple
                        )
                        target_rho_bar = jnp.zeros_like(target_rho)
                        for anchor_field, field_bar in zip(
                            anchor_response_fields,
                            response_field_bar_tuple,
                            strict=True,
                        ):
                            target_rho_bar = target_rho_bar + self._pullback_interpolate_anchor_target_rho(
                                anchor_indices,
                                anchor_field,
                                target_rho,
                                field_bar,
                            )
                        face_channels_bar = dataclasses.replace(
                            face_channels_bar,
                            rho=face_channels_bar.rho + target_rho_bar,
                        )
                        return _support_bar_from_face_bars(
                            support,
                            face_channels_bar,
                            face_prepared_bar,
                        )
                    # The old path below computes every local anchor response
                    # once for the coordinate transpose, then computes the
                    # exact same primal values again as the forward half of
                    # each local JAX VJP.  Keep the VJP primal output and use
                    # it for the coordinate transpose instead.  The collected
                    # array is temporary within this pullback and has the same
                    # shape as the old `anchor_response`; no lagged cache or
                    # checkpoint payload is enlarged.
                    raw_anchor_response_bar = self._pullback_interpolated_anchor_response_fields(
                        anchor_indices=anchor_indices,
                        anchor_rho=anchor_rho,
                        target_rho=target_rho,
                        field_bars=response_field_bars,
                    )
                    raw_anchor_response_fields = _interpolated_response_field_bar_tuple(
                        raw_anchor_response_bar
                    )
                    anchor_response_fields0 = tuple(
                        jnp.zeros(
                            (n_anchor,) + field_bar.shape[1:],
                            dtype=field_bar.dtype,
                        )
                        for field_bar in response_field_bar_tuple
                    )
                    anchor_positions = jnp.arange(n_anchor, dtype=jnp.int32)

                    def _accumulate_anchor_from_vjp(carry, anchor_pos):
                        channels_carry, prepared_carry, anchor_fields_carry = carry
                        radius_index = jax.lax.dynamic_index_in_dim(
                            anchor_indices,
                            anchor_pos,
                            axis=0,
                            keepdims=False,
                        )
                        local_field_bars = tuple(
                            jax.lax.dynamic_index_in_dim(
                                field_bar,
                                anchor_pos,
                                axis=0,
                                keepdims=False,
                            )
                            for field_bar in raw_anchor_response_fields
                        )
                        local_response, prepared_local_bar, drds_local_bar = (
                            _local_face_support_pullback(
                                radius_index,
                                local_field_bars,
                                interpolated=True,
                                return_primal_response=True,
                            )
                        )
                        updated_anchor_fields = tuple(
                            anchor_field.at[anchor_pos].set(local_field)
                            for anchor_field, local_field in zip(
                                anchor_fields_carry,
                                local_response,
                                strict=True,
                            )
                        )
                        return (
                            dataclasses.replace(
                                channels_carry,
                                drds=channels_carry.drds.at[radius_index].add(drds_local_bar),
                            ),
                            _add_local_face_prepared_bar(
                                prepared_carry,
                                radius_index,
                                prepared_local_bar,
                            ),
                            updated_anchor_fields,
                        ), None

                    with _reverse_rebuild_profile_scope(
                        reverse_segment_profile_annotations,
                        "reverse_segment/rebuild_support/anchor_accumulation",
                    ):
                        (
                            face_channels_bar,
                            face_prepared_bar,
                            raw_anchor_response_fields,
                        ), _ = jax.lax.scan(
                            _accumulate_anchor_from_vjp,
                            (
                                face_channels_bar,
                                face_prepared_bar,
                                anchor_response_fields0,
                            ),
                            anchor_positions,
                        )
                    # Match `_map_radius_axis_regularized_at_axis0`: axis
                    # extrapolation is valid only when the first anchor is
                    # actually at rho=0.  Face grids can begin away from the
                    # axis, in which case the raw local responses are already
                    # the reference values.
                    if n_anchor < 4:
                        anchor_response_fields = raw_anchor_response_fields
                    else:
                        anchor_response_fields = jax.lax.cond(
                            jnp.isclose(anchor_rho[0], 0.0),
                            lambda _: self._regularize_axis_radius0(
                                raw_anchor_response_fields,
                                anchor_rho,
                            ),
                            lambda _: raw_anchor_response_fields,
                            operand=None,
                        )
                    if reverse_rebuild_inner_timing_component in {
                        "local_ntx_vjp_and_accumulation",
                        "local_ntx_vjp_transport_only",
                        "local_ntx_vjp_d_er_only",
                        "local_ntx_vjp_d_log_nu_star_only",
                    }:
                        # Diagnostic-only: retain the exact anchor-value
                        # interpolation transpose, local NTX VJP, and the
                        # support-bar scatter/accumulation, while omitting only
                        # the separate coordinate transpose below.
                        return _support_bar_from_face_bars(
                            support,
                            face_channels_bar,
                            face_prepared_bar,
                        )
                    target_rho_bar = jnp.zeros_like(target_rho)
                    for anchor_field, field_bar in zip(
                        anchor_response_fields,
                        response_field_bar_tuple,
                        strict=True,
                    ):
                        target_rho_bar = target_rho_bar + self._pullback_interpolate_anchor_target_rho(
                            anchor_indices,
                            anchor_field,
                            target_rho,
                            field_bar,
                        )
                    face_channels_bar = dataclasses.replace(
                        face_channels_bar,
                        rho=face_channels_bar.rho + target_rho_bar,
                    )
                    face_support_bar = _support_bar_from_face_bars(
                        support,
                        face_channels_bar,
                        face_prepared_bar,
                    )
                    if center_response_bar is None:
                        return face_support_bar
                    center_support_bar = self.pullback_build_lagged_response_support_payload(
                        state,
                        NTXExactLijLaggedResponse(center_response=center_response_bar),
                        support,
                    )
                    return jax.tree_util.tree_map(
                        lambda lhs, rhs: lhs + rhs,
                        face_support_bar,
                        center_support_bar,
                    )

                def _per_anchor_forward(radius_index):
                    prepared = jax.tree_util.tree_map(
                        lambda arr: jax.lax.dynamic_index_in_dim(arr, radius_index, axis=0, keepdims=False),
                        support.face_prepared,
                    )
                    drds_value = jax.lax.dynamic_index_in_dim(
                        support.face_channels.drds,
                        radius_index,
                        axis=0,
                        keepdims=False,
                    )
                    er_value = jax.lax.dynamic_index_in_dim(face_state.Er, radius_index, axis=0, keepdims=False)
                    temperature_local = jax.lax.dynamic_index_in_dim(
                        face_temperature,
                        radius_index,
                        axis=1,
                        keepdims=False,
                    )
                    density_local = jax.lax.dynamic_index_in_dim(
                        face_density,
                        radius_index,
                        axis=1,
                        keepdims=False,
                    )
                    vthermal_local = jax.lax.dynamic_index_in_dim(
                        face_v_thermal,
                        radius_index,
                        axis=1,
                        keepdims=False,
                    )
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

                with _reverse_rebuild_profile_scope(
                    reverse_segment_profile_annotations,
                    "reverse_segment/rebuild_support/anchor_response_reconstruct",
                ):
                    anchor_response = self._map_radius_axis_regularized_at_axis0(
                        _per_anchor_forward,
                        anchor_indices,
                        anchor_rho,
                    )
                anchor_response_fields = (
                    anchor_response[0],
                    anchor_response[1],
                    anchor_response[2],
                    anchor_response[3],
                )

                with _reverse_rebuild_profile_scope(
                    reverse_segment_profile_annotations,
                    "reverse_segment/rebuild_support/interpolation_transpose",
                ):
                    target_rho_bar = jnp.zeros_like(target_rho)
                    for anchor_field, field_bar in zip(
                        anchor_response_fields,
                        response_field_bar_tuple,
                        strict=True,
                    ):
                        target_rho_bar = target_rho_bar + self._pullback_interpolate_anchor_target_rho(
                            anchor_indices,
                            anchor_field,
                            target_rho,
                            field_bar,
                        )
                face_channels_bar = dataclasses.replace(
                    face_channels_bar,
                    rho=face_channels_bar.rho + target_rho_bar,
                )

                with _reverse_rebuild_profile_scope(
                    reverse_segment_profile_annotations,
                    "reverse_segment/rebuild_support/interpolation_transpose",
                ):
                    raw_anchor_response_bar = self._pullback_interpolated_anchor_response_fields(
                        anchor_indices=anchor_indices,
                        anchor_rho=anchor_rho,
                        target_rho=target_rho,
                        field_bars=response_field_bars,
                    )
                raw_anchor_response_fields = _interpolated_response_field_bar_tuple(raw_anchor_response_bar)
                anchor_positions = jnp.arange(n_anchor, dtype=jnp.int32)

                def _pullback_one_anchor(anchor_pos):
                    radius_index = jax.lax.dynamic_index_in_dim(
                        anchor_indices,
                        anchor_pos,
                        axis=0,
                        keepdims=False,
                    )
                    local_field_bars = tuple(
                        jax.lax.dynamic_index_in_dim(
                            field_bar,
                            anchor_pos,
                            axis=0,
                            keepdims=False,
                        )
                        for field_bar in raw_anchor_response_fields
                    )
                    prepared_local_bar, drds_local_bar = _local_face_support_pullback(
                        radius_index,
                        local_field_bars,
                        interpolated=True,
                    )
                    return radius_index, prepared_local_bar, drds_local_bar

                def _accumulate_anchor(carry, anchor_pos):
                    channels_carry, prepared_carry = carry
                    radius_index, prepared_local_bar, drds_local_bar = _pullback_one_anchor(anchor_pos)
                    return (
                        dataclasses.replace(
                            channels_carry,
                            drds=channels_carry.drds.at[radius_index].add(drds_local_bar),
                        ),
                        _add_local_face_prepared_bar(prepared_carry, radius_index, prepared_local_bar),
                    ), None

                with _reverse_rebuild_profile_scope(
                    reverse_segment_profile_annotations,
                    "reverse_segment/rebuild_support/anchor_accumulation",
                ):
                    (face_channels_bar, face_prepared_bar), _ = jax.lax.scan(
                        _accumulate_anchor,
                        (face_channels_bar, face_prepared_bar),
                        anchor_positions,
                    )
                face_support_bar = _support_bar_from_face_bars(support, face_channels_bar, face_prepared_bar)
                if center_response_bar is None:
                    return face_support_bar
                center_support_bar = self.pullback_build_lagged_response_support_payload(
                    state,
                    NTXExactLijLaggedResponse(center_response=center_response_bar),
                    support,
                )
                return jax.tree_util.tree_map(lambda lhs, rhs: lhs + rhs, face_support_bar, center_support_bar)

            if isinstance(face_response_bar, NTXPreparedCoefficientResponse):
                radius_indices = jnp.arange(face_state.Er.shape[0], dtype=jnp.int32)

                def _accumulate_radius(carry, radius_index):
                    channels_carry, prepared_carry = carry
                    local_bar = jax.tree_util.tree_map(
                        lambda arr: jax.lax.dynamic_index_in_dim(arr, radius_index, axis=0, keepdims=False),
                        face_response_bar,
                    )
                    prepared_local_bar, drds_local_bar = _local_face_support_pullback(
                        radius_index,
                        local_bar,
                        interpolated=False,
                    )
                    return (
                        dataclasses.replace(
                            channels_carry,
                            drds=channels_carry.drds.at[radius_index].add(drds_local_bar),
                        ),
                        _add_local_face_prepared_bar(prepared_carry, radius_index, prepared_local_bar),
                    ), None

                (face_channels_bar, face_prepared_bar), _ = jax.lax.scan(
                    _accumulate_radius,
                    (face_channels_bar, face_prepared_bar),
                    radius_indices,
                )
                face_support_bar = _support_bar_from_face_bars(support, face_channels_bar, face_prepared_bar)
                if center_response_bar is None:
                    return face_support_bar
                center_support_bar = self.pullback_build_lagged_response_support_payload(
                    state,
                    NTXExactLijLaggedResponse(center_response=center_response_bar),
                    support,
                )
                return jax.tree_util.tree_map(lambda lhs, rhs: lhs + rhs, face_support_bar, center_support_bar)

            face_channels_delta0 = _float_delta_tree_like(support.face_channels)
            face_prepared_delta0 = _float_delta_tree_like(support.face_prepared)
            _, support_delta_pullback = jax.vjp(
                lambda face_channels_delta, face_prepared_delta: self.with_support_payload(
                    _support_with_face_delta(
                        support,
                        face_channels_delta,
                        face_prepared_delta,
                    )
                ).build_lagged_response(state).face_response,
                face_channels_delta0,
                face_prepared_delta0,
            )
            face_channels_bar, face_prepared_bar = support_delta_pullback(face_response_bar)
            face_support_bar = _support_bar_from_face_bars(support, face_channels_bar, face_prepared_bar)
            if center_response_bar is None:
                return face_support_bar
            center_support_bar = self.pullback_build_lagged_response_support_payload(
                state,
                NTXExactLijLaggedResponse(center_response=center_response_bar),
                support,
            )
            return jax.tree_util.tree_map(lambda lhs, rhs: lhs + rhs, face_support_bar, center_support_bar)
        if center_response_bar is None:
            return _support_bar_from_center_bars(
                support,
                _float_delta_tree_like(support.center_channels),
                _float_delta_tree_like(support.center_prepared),
            )

        center_channels_bar = _float_delta_tree_like(support.center_channels)
        center_prepared_bar = _float_delta_tree_like(support.center_prepared)
        density = safe_density(state.density, self.density_floor)
        temperature = state.temperature
        v_thermal = get_v_thermal(self.species.mass, temperature)
        collisionality_kind = _collisionality_kind(self.collisionality_model)
        species_indices = jnp.arange(int(self.species.number_species), dtype=jnp.int32)

        def _local_support_pullback(radius_index, local_field_bars, *, interpolated: bool):
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
            er_value = jax.lax.dynamic_index_in_dim(state.Er, radius_index, axis=0, keepdims=False)
            temperature_local = jax.lax.dynamic_index_in_dim(
                temperature,
                radius_index,
                axis=1,
                keepdims=False,
            )
            density_local = jax.lax.dynamic_index_in_dim(
                density,
                radius_index,
                axis=1,
                keepdims=False,
            )
            vthermal_local = jax.lax.dynamic_index_in_dim(
                v_thermal,
                radius_index,
                axis=1,
                keepdims=False,
            )
            prepared_delta0 = _float_delta_tree_like(prepared)
            drds_delta0 = jnp.zeros_like(drds_value)

            def _response_from_support_delta(prepared_delta, drds_delta):
                prepared_value = _add_float_delta_tree(prepared, prepared_delta)
                drds_local = drds_value + drds_delta
                if interpolated:
                    return jax.vmap(
                        lambda species_index: self._build_interpolated_moment_response_local(
                            prepared_value,
                            drds_value=drds_local,
                            species_index=species_index,
                            er_value=er_value,
                            temperature_local=temperature_local,
                            density_local=density_local,
                            vthermal_local=vthermal_local,
                            collisionality_kind=collisionality_kind,
                        )
                    )(species_indices)
                return jax.vmap(
                    lambda species_index: self._build_coefficient_response_local(
                        prepared_value,
                        drds_value=drds_local,
                        species_index=species_index,
                        er_value=er_value,
                        temperature_local=temperature_local,
                        density_local=density_local,
                        vthermal_local=vthermal_local,
                        collisionality_kind=collisionality_kind,
                    )
                )(species_indices)

            _, pullback = jax.vjp(
                _response_from_support_delta,
                prepared_delta0,
                drds_delta0,
            )
            return pullback(local_field_bars)

        def _add_local_prepared_bar(prepared_bar, radius_index, local_bar):
            return jax.tree_util.tree_map(
                lambda arr, local_arr: arr.at[radius_index].add(local_arr),
                prepared_bar,
                local_bar,
            )

        if isinstance(center_response_bar, NTXInterpolatedMomentResponse):
            n_radius = int(state.Er.shape[0])
            anchor_indices = self._response_anchor_indices(n_radius)
            anchor_rho = jnp.asarray(self.geometry.r_grid, dtype=jnp.float64)[anchor_indices]
            target_rho = jnp.asarray(support.center_channels.rho, dtype=jnp.float64)
            n_anchor = int(anchor_indices.shape[0])
            response_field_bars = self._interpolated_response_field_bars(center_response_bar)
            response_field_bar_tuple = _interpolated_response_field_bar_tuple(response_field_bars)

            def _per_anchor_forward(radius_index):
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
                er_value = jax.lax.dynamic_index_in_dim(state.Er, radius_index, axis=0, keepdims=False)
                temperature_local = jax.lax.dynamic_index_in_dim(
                    temperature,
                    radius_index,
                    axis=1,
                    keepdims=False,
                )
                density_local = jax.lax.dynamic_index_in_dim(
                    density,
                    radius_index,
                    axis=1,
                    keepdims=False,
                )
                vthermal_local = jax.lax.dynamic_index_in_dim(
                    v_thermal,
                    radius_index,
                    axis=1,
                    keepdims=False,
                )
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
                _per_anchor_forward,
                anchor_indices,
                anchor_rho,
            )
            anchor_response_fields = (
                anchor_response[0],
                anchor_response[1],
                anchor_response[2],
                anchor_response[3],
            )

            target_rho_bar = jnp.zeros_like(target_rho)
            for anchor_field, field_bar in zip(anchor_response_fields, response_field_bar_tuple, strict=True):
                target_rho_bar = target_rho_bar + self._pullback_interpolate_anchor_target_rho(
                    anchor_indices,
                    anchor_field,
                    target_rho,
                    field_bar,
                )
            center_channels_bar = dataclasses.replace(
                center_channels_bar,
                rho=center_channels_bar.rho + target_rho_bar,
            )

            raw_anchor_response_bar = self._pullback_interpolated_anchor_response_fields(
                anchor_indices=anchor_indices,
                anchor_rho=anchor_rho,
                target_rho=target_rho,
                field_bars=response_field_bars,
            )
            raw_anchor_response_fields = _interpolated_response_field_bar_tuple(raw_anchor_response_bar)
            anchor_positions = jnp.arange(n_anchor, dtype=jnp.int32)

            def _pullback_one_anchor(anchor_pos):
                radius_index = jax.lax.dynamic_index_in_dim(
                    anchor_indices,
                    anchor_pos,
                    axis=0,
                    keepdims=False,
                )
                local_field_bars = tuple(
                    jax.lax.dynamic_index_in_dim(
                        field_bar,
                        anchor_pos,
                        axis=0,
                        keepdims=False,
                    )
                    for field_bar in raw_anchor_response_fields
                )
                prepared_local_bar, drds_local_bar = _local_support_pullback(
                    radius_index,
                    local_field_bars,
                    interpolated=True,
                )
                return radius_index, prepared_local_bar, drds_local_bar

            def _accumulate_anchor(carry, anchor_pos):
                channels_carry, prepared_carry = carry
                radius_index, prepared_local_bar, drds_local_bar = _pullback_one_anchor(anchor_pos)
                return (
                    dataclasses.replace(
                        channels_carry,
                        drds=channels_carry.drds.at[radius_index].add(drds_local_bar),
                    ),
                    _add_local_prepared_bar(prepared_carry, radius_index, prepared_local_bar),
                ), None

            (center_channels_bar, center_prepared_bar), _ = jax.lax.scan(
                _accumulate_anchor,
                (center_channels_bar, center_prepared_bar),
                anchor_positions,
            )
            return _support_bar_from_center_bars(support, center_channels_bar, center_prepared_bar)

        if isinstance(center_response_bar, NTXPreparedCoefficientResponse):
            radius_indices = jnp.arange(state.Er.shape[0], dtype=jnp.int32)

            def _accumulate_radius(carry, radius_index):
                channels_carry, prepared_carry = carry
                local_bar = jax.tree_util.tree_map(
                    lambda arr: jax.lax.dynamic_index_in_dim(arr, radius_index, axis=0, keepdims=False),
                    center_response_bar,
                )
                prepared_local_bar, drds_local_bar = _local_support_pullback(
                    radius_index,
                    local_bar,
                    interpolated=False,
                )
                return (
                    dataclasses.replace(
                        channels_carry,
                        drds=channels_carry.drds.at[radius_index].add(drds_local_bar),
                    ),
                    _add_local_prepared_bar(prepared_carry, radius_index, prepared_local_bar),
                ), None

            (center_channels_bar, center_prepared_bar), _ = jax.lax.scan(
                _accumulate_radius,
                (center_channels_bar, center_prepared_bar),
                radius_indices,
            )
            return _support_bar_from_center_bars(support, center_channels_bar, center_prepared_bar)

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

    def pullback_build_lagged_response_support_payload_batched_interpolated_faces(
        self,
        state,
        lagged_response_bars,
        support,
    ):
        """Transpose interpolated face support for several objective RHS.

        This is deliberately narrower than the scalar public support-pullback:
        it covers the runtime benchmark's ``interpolate_from_faces`` response
        representation only.  Each radius constructs its local support VJP
        once, then applies that same transpose to the objective batch on the
        device.  The objective axis exists only in local cotangents and output
        support bars; state and the primal support payload stay unbatched.
        """
        if not isinstance(lagged_response_bars, NTXExactLijLaggedResponse):
            raise TypeError("batched NTX support pullback requires NTXExactLijLaggedResponse bars.")
        face_response_bars = lagged_response_bars.face_response
        if not isinstance(face_response_bars, NTXInterpolatedMomentResponse):
            raise NotImplementedError(
                "batched NTX support pullback currently requires interpolated face-response bars."
            )
        if lagged_response_bars.center_response is not None:
            raise NotImplementedError(
                "batched NTX support pullback currently requires center_response=None "
                "(the interpolate_from_faces runtime lane)."
            )

        objective_count = int(jnp.asarray(face_response_bars.reference_er).shape[0])
        face_state = build_face_transport_state(
            state,
            self.geometry,
            bc_density=self.bc_density,
            bc_temperature=self.bc_temperature,
            density_floor=self.density_floor,
            temperature_floor=self.temperature_floor,
        )
        face_density = safe_density(face_state.density, self.density_floor)
        face_temperature = face_state.temperature
        face_v_thermal = get_v_thermal(self.species.mass, face_temperature)
        collisionality_kind = _collisionality_kind(self.collisionality_model)
        species_indices = jnp.arange(int(self.species.number_species), dtype=jnp.int32)

        def _batched_zero_like(tree):
            return jax.tree_util.tree_map(
                lambda leaf: jnp.broadcast_to(
                    jnp.zeros_like(jnp.asarray(leaf)),
                    (objective_count,) + jnp.asarray(leaf).shape,
                ),
                tree,
            )

        face_channels_bar = _batched_zero_like(_float_delta_tree_like(support.face_channels))
        face_prepared_bar = _batched_zero_like(_float_delta_tree_like(support.face_prepared))
        n_radius = int(face_state.Er.shape[0])
        anchor_indices = self._response_anchor_indices(n_radius)
        anchor_rho = jnp.asarray(self.geometry.r_grid_half, dtype=jnp.float64)[anchor_indices]
        target_rho = jnp.asarray(support.face_channels.rho, dtype=jnp.float64)
        n_anchor = int(anchor_indices.shape[0])

        response_field_bars = jax.vmap(self._interpolated_response_field_bars)(face_response_bars)
        response_field_bar_tuple = _interpolated_response_field_bar_tuple(response_field_bars)

        def _per_anchor_forward(radius_index):
            prepared = jax.tree_util.tree_map(
                lambda arr: jax.lax.dynamic_index_in_dim(arr, radius_index, axis=0, keepdims=False),
                support.face_prepared,
            )
            drds_value = jax.lax.dynamic_index_in_dim(
                support.face_channels.drds, radius_index, axis=0, keepdims=False
            )
            er_value = jax.lax.dynamic_index_in_dim(face_state.Er, radius_index, axis=0, keepdims=False)
            temperature_local = jax.lax.dynamic_index_in_dim(
                face_temperature, radius_index, axis=1, keepdims=False
            )
            density_local = jax.lax.dynamic_index_in_dim(
                face_density, radius_index, axis=1, keepdims=False
            )
            vthermal_local = jax.lax.dynamic_index_in_dim(
                face_v_thermal, radius_index, axis=1, keepdims=False
            )
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
            _per_anchor_forward,
            anchor_indices,
            anchor_rho,
        )
        target_rho_bar = jnp.zeros((objective_count,) + target_rho.shape, dtype=target_rho.dtype)
        for anchor_field, field_bar in zip(
            (anchor_response[0], anchor_response[1], anchor_response[2], anchor_response[3]),
            response_field_bar_tuple,
            strict=True,
        ):
            target_rho_bar = target_rho_bar + jax.vmap(
                lambda one_field_bar: self._pullback_interpolate_anchor_target_rho(
                    anchor_indices,
                    anchor_field,
                    target_rho,
                    one_field_bar,
                )
            )(field_bar)
        face_channels_bar = dataclasses.replace(
            face_channels_bar,
            rho=face_channels_bar.rho + target_rho_bar,
        )

        raw_anchor_response_bar = jax.vmap(
            lambda one_field_bars: self._pullback_interpolated_anchor_response_fields(
                anchor_indices=anchor_indices,
                anchor_rho=anchor_rho,
                target_rho=target_rho,
                field_bars=one_field_bars,
            )
        )(response_field_bars)
        raw_anchor_response_fields = _interpolated_response_field_bar_tuple(raw_anchor_response_bar)
        anchor_positions = jnp.arange(n_anchor, dtype=jnp.int32)

        def _local_face_support_pullback(radius_index, local_field_bars):
            prepared = jax.tree_util.tree_map(
                lambda arr: jax.lax.dynamic_index_in_dim(arr, radius_index, axis=0, keepdims=False),
                support.face_prepared,
            )
            drds_value = jax.lax.dynamic_index_in_dim(
                support.face_channels.drds, radius_index, axis=0, keepdims=False
            )
            er_value = jax.lax.dynamic_index_in_dim(face_state.Er, radius_index, axis=0, keepdims=False)
            temperature_local = jax.lax.dynamic_index_in_dim(
                face_temperature, radius_index, axis=1, keepdims=False
            )
            density_local = jax.lax.dynamic_index_in_dim(
                face_density, radius_index, axis=1, keepdims=False
            )
            vthermal_local = jax.lax.dynamic_index_in_dim(
                face_v_thermal, radius_index, axis=1, keepdims=False
            )
            prepared_delta0 = _float_delta_tree_like(prepared)
            drds_delta0 = jnp.zeros_like(drds_value)

            def _response_from_support_delta(prepared_delta, drds_delta):
                prepared_value = _add_float_delta_tree(prepared, prepared_delta)
                drds_local = drds_value + drds_delta
                return jax.vmap(
                    lambda species_index: self._build_interpolated_moment_response_local(
                        prepared_value,
                        drds_value=drds_local,
                        species_index=species_index,
                        er_value=er_value,
                        temperature_local=temperature_local,
                        density_local=density_local,
                        vthermal_local=vthermal_local,
                        collisionality_kind=collisionality_kind,
                    )
                )(species_indices)

            _, local_pullback = jax.vjp(
                _response_from_support_delta,
                prepared_delta0,
                drds_delta0,
            )
            # ``prepared_delta0`` includes NTX pytrees with static dataclass
            # metadata. Vmapping the raw pullback result asks JAX to batch and
            # reconstruct that metadata (which is invalid). Map only its
            # numeric leaves, then restore the unchanged pytree structure.
            _, prepared_treedef = jax.tree_util.tree_flatten(prepared_delta0)

            def _local_pullback_leaves(one_local_field_bars):
                prepared_bar, drds_bar = local_pullback(one_local_field_bars)
                return (*jax.tree_util.tree_leaves(prepared_bar), drds_bar)

            batched_leaves = jax.vmap(_local_pullback_leaves)(local_field_bars)
            return prepared_treedef.unflatten(batched_leaves[:-1]), batched_leaves[-1]

        def _accumulate_anchor(carry, anchor_pos):
            channels_carry, prepared_carry = carry
            radius_index = jax.lax.dynamic_index_in_dim(
                anchor_indices, anchor_pos, axis=0, keepdims=False
            )
            local_field_bars = tuple(
                jax.lax.dynamic_index_in_dim(field_bar, anchor_pos, axis=1, keepdims=False)
                for field_bar in raw_anchor_response_fields
            )
            prepared_local_bar, drds_local_bar = _local_face_support_pullback(
                radius_index,
                local_field_bars,
            )
            return (
                dataclasses.replace(
                    channels_carry,
                    drds=channels_carry.drds.at[:, radius_index].add(drds_local_bar),
                ),
                jax.tree_util.tree_map(
                    lambda arr, local_arr: arr.at[:, radius_index].add(local_arr),
                    prepared_carry,
                    prepared_local_bar,
                ),
            ), None

        (face_channels_bar, face_prepared_bar), _ = jax.lax.scan(
            _accumulate_anchor,
            (face_channels_bar, face_prepared_bar),
            anchor_positions,
        )
        # Mirror the outer scalar ``lax.map`` result: every support leaf must
        # carry the objective axis, including inactive center/non-float leaves.
        return dataclasses.replace(
            _batched_zero_like(_float_delta_tree_like(support)),
            face_channels=face_channels_bar,
            face_prepared=face_prepared_bar,
        )

    def pullback_build_lagged_response_support_payload_batched_interpolated_faces_reuse_local_vjp_primal(
        self, state, lagged_response_bars, support,
    ):
        """Exact batched face transpose using the primal of each local VJP.

        The temporary anchor responses have no objective axis.  The VJP
        pullback alone is batched over objectives, so this keeps the existing
        batched rule's device multi-RHS behavior while removing its separate
        per-anchor NTX forward evaluation.
        """
        if not isinstance(lagged_response_bars, NTXExactLijLaggedResponse):
            raise TypeError("batched NTX support pullback requires NTXExactLijLaggedResponse bars.")
        face_response_bars = lagged_response_bars.face_response
        if not isinstance(face_response_bars, NTXInterpolatedMomentResponse):
            raise NotImplementedError("batched NTX support pullback requires interpolated face-response bars.")
        if lagged_response_bars.center_response is not None:
            raise NotImplementedError("batched NTX support pullback requires center_response=None.")
        objective_count = int(jnp.asarray(face_response_bars.reference_er).shape[0])
        face_state = build_face_transport_state(
            state, self.geometry, bc_density=self.bc_density, bc_temperature=self.bc_temperature,
            density_floor=self.density_floor, temperature_floor=self.temperature_floor,
        )
        face_density = safe_density(face_state.density, self.density_floor)
        face_temperature = face_state.temperature
        face_v_thermal = get_v_thermal(self.species.mass, face_temperature)
        collisionality_kind = _collisionality_kind(self.collisionality_model)
        species_indices = jnp.arange(int(self.species.number_species), dtype=jnp.int32)

        def _batched_zero_like(tree):
            return jax.tree_util.tree_map(
                lambda leaf: jnp.broadcast_to(jnp.zeros_like(jnp.asarray(leaf)),
                                               (objective_count,) + jnp.asarray(leaf).shape),
                tree,
            )

        face_channels_bar = _batched_zero_like(_float_delta_tree_like(support.face_channels))
        face_prepared_bar = _batched_zero_like(_float_delta_tree_like(support.face_prepared))
        n_radius = int(face_state.Er.shape[0])
        anchor_indices = self._response_anchor_indices(n_radius)
        anchor_rho = jnp.asarray(self.geometry.r_grid_half, dtype=jnp.float64)[anchor_indices]
        target_rho = jnp.asarray(support.face_channels.rho, dtype=jnp.float64)
        n_anchor = int(anchor_indices.shape[0])
        response_field_bars = jax.vmap(self._interpolated_response_field_bars)(face_response_bars)
        response_field_bar_tuple = _interpolated_response_field_bar_tuple(response_field_bars)
        raw_anchor_response_bar = jax.vmap(
            lambda bars: self._pullback_interpolated_anchor_response_fields(
                anchor_indices=anchor_indices, anchor_rho=anchor_rho,
                target_rho=target_rho, field_bars=bars,
            )
        )(response_field_bars)
        raw_anchor_response_fields = _interpolated_response_field_bar_tuple(raw_anchor_response_bar)
        anchor_response_fields0 = tuple(
            jnp.zeros((n_anchor,) + field_bar.shape[2:], dtype=field_bar.dtype)
            for field_bar in raw_anchor_response_fields
        )

        def _one_anchor(radius_index, local_field_bars):
            prepared = jax.tree_util.tree_map(
                lambda arr: jax.lax.dynamic_index_in_dim(arr, radius_index, axis=0, keepdims=False),
                support.face_prepared,
            )
            drds_value = jax.lax.dynamic_index_in_dim(support.face_channels.drds, radius_index, axis=0, keepdims=False)
            er_value = jax.lax.dynamic_index_in_dim(face_state.Er, radius_index, axis=0, keepdims=False)
            temperature_local = jax.lax.dynamic_index_in_dim(face_temperature, radius_index, axis=1, keepdims=False)
            density_local = jax.lax.dynamic_index_in_dim(face_density, radius_index, axis=1, keepdims=False)
            vthermal_local = jax.lax.dynamic_index_in_dim(face_v_thermal, radius_index, axis=1, keepdims=False)
            prepared_delta0 = _float_delta_tree_like(prepared)
            drds_delta0 = jnp.zeros_like(drds_value)

            def _response(delta_prepared, delta_drds):
                prepared_value = _add_float_delta_tree(prepared, delta_prepared)
                return jax.vmap(
                    lambda species_index: self._build_interpolated_moment_response_local(
                        prepared_value, drds_value=drds_value + delta_drds,
                        species_index=species_index, er_value=er_value,
                        temperature_local=temperature_local, density_local=density_local,
                        vthermal_local=vthermal_local, collisionality_kind=collisionality_kind,
                    )
                )(species_indices)

            primal_response, pullback = jax.vjp(_response, prepared_delta0, drds_delta0)
            _, treedef = jax.tree_util.tree_flatten(prepared_delta0)
            def _pullback_leaves(one_bars):
                prepared_bar, drds_bar = pullback(one_bars)
                return (*jax.tree_util.tree_leaves(prepared_bar), drds_bar)
            batched_leaves = jax.vmap(_pullback_leaves)(local_field_bars)
            return primal_response, treedef.unflatten(batched_leaves[:-1]), batched_leaves[-1]

        def _accumulate(carry, anchor_pos):
            channels, prepared_bars, anchor_fields = carry
            radius_index = jax.lax.dynamic_index_in_dim(anchor_indices, anchor_pos, axis=0, keepdims=False)
            local_field_bars = tuple(
                jax.lax.dynamic_index_in_dim(field_bar, anchor_pos, axis=1, keepdims=False)
                for field_bar in raw_anchor_response_fields
            )
            local_response, local_prepared, local_drds = _one_anchor(radius_index, local_field_bars)
            anchor_fields = tuple(
                field.at[anchor_pos].set(value)
                for field, value in zip(anchor_fields, local_response, strict=True)
            )
            return (
                dataclasses.replace(channels, drds=channels.drds.at[:, radius_index].add(local_drds)),
                jax.tree_util.tree_map(
                    lambda values, local: values.at[:, radius_index].add(local), prepared_bars, local_prepared
                ),
                anchor_fields,
            ), None

        (face_channels_bar, face_prepared_bar, raw_anchor_fields), _ = jax.lax.scan(
            _accumulate,
            (face_channels_bar, face_prepared_bar, anchor_response_fields0),
            jnp.arange(n_anchor, dtype=jnp.int32),
        )
        anchor_fields = raw_anchor_fields if n_anchor < 4 else jax.lax.cond(
            jnp.isclose(anchor_rho[0], 0.0),
            lambda _: self._regularize_axis_radius0(raw_anchor_fields, anchor_rho),
            lambda _: raw_anchor_fields, operand=None,
        )
        target_rho_bar = jnp.zeros((objective_count,) + target_rho.shape, dtype=target_rho.dtype)
        for anchor_field, field_bar in zip(anchor_fields, response_field_bar_tuple, strict=True):
            target_rho_bar = target_rho_bar + jax.vmap(
                lambda one_bar: self._pullback_interpolate_anchor_target_rho(
                    anchor_indices, anchor_field, target_rho, one_bar
                )
            )(field_bar)
        face_channels_bar = dataclasses.replace(face_channels_bar, rho=face_channels_bar.rho + target_rho_bar)
        return dataclasses.replace(
            _batched_zero_like(_float_delta_tree_like(support)),
            face_channels=face_channels_bar, face_prepared=face_prepared_bar,
        )

    def pullback_build_lagged_response_support_payload_batched_interpolated_faces_multi_rhs_shared_primal(
        self, state, lagged_response_bars, support, *, native_factorized_ntx_rhs: bool = False,
        native_compact_ntx_rhs: bool = False,
        native_compact_residual_ntx_rhs: bool = False,
        reuse_joint_moment_drds_jvp: bool = False,
        return_native_vmec_coefficient_bars: bool = False,
        native_vmec_direct_directional_product_rule: bool = False,
        native_direct_coefficient_pullback: bool = False,
        native_per_energy_call_boundary: bool = False,
    ):
        """Experimental exact batched support transpose with local NTX sharing.

        This private opt-in path is deliberately separate from the established
        ``separate_reuse_local_vjp_primal`` route.  At one anchor/species it
        forms NTX's primal modes, factorisation, and the two forward
        case-direction solves once; the prepared-support adjoints then carry
        only the objective RHS axis.  The temporary factorisation never leaves
        that local helper and no state is retained across anchors or steps.

        It is intentionally not selected by a CLI/reverse-mode dispatch until
        an exact comparison is performed on the benchmark machine.
        """
        if not isinstance(lagged_response_bars, NTXExactLijLaggedResponse):
            raise TypeError("batched NTX support pullback requires NTXExactLijLaggedResponse bars.")
        face_response_bars = lagged_response_bars.face_response
        if not isinstance(face_response_bars, NTXInterpolatedMomentResponse):
            raise NotImplementedError("batched NTX support pullback requires interpolated face-response bars.")
        if lagged_response_bars.center_response is not None:
            raise NotImplementedError("batched NTX support pullback requires center_response=None.")

        objective_count = int(jnp.asarray(face_response_bars.reference_er).shape[0])
        face_state = build_face_transport_state(
            state, self.geometry, bc_density=self.bc_density, bc_temperature=self.bc_temperature,
            density_floor=self.density_floor, temperature_floor=self.temperature_floor,
        )
        face_density = safe_density(face_state.density, self.density_floor)
        face_temperature = face_state.temperature
        face_v_thermal = get_v_thermal(self.species.mass, face_temperature)
        collisionality_kind = _collisionality_kind(self.collisionality_model)
        species_indices = jnp.arange(int(self.species.number_species), dtype=jnp.int32)

        def _batched_zero_like(tree):
            return jax.tree_util.tree_map(
                lambda leaf: jnp.broadcast_to(
                    jnp.zeros_like(jnp.asarray(leaf)),
                    (objective_count,) + jnp.asarray(leaf).shape,
                ),
                tree,
            )

        face_channels_bar = _batched_zero_like(_float_delta_tree_like(support.face_channels))
        face_prepared_bar = _batched_zero_like(_float_delta_tree_like(support.face_prepared))
        native_vmec_coefficient_bar = None
        if return_native_vmec_coefficient_bars:
            surface = support.face_prepared.surface
            native_vmec_coefficient_bar = {
                name: jnp.zeros(
                    (objective_count,) + jnp.asarray(getattr(surface, name)).shape,
                    dtype=jnp.asarray(getattr(surface, name)).dtype,
                )
                for name in (
                    "b_cos",
                    "jacobian_cos",
                    "b_sub_theta_cos",
                    "b_sub_zeta_cos",
                    "b_sup_theta_cos",
                    "b_sup_zeta_cos",
                    "b0",
                )
            }
        n_radius = int(face_state.Er.shape[0])
        anchor_indices = self._response_anchor_indices(n_radius)
        anchor_rho = jnp.asarray(self.geometry.r_grid_half, dtype=jnp.float64)[anchor_indices]
        target_rho = jnp.asarray(support.face_channels.rho, dtype=jnp.float64)
        n_anchor = int(anchor_indices.shape[0])
        response_field_bars = jax.vmap(self._interpolated_response_field_bars)(face_response_bars)
        response_field_bar_tuple = _interpolated_response_field_bar_tuple(response_field_bars)
        raw_anchor_response_bar = jax.vmap(
            lambda bars: self._pullback_interpolated_anchor_response_fields(
                anchor_indices=anchor_indices, anchor_rho=anchor_rho,
                target_rho=target_rho, field_bars=bars,
            )
        )(response_field_bars)
        raw_anchor_response_fields = _interpolated_response_field_bar_tuple(raw_anchor_response_bar)
        anchor_response_fields0 = tuple(
            jnp.zeros((n_anchor,) + field_bar.shape[2:], dtype=field_bar.dtype)
            for field_bar in raw_anchor_response_fields
        )

        def _one_anchor(radius_index, local_field_bars):
            prepared = jax.tree_util.tree_map(
                lambda arr: jax.lax.dynamic_index_in_dim(arr, radius_index, axis=0, keepdims=False),
                support.face_prepared,
            )
            drds_value = jax.lax.dynamic_index_in_dim(
                support.face_channels.drds, radius_index, axis=0, keepdims=False,
            )
            er_value = jax.lax.dynamic_index_in_dim(face_state.Er, radius_index, axis=0, keepdims=False)
            temperature_local = jax.lax.dynamic_index_in_dim(face_temperature, radius_index, axis=1, keepdims=False)
            density_local = jax.lax.dynamic_index_in_dim(face_density, radius_index, axis=1, keepdims=False)
            vthermal_local = jax.lax.dynamic_index_in_dim(face_v_thermal, radius_index, axis=1, keepdims=False)
            prepared_treedef = jax.tree_util.tree_structure(prepared)

            def _one_species(species_index, *species_field_bars):
                reference_nu_hat, reference_epsi_hat, vth_a = (
                    self._interpolated_moment_local_scan_primitives(
                        drds_value=drds_value,
                        species_index=species_index,
                        er_value=er_value,
                        temperature_local=temperature_local,
                        density_local=density_local,
                        vthermal_local=vthermal_local,
                        collisionality_kind=collisionality_kind,
                    )
                )
                if native_compact_residual_ntx_rhs:
                    local_support_pullback = (
                        self._pullback_interpolated_moment_prepared_support_and_drds_only_native_multi_rhs_compact_residual_reuse_moment_drds_jvp
                        if reuse_joint_moment_drds_jvp
                        else self._pullback_interpolated_moment_prepared_support_and_drds_only_native_multi_rhs_compact_residual
                    )
                elif native_compact_ntx_rhs:
                    local_support_pullback = (
                        self._pullback_interpolated_moment_prepared_support_and_drds_only_native_multi_rhs_compact
                    )
                elif return_native_vmec_coefficient_bars:
                    local_support_pullback = (
                        self._pullback_interpolated_moment_prepared_support_and_drds_only_native_multi_rhs_reuse_moment_drds_jvp_with_vmec_coefficients
                    )
                elif reuse_joint_moment_drds_jvp:
                    local_support_pullback = (
                        self._pullback_interpolated_moment_prepared_support_and_drds_only_native_multi_rhs_reuse_moment_drds_jvp
                    )
                elif native_factorized_ntx_rhs:
                    local_support_pullback = (
                        self._pullback_interpolated_moment_prepared_support_and_drds_only_native_multi_rhs
                    )
                else:
                    local_support_pullback = (
                        self._pullback_interpolated_moment_prepared_support_and_drds_only_multi_rhs
                    )
                local_kwargs = {}
                if (
                    native_factorized_ntx_rhs
                    or native_compact_ntx_rhs
                    or native_compact_residual_ntx_rhs
                    or return_native_vmec_coefficient_bars
                ):
                    local_kwargs["return_case_bars"] = True
                if return_native_vmec_coefficient_bars:
                    local_kwargs["native_vmec_coefficient_bars_only"] = True
                    local_kwargs["native_vmec_direct_directional_product_rule"] = (
                        native_vmec_direct_directional_product_rule
                    )
                    local_kwargs["native_direct_coefficient_pullback"] = (
                        native_direct_coefficient_pullback
                    )
                    local_kwargs["native_per_energy_call_boundary"] = (
                        native_per_energy_call_boundary
                    )
                local_result = local_support_pullback(
                    prepared,
                    drds_value=drds_value,
                    reference_nu_hat=reference_nu_hat,
                    reference_epsi_hat=reference_epsi_hat,
                    vth_a=vth_a,
                    field_bars=species_field_bars,
                    **local_kwargs,
                )
                if (
                    native_factorized_ntx_rhs
                    or native_compact_ntx_rhs
                    or native_compact_residual_ntx_rhs
                    or return_native_vmec_coefficient_bars
                ):
                    if return_native_vmec_coefficient_bars:
                        (
                            prepared_bar,
                            drds_bar,
                            primal_response,
                            (nu_hat_bars, epsi_hat_bars, vth_a_bars),
                            vmec_coefficient_bars,
                        ) = local_result
                    else:
                        (
                            prepared_bar,
                            drds_bar,
                            primal_response,
                            (nu_hat_bars, epsi_hat_bars, vth_a_bars),
                        ) = local_result
                    primitive_drds_bar, _er_bar, _temperature_bar, _density_bar = jax.vmap(
                        lambda nu_hat_bar, epsi_hat_bar, vth_a_bar: self._pullback_local_scan_inputs_and_drds_from_primitives(
                            drds_value=drds_value,
                            species_index=species_index,
                            er_value=er_value,
                            temperature_local=temperature_local,
                            density_local=density_local,
                            collisionality_kind=collisionality_kind,
                            reference_nu_hat_bar=nu_hat_bar,
                            reference_epsi_hat_bar=epsi_hat_bar,
                            vth_a_bar=vth_a_bar,
                        )
                    )(nu_hat_bars, epsi_hat_bars, vth_a_bars)
                    drds_bar = drds_bar + primitive_drds_bar
                else:
                    prepared_bar, drds_bar, primal_response = local_result
                result = (
                    # The multi-RHS NTX helper already converts static/float0
                    # prepared leaves into explicit zero arrays carrying the
                    # RHS axis.  Applying the scalar sanitizer here would
                    # replace only static metadata (for example Fourier mode
                    # arrays) by unbatched zeros while retaining batched
                    # floating geometry arrays, violating NTX dataclass shape
                    # invariants during this species ``vmap``.
                    tuple(jax.tree_util.tree_leaves(prepared_bar)),
                    drds_bar,
                    primal_response,
                )
                if return_native_vmec_coefficient_bars:
                    return (*result, vmec_coefficient_bars)
                return result

            species_result = jax.vmap(
                _one_species,
                in_axes=(0,) + (1,) * len(local_field_bars),
            )(species_indices, *local_field_bars)
            if return_native_vmec_coefficient_bars:
                (
                    prepared_species_bar_leaves,
                    drds_species_bars,
                    primal_response,
                    vmec_species_coefficient_bars,
                ) = species_result
            else:
                prepared_species_bar_leaves, drds_species_bars, primal_response = species_result
            prepared_bar = prepared_treedef.unflatten(
                tuple(jnp.sum(values, axis=0) for values in prepared_species_bar_leaves)
            )
            result = (primal_response, prepared_bar, jnp.sum(drds_species_bars, axis=0))
            if return_native_vmec_coefficient_bars:
                return (
                    *result,
                    jax.tree_util.tree_map(
                        lambda values: jnp.sum(values, axis=0),
                        vmec_species_coefficient_bars,
                    ),
                )
            return result

        def _accumulate(carry, anchor_pos):
            channels, prepared_bars, anchor_fields, coefficient_bars = carry
            radius_index = jax.lax.dynamic_index_in_dim(anchor_indices, anchor_pos, axis=0, keepdims=False)
            local_field_bars = tuple(
                jax.lax.dynamic_index_in_dim(field_bar, anchor_pos, axis=1, keepdims=False)
                for field_bar in raw_anchor_response_fields
            )
            anchor_result = _one_anchor(radius_index, local_field_bars)
            if return_native_vmec_coefficient_bars:
                local_response, local_prepared, local_drds, local_coefficient_bars = anchor_result
                coefficient_bars = {
                    name: values.at[:, radius_index].add(local_coefficient_bars[name])
                    for name, values in coefficient_bars.items()
                }
            else:
                local_response, local_prepared, local_drds = anchor_result
            anchor_fields = tuple(
                field.at[anchor_pos].set(value)
                for field, value in zip(anchor_fields, local_response, strict=True)
            )
            return (
                dataclasses.replace(channels, drds=channels.drds.at[:, radius_index].add(local_drds)),
                jax.tree_util.tree_map(
                    lambda values, local: values.at[:, radius_index].add(local),
                    prepared_bars, local_prepared,
                ),
                anchor_fields,
                coefficient_bars,
            ), None

        (face_channels_bar, face_prepared_bar, raw_anchor_fields, native_vmec_coefficient_bar), _ = jax.lax.scan(
            _accumulate,
            (face_channels_bar, face_prepared_bar, anchor_response_fields0, native_vmec_coefficient_bar),
            jnp.arange(n_anchor, dtype=jnp.int32),
        )
        anchor_fields = raw_anchor_fields if n_anchor < 4 else jax.lax.cond(
            jnp.isclose(anchor_rho[0], 0.0),
            lambda _: self._regularize_axis_radius0(raw_anchor_fields, anchor_rho),
            lambda _: raw_anchor_fields,
            operand=None,
        )
        target_rho_bar = jnp.zeros((objective_count,) + target_rho.shape, dtype=target_rho.dtype)
        for anchor_field, field_bar in zip(anchor_fields, response_field_bar_tuple, strict=True):
            target_rho_bar = target_rho_bar + jax.vmap(
                lambda one_bar: self._pullback_interpolate_anchor_target_rho(
                    anchor_indices, anchor_field, target_rho, one_bar,
                )
            )(field_bar)
        face_channels_bar = dataclasses.replace(face_channels_bar, rho=face_channels_bar.rho + target_rho_bar)
        support_bar = dataclasses.replace(
            _batched_zero_like(_float_delta_tree_like(support)),
            face_channels=face_channels_bar,
            face_prepared=face_prepared_bar,
        )
        if return_native_vmec_coefficient_bars:
            return support_bar, native_vmec_coefficient_bar
        return support_bar

    def pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_shared_primal(
        self, state, lagged_response_bars, support,
    ):
        """Opt-in native matrix-RHS counterpart of the rejected prior helper.

        This method is intentionally separate so the former experimental
        selector retains its original dispatch and remains an independent
        timing reference.
        """

        return self.pullback_build_lagged_response_support_payload_batched_interpolated_faces_multi_rhs_shared_primal(
            state,
            lagged_response_bars,
            support,
            native_factorized_ntx_rhs=True,
        )

    def pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_compact_shared_primal(
        self, state, lagged_response_bars, support,
    ):
        """Opt-in compact-return native matrix-RHS support transpose.

        The NTX factorisation and objective RHS batch are identical to the
        existing native selector.  Only its transient return payload is
        compacted before NEOPAX reduces over energy.
        """

        return self.pullback_build_lagged_response_support_payload_batched_interpolated_faces_multi_rhs_shared_primal(
            state,
            lagged_response_bars,
            support,
            native_factorized_ntx_rhs=True,
            native_compact_ntx_rhs=True,
        )

    def pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal(
        self, state, lagged_response_bars, support,
    ):
        """Opt-in native matrix-RHS transpose without duplicate drds JVPs."""
        return self.pullback_build_lagged_response_support_payload_batched_interpolated_faces_multi_rhs_shared_primal(
            state, lagged_response_bars, support,
            native_factorized_ntx_rhs=True,
            reuse_joint_moment_drds_jvp=True,
        )

    def pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients(
        self, state, lagged_response_bars, support,
    ):
        """Return normal support bars plus NTX-native face coefficient bars.

        The second result is deliberately out-of-band: it is a cotangent of
        VMEC face surface coefficients, not a leaf of the primal NTX support
        payload.  The reverse transport kernel must route it separately.
        """

        return self.pullback_build_lagged_response_support_payload_batched_interpolated_faces_multi_rhs_shared_primal(
            state,
            lagged_response_bars,
            support,
            native_factorized_ntx_rhs=True,
            reuse_joint_moment_drds_jvp=True,
            return_native_vmec_coefficient_bars=True,
        )

    def pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_directional_product_rule(
        self, state, lagged_response_bars, support,
    ):
        """Private opt-in forwarding the exact direct directional NTX rule.

        This differs from the established VMEC-coefficient hook only in the
        post-adjoint directional primitive contraction inside NTX.  The
        grouped NTX solve, factorisation, matrix-RHS adjoint and coefficient
        bridge are otherwise identical.
        """
        return self.pullback_build_lagged_response_support_payload_batched_interpolated_faces_multi_rhs_shared_primal(
            state,
            lagged_response_bars,
            support,
            native_factorized_ntx_rhs=True,
            reuse_joint_moment_drds_jvp=True,
            return_native_vmec_coefficient_bars=True,
            native_vmec_direct_directional_product_rule=True,
        )

    def pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_coefficient_pullback(
        self, state, lagged_response_bars, support,
    ):
        """Opt-in exact replacement of the upstream coefficient VJP/JVP pair.

        The grouped primal, NTX factorisation, matrix-RHS adjoints and native
        VMEC bridge are identical to the validated fast selector.  This keeps
        its post-adjoint direct directional contractions and additionally
        replaces the earlier coefficient-to-retained-mode VJP/JVP nest.
        """
        return self.pullback_build_lagged_response_support_payload_batched_interpolated_faces_multi_rhs_shared_primal(
            state,
            lagged_response_bars,
            support,
            native_factorized_ntx_rhs=True,
            reuse_joint_moment_drds_jvp=True,
            return_native_vmec_coefficient_bars=True,
            native_vmec_direct_directional_product_rule=True,
            native_direct_coefficient_pullback=True,
        )

    def pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_directional_product_rule_per_energy_call_boundary(
        self, state, lagged_response_bars, support,
    ):
        """Keep the validated grouped algebra behind a per-energy XLA call."""
        return self.pullback_build_lagged_response_support_payload_batched_interpolated_faces_multi_rhs_shared_primal(
            state,
            lagged_response_bars,
            support,
            native_factorized_ntx_rhs=True,
            reuse_joint_moment_drds_jvp=True,
            return_native_vmec_coefficient_bars=True,
            native_vmec_direct_directional_product_rule=True,
            native_per_energy_call_boundary=True,
        )

    def pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_compact_residual_reuse_moment_drds_jvp_shared_primal(
        self, state, lagged_response_bars, support,
    ):
        """Opt-in split-residual version of the validated native drds rule."""
        return self.pullback_build_lagged_response_support_payload_batched_interpolated_faces_multi_rhs_shared_primal(
            state,
            lagged_response_bars,
            support,
            native_factorized_ntx_rhs=True,
            native_compact_residual_ntx_rhs=True,
            reuse_joint_moment_drds_jvp=True,
        )

    def pullback_build_lagged_response_state_and_support_payload_batched_interpolated_faces(
        self,
        state,
        lagged_response_bars,
        support,
        *,
        packed_support_directional_adjoint: bool = False,
        reuse_local_vjp_primal_anchor_response: bool = False,
        compact_prepared_support_carry: bool = False,
        native_factorized_ntx_rhs: bool = False,
        reuse_joint_moment_drds_jvp: bool = False,
        return_native_vmec_coefficient_bars: bool = False,
        omit_generic_prepared_carry: bool = False,
    ):
        """Joint exact transpose of interpolated face state and NTX support.

        The regular scalar reverse constructs the local NTX response transpose
        once for the state path and again for the support path.  This narrow
        batched helper instead requests their joint local pullback from NTX:
        one implicit adjoint supplies the state-field, ``drds``, and prepared
        geometry cotangents for every objective RHS.  It deliberately covers
        only the benchmark's interpolated-face/no-center-response lane.
        """
        if not isinstance(lagged_response_bars, NTXExactLijLaggedResponse):
            raise TypeError("batched joint NTX pullback requires NTXExactLijLaggedResponse bars.")
        face_response_bars = lagged_response_bars.face_response
        if not isinstance(face_response_bars, NTXInterpolatedMomentResponse):
            raise NotImplementedError(
                "batched joint NTX pullback requires interpolated face-response bars."
            )
        if lagged_response_bars.center_response is not None:
            raise NotImplementedError(
                "batched joint NTX pullback requires center_response=None "
                "(the interpolate_from_faces runtime lane)."
            )

        objective_count = int(jnp.asarray(face_response_bars.reference_er).shape[0])

        def _batched_zero_like(tree):
            return jax.tree_util.tree_map(
                lambda leaf: jnp.broadcast_to(
                    jnp.zeros_like(jnp.asarray(leaf)),
                    (objective_count,) + jnp.asarray(leaf).shape,
                ),
                tree,
            )

        face_state = build_face_transport_state(
            state,
            self.geometry,
            bc_density=self.bc_density,
            bc_temperature=self.bc_temperature,
            density_floor=self.density_floor,
            temperature_floor=self.temperature_floor,
        )
        face_density = safe_density(face_state.density, self.density_floor)
        face_temperature = face_state.temperature
        collisionality_kind = _collisionality_kind(self.collisionality_model)
        species_indices = jnp.arange(int(self.species.number_species), dtype=jnp.int32)
        face_channels_bar = _batched_zero_like(_float_delta_tree_like(support.face_channels))
        face_prepared_bar = (
            None
            if omit_generic_prepared_carry
            else _batched_zero_like(_float_delta_tree_like(support.face_prepared))
        )
        if return_native_vmec_coefficient_bars and not native_factorized_ntx_rhs:
            raise ValueError(
                "native VMEC coefficient bars require the native matrix-RHS NTX path."
            )
        if omit_generic_prepared_carry and not return_native_vmec_coefficient_bars:
            raise ValueError(
                "omitting the generic prepared carry requires native VMEC coefficient bars."
            )
        native_vmec_coefficient_bar = None
        if return_native_vmec_coefficient_bars:
            surface = support.face_prepared.surface
            native_vmec_coefficient_bar = {
                name: jnp.zeros(
                    (objective_count,) + jnp.asarray(getattr(surface, name)).shape,
                    dtype=jnp.asarray(getattr(surface, name)).dtype,
                )
                for name in (
                    "b_cos",
                    "jacobian_cos",
                    "b_sub_theta_cos",
                    "b_sub_zeta_cos",
                    "b_sup_theta_cos",
                    "b_sup_zeta_cos",
                    "b0",
                )
            }
        n_radius = int(face_state.Er.shape[0])

        # The joint lowdot call returns a local prepared cotangent for every
        # active objective RHS.  Keeping that pytree as a scan carry creates a
        # separate carry component and scatter for every prepared leaf.  The
        # compact selector instead carries one numeric
        # ``(objective, radius, local-prepared-width)`` array and reconstructs
        # the identical pytree only after the anchor scan.  This changes no
        # NTX algebra and retains no data beyond this one rebuild pullback.
        prepared_treedef = jax.tree_util.tree_structure(support.face_prepared)
        if omit_generic_prepared_carry:
            prepared_primal_leaves = None
            prepared_bar_template_leaves = None
            prepared_local_shapes = None
            prepared_local_sizes = None
            face_prepared_flat_bar = None
        elif compact_prepared_support_carry:
            prepared_primal_leaves = tuple(jax.tree_util.tree_leaves(support.face_prepared))
            prepared_bar_template_leaves = tuple(jax.tree_util.tree_leaves(face_prepared_bar))
            prepared_local_shapes = tuple(
                jnp.asarray(leaf).shape[1:] for leaf in prepared_primal_leaves
            )
            if any(
                jnp.asarray(leaf).ndim < 1 or int(jnp.asarray(leaf).shape[0]) != n_radius
                for leaf in prepared_primal_leaves
            ):
                raise NotImplementedError(
                    "compact joint prepared-support carry requires every face_prepared "
                    "leaf to have the common leading radius axis."
                )
            prepared_local_sizes = tuple(
                int(jnp.asarray(leaf).size // n_radius) for leaf in prepared_primal_leaves
            )
            prepared_local_width = int(sum(prepared_local_sizes))
            prepared_packed_dtype = jnp.result_type(
                *(jnp.asarray(leaf).dtype for leaf in prepared_bar_template_leaves)
            )
            face_prepared_flat_bar = jnp.zeros(
                (objective_count, n_radius, prepared_local_width),
                dtype=prepared_packed_dtype,
            )
        else:
            prepared_primal_leaves = None
            prepared_bar_template_leaves = None
            prepared_local_shapes = None
            prepared_local_sizes = None
            face_prepared_flat_bar = None
        anchor_indices = self._response_anchor_indices(n_radius)
        anchor_rho = jnp.asarray(self.geometry.r_grid_half, dtype=jnp.float64)[anchor_indices]
        target_rho = jnp.asarray(support.face_channels.rho, dtype=jnp.float64)
        n_anchor = int(anchor_indices.shape[0])

        response_field_bars = jax.vmap(self._interpolated_response_field_bars)(face_response_bars)
        response_field_bar_tuple = _interpolated_response_field_bar_tuple(response_field_bars)

        if not reuse_local_vjp_primal_anchor_response:
            def _per_anchor_forward(radius_index):
                prepared = jax.tree_util.tree_map(
                    lambda arr: jax.lax.dynamic_index_in_dim(arr, radius_index, axis=0, keepdims=False),
                    support.face_prepared,
                )
                drds_value = jax.lax.dynamic_index_in_dim(
                    support.face_channels.drds, radius_index, axis=0, keepdims=False
                )
                er_value = jax.lax.dynamic_index_in_dim(face_state.Er, radius_index, axis=0, keepdims=False)
                temperature_local = jax.lax.dynamic_index_in_dim(
                    face_temperature, radius_index, axis=1, keepdims=False
                )
                density_local = jax.lax.dynamic_index_in_dim(
                    face_density, radius_index, axis=1, keepdims=False
                )
                return jax.vmap(
                    lambda species_index: self._build_interpolated_moment_response_local(
                        prepared,
                        drds_value=drds_value,
                        species_index=species_index,
                        er_value=er_value,
                        temperature_local=temperature_local,
                        density_local=density_local,
                        vthermal_local=get_v_thermal(self.species.mass, temperature_local),
                        collisionality_kind=collisionality_kind,
                    )
                )(species_indices)

            anchor_response = self._map_radius_axis_regularized_at_axis0(
                _per_anchor_forward, anchor_indices, anchor_rho
            )
            target_rho_bar = jnp.zeros((objective_count,) + target_rho.shape, dtype=target_rho.dtype)
            for anchor_field, field_bar in zip(
                (anchor_response[0], anchor_response[1], anchor_response[2], anchor_response[3]),
                response_field_bar_tuple,
                strict=True,
            ):
                target_rho_bar = target_rho_bar + jax.vmap(
                    lambda one_field_bar: self._pullback_interpolate_anchor_target_rho(
                        anchor_indices, anchor_field, target_rho, one_field_bar
                    )
                )(field_bar)
            face_channels_bar = dataclasses.replace(
                face_channels_bar,
                rho=face_channels_bar.rho + target_rho_bar,
            )
        raw_anchor_response_bar = jax.vmap(
            lambda one_field_bars: self._pullback_interpolated_anchor_response_fields(
                anchor_indices=anchor_indices,
                anchor_rho=anchor_rho,
                target_rho=target_rho,
                field_bars=one_field_bars,
            )
        )(response_field_bars)
        raw_anchor_response_fields = _interpolated_response_field_bar_tuple(raw_anchor_response_bar)
        face_density_bar = jnp.zeros((objective_count,) + face_density.shape, dtype=face_density.dtype)
        face_temperature_bar = jnp.zeros(
            (objective_count,) + face_temperature.shape, dtype=face_temperature.dtype
        )
        face_er_bar = jnp.asarray(face_response_bars.reference_er)
        anchor_positions = jnp.arange(n_anchor, dtype=jnp.int32)

        def _one_anchor(anchor_pos):
            radius_index = jax.lax.dynamic_index_in_dim(
                anchor_indices, anchor_pos, axis=0, keepdims=False
            )
            local_field_bars = tuple(
                jax.lax.dynamic_index_in_dim(field_bar, anchor_pos, axis=1, keepdims=False)
                for field_bar in raw_anchor_response_fields
            )
            prepared = jax.tree_util.tree_map(
                lambda arr: jax.lax.dynamic_index_in_dim(arr, radius_index, axis=0, keepdims=False),
                support.face_prepared,
            )
            drds_value = jax.lax.dynamic_index_in_dim(
                support.face_channels.drds, radius_index, axis=0, keepdims=False
            )
            er_value = jax.lax.dynamic_index_in_dim(face_state.Er, radius_index, axis=0, keepdims=False)
            temperature_local = jax.lax.dynamic_index_in_dim(
                face_temperature, radius_index, axis=1, keepdims=False
            )
            density_local = jax.lax.dynamic_index_in_dim(
                face_density, radius_index, axis=1, keepdims=False
            )
            if native_factorized_ntx_rhs:
                # ``local_field_bars`` is objective-major here.  Preserve
                # that entire RHS matrix through the native NTX helper: an
                # outer objective ``vmap`` would trace one scalar NTX
                # pullback per objective and defeat both the factor reuse and
                # the purpose of this separate initial-carry path.  The local
                # helper is species-major, so transpose only those two batch
                # axes; its returned state/support bars remain objective-major.
                native_local_field_bars = tuple(
                    jnp.swapaxes(field_bar, 0, 1)
                    for field_bar in local_field_bars
                )
                local_pullback = (
                    self._pullback_interpolated_moment_response_local_fields_and_prepared_support_and_drds_flat_prepared(
                        prepared,
                        drds_value=drds_value,
                        er_value=er_value,
                        temperature_local=temperature_local,
                        density_local=density_local,
                        collisionality_kind=collisionality_kind,
                        field_bars=native_local_field_bars,
                        packed_support_directional_adjoint=packed_support_directional_adjoint,
                        return_primal_response=reuse_local_vjp_primal_anchor_response,
                        native_factorized_ntx_rhs=True,
                        reuse_joint_moment_drds_jvp=reuse_joint_moment_drds_jvp,
                        return_native_vmec_coefficient_bars=(
                            return_native_vmec_coefficient_bars
                        ),
                        omit_generic_prepared_carry=omit_generic_prepared_carry,
                    )
                )
            else:
                local_pullback = jax.vmap(
                    lambda one_field_bars: self._pullback_interpolated_moment_response_local_fields_and_prepared_support_and_drds_flat_prepared(
                        prepared,
                        drds_value=drds_value,
                        er_value=er_value,
                        temperature_local=temperature_local,
                        density_local=density_local,
                        collisionality_kind=collisionality_kind,
                        field_bars=one_field_bars,
                        packed_support_directional_adjoint=packed_support_directional_adjoint,
                        return_primal_response=reuse_local_vjp_primal_anchor_response,
                    )
                )(local_field_bars)
            (
                drds_local_bar,
                er_local_bar,
                temperature_local_bar,
                density_local_bar,
                prepared_local_bar_leaves,
            ) = local_pullback[:5]
            native_vmec_local_bar = (
                local_pullback[5]
                if return_native_vmec_coefficient_bars
                else None
            )
            if reuse_local_vjp_primal_anchor_response:
                # The generic route has an outer objective axis and all
                # primal responses are identical, whereas the native route
                # above intentionally has no such axis.
                local_response = (
                    local_pullback[5]
                    if native_factorized_ntx_rhs
                    else jax.tree_util.tree_map(lambda leaf: leaf[0], local_pullback[5])
                )
            else:
                local_response = None
            # Match the established scalar state transpose: the axis anchor
            # has a regularized response representation and contributes only
            # through its explicit reference-Er channel, not through a local
            # moment/support pullback.
            is_axis_anchor = jnp.logical_and(
                jnp.asarray(n_anchor >= 4),
                jnp.logical_and(
                    jnp.asarray(anchor_pos == 0, dtype=jnp.bool_),
                    jnp.isclose(
                        jax.lax.dynamic_index_in_dim(anchor_rho, 0, axis=0, keepdims=False),
                        0.0,
                    ),
                ),
            )

            def _axis_zero(_):
                return (
                    tuple(jnp.zeros_like(leaf) for leaf in prepared_local_bar_leaves),
                    jnp.zeros_like(drds_local_bar),
                    jnp.zeros_like(er_local_bar),
                    jnp.zeros_like(temperature_local_bar),
                    jnp.zeros_like(density_local_bar),
                    (
                        jax.tree_util.tree_map(jnp.zeros_like, native_vmec_local_bar)
                        if return_native_vmec_coefficient_bars
                        else None
                    ),
                )

            def _non_axis(_):
                return (
                    prepared_local_bar_leaves,
                    drds_local_bar,
                    er_local_bar,
                    temperature_local_bar,
                    density_local_bar,
                    native_vmec_local_bar,
                )

            (
                prepared_local_bar_leaves,
                drds_local_bar,
                er_local_bar,
                temperature_local_bar,
                density_local_bar,
                native_vmec_local_bar,
            ) = jax.lax.cond(is_axis_anchor, _axis_zero, _non_axis, operand=None)
            return (
                radius_index,
                prepared_local_bar_leaves,
                drds_local_bar,
                er_local_bar,
                temperature_local_bar,
                density_local_bar,
                local_response,
                native_vmec_local_bar,
            )

        def _accumulate_anchor(carry, anchor_pos):
            (
                channels_carry,
                prepared_carry,
                density_carry,
                temperature_carry,
                er_carry,
                anchor_response_fields_carry,
                coefficient_carry,
            ) = carry
            (
                radius_index,
                prepared_local_bar_leaves,
                drds_local_bar,
                er_local_bar,
                temperature_local_bar,
                density_local_bar,
                local_response,
                native_vmec_local_bar,
            ) = _one_anchor(anchor_pos)
            if return_native_vmec_coefficient_bars:
                coefficient_carry = {
                    name: values.at[:, radius_index].add(native_vmec_local_bar[name])
                    for name, values in coefficient_carry.items()
                }
            if reuse_local_vjp_primal_anchor_response:
                anchor_response_fields_carry = tuple(
                    anchor_field.at[anchor_pos].set(local_field)
                    for anchor_field, local_field in zip(
                        anchor_response_fields_carry,
                        local_response,
                        strict=True,
                    )
                )
            return (
                dataclasses.replace(
                    channels_carry,
                    drds=channels_carry.drds.at[:, radius_index].add(drds_local_bar),
                ),
                (
                    None
                    if omit_generic_prepared_carry
                    else (
                        prepared_carry.at[:, radius_index].add(
                            jnp.concatenate(
                                tuple(
                                    jnp.reshape(local_leaf, (objective_count, -1))
                                    for local_leaf in prepared_local_bar_leaves
                                ),
                                axis=1,
                            )
                        )
                        if compact_prepared_support_carry
                        else jax.tree_util.tree_map(
                            lambda arr, local_arr: arr.at[:, radius_index].add(local_arr),
                            prepared_carry,
                            prepared_treedef.unflatten(prepared_local_bar_leaves),
                        )
                    )
                ),
                density_carry.at[:, :, radius_index].add(density_local_bar),
                temperature_carry.at[:, :, radius_index].add(temperature_local_bar),
                er_carry.at[:, radius_index].add(er_local_bar),
                anchor_response_fields_carry,
                coefficient_carry,
            ), None

        anchor_response_fields0 = tuple(
            jnp.zeros(
                (n_anchor,) + field_bar.shape[2:],
                dtype=field_bar.dtype,
            )
            for field_bar in response_field_bar_tuple
        )
        face_prepared_scan_bar = (
            None
            if omit_generic_prepared_carry
            else (
                face_prepared_flat_bar
                if compact_prepared_support_carry
                else face_prepared_bar
            )
        )

        (
            face_channels_bar,
            face_prepared_scan_bar,
            face_density_bar,
            face_temperature_bar,
            face_er_bar,
            raw_anchor_response_fields,
            native_vmec_coefficient_bar,
        ), _ = jax.lax.scan(
            _accumulate_anchor,
            (
                face_channels_bar,
                face_prepared_scan_bar,
                face_density_bar,
                face_temperature_bar,
                face_er_bar,
                anchor_response_fields0,
                native_vmec_coefficient_bar,
            ),
            anchor_positions,
        )

        if omit_generic_prepared_carry:
            face_prepared_bar = _batched_zero_like(_float_delta_tree_like(support.face_prepared))
        elif compact_prepared_support_carry:
            face_prepared_bar = face_prepared_scan_bar
            prepared_bar_leaves = []
            offset = 0
            for bar_template_leaf, local_shape, local_size in zip(
                prepared_bar_template_leaves,
                prepared_local_shapes,
                prepared_local_sizes,
                strict=True,
            ):
                prepared_bar_leaves.append(
                    jnp.reshape(
                        face_prepared_bar[:, :, offset : offset + local_size],
                        (objective_count, n_radius) + local_shape,
                    ).astype(jnp.asarray(bar_template_leaf).dtype)
                )
                offset += local_size
            face_prepared_bar = prepared_treedef.unflatten(prepared_bar_leaves)
        else:
            face_prepared_bar = face_prepared_scan_bar

        if reuse_local_vjp_primal_anchor_response:
            if n_anchor < 4:
                anchor_response_fields = raw_anchor_response_fields
            else:
                anchor_response_fields = jax.lax.cond(
                    jnp.isclose(anchor_rho[0], 0.0),
                    lambda _: self._regularize_axis_radius0(raw_anchor_response_fields, anchor_rho),
                    lambda _: raw_anchor_response_fields,
                    operand=None,
                )
            target_rho_bar = jnp.zeros(
                (objective_count,) + target_rho.shape,
                dtype=target_rho.dtype,
            )
            for anchor_field, field_bar in zip(
                anchor_response_fields,
                response_field_bar_tuple,
                strict=True,
            ):
                target_rho_bar = target_rho_bar + jax.vmap(
                    lambda one_field_bar: self._pullback_interpolate_anchor_target_rho(
                        anchor_indices, anchor_field, target_rho, one_field_bar
                    )
                )(field_bar)
            face_channels_bar = dataclasses.replace(
                face_channels_bar,
                rho=face_channels_bar.rho + target_rho_bar,
            )

        def _face_state_values(density_value, pressure_value, er_value):
            rebuilt_face_state = build_face_transport_state(
                dataclasses.replace(state, density=density_value, pressure=pressure_value, Er=er_value),
                self.geometry,
                bc_density=self.bc_density,
                bc_temperature=self.bc_temperature,
                density_floor=self.density_floor,
                temperature_floor=self.temperature_floor,
            )
            return (
                safe_density(rebuilt_face_state.density, self.density_floor),
                rebuilt_face_state.temperature,
                rebuilt_face_state.Er,
            )

        _, face_state_pullback = jax.vjp(
            _face_state_values, state.density, state.pressure, state.Er
        )
        density_bar, pressure_bar, er_bar = jax.vmap(face_state_pullback)(
            (face_density_bar, face_temperature_bar, face_er_bar)
        )
        state_bar = dataclasses.replace(
            _batched_zero_like(state),
            density=density_bar,
            pressure=pressure_bar,
            Er=er_bar,
        )
        support_bar = dataclasses.replace(
            _batched_zero_like(_float_delta_tree_like(support)),
            face_channels=face_channels_bar,
            face_prepared=face_prepared_bar,
        )
        if return_native_vmec_coefficient_bars:
            return state_bar, support_bar, native_vmec_coefficient_bar
        return state_bar, support_bar

    def pullback_build_lagged_response_state_and_support_payload_batched_interpolated_faces_reuse_local_vjp_primal(
        self,
        state,
        lagged_response_bars,
        support,
        **kwargs,
    ):
        """Joint face pullback that reuses each local VJP primal for interpolation."""
        return self.pullback_build_lagged_response_state_and_support_payload_batched_interpolated_faces(
            state,
            lagged_response_bars,
            support,
            reuse_local_vjp_primal_anchor_response=True,
            **kwargs,
        )

    def pullback_build_lagged_response_state_and_support_payload_batched_interpolated_faces_reuse_local_vjp_primal_compact_prepared_carry(
        self,
        state,
        lagged_response_bars,
        support,
        **kwargs,
    ):
        """Joint lowdot pullback with a single numeric prepared-bar carry."""
        return self.pullback_build_lagged_response_state_and_support_payload_batched_interpolated_faces(
            state,
            lagged_response_bars,
            support,
            reuse_local_vjp_primal_anchor_response=True,
            compact_prepared_support_carry=True,
            **kwargs,
        )

    def pullback_build_lagged_response_state_and_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal(
        self,
        state,
        lagged_response_bars,
        support,
        **kwargs,
    ):
        """Native matrix-RHS joint state/support transpose for initial carry.

        This is intentionally separate from the established generic joint
        selectors.  It reuses the native support helper's returned case bars
        to form state bars, so it does not request a second NTX solve.
        """

        return self.pullback_build_lagged_response_state_and_support_payload_batched_interpolated_faces(
            state,
            lagged_response_bars,
            support,
            native_factorized_ntx_rhs=True,
            reuse_joint_moment_drds_jvp=True,
            **kwargs,
        )

    def pullback_build_lagged_response_state_and_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients(
        self,
        state,
        lagged_response_bars,
        support,
        **kwargs,
    ):
        """Native joint NTX transpose with compact VMEC coefficient output.

        Direct realtime-geometry terms are intentionally not part of this
        helper.  The caller may evaluate that outer transpose independently,
        without staging it inside the native NTX matrix-RHS graph.
        """
        kwargs.pop("return_native_vmec_coefficient_bars", None)
        return self.pullback_build_lagged_response_state_and_support_payload_batched_interpolated_faces(
            state,
            lagged_response_bars,
            support,
            native_factorized_ntx_rhs=True,
            reuse_joint_moment_drds_jvp=True,
            return_native_vmec_coefficient_bars=True,
            **kwargs,
        )

    def pullback_build_lagged_response_state_and_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_no_prepared_carry(
        self,
        state,
        lagged_response_bars,
        support,
        **kwargs,
    ):
        """Opt-in native joint transpose without a zero prepared scan carry."""
        kwargs.pop("return_native_vmec_coefficient_bars", None)
        return self.pullback_build_lagged_response_state_and_support_payload_batched_interpolated_faces(
            state,
            lagged_response_bars,
            support,
            native_factorized_ntx_rhs=True,
            reuse_joint_moment_drds_jvp=True,
            return_native_vmec_coefficient_bars=True,
            omit_generic_prepared_carry=True,
            **kwargs,
        )

    def _evaluate_full_state_quadratic_axis_response(self, state, response, *, axis):
        """Evaluate the cached full-state quadratic model on centres or faces."""
        if axis not in {"center", "face"}:
            raise ValueError("axis must be 'center' or 'face'.")
        delta = dataclasses.replace(
            state,
            density=state.density - response.reference_state.density,
            pressure=state.pressure - response.reference_state.pressure,
            Er=state.Er - response.reference_state.Er,
        )
        evaluated = _build_evaluated_transport_state_directional(
            response.reference_state,
            delta,
            self.geometry,
            bc_density=self.bc_density,
            bc_temperature=self.bc_temperature,
            density_floor=self.density_floor,
            temperature_floor=self.temperature_floor,
        )
        axis_state = getattr(evaluated, axis)
        density_gradient = getattr(evaluated, f"density_grad_{axis}")
        temperature_gradient = getattr(evaluated, f"temperature_grad_{axis}")
        support, coefficients = self._static_support(), response.coefficient_response
        channels = getattr(support, f"{axis}_channels")
        radius_coordinates = (
            self.geometry.r_grid if axis == "center" else self.geometry.r_grid_half
        )
        vthermal0 = get_v_thermal(self.species.mass, axis_state.temperature.value)
        radii = jnp.arange(axis_state.Er.value.shape[0], dtype=jnp.int32)
        species_indices = jnp.arange(int(self.species.number_species), dtype=jnp.int32)

        def per_radius(radius):
            n = _jet_dynamic_index(axis_state.density, radius, axis=1)
            t = _jet_dynamic_index(axis_state.temperature, radius, axis=1)
            er = _jet_dynamic_index(axis_state.Er, radius)
            dn = _jet_dynamic_index(density_gradient, radius, axis=1)
            dt = _jet_dynamic_index(temperature_gradient, radius, axis=1)
            v0 = jax.lax.dynamic_index_in_dim(vthermal0, radius, axis=1, keepdims=False)
            drds = jax.lax.dynamic_index_in_dim(channels.drds, radius, axis=0, keepdims=False)

            def per_species(a):
                nu, ep, vth = _local_scan_inputs_directional_default(self.energy_grid, self.species, drds_value=drds, species_index=a, er_value=er, temperature_local=t, density_local=n, reference_vthermal_local=v0, er_v_floor=self.er_v_floor)
                fields = jax.tree_util.tree_map(lambda x: jax.lax.dynamic_index_in_dim(jax.lax.dynamic_index_in_dim(x, radius, axis=0, keepdims=False), a, axis=0, keepdims=False), coefficients)
                c = compose_ntx_coefficient_quadratic(
                    fields.reference_coefficients,
                    fields.dcoefficients_d_nu_hat,
                    fields.dcoefficients_d_epsi_hat,
                    fields.d2coefficients_d_nu_hat2,
                    fields.d2coefficients_d_nu_hat_d_epsi_hat,
                    fields.d2coefficients_d_epsi_hat2,
                    nu,
                    ep,
                )
                m = _transport_moments_from_coefficient_scan_directional(
                    self.energy_grid, c, drds_value=drds
                )
                lij = _lij_from_transport_moments_directional(self.species, m, species_index=a, vth_a=vth)
                return _assemble_face_fluxes_from_lij_directional_local(charge=self.species.charge[a], density=_jet_select_axis(n,a), temperature=_jet_select_axis(t,a), density_gradient=_jet_select_axis(dn,a), temperature_gradient=_jet_select_axis(dt,a), er=er, lij=lij)
            return jax.vmap(per_species)(species_indices)
        gamma, q, upar = self._map_radius_axis_regularized_at_axis0(
            per_radius, radii, radius_coordinates
        )
        # ``per_radius`` maps radius first.  The transport interface, like the
        # ordinary Lij evaluator, is species-first: (species, face).
        gamma, q, upar = (
            jax.tree_util.tree_map(lambda array: jnp.swapaxes(array, 0, 1), value)
            for value in (gamma, q, upar)
        )
        suffix = "" if axis == "center" else "_faces"
        return {
            f"Gamma{suffix}": _jet_evaluate(gamma),
            f"Q{suffix}": _jet_evaluate(q),
            f"Upar{suffix}": _jet_evaluate(upar),
        }

    def _evaluate_full_state_quadratic_face_response(self, state, response):
        return self._evaluate_full_state_quadratic_axis_response(
            state, response, axis="face"
        )

    def _evaluate_full_state_quadratic_center_response(self, state, response):
        return self._evaluate_full_state_quadratic_axis_response(
            state, response, axis="center"
        )

    def evaluate_full_state_quadratic_face_tangent_with_lagged_response(
        self,
        state: TransportState,
        state_direction: TransportState,
        lagged_response: NTXExactLijLaggedResponse,
    ) -> dict[str, jax.Array]:
        """Exact directional tangent of the cached quadratic NTX response.

        The full-state response is an explicit quadratic polynomial in the
        transport-state displacement.  Its centred polarization is therefore
        *exact*, not a finite-difference approximation:

        ``(F_Q(y + v) - F_Q(y - v)) / 2 = J_Q(y) v``.

        Each side uses the existing written second-order jet composition and
        cached NTX coefficient Hessian; no generic JVP and no NTX rebuild are
        involved.  Face-interpolated mode returns faces; direct-centre mode
        returns its complete cached centre-and-face representation. The
        equation-level mixed tangent consumes this primitive when constructing
        the opt-in stage Newton operator.
        """
        plus_state = dataclasses.replace(
            state,
            density=state.density + state_direction.density,
            pressure=state.pressure + state_direction.pressure,
            Er=state.Er + state_direction.Er,
        )
        minus_state = dataclasses.replace(
            state,
            density=state.density - state_direction.density,
            pressure=state.pressure - state_direction.pressure,
            Er=state.Er - state_direction.Er,
        )
        def _polarized(evaluate, response):
            if not isinstance(response, NTXFullStateQuadraticPreparedCoefficientResponse):
                raise NotImplementedError(
                    "Quadratic tangents require the full_state_quadratic_response payload."
                )
            plus = evaluate(plus_state, response)
            minus = evaluate(minus_state, response)
            return jax.tree_util.tree_map(
                lambda left, right: 0.5 * (left - right), plus, minus
            )

        if self._resolved_center_response_mode() in {
            "center_local_response",
            "interpolate_face_coefficients",
            "interpolate_face_coefficients_cubic",
            "interpolate_face_coefficients_physical_coordinates",
            "interpolate_face_coefficients_native_distance",
            "interpolate_face_coefficients_taylor_reliability",
        }:
            # Direct-centre mode owns two cached models: centres for local
            # terms and faces for conservative divergence.  Its tangent must
            # preserve the same complete representation as its value path.
            centre = _polarized(
                self._evaluate_full_state_quadratic_center_response,
                lagged_response.center_response,
            )
            faces = _polarized(
                self._evaluate_full_state_quadratic_face_response,
                lagged_response.face_response,
            )
            return {**centre, **faces}
        return _polarized(
            self._evaluate_full_state_quadratic_face_response,
            lagged_response.face_response,
        )

    def evaluate_with_lagged_response_tangent(
        self,
        state: TransportState,
        state_direction: TransportState,
        lagged_response: NTXExactLijLaggedResponse,
        **kwargs,
    ) -> dict[str, jax.Array]:
        """Custom tangent hook used only by the opt-in quadratic Newton path."""
        del kwargs
        if isinstance(
            lagged_response.face_response,
            NTXFullStateQuadraticPreparedCoefficientResponse,
        ) or isinstance(
            lagged_response.center_response,
            NTXFullStateQuadraticPreparedCoefficientResponse,
        ):
            return self.evaluate_full_state_quadratic_face_tangent_with_lagged_response(
                state, state_direction, lagged_response
            )
        return super().evaluate_with_lagged_response_tangent(
            state, state_direction, lagged_response
        )

    def evaluate_with_lagged_response(self, state, lagged_response, **kwargs):
        del kwargs
        if isinstance(
            lagged_response.face_response,
            NTXFullStateQuadraticPreparedCoefficientResponse,
        ) or isinstance(
            lagged_response.center_response,
            NTXFullStateQuadraticPreparedCoefficientResponse,
        ):
            if self._resolved_center_response_mode() in {
                "center_local_response",
                "interpolate_face_coefficients",
                "interpolate_face_coefficients_cubic",
                "interpolate_face_coefficients_physical_coordinates",
                "interpolate_face_coefficients_native_distance",
                "interpolate_face_coefficients_taylor_reliability",
            }:
                centre_fluxes = self._evaluate_full_state_quadratic_center_response(
                    state, lagged_response.center_response
                )
                face_fluxes = self._evaluate_full_state_quadratic_face_response(
                    state, lagged_response.face_response
                )
                return {**centre_fluxes, **face_fluxes}
            return self._evaluate_full_state_quadratic_face_response(
                state, lagged_response.face_response
            )
        evaluated = build_evaluated_transport_state(
            state,
            self.geometry,
            bc_density=self.bc_density,
            bc_temperature=self.bc_temperature,
            density_floor=self.density_floor,
            temperature_floor=self.temperature_floor,
        )
        density = evaluated.center.density
        temperature = evaluated.center.temperature

        support = self._static_support()
        collisionality_kind = _collisionality_kind(self.collisionality_model)
        v_thermal = get_v_thermal(self.species.mass, temperature)
        species_indices = jnp.arange(int(self.species.number_species), dtype=jnp.int32)

        def _lij_from_axis_response(
            response,
            *,
            channels,
            prepared_all,
            radius_coordinates,
            Er_axis,
            temperature_axis,
            density_axis,
            v_thermal_axis,
        ):
            if isinstance(response, NTXInterpolatedMomentResponse):
                radius_indices_axis = jnp.arange(Er_axis.shape[0], dtype=jnp.int32)

                def _current_log_nu_star_per_radius(radius_index):
                    drds_value = jax.lax.dynamic_index_in_dim(channels.drds, radius_index, axis=0, keepdims=False)
                    er_value = jax.lax.dynamic_index_in_dim(Er_axis, radius_index, axis=0, keepdims=False)
                    temperature_local = jax.lax.dynamic_index_in_dim(temperature_axis, radius_index, axis=1, keepdims=False)
                    density_local = jax.lax.dynamic_index_in_dim(density_axis, radius_index, axis=1, keepdims=False)
                    vthermal_local = jax.lax.dynamic_index_in_dim(v_thermal_axis, radius_index, axis=1, keepdims=False)
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
                        radius_indices_axis,
                        radius_coordinates,
                        unbatched=True,
                    ),
                    0,
                    1,
                )
                delta_er = Er_axis - response.reference_er
                delta_log_nu_star = current_log_nu_star - response.reference_log_nu_star
                transport_moments = (
                    response.reference_transport_moments
                    + response.dtransport_moments_d_er * delta_er[None, :, None]
                    + response.dtransport_moments_d_log_nu_star * delta_log_nu_star[:, :, None]
                )
                lij_axis = self._batched_lij_from_transport_moments(transport_moments, v_thermal_axis)
                _debug_arrays_if_any_nonfinite(
                    "ntx.evaluate_lagged.interpolated_axis",
                    (
                        ("Er_axis", Er_axis),
                        ("temperature_axis", temperature_axis),
                        ("density_axis", density_axis),
                        ("v_thermal_axis", v_thermal_axis),
                        ("current_log_nu_star", current_log_nu_star),
                        ("reference_log_nu_star", response.reference_log_nu_star),
                        ("delta_er", delta_er),
                        ("delta_log_nu_star", delta_log_nu_star),
                        ("reference_transport_moments", response.reference_transport_moments),
                        ("dtransport_moments_d_er", response.dtransport_moments_d_er),
                        ("dtransport_moments_d_log_nu_star", response.dtransport_moments_d_log_nu_star),
                        ("transport_moments", transport_moments),
                        ("lij_axis", lij_axis),
                    ),
                )
                return lij_axis

            if isinstance(response, NTXQuadraticPreparedCoefficientResponse):
                radius_indices_axis = jnp.arange(Er_axis.shape[0], dtype=jnp.int32)

                def _quadratic_transport_moments_per_radius(radius_index):
                    drds_value = jax.lax.dynamic_index_in_dim(
                        channels.drds, radius_index, axis=0, keepdims=False
                    )
                    er_value = jax.lax.dynamic_index_in_dim(
                        Er_axis, radius_index, axis=0, keepdims=False
                    )
                    temperature_local = jax.lax.dynamic_index_in_dim(
                        temperature_axis, radius_index, axis=1, keepdims=False
                    )
                    density_local = jax.lax.dynamic_index_in_dim(
                        density_axis, radius_index, axis=1, keepdims=False
                    )
                    vthermal_local = jax.lax.dynamic_index_in_dim(
                        v_thermal_axis, radius_index, axis=1, keepdims=False
                    )
                    current_nu_hat, current_epsi_hat = jax.vmap(
                        lambda species_index: self._local_scan_inputs(
                            drds_value=drds_value,
                            species_index=species_index,
                            er_value=er_value,
                            temperature_local=temperature_local,
                            density_local=density_local,
                            vthermal_local=vthermal_local,
                            collisionality_kind=collisionality_kind,
                        )[:2]
                    )(species_indices)
                    reference_nu_hat = jax.lax.dynamic_index_in_dim(
                        response.reference_nu_hat, radius_index, axis=0, keepdims=False
                    )
                    reference_epsi_hat = jax.lax.dynamic_index_in_dim(
                        response.reference_epsi_hat, radius_index, axis=0, keepdims=False
                    )
                    delta_nu_hat = current_nu_hat - reference_nu_hat
                    delta_epsi_hat = current_epsi_hat - reference_epsi_hat
                    coefficients = (
                        jax.lax.dynamic_index_in_dim(
                            response.reference_coefficients, radius_index, axis=0, keepdims=False
                        )
                        + jax.lax.dynamic_index_in_dim(
                            response.dcoefficients_d_nu_hat, radius_index, axis=0, keepdims=False
                        ) * delta_nu_hat[..., None]
                        + jax.lax.dynamic_index_in_dim(
                            response.dcoefficients_d_epsi_hat, radius_index, axis=0, keepdims=False
                        ) * delta_epsi_hat[..., None]
                        + 0.5
                        * (
                            jax.lax.dynamic_index_in_dim(
                                response.d2coefficients_d_nu_hat2, radius_index, axis=0, keepdims=False
                            ) * delta_nu_hat[..., None] ** 2
                            + 2.0
                            * jax.lax.dynamic_index_in_dim(
                                response.d2coefficients_d_nu_hat_d_epsi_hat,
                                radius_index,
                                axis=0,
                                keepdims=False,
                            )
                            * delta_nu_hat[..., None]
                            * delta_epsi_hat[..., None]
                            + jax.lax.dynamic_index_in_dim(
                                response.d2coefficients_d_epsi_hat2,
                                radius_index,
                                axis=0,
                                keepdims=False,
                            ) * delta_epsi_hat[..., None] ** 2
                        )
                    )
                    return jax.vmap(
                        lambda coefficient_scan: self._transport_moments_from_coefficient_scan(
                            coefficient_scan,
                            drds_value=drds_value,
                        )
                    )(coefficients)

                transport_moments = jnp.swapaxes(
                    self._map_radius_axis_regularized_at_axis0(
                        _quadratic_transport_moments_per_radius,
                        radius_indices_axis,
                        radius_coordinates,
                    ),
                    0,
                    1,
                )
                return self._batched_lij_from_transport_moments(
                    transport_moments,
                    v_thermal_axis,
                )

            radius_indices_axis = jnp.arange(Er_axis.shape[0], dtype=jnp.int32)

            def _transport_moment_tangent_per_radius(radius_index):
                prepared = jax.tree_util.tree_map(
                    lambda arr: jax.lax.dynamic_index_in_dim(arr, radius_index, axis=0, keepdims=False),
                    prepared_all,
                )
                drds_value = jax.lax.dynamic_index_in_dim(channels.drds, radius_index, axis=0, keepdims=False)
                er_value = jax.lax.dynamic_index_in_dim(Er_axis, radius_index, axis=0, keepdims=False)
                temperature_local = jax.lax.dynamic_index_in_dim(temperature_axis, radius_index, axis=1, keepdims=False)
                density_local = jax.lax.dynamic_index_in_dim(density_axis, radius_index, axis=1, keepdims=False)
                vthermal_local = jax.lax.dynamic_index_in_dim(v_thermal_axis, radius_index, axis=1, keepdims=False)
                ref_nu_radius = jax.lax.dynamic_index_in_dim(response.reference_nu_hat, radius_index, axis=0, keepdims=False)
                ref_epsi_radius = jax.lax.dynamic_index_in_dim(response.reference_epsi_hat, radius_index, axis=0, keepdims=False)
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
                radius_indices_axis,
                radius_coordinates,
            )
            transport_moments = response.reference_transport_moments + transport_moment_tangent_by_radius
            transport_moments = jnp.swapaxes(transport_moments, 0, 1)
            lij_axis = self._batched_lij_from_transport_moments(transport_moments, v_thermal_axis)
            _debug_arrays_if_any_nonfinite(
                "ntx.evaluate_lagged.direct_axis",
                (
                    ("Er_axis", Er_axis),
                    ("temperature_axis", temperature_axis),
                    ("density_axis", density_axis),
                    ("v_thermal_axis", v_thermal_axis),
                    ("reference_transport_moments", response.reference_transport_moments),
                    ("transport_moment_tangent_by_radius", transport_moment_tangent_by_radius),
                    ("transport_moments", transport_moments),
                    ("lij_axis", lij_axis),
                ),
            )
            return lij_axis

        def _assemble_face_fluxes_from_lij(face_state, face_density, lij_faces):
            dndr_faces = evaluated.density_grad_face
            dTdr_faces = evaluated.temperature_grad_face
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
            gamma_faces = -density_phys * (
                lij_faces[:, :, 0, 0] * a1
                + lij_faces[:, :, 0, 1] * a2
                + lij_faces[:, :, 0, 2] * a3[None, :]
            )
            q_faces = -temperature_phys * density_phys * (
                lij_faces[:, :, 1, 0] * a1
                + lij_faces[:, :, 1, 1] * a2
                + lij_faces[:, :, 1, 2] * a3[None, :]
            )
            upar_faces = -density_phys * (
                lij_faces[:, :, 2, 0] * a1
                + lij_faces[:, :, 2, 1] * a2
                + lij_faces[:, :, 2, 2] * a3[None, :]
            )
            _debug_arrays_if_any_nonfinite(
                "ntx.evaluate_lagged.face_flux_assembly",
                (
                    ("state_density_center", density),
                    ("state_temperature_center", temperature),
                    ("face_density", face_density),
                    ("face_temperature", face_state.temperature),
                    ("face_Er", face_state.Er),
                    ("dndr_faces", dndr_faces),
                    ("dTdr_faces", dTdr_faces),
                    ("A1", a1),
                    ("A2", a2),
                    ("A3", a3),
                    ("density_phys", density_phys),
                    ("temperature_phys", temperature_phys),
                    ("lij_faces", lij_faces),
                    ("Gamma_faces", gamma_faces),
                    ("Q_faces", q_faces),
                    ("Upar_faces", upar_faces),
                    ("Gamma_center_from_faces", jax.vmap(cell_centered_from_faces)(gamma_faces)),
                    ("Q_center_from_faces", jax.vmap(cell_centered_from_faces)(q_faces)),
                    ("Upar_center_from_faces", jax.vmap(cell_centered_from_faces)(upar_faces)),
                ),
            )
            return gamma_faces, q_faces, upar_faces

        face_state = evaluated.face
        face_density = evaluated.face.density
        face_temperature = face_state.temperature
        face_v_thermal = get_v_thermal(self.species.mass, face_temperature)
        _debug_arrays_if_any_nonfinite(
            "ntx.evaluate_lagged.input_state",
            (
                ("density_center", density),
                ("temperature_center", temperature),
                ("Er_center", state.Er),
                ("density_face", face_density),
                ("temperature_face", face_temperature),
                ("Er_face", face_state.Er),
                ("v_thermal_face", face_v_thermal),
            ),
        )
        face_response = lagged_response.face_response
        _debug_lagged_response_if_nonfinite("ntx.evaluate_lagged.face_response", face_response)
        if face_response is None:
            face_fluxes = self.evaluate_face_fluxes(state, face_state, face_response_mode="face_local_response")
        else:
            lij_faces = _lij_from_axis_response(
                face_response,
                channels=support.face_channels,
                prepared_all=support.face_prepared,
                radius_coordinates=self.geometry.r_grid_half,
                Er_axis=face_state.Er,
                temperature_axis=face_temperature,
                density_axis=face_density,
                v_thermal_axis=face_v_thermal,
            )
            gamma_faces, q_faces, upar_faces = _assemble_face_fluxes_from_lij(face_state, face_density, lij_faces)
            face_fluxes = {"Gamma": gamma_faces, "Q": q_faces, "Upar": upar_faces}
        _debug_arrays_if_any_nonfinite(
            "ntx.evaluate_lagged.face_flux_output",
            (
                ("Gamma_faces", face_fluxes.get("Gamma")),
                ("Q_faces", face_fluxes.get("Q")),
                ("Upar_faces", face_fluxes.get("Upar")),
                (
                    "Gamma_center_from_faces",
                    None
                    if face_fluxes.get("Gamma", None) is None
                    else jax.vmap(cell_centered_from_faces)(face_fluxes["Gamma"]),
                ),
                (
                    "Q_center_from_faces",
                    None
                    if face_fluxes.get("Q", None) is None
                    else jax.vmap(cell_centered_from_faces)(face_fluxes["Q"]),
                ),
                (
                    "Upar_center_from_faces",
                    None
                    if face_fluxes.get("Upar", None) is None
                    else jax.vmap(cell_centered_from_faces)(face_fluxes["Upar"]),
                ),
            ),
        )

        center_response = lagged_response.center_response
        _debug_lagged_response_if_nonfinite("ntx.evaluate_lagged.center_response", center_response)
        if center_response is None or self._resolved_center_response_mode() == "interpolate_from_faces":
            return {
                "Gamma_faces": face_fluxes["Gamma"],
                "Q_faces": face_fluxes["Q"],
                "Upar_faces": face_fluxes["Upar"],
            }

        if isinstance(center_response, NTXInterpolatedMomentResponse):
            if lagged_timing_enabled():
                jax.debug.callback(lambda: lagged_timing_start("ntx.evaluate_with_lagged_response.coarse"), ordered=True)
            lij = _lij_from_axis_response(
                center_response,
                channels=support.center_channels,
                prepared_all=support.center_prepared,
                radius_coordinates=self.geometry.r_grid,
                Er_axis=state.Er,
                temperature_axis=temperature,
                density_axis=density,
                v_thermal_axis=v_thermal,
            )
            gamma, q, upar = self._assemble_center_fluxes(
                state.Er,
                temperature,
                density,
                evaluated.density_grad_center,
                evaluated.temperature_grad_center,
                lij,
            )
            gamma, q, upar = self._regularize_center_fluxes_axis0(gamma, q, upar)
            if lagged_timing_enabled():
                jax.debug.callback(lambda: lagged_timing_end("ntx.evaluate_with_lagged_response.coarse"), ordered=True)
            return {
                "Gamma": gamma,
                "Q": q,
                "Upar": upar,
                "Gamma_faces": face_fluxes["Gamma"],
                "Q_faces": face_fluxes["Q"],
                "Upar_faces": face_fluxes["Upar"],
            }

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
            evaluated.density_grad_center,
            evaluated.temperature_grad_center,
            lij,
        )
        gamma, q, upar = self._regularize_center_fluxes_axis0(gamma, q, upar)
        return {
            "Gamma": gamma,
            "Q": q,
            "Upar": upar,
            "Gamma_faces": face_fluxes["Gamma"],
            "Q_faces": face_fluxes["Q"],
            "Upar_faces": face_fluxes["Upar"],
        }

    def pullback_evaluate_with_lagged_response(self, state, lagged_response, flux_bar, **kwargs):
        del kwargs
        face_response = lagged_response.face_response
        center_response = lagged_response.center_response

        def _compact_interpolated_face_response_pullback(response):
            evaluated = build_evaluated_transport_state(
                state,
                self.geometry,
                bc_density=self.bc_density,
                bc_temperature=self.bc_temperature,
                density_floor=self.density_floor,
                temperature_floor=self.temperature_floor,
            )
            face_state = evaluated.face
            face_density = evaluated.face.density
            face_temperature = face_state.temperature

            support = self._static_support()
            collisionality_kind = _collisionality_kind(self.collisionality_model)
            face_v_thermal = get_v_thermal(self.species.mass, face_temperature)
            species_indices = jnp.arange(int(self.species.number_species), dtype=jnp.int32)
            radius_indices = jnp.arange(face_state.Er.shape[0], dtype=jnp.int32)

            def _current_log_nu_star_per_radius(radius_index):
                drds_value = jax.lax.dynamic_index_in_dim(
                    support.face_channels.drds,
                    radius_index,
                    axis=0,
                    keepdims=False,
                )
                er_value = jax.lax.dynamic_index_in_dim(face_state.Er, radius_index, axis=0, keepdims=False)
                temperature_local = jax.lax.dynamic_index_in_dim(
                    face_temperature,
                    radius_index,
                    axis=1,
                    keepdims=False,
                )
                density_local = jax.lax.dynamic_index_in_dim(
                    face_density,
                    radius_index,
                    axis=1,
                    keepdims=False,
                )
                vthermal_local = jax.lax.dynamic_index_in_dim(
                    face_v_thermal,
                    radius_index,
                    axis=1,
                    keepdims=False,
                )
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
                    self.geometry.r_grid_half,
                    unbatched=True,
                ),
                0,
                1,
            )
            delta_er = face_state.Er - response.reference_er
            delta_log_nu_star = current_log_nu_star - response.reference_log_nu_star
            transport_moments = (
                response.reference_transport_moments
                + response.dtransport_moments_d_er * delta_er[None, :, None]
                + response.dtransport_moments_d_log_nu_star * delta_log_nu_star[:, :, None]
            )
            lij_faces = self._batched_lij_from_transport_moments(transport_moments, face_v_thermal)

            dndr_faces = evaluated.density_grad_face
            dTdr_faces = evaluated.temperature_grad_face
            a1 = jax.vmap(
                lambda charge, density_a, temperature_a, dndr_a, dTdr_a: get_Thermodynamical_Forces_A1(
                    charge,
                    density_a,
                    temperature_a,
                    dndr_a,
                    dTdr_a,
                    face_state.Er,
                )
            )(self.species.charge, face_density, face_temperature, dndr_faces, dTdr_faces)
            a2 = jax.vmap(get_Thermodynamical_Forces_A2)(face_temperature, dTdr_faces)
            a3 = get_Thermodynamical_Forces_A3(face_state.Er)
            density_phys = DENSITY_STATE_TO_PHYSICAL * face_density
            temperature_phys = TEMPERATURE_STATE_TO_PHYSICAL * face_temperature
            gamma_faces = -density_phys * (
                lij_faces[:, :, 0, 0] * a1
                + lij_faces[:, :, 0, 1] * a2
                + lij_faces[:, :, 0, 2] * a3[None, :]
            )
            q_faces = -temperature_phys * density_phys * (
                lij_faces[:, :, 1, 0] * a1
                + lij_faces[:, :, 1, 1] * a2
                + lij_faces[:, :, 1, 2] * a3[None, :]
            )
            upar_faces = -density_phys * (
                lij_faces[:, :, 2, 0] * a1
                + lij_faces[:, :, 2, 1] * a2
                + lij_faces[:, :, 2, 2] * a3[None, :]
            )
            face_output = {
                "Gamma_faces": gamma_faces,
                "Q_faces": q_faces,
                "Upar_faces": upar_faces,
            }
            face_flux_bar = (
                _face_flux_bar_with_interpolated_center_bars(face_output, flux_bar)
                if center_response is None or self._resolved_center_response_mode() == "interpolate_from_faces"
                else flux_bar
            )
            face_flux_bar = _complete_flux_bar_like(
                face_output,
                face_flux_bar,
                context="NTXExactLijRuntimeTransportModel.response.face_compact",
            )
            gamma_bar = face_flux_bar["Gamma_faces"]
            q_bar = face_flux_bar["Q_faces"]
            upar_bar = face_flux_bar["Upar_faces"]

            lij_bar = jnp.zeros_like(lij_faces)
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
            l11_fac = -inv_sqrt_pi * (mass / charge) ** 2 * face_v_thermal**3
            l13_fac = -inv_sqrt_pi * (mass / charge) * face_v_thermal**2
            l33_fac = -inv_sqrt_pi * face_v_thermal
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

            return NTXInterpolatedMomentResponse(
                reference_er=-jnp.sum(
                    transport_moments_bar * response.dtransport_moments_d_er,
                    axis=(0, 2),
                ),
                reference_log_nu_star=-jnp.sum(
                    transport_moments_bar * response.dtransport_moments_d_log_nu_star,
                    axis=2,
                ),
                reference_transport_moments=transport_moments_bar,
                dtransport_moments_d_er=transport_moments_bar * delta_er[None, :, None],
                dtransport_moments_d_log_nu_star=transport_moments_bar * delta_log_nu_star[:, :, None],
            )

        has_face_bar = any(
            key in flux_bar and flux_bar.get(key, None) is not None
            for key in ("Gamma_faces", "Q_faces", "Upar_faces")
        )
        face_response_bar_acc = None
        if face_response is not None and (
            has_face_bar
            or center_response is None
            or self._resolved_center_response_mode() == "interpolate_from_faces"
        ):
            if isinstance(face_response, NTXInterpolatedMomentResponse):
                face_response_bar_acc = _compact_interpolated_face_response_pullback(face_response)
            else:
                face_output, pullback = jax.vjp(
                    lambda response_value: self.evaluate_with_lagged_response(
                        state,
                        NTXExactLijLaggedResponse(
                            face_response=response_value,
                            center_response=center_response,
                        ),
                    ),
                    face_response,
                )
                face_flux_bar = (
                    _face_flux_bar_with_interpolated_center_bars(face_output, flux_bar)
                    if center_response is None or self._resolved_center_response_mode() == "interpolate_from_faces"
                    else flux_bar
                )
                face_flux_bar = _complete_flux_bar_like(
                    face_output,
                    face_flux_bar,
                    context="NTXExactLijRuntimeTransportModel.response.face",
                )
                (face_response_bar_acc,) = pullback(face_flux_bar)
            if center_response is None or self._resolved_center_response_mode() == "interpolate_from_faces":
                return NTXExactLijLaggedResponse(face_response=face_response_bar_acc)

        if isinstance(center_response, NTXInterpolatedMomentResponse):
            evaluated = build_evaluated_transport_state(
                state,
                self.geometry,
                bc_density=self.bc_density,
                bc_temperature=self.bc_temperature,
                density_floor=self.density_floor,
                temperature_floor=self.temperature_floor,
            )
            density = evaluated.center.density
            temperature = evaluated.center.temperature
            n_species = int(temperature.shape[0])

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
                evaluated.density_grad_center,
                evaluated.temperature_grad_center,
                lij,
            )
            center_flux_bar = _complete_flux_bar_like(
                {"Gamma": gamma, "Q": q, "Upar": upar},
                flux_bar,
                context="NTXExactLijRuntimeTransportModel.response.center_interpolated",
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
            )((center_flux_bar["Gamma"], center_flux_bar["Q"], center_flux_bar["Upar"]))

            dndr_all = evaluated.density_grad_center
            dTdr_all = evaluated.temperature_grad_center
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
                face_response=face_response_bar_acc,
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

            output, pb = jax.vjp(
                _reduced_response,
                center_response.reference_transport_moments,
                center_response.reference_nu_hat,
                center_response.reference_epsi_hat,
            )
            (
                reference_transport_moments_bar,
                reference_nu_hat_bar,
                reference_epsi_hat_bar,
            ) = pb(
                _complete_flux_bar_like(
                    output,
                    flux_bar,
                    context="NTXExactLijRuntimeTransportModel.response.center_prepared",
                )
            )
            return NTXExactLijLaggedResponse(
                face_response=face_response_bar_acc,
                center_response=NTXPreparedCoefficientResponse(
                    reference_transport_moments=reference_transport_moments_bar,
                    reference_nu_hat=reference_nu_hat_bar,
                    reference_epsi_hat=reference_epsi_hat_bar,
                )
            )

        output, pb = jax.vjp(
            lambda response_value: self.evaluate_with_lagged_response(
                state,
                NTXExactLijLaggedResponse(center_response=response_value),
            ),
            center_response,
        )
        (center_response_bar,) = pb(
            _complete_flux_bar_like(
                output,
                flux_bar,
                context="NTXExactLijRuntimeTransportModel.response.center_generic",
            )
        )
        return NTXExactLijLaggedResponse(
            face_response=face_response_bar_acc,
            center_response=center_response_bar,
        )

    def pullback_evaluate_with_lagged_response_support_payload(
        self,
        state,
        lagged_response,
        flux_bar,
        support,
        **kwargs,
    ):
        del kwargs
        if (
            lagged_response.center_response is None
            or self._resolved_center_response_mode() == "interpolate_from_faces"
        ):
            if isinstance(lagged_response.face_response, NTXInterpolatedMomentResponse):
                response = lagged_response.face_response
                evaluated = build_evaluated_transport_state(
                    state,
                    self.geometry,
                    bc_density=self.bc_density,
                    bc_temperature=self.bc_temperature,
                    density_floor=self.density_floor,
                    temperature_floor=self.temperature_floor,
                )
                face_state = evaluated.face
                face_density = evaluated.face.density
                face_temperature = face_state.temperature

                collisionality_kind = _collisionality_kind(self.collisionality_model)
                face_v_thermal = get_v_thermal(self.species.mass, face_temperature)
                species_indices = jnp.arange(int(self.species.number_species), dtype=jnp.int32)
                radius_indices = jnp.arange(face_state.Er.shape[0], dtype=jnp.int32)

                def _current_log_nu_star_from_drds(drds_values):
                    def _current_log_nu_star_per_radius(radius_index):
                        drds_value = jax.lax.dynamic_index_in_dim(
                            drds_values,
                            radius_index,
                            axis=0,
                            keepdims=False,
                        )
                        er_value = jax.lax.dynamic_index_in_dim(
                            face_state.Er,
                            radius_index,
                            axis=0,
                            keepdims=False,
                        )
                        temperature_local = jax.lax.dynamic_index_in_dim(
                            face_temperature,
                            radius_index,
                            axis=1,
                            keepdims=False,
                        )
                        density_local = jax.lax.dynamic_index_in_dim(
                            face_density,
                            radius_index,
                            axis=1,
                            keepdims=False,
                        )
                        vthermal_local = jax.lax.dynamic_index_in_dim(
                            face_v_thermal,
                            radius_index,
                            axis=1,
                            keepdims=False,
                        )
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

                    return jnp.swapaxes(
                        self._map_radius_axis_regularized_at_axis0(
                            _current_log_nu_star_per_radius,
                            radius_indices,
                            self.geometry.r_grid_half,
                            unbatched=True,
                        ),
                        0,
                        1,
                    )

                current_log_nu_star, drds_pullback = jax.vjp(
                    _current_log_nu_star_from_drds,
                    support.face_channels.drds,
                )
                delta_er = face_state.Er - response.reference_er
                delta_log_nu_star = current_log_nu_star - response.reference_log_nu_star
                transport_moments = (
                    response.reference_transport_moments
                    + response.dtransport_moments_d_er * delta_er[None, :, None]
                    + response.dtransport_moments_d_log_nu_star * delta_log_nu_star[:, :, None]
                )
                lij_faces = self._batched_lij_from_transport_moments(transport_moments, face_v_thermal)

                dndr_faces = evaluated.density_grad_face
                dTdr_faces = evaluated.temperature_grad_face
                a1 = jax.vmap(
                    lambda charge, density_a, temperature_a, dndr_a, dTdr_a: get_Thermodynamical_Forces_A1(
                        charge,
                        density_a,
                        temperature_a,
                        dndr_a,
                        dTdr_a,
                        face_state.Er,
                    )
                )(self.species.charge, face_density, face_temperature, dndr_faces, dTdr_faces)
                a2 = jax.vmap(get_Thermodynamical_Forces_A2)(face_temperature, dTdr_faces)
                a3 = get_Thermodynamical_Forces_A3(face_state.Er)
                density_phys = DENSITY_STATE_TO_PHYSICAL * face_density
                temperature_phys = TEMPERATURE_STATE_TO_PHYSICAL * face_temperature
                gamma_faces = -density_phys * (
                    lij_faces[:, :, 0, 0] * a1
                    + lij_faces[:, :, 0, 1] * a2
                    + lij_faces[:, :, 0, 2] * a3[None, :]
                )
                q_faces = -temperature_phys * density_phys * (
                    lij_faces[:, :, 1, 0] * a1
                    + lij_faces[:, :, 1, 1] * a2
                    + lij_faces[:, :, 1, 2] * a3[None, :]
                )
                upar_faces = -density_phys * (
                    lij_faces[:, :, 2, 0] * a1
                    + lij_faces[:, :, 2, 1] * a2
                    + lij_faces[:, :, 2, 2] * a3[None, :]
                )
                face_output = {
                    "Gamma_faces": gamma_faces,
                    "Q_faces": q_faces,
                    "Upar_faces": upar_faces,
                }
                face_flux_bar = _face_flux_bar_with_interpolated_center_bars(face_output, flux_bar)
                face_flux_bar = _complete_flux_bar_like(
                    face_output,
                    face_flux_bar,
                    context="NTXExactLijRuntimeTransportModel.support_payload.face_interpolated",
                )
                gamma_bar = face_flux_bar["Gamma_faces"]
                q_bar = face_flux_bar["Q_faces"]
                upar_bar = face_flux_bar["Upar_faces"]

                lij_bar = jnp.zeros_like(lij_faces)
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
                l11_fac = -inv_sqrt_pi * (mass / charge) ** 2 * face_v_thermal**3
                l13_fac = -inv_sqrt_pi * (mass / charge) * face_v_thermal**2
                l33_fac = -inv_sqrt_pi * face_v_thermal
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
                current_log_nu_star_bar = jnp.sum(
                    transport_moments_bar * response.dtransport_moments_d_log_nu_star,
                    axis=2,
                )
                (face_drds_bar,) = drds_pullback(current_log_nu_star_bar)
                face_channels_bar = dataclasses.replace(
                    _float_delta_tree_like(support.face_channels),
                    drds=face_drds_bar,
                )
                return _support_bar_from_face_bars(
                    support,
                    face_channels_bar,
                    _float_delta_tree_like(support.face_prepared),
                )

            face_channels_delta0 = _float_delta_tree_like(support.face_channels)
            face_prepared_delta0 = _float_delta_tree_like(support.face_prepared)
            face_output, face_support_pullback = jax.vjp(
                lambda face_channels_delta, face_prepared_delta: self.with_support_payload(
                    _support_with_face_delta(
                        support,
                        face_channels_delta,
                        face_prepared_delta,
                    )
                ).evaluate_with_lagged_response(state, lagged_response),
                face_channels_delta0,
                face_prepared_delta0,
            )
            face_flux_bar = _face_flux_bar_with_interpolated_center_bars(face_output, flux_bar)
            face_flux_bar = _complete_flux_bar_like(
                face_output,
                face_flux_bar,
                context="NTXExactLijRuntimeTransportModel.support_payload.face_interpolated",
            )
            face_channels_bar, face_prepared_bar = face_support_pullback(face_flux_bar)
            return _support_bar_from_face_bars(support, face_channels_bar, face_prepared_bar)

        center_delta0 = _float_delta_tree_like(support.center_channels)
        face_delta0 = _float_delta_tree_like(support.face_channels)
        output, channel_delta_pullback = jax.vjp(
            lambda center_delta, face_delta: self.with_support_payload(
                _support_with_channel_delta(support, center_delta, face_delta)
            ).evaluate_with_lagged_response(state, lagged_response),
            center_delta0,
            face_delta0,
        )
        flux_bar_for_output = (
            _face_flux_bar_with_interpolated_center_bars(output, flux_bar)
            if lagged_response.center_response is None
            or self._resolved_center_response_mode() == "interpolate_from_faces"
            else _complete_flux_bar_like(
                output,
                flux_bar,
                context="NTXExactLijRuntimeTransportModel.support_payload.center_local",
            )
        )
        center_bar, face_bar = channel_delta_pullback(flux_bar_for_output)
        return _support_bar_from_channel_bars(support, center_bar, face_bar)

    def pullback_evaluate_with_lagged_response_state(self, state, lagged_response, flux_bar, **kwargs):
        del kwargs
        face_response = lagged_response.face_response
        center_response = lagged_response.center_response
        state_bar_acc = jax.tree_util.tree_map(jnp.zeros_like, state)
        has_face_bar = any(
            key in flux_bar and flux_bar.get(key, None) is not None
            for key in ("Gamma_faces", "Q_faces", "Upar_faces")
        )

        if face_response is not None and (
            has_face_bar
            or center_response is None
            or self._resolved_center_response_mode() == "interpolate_from_faces"
        ):
            use_all_flux_bars_for_face = (
                center_response is None
                or self._resolved_center_response_mode() == "interpolate_from_faces"
            )
            if not isinstance(face_response, NTXInterpolatedMomentResponse):
                face_output, face_state_pullback = jax.vjp(
                    lambda state_value: self.evaluate_with_lagged_response(
                        state_value,
                        NTXExactLijLaggedResponse(face_response=face_response),
                    ),
                    state,
                )
                if use_all_flux_bars_for_face:
                    face_flux_bar = _face_flux_bar_with_interpolated_center_bars(face_output, flux_bar)
                else:
                    face_flux_bar = _complete_flux_bar_like(
                        face_output,
                        {
                            "Gamma_faces": flux_bar.get("Gamma_faces", None),
                            "Q_faces": flux_bar.get("Q_faces", None),
                            "Upar_faces": flux_bar.get("Upar_faces", None),
                        },
                        context="NTXExactLijRuntimeTransportModel.state.face",
                    )
                (face_state_bar,) = face_state_pullback(face_flux_bar)
                state_bar_acc = dataclasses.replace(
                    state_bar_acc,
                    density=state_bar_acc.density + face_state_bar.density,
                    pressure=state_bar_acc.pressure + face_state_bar.pressure,
                    Er=state_bar_acc.Er + face_state_bar.Er,
                )
                if center_response is None or self._resolved_center_response_mode() == "interpolate_from_faces":
                    return state_bar_acc

            evaluated = build_evaluated_transport_state(
                state,
                self.geometry,
                bc_density=self.bc_density,
                bc_temperature=self.bc_temperature,
                density_floor=self.density_floor,
                temperature_floor=self.temperature_floor,
            )
            face_state = evaluated.face
            face_density = face_state.density
            face_temperature = face_state.temperature
            face_er = face_state.Er
            dndr_faces = evaluated.density_grad_face
            dTdr_faces = evaluated.temperature_grad_face

            support = self._static_support()
            collisionality_kind = _collisionality_kind(self.collisionality_model)
            species_indices = jnp.arange(int(self.species.number_species), dtype=jnp.int32)
            radius_indices = jnp.arange(face_er.shape[0], dtype=jnp.int32)

            def _current_log_nu_star_from_face_inputs(density_faces, temperature_faces, er_faces):
                vthermal_faces = get_v_thermal(self.species.mass, temperature_faces)

                def _current_log_nu_star_per_radius(radius_index):
                    drds_value = jax.lax.dynamic_index_in_dim(
                        support.face_channels.drds,
                        radius_index,
                        axis=0,
                        keepdims=False,
                    )
                    er_value = jax.lax.dynamic_index_in_dim(
                        er_faces,
                        radius_index,
                        axis=0,
                        keepdims=False,
                    )
                    temperature_local = jax.lax.dynamic_index_in_dim(
                        temperature_faces,
                        radius_index,
                        axis=1,
                        keepdims=False,
                    )
                    density_local = jax.lax.dynamic_index_in_dim(
                        density_faces,
                        radius_index,
                        axis=1,
                        keepdims=False,
                    )
                    vthermal_local = jax.lax.dynamic_index_in_dim(
                        vthermal_faces,
                        radius_index,
                        axis=1,
                        keepdims=False,
                    )
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

                return jnp.swapaxes(
                    self._map_radius_axis_regularized_at_axis0(
                        _current_log_nu_star_per_radius,
                        radius_indices,
                        self.geometry.r_grid_half,
                        unbatched=True,
                    ),
                    0,
                    1,
                )

            current_log_nu_star, current_log_pullback = jax.vjp(
                _current_log_nu_star_from_face_inputs,
                face_density,
                face_temperature,
                face_er,
            )

            def _face_fluxes_from_local_arrays(
                density_faces,
                temperature_faces,
                er_faces,
                dndr_value,
                dTdr_value,
                log_nu_star_value,
            ):
                face_v_thermal = get_v_thermal(self.species.mass, temperature_faces)
                delta_er = er_faces - face_response.reference_er
                delta_log_nu_star = log_nu_star_value - face_response.reference_log_nu_star
                transport_moments = (
                    face_response.reference_transport_moments
                    + face_response.dtransport_moments_d_er * delta_er[None, :, None]
                    + face_response.dtransport_moments_d_log_nu_star * delta_log_nu_star[:, :, None]
                )
                lij_faces = self._batched_lij_from_transport_moments(transport_moments, face_v_thermal)
                a1 = jax.vmap(
                    lambda charge, density_a, temperature_a, dndr_a, dTdr_a: get_Thermodynamical_Forces_A1(
                        charge,
                        density_a,
                        temperature_a,
                        dndr_a,
                        dTdr_a,
                        er_faces,
                    )
                )(self.species.charge, density_faces, temperature_faces, dndr_value, dTdr_value)
                a2 = jax.vmap(get_Thermodynamical_Forces_A2)(temperature_faces, dTdr_value)
                a3 = get_Thermodynamical_Forces_A3(er_faces)
                density_phys = DENSITY_STATE_TO_PHYSICAL * density_faces
                temperature_phys = TEMPERATURE_STATE_TO_PHYSICAL * temperature_faces
                gamma_faces = -density_phys * (
                    lij_faces[:, :, 0, 0] * a1
                    + lij_faces[:, :, 0, 1] * a2
                    + lij_faces[:, :, 0, 2] * a3[None, :]
                )
                q_faces = -temperature_phys * density_phys * (
                    lij_faces[:, :, 1, 0] * a1
                    + lij_faces[:, :, 1, 1] * a2
                    + lij_faces[:, :, 1, 2] * a3[None, :]
                )
                upar_faces = -density_phys * (
                    lij_faces[:, :, 2, 0] * a1
                    + lij_faces[:, :, 2, 1] * a2
                    + lij_faces[:, :, 2, 2] * a3[None, :]
                )
                return {
                    "Gamma_faces": gamma_faces,
                    "Q_faces": q_faces,
                    "Upar_faces": upar_faces,
                }

            face_output, face_algebra_pullback = jax.vjp(
                _face_fluxes_from_local_arrays,
                face_density,
                face_temperature,
                face_er,
                dndr_faces,
                dTdr_faces,
                current_log_nu_star,
            )
            face_flux_bar = (
                _face_flux_bar_with_interpolated_center_bars(face_output, flux_bar)
                if use_all_flux_bars_for_face
                else _complete_flux_bar_like(
                    face_output,
                    {
                        "Gamma_faces": flux_bar.get("Gamma_faces", None),
                        "Q_faces": flux_bar.get("Q_faces", None),
                        "Upar_faces": flux_bar.get("Upar_faces", None),
                    },
                    context="NTXExactLijRuntimeTransportModel.state.face",
                )
            )
            (
                face_density_bar,
                face_temperature_bar,
                face_er_bar,
                dndr_faces_bar,
                dTdr_faces_bar,
                current_log_nu_star_bar,
            ) = face_algebra_pullback(face_flux_bar)
            (
                log_density_bar,
                log_temperature_bar,
                log_er_bar,
            ) = current_log_pullback(current_log_nu_star_bar)
            face_density_bar = face_density_bar + log_density_bar
            face_temperature_bar = face_temperature_bar + log_temperature_bar
            face_er_bar = face_er_bar + log_er_bar

            def _face_inputs_from_state(state_value):
                evaluated_value = build_evaluated_transport_state(
                    state_value,
                    self.geometry,
                    bc_density=self.bc_density,
                    bc_temperature=self.bc_temperature,
                    density_floor=self.density_floor,
                    temperature_floor=self.temperature_floor,
                )
                return (
                    evaluated_value.face.density,
                    evaluated_value.face.temperature,
                    evaluated_value.face.Er,
                    evaluated_value.density_grad_face,
                    evaluated_value.temperature_grad_face,
                )

            _, face_inputs_pullback = jax.vjp(_face_inputs_from_state, state)
            (face_state_bar,) = face_inputs_pullback(
                (
                    face_density_bar,
                    face_temperature_bar,
                    face_er_bar,
                    dndr_faces_bar,
                    dTdr_faces_bar,
                )
            )
            state_bar_acc = dataclasses.replace(
                state_bar_acc,
                density=state_bar_acc.density + face_state_bar.density,
                pressure=state_bar_acc.pressure + face_state_bar.pressure,
                Er=state_bar_acc.Er + face_state_bar.Er,
            )
            if center_response is None or self._resolved_center_response_mode() == "interpolate_from_faces":
                return state_bar_acc

        if isinstance(center_response, NTXInterpolatedMomentResponse):
            evaluated = build_evaluated_transport_state(
                state,
                self.geometry,
                bc_density=self.bc_density,
                bc_temperature=self.bc_temperature,
                density_floor=self.density_floor,
                temperature_floor=self.temperature_floor,
            )
            density = evaluated.center.density
            temperature = evaluated.center.temperature

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
                evaluated.density_grad_center,
                evaluated.temperature_grad_center,
                lij,
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

            dndr_all = evaluated.density_grad_center
            dTdr_all = evaluated.temperature_grad_center
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
                return _center_profile_gradient(
                    density_value,
                    self.geometry.r_grid_half,
                    bc_model=self.bc_density,
                )

            def _temperature_gradient_map(temperature_value):
                return _center_profile_gradient(
                    temperature_value,
                    self.geometry.r_grid_half,
                    bc_model=self.bc_temperature,
                )

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
            density_floor_arr = _broadcast_species_floor(jnp.asarray(state.density), self.density_floor)
            density_active = jnp.asarray(state.density) > density_floor_arr
            density_safe = safe_density(state.density, self.density_floor)
            pressure_bar = temperature_bar / density_safe
            density_bar = density_bar_direct - temperature_bar * state.pressure / (density_safe * density_safe)
            density_bar = density_bar * density_active.astype(density_bar.dtype)
            return dataclasses.replace(
                state_bar_acc,
                density=state_bar_acc.density + density_bar,
                pressure=state_bar_acc.pressure + pressure_bar,
                Er=state_bar_acc.Er + er_bar,
            )

        center_output, pb = jax.vjp(
            lambda state_value: self.evaluate_with_lagged_response(
                state_value,
                NTXExactLijLaggedResponse(center_response=center_response),
            ),
            state,
        )
        center_flux_bar = _complete_flux_bar_like(
            center_output,
            {
                "Gamma": flux_bar.get("Gamma", None),
                "Q": flux_bar.get("Q", None),
                "Upar": flux_bar.get("Upar", None),
            },
            context="NTXExactLijRuntimeTransportModel.state.center_generic",
        )
        (state_bar,) = pb(center_flux_bar)
        return dataclasses.replace(
            state_bar_acc,
            density=state_bar_acc.density + state_bar.density,
            pressure=state_bar_acc.pressure + state_bar.pressure,
            Er=state_bar_acc.Er + state_bar.Er,
        )

    def build_local_particle_flux_evaluator(self, state):
        evaluated = build_evaluated_transport_state(
            state,
            self.geometry,
            bc_density=self.bc_density,
            bc_temperature=self.bc_temperature,
            density_floor=self.density_floor,
            temperature_floor=self.temperature_floor,
        )
        density = evaluated.center.density
        temperature = evaluated.center.temperature
        support = self._static_support()
        center_prepared = support.center_prepared
        center_drds = support.center_channels.drds
        collisionality_kind = _collisionality_kind(self.collisionality_model)
        v_thermal = get_v_thermal(self.species.mass, temperature)
        species_indices = jnp.arange(int(self.species.number_species), dtype=jnp.int32)
        dndr_all = evaluated.density_grad_center
        dTdr_all = evaluated.temperature_grad_center

        def evaluator(radius_index, er_value):
            radius_index = jnp.asarray(radius_index, dtype=jnp.int32)
            er_scalar = jnp.asarray(er_value, dtype=state.Er.dtype)
            prepared = jax.tree_util.tree_map(
                lambda arr: jax.lax.dynamic_index_in_dim(arr, radius_index, axis=0, keepdims=False),
                center_prepared,
            )
            drds_value = jax.lax.dynamic_index_in_dim(center_drds, radius_index, axis=0, keepdims=False)
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

        evaluated = kwargs.get("evaluated_state")
        if evaluated is None:
            evaluated = build_evaluated_transport_state(
                state,
                self.geometry,
                bc_density=kwargs.get("bc_density", self.bc_density),
                bc_temperature=kwargs.get("bc_temperature", self.bc_temperature),
                density_floor=self.density_floor,
                temperature_floor=self.temperature_floor,
            )
        density = evaluated.center.density
        face_density = safe_density(face_state.density, self.density_floor)
        bc_density = kwargs.get("bc_density", self.bc_density)
        bc_temperature = kwargs.get("bc_temperature", self.bc_temperature)
        particle_face_closure_mode = str(kwargs.get("particle_face_closure_mode", "reconstructed")).strip().lower()
        if particle_face_closure_mode in {"ntss_like", "ntss", "half_point"}:
            dndr_faces = _ntss_like_face_gradient(
                density,
                self.geometry.r_grid_half,
                bc_model=bc_density,
            )
            dTdr_faces = _ntss_like_face_gradient(
                evaluated.center.temperature,
                self.geometry.r_grid_half,
                bc_model=bc_temperature,
            )
        else:
            dndr_faces = evaluated.density_grad_face
            dTdr_faces = evaluated.temperature_grad_face
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
    ntx_exact_center_response_mode=None,
    ntx_exact_radial_batch_size=None,
    ntx_exact_radial_batch_mode="simple",
    ntx_exact_scan_batch_size=None,
    ntx_exact_response_anchor_count=None,
    ntx_exact_use_remat=False,
    ntx_exact_derivative_mode="direct",
    ntx_exact_derivative_field_pullback_mode="compact_vjp",
    ntx_exact_derivative_pullback_boundary="inline",
    ntx_exact_derivative_pullback_algebra="ntx_helper",
    ntx_exact_er_v_floor=None,
    ntx_exact_full_state_quadratic_response=False,
    ntx_exact_debug_center_lij_comparison=False,
    lagged_response_taylor_order=1,
    ntx_exact_lij_support=None,
    preload_support=False,
    collisionality_model="default",
    bc_density=None,
    bc_temperature=None,
    density_floor=DEFAULT_TRANSPORT_DENSITY_FLOOR,
    temperature_floor=DEFAULT_TRANSPORT_TEMPERATURE_FLOOR,
    **kwargs,
):
    del kwargs
    if ntx_exact_center_response_mode is None:
        ntx_exact_center_response_mode = (
            "center_local_response"
            if str(ntx_exact_face_response_mode).strip().lower()
            in {"interpolate_center_response", "interpolate_center_fluxes", "center_interpolation"}
            else "interpolate_from_faces"
        )
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
        center_response_mode=NTXExactLijRuntimeTransportModel._normalize_center_response_mode(
            ntx_exact_center_response_mode
        ),
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
        full_state_quadratic_response=bool(ntx_exact_full_state_quadratic_response),
        debug_center_lij_comparison=bool(ntx_exact_debug_center_lij_comparison),
        collisionality_model=str(collisionality_model),
        bc_density=bc_density,
        bc_temperature=bc_temperature,
        density_floor=density_floor,
        temperature_floor=temperature_floor,
        support=ntx_exact_lij_support,
        lagged_response_taylor_order=int(lagged_response_taylor_order),
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
    ntx_scan_surfaces=None,
    preload_channels=False,
    prebuild_database=True,
    lagged_response_taylor_order=1,
    ntx_scan_coefficient_reverse_mode="generic",
    ntx_scan_record_primal=False,
    **kwargs,
):
    del kwargs
    model = NTXRuntimeScanTransportModel(
        species=species,
        energy_grid=energy_grid,
        geometry=geometry,
        vmec_file=None if vmec_file is None else str(vmec_file),
        boozer_file=None if boozer_file is None else str(boozer_file),
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
        scan_surfaces=None if ntx_scan_surfaces is None else tuple(ntx_scan_surfaces),
        database=None,
        lagged_response_taylor_order=int(lagged_response_taylor_order),
        coefficient_reverse_mode=str(ntx_scan_coefficient_reverse_mode),
        record_scan_primal=bool(ntx_scan_record_primal),
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


def _spectrax_response_label_kind_codes(labels):
    mapping = {
        "density_gradient": 0,
        "temperature_gradient": 1,
    }
    codes = []
    for raw in labels:
        key = str(raw.decode("utf-8") if isinstance(raw, bytes) else raw).strip().lower()
        if key == "base":
            raise ValueError("response_label='base' is not a valid perturbation channel for SPECTRAX FD response data.")
        if key not in mapping:
            raise ValueError(f"Unsupported SPECTRAX response_label {raw!r}.")
        codes.append(mapping[key])
    return jnp.asarray(codes, dtype=jnp.int32)


def _spectrax_response_labels_to_kind_and_species(labels, species_names):
    kind_codes = []
    species_indices = []
    lookup = {str(name).strip().lower(): idx for idx, name in enumerate(species_names)}
    for raw in labels:
        key = str(raw.decode("utf-8") if isinstance(raw, bytes) else raw).strip().lower()
        if key == "base":
            raise ValueError("response_label='base' is not a valid perturbation channel for SPECTRAX FD response data.")
        if key.startswith("fd_n_"):
            species_key = key[len("fd_n_") :]
            kind_code = 0
        elif key.startswith("fd_t_"):
            species_key = key[len("fd_t_") :]
            kind_code = 1
        elif key in {"fd_er", "fd_e_r"}:
            raise ValueError(
                "SPECTRAX response_label 'fd_er' is not yet supported by NEOPAX fluxes_r_file lagged_response_mode='fd'."
            )
        else:
            raise ValueError(f"Unsupported SPECTRAX response_label {raw!r}.")
        if species_key not in lookup:
            raise ValueError(
                f"Unknown SPECTRAX response_label species tag {species_key!r} for NEOPAX species {species_names!r}."
            )
        species_indices.append(lookup[species_key])
        kind_codes.append(kind_code)
    return (
        jnp.asarray(kind_codes, dtype=jnp.int32),
        jnp.asarray(species_indices, dtype=jnp.int32),
    )


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


def read_flux_profile_center_file(path, n_species):
    with h5py.File(path, "r") as f:
        keys = set(f.keys())

        def _first(*names):
            for name in names:
                if name in keys:
                    return f[name][...]
            return None

        r = _first("r_center", "rho_center", "r_grid_center", "radius_center")
        gamma = _first("Gamma_center", "gamma_center")
        q = _first("Q_center", "q_center")
        upar = _first("Upar_center", "upar_center", "u_par_center")

    if r is None or (gamma is None and q is None and upar is None):
        return None

    r_arr = jnp.ravel(jnp.asarray(r, dtype=float))
    gamma_arr = None if gamma is None else _normalize_flux_dataset(gamma, n_species)
    q_arr = None if q is None else _normalize_flux_dataset(q, n_species)
    upar_arr = None if upar is None else _normalize_flux_dataset(upar, n_species)
    return r_arr, gamma_arr, q_arr, upar_arr


def read_flux_profile_fd_response_file(path, n_species, species_names):
    with h5py.File(path, "r") as f:
        keys = set(f.keys())
        gamma_key = "Gamma_perturb" if "Gamma_perturb" in keys else "Gamma_perturbed"
        q_key = "Q_perturb" if "Q_perturb" in keys else "Q_perturbed"
        shared_required = {"perturb_delta", "perturb_present"}
        missing = []
        if gamma_key not in keys:
            missing.append("Gamma_perturb or Gamma_perturbed")
        if q_key not in keys:
            missing.append("Q_perturb or Q_perturbed")
        missing.extend(sorted(shared_required - keys))
        if missing:
            raise ValueError(
                f"Flux file '{path}' is missing SPECTRAX FD lagged-response datasets: {', '.join(missing)}."
            )
        gamma_perturb = _normalize_perturb_species_flux_dataset(f[gamma_key][...], n_species)
        q_perturb = _normalize_perturb_species_flux_dataset(f[q_key][...], n_species)
        perturb_delta = jnp.asarray(f["perturb_delta"][...], dtype=float)
        perturb_present = jnp.asarray(f["perturb_present"][...], dtype=bool)
        if {"perturb_kind", "perturb_species"}.issubset(keys):
            perturb_kind_codes = _spectrax_perturb_kind_codes(f["perturb_kind"][...])
            perturb_species_indices = _spectrax_perturb_species_indices(f["perturb_species"][...], species_names)
        elif {"response_label", "perturb_species"}.issubset(keys):
            perturb_kind_codes = _spectrax_response_label_kind_codes(f["response_label"][...])
            perturb_species_indices = _spectrax_perturb_species_indices(f["perturb_species"][...], species_names)
        elif "response_label" in keys:
            perturb_kind_codes, perturb_species_indices = _spectrax_response_labels_to_kind_and_species(
                f["response_label"][...],
                species_names,
            )
        else:
            raise ValueError(
                f"Flux file '{path}' is missing SPECTRAX FD perturbation labels. "
                "Expected either (perturb_kind, perturb_species) or response_label."
            )
    return (
        gamma_perturb,
        q_perturb,
        perturb_delta,
        perturb_present,
        perturb_kind_codes,
        perturb_species_indices,
    )


def _require_matching_fd_grid(file_r, target_r):
    file_r = jnp.asarray(file_r, dtype=jnp.float64)
    target_r = jnp.asarray(target_r, dtype=jnp.float64)
    if file_r.shape != target_r.shape or not bool(jnp.allclose(file_r, target_r, rtol=0.0, atol=1.0e-12)):
        raise ValueError(
            "fluxes_r_file lagged_response_mode='fd' currently requires the file radial grid "
            "to match the primary NEOPAX flux grid exactly."
        )


def _require_matching_flux_grid(file_r, target_r, *, label):
    file_r = jnp.asarray(file_r, dtype=jnp.float64)
    target_r = jnp.asarray(target_r, dtype=jnp.float64)
    if file_r.shape != target_r.shape or not bool(jnp.allclose(file_r, target_r, rtol=0.0, atol=1.0e-12)):
        raise ValueError(
            f"fluxes_r_file {label} radial grid must match the corresponding NEOPAX geometry grid exactly."
        )


def _require_interpolatable_flux_grid(file_r, target_r, target_name):
    file_r = jnp.asarray(file_r, dtype=jnp.float64)
    target_r = jnp.asarray(target_r, dtype=jnp.float64)
    if file_r.size < 2:
        raise ValueError(
            f"fluxes_r_file radial grid holds {file_r.size} entries; interpolation needs at least 2. "
            "The finiteness and ordering checks below are vacuous on a shorter grid."
        )
    if not bool(jnp.all(jnp.isfinite(file_r))):
        raise ValueError(
            "fluxes_r_file radial grid holds non-finite entries, which interpolate to NaN at every radius."
        )
    if not bool(jnp.all(jnp.diff(file_r) > 0.0)):
        raise ValueError(
            "fluxes_r_file radial grid must be strictly increasing. interpax takes its first and last "
            "entries as the interpolation bounds, so a descending grid interpolates to NaN everywhere "
            "and an unordered one silently reads from the wrong interval."
        )
    # Bounds come from the end points rather than min/max, matching how interpax reads them.
    file_lo, file_hi = float(file_r[0]), float(file_r[-1])
    target_lo, target_hi = float(jnp.min(target_r)), float(jnp.max(target_r))
    if not (file_lo <= target_lo and target_hi <= file_hi):
        raise ValueError(
            f"fluxes_r_file radial grid [{file_lo:.6e}, {file_hi:.6e}] does not cover "
            f"{target_name} [{target_lo:.6e}, {target_hi:.6e}]. Radii outside the file grid "
            "interpolate to NaN and would reach the transport solve unreported."
        )


def _require_fd_response_radial_shape(name, arr, target_r):
    arr = jnp.asarray(arr)
    target_n = int(jnp.asarray(target_r).shape[0])
    if arr.shape[-1] != target_n:
        raise ValueError(
            f"fluxes_r_file lagged_response_mode='fd' dataset {name} has radial length "
            f"{arr.shape[-1]}, expected {target_n} to match the primary NEOPAX flux grid."
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
    grid_location="face_centered",
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
    center_flux_mode = str(
        kwargs.pop(
            "center_flux_mode",
            kwargs.pop("center_response_mode", "interpolate_from_faces"),
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
    normalized_location_for_check = str(location).strip().lower()
    if normalized_location_for_check in {"face", "faces", "face_centered", "face-centred", "face_centred"}:
        _require_interpolatable_flux_grid(r_data, geometry.r_grid_half, "geometry.r_grid_half")
    elif normalized_location_for_check in {"cell", "cells", "center", "centers", "cell_centered", "cell-centred", "cell_centred"}:
        _require_interpolatable_flux_grid(r_data, geometry.r_grid, "geometry.r_grid")
    center_r_data = None
    center_gamma_data = None
    center_q_data = None
    center_upar_data = None
    if center_flux_mode in {"file_center", "file_centers", "direct_file_center", "direct"}:
        center_payload = read_flux_profile_center_file(path, species.number_species)
        if center_payload is None:
            warnings.warn(
                "fluxes_r_file center_flux_mode='file_center' requested, but the file does not contain "
                "center datasets (r_center/rho_center plus Gamma_center/Q_center/Upar_center).",
                RuntimeWarning,
                stacklevel=2,
            )
            raise ValueError(
                "fluxes_r_file center_flux_mode='file_center' requires center-grid datasets in the flux file."
            )
        center_r_data, center_gamma_data, center_q_data, center_upar_data = center_payload
        _require_interpolatable_flux_grid(center_r_data, geometry.r_grid, "geometry.r_grid")
        _require_matching_flux_grid(center_r_data, geometry.r_grid, label="center")
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
        _require_matching_fd_grid(r_data, geometry.r_grid_half)
        _require_fd_response_radial_shape("Gamma_perturb/Gamma_perturbed", gamma_perturb_data, geometry.r_grid_half)
        _require_fd_response_radial_shape("Q_perturb/Q_perturbed", q_perturb_data, geometry.r_grid_half)
        _require_fd_response_radial_shape("perturb_delta", perturb_delta_data, geometry.r_grid_half)
        _require_fd_response_radial_shape("perturb_present", perturb_present_data, geometry.r_grid_half)
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
        f"center_flux_mode={center_flux_mode} r.shape={tuple(r_data.shape)} q_scale={q_scale:.6e} {r_summary}"
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
        center_flux_mode=center_flux_mode,
        q_scale=q_scale,
        lagged_response_mode=lagged_response_mode,
        gamma_perturb_data=gamma_perturb_data,
        q_perturb_data=q_perturb_data,
        perturb_delta_data=perturb_delta_data,
        perturb_present_data=perturb_present_data,
        perturb_kind_codes=perturb_kind_codes,
        perturb_species_indices=perturb_species_indices,
        center_r_data=center_r_data,
        center_gamma_data=center_gamma_data,
        center_q_data=center_q_data,
        center_upar_data=center_upar_data,
    )


@dataclasses.dataclass(frozen=True, eq=False)
class FluxesRFileTransportModel(TransportFluxModelBase):
    species: Any
    geometry: Any
    r_data: Any
    gamma_data: Any = None
    q_data: Any = None
    upar_data: Any = None
    profile_location: str = "face_centered"
    center_flux_mode: str = "interpolate_from_faces"
    q_scale: float = 1.0
    lagged_response_mode: str = "none"
    gamma_perturb_data: Any = None
    q_perturb_data: Any = None
    perturb_delta_data: Any = None
    perturb_present_data: Any = None
    perturb_kind_codes: Any = None
    perturb_species_indices: Any = None
    center_r_data: Any = None
    center_gamma_data: Any = None
    center_q_data: Any = None
    center_upar_data: Any = None

    def with_q_scale(self, q_scale: float) -> "FluxesRFileTransportModel":
        return dataclasses.replace(self, q_scale=float(q_scale))

    def _interp_species_profile(self, data, target_r):
        if data is None:
            return jnp.zeros((self.species.number_species, target_r.shape[0]), dtype=target_r.dtype)
        return jax.vmap(lambda prof: interpax.interp1d(target_r, self.r_data, prof))(data)

    def _interp_center_species_profile(self, data, target_r):
        if data is None:
            return jnp.zeros((self.species.number_species, target_r.shape[0]), dtype=target_r.dtype)
        if self.center_r_data is None:
            raise ValueError("fluxes_r_file center grid data is missing.")
        return jax.vmap(lambda prof: interpax.interp1d(target_r, self.center_r_data, prof))(data)

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

    def _normalize_center_flux_mode(self):
        mode = str(self.center_flux_mode).strip().lower()
        aliases = {
            "interpolate": "interpolate_from_faces",
            "interpolate_from_faces": "interpolate_from_faces",
            "from_faces": "interpolate_from_faces",
            "cell_centered_from_faces": "interpolate_from_faces",
            "file_center": "file_center",
            "file_centers": "file_center",
            "direct": "file_center",
            "direct_file_center": "file_center",
        }
        if mode not in aliases:
            raise ValueError(
                f"Unsupported fluxes_r_file center_flux_mode '{self.center_flux_mode}'. "
                "Expected interpolate_from_faces or file_center."
            )
        return aliases[mode]

    def _data_on_cell_grid(self, data):
        center_mode = self._normalize_center_flux_mode()
        if center_mode == "file_center":
            if data is self.gamma_data:
                return self._interp_center_species_profile(self.center_gamma_data, self.geometry.r_grid)
            if data is self.q_data:
                return self._interp_center_species_profile(self.center_q_data, self.geometry.r_grid)
            if data is self.upar_data:
                return self._interp_center_species_profile(self.center_upar_data, self.geometry.r_grid)
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

    def _spectrax_fd_face_basis(self, state):
        center_basis = self._spectrax_fd_basis(state)
        return jax.vmap(faces_from_cell_centered)(center_basis)

    def __call__(self, state) -> dict:
        del state
        gamma = self._data_on_cell_grid(self.gamma_data)
        q = self.q_scale * self._data_on_cell_grid(self.q_data)
        upar = self._data_on_cell_grid(self.upar_data)
        face_fluxes = self.evaluate_face_fluxes(None, None)
        return {
            "Gamma": gamma,
            "Q": q,
            "Upar": upar,
            "Gamma_faces": face_fluxes["Gamma"],
            "Q_faces": face_fluxes["Q"],
            "Upar_faces": face_fluxes["Upar"],
        }

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
            reference_basis=self._spectrax_fd_face_basis(state),
            perturb_kind_codes=self.perturb_kind_codes,
            perturb_species_indices=self.perturb_species_indices,
            perturb_delta=jnp.asarray(self.perturb_delta_data, dtype=float),
            perturb_present=jnp.asarray(self.perturb_present_data, dtype=bool),
            gamma_perturb=jnp.asarray(self.gamma_perturb_data, dtype=float),
            q_perturb=self.q_scale * jnp.asarray(self.q_perturb_data, dtype=float),
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

        # Freeze the lagged response in NEOPAX state-space around the reference
        # state: use the basis map tangent at the reference rather than
        # reevaluating the nonlinear basis at the current Newton iterate.
        delta_state = jax.tree_util.tree_map(
            lambda current, reference: current - reference,
            state,
            lagged_response.reference_state,
        )
        delta_basis = jax.jvp(
            self._spectrax_fd_face_basis,
            (lagged_response.reference_state,),
            (delta_state,),
        )[1]
        perturb_present = lagged_response.perturb_present[:, None, :]
        safe_delta = jnp.where(
            perturb_present[:, 0, :],
            lagged_response.perturb_delta,
            1.0,
        )
        dgamma = jnp.where(
            perturb_present,
            (lagged_response.gamma_perturb - lagged_response.reference_flux["Gamma_faces"][None, :, :]) / safe_delta[:, None, :],
            0.0,
        )
        dq = jnp.where(
            perturb_present,
            (lagged_response.q_perturb - lagged_response.reference_flux["Q_faces"][None, :, :]) / safe_delta[:, None, :],
            0.0,
        )
        gamma_faces = lagged_response.reference_flux["Gamma_faces"] + jnp.sum(dgamma * delta_basis[:, None, :], axis=0)
        q_faces = lagged_response.reference_flux["Q_faces"] + jnp.sum(dq * delta_basis[:, None, :], axis=0)
        upar_faces = lagged_response.reference_flux["Upar_faces"]
        gamma = jax.vmap(cell_centered_from_faces)(gamma_faces)
        q = jax.vmap(cell_centered_from_faces)(q_faces)
        upar = jax.vmap(cell_centered_from_faces)(upar_faces)
        return {
            "Gamma": gamma,
            "Q": q,
            "Upar": upar,
            "Gamma_faces": gamma_faces,
            "Q_faces": q_faces,
            "Upar_faces": upar_faces,
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
        evaluated = kwargs.get("evaluated_state")
        if evaluated is None:
            evaluated = build_evaluated_transport_state(
                state,
                self.field,
                bc_density=bc_density,
                bc_temperature=bc_temperature,
            )
        dndr_faces = DENSITY_STATE_TO_PHYSICAL * evaluated.density_grad_face
        dTdr_faces = TEMPERATURE_STATE_TO_PHYSICAL * evaluated.temperature_grad_face
        gamma = -self.chi_n[:, None] * dndr_faces
        q = -(DENSITY_STATE_TO_PHYSICAL * face_state.density) * self.chi_t[:, None] * dTdr_faces
        upar = jnp.zeros_like(gamma)
        return {"Gamma": gamma, "Q": q, "Upar": upar}

    def _fluxes_from_face_fluxes(self, face_fluxes):
        return {
            "Gamma": jax.vmap(cell_centered_from_faces)(face_fluxes["Gamma"]),
            "Q": jax.vmap(cell_centered_from_faces)(face_fluxes["Q"]),
            "Upar": jax.vmap(cell_centered_from_faces)(face_fluxes["Upar"]),
            "Gamma_faces": face_fluxes["Gamma"],
            "Q_faces": face_fluxes["Q"],
            "Upar_faces": face_fluxes["Upar"],
        }

    def build_lagged_response(self, state, **kwargs):
        bc_density = kwargs.get("bc_density")
        bc_temperature = kwargs.get("bc_temperature")
        face_state = build_face_transport_state(
            state,
            self.field,
            bc_density=bc_density,
            bc_temperature=bc_temperature,
        )
        evaluated_state = build_evaluated_transport_state(
            state,
            self.field,
            bc_density=bc_density,
            bc_temperature=bc_temperature,
        )
        return FaceJVPTransportFluxResponse(
            reference_state=state,
            reference_face_flux=self.evaluate_face_fluxes(
                state,
                face_state,
                bc_density=bc_density,
                bc_temperature=bc_temperature,
                evaluated_state=evaluated_state,
            ),
        )

    def evaluate_with_lagged_response(self, state, lagged_response, **kwargs):
        bc_density = kwargs.get("bc_density")
        bc_temperature = kwargs.get("bc_temperature")
        delta_state = jax.tree_util.tree_map(
            lambda current, reference: current - reference,
            state,
            lagged_response.reference_state,
        )

        def _face_fluxes_from_state(state_value):
            face_state_value = build_face_transport_state(
                state_value,
                self.field,
                bc_density=bc_density,
                bc_temperature=bc_temperature,
            )
            evaluated_state_value = build_evaluated_transport_state(
                state_value,
                self.field,
                bc_density=bc_density,
                bc_temperature=bc_temperature,
            )
            return self.evaluate_face_fluxes(
                state_value,
                face_state_value,
                bc_density=bc_density,
                bc_temperature=bc_temperature,
                evaluated_state=evaluated_state_value,
            )

        tangent_face_flux = jax.jvp(
            _face_fluxes_from_state,
            (lagged_response.reference_state,),
            (delta_state,),
        )[1]
        face_fluxes = jax.tree_util.tree_map(
            lambda reference, tangent: reference + tangent,
            lagged_response.reference_face_flux,
            tangent_face_flux,
        )
        return self._fluxes_from_face_fluxes(face_fluxes)


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
        evaluated = kwargs.get("evaluated_state")
        if evaluated is None:
            evaluated = build_evaluated_transport_state(
                state,
                self.field,
                bc_density=bc_density,
                bc_temperature=bc_temperature,
            )
        total_power_mw = self._effective_total_power_mw(state)
        dndr_faces = DENSITY_STATE_TO_PHYSICAL * evaluated.density_grad_face
        dTdr_faces = TEMPERATURE_STATE_TO_PHYSICAL * evaluated.temperature_grad_face
        electron_idx = int(self.species.species_idx["e"])
        ne_face = jnp.maximum(jnp.asarray(face_state.density[electron_idx], dtype=state.density.dtype), 1.0e-12)
        p075 = jnp.where(total_power_mw < 0.0, jnp.asarray(3.0, dtype=state.density.dtype), jnp.power(total_power_mw, 0.75))
        density_coeff = jnp.asarray(self.chi_n, dtype=state.density.dtype)[:, None] * p075 / ne_face[None, :]
        heat_coeff = jnp.asarray(self.chi_t, dtype=state.density.dtype)[:, None] * p075 / ne_face[None, :]
        gamma = -density_coeff * dndr_faces
        q = -(DENSITY_STATE_TO_PHYSICAL * face_state.density) * heat_coeff * dTdr_faces
        upar = jnp.zeros_like(gamma)
        return {"Gamma": gamma, "Q": q, "Upar": upar}

    def _fluxes_from_face_fluxes(self, face_fluxes):
        return {
            "Gamma": jax.vmap(cell_centered_from_faces)(face_fluxes["Gamma"]),
            "Q": jax.vmap(cell_centered_from_faces)(face_fluxes["Q"]),
            "Upar": jax.vmap(cell_centered_from_faces)(face_fluxes["Upar"]),
            "Gamma_faces": face_fluxes["Gamma"],
            "Q_faces": face_fluxes["Q"],
            "Upar_faces": face_fluxes["Upar"],
        }

    def build_lagged_response(self, state, **kwargs):
        bc_density = kwargs.get("bc_density")
        bc_temperature = kwargs.get("bc_temperature")
        face_state = build_face_transport_state(
            state,
            self.field,
            bc_density=bc_density,
            bc_temperature=bc_temperature,
        )
        evaluated_state = build_evaluated_transport_state(
            state,
            self.field,
            bc_density=bc_density,
            bc_temperature=bc_temperature,
        )
        return FaceJVPTransportFluxResponse(
            reference_state=state,
            reference_face_flux=self.evaluate_face_fluxes(
                state,
                face_state,
                bc_density=bc_density,
                bc_temperature=bc_temperature,
                evaluated_state=evaluated_state,
            ),
        )

    def evaluate_with_lagged_response(self, state, lagged_response, **kwargs):
        bc_density = kwargs.get("bc_density")
        bc_temperature = kwargs.get("bc_temperature")
        delta_state = jax.tree_util.tree_map(
            lambda current, reference: current - reference,
            state,
            lagged_response.reference_state,
        )

        def _face_fluxes_from_state(state_value):
            face_state_value = build_face_transport_state(
                state_value,
                self.field,
                bc_density=bc_density,
                bc_temperature=bc_temperature,
            )
            evaluated_state_value = build_evaluated_transport_state(
                state_value,
                self.field,
                bc_density=bc_density,
                bc_temperature=bc_temperature,
            )
            return self.evaluate_face_fluxes(
                state_value,
                face_state_value,
                bc_density=bc_density,
                bc_temperature=bc_temperature,
                evaluated_state=evaluated_state_value,
            )

        tangent_face_flux = jax.jvp(
            _face_fluxes_from_state,
            (lagged_response.reference_state,),
            (delta_state,),
        )[1]
        face_fluxes = jax.tree_util.tree_map(
            lambda reference, tangent: reference + tangent,
            lagged_response.reference_face_flux,
            tangent_face_flux,
        )
        return self._fluxes_from_face_fluxes(face_fluxes)


@dataclasses.dataclass(frozen=True, eq=False)
class ReLUAnalyticalTurbulentTransportModel(TransportFluxModelBase):
    species: Any
    field: Any
    density_critical_gradient: Any = 1.0
    temperature_critical_gradient: Any = 1.0
    density_relu_slope: Any = 1.0
    temperature_relu_slope: Any = 1.0
    relu_power: Any = 1.0

    def with_transport_coeffs(
        self,
        *,
        chi_t=None,
        chi_n=None,
        pressure_source_model=None,
        total_power_mw=None,
        density_critical_gradient=None,
        temperature_critical_gradient=None,
        density_relu_slope=None,
        temperature_relu_slope=None,
        relu_power=None,
    ) -> "ReLUAnalyticalTurbulentTransportModel":
        del chi_t, chi_n, pressure_source_model, total_power_mw
        return dataclasses.replace(
            self,
            density_critical_gradient=(
                self.density_critical_gradient
                if density_critical_gradient is None
                else density_critical_gradient
            ),
            temperature_critical_gradient=(
                self.temperature_critical_gradient
                if temperature_critical_gradient is None
                else temperature_critical_gradient
            ),
            density_relu_slope=self.density_relu_slope if density_relu_slope is None else density_relu_slope,
            temperature_relu_slope=(
                self.temperature_relu_slope
                if temperature_relu_slope is None
                else temperature_relu_slope
            ),
            relu_power=self.relu_power if relu_power is None else relu_power,
        )

    def _relu_fluxes(self, state, face_state, *, bc_density=None, bc_temperature=None):
        dtype = state.density.dtype
        n_species = int(state.density.shape[0])

        def _species_column(value):
            arr = jnp.asarray(value, dtype=dtype)
            if arr.ndim == 0:
                arr = jnp.full((n_species,), arr, dtype=dtype)
            if arr.shape[0] < n_species:
                arr = jnp.pad(arr, (0, n_species - arr.shape[0]), mode="edge")
            return arr[:n_species, None]

        density_face = jnp.maximum(
            jnp.asarray(face_state.density, dtype=dtype),
            jnp.asarray(1.0e-30, dtype=dtype),
        )
        temperature_face = jnp.maximum(
            jnp.asarray(face_state.temperature, dtype=dtype),
            jnp.asarray(1.0e-30, dtype=dtype),
        )
        # T3D's ReLU model thresholds are specified for normalized gradients
        # -d ln(profile) / d rho, not physical minor-radius gradients.
        gradient_faces = getattr(self.field, "rho_grid_half", self.field.r_grid_half)
        dndr_faces = _face_profile_gradient(
            jnp.asarray(state.density, dtype=dtype),
            gradient_faces,
            bc_model=bc_density,
        )
        dTdr_faces = _face_profile_gradient(
            jnp.asarray(state.temperature, dtype=dtype),
            gradient_faces,
            bc_model=bc_temperature,
        )
        kn = -dndr_faces / density_face
        kT = -dTdr_faces / temperature_face
        crit_n = _species_column(self.density_critical_gradient)
        crit_T = _species_column(self.temperature_critical_gradient)
        slope_n = _species_column(self.density_relu_slope)
        slope_T = _species_column(self.temperature_relu_slope)
        power = jnp.asarray(self.relu_power, dtype=dtype)

        def _signed_power(value):
            return jnp.sign(value) * jnp.power(jnp.abs(value), power)

        gamma = jnp.sign(kn - crit_n) * jnp.power(jnp.abs(slope_n * (kn - crit_n)), power)
        relu_base = kT - crit_T * jnp.sign(kT)
        q = jnp.where(
            jnp.abs(kT) < crit_T,
            jnp.asarray(0.0, dtype=dtype),
            slope_T * _signed_power(relu_base),
        )
        return gamma, q

    def _to_neopax_physical_fluxes(self, state, face_state, gamma_raw, q_raw):
        """Convert T3D-like normalized ReLU fluxes to NEOPAX physical units.

        The ReLU algebra returns T3D-style gyroBohm-normalized particle and heat
        flux amplitudes. NEOPAX's transport equations expect particle flux in
        m^-2 s^-1 and heat flux in eV m^-2 s^-1.
        """

        dtype = state.density.dtype
        names = tuple(getattr(self.species, "names", ()))
        if "e" in names:
            electron_idx = int(names.index("e"))
            ref_candidates = [idx for idx in range(int(self.species.number_species)) if idx != electron_idx]
            ref_idx = int(ref_candidates[0]) if ref_candidates else 0
        else:
            ref_idx = 0

        ref_density = jnp.maximum(
            jnp.asarray(face_state.density[ref_idx], dtype=dtype),
            jnp.asarray(1.0e-30, dtype=dtype),
        )
        ref_temperature = jnp.maximum(
            jnp.asarray(face_state.temperature[ref_idx], dtype=dtype),
            jnp.asarray(1.0e-30, dtype=dtype),
        )

        a_minor = jnp.asarray(getattr(self.field, "a_b", 1.0), dtype=dtype)
        psia = jnp.asarray(getattr(self.field, "Psia_value", 1.0), dtype=dtype)
        b_ref = jnp.maximum(
            jnp.abs(psia) / (jnp.pi * jnp.maximum(a_minor**2, jnp.asarray(1.0e-30, dtype=dtype))),
            jnp.asarray(1.0e-30, dtype=dtype),
        )
        ref_pressure = ref_density * ref_temperature

        m_ref_mp = jnp.asarray(self.species.mass_mp[ref_idx], dtype=dtype)
        reference_thermal_speed = jnp.sqrt(jnp.asarray(1.0e3, dtype=dtype) * elementary_charge / proton_mass)
        reference_gyroradius = jnp.sqrt(jnp.asarray(1.0e3, dtype=dtype) * elementary_charge * proton_mass) / elementary_charge
        vt_ref = reference_thermal_speed / jnp.sqrt(jnp.maximum(m_ref_mp, jnp.asarray(1.0e-30, dtype=dtype)))
        rho_ref = reference_gyroradius * jnp.sqrt(jnp.maximum(m_ref_mp, jnp.asarray(1.0e-30, dtype=dtype)))
        t_ref = (a_minor / vt_ref) * jnp.square(a_minor / jnp.maximum(rho_ref, jnp.asarray(1.0e-30, dtype=dtype)))

        particle_source_ref = jnp.asarray(1.0e20, dtype=dtype) / t_ref
        pressure_power_ref_eV = jnp.asarray(1.0e23, dtype=dtype) / t_ref
        particle_factor = jnp.power(ref_pressure, 1.5) / jnp.sqrt(ref_density)
        heat_factor = jnp.power(ref_pressure, 2.5) / jnp.power(ref_density, 1.5)
        geometry_factor = a_minor / jnp.square(b_ref)

        gamma = (
            gamma_raw
            * particle_factor[None, :]
            * particle_source_ref
            * geometry_factor
        )
        q = (
            q_raw
            * heat_factor[None, :]
            * pressure_power_ref_eV
            * geometry_factor
        )
        return gamma, q

    def __call__(self, state) -> dict:
        face_state = build_face_transport_state(state, self.field)
        face_fluxes = self.evaluate_face_fluxes(state, face_state)
        return {
            "Gamma": jax.vmap(cell_centered_from_faces)(face_fluxes["Gamma"]),
            "Q": jax.vmap(cell_centered_from_faces)(face_fluxes["Q"]),
            "Upar": jnp.zeros_like(state.density),
        }

    def evaluate_face_fluxes(self, state, face_state, **kwargs):
        gamma_raw, q_raw = self._relu_fluxes(
            state,
            face_state,
            bc_density=kwargs.get("bc_density"),
            bc_temperature=kwargs.get("bc_temperature"),
        )
        gamma, q = self._to_neopax_physical_fluxes(state, face_state, gamma_raw, q_raw)
        return {
            "Gamma": gamma,
            "Q": q,
            "Upar": jnp.zeros_like(gamma),
        }

    def _fluxes_from_face_fluxes(self, face_fluxes):
        return {
            "Gamma": jax.vmap(cell_centered_from_faces)(face_fluxes["Gamma"]),
            "Q": jax.vmap(cell_centered_from_faces)(face_fluxes["Q"]),
            "Upar": jax.vmap(cell_centered_from_faces)(face_fluxes["Upar"]),
            "Gamma_faces": face_fluxes["Gamma"],
            "Q_faces": face_fluxes["Q"],
            "Upar_faces": face_fluxes["Upar"],
        }

    def build_lagged_response(self, state, **kwargs):
        bc_density = kwargs.get("bc_density")
        bc_temperature = kwargs.get("bc_temperature")
        face_state = build_face_transport_state(
            state,
            self.field,
            bc_density=bc_density,
            bc_temperature=bc_temperature,
        )
        return FaceJVPTransportFluxResponse(
            reference_state=state,
            reference_face_flux=self.evaluate_face_fluxes(
                state,
                face_state,
                bc_density=bc_density,
                bc_temperature=bc_temperature,
            ),
        )

    def evaluate_with_lagged_response(self, state, lagged_response, **kwargs):
        bc_density = kwargs.get("bc_density")
        bc_temperature = kwargs.get("bc_temperature")
        delta_state = jax.tree_util.tree_map(
            lambda current, reference: current - reference,
            state,
            lagged_response.reference_state,
        )

        def _face_fluxes_from_state(state_value):
            face_state_value = build_face_transport_state(
                state_value,
                self.field,
                bc_density=bc_density,
                bc_temperature=bc_temperature,
            )
            return self.evaluate_face_fluxes(
                state_value,
                face_state_value,
                bc_density=bc_density,
                bc_temperature=bc_temperature,
            )

        tangent_face_flux = jax.jvp(
            _face_fluxes_from_state,
            (lagged_response.reference_state,),
            (delta_state,),
        )[1]
        face_fluxes = jax.tree_util.tree_map(
            lambda reference, tangent: reference + tangent,
            lagged_response.reference_face_flux,
            tangent_face_flux,
        )
        return self._fluxes_from_face_fluxes(face_fluxes)

    def build_local_particle_flux_evaluator(self, state):
        fluxes = self(state)
        gamma_turb = fluxes["Gamma"]

        def evaluator(radius_index, er_value):
            del er_value
            return gamma_turb[:, radius_index]

        return evaluator


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
                              include_turbulent_particle_flux: bool = True,
                              geometry: Any = None,
                              center_flux_mode: str = "direct") -> CombinedTransportFluxModel:
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
        geometry=geometry,
        center_flux_mode=center_flux_mode,
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
    "turbulent_relu_analytical",
    lambda species,
    grid,
    field,
    chi_t,
    chi_n,
    pressure_source_model=None,
    total_power_mw=None,
    density_critical_gradient=1.0,
    temperature_critical_gradient=1.0,
    density_relu_slope=1.0,
    temperature_relu_slope=1.0,
    relu_power=1.0: ReLUAnalyticalTurbulentTransportModel(
        species=species,
        field=field,
        density_critical_gradient=density_critical_gradient,
        temperature_critical_gradient=temperature_critical_gradient,
        density_relu_slope=density_relu_slope,
        temperature_relu_slope=temperature_relu_slope,
        relu_power=relu_power,
    ),
)

register_transport_flux_model(
    "turbulent_power_relu_analytical",
    get_transport_flux_model("turbulent_relu_analytical"),
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
