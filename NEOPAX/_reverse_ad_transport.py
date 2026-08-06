"""Reusable transport reverse-AD report-builder helpers.

This module owns production-facing transport reverse-AD seams that have been
lifted out of benchmark reporting. It provides the compact segmented transport
cotangent path and the VMEC raw-block payload pullback used by optimization
callers, while keeping report formatting outside the AD graph.
"""

from __future__ import annotations

import copy
import contextlib
import dataclasses
import io
import inspect
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ._constants import elementary_charge
from ._geometry_autodiff import (
    boundary_param_entries,
    build_neopax_geometry_and_ntx_exact_lij_support_from_state,
    build_geometry_autodiff_context,
    GeometryRawBlockSolve,
    geometry_payload_pullback_from_param_vector_raw_block_transpose,
)
from ._orchestrator import prepare_transport_solver_components
from ._profiles import AnalyticalProfileModel
from ._reverse_ad_initial_er import (
    compact_initial_er_ntx_support_pullback_leaves,
    compact_initial_er_state_pullback,
    find_ntx_support_payload,
    initial_er_charge_flux_residual_er_derivative,
    initial_er_charge_flux_residual_scalar,
    initial_er_charge_flux_residuals,
    initial_er_selected_root_profile,
    runtime_with_geometry_payload,
    runtime_with_ntx_support_payload,
)
from ._reverse_ad_parameters import (
    PROFILE_PARAMETER_ORDER,
    ReverseADParameterSet,
    discover_vmec_boundary_parameter_specs,
    normalize_vmec_boundary_families,
    vmec_boundary_tuples,
)
from ._transport_flux_models import (
    DENSITY_STATE_TO_PHYSICAL,
    PRESSURE_SOURCE_STATE_TO_MW_M3,
    _add_float_delta_tree,
    _float_delta_tree_like,
    _sanitize_float_delta_bar_tree,
)
from ._transport_solvers import (
    RADAUSolver,
    NewtonThetaMethodSolver,
    ThetaMethodSolver,
    _RadauAcceptedStepReducedCotangent,
    _ThetaStepState,
    _build_prepared_radau_accepted_rollout,
    _build_prepared_radau_execution_context,
    _extract_fixed_temperature_projection,
    _extract_state_regularization,
    _flat_rhs_factory,
    _flat_rhs_build_support_pullback_factory,
    _flat_rhs_lagged_response_pullback_factory,
    _flat_rhs_lagged_response_support_pullback_factory,
    _flat_rhs_state_pullback_factory,
    _flat_rhs_with_lagged_response_factory,
    _lagged_response_hooks,
    _make_radau_initial_step_state,
    _make_solver_state_transform,
    _radau_adaptive_schedule_rollout,
    _theta_basic_accepted_step_attempt,
    _theta_basic_adaptive_schedule_rollout,
    _theta_basic_step_from_attempt_fn,
    _theta_initial_reuse_state,
    _theta_make_attempt_context,
    _theta_newton_accepted_step_attempt,
    _theta_newton_adaptive_schedule_rollout,
    _theta_newton_step_from_attempt_fn,
    _theta_prepare_lagged_response,
    _project_flat_state_if_needed,
    _radau_adaptive_final_y_realized_schedule_vjp_fwd,
    _radau_align_tangent_tree_to_primal,
    _radau_carry_from_step_state,
    _radau_eval_rhs,
    _radau_segment_reduced_cotangent_bwd_batched_with_support_call,
    _radau_sanitize_support_delta_bar_tree,
    _radau_zero_support_delta_tree_like,
)


TransportReverseReport = Mapping[str, object]
TransportReverseReportRunner = Callable[[], TransportReverseReport]
TransportReverseSupportSegmentExecutor = Callable[[object, bool], TransportReverseReport]
TransportReverseSupportSegmentProbe = Callable[..., TransportReverseReport]
TransportReverseArgsUpdater = Callable[[object], object]
TransportReverseReportBuilder = Callable[
    [tuple[str, ...], ReverseADParameterSet, Mapping[str, object] | None],
    TransportReverseReport,
]
TransportReverseTableResultBuilder = Callable[
    [tuple[str, ...], ReverseADParameterSet, Mapping[str, object] | None],
    "RealtimeGeometryTransportReverseTableResult",
]
NtxSupportPayloadBuilder = Callable[[object], object]
ReverseStaticSetupBuilder = Callable[..., object]
GeometryDiagnosticsBuilder = Callable[[object], Mapping[str, object]]


@dataclasses.dataclass(frozen=True, slots=True)
class RealtimeGeometryTransportReverseTableContext:
    """Runtime inputs required by the grouped realtime-geometry reverse table."""

    config: Mapping[str, Any]
    baseline_values: object
    baseline_runtime: object
    baseline_state: object
    profile_cfg: Mapping[str, Any]
    neoclassical_cfg: Mapping[str, Any]


@dataclasses.dataclass(frozen=True, slots=True)
class RealtimeGeometryTransportReverseTableRequest:
    """Non-printing grouped transport reverse table request.

    This request object is intentionally small and benchmark-agnostic.  It
    gives the production lane a stable handoff point before the heavy grouped
    runner itself is moved out of the benchmark.
    """

    objective_names: tuple[str, ...]
    parameter_set: ReverseADParameterSet
    context: RealtimeGeometryTransportReverseTableContext
    options: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "objective_names",
            normalize_transport_objective_names(self.objective_names),
        )
        _validate_transport_reverse_parameter_set(self.parameter_set)
        if not isinstance(self.context, RealtimeGeometryTransportReverseTableContext):
            raise TypeError(
                "context must be a RealtimeGeometryTransportReverseTableContext; "
                f"got {type(self.context).__name__}."
            )


@dataclasses.dataclass(frozen=True, slots=True)
class RealtimeGeometryTransportReverseTableResult:
    """JAX-native grouped transport reverse table result.

    This object is the optimization-facing result boundary: it keeps objective
    values and Jacobian blocks as device arrays until a benchmark/reporting
    layer explicitly asks for host dictionaries.
    """

    objective_labels: tuple[str, ...]
    profile_parameter_labels: tuple[str, ...]
    geometry_parameter_labels: tuple[str, ...]
    objective_values: object
    profile_gradient_matrix: object
    geometry_gradient_matrix: object

    def __post_init__(self) -> None:
        objective_names = tuple(str(name) for name in self.objective_labels)
        profile_names = tuple(str(name) for name in self.profile_parameter_labels)
        geometry_names = tuple(str(name) for name in self.geometry_parameter_labels)
        if not objective_names or any(not name for name in objective_names):
            raise ValueError("objective_labels must contain at least one non-empty label.")
        if len(set(objective_names)) != len(objective_names):
            raise ValueError(f"objective_labels must be unique; got {objective_names!r}.")
        if len(set(profile_names)) != len(profile_names):
            raise ValueError(f"profile_parameter_labels must be unique; got {profile_names!r}.")
        if len(set(geometry_names)) != len(geometry_names):
            raise ValueError(f"geometry_parameter_labels must be unique; got {geometry_names!r}.")
        object.__setattr__(self, "objective_labels", objective_names)
        object.__setattr__(self, "profile_parameter_labels", profile_names)
        object.__setattr__(self, "geometry_parameter_labels", geometry_names)
        objective_values_shape = tuple(getattr(self.objective_values, "shape", ()))
        profile_gradient_shape = tuple(getattr(self.profile_gradient_matrix, "shape", ()))
        geometry_gradient_shape = tuple(getattr(self.geometry_gradient_matrix, "shape", ()))
        if objective_values_shape != (len(objective_names),):
            raise ValueError(
                "objective_values must have shape (objective_count,); "
                f"got {objective_values_shape}, objective_count={len(objective_names)}."
            )
        if profile_gradient_shape != (len(objective_names), len(profile_names)):
            raise ValueError(
                "profile_gradient_matrix must have shape "
                "(objective_count, profile_parameter_count); "
                f"got {profile_gradient_shape}, expected {(len(objective_names), len(profile_names))}."
            )
        if geometry_gradient_shape != (len(objective_names), len(geometry_names)):
            raise ValueError(
                "geometry_gradient_matrix must have shape "
                "(objective_count, geometry_parameter_count); "
                f"got {geometry_gradient_shape}, expected {(len(objective_names), len(geometry_names))}."
            )


@dataclasses.dataclass(frozen=True, slots=True)
class RealtimeGeometryPayloadPullbackResult:
    """VMEC-harmonic pullback result for transport support-payload cotangents."""

    geometry_gradient_matrix: object
    geometry_branch_gradient_matrix: object | None
    ntx_support_branch_gradient_matrix: object | None
    component_gradient_matrices: Mapping[str, object]
    component_geometry_branch_matrices: Mapping[str, object]
    component_ntx_support_branch_matrices: Mapping[str, object]
    pullback_mode: str = "payload_state_raw_block_transpose"


@dataclasses.dataclass(frozen=True, slots=True)
class RealtimeGeometryTransportReverseAssemblyResult:
    """Non-printing grouped transport table plus VMEC payload pullback result."""

    table_result: RealtimeGeometryTransportReverseTableResult
    payload_pullback_result: RealtimeGeometryPayloadPullbackResult


@dataclasses.dataclass(frozen=True, slots=True)
class RealtimeGeometryTransportReverseGroupedInputs:
    """Context plus grouped runner for realtime-geometry transport reverse tables."""

    table_context: RealtimeGeometryTransportReverseTableContext
    run_grouped_report: TransportReverseReportRunner


@dataclasses.dataclass(frozen=True, slots=True)
class RealtimeGeometrySupportSegmentCoreSetup:
    """Reusable setup for the realtime-geometry segmented support reverse core."""

    combined_geometry_payload: bool
    ntx_surface_backend: str
    ntx_support_payload: object
    support_payload: object
    profile_values: object
    support_probe_cotangent_mode: str
    reverse_setup: object
    early_geometry_diagnostics: Mapping[str, object] | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class RealtimeGeometrySupportCotangentResult:
    """Grouped transport profile/support cotangents before VMEC payload pullback."""

    objective_values: object
    profile_gradient_matrix: object
    support_bars: object
    support_component_bars_by_name: Mapping[str, object]
    support_reuse_count: int
    support_rebuild_count: int
    initial_cache_pullback_used: bool
    initial_cache_pullback_skipped: bool


@dataclasses.dataclass(frozen=True, slots=True)
class RealtimeGeometrySupportReverseDependencies:
    """Benchmark-supplied callbacks for the migrated support reverse kernel."""

    initial_er_root_enabled: Callable[[Mapping[str, Any], str], bool]
    initial_state_for_parameter_vector: Callable[..., object]
    state_with_initial_er_root_ad: Callable[..., object]
    reverse_initial_carry_from_state_with_static_setup: Callable[..., object]
    objective_scalar_by_index: Callable[[object, object, int], object]
    add_trees: Callable[[object, object], object]
    initial_er_selected_root_profile: Callable[..., object]
    initial_er_charge_flux_residuals: Callable[..., object]
    initial_er_charge_flux_residual_scalar: Callable[..., object]
    initial_er_charge_flux_residual_er_derivative: Callable[..., object]
    compact_initial_er_state_pullback: Callable[..., object]
    compact_initial_er_ntx_support_pullback_leaves: Callable[..., object]
    runtime_with_geometry_payload: Callable[[object, object], object]
    runtime_with_ntx_support_payload: Callable[[object, object], object]

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if not callable(value):
                raise TypeError(f"{field.name} must be callable.")


@dataclasses.dataclass(frozen=True, slots=True)
class RealtimeGeometryReverseStaticSetup:
    """Static solver reverse setup shared by benchmarks and optimization callers."""

    solver: object
    solve_vector_field: object
    prepared_rollout: object
    execution_context: object
    stop_after_accepted_steps: int | None
    max_total_steps: int
    reverse_segment_length: int | None
    require_final_time: bool = False


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class _ThetaReverseCarry:
    t: Any
    y: Any
    lagged_response_cache: Any
    lagged_response_valid: Any
    lagged_reference_y: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class _ThetaAcceptedStepReducedCotangent:
    y: Any
    lagged_response_cache: Any
    lagged_reference_y: Any


@dataclasses.dataclass(frozen=True, eq=False)
class _ThetaReversePhysicsContext:
    unpack_flat: Any
    pack_flat: Any = None
    project_flat: Any = None
    flat_rhs: Any = None
    flat_rhs_with_lagged_response: Any = None
    pullback_build_lagged_response: Any = None
    flat_rhs_lagged_response_pullback: Any = None
    flat_rhs_state_pullback: Any = None
    flat_rhs_build_support_pullback: Any = None
    flat_rhs_lagged_response_support_pullback: Any = None
    reverse_lagged_branch_schedule: tuple[bool, ...] | None = None


@dataclasses.dataclass(frozen=True, eq=False)
class _ThetaPreparedReverseRollout:
    solver: Any
    state: Any
    vector_field: Any
    species: Any
    temperature_active_mask: Any
    fixed_temperature_profile: Any
    density_floor: Any
    temperature_floor: Any
    initial_carry: _ThetaReverseCarry
    physics_context: _ThetaReversePhysicsContext


@dataclasses.dataclass(frozen=True, eq=False)
class _ThetaReverseExecutionContext:
    solver: Any
    prepared_rollout: _ThetaPreparedReverseRollout
    physics_context: _ThetaReversePhysicsContext


@dataclasses.dataclass(frozen=True, eq=False)
class _ThetaReverseScheduleTrace:
    accepted_mask: Any
    active_mask: Any
    t_start: Any
    y_start: Any
    lagged_response_cache_start: Any
    lagged_response_valid_start: Any
    err_norms: Any
    attempted_dts: Any
    next_dts: Any
    step_ts: Any
    next_recent_reject_count: Any
    next_regrowth_cooldown: Any
    next_easy_growth_streak: Any
    next_lagged_response_valid: Any


@dataclasses.dataclass(frozen=True, eq=False)
class _ThetaReverseScheduleRolloutResult:
    final_step_state: Any
    final_carry: _ThetaReverseCarry
    trace: _ThetaReverseScheduleTrace
    attempt_count: Any
    accepted_count: Any
    completed: Any
    failed: Any
    fail_code: Any


TRANSPORT_REVERSE_OBJECTIVE_LABELS: tuple[str, ...] = (
    "softmax_Er",
    "smooth_root_proxy",
    "Er_transition_left",
    "Er_transition_right",
    "Er2_volume_average",
    "Er_volume_average",
    "electron_temperature_volume_average_keV",
    "total_pressure_volume_average",
    "alpha_power_volume_average_mw_m3",
    "bootstrap_current_softmax_abs_scaled",
)
TRANSPORT_REVERSE_PROFILE_PARAMETER_ORDER: tuple[str, ...] = (
    "n0",
    "T0",
    "density_shape_power",
    "temperature_shape_power",
)


def initial_er_root_ad_mode(value: str | None) -> str:
    mode = str(value or "off").strip().lower()
    aliases = {
        "none": "off",
        "false": "off",
        "0": "off",
        "jax": "jax_selected_root",
        "selected_jax": "jax_selected_root",
        "jax_selected": "jax_selected_root",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"off", "jax_selected_root"}:
        raise ValueError("initial_er_root_ad must be one of: off, jax_selected_root")
    return mode


def initial_er_root_enabled(config: Mapping[str, Any], mode: str) -> bool:
    mode = initial_er_root_ad_mode(mode)
    if mode == "off":
        return False
    profiles_cfg = config.get("profiles", {})
    init_mode = str(profiles_cfg.get("er_initialization_mode", "analytical")).strip().lower()
    return init_mode in {
        "ambipolar_min_entropy",
        "ambipolar_best_root",
        "ambipolarity_best_root",
    }


def add_trees(lhs, rhs):
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs
    return jax.tree_util.tree_map(lambda a, b: a + b, lhs, rhs)


def parameterized_profile_set(
    profile_cfg: Mapping[str, Any],
    geometry,
    n_species: int,
    *,
    parameter_name: str,
    parameter_value,
):
    cfg = dict(profile_cfg)
    cfg[parameter_name] = parameter_value
    model = AnalyticalProfileModel(
        n0=cfg.get("n0", 4.21),
        n_edge=cfg.get("n_edge", 0.6),
        T0=cfg.get("T0", 17.8),
        T_edge=cfg.get("T_edge", 0.7),
        c_density=None if cfg.get("c_density") is None else tuple(cfg.get("c_density")),
        c_temperature=None if cfg.get("c_temperature") is None else tuple(cfg.get("c_temperature")),
        density_shape_power=cfg.get("density_shape_power", 2.0),
        density_shape_alpha=cfg.get("density_shape_alpha", 1.0),
        temperature_shape_power=cfg.get("temperature_shape_power", 2.0),
        temperature_shape_alpha=cfg.get("temperature_shape_alpha", 1.0),
        n_scale=cfg.get("n_scale", 1.0),
        T_scale=cfg.get("T_scale", 1.0),
        er0_scale=cfg.get("er0_scale", 100.0),
        er0_peak_rho=cfg.get("er0_peak_rho", 0.8),
        charge_qp=None if cfg.get("charge_qp") is None else tuple(cfg.get("charge_qp")),
    )
    return model.build(geometry, n_species)


def initial_state_for_parameter_vector(
    parameter_values,
    *,
    baseline_state,
    profile_cfg: Mapping[str, Any],
    runtime,
    config: Mapping[str, Any] | None = None,
    initial_er_root_ad: str = "off",
):
    cfg = dict(profile_cfg)
    values_arr = jnp.asarray(parameter_values)
    if int(values_arr.shape[0]) == len(PROFILE_PARAMETER_ORDER):
        parameter_order = PROFILE_PARAMETER_ORDER
    else:
        parameter_order = TRANSPORT_REVERSE_PROFILE_PARAMETER_ORDER
    for name, value in zip(parameter_order, values_arr):
        cfg[name] = value
    profile_set = parameterized_profile_set(
        cfg,
        runtime.geometry,
        runtime.species.number_species,
        parameter_name=TRANSPORT_REVERSE_PROFILE_PARAMETER_ORDER[0],
        parameter_value=cfg[TRANSPORT_REVERSE_PROFILE_PARAMETER_ORDER[0]],
    )
    density_state = jnp.asarray(profile_set.density, dtype=baseline_state.density.dtype) / 1.0e20
    temperature_state = jnp.asarray(profile_set.temperature, dtype=baseline_state.pressure.dtype) / 1.0e3
    pressure_state = density_state * temperature_state
    state = dataclasses.replace(
        baseline_state,
        density=density_state,
        pressure=pressure_state,
    )
    mode = initial_er_root_ad_mode(initial_er_root_ad)
    if mode != "off":
        if config is None:
            raise ValueError("config is required when initial_er_root_ad is enabled.")
        state = state_with_initial_er_root_ad(state, config=config, runtime=runtime, mode=mode)
    return state


def initial_er_root_state_bar(state, er_profile, finite_mask, state_bar, *, runtime):
    dres_der = initial_er_charge_flux_residual_er_derivative(
        state,
        er_profile,
        runtime=runtime,
    )
    safe_dres_der = jnp.where(
        jnp.abs(dres_der) > jnp.asarray(1.0e-30, dtype=dres_der.dtype),
        dres_der,
        jnp.inf,
    )
    residual_bar = jnp.where(
        finite_mask,
        -jnp.asarray(state_bar.Er) / safe_dres_der,
        0.0,
    )
    state_residual_bar = compact_initial_er_state_pullback(
        residual_scalar_fn=initial_er_charge_flux_residual_scalar,
        state=state,
        er_profile=er_profile,
        residual_bars=residual_bar,
        runtime=runtime,
    )
    direct_bar = dataclasses.replace(state_bar, Er=jnp.zeros_like(state.Er))
    return add_trees(direct_bar, state_residual_bar)


def state_with_initial_er_root_ad(state, *, config: Mapping[str, Any], runtime, mode: str):
    if not initial_er_root_enabled(config, mode):
        return state

    @jax.custom_vjp
    def _replace_er_with_selected_root(state_inner):
        er_profile, _ = initial_er_selected_root_profile(state_inner, config=dict(config), runtime=runtime)
        return dataclasses.replace(state_inner, Er=er_profile)

    def _replace_er_fwd(state_inner):
        er_profile, finite_mask = initial_er_selected_root_profile(
            state_inner,
            config=dict(config),
            runtime=runtime,
        )
        return dataclasses.replace(state_inner, Er=er_profile), (state_inner, er_profile, finite_mask)

    def _replace_er_bwd(residuals, state_bar):
        state_inner, er_profile, finite_mask = residuals
        return (
            initial_er_root_state_bar(
                state_inner,
                er_profile,
                finite_mask,
                state_bar,
                runtime=runtime,
            ),
        )

    _replace_er_with_selected_root.defvjp(_replace_er_fwd, _replace_er_bwd)
    return _replace_er_with_selected_root(state)


def softmax_objective(er_profile: jax.Array, *, beta: float = 16.0) -> jax.Array:
    beta_arr = jnp.asarray(beta, dtype=er_profile.dtype)
    return jax.scipy.special.logsumexp(beta_arr * er_profile) / beta_arr


def smooth_root_proxy(
    er_profile: jax.Array,
    rho_grid: jax.Array,
    *,
    beta: float = 24.0,
    eps: float = 1.0e-4,
):
    beta_arr = jnp.asarray(beta, dtype=er_profile.dtype)
    eps_arr = jnp.asarray(eps, dtype=er_profile.dtype)
    smooth_abs = jnp.sqrt(er_profile * er_profile + eps_arr * eps_arr)
    weights = jnp.exp(-beta_arr * smooth_abs)
    return jnp.sum(rho_grid * weights) / jnp.maximum(jnp.sum(weights), jnp.asarray(1.0e-30, dtype=er_profile.dtype))


def volume_average(profile: jax.Array, geometry) -> jax.Array:
    volume = jnp.trapezoid(jnp.asarray(geometry.Vprime), x=jnp.asarray(geometry.r_grid))
    integral = jnp.trapezoid(profile * jnp.asarray(geometry.Vprime), x=jnp.asarray(geometry.r_grid))
    return integral / jnp.maximum(volume, jnp.asarray(1.0e-30, dtype=integral.dtype))


def alpha_power_volume_average(final_state, runtime) -> jax.Array:
    source_models = runtime.models.source or {}
    pressure_source_model = source_models.get("temperature") if isinstance(source_models, dict) else None
    if pressure_source_model is None:
        return jnp.asarray(0.0, dtype=final_state.pressure.dtype)
    raw_sources = pressure_source_model(final_state)
    alpha_power = raw_sources.get("AlphaPower") if isinstance(raw_sources, dict) else None
    if alpha_power is None:
        return jnp.asarray(0.0, dtype=final_state.pressure.dtype)
    alpha_mw_m3 = PRESSURE_SOURCE_STATE_TO_MW_M3 * jnp.asarray(alpha_power, dtype=final_state.pressure.dtype)
    return volume_average(alpha_mw_m3, runtime.geometry)


def electron_temperature_volume_average(final_state, runtime) -> jax.Array:
    species_idx = getattr(runtime.species, "species_idx", {})
    electron_idx = species_idx.get("e", 0)
    temperature = jnp.asarray(final_state.temperature[electron_idx], dtype=final_state.pressure.dtype)
    return volume_average(temperature, runtime.geometry)


def total_pressure_volume_average(final_state, runtime) -> jax.Array:
    total_pressure = jnp.sum(jnp.asarray(final_state.pressure, dtype=final_state.pressure.dtype), axis=0)
    return volume_average(total_pressure, runtime.geometry)


def bootstrap_current_softmax_abs_scaled(
    final_state,
    runtime,
    *,
    beta: float = 128.0,
    eps: float = 1.0e-12,
) -> jax.Array:
    """Direct momentum-corrected smooth max(abs(Jboot)).

    A value of 0.1 corresponds to 10 kA/m^2 in physical current density.
    """

    flux_model = getattr(getattr(runtime, "models", None), "flux", None)
    neoclassical_model = getattr(flux_model, "neoclassical_model", flux_model)
    corrected_fluxes_fn = getattr(neoclassical_model, "evaluate_momentum_corrected_fluxes", None)
    if not callable(corrected_fluxes_fn):
        raise NotImplementedError(
            "bootstrap_current_softmax_abs_scaled requires a realtime NTX model with "
            "evaluate_momentum_corrected_fluxes; refusing to use the static database "
            "or uncorrected lagged Upar path."
        )
    corrected_fluxes = corrected_fluxes_fn(final_state)
    upar = corrected_fluxes.get("Upar_neo", corrected_fluxes.get("Upar", None))
    if upar is None:
        raise ValueError("momentum-corrected realtime NTX fluxes did not return Upar.")
    charge_qp = jnp.asarray(runtime.species.charge_qp, dtype=final_state.pressure.dtype)
    current_weights = jnp.sign(charge_qp)
    upar_arr = jnp.asarray(upar, dtype=final_state.pressure.dtype)
    upar_physical = jnp.asarray(DENSITY_STATE_TO_PHYSICAL, dtype=upar_arr.dtype) * upar_arr
    if int(upar_arr.shape[0]) == int(charge_qp.shape[0]):
        jboot = jnp.sum(upar_physical * current_weights[:, None], axis=0)
    else:
        jboot = jnp.sum(upar_physical * current_weights[None, :], axis=1)
    jboot = jboot * jnp.asarray(elementary_charge * 1.0e-5, dtype=final_state.pressure.dtype)
    smooth_abs = jnp.sqrt(jboot * jboot + jnp.asarray(eps, dtype=jboot.dtype) ** 2)
    beta_arr = jnp.asarray(beta, dtype=jboot.dtype)
    return jax.scipy.special.logsumexp(beta_arr * smooth_abs) / beta_arr


def bootstrap_current_softmax_abs_value_and_upar_bar(
    final_state,
    runtime,
    fluxes: Mapping[str, Any],
    *,
    beta: float = 128.0,
    eps: float = 1.0e-12,
) -> tuple[jax.Array, jax.Array]:
    """Return smooth max(abs(Jboot)) and the compact corrected-Upar cotangent."""

    upar = fluxes.get("Upar_neo", fluxes.get("Upar", None))
    if upar is None:
        raise ValueError("bootstrap current objective requires Upar or Upar_neo fluxes.")
    dtype = jnp.asarray(final_state.pressure).dtype
    charge_qp = jnp.asarray(runtime.species.charge_qp, dtype=dtype)
    current_weights = jnp.sign(charge_qp)
    upar_arr = jnp.asarray(upar, dtype=dtype)
    scale = jnp.asarray(elementary_charge * 1.0e-5, dtype=dtype)
    upar_physical_scale = jnp.asarray(DENSITY_STATE_TO_PHYSICAL, dtype=dtype)
    upar_physical = upar_physical_scale * upar_arr
    if int(upar_arr.shape[0]) == int(charge_qp.shape[0]):
        jboot = jnp.sum(upar_physical * current_weights[:, None], axis=0) * scale
        species_axis_first = True
    else:
        jboot = jnp.sum(upar_physical * current_weights[None, :], axis=1) * scale
        species_axis_first = False

    smooth_abs = jnp.sqrt(jboot * jboot + jnp.asarray(eps, dtype=dtype) ** 2)
    beta_arr = jnp.asarray(beta, dtype=dtype)
    value = jax.scipy.special.logsumexp(beta_arr * smooth_abs) / beta_arr
    smooth_abs_bar = jax.nn.softmax(beta_arr * smooth_abs)
    jboot_bar = smooth_abs_bar * jboot / jnp.maximum(smooth_abs, jnp.asarray(1.0e-30, dtype=dtype))
    if species_axis_first:
        upar_bar = current_weights[:, None] * (upar_physical_scale * scale * jboot_bar)[None, :]
    else:
        upar_bar = (upar_physical_scale * scale * jboot_bar)[:, None] * current_weights[None, :]
    return value, upar_bar


def objective_scalar_by_index(final_state, runtime, objective_index: int):
    objective_name = TRANSPORT_REVERSE_OBJECTIVE_LABELS[int(objective_index)]
    er = jnp.asarray(final_state.Er)
    if objective_name == "softmax_Er":
        return softmax_objective(er)
    if objective_name == "smooth_root_proxy":
        rho = jnp.asarray(runtime.geometry.rho_grid, dtype=er.dtype)
        return smooth_root_proxy(er, rho)
    if objective_name == "Er_transition_left":
        return er[max(0, min(20, int(er.shape[-1]) - 1))]
    if objective_name == "Er_transition_right":
        return er[max(0, min(21, int(er.shape[-1]) - 1))]
    if objective_name == "Er2_volume_average":
        return volume_average(er * er, runtime.geometry)
    if objective_name == "Er_volume_average":
        return volume_average(er, runtime.geometry)
    if objective_name == "electron_temperature_volume_average_keV":
        return electron_temperature_volume_average(final_state, runtime)
    if objective_name == "total_pressure_volume_average":
        return total_pressure_volume_average(final_state, runtime)
    if objective_name == "alpha_power_volume_average_mw_m3":
        return alpha_power_volume_average(final_state, runtime)
    if objective_name == "bootstrap_current_softmax_abs_scaled":
        return bootstrap_current_softmax_abs_scaled(final_state, runtime)
    raise ValueError(f"Unknown objective index {objective_index}: {objective_name!r}")


def lagged_response_pullback_from_owner(solve_vector_field):
    owner = getattr(solve_vector_field, "__self__", None)
    if owner is None:
        return None
    pullback_fn = getattr(owner, "pullback_build_lagged_response", None)
    return pullback_fn if callable(pullback_fn) else None


def _require_reverse_solver_radau(solver, capability: str) -> None:
    if isinstance(solver, RADAUSolver):
        return
    solver_name = type(solver).__name__
    raise ValueError(
        f"{capability} currently requires RADAUSolver because the reverse path "
        "uses Radau-private accepted-step rollout/VJP/segment rules; "
        f"got {solver_name}. Add the theta implementation behind this solver-neutral "
        "dispatch layer before enabling theta for this reverse capability."
    )


def _require_reverse_execution_context_radau(execution_context, capability: str) -> None:
    kernel_context = getattr(execution_context, "kernel_context", None)
    if hasattr(kernel_context, "radau_transform"):
        return
    context_name = type(execution_context).__name__
    raise ValueError(
        f"{capability} currently requires a Radau reverse execution context because "
        "the reverse path uses Radau-private accepted-step rollout/VJP/segment rules; "
        f"got {context_name}. Add the theta implementation behind this solver-neutral "
        "dispatch layer before enabling theta for this reverse capability."
    )


def _is_theta_reverse_solver(solver) -> bool:
    return isinstance(solver, (ThetaMethodSolver, NewtonThetaMethodSolver))


def _build_prepared_theta_reverse_rollout(*, solver, state, vector_field, species):
    temperature_active_mask, fixed_temperature_profile = _extract_fixed_temperature_projection(vector_field)
    density_floor, temperature_floor = _extract_state_regularization(vector_field)
    flat_state0, unpack_flat, _unpack_packed, pack_flat, project_flat = _make_solver_state_transform(
        state,
        species,
        temperature_active_mask=temperature_active_mask,
        fixed_temperature_profile=fixed_temperature_profile,
        density_floor=density_floor,
        temperature_floor=temperature_floor,
    )
    args = (species,)
    kwargs = {}
    flat_rhs = _flat_rhs_factory(
        unpack_flat,
        vector_field,
        args,
        kwargs,
        project_flat=project_flat,
    )
    flat_rhs_with_lagged_response = _flat_rhs_with_lagged_response_factory(
        unravel=unpack_flat,
        vector_field=vector_field,
        args=args,
        kwargs=kwargs,
        project_flat=project_flat,
    )
    flat_rhs_lagged_response_pullback = _flat_rhs_lagged_response_pullback_factory(
        unpack_flat,
        vector_field,
        args,
        kwargs,
        project_flat=project_flat,
    )
    flat_rhs_state_pullback = _flat_rhs_state_pullback_factory(
        unpack_flat,
        pack_flat,
        vector_field,
        args,
        kwargs,
        project_flat=project_flat,
    )
    pullback_build_lagged_response = lagged_response_pullback_from_owner(vector_field)
    flat_rhs_build_support_pullback = _flat_rhs_build_support_pullback_factory(
        unpack_flat,
        vector_field,
        args,
        kwargs,
        project_flat=project_flat,
    )
    flat_rhs_lagged_response_support_pullback = _flat_rhs_lagged_response_support_pullback_factory(
        unpack_flat,
        vector_field,
        args,
        kwargs,
        project_flat=project_flat,
    )
    dtype = jnp.asarray(flat_state0).dtype
    initial_carry = _ThetaReverseCarry(
        t=jnp.asarray(getattr(solver, "t0", 0.0), dtype=dtype),
        y=flat_state0,
        lagged_response_cache=None,
        lagged_response_valid=jnp.asarray(False),
        lagged_reference_y=flat_state0,
    )
    return _ThetaPreparedReverseRollout(
        solver=solver,
        state=state,
        vector_field=vector_field,
        species=species,
        temperature_active_mask=temperature_active_mask,
        fixed_temperature_profile=fixed_temperature_profile,
        density_floor=density_floor,
        temperature_floor=temperature_floor,
        initial_carry=initial_carry,
        physics_context=_ThetaReversePhysicsContext(
            unpack_flat=unpack_flat,
            pack_flat=pack_flat,
            project_flat=project_flat,
            flat_rhs=flat_rhs,
            flat_rhs_with_lagged_response=flat_rhs_with_lagged_response,
            pullback_build_lagged_response=pullback_build_lagged_response,
            flat_rhs_lagged_response_pullback=flat_rhs_lagged_response_pullback,
            flat_rhs_state_pullback=flat_rhs_state_pullback,
            flat_rhs_build_support_pullback=flat_rhs_build_support_pullback,
            flat_rhs_lagged_response_support_pullback=flat_rhs_lagged_response_support_pullback,
        ),
    )


def _build_theta_reverse_execution_context(*, solver, prepared_rollout):
    return _ThetaReverseExecutionContext(
        solver=solver,
        prepared_rollout=prepared_rollout,
        physics_context=prepared_rollout.physics_context,
    )


def _theta_solver_with_reverse_probe_limits(solver, *, max_total_steps, stop_after_accepted_steps):
    replacements = {}
    if max_total_steps is not None:
        replacements["max_steps"] = int(max(1, max_total_steps))
    if stop_after_accepted_steps is not None:
        replacements["stop_after_accepted_steps"] = int(max(1, stop_after_accepted_steps))
    if not replacements:
        return solver
    init_params = inspect.signature(type(solver).__init__).parameters
    accepted_kinds = {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }
    solver_kwargs = {
        name: getattr(solver, name)
        for name, param in init_params.items()
        if name != "self"
        and hasattr(solver, name)
        and param.kind in accepted_kinds
    }
    solver_kwargs.update(replacements)
    return type(solver)(**solver_kwargs)


def _theta_reverse_adaptive_schedule_rollout(
    execution_context,
    initial_carry,
    *,
    max_total_steps,
    stop_after_accepted_steps,
):
    """Run theta's forward schedule probe and expose Radau-like metadata.

    Theta and Newton-theta use the shared theta attempt helpers directly so
    reverse setup can see real accepted/rejected schedule rows.  The remaining
    theta reverse work is the realized-schedule VJP/segmented cotangent pass.
    """

    prepared = execution_context.prepared_rollout
    solver = _theta_solver_with_reverse_probe_limits(
        execution_context.solver,
        max_total_steps=max_total_steps,
        stop_after_accepted_steps=stop_after_accepted_steps,
    )
    initial_state = prepared.physics_context.unpack_flat(initial_carry.y)
    if isinstance(solver, ThetaMethodSolver):
        rollout = _theta_basic_adaptive_schedule_rollout(
            solver,
            initial_state,
            prepared.vector_field,
            prepared.species,
            max_total_steps=max_total_steps,
            stop_after_accepted_steps=stop_after_accepted_steps,
        )
        final_carry = _ThetaReverseCarry(
            t=rollout.final_step_state.t,
            y=rollout.final_step_state.y,
            lagged_response_cache=rollout.final_step_state.reuse_state.lagged_response_cache,
            lagged_response_valid=rollout.final_step_state.reuse_state.lagged_response_valid,
            lagged_reference_y=rollout.final_step_state.reuse_state.lagged_reference_y,
        )
        trace = _ThetaReverseScheduleTrace(
            accepted_mask=rollout.trace.accepted_mask,
            active_mask=rollout.trace.active_mask,
            t_start=rollout.trace.t_start,
            y_start=rollout.trace.y_start,
            lagged_response_cache_start=rollout.trace.lagged_response_cache_start,
            lagged_response_valid_start=rollout.trace.lagged_response_valid_start,
            err_norms=rollout.trace.err_norms,
            attempted_dts=rollout.trace.attempted_dts,
            next_dts=rollout.trace.next_dts,
            step_ts=rollout.trace.step_ts,
            next_recent_reject_count=rollout.trace.next_recent_reject_count,
            next_regrowth_cooldown=rollout.trace.next_regrowth_cooldown,
            next_easy_growth_streak=rollout.trace.next_easy_growth_streak,
            next_lagged_response_valid=rollout.trace.next_lagged_response_valid,
        )
        return _ThetaReverseScheduleRolloutResult(
            final_step_state=rollout.final_step_state,
            final_carry=final_carry,
            trace=trace,
            attempt_count=rollout.attempt_count,
            accepted_count=rollout.accepted_count,
            completed=rollout.completed,
            failed=rollout.failed,
            fail_code=rollout.fail_code,
        )
    if isinstance(solver, NewtonThetaMethodSolver):
        rollout = _theta_newton_adaptive_schedule_rollout(
            solver,
            initial_state,
            prepared.vector_field,
            prepared.species,
            max_total_steps=max_total_steps,
            stop_after_accepted_steps=stop_after_accepted_steps,
        )
        final_carry = _ThetaReverseCarry(
            t=rollout.final_step_state.t,
            y=rollout.final_step_state.y,
            lagged_response_cache=rollout.final_step_state.reuse_state.lagged_response_cache,
            lagged_response_valid=rollout.final_step_state.reuse_state.lagged_response_valid,
            lagged_reference_y=rollout.final_step_state.reuse_state.lagged_reference_y,
        )
        trace = _ThetaReverseScheduleTrace(
            accepted_mask=rollout.trace.accepted_mask,
            active_mask=rollout.trace.active_mask,
            t_start=rollout.trace.t_start,
            y_start=rollout.trace.y_start,
            lagged_response_cache_start=rollout.trace.lagged_response_cache_start,
            lagged_response_valid_start=rollout.trace.lagged_response_valid_start,
            err_norms=rollout.trace.err_norms,
            attempted_dts=rollout.trace.attempted_dts,
            next_dts=rollout.trace.next_dts,
            step_ts=rollout.trace.step_ts,
            next_recent_reject_count=rollout.trace.next_recent_reject_count,
            next_regrowth_cooldown=rollout.trace.next_regrowth_cooldown,
            next_easy_growth_streak=rollout.trace.next_easy_growth_streak,
            next_lagged_response_valid=rollout.trace.next_lagged_response_valid,
        )
        return _ThetaReverseScheduleRolloutResult(
            final_step_state=rollout.final_step_state,
            final_carry=final_carry,
            trace=trace,
            attempt_count=rollout.attempt_count,
            accepted_count=rollout.accepted_count,
            completed=rollout.completed,
            failed=rollout.failed,
            fail_code=rollout.fail_code,
        )

    result = solver.solve(prepared.state, prepared.vector_field, prepared.species)
    final_flat, *_ = _make_solver_state_transform(
        result["final_state"],
        prepared.species,
        temperature_active_mask=prepared.temperature_active_mask,
        fixed_temperature_profile=prepared.fixed_temperature_profile,
        density_floor=prepared.density_floor,
        temperature_floor=prepared.temperature_floor,
    )
    accepted_count = jnp.asarray(result.get("n_steps", 0), dtype=jnp.int32)
    attempt_count_value = int(np.asarray(jax.device_get(accepted_count)))
    dtype = jnp.asarray(initial_carry.y).dtype
    active_mask = jnp.ones((attempt_count_value,), dtype=bool)
    accepted_mask = jnp.ones((attempt_count_value,), dtype=bool)
    zero_float_trace = jnp.zeros((attempt_count_value,), dtype=dtype)
    zero_int_trace = jnp.zeros((attempt_count_value,), dtype=jnp.int32)
    lagged_valid = result.get("final_reuse_lagged_response_valid", False)
    if lagged_valid is None:
        lagged_valid = False
    lagged_valid_trace = jnp.full((attempt_count_value,), lagged_valid, dtype=bool)
    final_carry = _ThetaReverseCarry(
        t=result["final_time"],
        y=final_flat,
        lagged_response_cache=None,
        lagged_response_valid=jnp.asarray(lagged_valid, dtype=bool),
        lagged_reference_y=final_flat,
    )
    trace = _ThetaReverseScheduleTrace(
        accepted_mask=accepted_mask,
        active_mask=active_mask,
        t_start=zero_float_trace,
        y_start=jnp.broadcast_to(final_flat[None, ...], (attempt_count_value,) + jnp.shape(final_flat)),
        lagged_response_cache_start=None,
        lagged_response_valid_start=lagged_valid_trace,
        err_norms=zero_float_trace,
        attempted_dts=zero_float_trace,
        next_dts=zero_float_trace,
        step_ts=zero_float_trace,
        next_recent_reject_count=zero_int_trace,
        next_regrowth_cooldown=zero_int_trace,
        next_easy_growth_streak=zero_int_trace,
        next_lagged_response_valid=lagged_valid_trace,
    )
    return _ThetaReverseScheduleRolloutResult(
        final_step_state=None,
        final_carry=final_carry,
        trace=trace,
        attempt_count=accepted_count,
        accepted_count=accepted_count,
        completed=result["done"],
        failed=result["failed"],
        fail_code=result["fail_code"],
    )


def _build_prepared_reverse_accepted_rollout(*, solver, state, vector_field, species):
    # Solver-neutral seam: theta support should be added here, not in callers.
    if _is_theta_reverse_solver(solver):
        return _build_prepared_theta_reverse_rollout(
            solver=solver,
            state=state,
            vector_field=vector_field,
            species=species,
        )
    _require_reverse_solver_radau(solver, "prepared reverse accepted rollout")
    return _build_prepared_radau_accepted_rollout(
        solver=solver,
        state=state,
        vector_field=vector_field,
        species=species,
    )


def _build_prepared_reverse_execution_context(*, solver, prepared_rollout):
    if _is_theta_reverse_solver(solver):
        return _build_theta_reverse_execution_context(
            solver=solver,
            prepared_rollout=prepared_rollout,
        )
    _require_reverse_solver_radau(solver, "prepared reverse execution context")
    return _build_prepared_radau_execution_context(
        solver=solver,
        prepared_rollout=prepared_rollout,
    )


def _reverse_adaptive_schedule_rollout(
    execution_context,
    initial_carry,
    *,
    max_total_steps,
    stop_after_accepted_steps,
):
    if isinstance(execution_context, _ThetaReverseExecutionContext):
        return _theta_reverse_adaptive_schedule_rollout(
            execution_context,
            initial_carry,
            max_total_steps=max_total_steps,
            stop_after_accepted_steps=stop_after_accepted_steps,
        )
    _require_reverse_execution_context_radau(execution_context, "reverse adaptive schedule rollout")
    return _radau_adaptive_schedule_rollout(
        execution_context,
        initial_carry,
        max_total_steps=max_total_steps,
        stop_after_accepted_steps=stop_after_accepted_steps,
    )


def _theta_adaptive_final_y_realized_schedule_vjp_fwd(
    execution_context,
    max_total_steps,
    stop_after_accepted_steps,
    reverse_segment_length,
    initial_carry,
):
    rollout = _theta_reverse_adaptive_schedule_rollout(
        execution_context,
        initial_carry,
        max_total_steps=max_total_steps,
        stop_after_accepted_steps=stop_after_accepted_steps,
    )
    active_mask = jax.lax.stop_gradient(rollout.trace.active_mask)
    accepted_mask = jax.lax.stop_gradient(rollout.trace.accepted_mask)
    attempted_dts = jax.lax.stop_gradient(rollout.trace.attempted_dts)
    next_dts = jax.lax.stop_gradient(rollout.trace.next_dts)
    next_recent_reject_count = jax.lax.stop_gradient(rollout.trace.next_recent_reject_count)
    next_regrowth_cooldown = jax.lax.stop_gradient(rollout.trace.next_regrowth_cooldown)
    next_easy_growth_streak = jax.lax.stop_gradient(rollout.trace.next_easy_growth_streak)
    next_lagged_response_valid = jax.lax.stop_gradient(rollout.trace.next_lagged_response_valid)
    segment_start_carries = None
    segmented_final_carry = None
    segmented_replay_arrays = None
    if reverse_segment_length is not None and int(reverse_segment_length) > 0:
        segment_length = int(reverse_segment_length)
        accepted_limit = int(stop_after_accepted_steps) if stop_after_accepted_steps is not None else int(max_total_steps)
        segment_count = (accepted_limit + segment_length - 1) // segment_length
        padded_count = segment_count * segment_length
        accepted_active_mask = jnp.logical_and(active_mask, accepted_mask)
        accepted_count = jnp.minimum(
            jnp.sum(accepted_active_mask.astype(jnp.int32)),
            jnp.asarray(accepted_limit, dtype=jnp.int32),
        )
        accepted_positions = jnp.nonzero(
            accepted_active_mask,
            size=accepted_limit,
            fill_value=0,
        )[0]

        def _compact_and_pad(values):
            compact = jnp.take(values, accepted_positions, axis=0)
            pad_count = padded_count - accepted_limit
            if pad_count == 0:
                return compact
            pad_values = jnp.repeat(compact[-1:], pad_count, axis=0)
            return jnp.concatenate([compact, pad_values], axis=0)

        def _compact_and_pad_tree(tree):
            if tree is None:
                return None
            return jax.tree_util.tree_map(_compact_and_pad, tree)

        replay_active_mask = jnp.concatenate(
            [
                jnp.arange(accepted_limit, dtype=jnp.int32) < accepted_count,
                jnp.zeros((padded_count - accepted_limit,), dtype=jnp.bool_),
            ],
            axis=0,
        )
        replay_attempted_dts = _compact_and_pad(attempted_dts)
        replay_next_dts = _compact_and_pad(next_dts)
        replay_next_recent_reject_count = _compact_and_pad(next_recent_reject_count)
        replay_next_regrowth_cooldown = _compact_and_pad(next_regrowth_cooldown)
        replay_next_easy_growth_streak = _compact_and_pad(next_easy_growth_streak)
        replay_next_lagged_response_valid = _compact_and_pad(next_lagged_response_valid)
        segmented_replay_arrays = (
            replay_active_mask.reshape((segment_count, segment_length)),
            replay_attempted_dts.reshape((segment_count, segment_length)),
            replay_next_dts.reshape((segment_count, segment_length)),
            replay_next_recent_reject_count.reshape((segment_count, segment_length)),
            replay_next_regrowth_cooldown.reshape((segment_count, segment_length)),
            replay_next_easy_growth_streak.reshape((segment_count, segment_length)),
            replay_next_lagged_response_valid.reshape((segment_count, segment_length)),
        )
        compact_start_carries = _ThetaReverseCarry(
            t=_compact_and_pad(jax.lax.stop_gradient(rollout.trace.t_start)),
            y=_compact_and_pad(jax.lax.stop_gradient(rollout.trace.y_start)),
            lagged_response_cache=_compact_and_pad_tree(
                jax.lax.stop_gradient(rollout.trace.lagged_response_cache_start)
            ),
            lagged_response_valid=_compact_and_pad(
                jax.lax.stop_gradient(rollout.trace.lagged_response_valid_start)
            ),
            lagged_reference_y=_compact_and_pad(jax.lax.stop_gradient(rollout.trace.y_start)),
        )
        segment_start_carries = _ThetaReverseCarry(
            t=compact_start_carries.t.reshape((segment_count, segment_length) + compact_start_carries.t.shape[1:])[:, 0],
            y=compact_start_carries.y.reshape((segment_count, segment_length) + compact_start_carries.y.shape[1:])[:, 0],
            lagged_response_cache=(
                None
                if compact_start_carries.lagged_response_cache is None
                else jax.tree_util.tree_map(
                    lambda value: value.reshape((segment_count, segment_length) + value.shape[1:])[:, 0],
                    compact_start_carries.lagged_response_cache,
                )
            ),
            lagged_response_valid=compact_start_carries.lagged_response_valid.reshape(
                (segment_count, segment_length) + compact_start_carries.lagged_response_valid.shape[1:]
            )[:, 0],
            lagged_reference_y=compact_start_carries.lagged_reference_y.reshape(
                (segment_count, segment_length) + compact_start_carries.lagged_reference_y.shape[1:]
            )[:, 0],
        )
        segmented_final_carry = rollout.final_carry
    residuals = (
        initial_carry,
        active_mask,
        accepted_mask,
        attempted_dts,
        next_dts,
        next_recent_reject_count,
        next_regrowth_cooldown,
        next_easy_growth_streak,
        next_lagged_response_valid,
        segment_start_carries,
        segmented_final_carry,
        segmented_replay_arrays,
    )
    return rollout.final_carry.y, residuals


def _reverse_adaptive_final_y_realized_schedule_vjp_fwd(
    execution_context,
    max_total_steps,
    stop_after_accepted_steps,
    reverse_segment_length,
    initial_carry,
):
    if isinstance(execution_context, _ThetaReverseExecutionContext):
        return _theta_adaptive_final_y_realized_schedule_vjp_fwd(
            execution_context,
            max_total_steps,
            stop_after_accepted_steps,
            reverse_segment_length,
            initial_carry,
        )
    _require_reverse_execution_context_radau(execution_context, "reverse realized-schedule VJP forward")
    return _radau_adaptive_final_y_realized_schedule_vjp_fwd(
        execution_context,
        max_total_steps,
        stop_after_accepted_steps,
        reverse_segment_length,
        initial_carry,
    )


def _reverse_zero_support_delta_tree_like(solver, support_payload):
    if _is_theta_reverse_solver(solver):
        return _radau_zero_support_delta_tree_like(support_payload)
    _require_reverse_solver_radau(solver, "reverse support-payload zero cotangent")
    return _radau_zero_support_delta_tree_like(support_payload)


def _reverse_align_tangent_tree_to_primal(execution_context, tangent_tree, primal_tree):
    if isinstance(execution_context, _ThetaReverseExecutionContext):
        return _radau_align_tangent_tree_to_primal(tangent_tree, primal_tree)
    _require_reverse_execution_context_radau(execution_context, "reverse tangent-tree alignment")
    return _radau_align_tangent_tree_to_primal(tangent_tree, primal_tree)


def _theta_lagged_rhs_support_pullback(
    physics_context,
    *,
    t_value,
    y_value,
    lagged_response,
    rhs_bar,
    support_payload,
):
    if (
        lagged_response is None
        or support_payload is None
        or physics_context.flat_rhs_lagged_response_support_pullback is None
    ):
        return _radau_zero_support_delta_tree_like(support_payload)
    return _radau_sanitize_support_delta_bar_tree(
        support_payload,
        physics_context.flat_rhs_lagged_response_support_pullback(
            t_value,
            y_value,
            lagged_response,
            rhs_bar,
            support_payload,
        ),
    )


def _theta_support_cotangent_for_vjp(primal, bar):
    def _sanitize_leaf(primal_leaf, bar_leaf):
        arr = jnp.asarray(primal_leaf)
        if jnp.issubdtype(arr.dtype, jnp.inexact):
            return jnp.asarray(bar_leaf, dtype=arr.dtype)
        return jnp.zeros(arr.shape, dtype=jax.dtypes.float0)

    return jax.tree_util.tree_map(_sanitize_leaf, primal, bar)


def _theta_segment_reduced_cotangent_bwd_batched_with_support_call(
    execution_context,
    cotangent_mode,
    reduced_bars,
    segment_start_carry,
    segment_arrays,
    support_payload,
):
    requested_mode = str(cotangent_mode).strip().lower()
    # Theta has a one-state implicit residual, so the normal "full" reverse lane
    # dispatches to the theta residual-transpose implementation directly.  The
    # theta_* names are kept only as diagnostics/aliases for isolating pieces.
    mode = "theta_implicit_transpose_probe" if requested_mode == "full" else requested_mode
    zero_step_bwd = mode in {"zero_step_bwd", "step_bwd_zero", "zero_accepted_step_bwd"}
    objective_count = jnp.asarray(reduced_bars.y).shape[0]
    zero_support_leaves = tuple(jax.tree_util.tree_leaves(_radau_zero_support_delta_tree_like(support_payload)))
    zero_support_bar_leaves = tuple(
        jnp.broadcast_to(jnp.asarray(leaf)[None, ...], (objective_count,) + jnp.asarray(leaf).shape)
        for leaf in zero_support_leaves
    )
    if zero_step_bwd:
        zero_reduced_bars = _ThetaAcceptedStepReducedCotangent(
            y=jnp.zeros(
                (objective_count,) + jnp.shape(segment_start_carry.y),
                dtype=jnp.asarray(segment_start_carry.y).dtype,
            ),
            lagged_response_cache=_reverse_align_tangent_tree_to_primal(
                execution_context,
                None,
                reduced_bars.lagged_response_cache,
            ),
            lagged_reference_y=jnp.zeros_like(reduced_bars.lagged_reference_y),
        )
        return zero_reduced_bars, zero_support_bar_leaves
    state_only_modes = {
        "state_only",
        "final_state",
        "theta_state_only",
        "theta_zero_lagged",
        "theta_compact_support_probe",
        "theta_implicit_transpose_probe",
    }
    if mode in state_only_modes:
        solver = execution_context.solver
        prepared = execution_context.prepared_rollout
        physics_context = prepared.physics_context
        unpack_flat = physics_context.unpack_flat
        vector_field = prepared.vector_field
        species = prepared.species
        dtype = jnp.asarray(segment_start_carry.y).dtype
        state_dim = jnp.asarray(segment_start_carry.y).shape[0]
        project_flat = physics_context.project_flat
        flat_rhs = physics_context.flat_rhs
        flat_rhs_with_lagged_response = physics_context.flat_rhs_with_lagged_response
        build_lagged_response, _ = _lagged_response_hooks(vector_field)
        rhs_mode = str(getattr(solver, "rhs_mode", "black_box")).strip().lower()
        use_lagged_linear_response = rhs_mode == "lagged_linear_state"
        use_transport_lagged_response = rhs_mode in {"lagged_transport_response", "lagged_response"}
        predictor_mode = getattr(solver, "predictor_mode", "linearized")
        theta = jnp.asarray(solver.theta_implicit, dtype=dtype)
        identity_n = jnp.eye(state_dim, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        t_final = jnp.asarray(solver.t1, dtype=dtype)

        def _make_theta_replay_step_fn(
            flat_rhs_replay,
            flat_rhs_with_lagged_response_replay,
            build_lagged_response_replay=build_lagged_response,
        ):
            if isinstance(solver, ThetaMethodSolver):
                n_linearized_solves = 1 + (solver.n_corrector_steps if solver.use_predictor_corrector else 0)

                def _attempt_fn(attempt_context):
                    return _theta_basic_accepted_step_attempt(
                        attempt_context,
                        predictor_mode=predictor_mode,
                        n_linearized_solves=n_linearized_solves,
                        theta=theta,
                        one=one,
                        identity_n=identity_n,
                        flat_rhs=flat_rhs_replay,
                        flat_rhs_with_lagged_response=flat_rhs_with_lagged_response_replay,
                        use_lagged_linear_response=use_lagged_linear_response,
                        project_flat=project_flat,
                        dtype=dtype,
                        tol=jnp.asarray(solver.tol, dtype=dtype),
                    )

                def _step_fn(step_state):
                    return _theta_basic_step_from_attempt_fn(
                        step_state,
                        attempt_fn=_attempt_fn,
                        t_final=t_final,
                        flat_rhs=flat_rhs_replay,
                        build_lagged_response=build_lagged_response_replay,
                        unpack_flat=unpack_flat,
                        project_flat=project_flat,
                        use_transport_lagged_response=use_transport_lagged_response,
                        dtype=dtype,
                    )

                return _step_fn

            n_linearized_solves = 1 + (solver.n_corrector_steps if solver.use_predictor_corrector else 0)
            lagged_response_reuse_mode = str(getattr(solver, "lagged_response_reuse_mode", "retry_only")).strip().lower()
            lagged_response_reuse_rtol = jnp.asarray(getattr(solver, "lagged_response_reuse_rtol", 5.0e-2), dtype=dtype)
            lagged_response_reuse_atol = jnp.asarray(getattr(solver, "lagged_response_reuse_atol", 1.0e-8), dtype=dtype)
            delta_reduction_factor = jnp.asarray(solver.delta_reduction_factor, dtype=dtype)
            tau_min = jnp.asarray(solver.tau_min, dtype=dtype)
            jacobian_reuse_rtol = jnp.asarray(getattr(solver, "jacobian_reuse_rtol", 0.1), dtype=dtype)
            max_jacobian_age = jnp.asarray(getattr(solver, "max_jacobian_age", 8), dtype=jnp.int32)
            freeze_attempt_linearization = str(
                getattr(solver, "jacobian_reuse_mode", "refresh_each_iteration")
            ).strip().lower() == "freeze_attempt"

            def _attempt_fn(attempt_context):
                return _theta_newton_accepted_step_attempt(
                    attempt_context,
                    predictor_mode=predictor_mode,
                    n_linearized_solves=n_linearized_solves,
                    theta=theta,
                    one=one,
                    identity_n=identity_n,
                    flat_rhs=flat_rhs_replay,
                    flat_rhs_with_lagged_response=flat_rhs_with_lagged_response_replay,
                    use_lagged_linear_response=use_lagged_linear_response,
                    use_transport_lagged_response=use_transport_lagged_response,
                    lagged_response_reuse_mode=lagged_response_reuse_mode,
                    jacobian_reuse_rtol=jacobian_reuse_rtol,
                    max_jacobian_age=max_jacobian_age,
                    delta_reduction_factor=delta_reduction_factor,
                    tau_min=tau_min,
                    project_flat=project_flat,
                    dtype=dtype,
                    tol=jnp.asarray(solver.tol, dtype=dtype),
                    maxiter=jnp.asarray(solver.maxiter, dtype=jnp.int32),
                )

            def _step_fn(step_state):
                return _theta_newton_step_from_attempt_fn(
                    step_state,
                    attempt_fn=_attempt_fn,
                    t_final=t_final,
                    flat_rhs=flat_rhs_replay,
                    build_lagged_response=build_lagged_response_replay,
                    unpack_flat=unpack_flat,
                    project_flat=project_flat,
                    use_transport_lagged_response=use_transport_lagged_response,
                    lagged_response_reuse_mode=lagged_response_reuse_mode,
                    lagged_response_reuse_rtol=lagged_response_reuse_rtol,
                    lagged_response_reuse_atol=lagged_response_reuse_atol,
                    dt_min=jnp.asarray(solver.min_step, dtype=dtype),
                    dt_max=jnp.asarray(solver.max_step, dtype=dtype),
                    safety_factor=jnp.asarray(solver.safety_factor, dtype=dtype),
                    min_step_factor=jnp.asarray(solver.min_step_factor, dtype=dtype),
                    max_step_factor=jnp.asarray(solver.max_step_factor, dtype=dtype),
                    controller_mode=str(getattr(solver, "controller_mode", "current")).strip().lower(),
                    delta_reduction_factor=delta_reduction_factor,
                    freeze_attempt_linearization=freeze_attempt_linearization,
                    dtype=dtype,
                )

            return _step_fn

        _step_fn = _make_theta_replay_step_fn(flat_rhs, flat_rhs_with_lagged_response)

        (
            active_mask,
            attempted_dts,
            next_dts,
            next_recent_reject_count,
            next_regrowth_cooldown,
            next_easy_growth_streak,
            next_lagged_response_valid,
        ) = segment_arrays

        if mode == "theta_implicit_transpose_probe":
            if not use_transport_lagged_response:
                raise NotImplementedError(
                    "theta_implicit_transpose_probe currently targets lagged transport-response theta runs."
                )

            def _initial_theta_step_state_from_carry(carry_value):
                reuse_state = dataclasses.replace(
                    _theta_initial_reuse_state(state_dim, dtype),
                    lagged_response_cache=carry_value.lagged_response_cache,
                    lagged_response_available=jnp.asarray(carry_value.lagged_response_cache is not None),
                    lagged_response_valid=carry_value.lagged_response_valid,
                    lagged_reference_y=carry_value.lagged_reference_y,
                )
                return _ThetaStepState(
                    t=carry_value.t,
                    y=carry_value.y,
                    dt=attempted_dts[0],
                    status=jnp.asarray([0, 0, 0], dtype=jnp.int32),
                    prev_error=jnp.asarray(1.0, dtype=dtype),
                    prev_dt=jnp.asarray(0.0, dtype=dtype),
                    recent_reject_count=jnp.asarray(0, dtype=jnp.int32),
                    regrowth_cooldown=jnp.asarray(0, dtype=jnp.int32),
                    easy_growth_streak=jnp.asarray(0, dtype=jnp.int32),
                    prev_theta_final=jnp.asarray(0.0, dtype=dtype),
                    prev_newton_iter_count=jnp.asarray(0, dtype=jnp.int32),
                    reuse_state=reuse_state,
                )

            def _collect_start_states(carry, slot_values):
                active, dt_value, next_dt_value, recent_reject, cooldown, streak, lagged_valid = slot_values
                carry = dataclasses.replace(carry, dt=dt_value)

                def _run(_):
                    next_state, _info = _step_fn(carry)
                    next_reuse = dataclasses.replace(
                        next_state.reuse_state,
                        lagged_response_valid=lagged_valid,
                    )
                    return dataclasses.replace(
                        next_state,
                        dt=next_dt_value,
                        recent_reject_count=recent_reject,
                        regrowth_cooldown=cooldown,
                        easy_growth_streak=streak,
                        reuse_state=next_reuse,
                    )

                return jax.lax.cond(active, _run, lambda _: carry, operand=None), carry

            _segment_final_state, step_start_states = jax.lax.scan(
                _collect_start_states,
                _initial_theta_step_state_from_carry(segment_start_carry),
                segment_arrays,
            )

            def _rhs_state_pullback(t_value, y_value, lagged_response_value, rhs_bar):
                if lagged_response_value is not None and physics_context.flat_rhs_state_pullback is not None:
                    return physics_context.flat_rhs_state_pullback(
                        t_value,
                        y_value,
                        lagged_response_value,
                        rhs_bar,
                    )

                def _rhs_from_y(y_inner):
                    return _radau_eval_rhs(
                        t_value,
                        y_inner,
                        lagged_response_value,
                        flat_rhs,
                        flat_rhs_with_lagged_response,
                    )

                _, rhs_pullback = jax.vjp(_rhs_from_y, y_value)
                (y_bar,) = rhs_pullback(rhs_bar)
                return y_bar

            def _rhs_lagged_pullback(t_value, y_value, lagged_response_value, rhs_bar):
                if lagged_response_value is None:
                    return None
                if physics_context.flat_rhs_lagged_response_pullback is not None:
                    return physics_context.flat_rhs_lagged_response_pullback(
                        t_value,
                        y_value,
                        lagged_response_value,
                        rhs_bar,
                    )

                def _rhs_from_lagged(lagged_inner):
                    return flat_rhs_with_lagged_response(t_value, y_value, lagged_inner)

                _, lagged_pullback = jax.vjp(_rhs_from_lagged, lagged_response_value)
                (lagged_bar,) = lagged_pullback(rhs_bar)
                return lagged_bar

            def _batched_zero_like_tree(tree):
                if tree is None:
                    return None
                return jax.tree_util.tree_map(
                    lambda leaf: jnp.zeros(
                        (objective_count,) + jnp.asarray(leaf).shape,
                        dtype=jnp.asarray(leaf).dtype,
                    ),
                    tree,
                )

            def _single_step_bwd(carry, slot_xs):
                slot_bars, support_bar_leaves = carry
                step_state, slot_values = slot_xs
                (
                    active,
                    dt_value,
                    _next_dt_value,
                    _recent_reject,
                    _cooldown,
                    _streak,
                    _lagged_valid,
                ) = slot_values

                def _zero_step(_):
                    return slot_bars, support_bar_leaves

                def _do_step(_):
                    carry_for_step = dataclasses.replace(step_state, dt=dt_value)
                    next_state, _info = _step_fn(carry_for_step)
                    h_value = jnp.asarray(dt_value, dtype=dtype)
                    t_old = carry_for_step.t
                    t_new = t_old + h_value
                    y_old = carry_for_step.y
                    y_new = next_state.y

                    if isinstance(solver, ThetaMethodSolver):
                        attempt_context = _theta_make_attempt_context(
                            carry_for_step,
                            t_final=t_final,
                            flat_rhs=flat_rhs,
                            build_lagged_response=build_lagged_response,
                            unpack_flat=unpack_flat,
                            project_flat=project_flat,
                            use_transport_lagged_response=use_transport_lagged_response,
                        )
                        lagged_response = attempt_context.lagged_response
                        lagged_response_reused = jnp.asarray(False)
                        lagged_reference_y = attempt_context.lagged_reference_y
                    else:
                        lagged_response, lagged_reference_y, lagged_response_reused = _theta_prepare_lagged_response(
                            carry_for_step,
                            use_transport_lagged_response=use_transport_lagged_response,
                            lagged_response_reuse_mode=lagged_response_reuse_mode,
                            lagged_response_reuse_rtol=lagged_response_reuse_rtol,
                            lagged_response_reuse_atol=lagged_response_reuse_atol,
                            unpack_flat=unpack_flat,
                            project_flat=project_flat,
                            build_lagged_response=build_lagged_response,
                        )

                    def _rhs_new(y_value):
                        return _radau_eval_rhs(
                            t_new,
                            y_value,
                            lagged_response,
                            flat_rhs,
                            flat_rhs_with_lagged_response,
                        )

                    jac_new = jax.jacfwd(_rhs_new)(y_new)
                    system_t = (jnp.eye(state_dim, dtype=dtype) - h_value * theta * jac_new).T
                    lambda_rows = jax.vmap(
                        lambda rhs: jnp.linalg.solve(system_t, jnp.asarray(rhs, dtype=dtype))
                    )(slot_bars.y)

                    f_old_coeff = h_value * (one - theta)
                    f_new_coeff = h_value * theta
                    old_rhs_bars = f_old_coeff * lambda_rows
                    new_rhs_bars = f_new_coeff * lambda_rows

                    old_rhs_y_bars = jax.vmap(
                        lambda rhs_bar: _rhs_state_pullback(t_old, y_old, None, rhs_bar)
                    )(old_rhs_bars)
                    y_old_bars = lambda_rows + old_rhs_y_bars

                    lagged_rhs_bars = jax.vmap(
                        lambda rhs_bar: _rhs_lagged_pullback(t_new, y_new, lagged_response, rhs_bar)
                    )(new_rhs_bars)
                    support_rhs_bars = jax.vmap(
                        lambda rhs_bar: _theta_lagged_rhs_support_pullback(
                            physics_context,
                            t_value=t_new,
                            y_value=y_new,
                            lagged_response=lagged_response,
                            rhs_bar=rhs_bar,
                            support_payload=support_payload,
                        )
                    )(new_rhs_bars)
                    support_bar_leaves_next = tuple(
                        accumulated + increment
                        for accumulated, increment in zip(
                            support_bar_leaves,
                            jax.tree_util.tree_leaves(support_rhs_bars),
                            strict=True,
                        )
                    )

                    total_lagged_bars = _radau_align_tangent_tree_to_primal(
                        slot_bars.lagged_response_cache,
                        lagged_response,
                    )
                    total_lagged_bars = jax.tree_util.tree_map(
                        lambda lhs, rhs: lhs + rhs,
                        total_lagged_bars,
                        lagged_rhs_bars,
                    )
                    reference_bars = jnp.asarray(slot_bars.lagged_reference_y, dtype=dtype)

                    def _reuse_lagged(_):
                        return (
                            y_old_bars,
                            _radau_align_tangent_tree_to_primal(
                                total_lagged_bars,
                                carry_for_step.reuse_state.lagged_response_cache,
                            ),
                            reference_bars,
                            support_bar_leaves_next,
                        )

                    def _rebuild_lagged(_):
                        if physics_context.pullback_build_lagged_response is not None:
                            projected_y = _project_flat_state_if_needed(y_old, project_flat)
                            rebuild_state = unpack_flat(projected_y)
                            rebuild_state_bars = jax.vmap(
                                lambda lagged_bar: physics_context.pullback_build_lagged_response(
                                    rebuild_state,
                                    lagged_bar,
                                )
                            )(total_lagged_bars)
                            rebuild_flat_bars = jax.vmap(physics_context.pack_flat)(rebuild_state_bars)
                            if project_flat is not None:
                                _, project_pullback = jax.vjp(project_flat, y_old)
                                rebuild_flat_bars = jax.vmap(lambda bar: project_pullback(bar)[0])(rebuild_flat_bars)
                        else:
                            def _build_from_flat(flat_inner):
                                projected_inner = _project_flat_state_if_needed(flat_inner, project_flat)
                                return build_lagged_response(unpack_flat(projected_inner))

                            _, build_pullback = jax.vjp(_build_from_flat, y_old)
                            rebuild_flat_bars = jax.vmap(lambda lagged_bar: build_pullback(lagged_bar)[0])(
                                total_lagged_bars
                            )
                        y_old_bars_rebuild = y_old_bars + rebuild_flat_bars + reference_bars
                        if physics_context.flat_rhs_build_support_pullback is not None:
                            rebuild_support_bars = jax.vmap(
                                lambda lagged_bar: _radau_sanitize_support_delta_bar_tree(
                                    support_payload,
                                    physics_context.flat_rhs_build_support_pullback(
                                        y_old,
                                        lagged_bar,
                                        support_payload,
                                    ),
                                )
                            )(total_lagged_bars)
                            support_bar_leaves_rebuild = tuple(
                                accumulated + increment
                                for accumulated, increment in zip(
                                    support_bar_leaves_next,
                                    jax.tree_util.tree_leaves(rebuild_support_bars),
                                    strict=True,
                                )
                            )
                        else:
                            support_bar_leaves_rebuild = support_bar_leaves_next
                        return (
                            y_old_bars_rebuild,
                            _batched_zero_like_tree(carry_for_step.reuse_state.lagged_response_cache),
                            jnp.zeros_like(reference_bars),
                            support_bar_leaves_rebuild,
                        )

                    y_prev, lagged_prev, reference_prev, support_bar_leaves_out = jax.lax.cond(
                        lagged_response_reused,
                        _reuse_lagged,
                        _rebuild_lagged,
                        operand=None,
                    )
                    return (
                        _ThetaAcceptedStepReducedCotangent(
                            y=y_prev,
                            lagged_response_cache=lagged_prev,
                            lagged_reference_y=reference_prev,
                        ),
                        support_bar_leaves_out,
                    )

                return jax.lax.cond(active, _do_step, _zero_step, operand=None)

            start_bars, support_bar_leaves = jax.lax.scan(
                _single_step_bwd,
                (reduced_bars, zero_support_bar_leaves),
                (step_start_states, segment_arrays),
                reverse=True,
            )[0]
            return start_bars, support_bar_leaves

        @jax.custom_vjp
        def _compact_lagged_rhs_with_support(t_value, y_value, lagged_response_value, support_value):
            return flat_rhs_with_lagged_response(t_value, y_value, lagged_response_value)

        def _compact_lagged_rhs_with_support_fwd(t_value, y_value, lagged_response_value, support_value):
            rhs_value = flat_rhs_with_lagged_response(t_value, y_value, lagged_response_value)
            return rhs_value, (t_value, y_value, lagged_response_value, support_value)

        def _compact_lagged_rhs_with_support_bwd(residual, rhs_bar):
            t_value, y_value, lagged_response_value, support_value = residual
            rhs_bar = jnp.asarray(rhs_bar, dtype=jnp.asarray(y_value).dtype)
            if physics_context.flat_rhs_state_pullback is not None:
                y_bar = physics_context.flat_rhs_state_pullback(
                    t_value,
                    y_value,
                    lagged_response_value,
                    rhs_bar,
                )
            else:
                def _rhs_from_y(y_inner):
                    return flat_rhs_with_lagged_response(
                        t_value,
                        y_inner,
                        lagged_response_value,
                    )

                _, y_pullback = jax.vjp(_rhs_from_y, y_value)
                (y_bar,) = y_pullback(rhs_bar)
            if physics_context.flat_rhs_lagged_response_pullback is not None:
                lagged_bar = physics_context.flat_rhs_lagged_response_pullback(
                    t_value,
                    y_value,
                    lagged_response_value,
                    rhs_bar,
                )
            else:
                def _rhs_from_lagged(lagged_inner):
                    return flat_rhs_with_lagged_response(
                        t_value,
                        y_value,
                        lagged_inner,
                    )

                _, lagged_pullback = jax.vjp(_rhs_from_lagged, lagged_response_value)
                (lagged_bar,) = lagged_pullback(rhs_bar)
            support_bar = _theta_lagged_rhs_support_pullback(
                physics_context,
                t_value=t_value,
                y_value=y_value,
                lagged_response=lagged_response_value,
                rhs_bar=rhs_bar,
                support_payload=support_value,
            )
            support_bar = _theta_support_cotangent_for_vjp(support_value, support_bar)
            return (
                jnp.zeros_like(t_value),
                y_bar,
                lagged_bar,
                support_bar,
            )

        _compact_lagged_rhs_with_support.defvjp(
            _compact_lagged_rhs_with_support_fwd,
            _compact_lagged_rhs_with_support_bwd,
        )

        def _build_lagged_response_from_flat(flat_y_value):
            if build_lagged_response is None:
                return None
            projected_y = _project_flat_state_if_needed(flat_y_value, project_flat)
            return build_lagged_response(unpack_flat(projected_y))

        @jax.custom_vjp
        def _compact_build_lagged_response_with_support(flat_y_value, support_value):
            return _build_lagged_response_from_flat(flat_y_value)

        def _compact_build_lagged_response_with_support_fwd(flat_y_value, support_value):
            lagged_value = _build_lagged_response_from_flat(flat_y_value)
            return lagged_value, (flat_y_value, support_value, lagged_value)

        def _compact_build_lagged_response_with_support_bwd(residual, lagged_bar):
            flat_y_value, support_value, lagged_value = residual
            if lagged_value is None:
                return (
                    jnp.zeros_like(flat_y_value),
                    _theta_support_cotangent_for_vjp(
                        support_value,
                        _radau_zero_support_delta_tree_like(support_value),
                    ),
                )
            projected_y = _project_flat_state_if_needed(flat_y_value, project_flat)
            state_value = unpack_flat(projected_y)
            if physics_context.pullback_build_lagged_response is not None:
                state_bar = physics_context.pullback_build_lagged_response(
                    state_value,
                    lagged_bar,
                )
                flat_y_bar = physics_context.pack_flat(state_bar)
                if project_flat is not None:
                    _, project_pullback = jax.vjp(project_flat, flat_y_value)
                    (flat_y_bar,) = project_pullback(flat_y_bar)
            else:
                def _build_from_flat(flat_inner):
                    projected_inner = _project_flat_state_if_needed(flat_inner, project_flat)
                    return build_lagged_response(unpack_flat(projected_inner))

                _, build_pullback = jax.vjp(_build_from_flat, flat_y_value)
                (flat_y_bar,) = build_pullback(lagged_bar)
            if physics_context.flat_rhs_build_support_pullback is not None:
                support_bar = physics_context.flat_rhs_build_support_pullback(
                    flat_y_value,
                    lagged_bar,
                    support_value,
                )
            else:
                support_bar = _radau_zero_support_delta_tree_like(support_value)
            support_bar = _theta_support_cotangent_for_vjp(support_value, support_bar)
            return flat_y_bar, support_bar

        _compact_build_lagged_response_with_support.defvjp(
            _compact_build_lagged_response_with_support_fwd,
            _compact_build_lagged_response_with_support_bwd,
        )

        def _make_final_reduced_from_start(step_fn_for_replay):
            def _final_reduced_from_start(
                start_y,
                start_lagged_cache,
                start_lagged_reference_y,
            ):
                reuse_state = dataclasses.replace(
                    _theta_initial_reuse_state(state_dim, dtype),
                    lagged_response_cache=start_lagged_cache,
                    lagged_response_available=jnp.asarray(start_lagged_cache is not None),
                    lagged_response_valid=segment_start_carry.lagged_response_valid,
                    lagged_reference_y=start_lagged_reference_y,
                )
                step_state0 = _ThetaStepState(
                    t=segment_start_carry.t,
                    y=start_y,
                    dt=attempted_dts[0],
                    status=jnp.asarray([0, 0, 0], dtype=jnp.int32),
                    prev_error=jnp.asarray(1.0, dtype=dtype),
                    prev_dt=jnp.asarray(0.0, dtype=dtype),
                    recent_reject_count=jnp.asarray(0, dtype=jnp.int32),
                    regrowth_cooldown=jnp.asarray(0, dtype=jnp.int32),
                    easy_growth_streak=jnp.asarray(0, dtype=jnp.int32),
                    prev_theta_final=jnp.asarray(0.0, dtype=dtype),
                    prev_newton_iter_count=jnp.asarray(0, dtype=jnp.int32),
                    reuse_state=reuse_state,
                )

                def _slot(carry, slot_values):
                    active, dt_value, next_dt_value, recent_reject, cooldown, streak, lagged_valid = slot_values
                    carry = dataclasses.replace(carry, dt=dt_value)

                    def _run(_):
                        next_state, _info = step_fn_for_replay(carry)
                        next_reuse = dataclasses.replace(
                            next_state.reuse_state,
                            lagged_response_valid=lagged_valid,
                        )
                        return dataclasses.replace(
                            next_state,
                            dt=next_dt_value,
                            recent_reject_count=recent_reject,
                            regrowth_cooldown=cooldown,
                            easy_growth_streak=streak,
                            reuse_state=next_reuse,
                        )

                    return jax.lax.cond(active, _run, lambda _: carry, operand=None), None

                final_state, _ = jax.lax.scan(
                    _slot,
                    step_state0,
                    (
                        active_mask,
                        attempted_dts,
                        next_dts,
                        next_recent_reject_count,
                        next_regrowth_cooldown,
                        next_easy_growth_streak,
                        next_lagged_response_valid,
                    ),
                )
                return _ThetaAcceptedStepReducedCotangent(
                    y=final_state.y,
                    lagged_response_cache=final_state.reuse_state.lagged_response_cache,
                    lagged_reference_y=final_state.reuse_state.lagged_reference_y,
                )

            return _final_reduced_from_start

        _final_reduced_from_start = _make_final_reduced_from_start(_step_fn)

        def _take_batched_tree_axis0(tree, index):
            if tree is None:
                return None
            return jax.tree_util.tree_map(lambda value: value[index], tree)

        def _zero_tree_like(tree):
            if tree is None:
                return None
            return jax.tree_util.tree_map(jnp.zeros_like, tree)

        def _single_reduced_pullback(objective_index):
            reduced_lagged_bar_i = _take_batched_tree_axis0(
                reduced_bars.lagged_response_cache,
                objective_index,
            )
            if mode == "theta_zero_lagged":
                reduced_lagged_bar_i = _zero_tree_like(reduced_lagged_bar_i)
            reduced_bar_i = _ThetaAcceptedStepReducedCotangent(
                y=reduced_bars.y[objective_index],
                lagged_response_cache=reduced_lagged_bar_i,
                lagged_reference_y=reduced_bars.lagged_reference_y[objective_index],
            )

            if segment_start_carry.lagged_response_cache is None:
                def _final_reduced_from_start_no_cache(start_y, start_lagged_reference_y):
                    return _final_reduced_from_start(
                        start_y,
                        None,
                        start_lagged_reference_y,
                    )

                _, pullback = jax.vjp(
                    _final_reduced_from_start_no_cache,
                    segment_start_carry.y,
                    segment_start_carry.lagged_reference_y,
                )
                start_y_bar_i, start_reference_bar_i = pullback(reduced_bar_i)
                start_cache_bar_i = None
            else:
                _, pullback = jax.vjp(
                    _final_reduced_from_start,
                    segment_start_carry.y,
                    segment_start_carry.lagged_response_cache,
                    segment_start_carry.lagged_reference_y,
                )
                start_y_bar_i, start_cache_bar_i, start_reference_bar_i = pullback(reduced_bar_i)
            return _ThetaAcceptedStepReducedCotangent(
                y=start_y_bar_i,
                lagged_response_cache=start_cache_bar_i,
                lagged_reference_y=start_reference_bar_i,
            )

        objective_indices = jnp.arange(objective_count, dtype=jnp.int32)
        if mode == "theta_compact_support_probe":
            def _single_support_pullback(objective_index):
                reduced_bar_i = _ThetaAcceptedStepReducedCotangent(
                    y=reduced_bars.y[objective_index],
                    lagged_response_cache=_take_batched_tree_axis0(
                        reduced_bars.lagged_response_cache,
                        objective_index,
                    ),
                    lagged_reference_y=reduced_bars.lagged_reference_y[objective_index],
                )

                def _make_support_step_fn(support_value):
                    def _flat_rhs_with_lagged_response_support(t_value, y_value, lagged_response_value):
                        return _compact_lagged_rhs_with_support(
                            t_value,
                            y_value,
                            lagged_response_value,
                            support_value,
                        )

                    def _build_lagged_response_support(state_value):
                        return _compact_build_lagged_response_with_support(
                            physics_context.pack_flat(state_value),
                            support_value,
                        )

                    return _make_theta_replay_step_fn(
                        flat_rhs,
                        _flat_rhs_with_lagged_response_support,
                        build_lagged_response_replay=_build_lagged_response_support,
                    )

                if segment_start_carry.lagged_response_cache is None:
                    def _final_reduced_from_start_no_cache_support(
                        start_y,
                        start_lagged_reference_y,
                        support_value,
                    ):
                        support_final_reduced_from_start = _make_final_reduced_from_start(
                            _make_support_step_fn(support_value)
                        )
                        return support_final_reduced_from_start(
                            start_y,
                            None,
                            start_lagged_reference_y,
                        )

                    _, pullback = jax.vjp(
                        _final_reduced_from_start_no_cache_support,
                        segment_start_carry.y,
                        segment_start_carry.lagged_reference_y,
                        support_payload,
                    )
                    start_y_bar_i, start_reference_bar_i, support_bar_i = pullback(reduced_bar_i)
                    start_cache_bar_i = None
                else:
                    def _final_reduced_from_start_support(
                        start_y,
                        start_lagged_cache,
                        start_lagged_reference_y,
                        support_value,
                    ):
                        support_final_reduced_from_start = _make_final_reduced_from_start(
                            _make_support_step_fn(support_value)
                        )
                        return support_final_reduced_from_start(
                            start_y,
                            start_lagged_cache,
                            start_lagged_reference_y,
                        )

                    _, pullback = jax.vjp(
                        _final_reduced_from_start_support,
                        segment_start_carry.y,
                        segment_start_carry.lagged_response_cache,
                        segment_start_carry.lagged_reference_y,
                        support_payload,
                    )
                    start_y_bar_i, start_cache_bar_i, start_reference_bar_i, support_bar_i = pullback(reduced_bar_i)
                return (
                    _ThetaAcceptedStepReducedCotangent(
                        y=start_y_bar_i,
                        lagged_response_cache=start_cache_bar_i,
                        lagged_reference_y=start_reference_bar_i,
                    ),
                    _radau_sanitize_support_delta_bar_tree(support_payload, support_bar_i),
                )

            start_bars, support_bars = jax.vmap(_single_support_pullback)(objective_indices)
            return start_bars, tuple(jax.tree_util.tree_leaves(support_bars))

        start_bars = jax.vmap(_single_reduced_pullback)(objective_indices)
        return start_bars, zero_support_bar_leaves

    raise NotImplementedError(
        "theta full-transport reverse has real schedule/segment forward artifacts, "
        "but this cotangent diagnostic mode is not implemented for theta. "
        "Use reverse_stage_cotangent_mode='full' for the solver-selected theta "
        "implicit-transpose lane, or one of zero_step_bwd/theta_state_only/"
        "theta_compact_support_probe/theta_implicit_transpose_probe for diagnostics."
    )


def _reverse_segment_reduced_cotangent_bwd_batched_with_support_call(
    execution_context,
    cotangent_mode,
    reduced_bars,
    segment_start_carry,
    segment_arrays,
    support_payload,
):
    if isinstance(execution_context, _ThetaReverseExecutionContext):
        return _theta_segment_reduced_cotangent_bwd_batched_with_support_call(
            execution_context,
            cotangent_mode,
            reduced_bars,
            segment_start_carry,
            segment_arrays,
            support_payload,
        )
    _require_reverse_execution_context_radau(
        execution_context,
        "reverse segmented support-payload cotangent sweep",
    )
    return _radau_segment_reduced_cotangent_bwd_batched_with_support_call(
        execution_context,
        cotangent_mode,
        reduced_bars,
        segment_start_carry,
        segment_arrays,
        support_payload,
    )


def _reverse_reduced_cotangent(
    execution_context,
    *,
    y,
    lagged_response_cache,
    lagged_reference_y,
):
    if isinstance(execution_context, _ThetaReverseExecutionContext):
        return _ThetaAcceptedStepReducedCotangent(
            y=y,
            lagged_response_cache=lagged_response_cache,
            lagged_reference_y=lagged_reference_y,
        )
    _require_reverse_execution_context_radau(execution_context, "reverse reduced cotangent")
    return _RadauAcceptedStepReducedCotangent(
        y=y,
        lagged_response_cache=lagged_response_cache,
        lagged_reference_y=lagged_reference_y,
    )


def _theta_initial_carry_from_state_with_static_setup(
    *,
    solver,
    state,
    solve_vector_field,
    species,
    prepared_rollout_static,
):
    temperature_active_mask, fixed_temperature_profile = _extract_fixed_temperature_projection(solve_vector_field)
    density_floor, temperature_floor = _extract_state_regularization(solve_vector_field)

    def _flat_state_from_state(state_value):
        flat_state, unpack_flat, _unpack_packed, _pack_state, project_flat = _make_solver_state_transform(
            state_value,
            species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
            density_floor=density_floor,
            temperature_floor=temperature_floor,
        )
        return flat_state, unpack_flat, project_flat

    flat_state0, unpack_flat, project_flat = _flat_state_from_state(state)
    dtype = jnp.asarray(flat_state0).dtype
    build_lagged_response, _ = _lagged_response_hooks(solve_vector_field)
    rhs_mode = str(getattr(solver, "rhs_mode", "black_box")).strip().lower()
    use_transport_lagged_response = rhs_mode in {"lagged_transport_response", "lagged_response"}
    if use_transport_lagged_response and build_lagged_response is not None:
        lagged_reference_y = _project_flat_state_if_needed(flat_state0, project_flat)
        lagged_response_cache = build_lagged_response(unpack_flat(lagged_reference_y))
        lagged_response_valid = jnp.asarray(True)
    else:
        lagged_reference_y = flat_state0
        lagged_response_cache = None
        lagged_response_valid = jnp.asarray(False)
    return _ThetaReverseCarry(
        t=jnp.asarray(getattr(solver, "t0", 0.0), dtype=dtype),
        y=flat_state0,
        lagged_response_cache=lagged_response_cache,
        lagged_response_valid=lagged_response_valid,
        lagged_reference_y=lagged_reference_y,
    )


def reverse_initial_carry_from_state_with_static_setup(
    *,
    solver,
    state,
    solve_vector_field,
    species,
    prepared_rollout_static,
):
    """Build the initial carry with the validated reverse-local lagged pullback."""

    if _is_theta_reverse_solver(solver):
        return _theta_initial_carry_from_state_with_static_setup(
            solver=solver,
            state=state,
            solve_vector_field=solve_vector_field,
            species=species,
            prepared_rollout_static=prepared_rollout_static,
        )

    _require_reverse_solver_radau(solver, "reverse initial carry")
    temperature_active_mask, fixed_temperature_profile = _extract_fixed_temperature_projection(solve_vector_field)
    density_floor, temperature_floor = _extract_state_regularization(solve_vector_field)
    kernel_context = prepared_rollout_static.kernel_context
    physics_context = prepared_rollout_static.physics_context
    initial_carry_static = prepared_rollout_static.initial_carry
    lagged_pullback_fn = lagged_response_pullback_from_owner(solve_vector_field)

    def _flat_state_from_state(state_value):
        flat_state, *_ = _make_solver_state_transform(
            state_value,
            species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
            density_floor=density_floor,
            temperature_floor=temperature_floor,
        )
        return flat_state

    def _build_state_from_flat(flat_value, unpack_flat, project_flat):
        return unpack_flat(_project_flat_state_if_needed(flat_value, project_flat))

    @jax.custom_vjp
    def _build_initial_carry(state_value):
        flat_state0, unpack_flat, _unpack_packed, _pack_state, project_flat = _make_solver_state_transform(
            state_value,
            species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
            density_floor=density_floor,
            temperature_floor=temperature_floor,
        )
        lagged_state0 = _build_state_from_flat(flat_state0, unpack_flat, project_flat)
        initial_lagged_response = (
            physics_context.build_lagged_response(lagged_state0)
            if (kernel_context.use_transport_lagged_response and physics_context.build_lagged_response is not None)
            else None
        )
        initial_rhs = _radau_eval_rhs(
            initial_carry_static.t,
            flat_state0,
            initial_lagged_response,
            physics_context.flat_rhs,
            physics_context.flat_rhs_with_lagged_response,
        )
        step_state0 = _make_radau_initial_step_state(
            initial_carry_static.t,
            flat_state0,
            initial_carry_static.dt,
            kernel_context.dtype,
            initial_rhs,
            kernel_context.num_stages,
            initial_carry_static.real_lu,
            initial_carry_static.real_piv,
            initial_carry_static.complex_lu,
            initial_carry_static.complex_piv,
            initial_lagged_response,
            jnp.asarray(kernel_context.use_transport_lagged_response),
            flat_state0,
        )
        return _radau_carry_from_step_state(step_state0)

    def _build_initial_carry_fwd(state_value):
        flat_state0, unpack_flat, _unpack_packed, _pack_state, project_flat = _make_solver_state_transform(
            state_value,
            species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
            density_floor=density_floor,
            temperature_floor=temperature_floor,
        )
        lagged_state0 = _build_state_from_flat(flat_state0, unpack_flat, project_flat)
        initial_lagged_response = (
            physics_context.build_lagged_response(lagged_state0)
            if (kernel_context.use_transport_lagged_response and physics_context.build_lagged_response is not None)
            else None
        )
        initial_rhs = _radau_eval_rhs(
            initial_carry_static.t,
            flat_state0,
            initial_lagged_response,
            physics_context.flat_rhs,
            physics_context.flat_rhs_with_lagged_response,
        )
        step_state0 = _make_radau_initial_step_state(
            initial_carry_static.t,
            flat_state0,
            initial_carry_static.dt,
            kernel_context.dtype,
            initial_rhs,
            kernel_context.num_stages,
            initial_carry_static.real_lu,
            initial_carry_static.real_piv,
            initial_carry_static.complex_lu,
            initial_carry_static.complex_piv,
            initial_lagged_response,
            jnp.asarray(kernel_context.use_transport_lagged_response),
            flat_state0,
        )
        carry0 = _radau_carry_from_step_state(step_state0)
        residual = (state_value, flat_state0, lagged_state0, initial_lagged_response)
        return carry0, residual

    def _build_initial_carry_bwd(residual, carry_bar):
        state_value, flat_state0, lagged_state0, initial_lagged_response = residual
        _, unpack_flat, _unpack_packed, _pack_state, project_flat = _make_solver_state_transform(
            state_value,
            species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
            density_floor=density_floor,
            temperature_floor=temperature_floor,
        )
        flat_bar = jnp.asarray(carry_bar.y)
        flat_bar = flat_bar + jnp.asarray(carry_bar.lagged_reference_y)

        prev_stages_bar = jnp.asarray(carry_bar.prev_stages).reshape((kernel_context.num_stages, -1))
        rhs_bar = jnp.sum(prev_stages_bar, axis=0)
        lagged_bar = carry_bar.lagged_response_cache

        def _tree_max_abs(tree):
            values = []
            for leaf in jax.tree_util.tree_leaves(tree):
                arr = jnp.asarray(leaf)
                if arr.dtype == jax.dtypes.float0:
                    continue
                if jnp.issubdtype(arr.dtype, jnp.number):
                    values.append(jnp.max(jnp.abs(arr)))
            if not values:
                return jnp.asarray(0.0, dtype=flat_state0.dtype)
            return jnp.max(jnp.stack([jnp.asarray(value, dtype=flat_state0.dtype) for value in values]))

        def _zero_flat_bar():
            return jnp.zeros_like(flat_state0)

        def _rhs_state_pullback_fallback(lagged_response_value):
            def _rhs_from_flat(flat_value):
                return _radau_eval_rhs(
                    initial_carry_static.t,
                    flat_value,
                    lagged_response_value,
                    physics_context.flat_rhs,
                    physics_context.flat_rhs_with_lagged_response,
                )

            _, rhs_pullback = jax.vjp(_rhs_from_flat, flat_state0)
            (rhs_flat_bar,) = rhs_pullback(rhs_bar)
            return rhs_flat_bar

        def _nonzero_rhs_state_pullback(_):
            if physics_context.flat_rhs_state_pullback is not None:
                rhs_flat_bar_value = physics_context.flat_rhs_state_pullback(
                    initial_carry_static.t,
                    flat_state0,
                    initial_lagged_response,
                    rhs_bar,
                )
                if project_flat is not None:
                    _, project_pullback = jax.vjp(project_flat, flat_state0)
                    (rhs_flat_bar_value,) = project_pullback(rhs_flat_bar_value)
                return rhs_flat_bar_value
            return _rhs_state_pullback_fallback(initial_lagged_response)

        rhs_flat_bar = jax.lax.cond(
            _tree_max_abs(rhs_bar) > 0.0,
            _nonzero_rhs_state_pullback,
            lambda _: _zero_flat_bar(),
            operand=None,
        )
        flat_bar = flat_bar + rhs_flat_bar

        if initial_lagged_response is not None:
            def _rhs_from_flat_and_lagged(flat_value, lagged_value):
                return _radau_eval_rhs(
                    initial_carry_static.t,
                    flat_value,
                    lagged_value,
                    physics_context.flat_rhs,
                    physics_context.flat_rhs_with_lagged_response,
                )

            def _zero_lagged_bar():
                return _radau_align_tangent_tree_to_primal(None, initial_lagged_response)

            def _nonzero_rhs_lagged_pullback(_):
                if physics_context.flat_rhs_lagged_response_pullback is not None:
                    return physics_context.flat_rhs_lagged_response_pullback(
                        initial_carry_static.t,
                        flat_state0,
                        initial_lagged_response,
                        rhs_bar,
                    )
                _, rhs_pullback = jax.vjp(_rhs_from_flat_and_lagged, flat_state0, initial_lagged_response)
                _rhs_flat_bar_unused, rhs_lagged_bar_value = rhs_pullback(rhs_bar)
                return rhs_lagged_bar_value

            rhs_lagged_bar = jax.lax.cond(
                _tree_max_abs(rhs_bar) > 0.0,
                _nonzero_rhs_lagged_pullback,
                lambda _: _zero_lagged_bar(),
                operand=None,
            )
            lagged_bar = add_trees(lagged_bar, rhs_lagged_bar)

            if lagged_pullback_fn is not None:
                lagged_state_bar = lagged_pullback_fn(lagged_state0, lagged_bar)
            else:
                def _nonzero_lagged_state_pullback(_):
                    def _build_lagged_from_state(lagged_state_value):
                        return physics_context.build_lagged_response(lagged_state_value)

                    _, lagged_pullback = jax.vjp(_build_lagged_from_state, lagged_state0)
                    (lagged_state_bar_value,) = lagged_pullback(lagged_bar)
                    return lagged_state_bar_value

                lagged_state_bar = jax.lax.cond(
                    _tree_max_abs(lagged_bar) > 0.0,
                    _nonzero_lagged_state_pullback,
                    lambda _: jax.tree_util.tree_map(jnp.zeros_like, lagged_state0),
                    operand=None,
                )

            def _lagged_state_from_flat(flat_value):
                return _build_state_from_flat(flat_value, unpack_flat, project_flat)

            def _nonzero_lagged_state_flat_pullback(_):
                _, lagged_state_flat_pullback = jax.vjp(_lagged_state_from_flat, flat_state0)
                (lagged_flat_bar_value,) = lagged_state_flat_pullback(lagged_state_bar)
                return lagged_flat_bar_value

            lagged_flat_bar = jax.lax.cond(
                _tree_max_abs(lagged_state_bar) > 0.0,
                _nonzero_lagged_state_flat_pullback,
                lambda _: _zero_flat_bar(),
                operand=None,
            )
            flat_bar = flat_bar + lagged_flat_bar

        _, state_pullback = jax.vjp(_flat_state_from_state, state_value)
        (state_bar,) = state_pullback(flat_bar)
        return (state_bar,)

    _build_initial_carry.defvjp(_build_initial_carry_fwd, _build_initial_carry_bwd)
    return _build_initial_carry(state)


def prepare_reverse_static_setup(
    parameter_values,
    *,
    config: Mapping[str, Any],
    runtime,
    baseline_state,
    profile_cfg: Mapping[str, Any],
    initial_er_root_ad: str = "off",
    accepted_step_limit_override: int | None = None,
    reverse_segment_length: int | str | None = None,
    reverse_direct_stage_adjoint: bool = False,
    reverse_stage_adjoint_solve_mode: str = "structured",
    reverse_rhs_transpose_mode: str = "generic",
    reverse_stage_cotangent_mode: str = "full",
    reverse_step_bwd_mode: str = "current",
    reverse_stage_adjoint_memory_mode: str = "default",
    reverse_stage_adjoint_iter_maxiter: int = 40,
    reverse_stage_adjoint_iter_tol: float = 1.0e-10,
    max_reverse_accepted_steps: int | None = None,
) -> RealtimeGeometryReverseStaticSetup:
    state0_static = initial_state_for_parameter_vector(
        parameter_values,
        config=config,
        initial_er_root_ad=initial_er_root_ad,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        runtime=runtime,
    )
    prepared_components_static = prepare_transport_solver_components(dict(config), runtime, state0_static)
    solver = prepared_components_static["solver"]
    solve_vector_field_static = prepared_components_static["solve_vector_field"]
    prepared_rollout_static = _build_prepared_reverse_accepted_rollout(
        solver=solver,
        state=state0_static,
        vector_field=solve_vector_field_static,
        species=runtime.species,
    )
    execution_context = _build_prepared_reverse_execution_context(
        solver=solver,
        prepared_rollout=prepared_rollout_static,
    )
    if reverse_direct_stage_adjoint:
        execution_context = dataclasses.replace(
            execution_context,
            physics_context=dataclasses.replace(
                execution_context.physics_context,
                reverse_direct_stage_adjoint=True,
                reverse_stage_adjoint_solve_mode=str(reverse_stage_adjoint_solve_mode),
                reverse_rhs_transpose_mode=str(reverse_rhs_transpose_mode),
                reverse_stage_cotangent_mode=str(reverse_stage_cotangent_mode),
                reverse_step_bwd_mode=str(reverse_step_bwd_mode),
                reverse_stage_adjoint_memory_mode=str(reverse_stage_adjoint_memory_mode),
                reverse_stage_adjoint_iter_maxiter=int(reverse_stage_adjoint_iter_maxiter),
                reverse_stage_adjoint_iter_tol=float(reverse_stage_adjoint_iter_tol),
            ),
        )
    stop_after_accepted_steps = (
        int(accepted_step_limit_override)
        if accepted_step_limit_override is not None
        else getattr(solver, "stop_after_accepted_steps", None)
    )
    requested_full_final_time = stop_after_accepted_steps is None
    max_total_steps = int(max(1, getattr(solver, "max_steps", 1)))
    reverse_segment_auto_quarter = (
        isinstance(reverse_segment_length, str)
        and str(reverse_segment_length).strip().lower()
        in {"auto", "auto_quarter", "quarter", "accepted_quarter"}
    )
    reverse_segment_length_eff = None
    if reverse_segment_length is not None and not reverse_segment_auto_quarter:
        reverse_segment_length_eff = int(reverse_segment_length)
    needs_schedule_probe = (
        stop_after_accepted_steps is not None
        or reverse_segment_auto_quarter
        or (requested_full_final_time and reverse_segment_length_eff is not None)
        or (requested_full_final_time and max_reverse_accepted_steps is not None)
    )
    if needs_schedule_probe:
        probe_max_total_steps = max_total_steps
        probe_stop_after_accepted_steps = stop_after_accepted_steps
        if requested_full_final_time and max_reverse_accepted_steps is not None:
            probe_stop_after_accepted_steps = int(max_reverse_accepted_steps)
        if probe_stop_after_accepted_steps is not None:
            probe_max_total_steps = min(
                max_total_steps,
                max(
                    int(probe_stop_after_accepted_steps) * 16,
                    int(probe_stop_after_accepted_steps) + 16,
                ),
            )
        max_total_steps = min(
            max_total_steps,
            max(
                int(probe_stop_after_accepted_steps) * 16,
                int(probe_stop_after_accepted_steps) + 16,
            ),
        ) if probe_stop_after_accepted_steps is not None else max_total_steps
        schedule_probe = _reverse_adaptive_schedule_rollout(
            execution_context,
            prepared_rollout_static.initial_carry,
            max_total_steps=probe_max_total_steps,
            stop_after_accepted_steps=probe_stop_after_accepted_steps,
        )
        actual_attempt_count = int(np.asarray(jax.device_get(schedule_probe.attempt_count)))
        active_mask_np = np.asarray(jax.device_get(schedule_probe.trace.active_mask), dtype=bool)
        accepted_mask_np = np.asarray(jax.device_get(schedule_probe.trace.accepted_mask), dtype=bool)
        accepted_count_np = int(np.sum(np.logical_and(active_mask_np, accepted_mask_np)))
        if requested_full_final_time and max_reverse_accepted_steps is not None:
            final_time = float(np.asarray(jax.device_get(schedule_probe.final_carry.t)))
            target_time = float(getattr(solver, "t1", final_time))
            if final_time < target_time - 1.0e-12 and accepted_count_np >= int(max_reverse_accepted_steps):
                raise RuntimeError(
                    "full transport reverse trial exceeded optimization accepted-step guard "
                    f"(accepted_steps={accepted_count_np}, max_reverse_accepted_steps="
                    f"{int(max_reverse_accepted_steps)}, final_time={final_time:.16e}, "
                    f"target_time={target_time:.16e}); treating trial as failed for optimization."
                )
        if reverse_segment_auto_quarter:
            reverse_segment_length_eff = max(1, (accepted_count_np + 3) // 4)
        if requested_full_final_time and reverse_segment_length_eff is not None:
            stop_after_accepted_steps = max(1, accepted_count_np)
        replay_accepted_limit = (
            int(stop_after_accepted_steps)
            if stop_after_accepted_steps is not None
            else int(probe_stop_after_accepted_steps)
            if probe_stop_after_accepted_steps is not None
            else 1
        )
        max_total_steps = min(
            probe_max_total_steps,
            max(actual_attempt_count + 2, replay_accepted_limit),
        )
        if stop_after_accepted_steps is None:
            accepted_limit = int(max_total_steps)
        else:
            accepted_limit = int(stop_after_accepted_steps)
        next_lagged_valid_np = np.asarray(
            jax.device_get(schedule_probe.trace.next_lagged_response_valid),
            dtype=bool,
        )
        accepted_positions = np.nonzero(np.logical_and(active_mask_np, accepted_mask_np))[0][:accepted_limit]
        incoming_valid = bool(np.asarray(jax.device_get(prepared_rollout_static.initial_carry.lagged_response_valid)))
        lagged_branch_schedule: list[bool] = []
        for accepted_position in accepted_positions:
            lagged_branch_schedule.append(bool(incoming_valid))
            incoming_valid = bool(next_lagged_valid_np[int(accepted_position)])
        if len(lagged_branch_schedule) < accepted_limit:
            lagged_branch_schedule.extend([bool(incoming_valid)] * (accepted_limit - len(lagged_branch_schedule)))
        execution_context = dataclasses.replace(
            execution_context,
            physics_context=dataclasses.replace(
                execution_context.physics_context,
                reverse_lagged_branch_schedule=tuple(lagged_branch_schedule),
            ),
        )
    return RealtimeGeometryReverseStaticSetup(
        solver=solver,
        solve_vector_field=solve_vector_field_static,
        prepared_rollout=prepared_rollout_static,
        execution_context=execution_context,
        stop_after_accepted_steps=stop_after_accepted_steps,
        max_total_steps=max_total_steps,
        reverse_segment_length=reverse_segment_length_eff,
        require_final_time=bool(requested_full_final_time and reverse_segment_length_eff is not None),
    )


def default_realtime_geometry_support_reverse_dependencies() -> RealtimeGeometrySupportReverseDependencies:
    """Return the production/default dependency bundle for the compact reverse rule."""

    return RealtimeGeometrySupportReverseDependencies(
        initial_er_root_enabled=initial_er_root_enabled,
        initial_state_for_parameter_vector=initial_state_for_parameter_vector,
        state_with_initial_er_root_ad=state_with_initial_er_root_ad,
        reverse_initial_carry_from_state_with_static_setup=reverse_initial_carry_from_state_with_static_setup,
        objective_scalar_by_index=objective_scalar_by_index,
        add_trees=add_trees,
        initial_er_selected_root_profile=initial_er_selected_root_profile,
        initial_er_charge_flux_residuals=initial_er_charge_flux_residuals,
        initial_er_charge_flux_residual_scalar=initial_er_charge_flux_residual_scalar,
        initial_er_charge_flux_residual_er_derivative=initial_er_charge_flux_residual_er_derivative,
        compact_initial_er_state_pullback=compact_initial_er_state_pullback,
        compact_initial_er_ntx_support_pullback_leaves=compact_initial_er_ntx_support_pullback_leaves,
        runtime_with_geometry_payload=runtime_with_geometry_payload,
        runtime_with_ntx_support_payload=runtime_with_ntx_support_payload,
    )


def realtime_geometry_transport_reverse_table_context(
    *,
    config: Mapping[str, Any],
    baseline_values,
    baseline_runtime,
    baseline_state,
    profile_cfg: Mapping[str, Any],
    neoclassical_cfg: Mapping[str, Any],
) -> RealtimeGeometryTransportReverseTableContext:
    """Build the grouped realtime-geometry transport reverse table context."""

    return RealtimeGeometryTransportReverseTableContext(
        config=config,
        baseline_values=baseline_values,
        baseline_runtime=baseline_runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        neoclassical_cfg=neoclassical_cfg,
    )


def prepare_realtime_geometry_support_segment_core_setup(
    *,
    args,
    config: Mapping[str, Any],
    baseline_values,
    baseline_runtime,
    baseline_state,
    profile_cfg: Mapping[str, Any],
    neoclassical_cfg: Mapping[str, Any],
    parameter_order: Sequence[str],
    find_ntx_support_payload: NtxSupportPayloadBuilder,
    prepare_reverse_static_setup: ReverseStaticSetupBuilder,
    geometry_volume_diagnostics: GeometryDiagnosticsBuilder | None = None,
) -> RealtimeGeometrySupportSegmentCoreSetup:
    """Prepare reusable inputs for a segmented realtime-geometry support pullback.

    This keeps the benchmark-owned heavy reverse sweep unchanged while moving
    the stable support-payload selection and reverse static setup into the
    internal transport reverse-AD API.
    """

    if not callable(find_ntx_support_payload):
        raise TypeError("find_ntx_support_payload must be callable.")
    if not callable(prepare_reverse_static_setup):
        raise TypeError("prepare_reverse_static_setup must be callable.")
    if geometry_volume_diagnostics is not None and not callable(geometry_volume_diagnostics):
        raise TypeError("geometry_volume_diagnostics must be callable when provided.")
    parameter_labels = tuple(str(name) for name in parameter_order)
    combined_geometry_payload = str(args.realtime_geometry_gradient_path) == "reverse_payload"
    ntx_surface_backend = str(neoclassical_cfg.get("ntx_exact_surface_backend", "booz"))
    ntx_support_payload = find_ntx_support_payload(baseline_runtime)
    support_payload = (
        {"geometry": baseline_runtime.geometry, "ntx_support": ntx_support_payload}
        if combined_geometry_payload
        else ntx_support_payload
    )
    profile_values = baseline_values[: len(parameter_labels)]
    support_probe_cotangent_mode = str(args.reverse_stage_cotangent_mode)
    reverse_setup = prepare_reverse_static_setup(
        profile_values,
        config=config,
        runtime=baseline_runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        initial_er_root_ad=args.initial_er_root_ad,
        accepted_step_limit_override=args.accepted_step_limit,
        reverse_segment_length=args.reverse_segment_length,
        reverse_direct_stage_adjoint=True,
        reverse_stage_adjoint_solve_mode=args.reverse_stage_adjoint_solve_mode,
        reverse_rhs_transpose_mode=args.reverse_rhs_transpose_mode,
        reverse_stage_cotangent_mode=support_probe_cotangent_mode,
        reverse_step_bwd_mode=args.reverse_step_bwd_mode,
        reverse_stage_adjoint_memory_mode=args.reverse_stage_adjoint_memory_mode,
        reverse_stage_adjoint_iter_maxiter=args.reverse_stage_adjoint_iter_maxiter,
        reverse_stage_adjoint_iter_tol=args.reverse_stage_adjoint_iter_tol,
    )
    early_geometry_diagnostics = (
        None
        if geometry_volume_diagnostics is None
        else geometry_volume_diagnostics(baseline_runtime.geometry)
    )
    return RealtimeGeometrySupportSegmentCoreSetup(
        combined_geometry_payload=combined_geometry_payload,
        ntx_surface_backend=ntx_surface_backend,
        ntx_support_payload=ntx_support_payload,
        support_payload=support_payload,
        profile_values=profile_values,
        support_probe_cotangent_mode=support_probe_cotangent_mode,
        reverse_setup=reverse_setup,
        early_geometry_diagnostics=early_geometry_diagnostics,
    )


def realtime_geometry_reverse_all_objectives_support_payload_bar_for_parameter_vector(
    parameter_values,
    *,
    config: Mapping[str, Any],
    runtime,
    baseline_state,
    profile_cfg: Mapping[str, Any],
    reverse_setup,
    support_payload,
    initial_er_root_ad: str = "off",
    objective_labels: Sequence[str],
    dependencies: RealtimeGeometrySupportReverseDependencies,
    progress_prefix: str = "[autodiff-gate]",
):
    """Return all objective values, profile gradients, and support cotangents.

    This is the migrated implementation of the realtime-geometry grouped
    support reverse kernel. Benchmark-specific objective/profile/root helpers
    are still supplied explicitly through `dependencies` so this move does not
    alter the JAX graph or numerical path.
    """

    if not isinstance(dependencies, RealtimeGeometrySupportReverseDependencies):
        raise TypeError(
            "dependencies must be a RealtimeGeometrySupportReverseDependencies instance."
        )
    objective_labels = tuple(str(label) for label in objective_labels)
    if not objective_labels:
        raise ValueError("objective_labels must contain at least one objective.")
    if reverse_setup.reverse_segment_length is None or int(reverse_setup.reverse_segment_length) <= 0:
        raise ValueError("support payload reverse probe requires --reverse-segment-length.")
    step_bwd_mode = str(
        getattr(reverse_setup.execution_context.physics_context, "reverse_step_bwd_mode", "current")
    ).strip().lower()
    if step_bwd_mode not in {
        "reduced_cotangent",
        "reduced_cotangent_lean_replay",
        "reduced_cotangent_recompute_replay",
        "lean_replay",
        "recompute_replay",
        "reduced",
        "state_only",
        "final_state",
    }:
        raise ValueError("support payload reverse probe requires a reduced-cotangent reverse step bwd mode.")

    def _zero_tangent_like(x):
        arr = jnp.asarray(x)
        if jnp.issubdtype(arr.dtype, jnp.inexact):
            return jnp.zeros_like(arr)
        return jnp.zeros(arr.shape, dtype=jax.dtypes.float0)

    def _take_tree_axis0(tree, index: int):
        return jax.tree_util.tree_map(lambda value: value[index], tree)

    def _batched_zero_tangent_tree_like(primal_tree, batch_size: int):
        zero_tree = _reverse_align_tangent_tree_to_primal(
            reverse_setup.execution_context,
            None,
            primal_tree,
        )
        return jax.tree_util.tree_map(
            lambda leaf: jnp.broadcast_to(
                jnp.asarray(leaf)[None, ...],
                (batch_size,) + jnp.asarray(leaf).shape,
            ),
            zero_tree,
        )

    initial_er_root_enabled = dependencies.initial_er_root_enabled(config, initial_er_root_ad)

    def _state_from_profiles(p):
        return dependencies.initial_state_for_parameter_vector(
            p,
            config=config,
            initial_er_root_ad="off",
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
        )

    phase_start = time.perf_counter()
    pre_root_initial_state, profile_state_pullback = jax.vjp(_state_from_profiles, parameter_values)
    initial_state = (
        dependencies.state_with_initial_er_root_ad(
            pre_root_initial_state,
            config=config,
            runtime=runtime,
            mode=initial_er_root_ad,
        )
        if initial_er_root_enabled
        else pre_root_initial_state
    )
    initial_state = jax.block_until_ready(initial_state)
    print(
        f"{progress_prefix} progress: support reverse profile-state vjp ready "
        f"elapsed_s={time.perf_counter() - phase_start:.3f}",
        flush=True,
    )

    def _carry_from_state(state_value):
        return dependencies.reverse_initial_carry_from_state_with_static_setup(
            solver=reverse_setup.solver,
            state=state_value,
            solve_vector_field=reverse_setup.solve_vector_field,
            species=runtime.species,
            prepared_rollout_static=reverse_setup.prepared_rollout,
        )

    phase_start = time.perf_counter()
    initial_carry, initial_state_pullback = jax.vjp(_carry_from_state, initial_state)
    initial_carry = jax.block_until_ready(initial_carry)
    print(
        f"{progress_prefix} progress: support reverse initial carry vjp ready "
        f"elapsed_s={time.perf_counter() - phase_start:.3f}",
        flush=True,
    )

    phase_start = time.perf_counter()
    final_y, residuals = _reverse_adaptive_final_y_realized_schedule_vjp_fwd(
        reverse_setup.execution_context,
        reverse_setup.max_total_steps,
        reverse_setup.stop_after_accepted_steps,
        reverse_setup.reverse_segment_length,
        initial_carry,
    )
    final_y, residuals = jax.block_until_ready((final_y, residuals))
    print(
        f"{progress_prefix} progress: support reverse realized-schedule vjp forward ready "
        f"elapsed_s={time.perf_counter() - phase_start:.3f}",
        flush=True,
    )

    (
        carry0,
        active_mask,
        accepted_mask,
        attempted_dts,
        next_dts,
        next_recent_reject_count,
        next_regrowth_cooldown,
        next_easy_growth_streak,
        next_lagged_response_valid,
        segment_start_carries,
        segmented_final_carry,
        segmented_replay_arrays,
    ) = residuals
    del (
        active_mask,
        accepted_mask,
        attempted_dts,
        next_dts,
        next_recent_reject_count,
        next_regrowth_cooldown,
        next_easy_growth_streak,
        next_lagged_response_valid,
    )
    if segment_start_carries is None or segmented_final_carry is None or segmented_replay_arrays is None:
        raise ValueError("support payload reverse probe requires segmented reverse residuals.")

    final_y_for_objective = segmented_final_carry.y
    if bool(getattr(reverse_setup, "require_final_time", False)):
        final_time = jnp.asarray(segmented_final_carry.t)
        target_time = jnp.asarray(getattr(reverse_setup.solver, "t1", final_time), dtype=final_time.dtype)
        final_time_ready, target_time_ready = jax.block_until_ready((final_time, target_time))
        if float(final_time_ready) < float(target_time_ready) - 1.0e-12:
            raise RuntimeError(
                "full transport reverse trial did not reach solver final time "
                f"(final_time={float(final_time_ready):.16e}, target_time={float(target_time_ready):.16e}); "
                "treating trial as failed for optimization."
            )

    objective_count = int(len(objective_labels))
    objective_values_rows = []
    final_y_bar_rows = []
    objective_payload_bar_rows = []
    combined_geometry_payload = isinstance(support_payload, dict) and "geometry" in support_payload
    zero_payload_bar = _reverse_zero_support_delta_tree_like(reverse_setup.solver, support_payload)
    for objective_i in range(objective_count):
        objective_name = objective_labels[objective_i]
        if objective_name == "bootstrap_current_softmax_abs_scaled":
            final_state_for_bootstrap = reverse_setup.prepared_rollout.physics_context.unpack_flat(
                final_y_for_objective
            )
            flux_model = getattr(getattr(runtime, "models", None), "flux", None)
            neoclassical_model = getattr(flux_model, "neoclassical_model", flux_model)
            corrected_fluxes_fn = getattr(neoclassical_model, "evaluate_momentum_corrected_fluxes", None)
            state_pullback_fn = getattr(neoclassical_model, "pullback_momentum_corrected_upar_state_by_radius", None)
            support_pullback_fn = getattr(
                neoclassical_model,
                "pullback_momentum_corrected_upar_support_by_radius",
                None,
            )
            geometry_pullback_fn = getattr(
                neoclassical_model,
                "pullback_momentum_corrected_upar_geometry_by_radius",
                None,
            )
            if not callable(corrected_fluxes_fn):
                raise NotImplementedError(
                    "bootstrap_current_softmax_abs_scaled requires realtime NTX "
                    "evaluate_momentum_corrected_fluxes for compact full-transport AD."
                )
            if not callable(state_pullback_fn) or not callable(support_pullback_fn):
                raise NotImplementedError(
                    "bootstrap_current_softmax_abs_scaled requires compact corrected-Upar "
                    "state and support pullbacks on the realtime NTX model."
                )
            corrected_fluxes = corrected_fluxes_fn(final_state_for_bootstrap)
            objective_value, upar_bar = bootstrap_current_softmax_abs_value_and_upar_bar(
                final_state_for_bootstrap,
                runtime,
                corrected_fluxes,
            )
            final_state_bar = state_pullback_fn(final_state_for_bootstrap, upar_bar)
            _, unpack_pullback = jax.vjp(
                reverse_setup.prepared_rollout.physics_context.unpack_flat,
                final_y_for_objective,
            )
            final_y_bar_rows.append(unpack_pullback(final_state_bar)[0])
            objective_values_rows.append(objective_value)
            if combined_geometry_payload:
                if not callable(geometry_pullback_fn):
                    raise NotImplementedError(
                        "bootstrap_current_softmax_abs_scaled requires compact corrected-Upar "
                        "geometry pullback for combined realtime geometry payloads."
                    )
                geometry = support_payload["geometry"]
                ntx_support = support_payload["ntx_support"]
                geometry_objective_bar = geometry_pullback_fn(
                    final_state_for_bootstrap,
                    upar_bar,
                    geometry,
                    ntx_support,
                )
                support_bar_leaves = support_pullback_fn(
                    final_state_for_bootstrap,
                    upar_bar,
                    ntx_support,
                )
                _, ntx_treedef = jax.tree_util.tree_flatten(ntx_support)
                objective_payload_bar_rows.append(
                    {
                        "geometry": _sanitize_float_delta_bar_tree(geometry, geometry_objective_bar),
                        "ntx_support": ntx_treedef.unflatten(tuple(support_bar_leaves)),
                    }
                )
            else:
                support_bar_leaves = support_pullback_fn(
                    final_state_for_bootstrap,
                    upar_bar,
                    support_payload,
                )
                _, support_treedef = jax.tree_util.tree_flatten(support_payload)
                objective_payload_bar_rows.append(
                    support_treedef.unflatten(tuple(support_bar_leaves))
                )
            continue

        def _objective_from_final_y(final_y_value, objective_index=objective_i):
            final_state = reverse_setup.prepared_rollout.physics_context.unpack_flat(final_y_value)
            return dependencies.objective_scalar_by_index(final_state, runtime, objective_index)

        objective_value, objective_pullback = jax.vjp(_objective_from_final_y, final_y_for_objective)
        objective_values_rows.append(objective_value)
        final_y_bar_rows.append(objective_pullback(jnp.ones_like(objective_value))[0])
        if combined_geometry_payload:
            final_state_for_geometry = reverse_setup.prepared_rollout.physics_context.unpack_flat(
                final_y_for_objective
            )
            geometry = support_payload["geometry"]
            geometry_delta0 = _float_delta_tree_like(geometry)

            def _objective_from_geometry_delta(geometry_delta, objective_index=objective_i):
                runtime_with_geometry = dataclasses.replace(
                    runtime,
                    geometry=_add_float_delta_tree(geometry, geometry_delta),
                )
                return dependencies.objective_scalar_by_index(
                    final_state_for_geometry,
                    runtime_with_geometry,
                    objective_index,
                )

            _, geometry_objective_pullback = jax.vjp(_objective_from_geometry_delta, geometry_delta0)
            (geometry_objective_bar,) = geometry_objective_pullback(jnp.ones_like(objective_value))
            objective_payload_bar_rows.append(
                {
                    "geometry": _sanitize_float_delta_bar_tree(geometry, geometry_objective_bar),
                    "ntx_support": zero_payload_bar["ntx_support"],
                }
            )
        else:
            objective_payload_bar_rows.append(zero_payload_bar)
    objective_values = jnp.stack(objective_values_rows, axis=0)
    final_y_bars = jnp.stack(final_y_bar_rows, axis=0)

    cotangent_mode = str(
        getattr(reverse_setup.execution_context.physics_context, "reverse_stage_cotangent_mode", "full")
    ).strip().lower()
    segment_count = int(jax.tree_util.tree_leaves(segmented_replay_arrays)[0].shape[0])

    reduced_bars = _reverse_reduced_cotangent(
        reverse_setup.execution_context,
        y=final_y_bars,
        lagged_response_cache=_batched_zero_tangent_tree_like(
            segmented_final_carry.lagged_response_cache,
            objective_count,
        ),
        lagged_reference_y=jnp.zeros(
            (objective_count,) + jnp.shape(segmented_final_carry.lagged_reference_y),
            dtype=jnp.asarray(segmented_final_carry.lagged_reference_y).dtype,
        ),
    )
    _zero_support_leaves, support_treedef = jax.tree_util.tree_flatten(zero_payload_bar)
    objective_payload_bar_leaves = tuple(
        jax.tree_util.tree_leaves(payload_bar)
        for payload_bar in objective_payload_bar_rows
    )
    support_bar_leaves = tuple(
        jnp.stack(
            [
                jnp.asarray(objective_payload_bar_leaves[objective_i][leaf_i])
                for objective_i in range(objective_count)
            ],
            axis=0,
        )
        for leaf_i in range(len(_zero_support_leaves))
    )
    objective_support_bar_leaves = support_bar_leaves
    step_support_bar_leaves_accum = tuple(jnp.zeros_like(leaf) for leaf in support_bar_leaves)
    initial_cache_support_bar_leaves_accum = tuple(jnp.zeros_like(leaf) for leaf in support_bar_leaves)
    support_reuse_count = 0
    support_rebuild_count = 0
    phase_start = time.perf_counter()
    for segment_index in range(segment_count - 1, -1, -1):
        segment_start_carry = _take_tree_axis0(segment_start_carries, segment_index)
        segment_arrays = _take_tree_axis0(segmented_replay_arrays, segment_index)
        reduced_bars, segment_support_bar_leaves = (
            _reverse_segment_reduced_cotangent_bwd_batched_with_support_call(
                reverse_setup.execution_context,
                cotangent_mode,
                reduced_bars,
                segment_start_carry,
                segment_arrays,
                support_payload,
            )
        )
        support_bar_leaves = tuple(
            accumulated + increment
            for accumulated, increment in zip(support_bar_leaves, segment_support_bar_leaves)
        )
        step_support_bar_leaves_accum = tuple(
            accumulated + increment
            for accumulated, increment in zip(step_support_bar_leaves_accum, segment_support_bar_leaves)
        )
        segment_lagged_valid = np.asarray(jax.device_get(segment_arrays[6])).reshape(-1)
        support_reuse_count += int(np.count_nonzero(segment_lagged_valid))
        support_rebuild_count += int(segment_lagged_valid.size - np.count_nonzero(segment_lagged_valid))
    reduced_bars, support_bar_leaves = jax.block_until_ready((reduced_bars, support_bar_leaves))
    print(
        f"{progress_prefix} progress: support reverse segmented cotangent sweep ready "
        f"elapsed_s={time.perf_counter() - phase_start:.3f}",
        flush=True,
    )

    initial_lagged_response_valid = bool(np.asarray(jax.device_get(carry0.lagged_response_valid)))
    build_support_pullback = reverse_setup.execution_context.physics_context.flat_rhs_build_support_pullback
    allow_initial_cache_support_pullback = cotangent_mode in {
        "full",
        "full_initial_cache_support_pullback",
        "initial_cache_support_pullback",
    }
    initial_cache_pullback_used = False
    initial_cache_pullback_skipped = False
    if initial_lagged_response_valid and build_support_pullback is not None and allow_initial_cache_support_pullback:
        initial_cache_support_bars = jax.lax.map(
            lambda lagged_bar: build_support_pullback(
                carry0.y,
                lagged_bar,
                support_payload,
            ),
            reduced_bars.lagged_response_cache,
        )
        initial_cache_support_bar_leaves = jax.tree_util.tree_leaves(initial_cache_support_bars)
        support_bar_leaves = tuple(
            accumulated + increment
            for accumulated, increment in zip(support_bar_leaves, initial_cache_support_bar_leaves)
        )
        initial_cache_support_bar_leaves_accum = tuple(
            accumulated + increment
            for accumulated, increment in zip(
                initial_cache_support_bar_leaves_accum,
                initial_cache_support_bar_leaves,
            )
        )
        initial_cache_pullback_used = True
    elif initial_lagged_response_valid and build_support_pullback is not None:
        initial_cache_pullback_skipped = True

    def _full_carry_bar_from_reduced(reduced_bar):
        return dataclasses.replace(
            jax.tree_util.tree_map(_zero_tangent_like, carry0),
            y=reduced_bar.y,
            lagged_response_cache=reduced_bar.lagged_response_cache,
            lagged_reference_y=reduced_bar.lagged_reference_y,
        )

    carry0_bars = jax.vmap(_full_carry_bar_from_reduced)(reduced_bars)
    initial_state_bars = jax.vmap(lambda carry0_bar: initial_state_pullback(carry0_bar)[0])(carry0_bars)
    initial_er_root_support_bars = None
    if initial_er_root_enabled:
        phase_start = time.perf_counter()
        er_profile, finite_mask = dependencies.initial_er_selected_root_profile(
            pre_root_initial_state,
            config=config,
            runtime=runtime,
        )

        er_profile = jnp.asarray(er_profile, dtype=pre_root_initial_state.Er.dtype)
        finite_mask = jnp.asarray(finite_mask, dtype=bool)

        dres_der = dependencies.initial_er_charge_flux_residual_er_derivative(
            pre_root_initial_state,
            er_profile,
            runtime=runtime,
        )
        safe_dres_der = jnp.where(
            jnp.abs(dres_der) > jnp.asarray(1.0e-30, dtype=dres_der.dtype),
            dres_der,
            jnp.inf,
        )
        residual_bars = jnp.where(
            finite_mask[None, :],
            -jnp.asarray(initial_state_bars.Er) / safe_dres_der[None, :],
            0.0,
        )

        state_residual_bars = dependencies.compact_initial_er_state_pullback(
            residual_scalar_fn=dependencies.initial_er_charge_flux_residual_scalar,
            state=pre_root_initial_state,
            er_profile=er_profile,
            residual_bars=residual_bars,
            runtime=runtime,
        )
        direct_initial_state_bars = dataclasses.replace(
            initial_state_bars,
            Er=jnp.zeros_like(initial_state_bars.Er),
        )
        pre_root_initial_state_bars = dependencies.add_trees(
            direct_initial_state_bars,
            state_residual_bars,
        )

        if combined_geometry_payload:
            geometry = support_payload["geometry"]
            ntx_support = support_payload["ntx_support"]
            geometry_delta0 = _float_delta_tree_like(geometry)

            def _residuals_from_geometry_delta(geometry_delta):
                runtime_with_geometry = dependencies.runtime_with_geometry_payload(
                    runtime,
                    _add_float_delta_tree(geometry, geometry_delta),
                )
                runtime_with_geometry = dependencies.runtime_with_ntx_support_payload(
                    runtime_with_geometry,
                    ntx_support,
                )
                return dependencies.initial_er_charge_flux_residuals(
                    pre_root_initial_state,
                    er_profile,
                    runtime=runtime_with_geometry,
                )

            _, geometry_pullback = jax.vjp(_residuals_from_geometry_delta, geometry_delta0)
            geometry_bars = jax.vmap(lambda residual_bar: geometry_pullback(residual_bar)[0])(
                residual_bars
            )
            ntx_runtime = dependencies.runtime_with_geometry_payload(runtime, geometry)
            ntx_bar_leaves = dependencies.compact_initial_er_ntx_support_pullback_leaves(
                runtime=ntx_runtime,
                state=pre_root_initial_state,
                er_profile=er_profile,
                residual_bars=residual_bars,
                support=ntx_support,
            )
            initial_er_root_support_bars = (
                tuple(jax.tree_util.tree_leaves(geometry_bars)) + tuple(ntx_bar_leaves)
            )
        else:
            initial_er_root_support_bars = dependencies.compact_initial_er_ntx_support_pullback_leaves(
                runtime=runtime,
                state=pre_root_initial_state,
                er_profile=er_profile,
                residual_bars=residual_bars,
                support=support_payload,
            )
        pre_root_initial_state_bars, initial_er_root_support_bars = jax.block_until_ready(
            (pre_root_initial_state_bars, initial_er_root_support_bars)
        )
        print(
            f"{progress_prefix} progress: initial-Er root boundary compact pullback ready "
            f"elapsed_s={time.perf_counter() - phase_start:.3f}",
            flush=True,
        )
        initial_state_bars = pre_root_initial_state_bars

    gradient_matrix = jax.vmap(lambda state_bar: profile_state_pullback(state_bar)[0])(initial_state_bars)
    if initial_er_root_support_bars is not None:
        raw_initial_er_root_support_bar_leaves = tuple(initial_er_root_support_bars)
        if len(raw_initial_er_root_support_bar_leaves) != len(support_bar_leaves):
            raise ValueError(
                "Initial-Er root support pullback produced "
                f"{len(raw_initial_er_root_support_bar_leaves)} leaves, but support payload expects "
                f"{len(support_bar_leaves)} leaves."
            )
        for leaf_i, (accumulated, increment) in enumerate(
            zip(support_bar_leaves, raw_initial_er_root_support_bar_leaves, strict=True)
        ):
            if jnp.asarray(increment).shape != jnp.asarray(accumulated).shape:
                raise ValueError(
                    "Initial-Er root support pullback leaf shape mismatch at "
                    f"leaf {leaf_i}: got {jnp.asarray(increment).shape}, "
                    f"expected {jnp.asarray(accumulated).shape}."
                )
        initial_er_root_support_bar_leaves = tuple(
            jnp.zeros_like(accumulated)
            if jnp.asarray(increment).dtype == jax.dtypes.float0
            else jnp.asarray(increment)
            for accumulated, increment in zip(
                support_bar_leaves,
                raw_initial_er_root_support_bar_leaves,
            )
        )
        support_bar_leaves = tuple(
            accumulated + increment
            for accumulated, increment in zip(support_bar_leaves, initial_er_root_support_bar_leaves)
        )
    support_bars = tuple(
        support_treedef.unflatten(
            [jnp.asarray(leaf)[objective_i] for leaf in support_bar_leaves]
        )
        for objective_i in range(objective_count)
    )
    component_support_bars_by_name = {
        "objective_explicit": tuple(
            support_treedef.unflatten(
                [jnp.asarray(leaf)[objective_i] for leaf in objective_support_bar_leaves]
            )
            for objective_i in range(objective_count)
        ),
        "transport_rhs": tuple(
            support_treedef.unflatten(
                [jnp.asarray(leaf)[objective_i] for leaf in step_support_bar_leaves_accum]
            )
            for objective_i in range(objective_count)
        ),
        "initial_cache": tuple(
            support_treedef.unflatten(
                [jnp.asarray(leaf)[objective_i] for leaf in initial_cache_support_bar_leaves_accum]
            )
            for objective_i in range(objective_count)
        ),
    }
    if initial_er_root_support_bars is not None:
        component_support_bars_by_name["initial_er_root"] = tuple(
            support_treedef.unflatten(
                [jnp.asarray(leaf)[objective_i] for leaf in initial_er_root_support_bar_leaves]
            )
            for objective_i in range(objective_count)
        )
    if combined_geometry_payload:
        geometry = support_payload["geometry"]
        geometry_delta0 = _float_delta_tree_like(geometry)

        def _initial_state_from_geometry_delta(geometry_delta):
            runtime_with_geometry = dataclasses.replace(
                runtime,
                geometry=_add_float_delta_tree(geometry, geometry_delta),
            )
            return dependencies.initial_state_for_parameter_vector(
                parameter_values,
                config=config,
                initial_er_root_ad="off",
                baseline_state=baseline_state,
                profile_cfg=profile_cfg,
                runtime=runtime_with_geometry,
            )

        _, initial_geometry_pullback = jax.vjp(_initial_state_from_geometry_delta, geometry_delta0)
        initial_geometry_bars = jax.vmap(
            lambda state_bar: initial_geometry_pullback(state_bar)[0]
        )(initial_state_bars)
        component_support_bars_by_name["initial_profile"] = tuple(
            {
                "geometry": _sanitize_float_delta_bar_tree(
                    support_payload["geometry"],
                    jax.tree_util.tree_map(lambda value: value[objective_i], initial_geometry_bars),
                ),
                "ntx_support": zero_payload_bar["ntx_support"],
            }
            for objective_i in range(objective_count)
        )
        support_bars = tuple(
            {
                "geometry": _sanitize_float_delta_bar_tree(
                    support_payload["geometry"],
                    dependencies.add_trees(
                        support_bar["geometry"],
                        jax.tree_util.tree_map(lambda value: value[objective_i], initial_geometry_bars),
                    ),
                ),
                "ntx_support": support_bar["ntx_support"],
            }
            for objective_i, support_bar in enumerate(support_bars)
        )
    return (
        objective_values,
        gradient_matrix,
        support_bars,
        component_support_bars_by_name,
        support_reuse_count,
        support_rebuild_count,
        initial_cache_pullback_used,
        initial_cache_pullback_skipped,
    )


def realtime_geometry_support_cotangents_from_parameter_vector(
    *,
    reverse_all_objectives_support_payload_bar: Callable[..., object] | None = None,
    profile_values,
    config: Mapping[str, Any],
    baseline_runtime,
    baseline_state,
    profile_cfg: Mapping[str, Any],
    reverse_setup,
    support_payload,
    initial_er_root_ad: str,
    block_until_ready: bool = True,
) -> RealtimeGeometrySupportCotangentResult:
    """Run the grouped all-objective transport support-cotangent pullback.

    By default this uses the internal compact reverse dependencies. A callback
    can still be supplied by legacy benchmark adapters, but optimization callers
    should use the default internal path.
    """

    if reverse_all_objectives_support_payload_bar is None:
        def reverse_all_objectives_support_payload_bar(*args, **kwargs):
            return realtime_geometry_reverse_all_objectives_support_payload_bar_for_parameter_vector(
                *args,
                objective_labels=TRANSPORT_REVERSE_OBJECTIVE_LABELS,
                dependencies=default_realtime_geometry_support_reverse_dependencies(),
                **kwargs,
            )
    if not callable(reverse_all_objectives_support_payload_bar):
        raise TypeError("reverse_all_objectives_support_payload_bar must be callable.")
    callback_result = reverse_all_objectives_support_payload_bar(
        profile_values,
        config=config,
        runtime=baseline_runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        reverse_setup=reverse_setup,
        support_payload=support_payload,
        initial_er_root_ad=initial_er_root_ad,
    )
    if not isinstance(callback_result, tuple) or len(callback_result) != 8:
        raise TypeError(
            "reverse_all_objectives_support_payload_bar must return an 8-tuple: "
            "(objective_values, profile_gradient_matrix, support_bars, "
            "support_component_bars_by_name, support_reuse_count, "
            "support_rebuild_count, initial_cache_pullback_used, "
            "initial_cache_pullback_skipped)."
        )
    (
        objective_values,
        profile_gradient_matrix,
        support_bars,
        support_component_bars_by_name,
        support_reuse_count,
        support_rebuild_count,
        initial_cache_pullback_used,
        initial_cache_pullback_skipped,
    ) = callback_result
    if block_until_ready:
        objective_values, profile_gradient_matrix, support_bars, support_component_bars_by_name = (
            jax.block_until_ready(
                (
                    objective_values,
                    profile_gradient_matrix,
                    support_bars,
                    support_component_bars_by_name,
                )
            )
        )
    return RealtimeGeometrySupportCotangentResult(
        objective_values=objective_values,
        profile_gradient_matrix=profile_gradient_matrix,
        support_bars=support_bars,
        support_component_bars_by_name=support_component_bars_by_name,
        support_reuse_count=int(support_reuse_count),
        support_rebuild_count=int(support_rebuild_count),
        initial_cache_pullback_used=bool(initial_cache_pullback_used),
        initial_cache_pullback_skipped=bool(initial_cache_pullback_skipped),
    )


def realtime_geometry_transport_reverse_support_segment_executor(
    *,
    support_segment_probe: TransportReverseSupportSegmentProbe,
    config: Mapping[str, Any],
    baseline_values,
    baseline_runtime,
    baseline_state,
    profile_cfg: Mapping[str, Any],
    neoclassical_cfg: Mapping[str, Any],
) -> TransportReverseSupportSegmentExecutor:
    """Build an executor wrapper around a segmented reverse probe.

    The supplied probe may be the internal grouped implementation or a
    benchmark diagnostic wrapper. This owns the reusable calling convention:
    grouped optimization paths must request an internal report and receive the
    shared runtime/config inputs.
    """

    if not callable(support_segment_probe):
        raise TypeError("support_segment_probe must be callable.")
    context = realtime_geometry_transport_reverse_table_context(
        config=config,
        baseline_values=baseline_values,
        baseline_runtime=baseline_runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        neoclassical_cfg=neoclassical_cfg,
    )

    def _executor(builder_args, return_report: bool) -> TransportReverseReport:
        if not return_report:
            raise ValueError("Grouped optimization builders require return_report=True.")
        return run_realtime_geometry_support_segment_reverse_table_core(
            support_segment_probe=support_segment_probe,
            args=builder_args,
            context=context,
            suppress_output=False,
        )

    return _executor


def _format_geometry_param_spec(param_spec: tuple[str, int, int]) -> str:
    family, m, n = param_spec
    return f"vmec:{str(family).strip().upper()}:{int(m)}:{int(n)}"


def _parse_reverse_geometry_parameter(parameter_name: str) -> tuple[str, int, int]:
    parts = str(parameter_name).split(":")
    if len(parts) != 3:
        raise ValueError(
            "Reverse geometry parameters must use FAMILY:m:n, for example 'RBC:1:0'."
        )
    family = parts[0].strip().upper()
    if not family:
        raise ValueError("Reverse geometry parameter family cannot be empty.")
    try:
        return family, int(parts[1]), int(parts[2])
    except ValueError as exc:
        raise ValueError(
            "Reverse geometry parameters must use integer m/n values, "
            "for example 'RBC:1:0'."
        ) from exc


def _geometry_context_from_reverse_args(config: Mapping[str, Any], args):
    geometry_parameter = str(getattr(args, "reverse_geometry_parameter", "RBC:1:0"))
    if geometry_parameter.strip().lower() == "all":
        family, m, n = ("RBC", 0, 0)
    else:
        family, m, n = _parse_reverse_geometry_parameter(geometry_parameter)
    geom_cfg = config.get("geometry", {})
    vmec_input_file = geom_cfg.get("vmec_input_file")
    if vmec_input_file is None:
        raise ValueError("Realtime geometry reverse mode requires geometry.vmec_input_file.")
    return build_geometry_autodiff_context(
        vmec_input_file,
        param_family=family,
        param_m=m,
        param_n=n,
        mboz=int(geom_cfg.get("mboz", geom_cfg.get("vmec_mboz", 12))),
        nboz=int(geom_cfg.get("nboz", geom_cfg.get("vmec_nboz", 12))),
    )


def _geometry_param_specs_from_reverse_args(args, geometry_context) -> tuple[tuple[str, int, int], ...]:
    geometry_parameter = str(getattr(args, "reverse_geometry_parameter", "RBC:1:0")).strip()
    if geometry_parameter.lower() != "all":
        return (_parse_reverse_geometry_parameter(geometry_parameter),)
    families = normalize_vmec_boundary_families(
        getattr(args, "reverse_geometry_families", "RBC,ZBS")
    )
    try:
        specs = discover_vmec_boundary_parameter_specs(
            geometry_context,
            families=families,
            nonzero_only=not bool(getattr(args, "reverse_geometry_include_zero_harmonics", False)),
        )
    except ValueError as exc:
        raise ValueError(
            "No VMEC boundary harmonics matched the requested all-harmonic selector. "
            "Try enabling zero harmonics or changing the requested families."
        ) from exc
    return vmec_boundary_tuples(specs)


def _baseline_geometry_delta_vector_for_specs(
    geom_cfg: Mapping[str, Any],
    geometry_param_specs: Sequence[tuple[str, int, int]],
) -> jax.Array:
    deltas = np.zeros((len(geometry_param_specs),), dtype=np.float64)
    configured_delta = float(geom_cfg.get("vmec_param_delta", 0.0))
    if configured_delta != 0.0:
        configured_spec = (
            str(geom_cfg.get("vmec_param_family", "RBC")).strip().upper(),
            int(geom_cfg.get("vmec_param_m", 0)),
            int(geom_cfg.get("vmec_param_n", 0)),
        )
        for i, spec in enumerate(geometry_param_specs):
            normalized_spec = (str(spec[0]).strip().upper(), int(spec[1]), int(spec[2]))
            if normalized_spec == configured_spec:
                deltas[i] = configured_delta
                break
    return jnp.asarray(deltas, dtype=jnp.float64)


def _array_finite_summary(value) -> dict[str, Any]:
    arr = np.asarray(jax.device_get(jnp.asarray(value)))
    finite = np.isfinite(arr)
    finite_values = arr[finite]
    first_nonfinite_index = None
    if not bool(np.all(finite)):
        first_nonfinite_index = [int(i) for i in np.argwhere(~finite)[0].tolist()]
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "all_finite": bool(np.all(finite)),
        "nan_count": int(np.isnan(arr).sum()),
        "posinf_count": int(np.isposinf(arr).sum()),
        "neginf_count": int(np.isneginf(arr).sum()),
        "finite_min": None if finite_values.size == 0 else float(np.min(finite_values)),
        "finite_max": None if finite_values.size == 0 else float(np.max(finite_values)),
        "first_nonfinite_index": first_nonfinite_index,
    }


def geometry_volume_diagnostics(geometry) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for name in (
        "a_b",
        "R0",
        "rho_grid",
        "rho_grid_half",
        "r_grid",
        "r_grid_half",
        "Vprime",
        "Vprime_half",
        "overVprime",
    ):
        if hasattr(geometry, name):
            diagnostics[name] = _array_finite_summary(getattr(geometry, name))
    if hasattr(geometry, "Vprime") and hasattr(geometry, "r_grid"):
        volume = jnp.trapezoid(
            jnp.asarray(geometry.Vprime),
            x=jnp.asarray(geometry.r_grid),
        )
        diagnostics["integrated_volume"] = _array_finite_summary(volume)
        diagnostics["integrated_volume_value"] = float(np.asarray(jax.device_get(volume)))
    return diagnostics


def run_internal_realtime_geometry_support_segment_probe(
    *,
    args,
    context: RealtimeGeometryTransportReverseTableContext,
    return_report: bool = True,
    suppress_diagnostics: bool = True,
) -> TransportReverseReport:
    """Run the benchmark-matched grouped realtime-geometry reverse table internally.

    This is the all-objective grouped path used by optimization callers. It
    keeps the compact support cotangent sweep and raw-block VMEC payload
    pullback together, without importing benchmark modules.
    """

    if not return_report:
        raise ValueError("Internal grouped realtime-geometry reverse requires return_report=True.")
    if str(getattr(args, "objective", "all")) != "all":
        raise ValueError("Internal grouped realtime-geometry reverse currently requires objective='all'.")
    config = context.config
    baseline_runtime = context.baseline_runtime
    baseline_state = context.baseline_state
    profile_cfg = context.profile_cfg
    neoclassical_cfg = context.neoclassical_cfg
    core_setup = prepare_realtime_geometry_support_segment_core_setup(
        args=args,
        config=config,
        baseline_values=context.baseline_values,
        baseline_runtime=baseline_runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        neoclassical_cfg=neoclassical_cfg,
        parameter_order=TRANSPORT_REVERSE_PROFILE_PARAMETER_ORDER,
        find_ntx_support_payload=find_ntx_support_payload,
        prepare_reverse_static_setup=prepare_reverse_static_setup,
        geometry_volume_diagnostics=None if suppress_diagnostics else geometry_volume_diagnostics,
    )
    t_start = time.perf_counter()
    t_phase = time.perf_counter()
    support_cotangent_result = realtime_geometry_support_cotangents_from_parameter_vector(
        profile_values=core_setup.profile_values,
        config=config,
        baseline_runtime=baseline_runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        reverse_setup=core_setup.reverse_setup,
        support_payload=core_setup.support_payload,
        initial_er_root_ad=getattr(args, "initial_er_root_ad", "off"),
    )
    if not suppress_diagnostics:
        print(
            "[autodiff-gate] progress: transport reverse profile/support cotangents complete "
            f"elapsed_s={time.perf_counter() - t_phase:.3f}",
            flush=True,
        )
    geom_cfg = config.get("geometry", {})
    geometry_context = _geometry_context_from_reverse_args(config, args)
    geometry_param_specs = _geometry_param_specs_from_reverse_args(args, geometry_context)
    geometry_param_entries = boundary_param_entries(geometry_context, geometry_param_specs)
    geometry_param_labels = tuple(_format_geometry_param_spec(spec) for spec in geometry_param_specs)
    baseline_geometry_deltas = _baseline_geometry_delta_vector_for_specs(
        geom_cfg,
        geometry_param_specs,
    )
    t_phase = time.perf_counter()
    assembly_result = realtime_geometry_transport_reverse_table_from_payload_cotangents(
        objective_labels=TRANSPORT_REVERSE_OBJECTIVE_LABELS,
        profile_parameter_labels=TRANSPORT_REVERSE_PROFILE_PARAMETER_ORDER,
        geometry_parameter_labels=geometry_param_labels,
        objective_values=support_cotangent_result.objective_values,
        profile_gradient_matrix=support_cotangent_result.profile_gradient_matrix,
        geometry_context=geometry_context,
        baseline_geometry_deltas=baseline_geometry_deltas,
        geometry_param_specs=geometry_param_specs,
        support_bars=tuple(support_cotangent_result.support_bars),
        support_component_bars_by_name=support_cotangent_result.support_component_bars_by_name,
        include_component_pullbacks=bool(getattr(args, "realtime_geometry_component_pullbacks", False)),
        combined_geometry_payload=core_setup.combined_geometry_payload,
        n_r=int(geom_cfg.get("n_radial", 51)),
        n_theta=int(neoclassical_cfg.get("ntx_exact_n_theta", 25)),
        n_zeta=int(neoclassical_cfg.get("ntx_exact_n_zeta", 25)),
        n_xi=int(neoclassical_cfg.get("ntx_exact_n_xi", 64)),
        surface_backend=core_setup.ntx_surface_backend,
        max_iter=geom_cfg.get("vmec_max_iter"),
        solver_device=str(geom_cfg.get("vmec_implicit_solver_device", "default")),
        progress_label=None if suppress_diagnostics else "[autodiff-gate] realtime geometry payload pullback:",
        return_branch_gradients=False,
    )
    if not suppress_diagnostics:
        print(
            "[autodiff-gate] progress: geometry support pullback complete "
            f"mode={assembly_result.payload_pullback_result.pullback_mode} "
            f"elapsed_s={time.perf_counter() - t_phase:.3f}",
            flush=True,
        )
    table_result = assembly_result.table_result
    metadata_entries = realtime_geometry_transport_reverse_metadata_entries(
        parameter_mode=str(getattr(args, "reverse_parameter_mode", "profiles_plus_realtime_geometry")),
        config_path=str(getattr(args, "config", "")),
        objective_labels=TRANSPORT_REVERSE_OBJECTIVE_LABELS,
        profile_parameter_labels=TRANSPORT_REVERSE_PROFILE_PARAMETER_ORDER,
        profile_values=core_setup.profile_values,
        geometry_parameter_labels=geometry_param_labels,
        geometry_parameter_entries=geometry_param_entries,
        baseline_geometry_deltas=baseline_geometry_deltas,
        geometry_parameter_specs=geometry_param_specs,
        geometry_parameter_selector=str(getattr(args, "reverse_geometry_parameter", "RBC:1:0")),
        accepted_step_limit=None
        if getattr(args, "accepted_step_limit", None) is None
        else int(getattr(args, "accepted_step_limit")),
        reverse_segment_length=None
        if getattr(args, "reverse_segment_length", None) is None
        else int(getattr(args, "reverse_segment_length")),
        reverse_stage_cotangent_mode_requested=str(getattr(args, "reverse_stage_cotangent_mode", "full")),
        reverse_stage_cotangent_mode_effective=core_setup.support_probe_cotangent_mode,
        ntx_exact_derivative_mode=str(getattr(args, "ntx_exact_derivative_mode", "direct")),
        ntx_exact_derivative_field_pullback_mode=str(
            getattr(args, "ntx_exact_derivative_field_pullback_mode", "compact_vjp")
        ),
        ntx_exact_surface_backend=str(neoclassical_cfg.get("ntx_exact_surface_backend", "booz")),
        realtime_geometry_gradient_path=str(getattr(args, "realtime_geometry_gradient_path", "reverse_payload")),
        realtime_geometry_component_pullbacks=bool(getattr(args, "realtime_geometry_component_pullbacks", False)),
        realtime_geometry_support_bar_diagnostics_skipped=bool(suppress_diagnostics),
        realtime_geometry_derivative_complete=bool(core_setup.combined_geometry_payload),
        geometry_support_pullback_mode=assembly_result.payload_pullback_result.pullback_mode,
        realtime_geometry_diagnostics={},
        support_payload_summary={},
        support_bar_summary_by_objective={},
        support_bar_l2_by_objective={},
        support_bar_branch_diagnostics_by_objective={},
        support_reuse_count=support_cotangent_result.support_reuse_count,
        support_rebuild_count=support_cotangent_result.support_rebuild_count,
        support_initial_cache_pullback_used=support_cotangent_result.initial_cache_pullback_used,
        support_initial_cache_pullback_skipped=support_cotangent_result.initial_cache_pullback_skipped,
        elapsed_s=time.perf_counter() - t_start,
    )
    return {
        **metadata_entries,
        **transport_reverse_table_report_entries(table_result=table_result),
        "geometry_gradient_reverse_ad_by_branch": None,
        "geometry_gradient_reverse_ad_by_component": {},
        "geometry_gradient_reverse_ad_final_state_components": {},
        "geometry_gradient_reverse_ad_by_component_and_branch": {},
        "transport_reverse_table_result": table_result,
    }


def run_realtime_geometry_support_segment_reverse_table_core(
    *,
    support_segment_probe: TransportReverseSupportSegmentProbe,
    args,
    context: RealtimeGeometryTransportReverseTableContext,
    suppress_output: bool = True,
) -> TransportReverseReport:
    """Run the segmented support probe as a non-printing reverse table core.

    This function owns the internal core calling convention: execute the probe
    with `return_report=True`, thread the table context, and require the
    JAX-native table result in the returned report.
    """

    if not callable(support_segment_probe):
        raise TypeError("support_segment_probe must be callable.")

    def _call_probe() -> TransportReverseReport:
        return support_segment_probe(
            args=args,
            config=context.config,
            baseline_values=context.baseline_values,
            baseline_runtime=context.baseline_runtime,
            baseline_state=context.baseline_state,
            profile_cfg=context.profile_cfg,
            neoclassical_cfg=context.neoclassical_cfg,
            return_report=True,
        )

    if suppress_output:
        with contextlib.redirect_stdout(io.StringIO()):
            report = _call_probe()
    else:
        report = _call_probe()
    table_result = report.get("transport_reverse_table_result") if isinstance(report, Mapping) else None
    if not isinstance(table_result, RealtimeGeometryTransportReverseTableResult):
        raise TypeError(
            "Segmented realtime-geometry reverse core did not return a "
            "RealtimeGeometryTransportReverseTableResult under "
            "'transport_reverse_table_result'."
        )
    return report


def _copy_args_with_objective_all(args):
    copied_args = copy.copy(args)
    setattr(copied_args, "objective", "all")
    return copied_args


def realtime_geometry_transport_reverse_grouped_inputs(
    *,
    args,
    config: Mapping[str, Any],
    baseline_values,
    baseline_runtime,
    baseline_state,
    profile_cfg: Mapping[str, Any],
    neoclassical_cfg: Mapping[str, Any],
    support_segment_executor: TransportReverseSupportSegmentExecutor,
    update_args_for_all_objectives: TransportReverseArgsUpdater | None = None,
) -> RealtimeGeometryTransportReverseGroupedInputs:
    """Build shared context and grouped runner for realtime-geometry reverse tables.

    The support executor owns the segmented reverse sweep. This helper owns the
    reusable context construction and objective='all' grouped-runner contract.
    """

    table_context = realtime_geometry_transport_reverse_table_context(
        config=config,
        baseline_values=baseline_values,
        baseline_runtime=baseline_runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        neoclassical_cfg=neoclassical_cfg,
    )
    update_args = (
        _copy_args_with_objective_all
        if update_args_for_all_objectives is None
        else update_args_for_all_objectives
    )
    return RealtimeGeometryTransportReverseGroupedInputs(
        table_context=table_context,
        run_grouped_report=realtime_geometry_transport_reverse_grouped_runner(
            args=args,
            support_segment_executor=support_segment_executor,
            update_args_for_all_objectives=update_args,
        ),
    )


def realtime_geometry_transport_reverse_table_request(
    *,
    objective_names: Sequence[str],
    parameter_set: ReverseADParameterSet,
    context: RealtimeGeometryTransportReverseTableContext,
    options: Mapping[str, object] | None = None,
) -> RealtimeGeometryTransportReverseTableRequest:
    """Build and validate a grouped realtime-geometry reverse table request."""

    return RealtimeGeometryTransportReverseTableRequest(
        objective_names=tuple(str(name) for name in objective_names),
        parameter_set=parameter_set,
        context=context,
        options=options,
    )


def realtime_geometry_transport_reverse_table_result(
    *,
    objective_labels: Sequence[str],
    profile_parameter_labels: Sequence[str],
    geometry_parameter_labels: Sequence[str],
    objective_values,
    profile_gradient_matrix,
    geometry_gradient_matrix,
) -> RealtimeGeometryTransportReverseTableResult:
    """Build the JAX-native grouped transport reverse table result."""

    return RealtimeGeometryTransportReverseTableResult(
        objective_labels=tuple(str(name) for name in objective_labels),
        profile_parameter_labels=tuple(str(name) for name in profile_parameter_labels),
        geometry_parameter_labels=tuple(str(name) for name in geometry_parameter_labels),
        objective_values=objective_values,
        profile_gradient_matrix=profile_gradient_matrix,
        geometry_gradient_matrix=geometry_gradient_matrix,
    )


def _validate_transport_reverse_parameter_set(parameter_set: ReverseADParameterSet) -> None:
    """Validate the transport reverse parameter layout placeholder.

    The current report runner still owns parameter handling.  Keeping this
    explicit hook makes the production seam ready for step 2 without changing
    the public helper signature.
    """

    if not isinstance(parameter_set, ReverseADParameterSet):
        raise TypeError(
            "parameter_set must be a ReverseADParameterSet; "
            f"got {type(parameter_set).__name__}."
        )


def normalize_transport_objective_names(
    objective_names: Sequence[str],
    *,
    objective_labels: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Validate and normalize grouped transport objective names."""

    requested_objectives = tuple(str(name).strip() for name in objective_names)
    if not requested_objectives or any(not name for name in requested_objectives):
        raise ValueError("At least one non-empty transport objective name is required.")
    if objective_labels is not None:
        available_labels = tuple(str(name) for name in objective_labels)
        missing = tuple(name for name in requested_objectives if name not in available_labels)
        if missing:
            available = ", ".join(available_labels)
            raise ValueError(
                "Grouped transport reverse report builder received unknown "
                f"transport objectives {missing!r}. Available objectives: {available}."
            )
    return requested_objectives


def build_realtime_geometry_transport_reverse_report(
    *,
    objective_names: Sequence[str],
    parameter_set: ReverseADParameterSet,
    objective_labels: Sequence[str],
    run_grouped_report: TransportReverseReportRunner,
    options: Mapping[str, object] | None = None,
    quiet_default: bool = True,
) -> TransportReverseReport:
    """Build a grouped realtime-geometry transport reverse report.

    This is the production-facing seam for the validated all-objective reverse
    path.  For now the heavy reverse execution is supplied as
    `run_grouped_report`; later extraction can move that runner into this
    module without changing the optimization backend contract.
    """

    _validate_transport_reverse_parameter_set(parameter_set)
    normalize_transport_objective_names(objective_names, objective_labels=objective_labels)
    quiet_option = quiet_default if options is None else options.get("quiet", quiet_default)
    quiet = bool(quiet_option)
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            return run_grouped_report()
    return run_grouped_report()


def run_realtime_geometry_transport_reverse_table(
    *,
    objective_names: Sequence[str],
    parameter_set: ReverseADParameterSet,
    objective_labels: Sequence[str],
    run_grouped_report: TransportReverseReportRunner,
    options: Mapping[str, object] | None = None,
    quiet_default: bool = True,
) -> TransportReverseReport:
    """Run the validated grouped realtime-geometry transport reverse table.

    This is the named internal execution seam for the production optimization
    lane. For realtime-geometry transport, the grouped report runner is the
    validated memory/graph behavior used by the benchmark. Direct table-result
    builders should only become the default after matching this path.
    """

    return build_realtime_geometry_transport_reverse_report(
        objective_names=objective_names,
        parameter_set=parameter_set,
        objective_labels=objective_labels,
        run_grouped_report=run_grouped_report,
        options=options,
        quiet_default=quiet_default,
    )


def build_realtime_geometry_transport_reverse_table_result(
    *,
    objective_names: Sequence[str],
    parameter_set: ReverseADParameterSet,
    objective_labels: Sequence[str],
    run_grouped_report: TransportReverseReportRunner,
    options: Mapping[str, object] | None = None,
    quiet_default: bool = True,
) -> RealtimeGeometryTransportReverseTableResult:
    """Run the validated grouped path and extract its JAX-native table result."""

    report = build_realtime_geometry_transport_reverse_report(
        objective_names=objective_names,
        parameter_set=parameter_set,
        objective_labels=objective_labels,
        run_grouped_report=run_grouped_report,
        options=options,
        quiet_default=quiet_default,
    )
    table_result = report.get("transport_reverse_table_result") if isinstance(report, Mapping) else None
    if not isinstance(table_result, RealtimeGeometryTransportReverseTableResult):
        raise TypeError(
            "Grouped realtime-geometry reverse runner did not return a "
            "RealtimeGeometryTransportReverseTableResult under "
            "'transport_reverse_table_result'."
        )
    return table_result


def transport_realtime_geometry_reverse_table(
    *,
    request: RealtimeGeometryTransportReverseTableRequest | None = None,
    objective_names: Sequence[str] | None = None,
    parameter_set: ReverseADParameterSet | None = None,
    context: RealtimeGeometryTransportReverseTableContext | None = None,
    table_result_builder: TransportReverseTableResultBuilder | None = None,
    run_grouped_report: TransportReverseReportRunner | None = None,
    objective_labels: Sequence[str] | None = None,
    options: Mapping[str, object] | None = None,
    quiet_default: bool = True,
) -> RealtimeGeometryTransportReverseTableResult:
    """Evaluate the realtime-geometry transport reverse table.

    This is the stable internal API boundary for optimization callers. For the
    realtime-geometry transport lane, `run_grouped_report` is currently the
    benchmark-matched path. A direct `table_result_builder` is still accepted
    for controlled experiments, but it is not the validated default.
    """

    if request is None:
        if objective_names is None:
            raise ValueError("objective_names must be provided when request is omitted.")
        if parameter_set is None:
            raise ValueError("parameter_set must be provided when request is omitted.")
        if context is None:
            raise ValueError("context must be provided when request is omitted.")
        request = realtime_geometry_transport_reverse_table_request(
            objective_names=objective_names,
            parameter_set=parameter_set,
            context=context,
            options=options,
        )
    else:
        if objective_names is not None or parameter_set is not None or context is not None:
            raise ValueError(
                "Pass either request or objective_names/parameter_set/context, not both."
            )
        if options is not None:
            request = dataclasses.replace(request, options=options)

    if table_result_builder is not None and run_grouped_report is not None:
        raise ValueError("Pass only one of table_result_builder or run_grouped_report.")
    if table_result_builder is not None:
        return table_result_builder(
            request.objective_names,
            request.parameter_set,
            request.options,
        )
    if run_grouped_report is None:
        raise ValueError("Either table_result_builder or run_grouped_report must be provided.")
    if objective_labels is None:
        raise ValueError("objective_labels must be provided with run_grouped_report.")
    return build_realtime_geometry_transport_reverse_table_result(
        objective_names=request.objective_names,
        parameter_set=request.parameter_set,
        objective_labels=objective_labels,
        run_grouped_report=run_grouped_report,
        options=request.options,
        quiet_default=quiet_default,
    )


def internal_realtime_geometry_transport_reverse_table_result_builder(
    *,
    table_context: RealtimeGeometryTransportReverseTableContext,
    geometry_context,
    baseline_geometry_deltas=None,
    combined_geometry_payload: bool = True,
    n_r: int = 51,
    n_theta: int = 25,
    n_zeta: int = 25,
    n_xi: int = 64,
    surface_backend: str = "vmec",
    max_iter=None,
    solver_device: str = "default",
    accepted_step_limit: int | None = None,
    reverse_segment_length: int | None = 1,
    initial_er_root_ad: str = "off",
    reverse_stage_adjoint_solve_mode: str = "bicgstab",
    reverse_rhs_transpose_mode: str = "explicit_ntx_interpolated",
    reverse_stage_cotangent_mode: str = "full",
    reverse_step_bwd_mode: str = "reduced_cotangent",
    reverse_stage_adjoint_memory_mode: str = "default",
    reverse_stage_adjoint_iter_maxiter: int = 40,
    reverse_stage_adjoint_iter_tol: float = 1.0e-10,
    max_reverse_accepted_steps: int | None = None,
    progress_label: str | None = None,
    raw_block_solve: GeometryRawBlockSolve | None = None,
) -> TransportReverseTableResultBuilder:
    """Build an experimental direct full transport reverse table builder.

    This is useful for extraction work, but the grouped runner remains the
    validated optimization path until this builder is proven to match the
    benchmark memory behavior.
    """

    if not isinstance(table_context, RealtimeGeometryTransportReverseTableContext):
        raise TypeError("table_context must be a RealtimeGeometryTransportReverseTableContext.")
    baseline_profile_values = jnp.asarray(
        table_context.baseline_values[: len(TRANSPORT_REVERSE_PROFILE_PARAMETER_ORDER)]
    )

    def _row_indices(objective_names: Sequence[str]) -> tuple[int, ...]:
        labels = tuple(TRANSPORT_REVERSE_OBJECTIVE_LABELS)
        lookup = {name: i for i, name in enumerate(labels)}
        return tuple(lookup[str(name)] for name in normalize_transport_objective_names(objective_names, objective_labels=labels))

    def _builder(
        objective_names: tuple[str, ...],
        parameter_set: ReverseADParameterSet,
        options: Mapping[str, object] | None,
    ) -> RealtimeGeometryTransportReverseTableResult:
        _validate_transport_reverse_parameter_set(parameter_set)
        opts = {} if options is None else dict(options)
        active_accepted_step_limit = opts.get("accepted_step_limit", accepted_step_limit)
        active_reverse_segment_length = opts.get("reverse_segment_length", reverse_segment_length)
        active_max_reverse_accepted_steps = opts.get(
            "max_reverse_accepted_steps",
            max_reverse_accepted_steps,
        )
        active_initial_er_root_ad = str(opts.get("initial_er_root_ad", initial_er_root_ad))
        active_raw_block_solve = opts.get("raw_block_solve", raw_block_solve)
        active_profile_values = jnp.asarray(
            opts.get("profile_values", baseline_profile_values),
            dtype=baseline_profile_values.dtype,
        )
        active_runtime = table_context.baseline_runtime
        active_support_payload = None
        use_runtime_payload = bool(opts.get("use_runtime_payload", active_raw_block_solve is None))
        if active_raw_block_solve is not None and not use_runtime_payload:
            active_support_payload = build_neopax_geometry_and_ntx_exact_lij_support_from_state(
                geometry_context,
                active_raw_block_solve.state,
                n_r=int(opts.get("n_r", n_r)),
                n_theta=int(opts.get("n_theta", n_theta)),
                n_zeta=int(opts.get("n_zeta", n_zeta)),
                n_xi=int(opts.get("n_xi", n_xi)),
                surface_backend=str(opts.get("surface_backend", surface_backend)),
            )
            active_runtime = runtime_with_geometry_payload(
                active_runtime,
                active_support_payload["geometry"],
            )
            active_runtime = runtime_with_ntx_support_payload(
                active_runtime,
                active_support_payload["ntx_support"],
            )
        active_reverse_setup = prepare_reverse_static_setup(
            active_profile_values,
            config=table_context.config,
            runtime=active_runtime,
            baseline_state=table_context.baseline_state,
            profile_cfg=table_context.profile_cfg,
            initial_er_root_ad=active_initial_er_root_ad,
            accepted_step_limit_override=(
                None if active_accepted_step_limit is None else int(active_accepted_step_limit)
            ),
            reverse_segment_length=(
                None
                if active_reverse_segment_length is None
                else active_reverse_segment_length
                if isinstance(active_reverse_segment_length, str)
                else int(active_reverse_segment_length)
            ),
            reverse_direct_stage_adjoint=True,
            reverse_stage_adjoint_solve_mode=str(
                opts.get("reverse_stage_adjoint_solve_mode", reverse_stage_adjoint_solve_mode)
            ),
            reverse_rhs_transpose_mode=str(opts.get("reverse_rhs_transpose_mode", reverse_rhs_transpose_mode)),
            reverse_stage_cotangent_mode=str(opts.get("reverse_stage_cotangent_mode", reverse_stage_cotangent_mode)),
            reverse_step_bwd_mode=str(opts.get("reverse_step_bwd_mode", reverse_step_bwd_mode)),
            reverse_stage_adjoint_memory_mode=str(
                opts.get("reverse_stage_adjoint_memory_mode", reverse_stage_adjoint_memory_mode)
            ),
            reverse_stage_adjoint_iter_maxiter=int(
                opts.get("reverse_stage_adjoint_iter_maxiter", reverse_stage_adjoint_iter_maxiter)
            ),
            reverse_stage_adjoint_iter_tol=float(
                opts.get("reverse_stage_adjoint_iter_tol", reverse_stage_adjoint_iter_tol)
            ),
            max_reverse_accepted_steps=(
                None
                if active_max_reverse_accepted_steps is None
                else int(active_max_reverse_accepted_steps)
            ),
        )
        ntx_support_payload = (
            active_support_payload["ntx_support"]
            if active_support_payload is not None
            else find_ntx_support_payload(active_runtime)
        )
        support_payload = (
            {
                "geometry": (
                    active_support_payload["geometry"]
                    if active_support_payload is not None
                    else active_runtime.geometry
                ),
                "ntx_support": ntx_support_payload,
            }
            if combined_geometry_payload
            else ntx_support_payload
        )
        support_result = realtime_geometry_support_cotangents_from_parameter_vector(
            profile_values=active_profile_values,
            config=table_context.config,
            baseline_runtime=active_runtime,
            baseline_state=table_context.baseline_state,
            profile_cfg=table_context.profile_cfg,
            reverse_setup=active_reverse_setup,
            support_payload=support_payload,
            initial_er_root_ad=active_initial_er_root_ad,
        )
        rows = _row_indices(objective_names)
        support_bars = tuple(support_result.support_bars[i] for i in rows)
        component_bars = {
            name: tuple(values[i] for i in rows)
            for name, values in support_result.support_component_bars_by_name.items()
        }
        all_objective_values = jnp.asarray(support_result.objective_values)
        all_profile_gradient_matrix = jnp.asarray(support_result.profile_gradient_matrix)
        if int(all_profile_gradient_matrix.shape[1]) == len(PROFILE_PARAMETER_ORDER):
            all_profile_parameter_labels = PROFILE_PARAMETER_ORDER
        else:
            all_profile_parameter_labels = TRANSPORT_REVERSE_PROFILE_PARAMETER_ORDER
        profile_lookup = {
            name: i for i, name in enumerate(all_profile_parameter_labels)
        }
        profile_cols = tuple(profile_lookup[spec.name] for spec in parameter_set.profile_specs)
        selected_objective_values = all_objective_values[jnp.asarray(rows, dtype=jnp.int32)]
        selected_profile_matrix = all_profile_gradient_matrix[
            jnp.asarray(rows, dtype=jnp.int32),
            :,
        ]
        if profile_cols:
            selected_profile_matrix = selected_profile_matrix[:, jnp.asarray(profile_cols, dtype=jnp.int32)]
        else:
            selected_profile_matrix = selected_profile_matrix[:, :0]
        vmec_specs = tuple(spec.as_tuple() for spec in parameter_set.vmec_boundary_specs)
        if baseline_geometry_deltas is None:
            active_baseline_geometry_deltas = jnp.zeros((len(vmec_specs),), dtype=jnp.float64)
        else:
            active_baseline_geometry_deltas = jnp.asarray(baseline_geometry_deltas, dtype=jnp.float64)
        assembly = realtime_geometry_transport_reverse_table_from_payload_cotangents(
            objective_labels=objective_names,
            profile_parameter_labels=tuple(spec.label for spec in parameter_set.profile_specs),
            geometry_parameter_labels=tuple(spec.vmec_label for spec in parameter_set.vmec_boundary_specs),
            objective_values=selected_objective_values,
            profile_gradient_matrix=selected_profile_matrix,
            geometry_context=geometry_context,
            baseline_geometry_deltas=active_baseline_geometry_deltas,
            geometry_param_specs=vmec_specs,
            support_bars=support_bars,
            support_component_bars_by_name=component_bars,
            include_component_pullbacks=False,
            combined_geometry_payload=combined_geometry_payload,
            n_r=int(opts.get("n_r", n_r)),
            n_theta=int(opts.get("n_theta", n_theta)),
            n_zeta=int(opts.get("n_zeta", n_zeta)),
            n_xi=int(opts.get("n_xi", n_xi)),
            surface_backend=str(opts.get("surface_backend", surface_backend)),
            max_iter=opts.get("max_iter", max_iter),
            solver_device=str(opts.get("solver_device", solver_device)),
            progress_label=progress_label,
            return_branch_gradients=bool(opts.get("return_branch_gradients", False)),
            raw_block_solve=active_raw_block_solve,
        )
        return assembly.table_result

    return _builder


def realtime_geometry_payload_pullback_result(
    *,
    geometry_context,
    baseline_geometry_deltas,
    geometry_param_specs: Sequence[tuple[str, int, int]],
    support_bars: Sequence[object],
    support_component_bars_by_name: Mapping[str, Sequence[object]] | None = None,
    include_component_pullbacks: bool = False,
    combined_geometry_payload: bool = True,
    n_r: int = 51,
    n_theta: int = 25,
    n_zeta: int = 25,
    n_xi: int = 64,
    surface_backend: str = "booz",
    max_iter=None,
    solver_device: str = "default",
    progress_label: str | None = None,
    raw_block_solve: GeometryRawBlockSolve | None = None,
    return_branch_gradients: bool = True,
) -> RealtimeGeometryPayloadPullbackResult:
    """Pull transport support-payload cotangents back to VMEC boundary harmonics.

    This is the raw-block-transpose path validated by the geometry benchmarks.
    It only moves the benchmark's existing orchestration into an internal helper;
    callers still supply the already-computed support cotangents.
    """

    component_bars = {} if support_component_bars_by_name is None else dict(support_component_bars_by_name)
    support_component_names = tuple(component_bars) if include_component_pullbacks else tuple()
    geometry_pullback_payload_bars = tuple(support_bars)
    for component_name in support_component_names:
        geometry_pullback_payload_bars = (
            *geometry_pullback_payload_bars,
            *tuple(component_bars[component_name]),
        )

    geometry_gradient_result = geometry_payload_pullback_from_param_vector_raw_block_transpose(
        geometry_context,
        baseline_geometry_deltas,
        tuple(geometry_param_specs),
        geometry_pullback_payload_bars,
        combined_payload=combined_geometry_payload,
        n_r=int(n_r),
        n_theta=int(n_theta),
        n_zeta=int(n_zeta),
        n_xi=int(n_xi),
        surface_backend=surface_backend,
        max_iter=max_iter,
        solver_device=str(solver_device),
        progress_label=progress_label,
        return_branch_gradients=bool(return_branch_gradients),
        raw_block_solve=raw_block_solve,
    )
    geometry_gradient_result = jax.block_until_ready(geometry_gradient_result)
    objective_count = int(len(tuple(support_bars)))

    def _split_component_rows(matrix):
        if matrix is None:
            return None, {}
        total_matrix = matrix[:objective_count]
        component_matrices = {}
        row0 = objective_count
        for component_name in support_component_names:
            row1 = row0 + objective_count
            component_matrices[component_name] = matrix[row0:row1]
            row0 = row1
        return total_matrix, component_matrices

    if isinstance(geometry_gradient_result, Mapping):
        geometry_gradient_matrix, component_gradient_matrices = _split_component_rows(
            geometry_gradient_result["combined"]
        )
        geometry_branch_gradient_matrix, component_geometry_branch_matrices = _split_component_rows(
            geometry_gradient_result.get("geometry")
        )
        ntx_support_branch_gradient_matrix, component_ntx_support_branch_matrices = _split_component_rows(
            geometry_gradient_result.get("ntx_support")
        )
    else:
        geometry_gradient_matrix, component_gradient_matrices = _split_component_rows(
            geometry_gradient_result
        )
        geometry_branch_gradient_matrix = None
        ntx_support_branch_gradient_matrix = None
        component_geometry_branch_matrices = {}
        component_ntx_support_branch_matrices = {}

    return RealtimeGeometryPayloadPullbackResult(
        geometry_gradient_matrix=geometry_gradient_matrix,
        geometry_branch_gradient_matrix=geometry_branch_gradient_matrix,
        ntx_support_branch_gradient_matrix=ntx_support_branch_gradient_matrix,
        component_gradient_matrices=component_gradient_matrices,
        component_geometry_branch_matrices=component_geometry_branch_matrices,
        component_ntx_support_branch_matrices=component_ntx_support_branch_matrices,
    )


def realtime_geometry_transport_reverse_table_from_payload_cotangents(
    *,
    objective_labels: Sequence[str],
    profile_parameter_labels: Sequence[str],
    geometry_parameter_labels: Sequence[str],
    objective_values,
    profile_gradient_matrix,
    geometry_context,
    baseline_geometry_deltas,
    geometry_param_specs: Sequence[tuple[str, int, int]],
    support_bars: Sequence[object],
    support_component_bars_by_name: Mapping[str, Sequence[object]] | None = None,
    include_component_pullbacks: bool = False,
    combined_geometry_payload: bool = True,
    n_r: int = 51,
    n_theta: int = 25,
    n_zeta: int = 25,
    n_xi: int = 64,
    surface_backend: str = "booz",
    max_iter=None,
    solver_device: str = "default",
    progress_label: str | None = None,
    raw_block_solve: GeometryRawBlockSolve | None = None,
    return_branch_gradients: bool = True,
) -> RealtimeGeometryTransportReverseAssemblyResult:
    """Assemble the JAX transport reverse table from support-payload cotangents."""

    payload_pullback_result = realtime_geometry_payload_pullback_result(
        geometry_context=geometry_context,
        baseline_geometry_deltas=baseline_geometry_deltas,
        geometry_param_specs=geometry_param_specs,
        support_bars=support_bars,
        support_component_bars_by_name=support_component_bars_by_name,
        include_component_pullbacks=include_component_pullbacks,
        combined_geometry_payload=combined_geometry_payload,
        n_r=n_r,
        n_theta=n_theta,
        n_zeta=n_zeta,
        n_xi=n_xi,
        surface_backend=surface_backend,
        max_iter=max_iter,
        solver_device=solver_device,
        progress_label=progress_label,
        raw_block_solve=raw_block_solve,
        return_branch_gradients=bool(return_branch_gradients),
    )
    table_result = realtime_geometry_transport_reverse_table_result(
        objective_labels=objective_labels,
        profile_parameter_labels=profile_parameter_labels,
        geometry_parameter_labels=geometry_parameter_labels,
        objective_values=objective_values,
        profile_gradient_matrix=profile_gradient_matrix,
        geometry_gradient_matrix=payload_pullback_result.geometry_gradient_matrix,
    )
    return RealtimeGeometryTransportReverseAssemblyResult(
        table_result=table_result,
        payload_pullback_result=payload_pullback_result,
    )


def transport_reverse_table_report_entries(
    *,
    table_result: RealtimeGeometryTransportReverseTableResult | None = None,
    objective_labels: Sequence[str] | None = None,
    profile_parameter_labels: Sequence[str] | None = None,
    geometry_parameter_labels: Sequence[str] | None = None,
    objective_values=None,
    profile_gradient_matrix=None,
    geometry_gradient_matrix=None,
) -> dict[str, object]:
    """Assemble reusable non-printing transport reverse table report entries."""

    if table_result is not None:
        objective_labels = table_result.objective_labels
        profile_parameter_labels = table_result.profile_parameter_labels
        geometry_parameter_labels = table_result.geometry_parameter_labels
        objective_values = table_result.objective_values
        profile_gradient_matrix = table_result.profile_gradient_matrix
        geometry_gradient_matrix = table_result.geometry_gradient_matrix
    if (
        objective_labels is None
        or profile_parameter_labels is None
        or geometry_parameter_labels is None
        or objective_values is None
        or profile_gradient_matrix is None
        or geometry_gradient_matrix is None
    ):
        raise TypeError(
            "transport_reverse_table_report_entries requires either table_result or "
            "all explicit labels and arrays."
        )
    objective_names = tuple(str(name) for name in objective_labels)
    profile_names = tuple(str(name) for name in profile_parameter_labels)
    geometry_names = tuple(str(name) for name in geometry_parameter_labels)
    if len(set(objective_names)) != len(objective_names):
        raise ValueError(f"objective_labels must be unique; got {objective_names!r}.")
    if len(set(profile_names)) != len(profile_names):
        raise ValueError(f"profile_parameter_labels must be unique; got {profile_names!r}.")
    if len(set(geometry_names)) != len(geometry_names):
        raise ValueError(f"geometry_parameter_labels must be unique; got {geometry_names!r}.")
    objective_values_np = np.asarray(jax.device_get(objective_values), dtype=float)
    profile_gradient_np = np.asarray(jax.device_get(profile_gradient_matrix), dtype=float)
    geometry_gradient_np = np.asarray(jax.device_get(geometry_gradient_matrix), dtype=float)

    objective_count = len(objective_names)
    if objective_values_np.shape != (objective_count,):
        raise ValueError(
            "objective_values must have shape (objective_count,); "
            f"got {objective_values_np.shape}, objective_count={objective_count}."
        )
    if profile_gradient_np.shape != (objective_count, len(profile_names)):
        raise ValueError(
            "profile_gradient_matrix must have shape "
            "(objective_count, profile_parameter_count); "
            f"got {profile_gradient_np.shape}, expected {(objective_count, len(profile_names))}."
        )
    if geometry_gradient_np.shape != (objective_count, len(geometry_names)):
        raise ValueError(
            "geometry_gradient_matrix must have shape "
            "(objective_count, geometry_parameter_count); "
            f"got {geometry_gradient_np.shape}, expected {(objective_count, len(geometry_names))}."
        )

    objective_finite_np = np.isfinite(objective_values_np)
    return {
        "objective_values": {
            name: float(value)
            for name, value in zip(objective_names, objective_values_np.tolist())
        },
        "objective_finite": {
            name: bool(value)
            for name, value in zip(objective_names, objective_finite_np.tolist())
        },
        "profile_gradient_all_finite_by_objective": {
            objective_name: bool(np.all(np.isfinite(profile_gradient_np[objective_i])))
            for objective_i, objective_name in enumerate(objective_names)
        },
        "geometry_gradient_all_finite_by_objective": {
            objective_name: bool(np.all(np.isfinite(geometry_gradient_np[objective_i])))
            for objective_i, objective_name in enumerate(objective_names)
        },
        "profile_gradient_reverse_ad": {
            objective_name: {
                parameter_name: float(value)
                for parameter_name, value in zip(
                    profile_names,
                    profile_gradient_np[objective_i].tolist(),
                )
            }
            for objective_i, objective_name in enumerate(objective_names)
        },
        "geometry_gradient_reverse_ad": {
            objective_name: {
                geometry_label: float(value)
                for geometry_label, value in zip(
                    geometry_names,
                    geometry_gradient_np[objective_i].tolist(),
                )
            }
            for objective_i, objective_name in enumerate(objective_names)
        },
    }


def realtime_geometry_transport_reverse_metadata_entries(
    *,
    parameter_mode: str,
    config_path: str,
    objective_labels: Sequence[str],
    profile_parameter_labels: Sequence[str],
    profile_values,
    geometry_parameter_labels: Sequence[str],
    geometry_parameter_entries: Sequence[Mapping[str, object]],
    baseline_geometry_deltas,
    geometry_parameter_specs: Sequence[tuple[str, int, int]],
    geometry_parameter_selector: str,
    accepted_step_limit: int | None,
    reverse_segment_length: int | None,
    reverse_stage_cotangent_mode_requested: str,
    reverse_stage_cotangent_mode_effective: str,
    ntx_exact_derivative_mode: str,
    ntx_exact_derivative_field_pullback_mode: str,
    ntx_exact_surface_backend: str,
    realtime_geometry_gradient_path: str,
    realtime_geometry_component_pullbacks: bool,
    realtime_geometry_support_bar_diagnostics_skipped: bool,
    realtime_geometry_derivative_complete: bool,
    geometry_support_pullback_mode: str,
    realtime_geometry_diagnostics: Mapping[str, object],
    support_payload_summary: Mapping[str, object],
    support_bar_summary_by_objective: Mapping[str, object],
    support_bar_l2_by_objective: Mapping[str, object],
    support_bar_branch_diagnostics_by_objective: Mapping[str, object],
    support_reuse_count: int,
    support_rebuild_count: int,
    support_initial_cache_pullback_used: bool,
    support_initial_cache_pullback_skipped: bool,
    elapsed_s: float,
) -> dict[str, object]:
    """Assemble generic realtime-geometry transport reverse report metadata."""

    objective_names = tuple(str(name) for name in objective_labels)
    profile_names = tuple(str(name) for name in profile_parameter_labels)
    geometry_names = tuple(str(name) for name in geometry_parameter_labels)
    geometry_deltas = np.asarray(jax.device_get(baseline_geometry_deltas), dtype=float).tolist()
    if len(geometry_names) != len(geometry_parameter_entries) or len(geometry_names) != len(geometry_deltas):
        raise ValueError(
            "Geometry parameter labels, entries, and baseline deltas must have matching lengths; "
            f"got labels={len(geometry_names)}, entries={len(geometry_parameter_entries)}, "
            f"deltas={len(geometry_deltas)}."
        )
    return {
        "mode": "transport_reverse_ad_only",
        "parameter_mode": str(parameter_mode),
        "config_path": str(config_path),
        "objective_name": "all",
        "objective_order": list(objective_names),
        "parameter_order": list(profile_names),
        "profile_baseline_values": np.asarray(jax.device_get(profile_values), dtype=float).tolist(),
        "geometry_baseline_values": {
            geometry_label: float(entry["baseline_coefficient"]) + float(delta)
            for geometry_label, entry, delta in zip(
                geometry_names,
                geometry_parameter_entries,
                geometry_deltas,
            )
        },
        "geometry_parameter_order": list(geometry_names),
        "geometry_parameter_specs": [
            {"family": family, "m": int(m), "n": int(n)}
            for family, m, n in geometry_parameter_specs
        ],
        "geometry_parameter_selector": str(geometry_parameter_selector),
        "geometry_parameter_count": int(len(geometry_parameter_specs)),
        "accepted_step_limit": None if accepted_step_limit is None else int(accepted_step_limit),
        "reverse_segment_length": None if reverse_segment_length is None else int(reverse_segment_length),
        "reverse_stage_cotangent_mode_requested": str(reverse_stage_cotangent_mode_requested),
        "reverse_stage_cotangent_mode_effective": str(reverse_stage_cotangent_mode_effective),
        "ntx_exact_derivative_mode": str(ntx_exact_derivative_mode),
        "ntx_exact_derivative_field_pullback_mode": str(ntx_exact_derivative_field_pullback_mode),
        "ntx_exact_surface_backend": str(ntx_exact_surface_backend),
        "realtime_geometry_gradient_path": str(realtime_geometry_gradient_path),
        "realtime_geometry_component_pullbacks": bool(realtime_geometry_component_pullbacks),
        "realtime_geometry_support_bar_diagnostics_skipped": bool(
            realtime_geometry_support_bar_diagnostics_skipped
        ),
        "realtime_primal_runtime_builder": "build_runtime_context",
        "realtime_geometry_derivative_boundary": (
            "runtime_geometry_and_ntx_exact_lij_support_payload"
            if realtime_geometry_derivative_complete
            else "ntx_exact_lij_support_payload_only_diagnostic"
        ),
        "realtime_geometry_derivative_complete": bool(realtime_geometry_derivative_complete),
        "geometry_support_pullback_mode": str(geometry_support_pullback_mode),
        "realtime_geometry_support_bwd_mode": "grouped_batched_fused_support",
        "realtime_geometry_diagnostics": realtime_geometry_diagnostics,
        "support_payload_summary": support_payload_summary,
        "support_bar_summary_by_objective": support_bar_summary_by_objective,
        "support_bar_l2_by_objective": support_bar_l2_by_objective,
        "support_bar_branch_diagnostics_by_objective": support_bar_branch_diagnostics_by_objective,
        "support_reuse_count": int(support_reuse_count),
        "support_rebuild_count": int(support_rebuild_count),
        "support_initial_cache_pullback_used": bool(support_initial_cache_pullback_used),
        "support_initial_cache_pullback_skipped": bool(support_initial_cache_pullback_skipped),
        "elapsed_s": float(elapsed_s),
    }


def realtime_geometry_transport_reverse_diagnostic_gradient_entries(
    *,
    objective_labels: Sequence[str],
    geometry_parameter_labels: Sequence[str],
    geometry_gradient_matrix_np,
    geometry_branch_gradient_matrix_np=None,
    ntx_support_branch_gradient_matrix_np=None,
    component_gradient_np_by_name: Mapping[str, object] | None = None,
    component_geometry_branch_np_by_name: Mapping[str, object] | None = None,
    component_ntx_support_branch_np_by_name: Mapping[str, object] | None = None,
    include_component_pullbacks: bool = False,
) -> dict[str, object]:
    """Assemble branch/component geometry-gradient diagnostics from host arrays."""

    objective_names = tuple(str(name) for name in objective_labels)
    geometry_names = tuple(str(name) for name in geometry_parameter_labels)
    component_gradient_np_by_name = (
        {} if component_gradient_np_by_name is None else component_gradient_np_by_name
    )
    component_geometry_branch_np_by_name = (
        {} if component_geometry_branch_np_by_name is None else component_geometry_branch_np_by_name
    )
    component_ntx_support_branch_np_by_name = (
        {} if component_ntx_support_branch_np_by_name is None else component_ntx_support_branch_np_by_name
    )
    if geometry_branch_gradient_matrix_np is None or ntx_support_branch_gradient_matrix_np is None:
        branch_entries = None
    else:
        branch_entries = {
            objective_name: {
                "geometry": {
                    geometry_label: float(value)
                    for geometry_label, value in zip(
                        geometry_names,
                        geometry_branch_gradient_matrix_np[objective_i].tolist(),
                    )
                },
                "ntx_support": {
                    geometry_label: float(value)
                    for geometry_label, value in zip(
                        geometry_names,
                        ntx_support_branch_gradient_matrix_np[objective_i].tolist(),
                    )
                },
                "combined": {
                    geometry_label: float(value)
                    for geometry_label, value in zip(
                        geometry_names,
                        geometry_gradient_matrix_np[objective_i].tolist(),
                    )
                },
            }
            for objective_i, objective_name in enumerate(objective_names)
        }
    if not include_component_pullbacks:
        return {
            "geometry_gradient_reverse_ad_by_branch": branch_entries,
            "geometry_gradient_reverse_ad_by_component": {},
            "geometry_gradient_reverse_ad_final_state_components": {},
            "geometry_gradient_reverse_ad_by_component_and_branch": {},
        }
    component_entries = {
        objective_name: {
            component_name: {
                geometry_label: float(value)
                for geometry_label, value in zip(
                    geometry_names,
                    component_matrix[objective_i].tolist(),
                )
            }
            for component_name, component_matrix in component_gradient_np_by_name.items()
        }
        for objective_i, objective_name in enumerate(objective_names)
    }
    final_state_component_entries = {
        objective_name: {
            geometry_label: float(
                sum(
                    component_matrix[objective_i, geometry_i]
                    for component_name, component_matrix in component_gradient_np_by_name.items()
                    if component_name != "objective_explicit"
                )
            )
            for geometry_i, geometry_label in enumerate(geometry_names)
        }
        for objective_i, objective_name in enumerate(objective_names)
    }
    component_branch_entries = {
        objective_name: {
            component_name: {
                "geometry": {
                    geometry_label: float(value)
                    for geometry_label, value in zip(
                        geometry_names,
                        component_geometry_branch_np_by_name.get(component_name, component_matrix)[
                            objective_i
                        ].tolist(),
                    )
                },
                "ntx_support": {
                    geometry_label: float(value)
                    for geometry_label, value in zip(
                        geometry_names,
                        component_ntx_support_branch_np_by_name.get(component_name, component_matrix)[
                            objective_i
                        ].tolist(),
                    )
                },
                "combined": {
                    geometry_label: float(value)
                    for geometry_label, value in zip(
                        geometry_names,
                        component_matrix[objective_i].tolist(),
                    )
                },
            }
            for component_name, component_matrix in component_gradient_np_by_name.items()
        }
        for objective_i, objective_name in enumerate(objective_names)
    }
    return {
        "geometry_gradient_reverse_ad_by_branch": branch_entries,
        "geometry_gradient_reverse_ad_by_component": component_entries,
        "geometry_gradient_reverse_ad_final_state_components": final_state_component_entries,
        "geometry_gradient_reverse_ad_by_component_and_branch": component_branch_entries,
    }


def realtime_geometry_transport_reverse_grouped_runner(
    *,
    args,
    support_segment_executor: TransportReverseSupportSegmentExecutor,
    update_args_for_all_objectives: TransportReverseArgsUpdater,
) -> TransportReverseReportRunner:
    """Build the grouped all-objective runner used by transport reverse builders.

    The supplied executor still owns the benchmark-validated segmented reverse
    math. This wrapper owns the grouped contract: force objective='all', request
    an internal report return, and require a JAX-native table result.
    """

    if not callable(support_segment_executor):
        raise TypeError("support_segment_executor must be callable.")
    if not callable(update_args_for_all_objectives):
        raise TypeError("update_args_for_all_objectives must be callable.")

    def _run_grouped_report() -> TransportReverseReport:
        grouped_args = update_args_for_all_objectives(args)
        report = support_segment_executor(grouped_args, True)
        table_result = report.get("transport_reverse_table_result") if isinstance(report, Mapping) else None
        if not isinstance(table_result, RealtimeGeometryTransportReverseTableResult):
            raise TypeError(
                "Grouped realtime-geometry reverse executor did not return a "
                "RealtimeGeometryTransportReverseTableResult under "
                "'transport_reverse_table_result'."
            )
        return report

    return _run_grouped_report


def grouped_transport_reverse_report_builder(
    *,
    objective_labels: Sequence[str],
    run_grouped_report: TransportReverseReportRunner,
    table_context: RealtimeGeometryTransportReverseTableContext | None = None,
    quiet_default: bool = True,
) -> TransportReverseReportBuilder:
    """Build a grouped transport report builder around a validated runner.

    `run_grouped_report` should execute the validated all-objective reverse
    path and return its report dictionary. Objective selection is handled later
    by the table adapter; this builder only validates that all requested names
    exist and controls whether benchmark-style progress output is suppressed.
    """

    labels = tuple(str(name) for name in objective_labels)

    def _builder(
        objective_names: tuple[str, ...],
        parameter_set: ReverseADParameterSet,
        options: Mapping[str, object] | None,
    ) -> TransportReverseReport:
        if table_context is not None:
            request = realtime_geometry_transport_reverse_table_request(
                objective_names=objective_names,
                parameter_set=parameter_set,
                context=table_context,
                options=options,
            )
            objective_names = request.objective_names
            parameter_set = request.parameter_set
            options = request.options
        return run_realtime_geometry_transport_reverse_table(
            objective_names=objective_names,
            parameter_set=parameter_set,
            objective_labels=labels,
            run_grouped_report=run_grouped_report,
            options=options,
            quiet_default=quiet_default,
        )

    return _builder


def grouped_transport_reverse_table_result_builder(
    *,
    objective_labels: Sequence[str],
    run_grouped_report: TransportReverseReportRunner,
    table_context: RealtimeGeometryTransportReverseTableContext | None = None,
    quiet_default: bool = True,
) -> TransportReverseTableResultBuilder:
    """Build a grouped transport table-result builder around a validated runner."""

    labels = tuple(str(name) for name in objective_labels)

    def _builder(
        objective_names: tuple[str, ...],
        parameter_set: ReverseADParameterSet,
        options: Mapping[str, object] | None,
    ) -> RealtimeGeometryTransportReverseTableResult:
        if table_context is not None:
            return transport_realtime_geometry_reverse_table(
                objective_names=objective_names,
                parameter_set=parameter_set,
                context=table_context,
                run_grouped_report=run_grouped_report,
                objective_labels=labels,
                options=options,
                quiet_default=quiet_default,
            )
        return build_realtime_geometry_transport_reverse_table_result(
            objective_names=objective_names,
            parameter_set=parameter_set,
            objective_labels=labels,
            run_grouped_report=run_grouped_report,
            options=options,
            quiet_default=quiet_default,
        )

    return _builder
