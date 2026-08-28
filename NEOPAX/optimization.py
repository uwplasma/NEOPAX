"""Optimization helpers built on NEOPAX reverse-AD internals."""

from __future__ import annotations

import dataclasses
import copy
import gc
import os
import time
from collections.abc import Sequence
from typing import Callable, Mapping
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from ._geometry_autodiff import (
    _boozer_surface_indices_and_rho,
    _neopax_geometry_requested_sample_rho,
    _input_with_boundary_deltas,
    boundary_param_entries,
    build_neopax_geometry_and_ntx_exact_lij_support_from_state,
    build_geometry_autodiff_context,
    geometry_raw_block_stage,
    geometry_raw_block_solve_from_param_vector,
)
from ._constants import elementary_charge
from ._orchestrator import build_runtime_context
from ._orchestrator import run_config
from ._reverse_ad_initial_er import (
    find_ntx_support_payload,
    initial_er_selected_root_profile,
    runtime_with_geometry_payload,
    runtime_with_ntx_support_payload,
)
from ._reverse_ad_optimization import (
    LeastSquaresEvaluation,
    LeastSquaresResult,
    LeastSquaresTerm,
    ObjectiveTableResult,
    ObjectiveRef,
    evaluate_geometry_transport_realtime_geometry_least_squares,
    evaluate_transport_realtime_geometry_least_squares,
    evaluate_geometry_initial_er_root_only_least_squares_benchmark_tables,
    evaluate_geometry_initial_er_root_only_least_squares_optimization,
    _optimization_root_to_payload_cotangents,
    _optimization_payload_to_vmec_table,
    geometry,
    normalize_least_squares_terms,
    normalize_transport_objective_names,
    realtime_geometry_transport_reverse_table_request,
    geometry_full_ad_reverse_table,
    scale_least_squares_evaluation_columns,
    transport,
)
from ._optimization_initial_root_stage import (
    build_geometry_payload_optimization_stage,
    build_compiled_geometry_initial_root_stage,
    build_geometry_initial_root_optimization_stage,
    initial_root_stage_layout,
)
from ._reverse_ad_parameters import (
    PROFILE_PARAMETER_ORDER,
    ProfileParameterSpec,
    VmexBoundaryParameterization,
    parse_profile_parameter_specs,
    parse_vmec_boundary_parameter_specs,
    reverse_ad_optimization_parameter_set,
    vmex_boundary_parameterization,
)
from ._reverse_ad_transport import (
    TRANSPORT_REVERSE_OBJECTIVE_LABELS,
    internal_realtime_geometry_transport_reverse_table_result_builder,
    initial_state_for_parameter_vector,
    realtime_geometry_transport_reverse_grouped_inputs,
    realtime_geometry_transport_reverse_support_segment_executor,
    realtime_geometry_transport_reverse_table_context,
    run_internal_realtime_geometry_support_segment_probe,
)
from ._transport_flux_models import DENSITY_STATE_TO_PHYSICAL
from .api import prepare_config


@dataclasses.dataclass(frozen=True, slots=True)
class GeometryObjectiveTransform:
    """Script-defined scalar objective backed by one AD-table row."""

    base: ObjectiveRef
    value_fn: Callable[[object], object]
    derivative_fn: Callable[[object], object]
    label: str

    @property
    def objective(self) -> ObjectiveRef:
        return self.base


@dataclasses.dataclass(frozen=True, slots=True)
class GeometryLeastSquaresTerm:
    """Geometry least-squares term with an optional scalar chain-rule transform."""

    objective: ObjectiveRef
    target: float
    weight: float
    label: str | None = None
    value_fn: Callable[[object], object] | None = None
    derivative_fn: Callable[[object], object] | None = None

    @property
    def residual_label(self) -> str:
        return self.label or self.objective.label


@dataclasses.dataclass(frozen=True, slots=True)
class RepeatedEvaluationMemorySample:
    """One fully materialized optimizer evaluation in a memory audit."""

    iteration: int
    elapsed_s: float
    resident_memory_bytes: int | None
    residual_norm: float
    jacobian_shape: tuple[int, int]


def _process_resident_memory_bytes() -> int | None:
    """Return the current process working set without requiring ``psutil``.

    This deliberately measures the whole process: JAX/XLA executable and device
    allocations are native allocations, so Python allocation tracers alone are
    not useful for the optimizer-retention investigation.
    """

    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class _ProcessMemoryCountersEx(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t),
                ]

            counters = _ProcessMemoryCountersEx()
            counters.cb = ctypes.sizeof(counters)
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(),
                ctypes.byref(counters),
                counters.cb,
            )
            return int(counters.WorkingSetSize) if ok else None
        except (AttributeError, OSError):
            return None
    try:
        with open("/proc/self/statm", encoding="ascii") as stream:
            resident_pages = int(stream.read().split()[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return None


def repeated_evaluation_memory_samples(
    problem,
    *,
    repeats: int = 5,
    warmup: int = 1,
    scaled_parameter_values=None,
    on_sample: Callable[[RepeatedEvaluationMemorySample], None] | None = None,
) -> tuple[RepeatedEvaluationMemorySample, ...]:
    """Measure retained process memory across identical optimizer evaluations.

    ``warmup`` evaluations are excluded so the caller can distinguish first-use
    compilation from growth that continues across identical geometry plus
    initial-Er-root trials. This is diagnostic-only: it calls the existing
    ``problem.evaluate`` unchanged and retains neither its arrays nor VJP data.
    """

    if int(repeats) != repeats or int(repeats) < 1:
        raise ValueError("repeats must be a positive integer.")
    if int(warmup) != warmup or int(warmup) < 0:
        raise ValueError("warmup must be a non-negative integer.")
    x = problem.x0 if scaled_parameter_values is None else scaled_parameter_values
    for _ in range(int(warmup)):
        evaluation = problem.evaluate(x)
        jax.block_until_ready((evaluation.residuals, evaluation.jacobian))
        del evaluation
    gc.collect()

    samples = []
    for iteration in range(int(repeats)):
        started = time.perf_counter()
        evaluation = problem.evaluate(x)
        residuals, jacobian = jax.block_until_ready(
            (evaluation.residuals, evaluation.jacobian)
        )
        residual_norm = float(np.linalg.norm(np.asarray(jax.device_get(residuals))))
        jacobian_shape = tuple(int(size) for size in jacobian.shape)
        elapsed_s = time.perf_counter() - started
        del evaluation, residuals, jacobian
        gc.collect()
        sample = RepeatedEvaluationMemorySample(
            iteration=iteration,
            elapsed_s=float(elapsed_s),
            resident_memory_bytes=_process_resident_memory_bytes(),
            residual_norm=residual_norm,
            jacobian_shape=jacobian_shape,
        )
        samples.append(sample)
        if on_sample is not None:
            on_sample(sample)
    return tuple(samples)


@dataclasses.dataclass(frozen=True, slots=True)
class GeometryLeastSquaresProblem:
    """Geometry-only least-squares problem using the validated geometry AD table."""

    context: object
    parameterization: VmexBoundaryParameterization
    terms: tuple[GeometryLeastSquaresTerm, ...]
    lane: str = "ad"
    max_iter: int | None = None
    step_size: float | None = None
    solver_device: str | None = "default"

    @property
    def parameter_count(self) -> int:
        return len(self.parameterization.specs)

    @property
    def parameter_labels(self) -> tuple[str, ...]:
        return self.parameterization.labels

    @property
    def x0(self):
        return jnp.zeros((self.parameter_count,), dtype=jnp.float64)

    @property
    def x_scale(self):
        return self.parameterization.x_scale

    def evaluate(self, scaled_parameter_values=None) -> LeastSquaresEvaluation:
        """Evaluate residuals/Jacobian with respect to scaled optimizer variables."""

        if scaled_parameter_values is None:
            scaled_values = self.x0
        else:
            scaled_values = jnp.asarray(scaled_parameter_values, dtype=jnp.float64)
        if tuple(scaled_values.shape) != (self.parameter_count,):
            raise ValueError(
                "scaled_parameter_values must have shape "
                f"({self.parameter_count},); got {tuple(scaled_values.shape)}."
            )
        physical_values = self.parameterization.scaled_to_physical_delta(scaled_values)
        parameter_set = reverse_ad_optimization_parameter_set(
            include_profiles=False,
            vmec_boundary=self.parameterization.specs,
        )
        objective_names = tuple(dict.fromkeys(term.objective.name for term in self.terms))
        t_start = time.perf_counter()
        table = geometry_full_ad_reverse_table(
            context=self.context,
            parameter_set=parameter_set,
            objective_names=objective_names,
            parameter_values=physical_values,
            lane=self.lane,
            max_iter=self.max_iter,
            step_size=self.step_size,
            final_vmec_pullback_mode="raw_block_transpose",
            solver_device=self.solver_device,
        )
        result = _assemble_geometry_least_squares_result(
            self.terms,
            parameter_set=parameter_set,
            table=table,
        )
        residuals = jax.block_until_ready(result.residuals)
        jacobian = jax.block_until_ready(result.jacobian)
        evaluation = LeastSquaresEvaluation(
            result=result,
            residuals=residuals,
            jacobian=jacobian,
            elapsed_s=float(time.perf_counter() - t_start),
        )
        return scale_least_squares_evaluation_columns(evaluation, self.x_scale)

    def residuals(self, scaled_parameter_values=None) -> np.ndarray:
        return np.asarray(jax.device_get(self.evaluate(scaled_parameter_values).residuals), dtype=float)

    def jacobian(self, scaled_parameter_values=None) -> np.ndarray:
        return np.asarray(jax.device_get(self.evaluate(scaled_parameter_values).jacobian), dtype=float)

    def input_from_scaled_parameters(self, scaled_parameter_values=None):
        """Return a VMEX input object with the current scaled boundary deltas applied."""

        if scaled_parameter_values is None:
            scaled_values = self.x0
        else:
            scaled_values = jnp.asarray(scaled_parameter_values, dtype=jnp.float64)
        physical_values = self.parameterization.scaled_to_physical_delta(scaled_values)
        entries = boundary_param_entries(self.context, self.parameterization.vmec_tuples)
        return _input_with_boundary_deltas(self.context, physical_values, entries)


@dataclasses.dataclass(frozen=True, slots=True)
class GeometryInitialErRootLeastSquaresProblem:
    """Geometry objectives plus initial-Er root-only transport objectives.

    This optimization-facing wrapper intentionally composes the benchmark-
    validated internal reverse tables. It does not route through the experimental
    fused payload path that caused the mixed-script OOMs.
    """

    config: dict
    context: object
    runtime: object
    baseline_state: object
    baseline_profile_values: object
    profile_scales: object
    parameter_set: object
    geometry_parameterization: VmexBoundaryParameterization | None
    profile_specs: tuple[ProfileParameterSpec, ...]
    terms: tuple[GeometryLeastSquaresTerm | LeastSquaresTerm, ...]
    n_r: int
    n_theta: int
    n_zeta: int
    n_xi: int
    surface_backend: str = "vmec"
    geometry_lane: str = "ad"
    geometry_max_iter: int | None = None
    geometry_step_size: float | None = None
    geometry_solver_device: str | None = "default"
    root_options: Mapping[str, object] | None = None
    raw_block_stage: object | None = None
    optimization_stage_layout: object | None = None
    optimization_stage: object | None = None
    payload_optimization_stage: object | None = None
    reverse_stage_mode: str = "off"

    @property
    def parameter_count(self) -> int:
        return len(self.parameter_set.specs)

    @property
    def parameter_labels(self) -> tuple[str, ...]:
        return self.parameter_set.labels

    @property
    def x0(self):
        values = []
        profile_lookup = {name: i for i, name in enumerate(PROFILE_PARAMETER_ORDER)}
        profile_scale_lookup = {
            name: self.profile_scales[i]
            for i, name in enumerate(PROFILE_PARAMETER_ORDER)
        }
        for spec in self.parameter_set.specs:
            if isinstance(spec, ProfileParameterSpec):
                values.append(self.baseline_profile_values[profile_lookup[spec.name]] / profile_scale_lookup[spec.name])
            else:
                values.append(jnp.asarray(0.0, dtype=jnp.float64))
        return jnp.asarray(values, dtype=jnp.float64)

    @property
    def x_scale(self):
        geometry_scales = {}
        if self.geometry_parameterization is not None:
            geometry_scales = {
                spec: scale
                for spec, scale in zip(
                    self.geometry_parameterization.specs,
                    self.geometry_parameterization.scales,
                    strict=True,
                )
            }
        scales = []
        profile_scale_lookup = {
            name: self.profile_scales[i]
            for i, name in enumerate(PROFILE_PARAMETER_ORDER)
        }
        for spec in self.parameter_set.specs:
            if isinstance(spec, ProfileParameterSpec):
                scales.append(profile_scale_lookup[spec.name])
            else:
                scales.append(float(geometry_scales.get(spec, 1.0)))
        return jnp.asarray(scales, dtype=jnp.float64)

    def _scaled_to_physical(self, scaled_parameter_values):
        scaled_values = jnp.asarray(scaled_parameter_values, dtype=jnp.float64)
        if tuple(scaled_values.shape) != (self.parameter_count,):
            raise ValueError(
                "scaled_parameter_values must have shape "
                f"({self.parameter_count},); got {tuple(scaled_values.shape)}."
            )
        physical_values = []
        scale = self.x_scale
        for i, spec in enumerate(self.parameter_set.specs):
            if isinstance(spec, ProfileParameterSpec):
                physical_values.append(scaled_values[i] * scale[i])
            else:
                physical_values.append(scaled_values[i] * scale[i])
        return jnp.asarray(physical_values, dtype=jnp.float64)

    def _pre_root_state_from_profile_values(self, profile_values):
        return initial_state_for_parameter_vector(
            profile_values,
            config=self.config,
            initial_er_root_ad="off",
            baseline_state=self.baseline_state,
            profile_cfg=self.config.get("profiles", {}),
            runtime=self.runtime,
        )

    def evaluate(self, scaled_parameter_values=None) -> LeastSquaresEvaluation:
        """Evaluate residuals/Jacobian with respect to scaled optimizer variables."""

        if scaled_parameter_values is None:
            scaled_values = self.x0
        else:
            scaled_values = jnp.asarray(scaled_parameter_values, dtype=jnp.float64)
        physical_values = self._scaled_to_physical(scaled_values)
        base_terms = _base_terms_for_mixed_initial_er_root(self.terms)
        base_evaluation = evaluate_geometry_initial_er_root_only_least_squares_benchmark_tables(
            self.config,
            parameter_set=self.parameter_set,
            parameter_values=physical_values,
            terms=base_terms,
            geometry_context=self.context,
            runtime=self.runtime,
            baseline_profile_values=self.baseline_profile_values,
            pre_root_state_from_profile_values=self._pre_root_state_from_profile_values,
            n_r=self.n_r,
            n_theta=self.n_theta,
            n_zeta=self.n_zeta,
            n_xi=self.n_xi,
            surface_backend=self.surface_backend,
            geometry_lane=self.geometry_lane,
            geometry_max_iter=self.geometry_max_iter,
            geometry_step_size=self.geometry_step_size,
            geometry_solver_device=self.geometry_solver_device,
            root_options=self.root_options,
            raw_block_stage=self.raw_block_stage,
            payload_optimization_stage=self.payload_optimization_stage,
        )
        result = _assemble_mixed_initial_er_root_result(
            self.terms,
            base_evaluation=base_evaluation,
        )
        residuals = jax.block_until_ready(result.residuals)
        jacobian = jax.block_until_ready(result.jacobian)
        evaluation = LeastSquaresEvaluation(
            result=result,
            residuals=residuals,
            jacobian=jacobian,
            elapsed_s=float(base_evaluation.elapsed_s),
        )
        return scale_least_squares_evaluation_columns(evaluation, self.x_scale)

    def residuals(self, scaled_parameter_values=None) -> np.ndarray:
        return np.asarray(jax.device_get(self.evaluate(scaled_parameter_values).residuals), dtype=float)

    def jacobian(self, scaled_parameter_values=None) -> np.ndarray:
        return np.asarray(jax.device_get(self.evaluate(scaled_parameter_values).jacobian), dtype=float)

    def input_from_scaled_parameters(self, scaled_parameter_values=None):
        """Return a VMEX input object with the current scaled boundary deltas applied."""

        if self.geometry_parameterization is None:
            raise ValueError("This problem has no geometry parameterization.")
        scaled_values = self.x0 if scaled_parameter_values is None else jnp.asarray(
            scaled_parameter_values,
            dtype=jnp.float64,
        )
        physical_values = self._scaled_to_physical(scaled_values)
        geometry_values = jnp.asarray(
            [
                physical_values[i]
                for i, spec in enumerate(self.parameter_set.specs)
                if not isinstance(spec, ProfileParameterSpec)
            ],
            dtype=jnp.float64,
        )
        entries = boundary_param_entries(self.context, self.geometry_parameterization.vmec_tuples)
        return _input_with_boundary_deltas(self.context, geometry_values, entries)

    def _initial_er_root_state_runtime_from_scaled_parameters(self, scaled_parameter_values=None):
        """Return rho, rooted initial state, runtime, and finite mask for optimizer variables."""

        scaled_values = self.x0 if scaled_parameter_values is None else jnp.asarray(
            scaled_parameter_values,
            dtype=jnp.float64,
        )
        physical_values = self._scaled_to_physical(scaled_values)
        profile_values = list(jnp.asarray(self.baseline_profile_values, dtype=jnp.float64))
        geometry_values = []
        profile_lookup = {name: i for i, name in enumerate(PROFILE_PARAMETER_ORDER)}
        for i, spec in enumerate(self.parameter_set.specs):
            if isinstance(spec, ProfileParameterSpec):
                profile_values[profile_lookup[spec.name]] = physical_values[i]
            else:
                geometry_values.append(physical_values[i])
        profile_values = jnp.asarray(profile_values, dtype=jnp.float64)
        runtime_for_geometry = self.runtime
        if geometry_values:
            geometry_values = jnp.asarray(geometry_values, dtype=jnp.float64)
            if not bool(np.allclose(np.asarray(jax.device_get(geometry_values), dtype=float), 0.0)):
                raw_block_solve = geometry_raw_block_solve_from_param_vector(
                    self.context,
                    geometry_values,
                    tuple(spec.as_tuple() for spec in self.parameter_set.vmec_boundary_specs),
                    max_iter=self.geometry_max_iter,
                    solver_device=self.geometry_solver_device,
                )
                payload = build_neopax_geometry_and_ntx_exact_lij_support_from_state(
                    self.context,
                    raw_block_solve.state,
                    n_r=self.n_r,
                    n_theta=self.n_theta,
                    n_zeta=self.n_zeta,
                    n_xi=self.n_xi,
                    surface_backend=self.surface_backend,
                )
                runtime_for_geometry = runtime_with_geometry_payload(runtime_for_geometry, payload["geometry"])
                runtime_for_geometry = runtime_with_ntx_support_payload(runtime_for_geometry, payload["ntx_support"])
        pre_root_state = self._pre_root_state_from_profile_values(profile_values)
        er_profile, finite_mask = initial_er_selected_root_profile(
            pre_root_state,
            config=dict(self.config),
            runtime=runtime_for_geometry,
        )
        rho_grid = jnp.asarray(runtime_for_geometry.geometry.rho_grid, dtype=jnp.asarray(er_profile).dtype)
        rooted_state = dataclasses.replace(pre_root_state, Er=er_profile)
        return rho_grid, rooted_state, runtime_for_geometry, finite_mask

    def initial_er_profile_from_scaled_parameters(self, scaled_parameter_values=None):
        """Return rho, selected initial ambipolar Er, and finite mask for optimizer variables."""

        rho_grid, rooted_state, _runtime_for_geometry, finite_mask = (
            self._initial_er_root_state_runtime_from_scaled_parameters(scaled_parameter_values)
        )
        return rho_grid, rooted_state.Er, finite_mask

    def initial_root_profiles_from_scaled_parameters(self, scaled_parameter_values=None):
        """Return rho, density, temperature, selected Er, and finite root mask."""

        rho_grid, rooted_state, _runtime_for_geometry, finite_mask = (
            self._initial_er_root_state_runtime_from_scaled_parameters(scaled_parameter_values)
        )
        return rho_grid, rooted_state.density, rooted_state.temperature, rooted_state.Er, finite_mask

    def bootstrap_current_profile_from_scaled_parameters(self, scaled_parameter_values=None):
        """Return rho, momentum-corrected bootstrap current profile, and finite mask.

        The returned current uses the same scaled units as the optimization
        objective: one unit corresponds to ``1e5 A/m^2``.
        """

        rho_grid, rooted_state, runtime_for_geometry, finite_mask = (
            self._initial_er_root_state_runtime_from_scaled_parameters(scaled_parameter_values)
        )
        flux_model = runtime_for_geometry.models.flux
        neoclassical_model = getattr(flux_model, "neoclassical_model", flux_model)
        corrected_fluxes_fn = getattr(neoclassical_model, "evaluate_momentum_corrected_fluxes", None)
        if corrected_fluxes_fn is None:
            raise ValueError("Bootstrap-current profile requires evaluate_momentum_corrected_fluxes.")
        fluxes = corrected_fluxes_fn(rooted_state)

        def flux_value(name: str):
            if hasattr(fluxes, "get"):
                return fluxes.get(name, None)
            return getattr(fluxes, name, None)

        upar = flux_value("Upar_neo")
        if upar is None:
            upar = flux_value("Upar")
        if upar is None:
            raise ValueError("Momentum-corrected fluxes did not provide Upar_neo or Upar.")
        upar_arr = jnp.asarray(upar, dtype=jnp.asarray(rooted_state.pressure).dtype)
        charge_qp = jnp.asarray(runtime_for_geometry.species.charge_qp, dtype=upar_arr.dtype)
        current_weights = jnp.sign(charge_qp)
        upar_physical = jnp.asarray(DENSITY_STATE_TO_PHYSICAL, dtype=upar_arr.dtype) * upar_arr
        scale = jnp.asarray(elementary_charge * 1.0e-5, dtype=upar_arr.dtype)
        if int(upar_arr.shape[0]) == int(charge_qp.shape[0]):
            jboot = jnp.sum(upar_physical * current_weights[:, None], axis=0) * scale
        else:
            jboot = jnp.sum(upar_physical * current_weights[None, :], axis=1) * scale
        return rho_grid, jboot, finite_mask


@dataclasses.dataclass(frozen=True, slots=True)
class ProfileFullTransportLeastSquaresProblem:
    """Profile-only full-transport least-squares problem.

    This is the full Radau-transport analogue of the profile-only initial-Er
    root helper. The benchmark-validated realtime transport reverse table owns
    the AD calculation; this wrapper only handles scaled profile variables and
    user-facing least-squares terms.
    """

    config: dict
    runtime: object
    baseline_state: object
    profile_cfg: Mapping[str, object]
    neoclassical_cfg: Mapping[str, object]
    baseline_profile_values: object
    profile_scales: object
    parameter_set: object
    terms: tuple[GeometryLeastSquaresTerm | LeastSquaresTerm, ...]
    table_context: object
    run_grouped_report: object
    options: Mapping[str, object]

    @property
    def parameter_count(self) -> int:
        return len(self.parameter_set.specs)

    @property
    def parameter_labels(self) -> tuple[str, ...]:
        return self.parameter_set.labels

    @property
    def x0(self):
        values = []
        profile_lookup = {name: i for i, name in enumerate(PROFILE_PARAMETER_ORDER)}
        for spec in self.parameter_set.profile_specs:
            base_value = self.baseline_profile_values[profile_lookup[spec.name]]
            scale = self.profile_scales[profile_lookup[spec.name]]
            values.append(base_value / scale)
        return jnp.asarray(values, dtype=jnp.float64)

    @property
    def x_scale(self):
        profile_scale_lookup = {
            name: self.profile_scales[i]
            for i, name in enumerate(PROFILE_PARAMETER_ORDER)
        }
        return jnp.asarray(
            [profile_scale_lookup[spec.name] for spec in self.parameter_set.profile_specs],
            dtype=jnp.float64,
        )

    def _scaled_to_physical(self, scaled_parameter_values):
        scaled_values = jnp.asarray(scaled_parameter_values, dtype=jnp.float64)
        if tuple(scaled_values.shape) != (self.parameter_count,):
            raise ValueError(
                "scaled_parameter_values must have shape "
                f"({self.parameter_count},); got {tuple(scaled_values.shape)}."
            )
        return scaled_values * self.x_scale

    def evaluate(self, scaled_parameter_values=None) -> LeastSquaresEvaluation:
        scaled_values = self.x0 if scaled_parameter_values is None else scaled_parameter_values
        physical_values = self._scaled_to_physical(scaled_values)
        base_terms = _base_terms_for_mixed_initial_er_root(self.terms)
        transport_objectives = tuple(
            dict.fromkeys(term.objective.name for term in base_terms if term.objective.family == "transport")
        )
        if not transport_objectives:
            raise ValueError("Profile full-transport optimization requires at least one transport objective.")
        options = dict(self.options)
        request = realtime_geometry_transport_reverse_table_request(
            objective_names=transport_objectives,
            parameter_set=self.parameter_set,
            context=self.table_context,
            options=options,
        )
        base_evaluation = evaluate_transport_realtime_geometry_least_squares(
            self.config,
            request=request,
            terms=base_terms,
            run_grouped_report=self.run_grouped_report,
            objective_labels=TRANSPORT_REVERSE_OBJECTIVE_LABELS,
            options=options | {"profile_values": physical_values},
            quiet_default=True,
        )
        result = _assemble_mixed_initial_er_root_result(
            self.terms,
            base_evaluation=base_evaluation,
        )
        residuals = jax.block_until_ready(result.residuals)
        jacobian = jax.block_until_ready(result.jacobian)
        evaluation = LeastSquaresEvaluation(
            result=result,
            residuals=residuals,
            jacobian=jacobian,
            elapsed_s=float(base_evaluation.elapsed_s),
        )
        return scale_least_squares_evaluation_columns(evaluation, self.x_scale)

    def residuals(self, scaled_parameter_values=None) -> np.ndarray:
        return np.asarray(jax.device_get(self.evaluate(scaled_parameter_values).residuals), dtype=float)

    def jacobian(self, scaled_parameter_values=None) -> np.ndarray:
        return np.asarray(jax.device_get(self.evaluate(scaled_parameter_values).jacobian), dtype=float)

    def config_from_scaled_parameters(self, scaled_parameter_values=None):
        physical_values = self._scaled_to_physical(self.x0 if scaled_parameter_values is None else scaled_parameter_values)
        config_eff = copy.deepcopy(self.config)
        profiles = config_eff.setdefault("profiles", {})
        for spec, value in zip(self.parameter_set.profile_specs, physical_values, strict=True):
            profiles[spec.name] = float(np.asarray(jax.device_get(value)))
        return config_eff

    def final_transport_profiles_from_scaled_parameters(self, scaled_parameter_values=None):
        config_eff = self.config_from_scaled_parameters(scaled_parameter_values)
        result = run_config(config_eff)
        final_state = result.get("final_state") if isinstance(result, dict) else getattr(result, "final_state", None)
        if final_state is None:
            ys = result.get("ys") if isinstance(result, dict) else getattr(result, "ys", None)
            if ys is not None:
                final_state = jax.tree_util.tree_map(lambda leaf: leaf[-1], ys)
        if final_state is None:
            raise RuntimeError("Full transport run did not expose a final state.")
        rho_grid = jnp.asarray(self.runtime.geometry.rho_grid, dtype=jnp.asarray(final_state.Er).dtype)
        return rho_grid, final_state


@dataclasses.dataclass(frozen=True, slots=True)
class GeometryFullTransportLeastSquaresProblem:
    """Geometry-only QI + full-transport least-squares problem."""

    config: dict
    context: object
    runtime: object
    baseline_state: object
    parameterization: VmexBoundaryParameterization
    parameter_set: object
    terms: tuple[GeometryLeastSquaresTerm | LeastSquaresTerm, ...]
    table_context: object
    table_result_builder: object
    options: Mapping[str, object]
    geometry_lane: str = "ad"
    geometry_max_iter: int | None = None
    geometry_step_size: float | None = None
    geometry_solver_device: str | None = "default"

    @property
    def parameter_count(self) -> int:
        return len(self.parameter_set.specs)

    @property
    def parameter_labels(self) -> tuple[str, ...]:
        return self.parameter_set.labels

    @property
    def x0(self):
        return jnp.zeros((self.parameter_count,), dtype=jnp.float64)

    @property
    def x_scale(self):
        return self.parameterization.x_scale

    def _scaled_to_physical(self, scaled_parameter_values):
        scaled_values = jnp.asarray(scaled_parameter_values, dtype=jnp.float64)
        if tuple(scaled_values.shape) != (self.parameter_count,):
            raise ValueError(
                "scaled_parameter_values must have shape "
                f"({self.parameter_count},); got {tuple(scaled_values.shape)}."
            )
        return self.parameterization.scaled_to_physical_delta(scaled_values)

    def evaluate(self, scaled_parameter_values=None) -> LeastSquaresEvaluation:
        physical_values = self._scaled_to_physical(
            self.x0 if scaled_parameter_values is None else scaled_parameter_values
        )
        base_terms = _base_terms_for_mixed_initial_er_root(self.terms)
        transport_objectives = tuple(
            dict.fromkeys(term.objective.name for term in base_terms if term.objective.family == "transport")
        )
        options = dict(self.options)
        request = realtime_geometry_transport_reverse_table_request(
            objective_names=transport_objectives,
            parameter_set=self.parameter_set,
            context=self.table_context,
            options=options,
        )
        base_evaluation = evaluate_geometry_transport_realtime_geometry_least_squares(
            self.config,
            request=request,
            terms=base_terms,
            geometry_context=self.context,
            parameter_values=physical_values,
            table_result_builder=self.table_result_builder,
            objective_labels=TRANSPORT_REVERSE_OBJECTIVE_LABELS,
            options=options,
            quiet_default=True,
            geometry_lane=self.geometry_lane,
            geometry_max_iter=self.geometry_max_iter,
            geometry_step_size=self.geometry_step_size,
            geometry_final_vmec_pullback_mode="raw_block_transpose",
            geometry_solver_device=self.geometry_solver_device,
            share_raw_block_solve=True,
        )
        result = _assemble_mixed_initial_er_root_result(
            self.terms,
            base_evaluation=base_evaluation,
        )
        residuals = jax.block_until_ready(result.residuals)
        jacobian = jax.block_until_ready(result.jacobian)
        evaluation = LeastSquaresEvaluation(
            result=result,
            residuals=residuals,
            jacobian=jacobian,
            elapsed_s=float(base_evaluation.elapsed_s),
        )
        return scale_least_squares_evaluation_columns(evaluation, self.x_scale)

    def residuals(self, scaled_parameter_values=None) -> np.ndarray:
        return np.asarray(jax.device_get(self.evaluate(scaled_parameter_values).residuals), dtype=float)

    def jacobian(self, scaled_parameter_values=None) -> np.ndarray:
        return np.asarray(jax.device_get(self.evaluate(scaled_parameter_values).jacobian), dtype=float)

    def input_from_scaled_parameters(self, scaled_parameter_values=None):
        physical_values = self._scaled_to_physical(
            self.x0 if scaled_parameter_values is None else scaled_parameter_values
        )
        entries = boundary_param_entries(self.context, self.parameterization.vmec_tuples)
        return _input_with_boundary_deltas(self.context, physical_values, entries)


def geometry_objective(name: str | ObjectiveRef) -> ObjectiveRef:
    """Return a named geometry objective reference."""

    if isinstance(name, ObjectiveRef):
        if name.family != "geometry":
            raise ValueError(f"Expected a geometry ObjectiveRef, got {name.family!r}.")
        return name
    return ObjectiveRef("geometry", str(name))


def transport_objective(name: str | ObjectiveRef) -> ObjectiveRef:
    """Return a named transport objective reference."""

    if isinstance(name, ObjectiveRef):
        if name.family != "transport":
            raise ValueError(f"Expected a transport ObjectiveRef, got {name.family!r}.")
        return name
    return ObjectiveRef("transport", str(name))


def transformed_geometry_objective(
    base: str | ObjectiveRef,
    value_fn: Callable[[object], object],
    derivative_fn: Callable[[object], object] | None = None,
    *,
    label: str,
) -> GeometryObjectiveTransform:
    """Build a scalar transformed geometry objective from one table row."""

    if derivative_fn is None:
        derivative_fn = jax.grad(lambda x: jnp.asarray(value_fn(x), dtype=jnp.float64))
    return GeometryObjectiveTransform(
        base=geometry_objective(base),
        value_fn=value_fn,
        derivative_fn=derivative_fn,
        label=str(label),
    )


def transformed_transport_objective(
    base: str | ObjectiveRef,
    value_fn: Callable[[object], object],
    derivative_fn: Callable[[object], object] | None = None,
    *,
    label: str,
) -> GeometryObjectiveTransform:
    """Build a scalar transformed transport objective from one table row."""

    if derivative_fn is None:
        derivative_fn = jax.grad(lambda x: jnp.asarray(value_fn(x), dtype=jnp.float64))
    return GeometryObjectiveTransform(
        base=transport_objective(base),
        value_fn=value_fn,
        derivative_fn=derivative_fn,
        label=str(label),
    )


def _normalize_geometry_least_squares_terms(
    terms: Sequence[
        GeometryLeastSquaresTerm
        | LeastSquaresTerm
        | tuple[ObjectiveRef | GeometryObjectiveTransform | str, float, float]
    ],
) -> tuple[GeometryLeastSquaresTerm, ...]:
    normalized: list[GeometryLeastSquaresTerm] = []
    for term in terms:
        if isinstance(term, GeometryLeastSquaresTerm):
            normalized.append(term)
            continue
        if isinstance(term, LeastSquaresTerm):
            normalized.append(
                GeometryLeastSquaresTerm(
                    objective=geometry_objective(term.objective),
                    target=float(term.target),
                    weight=float(term.weight),
                    label=term.label,
                )
            )
            continue
        if not isinstance(term, tuple) or len(term) != 3:
            raise TypeError(
                "Geometry terms must be GeometryLeastSquaresTerm instances, "
                "LeastSquaresTerm instances, or (objective, target, weight) tuples."
            )
        objective, target, weight = term
        if isinstance(objective, GeometryObjectiveTransform):
            if objective.base.family != "geometry":
                raise ValueError(
                    "geometry_least_squares_problem supports only transformed "
                    f"geometry objectives; got {objective.base.family!r}."
                )
            normalized.append(
                GeometryLeastSquaresTerm(
                    objective=objective.base,
                    target=float(target),
                    weight=float(weight),
                    label=objective.label,
                    value_fn=objective.value_fn,
                    derivative_fn=objective.derivative_fn,
                )
            )
        else:
            normalized.append(
                GeometryLeastSquaresTerm(
                    objective=geometry_objective(objective),
                    target=float(target),
                    weight=float(weight),
                )
            )
    if not normalized:
        raise ValueError("At least one geometry least-squares term is required.")
    for term in normalized:
        if term.weight < 0.0:
            raise ValueError(f"Geometry least-squares weights must be non-negative; got {term.weight}.")
    return tuple(normalized)


def _normalize_initial_er_root_least_squares_terms(
    terms: Sequence[
        GeometryLeastSquaresTerm | LeastSquaresTerm | tuple[ObjectiveRef | str, float, float]
    ],
) -> tuple[GeometryLeastSquaresTerm | LeastSquaresTerm, ...]:
    normalized: list[GeometryLeastSquaresTerm | LeastSquaresTerm] = []
    for term in terms:
        if isinstance(term, GeometryLeastSquaresTerm):
            normalized.append(
                term
            )
            continue
        if isinstance(term, LeastSquaresTerm):
            if term.objective.family not in {"geometry", "transport"}:
                raise ValueError(
                    "Initial-Er root optimization supports only geometry and transport terms; "
                    f"got {term.objective.family!r}."
                )
            normalized.append(term)
            continue
        if not isinstance(term, tuple) or len(term) != 3:
            raise TypeError(
                "Mixed geometry/initial-Er terms must be LeastSquaresTerm instances "
                "or (objective, target, weight) tuples."
            )
        objective, target, weight = term
        if isinstance(objective, GeometryObjectiveTransform):
            normalized.append(
                GeometryLeastSquaresTerm(
                    objective=objective.base,
                    target=float(target),
                    weight=float(weight),
                    label=objective.label,
                    value_fn=objective.value_fn,
                    derivative_fn=objective.derivative_fn,
                )
            )
            continue
        if isinstance(objective, ObjectiveRef):
            objective_ref = objective
        else:
            objective_ref = geometry_objective(objective)
        if objective_ref.family not in {"geometry", "transport"}:
            raise ValueError(
                "Initial-Er root optimization supports only geometry and transport terms; "
                f"got {objective_ref.family!r}."
            )
        normalized.append(
            LeastSquaresTerm(
                objective=objective_ref,
                target=float(target),
                weight=float(weight),
            )
        )
    if not normalized:
        raise ValueError("At least one least-squares term is required.")
    for term in normalized:
        if term.weight < 0.0:
            raise ValueError(f"Least-squares weights must be non-negative; got {term.weight}.")
    return tuple(normalized)


def _base_terms_for_mixed_initial_er_root(
    terms: Sequence[GeometryLeastSquaresTerm | LeastSquaresTerm],
) -> tuple[LeastSquaresTerm, ...]:
    base_terms = []
    for term in terms:
        if isinstance(term, GeometryLeastSquaresTerm):
            base_terms.append(
                LeastSquaresTerm(
                    objective=term.objective,
                    target=0.0,
                    weight=1.0,
                    label=term.residual_label,
                )
            )
        else:
            base_terms.append(
                LeastSquaresTerm(
                    objective=term.objective,
                    target=0.0,
                    weight=1.0,
                    label=term.residual_label,
                )
            )
    return tuple(base_terms)


def _assemble_mixed_initial_er_root_result(
    terms: Sequence[GeometryLeastSquaresTerm | LeastSquaresTerm],
    *,
    base_evaluation: LeastSquaresEvaluation,
) -> LeastSquaresResult:
    base_values = jnp.asarray(base_evaluation.result.residuals)
    base_jacobian = jnp.asarray(base_evaluation.result.jacobian)
    residual_rows = []
    jacobian_rows = []
    residual_labels = []
    objective_values: dict[str, object] = {}
    label_counts: dict[str, int] = {}
    for term_i, term in enumerate(terms):
        base_value = base_values[term_i]
        base_jacobian_row = base_jacobian[term_i]
        if isinstance(term, GeometryLeastSquaresTerm) and term.value_fn is not None:
            value = term.value_fn(base_value)
            if term.derivative_fn is None:
                chain = jax.grad(lambda x: jnp.asarray(term.value_fn(x), dtype=jnp.float64))(base_value)
            else:
                chain = term.derivative_fn(base_value)
        else:
            value = base_value
            chain = jnp.asarray(1.0, dtype=base_value.dtype)
        scale = jnp.asarray(np.sqrt(float(term.weight)), dtype=base_value.dtype)
        residual_rows.append(scale * (value - jnp.asarray(term.target, dtype=base_value.dtype)))
        jacobian_rows.append(scale * chain * base_jacobian_row)
        base_label = term.residual_label
        label_count = label_counts.get(base_label, 0)
        label_counts[base_label] = label_count + 1
        residual_label = base_label if label_count == 0 else f"{base_label}#{label_count + 1}"
        residual_labels.append(residual_label)
        objective_values[residual_label] = value
    return LeastSquaresResult(
        residuals=jnp.stack(residual_rows),
        jacobian=jnp.stack(jacobian_rows),
        residual_labels=tuple(residual_labels),
        parameter_labels=base_evaluation.result.parameter_labels,
        objective_values=objective_values,
    )


def _result_lookup(result: ObjectiveTableResult) -> dict[str, int]:
    return {name: index for index, name in enumerate(result.objective_names)}


def _assemble_geometry_least_squares_result(
    terms: Sequence[GeometryLeastSquaresTerm],
    *,
    parameter_set,
    table: ObjectiveTableResult,
) -> LeastSquaresResult:
    lookup = _result_lookup(table)
    values = jnp.asarray(table.values)
    jacobian_table = jnp.asarray(table.jacobian)
    residual_rows = []
    jacobian_rows = []
    residual_labels = []
    objective_values: dict[str, object] = {}
    label_counts: dict[str, int] = {}
    for term in terms:
        row_index = lookup[term.objective.name]
        base_value = values[row_index]
        base_jacobian = jacobian_table[row_index]
        if term.value_fn is None:
            value = base_value
            chain = jnp.asarray(1.0, dtype=base_value.dtype)
        else:
            value = term.value_fn(base_value)
            chain = term.derivative_fn(base_value)
        scale = jnp.asarray(np.sqrt(float(term.weight)), dtype=base_value.dtype)
        residual_rows.append(scale * (value - jnp.asarray(term.target, dtype=base_value.dtype)))
        jacobian_rows.append(scale * chain * base_jacobian)
        base_label = term.residual_label
        label_count = label_counts.get(base_label, 0)
        label_counts[base_label] = label_count + 1
        residual_label = base_label if label_count == 0 else f"{base_label}#{label_count + 1}"
        residual_labels.append(residual_label)
        objective_values[residual_label] = value
    return LeastSquaresResult(
        residuals=jnp.stack(residual_rows),
        jacobian=jnp.stack(jacobian_rows),
        residual_labels=tuple(residual_labels),
        parameter_labels=parameter_set.vmec_prefixed_labels,
        objective_values=objective_values,
    )


def geometry_least_squares_problem(
    vmec_input,
    terms: Sequence[
        GeometryLeastSquaresTerm
        | LeastSquaresTerm
        | tuple[ObjectiveRef | GeometryObjectiveTransform | str, float, float]
    ],
    *,
    max_mode: int | None = None,
    parameters: str | Sequence[str] | None = None,
    families: str | Sequence[str] | None = "RBC,ZBS",
    scale_mode: str = "ess",
    ess_alpha: float = 1.0,
    mboz: int = 18,
    nboz: int = 18,
    surfaces: Sequence[float] = (0.1, 0.28, 0.46, 0.64, 0.82, 1.0),
    lane: str = "ad",
    max_iter: int | None = None,
    step_size: float | None = None,
    solver_device: str | None = "default",
) -> GeometryLeastSquaresProblem:
    """Build a VMEX-style geometry least-squares problem.

    Use ``max_mode`` for VMEX packed boundary DOFs, or pass explicit
    ``parameters`` such as ``"RBC:1:0,ZBS:1:0"`` for diagnostic runs.
    """

    context = build_geometry_autodiff_context(
        vmec_input,
        param_family="RBC",
        param_m=1,
        param_n=0,
        mboz=int(mboz),
        nboz=int(nboz),
        surface_s=tuple(float(s) for s in surfaces),
    )
    if parameters is not None:
        specs = parse_vmec_boundary_parameter_specs(
            ",".join(parameters) if not isinstance(parameters, str) else parameters
        )
        parameterization = VmexBoundaryParameterization(
            specs=specs,
            scales=tuple(1.0 for _ in specs),
            scale_mode="unit",
        )
    else:
        if max_mode is None:
            raise ValueError("Either max_mode or explicit parameters must be provided.")
        parameterization = vmex_boundary_parameterization(
            context,
            max_mode=int(max_mode),
            families=families,
            scale_mode=scale_mode,
            ess_alpha=float(ess_alpha),
        )
    normalized_terms = _normalize_geometry_least_squares_terms(terms)
    return GeometryLeastSquaresProblem(
        context=context,
        parameterization=parameterization,
        terms=normalized_terms,
        lane=lane,
        max_iter=max_iter,
        step_size=step_size,
        solver_device=solver_device,
    )


def _prepare_initial_er_root_config(config_path, *, device: str | None, vmec_input) -> dict:
    config = prepare_config(config_path, device=device)
    config = copy.deepcopy(config)
    config.setdefault("general", {})["mode"] = "transport"
    transport_output = config.setdefault("transport_output", {})
    transport_output["transport_plot"] = False
    transport_output["transport_write_hdf5"] = False
    transport_output["transport_compare_ambipolarity_residual"] = False
    transport_output["transport_scan_ambipolarity_residual"] = False
    solver_cfg = config.setdefault("transport_solver", {})
    solver_cfg["debug_stage_markers"] = False
    solver_cfg["debug_disable_jit"] = False
    solver_cfg["debug_walltime_attempts"] = False
    config.setdefault("neoclassical", {})["ntx_exact_derivative_mode"] = "direct"
    config.setdefault("neoclassical", {})["ntx_exact_derivative_field_pullback_mode"] = "compact_vjp"
    if vmec_input is not None:
        config.setdefault("geometry", {})["vmec_input_file"] = str(vmec_input)
    return config


def _profile_values_from_config(config: dict, dtype) -> jnp.ndarray:
    profiles = config.get("profiles", {})
    defaults = {
        "n0": 4.21,
        "T0": 17.8,
        "density_shape_power": 2.0,
        "temperature_shape_power": 2.0,
        "density_shape_alpha": 1.0,
        "temperature_shape_alpha": 1.0,
    }
    values = []
    for name in PROFILE_PARAMETER_ORDER:
        raw = profiles.get(name, defaults[name])
        if isinstance(raw, (list, tuple)):
            raw = raw[0]
        values.append(float(raw))
    return jnp.asarray(values, dtype=dtype)


def _profile_scales_from_values(values, mode: str):
    mode_eff = str(mode).strip().lower()
    values_arr = jnp.asarray(values, dtype=jnp.float64)
    if mode_eff in ("identity", "none", "unit"):
        return jnp.ones_like(values_arr)
    if mode_eff in ("nominal", "baseline"):
        return jnp.maximum(jnp.abs(values_arr), jnp.asarray(1.0e-12, dtype=values_arr.dtype))
    raise ValueError("profile_scale_mode must be 'identity' or 'nominal'.")


def geometry_initial_er_root_only_least_squares_problem(
    config,
    terms: Sequence[
        GeometryLeastSquaresTerm
        | LeastSquaresTerm
        | tuple[ObjectiveRef | GeometryObjectiveTransform | str, float, float]
    ],
    *,
    vmec_input=None,
    max_mode: int | None = None,
    parameters: str | Sequence[str] | None = None,
    include_profiles: bool = False,
    profile_parameters: str | Sequence[str] | None = "n0,T0,density_shape_power,temperature_shape_power",
    profile_scale_mode: str = "identity",
    families: str | Sequence[str] | None = "RBC,ZBS",
    scale_mode: str = "ess",
    ess_alpha: float = 1.0,
    mboz: int = 18,
    nboz: int = 18,
    surfaces: Sequence[float] = (0.1, 0.28, 0.46, 0.64, 0.82, 1.0),
    n_r: int | None = None,
    n_theta: int | None = None,
    n_zeta: int | None = None,
    n_xi: int | None = None,
    surface_backend: str | None = None,
    geometry_lane: str = "ad",
    geometry_max_iter: int | None = None,
    geometry_step_size: float | None = None,
    geometry_solver_device: str | None = "default",
    device: str | None = "default",
    root_options: Mapping[str, object] | None = None,
    reverse_stage_mode: str = "off",
) -> GeometryInitialErRootLeastSquaresProblem:
    """Build an optimizer problem for geometry terms plus initial-Er root terms.

    The returned problem is intentionally thin and script-friendly: users define
    least-squares terms in the optimization script, while this helper owns the
    transport runtime setup and validated reverse-table calls.
    """

    mode = str(reverse_stage_mode).strip().lower()
    if mode not in {"off", "payload_optimization", "vmex_like"}:
        raise ValueError(
            "reverse_stage_mode must be 'off', 'payload_optimization', or 'vmex_like'."
        )
    if mode == "vmex_like":
        raise NotImplementedError(
            "reverse_stage_mode='vmex_like' is disabled while the incomplete "
            "outer-JIT experiment is replaced by retained existing reverse kernels."
        )
    config_eff = _prepare_initial_er_root_config(config, device=device, vmec_input=vmec_input)
    geom_cfg = config_eff.get("geometry", {})
    neoclassical_cfg = config_eff.get("neoclassical", {})
    vmec_input_eff = geom_cfg.get("vmec_input_file")
    if vmec_input_eff is None:
        raise ValueError("geometry.vmec_input_file is required.")
    context = build_geometry_autodiff_context(
        vmec_input_eff,
        param_family="RBC",
        param_m=1,
        param_n=0,
        mboz=int(mboz),
        nboz=int(nboz),
        surface_s=tuple(float(s) for s in surfaces),
    )
    if parameters is not None:
        specs = parse_vmec_boundary_parameter_specs(
            ",".join(parameters) if not isinstance(parameters, str) else parameters
        )
        geometry_parameterization = VmexBoundaryParameterization(
            specs=specs,
            scales=tuple(1.0 for _ in specs),
            scale_mode="unit",
        )
    else:
        if max_mode is None and not include_profiles:
            raise ValueError("Either max_mode or explicit geometry parameters must be provided.")
        geometry_parameterization = (
            None
            if max_mode is None
            else vmex_boundary_parameterization(
                context,
                max_mode=int(max_mode),
                families=families,
                scale_mode=scale_mode,
                ess_alpha=float(ess_alpha),
            )
        )
    profile_specs = (
        parse_profile_parameter_specs(profile_parameters)
        if include_profiles
        else ()
    )
    parameter_set = reverse_ad_optimization_parameter_set(
        include_profiles=bool(profile_specs),
        profiles=tuple(spec.name for spec in profile_specs) if profile_specs else None,
        vmec_boundary=() if geometry_parameterization is None else geometry_parameterization.specs,
    )
    runtime, baseline_state = build_runtime_context(config_eff)
    if baseline_state is None:
        raise RuntimeError("transport runtime did not return an initial state.")
    baseline_profile_values = _profile_values_from_config(
        config_eff,
        jnp.asarray(baseline_state.pressure).dtype,
    )
    profile_scales = _profile_scales_from_values(baseline_profile_values, profile_scale_mode)
    if geometry_max_iter is None:
        geometry_max_iter = geom_cfg.get("vmec_max_iter")
    if geometry_solver_device is None:
        geometry_solver_device = geom_cfg.get("vmec_implicit_solver_device", "default")
    normalized_terms = _normalize_initial_er_root_least_squares_terms(terms)
    raw_block_stage = (
        None
        if geometry_parameterization is None
        else geometry_raw_block_stage(
            context,
            tuple(spec.as_tuple() for spec in geometry_parameterization.specs),
            max_iter=geometry_max_iter,
        )
    )
    optimization_stage_layout = None
    optimization_stage = None
    payload_optimization_stage = None
    if mode == "payload_optimization":
        payload_optimization_stage = build_geometry_payload_optimization_stage(
            geometry_context=context,
            n_r=int(n_r if n_r is not None else geom_cfg.get("n_radial", 51)),
            n_theta=int(n_theta if n_theta is not None else neoclassical_cfg.get("ntx_exact_n_theta", 25)),
            n_zeta=int(n_zeta if n_zeta is not None else neoclassical_cfg.get("ntx_exact_n_zeta", 25)),
            n_xi=int(n_xi if n_xi is not None else neoclassical_cfg.get("ntx_exact_n_xi", 64)),
            surface_backend=str(surface_backend or neoclassical_cfg.get("ntx_exact_surface_backend", "vmec")),
            flux_model=str(neoclassical_cfg.get("flux_model", "ntx_database")),
        )
    if mode == "vmex_like":
        normalized_stage_terms = normalize_least_squares_terms(terms)
        optimization_stage_layout = initial_root_stage_layout(
            config=config_eff,
            objective_names=tuple(term.objective.name for term in normalized_stage_terms),
            geometry_param_specs=tuple(spec.as_tuple() for spec in geometry_parameterization.specs),
            n_r=n_r,
            n_theta=n_theta,
            n_zeta=n_zeta,
            n_xi=n_xi,
            surface_backend=surface_backend,
        )
        if raw_block_stage is None:
            raise ValueError("vmex_like initial-root optimization requires VMEC boundary parameters.")
        stage_transport_objectives = tuple(
            term.objective.name for term in normalized_stage_terms if term.objective.family == "transport"
        )
        stage_profile_specs = tuple(ProfileParameterSpec(name) for name in PROFILE_PARAMETER_ORDER)
        stage_support_payload = find_ntx_support_payload(runtime)
        stage_n_r = int(n_r if n_r is not None else geom_cfg.get("n_radial", 51))
        stage_boozer_surface_sampling = _boozer_surface_indices_and_rho(
            context.static,
            _neopax_geometry_requested_sample_rho(context, n_r=stage_n_r),
        )
        stage_r00_center = np.linspace(0.0, 1.0, stage_n_r, dtype=float)
        stage_r00_faces = (
            np.asarray([0.0, 1.0], dtype=float)
            if stage_n_r == 1
            else np.concatenate(
                [np.asarray([0.0]), 0.5 * (stage_r00_center[:-1] + stage_r00_center[1:]), np.asarray([1.0])]
            )
        )
        stage_r00_boozer_surface_sampling = _boozer_surface_indices_and_rho(
            context.static, np.unique(np.concatenate([stage_r00_center, stage_r00_faces]))
        )

        def _stage_pre_root_state(profile_values):
            return initial_state_for_parameter_vector(
                profile_values, config=config_eff, initial_er_root_ad="off",
                baseline_state=baseline_state, profile_cfg=config_eff.get("profiles", {}), runtime=runtime,
            )

        def _stage_root(raw_block_solve, profile_values):
            return _optimization_root_to_payload_cotangents(
                config=config_eff, requested_objectives=stage_transport_objectives, runtime=runtime,
                profile_values_arr=profile_values, pre_root_state_from_profile_values=_stage_pre_root_state,
                geometry_context=context, n_r=stage_n_r,
                n_theta=int(n_theta if n_theta is not None else neoclassical_cfg.get("ntx_exact_n_theta", 25)),
                n_zeta=int(n_zeta if n_zeta is not None else neoclassical_cfg.get("ntx_exact_n_zeta", 25)),
                n_xi=int(n_xi if n_xi is not None else neoclassical_cfg.get("ntx_exact_n_xi", 64)),
                surface_backend=str(surface_backend or neoclassical_cfg.get("ntx_exact_surface_backend", "vmec")),
                raw_block_solve=raw_block_solve, support_payload=stage_support_payload,
                use_runtime_payload=False, profile_specs=stage_profile_specs, options=dict(root_options or {}),
                boozer_surface_sampling=stage_boozer_surface_sampling,
                r00_boozer_surface_sampling=stage_r00_boozer_surface_sampling,
            )

        def _stage_payload(raw_block_solve, geometry_deltas, values, profile_gradient, support_bars):
            return _optimization_payload_to_vmec_table(
                objective_labels=stage_transport_objectives,
                profile_parameter_labels=tuple(spec.name for spec in stage_profile_specs),
                geometry_parameter_labels=tuple(spec.label for spec in parameter_set.vmec_boundary_specs),
                objective_values=values, profile_gradient_matrix=profile_gradient,
                geometry_context=context, baseline_geometry_deltas=geometry_deltas,
                geometry_param_specs=tuple(spec.as_tuple() for spec in parameter_set.vmec_boundary_specs),
                support_bars=support_bars, support_component_bars_by_name={},
                include_component_pullbacks=False, combined_geometry_payload=True,
                n_r=int(n_r if n_r is not None else geom_cfg.get("n_radial", 51)),
                n_theta=int(n_theta if n_theta is not None else neoclassical_cfg.get("ntx_exact_n_theta", 25)),
                n_zeta=int(n_zeta if n_zeta is not None else neoclassical_cfg.get("ntx_exact_n_zeta", 25)),
                n_xi=int(n_xi if n_xi is not None else neoclassical_cfg.get("ntx_exact_n_xi", 64)),
                surface_backend=str(surface_backend or neoclassical_cfg.get("ntx_exact_surface_backend", "vmec")),
                max_iter=geometry_max_iter, solver_device=geometry_solver_device,
                progress_label=None, raw_block_solve=raw_block_solve, return_branch_gradients=False,
            )

        optimization_stage = build_compiled_geometry_initial_root_stage(
            layout=optimization_stage_layout,
            raw_block_stage=raw_block_stage,
            root_impl=_stage_root,
            payload_impl=_stage_payload,
        )
    return GeometryInitialErRootLeastSquaresProblem(
        config=config_eff,
        context=context,
        runtime=runtime,
        baseline_state=baseline_state,
        baseline_profile_values=baseline_profile_values,
        profile_scales=profile_scales,
        parameter_set=parameter_set,
        geometry_parameterization=geometry_parameterization,
        profile_specs=profile_specs,
        terms=normalized_terms,
        n_r=int(n_r if n_r is not None else geom_cfg.get("n_radial", 51)),
        n_theta=int(n_theta if n_theta is not None else neoclassical_cfg.get("ntx_exact_n_theta", 25)),
        n_zeta=int(n_zeta if n_zeta is not None else neoclassical_cfg.get("ntx_exact_n_zeta", 25)),
        n_xi=int(n_xi if n_xi is not None else neoclassical_cfg.get("ntx_exact_n_xi", 64)),
        surface_backend=str(
            surface_backend
            if surface_backend is not None
            else neoclassical_cfg.get(
                "ntx_exact_surface_backend",
                neoclassical_cfg.get("ntx_surface_backend", "vmec"),
            )
        ),
        geometry_lane=geometry_lane,
        geometry_max_iter=geometry_max_iter,
        geometry_step_size=geometry_step_size,
        geometry_solver_device=geometry_solver_device,
        root_options=None if root_options is None else dict(root_options),
        raw_block_stage=raw_block_stage,
        optimization_stage_layout=optimization_stage_layout,
        optimization_stage=optimization_stage,
        payload_optimization_stage=payload_optimization_stage,
        reverse_stage_mode=mode,
    )


def _prepare_full_transport_config(config_path, *, device: str | None) -> dict:
    config = prepare_config(config_path, device=device)
    config = copy.deepcopy(config)
    config.setdefault("general", {})["mode"] = "transport"
    transport_output = config.setdefault("transport_output", {})
    transport_output["transport_plot"] = False
    transport_output["transport_write_hdf5"] = False
    transport_output["transport_compare_ambipolarity_residual"] = False
    transport_output["transport_scan_ambipolarity_residual"] = False
    solver_cfg = config.setdefault("transport_solver", {})
    solver_cfg["debug_stage_markers"] = False
    solver_cfg["debug_disable_jit"] = False
    solver_cfg["debug_walltime_attempts"] = False
    config.setdefault("neoclassical", {})["ntx_exact_derivative_mode"] = "direct"
    config.setdefault("neoclassical", {})["ntx_exact_derivative_field_pullback_mode"] = "compact_vjp"
    return config


def full_transport_profile_least_squares_problem(
    config,
    terms: Sequence[
        GeometryLeastSquaresTerm
        | LeastSquaresTerm
        | tuple[ObjectiveRef | GeometryObjectiveTransform | str, float, float]
    ],
    *,
    profile_parameters: str | Sequence[str] | None = "n0,T0,density_shape_power,temperature_shape_power",
    profile_scale_mode: str = "nominal",
    device: str | None = "default",
    accepted_step_limit: int | None = 16,
    reverse_segment_length: int | str | None = 4,
    initial_er_root_ad: str = "jax_selected_root",
    radau_jacobian_reuse_mode: str = "legacy",
    reverse_stage_adjoint_solve_mode: str = "bicgstab",
    reverse_rhs_transpose_mode: str = "explicit_ntx_interpolated",
    reverse_stage_cotangent_mode: str = "full",
    reverse_step_bwd_mode: str = "reduced_cotangent",
    reverse_stage_adjoint_memory_mode: str = "default",
    reverse_stage_adjoint_iter_maxiter: int = 40,
    reverse_stage_adjoint_iter_tol: float = 1.0e-10,
    reverse_stage_adjoint_woodbury_rank: int = 24,
) -> ProfileFullTransportLeastSquaresProblem:
    """Build a profile-only optimizer problem for full Radau transport objectives."""

    config_eff = _prepare_full_transport_config(config, device=device)
    solver_cfg = config_eff.setdefault("transport_solver", {})
    solver_cfg["radau_jacobian_reuse_mode"] = str(radau_jacobian_reuse_mode)
    runtime, baseline_state = build_runtime_context(config_eff)
    if baseline_state is None:
        raise RuntimeError("transport runtime did not return an initial state.")
    baseline_profile_values = _profile_values_from_config(
        config_eff,
        jnp.asarray(baseline_state.pressure).dtype,
    )
    profile_scales = _profile_scales_from_values(baseline_profile_values, profile_scale_mode)
    profile_specs = parse_profile_parameter_specs(profile_parameters)
    parameter_set = reverse_ad_optimization_parameter_set(
        include_profiles=True,
        profiles=tuple(spec.name for spec in profile_specs),
        vmec_boundary=(),
    )
    normalized_terms = _normalize_initial_er_root_least_squares_terms(terms)
    profile_cfg = copy.deepcopy(config_eff.get("profiles", {}))
    profile_cfg.setdefault("model", "standard_analytical")
    neoclassical_cfg = dict(config_eff.get("neoclassical", {}))
    table_context = realtime_geometry_transport_reverse_table_context(
        config=config_eff,
        baseline_values=baseline_profile_values,
        baseline_runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        neoclassical_cfg=neoclassical_cfg,
    )

    args = SimpleNamespace(
        realtime_geometry_gradient_path="reverse_payload",
        accepted_step_limit=accepted_step_limit,
        reverse_segment_length=reverse_segment_length,
        initial_er_root_ad=str(initial_er_root_ad),
        initial_Er_root_ad=str(initial_er_root_ad),
        reverse_stage_adjoint_solve_mode=str(reverse_stage_adjoint_solve_mode),
        reverse_rhs_transpose_mode=str(reverse_rhs_transpose_mode),
        reverse_stage_cotangent_mode=str(reverse_stage_cotangent_mode),
        reverse_step_bwd_mode=str(reverse_step_bwd_mode),
        reverse_stage_adjoint_memory_mode=str(reverse_stage_adjoint_memory_mode),
        reverse_stage_adjoint_iter_maxiter=int(reverse_stage_adjoint_iter_maxiter),
        reverse_stage_adjoint_iter_tol=float(reverse_stage_adjoint_iter_tol),
        reverse_stage_adjoint_woodbury_rank=int(reverse_stage_adjoint_woodbury_rank),
    )

    def _internal_support_segment_probe(
        *,
        args,
        config,
        baseline_values,
        baseline_runtime,
        baseline_state,
        profile_cfg,
        neoclassical_cfg,
        return_report=False,
    ):
        return run_internal_realtime_geometry_support_segment_probe(
            args=args,
            context=realtime_geometry_transport_reverse_table_context(
                config=config,
                baseline_values=baseline_values,
                baseline_runtime=baseline_runtime,
                baseline_state=baseline_state,
                profile_cfg=profile_cfg,
                neoclassical_cfg=neoclassical_cfg,
            ),
            return_report=return_report,
            suppress_diagnostics=True,
        )

    support_segment_executor = realtime_geometry_transport_reverse_support_segment_executor(
        support_segment_probe=_internal_support_segment_probe,
        config=config_eff,
        baseline_values=baseline_profile_values,
        baseline_runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        neoclassical_cfg=neoclassical_cfg,
    )
    grouped_inputs = realtime_geometry_transport_reverse_grouped_inputs(
        args=args,
        config=config_eff,
        baseline_values=baseline_profile_values,
        baseline_runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        neoclassical_cfg=neoclassical_cfg,
        support_segment_executor=support_segment_executor,
    )
    options = {
        "quiet": True,
        "accepted_step_limit": None if accepted_step_limit is None else int(accepted_step_limit),
        "reverse_segment_length": (
            None
            if reverse_segment_length is None
            else reverse_segment_length
            if isinstance(reverse_segment_length, str)
            else int(reverse_segment_length)
        ),
        "initial_er_root_ad": str(initial_er_root_ad),
        "reverse_stage_adjoint_solve_mode": str(reverse_stage_adjoint_solve_mode),
        "reverse_rhs_transpose_mode": str(reverse_rhs_transpose_mode),
        "reverse_stage_cotangent_mode": str(reverse_stage_cotangent_mode),
        "reverse_step_bwd_mode": str(reverse_step_bwd_mode),
        "reverse_stage_adjoint_memory_mode": str(reverse_stage_adjoint_memory_mode),
        "reverse_stage_adjoint_iter_maxiter": int(reverse_stage_adjoint_iter_maxiter),
        "reverse_stage_adjoint_iter_tol": float(reverse_stage_adjoint_iter_tol),
    }
    return ProfileFullTransportLeastSquaresProblem(
        config=config_eff,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        neoclassical_cfg=neoclassical_cfg,
        baseline_profile_values=baseline_profile_values,
        profile_scales=profile_scales,
        parameter_set=parameter_set,
        terms=normalized_terms,
        table_context=grouped_inputs.table_context,
        run_grouped_report=grouped_inputs.run_grouped_report,
        options=options,
    )


def geometry_full_transport_least_squares_problem(
    config,
    terms: Sequence[
        GeometryLeastSquaresTerm
        | LeastSquaresTerm
        | tuple[ObjectiveRef | GeometryObjectiveTransform | str, float, float]
    ],
    *,
    vmec_input=None,
    max_mode: int | None = None,
    parameters: str | Sequence[str] | None = None,
    families: str | Sequence[str] | None = "RBC,ZBS",
    scale_mode: str = "ess",
    ess_alpha: float = 1.0,
    mboz: int = 18,
    nboz: int = 18,
    surfaces: Sequence[float] = (0.1, 0.28, 0.46, 0.64, 0.82, 1.0),
    n_r: int | None = None,
    n_theta: int | None = None,
    n_zeta: int | None = None,
    n_xi: int | None = None,
    surface_backend: str | None = None,
    geometry_lane: str = "ad",
    geometry_max_iter: int | None = None,
    geometry_step_size: float | None = None,
    geometry_solver_device: str | None = "default",
    device: str | None = "default",
    accepted_step_limit: int | None = None,
    reverse_segment_length: int | str | None = 4,
    initial_er_root_ad: str = "jax_selected_root",
    radau_jacobian_reuse_mode: str = "legacy",
    reverse_stage_adjoint_solve_mode: str = "bicgstab",
    reverse_rhs_transpose_mode: str = "explicit_ntx_interpolated",
    reverse_stage_cotangent_mode: str = "full",
    reverse_step_bwd_mode: str = "reduced_cotangent",
    reverse_stage_adjoint_memory_mode: str = "default",
    reverse_stage_adjoint_iter_maxiter: int = 40,
    reverse_stage_adjoint_iter_tol: float = 1.0e-10,
    reverse_stage_adjoint_woodbury_rank: int = 24,
    max_reverse_accepted_steps: int | None = None,
) -> GeometryFullTransportLeastSquaresProblem:
    """Build a geometry-only optimizer problem for full Radau transport objectives."""

    config_eff = _prepare_full_transport_config(config, device=device)
    config_eff.setdefault("transport_solver", {})["radau_jacobian_reuse_mode"] = str(
        radau_jacobian_reuse_mode
    )
    if vmec_input is not None:
        config_eff.setdefault("geometry", {})["vmec_input_file"] = str(vmec_input)
    geom_cfg = config_eff.get("geometry", {})
    neoclassical_cfg = config_eff.get("neoclassical", {})
    vmec_input_eff = geom_cfg.get("vmec_input_file")
    if vmec_input_eff is None:
        raise ValueError("geometry.vmec_input_file is required.")
    context = build_geometry_autodiff_context(
        vmec_input_eff,
        param_family="RBC",
        param_m=1,
        param_n=0,
        mboz=int(mboz),
        nboz=int(nboz),
        surface_s=tuple(float(s) for s in surfaces),
    )
    if parameters is not None:
        specs = parse_vmec_boundary_parameter_specs(
            ",".join(parameters) if not isinstance(parameters, str) else parameters
        )
        parameterization = VmexBoundaryParameterization(
            specs=specs,
            scales=tuple(1.0 for _ in specs),
            scale_mode="unit",
        )
    else:
        if max_mode is None:
            raise ValueError("Either max_mode or explicit geometry parameters must be provided.")
        parameterization = vmex_boundary_parameterization(
            context,
            max_mode=int(max_mode),
            families=families,
            scale_mode=scale_mode,
            ess_alpha=float(ess_alpha),
        )
    parameter_set = reverse_ad_optimization_parameter_set(
        include_profiles=False,
        vmec_boundary=parameterization.specs,
    )
    runtime, baseline_state = build_runtime_context(config_eff)
    if baseline_state is None:
        raise RuntimeError("transport runtime did not return an initial state.")
    baseline_profile_values = _profile_values_from_config(
        config_eff,
        jnp.asarray(baseline_state.pressure).dtype,
    )
    baseline_values = jnp.concatenate(
        [
            baseline_profile_values,
            jnp.zeros((len(parameterization.specs),), dtype=baseline_profile_values.dtype),
        ],
        axis=0,
    )
    profile_cfg = copy.deepcopy(config_eff.get("profiles", {}))
    profile_cfg.setdefault("model", "standard_analytical")
    neoclassical_cfg = dict(neoclassical_cfg)
    table_context = realtime_geometry_transport_reverse_table_context(
        config=config_eff,
        baseline_values=baseline_values,
        baseline_runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        neoclassical_cfg=neoclassical_cfg,
    )
    n_r_eff = int(n_r if n_r is not None else geom_cfg.get("n_radial", 51))
    n_theta_eff = int(n_theta if n_theta is not None else neoclassical_cfg.get("ntx_exact_n_theta", 25))
    n_zeta_eff = int(n_zeta if n_zeta is not None else neoclassical_cfg.get("ntx_exact_n_zeta", 25))
    n_xi_eff = int(n_xi if n_xi is not None else neoclassical_cfg.get("ntx_exact_n_xi", 64))
    surface_backend_eff = str(
        surface_backend
        if surface_backend is not None
        else neoclassical_cfg.get(
            "ntx_exact_surface_backend",
            neoclassical_cfg.get("ntx_surface_backend", "vmec"),
        )
    )
    if geometry_max_iter is None:
        geometry_max_iter = geom_cfg.get("vmec_max_iter")
    if geometry_solver_device is None:
        geometry_solver_device = geom_cfg.get("vmec_implicit_solver_device", "default")
    options = {
        "quiet": True,
        "accepted_step_limit": None if accepted_step_limit is None else int(accepted_step_limit),
        "reverse_segment_length": (
            None
            if reverse_segment_length is None
            else reverse_segment_length
            if isinstance(reverse_segment_length, str)
            else int(reverse_segment_length)
        ),
        "initial_er_root_ad": str(initial_er_root_ad),
        "reverse_stage_adjoint_solve_mode": str(reverse_stage_adjoint_solve_mode),
        "reverse_rhs_transpose_mode": str(reverse_rhs_transpose_mode),
        "reverse_stage_cotangent_mode": str(reverse_stage_cotangent_mode),
        "reverse_step_bwd_mode": str(reverse_step_bwd_mode),
        "reverse_stage_adjoint_memory_mode": str(reverse_stage_adjoint_memory_mode),
        "reverse_stage_adjoint_iter_maxiter": int(reverse_stage_adjoint_iter_maxiter),
        "reverse_stage_adjoint_iter_tol": float(reverse_stage_adjoint_iter_tol),
        "reverse_stage_adjoint_woodbury_rank": int(reverse_stage_adjoint_woodbury_rank),
        "max_reverse_accepted_steps": (
            None if max_reverse_accepted_steps is None else int(max_reverse_accepted_steps)
        ),
        "n_r": n_r_eff,
        "n_theta": n_theta_eff,
        "n_zeta": n_zeta_eff,
        "n_xi": n_xi_eff,
        "surface_backend": surface_backend_eff,
        "max_iter": geometry_max_iter,
        "solver_device": geometry_solver_device,
    }
    table_result_builder = internal_realtime_geometry_transport_reverse_table_result_builder(
        table_context=table_context,
        geometry_context=context,
        baseline_geometry_deltas=jnp.zeros((len(parameterization.specs),), dtype=jnp.float64),
        combined_geometry_payload=True,
        n_r=n_r_eff,
        n_theta=n_theta_eff,
        n_zeta=n_zeta_eff,
        n_xi=n_xi_eff,
        surface_backend=surface_backend_eff,
        max_iter=geometry_max_iter,
        solver_device=str(geometry_solver_device),
        accepted_step_limit=accepted_step_limit,
        reverse_segment_length=reverse_segment_length,
        initial_er_root_ad=str(initial_er_root_ad),
        reverse_stage_adjoint_solve_mode=str(reverse_stage_adjoint_solve_mode),
        reverse_rhs_transpose_mode=str(reverse_rhs_transpose_mode),
        reverse_stage_cotangent_mode=str(reverse_stage_cotangent_mode),
        reverse_step_bwd_mode=str(reverse_step_bwd_mode),
        reverse_stage_adjoint_memory_mode=str(reverse_stage_adjoint_memory_mode),
        reverse_stage_adjoint_iter_maxiter=int(reverse_stage_adjoint_iter_maxiter),
        reverse_stage_adjoint_iter_tol=float(reverse_stage_adjoint_iter_tol),
        reverse_stage_adjoint_woodbury_rank=int(reverse_stage_adjoint_woodbury_rank),
        max_reverse_accepted_steps=(
            None if max_reverse_accepted_steps is None else int(max_reverse_accepted_steps)
        ),
        progress_label="[optimization] full transport geometry payload pullback:",
    )
    normalized_terms = _normalize_initial_er_root_least_squares_terms(terms)
    return GeometryFullTransportLeastSquaresProblem(
        config=config_eff,
        context=context,
        runtime=runtime,
        baseline_state=baseline_state,
        parameterization=parameterization,
        parameter_set=parameter_set,
        terms=normalized_terms,
        table_context=table_context,
        table_result_builder=table_result_builder,
        options=options,
        geometry_lane=geometry_lane,
        geometry_max_iter=geometry_max_iter,
        geometry_step_size=geometry_step_size,
        geometry_solver_device=geometry_solver_device,
    )


def least_squares(problem: GeometryLeastSquaresProblem, **kwargs):
    """Run SciPy least_squares on a NEOPAX geometry least-squares problem."""

    from scipy.optimize import least_squares as scipy_least_squares

    iteration_reporter = kwargs.pop("iteration_reporter", None)
    initial_evaluation = kwargs.pop("initial_evaluation", None)
    cache: dict[tuple[float, ...], LeastSquaresEvaluation] = {}
    failed_cache: dict[tuple[float, ...], tuple[np.ndarray, np.ndarray, str]] = {}
    verbose = int(kwargs.get("verbose", 0) or 0)
    state: dict[str, object] = {"nres": None, "npar": problem.parameter_count, "eval_count": 0}

    def _key(x):
        return tuple(np.asarray(x, dtype=float).tolist())

    def _evaluate(x):
        key = _key(x)
        if key in failed_cache:
            raise RuntimeError(f"cached failed least-squares trial: {failed_cache[key][2]}")
        evaluation = cache.get(key)
        if evaluation is None:
            evaluation = problem.evaluate(jnp.asarray(x, dtype=jnp.float64))
            cache.clear()
            cache[key] = evaluation
        return evaluation

    def _fun(x):
        key = _key(x)
        failed = failed_cache.get(key)
        if failed is not None:
            residuals, _jacobian, reason = failed
            state["eval_count"] = int(state["eval_count"]) + 1
            if verbose:
                print(
                    f"[NEOPAX least_squares] eval={int(state['eval_count'])} "
                    f"cached failed trial -> penalty residual: {reason}",
                    flush=True,
                )
            return residuals
        try:
            residuals = np.asarray(jax.device_get(_evaluate(x).residuals), dtype=float)
        except Exception as exc:
            if state["nres"] is None:
                raise
            state["eval_count"] = int(state["eval_count"]) + 1
            residuals = np.full((int(state["nres"]),), 1.0e6, dtype=float)
            jacobian = np.zeros((int(state["nres"]), int(state["npar"])), dtype=float)
            failed_cache[key] = (residuals, jacobian, str(exc))
            cache.clear()
            if verbose:
                print(
                    f"[NEOPAX least_squares] eval={int(state['eval_count'])} "
                    f"trial solve failed -> penalty residual: {exc}",
                    flush=True,
                )
            return residuals
        residuals = np.where(np.isfinite(residuals), residuals, 1.0e6)
        state["nres"] = int(residuals.size)
        state["eval_count"] = int(state["eval_count"]) + 1
        if verbose:
            cost = 0.5 * float(residuals @ residuals)
            details = ""
            if iteration_reporter is not None:
                details_text = str(iteration_reporter(_evaluate(x))).strip()
                if details_text:
                    details = f" {details_text}"
            print(
                f"[NEOPAX least_squares] eval={int(state['eval_count'])} "
                f"cost={cost:.6e} residual_norm={float(np.linalg.norm(residuals)):.6e}{details}",
                flush=True,
            )
        return residuals

    def _jac(x):
        key = _key(x)
        failed = failed_cache.get(key)
        if failed is not None:
            _residuals, jacobian, reason = failed
            if verbose:
                print(
                    f"[NEOPAX least_squares] cached failed trial -> zero jacobian: {reason}",
                    flush=True,
                )
            return jacobian
        try:
            jacobian = np.asarray(jax.device_get(_evaluate(x).jacobian), dtype=float)
        except Exception as exc:
            if state["nres"] is None:
                raise
            jacobian = np.zeros((int(state["nres"]), int(state["npar"])), dtype=float)
            residuals = np.full((int(state["nres"]),), 1.0e6, dtype=float)
            failed_cache[key] = (residuals, jacobian, str(exc))
            cache.clear()
            if verbose:
                print(
                    f"[NEOPAX least_squares] trial jacobian failed -> zero jacobian: {exc}",
                    flush=True,
                )
            return jacobian
        return np.where(np.isfinite(jacobian), jacobian, 0.0)

    x0 = np.asarray(jax.device_get(problem.x0), dtype=float)
    if initial_evaluation is not None:
        initial_residuals = np.asarray(jax.device_get(initial_evaluation.residuals), dtype=float)
        initial_jacobian = np.asarray(jax.device_get(initial_evaluation.jacobian), dtype=float)
        if initial_residuals.ndim != 1:
            raise ValueError(
                "initial_evaluation residuals must be one-dimensional; "
                f"got shape={initial_residuals.shape}."
            )
        if initial_jacobian.ndim != 2:
            raise ValueError(
                "initial_evaluation jacobian must be two-dimensional; "
                f"got shape={initial_jacobian.shape}."
            )
        if int(initial_jacobian.shape[0]) != int(initial_residuals.shape[0]):
            raise ValueError(
                "initial_evaluation jacobian row count must match residual count; "
                f"residuals.shape={initial_residuals.shape}, jacobian.shape={initial_jacobian.shape}."
            )
        if int(initial_jacobian.shape[1]) != int(problem.parameter_count):
            raise ValueError(
                "initial_evaluation jacobian column count must match problem parameter count; "
                f"jacobian.shape={initial_jacobian.shape}, parameter_count={problem.parameter_count}."
            )
        cache[_key(x0)] = initial_evaluation
    x_scale = np.asarray(jax.device_get(problem.x_scale), dtype=float)
    kwargs.setdefault("x_scale", x_scale)
    return scipy_least_squares(_fun, x0, jac=_jac, **kwargs)


__all__ = [
    "GeometryLeastSquaresTerm",
    "GeometryObjectiveTransform",
    "GeometryInitialErRootLeastSquaresProblem",
    "GeometryFullTransportLeastSquaresProblem",
    "GeometryLeastSquaresProblem",
    "ProfileFullTransportLeastSquaresProblem",
    "RepeatedEvaluationMemorySample",
    "full_transport_profile_least_squares_problem",
    "geometry_full_transport_least_squares_problem",
    "geometry",
    "geometry_objective",
    "geometry_initial_er_root_only_least_squares_problem",
    "geometry_least_squares_problem",
    "least_squares",
    "repeated_evaluation_memory_samples",
    "transformed_geometry_objective",
    "transformed_transport_objective",
    "transport",
    "transport_objective",
]
