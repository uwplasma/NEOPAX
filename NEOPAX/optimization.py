"""Optimization helpers built on NEOPAX reverse-AD internals."""

from __future__ import annotations

import dataclasses
import copy
import time
from collections.abc import Sequence
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from ._geometry_autodiff import (
    _input_with_boundary_deltas,
    boundary_param_entries,
    build_neopax_geometry_and_ntx_exact_lij_support_from_state,
    build_geometry_autodiff_context,
    geometry_raw_block_solve_from_param_vector,
)
from ._orchestrator import build_runtime_context
from ._reverse_ad_initial_er import (
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
    evaluate_geometry_initial_er_root_only_least_squares_benchmark_tables,
    geometry,
    geometry_full_ad_reverse_table,
    scale_least_squares_evaluation_columns,
    transport,
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
from ._reverse_ad_transport import initial_state_for_parameter_vector
from .api import prepare_config


@dataclasses.dataclass(frozen=True, slots=True)
class GeometryObjectiveTransform:
    """Script-defined scalar geometry objective backed by one AD-table row."""

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
    parameter_set: object
    geometry_parameterization: VmexBoundaryParameterization | None
    profile_specs: tuple[ProfileParameterSpec, ...]
    terms: tuple[LeastSquaresTerm, ...]
    n_r: int
    n_theta: int
    n_zeta: int
    n_xi: int
    surface_backend: str = "vmec"
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
        values = []
        profile_lookup = {name: i for i, name in enumerate(PROFILE_PARAMETER_ORDER)}
        for spec in self.parameter_set.specs:
            if isinstance(spec, ProfileParameterSpec):
                values.append(self.baseline_profile_values[profile_lookup[spec.name]])
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
        for spec in self.parameter_set.specs:
            if isinstance(spec, ProfileParameterSpec):
                scales.append(1.0)
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
                physical_values.append(scaled_values[i])
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
        evaluation = evaluate_geometry_initial_er_root_only_least_squares_benchmark_tables(
            self.config,
            parameter_set=self.parameter_set,
            parameter_values=physical_values,
            terms=self.terms,
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

    def initial_er_profile_from_scaled_parameters(self, scaled_parameter_values=None):
        """Return rho, selected initial ambipolar Er, and finite mask for optimizer variables."""

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
        return rho_grid, er_profile, finite_mask


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
) -> tuple[LeastSquaresTerm, ...]:
    normalized: list[LeastSquaresTerm] = []
    for term in terms:
        if isinstance(term, GeometryLeastSquaresTerm):
            if term.value_fn is not None or term.derivative_fn is not None:
                raise NotImplementedError(
                    "Transformed geometry terms are supported by geometry_least_squares_problem, "
                    "but not yet by the mixed geometry + initial-Er root optimizer. "
                    "Use the base geometry objective row in mixed optimization for now."
                )
            normalized.append(
                LeastSquaresTerm(
                    objective=term.objective,
                    target=term.target,
                    weight=term.weight,
                    label=term.label,
                )
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
            raise NotImplementedError(
                "Transformed geometry terms are supported by geometry_least_squares_problem, "
                "but not yet by the mixed geometry + initial-Er root optimizer. "
                "Use the base geometry objective row in mixed optimization for now."
            )
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
    config.setdefault("neoclassical", {})["ntx_exact_derivative_field_pullback_mode"] = "generic_jvp"
    if vmec_input is not None:
        config.setdefault("geometry", {})["vmec_input_file"] = str(vmec_input)
    return config


def _profile_values_from_config(config: dict, dtype) -> jnp.ndarray:
    profiles = config.get("profiles", {})
    values = []
    for name in PROFILE_PARAMETER_ORDER:
        raw = profiles[name]
        if isinstance(raw, (list, tuple)):
            raw = raw[0]
        values.append(float(raw))
    return jnp.asarray(values, dtype=dtype)


def geometry_initial_er_root_only_least_squares_problem(
    config,
    terms: Sequence[LeastSquaresTerm | tuple[ObjectiveRef | str, float, float]],
    *,
    vmec_input=None,
    max_mode: int | None = None,
    parameters: str | Sequence[str] | None = None,
    include_profiles: bool = False,
    profile_parameters: str | Sequence[str] | None = "n0,T0,density_shape_power,temperature_shape_power",
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
) -> GeometryInitialErRootLeastSquaresProblem:
    """Build an optimizer problem for geometry terms plus initial-Er root terms.

    The returned problem is intentionally thin and script-friendly: users define
    least-squares terms in the optimization script, while this helper owns the
    transport runtime setup and validated reverse-table calls.
    """

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
        if max_mode is None:
            raise ValueError("Either max_mode or explicit geometry parameters must be provided.")
        geometry_parameterization = vmex_boundary_parameterization(
            context,
            max_mode=int(max_mode),
            families=families,
            scale_mode=scale_mode,
            ess_alpha=float(ess_alpha),
        )
    profile_specs = (
        parse_profile_parameter_specs(profile_parameters)
        if include_profiles
        else ()
    )
    parameter_set = reverse_ad_optimization_parameter_set(
        include_profiles=bool(profile_specs),
        profiles=tuple(spec.name for spec in profile_specs) if profile_specs else None,
        vmec_boundary=geometry_parameterization.specs,
    )
    runtime, baseline_state = build_runtime_context(config_eff)
    if baseline_state is None:
        raise RuntimeError("transport runtime did not return an initial state.")
    baseline_profile_values = _profile_values_from_config(
        config_eff,
        jnp.asarray(baseline_state.pressure).dtype,
    )
    if geometry_max_iter is None:
        geometry_max_iter = geom_cfg.get("vmec_max_iter")
    if geometry_solver_device is None:
        geometry_solver_device = geom_cfg.get("vmec_implicit_solver_device", "default")
    normalized_terms = _normalize_initial_er_root_least_squares_terms(terms)
    return GeometryInitialErRootLeastSquaresProblem(
        config=config_eff,
        context=context,
        runtime=runtime,
        baseline_state=baseline_state,
        baseline_profile_values=baseline_profile_values,
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
    )


def least_squares(problem: GeometryLeastSquaresProblem, **kwargs):
    """Run SciPy least_squares on a NEOPAX geometry least-squares problem."""

    from scipy.optimize import least_squares as scipy_least_squares

    iteration_reporter = kwargs.pop("iteration_reporter", None)
    cache: dict[tuple[float, ...], LeastSquaresEvaluation] = {}
    verbose = int(kwargs.get("verbose", 0) or 0)
    state: dict[str, object] = {"nres": None, "npar": problem.parameter_count, "eval_count": 0}

    def _key(x):
        return tuple(np.asarray(x, dtype=float).tolist())

    def _evaluate(x):
        key = _key(x)
        evaluation = cache.get(key)
        if evaluation is None:
            evaluation = problem.evaluate(jnp.asarray(x, dtype=jnp.float64))
            cache.clear()
            cache[key] = evaluation
        return evaluation

    def _fun(x):
        try:
            residuals = np.asarray(jax.device_get(_evaluate(x).residuals), dtype=float)
        except Exception as exc:
            if state["nres"] is None:
                raise
            state["eval_count"] = int(state["eval_count"]) + 1
            if verbose:
                print(
                    f"[NEOPAX least_squares] eval={int(state['eval_count'])} "
                    f"trial solve failed -> penalty residual: {exc}",
                    flush=True,
                )
            return np.full((int(state["nres"]),), 1.0e6, dtype=float)
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
        try:
            jacobian = np.asarray(jax.device_get(_evaluate(x).jacobian), dtype=float)
        except Exception as exc:
            if state["nres"] is None:
                raise
            if verbose:
                print(
                    f"[NEOPAX least_squares] trial jacobian failed -> zero jacobian: {exc}",
                    flush=True,
                )
            return np.zeros((int(state["nres"]), int(state["npar"])), dtype=float)
        return np.where(np.isfinite(jacobian), jacobian, 0.0)

    x0 = np.asarray(jax.device_get(problem.x0), dtype=float)
    x_scale = np.asarray(jax.device_get(problem.x_scale), dtype=float)
    kwargs.setdefault("x_scale", x_scale)
    return scipy_least_squares(_fun, x0, jac=_jac, **kwargs)


__all__ = [
    "GeometryLeastSquaresTerm",
    "GeometryObjectiveTransform",
    "GeometryInitialErRootLeastSquaresProblem",
    "GeometryLeastSquaresProblem",
    "geometry",
    "geometry_objective",
    "geometry_initial_er_root_only_least_squares_problem",
    "geometry_least_squares_problem",
    "least_squares",
    "transformed_geometry_objective",
    "transport",
    "transport_objective",
]
