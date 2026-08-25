from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_NTX_EXACT_DERIVATIVE_MODE = "direct"
DEFAULT_NTX_EXACT_DERIVATIVE_FIELD_PULLBACK_MODE = "compact_vjp"
DEFAULT_NTX_EXACT_DERIVATIVE_PULLBACK_BOUNDARY = "inline"
DEFAULT_NTX_EXACT_DERIVATIVE_PULLBACK_ALGEBRA = "ntx_helper"


@contextlib.contextmanager
def _maybe_reverse_segment_profiler_trace(trace_dir: str | None):
    """Capture one benchmark evaluation for XProf without changing its math."""
    if trace_dir in (None, ""):
        yield
        return
    output_dir = Path(trace_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "[autodiff-gate] progress: starting reverse-segment XProf trace "
        f"directory={output_dir}",
        flush=True,
    )
    jax.profiler.start_trace(str(output_dir), create_perfetto_trace=True)
    try:
        yield
    finally:
        jax.profiler.stop_trace()
        print(
            "[autodiff-gate] progress: reverse-segment XProf trace written "
            f"directory={output_dir}",
            flush=True,
        )


def _apply_transport_solver_backend_override(config: dict, backend_override: str | None) -> None:
    """Switch the benchmark TOML between solver backends without editing the file."""
    if backend_override in (None, "", "config"):
        return
    backend = str(backend_override).strip().lower()
    solver_cfg = config.setdefault("transport_solver", {})
    solver_cfg["transport_solver_backend"] = backend
    solver_cfg["integrator"] = backend
    if backend in {"theta", "theta_newton"}:
        solver_cfg.setdefault("theta_implicit", 1.0)
        solver_cfg.setdefault("theta_predictor_mode", "linearized")
        solver_cfg.setdefault(
            "theta_rhs_mode",
            solver_cfg.get("radau_rhs_mode", solver_cfg.get("rhs_mode", "lagged_response")),
        )
        if backend == "theta_newton":
            solver_cfg.setdefault("theta_controller_mode", "current")
            solver_cfg.setdefault("theta_jacobian_reuse_mode", "refresh_each_iteration")
            solver_cfg.setdefault(
                "theta_lagged_response_reuse_mode",
                solver_cfg.get("lagged_response_reuse_mode", "retry_only"),
            )
            solver_cfg.setdefault(
                "theta_lagged_response_reuse_rtol",
                solver_cfg.get("lagged_response_reuse_rtol", 5.0e-2),
            )
            solver_cfg.setdefault(
                "theta_lagged_response_reuse_atol",
                solver_cfg.get("lagged_response_reuse_atol", 1.0e-8),
            )


def _run_transport_solver_forward_smoke(*, args, solver, solve_vector_field, runtime, baseline_state) -> None:
    phase_start = time.perf_counter()
    print(
        "[autodiff-gate] progress: running transport solver forward smoke "
        f"solver={type(solver).__name__}",
        flush=True,
    )
    result = solver.solve(baseline_state, solve_vector_field, runtime.species)
    final_state = result["final_state"]
    objective_values = _objective_vector(final_state, runtime)
    jax.block_until_ready(objective_values)
    objective_values_np = np.asarray(jax.device_get(objective_values), dtype=float)
    print(
        "[autodiff-gate] mode=transport_solver_forward_smoke "
        f"solver={type(solver).__name__} objective=all "
        f"elapsed_s={time.perf_counter() - phase_start:.3f}",
        flush=True,
    )
    print("[autodiff-gate] forward-smoke objective values:", flush=True)
    for label, value in zip(OBJECTIVE_LABELS, objective_values_np, strict=False):
        print(f"  - {label}: value={value:.16e}", flush=True)
    if "n_steps" in result:
        print(
            "[autodiff-gate] forward-smoke solver summary: "
            f"n_steps={int(np.asarray(jax.device_get(result['n_steps'])))} "
            f"done={bool(np.asarray(jax.device_get(result.get('done', False))))} "
            f"failed={bool(np.asarray(jax.device_get(result.get('failed', False))))} "
            f"fail_code={int(np.asarray(jax.device_get(result.get('fail_code', 0))))}",
            flush=True,
        )


from benchmark_transport_forward_fd_lane import (  # noqa: E402
    DEFAULT_CONFIG,
    OBJECTIVE_LABELS,
    _adaptive_rollout_diagnostics,
    _alpha_power_volume_average,
    _baseline_profile_cfg,
    _bootstrap_current_softmax_abs_scaled,
    _electron_temperature_volume_average,
    _objective_vector,
    _parameterized_profile_set,
    _prepare_benchmark_config,
    _smooth_root_proxy,
    _softmax_objective,
    _total_pressure_volume_average,
    _volume_average,
)
from NEOPAX._geometry_autodiff import (  # noqa: E402
    boundary_param_entries,
    build_geometry_autodiff_context,
)
from NEOPAX._orchestrator import build_runtime_context  # noqa: E402
from NEOPAX._orchestrator import prepare_transport_solver_components  # noqa: E402
from NEOPAX._reverse_ad_initial_er import (  # noqa: E402
    compact_initial_er_state_pullback,
    compact_initial_er_ntx_support_pullback_leaves,
    find_ntx_exact_support_model,
    find_ntx_support_payload as _core_find_ntx_support_payload,
    initial_er_charge_flux_residual_er_derivative as _core_initial_er_charge_flux_residual_er_derivative,
    initial_er_charge_flux_residual_scalar as _core_initial_er_charge_flux_residual_scalar,
    initial_er_charge_flux_residuals as _core_initial_er_charge_flux_residuals,
    initial_er_root_setup as _core_initial_er_root_setup,
    initial_er_selected_root_profile as _core_initial_er_selected_root_profile,
    runtime_with_geometry_payload as _core_runtime_with_geometry_payload,
    runtime_with_ntx_support_payload as _core_runtime_with_ntx_support_payload,
)
from NEOPAX._reverse_ad_parameters import (  # noqa: E402
    PROFILE_PARAMETER_ORDER,
    VmecBoundaryParameterSpec,
    discover_vmec_boundary_parameter_specs,
    normalize_vmec_boundary_families,
    reverse_ad_optimization_parameter_set,
    reverse_ad_parameter_set,
    vmec_boundary_tuples,
)
from NEOPAX._reverse_ad_optimization import (  # noqa: E402
    build_initial_er_root_only_least_squares_runner,
    build_transport_realtime_geometry_least_squares_runner,
    evaluate_geometry_transport_realtime_geometry_least_squares,
    evaluate_geometry_initial_er_root_only_least_squares_benchmark_tables,
    geometry as geometry_objectives,
    geometry_active_initial_er_root_only_reverse_table,
    LeastSquaresEvaluation,
    INITIAL_ER_ROOT_ONLY_EXPLICIT_OBJECTIVES,
    INITIAL_ER_ROOT_ONLY_OBJECTIVES,
    residuals_and_jacobian_reverse_ad,
    LeastSquaresTerm,
    transport_least_squares_terms,
)
from NEOPAX._reverse_ad_transport import (  # noqa: E402
    TRANSPORT_REVERSE_OBJECTIVE_LABELS,
    grouped_transport_reverse_report_builder,
    grouped_transport_reverse_table_result_builder,
    internal_realtime_geometry_transport_reverse_table_result_builder,
    prepare_realtime_geometry_support_segment_core_setup,
    realtime_geometry_transport_reverse_table_request,
    RealtimeGeometrySupportReverseDependencies,
    realtime_geometry_reverse_all_objectives_support_payload_bar_for_parameter_vector,
    realtime_geometry_support_cotangents_from_parameter_vector,
    realtime_geometry_transport_reverse_table_from_payload_cotangents,
    realtime_geometry_transport_reverse_grouped_inputs,
    realtime_geometry_transport_reverse_support_segment_executor,
    realtime_geometry_transport_reverse_diagnostic_gradient_entries,
    realtime_geometry_transport_reverse_metadata_entries,
    reverse_initial_carry_from_state_with_static_setup as _core_reverse_initial_carry_from_state_with_static_setup,
    transport_reverse_table_report_entries,
)
from NEOPAX._transport_flux_models import (  # noqa: E402
    _add_float_delta_tree,
    _float_delta_tree_like,
    _sanitize_float_delta_bar_tree,
)
from NEOPAX._transport_solvers import (  # noqa: E402
    _RadauAcceptedStepReducedCotangent,
    _build_prepared_radau_accepted_rollout,
    _build_prepared_radau_execution_context,
    _extract_fixed_temperature_projection,
    _extract_state_regularization,
    _make_radau_initial_step_state,
    _make_solver_state_transform,
    _project_flat_state_if_needed,
    _radau_adaptive_final_state_rollout,
    _radau_adaptive_final_y_realized_schedule_vjp,
    _radau_adaptive_final_y_realized_schedule_vjp_bwd,
    _radau_adaptive_final_y_realized_schedule_vjp_fwd,
    _radau_adaptive_schedule_rollout,
    _radau_align_tangent_tree_to_primal,
    _radau_carry_from_step_state,
    _radau_debug_local_accepted_step_transpose,
    _radau_debug_local_stage_transpose_matvec,
    _radau_eval_rhs,
    _radau_add_support_delta_trees,
    _radau_replay_realized_accepted_slot,
    _radau_segment_reduced_cotangent_bwd_batched_call,
    _radau_segment_reduced_cotangent_bwd_batched_with_support_call,
    _radau_segment_reduced_cotangent_bwd_call,
    _radau_segment_reduced_cotangent_bwd_with_support_call,
    _radau_single_slot_support_cotangent_bwd_call,
    _radau_zero_support_delta_tree_like,
)


PARAMETER_ORDER = PROFILE_PARAMETER_ORDER
_PROFILE_PARAMETER_DEFAULTS = {
    "n0": 4.21,
    "T0": 17.8,
    "density_shape_power": 2.0,
    "temperature_shape_power": 2.0,
    "density_shape_alpha": 1.0,
    "temperature_shape_alpha": 1.0,
}
_REALTIME_GEOMETRY_BACKENDS = {"vmec_jax_booz_xform_jax", "vmec_runtime", "vmec_realtime"}


def _profile_cfg_scalar_value(profile_cfg: dict[str, Any], name: str) -> float:
    raw = profile_cfg.get(name, _PROFILE_PARAMETER_DEFAULTS[name])
    if isinstance(raw, (list, tuple)):
        raw = raw[0]
    return float(raw)


def _initial_er_root_ad_mode(value: str | None) -> str:
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
        raise ValueError("--initial-Er-root-ad must be one of: off, jax_selected_root")
    return mode


def _initial_er_root_enabled(config: dict[str, Any], mode: str) -> bool:
    mode = _initial_er_root_ad_mode(mode)
    if mode == "off":
        return False
    profiles_cfg = config.get("profiles", {})
    init_mode = str(profiles_cfg.get("er_initialization_mode", "analytical")).strip().lower()
    return init_mode in {
        "ambipolar_min_entropy",
        "ambipolar_best_root",
        "ambipolarity_best_root",
    }


def _initial_er_root_setup(config: dict[str, Any], runtime):
    return _core_initial_er_root_setup(config, runtime)


def _initial_er_selected_root_profile(state, *, config: dict[str, Any], runtime):
    return _core_initial_er_selected_root_profile(state, config=config, runtime=runtime)


def _initial_er_charge_flux_residuals(state, er_profile, *, runtime):
    return _core_initial_er_charge_flux_residuals(state, er_profile, runtime=runtime)


def _initial_er_charge_flux_residual_scalar(state, er_profile, radius_index, *, runtime):
    return _core_initial_er_charge_flux_residual_scalar(
        state,
        er_profile,
        radius_index,
        runtime=runtime,
    )


def _initial_er_charge_flux_residual_er_derivative(state, er_profile, *, runtime):
    return _core_initial_er_charge_flux_residual_er_derivative(
        state,
        er_profile,
        runtime=runtime,
    )


def _initial_er_residual_bar(state, er_profile, er_bar, finite_mask, *, runtime):
    er_profile = jnp.asarray(er_profile, dtype=state.Er.dtype)
    er_bar = jnp.asarray(er_bar, dtype=state.Er.dtype)
    finite_mask = jnp.asarray(finite_mask, dtype=bool)

    dres_der = _initial_er_charge_flux_residual_er_derivative(
        state,
        er_profile,
        runtime=runtime,
    )
    safe_dres_der = jnp.where(
        jnp.abs(dres_der) > jnp.asarray(1.0e-30, dtype=dres_der.dtype),
        dres_der,
        jnp.inf,
    )
    return jnp.where(finite_mask, -er_bar / safe_dres_der, 0.0)


def _initial_er_root_state_bar(state, er_profile, finite_mask, state_bar, *, runtime):
    residual_bar = _initial_er_residual_bar(
        state,
        er_profile,
        state_bar.Er,
        finite_mask,
        runtime=runtime,
    )
    state_residual_bar = compact_initial_er_state_pullback(
        residual_scalar_fn=_initial_er_charge_flux_residual_scalar,
        state=state,
        er_profile=er_profile,
        residual_bars=residual_bar,
        runtime=runtime,
    )
    direct_bar = dataclasses.replace(state_bar, Er=jnp.zeros_like(state.Er))
    return _add_trees(direct_bar, state_residual_bar)


def _state_with_initial_er_root_ad(state, *, config: dict[str, Any], runtime, mode: str):
    if not _initial_er_root_enabled(config, mode):
        return state

    @jax.custom_vjp
    def _replace_er_with_selected_root(state_inner):
        er_profile, _ = _initial_er_selected_root_profile(state_inner, config=config, runtime=runtime)
        return dataclasses.replace(state_inner, Er=er_profile)

    def _replace_er_fwd(state_inner):
        er_profile, finite_mask = _initial_er_selected_root_profile(
            state_inner,
            config=config,
            runtime=runtime,
        )
        return dataclasses.replace(state_inner, Er=er_profile), (state_inner, er_profile, finite_mask)

    def _replace_er_bwd(residuals, state_bar):
        state_inner, er_profile, finite_mask = residuals
        return (
            _initial_er_root_state_bar(
                state_inner,
                er_profile,
                finite_mask,
                state_bar,
                runtime=runtime,
            ),
        )

    _replace_er_with_selected_root.defvjp(_replace_er_fwd, _replace_er_bwd)
    return _replace_er_with_selected_root(state)


def _objective_scalar_by_index(final_state, runtime, objective_index: int):
    """Evaluate one objective without constructing unrelated possibly-nonfinite objectives."""

    objective_name = OBJECTIVE_LABELS[int(objective_index)]
    er = jnp.asarray(final_state.Er)
    if objective_name == "softmax_Er":
        return _softmax_objective(er)
    if objective_name == "smooth_root_proxy":
        rho = jnp.asarray(runtime.geometry.rho_grid, dtype=er.dtype)
        return _smooth_root_proxy(er, rho)
    if objective_name == "Er_transition_left":
        return er[max(0, min(20, int(er.shape[-1]) - 1))]
    if objective_name == "Er_transition_right":
        return er[max(0, min(21, int(er.shape[-1]) - 1))]
    if objective_name == "Er2_volume_average":
        return _volume_average(er * er, runtime.geometry)
    if objective_name == "Er_volume_average":
        return _volume_average(er, runtime.geometry)
    if objective_name == "electron_temperature_volume_average_keV":
        return _electron_temperature_volume_average(final_state, runtime)
    if objective_name == "total_pressure_volume_average":
        return _total_pressure_volume_average(final_state, runtime)
    if objective_name == "alpha_power_volume_average_mw_m3":
        return _alpha_power_volume_average(final_state, runtime)
    if objective_name == "bootstrap_current_softmax_abs_scaled":
        return _bootstrap_current_softmax_abs_scaled(final_state, runtime)
    raise ValueError(f"Unknown objective index {objective_index}: {objective_name!r}")


def _benchmark_device_context(config: dict[str, Any]):
    general_cfg = config.get("general", {})
    device = str(general_cfg.get("device", "auto")).strip().lower()
    if device in {"", "none", "null", "auto"}:
        return contextlib.nullcontext()
    if device not in {"cpu", "gpu"}:
        raise ValueError("general.device must be one of: auto, cpu, gpu")
    try:
        devices = jax.local_devices(backend=device)
    except Exception as exc:
        available = sorted({local_device.platform for local_device in jax.local_devices()})
        raise ValueError(
            f"Requested general.device={device!r}, but JAX could not query that backend. "
            f"Available local platforms: {available}"
        ) from exc
    if not devices:
        available = sorted({local_device.platform for local_device in jax.local_devices()})
        raise ValueError(
            f"Requested general.device={device!r}, but no local JAX devices were found. "
            f"Available local platforms: {available}"
        )
    return jax.default_device(devices[0])


def _array_device_summary(value) -> str:
    try:
        return str(value.device)
    except Exception:
        try:
            return ",".join(str(device) for device in value.devices())
        except Exception:
            return "unknown"


@dataclasses.dataclass(frozen=True)
class _ReverseStaticSetup:
    solver: object
    solve_vector_field: object
    prepared_rollout: object
    execution_context: object
    stop_after_accepted_steps: int | None
    max_total_steps: int
    reverse_segment_length: int | None
    # Scalar schedule arrays only; no extra carry or per-step tape.
    schedule_artifact: object | None = None


def _replace_ntx_support_payload_in_model(model, support):
    with_support_payload = getattr(model, "with_support_payload", None)
    if callable(with_support_payload):
        return with_support_payload(support), True
    if not dataclasses.is_dataclass(model) or isinstance(model, type):
        return model, False
    updates = {}
    changed = False
    for field in dataclasses.fields(model):
        value = getattr(model, field.name)
        new_value, child_changed = _replace_ntx_support_payload_in_model(value, support)
        if child_changed:
            updates[field.name] = new_value
            changed = True
    if not changed:
        return model, False
    return dataclasses.replace(model, **updates), True


def _runtime_with_ntx_support_payload(runtime, support):
    return _core_runtime_with_ntx_support_payload(runtime, support)


def _replace_geometry_payload_in_model(model, geometry):
    if model is None or not dataclasses.is_dataclass(model) or isinstance(model, type):
        return model, False
    updates = {}
    changed = False
    for field in dataclasses.fields(model):
        value = getattr(model, field.name)
        if field.name in {"geometry", "field"}:
            if value is not geometry:
                updates[field.name] = geometry
                changed = True
            continue
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            new_value, child_changed = _replace_geometry_payload_in_model(value, geometry)
            if child_changed:
                updates[field.name] = new_value
                changed = True
    if not changed:
        return model, False
    return dataclasses.replace(model, **updates), True


def _runtime_with_geometry_payload(runtime, geometry):
    return _core_runtime_with_geometry_payload(runtime, geometry)


def _find_ntx_support_payload_in_model(model):
    support = getattr(model, "support", None)
    if support is not None and hasattr(model, "with_support_payload"):
        return support
    if dataclasses.is_dataclass(model) and not isinstance(model, type):
        for field in dataclasses.fields(model):
            found = _find_ntx_support_payload_in_model(getattr(model, field.name))
            if found is not None:
                return found
    return None


def _find_ntx_support_payload(runtime):
    return _core_find_ntx_support_payload(runtime)


def _find_ntx_exact_support_model(model):
    return find_ntx_exact_support_model(model)


def _compact_initial_er_ntx_support_pullback_leaves(
    *,
    runtime,
    state,
    er_profile,
    residual_bars,
    support,
):
    return compact_initial_er_ntx_support_pullback_leaves(
        runtime=runtime,
        state=state,
        er_profile=er_profile,
        residual_bars=residual_bars,
        support=support,
    )


def _payload_leaf_summary(payload) -> dict[str, Any]:
    leaves = jax.tree_util.tree_leaves(payload)
    array_leaves = [jnp.asarray(leaf) for leaf in leaves if hasattr(leaf, "shape")]
    finite_leaves = [
        bool(jnp.all(jnp.isfinite(leaf)))
        for leaf in array_leaves
        if jnp.issubdtype(leaf.dtype, jnp.inexact)
    ]
    total_bytes = int(
        sum(int(leaf.size) * int(leaf.dtype.itemsize) for leaf in array_leaves)
    )
    return {
        "n_leaves": int(len(leaves)),
        "n_array_leaves": int(len(array_leaves)),
        "total_array_bytes": total_bytes,
        "all_floating_leaves_finite": bool(all(finite_leaves)) if finite_leaves else True,
        "first_array_leaves": [
            {
                "shape": list(leaf.shape),
                "dtype": str(leaf.dtype),
            }
            for leaf in array_leaves[:12]
        ],
    }


def _tree_path_label(path) -> str:
    parts: list[str] = []
    for entry in path:
        key = getattr(entry, "key", None)
        name = getattr(entry, "name", None)
        idx = getattr(entry, "idx", None)
        if key is not None:
            parts.append(str(key))
        elif name is not None:
            parts.append(str(name))
        elif idx is not None:
            parts.append(str(idx))
        else:
            parts.append(str(entry))
    return ".".join(parts) if parts else "<root>"


def _payload_nonfinite_leaf_summaries(payload, *, limit: int = 8) -> list[dict[str, Any]]:
    entries = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(payload)[0]:
        if not hasattr(leaf, "shape"):
            continue
        arr_jax = jnp.asarray(leaf)
        if not jnp.issubdtype(arr_jax.dtype, jnp.inexact):
            continue
        arr = np.asarray(jax.device_get(arr_jax))
        finite = np.isfinite(arr)
        if bool(np.all(finite)):
            continue
        finite_values = arr[finite]
        first_index = [int(i) for i in np.argwhere(~finite)[0].tolist()]
        entries.append(
            {
                "path": _tree_path_label(path),
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "nan_count": int(np.isnan(arr).sum()),
                "posinf_count": int(np.isposinf(arr).sum()),
                "neginf_count": int(np.isneginf(arr).sum()),
                "finite_min": None if finite_values.size == 0 else float(np.min(finite_values)),
                "finite_max": None if finite_values.size == 0 else float(np.max(finite_values)),
                "first_nonfinite_index": first_index,
            }
        )
        if len(entries) >= int(limit):
            break
    return entries


def _payload_branch_diagnostics(payload) -> dict[str, Any]:
    def _branch(tree) -> dict[str, Any]:
        return {
            "l2": _tree_array_l2_norm(tree),
            "summary": _payload_leaf_summary(tree),
            "first_nonfinite_leaves": _payload_nonfinite_leaf_summaries(tree),
        }

    diagnostics: dict[str, Any] = {"root": _branch(payload)}
    ntx_payload = payload
    if isinstance(payload, dict):
        for branch_name in ("geometry", "ntx_support"):
            if branch_name in payload:
                diagnostics[branch_name] = _branch(payload[branch_name])
        if "ntx_support" in payload:
            ntx_payload = payload["ntx_support"]
    for branch_name in (
        "center_channels",
        "face_channels",
        "center_prepared",
        "face_prepared",
        "center_surfaces",
        "face_surfaces",
    ):
        if hasattr(ntx_payload, branch_name):
            diagnostics[f"ntx_support.{branch_name}"] = _branch(
                getattr(ntx_payload, branch_name)
            )
    return diagnostics


def _tree_array_l2_norm(payload) -> float:
    leaves = jax.tree_util.tree_leaves(payload)
    total = jnp.asarray(0.0, dtype=jnp.float64)
    for leaf in leaves:
        if not hasattr(leaf, "shape"):
            continue
        arr = jnp.asarray(leaf)
        if jnp.issubdtype(arr.dtype, jnp.inexact):
            total = total + jnp.sum(arr.astype(jnp.float64) * arr.astype(jnp.float64))
    return float(np.asarray(jax.device_get(jnp.sqrt(total))))


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


def _geometry_volume_diagnostics(geometry) -> dict[str, Any]:
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


def _report_path(objective_name: str) -> Path:
    outdir = ROOT / "outputs" / "autodiff_transport_lagged_ntx" / "reverse_ad"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / f"transport_reverse_ad_only_{objective_name}.json"


def _check_compact_ntx_derivative_pullback_available() -> None:
    try:
        from ntx._solver_prepared import solve_prepared_coefficient_vector_derivative_vjp  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "[autodiff-gate] the default compact NTX field pullback requires "
            "the matching NTX patch/export: "
            "solve_prepared_coefficient_vector_derivative_vjp. Sync/apply the NTX "
            "changes before running this mode."
        ) from exc


def _initial_state_for_parameter_vector(
    parameter_values,
    *,
    baseline_state,
    profile_cfg: dict,
    runtime,
    config: dict[str, Any] | None = None,
    initial_er_root_ad: str = "off",
):
    cfg = dict(profile_cfg)
    for name, value in zip(PARAMETER_ORDER, parameter_values):
        cfg[name] = value
    profile_set = _parameterized_profile_set(
        cfg,
        runtime.geometry,
        runtime.species.number_species,
        parameter_name=PARAMETER_ORDER[0],
        parameter_value=cfg[PARAMETER_ORDER[0]],
    )
    density_state = jnp.asarray(profile_set.density, dtype=baseline_state.density.dtype) / 1.0e20
    temperature_state = jnp.asarray(profile_set.temperature, dtype=baseline_state.pressure.dtype) / 1.0e3
    pressure_state = density_state * temperature_state
    state = dataclasses.replace(
        baseline_state,
        density=density_state,
        pressure=pressure_state,
    )
    mode = _initial_er_root_ad_mode(initial_er_root_ad)
    if mode != "off":
        if config is None:
            raise ValueError("config is required when initial_er_root_ad is enabled.")
        state = _state_with_initial_er_root_ad(state, config=config, runtime=runtime, mode=mode)
    return state


def _parse_reverse_geometry_parameter(parameter_name: str) -> tuple[str, int, int]:
    parts = str(parameter_name).split(":")
    if len(parts) != 3:
        raise ValueError(
            "Reverse geometry parameters must use the syntax 'FAMILY:m:n', "
            "for example 'RBC:1:0'."
        )
    family = parts[0].strip().upper()
    try:
        m = int(parts[1])
        n = int(parts[2])
    except ValueError as exc:
        raise ValueError(
            "Reverse geometry parameters must use integer m/n values, "
            "for example 'RBC:1:0'."
        ) from exc
    if not family:
        raise ValueError("Reverse geometry parameter family cannot be empty.")
    return family, m, n


def _format_reverse_geometry_parameter(parameter_name: str) -> str:
    family, m, n = _parse_reverse_geometry_parameter(parameter_name)
    return f"vmec:{family}:{m}:{n}"


def _format_geometry_param_spec(param_spec: tuple[str, int, int]) -> str:
    family, m, n = param_spec
    return f"vmec:{str(family).strip().upper()}:{int(m)}:{int(n)}"


def _parse_reverse_geometry_families(value: str | None) -> tuple[str, ...]:
    try:
        return normalize_vmec_boundary_families(value)
    except ValueError as exc:
        raise ValueError(f"--reverse-geometry-families {exc}") from exc


def _geometry_context_from_config(config: dict[str, Any], geometry_parameter: str):
    if str(geometry_parameter).strip().lower() == "all":
        family, m, n = ("RBC", 0, 0)
    else:
        first_geometry_parameter = str(geometry_parameter).split(",", 1)[0].strip()
        family, m, n = _parse_reverse_geometry_parameter(first_geometry_parameter)
    geom_cfg = config.get("geometry", {})
    backend = str(geom_cfg.get("backend", "")).strip().lower()
    if backend not in _REALTIME_GEOMETRY_BACKENDS:
        raise ValueError(
            "Realtime geometry reverse parameters require geometry.backend to be one of "
            f"{sorted(_REALTIME_GEOMETRY_BACKENDS)}; got backend={backend!r}."
        )
    vmec_input_file = geom_cfg.get("vmec_input_file")
    if vmec_input_file is None:
        raise ValueError("Realtime geometry reverse mode requires geometry.vmec_input_file.")
    surface_s_raw = geom_cfg.get(
        "surface_s",
        geom_cfg.get("vmec_surface_s", (0.1, 0.28, 0.46, 0.64, 0.82, 1.0)),
    )
    if isinstance(surface_s_raw, str):
        surface_s = tuple(float(item.strip()) for item in surface_s_raw.split(",") if item.strip())
    else:
        surface_s = tuple(float(item) for item in surface_s_raw)
    return build_geometry_autodiff_context(
        vmec_input_file,
        param_family=family,
        param_m=m,
        param_n=n,
        mboz=int(geom_cfg.get("mboz", geom_cfg.get("vmec_mboz", 18))),
        nboz=int(geom_cfg.get("nboz", geom_cfg.get("vmec_nboz", 18))),
        surface_s=surface_s,
    )


def _reverse_geometry_parameter_order(geometry_parameter: str) -> tuple[str, ...]:
    if str(geometry_parameter).strip().lower() == "all":
        return (*PARAMETER_ORDER, "vmec:all")
    return (
        *PARAMETER_ORDER,
        *(
            _format_reverse_geometry_parameter(raw_parameter.strip())
            for raw_parameter in str(geometry_parameter).split(",")
            if raw_parameter.strip()
        ),
    )


def _geometry_param_specs_from_parameter_name(geometry_parameter: str) -> tuple[tuple[str, int, int], ...]:
    specs = tuple(
        _parse_reverse_geometry_parameter(raw_parameter.strip())
        for raw_parameter in str(geometry_parameter).split(",")
        if raw_parameter.strip()
    )
    if not specs:
        raise ValueError("At least one reverse geometry parameter must be provided.")
    return specs


def _all_geometry_param_specs_from_context(
    geometry_context,
    *,
    families: tuple[str, ...],
    nonzero_only: bool = True,
) -> tuple[tuple[str, int, int], ...]:
    try:
        specs = discover_vmec_boundary_parameter_specs(
            geometry_context,
            families=families,
            nonzero_only=nonzero_only,
        )
    except ValueError as exc:
        raise ValueError(
            "No VMEC boundary harmonics matched the requested all-harmonic selector. "
            "Try --reverse-geometry-include-zero-harmonics or a different "
            "--reverse-geometry-families value."
        ) from exc
    return vmec_boundary_tuples(specs)


def _geometry_param_specs_from_args(args, geometry_context) -> tuple[tuple[str, int, int], ...]:
    geometry_parameter = str(args.reverse_geometry_parameter).strip()
    if geometry_parameter.lower() != "all":
        return _geometry_param_specs_from_parameter_name(geometry_parameter)
    return _all_geometry_param_specs_from_context(
        geometry_context,
        families=_parse_reverse_geometry_families(args.reverse_geometry_families),
        nonzero_only=not bool(args.reverse_geometry_include_zero_harmonics),
    )


def _baseline_geometry_delta_vector_for_specs(
    geom_cfg: dict[str, Any],
    geometry_param_specs: Sequence[tuple[str, int, int]],
) -> jnp.ndarray:
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


def _add_trees(lhs, rhs):
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs
    return jax.tree_util.tree_map(lambda a, b: a + b, lhs, rhs)


def _lagged_response_pullback_from_owner(solve_vector_field):
    owner = getattr(solve_vector_field, "__self__", None)
    if owner is None:
        return None
    pullback_fn = getattr(owner, "pullback_build_lagged_response", None)
    return pullback_fn if callable(pullback_fn) else None


def _reverse_initial_carry_from_state_with_static_setup(
    *,
    solver,
    state,
    solve_vector_field,
    species,
    prepared_rollout_static,
    return_native_joint_pullback: bool = False,
):
    """Build the initial carry with a reverse-local model-aware lagged pullback."""

    # Preserve the benchmark's legacy local custom-VJP exactly for every
    # established selector.  Only the new explicit opt-in uses the core
    # adapter, which exposes the joint state/support result unavailable from
    # a normal ``jax.vjp`` closure over state alone.
    if return_native_joint_pullback:
        return _core_reverse_initial_carry_from_state_with_static_setup(
            solver=solver,
            state=state,
            solve_vector_field=solve_vector_field,
            species=species,
            prepared_rollout_static=prepared_rollout_static,
            return_native_joint_pullback=True,
        )

    temperature_active_mask, fixed_temperature_profile = _extract_fixed_temperature_projection(solve_vector_field)
    density_floor, temperature_floor = _extract_state_regularization(solve_vector_field)
    kernel_context = prepared_rollout_static.kernel_context
    physics_context = prepared_rollout_static.physics_context
    initial_carry_static = prepared_rollout_static.initial_carry
    lagged_pullback_fn = _lagged_response_pullback_from_owner(solve_vector_field)

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
            lagged_bar = _add_trees(lagged_bar, rhs_lagged_bar)

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


def _reverse_objective_for_parameter_vector(
    parameter_values,
    *,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict,
    objective_index: int,
    reverse_setup: _ReverseStaticSetup,
    initial_er_root_ad: str = "off",
):
    state0 = _initial_state_for_parameter_vector(
        parameter_values,
        config=config,
        initial_er_root_ad=initial_er_root_ad,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        runtime=runtime,
    )
    initial_carry = _reverse_initial_carry_from_state_with_static_setup(
        solver=reverse_setup.solver,
        state=state0,
        solve_vector_field=reverse_setup.solve_vector_field,
        species=runtime.species,
        prepared_rollout_static=reverse_setup.prepared_rollout,
    )
    final_y = _radau_adaptive_final_y_realized_schedule_vjp(
        reverse_setup.execution_context,
        reverse_setup.max_total_steps,
        reverse_setup.stop_after_accepted_steps,
        reverse_setup.reverse_segment_length,
        initial_carry,
    )
    final_state = reverse_setup.prepared_rollout.physics_context.unpack_flat(final_y)
    return _objective_scalar_by_index(final_state, runtime, objective_index)


def _reverse_objective_vector_for_parameter_vector(
    parameter_values,
    *,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict,
    reverse_setup: _ReverseStaticSetup,
    initial_er_root_ad: str = "off",
):
    state0 = _initial_state_for_parameter_vector(
        parameter_values,
        config=config,
        initial_er_root_ad=initial_er_root_ad,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        runtime=runtime,
    )
    initial_carry = _reverse_initial_carry_from_state_with_static_setup(
        solver=reverse_setup.solver,
        state=state0,
        solve_vector_field=reverse_setup.solve_vector_field,
        species=runtime.species,
        prepared_rollout_static=reverse_setup.prepared_rollout,
    )
    final_y = _radau_adaptive_final_y_realized_schedule_vjp(
        reverse_setup.execution_context,
        reverse_setup.max_total_steps,
        reverse_setup.stop_after_accepted_steps,
        reverse_setup.reverse_segment_length,
        initial_carry,
    )
    final_state = reverse_setup.prepared_rollout.physics_context.unpack_flat(final_y)
    return _objective_vector(final_state, runtime)


def _reverse_final_y_for_parameter_vector(
    parameter_values,
    *,
    runtime,
    baseline_state,
    profile_cfg: dict,
    reverse_setup: _ReverseStaticSetup,
):
    state0 = _initial_state_for_parameter_vector(
        parameter_values,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        runtime=runtime,
    )
    initial_carry = _reverse_initial_carry_from_state_with_static_setup(
        solver=reverse_setup.solver,
        state=state0,
        solve_vector_field=reverse_setup.solve_vector_field,
        species=runtime.species,
        prepared_rollout_static=reverse_setup.prepared_rollout,
    )
    return _radau_adaptive_final_y_realized_schedule_vjp(
        reverse_setup.execution_context,
        reverse_setup.max_total_steps,
        reverse_setup.stop_after_accepted_steps,
        reverse_setup.reverse_segment_length,
        initial_carry,
    )


def _reverse_all_objectives_vmap_pullback_for_parameter_vector(
    parameter_values,
    *,
    runtime,
    baseline_state,
    profile_cfg: dict,
    reverse_setup: _ReverseStaticSetup,
):
    """Compute all objective rows by batching final-y cotangent pullbacks."""

    def _final_y_from_parameters(parameter_values_value):
        return _reverse_final_y_for_parameter_vector(
            parameter_values_value,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            reverse_setup=reverse_setup,
        )

    final_y, final_y_pullback = jax.vjp(_final_y_from_parameters, parameter_values)

    def _objective_vector_from_final_y(final_y_value):
        final_state = reverse_setup.prepared_rollout.physics_context.unpack_flat(final_y_value)
        return _objective_vector(final_state, runtime)

    objective_values, objective_pullback = jax.vjp(_objective_vector_from_final_y, final_y)
    objective_basis = jnp.eye(len(OBJECTIVE_LABELS), dtype=jnp.asarray(objective_values).dtype)
    final_y_bars = jax.vmap(lambda basis: objective_pullback(basis)[0])(objective_basis)
    gradient_matrix = jax.vmap(lambda final_y_bar: final_y_pullback(final_y_bar)[0])(final_y_bars)
    return objective_values, gradient_matrix


def _reverse_all_objectives_multi_rhs_reduced_for_parameter_vector(
    parameter_values,
    *,
    runtime,
    baseline_state,
    profile_cfg: dict,
    reverse_setup: _ReverseStaticSetup,
):
    """Compute all objective rows with a shared segmented reduced reverse replay."""

    if reverse_setup.reverse_segment_length is None or int(reverse_setup.reverse_segment_length) <= 0:
        raise ValueError("multi_rhs_reduced requires --reverse-segment-length.")
    step_bwd_mode = str(
        getattr(reverse_setup.execution_context.physics_context, "reverse_step_bwd_mode", "current")
    ).strip().lower()
    if step_bwd_mode not in {
        "reduced_cotangent",
        "reduced_cotangent_call_boundary",
        "reduced_cotangent_lean_replay",
        "reduced_cotangent_recompute_replay",
        "lean_replay",
        "recompute_replay",
        "reduced",
        "state_only",
        "final_state",
    }:
        raise ValueError("multi_rhs_reduced requires a reduced-cotangent reverse step bwd mode.")

    def _zero_tangent_like(x):
        arr = jnp.asarray(x)
        if jnp.issubdtype(arr.dtype, jnp.inexact):
            return jnp.zeros_like(arr)
        return jnp.zeros(arr.shape, dtype=jax.dtypes.float0)

    def _take_tree_axis0(tree, index: int):
        return jax.tree_util.tree_map(lambda value: value[index], tree)

    def _carry_from_parameters(p):
        return _reverse_initial_carry_for_parameter_vector(
            p,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            reverse_setup=reverse_setup,
        )

    initial_carry, initial_carry_pullback = jax.vjp(_carry_from_parameters, parameter_values)
    final_y, residuals = _radau_adaptive_final_y_realized_schedule_vjp_fwd(
        reverse_setup.execution_context,
        reverse_setup.max_total_steps,
        reverse_setup.stop_after_accepted_steps,
        reverse_setup.reverse_segment_length,
        initial_carry,
    )

    def _objective_vector_from_final_y(final_y_value):
        final_state = reverse_setup.prepared_rollout.physics_context.unpack_flat(final_y_value)
        return _objective_vector(final_state, runtime)

    objective_values, objective_pullback = jax.vjp(_objective_vector_from_final_y, final_y)
    objective_basis = jnp.eye(len(OBJECTIVE_LABELS), dtype=jnp.asarray(objective_values).dtype)
    final_y_bars = jax.vmap(lambda basis: objective_pullback(basis)[0])(objective_basis)

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
    if segment_start_carries is None or segmented_final_carry is None or segmented_replay_arrays is None:
        raise ValueError("multi_rhs_reduced requires segmented reverse residuals.")

    objective_count = int(len(OBJECTIVE_LABELS))

    def _batched_zero_tangent_tree_like(primal_tree, batch_size: int):
        zero_tree = _radau_align_tangent_tree_to_primal(None, primal_tree)
        return jax.tree_util.tree_map(
            lambda leaf: jnp.broadcast_to(
                jnp.asarray(leaf)[None, ...],
                (batch_size,) + jnp.asarray(leaf).shape,
            ),
            zero_tree,
        )

    reduced_bars = _RadauAcceptedStepReducedCotangent(
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
    cotangent_mode = str(
        getattr(reverse_setup.execution_context.physics_context, "reverse_stage_cotangent_mode", "full")
    ).strip().lower()
    segment_count = int(jax.tree_util.tree_leaves(segmented_replay_arrays)[0].shape[0])
    for segment_index in range(segment_count - 1, -1, -1):
        segment_start_carry = _take_tree_axis0(segment_start_carries, segment_index)
        segment_arrays = _take_tree_axis0(segmented_replay_arrays, segment_index)
        reduced_bars = _radau_segment_reduced_cotangent_bwd_batched_call(
            reverse_setup.execution_context,
            cotangent_mode,
            reduced_bars,
            segment_start_carry,
            segment_arrays,
        )

    def _full_carry_bar_from_reduced(reduced_bar):
        return dataclasses.replace(
            jax.tree_util.tree_map(_zero_tangent_like, carry0),
            y=reduced_bar.y,
            lagged_response_cache=reduced_bar.lagged_response_cache,
            lagged_reference_y=reduced_bar.lagged_reference_y,
        )

    carry0_bars = jax.vmap(_full_carry_bar_from_reduced)(reduced_bars)
    gradient_matrix = jax.vmap(lambda carry0_bar: initial_carry_pullback(carry0_bar)[0])(carry0_bars)
    return objective_values, gradient_matrix


def _reverse_objective_support_payload_bar_for_parameter_vector(
    parameter_values,
    *,
    runtime,
    baseline_state,
    profile_cfg: dict,
    reverse_setup: _ReverseStaticSetup,
    objective_index: int,
    support_payload,
):
    """Return one objective value, profile gradients, and realtime support cotangent."""

    if reverse_setup.reverse_segment_length is None or int(reverse_setup.reverse_segment_length) <= 0:
        raise ValueError("support payload reverse probe requires --reverse-segment-length.")
    step_bwd_mode = str(
        getattr(reverse_setup.execution_context.physics_context, "reverse_step_bwd_mode", "current")
    ).strip().lower()
    if step_bwd_mode not in {
        "reduced_cotangent",
        "reduced_cotangent_call_boundary",
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

    def _carry_from_parameters(p):
        return _reverse_initial_carry_for_parameter_vector(
            p,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            reverse_setup=reverse_setup,
        )

    initial_carry, initial_carry_pullback = jax.vjp(_carry_from_parameters, parameter_values)
    final_y, residuals = _radau_adaptive_final_y_realized_schedule_vjp_fwd(
        reverse_setup.execution_context,
        reverse_setup.max_total_steps,
        reverse_setup.stop_after_accepted_steps,
        reverse_setup.reverse_segment_length,
        initial_carry,
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

    # The support probe's backward pass is the segmented accepted-replay map.
    # Seed objective cotangents at that same replay final state, not at the
    # full adaptive schedule final state, so FD/reverse compare the same map.
    final_y_for_objective = segmented_final_carry.y

    def _objective_from_final_y(final_y_value):
        final_state = reverse_setup.prepared_rollout.physics_context.unpack_flat(final_y_value)
        return _objective_scalar_by_index(final_state, runtime, objective_index)

    objective_value, objective_pullback = jax.vjp(_objective_from_final_y, final_y_for_objective)
    (final_y_bar,) = objective_pullback(jnp.ones_like(objective_value))

    reduced_bar = _RadauAcceptedStepReducedCotangent(
        y=final_y_bar,
        lagged_response_cache=_radau_align_tangent_tree_to_primal(
            None,
            segmented_final_carry.lagged_response_cache,
        ),
        lagged_reference_y=jnp.zeros_like(segmented_final_carry.lagged_reference_y),
    )
    support_bar = _radau_zero_support_delta_tree_like(support_payload)
    cotangent_mode = str(
        getattr(reverse_setup.execution_context.physics_context, "reverse_stage_cotangent_mode", "full")
    ).strip().lower()
    segment_count = int(jax.tree_util.tree_leaves(segmented_replay_arrays)[0].shape[0])
    support_reuse_count = 0
    support_rebuild_count = 0
    for segment_index in range(segment_count - 1, -1, -1):
        segment_start_carry = _take_tree_axis0(segment_start_carries, segment_index)
        segment_arrays = _take_tree_axis0(segmented_replay_arrays, segment_index)
        reduced_bar, segment_support_bar = _radau_segment_reduced_cotangent_bwd_with_support_call(
            reverse_setup.execution_context,
            cotangent_mode,
            reduced_bar,
            segment_start_carry,
            segment_arrays,
            support_payload,
        )
        support_bar = _radau_add_support_delta_trees(support_bar, segment_support_bar, support_payload)
        segment_lagged_valid = np.asarray(jax.device_get(segment_arrays[6])).reshape(-1)
        support_reuse_count += int(np.count_nonzero(segment_lagged_valid))
        support_rebuild_count += int(segment_lagged_valid.size - np.count_nonzero(segment_lagged_valid))

    initial_cache_pullback_used = False
    initial_cache_pullback_skipped = False
    initial_lagged_response_valid = bool(np.asarray(jax.device_get(carry0.lagged_response_valid)))
    build_support_pullback = reverse_setup.execution_context.physics_context.flat_rhs_build_support_pullback
    allow_initial_cache_support_pullback = cotangent_mode in {
        "full_initial_cache_support_pullback",
        "initial_cache_support_pullback",
    }
    if initial_lagged_response_valid and build_support_pullback is not None and allow_initial_cache_support_pullback:
        initial_cache_support_bar = build_support_pullback(
            carry0.y,
            reduced_bar.lagged_response_cache,
            support_payload,
        )
        support_bar = _radau_add_support_delta_trees(
            support_bar,
            initial_cache_support_bar,
            support_payload,
        )
        initial_cache_pullback_used = True
    elif initial_lagged_response_valid and build_support_pullback is not None:
        initial_cache_pullback_skipped = True

    carry0_bar = dataclasses.replace(
        jax.tree_util.tree_map(_zero_tangent_like, carry0),
        y=reduced_bar.y,
        lagged_response_cache=reduced_bar.lagged_response_cache,
        lagged_reference_y=reduced_bar.lagged_reference_y,
    )
    (profile_parameter_bar,) = initial_carry_pullback(carry0_bar)
    return (
        objective_value,
        profile_parameter_bar,
        support_bar,
        support_reuse_count,
        support_rebuild_count,
        initial_cache_pullback_used,
        initial_cache_pullback_skipped,
    )


def _reverse_all_objectives_support_payload_bar_for_parameter_vector(
    parameter_values,
    *,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict,
    reverse_setup: _ReverseStaticSetup,
    support_payload,
    initial_er_root_ad: str = "off",
):
    """Return all objective values, profile gradients, and realtime support cotangents."""

    dependencies = RealtimeGeometrySupportReverseDependencies(
        initial_er_root_enabled=_initial_er_root_enabled,
        initial_state_for_parameter_vector=_initial_state_for_parameter_vector,
        state_with_initial_er_root_ad=_state_with_initial_er_root_ad,
        reverse_initial_carry_from_state_with_static_setup=(
            _reverse_initial_carry_from_state_with_static_setup
        ),
        objective_scalar_by_index=_objective_scalar_by_index,
        add_trees=_add_trees,
        initial_er_selected_root_profile=_initial_er_selected_root_profile,
        initial_er_charge_flux_residuals=_initial_er_charge_flux_residuals,
        initial_er_charge_flux_residual_scalar=_initial_er_charge_flux_residual_scalar,
        initial_er_charge_flux_residual_er_derivative=(
            _initial_er_charge_flux_residual_er_derivative
        ),
        compact_initial_er_state_pullback=compact_initial_er_state_pullback,
        compact_initial_er_ntx_support_pullback_leaves=(
            _compact_initial_er_ntx_support_pullback_leaves
        ),
        runtime_with_geometry_payload=_runtime_with_geometry_payload,
        runtime_with_ntx_support_payload=_runtime_with_ntx_support_payload,
    )
    return realtime_geometry_reverse_all_objectives_support_payload_bar_for_parameter_vector(
        parameter_values,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        reverse_setup=reverse_setup,
        support_payload=support_payload,
        initial_er_root_ad=initial_er_root_ad,
        objective_labels=OBJECTIVE_LABELS,
        dependencies=dependencies,
    )


def _reverse_objective_initial_state_bar(
    initial_state,
    *,
    runtime,
    reverse_setup: _ReverseStaticSetup,
    objective_index: int,
):
    """Return objective value and compact cotangent wrt the initial transport state."""

    if reverse_setup.reverse_segment_length is None or int(reverse_setup.reverse_segment_length) <= 0:
        raise ValueError("initial-carry boundary probe requires --reverse-segment-length.")
    step_bwd_mode = str(
        getattr(reverse_setup.execution_context.physics_context, "reverse_step_bwd_mode", "current")
    ).strip().lower()
    if step_bwd_mode not in {
        "reduced_cotangent",
        "reduced_cotangent_call_boundary",
        "reduced_cotangent_lean_replay",
        "reduced_cotangent_recompute_replay",
        "lean_replay",
        "recompute_replay",
        "reduced",
        "state_only",
        "final_state",
    }:
        raise ValueError("initial-carry boundary probe requires a reduced-cotangent reverse step bwd mode.")

    def _zero_tangent_like(x):
        arr = jnp.asarray(x)
        if jnp.issubdtype(arr.dtype, jnp.inexact):
            return jnp.zeros_like(arr)
        return jnp.zeros(arr.shape, dtype=jax.dtypes.float0)

    def _take_tree_axis0(tree, index: int):
        return jax.tree_util.tree_map(lambda value: value[index], tree)

    def _carry_from_state(state_value):
        return _reverse_initial_carry_from_state_with_static_setup(
            solver=reverse_setup.solver,
            state=state_value,
            solve_vector_field=reverse_setup.solve_vector_field,
            species=runtime.species,
            prepared_rollout_static=reverse_setup.prepared_rollout,
        )

    initial_carry, initial_state_pullback = jax.vjp(_carry_from_state, initial_state)
    final_y, residuals = _radau_adaptive_final_y_realized_schedule_vjp_fwd(
        reverse_setup.execution_context,
        reverse_setup.max_total_steps,
        reverse_setup.stop_after_accepted_steps,
        reverse_setup.reverse_segment_length,
        initial_carry,
    )

    def _objective_from_final_y(final_y_value):
        final_state = reverse_setup.prepared_rollout.physics_context.unpack_flat(final_y_value)
        return _objective_scalar_by_index(final_state, runtime, objective_index)

    objective_value, objective_pullback = jax.vjp(_objective_from_final_y, final_y)
    (final_y_bar,) = objective_pullback(jnp.ones_like(objective_value))

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
        raise ValueError("initial-carry boundary probe requires segmented reverse residuals.")

    reduced_bar = _RadauAcceptedStepReducedCotangent(
        y=final_y_bar,
        lagged_response_cache=_radau_align_tangent_tree_to_primal(
            None,
            segmented_final_carry.lagged_response_cache,
        ),
        lagged_reference_y=jnp.zeros_like(segmented_final_carry.lagged_reference_y),
    )
    cotangent_mode = str(
        getattr(reverse_setup.execution_context.physics_context, "reverse_stage_cotangent_mode", "full")
    ).strip().lower()
    segment_count = int(jax.tree_util.tree_leaves(segmented_replay_arrays)[0].shape[0])
    for segment_index in range(segment_count - 1, -1, -1):
        segment_start_carry = _take_tree_axis0(segment_start_carries, segment_index)
        segment_arrays = _take_tree_axis0(segmented_replay_arrays, segment_index)
        reduced_bar = _radau_segment_reduced_cotangent_bwd_call(
            reverse_setup.execution_context,
            cotangent_mode,
            reduced_bar,
            segment_start_carry,
            segment_arrays,
        )

    carry0_bar = dataclasses.replace(
        jax.tree_util.tree_map(_zero_tangent_like, carry0),
        y=reduced_bar.y,
        lagged_response_cache=reduced_bar.lagged_response_cache,
        lagged_reference_y=reduced_bar.lagged_reference_y,
    )
    (initial_state_bar,) = initial_state_pullback(carry0_bar)
    carry0_bar_without_lagged_cache = dataclasses.replace(
        carry0_bar,
        lagged_response_cache=_radau_align_tangent_tree_to_primal(None, carry0.lagged_response_cache),
    )
    (initial_state_bar_without_lagged_cache,) = initial_state_pullback(carry0_bar_without_lagged_cache)
    carry0_zero_bar = jax.tree_util.tree_map(_zero_tangent_like, carry0)
    carry0_y_only_bar = dataclasses.replace(carry0_zero_bar, y=reduced_bar.y)
    carry0_lagged_reference_only_bar = dataclasses.replace(
        carry0_zero_bar,
        lagged_reference_y=reduced_bar.lagged_reference_y,
    )
    carry0_lagged_cache_only_bar = dataclasses.replace(
        carry0_zero_bar,
        lagged_response_cache=reduced_bar.lagged_response_cache,
    )
    (initial_state_bar_y_only,) = initial_state_pullback(carry0_y_only_bar)
    (initial_state_bar_lagged_reference_only,) = initial_state_pullback(
        carry0_lagged_reference_only_bar
    )
    (initial_state_bar_lagged_cache_only,) = initial_state_pullback(carry0_lagged_cache_only_bar)
    diagnostics = {
        "reduced_y_bar_l2": _tree_array_l2_norm(reduced_bar.y),
        "reduced_y_bar_summary": _payload_leaf_summary(reduced_bar.y),
        "reduced_lagged_response_cache_bar_l2": _tree_array_l2_norm(reduced_bar.lagged_response_cache),
        "reduced_lagged_response_cache_bar_summary": _payload_leaf_summary(reduced_bar.lagged_response_cache),
        "reduced_lagged_reference_y_bar_l2": _tree_array_l2_norm(reduced_bar.lagged_reference_y),
        "reduced_lagged_reference_y_bar_summary": _payload_leaf_summary(reduced_bar.lagged_reference_y),
        "carry0_bar_l2": _tree_array_l2_norm(carry0_bar),
        "carry0_bar_summary": _payload_leaf_summary(carry0_bar),
        "initial_state_bar_without_lagged_cache_l2": _tree_array_l2_norm(
            initial_state_bar_without_lagged_cache
        ),
        "initial_state_bar_without_lagged_cache_summary": _payload_leaf_summary(
            initial_state_bar_without_lagged_cache
        ),
        "initial_state_bar_without_lagged_cache_field_l2": {
            "density": _tree_array_l2_norm(initial_state_bar_without_lagged_cache.density),
            "pressure": _tree_array_l2_norm(initial_state_bar_without_lagged_cache.pressure),
            "Er": _tree_array_l2_norm(initial_state_bar_without_lagged_cache.Er),
        },
        "initial_state_bar_y_only_l2": _tree_array_l2_norm(initial_state_bar_y_only),
        "initial_state_bar_y_only_summary": _payload_leaf_summary(initial_state_bar_y_only),
        "initial_state_bar_y_only_field_l2": {
            "density": _tree_array_l2_norm(initial_state_bar_y_only.density),
            "pressure": _tree_array_l2_norm(initial_state_bar_y_only.pressure),
            "Er": _tree_array_l2_norm(initial_state_bar_y_only.Er),
        },
        "initial_state_bar_lagged_reference_only_l2": _tree_array_l2_norm(
            initial_state_bar_lagged_reference_only
        ),
        "initial_state_bar_lagged_reference_only_summary": _payload_leaf_summary(
            initial_state_bar_lagged_reference_only
        ),
        "initial_state_bar_lagged_cache_only_l2": _tree_array_l2_norm(initial_state_bar_lagged_cache_only),
        "initial_state_bar_lagged_cache_only_summary": _payload_leaf_summary(
            initial_state_bar_lagged_cache_only
        ),
    }
    return objective_value, initial_state_bar, diagnostics


def _reverse_all_objectives_initial_state_boundary(
    initial_state,
    *,
    runtime,
    reverse_setup: _ReverseStaticSetup,
):
    """Return all objective values and compact initial-state cotangents."""

    if reverse_setup.reverse_segment_length is None or int(reverse_setup.reverse_segment_length) <= 0:
        raise ValueError("initial-carry boundary probe requires --reverse-segment-length.")
    step_bwd_mode = str(
        getattr(reverse_setup.execution_context.physics_context, "reverse_step_bwd_mode", "current")
    ).strip().lower()
    if step_bwd_mode not in {
        "reduced_cotangent",
        "reduced_cotangent_call_boundary",
        "reduced_cotangent_lean_replay",
        "reduced_cotangent_recompute_replay",
        "lean_replay",
        "recompute_replay",
        "reduced",
        "state_only",
        "final_state",
    }:
        raise ValueError("initial-carry boundary probe requires a reduced-cotangent reverse step bwd mode.")

    def _zero_tangent_like(x):
        arr = jnp.asarray(x)
        if jnp.issubdtype(arr.dtype, jnp.inexact):
            return jnp.zeros_like(arr)
        return jnp.zeros(arr.shape, dtype=jax.dtypes.float0)

    def _take_tree_axis0(tree, index: int):
        return jax.tree_util.tree_map(lambda value: value[index], tree)

    def _batched_zero_tangent_tree_like(primal_tree, batch_size: int):
        zero_tree = _radau_align_tangent_tree_to_primal(None, primal_tree)
        return jax.tree_util.tree_map(
            lambda leaf: jnp.broadcast_to(
                jnp.asarray(leaf)[None, ...],
                (batch_size,) + jnp.asarray(leaf).shape,
            ),
            zero_tree,
        )

    def _carry_from_state(state_value):
        return _reverse_initial_carry_from_state_with_static_setup(
            solver=reverse_setup.solver,
            state=state_value,
            solve_vector_field=reverse_setup.solve_vector_field,
            species=runtime.species,
            prepared_rollout_static=reverse_setup.prepared_rollout,
        )

    initial_carry, initial_state_pullback = jax.vjp(_carry_from_state, initial_state)
    final_y, residuals = _radau_adaptive_final_y_realized_schedule_vjp_fwd(
        reverse_setup.execution_context,
        reverse_setup.max_total_steps,
        reverse_setup.stop_after_accepted_steps,
        reverse_setup.reverse_segment_length,
        initial_carry,
    )

    def _objective_vector_from_final_y(final_y_value):
        final_state = reverse_setup.prepared_rollout.physics_context.unpack_flat(final_y_value)
        return _objective_vector(final_state, runtime)

    objective_values, objective_pullback = jax.vjp(_objective_vector_from_final_y, final_y)
    objective_count = int(len(OBJECTIVE_LABELS))
    objective_basis = jnp.eye(objective_count, dtype=jnp.asarray(objective_values).dtype)
    final_y_bars = jax.vmap(lambda basis: objective_pullback(basis)[0])(objective_basis)

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
        raise ValueError("initial-carry boundary probe requires segmented reverse residuals.")

    reduced_bars = _RadauAcceptedStepReducedCotangent(
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
    cotangent_mode = str(
        getattr(reverse_setup.execution_context.physics_context, "reverse_stage_cotangent_mode", "full")
    ).strip().lower()
    segment_count = int(jax.tree_util.tree_leaves(segmented_replay_arrays)[0].shape[0])
    for segment_index in range(segment_count - 1, -1, -1):
        segment_start_carry = _take_tree_axis0(segment_start_carries, segment_index)
        segment_arrays = _take_tree_axis0(segmented_replay_arrays, segment_index)
        reduced_bars = _radau_segment_reduced_cotangent_bwd_batched_call(
            reverse_setup.execution_context,
            cotangent_mode,
            reduced_bars,
            segment_start_carry,
            segment_arrays,
        )

    def _full_carry_bar_from_reduced(reduced_bar):
        return dataclasses.replace(
            jax.tree_util.tree_map(_zero_tangent_like, carry0),
            y=reduced_bar.y,
            lagged_response_cache=reduced_bar.lagged_response_cache,
            lagged_reference_y=reduced_bar.lagged_reference_y,
        )

    carry0_bars = jax.vmap(_full_carry_bar_from_reduced)(reduced_bars)
    initial_state_bars = jax.vmap(lambda carry0_bar: initial_state_pullback(carry0_bar)[0])(
        carry0_bars
    )
    return objective_values, initial_state_bars, reduced_bars


def _reverse_final_y_objective_cotangent_for_parameter_vector(
    parameter_values,
    *,
    runtime,
    baseline_state,
    profile_cfg: dict,
    objective_index: int,
    reverse_setup: _ReverseStaticSetup,
):
    state0 = _initial_state_for_parameter_vector(
        parameter_values,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        runtime=runtime,
    )
    initial_carry = _reverse_initial_carry_from_state_with_static_setup(
        solver=reverse_setup.solver,
        state=state0,
        solve_vector_field=reverse_setup.solve_vector_field,
        species=runtime.species,
        prepared_rollout_static=reverse_setup.prepared_rollout,
    )
    final_y = _radau_adaptive_final_y_realized_schedule_vjp(
        reverse_setup.execution_context,
        reverse_setup.max_total_steps,
        reverse_setup.stop_after_accepted_steps,
        reverse_setup.reverse_segment_length,
        initial_carry,
    )

    def _objective_from_final_y(final_y_value):
        final_state = reverse_setup.prepared_rollout.physics_context.unpack_flat(final_y_value)
        return _objective_scalar_by_index(final_state, runtime, objective_index)

    objective_value = _objective_from_final_y(final_y)
    final_y_bar = jax.grad(_objective_from_final_y)(final_y)
    return objective_value, final_y_bar


def _reverse_initial_carry_for_parameter_vector(
    parameter_values,
    *,
    runtime,
    baseline_state,
    profile_cfg: dict,
    reverse_setup: _ReverseStaticSetup,
):
    state0 = _initial_state_for_parameter_vector(
        parameter_values,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        runtime=runtime,
    )
    return _reverse_initial_carry_from_state_with_static_setup(
        solver=reverse_setup.solver,
        state=state0,
        solve_vector_field=reverse_setup.solve_vector_field,
        species=runtime.species,
        prepared_rollout_static=reverse_setup.prepared_rollout,
    )


def _make_reverse_gradient_split_custom_vjp_fn(
    *,
    runtime,
    baseline_state,
    profile_cfg: dict,
    objective_index: int,
    reverse_setup: _ReverseStaticSetup,
    jit_kernels: bool,
):
    """Build a reusable split custom-VJP gradient pipeline."""

    def _carry_from_parameters(p):
        return _reverse_initial_carry_for_parameter_vector(
            p,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            reverse_setup=reverse_setup,
        )

    def _rollout_fwd(p):
        initial_carry = _carry_from_parameters(p)
        return _radau_adaptive_final_y_realized_schedule_vjp_fwd(
            reverse_setup.execution_context,
            reverse_setup.max_total_steps,
            reverse_setup.stop_after_accepted_steps,
            reverse_setup.reverse_segment_length,
            initial_carry,
        )

    def _objective_from_final_y(final_y):
        final_state = reverse_setup.prepared_rollout.physics_context.unpack_flat(final_y)
        return _objective_scalar_by_index(final_state, runtime, objective_index)

    def _rollout_bwd(residuals, final_y_bar):
        (carry0_bar,) = _radau_adaptive_final_y_realized_schedule_vjp_bwd(
            reverse_setup.execution_context,
            reverse_setup.max_total_steps,
            reverse_setup.stop_after_accepted_steps,
            reverse_setup.reverse_segment_length,
            residuals,
            final_y_bar,
        )
        return carry0_bar

    def _zero_tangent_like(x):
        arr = jnp.asarray(x)
        if jnp.issubdtype(arr.dtype, jnp.inexact):
            return jnp.zeros_like(arr)
        return jnp.zeros(arr.shape, dtype=jax.dtypes.float0)

    def _take_tree_axis0(tree, index: int):
        return jax.tree_util.tree_map(lambda value: value[index], tree)

    def _full_carry_bar_from_reduced(carry_value, reduced_bar_value):
        return dataclasses.replace(
            jax.tree_util.tree_map(_zero_tangent_like, carry_value),
            y=reduced_bar_value.y,
            lagged_response_cache=reduced_bar_value.lagged_response_cache,
            lagged_reference_y=reduced_bar_value.lagged_reference_y,
        )

    def _rollout_bwd_host_segments(residuals, final_y_bar):
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
        if segment_start_carries is None or segmented_final_carry is None or segmented_replay_arrays is None:
            raise ValueError("split-vjp segment host mode requires --reverse-segment-length.")
        segment_count = int(jax.tree_util.tree_leaves(segmented_replay_arrays)[0].shape[0])
        reduced_bar = _RadauAcceptedStepReducedCotangent(
            y=final_y_bar,
            lagged_response_cache=_radau_align_tangent_tree_to_primal(
                None,
                segmented_final_carry.lagged_response_cache,
            ),
            lagged_reference_y=jnp.zeros_like(segmented_final_carry.lagged_reference_y),
        )
        cotangent_mode = str(
            getattr(reverse_setup.execution_context.physics_context, "reverse_stage_cotangent_mode", "full")
        ).strip().lower()
        for segment_index in range(segment_count - 1, -1, -1):
            segment_start_carry = _take_tree_axis0(segment_start_carries, segment_index)
            segment_arrays = _take_tree_axis0(segmented_replay_arrays, segment_index)
            reduced_bar = _radau_segment_reduced_cotangent_bwd_call(
                reverse_setup.execution_context,
                cotangent_mode,
                reduced_bar,
                segment_start_carry,
                segment_arrays,
            )
            reduced_bar = jax.block_until_ready(reduced_bar)
        return _full_carry_bar_from_reduced(carry0, reduced_bar)

    def _parameter_pullback(p, carry0_bar):
        _, pullback = jax.vjp(_carry_from_parameters, p)
        (parameter_bar,) = pullback(carry0_bar)
        return parameter_bar

    use_host_segment_bwd = str(reverse_setup.reverse_segment_length) not in {"None", "0"} and str(
        getattr(reverse_setup.execution_context.physics_context, "reverse_step_bwd_mode", "current")
    ).strip().lower() in {"reduced_cotangent_host_segments", "host_segments"}

    if jit_kernels:
        rollout_fwd = jax.jit(_rollout_fwd)
        objective_final_y_grad = jax.jit(jax.grad(_objective_from_final_y))
        rollout_bwd = _rollout_bwd_host_segments if use_host_segment_bwd else jax.jit(_rollout_bwd)
        parameter_pullback = jax.jit(_parameter_pullback)
    else:
        rollout_fwd = _rollout_fwd
        objective_final_y_grad = jax.grad(_objective_from_final_y)
        rollout_bwd = _rollout_bwd_host_segments if use_host_segment_bwd else _rollout_bwd
        parameter_pullback = _parameter_pullback

    def _compute(parameter_values):
        final_y, residuals = rollout_fwd(parameter_values)
        final_y = jax.block_until_ready(final_y)
        final_y_bar = objective_final_y_grad(final_y)
        final_y_bar = jax.block_until_ready(final_y_bar)
        carry0_bar = rollout_bwd(residuals, final_y_bar)
        carry0_bar = jax.block_until_ready(carry0_bar)
        parameter_bar = parameter_pullback(parameter_values, carry0_bar)
        return jax.block_until_ready(parameter_bar)

    return _compute


def _prepare_reverse_static_setup(
    parameter_values,
    *,
    config: dict,
    runtime,
    baseline_state,
    profile_cfg: dict,
    initial_er_root_ad: str = "off",
    accepted_step_limit_override: int | None = None,
    reverse_segment_length: int | None = None,
    reverse_direct_stage_adjoint: bool = False,
    reverse_stage_adjoint_solve_mode: str = "structured",
    reverse_rhs_transpose_mode: str = "generic",
    reverse_rhs_pullback_mode: str = "separate",
    reverse_final_objective_cotangent_mode: str = "scalar",
    reverse_stage_cotangent_mode: str = "full",
    reverse_step_bwd_mode: str = "current",
    reverse_stage_adjoint_memory_mode: str = "default",
    reverse_stage_adjoint_iter_maxiter: int = 40,
    reverse_stage_adjoint_iter_tol: float = 1.0e-10,
    reverse_stage_adjoint_woodbury_rank: int = 24,
    reverse_single_segment_vjp_forward_mode: str = "legacy",
    reverse_schedule_artifact_mode: str = "legacy",
) -> _ReverseStaticSetup:
    state0_static = _initial_state_for_parameter_vector(
        parameter_values,
        config=config,
        initial_er_root_ad=initial_er_root_ad,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        runtime=runtime,
    )
    prepared_components_static = prepare_transport_solver_components(config, runtime, state0_static)
    solver = prepared_components_static["solver"]
    solve_vector_field_static = prepared_components_static["solve_vector_field"]
    prepared_rollout_static = _build_prepared_radau_accepted_rollout(
        solver=solver,
        state=state0_static,
        vector_field=solve_vector_field_static,
        species=runtime.species,
    )
    execution_context = _build_prepared_radau_execution_context(
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
                reverse_rhs_pullback_mode=str(reverse_rhs_pullback_mode),
                reverse_final_objective_cotangent_mode=str(
                    reverse_final_objective_cotangent_mode
                ),
                reverse_stage_cotangent_mode=str(reverse_stage_cotangent_mode),
                reverse_step_bwd_mode=str(reverse_step_bwd_mode),
                reverse_stage_adjoint_memory_mode=str(reverse_stage_adjoint_memory_mode),
                reverse_stage_adjoint_iter_maxiter=int(reverse_stage_adjoint_iter_maxiter),
                reverse_stage_adjoint_iter_tol=float(reverse_stage_adjoint_iter_tol),
                reverse_stage_adjoint_woodbury_rank=int(reverse_stage_adjoint_woodbury_rank),
                reverse_single_segment_vjp_forward_mode=str(reverse_single_segment_vjp_forward_mode),
            ),
        )
    stop_after_accepted_steps = (
        int(accepted_step_limit_override)
        if accepted_step_limit_override is not None
        else getattr(solver, "stop_after_accepted_steps", None)
    )
    schedule_artifact = None
    schedule_artifact_mode = str(reverse_schedule_artifact_mode).strip().lower()
    if schedule_artifact_mode not in {"legacy", "reuse_static_probe"}:
        raise ValueError(
            "Unknown reverse_schedule_artifact_mode "
            f"'{reverse_schedule_artifact_mode}'."
        )
    if schedule_artifact_mode == "reuse_static_probe" and stop_after_accepted_steps is None:
        raise ValueError(
            "reuse_static_probe requires --accepted-step-limit in this benchmark."
        )
    if (
        schedule_artifact_mode == "reuse_static_probe"
        and str(reverse_single_segment_vjp_forward_mode).strip().lower()
        == "reuse_adaptive_rollout"
    ):
        raise ValueError(
            "reuse_static_probe already removes the adaptive rollout and cannot be combined "
            "with reverse_single_segment_vjp_forward_mode='reuse_adaptive_rollout'."
        )
    max_total_steps = int(max(1, getattr(solver, "max_steps", 1)))
    if stop_after_accepted_steps is not None:
        max_total_steps = min(
            max_total_steps,
            max(int(stop_after_accepted_steps) * 16, int(stop_after_accepted_steps) + 16),
        )
        schedule_probe = _radau_adaptive_schedule_rollout(
            execution_context,
            prepared_rollout_static.initial_carry,
            max_total_steps=max_total_steps,
            stop_after_accepted_steps=stop_after_accepted_steps,
        )
        actual_attempt_count = int(np.asarray(jax.device_get(schedule_probe.attempt_count)))
        max_total_steps = min(
            max_total_steps,
            max(actual_attempt_count + 2, int(stop_after_accepted_steps)),
        )
        accepted_limit = int(stop_after_accepted_steps)
        active_mask_np = np.asarray(jax.device_get(schedule_probe.trace.active_mask), dtype=bool)
        accepted_mask_np = np.asarray(jax.device_get(schedule_probe.trace.accepted_mask), dtype=bool)
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
        if schedule_artifact_mode == "reuse_static_probe":
            # Scalar schedule trace only; no carry or step/stage tape.
            # Keep the same reduced graph length that legacy uses after its
            # adaptive rollout, not the conservative probe guard length.
            schedule_artifact = jax.tree_util.tree_map(
                lambda value: value[:max_total_steps],
                schedule_probe.trace,
            )
            print(
                "[autodiff-gate] progress: static reverse schedule artifact retained "
                f"attempt_slots={max_total_steps} (trace only; no carry tape)",
                flush=True,
            )
    return _ReverseStaticSetup(
        solver=solver,
        solve_vector_field=solve_vector_field_static,
        prepared_rollout=prepared_rollout_static,
        execution_context=execution_context,
        stop_after_accepted_steps=stop_after_accepted_steps,
        max_total_steps=max_total_steps,
        reverse_segment_length=reverse_segment_length,
        schedule_artifact=schedule_artifact,
    )


def _baseline_rollout_for_diagnostics(
    parameter_values,
    *,
    config: dict,
    runtime,
    baseline_state,
    profile_cfg: dict,
    accepted_step_limit_override: int | None = None,
):
    state0 = _initial_state_for_parameter_vector(
        parameter_values,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        runtime=runtime,
    )
    prepared_components = prepare_transport_solver_components(config, runtime, state0)
    solver = prepared_components["solver"]
    solve_vector_field = prepared_components["solve_vector_field"]
    prepared_rollout = _build_prepared_radau_accepted_rollout(
        solver=solver,
        state=state0,
        vector_field=solve_vector_field,
        species=runtime.species,
    )
    execution_context = _build_prepared_radau_execution_context(
        solver=solver,
        prepared_rollout=prepared_rollout,
    )
    stop_after_accepted_steps = (
        int(accepted_step_limit_override)
        if accepted_step_limit_override is not None
        else getattr(solver, "stop_after_accepted_steps", None)
    )
    max_total_steps = int(max(1, getattr(solver, "max_steps", 1)))
    if stop_after_accepted_steps is not None:
        max_total_steps = min(
            max_total_steps,
            max(int(stop_after_accepted_steps) * 16, int(stop_after_accepted_steps) + 16),
        )
    return _radau_adaptive_schedule_rollout(
        execution_context,
        prepared_rollout.initial_carry,
        max_total_steps=max_total_steps,
        stop_after_accepted_steps=stop_after_accepted_steps,
    )


def _reverse_geometry_rollout_parts_for_parameter_vector(
    parameter_values,
    *,
    config: dict[str, Any],
    geometry_context,
    profile_cfg: dict,
    geometry_parameter_name: str,
    fixed_initial_er=None,
    accepted_step_limit_override: int | None = None,
    reverse_segment_length: int | None = None,
    solver_override=None,
):
    del (
        parameter_values,
        config,
        geometry_context,
        profile_cfg,
        geometry_parameter_name,
        fixed_initial_er,
        accepted_step_limit_override,
        reverse_segment_length,
        solver_override,
    )
    raise NotImplementedError(
        "The old whole-runtime realtime geometry AD rollout path is disabled. "
        "Use --realtime-geometry-gradient-path reverse_payload for the compact "
        "combined runtime.geometry + NTX-support payload lane."
    )


def _reverse_geometry_objective_vector_for_parameter_vector(
    parameter_values,
    *,
    config: dict[str, Any],
    geometry_context,
    profile_cfg: dict,
    geometry_parameter_name: str,
    fixed_initial_er=None,
    accepted_step_limit_override: int | None = None,
    reverse_segment_length: int | None = None,
    solver_override=None,
):
    del reverse_segment_length
    runtime, prepared_rollout, execution_context, stop_after_accepted_steps, max_total_steps = (
        _reverse_geometry_rollout_parts_for_parameter_vector(
            parameter_values,
            config=config,
            geometry_context=geometry_context,
            profile_cfg=profile_cfg,
            geometry_parameter_name=geometry_parameter_name,
            fixed_initial_er=fixed_initial_er,
            accepted_step_limit_override=accepted_step_limit_override,
            solver_override=solver_override,
        )
    )
    rollout = _radau_adaptive_schedule_rollout(
        execution_context,
        prepared_rollout.initial_carry,
        max_total_steps=max_total_steps,
        stop_after_accepted_steps=stop_after_accepted_steps,
    )
    final_state = prepared_rollout.physics_context.unpack_flat(rollout.final_carry.y)
    return _objective_vector(final_state, runtime)


def _make_reverse_geometry_objective_vector_with_schedule_vjp(
    *,
    config: dict[str, Any],
    geometry_context,
    profile_cfg: dict,
    geometry_parameter_name: str,
    fixed_initial_er=None,
    accepted_step_limit_override: int | None = None,
    solver_override=None,
):
    @jax.custom_vjp
    def _objective(parameter_values):
        return _reverse_geometry_objective_vector_for_parameter_vector(
            parameter_values,
            config=config,
            geometry_context=geometry_context,
            profile_cfg=profile_cfg,
            geometry_parameter_name=geometry_parameter_name,
            fixed_initial_er=fixed_initial_er,
            accepted_step_limit_override=accepted_step_limit_override,
            solver_override=solver_override,
        )

    def _fwd(parameter_values):
        runtime, prepared_rollout, execution_context, stop_after_accepted_steps, max_total_steps = (
            _reverse_geometry_rollout_parts_for_parameter_vector(
                parameter_values,
                config=config,
                geometry_context=geometry_context,
                profile_cfg=profile_cfg,
                geometry_parameter_name=geometry_parameter_name,
                fixed_initial_er=fixed_initial_er,
                accepted_step_limit_override=accepted_step_limit_override,
                solver_override=solver_override,
            )
        )
        rollout = _radau_adaptive_schedule_rollout(
            execution_context,
            prepared_rollout.initial_carry,
            max_total_steps=max_total_steps,
            stop_after_accepted_steps=stop_after_accepted_steps,
        )
        final_state = prepared_rollout.physics_context.unpack_flat(rollout.final_carry.y)
        value = _objective_vector(final_state, runtime)
        trace = rollout.trace
        residuals = (
            parameter_values,
            jax.lax.stop_gradient(trace.active_mask),
            jax.lax.stop_gradient(trace.accepted_mask),
            jax.lax.stop_gradient(trace.attempted_dts),
            jax.lax.stop_gradient(trace.next_dts),
            jax.lax.stop_gradient(trace.next_recent_reject_count),
            jax.lax.stop_gradient(trace.next_regrowth_cooldown),
            jax.lax.stop_gradient(trace.next_easy_growth_streak),
            jax.lax.stop_gradient(trace.next_lagged_response_valid),
        )
        return value, residuals

    def _bwd(residuals, objective_bar):
        (
            parameter_values,
            _active_mask,
            _accepted_mask,
            _attempted_dts,
            _next_dts,
            _next_recent_reject_count,
            _next_regrowth_cooldown,
            _next_easy_growth_streak,
            _next_lagged_response_valid,
        ) = residuals

        del (
            parameter_values,
            objective_bar,
        )
        raise NotImplementedError(
            "Realtime geometry reverse AD must use a reverse-only geometry-payload "
            "accepted-step transpose. The VMEC implicit lane is custom_vjp-only "
            "and cannot be used through jax.jvp, while the profile replay VJP "
            "keeps physics_context static and therefore cannot carry traced "
            "geometry/NTX support. Keep --reverse-parameter-mode profiles for "
            "frozen/profile reverse AD until the geometry-payload reverse path "
            "is implemented."
        )

    _objective.defvjp(_fwd, _bwd)
    return _objective


def _run_realtime_geometry_payload_boundary_probe(
    *,
    args,
    config: dict[str, Any],
    baseline_values,
    baseline_runtime,
    baseline_state,
    profile_cfg: dict,
    neoclassical_cfg: dict[str, Any],
):
    support_payload = _find_ntx_support_payload(baseline_runtime)
    payload_summary = _payload_leaf_summary(support_payload)
    swapped_runtime = _runtime_with_ntx_support_payload(baseline_runtime, support_payload)
    baseline_profile_state = _initial_state_for_parameter_vector(
        baseline_values[: len(PARAMETER_ORDER)],
        config=config,
        initial_er_root_ad=args.initial_er_root_ad,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        runtime=baseline_runtime,
    )
    swapped_profile_state = _initial_state_for_parameter_vector(
        baseline_values[: len(PARAMETER_ORDER)],
        config=config,
        initial_er_root_ad=args.initial_er_root_ad,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        runtime=swapped_runtime,
    )
    baseline_components = prepare_transport_solver_components(
        config,
        baseline_runtime,
        baseline_profile_state,
    )
    swapped_components = prepare_transport_solver_components(
        config,
        swapped_runtime,
        swapped_profile_state,
        solver_override=baseline_components["solver"],
    )

    def _rollout_objectives(runtime, components):
        solver = components["solver"]
        prepared_rollout = _build_prepared_radau_accepted_rollout(
            solver=solver,
            state=components["solve_state"],
            vector_field=components["solve_vector_field"],
            species=runtime.species,
        )
        execution_context = _build_prepared_radau_execution_context(
            solver=solver,
            prepared_rollout=prepared_rollout,
        )
        stop_after_accepted_steps = (
            int(args.accepted_step_limit)
            if args.accepted_step_limit is not None
            else getattr(solver, "stop_after_accepted_steps", None)
        )
        max_total_steps = int(max(1, getattr(solver, "max_steps", 1)))
        if stop_after_accepted_steps is not None:
            max_total_steps = min(
                max_total_steps,
                max(int(stop_after_accepted_steps) * 16, int(stop_after_accepted_steps) + 16),
            )
        rollout = _radau_adaptive_schedule_rollout(
            execution_context,
            prepared_rollout.initial_carry,
            max_total_steps=max_total_steps,
            stop_after_accepted_steps=stop_after_accepted_steps,
        )
        final_state = prepared_rollout.physics_context.unpack_flat(rollout.final_carry.y)
        return _objective_vector(final_state, runtime), rollout

    print(
        "[autodiff-gate] progress: probing realtime geometry support-payload boundary",
        flush=True,
    )
    baseline_objectives, baseline_rollout = _rollout_objectives(baseline_runtime, baseline_components)
    swapped_objectives, swapped_rollout = _rollout_objectives(swapped_runtime, swapped_components)
    baseline_objectives = jax.block_until_ready(baseline_objectives)
    swapped_objectives = jax.block_until_ready(swapped_objectives)
    baseline_np = np.asarray(jax.device_get(baseline_objectives), dtype=float)
    swapped_np = np.asarray(jax.device_get(swapped_objectives), dtype=float)
    abs_delta = np.abs(swapped_np - baseline_np)
    rel_delta = abs_delta / np.maximum(1.0e-300, np.maximum(np.abs(baseline_np), np.abs(swapped_np)))
    finite_objectives = np.isfinite(baseline_np) & np.isfinite(swapped_np)
    finite_delta = finite_objectives & np.isfinite(abs_delta) & np.isfinite(rel_delta)
    nonfinite_objective_names = [
        name
        for name, finite in zip(OBJECTIVE_LABELS, finite_objectives.tolist())
        if not bool(finite)
    ]
    max_abs_delta = float(np.max(abs_delta[finite_delta])) if np.any(finite_delta) else None
    max_rel_delta = float(np.max(rel_delta[finite_delta])) if np.any(finite_delta) else None
    report = {
        "mode": "transport_reverse_ad_only",
        "parameter_mode": str(args.reverse_parameter_mode),
        "config_path": str(Path(args.config)),
        "objective_name": args.objective,
        "objective_order": list(OBJECTIVE_LABELS),
        "parameter_order": _reverse_geometry_parameter_order(str(args.reverse_geometry_parameter)),
        "baseline_values": np.asarray(jax.device_get(baseline_values), dtype=float).tolist(),
        "accepted_step_limit": None if args.accepted_step_limit is None else int(args.accepted_step_limit),
        "ntx_exact_derivative_mode": str(args.ntx_exact_derivative_mode),
        "ntx_exact_derivative_field_pullback_mode": str(args.ntx_exact_derivative_field_pullback_mode),
        "ntx_exact_surface_backend": str(neoclassical_cfg.get("ntx_exact_surface_backend", "booz")),
        "realtime_geometry_gradient_path": "payload_boundary_probe",
        "support_payload_summary": payload_summary,
        "baseline_attempt_count": int(np.asarray(jax.device_get(jnp.sum(baseline_rollout.trace.active_mask.astype(jnp.int32))))),
        "baseline_accepted_count": int(np.asarray(jax.device_get(jnp.sum(baseline_rollout.trace.accepted_mask.astype(jnp.int32))))),
        "swapped_attempt_count": int(np.asarray(jax.device_get(jnp.sum(swapped_rollout.trace.active_mask.astype(jnp.int32))))),
        "swapped_accepted_count": int(np.asarray(jax.device_get(jnp.sum(swapped_rollout.trace.accepted_mask.astype(jnp.int32))))),
        "objective_values_baseline": {
            name: float(value) for name, value in zip(OBJECTIVE_LABELS, baseline_np.tolist())
        },
        "objective_values_swapped_payload": {
            name: float(value) for name, value in zip(OBJECTIVE_LABELS, swapped_np.tolist())
        },
        "objective_abs_delta": {
            name: float(value) for name, value in zip(OBJECTIVE_LABELS, abs_delta.tolist())
        },
        "objective_rel_delta": {
            name: float(value) for name, value in zip(OBJECTIVE_LABELS, rel_delta.tolist())
        },
        "objective_finite": {
            name: bool(value) for name, value in zip(OBJECTIVE_LABELS, finite_objectives.tolist())
        },
        "nonfinite_objectives": nonfinite_objective_names,
        "max_finite_objective_abs_delta": max_abs_delta,
        "max_finite_objective_rel_delta": max_rel_delta,
        "gradient_reverse_ad": None,
    }
    max_abs_text = "nan" if max_abs_delta is None else f"{max_abs_delta:.6e}"
    max_rel_text = "nan" if max_rel_delta is None else f"{max_rel_delta:.6e}"
    print(
        "[autodiff-gate] mode=transport_reverse_ad_only "
        "parameter_mode=profiles_plus_realtime_geometry "
        "realtime_geometry_gradient_path=payload_boundary_probe "
        f"payload_array_leaves={payload_summary['n_array_leaves']} "
        f"payload_total_array_bytes={payload_summary['total_array_bytes']} "
        f"max_finite_objective_abs_delta={max_abs_text} "
        f"max_finite_objective_rel_delta={max_rel_text}",
        flush=True,
    )
    if nonfinite_objective_names:
        print(
            "[autodiff-gate] payload-boundary nonfinite objectives: "
            + ", ".join(nonfinite_objective_names),
            flush=True,
        )
    print("[autodiff-gate] payload-boundary objective deltas:")
    for objective_name in OBJECTIVE_LABELS:
        finite_status = "finite" if report["objective_finite"][objective_name] else "nonfinite"
        print(
            f"  - {objective_name}: "
            f"baseline={report['objective_values_baseline'][objective_name]:.16e} "
            f"swapped={report['objective_values_swapped_payload'][objective_name]:.16e} "
            f"abs_delta={report['objective_abs_delta'][objective_name]:.6e} "
            f"rel_delta={report['objective_rel_delta'][objective_name]:.6e} "
            f"status={finite_status}"
        )
    outpath = _report_path("realtime_geometry_payload_boundary")
    outpath.write_text(json.dumps(report, indent=2))
    print(f"Wrote {outpath.relative_to(ROOT)}")


def _run_realtime_geometry_support_pullback_probe(
    *,
    args,
    config: dict[str, Any],
    baseline_values,
    baseline_runtime,
    baseline_state,
    profile_cfg: dict,
    neoclassical_cfg: dict[str, Any],
):
    support_payload = _find_ntx_support_payload(baseline_runtime)
    baseline_profile_state = _initial_state_for_parameter_vector(
        baseline_values[: len(PARAMETER_ORDER)],
        config=config,
        initial_er_root_ad=args.initial_er_root_ad,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        runtime=baseline_runtime,
    )
    components = prepare_transport_solver_components(
        config,
        baseline_runtime,
        baseline_profile_state,
    )
    equation_system = components["equation_system"]
    solver = components["solver"]
    t0 = jnp.asarray(getattr(solver, "t0", 0.0), dtype=jnp.float64)
    lagged_response = equation_system.build_lagged_response(baseline_profile_state)
    rhs = equation_system.evaluate_with_lagged_response(
        t0,
        baseline_profile_state,
        baseline_runtime.species,
        lagged_response,
    )
    zero_rhs_bar = jax.tree_util.tree_map(jnp.zeros_like, rhs)

    def _rhs_bar_for(component_name: str):
        return dataclasses.replace(
            zero_rhs_bar,
            **{
                component_name: jnp.ones_like(
                    getattr(zero_rhs_bar, component_name)
                )
            },
        )

    support_bars = {}
    for component_name in ("density", "pressure", "Er"):
        support_bar_value = equation_system.pullback_evaluate_with_lagged_response_support_payload(
            t0,
            baseline_profile_state,
            baseline_runtime.species,
            lagged_response,
            _rhs_bar_for(component_name),
            support_payload,
        )
        support_bars[f"rhs_{component_name}"] = jax.block_until_ready(support_bar_value)

    if bool(args.support_pullback_probe_include_build):
        lagged_response_bar = jax.tree_util.tree_map(
            lambda leaf: (
                jnp.ones_like(leaf)
                if hasattr(leaf, "shape") and jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.inexact)
                else leaf
            ),
            lagged_response,
        )
        support_bars["build_lagged_response"] = jax.block_until_ready(
            equation_system.pullback_build_lagged_response_support_payload(
                baseline_profile_state,
                lagged_response_bar,
                support_payload,
            )
        )
    support_summary = _payload_leaf_summary(support_payload)
    support_bar_summaries = {
        name: _payload_leaf_summary(support_bar)
        for name, support_bar in support_bars.items()
    }
    support_bar_l2 = {
        name: _tree_array_l2_norm(support_bar)
        for name, support_bar in support_bars.items()
    }
    report = {
        "mode": "transport_reverse_ad_only",
        "parameter_mode": str(args.reverse_parameter_mode),
        "config_path": str(Path(args.config)),
        "objective_name": args.objective,
        "objective_order": list(OBJECTIVE_LABELS),
        "parameter_order": _reverse_geometry_parameter_order(str(args.reverse_geometry_parameter)),
        "baseline_values": np.asarray(jax.device_get(baseline_values), dtype=float).tolist(),
        "ntx_exact_derivative_mode": str(args.ntx_exact_derivative_mode),
        "ntx_exact_derivative_field_pullback_mode": str(args.ntx_exact_derivative_field_pullback_mode),
        "ntx_exact_surface_backend": str(neoclassical_cfg.get("ntx_exact_surface_backend", "booz")),
        "realtime_geometry_gradient_path": "support_pullback_probe",
        "support_payload_summary": support_summary,
        "support_bar_summary": support_bar_summaries,
        "support_bar_l2": support_bar_l2,
        "support_pullback_probe_include_build": bool(args.support_pullback_probe_include_build),
    }
    print(
        "[autodiff-gate] mode=transport_reverse_ad_only "
        "parameter_mode=profiles_plus_realtime_geometry "
        "realtime_geometry_gradient_path=support_pullback_probe ",
        flush=True,
    )
    for name in support_bars:
        summary = support_bar_summaries[name]
        print(
            f"[autodiff-gate] support_pullback {name}: "
            f"l2={support_bar_l2[name]:.6e} "
            f"array_leaves={summary['n_array_leaves']} "
            f"all_finite={summary['all_floating_leaves_finite']}",
            flush=True,
        )
    outpath = _report_path("realtime_geometry_support_pullback")
    outpath.write_text(json.dumps(report, indent=2))
    print(f"Wrote {outpath.relative_to(ROOT)}")


def _run_realtime_geometry_support_segment_probe(
    *,
    args,
    config: dict[str, Any],
    baseline_values,
    baseline_runtime,
    baseline_state,
    profile_cfg: dict,
    neoclassical_cfg: dict[str, Any],
    return_report: bool = False,
):
    if return_report and str(args.objective) != "all":
        raise ValueError("return_report=True is only supported for the grouped objective='all' path.")
    core_setup = prepare_realtime_geometry_support_segment_core_setup(
        args=args,
        config=config,
        baseline_values=baseline_values,
        baseline_runtime=baseline_runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        neoclassical_cfg=neoclassical_cfg,
        parameter_order=PARAMETER_ORDER,
        find_ntx_support_payload=_find_ntx_support_payload,
        prepare_reverse_static_setup=_prepare_reverse_static_setup,
        geometry_volume_diagnostics=_geometry_volume_diagnostics,
    )
    combined_geometry_payload = core_setup.combined_geometry_payload
    ntx_surface_backend = core_setup.ntx_surface_backend
    support_payload = core_setup.support_payload
    profile_values = core_setup.profile_values
    support_probe_cotangent_mode = core_setup.support_probe_cotangent_mode
    reverse_setup = core_setup.reverse_setup
    if args.objective == "all":
        early_geometry_diagnostics = core_setup.early_geometry_diagnostics
        if early_geometry_diagnostics is None:
            early_geometry_diagnostics = _geometry_volume_diagnostics(baseline_runtime.geometry)
        print("[autodiff-gate] realtime geometry pre-reverse diagnostics:")
        print(
            "[autodiff-gate] realtime geometry NTX surface backend: "
            f"{ntx_surface_backend}",
            flush=True,
        )
        for field_name in ("a_b", "R0", "r_grid", "Vprime", "Vprime_half", "overVprime", "integrated_volume"):
            if field_name not in early_geometry_diagnostics:
                continue
            summary = early_geometry_diagnostics[field_name]
            value_suffix = ""
            if field_name == "integrated_volume":
                value_suffix = f" value={early_geometry_diagnostics['integrated_volume_value']:.16e}"
            print(
                f"  - {field_name}: all_finite={summary['all_finite']} "
                f"nan_count={summary['nan_count']} "
                f"finite_min={summary['finite_min']} "
                f"finite_max={summary['finite_max']} "
                f"first_nonfinite_index={summary['first_nonfinite_index']}"
                f"{value_suffix}",
                flush=True,
            )
        print(
            "[autodiff-gate] progress: probing realized-schedule reverse support payload cotangents "
            "for all objectives",
            flush=True,
        )
        t_start = time.perf_counter()
        t_phase = time.perf_counter()
        support_cotangent_result = realtime_geometry_support_cotangents_from_parameter_vector(
            reverse_all_objectives_support_payload_bar=(
                _reverse_all_objectives_support_payload_bar_for_parameter_vector
            ),
            profile_values=profile_values,
            config=config,
            baseline_runtime=baseline_runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            reverse_setup=reverse_setup,
            support_payload=support_payload,
            initial_er_root_ad=args.initial_er_root_ad,
        )
        objective_values = support_cotangent_result.objective_values
        profile_gradient_matrix = support_cotangent_result.profile_gradient_matrix
        support_bars = support_cotangent_result.support_bars
        support_component_bars_by_name = support_cotangent_result.support_component_bars_by_name
        native_vmec_face_coefficient_bars = (
            support_cotangent_result.native_vmec_face_coefficient_bars
        )
        support_reuse_count = support_cotangent_result.support_reuse_count
        support_rebuild_count = support_cotangent_result.support_rebuild_count
        initial_cache_pullback_used = support_cotangent_result.initial_cache_pullback_used
        initial_cache_pullback_skipped = support_cotangent_result.initial_cache_pullback_skipped
        print(
            "[autodiff-gate] progress: transport reverse profile/support cotangents complete "
            f"elapsed_s={time.perf_counter() - t_phase:.3f}",
            flush=True,
        )
        skip_support_bar_diagnostics = bool(args.skip_realtime_geometry_support_bar_diagnostics)
        if skip_support_bar_diagnostics:
            print(
                "[autodiff-gate] realtime support/geometry-payload cotangent precheck skipped "
                "(--skip-realtime-geometry-support-bar-diagnostics)",
                flush=True,
            )
        else:
            pre_support_all_finite = True
            print("[autodiff-gate] realtime support/geometry-payload cotangent precheck:")
            for objective_i, objective_name in enumerate(OBJECTIVE_LABELS):
                support_bar = support_bars[objective_i]
                branch_diagnostics = _payload_branch_diagnostics(support_bar)
                root_summary = branch_diagnostics["root"]["summary"]
                if not root_summary["all_floating_leaves_finite"]:
                    pre_support_all_finite = False
                print(
                    f"  - {objective_name}: "
                    f"support_bar_l2={branch_diagnostics['root']['l2']:.6e} "
                    f"support_bar_all_finite={root_summary['all_floating_leaves_finite']}"
                )
                if not root_summary["all_floating_leaves_finite"]:
                    for branch_name, branch_summary in branch_diagnostics.items():
                        branch_leaf_summary = branch_summary["summary"]
                        if branch_leaf_summary["all_floating_leaves_finite"]:
                            continue
                        nonfinite_leaves = branch_summary["first_nonfinite_leaves"]
                        first_nonfinite = None if not nonfinite_leaves else nonfinite_leaves[0]
                        print(
                            f"      first bad branch={branch_name} "
                            f"l2={branch_summary['l2']:.6e} "
                            f"array_leaves={branch_leaf_summary['n_array_leaves']} "
                            f"first_nonfinite_leaf={first_nonfinite}",
                            flush=True,
                        )
                        break
            if not pre_support_all_finite:
                raise FloatingPointError(
                    "Realtime geometry payload pullback skipped because transport reverse "
                    "produced nonfinite support/geometry payload cotangents. See the "
                    "precheck branch output above for the first bad payload leaf."
                )

        geom_cfg = config.get("geometry", {})
        geometry_parameter_name = str(args.reverse_geometry_parameter)
        geometry_context = _geometry_context_from_config(config, geometry_parameter_name)
        geometry_param_specs = _geometry_param_specs_from_args(args, geometry_context)
        geometry_param_entries = boundary_param_entries(geometry_context, geometry_param_specs)
        geometry_param_labels = tuple(_format_geometry_param_spec(spec) for spec in geometry_param_specs)
        baseline_geometry_deltas = _baseline_geometry_delta_vector_for_specs(
            geom_cfg,
            geometry_param_specs,
        )

        t_phase = time.perf_counter()
        print(
            "[autodiff-gate] progress: building geometry support pullback "
            f"for {geometry_parameter_name} "
            f"(harmonic_count={len(geometry_param_specs)})",
            flush=True,
        )
        include_component_pullbacks = bool(args.realtime_geometry_component_pullbacks)
        assembly_result = realtime_geometry_transport_reverse_table_from_payload_cotangents(
            objective_labels=OBJECTIVE_LABELS,
            profile_parameter_labels=PARAMETER_ORDER,
            geometry_parameter_labels=geometry_param_labels,
            objective_values=objective_values,
            profile_gradient_matrix=profile_gradient_matrix,
            geometry_context=geometry_context,
            baseline_geometry_deltas=baseline_geometry_deltas,
            geometry_param_specs=geometry_param_specs,
            support_bars=tuple(support_bars),
            support_component_bars_by_name=support_component_bars_by_name,
            native_vmec_face_coefficient_bars=native_vmec_face_coefficient_bars,
            include_component_pullbacks=include_component_pullbacks,
            combined_geometry_payload=combined_geometry_payload,
            n_r=int(geom_cfg.get("n_radial", 51)),
            n_theta=int(neoclassical_cfg.get("ntx_exact_n_theta", 25)),
            n_zeta=int(neoclassical_cfg.get("ntx_exact_n_zeta", 25)),
            n_xi=int(neoclassical_cfg.get("ntx_exact_n_xi", 64)),
            surface_backend=ntx_surface_backend,
            max_iter=geom_cfg.get("vmec_max_iter"),
            solver_device=str(geom_cfg.get("vmec_implicit_solver_device", "default")),
            progress_label="[autodiff-gate] realtime geometry payload pullback:",
            return_branch_gradients=not (
                bool(getattr(args, "optimization_api_smoke", False))
                or bool(getattr(args, "full_transport_shared_payload_smoke", False))
            ),
        )
        table_result = assembly_result.table_result
        geometry_pullback_result = assembly_result.payload_pullback_result
        geometry_pullback_mode = geometry_pullback_result.pullback_mode
        geometry_gradient_matrix = geometry_pullback_result.geometry_gradient_matrix
        geometry_branch_gradient_matrix = geometry_pullback_result.geometry_branch_gradient_matrix
        ntx_support_branch_gradient_matrix = geometry_pullback_result.ntx_support_branch_gradient_matrix
        component_gradient_matrices = geometry_pullback_result.component_gradient_matrices
        component_geometry_branch_matrices = geometry_pullback_result.component_geometry_branch_matrices
        component_ntx_support_branch_matrices = geometry_pullback_result.component_ntx_support_branch_matrices
        print(
            "[autodiff-gate] progress: geometry support pullback complete "
            f"mode={geometry_pullback_mode} elapsed_s={time.perf_counter() - t_phase:.3f}",
            flush=True,
        )
        elapsed_s = time.perf_counter() - t_start

        geometry_branch_gradient_np = (
            None
            if geometry_branch_gradient_matrix is None
            else np.asarray(jax.device_get(geometry_branch_gradient_matrix), dtype=float)
        )
        ntx_support_branch_gradient_np = (
            None
            if ntx_support_branch_gradient_matrix is None
            else np.asarray(jax.device_get(ntx_support_branch_gradient_matrix), dtype=float)
        )
        component_gradient_np_by_name = {
            component_name: np.asarray(jax.device_get(component_matrix), dtype=float)
            for component_name, component_matrix in component_gradient_matrices.items()
        }
        component_geometry_branch_np_by_name = {
            component_name: np.asarray(jax.device_get(component_matrix), dtype=float)
            for component_name, component_matrix in component_geometry_branch_matrices.items()
        }
        component_ntx_support_branch_np_by_name = {
            component_name: np.asarray(jax.device_get(component_matrix), dtype=float)
            for component_name, component_matrix in component_ntx_support_branch_matrices.items()
        }
        table_report_entries = transport_reverse_table_report_entries(
            table_result=table_result,
        )
        geometry_gradient_np = np.asarray(jax.device_get(geometry_gradient_matrix), dtype=float)
        realtime_geometry_diagnostics = _geometry_volume_diagnostics(baseline_runtime.geometry)
        support_bar_summary_by_objective = {}
        support_bar_l2_by_objective = {}
        support_bar_branch_diagnostics_by_objective = {}
        if not skip_support_bar_diagnostics:
            for objective_i, objective_name in enumerate(OBJECTIVE_LABELS):
                support_bar = support_bars[objective_i]
                support_bar_summary_by_objective[objective_name] = _payload_leaf_summary(support_bar)
                support_bar_l2_by_objective[objective_name] = _tree_array_l2_norm(support_bar)
                support_bar_branch_diagnostics_by_objective[objective_name] = (
                    _payload_branch_diagnostics(support_bar)
                )

        support_summary = (
            {}
            if skip_support_bar_diagnostics
            else _payload_leaf_summary(support_payload)
        )
        metadata_entries = realtime_geometry_transport_reverse_metadata_entries(
            parameter_mode=str(args.reverse_parameter_mode),
            config_path=str(Path(args.config)),
            objective_labels=OBJECTIVE_LABELS,
            profile_parameter_labels=PARAMETER_ORDER,
            profile_values=profile_values,
            geometry_parameter_labels=geometry_param_labels,
            geometry_parameter_entries=geometry_param_entries,
            baseline_geometry_deltas=baseline_geometry_deltas,
            geometry_parameter_specs=geometry_param_specs,
            geometry_parameter_selector=str(geometry_parameter_name),
            accepted_step_limit=None
            if args.accepted_step_limit is None
            else int(args.accepted_step_limit),
            reverse_segment_length=None
            if args.reverse_segment_length is None
            else int(args.reverse_segment_length),
            reverse_stage_cotangent_mode_requested=str(args.reverse_stage_cotangent_mode),
            reverse_stage_cotangent_mode_effective=support_probe_cotangent_mode,
            ntx_exact_derivative_mode=str(args.ntx_exact_derivative_mode),
            ntx_exact_derivative_field_pullback_mode=str(
                args.ntx_exact_derivative_field_pullback_mode
            ),
            ntx_exact_surface_backend=str(neoclassical_cfg.get("ntx_exact_surface_backend", "booz")),
            realtime_geometry_gradient_path=str(args.realtime_geometry_gradient_path),
            realtime_geometry_component_pullbacks=bool(include_component_pullbacks),
            realtime_geometry_support_bar_diagnostics_skipped=bool(skip_support_bar_diagnostics),
            realtime_geometry_derivative_complete=bool(combined_geometry_payload),
            geometry_support_pullback_mode=geometry_pullback_mode,
            realtime_geometry_diagnostics=realtime_geometry_diagnostics,
            support_payload_summary=support_summary,
            support_bar_summary_by_objective=support_bar_summary_by_objective,
            support_bar_l2_by_objective=support_bar_l2_by_objective,
            support_bar_branch_diagnostics_by_objective=support_bar_branch_diagnostics_by_objective,
            support_reuse_count=support_reuse_count,
            support_rebuild_count=support_rebuild_count,
            support_initial_cache_pullback_used=initial_cache_pullback_used,
            support_initial_cache_pullback_skipped=initial_cache_pullback_skipped,
            elapsed_s=elapsed_s,
        )
        diagnostic_gradient_entries = realtime_geometry_transport_reverse_diagnostic_gradient_entries(
            objective_labels=OBJECTIVE_LABELS,
            geometry_parameter_labels=geometry_param_labels,
            geometry_gradient_matrix_np=geometry_gradient_np,
            geometry_branch_gradient_matrix_np=geometry_branch_gradient_np,
            ntx_support_branch_gradient_matrix_np=ntx_support_branch_gradient_np,
            component_gradient_np_by_name=component_gradient_np_by_name,
            component_geometry_branch_np_by_name=component_geometry_branch_np_by_name,
            component_ntx_support_branch_np_by_name=component_ntx_support_branch_np_by_name,
            include_component_pullbacks=include_component_pullbacks,
        )
        report = {
            **metadata_entries,
            **table_report_entries,
            **diagnostic_gradient_entries,
        }
        if return_report:
            return {
                **report,
                "transport_reverse_table_result": table_result,
            }
        print(
            "[autodiff-gate] mode=transport_reverse_ad_only "
            "parameter_mode=profiles_plus_realtime_geometry "
            f"realtime_geometry_gradient_path={args.realtime_geometry_gradient_path} "
            "objective=all "
            "realtime_geometry_support_bwd_mode=grouped_batched_fused_support "
            f"reverse_stage_cotangent_mode_effective={support_probe_cotangent_mode} "
            f"support_reuse_count={support_reuse_count} "
            f"support_rebuild_count={support_rebuild_count} "
            f"support_initial_cache_pullback_used={initial_cache_pullback_used} "
            f"support_initial_cache_pullback_skipped={initial_cache_pullback_skipped} "
            f"elapsed_s={elapsed_s:.3f}",
            flush=True,
        )
        print("[autodiff-gate] objective values:")
        for objective_name, value in report["objective_values"].items():
            print(f"  - {objective_name}: value={value:.16e}")
        print("[autodiff-gate] realtime geometry diagnostics:")
        for field_name in ("a_b", "R0", "r_grid", "Vprime", "Vprime_half", "overVprime", "integrated_volume"):
            if field_name not in realtime_geometry_diagnostics:
                continue
            summary = realtime_geometry_diagnostics[field_name]
            value_suffix = ""
            if field_name == "integrated_volume":
                value_suffix = f" value={realtime_geometry_diagnostics['integrated_volume_value']:.16e}"
            print(
                f"  - {field_name}: all_finite={summary['all_finite']} "
                f"nan_count={summary['nan_count']} "
                f"finite_min={summary['finite_min']} "
                f"finite_max={summary['finite_max']} "
                f"first_nonfinite_index={summary['first_nonfinite_index']}"
                f"{value_suffix}"
            )
        print("[autodiff-gate] reverse profile gradients by objective:")
        for objective_name in OBJECTIVE_LABELS:
            print(
                f"  - {objective_name}: "
                f"objective_finite={report['objective_finite'][objective_name]} "
                f"profile_gradient_all_finite="
                f"{report['profile_gradient_all_finite_by_objective'][objective_name]}"
            )
            for parameter_name in PARAMETER_ORDER:
                value = report["profile_gradient_reverse_ad"][objective_name][parameter_name]
                print(f"      d{objective_name}/d{parameter_name}: ad={value:.6e}")
        print("[autodiff-gate] reverse geometry gradients by objective:")
        print(
            "[autodiff-gate] reverse geometry parameter count: "
            f"{len(geometry_param_labels)}",
            flush=True,
        )
        for objective_name in OBJECTIVE_LABELS:
            values_by_parameter = report["geometry_gradient_reverse_ad"][objective_name]
            finite = report["geometry_gradient_all_finite_by_objective"][objective_name]
            print(f"  - {objective_name}: geometry_gradient_all_finite={finite}")
            if len(geometry_param_labels) <= int(args.reverse_geometry_print_limit):
                for geometry_label in geometry_param_labels:
                    value = values_by_parameter[geometry_label]
                    branch_suffix = ""
                    branch_payload = report["geometry_gradient_reverse_ad_by_branch"]
                    if branch_payload is not None:
                        geometry_value = branch_payload[objective_name]["geometry"][geometry_label]
                        ntx_value = branch_payload[objective_name]["ntx_support"][geometry_label]
                        branch_suffix = (
                            f" geometry_branch={geometry_value:.6e} "
                            f"ntx_support_branch={ntx_value:.6e}"
                        )
                    print(f"      d{objective_name}/d{geometry_label}: ad={value:.6e}{branch_suffix}")
                    component_payload = report["geometry_gradient_reverse_ad_by_component"]
                    if component_payload:
                        component_parts = [
                            f"{component_name}={component_payload[objective_name][component_name][geometry_label]:.6e}"
                            for component_name in support_component_names
                        ]
                        print("        components: " + " ".join(component_parts))
                        component_branch_payload = report.get(
                            "geometry_gradient_reverse_ad_by_component_and_branch",
                            {},
                        )
                        if component_branch_payload:
                            for component_name in support_component_names:
                                branch_values = component_branch_payload[objective_name][component_name]
                                geometry_component = branch_values["geometry"][geometry_label]
                                ntx_component = branch_values["ntx_support"][geometry_label]
                                print(
                                    "        component_branches: "
                                    f"{component_name}.geometry={geometry_component:.6e} "
                                    f"{component_name}.ntx_support={ntx_component:.6e}"
                                )
                        final_state_components = report.get(
                            "geometry_gradient_reverse_ad_final_state_components",
                            {},
                        )
                        if objective_name in final_state_components:
                            dynamic_value = final_state_components[objective_name][geometry_label]
                            print(
                                "        final_state_components_sum="
                                f"{dynamic_value:.6e} "
                                "(compare to FD fd_final_state_geometry)"
                            )
            else:
                objective_values_arr = geometry_gradient_np[OBJECTIVE_LABELS.index(objective_name)]
                top_k = int(max(1, args.reverse_geometry_print_top_k))
                top_k = min(top_k, len(geometry_param_labels))
                order = np.argsort(-np.abs(objective_values_arr))[:top_k]
                print(
                    f"      printing top {top_k} |gradient| entries "
                    f"(full table is in JSON)",
                    flush=True,
                )
                for param_i in order.tolist():
                    geometry_label = geometry_param_labels[int(param_i)]
                    value = values_by_parameter[geometry_label]
                    branch_suffix = ""
                    branch_payload = report["geometry_gradient_reverse_ad_by_branch"]
                    if branch_payload is not None:
                        geometry_value = branch_payload[objective_name]["geometry"][geometry_label]
                        ntx_value = branch_payload[objective_name]["ntx_support"][geometry_label]
                        branch_suffix = (
                            f" geometry_branch={geometry_value:.6e} "
                            f"ntx_support_branch={ntx_value:.6e}"
                        )
                    print(f"      d{objective_name}/d{geometry_label}: ad={value:.6e}{branch_suffix}")
                    component_payload = report["geometry_gradient_reverse_ad_by_component"]
                    if component_payload:
                        component_parts = [
                            f"{component_name}={component_payload[objective_name][component_name][geometry_label]:.6e}"
                            for component_name in support_component_names
                        ]
                        print("        components: " + " ".join(component_parts))
                        component_branch_payload = report.get(
                            "geometry_gradient_reverse_ad_by_component_and_branch",
                            {},
                        )
                        if component_branch_payload:
                            for component_name in support_component_names:
                                branch_values = component_branch_payload[objective_name][component_name]
                                geometry_component = branch_values["geometry"][geometry_label]
                                ntx_component = branch_values["ntx_support"][geometry_label]
                                print(
                                    "        component_branches: "
                                    f"{component_name}.geometry={geometry_component:.6e} "
                                    f"{component_name}.ntx_support={ntx_component:.6e}"
                                )
                        final_state_components = report.get(
                            "geometry_gradient_reverse_ad_final_state_components",
                            {},
                        )
                        if objective_name in final_state_components:
                            dynamic_value = final_state_components[objective_name][geometry_label]
                            print(
                                "        final_state_components_sum="
                                f"{dynamic_value:.6e} "
                                "(compare to FD fd_final_state_geometry)"
                            )
        if skip_support_bar_diagnostics:
            print(
                "[autodiff-gate] realtime support/geometry-payload cotangent diagnostics skipped",
                flush=True,
            )
        else:
            print("[autodiff-gate] realtime support/geometry-payload cotangents by objective:")
            for objective_name in OBJECTIVE_LABELS:
                summary = support_bar_summary_by_objective[objective_name]
                print(
                    f"  - {objective_name}: "
                    f"support_bar_l2={support_bar_l2_by_objective[objective_name]:.6e} "
                    f"support_bar_array_leaves={summary['n_array_leaves']} "
                    f"support_bar_all_finite={summary['all_floating_leaves_finite']}"
                )
                branch_diagnostics = support_bar_branch_diagnostics_by_objective[objective_name]
                for branch_name, branch_summary in branch_diagnostics.items():
                    branch_leaf_summary = branch_summary["summary"]
                    nonfinite_leaves = branch_summary["first_nonfinite_leaves"]
                    first_nonfinite = None if not nonfinite_leaves else nonfinite_leaves[0]["path"]
                    print(
                        f"      {branch_name}: "
                        f"l2={branch_summary['l2']:.6e} "
                        f"array_leaves={branch_leaf_summary['n_array_leaves']} "
                        f"all_finite={branch_leaf_summary['all_floating_leaves_finite']} "
                        f"first_nonfinite_leaf={first_nonfinite}"
                    )
        outpath = _report_path("realtime_geometry_support_segment")
        outpath.write_text(json.dumps(report, indent=2))
        print(f"Wrote {outpath.relative_to(ROOT)}")
        return

    objective_name = str(args.objective)
    objective_index = OBJECTIVE_LABELS.index(objective_name)
    print(
        "[autodiff-gate] progress: probing realized-schedule reverse support payload cotangent "
        f"for {objective_name}",
        flush=True,
    )
    t_start = time.perf_counter()
    (
        objective_value,
        profile_bar,
        support_bar,
        support_reuse_count,
        support_rebuild_count,
        initial_cache_pullback_used,
        initial_cache_pullback_skipped,
    ) = _reverse_objective_support_payload_bar_for_parameter_vector(
        profile_values,
        runtime=baseline_runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        reverse_setup=reverse_setup,
        objective_index=objective_index,
        support_payload=support_payload,
    )
    objective_value, profile_bar, support_bar = jax.block_until_ready(
        (objective_value, profile_bar, support_bar)
    )
    elapsed_s = time.perf_counter() - t_start
    support_summary = _payload_leaf_summary(support_payload)
    support_bar_summary = _payload_leaf_summary(support_bar)
    support_bar_l2 = _tree_array_l2_norm(support_bar)
    profile_bar_np = np.asarray(jax.device_get(profile_bar), dtype=float)
    report = {
        "mode": "transport_reverse_ad_only",
        "parameter_mode": str(args.reverse_parameter_mode),
        "config_path": str(Path(args.config)),
        "objective_name": objective_name,
        "parameter_order": list(PARAMETER_ORDER),
        "profile_baseline_values": np.asarray(jax.device_get(profile_values), dtype=float).tolist(),
        "objective_value": float(np.asarray(jax.device_get(objective_value), dtype=float)),
        "profile_gradient_reverse_ad": {
            name: float(value) for name, value in zip(PARAMETER_ORDER, profile_bar_np.tolist())
        },
        "accepted_step_limit": None if args.accepted_step_limit is None else int(args.accepted_step_limit),
        "reverse_segment_length": None if args.reverse_segment_length is None else int(args.reverse_segment_length),
        "reverse_stage_cotangent_mode_requested": str(args.reverse_stage_cotangent_mode),
        "reverse_stage_cotangent_mode_effective": support_probe_cotangent_mode,
        "ntx_exact_derivative_mode": str(args.ntx_exact_derivative_mode),
        "ntx_exact_derivative_field_pullback_mode": str(args.ntx_exact_derivative_field_pullback_mode),
        "ntx_exact_surface_backend": str(neoclassical_cfg.get("ntx_exact_surface_backend", "booz")),
        "realtime_geometry_gradient_path": str(args.realtime_geometry_gradient_path),
        "realtime_primal_runtime_builder": "build_runtime_context",
        "realtime_geometry_derivative_boundary": (
            "runtime_geometry_and_ntx_exact_lij_support_payload"
            if combined_geometry_payload
            else "ntx_exact_lij_support_payload_only_diagnostic"
        ),
        "realtime_geometry_derivative_complete": bool(combined_geometry_payload),
        "support_payload_summary": support_summary,
        "support_bar_summary": support_bar_summary,
        "support_bar_l2": support_bar_l2,
        "support_reuse_count": int(support_reuse_count),
        "support_rebuild_count": int(support_rebuild_count),
        "support_initial_cache_pullback_used": bool(initial_cache_pullback_used),
        "support_initial_cache_pullback_skipped": bool(initial_cache_pullback_skipped),
        "elapsed_s": float(elapsed_s),
    }
    print(
        "[autodiff-gate] mode=transport_reverse_ad_only "
        "parameter_mode=profiles_plus_realtime_geometry "
        f"realtime_geometry_gradient_path={args.realtime_geometry_gradient_path} "
        f"objective={objective_name} "
        f"reverse_stage_cotangent_mode_effective={support_probe_cotangent_mode} "
        f"value={report['objective_value']:.16e} "
        f"support_bar_l2={support_bar_l2:.6e} "
        f"support_reuse_count={support_reuse_count} "
        f"support_rebuild_count={support_rebuild_count} "
        f"support_initial_cache_pullback_used={initial_cache_pullback_used} "
        f"support_initial_cache_pullback_skipped={initial_cache_pullback_skipped} "
        f"support_bar_array_leaves={support_bar_summary['n_array_leaves']} "
        f"support_bar_all_finite={support_bar_summary['all_floating_leaves_finite']} "
        f"elapsed_s={elapsed_s:.3f}",
        flush=True,
    )
    print("[autodiff-gate] profile-gradient sidecar from same reverse pass:")
    for parameter_name in PARAMETER_ORDER:
        print(
            f"  - {parameter_name}: ad={report['profile_gradient_reverse_ad'][parameter_name]:.6e}"
        )
    outpath = _report_path("realtime_geometry_support_segment")
    outpath.write_text(json.dumps(report, indent=2))
    print(f"Wrote {outpath.relative_to(ROOT)}")


def _make_realtime_geometry_support_segment_builder_inputs(
    *,
    args,
    config: dict[str, Any],
    baseline_values,
    baseline_runtime,
    baseline_state,
    profile_cfg: dict,
    neoclassical_cfg: dict[str, Any],
):
    """Return the shared grouped-runner inputs for transport optimization builders."""

    support_segment_executor = realtime_geometry_transport_reverse_support_segment_executor(
        support_segment_probe=_run_realtime_geometry_support_segment_probe,
        config=config,
        baseline_values=baseline_values,
        baseline_runtime=baseline_runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        neoclassical_cfg=neoclassical_cfg,
    )

    grouped_inputs = realtime_geometry_transport_reverse_grouped_inputs(
        args=args,
        config=config,
        baseline_values=baseline_values,
        baseline_runtime=baseline_runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        neoclassical_cfg=neoclassical_cfg,
        support_segment_executor=support_segment_executor,
    )
    return grouped_inputs.table_context, grouped_inputs.run_grouped_report


def _make_realtime_geometry_support_segment_report_builder(
    *,
    args,
    config: dict[str, Any],
    baseline_values,
    baseline_runtime,
    baseline_state,
    profile_cfg: dict,
    neoclassical_cfg: dict[str, Any],
):
    """Return a grouped transport reverse report builder for optimization adapters."""

    table_context, run_grouped_report = _make_realtime_geometry_support_segment_builder_inputs(
        args=args,
        config=config,
        baseline_values=baseline_values,
        baseline_runtime=baseline_runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        neoclassical_cfg=neoclassical_cfg,
    )

    return grouped_transport_reverse_report_builder(
        objective_labels=OBJECTIVE_LABELS,
        run_grouped_report=run_grouped_report,
        table_context=table_context,
    )


def _make_realtime_geometry_support_segment_table_result_builder(
    *,
    args,
    config: dict[str, Any],
    baseline_values,
    baseline_runtime,
    baseline_state,
    profile_cfg: dict,
    neoclassical_cfg: dict[str, Any],
):
    """Return a grouped transport reverse table-result builder for optimization adapters."""

    table_context, run_grouped_report = _make_realtime_geometry_support_segment_builder_inputs(
        args=args,
        config=config,
        baseline_values=baseline_values,
        baseline_runtime=baseline_runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        neoclassical_cfg=neoclassical_cfg,
    )

    return grouped_transport_reverse_table_result_builder(
        objective_labels=OBJECTIVE_LABELS,
        run_grouped_report=run_grouped_report,
        table_context=table_context,
    )


def _run_realtime_geometry_optimization_api_smoke(
    *,
    args,
    config: dict[str, Any],
    geometry_context,
    baseline_values,
    baseline_runtime,
    baseline_state,
    profile_cfg: dict,
    neoclassical_cfg: dict[str, Any],
):
    """Exercise the production-style transport least-squares API on the benchmark path."""

    geometry_param_specs = _geometry_param_specs_from_args(args, geometry_context)
    include_profile_dofs = str(getattr(args, "optimization_api_profile_dofs", "include")) == "include"
    profile_parameter_order = (
        PROFILE_PARAMETER_ORDER
        if bool(getattr(args, "full_transport_shared_payload_smoke", False))
        else PARAMETER_ORDER
    )
    parameter_set = reverse_ad_optimization_parameter_set(
        include_profiles=include_profile_dofs,
        profiles=profile_parameter_order if include_profile_dofs else None,
        vmec_boundary=tuple(
            VmecBoundaryParameterSpec(family, m, n)
            for family, m, n in geometry_param_specs
        ),
    )
    objective_labels = (
        TRANSPORT_REVERSE_OBJECTIVE_LABELS
        if bool(getattr(args, "full_transport_shared_payload_smoke", False))
        else tuple(OBJECTIVE_LABELS)
    )
    objective_names = (
        objective_labels
        if str(args.objective) == "all"
        else (str(args.objective),)
    )
    terms = transport_least_squares_terms(objective_names)
    if bool(getattr(args, "full_transport_shared_payload_smoke", False)):
        terms = tuple(terms) + (
            LeastSquaresTerm(geometry_objectives.boozer_qi_objective),
            LeastSquaresTerm(geometry_objectives.boozer_maxj_objective),
            LeastSquaresTerm(geometry_objectives.vmec_aspect_ratio),
            LeastSquaresTerm(geometry_objectives.vmec_iota_mean),
            LeastSquaresTerm(geometry_objectives.vmec_magnetic_well),
            LeastSquaresTerm(geometry_objectives.vmec_mirror_ratio),
        )
    if bool(getattr(args, "full_transport_shared_payload_smoke", False)):
        geom_cfg_for_context = config.get("geometry", {})
        context_geometry_values0 = _baseline_geometry_delta_vector_for_specs(
            geom_cfg_for_context,
            geometry_param_specs,
        )
        context_profile_values0 = jnp.asarray(
            [
                _profile_cfg_scalar_value(profile_cfg, name)
                for name in PROFILE_PARAMETER_ORDER
            ],
            dtype=jnp.asarray(baseline_state.pressure).dtype,
        )
        context_baseline_values = jnp.concatenate(
            [
                context_profile_values0,
                jnp.asarray(context_geometry_values0, dtype=context_profile_values0.dtype),
            ],
            axis=0,
        )
    else:
        context_baseline_values = baseline_values
    table_context, run_grouped_report = _make_realtime_geometry_support_segment_builder_inputs(
        args=args,
        config=config,
        baseline_values=context_baseline_values,
        baseline_runtime=baseline_runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        neoclassical_cfg=neoclassical_cfg,
    )
    print(
        "[autodiff-gate] progress: running realtime geometry optimization API smoke",
        flush=True,
    )
    if bool(getattr(args, "full_transport_shared_payload_smoke", False)):
        print(
            "[autodiff-gate] progress: full-transport reverse stage-adjoint "
            f"solve mode={args.reverse_stage_adjoint_solve_mode} "
            f"rhs_pullback_mode={args.reverse_rhs_pullback_mode} "
            f"initial_cache_support_pullback_mode={args.reverse_initial_cache_support_pullback_mode} "
            f"rebuild_support_pullback_mode={args.reverse_rebuild_support_pullback_mode} "
            f"segment_jit_diagnostics={args.reverse_segment_jit_diagnostics} "
            f"segment_input_diagnostics={args.reverse_segment_input_diagnostics} "
            f"segment_start_replay_mode={args.reverse_segment_start_replay_mode} "
            f"segment_primal_record_mode={args.reverse_segment_primal_record_mode} "
            f"step_bwd_mode={args.reverse_step_bwd_mode}",
            flush=True,
        )
        geom_cfg = config.get("geometry", {})
        neoclassical_cfg = config.get("neoclassical", {})
        geometry_values0 = _baseline_geometry_delta_vector_for_specs(
            geom_cfg,
            geometry_param_specs,
        )
        profile_values0 = jnp.asarray(
            [
                _profile_cfg_scalar_value(profile_cfg, spec.name)
                for spec in parameter_set.profile_specs
            ],
            dtype=jnp.asarray(baseline_state.pressure).dtype,
        )
        parameter_values = jnp.asarray(
            (
                list(np.asarray(profile_values0, dtype=float))
                if include_profile_dofs
                else []
            )
            + list(np.asarray(geometry_values0, dtype=float)),
            dtype=jnp.asarray(baseline_state.pressure).dtype,
        )
        table_result_builder = internal_realtime_geometry_transport_reverse_table_result_builder(
            table_context=table_context,
            geometry_context=geometry_context,
            baseline_geometry_deltas=geometry_values0,
            combined_geometry_payload=True,
            n_r=int(geom_cfg.get("n_radial", 51)),
            n_theta=int(neoclassical_cfg.get("ntx_exact_n_theta", 25)),
            n_zeta=int(neoclassical_cfg.get("ntx_exact_n_zeta", 25)),
            n_xi=int(neoclassical_cfg.get("ntx_exact_n_xi", 64)),
            surface_backend=str(
                neoclassical_cfg.get(
                    "ntx_exact_surface_backend",
                    neoclassical_cfg.get("ntx_surface_backend", "vmec"),
                )
            ),
            max_iter=geom_cfg.get("vmec_max_iter"),
            solver_device=str(geom_cfg.get("vmec_implicit_solver_device", "default")),
            accepted_step_limit=args.accepted_step_limit,
            reverse_segment_length=args.reverse_segment_length,
            initial_er_root_ad=str(args.initial_er_root_ad),
            reverse_stage_adjoint_solve_mode=str(args.reverse_stage_adjoint_solve_mode),
            reverse_rhs_transpose_mode=str(args.reverse_rhs_transpose_mode),
            reverse_rhs_pullback_mode=str(args.reverse_rhs_pullback_mode),
            reverse_initial_cache_support_pullback_mode=str(
                args.reverse_initial_cache_support_pullback_mode
            ),
            reverse_rebuild_support_pullback_mode=str(
                args.reverse_rebuild_support_pullback_mode
            ),
            reverse_segment_jit_diagnostics=bool(args.reverse_segment_jit_diagnostics),
            reverse_segment_input_diagnostics=bool(args.reverse_segment_input_diagnostics),
            reverse_rebuild_component_timing=bool(args.reverse_rebuild_component_timing),
            reverse_segment_profile_annotations=bool(
                args.reverse_segment_profiler_trace_dir
            ),
            reverse_segment_start_replay_mode=str(args.reverse_segment_start_replay_mode),
            reverse_segment_primal_record_mode=str(args.reverse_segment_primal_record_mode),
            reverse_final_objective_cotangent_mode=str(
                args.reverse_final_objective_cotangent_mode
            ),
            reverse_step_bwd_mode=str(args.reverse_step_bwd_mode),
            reverse_schedule_artifact_mode=str(args.reverse_schedule_artifact_mode),
            max_reverse_accepted_steps=(
                None
                if args.max_reverse_accepted_steps is None
                else int(args.max_reverse_accepted_steps)
            ),
            progress_label="[autodiff-gate] full-transport shared payload:",
        )
        request = realtime_geometry_transport_reverse_table_request(
            objective_names=objective_names,
            parameter_set=parameter_set,
            context=table_context,
            options={"quiet": True},
        )
        with _maybe_reverse_segment_profiler_trace(
            args.reverse_segment_profiler_trace_dir
        ):
            evaluation = evaluate_geometry_transport_realtime_geometry_least_squares(
                config,
                request=request,
                terms=terms,
                geometry_context=geometry_context,
                parameter_values=parameter_values,
                table_result_builder=table_result_builder,
                objective_labels=objective_labels,
                options={
                    "quiet": True,
                    "reverse_table_timing_diagnostics": bool(
                        args.reverse_table_timing_diagnostics
                    ),
                },
                quiet_default=True,
                geometry_max_iter=geom_cfg.get("vmec_max_iter"),
                geometry_solver_device=str(geom_cfg.get("vmec_implicit_solver_device", "default")),
            )
    else:
        runner = build_transport_realtime_geometry_least_squares_runner(
            config,
            objective_names=objective_names,
            parameter_set=parameter_set,
            table_context=table_context,
            run_grouped_report=run_grouped_report,
            objective_labels=objective_labels,
            options={"quiet": True},
        )
        evaluation = runner(terms)
    result = evaluation.result
    residuals = evaluation.residuals
    jacobian = evaluation.jacobian
    elapsed_s = evaluation.elapsed_s
    residuals_np = np.asarray(jax.device_get(residuals), dtype=float)
    jacobian_np = np.asarray(jax.device_get(jacobian), dtype=float)
    objective_values_np = {
        label: float(np.asarray(jax.device_get(value), dtype=float))
        for label, value in result.objective_values.items()
    }
    report = {
        "mode": "transport_reverse_ad_only_full_transport_shared_payload_smoke"
        if bool(getattr(args, "full_transport_shared_payload_smoke", False))
        else "transport_reverse_ad_only_optimization_api_smoke",
        "parameter_mode": str(args.reverse_parameter_mode),
        "config_path": str(Path(args.config)),
        "objective_name": str(args.objective),
        "objective_order": list(objective_names),
        "residual_labels": list(result.residual_labels),
        "parameter_order": list(result.parameter_labels),
        "objective_values": objective_values_np,
        "residuals": residuals_np.tolist(),
        "jacobian": jacobian_np.tolist(),
        "optimization_api_profile_dofs": str(getattr(args, "optimization_api_profile_dofs", "include")),
        "accepted_step_limit": None if args.accepted_step_limit is None else int(args.accepted_step_limit),
        "reverse_segment_length": None if args.reverse_segment_length is None else int(args.reverse_segment_length),
        "realtime_geometry_gradient_path": str(args.realtime_geometry_gradient_path),
        "initial_er_root_ad": str(args.initial_er_root_ad),
        "reverse_rhs_pullback_mode": str(args.reverse_rhs_pullback_mode),
        "reverse_segment_start_replay_mode": str(args.reverse_segment_start_replay_mode),
        "reverse_segment_primal_record_mode": str(args.reverse_segment_primal_record_mode),
        "reverse_table_timing_diagnostics": bool(args.reverse_table_timing_diagnostics),
        "reverse_segment_profiler_trace_dir": args.reverse_segment_profiler_trace_dir,
        "shared_payload_smoke": bool(getattr(args, "full_transport_shared_payload_smoke", False)),
        "shared_payload_note": (
            "Full transport shared-path smoke uses the internal realtime-geometry "
            "transport table-result builder once and writes JSON for offline "
            "comparison against saved reference benchmark output."
        ),
        "elapsed_s": float(elapsed_s),
    }
    print(
        "[autodiff-gate] mode="
        f"{report['mode']} "
        f"objective={args.objective} "
        f"residual_count={len(result.residual_labels)} "
        f"parameter_count={len(result.parameter_labels)} "
        f"elapsed_s={elapsed_s:.3f}",
        flush=True,
    )
    print("[autodiff-gate] optimization API residuals/Jacobian rows:")
    for row_i, label in enumerate(result.residual_labels):
        print(f"  - {label}: residual={residuals_np[row_i]:.16e}")
        for parameter_name, value in zip(result.parameter_labels, jacobian_np[row_i].tolist()):
            print(f"      d{label}/d{parameter_name}: jac={value:.16e}")
    outpath = _report_path(
        "full_transport_shared_payload_smoke"
        if bool(getattr(args, "full_transport_shared_payload_smoke", False))
        else "optimization_api_smoke"
    )
    outpath.write_text(json.dumps(report, indent=2))
    print(f"Wrote {outpath.relative_to(ROOT)}")


def _run_initial_er_root_only_optimization_api_smoke(
    *,
    args,
    config: dict[str, Any],
    baseline_runtime,
    baseline_state,
    profile_cfg: dict,
    geometry_context=None,
    neoclassical_cfg: dict[str, Any] | None = None,
):
    """Exercise initial ambipolar-Er objective optimization without time rollout."""

    if not _initial_er_root_enabled(config, str(args.initial_er_root_ad)):
        raise SystemExit(
            "[autodiff-gate] --initial-Er-root-only-optimization-smoke requires "
            "an ambipolar Er initialization mode in the TOML and "
            "--initial-Er-root-ad jax_selected_root."
        )
    include_profile_dofs = str(getattr(args, "optimization_api_profile_dofs", "include")) == "include"
    include_geometry_dofs = str(args.reverse_parameter_mode) == "profiles_plus_realtime_geometry"
    if str(args.objective) == "all":
        objective_names = (
            (*INITIAL_ER_ROOT_ONLY_OBJECTIVES, "bootstrap_current_softmax_abs_scaled")
            if include_geometry_dofs
            else INITIAL_ER_ROOT_ONLY_OBJECTIVES
        )
    else:
        objective_names = (str(args.objective),)
    unsupported = tuple(name for name in objective_names if name not in INITIAL_ER_ROOT_ONLY_EXPLICIT_OBJECTIVES)
    if unsupported:
        allowed = ", ".join(INITIAL_ER_ROOT_ONLY_EXPLICIT_OBJECTIVES)
        raise SystemExit(
            "[autodiff-gate] initial-Er root-only smoke supports only Er objectives; "
            f"unsupported={unsupported!r}, choices are: {allowed}."
        )
    if "bootstrap_current_softmax_abs_scaled" in objective_names and not include_geometry_dofs:
        raise SystemExit(
            "[autodiff-gate] bootstrap_current_softmax_abs_scaled currently requires the "
            "geometry-active compact root-only path. Use "
            "--reverse-parameter-mode profiles_plus_realtime_geometry and "
            "--reverse-geometry-parameter ..."
        )
    geometry_param_specs = (
        _geometry_param_specs_from_args(args, geometry_context)
        if include_geometry_dofs
        else ()
    )
    parameter_set = reverse_ad_optimization_parameter_set(
        include_profiles=include_profile_dofs,
        profiles=PARAMETER_ORDER if include_profile_dofs else None,
        vmec_boundary=tuple(
            VmecBoundaryParameterSpec(family, m, n)
            for family, m, n in geometry_param_specs
        ),
    )
    if not parameter_set.specs:
        raise SystemExit(
            "[autodiff-gate] initial-Er root-only smoke has no active optimization parameters. "
            "Use profile DOFs, realtime geometry DOFs, or both."
        )
    geom_cfg = config.get("geometry", {})
    profile_values0 = jnp.asarray(
        [
            _profile_cfg_scalar_value(profile_cfg, spec.name)
            for spec in parameter_set.profile_specs
        ],
        dtype=jnp.asarray(baseline_state.pressure).dtype,
    )
    geometry_values0 = (
        _baseline_geometry_delta_vector_for_specs(geom_cfg, geometry_param_specs)
        if include_geometry_dofs
        else jnp.asarray((), dtype=jnp.float64)
    )
    parameter_values = jnp.asarray(
        (
            list(np.asarray(profile_values0, dtype=float))
            if include_profile_dofs
            else []
        )
        + (
            list(np.asarray(geometry_values0, dtype=float))
            if include_geometry_dofs
            else []
        ),
        dtype=jnp.asarray(baseline_state.pressure).dtype,
    )
    neoclassical_cfg = {} if neoclassical_cfg is None else neoclassical_cfg

    def _rooted_state_from_profile_values(profile_values):
        return _initial_state_for_parameter_vector(
            profile_values,
            config=config,
            initial_er_root_ad=args.initial_er_root_ad,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            runtime=baseline_runtime,
        )

    terms = transport_least_squares_terms(objective_names)
    print(
        "[autodiff-gate] progress: running initial-Er root-only optimization API smoke",
        flush=True,
    )
    shared_compare_report = None
    if not include_geometry_dofs:
        runner = build_initial_er_root_only_least_squares_runner(
            config,
            runtime=baseline_runtime,
            parameter_set=parameter_set,
            rooted_state_from_parameter_vector=_rooted_state_from_profile_values,
            objective_names=objective_names,
        )
        evaluation = runner(parameter_values, terms)
    else:
        baseline_geometry_deltas = _baseline_geometry_delta_vector_for_specs(
            geom_cfg,
            geometry_param_specs,
        )

        def _pre_root_state_from_profile_values(profile_values):
            return _initial_state_for_parameter_vector(
                profile_values,
                config=config,
                initial_er_root_ad="off",
                baseline_state=baseline_state,
                profile_cfg=profile_cfg,
                runtime=baseline_runtime,
            )

        if bool(getattr(args, "initial_er_root_shared_payload_compare_smoke", False)):
            print(
                "[autodiff-gate] progress: running initial-Er shared-payload root-only smoke",
                flush=True,
            )
            mixed_terms = tuple(terms) + (
                LeastSquaresTerm(geometry_objectives.boozer_qi_objective),
                LeastSquaresTerm(geometry_objectives.boozer_maxj_objective),
                LeastSquaresTerm(geometry_objectives.vmec_aspect_ratio),
                LeastSquaresTerm(geometry_objectives.vmec_iota_mean),
                LeastSquaresTerm(geometry_objectives.vmec_magnetic_well),
                LeastSquaresTerm(geometry_objectives.vmec_mirror_ratio),
            )
            evaluation = evaluate_geometry_initial_er_root_only_least_squares_benchmark_tables(
                config,
                parameter_set=parameter_set,
                parameter_values=parameter_values,
                terms=mixed_terms,
                geometry_context=geometry_context,
                runtime=baseline_runtime,
                baseline_profile_values=profile_values0,
                pre_root_state_from_profile_values=_pre_root_state_from_profile_values,
                n_r=int(geom_cfg.get("n_radial", 51)),
                n_theta=int(neoclassical_cfg.get("ntx_exact_n_theta", 25)),
                n_zeta=int(neoclassical_cfg.get("ntx_exact_n_zeta", 25)),
                n_xi=int(neoclassical_cfg.get("ntx_exact_n_xi", 64)),
                surface_backend=str(
                    neoclassical_cfg.get(
                        "ntx_exact_surface_backend",
                        neoclassical_cfg.get("ntx_surface_backend", "vmec"),
                    )
                ),
                geometry_max_iter=geom_cfg.get("vmec_max_iter"),
                geometry_solver_device=str(geom_cfg.get("vmec_implicit_solver_device", "default")),
            )
            result = evaluation.result
            residuals_np = np.asarray(jax.device_get(evaluation.residuals), dtype=float)
            jacobian_np = np.asarray(jax.device_get(evaluation.jacobian), dtype=float)
            objective_values_np = {
                label: float(np.asarray(jax.device_get(value), dtype=float))
                for label, value in result.objective_values.items()
            }
            report = {
                "mode": "transport_reverse_ad_only_initial_er_root_shared_payload_smoke",
                "config_path": str(Path(args.config)),
                "objective_name": str(args.objective),
                "objective_order": list(objective_names),
                "mixed_residual_labels": list(result.residual_labels),
                "parameter_order": list(result.parameter_labels),
                "optimization_api_profile_dofs": str(getattr(args, "optimization_api_profile_dofs", "include")),
                "geometry_parameter_specs": [_format_geometry_param_spec(spec) for spec in geometry_param_specs],
                "objective_values": objective_values_np,
                "residuals": residuals_np.tolist(),
                "jacobian": jacobian_np.tolist(),
                "initial_er_root_ad": str(args.initial_er_root_ad),
                "elapsed_s": float(evaluation.elapsed_s),
            }
            print(
                "[autodiff-gate] mode=transport_reverse_ad_only_initial_er_root_shared_payload_smoke "
                f"objective={args.objective} "
                f"residual_count={len(result.residual_labels)} "
                f"parameter_count={len(result.parameter_labels)} "
                f"elapsed_s={evaluation.elapsed_s:.3f}",
                flush=True,
            )
            print("[autodiff-gate] initial-Er shared-payload residuals/Jacobian rows:")
            for row_i, label in enumerate(result.residual_labels):
                print(f"  - {label}: residual={residuals_np[row_i]:.16e}")
                for parameter_name, value in zip(result.parameter_labels, jacobian_np[row_i].tolist()):
                    print(f"      d{label}/d{parameter_name}: jac={value:.16e}")
            outpath = _report_path("initial_er_root_shared_payload_smoke")
            outpath.write_text(json.dumps(report, indent=2))
            print(f"Wrote {outpath.relative_to(ROOT)}")
            return

        table_result = geometry_active_initial_er_root_only_reverse_table(
            config=config,
            objective_names=objective_names,
            parameter_set=parameter_set,
            parameter_values=parameter_values,
            runtime=baseline_runtime,
            profile_values=profile_values0,
            pre_root_state_from_profile_values=_pre_root_state_from_profile_values,
            geometry_context=geometry_context,
            baseline_geometry_deltas=baseline_geometry_deltas,
            n_r=int(geom_cfg.get("n_radial", 51)),
            n_theta=int(neoclassical_cfg.get("ntx_exact_n_theta", 25)),
            n_zeta=int(neoclassical_cfg.get("ntx_exact_n_zeta", 25)),
            n_xi=int(neoclassical_cfg.get("ntx_exact_n_xi", 64)),
            surface_backend=str(
                neoclassical_cfg.get(
                    "ntx_exact_surface_backend",
                    neoclassical_cfg.get("ntx_surface_backend", "vmec"),
                )
            ),
            max_iter=geom_cfg.get("vmec_max_iter"),
            solver_device=str(geom_cfg.get("vmec_implicit_solver_device", "default")),
            progress_label="[autodiff-gate] initial-Er root-only geometry payload pullback:",
        )
        t_eval = time.perf_counter()
        result = residuals_and_jacobian_reverse_ad(
            config,
            parameter_set=parameter_set,
            terms=terms,
            backends={"transport": lambda _names, _parameter_set, _options: table_result},
        )
        residuals = jax.block_until_ready(result.residuals)
        jacobian = jax.block_until_ready(result.jacobian)
        evaluation = LeastSquaresEvaluation(
            result=result,
            residuals=residuals,
            jacobian=jacobian,
            elapsed_s=float(time.perf_counter() - t_eval),
        )
    result = evaluation.result
    residuals_np = np.asarray(jax.device_get(evaluation.residuals), dtype=float)
    jacobian_np = np.asarray(jax.device_get(evaluation.jacobian), dtype=float)
    objective_values_np = {
        label: float(np.asarray(jax.device_get(value), dtype=float))
        for label, value in result.objective_values.items()
    }
    report = {
        "mode": "transport_reverse_ad_only_initial_er_root_only_optimization_api_smoke",
        "config_path": str(Path(args.config)),
        "objective_name": str(args.objective),
        "objective_order": list(objective_names),
        "residual_labels": list(result.residual_labels),
        "parameter_order": list(result.parameter_labels),
        "optimization_api_profile_dofs": str(getattr(args, "optimization_api_profile_dofs", "include")),
        "geometry_parameter_specs": [_format_geometry_param_spec(spec) for spec in geometry_param_specs],
        "objective_values": objective_values_np,
        "residuals": residuals_np.tolist(),
        "jacobian": jacobian_np.tolist(),
        "initial_er_root_ad": str(args.initial_er_root_ad),
        "elapsed_s": float(evaluation.elapsed_s),
        "shared_payload_compare": shared_compare_report,
    }
    print(
        "[autodiff-gate] mode=transport_reverse_ad_only_initial_er_root_only_optimization_api_smoke "
        f"objective={args.objective} "
        f"residual_count={len(result.residual_labels)} "
        f"parameter_count={len(result.parameter_labels)} "
        f"elapsed_s={evaluation.elapsed_s:.3f}",
        flush=True,
    )
    print("[autodiff-gate] initial-Er root-only optimization residuals/Jacobian rows:")
    for row_i, label in enumerate(result.residual_labels):
        print(f"  - {label}: residual={residuals_np[row_i]:.16e}")
        for parameter_name, value in zip(result.parameter_labels, jacobian_np[row_i].tolist()):
            print(f"      d{label}/d{parameter_name}: jac={value:.16e}")
    outpath = _report_path("initial_er_root_only_optimization_api_smoke")
    outpath.write_text(json.dumps(report, indent=2))
    print(f"Wrote {outpath.relative_to(ROOT)}")


def _run_realtime_geometry_initial_carry_boundary_probe(
    *,
    args,
    config: dict[str, Any],
    baseline_values,
    baseline_runtime,
    baseline_state,
    profile_cfg: dict,
    neoclassical_cfg: dict[str, Any],
):
    profile_values = baseline_values[: len(PARAMETER_ORDER)]
    def _state_from_profiles(values):
        return _initial_state_for_parameter_vector(
            values,
            config=config,
            initial_er_root_ad=args.initial_er_root_ad,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            runtime=baseline_runtime,
        )

    baseline_profile_state, profile_state_pullback = jax.vjp(_state_from_profiles, profile_values)
    reverse_setup = _prepare_reverse_static_setup(
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
        reverse_rhs_pullback_mode=args.reverse_rhs_pullback_mode,
        reverse_stage_cotangent_mode=args.reverse_stage_cotangent_mode,
        reverse_step_bwd_mode=args.reverse_step_bwd_mode,
        reverse_stage_adjoint_memory_mode=args.reverse_stage_adjoint_memory_mode,
        reverse_stage_adjoint_iter_maxiter=args.reverse_stage_adjoint_iter_maxiter,
        reverse_stage_adjoint_iter_tol=args.reverse_stage_adjoint_iter_tol,
        reverse_stage_adjoint_woodbury_rank=args.reverse_stage_adjoint_woodbury_rank,
    )
    if args.objective == "all":
        print(
            "[autodiff-gate] progress: probing compact initial-carry boundary cotangents "
            "for all objectives",
            flush=True,
        )
        t_start = time.perf_counter()
        objective_values, initial_state_bars, reduced_bars = (
            _reverse_all_objectives_initial_state_boundary(
                baseline_profile_state,
                runtime=baseline_runtime,
                reverse_setup=reverse_setup,
            )
        )
        profile_gradient_matrix = jax.vmap(
            lambda state_bar: profile_state_pullback(state_bar)[0]
        )(initial_state_bars)
        objective_values, initial_state_bars, reduced_bars, profile_gradient_matrix = (
            jax.block_until_ready(
                (objective_values, initial_state_bars, reduced_bars, profile_gradient_matrix)
            )
        )
        elapsed_s = time.perf_counter() - t_start

        def _take_axis0(tree, index: int):
            return jax.tree_util.tree_map(lambda value: value[index], tree)

        objective_values_np = np.asarray(jax.device_get(objective_values), dtype=float)
        profile_gradient_np = np.asarray(jax.device_get(profile_gradient_matrix), dtype=float)
        boundary_diagnostics_by_objective = {}
        for objective_i, objective_label in enumerate(OBJECTIVE_LABELS):
            state_bar = _take_axis0(initial_state_bars, objective_i)
            reduced_bar = _take_axis0(reduced_bars, objective_i)
            state_bar_summary = _payload_leaf_summary(state_bar)
            field_l2 = {
                "density": _tree_array_l2_norm(state_bar.density),
                "pressure": _tree_array_l2_norm(state_bar.pressure),
                "Er": _tree_array_l2_norm(state_bar.Er),
            }
            boundary_diagnostics_by_objective[objective_label] = {
                "initial_state_bar_l2": _tree_array_l2_norm(state_bar),
                "initial_state_bar_field_l2": field_l2,
                "initial_state_bar_summary": state_bar_summary,
                "reduced_y_bar_l2": _tree_array_l2_norm(reduced_bar.y),
                "reduced_y_bar_summary": _payload_leaf_summary(reduced_bar.y),
                "reduced_lagged_response_cache_bar_l2": _tree_array_l2_norm(
                    reduced_bar.lagged_response_cache
                ),
                "reduced_lagged_response_cache_bar_summary": _payload_leaf_summary(
                    reduced_bar.lagged_response_cache
                ),
                "reduced_lagged_reference_y_bar_l2": _tree_array_l2_norm(
                    reduced_bar.lagged_reference_y
                ),
                "reduced_lagged_reference_y_bar_summary": _payload_leaf_summary(
                    reduced_bar.lagged_reference_y
                ),
            }

        report = {
            "mode": "transport_reverse_ad_only",
            "parameter_mode": str(args.reverse_parameter_mode),
            "config_path": str(Path(args.config)),
            "objective_name": "all",
            "objective_order": list(OBJECTIVE_LABELS),
            "objective_values": {
                name: float(value) for name, value in zip(OBJECTIVE_LABELS, objective_values_np.tolist())
            },
            "parameter_order": list(PARAMETER_ORDER),
            "profile_gradient_reverse_ad": {
                objective_name: {
                    parameter_name: float(value)
                    for parameter_name, value in zip(
                        PARAMETER_ORDER,
                        profile_gradient_np[objective_i].tolist(),
                    )
                }
                for objective_i, objective_name in enumerate(OBJECTIVE_LABELS)
            },
            "accepted_step_limit": None
            if args.accepted_step_limit is None
            else int(args.accepted_step_limit),
            "reverse_segment_length": None
            if args.reverse_segment_length is None
            else int(args.reverse_segment_length),
            "reverse_stage_cotangent_mode": str(args.reverse_stage_cotangent_mode),
            "ntx_exact_derivative_mode": str(args.ntx_exact_derivative_mode),
            "ntx_exact_derivative_field_pullback_mode": str(
                args.ntx_exact_derivative_field_pullback_mode
            ),
            "ntx_exact_surface_backend": str(neoclassical_cfg.get("ntx_exact_surface_backend", "booz")),
            "realtime_geometry_gradient_path": "initial_carry_boundary_probe",
            "boundary_diagnostics_by_objective": boundary_diagnostics_by_objective,
            "elapsed_s": float(elapsed_s),
        }
        print(
            "[autodiff-gate] mode=transport_reverse_ad_only "
            "parameter_mode=profiles_plus_realtime_geometry "
            "realtime_geometry_gradient_path=initial_carry_boundary_probe "
            "objective=all "
            f"elapsed_s={elapsed_s:.3f}",
            flush=True,
        )
        print("[autodiff-gate] objective values:")
        for objective_name, value in report["objective_values"].items():
            print(f"  - {objective_name}: value={value:.16e}")
        print("[autodiff-gate] reverse profile gradients by objective:")
        for objective_name in OBJECTIVE_LABELS:
            print(f"  - {objective_name}:")
            for parameter_name in PARAMETER_ORDER:
                value = report["profile_gradient_reverse_ad"][objective_name][parameter_name]
                print(f"      d{objective_name}/d{parameter_name}: ad={value:.6e}")
        print("[autodiff-gate] initial-carry boundary diagnostics by objective:")
        for objective_name in OBJECTIVE_LABELS:
            diag = report["boundary_diagnostics_by_objective"][objective_name]
            field_l2 = diag["initial_state_bar_field_l2"]
            state_finite = diag["initial_state_bar_summary"]["all_floating_leaves_finite"]
            lagged_finite = diag["reduced_lagged_response_cache_bar_summary"][
                "all_floating_leaves_finite"
            ]
            print(
                f"  - {objective_name}: "
                f"initial_state_bar_l2={diag['initial_state_bar_l2']:.6e} "
                f"density_bar_l2={field_l2['density']:.6e} "
                f"pressure_bar_l2={field_l2['pressure']:.6e} "
                f"Er_bar_l2={field_l2['Er']:.6e} "
                f"lagged_cache_bar_l2={diag['reduced_lagged_response_cache_bar_l2']:.6e} "
                f"state_bar_all_finite={state_finite} "
                f"lagged_cache_bar_all_finite={lagged_finite}"
            )
        outpath = _report_path("realtime_geometry_initial_carry_boundary")
        outpath.write_text(json.dumps(report, indent=2))
        print(f"Wrote {outpath.relative_to(ROOT)}")
        return

    objective_name = str(args.objective)
    objective_index = OBJECTIVE_LABELS.index(objective_name)
    print(
        "[autodiff-gate] progress: probing compact initial-carry boundary cotangent "
        f"for {objective_name}",
        flush=True,
    )
    t_start = time.perf_counter()
    objective_value, initial_state_bar, boundary_diagnostics = _reverse_objective_initial_state_bar(
        baseline_profile_state,
        runtime=baseline_runtime,
        reverse_setup=reverse_setup,
        objective_index=objective_index,
    )
    objective_value, initial_state_bar, boundary_diagnostics = jax.block_until_ready(
        (objective_value, initial_state_bar, boundary_diagnostics)
    )
    elapsed_s = time.perf_counter() - t_start

    state_bar_summary = _payload_leaf_summary(initial_state_bar)
    field_l2 = {
        "density": _tree_array_l2_norm(initial_state_bar.density),
        "pressure": _tree_array_l2_norm(initial_state_bar.pressure),
        "Er": _tree_array_l2_norm(initial_state_bar.Er),
    }
    report = {
        "mode": "transport_reverse_ad_only",
        "parameter_mode": str(args.reverse_parameter_mode),
        "config_path": str(Path(args.config)),
        "objective_name": objective_name,
        "objective_value": float(np.asarray(jax.device_get(objective_value), dtype=float)),
        "accepted_step_limit": None if args.accepted_step_limit is None else int(args.accepted_step_limit),
        "reverse_segment_length": None if args.reverse_segment_length is None else int(args.reverse_segment_length),
        "reverse_stage_cotangent_mode": str(args.reverse_stage_cotangent_mode),
        "ntx_exact_derivative_mode": str(args.ntx_exact_derivative_mode),
        "ntx_exact_derivative_field_pullback_mode": str(args.ntx_exact_derivative_field_pullback_mode),
        "ntx_exact_surface_backend": str(neoclassical_cfg.get("ntx_exact_surface_backend", "booz")),
        "realtime_geometry_gradient_path": "initial_carry_boundary_probe",
        "initial_state_bar_summary": state_bar_summary,
        "initial_state_bar_l2": _tree_array_l2_norm(initial_state_bar),
        "initial_state_bar_field_l2": field_l2,
        "boundary_diagnostics": boundary_diagnostics,
        "elapsed_s": float(elapsed_s),
    }
    print(
        "[autodiff-gate] mode=transport_reverse_ad_only "
        "parameter_mode=profiles_plus_realtime_geometry "
        "realtime_geometry_gradient_path=initial_carry_boundary_probe "
        f"objective={objective_name} "
        f"value={report['objective_value']:.16e} "
        f"initial_state_bar_l2={report['initial_state_bar_l2']:.6e} "
        f"density_bar_l2={field_l2['density']:.6e} "
        f"pressure_bar_l2={field_l2['pressure']:.6e} "
        f"Er_bar_l2={field_l2['Er']:.6e} "
        f"state_bar_all_finite={state_bar_summary['all_floating_leaves_finite']} "
        f"reduced_y_bar_all_finite="
        f"{boundary_diagnostics['reduced_y_bar_summary']['all_floating_leaves_finite']} "
        f"reduced_lagged_cache_bar_l2="
        f"{boundary_diagnostics['reduced_lagged_response_cache_bar_l2']:.6e} "
        f"reduced_lagged_cache_bar_all_finite="
        f"{boundary_diagnostics['reduced_lagged_response_cache_bar_summary']['all_floating_leaves_finite']} "
        f"no_lagged_cache_state_bar_all_finite="
        f"{boundary_diagnostics['initial_state_bar_without_lagged_cache_summary']['all_floating_leaves_finite']} "
        f"y_only_state_bar_all_finite="
        f"{boundary_diagnostics['initial_state_bar_y_only_summary']['all_floating_leaves_finite']} "
        f"lagged_reference_only_state_bar_all_finite="
        f"{boundary_diagnostics['initial_state_bar_lagged_reference_only_summary']['all_floating_leaves_finite']} "
        f"lagged_cache_only_state_bar_all_finite="
        f"{boundary_diagnostics['initial_state_bar_lagged_cache_only_summary']['all_floating_leaves_finite']} "
        f"elapsed_s={elapsed_s:.3f}",
        flush=True,
    )
    outpath = _report_path("realtime_geometry_initial_carry_boundary")
    outpath.write_text(json.dumps(report, indent=2))
    print(f"Wrote {outpath.relative_to(ROOT)}")


def _run_local_stage_matvec_diagnostic_report(
    *,
    args,
    reverse_setup: _ReverseStaticSetup,
    accepted_step_index: int,
    mode_label: str = "transport_reverse_ad_only_local_stage_matvec_diagnostic",
):
    if accepted_step_index < 0:
        raise SystemExit("[autodiff-gate] --local-transpose-diagnostic-accepted-step must be >= 0.")
    print("[autodiff-gate] progress: running local stage transpose matvec diagnostic", flush=True)
    baseline_rollout = _radau_adaptive_schedule_rollout(
        reverse_setup.execution_context,
        reverse_setup.prepared_rollout.initial_carry,
        max_total_steps=reverse_setup.max_total_steps,
        stop_after_accepted_steps=reverse_setup.stop_after_accepted_steps,
    )
    diagnostic = _radau_debug_local_stage_transpose_matvec(
        reverse_setup.execution_context,
        reverse_setup.prepared_rollout.initial_carry,
        baseline_rollout.trace,
        accepted_step_index=accepted_step_index,
    )
    diagnostic = jax.device_get(diagnostic)
    report = {
        "mode": mode_label,
        "config_path": str(Path(args.config)),
        "objective_name": args.objective,
        "accepted_step_limit": None if args.accepted_step_limit is None else int(args.accepted_step_limit),
        "diagnostic_accepted_step_index": accepted_step_index,
        "target_attempt_index": int(diagnostic["target_attempt_index"]),
        "found_target": bool(diagnostic["found_target"]),
        "lagged_response_valid_in": bool(diagnostic["lagged_response_valid_in"]),
        "local_branch_reuse": bool(diagnostic["local_branch_reuse"]),
        "compact_l2": float(diagnostic["compact_l2"]),
        "dense_l2": float(diagnostic["dense_l2"]),
        "diff_l2": float(diagnostic["diff_l2"]),
        "rel_err": float(diagnostic["rel_err"]),
        "max_abs_diff": float(diagnostic["max_abs_diff"]),
        "generic_diff_l2": float(diagnostic["generic_diff_l2"]),
        "generic_rel_err": float(diagnostic["generic_rel_err"]),
        "generic_max_abs_diff": float(diagnostic["generic_max_abs_diff"]),
        "full_lagged_generic_diff_l2": float(diagnostic["full_lagged_generic_diff_l2"]),
        "full_lagged_generic_pressure_diff_l2": float(diagnostic["full_lagged_generic_pressure_diff_l2"]),
        "explicit_vs_full_lagged_generic_diff_l2": float(
            diagnostic["explicit_vs_full_lagged_generic_diff_l2"]
        ),
        "explicit_vs_full_lagged_generic_pressure_diff_l2": float(
            diagnostic["explicit_vs_full_lagged_generic_pressure_diff_l2"]
        ),
        "explicit_rhs_state_diff_l2": float(diagnostic["explicit_rhs_state_diff_l2"]),
        "explicit_rhs_state_rel_err": float(diagnostic["explicit_rhs_state_rel_err"]),
        "explicit_rhs_state_max_abs_diff": float(diagnostic["explicit_rhs_state_max_abs_diff"]),
        "projected_explicit_rhs_state_diff_l2": float(diagnostic["projected_explicit_rhs_state_diff_l2"]),
        "projected_explicit_rhs_state_rel_err": float(diagnostic["projected_explicit_rhs_state_rel_err"]),
        "projected_explicit_rhs_state_pressure_diff_l2": float(
            diagnostic["projected_explicit_rhs_state_pressure_diff_l2"]
        ),
        "split_rhs_state_diff_l2": float(diagnostic["split_rhs_state_diff_l2"]),
        "split_rhs_state_rel_err": float(diagnostic["split_rhs_state_rel_err"]),
        "split_rhs_state_max_abs_diff": float(diagnostic["split_rhs_state_max_abs_diff"]),
        "flux_vs_generic_residual_diff_l2": float(diagnostic["flux_vs_generic_residual_diff_l2"]),
        "direct_vs_generic_residual_diff_l2": float(diagnostic["direct_vs_generic_residual_diff_l2"]),
        "flux_vs_generic_pressure_diff_l2": float(diagnostic["flux_vs_generic_pressure_diff_l2"]),
        "direct_vs_generic_pressure_diff_l2": float(diagnostic["direct_vs_generic_pressure_diff_l2"]),
        "compact_flux_vs_generic_flux_diff_l2": float(diagnostic["compact_flux_vs_generic_flux_diff_l2"]),
        "compact_flux_vs_generic_flux_pressure_diff_l2": float(
            diagnostic["compact_flux_vs_generic_flux_pressure_diff_l2"]
        ),
        "compact_direct_vs_generic_direct_diff_l2": float(diagnostic["compact_direct_vs_generic_direct_diff_l2"]),
        "compact_direct_vs_generic_direct_pressure_diff_l2": float(
            diagnostic["compact_direct_vs_generic_direct_pressure_diff_l2"]
        ),
        "joint_generic_diff_l2": float(diagnostic["joint_generic_diff_l2"]),
        "joint_generic_pressure_diff_l2": float(diagnostic["joint_generic_pressure_diff_l2"]),
        "explicit_rhs_state_density_diff_l2": float(diagnostic["explicit_rhs_state_density_diff_l2"]),
        "explicit_rhs_state_pressure_diff_l2": float(diagnostic["explicit_rhs_state_pressure_diff_l2"]),
        "explicit_rhs_state_er_tail_diff_l2": float(diagnostic["explicit_rhs_state_er_tail_diff_l2"]),
        "radial_state_dim": int(diagnostic["radial_state_dim"]),
        "radial_density_size": int(diagnostic["radial_density_size"]),
        "radial_pressure_size": int(diagnostic["radial_pressure_size"]),
        "radial_er_size": int(diagnostic["radial_er_size"]),
        "radial_active_er_size": int(diagnostic["radial_active_er_size"]),
        "radial_extra_size": int(diagnostic["radial_extra_size"]),
        "radial_block_count": int(diagnostic["radial_block_count"]),
        "radial_block_dim": int(diagnostic["radial_block_dim"]),
        "radial_variables_per_cell": int(diagnostic["radial_variables_per_cell"]),
        "radial_extra_per_cell_count": int(diagnostic["radial_extra_per_cell_count"]),
        "radial_off_tridiagonal_l2": float(diagnostic["radial_off_tridiagonal_l2"]),
        "radial_off_tridiagonal_rel_l2": float(diagnostic["radial_off_tridiagonal_rel_l2"]),
        "radial_off_tridiagonal_max_abs": float(diagnostic["radial_off_tridiagonal_max_abs"]),
        "radial_explicit_rhs_component_l2": float(diagnostic["radial_explicit_rhs_component_l2"]),
        "radial_explicit_rhs_component_off_l2": float(diagnostic["radial_explicit_rhs_component_off_l2"]),
        "radial_explicit_rhs_component_off_rel_l2": float(diagnostic["radial_explicit_rhs_component_off_rel_l2"]),
        "radial_explicit_rhs_component_off_max_abs": float(diagnostic["radial_explicit_rhs_component_off_max_abs"]),
        "radial_flux_rhs_component_l2": float(diagnostic["radial_flux_rhs_component_l2"]),
        "radial_flux_rhs_component_off_l2": float(diagnostic["radial_flux_rhs_component_off_l2"]),
        "radial_flux_rhs_component_off_rel_l2": float(diagnostic["radial_flux_rhs_component_off_rel_l2"]),
        "radial_flux_rhs_component_off_max_abs": float(diagnostic["radial_flux_rhs_component_off_max_abs"]),
        "radial_direct_rhs_component_l2": float(diagnostic["radial_direct_rhs_component_l2"]),
        "radial_direct_rhs_component_off_l2": float(diagnostic["radial_direct_rhs_component_off_l2"]),
        "radial_direct_rhs_component_off_rel_l2": float(diagnostic["radial_direct_rhs_component_off_rel_l2"]),
        "radial_direct_rhs_component_off_max_abs": float(diagnostic["radial_direct_rhs_component_off_max_abs"]),
        "radial_direct_density_rhs_component_l2": float(diagnostic["radial_direct_density_rhs_component_l2"]),
        "radial_direct_density_rhs_component_off_l2": float(diagnostic["radial_direct_density_rhs_component_off_l2"]),
        "radial_direct_density_rhs_component_off_rel_l2": float(diagnostic["radial_direct_density_rhs_component_off_rel_l2"]),
        "radial_direct_density_rhs_component_off_max_abs": float(diagnostic["radial_direct_density_rhs_component_off_max_abs"]),
        "radial_direct_pressure_rhs_component_l2": float(diagnostic["radial_direct_pressure_rhs_component_l2"]),
        "radial_direct_pressure_rhs_component_off_l2": float(diagnostic["radial_direct_pressure_rhs_component_off_l2"]),
        "radial_direct_pressure_rhs_component_off_rel_l2": float(diagnostic["radial_direct_pressure_rhs_component_off_rel_l2"]),
        "radial_direct_pressure_rhs_component_off_max_abs": float(diagnostic["radial_direct_pressure_rhs_component_off_max_abs"]),
        "radial_direct_er_rhs_component_l2": float(diagnostic["radial_direct_er_rhs_component_l2"]),
        "radial_direct_er_rhs_component_off_l2": float(diagnostic["radial_direct_er_rhs_component_off_l2"]),
        "radial_direct_er_rhs_component_off_rel_l2": float(diagnostic["radial_direct_er_rhs_component_off_rel_l2"]),
        "radial_direct_er_rhs_component_off_max_abs": float(diagnostic["radial_direct_er_rhs_component_off_max_abs"]),
        "radial_direct_er_diffusion_rhs_component_l2": float(diagnostic["radial_direct_er_diffusion_rhs_component_l2"]),
        "radial_direct_er_diffusion_rhs_component_off_l2": float(diagnostic["radial_direct_er_diffusion_rhs_component_off_l2"]),
        "radial_direct_er_diffusion_rhs_component_off_rel_l2": float(diagnostic["radial_direct_er_diffusion_rhs_component_off_rel_l2"]),
        "radial_direct_er_diffusion_rhs_component_off_max_abs": float(diagnostic["radial_direct_er_diffusion_rhs_component_off_max_abs"]),
        "radial_direct_er_ambipolar_rhs_component_l2": float(diagnostic["radial_direct_er_ambipolar_rhs_component_l2"]),
        "radial_direct_er_ambipolar_rhs_component_off_l2": float(diagnostic["radial_direct_er_ambipolar_rhs_component_off_l2"]),
        "radial_direct_er_ambipolar_rhs_component_off_rel_l2": float(diagnostic["radial_direct_er_ambipolar_rhs_component_off_rel_l2"]),
        "radial_direct_er_ambipolar_rhs_component_off_max_abs": float(diagnostic["radial_direct_er_ambipolar_rhs_component_off_max_abs"]),
        "radial_direct_er_ambi_coeff_rhs_component_l2": float(diagnostic["radial_direct_er_ambi_coeff_rhs_component_l2"]),
        "radial_direct_er_ambi_coeff_rhs_component_off_l2": float(diagnostic["radial_direct_er_ambi_coeff_rhs_component_off_l2"]),
        "radial_direct_er_ambi_coeff_rhs_component_off_rel_l2": float(diagnostic["radial_direct_er_ambi_coeff_rhs_component_off_rel_l2"]),
        "radial_direct_er_ambi_coeff_rhs_component_off_max_abs": float(diagnostic["radial_direct_er_ambi_coeff_rhs_component_off_max_abs"]),
        "radial_direct_er_ambi_charge_flux_rhs_component_l2": float(diagnostic["radial_direct_er_ambi_charge_flux_rhs_component_l2"]),
        "radial_direct_er_ambi_charge_flux_rhs_component_off_l2": float(diagnostic["radial_direct_er_ambi_charge_flux_rhs_component_off_l2"]),
        "radial_direct_er_ambi_charge_flux_rhs_component_off_rel_l2": float(diagnostic["radial_direct_er_ambi_charge_flux_rhs_component_off_rel_l2"]),
        "radial_direct_er_ambi_charge_flux_rhs_component_off_max_abs": float(diagnostic["radial_direct_er_ambi_charge_flux_rhs_component_off_max_abs"]),
        "radial_joint_generic_rhs_component_l2": float(diagnostic["radial_joint_generic_rhs_component_l2"]),
        "radial_joint_generic_rhs_component_off_l2": float(diagnostic["radial_joint_generic_rhs_component_off_l2"]),
        "radial_joint_generic_rhs_component_off_rel_l2": float(diagnostic["radial_joint_generic_rhs_component_off_rel_l2"]),
        "radial_joint_generic_rhs_component_off_max_abs": float(diagnostic["radial_joint_generic_rhs_component_off_max_abs"]),
        "radial_offset0_l2": float(diagnostic["radial_offset0_l2"]),
        "radial_offset1_l2": float(diagnostic["radial_offset1_l2"]),
        "radial_offset2_l2": float(diagnostic["radial_offset2_l2"]),
        "radial_offset3_l2": float(diagnostic["radial_offset3_l2"]),
        "radial_offset4_l2": float(diagnostic["radial_offset4_l2"]),
        "radial_offset_ge5_l2": float(diagnostic["radial_offset_ge5_l2"]),
        "radial_max_significant_offset": int(diagnostic["radial_max_significant_offset"]),
        "radial_off_tridiagonal_significant_count": int(diagnostic["radial_off_tridiagonal_significant_count"]),
        "radial_band_builder_block_dim": int(diagnostic["radial_band_builder_block_dim"]),
        "radial_band_builder_permutation_max_abs_diff": int(
            diagnostic["radial_band_builder_permutation_max_abs_diff"]
        ),
        "radial_band_builder_diff_l2": float(diagnostic["radial_band_builder_diff_l2"]),
        "radial_band_builder_rel_err": float(diagnostic["radial_band_builder_rel_err"]),
        "radial_band_builder_max_abs_diff": float(diagnostic["radial_band_builder_max_abs_diff"]),
        "radial_off_tridiagonal_svals_top8": [
            float(diagnostic[f"radial_off_tridiagonal_sval{i}"]) for i in range(8)
        ],
        "radial_off_tridiagonal_rank_999": int(diagnostic["radial_off_tridiagonal_rank_999"]),
        "radial_off_tridiagonal_rank_9999": int(diagnostic["radial_off_tridiagonal_rank_9999"]),
        "radial_off_tridiagonal_numerical_rank": int(diagnostic["radial_off_tridiagonal_numerical_rank"]),
        "radial_woodbury_rank12_solve_diff_l2": float(diagnostic["radial_woodbury_rank12_solve_diff_l2"]),
        "radial_woodbury_rank12_solve_rel_err": float(diagnostic["radial_woodbury_rank12_solve_rel_err"]),
        "radial_woodbury_rank12_solve_max_abs": float(diagnostic["radial_woodbury_rank12_solve_max_abs"]),
        "radial_woodbury_rank16_solve_diff_l2": float(diagnostic["radial_woodbury_rank16_solve_diff_l2"]),
        "radial_woodbury_rank16_solve_rel_err": float(diagnostic["radial_woodbury_rank16_solve_rel_err"]),
        "radial_woodbury_rank16_solve_max_abs": float(diagnostic["radial_woodbury_rank16_solve_max_abs"]),
        "radial_woodbury_rank24_solve_diff_l2": float(diagnostic["radial_woodbury_rank24_solve_diff_l2"]),
        "radial_woodbury_rank24_solve_rel_err": float(diagnostic["radial_woodbury_rank24_solve_rel_err"]),
        "radial_woodbury_rank24_solve_max_abs": float(diagnostic["radial_woodbury_rank24_solve_max_abs"]),
        "radial_woodbury_rank32_solve_diff_l2": float(diagnostic["radial_woodbury_rank32_solve_diff_l2"]),
        "radial_woodbury_rank32_solve_rel_err": float(diagnostic["radial_woodbury_rank32_solve_rel_err"]),
        "radial_woodbury_rank32_solve_max_abs": float(diagnostic["radial_woodbury_rank32_solve_max_abs"]),
        "radial_woodbury_rank48_solve_diff_l2": float(diagnostic["radial_woodbury_rank48_solve_diff_l2"]),
        "radial_woodbury_rank48_solve_rel_err": float(diagnostic["radial_woodbury_rank48_solve_rel_err"]),
        "radial_woodbury_rank48_solve_max_abs": float(diagnostic["radial_woodbury_rank48_solve_max_abs"]),
    }
    print(
        "[autodiff-gate] local stage matvec diagnostic: "
        f"accepted_step_index={accepted_step_index} "
        f"target_attempt_index={report['target_attempt_index']} "
        f"found_target={report['found_target']} "
        f"lagged_response_valid_in={report['lagged_response_valid_in']} "
        f"local_branch_reuse={report['local_branch_reuse']}",
        flush=True,
    )
    print(
        "[autodiff-gate] local stage matvec diagnostic values: "
        f"compact_l2={report['compact_l2']:.6e} "
        f"dense_l2={report['dense_l2']:.6e} "
        f"diff_l2={report['diff_l2']:.6e} "
        f"rel_err={report['rel_err']:.6e} "
        f"max_abs_diff={report['max_abs_diff']:.6e}",
        flush=True,
    )
    print(
        "[autodiff-gate] local stage generic-vjp diagnostic values: "
        f"generic_diff_l2={report['generic_diff_l2']:.6e} "
        f"generic_rel_err={report['generic_rel_err']:.6e} "
        f"generic_max_abs_diff={report['generic_max_abs_diff']:.6e} "
        f"full_lagged_generic_l2={report['full_lagged_generic_diff_l2']:.6e} "
        f"full_lagged_generic_pressure_l2={report['full_lagged_generic_pressure_diff_l2']:.6e}",
        flush=True,
    )
    print(
        "[autodiff-gate] local stage explicit-rhs-state diagnostic values: "
        f"explicit_diff_l2={report['explicit_rhs_state_diff_l2']:.6e} "
        f"explicit_rel_err={report['explicit_rhs_state_rel_err']:.6e} "
        f"explicit_max_abs_diff={report['explicit_rhs_state_max_abs_diff']:.6e} "
        f"projected_explicit_diff_l2={report['projected_explicit_rhs_state_diff_l2']:.6e} "
        f"projected_explicit_rel_err={report['projected_explicit_rhs_state_rel_err']:.6e} "
        f"split_diff_l2={report['split_rhs_state_diff_l2']:.6e} "
        f"split_rel_err={report['split_rhs_state_rel_err']:.6e} "
        f"split_max_abs_diff={report['split_rhs_state_max_abs_diff']:.6e}",
        flush=True,
    )
    print(
        "[autodiff-gate] local stage explicit-rhs-state split attribution: "
        f"flux_vs_generic_residual_l2={report['flux_vs_generic_residual_diff_l2']:.6e} "
        f"direct_vs_generic_residual_l2={report['direct_vs_generic_residual_diff_l2']:.6e} "
        f"flux_pressure_l2={report['flux_vs_generic_pressure_diff_l2']:.6e} "
        f"direct_pressure_l2={report['direct_vs_generic_pressure_diff_l2']:.6e} "
        f"compact_flux_vs_generic_flux_l2={report['compact_flux_vs_generic_flux_diff_l2']:.6e} "
        f"compact_flux_vs_generic_flux_pressure_l2="
        f"{report['compact_flux_vs_generic_flux_pressure_diff_l2']:.6e} "
        f"compact_direct_vs_generic_direct_l2={report['compact_direct_vs_generic_direct_diff_l2']:.6e} "
        f"compact_direct_vs_generic_direct_pressure_l2="
        f"{report['compact_direct_vs_generic_direct_pressure_diff_l2']:.6e} "
        f"joint_generic_l2={report['joint_generic_diff_l2']:.6e} "
        f"joint_generic_pressure_l2={report['joint_generic_pressure_diff_l2']:.6e} "
        f"explicit_vs_full_lagged_generic_l2={report['explicit_vs_full_lagged_generic_diff_l2']:.6e} "
        f"explicit_vs_full_lagged_generic_pressure_l2="
        f"{report['explicit_vs_full_lagged_generic_pressure_diff_l2']:.6e}",
        flush=True,
    )
    print(
        "[autodiff-gate] local stage explicit-rhs-state component errors: "
        f"density_l2={report['explicit_rhs_state_density_diff_l2']:.6e} "
        f"pressure_l2={report['explicit_rhs_state_pressure_diff_l2']:.6e} "
        f"projected_pressure_l2={report['projected_explicit_rhs_state_pressure_diff_l2']:.6e} "
        f"er_tail_l2={report['explicit_rhs_state_er_tail_diff_l2']:.6e}",
        flush=True,
    )
    print(
        "[autodiff-gate] local stage radial-block diagnostic values: "
        f"state_dim={report['radial_state_dim']} "
        f"density_size={report['radial_density_size']} "
        f"pressure_size={report['radial_pressure_size']} "
        f"er_size={report['radial_er_size']} "
        f"active_er_size={report['radial_active_er_size']} "
        f"extra_size={report['radial_extra_size']} "
        f"block_count={report['radial_block_count']} "
        f"block_dim={report['radial_block_dim']} "
        f"variables_per_cell={report['radial_variables_per_cell']} "
        f"extra_per_cell={report['radial_extra_per_cell_count']} "
        f"off_tridiagonal_l2={report['radial_off_tridiagonal_l2']:.6e} "
        f"off_tridiagonal_rel_l2={report['radial_off_tridiagonal_rel_l2']:.6e} "
        f"off_tridiagonal_max_abs={report['radial_off_tridiagonal_max_abs']:.6e}",
        flush=True,
    )
    print(
        "[autodiff-gate] local stage radial-block bandwidth values: "
        f"offset0_l2={report['radial_offset0_l2']:.6e} "
        f"offset1_l2={report['radial_offset1_l2']:.6e} "
        f"offset2_l2={report['radial_offset2_l2']:.6e} "
        f"offset3_l2={report['radial_offset3_l2']:.6e} "
        f"offset4_l2={report['radial_offset4_l2']:.6e} "
        f"offset_ge5_l2={report['radial_offset_ge5_l2']:.6e} "
        f"max_significant_offset={report['radial_max_significant_offset']} "
        f"off_tridiagonal_significant_count={report['radial_off_tridiagonal_significant_count']}",
        flush=True,
    )
    print(
        "[autodiff-gate] local stage radial-band builder check: "
        f"block_dim={report['radial_band_builder_block_dim']} "
        f"permutation_max_abs_diff={report['radial_band_builder_permutation_max_abs_diff']} "
        f"diff_l2={report['radial_band_builder_diff_l2']:.6e} "
        f"rel_err={report['radial_band_builder_rel_err']:.6e} "
        f"max_abs_diff={report['radial_band_builder_max_abs_diff']:.6e}",
        flush=True,
    )
    _missing_component_value = float("nan")
    print(
        "[autodiff-gate] local stage off-band component attribution: "
        f"explicit_l2={report.get('radial_explicit_rhs_component_l2', _missing_component_value):.6e} "
        f"explicit_off_l2={report.get('radial_explicit_rhs_component_off_l2', _missing_component_value):.6e} "
        f"explicit_off_rel={report.get('radial_explicit_rhs_component_off_rel_l2', _missing_component_value):.6e} "
        f"explicit_off_max={report.get('radial_explicit_rhs_component_off_max_abs', _missing_component_value):.6e} "
        f"flux_l2={report.get('radial_flux_rhs_component_l2', _missing_component_value):.6e} "
        f"flux_off_l2={report.get('radial_flux_rhs_component_off_l2', _missing_component_value):.6e} "
        f"flux_off_rel={report.get('radial_flux_rhs_component_off_rel_l2', _missing_component_value):.6e} "
        f"flux_off_max={report.get('radial_flux_rhs_component_off_max_abs', _missing_component_value):.6e} "
        f"direct_l2={report.get('radial_direct_rhs_component_l2', _missing_component_value):.6e} "
        f"direct_off_l2={report.get('radial_direct_rhs_component_off_l2', _missing_component_value):.6e} "
        f"direct_off_rel={report.get('radial_direct_rhs_component_off_rel_l2', _missing_component_value):.6e} "
        f"direct_off_max={report.get('radial_direct_rhs_component_off_max_abs', _missing_component_value):.6e} "
        f"joint_generic_l2={report.get('radial_joint_generic_rhs_component_l2', _missing_component_value):.6e} "
        f"joint_generic_off_l2={report.get('radial_joint_generic_rhs_component_off_l2', _missing_component_value):.6e} "
        f"joint_generic_off_rel={report.get('radial_joint_generic_rhs_component_off_rel_l2', _missing_component_value):.6e} "
        f"joint_generic_off_max={report.get('radial_joint_generic_rhs_component_off_max_abs', _missing_component_value):.6e}",
        flush=True,
    )
    print(
        "[autodiff-gate] local stage direct off-band equation attribution: "
        f"density_l2={report['radial_direct_density_rhs_component_l2']:.6e} "
        f"density_off_l2={report['radial_direct_density_rhs_component_off_l2']:.6e} "
        f"density_off_rel={report['radial_direct_density_rhs_component_off_rel_l2']:.6e} "
        f"density_off_max={report['radial_direct_density_rhs_component_off_max_abs']:.6e} "
        f"pressure_l2={report['radial_direct_pressure_rhs_component_l2']:.6e} "
        f"pressure_off_l2={report['radial_direct_pressure_rhs_component_off_l2']:.6e} "
        f"pressure_off_rel={report['radial_direct_pressure_rhs_component_off_rel_l2']:.6e} "
        f"pressure_off_max={report['radial_direct_pressure_rhs_component_off_max_abs']:.6e} "
        f"er_l2={report['radial_direct_er_rhs_component_l2']:.6e} "
        f"er_off_l2={report['radial_direct_er_rhs_component_off_l2']:.6e} "
        f"er_off_rel={report['radial_direct_er_rhs_component_off_rel_l2']:.6e} "
        f"er_off_max={report['radial_direct_er_rhs_component_off_max_abs']:.6e}",
        flush=True,
    )
    print(
        "[autodiff-gate] local stage Er direct subterm attribution: "
        f"diffusion_l2={report['radial_direct_er_diffusion_rhs_component_l2']:.6e} "
        f"diffusion_off_l2={report['radial_direct_er_diffusion_rhs_component_off_l2']:.6e} "
        f"diffusion_off_rel={report['radial_direct_er_diffusion_rhs_component_off_rel_l2']:.6e} "
        f"diffusion_off_max={report['radial_direct_er_diffusion_rhs_component_off_max_abs']:.6e} "
        f"ambipolar_l2={report['radial_direct_er_ambipolar_rhs_component_l2']:.6e} "
        f"ambipolar_off_l2={report['radial_direct_er_ambipolar_rhs_component_off_l2']:.6e} "
        f"ambipolar_off_rel={report['radial_direct_er_ambipolar_rhs_component_off_rel_l2']:.6e} "
        f"ambipolar_off_max={report['radial_direct_er_ambipolar_rhs_component_off_max_abs']:.6e} "
        f"coeff_l2={report['radial_direct_er_ambi_coeff_rhs_component_l2']:.6e} "
        f"coeff_off_l2={report['radial_direct_er_ambi_coeff_rhs_component_off_l2']:.6e} "
        f"coeff_off_rel={report['radial_direct_er_ambi_coeff_rhs_component_off_rel_l2']:.6e} "
        f"coeff_off_max={report['radial_direct_er_ambi_coeff_rhs_component_off_max_abs']:.6e} "
        f"charge_flux_l2={report['radial_direct_er_ambi_charge_flux_rhs_component_l2']:.6e} "
        f"charge_flux_off_l2={report['radial_direct_er_ambi_charge_flux_rhs_component_off_l2']:.6e} "
        f"charge_flux_off_rel={report['radial_direct_er_ambi_charge_flux_rhs_component_off_rel_l2']:.6e} "
        f"charge_flux_off_max={report['radial_direct_er_ambi_charge_flux_rhs_component_off_max_abs']:.6e}",
        flush=True,
    )
    print(
        "[autodiff-gate] local stage off-tridiagonal low-rank values: "
        f"top8_singular_values="
        f"{','.join(f'{value:.6e}' for value in report['radial_off_tridiagonal_svals_top8'])} "
        f"rank_999={report['radial_off_tridiagonal_rank_999']} "
        f"rank_9999={report['radial_off_tridiagonal_rank_9999']} "
        f"numerical_rank={report['radial_off_tridiagonal_numerical_rank']}",
        flush=True,
    )
    print(
        "[autodiff-gate] local stage Woodbury solve diagnostic values: "
        f"rank12_l2={report['radial_woodbury_rank12_solve_diff_l2']:.6e} "
        f"rank12_rel={report['radial_woodbury_rank12_solve_rel_err']:.6e} "
        f"rank12_max={report['radial_woodbury_rank12_solve_max_abs']:.6e} "
        f"rank16_l2={report['radial_woodbury_rank16_solve_diff_l2']:.6e} "
        f"rank16_rel={report['radial_woodbury_rank16_solve_rel_err']:.6e} "
        f"rank16_max={report['radial_woodbury_rank16_solve_max_abs']:.6e} "
        f"rank24_l2={report['radial_woodbury_rank24_solve_diff_l2']:.6e} "
        f"rank24_rel={report['radial_woodbury_rank24_solve_rel_err']:.6e} "
        f"rank24_max={report['radial_woodbury_rank24_solve_max_abs']:.6e} "
        f"rank32_l2={report['radial_woodbury_rank32_solve_diff_l2']:.6e} "
        f"rank32_rel={report['radial_woodbury_rank32_solve_rel_err']:.6e} "
        f"rank32_max={report['radial_woodbury_rank32_solve_max_abs']:.6e} "
        f"rank48_l2={report['radial_woodbury_rank48_solve_diff_l2']:.6e} "
        f"rank48_rel={report['radial_woodbury_rank48_solve_rel_err']:.6e} "
        f"rank48_max={report['radial_woodbury_rank48_solve_max_abs']:.6e}",
        flush=True,
    )
    outpath = _report_path(args.objective)
    outpath.write_text(json.dumps(report, indent=2))
    print(f"Wrote {outpath.relative_to(ROOT)}")
    return report


def _run_realtime_geometry_reverse_mode(
    *,
    args,
    config: dict[str, Any],
    profile_cfg: dict,
    effective_ntx_exact_derivative_mode: str,
    neoclassical_cfg: dict[str, Any],
):
    if args.timing_mode == "split-vjp-warm":
        raise SystemExit(
            "[autodiff-gate] realtime-geometry reverse mode does not support "
            "--timing-mode split-vjp-warm. Use jit-warm, jit-compile-only, or eager."
        )
    local_transpose_diagnostic_accepted_step = args.local_transpose_diagnostic_accepted_step
    if local_transpose_diagnostic_accepted_step is not None and not bool(args.stage_matvec_diagnostic):
        raise SystemExit(
            "[autodiff-gate] realtime-geometry local transpose diagnostics currently "
            "support only --stage-matvec-diagnostic."
        )
    if bool(args.diagnose_final_objective_cotangent):
        raise SystemExit(
            "[autodiff-gate] --diagnose-final-objective-cotangent is only "
            "available for the profile-only static reverse setup."
        )
    if str(args.reverse_all_objectives_mode) != "jacrev":
        raise SystemExit(
            "[autodiff-gate] --reverse-all-objectives-mode vmap_pullback is "
            "currently implemented only for the profile-only static reverse setup."
        )
    geometry_parameter = str(args.reverse_geometry_parameter)
    parameter_order = _reverse_geometry_parameter_order(geometry_parameter)
    geometry_context = _geometry_context_from_config(config, geometry_parameter)
    geom_cfg = config.get("geometry", {})
    baseline_values = jnp.asarray(
        [_profile_cfg_scalar_value(profile_cfg, name) for name in PARAMETER_ORDER]
        + [float(geom_cfg.get("vmec_param_delta", 0.0))],
        dtype=jnp.float64,
    )
    baseline_geometry_delta = float(geom_cfg.get("vmec_param_delta", 0.0))
    # Use the same entrypoint as the realtime forward solver for the primal
    # runtime.  Geometry-context helpers below are only for derivative-side
    # support-payload pullbacks against VMEC harmonics.
    phase_start = time.perf_counter()
    baseline_runtime, baseline_state = build_runtime_context(config)
    jax.block_until_ready(jax.tree_util.tree_leaves(baseline_state))
    print(
        "[autodiff-gate] progress: realtime geometry runtime build ready "
        f"elapsed_s={time.perf_counter() - phase_start:.3f}",
        flush=True,
    )
    if bool(args.initial_er_root_only_optimization_smoke):
        _run_initial_er_root_only_optimization_api_smoke(
            args=args,
            config=config,
            baseline_runtime=baseline_runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            geometry_context=geometry_context,
            neoclassical_cfg=neoclassical_cfg,
        )
        return
    phase_start = time.perf_counter()
    baseline_profile_state = _initial_state_for_parameter_vector(
        baseline_values[: len(PARAMETER_ORDER)],
        config=config,
        initial_er_root_ad=args.initial_er_root_ad,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        runtime=baseline_runtime,
    )
    baseline_components = prepare_transport_solver_components(config, baseline_runtime, baseline_profile_state)
    jax.block_until_ready(jax.tree_util.tree_leaves(baseline_profile_state))
    print(
        "[autodiff-gate] progress: realtime geometry solver components ready "
        f"elapsed_s={time.perf_counter() - phase_start:.3f}",
        flush=True,
    )
    static_solver = baseline_components["solver"]
    print(
        "[autodiff-gate] realtime geometry device: "
        f"default_backend={jax.default_backend()} "
        f"baseline_values_device={_array_device_summary(baseline_values)} "
        f"local_devices={[str(device) for device in jax.local_devices()]}",
        flush=True,
    )
    if bool(getattr(args, "transport_solver_forward_smoke", False)):
        _run_transport_solver_forward_smoke(
            args=args,
            solver=static_solver,
            solve_vector_field=baseline_components["solve_vector_field"],
            runtime=baseline_runtime,
            baseline_state=baseline_profile_state,
        )
        return
    if bool(args.optimization_api_smoke) or bool(args.full_transport_shared_payload_smoke):
        if str(args.realtime_geometry_gradient_path) != "reverse_payload":
            raise SystemExit(
                "[autodiff-gate] --optimization-api-smoke/--full-transport-shared-payload-smoke currently requires "
                "--realtime-geometry-gradient-path reverse_payload so it exercises "
                "the validated full realtime geometry table."
            )
        if local_transpose_diagnostic_accepted_step is not None:
            core_setup = prepare_realtime_geometry_support_segment_core_setup(
                args=args,
                config=config,
                baseline_values=baseline_values,
                baseline_runtime=baseline_runtime,
                baseline_state=baseline_state,
                profile_cfg=profile_cfg,
                neoclassical_cfg=neoclassical_cfg,
                parameter_order=PARAMETER_ORDER,
                find_ntx_support_payload=_find_ntx_support_payload,
                prepare_reverse_static_setup=_prepare_reverse_static_setup,
                geometry_volume_diagnostics=_geometry_volume_diagnostics,
            )
            _run_local_stage_matvec_diagnostic_report(
                args=args,
                reverse_setup=core_setup.reverse_setup,
                accepted_step_index=int(local_transpose_diagnostic_accepted_step),
                mode_label="transport_reverse_ad_only_realtime_geometry_local_stage_matvec_diagnostic",
            )
            return
        _run_realtime_geometry_optimization_api_smoke(
            args=args,
            config=config,
            geometry_context=geometry_context,
            baseline_values=baseline_values,
            baseline_runtime=baseline_runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            neoclassical_cfg=neoclassical_cfg,
        )
        return
    if str(args.realtime_geometry_gradient_path) == "payload_boundary_probe":
        _run_realtime_geometry_payload_boundary_probe(
            args=args,
            config=config,
            baseline_values=baseline_values,
            baseline_runtime=baseline_runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            neoclassical_cfg=neoclassical_cfg,
        )
        return
    if str(args.realtime_geometry_gradient_path) == "support_pullback_probe":
        _run_realtime_geometry_support_pullback_probe(
            args=args,
            config=config,
            baseline_values=baseline_values,
            baseline_runtime=baseline_runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            neoclassical_cfg=neoclassical_cfg,
        )
        return
    if str(args.realtime_geometry_gradient_path) in {"support_segment_probe", "reverse_payload"}:
        _run_realtime_geometry_support_segment_probe(
            args=args,
            config=config,
            baseline_values=baseline_values,
            baseline_runtime=baseline_runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            neoclassical_cfg=neoclassical_cfg,
        )
        return
    if str(args.realtime_geometry_gradient_path) == "initial_carry_boundary_probe":
        _run_realtime_geometry_initial_carry_boundary_probe(
            args=args,
            config=config,
            baseline_values=baseline_values,
            baseline_runtime=baseline_runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            neoclassical_cfg=neoclassical_cfg,
        )
        return
    raise SystemExit(
        "[autodiff-gate] unsupported realtime geometry gradient path "
        f"{args.realtime_geometry_gradient_path!r}. Supported diagnostics are "
        "payload_boundary_probe, support_pullback_probe, support_segment_probe, "
        "and initial_carry_boundary_probe. The final combined path is "
        "reverse_payload."
    )
    objective_index = None if args.objective == "all" else OBJECTIVE_LABELS.index(args.objective)

    objective_vector_fn = _make_reverse_geometry_objective_vector_with_schedule_vjp(
        config=config,
        geometry_context=geometry_context,
        profile_cfg=profile_cfg,
        geometry_parameter_name=geometry_parameter,
        fixed_initial_er=jax.lax.stop_gradient(baseline_state.Er),
        accepted_step_limit_override=args.accepted_step_limit,
        solver_override=static_solver,
    )
    objective_fn = (
        objective_vector_fn
        if args.objective == "all"
        else lambda p: objective_vector_fn(p)[objective_index]  # noqa: E731
    )

    print(
        "[autodiff-gate] progress: running reverse realtime-geometry AD"
        + (" for all objectives" if args.objective == "all" else ""),
        flush=True,
    )
    reverse_compile_plus_execute_s = None
    reverse_execute_s = None
    reverse_execute_times_s: list[float] = []
    gradient = None
    t_reverse_start = time.perf_counter()

    if args.objective == "all":
        if args.timing_mode == "jit-compile-only":
            jac_fn = jax.jit(jax.jacrev(objective_fn))
            compiled_jac_fn = jac_fn.lower(baseline_values).compile()
            del compiled_jac_fn
            reverse_total_s = time.perf_counter() - t_reverse_start
        elif args.timing_mode == "jit-warm":
            jac_fn = jax.jit(jax.jacrev(objective_fn))
            first_gradient = jac_fn(baseline_values)
            first_gradient = jax.block_until_ready(first_gradient)
            reverse_compile_plus_execute_s = time.perf_counter() - t_reverse_start
            gradient = first_gradient
            for _ in range(max(1, int(args.warm_repeats))):
                t_execute_start = time.perf_counter()
                gradient = jac_fn(baseline_values)
                gradient = jax.block_until_ready(gradient)
                reverse_execute_times_s.append(time.perf_counter() - t_execute_start)
            reverse_execute_s = float(np.mean(reverse_execute_times_s))
            reverse_total_s = reverse_compile_plus_execute_s + float(np.sum(reverse_execute_times_s))
        else:
            gradient = jax.jacrev(objective_fn)(baseline_values)
            gradient = jax.block_until_ready(gradient)
            reverse_total_s = time.perf_counter() - t_reverse_start
    else:
        if args.timing_mode == "jit-compile-only":
            grad_fn = jax.jit(jax.grad(objective_fn))
            compiled_grad_fn = grad_fn.lower(baseline_values).compile()
            del compiled_grad_fn
            reverse_total_s = time.perf_counter() - t_reverse_start
        elif args.timing_mode == "jit-warm":
            grad_fn = jax.jit(jax.grad(objective_fn))
            first_gradient = grad_fn(baseline_values)
            first_gradient = jax.block_until_ready(first_gradient)
            reverse_compile_plus_execute_s = time.perf_counter() - t_reverse_start
            gradient = first_gradient
            for _ in range(max(1, int(args.warm_repeats))):
                t_execute_start = time.perf_counter()
                gradient = grad_fn(baseline_values)
                gradient = jax.block_until_ready(gradient)
                reverse_execute_times_s.append(time.perf_counter() - t_execute_start)
            reverse_execute_s = float(np.mean(reverse_execute_times_s))
            reverse_total_s = reverse_compile_plus_execute_s + float(np.sum(reverse_execute_times_s))
        else:
            gradient = jax.grad(objective_fn)(baseline_values)
            gradient = jax.block_until_ready(gradient)
            reverse_total_s = time.perf_counter() - t_reverse_start

    objective_values_by_name = None
    if args.timing_mode != "jit-compile-only":
        objective_values = objective_vector_fn(baseline_values)
        objective_values = jax.block_until_ready(objective_values)
        objective_values_np = np.asarray(jax.device_get(objective_values), dtype=float)
        objective_values_by_name = {
            objective_name: float(value)
            for objective_name, value in zip(OBJECTIVE_LABELS, objective_values_np.tolist())
        }

    gradient_payload = None
    if gradient is not None:
        gradient_np = np.asarray(jax.device_get(gradient), dtype=float)
        if args.objective == "all":
            gradient_payload = {
                objective_name: {
                    parameter_name: float(value)
                    for parameter_name, value in zip(parameter_order, gradient_np[objective_i].tolist())
                }
                for objective_i, objective_name in enumerate(OBJECTIVE_LABELS)
            }
        else:
            gradient_payload = {
                parameter_name: float(value)
                for parameter_name, value in zip(parameter_order, gradient_np.tolist())
            }

    report = {
        "mode": "transport_reverse_ad_only",
        "parameter_mode": str(args.reverse_parameter_mode),
        "config_path": str(Path(args.config)),
        "objective_name": args.objective,
        "objective_order": list(OBJECTIVE_LABELS) if args.objective == "all" else None,
        "objective_values": objective_values_by_name,
        "parameter_order": list(parameter_order),
        "baseline_values": np.asarray(jax.device_get(baseline_values), dtype=float).tolist(),
        "accepted_step_limit": None if args.accepted_step_limit is None else int(args.accepted_step_limit),
        "ntx_exact_derivative_mode": str(args.ntx_exact_derivative_mode),
        "effective_ntx_exact_derivative_mode": effective_ntx_exact_derivative_mode,
        "ntx_exact_derivative_field_pullback_mode": str(args.ntx_exact_derivative_field_pullback_mode),
        "ntx_exact_derivative_pullback_boundary": str(args.ntx_exact_derivative_pullback_boundary),
        "ntx_exact_derivative_pullback_algebra": str(args.ntx_exact_derivative_pullback_algebra),
        "reverse_ntx_prepared_solve_boundary": str(args.reverse_ntx_prepared_solve_boundary),
        "ntx_exact_radial_batch_size": neoclassical_cfg.get("ntx_exact_radial_batch_size"),
        "ntx_exact_radial_batch_mode": neoclassical_cfg.get("ntx_exact_radial_batch_mode", "simple"),
        "ntx_exact_scan_batch_size": neoclassical_cfg.get("ntx_exact_scan_batch_size"),
        "ntx_exact_preload_support": neoclassical_cfg.get("preload_support", "config"),
        "radau_jacobian_reuse_mode": None if args.radau_jacobian_reuse_mode is None else str(args.radau_jacobian_reuse_mode),
        "reverse_geometry_parameter": str(args.reverse_geometry_parameter),
        "realtime_geometry_gradient_path": "pending_reverse_geometry_payload",
        "timing_mode": str(args.timing_mode),
        "reverse_total_s": float(reverse_total_s),
        "reverse_compile_plus_execute_s": None if reverse_compile_plus_execute_s is None else float(reverse_compile_plus_execute_s),
        "reverse_execute_s": None if reverse_execute_s is None else float(reverse_execute_s),
        "reverse_execute_times_s": [float(value) for value in reverse_execute_times_s],
        "gradient_reverse_ad": gradient_payload,
    }

    print(
        f"[autodiff-gate] mode=transport_reverse_ad_only objective={args.objective} "
        f"parameter_mode={args.reverse_parameter_mode} "
        f"parameters={list(parameter_order)} "
        f"radau_jacobian_reuse_mode={args.radau_jacobian_reuse_mode} "
        f"effective_ntx_exact_derivative_mode={effective_ntx_exact_derivative_mode} "
        f"ntx_exact_derivative_field_pullback_mode={args.ntx_exact_derivative_field_pullback_mode} "
        f"ntx_exact_derivative_pullback_boundary={args.ntx_exact_derivative_pullback_boundary} "
        f"ntx_exact_derivative_pullback_algebra={args.ntx_exact_derivative_pullback_algebra} "
        f"reverse_ntx_prepared_solve_boundary={args.reverse_ntx_prepared_solve_boundary} "
        f"ntx_exact_preload_support={neoclassical_cfg.get('preload_support', 'config')} "
        "realtime_geometry_gradient_path=pending_reverse_geometry_payload "
        f"timing_mode={args.timing_mode} "
        f"reverse_total_s={reverse_total_s:.6e}"
    )
    if reverse_compile_plus_execute_s is not None:
        print(
            f"[autodiff-gate] timing reverse_compile_plus_execute_s={reverse_compile_plus_execute_s:.6e} "
            f"reverse_execute_s_mean={reverse_execute_s:.6e} "
            f"reverse_execute_s_min={min(reverse_execute_times_s):.6e} "
            f"reverse_execute_repeats={len(reverse_execute_times_s)}"
        )
        print(
            "[autodiff-gate] timing reverse_execute_times_s="
            + ",".join(f"{float(value):.6e}" for value in reverse_execute_times_s)
        )
    if gradient_payload is not None:
        if args.objective == "all":
            if objective_values_by_name is not None:
                print("[autodiff-gate] objective values:")
                for objective_name in OBJECTIVE_LABELS:
                    print(f"  - {objective_name}: value={objective_values_by_name[objective_name]:.6e}")
            print("[autodiff-gate] reverse gradients by objective:")
            for objective_name in OBJECTIVE_LABELS:
                print(f"  - {objective_name}:")
                for parameter_name in parameter_order:
                    print(
                        f"      d{objective_name}/d{parameter_name}: "
                        f"rev={gradient_payload[objective_name][parameter_name]:.16e}"
                    )
        else:
            print("[autodiff-gate] reverse gradients:")
            for parameter_name in parameter_order:
                print(
                    f"  - d{args.objective}/d{parameter_name}: "
                    f"rev={gradient_payload[parameter_name]:.16e}"
                )

    outpath = _report_path(args.objective)
    outpath.write_text(json.dumps(report, indent=2))
    print(f"Wrote {outpath.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reverse-only adaptive benchmark lane using the current reverse-capable realized-schedule helper."
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="Benchmark TOML.")
    parser.add_argument(
        "--objective",
        type=str,
        default="softmax_Er",
        choices=tuple(dict.fromkeys(tuple(OBJECTIVE_LABELS) + tuple(INITIAL_ER_ROOT_ONLY_EXPLICIT_OBJECTIVES) + ("all",))),
        help=(
            "Scalar objective for reverse mode. Use 'all' to return the "
            "objective-by-parameter reverse derivative matrix for every metric."
        ),
    )
    parser.add_argument(
        "--reverse-all-objectives-mode",
        type=str,
        default="jacrev",
        choices=("jacrev", "vmap_pullback", "multi_rhs_reduced"),
        help=(
            "Implementation used when --objective all. 'jacrev' preserves the "
            "current vector-objective reverse path. 'vmap_pullback' rolls out "
            "final_y once, builds one final-y cotangent per objective, and vmaps "
            "the realized-schedule pullback over those cotangents. "
            "'multi_rhs_reduced' shares segmented replay across objective rows "
            "and maps the existing reduced step adjoint inside each slot."
        ),
    )
    parser.add_argument(
        "--reverse-parameter-mode",
        type=str,
        default="profiles",
        choices=("profiles", "profiles_plus_realtime_geometry"),
        help=(
            "Differentiated parameter set. 'profiles' preserves the current "
            "profile-only reverse lane. 'profiles_plus_realtime_geometry' adds "
            "selected realtime VMEC boundary parameters to the active profile parameters."
        ),
    )
    parser.add_argument(
        "--reverse-geometry-parameter",
        type=str,
        default="RBC:1:0",
        help=(
            "Realtime VMEC geometry parameter used when --reverse-parameter-mode "
            "is profiles_plus_realtime_geometry. Syntax: FAMILY:m:n, e.g. RBC:1:0. "
            "Use comma-separated values for multiple harmonics, e.g. RBC:1:0,ZBS:1:0. "
            "Use 'all' to pull back to all selected VMEC boundary harmonics."
        ),
    )
    parser.add_argument(
        "--reverse-geometry-families",
        type=str,
        default="RBC,ZBS",
        help=(
            "Comma-separated harmonic families used when --reverse-geometry-parameter all. "
            "Currently supports RBC and ZBS."
        ),
    )
    parser.add_argument(
        "--reverse-geometry-include-zero-harmonics",
        action="store_true",
        help=(
            "When --reverse-geometry-parameter all, include zero-valued boundary "
            "harmonics as well as nonzero harmonics."
        ),
    )
    parser.add_argument(
        "--reverse-geometry-print-limit",
        type=int,
        default=16,
        help=(
            "Print every geometry harmonic gradient only when the selected harmonic "
            "count is at most this value; otherwise print top-k by magnitude."
        ),
    )
    parser.add_argument(
        "--reverse-geometry-print-top-k",
        type=int,
        default=12,
        help=(
            "Number of largest-magnitude geometry harmonic gradients printed per "
            "objective when the full selected table is too large for stdout."
        ),
    )
    parser.add_argument(
        "--realtime-geometry-gradient-path",
        type=str,
        default="reverse_payload",
        choices=(
            "reverse_payload",
            "payload_boundary_probe",
            "support_pullback_probe",
            "support_segment_probe",
            "initial_carry_boundary_probe",
        ),
        help=(
            "Implementation used only with --reverse-parameter-mode "
            "profiles_plus_realtime_geometry. 'payload_boundary_probe' verifies "
            "the static-object/differentiable-NTX-support boundary without "
            "claiming to compute the missing geometry reverse gradient. "
            "'support_pullback_probe' checks the local lagged-RHS support "
            "cotangent hook that the full reverse payload path needs. "
            "'support_segment_probe' runs the realized-schedule reduced reverse "
            "replay and accumulates cotangents wrt the explicit support payload. "
            "'initial_carry_boundary_probe' returns the compact cotangent wrt "
            "the initial transport state, matching the profile reverse boundary "
            "before composing with VMEC/NTX geometry parameters. "
            "'reverse_payload' runs the same realized-schedule reduced reverse "
            "replay with a combined runtime.geometry + NTX-support payload, then "
            "contracts that payload back to the selected VMEC harmonic."
        ),
    )
    parser.add_argument(
        "--support-pullback-probe-include-build",
        action="store_true",
        help=(
            "Only for --realtime-geometry-gradient-path support_pullback_probe. "
            "Also probe the generic support VJP through build_lagged_response. "
            "This is intentionally off by default because that map materializes "
            "a very large NTX graph and can OOM."
        ),
    )
    parser.add_argument(
        "--realtime-geometry-component-pullbacks",
        action="store_true",
        help=(
            "For profiles_plus_realtime_geometry reverse_payload runs, also pull "
            "objective_explicit/transport_rhs/initial_cache/initial_profile "
            "diagnostic support components back to VMEC harmonics. Off by default "
            "to avoid several extra payload-to-VMEC RHS batches once the "
            "decomposition has been validated."
        ),
    )
    parser.add_argument(
        "--skip-realtime-geometry-support-bar-diagnostics",
        action="store_true",
        help=(
            "For profiles_plus_realtime_geometry reverse_payload all-objective runs, "
            "skip support/geometry payload cotangent l2/finiteness tree diagnostics "
            "and JSON summaries. This does not skip the support cotangents used for "
            "the derivative; it only avoids extra host-side diagnostic scans."
        ),
    )
    parser.add_argument(
        "--optimization-api-smoke",
        action="store_true",
        help=(
            "For profiles_plus_realtime_geometry runs, exercise the production-style "
            "least-squares API through the direct JAX table-result builder and then "
            "exit. This uses the same validated grouped reverse runner but does not "
            "write the normal benchmark report."
        ),
    )
    parser.add_argument(
        "--reverse-final-objective-cotangent-mode",
        choices=("scalar", "grouped_vjp"),
        default="scalar",
        help=(
            "Terminal objective-cotangent construction for the segmented realtime "
            "geometry reverse path. 'scalar' preserves the per-objective VJPs. "
            "'grouped_vjp' groups all non-bootstrap objectives into one final-state "
            "VJP and one explicit-geometry VJP; bootstrap keeps its compact rule."
        ),
    )
    parser.add_argument(
        "--full-transport-shared-payload-smoke",
        action="store_true",
        help=(
            "For profiles_plus_realtime_geometry reverse_payload runs, execute only "
            "the full transport internal realtime-geometry table path and write a "
            "shared-path JSON report for offline comparison against saved reference "
            "benchmark output. This does not run the reference path in the same process."
        ),
    )
    parser.add_argument(
        "--max-reverse-accepted-steps",
        type=int,
        default=None,
        help=(
            "When reverse AD is run to the solver t_final rather than an explicit "
            "--accepted-step-limit, use this as a schedule-discovery guard. The "
            "reverse path still requires reaching t_final; exceeding this guard is "
            "reported as a failed trial instead of compiling a max_steps-sized trace."
        ),
    )
    parser.add_argument(
        "--transport-solver-forward-smoke",
        action="store_true",
        help=(
            "For profiles_plus_realtime_geometry setup, run only the production transport "
            "solver forward from the benchmark initial state and print objective values. "
            "This is useful for theta/theta_newton backend checks before running reverse."
        ),
    )
    parser.add_argument(
        "--optimization-api-profile-dofs",
        choices=("include", "exclude"),
        default="include",
        help=(
            "Only for --optimization-api-smoke. Choose whether the optimization "
            "parameter vector includes profile DOFs in addition to realtime VMEC "
            "geometry DOFs. Use 'exclude' for geometry-only optimization."
        ),
    )
    parser.add_argument(
        "--initial-Er-root-only-optimization-smoke",
        dest="initial_er_root_only_optimization_smoke",
        action="store_true",
        help=(
            "Exercise the initial ambipolar-Er optimization API for Er objectives "
            "without preparing or running the Radau time-evolution solver."
        ),
    )
    parser.add_argument(
        "--initial-Er-root-shared-payload-compare-smoke",
        dest="initial_er_root_shared_payload_compare_smoke",
        action="store_true",
        help=(
            "With --initial-Er-root-only-optimization-smoke and realtime geometry "
            "DOFs, run only the shared-payload/fused root-only path and write "
            "its own report for offline comparison against saved reference JSON."
        ),
    )
    parser.add_argument(
        "--initial-Er-root-ad",
        dest="initial_er_root_ad",
        default="off",
        choices=("off", "jax_selected_root"),
        help=(
            "Opt-in AD treatment for ambipolar initial Er. 'off' preserves the "
            "validated benchmark behavior. 'jax_selected_root' recomputes the "
            "same selected best-root profile with a JAX-returning root path so "
            "the initial-Er boundary can participate in VJP/JVP diagnostics."
        ),
    )
    parser.add_argument("--device", type=str, default=None, help="Optional device override.")
    parser.add_argument(
        "--transport-solver-backend-override",
        choices=("config", "radau", "theta", "theta_newton"),
        default="config",
        help=(
            "Override transport_solver_backend/integrator in memory for this benchmark run. "
            "Use theta_newton to test the theta/TORAX-style production solve against the "
            "same benchmark TOML without editing the validated Radau config. Normal reverse "
            "uses reverse_stage_cotangent_mode='full' and dispatches by the configured solver."
        ),
    )
    parser.add_argument(
        "--accepted-step-limit",
        type=int,
        default=None,
        help="Optional accepted-step prefix to stop the adaptive rollout.",
    )
    parser.set_defaults(
        ntx_exact_derivative_mode=DEFAULT_NTX_EXACT_DERIVATIVE_MODE,
        ntx_exact_derivative_field_pullback_mode=DEFAULT_NTX_EXACT_DERIVATIVE_FIELD_PULLBACK_MODE,
        ntx_exact_derivative_pullback_boundary=DEFAULT_NTX_EXACT_DERIVATIVE_PULLBACK_BOUNDARY,
        ntx_exact_derivative_pullback_algebra=DEFAULT_NTX_EXACT_DERIVATIVE_PULLBACK_ALGEBRA,
        reverse_ntx_prepared_solve_boundary="default",
    )
    parser.add_argument(
        "--ntx-exact-derivative-pullback-algebra",
        choices=("ntx_helper", "ntx_helper_lowdot_fused"),
        default=DEFAULT_NTX_EXACT_DERIVATIVE_PULLBACK_ALGEBRA,
        help=(
            "Exact-runtime NTX local derivative pullback algebra. 'ntx_helper' "
            "is the validated default. 'ntx_helper_lowdot_fused' is an isolated "
            "experimental mode that fuses the base, d/dEr, and d/dlog(nu) local "
            "contractions; it does not select the joint prepared-support path."
        ),
    )
    parser.add_argument(
        "--ntx-radial-batch-size",
        type=int,
        default=None,
        help=(
            "Exact-runtime NTX radial batch size for this reverse-AD lane. "
            "Unset/0 preserves the config default; values >1 enable the "
            "runtime radial mapper selected by --ntx-radial-batch-mode."
        ),
    )
    parser.add_argument(
        "--ntx-radial-batch-mode",
        default=None,
        choices=("simple", "lax_map", "vmap", "hybrid"),
        help=(
            "Exact-runtime NTX radial mapper override. 'hybrid' uses chunked "
            "lax.map over radial batches with vmap inside each chunk."
        ),
    )
    parser.add_argument(
        "--ntx-scan-batch-size",
        type=int,
        default=None,
        help=(
            "Exact-runtime NTX coefficient-scan batch size across energy-grid "
            "cases. Unset/0 preserves the config default; values >1 chunk the "
            "energy/collisionality scan."
        ),
    )
    parser.add_argument(
        "--ntx-exact-preload-support",
        choices=("config", "true", "false"),
        default="config",
        help=(
            "Reverse-lane-only preload_support override for ntx_exact_lij_runtime. "
            "'config' preserves the current orchestrator/config behavior. 'false' "
            "tests whether preloaded NTX support arrays captured by the static "
            "reverse custom-VJP context are driving compile-memory constants."
        ),
    )
    parser.add_argument(
        "--radau-jacobian-reuse-mode",
        type=str,
        default=None,
        help="Optional Radau Jacobian reuse mode override, e.g. legacy or retry_only.",
    )
    parser.add_argument(
        "--baseline-diagnostics",
        action="store_true",
        help="Run an extra primal schedule rollout to print attempt/accepted counts before reverse AD.",
    )
    parser.add_argument(
        "--reverse-segment-length",
        type=int,
        default=None,
        help=(
            "Optional reverse checkpoint segment length for accepted-step replay. "
            "Omit for the current unsegmented reference path."
        ),
    )
    parser.add_argument(
        "--reverse-direct-stage-adjoint",
        action="store_true",
        help=(
            "Use the reverse-only structured accepted-step adjoint. This is the default; "
            "the flag is kept as an explicit marker for old command lines."
        ),
    )
    parser.add_argument(
        "--reverse-transpose-fallback",
        action="store_true",
        help=(
            "Use the older transpose-of-forward-tangent helper instead of the "
            "structured reverse accepted-step adjoint. Intended only for comparisons."
        ),
    )
    parser.add_argument(
        "--reverse-stage-adjoint-solve-mode",
        choices=(
            "structured",
            "bicgstab",
            "block",
            "block_colored_ntss_midpoint",
            "block_explicit_ntx_jacobian",
            "block_frozen_forward_jacobian",
            "gmres",
            "exact_block_compact",
            "woodbury_compact",
            "woodbury_matvec_compact",
            "woodbury_er_coeff_compact",
        ),
        default="structured",
        help=(
            "Reverse stage-adjoint linear solve. 'structured' uses the Radau "
            "transformed LU transpose approximation and is the lightweight default; "
            "'bicgstab' is the lower-memory exact iterative candidate; 'block' and "
            "'gmres' are correctness oracles but are memory/compile heavy; "
            "'block_colored_ntss_midpoint' is an isolated exact candidate for the "
            "NTSS-midpoint model: it reconstructs the dense block transpose from "
            "colored local actions plus the analytic rank-three correction, then "
            "uses the same dense multi-RHS solve as 'block'; "
            "'block_explicit_ntx_jacobian' keeps the exact block system but materializes "
            "each fixed-lagged NTX stage Jacobian from the explicit state pullback; "
            "'block_frozen_forward_jacobian' uses each replayed primal step's frozen "
            "jacobian_out for every stage and is a non-exact comparison mode; "
            "'exact_block_compact' requires a non-dense exact compact solve hook; "
            "'woodbury_compact' uses the experimental rank-truncated block-Woodbury solve; "
            "'woodbury_matvec_compact' builds the same Woodbury system from the compact "
            "transpose matvec instead of jacfwd; 'woodbury_er_coeff_compact' uses compact "
            "radial bands plus a skinny Er-coefficient Woodbury correction."
        ),
    )
    parser.add_argument(
        "--reverse-rhs-pullback-mode",
        choices=("separate", "fused_ntx"),
        default="separate",
        help=(
            "Exact fixed-lagged RHS pullback dispatch. 'separate' preserves the "
            "reference state/lagged/support calls. 'fused_ntx' is an opt-in "
            "NEOPAX-only experiment that shares NTX shared-flux assembly; the "
            "realtime-geometry payload path retains its established pullback."
        ),
    )
    parser.add_argument(
        "--reverse-initial-cache-support-pullback-mode",
        choices=(
            "scalar",
            "ntx_batched_interpolated_faces",
            "ntx_native_joint_state_and_support",
        ),
        default="scalar",
        help=(
            "Initial lagged-cache support transpose. 'scalar' preserves the reference "
            "lax.map path. 'ntx_batched_interpolated_faces' is an exact, opt-in "
            "multi-objective NTX face-interpolation transpose; it is limited to the "
            "realtime interpolate_from_faces configuration and has no scalar fallback. "
            "'ntx_native_joint_state_and_support' is a separate opt-in path that "
            "uses one native multi-RHS initial lagged transpose for both state and "
            "support cotangents."
        ),
    )
    parser.add_argument(
        "--reverse-rebuild-support-pullback-mode",
        choices=(
            "separate",
            "separate_reuse_local_vjp_primal",
            "separate_reuse_local_vjp_primal_geometry_only_prepared",
            "separate_reuse_local_vjp_primal_geometry_implicit_ntx_two_directional",
            "separate_reuse_local_vjp_primal_support_only_ntx_implicit",
            "separate_reuse_local_vjp_primal_factorized_ntx_two_directional",
            "ntx_batched_interpolated_faces",
            "ntx_batched_interpolated_faces_reuse_local_vjp_primal",
            "ntx_batched_interpolated_faces_multi_rhs_shared_primal",
            "ntx_batched_interpolated_faces_native_multi_rhs_shared_primal",
            "ntx_batched_interpolated_faces_native_multi_rhs_compact_shared_primal",
            "ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal",
            "ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients",
            "ntx_batched_interpolated_faces_native_multi_rhs_compact_residual_reuse_moment_drds_jvp_shared_primal",
            "ntx_joint_implicit_interpolated_faces",
            "ntx_joint_implicit_interpolated_faces_packed_support_adjoint",
            "ntx_joint_implicit_interpolated_faces_reuse_local_vjp_primal",
            "ntx_joint_implicit_interpolated_faces_reuse_local_vjp_primal_compact_prepared_carry",
        ),
        default="separate",
        help=(
            "Lagged-response rebuild support transpose inside each reverse step. "
            "'separate' preserves the reference vmapped scalar path. "
            "'separate_reuse_local_vjp_primal' is an exact experimental variant "
            "that uses the primal output of each existing local NTX VJP instead "
            "of a separate anchor-response construction. "
            "'separate_reuse_local_vjp_primal_geometry_only_prepared' is an exact "
            "runtime-grid-fixed variant of that mode: it differentiates the local "
            "NTX response with respect to GeometryOnGrid and drds only, keeping "
            "the fixed d_theta/d_zeta operators outside the VJP. "
            "'separate_reuse_local_vjp_primal_geometry_implicit_ntx_two_directional' "
            "is an isolated exact NTX implicit-adjoint variant returning only "
            "GeometryOnGrid support bars while retaining the same local primal "
            "response reuse. "
            "'separate_reuse_local_vjp_primal_support_only_ntx_implicit' is an "
            "isolated exact experimental variant that retains that anchor-primal "
            "reuse while omitting NTX case/profile bars unused by the rebuild "
            "support transpose. "
            "'separate_reuse_local_vjp_primal_factorized_ntx_two_directional' is an "
            "isolated exact experimental rebuild-support mode: for each energy it "
            "uses NTX's one-factorization base-plus-two-directional primitive and "
            "its matching local prepared custom VJP. "
            "'ntx_batched_interpolated_faces' is an exact opt-in multi-objective "
            "NTX face-interpolation transpose for the rebuild branch; it has no "
            "fallback for unsupported response representations. "
            "'ntx_batched_interpolated_faces_multi_rhs_shared_primal' is an "
            "isolated exact experiment: per anchor/species it shares NTX's local "
            "primal factorisation and two forward case directions across the "
            "objective RHS batch through the older objective-vmapped support rule. "
            "'ntx_batched_interpolated_faces_native_multi_rhs_shared_primal' is "
            "the separate native matrix-RHS experiment: it packs all implicit "
            "adjoint RHS columns before the final support-gradient contraction "
            "without a scalar objective VJP loop. "
            "'ntx_batched_interpolated_faces_native_multi_rhs_compact_shared_primal' "
            "uses the same native matrix-RHS algebra but compacts its temporary "
            "prepared and case-bar payload before NEOPAX reduces over energy. "
            "'ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal' "
            "uses the native matrix-RHS algebra while reusing the directional drds "
            "JVPs already returned by its joint local moment pullback. "
            "'ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients' "
            "is the isolated AD-comparison variant of that native mode: it carries "
            "the grouped NTX face-coefficient bars to the existing VMEC raw-block "
            "transpose and replaces the generic face-prepared support branch. "
            "'ntx_joint_implicit_interpolated_faces' is the exact experimental "
            "mode that obtains rebuild state and NTX-support bars jointly from "
            "the same local NTX implicit adjoint. "
            "'ntx_joint_implicit_interpolated_faces_packed_support_adjoint' is "
            "the same experimental joint lane, but uses NTX's separately tested "
            "paired directional support-adjoint solve. "
            "'ntx_joint_implicit_interpolated_faces_reuse_local_vjp_primal' "
            "keeps the joint state/support adjoint but reuses its local primal "
            "response for the interpolation-coordinate transpose. "
            "'ntx_joint_implicit_interpolated_faces_reuse_local_vjp_primal_compact_prepared_carry' "
            "keeps that same joint lowdot calculation while packing only the "
            "prepared-support anchor-scan carry; it is an isolated exact "
            "compile-layout experiment."
        ),
    )
    parser.add_argument(
        "--reverse-segment-jit-diagnostics",
        action="store_true",
        help=(
            "Print JAX in-process trace-cache counters before/after each reverse segment. "
            "Diagnostic only: it does not alter reverse computation and is not an XLA "
            "persistent-cache metric."
        ),
    )
    parser.add_argument(
        "--reverse-segment-input-diagnostics",
        action="store_true",
        help=(
            "After each already-synchronized reverse segment, print active slots, "
            "accepted dt range, lagged reuse/rebuild pattern, and incoming Jacobian "
            "cache metadata. Diagnostic only; it does not alter reverse computation."
        ),
    )
    parser.add_argument(
        "--reverse-rebuild-component-timing",
        action="store_true",
        help=(
            "Diagnostic-only: separately JIT and time one representative rebuild "
            "state transpose and support transpose with device synchronization. "
            "With --reverse-segment-primal-record-mode reuse_segment_primal_record, "
            "also reports the logical record payload and separately times record replay "
            "and record-consuming reverse without rebuild transposes. This adds extra "
            "work and is not a normal reverse timing."
        ),
    )
    parser.add_argument(
        "--reverse-table-timing-diagnostics",
        action="store_true",
        help=(
            "Print host timing boundaries for reverse-table setup, transport-table "
            "execution, geometry-table execution, and final assembly. Diagnostic "
            "only; it does not change reverse math or retained payloads."
        ),
    )
    parser.add_argument(
        "--reverse-segment-profiler-trace-dir",
        type=str,
        default=None,
        help=(
            "Optional directory for an XProf trace of the normal reverse benchmark. "
            "This enables XProf labels for the stage solve, fixed-lagged RHS "
            "transposes, and rebuild transposes; it does not split the JIT or "
            "change reverse mathematics."
        ),
    )
    parser.add_argument(
        "--reverse-segment-start-replay-mode",
        choices=("legacy", "minimal"),
        default="legacy",
        help=(
            "Segment-start carry reconstruction after the accepted schedule is fixed. "
            "legacy preserves the full accepted-step replay. minimal uses the exact "
            "reverse-minimal Radau reconstruction and skips unused adaptive diagnostics."
        ),
    )
    parser.add_argument(
        "--reverse-segment-primal-record-mode",
        choices=("reconstruct", "reuse_segment_primal_record"),
        default="reconstruct",
        help=(
            "Experimental exact bounded-memory segment mode. "
            "reuse_segment_primal_record retains each minimal accepted-step primal "
            "record only for the active reverse segment, so the step adjoint reuses "
            "it instead of reconstructing the same accepted attempt. Requires "
            "--reverse-segment-start-replay-mode minimal. Default reconstruct "
            "preserves the current behavior."
        ),
    )
    parser.add_argument(
        "--reverse-single-segment-vjp-forward-mode",
        choices=("legacy", "reuse_adaptive_rollout"),
        default="legacy",
        help=(
            "Experimental one-segment-only VJP-forward setup. "
            "reuse_adaptive_rollout reuses carry0 and the adaptive final carry, "
            "skipping the otherwise redundant accepted-schedule replay. "
            "Default legacy preserves the current path."
        ),
    )
    parser.add_argument(
        "--reverse-schedule-artifact-mode",
        choices=("legacy", "reuse_static_probe"),
        default="legacy",
        help=(
            "Experimental Radau shared-payload mode. reuse_static_probe reuses "
            "the compact adaptive schedule already built during reverse setup, "
            "rather than executing a second adaptive schedule rollout in the "
            "manual VJP forward. It stores no carries or per-step primal tape."
        ),
    )
    parser.add_argument(
        "--reverse-stage-adjoint-woodbury-rank",
        type=int,
        default=24,
        help="Rank used by --reverse-stage-adjoint-solve-mode woodbury_compact.",
    )
    parser.add_argument(
        "--reverse-rhs-transpose-mode",
        choices=("generic", "explicit_ntx_interpolated"),
        default="generic",
        help=(
            "RHS-state transpose used inside exact reverse stage-adjoint matvecs. "
            "'generic' is the known-good JAX VJP reference; "
            "'explicit_ntx_interpolated' opts into the experimental explicit NTX state pullback."
        ),
    )
    parser.add_argument(
        "--reverse-stage-cotangent-mode",
        choices=(
            "full",
            "zero_lagged",
            "zero_rhs_state",
            "zero_rhs_direct",
            "zero_rhs_flux",
            "zero_stage_solve",
            "zero_rebuild_pullback",
            "zero_rebuild_anchor_fields",
            "zero_rebuild_local_moment_pullback",
            "scan_rebuild_local_moment_pullback",
            "scan_rebuild_anchor_pullback",
            "zero_step_bwd",
            "theta_state_only",
            "theta_zero_lagged",
            "theta_compact_support_probe",
            "theta_implicit_transpose_probe",
            "force_reuse_bwd",
            "force_rebuild_bwd",
            "dynamic_call_bwd",
        ),
        default="full",
        help=(
            "Diagnostic-only branch toggle for exact stage adjoints. 'full' is the normal "
            "reverse lane; 'zero_lagged' drops stage lagged-response cotangents; "
            "'zero_rhs_state' drops stage RHS-state cotangents, including inside the exact "
            "iterative transpose matvec; 'zero_rhs_direct' keeps only shared-flux state "
            "cotangents; 'zero_rhs_flux' keeps only direct equation-assembly state "
            "cotangents; 'zero_stage_solve' bypasses the exact stage-adjoint solve and "
            "residual-input pullback; 'zero_rebuild_pullback' skips only the lagged-response "
            "rebuild pullback in rebuild branches; 'zero_rebuild_anchor_fields' keeps only "
            "the direct reference-Er part of the NTX interpolated rebuild pullback; "
            "'zero_rebuild_local_moment_pullback' keeps the rebuild interpolation transpose "
            "but skips the local NTX moment-response pullback; "
            "'scan_rebuild_local_moment_pullback' keeps the local NTX moment-response "
            "pullback exact but scans over species instead of materializing a species stack; "
            "'scan_rebuild_anchor_pullback' additionally scans over rebuild anchors and "
            "accumulates state bars directly; "
            "'zero_step_bwd' bypasses the accepted-step "
            "backward body inside segmented replay; for theta solvers, 'full' dispatches "
            "to the one-step theta implicit residual transpose; theta-only diagnostics "
            "'theta_state_only' and 'theta_zero_lagged' replay the theta realized schedule "
            "through state/carry cotangents but intentionally return zero support-payload bars; "
            "'theta_compact_support_probe' additionally threads support as a VJP primal and "
            "uses compact lagged-RHS support pullbacks for diagnosis; "
            "'theta_implicit_transpose_probe' uses a one-step theta residual transpose "
            "and compact RHS/rebuild support pullbacks; "
            "'force_reuse_bwd' and 'force_rebuild_bwd' "
            "compile only one lagged-response backward branch for diagnosis. Most non-full "
            "diagnostic modes intentionally change gradients unless the forced branch matches the "
            "realized primal branch for every accepted step; 'scan_rebuild_local_moment_pullback' "
            "and 'scan_rebuild_anchor_pullback' are intended to preserve gradients; "
            "'dynamic_call_bwd' keeps the dynamic branch but puts each branch body behind "
            "a non-inlined compiled call boundary."
        ),
    )
    parser.add_argument(
        "--reverse-step-bwd-mode",
        choices=(
            "current",
            "manual_split",
            "reduced_cotangent",
            "reduced_cotangent_call_boundary",
            "reduced_cotangent_lean_replay",
            "reduced_cotangent_recompute_replay",
            "reduced_cotangent_host_segments",
        ),
        default="current",
        help=(
            "Accepted-step backward implementation selector. 'current' keeps the "
            "existing reverse path. 'manual_split' is reserved for the upcoming "
            "split/manual accepted-step adjoint and currently routes through the "
            "same implementation while plumbing is validated. 'reduced_cotangent' "
            "uses a reduced final-state cotangent contract inside the segmented "
            "accepted-step reverse scan. 'reduced_cotangent_call_boundary' keeps "
            "the same exact reduced-cotangent equations but places each batched "
            "step-with-support adjoint behind a non-inlined JIT call boundary. "
            "'reduced_cotangent_lean_replay' stores "
            "the per-slot replay tape after masking forward-only Radau cache fields. "
            "'reduced_cotangent_recompute_replay' recomputes each slot start from "
            "the segment checkpoint instead of storing the full per-slot carry tape. "
            "'reduced_cotangent_host_segments' is only "
            "for split-vjp timing and orchestrates segment backward kernels outside "
            "the monolithic rollout-bwd JIT."
        ),
    )
    parser.add_argument(
        "--reverse-stage-adjoint-memory-mode",
        choices=("default", "remat_matvec", "stream_rhs", "stage_call_boundary"),
        default="default",
        help=(
            "Memory strategy inside the exact reverse stage-adjoint matvec. "
            "'default' keeps the current graph. 'remat_matvec' checkpoints the "
            "per-stage RHS transpose inside the Krylov matvec while preserving "
            "the outer lax.scan structure. 'stream_rhs' accumulates each stage "
            "RHS transpose contribution into A.T @ J.T @ lambda without first "
            "materializing the full per-stage stack. 'stage_call_boundary' puts the "
            "reduced-cotangent stage adjoint solve plus residual-input pullback "
            "behind a non-inlined JIT call boundary."
        ),
    )
    parser.add_argument(
        "--reverse-stage-adjoint-iter-maxiter",
        type=int,
        default=40,
        help=(
            "Maximum Krylov iterations for exact iterative reverse stage-adjoint "
            "modes ('bicgstab'/'gmres'). Defaults to the current conservative value."
        ),
    )
    parser.add_argument(
        "--reverse-stage-adjoint-iter-tol",
        type=float,
        default=1.0e-10,
        help=(
            "Relative tolerance for exact iterative reverse stage-adjoint modes "
            "('bicgstab'/'gmres'). Defaults to the current conservative value."
        ),
    )
    parser.add_argument(
        "--timing-mode",
        choices=("eager", "jit-warm", "jit-compile-only", "split-vjp-warm"),
        default="eager",
        help=(
            "Timing harness. 'eager' preserves the original un-jitted grad timing; "
            "'jit-warm' reports first jit call and second warm execute call separately; "
            "'jit-compile-only' lowers and compiles the jitted gradient, then exits "
            "without executing it; 'split-vjp-warm' compiles the custom-VJP forward, "
            "objective cotangent, rollout backward, and parameter pullback as separate "
            "kernels instead of one monolithic jitted grad."
        ),
    )
    parser.add_argument(
        "--warm-repeats",
        type=int,
        default=1,
        help="Number of post-compile warm executions to time when --timing-mode=jit-warm.",
    )
    parser.add_argument(
        "--local-transpose-diagnostic-accepted-step",
        type=int,
        default=None,
        help=(
            "Diagnostic-only mode: run one local accepted-step dot-product transpose "
            "check at this zero-based accepted-step ordinal, then exit."
        ),
    )
    parser.add_argument(
        "--local-transpose-diagnostic-seed-mode",
        type=str,
        default="y",
        choices=(
            "y",
            "prev_stages",
            "lagged_cache",
            "lagged_reference",
            "y_lagged_cache",
            "y_lagged_reference",
            "lagged_cache_reference",
            "all",
        ),
        help="Seed channel for --local-transpose-diagnostic-accepted-step.",
    )
    parser.add_argument(
        "--local-transpose-diagnostic-input-seed-mode",
        type=str,
        default=None,
        choices=(
            "y",
            "prev_stages",
            "lagged_cache",
            "lagged_reference",
            "y_lagged_cache",
            "y_lagged_reference",
            "lagged_cache_reference",
            "all",
        ),
        help="Optional input tangent seed channel. Defaults to --local-transpose-diagnostic-seed-mode.",
    )
    parser.add_argument(
        "--local-transpose-diagnostic-output-seed-mode",
        type=str,
        default=None,
        choices=(
            "y",
            "prev_stages",
            "lagged_cache",
            "lagged_reference",
            "y_lagged_cache",
            "y_lagged_reference",
            "lagged_cache_reference",
            "all",
        ),
        help="Optional output cotangent seed channel. Defaults to --local-transpose-diagnostic-seed-mode.",
    )
    parser.add_argument(
        "--stage-matvec-diagnostic",
        action="store_true",
        help=(
            "With --local-transpose-diagnostic-accepted-step, compare the compact "
            "exact Radau stage transpose matvec against the dense block matrix, "
            "then exit. Diagnostic only; does not compute parameter gradients."
        ),
    )
    parser.add_argument(
        "--diagnose-final-objective-cotangent",
        action="store_true",
        help=(
            "For scalar objectives, also print the norm/max/nonzero count of "
            "grad(objective(final_y)) before the realized-schedule reverse rule."
        ),
    )
    args = parser.parse_args()
    if (
        str(args.reverse_schedule_artifact_mode) == "reuse_static_probe"
        and not bool(args.full_transport_shared_payload_smoke)
    ):
        raise SystemExit(
            "[autodiff-gate] --reverse-schedule-artifact-mode reuse_static_probe is currently "
            "implemented only for --full-transport-shared-payload-smoke."
        )
    if int(args.reverse_stage_adjoint_iter_maxiter) <= 0:
        raise SystemExit("[autodiff-gate] --reverse-stage-adjoint-iter-maxiter must be positive.")
    if float(args.reverse_stage_adjoint_iter_tol) <= 0.0:
        raise SystemExit("[autodiff-gate] --reverse-stage-adjoint-iter-tol must be positive.")
    if int(args.reverse_stage_adjoint_woodbury_rank) <= 0:
        raise SystemExit("[autodiff-gate] --reverse-stage-adjoint-woodbury-rank must be positive.")
    if (
        str(args.reverse_rhs_transpose_mode) == "explicit_ntx_interpolated"
        and str(args.reverse_stage_adjoint_solve_mode) == "gmres"
    ):
        raise SystemExit(
            "[autodiff-gate] --reverse-rhs-transpose-mode explicit_ntx_interpolated is not ready for "
            "JAX scipy GMRES. Use bicgstab for this experimental mode while the NTX RHS-state "
            "transpose is being specialized."
        )
    reverse_segment_length = None
    if args.reverse_segment_length is not None:
        reverse_segment_length = int(args.reverse_segment_length)
        if reverse_segment_length <= 0:
            raise SystemExit("[autodiff-gate] --reverse-segment-length must be positive when provided.")
    reverse_direct_stage_adjoint = not bool(args.reverse_transpose_fallback)
    if str(args.ntx_exact_derivative_field_pullback_mode) == "compact_vjp":
        _check_compact_ntx_derivative_pullback_available()
    effective_ntx_exact_derivative_mode = str(args.ntx_exact_derivative_mode)

    config = _prepare_benchmark_config(
        Path(args.config),
        device=args.device,
        ntx_exact_derivative_mode=effective_ntx_exact_derivative_mode,
        ntx_exact_derivative_field_pullback_mode=args.ntx_exact_derivative_field_pullback_mode,
        ntx_exact_derivative_pullback_boundary=args.ntx_exact_derivative_pullback_boundary,
        ntx_exact_derivative_pullback_algebra=args.ntx_exact_derivative_pullback_algebra,
        radau_jacobian_reuse_mode=args.radau_jacobian_reuse_mode,
    )
    _apply_transport_solver_backend_override(config, args.transport_solver_backend_override)
    if bool(args.transport_solver_forward_smoke) and args.accepted_step_limit is not None:
        config.setdefault("transport_solver", {})["stop_after_accepted_steps"] = int(args.accepted_step_limit)
    neoclassical_cfg = config.setdefault("neoclassical", {})
    if args.ntx_radial_batch_size not in (None, 0):
        neoclassical_cfg["ntx_exact_radial_batch_size"] = int(args.ntx_radial_batch_size)
    if args.ntx_radial_batch_mode not in (None, ""):
        neoclassical_cfg["ntx_exact_radial_batch_mode"] = str(args.ntx_radial_batch_mode)
    if args.ntx_scan_batch_size not in (None, 0):
        neoclassical_cfg["ntx_exact_scan_batch_size"] = int(args.ntx_scan_batch_size)
    if args.ntx_exact_preload_support != "config":
        neoclassical_cfg["preload_support"] = args.ntx_exact_preload_support == "true"
    profile_cfg = _baseline_profile_cfg(config)
    if args.reverse_parameter_mode == "profiles_plus_realtime_geometry":
        with _benchmark_device_context(config):
            _run_realtime_geometry_reverse_mode(
                args=args,
                config=config,
                profile_cfg=profile_cfg,
                effective_ntx_exact_derivative_mode=effective_ntx_exact_derivative_mode,
                neoclassical_cfg=neoclassical_cfg,
            )
        return

    runtime, baseline_state = build_runtime_context(config)
    if bool(args.initial_er_root_only_optimization_smoke):
        _run_initial_er_root_only_optimization_api_smoke(
            args=args,
            config=config,
            baseline_runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
        )
        return
    baseline_values = jnp.asarray(
        [_profile_cfg_scalar_value(profile_cfg, name) for name in PARAMETER_ORDER],
        dtype=jnp.asarray(baseline_state.pressure).dtype,
    )
    objective_index = None if args.objective == "all" else OBJECTIVE_LABELS.index(args.objective)
    reverse_setup = _prepare_reverse_static_setup(
        baseline_values,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        initial_er_root_ad=args.initial_er_root_ad,
        accepted_step_limit_override=args.accepted_step_limit,
        reverse_segment_length=reverse_segment_length,
        reverse_direct_stage_adjoint=reverse_direct_stage_adjoint,
        reverse_stage_adjoint_solve_mode=str(args.reverse_stage_adjoint_solve_mode),
        reverse_rhs_transpose_mode=str(args.reverse_rhs_transpose_mode),
        reverse_rhs_pullback_mode=str(args.reverse_rhs_pullback_mode),
        reverse_stage_cotangent_mode=str(args.reverse_stage_cotangent_mode),
        reverse_step_bwd_mode=str(args.reverse_step_bwd_mode),
        reverse_stage_adjoint_memory_mode=str(args.reverse_stage_adjoint_memory_mode),
        reverse_stage_adjoint_iter_maxiter=int(args.reverse_stage_adjoint_iter_maxiter),
        reverse_stage_adjoint_iter_tol=float(args.reverse_stage_adjoint_iter_tol),
        reverse_stage_adjoint_woodbury_rank=int(args.reverse_stage_adjoint_woodbury_rank),
    )

    if args.local_transpose_diagnostic_accepted_step is not None:
        accepted_step_index = int(args.local_transpose_diagnostic_accepted_step)
        if accepted_step_index < 0:
            raise SystemExit("[autodiff-gate] --local-transpose-diagnostic-accepted-step must be >= 0.")
        if bool(args.stage_matvec_diagnostic):
            _run_local_stage_matvec_diagnostic_report(
                args=args,
                reverse_setup=reverse_setup,
                accepted_step_index=accepted_step_index,
            )
            return
        print("[autodiff-gate] progress: running local accepted-step transpose diagnostic", flush=True)
        baseline_rollout = _radau_adaptive_schedule_rollout(
            reverse_setup.execution_context,
            reverse_setup.prepared_rollout.initial_carry,
            max_total_steps=reverse_setup.max_total_steps,
            stop_after_accepted_steps=reverse_setup.stop_after_accepted_steps,
        )
        diagnostic = _radau_debug_local_accepted_step_transpose(
            reverse_setup.execution_context,
            reverse_setup.prepared_rollout.initial_carry,
            baseline_rollout.trace,
            accepted_step_index=accepted_step_index,
            seed_mode=args.local_transpose_diagnostic_seed_mode,
            input_seed_mode=args.local_transpose_diagnostic_input_seed_mode,
            output_seed_mode=args.local_transpose_diagnostic_output_seed_mode,
        )
        diagnostic = jax.device_get(diagnostic)
        input_seed_mode = (
            args.local_transpose_diagnostic_seed_mode
            if args.local_transpose_diagnostic_input_seed_mode is None
            else args.local_transpose_diagnostic_input_seed_mode
        )
        output_seed_mode = (
            args.local_transpose_diagnostic_seed_mode
            if args.local_transpose_diagnostic_output_seed_mode is None
            else args.local_transpose_diagnostic_output_seed_mode
        )
        report = {
            "mode": "transport_reverse_ad_only_local_transpose_diagnostic",
            "config_path": str(Path(args.config)),
            "objective_name": args.objective,
            "accepted_step_limit": None if args.accepted_step_limit is None else int(args.accepted_step_limit),
            "diagnostic_accepted_step_index": accepted_step_index,
            "diagnostic_seed_mode": str(args.local_transpose_diagnostic_seed_mode),
            "diagnostic_input_seed_mode": str(input_seed_mode),
            "diagnostic_output_seed_mode": str(output_seed_mode),
            "target_attempt_index": int(diagnostic.target_attempt_index),
            "found_target": bool(diagnostic.found_target),
            "lagged_response_valid_in": bool(diagnostic.lagged_response_valid_in),
            "local_branch_reuse": bool(diagnostic.local_branch_reuse),
            "lhs_v_dot_ju": float(diagnostic.lhs_v_dot_ju),
            "rhs_jtv_dot_u": float(diagnostic.rhs_jtv_dot_u),
            "abs_err": float(diagnostic.abs_err),
            "rel_err": float(diagnostic.rel_err),
        }
        print(
            "[autodiff-gate] local transpose diagnostic: "
            f"accepted_step_index={accepted_step_index} "
            f"seed_mode={args.local_transpose_diagnostic_seed_mode} "
            f"input_seed_mode={input_seed_mode} "
            f"output_seed_mode={output_seed_mode} "
            f"target_attempt_index={report['target_attempt_index']} "
            f"found_target={report['found_target']} "
            f"lagged_response_valid_in={report['lagged_response_valid_in']} "
            f"local_branch_reuse={report['local_branch_reuse']}"
        )
        print(
            "[autodiff-gate] local transpose diagnostic values: "
            f"lhs_v_dot_ju={report['lhs_v_dot_ju']:.6e} "
            f"rhs_jtv_dot_u={report['rhs_jtv_dot_u']:.6e} "
            f"abs_err={report['abs_err']:.6e} "
            f"rel_err={report['rel_err']:.6e}"
        )
        outpath = _report_path(args.objective)
        outpath.write_text(json.dumps(report, indent=2))
        print(f"Wrote {outpath.relative_to(ROOT)}")
        return

    baseline_diag = None
    if args.baseline_diagnostics:
        print("[autodiff-gate] progress: running baseline adaptive rollout for reverse AD lane", flush=True)
        baseline_rollout = _baseline_rollout_for_diagnostics(
            baseline_values,
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            accepted_step_limit_override=args.accepted_step_limit,
        )
        baseline_diag = _adaptive_rollout_diagnostics(baseline_rollout)

    if args.objective == "all":
        if args.timing_mode == "split-vjp-warm":
            raise SystemExit(
                "[autodiff-gate] --objective all is not supported with "
                "--timing-mode split-vjp-warm yet; use jit-warm, jit-compile-only, "
                "or eager for the full metric Jacobian."
            )

        objective_vector_fn = lambda p: _reverse_objective_vector_for_parameter_vector(  # noqa: E731
            p,
            config=config,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            reverse_setup=reverse_setup,
            initial_er_root_ad=args.initial_er_root_ad,
        )

        print("[autodiff-gate] progress: running reverse custom-VJP for all objectives", flush=True)
        reverse_compile_plus_execute_s = None
        reverse_execute_s = None
        reverse_execute_times_s: list[float] = []
        objective_values = None
        t_reverse_start = time.perf_counter()
        all_objectives_direct_fn = _reverse_all_objectives_vmap_pullback_for_parameter_vector
        if args.reverse_all_objectives_mode == "multi_rhs_reduced":
            all_objectives_direct_fn = _reverse_all_objectives_multi_rhs_reduced_for_parameter_vector
        if args.timing_mode == "jit-compile-only":
            if args.reverse_all_objectives_mode in {"vmap_pullback", "multi_rhs_reduced"}:
                all_objectives_fn = jax.jit(
                    lambda p: all_objectives_direct_fn(  # noqa: E731
                        p,
                        runtime=runtime,
                        baseline_state=baseline_state,
                        profile_cfg=profile_cfg,
                        reverse_setup=reverse_setup,
                    )
                )
                compiled_all_objectives_fn = all_objectives_fn.lower(baseline_values).compile()
                del compiled_all_objectives_fn
            else:
                jac_fn = jax.jit(jax.jacrev(objective_vector_fn))
                compiled_jac_fn = jac_fn.lower(baseline_values).compile()
                del compiled_jac_fn
            gradient_matrix = None
            reverse_total_s = time.perf_counter() - t_reverse_start
        elif args.timing_mode == "jit-warm":
            if args.reverse_all_objectives_mode in {"vmap_pullback", "multi_rhs_reduced"}:
                all_objectives_fn = jax.jit(
                    lambda p: all_objectives_direct_fn(  # noqa: E731
                        p,
                        runtime=runtime,
                        baseline_state=baseline_state,
                        profile_cfg=profile_cfg,
                        reverse_setup=reverse_setup,
                    )
                )
                first_objective_values, first_gradient_matrix = all_objectives_fn(baseline_values)
                first_objective_values = jax.block_until_ready(first_objective_values)
                first_gradient_matrix = jax.block_until_ready(first_gradient_matrix)
                objective_values = first_objective_values
            else:
                jac_fn = jax.jit(jax.jacrev(objective_vector_fn))
                first_gradient_matrix = jac_fn(baseline_values)
                first_gradient_matrix = jax.block_until_ready(first_gradient_matrix)
            reverse_compile_plus_execute_s = time.perf_counter() - t_reverse_start

            gradient_matrix = first_gradient_matrix
            for _ in range(max(1, int(args.warm_repeats))):
                t_execute_start = time.perf_counter()
                if args.reverse_all_objectives_mode in {"vmap_pullback", "multi_rhs_reduced"}:
                    objective_values, gradient_matrix = all_objectives_fn(baseline_values)
                    objective_values = jax.block_until_ready(objective_values)
                    gradient_matrix = jax.block_until_ready(gradient_matrix)
                else:
                    gradient_matrix = jac_fn(baseline_values)
                    gradient_matrix = jax.block_until_ready(gradient_matrix)
                reverse_execute_times_s.append(time.perf_counter() - t_execute_start)
            reverse_execute_s = float(np.mean(reverse_execute_times_s))
            reverse_total_s = reverse_compile_plus_execute_s + float(np.sum(reverse_execute_times_s))
        else:
            if args.reverse_all_objectives_mode in {"vmap_pullback", "multi_rhs_reduced"}:
                objective_values, gradient_matrix = all_objectives_direct_fn(
                    baseline_values,
                    runtime=runtime,
                    baseline_state=baseline_state,
                    profile_cfg=profile_cfg,
                    reverse_setup=reverse_setup,
                )
                objective_values = jax.block_until_ready(objective_values)
                gradient_matrix = jax.block_until_ready(gradient_matrix)
            else:
                gradient_matrix = jax.jacrev(objective_vector_fn)(baseline_values)
                gradient_matrix = jax.block_until_ready(gradient_matrix)
            reverse_total_s = time.perf_counter() - t_reverse_start

        reverse_checkpoint_count = None
        if reverse_segment_length is not None:
            reverse_checkpoint_base = (
                int(args.accepted_step_limit)
                if args.accepted_step_limit is not None
                else int(reverse_setup.max_total_steps)
            )
            reverse_checkpoint_count = int(
                (reverse_checkpoint_base + int(reverse_segment_length) - 1)
                // int(reverse_segment_length)
            )
        reverse_lagged_branch_schedule = getattr(
            reverse_setup.execution_context.physics_context,
            "reverse_lagged_branch_schedule",
            None,
        )
        reverse_lagged_reuse_count = None
        reverse_lagged_rebuild_count = None
        if reverse_lagged_branch_schedule is not None:
            reverse_lagged_reuse_count = int(sum(bool(value) for value in reverse_lagged_branch_schedule))
            reverse_lagged_rebuild_count = int(len(reverse_lagged_branch_schedule) - reverse_lagged_reuse_count)

        objective_values_by_name = None
        if args.timing_mode != "jit-compile-only":
            if objective_values is None:
                objective_values = objective_vector_fn(baseline_values)
                objective_values = jax.block_until_ready(objective_values)
            objective_values_np = np.asarray(jax.device_get(objective_values), dtype=float)
            objective_values_by_name = {
                objective_name: float(value)
                for objective_name, value in zip(OBJECTIVE_LABELS, objective_values_np.tolist())
            }

        gradient_by_objective = None
        if gradient_matrix is not None:
            gradient_np = np.asarray(jax.device_get(gradient_matrix), dtype=float)
            gradient_by_objective = {
                objective_name: {
                    parameter_name: float(value)
                    for parameter_name, value in zip(PARAMETER_ORDER, gradient_np[objective_i].tolist())
                }
                for objective_i, objective_name in enumerate(OBJECTIVE_LABELS)
            }

        report = {
            "mode": "transport_reverse_ad_only",
            "config_path": str(Path(args.config)),
            "objective_name": "all",
            "objective_order": list(OBJECTIVE_LABELS),
            "reverse_all_objectives_mode": str(args.reverse_all_objectives_mode),
            "objective_values": objective_values_by_name,
            "parameter_order": list(PARAMETER_ORDER),
            "baseline_values": np.asarray(jax.device_get(baseline_values), dtype=float).tolist(),
            "accepted_step_limit": None if args.accepted_step_limit is None else int(args.accepted_step_limit),
            "max_total_steps": int(reverse_setup.max_total_steps),
            "reverse_checkpoint_count": reverse_checkpoint_count,
            "ntx_exact_derivative_mode": str(args.ntx_exact_derivative_mode),
            "effective_ntx_exact_derivative_mode": effective_ntx_exact_derivative_mode,
            "ntx_exact_derivative_field_pullback_mode": str(args.ntx_exact_derivative_field_pullback_mode),
            "ntx_exact_derivative_pullback_boundary": str(args.ntx_exact_derivative_pullback_boundary),
            "ntx_exact_derivative_pullback_algebra": str(args.ntx_exact_derivative_pullback_algebra),
            "reverse_ntx_prepared_solve_boundary": str(args.reverse_ntx_prepared_solve_boundary),
            "ntx_exact_radial_batch_size": neoclassical_cfg.get("ntx_exact_radial_batch_size"),
            "ntx_exact_radial_batch_mode": neoclassical_cfg.get("ntx_exact_radial_batch_mode", "simple"),
            "ntx_exact_scan_batch_size": neoclassical_cfg.get("ntx_exact_scan_batch_size"),
            "ntx_exact_preload_support": neoclassical_cfg.get("preload_support", "config"),
            "radau_jacobian_reuse_mode": None if args.radau_jacobian_reuse_mode is None else str(args.radau_jacobian_reuse_mode),
            "reverse_segment_length": reverse_segment_length,
            "reverse_lagged_reuse_count": reverse_lagged_reuse_count,
            "reverse_lagged_rebuild_count": reverse_lagged_rebuild_count,
            "reverse_direct_stage_adjoint": bool(reverse_direct_stage_adjoint),
            "reverse_stage_adjoint_solve_mode": str(args.reverse_stage_adjoint_solve_mode),
            "reverse_stage_adjoint_woodbury_rank": int(args.reverse_stage_adjoint_woodbury_rank),
            "reverse_rhs_transpose_mode": str(args.reverse_rhs_transpose_mode),
            "reverse_stage_cotangent_mode": str(args.reverse_stage_cotangent_mode),
            "reverse_step_bwd_mode": str(args.reverse_step_bwd_mode),
            "reverse_stage_adjoint_memory_mode": str(args.reverse_stage_adjoint_memory_mode),
            "reverse_stage_adjoint_iter_maxiter": int(args.reverse_stage_adjoint_iter_maxiter),
            "reverse_stage_adjoint_iter_tol": float(args.reverse_stage_adjoint_iter_tol),
            "reverse_transpose_fallback": bool(args.reverse_transpose_fallback),
            "timing_mode": str(args.timing_mode),
            "reverse_total_s": float(reverse_total_s),
            "reverse_compile_plus_execute_s": None if reverse_compile_plus_execute_s is None else float(reverse_compile_plus_execute_s),
            "reverse_execute_s": None if reverse_execute_s is None else float(reverse_execute_s),
            "reverse_execute_times_s": [float(value) for value in reverse_execute_times_s],
            "gradient_reverse_ad_by_objective": gradient_by_objective,
            "rollout_path": {
                "baseline": baseline_diag,
            },
        }

        print(
            f"[autodiff-gate] mode=transport_reverse_ad_only objective=all "
            f"reverse_all_objectives_mode={args.reverse_all_objectives_mode} "
            f"objectives={list(OBJECTIVE_LABELS)} "
            f"parameters={list(PARAMETER_ORDER)} "
            f"radau_jacobian_reuse_mode={args.radau_jacobian_reuse_mode} "
            f"effective_ntx_exact_derivative_mode={effective_ntx_exact_derivative_mode} "
            f"ntx_exact_derivative_field_pullback_mode={args.ntx_exact_derivative_field_pullback_mode} "
            f"ntx_exact_derivative_pullback_boundary={args.ntx_exact_derivative_pullback_boundary} "
            f"ntx_exact_derivative_pullback_algebra={args.ntx_exact_derivative_pullback_algebra} "
            f"reverse_ntx_prepared_solve_boundary={args.reverse_ntx_prepared_solve_boundary} "
            f"ntx_exact_preload_support={neoclassical_cfg.get('preload_support', 'config')} "
            f"reverse_total_s={reverse_total_s:.6e}"
        )
        if reverse_compile_plus_execute_s is not None:
            print(
                f"[autodiff-gate] timing reverse_compile_plus_execute_s={reverse_compile_plus_execute_s:.6e} "
                f"reverse_execute_s_mean={reverse_execute_s:.6e} "
                f"reverse_execute_s_min={min(reverse_execute_times_s):.6e} "
                f"reverse_execute_repeats={len(reverse_execute_times_s)}"
            )
            print(
                "[autodiff-gate] timing reverse_execute_times_s="
                + ",".join(f"{float(value):.6e}" for value in reverse_execute_times_s)
            )
        if gradient_by_objective is not None:
            if objective_values_by_name is not None:
                print("[autodiff-gate] objective values:")
                for objective_name in OBJECTIVE_LABELS:
                    print(f"  - {objective_name}: value={objective_values_by_name[objective_name]:.6e}")
            print("[autodiff-gate] reverse gradients by objective:")
            for objective_name in OBJECTIVE_LABELS:
                print(f"  - {objective_name}:")
                for parameter_name in PARAMETER_ORDER:
                    print(
                        f"      d{objective_name}/d{parameter_name}: "
                        f"rev={gradient_by_objective[objective_name][parameter_name]:.16e}"
                    )
        outpath = _report_path("all")
        outpath.write_text(json.dumps(report, indent=2))
        print(f"Wrote {outpath.relative_to(ROOT)}")
        return

    objective_fn = lambda p: _reverse_objective_for_parameter_vector(  # noqa: E731
        p,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        objective_index=objective_index,
        reverse_setup=reverse_setup,
        initial_er_root_ad=args.initial_er_root_ad,
    )

    print("[autodiff-gate] progress: running reverse custom-VJP", flush=True)
    reverse_compile_plus_execute_s = None
    reverse_execute_s = None
    reverse_execute_times_s: list[float] = []
    t_reverse_start = time.perf_counter()
    if args.timing_mode == "jit-compile-only":
        grad_fn = jax.jit(jax.grad(objective_fn))
        compiled_grad_fn = grad_fn.lower(baseline_values).compile()
        del compiled_grad_fn
        reverse_total_s = time.perf_counter() - t_reverse_start
        reverse_checkpoint_count = None
        if reverse_segment_length is not None:
            reverse_checkpoint_base = (
                int(args.accepted_step_limit)
                if args.accepted_step_limit is not None
                else int(reverse_setup.max_total_steps)
            )
            reverse_checkpoint_count = int(
                (reverse_checkpoint_base + int(reverse_segment_length) - 1)
                // int(reverse_segment_length)
            )
        reverse_lagged_branch_schedule = getattr(
            reverse_setup.execution_context.physics_context,
            "reverse_lagged_branch_schedule",
            None,
        )
        reverse_lagged_reuse_count = None
        reverse_lagged_rebuild_count = None
        if reverse_lagged_branch_schedule is not None:
            reverse_lagged_reuse_count = int(sum(bool(value) for value in reverse_lagged_branch_schedule))
            reverse_lagged_rebuild_count = int(len(reverse_lagged_branch_schedule) - reverse_lagged_reuse_count)
        report = {
            "mode": "transport_reverse_ad_only",
            "config_path": str(Path(args.config)),
            "objective_name": args.objective,
            "parameter_order": list(PARAMETER_ORDER),
            "baseline_values": np.asarray(jax.device_get(baseline_values), dtype=float).tolist(),
            "accepted_step_limit": None if args.accepted_step_limit is None else int(args.accepted_step_limit),
            "max_total_steps": int(reverse_setup.max_total_steps),
            "reverse_checkpoint_count": reverse_checkpoint_count,
            "ntx_exact_derivative_mode": str(args.ntx_exact_derivative_mode),
            "effective_ntx_exact_derivative_mode": effective_ntx_exact_derivative_mode,
            "ntx_exact_derivative_field_pullback_mode": str(args.ntx_exact_derivative_field_pullback_mode),
            "ntx_exact_derivative_pullback_boundary": str(args.ntx_exact_derivative_pullback_boundary),
            "ntx_exact_derivative_pullback_algebra": str(args.ntx_exact_derivative_pullback_algebra),
            "reverse_ntx_prepared_solve_boundary": str(args.reverse_ntx_prepared_solve_boundary),
            "ntx_exact_radial_batch_size": neoclassical_cfg.get("ntx_exact_radial_batch_size"),
            "ntx_exact_radial_batch_mode": neoclassical_cfg.get("ntx_exact_radial_batch_mode", "simple"),
            "ntx_exact_scan_batch_size": neoclassical_cfg.get("ntx_exact_scan_batch_size"),
            "ntx_exact_preload_support": neoclassical_cfg.get("preload_support", "config"),
            "radau_jacobian_reuse_mode": None if args.radau_jacobian_reuse_mode is None else str(args.radau_jacobian_reuse_mode),
            "reverse_segment_length": reverse_segment_length,
            "reverse_lagged_reuse_count": reverse_lagged_reuse_count,
            "reverse_lagged_rebuild_count": reverse_lagged_rebuild_count,
            "reverse_direct_stage_adjoint": bool(reverse_direct_stage_adjoint),
            "reverse_stage_adjoint_solve_mode": str(args.reverse_stage_adjoint_solve_mode),
            "reverse_stage_adjoint_woodbury_rank": int(args.reverse_stage_adjoint_woodbury_rank),
            "reverse_rhs_transpose_mode": str(args.reverse_rhs_transpose_mode),
            "reverse_stage_cotangent_mode": str(args.reverse_stage_cotangent_mode),
            "reverse_step_bwd_mode": str(args.reverse_step_bwd_mode),
            "reverse_stage_adjoint_memory_mode": str(args.reverse_stage_adjoint_memory_mode),
            "reverse_stage_adjoint_iter_maxiter": int(args.reverse_stage_adjoint_iter_maxiter),
            "reverse_stage_adjoint_iter_tol": float(args.reverse_stage_adjoint_iter_tol),
            "reverse_transpose_fallback": bool(args.reverse_transpose_fallback),
            "timing_mode": str(args.timing_mode),
            "reverse_total_s": float(reverse_total_s),
            "gradient_reverse_ad": None,
            "rollout_path": {
                "baseline": baseline_diag,
            },
        }
        print(
            f"[autodiff-gate] mode=transport_reverse_ad_only objective={args.objective} "
            f"parameters={list(PARAMETER_ORDER)} "
            f"radau_jacobian_reuse_mode={args.radau_jacobian_reuse_mode} "
            f"effective_ntx_exact_derivative_mode={effective_ntx_exact_derivative_mode} "
            f"ntx_exact_derivative_field_pullback_mode={args.ntx_exact_derivative_field_pullback_mode} "
            f"ntx_exact_derivative_pullback_boundary={args.ntx_exact_derivative_pullback_boundary} "
            f"ntx_exact_derivative_pullback_algebra={args.ntx_exact_derivative_pullback_algebra} "
            f"reverse_ntx_prepared_solve_boundary={args.reverse_ntx_prepared_solve_boundary} "
            f"ntx_exact_radial_batch_size={neoclassical_cfg.get('ntx_exact_radial_batch_size')} "
            f"ntx_exact_radial_batch_mode={neoclassical_cfg.get('ntx_exact_radial_batch_mode', 'simple')} "
            f"ntx_exact_scan_batch_size={neoclassical_cfg.get('ntx_exact_scan_batch_size')} "
            f"ntx_exact_preload_support={neoclassical_cfg.get('preload_support', 'config')} "
            f"max_total_steps={reverse_setup.max_total_steps} "
            f"reverse_checkpoint_count={reverse_checkpoint_count} "
            f"reverse_segment_length={reverse_segment_length} "
            f"reverse_lagged_reuse_count={reverse_lagged_reuse_count} "
            f"reverse_lagged_rebuild_count={reverse_lagged_rebuild_count} "
            f"reverse_direct_stage_adjoint={bool(reverse_direct_stage_adjoint)} "
            f"reverse_stage_adjoint_solve_mode={args.reverse_stage_adjoint_solve_mode} "
            f"reverse_rhs_transpose_mode={args.reverse_rhs_transpose_mode} "
            f"reverse_stage_cotangent_mode={args.reverse_stage_cotangent_mode} "
            f"reverse_step_bwd_mode={args.reverse_step_bwd_mode} "
            f"reverse_stage_adjoint_memory_mode={args.reverse_stage_adjoint_memory_mode} "
            f"timing_mode={args.timing_mode} "
            f"reverse_compile_s={reverse_total_s:.6e}"
        )
        outpath = _report_path(args.objective)
        outpath.write_text(json.dumps(report, indent=2))
        print(f"Wrote {outpath.relative_to(ROOT)}")
        return
    if args.timing_mode == "jit-warm":
        grad_fn = jax.jit(jax.grad(objective_fn))
        first_gradient = grad_fn(baseline_values)
        first_gradient = jax.block_until_ready(first_gradient)
        reverse_compile_plus_execute_s = time.perf_counter() - t_reverse_start

        gradient_rev = first_gradient
        for _ in range(max(1, int(args.warm_repeats))):
            t_execute_start = time.perf_counter()
            gradient_rev = grad_fn(baseline_values)
            gradient_rev = jax.block_until_ready(gradient_rev)
            reverse_execute_times_s.append(time.perf_counter() - t_execute_start)
        reverse_execute_s = float(np.mean(reverse_execute_times_s))
        reverse_total_s = reverse_compile_plus_execute_s + float(np.sum(reverse_execute_times_s))
    elif args.timing_mode == "split-vjp-warm":
        split_grad_fn = _make_reverse_gradient_split_custom_vjp_fn(
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            objective_index=objective_index,
            reverse_setup=reverse_setup,
            jit_kernels=True,
        )
        first_gradient = split_grad_fn(baseline_values)
        first_gradient = jax.block_until_ready(first_gradient)
        reverse_compile_plus_execute_s = time.perf_counter() - t_reverse_start

        gradient_rev = first_gradient
        for _ in range(max(1, int(args.warm_repeats))):
            t_execute_start = time.perf_counter()
            gradient_rev = split_grad_fn(baseline_values)
            gradient_rev = jax.block_until_ready(gradient_rev)
            reverse_execute_times_s.append(time.perf_counter() - t_execute_start)
        reverse_execute_s = float(np.mean(reverse_execute_times_s))
        reverse_total_s = reverse_compile_plus_execute_s + float(np.sum(reverse_execute_times_s))
    else:
        gradient_rev = jax.grad(objective_fn)(baseline_values)
        gradient_rev = jax.block_until_ready(gradient_rev)
        reverse_total_s = time.perf_counter() - t_reverse_start
    grad_np = np.asarray(jax.device_get(gradient_rev), dtype=float)
    final_objective_cotangent_diagnostic = None
    if bool(args.diagnose_final_objective_cotangent):
        if args.objective == "all":
            raise SystemExit("[autodiff-gate] --diagnose-final-objective-cotangent requires a scalar objective.")
        final_cotangent_fn = jax.jit(
            lambda p: _reverse_final_y_objective_cotangent_for_parameter_vector(  # noqa: E731
                p,
                runtime=runtime,
                baseline_state=baseline_state,
                profile_cfg=profile_cfg,
                objective_index=objective_index,
                reverse_setup=reverse_setup,
            )
        )
        objective_value_diag, final_y_bar_diag = final_cotangent_fn(baseline_values)
        objective_value_diag, final_y_bar_diag = jax.block_until_ready((objective_value_diag, final_y_bar_diag))
        final_y_bar_np = np.asarray(jax.device_get(final_y_bar_diag), dtype=float)
        final_objective_cotangent_diagnostic = {
            "objective_value": float(jax.device_get(objective_value_diag)),
            "final_y_bar_l2": float(np.linalg.norm(final_y_bar_np)),
            "final_y_bar_linf": float(np.max(np.abs(final_y_bar_np))) if final_y_bar_np.size else 0.0,
            "final_y_bar_nonzero_count": int(np.count_nonzero(final_y_bar_np)),
            "final_y_bar_size": int(final_y_bar_np.size),
        }
    reverse_checkpoint_count = None
    if reverse_segment_length is not None:
        reverse_checkpoint_base = (
            int(args.accepted_step_limit)
            if args.accepted_step_limit is not None
            else int(reverse_setup.max_total_steps)
        )
        reverse_checkpoint_count = int(
            (reverse_checkpoint_base + int(reverse_segment_length) - 1)
            // int(reverse_segment_length)
        )
    reverse_lagged_branch_schedule = getattr(
        reverse_setup.execution_context.physics_context,
        "reverse_lagged_branch_schedule",
        None,
    )
    reverse_lagged_reuse_count = None
    reverse_lagged_rebuild_count = None
    if reverse_lagged_branch_schedule is not None:
        reverse_lagged_reuse_count = int(sum(bool(value) for value in reverse_lagged_branch_schedule))
        reverse_lagged_rebuild_count = int(len(reverse_lagged_branch_schedule) - reverse_lagged_reuse_count)

    report = {
        "mode": "transport_reverse_ad_only",
        "config_path": str(Path(args.config)),
        "objective_name": args.objective,
        "parameter_order": list(PARAMETER_ORDER),
        "baseline_values": np.asarray(jax.device_get(baseline_values), dtype=float).tolist(),
        "accepted_step_limit": None if args.accepted_step_limit is None else int(args.accepted_step_limit),
        "max_total_steps": int(reverse_setup.max_total_steps),
        "reverse_checkpoint_count": reverse_checkpoint_count,
        "ntx_exact_derivative_mode": str(args.ntx_exact_derivative_mode),
        "effective_ntx_exact_derivative_mode": effective_ntx_exact_derivative_mode,
        "ntx_exact_derivative_field_pullback_mode": str(args.ntx_exact_derivative_field_pullback_mode),
        "ntx_exact_derivative_pullback_boundary": str(args.ntx_exact_derivative_pullback_boundary),
        "ntx_exact_derivative_pullback_algebra": str(args.ntx_exact_derivative_pullback_algebra),
        "reverse_ntx_prepared_solve_boundary": str(args.reverse_ntx_prepared_solve_boundary),
        "ntx_exact_radial_batch_size": neoclassical_cfg.get("ntx_exact_radial_batch_size"),
        "ntx_exact_radial_batch_mode": neoclassical_cfg.get("ntx_exact_radial_batch_mode", "simple"),
        "ntx_exact_scan_batch_size": neoclassical_cfg.get("ntx_exact_scan_batch_size"),
        "ntx_exact_preload_support": neoclassical_cfg.get("preload_support", "config"),
        "radau_jacobian_reuse_mode": None if args.radau_jacobian_reuse_mode is None else str(args.radau_jacobian_reuse_mode),
        "reverse_segment_length": reverse_segment_length,
        "reverse_lagged_reuse_count": reverse_lagged_reuse_count,
        "reverse_lagged_rebuild_count": reverse_lagged_rebuild_count,
        "reverse_direct_stage_adjoint": bool(reverse_direct_stage_adjoint),
        "reverse_stage_adjoint_solve_mode": str(args.reverse_stage_adjoint_solve_mode),
        "reverse_stage_adjoint_woodbury_rank": int(args.reverse_stage_adjoint_woodbury_rank),
        "reverse_rhs_transpose_mode": str(args.reverse_rhs_transpose_mode),
        "reverse_stage_cotangent_mode": str(args.reverse_stage_cotangent_mode),
        "reverse_step_bwd_mode": str(args.reverse_step_bwd_mode),
        "reverse_stage_adjoint_memory_mode": str(args.reverse_stage_adjoint_memory_mode),
        "reverse_stage_adjoint_iter_maxiter": int(args.reverse_stage_adjoint_iter_maxiter),
        "reverse_stage_adjoint_iter_tol": float(args.reverse_stage_adjoint_iter_tol),
        "reverse_transpose_fallback": bool(args.reverse_transpose_fallback),
        "timing_mode": str(args.timing_mode),
        "reverse_total_s": float(reverse_total_s),
        "reverse_compile_plus_execute_s": None if reverse_compile_plus_execute_s is None else float(reverse_compile_plus_execute_s),
        "reverse_execute_s": None if reverse_execute_s is None else float(reverse_execute_s),
        "reverse_execute_times_s": [float(value) for value in reverse_execute_times_s],
        "gradient_reverse_ad": grad_np.tolist(),
        "final_objective_cotangent_diagnostic": final_objective_cotangent_diagnostic,
        "rollout_path": {
            "baseline": baseline_diag,
        },
    }

    print(
        f"[autodiff-gate] mode=transport_reverse_ad_only objective={args.objective} "
        f"parameters={list(PARAMETER_ORDER)} "
        f"radau_jacobian_reuse_mode={args.radau_jacobian_reuse_mode} "
        f"effective_ntx_exact_derivative_mode={effective_ntx_exact_derivative_mode} "
        f"ntx_exact_derivative_field_pullback_mode={args.ntx_exact_derivative_field_pullback_mode} "
        f"ntx_exact_derivative_pullback_boundary={args.ntx_exact_derivative_pullback_boundary} "
        f"ntx_exact_derivative_pullback_algebra={args.ntx_exact_derivative_pullback_algebra} "
        f"reverse_ntx_prepared_solve_boundary={args.reverse_ntx_prepared_solve_boundary} "
        f"ntx_exact_radial_batch_size={neoclassical_cfg.get('ntx_exact_radial_batch_size')} "
        f"ntx_exact_radial_batch_mode={neoclassical_cfg.get('ntx_exact_radial_batch_mode', 'simple')} "
        f"ntx_exact_scan_batch_size={neoclassical_cfg.get('ntx_exact_scan_batch_size')} "
        f"ntx_exact_preload_support={neoclassical_cfg.get('preload_support', 'config')} "
        f"max_total_steps={reverse_setup.max_total_steps} "
        f"reverse_checkpoint_count={reverse_checkpoint_count} "
        f"reverse_segment_length={reverse_segment_length} "
        f"reverse_lagged_reuse_count={reverse_lagged_reuse_count} "
        f"reverse_lagged_rebuild_count={reverse_lagged_rebuild_count} "
        f"reverse_direct_stage_adjoint={bool(reverse_direct_stage_adjoint)} "
        f"reverse_stage_adjoint_solve_mode={args.reverse_stage_adjoint_solve_mode} "
        f"reverse_stage_adjoint_woodbury_rank={args.reverse_stage_adjoint_woodbury_rank} "
        f"reverse_rhs_transpose_mode={args.reverse_rhs_transpose_mode} "
        f"reverse_stage_cotangent_mode={args.reverse_stage_cotangent_mode} "
        f"reverse_step_bwd_mode={args.reverse_step_bwd_mode} "
        f"reverse_stage_adjoint_memory_mode={args.reverse_stage_adjoint_memory_mode} "
        f"reverse_stage_adjoint_iter_maxiter={args.reverse_stage_adjoint_iter_maxiter} "
        f"reverse_stage_adjoint_iter_tol={args.reverse_stage_adjoint_iter_tol:.6e} "
        f"timing_mode={args.timing_mode} "
        f"reverse_total_s={reverse_total_s:.6e}"
    )
    if reverse_compile_plus_execute_s is not None:
        print(
            f"[autodiff-gate] timing reverse_compile_plus_execute_s={reverse_compile_plus_execute_s:.6e} "
            f"reverse_execute_s_mean={reverse_execute_s:.6e} "
            f"reverse_execute_s_min={min(reverse_execute_times_s):.6e} "
            f"reverse_execute_repeats={len(reverse_execute_times_s)}"
        )
        print(
            "[autodiff-gate] timing reverse_execute_times_s="
            + ",".join(f"{float(value):.6e}" for value in reverse_execute_times_s)
        )
    if baseline_diag is not None:
        print(
            f"[autodiff-gate] rollout baseline: attempt_count={baseline_diag.get('attempt_count')} "
            f"accepted_count={baseline_diag.get('accepted_count')} "
            f"completed={baseline_diag.get('completed')} failed={baseline_diag.get('failed')} "
            f"fail_code={baseline_diag.get('fail_code')}"
        )
    if final_objective_cotangent_diagnostic is not None:
        print(
            "[autodiff-gate] final objective cotangent: "
            f"value={final_objective_cotangent_diagnostic['objective_value']:.16e} "
            f"l2={final_objective_cotangent_diagnostic['final_y_bar_l2']:.16e} "
            f"linf={final_objective_cotangent_diagnostic['final_y_bar_linf']:.16e} "
            f"nonzero_count={final_objective_cotangent_diagnostic['final_y_bar_nonzero_count']}/"
            f"{final_objective_cotangent_diagnostic['final_y_bar_size']}"
        )
    print("[autodiff-gate] reverse gradients:")
    for name, value in zip(PARAMETER_ORDER, grad_np.tolist()):
        print(f"  - d{args.objective}/d{name}: rev={float(value):.16e}")
    outpath = _report_path(args.objective)
    outpath.write_text(json.dumps(report, indent=2))
    print(f"Wrote {outpath.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
