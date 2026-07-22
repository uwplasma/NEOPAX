from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_transport_forward_fd_lane import (  # noqa: E402
    DEFAULT_CONFIG,
    OBJECTIVE_LABELS,
    _adaptive_rollout_diagnostics,
    _alpha_power_volume_average,
    _baseline_profile_cfg,
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
    build_geometry_autodiff_context,
    build_neopax_geometry_and_ntx_exact_lij_support_from_param_vector,
    build_ntx_exact_lij_support_from_param_vector,
    geometry_payload_pullback_from_param_vector_raw_block_transpose,
)
from NEOPAX._orchestrator import build_runtime_context  # noqa: E402
from NEOPAX._orchestrator import prepare_transport_solver_components  # noqa: E402
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
    _radau_eval_rhs,
    _radau_add_support_delta_trees,
    _radau_segment_reduced_cotangent_bwd_batched_call,
    _radau_segment_reduced_cotangent_bwd_call,
    _radau_single_slot_support_cotangent_bwd_call,
    _radau_single_slot_support_cotangent_bwd_flat_batched_call,
    _radau_zero_support_delta_tree_like,
)


PARAMETER_ORDER = ("n0", "T0", "density_shape_power", "temperature_shape_power")
_REALTIME_GEOMETRY_BACKENDS = {"vmec_jax_booz_xform_jax", "vmec_runtime", "vmec_realtime"}


def _objective_scalar_by_index(final_state, runtime, objective_index: int):
    """Evaluate one objective without constructing unrelated possibly-nonfinite objectives."""

    objective_name = OBJECTIVE_LABELS[int(objective_index)]
    er = jnp.asarray(final_state.Er)
    if objective_name == "softmax_Er":
        return _softmax_objective(er)
    if objective_name == "smooth_root_proxy":
        rho = jnp.asarray(runtime.geometry.rho_grid, dtype=er.dtype)
        return _smooth_root_proxy(er, rho)
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
    flux_model, changed = _replace_ntx_support_payload_in_model(runtime.models.flux, support)
    if not changed:
        raise ValueError("Could not find an NTX exact-runtime model that accepts an explicit support payload.")
    return dataclasses.replace(
        runtime,
        models=dataclasses.replace(runtime.models, flux=flux_model),
    )


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
    support = _find_ntx_support_payload_in_model(runtime.models.flux)
    if support is None:
        raise ValueError("No preloaded NTX exact-runtime support payload was found in the realtime runtime.")
    return support


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
            "[autodiff-gate] --ntx-exact-derivative-field-pullback-mode compact_vjp "
            "requires the matching NTX patch/export: "
            "solve_prepared_coefficient_vector_derivative_vjp. Sync/apply the NTX "
            "changes before running this mode."
        ) from exc


def _initial_state_for_parameter_vector(
    parameter_values,
    *,
    baseline_state,
    profile_cfg: dict,
    runtime,
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
    return dataclasses.replace(
        baseline_state,
        density=density_state,
        pressure=pressure_state,
    )


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


def _geometry_context_from_config(config: dict[str, Any], geometry_parameter: str):
    family, m, n = _parse_reverse_geometry_parameter(geometry_parameter)
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
    return build_geometry_autodiff_context(
        vmec_input_file,
        param_family=family,
        param_m=m,
        param_n=n,
        mboz=int(geom_cfg.get("mboz", geom_cfg.get("vmec_mboz", 12))),
        nboz=int(geom_cfg.get("nboz", geom_cfg.get("vmec_nboz", 12))),
    )


def _reverse_geometry_parameter_order(geometry_parameter: str) -> tuple[str, ...]:
    return (*PARAMETER_ORDER, _format_reverse_geometry_parameter(geometry_parameter))


def _geometry_param_specs_from_parameter_name(geometry_parameter: str) -> tuple[tuple[str, int, int], ...]:
    return (_parse_reverse_geometry_parameter(geometry_parameter),)


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
):
    """Build the initial carry with a reverse-local model-aware lagged pullback."""

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
    final_state = reverse_setup.prepared_rollout.physics_context.unpack_flat(final_y)
    return _objective_scalar_by_index(final_state, runtime, objective_index)


def _reverse_objective_vector_for_parameter_vector(
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
    if int(reverse_setup.reverse_segment_length) != 1:
        raise ValueError("support payload reverse probe currently requires --reverse-segment-length 1.")
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
        raise ValueError("support payload reverse probe requires segmented reverse residuals.")

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
    next_reduced_bars_by_segment = [None] * segment_count
    for segment_index in range(segment_count - 1, -1, -1):
        segment_start_carry = _take_tree_axis0(segment_start_carries, segment_index)
        segment_arrays = _take_tree_axis0(segmented_replay_arrays, segment_index)
        next_reduced_bars_by_segment[segment_index] = reduced_bar
        reduced_bar = _radau_segment_reduced_cotangent_bwd_call(
            reverse_setup.execution_context,
            cotangent_mode,
            reduced_bar,
            segment_start_carry,
            segment_arrays,
        )

    support_reuse_count = 0
    support_rebuild_count = 0
    for segment_index in range(segment_count - 1, -1, -1):
        segment_start_carry = _take_tree_axis0(segment_start_carries, segment_index)
        segment_arrays = _take_tree_axis0(segmented_replay_arrays, segment_index)
        slot_arrays = _take_tree_axis0(segment_arrays, 0)
        slot_lagged_response_valid = bool(
            np.asarray(jax.device_get(segment_start_carry.lagged_response_valid))
        )
        slot_cotangent_mode = "force_reuse_bwd" if slot_lagged_response_valid else "force_rebuild_bwd"
        support_reuse_count += int(slot_lagged_response_valid)
        support_rebuild_count += int(not slot_lagged_response_valid)
        segment_support_bar = _radau_single_slot_support_cotangent_bwd_call(
            reverse_setup.execution_context,
            slot_cotangent_mode,
            next_reduced_bars_by_segment[segment_index],
            segment_start_carry,
            slot_arrays,
            support_payload,
        )
        support_bar = _radau_add_support_delta_trees(support_bar, segment_support_bar, support_payload)

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
    runtime,
    baseline_state,
    profile_cfg: dict,
    reverse_setup: _ReverseStaticSetup,
    support_payload,
):
    """Return all objective values, profile gradients, and realtime support cotangents.

    This is the realtime-geometry extension of the regular multi-RHS reduced
    profile reverse path: profile cotangents and optional support cotangents are
    propagated through the same realized-schedule reverse pass.
    """

    if reverse_setup.reverse_segment_length is None or int(reverse_setup.reverse_segment_length) <= 0:
        raise ValueError("support payload reverse probe requires --reverse-segment-length.")
    if int(reverse_setup.reverse_segment_length) != 1:
        raise ValueError("support payload reverse probe currently requires --reverse-segment-length 1.")
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
        zero_tree = _radau_align_tangent_tree_to_primal(None, primal_tree)
        return jax.tree_util.tree_map(
            lambda leaf: jnp.broadcast_to(
                jnp.asarray(leaf)[None, ...],
                (batch_size,) + jnp.asarray(leaf).shape,
            ),
            zero_tree,
        )

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

    objective_count = int(len(OBJECTIVE_LABELS))
    objective_values_rows = []
    final_y_bar_rows = []
    objective_payload_bar_rows = []
    combined_geometry_payload = isinstance(support_payload, dict) and "geometry" in support_payload
    zero_payload_bar = _radau_zero_support_delta_tree_like(support_payload)
    for objective_i in range(objective_count):
        def _objective_from_final_y(final_y_value, objective_index=objective_i):
            final_state = reverse_setup.prepared_rollout.physics_context.unpack_flat(final_y_value)
            return _objective_scalar_by_index(final_state, runtime, objective_index)

        objective_value, objective_pullback = jax.vjp(_objective_from_final_y, final_y)
        objective_values_rows.append(objective_value)
        final_y_bar_rows.append(objective_pullback(jnp.ones_like(objective_value))[0])
        if combined_geometry_payload:
            final_state_for_geometry = reverse_setup.prepared_rollout.physics_context.unpack_flat(final_y)
            geometry = support_payload["geometry"]
            geometry_delta0 = _float_delta_tree_like(geometry)

            def _objective_from_geometry_delta(geometry_delta, objective_index=objective_i):
                runtime_with_geometry = dataclasses.replace(
                    runtime,
                    geometry=_add_float_delta_tree(geometry, geometry_delta),
                )
                return _objective_scalar_by_index(
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
    cotangent_mode = str(
        getattr(reverse_setup.execution_context.physics_context, "reverse_stage_cotangent_mode", "full")
    ).strip().lower()
    segment_count = int(jax.tree_util.tree_leaves(segmented_replay_arrays)[0].shape[0])
    next_reduced_bars_by_segment = [None] * segment_count
    for segment_index in range(segment_count - 1, -1, -1):
        segment_start_carry = _take_tree_axis0(segment_start_carries, segment_index)
        segment_arrays = _take_tree_axis0(segmented_replay_arrays, segment_index)
        next_reduced_bars_by_segment[segment_index] = reduced_bars
        reduced_bars = _radau_segment_reduced_cotangent_bwd_batched_call(
            reverse_setup.execution_context,
            cotangent_mode,
            reduced_bars,
            segment_start_carry,
            segment_arrays,
        )

    support_reuse_count = 0
    support_rebuild_count = 0
    for segment_index in range(segment_count - 1, -1, -1):
        segment_start_carry = _take_tree_axis0(segment_start_carries, segment_index)
        segment_arrays = _take_tree_axis0(segmented_replay_arrays, segment_index)
        slot_arrays = _take_tree_axis0(segment_arrays, 0)
        slot_lagged_response_valid = bool(
            np.asarray(jax.device_get(segment_start_carry.lagged_response_valid))
        )
        slot_cotangent_mode = "force_reuse_bwd" if slot_lagged_response_valid else "force_rebuild_bwd"
        support_reuse_count += int(slot_lagged_response_valid)
        support_rebuild_count += int(not slot_lagged_response_valid)
        step_support_bar_leaves = _radau_single_slot_support_cotangent_bwd_flat_batched_call(
            reverse_setup.execution_context,
            slot_cotangent_mode,
            next_reduced_bars_by_segment[segment_index],
            segment_start_carry,
            slot_arrays,
            support_payload,
        )
        support_bar_leaves = tuple(
            accumulated + increment
            for accumulated, increment in zip(support_bar_leaves, step_support_bar_leaves)
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
    gradient_matrix = jax.vmap(lambda carry0_bar: initial_carry_pullback(carry0_bar)[0])(carry0_bars)
    support_bars = tuple(
        support_treedef.unflatten(
            [jnp.asarray(leaf)[objective_i] for leaf in support_bar_leaves]
        )
        for objective_i in range(objective_count)
    )
    return (
        objective_values,
        gradient_matrix,
        support_bars,
        support_reuse_count,
        support_rebuild_count,
        initial_cache_pullback_used,
        initial_cache_pullback_skipped,
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
    accepted_step_limit_override: int | None = None,
    reverse_segment_length: int | None = None,
    reverse_direct_stage_adjoint: bool = False,
    reverse_stage_adjoint_solve_mode: str = "structured",
    reverse_rhs_transpose_mode: str = "generic",
    reverse_stage_cotangent_mode: str = "full",
    reverse_step_bwd_mode: str = "current",
    reverse_stage_adjoint_memory_mode: str = "default",
    reverse_stage_adjoint_iter_maxiter: int = 40,
    reverse_stage_adjoint_iter_tol: float = 1.0e-10,
) -> _ReverseStaticSetup:
    state0_static = _initial_state_for_parameter_vector(
        parameter_values,
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
    return _ReverseStaticSetup(
        solver=solver,
        solve_vector_field=solve_vector_field_static,
        prepared_rollout=prepared_rollout_static,
        execution_context=execution_context,
        stop_after_accepted_steps=stop_after_accepted_steps,
        max_total_steps=max_total_steps,
        reverse_segment_length=reverse_segment_length,
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
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        runtime=baseline_runtime,
    )
    swapped_profile_state = _initial_state_for_parameter_vector(
        baseline_values[: len(PARAMETER_ORDER)],
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
):
    combined_geometry_payload = str(args.realtime_geometry_gradient_path) == "reverse_payload"
    ntx_support_payload = _find_ntx_support_payload(baseline_runtime)
    support_payload = (
        {"geometry": baseline_runtime.geometry, "ntx_support": ntx_support_payload}
        if combined_geometry_payload
        else ntx_support_payload
    )
    profile_values = baseline_values[: len(PARAMETER_ORDER)]
    support_probe_cotangent_mode = str(args.reverse_stage_cotangent_mode)
    reverse_setup = _prepare_reverse_static_setup(
        profile_values,
        config=config,
        runtime=baseline_runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
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
    if args.objective == "all":
        early_geometry_diagnostics = _geometry_volume_diagnostics(baseline_runtime.geometry)
        print("[autodiff-gate] realtime geometry pre-reverse diagnostics:")
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
        (
            objective_values,
            profile_gradient_matrix,
            support_bars,
            support_reuse_count,
            support_rebuild_count,
            initial_cache_pullback_used,
            initial_cache_pullback_skipped,
        ) = _reverse_all_objectives_support_payload_bar_for_parameter_vector(
            profile_values,
            runtime=baseline_runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            reverse_setup=reverse_setup,
            support_payload=support_payload,
        )
        objective_values, profile_gradient_matrix, support_bars = jax.block_until_ready(
            (objective_values, profile_gradient_matrix, support_bars)
        )
        print(
            "[autodiff-gate] progress: transport reverse profile/support cotangents complete "
            f"elapsed_s={time.perf_counter() - t_phase:.3f}",
            flush=True,
        )
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
        geometry_param_specs = _geometry_param_specs_from_parameter_name(geometry_parameter_name)
        baseline_geometry_deltas = jnp.asarray(
            [float(geom_cfg.get("vmec_param_delta", 0.0))],
            dtype=jnp.float64,
        )

        def _support_from_geometry_deltas(geometry_deltas):
            if combined_geometry_payload:
                return build_neopax_geometry_and_ntx_exact_lij_support_from_param_vector(
                    geometry_context,
                    geometry_deltas,
                    geometry_param_specs,
                    lane="ad",
                    n_r=int(geom_cfg.get("n_radial", 51)),
                    n_theta=int(neoclassical_cfg.get("ntx_exact_n_theta", 25)),
                    n_zeta=int(neoclassical_cfg.get("ntx_exact_n_zeta", 25)),
                    n_xi=int(neoclassical_cfg.get("ntx_exact_n_xi", 64)),
                    surface_backend=str(neoclassical_cfg.get("ntx_exact_surface_backend", "booz")),
                    max_iter=geom_cfg.get("vmec_max_iter"),
                    step_size=geom_cfg.get("vmec_step_size"),
                    jacobian_penalty=float(geom_cfg.get("vmec_jacobian_penalty", 1.0e3)),
                )
            return build_ntx_exact_lij_support_from_param_vector(
                geometry_context,
                geometry_deltas,
                geometry_param_specs,
                lane="ad",
                n_r=int(geom_cfg.get("n_radial", 51)),
                n_theta=int(neoclassical_cfg.get("ntx_exact_n_theta", 25)),
                n_zeta=int(neoclassical_cfg.get("ntx_exact_n_zeta", 25)),
                n_xi=int(neoclassical_cfg.get("ntx_exact_n_xi", 64)),
                surface_backend=str(neoclassical_cfg.get("ntx_exact_surface_backend", "booz")),
                max_iter=geom_cfg.get("vmec_max_iter"),
                step_size=geom_cfg.get("vmec_step_size"),
                jacobian_penalty=float(geom_cfg.get("vmec_jacobian_penalty", 1.0e3)),
            )

        t_phase = time.perf_counter()
        print(
            "[autodiff-gate] progress: building geometry support pullback "
            f"for {geometry_parameter_name}",
            flush=True,
        )
        geometry_gradient_matrix = geometry_payload_pullback_from_param_vector_raw_block_transpose(
            geometry_context,
            baseline_geometry_deltas,
            geometry_param_specs,
            tuple(support_bars),
            combined_payload=combined_geometry_payload,
            n_r=int(geom_cfg.get("n_radial", 51)),
            n_theta=int(neoclassical_cfg.get("ntx_exact_n_theta", 25)),
            n_zeta=int(neoclassical_cfg.get("ntx_exact_n_zeta", 25)),
            n_xi=int(neoclassical_cfg.get("ntx_exact_n_xi", 64)),
            surface_backend=str(neoclassical_cfg.get("ntx_exact_surface_backend", "booz")),
            max_iter=geom_cfg.get("vmec_max_iter"),
            solver_device=str(geom_cfg.get("vmec_implicit_solver_device", "default")),
            progress_label="[autodiff-gate] realtime geometry payload pullback:",
        )
        geometry_pullback_mode = "payload_state_raw_block_transpose"
        geometry_gradient_matrix = jax.block_until_ready(geometry_gradient_matrix)
        print(
            "[autodiff-gate] progress: geometry support pullback complete "
            f"mode={geometry_pullback_mode} elapsed_s={time.perf_counter() - t_phase:.3f}",
            flush=True,
        )
        elapsed_s = time.perf_counter() - t_start

        objective_values_np = np.asarray(jax.device_get(objective_values), dtype=float)
        profile_gradient_np = np.asarray(jax.device_get(profile_gradient_matrix), dtype=float)
        geometry_gradient_np = np.asarray(jax.device_get(geometry_gradient_matrix), dtype=float)
        objective_finite_np = np.isfinite(objective_values_np)
        profile_gradient_finite_by_objective = {
            objective_name: bool(np.all(np.isfinite(profile_gradient_np[objective_i])))
            for objective_i, objective_name in enumerate(OBJECTIVE_LABELS)
        }
        geometry_gradient_finite_by_objective = {
            objective_name: bool(np.all(np.isfinite(geometry_gradient_np[objective_i])))
            for objective_i, objective_name in enumerate(OBJECTIVE_LABELS)
        }
        realtime_geometry_diagnostics = _geometry_volume_diagnostics(baseline_runtime.geometry)
        support_bar_summary_by_objective = {}
        support_bar_l2_by_objective = {}
        support_bar_branch_diagnostics_by_objective = {}
        for objective_i, objective_name in enumerate(OBJECTIVE_LABELS):
            support_bar = support_bars[objective_i]
            support_bar_summary_by_objective[objective_name] = _payload_leaf_summary(support_bar)
            support_bar_l2_by_objective[objective_name] = _tree_array_l2_norm(support_bar)
            support_bar_branch_diagnostics_by_objective[objective_name] = (
                _payload_branch_diagnostics(support_bar)
            )

        support_summary = _payload_leaf_summary(support_payload)
        report = {
            "mode": "transport_reverse_ad_only",
            "parameter_mode": str(args.reverse_parameter_mode),
            "config_path": str(Path(args.config)),
            "objective_name": "all",
            "objective_order": list(OBJECTIVE_LABELS),
            "parameter_order": list(PARAMETER_ORDER),
            "profile_baseline_values": np.asarray(jax.device_get(profile_values), dtype=float).tolist(),
            "objective_values": {
                name: float(value) for name, value in zip(OBJECTIVE_LABELS, objective_values_np.tolist())
            },
            "objective_finite": {
                name: bool(value) for name, value in zip(OBJECTIVE_LABELS, objective_finite_np.tolist())
            },
            "profile_gradient_all_finite_by_objective": profile_gradient_finite_by_objective,
            "geometry_gradient_all_finite_by_objective": geometry_gradient_finite_by_objective,
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
            "geometry_gradient_reverse_ad": {
                objective_name: {
                    _format_reverse_geometry_parameter(geometry_parameter_name): float(
                        geometry_gradient_np[objective_i, 0]
                    )
                }
                for objective_i, objective_name in enumerate(OBJECTIVE_LABELS)
            },
            "geometry_baseline_values": {
                _format_reverse_geometry_parameter(geometry_parameter_name): float(
                    np.asarray(jax.device_get(baseline_geometry_deltas[0]), dtype=float)
                )
            },
            "accepted_step_limit": None
            if args.accepted_step_limit is None
            else int(args.accepted_step_limit),
            "reverse_segment_length": None
            if args.reverse_segment_length is None
            else int(args.reverse_segment_length),
            "reverse_stage_cotangent_mode_requested": str(args.reverse_stage_cotangent_mode),
            "reverse_stage_cotangent_mode_effective": support_probe_cotangent_mode,
            "ntx_exact_derivative_mode": str(args.ntx_exact_derivative_mode),
            "ntx_exact_derivative_field_pullback_mode": str(
                args.ntx_exact_derivative_field_pullback_mode
            ),
            "ntx_exact_surface_backend": str(neoclassical_cfg.get("ntx_exact_surface_backend", "booz")),
            "realtime_geometry_gradient_path": str(args.realtime_geometry_gradient_path),
            "realtime_primal_runtime_builder": "build_runtime_context",
            "realtime_geometry_derivative_boundary": (
                "runtime_geometry_and_ntx_exact_lij_support_payload"
                if combined_geometry_payload
                else "ntx_exact_lij_support_payload_only_diagnostic"
            ),
            "realtime_geometry_derivative_complete": bool(combined_geometry_payload),
            "geometry_support_pullback_mode": geometry_pullback_mode,
            "realtime_geometry_diagnostics": realtime_geometry_diagnostics,
            "support_payload_summary": support_summary,
            "support_bar_summary_by_objective": support_bar_summary_by_objective,
            "support_bar_l2_by_objective": support_bar_l2_by_objective,
            "support_bar_branch_diagnostics_by_objective": (
                support_bar_branch_diagnostics_by_objective
            ),
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
            "objective=all "
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
        geometry_label = _format_reverse_geometry_parameter(geometry_parameter_name)
        print("[autodiff-gate] reverse geometry gradients by objective:")
        for objective_name in OBJECTIVE_LABELS:
            value = report["geometry_gradient_reverse_ad"][objective_name][geometry_label]
            print(f"  - d{objective_name}/d{geometry_label}: ad={value:.6e}")
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
        accepted_step_limit_override=args.accepted_step_limit,
        reverse_segment_length=args.reverse_segment_length,
        reverse_direct_stage_adjoint=True,
        reverse_stage_adjoint_solve_mode=args.reverse_stage_adjoint_solve_mode,
        reverse_rhs_transpose_mode=args.reverse_rhs_transpose_mode,
        reverse_stage_cotangent_mode=args.reverse_stage_cotangent_mode,
        reverse_step_bwd_mode=args.reverse_step_bwd_mode,
        reverse_stage_adjoint_memory_mode=args.reverse_stage_adjoint_memory_mode,
        reverse_stage_adjoint_iter_maxiter=args.reverse_stage_adjoint_iter_maxiter,
        reverse_stage_adjoint_iter_tol=args.reverse_stage_adjoint_iter_tol,
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
    if args.local_transpose_diagnostic_accepted_step is not None:
        raise SystemExit(
            "[autodiff-gate] local accepted-step transpose diagnostics are only "
            "available for the profile-only static reverse setup."
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
        [float(profile_cfg[name]) for name in PARAMETER_ORDER]
        + [float(geom_cfg.get("vmec_param_delta", 0.0))],
        dtype=jnp.float64,
    )
    baseline_geometry_delta = float(geom_cfg.get("vmec_param_delta", 0.0))
    # Use the same entrypoint as the realtime forward solver for the primal
    # runtime.  Geometry-context helpers below are only for derivative-side
    # support-payload pullbacks against VMEC harmonics.
    baseline_runtime, baseline_state = build_runtime_context(config)
    baseline_profile_state = _initial_state_for_parameter_vector(
        baseline_values[: len(PARAMETER_ORDER)],
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        runtime=baseline_runtime,
    )
    baseline_components = prepare_transport_solver_components(config, baseline_runtime, baseline_profile_state)
    static_solver = baseline_components["solver"]
    print(
        "[autodiff-gate] realtime geometry device: "
        f"default_backend={jax.default_backend()} "
        f"baseline_values_device={_array_device_summary(baseline_values)} "
        f"local_devices={[str(device) for device in jax.local_devices()]}",
        flush=True,
    )
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
        choices=tuple(OBJECTIVE_LABELS) + ("all",),
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
            "one realtime VMEC boundary parameter to the four profile parameters."
        ),
    )
    parser.add_argument(
        "--reverse-geometry-parameter",
        type=str,
        default="RBC:1:0",
        help=(
            "Realtime VMEC geometry parameter used when --reverse-parameter-mode "
            "is profiles_plus_realtime_geometry. Syntax: FAMILY:m:n, e.g. RBC:1:0."
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
    parser.add_argument("--device", type=str, default=None, help="Optional device override.")
    parser.add_argument(
        "--accepted-step-limit",
        type=int,
        default=None,
        help="Optional accepted-step prefix to stop the adaptive rollout.",
    )
    parser.add_argument(
        "--ntx-exact-derivative-mode",
        default="direct",
        choices=(
            "direct",
            "custom_jvp",
            "custom_vjp",
            "recompute_vjp",
            "iterative_vjp",
            "iterative_jvp",
        ),
        help="NTX exact-runtime derivative mode.",
    )
    parser.add_argument(
        "--ntx-exact-derivative-field-pullback-mode",
        default="compact_vjp",
        choices=("generic_jvp", "compact_vjp"),
        help=(
            "Reverse-only NTX derivative-field pullback mode. 'compact_vjp' uses "
            "the NTX compact second-order coefficient-solve VJP helper and is the "
            "intended reverse-lane path. 'generic_jvp' keeps the older fallback "
            "that can compile through NTX factorization JVPs."
        ),
    )
    parser.add_argument(
        "--ntx-exact-derivative-pullback-boundary",
        default="inline",
        choices=("inline", "per_energy_jit"),
        help=(
            "Reverse-only boundary mode for the compact NTX derivative-field "
            "pullback. 'inline' keeps the current monolithic reverse kernel. "
            "'per_energy_jit' wraps each per-energy derivative pullback in its "
            "own JIT call boundary to test XLA graph partitioning."
        ),
    )
    parser.add_argument(
        "--ntx-exact-derivative-pullback-algebra",
        default="ntx_helper",
        choices=(
            "ntx_helper",
            "scalar_contract",
            "scalar_contract_lowdot",
            "scalar_contract_lowdot_sequential",
            "scalar_contract_lowdot_ntx",
            "scalar_contract_lowdot_recompute",
            "scalar_contract_matrix_free",
        ),
        help=(
            "Reverse-only algebra mode for compact NTX derivative-field "
            "pullbacks. 'ntx_helper' uses NTX's current compact helper. "
            "'scalar_contract' uses a NEOPAX-local scalar-contraction path "
            "that avoids Python-unrolled mode loops where possible. "
            "'scalar_contract_lowdot' additionally avoids full tangent-mode "
            "stacks for the field-bar contraction. "
            "'scalar_contract_lowdot_sequential' keeps that exact algebra but "
            "assembles the energy-scan bars with a sequential JAX loop to test "
            "whether factorization temporaries stop being live across the full "
            "energy axis. "
            "'scalar_contract_lowdot_ntx' moves the fused lowdot algebra into "
            "NTX while keeping NEOPAX's transport-moment cotangent mapping. "
            "'scalar_contract_lowdot_recompute' recomputes the lowdot adjoint "
            "before field-dot contractions to test peak-memory reduction. "
            "'scalar_contract_matrix_free' avoids saved LU-factor tensors by "
            "using Krylov solves on the NTX block operator."
        ),
    )
    parser.add_argument(
        "--reverse-ntx-prepared-solve-boundary",
        default="default",
        choices=("default", "custom_vjp", "recompute_vjp"),
        help=(
            "Reverse-only diagnostic boundary for the NTX prepared coefficient solve. "
            "'default' preserves --ntx-exact-derivative-mode. 'custom_vjp' forces "
            "the response solve through NTX's custom-VJP boundary without changing "
            "the forward-AD lane. 'recompute_vjp' uses an exact custom-VJP boundary "
            "that rebuilds the NTX factorization in backward instead of saving it "
            "in the forward residual."
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
        choices=("structured", "bicgstab", "block", "gmres"),
        default="structured",
        help=(
            "Reverse stage-adjoint linear solve. 'structured' uses the Radau "
            "transformed LU transpose approximation and is the lightweight default; "
            "'bicgstab' is the lower-memory exact iterative candidate; 'block' and "
            "'gmres' are correctness oracles but are memory/compile heavy."
        ),
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
            "backward body inside segmented replay; 'force_reuse_bwd' and 'force_rebuild_bwd' "
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
            "accepted-step reverse scan. 'reduced_cotangent_lean_replay' stores "
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
        "--diagnose-final-objective-cotangent",
        action="store_true",
        help=(
            "For scalar objectives, also print the norm/max/nonzero count of "
            "grad(objective(final_y)) before the realized-schedule reverse rule."
        ),
    )
    args = parser.parse_args()
    if int(args.reverse_stage_adjoint_iter_maxiter) <= 0:
        raise SystemExit("[autodiff-gate] --reverse-stage-adjoint-iter-maxiter must be positive.")
    if float(args.reverse_stage_adjoint_iter_tol) <= 0.0:
        raise SystemExit("[autodiff-gate] --reverse-stage-adjoint-iter-tol must be positive.")
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
    if str(args.reverse_ntx_prepared_solve_boundary) in {"custom_vjp", "recompute_vjp"}:
        if str(args.ntx_exact_derivative_mode) not in {
            "direct",
            "custom_vjp",
            "recompute_vjp",
        }:
            raise SystemExit(
                "[autodiff-gate] --reverse-ntx-prepared-solve-boundary custom_vjp "
                "or recompute_vjp is only compatible with --ntx-exact-derivative-mode "
                "direct, custom_vjp, or recompute_vjp."
            )
        effective_ntx_exact_derivative_mode = str(args.reverse_ntx_prepared_solve_boundary)

    config = _prepare_benchmark_config(
        Path(args.config),
        device=args.device,
        ntx_exact_derivative_mode=effective_ntx_exact_derivative_mode,
        ntx_exact_derivative_field_pullback_mode=args.ntx_exact_derivative_field_pullback_mode,
        ntx_exact_derivative_pullback_boundary=args.ntx_exact_derivative_pullback_boundary,
        ntx_exact_derivative_pullback_algebra=args.ntx_exact_derivative_pullback_algebra,
        radau_jacobian_reuse_mode=args.radau_jacobian_reuse_mode,
    )
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
    baseline_values = jnp.asarray(
        [float(profile_cfg[name]) for name in PARAMETER_ORDER],
        dtype=jnp.asarray(baseline_state.pressure).dtype,
    )
    objective_index = None if args.objective == "all" else OBJECTIVE_LABELS.index(args.objective)
    reverse_setup = _prepare_reverse_static_setup(
        baseline_values,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        accepted_step_limit_override=args.accepted_step_limit,
        reverse_segment_length=reverse_segment_length,
        reverse_direct_stage_adjoint=reverse_direct_stage_adjoint,
        reverse_stage_adjoint_solve_mode=str(args.reverse_stage_adjoint_solve_mode),
        reverse_rhs_transpose_mode=str(args.reverse_rhs_transpose_mode),
        reverse_stage_cotangent_mode=str(args.reverse_stage_cotangent_mode),
        reverse_step_bwd_mode=str(args.reverse_step_bwd_mode),
        reverse_stage_adjoint_memory_mode=str(args.reverse_stage_adjoint_memory_mode),
        reverse_stage_adjoint_iter_maxiter=int(args.reverse_stage_adjoint_iter_maxiter),
        reverse_stage_adjoint_iter_tol=float(args.reverse_stage_adjoint_iter_tol),
    )

    if args.local_transpose_diagnostic_accepted_step is not None:
        accepted_step_index = int(args.local_transpose_diagnostic_accepted_step)
        if accepted_step_index < 0:
            raise SystemExit("[autodiff-gate] --local-transpose-diagnostic-accepted-step must be >= 0.")
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
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            reverse_setup=reverse_setup,
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
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        objective_index=objective_index,
        reverse_setup=reverse_setup,
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
