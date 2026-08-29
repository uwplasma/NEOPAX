from __future__ import annotations

import argparse
import atexit
import contextlib
import dataclasses
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = ROOT.parent
for path in (ROOT, WORKSPACE_ROOT / "vmec_jax"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_transport_autodiff_lagged_ntx import (  # noqa: E402
    DEFAULT_CONFIG,
    OBJECTIVE_LABELS,
    _adaptive_rollout_diagnostics,
    _baseline_profile_cfg,
    _fd_step,
    _forward_benchmark_adaptive_rollout_objectives_for_parameter_on_frozen_trace,
    _objective_vector,
    _prepare_benchmark_config,
    _truncate_rollout_trace_by_accepted_steps,
)
from benchmark_transport_reverse_ad_only import (  # noqa: E402
    PARAMETER_ORDER,
    _find_ntx_support_payload,
    _geometry_context_from_config,
    _geometry_param_specs_from_parameter_name,
    _initial_state_for_parameter_vector,
    _initial_er_root_ad_mode,
    _profile_cfg_scalar_value,
    _runtime_with_ntx_support_payload,
)
from NEOPAX._reverse_ad_optimization import (  # noqa: E402
    INITIAL_ER_ROOT_ONLY_EXPLICIT_OBJECTIVES,
    _initial_er_root_only_objective_values,
    normalize_initial_er_root_only_objective_names,
)
from NEOPAX._reverse_ad_initial_er import (  # noqa: E402
    initial_er_charge_flux_residual_er_derivative,
    initial_er_charge_flux_residuals,
)
from NEOPAX._geometry_autodiff import (  # noqa: E402
    _geometry_full_ad_objectives_from_state,
    _implicit_params_with_boundary_deltas,
    boundary_param_entries,
    build_runtime_context_for_geometry_param,
    build_runtime_context_for_vmec_state,
    geometry_observable_names_for_kind,
)
from NEOPAX._orchestrator import (  # noqa: E402
    _execution_device_context,
    build_runtime_context,
    prepare_transport_solver_components,
)
from NEOPAX._transport_solvers import (  # noqa: E402
    _build_prepared_radau_accepted_rollout,
    _build_prepared_radau_execution_context,
    _radau_adaptive_schedule_rollout,
    _radau_forward_fd_run_prepared_on_realized_trace,
    _radau_run_prepared_on_realized_trace,
)
from vmex.core import implicit as im  # noqa: E402


def _report_path(parameter_name: str, *, root_fd_lane: str = "selected") -> Path:
    safe_name = parameter_name.replace(":", "_")
    outdir = ROOT / "outputs" / "autodiff_transport_lagged_ntx" / "realtime_geometry_fd"
    outdir.mkdir(parents=True, exist_ok=True)
    lane_suffix = "" if str(root_fd_lane) == "selected" else f"_{root_fd_lane}"
    return outdir / f"{safe_name}_forward_fd{lane_suffix}_summary.json"


def _root_only_report_path(parameter_name: str, *, root_fd_lane: str = "selected") -> Path:
    safe_name = parameter_name.replace(":", "_")
    outdir = ROOT / "outputs" / "autodiff_transport_lagged_ntx" / "realtime_geometry_fd"
    outdir.mkdir(parents=True, exist_ok=True)
    lane_suffix = "" if str(root_fd_lane) == "selected" else f"_{root_fd_lane}"
    return outdir / f"{safe_name}_initial_er_root_only_fd{lane_suffix}_summary.json"


def _tree_all_finite(tree) -> bool:
    for leaf in jax.tree_util.tree_leaves(tree):
        arr = np.asarray(jax.device_get(leaf))
        if np.issubdtype(arr.dtype, np.inexact) and not np.all(np.isfinite(arr)):
            return False
    return True


def _profile_state_from_values(
    values,
    *,
    config: dict[str, Any] | None = None,
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    initial_er_root_ad: str = "off",
):
    return _initial_state_for_parameter_vector(
        values,
        config=config,
        initial_er_root_ad=initial_er_root_ad,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        runtime=runtime,
    )


def _prepare_rollout(config: dict[str, Any], runtime, state0, *, solver_override=None):
    kwargs = {}
    if solver_override is not None and "solver_override" in inspect.signature(prepare_transport_solver_components).parameters:
        kwargs["solver_override"] = solver_override
    components = prepare_transport_solver_components(config, runtime, state0, **kwargs)
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
    return components, prepared_rollout, execution_context


def _schedule_rollout(config: dict[str, Any], runtime, state0, *, accepted_step_limit: int | None):
    components, prepared_rollout, execution_context = _prepare_rollout(config, runtime, state0)
    solver = components["solver"]
    stop_after_accepted_steps = (
        int(accepted_step_limit)
        if accepted_step_limit is not None
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
    return components, rollout, final_state


def _objectives_on_realtime_geometry_frozen_trace(
    *,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    profile_values,
    frozen_trace,
    replay_mode: str,
    solver_override=None,
    initial_er_root_ad: str = "off",
    initial_er_root_fd_root_lane: str = "selected",
    baseline_er_profile=None,
    baseline_residual=None,
    baseline_dres_der=None,
):
    root_lane = str(initial_er_root_fd_root_lane).strip().lower()
    if root_lane == "frozen_linearized":
        if baseline_er_profile is None or baseline_residual is None or baseline_dres_der is None:
            raise ValueError("frozen-linearized full-transport initial-Er FD requires baseline root data.")
        state0 = _state_with_frozen_linearized_initial_er_root(
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            profile_values=profile_values,
            baseline_er_profile=baseline_er_profile,
            baseline_residual=baseline_residual,
            baseline_dres_der=baseline_dres_der,
        )
    else:
        state0 = _profile_state_from_values(
            profile_values,
            config=config,
            initial_er_root_ad=initial_er_root_ad,
            runtime=runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
        )
    components, prepared_rollout, execution_context = _prepare_rollout(
        config,
        runtime,
        state0,
        solver_override=solver_override,
    )
    replay_mode_normalized = str(replay_mode).strip().lower()
    if replay_mode_normalized == "accepted":
        replay = _radau_forward_fd_run_prepared_on_realized_trace(
            prepared_rollout,
            execution_context,
            frozen_trace,
            replay_mode="accepted",
            carry0=prepared_rollout.initial_carry,
        )
    else:
        replay = _radau_run_prepared_on_realized_trace(
            prepared_rollout,
            execution_context,
            frozen_trace,
            replay_mode=replay_mode_normalized,
            carry0=prepared_rollout.initial_carry,
        )
    return _objective_vector(replay["final_state"], runtime), replay


def _runtime_for_geometry_delta(
    config: dict[str, Any],
    geometry_parameter: str,
    delta_value,
    *,
    return_vmec_state: bool = False,
):
    geom_cfg = config.get("geometry", {})
    geometry_context = _geometry_context_from_config(config, geometry_parameter)
    specs = _geometry_param_specs_from_parameter_name(geometry_parameter)
    if len(specs) != 1:
        raise ValueError("This FD benchmark currently expects one geometry parameter.")
    result = build_runtime_context_for_geometry_param(
        config,
        geometry_context,
        jnp.asarray(delta_value, dtype=jnp.float64),
        lane=str(geom_cfg.get("vmec_lane", "forward")).strip().lower(),
        n_r=int(geom_cfg.get("n_radial", 51)),
        max_iter=geom_cfg.get("vmec_max_iter"),
        step_size=geom_cfg.get("vmec_step_size"),
        jacobian_penalty=float(geom_cfg.get("vmec_jacobian_penalty", 1.0e3)),
        return_vmec_state=return_vmec_state,
    )
    if return_vmec_state:
        runtime, transport_state, state_vmec = result
        return runtime, transport_state, state_vmec, geometry_context
    return result


def _geometry_full_objective_vector(context, state_vmec) -> jax.Array:
    """Canonical scalar geometry rows shared with the reverse-table path."""

    names = geometry_observable_names_for_kind("geometry_full_ad_objectives")
    values = _geometry_full_ad_objectives_from_state(context, state_vmec)
    return jnp.stack([jnp.asarray(values[name], dtype=jnp.float64).reshape(()) for name in names])


def _param_unit_tangent_like(params, entry: dict[str, Any]):
    leaves = {
        field.name: jnp.zeros_like(getattr(params, field.name))
        for field in dataclasses.fields(params)
    }
    field_name = str(entry["input_field"])
    leaves[field_name] = leaves[field_name].at[
        int(entry["n_offset"]),
        int(entry["m_index"]),
    ].set(1.0)
    return dataclasses.replace(params, **leaves)


def _manual_implicit_forward_state_tangent(*, params, param_tangent, cfg, x_star, dof_mask):
    if hasattr(im, "implicit_state_tangent_raw_block"):
        return im.implicit_state_tangent_raw_block(
            params,
            cfg,
            x_star,
            dof_mask,
            param_tangent,
            probe_chunk_size=1,
        )

    frozen = jax.lax.stop_gradient(x_star)
    edge_mask = im._edge_mask(cfg)
    P = im._dof_projector(cfg, dof_mask)
    F = im.residual_fn(cfg, frozen, dof_mask)
    z_star = P(x_star)

    rhs = jax.tree.map(
        jnp.negative,
        jax.jvp(lambda prm: F(z_star, prm), (params,), (param_tangent,))[1],
    )
    dz, _ = im._adjoint_solve(
        lambda v: jax.jvp(lambda z: F(z, params), (z_star,), (v,))[1],
        rhs,
        cfg,
    )

    def assemble_from_z_params(z, prm):
        return im._assemble(
            z,
            im.runtime_from_params(prm, cfg),
            frozen,
            P,
            edge_mask,
        )

    return jax.jvp(
        assemble_from_z_params,
        (z_star, params),
        (dz, param_tangent),
    )[1]


def _frozen_linearized_vmec_geometry_bundle(
    config: dict[str, Any],
    geometry_parameter: str,
    *,
    baseline_delta: float,
):
    geom_cfg = config.get("geometry", {})
    geometry_context = _geometry_context_from_config(config, geometry_parameter)
    specs = _geometry_param_specs_from_parameter_name(geometry_parameter)
    if len(specs) != 1:
        raise ValueError("This FD benchmark currently expects one geometry parameter.")
    (entry,) = boundary_param_entries(geometry_context, specs)
    solver_device = str(geom_cfg.get("vmec_implicit_solver_device", "default"))
    params0 = _implicit_params_with_boundary_deltas(
        geometry_context,
        im,
        jnp.asarray([0.0], dtype=jnp.float64),
        (entry,),
        solver_device=solver_device,
    )
    params = _implicit_params_with_boundary_deltas(
        geometry_context,
        im,
        jnp.asarray([baseline_delta], dtype=jnp.float64),
        (entry,),
        solver_device=solver_device,
    )
    cfg_kwargs = {
        "mode": "cli",
        "multigrid": True,
    }
    if geom_cfg.get("vmec_max_iter") is not None:
        cfg_kwargs["max_iterations"] = int(geom_cfg["vmec_max_iter"])
    cfg = im.make_config(geometry_context.indata, **cfg_kwargs)

    print("[autodiff-gate] progress: baseline implicit VMEC solve for frozen-linearized geometry FD", flush=True)
    state_star, dof_mask = im.solve_implicit_with_aux(params, cfg)
    param_tangent = _param_unit_tangent_like(params0, entry)
    state_tangent = _manual_implicit_forward_state_tangent(
        params=params,
        param_tangent=param_tangent,
        cfg=cfg,
        x_star=state_star,
        dof_mask=dof_mask,
    )
    coefficient_value = float(entry["baseline_coefficient"]) + float(baseline_delta)
    return {
        "context": geometry_context,
        "state_star": state_star,
        "state_tangent": state_tangent,
        "coefficient_value": coefficient_value,
        "n_r": int(geom_cfg.get("n_radial", 51)),
    }


def _runtime_for_frozen_linearized_geometry_step(
    config: dict[str, Any],
    bundle: dict[str, Any],
    *,
    step_scale: float,
    fixed_initial_er=None,
):
    state = jax.tree.map(
        lambda value, tangent: value + jnp.asarray(step_scale, dtype=jnp.float64) * tangent,
        bundle["state_star"],
        bundle["state_tangent"],
    )
    runtime, state = build_runtime_context_for_vmec_state(
        config,
        bundle["context"],
        state,
        n_r=int(bundle["n_r"]),
    )
    if fixed_initial_er is not None:
        state = dataclasses.replace(state, Er=jnp.asarray(fixed_initial_er, dtype=state.Er.dtype))
    return runtime, state


def _geometry_fd_objectives(
    *,
    config: dict[str, Any],
    geometry_parameter: str,
    geometry_delta: float,
    profile_values,
    profile_cfg: dict[str, Any],
    frozen_trace,
    replay_mode: str,
    geometry_fd_lane: str,
    frozen_linearized_bundle: dict[str, Any] | None = None,
    fixed_initial_er=None,
    initial_er_root_ad: str = "off",
    initial_er_root_fd_root_lane: str = "selected",
    baseline_er_profile=None,
    baseline_residual=None,
    baseline_dres_der=None,
    include_vmec_main_geometry_objectives: bool = False,
):
    if str(geometry_fd_lane).strip().lower() == "frozen_linearized":
        if frozen_linearized_bundle is None:
            raise ValueError("frozen_linearized geometry FD requires a baseline VMEC tangent bundle.")
        baseline_delta = float(config.get("geometry", {}).get("vmec_param_delta", 0.0))
        runtime, baseline_state = _runtime_for_frozen_linearized_geometry_step(
            config,
            frozen_linearized_bundle,
            step_scale=float(geometry_delta) - baseline_delta,
            fixed_initial_er=fixed_initial_er,
        )
        state_vmec = jax.tree.map(
            lambda value, tangent: value
            + jnp.asarray(float(geometry_delta) - baseline_delta, dtype=jnp.float64) * tangent,
            frozen_linearized_bundle["state_star"],
            frozen_linearized_bundle["state_tangent"],
        )
        geometry_context = frozen_linearized_bundle["context"]
    else:
        runtime, baseline_state, state_vmec, geometry_context = _runtime_for_geometry_delta(
            config,
            geometry_parameter,
            geometry_delta,
            return_vmec_state=True,
        )
    transport_objectives, replay = _objectives_on_realtime_geometry_frozen_trace(
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        profile_values=profile_values,
        frozen_trace=frozen_trace,
        replay_mode=replay_mode,
        initial_er_root_ad=initial_er_root_ad,
        initial_er_root_fd_root_lane=initial_er_root_fd_root_lane,
        baseline_er_profile=baseline_er_profile,
        baseline_residual=baseline_residual,
        baseline_dres_der=baseline_dres_der,
    )
    if not include_vmec_main_geometry_objectives:
        return transport_objectives, replay
    return jnp.concatenate(
        [transport_objectives, _geometry_full_objective_vector(geometry_context, state_vmec)]
    ), replay


def _root_only_objective_names(raw: str) -> tuple[str, ...]:
    if str(raw).strip().lower() == "all":
        return INITIAL_ER_ROOT_ONLY_EXPLICIT_OBJECTIVES
    return normalize_initial_er_root_only_objective_names(raw)


def _initial_er_root_only_objectives(
    *,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    profile_values,
    objective_names: tuple[str, ...],
    initial_er_root_ad: str,
):
    state = _profile_state_from_values(
        profile_values,
        config=config,
        initial_er_root_ad=initial_er_root_ad,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
    )
    return _initial_er_root_only_objective_values(state, runtime, objective_names), state


def _initial_er_root_only_frozen_linearized_objectives(
    *,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    profile_values,
    objective_names: tuple[str, ...],
    baseline_er_profile,
    baseline_residual,
    baseline_dres_der,
):
    del config
    pre_root_state = _profile_state_from_values(
        profile_values,
        initial_er_root_ad="off",
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
    )
    residual_at_baseline_er = initial_er_charge_flux_residuals(
        pre_root_state,
        jnp.asarray(baseline_er_profile, dtype=pre_root_state.Er.dtype),
        runtime=runtime,
    )
    residual_delta = residual_at_baseline_er - jnp.asarray(baseline_residual, dtype=residual_at_baseline_er.dtype)
    safe_dres_der = jnp.where(
        jnp.abs(baseline_dres_der) > jnp.asarray(1.0e-30, dtype=residual_delta.dtype),
        jnp.asarray(baseline_dres_der, dtype=residual_delta.dtype),
        jnp.inf,
    )
    er_profile = jnp.asarray(baseline_er_profile, dtype=pre_root_state.Er.dtype) - residual_delta / safe_dres_der
    rooted_state = dataclasses.replace(pre_root_state, Er=er_profile)
    return _initial_er_root_only_objective_values(rooted_state, runtime, objective_names), rooted_state


def _state_with_frozen_linearized_initial_er_root(
    *,
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    profile_values,
    baseline_er_profile,
    baseline_residual,
    baseline_dres_der,
):
    pre_root_state = _profile_state_from_values(
        profile_values,
        initial_er_root_ad="off",
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
    )
    residual_at_baseline_er = initial_er_charge_flux_residuals(
        pre_root_state,
        jnp.asarray(baseline_er_profile, dtype=pre_root_state.Er.dtype),
        runtime=runtime,
    )
    residual_delta = residual_at_baseline_er - jnp.asarray(baseline_residual, dtype=residual_at_baseline_er.dtype)
    safe_dres_der = jnp.where(
        jnp.abs(baseline_dres_der) > jnp.asarray(1.0e-30, dtype=residual_delta.dtype),
        jnp.asarray(baseline_dres_der, dtype=residual_delta.dtype),
        jnp.inf,
    )
    er_profile = jnp.asarray(baseline_er_profile, dtype=pre_root_state.Er.dtype) - residual_delta / safe_dres_der
    return dataclasses.replace(pre_root_state, Er=er_profile)


def _root_only_er_fd_diagnostics(minus_state, baseline_state, plus_state, h: float) -> dict[str, Any]:
    er_minus = np.asarray(jax.device_get(minus_state.Er), dtype=float)
    er_baseline = np.asarray(jax.device_get(baseline_state.Er), dtype=float)
    er_plus = np.asarray(jax.device_get(plus_state.Er), dtype=float)
    d_er = (er_plus - er_minus) / (2.0 * float(h))
    abs_d_er = np.abs(d_er)
    top_count = int(min(8, abs_d_er.size))
    top_indices = np.argsort(abs_d_er)[-top_count:][::-1]
    return {
        "Er_minus": er_minus.tolist(),
        "Er_baseline": er_baseline.tolist(),
        "Er_plus": er_plus.tolist(),
        "dEr_fd": d_er.tolist(),
        "dEr_fd_min": float(np.min(d_er)),
        "dEr_fd_max": float(np.max(d_er)),
        "dEr_fd_max_abs": float(np.max(abs_d_er)),
        "dEr_fd_top_abs_indices": [int(index) for index in top_indices.tolist()],
        "dEr_fd_top_abs_values": [float(d_er[index]) for index in top_indices.tolist()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Forward finite-difference benchmark for realtime VMEC/Boozer geometry. "
            "The primal runtime is built through the same realtime forward solver path."
        )
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="Benchmark TOML.")
    parser.add_argument(
        "--parameter",
        type=str,
        default="RBC:1:0",
        help=(
            "Profile parameter name (n0, T0, density_shape_power, temperature_shape_power) "
            "or realtime geometry harmonic FAMILY:m:n."
        ),
    )
    parser.add_argument("--fd-rel-step", type=float, default=3.0e-7)
    parser.add_argument("--fd-abs-step", type=float, default=1.0e-10)
    parser.add_argument("--accepted-step-limit", type=int, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="default",
        help=(
            "Benchmark placement. 'default' uses JAX's current default device, "
            "'auto' leaves placement untouched, and 'cpu'/'gpu' force that backend."
        ),
    )
    parser.add_argument(
        "--geometry-fd-lane",
        choices=("frozen_linearized", "nonlinear_resolve"),
        default="frozen_linearized",
        help=(
            "Geometry-parameter FD oracle. 'frozen_linearized' freezes the baseline VMEC "
            "implicit solve and perturbs state_star +/- h*dstate/dp; 'nonlinear_resolve' "
            "keeps the older full VMEC re-solve endpoint diagnostic."
        ),
    )
    parser.add_argument(
        "--replay-mode",
        choices=("attempt", "accepted"),
        default="accepted",
        help=(
            "Frozen trace replay mode. 'accepted' uses the same solver-native fixed accepted-dt "
            "map as benchmark_transport_frozen_fd_only; 'attempt' keeps the older diagnostic trace replay."
        ),
    )
    parser.add_argument("--radau-jacobian-reuse-mode", default=None)
    parser.add_argument(
        "--split-payload-fd-diagnostic",
        action="store_true",
        help=(
            "For realtime geometry parameters, additionally split the final-state FD "
            "diagnostic into geometry-metric-only and NTX-support-only branches."
        ),
    )
    parser.add_argument(
        "--include-vmec-main-geometry-objectives",
        action="store_true",
        help=(
            "Append the shared VMEX-main geometry table rows (including QI, "
            "max-J, and the scalar physical-Mercier softmax objective) to the "
            "transport FD vector. This is valid only for a realtime geometry "
            "parameter. For reverse-AD parity, use --geometry-fd-lane nonlinear_resolve."
        ),
    )
    parser.add_argument(
        "--initial-Er-root-ad",
        dest="initial_er_root_ad",
        default="off",
        choices=("off", "jax_selected_root"),
        help=(
            "Opt-in FD diagnostic for ambipolar initial Er. 'off' preserves the "
            "current oracle. 'jax_selected_root' recomputes the selected best-root "
            "profile with the same JAX-returning path used by the reverse benchmark."
        ),
    )
    parser.add_argument(
        "--initial-Er-root-only-fd",
        dest="initial_er_root_only_fd",
        action="store_true",
        help=(
            "Stop after the initial ambipolar Er construction and finite-difference "
            "root-only scalar objectives. This reuses the realtime geometry/profile "
            "FD setup in this script but does not run the Radau time evolution."
        ),
    )
    parser.add_argument(
        "--initial-Er-root-only-fd-root-lane",
        dest="initial_er_root_only_fd_root_lane",
        choices=("selected", "frozen_linearized"),
        default="selected",
        help=(
            "Root-only FD oracle. 'selected' reruns the best-root selection at p +/- h. "
            "'frozen_linearized' freezes the baseline selected root branch and applies "
            "the same implicit residual linearization used by the AD rule."
        ),
    )
    parser.add_argument(
        "--initial-Er-root-fd-root-lane",
        dest="initial_er_root_fd_root_lane",
        choices=("selected", "frozen_linearized"),
        default="selected",
        help=(
            "Full-transport FD oracle for the initial-Er root. 'selected' preserves "
            "the existing behavior and reruns best-root selection at p +/- h. "
            "'frozen_linearized' freezes the baseline selected root branch before "
            "the Radau replay, using the same local implicit linearization as the "
            "root-only FD diagnostic."
        ),
    )
    parser.add_argument(
        "--root-only-objective",
        default="all",
        help=(
            "Comma-separated initial-Er root-only objectives for "
            "--initial-Er-root-only-fd, or 'all'. Choices are: "
            + ", ".join(INITIAL_ER_ROOT_ONLY_EXPLICIT_OBJECTIVES)
        ),
    )
    args = parser.parse_args()
    initial_er_root_ad = _initial_er_root_ad_mode(args.initial_er_root_ad)

    device_arg = str(args.device).strip().lower() if args.device is not None else "default"
    config_device = None if device_arg == "default" else args.device
    config = _prepare_benchmark_config(
        config_path=Path(args.config),
        device=config_device,
        ntx_exact_derivative_mode="direct",
        radau_jacobian_reuse_mode=args.radau_jacobian_reuse_mode,
    )
    if device_arg == "default":
        default_backend = jax.default_backend()
        selected_device = jax.local_devices(backend=default_backend)[0]
        device_context = jax.default_device(selected_device)
    elif device_arg == "auto":
        selected_device = None
        device_context = contextlib.nullcontext()
    else:
        selected_device = None
        device_context = _execution_device_context(config)
    device_context.__enter__()
    atexit.register(lambda: device_context.__exit__(None, None, None))
    print(
        "[autodiff-gate] device placement: "
        f"requested={device_arg} default_backend={jax.default_backend()} "
        f"selected_device={selected_device}",
        flush=True,
    )
    profile_cfg = _baseline_profile_cfg(config)
    parameter_name = str(args.parameter)
    geometry_fd_lane = str(args.geometry_fd_lane).strip().lower()
    parameter_is_profile = parameter_name in PARAMETER_ORDER
    root_only_fd = bool(args.initial_er_root_only_fd)
    include_vmec_main_geometry_objectives = bool(args.include_vmec_main_geometry_objectives)
    if include_vmec_main_geometry_objectives and (parameter_is_profile or root_only_fd):
        raise SystemExit(
            "--include-vmec-main-geometry-objectives requires a full-transport "
            "realtime geometry parameter; it is not defined for profile or root-only FD."
        )
    frozen_linearized_bundle = None
    if parameter_is_profile or geometry_fd_lane != "frozen_linearized":
        baseline_runtime, baseline_state = build_runtime_context(config)
    else:
        baseline_delta = float(config.get("geometry", {}).get("vmec_param_delta", 0.0))
        frozen_linearized_bundle = _frozen_linearized_vmec_geometry_bundle(
            config,
            parameter_name,
            baseline_delta=baseline_delta,
        )
        baseline_runtime, baseline_state = _runtime_for_frozen_linearized_geometry_step(
            config,
            frozen_linearized_bundle,
            step_scale=0.0,
        )
    profile_values = jnp.asarray(
        [_profile_cfg_scalar_value(profile_cfg, name) for name in PARAMETER_ORDER],
        dtype=jnp.float64,
    )
    baseline_profile_state = _profile_state_from_values(
        profile_values,
        config=config,
        initial_er_root_ad="off" if root_only_fd else initial_er_root_ad,
        runtime=baseline_runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
    )

    if root_only_fd:
        if initial_er_root_ad == "off":
            raise SystemExit(
                "[autodiff-gate] --initial-Er-root-only-fd requires "
                "--initial-Er-root-ad jax_selected_root."
            )
        objective_names = _root_only_objective_names(str(args.root_only_objective))
        if parameter_is_profile:
            parameter_kind = "profile"
            param_index = PARAMETER_ORDER.index(parameter_name)
            baseline_value = _profile_cfg_scalar_value(profile_cfg, parameter_name)
            h = _fd_step(baseline_value, rel_step=args.fd_rel_step, abs_step=args.fd_abs_step)
            minus_runtime = baseline_runtime
            plus_runtime = baseline_runtime
            minus_state = baseline_state
            plus_state = baseline_state
            minus_profile_values = profile_values.at[param_index].set(
                jnp.asarray(baseline_value - h, dtype=profile_values.dtype)
            )
            plus_profile_values = profile_values.at[param_index].set(
                jnp.asarray(baseline_value + h, dtype=profile_values.dtype)
            )
        else:
            parameter_kind = "realtime_geometry"
            geom_cfg = config.get("geometry", {})
            baseline_value = float(geom_cfg.get("vmec_param_delta", 0.0))
            if geometry_fd_lane == "frozen_linearized":
                if frozen_linearized_bundle is None:
                    raise ValueError("Missing frozen-linearized geometry bundle.")
                step_scale_value = float(frozen_linearized_bundle["coefficient_value"])
            else:
                geometry_context = _geometry_context_from_config(config, parameter_name)
                specs = _geometry_param_specs_from_parameter_name(parameter_name)
                (entry,) = boundary_param_entries(geometry_context, specs)
                step_scale_value = float(entry["baseline_coefficient"]) + baseline_value
            h = _fd_step(step_scale_value, rel_step=args.fd_rel_step, abs_step=args.fd_abs_step)
            if geometry_fd_lane == "frozen_linearized":
                minus_runtime, minus_state = _runtime_for_frozen_linearized_geometry_step(
                    config,
                    frozen_linearized_bundle,
                    step_scale=(baseline_value - h) - baseline_value,
                )
                plus_runtime, plus_state = _runtime_for_frozen_linearized_geometry_step(
                    config,
                    frozen_linearized_bundle,
                    step_scale=(baseline_value + h) - baseline_value,
                )
            else:
                minus_runtime, minus_state = _runtime_for_geometry_delta(config, parameter_name, baseline_value - h)
                plus_runtime, plus_state = _runtime_for_geometry_delta(config, parameter_name, baseline_value + h)
            minus_profile_values = profile_values
            plus_profile_values = profile_values

        print("[autodiff-gate] progress: running initial-Er root-only baseline", flush=True)
        baseline_objectives, baseline_rooted_state = _initial_er_root_only_objectives(
            config=config,
            runtime=baseline_runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            profile_values=profile_values,
            objective_names=objective_names,
            initial_er_root_ad=initial_er_root_ad,
        )
        root_fd_lane = str(args.initial_er_root_only_fd_root_lane)
        if root_fd_lane == "frozen_linearized":
            baseline_pre_root_state = _profile_state_from_values(
                profile_values,
                initial_er_root_ad="off",
                runtime=baseline_runtime,
                baseline_state=baseline_state,
                profile_cfg=profile_cfg,
            )
            baseline_er_profile = jnp.asarray(baseline_rooted_state.Er, dtype=baseline_pre_root_state.Er.dtype)
            baseline_residual = initial_er_charge_flux_residuals(
                baseline_pre_root_state,
                baseline_er_profile,
                runtime=baseline_runtime,
            )
            baseline_dres_der = initial_er_charge_flux_residual_er_derivative(
                baseline_pre_root_state,
                baseline_er_profile,
                runtime=baseline_runtime,
            )
            print(
                "[autodiff-gate] progress: running initial-Er root-only fd_minus "
                "(frozen-linearized root branch)",
                flush=True,
            )
            minus_objectives, minus_rooted_state = _initial_er_root_only_frozen_linearized_objectives(
                config=config,
                runtime=minus_runtime,
                baseline_state=minus_state,
                profile_cfg=profile_cfg,
                profile_values=minus_profile_values,
                objective_names=objective_names,
                baseline_er_profile=baseline_er_profile,
                baseline_residual=baseline_residual,
                baseline_dres_der=baseline_dres_der,
            )
        else:
            print("[autodiff-gate] progress: running initial-Er root-only fd_minus", flush=True)
            minus_objectives, minus_rooted_state = _initial_er_root_only_objectives(
                config=config,
                runtime=minus_runtime,
                baseline_state=minus_state,
                profile_cfg=profile_cfg,
                profile_values=minus_profile_values,
                objective_names=objective_names,
                initial_er_root_ad=initial_er_root_ad,
            )
        print("[autodiff-gate] progress: running initial-Er root-only fd_plus", flush=True)
        if root_fd_lane == "frozen_linearized":
            plus_objectives, plus_rooted_state = _initial_er_root_only_frozen_linearized_objectives(
                config=config,
                runtime=plus_runtime,
                baseline_state=plus_state,
                profile_cfg=profile_cfg,
                profile_values=plus_profile_values,
                objective_names=objective_names,
                baseline_er_profile=baseline_er_profile,
                baseline_residual=baseline_residual,
                baseline_dres_der=baseline_dres_der,
            )
        else:
            plus_objectives, plus_rooted_state = _initial_er_root_only_objectives(
                config=config,
                runtime=plus_runtime,
                baseline_state=plus_state,
                profile_cfg=profile_cfg,
                profile_values=plus_profile_values,
                objective_names=objective_names,
                initial_er_root_ad=initial_er_root_ad,
            )
        minus_objectives, baseline_objectives, plus_objectives = jax.block_until_ready(
            (minus_objectives, baseline_objectives, plus_objectives)
        )
        gradient_fd = jax.block_until_ready((plus_objectives - minus_objectives) / (2.0 * h))
        baseline_np = np.asarray(jax.device_get(baseline_objectives), dtype=float)
        gradient_np = np.asarray(jax.device_get(gradient_fd), dtype=float)
        minus_np = np.asarray(jax.device_get(minus_objectives), dtype=float)
        plus_np = np.asarray(jax.device_get(plus_objectives), dtype=float)
        er_fd_diagnostics = _root_only_er_fd_diagnostics(
            minus_rooted_state,
            baseline_rooted_state,
            plus_rooted_state,
            float(h),
        )
        report = {
            "mode": "transport_realtime_geometry_initial_er_root_only_forward_fd",
            "config_path": str(Path(args.config)),
            "parameter_name": parameter_name,
            "parameter_kind": parameter_kind,
            "baseline_value": float(baseline_value),
            "fd_step": float(h),
            "geometry_fd_lane": str(geometry_fd_lane),
            "root_fd_lane": str(root_fd_lane),
            "geometry_backend": str(config.get("geometry", {}).get("backend")),
            "vmec_lane": str(config.get("geometry", {}).get("vmec_lane", "forward")),
            "ntx_exact_surface_backend": str(
                config.get("neoclassical", {}).get("ntx_exact_surface_backend", "booz")
            ),
            "initial_er_root_ad": str(initial_er_root_ad),
            "objective_labels": list(objective_names),
            "objective_values": baseline_np.tolist(),
            "objective_minus": minus_np.tolist(),
            "objective_plus": plus_np.tolist(),
            "gradient_fd": gradient_np.tolist(),
            "Er_profile_fd_diagnostics": er_fd_diagnostics,
            "baseline_rooted_state_finite": _tree_all_finite(baseline_rooted_state),
            "minus_rooted_state_finite": _tree_all_finite(minus_rooted_state),
            "plus_rooted_state_finite": _tree_all_finite(plus_rooted_state),
        }
        print(
            "[autodiff-gate] mode=transport_realtime_geometry_initial_er_root_only_forward_fd "
            f"parameter={parameter_name} parameter_kind={parameter_kind} "
            f"baseline_value={baseline_value:.6e} fd_step={h:.6e} "
            f"geometry_fd_lane={geometry_fd_lane} root_fd_lane={root_fd_lane}",
            flush=True,
        )
        print("[autodiff-gate] initial-Er root-only objective values:")
        for label, value in zip(objective_names, baseline_np.tolist()):
            print(f"  - {label}: value={float(value):.16e}")
        print("[autodiff-gate] initial-Er root-only finite-difference gradients:")
        for label, value in zip(objective_names, gradient_np.tolist()):
            print(f"  - {label}: fd={float(value):.6e}")
        print(
            "[autodiff-gate] initial-Er root-only dEr/dr_param profile diagnostic: "
            f"min={er_fd_diagnostics['dEr_fd_min']:.6e} "
            f"max={er_fd_diagnostics['dEr_fd_max']:.6e} "
            f"max_abs={er_fd_diagnostics['dEr_fd_max_abs']:.6e}"
        )
        print("[autodiff-gate] initial-Er root-only largest |dEr/dr_param| radial indices:")
        for index, value in zip(
            er_fd_diagnostics["dEr_fd_top_abs_indices"],
            er_fd_diagnostics["dEr_fd_top_abs_values"],
        ):
            print(f"  - i={index}: dEr_fd={value:.6e}")
        outpath = _root_only_report_path(parameter_name, root_fd_lane=root_fd_lane)
        outpath.write_text(json.dumps(report, indent=2))
        print(f"Wrote {outpath.relative_to(ROOT)}")
        return

    print("[autodiff-gate] progress: running baseline realtime rollout for FD trace", flush=True)
    baseline_components, baseline_rollout, _baseline_final_state = _schedule_rollout(
        config,
        baseline_runtime,
        baseline_profile_state,
        accepted_step_limit=args.accepted_step_limit,
    )
    frozen_trace = _truncate_rollout_trace_by_accepted_steps(
        baseline_rollout.trace,
        args.accepted_step_limit,
    )
    baseline_objectives = _objective_vector(_baseline_final_state, baseline_runtime)
    objective_labels = tuple(OBJECTIVE_LABELS)
    if include_vmec_main_geometry_objectives:
        baseline_delta = float(config.get("geometry", {}).get("vmec_param_delta", 0.0))
        if geometry_fd_lane == "frozen_linearized":
            if frozen_linearized_bundle is None:
                raise ValueError("Missing frozen-linearized geometry bundle.")
            baseline_vmec_state = frozen_linearized_bundle["state_star"]
            baseline_geometry_context = frozen_linearized_bundle["context"]
        else:
            _runtime_unused, _state_unused, baseline_vmec_state, baseline_geometry_context = (
                _runtime_for_geometry_delta(
                    config,
                    parameter_name,
                    baseline_delta,
                    return_vmec_state=True,
                )
            )
        baseline_objectives = jnp.concatenate(
            [
                baseline_objectives,
                _geometry_full_objective_vector(
                    baseline_geometry_context, baseline_vmec_state
                ),
            ]
        )
        objective_labels = (
            *objective_labels,
            *geometry_observable_names_for_kind("geometry_full_ad_objectives"),
        )
    baseline_objectives = jax.block_until_ready(baseline_objectives)
    full_root_fd_lane = str(args.initial_er_root_fd_root_lane).strip().lower()
    if full_root_fd_lane == "frozen_linearized" and initial_er_root_ad == "off":
        raise SystemExit(
            "[autodiff-gate] --initial-Er-root-fd-root-lane frozen_linearized "
            "requires --initial-Er-root-ad jax_selected_root."
        )
    baseline_er_profile = None
    baseline_residual = None
    baseline_dres_der = None
    if full_root_fd_lane == "frozen_linearized":
        print(
            "[autodiff-gate] progress: preparing frozen-linearized initial-Er root branch "
            "for full-transport FD",
            flush=True,
        )
        baseline_pre_root_state = _profile_state_from_values(
            profile_values,
            initial_er_root_ad="off",
            runtime=baseline_runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
        )
        baseline_er_profile = jnp.asarray(baseline_profile_state.Er, dtype=baseline_pre_root_state.Er.dtype)
        baseline_residual = initial_er_charge_flux_residuals(
            baseline_pre_root_state,
            baseline_er_profile,
            runtime=baseline_runtime,
        )
        baseline_dres_der = initial_er_charge_flux_residual_er_derivative(
            baseline_pre_root_state,
            baseline_er_profile,
            runtime=baseline_runtime,
        )
        baseline_residual, baseline_dres_der = jax.block_until_ready((baseline_residual, baseline_dres_der))
    fixed_initial_er = (
        jax.lax.stop_gradient(baseline_state.Er)
        if (
            not parameter_is_profile
            and geometry_fd_lane == "frozen_linearized"
            and initial_er_root_ad == "off"
        )
        else None
    )

    if parameter_is_profile:
        parameter_kind = "profile"
        fixed_final_state_geometry_fd = None
        baseline_geometry_final_state_fd = None
        geometry_only_final_state_fd = None
        ntx_support_only_final_state_fd = None
        local_rhs_geometry_fd = None
        local_rhs_geometry_payload_fd = None
        param_index = PARAMETER_ORDER.index(parameter_name)
        baseline_value = _profile_cfg_scalar_value(profile_cfg, parameter_name)
        h = _fd_step(baseline_value, rel_step=args.fd_rel_step, abs_step=args.fd_abs_step)
        print("[autodiff-gate] progress: running profile fd_minus replay", flush=True)
        minus_profile_values = profile_values.at[param_index].set(
            jnp.asarray(baseline_value - h, dtype=profile_values.dtype)
        )
        minus_objectives, minus_replay = _objectives_on_realtime_geometry_frozen_trace(
            config=config,
            runtime=baseline_runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            profile_values=minus_profile_values,
            frozen_trace=frozen_trace,
            replay_mode=args.replay_mode,
            initial_er_root_ad=initial_er_root_ad,
            initial_er_root_fd_root_lane=full_root_fd_lane,
            baseline_er_profile=baseline_er_profile,
            baseline_residual=baseline_residual,
            baseline_dres_der=baseline_dres_der,
        )
        print("[autodiff-gate] progress: running profile fd_plus replay", flush=True)
        plus_profile_values = profile_values.at[param_index].set(
            jnp.asarray(baseline_value + h, dtype=profile_values.dtype)
        )
        plus_objectives, plus_replay = _objectives_on_realtime_geometry_frozen_trace(
            config=config,
            runtime=baseline_runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            profile_values=plus_profile_values,
            frozen_trace=frozen_trace,
            replay_mode=args.replay_mode,
            initial_er_root_ad=initial_er_root_ad,
            initial_er_root_fd_root_lane=full_root_fd_lane,
            baseline_er_profile=baseline_er_profile,
            baseline_residual=baseline_residual,
            baseline_dres_der=baseline_dres_der,
        )
        del param_index
    else:
        parameter_kind = "realtime_geometry"
        geom_cfg = config.get("geometry", {})
        baseline_value = float(geom_cfg.get("vmec_param_delta", 0.0))
        if geometry_fd_lane == "frozen_linearized":
            if frozen_linearized_bundle is None:
                raise ValueError("Missing frozen-linearized geometry bundle.")
            step_scale_value = float(frozen_linearized_bundle["coefficient_value"])
        else:
            geometry_context = _geometry_context_from_config(config, parameter_name)
            specs = _geometry_param_specs_from_parameter_name(parameter_name)
            (entry,) = boundary_param_entries(geometry_context, specs)
            step_scale_value = float(entry["baseline_coefficient"]) + baseline_value
        h = _fd_step(step_scale_value, rel_step=args.fd_rel_step, abs_step=args.fd_abs_step)
        print("[autodiff-gate] progress: running realtime geometry fd_minus replay", flush=True)
        minus_objectives, minus_replay = _geometry_fd_objectives(
            config=config,
            geometry_parameter=parameter_name,
            geometry_delta=baseline_value - h,
            profile_values=profile_values,
            profile_cfg=profile_cfg,
            frozen_trace=frozen_trace,
            replay_mode=args.replay_mode,
            geometry_fd_lane=geometry_fd_lane,
            frozen_linearized_bundle=frozen_linearized_bundle,
            fixed_initial_er=fixed_initial_er,
            initial_er_root_ad=initial_er_root_ad,
            initial_er_root_fd_root_lane=full_root_fd_lane,
            baseline_er_profile=baseline_er_profile,
            baseline_residual=baseline_residual,
            baseline_dres_der=baseline_dres_der,
            include_vmec_main_geometry_objectives=include_vmec_main_geometry_objectives,
        )
        print(
            "[autodiff-gate] progress: running fixed-final-state explicit geometry FD diagnostic",
            flush=True,
        )
        if geometry_fd_lane == "frozen_linearized":
            minus_runtime_fixed, _ = _runtime_for_frozen_linearized_geometry_step(
                config,
                frozen_linearized_bundle,
                step_scale=(baseline_value - h) - baseline_value,
                fixed_initial_er=fixed_initial_er,
            )
            plus_runtime_fixed, _ = _runtime_for_frozen_linearized_geometry_step(
                config,
                frozen_linearized_bundle,
                step_scale=(baseline_value + h) - baseline_value,
                fixed_initial_er=fixed_initial_er,
            )
        else:
            minus_runtime_fixed, _ = _runtime_for_geometry_delta(config, parameter_name, baseline_value - h)
            plus_runtime_fixed, _ = _runtime_for_geometry_delta(config, parameter_name, baseline_value + h)
        fixed_final_state_geometry_fd = (
            _objective_vector(_baseline_final_state, plus_runtime_fixed)
            - _objective_vector(_baseline_final_state, minus_runtime_fixed)
        ) / (2.0 * h)
        print("[autodiff-gate] progress: running realtime geometry fd_plus replay", flush=True)
        plus_objectives, plus_replay = _geometry_fd_objectives(
            config=config,
            geometry_parameter=parameter_name,
            geometry_delta=baseline_value + h,
            profile_values=profile_values,
            profile_cfg=profile_cfg,
            frozen_trace=frozen_trace,
            replay_mode=args.replay_mode,
            geometry_fd_lane=geometry_fd_lane,
            frozen_linearized_bundle=frozen_linearized_bundle,
            fixed_initial_er=fixed_initial_er,
            initial_er_root_ad=initial_er_root_ad,
            initial_er_root_fd_root_lane=full_root_fd_lane,
            baseline_er_profile=baseline_er_profile,
            baseline_residual=baseline_residual,
            baseline_dres_der=baseline_dres_der,
            include_vmec_main_geometry_objectives=include_vmec_main_geometry_objectives,
        )
        baseline_geometry_final_state_fd = (
            _objective_vector(plus_replay["final_state"], baseline_runtime)
            - _objective_vector(minus_replay["final_state"], baseline_runtime)
        ) / (2.0 * h)
        geometry_only_final_state_fd = None
        ntx_support_only_final_state_fd = None
        local_rhs_geometry_fd = None
        local_rhs_geometry_payload_fd = None
        if args.split_payload_fd_diagnostic:
            print(
                "[autodiff-gate] progress: running split payload final-state FD diagnostic",
                flush=True,
            )
            baseline_support = _find_ntx_support_payload(baseline_runtime)
            minus_support = _find_ntx_support_payload(minus_runtime_fixed)
            plus_support = _find_ntx_support_payload(plus_runtime_fixed)
            minus_geometry_only_runtime = _runtime_with_ntx_support_payload(
                minus_runtime_fixed,
                baseline_support,
            )
            plus_geometry_only_runtime = _runtime_with_ntx_support_payload(
                plus_runtime_fixed,
                baseline_support,
            )
            minus_support_only_runtime = _runtime_with_ntx_support_payload(
                baseline_runtime,
                minus_support,
            )
            plus_support_only_runtime = _runtime_with_ntx_support_payload(
                baseline_runtime,
                plus_support,
            )
            _minus_geometry_only_objectives, minus_geometry_only_replay = _objectives_on_realtime_geometry_frozen_trace(
                config=config,
                runtime=minus_geometry_only_runtime,
                baseline_state=baseline_state,
                profile_cfg=profile_cfg,
                profile_values=profile_values,
                frozen_trace=frozen_trace,
                replay_mode=args.replay_mode,
                initial_er_root_ad=initial_er_root_ad,
                initial_er_root_fd_root_lane=full_root_fd_lane,
                baseline_er_profile=baseline_er_profile,
                baseline_residual=baseline_residual,
                baseline_dres_der=baseline_dres_der,
            )
            _plus_geometry_only_objectives, plus_geometry_only_replay = _objectives_on_realtime_geometry_frozen_trace(
                config=config,
                runtime=plus_geometry_only_runtime,
                baseline_state=baseline_state,
                profile_cfg=profile_cfg,
                profile_values=profile_values,
                frozen_trace=frozen_trace,
                replay_mode=args.replay_mode,
                initial_er_root_ad=initial_er_root_ad,
                initial_er_root_fd_root_lane=full_root_fd_lane,
                baseline_er_profile=baseline_er_profile,
                baseline_residual=baseline_residual,
                baseline_dres_der=baseline_dres_der,
            )
            _minus_support_only_objectives, minus_support_only_replay = _objectives_on_realtime_geometry_frozen_trace(
                config=config,
                runtime=minus_support_only_runtime,
                baseline_state=baseline_state,
                profile_cfg=profile_cfg,
                profile_values=profile_values,
                frozen_trace=frozen_trace,
                replay_mode=args.replay_mode,
                initial_er_root_ad=initial_er_root_ad,
                initial_er_root_fd_root_lane=full_root_fd_lane,
                baseline_er_profile=baseline_er_profile,
                baseline_residual=baseline_residual,
                baseline_dres_der=baseline_dres_der,
            )
            _plus_support_only_objectives, plus_support_only_replay = _objectives_on_realtime_geometry_frozen_trace(
                config=config,
                runtime=plus_support_only_runtime,
                baseline_state=baseline_state,
                profile_cfg=profile_cfg,
                profile_values=profile_values,
                frozen_trace=frozen_trace,
                replay_mode=args.replay_mode,
                initial_er_root_ad=initial_er_root_ad,
                initial_er_root_fd_root_lane=full_root_fd_lane,
                baseline_er_profile=baseline_er_profile,
                baseline_residual=baseline_residual,
                baseline_dres_der=baseline_dres_der,
            )
            geometry_only_final_state_fd = (
                _objective_vector(plus_geometry_only_replay["final_state"], baseline_runtime)
                - _objective_vector(minus_geometry_only_replay["final_state"], baseline_runtime)
            ) / (2.0 * h)
            ntx_support_only_final_state_fd = (
                _objective_vector(plus_support_only_replay["final_state"], baseline_runtime)
                - _objective_vector(minus_support_only_replay["final_state"], baseline_runtime)
            ) / (2.0 * h)
            print(
                "[autodiff-gate] progress: running local RHS geometry-payload FD diagnostic",
                flush=True,
            )
            baseline_equation_system = baseline_components["equation_system"]
            baseline_solver = baseline_components["solver"]
            t0 = jnp.asarray(getattr(baseline_solver, "t0", 0.0), dtype=jnp.float64)
            baseline_lagged_response = baseline_equation_system.build_lagged_response(
                baseline_profile_state
            )

            def _rhs_component_sums(rhs_value):
                return jnp.asarray(
                    [
                        jnp.sum(jnp.asarray(rhs_value.density, dtype=jnp.float64)),
                        jnp.sum(jnp.asarray(rhs_value.pressure, dtype=jnp.float64)),
                        jnp.sum(jnp.asarray(rhs_value.Er, dtype=jnp.float64)),
                    ],
                    dtype=jnp.float64,
                )

            def _local_prepared_rhs_sums(runtime_value):
                components_value = prepare_transport_solver_components(
                    config,
                    runtime_value,
                    baseline_profile_state,
                )
                rhs_value = components_value["equation_system"].evaluate_with_lagged_response(
                    t0,
                    baseline_profile_state,
                    runtime_value.species,
                    baseline_lagged_response,
                )
                return _rhs_component_sums(rhs_value)

            def _local_payload_rhs_sums(runtime_value):
                rhs_value = baseline_equation_system.with_geometry_payload(
                    runtime_value.geometry
                ).evaluate_with_lagged_response(
                    t0,
                    baseline_profile_state,
                    baseline_runtime.species,
                    baseline_lagged_response,
                )
                return _rhs_component_sums(rhs_value)

            local_rhs_geometry_fd = (
                _local_prepared_rhs_sums(plus_geometry_only_runtime)
                - _local_prepared_rhs_sums(minus_geometry_only_runtime)
            ) / (2.0 * h)
            local_rhs_geometry_payload_fd = (
                _local_payload_rhs_sums(plus_geometry_only_runtime)
                - _local_payload_rhs_sums(minus_geometry_only_runtime)
            ) / (2.0 * h)

    (
        minus_objectives,
        plus_objectives,
        fixed_final_state_geometry_fd,
        baseline_geometry_final_state_fd,
        geometry_only_final_state_fd,
        ntx_support_only_final_state_fd,
        local_rhs_geometry_fd,
        local_rhs_geometry_payload_fd,
    ) = (
        jax.block_until_ready(
            (
                minus_objectives,
                plus_objectives,
                fixed_final_state_geometry_fd,
                baseline_geometry_final_state_fd,
                geometry_only_final_state_fd,
                ntx_support_only_final_state_fd,
                local_rhs_geometry_fd,
                local_rhs_geometry_payload_fd,
            )
        )
    )
    gradient_fd = (plus_objectives - minus_objectives) / (2.0 * h)
    gradient_fd = jax.block_until_ready(gradient_fd)
    gradient_np = np.asarray(jax.device_get(gradient_fd), dtype=float)
    baseline_np = np.asarray(jax.device_get(baseline_objectives), dtype=float)
    fixed_final_state_geometry_fd_np = (
        None
        if fixed_final_state_geometry_fd is None
        else np.asarray(jax.device_get(fixed_final_state_geometry_fd), dtype=float)
    )
    baseline_geometry_final_state_fd_np = (
        None
        if baseline_geometry_final_state_fd is None
        else np.asarray(jax.device_get(baseline_geometry_final_state_fd), dtype=float)
    )
    geometry_only_final_state_fd_np = (
        None
        if geometry_only_final_state_fd is None
        else np.asarray(jax.device_get(geometry_only_final_state_fd), dtype=float)
    )
    ntx_support_only_final_state_fd_np = (
        None
        if ntx_support_only_final_state_fd is None
        else np.asarray(jax.device_get(ntx_support_only_final_state_fd), dtype=float)
    )
    local_rhs_geometry_fd_np = (
        None
        if local_rhs_geometry_fd is None
        else np.asarray(jax.device_get(local_rhs_geometry_fd), dtype=float)
    )
    local_rhs_geometry_payload_fd_np = (
        None
        if local_rhs_geometry_payload_fd is None
        else np.asarray(jax.device_get(local_rhs_geometry_payload_fd), dtype=float)
    )

    report = {
        "mode": "transport_realtime_geometry_forward_fd",
        "config_path": str(Path(args.config)),
        "parameter_name": parameter_name,
        "parameter_kind": parameter_kind,
        "baseline_value": float(baseline_value),
        "fd_step": float(h),
        "replay_mode": str(args.replay_mode),
        "geometry_fd_lane": str(geometry_fd_lane),
        "initial_er_root_ad": str(initial_er_root_ad),
        "initial_er_root_fd_root_lane": str(full_root_fd_lane),
        "accepted_step_limit": None if args.accepted_step_limit is None else int(args.accepted_step_limit),
        "radau_jacobian_reuse_mode": None
        if args.radau_jacobian_reuse_mode is None
        else str(args.radau_jacobian_reuse_mode),
        "geometry_backend": str(config.get("geometry", {}).get("backend")),
        "vmec_lane": str(config.get("geometry", {}).get("vmec_lane", "forward")),
        "ntx_exact_surface_backend": str(
            config.get("neoclassical", {}).get("ntx_exact_surface_backend", "booz")
        ),
        "objective_labels": list(objective_labels),
        "transport_objective_labels": list(OBJECTIVE_LABELS),
        "objective_values": baseline_np.tolist(),
        "gradient_fd": gradient_np.tolist(),
        "fixed_final_state_objective_geometry_fd": None
        if fixed_final_state_geometry_fd_np is None
        else fixed_final_state_geometry_fd_np.tolist(),
        "baseline_geometry_final_state_fd": None
        if baseline_geometry_final_state_fd_np is None
        else baseline_geometry_final_state_fd_np.tolist(),
        "geometry_only_final_state_fd": None
        if geometry_only_final_state_fd_np is None
        else geometry_only_final_state_fd_np.tolist(),
        "ntx_support_only_final_state_fd": None
        if ntx_support_only_final_state_fd_np is None
        else ntx_support_only_final_state_fd_np.tolist(),
        "local_rhs_geometry_fd_component_sums": None
        if local_rhs_geometry_fd_np is None
        else {
            name: float(value)
            for name, value in zip(
                ("density", "pressure", "Er"),
                local_rhs_geometry_fd_np.tolist(),
            )
        },
        "local_rhs_geometry_payload_fd_component_sums": None
        if local_rhs_geometry_payload_fd_np is None
        else {
            name: float(value)
            for name, value in zip(
                ("density", "pressure", "Er"),
                local_rhs_geometry_payload_fd_np.tolist(),
            )
        },
        "baseline_rollout": _adaptive_rollout_diagnostics(baseline_rollout),
        "minus_replay": {
            "final_state_finite": _tree_all_finite(minus_replay["final_state"]),
            "final_carry_finite": _tree_all_finite(minus_replay["final_carry"]),
        },
        "plus_replay": {
            "final_state_finite": _tree_all_finite(plus_replay["final_state"]),
            "final_carry_finite": _tree_all_finite(plus_replay["final_carry"]),
        },
    }
    print(
        "[autodiff-gate] mode=transport_realtime_geometry_forward_fd "
        f"parameter={parameter_name} parameter_kind={parameter_kind} "
        f"baseline_value={baseline_value:.6e} fd_step={h:.6e} "
        f"replay_mode={args.replay_mode} geometry_fd_lane={geometry_fd_lane} "
        f"root_fd_lane={full_root_fd_lane}"
    )
    print("[autodiff-gate] objective values:")
    for label, value in zip(objective_labels, baseline_np.tolist()):
        print(f"  - {label}: value={float(value):.16e}")
    print("[autodiff-gate] objective finite-difference gradients:")
    for label, value in zip(objective_labels, gradient_np.tolist()):
        print(f"  - {label}: fd={float(value):.6e}")
    if fixed_final_state_geometry_fd_np is not None:
        print("[autodiff-gate] fixed-final-state explicit geometry finite-difference gradients:")
        for label, value in zip(OBJECTIVE_LABELS, fixed_final_state_geometry_fd_np.tolist()):
            print(f"  - {label}: fd_explicit_geometry={float(value):.6e}")
    if baseline_geometry_final_state_fd_np is not None:
        print("[autodiff-gate] baseline-geometry final-state finite-difference gradients:")
        for label, value in zip(OBJECTIVE_LABELS, baseline_geometry_final_state_fd_np.tolist()):
            print(f"  - {label}: fd_final_state_geometry={float(value):.6e}")
    if geometry_only_final_state_fd_np is not None:
        print("[autodiff-gate] geometry-only final-state finite-difference gradients:")
        for label, value in zip(OBJECTIVE_LABELS, geometry_only_final_state_fd_np.tolist()):
            print(f"  - {label}: fd_final_state_geometry_branch={float(value):.6e}")
    if ntx_support_only_final_state_fd_np is not None:
        print("[autodiff-gate] NTX-support-only final-state finite-difference gradients:")
        for label, value in zip(OBJECTIVE_LABELS, ntx_support_only_final_state_fd_np.tolist()):
            print(f"  - {label}: fd_final_state_ntx_support_branch={float(value):.6e}")
    if local_rhs_geometry_fd_np is not None and local_rhs_geometry_payload_fd_np is not None:
        print("[autodiff-gate] local RHS geometry finite-difference component sums:")
        for label, prepared_value, payload_value in zip(
            ("density", "pressure", "Er"),
            local_rhs_geometry_fd_np.tolist(),
            local_rhs_geometry_payload_fd_np.tolist(),
        ):
            print(
                f"  - {label}: "
                f"fd_prepared_geometry={float(prepared_value):.6e} "
                f"fd_with_geometry_payload={float(payload_value):.6e} "
                f"diff={float(payload_value - prepared_value):.6e}"
            )

    outpath = _report_path(parameter_name, root_fd_lane=full_root_fd_lane)
    outpath.write_text(json.dumps(report, indent=2))
    print(f"Wrote {outpath.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
