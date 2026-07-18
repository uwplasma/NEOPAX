from __future__ import annotations

import argparse
import dataclasses
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
    _geometry_context_from_config,
    _geometry_param_specs_from_parameter_name,
    _initial_state_for_parameter_vector,
)
from NEOPAX._geometry_autodiff import (  # noqa: E402
    _implicit_params_with_boundary_deltas,
    boundary_param_entries,
    build_runtime_context_for_geometry_param,
    build_runtime_context_for_vmec_state,
)
from NEOPAX._orchestrator import build_runtime_context, prepare_transport_solver_components  # noqa: E402
from NEOPAX._transport_solvers import (  # noqa: E402
    _build_prepared_radau_accepted_rollout,
    _build_prepared_radau_execution_context,
    _radau_adaptive_schedule_rollout,
    _radau_run_prepared_on_realized_trace,
)
from vmec_jax.core import implicit as im  # noqa: E402


def _report_path(parameter_name: str) -> Path:
    safe_name = parameter_name.replace(":", "_")
    outdir = ROOT / "outputs" / "autodiff_transport_lagged_ntx" / "realtime_geometry_fd"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / f"{safe_name}_forward_fd_summary.json"


def _tree_all_finite(tree) -> bool:
    for leaf in jax.tree_util.tree_leaves(tree):
        arr = np.asarray(jax.device_get(leaf))
        if np.issubdtype(arr.dtype, np.inexact) and not np.all(np.isfinite(arr)):
            return False
    return True


def _profile_state_from_values(values, *, runtime, baseline_state, profile_cfg: dict[str, Any]):
    return _initial_state_for_parameter_vector(
        values,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        runtime=runtime,
    )


def _prepare_rollout(config: dict[str, Any], runtime, state0, *, solver_override=None):
    components = prepare_transport_solver_components(
        config,
        runtime,
        state0,
        solver_override=solver_override,
    )
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
):
    state0 = _profile_state_from_values(
        profile_values,
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
    replay = _radau_run_prepared_on_realized_trace(
        prepared_rollout,
        execution_context,
        frozen_trace,
        replay_mode=str(replay_mode).strip().lower(),
        carry0=prepared_rollout.initial_carry,
    )
    return _objective_vector(replay["final_state"], runtime), replay


def _runtime_for_geometry_delta(config: dict[str, Any], geometry_parameter: str, delta_value):
    geom_cfg = config.get("geometry", {})
    geometry_context = _geometry_context_from_config(config, geometry_parameter)
    specs = _geometry_param_specs_from_parameter_name(geometry_parameter)
    if len(specs) != 1:
        raise ValueError("This FD benchmark currently expects one geometry parameter.")
    return build_runtime_context_for_geometry_param(
        config,
        geometry_context,
        jnp.asarray(delta_value, dtype=jnp.float64),
        lane=str(geom_cfg.get("vmec_lane", "forward")).strip().lower(),
        n_r=int(geom_cfg.get("n_radial", 51)),
        max_iter=geom_cfg.get("vmec_max_iter"),
        step_size=geom_cfg.get("vmec_step_size"),
        jacobian_penalty=float(geom_cfg.get("vmec_jacobian_penalty", 1.0e3)),
    )


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
    params0 = im.params_from_input(geometry_context.indata)
    params = _implicit_params_with_boundary_deltas(
        geometry_context,
        im,
        jnp.asarray([baseline_delta], dtype=jnp.float64),
        (entry,),
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
    else:
        runtime, baseline_state = _runtime_for_geometry_delta(config, geometry_parameter, geometry_delta)
    return _objectives_on_realtime_geometry_frozen_trace(
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        profile_values=profile_values,
        frozen_trace=frozen_trace,
        replay_mode=replay_mode,
    )


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
    parser.add_argument("--device", type=str, default=None)
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
        default="attempt",
        help="Replay the baseline realized attempt trace or only accepted times.",
    )
    parser.add_argument("--radau-jacobian-reuse-mode", default=None)
    args = parser.parse_args()

    config = _prepare_benchmark_config(
        config_path=Path(args.config),
        device=args.device,
        ntx_exact_derivative_mode="direct",
        radau_jacobian_reuse_mode=args.radau_jacobian_reuse_mode,
    )
    profile_cfg = _baseline_profile_cfg(config)
    parameter_name = str(args.parameter)
    geometry_fd_lane = str(args.geometry_fd_lane).strip().lower()
    parameter_is_profile = parameter_name in PARAMETER_ORDER
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
    profile_values = jnp.asarray([float(profile_cfg[name]) for name in PARAMETER_ORDER], dtype=jnp.float64)
    baseline_profile_state = _profile_state_from_values(
        profile_values,
        runtime=baseline_runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
    )

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
    baseline_objectives = jax.block_until_ready(baseline_objectives)
    fixed_initial_er = (
        jax.lax.stop_gradient(baseline_state.Er)
        if (not parameter_is_profile and geometry_fd_lane == "frozen_linearized")
        else None
    )

    if parameter_is_profile:
        parameter_kind = "profile"
        param_index = PARAMETER_ORDER.index(parameter_name)
        baseline_value = float(profile_cfg[parameter_name])
        h = _fd_step(baseline_value, rel_step=args.fd_rel_step, abs_step=args.fd_abs_step)
        print("[autodiff-gate] progress: running profile fd_minus replay", flush=True)
        minus_objectives, minus_replay = _forward_benchmark_adaptive_rollout_objectives_for_parameter_on_frozen_trace(
            jnp.asarray(baseline_value - h, dtype=jnp.float64),
            config=config,
            runtime=baseline_runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=parameter_name,
            frozen_trace=frozen_trace,
            replay_mode=args.replay_mode,
            use_ad_lane=False,
        )
        print("[autodiff-gate] progress: running profile fd_plus replay", flush=True)
        plus_objectives, plus_replay = _forward_benchmark_adaptive_rollout_objectives_for_parameter_on_frozen_trace(
            jnp.asarray(baseline_value + h, dtype=jnp.float64),
            config=config,
            runtime=baseline_runtime,
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            parameter_name=parameter_name,
            frozen_trace=frozen_trace,
            replay_mode=args.replay_mode,
            use_ad_lane=False,
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
        )
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
        )

    minus_objectives, plus_objectives = jax.block_until_ready((minus_objectives, plus_objectives))
    gradient_fd = (plus_objectives - minus_objectives) / (2.0 * h)
    gradient_fd = jax.block_until_ready(gradient_fd)
    gradient_np = np.asarray(jax.device_get(gradient_fd), dtype=float)
    baseline_np = np.asarray(jax.device_get(baseline_objectives), dtype=float)

    report = {
        "mode": "transport_realtime_geometry_forward_fd",
        "config_path": str(Path(args.config)),
        "parameter_name": parameter_name,
        "parameter_kind": parameter_kind,
        "baseline_value": float(baseline_value),
        "fd_step": float(h),
        "replay_mode": str(args.replay_mode),
        "geometry_fd_lane": str(geometry_fd_lane),
        "accepted_step_limit": None if args.accepted_step_limit is None else int(args.accepted_step_limit),
        "radau_jacobian_reuse_mode": None
        if args.radau_jacobian_reuse_mode is None
        else str(args.radau_jacobian_reuse_mode),
        "geometry_backend": str(config.get("geometry", {}).get("backend")),
        "vmec_lane": str(config.get("geometry", {}).get("vmec_lane", "forward")),
        "ntx_exact_surface_backend": str(
            config.get("neoclassical", {}).get("ntx_exact_surface_backend", "booz")
        ),
        "objective_labels": list(OBJECTIVE_LABELS),
        "objective_values": baseline_np.tolist(),
        "gradient_fd": gradient_np.tolist(),
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
        f"replay_mode={args.replay_mode} geometry_fd_lane={geometry_fd_lane}"
    )
    print("[autodiff-gate] objective values:")
    for label, value in zip(OBJECTIVE_LABELS, baseline_np.tolist()):
        print(f"  - {label}: value={float(value):.16e}")
    print("[autodiff-gate] objective finite-difference gradients:")
    for label, value in zip(OBJECTIVE_LABELS, gradient_np.tolist()):
        print(f"  - {label}: fd={float(value):.6e}")

    outpath = _report_path(parameter_name)
    outpath.write_text(json.dumps(report, indent=2))
    print(f"Wrote {outpath.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
