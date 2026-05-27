from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from NEOPAX._geometry_autodiff import (
    build_geometry_autodiff_context,
    build_runtime_context_for_geometry_param,
    rel_error,
)
from NEOPAX._orchestrator import prepare_transport_solver_components
from NEOPAX._transport_solvers import (
    _build_prepared_radau_accepted_rollout,
    _build_prepared_radau_execution_context,
    _make_solver_state_transform,
    _radau_adaptive_final_state_rollout,
    _radau_run_prepared_on_realized_trace,
)
from benchmark_transport_autodiff_lagged_ntx import (
    OBJECTIVE_LABELS,
    _adaptive_rollout_diagnostics,
    _objective_vector,
    _prepare_benchmark_config,
    _truncate_rollout_trace_by_accepted_steps,
)


def _parse_surface_s(text: str) -> tuple[float, ...]:
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("At least one Boozer surface must be provided.")
    return tuple(values)


def _fd_step(base_value: float, *, fd_rel_step: float, fd_abs_step: float) -> float:
    return max(abs(float(base_value)) * float(fd_rel_step), float(fd_abs_step))


def _build_geometry_runtime_and_rollout(
    *,
    config: dict,
    context,
    delta,
    n_radial: int,
    vmec_max_iter: int,
    vmec_step_size: float,
    vmec_jacobian_penalty: float,
    checkpoint_index: int,
):
    runtime, state0 = build_runtime_context_for_geometry_param(
        config,
        context,
        delta,
        n_r=n_radial,
        max_iter=vmec_max_iter,
        step_size=vmec_step_size,
        jacobian_penalty=vmec_jacobian_penalty,
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
    max_total_steps = int(max(1, getattr(solver, "max_steps", 1)))
    rollout = _radau_adaptive_final_state_rollout(
        execution_context,
        prepared_rollout.initial_carry,
        max_total_steps=max_total_steps,
        stop_after_accepted_steps=int(checkpoint_index),
    )
    final_state = prepared_rollout.physics_context.unpack_flat(rollout.final_carry.y)
    return runtime, state0, prepared_rollout, execution_context, rollout, final_state


def _replay_objective_and_state(
    delta,
    *,
    config: dict,
    context,
    replay_trace,
    replay_mode: str,
    n_radial: int,
    vmec_max_iter: int,
    vmec_step_size: float,
    vmec_jacobian_penalty: float,
    pack_state,
):
    runtime, state0 = build_runtime_context_for_geometry_param(
        config,
        context,
        delta,
        n_r=n_radial,
        max_iter=vmec_max_iter,
        step_size=vmec_step_size,
        jacobian_penalty=vmec_jacobian_penalty,
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
    replay = _radau_run_prepared_on_realized_trace(
        prepared_rollout,
        execution_context,
        replay_trace,
        replay_mode=replay_mode,
    )
    final_state = replay["final_state"]
    return _objective_vector(final_state, runtime), pack_state(final_state)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen realized-trace AD-vs-FD transport benchmark for VMEC geometry parameters."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path("NEOPAX/examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_benchmark.toml")),
        help="NEOPAX transport config; intended for the NTX exact lagged runtime path.",
    )
    parser.add_argument(
        "--vmec-input",
        type=str,
        default=str(Path("NEOPAX/examples/inputs/input.QI_nfp2_newNT_opt_hires")),
        help="VMEC input file used to rebuild geometry.",
    )
    parser.add_argument("--param-family", type=str, default="RBC", choices=("RBC", "ZBS"))
    parser.add_argument("--param-m", type=int, default=1)
    parser.add_argument("--param-n", type=int, default=0)
    parser.add_argument("--surface-s", type=str, default="0.25,0.5,0.75")
    parser.add_argument("--mboz", type=int, default=12)
    parser.add_argument("--nboz", type=int, default=12)
    parser.add_argument("--n-radial", type=int, default=51)
    parser.add_argument("--vmec-max-iter", type=int, default=2)
    parser.add_argument("--vmec-step-size", type=float, default=5.0e-3)
    parser.add_argument("--vmec-jacobian-penalty", type=float, default=1.0e3)
    parser.add_argument("--checkpoint-index", type=int, default=115)
    parser.add_argument("--fd-rel-step", type=float, default=1.0e-6)
    parser.add_argument("--fd-abs-step", type=float, default=1.0e-8)
    parser.add_argument("--replay-mode", type=str, default="attempt", choices=("attempt", "accepted"))
    parser.add_argument("--with-five-point", action="store_true")
    args = parser.parse_args()

    config = _prepare_benchmark_config(Path(args.config), device=None)
    surface_s = _parse_surface_s(args.surface_s)
    context = build_geometry_autodiff_context(
        args.vmec_input,
        param_family=args.param_family,
        param_m=args.param_m,
        param_n=args.param_n,
        mboz=args.mboz,
        nboz=args.nboz,
        surface_s=surface_s,
    )
    fd_step = _fd_step(context.baseline_coefficient, fd_rel_step=args.fd_rel_step, fd_abs_step=args.fd_abs_step)
    minus_value = -fd_step
    plus_value = fd_step
    minus2_value = -2.0 * fd_step
    plus2_value = 2.0 * fd_step

    print(
        "[autodiff-gate] geometry-frozen baseline setup: "
        f"family={context.param_family} m={context.param_m} n={context.param_n} "
        f"baseline_coefficient={context.baseline_coefficient:.6e} fd_step={fd_step:.6e}",
        flush=True,
    )

    baseline_runtime, baseline_state, baseline_prepared_rollout, _baseline_execution_context, baseline_rollout, _baseline_final_state = _build_geometry_runtime_and_rollout(
        config=config,
        context=context,
        delta=jnp.asarray(0.0, dtype=jnp.float64),
        n_radial=args.n_radial,
        vmec_max_iter=args.vmec_max_iter,
        vmec_step_size=args.vmec_step_size,
        vmec_jacobian_penalty=args.vmec_jacobian_penalty,
        checkpoint_index=args.checkpoint_index,
    )
    baseline_diag = _adaptive_rollout_diagnostics(baseline_rollout)
    accepted_mask_np = np.asarray(jax.device_get(baseline_rollout.trace.accepted_mask), dtype=bool)
    accepted_attempt_indices = np.flatnonzero(accepted_mask_np)
    if accepted_attempt_indices.size < int(args.checkpoint_index):
        raise ValueError(
            f"Need at least {args.checkpoint_index} accepted attempts for frozen geometry comparison; "
            f"found {accepted_attempt_indices.size}."
        )
    accepted_attempt_indices = [int(v) for v in accepted_attempt_indices[: int(args.checkpoint_index)]]
    replay_trace = _truncate_rollout_trace_by_accepted_steps(
        baseline_rollout.trace,
        accepted_step_limit=int(args.checkpoint_index),
    )
    step_ts = np.asarray(jax.device_get(baseline_rollout.trace.step_ts), dtype=float)

    print(
        "[autodiff-gate] geometry-frozen baseline summary: "
        f"attempt_count={baseline_diag['attempt_count']} "
        f"accepted_count={baseline_diag['accepted_count']} "
        f"checkpoint_index={int(args.checkpoint_index)} "
        f"checkpoint_attempt={accepted_attempt_indices[-1]}",
        flush=True,
    )

    _flat_state0, _unpack_flat_tmp, _unpack_packed_tmp, pack_state, _project_flat_tmp = _make_solver_state_transform(
        baseline_state,
        baseline_runtime.species,
    )

    objective_and_state_fn = lambda delta: _replay_objective_and_state(  # noqa: E731
        delta,
        config=config,
        context=context,
        replay_trace=replay_trace,
        replay_mode=args.replay_mode,
        n_radial=args.n_radial,
        vmec_max_iter=args.vmec_max_iter,
        vmec_step_size=args.vmec_step_size,
        vmec_jacobian_penalty=args.vmec_jacobian_penalty,
        pack_state=pack_state,
    )

    print("[autodiff-gate] geometry-frozen progress: running custom AD", flush=True)
    (_, _baseline_state_flat), (objective_ad, state_ad) = jax.jvp(
        objective_and_state_fn,
        (jnp.asarray(0.0, dtype=jnp.float64),),
        (jnp.asarray(1.0, dtype=jnp.float64),),
    )

    print("[autodiff-gate] geometry-frozen progress: running fd_minus replay", flush=True)
    objectives_minus, state_minus = objective_and_state_fn(jnp.asarray(minus_value, dtype=jnp.float64))
    print("[autodiff-gate] geometry-frozen progress: running fd_plus replay", flush=True)
    objectives_plus, state_plus = objective_and_state_fn(jnp.asarray(plus_value, dtype=jnp.float64))

    gradient_fd = (objectives_plus - objectives_minus) / (2.0 * fd_step)
    state_fd = (state_plus - state_minus) / (2.0 * fd_step)

    grad_ad_np = np.asarray(jax.device_get(objective_ad), dtype=float)
    grad_fd_np = np.asarray(jax.device_get(gradient_fd), dtype=float)
    state_ad_np = np.asarray(jax.device_get(state_ad), dtype=float)
    state_fd_np = np.asarray(jax.device_get(state_fd), dtype=float)

    grad_fd_five_point_np = None
    state_fd_five_point_np = None
    if args.with_five_point:
        print("[autodiff-gate] geometry-frozen progress: running fd_minus2 replay", flush=True)
        objectives_minus2, state_minus2 = objective_and_state_fn(jnp.asarray(minus2_value, dtype=jnp.float64))
        print("[autodiff-gate] geometry-frozen progress: running fd_plus2 replay", flush=True)
        objectives_plus2, state_plus2 = objective_and_state_fn(jnp.asarray(plus2_value, dtype=jnp.float64))
        gradient_fd_five_point = (-objectives_plus2 + 8.0 * objectives_plus - 8.0 * objectives_minus + objectives_minus2) / (12.0 * fd_step)
        state_fd_five_point = (-state_plus2 + 8.0 * state_plus - 8.0 * state_minus + state_minus2) / (12.0 * fd_step)
        grad_fd_five_point_np = np.asarray(jax.device_get(gradient_fd_five_point), dtype=float)
        state_fd_five_point_np = np.asarray(jax.device_get(state_fd_five_point), dtype=float)

    print(
        "[autodiff-gate] mode=geometry_realized_trace_checkpoint_frozen_fd "
        f"parameter={context.param_family}({context.param_m},{context.param_n}) "
        f"baseline_value={context.baseline_coefficient:.6e} fd_step={fd_step:.6e} "
        f"checkpoint_index={int(args.checkpoint_index)} checkpoint_time={float(step_ts[accepted_attempt_indices[-1]]):.6e}"
    )
    print(
        "[autodiff-gate] rollout baseline: "
        f"attempt_count={baseline_diag['attempt_count']} accepted_count={baseline_diag['accepted_count']} "
        f"completed={baseline_diag['completed']} failed={baseline_diag['failed']} fail_code={baseline_diag['fail_code']}"
    )

    print("[autodiff-gate] objective errors:")
    for idx, label in enumerate(OBJECTIVE_LABELS):
        rel_center = abs(grad_ad_np[idx] - grad_fd_np[idx]) / max(abs(grad_fd_np[idx]), 1.0e-10)
        line = (
            f"  - {label}: custom_ad={grad_ad_np[idx]:.6e} fd={grad_fd_np[idx]:.6e} "
            f"custom_vs_fd_rel_err={rel_center:.6e}"
        )
        if grad_fd_five_point_np is not None:
            rel_five = abs(grad_ad_np[idx] - grad_fd_five_point_np[idx]) / max(abs(grad_fd_five_point_np[idx]), 1.0e-10)
            center_vs_five = abs(grad_fd_np[idx] - grad_fd_five_point_np[idx]) / max(abs(grad_fd_five_point_np[idx]), 1.0e-10)
            line += (
                f" fd_five_point={grad_fd_five_point_np[idx]:.6e}"
                f" custom_vs_fd_five_point_rel_err={rel_five:.6e}"
                f" center_vs_fd_five_point_rel_err={center_vs_five:.6e}"
            )
        print(line)

    full_rel = float(np.linalg.norm(state_ad_np - state_fd_np) / max(float(np.linalg.norm(state_fd_np)), 1.0e-10))
    print(f"[autodiff-gate] state tangent errors:\n  - custom_vs_fd: full_rel_err={full_rel:.6e}")
    if state_fd_five_point_np is not None:
        full_rel_five = float(
            np.linalg.norm(state_ad_np - state_fd_five_point_np)
            / max(float(np.linalg.norm(state_fd_five_point_np)), 1.0e-10)
        )
        center_vs_five = float(
            np.linalg.norm(state_fd_np - state_fd_five_point_np)
            / max(float(np.linalg.norm(state_fd_five_point_np)), 1.0e-10)
        )
        print(
            "[autodiff-gate] state tangent errors (five-point):\n"
            f"  - custom_vs_fd_five_point: full_rel_err={full_rel_five:.6e}\n"
            f"  - center_vs_fd_five_point: full_rel_err={center_vs_five:.6e}"
        )


if __name__ == "__main__":
    main()
