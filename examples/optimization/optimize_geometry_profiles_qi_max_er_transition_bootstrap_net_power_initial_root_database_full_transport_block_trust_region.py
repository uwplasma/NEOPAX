#!/usr/bin/env python
"""One-run geometry/profile optimization with physical block trust limits.

This experiment starts geometry and profile degrees of freedom together at
the original seed. Geometry uses the established ESS coordinates. Profiles
use centered nominal deltas, so all optimizer coordinates start at zero and a
profile step of 0.1 is a ten-percent change from the seed profile. One joint
Gauss--Newton model is constrained by independent geometry and profile trust
limits, and every step is accepted or rejected using the actual nonlinear
VMEC/NTX/full-transport cost.

The established geometry-only and ordinary combined optimization scripts are
not called or modified by this example.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import jax
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Preserve the validated VMEX-first initialization order.
import vmex as vj  # noqa: E402,F401
from vmex import optimize as vmex_opt  # noqa: E402,F401

from examples.optimization import (  # noqa: E402
    optimize_geometry_profiles_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport
    as combined_example,
)
from NEOPAX._block_trust_region import (  # noqa: E402
    block_trust_region_least_squares,
)


OUT_DIR = (
    ROOT
    / "outputs"
    / "geometry_profiles_qi_max_er_transition_bootstrap_net_power_initial_root_"
    "database_full_transport_block_trust_region_optimization"
)


def parser() -> argparse.ArgumentParser:
    out = combined_example.parser()
    out.description = __doc__
    out.set_defaults(out_dir=OUT_DIR, profile_dofs=True)
    out.add_argument("--geometry-initial-radius", type=float, default=1.0)
    out.add_argument("--geometry-min-radius", type=float, default=1.0e-4)
    out.add_argument("--geometry-max-radius", type=float, default=4.0)
    out.add_argument(
        "--profile-initial-fraction-limit", type=float, default=0.10
    )
    out.add_argument(
        "--profile-min-fraction-limit", type=float, default=1.0e-3
    )
    out.add_argument(
        "--profile-max-fraction-limit", type=float, default=0.25
    )
    out.add_argument("--block-proximal-weight", type=float, default=1.0e-8)
    return out


def _problem_kwargs(args: argparse.Namespace, physical_pitches):
    return {
        "vmec_input": args.seed_input.resolve(),
        "max_mode": None,
        "families": combined_example.GEOMETRY_FAMILIES,
        "scale_mode": combined_example.GEOMETRY_SCALE_MODE,
        "ess_alpha": combined_example.ESS_ALPHA,
        "mboz": combined_example.QI_MBOZ,
        "nboz": combined_example.QI_NBOZ,
        "surfaces": tuple(float(value) for value in combined_example.SURFACES),
        "n_theta": int(args.database_n_theta),
        "n_zeta": int(args.database_n_phi),
        "n_xi": int(args.database_n_xi),
        "geometry_max_iter": combined_example.GEOMETRY_MAX_ITER,
        "geometry_solver_device": combined_example.SOLVER_DEVICE,
        "device": combined_example.SOLVER_DEVICE,
        "accepted_step_limit": (
            combined_example.FULL_TRANSPORT_ACCEPTED_STEP_LIMIT
        ),
        "reverse_segment_length": combined_example.REVERSE_SEGMENT_LENGTH,
        "max_reverse_accepted_steps": (
            combined_example.MAX_REVERSE_ACCEPTED_STEPS
        ),
        "initial_er_root_ad": "jax_selected_root",
        "er_transition_left_index": int(args.er_transition_left_index),
        "er_transition_right_index": int(args.er_transition_right_index),
        "er_transition_rho_min": float(args.er_transition_rho_min),
        "er_transition_rho_max": float(args.er_transition_rho_max),
        "er_transition_rho_target": float(args.er_transition_rho_target),
        "er_transition_temperature_kv_m": float(
            args.er_transition_temperature_kv_m
        ),
        "er_transition_rho_softness": float(
            args.er_transition_rho_softness
        ),
        "er_transition_softmax_beta": float(
            args.er_transition_softmax_beta
        ),
        "radau_jacobian_reuse_mode": "legacy",
        "reverse_stage_adjoint_solve_mode": "block",
        "reverse_rhs_transpose_mode": "explicit_database",
        "reverse_stage_cotangent_mode": "full",
        "reverse_step_bwd_mode": "reduced_cotangent_call_boundary",
        "reverse_stage_adjoint_memory_mode": "default",
        "print_final_softmax_er": combined_example.PRINT_FINAL_SOFTMAX_ER,
        "reverse_stage_mode": combined_example.REVERSE_STAGE_MODE,
        "qi_maxj_settings": combined_example.qi_maxj_backend_settings(
            physical_pitches
        ),
        "include_profiles": True,
        "profile_parameters": combined_example.PROFILE_PARAMETERS,
        "profile_scale_mode": combined_example.PROFILE_SCALE_MODE,
        "profile_coordinate_mode": "delta",
    }


def _build_problem(
    args: argparse.Namespace,
    *,
    config,
    vmec_input: Path,
    max_mode: int,
    physical_pitches,
):
    kwargs = _problem_kwargs(args, physical_pitches)
    kwargs["vmec_input"] = vmec_input
    kwargs["max_mode"] = int(max_mode)
    return combined_example.opt.geometry_full_transport_least_squares_problem(
        config,
        combined_example.active_terms(args),
        **kwargs,
    )


def _validate_args(args: argparse.Namespace) -> None:
    if not args.profile_dofs:
        raise ValueError(
            "This new block optimizer is only for the combined problem. "
            "Use the established geometry-only script for --no-profile-dofs."
        )
    if args.postprocess_existing:
        raise ValueError("--postprocess-existing is not supported here.")
    if int(args.max_nfev) < 1:
        raise ValueError("--max-nfev must be positive.")
    for name in (
        "geometry_initial_radius",
        "geometry_min_radius",
        "geometry_max_radius",
        "profile_initial_fraction_limit",
        "profile_min_fraction_limit",
        "profile_max_fraction_limit",
    ):
        if float(getattr(args, name)) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if not (
        args.geometry_min_radius
        <= args.geometry_initial_radius
        <= args.geometry_max_radius
    ):
        raise ValueError("Geometry trust radii must be ordered min <= initial <= max.")
    if not (
        args.profile_min_fraction_limit
        <= args.profile_initial_fraction_limit
        <= args.profile_max_fraction_limit
    ):
        raise ValueError(
            "Profile fraction limits must be ordered min <= initial <= max."
        )
    if float(args.block_proximal_weight) < 0.0:
        raise ValueError("--block-proximal-weight must be nonnegative.")


def _print_evaluation(tag: str, problem, x, evaluation) -> None:
    residuals = np.asarray(jax.device_get(evaluation.residuals), dtype=float)
    jacobian = np.asarray(jax.device_get(evaluation.jacobian), dtype=float)
    print(
        f"[{tag}] elapsed_s={evaluation.elapsed_s:.3f} "
        f"{combined_example.geometry_example.iteration_diagnostics(evaluation)}",
        flush=True,
    )
    print(
        "  physical_profiles="
        f"{combined_example.profile_parameter_values(problem, x)}",
        flush=True,
    )
    print(f"  residual_norm={np.linalg.norm(residuals):.6e}", flush=True)
    print(f"  jacobian_shape={jacobian.shape}", flush=True)


def main() -> int:
    args = parser().parse_args()
    _validate_args(args)
    seed_input = args.seed_input.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    current_input = seed_input
    current_config = combined_example.transport_config(args)
    frozen_physical_pitches = combined_example.PHYSICAL_J_PITCHES
    max_modes = (
        (int(combined_example.MAX_MODE_SCHEDULE),)
        if np.isscalar(combined_example.MAX_MODE_SCHEDULE)
        else tuple(int(value) for value in combined_example.MAX_MODE_SCHEDULE)
    )
    initial_input = initial_config = initial_profiles = None
    optimized_input = optimized_config = optimized_profiles = None
    last_problem = last_result = None

    for max_mode in max_modes:
        print(
            "\n===== geometry + profile block-trust-region full transport "
            f"stage, max_mode={max_mode}, grid=({args.database_n_theta},"
            f"{args.database_n_phi},{args.database_n_xi}), "
            f"J_backend={combined_example.QI_MAXJ_BACKEND} =====",
            flush=True,
        )
        problem = _build_problem(
            args,
            config=current_config,
            vmec_input=current_input,
            max_mode=max_mode,
            physical_pitches=frozen_physical_pitches,
        )
        if (
            combined_example.QI_MAXJ_BACKEND.strip().lower() == "physical"
            and frozen_physical_pitches is None
        ):
            frozen_physical_pitches = tuple(
                float(value)
                for value in problem.context.qi_maxj_physical_pitches
            )
        problem = combined_example.CombinedTrialSavingProblem(
            problem,
            out_dir / f"block_trust_inputs_m{max_mode}",
            max_mode,
        )
        x0 = np.asarray(jax.device_get(problem.x0), dtype=float)
        if not np.array_equal(x0, np.zeros_like(x0)):
            raise AssertionError(
                "Centered geometry/profile coordinates must start exactly at zero."
            )
        if initial_input is None:
            initial_input = problem.input_from_scaled_parameters(x0)
            initial_config = problem.config_from_scaled_parameters(x0)
            initial_profiles = combined_example.profile_parameter_values(
                problem, x0
            )

        print(
            f"[setup] parameter_count={problem.parameter_count} "
            f"parameters={list(problem.parameter_labels)} "
            "geometry_coordinates=ESS_delta profile_coordinates="
            "centered_nominal_delta optimizer=block_trust_region",
            flush=True,
        )
        initial_evaluation = problem.evaluate(x0)
        _print_evaluation("initial", problem, x0, initial_evaluation)
        lower, upper = combined_example.scaled_bounds(problem)
        profile_mask = np.asarray(
            [
                label in combined_example.PROFILE_PHYSICAL_LOWER
                for label in problem.parameter_labels
            ],
            dtype=bool,
        )
        result = block_trust_region_least_squares(
            problem,
            profile_mask=profile_mask,
            bounds=(lower, upper),
            initial_evaluation=initial_evaluation,
            max_nfev=int(args.max_nfev),
            geometry_initial_radius=float(args.geometry_initial_radius),
            geometry_min_radius=float(args.geometry_min_radius),
            geometry_max_radius=float(args.geometry_max_radius),
            profile_initial_fraction_limit=float(
                args.profile_initial_fraction_limit
            ),
            profile_min_fraction_limit=float(
                args.profile_min_fraction_limit
            ),
            profile_max_fraction_limit=float(
                args.profile_max_fraction_limit
            ),
            proximal_weight=float(args.block_proximal_weight),
            ftol=combined_example.FTOL,
            xtol=combined_example.XTOL,
            iteration_reporter=(
                combined_example.geometry_example.iteration_diagnostics
            ),
            verbose=1,
        )
        x_opt = np.asarray(result.x, dtype=float)
        _print_evaluation(
            f"block trust stage {max_mode}",
            problem,
            x_opt,
            result.evaluation,
        )
        optimized_input = problem.input_from_scaled_parameters(x_opt)
        optimized_config = problem.config_from_scaled_parameters(x_opt)
        optimized_profiles = combined_example.profile_parameter_values(
            problem, x_opt
        )
        stage_input = out_dir / f"input.QI_neopax_block_trust_stage_m{max_mode}"
        optimized_input.to_indata(stage_input)
        print(f"wrote {stage_input}", flush=True)
        current_input = stage_input
        current_config = optimized_config
        last_problem = problem
        last_result = result

    required = (
        initial_input,
        initial_config,
        initial_profiles,
        optimized_input,
        optimized_config,
        optimized_profiles,
        last_problem,
        last_result,
    )
    if any(value is None for value in required):
        raise RuntimeError("No block trust-region optimization stage was executed.")

    summary = {
        "seed_input": str(seed_input),
        "transport_config": str(combined_example.TRANSPORT_CONFIG),
        "database_grid": [
            int(args.database_n_theta),
            int(args.database_n_phi),
            int(args.database_n_xi),
        ],
        "max_mode_schedule": list(max_modes),
        "optimizer": "joint_block_trust_region",
        "geometry_coordinate_mode": "ESS_delta",
        "profile_scale_mode": combined_example.PROFILE_SCALE_MODE,
        "profile_coordinate_mode": "delta",
        "initial_physical_profiles": initial_profiles,
        "optimized_physical_profiles": optimized_profiles,
        "parameter_labels": list(last_problem.parameter_labels),
        "x_scaled_delta": np.asarray(last_result.x, dtype=float).tolist(),
        "block_trust_settings": {
            "geometry_initial_radius": float(args.geometry_initial_radius),
            "geometry_min_radius": float(args.geometry_min_radius),
            "geometry_max_radius": float(args.geometry_max_radius),
            "profile_initial_fraction_limit": float(
                args.profile_initial_fraction_limit
            ),
            "profile_min_fraction_limit": float(
                args.profile_min_fraction_limit
            ),
            "profile_max_fraction_limit": float(
                args.profile_max_fraction_limit
            ),
            "proximal_weight": float(args.block_proximal_weight),
        },
        "cost": float(last_result.cost),
        "optimality": float(last_result.optimality),
        "nfev": int(last_result.nfev),
        "nit": int(last_result.nit),
        "accepted_steps": int(last_result.accepted_steps),
        "rejected_steps": int(last_result.rejected_steps),
        "final_geometry_radius": float(last_result.geometry_radius),
        "final_profile_fraction_limit": float(
            last_result.profile_fraction_limit
        ),
        "status": int(last_result.status),
        "message": str(last_result.message),
    }
    summary_path = out_dir / "optimization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}", flush=True)
    combined_example.write_outputs(
        initial_input=initial_input,
        optimized_input=optimized_input,
        initial_config=initial_config,
        optimized_config=optimized_config,
        initial_profiles=initial_profiles,
        optimized_profiles=optimized_profiles,
        out_dir=out_dir,
        seed_input=seed_input,
        make_initial_plots=args.initial_plots,
        physical_pitches=frozen_physical_pitches,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
