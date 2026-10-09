#!/usr/bin/env python
"""Simple ESS-geometry plus nominal-profile full-transport least squares.

This standalone baseline deliberately contains no experimental block solver or
additional optimizer scaling.  It uses the validated ESS geometry coordinates,
all six nominally scaled analytical-profile coordinates, and the established
``NEOPAX.optimization.least_squares`` SciPy-TRF path with its unit optimizer
metric.  Profile coordinates are absolute nominal coordinates, so their seed
values are exactly one while the ESS geometry deltas start at zero.

The script exists as a clean reference for deciding how to increase profile
motion without changing the validated geometry/objective implementation.
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

# Preserve the validated geometry-only import/initialization order.
import vmex as vj  # noqa: E402,F401
from vmex import optimize as vmex_opt  # noqa: E402,F401

from examples.optimization import (  # noqa: E402
    optimize_geometry_profiles_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport
    as combined_example,
)
from NEOPAX import optimization as opt  # noqa: E402


OUT_DIR = (
    ROOT
    / "outputs"
    / "geometry_profiles_qi_max_er_transition_bootstrap_net_power_initial_root_"
    "database_full_transport_simple_least_squares_optimization"
)


def parser() -> argparse.ArgumentParser:
    out = combined_example.parser()
    out.description = __doc__
    out.set_defaults(out_dir=OUT_DIR, profile_dofs=True)
    return out


def _validate_args(args: argparse.Namespace) -> None:
    if not args.profile_dofs:
        raise ValueError(
            "This baseline is specifically the combined geometry/profile "
            "problem. Use the established geometry-only script when profile "
            "DoFs are disabled."
        )
    if args.postprocess_existing:
        raise ValueError("--postprocess-existing is not supported here.")
    if int(args.max_nfev) < 1:
        raise ValueError("--max-nfev must be positive.")
    for name in ("database_n_theta", "database_n_phi", "database_n_xi"):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    config = combined_example.transport_config(args)
    n_radial = int(config.get("geometry", {}).get("n_radial", 51))
    for name in ("er_transition_left_index", "er_transition_right_index"):
        index = int(getattr(args, name))
        if not 0 <= index < n_radial:
            raise ValueError(
                f"--{name.replace('_', '-')} must be in [0, {n_radial}); "
                f"got {index}."
            )
    if not (
        0.0
        <= float(args.er_transition_rho_min)
        < float(args.er_transition_rho_max)
        <= 1.0
    ):
        raise ValueError(
            "Er transition radial window must satisfy 0 <= min < max <= 1."
        )
    if not (
        float(args.er_transition_rho_min)
        <= float(args.er_transition_rho_target)
        <= float(args.er_transition_rho_max)
    ):
        raise ValueError("Er transition target must lie inside the radial window.")
    if min(
        float(args.er_transition_strength_target),
        float(args.er_transition_temperature_kv_m),
        float(args.er_transition_rho_softness),
        float(args.er_transition_softmax_beta),
    ) <= 0.0:
        raise ValueError(
            "Er transition strength/temperature/softness/beta must be positive."
        )


def _problem_kwargs(args: argparse.Namespace, physical_pitches):
    """Return the established geometry problem plus nominal profile DoFs."""

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
        "profile_scale_mode": "nominal",
        "profile_coordinate_mode": "absolute",
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
    return opt.geometry_full_transport_least_squares_problem(
        config,
        combined_example.active_terms(args),
        **kwargs,
    )


def _least_squares_options(args, initial_evaluation, bounds):
    """Return only the established SciPy-TRF options for this baseline."""

    return {
        "max_nfev": int(args.max_nfev),
        "ftol": combined_example.FTOL,
        "xtol": combined_example.XTOL,
        "verbose": 1,
        "iteration_reporter": (
            combined_example.geometry_example.iteration_diagnostics
        ),
        "initial_evaluation": initial_evaluation,
        "bounds": bounds,
        # Intentionally no x_scale: opt.least_squares supplies its validated
        # unit vector in the already scaled ESS/nominal coordinates.
    }


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
            "\n===== simple SciPy least-squares: ESS geometry + nominal "
            f"profiles, max_mode={max_mode}, grid=({args.database_n_theta},"
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
            out_dir / f"simple_least_squares_inputs_m{max_mode}",
            max_mode,
        )
        x0 = np.asarray(jax.device_get(problem.x0), dtype=float)
        profile_mask = np.asarray(
            [
                label in combined_example.PROFILE_PHYSICAL_LOWER
                for label in problem.parameter_labels
            ],
            dtype=bool,
        )
        if not np.all(x0[profile_mask] == 1.0):
            raise AssertionError(
                "Nominal absolute profile coordinates must start exactly at one."
            )
        if not np.all(x0[~profile_mask] == 0.0):
            raise AssertionError("ESS geometry deltas must start exactly at zero.")
        if initial_input is None:
            initial_input = problem.input_from_scaled_parameters(x0)
            initial_config = problem.config_from_scaled_parameters(x0)
            initial_profiles = combined_example.profile_parameter_values(
                problem, x0
            )

        print(
            f"[setup] parameter_count={problem.parameter_count} "
            f"parameters={list(problem.parameter_labels)} "
            "geometry_coordinates=ESS_delta "
            "profile_coordinates=absolute_nominal "
            "optimizer=scipy_least_squares_TRF optimizer_x_scale=unit",
            flush=True,
        )
        initial_evaluation = combined_example.report(
            "initial", problem, x0
        )
        bounds = combined_example.scaled_bounds(problem)
        last_result = opt.least_squares(
            problem,
            **_least_squares_options(args, initial_evaluation, bounds),
        )
        x_opt = np.asarray(last_result.x, dtype=float)
        combined_example.report(
            f"simple combined least-squares stage {max_mode}",
            problem,
            x_opt,
        )
        optimized_input = problem.input_from_scaled_parameters(x_opt)
        optimized_config = problem.config_from_scaled_parameters(x_opt)
        optimized_profiles = combined_example.profile_parameter_values(
            problem, x_opt
        )
        stage_input = (
            out_dir / f"input.QI_neopax_simple_least_squares_stage_m{max_mode}"
        )
        optimized_input.to_indata(stage_input)
        print(f"wrote {stage_input}", flush=True)
        current_input = stage_input
        current_config = optimized_config
        last_problem = problem

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
        raise RuntimeError("No simple combined optimization stage was executed.")

    summary = {
        "seed_input": str(seed_input),
        "transport_config": str(combined_example.TRANSPORT_CONFIG),
        "database_grid": [
            int(args.database_n_theta),
            int(args.database_n_phi),
            int(args.database_n_xi),
        ],
        "max_mode_schedule": list(max_modes),
        "optimizer": "scipy_least_squares_TRF",
        "optimizer_x_scale": "unit",
        "geometry_coordinate_mode": "ESS_delta",
        "profile_scale_mode": "nominal",
        "profile_coordinate_mode": "absolute",
        "profile_dofs_enabled": True,
        "initial_physical_profiles": initial_profiles,
        "optimized_physical_profiles": optimized_profiles,
        "parameter_labels": list(last_problem.parameter_labels),
        "x_scaled": np.asarray(last_result.x, dtype=float).tolist(),
        "objectives": {
            "max_er": bool(args.max_er),
            "root_objectives": bool(args.root_objectives),
            "transition_location": bool(args.transition_location_objective),
            "bootstrap_penalty": bool(args.bootstrap_penalty),
            "net_power": bool(args.net_power),
        },
        "cost": float(last_result.cost),
        "optimality": float(last_result.optimality),
        "nfev": int(last_result.nfev),
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
