#!/usr/bin/env python
"""Optimize geometry first, then add a profile correction to each step.

This is an opt-in, standalone full-transport optimization lane.  At every
accepted nonlinear state it:

1. constructs an ESS-coordinate geometry Gauss--Newton proposal;
2. freezes that geometry proposal;
3. constructs a nominal-profile correction for the residual predicted after
   the geometry proposal; and
4. evaluates the combined proposal once with VMEC/NTX/full transport.

A rejected combined proposal contracts the profile correction first.  The
geometry radius is contracted only after the corresponding exact
geometry-only proposal is also rejected.  Thus profiles can improve a
geometry-led path without replacing its step in the local model.

The established geometry-only, ordinary combined, and symmetric block-trust
optimization paths are not modified or called by this script.  Their shared
physics, objective, parameter, and output definitions are imported so this
experiment starts from exactly the same configured problem.
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

# Preserve the validated VMEX-first initialization order used by the
# established geometry-only path.
import vmex as vj  # noqa: E402,F401
from vmex import optimize as vmex_opt  # noqa: E402,F401

from examples.optimization import (  # noqa: E402
    optimize_geometry_profiles_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport
    as combined_example,
)
from NEOPAX._geometry_primary_trust_region import (  # noqa: E402
    geometry_primary_profile_correction_least_squares,
)


OUT_DIR = (
    ROOT
    / "outputs"
    / "geometry_profiles_qi_max_er_transition_bootstrap_net_power_initial_root_"
    "database_full_transport_geometry_primary_optimization"
)

# These defaults are copied by reference from the established combined
# example, whose geometry settings in turn come directly from the validated
# geometry-only example.  They remain editable here and on the CLI without
# mutating either existing optimization lane.
QI_WEIGHT = combined_example.QI_WEIGHT
MAXJ_WEIGHT = combined_example.MAXJ_WEIGHT
MIRROR_WEIGHT = combined_example.MIRROR_WEIGHT
ASPECT_WEIGHT = combined_example.ASPECT_WEIGHT
IOTA_WEIGHT = combined_example.IOTA_WEIGHT
MAX_ER_WEIGHT = combined_example.MAX_ER_WEIGHT
ER_TRANSITION_LEFT_WEIGHT = combined_example.ER_TRANSITION_LEFT_WEIGHT
ER_TRANSITION_RIGHT_WEIGHT = combined_example.ER_TRANSITION_RIGHT_WEIGHT
ER_TRANSITION_RHO_WEIGHT = combined_example.ER_TRANSITION_RHO_WEIGHT
ER_TRANSITION_STRENGTH_WEIGHT = combined_example.ER_TRANSITION_STRENGTH_WEIGHT
BOOTSTRAP_WEIGHT = combined_example.BOOTSTRAP_WEIGHT
NET_POWER_WEIGHT = combined_example.NET_POWER_WEIGHT


def _add_weight_arguments(out: argparse.ArgumentParser) -> None:
    for flag, default in (
        ("qi", QI_WEIGHT),
        ("maxj", MAXJ_WEIGHT),
        ("mirror", MIRROR_WEIGHT),
        ("aspect", ASPECT_WEIGHT),
        ("iota", IOTA_WEIGHT),
        ("max-er", MAX_ER_WEIGHT),
        ("er-transition-left", ER_TRANSITION_LEFT_WEIGHT),
        ("er-transition-right", ER_TRANSITION_RIGHT_WEIGHT),
        ("er-transition-rho", ER_TRANSITION_RHO_WEIGHT),
        ("er-transition-strength", ER_TRANSITION_STRENGTH_WEIGHT),
        ("bootstrap", BOOTSTRAP_WEIGHT),
        ("net-power", NET_POWER_WEIGHT),
    ):
        out.add_argument(
            f"--{flag}-weight",
            type=float,
            default=float(default),
            help=f"least-squares weight for {flag} (default: {default:g})",
        )


def parser() -> argparse.ArgumentParser:
    """Return this lane's CLI while retaining the shared problem defaults."""

    out = combined_example.parser()
    out.description = __doc__
    out.set_defaults(out_dir=OUT_DIR, profile_dofs=True)
    _add_weight_arguments(out)
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


def objective_weights(args: argparse.Namespace) -> dict[str, float]:
    return {
        "qi": float(args.qi_weight),
        "maxj": float(args.maxj_weight),
        "mirror": float(args.mirror_weight),
        "aspect": float(args.aspect_weight),
        "iota": float(args.iota_weight),
        "max_er": float(args.max_er_weight),
        "er_transition_left": float(args.er_transition_left_weight),
        "er_transition_right": float(args.er_transition_right_weight),
        "er_transition_rho": float(args.er_transition_rho_weight),
        "er_transition_strength": float(args.er_transition_strength_weight),
        "bootstrap": float(args.bootstrap_weight),
        "net_power": float(args.net_power_weight),
    }


def active_terms(args: argparse.Namespace):
    """Apply this lane's local weights to the shared objective definitions."""

    opt = combined_example.opt
    geometry_example = combined_example.geometry_example
    weights_by_label = {
        opt.geometry.boozer_qi_objective.label: float(args.qi_weight),
        opt.geometry.boozer_maxj_objective.label: float(args.maxj_weight),
        geometry_example.mirror_penalization.label: float(args.mirror_weight),
        opt.geometry.vmec_aspect_ratio.label: float(args.aspect_weight),
        opt.geometry.vmec_iota_mean.label: float(args.iota_weight),
        opt.transport.softmax_Er.label: float(args.max_er_weight),
        opt.transport.Er_transition_left.label: float(
            args.er_transition_left_weight
        ),
        opt.transport.Er_transition_right.label: float(
            args.er_transition_right_weight
        ),
        opt.transport.Er_transition_location_moment.label: float(
            args.er_transition_rho_weight
        ),
        "Er_transition_strength_deficit": float(
            args.er_transition_strength_weight
        ),
        geometry_example.bootstrap_penalty.label: float(args.bootstrap_weight),
        opt.transport.net_total_power_volume_average_mw_m3.label: float(
            args.net_power_weight
        ),
    }
    return tuple(
        (
            objective,
            target,
            weights_by_label.get(objective.label, float(inherited_weight)),
        )
        for objective, target, inherited_weight in combined_example.active_terms(args)
    )


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
        active_terms(args),
        **kwargs,
    )


def _validate_args(args) -> None:
    if not args.profile_dofs:
        raise ValueError(
            "The geometry-primary correction lane requires profile DoFs. "
            "Use the established geometry-only script for geometry only."
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
    negative_weights = {
        name: value
        for name, value in objective_weights(args).items()
        if value < 0.0
    }
    if negative_weights:
        raise ValueError(
            "Least-squares objective weights must be nonnegative; got "
            f"{negative_weights}."
        )
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
            "\n===== geometry-primary + profile-correction full transport "
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
            out_dir / f"geometry_primary_inputs_m{max_mode}",
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
            "centered_nominal_delta "
            "optimizer=geometry_primary_profile_correction",
            flush=True,
        )
        print(
            f"[setup] objective_weights={objective_weights(args)}",
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
        result = geometry_primary_profile_correction_least_squares(
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
            f"geometry-primary stage {max_mode}",
            problem,
            x_opt,
            result.evaluation,
        )
        optimized_input = problem.input_from_scaled_parameters(x_opt)
        optimized_config = problem.config_from_scaled_parameters(x_opt)
        optimized_profiles = combined_example.profile_parameter_values(
            problem, x_opt
        )
        stage_input = (
            out_dir / f"input.QI_neopax_geometry_primary_stage_m{max_mode}"
        )
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
        raise RuntimeError("No geometry-primary optimization stage was executed.")

    summary = {
        "seed_input": str(seed_input),
        "transport_config": str(combined_example.TRANSPORT_CONFIG),
        "database_grid": [
            int(args.database_n_theta),
            int(args.database_n_phi),
            int(args.database_n_xi),
        ],
        "max_mode_schedule": list(max_modes),
        "optimizer": "geometry_primary_profile_correction_trust_region",
        "step_policy": (
            "geometry_first_then_profile_correction; "
            "contract_profile_before_geometry"
        ),
        "geometry_coordinate_mode": "ESS_delta",
        "profile_scale_mode": combined_example.PROFILE_SCALE_MODE,
        "profile_coordinate_mode": "delta",
        "initial_physical_profiles": initial_profiles,
        "optimized_physical_profiles": optimized_profiles,
        "parameter_labels": list(last_problem.parameter_labels),
        "objective_weights": objective_weights(args),
        "x_scaled_delta": np.asarray(last_result.x, dtype=float).tolist(),
        "trust_settings": {
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
        "profile_contractions": int(last_result.profile_contractions),
        "geometry_contractions": int(last_result.geometry_contractions),
        "geometry_only_trials": int(last_result.geometry_only_trials),
        "final_geometry_radius": float(last_result.geometry_radius),
        "final_profile_fraction_limit": float(
            last_result.profile_fraction_limit
        ),
        "status": int(last_result.status),
        "success": bool(last_result.success),
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
