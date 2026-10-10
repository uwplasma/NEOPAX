#!/usr/bin/env python
"""Geometry warm start followed by combined geometry/profile least squares.

This is the deliberately sequential reference experiment for the combined
full-transport problem:

1. Run the established geometry-only SciPy-TRF optimization with its exact
   ESS parameterization and objective definitions.
2. Rebuild the same full-transport problem with the same baseline and ESS
   scales, initialize its geometry block at the accepted phase-1 coordinates,
   add all six nominally scaled analytical-profile DoFs, and run ordinary
   nonlinear SciPy-TRF least squares again.

The validated geometry-only example and its library path are not modified.
The second phase starts at the geometry-only result with nominal profiles, so
its first cost is the geometry warm-start cost.  SciPy may then accept only
combined steps that reduce that cost.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import jax
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Preserve the validated geometry-only process initialization order.
import vmex as vj  # noqa: E402,F401
from vmex import optimize as vmex_opt  # noqa: E402,F401

from NEOPAX import optimization as opt  # noqa: E402
from examples.optimization import (  # noqa: E402
    optimize_geometry_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport
    as geometry_example,
)


OUT_DIR = (
    ROOT
    / "outputs"
    / "geometry_warm_start_profiles_qi_max_er_transition_bootstrap_net_power_"
    "initial_root_database_full_transport_optimization"
)

# Phase 2 uses physical relative deltas by default: p = p_warm * (1 + delta).
# Unit SciPy scaling is intentional.  The failed 3.7x trust-scale experiment
# is not silently carried into this reference run.
PROFILE_COORDINATE_MODE = "delta"
PROFILE_TRUST_MULTIPLIER = 1.0


def parser() -> argparse.ArgumentParser:
    out = geometry_example.parser()
    out.description = __doc__
    out.set_defaults(out_dir=OUT_DIR)
    out.add_argument(
        "--geometry-max-nfev",
        type=int,
        default=None,
        help="Geometry-only phase budget; defaults to --max-nfev.",
    )
    out.add_argument(
        "--combined-max-nfev",
        type=int,
        default=None,
        help="Combined geometry/profile phase budget; defaults to --max-nfev.",
    )
    out.add_argument(
        "--profile-coordinate-mode",
        choices=("delta", "absolute"),
        default=PROFILE_COORDINATE_MODE,
        help=(
            "Profile coordinates in phase 2. 'delta' starts at zero and maps "
            "p=p_warm*(1+delta); 'absolute' starts nominal coordinates at one."
        ),
    )
    out.add_argument(
        "--profile-trust-multiplier",
        type=float,
        default=PROFILE_TRUST_MULTIPLIER,
        help=(
            "SciPy x_scale for profile coordinates in phase 2. Geometry "
            "coordinates always remain one. The reference default is one."
        ),
    )
    return out


def _phase_budget(value: int | None, fallback: int) -> int:
    result = int(fallback if value is None else value)
    if result < 1:
        raise ValueError("Both optimization phase budgets must be positive.")
    return result


def _geometry_problem_kwargs(args: argparse.Namespace, physical_pitches):
    """Exact keyword bundle used by the established geometry-only example."""

    return {
        "max_mode": None,
        "families": geometry_example.GEOMETRY_FAMILIES,
        "scale_mode": geometry_example.SCALE_MODE,
        "ess_alpha": geometry_example.ESS_ALPHA,
        "mboz": geometry_example.QI_MBOZ,
        "nboz": geometry_example.QI_NBOZ,
        "surfaces": tuple(float(value) for value in geometry_example.SURFACES),
        "n_theta": int(args.database_n_theta),
        "n_zeta": int(args.database_n_phi),
        "n_xi": int(args.database_n_xi),
        "geometry_max_iter": geometry_example.GEOMETRY_MAX_ITER,
        "geometry_solver_device": geometry_example.SOLVER_DEVICE,
        "device": geometry_example.SOLVER_DEVICE,
        "accepted_step_limit": geometry_example.FULL_TRANSPORT_ACCEPTED_STEP_LIMIT,
        "reverse_segment_length": geometry_example.REVERSE_SEGMENT_LENGTH,
        "max_reverse_accepted_steps": geometry_example.MAX_REVERSE_ACCEPTED_STEPS,
        "initial_er_root_ad": "jax_selected_root",
        "er_transition_left_index": int(args.er_transition_left_index),
        "er_transition_right_index": int(args.er_transition_right_index),
        "er_transition_rho_min": float(args.er_transition_rho_min),
        "er_transition_rho_max": float(args.er_transition_rho_max),
        "er_transition_rho_target": float(args.er_transition_rho_target),
        "er_transition_temperature_kv_m": float(
            args.er_transition_temperature_kv_m
        ),
        "er_transition_rho_softness": float(args.er_transition_rho_softness),
        "er_transition_softmax_beta": float(args.er_transition_softmax_beta),
        "radau_jacobian_reuse_mode": "legacy",
        "reverse_stage_adjoint_solve_mode": "block",
        "reverse_rhs_transpose_mode": "explicit_database",
        "reverse_stage_cotangent_mode": "full",
        "reverse_step_bwd_mode": "reduced_cotangent_call_boundary",
        "reverse_stage_adjoint_memory_mode": "default",
        "print_final_softmax_er": geometry_example.PRINT_FINAL_SOFTMAX_ER,
        "reverse_stage_mode": geometry_example.REVERSE_STAGE_MODE,
        "qi_maxj_settings": geometry_example.qi_maxj_backend_settings(
            physical_pitches
        ),
    }


def _build_geometry_problem(
    args: argparse.Namespace,
    *,
    config,
    vmec_input: Path,
    max_mode: int,
    physical_pitches,
):
    kwargs = _geometry_problem_kwargs(args, physical_pitches)
    kwargs["vmec_input"] = vmec_input
    kwargs["max_mode"] = int(max_mode)
    # Do not pass profile arguments here. This is the unchanged geometry-only
    # API default and therefore has exactly the established 24-column layout.
    return opt.geometry_full_transport_least_squares_problem(
        config,
        geometry_example.active_terms(args),
        **kwargs,
    )


def _cost(evaluation) -> float:
    residuals = np.asarray(jax.device_get(evaluation.residuals), dtype=float)
    return 0.5 * float(np.dot(residuals, residuals))


def _run_geometry_phase(
    args: argparse.Namespace,
    *,
    config,
    seed_input: Path,
    out_dir: Path,
    max_modes: tuple[int, ...],
    max_nfev: int,
    physical_pitches,
):
    """Run the established geometry-only nonlinear least-squares phase."""

    current_input = seed_input
    initial_input = None
    final_input = None
    final_problem = None
    final_result = None
    final_baseline_input = None
    initial_cost = None

    for max_mode in max_modes:
        stage_baseline_input = current_input
        print(
            "\n===== phase 1/2: validated geometry-only full transport "
            f"stage, max_mode={max_mode}, grid=({args.database_n_theta},"
            f"{args.database_n_phi},{args.database_n_xi}), "
            f"J_backend={geometry_example.QI_MAXJ_BACKEND} =====",
            flush=True,
        )
        problem = _build_geometry_problem(
            args,
            config=config,
            vmec_input=current_input,
            max_mode=max_mode,
            physical_pitches=physical_pitches,
        )
        if (
            geometry_example.QI_MAXJ_BACKEND.strip().lower() == "physical"
            and physical_pitches is None
        ):
            physical_pitches = tuple(
                float(value) for value in problem.context.qi_maxj_physical_pitches
            )
            print(
                "[setup] frozen_physical_J_pitches_T^-1="
                + ",".join(f"{value:.16g}" for value in physical_pitches),
                flush=True,
            )

        problem = opt.GeometryInputSavingProblem(
            problem,
            out_dir / f"geometry_warm_start_inputs_m{max_mode}",
            filename_prefix=(
                "input.QI_neopax_geometry_warm_start_eval"
            ),
        )
        x0 = np.asarray(jax.device_get(problem.x0), dtype=float)
        if not np.all(x0 == 0.0):
            raise AssertionError(
                "The validated geometry-only ESS coordinates must start at zero."
            )
        if initial_input is None:
            initial_input = problem.input_from_scaled_parameters(x0)

        print(
            f"[setup] phase=geometry_only parameter_count={problem.parameter_count} "
            f"parameters={list(problem.parameter_labels)} "
            f"max_nfev={max_nfev} geometry_coordinates=ESS_delta "
            "optimizer=scipy_least_squares_TRF optimizer_x_scale=unit",
            flush=True,
        )
        initial_evaluation = geometry_example.report("geometry initial", problem, x0)
        if initial_cost is None:
            initial_cost = _cost(initial_evaluation)
        result = opt.least_squares(
            problem,
            max_nfev=max_nfev,
            ftol=geometry_example.FTOL,
            xtol=geometry_example.XTOL,
            verbose=1,
            iteration_reporter=geometry_example.iteration_diagnostics,
            initial_evaluation=initial_evaluation,
        )
        x_opt = np.asarray(result.x, dtype=float)
        geometry_example.report(
            f"geometry warm-start stage {max_mode}", problem, x_opt
        )
        final_input = problem.input_from_scaled_parameters(x_opt)
        stage_input = out_dir / (
            f"input.QI_neopax_geometry_warm_start_stage_m{max_mode}"
        )
        final_input.to_indata(stage_input)
        print(f"wrote {stage_input}", flush=True)
        current_input = stage_input
        final_problem = problem
        final_result = result
        final_baseline_input = stage_baseline_input

    if any(
        value is None
        for value in (
            initial_input,
            final_input,
            final_problem,
            final_result,
            final_baseline_input,
        )
    ):
        raise RuntimeError("No geometry-only warm-start stage was executed.")

    warm_path = out_dir / "input.QI_neopax_geometry_warm_start_optimized"
    final_input.to_indata(warm_path)
    print(f"wrote {warm_path}", flush=True)
    return {
        "initial_input": initial_input,
        "warm_input": final_input,
        "warm_path": warm_path,
        "baseline_input": final_baseline_input,
        "problem": final_problem,
        "result": final_result,
        "initial_cost": initial_cost,
        "physical_pitches": physical_pitches,
    }


def _combined_starting_point(geometry_problem, geometry_result, combined_problem):
    """Append nominal profile coordinates without rebasing geometry.

    Geometry values and scales are matched by label so this remains correct if
    the mixed parameter set stores its profile block before its boundary block.
    """

    geometry_labels = tuple(geometry_problem.parameter_labels)
    geometry_x = np.asarray(geometry_result.x, dtype=float)
    geometry_scales = np.asarray(
        jax.device_get(geometry_problem.x_scale), dtype=float
    )
    combined_labels = tuple(combined_problem.parameter_labels)
    combined_scales = np.asarray(
        jax.device_get(combined_problem.x_scale), dtype=float
    )
    if geometry_x.shape != (len(geometry_labels),):
        raise ValueError("The accepted geometry vector has an unexpected shape.")

    geometry_by_label = dict(zip(geometry_labels, geometry_x, strict=True))
    scale_by_label = dict(zip(geometry_labels, geometry_scales, strict=True))
    missing = [label for label in geometry_labels if label not in combined_labels]
    if missing:
        raise ValueError(
            "The combined problem is missing accepted geometry coordinates: "
            + ", ".join(missing)
        )

    start = np.asarray(jax.device_get(combined_problem.x0), dtype=float).copy()
    geometry_scale_differences = []
    for index, label in enumerate(combined_labels):
        if label not in geometry_by_label:
            continue
        start[index] = geometry_by_label[label]
        geometry_scale_differences.append(
            abs(combined_scales[index] - scale_by_label[label])
        )
    max_scale_difference = max(geometry_scale_differences, default=0.0)
    if max_scale_difference != 0.0:
        raise AssertionError(
            "Phase 2 changed the phase-1 ESS geometry scales; "
            f"max_abs_difference={max_scale_difference:.16e}."
        )
    return start


def main() -> int:
    args = parser().parse_args()
    geometry_budget = _phase_budget(args.geometry_max_nfev, args.max_nfev)
    combined_budget = _phase_budget(args.combined_max_nfev, args.max_nfev)
    if (
        not np.isfinite(float(args.profile_trust_multiplier))
        or float(args.profile_trust_multiplier) <= 0.0
    ):
        raise ValueError("--profile-trust-multiplier must be finite and positive.")

    seed_input = args.seed_input.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    forward_config = geometry_example.transport_config(args)
    max_modes = geometry_example.max_mode_schedule_values()

    geometry_phase = _run_geometry_phase(
        args,
        config=forward_config,
        seed_input=seed_input,
        out_dir=out_dir,
        max_modes=max_modes,
        max_nfev=geometry_budget,
        physical_pitches=geometry_example.PHYSICAL_J_PITCHES,
    )

    # Import the combined helper only after phase 1. This keeps all imports
    # preceding the geometry-only solve identical to the validated example.
    from examples.optimization import (  # noqa: E402
        optimize_geometry_profiles_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_simple_least_squares
        as combined_helper,
    )

    combined_helper._validate_args(args)
    # Keep the final geometry phase's original baseline. The accepted geometry
    # is represented by its existing ESS coordinate vector, not by replacing
    # the baseline and resetting those coordinates to zero.
    current_input = geometry_phase["baseline_input"]
    current_config = copy.deepcopy(forward_config)
    physical_pitches = geometry_phase["physical_pitches"]
    # Phase 1 has already completed the whole resolution continuation. Keep
    # phase 2 at that final resolution so its x=0 point is exactly the final
    # geometry-only problem, now augmented by profile columns.
    combined_modes = (max_modes[-1],)
    nominal_profiles = None
    optimized_profiles = None
    optimized_config = None
    final_input = None
    final_problem = None
    final_result = None
    combined_initial_cost = None

    for max_mode in combined_modes:
        print(
            "\n===== phase 2/2: geometry-warm-start combined geometry/profile "
            f"full transport stage, max_mode={max_mode}, "
            f"grid=({args.database_n_theta},{args.database_n_phi},"
            f"{args.database_n_xi}), J_backend="
            f"{geometry_example.QI_MAXJ_BACKEND} =====",
            flush=True,
        )
        problem = combined_helper._build_problem(
            args,
            config=current_config,
            vmec_input=current_input,
            max_mode=max_mode,
            physical_pitches=physical_pitches,
        )
        problem = combined_helper.TrialSavingProblem(
            problem,
            out_dir / f"combined_warm_start_inputs_m{max_mode}",
        )
        x0 = _combined_starting_point(
            geometry_phase["problem"], geometry_phase["result"], problem
        )
        profile_names = frozenset(combined_helper.PROFILE_PARAMETERS.split(","))
        profile_mask = np.asarray(
            [label in profile_names for label in problem.parameter_labels],
            dtype=bool,
        )
        expected_profile_x0 = (
            0.0 if args.profile_coordinate_mode == "delta" else 1.0
        )
        if not np.all(x0[profile_mask] == expected_profile_x0):
            raise AssertionError(
                "Combined nominal profile coordinates have an unexpected origin."
            )
        accepted_geometry_by_label = dict(
            zip(
                geometry_phase["problem"].parameter_labels,
                np.asarray(geometry_phase["result"].x, dtype=float),
                strict=True,
            )
        )
        expected_geometry_x = np.asarray(
            [
                accepted_geometry_by_label[label]
                for label in np.asarray(problem.parameter_labels)[~profile_mask]
            ],
            dtype=float,
        )
        if not np.array_equal(x0[~profile_mask], expected_geometry_x):
            raise AssertionError(
                "Combined ESS coordinates do not equal the accepted geometry "
                "coordinates from phase 1."
            )

        reconstructed_path = out_dir / (
            "input.QI_neopax_combined_initial_reconstructed_from_phase_1"
        )
        problem.input_from_scaled_parameters(x0).to_indata(reconstructed_path)
        if reconstructed_path.read_bytes() != geometry_phase["warm_path"].read_bytes():
            raise AssertionError(
                "Phase 2 did not reconstruct the exact accepted phase-1 VMEC "
                "input from the preserved ESS coordinates."
            )
        print(
            "[warm-start coordinate parity] preserved_phase_1_ESS_scales=True "
            "reconstructed_boundary_exact=True",
            flush=True,
        )

        if nominal_profiles is None:
            nominal_profiles = combined_helper.profile_parameter_values(problem, x0)

        trust_scale = combined_helper.optimizer_trust_region_x_scale(
            problem, args.profile_trust_multiplier
        )
        if not np.all(trust_scale[~profile_mask] == 1.0):
            raise AssertionError("Every ESS geometry optimizer scale must remain one.")
        print(
            f"[setup] phase=combined parameter_count={problem.parameter_count} "
            f"parameters={list(problem.parameter_labels)} max_nfev={combined_budget} "
            "geometry_coordinates=preserved_phase_1_ESS_delta "
            f"profile_coordinates={args.profile_coordinate_mode}_nominal "
            f"profile_trust_multiplier={args.profile_trust_multiplier:.8g} "
            "geometry_trust_multiplier=1 optimizer=scipy_least_squares_TRF",
            flush=True,
        )
        initial_evaluation = combined_helper.report(
            "combined initial at accepted geometry warm start", problem, x0
        )
        if combined_initial_cost is None:
            combined_initial_cost = _cost(initial_evaluation)
            geometry_warm_cost = float(geometry_phase["result"].cost)
            warm_cost_difference = combined_initial_cost - geometry_warm_cost
            print(
                "[warm-start parity] "
                f"geometry_final_cost={geometry_warm_cost:.16e} "
                f"combined_initial_cost={combined_initial_cost:.16e} "
                f"difference={warm_cost_difference:.16e}",
                flush=True,
            )
        least_squares_options = combined_helper._least_squares_options(
            args,
            initial_evaluation,
            combined_helper.scaled_bounds(problem),
            trust_scale,
        )
        least_squares_options["max_nfev"] = combined_budget
        result = opt.least_squares(problem, **least_squares_options)
        x_opt = np.asarray(result.x, dtype=float)
        combined_helper.report(
            f"combined warm-start stage {max_mode}", problem, x_opt
        )
        final_input = problem.input_from_scaled_parameters(x_opt)
        optimized_profiles = combined_helper.profile_parameter_values(problem, x_opt)
        optimized_config = combined_helper.postprocess_config_with_profiles(
            forward_config, optimized_profiles
        )
        stage_input = out_dir / (
            f"input.QI_neopax_geometry_profiles_warm_start_stage_m{max_mode}"
        )
        final_input.to_indata(stage_input)
        print(f"wrote {stage_input}", flush=True)
        current_input = stage_input
        current_config = optimized_config
        final_problem = problem
        final_result = result

    required = (
        nominal_profiles,
        optimized_profiles,
        optimized_config,
        final_input,
        final_problem,
        final_result,
    )
    if any(value is None for value in required):
        raise RuntimeError("No combined warm-start stage was executed.")

    initial_config = combined_helper.postprocess_config_with_profiles(
        forward_config, nominal_profiles
    )
    geometry_warm_cost = float(geometry_phase["result"].cost)
    warm_cost_difference = float(combined_initial_cost) - geometry_warm_cost
    summary = {
        "method": "geometry_warm_start_then_combined_scipy_least_squares_TRF",
        "seed_input": str(seed_input),
        "transport_config": str(geometry_example.TRANSPORT_CONFIG),
        "database_grid": [
            int(args.database_n_theta),
            int(args.database_n_phi),
            int(args.database_n_xi),
        ],
        "geometry_max_mode_schedule": list(max_modes),
        "combined_max_mode_schedule": list(combined_modes),
        "geometry_phase": {
            "max_nfev": geometry_budget,
            "parameter_labels": list(geometry_phase["problem"].parameter_labels),
            "initial_cost": float(geometry_phase["initial_cost"]),
            "cost": float(geometry_phase["result"].cost),
            "optimality": float(geometry_phase["result"].optimality),
            "nfev": int(geometry_phase["result"].nfev),
            "status": int(geometry_phase["result"].status),
            "message": str(geometry_phase["result"].message),
            "x_scaled": np.asarray(
                geometry_phase["result"].x, dtype=float
            ).tolist(),
            "accepted_geometry_input": str(geometry_phase["warm_path"]),
            "geometry_coordinate_baseline_input": str(
                geometry_phase["baseline_input"]
            ),
        },
        "combined_phase": {
            "max_nfev": combined_budget,
            "parameter_labels": list(final_problem.parameter_labels),
            "profile_coordinate_mode": str(args.profile_coordinate_mode),
            "profile_scale_mode": "nominal",
            "profile_trust_multiplier": float(args.profile_trust_multiplier),
            "geometry_trust_multiplier": 1.0,
            "geometry_coordinate_origin": "preserved_phase_1_baseline_and_scales",
            "initial_cost_at_geometry_warm_start": float(combined_initial_cost),
            "initial_cost_minus_geometry_final_cost": warm_cost_difference,
            "cost": float(final_result.cost),
            "optimality": float(final_result.optimality),
            "nfev": int(final_result.nfev),
            "status": int(final_result.status),
            "message": str(final_result.message),
            "x_scaled": np.asarray(final_result.x, dtype=float).tolist(),
        },
        "initial_physical_profiles": nominal_profiles,
        "optimized_physical_profiles": optimized_profiles,
        "objectives": {
            "max_er": bool(args.max_er),
            "root_objectives": bool(args.root_objectives),
            "transition_location": bool(args.transition_location_objective),
            "bootstrap_penalty": bool(args.bootstrap_penalty),
            "net_power": bool(args.net_power),
        },
    }
    summary_path = out_dir / "optimization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}", flush=True)

    combined_helper.write_outputs(
        initial_input=geometry_phase["initial_input"],
        optimized_input=final_input,
        initial_config=initial_config,
        optimized_config=optimized_config,
        initial_profiles=nominal_profiles,
        optimized_profiles=optimized_profiles,
        out_dir=out_dir,
        seed_input=seed_input,
        make_initial_plots=args.initial_plots,
        physical_pitches=physical_pitches,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
