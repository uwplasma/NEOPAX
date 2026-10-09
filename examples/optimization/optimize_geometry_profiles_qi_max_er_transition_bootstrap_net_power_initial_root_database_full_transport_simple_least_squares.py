#!/usr/bin/env python
"""Simple ESS-geometry plus nominal-profile full-transport least squares.

This standalone experiment deliberately contains no custom block solver.  It
uses the validated ESS geometry coordinates, all six nominally scaled
analytical-profile coordinates, and the established
``NEOPAX.optimization.least_squares`` SciPy-TRF path.  A single configurable
SciPy trust multiplier is applied only to the profile block; every geometry
entry remains exactly one.  Profile coordinates are absolute nominal
coordinates, so their seed values are exactly one while the ESS geometry
deltas start at zero.

The script exists as a clean reference for deciding how to increase profile
motion without changing the validated geometry/objective implementation.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Preserve the validated geometry-only import/initialization order.
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
    / "geometry_profiles_qi_max_er_transition_bootstrap_net_power_initial_root_"
    "database_full_transport_simple_least_squares_optimization"
)

PROFILE_PARAMETERS = (
    "n0,T0,density_shape_power,temperature_shape_power,"
    "density_shape_alpha,temperature_shape_alpha"
)
PROFILE_SCALE_MODE = "nominal"
PROFILE_COORDINATE_MODE = "absolute"
# The earlier fully Jacobian-scaled experiment gave the profile block about
# 3.7 times the median trust scale of the ESS geometry block.  Reproduce only
# that block-level effect here: no individual geometry or profile column is
# Jacobian-scaled, and the geometry block remains exactly unit-scaled.
PROFILE_TRUST_MULTIPLIER = 3.7
PROFILE_PHYSICAL_LOWER = {
    "n0": 0.6,
    "T0": 5.0,
    "density_shape_power": 0.1,
    "temperature_shape_power": 0.1,
    "density_shape_alpha": 0.1,
    "temperature_shape_alpha": 0.1,
}
PROFILE_PHYSICAL_UPPER = {
    "n0": 10.0,
    "T0": 25.0,
    "density_shape_power": 12.0,
    "temperature_shape_power": 12.0,
    "density_shape_alpha": 12.0,
    "temperature_shape_alpha": 12.0,
}


def parser() -> argparse.ArgumentParser:
    # Start from the geometry-only CLI itself.  This script adds profile
    # columns to its problem; it does not pass through the older combined
    # optimization script.
    out = geometry_example.parser()
    out.description = __doc__
    out.set_defaults(out_dir=OUT_DIR)
    out.add_argument(
        "--profile-trust-multiplier",
        type=float,
        default=PROFILE_TRUST_MULTIPLIER,
        help=(
            "Fixed SciPy trust-region x_scale applied only to profile "
            "coordinates. Geometry coordinates always retain x_scale=1. "
            "Use 1 to reproduce the previous unit-metric baseline."
        ),
    )
    return out


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.max_nfev) < 1:
        raise ValueError("--max-nfev must be positive.")
    if not np.isfinite(float(args.profile_trust_multiplier)) or float(
        args.profile_trust_multiplier
    ) <= 0.0:
        raise ValueError("--profile-trust-multiplier must be finite and positive.")
    for name in ("database_n_theta", "database_n_phi", "database_n_xi"):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    config = geometry_example.transport_config(args)
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
        "accepted_step_limit": (
            geometry_example.FULL_TRANSPORT_ACCEPTED_STEP_LIMIT
        ),
        "reverse_segment_length": geometry_example.REVERSE_SEGMENT_LENGTH,
        "max_reverse_accepted_steps": (
            geometry_example.MAX_REVERSE_ACCEPTED_STEPS
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
        "print_final_softmax_er": geometry_example.PRINT_FINAL_SOFTMAX_ER,
        "reverse_stage_mode": geometry_example.REVERSE_STAGE_MODE,
        "qi_maxj_settings": geometry_example.qi_maxj_backend_settings(
            physical_pitches
        ),
        "include_profiles": True,
        "profile_parameters": PROFILE_PARAMETERS,
        "profile_scale_mode": PROFILE_SCALE_MODE,
        "profile_coordinate_mode": PROFILE_COORDINATE_MODE,
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
        geometry_example.active_terms(args),
        **kwargs,
    )


def optimizer_trust_region_x_scale(problem, profile_multiplier: float):
    """Scale only profile trust coordinates; preserve unit ESS geometry."""

    multiplier = float(profile_multiplier)
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("profile_multiplier must be finite and positive.")
    labels = tuple(problem.parameter_labels)
    profile_names = frozenset(PROFILE_PARAMETERS.split(","))
    return np.asarray(
        [multiplier if label in profile_names else 1.0 for label in labels],
        dtype=float,
    )


def _least_squares_options(args, initial_evaluation, bounds, x_scale):
    """Return only the established SciPy-TRF options for this baseline."""

    x_scale_array = np.asarray(x_scale, dtype=float)
    return {
        "max_nfev": int(args.max_nfev),
        "ftol": geometry_example.FTOL,
        "xtol": geometry_example.XTOL,
        "verbose": 1,
        "iteration_reporter": geometry_example.iteration_diagnostics,
        "initial_evaluation": initial_evaluation,
        "bounds": bounds,
        "x_scale": x_scale_array,
    }


def profile_parameter_values(problem, x) -> dict[str, float]:
    values = np.asarray(
        jax.device_get(problem.profile_values_from_scaled_parameters(x)),
        dtype=float,
    )
    return dict(
        zip(PROFILE_PARAMETERS.split(","), values.tolist(), strict=True)
    )


def postprocess_config_with_profiles(base_config, profile_values):
    """Return the ordinary forward config with the selected profiles.

    The optimization builder intentionally silences Radau/initial-root debug
    output.  Post-processing must instead start from the original geometry
    example config, preserving its progress/iteration flags, and change only
    the active analytical-profile values.
    """

    config = copy.deepcopy(base_config)
    profiles = config.setdefault("profiles", {})
    for name, value in profile_values.items():
        profiles[name] = float(value)
    general = config.setdefault("general", {})
    if str(general.get("device", "auto")).strip().lower() == "default":
        general["device"] = "auto"
    return config


def scaled_bounds(problem):
    """Bound only profile variables; every ESS geometry variable is free."""

    scales = np.asarray(jax.device_get(problem.x_scale), dtype=float)
    lower, upper = [], []
    for label, scale in zip(problem.parameter_labels, scales, strict=True):
        lower.append(PROFILE_PHYSICAL_LOWER.get(label, -np.inf) / scale)
        upper.append(PROFILE_PHYSICAL_UPPER.get(label, np.inf) / scale)
    return np.asarray(lower), np.asarray(upper)


class TrialSavingProblem:
    """Save each unique trial without changing the delegated evaluation."""

    def __init__(self, problem, out_dir: Path):
        self.problem = problem
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._seen = set()
        self._next_index = 0

    def __getattr__(self, name):
        return getattr(self.problem, name)

    def evaluate(self, scaled_parameter_values=None):
        x = self.x0 if scaled_parameter_values is None else jnp.asarray(
            scaled_parameter_values, dtype=jnp.float64
        )
        host = np.ascontiguousarray(
            np.asarray(jax.device_get(x), dtype=np.float64)
        )
        key = (host.shape, host.tobytes())
        if key not in self._seen:
            index = self._next_index
            self._next_index += 1
            input_path = self.out_dir / (
                "input.QI_neopax_database_full_transport_combined_"
                f"eval_{index:04d}"
            )
            profile_path = self.out_dir / f"profile_parameters_eval_{index:04d}.json"
            self.problem.input_from_scaled_parameters(x).to_indata(input_path)
            profile_path.write_text(
                json.dumps(profile_parameter_values(self.problem, x), indent=2),
                encoding="utf-8",
            )
            self._seen.add(key)
            print(f"wrote {input_path}", flush=True)
            print(f"wrote {profile_path}", flush=True)
        return self.problem.evaluate(x)


def report(tag, problem, x):
    evaluation = problem.evaluate(x)
    residuals = np.asarray(jax.device_get(evaluation.residuals), dtype=float)
    jacobian = np.asarray(jax.device_get(evaluation.jacobian), dtype=float)
    print(
        f"[{tag}] elapsed_s={evaluation.elapsed_s:.3f} "
        f"{geometry_example.iteration_diagnostics(evaluation)}",
        flush=True,
    )
    print(f"  physical_profiles={profile_parameter_values(problem, x)}", flush=True)
    print(f"  residual_norm={np.linalg.norm(residuals):.6e}", flush=True)
    print(f"  jacobian_shape={jacobian.shape}", flush=True)
    return evaluation


def write_outputs(
    *,
    initial_input,
    optimized_input,
    initial_config,
    optimized_config,
    initial_profiles,
    optimized_profiles,
    out_dir: Path,
    seed_input: Path,
    make_initial_plots: bool,
    physical_pitches,
):
    """Run the same fresh post-processing solves as the established cases."""

    out_dir.mkdir(parents=True, exist_ok=True)
    initial_input.to_indata(out_dir / seed_input.name)
    optimized_path = (
        out_dir / "input.QI_neopax_database_full_transport_combined_optimized"
    )
    optimized_input.to_indata(optimized_path)
    print(f"wrote {optimized_path}", flush=True)

    for label, values in (
        ("initial", initial_profiles),
        ("optimized", optimized_profiles),
    ):
        path = out_dir / label / f"profile_parameters_{label}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(values, indent=2), encoding="utf-8")
        print(f"wrote {path}", flush=True)

    if make_initial_plots:
        geometry_example.write_geometry_artifacts(
            initial_input,
            "initial",
            out_dir,
            physical_pitches=physical_pitches,
        )
        # This is a new forward transport solve using the initial profile
        # configuration, not a cached optimization evaluation.
        geometry_example.write_transport_report(
            initial_input,
            "initial",
            initial_config,
            out_dir,
        )

    geometry_example.write_geometry_artifacts(
        optimized_input,
        "optimized",
        out_dir,
        physical_pitches=physical_pitches,
    )
    # This is a second new forward solve using both the optimized boundary and
    # optimized analytical profiles.  It writes the usual HDF5/transport plots
    # and the momentum-corrected bootstrap-current evolution CSV/PNG.
    geometry_example.write_transport_report(
        optimized_input,
        "optimized",
        optimized_config,
        out_dir,
    )

    if make_initial_plots:
        # Importing this plotting-only helper after both transport solves keeps
        # it outside the optimization initialization and trajectory.
        from examples.optimization import (
            optimize_geometry_profiles_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport
            as profile_output,
        )

        profile_output.write_initial_optimized_profile_comparison(out_dir)


def main() -> int:
    args = parser().parse_args()
    _validate_args(args)
    seed_input = args.seed_input.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    current_input = seed_input
    forward_config = geometry_example.transport_config(args)
    current_config = copy.deepcopy(forward_config)
    frozen_physical_pitches = geometry_example.PHYSICAL_J_PITCHES
    max_modes = (
        (int(geometry_example.MAX_MODE_SCHEDULE),)
        if np.isscalar(geometry_example.MAX_MODE_SCHEDULE)
        else tuple(int(value) for value in geometry_example.MAX_MODE_SCHEDULE)
    )
    initial_input = initial_config = initial_profiles = None
    optimized_input = optimized_config = optimized_profiles = None
    last_problem = last_result = None

    for max_mode in max_modes:
        print(
            "\n===== simple SciPy least-squares: ESS geometry + nominal "
            f"profiles, max_mode={max_mode}, grid=({args.database_n_theta},"
            f"{args.database_n_phi},{args.database_n_xi}), "
            f"J_backend={geometry_example.QI_MAXJ_BACKEND} =====",
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
            geometry_example.QI_MAXJ_BACKEND.strip().lower() == "physical"
            and frozen_physical_pitches is None
        ):
            frozen_physical_pitches = tuple(
                float(value)
                for value in problem.context.qi_maxj_physical_pitches
            )
        problem = TrialSavingProblem(
            problem,
            out_dir / f"simple_least_squares_inputs_m{max_mode}",
        )
        x0 = np.asarray(jax.device_get(problem.x0), dtype=float)
        profile_mask = np.asarray(
            [
                label in PROFILE_PHYSICAL_LOWER
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
            initial_profiles = profile_parameter_values(problem, x0)
            initial_config = postprocess_config_with_profiles(
                forward_config, initial_profiles
            )

        print(
            f"[setup] parameter_count={problem.parameter_count} "
            f"parameters={list(problem.parameter_labels)} "
            "geometry_coordinates=ESS_delta "
            "profile_coordinates=absolute_nominal "
            "optimizer=scipy_least_squares_TRF "
            f"profile_trust_multiplier={args.profile_trust_multiplier:.8g} "
            "geometry_trust_multiplier=1",
            flush=True,
        )
        initial_evaluation = report("initial", problem, x0)
        bounds = scaled_bounds(problem)
        trust_region_x_scale = optimizer_trust_region_x_scale(
            problem, args.profile_trust_multiplier
        )
        if not np.all(trust_region_x_scale[~profile_mask] == 1.0):
            raise AssertionError("Every ESS geometry trust multiplier must be one.")
        last_result = opt.least_squares(
            problem,
            **_least_squares_options(
                args,
                initial_evaluation,
                bounds,
                trust_region_x_scale,
            ),
        )
        x_opt = np.asarray(last_result.x, dtype=float)
        report(
            f"simple combined least-squares stage {max_mode}",
            problem,
            x_opt,
        )
        optimized_input = problem.input_from_scaled_parameters(x_opt)
        optimized_profiles = profile_parameter_values(problem, x_opt)
        optimized_config = postprocess_config_with_profiles(
            forward_config, optimized_profiles
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
        "transport_config": str(geometry_example.TRANSPORT_CONFIG),
        "database_grid": [
            int(args.database_n_theta),
            int(args.database_n_phi),
            int(args.database_n_xi),
        ],
        "max_mode_schedule": list(max_modes),
        "optimizer": "scipy_least_squares_TRF",
        "optimizer_x_scale": np.asarray(
            trust_region_x_scale, dtype=float
        ).tolist(),
        "profile_trust_multiplier": float(args.profile_trust_multiplier),
        "geometry_trust_multiplier": 1.0,
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

    write_outputs(
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
