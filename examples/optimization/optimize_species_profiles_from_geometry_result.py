#!/usr/bin/env python
"""Stage 2 on a saved optimized geometry with independent species profiles.

Default e/D/T temperature and D/T density amplitude/shape DOFs give 15
variables. Electron density follows quasineutrality; initially absent helium
and frozen temperatures remain fixed. Uses the existing persistent full-time
transport reverse AD and bounded SciPy least squares, with the geometry-only
root/bootstrap/net-power targets and default weights. CLI weight overrides
allow the transport objectives to be rebalanced for a separate experiment.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

if __name__ == "__main__" and "--enable-cuda-command-buffers" not in sys.argv:
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "").rstrip() +
                               " --xla_gpu_enable_command_buffer=").strip()

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import numpy as np

from NEOPAX import optimization as opt
from NEOPAX._reverse_ad_parameters import ProfileParameterSpec
from NEOPAX._reverse_ad_transport import initial_state_for_parameter_vector
from NEOPAX._species_profile_parameters import species_index, species_profile_values
from examples.optimization import optimize_geometry_profiles_from_geometry_result as stage2
from examples.benchmarks import benchmark_er_root_candidates_optimization_start as baseline
from examples.benchmarks.audit_saved_geometry_root_baseline import compare_saved_transport

geometry = stage2.combined.geometry_example
DENSITY = ("n0", "density_shape_power", "density_shape_alpha")
TEMPERATURE = ("T0", "temperature_shape_power", "temperature_shape_alpha")


def density_multiplier(config, index):
    names = config["species"]["names"]
    factor = 1.
    for key in ("c_density", "n_scale"):
        raw = np.asarray(config.get("profiles", {}).get(key, 1.), dtype=float)
        if raw.ndim == 0:
            factor *= float(raw)
        elif len(raw) == len(names):
            factor *= float(raw[index])
        elif len(raw) == len(names) - 1:
            electron = names.index("e")
            factor *= 1. if index == electron else float(raw[index - (index > electron)])
        else:
            raise ValueError(f"{key} has an incompatible species layout.")
    return factor


def independent_parameters(config, requested=None):
    names = tuple(config["species"]["names"])
    charges = config["species"]["charge_qp"]
    active = config.get("equations", {}).get("toggle_temperature", [True] * len(names))
    if "e" not in names or charges[names.index("e")] >= 0:
        raise ValueError("This example requires an electron species named 'e' with negative charge.")
    labels = ([item.strip() for item in requested.split(",") if item.strip()] if requested else
              [f"{base}[{name}]" for i, name in enumerate(names) if active[i] for base in TEMPERATURE] +
              [f"{base}[{name}]" for i, name in enumerate(names)
               if charges[i] > 0 and density_multiplier(config, i) != 0 for base in DENSITY])
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("Choose at least one unique species profile parameter.")
    for label in labels:
        spec = ProfileParameterSpec(label)
        if spec.species is None:
            raise ValueError("Use species labels such as T0[e] or n0[D].")
        index = species_index(spec.species, config)
        if spec.base_name in DENSITY and (charges[index] <= 0 or density_multiplier(config, index) == 0):
            raise ValueError(f"{label} is dependent or initially absent; select a populated ion density.")
        if spec.base_name in TEMPERATURE and not active[index]:
            raise ValueError(f"{label} is a prescribed frozen temperature, not an initial-temperature DOF.")
    return tuple(labels)


def nonnegative_weight(value):
    weight = float(value)
    if not np.isfinite(weight) or weight < 0.:
        raise argparse.ArgumentTypeError("weight must be finite and non-negative")
    return weight


def parser():
    out = stage2.parser()
    out.description = __doc__
    out.set_defaults(out_dir=ROOT / "outputs/species_profiles_from_geometry_result",
                     profile_only=True, max_reverse_accepted_steps=None)
    out.add_argument("--species-profile-parameters", help="Optional comma-separated indexed DOFs; default 15 for e/D/T/He.")
    out.add_argument("--plan-only", action="store_true", help="Read saved inputs and print DOFs/targets; build no runtime.")
    out.add_argument("--root-weight", type=nonnegative_weight,
                     help="Override both root-position and strength-deficit weights; default keeps the geometry example's weights.")
    out.add_argument("--bootstrap-weight", type=nonnegative_weight,
                     help="Override the bootstrap-current penalty weight.")
    out.add_argument("--net-power-weight", type=nonnegative_weight,
                     help="Override the net-power weight (residual is in MW/m^3).")
    return out


def prepare(args):
    saved = stage2.load_geometry_result(args.geometry_result_dir)
    stage2.configure_saved_settings(args, saved)
    cli = SimpleNamespace(geometry_result_dir=args.geometry_result_dir, seed_input=None,
        database_n_theta=args.database_n_theta, database_n_phi=args.database_n_phi,
        database_n_xi=args.database_n_xi, transport_max_steps=args.transport_max_steps,
        max_reverse_accepted_steps=args.max_reverse_accepted_steps, t_final=None, inner_radius=None)
    config, _, metadata = baseline.prepare_case(cli)
    args.max_reverse_accepted_steps = int(metadata["max_reverse_accepted_steps"])
    labels = independent_parameters(config, args.species_profile_parameters)
    config.setdefault("optimization", {}).update(species_profile_parameters=list(labels),
                                                species_profile_quasineutral=True)
    overrides = {
        "Er_transition_location_moment": args.root_weight,
        "Er_transition_strength_deficit": args.root_weight,
        "bootstrap_current_penalty": args.bootstrap_weight,
        "net_total_power_volume_average_mw_m3": args.net_power_weight,
    }
    terms = []
    for term in opt._normalize_initial_er_root_least_squares_terms(geometry.active_terms(args)):
        if term.objective.family == "transport":
            weight = overrides.get(term.label or term.objective.name)
            terms.append(term if weight is None else replace(term, weight=weight))
    terms = tuple(terms)
    values = np.asarray(species_profile_values(config), dtype=float)
    lower = np.asarray([stage2.combined.PROFILE_PHYSICAL_LOWER[ProfileParameterSpec(p).base_name] for p in labels])
    upper = np.asarray([stage2.combined.PROFILE_PHYSICAL_UPPER[ProfileParameterSpec(p).base_name] for p in labels])
    if not np.all(np.isfinite(values)) or np.any(values < lower) or np.any(values > upper):
        raise ValueError("Saved initial profile parameters must lie inside their physical bounds.")
    return config, labels, terms, lower, upper, metadata


def write_json(path, value):
    def encode(item):
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, (jax.Array, np.ndarray, np.generic)):
            return np.asarray(jax.device_get(item)).tolist()
        raise TypeError(f"Unsupported saved value: {type(item).__name__}")
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=encode), encoding="utf-8")


def initial_profiles(problem, x):
    state = initial_state_for_parameter_vector(problem._scaled_to_physical(x),
        config=problem.config, profile_cfg=problem.profile_cfg, runtime=problem.runtime,
        baseline_state=problem.baseline_state, initial_er_root_ad="off")
    return {key: np.asarray(jax.device_get(getattr(state, key))) for key in ("density", "temperature")}


def save_initial_comparison(problem, first, last, out, *, plot=True):
    before, after = initial_profiles(problem, first), initial_profiles(problem, last)
    rho = np.asarray(jax.device_get(problem.runtime.geometry.rho_grid))
    names = problem.config["species"]["names"]
    with (out / "initial_optimized_initial_profiles.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["rho", *[f"{field}_{name}_{point}" for field in before for name in names for point in ("initial", "optimized")]])
        for j, radius in enumerate(rho):
            writer.writerow([radius, *[data[field][i, j] for field in before for i in range(len(names)) for data in (before, after)]])
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for axis, field, unit in zip(axes, before, ("1e20 m^-3", "keV"), strict=True):
            for i, name in enumerate(names):
                line, = axis.plot(rho, before[field][i], "--", label=f"{name} initial")
                axis.plot(rho, after[field][i], color=line.get_color(), label=f"{name} optimized")
            axis.set(xlabel="rho", ylabel=f"Initial {field} [{unit}]")
            axis.legend(fontsize=7)
        fig.savefig(out / "initial_profiles_initial_vs_optimized.png", dpi=150)
        plt.close(fig)


def main(argv=None):
    args = parser().parse_args(argv)
    if args.replay_trial is not None or args.plot_profiles_only:
        raise ValueError("Use the original stage-2 script for its replay/plot-only modes.")
    config, labels, terms, lower, upper, metadata = prepare(args)
    manifest = dict(geometry=metadata, parameter_labels=labels,
        initial_parameters=dict(zip(labels, np.asarray(species_profile_values(config)).tolist(), strict=True)),
        physical_lower_bounds=lower.tolist(), physical_upper_bounds=upper.tolist(),
        electron_density="derived from ion charge densities", geometry_fixed=True,
        profile_coordinate_mode=args.profile_coordinate_mode,
        active_terms=[dict(label=getattr(term.objective, "name", ""), target=term.target, weight=term.weight)
                     for term in opt._normalize_initial_er_root_least_squares_terms(terms)])
    if args.plan_only:
        print(json.dumps(manifest, indent=2))
        return 0
    out = args.out_dir.resolve()
    if out == args.geometry_result_dir.resolve() or (out / "optimization_summary.json").exists():
        raise ValueError("Use a fresh output directory distinct from the saved geometry result.")
    stage2.combined._validate_args(args)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "case_manifest.json", manifest)
    shutil.copy2(metadata["seed_input"], out / "input.QI_fixed_optimized_geometry")
    observing = dict(evaluation=0, x=None, baseline_checked=False)

    def observe(state, runtime, final_time):
        rho, er = (np.asarray(jax.device_get(value)) for value in (runtime.geometry.rho_grid, state.Er))
        time = float(jax.device_get(final_time))
        if not observing["baseline_checked"]:
            parity = compare_saved_transport(args.geometry_result_dir,
                dict(terminal_rho=rho, terminal_Er=er, final_time=time),
                dict(inner_radius=args.er_transition_rho_min, outer_radius=args.er_transition_rho_max,
                     target=args.er_transition_rho_target))
            write_json(out / "baseline_saved_forward_parity.json", parity)
            if parity["available"] and not parity["matches"]:
                raise ValueError("Expanded-DOF baseline does not reproduce the saved transport field; stopped before reverse AD.")
            observing["baseline_checked"] = True
        np.savez(out / f"terminal_state_eval_{observing['evaluation']:04d}.npz", rho=rho, Er=er,
                 density=np.asarray(jax.device_get(state.density)), temperature=np.asarray(jax.device_get(state.temperature)),
                 final_time=time, x=observing["x"], parameter_labels=np.asarray(labels))

    problem = opt.full_transport_profile_least_squares_problem(config, terms,
        profile_parameters=labels, profile_scale_mode="nominal", profile_coordinate_mode=args.profile_coordinate_mode,
        device=geometry.SOLVER_DEVICE, accepted_step_limit=None,
        reverse_segment_length=geometry.REVERSE_SEGMENT_LENGTH,
        max_reverse_accepted_steps=args.max_reverse_accepted_steps,
        initial_er_root_ad="jax_selected_root", radau_jacobian_reuse_mode="legacy",
        er_transition_left_index=args.er_transition_left_index, er_transition_right_index=args.er_transition_right_index,
        er_transition_rho_min=args.er_transition_rho_min, er_transition_rho_max=args.er_transition_rho_max,
        er_transition_rho_target=args.er_transition_rho_target,
        er_transition_temperature_kv_m=args.er_transition_temperature_kv_m,
        er_transition_rho_softness=args.er_transition_rho_softness,
        er_transition_softmax_beta=args.er_transition_softmax_beta,
        reverse_stage_adjoint_solve_mode="block", reverse_rhs_transpose_mode="explicit_database",
        reverse_step_bwd_mode="reduced_cotangent_call_boundary",
        reverse_stage_mode="database_full_transport_optimization", terminal_state_observer=observe)
    raw_problem = problem

    class SavingProblem:
        def __getattr__(self, name):
            return getattr(raw_problem, name)

        def evaluate(self, x=None):
            x = raw_problem.x0 if x is None else x
            observing["evaluation"] += 1
            observing["x"] = np.asarray(jax.device_get(x))
            values = np.asarray(jax.device_get(raw_problem._scaled_to_physical(x)))
            write_json(out / f"profile_parameters_eval_{observing['evaluation']:04d}.json",
                       dict(zip(labels, values.tolist(), strict=True)))
            return raw_problem.evaluate(x)

    problem = SavingProblem()
    if args.diagnose_memory:
        problem = stage2.warm.MemoryDiagnosticProblem(problem, "species_profiles_from_saved_geometry")
    x0 = np.asarray(problem.x0)
    print(f"[species-stage2] geometry_fixed=True parameter_count={len(labels)} parameters={','.join(labels)}", flush=True)
    initial = problem.evaluate(x0)
    print("[species-stage2 initial] " + geometry.iteration_diagnostics(initial), flush=True)
    scales = np.asarray(problem.x_scale)
    offsets = np.asarray(species_profile_values(config)) if args.profile_coordinate_mode == "delta" else 0.
    bounds = ((lower-offsets)/scales, (upper-offsets)/scales)
    options = dict(max_nfev=args.max_nfev, ftol=geometry.FTOL, xtol=geometry.XTOL,
        bounds=bounds, x_scale=np.full(len(labels), args.profile_trust_multiplier),
        initial_evaluation=initial, iteration_reporter=geometry.iteration_diagnostics, verbose=1)
    if args.diagnose_optimization:
        options["diagnostics_dir"] = out / "optimization_diagnostics"
    result = opt.least_squares(problem, x0=x0, **options)
    physical = np.asarray(jax.device_get(raw_problem._scaled_to_physical(result.x)))
    write_json(out / "profile_parameters_optimized.json", dict(zip(labels, physical.tolist(), strict=True)))
    write_json(out / "optimized_transport_config.json", raw_problem.config_from_scaled_parameters(result.x))
    write_json(out / "optimization_summary.json", dict(**manifest,
        config=config, x0=x0.tolist(), x=np.asarray(result.x).tolist(),
        optimized_parameters=dict(zip(labels, physical.tolist(), strict=True)),
        initial_cost=float(0.5*np.vdot(initial.residuals, initial.residuals)), cost=float(result.cost),
        nfev=int(result.nfev), status=int(result.status), message=str(result.message),
        optimality=float(result.optimality)))
    save_initial_comparison(raw_problem, x0, result.x, out, plot=args.plot_initial_profiles)
    print(f"[species-stage2 result] cost={result.cost:.9e} nfev={result.nfev} status={result.status} wrote={out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
