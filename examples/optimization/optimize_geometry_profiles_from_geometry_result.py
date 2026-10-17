#!/usr/bin/env python
"""Run stage 2 from a saved geometry-only result, with optional fixed geometry.

No geometry-only optimization is run. The saved geometry coordinates are
restored against their original baseline, retaining the established ESS map.
Memory diagnostics are enabled and CUDA command buffers are disabled by default.
Initial density/temperature comparisons use analytical profiles only, without
additional transport solves. Use --profile-only to freeze the saved geometry.
"""

from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


def _configure_command_buffers(argv):
    # XLA reads these flags when its backend initializes, so configure the
    # standalone process before importing JAX or either optimization example.
    if "--enable-cuda-command-buffers" not in argv:
        flags = os.environ.get("XLA_FLAGS", "").rstrip()
        os.environ["XLA_FLAGS"] = f"{flags} --xla_gpu_enable_command_buffer=".strip()


if __name__ == "__main__":
    _configure_command_buffers(sys.argv[1:])

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Match the established VMEX/NEOPAX process initialization order.
import vmex as vj  # noqa: E402,F401
from vmex import optimize as vmex_opt  # noqa: E402,F401
from NEOPAX import optimization as opt  # noqa: E402
from examples.optimization import (  # noqa: E402
    optimize_geometry_profiles_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_simple_least_squares
    as combined,
    optimize_geometry_warm_start_profiles_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport
    as warm,
)


class Stage2ArgumentParser(argparse.ArgumentParser):
    def parse_args(self, *args, **kwargs):
        parsed = super().parse_args(*args, **kwargs)
        if (parsed.replay_diagnostics is None) != (parsed.replay_trial is None):
            self.error("--replay-diagnostics and --replay-trial must be supplied together.")
        if parsed.replay_trial is not None and parsed.replay_trial < 1:
            self.error("--replay-trial must be positive.")
        if parsed.plot_profiles_only and parsed.replay_trial is not None:
            self.error("--plot-profiles-only cannot be combined with trial replay.")
        if parsed.out_dir is None:
            name = (
                "profiles_from_geometry_result" if parsed.profile_only else "combined_from_geometry_result"
            )
            if parsed.replay_trial is not None:
                name += f"_replay_trial_{parsed.replay_trial:04d}"
            parsed.out_dir = ROOT / "outputs" / name
        return parsed


def parser():
    out = Stage2ArgumentParser(parents=[combined.parser()], add_help=False)
    out.description = __doc__
    out.set_defaults(
        seed_input=None, out_dir=None,
        database_n_theta=None, database_n_phi=None, database_n_xi=None,
        profile_trust_multiplier=1.0,
        max_er=None, root_objectives=None, transition_location_objective=None,
        bootstrap_penalty=None, net_power=None,
    )
    out.add_argument("--geometry-result-dir", type=Path, required=True,
                     help="Output directory containing a geometry-only optimization summary.")
    out.add_argument("--diagnose-memory", action=argparse.BooleanOptionalAction, default=True)
    out.add_argument("--profile-only", action="store_true",
                     help="Optimize only the six profile DoFs; keep the saved geometry coordinates fixed.")
    out.add_argument("--plot-initial-profiles", action=argparse.BooleanOptionalAction, default=True,
                     help="Compare starting/optimized density and temperature at t=0; no extra transport solve.")
    out.add_argument("--plot-profiles-only", action="store_true",
                     help="Plot an existing stage-2 --out-dir and exit without building a physics problem.")
    out.add_argument("--replay-diagnostics", type=Path,
                     help="Existing diagnostics run directory or events.jsonl to read for a one-trial replay.")
    out.add_argument("--replay-trial", type=int,
                     help="Reevaluate this saved trial once, with reverse checks and no optimizer or initial evaluation.")
    out.add_argument("--enable-cuda-command-buffers", action="store_true",
                     help="Enable CUDA graphs; the default avoids the reported graph-instantiation OOM.")
    return out


def load_geometry_result(directory):
    """Read standalone geometry results or the geometry phase of a warm run."""

    directory = Path(directory).resolve()
    summary_path = directory / "geometry_phase_summary.json"
    if not summary_path.is_file():
        summary_path = directory / "optimization_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    phase = summary.get("geometry_phase", summary)
    modes = tuple(summary.get("geometry_max_mode_schedule", summary.get("max_mode_schedule", ())))
    if not modes:
        raise ValueError("The saved geometry result has no max-mode schedule.")
    labels = tuple(phase["parameter_labels"])
    if not labels or any(not label.startswith(("RBC:", "ZBS:")) for label in labels):
        raise ValueError("The saved result must contain only RBC/ZBS geometry DoFs.")
    coordinates = np.asarray(phase.get("x_scaled", phase.get("x")), dtype=float)
    if coordinates.shape != (len(labels),) or not np.all(np.isfinite(coordinates)):
        raise ValueError("The saved geometry coordinate vector has an invalid shape or value.")
    baseline = phase.get("geometry_coordinate_baseline_input")
    if baseline is None:
        baseline = summary.get("geometry_coordinate_baseline_input")
    if baseline is None:
        baseline = (
            summary["seed_input"] if len(modes) == 1 else str(
                directory / f"input.QI_neopax_database_full_transport_net_power_stage_m{modes[-2]}"
            )
        )
    return SimpleNamespace(
        directory=directory, summary_path=summary_path, summary=summary,
        labels=labels, coordinates=coordinates, max_mode=int(modes[-1]),
        baseline_input=Path(baseline),
        scales=phase.get("coordinate_scales"),
    )


def configure_saved_settings(args, result):
    """Use saved settings unless the caller explicitly overrides a switch/grid."""

    geometry = combined.geometry_example
    config = result.summary.get("config", {})
    neo = config.get("neoclassical", {})
    grid = result.summary.get("database_grid", [
        neo.get("ntx_scan_n_theta", geometry.DATABASE_N_THETA),
        neo.get("ntx_scan_n_zeta", geometry.DATABASE_N_PHI),
        neo.get("ntx_scan_n_xi", geometry.DATABASE_N_XI),
    ])
    for name, value in zip(("database_n_theta", "database_n_phi", "database_n_xi"), grid, strict=True):
        if getattr(args, name) is None:
            setattr(args, name, int(value))
    switches = result.summary.get("objectives", {})
    for name, key, fallback in (
        ("max_er", "max_er", geometry.USE_MAX_ER_OBJECTIVE),
        ("root_objectives", "root_objectives", geometry.USE_ER_TRANSITION_OBJECTIVES),
        ("transition_location_objective", "transition_location", geometry.USE_ER_TRANSITION_LOCATION_OBJECTIVE),
        ("bootstrap_penalty", "bootstrap_penalty", geometry.USE_BOOTSTRAP_PENALTY),
        ("net_power", "net_power", geometry.USE_NET_POWER_OBJECTIVE),
    ):
        if getattr(args, name) is None:
            setattr(args, name, bool(switches.get(key, fallback)))
    if args.seed_input is None:
        args.seed_input = result.baseline_input


def starting_point(problem, result):
    """Append nominal profiles while preserving saved geometry by label."""

    lookup = dict(zip(problem.parameter_labels, np.asarray(jax.device_get(problem.x_scale)), strict=True))
    geometry_labels = tuple(label for label in problem.parameter_labels if label not in combined.PROFILE_PARAMETERS.split(","))
    if geometry_labels != result.labels:
        raise ValueError("The saved geometry parameter layout differs from the current ESS layout.")
    scales = (
        np.asarray(result.scales, dtype=float) if result.scales is not None
        else np.asarray([lookup[label] for label in result.labels], dtype=float)
    )
    return warm._combined_starting_point(
        SimpleNamespace(parameter_labels=result.labels, x_scale=scales),
        SimpleNamespace(x=result.coordinates), problem,
    )


class ProfileOnlyProblem:
    """Expose profile columns of the same evaluation, freezing saved geometry.

    The underlying combined evaluator and objective weights are unchanged,
    including its reverse-AD work. Only the optimizer's variables and Jacobian
    columns are restricted. This isolates the effect of freeing geometry.
    """

    def __init__(self, problem, full_start):
        self.problem = problem
        self.parameter_labels = tuple(combined.PROFILE_PARAMETERS.split(","))
        self.parameter_count = len(self.parameter_labels)
        self._indices = np.asarray([problem.parameter_labels.index(name) for name in self.parameter_labels])
        self._full_start = np.asarray(full_start, dtype=float).copy()
        if self._full_start.shape != (problem.parameter_count,):
            raise ValueError("Fixed-geometry starting point has the wrong shape.")
        self.x0 = self._full_start[self._indices].copy()
        self.x_scale = np.asarray(jax.device_get(problem.x_scale))[self._indices].copy()

    def __getattr__(self, name):
        return getattr(self.problem, name)

    def full_parameters(self, x=None):
        values = self.x0 if x is None else np.asarray(x, dtype=float)
        if values.shape != (self.parameter_count,):
            raise ValueError(f"Expected {self.parameter_count} profile coordinates; got {values.shape}.")
        full = self._full_start.copy()
        full[self._indices] = values
        return full

    def evaluate(self, x=None):
        evaluation = self.problem.evaluate(self.full_parameters(x))
        jacobian = evaluation.jacobian[:, self._indices]
        result = dataclasses.replace(
            evaluation.result, jacobian=jacobian, parameter_labels=self.parameter_labels,
        )
        return dataclasses.replace(evaluation, result=result, jacobian=jacobian)

    def input_from_scaled_parameters(self, x=None):
        return self.problem.input_from_scaled_parameters(self.full_parameters(x))

    def profile_values_from_scaled_parameters(self, x=None):
        return self.problem.profile_values_from_scaled_parameters(self.full_parameters(x))

    def config_from_scaled_parameters(self, x=None):
        return self.problem.config_from_scaled_parameters(self.full_parameters(x))


def initial_profile_snapshot(problem, x):
    """Rebuild only the analytical initial n/T map already used by reverse AD."""
    from NEOPAX._reverse_ad_transport import initial_state_for_parameter_vector_compact
    from NEOPAX._state import safe_density

    state = initial_state_for_parameter_vector_compact(
        problem.profile_values_from_scaled_parameters(x),
        baseline_state=problem.baseline_state, profile_cfg=problem.config.get("profiles", {}),
        geometry=problem.runtime.geometry, number_species=problem.runtime.species.number_species,
        config=problem.config, initial_er_root_ad="off",
    )
    solver = problem.config.get("transport_solver", {})
    density_floor = solver.get("density_floor", problem.config.get("solver", {}).get("density_floor", 1e-6))
    rho, density, temperature = jax.device_get((
        problem.runtime.geometry.rho_grid, state.density,
        state.pressure / safe_density(state.density, density_floor),
    ))
    names = problem.config.get("species", {}).get("names")
    if names is None:
        names = [f"species_{i}" for i in range(density.shape[0])]
    return {"rho": np.asarray(rho), "species": tuple(names),
            "density": np.asarray(density), "temperature": np.asarray(temperature)}


def write_initial_profile_comparison(problem, initial_x, optimized_x, out_dir):
    """Plot t=0 profiles and their small differences without any physics replay."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    initial = initial_profile_snapshot(problem, initial_x)
    optimized = initial_profile_snapshot(problem, optimized_x)
    np.testing.assert_allclose(initial["rho"], optimized["rho"], rtol=0, atol=1e-12)
    rho, species = initial["rho"], initial["species"]
    png = out_dir / "initial_profiles_initial_vs_optimized.png"
    csv_path = out_dir / "initial_optimized_initial_profiles.csv"
    figure = Figure(figsize=(12, 9))
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2, sharex=True)
    quantities = (("density", r"$n$ [$10^{20}$ m$^{-3}$]", "1e20_m-3"),
                  ("temperature", r"$T$ [keV]", "keV"))
    for col, (quantity, ylabel, _) in enumerate(quantities):
        for index, name in enumerate(species):
            color = f"C{index % 10}"
            for snapshot, style in ((initial, "--"), (optimized, "-")):
                axes[0, col].plot(rho, snapshot[quantity][index], style, color=color)
            axes[1, col].plot(rho, optimized[quantity][index] - initial[quantity][index],
                              color=color, label=name)
        axes[0, col].set(title=quantity.capitalize(), ylabel=ylabel)
        axes[1, col].set(ylabel=f"Optimized − initial: {ylabel}", xlabel=r"$\rho$")
        axes[1, col].axhline(0, color="0.6", linewidth=.7)
        for axis in axes[:, col]:
            axis.grid(alpha=.25)
    handles = [Line2D([], [], color=f"C{i % 10}", label=name) for i, name in enumerate(species)]
    handles += [Line2D([], [], color="black", linestyle=style, label=label)
                for label, style in (("Starting case", "--"), ("Optimized case", "-"))]
    figure.legend(handles=handles, loc="lower center", ncol=3)
    figure.suptitle("Initial profiles (t = 0): starting and optimized cases")
    figure.tight_layout(rect=(0, .09, 1, .96))
    figure.savefig(png, dpi=180)
    records = {"rho": rho}
    for quantity, _, units in quantities:
        for label, snapshot in (("initial", initial), ("optimized", optimized)):
            for index, name in enumerate(species):
                records[f"{label}_{name}_{quantity}_{units}"] = snapshot[quantity][index]
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(records)
        writer.writerows(zip(*records.values(), strict=True))
    print(f"wrote {png}", flush=True)
    print(f"wrote {csv_path}", flush=True)
    return {"png": str(png), "csv": str(csv_path)}


def plot_saved_initial_profiles(out_dir, fallback_config):
    """Plot completed runs, including summaries written before plot support."""
    import jax.numpy as jnp
    from NEOPAX._state import TransportState

    summary = json.loads((out_dir / "optimization_summary.json").read_text(encoding="utf-8"))
    config = summary.get("config", fallback_config)
    initial = summary.get("initial_physical_profiles")
    if initial is None:
        initial_path = out_dir / "profile_parameters_initial.json"
        initial = (
            json.loads(initial_path.read_text(encoding="utf-8")) if initial_path.is_file()
            else dict(zip(combined.PROFILE_PARAMETERS.split(","),
                          np.asarray(jax.device_get(opt._profile_values_from_config(config, jnp.float64))).tolist(),
                          strict=True))
        )
    optimized = summary.get("optimized_physical_profiles")
    if optimized is None:
        optimized = json.loads((out_dir / "profile_parameters_optimized.json").read_text(encoding="utf-8"))
    n_radial = int(config.get("geometry", {}).get("n_radial", 51))
    species_cfg = config.get("species", {})
    n_species = int(species_cfg.get("n_species", len(species_cfg.get("names", ("e", "D", "T")))))
    # Realtime VMEC uses uniform rho cell centers. Analytical n/T depends only
    # on r_grid / r_grid_half[-1], so absolute minor radius cancels exactly.
    faces = jnp.linspace(0., 1., n_radial + 1)
    rho = .5 * (faces[:-1] + faces[1:])
    geometry = SimpleNamespace(r_grid=rho, r_grid_half=faces, rho_grid=rho)
    problem = SimpleNamespace(
        config=config,
        runtime=SimpleNamespace(geometry=geometry, species=SimpleNamespace(number_species=n_species)),
        baseline_state=TransportState(density=jnp.zeros((n_species, n_radial)),
                                      pressure=jnp.ones((n_species, n_radial)), Er=jnp.zeros(n_radial)),
        profile_values_from_scaled_parameters=lambda values: jnp.asarray(values),
    )
    names = combined.PROFILE_PARAMETERS.split(",")
    return write_initial_profile_comparison(
        problem, [initial[name] for name in names], [optimized[name] for name in names], out_dir,
    )


def load_replay_trial(path, trial, problem):
    """Read successful or failed coordinates from existing diagnostics."""
    path = Path(path)
    if path.is_dir():
        path /= "events.jsonl"
    setup = None
    selected = None
    with path.open(encoding="utf-8") as source:
        for line in source:
            event = json.loads(line)
            if event["event"] == "setup":
                setup = event
            if event["event"] in {"trial", "failed_trial"} and event["trial"] == trial:
                selected = event
                break
    if setup is None or selected is None:
        raise ValueError(f"Missing setup or trial {trial} in {path}.")
    if tuple(setup["parameter_labels"]) != tuple(problem.parameter_labels):
        raise ValueError("Saved replay parameter labels differ; use the original --profile-only mode.")
    if setup.get("profile_coordinate_mode") != problem.profile_coordinate_mode:
        raise ValueError("Saved replay profile coordinate mode differs from this problem.")
    np.testing.assert_allclose(setup["physical_scales"], jax.device_get(problem.x_scale), rtol=1e-12, atol=0)
    if setup.get("baseline_profile_values") is not None:
        np.testing.assert_allclose(setup["baseline_profile_values"],
                                   jax.device_get(problem.baseline_profile_values), rtol=1e-12, atol=0)
    x = np.asarray(selected["x"], dtype=float)
    if x.shape != (problem.parameter_count,) or not np.all(np.isfinite(x)):
        raise ValueError("Saved replay coordinates have an invalid shape or value.")
    return path, setup, selected, x


def replay_saved_trial(problem, *, source, trial, out_dir):
    """One full evaluation of saved coordinates, with no penalty fallback."""
    path, setup, previous, x = load_replay_trial(source, trial, problem)
    out_dir = Path(out_dir)
    sink = opt.LeastSquaresDiagnostics(
        out_dir / "replay_diagnostics", problem,
        bounds=combined.scaled_bounds(problem),
        trust_scale=setup.get("trust_scales") or 1.,
    )
    summary = {"source_events": str(path.resolve()), "source_trial": int(trial),
               "previous_event": previous["event"], "previous_reason": previous.get("reason"),
               "parameter_labels": list(problem.parameter_labels), "x": x.tolist(),
               "diagnostics_dir": str(sink.directory), "physics_evaluations": 1}
    print(f"[trial replay] source_trial={trial} previous_event={previous['event']} "
          f"parameter_count={problem.parameter_count} optimizer=skipped", flush=True)
    try:
        with sink.evaluation_scope():
            evaluation = problem.evaluate(x)
        residual, jacobian = jax.device_get((evaluation.residuals, evaluation.jacobian))
        if not (np.all(np.isfinite(residual)) and np.all(np.isfinite(jacobian))):
            np.savez(out_dir / "replay_nonfinite.npz", x=x, residuals=residual, jacobian=jacobian)
            raise FloatingPointError("Trial replay returned a nonfinite residual or Jacobian.")
        sink.evaluation(x, evaluation)
        summary.update(success=True, cost=warm._cost(evaluation), jacobian_shape=list(jacobian.shape),
                       elapsed_s=float(evaluation.elapsed_s),
                       objective_values={name: float(jax.device_get(value))
                                         for name, value in evaluation.result.objective_values.items()})
        print(f"[trial replay] finite=True cost={summary['cost']:.8e} "
              f"{combined.geometry_example.iteration_diagnostics(evaluation)}", flush=True)
    except Exception as exc:
        sink.failure(x, exc, phase="saved_trial_replay")
        summary.update(success=False, reason=str(exc), exception_type=type(exc).__name__)
        raise
    finally:
        (out_dir / "replay_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


def main():
    args = parser().parse_args()
    saved = load_geometry_result(args.geometry_result_dir)
    configure_saved_settings(args, saved)
    combined._validate_args(args)
    out_dir = args.out_dir.resolve()
    if out_dir == saved.directory:
        raise ValueError("--out-dir must differ from the saved geometry-result directory.")
    config = (
        copy.deepcopy(saved.summary["config"]) if "config" in saved.summary
        else combined.geometry_example.transport_config(args)
    )
    if args.plot_profiles_only:
        plot_saved_initial_profiles(out_dir, config)
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    config.setdefault("neoclassical", {}).update(
        ntx_scan_n_theta=args.database_n_theta,
        ntx_scan_n_zeta=args.database_n_phi,
        ntx_scan_n_xi=args.database_n_xi,
    )
    problem = combined._build_problem(
        args, config=config, vmec_input=args.seed_input.resolve(),
        max_mode=saved.max_mode,
        physical_pitches=combined.geometry_example.PHYSICAL_J_PITCHES,
    )
    full_labels = tuple(problem.parameter_labels)
    full_x0 = starting_point(problem, saved)
    if args.profile_only:
        problem = ProfileOnlyProblem(problem, full_x0)
    x0 = np.asarray(problem.x0).copy() if args.profile_only else full_x0
    trial_subdir = "replay_inputs" if args.replay_trial is not None else "combined_inputs"
    problem = combined.TrialSavingProblem(problem, out_dir / trial_subdir)
    if args.diagnose_memory:
        phase = "profiles_from_saved_geometry" if args.profile_only else "combined_from_saved_geometry"
        problem = warm.MemoryDiagnosticProblem(problem, phase)
    if args.replay_trial is not None:
        return replay_saved_trial(problem, source=args.replay_diagnostics, trial=args.replay_trial, out_dir=out_dir)
    problem.input_from_scaled_parameters(x0).to_indata(out_dir / "input.QI_combined_initial")
    initial_profiles = combined.profile_parameter_values(problem, x0)
    (out_dir / "profile_parameters_initial.json").write_text(json.dumps(initial_profiles, indent=2), encoding="utf-8")
    print(f"[setup] loaded_geometry_result={saved.summary_path} max_mode={saved.max_mode} "
          f"parameter_count={problem.parameter_count} profile_only={args.profile_only} geometry_stage=skipped "
          f"XLA_FLAGS={os.environ.get('XLA_FLAGS', '')}", flush=True)
    initial = combined.report("combined initial at saved geometry", problem, x0)
    options = combined._least_squares_options(
        args, initial, combined.scaled_bounds(problem),
        combined.optimizer_trust_region_x_scale(problem, args.profile_trust_multiplier),
    )
    result = opt.least_squares(problem, x0=x0, **options)
    profiles = combined.profile_parameter_values(problem, result.x)
    problem.input_from_scaled_parameters(result.x).to_indata(out_dir / "input.QI_combined_optimized")
    (out_dir / "profile_parameters_optimized.json").write_text(json.dumps(profiles, indent=2), encoding="utf-8")
    summary = {
        "geometry_result": str(saved.summary_path), "geometry_baseline_input": str(args.seed_input.resolve()),
        "config": config,
        "parameter_labels": list(problem.parameter_labels), "x0": x0.tolist(),
        "x": np.asarray(result.x).tolist(), "initial_cost": warm._cost(initial),
        "cost": float(result.cost), "nfev": int(result.nfev),
        "status": int(result.status), "message": str(result.message),
        "optimization_mode": "profiles_only" if args.profile_only else "geometry_and_profiles",
        "full_parameter_labels": list(full_labels), "full_x0": full_x0.tolist(),
        "full_x": (problem.full_parameters(result.x) if args.profile_only else np.asarray(result.x)).tolist(),
        "initial_physical_profiles": initial_profiles,
        "optimized_physical_profiles": profiles, "XLA_FLAGS": os.environ.get("XLA_FLAGS", ""),
    }
    (out_dir / "optimization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.plot_initial_profiles:
        summary["initial_profile_comparison"] = write_initial_profile_comparison(problem, x0, result.x, out_dir)
        (out_dir / "optimization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[combined result] initial_cost={summary['initial_cost']:.8e} cost={result.cost:.8e} "
          f"nfev={result.nfev} status={result.status} wrote={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
