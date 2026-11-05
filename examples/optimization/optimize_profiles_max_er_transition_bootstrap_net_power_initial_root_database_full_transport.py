#!/usr/bin/env python
"""Standalone database full-transport optimization of plasma profiles.

The default fixed vacuum geometry is the 51-surface version of the wHe
realtime-geometry benchmark equilibrium; ``--seed-input`` can override it.
The four profile parameters
are optimized for root position/strength, bootstrap current and 300 MW net
power. Every accepted optimizer evaluation must complete ``t_final=2``.
Initial-profile comparisons use analytical maps only. ``--plot-profiles-only``
plots saved profile parameters without VMEC, NTX, transport or optimization.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from NEOPAX import optimization as opt  # noqa: E402
from NEOPAX._orchestrator import load_config, run_config  # noqa: E402


# --------------------------- user settings ---------------------------------
# Same physical equilibrium as the wHe benchmark, with VMEC NS_ARRAY=[24,51].
# The *_true input continues that same equilibrium to 101 and 201 surfaces.
SEED_INPUT = ROOT / "examples" / "inputs" / "input.QI_nfp2_newNT_opt_hires"
TRANSPORT_CONFIG = (
    ROOT / "examples" / "optimization" / "full_transport_database_vacuum_t2.toml"
)
OUT_DIR = (
    ROOT
    / "outputs"
    / "profiles_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_optimization"
)

DATABASE_N_THETA = 25
DATABASE_N_PHI = 31  # NTX calls this toroidal coordinate zeta.
DATABASE_N_XI = 64
TRANSPORT_RADIAL_POINTS = 51

PROFILE_PARAMETERS = "n0,T0,density_shape_power,temperature_shape_power"
PROFILE_SCALE_MODE = "nominal"
PROFILE_PHYSICAL_LOWER = {
    "n0": 2.0,
    "T0": 7.0,
    "density_shape_power": 0.1,
    "temperature_shape_power": 0.1,
}
PROFILE_PHYSICAL_UPPER = {
    "n0": 10.0,
    "T0": 25.0,
    "density_shape_power": 12.0,
    "temperature_shape_power": 12.0,
}

FULL_TRANSPORT_ACCEPTED_STEP_LIMIT = None
REVERSE_SEGMENT_LENGTH = 50
MAX_REVERSE_ACCEPTED_STEPS = 500
TRANSPORT_MAX_STEPS = 1000
TRANSPORT_FINAL_TIME = 2.0
REVERSE_STAGE_MODE = "database_full_transport_optimization"
PRINT_FINAL_SOFTMAX_ER = True

MAX_ER_TARGET = 25.0
ER_TRANSITION_LEFT_TARGET = 26.0
ER_TRANSITION_RIGHT_TARGET = -10.0
ER_TRANSITION_LEFT_INDEX = 25
ER_TRANSITION_RIGHT_INDEX = 26
ER_TRANSITION_RHO_TARGET = 0.514
ER_TRANSITION_RHO_MIN = 0.25
ER_TRANSITION_RHO_MAX = 0.75
ER_TRANSITION_TEMPERATURE_KV_M = 2.0
ER_TRANSITION_RHO_SOFTNESS = 0.05
ER_TRANSITION_SOFTMAX_BETA = 16.0
BOOTSTRAP_LIMIT_SCALED = 0.1
NET_POWER_TARGET_MW = 300.0
NET_POWER_REFERENCE_VOLUME_M3 = 331.0187969899648
NET_POWER_TARGET_MW_M3 = NET_POWER_TARGET_MW / NET_POWER_REFERENCE_VOLUME_M3

MAX_ER_WEIGHT = 0.5
ER_TRANSITION_LEFT_WEIGHT = 0.09
ER_TRANSITION_RIGHT_WEIGHT = 0.09
ER_TRANSITION_RHO_WEIGHT = 100.0
ER_TRANSITION_STRENGTH_TARGET = 0.90
ER_TRANSITION_STRENGTH_WEIGHT = 100.0
BOOTSTRAP_WEIGHT = 2.0
NET_POWER_WEIGHT = 1.0

# Use the coupled crossing location/existence terms by default. The older
# fixed-cell left/right terms and maximum-Er term remain explicit opt-ins.
USE_MAX_ER_OBJECTIVE = False
USE_ER_TRANSITION_OBJECTIVES = False
USE_ER_TRANSITION_LOCATION_OBJECTIVE = True
USE_BOOTSTRAP_PENALTY = True
USE_NET_POWER_OBJECTIVE = True

NFEV = 30
FTOL = 1.0e-6
XTOL = 1.0e-10
SOLVER_DEVICE = "default"
MAKE_INITIAL_TRANSPORT_REPORT = True
MAKE_OPTIMIZED_TRANSPORT_REPORT = True
PLOT_INITIAL_PROFILES = True


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description=__doc__)
    out.add_argument("--seed-input", type=Path, default=SEED_INPUT)
    out.add_argument("--out-dir", type=Path, default=OUT_DIR)
    out.add_argument("--max-nfev", type=int, default=NFEV)
    out.add_argument("--database-n-theta", type=int, default=DATABASE_N_THETA)
    out.add_argument("--database-n-phi", type=int, default=DATABASE_N_PHI)
    out.add_argument("--database-n-xi", type=int, default=DATABASE_N_XI)
    out.add_argument(
        "--plot-initial-profiles", action=argparse.BooleanOptionalAction,
        default=PLOT_INITIAL_PROFILES,
        help="Compare starting and optimized input density/temperature profiles (no transport solve).",
    )
    out.add_argument(
        "--plot-profiles-only", action="store_true",
        help="Plot existing saved profile parameters in --out-dir without running optimization or transport.",
    )
    out.add_argument(
        "--er-transition-left-index",
        type=int,
        default=ER_TRANSITION_LEFT_INDEX,
        help="Final-time Er radial-cell index for the left transition target.",
    )
    out.add_argument(
        "--er-transition-right-index",
        type=int,
        default=ER_TRANSITION_RIGHT_INDEX,
        help="Final-time Er radial-cell index for the right transition target.",
    )
    out.add_argument(
        "--max-er",
        action=argparse.BooleanOptionalAction,
        default=USE_MAX_ER_OBJECTIVE,
        help="Enable/disable the maximum-Er objective.",
    )
    out.add_argument(
        "--root-objectives",
        action=argparse.BooleanOptionalAction,
        default=USE_ER_TRANSITION_OBJECTIVES,
        help="Enable/disable the older fixed-cell Er-left/right objectives.",
    )
    out.add_argument(
        "--transition-location-objective",
        action=argparse.BooleanOptionalAction,
        default=USE_ER_TRANSITION_LOCATION_OBJECTIVE,
        help="Enable/disable the coupled final-time Er transition location/existence objectives.",
    )
    out.add_argument(
        "--er-transition-rho-target", type=float, default=ER_TRANSITION_RHO_TARGET
    )
    out.add_argument(
        "--er-transition-strength-target",
        type=float,
        default=ER_TRANSITION_STRENGTH_TARGET,
        help="Required soft-selected crossing margin in units of the Er temperature scale.",
    )
    out.add_argument(
        "--er-transition-rho-min", type=float, default=ER_TRANSITION_RHO_MIN
    )
    out.add_argument(
        "--er-transition-rho-max", type=float, default=ER_TRANSITION_RHO_MAX
    )
    out.add_argument(
        "--er-transition-temperature-kv-m",
        type=float,
        default=ER_TRANSITION_TEMPERATURE_KV_M,
    )
    out.add_argument(
        "--er-transition-rho-softness",
        type=float,
        default=ER_TRANSITION_RHO_SOFTNESS,
        help="Radial width of the soft preference around the requested transition.",
    )
    out.add_argument(
        "--er-transition-softmax-beta",
        type=float,
        default=ER_TRANSITION_SOFTMAX_BETA,
        help="Sharpness of the soft selection over all radial transition faces.",
    )
    out.add_argument(
        "--bootstrap-penalty",
        action=argparse.BooleanOptionalAction,
        default=USE_BOOTSTRAP_PENALTY,
        help="Enable/disable the bootstrap-current penalty.",
    )
    out.add_argument(
        "--net-power",
        action=argparse.BooleanOptionalAction,
        default=USE_NET_POWER_OBJECTIVE,
        help="Enable/disable the 300 MW net-power objective.",
    )
    out.add_argument(
        "--initial-transport-report",
        action=argparse.BooleanOptionalAction,
        default=MAKE_INITIAL_TRANSPORT_REPORT,
        help="Rerun NEOPAX and write the usual initial transport plots/HDF5.",
    )
    out.add_argument(
        "--optimized-transport-report",
        action=argparse.BooleanOptionalAction,
        default=MAKE_OPTIMIZED_TRANSPORT_REPORT,
        help="Rerun NEOPAX and write the usual optimized transport plots/HDF5.",
    )
    return out


def transport_config(args: argparse.Namespace) -> dict:
    config = copy.deepcopy(load_config(TRANSPORT_CONFIG))
    geometry = config.setdefault("geometry", {})
    geometry["vmec_input_file"] = str(args.seed_input.resolve())
    geometry["n_radial"] = int(TRANSPORT_RADIAL_POINTS)
    neoclassical = config.setdefault("neoclassical", {})
    neoclassical["ntx_scan_n_theta"] = int(args.database_n_theta)
    neoclassical["ntx_scan_n_zeta"] = int(args.database_n_phi)
    neoclassical["ntx_scan_n_xi"] = int(args.database_n_xi)
    solver = config.setdefault("transport_solver", {})
    solver["t_final"] = float(TRANSPORT_FINAL_TIME)
    solver["max_steps"] = int(TRANSPORT_MAX_STEPS)
    return config


def positive_part(value):
    return jnp.maximum(value, 0.0)


def smooth_positive_part(value, eps: float = 1.0e-6):
    value_arr = jnp.asarray(value)
    eps_arr = jnp.asarray(eps, dtype=value_arr.dtype)
    return 0.5 * (value_arr + jnp.sqrt(value_arr * value_arr + eps_arr * eps_arr))


bootstrap_penalty = opt.transformed_transport_objective(
    opt.transport.bootstrap_current_softmax_abs_scaled,
    lambda value: positive_part(value - BOOTSTRAP_LIMIT_SCALED),
    label="bootstrap_current_penalty",
)


def active_terms(args: argparse.Namespace):
    terms = []
    if args.max_er:
        terms.append((opt.transport.softmax_Er, MAX_ER_TARGET, MAX_ER_WEIGHT))
    if args.root_objectives:
        terms.extend(
            (
                (
                    opt.transport.Er_transition_left,
                    ER_TRANSITION_LEFT_TARGET,
                    ER_TRANSITION_LEFT_WEIGHT,
                ),
                (
                    opt.transport.Er_transition_right,
                    ER_TRANSITION_RIGHT_TARGET,
                    ER_TRANSITION_RIGHT_WEIGHT,
                ),
            )
        )
    if args.transition_location_objective:
        transition_strength_deficit = opt.transformed_transport_objective(
            opt.transport.Er_transition_strength,
            lambda strength: smooth_positive_part(
                args.er_transition_strength_target - strength
            ),
            label="Er_transition_strength_deficit",
        )
        terms.extend(
            (
                # Report the soft-selected location; the location-moment row
                # moves its best candidate while the strength row creates it.
                (opt.transport.Er_transition_rho, 0.0, 0.0),
                (
                    opt.transport.Er_transition_location_moment,
                    0.0,
                    ER_TRANSITION_RHO_WEIGHT,
                ),
                (
                    opt.transport.Er_transition_strength,
                    0.0,
                    0.0,
                ),
                (
                    transition_strength_deficit,
                    0.0,
                    ER_TRANSITION_STRENGTH_WEIGHT,
                ),
            )
        )
    if args.bootstrap_penalty:
        terms.append((bootstrap_penalty, 0.0, BOOTSTRAP_WEIGHT))
    if args.net_power:
        terms.append(
            (
                opt.transport.net_total_power_volume_average_mw_m3,
                NET_POWER_TARGET_MW_M3,
                NET_POWER_WEIGHT,
            )
        )
    if not terms:
        raise ValueError("Enable at least one full-transport objective.")
    return tuple(terms)


def iteration_diagnostics(evaluation):
    values = {
        label: float(np.asarray(jax.device_get(value), dtype=float))
        for label, value in evaluation.result.objective_values.items()
    }
    residuals = np.asarray(jax.device_get(evaluation.residuals), dtype=float)
    residual_lookup = {
        label: float(residuals[i])
        for i, label in enumerate(evaluation.result.residual_labels)
    }

    def value(*labels):
        for label in labels:
            if label in values:
                return values[label]
        return np.nan

    def component_cost(*labels):
        for label in labels:
            if label in residual_lookup:
                residual = residual_lookup[label]
                return 0.5 * residual * residual
        return np.nan

    net_power_average = value(
        "transport:net_total_power_volume_average_mw_m3",
        "net_total_power_volume_average_mw_m3",
    )
    return (
        f"total_cost={0.5 * float(np.dot(residuals, residuals)):.8e} "
        f"softmax_Er={value('transport:softmax_Er', 'softmax_Er'):.8e} "
        f"Er_cost={component_cost('transport:softmax_Er', 'softmax_Er'):.8e} "
        f"Er_left={value('transport:Er_transition_left', 'Er_transition_left'):.8e} "
        f"Er_left_cost={component_cost('transport:Er_transition_left', 'Er_transition_left'):.8e} "
        f"Er_right={value('transport:Er_transition_right', 'Er_transition_right'):.8e} "
        f"Er_right_cost={component_cost('transport:Er_transition_right', 'Er_transition_right'):.8e} "
        f"Er_transition_rho={value('transport:Er_transition_rho', 'Er_transition_rho'):.8e} "
        f"Er_transition_location_moment={value('transport:Er_transition_location_moment', 'Er_transition_location_moment'):.8e} "
        f"Er_transition_location_cost={component_cost('transport:Er_transition_location_moment', 'Er_transition_location_moment'):.8e} "
        f"Er_transition_strength={value('transport:Er_transition_strength', 'Er_transition_strength'):.8e} "
        f"Er_transition_strength_deficit={value('Er_transition_strength_deficit'):.8e} "
        f"Er_transition_strength_cost={component_cost('Er_transition_strength_deficit'):.8e} "
        f"bootstrap_penalty={value('bootstrap_current_penalty', 'transport:bootstrap_current_penalty'):.8e} "
        f"bootstrap_cost={component_cost('bootstrap_current_penalty', 'transport:bootstrap_current_penalty'):.8e} "
        f"net_power_average_MW_m3={net_power_average:.8e} "
        f"net_power_MW={net_power_average * NET_POWER_REFERENCE_VOLUME_M3:.8e} "
        f"net_power_cost={component_cost('transport:net_total_power_volume_average_mw_m3', 'net_total_power_volume_average_mw_m3'):.8e}"
    )


def physical_profile_values(problem, x):
    values = np.asarray(
        jax.device_get(jnp.asarray(x, dtype=jnp.float64) * problem.x_scale),
        dtype=float,
    )
    return dict(zip(problem.parameter_labels, values.tolist(), strict=True))


def report(tag, problem, x):
    evaluation = problem.evaluate(x)
    residuals = np.asarray(jax.device_get(evaluation.residuals), dtype=float)
    jacobian = np.asarray(jax.device_get(evaluation.jacobian), dtype=float)
    print(
        f"[{tag}] elapsed_s={evaluation.elapsed_s:.3f} "
        f"{iteration_diagnostics(evaluation)}",
        flush=True,
    )
    print(f"  physical_profiles={physical_profile_values(problem, x)}", flush=True)
    print(f"  residual_norm={float(np.linalg.norm(residuals)):.6e}", flush=True)
    print(f"  jacobian_shape={jacobian.shape}", flush=True)
    return evaluation


def scaled_profile_bounds(problem):
    scales = np.asarray(jax.device_get(problem.x_scale), dtype=float)
    lower = []
    upper = []
    for label, scale in zip(problem.parameter_labels, scales, strict=True):
        lower.append(PROFILE_PHYSICAL_LOWER.get(label, -np.inf) / scale)
        upper.append(PROFILE_PHYSICAL_UPPER.get(label, np.inf) / scale)
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


class ProfileParameterSavingProblem:
    """Save the physical profile parameters before every new evaluation."""

    def __init__(self, problem, out_dir: Path):
        self.problem = problem
        self.out_dir = out_dir
        self.evaluation_count = 0
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def __getattr__(self, name):
        return getattr(self.problem, name)

    def evaluate(self, scaled_parameter_values=None):
        values = self.x0 if scaled_parameter_values is None else scaled_parameter_values
        path = self.out_dir / f"profile_parameters_eval_{self.evaluation_count:04d}.json"
        path.write_text(
            json.dumps(physical_profile_values(self, values), indent=2),
            encoding="utf-8",
        )
        self.evaluation_count += 1
        print(f"wrote {path}", flush=True)
        return self.problem.evaluate(scaled_parameter_values)


def write_profile_parameters(problem, x, out_dir: Path, label: str):
    path = out_dir / label / f"profile_parameters_{label}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(physical_profile_values(problem, x), indent=2),
        encoding="utf-8",
    )
    print(f"wrote {path}")


def write_transport_report(problem, x, out_dir: Path, label: str):
    artifact_dir = out_dir / label / "transport"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    config = problem.config_from_scaled_parameters(x)
    output = config.setdefault("transport_output", {})
    output["transport_plot"] = True
    output["transport_write_hdf5"] = True
    output["transport_bootstrap_current_evolution"] = True
    output["transport_plot_n_times"] = -1
    output["transport_output_dir"] = str(artifact_dir)
    print(f"[transport-report] running {label} profile configuration", flush=True)
    run_config(config)
    print(f"[transport-report] wrote usual outputs in {artifact_dir}", flush=True)


def write_initial_profile_comparison(config, before, after, out_dir, *, geometry=None):
    """Compare input n/T through the normal analytical profile map only."""
    from types import SimpleNamespace
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from NEOPAX._orchestrator import _build_species, _build_state
    from NEOPAX._state import safe_density

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if geometry is None:
        # Live VMEC uses uniform rho cell centers. Absolute minor radius
        # cancels from the analytical n/T map, which depends on r/a only.
        n = int(config.get("geometry", {}).get("n_radial", 51))
        faces = jnp.linspace(0., 1., n + 1)
        rho = .5 * (faces[:-1] + faces[1:])
        geometry = SimpleNamespace(r_grid=rho, r_grid_half=faces, rho_grid=rho)
    rho = np.asarray(jax.device_get(geometry.rho_grid))
    species = _build_species(config)
    names = tuple(species.names)
    snapshots = {}
    for label, parameters in (("initial", before), ("optimized", after)):
        current = copy.deepcopy(config)
        current.setdefault("profiles", {}).update(parameters)
        state = _build_state(current, geometry, species)
        density_floor = current.get("transport_solver", {}).get("density_floor", 1.0e-6)
        density, temperature = jax.device_get((
            state.density, state.pressure / safe_density(state.density, density_floor),
        ))
        snapshots[label] = dict(density=np.asarray(density), temperature=np.asarray(temperature))

    figure = Figure(figsize=(11.5, 8.2))
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2, sharex=True)
    columns = {"rho": rho}
    for col, (quantity, unit) in enumerate((("density", r"$10^{20}$ m$^{-3}$"), ("temperature", "keV"))):
        for i, name in enumerate(names):
            color = f"C{i % 10}"
            for label, style in (("initial", "--"), ("optimized", "-")):
                values = snapshots[label][quantity][i]
                axes[0, col].plot(rho, values, style, color=color)
                columns[f"{label}_{name}_{quantity}"] = values
            axes[1, col].plot(rho, snapshots["optimized"][quantity][i]-snapshots["initial"][quantity][i],
                              color=color)
        axes[0, col].set(title=quantity.capitalize(), ylabel=f"{quantity.capitalize()} [{unit}]")
        axes[1, col].set(xlabel=r"$\rho$", ylabel=f"Optimized - initial [{unit}]")
        axes[1, col].axhline(0., color="0.6", linewidth=.7)
        for ax in axes[:, col]:
            ax.grid(alpha=.2)
    handles = [Line2D([], [], color=f"C{i % 10}", label=name) for i, name in enumerate(names)]
    handles += [Line2D([], [], color="black", linestyle=style, label=label)
                for label, style in (("Starting case", "--"), ("Optimized case", "-"))]
    figure.legend(handles=handles, loc="lower center", ncol=3)
    figure.suptitle("Input profiles (t = 0): starting and optimized cases")
    figure.tight_layout(rect=(0, .09, 1, .96))
    paths = {}
    for suffix in ("png", "pdf"):
        path = out_dir / f"initial_profiles_initial_vs_optimized.{suffix}"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        paths[suffix] = str(path)
    csv_path = out_dir / "initial_optimized_initial_profiles.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values(), strict=True))
    paths["csv"] = str(csv_path)
    print(f"[profile-comparison] wrote {paths}", flush=True)
    return paths


def plot_saved_initial_profiles(args):
    """Replot completed parameter files, including older report-crashed runs."""
    out_dir = args.out_dir.resolve()
    summary = json.loads((out_dir / "optimization_summary.json").read_text(encoding="utf-8"))
    config = summary.get("profile_plot_config")
    if config is None:
        config_path = Path(summary.get("transport_config", TRANSPORT_CONFIG))
        config = load_config(config_path if config_path.is_file() else TRANSPORT_CONFIG)
    parameters = {}
    for label in ("initial", "optimized"):
        values = summary.get(f"{label}_physical_profiles")
        if values is None:
            path = out_dir / label / f"profile_parameters_{label}.json"
            values = json.loads(path.read_text(encoding="utf-8"))
        parameters[label] = values
    return write_initial_profile_comparison(config, parameters["initial"], parameters["optimized"], out_dir)


def write_final_profile_comparison(out_dir):
    """Compare terminal density, temperature and Er from the two saved reports."""
    import h5py
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    reports = {}
    for label in ("initial", "optimized"):
        with h5py.File(out_dir / label / "transport" / "transport_solution.h5") as stream:
            reports[label] = {key:np.asarray(stream[key]) for key in ("rho", "ts", "density", "temperature", "Er")}
    first, last = reports["initial"], reports["optimized"]
    np.testing.assert_allclose(first["rho"], last["rho"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(first["ts"][-1], last["ts"][-1], rtol=0, atol=1e-12)
    names = ("e", "D", "T", "He")
    figure = Figure(figsize=(15, 4.7))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    for col, (quantity, unit) in enumerate((("density", r"$10^{20}$ m$^{-3}$"), ("temperature", "keV"))):
        for i in range(first[quantity].shape[1]):
            for label, style in (("initial", "--"), ("optimized", "-")):
                axes[col].plot(reports[label]["rho"], reports[label][quantity][-1, i], style, color=f"C{i % 10}")
        axes[col].set(title=quantity.capitalize(), xlabel=r"$\rho$", ylabel=f"{quantity.capitalize()} [{unit}]")
    for label, style in (("initial", "--"), ("optimized", "-")):
        axes[2].plot(reports[label]["rho"], reports[label]["Er"][-1], style, color="C4", label=label.capitalize())
    axes[2].axhline(0., color="0.6", linewidth=.7)
    axes[2].set(title="Electric field", xlabel=r"$\rho$", ylabel=r"$E_r$ [kV/m]")
    for ax in axes:
        ax.grid(alpha=.2)
    handles = [Line2D([], [], color=f"C{i}", label=name)
               for i,name in enumerate(names[:first["density"].shape[1]])]
    handles += [Line2D([], [], color="black", linestyle=style, label=label)
                for label, style in (("Starting case", "--"), ("Optimized case", "-"))]
    figure.legend(handles=handles, loc="lower center", ncol=6)
    figure.suptitle(f"Final transport profiles (t = {first['ts'][-1]:g} s)")
    figure.tight_layout(rect=(0, .09, 1, .96))
    paths = {}
    for suffix in ("png", "pdf"):
        path = out_dir / f"final_profiles_initial_vs_optimized.{suffix}"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        paths[suffix] = str(path)
    print(f"[profile-comparison] wrote {paths}", flush=True)
    return paths


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.plot_profiles_only:
        plot_saved_initial_profiles(args)
        return 0
    if args.max_nfev < 1:
        raise ValueError("--max-nfev must be positive.")
    for name in ("database_n_theta", "database_n_phi", "database_n_xi"):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")

    config = transport_config(args)
    n_radial = int(config.get("geometry", {}).get("n_radial", 51))
    for name in ("er_transition_left_index", "er_transition_right_index"):
        index = int(getattr(args, name))
        if not 0 <= index < n_radial:
            raise ValueError(
                f"--{name.replace('_', '-')} must be in [0, {n_radial}); "
                f"got {index}."
            )
    if not 0.0 <= args.er_transition_rho_min < args.er_transition_rho_max <= 1.0:
        raise ValueError("Er transition radial window must satisfy 0 <= min < max <= 1.")
    if not (
        args.er_transition_rho_min
        <= args.er_transition_rho_target
        <= args.er_transition_rho_max
    ):
        raise ValueError("Er transition target must lie inside the radial window.")
    if args.er_transition_strength_target <= 0.0:
        raise ValueError("--er-transition-strength-target must be positive.")
    if args.er_transition_temperature_kv_m <= 0.0:
        raise ValueError("--er-transition-temperature-kv-m must be positive.")
    if args.er_transition_rho_softness <= 0.0:
        raise ValueError("--er-transition-rho-softness must be positive.")
    if args.er_transition_softmax_beta <= 0.0:
        raise ValueError("--er-transition-softmax-beta must be positive.")
    terms = active_terms(args)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if (
        args.max_er
        and args.root_objectives
        and ER_TRANSITION_LEFT_TARGET > MAX_ER_TARGET
    ):
        print(
            "[setup] warning: Er_left target exceeds the softmax-Er target; "
            "the two objectives cannot both have zero residual.",
            flush=True,
        )

    print(
        "\n===== database full-transport profile optimization "
        f"grid=({args.database_n_theta},{args.database_n_phi},"
        f"{args.database_n_xi}) =====",
        flush=True,
    )
    print(
        f"[setup] fixed_geometry seed_input={args.seed_input.resolve()} "
        f"transport_n_radial={n_radial}",
        flush=True,
    )
    problem = opt.full_transport_profile_least_squares_problem(
        config,
        terms,
        profile_parameters=PROFILE_PARAMETERS,
        profile_scale_mode=PROFILE_SCALE_MODE,
        device=SOLVER_DEVICE,
        accepted_step_limit=FULL_TRANSPORT_ACCEPTED_STEP_LIMIT,
        reverse_segment_length=REVERSE_SEGMENT_LENGTH,
        max_reverse_accepted_steps=MAX_REVERSE_ACCEPTED_STEPS,
        initial_er_root_ad="jax_selected_root",
        er_transition_left_index=args.er_transition_left_index,
        er_transition_right_index=args.er_transition_right_index,
        er_transition_rho_min=args.er_transition_rho_min,
        er_transition_rho_max=args.er_transition_rho_max,
        er_transition_rho_target=args.er_transition_rho_target,
        er_transition_temperature_kv_m=args.er_transition_temperature_kv_m,
        er_transition_rho_softness=args.er_transition_rho_softness,
        er_transition_softmax_beta=args.er_transition_softmax_beta,
        radau_jacobian_reuse_mode="legacy",
        reverse_stage_adjoint_solve_mode="block",
        reverse_rhs_transpose_mode="explicit_database",
        reverse_stage_cotangent_mode="full",
        reverse_step_bwd_mode="reduced_cotangent_call_boundary",
        reverse_stage_adjoint_memory_mode="default",
        print_final_softmax_er=PRINT_FINAL_SOFTMAX_ER,
        reverse_stage_mode=REVERSE_STAGE_MODE,
    )
    problem = ProfileParameterSavingProblem(
        problem,
        out_dir / "profile_evaluations",
    )
    x0 = np.asarray(jax.device_get(problem.x0), dtype=float)
    print(
        f"[setup] parameter_count={problem.parameter_count} "
        f"parameters={list(problem.parameter_labels)}",
        flush=True,
    )
    print(
        "[setup] full_transport "
        f"t_final={TRANSPORT_FINAL_TIME} max_steps={TRANSPORT_MAX_STEPS} "
        f"accepted_step_limit={FULL_TRANSPORT_ACCEPTED_STEP_LIMIT} "
        f"reverse_segment_length={REVERSE_SEGMENT_LENGTH} "
        f"max_reverse_accepted_steps={MAX_REVERSE_ACCEPTED_STEPS} "
        f"Er_transition_indices=({args.er_transition_left_index},"
        f"{args.er_transition_right_index}) "
        f"Er_transition_location={args.transition_location_objective} "
        f"Er_transition_rho_target={args.er_transition_rho_target} "
        f"Er_transition_strength_target={args.er_transition_strength_target} "
        f"Er_transition_rho_window=({args.er_transition_rho_min},"
        f"{args.er_transition_rho_max}) "
        f"Er_transition_rho_softness={args.er_transition_rho_softness} "
        f"Er_transition_softmax_beta={args.er_transition_softmax_beta}",
        flush=True,
    )
    initial_evaluation = report("initial", problem, x0)
    result = opt.least_squares(
        problem,
        max_nfev=args.max_nfev,
        ftol=FTOL,
        xtol=XTOL,
        x_scale=np.ones_like(x0),
        bounds=scaled_profile_bounds(problem),
        verbose=1,
        iteration_reporter=iteration_diagnostics,
        initial_evaluation=initial_evaluation,
    )
    x_opt = np.asarray(result.x, dtype=float)
    report("profile optimized", problem, x_opt)

    write_profile_parameters(problem, x0, out_dir, "initial")
    write_profile_parameters(problem, x_opt, out_dir, "optimized")
    summary = {
        "seed_input": str(args.seed_input.resolve()),
        "transport_config": str(TRANSPORT_CONFIG),
        "database_grid": [
            args.database_n_theta,
            args.database_n_phi,
            args.database_n_xi,
        ],
        "transport_final_time": TRANSPORT_FINAL_TIME,
        "transport_max_steps": TRANSPORT_MAX_STEPS,
        "transport_n_radial": n_radial,
        "accepted_step_limit": FULL_TRANSPORT_ACCEPTED_STEP_LIMIT,
        "reverse_segment_length": REVERSE_SEGMENT_LENGTH,
        "max_reverse_accepted_steps": MAX_REVERSE_ACCEPTED_STEPS,
        "Er_transition_indices": [
            args.er_transition_left_index,
            args.er_transition_right_index,
        ],
        "objectives": {
            "max_er": bool(args.max_er),
            "root_objectives": bool(args.root_objectives),
            "transition_location_objective": bool(args.transition_location_objective),
            "bootstrap_penalty": bool(args.bootstrap_penalty),
            "net_power": bool(args.net_power),
        },
        "Er_transition_rho_target": args.er_transition_rho_target,
        "Er_transition_strength_target": args.er_transition_strength_target,
        "parameter_labels": list(problem.parameter_labels),
        "initial_physical_profiles": physical_profile_values(problem, x0),
        "optimized_physical_profiles": physical_profile_values(problem, x_opt),
        "x_scaled": x_opt.tolist(),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "status": int(result.status),
        "message": str(result.message),
    }
    summary["profile_plot_config"] = {
        key: copy.deepcopy(config[key])
        for key in ("geometry", "species", "profiles", "transport_solver", "equations")
        if key in config
    }
    if args.plot_initial_profiles:
        summary["initial_profile_comparison"] = write_initial_profile_comparison(
            config, summary["initial_physical_profiles"], summary["optimized_physical_profiles"],
            out_dir, geometry=problem.runtime.geometry,
        )
    summary_path = out_dir / "optimization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")

    if args.initial_transport_report:
        write_transport_report(problem, x0, out_dir, "initial")
    if args.optimized_transport_report:
        write_transport_report(problem, x_opt, out_dir, "optimized")
    if args.initial_transport_report and args.optimized_transport_report:
        summary["final_profile_comparison"] = write_final_profile_comparison(out_dir)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
