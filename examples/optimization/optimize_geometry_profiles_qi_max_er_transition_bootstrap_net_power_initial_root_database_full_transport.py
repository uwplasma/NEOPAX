#!/usr/bin/env python
"""Optimize VMEX boundary and analytical profiles through full transport.

This is the combined-DoF counterpart of the standalone geometry database
full-transport example.  It retains the same QI, maximum-J, final-time Er,
bootstrap-current, and net-power objectives while optimizing the selected
RBC/ZBS boundary modes together with all six analytical-profile parameters,
including both profile-alpha shape parameters.
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

from NEOPAX import optimization as opt  # noqa: E402
from NEOPAX._orchestrator import load_config  # noqa: E402
from examples.optimization import (  # noqa: E402
    optimize_geometry_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport
    as geometry_example,
)


# --------------------------- editable settings -----------------------------
SEED_INPUT = ROOT / "examples" / "inputs" / "input.QI_nfp2_initial"
TRANSPORT_CONFIG = (
    ROOT / "examples" / "optimization" / "full_transport_database_vacuum_t2.toml"
)
OUT_DIR = (
    ROOT
    / "outputs"
    / "geometry_profiles_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_optimization"
)

DATABASE_N_THETA, DATABASE_N_PHI, DATABASE_N_XI = 25, 25, 33
SURFACES = np.asarray(
    [
        1 / 51,
        5 / 51,
        10 / 51,
        15 / 51,
        20 / 51,
        25 / 51,
        30 / 51,
        35 / 51,
        40 / 51,
        45 / 51,
        51 / 51,
    ],
    dtype=float,
)
QI_MBOZ = 18
QI_NBOZ = 18
QI_MAXJ_BACKEND = "surrogate"
PHYSICAL_J_PITCHES = None
PHYSICAL_J_TRAPPING_DEPTHS = (0.35, 0.55, 0.75)
PHYSICAL_J_NALPHA, PHYSICAL_J_POINTS_PER_PERIOD = 5, 24
PHYSICAL_J_NUM_PERIODS, PHYSICAL_J_MAX_WELLS = 6, 16
PHYSICAL_J_QUADRATURE_ORDER, PHYSICAL_MAXJ_TARGET = 16, 0.0

MAX_MODE_SCHEDULE = 2
GEOMETRY_FAMILIES = "RBC,ZBS"
GEOMETRY_SCALE_MODE = "ess"
ESS_ALPHA = 1.2

# All supported analytical-profile DoFs, explicitly editable here.
PROFILE_PARAMETERS = (
    "n0,T0,density_shape_power,temperature_shape_power,"
    "density_shape_alpha,temperature_shape_alpha"
)
PROFILE_SCALE_MODE = "nominal"
PROFILE_PHYSICAL_LOWER = {
    "n0": 2.0,
    "T0": 7.0,
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

FULL_TRANSPORT_ACCEPTED_STEP_LIMIT = None
REVERSE_SEGMENT_LENGTH = 50
MAX_REVERSE_ACCEPTED_STEPS = 500
TRANSPORT_MAX_STEPS = 1000
TRANSPORT_FINAL_TIME = 2.0
REVERSE_STAGE_MODE = "database_full_transport_optimization"
PRINT_FINAL_SOFTMAX_ER = True

ASPECT_TARGET, IOTA_TARGET, MIRROR_TARGET = 10.0, -0.61, 0.25
MAX_ER_TARGET = 25.0
ER_TRANSITION_LEFT_TARGET, ER_TRANSITION_RIGHT_TARGET = 26.0, -10.0
ER_TRANSITION_LEFT_INDEX, ER_TRANSITION_RIGHT_INDEX = 25, 26
ER_TRANSITION_RHO_TARGET = 0.514
ER_TRANSITION_RHO_MIN, ER_TRANSITION_RHO_MAX = 0.05, 0.9
ER_TRANSITION_TEMPERATURE_KV_M = 2.0
ER_TRANSITION_RHO_SOFTNESS = 0.05
ER_TRANSITION_SOFTMAX_BETA = 16.0
ER_TRANSITION_STRENGTH_TARGET = 0.90
BOOTSTRAP_LIMIT_SCALED = 0.1
NET_POWER_TARGET_MW = 300.0
NET_POWER_REFERENCE_VOLUME_M3 = 331.0187969899648
NET_POWER_TARGET_MW_M3 = NET_POWER_TARGET_MW / NET_POWER_REFERENCE_VOLUME_M3

QI_WEIGHT, MAXJ_WEIGHT = 1.0, 0.0
ASPECT_WEIGHT, IOTA_WEIGHT, MIRROR_WEIGHT = 1.0, 1.0, 100.0
MAX_ER_WEIGHT = 0.5
ER_TRANSITION_LEFT_WEIGHT, ER_TRANSITION_RIGHT_WEIGHT = 0.09, 0.09
ER_TRANSITION_RHO_WEIGHT = 1000.0
ER_TRANSITION_STRENGTH_WEIGHT = 100.0
BOOTSTRAP_WEIGHT, NET_POWER_WEIGHT = 4.0, 10000.0

USE_MAX_ER_OBJECTIVE = False
USE_ER_TRANSITION_OBJECTIVES = False
USE_ER_TRANSITION_LOCATION_OBJECTIVE = True
USE_BOOTSTRAP_PENALTY = True
USE_NET_POWER_OBJECTIVE = True

NFEV = 30
FTOL, XTOL = 1.0e-6, 1.0e-10
GEOMETRY_MAX_ITER = None
SOLVER_DEVICE = "default"


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description=__doc__)
    out.add_argument("--seed-input", type=Path, default=SEED_INPUT)
    out.add_argument("--out-dir", type=Path, default=OUT_DIR)
    out.add_argument(
        "--postprocess-existing",
        action="store_true",
        help=(
            "read existing initial/optimized transport_solution.h5 files and "
            "write only the density/temperature profile comparison"
        ),
    )
    out.add_argument("--max-nfev", type=int, default=NFEV)
    out.add_argument("--database-n-theta", type=int, default=DATABASE_N_THETA)
    out.add_argument("--database-n-phi", type=int, default=DATABASE_N_PHI)
    out.add_argument("--database-n-xi", type=int, default=DATABASE_N_XI)
    out.add_argument(
        "--er-transition-left-index", type=int, default=ER_TRANSITION_LEFT_INDEX
    )
    out.add_argument(
        "--er-transition-right-index", type=int, default=ER_TRANSITION_RIGHT_INDEX
    )
    out.add_argument(
        "--max-er",
        action=argparse.BooleanOptionalAction,
        default=USE_MAX_ER_OBJECTIVE,
    )
    out.add_argument(
        "--root-objectives",
        action=argparse.BooleanOptionalAction,
        default=USE_ER_TRANSITION_OBJECTIVES,
    )
    out.add_argument(
        "--transition-location-objective",
        action=argparse.BooleanOptionalAction,
        default=USE_ER_TRANSITION_LOCATION_OBJECTIVE,
    )
    out.add_argument(
        "--bootstrap-penalty",
        action=argparse.BooleanOptionalAction,
        default=USE_BOOTSTRAP_PENALTY,
    )
    out.add_argument(
        "--net-power",
        action=argparse.BooleanOptionalAction,
        default=USE_NET_POWER_OBJECTIVE,
    )
    out.add_argument(
        "--initial-plots", action=argparse.BooleanOptionalAction, default=True
    )
    out.add_argument(
        "--er-transition-rho-target", type=float, default=ER_TRANSITION_RHO_TARGET
    )
    out.add_argument(
        "--er-transition-strength-target",
        type=float,
        default=ER_TRANSITION_STRENGTH_TARGET,
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
    )
    out.add_argument(
        "--er-transition-softmax-beta",
        type=float,
        default=ER_TRANSITION_SOFTMAX_BETA,
    )
    return out


def transport_config(args: argparse.Namespace) -> dict:
    config = copy.deepcopy(load_config(TRANSPORT_CONFIG))
    neoclassical = config.setdefault("neoclassical", {})
    neoclassical["ntx_scan_n_theta"] = int(args.database_n_theta)
    neoclassical["ntx_scan_n_zeta"] = int(args.database_n_phi)
    neoclassical["ntx_scan_n_xi"] = int(args.database_n_xi)
    solver = config.setdefault("transport_solver", {})
    solver["t_final"] = float(TRANSPORT_FINAL_TIME)
    solver["max_steps"] = int(TRANSPORT_MAX_STEPS)
    return config


def qi_maxj_backend_settings(physical_pitches=None):
    return opt.QImaxJBackendSettings(
        backend=QI_MAXJ_BACKEND,
        physical_pitches=(
            PHYSICAL_J_PITCHES if physical_pitches is None else physical_pitches
        ),
        trapping_depths=PHYSICAL_J_TRAPPING_DEPTHS,
        physical_nalpha=PHYSICAL_J_NALPHA,
        physical_points_per_period=PHYSICAL_J_POINTS_PER_PERIOD,
        physical_num_periods=PHYSICAL_J_NUM_PERIODS,
        physical_max_wells=PHYSICAL_J_MAX_WELLS,
        physical_quadrature_order=PHYSICAL_J_QUADRATURE_ORDER,
        physical_maxj_target=PHYSICAL_MAXJ_TARGET,
    )


def positive_part(value):
    return jnp.maximum(value, 0.0)


def smooth_positive_part(value, eps: float = 1.0e-6):
    value_arr = jnp.asarray(value)
    eps_arr = jnp.asarray(eps, dtype=value_arr.dtype)
    return 0.5 * (
        value_arr + jnp.sqrt(value_arr * value_arr + eps_arr * eps_arr)
    )


mirror_penalization = opt.transformed_geometry_objective(
    opt.geometry.vmec_mirror_ratio,
    lambda value: positive_part(value - MIRROR_TARGET),
    label="mirror_penalization",
)
bootstrap_penalty = opt.transformed_transport_objective(
    opt.transport.bootstrap_current_softmax_abs_scaled,
    lambda value: positive_part(value - BOOTSTRAP_LIMIT_SCALED),
    label="bootstrap_current_penalty",
)


def active_terms(args: argparse.Namespace):
    terms = [
        (opt.geometry.boozer_qi_objective, 0.0, QI_WEIGHT),
        (opt.geometry.boozer_maxj_objective, 0.0, MAXJ_WEIGHT),
        (mirror_penalization, 0.0, MIRROR_WEIGHT),
        (opt.geometry.vmec_aspect_ratio, ASPECT_TARGET, ASPECT_WEIGHT),
        (opt.geometry.vmec_iota_mean, IOTA_TARGET, IOTA_WEIGHT),
    ]
    if args.max_er:
        terms.append((opt.transport.softmax_Er, MAX_ER_TARGET, MAX_ER_WEIGHT))
    if args.root_objectives:
        terms.extend(
            [
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
            ]
        )
    if args.transition_location_objective:
        strength_deficit = opt.transformed_transport_objective(
            opt.transport.Er_transition_strength,
            lambda strength: smooth_positive_part(
                args.er_transition_strength_target - strength
            ),
            label="Er_transition_strength_deficit",
        )
        terms.extend(
            [
                (opt.transport.Er_transition_rho, 0.0, 0.0),
                (
                    opt.transport.Er_transition_location_moment,
                    0.0,
                    ER_TRANSITION_RHO_WEIGHT,
                ),
                (opt.transport.Er_transition_strength, 0.0, 0.0),
                (strength_deficit, 0.0, ER_TRANSITION_STRENGTH_WEIGHT),
            ]
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
    if not any(
        (
            term[0].objective.family
            if hasattr(term[0], "objective")
            else term[0].family
        )
        == "transport"
        for term in terms
    ):
        raise ValueError("Enable at least one full-transport objective.")
    return tuple(terms)


def profile_parameter_values(problem, x) -> dict[str, float]:
    values = np.asarray(
        jax.device_get(problem.profile_values_from_scaled_parameters(x)), dtype=float
    )
    names = (
        "n0",
        "T0",
        "density_shape_power",
        "temperature_shape_power",
        "density_shape_alpha",
        "temperature_shape_alpha",
    )
    return dict(zip(names, values.tolist(), strict=True))


def scaled_bounds(problem):
    scales = np.asarray(jax.device_get(problem.x_scale), dtype=float)
    lower, upper = [], []
    for label, scale in zip(problem.parameter_labels, scales, strict=True):
        lower.append(PROFILE_PHYSICAL_LOWER.get(label, -np.inf) / scale)
        upper.append(PROFILE_PHYSICAL_UPPER.get(label, np.inf) / scale)
    return np.asarray(lower), np.asarray(upper)


class CombinedTrialSavingProblem:
    """Save geometry input and physical profiles before each unique trial."""

    def __init__(self, problem, out_dir: Path, max_mode: int):
        self.problem = problem
        self.out_dir = out_dir
        self.max_mode = int(max_mode)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._seen = {}
        self._next_index = 0

    def __getattr__(self, name):
        return getattr(self.problem, name)

    def evaluate(self, scaled_parameter_values=None):
        x = self.x0 if scaled_parameter_values is None else jnp.asarray(
            scaled_parameter_values, dtype=jnp.float64
        )
        host = np.ascontiguousarray(np.asarray(jax.device_get(x), dtype=np.float64))
        key = (host.shape, host.tobytes())
        if key not in self._seen:
            index = self._next_index
            self._next_index += 1
            input_path = self.out_dir / (
                f"input.QI_neopax_database_full_transport_combined_eval_{index:04d}"
            )
            profile_path = self.out_dir / f"profile_parameters_eval_{index:04d}.json"
            self.problem.input_from_scaled_parameters(x).to_indata(input_path)
            profile_path.write_text(
                json.dumps(profile_parameter_values(self.problem, x), indent=2),
                encoding="utf-8",
            )
            self._seen[key] = (input_path, profile_path)
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


def _transport_profile_snapshots(h5_path: Path) -> dict:
    """Load the first and last finite e/D/T profiles from a transport HDF5."""
    import h5py

    with h5py.File(h5_path, "r") as h5_file:
        missing = {
            name
            for name in ("rho", "ts", "density", "temperature")
            if name not in h5_file
        }
        if missing:
            raise ValueError(
                f"{h5_path} is missing required datasets: {sorted(missing)}"
            )
        rho = np.asarray(h5_file["rho"], dtype=float).reshape(-1)
        times = np.asarray(h5_file["ts"], dtype=float).reshape(-1)
        raw_profiles = {
            name: np.asarray(h5_file[name], dtype=float)
            for name in ("density", "temperature")
        }

    finite_time_indices = np.flatnonzero(np.isfinite(times))
    if finite_time_indices.size == 0:
        raise ValueError(f"{h5_path} contains no finite saved transport times.")
    initial_index = int(finite_time_indices[np.argmin(times[finite_time_indices])])
    final_index = int(finite_time_indices[np.argmax(times[finite_time_indices])])

    def _time_species_rho(name: str, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float).squeeze()
        if values.ndim == 2 and times.size == 1:
            values = values[np.newaxis, ...]
        if values.ndim != 3:
            raise ValueError(
                f"Cannot interpret {name} in {h5_path}: expected a three-axis "
                f"time/species/rho array, got shape={values.shape}."
            )

        # The writer uses (time, species, rho).  Infer the axes as a guard
        # against older files that may store an equivalent permutation.
        if values.shape[0] == times.size and values.shape[-1] == rho.size:
            normalized = values
        else:
            time_axes = [
                axis for axis, size in enumerate(values.shape) if size == times.size
            ]
            rho_axes = [
                axis for axis, size in enumerate(values.shape) if size == rho.size
            ]
            candidates = [
                (time_axis, rho_axis)
                for time_axis in time_axes
                for rho_axis in rho_axes
                if time_axis != rho_axis
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"Cannot identify unique time/rho axes for {name} in {h5_path}: "
                    f"shape={values.shape}, n_times={times.size}, n_rho={rho.size}."
                )
            time_axis, rho_axis = candidates[0]
            species_axis = next(
                axis for axis in range(values.ndim) if axis not in (time_axis, rho_axis)
            )
            normalized = np.moveaxis(
                values,
                (time_axis, species_axis, rho_axis),
                (0, 1, 2),
            )
        if normalized.shape[1] < 3:
            raise ValueError(
                f"{h5_path} contains only {normalized.shape[1]} species; e, D, and T "
                "profiles are required."
            )
        return normalized

    profiles = {
        name: _time_species_rho(name, values)
        for name, values in raw_profiles.items()
    }
    return {
        "rho": rho,
        "initial_time": float(times[initial_index]),
        "final_time": float(times[final_index]),
        "density_initial": profiles["density"][initial_index, :3],
        "density_final": profiles["density"][final_index, :3],
        "temperature_initial": profiles["temperature"][initial_index, :3],
        "temperature_final": profiles["temperature"][final_index, :3],
    }


def write_initial_optimized_profile_comparison(
    out_dir: Path,
) -> tuple[Path, Path, Path]:
    """Compare initial/optimized configurations at initial and final time."""
    import csv

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    h5_paths = {
        label: out_dir / label / "transport" / "transport_solution.h5"
        for label in ("initial", "optimized")
    }
    missing = [str(path) for path in h5_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Run the initial and optimized forward transport reports before the "
            "profile comparison. Missing: " + ", ".join(missing)
        )

    snapshots = {
        label: _transport_profile_snapshots(path)
        for label, path in h5_paths.items()
    }
    rho = snapshots["initial"]["rho"]
    optimized_rho = snapshots["optimized"]["rho"]
    if rho.shape != optimized_rho.shape or not np.allclose(
        rho,
        optimized_rho,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "Initial and optimized transport outputs use different rho grids; "
            "a side-by-side pointwise comparison is not valid."
        )

    species = ("e", "D", "T")
    colors = {"e": "C0", "D": "C1", "T": "C2"}
    configurations = (("initial", "--"), ("optimized", "-"))
    time_states = ("initial", "final")
    quantities = (
        ("density", r"$n$ [$10^{20}\,\mathrm{m}^{-3}$]"),
        ("temperature", r"$T$ [$\mathrm{keV}$]"),
    )

    figure_paths = {
        time_state: out_dir
        / f"{time_state}_profiles_initial_vs_optimized.png"
        for time_state in time_states
    }
    csv_path = out_dir / "initial_optimized_density_temperature_profiles.csv"

    for time_state in time_states:
        fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.6), sharex=True)
        for axis, (quantity, ylabel) in zip(axes, quantities):
            axis.set_title(quantity.capitalize(), fontsize=19)
            for label, linestyle in configurations:
                snapshot = snapshots[label]
                values = snapshot[f"{quantity}_{time_state}"]
                for species_index, species_name in enumerate(species):
                    axis.plot(
                        rho,
                        values[species_index],
                        color=colors[species_name],
                        linestyle=linestyle,
                        linewidth=3.0,
                    )
            axis.set_xlabel(r"$\rho$", fontsize=20)
            axis.set_ylabel(ylabel, fontsize=20)
            axis.grid(False)
            axis.tick_params(axis="both", labelsize=16, width=1.0, length=4)
            axis.margins(x=0.04, y=0.08)
            for spine in axis.spines.values():
                spine.set_linewidth(1.0)
                spine.set_color("0.35")

        species_handles = [
            Line2D([0], [0], color=colors[name], linewidth=3.0, label=name)
            for name in species
        ]
        configuration_handles = [
            Line2D(
                [0],
                [0],
                color="black",
                linestyle=linestyle,
                linewidth=3.0,
                label=f"{label.capitalize()} configuration",
            )
            for label, linestyle in configurations
        ]
        figure_species_legend = fig.legend(
            handles=species_handles,
            title="Species",
            loc="lower center",
            bbox_to_anchor=(0.35, 0.005),
            ncol=3,
            fontsize=15,
            title_fontsize=15,
            frameon=True,
        )
        fig.add_artist(figure_species_legend)
        fig.legend(
            handles=configuration_handles,
            title="Configuration",
            loc="lower center",
            bbox_to_anchor=(0.72, 0.005),
            ncol=2,
            fontsize=15,
            title_fontsize=15,
            frameon=True,
        )
        initial_time = snapshots["initial"][f"{time_state}_time"]
        optimized_time = snapshots["optimized"][f"{time_state}_time"]
        if np.isclose(initial_time, optimized_time, rtol=0.0, atol=1.0e-12):
            time_label = rf"$t={initial_time:.3g}\,\mathrm{{s}}$"
        else:
            time_label = (
                rf"$t_{{\mathrm{{initial\ config}}}}={initial_time:.3g}\,\mathrm{{s}}$, "
                rf"$t_{{\mathrm{{optimized\ config}}}}={optimized_time:.3g}\,\mathrm{{s}}$"
            )
        fig.suptitle(
            f"{time_state.capitalize()} transport profiles: " + time_label,
            fontsize=20,
        )
        fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.94))
        fig.savefig(figure_paths[time_state], dpi=320, bbox_inches="tight")
        plt.close(fig)

    fieldnames = ["rho"]
    for quantity, _ in quantities:
        units = "1e20_m-3" if quantity == "density" else "keV"
        for label in ("initial", "optimized"):
            for time_state in time_states:
                for species_name in species:
                    fieldnames.append(
                        f"{label}_{time_state}_{species_name}_{quantity}_{units}"
                    )
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for radial_index, rho_value in enumerate(rho):
            row = {"rho": f"{rho_value:.16e}"}
            for quantity, _ in quantities:
                units = "1e20_m-3" if quantity == "density" else "keV"
                for label in ("initial", "optimized"):
                    for time_state in time_states:
                        values = snapshots[label][f"{quantity}_{time_state}"]
                        for species_index, species_name in enumerate(species):
                            key = (
                                f"{label}_{time_state}_{species_name}_"
                                f"{quantity}_{units}"
                            )
                            row[key] = f"{values[species_index, radial_index]:.16e}"
            writer.writerow(row)

    for figure_path in figure_paths.values():
        print(f"wrote {figure_path}", flush=True)
    print(f"wrote {csv_path}", flush=True)
    return figure_paths["initial"], figure_paths["final"], csv_path


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
    out_dir.mkdir(parents=True, exist_ok=True)
    initial_input.to_indata(out_dir / seed_input.name)
    optimized_path = (
        out_dir / "input.QI_neopax_database_full_transport_combined_optimized"
    )
    optimized_input.to_indata(optimized_path)
    print(f"wrote {optimized_path}")
    for label, values in (
        ("initial", initial_profiles),
        ("optimized", optimized_profiles),
    ):
        path = out_dir / label / f"profile_parameters_{label}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(values, indent=2), encoding="utf-8")
        print(f"wrote {path}")
    if make_initial_plots:
        geometry_example.write_geometry_artifacts(
            initial_input,
            "initial",
            out_dir,
            physical_pitches=physical_pitches,
        )
        geometry_example.write_transport_report(
            initial_input, "initial", initial_config, out_dir
        )
    geometry_example.write_geometry_artifacts(
        optimized_input,
        "optimized",
        out_dir,
        physical_pitches=physical_pitches,
    )
    geometry_example.write_transport_report(
        optimized_input, "optimized", optimized_config, out_dir
    )
    if make_initial_plots:
        write_initial_optimized_profile_comparison(out_dir)


def main() -> int:
    args = parser().parse_args()
    if args.postprocess_existing:
        write_initial_optimized_profile_comparison(args.out_dir.resolve())
        return 0
    if args.max_nfev < 1:
        raise ValueError("--max-nfev must be positive.")
    for name in ("database_n_theta", "database_n_phi", "database_n_xi"):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if not 0.0 <= args.er_transition_rho_min < args.er_transition_rho_max <= 1.0:
        raise ValueError("Er transition radial window must satisfy 0 <= min < max <= 1.")
    if not args.er_transition_rho_min <= args.er_transition_rho_target <= args.er_transition_rho_max:
        raise ValueError("Er transition target must lie inside the radial window.")
    if min(
        args.er_transition_strength_target,
        args.er_transition_temperature_kv_m,
        args.er_transition_rho_softness,
        args.er_transition_softmax_beta,
    ) <= 0.0:
        raise ValueError("Er transition strength/temperature/softness/beta must be positive.")

    seed_input = args.seed_input.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    current_input = seed_input
    current_config = transport_config(args)
    n_radial = int(current_config.get("geometry", {}).get("n_radial", 51))
    for name in ("er_transition_left_index", "er_transition_right_index"):
        index = int(getattr(args, name))
        if not 0 <= index < n_radial:
            raise ValueError(
                f"--{name.replace('_', '-')} must be in [0, {n_radial}); "
                f"got {index}."
            )
    terms = active_terms(args)
    frozen_physical_pitches = PHYSICAL_J_PITCHES
    initial_input = initial_config = initial_profiles = None
    optimized_input = optimized_config = optimized_profiles = None
    last_problem = last_result = None

    max_modes = (
        (int(MAX_MODE_SCHEDULE),)
        if np.isscalar(MAX_MODE_SCHEDULE)
        else tuple(int(value) for value in MAX_MODE_SCHEDULE)
    )
    for max_mode in max_modes:
        print(
            "\n===== combined geometry + six-profile-DoF database full-transport "
            f"stage, max_mode={max_mode}, grid=({args.database_n_theta},"
            f"{args.database_n_phi},{args.database_n_xi}), "
            f"J_backend={QI_MAXJ_BACKEND} =====",
            flush=True,
        )
        problem = opt.geometry_full_transport_least_squares_problem(
            current_config,
            terms,
            vmec_input=current_input,
            max_mode=max_mode,
            include_profiles=True,
            profile_parameters=PROFILE_PARAMETERS,
            profile_scale_mode=PROFILE_SCALE_MODE,
            families=GEOMETRY_FAMILIES,
            scale_mode=GEOMETRY_SCALE_MODE,
            ess_alpha=ESS_ALPHA,
            mboz=QI_MBOZ,
            nboz=QI_NBOZ,
            surfaces=tuple(float(value) for value in SURFACES),
            n_theta=args.database_n_theta,
            n_zeta=args.database_n_phi,
            n_xi=args.database_n_xi,
            geometry_max_iter=GEOMETRY_MAX_ITER,
            geometry_solver_device=SOLVER_DEVICE,
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
            qi_maxj_settings=qi_maxj_backend_settings(frozen_physical_pitches),
        )
        if QI_MAXJ_BACKEND.strip().lower() == "physical" and frozen_physical_pitches is None:
            frozen_physical_pitches = tuple(
                float(value) for value in problem.context.qi_maxj_physical_pitches
            )
        problem = CombinedTrialSavingProblem(
            problem, out_dir / f"combined_inputs_m{max_mode}", max_mode
        )
        x0 = np.asarray(jax.device_get(problem.x0), dtype=float)
        if initial_input is None:
            initial_input = problem.input_from_scaled_parameters(x0)
            initial_config = problem.config_from_scaled_parameters(x0)
            initial_profiles = profile_parameter_values(problem, x0)
        print(
            f"[setup] parameter_count={problem.parameter_count} "
            f"parameters={list(problem.parameter_labels)}",
            flush=True,
        )
        initial_evaluation = report("initial", problem, x0)
        last_result = opt.least_squares(
            problem,
            max_nfev=args.max_nfev,
            ftol=FTOL,
            xtol=XTOL,
            x_scale=np.ones_like(x0),
            bounds=scaled_bounds(problem),
            verbose=1,
            iteration_reporter=geometry_example.iteration_diagnostics,
            initial_evaluation=initial_evaluation,
        )
        x_opt = np.asarray(last_result.x, dtype=float)
        report(f"combined stage {max_mode}", problem, x_opt)
        optimized_input = problem.input_from_scaled_parameters(x_opt)
        optimized_config = problem.config_from_scaled_parameters(x_opt)
        optimized_profiles = profile_parameter_values(problem, x_opt)
        stage_input = out_dir / f"input.QI_neopax_combined_stage_m{max_mode}"
        optimized_input.to_indata(stage_input)
        print(f"wrote {stage_input}")
        current_input = stage_input
        current_config = optimized_config
        last_problem = problem

    if any(
        value is None
        for value in (
            initial_input,
            initial_config,
            initial_profiles,
            optimized_input,
            optimized_config,
            optimized_profiles,
            last_problem,
            last_result,
        )
    ):
        raise RuntimeError("No combined optimization stage was executed.")

    summary = {
        "seed_input": str(seed_input),
        "transport_config": str(TRANSPORT_CONFIG),
        "database_grid": [
            args.database_n_theta,
            args.database_n_phi,
            args.database_n_xi,
        ],
        "max_mode_schedule": list(max_modes),
        "profile_parameters": PROFILE_PARAMETERS.split(","),
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
        "status": int(last_result.status),
        "message": str(last_result.message),
    }
    summary_path = out_dir / "optimization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")
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
