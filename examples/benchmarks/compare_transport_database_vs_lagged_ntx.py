from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_DATABASE_CONFIG = ROOT / "examples" / "Solve_Transport_Equations" / "Solve_Transport_equations_noHe_radau.toml"
DEFAULT_LAGGED_CONFIG = (
    ROOT
    / "examples"
    / "Solve_Transport_Equations"
    / "Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime.toml"
)
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "transport_compare_database_vs_lagged_ntx"
DEFAULT_REFERENCE_LABEL = "NTSS"


def _prepare_run_config(config_path: Path, *, output_dir: Path) -> dict:
    import NEOPAX

    config = copy.deepcopy(NEOPAX.load_config(config_path))
    transport_output = config.setdefault("transport_output", {})
    transport_output["transport_output_dir"] = str(output_dir)
    transport_output["transport_write_hdf5"] = True
    transport_output["transport_plot"] = False
    return config


def _run_case(config_path: Path, *, output_dir: Path, device: str | None) -> Path:
    import NEOPAX

    config = _prepare_run_config(config_path, output_dir=output_dir)
    run_kwargs = {}
    if device is not None:
        run_kwargs["device"] = device
    result = NEOPAX.run(config, **run_kwargs)
    resolved_output_dir = result.output_dir if result.output_dir is not None else output_dir
    solution_path = Path(resolved_output_dir) / "transport_solution.h5"
    if not solution_path.exists():
        raise FileNotFoundError(f"Expected transport output at '{solution_path}' but it was not written.")
    return solution_path


def _load_transport_solution(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        rho = np.asarray(handle["rho"][()], dtype=float)
        er_all = np.asarray(handle["Er"][()], dtype=float)
        pressure_all = np.asarray(handle["pressure"][()], dtype=float)
        density_all = np.asarray(handle["density"][()], dtype=float)
        temperature_all = np.asarray(handle["temperature"][()], dtype=float)
        ts = np.asarray(handle["ts"][()], dtype=float) if "ts" in handle else np.zeros((0,), dtype=float)
    return {
        "rho": rho,
        "Er_initial": er_all[0],
        "Er_final": er_all[-1],
        "pressure_initial": pressure_all[0],
        "pressure_final": pressure_all[-1],
        "density_initial": density_all[0],
        "density_final": density_all[-1],
        "temperature_initial": temperature_all[0],
        "temperature_final": temperature_all[-1],
        "ts": ts,
    }


def _load_reference_profiles(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        rho = np.asarray(handle["r"][()], dtype=float)
        er = np.asarray(handle["Er"][()], dtype=float)
        ne = np.asarray(handle["ne"][()], dtype=float)
        n_d = np.asarray(handle["nD"][()], dtype=float)
        te = np.asarray(handle["Te"][()], dtype=float)
        td = np.asarray(handle["TD"][()], dtype=float)
        tt = np.asarray(handle["Tt"][()], dtype=float)

    # NTSS no-He files expose ne and nD directly. Recover nT from quasineutrality.
    n_t = ne - n_d
    pressure = np.vstack([ne * te, n_d * td, n_t * tt])
    return {
        "rho": rho,
        "Er": er,
        "pressure": pressure,
    }


def _species_names_from_config(config_path: Path) -> list[str]:
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    names = config.get("species", {}).get("names", [])
    return [str(name) for name in names] if names else ["e", "D", "T"]


def _reference_files_from_config(config_path: Path) -> tuple[Path, Path]:
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    output_cfg = config.get("transport_output", {})
    initial_text = output_cfg.get("transport_initial_reference_file")
    final_text = output_cfg.get("transport_final_reference_file")
    if not initial_text:
        raise ValueError(f"No transport_initial_reference_file found in '{config_path}'.")
    if not final_text:
        raise ValueError(f"No transport_final_reference_file found in '{config_path}'.")
    return ROOT / Path(str(initial_text)), ROOT / Path(str(final_text))


def _plot_er(
    output_path: Path,
    database_case: dict[str, np.ndarray],
    lagged_case: dict[str, np.ndarray],
    reference_initial: dict[str, np.ndarray],
    reference_final: dict[str, np.ndarray],
    *,
    reference_label: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 8.0), sharex=True)
    panels = [
        ("Initial Er Comparison", "Er_initial", reference_initial),
        ("Final Er Comparison", "Er_final", reference_final),
    ]
    for ax, (title, key, reference_case) in zip(axes, panels):
        ax.plot(database_case["rho"], database_case[key], linewidth=2.0, label="database NTX")
        ax.plot(lagged_case["rho"], lagged_case[key], linewidth=2.0, label="lagged NTX")
        ax.plot(reference_case["rho"], reference_case["Er"], "--", linewidth=2.0, label=reference_label)
        ax.set_ylabel("Er")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("rho")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _plot_pressure(
    output_path: Path,
    database_case: dict[str, np.ndarray],
    lagged_case: dict[str, np.ndarray],
    reference_initial: dict[str, np.ndarray],
    reference_final: dict[str, np.ndarray],
    species_names: list[str],
    *,
    reference_label: str,
) -> None:
    n_species = len(species_names)
    fig, axes = plt.subplots(n_species, 2, figsize=(12.0, 3.0 * n_species), sharex=True)
    if n_species == 1:
        axes = np.asarray([axes])

    for idx in range(n_species):
        ax_initial = axes[idx, 0]
        ax_final = axes[idx, 1]
        ax_initial.plot(
            database_case["rho"], database_case["pressure_initial"][idx], linewidth=2.0, label="database NTX"
        )
        ax_initial.plot(
            lagged_case["rho"], lagged_case["pressure_initial"][idx], linewidth=2.0, label="lagged NTX"
        )
        ax_initial.plot(
            reference_initial["rho"], reference_initial["pressure"][idx], "--", linewidth=2.0, label=reference_label
        )
        ax_initial.set_ylabel(f"{species_names[idx]} pressure")
        ax_initial.set_title(f"{species_names[idx]} Initial")
        ax_initial.grid(True, alpha=0.3)
        ax_initial.legend()

        ax_final.plot(
            database_case["rho"], database_case["pressure_final"][idx], linewidth=2.0, label="database NTX"
        )
        ax_final.plot(
            lagged_case["rho"], lagged_case["pressure_final"][idx], linewidth=2.0, label="lagged NTX"
        )
        ax_final.plot(
            reference_final["rho"], reference_final["pressure"][idx], "--", linewidth=2.0, label=reference_label
        )
        ax_final.set_title(f"{species_names[idx]} Final")
        ax_final.grid(True, alpha=0.3)
        ax_final.legend()

    axes[-1, 0].set_xlabel("rho")
    axes[-1, 1].set_xlabel("rho")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _write_summary(
    output_path: Path,
    *,
    database_solution: Path,
    lagged_solution: Path,
    initial_reference_file: Path,
    final_reference_file: Path,
    er_plot: Path,
    pressure_plot: Path,
) -> None:
    payload = {
        "database_solution_h5": str(database_solution),
        "lagged_solution_h5": str(lagged_solution),
        "initial_reference_h5": str(initial_reference_file),
        "final_reference_h5": str(final_reference_file),
        "plots": {
            "Er": str(er_plot),
            "pressure": str(pressure_plot),
        },
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the database and lagged NTX transport cases and compare final Er/pressure against NTSS."
    )
    parser.add_argument("--database-config", type=Path, default=DEFAULT_DATABASE_CONFIG)
    parser.add_argument("--lagged-config", type=Path, default=DEFAULT_LAGGED_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--initial-reference-file", type=Path, default=None)
    parser.add_argument("--final-reference-file", type=Path, default=None)
    parser.add_argument("--reference-label", type=str, default=DEFAULT_REFERENCE_LABEL)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--skip-runs",
        action="store_true",
        help="Reuse existing transport_solution.h5 files in the output directories instead of rerunning NEOPAX.",
    )
    args = parser.parse_args()

    os.chdir(ROOT)

    output_root = args.output_root
    database_output_dir = output_root / "database_ntx"
    lagged_output_dir = output_root / "lagged_ntx"
    comparison_output_dir = output_root / "comparison"
    database_output_dir.mkdir(parents=True, exist_ok=True)
    lagged_output_dir.mkdir(parents=True, exist_ok=True)
    comparison_output_dir.mkdir(parents=True, exist_ok=True)

    database_solution_path = database_output_dir / "transport_solution.h5"
    lagged_solution_path = lagged_output_dir / "transport_solution.h5"

    if args.skip_runs:
        if not database_solution_path.exists():
            raise FileNotFoundError(f"--skip-runs was set but '{database_solution_path}' does not exist.")
        if not lagged_solution_path.exists():
            raise FileNotFoundError(f"--skip-runs was set but '{lagged_solution_path}' does not exist.")
    else:
        print(f"[compare-transport] running database NTX case: {args.database_config}")
        database_solution_path = _run_case(args.database_config, output_dir=database_output_dir, device=args.device)
        print(f"[compare-transport] running lagged NTX case: {args.lagged_config}")
        lagged_solution_path = _run_case(args.lagged_config, output_dir=lagged_output_dir, device=args.device)

    initial_reference_file = args.initial_reference_file
    final_reference_file = args.final_reference_file
    if initial_reference_file is None or final_reference_file is None:
        cfg_initial_reference, cfg_final_reference = _reference_files_from_config(args.database_config)
        if initial_reference_file is None:
            initial_reference_file = cfg_initial_reference
        if final_reference_file is None:
            final_reference_file = cfg_final_reference
    if not initial_reference_file.is_absolute():
        initial_reference_file = ROOT / initial_reference_file
    if not final_reference_file.is_absolute():
        final_reference_file = ROOT / final_reference_file

    species_names = _species_names_from_config(args.database_config)
    database_case = _load_transport_solution(database_solution_path)
    lagged_case = _load_transport_solution(lagged_solution_path)
    reference_initial = _load_reference_profiles(initial_reference_file)
    reference_final = _load_reference_profiles(final_reference_file)

    if database_case["pressure_final"].shape[0] != len(species_names):
        raise ValueError(
            f"Species list length {len(species_names)} does not match pressure shape {database_case['pressure_final'].shape}."
        )
    if reference_initial["pressure"].shape[0] < len(species_names):
        raise ValueError(
            f"Reference file '{initial_reference_file}' only provides {reference_initial['pressure'].shape[0]} pressure channels."
        )
    if reference_final["pressure"].shape[0] < len(species_names):
        raise ValueError(
            f"Reference file '{final_reference_file}' only provides {reference_final['pressure'].shape[0]} pressure channels."
        )

    er_plot = comparison_output_dir / "compare_Er_database_vs_lagged_vs_ntss.png"
    pressure_plot = comparison_output_dir / "compare_pressure_database_vs_lagged_vs_ntss.png"
    summary_json = comparison_output_dir / "comparison_summary.json"

    _plot_er(
        er_plot,
        database_case,
        lagged_case,
        reference_initial,
        reference_final,
        reference_label=args.reference_label,
    )
    _plot_pressure(
        pressure_plot,
        database_case,
        lagged_case,
        reference_initial,
        reference_final,
        species_names,
        reference_label=args.reference_label,
    )
    _write_summary(
        summary_json,
        database_solution=database_solution_path,
        lagged_solution=lagged_solution_path,
        initial_reference_file=initial_reference_file,
        final_reference_file=final_reference_file,
        er_plot=er_plot,
        pressure_plot=pressure_plot,
    )

    print(f"[compare-transport] wrote {er_plot}")
    print(f"[compare-transport] wrote {pressure_plot}")
    print(f"[compare-transport] wrote {summary_json}")


if __name__ == "__main__":
    main()
