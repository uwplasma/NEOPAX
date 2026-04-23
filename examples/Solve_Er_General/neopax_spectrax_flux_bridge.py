#!/usr/bin/env python3
"""Bridge NEOPAX profile outputs to local SPECTRAX-GK nonlinear flux scans.

This script has three user-facing subcommands:

- ``prepare``: read a NEOPAX HDF5 result plus the originating TOML, pick
  several radial locations, and write a manifest describing one local
  SPECTRAX-GK nonlinear run per radius.
- ``run``: execute the prepared runs in parallel, either round-robin across
  visible GPUs or across a CPU worker pool.
- ``collect``: gather the final nonlinear heat / particle fluxes from the
  generated SPECTRAX-GK diagnostics CSV files into one HDF5 summary.

The bridge is intentionally conservative:
- it treats NEOPAX as the source of local profiles ``n_s(rho), T_s(rho), Er(rho)``
- it estimates local logarithmic gradients from those profiles
- it maps ``rho -> torflux`` using the common assumption ``torflux = rho**2``
- it launches one *independent* SPECTRAX-GK flux-tube run per selected radius

That makes it suitable for a first external workflow before the same logic is
embedded directly inside the NEOPAX transport solve.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any

import h5py
import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]


DEFAULT_SPECTRAX_ROOT = Path(__file__).resolve().parents[3] / "SPECTRAX-GK"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "spectrax_flux_scan"


@dataclass(frozen=True)
class SpeciesMeta:
    name: str
    charge: float
    mass_mp: float


@dataclass(frozen=True)
class ProfileSnapshot:
    rho: np.ndarray
    density: np.ndarray  # shape (ns, nr)
    temperature: np.ndarray  # shape (ns, nr)
    er: np.ndarray  # shape (nr,)
    time_value: float | None


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _resolve_relative(base: Path, value: str | None) -> str | None:
    if value is None:
        return None
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    if path.is_absolute():
        return str(path.resolve())
    return str((base / path).resolve())


def _infer_neopax_root(config_path: Path) -> Path:
    parent = config_path.resolve().parent
    if parent.parent.name == "examples":
        return parent.parent.parent
    return parent


def _parse_species_from_neopax_config(cfg: dict[str, Any]) -> list[SpeciesMeta]:
    species_cfg = cfg.get("species", {})
    names = list(species_cfg.get("names", []))
    masses = list(species_cfg.get("mass_mp", []))
    charges = list(species_cfg.get("charge_qp", []))
    if not names or len(names) != len(masses) or len(names) != len(charges):
        raise ValueError("NEOPAX config [species] must define matching names, mass_mp, and charge_qp arrays")
    return [
        SpeciesMeta(name=str(name), charge=float(charge), mass_mp=float(mass))
        for name, mass, charge in zip(names, masses, charges)
    ]


def _canonical_species_key(name: str) -> str:
    return str(name).strip().lower()


def _infer_transport_snapshot(h5_path: Path, *, time_index: int) -> ProfileSnapshot:
    with h5py.File(h5_path, "r") as f:
        if {"rho", "density", "temperature", "Er"}.issubset(f.keys()):
            rho = np.asarray(f["rho"][()], dtype=float)
            density_all = np.asarray(f["density"][()], dtype=float)
            temperature_all = np.asarray(f["temperature"][()], dtype=float)
            er_all = np.asarray(f["Er"][()], dtype=float)
            ts = np.asarray(f["ts"][()], dtype=float) if "ts" in f else None
            idx = time_index if time_index >= 0 else density_all.shape[0] + time_index
            if idx < 0 or idx >= density_all.shape[0]:
                raise IndexError(f"time index {time_index} out of range for {h5_path}")
            return ProfileSnapshot(
                rho=rho,
                density=np.asarray(density_all[idx], dtype=float),
                temperature=np.asarray(temperature_all[idx], dtype=float),
                er=np.asarray(er_all[idx], dtype=float),
                time_value=None if ts is None else float(ts[idx]),
            )
    raise KeyError("This HDF5 file does not look like a NEOPAX transport_solution.h5 output")


def _infer_ntss_snapshot(h5_path: Path, species: list[SpeciesMeta]) -> ProfileSnapshot:
    density_map = {
        "e": "ne",
        "electron": "ne",
        "electrons": "ne",
        "d": "nD",
        "deuterium": "nD",
        "t": "nT",
        "tritium": "nT",
        "he": "nHe",
        "helium": "nHe",
    }
    temperature_map = {
        "e": "Te",
        "electron": "Te",
        "electrons": "Te",
        "d": "TD",
        "deuterium": "TD",
        "t": "TT",
        "tritium": "TT",
        "he": "THe",
        "helium": "THe",
    }
    with h5py.File(h5_path, "r") as f:
        if "r" not in f or "Er" not in f:
            raise KeyError("Flat NTSS-like file must at least contain datasets r and Er")
        rho = np.asarray(f["r"][()], dtype=float)
        er = np.asarray(f["Er"][()], dtype=float)
        density = np.zeros((len(species), rho.size), dtype=float)
        temperature = np.zeros((len(species), rho.size), dtype=float)
        for i, sp in enumerate(species):
            key = _canonical_species_key(sp.name)
            dset_n = density_map.get(key)
            dset_t = temperature_map.get(key)
            if dset_n is None or dset_t is None:
                raise KeyError(f"Do not know how to map species {sp.name!r} into NTSS flat-file datasets")
            if dset_n not in f or dset_t not in f:
                raise KeyError(f"Missing datasets {dset_n!r} / {dset_t!r} in {h5_path}")
            density[i] = np.asarray(f[dset_n][()], dtype=float)
            temperature[i] = np.asarray(f[dset_t][()], dtype=float)
    return ProfileSnapshot(rho=rho, density=density, temperature=temperature, er=er, time_value=None)


def load_neopax_snapshot(h5_path: Path, species: list[SpeciesMeta], *, time_index: int) -> ProfileSnapshot:
    try:
        return _infer_transport_snapshot(h5_path, time_index=time_index)
    except KeyError:
        return _infer_ntss_snapshot(h5_path, species)


def _safe_log_gradient(values: np.ndarray, rho: np.ndarray, *, floor: float) -> np.ndarray:
    arr = np.maximum(np.asarray(values, dtype=float), float(floor))
    logr = np.log(arr)
    return -np.gradient(logr, np.asarray(rho, dtype=float), edge_order=2)


def _safe_log_gradient_torflux(values: np.ndarray, rho: np.ndarray, *, floor: float) -> np.ndarray:
    arr = np.maximum(np.asarray(values, dtype=float), float(floor))
    logr = np.log(arr)
    torflux = np.asarray(rho, dtype=float) ** 2
    return -np.gradient(logr, torflux, edge_order=2)


def _select_reference_ion_index(species: list[SpeciesMeta], preferred_name: str | None = None) -> int:
    if preferred_name is not None:
        key = _canonical_species_key(preferred_name)
        for idx, sp in enumerate(species):
            if _canonical_species_key(sp.name) == key:
                return idx
        raise ValueError(f"reference ion {preferred_name!r} not found in NEOPAX species list")
    for idx, sp in enumerate(species):
        if sp.charge > 0.0:
            return idx
    raise ValueError("No positively charged species found; cannot choose a reference ion")


def _parse_index_list(text: str | None) -> list[int] | None:
    if text is None or not text.strip():
        return None
    out = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.append(int(chunk))
    return out


def _choose_radius_indices(
    rho: np.ndarray,
    *,
    explicit: list[int] | None,
    rho_min: float,
    rho_max: float,
    num_radii: int,
) -> list[int]:
    if explicit is not None:
        idxs = sorted(set(int(i) for i in explicit))
        for idx in idxs:
            if idx < 0 or idx >= rho.size:
                raise IndexError(f"rho index {idx} out of range [0, {rho.size - 1}]")
        return idxs
    mask = (rho >= float(rho_min)) & (rho <= float(rho_max))
    candidates = np.where(mask)[0]
    if candidates.size == 0:
        raise ValueError("No radii satisfy the requested rho range")
    if int(num_radii) >= candidates.size:
        return [int(v) for v in candidates]
    picks = np.linspace(0, candidates.size - 1, int(num_radii))
    return sorted(set(int(candidates[int(round(p))]) for p in picks))


def _build_manifest(
    *,
    neopax_result: Path,
    neopax_config: Path,
    spectrax_root: Path,
    output_dir: Path,
    snapshot: ProfileSnapshot,
    species: list[SpeciesMeta],
    electron_model: str,
    reference_ion: str | None,
    rho_indices: list[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    rho = np.asarray(snapshot.rho, dtype=float)
    density = np.asarray(snapshot.density, dtype=float)
    temperature = np.asarray(snapshot.temperature, dtype=float)
    er = np.asarray(snapshot.er, dtype=float)
    if density.shape[0] != len(species) or temperature.shape[0] != len(species):
        raise ValueError("NEOPAX HDF5 species dimension does not match the species list from the TOML")

    gradient_coordinate = str(args.gradient_coordinate).strip().lower()
    gradient_scale = float(args.gradient_scale)
    if gradient_coordinate not in {"rho", "torflux", "rho_with_scale"}:
        raise ValueError("--gradient-coordinate must be one of: rho, torflux, rho_with_scale")

    if gradient_coordinate == "rho":
        density_grad = np.vstack([
            _safe_log_gradient(density[i], rho, floor=float(args.density_floor))
            for i in range(density.shape[0])
        ])
        temperature_grad = np.vstack([
            _safe_log_gradient(temperature[i], rho, floor=float(args.temperature_floor))
            for i in range(temperature.shape[0])
        ])
    elif gradient_coordinate == "torflux":
        density_grad = np.vstack([
            _safe_log_gradient_torflux(density[i], rho, floor=float(args.density_floor))
            for i in range(density.shape[0])
        ])
        temperature_grad = np.vstack([
            _safe_log_gradient_torflux(temperature[i], rho, floor=float(args.temperature_floor))
            for i in range(temperature.shape[0])
        ])
    else:
        density_grad = gradient_scale * np.vstack([
            _safe_log_gradient(density[i], rho, floor=float(args.density_floor))
            for i in range(density.shape[0])
        ])
        temperature_grad = gradient_scale * np.vstack([
            _safe_log_gradient(temperature[i], rho, floor=float(args.temperature_floor))
            for i in range(temperature.shape[0])
        ])

    ref_idx = _select_reference_ion_index(species, preferred_name=reference_ion)
    ref_density = np.maximum(density[ref_idx], float(args.density_floor))
    ref_temperature = np.maximum(temperature[ref_idx], float(args.temperature_floor))

    neopax_root = _infer_neopax_root(neopax_config)

    vmec_path = _resolve_relative(neopax_root, args.vmec_file_override)
    if vmec_path is None:
        cfg = _load_toml(neopax_config)
        geometry_cfg = cfg.get("geometry", {})
        vmec_path = _resolve_relative(neopax_root, geometry_cfg.get("vmec_file"))
    if vmec_path is None:
        raise ValueError("Could not resolve a VMEC file path from the NEOPAX config")

    booz_path = _resolve_relative(neopax_root, args.boozer_file_override)
    if booz_path is None:
        cfg = _load_toml(neopax_config)
        geometry_cfg = cfg.get("geometry", {})
        booz_path = _resolve_relative(neopax_root, geometry_cfg.get("boozer_file"))

    electron_idx = None
    for idx, sp in enumerate(species):
        if sp.charge < 0.0:
            electron_idx = idx
            break

    runs: list[dict[str, Any]] = []
    runtime_species_names: list[str] = []
    if str(electron_model).lower() == "adiabatic":
        runtime_species_names = [sp.name for sp in species if sp.charge > 0.0]
    elif str(electron_model).lower() == "kinetic":
        runtime_species_names = [sp.name for sp in species]
    else:
        raise ValueError("electron_model must be either 'adiabatic' or 'kinetic'")

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_geom_dir = output_dir / "geometry_cache"
    generated_geom_dir.mkdir(parents=True, exist_ok=True)

    for ordinal, rho_idx in enumerate(rho_indices):
        rho_val = float(rho[rho_idx])
        torflux = float(rho_val ** 2)
        ref_n = float(ref_density[rho_idx])
        ref_t = float(ref_temperature[rho_idx])
        if ref_n <= 0.0 or ref_t <= 0.0:
            raise ValueError(f"Reference density/temperature must stay positive at rho index {rho_idx}")

        runtime_species: list[dict[str, Any]] = []
        for sp_idx, sp in enumerate(species):
            include = str(electron_model).lower() == "kinetic" or sp.charge > 0.0
            if not include:
                continue
            is_electron = sp.charge < 0.0
            runtime_species.append(
                {
                    "name": sp.name,
                    "charge": float(sp.charge),
                    "mass": float(sp.mass_mp),
                    "density": float(density[sp_idx, rho_idx] / ref_n),
                    "temperature": float(temperature[sp_idx, rho_idx] / ref_t),
                    "tprim": float(args.tprim_scale * temperature_grad[sp_idx, rho_idx]),
                    "fprim": float(args.fprim_scale * density_grad[sp_idx, rho_idx]),
                    "nu": float(args.nu_electron if is_electron else args.nu_ion),
                    "density_physical": float(density[sp_idx, rho_idx]),
                    "temperature_physical": float(temperature[sp_idx, rho_idx]),
                }
            )

        tau_e = None
        if electron_idx is not None:
            te_val = max(float(temperature[electron_idx, rho_idx]), float(args.temperature_floor))
            tau_e = float(ref_t / te_val)
        elif str(electron_model).lower() == "adiabatic":
            tau_e = float(args.tau_e_override)

        base_name = f"rho_{rho_idx:03d}_r{rho_val:.4f}".replace(".", "p")
        output_prefix = str((output_dir / base_name).resolve())
        geometry_file = str((generated_geom_dir / f"{base_name}.eik.nc").resolve())
        runs.append(
            {
                "index": ordinal,
                "rho_index": int(rho_idx),
                "rho": rho_val,
                "torflux": torflux,
                "Er": float(er[rho_idx]),
                "output_prefix": output_prefix,
                "geometry_file": geometry_file,
                "runtime_species": runtime_species,
                "tau_e": tau_e,
            }
        )

    return {
        "schema_version": 1,
        "neopax_result": str(neopax_result.resolve()),
        "neopax_config": str(neopax_config.resolve()),
        "spectrax_root": str(spectrax_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "snapshot_time": snapshot.time_value,
        "electron_model": str(electron_model).lower(),
        "runtime_species_names": runtime_species_names,
        "booz_file": booz_path,
        "vmec_file": vmec_path,
        "grid": {
            "Nx": int(args.nx),
            "Ny": int(args.ny),
            "Nz": int(args.nz),
            "Lx": float(args.lx),
            "Ly": float(args.ly),
            "boundary": str(args.boundary),
            "y0": float(args.y0),
            "ntheta": int(args.ntheta),
            "nperiod": int(args.nperiod),
        },
        "time": {
            "t_max": float(args.t_max),
            "dt": float(args.dt),
            "method": str(args.method),
            "use_diffrax": bool(args.use_diffrax),
            "sample_stride": int(args.sample_stride),
            "diagnostics_stride": int(args.diagnostics_stride),
            "fixed_dt": bool(args.fixed_dt),
            "cfl": float(args.cfl),
            "state_sharding": None if args.state_sharding.lower() == "none" else str(args.state_sharding),
        },
        "physics": {
            "electrostatic": True,
            "electromagnetic": False,
            "adiabatic_electrons": str(electron_model).lower() == "adiabatic",
            "collisions": True,
            "hypercollisions": True,
            "beta": float(args.beta),
        },
        "collisions": {
            "nu_hermite": float(args.nu_hermite),
            "nu_laguerre": float(args.nu_laguerre),
            "nu_hyper": float(args.nu_hyper),
            "p_hyper": float(args.p_hyper),
            "hypercollisions_const": float(args.hypercollisions_const),
            "hypercollisions_kz": float(args.hypercollisions_kz),
            "D_hyper": float(args.d_hyper),
            "damp_ends_amp": float(args.damp_ends_amp),
            "damp_ends_widthfrac": float(args.damp_ends_widthfrac),
        },
        "normalization": {
            "contract": str(args.normalization_contract),
            "diagnostic_norm": str(args.diagnostic_norm),
        },
        "terms": {
            "streaming": 1.0,
            "mirror": 1.0,
            "curvature": 1.0,
            "gradb": 1.0,
            "diamagnetic": 1.0,
            "collisions": 1.0,
            "hypercollisions": 1.0,
            "hyperdiffusion": float(args.hyperdiffusion),
            "end_damping": 1.0,
            "apar": 0.0,
            "bpar": 0.0,
            "nonlinear": 1.0,
        },
        "run": {
            "ky": float(args.ky),
            "Nl": int(args.nl),
            "Nm": int(args.nm),
        },
        "gradient_mapping": {
            "coordinate": gradient_coordinate,
            "scale": gradient_scale,
        },
        "init": {
            "init_field": str(args.init_field),
            "init_amp": float(args.init_amp),
            "gaussian_init": False,
            "init_single": False,
        },
        "geometry": {
            "model": "vmec",
            "alpha": float(args.alpha),
            "npol": float(args.npol),
            "geometry_backend": "internal",
        },
        "species_meta": [
            {"name": sp.name, "charge": sp.charge, "mass_mp": sp.mass_mp}
            for sp in species
        ],
        "runs": runs,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_runs_csv(path: Path, manifest: dict[str, Any]) -> None:
    rows = []
    for run in manifest["runs"]:
        row = {
            "index": run["index"],
            "rho_index": run["rho_index"],
            "rho": run["rho"],
            "torflux": run["torflux"],
            "Er": run["Er"],
            "output_prefix": run["output_prefix"],
            "geometry_file": run["geometry_file"],
        }
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["index"])
        writer.writeheader()
        writer.writerows(rows)


def _build_normalization_audit_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    electron_model = str(manifest["electron_model"])
    gradient_coordinate = str(manifest.get("gradient_mapping", {}).get("coordinate", "rho"))
    gradient_scale = float(manifest.get("gradient_mapping", {}).get("scale", 1.0))
    for run in manifest["runs"]:
        rho = float(run["rho"])
        torflux = float(run["torflux"])
        tau_e = run["tau_e"]
        for sp in run["runtime_species"]:
            grad_rho = float(sp["tprim"])
            dens_grad_rho = float(sp["fprim"])
            if abs(rho) > 1.0e-12:
                grad_torflux = grad_rho / (2.0 * rho)
                dens_grad_torflux = dens_grad_rho / (2.0 * rho)
            else:
                grad_torflux = math.nan
                dens_grad_torflux = math.nan
            rows.append(
                {
                    "run_index": int(run["index"]),
                    "rho_index": int(run["rho_index"]),
                    "rho": rho,
                    "torflux": torflux,
                    "electron_model": electron_model,
                    "species_name": str(sp["name"]),
                    "charge": float(sp["charge"]),
                    "mass_mp": float(sp["mass"]),
                    "density_physical": float(sp["density_physical"]),
                    "temperature_physical": float(sp["temperature_physical"]),
                    "density_normalized_to_ref_ion": float(sp["density"]),
                    "temperature_normalized_to_ref_ion": float(sp["temperature"]),
                    "gradient_coordinate": gradient_coordinate,
                    "gradient_scale": gradient_scale,
                    "fprim_used": dens_grad_rho,
                    "tprim_used": grad_rho,
                    "fprim_if_interpreted_per_torflux": dens_grad_torflux,
                    "tprim_if_interpreted_per_torflux": grad_torflux,
                    "tau_e_used": math.nan if tau_e is None else float(tau_e),
                    "Er_input": float(run["Er"]),
                }
            )
    return rows


def _write_normalization_audit(output_dir: Path, manifest: dict[str, Any]) -> None:
    rows = _build_normalization_audit_rows(manifest)
    if not rows:
        return
    csv_path = output_dir / "normalization_audit.csv"
    json_path = output_dir / "normalization_audit.json"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "assumptions": {
            "reference_species_normalization": "All runtime densities and temperatures are normalized to the chosen reference ion at the same radius.",
            "gradient_coordinate_used": (
                "tprim/fprim are computed according to manifest.gradient_mapping.coordinate: "
                "'rho' -> -d ln(X)/d rho, "
                "'torflux' -> -d ln(X)/d torflux, "
                "'rho_with_scale' -> gradient_scale * (-d ln(X)/d rho)."
            ),
            "torflux_mapping": "torflux is currently assumed to be rho^2.",
            "alternate_torflux_gradient_columns": "The audit CSV also includes the derived values tprim_if_interpreted_per_torflux and fprim_if_interpreted_per_torflux.",
        },
        "rows": rows,
    }
    _write_json(json_path, payload)


def _import_spectrax_runtime(spectrax_root: Path):
    src_path = spectrax_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from spectraxgk.config import GeometryConfig, GridConfig, InitializationConfig, TimeConfig
    from spectraxgk.runtime_artifacts import run_runtime_nonlinear_with_artifacts
    from spectraxgk.runtime_config import (
        RuntimeCollisionConfig,
        RuntimeConfig,
        RuntimeNormalizationConfig,
        RuntimePhysicsConfig,
        RuntimeSpeciesConfig,
        RuntimeTermsConfig,
    )

    return {
        "GeometryConfig": GeometryConfig,
        "GridConfig": GridConfig,
        "InitializationConfig": InitializationConfig,
        "TimeConfig": TimeConfig,
        "run_runtime_nonlinear_with_artifacts": run_runtime_nonlinear_with_artifacts,
        "RuntimeCollisionConfig": RuntimeCollisionConfig,
        "RuntimeConfig": RuntimeConfig,
        "RuntimeNormalizationConfig": RuntimeNormalizationConfig,
        "RuntimePhysicsConfig": RuntimePhysicsConfig,
        "RuntimeSpeciesConfig": RuntimeSpeciesConfig,
        "RuntimeTermsConfig": RuntimeTermsConfig,
    }


def _build_runtime_config_from_manifest(manifest: dict[str, Any], run_spec: dict[str, Any]):
    spectrax_root = Path(manifest["spectrax_root"])
    api = _import_spectrax_runtime(spectrax_root)
    GridConfig = api["GridConfig"]
    TimeConfig = api["TimeConfig"]
    GeometryConfig = api["GeometryConfig"]
    InitializationConfig = api["InitializationConfig"]
    RuntimeConfig = api["RuntimeConfig"]
    RuntimeSpeciesConfig = api["RuntimeSpeciesConfig"]
    RuntimePhysicsConfig = api["RuntimePhysicsConfig"]
    RuntimeCollisionConfig = api["RuntimeCollisionConfig"]
    RuntimeNormalizationConfig = api["RuntimeNormalizationConfig"]
    RuntimeTermsConfig = api["RuntimeTermsConfig"]

    grid_cfg = manifest["grid"]
    time_cfg = manifest["time"]
    geom_cfg = manifest["geometry"]
    init_cfg = manifest["init"]
    phys_cfg = manifest["physics"]
    coll_cfg = manifest["collisions"]
    norm_cfg = manifest["normalization"]
    terms_cfg = manifest["terms"]

    species = tuple(
        RuntimeSpeciesConfig(
            name=str(sp["name"]),
            charge=float(sp["charge"]),
            mass=float(sp["mass"]),
            density=float(sp["density"]),
            temperature=float(sp["temperature"]),
            tprim=float(sp["tprim"]),
            fprim=float(sp["fprim"]),
            nu=float(sp["nu"]),
            kinetic=True,
        )
        for sp in run_spec["runtime_species"]
    )

    cfg = RuntimeConfig(
        grid=GridConfig(
            Nx=int(grid_cfg["Nx"]),
            Ny=int(grid_cfg["Ny"]),
            Nz=int(grid_cfg["Nz"]),
            Lx=float(grid_cfg["Lx"]),
            Ly=float(grid_cfg["Ly"]),
            boundary=str(grid_cfg["boundary"]),
            y0=float(grid_cfg["y0"]),
            ntheta=int(grid_cfg["ntheta"]),
            nperiod=int(grid_cfg["nperiod"]),
        ),
        time=TimeConfig(
            t_max=float(time_cfg["t_max"]),
            dt=float(time_cfg["dt"]),
            method=str(time_cfg["method"]),
            use_diffrax=bool(time_cfg["use_diffrax"]),
            sample_stride=int(time_cfg["sample_stride"]),
            diagnostics_stride=int(time_cfg["diagnostics_stride"]),
            fixed_dt=bool(time_cfg["fixed_dt"]),
            cfl=float(time_cfg["cfl"]),
            state_sharding=time_cfg["state_sharding"],
        ),
        geometry=GeometryConfig(
            model=str(geom_cfg["model"]),
            vmec_file=str(manifest["vmec_file"]),
            geometry_file=str(run_spec["geometry_file"]),
            geometry_backend=str(geom_cfg["geometry_backend"]),
            torflux=float(run_spec["torflux"]),
            alpha=float(geom_cfg["alpha"]),
            npol=float(geom_cfg["npol"]),
        ),
        init=InitializationConfig(
            init_field=str(init_cfg["init_field"]),
            init_amp=float(init_cfg["init_amp"]),
            gaussian_init=bool(init_cfg["gaussian_init"]),
            init_single=bool(init_cfg["init_single"]),
        ),
        species=species,
        physics=RuntimePhysicsConfig(
            linear=False,
            nonlinear=True,
            electrostatic=bool(phys_cfg["electrostatic"]),
            electromagnetic=bool(phys_cfg["electromagnetic"]),
            adiabatic_electrons=bool(phys_cfg["adiabatic_electrons"]),
            tau_e=1.0 if run_spec["tau_e"] is None else float(run_spec["tau_e"]),
            beta=float(phys_cfg["beta"]),
            collisions=bool(phys_cfg["collisions"]),
            hypercollisions=bool(phys_cfg["hypercollisions"]),
        ),
        collisions=RuntimeCollisionConfig(
            nu_hermite=float(coll_cfg["nu_hermite"]),
            nu_laguerre=float(coll_cfg["nu_laguerre"]),
            nu_hyper=float(coll_cfg["nu_hyper"]),
            p_hyper=float(coll_cfg["p_hyper"]),
            hypercollisions_const=float(coll_cfg["hypercollisions_const"]),
            hypercollisions_kz=float(coll_cfg["hypercollisions_kz"]),
            D_hyper=float(coll_cfg["D_hyper"]),
            damp_ends_amp=float(coll_cfg["damp_ends_amp"]),
            damp_ends_widthfrac=float(coll_cfg["damp_ends_widthfrac"]),
        ),
        normalization=RuntimeNormalizationConfig(
            contract=str(norm_cfg["contract"]),
            diagnostic_norm=str(norm_cfg["diagnostic_norm"]),
        ),
        terms=RuntimeTermsConfig(**terms_cfg),
    )
    return cfg, api["run_runtime_nonlinear_with_artifacts"]


def cmd_prepare(args: argparse.Namespace) -> int:
    neopax_result = Path(args.neopax_result).resolve()
    neopax_config = Path(args.neopax_config).resolve()
    spectrax_root = Path(args.spectrax_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    cfg = _load_toml(neopax_config)
    species = _parse_species_from_neopax_config(cfg)
    snapshot = load_neopax_snapshot(neopax_result, species, time_index=int(args.time_index))
    rho_indices = _choose_radius_indices(
        snapshot.rho,
        explicit=_parse_index_list(args.rho_indices),
        rho_min=float(args.rho_min),
        rho_max=float(args.rho_max),
        num_radii=int(args.num_radii),
    )
    manifest = _build_manifest(
        neopax_result=neopax_result,
        neopax_config=neopax_config,
        spectrax_root=spectrax_root,
        output_dir=output_dir,
        snapshot=snapshot,
        species=species,
        electron_model=args.electron_model,
        reference_ion=args.reference_ion,
        rho_indices=rho_indices,
        args=args,
    )
    manifest_path = output_dir / "manifest.json"
    csv_path = output_dir / "runs.csv"
    _write_json(manifest_path, manifest)
    _write_runs_csv(csv_path, manifest)
    _write_normalization_audit(output_dir, manifest)
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote run table: {csv_path}")
    print(f"Wrote normalization audit: {output_dir / 'normalization_audit.csv'}")
    print(f"Prepared {len(manifest['runs'])} SPECTRAX-GK runs from {neopax_result.name}")
    return 0


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_run_one(args: argparse.Namespace) -> int:
    manifest = _load_manifest(Path(args.manifest).resolve())
    run_spec = manifest["runs"][int(args.index)]
    cfg, runner = _build_runtime_config_from_manifest(manifest, run_spec)
    result, paths = runner(
        cfg,
        out=run_spec["output_prefix"],
        ky_target=float(manifest["run"]["ky"]),
        Nl=int(manifest["run"]["Nl"]),
        Nm=int(manifest["run"]["Nm"]),
        dt=float(manifest["time"]["dt"]),
        steps=None,
        method=str(manifest["time"]["method"]),
        sample_stride=int(manifest["time"]["sample_stride"]),
        diagnostics_stride=int(manifest["time"]["diagnostics_stride"]),
        diagnostics=True,
        show_progress=False,
    )
    diag = result.diagnostics
    heat_last = float(np.asarray(diag.heat_flux_t)[-1]) if diag is not None and np.asarray(diag.heat_flux_t).size else float("nan")
    pflux_last = float(np.asarray(diag.particle_flux_t)[-1]) if diag is not None and np.asarray(diag.particle_flux_t).size else float("nan")
    print(
        json.dumps(
            {
                "index": int(run_spec["index"]),
                "rho": float(run_spec["rho"]),
                "output_prefix": run_spec["output_prefix"],
                "paths": paths,
                "heat_flux_last": heat_last,
                "particle_flux_last": pflux_last,
            }
        )
    )
    return 0


def _launch_subprocess(
    *,
    manifest_path: Path,
    index: int,
    env_overrides: dict[str, str],
) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env.update(env_overrides)
    return subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "run-one",
            "--manifest",
            str(manifest_path),
            "--index",
            str(index),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )


def cmd_run(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve()
    manifest = _load_manifest(manifest_path)
    runs = manifest["runs"]
    max_parallel = int(args.max_parallel)
    if max_parallel < 1:
        raise ValueError("--max-parallel must be >= 1")

    mode = str(args.backend).lower()
    if mode not in {"cpu", "gpu"}:
        raise ValueError("--backend must be 'cpu' or 'gpu'")

    gpu_ids = []
    if mode == "gpu":
        gpu_ids = _parse_index_list(args.gpu_ids) or [0]
        max_parallel = min(max_parallel, len(gpu_ids))

    pending = list(range(len(runs)))
    active: dict[Any, tuple[subprocess.Popen[str], int, dict[str, str]]] = {}
    failures = 0

    def _env_for_slot(slot: int) -> dict[str, str]:
        env = {
            "PYTHONPATH": str((Path(manifest["spectrax_root"]) / "src").resolve()),
        }
        if mode == "gpu":
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[slot % len(gpu_ids)])
            env["JAX_PLATFORM_NAME"] = "gpu"
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        else:
            env["JAX_PLATFORM_NAME"] = "cpu"
            env["OMP_NUM_THREADS"] = str(int(args.threads_per_run))
        return env

    with ThreadPoolExecutor(max_workers=max_parallel) as _unused:
        while pending or active:
            while pending and len(active) < max_parallel:
                slot = len(active)
                run_idx = pending.pop(0)
                env = _env_for_slot(slot)
                proc = _launch_subprocess(manifest_path=manifest_path, index=run_idx, env_overrides=env)
                active[proc] = (proc, run_idx, env)
                where = env.get("CUDA_VISIBLE_DEVICES", f"cpu x{env.get('OMP_NUM_THREADS', '1')}")
                print(f"started run {run_idx} on {where}")

            if not active:
                break

            done = []
            for proc, run_idx, env in list(active.values()):
                rc = proc.poll()
                if rc is None:
                    continue
                stdout, stderr = proc.communicate()
                if rc == 0:
                    line = stdout.strip().splitlines()[-1] if stdout.strip() else ""
                    print(f"finished run {run_idx}: {line}")
                else:
                    failures += 1
                    print(f"run {run_idx} failed with code {rc}")
                    if stdout.strip():
                        print(stdout.strip())
                    if stderr.strip():
                        print(stderr.strip())
                done.append(proc)
            for proc in done:
                active.pop(proc, None)

            if active and not done:
                wait_timeout = float(args.poll_interval)
                import time

                time.sleep(wait_timeout)

    if failures:
        print(f"{failures} runs failed")
        return 1
    print("All runs finished successfully")
    return 0


def _read_last_row_csv(path: Path) -> dict[str, float]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    if data.size == 0:
        raise ValueError(f"No rows found in diagnostics CSV {path}")
    row = data[-1] if getattr(data, "shape", ()) else data
    out: dict[str, float] = {}
    for name in data.dtype.names or ():
        out[str(name)] = float(row[name])
    return out


def cmd_collect(args: argparse.Namespace) -> int:
    manifest = _load_manifest(Path(args.manifest).resolve())
    runs = manifest["runs"]
    species_names = list(manifest["runtime_species_names"])
    out_h5 = Path(args.out).resolve()
    out_h5.parent.mkdir(parents=True, exist_ok=True)

    n = len(runs)
    rho = np.full(n, np.nan, dtype=float)
    rho_index = np.full(n, -1, dtype=int)
    torflux = np.full(n, np.nan, dtype=float)
    er = np.full(n, np.nan, dtype=float)
    heat_flux = np.full(n, np.nan, dtype=float)
    particle_flux = np.full(n, np.nan, dtype=float)
    heat_flux_species = np.full((n, len(species_names)), np.nan, dtype=float)
    particle_flux_species = np.full((n, len(species_names)), np.nan, dtype=float)

    for i, run in enumerate(runs):
        rho[i] = float(run["rho"])
        rho_index[i] = int(run["rho_index"])
        torflux[i] = float(run["torflux"])
        er[i] = float(run["Er"])
        diag_csv = Path(f"{run['output_prefix']}.diagnostics.csv")
        if not diag_csv.exists():
            print(f"skipping missing diagnostics: {diag_csv}")
            continue
        row = _read_last_row_csv(diag_csv)
        heat_flux[i] = row.get("heat_flux", math.nan)
        particle_flux[i] = row.get("particle_flux", math.nan)
        for s_idx, _name in enumerate(species_names):
            heat_flux_species[i, s_idx] = row.get(f"heat_flux_s{s_idx}", math.nan)
            particle_flux_species[i, s_idx] = row.get(f"particle_flux_s{s_idx}", math.nan)

    with h5py.File(out_h5, "w") as f:
        f.create_dataset("rho", data=rho)
        f.create_dataset("rho_index", data=rho_index)
        f.create_dataset("torflux", data=torflux)
        f.create_dataset("Er", data=er)
        f.create_dataset("heat_flux_total", data=heat_flux)
        f.create_dataset("particle_flux_total", data=particle_flux)
        grp = f.create_group("species")
        dt = h5py.string_dtype(encoding="utf-8")
        grp.create_dataset("names", data=np.asarray(species_names, dtype=object), dtype=dt)
        grp.create_dataset("heat_flux", data=heat_flux_species)
        grp.create_dataset("particle_flux", data=particle_flux_species)
        meta = f.create_group("meta")
        meta.attrs["manifest"] = str(Path(args.manifest).resolve())
        meta.attrs["electron_model"] = str(manifest["electron_model"])
        meta.attrs["neopax_result"] = str(manifest["neopax_result"])
        meta.attrs["neopax_config"] = str(manifest["neopax_config"])

    print(f"Wrote collected flux summary: {out_h5}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare", help="Create a SPECTRAX run manifest from a NEOPAX result")
    prep.add_argument("--neopax-result", required=True, help="Path to a NEOPAX HDF5 result file")
    prep.add_argument("--neopax-config", required=True, help="Path to the originating NEOPAX TOML")
    prep.add_argument("--spectrax-root", default=str(DEFAULT_SPECTRAX_ROOT), help="Path to the SPECTRAX-GK checkout")
    prep.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for the manifest and SPECTRAX outputs")
    prep.add_argument("--time-index", type=int, default=-1, help="Time slice for transport_solution.h5 inputs; default: final")
    prep.add_argument("--electron-model", choices=("adiabatic", "kinetic"), default="adiabatic")
    prep.add_argument("--reference-ion", default=None, help="Species name used as the normalization reference ion")
    prep.add_argument("--rho-indices", default=None, help="Explicit comma-separated rho indices, e.g. 5,10,20")
    prep.add_argument("--rho-min", type=float, default=0.15)
    prep.add_argument("--rho-max", type=float, default=0.85)
    prep.add_argument("--num-radii", type=int, default=5)
    prep.add_argument("--vmec-file-override", default=None, help="Override the VMEC path from the NEOPAX config")
    prep.add_argument("--boozer-file-override", default=None, help="Reserved for later internal coupling metadata")
    prep.add_argument("--density-floor", type=float, default=1.0e-8)
    prep.add_argument("--temperature-floor", type=float, default=1.0e-8)
    prep.add_argument("--gradient-coordinate", choices=("rho", "torflux", "rho_with_scale"), default="rho")
    prep.add_argument("--gradient-scale", type=float, default=1.0)
    prep.add_argument("--tprim-scale", type=float, default=1.0)
    prep.add_argument("--fprim-scale", type=float, default=1.0)
    prep.add_argument("--tau-e-override", type=float, default=1.0)
    prep.add_argument("--nu-ion", type=float, default=0.01)
    prep.add_argument("--nu-electron", type=float, default=0.0)
    prep.add_argument("--nx", type=int, default=96)
    prep.add_argument("--ny", type=int, default=96)
    prep.add_argument("--nz", type=int, default=48)
    prep.add_argument("--lx", type=float, default=62.8)
    prep.add_argument("--ly", type=float, default=62.8)
    prep.add_argument("--boundary", default="fix aspect")
    prep.add_argument("--y0", type=float, default=21.0)
    prep.add_argument("--ntheta", type=int, default=48)
    prep.add_argument("--nperiod", type=int, default=1)
    prep.add_argument("--t-max", type=float, default=200.0)
    prep.add_argument("--dt", type=float, default=0.1)
    prep.add_argument("--method", default="rk3")
    prep.add_argument("--use-diffrax", action="store_true")
    prep.add_argument("--fixed-dt", action="store_true")
    prep.add_argument("--sample-stride", type=int, default=50)
    prep.add_argument("--diagnostics-stride", type=int, default=50)
    prep.add_argument("--cfl", type=float, default=1.0)
    prep.add_argument("--state-sharding", default="none", help="none, auto, ky, ...; used inside a single SPECTRAX run")
    prep.add_argument("--ky", type=float, default=1.0 / 21.0)
    prep.add_argument("--nl", type=int, default=4)
    prep.add_argument("--nm", type=int, default=8)
    prep.add_argument("--init-field", default="density")
    prep.add_argument("--init-amp", type=float, default=1.0e-3)
    prep.add_argument("--alpha", type=float, default=0.0)
    prep.add_argument("--npol", type=float, default=1.0)
    prep.add_argument("--beta", type=float, default=0.0)
    prep.add_argument("--nu-hermite", type=float, default=1.0)
    prep.add_argument("--nu-laguerre", type=float, default=2.0)
    prep.add_argument("--nu-hyper", type=float, default=0.0)
    prep.add_argument("--p-hyper", type=float, default=4.0)
    prep.add_argument("--hypercollisions-const", type=float, default=0.0)
    prep.add_argument("--hypercollisions-kz", type=float, default=1.0)
    prep.add_argument("--d-hyper", type=float, default=0.05)
    prep.add_argument("--damp-ends-amp", type=float, default=0.1)
    prep.add_argument("--damp-ends-widthfrac", type=float, default=0.125)
    prep.add_argument("--hyperdiffusion", type=float, default=1.0)
    prep.add_argument("--normalization-contract", default="kinetic")
    prep.add_argument("--diagnostic-norm", default="gx")
    prep.set_defaults(func=cmd_prepare)

    run = sub.add_parser("run", help="Execute runs from a previously prepared manifest")
    run.add_argument("--manifest", required=True)
    run.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    run.add_argument("--gpu-ids", default="0", help="Comma-separated CUDA device ids for round-robin scheduling")
    run.add_argument("--max-parallel", type=int, default=1)
    run.add_argument("--threads-per-run", type=int, default=1)
    run.add_argument("--poll-interval", type=float, default=2.0)
    run.set_defaults(func=cmd_run)

    run_one = sub.add_parser("run-one", help=argparse.SUPPRESS)
    run_one.add_argument("--manifest", required=True)
    run_one.add_argument("--index", required=True, type=int)
    run_one.set_defaults(func=cmd_run_one)

    collect = sub.add_parser("collect", help="Collect final SPECTRAX fluxes from diagnostics CSV files")
    collect.add_argument("--manifest", required=True)
    collect.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR / "flux_summary.h5"))
    collect.set_defaults(func=cmd_collect)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
