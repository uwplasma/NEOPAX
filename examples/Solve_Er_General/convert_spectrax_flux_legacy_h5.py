#!/usr/bin/env python3
"""Convert legacy collected SPECTRAX flux HDF5 files to the new NEOPAX layout.

Legacy input layout expected:
- /rho
- /rho_index
- /torflux
- /Er
- /heat_flux_total
- /particle_flux_total
- /species/names
- /species/heat_flux          shape (n_radial, n_species)
- /species/particle_flux      shape (n_radial, n_species)

New output layout written:
- /r                          shape (n_radial,)
- /Gamma                      shape (n_species, n_radial)
- /Q                          shape (n_species, n_radial)
- /Upar                       shape (n_species, n_radial)
- /rho
- /rho_index
- /torflux
- /Er
- /heat_flux_total
- /particle_flux_total
- /meta/species_names
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.constants import elementary_charge, proton_mass


NEOPAX_DENSITY_REFERENCE_M3 = 1.0e20
NEOPAX_TEMPERATURE_REFERENCE_EV = 1.0e3


def _read_string_array(dataset: Any) -> list[str]:
    values = dataset[...]
    out: list[str] = []
    for item in np.ravel(values):
        if isinstance(item, bytes):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return out


def _prepend_axis_zero_if_needed(
    rho: np.ndarray,
    rho_index: np.ndarray,
    torflux: np.ndarray,
    er: np.ndarray,
    heat_total: np.ndarray,
    particle_total: np.ndarray,
    gamma: np.ndarray,
    q: np.ndarray,
    upar: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if rho.size == 0:
        return rho, rho_index, torflux, er, heat_total, particle_total, gamma, q, upar
    expected = np.arange(1, rho.size + 1, dtype=int)
    if not np.array_equal(rho_index, expected):
        return rho, rho_index, torflux, er, heat_total, particle_total, gamma, q, upar
    rho = np.concatenate(([0.0], rho))
    rho_index = np.concatenate(([0], rho_index))
    torflux = np.concatenate(([0.0], torflux))
    er = np.concatenate(([0.0], er))
    heat_total = np.concatenate(([0.0], heat_total))
    particle_total = np.concatenate(([0.0], particle_total))
    gamma = np.concatenate((np.zeros((gamma.shape[0], 1), dtype=gamma.dtype), gamma), axis=1)
    q = np.concatenate((np.zeros((q.shape[0], 1), dtype=q.dtype), q), axis=1)
    upar = np.concatenate((np.zeros((upar.shape[0], 1), dtype=upar.dtype), upar), axis=1)
    return rho, rho_index, torflux, er, heat_total, particle_total, gamma, q, upar


def _thermal_speed_ms(temperature_keV: float, mass_mp: float) -> float:
    temp_eV = float(temperature_keV) * NEOPAX_TEMPERATURE_REFERENCE_EV
    mass_kg = float(mass_mp) * proton_mass
    return float(np.sqrt(2.0 * temp_eV * elementary_charge / mass_kg))


def _spectrax_flux_to_neopax_units(
    flux_gb: float,
    *,
    density_ref_state: float,
    temperature_ref_keV: float,
    mass_ref_mp: float,
    rho_star_physical: float,
    kind: str,
) -> float:
    n_ref_m3 = float(density_ref_state) * NEOPAX_DENSITY_REFERENCE_M3
    t_ref_eV = float(temperature_ref_keV) * NEOPAX_TEMPERATURE_REFERENCE_EV
    vth_ref = _thermal_speed_ms(float(temperature_ref_keV), float(mass_ref_mp))
    if kind == "Gamma":
        scale = n_ref_m3 * vth_ref * float(rho_star_physical) ** 2
    elif kind == "Q":
        scale = n_ref_m3 * t_ref_eV * vth_ref * float(rho_star_physical) ** 2
    else:
        raise ValueError(f"Unknown flux kind {kind!r}")
    return float(flux_gb) * float(scale)


def _load_manifest_for_legacy(fin: h5py.File, src: Path, manifest_override: Path | None) -> dict[str, Any] | None:
    manifest_path = manifest_override
    if manifest_path is None and "meta" in fin:
        meta = fin["meta"]
        manifest_attr = meta.attrs.get("manifest")
        if manifest_attr is not None:
            manifest_path = Path(str(manifest_attr))
    if manifest_path is None:
        return None
    manifest_path = manifest_path.expanduser()
    if not manifest_path.is_absolute():
        manifest_path = (src.parent / manifest_path).resolve()
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _normalized_from_manifest(
    *,
    manifest: dict[str, Any],
    rho_index: np.ndarray,
    species_names: list[str],
    q_raw: np.ndarray,
    gamma_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    runs = list(manifest.get("runs", []))
    runtime_species_names = list(manifest.get("runtime_species_names", []))
    full_species_names = [str(sp["name"]) for sp in manifest.get("species_meta", [])] or species_names
    runtime_to_full = {name: idx for idx, name in enumerate(full_species_names)}
    runs_by_rho_index = {int(run["rho_index"]): run for run in runs}
    q_out = np.zeros((len(full_species_names), q_raw.shape[1]), dtype=float)
    gamma_out = np.zeros((len(full_species_names), gamma_raw.shape[1]), dtype=float)

    ref_name = str(manifest.get("normalization", {}).get("reference_species_name", "")).strip().lower()
    a_minor_default = float(manifest.get("geometry", {}).get("a_minor", 1.0))
    for col, ridx in enumerate(rho_index):
        if int(ridx) == 0:
            continue
        run = runs_by_rho_index.get(int(ridx))
        if run is None:
            continue
        ref_runtime_species = next(
            (sp for sp in run["runtime_species"] if str(sp.get("name", "")).strip().lower() == ref_name),
            run["runtime_species"][0],
        )
        rho_star_physical = float(run.get("rho_star_physical", 1.0))
        a_minor = float(run.get("a_minor", a_minor_default))
        for runtime_idx, runtime_name in enumerate(runtime_species_names):
            full_idx = runtime_to_full.get(runtime_name)
            if full_idx is None or runtime_idx >= q_raw.shape[0] or runtime_idx >= gamma_raw.shape[0]:
                continue
            q_out[full_idx, col] = _spectrax_flux_to_neopax_units(
                q_raw[runtime_idx, col],
                density_ref_state=float(ref_runtime_species["density_reference_physical"]),
                temperature_ref_keV=float(ref_runtime_species["temperature_reference_physical"]),
                mass_ref_mp=float(ref_runtime_species["mass"]),
                rho_star_physical=rho_star_physical,
                kind="Q",
            ) * a_minor
            gamma_out[full_idx, col] = _spectrax_flux_to_neopax_units(
                gamma_raw[runtime_idx, col],
                density_ref_state=float(ref_runtime_species["density_reference_physical"]),
                temperature_ref_keV=float(ref_runtime_species["temperature_reference_physical"]),
                mass_ref_mp=float(ref_runtime_species["mass"]),
                rho_star_physical=rho_star_physical,
                kind="Gamma",
            ) * a_minor
    return q_out, gamma_out


def convert_file(src: Path, dst: Path, *, manifest_override: Path | None = None) -> None:
    src = src.resolve()
    dst = dst.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(src, "r") as fin:
        rho = np.asarray(fin["rho"][...], dtype=float)
        rho_index = np.asarray(fin["rho_index"][...], dtype=int) if "rho_index" in fin else np.arange(rho.size, dtype=int)
        torflux = np.asarray(fin["torflux"][...], dtype=float) if "torflux" in fin else rho**2
        er = np.asarray(fin["Er"][...], dtype=float) if "Er" in fin else np.zeros_like(rho)
        heat_total = np.asarray(fin["heat_flux_total"][...], dtype=float) if "heat_flux_total" in fin else np.full_like(rho, np.nan)
        particle_total = np.asarray(fin["particle_flux_total"][...], dtype=float) if "particle_flux_total" in fin else np.full_like(rho, np.nan)

        if "species" not in fin:
            raise KeyError("Legacy file must contain /species group")
        species = fin["species"]
        species_names = _read_string_array(species["names"])
        heat_species = np.asarray(species["heat_flux"][...], dtype=float)
        particle_species = np.asarray(species["particle_flux"][...], dtype=float)

        if heat_species.ndim != 2 or particle_species.ndim != 2:
            raise ValueError("Legacy /species/heat_flux and /species/particle_flux must be rank-2 arrays")
        if heat_species.shape != particle_species.shape:
            raise ValueError("Legacy species heat and particle flux arrays must have matching shapes")
        if heat_species.shape[0] != rho.size:
            raise ValueError("Legacy species arrays must have leading dimension n_radial")
        if heat_species.shape[1] != len(species_names):
            raise ValueError("Legacy species names count must match species-array width")

        gamma = np.asarray(particle_species.T, dtype=float)
        q = np.asarray(heat_species.T, dtype=float)
        upar = np.zeros_like(gamma)
        rho, rho_index, torflux, er, heat_total, particle_total, gamma, q, upar = _prepend_axis_zero_if_needed(
            rho,
            rho_index,
            torflux,
            er,
            heat_total,
            particle_total,
            gamma,
            q,
            upar,
        )
        gamma = np.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)
        q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
        upar = np.nan_to_num(upar, nan=0.0, posinf=0.0, neginf=0.0)
        er = np.nan_to_num(er, nan=0.0, posinf=0.0, neginf=0.0)
        heat_total = np.nan_to_num(heat_total, nan=0.0, posinf=0.0, neginf=0.0)
        particle_total = np.nan_to_num(particle_total, nan=0.0, posinf=0.0, neginf=0.0)
        manifest = _load_manifest_for_legacy(fin, src, manifest_override)
        normalized = False
        output_species_names = list(species_names)
        if manifest is not None:
            try:
                q_norm, gamma_norm = _normalized_from_manifest(
                    manifest=manifest,
                    rho_index=rho_index,
                    species_names=species_names,
                    q_raw=q,
                    gamma_raw=gamma,
                )
                if q_norm.shape[1] == q.shape[1] and gamma_norm.shape[1] == gamma.shape[1]:
                    q = np.nan_to_num(q_norm, nan=0.0, posinf=0.0, neginf=0.0)
                    gamma = np.nan_to_num(gamma_norm, nan=0.0, posinf=0.0, neginf=0.0)
                    upar = np.zeros_like(gamma)
                    output_species_names = [str(sp["name"]) for sp in manifest.get("species_meta", [])] or output_species_names
                    normalized = True
            except Exception:
                normalized = False

    with h5py.File(dst, "w") as fout:
        fout.create_dataset("r", data=rho)
        fout.create_dataset("Gamma", data=gamma)
        fout.create_dataset("Q", data=q)
        fout.create_dataset("Upar", data=upar)
        fout.create_dataset("rho", data=rho)
        fout.create_dataset("rho_index", data=rho_index)
        fout.create_dataset("torflux", data=torflux)
        fout.create_dataset("Er", data=er)
        fout.create_dataset("heat_flux_total", data=heat_total)
        fout.create_dataset("particle_flux_total", data=particle_total)
        meta = fout.create_group("meta")
        dt = h5py.string_dtype(encoding="utf-8")
        meta.create_dataset("species_names", data=np.asarray(output_species_names, dtype=object), dtype=dt)
        meta.attrs["converted_from"] = str(src)
        meta.attrs["conversion"] = (
            "legacy species/(heat_flux,particle_flux) transposed to root /(Q,Gamma)"
            if not normalized
            else "legacy raw species fluxes converted to NEOPAX physical Gamma/Q using the legacy manifest metadata"
        )
        meta.attrs["nan_policy"] = "all NaN/inf values replaced by 0.0 during conversion"
        meta.attrs["normalized_to_neopax_units"] = bool(normalized)
        meta.attrs["root_flux_layout"] = "r:(n_radial,), Gamma/Q/Upar:(n_species,n_radial)"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Legacy collected flux HDF5 file")
    parser.add_argument("--output", required=True, help="Converted output HDF5 file")
    parser.add_argument("--manifest", default=None, help="Optional explicit legacy manifest.json to use for physical Gamma/Q normalization")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    convert_file(
        Path(args.input),
        Path(args.output),
        manifest_override=None if args.manifest is None else Path(args.manifest),
    )
    print(f"Converted {Path(args.input).resolve()} -> {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
