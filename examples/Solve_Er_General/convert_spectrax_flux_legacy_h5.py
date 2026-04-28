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
from pathlib import Path
from typing import Any

import h5py
import numpy as np


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


def convert_file(src: Path, dst: Path) -> None:
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
        meta.create_dataset("species_names", data=np.asarray(species_names, dtype=object), dtype=dt)
        meta.attrs["converted_from"] = str(src)
        meta.attrs["conversion"] = "legacy species/(heat_flux,particle_flux) transposed to root /(Q,Gamma)"
        meta.attrs["root_flux_layout"] = "r:(n_radial,), Gamma/Q/Upar:(n_species,n_radial)"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Legacy collected flux HDF5 file")
    parser.add_argument("--output", required=True, help="Converted output HDF5 file")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    convert_file(Path(args.input), Path(args.output))
    print(f"Converted {Path(args.input).resolve()} -> {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
