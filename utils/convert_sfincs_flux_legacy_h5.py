"""Add a zero axis point to legacy sfincs_jax flux scan HDF5 files.

This utility updates older ``sfincs_jax_flux_profiles.h5``-style files that
were written without the ``rho=0`` / ``r=0`` point when the scan skipped the
magnetic axis. It prepends a zero radial point and zero ``Gamma``, ``Q``, and
``Upar`` columns when needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


RADIAL_DATASETS = {"r", "rho", "rHat"}
FLUX_DATASETS = {"Gamma", "Q", "Upar"}


def _needs_padding(fin: h5py.File) -> bool:
    if "r" not in fin:
        raise KeyError("Input file must contain dataset 'r'.")
    r = np.asarray(fin["r"][()], dtype=np.float64)
    if r.ndim != 1:
        raise ValueError("Dataset 'r' must be 1D.")
    return r.size == 0 or not np.isclose(r[0], 0.0)


def _copy_dataset(name: str, fin: h5py.File, fout: h5py.File, *, pad_axis: bool, n_species: int | None) -> None:
    data = fin[name][()]

    if pad_axis and name in RADIAL_DATASETS:
        arr = np.asarray(data)
        if arr.ndim != 1:
            raise ValueError(f"Dataset '{name}' must be 1D to pad the axis point.")
        out = np.concatenate([np.asarray([0.0], dtype=arr.dtype), arr])
        fout.create_dataset(name, data=out)
        return

    if pad_axis and name in FLUX_DATASETS:
        arr = np.asarray(data)
        if arr.ndim != 2:
            raise ValueError(f"Dataset '{name}' must be 2D with shape (n_species, n_radial).")
        zeros = np.zeros((arr.shape[0], 1), dtype=arr.dtype)
        out = np.concatenate([zeros, arr], axis=1)
        fout.create_dataset(name, data=out)
        return

    fout.create_dataset(name, data=data)


def convert_file(src: Path, dst: Path, *, overwrite: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists() and not overwrite:
        raise FileExistsError(f"{dst} already exists. Use --overwrite to replace it.")

    with h5py.File(src, "r") as fin:
        pad_axis = _needs_padding(fin)
        n_species = None
        if "Gamma" in fin:
            gamma = np.asarray(fin["Gamma"][()])
            if gamma.ndim == 2:
                n_species = int(gamma.shape[0])

        with h5py.File(dst, "w") as fout:
            for key, value in fin.attrs.items():
                fout.attrs[key] = value
            fout.attrs["axis_zero_padded"] = bool(pad_axis)
            fout.attrs["legacy_axis_conversion"] = True
            fout.attrs["legacy_axis_conversion_source"] = str(src)

            for name in fin.keys():
                _copy_dataset(name, fin, fout, pad_axis=pad_axis, n_species=n_species)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_h5", help="Legacy HDF5 file to convert.")
    p.add_argument(
        "--output-h5",
        default=None,
        help="Destination HDF5 file. Default: write '<input stem>_with_axis.h5'.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting the output file if it already exists.",
    )
    p.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite the input file in place.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    src = Path(args.input_h5).expanduser().resolve()
    if args.in_place and args.output_h5 is not None:
        raise ValueError("--in-place and --output-h5 are mutually exclusive.")

    if args.in_place:
        dst = src.with_suffix(src.suffix + ".tmp")
        convert_file(src, dst, overwrite=True)
        dst.replace(src)
        print(f"rewrote {src}")
        return 0

    if args.output_h5 is None:
        dst = src.with_name(src.stem + "_with_axis" + src.suffix)
    else:
        dst = Path(args.output_h5).expanduser().resolve()

    convert_file(src, dst, overwrite=bool(args.overwrite))
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
