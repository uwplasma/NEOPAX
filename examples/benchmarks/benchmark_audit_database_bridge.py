"""Audit raw NTX HDF5 coefficients against NEOPAX bridge/load conventions.

This compares three layers at an exact stored scan node:
1. raw HDF5 coefficients from the NTX-produced file
2. expected bridged values from the NEOPAX load formulas
3. values reconstructed from the loaded NEOPAX database object / interpolation kernel
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import NEOPAX
from NEOPAX._database import Monoenergetic
from NEOPAX._monoenergetic_interpolators import monoenergetic_interpolation_kernel
from NEOPAX._orchestrator import build_runtime_context

DEFAULT_DATABASE_CONFIG = Path("examples/benchmarks/Calculate_Fluxes_noHe_ntx_database_benchmark.toml")


def _prepare_config(config_path: Path, *, device: str, er_init_mode: str):
    config = NEOPAX.prepare_config(config_path, device=device)
    config = copy.deepcopy(config)
    config.setdefault("profiles", {})["er_initialization_mode"] = str(er_init_mode)
    return config


def _find_database_path(config: dict) -> Path:
    path = config["neoclassical"]["neoclassical_file"]
    return Path(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--database-config", default=str(DEFAULT_DATABASE_CONFIG))
    parser.add_argument(
        "--er-init-mode",
        default="analytical",
        choices=["keep", "analytical", "ambipolar_min_entropy"],
    )
    parser.add_argument("--rho-index", type=int, default=1)
    parser.add_argument("--nu-index", type=int, default=5)
    parser.add_argument("--er-index", type=int, default=5)
    args = parser.parse_args()

    cfg = _prepare_config(Path(args.database_config), device=args.device, er_init_mode=args.er_init_mode)
    runtime, _state = build_runtime_context(cfg)
    db_path = _find_database_path(cfg)
    db_abs = (ROOT / db_path).resolve() if not db_path.is_absolute() else db_path.resolve()

    with h5py.File(db_abs, "r") as handle:
        rho = jnp.asarray(handle["rho"][()])
        nu_v = jnp.asarray(handle["nu_v"][()])
        Er = jnp.asarray(handle["Er"][()])
        Es = jnp.asarray(handle["Es"][()])
        drds = jnp.asarray(handle["drds"][()])
        D11 = jnp.asarray(handle["D11"][()])
        D13 = jnp.asarray(handle["D13"][()])
        D33 = jnp.asarray(handle["D33"][()])

    ir = int(args.rho_index)
    inu = int(args.nu_index)
    ier = int(args.er_index)

    raw_d11 = float(D11[ir, inu, ier])
    raw_d13 = float(D13[ir, inu, ier])
    raw_d33 = float(D33[ir, inu, ier])
    drds_value = float(drds[ir])
    nu_value = float(nu_v[inu])
    er_value = float(Er[ir, ier])
    es_value = float(Es[ir, ier])
    r_value = float(runtime.geometry.a_b * rho[ir])

    bridged_d11 = raw_d11 * drds_value**2
    bridged_d13 = raw_d13 * drds_value
    bridged_d33 = raw_d33 * nu_value
    bridged_d11_log = np.log10(max(bridged_d11, 1.0e-20))
    bridged_er_axis = np.log10(max(1.0e-8, abs(float(Er[0, ier])) / max(r_value, 1.0e-30)))

    generic_db = Monoenergetic.read_ntx(runtime.geometry.a_b, str(db_abs))
    generic_d11_log = float(generic_db.D11_log[ir, inu, ier])
    generic_d13 = float(generic_db.D13[ir, inu, ier])
    generic_d33 = float(generic_db.D33[ir, inu, ier])
    generic_er_axis = float(generic_db.Er_list[ir, ier])

    runtime_db = runtime.database
    kernel = monoenergetic_interpolation_kernel(runtime_db)
    query = kernel(
        jnp.asarray(r_value, dtype=jnp.float64),
        jnp.asarray(nu_value, dtype=jnp.float64),
        jnp.asarray(er_value, dtype=jnp.float64),
        runtime_db,
    )
    query = np.asarray(query, dtype=float)

    print(f"[db-audit] database_file={db_abs}")
    print(f"[db-audit] rho_index={ir} nu_index={inu} er_index={ier}")
    print(f"[db-audit] rho={float(rho[ir]):.6e} r={r_value:.6e}")
    print(f"[db-audit] nu_v={nu_value:.6e}")
    print(f"[db-audit] Er(node)={er_value:.6e}")
    print(f"[db-audit] Es(node)={es_value:.6e}")
    print(f"[db-audit] drds={drds_value:.6e}")
    print()
    print("raw file coefficients")
    print(f"  D11_raw = {raw_d11:.12e}")
    print(f"  D13_raw = {raw_d13:.12e}")
    print(f"  D33_raw = {raw_d33:.12e}")
    print()
    print("expected NEOPAX bridge from raw file")
    print(f"  D11_log_expected = {bridged_d11_log:.12e}")
    print(f"  D13_expected     = {bridged_d13:.12e}")
    print(f"  D33_expected     = {bridged_d33:.12e}")
    print(f"  Er_axis_expected = {bridged_er_axis:.12e}")
    print()
    print("generic Monoenergetic.read_ntx loaded arrays")
    print(f"  D11_log_loaded = {generic_d11_log:.12e}")
    print(f"  D13_loaded     = {generic_d13:.12e}")
    print(f"  D33_loaded     = {generic_d33:.12e}")
    print(f"  Er_axis_loaded = {generic_er_axis:.12e}")
    print()
    print("runtime database kernel queried exactly at stored node")
    print(f"  query[0]=D11_log_interp = {query[0]:.12e}")
    print(f"  query[1]=D13_interp     = {query[1]:.12e}")
    print(f"  query[2]=D33_interp     = {query[2]:.12e}")
    print()
    print("deltas vs expected bridge")
    print(f"  D11_log: {abs(query[0] - bridged_d11_log):.12e}")
    print(f"  D13:     {abs(query[1] - bridged_d13):.12e}")
    print(f"  D33:     {abs(query[2] - bridged_d33):.12e}")


if __name__ == "__main__":
    main()
