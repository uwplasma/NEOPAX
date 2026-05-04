"""Compare one raw HDF5 scan node against NTX scan and prepared solves at that same node.

This isolates whether the remaining mismatch is:
- already present between the stored scan node and current NTX solves, or
- introduced later by NEOPAX exact-runtime bridge/assembly.
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
from NEOPAX._orchestrator import build_runtime_context
from NEOPAX._transport_flux_models import NTXExactLijRuntimeTransportModel, _import_ntx

DEFAULT_DATABASE_CONFIG = Path("examples/benchmarks/Calculate_Fluxes_noHe_ntx_database_benchmark.toml")
DEFAULT_EXACT_CONFIG = Path("examples/benchmarks/Calculate_Fluxes_noHe_ntx_exact_lij_runtime_benchmark.toml")


def _prepare_config(
    config_path: Path,
    *,
    device: str,
    er_init_mode: str,
    force_exact: bool = False,
    resolution: tuple[int, int, int] | None = None,
):
    config = NEOPAX.prepare_config(config_path, device=device)
    config = copy.deepcopy(config)
    config.setdefault("profiles", {})["er_initialization_mode"] = str(er_init_mode)
    if force_exact:
        neo = config.setdefault("neoclassical", {})
        neo["flux_model"] = "ntx_exact_lij_runtime"
        neo["entropy_model"] = "ntx_exact_lij_runtime"
    if resolution is not None:
        n_theta, n_zeta, n_xi = resolution
        neo = config.setdefault("neoclassical", {})
        neo["ntx_exact_n_theta"] = int(n_theta)
        neo["ntx_exact_n_zeta"] = int(n_zeta)
        neo["ntx_exact_n_xi"] = int(n_xi)
    return config


def _parse_resolution(spec: str) -> tuple[int, int, int]:
    parts = [piece.strip() for piece in str(spec).split(",")]
    if len(parts) != 3:
        raise ValueError(f"Resolution '{spec}' must be in 'n_theta,n_zeta,n_xi' format.")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _extract_neoclassical_model(model):
    return getattr(model, "neoclassical_model", model)


def _find_database_path(config: dict) -> Path:
    return Path(config["neoclassical"]["neoclassical_file"])


def _find_vmec_path(config: dict) -> Path:
    return Path(config["geometry"]["vmec_file"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--database-config", default=str(DEFAULT_DATABASE_CONFIG))
    parser.add_argument("--exact-config", default=str(DEFAULT_EXACT_CONFIG))
    parser.add_argument(
        "--er-init-mode",
        default="analytical",
        choices=["keep", "analytical", "ambipolar_min_entropy"],
    )
    parser.add_argument("--rho-index", type=int, default=1)
    parser.add_argument("--nu-index", type=int, default=5)
    parser.add_argument("--er-index", type=int, default=5)
    parser.add_argument("--resolution", default="25,25,63")
    args = parser.parse_args()

    resolution = _parse_resolution(args.resolution)
    db_cfg = _prepare_config(Path(args.database_config), device=args.device, er_init_mode=args.er_init_mode)
    db_runtime, _ = build_runtime_context(db_cfg)
    db_path = _find_database_path(db_cfg)
    db_abs = (ROOT / db_path).resolve() if not db_path.is_absolute() else db_path.resolve()

    with h5py.File(db_abs, "r") as handle:
        rho = np.asarray(handle["rho"][()], dtype=float)
        nu_v = np.asarray(handle["nu_v"][()], dtype=float)
        Es = np.asarray(handle["Es"][()], dtype=float)
        D11 = np.asarray(handle["D11"][()], dtype=float)
        D13 = np.asarray(handle["D13"][()], dtype=float)
        D31 = np.asarray(handle["D31"][()], dtype=float) if "D31" in handle else None
        D33 = np.asarray(handle["D33"][()], dtype=float)

    ir = int(args.rho_index)
    inu = int(args.nu_index)
    ier = int(args.er_index)
    rho_value = float(rho[ir])
    nu_hat_value = float(nu_v[inu])
    epsi_hat_value = float(Es[ir, ier])

    ex_cfg = _prepare_config(
        Path(args.exact_config),
        device=args.device,
        er_init_mode=args.er_init_mode,
        force_exact=True,
        resolution=resolution,
    )
    ex_runtime, _ = build_runtime_context(ex_cfg)
    ex_model = _extract_neoclassical_model(ex_runtime.models.flux)
    if not isinstance(ex_model, NTXExactLijRuntimeTransportModel):
        raise TypeError("Expected ntx_exact_lij_runtime neoclassical model.")
    ex_model = ex_model.with_static_support()
    ntx = _import_ntx()

    neo = ex_cfg["neoclassical"]
    grid_spec = ntx.GridSpec(
        n_theta=int(neo["ntx_exact_n_theta"]),
        n_zeta=int(neo["ntx_exact_n_zeta"]),
        n_xi=int(neo["ntx_exact_n_xi"]),
    )
    vmec_path = _find_vmec_path(ex_cfg)
    vmec_abs = (ROOT / vmec_path).resolve() if not vmec_path.is_absolute() else vmec_path.resolve()
    surface = ntx.surface_from_vmec_jax_vmec_wout_file(str(vmec_abs), s=float(rho_value**2))
    prepared = ntx.prepare_monoenergetic_system(surface, grid_spec)

    prepared_raw = ntx.solve_prepared_coefficient_vector(
        prepared,
        ntx.MonoenergeticCase(
            nu_hat=jnp.asarray(nu_hat_value, dtype=jnp.float64),
            epsi_hat=jnp.asarray(epsi_hat_value, dtype=jnp.float64),
        ),
    )
    prepared_raw = np.asarray(jax.device_get(prepared_raw), dtype=float)

    scan_result = ntx.solve_monoenergetic_scan(
        surface,
        grid_spec,
        jnp.asarray([nu_hat_value], dtype=jnp.float64),
        epsi_hat=jnp.asarray([epsi_hat_value], dtype=jnp.float64),
    )
    scan_raw = np.asarray(
        [
            float(jnp.asarray(scan_result["D11"])[0, 0]),
            float(jnp.asarray(scan_result["D31"])[0, 0]),
            float(jnp.asarray(scan_result["D13"])[0, 0]),
            float(jnp.asarray(scan_result["D33"])[0, 0]),
            float(jnp.asarray(scan_result["D33_spitzer"])[0, 0]),
        ],
        dtype=float,
    )

    print(f"[file-vs-exact] database_file={db_abs}")
    print(f"[file-vs-exact] vmec_file={vmec_abs}")
    print(f"[file-vs-exact] requested_node=(rho_idx={ir}, nu_idx={inu}, er_idx={ier})")
    print(f"[file-vs-exact] rho_file={rho_value:.12e}")
    print(f"[file-vs-exact] rho_exact={rho_value:.12e}")
    print(f"[file-vs-exact] s_exact={float(rho_value**2):.12e}")
    print(f"[file-vs-exact] nu_hat={nu_hat_value:.12e}")
    print(f"[file-vs-exact] epsi_hat_from_file_Es={epsi_hat_value:.12e}")
    print()
    print("raw file node")
    print(f"  D11_file = {D11[ir, inu, ier]:.12e}")
    if D31 is not None:
        print(f"  D31_file = {D31[ir, inu, ier]:.12e}")
    print(f"  D13_file = {D13[ir, inu, ier]:.12e}")
    print(f"  D33_file = {D33[ir, inu, ier]:.12e}")
    print()
    print("NTX scan raw")
    print(f"  D11_scan = {scan_raw[0]:.12e}")
    print(f"  D31_scan = {scan_raw[1]:.12e}")
    print(f"  D13_scan = {scan_raw[2]:.12e}")
    print(f"  D33_scan = {scan_raw[3]:.12e}")
    print(f"  D33_spitzer_scan = {scan_raw[4]:.12e}")
    print()
    print("NTX prepared raw")
    print(f"  D11_prepared = {prepared_raw[0]:.12e}")
    print(f"  D31_prepared = {prepared_raw[1]:.12e}")
    print(f"  D13_prepared = {prepared_raw[2]:.12e}")
    print(f"  D33_prepared = {prepared_raw[3]:.12e}")
    print(f"  D33_spitzer_prepared = {prepared_raw[4]:.12e}")
    print()
    print("absolute deltas: file vs scan")
    print(f"  D11 = {abs(scan_raw[0] - D11[ir, inu, ier]):.12e}")
    if D31 is not None:
        print(f"  D31 = {abs(scan_raw[1] - D31[ir, inu, ier]):.12e}")
    print(f"  D13 = {abs(scan_raw[2] - D13[ir, inu, ier]):.12e}")
    print(f"  D33 = {abs(scan_raw[3] - D33[ir, inu, ier]):.12e}")
    print()
    print("absolute deltas: file vs prepared")
    print(f"  D11 = {abs(prepared_raw[0] - D11[ir, inu, ier]):.12e}")
    if D31 is not None:
        print(f"  D31 = {abs(prepared_raw[1] - D31[ir, inu, ier]):.12e}")
    print(f"  D13 = {abs(prepared_raw[2] - D13[ir, inu, ier]):.12e}")
    print(f"  D33 = {abs(prepared_raw[3] - D33[ir, inu, ier]):.12e}")
    print()
    print("absolute deltas: scan vs prepared")
    print(f"  D11 = {abs(prepared_raw[0] - scan_raw[0]):.12e}")
    print(f"  D31 = {abs(prepared_raw[1] - scan_raw[1]):.12e}")
    print(f"  D13 = {abs(prepared_raw[2] - scan_raw[2]):.12e}")
    print(f"  D33 = {abs(prepared_raw[3] - scan_raw[3]):.12e}")


if __name__ == "__main__":
    main()
