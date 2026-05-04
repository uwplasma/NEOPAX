"""Compare database-queried and exact NTX Dij at multiple stored nodes.

This script evaluates the NEOPAX database interpolation kernel exactly at stored
`rho`, `nu_v`, and `Er` nodes from the NTX-built HDF5 file, then compares the
resulting physical monoenergetic coefficients against exact prepared NTX solves
at those same nodes.

It is a direct confirmation test for the hypothesis that:
- node-level database queries should agree with exact NTX,
- and any remaining mismatch is an off-node effect.
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
from NEOPAX._monoenergetic_interpolators import monoenergetic_interpolation_kernel
from NEOPAX._orchestrator import build_runtime_context
from NEOPAX._transport_flux_models import _import_ntx

DEFAULT_DATABASE_CONFIG = Path("examples/benchmarks/Calculate_Fluxes_noHe_ntx_database_benchmark.toml")
DEFAULT_EXACT_CONFIG = Path("examples/benchmarks/Calculate_Fluxes_noHe_ntx_exact_lij_runtime_benchmark.toml")


def _prepare_config(
    config_path: Path,
    *,
    device: str,
    er_init_mode: str,
    flux_model: str,
    interpolation_mode: str | None = None,
    resolution: tuple[int, int, int] | None = None,
):
    config = NEOPAX.prepare_config(config_path, device=device)
    config = copy.deepcopy(config)
    config.setdefault("profiles", {})["er_initialization_mode"] = str(er_init_mode)
    neoclassical = config.setdefault("neoclassical", {})
    neoclassical["flux_model"] = str(flux_model)
    neoclassical["entropy_model"] = str(flux_model)
    if interpolation_mode is not None:
        neoclassical["interpolation_mode"] = str(interpolation_mode)
    if resolution is not None:
        n_theta, n_zeta, n_xi = resolution
        neoclassical["ntx_exact_n_theta"] = int(n_theta)
        neoclassical["ntx_exact_n_zeta"] = int(n_zeta)
        neoclassical["ntx_exact_n_xi"] = int(n_xi)
    return config


def _parse_resolution(spec: str) -> tuple[int, int, int]:
    parts = [piece.strip() for piece in str(spec).split(",")]
    if len(parts) != 3:
        raise ValueError(f"Resolution '{spec}' must be in 'n_theta,n_zeta,n_xi' format.")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _parse_index_spec(spec: str) -> list[int]:
    items: list[int] = []
    for chunk in str(spec).split(","):
        text = chunk.strip()
        if not text:
            continue
        if ":" in text:
            parts = [piece.strip() for piece in text.split(":")]
            if len(parts) not in {2, 3}:
                raise ValueError(f"Invalid range spec '{text}'")
            start = int(parts[0])
            stop = int(parts[1])
            step = int(parts[2]) if len(parts) == 3 else 1
            items.extend(list(range(start, stop, step)))
        else:
            items.append(int(text))
    if not items:
        raise ValueError(f"Empty index spec '{spec}'")
    return items


def _database_channels_to_physical(db_coeffs: jax.Array, nu_hat_value: float) -> np.ndarray:
    return np.asarray(
        (
            -10.0 ** float(db_coeffs[0]),
            -float(db_coeffs[1]),
            -float(db_coeffs[2]) / max(float(nu_hat_value), 1.0e-30),
        ),
        dtype=float,
    )


def _exact_raw_to_physical(exact_coeffs: np.ndarray, drds_value: float) -> np.ndarray:
    return np.asarray(
        (
            -float(exact_coeffs[0]) * drds_value**2,
            -float(exact_coeffs[2]) * drds_value,
            -float(exact_coeffs[3]),
        ),
        dtype=float,
    )


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
    parser.add_argument("--rho-indices", default="0,1,2")
    parser.add_argument("--nu-indices", default="0,2,5")
    parser.add_argument("--er-indices", default="1,3,6,9")
    parser.add_argument("--resolution", default="25,25,63")
    parser.add_argument(
        "--interpolation-mode",
        default=None,
        help="Optional database interpolation mode override. Defaults to the config value.",
    )
    args = parser.parse_args()

    resolution = _parse_resolution(args.resolution)
    rho_indices = _parse_index_spec(args.rho_indices)
    nu_indices = _parse_index_spec(args.nu_indices)
    er_indices = _parse_index_spec(args.er_indices)

    db_cfg = _prepare_config(
        Path(args.database_config),
        device=args.device,
        er_init_mode=args.er_init_mode,
        flux_model="ntx_database",
        interpolation_mode=args.interpolation_mode,
    )
    runtime, _ = build_runtime_context(db_cfg)
    db_path = Path(db_cfg["neoclassical"]["neoclassical_file"])
    db_abs = (ROOT / db_path).resolve() if not db_path.is_absolute() else db_path.resolve()
    kernel = monoenergetic_interpolation_kernel(runtime.database)

    with h5py.File(db_abs, "r") as handle:
        rho = np.asarray(handle["rho"][()], dtype=float)
        nu_v = np.asarray(handle["nu_v"][()], dtype=float)
        Er = np.asarray(handle["Er"][()], dtype=float)
        Es = np.asarray(handle["Es"][()], dtype=float)

    ex_cfg = _prepare_config(
        Path(args.exact_config),
        device=args.device,
        er_init_mode=args.er_init_mode,
        flux_model="ntx_exact_lij_runtime",
        resolution=resolution,
    )
    ntx = _import_ntx()
    neo = ex_cfg["neoclassical"]
    grid_spec = ntx.GridSpec(
        n_theta=int(neo["ntx_exact_n_theta"]),
        n_zeta=int(neo["ntx_exact_n_zeta"]),
        n_xi=int(neo["ntx_exact_n_xi"]),
    )
    vmec_path = Path(ex_cfg["geometry"]["vmec_file"])
    vmec_abs = (ROOT / vmec_path).resolve() if not vmec_path.is_absolute() else vmec_path.resolve()

    prepared_by_rho: dict[int, object] = {}
    max_abs = np.zeros(3, dtype=float)
    max_rel = np.zeros(3, dtype=float)
    worst_meta: dict[str, tuple[int, int, int] | None] = {"D11": None, "D13": None, "D33": None}

    print(f"[node-compare] database_file={db_abs}")
    print(f"[node-compare] interpolation_mode={db_cfg['neoclassical'].get('interpolation_mode')}")
    print(f"[node-compare] resolution={resolution}")
    print()
    print(
        f"{'rho':>3} {'nu':>3} {'er':>3} "
        f"{'D11_abs':>12} {'D11_rel':>12} "
        f"{'D13_abs':>12} {'D13_rel':>12} "
        f"{'D33_abs':>12} {'D33_rel':>12}"
    )
    print("-" * 90)

    for ir in rho_indices:
        if ir not in prepared_by_rho:
            surface = ntx.surface_from_vmec_jax_vmec_wout_file(str(vmec_abs), s=float(rho[ir] ** 2))
            prepared_by_rho[ir] = ntx.prepare_monoenergetic_system(surface, grid_spec)
        prepared = prepared_by_rho[ir]
        drds_value = float(Es[ir, 1] / Er[ir, 1]) if Er[ir, 1] != 0.0 else float(Es[ir, 2] / Er[ir, 2])

        for inu in nu_indices:
            nu_value = float(nu_v[inu])
            for ier in er_indices:
                er_value = float(Er[ir, ier])
                es_value = float(Es[ir, ier])

                r_value = float(runtime.geometry.a_b * rho[ir])
                db_coeff = kernel(r_value, nu_value, er_value, runtime.database)
                db_phys = _database_channels_to_physical(db_coeff, nu_value)

                exact_raw = np.asarray(
                    jax.device_get(
                        ntx.solve_prepared_coefficient_vector(
                            prepared,
                            ntx.MonoenergeticCase(
                                nu_hat=jnp.asarray(nu_value, dtype=jnp.float64),
                                epsi_hat=jnp.asarray(es_value, dtype=jnp.float64),
                            ),
                        )
                    ),
                    dtype=float,
                )
                exact_phys = _exact_raw_to_physical(exact_raw, drds_value)

                abs_delta = np.abs(exact_phys - db_phys)
                rel_delta = abs_delta / np.maximum(np.abs(exact_phys), 1.0e-30)

                print(
                    f"{ir:>3d} {inu:>3d} {ier:>3d} "
                    f"{abs_delta[0]:12.4e} {rel_delta[0]:12.4e} "
                    f"{abs_delta[1]:12.4e} {rel_delta[1]:12.4e} "
                    f"{abs_delta[2]:12.4e} {rel_delta[2]:12.4e}"
                )

                labels = ("D11", "D13", "D33")
                for idx, label in enumerate(labels):
                    if abs_delta[idx] > max_abs[idx]:
                        max_abs[idx] = abs_delta[idx]
                        max_rel[idx] = rel_delta[idx]
                        worst_meta[label] = (ir, inu, ier)

    print()
    print("[node-compare] maxima")
    for idx, label in enumerate(("D11", "D13", "D33")):
        print(
            f"  {label}: abs_max={max_abs[idx]:.6e} rel_max={max_rel[idx]:.6e} "
            f"at node={worst_meta[label]}"
        )


if __name__ == "__main__":
    main()
