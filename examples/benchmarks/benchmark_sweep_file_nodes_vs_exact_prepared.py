"""Sweep raw file nodes against exact prepared NTX at one stored rho/nu slice.

This removes database interpolation entirely. For one stored rho index and one
stored nu index, it sweeps across the file's electric-field node axis and
compares:
- raw NTX coefficients stored in the HDF5 file
- raw exact prepared NTX solve at the same rho, nu_hat, and Es node
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
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


def _extract_neoclassical_model(model):
    return getattr(model, "neoclassical_model", model)


def _parse_resolution(spec: str) -> tuple[int, int, int]:
    parts = [piece.strip() for piece in str(spec).split(",")]
    if len(parts) != 3:
        raise ValueError(f"Resolution '{spec}' must be in 'n_theta,n_zeta,n_xi' format.")
    return int(parts[0]), int(parts[1]), int(parts[2])


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
    parser.add_argument("--resolution", default="25,25,63")
    args = parser.parse_args()

    resolution = _parse_resolution(args.resolution)

    db_cfg = _prepare_config(Path(args.database_config), device=args.device, er_init_mode=args.er_init_mode)
    db_runtime, _ = build_runtime_context(db_cfg)
    db_path = Path(db_cfg["neoclassical"]["neoclassical_file"])
    db_abs = (ROOT / db_path).resolve() if not db_path.is_absolute() else db_path.resolve()

    with h5py.File(db_abs, "r") as handle:
        rho = np.asarray(handle["rho"][()], dtype=float)
        nu_v = np.asarray(handle["nu_v"][()], dtype=float)
        Er = np.asarray(handle["Er"][()], dtype=float)
        Es = np.asarray(handle["Es"][()], dtype=float)
        D11 = np.asarray(handle["D11"][()], dtype=float)
        D13 = np.asarray(handle["D13"][()], dtype=float)
        D31 = np.asarray(handle["D31"][()], dtype=float) if "D31" in handle else None
        D33 = np.asarray(handle["D33"][()], dtype=float)

    ir = int(args.rho_index)
    inu = int(args.nu_index)
    rho_value = float(rho[ir])
    nu_hat_value = float(nu_v[inu])

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
    vmec_path = Path(ex_cfg["geometry"]["vmec_file"])
    vmec_abs = (ROOT / vmec_path).resolve() if not vmec_path.is_absolute() else vmec_path.resolve()
    surface = ntx.surface_from_vmec_jax_vmec_wout_file(str(vmec_abs), s=float(rho_value**2))
    prepared = ntx.prepare_monoenergetic_system(surface, grid_spec)

    epsi_hat_axis = jnp.asarray(Es[ir, :], dtype=jnp.float64)
    exact_raw = jax.vmap(
        lambda epsi_hat_value: ntx.solve_prepared_coefficient_vector(
            prepared,
            ntx.MonoenergeticCase(
                nu_hat=jnp.asarray(nu_hat_value, dtype=jnp.float64),
                epsi_hat=jnp.asarray(epsi_hat_value, dtype=jnp.float64),
            ),
        )
    )(epsi_hat_axis)
    exact_raw = np.asarray(jax.device_get(exact_raw), dtype=float)

    output_dir = Path("outputs/benchmark_file_node_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)
    x = np.asarray(Er[ir, :], dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    file_curves = [D11[ir, inu, :], D31[ir, inu, :], D13[ir, inu, :], D33[ir, inu, :]]
    exact_curves = [exact_raw[:, 0], exact_raw[:, 1], exact_raw[:, 2], exact_raw[:, 3]]
    labels = ("D11", "D31", "D13", "D33")
    for ax, file_vals, exact_vals, label in zip(axes, file_curves, exact_curves, labels):
        ax.plot(x, file_vals, label="raw file", linewidth=2.4)
        ax.plot(x, exact_vals, label="exact prepared", linestyle="--", linewidth=2.0)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("Er node from file")
    fig.suptitle(
        f"rho_idx={ir}, nu_idx={inu}, rho={rho_value:.4f}, nu_hat={nu_hat_value:.4e}, "
        f"resolution={resolution}"
    )
    fig.tight_layout()
    plot_path = output_dir / f"file_node_sweep_rho_{ir}_nu_{inu}_{resolution[0]}_{resolution[1]}_{resolution[2]}.png"
    fig.savefig(plot_path, dpi=170)
    plt.close(fig)

    print(f"[file-node-sweep] database_file={db_abs}")
    print(f"[file-node-sweep] rho_index={ir} nu_index={inu}")
    print(f"[file-node-sweep] rho={rho_value:.12e}")
    print(f"[file-node-sweep] nu_hat={nu_hat_value:.12e}")
    print(f"[file-node-sweep] Er_range=[{float(np.min(Er[ir, :])):.6e}, {float(np.max(Er[ir, :])):.6e}]")
    print(f"[file-node-sweep] Es_range=[{float(np.min(Es[ir, :])):.6e}, {float(np.max(Es[ir, :])):.6e}]")
    print(f"[file-node-sweep] plot={plot_path}")
    print()
    print(f"{'quantity':<8} {'abs_max':>14} {'rel_max':>14}")
    print("-" * 40)
    for label, file_vals, exact_vals in zip(labels, file_curves, exact_curves):
        abs_max = float(np.max(np.abs(exact_vals - file_vals)))
        rel_max = float(np.max(np.abs(exact_vals - file_vals) / np.maximum(np.abs(file_vals), 1.0e-30)))
        print(f"{label:<8} {abs_max:14.6e} {rel_max:14.6e}")


if __name__ == "__main__":
    main()
