"""Plot log(D11) vs log(nu/v) for every field line at each database radius.

For each selected radius node in the database file, this script creates one
figure. Inside each figure:

- x-axis: ln(nu/v) at the database nodes
- y-axis: ln(D11)
- one color per field node
- exact NTX node solves and database-node values are overlaid

This is a node-level diagnostic: it compares the preprocessed/database path to
exact prepared NTX solves at the stored `(rho, nu_v, Er, Es)` nodes of the HDF5
database file.
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


def _parse_index_spec(spec: str, size: int) -> list[int]:
    text = str(spec).strip().lower()
    if text in {"all", "*"}:
        return list(range(size))

    items: list[int] = []
    for chunk in str(spec).split(","):
        part = chunk.strip()
        if not part:
            continue
        if ":" in part:
            bits = [piece.strip() for piece in part.split(":")]
            if len(bits) not in {2, 3}:
                raise ValueError(f"Invalid range spec '{part}'")
            start = int(bits[0])
            stop = int(bits[1])
            step = int(bits[2]) if len(bits) == 3 else 1
            items.extend(list(range(start, stop, step)))
        else:
            items.append(int(part))
    if not items:
        raise ValueError(f"Empty index spec '{spec}'")
    return items


def _database_d11_positive(db_coeffs: jax.Array) -> float:
    return float(10.0 ** jnp.asarray(db_coeffs[0], dtype=jnp.float64))


def _exact_d11_positive(exact_coeffs: np.ndarray, drds_value: float) -> float:
    return float(np.asarray(exact_coeffs[0], dtype=float) * drds_value**2)


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
    parser.add_argument("--resolution", default="25,25,63")
    parser.add_argument("--radius-indices", default="all")
    parser.add_argument("--field-indices", default="all")
    parser.add_argument("--output-dir", default="outputs/benchmark_d11_database_vs_exact_field_lines")
    parser.add_argument(
        "--interpolation-mode",
        default=None,
        help="Optional database interpolation mode override. Defaults to the config value.",
    )
    args = parser.parse_args()

    resolution = _parse_resolution(args.resolution)
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

    radius_indices = _parse_index_spec(args.radius_indices, rho.shape[0])
    field_indices = _parse_index_spec(args.field_indices, Er.shape[1])

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

    prepared_by_radius: dict[int, object] = {}
    x_ln_nu = np.log(np.asarray(nu_v, dtype=float))
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[d11-field-lines] database_file={db_abs}")
    print(f"[d11-field-lines] vmec_file={vmec_abs}")
    print(f"[d11-field-lines] interpolation_mode={db_cfg['neoclassical'].get('interpolation_mode')}")
    print(f"[d11-field-lines] resolution={resolution}")
    print(f"[d11-field-lines] radii={radius_indices}")
    print(f"[d11-field-lines] fields={field_indices}")

    colors = plt.cm.viridis(np.linspace(0.05, 0.95, max(len(field_indices), 2)))

    for ir in radius_indices:
        if ir not in prepared_by_radius:
            surface = ntx.surface_from_vmec_jax_vmec_wout_file(str(vmec_abs), s=float(rho[ir] ** 2))
            prepared_by_radius[ir] = ntx.prepare_monoenergetic_system(surface, grid_spec)
        prepared = prepared_by_radius[ir]

        drds_value = float(Es[ir, 1] / Er[ir, 1]) if abs(Er[ir, 1]) > 0.0 else float(Es[ir, 2] / Er[ir, 2])
        r_value = float(runtime.geometry.a_b * rho[ir])

        fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)

        for color, ier in zip(colors, field_indices):
            db_vals: list[float] = []
            exact_vals: list[float] = []
            er_value = float(Er[ir, ier])
            es_value = float(Es[ir, ier])

            for nu_value in nu_v:
                db_coeff = kernel(r_value, float(nu_value), er_value, runtime.database)
                db_vals.append(_database_d11_positive(db_coeff))

                exact_raw = np.asarray(
                    jax.device_get(
                        ntx.solve_prepared_coefficient_vector(
                            prepared,
                            ntx.MonoenergeticCase(
                                nu_hat=jnp.asarray(float(nu_value), dtype=jnp.float64),
                                epsi_hat=jnp.asarray(es_value, dtype=jnp.float64),
                            ),
                        )
                    ),
                    dtype=float,
                )
                exact_vals.append(_exact_d11_positive(exact_raw, drds_value))

            db_ln = np.log(np.maximum(np.asarray(db_vals, dtype=float), 1.0e-300))
            exact_ln = np.log(np.maximum(np.asarray(exact_vals, dtype=float), 1.0e-300))
            label_base = f"field {ier} | Er/v={er_value:.3e}"
            ax.plot(x_ln_nu, exact_ln, color=color, linewidth=2.0, label=f"{label_base} exact")
            ax.plot(
                x_ln_nu,
                db_ln,
                color=color,
                linewidth=1.8,
                linestyle="--",
                marker="o",
                markersize=3.5,
                label=f"{label_base} db",
            )

        ax.set_title(f"rho index={ir}, rho={rho[ir]:.6f}")
        ax.set_xlabel("ln(nu/v)")
        ax.set_ylabel("ln(D11)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

        out_path = output_dir / f"d11_field_lines_radius_{ir:03d}.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        print(f"  [radius {ir}] plot={out_path}")


if __name__ == "__main__":
    main()
