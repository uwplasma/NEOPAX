"""Plot ln(D11) vs ln(nu/v) for radius/field midpoints.

This is the midpoint companion to `benchmark_plot_d11_database_vs_exact_field_lines.py`.

For each selected lower radius index `i`, this script creates one figure for the
midpoint radius:

- `rho_mid = 0.5 * (rho[i] + rho[i+1])`

Inside each figure:

- x-axis: ln(nu/v) at the database nodes
- y-axis: ln(D11)
- one color per selected field midpoint
- database query at `(rho_mid, Er_mid)`
- exact prepared NTX solve at `(rho_mid, Es_mid)`

where:

- `Er_mid = 0.5 * (Er[i, j] + Er[i, j+1])`
- `drds_mid = a_b / (2 * rho_mid)`
- `Es_mid = Er_mid * drds_mid`
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
from NEOPAX._database import D11_POSITIVE_FLOOR
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
    d11 = float(np.asarray(exact_coeffs[0], dtype=float) * drds_value**2)
    return max(d11, float(D11_POSITIVE_FLOOR))


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
    parser.add_argument("--output-dir", default="outputs/benchmark_d11_database_vs_exact_field_lines_midpoints")
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

    radius_indices = [idx for idx in _parse_index_spec(args.radius_indices, rho.shape[0] - 1) if idx < rho.shape[0] - 1]
    field_indices = [idx for idx in _parse_index_spec(args.field_indices, Er.shape[1] - 1) if idx < Er.shape[1] - 1]

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

    prepared_by_rho_mid: dict[int, object] = {}
    x_ln_nu = np.log(np.asarray(nu_v, dtype=float))
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[d11-field-lines-mid] database_file={db_abs}")
    print(f"[d11-field-lines-mid] vmec_file={vmec_abs}")
    print(f"[d11-field-lines-mid] interpolation_mode={db_cfg['neoclassical'].get('interpolation_mode')}")
    print(f"[d11-field-lines-mid] resolution={resolution}")
    print(f"[d11-field-lines-mid] radii(lower idx)={radius_indices}")
    print(f"[d11-field-lines-mid] fields(lower idx)={field_indices}")

    colors = plt.cm.nipy_spectral(np.linspace(0.03, 0.97, max(len(field_indices), 2)))
    a_b = float(runtime.geometry.a_b)

    for ir in radius_indices:
        rho_mid = 0.5 * (rho[ir] + rho[ir + 1])
        if ir not in prepared_by_rho_mid:
            surface = ntx.surface_from_vmec_jax_vmec_wout_file(str(vmec_abs), s=float(rho_mid**2))
            prepared_by_rho_mid[ir] = ntx.prepare_monoenergetic_system(surface, grid_spec)
        prepared = prepared_by_rho_mid[ir]

        drds_mid = a_b / (2.0 * rho_mid)
        r_mid = a_b * rho_mid

        fig, ax = plt.subplots(figsize=(12, 7))

        for color, ier in zip(colors, field_indices):
            db_vals: list[float] = []
            exact_vals: list[float] = []
            er_mid = 0.5 * (Er[ir, ier] + Er[ir, ier + 1])
            es_mid = er_mid * drds_mid

            for nu_value in nu_v:
                db_coeff = kernel(r_mid, float(nu_value), float(er_mid), runtime.database)
                db_vals.append(_database_d11_positive(db_coeff))

                exact_raw = np.asarray(
                    jax.device_get(
                        ntx.solve_prepared_coefficient_vector(
                            prepared,
                            ntx.MonoenergeticCase(
                                nu_hat=jnp.asarray(float(nu_value), dtype=jnp.float64),
                                epsi_hat=jnp.asarray(float(es_mid), dtype=jnp.float64),
                            ),
                        )
                    ),
                    dtype=float,
                )
                exact_vals.append(_exact_d11_positive(exact_raw, drds_mid))

            db_ln = np.log(np.maximum(np.asarray(db_vals, dtype=float), 1.0e-300))
            exact_ln = np.log(np.maximum(np.asarray(exact_vals, dtype=float), 1.0e-300))
            label_base = f"field mid {ier}-{ier+1} | Er/v={er_mid:.3e}"
            ax.plot(
                x_ln_nu,
                exact_ln,
                color=color,
                linewidth=1.8,
                marker="o",
                markersize=4.0,
                label=f"{label_base} exact",
            )
            ax.plot(
                x_ln_nu,
                db_ln,
                color=color,
                linewidth=1.8,
                linestyle="--",
                marker="s",
                markersize=4.0,
                label=f"{label_base} db",
            )

        ax.set_title(f"rho midpoint {ir}-{ir+1}, rho={rho_mid:.6f}")
        ax.set_xlabel("ln(nu/v)")
        ax.set_ylabel("ln(D11)")
        ax.grid(True, alpha=0.3)
        ax.legend(
            fontsize=8,
            ncol=2,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
        )

        fig.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
        out_path = output_dir / f"d11_field_lines_rhomid_{ir:03d}_{ir+1:03d}.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        print(f"  [rho mid {ir}-{ir+1}] plot={out_path}")


if __name__ == "__main__":
    main()
