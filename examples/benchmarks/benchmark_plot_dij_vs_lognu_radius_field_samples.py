"""Plot Dij vs ln(nu_v) for node and midpoint radius samples.

For one chosen lower rho index j, this script generates curves at:
- node radius rho[j]
- midpoint radius 0.5 * (rho[j] + rho[j+1])

For each selected field-node index, it also includes:
- the node Er/v value itself
- the midpoint Er/v values between consecutive selected field nodes

For nu_v it uses a combined sampling:
- every stored nu_v node
- every geometric midpoint between adjacent nu_v nodes

Each figure corresponds to one radius sample and one Er/v sample, and plots:
- ln(D11)
- D13
- D33

against:
- ln(nu_v)

Datasets shown:
- exact
- generic
- generic_loger_no_r
- preprocessed_3d
- preprocessed_3d_radial_ntss1d
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
DEFAULT_MODES = (
    "generic",
    "generic_loger_no_r",
    "preprocessed_3d",
    "preprocessed_3d_radial_ntss1d",
)


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
        items.append(int(text))
    if not items:
        raise ValueError(f"Empty index spec '{spec}'")
    return items


def _db_to_plot_physical(db_coeffs: jax.Array, nu_hat_value: jax.Array) -> jax.Array:
    return jnp.array(
        (
            10.0 ** jnp.asarray(db_coeffs[0], dtype=jnp.float64),
            jnp.asarray(db_coeffs[1], dtype=jnp.float64),
            jnp.asarray(db_coeffs[2], dtype=jnp.float64) / jnp.maximum(jnp.asarray(nu_hat_value, dtype=jnp.float64), 1.0e-30),
        ),
        dtype=jnp.float64,
    )


def _exact_to_plot_physical(exact_coeffs: jax.Array, drds_value: jax.Array) -> jax.Array:
    return jnp.array(
        (
            jnp.asarray(exact_coeffs[0], dtype=jnp.float64) * drds_value**2,
            jnp.asarray(exact_coeffs[2], dtype=jnp.float64) * drds_value,
            jnp.asarray(exact_coeffs[3], dtype=jnp.float64),
        ),
        dtype=jnp.float64,
    )


def _combined_nu_samples(nu_nodes: np.ndarray) -> np.ndarray:
    values: list[float] = []
    for i in range(len(nu_nodes) - 1):
        values.append(float(nu_nodes[i]))
        values.append(float(np.sqrt(nu_nodes[i] * nu_nodes[i + 1])))
    values.append(float(nu_nodes[-1]))
    return np.asarray(values, dtype=float)


def _combined_field_samples(er_row: np.ndarray, selected_indices: list[int]) -> list[tuple[str, float]]:
    picked = sorted(set(int(i) for i in selected_indices))
    samples: list[tuple[str, float]] = []
    for idx in picked:
        samples.append((f"node_{idx}", float(er_row[idx])))
    for left, right in zip(picked[:-1], picked[1:]):
        samples.append((f"mid_{left}_{right}", 0.5 * float(er_row[left] + er_row[right])))
    return samples


def _plot_one(
    x_ln_nu: np.ndarray,
    curves: dict[str, np.ndarray],
    rho_label: str,
    er_label: str,
    er_value: float,
    output_path: Path,
):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    colors = {
        "exact": "black",
        "generic": "C0",
        "generic_loger_no_r": "C3",
        "preprocessed_3d": "C1",
        "preprocessed_3d_radial_ntss1d": "C2",
    }
    for ax, idx, ylabel in zip(axes, (0, 1, 2), ("ln(D11)", "D13", "D33")):
        for label, values in curves.items():
            if idx == 0:
                y = np.log(np.maximum(values[:, idx], 1.0e-300))
            else:
                y = values[:, idx]
            ax.plot(x_ln_nu, y, color=colors[label], linewidth=2.0, label=label)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("ln(nu_v)")
    fig.suptitle(f"{rho_label}, {er_label}, Er/v={er_value:.6e}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


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
    parser.add_argument("--field-indices", default="1,3,6,9")
    parser.add_argument("--resolution", default="25,25,63")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    args = parser.parse_args()

    resolution = _parse_resolution(args.resolution)
    field_indices = _parse_index_spec(args.field_indices)
    modes = [item.strip() for item in str(args.modes).split(",") if item.strip()]

    db_cfg = _prepare_config(
        Path(args.database_config),
        device=args.device,
        er_init_mode=args.er_init_mode,
        flux_model="ntx_database",
    )
    runtime, _ = build_runtime_context(db_cfg)
    db_path = Path(db_cfg["neoclassical"]["neoclassical_file"])
    db_abs = (ROOT / db_path).resolve() if not db_path.is_absolute() else db_path.resolve()
    with h5py.File(db_abs, "r") as handle:
        rho = np.asarray(handle["rho"][()], dtype=float)
        nu_v = np.asarray(handle["nu_v"][()], dtype=float)
        er_nodes = np.asarray(handle["Er"][()], dtype=float)
        es_nodes = np.asarray(handle["Es"][()], dtype=float)

    rho_index = int(args.rho_index)
    if rho_index < 0 or rho_index + 1 >= len(rho):
        raise ValueError(f"rho-index {rho_index} must satisfy 0 <= idx < {len(rho)-1}")

    rho_node = float(rho[rho_index])
    rho_mid = 0.5 * float(rho[rho_index] + rho[rho_index + 1])
    radius_samples = [
        ("rho_node", rho_node),
        ("rho_mid", rho_mid),
    ]

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

    mode_runtimes: dict[str, tuple[object, object]] = {}
    for mode in modes:
        cfg = _prepare_config(
            Path(args.database_config),
            device=args.device,
            er_init_mode=args.er_init_mode,
            flux_model="ntx_database",
            interpolation_mode=mode,
        )
        mode_runtime, _ = build_runtime_context(cfg)
        mode_runtimes[mode] = (mode_runtime, monoenergetic_interpolation_kernel(mode_runtime.database))

    nu_samples = _combined_nu_samples(np.asarray(nu_v, dtype=float))
    x_ln_nu = np.log(nu_samples)
    field_samples = _combined_field_samples(er_nodes[rho_index], field_indices)

    output_dir = Path("outputs/benchmark_dij_vs_lognu_radius_field_samples")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[dij-lnu] database_file={db_abs}")
    print(f"[dij-lnu] rho_index={rho_index} rho_node={rho_node:.6e} rho_mid={rho_mid:.6e}")
    print(f"[dij-lnu] field_samples={[name for name, _ in field_samples]}")
    print(f"[dij-lnu] resolution={resolution}")
    print()

    for rho_label, rho_value in radius_samples:
        r_value = float(runtime.geometry.a_b * rho_value)
        prepared = ntx.prepare_monoenergetic_system(
            ntx.surface_from_vmec_jax_vmec_wout_file(str(vmec_abs), s=float(rho_value**2)),
            grid_spec,
        )
        drds_value = float(prepared.geometry.transport_psi_scale)

        for field_label, er_value in field_samples:
            drds_plot = drds_value
            if rho_label == "rho_node" and field_label.startswith("node_"):
                field_idx = int(field_label.split("_")[1])
                drds_plot = float(es_nodes[rho_index, field_idx] / er_nodes[rho_index, field_idx])
            curves: dict[str, np.ndarray] = {}
            exact_vals = []
            for nu_value in nu_samples:
                exact_vals.append(
                    np.asarray(
                        _exact_to_plot_physical(
                            ntx.solve_prepared_coefficient_vector(
                                prepared,
                                ntx.MonoenergeticCase(
                                    nu_hat=jnp.asarray(nu_value, dtype=jnp.float64),
                                    epsi_hat=jnp.asarray(er_value * drds_plot, dtype=jnp.float64),
                                ),
                            ),
                            jnp.asarray(drds_plot, dtype=jnp.float64),
                        ),
                        dtype=float,
                    )
                )
            curves["exact"] = np.asarray(exact_vals)

            for mode in modes:
                mode_runtime, kernel = mode_runtimes[mode]
                vals = []
                for nu_value in nu_samples:
                    vals.append(
                        np.asarray(
                            _db_to_plot_physical(
                                kernel(r_value, float(nu_value), er_value, mode_runtime.database),
                                jnp.asarray(nu_value, dtype=jnp.float64),
                            ),
                            dtype=float,
                        )
                    )
                curves[mode] = np.asarray(vals)

            output_path = output_dir / (
                f"dij_lnu_rhoidx_{rho_index}_{rho_label}_{field_label}_{resolution[0]}_{resolution[1]}_{resolution[2]}.png"
            )
            _plot_one(x_ln_nu, curves, rho_label, field_label, er_value, output_path)
            print(f"[plot] rho={rho_label} field={field_label} Er/v={er_value:.6e} path={output_path}")


if __name__ == "__main__":
    main()
