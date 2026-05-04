"""Isolate which database interpolation axis is driving the off-node mismatch.

This script compares database interpolation against exact NTX using the
physically consistent conventions:
- database queried with Er/v
- exact NTX solved with Es/v

It evaluates three cases at a chosen runtime radius/species/energy:
1. on-grid rho, off-grid nu
2. off-grid rho, on-grid nu
3. off-grid rho, off-grid nu

The electric-field axis is kept on the stored file nodes to avoid mixing in
field interpolation error while we diagnose rho/nu interpolation.
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
from NEOPAX._neoclassical import _collisionality_kind, _nu_over_vnew_local
from NEOPAX._orchestrator import build_runtime_context
from NEOPAX._state import get_v_thermal, safe_density
from NEOPAX._transport_flux_models import NTXExactLijRuntimeTransportModel, _import_ntx

DEFAULT_DATABASE_CONFIG = Path("examples/benchmarks/Calculate_Fluxes_noHe_ntx_database_benchmark.toml")
DEFAULT_EXACT_CONFIG = Path("examples/benchmarks/Calculate_Fluxes_noHe_ntx_exact_lij_runtime_benchmark.toml")


def _prepare_config(
    config_path: Path,
    *,
    device: str,
    er_init_mode: str,
    flux_model: str,
    resolution: tuple[int, int, int] | None = None,
):
    config = NEOPAX.prepare_config(config_path, device=device)
    config = copy.deepcopy(config)
    config.setdefault("profiles", {})["er_initialization_mode"] = str(er_init_mode)
    neoclassical = config.setdefault("neoclassical", {})
    neoclassical["flux_model"] = str(flux_model)
    neoclassical["entropy_model"] = str(flux_model)
    if resolution is not None:
        n_theta, n_zeta, n_xi = resolution
        neoclassical["ntx_exact_n_theta"] = int(n_theta)
        neoclassical["ntx_exact_n_zeta"] = int(n_zeta)
        neoclassical["ntx_exact_n_xi"] = int(n_xi)
    return config


def _extract_neoclassical_model(model):
    return getattr(model, "neoclassical_model", model)


def _parse_resolution(spec: str) -> tuple[int, int, int]:
    parts = [piece.strip() for piece in str(spec).split(",")]
    if len(parts) != 3:
        raise ValueError(f"Resolution '{spec}' must be in 'n_theta,n_zeta,n_xi' format.")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _database_channels_to_physical(db_coeffs: jax.Array, nu_hat_value: jax.Array) -> jax.Array:
    return jnp.array(
        (
            -10.0 ** jnp.asarray(db_coeffs[0], dtype=jnp.float64),
            -jnp.asarray(db_coeffs[1], dtype=jnp.float64),
            -jnp.asarray(db_coeffs[2], dtype=jnp.float64) / jnp.maximum(jnp.asarray(nu_hat_value, dtype=jnp.float64), 1.0e-30),
        ),
        dtype=jnp.float64,
    )


def _exact_raw_to_physical(exact_coeffs: jax.Array, drds_value: jax.Array) -> jax.Array:
    return jnp.array(
        (
            -jnp.asarray(exact_coeffs[0], dtype=jnp.float64) * drds_value**2,
            -jnp.asarray(exact_coeffs[2], dtype=jnp.float64) * drds_value,
            -jnp.asarray(exact_coeffs[3], dtype=jnp.float64),
        ),
        dtype=jnp.float64,
    )


def _nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


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
    parser.add_argument(
        "--state-source-model",
        default="database",
        choices=["database", "exact"],
    )
    parser.add_argument("--radius-index", type=int, default=10)
    parser.add_argument("--species-index", type=int, default=1)
    parser.add_argument("--energy-index", type=int, default=8)
    parser.add_argument("--num-field-nodes", type=int, default=12)
    parser.add_argument("--resolution", default="25,25,63")
    args = parser.parse_args()

    resolution = _parse_resolution(args.resolution)
    db_cfg_path = Path(args.database_config)
    ex_cfg_path = Path(args.exact_config)

    state_cfg_path = db_cfg_path if args.state_source_model == "database" else ex_cfg_path
    state_flux_model = "ntx_database" if args.state_source_model == "database" else "ntx_exact_lij_runtime"
    shared_state_config = _prepare_config(
        state_cfg_path,
        device=args.device,
        er_init_mode=args.er_init_mode,
        flux_model=state_flux_model,
    )
    runtime, state = build_runtime_context(shared_state_config)

    db_cfg = _prepare_config(Path(args.database_config), device=args.device, er_init_mode=args.er_init_mode, flux_model="ntx_database")
    db_path = Path(db_cfg["neoclassical"]["neoclassical_file"])
    db_abs = (ROOT / db_path).resolve() if not db_path.is_absolute() else db_path.resolve()

    with h5py.File(db_abs, "r") as handle:
        rho_nodes = np.asarray(handle["rho"][()], dtype=float)
        nu_nodes = np.asarray(handle["nu_v"][()], dtype=float)
        er_nodes = np.asarray(handle["Er"][()], dtype=float)
        es_nodes = np.asarray(handle["Es"][()], dtype=float)

    exact_config = _prepare_config(
        ex_cfg_path,
        device=args.device,
        er_init_mode=args.er_init_mode,
        flux_model="ntx_exact_lij_runtime",
        resolution=resolution,
    )
    exact_runtime, _ = build_runtime_context(exact_config)
    exact_model = _extract_neoclassical_model(exact_runtime.models.flux)
    if not isinstance(exact_model, NTXExactLijRuntimeTransportModel):
        raise TypeError("Expected ntx_exact_lij_runtime neoclassical model.")
    exact_model = exact_model.with_static_support()
    support = exact_model._static_support()
    ntx = _import_ntx()

    radius_index = int(args.radius_index)
    species_index = int(args.species_index)
    energy_index = int(args.energy_index)

    density = safe_density(state.density)
    temperature = state.temperature
    er_profile = state.Er
    v_thermal = get_v_thermal(runtime.species.mass, temperature)

    rho_runtime_surface = float(runtime.geometry.rho_grid[radius_index])
    r_runtime = float(runtime.geometry.r_grid[radius_index])
    density_local = density[:, radius_index]
    temperature_local = temperature[:, radius_index]
    vthermal_local = v_thermal[:, radius_index]
    vth_a = jnp.asarray(vthermal_local[species_index], dtype=jnp.float64)
    v_new_a = runtime.energy_grid.v_norm * vth_a
    nu_hat_a = _nu_over_vnew_local(
        runtime.species,
        species_index,
        v_new_a,
        density_local,
        temperature_local,
        vthermal_local,
        _collisionality_kind(exact_model.collisionality_model),
    )
    nu_runtime = float(nu_hat_a[energy_index])
    rho_node_idx = _nearest_index(rho_nodes, rho_runtime_surface)
    nu_node_idx = _nearest_index(nu_nodes, nu_runtime)
    rho_on = float(rho_nodes[rho_node_idx])
    r_on = float(runtime.geometry.a_b * rho_on)
    nu_on = float(nu_nodes[nu_node_idx])

    node_count = min(int(args.num_field_nodes), int(er_nodes.shape[1]))
    field_indices = np.arange(node_count, dtype=int)

    kernel = monoenergetic_interpolation_kernel(runtime.database)
    neo = exact_config["neoclassical"]
    grid_spec = ntx.GridSpec(
        n_theta=int(neo["ntx_exact_n_theta"]),
        n_zeta=int(neo["ntx_exact_n_zeta"]),
        n_xi=int(neo["ntx_exact_n_xi"]),
    )
    vmec_path = Path(exact_config["geometry"]["vmec_file"])
    vmec_abs = (ROOT / vmec_path).resolve() if not vmec_path.is_absolute() else vmec_path.resolve()

    def _solve_slice(r_surface_value: float, rho_surface_value: float, nu_value: float, rho_index_for_field: int):
        surface = ntx.surface_from_vmec_jax_vmec_wout_file(str(vmec_abs), s=float(rho_surface_value**2))
        prepared = ntx.prepare_monoenergetic_system(surface, grid_spec)
        drds_value = jnp.asarray(es_nodes[rho_index_for_field, 1] / np.maximum(er_nodes[rho_index_for_field, 1], 1.0e-30), dtype=jnp.float64)

        db_list = []
        exact_list = []
        er_over_v_list = []
        for ier in field_indices:
            er_over_v_value = float(er_nodes[rho_index_for_field, ier])
            es_over_v_value = float(es_nodes[rho_index_for_field, ier])
            db_coeff = _database_channels_to_physical(
                kernel(r_surface_value, nu_value, er_over_v_value, runtime.database),
                jnp.asarray(nu_value, dtype=jnp.float64),
            )
            exact_coeff = _exact_raw_to_physical(
                ntx.solve_prepared_coefficient_vector(
                    prepared,
                    ntx.MonoenergeticCase(
                        nu_hat=jnp.asarray(nu_value, dtype=jnp.float64),
                        epsi_hat=jnp.asarray(es_over_v_value, dtype=jnp.float64),
                    ),
                ),
                drds_value,
            )
            db_list.append(np.asarray(db_coeff, dtype=float))
            exact_list.append(np.asarray(exact_coeff, dtype=float))
            er_over_v_list.append(er_over_v_value)
        return np.asarray(er_over_v_list), np.asarray(db_list), np.asarray(exact_list)

    cases = [
        ("on_rho_off_nu", r_on, rho_on, nu_runtime, rho_node_idx),
        ("off_rho_on_nu", r_runtime, rho_runtime_surface, nu_on, rho_node_idx),
        ("off_rho_off_nu", r_runtime, rho_runtime_surface, nu_runtime, rho_node_idx),
    ]

    output_dir = Path("outputs/benchmark_interpolation_isolation")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[interp-isolation] rho_runtime_surface={rho_runtime_surface:.12e} r_runtime={r_runtime:.12e}")
    print(f"[interp-isolation] rho_on={rho_on:.12e} r_on={r_on:.12e} rho_node_idx={rho_node_idx}")
    print(f"[interp-isolation] nu_runtime={nu_runtime:.12e} nu_on={nu_on:.12e} nu_node_idx={nu_node_idx}")
    print(f"[interp-isolation] species_index={species_index} energy_index={energy_index}")
    print(f"[interp-isolation] resolution={resolution}")
    print()

    for case_name, r_value, rho_surface_value, nu_value, rho_field_idx in cases:
        x, db_vals, exact_vals = _solve_slice(r_value, rho_surface_value, nu_value, rho_field_idx)
        abs_delta = np.abs(exact_vals - db_vals)
        rel_delta = abs_delta / np.maximum(np.abs(db_vals), 1.0e-30)

        fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
        for ax, idx, label in zip(axes, (0, 1, 2), ("D11", "D13", "D33")):
            ax.plot(x, db_vals[:, idx], label="database", linewidth=2.4)
            ax.plot(x, exact_vals[:, idx], label="exact NTX", linestyle="--", linewidth=2.0)
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
            ax.legend()
        axes[-1].set_xlabel("Er / v node")
        fig.suptitle(f"{case_name}: rho={rho_surface_value:.4f}, r={r_value:.4f}, nu={nu_value:.4e}")
        fig.tight_layout()
        plot_path = output_dir / f"{case_name}_{resolution[0]}_{resolution[1]}_{resolution[2]}.png"
        fig.savefig(plot_path, dpi=170)
        plt.close(fig)

        print(f"[{case_name}] rho_surface={rho_surface_value:.12e} r={r_value:.12e} nu={nu_value:.12e} plot={plot_path}")
        print(f"{'quantity':<8} {'abs_max':>14} {'rel_max':>14}")
        print("-" * 40)
        for idx, label in enumerate(("D11", "D13", "D33")):
            print(f"{label:<8} {float(np.max(abs_delta[:, idx])):14.6e} {float(np.max(rel_delta[:, idx])):14.6e}")
        print()


if __name__ == "__main__":
    main()
