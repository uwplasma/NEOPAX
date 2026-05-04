"""Compare database interpolation modes against exact NTX on one off-node case.

This focuses on the monoenergetic database interpolation mode itself. It uses:
- the same off-node runtime rho and nu from a chosen radius/species/energy
- the stored file field nodes as the field sweep
- database queried with Er/v
- exact NTX solved with Es/v

It then compares several database interpolation modes against exact NTX:
- preprocessed_3d_radial_ntss1d
- preprocessed_3d_radial
- preprocessed_3d
- generic
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

    db_cfg = _prepare_config(
        db_cfg_path,
        device=args.device,
        er_init_mode=args.er_init_mode,
        flux_model="ntx_database",
    )
    db_path = Path(db_cfg["neoclassical"]["neoclassical_file"])
    db_abs = (ROOT / db_path).resolve() if not db_path.is_absolute() else db_path.resolve()

    with h5py.File(db_abs, "r") as handle:
        rho_nodes = np.asarray(handle["rho"][()], dtype=float)
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
    v_thermal = get_v_thermal(runtime.species.mass, temperature)

    rho_runtime = float(runtime.geometry.r_grid[radius_index])
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
    rho_node_idx = int(np.argmin(np.abs(rho_nodes - rho_runtime)))
    rho_field = float(rho_nodes[rho_node_idx])

    node_count = min(int(args.num_field_nodes), int(er_nodes.shape[1]))
    field_indices = np.arange(node_count, dtype=int)
    er_over_v_axis = np.asarray(er_nodes[rho_node_idx, field_indices], dtype=float)
    es_over_v_axis = np.asarray(es_nodes[rho_node_idx, field_indices], dtype=float)

    neo = exact_config["neoclassical"]
    grid_spec = ntx.GridSpec(
        n_theta=int(neo["ntx_exact_n_theta"]),
        n_zeta=int(neo["ntx_exact_n_zeta"]),
        n_xi=int(neo["ntx_exact_n_xi"]),
    )
    vmec_path = Path(exact_config["geometry"]["vmec_file"])
    vmec_abs = (ROOT / vmec_path).resolve() if not vmec_path.is_absolute() else vmec_path.resolve()
    surface = ntx.surface_from_vmec_jax_vmec_wout_file(str(vmec_abs), s=float(rho_runtime**2))
    prepared = ntx.prepare_monoenergetic_system(surface, grid_spec)
    drds_value = jnp.asarray(support.center_channels.drds[radius_index], dtype=jnp.float64)

    exact_vals = []
    for es_over_v_value in es_over_v_axis:
        exact_vals.append(
            np.asarray(
                _exact_raw_to_physical(
                    ntx.solve_prepared_coefficient_vector(
                        prepared,
                        ntx.MonoenergeticCase(
                            nu_hat=jnp.asarray(nu_runtime, dtype=jnp.float64),
                            epsi_hat=jnp.asarray(es_over_v_value, dtype=jnp.float64),
                        ),
                    ),
                    drds_value,
                ),
                dtype=float,
            )
        )
    exact_vals = np.asarray(exact_vals)

    modes = [
        "preprocessed_3d_radial_ntss1d",
        "preprocessed_3d_radial",
        "preprocessed_3d",
        "generic",
    ]
    mode_results: dict[str, np.ndarray] = {}

    for mode in modes:
        mode_cfg = _prepare_config(
            db_cfg_path,
            device=args.device,
            er_init_mode=args.er_init_mode,
            flux_model="ntx_database",
            interpolation_mode=mode,
        )
        mode_runtime, _ = build_runtime_context(mode_cfg)
        kernel = monoenergetic_interpolation_kernel(mode_runtime.database)
        values = []
        for er_over_v_value in er_over_v_axis:
            values.append(
                np.asarray(
                    _database_channels_to_physical(
                        kernel(rho_runtime, nu_runtime, er_over_v_value, mode_runtime.database),
                        jnp.asarray(nu_runtime, dtype=jnp.float64),
                    ),
                    dtype=float,
                )
            )
        mode_results[mode] = np.asarray(values)

    output_dir = Path("outputs/benchmark_interpolation_mode_compare")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    colors = {
        "preprocessed_3d_radial_ntss1d": "C0",
        "preprocessed_3d_radial": "C1",
        "preprocessed_3d": "C2",
        "generic": "C3",
    }
    for ax, idx, label in zip(axes, (0, 1, 2), ("D11", "D13", "D33")):
        ax.plot(er_over_v_axis, exact_vals[:, idx], color="black", linestyle="--", linewidth=2.4, label="exact NTX")
        for mode in modes:
            ax.plot(
                er_over_v_axis,
                mode_results[mode][:, idx],
                color=colors[mode],
                linewidth=2.0,
                label=mode,
            )
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("Er / v node")
    fig.suptitle(
        f"rho_runtime={rho_runtime:.4f}, nu_runtime={nu_runtime:.4e}, "
        f"rho_field={rho_field:.4f}, resolution={resolution}"
    )
    fig.tight_layout()
    plot_path = output_dir / f"mode_compare_radius_{radius_index}_species_{species_index}_energy_{energy_index}_{resolution[0]}_{resolution[1]}_{resolution[2]}.png"
    fig.savefig(plot_path, dpi=170)
    plt.close(fig)

    print(f"[mode-compare] radius_index={radius_index} species_index={species_index} energy_index={energy_index}")
    print(f"[mode-compare] rho_runtime={rho_runtime:.12e} rho_field_nodes={rho_field:.12e}")
    print(f"[mode-compare] nu_runtime={nu_runtime:.12e}")
    print(f"[mode-compare] plot={plot_path}")
    print()
    for mode in modes:
        vals = mode_results[mode]
        abs_delta = np.abs(vals - exact_vals)
        rel_delta = abs_delta / np.maximum(np.abs(exact_vals), 1.0e-30)
        print(f"[{mode}]")
        print(f"{'quantity':<8} {'abs_max':>14} {'rel_max':>14}")
        print("-" * 40)
        for idx, label in enumerate(("D11", "D13", "D33")):
            print(f"{label:<8} {float(np.max(abs_delta[:, idx])):14.6e} {float(np.max(rel_delta[:, idx])):14.6e}")
        print()


if __name__ == "__main__":
    main()
