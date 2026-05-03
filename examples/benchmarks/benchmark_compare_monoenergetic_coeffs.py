"""Compare raw monoenergetic coefficients between database and exact-runtime NTX.

This diagnostic is intentionally local:
- pick one radius index
- pick one species index
- build one initialized shared state
- compare raw database-interpolated D11/D13/D33 against exact-runtime prepared solves

It also compares two exact-runtime input conventions:
- current NEOPAX transport path: `epsi_hat = (Er / v) * drds`
- explicit `er_hat` path, derived from the same resolved `epsi_hat`

It reports both:
- raw exact NTX coefficients
- exact coefficients mapped through the same NEOPAX bridge convention as the database
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

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
from NEOPAX._state import get_v_thermal, safe_density
from NEOPAX._transport_flux_models import (
    NTXExactLijRuntimeTransportModel,
    _collisionality_kind,
    _import_ntx,
)

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


def _species_label(runtime, species_index: int) -> str:
    names = getattr(runtime.species, "names", None)
    if names is not None and species_index < len(names):
        return str(names[species_index])
    return f"s{species_index}"


def _plot_coefficients(output_dir: Path, x_grid, curves: dict[str, np.ndarray], title: str, stem: str):
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    quantity_order = [("D11", 0), ("D13", 1), ("D33", 2)]
    styles = {
        "database": ("solid", 2.4),
        "exact_epsihat": ("--", 2.0),
        "exact_erhat": (":", 2.0),
    }

    for ax, (name, idx) in zip(axes, quantity_order):
        for label, values in curves.items():
            linestyle, linewidth = styles.get(label, ("-.", 2.0))
            ax.plot(x_grid, values[:, idx], linestyle=linestyle, linewidth=linewidth, label=label)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("x")
    fig.suptitle(title)
    fig.tight_layout()
    out = output_dir / f"{stem}.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def _database_channels_to_physical(db_coeffs: jax.Array, nu_hat_a: jax.Array) -> jax.Array:
    """Convert stored database channels to physical D11/D13/D33."""
    return jnp.stack(
        (
            -10.0 ** jnp.asarray(db_coeffs[:, 0], dtype=jnp.float64),
            -jnp.asarray(db_coeffs[:, 1], dtype=jnp.float64),
            -jnp.asarray(db_coeffs[:, 2], dtype=jnp.float64) / jnp.maximum(jnp.asarray(nu_hat_a, dtype=jnp.float64), 1.0e-30),
        ),
        axis=1,
    )


def _exact_coeff_vector_to_physical(exact_coeffs: jax.Array) -> jax.Array:
    """Select the comparable D11/D13/D33 channels from the NTX coefficient vector."""
    return jnp.stack(
        (
            jnp.asarray(exact_coeffs[:, 0], dtype=jnp.float64),
            jnp.asarray(exact_coeffs[:, 2], dtype=jnp.float64),
            jnp.asarray(exact_coeffs[:, 3], dtype=jnp.float64),
        ),
        axis=1,
    )


def _exact_coeff_vector_to_neopax_physical(exact_coeffs: jax.Array, drds_value: jax.Array) -> jax.Array:
    drds_value = jnp.asarray(drds_value, dtype=jnp.float64)
    return jnp.stack(
        (
            jnp.asarray(exact_coeffs[:, 0], dtype=jnp.float64) * drds_value**2,
            jnp.asarray(exact_coeffs[:, 2], dtype=jnp.float64) * drds_value,
            jnp.asarray(exact_coeffs[:, 3], dtype=jnp.float64),
        ),
        axis=1,
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
        help="Which model initializes the shared state before local coefficient comparison.",
    )
    parser.add_argument("--radius-index", type=int, default=10)
    parser.add_argument("--species-index", type=int, default=1)
    parser.add_argument("--resolution", default="5,21,32")
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

    radius_index = int(args.radius_index)
    species_index = int(args.species_index)
    if radius_index < 0 or radius_index >= int(state.Er.shape[0]):
        raise ValueError(f"radius-index {radius_index} out of bounds for n_radius={int(state.Er.shape[0])}")
    if species_index < 0 or species_index >= int(runtime.species.number_species):
        raise ValueError(
            f"species-index {species_index} out of bounds for n_species={int(runtime.species.number_species)}"
        )

    density = safe_density(state.density)
    temperature = state.temperature
    er_profile = state.Er
    v_thermal = get_v_thermal(runtime.species.mass, temperature)

    ex_config = _prepare_config(
        ex_cfg_path,
        device=args.device,
        er_init_mode=args.er_init_mode,
        flux_model="ntx_exact_lij_runtime",
        resolution=resolution,
    )
    ex_runtime, _ = build_runtime_context(ex_config)
    exact_model = _extract_neoclassical_model(ex_runtime.models.flux)
    if not isinstance(exact_model, NTXExactLijRuntimeTransportModel):
        raise TypeError("Expected ntx_exact_lij_runtime neoclassical model.")
    exact_model = exact_model.with_static_support()

    radius_value = float(runtime.geometry.r_grid[radius_index])
    er_value = float(er_profile[radius_index])
    temperature_local = temperature[:, radius_index]
    density_local = density[:, radius_index]
    vthermal_local = v_thermal[:, radius_index]
    vth_a = vthermal_local[species_index]
    v_new_a = runtime.energy_grid.v_norm * vth_a
    drds_value = jnp.asarray(exact_model._static_support().center_channels.drds[radius_index], dtype=jnp.float64)

    kernel = monoenergetic_interpolation_kernel(runtime.database)
    nu_hat_a, epsi_hat_a, _ = exact_model._local_scan_inputs(
        drds_value=drds_value,
        species_index=species_index,
        er_value=jnp.asarray(er_value, dtype=jnp.float64),
        temperature_local=temperature_local,
        density_local=density_local,
        vthermal_local=vthermal_local,
        collisionality_kind=_collisionality_kind(exact_model.collisionality_model),
    )

    er_vnew_a = jnp.asarray(er_value * 1.0e3 / v_new_a, dtype=jnp.float64)
    db_coeffs_raw = jax.vmap(kernel, in_axes=(None, 0, 0, None))(radius_value, nu_hat_a, er_vnew_a, runtime.database)

    support = exact_model._static_support()
    prepared = jax.tree_util.tree_map(
        lambda arr: jax.lax.dynamic_index_in_dim(arr, radius_index, axis=0, keepdims=False),
        support.center_prepared,
    )
    exact_epsihat_raw = jax.vmap(
        lambda nu_hat_value, epsi_hat_value: exact_model._solve_coefficient_scan_prepared(
            prepared,
            jnp.asarray([nu_hat_value], dtype=jnp.float64),
            jnp.asarray([epsi_hat_value], dtype=jnp.float64),
        )[0]
    )(nu_hat_a, epsi_hat_a)

    transport_scale = jnp.asarray(prepared.geometry.transport_psi_scale, dtype=jnp.float64)
    er_hat_a = epsi_hat_a * transport_scale
    ntx = _import_ntx()
    exact_erhat_raw = jax.vmap(
        lambda nu_hat_value, er_hat_value: ntx.solve_prepared_coefficient_vector(
            prepared,
            ntx.MonoenergeticCase(
                nu_hat=jnp.asarray(nu_hat_value, dtype=jnp.float64),
                er_hat=jnp.asarray(er_hat_value, dtype=jnp.float64),
            ),
        )
    )(nu_hat_a, er_hat_a)

    db_coeffs = _database_channels_to_physical(db_coeffs_raw, nu_hat_a)
    exact_epsihat_raw_phys = _exact_coeff_vector_to_physical(exact_epsihat_raw)
    exact_erhat_raw_phys = _exact_coeff_vector_to_physical(exact_erhat_raw)
    exact_epsihat = _exact_coeff_vector_to_neopax_physical(exact_epsihat_raw, drds_value)
    exact_erhat = _exact_coeff_vector_to_neopax_physical(exact_erhat_raw, drds_value)

    jax.block_until_ready(db_coeffs)
    jax.block_until_ready(exact_epsihat)
    jax.block_until_ready(exact_erhat)

    output_dir = Path("outputs/benchmark_monoenergetic_compare")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_path = _plot_coefficients(
        output_dir,
        np.asarray(runtime.energy_grid.x),
        {
            "database": np.asarray(db_coeffs),
            "exact_epsihat": np.asarray(exact_epsihat),
            "exact_erhat": np.asarray(exact_erhat),
        },
        title=(
            f"radius={radius_index} rho={float(getattr(runtime.geometry, 'rho_grid', runtime.geometry.r_grid)[radius_index]):.4f}, "
            f"species={_species_label(runtime, species_index)}, Er_init={args.er_init_mode}, "
            f"resolution={resolution}"
        ),
        stem=f"monoenergetic_radius_{radius_index}_species_{species_index}_{resolution[0]}_{resolution[1]}_{resolution[2]}",
    )

    def _delta(reference, candidate, index):
        abs_max = float(jnp.max(jnp.abs(candidate[:, index] - reference[:, index])))
        rel_max = float(
            jnp.max(jnp.abs(candidate[:, index] - reference[:, index]) / jnp.maximum(jnp.abs(reference[:, index]), 1.0e-30))
        )
        return abs_max, rel_max

    print(f"[mono-compare] device={args.device}")
    print(f"[mono-compare] er_init_mode={args.er_init_mode}")
    print(f"[mono-compare] state_source_model={args.state_source_model}")
    print(f"[mono-compare] radius_index={radius_index}")
    print(f"[mono-compare] species_index={species_index} ({_species_label(runtime, species_index)})")
    print(f"[mono-compare] resolution={resolution}")
    print(f"[mono-compare] drds={float(drds_value):.6e}")
    print(f"[mono-compare] plot={plot_path}")
    print()
    print("raw exact coefficients vs database")
    print("quantity    epsihat_abs      epsihat_rel        erhat_abs        erhat_rel")
    print("---------------------------------------------------------------------------")
    for name, idx in (("D11", 0), ("D13", 1), ("D33", 2)):
        abs_eps, rel_eps = _delta(db_coeffs, exact_epsihat_raw_phys, idx)
        abs_erh, rel_erh = _delta(db_coeffs, exact_erhat_raw_phys, idx)
        print(f"{name:<10}{abs_eps:>14.6e}{rel_eps:>18.6e}{abs_erh:>18.6e}{rel_erh:>18.6e}")
    print()
    print("NEOPAX-bridge-scaled exact coefficients vs database")
    print("quantity    epsihat_abs      epsihat_rel        erhat_abs        erhat_rel")
    print("---------------------------------------------------------------------------")
    for name, idx in (("D11", 0), ("D13", 1), ("D33", 2)):
        abs_eps, rel_eps = _delta(db_coeffs, exact_epsihat, idx)
        abs_erh, rel_erh = _delta(db_coeffs, exact_erhat, idx)
        print(f"{name:<10}{abs_eps:>14.6e}{rel_eps:>18.6e}{abs_erh:>18.6e}{rel_erh:>18.6e}")


if __name__ == "__main__":
    main()
