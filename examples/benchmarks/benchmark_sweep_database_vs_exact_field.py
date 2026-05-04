"""Sweep field input locally and compare database interpolation vs exact NTX.

This uses the conventions we agreed are physically consistent:
- database path queried with Er / v
- exact NTX path solved with Es / v = (Er / v) * drds

At one radius and species, it fixes one transport-energy point (or a chosen
monoenergetic nu_hat) and sweeps the local field range to measure off-node
surrogate error, especially in D11.
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
from NEOPAX._neoclassical import _collisionality_kind, _nu_over_vnew_local
from NEOPAX._orchestrator import build_runtime_context
from NEOPAX._state import get_v_thermal, safe_density
from NEOPAX._transport_flux_models import NTXExactLijRuntimeTransportModel

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
    parser.add_argument("--num-points", type=int, default=25)
    parser.add_argument("--field-factor-min", type=float, default=0.5)
    parser.add_argument("--field-factor-max", type=float, default=1.5)
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

    radius_index = int(args.radius_index)
    species_index = int(args.species_index)
    energy_index = int(args.energy_index)

    density = safe_density(state.density)
    temperature = state.temperature
    er_profile = state.Er
    v_thermal = get_v_thermal(runtime.species.mass, temperature)

    radius_value = jnp.asarray(runtime.geometry.r_grid[radius_index], dtype=jnp.float64)
    er_value = jnp.asarray(er_profile[radius_index], dtype=jnp.float64)
    density_local = density[:, radius_index]
    temperature_local = temperature[:, radius_index]
    vthermal_local = v_thermal[:, radius_index]
    vth_a = jnp.asarray(vthermal_local[species_index], dtype=jnp.float64)
    v_new_a = runtime.energy_grid.v_norm * vth_a
    drds_value = jnp.asarray(support.center_channels.drds[radius_index], dtype=jnp.float64)

    collisionality_kind = _collisionality_kind(exact_model.collisionality_model)
    nu_hat_a = _nu_over_vnew_local(
        runtime.species,
        species_index,
        v_new_a,
        density_local,
        temperature_local,
        vthermal_local,
        collisionality_kind,
    )
    nu_hat_value = jnp.asarray(nu_hat_a[energy_index], dtype=jnp.float64)
    er_over_v_ref = jnp.asarray((er_value * 1.0e3) / v_new_a[energy_index], dtype=jnp.float64)

    field_factors = jnp.linspace(
        jnp.asarray(args.field_factor_min, dtype=jnp.float64),
        jnp.asarray(args.field_factor_max, dtype=jnp.float64),
        int(args.num_points),
    )
    er_over_v_grid = er_over_v_ref * field_factors
    es_over_v_grid = er_over_v_grid * drds_value

    kernel = monoenergetic_interpolation_kernel(runtime.database)
    db_coeffs = jax.vmap(
        lambda er_over_v_value: _database_channels_to_physical(
            kernel(radius_value, nu_hat_value, er_over_v_value, runtime.database),
            nu_hat_value,
        )
    )(er_over_v_grid)

    prepared = jax.tree_util.tree_map(
        lambda arr: jax.lax.dynamic_index_in_dim(arr, radius_index, axis=0, keepdims=False),
        support.center_prepared,
    )
    exact_coeffs = jax.vmap(
        lambda es_over_v_value: _exact_raw_to_physical(
            exact_model._solve_coefficient_scan_prepared(
                prepared,
                jnp.asarray([nu_hat_value], dtype=jnp.float64),
                jnp.asarray([es_over_v_value], dtype=jnp.float64),
            )[0],
            drds_value,
        )
    )(es_over_v_grid)

    abs_delta = jnp.abs(exact_coeffs - db_coeffs)
    rel_delta = abs_delta / jnp.maximum(jnp.abs(db_coeffs), 1.0e-30)

    output_dir = Path("outputs/benchmark_runtime_field_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)
    x = np.asarray(er_over_v_grid)
    db_np = np.asarray(db_coeffs)
    exact_np = np.asarray(exact_coeffs)

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for ax, idx, name in zip(axes, (0, 1, 2), ("D11", "D13", "D33")):
        ax.plot(x, db_np[:, idx], label="database", linewidth=2.4)
        ax.plot(x, exact_np[:, idx], label="exact NTX", linestyle="--", linewidth=2.0)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("Er / v")
    fig.suptitle(
        f"rho_idx={radius_index}, species_idx={species_index}, energy_idx={energy_index}, "
        f"nu_hat={float(nu_hat_value):.4e}, drds={float(drds_value):.4e}"
    )
    fig.tight_layout()
    plot_path = output_dir / (
        f"field_sweep_radius_{radius_index}_species_{species_index}_energy_{energy_index}_"
        f"{resolution[0]}_{resolution[1]}_{resolution[2]}.png"
    )
    fig.savefig(plot_path, dpi=170)
    plt.close(fig)

    print(f"[field-sweep] device={args.device}")
    print(f"[field-sweep] er_init_mode={args.er_init_mode}")
    print(f"[field-sweep] state_source_model={args.state_source_model}")
    print(f"[field-sweep] radius_index={radius_index}")
    print(f"[field-sweep] species_index={species_index}")
    print(f"[field-sweep] energy_index={energy_index}")
    print(f"[field-sweep] resolution={resolution}")
    print(f"[field-sweep] rho={float(radius_value):.12e}")
    print(f"[field-sweep] drds={float(drds_value):.12e}")
    print(f"[field-sweep] nu_hat={float(nu_hat_value):.12e}")
    print(f"[field-sweep] Er_over_v_ref={float(er_over_v_ref):.12e}")
    print(f"[field-sweep] Er_over_v_range=[{float(jnp.min(er_over_v_grid)):.6e}, {float(jnp.max(er_over_v_grid)):.6e}]")
    print(f"[field-sweep] Es_over_v_range=[{float(jnp.min(es_over_v_grid)):.6e}, {float(jnp.max(es_over_v_grid)):.6e}]")
    print(f"[field-sweep] plot={plot_path}")
    print()
    print(f"{'quantity':<8} {'abs_max':>14} {'rel_max':>14}")
    print("-" * 40)
    for idx, label in enumerate(("D11", "D13", "D33")):
        print(
            f"{label:<8} "
            f"{float(jnp.max(abs_delta[:, idx])):14.6e} "
            f"{float(jnp.max(rel_delta[:, idx])):14.6e}"
        )


if __name__ == "__main__":
    main()
