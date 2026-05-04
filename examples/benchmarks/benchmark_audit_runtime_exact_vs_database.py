"""Audit runtime exact NTX assembly against the database path at one radius/species.

This isolates the first point where the exact-runtime path can diverge from the
trusted database path on the same initialized transport state.

For one chosen radius/species, it compares:
- database-reconstructed monoenergetic coefficients queried at runtime
- exact NTX coefficients bridged with the same NEOPAX raw->transport mapping

for two exact NTX field-input conventions:
- current runtime convention: epsi_hat = Er / v
- candidate NTX convention: epsi_hat = Es / v = (Er / v) * drds

It reports coefficient, transport-moment, and Lij deltas.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
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


def _database_channels_to_physical(db_coeffs: jax.Array, nu_hat_a: jax.Array) -> jax.Array:
    return jnp.stack(
        (
            -10.0 ** jnp.asarray(db_coeffs[:, 0], dtype=jnp.float64),
            -jnp.asarray(db_coeffs[:, 1], dtype=jnp.float64),
            -jnp.asarray(db_coeffs[:, 2], dtype=jnp.float64) / jnp.maximum(jnp.asarray(nu_hat_a, dtype=jnp.float64), 1.0e-30),
        ),
        axis=1,
    )


def _exact_raw_to_physical(exact_coeffs: jax.Array, drds_value: jax.Array) -> jax.Array:
    return jnp.stack(
        (
            -jnp.asarray(exact_coeffs[:, 0], dtype=jnp.float64) * drds_value**2,
            -jnp.asarray(exact_coeffs[:, 2], dtype=jnp.float64) * drds_value,
            -jnp.asarray(exact_coeffs[:, 3], dtype=jnp.float64),
        ),
        axis=1,
    )


def _transport_moments_from_physical(energy_grid, physical_coeffs: jax.Array) -> jax.Array:
    d11_a = physical_coeffs[:, 0]
    d13_a = physical_coeffs[:, 1]
    d33_a = physical_coeffs[:, 2]
    return jnp.stack(
        (
            jnp.sum(energy_grid.L11_weight * energy_grid.xWeights * d11_a),
            jnp.sum(energy_grid.L12_weight * energy_grid.xWeights * d11_a),
            jnp.sum(energy_grid.L22_weight * energy_grid.xWeights * d11_a),
            jnp.sum(energy_grid.L13_weight * energy_grid.xWeights * d13_a),
            jnp.sum(energy_grid.L23_weight * energy_grid.xWeights * d13_a),
            jnp.sum(energy_grid.L33_weight * energy_grid.xWeights * d33_a),
        ),
        axis=0,
    )


def _max_abs_rel(reference: jax.Array, candidate: jax.Array) -> tuple[float, float]:
    abs_max = float(jnp.max(jnp.abs(candidate - reference)))
    rel_max = float(jnp.max(jnp.abs(candidate - reference) / jnp.maximum(jnp.abs(reference), 1.0e-30)))
    return abs_max, rel_max


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
        help="Which model initializes the shared transport state.",
    )
    parser.add_argument("--radius-index", type=int, default=10)
    parser.add_argument("--species-index", type=int, default=1)
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

    radius_index = int(args.radius_index)
    species_index = int(args.species_index)

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

    epsi_hat_er_over_v = er_value * 1.0e3 / v_new_a
    epsi_hat_es_over_v = epsi_hat_er_over_v * drds_value

    kernel = monoenergetic_interpolation_kernel(runtime.database)
    db_coeffs_raw = jax.vmap(kernel, in_axes=(None, 0, 0, None))(
        radius_value,
        nu_hat_a,
        epsi_hat_er_over_v,
        runtime.database,
    )
    db_physical = _database_channels_to_physical(db_coeffs_raw, nu_hat_a)
    db_moments = _transport_moments_from_physical(runtime.energy_grid, db_physical)
    db_lij = exact_model._lij_from_transport_moments(
        db_moments,
        species_index=species_index,
        vth_a=vth_a,
    )

    prepared = jax.tree_util.tree_map(
        lambda arr: jax.lax.dynamic_index_in_dim(arr, radius_index, axis=0, keepdims=False),
        support.center_prepared,
    )

    coeffs_er_over_v = exact_model._solve_coefficient_scan_prepared(
        prepared,
        nu_hat_a,
        epsi_hat_er_over_v,
    )
    coeffs_es_over_v = exact_model._solve_coefficient_scan_prepared(
        prepared,
        nu_hat_a,
        epsi_hat_es_over_v,
    )

    exact_phys_er_over_v = _exact_raw_to_physical(coeffs_er_over_v, drds_value)
    exact_phys_es_over_v = _exact_raw_to_physical(coeffs_es_over_v, drds_value)

    exact_moments_er_over_v = _transport_moments_from_physical(runtime.energy_grid, exact_phys_er_over_v)
    exact_moments_es_over_v = _transport_moments_from_physical(runtime.energy_grid, exact_phys_es_over_v)

    exact_lij_er_over_v = exact_model._lij_from_transport_moments(
        exact_moments_er_over_v,
        species_index=species_index,
        vth_a=vth_a,
    )
    exact_lij_es_over_v = exact_model._lij_from_transport_moments(
        exact_moments_es_over_v,
        species_index=species_index,
        vth_a=vth_a,
    )

    jax.block_until_ready(db_lij)
    jax.block_until_ready(exact_lij_er_over_v)
    jax.block_until_ready(exact_lij_es_over_v)

    labels = ("D11", "D13", "D33")
    print(f"[runtime-audit] device={args.device}")
    print(f"[runtime-audit] er_init_mode={args.er_init_mode}")
    print(f"[runtime-audit] state_source_model={args.state_source_model}")
    print(f"[runtime-audit] radius_index={radius_index}")
    print(f"[runtime-audit] species_index={species_index}")
    print(f"[runtime-audit] resolution={resolution}")
    print(f"[runtime-audit] rho={float(radius_value):.12e}")
    print(f"[runtime-audit] Er={float(er_value):.12e}")
    print(f"[runtime-audit] drds={float(drds_value):.12e}")
    print(f"[runtime-audit] nu_hat_range=[{float(jnp.min(nu_hat_a)):.6e}, {float(jnp.max(nu_hat_a)):.6e}]")
    print(
        f"[runtime-audit] epsi_hat_ranges: "
        f"Er/v=[{float(jnp.min(epsi_hat_er_over_v)):.6e}, {float(jnp.max(epsi_hat_er_over_v)):.6e}] "
        f"Es/v=[{float(jnp.min(epsi_hat_es_over_v)):.6e}, {float(jnp.max(epsi_hat_es_over_v)):.6e}]"
    )
    print()
    print("monoenergetic physical coefficients vs database")
    print(f"{'quantity':<8} {'Er/v_abs':>14} {'Er/v_rel':>14} {'Es/v_abs':>14} {'Es/v_rel':>14}")
    print("-" * 68)
    for idx, label in enumerate(labels):
        abs_er, rel_er = _max_abs_rel(db_physical[:, idx], exact_phys_er_over_v[:, idx])
        abs_es, rel_es = _max_abs_rel(db_physical[:, idx], exact_phys_es_over_v[:, idx])
        print(f"{label:<8} {abs_er:14.6e} {rel_er:14.6e} {abs_es:14.6e} {rel_es:14.6e}")
    print()
    print("transport moments vs database")
    print(f"{'moment':<8} {'Er/v_abs':>14} {'Er/v_rel':>14} {'Es/v_abs':>14} {'Es/v_rel':>14}")
    print("-" * 68)
    for idx, label in enumerate(("M11", "M12", "M22", "M13", "M23", "M33")):
        abs_er, rel_er = _max_abs_rel(db_moments[idx], exact_moments_er_over_v[idx])
        abs_es, rel_es = _max_abs_rel(db_moments[idx], exact_moments_es_over_v[idx])
        print(f"{label:<8} {abs_er:14.6e} {rel_er:14.6e} {abs_es:14.6e} {rel_es:14.6e}")
    print()
    print("Lij matrix vs database")
    print(f"{'entry':<8} {'Er/v_abs':>14} {'Er/v_rel':>14} {'Es/v_abs':>14} {'Es/v_rel':>14}")
    print("-" * 68)
    for i, j, label in (
        (0, 0, "L11"),
        (0, 1, "L12"),
        (1, 1, "L22"),
        (0, 2, "L13"),
        (1, 2, "L23"),
        (2, 2, "L33"),
    ):
        abs_er, rel_er = _max_abs_rel(db_lij[i, j], exact_lij_er_over_v[i, j])
        abs_es, rel_es = _max_abs_rel(db_lij[i, j], exact_lij_es_over_v[i, j])
        print(f"{label:<8} {abs_er:14.6e} {rel_er:14.6e} {abs_es:14.6e} {rel_es:14.6e}")


if __name__ == "__main__":
    main()
