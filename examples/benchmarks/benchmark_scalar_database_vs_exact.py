"""Scalar audit of database vs exact NTX at one runtime energy point.

This prints the exact scalar inputs and outputs for:
- the actual off-node runtime case at one radius/species/energy
- a nearby stored nonzero field node at the nearest stored rho

It is meant to answer whether the remaining mismatch is in:
- field-coordinate equivalence, or
- bridged coefficient equivalence.
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
    parser.add_argument("--radius-index", type=int, default=10)
    parser.add_argument("--species-index", type=int, default=1)
    parser.add_argument("--energy-index", type=int, default=8)
    parser.add_argument("--field-node-index", type=int, default=3)
    parser.add_argument("--resolution", default="25,25,63")
    args = parser.parse_args()

    resolution = _parse_resolution(args.resolution)
    db_cfg = _prepare_config(
        Path(args.database_config),
        device=args.device,
        er_init_mode=args.er_init_mode,
        flux_model="ntx_database",
    )
    runtime, state = build_runtime_context(db_cfg)
    db_path = Path(db_cfg["neoclassical"]["neoclassical_file"])
    db_abs = (ROOT / db_path).resolve() if not db_path.is_absolute() else db_path.resolve()

    with h5py.File(db_abs, "r") as handle:
        rho_nodes = np.asarray(handle["rho"][()], dtype=float)
        er_nodes = np.asarray(handle["Er"][()], dtype=float)
        es_nodes = np.asarray(handle["Es"][()], dtype=float)

    exact_cfg = _prepare_config(
        Path(args.exact_config),
        device=args.device,
        er_init_mode=args.er_init_mode,
        flux_model="ntx_exact_lij_runtime",
        resolution=resolution,
    )
    exact_runtime, _ = build_runtime_context(exact_cfg)
    exact_model = _extract_neoclassical_model(exact_runtime.models.flux)
    if not isinstance(exact_model, NTXExactLijRuntimeTransportModel):
        raise TypeError("Expected ntx_exact_lij_runtime neoclassical model.")
    exact_model = exact_model.with_static_support()
    support = exact_model._static_support()
    ntx = _import_ntx()

    radius_index = int(args.radius_index)
    species_index = int(args.species_index)
    energy_index = int(args.energy_index)
    field_node_index = int(args.field_node_index)

    density = safe_density(state.density)
    temperature = state.temperature
    v_thermal = get_v_thermal(runtime.species.mass, temperature)
    er_profile = state.Er

    rho_runtime = float(runtime.geometry.r_grid[radius_index])
    er_runtime = float(er_profile[radius_index])
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
    er_over_v_runtime = float((er_runtime * 1.0e3) / v_new_a[energy_index])
    drds_runtime = float(support.center_channels.drds[radius_index])
    es_over_v_runtime = er_over_v_runtime * drds_runtime

    rho_node_idx = _nearest_index(rho_nodes, rho_runtime)
    rho_node = float(rho_nodes[rho_node_idx])
    er_over_v_node = float(er_nodes[rho_node_idx, field_node_index])
    es_over_v_node = float(es_nodes[rho_node_idx, field_node_index])

    kernel = monoenergetic_interpolation_kernel(runtime.database)
    db_raw_runtime = kernel(rho_runtime, nu_runtime, er_over_v_runtime, runtime.database)
    db_phys_runtime = _database_channels_to_physical(db_raw_runtime, jnp.asarray(nu_runtime, dtype=jnp.float64))

    db_raw_node = kernel(rho_node, nu_runtime, er_over_v_node, runtime.database)
    db_phys_node = _database_channels_to_physical(db_raw_node, jnp.asarray(nu_runtime, dtype=jnp.float64))

    neo = exact_cfg["neoclassical"]
    grid_spec = ntx.GridSpec(
        n_theta=int(neo["ntx_exact_n_theta"]),
        n_zeta=int(neo["ntx_exact_n_zeta"]),
        n_xi=int(neo["ntx_exact_n_xi"]),
    )
    vmec_path = Path(exact_cfg["geometry"]["vmec_file"])
    vmec_abs = (ROOT / vmec_path).resolve() if not vmec_path.is_absolute() else vmec_path.resolve()
    surface_runtime = ntx.surface_from_vmec_jax_vmec_wout_file(str(vmec_abs), s=float(rho_runtime**2))
    prepared_runtime = ntx.prepare_monoenergetic_system(surface_runtime, grid_spec)
    exact_raw_runtime = ntx.solve_prepared_coefficient_vector(
        prepared_runtime,
        ntx.MonoenergeticCase(
            nu_hat=jnp.asarray(nu_runtime, dtype=jnp.float64),
            epsi_hat=jnp.asarray(es_over_v_runtime, dtype=jnp.float64),
        ),
    )
    exact_phys_runtime = _exact_raw_to_physical(exact_raw_runtime, jnp.asarray(drds_runtime, dtype=jnp.float64))

    surface_node = ntx.surface_from_vmec_jax_vmec_wout_file(str(vmec_abs), s=float(rho_node**2))
    prepared_node = ntx.prepare_monoenergetic_system(surface_node, grid_spec)
    drds_node = float(es_over_v_node / max(er_over_v_node, 1.0e-30)) if er_over_v_node != 0.0 else drds_runtime
    exact_raw_node = ntx.solve_prepared_coefficient_vector(
        prepared_node,
        ntx.MonoenergeticCase(
            nu_hat=jnp.asarray(nu_runtime, dtype=jnp.float64),
            epsi_hat=jnp.asarray(es_over_v_node, dtype=jnp.float64),
        ),
    )
    exact_phys_node = _exact_raw_to_physical(exact_raw_node, jnp.asarray(drds_node, dtype=jnp.float64))

    print("[scalar-audit] runtime off-node case")
    print(f"  rho_runtime         = {rho_runtime:.12e}")
    print(f"  nu_runtime          = {nu_runtime:.12e}")
    print(f"  Er_runtime          = {er_runtime:.12e}")
    print(f"  Er_over_v_runtime   = {er_over_v_runtime:.12e}")
    print(f"  Es_over_v_runtime   = {es_over_v_runtime:.12e}")
    print(f"  drds_runtime        = {drds_runtime:.12e}")
    print(f"  grid_er_internal    = {float(jnp.log10(max(1.0e-8, abs(er_over_v_runtime / rho_runtime)))):.12e}")
    print("  database physical   =", np.asarray(db_phys_runtime, dtype=float))
    print("  exact physical      =", np.asarray(exact_phys_runtime, dtype=float))
    print("  abs delta           =", np.asarray(jnp.abs(exact_phys_runtime - db_phys_runtime), dtype=float))
    print()
    print("[scalar-audit] nearby nonzero stored-field case")
    print(f"  rho_node_idx        = {rho_node_idx}")
    print(f"  rho_node            = {rho_node:.12e}")
    print(f"  field_node_index    = {field_node_index}")
    print(f"  Er_over_v_node      = {er_over_v_node:.12e}")
    print(f"  Es_over_v_node      = {es_over_v_node:.12e}")
    print(f"  drds_node           = {drds_node:.12e}")
    print(f"  grid_er_internal    = {float(jnp.log10(max(1.0e-8, abs(er_over_v_node / rho_node)))):.12e}")
    print("  database physical   =", np.asarray(db_phys_node, dtype=float))
    print("  exact physical      =", np.asarray(exact_phys_node, dtype=float))
    print("  abs delta           =", np.asarray(jnp.abs(exact_phys_node - db_phys_node), dtype=float))


if __name__ == "__main__":
    main()
