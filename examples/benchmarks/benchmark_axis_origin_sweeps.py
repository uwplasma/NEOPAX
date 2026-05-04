"""Isolate off-node error by varying one monoenergetic input at a time.

For one chosen runtime case, this script builds three sweeps:
- vary only radius, keep nu and field on stored node values
- vary only nu, keep radius and field on stored node values
- vary only field, keep radius and nu on stored node values

For each sweep it compares:
- generic
- preprocessed_3d
- preprocessed_3d_radial_ntss1d
- exact NTX

This helps identify which axis contributes most to the off-node mismatch.
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


def _nearest_idx(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def _midpoint_sweep_indices(center_idx: int, window: int, n_values: int) -> list[int]:
    lo = max(0, center_idx - window)
    hi = min(n_values - 1, center_idx + window + 1)
    return list(range(lo, hi))


def _print_mode_table(name: str, results: dict[str, np.ndarray], exact_vals: np.ndarray):
    print(f"[{name}]")
    for mode, vals in results.items():
        abs_delta = np.abs(vals - exact_vals)
        rel_delta = abs_delta / np.maximum(np.abs(exact_vals), 1.0e-30)
        print(f"  [{mode}]")
        print(f"    {'quantity':<8} {'abs_max':>14} {'rel_max':>14}")
        for idx, label in enumerate(("D11", "D13", "D33")):
            print(f"    {label:<8} {float(np.max(abs_delta[:, idx])):14.6e} {float(np.max(rel_delta[:, idx])):14.6e}")
    print()


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
    parser.add_argument("--field-index", type=int, default=6)
    parser.add_argument("--rho-window", type=int, default=1)
    parser.add_argument("--nu-window", type=int, default=1)
    parser.add_argument("--field-window", type=int, default=2)
    parser.add_argument("--resolution", default="25,25,63")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    args = parser.parse_args()

    resolution = _parse_resolution(args.resolution)
    modes = [item.strip() for item in str(args.modes).split(",") if item.strip()]

    db_cfg_path = Path(args.database_config)
    ex_cfg_path = Path(args.exact_config)

    state_cfg = _prepare_config(
        db_cfg_path,
        device=args.device,
        er_init_mode=args.er_init_mode,
        flux_model="ntx_database",
    )
    runtime, state = build_runtime_context(state_cfg)

    db_path = Path(state_cfg["neoclassical"]["neoclassical_file"])
    db_abs = (ROOT / db_path).resolve() if not db_path.is_absolute() else db_path.resolve()
    with h5py.File(db_abs, "r") as handle:
        rho_nodes = np.asarray(handle["rho"][()], dtype=float)
        nu_nodes = np.asarray(handle["nu_v"][()], dtype=float)
        er_nodes = np.asarray(handle["Er"][()], dtype=float)
        es_nodes = np.asarray(handle["Es"][()], dtype=float)

    ex_cfg = _prepare_config(
        ex_cfg_path,
        device=args.device,
        er_init_mode=args.er_init_mode,
        flux_model="ntx_exact_lij_runtime",
        resolution=resolution,
    )
    exact_runtime, _ = build_runtime_context(ex_cfg)
    exact_model = _extract_neoclassical_model(exact_runtime.models.flux)
    if not isinstance(exact_model, NTXExactLijRuntimeTransportModel):
        raise TypeError("Expected ntx_exact_lij_runtime neoclassical model.")
    exact_model = exact_model.with_static_support()
    support = exact_model._static_support()
    ntx = _import_ntx()

    density = safe_density(state.density)
    temperature = state.temperature
    v_thermal = get_v_thermal(runtime.species.mass, temperature)

    radius_index = int(args.radius_index)
    species_index = int(args.species_index)
    energy_index = int(args.energy_index)
    field_index = int(args.field_index)

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

    rho_node_idx = _nearest_idx(rho_nodes, rho_runtime_surface)
    nu_node_idx = _nearest_idx(nu_nodes, nu_runtime)
    rho_node = float(rho_nodes[rho_node_idx])
    r_node = float(runtime.geometry.a_b * rho_node)
    nu_node = float(nu_nodes[nu_node_idx])
    er_node = float(er_nodes[rho_node_idx, field_index])
    es_node = float(es_nodes[rho_node_idx, field_index])

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
            db_cfg_path,
            device=args.device,
            er_init_mode=args.er_init_mode,
            flux_model="ntx_database",
            interpolation_mode=mode,
        )
        mode_runtime, _ = build_runtime_context(cfg)
        mode_runtimes[mode] = (mode_runtime, monoenergetic_interpolation_kernel(mode_runtime.database))

    print(f"[axis-sweeps] database_file={db_abs}")
    print(f"[axis-sweeps] resolution={resolution}")
    print(f"[axis-sweeps] runtime case: radius_index={radius_index} rho_runtime_surface={rho_runtime_surface:.6e} r_runtime={r_runtime:.6e}")
    print(f"[axis-sweeps] species_index={species_index} energy_index={energy_index} nu_runtime={nu_runtime:.6e}")
    print(f"[axis-sweeps] nearest nodes: rho_idx={rho_node_idx} rho_node={rho_node:.6e} nu_idx={nu_node_idx} nu_node={nu_node:.6e} field_idx={field_index} Er/v_node={er_node:.6e}")
    print()

    # 1. Vary rho only: use rho midpoints, keep nu and Er/v fixed on node values.
    rho_sweep_idx = _midpoint_sweep_indices(rho_node_idx, args.rho_window, len(rho_nodes))
    rho_exact = []
    rho_results = {mode: [] for mode in modes}
    for ir in rho_sweep_idx:
        if ir + 1 >= len(rho_nodes):
            continue
        rho_val = 0.5 * float(rho_nodes[ir] + rho_nodes[ir + 1])
        r_val = float(runtime.geometry.a_b * rho_val)
        surface = ntx.surface_from_vmec_jax_vmec_wout_file(str(vmec_abs), s=float(rho_val**2))
        prepared = ntx.prepare_monoenergetic_system(surface, grid_spec)
        drds_val = float(prepared.geometry.transport_psi_scale)
        es_val = float(er_node * drds_val)
        rho_exact.append(
            np.asarray(
                _exact_raw_to_physical(
                    ntx.solve_prepared_coefficient_vector(
                        prepared,
                        ntx.MonoenergeticCase(
                            nu_hat=jnp.asarray(nu_node, dtype=jnp.float64),
                            epsi_hat=jnp.asarray(es_val, dtype=jnp.float64),
                        ),
                    ),
                    jnp.asarray(drds_val, dtype=jnp.float64),
                ),
                dtype=float,
            )
        )
        for mode in modes:
            mode_runtime, kernel = mode_runtimes[mode]
            rho_results[mode].append(
                np.asarray(
                    _database_channels_to_physical(
                        kernel(r_val, nu_node, er_node, mode_runtime.database),
                        jnp.asarray(nu_node, dtype=jnp.float64),
                    ),
                    dtype=float,
                )
            )
    _print_mode_table("vary-rho-only", {k: np.asarray(v) for k, v in rho_results.items()}, np.asarray(rho_exact))

    # 2. Vary nu only: use nu midpoints, keep rho and field on node values.
    nu_sweep_idx = _midpoint_sweep_indices(nu_node_idx, args.nu_window, len(nu_nodes))
    surface_node = ntx.surface_from_vmec_jax_vmec_wout_file(str(vmec_abs), s=float(rho_node**2))
    prepared_node = ntx.prepare_monoenergetic_system(surface_node, grid_spec)
    drds_node = float(es_node / er_node)
    nu_exact = []
    nu_results = {mode: [] for mode in modes}
    for inu in nu_sweep_idx:
        if inu + 1 >= len(nu_nodes):
            continue
        nu_val = float(np.sqrt(nu_nodes[inu] * nu_nodes[inu + 1]))
        nu_exact.append(
            np.asarray(
                _exact_raw_to_physical(
                    ntx.solve_prepared_coefficient_vector(
                        prepared_node,
                        ntx.MonoenergeticCase(
                            nu_hat=jnp.asarray(nu_val, dtype=jnp.float64),
                            epsi_hat=jnp.asarray(es_node, dtype=jnp.float64),
                        ),
                    ),
                    jnp.asarray(drds_node, dtype=jnp.float64),
                ),
                dtype=float,
            )
        )
        for mode in modes:
            mode_runtime, kernel = mode_runtimes[mode]
            nu_results[mode].append(
                np.asarray(
                    _database_channels_to_physical(
                        kernel(r_node, nu_val, er_node, mode_runtime.database),
                        jnp.asarray(nu_val, dtype=jnp.float64),
                    ),
                    dtype=float,
                )
            )
    _print_mode_table("vary-nu-only", {k: np.asarray(v) for k, v in nu_results.items()}, np.asarray(nu_exact))

    # 3. Vary field only: use field midpoints on the nearest rho row, keep rho and nu on node values.
    er_sweep_idx = _midpoint_sweep_indices(field_index, args.field_window, er_nodes.shape[1])
    er_exact = []
    er_results = {mode: [] for mode in modes}
    for ier in er_sweep_idx:
        if ier + 1 >= er_nodes.shape[1]:
            continue
        er_val = 0.5 * float(er_nodes[rho_node_idx, ier] + er_nodes[rho_node_idx, ier + 1])
        es_val = 0.5 * float(es_nodes[rho_node_idx, ier] + es_nodes[rho_node_idx, ier + 1])
        er_exact.append(
            np.asarray(
                _exact_raw_to_physical(
                    ntx.solve_prepared_coefficient_vector(
                        prepared_node,
                        ntx.MonoenergeticCase(
                            nu_hat=jnp.asarray(nu_node, dtype=jnp.float64),
                            epsi_hat=jnp.asarray(es_val, dtype=jnp.float64),
                        ),
                    ),
                    jnp.asarray(drds_node, dtype=jnp.float64),
                ),
                dtype=float,
            )
        )
        for mode in modes:
            mode_runtime, kernel = mode_runtimes[mode]
            er_results[mode].append(
                np.asarray(
                    _database_channels_to_physical(
                        kernel(r_node, nu_node, er_val, mode_runtime.database),
                        jnp.asarray(nu_node, dtype=jnp.float64),
                    ),
                    dtype=float,
                )
            )
    _print_mode_table("vary-field-only", {k: np.asarray(v) for k, v in er_results.items()}, np.asarray(er_exact))


if __name__ == "__main__":
    main()
