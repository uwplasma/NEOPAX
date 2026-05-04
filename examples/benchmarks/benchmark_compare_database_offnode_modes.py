"""Compare off-node database interpolation modes against exact NTX over several regimes.

This script extends the earlier one-case mode comparison by letting us sweep:
- multiple runtime radius indices
- multiple species indices
- multiple transport energy indices
- a configurable subset of stored field nodes

For each runtime case it:
- takes the off-node runtime `r` and `nu_hat`
- uses stored file field nodes from the nearest stored rho row as the field axis
- queries database modes with `Er / v`
- solves exact NTX with `Es / v`

It then reports max absolute and relative errors for `D11`, `D13`, `D33`.
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


def _parse_index_spec(spec: str) -> list[int]:
    items: list[int] = []
    for chunk in str(spec).split(","):
        text = chunk.strip()
        if not text:
            continue
        if ":" in text:
            parts = [piece.strip() for piece in text.split(":")]
            if len(parts) not in {2, 3}:
                raise ValueError(f"Invalid range spec '{text}'")
            start = int(parts[0])
            stop = int(parts[1])
            step = int(parts[2]) if len(parts) == 3 else 1
            items.extend(list(range(start, stop, step)))
        else:
            items.append(int(text))
    if not items:
        raise ValueError(f"Empty index spec '{spec}'")
    return items


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
    parser.add_argument("--radius-indices", default="2,10,20")
    parser.add_argument("--species-indices", default="1")
    parser.add_argument("--energy-indices", default="0,8,15")
    parser.add_argument("--field-indices", default="1,3,6,9")
    parser.add_argument(
        "--include-node-baseline",
        action="store_true",
        default=True,
        help="Also compare one nearby exact stored node for each runtime case.",
    )
    parser.add_argument("--resolution", default="25,25,63")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    args = parser.parse_args()

    resolution = _parse_resolution(args.resolution)
    radius_indices = _parse_index_spec(args.radius_indices)
    species_indices = _parse_index_spec(args.species_indices)
    energy_indices = _parse_index_spec(args.energy_indices)
    field_indices = _parse_index_spec(args.field_indices)
    modes = [item.strip() for item in str(args.modes).split(",") if item.strip()]

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

    density = safe_density(state.density)
    temperature = state.temperature
    v_thermal = get_v_thermal(runtime.species.mass, temperature)

    neo = exact_config["neoclassical"]
    grid_spec = ntx.GridSpec(
        n_theta=int(neo["ntx_exact_n_theta"]),
        n_zeta=int(neo["ntx_exact_n_zeta"]),
        n_xi=int(neo["ntx_exact_n_xi"]),
    )
    vmec_path = Path(exact_config["geometry"]["vmec_file"])
    vmec_abs = (ROOT / vmec_path).resolve() if not vmec_path.is_absolute() else vmec_path.resolve()

    mode_runtimes: dict[str, tuple[object, object]] = {}
    for mode in modes:
        mode_cfg = _prepare_config(
            db_cfg_path,
            device=args.device,
            er_init_mode=args.er_init_mode,
            flux_model="ntx_database",
            interpolation_mode=mode,
        )
        mode_runtime, _ = build_runtime_context(mode_cfg)
        mode_runtimes[mode] = (mode_runtime, monoenergetic_interpolation_kernel(mode_runtime.database))

    overall: dict[str, dict[str, tuple[float, float, tuple[int, int, int, int] | None]]] = {
        mode: {
            "D11": (0.0, 0.0, None),
            "D13": (0.0, 0.0, None),
            "D33": (0.0, 0.0, None),
        }
        for mode in modes
    }

    print(f"[offnode-modes] database_file={db_abs}")
    print(f"[offnode-modes] state_source_model={args.state_source_model}")
    print(f"[offnode-modes] resolution={resolution}")
    print(f"[offnode-modes] modes={modes}")
    print()

    for radius_index in radius_indices:
        rho_runtime_surface = float(runtime.geometry.rho_grid[radius_index])
        r_runtime = float(runtime.geometry.r_grid[radius_index])
        rho_node_idx = int(np.argmin(np.abs(rho_nodes - rho_runtime_surface)))
        rho_node = float(rho_nodes[rho_node_idx])
        r_node = float(runtime.geometry.a_b * rho_node)
        er_over_v_axis = np.asarray(er_nodes[rho_node_idx, field_indices], dtype=float)
        es_over_v_axis = np.asarray(es_nodes[rho_node_idx, field_indices], dtype=float)

        surface = ntx.surface_from_vmec_jax_vmec_wout_file(str(vmec_abs), s=float(rho_runtime_surface**2))
        prepared = ntx.prepare_monoenergetic_system(surface, grid_spec)
        drds_value = jnp.asarray(support.center_channels.drds[radius_index], dtype=jnp.float64)

        surface_node = ntx.surface_from_vmec_jax_vmec_wout_file(str(vmec_abs), s=float(rho_node**2))
        prepared_node = ntx.prepare_monoenergetic_system(surface_node, grid_spec)

        for species_index in species_indices:
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

            for energy_index in energy_indices:
                nu_runtime = float(nu_hat_a[energy_index])

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

                print(
                    f"[case] radius_index={radius_index} rho_runtime_surface={rho_runtime_surface:.6e} "
                    f"r_runtime={r_runtime:.6e} rho_field_row={rho_node:.6e} "
                    f"species_index={species_index} energy_index={energy_index} nu_runtime={nu_runtime:.6e}"
                )
                print(
                    f"       Er/v range=[{float(np.min(er_over_v_axis)):.6e}, {float(np.max(er_over_v_axis)):.6e}] "
                    f"Es/v range=[{float(np.min(es_over_v_axis)):.6e}, {float(np.max(es_over_v_axis)):.6e}]"
                )
                if args.include_node_baseline:
                    nu_node_idx = int(np.argmin(np.abs(np.asarray(runtime.database.nu_log, dtype=float) - np.log10(nu_runtime))))
                    nu_node = float(10.0 ** np.asarray(runtime.database.nu_log[nu_node_idx], dtype=float))
                    drds_node = float(es_nodes[rho_node_idx, field_indices[0]] / er_nodes[rho_node_idx, field_indices[0]])
                    print(
                        f"       node-baseline: rho_node_idx={rho_node_idx} rho_node={rho_node:.6e} "
                        f"r_node={r_node:.6e} nu_node_idx={nu_node_idx} nu_node={nu_node:.6e}"
                    )
                    for mode in modes:
                        mode_runtime, kernel = mode_runtimes[mode]
                        baseline_abs = []
                        baseline_rel = []
                        for field_idx in field_indices:
                            er_node = float(er_nodes[rho_node_idx, field_idx])
                            es_node = float(es_nodes[rho_node_idx, field_idx])
                            db_node = np.asarray(
                                _database_channels_to_physical(
                                    kernel(r_node, nu_node, er_node, mode_runtime.database),
                                    jnp.asarray(nu_node, dtype=jnp.float64),
                                ),
                                dtype=float,
                            )
                            exact_node = np.asarray(
                                _exact_raw_to_physical(
                                    ntx.solve_prepared_coefficient_vector(
                                        prepared_node,
                                        ntx.MonoenergeticCase(
                                            nu_hat=jnp.asarray(nu_node, dtype=jnp.float64),
                                            epsi_hat=jnp.asarray(es_node, dtype=jnp.float64),
                                        ),
                                    ),
                                    jnp.asarray(drds_node, dtype=jnp.float64),
                                ),
                                dtype=float,
                            )
                            abs_delta_node = np.abs(db_node - exact_node)
                            rel_delta_node = abs_delta_node / np.maximum(np.abs(exact_node), 1.0e-30)
                            baseline_abs.append(abs_delta_node)
                            baseline_rel.append(rel_delta_node)
                        baseline_abs = np.asarray(baseline_abs)
                        baseline_rel = np.asarray(baseline_rel)
                        print(f"  [{mode}] node-baseline")
                        print(f"    {'quantity':<8} {'abs_max':>14} {'rel_max':>14}")
                        for idx, label in enumerate(("D11", "D13", "D33")):
                            print(
                                f"    {label:<8} {float(np.max(baseline_abs[:, idx])):14.6e} "
                                f"{float(np.max(baseline_rel[:, idx])):14.6e}"
                            )
                for mode in modes:
                    mode_runtime, kernel = mode_runtimes[mode]
                    vals = []
                    for er_over_v_value in er_over_v_axis:
                        vals.append(
                            np.asarray(
                                _database_channels_to_physical(
                                    kernel(r_runtime, nu_runtime, er_over_v_value, mode_runtime.database),
                                    jnp.asarray(nu_runtime, dtype=jnp.float64),
                                ),
                                dtype=float,
                            )
                        )
                    vals = np.asarray(vals)
                    abs_delta = np.abs(vals - exact_vals)
                    rel_delta = abs_delta / np.maximum(np.abs(exact_vals), 1.0e-30)

                    print(f"  [{mode}]")
                    print(f"    {'quantity':<8} {'abs_max':>14} {'rel_max':>14}")
                    for idx, label in enumerate(("D11", "D13", "D33")):
                        abs_max = float(np.max(abs_delta[:, idx]))
                        rel_max = float(np.max(rel_delta[:, idx]))
                        print(f"    {label:<8} {abs_max:14.6e} {rel_max:14.6e}")
                        prev_abs, _prev_rel, _prev_meta = overall[mode][label]
                        if abs_max > prev_abs:
                            overall[mode][label] = (
                                abs_max,
                                rel_max,
                                (radius_index, species_index, energy_index, rho_node_idx),
                            )
                print()

    print("[offnode-modes] overall maxima")
    for mode in modes:
        print(f"  [{mode}]")
        for label in ("D11", "D13", "D33"):
            abs_max, rel_max, meta = overall[mode][label]
            print(f"    {label}: abs_max={abs_max:.6e} rel_max={rel_max:.6e} at case={meta}")


if __name__ == "__main__":
    main()
