"""Dump the local scan inputs used in ambipolar root finding.

This utility writes one CSV row per:

- radius index
- Er sample
- species
- energy-grid point x

and reports the quantities that enter the local ambipolar neoclassical scan:

- raw Er / v
- Es / v with Es = Er * dr/ds
- the active field-over-v value actually used by the configured model
- nu / v
- ln(nu / v)
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import NEOPAX
from NEOPAX._neoclassical import _collisionality_kind, _nu_over_vnew
from NEOPAX._orchestrator import build_runtime_context
from NEOPAX._state import get_v_thermal, safe_density
from NEOPAX._transport_flux_models import NTXDatabaseTransportModel, NTXExactLijRuntimeTransportModel

DEFAULT_CONFIG = Path("examples/Solve_Ambipolarity/ambiplarity_benchmark_Er_r_Large.toml")


def _prepare_config(config_path: Path, *, device: str):
    return copy.deepcopy(NEOPAX.prepare_config(config_path, device=device))


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
            pieces = [piece.strip() for piece in part.split(":")]
            if len(pieces) not in {2, 3}:
                raise ValueError(f"Invalid range spec '{part}'")
            start = int(pieces[0])
            stop = int(pieces[1])
            step = int(pieces[2]) if len(pieces) == 3 else 1
            items.extend(list(range(start, stop, step)))
        else:
            items.append(int(part))
    if not items:
        raise ValueError(f"Empty radius index spec '{spec}'")
    return items


def _extract_neoclassical_model(flux_model):
    return getattr(flux_model, "neoclassical_model", flux_model)


def _active_scan_inputs(
    model,
    *,
    state,
    radius_index: int,
    species_index: int,
    er_value: float,
    density,
    temperature,
    v_thermal,
):
    geometry = model.geometry
    collisionality_kind = _collisionality_kind(getattr(model, "collisionality_model", None))
    temperature_local = temperature[:, radius_index]
    density_local = density[:, radius_index]
    vthermal_local = v_thermal[:, radius_index]
    vth_a = jnp.asarray(vthermal_local[species_index], dtype=jnp.float64)
    v_new_a = jnp.asarray(model.energy_grid.v_norm, dtype=jnp.float64) * vth_a

    rho_value = float(np.asarray(geometry.rho_grid[radius_index], dtype=float))
    drds_value = float(geometry.a_b / max(2.0 * rho_value, 1.0e-30))
    er_over_v = jnp.asarray(er_value, dtype=jnp.float64) * 1.0e3 / v_new_a
    es_over_v = er_over_v * jnp.asarray(drds_value, dtype=jnp.float64)

    if isinstance(model, NTXExactLijRuntimeTransportModel):
        nu_hat_a, active_field_over_v, _ = model._local_scan_inputs(
            drds_value=jnp.asarray(drds_value, dtype=jnp.float64),
            species_index=species_index,
            er_value=jnp.asarray(er_value, dtype=jnp.float64),
            temperature_local=temperature_local,
            density_local=density_local,
            vthermal_local=vthermal_local,
            collisionality_kind=collisionality_kind,
        )
        active_field_kind = "Es_over_v"
    elif isinstance(model, NTXDatabaseTransportModel):
        nu_hat_a = _nu_over_vnew(
            model.species,
            species_index,
            v_new_a,
            radius_index,
            density,
            temperature,
            v_thermal,
            collisionality_kind,
        )
        active_field_over_v = er_over_v
        active_field_kind = "Er_over_v"
    else:
        raise TypeError(f"Unsupported neoclassical model type {type(model).__name__}")

    return {
        "rho": rho_value,
        "r": float(np.asarray(geometry.r_grid[radius_index], dtype=float)),
        "drds": drds_value,
        "vth": float(np.asarray(vth_a, dtype=float)),
        "v_new": np.asarray(v_new_a, dtype=float),
        "er_over_v": np.asarray(er_over_v, dtype=float),
        "es_over_v": np.asarray(es_over_v, dtype=float),
        "active_field_over_v": np.asarray(active_field_over_v, dtype=float),
        "active_field_kind": active_field_kind,
        "nu_over_v": np.asarray(nu_hat_a, dtype=float),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--radius-indices", default="all")
    parser.add_argument("--er-min", type=float, default=0.0)
    parser.add_argument("--er-max", type=float, default=50.0)
    parser.add_argument("--er-count", type=int, default=51)
    parser.add_argument(
        "--output-csv",
        default="outputs/benchmark_ambipolar_root_scan_inputs/ambipolar_root_scan_inputs.csv",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Also print all rows to stdout.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = _prepare_config(config_path, device=args.device)
    runtime, state = build_runtime_context(cfg)
    if state is None:
        raise RuntimeError("This config did not build a transport state.")

    model = _extract_neoclassical_model(runtime.models.flux)
    if not isinstance(model, (NTXDatabaseTransportModel, NTXExactLijRuntimeTransportModel)):
        raise TypeError(
            f"Expected NTX database or exact runtime model, got {type(model).__name__}."
        )

    density = safe_density(state.density)
    temperature = state.temperature
    v_thermal = get_v_thermal(runtime.species.mass, temperature)
    n_r = int(runtime.geometry.r_grid.shape[0])
    radius_indices = _parse_index_spec(args.radius_indices, n_r)
    er_values = np.linspace(float(args.er_min), float(args.er_max), int(args.er_count), dtype=float)

    species_names = [str(name) for name in getattr(runtime.species, "names", ())]
    if len(species_names) != int(runtime.species.number_species):
        species_names = [f"species_{idx}" for idx in range(int(runtime.species.number_species))]
    x_values = np.asarray(runtime.energy_grid.x, dtype=float)
    v_norm = np.asarray(runtime.energy_grid.v_norm, dtype=float)

    output_path = ROOT / args.output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)

    field_kind = "Es_over_v" if isinstance(model, NTXExactLijRuntimeTransportModel) else "Er_over_v"
    print(f"[ambipolar-root-inputs] config={config_path.resolve() if config_path.is_absolute() else (ROOT / config_path).resolve()}")
    print(f"[ambipolar-root-inputs] model={type(model).__name__}")
    print(f"[ambipolar-root-inputs] active_field_kind={field_kind}")
    print(f"[ambipolar-root-inputs] radii={radius_indices}")
    print(f"[ambipolar-root-inputs] Er_range=[{float(args.er_min):.6e}, {float(args.er_max):.6e}] n_er={int(args.er_count)}")
    print(f"[ambipolar-root-inputs] n_x={len(x_values)}")

    header = [
        "radius_index",
        "rho",
        "r",
        "drds",
        "Er",
        "species_index",
        "species_name",
        "x_index",
        "x",
        "v_norm",
        "vth",
        "v_new",
        "Er_over_v",
        "Es_over_v",
        "active_field_kind",
        "active_field_over_v",
        "nu_over_v",
        "ln_nu_over_v",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)

        if args.stdout:
            print(",".join(header))

        for radius_index in radius_indices:
            for er_value in er_values:
                for species_index, species_name in enumerate(species_names):
                    scan = _active_scan_inputs(
                        model,
                        state=state,
                        radius_index=radius_index,
                        species_index=species_index,
                        er_value=float(er_value),
                        density=density,
                        temperature=temperature,
                        v_thermal=v_thermal,
                    )
                    ln_nu_over_v = np.log(np.maximum(scan["nu_over_v"], 1.0e-300))
                    for x_index, (x_value, v_norm_value) in enumerate(zip(x_values, v_norm, strict=True)):
                        row = [
                            int(radius_index),
                            float(scan["rho"]),
                            float(scan["r"]),
                            float(scan["drds"]),
                            float(er_value),
                            int(species_index),
                            species_name,
                            int(x_index),
                            float(x_value),
                            float(v_norm_value),
                            float(scan["vth"]),
                            float(scan["v_new"][x_index]),
                            float(scan["er_over_v"][x_index]),
                            float(scan["es_over_v"][x_index]),
                            scan["active_field_kind"],
                            float(scan["active_field_over_v"][x_index]),
                            float(scan["nu_over_v"][x_index]),
                            float(ln_nu_over_v[x_index]),
                        ]
                        writer.writerow(row)
                        if args.stdout:
                            print(",".join(str(value) for value in row))

    print(f"[ambipolar-root-inputs] wrote={output_path}")


if __name__ == "__main__":
    main()
