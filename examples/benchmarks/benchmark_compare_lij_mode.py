"""Directly compare database and exact-runtime NTX Lij on a shared state.

This benchmark is meant to validate the neoclassical transport matrices
themselves before looking at solver behavior. It:

- builds one shared initialized state
- evaluates database Lij on centers and faces
- evaluates exact-runtime Lij on the same centers and faces
- reports max absolute/relative deltas
- writes overlay plots for the six independent Lij entries
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import NEOPAX
from NEOPAX._boundary_conditions import build_boundary_condition_model
from NEOPAX._neoclassical import (
    _collisionality_kind,
    get_Lij_matrix_at_radius,
    get_Neoclassical_Fluxes,
)
from NEOPAX._orchestrator import (
    _build_flux_model,
    _normalized_boundary_cfg_for_transport,
    build_runtime_context,
)
from NEOPAX._source_models import build_source_models_from_config
from NEOPAX._state import get_v_thermal, safe_density
from NEOPAX._transport_flux_models import (
    NTXExactLijRuntimeTransportModel,
    build_face_transport_state,
)

DEFAULT_DATABASE_CONFIG = Path("examples/benchmarks/Calculate_Fluxes_noHe_ntx_database_benchmark.toml")
DEFAULT_EXACT_CONFIG = Path("examples/benchmarks/Calculate_Fluxes_noHe_ntx_exact_lij_runtime_benchmark.toml")
DEFAULT_RESOLUTIONS = ["5,21,32"]
INDEPENDENT_LIJ_ENTRIES = [
    ("L11", 0, 0),
    ("L12", 0, 1),
    ("L22", 1, 1),
    ("L13", 0, 2),
    ("L23", 1, 2),
    ("L33", 2, 2),
]
ENTRY_MAP = {name: (name, row, col) for name, row, col in INDEPENDENT_LIJ_ENTRIES}


def _parse_resolution(spec: str) -> tuple[int, int, int]:
    parts = [piece.strip() for piece in str(spec).split(",")]
    if len(parts) != 3:
        raise ValueError(f"Resolution '{spec}' must be in 'n_theta,n_zeta,n_xi' format.")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _species_names(n_species: int):
    default = ["e", "D", "T"]
    if n_species <= len(default):
        return default[:n_species]
    return default + [f"s{i}" for i in range(len(default), n_species)]


def _max_abs_rel_delta(reference: jax.Array, candidate: jax.Array) -> tuple[float, float]:
    diff = jnp.abs(candidate - reference)
    abs_max = float(jnp.max(diff))
    rel = diff / jnp.maximum(jnp.abs(reference), 1.0e-30)
    rel_max = float(jnp.max(rel))
    return abs_max, rel_max


def _rho_center_face(geometry):
    rho_center = getattr(geometry, "rho_grid", None)
    if rho_center is None:
        rho_center = jnp.asarray(geometry.r_grid, dtype=jnp.float64) / jnp.asarray(geometry.a_b, dtype=jnp.float64)
    rho_face = jnp.asarray(geometry.r_grid_half, dtype=jnp.float64) / jnp.asarray(geometry.a_b, dtype=jnp.float64)
    return np.asarray(rho_center), np.asarray(rho_face)


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


def _build_neoclassical_model(
    config: dict,
    *,
    species,
    energy_grid,
    geometry,
    database,
):
    source_models = build_source_models_from_config(config, species)
    model = _build_flux_model(
        config,
        species,
        energy_grid,
        geometry,
        database,
        source_models=source_models,
    )
    neo_model = _extract_neoclassical_model(model)
    if isinstance(neo_model, NTXExactLijRuntimeTransportModel):
        neo_model = neo_model.with_static_support()
    return neo_model


def _build_boundary_models(config: dict, geometry, species):
    boundary_cfg = _normalized_boundary_cfg_for_transport(config.get("boundary", {}))
    bc = {}
    for key in ("density", "temperature", "Er"):
        if key in boundary_cfg:
            bc[key] = build_boundary_condition_model(
                boundary_cfg[key],
                geometry.dr,
                species_names=species.names if key in {"density", "temperature"} else None,
            )
    return bc


def _database_lij_faces(model, face_state):
    geometry = model.geometry
    species = model.species
    density_faces = safe_density(face_state.density)
    v_thermal_faces = get_v_thermal(species.mass, face_state.temperature)
    collisionality_kind = _collisionality_kind(model.collisionality_model)

    return jax.vmap(
        lambda species_index: jax.vmap(
            lambda radius_value, er_value, temperature_local, density_local, vthermal_local: get_Lij_matrix_at_radius(
                species,
                model.energy_grid,
                geometry,
                model.database,
                species_index,
                radius_value,
                er_value,
                temperature_local,
                density_local,
                vthermal_local,
                collisionality_kind,
            ),
            in_axes=(0, 0, 1, 1, 1),
        )(
            geometry.r_grid_half,
            face_state.Er,
            face_state.temperature,
            density_faces,
            v_thermal_faces,
        )
    )(species.species_indices)


def _plot_lij_quantity(output_dir: Path, rho, cases: list[dict], quantity: tuple[str, int, int], location: str):
    name, row, col = quantity
    ref = jnp.asarray(cases[0][f"{location}_lij"])
    species_names = _species_names(int(ref.shape[0]))
    fig, axes = plt.subplots(int(ref.shape[0]), 1, figsize=(10, 3 * int(ref.shape[0])), sharex=True)
    if int(ref.shape[0]) == 1:
        axes = [axes]
    styles = [
        ("solid", 2.4),
        ("--", 2.0),
        (":", 2.0),
        ("-.", 2.0),
        ((0, (5, 1)), 2.0),
        ((0, (3, 1, 1, 1)), 2.0),
    ]
    for i, ax in enumerate(axes):
        for case_idx, case in enumerate(cases):
            arr = jnp.asarray(case[f"{location}_lij"])
            linestyle, linewidth = styles[case_idx % len(styles)]
            ax.plot(rho, arr[i, :, row, col], linestyle=linestyle, linewidth=linewidth, label=case["label"])
        ax.set_ylabel(f"{name}[{species_names[i]}]")
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("rho")
    fig.tight_layout()
    out = output_dir / f"compare_{location}_{name}.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--database-config", default=str(DEFAULT_DATABASE_CONFIG))
    parser.add_argument("--exact-config", default=str(DEFAULT_EXACT_CONFIG))
    parser.add_argument(
        "--er-init-mode",
        default="keep",
        choices=["keep", "analytical", "ambipolar_min_entropy"],
        help="Override profiles.er_initialization_mode before building the shared state.",
    )
    parser.add_argument(
        "--state-source-model",
        default="database",
        choices=["database", "exact"],
        help="Which neoclassical model initializes the shared state before both Lij evaluations.",
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        default=DEFAULT_RESOLUTIONS,
        help="Exact-runtime resolution sweep entries in 'n_theta,n_zeta,n_xi' format.",
    )
    parser.add_argument(
        "--entries",
        nargs="+",
        default=["all"],
        choices=["all", "L11", "L12", "L22", "L13", "L23", "L33"],
        help="Which independent Lij entries to plot. Use 'all' or a subset like 'L11'.",
    )
    args = parser.parse_args()

    db_cfg_path = Path(args.database_config)
    ex_cfg_path = Path(args.exact_config)
    resolutions = [_parse_resolution(spec) for spec in args.resolutions]

    state_cfg_path = db_cfg_path if args.state_source_model == "database" else ex_cfg_path
    state_flux_model = "ntx_database" if args.state_source_model == "database" else "ntx_exact_lij_runtime"
    shared_state_config = _prepare_config(
        state_cfg_path,
        device=args.device,
        er_init_mode=args.er_init_mode,
        flux_model=state_flux_model,
    )
    runtime, state = build_runtime_context(shared_state_config)
    density = safe_density(state.density)

    bc = _build_boundary_models(shared_state_config, runtime.geometry, runtime.species)
    face_state = build_face_transport_state(
        state,
        runtime.geometry,
        bc_density=bc.get("density"),
        bc_temperature=bc.get("temperature"),
        bc_er=bc.get("Er"),
    )

    print(f"[lij-compare] device={args.device}")
    print(f"[lij-compare] database_config={db_cfg_path}")
    print(f"[lij-compare] exact_config={ex_cfg_path}")
    print(f"[lij-compare] er_init_mode={args.er_init_mode}")
    print(f"[lij-compare] state_source_model={args.state_source_model}")
    print(f"[lij-compare] resolutions={resolutions}")
    print(f"[lij-compare] entries={args.entries}")

    selected_entries = (
        INDEPENDENT_LIJ_ENTRIES
        if "all" in args.entries
        else [ENTRY_MAP[name] for name in args.entries]
    )

    rho_center, rho_face = _rho_center_face(runtime.geometry)
    output_dir = Path("outputs/benchmark_lij_compare")
    output_dir.mkdir(parents=True, exist_ok=True)

    db_model = _build_neoclassical_model(
        _prepare_config(
            db_cfg_path,
            device=args.device,
            er_init_mode=args.er_init_mode,
            flux_model="ntx_database",
        ),
        species=runtime.species,
        energy_grid=runtime.energy_grid,
        geometry=runtime.geometry,
        database=runtime.database,
    )

    t0 = time.perf_counter()
    db_center_lij, _, _, _ = get_Neoclassical_Fluxes(
        runtime.species,
        runtime.energy_grid,
        runtime.geometry,
        runtime.database,
        state.Er,
        state.temperature,
        density,
        collisionality_model=db_model.collisionality_model,
    )
    db_face_lij = _database_lij_faces(db_model, face_state)
    jax.block_until_ready(db_center_lij)
    jax.block_until_ready(db_face_lij)
    db_wall_s = time.perf_counter() - t0

    cases = [
        {
            "label": "database",
            "center_lij": db_center_lij,
            "face_lij": db_face_lij,
            "wall_s": db_wall_s,
            "center_abs": 0.0,
            "center_rel": 0.0,
            "face_abs": 0.0,
            "face_rel": 0.0,
        }
    ]

    for resolution in resolutions:
        label = f"exact:({resolution[0]},{resolution[1]},{resolution[2]})"
        exact_model = _build_neoclassical_model(
            _prepare_config(
                ex_cfg_path,
                device=args.device,
                er_init_mode=args.er_init_mode,
                flux_model="ntx_exact_lij_runtime",
                resolution=resolution,
            ),
            species=runtime.species,
            energy_grid=runtime.energy_grid,
            geometry=runtime.geometry,
            database=runtime.database,
        )

        t0 = time.perf_counter()
        ex_center_lij = exact_model._lij_center(state.Er, state.temperature, density)
        ex_face_lij = exact_model._lij_faces(face_state.Er, face_state.temperature, safe_density(face_state.density))
        jax.block_until_ready(ex_center_lij)
        jax.block_until_ready(ex_face_lij)
        wall_s = time.perf_counter() - t0

        center_abs, center_rel = _max_abs_rel_delta(db_center_lij, ex_center_lij)
        face_abs, face_rel = _max_abs_rel_delta(db_face_lij, ex_face_lij)
        cases.append(
            {
                "label": label,
                "center_lij": ex_center_lij,
                "face_lij": ex_face_lij,
                "wall_s": wall_s,
                "center_abs": center_abs,
                "center_rel": center_rel,
                "face_abs": face_abs,
                "face_rel": face_rel,
            }
        )

    center_plots = {}
    face_plots = {}
    for quantity in selected_entries:
        center_plots[quantity[0]] = _plot_lij_quantity(output_dir, rho_center, cases, quantity, "center")
        face_plots[quantity[0]] = _plot_lij_quantity(output_dir, rho_face, cases, quantity, "face")

    print()
    print("case                         wall_s    center_abs_max    center_rel_max      face_abs_max      face_rel_max")
    print("-----------------------------------------------------------------------------------------------------------")
    for case in cases:
        print(
            f"{case['label']:<26}"
            f"{case['wall_s']:>8.3f}"
            f"{case['center_abs']:>18.6e}"
            f"{case['center_rel']:>18.6e}"
            f"{case['face_abs']:>18.6e}"
            f"{case['face_rel']:>18.6e}"
        )

    print()
    print(f"[lij-compare] output_dir={output_dir}")
    for key, path in center_plots.items():
        print(f"[lij-compare] center_plot_{key}={path}")
    for key, path in face_plots.items():
        print(f"[lij-compare] face_plot_{key}={path}")


if __name__ == "__main__":
    main()
