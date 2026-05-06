"""Audit database local Lij assembly and state parity against exact config.

This benchmark targets the remaining ambiguity after validating the exact
runtime path:

1. Are the database and exact ambipolar configs building the same local state
   (`density`, `temperature`, `Er`, `A1`, `A2`, `A3`) at selected radii?
2. For the database config alone, does `get_Lij_matrix_local(...)` match a
   manual reconstruction from the monoenergetic database kernel and the energy
   weights?

If both are true, then any remaining mismatch must live in how the two models
map the same physical `Er` into their monoenergetic field coordinate.
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
from NEOPAX._cell_variable import get_gradient_density, get_gradient_temperature
from NEOPAX._monoenergetic_interpolators import monoenergetic_interpolation_kernel
from NEOPAX._neoclassical import _nu_over_vnew_local, get_Lij_matrix_local
from NEOPAX._orchestrator import build_runtime_context
from NEOPAX._species import (
    get_Thermodynamical_Forces_A1,
    get_Thermodynamical_Forces_A2,
    get_Thermodynamical_Forces_A3,
)
from NEOPAX._state import get_v_thermal, safe_density
from NEOPAX._transport_flux_models import (
    NTXDatabaseTransportModel,
    NTXExactLijRuntimeTransportModel,
    _collisionality_kind,
    _extract_right_constraints,
)

DEFAULT_DATABASE_CONFIG = Path("examples/Solve_Ambipolarity/ambiplarity_benchmark_Er_r_Large.toml")
DEFAULT_EXACT_CONFIG = Path("examples/Solve_Ambipolarity/ambiplarity_benchmark_Er_r_Large_exact.toml")


def _prepare_config(config_path: Path, *, device: str):
    return copy.deepcopy(NEOPAX.prepare_config(config_path, device=device))


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


def _extract_neoclassical_model(flux_model):
    return getattr(flux_model, "neoclassical_model", flux_model)


def _right_constraints(model, density, temperature):
    density_right_constraint, density_right_grad_constraint = _extract_right_constraints(
        getattr(model, "bc_density", None),
        density,
    )
    temperature_right_constraint, temperature_right_grad_constraint = _extract_right_constraints(
        getattr(model, "bc_temperature", None),
        temperature,
    )
    return (
        density_right_constraint,
        density_right_grad_constraint,
        temperature_right_constraint,
        temperature_right_grad_constraint,
    )


def _compute_forces(model, geometry, species, density, temperature, er_profile):
    (
        density_right_constraint,
        density_right_grad_constraint,
        temperature_right_constraint,
        temperature_right_grad_constraint,
    ) = _right_constraints(model, density, temperature)

    dndr_all = jax.vmap(
        lambda density_a, right_value, right_grad: get_gradient_density(
            density_a,
            geometry.r_grid,
            geometry.r_grid_half,
            geometry.dr,
            right_face_constraint=right_value,
            right_face_grad_constraint=right_grad,
        )
    )(density, density_right_constraint, density_right_grad_constraint)
    dTdr_all = jax.vmap(
        lambda temperature_a, right_value, right_grad: get_gradient_temperature(
            temperature_a,
            geometry.r_grid,
            geometry.r_grid_half,
            geometry.dr,
            right_face_constraint=right_value,
            right_face_grad_constraint=right_grad,
        )
    )(temperature, temperature_right_constraint, temperature_right_grad_constraint)

    a1 = jax.vmap(
        lambda charge, density_a, temperature_a, dndr_a, dTdr_a: get_Thermodynamical_Forces_A1(
            charge,
            density_a,
            temperature_a,
            dndr_a,
            dTdr_a,
            er_profile,
        )
    )(species.charge, density, temperature, dndr_all, dTdr_all)
    a2 = jax.vmap(get_Thermodynamical_Forces_A2)(temperature, dTdr_all)
    a3 = get_Thermodynamical_Forces_A3(er_profile)
    return dndr_all, dTdr_all, a1, a2, a3


def _database_channels_to_dij(db_coeffs_raw: jax.Array, nu_hat_a: jax.Array) -> jax.Array:
    return jnp.stack(
        (
            -10.0 ** jnp.asarray(db_coeffs_raw[:, 0], dtype=jnp.float64),
            -jnp.asarray(db_coeffs_raw[:, 1], dtype=jnp.float64),
            -jnp.asarray(db_coeffs_raw[:, 2], dtype=jnp.float64) / jnp.maximum(jnp.asarray(nu_hat_a, dtype=jnp.float64), 1.0e-30),
        ),
        axis=1,
    )


def _manual_lij_from_database(runtime, species_index: int, radius_index: int, er_value: float, density, temperature, v_thermal):
    species = runtime.species
    energy_grid = runtime.energy_grid
    geometry = runtime.geometry
    database = runtime.database
    collisionality_kind = _collisionality_kind(_extract_neoclassical_model(runtime.models.flux).collisionality_model)
    kernel = monoenergetic_interpolation_kernel(database)

    vth_a = v_thermal[species_index, radius_index]
    v_new_a = energy_grid.v_norm * vth_a
    nu_hat_a = _nu_over_vnew_local(
        species,
        species_index,
        v_new_a,
        density[:, radius_index],
        temperature[:, radius_index],
        v_thermal[:, radius_index],
        collisionality_kind,
    )
    er_vnew_a = jnp.asarray(er_value, dtype=jnp.float64) * 1.0e3 / v_new_a
    radius_value = jnp.asarray(geometry.r_grid[radius_index], dtype=jnp.float64)
    db_coeffs_raw = jax.vmap(kernel, in_axes=(None, 0, 0, None))(radius_value, nu_hat_a, er_vnew_a, database)
    dij = _database_channels_to_dij(db_coeffs_raw, nu_hat_a)

    d11_a = -dij[:, 0]
    d13_a = -dij[:, 1]
    d33_a = -jnp.true_divide(dij[:, 2], nu_hat_a)

    charge = jnp.asarray(species.charge[species_index], dtype=jnp.float64)
    mass = jnp.asarray(species.mass[species_index], dtype=jnp.float64)
    vth = jnp.asarray(vth_a, dtype=jnp.float64)
    l11_fac = (-1.0 / charge) * 1.5 * jnp.sqrt(jnp.pi) * mass**2 * vth / 4.0
    l13_fac = (-1.0 / charge) * jnp.sqrt(jnp.pi) * mass * vth / 2.0
    l33_fac = -jnp.asarray(1.0 / jnp.sqrt(jnp.pi), dtype=jnp.float64) * vth

    lij = jnp.zeros((3, 3), dtype=jnp.float64)
    lij = lij.at[0, 0].set(l11_fac * jnp.sum(energy_grid.L11_weight * energy_grid.xWeights * d11_a))
    lij = lij.at[0, 1].set(l11_fac * jnp.sum(energy_grid.L12_weight * energy_grid.xWeights * d11_a))
    lij = lij.at[1, 0].set(lij[0, 1])
    lij = lij.at[1, 1].set(l11_fac * jnp.sum(energy_grid.L22_weight * energy_grid.xWeights * d11_a))
    lij = lij.at[0, 2].set(l13_fac * jnp.sum(energy_grid.L13_weight * energy_grid.xWeights * d13_a))
    lij = lij.at[1, 2].set(l13_fac * jnp.sum(energy_grid.L23_weight * energy_grid.xWeights * d13_a))
    lij = lij.at[2, 0].set(-lij[0, 2])
    lij = lij.at[2, 1].set(-lij[1, 2])
    lij = lij.at[2, 2].set(l33_fac * jnp.sum(energy_grid.L33_weight * energy_grid.xWeights * d33_a))
    return lij


def _max_abs_rel(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    abs_max = float(np.max(np.abs(candidate - reference)))
    rel_max = float(np.max(np.abs(candidate - reference) / np.maximum(np.abs(reference), 1.0e-30)))
    return abs_max, rel_max


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--database-config", default=str(DEFAULT_DATABASE_CONFIG))
    parser.add_argument("--exact-config", default=str(DEFAULT_EXACT_CONFIG))
    parser.add_argument("--radius-indices", default="20,30,40")
    parser.add_argument("--species-index", type=int, default=1)
    parser.add_argument("--er-min", type=float, default=None)
    parser.add_argument("--er-max", type=float, default=None)
    parser.add_argument("--n-er", type=int, default=None)
    args = parser.parse_args()

    db_cfg = _prepare_config(Path(args.database_config), device=args.device)
    ex_cfg = _prepare_config(Path(args.exact_config), device=args.device)

    db_runtime, db_state = build_runtime_context(db_cfg)
    ex_runtime, ex_state = build_runtime_context(ex_cfg)
    if db_state is None or ex_state is None:
        raise RuntimeError("Both configs must build a transport state.")

    db_model = _extract_neoclassical_model(db_runtime.models.flux)
    ex_model = _extract_neoclassical_model(ex_runtime.models.flux)
    if not isinstance(db_model, NTXDatabaseTransportModel):
        raise TypeError("database-config must use NTXDatabaseTransportModel")
    if not isinstance(ex_model, NTXExactLijRuntimeTransportModel):
        raise TypeError("exact-config must use NTXExactLijRuntimeTransportModel")

    radius_indices = _parse_index_spec(args.radius_indices)
    species_index = int(args.species_index)
    amb_cfg = ex_cfg.get("ambipolarity", {})
    er_min = float(args.er_min if args.er_min is not None else amb_cfg.get("er_ambipolar_scan_min", -50.0))
    er_max = float(args.er_max if args.er_max is not None else amb_cfg.get("er_ambipolar_scan_max", 50.0))
    n_er = int(args.n_er if args.n_er is not None else amb_cfg.get("er_ambipolar_n_coarse", 300))
    er_values = np.linspace(er_min, er_max, n_er, dtype=float)

    db_density = safe_density(db_state.density)
    db_temperature = db_state.temperature
    ex_density = safe_density(ex_state.density)
    ex_temperature = ex_state.temperature
    db_vthermal = get_v_thermal(db_runtime.species.mass, db_temperature)
    rho = np.asarray(db_runtime.geometry.r_grid / db_runtime.geometry.a_b, dtype=float)

    print(f"[db-state-lij-audit] database_config={Path(args.database_config).resolve()}")
    print(f"[db-state-lij-audit] exact_config={Path(args.exact_config).resolve()}")
    print(f"[db-state-lij-audit] species_index={species_index}")
    print(f"[db-state-lij-audit] er_scan=[{er_min:.6e}, {er_max:.6e}] n_er={n_er}")

    for radius_index in radius_indices:
        density_abs, density_rel = _max_abs_rel(
            np.asarray(db_density[:, radius_index], dtype=float),
            np.asarray(ex_density[:, radius_index], dtype=float),
        )
        temperature_abs, temperature_rel = _max_abs_rel(
            np.asarray(db_temperature[:, radius_index], dtype=float),
            np.asarray(ex_temperature[:, radius_index], dtype=float),
        )
        er_state_abs, er_state_rel = _max_abs_rel(
            np.asarray([db_state.Er[radius_index]], dtype=float),
            np.asarray([ex_state.Er[radius_index]], dtype=float),
        )

        a1_abs_max = 0.0
        a1_rel_max = 0.0
        a2_abs_max = 0.0
        a2_rel_max = 0.0
        a3_abs_max = 0.0
        a3_rel_max = 0.0
        lij_abs_max = np.zeros(6, dtype=float)
        lij_rel_max = np.zeros(6, dtype=float)

        for er_value in er_values:
            db_er_profile = np.asarray(db_state.Er, dtype=float).copy()
            ex_er_profile = np.asarray(ex_state.Er, dtype=float).copy()
            db_er_profile[radius_index] = float(er_value)
            ex_er_profile[radius_index] = float(er_value)

            _, _, db_a1, db_a2, db_a3 = _compute_forces(
                db_model,
                db_runtime.geometry,
                db_runtime.species,
                db_density,
                db_temperature,
                jnp.asarray(db_er_profile, dtype=db_state.Er.dtype),
            )
            _, _, ex_a1, ex_a2, ex_a3 = _compute_forces(
                ex_model,
                ex_runtime.geometry,
                ex_runtime.species,
                ex_density,
                ex_temperature,
                jnp.asarray(ex_er_profile, dtype=ex_state.Er.dtype),
            )

            a1_abs, a1_rel = _max_abs_rel(
                np.asarray(db_a1[:, radius_index], dtype=float),
                np.asarray(ex_a1[:, radius_index], dtype=float),
            )
            a2_abs, a2_rel = _max_abs_rel(
                np.asarray(db_a2[:, radius_index], dtype=float),
                np.asarray(ex_a2[:, radius_index], dtype=float),
            )
            a3_abs, a3_rel = _max_abs_rel(
                np.asarray([db_a3[radius_index]], dtype=float),
                np.asarray([ex_a3[radius_index]], dtype=float),
            )
            a1_abs_max = max(a1_abs_max, a1_abs)
            a1_rel_max = max(a1_rel_max, a1_rel)
            a2_abs_max = max(a2_abs_max, a2_abs)
            a2_rel_max = max(a2_rel_max, a2_rel)
            a3_abs_max = max(a3_abs_max, a3_abs)
            a3_rel_max = max(a3_rel_max, a3_rel)

            lij_model = get_Lij_matrix_local(
                db_runtime.species,
                db_runtime.energy_grid,
                db_runtime.geometry,
                db_runtime.database,
                species_index,
                radius_index,
                jnp.asarray(er_value, dtype=db_state.Er.dtype),
                db_temperature,
                db_density,
                db_vthermal,
                _collisionality_kind(db_model.collisionality_model),
            )
            lij_manual = _manual_lij_from_database(
                db_runtime,
                species_index,
                radius_index,
                float(er_value),
                db_density,
                db_temperature,
                db_vthermal,
            )
            lij_model_np = np.asarray(jax.device_get(lij_model), dtype=float)
            lij_manual_np = np.asarray(jax.device_get(lij_manual), dtype=float)
            for i, (row, col) in enumerate(((0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2))):
                abs_max, rel_max = _max_abs_rel(
                    np.asarray([lij_model_np[row, col]]),
                    np.asarray([lij_manual_np[row, col]]),
                )
                lij_abs_max[i] = max(lij_abs_max[i], abs_max)
                lij_rel_max[i] = max(lij_rel_max[i], rel_max)

        print(f"\n[radius] idx={radius_index} rho={rho[radius_index]:.6e}")
        print(f"  state density: abs_max={density_abs:.6e} rel_max={density_rel:.6e}")
        print(f"  state temperature: abs_max={temperature_abs:.6e} rel_max={temperature_rel:.6e}")
        print(f"  state Er(initial): abs_max={er_state_abs:.6e} rel_max={er_state_rel:.6e}")
        print(f"  A1 over scan: abs_max={a1_abs_max:.6e} rel_max={a1_rel_max:.6e}")
        print(f"  A2 over scan: abs_max={a2_abs_max:.6e} rel_max={a2_rel_max:.6e}")
        print(f"  A3 over scan: abs_max={a3_abs_max:.6e} rel_max={a3_rel_max:.6e}")
        print("  database get_Lij_matrix_local vs manual kernel reconstruction")
        for label, abs_max, rel_max in zip(
            ("L11", "L12", "L22", "L13", "L23", "L33"),
            lij_abs_max,
            lij_rel_max,
        ):
            print(f"    {label}: abs_max={abs_max:.6e} rel_max={rel_max:.6e}")


if __name__ == "__main__":
    main()
