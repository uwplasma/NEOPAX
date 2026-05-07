"""Audit per-energy D11 contributions to L11, L12, and Gamma.

This benchmark compares the database NTX path and the exact-runtime NTX path
at a single radius/species/Er point and prints the per-energy contributions that
build:

- L11 from D11
- L12 from D11
- Gamma = -n (L11 * A1 + L12 * A2)

It is meant to test whether a Gamma mismatch comes from:

- different monoenergetic D11 values
- quadrature weighting over x
- or a bug in the D11 -> L11/L12 -> Gamma assembly
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
from NEOPAX._database import D11_POSITIVE_FLOOR
from NEOPAX._monoenergetic_interpolators import monoenergetic_interpolation_kernel
from NEOPAX._neoclassical import _collisionality_kind, _nu_over_vnew, _nu_over_vnew_local
from NEOPAX._orchestrator import build_runtime_context
from NEOPAX._species import get_Thermodynamical_Forces_A1, get_Thermodynamical_Forces_A2
from NEOPAX._state import get_v_thermal, safe_density
from NEOPAX._transport_flux_models import (
    NTXDatabaseTransportModel,
    NTXExactLijRuntimeTransportModel,
    _extract_right_constraints,
)

DEFAULT_DATABASE_CONFIG = Path("examples/Solve_Ambipolarity/ambiplarity_benchmark_Er_r_Large.toml")
DEFAULT_EXACT_CONFIG = Path("examples/Solve_Ambipolarity/ambiplarity_benchmark_Er_r_Large_exact.toml")


def _prepare_config(config_path: Path, *, device: str):
    return copy.deepcopy(NEOPAX.prepare_config(config_path, device=device))


def _extract_neoclassical_model(flux_model):
    return getattr(flux_model, "neoclassical_model", flux_model)


def _species_names(species) -> list[str]:
    names = list(getattr(species, "names", ()))
    if len(names) != int(species.number_species):
        names = [f"species_{idx}" for idx in range(int(species.number_species))]
    return [str(name) for name in names]


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


def _compute_forces(model, geometry, species, density, temperature, er_profile, radius_index: int, species_index: int):
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

    a1_all = jax.vmap(
        lambda charge, density_a, temperature_a, dndr_a, dTdr_a: get_Thermodynamical_Forces_A1(
            charge,
            density_a,
            temperature_a,
            dndr_a,
            dTdr_a,
            er_profile,
        )
    )(species.charge, density, temperature, dndr_all, dTdr_all)
    a2_all = jax.vmap(get_Thermodynamical_Forces_A2)(temperature, dTdr_all)

    return (
        float(np.asarray(a1_all[species_index, radius_index], dtype=float)),
        float(np.asarray(a2_all[species_index, radius_index], dtype=float)),
    )


def _database_d11_physical(model, state, *, radius_index: int, species_index: int, er_value: float, density, temperature, v_thermal):
    kernel = monoenergetic_interpolation_kernel(model.database)
    vth_a = jnp.asarray(v_thermal[species_index, radius_index], dtype=jnp.float64)
    v_new_a = jnp.asarray(model.energy_grid.v_norm, dtype=jnp.float64) * vth_a
    nu_over_v = _nu_over_vnew(
        model.species,
        species_index,
        v_new_a,
        radius_index,
        density,
        temperature,
        v_thermal,
        _collisionality_kind(model.collisionality_model),
    )
    er_over_v = jnp.asarray(er_value, dtype=jnp.float64) * 1.0e3 / v_new_a
    r_value = jnp.asarray(model.geometry.r_grid[radius_index], dtype=jnp.float64)
    coeffs = jax.vmap(kernel, in_axes=(None, 0, 0, None))(r_value, nu_over_v, er_over_v, model.database)
    d11_physical = 10.0 ** jnp.asarray(coeffs[:, 0], dtype=jnp.float64)
    d11_physical = jnp.maximum(d11_physical, jnp.asarray(D11_POSITIVE_FLOOR, dtype=jnp.float64))
    return (
        np.asarray(d11_physical, dtype=float),
        np.asarray(nu_over_v, dtype=float),
        np.asarray(er_over_v, dtype=float),
        float(np.asarray(vth_a, dtype=float)),
    )


def _exact_d11_physical(model, *, radius_index: int, species_index: int, er_value: float, density, temperature, v_thermal):
    support = model._static_support()
    prepared = jax.tree_util.tree_map(
        lambda arr: jax.lax.dynamic_index_in_dim(arr, radius_index, axis=0, keepdims=False),
        support.center_prepared,
    )
    drds_value = jnp.asarray(support.center_channels.drds[radius_index], dtype=jnp.float64)
    temperature_local = temperature[:, radius_index]
    density_local = density[:, radius_index]
    vthermal_local = v_thermal[:, radius_index]

    nu_over_v, epsi_hat_a, vth_a = model._local_scan_inputs(
        drds_value=drds_value,
        species_index=species_index,
        er_value=jnp.asarray(er_value, dtype=jnp.float64),
        temperature_local=temperature_local,
        density_local=density_local,
        vthermal_local=vthermal_local,
        collisionality_kind=_collisionality_kind(model.collisionality_model),
    )
    coeffs = model._solve_coefficient_scan_prepared(prepared, nu_over_v, epsi_hat_a)
    d11_physical = jnp.asarray(coeffs[:, 0], dtype=jnp.float64) * drds_value**2
    d11_physical = jnp.maximum(d11_physical, jnp.asarray(D11_POSITIVE_FLOOR, dtype=jnp.float64))
    return (
        np.asarray(d11_physical, dtype=float),
        np.asarray(nu_over_v, dtype=float),
        np.asarray(epsi_hat_a, dtype=float),
        float(np.asarray(vth_a, dtype=float)),
    )


def _contribution_table(energy_grid, species, species_index: int, vth_a: float, d11_physical: np.ndarray, a1: float, a2: float):
    charge = float(np.asarray(species.charge[species_index], dtype=float))
    mass = float(np.asarray(species.mass[species_index], dtype=float))
    l11_fac = -1.0 / np.sqrt(np.pi) * (mass / charge) ** 2 * vth_a**3

    weighted_l11 = np.asarray(energy_grid.L11_weight, dtype=float) * np.asarray(energy_grid.xWeights, dtype=float)
    weighted_l12 = np.asarray(energy_grid.L12_weight, dtype=float) * np.asarray(energy_grid.xWeights, dtype=float)
    d11_a = -np.asarray(d11_physical, dtype=float)

    m11_terms = weighted_l11 * d11_a
    m12_terms = weighted_l12 * d11_a
    l11_terms = l11_fac * m11_terms
    l12_terms = l11_fac * m12_terms

    return {
        "m11_terms": m11_terms,
        "m12_terms": m12_terms,
        "l11_terms": l11_terms,
        "l12_terms": l12_terms,
        "gamma_l11_terms": l11_terms * a1,
        "gamma_l12_terms": l12_terms * a2,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--database-config", default=str(DEFAULT_DATABASE_CONFIG))
    parser.add_argument("--exact-config", default=str(DEFAULT_EXACT_CONFIG))
    parser.add_argument("--radius-index", type=int, default=38)
    parser.add_argument("--species-index", type=int, default=0)
    parser.add_argument("--er-value", type=float, default=None)
    args = parser.parse_args()

    db_runtime, db_state = build_runtime_context(_prepare_config(Path(args.database_config), device=args.device))
    ex_runtime, ex_state = build_runtime_context(_prepare_config(Path(args.exact_config), device=args.device))
    if db_state is None or ex_state is None:
        raise RuntimeError("Both configs must build a transport state.")

    db_model = _extract_neoclassical_model(db_runtime.models.flux)
    ex_model = _extract_neoclassical_model(ex_runtime.models.flux)
    if not isinstance(db_model, NTXDatabaseTransportModel):
        raise TypeError(f"Database config must produce NTXDatabaseTransportModel, got {type(db_model).__name__}")
    if not isinstance(ex_model, NTXExactLijRuntimeTransportModel):
        raise TypeError(f"Exact config must produce NTXExactLijRuntimeTransportModel, got {type(ex_model).__name__}")
    ex_model = ex_model.with_static_support()

    radius_index = int(args.radius_index)
    species_index = int(args.species_index)
    species_name = _species_names(db_runtime.species)[species_index]
    er_value = float(np.asarray(db_state.Er[radius_index], dtype=float)) if args.er_value is None else float(args.er_value)

    density_db = safe_density(db_state.density)
    temperature_db = db_state.temperature
    v_thermal_db = get_v_thermal(db_runtime.species.mass, temperature_db)
    density_ex = safe_density(ex_state.density)
    temperature_ex = ex_state.temperature
    v_thermal_ex = get_v_thermal(ex_runtime.species.mass, temperature_ex)

    er_profile_db = np.asarray(db_state.Er, dtype=float).copy()
    er_profile_db[radius_index] = er_value
    er_profile_ex = np.asarray(ex_state.Er, dtype=float).copy()
    er_profile_ex[radius_index] = er_value

    a1_db, a2_db = _compute_forces(
        db_model,
        db_runtime.geometry,
        db_runtime.species,
        density_db,
        temperature_db,
        jnp.asarray(er_profile_db, dtype=db_state.Er.dtype),
        radius_index,
        species_index,
    )
    a1_ex, a2_ex = _compute_forces(
        ex_model,
        ex_runtime.geometry,
        ex_runtime.species,
        density_ex,
        temperature_ex,
        jnp.asarray(er_profile_ex, dtype=ex_state.Er.dtype),
        radius_index,
        species_index,
    )

    d11_db, nu_db, field_db, vth_db = _database_d11_physical(
        db_model,
        db_state,
        radius_index=radius_index,
        species_index=species_index,
        er_value=er_value,
        density=density_db,
        temperature=temperature_db,
        v_thermal=v_thermal_db,
    )
    d11_ex, nu_ex, field_ex, vth_ex = _exact_d11_physical(
        ex_model,
        radius_index=radius_index,
        species_index=species_index,
        er_value=er_value,
        density=density_ex,
        temperature=temperature_ex,
        v_thermal=v_thermal_ex,
    )

    contrib_db = _contribution_table(db_runtime.energy_grid, db_runtime.species, species_index, vth_db, d11_db, a1_db, a2_db)
    contrib_ex = _contribution_table(ex_runtime.energy_grid, ex_runtime.species, species_index, vth_ex, d11_ex, a1_ex, a2_ex)

    x_values = np.asarray(db_runtime.energy_grid.x, dtype=float)
    print(f"[gamma-d11-audit] radius_index={radius_index} rho={float(np.asarray(db_runtime.geometry.rho_grid[radius_index], dtype=float)):.6e}")
    print(f"[gamma-d11-audit] species_index={species_index} species={species_name}")
    print(f"[gamma-d11-audit] Er={er_value:.6e}")
    print(f"[gamma-d11-audit] A1_db={a1_db:.6e} A2_db={a2_db:.6e}")
    print(f"[gamma-d11-audit] A1_exact={a1_ex:.6e} A2_exact={a2_ex:.6e}")
    print()
    print("x-wise contributions")
    print(
        "x_idx x"
        "         db_field/v    ex_field/v    db_ln_nu     ex_ln_nu"
        "      db_D11        ex_D11"
        "      db_L11_term   ex_L11_term"
        "      db_L12_term   ex_L12_term"
        "      db_G11_term   ex_G11_term"
        "      db_G12_term   ex_G12_term"
    )
    for i, x_value in enumerate(x_values):
        print(
            f"{i:>3d} {x_value:>10.6e}"
            f" {field_db[i]:>12.6e} {field_ex[i]:>12.6e}"
            f" {np.log(max(nu_db[i], 1.0e-300)):>12.6e} {np.log(max(nu_ex[i], 1.0e-300)):>12.6e}"
            f" {d11_db[i]:>12.6e} {d11_ex[i]:>12.6e}"
            f" {contrib_db['l11_terms'][i]:>12.6e} {contrib_ex['l11_terms'][i]:>12.6e}"
            f" {contrib_db['l12_terms'][i]:>12.6e} {contrib_ex['l12_terms'][i]:>12.6e}"
            f" {contrib_db['gamma_l11_terms'][i]:>12.6e} {contrib_ex['gamma_l11_terms'][i]:>12.6e}"
            f" {contrib_db['gamma_l12_terms'][i]:>12.6e} {contrib_ex['gamma_l12_terms'][i]:>12.6e}"
        )

    l11_db = float(np.sum(contrib_db["l11_terms"]))
    l11_ex = float(np.sum(contrib_ex["l11_terms"]))
    l12_db = float(np.sum(contrib_db["l12_terms"]))
    l12_ex = float(np.sum(contrib_ex["l12_terms"]))
    gamma_db = float(-1.0e20 * np.asarray(density_db[species_index, radius_index], dtype=float) * (np.sum(contrib_db["gamma_l11_terms"]) + np.sum(contrib_db["gamma_l12_terms"])))
    gamma_ex = float(-1.0e20 * np.asarray(density_ex[species_index, radius_index], dtype=float) * (np.sum(contrib_ex["gamma_l11_terms"]) + np.sum(contrib_ex["gamma_l12_terms"])))

    print()
    print("[totals]")
    print(f"  L11_db={l11_db:.6e} L11_exact={l11_ex:.6e} rel={(abs(l11_ex-l11_db)/max(abs(l11_db),1.0e-30)):.6e}")
    print(f"  L12_db={l12_db:.6e} L12_exact={l12_ex:.6e} rel={(abs(l12_ex-l12_db)/max(abs(l12_db),1.0e-30)):.6e}")
    print(f"  Gamma_db={gamma_db:.6e} Gamma_exact={gamma_ex:.6e} rel={(abs(gamma_ex-gamma_db)/max(abs(gamma_db),1.0e-30)):.6e}")


if __name__ == "__main__":
    main()
