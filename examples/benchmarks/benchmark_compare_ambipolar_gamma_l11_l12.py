"""Compare ambipolar Gamma(Er), L11(Er), and L12(Er) for database vs exact NTX.

This benchmark is meant to diagnose cases where ambipolar roots differ even
though node-level or even off-node Dij agreement looks decent. It evaluates the
same local neoclassical quantities used by the ambipolar root finder at chosen
radius indices for:

- a database-based NTX transport model
- the real-time exact NTX Lij runtime model

and then compares:

- charged ambipolar residual sum_s Z_s * Gamma_s
- species particle fluxes Gamma_s
- L11 and L12 for each species
- A1, A2, and A3 at the chosen radius
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
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
from NEOPAX._cell_variable import get_gradient_density, get_gradient_temperature
from NEOPAX._neoclassical import get_Lij_matrix_local
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

DEFAULT_DATABASE_CONFIG = Path("examples/Solve_Ambipolarity/ambiplarity_benchmark.toml")
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


def _assert_supported_model(model, label: str):
    if not isinstance(model, (NTXDatabaseTransportModel, NTXExactLijRuntimeTransportModel)):
        raise TypeError(
            f"{label} neoclassical model must be NTX database or NTX exact runtime, got {type(model).__name__}."
        )


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
    return a1, a2, a3


def _lij_database(model, state, radius_index: int, er_value: float):
    density = safe_density(state.density)
    temperature = state.temperature
    geometry = model.geometry
    species = model.species
    energy_grid = model.energy_grid
    database = model.database
    collisionality_kind = _collisionality_kind(model.collisionality_model)
    v_thermal = get_v_thermal(species.mass, temperature)
    species_indices = jnp.arange(int(species.number_species), dtype=jnp.int32)
    return jax.vmap(
        lambda species_index: get_Lij_matrix_local(
            species,
            energy_grid,
            geometry,
            database,
            species_index,
            radius_index,
            jnp.asarray(er_value, dtype=state.Er.dtype),
            temperature,
            density,
            v_thermal,
            collisionality_kind,
        )
    )(species_indices)


def _lij_exact(model, state, radius_index: int, er_value: float):
    density = safe_density(state.density)
    temperature = state.temperature
    species = model.species
    collisionality_kind = _collisionality_kind(model.collisionality_model)
    v_thermal = get_v_thermal(species.mass, temperature)
    species_indices = jnp.arange(int(species.number_species), dtype=jnp.int32)
    support = model._static_support()
    prepared = jax.tree_util.tree_map(
        lambda arr: jax.lax.dynamic_index_in_dim(arr, radius_index, axis=0, keepdims=False),
        support.center_prepared,
    )
    drds_value = jax.lax.dynamic_index_in_dim(support.center_channels.drds, radius_index, axis=0, keepdims=False)
    temperature_local = jax.lax.dynamic_index_in_dim(temperature, radius_index, axis=1, keepdims=False)
    density_local = jax.lax.dynamic_index_in_dim(density, radius_index, axis=1, keepdims=False)
    vthermal_local = jax.lax.dynamic_index_in_dim(v_thermal, radius_index, axis=1, keepdims=False)
    return jax.vmap(
        lambda species_index: model._solve_lij_prepared_local(
            prepared,
            drds_value=drds_value,
            species_index=species_index,
            er_value=jnp.asarray(er_value, dtype=state.Er.dtype),
            temperature_local=temperature_local,
            density_local=density_local,
            vthermal_local=vthermal_local,
            collisionality_kind=collisionality_kind,
        )
    )(species_indices)


def _evaluate_radius(model, state, radius_index: int, er_values: np.ndarray):
    geometry = model.geometry
    species = model.species
    density = safe_density(state.density)
    temperature = state.temperature

    per_species_density_phys = 1.0e20 * np.asarray(density[:, radius_index], dtype=float)
    species_charge_qp = np.asarray(species.charge_qp, dtype=float)

    gamma_species = []
    l11_species = []
    l12_species = []
    a1_species = []
    a2_species = []
    a3_scalar = []

    for er_value in er_values:
        er_profile = np.asarray(state.Er, dtype=float).copy()
        er_profile[radius_index] = float(er_value)
        er_profile_jax = jnp.asarray(er_profile, dtype=state.Er.dtype)
        a1_all, a2_all, a3_all = _compute_forces(model, geometry, species, density, temperature, er_profile_jax)
        a1_local = np.asarray(a1_all[:, radius_index], dtype=float)
        a2_local = np.asarray(a2_all[:, radius_index], dtype=float)
        a3_local = float(np.asarray(a3_all[radius_index], dtype=float))

        if isinstance(model, NTXDatabaseTransportModel):
            lij = _lij_database(model, state, radius_index, float(er_value))
        elif isinstance(model, NTXExactLijRuntimeTransportModel):
            lij = _lij_exact(model, state, radius_index, float(er_value))
        else:
            raise TypeError(f"Unsupported model type {type(model).__name__}")

        lij_np = np.asarray(lij, dtype=float)
        gamma_local = -per_species_density_phys * (
            lij_np[:, 0, 0] * a1_local
            + lij_np[:, 0, 1] * a2_local
            + lij_np[:, 0, 2] * a3_local
        )

        gamma_species.append(gamma_local)
        l11_species.append(lij_np[:, 0, 0])
        l12_species.append(lij_np[:, 0, 1])
        a1_species.append(a1_local)
        a2_species.append(a2_local)
        a3_scalar.append(a3_local)

    gamma_species_np = np.asarray(gamma_species, dtype=float)
    l11_np = np.asarray(l11_species, dtype=float)
    l12_np = np.asarray(l12_species, dtype=float)
    a1_np = np.asarray(a1_species, dtype=float)
    a2_np = np.asarray(a2_species, dtype=float)
    a3_np = np.asarray(a3_scalar, dtype=float)
    charged_gamma = np.sum(species_charge_qp[None, :] * gamma_species_np, axis=1)

    return {
        "gamma_species": gamma_species_np,
        "charged_gamma": charged_gamma,
        "l11": l11_np,
        "l12": l12_np,
        "a1": a1_np,
        "a2": a2_np,
        "a3": a3_np,
    }


def _relative_max(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.maximum(np.abs(b), 1.0e-30)
    return float(np.max(np.abs(a - b) / denom))


def _relative_max_with_location(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> tuple[float, float, float, float]:
    denom = np.maximum(np.abs(b), 1.0e-30)
    rel = np.abs(a - b) / denom
    idx = int(np.argmax(rel))
    return float(rel[idx]), float(x[idx]), float(a[idx]), float(b[idx])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--database-config", default=str(DEFAULT_DATABASE_CONFIG))
    parser.add_argument("--exact-config", default=str(DEFAULT_EXACT_CONFIG))
    parser.add_argument("--radius-indices", default="20,30,40")
    parser.add_argument("--er-min", type=float, default=None)
    parser.add_argument("--er-max", type=float, default=None)
    parser.add_argument("--n-er", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/benchmark_ambipolar_gamma_l11_l12")
    args = parser.parse_args()

    db_cfg = _prepare_config(Path(args.database_config), device=args.device)
    ex_cfg = _prepare_config(Path(args.exact_config), device=args.device)

    db_runtime, db_state = build_runtime_context(db_cfg)
    ex_runtime, ex_state = build_runtime_context(ex_cfg)
    if db_state is None or ex_state is None:
        raise RuntimeError("Both configs must build a transport state.")

    db_model = _extract_neoclassical_model(db_runtime.models.flux)
    ex_model = _extract_neoclassical_model(ex_runtime.models.flux)
    _assert_supported_model(db_model, "database")
    _assert_supported_model(ex_model, "exact")

    radius_indices = _parse_index_spec(args.radius_indices)
    amb_cfg = ex_cfg.get("ambipolarity", {})
    er_min = float(args.er_min if args.er_min is not None else amb_cfg.get("er_ambipolar_scan_min", -50.0))
    er_max = float(args.er_max if args.er_max is not None else amb_cfg.get("er_ambipolar_scan_max", 50.0))
    n_er = int(args.n_er if args.n_er is not None else amb_cfg.get("er_ambipolar_n_coarse", 300))
    er_values = np.linspace(er_min, er_max, n_er, dtype=float)

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    species_names = tuple(str(name) for name in db_runtime.species.names)
    rho = np.asarray(db_runtime.geometry.r_grid / db_runtime.geometry.a_b, dtype=float)

    print(f"[gamma-l11-l12] database_config={Path(args.database_config).resolve()}")
    print(f"[gamma-l11-l12] exact_config={Path(args.exact_config).resolve()}")
    print(f"[gamma-l11-l12] er_scan=[{er_min:.6e}, {er_max:.6e}] n_er={n_er}")

    for radius_index in radius_indices:
        db_eval = _evaluate_radius(db_model, db_state, radius_index, er_values)
        ex_eval = _evaluate_radius(ex_model, ex_state, radius_index, er_values)

        print(
            f"\n[radius] idx={radius_index} rho={rho[radius_index]:.6e} "
            f"A3_db_max={np.max(np.abs(db_eval['a3'])):.6e} "
            f"A3_exact_max={np.max(np.abs(ex_eval['a3'])):.6e}"
        )
        charged_rel, charged_er_at_max, charged_db_at_max, charged_exact_at_max = _relative_max_with_location(
            db_eval["charged_gamma"],
            ex_eval["charged_gamma"],
            er_values,
        )
        print(
            "  charged_gamma: "
            f"abs_max={np.max(np.abs(db_eval['charged_gamma'] - ex_eval['charged_gamma'])):.6e} "
            f"rel_max={charged_rel:.6e} "
            f"at Er={charged_er_at_max:.6e} "
            f"(db={charged_db_at_max:.6e}, exact={charged_exact_at_max:.6e})"
        )
        for s_idx, s_name in enumerate(species_names):
            l11_rel, l11_er_at_max, l11_db_at_max, l11_exact_at_max = _relative_max_with_location(
                db_eval['l11'][:, s_idx],
                ex_eval['l11'][:, s_idx],
                er_values,
            )
            l12_rel, l12_er_at_max, l12_db_at_max, l12_exact_at_max = _relative_max_with_location(
                db_eval['l12'][:, s_idx],
                ex_eval['l12'][:, s_idx],
                er_values,
            )
            gamma_rel, gamma_er_at_max, gamma_db_at_max, gamma_exact_at_max = _relative_max_with_location(
                db_eval['gamma_species'][:, s_idx],
                ex_eval['gamma_species'][:, s_idx],
                er_values,
            )
            print(
                f"  {s_name}: "
                f"L11_rel_max={l11_rel:.6e} at Er={l11_er_at_max:.6e} "
                f"(db={l11_db_at_max:.6e}, exact={l11_exact_at_max:.6e}) "
                f"L12_rel_max={l12_rel:.6e} at Er={l12_er_at_max:.6e} "
                f"(db={l12_db_at_max:.6e}, exact={l12_exact_at_max:.6e}) "
                f"Gamma_rel_max={gamma_rel:.6e} at Er={gamma_er_at_max:.6e} "
                f"(db={gamma_db_at_max:.6e}, exact={gamma_exact_at_max:.6e})"
            )

        fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
        ax = axes[0, 0]
        ax.plot(er_values, ex_eval["charged_gamma"], color="black", label="exact")
        ax.plot(er_values, db_eval["charged_gamma"], color="tab:blue", linestyle="--", label="database")
        ax.axhline(0.0, color="0.7", linewidth=1.0)
        ax.set_title("Charged Ambipolar Residual")
        ax.set_xlabel("Er")
        ax.set_ylabel(r"$\sum_s Z_s \Gamma_s$")
        ax.legend()

        ax = axes[0, 1]
        for s_idx, s_name in enumerate(species_names):
            ax.plot(er_values, ex_eval["gamma_species"][:, s_idx], label=f"{s_name} exact")
            ax.plot(er_values, db_eval["gamma_species"][:, s_idx], linestyle="--", label=f"{s_name} db")
        ax.axhline(0.0, color="0.7", linewidth=1.0)
        ax.set_title("Species Gamma")
        ax.set_xlabel("Er")
        ax.set_ylabel(r"$\Gamma_s$")
        ax.legend(fontsize=8, ncol=2)

        ax = axes[1, 0]
        for s_idx, s_name in enumerate(species_names):
            ax.plot(er_values, ex_eval["l11"][:, s_idx], label=f"{s_name} exact")
            ax.plot(er_values, db_eval["l11"][:, s_idx], linestyle="--", label=f"{s_name} db")
        ax.set_title("L11")
        ax.set_xlabel("Er")
        ax.set_ylabel("L11")
        ax.legend(fontsize=8, ncol=2)

        ax = axes[1, 1]
        for s_idx, s_name in enumerate(species_names):
            ax.plot(er_values, ex_eval["l12"][:, s_idx], label=f"{s_name} exact")
            ax.plot(er_values, db_eval["l12"][:, s_idx], linestyle="--", label=f"{s_name} db")
        ax.set_title("L12")
        ax.set_xlabel("Er")
        ax.set_ylabel("L12")
        ax.legend(fontsize=8, ncol=2)

        fig.suptitle(
            f"radius_index={radius_index}, rho={rho[radius_index]:.6f} | "
            f"A3_exact_max={np.max(np.abs(ex_eval['a3'])):.2e}"
        )
        out_path = out_dir / f"gamma_l11_l12_radius_{radius_index:03d}.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
