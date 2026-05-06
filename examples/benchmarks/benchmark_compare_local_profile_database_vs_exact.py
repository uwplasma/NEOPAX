"""Compare local profile-point monoenergetic curves: database Er-path vs exact Es/v-path.

This benchmark is intentionally apples-to-apples for the user question:

- database side:
  use the same local database query path driven by the profile `Er` value
  (i.e. the same `Er -> Er/v_new` mapping used by `get_Lij_matrix_local`)

- exact side:
  use the exact runtime local scan inputs, which feed NTX with `Es/v_new`
  through `_local_scan_inputs(...)`

It reports and plots local `D11(x)`, `D13(x)`, and `D33(x)` at selected
profile radii and species.
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
from NEOPAX._neoclassical import _nu_over_vnew
from NEOPAX._orchestrator import build_runtime_context
from NEOPAX._state import get_v_thermal, safe_density
from NEOPAX._transport_flux_models import (
    NTXDatabaseTransportModel,
    NTXExactLijRuntimeTransportModel,
    _collisionality_kind,
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


def _species_label(runtime, species_index: int) -> str:
    names = getattr(runtime.species, "names", None)
    if names is not None and species_index < len(names):
        return str(names[species_index])
    return f"s{species_index}"


def _db_channels_to_physical(db_coeffs_raw: jax.Array, nu_hat_a: jax.Array) -> jax.Array:
    return jnp.stack(
        (
            10.0 ** jnp.asarray(db_coeffs_raw[:, 0], dtype=jnp.float64),
            jnp.asarray(db_coeffs_raw[:, 1], dtype=jnp.float64),
            jnp.asarray(db_coeffs_raw[:, 2], dtype=jnp.float64) / jnp.maximum(jnp.asarray(nu_hat_a, dtype=jnp.float64), 1.0e-30),
        ),
        axis=1,
    )


def _exact_raw_to_physical(exact_coeffs: jax.Array, drds_value: jax.Array) -> jax.Array:
    return jnp.stack(
        (
            jnp.asarray(exact_coeffs[:, 0], dtype=jnp.float64) * drds_value**2,
            jnp.asarray(exact_coeffs[:, 2], dtype=jnp.float64) * drds_value,
            jnp.asarray(exact_coeffs[:, 3], dtype=jnp.float64),
        ),
        axis=1,
    )


def _delta_stats(reference: np.ndarray, candidate: np.ndarray, x_grid: np.ndarray) -> tuple[float, float, float, float, float]:
    abs_delta = np.abs(candidate - reference)
    rel_delta = abs_delta / np.maximum(np.abs(reference), 1.0e-30)
    idx = int(np.argmax(rel_delta))
    return (
        float(np.max(abs_delta)),
        float(rel_delta[idx]),
        float(x_grid[idx]),
        float(reference[idx]),
        float(candidate[idx]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--database-config", default=str(DEFAULT_DATABASE_CONFIG))
    parser.add_argument("--exact-config", default=str(DEFAULT_EXACT_CONFIG))
    parser.add_argument("--radius-indices", default="20,30,40")
    parser.add_argument("--species-index", type=int, default=1)
    parser.add_argument("--output-dir", default="outputs/benchmark_local_profile_database_vs_exact")
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
    ex_model = ex_model.with_static_support()
    support = ex_model._static_support()

    radius_indices = _parse_index_spec(args.radius_indices)
    species_index = int(args.species_index)
    x_grid = np.asarray(db_runtime.energy_grid.x, dtype=float)
    rho = np.asarray(db_runtime.geometry.r_grid / db_runtime.geometry.a_b, dtype=float)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    db_density = safe_density(db_state.density)
    db_temperature = db_state.temperature
    db_vthermal = get_v_thermal(db_runtime.species.mass, db_temperature)
    db_kernel = monoenergetic_interpolation_kernel(db_runtime.database)

    print(f"[local-profile-db-vs-exact] database_config={Path(args.database_config).resolve()}")
    print(f"[local-profile-db-vs-exact] exact_config={Path(args.exact_config).resolve()}")
    print(f"[local-profile-db-vs-exact] species_index={species_index} ({_species_label(db_runtime, species_index)})")

    for radius_index in radius_indices:
        er_value = float(np.asarray(db_state.Er[radius_index], dtype=float))
        radius_value = jnp.asarray(db_runtime.geometry.r_grid[radius_index], dtype=jnp.float64)
        vth_a = db_vthermal[species_index, radius_index]
        v_new_a = db_runtime.energy_grid.v_norm * vth_a
        nu_hat_a = _nu_over_vnew(
            db_runtime.species,
            species_index,
            v_new_a,
            radius_index,
            db_density,
            db_temperature,
            db_vthermal,
            _collisionality_kind(db_model.collisionality_model),
        )
        er_vnew_a = jnp.asarray(er_value, dtype=jnp.float64) * 1.0e3 / v_new_a
        db_coeffs_raw = jax.vmap(db_kernel, in_axes=(None, 0, 0, None))(
            radius_value,
            nu_hat_a,
            er_vnew_a,
            db_runtime.database,
        )
        db_phys = _db_channels_to_physical(db_coeffs_raw, nu_hat_a)

        prepared = jax.tree_util.tree_map(
            lambda arr: jax.lax.dynamic_index_in_dim(arr, radius_index, axis=0, keepdims=False),
            support.center_prepared,
        )
        drds_value = jax.lax.dynamic_index_in_dim(support.center_channels.drds, radius_index, axis=0, keepdims=False)
        temperature_local = jax.lax.dynamic_index_in_dim(db_temperature, radius_index, axis=1, keepdims=False)
        density_local = jax.lax.dynamic_index_in_dim(db_density, radius_index, axis=1, keepdims=False)
        vthermal_local = jax.lax.dynamic_index_in_dim(db_vthermal, radius_index, axis=1, keepdims=False)

        nu_hat_exact, epsi_hat_exact, _ = ex_model._local_scan_inputs(
            drds_value=drds_value,
            species_index=species_index,
            er_value=jnp.asarray(er_value, dtype=jnp.float64),
            temperature_local=temperature_local,
            density_local=density_local,
            vthermal_local=vthermal_local,
            collisionality_kind=_collisionality_kind(ex_model.collisionality_model),
        )
        exact_raw = ex_model._solve_coefficient_scan_prepared(prepared, nu_hat_exact, epsi_hat_exact)
        exact_phys = _exact_raw_to_physical(exact_raw, drds_value)

        db_np = np.asarray(jax.device_get(db_phys), dtype=float)
        exact_np = np.asarray(jax.device_get(exact_phys), dtype=float)

        print(f"\n[radius] idx={radius_index} rho={rho[radius_index]:.6e} Er={er_value:.6e} drds={float(np.asarray(drds_value)):.6e}")
        print(
            f"  field ranges: database Er/v=[{float(jnp.min(er_vnew_a)):.6e}, {float(jnp.max(er_vnew_a)):.6e}] "
            f"exact Es/v=[{float(jnp.min(epsi_hat_exact)):.6e}, {float(jnp.max(epsi_hat_exact)):.6e}]"
        )
        for label, idx in (("D11", 0), ("D13", 1), ("D33", 2)):
            abs_max, rel_max, x_at_max, db_at_max, ex_at_max = _delta_stats(db_np[:, idx], exact_np[:, idx], x_grid)
            print(
                f"  {label}: abs_max={abs_max:.6e} rel_max={rel_max:.6e} "
                f"at x={x_at_max:.6e} (db={db_at_max:.6e}, exact={ex_at_max:.6e})"
            )

        fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, constrained_layout=True)
        for ax, (label, idx) in zip(axes, (("D11", 0), ("D13", 1), ("D33", 2))):
            ax.plot(x_grid, db_np[:, idx], label="database (Er path)", linewidth=2.2)
            ax.plot(x_grid, exact_np[:, idx], label="exact runtime (Es/v path)", linewidth=2.0, linestyle="--")
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
            ax.legend()
        axes[-1].set_xlabel("x")
        fig.suptitle(
            f"radius_index={radius_index}, rho={rho[radius_index]:.6f}, "
            f"species={_species_label(db_runtime, species_index)}, Er={er_value:.6e}"
        )
        out_path = output_dir / f"local_profile_radius_{radius_index:03d}_species_{species_index}.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        print(f"  plot={out_path}")


if __name__ == "__main__":
    main()
