"""Compare exact-runtime monoenergetic D11 against direct prepared NTX solves.

This benchmark answers a narrow question:

- for the same local `(rho, species, Er)` case,
- and the same energy-point scan used by the exact runtime path,
- does the exact runtime coefficient path already differ from a direct
  `ntx.solve_prepared_coefficient_vector(...)` call?

If these agree, then any remaining mismatch in ambipolar `L11/L12/Gamma`
must be introduced later than the monoenergetic coefficient solve itself.
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
from NEOPAX._orchestrator import build_runtime_context
from NEOPAX._state import get_v_thermal, safe_density
from NEOPAX._transport_flux_models import (
    NTXExactLijRuntimeTransportModel,
    _collisionality_kind,
    _import_ntx,
)

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


def _direct_prepared_scan(ntx, prepared, nu_hat_a: jax.Array, epsi_hat_a: jax.Array) -> jax.Array:
    return jax.vmap(
        lambda nu_hat_value, epsi_hat_value: ntx.solve_prepared_coefficient_vector(
            prepared,
            ntx.MonoenergeticCase(
                nu_hat=jnp.asarray(nu_hat_value, dtype=jnp.float64),
                epsi_hat=jnp.asarray(epsi_hat_value, dtype=jnp.float64),
            ),
        )
    )(nu_hat_a, epsi_hat_a)


def _rel_abs_with_location(reference: np.ndarray, candidate: np.ndarray, x: np.ndarray) -> tuple[float, float, float, float]:
    abs_delta = np.abs(candidate - reference)
    rel_delta = abs_delta / np.maximum(np.abs(reference), 1.0e-30)
    idx = int(np.argmax(rel_delta))
    return float(rel_delta[idx]), float(abs_delta[idx]), float(x[idx]), float(reference[idx])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--exact-config", default=str(DEFAULT_EXACT_CONFIG))
    parser.add_argument("--radius-indices", default="20,30,40")
    parser.add_argument("--species-index", type=int, default=1)
    parser.add_argument("--er-min", type=float, default=None)
    parser.add_argument("--er-max", type=float, default=None)
    parser.add_argument("--n-er", type=int, default=None)
    args = parser.parse_args()

    cfg = _prepare_config(Path(args.exact_config), device=args.device)
    runtime, state = build_runtime_context(cfg)
    if state is None:
        raise RuntimeError("Exact config must build a transport state.")

    model = _extract_neoclassical_model(runtime.models.flux)
    if not isinstance(model, NTXExactLijRuntimeTransportModel):
        raise TypeError("Expected ntx_exact_lij_runtime neoclassical model.")
    model = model.with_static_support()
    support = model._static_support()
    ntx = _import_ntx()

    species_index = int(args.species_index)
    radius_indices = _parse_index_spec(args.radius_indices)
    if species_index < 0 or species_index >= int(runtime.species.number_species):
        raise ValueError(
            f"species-index {species_index} out of bounds for n_species={int(runtime.species.number_species)}"
        )

    amb_cfg = cfg.get("ambipolarity", {})
    er_min = float(args.er_min if args.er_min is not None else amb_cfg.get("er_ambipolar_scan_min", -50.0))
    er_max = float(args.er_max if args.er_max is not None else amb_cfg.get("er_ambipolar_scan_max", 50.0))
    n_er = int(args.n_er if args.n_er is not None else amb_cfg.get("er_ambipolar_n_coarse", 300))
    er_values = np.linspace(er_min, er_max, n_er, dtype=float)

    density = safe_density(state.density)
    temperature = state.temperature
    v_thermal = get_v_thermal(runtime.species.mass, temperature)
    collisionality_kind = _collisionality_kind(model.collisionality_model)
    rho = np.asarray(runtime.geometry.r_grid / runtime.geometry.a_b, dtype=float)
    x_grid = np.asarray(runtime.energy_grid.x, dtype=float)

    print(f"[runtime-vs-prepared-d11] exact_config={Path(args.exact_config).resolve()}")
    print(f"[runtime-vs-prepared-d11] species_index={species_index} ({_species_label(runtime, species_index)})")
    print(f"[runtime-vs-prepared-d11] er_scan=[{er_min:.6e}, {er_max:.6e}] n_er={n_er}")

    for radius_index in radius_indices:
        prepared = jax.tree_util.tree_map(
            lambda arr: jax.lax.dynamic_index_in_dim(arr, radius_index, axis=0, keepdims=False),
            support.center_prepared,
        )
        drds_value = jax.lax.dynamic_index_in_dim(support.center_channels.drds, radius_index, axis=0, keepdims=False)
        temperature_local = jax.lax.dynamic_index_in_dim(temperature, radius_index, axis=1, keepdims=False)
        density_local = jax.lax.dynamic_index_in_dim(density, radius_index, axis=1, keepdims=False)
        vthermal_local = jax.lax.dynamic_index_in_dim(v_thermal, radius_index, axis=1, keepdims=False)

        d11_rel_max = -1.0
        d11_abs_at_max = 0.0
        d11_er_at_max = 0.0
        d11_x_at_max = 0.0
        d11_runtime_at_max = 0.0
        d11_direct_at_max = 0.0

        d13_rel_max = -1.0
        d13_abs_at_max = 0.0
        d13_er_at_max = 0.0
        d13_x_at_max = 0.0
        d13_runtime_at_max = 0.0
        d13_direct_at_max = 0.0

        d33_rel_max = -1.0
        d33_abs_at_max = 0.0
        d33_er_at_max = 0.0
        d33_x_at_max = 0.0
        d33_runtime_at_max = 0.0
        d33_direct_at_max = 0.0

        for er_value in er_values:
            nu_hat_a, epsi_hat_a, _ = model._local_scan_inputs(
                drds_value=drds_value,
                species_index=species_index,
                er_value=jnp.asarray(er_value, dtype=jnp.float64),
                temperature_local=temperature_local,
                density_local=density_local,
                vthermal_local=vthermal_local,
                collisionality_kind=collisionality_kind,
            )
            runtime_raw = model._solve_coefficient_scan_prepared(prepared, nu_hat_a, epsi_hat_a)
            direct_raw = _direct_prepared_scan(ntx, prepared, nu_hat_a, epsi_hat_a)

            runtime_np = np.asarray(jax.device_get(runtime_raw), dtype=float)
            direct_np = np.asarray(jax.device_get(direct_raw), dtype=float)

            d11_rel, d11_abs, d11_x, d11_ref = _rel_abs_with_location(runtime_np[:, 0], direct_np[:, 0], x_grid)
            if d11_rel > d11_rel_max:
                idx = int(np.argmax(np.abs(direct_np[:, 0] - runtime_np[:, 0]) / np.maximum(np.abs(runtime_np[:, 0]), 1.0e-30)))
                d11_rel_max = d11_rel
                d11_abs_at_max = d11_abs
                d11_er_at_max = float(er_value)
                d11_x_at_max = d11_x
                d11_runtime_at_max = float(runtime_np[idx, 0])
                d11_direct_at_max = float(direct_np[idx, 0])

            d13_rel, d13_abs, d13_x, d13_ref = _rel_abs_with_location(runtime_np[:, 2], direct_np[:, 2], x_grid)
            if d13_rel > d13_rel_max:
                idx = int(np.argmax(np.abs(direct_np[:, 2] - runtime_np[:, 2]) / np.maximum(np.abs(runtime_np[:, 2]), 1.0e-30)))
                d13_rel_max = d13_rel
                d13_abs_at_max = d13_abs
                d13_er_at_max = float(er_value)
                d13_x_at_max = d13_x
                d13_runtime_at_max = float(runtime_np[idx, 2])
                d13_direct_at_max = float(direct_np[idx, 2])

            d33_rel, d33_abs, d33_x, d33_ref = _rel_abs_with_location(runtime_np[:, 3], direct_np[:, 3], x_grid)
            if d33_rel > d33_rel_max:
                idx = int(np.argmax(np.abs(direct_np[:, 3] - runtime_np[:, 3]) / np.maximum(np.abs(runtime_np[:, 3]), 1.0e-30)))
                d33_rel_max = d33_rel
                d33_abs_at_max = d33_abs
                d33_er_at_max = float(er_value)
                d33_x_at_max = d33_x
                d33_runtime_at_max = float(runtime_np[idx, 3])
                d33_direct_at_max = float(direct_np[idx, 3])

        print(f"\n[radius] idx={radius_index} rho={rho[radius_index]:.6e} drds={float(np.asarray(drds_value)):.6e}")
        print(
            f"  D11_rel_max={d11_rel_max:.6e} abs_max={d11_abs_at_max:.6e} "
            f"at Er={d11_er_at_max:.6e} x={d11_x_at_max:.6e} "
            f"(runtime={d11_runtime_at_max:.6e}, direct={d11_direct_at_max:.6e})"
        )
        print(
            f"  D13_rel_max={d13_rel_max:.6e} abs_max={d13_abs_at_max:.6e} "
            f"at Er={d13_er_at_max:.6e} x={d13_x_at_max:.6e} "
            f"(runtime={d13_runtime_at_max:.6e}, direct={d13_direct_at_max:.6e})"
        )
        print(
            f"  D33_rel_max={d33_rel_max:.6e} abs_max={d33_abs_at_max:.6e} "
            f"at Er={d33_er_at_max:.6e} x={d33_x_at_max:.6e} "
            f"(runtime={d33_runtime_at_max:.6e}, direct={d33_direct_at_max:.6e})"
        )


if __name__ == "__main__":
    main()
