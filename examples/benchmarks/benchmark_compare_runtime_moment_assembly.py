"""Compare exact-runtime moment/Lij assembly against manual reconstruction.

This benchmark uses the *same* exact-runtime raw coefficient scan and checks:

1. raw coefficients from `_solve_coefficient_scan_prepared(...)`
2. transport moments assembled by:
   - `NTXExactLijRuntimeTransportModel._transport_moments_from_coefficient_scan`
   - a manual weighted sum using the energy-grid weights
3. Lij assembled by:
   - `NTXExactLijRuntimeTransportModel._lij_from_transport_moments`
   - `NTXExactLijRuntimeTransportModel._solve_lij_prepared_local`

If all of these match, then any remaining mismatch must be outside the exact
runtime post-coefficient assembly path.
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


def _manual_transport_moments(energy_grid, coeff_scan: jax.Array, drds_value: jax.Array) -> jax.Array:
    d11_a = -(jnp.asarray(coeff_scan[:, 0], dtype=jnp.float64) * drds_value**2)
    d13_a = -(jnp.asarray(coeff_scan[:, 2], dtype=jnp.float64) * drds_value)
    d33_a = -jnp.asarray(coeff_scan[:, 3], dtype=jnp.float64)
    return jnp.stack(
        (
            jnp.sum(energy_grid.L11_weight * energy_grid.xWeights * d11_a),
            jnp.sum(energy_grid.L12_weight * energy_grid.xWeights * d11_a),
            jnp.sum(energy_grid.L22_weight * energy_grid.xWeights * d11_a),
            jnp.sum(energy_grid.L13_weight * energy_grid.xWeights * d13_a),
            jnp.sum(energy_grid.L23_weight * energy_grid.xWeights * d13_a),
            jnp.sum(energy_grid.L33_weight * energy_grid.xWeights * d33_a),
        ),
        axis=0,
    )


def _max_abs_rel(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    abs_max = float(np.max(np.abs(candidate - reference)))
    rel_max = float(np.max(np.abs(candidate - reference) / np.maximum(np.abs(reference), 1.0e-30)))
    return abs_max, rel_max


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

    print(f"[runtime-moment-assembly] exact_config={Path(args.exact_config).resolve()}")
    print(f"[runtime-moment-assembly] species_index={species_index} ({_species_label(runtime, species_index)})")
    print(f"[runtime-moment-assembly] er_scan=[{er_min:.6e}, {er_max:.6e}] n_er={n_er}")

    for radius_index in radius_indices:
        prepared = jax.tree_util.tree_map(
            lambda arr: jax.lax.dynamic_index_in_dim(arr, radius_index, axis=0, keepdims=False),
            support.center_prepared,
        )
        drds_value = jax.lax.dynamic_index_in_dim(support.center_channels.drds, radius_index, axis=0, keepdims=False)
        temperature_local = jax.lax.dynamic_index_in_dim(temperature, radius_index, axis=1, keepdims=False)
        density_local = jax.lax.dynamic_index_in_dim(density, radius_index, axis=1, keepdims=False)
        vthermal_local = jax.lax.dynamic_index_in_dim(v_thermal, radius_index, axis=1, keepdims=False)
        vth_a = jax.lax.dynamic_index_in_dim(vthermal_local, species_index, axis=0, keepdims=False)

        coeff_rel_max = np.zeros(3, dtype=float)
        moment_rel_max = np.zeros(6, dtype=float)
        lij_rel_max = np.zeros(6, dtype=float)
        lij_local_rel_max = np.zeros(6, dtype=float)

        coeff_abs_max = np.zeros(3, dtype=float)
        moment_abs_max = np.zeros(6, dtype=float)
        lij_abs_max = np.zeros(6, dtype=float)
        lij_local_abs_max = np.zeros(6, dtype=float)

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

            coeff_scan = model._solve_coefficient_scan_prepared(prepared, nu_hat_a, epsi_hat_a)
            moments_model = model._transport_moments_from_coefficient_scan(coeff_scan, drds_value=drds_value)
            moments_manual = _manual_transport_moments(runtime.energy_grid, coeff_scan, drds_value)

            lij_from_model_moments = model._lij_from_transport_moments(
                moments_model,
                species_index=species_index,
                vth_a=vth_a,
            )
            lij_from_manual_moments = model._lij_from_transport_moments(
                moments_manual,
                species_index=species_index,
                vth_a=vth_a,
            )
            lij_local = model._solve_lij_prepared_local(
                prepared,
                drds_value=drds_value,
                species_index=species_index,
                er_value=jnp.asarray(er_value, dtype=jnp.float64),
                temperature_local=temperature_local,
                density_local=density_local,
                vthermal_local=vthermal_local,
                collisionality_kind=collisionality_kind,
            )

            coeff_np = np.asarray(jax.device_get(coeff_scan), dtype=float)
            moments_model_np = np.asarray(jax.device_get(moments_model), dtype=float)
            moments_manual_np = np.asarray(jax.device_get(moments_manual), dtype=float)
            lij_model_np = np.asarray(jax.device_get(lij_from_model_moments), dtype=float)
            lij_manual_np = np.asarray(jax.device_get(lij_from_manual_moments), dtype=float)
            lij_local_np = np.asarray(jax.device_get(lij_local), dtype=float)

            coeff_ref = coeff_np
            coeff_cand = coeff_np
            for i in range(3):
                abs_max, rel_max = _max_abs_rel(coeff_ref[:, (0, 2, 3)[i]], coeff_cand[:, (0, 2, 3)[i]])
                coeff_abs_max[i] = max(coeff_abs_max[i], abs_max)
                coeff_rel_max[i] = max(coeff_rel_max[i], rel_max)

            for i in range(6):
                abs_max, rel_max = _max_abs_rel(np.asarray([moments_model_np[i]]), np.asarray([moments_manual_np[i]]))
                moment_abs_max[i] = max(moment_abs_max[i], abs_max)
                moment_rel_max[i] = max(moment_rel_max[i], rel_max)

            entries = ((0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2))
            for i, (row, col) in enumerate(entries):
                abs_max, rel_max = _max_abs_rel(np.asarray([lij_model_np[row, col]]), np.asarray([lij_manual_np[row, col]]))
                lij_abs_max[i] = max(lij_abs_max[i], abs_max)
                lij_rel_max[i] = max(lij_rel_max[i], rel_max)
                abs_max, rel_max = _max_abs_rel(np.asarray([lij_model_np[row, col]]), np.asarray([lij_local_np[row, col]]))
                lij_local_abs_max[i] = max(lij_local_abs_max[i], abs_max)
                lij_local_rel_max[i] = max(lij_local_rel_max[i], rel_max)

        print(f"\n[radius] idx={radius_index} rho={rho[radius_index]:.6e} drds={float(np.asarray(drds_value)):.6e}")
        print("  moments_model vs moments_manual")
        for label, abs_max, rel_max in zip(
            ("M11", "M12", "M22", "M13", "M23", "M33"),
            moment_abs_max,
            moment_rel_max,
        ):
            print(f"    {label}: abs_max={abs_max:.6e} rel_max={rel_max:.6e}")
        print("  Lij(model moments) vs Lij(manual moments)")
        for label, abs_max, rel_max in zip(
            ("L11", "L12", "L22", "L13", "L23", "L33"),
            lij_abs_max,
            lij_rel_max,
        ):
            print(f"    {label}: abs_max={abs_max:.6e} rel_max={rel_max:.6e}")
        print("  Lij(model moments) vs _solve_lij_prepared_local")
        for label, abs_max, rel_max in zip(
            ("L11", "L12", "L22", "L13", "L23", "L33"),
            lij_local_abs_max,
            lij_local_rel_max,
        ):
            print(f"    {label}: abs_max={abs_max:.6e} rel_max={rel_max:.6e}")


if __name__ == "__main__":
    main()
