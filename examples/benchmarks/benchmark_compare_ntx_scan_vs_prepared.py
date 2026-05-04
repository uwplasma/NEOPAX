"""Compare NTX scan and prepared-solve raw coefficients on one NEOPAX runtime surface.

This isolates whether a mismatch is happening:
- inside NTX itself (scan vs prepared direct solve), or
- later in NEOPAX input construction / bridge / transport assembly.
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

DEFAULT_EXACT_CONFIG = Path("examples/benchmarks/Calculate_Fluxes_noHe_ntx_exact_lij_runtime_benchmark.toml")


def _prepare_config(
    config_path: Path,
    *,
    device: str,
    er_init_mode: str,
    resolution: tuple[int, int, int] | None = None,
):
    config = NEOPAX.prepare_config(config_path, device=device)
    config = copy.deepcopy(config)
    config.setdefault("profiles", {})["er_initialization_mode"] = str(er_init_mode)
    neoclassical = config.setdefault("neoclassical", {})
    neoclassical["flux_model"] = "ntx_exact_lij_runtime"
    if resolution is not None:
        n_theta, n_zeta, n_xi = resolution
        neoclassical["ntx_exact_n_theta"] = int(n_theta)
        neoclassical["ntx_exact_n_zeta"] = int(n_zeta)
        neoclassical["ntx_exact_n_xi"] = int(n_xi)
    return config


def _parse_resolution(spec: str) -> tuple[int, int, int]:
    parts = [piece.strip() for piece in str(spec).split(",")]
    if len(parts) != 3:
        raise ValueError(f"Resolution '{spec}' must be in 'n_theta,n_zeta,n_xi' format.")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _extract_neoclassical_model(model):
    return getattr(model, "neoclassical_model", model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--exact-config", default=str(DEFAULT_EXACT_CONFIG))
    parser.add_argument(
        "--er-init-mode",
        default="analytical",
        choices=["keep", "analytical", "ambipolar_min_entropy"],
    )
    parser.add_argument("--radius-index", type=int, default=10)
    parser.add_argument("--species-index", type=int, default=1)
    parser.add_argument("--resolution", default="25,25,63")
    args = parser.parse_args()

    resolution = _parse_resolution(args.resolution)
    cfg = _prepare_config(
        Path(args.exact_config),
        device=args.device,
        er_init_mode=args.er_init_mode,
        resolution=resolution,
    )
    runtime, state = build_runtime_context(cfg)

    model = _extract_neoclassical_model(runtime.models.flux)
    if not isinstance(model, NTXExactLijRuntimeTransportModel):
        raise TypeError("Expected ntx_exact_lij_runtime neoclassical model.")
    model = model.with_static_support()
    support = model._static_support()
    ntx = _import_ntx()

    radius_index = int(args.radius_index)
    species_index = int(args.species_index)

    density = safe_density(state.density)
    temperature = state.temperature
    v_thermal = get_v_thermal(runtime.species.mass, temperature)
    collisionality_kind = _collisionality_kind(model.collisionality_model)

    prepared = jax.tree_util.tree_map(
        lambda arr: jax.lax.dynamic_index_in_dim(arr, radius_index, axis=0, keepdims=False),
        support.center_prepared,
    )
    drds_value = jax.lax.dynamic_index_in_dim(support.center_channels.drds, radius_index, axis=0, keepdims=False)
    er_value = jax.lax.dynamic_index_in_dim(state.Er, radius_index, axis=0, keepdims=False)
    temperature_local = jax.lax.dynamic_index_in_dim(temperature, radius_index, axis=1, keepdims=False)
    density_local = jax.lax.dynamic_index_in_dim(density, radius_index, axis=1, keepdims=False)
    vthermal_local = jax.lax.dynamic_index_in_dim(v_thermal, radius_index, axis=1, keepdims=False)

    nu_hat_a, epsi_hat_a, _ = model._local_scan_inputs(
        drds_value=drds_value,
        species_index=species_index,
        er_value=er_value,
        temperature_local=temperature_local,
        density_local=density_local,
        vthermal_local=vthermal_local,
        collisionality_kind=collisionality_kind,
    )

    direct_raw = jax.vmap(
        lambda nu_hat_value, epsi_hat_value: ntx.solve_prepared_coefficient_vector(
            prepared,
            ntx.MonoenergeticCase(
                nu_hat=jnp.asarray(nu_hat_value, dtype=jnp.float64),
                epsi_hat=jnp.asarray(epsi_hat_value, dtype=jnp.float64),
            ),
        )
    )(nu_hat_a, epsi_hat_a)

    scan = ntx.solve_monoenergetic_scan(
        prepared.surface,
        prepared.grid,
        nu_hat_a,
        epsi_hat=epsi_hat_a,
    )
    scan_raw = jnp.stack(
        (
            jnp.asarray(scan["D11"], dtype=jnp.float64),
            jnp.asarray(scan["D31"], dtype=jnp.float64),
            jnp.asarray(scan["D13"], dtype=jnp.float64),
            jnp.asarray(scan["D33"], dtype=jnp.float64),
            jnp.asarray(scan["D33_spitzer"], dtype=jnp.float64),
        ),
        axis=1,
    )

    jax.block_until_ready(direct_raw)
    jax.block_until_ready(scan_raw)

    labels = ("D11", "D31", "D13", "D33", "D33_spitzer")

    def _delta(index: int):
        ref = scan_raw[:, index]
        cand = direct_raw[:, index]
        abs_max = float(jnp.max(jnp.abs(cand - ref)))
        rel_max = float(jnp.max(jnp.abs(cand - ref) / jnp.maximum(jnp.abs(ref), 1.0e-30)))
        return abs_max, rel_max

    print(f"[ntx-compare] device={args.device}")
    print(f"[ntx-compare] er_init_mode={args.er_init_mode}")
    print(f"[ntx-compare] radius_index={radius_index}")
    print(f"[ntx-compare] species_index={species_index}")
    print(f"[ntx-compare] resolution={resolution}")
    print(f"[ntx-compare] rho={float(runtime.geometry.r_grid[radius_index]):.6e}")
    print(f"[ntx-compare] drds={float(drds_value):.6e}")
    print(
        f"[ntx-compare] nu_hat_range=[{float(jnp.min(nu_hat_a)):.6e}, {float(jnp.max(nu_hat_a)):.6e}] "
        f"epsi_hat_range=[{float(jnp.min(epsi_hat_a)):.6e}, {float(jnp.max(epsi_hat_a)):.6e}]"
    )
    print()
    print("prepared direct raw vs scan raw")
    print("quantity          abs_max            rel_max")
    print("---------------------------------------------")
    for idx, label in enumerate(labels):
        abs_max, rel_max = _delta(idx)
        print(f"{label:<12}{abs_max:>16.6e}{rel_max:>18.6e}")


if __name__ == "__main__":
    main()
