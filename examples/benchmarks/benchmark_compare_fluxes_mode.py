"""Compare native mode='fluxes' outputs for database and exact-runtime models."""

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

DEFAULT_DATABASE_CONFIG = Path("examples/benchmarks/Calculate_Fluxes_noHe_ntx_database_benchmark.toml")
DEFAULT_EXACT_CONFIG = Path("examples/benchmarks/Calculate_Fluxes_noHe_ntx_exact_lij_runtime_benchmark.toml")


def _run_config(config_path: Path, *, device: str):
    config = NEOPAX.prepare_config(config_path, device=device)
    config = copy.deepcopy(config)
    config.setdefault("general", {})["mode"] = "fluxes"
    t0 = time.perf_counter()
    result = NEOPAX.run(config)
    raw = result.raw_result if hasattr(result, "raw_result") else {}
    rho = jnp.asarray(raw["rho"])
    fluxes = {key: jnp.asarray(value) for key, value in raw["fluxes"].items()}
    jax.block_until_ready(rho)
    for value in fluxes.values():
        jax.block_until_ready(value)
    dt = time.perf_counter() - t0
    return rho, fluxes, dt, Path(raw["output_dir"]) if isinstance(raw, dict) and raw.get("output_dir") is not None else None


def _max_delta(a: dict, b: dict) -> float:
    max_delta = 0.0
    for key in sorted(a.keys()):
        if key not in b:
            return float("nan")
        arr_a = jnp.asarray(a[key])
        arr_b = jnp.asarray(b[key])
        if arr_a.shape != arr_b.shape:
            return float("nan")
        max_delta = max(max_delta, float(jnp.max(jnp.abs(arr_b - arr_a))))
    return max_delta


def _species_names(n_species: int):
    default = ["e", "D", "T"]
    if n_species <= len(default):
        return default[:n_species]
    return default + [f"s{i}" for i in range(len(default), n_species)]


def _plot_quantity(output_dir: Path, rho, database_fluxes, exact_fluxes, quantity: str):
    db = jnp.asarray(database_fluxes[quantity])
    ex = jnp.asarray(exact_fluxes[quantity])
    if db.ndim != 2 or ex.ndim != 2:
        return None
    species_names = _species_names(int(db.shape[0]))
    fig, axes = plt.subplots(int(db.shape[0]), 1, figsize=(9, 3 * int(db.shape[0])), sharex=True)
    if int(db.shape[0]) == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(rho, db[i], linewidth=2.0, label="database")
        ax.plot(rho, ex[i], linewidth=2.0, linestyle="--", label="exact_runtime")
        ax.set_ylabel(f"{quantity}[{species_names[i]}]")
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("rho")
    fig.tight_layout()
    out = output_dir / f"compare_{quantity}.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--database-config", default=str(DEFAULT_DATABASE_CONFIG))
    parser.add_argument("--exact-config", default=str(DEFAULT_EXACT_CONFIG))
    args = parser.parse_args()

    db_cfg = Path(args.database_config)
    ex_cfg = Path(args.exact_config)
    print(f"[flux-compare] device={args.device}")
    print(f"[flux-compare] database_config={db_cfg}")
    print(f"[flux-compare] exact_config={ex_cfg}")

    rho_db, fluxes_db, dt_db, out_db = _run_config(db_cfg, device=args.device)
    rho_ex, fluxes_ex, dt_ex, out_ex = _run_config(ex_cfg, device=args.device)

    rho_delta = float(jnp.max(jnp.abs(rho_ex - rho_db)))
    flux_delta = _max_delta(fluxes_db, fluxes_ex)

    output_dir = Path("outputs/benchmark_fluxes_compare")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = {}
    for quantity in ("Gamma", "Q", "Upar", "Gamma_neo", "Q_neo", "Upar_neo"):
        if quantity in fluxes_db and quantity in fluxes_ex:
            plot_paths[quantity] = _plot_quantity(output_dir, np.asarray(rho_db), fluxes_db, fluxes_ex, quantity)

    print()
    print("model                    wall_s")
    print("------------------------------")
    print(f"{'ntx_database':<24}{dt_db:>8.3f}")
    print(f"{'ntx_exact_lij_runtime':<24}{dt_ex:>8.3f}")
    print()
    print(f"[flux-compare] rho_max_delta={rho_delta:.6e}")
    print(f"[flux-compare] flux_max_delta={flux_delta:.6e}")
    print(f"[flux-compare] output_dir={output_dir}")
    if out_db is not None:
        print(f"[flux-compare] database_native_output={out_db}")
    if out_ex is not None:
        print(f"[flux-compare] exact_native_output={out_ex}")
    for key, path in plot_paths.items():
        if path is not None:
            print(f"[flux-compare] plot_{key}={path}")


if __name__ == "__main__":
    main()
