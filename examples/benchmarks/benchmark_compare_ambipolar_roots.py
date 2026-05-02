"""Run and compare ambipolar best-root profiles for database and exact-runtime
benchmark configs, and save an overlay plot.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import NEOPAX

DEFAULT_DB_CONFIG = Path("examples/benchmarks/Solve_Ambipolarity_noHe_ntx_database_benchmark.toml")
DEFAULT_EXACT_CONFIG = Path("examples/benchmarks/Solve_Ambipolarity_noHe_ntx_exact_lij_runtime_benchmark.toml")


def _run_once(config_path: Path, device: str):
    config = NEOPAX.prepare_config(config_path, mode="ambipolarity", device=device)
    t0 = time.perf_counter()
    result = NEOPAX.run(config)
    best_root = jnp.asarray(result.raw_result["best_root"])
    rho = jnp.asarray(result.raw_result["rho"])
    jax.block_until_ready(best_root)
    dt = time.perf_counter() - t0
    return rho, best_root, dt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--database-config", default=str(DEFAULT_DB_CONFIG))
    parser.add_argument("--exact-config", default=str(DEFAULT_EXACT_CONFIG))
    parser.add_argument(
        "--output",
        default="outputs/benchmark_ambipolarity_compare/compare_ambipolar_best_roots.png",
        help="Output PNG path.",
    )
    args = parser.parse_args()

    db_cfg = Path(args.database_config)
    exact_cfg = Path(args.exact_config)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[amb-compare] device={args.device}")
    print(f"[amb-compare] database_config={db_cfg}")
    print(f"[amb-compare] exact_config={exact_cfg}")

    rho_db, root_db, dt_db = _run_once(db_cfg, args.device)
    rho_exact, root_exact, dt_exact = _run_once(exact_cfg, args.device)

    if rho_db.shape != rho_exact.shape or jnp.max(jnp.abs(rho_db - rho_exact)) > 1.0e-12:
        raise ValueError("Database and exact-runtime rho grids do not match.")

    max_delta = float(jnp.max(jnp.abs(root_exact - root_db)))

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(rho_db, root_db, label=f"ntx_database ({dt_db:.2f}s)", linewidth=2.0)
    ax.plot(rho_exact, root_exact, label=f"ntx_exact_lij_runtime ({dt_exact:.2f}s)", linewidth=2.0)
    ax.set_xlabel(r"$\\rho$")
    ax.set_ylabel("Best-root Er")
    ax.set_title(f"Ambipolar Best Roots Comparison\nmax|delta| = {max_delta:.3e}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)

    print(f"[amb-compare] ntx_database_s={dt_db:.3f}")
    print(f"[amb-compare] ntx_exact_lij_runtime_s={dt_exact:.3f}")
    print(f"[amb-compare] max_abs_delta={max_delta:.6e}")
    print(f"[amb-compare] output={out_path}")


if __name__ == "__main__":
    main()
