"""Compare native mode='fluxes' outputs for database and exact-runtime models.

One run can compare:
- database baseline
- exact-runtime with interpolate_center_response
- exact-runtime with face_local_response
- multiple exact-runtime resolutions
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

DEFAULT_DATABASE_CONFIG = Path("examples/benchmarks/Calculate_Fluxes_noHe_ntx_database_benchmark.toml")
DEFAULT_EXACT_CONFIG = Path("examples/benchmarks/Calculate_Fluxes_noHe_ntx_exact_lij_runtime_benchmark.toml")
DEFAULT_FACE_MODES = ["interpolate_center_response", "face_local_response"]
DEFAULT_RESOLUTIONS = ["5,21,32"]


def _parse_resolution(spec: str) -> tuple[int, int, int]:
    parts = [piece.strip() for piece in str(spec).split(",")]
    if len(parts) != 3:
        raise ValueError(f"Resolution '{spec}' must be in 'n_theta,n_zeta,n_xi' format.")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _run_config(
    config_path: Path,
    *,
    device: str,
    er_init_mode: str,
    flux_model: str | None = None,
    face_mode: str | None = None,
    resolution: tuple[int, int, int] | None = None,
    output_dir: Path | None = None,
):
    config = NEOPAX.prepare_config(config_path, device=device)
    config = copy.deepcopy(config)
    config.setdefault("general", {})["mode"] = "fluxes"
    config.setdefault("profiles", {})["er_initialization_mode"] = str(er_init_mode)
    neoclassical = config.setdefault("neoclassical", {})
    if flux_model is not None:
        neoclassical["flux_model"] = str(flux_model)
    if face_mode is not None:
        neoclassical["ntx_exact_face_response_mode"] = str(face_mode)
    if resolution is not None:
        n_theta, n_zeta, n_xi = resolution
        neoclassical["ntx_exact_n_theta"] = int(n_theta)
        neoclassical["ntx_exact_n_zeta"] = int(n_zeta)
        neoclassical["ntx_exact_n_xi"] = int(n_xi)
    if output_dir is not None:
        config.setdefault("fluxes", {})["fluxes_output_dir"] = str(output_dir)

    t0 = time.perf_counter()
    result = NEOPAX.run(config)
    raw = result.raw_result if hasattr(result, "raw_result") else {}
    rho = jnp.asarray(raw["rho"])
    fluxes = {key: jnp.asarray(value) for key, value in raw["fluxes"].items()}
    jax.block_until_ready(rho)
    for value in fluxes.values():
        jax.block_until_ready(value)
    dt = time.perf_counter() - t0
    native_output = Path(raw["output_dir"]) if isinstance(raw, dict) and raw.get("output_dir") is not None else None
    return rho, fluxes, dt, native_output


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


def _plot_quantity(output_dir: Path, rho, cases: list[dict], quantity: str):
    ref = jnp.asarray(cases[0]["fluxes"][quantity])
    if ref.ndim != 2:
        return None
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
            arr = jnp.asarray(case["fluxes"][quantity])
            linestyle, linewidth = styles[case_idx % len(styles)]
            ax.plot(rho, arr[i], linestyle=linestyle, linewidth=linewidth, label=case["label"])
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
    parser.add_argument(
        "--er-init-mode",
        default="keep",
        choices=["keep", "analytical", "ambipolar_min_entropy"],
        help="Override profiles.er_initialization_mode for all native fluxes-mode runs.",
    )
    parser.add_argument(
        "--face-modes",
        nargs="+",
        default=DEFAULT_FACE_MODES,
        choices=["interpolate_center_response", "face_local_response"],
        help="Exact-runtime face response modes to compare in the same run.",
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        default=DEFAULT_RESOLUTIONS,
        help="Exact-runtime resolution sweep entries in 'n_theta,n_zeta,n_xi' format.",
    )
    args = parser.parse_args()

    db_cfg = Path(args.database_config)
    ex_cfg = Path(args.exact_config)
    resolutions = [_parse_resolution(spec) for spec in args.resolutions]

    print(f"[flux-compare] device={args.device}")
    print(f"[flux-compare] database_config={db_cfg}")
    print(f"[flux-compare] exact_config={ex_cfg}")
    print(f"[flux-compare] er_init_mode={args.er_init_mode}")
    print(f"[flux-compare] face_modes={args.face_modes}")
    print(f"[flux-compare] resolutions={resolutions}")

    compare_output_dir = Path("outputs/benchmark_fluxes_compare")
    compare_output_dir.mkdir(parents=True, exist_ok=True)

    rho_db, fluxes_db, dt_db, out_db = _run_config(
        db_cfg,
        device=args.device,
        er_init_mode=args.er_init_mode,
        flux_model="ntx_database",
        output_dir=compare_output_dir / "native_database",
    )

    cases = [
        {
            "label": "database",
            "rho": rho_db,
            "fluxes": fluxes_db,
            "wall_s": dt_db,
            "native_output": out_db,
            "rho_delta": 0.0,
            "flux_delta": 0.0,
        }
    ]

    for face_mode in args.face_modes:
        for resolution in resolutions:
            n_theta, n_zeta, n_xi = resolution
            label = f"exact:{face_mode}:({n_theta},{n_zeta},{n_xi})"
            slug = f"{face_mode}_{n_theta}_{n_zeta}_{n_xi}"
            rho_ex, fluxes_ex, dt_ex, out_ex = _run_config(
                ex_cfg,
                device=args.device,
                er_init_mode=args.er_init_mode,
                flux_model="ntx_exact_lij_runtime",
                face_mode=face_mode,
                resolution=resolution,
                output_dir=compare_output_dir / f"native_{slug}",
            )
            cases.append(
                {
                    "label": label,
                    "rho": rho_ex,
                    "fluxes": fluxes_ex,
                    "wall_s": dt_ex,
                    "native_output": out_ex,
                    "rho_delta": float(jnp.max(jnp.abs(rho_ex - rho_db))),
                    "flux_delta": _max_delta(fluxes_db, fluxes_ex),
                }
            )

    plot_paths = {}
    for quantity in ("Gamma", "Q", "Upar", "Gamma_neo", "Q_neo", "Upar_neo"):
        if all(quantity in case["fluxes"] for case in cases):
            plot_paths[quantity] = _plot_quantity(compare_output_dir, np.asarray(rho_db), cases, quantity)

    print()
    print("case                                       wall_s    rho_max_delta    flux_max_delta")
    print("------------------------------------------------------------------------------------")
    for case in cases:
        print(
            f"{case['label']:<42}"
            f"{case['wall_s']:>8.3f}"
            f"{case['rho_delta']:>17.6e}"
            f"{case['flux_delta']:>18.6e}"
        )

    print()
    print(f"[flux-compare] output_dir={compare_output_dir}")
    for case in cases:
        if case["native_output"] is not None:
            print(f"[flux-compare] native_output[{case['label']}]={case['native_output']}")
    for key, path in plot_paths.items():
        if path is not None:
            print(f"[flux-compare] plot_{key}={path}")


if __name__ == "__main__":
    main()
