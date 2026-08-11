#!/usr/bin/env python
"""Solve initial ambipolar Er from a wout file and plot it like optimization scripts.

The wout supplies the geometry. The transport/ambipolarity/profile settings are
taken from a TOML config. No black root-transition marker is drawn.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from NEOPAX._orchestrator import build_runtime_context, load_config  # noqa: E402
from NEOPAX._reverse_ad_initial_er import initial_er_selected_root_profile  # noqa: E402


DEFAULT_CONFIG = (
    ROOT
    / "examples"
    / "Solve_Transport_Equations"
    / "Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime.toml"
)


def _candidate_boozer_files(wout_path: Path) -> list[Path]:
    names = [
        f"boozermn_{wout_path.name}",
        f"boozmn_{wout_path.name}",
        f"boozermn_{wout_path.stem}.nc",
        f"boozmn_{wout_path.stem}.nc",
    ]
    if wout_path.name.startswith("wout_"):
        suffix = wout_path.name[len("wout_") :]
        names.extend([f"boozermn_wout_{suffix}", f"boozmn_wout_{suffix}", f"boozermn_{suffix}", f"boozmn_{suffix}"])

    directories = [wout_path.parent, wout_path.parent.parent, ROOT / "examples" / "inputs"]
    candidates: list[Path] = []
    for directory in directories:
        if not directory.exists():
            continue
        for name in names:
            candidate = directory / name
            if candidate.exists() and candidate not in candidates:
                candidates.append(candidate)
    return candidates


def _generate_boozer_file(
    wout_path: Path,
    *,
    mboz: int,
    nboz: int,
    jit_boozer: bool,
    verbose: bool,
) -> Path:
    from vmex.core.boozer import run_booz_xform

    print(
        "[initial-er-root-plot] matching Boozer file not found; "
        f"generating boozmn from wout mboz={int(mboz)} nboz={int(nboz)}"
    )
    return run_booz_xform(
        wout_path,
        mbooz=int(mboz),
        nbooz=int(nboz),
        surfaces=None,
        outdir=wout_path.parent,
        jit=bool(jit_boozer),
        verbose=bool(verbose),
    ).resolve()


def _resolve_boozer_file(
    wout_path: Path,
    explicit_boozer: Path | None,
    *,
    generate_boozer: bool,
    mboz: int,
    nboz: int,
    jit_boozer: bool,
    verbose_boozer: bool,
) -> Path:
    if explicit_boozer is not None:
        resolved = explicit_boozer.resolve()
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        return resolved
    candidates = _candidate_boozer_files(wout_path)
    if candidates:
        return candidates[0].resolve()
    if generate_boozer:
        return _generate_boozer_file(
            wout_path,
            mboz=mboz,
            nboz=nboz,
            jit_boozer=jit_boozer,
            verbose=verbose_boozer,
        )
    raise FileNotFoundError(
        "Could not find matching Boozer file for "
        f"{wout_path}. Pass --boozer-file explicitly or keep --generate-boozer enabled."
    )


def _config_for_wout(config_path: Path, wout_path: Path, boozer_path: Path) -> dict:
    config = copy.deepcopy(load_config(config_path))
    config.setdefault("general", {})["mode"] = "ambipolarity"
    geometry = config.setdefault("geometry", {})
    # Force the non-AD file-geometry runtime: this script solves Er for a fixed wout.
    for key in ("backend", "vmec_input_file", "vmec_lane", "vmec_param_delta"):
        geometry.pop(key, None)
    geometry["vmec_file"] = str(wout_path)
    geometry["boozer_file"] = str(boozer_path)
    return config


def solve_initial_er_root_from_wout(
    wout_path: Path,
    *,
    config_path: Path,
    boozer_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    config = _config_for_wout(config_path, wout_path, boozer_path)
    runtime, state = build_runtime_context(config)
    if state is None:
        raise RuntimeError("NEOPAX runtime did not build an initial transport state.")
    er_profile, finite_mask = initial_er_selected_root_profile(state, config=config, runtime=runtime)
    rho = getattr(runtime.geometry, "rho_grid", None)
    if rho is None:
        raise RuntimeError("Runtime geometry did not expose rho_grid.")
    return (
        np.asarray(jax.device_get(rho), dtype=float),
        np.asarray(jax.device_get(er_profile), dtype=float),
        np.asarray(jax.device_get(finite_mask), dtype=bool),
    )


def plot_er_profile(rho: np.ndarray, er: np.ndarray, output_path: Path, *, label: str) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    ax.plot(rho, er, color="red", linewidth=3.2, solid_capstyle="round", label=label)
    ax.set_xlabel(r"$\rho$", fontsize=20)
    ax.set_ylabel(r"$E_r$ [$\mathrm{kV}/\mathrm{m}$]", fontsize=20)
    ax.tick_params(axis="both", labelsize=16, width=1.0, length=4)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("0.35")
    ax.margins(x=0.04, y=0.08)
    if label:
        ax.legend(loc="lower left", fontsize=15, frameon=True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wout", type=Path, help="Path to wout_*.nc.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Transport/ambipolarity TOML config.")
    parser.add_argument("--boozer-file", type=Path, default=None, help="Matching boozermn/boozmn file.")
    parser.add_argument(
        "--generate-boozer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate boozmn_<case>.nc from the wout if no matching Boozer file exists.",
    )
    parser.add_argument("--mboz", type=int, default=18, help="Generated Boozer poloidal resolution.")
    parser.add_argument("--nboz", type=int, default=18, help="Generated Boozer toroidal resolution.")
    parser.add_argument("--jit-boozer", action="store_true", help="Use JIT inside booz_xform_jax generation.")
    parser.add_argument("--verbose-boozer", action="store_true", help="Print booz_xform_jax generation details.")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path.")
    parser.add_argument("--label", default="Er", help="Legend label. Use empty string to hide the legend.")
    args = parser.parse_args()

    wout_path = args.wout.resolve()
    if not wout_path.exists():
        raise FileNotFoundError(wout_path)
    config_path = args.config.resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    boozer_path = _resolve_boozer_file(
        wout_path,
        args.boozer_file,
        generate_boozer=bool(args.generate_boozer),
        mboz=int(args.mboz),
        nboz=int(args.nboz),
        jit_boozer=bool(args.jit_boozer),
        verbose_boozer=bool(args.verbose_boozer),
    )
    output_path = args.out
    if output_path is None:
        output_path = wout_path.parent / f"{wout_path.stem}_initial_er_root.png"
    output_path = output_path.resolve()

    print(f"[initial-er-root-plot] wout={wout_path}")
    print(f"[initial-er-root-plot] boozer_file={boozer_path}")
    print(f"[initial-er-root-plot] config={config_path}")
    rho, er, finite_mask = solve_initial_er_root_from_wout(
        wout_path,
        config_path=config_path,
        boozer_path=boozer_path,
    )
    csv_path = output_path.with_suffix(".csv")
    np.savetxt(
        csv_path,
        np.column_stack([rho, er, finite_mask.astype(float)]),
        delimiter=",",
        header="rho,Er,finite_mask",
        comments="",
    )
    plot_er_profile(rho, er, output_path, label=args.label)
    print(f"[initial-er-root-plot] wrote {csv_path}")
    print(f"[initial-er-root-plot] wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
