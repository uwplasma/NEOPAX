#!/usr/bin/env python
"""Plot a one-surface |B| contour from a VMEX/VMEC wout file.

The default surface is s=5/51, matching the near-axis diagnostic surface used
in the NEOPAX optimization scripts.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_surface_s(text: str) -> float:
    try:
        value = float(Fraction(text))
    except ValueError:
        value = float(text)
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise argparse.ArgumentTypeError("--surface-s must be finite and in [0, 1].")
    return value


def _surface_index_from_s(wout, surface_s: float) -> tuple[int, float]:
    ns = int(getattr(wout, "ns"))
    if ns < 2:
        return 0, 0.0
    index = int(np.clip(round(surface_s * float(ns - 1)), 0, ns - 1))
    actual_s = float(index) / float(ns - 1)
    return index, actual_s


def _safe_contour_levels(values: np.ndarray, count: int):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return int(count)
    vmin = float(finite.min())
    vmax = float(finite.max())
    if not vmax > vmin:
        pad = max(abs(vmin), 1.0) * 1.0e-8
        vmin -= pad
        vmax += pad
    return np.linspace(vmin, vmax, int(count))


def plot_boozer_b_contour_from_wout(
    wout_path: Path,
    out_dir: Path,
    *,
    surface_s: float,
    surface_index: int | None,
    ntheta: int,
    nphi: int,
    levels: int,
    dpi: int,
) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import vmex as vj
    from vmex.core.plotting import surface_modB

    wout = vj.read_wout(wout_path)
    ns = int(getattr(wout, "ns"))
    if surface_index is None:
        s_index, actual_s = _surface_index_from_s(wout, surface_s)
    else:
        s_index = int(np.clip(surface_index, 0, max(ns - 1, 0)))
        actual_s = 0.0 if ns < 2 else float(s_index) / float(ns - 1)

    theta = np.linspace(0.0, 2.0 * np.pi, int(ntheta))
    phi = np.linspace(0.0, 2.0 * np.pi / int(wout.nfp), int(nphi))
    b_grid = np.asarray(surface_modB(wout, s_index=s_index, theta=theta, phi=phi), dtype=float)

    out_dir.mkdir(parents=True, exist_ok=True)
    surface_tag = f"s_{actual_s:.8f}".replace(".", "p")
    stem = wout_path.stem
    csv_path = out_dir / f"{stem}_B_boozer_contour_{surface_tag}.csv"
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    np.savetxt(
        csv_path,
        np.column_stack([theta_grid.reshape(-1), phi_grid.reshape(-1), b_grid.reshape(-1)]),
        delimiter=",",
        header="theta,phi,B",
        comments="",
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    fig.patch.set_facecolor("white")
    contour = ax.contour(
        phi,
        theta,
        b_grid,
        levels=_safe_contour_levels(b_grid, levels),
        cmap="viridis",
        linewidths=1.0,
    )
    colorbar = fig.colorbar(contour, ax=ax, pad=0.05)
    colorbar.set_label(r"$|B|$ [T]", fontsize=11)
    colorbar.ax.tick_params(labelsize=9)
    ax.set_title(rf"$|B|$ at $s={actual_s:.5f}$ (one field period)", fontsize=12)
    ax.set_xlabel(r"toroidal angle $\phi$", fontsize=11)
    ax.set_ylabel(r"poloidal angle $\theta$", fontsize=11)
    ax.set_yticks([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2.0 * np.pi])
    ax.set_yticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_xlim(float(phi.min()), float(phi.max()))
    ax.set_ylim(float(theta.min()), float(theta.max()))
    ax.tick_params(axis="both", labelsize=9)
    fig.tight_layout()

    png_path = out_dir / f"{stem}_B_boozer_contour_{surface_tag}.png"
    fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)

    print(
        "[boozer-b-contour] "
        f"wout={wout_path} requested_s={surface_s:.8f} "
        f"surface_index={s_index} actual_s={actual_s:.8f}"
    )
    print(f"[boozer-b-contour] wrote {csv_path}")
    print(f"[boozer-b-contour] wrote {png_path}")
    return png_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wout", required=True, type=Path, help="Path to a VMEX/VMEC wout_*.nc file.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "boozer_b_contours",
        help="Directory for the PNG/CSV outputs.",
    )
    parser.add_argument("--surface-s", type=_parse_surface_s, default=_parse_surface_s("5/51"))
    parser.add_argument("--surface-index", type=int, default=None, help="Override --surface-s with an exact index.")
    parser.add_argument("--ntheta", type=int, default=160)
    parser.add_argument("--nphi", type=int, default=180)
    parser.add_argument("--levels", type=int, default=28)
    parser.add_argument("--dpi", type=int, default=320)
    args = parser.parse_args()

    plot_boozer_b_contour_from_wout(
        args.wout,
        args.out_dir,
        surface_s=args.surface_s,
        surface_index=args.surface_index,
        ntheta=args.ntheta,
        nphi=args.nphi,
        levels=args.levels,
        dpi=args.dpi,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
