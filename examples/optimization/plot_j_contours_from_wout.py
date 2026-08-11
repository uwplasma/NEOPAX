#!/usr/bin/env python
"""Plot J-invariant QI contour diagnostics from a VMEX/VMEC wout file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_SURFACES = (
    1 / 51,
    5 / 51,
    10 / 51,
    15 / 51,
    20 / 51,
    25 / 51,
    30 / 51,
    35 / 51,
    40 / 51,
    45 / 51,
    50 / 51,
)
DEFAULT_LAMBDA_SAMPLES = (0.1, 0.3, 0.5, 0.7, 0.9)


def _parse_float_list(text: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in text.split(",") if item.strip())


def _boozer_tables_from_wout(wout_path: Path, *, surfaces, mboz: int, nboz: int, jit_boozer: bool):
    import vmex as vj
    from booz_xform_jax import Booz_xform

    wout = vj.read_wout(wout_path)
    bx = Booz_xform(verbose=0, mboz=int(mboz), nboz=int(nboz))
    bx.read_wout_data(wout)

    s_in = np.asarray(bx.s_in, dtype=float)
    requested_surfaces = np.atleast_1d(np.asarray(surfaces, dtype=float))
    indices = sorted({int(np.argmin(np.abs(s_in - value))) for value in requested_surfaces})
    bx.compute_surfs = indices
    bx.run(jit=bool(jit_boozer))

    bmnc_b = np.asarray(bx.bmnc_b, dtype=float)
    xm_b = np.asarray(bx.xm_b, dtype=float)
    if bmnc_b.shape[0] == xm_b.shape[0]:
        bmnc_b = bmnc_b.T

    iota_b = np.asarray(bx.iota, dtype=float)[indices]
    boozer_i = np.asarray(bx.Boozer_I, dtype=float)
    boozer_g = np.asarray(bx.Boozer_G, dtype=float)
    gi_b = boozer_g + iota_b * boozer_i

    return {
        "bmnc_b": bmnc_b,
        "xm_b": xm_b,
        "xn_b": np.asarray(bx.xn_b, dtype=float),
        "iota_b": iota_b,
        "gi_b": gi_b,
        "s_b": s_in[indices],
        "nfp": int(bx.nfp),
    }


def _j_invariant_from_wout(
    wout_path: Path,
    *,
    surfaces,
    mboz: int,
    nboz: int,
    nphi: int,
    nalpha: int,
    n_bounce: int,
    p_j: float,
    p_lambda: float,
    nphi_int: int,
    jit_boozer: bool,
):
    from vmex.core.omnigenity_j import j_invariant_qi_maxj_residual_from_boozer

    booz = _boozer_tables_from_wout(
        wout_path,
        surfaces=surfaces,
        mboz=mboz,
        nboz=nboz,
        jit_boozer=jit_boozer,
    )
    return j_invariant_qi_maxj_residual_from_boozer(
        bmnc_b=booz["bmnc_b"],
        xm_b=booz["xm_b"],
        xn_b=booz["xn_b"],
        iota_b=booz["iota_b"],
        gi_b=booz["gi_b"],
        s_b=booz["s_b"],
        nfp=booz["nfp"],
        nphi=nphi,
        nalpha=nalpha,
        n_bounce=n_bounce,
        p_j=p_j,
        p_lambda=p_lambda,
        nphi_int=nphi_int,
        include_qi=True,
        include_maxj=False,
    )


def plot_j_polar_contours(out, out_dir: Path, *, p_lambda: float, lambda_samples):
    import matplotlib.pyplot as plt

    alpha = np.asarray(jax.device_get(out["alpha"]), dtype=float)
    surfaces = np.asarray(jax.device_get(out["surfaces"]), dtype=float)
    ji = np.asarray(jax.device_get(out["ji"]), dtype=float)
    jc = np.asarray(jax.device_get(out["jc"]), dtype=float)
    n_bounce = int(ji.shape[-1])
    lambda_grid = np.power(
        np.arange(n_bounce, dtype=float) / max(n_bounce - 1, 1),
        float(p_lambda),
    )

    theta = np.concatenate([alpha, alpha[:1] + 2.0 * np.pi])
    theta_grid, radius_grid = np.meshgrid(theta, surfaces, indexing="xy")
    sample_idx = sorted(
        {
            int(np.clip(round(lam * (n_bounce - 1)), 0, n_bounce - 1))
            for lam in lambda_samples
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, data in (("ji", ji), ("jc", jc)):
        for idx in sample_idx:
            values = data[:, :, idx]
            values_periodic = np.concatenate([values, values[:, :1]], axis=1)
            display_name = r"$\mathcal{J}$" if name == "ji" else r"$J_C$"
            title_name = r"$\mathcal{J}$" if name == "ji" else r"$J_C$"
            fig = plt.figure(figsize=(5.4, 5.8))
            ax_polar = fig.add_subplot(1, 1, 1, projection="polar")
            contour = ax_polar.contourf(theta_grid, radius_grid, values_periodic, levels=40, cmap="plasma")
            ax_polar.set_title(f"Second adiabatic invariant, {title_name}", fontsize=15, pad=20)
            ax_polar.set_ylim(0.0, float(surfaces.max()))
            ax_polar.set_thetagrids(np.arange(0, 360, 45), fontsize=8)
            radial_ticks = np.linspace(0.2, float(surfaces.max()), 5)
            ax_polar.set_rticks(radial_ticks)
            ax_polar.set_yticklabels([f"{tick:.1f}" for tick in radial_ticks], fontsize=8)
            ax_polar.set_rlabel_position(45)
            ax_polar.grid(color="white", linewidth=0.8, alpha=0.45)
            colorbar = fig.colorbar(contour, ax=ax_polar, pad=0.12, shrink=0.78)
            colorbar.set_label(display_name, fontsize=11)
            colorbar.ax.tick_params(labelsize=8)
            fig.text(0.5, 0.035, rf"$\lambda$ = {lambda_grid[idx]:.2f}", ha="center", va="center", fontsize=15)
            fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
            path = out_dir / f"{name}_polar_lambda_{idx:02d}.png"
            fig.savefig(path, dpi=320, bbox_inches="tight")
            plt.close(fig)
            written.append(path)
            print(f"wrote {path}")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wout", type=Path, help="Path to wout_*.nc.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for PNGs.")
    parser.add_argument("--surfaces", default=",".join(f"{value:.16g}" for value in DEFAULT_SURFACES))
    parser.add_argument("--lambda-samples", default=",".join(str(value) for value in DEFAULT_LAMBDA_SAMPLES))
    parser.add_argument("--mboz", type=int, default=18)
    parser.add_argument("--nboz", type=int, default=18)
    parser.add_argument("--nphi", type=int, default=101)
    parser.add_argument("--nalpha", type=int, default=51)
    parser.add_argument("--n-bounce", type=int, default=66)
    parser.add_argument("--p-j", type=float, default=1.0)
    parser.add_argument("--p-lambda", type=float, default=1.0)
    parser.add_argument("--nphi-int", type=int, default=128)
    parser.add_argument("--jit-boozer", action="store_true")
    args = parser.parse_args()

    wout_path = args.wout.resolve()
    if not wout_path.exists():
        raise FileNotFoundError(wout_path)
    out_dir = args.out_dir or (wout_path.parent / f"{wout_path.stem}_j_contours")

    jax.config.update("jax_enable_x64", True)
    surfaces = _parse_float_list(args.surfaces)
    lambda_samples = _parse_float_list(args.lambda_samples)
    print(
        "[j-contours-from-wout] "
        f"wout={wout_path} out_dir={out_dir} surfaces={','.join(f'{s:.3g}' for s in surfaces)}"
    )

    out = _j_invariant_from_wout(
        wout_path,
        surfaces=surfaces,
        mboz=args.mboz,
        nboz=args.nboz,
        nphi=args.nphi,
        nalpha=args.nalpha,
        n_bounce=args.n_bounce,
        p_j=args.p_j,
        p_lambda=args.p_lambda,
        nphi_int=args.nphi_int,
        jit_boozer=args.jit_boozer,
    )
    plot_j_polar_contours(out, out_dir, p_lambda=args.p_lambda, lambda_samples=lambda_samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
