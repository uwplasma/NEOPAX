#!/usr/bin/env python
"""Compare nonlinear adiabatic-electron GKX heat flux on saved equilibria.

Read initial/ and optimized/ wout files from a NEOPAX optimization directory.
Both runs use the same template and imposed ion a/L_T and a/L_n. The surface
argument is s (normalized toroidal flux), not rho=sqrt(s). No equilibrium or
transport optimization is run. --prepare-only writes inputs without invoking GKX.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.optimization import run_gkx_optimized_flux_scan as scan


def _find_wout(folder: Path, label: str, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label} wout: {path}")
        return path
    directory = folder / label
    preferred = directory / f"wout_QI_neopax_geometry_full_transport_{label}.nc"
    if preferred.is_file():
        return preferred.resolve()
    choices = sorted(path.resolve() for path in directory.glob("wout*.nc")
                     if not path.name.endswith((".eik.nc", ".out.nc", ".big.nc", ".restart.nc")))
    if len(choices) == 1:
        return choices[0]
    raise ValueError(
        f"Expected one equilibrium wout in {directory}, found {len(choices)}. "
        f"Select it explicitly with --{label}-wout. Candidates: {choices}"
    )


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _comparison_template(template: dict, alpha: float) -> dict:
    data = copy.deepcopy(template)
    if float(data["species"][0].get("charge", 0.)) <= 0.:
        raise ValueError("The adiabatic-electron template must contain one positive-ion species.")
    data.setdefault("geometry", {})["alpha"] = alpha
    data.setdefault("physics", {}).update(electrostatic=True, electromagnetic=False, beta=0.)
    data.setdefault("terms", {}).update(apar=0., bpar=0.)
    # The comparison is a fresh initial-value problem for each equilibrium.
    # Explicit step caps in tutorial templates must not override --t-max.
    data.setdefault("run", {}).pop("steps", None)
    data["run"]["diagnostics"] = True
    for key in ("init_file", "init_file_scale", "init_file_mode"):
        data.setdefault("init", {}).pop(key, None)
    return data


def _late_statistics(time, flux, fraction):
    """Time-weighted statistics; adaptive samples need not be equally spaced."""
    import numpy as np

    time, flux = np.asarray(time, dtype=float), np.asarray(flux, dtype=float)
    if len(time) < 2 or not np.all(np.diff(time) > 0.):
        raise ValueError("Need at least two heat-flux samples with strictly increasing times.")
    if not np.isfinite(time).all() or not np.isfinite(flux).all():
        raise ValueError("Nonfinite heat-flux history cannot be compared.")
    start = float(time[-1] - fraction*(time[-1]-time[0]))
    later = time > start
    t = np.r_[start, time[later]]
    q = np.r_[np.interp(start, time, flux), flux[later]]
    dt = np.diff(t)
    duration = t[-1]-t[0]
    mean = float(np.sum(dt*(q[:-1]+q[1:])*.5)/duration)
    # Exact variance of the piecewise-linear interpolant, rather than a
    # sample-count weighted variance that favors small adaptive timesteps.
    a, b = q[:-1]-mean, q[1:]-mean
    variance = float(np.sum(dt*(a*a+a*b+b*b)/3.)/duration)
    return dict(t_end=float(time[-1]), average_window_start=start,
                average_samples=int(np.count_nonzero(time >= start)),
                ion_heat_flux_mean=mean, ion_heat_flux_std=math.sqrt(max(variance, 0.)),
                ion_heat_flux_last=float(flux[-1]))


def _write_comparison(points: list[dict], output_dir: Path, fraction: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    histories, summaries = [], []
    for point in points:
        time, flux = scan._read_heat_flux(Path(point["output"]))
        stats = _late_statistics(time, flux, fraction)
        histories.append((point, time, flux))
        summaries.append({**point, **stats})
    with (output_dir / "heat_flux_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("equilibrium", "s", "alpha", "time", "ion_heat_flux"))
        for point, time, flux in histories:
            writer.writerows((point["equilibrium"], point["s"], point["alpha"], t, q)
                             for t, q in zip(time, flux, strict=True))
    with (output_dir / "heat_flux_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    initial, optimized = (row["ion_heat_flux_mean"] for row in summaries)
    report = dict(cases=summaries, optimized_minus_initial=optimized-initial,
                  optimized_over_initial=None if initial == 0. else optimized/initial)
    (output_dir / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for index, (point, time, flux) in enumerate(histories):
        axes[0].plot(time, flux, label=point["equilibrium"])
        axes[1].bar(index, summaries[index]["ion_heat_flux_mean"],
                    yerr=summaries[index]["ion_heat_flux_std"], capsize=4)
    axes[0].set(xlabel="GKX normalized time", ylabel="Ion heat flux [GKX normalization]")
    axes[0].legend()
    axes[1].set(xticks=[0, 1], xticklabels=["initial", "optimized"],
                ylabel="Late-time mean heat flux (bars: temporal std)")
    point = points[0]
    fig.suptitle(f"Adiabatic electrons: s={point['s']:g}, alpha={point['alpha']:g}, "
                 f"a/L_T={point['tprim']:g}, a/L_n={point['fprim']:g}")
    fig.savefig(output_dir / "heat_flux_comparison.png", dpi=200)
    plt.close(fig)
    print(f"[adiabatic comparison] mean heat flux: initial={initial:.8g} optimized={optimized:.8g}", flush=True)
    print(f"[adiabatic comparison] wrote comparison.json, CSVs and heat_flux_comparison.png in {output_dir}")


def parser():
    out = argparse.ArgumentParser(description=__doc__)
    out.add_argument("output_folder", type=Path, help="Geometry-only optimization output with initial/ and optimized/ wouts")
    out.add_argument("--initial-wout", type=Path)
    out.add_argument("--optimized-wout", type=Path)
    out.add_argument("--s", type=float, default=.5, help="Normalized toroidal flux s, not rho (default: 0.5)")
    out.add_argument("--LT", "--tprim", dest="tprim", type=float, default=3., help="Ion normalized inverse temperature gradient a/L_T (default: 3)")
    out.add_argument("--Ln", "--fprim", dest="fprim", type=float, default=0., help="Ion normalized inverse density gradient a/L_n (default: 0)")
    out.add_argument("--alpha", type=float, default=0., help="Field-line label (default: 0)")
    out.add_argument("--t-max", type=float, default=700., help="Fixed normalized GKX horizon (default: 700)")
    out.add_argument("--average-fraction", type=float, default=.2, help="Final fraction of each history to average (default: 0.2)")
    out.add_argument("--gkx-root", type=Path, default=scan._default_gkx_root())
    out.add_argument("--gkx-python", type=Path, default=Path(sys.executable))
    out.add_argument("--template", type=Path, help="Numerical settings; default GKX/examples/common_input.toml")
    out.add_argument("--out-dir", "--output-dir", dest="output_dir", type=Path)
    mode = out.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true", help="Write both decks and manifest; do not invoke GKX")
    mode.add_argument("--plot-only", action="store_true", help="Compare existing outputs; do not invoke GKX")
    return out


def main(argv=None):
    args = parser().parse_args(argv)
    if not all(math.isfinite(v) for v in (args.s, args.alpha, args.tprim, args.fprim, args.t_max, args.average_fraction)):
        raise ValueError("Surface, gradients, alpha, time and averaging fraction must be finite.")
    if not 0. < args.s < 1. or args.t_max <= 0. or not 0. < args.average_fraction <= 1.:
        raise ValueError("Require 0 < s < 1, t_max > 0, and 0 < average-fraction <= 1.")
    folder = args.output_folder.expanduser().resolve()
    gkx_root = args.gkx_root.expanduser().resolve()
    template_path = (args.template or gkx_root / "examples/common_input.toml").expanduser().resolve()
    template = _comparison_template(scan._load_template(template_path), args.alpha)
    wouts = [_find_wout(folder, label, path) for label, path in
             (("initial", args.initial_wout), ("optimized", args.optimized_wout))]
    if wouts[0] == wouts[1]:
        raise ValueError("Initial and optimized wout paths must be distinct.")
    tag = f"s_{args.s:g}_LT_{args.tprim:g}_Ln_{args.fprim:g}_alpha_{args.alpha:g}".replace(".", "p")
    output_dir = (args.output_dir or folder / "gkx_adiabatic_comparison" / tag).expanduser().resolve()
    points = [dict(equilibrium=label, s=args.s, rho=math.sqrt(args.s), alpha=args.alpha,
                   tprim=args.tprim, fprim=args.fprim, wout=str(wout), wout_sha256=_sha256(wout),
                   deck=str(output_dir / label / "gkx.toml"), output=str(output_dir / label / "gkx.out.nc"))
              for label, wout in zip(("initial", "optimized"), wouts, strict=True)]
    manifest = dict(template=str(template_path), template_sha256=_sha256(template_path),
                    t_max=args.t_max, points=points)
    manifest_path = output_dir / "case_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Saved comparison inputs differ; choose a fresh --out-dir to avoid reusing unrelated GKX output.")
    elif any(Path(point["output"]).exists() for point in points):
        raise ValueError("Existing GKX outputs have no comparison manifest; choose a fresh --out-dir.")
    if not args.plot_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
        for point in points:
            scan._prepare_deck(template, wout=Path(point["wout"]), rho=point["rho"], torflux=args.s,
                               t_max=args.t_max, tprim=args.tprim, fprim=args.fprim,
                               run_dir=output_dir / point["equilibrium"])
            print(f"[adiabatic comparison] {point['equilibrium']}: {point['wout']}; "
                  f"s={args.s:g} alpha={args.alpha:g} a/L_T={args.tprim:g} a/L_n={args.fprim:g}", flush=True)
    if args.prepare_only:
        print(f"[adiabatic comparison] prepared inputs only: {output_dir}")
        return 0
    if not args.plot_only:
        for point in points:
            scan._run_deck(Path(point["deck"]), Path(point["output"]), t_max=args.t_max,
                           gkx_root=gkx_root, python=args.gkx_python.expanduser().resolve(), rerun_complete=False)
    _write_comparison(points, output_dir, args.average_fraction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
