#!/usr/bin/env python
"""Compare final DT alpha-power profiles from two NEOPAX transport HDF5 files."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


FIGSIZE = (6.8, 5.6)
DPI = 320
LINEWIDTH = 3.0
PRESSURE_SOURCE_STATE_TO_MW_M3 = 1.0 / 62.422
DEFAULT_SPECIES_ORDER = ("e", "D", "T")


def _read_dataset(handle: h5py.File, names: tuple[str, ...]) -> np.ndarray | None:
    for name in names:
        if name in handle:
            return np.asarray(handle[name], dtype=float)
    return None


def _final_species_profiles(values: np.ndarray, *, label: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 2:
        return values
    if values.ndim == 3:
        return values[-1]
    raise ValueError(f"{label} must be 2D or 3D with species/radius axes, got shape {values.shape}.")


def _final_radial_profile(values: np.ndarray, *, label: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return values
    if values.ndim == 2:
        return values[-1]
    raise ValueError(f"{label} must be 1D or 2D, got shape {values.shape}.")


def _species_lookup(species_order: str, n_species: int) -> dict[str, int]:
    names = tuple(part.strip() for part in species_order.split(",") if part.strip())
    if len(names) != n_species:
        raise ValueError(
            f"--species-order has {len(names)} entries but the HDF5 profiles have {n_species} species."
        )
    return {name: index for index, name in enumerate(names)}


def _dt_alpha_power_mw_m3(
    density: np.ndarray,
    temperature: np.ndarray,
    *,
    species_order: str,
) -> np.ndarray:
    """Return DT alpha power in MW/m^3 using the same analytical source formula."""
    density = _final_species_profiles(density, label="density")
    temperature = _final_species_profiles(temperature, label="temperature")
    if density.shape != temperature.shape:
        raise ValueError(f"density shape {density.shape} does not match temperature shape {temperature.shape}.")

    species_idx = _species_lookup(species_order, density.shape[0])
    try:
        d_index = species_idx["D"]
        t_index = species_idx["T"]
    except KeyError as exc:
        raise ValueError("--species-order must include D and T to compute DT alpha power.") from exc

    n_d = density[d_index]
    n_t = density[t_index]
    t_t = temperature[t_index]

    safe_t = np.maximum(t_t, 1.0e-300)
    t_inv_cuberoot = safe_t ** (-1.0 / 3.0)
    tt_shifted = safe_t + 1.0134
    wrk = (
        tt_shifted / (1.0 + 6.386e-3 * tt_shifted**2)
        + 1.877 * np.exp(-0.16176 * np.sqrt(safe_t) * safe_t)
    )
    dt_reaction_rate = 8.972e-19 * t_inv_cuberoot**2 * np.exp(-19.94 * t_inv_cuberoot) * wrk
    he_source = 1.0e20 * dt_reaction_rate * n_d * n_t
    return PRESSURE_SOURCE_STATE_TO_MW_M3 * 3.52e3 * he_source


def _wout_volume_weights(wout_path: Path | None, rho: np.ndarray) -> np.ndarray | None:
    if wout_path is None:
        return None
    if not wout_path.exists():
        raise FileNotFoundError(wout_path)
    try:
        from vmex.core.wout import read_wout
    except Exception:
        from vmex import read_wout  # type: ignore

    wout = read_wout(wout_path)
    vp = np.abs(np.asarray(getattr(wout, "vp"), dtype=float))
    if vp.ndim != 1 or vp.size < 2:
        return None
    s_grid = np.linspace(0.0, 1.0, vp.size)
    # VMEC vp is dV/ds. The transport H5 rho coordinate is normalized radius,
    # so use dV/drho proportional to vp(s=rho^2) * 2*rho; constants cancel.
    weights = np.interp(np.clip(rho, 0.0, 1.0) ** 2, s_grid, vp) * 2.0 * np.maximum(rho, 0.0)
    return weights


def _volume_average(
    rho: np.ndarray,
    profile: np.ndarray,
    *,
    handle: h5py.File | None = None,
    wout_path: Path | None = None,
) -> float:
    r_grid = None if handle is None else _read_dataset(handle, ("r_grid", "r"))
    vprime = None if handle is None else _read_dataset(handle, ("Vprime", "V_prime", "vprime"))
    if r_grid is not None and vprime is not None and r_grid.shape == profile.shape and vprime.shape == profile.shape:
        volume = np.trapezoid(vprime, x=r_grid)
        integral = np.trapezoid(profile * vprime, x=r_grid)
        return float(integral / max(abs(float(volume)), 1.0e-300))

    weights = _wout_volume_weights(wout_path, rho)
    if weights is None:
        weights = np.maximum(rho, 0.0)
    volume = np.trapezoid(weights, x=rho)
    integral = np.trapezoid(profile * weights, x=rho)
    return float(integral / max(abs(float(volume)), 1.0e-300))


def read_alpha_power(
    path: Path,
    *,
    wout_path: Path | None,
    species_order: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    with h5py.File(path, "r") as handle:
        rho = _read_dataset(handle, ("rho", "rho_grid"))
        if rho is None:
            raise KeyError(f"{path} does not contain a rho grid dataset.")
        rho = np.asarray(rho, dtype=float)

        alpha = _read_dataset(handle, ("alpha_power_mw_m3", "AlphaPower_mw_m3", "alpha_power", "AlphaPower"))
        avg_series = _read_dataset(handle, ("alpha_power_volume_average_mw_m3", "AlphaPower_volume_average_mw_m3"))
        if alpha is not None:
            profile = _final_radial_profile(alpha, label="alpha power")
            avg = (
                float(np.asarray(avg_series, dtype=float).reshape(-1)[-1])
                if avg_series is not None
                else _volume_average(rho, profile, handle=handle, wout_path=wout_path)
            )
            return rho, profile, avg

        density = _read_dataset(handle, ("density",))
        temperature = _read_dataset(handle, ("temperature",))
        missing = [
            name
            for name, values in (("density", density), ("temperature", temperature))
            if values is None
        ]
        if missing:
            available = ", ".join(sorted(handle.keys()))
            raise KeyError(
                f"{path} lacks saved alpha power and is missing datasets needed to recompute it: {missing}. "
                f"Available top-level datasets: {available}"
            )
        profile = _dt_alpha_power_mw_m3(density, temperature, species_order=species_order)
        avg = _volume_average(rho, profile, handle=handle, wout_path=wout_path)
        return rho, profile, avg


def plot_profiles(
    rho: np.ndarray,
    initial_profile: np.ndarray,
    optimized_profile: np.ndarray,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(rho, initial_profile, color="#1f77b4", linewidth=LINEWIDTH, linestyle="--", label="initial")
    ax.plot(rho, optimized_profile, color="#d62728", linewidth=LINEWIDTH, label="optimized")
    ax.set_xlabel(r"$\rho$", fontsize=20)
    ax.set_ylabel(r"$P_{\alpha}$ [$\mathrm{MW}\,\mathrm{m}^{-3}$]", fontsize=20)
    ax.tick_params(axis="both", labelsize=16, width=1.0, length=4)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("0.35")
    ax.legend(loc="best", fontsize=15, frameon=True)
    ax.margins(x=0.04, y=0.08)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("initial_h5", type=Path, help="Initial transport_solution.h5.")
    parser.add_argument("optimized_h5", type=Path, help="Optimized transport_solution.h5.")
    parser.add_argument("--initial-wout", type=Path, default=None, help="Initial WOUT for volume weights.")
    parser.add_argument("--optimized-wout", type=Path, default=None, help="Optimized WOUT for volume weights.")
    parser.add_argument("--species-order", default=",".join(DEFAULT_SPECIES_ORDER), help="Comma-separated H5 species order.")
    parser.add_argument("--out", type=Path, default=Path("alpha_power_initial_vs_optimized.png"), help="Output PNG path.")
    parser.add_argument("--csv", type=Path, default=None, help="Optional output CSV path.")
    parser.add_argument("--config", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    initial_h5 = args.initial_h5.resolve()
    optimized_h5 = args.optimized_h5.resolve()
    if not initial_h5.exists():
        raise FileNotFoundError(initial_h5)
    if not optimized_h5.exists():
        raise FileNotFoundError(optimized_h5)

    initial_wout = None if args.initial_wout is None else args.initial_wout.resolve()
    optimized_wout = None if args.optimized_wout is None else args.optimized_wout.resolve()
    rho_i, alpha_i, avg_i = read_alpha_power(initial_h5, wout_path=initial_wout, species_order=args.species_order)
    rho_o, alpha_o, avg_o = read_alpha_power(optimized_h5, wout_path=optimized_wout, species_order=args.species_order)
    if rho_i.shape != rho_o.shape or not np.allclose(rho_i, rho_o, rtol=1.0e-10, atol=1.0e-12):
        alpha_o = np.interp(rho_i, rho_o, alpha_o)
        rho = rho_i
    else:
        rho = rho_i

    rel_change = (avg_o - avg_i) / max(abs(avg_i), 1.0e-300)
    output_path = args.out.resolve()
    csv_path = args.csv.resolve() if args.csv is not None else output_path.with_suffix(".csv")

    plot_profiles(rho, alpha_i, alpha_o, output_path)
    np.savetxt(
        csv_path,
        np.column_stack([rho, alpha_i, alpha_o]),
        delimiter=",",
        header="rho,alpha_power_initial_mw_m3,alpha_power_optimized_mw_m3",
        comments="",
    )

    print(f"species_order={args.species_order}")
    print(f"initial_volume_average_alpha_power_mw_m3={avg_i:.16e}")
    print(f"optimized_volume_average_alpha_power_mw_m3={avg_o:.16e}")
    print(f"relative_change=(optimized-initial)/initial={rel_change:.16e}")
    print(f"wrote {csv_path}")
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
