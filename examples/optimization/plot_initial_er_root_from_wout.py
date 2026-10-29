#!/usr/bin/env python
"""Calculate ambipolar Er and momentum-corrected bootstrap current from a wout.

The wout supplies geometry; species and prescribed density/temperature profiles
come from a TOML config, or use --wout-pressure for electron/proton profiles
derived from its pressure with density proportional to field strength.
Both quantities use the same selected-root state, with
no optimization, transport time evolution or reverse AD. Use --no-bootstrap for
the original Er-only output. No black root-transition marker is drawn.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
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


def config_with_wout_pressure(config: dict, wout_path: Path, *, field_name: str = "b0"):
    """Build electron/proton profiles satisfying total VMEC pressure [Pa].

    Density scales linearly with |B|: ne(0) = 4.21e20 * |B|/9 [m^-3].
    Assume ne=ni, Te=Ti and n(s)/n(0)=sqrt(p(s)/p(0)); then the pressure
    determines T(s). Prescribed arrays use NEOPAX's physical m^-3/eV units.
    This reads a netCDF file only; it builds no geometry or transport runtime.
    """
    from netCDF4 import Dataset
    from NEOPAX._constants import elementary_charge, electron_mass, proton_mass

    if field_name not in {"b0", "volavgB"}:
        raise ValueError("The wout field must be 'b0' or 'volavgB'.")
    with Dataset(wout_path, "r") as dataset:
        if "presf" not in dataset.variables or field_name not in dataset.variables:
            raise ValueError(f"Wout pressure profiles require presf [Pa] and {field_name} [T].")
        pressure = np.asarray(np.ma.filled(dataset.variables["presf"][:], np.nan), dtype=float)
        field = np.asarray(np.ma.filled(dataset.variables[field_name][...], np.nan), dtype=float)
        s = (np.asarray(np.ma.filled(dataset.variables["phi"][:], np.nan), dtype=float)
             if "phi" in dataset.variables else np.linspace(0., 1., pressure.size))
    if field.size != 1 or not np.isfinite(field).all() or float(np.abs(field).item()) <= 0.:
        raise ValueError(f"Wout {field_name} must be a finite nonzero scalar in tesla.")
    b_ref = float(np.abs(field).item())
    if pressure.ndim != 1 or pressure.size < 2 or not np.isfinite(pressure).all():
        raise ValueError("Wout presf must be a finite one-dimensional full-mesh pressure profile.")
    if pressure[0] <= 0. or np.min(pressure) < -1e-12 * max(1., np.max(np.abs(pressure))):
        raise ValueError("Wout pressure must be nonnegative with positive on-axis pressure; vacuum pressure cannot set temperatures.")
    pressure = np.maximum(pressure, 0.)
    if s.shape != pressure.shape or not np.isfinite(s).all() or s[-1] == 0.:
        raise ValueError("Wout phi must be finite and match the presf full mesh, with nonzero edge flux.")
    s = s / s[-1]  # VMEC s is normalized toroidal flux; NEOPAX rho=sqrt(s).
    if abs(s[0]) > 1e-10 or np.any(np.diff(s) <= 0.):
        raise ValueError("Wout normalized toroidal flux must increase from zero to one.")
    s[0] = 0.

    result = copy.deepcopy(config)
    geometry = result.setdefault("geometry", {})
    n_radial = int(geometry.get("n_radial", 51))
    rho_edge = float(geometry.get("rho_edge", 1.))
    if n_radial < 2 or not np.isfinite(rho_edge) or not 0. < rho_edge <= 1.:
        raise ValueError("Pressure-derived profiles require n_radial >= 2 and 0 < rho_edge <= 1.")
    # Match the file-geometry finite-volume cell centers without constructing it.
    rho = (np.arange(n_radial) + .5) * rho_edge / n_radial
    p_cells = np.interp(rho**2, s, pressure)
    if np.any(p_cells <= 0.):
        raise ValueError("Wout pressure must be positive at all sampled cell centers to calculate finite-density/temperature fluxes.")
    ne_axis = 4.21e20 * b_ref / 9.
    ne = ne_axis * np.sqrt(p_cells / pressure[0])
    te_ev = p_cells / (2. * ne * float(elementary_charge))

    result["species"] = dict(n_species=2, names=["e", "H"],
                             mass_mp=[float(electron_mass / proton_mass), 1.], charge_qp=[-1., 1.])
    result["profiles"] = dict(model="prescribed", density=[ne.tolist(), ne.tolist()],
                              temperature=[te_ev.tolist(), te_ev.tolist()], Er=np.zeros(n_radial).tolist(),
                              er_initialization_mode="ambipolar_min_entropy")
    # Strip old species-specific boundaries, sources and transport toggles: this
    # is a fixed-profile ambipolarity calculation with exactly two species.
    result.setdefault("general", {})["mode"] = "ambipolarity"
    result["boundary"] = {}
    result["sources"] = {}
    result["turbulence"] = {"flux_model": "none"}
    result["classical"] = {"flux_model": "none"}
    for key in ("equations", "optimization", "transport_output"):
        result.pop(key, None)
    result["wout_pressure"] = dict(
        source=str(wout_path.resolve()), field_variable=field_name, field_abs_T=b_ref,
        reference_field_T=9., reference_ne_axis_m3=4.21e20, ne_axis_m3=ne_axis,
        pressure_axis_Pa=float(pressure[0]), Te_axis_keV=float(pressure[0] / (2. * ne_axis * elementary_charge * 1e3)),
        density_shape="sqrt(p(s)/p(0))", equal_electron_ion_temperature=True,
        rho=rho.tolist(), pressure_Pa=p_cells.tolist(),
    )
    return result


def save_wout_pressure_config(config: dict, output_path: Path) -> Path:
    """Write a reusable full TOML and a physical-units profile CSV."""
    import toml

    config_path = output_path.with_name(output_path.stem + "_profiles.toml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(toml.dumps(config), encoding="utf-8")
    info = config["wout_pressure"]
    density = np.asarray(config["profiles"]["density"])
    temperature = np.asarray(config["profiles"]["temperature"])
    csv_path = config_path.with_suffix(".csv")
    rho = np.asarray(info["rho"])
    np.savetxt(csv_path, np.column_stack([rho, rho**2, info["pressure_Pa"],
                                         density.T, temperature.T / 1e3]), delimiter=",",
               header="rho,s,pressure_Pa,ne_m3,np_m3,Te_keV,Tp_keV", comments="")
    print(f"[wout-pressure] |{info['field_variable']}|={info['field_abs_T']:.9g} T "
          f"ne_axis=np_axis={info['ne_axis_m3']:.9e} m^-3 Te_axis=Tp_axis={info['Te_axis_keV']:.9g} keV")
    print(f"[wout-pressure] wrote {config_path}")
    print(f"[wout-pressure] wrote {csv_path}")
    return config_path


def solve_initial_er_root_from_wout(
    wout_path: Path,
    *,
    config_path: Path,
    boozer_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preserve the original Er-only Python interface."""
    rho, er, _, finite_mask = solve_initial_er_and_bootstrap_from_wout(
        wout_path, config_path=config_path, boozer_path=boozer_path,
        include_bootstrap=False,
    )
    return rho, er, finite_mask


def solve_initial_er_and_bootstrap_from_wout(
    wout_path: Path,
    *,
    config_path: Path,
    boozer_path: Path,
    include_bootstrap: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Return rho, Er [kV/m], Jboot [1e5 A/m^2] and the finite-root mask.

    Bootstrap current is NaN where no finite ambipolar root was found. The Er
    array retains the root selector's fallback at those radii for compatibility;
    callers must use the mask. Only one runtime and one root solve are needed.
    """
    config = _config_for_wout(config_path, wout_path, boozer_path)
    runtime, state = build_runtime_context(config)
    if state is None:
        raise RuntimeError("NEOPAX runtime did not build an initial transport state.")
    er_profile, finite_mask = initial_er_selected_root_profile(state, config=config, runtime=runtime)
    rho = getattr(runtime.geometry, "rho_grid", None)
    if rho is None:
        raise RuntimeError("Runtime geometry did not expose rho_grid.")
    finite_mask = np.asarray(jax.device_get(finite_mask), dtype=bool)
    jboot = None
    if include_bootstrap:
        if not np.any(finite_mask):
            raise RuntimeError("No finite ambipolar roots were found; cannot calculate bootstrap current.")
        # Reuse the optimizer's corrected-flux evaluator and unit conversion,
        # without constructing or evaluating an optimization problem.
        from NEOPAX.optimization import GeometryInitialErRootLeastSquaresProblem

        rooted_state = dataclasses.replace(state, Er=er_profile)
        jboot = GeometryInitialErRootLeastSquaresProblem._bootstrap_current_from_rooted_state_runtime(
            rooted_state, runtime
        )
        jboot = np.asarray(jax.device_get(jboot), dtype=float)
        if jboot.shape != finite_mask.shape:
            raise ValueError("Bootstrap current and ambipolar-root radial shapes differ.")
        if np.any(finite_mask & ~np.isfinite(jboot)):
            raise FloatingPointError("Nonfinite momentum-corrected bootstrap current at a finite ambipolar root.")
        jboot = np.where(finite_mask, jboot, np.nan)
    return (
        np.asarray(jax.device_get(rho), dtype=float),
        np.asarray(jax.device_get(er_profile), dtype=float),
        jboot,
        finite_mask,
    )


def plot_er_profile(rho: np.ndarray, er: np.ndarray, output_path: Path, *, label: str) -> None:
    _plot_profile(rho, er, output_path, label=label, color="red",
                  ylabel=r"$E_r$ [$\mathrm{kV}/\mathrm{m}$]")


def plot_bootstrap_profile(rho: np.ndarray, jboot_scaled: np.ndarray, output_path: Path) -> None:
    _plot_profile(rho, 100.0 * jboot_scaled, output_path,
                  label="momentum-corrected bootstrap", color="tab:blue",
                  ylabel=r"$J^{\mathrm{bootstrap}}$ [$\mathrm{kA}/\mathrm{m}^2$]")


def _plot_profile(rho, values, output_path, *, label, color, ylabel) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    ax.plot(rho, values, color=color, linewidth=3.2, solid_capstyle="round", label=label)
    ax.set_xlabel(r"$\rho$", fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
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


def main(argv=None) -> int:
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
    parser.add_argument("--out", type=Path, default=None, help="Er output PNG path; CSV uses the same stem.")
    parser.add_argument("--bootstrap", action=argparse.BooleanOptionalAction, default=True,
                        help="Also calculate momentum-corrected bootstrap current (default: enabled).")
    parser.add_argument("--bootstrap-out", type=Path, default=None,
                        help="Bootstrap PNG path; default <Er output stem>_bootstrap_current.png.")
    parser.add_argument("--wout-pressure", action="store_true",
                        help="Use wout pressure for e + proton profiles, with ne(0)=4.21e20*|B|/9 and Te=Tp.")
    parser.add_argument("--wout-field", choices=("b0", "volavgB"), default="b0",
                        help="Wout field scalar for density scaling; b0 is the axis toroidal field (default: b0).")
    parser.add_argument("--profiles-only", action="store_true",
                        help="With --wout-pressure, write profiles TOML/CSV without Boozer, NTX or ambipolarity solves.")
    parser.add_argument("--label", default="Er", help="Legend label. Use empty string to hide the legend.")
    args = parser.parse_args(argv)
    if args.bootstrap_out is not None and not args.bootstrap:
        parser.error("--bootstrap-out requires --bootstrap.")
    if args.profiles_only and not args.wout_pressure:
        parser.error("--profiles-only requires --wout-pressure.")

    wout_path = args.wout.resolve()
    if not wout_path.exists():
        raise FileNotFoundError(wout_path)
    config_path = args.config.resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    output_path = args.out
    if output_path is None:
        output_path = wout_path.parent / f"{wout_path.stem}_initial_er_root.png"
    output_path = output_path.resolve()
    bootstrap_path = (args.bootstrap_out or output_path.with_name(
        output_path.stem + "_bootstrap_current.png"
    )).resolve()
    if args.bootstrap and (bootstrap_path == output_path or bootstrap_path.with_suffix(".csv") == output_path.with_suffix(".csv")):
        parser.error("Er and bootstrap output paths must be distinct.")

    pressure_config = None
    if args.wout_pressure:
        # This path is only a future geometry reference in --profiles-only mode;
        # generating Boozer coordinates is unnecessary just to export profiles.
        candidates = _candidate_boozer_files(wout_path)
        case = wout_path.stem.removeprefix("wout_")
        planned_boozer = (args.boozer_file or (candidates[0] if candidates else
                          wout_path.parent / f"boozmn_{case}.nc")).resolve()
        pressure_config = config_with_wout_pressure(
            _config_for_wout(config_path, wout_path, planned_boozer), wout_path, field_name=args.wout_field,
        )
        if args.profiles_only:
            save_wout_pressure_config(pressure_config, output_path)
            return 0
    boozer_path = _resolve_boozer_file(
        wout_path, args.boozer_file,
        generate_boozer=bool(args.generate_boozer), mboz=int(args.mboz), nboz=int(args.nboz),
        jit_boozer=bool(args.jit_boozer), verbose_boozer=bool(args.verbose_boozer),
    )
    if pressure_config is not None:
        pressure_config["geometry"]["boozer_file"] = str(boozer_path)
        config_path = save_wout_pressure_config(pressure_config, output_path)

    print(f"[initial-er-root-plot] wout={wout_path}")
    print(f"[initial-er-root-plot] boozer_file={boozer_path}")
    print(f"[initial-er-root-plot] config={config_path}")
    rho, er, jboot, finite_mask = solve_initial_er_and_bootstrap_from_wout(
        wout_path,
        config_path=config_path,
        boozer_path=boozer_path,
        include_bootstrap=args.bootstrap,
    )
    print(f"[initial-er-root-plot] finite ambipolar roots={np.count_nonzero(finite_mask)}/{finite_mask.size}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_path.with_suffix(".csv")
    np.savetxt(
        csv_path,
        np.column_stack([rho, er, finite_mask.astype(float)]),
        delimiter=",",
        header="rho,Er,finite_mask",
        comments="",
    )
    # Do not display the initializer's fallback Er as a solved ambipolar root.
    plotted_er = np.where(finite_mask, er, np.nan)
    plot_er_profile(rho, plotted_er, output_path, label=args.label)
    print(f"[initial-er-root-plot] wrote {csv_path}")
    print(f"[initial-er-root-plot] wrote {output_path}")
    if jboot is not None:
        bootstrap_path.parent.mkdir(parents=True, exist_ok=True)
        bootstrap_csv = bootstrap_path.with_suffix(".csv")
        np.savetxt(
            bootstrap_csv,
            np.column_stack([rho, plotted_er, jboot * 1.0e5, jboot * 100.0,
                             jboot, finite_mask.astype(float)]),
            delimiter=",",
            header="rho,Er_kV_m,Jboot_A_m2,Jboot_kA_m2,Jboot_scaled_1e5_A_m2,finite_mask",
            comments="",
        )
        plot_bootstrap_profile(rho, jboot, bootstrap_path)
        print(f"[initial-er-root-plot] wrote {bootstrap_csv}")
        print(f"[initial-er-root-plot] wrote {bootstrap_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
