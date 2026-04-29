#!/usr/bin/env python3
"""Convenience wrapper for the external NEOPAX -> SPECTRAX flux workflow.

This script is the NEOPAX-facing entrypoint for Step 1 of the coupling plan.
It reads a NEOPAX transport TOML, finds the corresponding transport HDF5
output, then delegates to ``neopax_spectrax_flux_bridge.py`` to:

- prepare the per-radius SPECTRAX runs,
- execute them,
- collect the final fluxes into one HDF5 file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]

try:
    import NEOPAX.utils.neopax_spectrax_flux_bridge as bridge
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import neopax_spectrax_flux_bridge as bridge  # type: ignore[no-redef]


def _load_toml(path: Path) -> dict:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _infer_neopax_root(config_path: Path) -> Path:
    parent = config_path.resolve().parent
    if parent.parent.name == "examples":
        return parent.parent.parent
    return parent


def _resolve_from_neopax_root(config_path: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    root = _infer_neopax_root(config_path)
    resolved = bridge._resolve_relative(root, value)
    return None if resolved is None else Path(resolved)


def _default_transport_solution_path(config_path: Path, cfg: dict) -> Path:
    output_cfg = cfg.get("transport_output", {})
    output_dir = _resolve_from_neopax_root(config_path, output_cfg.get("transport_output_dir"))
    if output_dir is None:
        raise ValueError("Could not resolve [transport_output].transport_output_dir from the NEOPAX config")
    return output_dir / "transport_solution.h5"


def _warn_if_hdf5_disabled(cfg: dict) -> None:
    output_cfg = cfg.get("transport_output", {})
    if not bool(output_cfg.get("transport_write_hdf5", False)):
        print(
            "warning: [transport_output].transport_write_hdf5 is false; "
            "NEOPAX will not write transport_solution.h5 unless you enable it.",
            file=sys.stderr,
        )


def cmd_all(args: argparse.Namespace) -> int:
    config_path = Path(args.neopax_config).resolve()
    cfg = _load_toml(config_path)
    if str(args.profiles_source).lower() == "transport_h5":
        _warn_if_hdf5_disabled(cfg)

    result_path = None
    if str(args.profiles_source).lower() == "transport_h5":
        result_path = (
            Path(args.neopax_result).resolve()
            if args.neopax_result is not None
            else _default_transport_solution_path(config_path, cfg)
        )
    output_dir = Path(args.output_dir).resolve()

    prepare_args = argparse.Namespace(
        profiles_source=args.profiles_source,
        neopax_result=None if result_path is None else str(result_path),
        neopax_config=str(config_path),
        spectrax_root=args.spectrax_root,
        spectrax_template=args.spectrax_template,
        output_dir=str(output_dir),
        time_index=args.time_index,
        analytical_n_radii=args.analytical_n_radii,
        electron_model=args.electron_model,
        reference_ion=args.reference_ion,
        rho_indices=args.rho_indices,
        rho_min=args.rho_min,
        rho_max=args.rho_max,
        num_radii=args.num_radii,
        vmec_file_override=args.vmec_file_override,
        boozer_file_override=args.boozer_file_override,
        density_floor=args.density_floor,
        temperature_floor=args.temperature_floor,
        gradient_coordinate=args.gradient_coordinate,
        gradient_scale=args.gradient_scale,
        tprim_scale=args.tprim_scale,
        fprim_scale=args.fprim_scale,
        tau_e_override=args.tau_e_override,
        nu_ion=args.nu_ion,
        nu_electron=args.nu_electron,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        lx=args.lx,
        ly=args.ly,
        boundary=args.boundary,
        y0=args.y0,
        ntheta=args.ntheta,
        nperiod=args.nperiod,
        t_max=args.t_max,
        dt=args.dt,
        method=args.method,
        use_diffrax=args.use_diffrax,
        fixed_dt=args.fixed_dt,
        sample_stride=args.sample_stride,
        diagnostics_stride=args.diagnostics_stride,
        cfl=args.cfl,
        state_sharding=args.state_sharding,
        ky=args.ky,
        nl=args.nl,
        nm=args.nm,
        init_field=args.init_field,
        init_amp=args.init_amp,
        alpha=args.alpha,
        npol=args.npol,
        beta=args.beta,
        nu_hermite=args.nu_hermite,
        nu_laguerre=args.nu_laguerre,
        nu_hyper=args.nu_hyper,
        p_hyper=args.p_hyper,
        hypercollisions_const=args.hypercollisions_const,
        hypercollisions_kz=args.hypercollisions_kz,
        d_hyper=args.d_hyper,
        damp_ends_amp=args.damp_ends_amp,
        damp_ends_widthfrac=args.damp_ends_widthfrac,
        hyperdiffusion=args.hyperdiffusion,
        normalization_contract=args.normalization_contract,
        diagnostic_norm=args.diagnostic_norm,
        rho_star_physical=args.rho_star_physical,
    )
    rc = bridge.cmd_prepare(prepare_args)
    if rc != 0:
        return rc

    manifest_path = output_dir / "manifest.json"
    run_args = argparse.Namespace(
        manifest=str(manifest_path),
        backend=args.backend,
        gpu_ids=args.gpu_ids,
        max_parallel=args.max_parallel,
        threads_per_run=args.threads_per_run,
        poll_interval=args.poll_interval,
        verbose_workers=args.verbose_workers,
    )
    rc = bridge.cmd_run(run_args)
    if rc != 0:
        return rc

    collect_args = argparse.Namespace(
        manifest=str(manifest_path),
        out=str(output_dir / "flux_summary.h5"),
        neopax_flux_out=str(output_dir / "neopax_fluxes.h5"),
        average_window=args.average_window,
        t_final=args.t_max,
        plot=args.plot,
        plot_run_heat_traces=args.plot_run_heat_traces,
    )
    return bridge.cmd_collect(collect_args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--neopax-config", required=True, help="NEOPAX transport TOML")
    p.add_argument("--neopax-result", default=None, help="Optional explicit path to transport_solution.h5")
    p.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "spectrax_flux_scan"),
        help="Directory for manifest, SPECTRAX outputs, and collected fluxes",
    )
    p.add_argument("--spectrax-root", default=str(bridge.DEFAULT_SPECTRAX_ROOT))
    p.add_argument("--spectrax-template", default=None, help="Base SPECTRAX runtime TOML used as the model template")
    p.add_argument("--profiles-source", choices=("transport_h5", "analytical"), default="transport_h5")
    p.add_argument("--time-index", type=int, default=-1)
    p.add_argument("--analytical-n-radii", type=int, default=51)
    p.add_argument("--electron-model", choices=("adiabatic", "kinetic"), default=None)
    p.add_argument("--reference-ion", default=None)
    p.add_argument("--rho-indices", default=None)
    p.add_argument("--rho-min", type=float, default=0.0)
    p.add_argument("--rho-max", type=float, default=1.0)
    p.add_argument("--num-radii", type=int, default=-1, help="Number of radii to sample; use <=0 for all available nonzero radii")
    p.add_argument("--vmec-file-override", default=None)
    p.add_argument("--boozer-file-override", default=None)
    p.add_argument("--density-floor", type=float, default=1.0e-8)
    p.add_argument("--temperature-floor", type=float, default=1.0e-8)
    p.add_argument("--gradient-coordinate", choices=("rho", "torflux", "rho_with_scale"), default="rho")
    p.add_argument("--gradient-scale", type=float, default=1.0)
    p.add_argument("--tprim-scale", type=float, default=1.0)
    p.add_argument("--fprim-scale", type=float, default=1.0)
    p.add_argument("--tau-e-override", type=float, default=None)
    p.add_argument("--nu-ion", type=float, default=0.01)
    p.add_argument("--nu-electron", type=float, default=0.0)
    p.add_argument("--nx", type=int, default=None, help="Nonlinear spectral resolution in kx / x")
    p.add_argument("--ny", type=int, default=None, help="Nonlinear spectral resolution in ky / y")
    p.add_argument("--nz", type=int, default=None, help="Parallel/grid resolution in z")
    p.add_argument("--lx", type=float, default=None)
    p.add_argument("--ly", type=float, default=None)
    p.add_argument("--boundary", default=None)
    p.add_argument("--y0", type=float, default=None)
    p.add_argument("--ntheta", type=int, default=None, help="Number of theta points for generated VMEC geometry")
    p.add_argument("--nperiod", type=int, default=None)
    p.add_argument("--t-max", type=float, default=None)
    p.add_argument("--t-final", dest="t_max", type=float, help="Alias for --t-max")
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--method", default=None)
    p.add_argument("--use-diffrax", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--fixed-dt", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--sample-stride", type=int, default=None)
    p.add_argument("--diagnostics-stride", type=int, default=None)
    p.add_argument("--cfl", type=float, default=None)
    p.add_argument("--state-sharding", default=None)
    p.add_argument("--ky", type=float, default=None, help="Default nonlinear reference ky retained unless overridden")
    p.add_argument("--nl", type=int, default=None)
    p.add_argument("--nm", type=int, default=None)
    p.add_argument("--init-field", default=None)
    p.add_argument("--init-amp", type=float, default=None)
    p.add_argument("--alpha", type=float, default=None, help="Field-line label alpha for the local geometry")
    p.add_argument("--npol", type=float, default=None)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--nu-hermite", type=float, default=None)
    p.add_argument("--nu-laguerre", type=float, default=None)
    p.add_argument("--nu-hyper", type=float, default=None)
    p.add_argument("--p-hyper", type=float, default=None)
    p.add_argument("--hypercollisions-const", type=float, default=None)
    p.add_argument("--hypercollisions-kz", type=float, default=None)
    p.add_argument("--d-hyper", type=float, default=None)
    p.add_argument("--damp-ends-amp", type=float, default=None)
    p.add_argument("--damp-ends-widthfrac", type=float, default=None)
    p.add_argument("--hyperdiffusion", type=float, default=None)
    p.add_argument("--normalization-contract", default=None)
    p.add_argument("--diagnostic-norm", default=None)
    p.add_argument("--rho-star-physical", type=float, default=None, help="Optional manual rho_star override; otherwise derive it per radius from VMEC geometry and the reference-ion profile")
    p.add_argument("--average-window", type=float, default=20.0, help="Average turbulent fluxes over the final time window")
    p.add_argument("--plot", action="store_true", help="Write PNG plots of Gamma and Q versus rho.")
    p.add_argument("--plot-run-heat-traces", action="store_true", help="Write per-run heat-flux time-trace PNGs from existing diagnostics CSV files.")
    p.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    p.add_argument("--gpu-ids", default="0")
    p.add_argument("--max-parallel", type=int, default=1)
    p.add_argument("--threads-per-run", type=int, default=1)
    p.add_argument("--poll-interval", type=float, default=2.0)
    p.add_argument("--verbose-workers", action="store_true", help="Show the stdout/stderr from each SPECTRAX worker run")
    p.set_defaults(func=cmd_all)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
