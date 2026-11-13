"""Regenerate state/source comparison plots from HDF5 without running transport.

No runtime geometry, monoenergetic database or flux model is constructed.
Only the configured algebraic source models and saved profile arrays are used.
"""

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args(argv)

    import h5py
    import jax.numpy as jnp
    import numpy as np

    from NEOPAX._boundary_conditions import build_boundary_condition_model
    from NEOPAX._orchestrator import (
        _build_species,
        _normalize_solver_config,
        _normalized_boundary_cfg_for_transport,
        load_config,
        plot_transport_solution,
    )
    from NEOPAX._source_models import build_source_models_from_config
    from NEOPAX._state import TransportState
    from NEOPAX._temperature_constraints import (
        helium_temperature_match_indices,
        match_helium_temperature_boundary,
    )

    cfg = load_config(args.config)
    species = _build_species(cfg)
    solver_cfg = _normalize_solver_config(cfg)
    with h5py.File(args.solution, "r") as stream:
        saved = {
            key: jnp.asarray(stream[key][()])
            for key in ("density", "pressure", "Er", "ts", "rho", "r_grid", "r_grid_half")
        }
        geometry_values = {
            key: jnp.asarray(stream[key][()])
            for key in ("Vprime", "Vprime_half") if key in stream
        }
        geometry_values["rho_grid_half"] = (
            jnp.asarray(stream["rho_face"][()])
            if "rho_face" in stream else saved["r_grid_half"] / saved["r_grid_half"][-1]
        )
    geometry = SimpleNamespace(
        r_grid=saved["r_grid"], r_grid_half=saved["r_grid_half"],
        rho_grid=saved["rho"], a_b=float(saved["r_grid_half"][-1]),
        dr=float(np.mean(np.diff(saved["r_grid_half"]))), **geometry_values,
    )
    boundary_cfg = _normalized_boundary_cfg_for_transport(cfg.get("boundary", {}))
    boundaries = {
        key: build_boundary_condition_model(
            value, geometry.dr,
            species_names=species.names if key in {"density", "temperature", "gamma"} else None,
        )
        for key, value in boundary_cfg.items()
        if key in {"density", "temperature", "Er", "gamma"}
    }
    if "temperature" in boundaries:
        boundaries["temperature"] = match_helium_temperature_boundary(
            boundaries["temperature"], helium_temperature_match_indices(cfg, species.names)
        )
    output = args.out_dir or args.solution.parent / "replotted"
    output.mkdir(parents=True, exist_ok=True)
    plot_cfg = cfg.get("transport_output", {})
    print(f"[replot] saved times: {np.asarray(saved['ts'])}; no transport or flux solve", flush=True)
    plot_transport_solution(
        saved["rho"],
        {"ts": saved["ts"], "ys": TransportState(
            density=saved["density"], pressure=saved["pressure"], Er=saved["Er"],
        )},
        output,
        n_times=plot_cfg.get("transport_plot_n_times", -1),
        reference_er_file=plot_cfg.get("transport_reference_er_file"),
        overlay_reference_er=plot_cfg.get("transport_overlay_reference_er", False),
        reference_profile_file=plot_cfg.get("transport_reference_profile_file"),
        initial_reference_file=plot_cfg.get("transport_initial_reference_file"),
        final_reference_file=plot_cfg.get("transport_final_reference_file"),
        initial_reference_label=plot_cfg.get("transport_initial_reference_label", "Initial ref"),
        final_reference_label=plot_cfg.get("transport_final_reference_label", "Final ref"),
        source_models=build_source_models_from_config(cfg, species),
        species=species,
        flux_model=None,
        geometry=geometry,
        boundary_models=boundaries,
        density_floor=solver_cfg.get("density_floor", 1e-6),
        temperature_floor=solver_cfg.get("temperature_floor"),
    )
    print(f"[replot] wrote state/source plots to {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
