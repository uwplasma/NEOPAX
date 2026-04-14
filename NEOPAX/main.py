"""
NEOPAX main orchestrator (torax-style): TOML-driven workflow dispatch for ambipolarity and transport.
"""

import sys
import os
import dataclasses
from ._species import Species
from ._state import TransportState
from ._database import Monoenergetic
import jax
import jax.numpy as jnp

try:
    import tomli as toml
except ImportError:
    import toml

from ._ambipolarity import solve_ambipolarity_roots_from_config
from ._source_models import build_source_models_from_config
from ._transport_flux_models import build_transport_flux_model
from ._entropy_models import get_entropy_model
# Add other imports as needed (profiles, grid, field, database, etc.)


def load_config(path):
    with open(path, "rb") as f:
        return toml.load(f)

def main(config_path):
    config = load_config(config_path)
    # --- Support [general] section for mode and other global options ---
    general = config.get("general", {})
    mode = general.get("mode", config.get("mode", "transport")).lower()


    # --- Robust config section reading ---
    energy_grid_cfg = config.get("energy_grid", {})
    geom_cfg = config.get("geometry", {})
    species_cfg = config.get("species", {})
    neoclassical_cfg = config.get("neoclassical", {})
    profiles_cfg = config.get("profiles", {})



    # --- Build species ---

    # Ensure mass_mp and charge_qp are always jnp.arrays of float
    mass_mp = species_cfg.get("mass_mp", [0.000544617, 2.0, 3.0])
    charge_qp = species_cfg.get("charge_qp", [-1.0, 1.0, 1.0])
    n_species = int(species_cfg.get("n_species", 3))
    # Convert to jnp.array(float) if needed
    mass_mp = jnp.array(mass_mp, dtype=float)
    charge_qp = jnp.array(charge_qp, dtype=float)
    species_indices = jnp.arange(n_species)
    species = Species(
        number_species=n_species,
        species_indices=species_indices,
        mass_mp=mass_mp,
        charge_qp=charge_qp
    )


    # --- Build energy grid (modular) ---
    n_radial = int(geom_cfg.get("n_radial", 51))
    n_x = int(energy_grid_cfg.get("n_x", 4))
    from ._energy_grid_models import get_energy_grid_model
    energy_grid = get_energy_grid_model("standard_laguerre", n_x=n_x, n_order=3)

    # --- Build grid (for species/radial indices, if needed) ---
    # grid = None  # Legacy: NEOPAX.Grid.create_standard(n_radial, n_x, n_species)
    # Only instantiate if legacy code still needs it; otherwise, remove.

    # --- Build geometry (modular) ---
    from ._geometry_models import get_geometry_model
    vmec_file = geom_cfg.get("vmec_file")
    boozer_file = geom_cfg.get("boozer_file")
    if vmec_file is not None and boozer_file is not None:
        geometry = get_geometry_model("vmec_booz", n_r=n_radial, vmec=vmec_file, booz=boozer_file)
    else:
        geometry = None

    # --- Optionally: get flux/entropy model names from neoclassical section ---
    flux_model_name = neoclassical_cfg.get("flux_model", "monkes_database")
    entropy_model_name = neoclassical_cfg.get("entropy_model", "monkes_database")

    # --- Build database if neoclassical_file is given ---
    neoclassical_file = neoclassical_cfg.get("neoclassical_file")
    if neoclassical_file and geometry is not None:
        database = Monoenergetic.read_monkes(geometry.a_b, neoclassical_file)
    else:
        database = None


    # --- Build state from profiles using _profiles.py logic ---
    from ._profiles import build_profiles

    if geometry is not None:
        profile_set = build_profiles(profiles_cfg, geometry, n_species)
        state = TransportState(
            temperature=profile_set.temperature,
            density=profile_set.density,
            Er=profile_set.Er
        )
    else:
        state = None

    # Build params dictionary for all models
    params = {
        "energy_grid": energy_grid,
        "geometry": geometry,
        "database": database,
        "species": species,
    }

    # --- Build sources, fluxes, entropy models from config ---
    source_models = build_source_models_from_config(config)

    if mode == "ambipolarity":
        from pathlib import Path
        from ._ambipolarity import (
            solve_ambipolarity_roots_from_config,
            pad_and_sort_roots_for_plotting,
            plot_roots,
            write_ambipolarity_hdf5
        )
        # Solve ambipolarity roots (JIT/differentiable)
        result = solve_ambipolarity_roots_from_config(state, config, params)
        # New return signature: (roots_all, entropies_all, best_roots, n_roots_all, do_plot, do_hdf5, output_dir)
        if not (isinstance(result, tuple) and len(result) == 7):
            raise RuntimeError("Ambipolarity solver must return a 7-tuple: (roots_all, entropies_all, best_roots, n_roots_all, do_plot, do_hdf5, output_dir)")
        roots_all, entropies_all, best_roots, n_roots_all, do_plot, do_hdf5, output_dir = result
        # Use geometry.rho_grid for rho if needed
        rho = geometry.rho_grid if geometry is not None and hasattr(geometry, "rho_grid") else None

        # Debug print shapes and types before padding
        print("[DEBUG] roots_all shape:", getattr(roots_all, 'shape', None), type(roots_all))
        print("[DEBUG] entropies_all shape:", getattr(entropies_all, 'shape', None), type(entropies_all))
        print("[DEBUG] best_roots shape:", getattr(best_roots, 'shape', None), type(best_roots))
        print("[DEBUG] n_roots_all shape:", getattr(n_roots_all, 'shape', None), type(n_roots_all))
        print("[DEBUG] rho shape:", getattr(rho, 'shape', None), type(rho))
        # Prepare for plotting/writing (not JIT/differentiable, so outside main numerics)
        roots_3, entropies_3, best_root = pad_and_sort_roots_for_plotting(
            roots_all, entropies_all, n_roots_all, best_roots=best_roots, max_roots=3
        )

        print("Ambipolarity roots result:")
        print(f"rho shape: {jnp.shape(rho)}")
        print(f"roots_3 shape: {jnp.shape(roots_3)}")
        print(f"entropies_3 shape: {jnp.shape(entropies_3)}")
        print(f"best_root shape: {jnp.shape(best_root)}")

        # Only plotting and HDF5 writing are non-JIT, non-differentiable
        # output_dir may be None or a string; ensure Path
        if output_dir is None:
            output_dir = Path("outputs")
        elif not isinstance(output_dir, Path):
            output_dir = Path(str(output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)

        if do_plot:
            out_png = plot_roots(rho, roots_3, entropies_3, best_root, output_dir)
            print(f"Ambipolarity roots plot saved to: {out_png}")
        if do_hdf5:
            out_h5 = write_ambipolarity_hdf5(rho, roots_3, entropies_3, best_root, output_dir)
            print(f"Ambipolarity roots HDF5 saved to: {out_h5}")
        return {
            "rho": rho,
            "roots_3": roots_3,
            "entropies_3": entropies_3,
            "best_root": best_root,
            "output_dir": output_dir,
        }
    elif mode == "fluxes":
        from pathlib import Path
        # --- Calculate fluxes for the current state and config ---
        result = calculate_fluxes_from_config(state, config, params)
        if not (isinstance(result, tuple) and len(result) == 4):
            raise RuntimeError("Fluxes solver must return a 4-tuple: (fluxes, do_plot, do_hdf5, output_dir)")
        fluxes, do_plot, do_hdf5, output_dir = result
        rho = geometry.rho_grid if geometry is not None and hasattr(geometry, "rho_grid") else None
        if output_dir is None:
            output_dir = Path("outputs")
        elif not isinstance(output_dir, Path):
            output_dir = Path(str(output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        if do_plot:
            out_png = plot_fluxes(rho, fluxes, output_dir)
            print(f"Fluxes plot saved to: {out_png}")
        if do_hdf5:
            out_h5 = write_fluxes_hdf5(rho, fluxes, output_dir)
            print(f"Fluxes HDF5 saved to: {out_h5}")
        return {"rho": rho, "fluxes": fluxes, "output_dir": output_dir}
    elif mode == "transport":
        # --- Transport solver: time evolution of profiles ---
        from ._main_solver import solve_transport_equations
        from ._boundary_conditions import build_boundary_condition_model
        from ._transport_equations import get_equation
        # Build field (geometry) for equations
        field = geometry
        state0 = state
        # --- Equation selection from [equations] section ---
        equations_cfg = config.get("equations", {})
        eqn_flags = {
            "density": equations_cfg.get("toggle_density", [True]*n_species),
            "temperature": equations_cfg.get("toggle_temperature", [True]*n_species),
            "Er": equations_cfg.get("toggle_Er", True),
        }
        # Build list of equations to evolve
        equations_to_evolve = []
        if any(eqn_flags["density"]):
            equations_to_evolve.append(get_equation("density")())
        if any(eqn_flags["temperature"]):
            equations_to_evolve.append(get_equation("temperature")())
        if eqn_flags["Er"]:
            equations_to_evolve.append(get_equation("Er")())

        # --- Solver parameters from [transport_solver] section ---
        solver_cfg = config.get("transport_solver", {})
        # Fallback to [solver] or [transport] if not present
        if not solver_cfg:
            solver_cfg = config.get("solver", config.get("transport", {}))

        # --- Read [boundary] section and build BC models ---
        boundary_cfg = config.get("boundary", {})
        bc = {}
        dr = getattr(field, "dr", 1.0)
        for key in ("density", "temperature", "Er"):
            if key in boundary_cfg:
                bc[key] = build_boundary_condition_model(boundary_cfg[key], dr)

        # --- Build params tuple for solver (species, grid, field, database, turbulent, solver_parameters) ---
        args = (species, None, field, database, None, solver_cfg)

        # --- Run the time solver ---
        result = solve_transport_equations(
            state0,
            args,
            equations=equations_to_evolve,
            source_models=source_models,
            bc=bc
        )
        print("Transport solver completed.")
        return result
    else:
        raise ValueError(f"Unknown mode '{mode}'. Supported: 'ambipolarity', 'transport', 'fluxes'.")
    
    
def calculate_fluxes_from_config(state, config, params):
    """
    Config-driven entrypoint for direct flux calculation (no root-finding).
    Reads config, builds models, computes fluxes for the current state.
    Returns (fluxes, do_plot, do_hdf5, output_dir)
    """
    general = config.get("general", {})
    fluxes_cfg = config.get("fluxes", {})
    neoclassical_cfg = config.get("neoclassical", {})
    # Select flux model
    flux_model_name = neoclassical_cfg.get("flux_model", "monkes_database")
    flux_model = build_transport_flux_model(flux_model_name)
    # Compute fluxes
    fluxes = flux_model(state, params.get("geometry"), params)
    # Output options
    do_plot = fluxes_cfg.get("fluxes_plot", False)
    do_hdf5 = fluxes_cfg.get("fluxes_write_hdf5", False)
    output_dir = fluxes_cfg.get("fluxes_output_dir", None)
    return fluxes, do_plot, do_hdf5, output_dir

def plot_fluxes(rho, fluxes, output_dir):
    """Stub: Plot fluxes vs rho. Implement as needed."""
    import matplotlib.pyplot as plt
    out_png = output_dir / "fluxes.png"
    # Example: plot Gamma for each species if present
    Gamma = fluxes.get("Gamma", None)
    if Gamma is not None:
        if Gamma.ndim == 2:
            for i in range(Gamma.shape[0]):
                plt.plot(rho, Gamma[i], label=f"Gamma[{i}]")
        else:
            plt.plot(rho, Gamma, label="Gamma")
        plt.xlabel("rho")
        plt.ylabel("Gamma")
        plt.title("Particle Fluxes vs rho")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_png, dpi=170)
        plt.close()
    return out_png

def write_fluxes_hdf5(rho, fluxes, output_dir):
    """Stub: Write fluxes to HDF5. Implement as needed."""
    import h5py
    out_h5 = output_dir / "fluxes.h5"
    with h5py.File(out_h5, "w") as f:
        if rho is not None:
            f.create_dataset("rho", data=jnp.asarray(rho))
        for key, val in fluxes.items():
            f.create_dataset(key, data=jnp.asarray(val))
    return out_h5

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m NEOPAX.main <config.toml>")
        sys.exit(1)
    main(sys.argv[1])
