"""Compute and plot ambipolar Er roots for the Solve_Er_General setup.

This script uses the same TOML-style inputs as run_from_toml.py, but instead of
integrating transport equations it only evaluates ambipolarity roots at each
radius using NEOPAX._ambipolarity.find_ambipolar_Er_min_entropy_jit.
"""

from __future__ import annotations

from pathlib import Path
import sys
import jax
import numpy as np
import jax.numpy as jnp
import h5py as h5
import interpax 

import NEOPAX
from NEOPAX._neoclassical import get_Neoclassical_Fluxes
from NEOPAX._ambipolarity import find_all_ambipolar_Er_roots_profile_jit, pad_and_sort_roots_for_plotting

from run_from_toml import _load_config, _cfg_get, _resolve_path, _build_initial_species, _project_root_from_config
# --- Load Er profile from file (as in Fluxes_Calculation_comparison) ---


def compute_ambipolar_roots(config_path: str, blocksize=None):
    cfg_path = Path(config_path).resolve()
    cfg = _load_config(cfg_path)
    root = _project_root_from_config(cfg_path)

    input_dir = _resolve_path(root, _cfg_get(cfg, "paths", "input_dir", default="examples/inputs"))
    output_dir = _resolve_path(root, _cfg_get(cfg, "paths", "output_dir", default="examples/Solve_Er_General"))
    output_dir.mkdir(parents=True, exist_ok=True)

    vmec_file = _resolve_path(input_dir, _cfg_get(cfg, "geometry", "vmec_file", default=cfg["geometry"]["vmec_file"]))
    boozer_file = _resolve_path(input_dir, _cfg_get(cfg, "geometry", "boozer_file", default=cfg["geometry"]["boozer_file"]))
    neoclassical_file = _resolve_path(
        input_dir,
        _cfg_get(cfg, "neoclassical", "file", default="Dij_NEOPAX_FULL_S_NEW_Er_Opt.h5"),
    )

    n_species = int(_cfg_get(cfg, "species", "n_species", default=3))
    n_x = int(_cfg_get(cfg, "grid", "n_x", default=4))
    n_radial = int(_cfg_get(cfg, "grid", "n_radial", default=51))

    grid = NEOPAX.Grid.create_standard(n_radial, n_x, n_species)
    field = NEOPAX.Field.read_vmec_booz(n_radial, str(vmec_file), str(boozer_file))
    species, state = _build_initial_species(cfg, field, grid, n_species, n_radial)
    database = NEOPAX.Monoenergetic.read_monkes(field.a_b, str(neoclassical_file))

    profiles_cfg = cfg.get("profiles", {})
    root_cfg = cfg.get("root_finding", {})
    er_min = float(root_cfg.get("er_ambipolar_scan_min", -20.0))
    er_max = float(root_cfg.get("er_ambipolar_scan_max", 20.0))
    n_coarse = int(root_cfg.get("er_ambipolar_n_coarse", 24))
    n_refine = int(root_cfg.get("er_ambipolar_n_refine", 8))
    max_roots = int(root_cfg.get("er_ambipolar_max_roots", 3))
    tol = float(root_cfg.get("er_ambipolar_tol", 1.0e-6))
    x_tol = float(root_cfg.get("er_ambipolar_x_tol", 1.0e-6))

    # --- Method selection from TOML ---
    method = root_cfg.get("er_ambipolar_method", "two_stage")
    use_float32 = str(root_cfg.get("er_ambipolar_use_float32", "false")).lower() in ("true", "1", "yes")
    # Allow blocksize to be set from TOML if not passed as argument
    if blocksize is None:
        blocksize_val = root_cfg.get("er_ambipolar_blocksize", None)
        if blocksize_val is not None:
            try:
                blocksize_val = int(blocksize_val)
                blocksize = blocksize_val if blocksize_val > 0 else None
            except Exception:
                blocksize = None

    if method == "multistart_clustered":
        from NEOPAX._ambipolarity import find_ambipolar_Er_multistart_clustered
        n_starts = int(root_cfg.get("er_ambipolar_n_starts", 24))
        maxiter = int(root_cfg.get("er_ambipolar_maxiter", 20))
        cluster_tol = float(root_cfg.get("er_ambipolar_cluster_tol", 1e-2))
        max_clusters = int(root_cfg.get("er_ambipolar_max_clusters", 3))
        # Vectorized multi-start root-finder for all radii
        def root_finder_for_radius(i):
            def gamma(er):
                er_vec = state.Er.at[i].set(er)
                _, gamma_neo, _, _ = get_Neoclassical_Fluxes(
                    species, grid, field, database, er_vec, state.temperature, state.density
                )
                return jnp.sum(species.charge_qp * gamma_neo[:, i])
            def entropy(er):
                er_vec = state.Er.at[i].set(er)
                _, gamma_neo, _, _ = get_Neoclassical_Fluxes(
                    species, grid, field, database, er_vec, state.temperature, state.density
                )
                return jnp.sum(jnp.abs(gamma_neo[:, i]))
            roots = find_ambipolar_Er_multistart_clustered(
                gamma,
                entropy_func=entropy,
                Er_range=(er_min, er_max),
                n_starts=n_starts,
                tol=tol,
                maxiter=maxiter,
                cluster_tol=cluster_tol,
                return_entropies=True
            )
            # Pad roots and entropies to max_clusters for static shape
            roots_arr = jnp.full((max_clusters,), jnp.nan)
            entropies_arr = jnp.full((max_clusters,), jnp.nan)
            n_found = len(roots[0])
            roots_arr = roots_arr.at[:n_found].set(roots[0][:max_clusters])
            entropies_arr = entropies_arr.at[:n_found].set(roots[1][:max_clusters])
            # Best root: min-entropy
            entropy_masked = jnp.where(jnp.isfinite(entropies_arr), entropies_arr, jnp.inf)
            best_idx = jnp.argmin(entropy_masked)
            best_root = jnp.where(n_found > 0, roots_arr[best_idx], 0.0)
            return roots_arr, entropies_arr, best_root, n_found
        n_radial = state.Er.shape[0]
        roots_all, entropies_all, best_roots, n_roots_all = jax.vmap(root_finder_for_radius)(jnp.arange(n_radial))
    elif method == "adaptive":
        from NEOPAX._ambipolarity import find_ambipolar_Er_min_entropy_jit_adaptive
        n_init = int(root_cfg.get("er_ambipolar_adaptive_n_init", 16))
        n_subdiv = int(root_cfg.get("er_ambipolar_adaptive_n_subdiv", 2))
        n_rounds = int(root_cfg.get("er_ambipolar_adaptive_n_rounds", 2))
        max_brackets = int(root_cfg.get("er_ambipolar_adaptive_max_brackets", 24))
        # Vectorized adaptive root-finder for all radii
        def root_finder_for_radius(i):
            def gamma(er):
                er_vec = state.Er.at[i].set(er)
                _, gamma_neo, _, _ = get_Neoclassical_Fluxes(
                    species, grid, field, database, er_vec, state.temperature, state.density
                )
                return jnp.sum(species.charge_qp * gamma_neo[:, i])
            def entropy(er):
                er_vec = state.Er.at[i].set(er)
                _, gamma_neo, _, _ = get_Neoclassical_Fluxes(
                    species, grid, field, database, er_vec, state.temperature, state.density
                )
                return jnp.sum(jnp.abs(gamma_neo[:, i]))
            return find_ambipolar_Er_min_entropy_jit_adaptive(
                gamma,
                entropy,
                Er_range=(er_min, er_max),
                n_init=n_init,
                n_subdiv=n_subdiv,
                n_rounds=n_rounds,
                max_brackets=max_brackets,
                n_refine=n_refine,
                max_roots=max_roots,
                tol=tol,
                x_tol=x_tol,
                maxiter=12,
                use_float32=use_float32,
            )
        n_radial = state.Er.shape[0]
        roots_all, entropies_all, best_roots, n_roots_all = jax.vmap(root_finder_for_radius)(jnp.arange(n_radial))
    else:
        # --- Vectorized root-finding over all radii (two-stage adaptive) ---
        roots_all, entropies_all, best_roots, n_roots_all = find_all_ambipolar_Er_roots_profile_jit(
            get_Neoclassical_Fluxes,
            species,
            grid,
            field,
            database,
            state,
            er_min,
            er_max,
            n_coarse=n_coarse,
            n_refine=n_refine,
            max_roots=max_roots,
            tol=tol,
            x_tol=x_tol,
            maxiter=12,
            blocksize=blocksize
        )

    # Always transpose so that axis 0 is root index, axis 1 is radius
    rho = field.rho_grid
    print("rho shape:", rho.shape)
    print("roots_all shape:", roots_all.shape)
    roots_3, entropies_3, best_root = pad_and_sort_roots_for_plotting(
        roots_all, entropies_all, n_roots_all, best_roots, max_roots=max_roots
    )
    # Also return species, state, grid, field, database for plotting profiles and diagnostics
    return field.rho_grid, roots_3, entropies_3, best_root, output_dir, species, state, grid, field, database


def plot_roots(rho, roots_3, entropies_3, best_root, output_dir: Path) -> Path:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required to plot ambipolar roots.") from exc


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 8.0), sharex=True)

    labels = ["root 1", "root 2", "root 3"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    # --- top panel: Er roots ---
    for k in range(3):
        ax1.plot(rho, roots_3[k], color=colors[k], linewidth=1.8, label=labels[k])
    ax1.plot(rho, best_root, color="black", linewidth=2.2, linestyle="--", label="min-entropy root")

    # --- Add: Plot Er profile from NTSS_Initial_Er_Opt.h5 if available ---
    import os
    import h5py
    import interpax
    input_path = os.path.join(output_dir, '../inputs/')
    er_file = os.path.join(input_path, 'NTSS_Initial_Er_Opt.h5')
    try:
        if os.path.isfile(er_file):
            with h5py.File(er_file, 'r') as f:
                r_data = f['r'][()]
                er_data = f['Er'][()]
                # Try to get rho grid from file, else use current rho
                # Interpolate Er onto current rho grid if needed
                if len(er_data) != len(rho):
                    er_interp = interpax.interp1d(r_data, er_data, rho)
                else:
                    er_interp = er_data
                ax1.plot(rho, er_interp, color='tab:red', linewidth=2.0, linestyle=':', label='Er from NTSS_Initial_Er_Opt.h5')
    except Exception as e:
        print(f"Could not plot Er from NTSS_Initial_Er_Opt.h5: {e}")

    ax1.set_ylabel(r"$E_r$")
    ax1.set_title("Ambipolar $E_r$ roots (up to three per radius)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    # --- bottom panel: entropy at each root branch ---
    for k in range(3):
        ax2.plot(rho, entropies_3[k], color=colors[k], linewidth=1.8, label=labels[k])
    ax2.set_xlabel(r"$\rho$")
    ax2.set_ylabel(r"$\sum_s |\Gamma_s|$ (entropy proxy)")
    ax2.set_title("Entropy proxy per root branch")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    out_png = output_dir / "Er_ambipolar_roots.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    return out_png


# --- New: Plot density and temperature profiles for all species ---
def plot_density_temperature_profiles(rho, state, output_dir, species=None):
    import matplotlib.pyplot as plt
    density = state.density
    temperature = state.temperature
    n_species = density.shape[0]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for s in range(n_species):
        label = f"Species {s+1}" if species is None or not hasattr(species, "names") or not species.names else species.names[s]
        ax1.plot(rho, density[s], label=label)
        ax2.plot(rho, temperature[s], label=label)
    ax1.set_ylabel("Density [arb]")
    ax2.set_ylabel("Temperature [arb]")
    ax2.set_xlabel(r"$\rho$")
    ax1.set_title("Density profiles (all species)")
    ax2.set_title("Temperature profiles (all species)")
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)
    ax1.legend()
    ax2.legend()
    fig.tight_layout()
    out_path = output_dir / "species_density_temperature_profiles.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def run(config_path: str, blocksize=None):
    rho, roots_3, entropies_3, best_root, output_dir, species, state, grid, field, database = compute_ambipolar_roots(config_path, blocksize=blocksize)
    out_png = plot_roots(np.array(rho), roots_3, entropies_3, best_root, output_dir)
    print(f"Wrote: {out_png}")
    # Plot density and temperature profiles
    out_prof = plot_density_temperature_profiles(np.array(rho), state, output_dir, species)
    print(f"Wrote: {out_prof}")

    # --- Diagnostic: Plot Gamma(Er) for rho=0.05 only ---
    import matplotlib.pyplot as plt
    er_min, er_max = -20.0, 20.0
    n_diag = 200
    er_grid = np.linspace(er_min, er_max, n_diag)
    idx = int(np.argmin(np.abs(np.array(rho) - 0.05)))
    def gamma_func(er_val):
        er_vec = np.array(state.Er)
        er_vec[idx] = er_val
        _, gamma_neo, _, _ = get_Neoclassical_Fluxes(
            species, grid, field, database, er_vec, state.temperature, state.density
        )
        return np.sum(species.charge_qp * gamma_neo[:, idx])
    gamma_vals = np.array([gamma_func(er) for er in er_grid])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(er_grid, gamma_vals, label=f"rho={rho[idx]:.3f}", color="tab:blue")
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel(r"$E_r$")
    ax.set_ylabel(r"$\sum_s q_s \Gamma_s$")
    ax.set_title(r"Ambipolarity function $\Gamma(E_r)$ at $\rho=0.05$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    diag_path = output_dir / "Gamma_vs_Er_rho_0p05.png"
    fig.tight_layout()
    fig.savefig(diag_path, dpi=150)
    plt.close(fig)
    print(f"Wrote: {diag_path}")

    # Print roots object for inspection
    print("roots_3 (shape):", roots_3.shape)
    print(roots_3)
    print("entropies_3 (shape):", entropies_3.shape)
    print(entropies_3)
    print("best_root (shape):", best_root.shape)
    print(best_root)
    print("\nNOTE: If no physical roots are found at a radius, best_root will be set to 0.0 by the root-finder.\n"
                "This is why you may see a line for the min-entropy root even when no true ambipolar roots exist.")


if __name__ == "__main__":
    default_cfg = Path(__file__).with_name("solve_er_general_er_only.toml")
    cfg = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else default_cfg
    # Allow blocksize to be set as second command-line argument
    blocksize = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run(str(cfg), blocksize=blocksize)
