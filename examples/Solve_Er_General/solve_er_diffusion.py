"""
Solve and plot the electric field diffusion equation using NEOPAX, with the same TOML-driven setup as plot_er_ambipolar_roots.py, but actually integrating the Er diffusion equation.

This script:
- Reads the TOML config (same as run_from_toml.py)
- Sets up grid, field, species, state, and database
- Sets toggles so only Er evolves (density and temperature fixed)
- Solves the Er diffusion equation using NEOPAX's modular solver
- Plots the resulting Er profile
"""

from pathlib import Path
import sys
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import h5py

import NEOPAX
from run_from_toml import _load_config, _cfg_get, _resolve_path, _build_initial_species, _project_root_from_config

def solve_er_diffusion(config_path: str):
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

    # --- Set toggles so only Er evolves ---
    toggles = {
        "evolve_density": jnp.array([False] * n_species, dtype=bool),
        "evolve_temperature": jnp.array([False] * n_species, dtype=bool),
        "evolve_Er": True,
        "electron_index": 0,
    }

    # --- Build solver parameters (copy from run_from_toml.py, but force toggles) ---
    solver_cfg = cfg["solver"]
    profiles_cfg = cfg.get("profiles", {})
    eq_cfg = cfg.get("equations", {})
    t0 = float(solver_cfg["t0"])
    t_final = float(solver_cfg["t_final"])
    dt = float(solver_cfg["dt"])
    n_save = int(solver_cfg["n_save"])
    ts_list = jnp.linspace(t0, t_final, n_save)

    parameters = NEOPAX.Solver_Parameters(
        integrator=str(solver_cfg.get("integrator", "diffrax_kvaerno5")),
        transport_solver_family=str(solver_cfg.get("transport_solver_family", "auto")),
        transport_solver_backend=str(
            solver_cfg.get("transport_solver_backend", solver_cfg.get("integrator", "diffrax_kvaerno5"))
        ),
        nonlinear_solver_tol=float(solver_cfg.get("nonlinear_solver_tol", 1e-8)),
        nonlinear_solver_maxiter=int(solver_cfg.get("nonlinear_solver_maxiter", 50)),
        anderson_history=int(solver_cfg.get("anderson_history", 5)),
        theta_implicit=float(solver_cfg.get("theta_implicit", 1.0)),
        use_predictor_corrector=bool(solver_cfg.get("use_predictor_corrector", True)),
        n_corrector_steps=int(solver_cfg.get("n_corrector_steps", 1)),
        theta_ptc_enabled=bool(solver_cfg.get("theta_ptc_enabled", True)),
        theta_ptc_dt_min_factor=float(solver_cfg.get("theta_ptc_dt_min_factor", 1.0e-4)),
        theta_ptc_dt_max_factor=float(solver_cfg.get("theta_ptc_dt_max_factor", 1.0e3)),
        theta_ptc_growth=float(solver_cfg.get("theta_ptc_growth", 1.5)),
        theta_ptc_shrink=float(solver_cfg.get("theta_ptc_shrink", 0.5)),
        theta_line_search_enabled=bool(solver_cfg.get("theta_line_search_enabled", True)),
        theta_line_search_contraction=float(solver_cfg.get("theta_line_search_contraction", 0.5)),
        theta_line_search_min_alpha=float(solver_cfg.get("theta_line_search_min_alpha", 1.0e-4)),
        theta_line_search_c=float(solver_cfg.get("theta_line_search_c", 1.0e-4)),
        theta_max_step_retries=int(solver_cfg.get("theta_max_step_retries", 8)),
        theta_linear_solver=str(solver_cfg.get("theta_linear_solver", "direct")),
        theta_gmres_tol=float(solver_cfg.get("theta_gmres_tol", 1.0e-8)),
        theta_gmres_maxiter=int(solver_cfg.get("theta_gmres_maxiter", 200)),
        theta_trust_region_enabled=bool(solver_cfg.get("theta_trust_region_enabled", False)),
        theta_trust_radius=float(solver_cfg.get("theta_trust_radius", 1.0)),
        theta_homotopy_steps=int(solver_cfg.get("theta_homotopy_steps", 1)),
        theta_differentiable_mode=bool(solver_cfg.get("theta_differentiable_mode", False)),
        er_ambipolar_scan_min=float(profiles_cfg.get("er_ambipolar_scan_min", -20.0)),
        er_ambipolar_scan_max=float(profiles_cfg.get("er_ambipolar_scan_max", 20.0)),
        er_ambipolar_n_scan=int(profiles_cfg.get("er_ambipolar_n_scan", 96)),
        er_ambipolar_tol=float(profiles_cfg.get("er_ambipolar_tol", 1.0e-6)),
        er_ambipolar_maxiter=int(profiles_cfg.get("er_ambipolar_maxiter", 30)),
        er_ambipolar_n_coarse=int(profiles_cfg.get("er_ambipolar_n_coarse", 24)),
        er_ambipolar_n_fine=int(profiles_cfg.get("er_ambipolar_n_fine", 8)),
        er_ambipolar_method=str(profiles_cfg.get("er_ambipolar_method", "jit_multires")),
        t0=t0,
        t_final=t_final,
        dt=dt,
        ts_list=ts_list,
        rtol=float(solver_cfg["rtol"]),
        atol=float(solver_cfg["atol"]),
        DEr=float(eq_cfg.get("er_DEr", 2.0)),
        Er_relax=float(eq_cfg.get("er_relax", 0.1)),
        er_mode="diffusion",
        evolve_Er=True,
        evolve_density=jnp.array([False] * n_species, dtype=bool),
        evolve_temperature=jnp.array([False] * n_species, dtype=bool),
        chi_temperature=jnp.asarray(_cfg_get(cfg, "turbulence", "chi_temperature", default=[0.5] * n_species)),
        chi_density=jnp.asarray(_cfg_get(cfg, "turbulence", "chi_density", default=[0.0] * n_species)),
        on_OmegaC=0.0,
    )

    # --- No sources or turbulence for pure Er diffusion ---
    source_models = None
    turbulent = None

    # --- Initial state ---
    y0 = NEOPAX.TransportState(
        density=state.density,
        temperature=state.temperature,
        Er=state.Er,
    )

    # --- Build reference fluxes for BCs ---
    from NEOPAX._neoclassical import get_Neoclassical_Fluxes
    _, gamma_neo_0, q_neo_0, _ = get_Neoclassical_Fluxes(
        species, grid, field, database,
        y0.Er, y0.temperature, y0.density,
    )
    gamma_total_0 = gamma_neo_0
    q_total_0 = q_neo_0
    from run_from_toml import _build_boundary_conditions
    bc = _build_boundary_conditions(cfg, field, gamma_total_0, q_total_0, species, er_initial=y0.Er)

    args = (species, grid, field, database, turbulent, parameters)

    # --- Solve the Er diffusion equation ---
    sol = NEOPAX.solve_transport_equations(y0, args, source_models=source_models, bc=bc)
    er_final = sol.ys.Er[-1, :]
    rho = field.rho_grid

    # --- Plot the solved Er profile ---
    plt.figure(figsize=(8, 5))
    plt.plot(rho, er_final, label="Er (diffusion, solved)", color="tab:purple", linewidth=2)
    plt.xlabel(r"$\\rho$")
    plt.ylabel(r"$E_r$")
    plt.title("Solved $E_r$ profile (diffusion equation)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_png = output_dir / "Er_diffusion_solved.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close()
    print(f"Wrote: {out_png}")

    # Optionally, save Er profile to HDF5
    try:
        with h5py.File(output_dir / "Er_diffusion_solved.h5", "w") as f:
            f.create_dataset("rho", data=rho)
            f.create_dataset("Er", data=er_final)
        print(f"Wrote: {output_dir / 'Er_diffusion_solved.h5'}")
    except Exception as e:
        print(f"Could not save Er profile to HDF5: {e}")

if __name__ == "__main__":
    default_cfg = Path(__file__).with_name("solve_er_general_er_only.toml")
    cfg = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else default_cfg
    solve_er_diffusion(str(cfg))
