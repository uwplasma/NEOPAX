import os
from pathlib import Path
import sys

import h5py as h5
import jax
import jax.numpy as jnp
import dataclasses

import NEOPAX
from NEOPAX._ambipolarity import find_ambipolar_Er_min_entropy_jit
from NEOPAX._boundary_conditions import build_boundary_condition_model as _build_boundary_condition_model
from NEOPAX._constants import elementary_charge
from NEOPAX._fem import set_dirichlet_ghosts
from NEOPAX._neoclassical import get_Neoclassical_Fluxes
from NEOPAX._profiles import build_profiles
from NEOPAX._runtime_models import TurbulenceState
from NEOPAX._sources import build_source_models_from_config
from NEOPAX._species import Species
from NEOPAX._turbulence import get_Turbulent_Fluxes_Analytical

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


jax.config.update("jax_enable_x64", True)


def _resolve_path(base: Path, maybe_relative: str) -> Path:
    p = Path(maybe_relative)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _load_config(config_path: Path) -> dict:
    with config_path.open("rb") as f:
        return tomllib.load(f)


def _project_root_from_config(config_path: Path) -> Path:
    examples_dir = config_path.parent.parent
    if examples_dir.name == "examples":
        return examples_dir.parent
    return config_path.parent


def _cfg_get(cfg: dict, *path, default=None):
    cur = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _build_boundary_conditions(cfg: dict, field, gamma_total_0, q_total_0, initial_species, er_initial=None):
    eq_cfg = cfg.get("equations", {})

    # Preferred schema: flat keys in [equations], e.g.
    # density_left_type, density_right_value,
    # temperature_left_gradient, er_right_decay_length.
    def _build_flat_bc_root(eq_cfg_dict: dict) -> dict:
        bc_root_local = {}
        aliases = {
            "density": ["density"],
            "temperature": ["temperature"],
            "Er": ["er", "Er", "electric_field"],
        }
        for equation_name, prefixes in aliases.items():
            equation_cfg = {}
            for side in ("left", "right"):
                side_cfg = {}
                for field_name in ("type", "value", "gradient", "decay_length"):
                    field_value = None
                    for prefix in prefixes:
                        key = f"{prefix}_{side}_{field_name}"
                        if key in eq_cfg_dict:
                            field_value = eq_cfg_dict[key]
                            break
                    if field_value is not None:
                        side_cfg[field_name] = field_value
                if len(side_cfg) > 0:
                    equation_cfg[side] = side_cfg
            if len(equation_cfg) > 0:
                bc_root_local[equation_name] = equation_cfg
        return bc_root_local

    bc_root = _build_flat_bc_root(eq_cfg)

    if not isinstance(bc_root, dict) or len(bc_root) == 0:
        return None

    bc = {}
    if "density" in bc_root:
        bc["density"] = _build_boundary_condition_model(
            bc_root.get("density", {}),
            dr=field.dr,
            reference_profiles=gamma_total_0,
        )
    if "temperature" in bc_root:
        bc["temperature"] = _build_boundary_condition_model(
            bc_root.get("temperature", {}),
            dr=field.dr,
            reference_profiles=q_total_0,
        )
    if "Er" in bc_root or "electric_field" in bc_root:
        er_cfg = bc_root.get("Er", bc_root.get("electric_field", {}))
        bc["Er"] = _build_boundary_condition_model(
            er_cfg,
            dr=field.dr,
            reference_profile=er_initial,
            reference_profiles=gamma_total_0,
        )
    return bc if len(bc) > 0 else None


def _build_initial_species(cfg: dict, field, grid, n_species: int, n_radial: int):
    species_cfg = cfg.get("species", {})
    mass = jnp.asarray(species_cfg.get("mass_mp", [1.0 / 1836.15267343, 2.0, 3.0]))
    charge = jnp.asarray(species_cfg.get("charge_qp", [-1.0, 1.0, 1.0]))

    profile_cfg = dict(cfg["profiles"])
    profile_cfg["charge_qp"] = [float(v) for v in charge[:n_species]]
    profile_set = build_profiles(profile_cfg, field, n_species)

    species = Species(
        number_species=n_species,
        species_indices=grid.species_indeces,
        mass_mp=mass[:n_species],
        charge_qp=charge[:n_species],
    )
    state = NEOPAX.TransportState(
        density=profile_set.density,
        temperature=profile_set.temperature,
        Er=profile_set.Er,
    )
    return species, state


def _build_toggle_controls(cfg: dict, n_species: int):
    """Build generic equation toggles from TOML.
    
    Parses three simple toggle arrays from TOML:
    - toggle_density: [true, false, true]   for each species
    - toggle_temperature: [true, true, false] for each species
    - toggle_Er: true/false for electric field
    
    Example (3-species, freeze only deuterium temperature):
        [equations]
        toggle_density = [true, true, true]          # All densities evolve
        toggle_temperature = [true, false, true]     # Deuterium T frozen
        toggle_Er = true                             # Er evolves
    """
    eq_cfg = cfg.get("equations", {})
    
    def _parse_bool(val, default=True):
        """Convert value to boolean."""
        if isinstance(val, bool):
            return bool(val)
        if isinstance(val, (int, float)):
            return float(val) != 0.0
        return default
    
    # Parse toggle arrays, pad/truncate to n_species
    def _get_toggle_array(key: str, default: bool) -> jnp.ndarray:
        """Get toggle array from config, ensure length = n_species."""
        if key not in eq_cfg:
            # Not specified: default all to True or False
            return jnp.asarray([default] * n_species, dtype=bool)
        
        val = eq_cfg[key]
        if isinstance(val, (list, tuple)):
            toggles = [_parse_bool(v, default) for v in val]
        else:
            # Single boolean: apply to all species
            toggles = [_parse_bool(val, default)] * n_species
        
        # Truncate if too many, pad with default if too few
        if len(toggles) > n_species:
            toggles = toggles[:n_species]
        elif len(toggles) < n_species:
            toggles.extend([default] * (n_species - len(toggles)))
        
        return jnp.asarray(toggles, dtype=bool)
    
    # Parse the three toggles
    density_toggles_list = list(_get_toggle_array("toggle_density", default=True))
    temperature_toggles_list = list(_get_toggle_array("toggle_temperature", default=True))
    evolve_er = _parse_bool(eq_cfg.get("toggle_Er", True), True)

    # Detect electron species (most negative charge_qp) and force its density
    # toggle to False: electron density is a dependent variable derived from
    # quasi-neutrality  n_e = sum_ions Z_i * n_i, not independently evolved.
    electron_index = -1
    species_cfg = cfg.get("species", {})
    charge_qp_list = species_cfg.get("charge_qp", [])
    if len(charge_qp_list) >= n_species:
        charges = [float(c) for c in charge_qp_list[:n_species]]
        idx = charges.index(min(charges))
        if charges[idx] < 0.0:
            electron_index = idx
            if density_toggles_list[idx]:  # plain Python bool — safe inside or outside JIT
                import warnings
                warnings.warn(
                    f"toggle_density[{idx}] (electron) overridden to False — "
                    "electron density is derived from quasi-neutrality.",
                    stacklevel=2,
                )
            density_toggles_list[idx] = False

    return {
        "evolve_density": jnp.asarray(density_toggles_list, dtype=bool),
        "evolve_temperature": jnp.asarray(temperature_toggles_list, dtype=bool),
        "evolve_Er": bool(evolve_er),
        "electron_index": int(electron_index),
    }


def _build_source_models(cfg: dict) -> dict | None:
    """Build optional source model dictionary from TOML source registry names."""
    return build_source_models_from_config(cfg)


def _build_initial_er_profile(cfg: dict, species: Species, state: NEOPAX.TransportState,
                               grid, field, database) -> jnp.ndarray:
    profiles_cfg = cfg.get("profiles", {})
    mode = str(profiles_cfg.get("er_initialization_mode", "parabolic")).lower()

    if mode in ("parabolic", "profile", "default"):
        return state.Er

    if mode in ("ambipolar", "ambipolar_auto", "ambipolar_ion"):
        if mode == "ambipolar_ion":
            root_selection = "ion_root"
        else:
            root_selection = str(
                profiles_cfg.get("er_ambipolar_root_selection", "electron_or_ion")
            ).lower()

        er_min = float(profiles_cfg.get("er_ambipolar_scan_min", -20.0))
        er_max = float(profiles_cfg.get("er_ambipolar_scan_max", 20.0))
        n_scan = int(profiles_cfg.get("er_ambipolar_n_scan", 200))
        tol = float(profiles_cfg.get("er_ambipolar_tol", 1.0e-6))

        er_guess = state.Er
        er_values = []
        n_radial = int(field.r_grid.shape[0])

        for i in range(n_radial):
            def gamma_func(er_val, _i=i):
                er_vec = er_guess.at[_i].set(er_val)
                _, gamma_neo, _, _ = get_Neoclassical_Fluxes(
                    species, grid, field, database,
                    er_vec, state.temperature, state.density,
                )
                return jnp.sum(species.charge_qp * gamma_neo[:, _i])

            def entropy_func(er_val, _i=i):
                er_vec = er_guess.at[_i].set(er_val)
                _, gamma_neo, _, _ = get_Neoclassical_Fluxes(
                    species, grid, field, database,
                    er_vec, state.temperature, state.density,
                )
                return jnp.sum(jnp.abs(gamma_neo[:, _i]))

            best_root, roots_all, _, valid_mask = find_ambipolar_Er_min_entropy_jit(
                gamma_func, entropy_func,
                Er_range=(er_min, er_max), n_scan=n_scan, tol=tol,
            )
            roots = roots_all[valid_mask]

            if int(roots.shape[0]) == 0:
                chosen = best_root
            else:
                sorted_roots = jnp.sort(roots)
                ion_root = sorted_roots[0]
                electron_root = sorted_roots[-1]
                if root_selection == "ion_root":
                    chosen = ion_root
                else:
                    chosen = electron_root if int(sorted_roots.shape[0]) > 1 else ion_root

            er_values.append(chosen)

        return jnp.asarray(er_values)

    raise ValueError(
        "Unsupported er_initialization_mode. Use one of: "
        "parabolic, ambipolar_auto, ambipolar_ion"
    )


def run(config_path: str):
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
        _cfg_get(cfg, "neoclassical", "file", default=_cfg_get(cfg, "neoclassiclal", "neoclassical_file", default="Dij_NEOPAX_FULL_S_NEW_Er_Opt.h5")),
    )

    n_species = int(_cfg_get(cfg, "species", "n_species", default=3))
    n_x = int(_cfg_get(cfg, "grid", "n_x", default=_cfg_get(cfg, "neoclassical", "n_x", default=4)))
    n_radial = int(_cfg_get(cfg, "grid", "n_radial", default=_cfg_get(cfg, "r_grid", "n_radial", default=51)))

    grid = NEOPAX.Grid.create_standard(n_radial, n_x, n_species)
    field = NEOPAX.Field.read_vmec_booz(n_radial, str(vmec_file), str(boozer_file))

    solver_cfg = cfg["solver"]
    eq_cfg = cfg.get("equations", {})
    profiles_cfg = cfg.get("profiles", {})

    t0 = float(solver_cfg["t0"])
    t_final = float(solver_cfg["t_final"])
    dt = float(solver_cfg["dt"])
    n_save = int(solver_cfg["n_save"])
    ts_list = jnp.linspace(t0, t_final, n_save)

    momentum_correction_flag = bool(
        _cfg_get(cfg, "neoclassical", "momentum_correction_flag", default=False)
    )
    neoclassical_transport_model = str(
        _cfg_get(cfg, "neoclassical", "transport_model", default="neoclassical")
    )
    turbulent_transport_model = str(
        _cfg_get(cfg, "turbulence", "transport_model", default="analytical")
    )

    # Build generic equation toggles from TOML
    toggles = _build_toggle_controls(cfg, n_species)

    # Extract electric field parameters from equations section
    er_mode = eq_cfg.get("er_mode", "diffusion")
    er_DEr = float(eq_cfg.get("er_DEr", 2.0))
    er_relax = float(eq_cfg.get("er_relax", 0.1))

    parameters = NEOPAX.Solver_Parameters(
        integrator=str(solver_cfg.get("integrator", "diffrax_kvaerno5")),
        er_ambipolar_scan_min=float(profiles_cfg.get("er_ambipolar_scan_min", -20.0)),
        er_ambipolar_scan_max=float(profiles_cfg.get("er_ambipolar_scan_max", 20.0)),
        er_ambipolar_n_scan=int(profiles_cfg.get("er_ambipolar_n_scan", 96)),
        er_ambipolar_tol=float(profiles_cfg.get("er_ambipolar_tol", 1.0e-6)),
        er_ambipolar_maxiter=int(profiles_cfg.get("er_ambipolar_maxiter", 30)),
        er_ambipolar_n_coarse=int(profiles_cfg.get("er_ambipolar_n_coarse", 24)),
        er_ambipolar_n_fine=int(profiles_cfg.get("er_ambipolar_n_fine", 8)),
        er_ambipolar_method=str(profiles_cfg.get("er_ambipolar_method", "jit_multires")),
        neoclassical_transport_model=neoclassical_transport_model,
        turbulent_transport_model=turbulent_transport_model,
        t0=t0,
        t_final=t_final,
        dt=dt,
        ts_list=ts_list,
        rtol=float(solver_cfg["rtol"]),
        atol=float(solver_cfg["atol"]),
        momentum_correction_flag=momentum_correction_flag,
        DEr=er_DEr,
        Er_relax=er_relax,
        er_mode=er_mode,
        evolve_Er=toggles["evolve_Er"],
        evolve_density=toggles["evolve_density"],
        evolve_temperature=toggles["evolve_temperature"],
        chi_temperature=jnp.asarray(
            _cfg_get(cfg, "turbulence", "chi_temperature", default=[0.5] * n_species)
        ),
        chi_density=jnp.asarray(
            _cfg_get(cfg, "turbulence", "chi_density", default=[0.0] * n_species)
        ),
        on_OmegaC=0.0,
    )
    source_models = _build_source_models(cfg)

    global_species, y0_profiles = _build_initial_species(cfg, field, grid, n_species, n_radial)
    database = NEOPAX.Monoenergetic.read_monkes(field.a_b, str(neoclassical_file))

    er_initial = _build_initial_er_profile(cfg, global_species, y0_profiles, grid, field, database)
    y0 = NEOPAX.TransportState(
        density=y0_profiles.density,
        temperature=y0_profiles.temperature,
        Er=er_initial,
    )

    turbulence_model_name = str(_cfg_get(cfg, "turbulence", "model", default="analytical")).lower()
    if turbulence_model_name == "analytical":
        turbulent = None
    else:
        turbulent = TurbulenceState(
            Gamma_turb=jnp.zeros_like(y0.density),
            Q_turb=jnp.zeros_like(y0.density),
        )

    # Build reference fluxes at t0 for BC inference when values are omitted.
    _, gamma_neo_0, q_neo_0, _ = get_Neoclassical_Fluxes(
        global_species, grid, field, database,
        y0.Er, y0.temperature, y0.density,
    )
    if turbulence_model_name == "analytical":
        gamma_turb_0, q_turb_0 = get_Turbulent_Fluxes_Analytical(
            global_species, grid,
            parameters.chi_temperature, parameters.chi_density,
            y0.temperature, y0.density,
            field=field,
        )
    else:
        gamma_turb_0 = turbulent.Gamma_turb
        q_turb_0 = turbulent.Q_turb

    gamma_total_0 = gamma_neo_0 + gamma_turb_0
    q_total_0 = q_neo_0 + q_turb_0
    bc = _build_boundary_conditions(cfg, field, gamma_total_0, q_total_0, global_species, er_initial=y0.Er)

    args = (global_species, grid, field, database, turbulent, parameters)

    sol = NEOPAX.solve_transport_equations(y0, args, source_models=source_models, bc=bc)
    er_final = sol.ys.Er[-1, :]

    T_final = sol.ys.temperature[-1]
    n_final = sol.ys.density[-1]

    # Bootstrap diagnostic from the base neoclassical model.
    _, _, _, upar_mom = get_Neoclassical_Fluxes(
        global_species, grid, field, database,
        er_final, T_final, n_final,
    )
    # Generic bootstrap: j = sum_i Z_i * upar_i * e  (charge_qp are Z numbers)
    j_boots = jnp.sum(global_species.charge_qp[None, :] * upar_mom, axis=1) * elementary_charge

    out_file = output_dir / "Er_Test.h5"
    with h5.File(out_file, "w") as f:
        f["rho"] = field.rho_grid
        f["Er"] = er_final
        f["Jboots"] = j_boots

    print(f"Wrote: {out_file}")
    print(f"max Er: {jnp.max(er_final)}")


if __name__ == "__main__":
    default_cfg = Path(__file__).with_name("solve_er_general.toml")
    cfg = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else default_cfg
    run(str(cfg))
