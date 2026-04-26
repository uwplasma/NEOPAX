"""
NEOPAX main orchestrator: TOML-driven workflow dispatch for ambipolarity,
transport, and direct flux evaluation.
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from ._ambipolarity import (
    pad_and_sort_roots_for_plotting,
    plot_roots,
    solve_ambipolarity_roots_from_config,
    solve_ambipolarity_roots_radial,
    write_ambipolarity_hdf5,
)
from ._database import Monoenergetic
from ._database_ntss_preprocessed import (
    NTSSPreprocessedMonoenergetic,
)
from ._database_preprocessed import (
    PreprocessedMonoenergetic3D,
    PreprocessedMonoenergetic3DNTSSRadius,
    PreprocessedMonoenergetic3DNTSSRadiusNTSS1D,
    PreprocessedMonoenergetic3DNTSSRadiusNTSS1DFixedNU,
)
from ._entropy_models import get_entropy_model
from ._profiles import build_profiles
from ._source_models import (
    assemble_density_source_components,
    assemble_pressure_source_components,
    build_source_models_from_config,
    sum_source_components,
)
from ._species import Species
from ._state import TransportState
from ._transport_flux_models import (
    FluxesRFileTransportModel,
    ZeroTransportModel,
    build_transport_flux_model,
    compute_total_power_breakdown_mw,
    compute_total_power_mw,
    get_transport_flux_model,
    read_flux_profile_file,
)

try:
    import tomli as toml
except ImportError:
    import toml


@dataclasses.dataclass(frozen=True)
class Models:
    flux: object = None
    source: object = None


@dataclasses.dataclass(frozen=True)
class RuntimeContext:
    species: Species
    energy_grid: object
    geometry: object | None
    database: object | None
    solver_parameters: dict
    models: Models


def load_config(path):
    text = Path(path).read_text(encoding="utf-8")
    return toml.loads(text)


def _normalize_solver_config(config: dict) -> dict:
    solver_cfg = config.get("transport_solver", {})
    if not solver_cfg:
        solver_cfg = config.get("solver", config.get("transport", {}))
    solver_cfg = dict(solver_cfg)
    solver_cfg["transport_solver_backend"] = str(
        solver_cfg.get("transport_solver_backend", solver_cfg.get("integrator", "diffrax_kvaerno5"))
    )
    solver_cfg["integrator"] = solver_cfg["transport_solver_backend"]
    solver_cfg["neoclassical_flux_model"] = config.get("neoclassical", {}).get("flux_model", "none")
    solver_cfg["turbulence_flux_model"] = config.get("turbulence", {}).get("flux_model", "none")
    solver_cfg.setdefault("Er_relax", 1.0)
    solver_cfg.setdefault("DEr", 1.0)
    solver_cfg.setdefault("density_floor", 1.0e-6)
    return solver_cfg


def _state_num_elements(state: TransportState | None) -> int:
    if state is None:
        return 0
    total = 0
    for arr in (state.density, state.pressure, state.Er):
        if hasattr(arr, "size"):
            total += int(arr.size)
    return total


def _apply_transport_solver_memory_heuristics(solver_cfg: dict, state: TransportState, n_equations: int) -> dict:
    """Keep solver config on the supported Diffrax-only path."""
    tuned = dict(solver_cfg)
    backend = str(tuned.get("transport_solver_backend", tuned.get("integrator", "diffrax_kvaerno5"))).strip().lower()
    tuned["transport_solver_backend"] = backend
    tuned["integrator"] = backend
    tuned["transport_solver_family"] = "ode"
    return tuned


def _build_species(config: dict) -> Species:
    species_cfg = config.get("species", {})
    n_species = int(species_cfg.get("n_species", 3))
    mass_mp = jnp.asarray(species_cfg.get("mass_mp", [0.000544617, 2.0, 3.0]), dtype=float)
    charge_qp = jnp.asarray(species_cfg.get("charge_qp", [-1.0, 1.0, 1.0]), dtype=float)
    names = tuple(species_cfg.get("names", ["e", "D", "T"]))
    return Species(
        number_species=n_species,
        species_indices=jnp.arange(n_species),
        mass_mp=mass_mp,
        charge_qp=charge_qp,
        names=names,
    )


def _build_energy_grid(config: dict):
    from ._energy_grid_models import get_energy_grid_model

    energy_grid_cfg = config.get("energy_grid", {})
    n_x = int(energy_grid_cfg.get("n_x", 4))
    return get_energy_grid_model("standard_laguerre", n_x=n_x, n_order=3)


def _build_geometry(config: dict):
    from ._geometry_models import get_geometry_model

    geom_cfg = config.get("geometry", {})
    n_radial = int(geom_cfg.get("n_radial", 51))
    vmec_file = geom_cfg.get("vmec_file")
    boozer_file = geom_cfg.get("boozer_file")
    if vmec_file is None or boozer_file is None:
        return None
    return get_geometry_model("vmec_booz", n_r=n_radial, vmec=vmec_file, booz=boozer_file)


def _build_database(config: dict, geometry):
    neoclassical_file = config.get("neoclassical", {}).get("neoclassical_file")
    if neoclassical_file and geometry is not None:
        interp_mode = str(
            config.get("neoclassical", {}).get("interpolation_mode", "generic")
        ).strip().lower()
        if interp_mode == "preprocessed_ntss":
            return NTSSPreprocessedMonoenergetic.read_monkes(geometry, neoclassical_file)
        if interp_mode == "preprocessed_3d":
            return PreprocessedMonoenergetic3D.read_monkes(geometry.a_b, neoclassical_file)
        if interp_mode == "preprocessed_3d_radial":
            return PreprocessedMonoenergetic3DNTSSRadius.read_monkes(geometry.a_b, neoclassical_file)
        if interp_mode == "preprocessed_3d_radial_ntss1d":
            return PreprocessedMonoenergetic3DNTSSRadiusNTSS1D.read_monkes(geometry.a_b, neoclassical_file)
        if interp_mode == "preprocessed_3d_ntss1d_fixed":
            return PreprocessedMonoenergetic3DNTSSRadiusNTSS1DFixedNU.read_monkes(geometry.a_b, neoclassical_file)
        return Monoenergetic.read_monkes(geometry.a_b, neoclassical_file)
    return None


def _build_state(config: dict, geometry, n_species: int):
    if geometry is None:
        return None
    profile_set = build_profiles(config.get("profiles", {}), geometry, n_species)
    return TransportState(
        density=profile_set.density / 1.0e20,
        pressure=(profile_set.temperature / 1.0e3) * (profile_set.density / 1.0e20),
        Er=profile_set.Er,
    )


def _maybe_initialize_er_from_ambipolarity(config: dict, runtime: RuntimeContext, state: TransportState | None):
    if state is None:
        return state

    profiles_cfg = config.get("profiles", {})
    init_mode = str(profiles_cfg.get("er_initialization_mode", "analytical")).strip().lower()
    removed_modes = {
        "ambipolar_min_entropy_fast",
        "ambipolar_min_entropy_tracked",
        "ambipolar_min_entropy_hybrid",
        "ambipolar_min_entropy_multibranch",
    }
    if init_mode in removed_modes:
        raise ValueError(
            f"Unsupported er_initialization_mode '{init_mode}'. "
            "Alternative ambipolar initializers were removed; use 'ambipolar_min_entropy'."
        )
    if init_mode not in {
        "ambipolar_min_entropy",
        "ambipolar_best_root",
        "ambipolarity_best_root",
    }:
        return state

    debug_stage_markers = bool(runtime.solver_parameters.get("debug_stage_markers", False))
    if debug_stage_markers:
        print(f"[NEOPAX] starting Er initialization: mode={init_mode}")
    t_start = time.perf_counter()

    amb_cfg = dict(config.get("ambipolarity", {}))
    model_name = str(amb_cfg.get("er_ambipolar_method", "two_stage")).lower()
    entropy_model_name = config.get("neoclassical", {}).get(
        "entropy_model",
        runtime.solver_parameters.get("neoclassical_flux_model", "monkes_database"),
    )
    entropy_model = get_entropy_model(entropy_model_name)
    params = {
        "species": runtime.species,
        "energy_grid": runtime.energy_grid,
        "geometry": runtime.geometry,
        "database": runtime.database,
        "solver_parameters": runtime.solver_parameters,
    }
    _, _, best_roots, _ = solve_ambipolarity_roots_radial(
        state=state,
        config=config,
        params=params,
        model_name=model_name,
        flux_model=runtime.models.flux,
        entropy_model=entropy_model,
        amb_cfg=amb_cfg,
    )
    best_roots = jnp.asarray(best_roots, dtype=state.Er.dtype)
    er_init = jnp.where(jnp.isfinite(best_roots), best_roots, state.Er)
    if debug_stage_markers:
        dt = time.perf_counter() - t_start
        n_finite = int(jnp.sum(jnp.isfinite(best_roots)))
        print(
            f"[NEOPAX] finished Er initialization: mode={init_mode} "
            f"elapsed_s={dt:.3f} finite_roots={n_finite}/{best_roots.shape[0]}"
        )
    return dataclasses.replace(state, Er=er_init)


def _build_flux_model(config: dict, species, energy_grid, geometry, database, source_models=None):
    def _model_name(section_cfg, default):
        return str(section_cfg.get("flux_model", section_cfg.get("model", default))).strip().lower()

    def _file_path(section_name, section_cfg):
        candidates = [
            section_cfg.get("fluxes_file"),
            section_cfg.get("file"),
            section_cfg.get(f"{section_name}_file"),
        ]
        if section_name == "neoclassical":
            candidates.insert(0, section_cfg.get("neoclassical_file"))
        for candidate in candidates:
            if candidate:
                return candidate
        return None

    def _build_file_model(section_name, section_cfg):
        path = _file_path(section_name, section_cfg)
        if path is None:
            raise ValueError(
                f"[{section_name}] fluxes_r_file requires a flux file. "
                f"Provide one of: fluxes_file, file, or {section_name}_file."
            )
        r_data, gamma_data, q_data, upar_data = read_flux_profile_file(path, species.number_species)
        return FluxesRFileTransportModel(
            species=species,
            geometry=geometry,
            r_data=r_data,
            gamma_data=gamma_data,
            q_data=q_data,
            upar_data=upar_data,
        )

    neoclassical_cfg = config.get("neoclassical", {})
    turbulence_cfg = config.get("turbulence", {})
    classical_cfg = config.get("classical", {})

    neoclassical_name = _model_name(neoclassical_cfg, "monkes_database")
    neoclassical_factory = get_transport_flux_model(neoclassical_name) if neoclassical_name != "fluxes_r_file" else None
    turbulence_cfg = config.get("turbulence", {})
    turbulence_name = _model_name(turbulence_cfg, "none")
    turbulence_factory = get_transport_flux_model(turbulence_name) if turbulence_name != "fluxes_r_file" else None
    classical_factory = (
        get_transport_flux_model(_model_name(classical_cfg, "none"))
        if "classical" in config
        and _model_name(classical_cfg, "none") != "fluxes_r_file"
        else None
    )

    neoclassical_model = (
        _build_file_model("neoclassical", neoclassical_cfg)
        if neoclassical_name == "fluxes_r_file"
        else neoclassical_factory(
            species,
            energy_grid,
            geometry,
            database,
            collisionality_model=neoclassical_cfg.get("collisionality_model", "default"),
        )
    )
    if turbulence_factory is None:
        turbulence_model = (
            _build_file_model("turbulence", turbulence_cfg)
            if turbulence_name == "fluxes_r_file"
            else ZeroTransportModel()
        )
    elif turbulence_name == "turbulent_analytical":
        chi_t = jnp.asarray(
            turbulence_cfg.get(
                "chi_temperature",
                turbulence_cfg.get("chi_t", [0.0] * species.number_species),
            ),
            dtype=float,
        )
        chi_n = jnp.asarray(
            turbulence_cfg.get(
                "chi_density",
                turbulence_cfg.get("chi_n", [0.0] * species.number_species),
            ),
            dtype=float,
        )
        turbulence_model = turbulence_factory(species, energy_grid, chi_t, chi_n, geometry)
    elif turbulence_name in {"turbulent_power_analytical", "ntss_power_over_n"}:
        chi_t = jnp.asarray(
            turbulence_cfg.get(
                "chi_temperature",
                turbulence_cfg.get("chi_t", [0.0] * species.number_species),
            ),
            dtype=float,
        )
        chi_n = jnp.asarray(
            turbulence_cfg.get(
                "chi_density",
                turbulence_cfg.get("chi_n", [0.0] * species.number_species),
            ),
            dtype=float,
        )
        if "coef_dano" in turbulence_cfg or "coef_d" in turbulence_cfg:
            chi_n = jnp.full((species.number_species,), float(turbulence_cfg.get("coef_dano", turbulence_cfg.get("coef_d", 0.0))), dtype=float)
        if any(key in turbulence_cfg for key in ("coef_xe", "coef_e", "coef_xi", "coef_i")):
            electron_idx = int(species.species_idx.get("e", 0))
            ion_value = float(turbulence_cfg.get("coef_xi", turbulence_cfg.get("coef_i", 0.0)))
            electron_value = float(turbulence_cfg.get("coef_xe", turbulence_cfg.get("coef_e", ion_value)))
            chi_t = jnp.full((species.number_species,), ion_value, dtype=float).at[electron_idx].set(electron_value)
        total_power_mw = turbulence_cfg.get("total_power_mw", turbulence_cfg.get("power_mw"))
        pressure_source_model = None if source_models is None else source_models.get("temperature")
        if total_power_mw is None and pressure_source_model is None:
            raise ValueError(
                f"[turbulence] flux_model='{turbulence_name}' requires a power source. "
                "Provide 'total_power_mw' (or 'power_mw') in [turbulence], or configure "
                "temperature sources so NEOPAX can build the scalar power automatically."
            )
        turbulence_model = turbulence_factory(
            species,
            energy_grid,
            geometry,
            chi_t,
            chi_n,
            pressure_source_model,
            total_power_mw,
        )
    else:
        turbulence_model = turbulence_factory(species, energy_grid, geometry, database)
    classical_model = (
        _build_file_model("classical", classical_cfg)
        if _model_name(classical_cfg, "none") == "fluxes_r_file"
        else (
            classical_factory(species, energy_grid, geometry, database)
            if classical_factory is not None
            else ZeroTransportModel()
        )
    )
    return build_transport_flux_model(neoclassical_model, turbulence_model, classical_model)


def build_runtime_context(config: dict) -> tuple[RuntimeContext, TransportState | None]:
    species = _build_species(config)
    energy_grid = _build_energy_grid(config)
    geometry = _build_geometry(config)
    database = _build_database(config, geometry)
    state = _build_state(config, geometry, species.number_species)
    solver_cfg = _normalize_solver_config(config)
    source_models = build_source_models_from_config(config, species)
    models = Models(
        flux=_build_flux_model(config, species, energy_grid, geometry, database, source_models=source_models),
        source=source_models,
    )
    runtime = RuntimeContext(
        species=species,
        energy_grid=energy_grid,
        geometry=geometry,
        database=database,
        solver_parameters=solver_cfg,
        models=models,
    )
    mode = str(config.get("general", {}).get("mode", config.get("mode", "transport"))).strip().lower()
    # Ambipolarity-only runs solve the root problem explicitly later, so avoid
    # paying for the same radial solve here during state initialization.
    if mode != "ambipolarity":
        state = _maybe_initialize_er_from_ambipolarity(config, runtime, state)
    return runtime, state


def run_transport(config: dict, runtime: RuntimeContext, state: TransportState):
    from ._boundary_conditions import build_boundary_condition_model
    from ._transport_equations import ComposedEquationSystem, build_equation_system
    from ._transport_solvers import build_time_solver

    field = runtime.geometry
    boundary_cfg = config.get("boundary", {})
    bc = {}
    dr = getattr(field, "dr", 1.0)
    for key in ("density", "temperature", "Er", "gamma"):
        if key in boundary_cfg:
            bc[key] = build_boundary_condition_model(
                boundary_cfg[key],
                dr,
                species_names=runtime.species.names if key in {"density", "temperature", "gamma"} else None,
            )

    er_bc_mode = str(runtime.solver_parameters.get("Er_right_boundary_mode", "config")).strip().lower()
    if er_bc_mode == "ambipolar_edge_root":
        amb_cfg = dict(config.get("ambipolarity", {}))
        model_name = str(amb_cfg.get("er_ambipolar_method", "two_stage")).lower()
        entropy_model_name = config.get("neoclassical", {}).get(
            "entropy_model",
            runtime.solver_parameters.get("neoclassical_flux_model", "monkes_database"),
        )
        entropy_model = get_entropy_model(entropy_model_name)
        _, _, best_roots, _, = solve_ambipolarity_roots_radial(
            state=state,
            config=config,
            params={
                "species": runtime.species,
                "energy_grid": runtime.energy_grid,
                "geometry": runtime.geometry,
                "database": runtime.database,
                "solver_parameters": runtime.solver_parameters,
            },
            model_name=model_name,
            flux_model=runtime.models.flux,
            entropy_model=entropy_model,
            amb_cfg=amb_cfg,
        )
        er_edge = float(jnp.asarray(best_roots)[-1])
        if "Er" in bc:
            bc["Er"] = dataclasses.replace(
                bc["Er"],
                right_type="dirichlet",
                right_value=jnp.asarray(er_edge),
                right_gradient=None,
            )
        if bool(runtime.solver_parameters.get("debug_stage_markers", False)):
            print(f"[NEOPAX] using ambipolar edge Er BC: Er_edge={er_edge}")

    equations_to_evolve = build_equation_system(
        config=config,
        species=runtime.species,
        field=runtime.geometry,
        flux_model=runtime.models.flux,
        source_models=runtime.models.source,
        solver_cfg=runtime.solver_parameters,
        boundary_models=bc,
    )
    solver_cfg = _apply_transport_solver_memory_heuristics(
        runtime.solver_parameters,
        state,
        len(equations_to_evolve),
    )
    shared_flux_model = runtime.models.flux if len(equations_to_evolve) > 1 else None
    temperature_active_mask = jnp.asarray(
        config.get("equations", {}).get(
            "toggle_temperature",
            [True] * getattr(runtime.species, "number_species", state.temperature.shape[0]),
        ),
        dtype=bool,
    )
    fixed_temperature_profile = state.temperature
    try:
        configured_profiles = build_profiles(
            config.get("profiles", {}),
            runtime.geometry,
            getattr(runtime.species, "number_species", state.temperature.shape[0]),
        )
        fixed_temperature_profile = configured_profiles.temperature / 1.0e3
    except Exception:
        fixed_temperature_profile = state.temperature
    equation_system = ComposedEquationSystem(
        tuple(equations_to_evolve),
        density_equation=next((eq for eq in equations_to_evolve if getattr(eq, "name", None) == "density"), None),
        temperature_equation=next((eq for eq in equations_to_evolve if getattr(eq, "name", None) == "temperature"), None),
        er_equation=next((eq for eq in equations_to_evolve if getattr(eq, "name", None) == "Er"), None),
        species=runtime.species,
        shared_flux_model=shared_flux_model,
        density_floor=solver_cfg.get("density_floor", 1.0e-6),
        temperature_floor=solver_cfg.get("temperature_floor"),
        temperature_active_mask=temperature_active_mask,
        fixed_temperature_profile=fixed_temperature_profile,
    )
    solver = build_time_solver(solver_cfg)
    backend_name = str(solver_cfg.get("transport_solver_backend", solver_cfg.get("integrator", ""))).strip().lower()
    debug_markers = bool(solver_cfg.get("debug_stage_markers", False))
    debug_disable_jit = bool(solver_cfg.get("debug_disable_jit", False))
    if debug_markers:
        print(
            "[NEOPAX] transport setup complete:",
            f"backend={solver_cfg.get('transport_solver_backend', solver_cfg.get('integrator'))}",
            f"n_equations={len(equations_to_evolve)}",
            f"state_size={_state_num_elements(state)}",
        )
        if hasattr(runtime.geometry, "Vprime") and hasattr(runtime.geometry, "r_grid"):
            total_volume = float(
                jnp.asarray(
                    jnp.trapezoid(
                        jnp.asarray(runtime.geometry.Vprime),
                        x=jnp.asarray(runtime.geometry.r_grid),
                    )
                )
            )
            print(
                "[NEOPAX] geometry integrated Vprime:",
                f"total_volume_m3={total_volume:.6e}",
            )
        try:
            turbulence_debug_model = getattr(runtime.models.flux, "turbulent_model", runtime.models.flux)
            if hasattr(turbulence_debug_model, "_effective_total_power_mw"):
                total_power_used = float(jnp.asarray(turbulence_debug_model._effective_total_power_mw(state)))
                print(
                    "[NEOPAX] turbulence power_over_n scalar power:",
                    f"total_power_mw={total_power_used:.6e}",
                )
                pressure_source_model = getattr(turbulence_debug_model, "pressure_source_model", None)
                breakdown = compute_total_power_breakdown_mw(
                    state,
                    pressure_source_model,
                    runtime.geometry,
                )
                if breakdown:
                    breakdown_text = " ".join(
                        f"{key}={float(jnp.asarray(value)):.6e}"
                        for key, value in breakdown.items()
                    )
                    print("[NEOPAX] turbulence power_over_n breakdown:", breakdown_text)
        except Exception as exc:
            print(f"[NEOPAX] turbulence power debug unavailable: {exc}")
        density_equation = next((eq for eq in equations_to_evolve if getattr(eq, "name", None) == "density"), None)
        temperature_equation = next((eq for eq in equations_to_evolve if getattr(eq, "name", None) == "temperature"), None)
        er_equation = next((eq for eq in equations_to_evolve if getattr(eq, "name", None) == "Er"), None)
        if density_equation is not None:
            if hasattr(density_equation, "active_species_mask"):
                print(
                    "[NEOPAX] density active_species_mask:",
                    jnp.asarray(density_equation.active_species_mask).tolist(),
                )
            if hasattr(density_equation, "independent_density_mask"):
                print(
                    "[NEOPAX] density independent_density_mask:",
                    jnp.asarray(density_equation.independent_density_mask).tolist(),
                )
            if hasattr(density_equation, "particle_flux_reconstruction"):
                print(
                    "[NEOPAX] density particle_flux_reconstruction:",
                    getattr(density_equation, "particle_flux_reconstruction"),
                )
            if hasattr(density_equation, "particle_face_closure_mode"):
                print(
                    "[NEOPAX] density particle_face_closure_mode:",
                    getattr(density_equation, "particle_face_closure_mode"),
                )
        if temperature_equation is not None and hasattr(temperature_equation, "active_species_mask"):
            print(
                "[NEOPAX] temperature active_species_mask:",
                jnp.asarray(temperature_equation.active_species_mask).tolist(),
            )
        if temperature_equation is not None:
            if hasattr(temperature_equation, "convection_reconstruction"):
                print(
                    "[NEOPAX] temperature convection_reconstruction:",
                    getattr(temperature_equation, "convection_reconstruction"),
                )
            if hasattr(temperature_equation, "heat_flux_reconstruction"):
                print(
                    "[NEOPAX] temperature heat_flux_reconstruction:",
                    getattr(temperature_equation, "heat_flux_reconstruction"),
                )
        rhs0 = equation_system.vector_field(jnp.asarray(0.0), state, runtime.species)
        try:
            working_state_debug, _ = equation_system._prepare_working_state(state)
        except Exception:
            working_state_debug = state
        density_rhs0 = getattr(rhs0, "density", None)
        if density_rhs0 is not None:
            density_rhs0_arr = jnp.asarray(density_rhs0)
            for i in range(density_rhs0_arr.shape[0]):
                arr = density_rhs0_arr[i]
                print(
                    f"[NEOPAX] initial density RHS summary[{i}]:",
                    f"max_abs={float(jnp.max(jnp.abs(arr))):.6e}",
                    f"min={float(jnp.min(arr)):.6e}",
                    f"max={float(jnp.max(arr)):.6e}",
                )
            if density_equation is not None:
                components = density_equation.debug_components(working_state_debug)
                for label, arr in components.items():
                    arr = jnp.asarray(arr)
                    if arr.ndim == 2:
                        for i in range(arr.shape[0]):
                            comp = arr[i]
                            finite_mask = jnp.isfinite(comp)
                            finite_count = int(jnp.sum(finite_mask))
                            total_count = comp.size
                            if finite_count > 0:
                                finite_vals = comp[finite_mask]
                                print(
                                    f"[NEOPAX] density component {label}[{i}]:",
                                    f"finite={finite_count}/{total_count}",
                                    f"min={float(jnp.min(finite_vals)):.6e}",
                                    f"max={float(jnp.max(finite_vals)):.6e}",
                                )
                            else:
                                print(
                                    f"[NEOPAX] density component {label}[{i}]:",
                                    f"finite=0/{total_count}",
                                    "all_nonfinite=true",
                                )
                    else:
                        finite_mask = jnp.isfinite(arr)
                        finite_count = int(jnp.sum(finite_mask))
                        total_count = arr.size
                        if finite_count > 0:
                            finite_vals = arr[finite_mask]
                            print(
                                f"[NEOPAX] density component {label}:",
                                f"finite={finite_count}/{total_count}",
                                f"min={float(jnp.min(finite_vals)):.6e}",
                                f"max={float(jnp.max(finite_vals)):.6e}",
                            )
                        else:
                            print(
                                f"[NEOPAX] density component {label}:",
                                f"finite=0/{total_count}",
                                "all_nonfinite=true",
                            )
                if hasattr(runtime.species, "species_idx"):
                    debug_species = [name for name in ("D", "T", "He") if name in runtime.species.species_idx]
                    center_mask = (runtime.geometry.r_grid >= 0.45) & (runtime.geometry.r_grid <= 0.65)
                    face_mask = (runtime.geometry.r_grid_half >= 0.45) & (runtime.geometry.r_grid_half <= 0.65)

                    def _fmt_window(x, y):
                        x_arr = jnp.asarray(x)
                        y_arr = jnp.asarray(y)
                        pairs = [f"({float(xv):.3f}, {float(yv):.6e})" for xv, yv in zip(x_arr.tolist(), y_arr.tolist())]
                        return "[" + ", ".join(pairs) + "]"

                    for debug_name in debug_species:
                        sidx = int(runtime.species.species_idx[debug_name])
                        gamma_faces_raw = components.get("Gamma_faces_raw", None)
                        if gamma_faces_raw is not None:
                            print(
                                f"[NEOPAX] {debug_name} debug Gamma_faces_raw window:",
                                _fmt_window(
                                    runtime.geometry.r_grid_half[face_mask],
                                    jnp.asarray(gamma_faces_raw)[sidx, face_mask],
                                ),
                            )

                        gamma_div_raw = components.get("gamma_divergence_raw", None)
                        if gamma_div_raw is not None:
                            print(
                                f"[NEOPAX] {debug_name} debug gamma_divergence_raw window:",
                                _fmt_window(
                                    runtime.geometry.r_grid[center_mask],
                                    jnp.asarray(gamma_div_raw)[sidx, center_mask],
                                ),
                            )

                        density_rhs_dbg = components.get("density_rhs", None)
                        if density_rhs_dbg is not None:
                            print(
                                f"[NEOPAX] {debug_name} debug density_rhs window:",
                                _fmt_window(
                                    runtime.geometry.r_grid[center_mask],
                                    jnp.asarray(density_rhs_dbg)[sidx, center_mask],
                                ),
                            )
                    try:
                        from ._transport_flux_models import _face_profile_gradient, build_face_transport_state

                        face_state0 = build_face_transport_state(
                            working_state_debug,
                            runtime.geometry,
                            bc_density=bc.get("density"),
                            bc_temperature=bc.get("temperature"),
                            bc_er=bc.get("Er"),
                            density_floor=solver_cfg.get("density_floor", 1.0e-6),
                            temperature_floor=solver_cfg.get("temperature_floor"),
                        )
                        dndr_faces0 = _face_profile_gradient(
                            working_state_debug.density,
                            runtime.geometry.r_grid_half,
                            bc_model=bc.get("density"),
                        )
                        dTdr_faces0 = _face_profile_gradient(
                            working_state_debug.temperature,
                            runtime.geometry.r_grid_half,
                            bc_model=bc.get("temperature"),
                        )
                        print(
                            "[NEOPAX] debug Er_faces window:",
                            _fmt_window(
                                runtime.geometry.r_grid_half[face_mask],
                                jnp.asarray(face_state0.Er)[face_mask],
                            ),
                        )
                        for debug_name in debug_species:
                            sidx = int(runtime.species.species_idx[debug_name])
                            print(
                                f"[NEOPAX] {debug_name} debug density_faces window:",
                                _fmt_window(
                                    runtime.geometry.r_grid_half[face_mask],
                                    jnp.asarray(face_state0.density)[sidx, face_mask],
                                ),
                            )
                            print(
                                f"[NEOPAX] {debug_name} debug temperature_faces window:",
                                _fmt_window(
                                    runtime.geometry.r_grid_half[face_mask],
                                    jnp.asarray(face_state0.temperature)[sidx, face_mask],
                                ),
                            )
                            print(
                                f"[NEOPAX] {debug_name} debug dndr_faces window:",
                                _fmt_window(
                                    runtime.geometry.r_grid_half[face_mask],
                                    jnp.asarray(dndr_faces0)[sidx, face_mask],
                                ),
                            )
                            print(
                                f"[NEOPAX] {debug_name} debug dTdr_faces window:",
                                _fmt_window(
                                    runtime.geometry.r_grid_half[face_mask],
                                    jnp.asarray(dTdr_faces0)[sidx, face_mask],
                                ),
                            )
                    except Exception as exc:
                        print(f"[NEOPAX] density face-state debug unavailable: {exc}")
        pressure_rhs0 = getattr(rhs0, "pressure", None)
        if pressure_rhs0 is not None:
            pressure_rhs0_arr = jnp.asarray(pressure_rhs0)
            for i in range(pressure_rhs0_arr.shape[0]):
                arr = pressure_rhs0_arr[i]
                print(
                    f"[NEOPAX] initial pressure RHS summary[{i}]:",
                    f"max_abs={float(jnp.max(jnp.abs(arr))):.6e}",
                    f"min={float(jnp.min(arr)):.6e}",
                    f"max={float(jnp.max(arr)):.6e}",
                )
            if temperature_equation is not None:
                components = temperature_equation.debug_components(working_state_debug)
                for label, arr in components.items():
                    arr = jnp.asarray(arr)
                    if arr.ndim == 2:
                        for i in range(arr.shape[0]):
                            comp = arr[i]
                            finite_mask = jnp.isfinite(comp)
                            finite_count = int(jnp.sum(finite_mask))
                            total_count = comp.size
                            if finite_count > 0:
                                finite_vals = comp[finite_mask]
                                print(
                                    f"[NEOPAX] pressure component {label}[{i}]:",
                                    f"finite={finite_count}/{total_count}",
                                    f"min={float(jnp.min(finite_vals)):.6e}",
                                    f"max={float(jnp.max(finite_vals)):.6e}",
                                )
                            else:
                                print(
                                    f"[NEOPAX] pressure component {label}[{i}]:",
                                    f"finite=0/{total_count}",
                                    "all_nonfinite=true",
                                )
                    else:
                        finite_mask = jnp.isfinite(arr)
                        finite_count = int(jnp.sum(finite_mask))
                        total_count = arr.size
                        if finite_count > 0:
                            finite_vals = arr[finite_mask]
                            print(
                                f"[NEOPAX] pressure component {label}:",
                                f"finite={finite_count}/{total_count}",
                                f"min={float(jnp.min(finite_vals)):.6e}",
                                f"max={float(jnp.max(finite_vals)):.6e}",
                            )
                        else:
                            print(
                                f"[NEOPAX] pressure component {label}:",
                                f"finite=0/{total_count}",
                                "all_nonfinite=true",
                            )
                if hasattr(runtime.species, "species_idx"):
                    debug_species = [name for name in ("D", "T", "He") if name in runtime.species.species_idx]
                    center_mask = (runtime.geometry.r_grid >= 0.45) & (runtime.geometry.r_grid <= 0.65)
                    face_mask = (runtime.geometry.r_grid_half >= 0.45) & (runtime.geometry.r_grid_half <= 0.65)

                    def _fmt_window(x, y):
                        x_arr = jnp.asarray(x)
                        y_arr = jnp.asarray(y)
                        pairs = [f"({float(xv):.3f}, {float(yv):.6e})" for xv, yv in zip(x_arr.tolist(), y_arr.tolist())]
                        return "[" + ", ".join(pairs) + "]"

                    for debug_name in debug_species:
                        sidx = int(runtime.species.species_idx[debug_name])

                        q_faces = components.get("Q_faces", None)
                        if q_faces is not None:
                            print(
                                f"[NEOPAX] {debug_name} debug Q_faces window:",
                                _fmt_window(
                                    runtime.geometry.r_grid_half[face_mask],
                                    jnp.asarray(q_faces)[sidx, face_mask],
                                ),
                            )

                        q_div = components.get("q_divergence", None)
                        if q_div is not None:
                            print(
                                f"[NEOPAX] {debug_name} debug q_divergence window:",
                                _fmt_window(
                                    runtime.geometry.r_grid[center_mask],
                                    jnp.asarray(q_div)[sidx, center_mask],
                                ),
                            )

                        thermal_rhs = components.get("thermal_flux_rhs", None)
                        if thermal_rhs is not None:
                            print(
                                f"[NEOPAX] {debug_name} debug thermal_flux_rhs window:",
                                _fmt_window(
                                    runtime.geometry.r_grid[center_mask],
                                    jnp.asarray(thermal_rhs)[sidx, center_mask],
                                ),
                            )
        er_rhs0 = getattr(rhs0, "Er", None)
        if er_rhs0 is not None:
            print(
                "[NEOPAX] initial Er RHS summary:",
                f"max_abs={float(jnp.max(jnp.abs(er_rhs0))):.6e}",
                f"min={float(jnp.min(er_rhs0)):.6e}",
                f"max={float(jnp.max(er_rhs0)):.6e}",
            )
            if er_equation is not None:
                components = er_equation.debug_components(working_state_debug)
                for label, arr in components.items():
                    arr = jnp.asarray(arr)
                    finite_mask = jnp.isfinite(arr)
                    finite_count = int(jnp.sum(finite_mask))
                    total_count = arr.size
                    if finite_count > 0:
                        finite_vals = arr[finite_mask]
                        print(
                            f"[NEOPAX] Er component {label}:",
                            f"finite={finite_count}/{total_count}",
                            f"min={float(jnp.min(finite_vals)):.6e}",
                            f"max={float(jnp.max(finite_vals)):.6e}",
                        )
                    else:
                        print(
                            f"[NEOPAX] Er component {label}:",
                            f"finite=0/{total_count}",
                            "all_nonfinite=true",
                        )
        print("[NEOPAX] entering solver.solve(...)")

    solve_state = state
    solve_vector_field = equation_system.vector_field

    def _block_until_ready_result(result_obj):
        try:
            import jax
            return jax.tree_util.tree_map(jax.block_until_ready, result_obj)
        except Exception:
            return result_obj

    solve_wall_start = None
    if debug_markers:
        solve_wall_start = time.perf_counter()

    if debug_disable_jit:
        import jax

        if debug_markers:
            print("[NEOPAX] debug_disable_jit=true, forcing eager execution for diagnosis")
        with jax.disable_jit(True):
            result = solver.solve(solve_state, solve_vector_field, runtime.species)
    else:
        result = solver.solve(solve_state, solve_vector_field, runtime.species)

    solve_wall_mid = time.perf_counter() if debug_markers else None
    if debug_markers:
        _block_until_ready_result(result)
        solve_wall_end = time.perf_counter()
        print(
            "[NEOPAX] solver timing:",
            f"host_return_elapsed_s={solve_wall_mid - solve_wall_start:.3f}",
            f"synchronized_elapsed_s={solve_wall_end - solve_wall_start:.3f}",
            f"device_tail_s={solve_wall_end - solve_wall_mid:.3f}",
        )
    if debug_markers:
        ys = getattr(result, "ys", None)
        diffrax_result = getattr(result, "result", None)
        diffrax_stats = getattr(result, "stats", None)
        if diffrax_result is not None:
            print(f"[NEOPAX] diffrax result: {diffrax_result}")
        if diffrax_stats is not None:
            try:
                print(f"[NEOPAX] diffrax stats: {diffrax_stats}")
            except Exception:
                pass
        if ys is not None:
            er_hist = getattr(ys, "Er", None)
            if er_hist is not None and jnp.asarray(er_hist).ndim >= 2:
                er_hist_arr = jnp.asarray(er_hist)
                delta = er_hist_arr[-1] - er_hist_arr[0]
                print(
                    "[NEOPAX] saved Er evolution summary:",
                    f"max_abs_delta={float(jnp.max(jnp.abs(delta))):.6e}",
                    f"max_abs_initial={float(jnp.max(jnp.abs(er_hist_arr[0]))):.6e}",
                    f"max_abs_final={float(jnp.max(jnp.abs(er_hist_arr[-1]))):.6e}",
                )
        if isinstance(result, dict):
            accepted_mask = result.get("accepted_mask", None)
            failed_mask = result.get("failed_mask", None)
            n_steps = result.get("n_steps", None)
            if accepted_mask is not None:
                accepted_count = int(jnp.sum(jnp.asarray(accepted_mask)))
                total_saved = int(jnp.asarray(accepted_mask).size)
                print(
                    "[NEOPAX] solver step summary:",
                    f"accepted_saved={accepted_count}/{total_saved}",
                    f"n_steps={int(n_steps) if n_steps is not None else 'na'}",
                    f"failed_any={bool(jnp.any(jnp.asarray(failed_mask))) if failed_mask is not None else False}",
                )
        print("[NEOPAX] solver.solve(...) returned")
    transport_cfg = config.get("transport_output", {})
    do_plot = transport_cfg.get("transport_plot", False)
    do_hdf5 = transport_cfg.get("transport_write_hdf5", False)
    do_residual_compare = transport_cfg.get("transport_compare_ambipolarity_residual", False)
    do_residual_scan = transport_cfg.get("transport_scan_ambipolarity_residual", False)
    output_dir = transport_cfg.get("transport_output_dir", None)
    plot_n_times = int(transport_cfg.get("transport_plot_n_times", 1))
    rho = runtime.geometry.rho_grid if runtime.geometry is not None and hasattr(runtime.geometry, "rho_grid") else None
    if do_plot or do_hdf5 or do_residual_compare:
        if output_dir is None:
            output_dir = Path("outputs")
        elif not isinstance(output_dir, Path):
            output_dir = Path(str(output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        if do_plot:
            plot_transport_solution(
                rho,
                result,
                output_dir,
                n_times=plot_n_times,
                reference_er_file=transport_cfg.get("transport_reference_er_file"),
                overlay_reference_er=bool(transport_cfg.get("transport_overlay_reference_er", False)),
                reference_profile_file=transport_cfg.get("transport_reference_profile_file"),
                source_models=runtime.models.source,
                species=runtime.species,
                flux_model=runtime.models.flux,
                geometry=runtime.geometry,
            )
        if do_hdf5:
            write_transport_hdf5(
                rho,
                result,
                output_dir,
                geometry=runtime.geometry,
                species=runtime.species,
                source_models=runtime.models.source,
            )
        if do_residual_compare:
            write_transport_ambipolarity_residual_comparison(
                state=state,
                runtime=runtime,
                transport_equations=equations_to_evolve,
                config=config,
                output_dir=output_dir,
            )
        if do_residual_scan:
            write_transport_ambipolarity_residual_scan(
                state=state,
                runtime=runtime,
                transport_equations=equations_to_evolve,
                config=config,
                output_dir=output_dir,
            )
    return result


def run_ambipolarity(config: dict, runtime: RuntimeContext, state: TransportState):
    result = solve_ambipolarity_roots_from_config(
        state=state,
        config=config,
        params={
            "species": runtime.species,
            "energy_grid": runtime.energy_grid,
            "geometry": runtime.geometry,
            "database": runtime.database,
            "solver_parameters": runtime.solver_parameters,
        },
        flux_model=runtime.models.flux,
    )
    if not (isinstance(result, tuple) and len(result) == 7):
        raise RuntimeError(
            "Ambipolarity solver must return a 7-tuple: "
            "(roots_all, entropies_all, best_roots, n_roots_all, do_plot, do_hdf5, output_dir)"
        )

    roots_all, entropies_all, best_roots, n_roots_all, do_plot, do_hdf5, output_dir = result
    amb_cfg = config.get("ambipolarity", {})
    rho = runtime.geometry.rho_grid if runtime.geometry is not None and hasattr(runtime.geometry, "rho_grid") else None
    roots_3, entropies_3, best_root = pad_and_sort_roots_for_plotting(
        roots_all,
        entropies_all,
        n_roots_all,
        best_roots=best_roots,
        max_roots=3,
    )

    if output_dir is None:
        output_dir = Path("outputs")
    elif not isinstance(output_dir, Path):
        output_dir = Path(str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    if do_plot:
        plot_roots(
            rho,
            roots_3,
            entropies_3,
            best_root,
            output_dir,
            overlay_reference_er=bool(amb_cfg.get("er_ambipolar_overlay_reference_er", True)),
            reference_er_file=amb_cfg.get("er_ambipolar_reference_er_file", "./examples/inputs/NTSS_Initial_Er_Opt.h5"),
            reference_er_label=amb_cfg.get("er_ambipolar_reference_er_label", "reference Er"),
        )
    if do_hdf5:
        write_ambipolarity_hdf5(rho, roots_3, entropies_3, best_root, output_dir)

    return {
        "rho": rho,
        "roots_3": roots_3,
        "entropies_3": entropies_3,
        "best_root": best_root,
        "output_dir": output_dir,
    }


def calculate_fluxes_from_config(state, config, params, flux_model=None):
    """
    Config-driven entrypoint for direct flux calculation (no root-finding).
    Returns (fluxes, do_plot, do_hdf5, output_dir)
    """
    fluxes_cfg = config.get("fluxes", {})
    if flux_model is None:
        flux_model = _build_flux_model(
            config,
            params["species"],
            params["energy_grid"],
            params["geometry"],
            params["database"],
        )
    fluxes = flux_model(state)
    do_plot = fluxes_cfg.get("fluxes_plot", False)
    do_hdf5 = fluxes_cfg.get("fluxes_write_hdf5", False)
    output_dir = fluxes_cfg.get("fluxes_output_dir", None)
    return fluxes, do_plot, do_hdf5, output_dir


def calculate_sources_from_config(state, config, params, source_models=None):
    sources_cfg = config.get("sources", {})
    if source_models is None:
        source_models = build_source_models_from_config(config, params["species"])
    source_models = source_models or {}

    density_raw = source_models.get("density")(state) if source_models.get("density") is not None else None
    pressure_raw = source_models.get("temperature")(state) if source_models.get("temperature") is not None else None

    density_components = assemble_density_source_components(density_raw, state, params["species"])
    pressure_components = assemble_pressure_source_components(pressure_raw, state, params["species"])

    sources = {
        "density_raw": density_raw,
        "pressure_raw": pressure_raw,
        "density_components": density_components,
        "pressure_components": pressure_components,
        "density_total": sum_source_components(density_components, state.density),
        "pressure_total": sum_source_components(pressure_components, state.pressure),
    }
    do_plot = sources_cfg.get("sources_plot", False)
    do_hdf5 = sources_cfg.get("sources_write_hdf5", False)
    output_dir = sources_cfg.get("sources_output_dir", None)
    return sources, do_plot, do_hdf5, output_dir


def plot_fluxes(rho, fluxes, output_dir):
    import matplotlib.pyplot as plt

    def _plot_flux_group(quantity_keys, ylabel, title, out_name):
        fig, ax = plt.subplots(figsize=(9, 4))
        plotted = False
        for key in quantity_keys:
            arr = fluxes.get(key, None)
            if arr is None:
                continue
            arr = jnp.asarray(arr)
            if arr.ndim == 2:
                for i in range(arr.shape[0]):
                    ax.plot(rho, arr[i], label=f"{key}[{i}]")
            else:
                ax.plot(rho, arr, label=key)
            plotted = True
        if not plotted:
            plt.close(fig)
            return None
        ax.set_xlabel("rho")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_png = output_dir / out_name
        fig.savefig(out_png, dpi=170)
        plt.close(fig)
        return out_png

    particle_png = _plot_flux_group(
        ["Gamma", "Gamma_neo", "Gamma_turb", "Gamma_classical"],
        "Particle Flux",
        "Particle Fluxes vs rho",
        "particle_fluxes.png",
    )
    heat_png = _plot_flux_group(
        ["Q", "Q_neo", "Q_turb", "Q_classical"],
        "Heat Flux",
        "Heat Fluxes vs rho",
        "heat_fluxes.png",
    )
    return {
        "particle": particle_png,
        "heat": heat_png,
    }


def plot_sources(rho, sources, output_dir):
    import matplotlib.pyplot as plt

    def _sanitize(name):
        return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(name))

    def _plot_species_profile(arr, ylabel, title, out_name, prefix):
        arr = jnp.asarray(arr)
        if arr.ndim != 2:
            return None
        fig, ax = plt.subplots(figsize=(9, 4))
        for i in range(arr.shape[0]):
            ax.plot(rho, arr[i], label=f"{prefix}[{i}]")
        ax.set_xlabel("rho")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_png = output_dir / out_name
        fig.savefig(out_png, dpi=170)
        plt.close(fig)
        return out_png

    written = {}
    density_total = sources.get("density_total")
    if density_total is not None and jnp.asarray(density_total).ndim == 2:
        written["density_total"] = _plot_species_profile(
            density_total,
            "Density Source",
            "Density Source Total vs rho",
            "density_sources_total.png",
            "density_total",
        )

    pressure_total = sources.get("pressure_total")
    if pressure_total is not None and jnp.asarray(pressure_total).ndim == 2:
        written["pressure_total"] = _plot_species_profile(
            pressure_total,
            "Pressure Source",
            "Pressure Source Total vs rho",
            "pressure_sources_total.png",
            "pressure_total",
        )

    for name, arr in sources.get("density_components", {}).items():
        written[f"density_{name}"] = _plot_species_profile(
            arr,
            "Density Source",
            f"Density Source: {name}",
            f"density_source_{_sanitize(name)}.png",
            name,
        )

    for name, arr in sources.get("pressure_components", {}).items():
        written[f"pressure_{name}"] = _plot_species_profile(
            arr,
            "Pressure Source",
            f"Pressure Source: {name}",
            f"pressure_source_{_sanitize(name)}.png",
            name,
        )

    return written


def write_fluxes_hdf5(rho, fluxes, output_dir):
    import h5py

    out_h5 = output_dir / "fluxes.h5"
    with h5py.File(out_h5, "w") as f:
        if rho is not None:
            f.create_dataset("rho", data=jnp.asarray(rho))
        for key, val in fluxes.items():
            f.create_dataset(key, data=jnp.asarray(val))
    return out_h5


def write_sources_hdf5(rho, sources, output_dir):
    import h5py

    out_h5 = output_dir / "sources.h5"
    with h5py.File(out_h5, "w") as f:
        if rho is not None:
            f.create_dataset("rho", data=jnp.asarray(rho))
        f.create_dataset("density_total", data=jnp.asarray(sources["density_total"]))
        f.create_dataset("pressure_total", data=jnp.asarray(sources["pressure_total"]))

        density_group = f.create_group("density_components")
        for key, value in sources.get("density_components", {}).items():
            density_group.create_dataset(key, data=jnp.asarray(value))

        pressure_group = f.create_group("pressure_components")
        for key, value in sources.get("pressure_components", {}).items():
            pressure_group.create_dataset(key, data=jnp.asarray(value))
    return out_h5


def plot_transport_solution(
    rho,
    solution,
    output_dir,
    n_times=1,
    reference_er_file=None,
    overlay_reference_er=False,
    reference_profile_file=None,
    source_models=None,
    species=None,
    flux_model=None,
    geometry=None,
):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from ._transport_flux_models import build_face_transport_state

    EV_TO_J = 1.602176634e-19
    HEAT_FLUX_W_TO_MW = 1.0e-6
    PRESSURE_SOURCE_STATE_TO_MW_M3 = 1.0 / 62.422

    heat_power_scale_center = None
    heat_power_scale_face = None
    if geometry is not None:
        if hasattr(geometry, "Vprime"):
            heat_power_scale_center = EV_TO_J * HEAT_FLUX_W_TO_MW * jnp.asarray(geometry.Vprime)
        if hasattr(geometry, "Vprime_half"):
            heat_power_scale_face = EV_TO_J * HEAT_FLUX_W_TO_MW * jnp.asarray(geometry.Vprime_half)

    def _convert_heat_flux_density_to_mw(arr):
        arr = jnp.asarray(arr)
        if heat_power_scale_center is not None and arr.shape[-1] == heat_power_scale_center.shape[0]:
            return arr * heat_power_scale_center
        if heat_power_scale_face is not None and arr.shape[-1] == heat_power_scale_face.shape[0]:
            return arr * heat_power_scale_face
        return EV_TO_J * HEAT_FLUX_W_TO_MW * arr

    def _face_to_center(arr):
        arr = jnp.asarray(arr)
        if rho is None or arr.ndim == 0:
            return arr
        if arr.shape[-1] == len(rho):
            return arr
        if arr.shape[-1] == len(rho) + 1:
            return 0.5 * (arr[..., :-1] + arr[..., 1:])
        return arr

    ys = getattr(solution, "ys", None)
    if ys is None:
        ys = solution.get("ys") if isinstance(solution, dict) else None
    if ys is None:
        return None

    ts = getattr(solution, "ts", None)
    if ts is None and isinstance(solution, dict):
        ts = solution.get("ts")

    def _select_time_slices(arr, kind):
        if arr is None:
            return []
        arr = jnp.asarray(arr)
        if rho is None or arr.ndim == 0:
            return [(None, arr)]
        radial_n = len(rho)
        if arr.shape[-1] != radial_n:
            return [(None, arr)]

        if kind == "scalar":
            if arr.ndim == 1:
                return [(None, arr)]
            if arr.ndim == 2:
                n_saved = arr.shape[0]
                if int(n_times) < 0:
                    n_pick = n_saved
                else:
                    n_pick = max(1, min(int(n_times), n_saved))
                idxs = jnp.linspace(0, n_saved - 1, n_pick).round().astype(int)
                idxs = jnp.unique(idxs)
                labels = None
                if ts is not None:
                    ts_arr = jnp.asarray(ts)
                    labels = [float(ts_arr[int(i)]) for i in idxs]
                return [
                    (labels[k] if labels is not None else None, arr[int(i)])
                    for k, i in enumerate(idxs)
                ]
            return [(None, arr)]

        # species x rho
        if arr.ndim == 2:
            return [(None, arr)]
        # time x species x rho
        if arr.ndim >= 3:
            n_saved = arr.shape[0]
            if int(n_times) < 0:
                n_pick = n_saved
            else:
                n_pick = max(1, min(int(n_times), n_saved))
            idxs = jnp.linspace(0, n_saved - 1, n_pick).round().astype(int)
            idxs = jnp.unique(idxs)
            labels = None
            if ts is not None:
                ts_arr = jnp.asarray(ts)
                labels = [float(ts_arr[int(i)]) for i in idxs]
            return [
                (labels[k] if labels is not None else None, arr[int(i)])
                for k, i in enumerate(idxs)
            ]
        return [(None, arr)]

    density_series = _select_time_slices(getattr(ys, "density", None), kind="species")
    pressure_series = _select_time_slices(getattr(ys, "pressure", None), kind="species")
    temperature_series = _select_time_slices(getattr(ys, "temperature", None), kind="species")
    er_series = _select_time_slices(getattr(ys, "Er", None), kind="scalar")
    species_names = list(getattr(species, "names", ())) if species is not None else []

    def _resolve_reference_path(path_value):
        if path_value is None:
            return None
        candidate = Path(path_value)
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        return candidate if candidate.is_file() else None

    def _load_ntss_reference_profiles(path_value):
        candidate = _resolve_reference_path(path_value)
        if candidate is None or rho is None:
            return {}
        try:
            import h5py
            import numpy as np

            with h5py.File(candidate, "r") as f:
                if "r" not in f:
                    return {}
                r_src = np.asarray(f["r"][()], dtype=float)
                if r_src.ndim != 1 or r_src.size == 0:
                    return {}
                rho_src = r_src / max(float(r_src[-1]), 1.0e-14)
                rho_dst = np.asarray(rho, dtype=float)

                def _interp_dataset(name):
                    if name not in f:
                        return None
                    values = np.asarray(f[name][()], dtype=float)
                    if values.ndim != 1:
                        return None
                    if values.shape[0] == rho_dst.shape[0] and np.allclose(rho_src, rho_dst):
                        return jnp.asarray(values)
                    return jnp.asarray(np.interp(rho_dst, rho_src, values))

                def _interp_first(*names):
                    for name in names:
                        values = _interp_dataset(name)
                        if values is not None:
                            return values
                    return None

                density_d = _interp_dataset("nD")
                profiles = {
                    "Er": _interp_dataset("Er"),
                    "density": {
                        "e": _interp_dataset("ne"),
                        "D": density_d,
                        "T": _interp_dataset("nT") if "nT" in f else density_d,
                        "He": _interp_dataset("nHe"),
                    },
                    "temperature": {
                        "e": _interp_dataset("Te"),
                        "D": _interp_dataset("TD"),
                        "T": _interp_dataset("Tt"),
                    },
                    "scalar": {},
                    "flux_species": {
                        "Gamma_neo": {},
                        "Gamma_turb": {},
                        "Q_total": {},
                        "Q_neo": {},
                        "Q_turb": {},
                    },
                }
                vr = _interp_dataset("Vr")
                flux_qe = _interp_dataset("FluxQe")
                flux_qi = _interp_dataset("FluxQI")
                flux_qd = _interp_dataset("FluxQD")
                flux_qt = _interp_dataset("FluxQT")
                flux_qe_neo = _interp_dataset("FluxQeNeo")
                flux_qi_neo = _interp_dataset("FluxQiNeo")
                flux_qd_neo = _interp_dataset("FluxQDNeo")
                flux_qt_neo = _interp_dataset("FluxQTNeo")
                flux_qe_ano = _interp_dataset("FluxQeAno")
                flux_qi_ano = _interp_dataset("FluxQiAno")
                flux_qd_ano = _interp_dataset("FluxQDAno")
                flux_qt_ano = _interp_dataset("FluxQTAno")
                flux_ge_neo = _interp_first("FluxNeo", "FluxeNeo", "FluxGeNeo")
                flux_gi_neo = _interp_first("FluxiNeo", "FluxGiNeo")
                flux_gd_neo = _interp_first("FluxDNeo", "FluxGDNeo")
                flux_gt_neo = _interp_first("FluxTNeo", "FluxGTNeo")
                flux_ge_ano = _interp_first("FluxAno", "FluxeAno", "FluxGeAno")
                flux_gi_ano = _interp_first("FluxiAno", "FluxGiAno")
                flux_gd_ano = _interp_first("FluxDAno", "FluxGDAno")
                flux_gt_ano = _interp_first("FluxTAno", "FluxGTAno")
                alpha_power = _interp_first("Pfusion", "AlphaPower")
                pbrems_power = _interp_first("Prad", "PBrems", "Pbrems")
                he_source = _interp_first("HeSource", "Hesource")
                zeff = _interp_first("Zeff")
                ecrh_power = _interp_dataset("ECRHPower")
                nbi_power_e = _interp_dataset("NBIPower_e")
                nbi_power_i = _interp_dataset("NBIPower_I")
                if pbrems_power is None:
                    ne_ref = _interp_dataset("ne")
                    te_ref = _interp_first("Te")
                    if zeff is not None and ne_ref is not None and te_ref is not None:
                        pbrems_power = (3.16e-1 * zeff * ne_ref * ne_ref * jnp.sqrt(jnp.maximum(te_ref, 0.0))) / 62.422
                if vr is not None:
                    if flux_ge_neo is not None:
                        profiles["flux_species"]["Gamma_neo"]["e"] = flux_ge_neo * vr
                    if flux_gd_neo is not None:
                        profiles["flux_species"]["Gamma_neo"]["D"] = flux_gd_neo * vr
                    if flux_gt_neo is not None:
                        profiles["flux_species"]["Gamma_neo"]["T"] = flux_gt_neo * vr
                    if flux_gi_neo is not None:
                        profiles["scalar"]["Gamma_neo_ion_sum"] = flux_gi_neo * vr
                    elif flux_gd_neo is not None and flux_gt_neo is not None:
                        profiles["scalar"]["Gamma_neo_ion_sum"] = (flux_gd_neo + flux_gt_neo) * vr
                    elif flux_ge_neo is not None:
                        profiles["scalar"]["Gamma_neo_ion_sum"] = flux_ge_neo * vr
                    if flux_ge_ano is not None:
                        profiles["flux_species"]["Gamma_turb"]["e"] = flux_ge_ano * vr
                    if flux_gd_ano is not None:
                        profiles["flux_species"]["Gamma_turb"]["D"] = flux_gd_ano * vr
                    if flux_gt_ano is not None:
                        profiles["flux_species"]["Gamma_turb"]["T"] = flux_gt_ano * vr
                    if flux_gi_ano is not None:
                        profiles["scalar"]["Gamma_turb_ion_sum"] = flux_gi_ano * vr
                    elif flux_gd_ano is not None and flux_gt_ano is not None:
                        profiles["scalar"]["Gamma_turb_ion_sum"] = (flux_gd_ano + flux_gt_ano) * vr
                    elif flux_ge_ano is not None:
                        profiles["scalar"]["Gamma_turb_ion_sum"] = flux_ge_ano * vr
                    if flux_qe is not None and flux_qi is not None:
                        profiles["scalar"]["Q_total_sum"] = (flux_qe + flux_qi) * vr
                        profiles["flux_species"]["Q_total"]["e"] = flux_qe * vr
                        if flux_qd is not None:
                            profiles["flux_species"]["Q_total"]["D"] = flux_qd * vr
                        elif density_d is not None:
                            profiles["flux_species"]["Q_total"]["D"] = flux_qi * vr
                        if flux_qt is not None:
                            profiles["flux_species"]["Q_total"]["T"] = flux_qt * vr
                        elif "nT" in f or density_d is not None:
                            profiles["flux_species"]["Q_total"]["T"] = flux_qi * vr
                        profiles["scalar"]["Q_total_ion_sum"] = flux_qi * vr
                    if flux_qe_neo is not None and flux_qi_neo is not None:
                        profiles["scalar"]["Q_neo_sum"] = (flux_qe_neo + flux_qi_neo) * vr
                        profiles["flux_species"]["Q_neo"]["e"] = flux_qe_neo * vr
                        if flux_qd_neo is not None:
                            profiles["flux_species"]["Q_neo"]["D"] = flux_qd_neo * vr
                        elif density_d is not None:
                            profiles["flux_species"]["Q_neo"]["D"] = flux_qi_neo * vr
                        if flux_qt_neo is not None:
                            profiles["flux_species"]["Q_neo"]["T"] = flux_qt_neo * vr
                        elif "nT" in f or density_d is not None:
                            profiles["flux_species"]["Q_neo"]["T"] = flux_qi_neo * vr
                        profiles["scalar"]["Q_neo_ion_sum"] = flux_qi_neo * vr
                    if flux_qe_ano is not None and flux_qi_ano is not None:
                        profiles["scalar"]["Q_turb_sum"] = (flux_qe_ano + flux_qi_ano) * vr
                        profiles["flux_species"]["Q_turb"]["e"] = flux_qe_ano * vr
                        if flux_qd_ano is not None:
                            profiles["flux_species"]["Q_turb"]["D"] = flux_qd_ano * vr
                        elif density_d is not None:
                            profiles["flux_species"]["Q_turb"]["D"] = flux_qi_ano * vr
                        if flux_qt_ano is not None:
                            profiles["flux_species"]["Q_turb"]["T"] = flux_qt_ano * vr
                        elif "nT" in f or density_d is not None:
                            profiles["flux_species"]["Q_turb"]["T"] = flux_qi_ano * vr
                        profiles["scalar"]["Q_turb_ion_sum"] = flux_qi_ano * vr
                if alpha_power is not None:
                    profiles["scalar"]["alpha_power"] = alpha_power
                if pbrems_power is not None:
                    profiles["scalar"]["pbrems_power"] = pbrems_power
                if he_source is not None:
                    profiles["scalar"]["he_source"] = he_source
                power_terms = [term for term in (alpha_power, ecrh_power, nbi_power_e, nbi_power_i) if term is not None]
                if power_terms:
                    total_power_source = power_terms[0]
                    for term in power_terms[1:]:
                        total_power_source = total_power_source + term
                    profiles["scalar"]["power_sources_total"] = total_power_source
                return profiles
        except Exception as exc:
            print(f"Could not load NTSS benchmark profiles: {exc}")
            return {}

    reference_dataset_file = reference_profile_file if reference_profile_file is not None else reference_er_file
    ntss_reference = _load_ntss_reference_profiles(reference_dataset_file)

    def _species_label(species_idx):
        if species_idx < len(species_names):
            return str(species_names[species_idx])
        return f"species[{species_idx}]"

    def _plot_species_time_series(series, ylabel, out_name):
        if not series:
            return None
        fig, ax = plt.subplots(figsize=(9, 4))
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        if not color_cycle:
            color_cycle = [f"C{i}" for i in range(max(1, len(series)))]
        species_count = int(jnp.asarray(series[0][1]).shape[0])
        linestyle_cycle = ["-", "--", ":", "-."]

        for time_idx, (time_label, values) in enumerate(series):
            color = color_cycle[time_idx % len(color_cycle)]
            for species_idx in range(values.shape[0]):
                linestyle = linestyle_cycle[species_idx % len(linestyle_cycle)]
                ax.plot(rho, values[species_idx], color=color, linestyle=linestyle, linewidth=1.8)
        reference_kind = None
        flux_reference_key = None
        out_name_lower = out_name.lower()
        if "density" in out_name_lower:
            reference_kind = "density"
        elif "temperature" in out_name_lower:
            reference_kind = "temperature"
        elif "transport_flux_q_total" in out_name_lower:
            flux_reference_key = "Q_total"
        elif "transport_flux_q_neo" in out_name_lower:
            flux_reference_key = "Q_neo"
        elif "transport_flux_q_turb" in out_name_lower:
            flux_reference_key = "Q_turb"
        elif "transport_flux_gamma_neo" in out_name_lower:
            flux_reference_key = "Gamma_neo"
        elif "transport_flux_gamma_turb" in out_name_lower:
            flux_reference_key = "Gamma_turb"
        if reference_kind is not None and ntss_reference:
            reference_profiles = ntss_reference.get(reference_kind, {})
            for species_idx in range(species_count):
                species_name = _species_label(species_idx)
                ref_values = reference_profiles.get(species_name)
                if ref_values is not None:
                    linestyle = linestyle_cycle[species_idx % len(linestyle_cycle)]
                    ax.plot(rho, ref_values, color="black", linestyle=linestyle, linewidth=2.2, alpha=0.9)
        elif flux_reference_key is not None and ntss_reference:
            reference_profiles = ntss_reference.get("flux_species", {}).get(flux_reference_key, {})
            for species_idx in range(species_count):
                species_name = _species_label(species_idx)
                ref_values = reference_profiles.get(species_name)
                if species_name in {"D", "T"} and ref_values is None and flux_reference_key in {"Gamma_neo", "Gamma_turb"}:
                    scalar_key = f"{flux_reference_key}_ion_sum"
                    ref_values = ntss_reference.get("scalar", {}).get(scalar_key)
                if ref_values is not None:
                    linestyle = linestyle_cycle[species_idx % len(linestyle_cycle)]
                    ax.plot(rho, ref_values, color="black", linestyle=linestyle, linewidth=2.2, alpha=0.9)

        time_handles = []
        for time_idx, (time_label, _) in enumerate(series):
            color = color_cycle[time_idx % len(color_cycle)]
            label = f"t={time_label:.3g}" if time_label is not None else f"series {time_idx}"
            time_handles.append(Line2D([0], [0], color=color, linestyle="-", linewidth=2.0, label=label))

        species_handles = []
        for species_idx in range(species_count):
            linestyle = linestyle_cycle[species_idx % len(linestyle_cycle)]
            species_handles.append(
                Line2D([0], [0], color="black", linestyle=linestyle, linewidth=2.0, label=_species_label(species_idx))
            )
        has_reference = False
        if reference_kind is not None:
            has_reference = any(
                ntss_reference.get(reference_kind, {}).get(_species_label(species_idx)) is not None
                for species_idx in range(species_count)
            )
        elif flux_reference_key is not None:
            reference_profiles = ntss_reference.get("flux_species", {}).get(flux_reference_key, {})
            has_reference = any(
                reference_profiles.get(_species_label(species_idx)) is not None
                or (
                    _species_label(species_idx) in {"D", "T"}
                    and flux_reference_key in {"Gamma_neo", "Gamma_turb"}
                    and ntss_reference.get("scalar", {}).get(f"{flux_reference_key}_ion_sum") is not None
                )
                for species_idx in range(species_count)
            )
        if has_reference:
            species_handles.append(
                Line2D([0], [0], color="black", linestyle="-", linewidth=2.2, label="NTSS reference")
            )

        ax.set_xlabel("rho")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        legend_times = ax.legend(handles=time_handles, title="Time", loc="upper left")
        ax.add_artist(legend_times)
        ax.legend(handles=species_handles, title="Species", loc="upper right")

        fig.tight_layout()
        out_png = output_dir / out_name
        fig.savefig(out_png, dpi=170)
        plt.close(fig)
        return out_png

    def _plot_individual_species_series(series, ylabel, out_stem):
        if not series:
            return {}
        written = {}
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        if not color_cycle:
            color_cycle = [f"C{i}" for i in range(max(1, len(series)))]
        species_count = int(jnp.asarray(series[0][1]).shape[0])
        for species_idx in range(species_count):
            fig, ax = plt.subplots(figsize=(9, 4))
            for time_idx, (time_label, values) in enumerate(series):
                color = color_cycle[time_idx % len(color_cycle)]
                label = f"t={time_label:.3g}" if time_label is not None else f"series {time_idx}"
                ax.plot(rho, values[species_idx], color=color, linewidth=1.8, label=label)
            reference_kind = "density" if "density" in out_stem.lower() else "temperature" if "temperature" in out_stem.lower() else None
            species_name = _species_label(species_idx)
            ref_values = None
            ref_label = "NTSS reference"
            if reference_kind is not None and ntss_reference:
                ref_values = ntss_reference.get(reference_kind, {}).get(species_name)
            elif "transport_flux_q_total" in out_stem.lower():
                ref_values = ntss_reference.get("flux_species", {}).get("Q_total", {}).get(species_name)
                if species_name in {"D", "T"} and ref_values is not None:
                    ref_label = "NTSS ion reference"
            elif "transport_flux_q_neo" in out_stem.lower():
                ref_values = ntss_reference.get("flux_species", {}).get("Q_neo", {}).get(species_name)
                if species_name in {"D", "T"} and ref_values is not None:
                    ref_label = "NTSS ion reference"
            elif "transport_flux_q_turb" in out_stem.lower():
                ref_values = ntss_reference.get("flux_species", {}).get("Q_turb", {}).get(species_name)
                if species_name in {"D", "T"} and ref_values is not None:
                    ref_label = "NTSS ion reference"
            elif "transport_flux_gamma_neo" in out_stem.lower():
                ref_values = ntss_reference.get("flux_species", {}).get("Gamma_neo", {}).get(species_name)
                if species_name in {"D", "T"} and ref_values is None:
                    ref_values = ntss_reference.get("scalar", {}).get("Gamma_neo_ion_sum")
                if species_name in {"D", "T"} and ref_values is not None:
                    ref_label = "NTSS ion reference"
            elif "transport_flux_gamma_turb" in out_stem.lower():
                ref_values = ntss_reference.get("flux_species", {}).get("Gamma_turb", {}).get(species_name)
                if species_name in {"D", "T"} and ref_values is None:
                    ref_values = ntss_reference.get("scalar", {}).get("Gamma_turb_ion_sum")
                if species_name in {"D", "T"} and ref_values is not None:
                    ref_label = "NTSS ion reference"
            if ref_values is not None:
                ax.plot(rho, ref_values, color="black", linewidth=2.2, linestyle="--", label=ref_label)
            ax.set_xlabel("rho")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel}: {_species_label(species_idx)}")
            ax.grid(True, alpha=0.3)
            ax.legend(title="Time")
            fig.tight_layout()
            out_png = output_dir / f"{out_stem}_{_species_label(species_idx)}.png"
            fig.savefig(out_png, dpi=170)
            plt.close(fig)
            written[_species_label(species_idx)] = out_png
        return written

    def _plot_scalar_time_series(series, ylabel, out_name, title=None, reference_key=None, reference_label="NTSS reference"):
        if not series:
            return None
        fig, ax = plt.subplots(figsize=(9, 4))
        for time_idx, (time_label, values) in enumerate(series):
            label = f"t={time_label:.3g}" if time_label is not None else f"series {time_idx}"
            ax.plot(rho, values, linewidth=1.8, label=label)
        if reference_key is not None:
            ref_values = ntss_reference.get("scalar", {}).get(reference_key)
            if ref_values is not None:
                ax.plot(rho, ref_values, color="black", linewidth=2.2, linestyle="--", label=reference_label)
        ax.set_xlabel("rho")
        ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Time")
        fig.tight_layout()
        out_png = output_dir / out_name
        fig.savefig(out_png, dpi=170)
        plt.close(fig)
        return out_png

    def _plot_pair_time_series(
        series_left,
        series_right,
        ylabel,
        out_name,
        left_label,
        right_label,
        title=None,
        reference_left_key=None,
        reference_right_key=None,
        reference_left_values=None,
        reference_right_values=None,
    ):
        if not series_left and not series_right:
            return None
        fig, ax = plt.subplots(figsize=(9, 4))
        for time_idx, (time_label, values) in enumerate(series_left):
            label = f"{left_label} t={time_label:.3g}" if time_label is not None else left_label
            ax.plot(rho, values, linewidth=1.8, label=label)
        for time_idx, (time_label, values) in enumerate(series_right):
            label = f"{right_label} t={time_label:.3g}" if time_label is not None else right_label
            ax.plot(rho, values, linewidth=1.8, linestyle="--", label=label)
        if reference_left_values is not None:
            ref_values = reference_left_values
        elif reference_left_key is not None:
            ref_values = ntss_reference.get("scalar", {}).get(reference_left_key)
        else:
            ref_values = None
        if ref_values is not None:
            ax.plot(rho, ref_values, color="black", linewidth=2.2, linestyle="-", label=f"NTSS {left_label}")
        if reference_right_values is not None:
            ref_values = reference_right_values
        elif reference_right_key is not None:
            ref_values = ntss_reference.get("scalar", {}).get(reference_right_key)
        else:
            ref_values = None
        if ref_values is not None:
            ax.plot(rho, ref_values, color="black", linewidth=2.2, linestyle=":", label=f"NTSS {right_label}")
        ax.set_xlabel("rho")
        ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Series")
        fig.tight_layout()
        out_png = output_dir / out_name
        fig.savefig(out_png, dpi=170)
        plt.close(fig)
        return out_png

    def _plot_geometry_profile(x, values, xlabel, ylabel, out_name, title=None):
        if x is None or values is None:
            return None
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(jnp.asarray(x), jnp.asarray(values), linewidth=2.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_png = output_dir / out_name
        fig.savefig(out_png, dpi=170)
        plt.close(fig)
        return out_png

    if density_series:
        _plot_species_time_series(density_series, "Density", "transport_density.png")
        _plot_individual_species_series(density_series, "Density", "transport_density")

    if temperature_series:
        _plot_species_time_series(temperature_series, "Temperature", "transport_temperature.png")
        _plot_individual_species_series(temperature_series, "Temperature", "transport_temperature")

    power_source_series = []
    he_source_series = []
    alpha_power_series = []
    pbrems_series = []
    power_exchange_series = []
    total_heat_flux_series = []
    neo_heat_flux_series = []
    turb_heat_flux_series = []
    neo_ion_heat_flux_series = []
    turb_ion_heat_flux_series = []
    neo_electron_heat_flux_series = []
    turb_electron_heat_flux_series = []
    neo_energy_like_e_series = []
    neo_energy_like_i_series = []
    turb_energy_like_e_series = []
    turb_energy_like_i_series = []
    chi_t_series = []
    chi_n_series = []
    if source_models is not None and density_series and pressure_series and er_series and species is not None:
        n_source_snapshots = min(len(density_series), len(pressure_series), len(er_series))
        for idx in range(n_source_snapshots):
            time_label = pressure_series[idx][0]
            snapshot_state = TransportState(
                density=jnp.asarray(density_series[idx][1]),
                pressure=jnp.asarray(pressure_series[idx][1]),
                Er=jnp.asarray(er_series[idx][1]),
            )
            sources, _, _, _ = calculate_sources_from_config(
                snapshot_state,
                {},
                {"species": species},
                source_models=source_models,
            )
            pressure_total = sources.get("pressure_total")
            if pressure_total is None:
                pressure_total_arr = None
            else:
                pressure_total_arr = jnp.asarray(pressure_total)
                power_source_series.append(
                    (time_label, PRESSURE_SOURCE_STATE_TO_MW_M3 * jnp.sum(pressure_total_arr, axis=0))
                )

            density_components = sources.get("density_components", {})
            density_raw = sources.get("density_raw")
            pressure_raw = sources.get("pressure_raw")
            he_component = density_components.get("HeSource")
            if he_component is None:
                raw_he = None
                if isinstance(density_raw, dict):
                    raw_he = density_raw.get("HeSource")
                if raw_he is None and isinstance(pressure_raw, dict):
                    raw_he = pressure_raw.get("HeSource")
                if raw_he is not None:
                    he_component = raw_he
            if he_component is not None:
                he_arr = jnp.asarray(he_component)
                he_idx = None
                if species is not None and hasattr(species, "species_idx"):
                    he_idx = species.species_idx.get("He")
                if he_idx is not None and he_arr.ndim == 2:
                    he_source_series.append((time_label, he_arr[int(he_idx)]))
                elif he_arr.ndim == 1:
                    he_source_series.append((time_label, he_arr))

            pressure_components = sources.get("pressure_components", {})
            power_exchange_component = pressure_components.get("power_exchange")
            if power_exchange_component is not None:
                power_exchange_arr = jnp.asarray(power_exchange_component)
                power_exchange_series.append(
                    (time_label, PRESSURE_SOURCE_STATE_TO_MW_M3 * jnp.sum(power_exchange_arr, axis=0))
                )
            alpha_component = pressure_components.get("alpha_power")
            if alpha_component is not None:
                alpha_arr = jnp.asarray(alpha_component)
                alpha_power_series.append(
                    (time_label, PRESSURE_SOURCE_STATE_TO_MW_M3 * jnp.sum(alpha_arr, axis=0))
                )
            pbrems_component = pressure_components.get("bremsstrahlung")
            if pbrems_component is not None:
                pbrems_arr = jnp.asarray(pbrems_component)
                pbrems_series.append(
                    (time_label, -PRESSURE_SOURCE_STATE_TO_MW_M3 * jnp.sum(pbrems_arr, axis=0))
                )

    flux_component_series: dict[str, list[tuple[float | None, jax.Array]]] = {
        "Gamma_neo": [],
        "Gamma_turb": [],
        "Q_total": [],
        "Q_neo": [],
        "Q_turb": [],
    }
    turbulence_plot_model = getattr(flux_model, "turbulent_model", flux_model)
    if flux_model is not None and density_series and pressure_series and er_series:
        n_flux_snapshots = min(len(density_series), len(pressure_series), len(er_series))
        for idx in range(n_flux_snapshots):
            time_label = density_series[idx][0]
            snapshot_state = TransportState(
                density=jnp.asarray(density_series[idx][1]),
                pressure=jnp.asarray(pressure_series[idx][1]),
                Er=jnp.asarray(er_series[idx][1]),
            )
            fluxes = flux_model(snapshot_state)
            q_total = fluxes.get("Q")
            if q_total is not None:
                q_total_arr = _convert_heat_flux_density_to_mw(q_total)
                flux_component_series["Q_total"].append((time_label, q_total_arr))
                total_heat_flux_series.append(
                    (time_label, jnp.sum(q_total_arr, axis=0) if q_total_arr.ndim == 2 else q_total_arr)
                )
            for key in tuple(flux_component_series.keys()):
                if key == "Q_total":
                    continue
                value = fluxes.get(key)
                if value is not None:
                    value_arr = jnp.asarray(value)
                    if key.startswith("Q_"):
                        value_arr = _convert_heat_flux_density_to_mw(value_arr)
                        scalar_sum = jnp.sum(value_arr, axis=0) if value_arr.ndim == 2 else value_arr
                        if key == "Q_neo":
                            neo_heat_flux_series.append((time_label, scalar_sum))
                            if value_arr.ndim == 2 and species is not None and hasattr(species, "species_idx"):
                                eidx = species.species_idx.get("e")
                                if eidx is not None:
                                    neo_electron_heat_flux_series.append((time_label, value_arr[int(eidx)]))
                                    ion_sum = jnp.sum(value_arr, axis=0) - value_arr[int(eidx)]
                                    neo_ion_heat_flux_series.append((time_label, ion_sum))
                        elif key == "Q_turb":
                            turb_heat_flux_series.append((time_label, scalar_sum))
                            if value_arr.ndim == 2 and species is not None and hasattr(species, "species_idx"):
                                eidx = species.species_idx.get("e")
                                if eidx is not None:
                                    turb_electron_heat_flux_series.append((time_label, value_arr[int(eidx)]))
                                    ion_sum = jnp.sum(value_arr, axis=0) - value_arr[int(eidx)]
                                    turb_ion_heat_flux_series.append((time_label, ion_sum))
                    flux_component_series[key].append((time_label, value_arr))

            if flux_model is not None and geometry is not None and species is not None and hasattr(species, "species_idx"):
                try:
                    face_state = build_face_transport_state(snapshot_state, geometry)
                    face_fluxes = flux_model.evaluate_face_fluxes(snapshot_state, face_state)
                    eidx = species.species_idx.get("e")
                    if eidx is not None and face_fluxes is not None:
                        for flux_key, gamma_key, e_store, i_store in (
                            ("Q_neo", "Gamma_neo", neo_energy_like_e_series, neo_energy_like_i_series),
                            ("Q_turb", "Gamma_turb", turb_energy_like_e_series, turb_energy_like_i_series),
                        ):
                            q_face = face_fluxes.get(flux_key)
                            g_face = face_fluxes.get(gamma_key)
                            if q_face is None or g_face is None:
                                continue
                            q_face = jnp.asarray(q_face)
                            g_face = jnp.asarray(g_face)
                            energy_like = _convert_heat_flux_density_to_mw(q_face + g_face * face_state.temperature)
                            energy_like = _face_to_center(energy_like)
                            e_store.append((time_label, energy_like[int(eidx)]))
                            i_store.append((time_label, jnp.sum(energy_like, axis=0) - energy_like[int(eidx)]))
                except Exception:
                    pass

            if turbulence_plot_model is not None and hasattr(turbulence_plot_model, "chi_t"):
                chi_t_base = jnp.asarray(getattr(turbulence_plot_model, "chi_t"))
                chi_n_base = jnp.asarray(getattr(turbulence_plot_model, "chi_n", jnp.zeros_like(chi_t_base)))
                if hasattr(turbulence_plot_model, "_effective_total_power_mw") and species is not None and hasattr(species, "species_idx"):
                    electron_idx = species.species_idx.get("e")
                    if electron_idx is not None:
                        total_power_mw = jnp.asarray(
                            turbulence_plot_model._effective_total_power_mw(snapshot_state),
                            dtype=snapshot_state.density.dtype,
                        )
                        p075 = jnp.where(
                            total_power_mw < 0.0,
                            jnp.asarray(3.0, dtype=snapshot_state.density.dtype),
                            jnp.power(total_power_mw, 0.75),
                        )
                        ne_center = jnp.maximum(
                            jnp.asarray(snapshot_state.density[int(electron_idx)], dtype=snapshot_state.density.dtype),
                            1.0e-12,
                        )
                        chi_t_arr = chi_t_base[:, None] * p075 / ne_center[None, :]
                        chi_n_arr = chi_n_base[:, None] * p075 / ne_center[None, :]
                    else:
                        chi_t_arr = jnp.repeat(chi_t_base[:, None], len(rho), axis=1)
                        chi_n_arr = jnp.repeat(chi_n_base[:, None], len(rho), axis=1)
                else:
                    chi_t_arr = jnp.repeat(chi_t_base[:, None], len(rho), axis=1)
                    chi_n_arr = jnp.repeat(chi_n_base[:, None], len(rho), axis=1)
                chi_t_series.append((time_label, chi_t_arr))
                chi_n_series.append((time_label, chi_n_arr))

    power_sources_png = _plot_scalar_time_series(
        power_source_series,
        "Total Power Source [MW/m^3]",
        "transport_power_sources_total.png",
        title="Summed Power Sources vs rho",
        reference_key="power_sources_total",
    )

    he_source_png = None
    if he_source_series:
        he_source_png = _plot_scalar_time_series(
            he_source_series,
            "Alpha Particle Source [1e20 m^-3 s^-1]",
            "transport_density_source_HeSource.png",
            title="Alpha Particle Source vs rho",
            reference_key="he_source",
        )

    total_heat_flux_png = _plot_scalar_time_series(
        total_heat_flux_series,
        "Total Heat Flux [MW]",
        "transport_flux_Q_total_sum.png",
        title="Summed Total Heat Flux vs rho",
        reference_key="Q_total_sum",
    )

    total_neo_heat_flux_png = _plot_scalar_time_series(
        neo_heat_flux_series,
        "Neo Heat Flux [MW]",
        "transport_flux_Q_neo_sum.png",
        title="Summed Neo Heat Flux vs rho",
        reference_key="Q_neo_sum",
    )

    total_turb_heat_flux_png = _plot_scalar_time_series(
        turb_heat_flux_series,
        "Turbulent Heat Flux [MW]",
        "transport_flux_Q_turb_sum.png",
        title="Summed Turbulent Heat Flux vs rho",
        reference_key="Q_turb_sum",
    )

    neo_ion_heat_flux_png = _plot_scalar_time_series(
        neo_ion_heat_flux_series,
        "Ion Neo Heat Flux [MW]",
        "transport_flux_Q_neo_ion_sum.png",
        title="Summed Ion Neo Heat Flux vs rho",
        reference_key="Q_neo_ion_sum",
        reference_label="NTSS ion reference",
    )

    turb_ion_heat_flux_png = _plot_scalar_time_series(
        turb_ion_heat_flux_series,
        "Ion Turbulent Heat Flux [MW]",
        "transport_flux_Q_turb_ion_sum.png",
        title="Summed Ion Turbulent Heat Flux vs rho",
        reference_key="Q_turb_ion_sum",
        reference_label="NTSS ion reference",
    )

    neo_q_plus_conv_png = _plot_pair_time_series(
        neo_energy_like_e_series,
        neo_energy_like_i_series,
        "Neo Energy Flux Approx. [MW]",
        "transport_flux_Q_neo_plus_conv_ei.png",
        "e",
        "ions",
        title="Neo Q + Gamma*T vs rho",
        reference_left_values=ntss_reference.get("flux_species", {}).get("Q_neo", {}).get("e"),
        reference_right_key="Q_neo_ion_sum",
    )

    turb_q_plus_conv_png = _plot_pair_time_series(
        turb_energy_like_e_series,
        turb_energy_like_i_series,
        "Turbulent Energy Flux Approx. [MW]",
        "transport_flux_Q_turb_plus_conv_ei.png",
        "e",
        "ions",
        title="Turbulent Q + Gamma*T vs rho",
        reference_left_values=ntss_reference.get("flux_species", {}).get("Q_turb", {}).get("e"),
        reference_right_key="Q_turb_ion_sum",
    )

    alpha_power_png = _plot_scalar_time_series(
        alpha_power_series,
        "Alpha Power [MW/m^3]",
        "transport_pressure_source_AlphaPower.png",
        title="Alpha Power vs rho",
        reference_key="alpha_power",
    )

    pbrems_png = _plot_scalar_time_series(
        pbrems_series,
        "Bremsstrahlung Power [MW/m^3]",
        "transport_pressure_source_PBrems.png",
        title="Bremsstrahlung Power vs rho",
        reference_key="pbrems_power",
    )

    power_exchange_png = _plot_scalar_time_series(
        power_exchange_series,
        "Power Exchange [MW/m^3]",
        "transport_pressure_source_power_exchange.png",
        title="Power Exchange vs rho",
    )

    vprime_png = _plot_geometry_profile(
        getattr(geometry, "r_grid", None) if geometry is not None else None,
        getattr(geometry, "Vprime", None) if geometry is not None else None,
        "r [m]",
        "V'(r) [m^3]",
        "transport_geometry_Vprime.png",
        title="Volume Derivative Used in Center-Grid Integrals",
    )

    vprime_half_png = _plot_geometry_profile(
        getattr(geometry, "r_grid_half", None) if geometry is not None else None,
        getattr(geometry, "Vprime_half", None) if geometry is not None else None,
        "r_face [m]",
        "V'(r_face) [m^3]",
        "transport_geometry_Vprime_half.png",
        title="Volume Derivative Used on Faces",
    )

    chi_t_png = _plot_species_time_series(
        chi_t_series,
        "Heat Diffusivity chi_t [m^2/s]",
        "transport_chi_t.png",
    ) if chi_t_series else None

    chi_n_png = _plot_species_time_series(
        chi_n_series,
        "Particle Diffusivity chi_n [m^2/s]",
        "transport_chi_n.png",
    ) if chi_n_series else None

    flux_plot_paths = {}
    flux_plot_specs = (
        ("Gamma_neo", "Neo Particle Flux", "transport_flux_Gamma_neo"),
        ("Gamma_turb", "Turbulent Particle Flux", "transport_flux_Gamma_turb"),
        ("Q_total", "Total Heat Flux [MW]", "transport_flux_Q_total"),
        ("Q_neo", "Neo Heat Flux [MW]", "transport_flux_Q_neo"),
        ("Q_turb", "Turbulent Heat Flux [MW]", "transport_flux_Q_turb"),
    )
    for key, ylabel, stem in flux_plot_specs:
        series = flux_component_series.get(key, [])
        if not series:
            continue
        flux_plot_paths[f"{key}_all"] = _plot_species_time_series(series, ylabel, f"{stem}.png")
        per_species = _plot_individual_species_series(series, ylabel, stem)
        for species_name, path in per_species.items():
            flux_plot_paths[f"{key}_{species_name}"] = path

    if er_series:
        fig, ax = plt.subplots(figsize=(9, 4))
        for time_label, er in er_series:
            label = "Er"
            if time_label is not None:
                label += f" t={time_label:.3g}"
            ax.plot(rho, er, label=label)
        if overlay_reference_er and rho is not None:
            try:
                er_ref = None
                import h5py
                import interpax

                candidate = _resolve_reference_path(
                    reference_er_file if reference_er_file is not None else reference_profile_file
                )
                if candidate is not None:
                    with h5py.File(candidate, "r") as f:
                        if "r" in f and "Er" in f:
                            r_data = f["r"][()]
                            er_data = f["Er"][()]
                            if len(er_data) != len(rho):
                                r_data = jnp.asarray(r_data)
                                rho_ref = r_data / jnp.maximum(r_data[-1], 1.0e-14)
                                er_ref = interpax.interp1d(rho_ref, er_data, rho)
                            else:
                                er_ref = er_data
                if er_ref is not None:
                    ax.plot(rho, er_ref, color="black", linewidth=2.2, linestyle="--", label=f"reference Er")
            except Exception as e:
                print(f"Could not plot transport reference Er: {e}")
        ax.set_xlabel("rho")
        ax.set_ylabel("Er")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "transport_Er.png", dpi=170)
        plt.close(fig)

    return {
        "density": output_dir / "transport_density.png" if density_series else None,
        "temperature": output_dir / "transport_temperature.png" if temperature_series else None,
        "Er": output_dir / "transport_Er.png" if er_series else None,
        "Vprime": vprime_png,
        "Vprime_half": vprime_half_png,
        "power_sources_total": power_sources_png,
        "helium_particle_source": he_source_png,
        "total_heat_flux": total_heat_flux_png,
        "neo_heat_flux": total_neo_heat_flux_png,
        "turbulent_heat_flux": total_turb_heat_flux_png,
        "alpha_power": alpha_power_png,
        "bremsstrahlung_power": pbrems_png,
        "power_exchange": power_exchange_png,
        **flux_plot_paths,
    }


def write_transport_hdf5(rho, solution, output_dir, geometry=None, species=None, source_models=None):
    import h5py

    ys = getattr(solution, "ys", None)
    if ys is None:
        ys = solution.get("ys") if isinstance(solution, dict) else None
    ts = getattr(solution, "ts", None)
    if ts is None:
        ts = solution.get("ts") if isinstance(solution, dict) else None
    dts = getattr(solution, "dts", None)
    if dts is None:
        dts = solution.get("dts") if isinstance(solution, dict) else None

    out_h5 = output_dir / "transport_solution.h5"
    with h5py.File(out_h5, "w") as f:
        if rho is not None:
            f.create_dataset("rho", data=jnp.asarray(rho))
        if ts is not None:
            f.create_dataset("ts", data=jnp.asarray(ts))
        if dts is not None:
            f.create_dataset("dts", data=jnp.asarray(dts))
        if ys is not None:
            density = getattr(ys, "density", None)
            pressure = getattr(ys, "pressure", None)
            temperature = getattr(ys, "temperature", None)
            er = getattr(ys, "Er", None)
            if density is not None:
                f.create_dataset("density", data=jnp.asarray(density))
            if pressure is not None:
                f.create_dataset("pressure", data=jnp.asarray(pressure))
            if temperature is not None:
                f.create_dataset("temperature", data=jnp.asarray(temperature))
            if er is not None:
                f.create_dataset("Er", data=jnp.asarray(er))
            if (
                density is not None
                and pressure is not None
                and er is not None
                and geometry is not None
                and species is not None
                and source_models is not None
                and source_models.get("temperature") is not None
            ):
                power_total = jax.vmap(
                    lambda dens, pres, er_prof: compute_total_power_mw(
                        TransportState(
                            density=jnp.asarray(dens),
                            pressure=jnp.asarray(pres),
                            Er=jnp.asarray(er_prof),
                        ),
                        species,
                        source_models.get("temperature"),
                        geometry,
                    )
                )(jnp.asarray(density), jnp.asarray(pressure), jnp.asarray(er))
                f.create_dataset("P_total_mw", data=jnp.asarray(power_total))
    return out_h5


def write_transport_ambipolarity_residual_comparison(state, runtime, transport_equations, config, output_dir):
    import h5py
    import jax
    import matplotlib.pyplot as plt

    from ._constants import elementary_charge
    from ._entropy_models import get_entropy_model
    from ._transport_equations import ElectricFieldEquation, _plasma_permitivity_from_prefactor

    er_equation = next((eq for eq in transport_equations if isinstance(eq, ElectricFieldEquation)), None)
    if er_equation is None:
        return None

    charge_qp = jnp.asarray(runtime.species.charge_qp)
    local_particle_flux = runtime.models.flux.build_local_particle_flux_evaluator(state)

    def _transport_charge_flux_for_state(test_state):
        if er_equation.source_mode == "ambipolar_local" and local_particle_flux is not None:
            return jax.vmap(
                lambda i, er: jnp.sum(charge_qp * local_particle_flux(i, er))
            )(jnp.arange(test_state.Er.shape[0]), test_state.Er)
        fluxes = runtime.models.flux(test_state)
        gamma = fluxes["Gamma"]
        gamma_faces = er_equation.gamma_faces_builder(gamma)
        ambipolar_flux_center = 0.5 * (gamma_faces[:, :-1] + gamma_faces[:, 1:])
        return jnp.sum(er_equation.charge_qp[:, None] * ambipolar_flux_center, axis=0)

    plasma_permitivity = _plasma_permitivity_from_prefactor(
        state,
        er_equation.species_mass,
        er_equation.permitivity_prefactor,
    )
    transport_charge_flux = _transport_charge_flux_for_state(state)
    transport_ambi_term = transport_charge_flux * elementary_charge * 1.0e-3 / plasma_permitivity

    if local_particle_flux is not None:
        local_charge_flux = jax.vmap(
            lambda i, er: jnp.sum(charge_qp * local_particle_flux(i, er))
        )(jnp.arange(state.Er.shape[0]), state.Er)
    else:
        local_charge_flux = transport_charge_flux

    rho = runtime.geometry.rho_grid if runtime.geometry is not None and hasattr(runtime.geometry, "rho_grid") else None
    out_h5 = output_dir / "transport_ambipolarity_residual_compare.h5"
    with h5py.File(out_h5, "w") as f:
        if rho is not None:
            f.create_dataset("rho", data=jnp.asarray(rho))
        f.create_dataset("transport_charge_flux", data=jnp.asarray(transport_charge_flux))
        f.create_dataset("transport_ambi_term", data=jnp.asarray(transport_ambi_term))
        f.create_dataset("ambipolar_charge_flux_local", data=jnp.asarray(local_charge_flux))

    if rho is not None:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(rho, transport_charge_flux, label="transport charge flux")
        ax.plot(rho, local_charge_flux, label="ambipolar local charge flux", linestyle="--")
        ax.set_xlabel("rho")
        ax.set_ylabel("charge-weighted flux")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "transport_ambipolarity_residual_compare.png", dpi=170)
        plt.close(fig)

    return out_h5


def write_transport_ambipolarity_residual_scan(state, runtime, transport_equations, config, output_dir):
    import dataclasses as py_dataclasses
    import h5py
    import jax
    import matplotlib.pyplot as plt

    from ._transport_equations import ElectricFieldEquation

    er_equation = next((eq for eq in transport_equations if isinstance(eq, ElectricFieldEquation)), None)
    if er_equation is None:
        return None

    transport_cfg = config.get("transport_output", {})
    scan_min = float(transport_cfg.get("transport_residual_scan_min", -50.0))
    scan_max = float(transport_cfg.get("transport_residual_scan_max", 50.0))
    n_scan = int(transport_cfg.get("transport_residual_scan_n", 101))
    scan_radii = transport_cfg.get("transport_residual_scan_radii", [0, -1])
    er_scan = jnp.linspace(scan_min, scan_max, n_scan)
    charge_qp = jnp.asarray(runtime.species.charge_qp)
    local_particle_flux = runtime.models.flux.build_local_particle_flux_evaluator(state)
    rho = runtime.geometry.rho_grid if runtime.geometry is not None and hasattr(runtime.geometry, "rho_grid") else None
    n_radial = state.Er.shape[0]

    resolved_radii = []
    for idx in scan_radii:
        i = int(idx)
        if i < 0:
            i = n_radial + i
        if 0 <= i < n_radial:
            resolved_radii.append(i)
    resolved_radii = sorted(set(resolved_radii))
    if not resolved_radii:
        resolved_radii = [0, n_radial - 1]

    out_h5 = output_dir / "transport_ambipolarity_residual_scan.h5"
    fig, axes = plt.subplots(len(resolved_radii), 1, figsize=(9, 4 * len(resolved_radii)), sharex=True)
    if len(resolved_radii) == 1:
        axes = [axes]

    with h5py.File(out_h5, "w") as f:
        f.create_dataset("Er_scan", data=jnp.asarray(er_scan))
        if rho is not None:
            f.create_dataset("rho", data=jnp.asarray(rho))

        for ax, i in zip(axes, resolved_radii):
            def transport_charge_flux_at(er_value):
                er_vec = state.Er.at[i].set(er_value)
                test_state = py_dataclasses.replace(state, Er=er_vec)
                if er_equation.source_mode == "ambipolar_local" and local_particle_flux is not None:
                    return jnp.sum(charge_qp * local_particle_flux(i, er_value))
                fluxes = runtime.models.flux(test_state)
                gamma = fluxes["Gamma"]
                gamma_faces = er_equation.gamma_faces_builder(gamma)
                ambipolar_flux_center = 0.5 * (gamma_faces[:, :-1] + gamma_faces[:, 1:])
                return jnp.sum(charge_qp * ambipolar_flux_center[:, i])

            transport_scan = jax.vmap(transport_charge_flux_at)(er_scan)

            if local_particle_flux is not None:
                ambipolar_scan = jax.vmap(
                    lambda er_value: jnp.sum(charge_qp * local_particle_flux(i, er_value))
                )(er_scan)
            else:
                ambipolar_scan = transport_scan

            group = f.create_group(f"radius_{i}")
            group.create_dataset("transport_charge_flux", data=jnp.asarray(transport_scan))
            group.create_dataset("ambipolar_charge_flux_local", data=jnp.asarray(ambipolar_scan))
            if rho is not None:
                group.attrs["rho_value"] = float(rho[i])
            group.attrs["index"] = i

            label_suffix = f"i={i}"
            if rho is not None:
                label_suffix += f", rho={float(rho[i]):.3g}"
            ax.plot(er_scan, transport_scan, label=f"transport ({label_suffix})")
            ax.plot(er_scan, ambipolar_scan, "--", label=f"ambipolar ({label_suffix})")
            ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
            ax.set_ylabel("charge-weighted flux")
            ax.legend()
            ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Er")
    fig.tight_layout()
    fig.savefig(output_dir / "transport_ambipolarity_residual_scan.png", dpi=170)
    plt.close(fig)
    return out_h5


def main(config_path):
    config = load_config(config_path)
    runtime, state = build_runtime_context(config)
    general = config.get("general", {})
    mode = general.get("mode", config.get("mode", "transport")).lower()

    if mode == "transport":
        return run_transport(config, runtime, state)

    if mode == "ambipolarity":
        return run_ambipolarity(config, runtime, state)

    if mode == "fluxes":
        fluxes, do_plot, do_hdf5, output_dir = calculate_fluxes_from_config(
            state,
            config,
            {
                "species": runtime.species,
                "energy_grid": runtime.energy_grid,
                "geometry": runtime.geometry,
                "database": runtime.database,
                "solver_parameters": runtime.solver_parameters,
            },
            flux_model=runtime.models.flux,
        )
        rho = runtime.geometry.rho_grid if runtime.geometry is not None and hasattr(runtime.geometry, "rho_grid") else None
        if output_dir is None:
            output_dir = Path("outputs")
        elif not isinstance(output_dir, Path):
            output_dir = Path(str(output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        if do_plot:
            plot_fluxes(rho, fluxes, output_dir)
        if do_hdf5:
            write_fluxes_hdf5(rho, fluxes, output_dir)
        return {"rho": rho, "fluxes": fluxes, "output_dir": output_dir}

    if mode == "sources":
        sources, do_plot, do_hdf5, output_dir = calculate_sources_from_config(
            state,
            config,
            {
                "species": runtime.species,
                "energy_grid": runtime.energy_grid,
                "geometry": runtime.geometry,
                "database": runtime.database,
                "solver_parameters": runtime.solver_parameters,
            },
            source_models=runtime.models.source,
        )
        rho = runtime.geometry.rho_grid if runtime.geometry is not None and hasattr(runtime.geometry, "rho_grid") else None
        if output_dir is None:
            output_dir = Path("outputs")
        elif not isinstance(output_dir, Path):
            output_dir = Path(str(output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        if do_plot:
            plot_sources(rho, sources, output_dir)
        if do_hdf5:
            write_sources_hdf5(rho, sources, output_dir)
        return {"rho": rho, "sources": sources, "output_dir": output_dir}

    raise ValueError(f"Unknown mode '{mode}'. Supported: 'ambipolarity', 'transport', 'fluxes', 'sources'.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m NEOPAX.main <config.toml>")
        sys.exit(1)
    main(sys.argv[1])
