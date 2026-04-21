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
    with open(path, "rb") as f:
        return toml.load(f)


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
        else neoclassical_factory(species, energy_grid, geometry, database)
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
    equation_system = ComposedEquationSystem(
        tuple(equations_to_evolve),
        species=runtime.species,
        shared_flux_model=shared_flux_model,
        density_floor=solver_cfg.get("density_floor", 1.0e-6),
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
        density_equation = next((eq for eq in equations_to_evolve if getattr(eq, "name", None) == "density"), None)
        temperature_equation = next((eq for eq in equations_to_evolve if getattr(eq, "name", None) == "temperature"), None)
        er_equation = next((eq for eq in equations_to_evolve if getattr(eq, "name", None) == "Er"), None)
        rhs0 = equation_system.vector_field(jnp.asarray(0.0), state, runtime.species)
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
                components = density_equation.debug_components(state)
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
                components = temperature_equation.debug_components(state)
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
        er_rhs0 = getattr(rhs0, "Er", None)
        if er_rhs0 is not None:
            print(
                "[NEOPAX] initial Er RHS summary:",
                f"max_abs={float(jnp.max(jnp.abs(er_rhs0))):.6e}",
                f"min={float(jnp.min(er_rhs0)):.6e}",
                f"max={float(jnp.max(er_rhs0)):.6e}",
            )
            if er_equation is not None:
                components = er_equation.debug_components(state)
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
                source_models=runtime.models.source,
                species=runtime.species,
                flux_model=runtime.models.flux,
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
        plot_roots(rho, roots_3, entropies_3, best_root, output_dir)
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
    source_models=None,
    species=None,
    flux_model=None,
):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

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

    def _plot_scalar_time_series(series, ylabel, out_name, title=None):
        if not series:
            return None
        fig, ax = plt.subplots(figsize=(9, 4))
        for time_idx, (time_label, values) in enumerate(series):
            label = f"t={time_label:.3g}" if time_label is not None else f"series {time_idx}"
            ax.plot(rho, values, linewidth=1.8, label=label)
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

    if density_series:
        _plot_species_time_series(density_series, "Density", "transport_density.png")
        _plot_individual_species_series(density_series, "Density", "transport_density")

    if temperature_series:
        _plot_species_time_series(temperature_series, "Temperature", "transport_temperature.png")
        _plot_individual_species_series(temperature_series, "Temperature", "transport_temperature")

    power_source_series = []
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
                continue
            power_source_series.append((time_label, jnp.sum(jnp.asarray(pressure_total), axis=0)))

    flux_component_series: dict[str, list[tuple[float | None, jax.Array]]] = {
        "Gamma_neo": [],
        "Gamma_turb": [],
        "Q_neo": [],
        "Q_turb": [],
    }
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
            for key in tuple(flux_component_series.keys()):
                value = fluxes.get(key)
                if value is not None:
                    flux_component_series[key].append((time_label, jnp.asarray(value)))

    power_sources_png = _plot_scalar_time_series(
        power_source_series,
        "Total Power Source",
        "transport_power_sources_total.png",
        title="Summed Power Sources vs rho",
    )

    flux_plot_paths = {}
    flux_plot_specs = (
        ("Gamma_neo", "Neo Particle Flux", "transport_flux_Gamma_neo"),
        ("Gamma_turb", "Turbulent Particle Flux", "transport_flux_Gamma_turb"),
        ("Q_neo", "Neo Heat Flux", "transport_flux_Q_neo"),
        ("Q_turb", "Turbulent Heat Flux", "transport_flux_Q_turb"),
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
                import h5py
                import interpax

                if reference_er_file is None:
                    candidate = output_dir / "../inputs/NTSS_Initial_Er_Opt.h5"
                else:
                    candidate = Path(reference_er_file)
                    if not candidate.is_absolute():
                        candidate = (Path.cwd() / candidate).resolve()
                if candidate.is_file():
                    with h5py.File(candidate, "r") as f:
                        r_data = f["r"][()]
                        er_data = f["Er"][()]
                    if len(er_data) != len(rho):
                        er_ref = interpax.interp1d(r_data, er_data, rho)
                    else:
                        er_ref = er_data
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
        "power_sources_total": power_sources_png,
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
