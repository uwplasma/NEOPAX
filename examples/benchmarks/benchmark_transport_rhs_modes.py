"""Benchmark shared transport rhs modes on selectable NEOPAX flux models.

Example:

    python examples/benchmarks/benchmark_transport_rhs_modes.py

    python examples/benchmarks/benchmark_transport_rhs_modes.py \
        --config examples/Solve_Transport_Equations/Solve_Transport_equations_noHe_radau.toml \
        --backend theta_newton \
        --flux-model ntx_database \
        --warmup 1 \
        --repeat 2
"""

from __future__ import annotations

import argparse
import copy
import os
import time
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import NEOPAX
from NEOPAX._orchestrator import build_runtime_context, _normalized_boundary_cfg_for_transport
from NEOPAX._boundary_conditions import build_boundary_condition_model
from NEOPAX._transport_equations import ComposedEquationSystem, build_equation_system
from NEOPAX._transport_solvers import (
    _RADAU_STAGE_CONFIGS,
    _extract_fixed_temperature_projection,
    _extract_state_regularization,
    _flat_rhs_factory,
    _flat_rhs_with_lagged_response_factory,
    _lagged_response_hooks,
    _make_radau_stage_predictor,
    _make_solver_state_transform,
    _project_flat_state_if_needed,
    _project_state_to_quasi_neutrality,
)


DEFAULT_CONFIG = Path("examples/benchmarks/Solve_Transport_equations_noHe_radau_benchmark.toml")
DEFAULT_FLUX_MODEL = "ntx_exact_lij_runtime"
DEFAULT_ER_INIT_MODE = "keep"
DEFAULT_NTX_EXACT_N_THETA = 5
DEFAULT_NTX_EXACT_N_ZETA = 21
DEFAULT_NTX_EXACT_N_XI = 32
DEFAULT_NTX_EXACT_FACE_RESPONSE_MODE = "interpolate_center_response"
DEFAULT_NTX_EXACT_RADIAL_BATCH_SIZE = None
DEFAULT_NTX_EXACT_RADIAL_BATCH_MODE = "simple"
DEFAULT_NTX_EXACT_SCAN_BATCH_SIZE = None
DEFAULT_NTX_EXACT_RESPONSE_ANCHOR_COUNT = None
DEFAULT_NTX_EXACT_USE_REMAT = False
DEFAULT_RHS_MODES = ["black_box", "lagged_response", "lagged_linear_state"]


def _block_on_result(result) -> None:
    for candidate in (result.final_time, result.n_steps, result.done, result.fail_code):
        if candidate is not None:
            jax.block_until_ready(candidate)
            return
    if result.accepted_mask is not None:
        jax.block_until_ready(result.accepted_mask)


def _tree_to_host_numpy(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), tree)


def _leaf_max_abs_delta(reference, other) -> float:
    ref_leaves = jax.tree_util.tree_leaves(reference)
    other_leaves = jax.tree_util.tree_leaves(other)
    if len(ref_leaves) != len(other_leaves):
        return float("nan")
    max_delta = 0.0
    for ref_leaf, other_leaf in zip(ref_leaves, other_leaves):
        ref_arr = np.asarray(ref_leaf)
        other_arr = np.asarray(other_leaf)
        if ref_arr.shape != other_arr.shape:
            return float("nan")
        delta = float(np.max(np.abs(other_arr - ref_arr)))
        max_delta = max(max_delta, delta)
    return max_delta


def _accepted_count(result) -> int | None:
    if result.accepted_mask is None:
        return None
    return int(jnp.sum(jnp.asarray(result.accepted_mask, dtype=jnp.int32)))


def _build_config(
    config_path: Path,
    *,
    backend: str,
    device: str,
    rhs_mode: str,
    flux_model: str,
    er_init_mode: str,
    n_theta: int | None,
    n_zeta: int | None,
    n_xi: int | None,
    ntx_face_response_mode: str,
    ntx_radial_batch_size: int | None,
    ntx_radial_batch_mode: str,
    ntx_scan_batch_size: int | None,
    ntx_response_anchor_count: int | None,
    ntx_use_remat: bool,
    radau_newton_divergence_mode: str,
    radau_newton_residual_norm: str,
    debug_initial_finiteness: bool = False,
    single_attempt: bool = False,
    max_steps: int | None = None,
    stop_after_accepted_steps: int | None = None,
):
    config = NEOPAX.prepare_config(config_path, backend=backend, device=device)
    config = copy.deepcopy(config)

    general = config.setdefault("general", {})
    general["mode"] = "transport"

    neoclassical = config.setdefault("neoclassical", {})
    if flux_model != "keep":
        neoclassical["flux_model"] = str(flux_model)
    active_flux_model = neoclassical.get("flux_model", "ntx_database")

    profiles = config.setdefault("profiles", {})
    if er_init_mode != "keep":
        profiles["er_initialization_mode"] = str(er_init_mode)

    if active_flux_model == "ntx_exact_lij_runtime":
        if n_theta is None:
            n_theta = neoclassical.get("ntx_exact_n_theta", DEFAULT_NTX_EXACT_N_THETA)
        if n_zeta is None:
            n_zeta = neoclassical.get("ntx_exact_n_zeta", DEFAULT_NTX_EXACT_N_ZETA)
        if n_xi is None:
            n_xi = neoclassical.get("ntx_exact_n_xi", DEFAULT_NTX_EXACT_N_XI)
        if n_theta is not None:
            neoclassical["ntx_exact_n_theta"] = int(n_theta)
        if n_zeta is not None:
            neoclassical["ntx_exact_n_zeta"] = int(n_zeta)
        if n_xi is not None:
            neoclassical["ntx_exact_n_xi"] = int(n_xi)
        if ntx_radial_batch_size not in (None, 0):
            neoclassical["ntx_exact_radial_batch_size"] = int(ntx_radial_batch_size)
        if str(ntx_radial_batch_mode).strip().lower() != DEFAULT_NTX_EXACT_RADIAL_BATCH_MODE:
            neoclassical["ntx_exact_radial_batch_mode"] = str(ntx_radial_batch_mode)
        if ntx_scan_batch_size not in (None, 0):
            neoclassical["ntx_exact_scan_batch_size"] = int(ntx_scan_batch_size)
        if ntx_response_anchor_count not in (None, 0):
            neoclassical["ntx_exact_response_anchor_count"] = int(ntx_response_anchor_count)
        if ntx_use_remat:
            neoclassical["ntx_exact_use_remat"] = True
        neoclassical["ntx_exact_face_response_mode"] = str(ntx_face_response_mode)

    ambipolarity = config.setdefault("ambipolarity", {})
    ambipolarity["er_ambipolar_plot"] = False
    ambipolarity["er_ambipolar_overlay_reference_er"] = False

    solver = config.setdefault("transport_solver", {})
    solver["transport_solver_backend"] = str(backend)
    solver["rhs_mode"] = str(rhs_mode)
    solver["debug_stage_markers"] = bool(debug_initial_finiteness)
    if str(backend).strip().lower() == "radau":
        solver["radau_rhs_mode"] = str(rhs_mode)
        solver["radau_newton_divergence_mode"] = str(radau_newton_divergence_mode)
        solver["radau_newton_residual_norm"] = str(radau_newton_residual_norm)
    if single_attempt:
        solver["max_steps"] = 1
        solver.pop("stop_after_accepted_steps", None)
    else:
        if max_steps is not None:
            solver["max_steps"] = int(max_steps)
        if stop_after_accepted_steps is not None:
            solver["stop_after_accepted_steps"] = int(stop_after_accepted_steps)

    transport_output = config.setdefault("transport_output", {})
    transport_output["transport_plot"] = False
    transport_output["transport_write_hdf5"] = False

    return config, active_flux_model


def _run_once(config):
    t_start = time.perf_counter()
    result = NEOPAX.run(config)
    _block_on_result(result)
    wall_seconds = time.perf_counter() - t_start
    raw = result.raw_result if hasattr(result, "raw_result") else None
    if isinstance(raw, dict):
        last_attempt_accepted = raw.get("last_attempt_accepted")
        last_attempt_converged = raw.get("last_attempt_converged")
        last_attempt_err_norm = raw.get("last_attempt_err_norm")
        last_attempt_fail_code = raw.get("last_attempt_fail_code")
        last_attempt_diverged = raw.get("last_attempt_diverged")
        last_attempt_nonfinite_stage_state = raw.get("last_attempt_nonfinite_stage_state")
        last_attempt_nonfinite_stage_residual = raw.get("last_attempt_nonfinite_stage_residual")
        last_attempt_finite_f0 = raw.get("last_attempt_finite_f0")
        last_attempt_finite_z0 = raw.get("last_attempt_finite_z0")
        last_attempt_finite_initial_residual = raw.get("last_attempt_finite_initial_residual")
        last_attempt_newton_iter_count = raw.get("last_attempt_newton_iter_count")
        last_attempt_final_residual_norm = raw.get("last_attempt_final_residual_norm")
        last_attempt_final_delta_norm = raw.get("last_attempt_final_delta_norm")
        last_attempt_theta_final = raw.get("last_attempt_theta_final")
        last_attempt_slow_contraction = raw.get("last_attempt_slow_contraction")
        last_attempt_residual_blowup = raw.get("last_attempt_residual_blowup")
        last_attempt_newton_nonfinite = raw.get("last_attempt_newton_nonfinite")
        if any(value is not None for value in (last_attempt_accepted, last_attempt_converged, last_attempt_err_norm, last_attempt_fail_code)):
            err_text = "None" if last_attempt_err_norm is None else f"{float(last_attempt_err_norm):.6e}"
            final_residual_text = "None" if last_attempt_final_residual_norm is None else f"{float(last_attempt_final_residual_norm):.6e}"
            final_delta_text = "None" if last_attempt_final_delta_norm is None else f"{float(last_attempt_final_delta_norm):.6e}"
            theta_text = "None" if last_attempt_theta_final is None else f"{float(last_attempt_theta_final):.6e}"
            print(
                "[benchmark] last_attempt:",
                f"accepted={last_attempt_accepted}",
                f"converged={last_attempt_converged}",
                f"fail_code={last_attempt_fail_code}",
                f"err_norm={err_text}",
                f"diverged={last_attempt_diverged}",
                f"nonfinite_stage_state={last_attempt_nonfinite_stage_state}",
                f"nonfinite_stage_residual={last_attempt_nonfinite_stage_residual}",
                f"finite_f0={last_attempt_finite_f0}",
                f"finite_z0={last_attempt_finite_z0}",
                f"finite_initial_residual={last_attempt_finite_initial_residual}",
                f"newton_iter_count={last_attempt_newton_iter_count}",
                f"final_residual_norm={final_residual_text}",
                f"final_delta_norm={final_delta_text}",
                f"theta_final={theta_text}",
                f"slow_contraction={last_attempt_slow_contraction}",
                f"residual_blowup={last_attempt_residual_blowup}",
                f"newton_nonfinite={last_attempt_newton_nonfinite}",
            )
    return result, wall_seconds


def _print_array_finiteness(label: str, arr) -> None:
    arr = jnp.asarray(arr)
    finite_mask = jnp.isfinite(arr)
    finite_count = int(jnp.sum(finite_mask))
    total = int(arr.size)
    if finite_count == 0:
        print(f"[benchmark] {label}: finite=0/{total} all_nonfinite=true")
        return
    vals = arr[finite_mask]
    message = (
        f"[benchmark] {label}: finite={finite_count}/{total} "
        f"min={float(jnp.min(vals)):.6e} max={float(jnp.max(vals)):.6e}"
    )
    if finite_count < total:
        nonfinite_idx = np.argwhere(~np.asarray(jax.device_get(finite_mask)))
        preview = nonfinite_idx[:6].tolist()
        message += f" nonfinite_idx={preview}"
    print(message)


def _print_tree_finiteness(prefix: str, tree) -> None:
    if isinstance(tree, dict):
        for key, value in tree.items():
            _print_array_finiteness(f"{prefix}.{key}", value)
        return
    _print_array_finiteness(prefix, tree)


def _print_initial_finiteness_probe(config: dict) -> None:
    runtime, state = build_runtime_context(config)
    if runtime.geometry is None or state is None:
        print("[benchmark] initial_probe: geometry_or_state_unavailable")
        return

    field = runtime.geometry
    boundary_cfg = _normalized_boundary_cfg_for_transport(config.get("boundary", {}))
    bc = {}
    dr = getattr(field, "dr", 1.0)
    for key in ("density", "temperature", "Er", "gamma"):
        if key in boundary_cfg:
            bc[key] = build_boundary_condition_model(
                boundary_cfg[key],
                dr,
                species_names=runtime.species.names if key in {"density", "temperature", "gamma"} else None,
            )

    equations_to_evolve = build_equation_system(
        config=config,
        species=runtime.species,
        field=runtime.geometry,
        flux_model=runtime.models.flux,
        source_models=runtime.models.source,
        solver_cfg=runtime.solver_parameters,
        boundary_models=bc,
    )
    shared_flux_model = runtime.models.flux if len(equations_to_evolve) > 1 else None
    temperature_active_mask = jnp.asarray(
        config.get("equations", {}).get(
            "toggle_temperature",
            [True] * getattr(runtime.species, "number_species", state.temperature.shape[0]),
        ),
        dtype=bool,
    )
    equation_system = ComposedEquationSystem(
        tuple(equations_to_evolve),
        density_equation=next((eq for eq in equations_to_evolve if getattr(eq, "name", None) == "density"), None),
        temperature_equation=next((eq for eq in equations_to_evolve if getattr(eq, "name", None) == "temperature"), None),
        er_equation=next((eq for eq in equations_to_evolve if getattr(eq, "name", None) == "Er"), None),
        species=runtime.species,
        shared_flux_model=shared_flux_model,
        density_floor=runtime.solver_parameters.get("density_floor", 1.0e-6),
        temperature_floor=runtime.solver_parameters.get("temperature_floor"),
        temperature_active_mask=temperature_active_mask,
        fixed_temperature_profile=state.temperature,
        er_bc_model=bc.get("Er"),
    )

    working_state, _ = equation_system._prepare_working_state(state)
    print("[benchmark] initial_probe: start")
    _print_array_finiteness("initial_probe.state.Er", state.Er)
    _print_array_finiteness("initial_probe.working_state.Er", working_state.Er)

    center_fluxes = None
    if shared_flux_model is not None:
        center_fluxes = shared_flux_model(working_state)
        _print_tree_finiteness("initial_probe.center_flux", center_fluxes)

    density_eq, temperature_eq, er_eq = equation_system._resolve_equations()
    if density_eq is not None:
        _print_tree_finiteness("initial_probe.density", density_eq.debug_components(working_state, fluxes=center_fluxes))
    if temperature_eq is not None:
        _print_tree_finiteness("initial_probe.temperature", temperature_eq.debug_components(working_state, fluxes=center_fluxes))
    if er_eq is not None:
        _print_tree_finiteness("initial_probe.er", er_eq.debug_components(working_state, fluxes=center_fluxes))

    rhs0 = equation_system.vector_field(jnp.asarray(0.0), state, runtime.species)
    _print_array_finiteness("initial_probe.rhs0.density", rhs0.density)
    _print_array_finiteness("initial_probe.rhs0.pressure", rhs0.pressure)
    _print_array_finiteness("initial_probe.rhs0.Er", rhs0.Er)

    solver_cfg = config.get("transport_solver", {})
    backend = str(solver_cfg.get("transport_solver_backend", solver_cfg.get("integrator", ""))).strip().lower()
    if backend == "radau":
        num_stages = int(solver_cfg.get("radau_num_stages", 3))
        stage_cfg = _RADAU_STAGE_CONFIGS.get(num_stages)
        if stage_cfg is not None:
            temperature_active_mask_solver, fixed_temperature_profile_solver = _extract_fixed_temperature_projection(
                equation_system.vector_field
            )
            density_floor, temperature_floor = _extract_state_regularization(equation_system.vector_field)
            solver_state0 = _project_state_to_quasi_neutrality(
                state,
                runtime.species,
                temperature_active_mask=temperature_active_mask_solver,
                fixed_temperature_profile=fixed_temperature_profile_solver,
                density_floor=density_floor,
                temperature_floor=temperature_floor,
            )
            flat_state0, unpack_flat, _unpack_packed, _pack_state, project_flat = _make_solver_state_transform(
                solver_state0,
                runtime.species,
                temperature_active_mask=temperature_active_mask_solver,
                fixed_temperature_profile=fixed_temperature_profile_solver,
                density_floor=density_floor,
                temperature_floor=temperature_floor,
            )
            flat_rhs = _flat_rhs_factory(
                unpack_flat,
                equation_system.vector_field,
                (runtime.species,),
                {},
                project_flat=project_flat,
            )
            build_lagged_response, _ = _lagged_response_hooks(equation_system.vector_field)
            flat_rhs_with_lagged_response = _flat_rhs_with_lagged_response_factory(
                unpack_flat,
                equation_system.vector_field,
                (runtime.species,),
                {},
                project_flat=project_flat,
            )
            rhs_mode_norm = str(solver_cfg.get("radau_rhs_mode", "black_box")).strip().lower()
            use_transport_lagged_response = rhs_mode_norm in {"lagged_transport_response", "lagged_response"}
            dtype = flat_state0.dtype
            c = jnp.asarray(stage_cfg.c, dtype=dtype)
            a = jnp.asarray(stage_cfg.a, dtype=dtype)
            t0 = jnp.asarray(solver_cfg.get("t0", 0.0), dtype=dtype)
            t_final = jnp.asarray(solver_cfg.get("t_final", solver_cfg.get("t1", 0.0)), dtype=dtype)
            dt_min = jnp.asarray(solver_cfg.get("min_step", 1.0e-14), dtype=dtype)
            dt_max = jnp.asarray(solver_cfg.get("max_step", solver_cfg.get("dt", 1.0)), dtype=dtype)
            base_dt = jnp.clip(jnp.asarray(solver_cfg.get("dt", 0.0), dtype=dtype), dt_min, dt_max)
            h_value = jnp.minimum(base_dt, t_final - t0)
            f0_flat = flat_rhs(t0, flat_state0)
            prev_stages = jnp.tile(f0_flat, num_stages)
            lagged_response0 = (
                build_lagged_response(unpack_flat(_project_flat_state_if_needed(flat_state0, project_flat)))
                if (use_transport_lagged_response and build_lagged_response is not None)
                else None
            )

            def _stage_eval_flat(t_value, flat_y):
                if lagged_response0 is not None:
                    return flat_rhs_with_lagged_response(t_value, flat_y, lagged_response0)
                return flat_rhs(t_value, flat_y)

            z0 = _make_radau_stage_predictor(
                f0_flat,
                prev_stages,
                jnp.asarray(0.0, dtype=dtype),
                h_value,
                c,
                dtype,
            )
            _print_array_finiteness("initial_probe.radau.flat_state0", flat_state0)
            _print_array_finiteness("initial_probe.radau.f0_flat", f0_flat)
            _print_array_finiteness("initial_probe.radau.z0", z0)
            stages0 = z0.reshape((num_stages, flat_state0.shape[0]))
            stage_states = flat_state0[None, :] + h_value * (a @ stages0)
            stage_times = t0 + c * h_value
            _print_array_finiteness("initial_probe.radau.stage_times", stage_times)
            for i in range(num_stages):
                stage_time_i = stage_times[i]
                stage_state_i = stage_states[i]
                projected_stage_i = _project_flat_state_if_needed(stage_state_i, project_flat)
                stage_rhs_i = _stage_eval_flat(stage_time_i, stage_state_i)
                stage_residual_i = stages0[i] - stage_rhs_i
                _print_array_finiteness(f"initial_probe.radau.stage_state[{i}]", stage_state_i)
                _print_array_finiteness(f"initial_probe.radau.stage_state_projected[{i}]", projected_stage_i)
                _print_array_finiteness(f"initial_probe.radau.stage_rhs[{i}]", stage_rhs_i)
                _print_array_finiteness(f"initial_probe.radau.stage_residual[{i}]", stage_residual_i)

            jacobian_ref = jax.jacfwd(lambda y: _stage_eval_flat(t0, y))(flat_state0)
            _print_array_finiteness("initial_probe.radau.jacobian_ref", jacobian_ref)

            h_jacobian = h_value * jacobian_ref
            identity_n = jnp.eye(flat_state0.shape[0], dtype=dtype)
            radau_real_eig = jnp.asarray(stage_cfg.real_eig, dtype=dtype)
            radau_complex_blocks = jnp.asarray(stage_cfg.complex_blocks, dtype=dtype)
            real_matrix = identity_n - radau_real_eig * h_jacobian
            _print_array_finiteness("initial_probe.radau.real_matrix", real_matrix)
            real_lu, real_piv = jax.scipy.linalg.lu_factor(real_matrix)
            _print_array_finiteness("initial_probe.radau.real_lu", real_lu)
            _print_array_finiteness("initial_probe.radau.real_piv", real_piv)

            if int(stage_cfg.complex_blocks.shape[0]) > 0:
                identity_2 = jnp.eye(2, dtype=dtype)
                complex_dim = 2 * flat_state0.shape[0]
                complex_dense_all = jnp.transpose(
                    identity_2[None, :, :, None, None] * identity_n[None, None, None, :, :]
                    - radau_complex_blocks[:, :, :, None, None] * h_jacobian[None, None, None, :, :],
                    (0, 1, 3, 2, 4),
                ).reshape((int(stage_cfg.complex_blocks.shape[0]), complex_dim, complex_dim))
                for i in range(int(stage_cfg.complex_blocks.shape[0])):
                    _print_array_finiteness(f"initial_probe.radau.complex_matrix[{i}]", complex_dense_all[i])
                    complex_lu_i, complex_piv_i = jax.scipy.linalg.lu_factor(complex_dense_all[i])
                    _print_array_finiteness(f"initial_probe.radau.complex_lu[{i}]", complex_lu_i)
                    _print_array_finiteness(f"initial_probe.radau.complex_piv[{i}]", complex_piv_i)

            radau_transform = jnp.asarray(stage_cfg.transform, dtype=dtype)
            radau_inv_transform = jnp.asarray(stage_cfg.inv_transform, dtype=dtype)
            residual0 = (stages0 - jax.vmap(_stage_eval_flat, in_axes=(0, 0))(stage_times, stage_states)).reshape((-1,))
            _print_array_finiteness("initial_probe.radau.residual0", residual0)
            rhs_stages0 = (-residual0).reshape((num_stages, flat_state0.shape[0]))
            rhs_transformed0 = radau_inv_transform @ rhs_stages0
            _print_array_finiteness("initial_probe.radau.rhs_transformed0", rhs_transformed0)
            rhs_real0 = rhs_transformed0[0]
            delta_real0 = jax.scipy.linalg.lu_solve((real_lu, real_piv), rhs_real0)
            _print_array_finiteness("initial_probe.radau.delta_real0", delta_real0)

            delta_complex_rows = []
            for i in range(int(stage_cfg.complex_blocks.shape[0])):
                rhs_complex_i = rhs_transformed0[1:].reshape((int(stage_cfg.complex_blocks.shape[0]), 2, flat_state0.shape[0]))[i]
                delta_complex_i = jax.scipy.linalg.lu_solve(
                    jax.scipy.linalg.lu_factor(complex_dense_all[i]),
                    rhs_complex_i.reshape((-1,)),
                ).reshape((2, flat_state0.shape[0]))
                _print_array_finiteness(f"initial_probe.radau.delta_complex[{i}]", delta_complex_i)
                delta_complex_rows.append(delta_complex_i)

            if delta_complex_rows:
                delta_transformed0 = jnp.concatenate(
                    [delta_real0[None, :], jnp.asarray(delta_complex_rows).reshape((2 * int(stage_cfg.complex_blocks.shape[0]), flat_state0.shape[0]))],
                    axis=0,
                )
            else:
                delta_transformed0 = delta_real0[None, :]
            _print_array_finiteness("initial_probe.radau.delta_transformed0", delta_transformed0)
            def _explicit_newton_update(z_cur, label: str):
                stages_cur = z_cur.reshape((num_stages, flat_state0.shape[0]))
                stage_states_cur = flat_state0[None, :] + h_value * (a @ stages_cur)
                stage_rhs_cur = jax.vmap(_stage_eval_flat, in_axes=(0, 0))(stage_times, stage_states_cur)
                residual_cur = (stages_cur - stage_rhs_cur).reshape((-1,))
                _print_array_finiteness(f"initial_probe.radau.stage_rhs{label}", stage_rhs_cur)
                _print_array_finiteness(f"initial_probe.radau.residual{label}", residual_cur)

                rhs_stages_cur = (-residual_cur).reshape((num_stages, flat_state0.shape[0]))
                rhs_transformed_cur = radau_inv_transform @ rhs_stages_cur
                _print_array_finiteness(f"initial_probe.radau.rhs_transformed{label}", rhs_transformed_cur)

                rhs_real_cur = rhs_transformed_cur[0]
                delta_real_cur = jax.scipy.linalg.lu_solve((real_lu, real_piv), rhs_real_cur)
                _print_array_finiteness(f"initial_probe.radau.delta_real{label}", delta_real_cur)

                delta_complex_rows_cur = []
                rhs_complex_blocks_cur = rhs_transformed_cur[1:].reshape(
                    (int(stage_cfg.complex_blocks.shape[0]), 2, flat_state0.shape[0])
                )
                for j in range(int(stage_cfg.complex_blocks.shape[0])):
                    delta_complex_cur = jax.scipy.linalg.lu_solve(
                        jax.scipy.linalg.lu_factor(complex_dense_all[j]),
                        rhs_complex_blocks_cur[j].reshape((-1,)),
                    ).reshape((2, flat_state0.shape[0]))
                    _print_array_finiteness(f"initial_probe.radau.delta_complex{label}[{j}]", delta_complex_cur)
                    delta_complex_rows_cur.append(delta_complex_cur)

                if delta_complex_rows_cur:
                    delta_transformed_cur = jnp.concatenate(
                        [
                            delta_real_cur[None, :],
                            jnp.asarray(delta_complex_rows_cur).reshape(
                                (2 * int(stage_cfg.complex_blocks.shape[0]), flat_state0.shape[0])
                            ),
                        ],
                        axis=0,
                    )
                else:
                    delta_transformed_cur = delta_real_cur[None, :]
                _print_array_finiteness(f"initial_probe.radau.delta_transformed{label}", delta_transformed_cur)
                delta_cur = (radau_transform @ delta_transformed_cur).reshape((-1,))
                _print_array_finiteness(f"initial_probe.radau.delta{label}", delta_cur)
                z_next = z_cur + delta_cur
                _print_array_finiteness(f"initial_probe.radau.z_next{label}", z_next)
                return z_next

            delta0 = (radau_transform @ delta_transformed0).reshape((-1,))
            _print_array_finiteness("initial_probe.radau.delta0", delta0)
            z1 = z0 + delta0
            _print_array_finiteness("initial_probe.radau.z1", z1)
            z2 = _explicit_newton_update(z1, "1")
            z3 = _explicit_newton_update(z2, "2")
            _print_array_finiteness("initial_probe.radau.z3", z3)

            tol = jnp.asarray(solver_cfg.get("nonlinear_solver_tol", solver_cfg.get("tol", 1.0e-8)), dtype=dtype)
            maxiter = int(solver_cfg.get("nonlinear_solver_maxiter", solver_cfg.get("maxiter", 20)))
            tiny_scalar = jnp.asarray(1.0e-30, dtype=dtype)
            zero_scalar = jnp.asarray(0.0, dtype=dtype)
            theta_diverge_threshold = jnp.asarray(0.99, dtype=dtype)
            residual_blowup_factor = jnp.asarray(2.0, dtype=dtype)
            newton_shrink_num = jnp.asarray(0.8, dtype=dtype)
            newton_shrink_min = jnp.asarray(0.1, dtype=dtype)
            newton_shrink_max = jnp.asarray(0.5, dtype=dtype)
            residual_norm_mode = str(solver_cfg.get("radau_newton_residual_norm", "raw")).strip().lower()
            newton_tol_mode = str(solver_cfg.get("radau_newton_tol_mode", "residual")).strip().lower()
            fnewt_mode = str(solver_cfg.get("radau_newton_fnewt_mode", "tol")).strip().lower()
            use_rms_residual_norm = residual_norm_mode in {"rms", "scaled", "normalized"}
            use_hairer_newton_tol = newton_tol_mode in {"hairer", "hairer_like", "ntss"}
            use_hairer_scaled_correction = use_hairer_newton_tol or fnewt_mode in {"hairer", "hairer_like", "ntss"}
            residual_size_sqrt = jnp.sqrt(jnp.asarray(num_stages * flat_state0.shape[0], dtype=dtype))
            state_scale_base = jnp.asarray(solver_cfg.get("atol", 1.0e-8), dtype=dtype) + jnp.asarray(
                solver_cfg.get("rtol", 1.0e-6), dtype=dtype
            ) * jnp.abs(flat_state0)
            stage_scale = jnp.broadcast_to(jnp.maximum(state_scale_base, tiny_scalar)[None, :], (num_stages, flat_state0.shape[0])).reshape((-1,))

            def _residual_norm_value(residual_vec):
                raw_norm = jnp.linalg.norm(residual_vec)
                return jnp.where(
                    use_rms_residual_norm,
                    raw_norm / jnp.maximum(residual_size_sqrt, jnp.asarray(1.0, dtype=dtype)),
                    raw_norm,
                )

            def _correction_norm_value(delta_vec):
                raw_norm = jnp.linalg.norm(delta_vec)
                scaled_norm = jnp.sqrt(jnp.mean((delta_vec / stage_scale) * (delta_vec / stage_scale)) + tiny_scalar)
                return jnp.where(use_hairer_scaled_correction, scaled_norm, raw_norm)

            def _residual_flat(z_flat):
                stages = z_flat.reshape((num_stages, flat_state0.shape[0]))
                stage_states = flat_state0[None, :] + h_value * (a @ stages)
                evals = jax.vmap(_stage_eval_flat, in_axes=(0, 0))(stage_times, stage_states)
                return (stages - evals).reshape((-1,))

            def _stage_solver(rhs):
                rhs_stages = rhs.reshape((num_stages, flat_state0.shape[0]))
                rhs_transformed = radau_inv_transform @ rhs_stages
                rhs_real = rhs_transformed[0]
                delta_real = jax.scipy.linalg.lu_solve((real_lu, real_piv), rhs_real)
                rhs_complex_pairs = rhs_transformed[1:].reshape(
                    (int(stage_cfg.complex_blocks.shape[0]), 2, flat_state0.shape[0])
                )
                delta_complex_pairs = []
                for j in range(int(stage_cfg.complex_blocks.shape[0])):
                    delta_complex_pair = jax.scipy.linalg.lu_solve(
                        jax.scipy.linalg.lu_factor(complex_dense_all[j]),
                        rhs_complex_pairs[j].reshape((-1,)),
                    ).reshape((2, flat_state0.shape[0]))
                    delta_complex_pairs.append(delta_complex_pair)
                if delta_complex_pairs:
                    delta_transformed = jnp.concatenate(
                        [
                            delta_real[None, :],
                            jnp.asarray(delta_complex_pairs).reshape(
                                (2 * int(stage_cfg.complex_blocks.shape[0]), flat_state0.shape[0])
                            ),
                        ],
                        axis=0,
                    )
                else:
                    delta_transformed = delta_real[None, :]
                return (radau_transform @ delta_transformed).reshape((-1,))

            predictor_defect_floor = jnp.asarray(1.0e-4, dtype=dtype)
            predictor_defect_cap = jnp.asarray(20.0, dtype=dtype)
            if fnewt_mode in {"hairer", "hairer_like", "ntss"}:
                uround = jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)
                expmns = jnp.asarray((num_stages + 1.0) / (2.0 * num_stages), dtype=dtype)
                safe_rtol = jnp.maximum(jnp.asarray(solver_cfg.get("rtol", 1.0e-6), dtype=dtype), uround * 10.0)
                rtol1 = jnp.asarray(0.1, dtype=dtype) * (safe_rtol ** expmns)
                expmi = jnp.asarray(1.0, dtype=dtype) / expmns
                predictor_fnewt = jnp.maximum(
                    jnp.asarray(10.0, dtype=dtype) * uround / rtol1,
                    jnp.minimum(
                        jnp.asarray(3.0e-2, dtype=dtype),
                        rtol1 ** (expmi - jnp.asarray(1.0, dtype=dtype)),
                    ),
                )
            else:
                predictor_fnewt = jnp.maximum(tol, tiny_scalar)

            def _probe_body(newton_state):
                (
                    iter_idx,
                    z_cur,
                    delta_norm,
                    residual_norm,
                    prev_newton_norm,
                    newton_metric,
                    prev_theta_ratio,
                    theta_est,
                    diverged,
                    shrink_suggest,
                    slow_contraction_any,
                    residual_blowup_any,
                    newton_nonfinite_any,
                ) = newton_state
                residual_cur = _residual_flat(z_cur)
                delta = _stage_solver(-residual_cur)
                delta = jnp.where(jnp.all(jnp.isfinite(delta)), delta, jnp.zeros_like(delta))
                z_next = z_cur + delta
                current_residual_norm = _residual_norm_value(residual_cur)
                current_delta_norm = jnp.linalg.norm(delta)
                current_newton_norm = _correction_norm_value(delta)
                safe_prev_delta = jnp.maximum(prev_newton_norm, tiny_scalar)
                theta_raw = current_newton_norm / safe_prev_delta
                newton_iter_num = iter_idx + jnp.asarray(1, dtype=jnp.int32)
                theta_candidate = jnp.where(
                    newton_iter_num == 2,
                    theta_raw,
                    jnp.sqrt(jnp.maximum(theta_raw * prev_theta_ratio, tiny_scalar)),
                )
                theta_valid = newton_iter_num > 1
                theta_candidate = jnp.where(theta_valid, theta_candidate, zero_scalar)
                theta_next = jnp.where(theta_valid, theta_candidate, theta_est)
                theta_ratio_next = jnp.where(theta_valid, theta_raw, prev_theta_ratio)
                residual_blowup = jnp.logical_and(iter_idx >= 1, current_residual_norm > residual_norm * residual_blowup_factor)
                nonfinite_state = jnp.logical_not(
                    jnp.logical_and(
                        jnp.logical_and(jnp.all(jnp.isfinite(delta)), jnp.isfinite(current_residual_norm)),
                        jnp.logical_and(jnp.isfinite(current_delta_norm), jnp.isfinite(current_newton_norm)),
                    )
                )
                predictor_active = jnp.logical_and(theta_valid, newton_iter_num < maxiter)
                remaining_iters = jnp.maximum(maxiter - 1 - newton_iter_num, jnp.asarray(0, dtype=jnp.int32))
                faccon = theta_candidate / jnp.maximum(jnp.asarray(1.0, dtype=dtype) - theta_candidate, tiny_scalar)
                predicted_defect = faccon * current_newton_norm * (theta_candidate ** remaining_iters) / predictor_fnewt
                qnewt = jnp.clip(predicted_defect, predictor_defect_floor, predictor_defect_cap)
                predictor_exponent = -jnp.asarray(1.0, dtype=dtype) / (jnp.asarray(maxiter + 3, dtype=dtype) - newton_iter_num.astype(dtype))
                predictor_shrink = jnp.clip(
                    newton_shrink_num * (qnewt ** predictor_exponent),
                    newton_shrink_min,
                    newton_shrink_max,
                )
                slow_contraction = jnp.logical_and(
                    predictor_active,
                    jnp.where(
                        theta_candidate < theta_diverge_threshold,
                        predicted_defect >= jnp.asarray(1.0, dtype=dtype),
                        jnp.asarray(True),
                    ),
                )
                convergence_metric = jnp.where(theta_valid, faccon * current_newton_norm, current_newton_norm)
                predictor_shrink = jnp.where(theta_candidate < theta_diverge_threshold, predictor_shrink, jnp.asarray(0.5, dtype=dtype))
                shrink_suggest_next = jnp.where(slow_contraction, predictor_shrink, shrink_suggest)
                diverged_next = jnp.logical_or(
                    diverged,
                    jnp.logical_or(
                        slow_contraction,
                        jnp.logical_or(residual_blowup, nonfinite_state),
                    ),
                )
                return (
                    iter_idx + 1,
                    z_next,
                    current_delta_norm,
                    current_residual_norm,
                    current_newton_norm,
                    convergence_metric,
                    theta_ratio_next,
                    theta_next,
                    diverged_next,
                    shrink_suggest_next,
                    jnp.logical_or(slow_contraction_any, slow_contraction),
                    jnp.logical_or(residual_blowup_any, residual_blowup),
                    jnp.logical_or(newton_nonfinite_any, nonfinite_state),
                )

            def _probe_cond(newton_state):
                iter_idx, _, delta_norm, residual_norm, _, newton_metric, _, _, diverged, _, _, _, _ = newton_state
                active = jnp.where(
                    use_hairer_newton_tol,
                    newton_metric > predictor_fnewt,
                    jnp.logical_or(residual_norm > tol, delta_norm > tol),
                )
                return jnp.logical_and(jnp.logical_and(iter_idx < maxiter, active), jnp.logical_not(diverged))

            probe_init = (
                jnp.asarray(0, dtype=jnp.int32),
                z0,
                jnp.asarray(jnp.inf, dtype=dtype),
                jnp.asarray(jnp.inf, dtype=dtype),
                jnp.asarray(jnp.inf, dtype=dtype),
                jnp.asarray(jnp.inf, dtype=dtype),
                zero_scalar,
                zero_scalar,
                jnp.asarray(False),
                jnp.asarray(1.0, dtype=dtype),
                jnp.asarray(False),
                jnp.asarray(False),
                jnp.asarray(False),
            )
            probe_final = jax.lax.while_loop(_probe_cond, _probe_body, probe_init)
            probe_iter, probe_z, probe_delta_norm, probe_residual_norm, _probe_prev_newton, probe_newton_metric, _probe_prev_theta_ratio, probe_theta, probe_diverged, _probe_shrink_suggest, probe_slow_any, probe_blowup_any, probe_nonfinite_any = probe_final
            probe_final_residual = _residual_flat(probe_z)
            _print_array_finiteness("initial_probe.radau.while_loop_z", probe_z)
            _print_array_finiteness("initial_probe.radau.while_loop_final_residual", probe_final_residual)
            print(
                "[benchmark] initial_probe.radau.while_loop:",
                f"iter={int(probe_iter)}",
                f"delta_norm={float(probe_delta_norm):.6e}",
                f"residual_norm={float(probe_residual_norm):.6e}",
                f"newton_metric={float(probe_newton_metric):.6e}",
                f"newton_tol_mode={newton_tol_mode}",
                f"predictor_fnewt={float(predictor_fnewt):.6e}",
                f"theta={float(probe_theta):.6e}",
                f"diverged={bool(probe_diverged)}",
                f"slow_any={bool(probe_slow_any)}",
                f"blowup_any={bool(probe_blowup_any)}",
                f"nonfinite_any={bool(probe_nonfinite_any)}",
            )
    print("[benchmark] initial_probe: end")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Base transport config to benchmark. Default: no-He Radau benchmark case.",
    )
    parser.add_argument(
        "--backend",
        default="radau",
        choices=["radau", "theta", "theta_newton"],
        help="Transport solver backend to benchmark.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Execution device to request from NEOPAX/JAX.",
    )
    parser.add_argument(
        "--flux-model",
        default=DEFAULT_FLUX_MODEL,
        choices=["keep", "ntx_database", "ntx_exact_lij_runtime", "ntx_scan_runtime"],
        help=(
            "Neoclassical flux model to benchmark. "
            "'keep' preserves whatever the base TOML already uses."
        ),
    )
    parser.add_argument(
        "--er-init-mode",
        default=DEFAULT_ER_INIT_MODE,
        choices=["keep", "analytical", "ambipolar_min_entropy"],
        help=(
            "Override profiles.er_initialization_mode for the benchmark. "
            "Default is 'keep', so the benchmark TOML controls whether Er starts "
            "from the ambipolar root finder or an analytical profile."
        ),
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per rhs_mode before timing.")
    parser.add_argument("--repeat", type=int, default=1, help="Timed runs per rhs_mode.")
    parser.add_argument(
        "--rhs-modes",
        nargs="+",
        default=DEFAULT_RHS_MODES,
        help="RHS modes to compare.",
    )
    parser.add_argument(
        "--ntx-n-theta",
        type=int,
        default=None,
        help="Override ntx_exact_n_theta when benchmarking ntx_exact_lij_runtime.",
    )
    parser.add_argument(
        "--ntx-n-zeta",
        type=int,
        default=None,
        help="Override ntx_exact_n_zeta when benchmarking ntx_exact_lij_runtime.",
    )
    parser.add_argument(
        "--ntx-n-xi",
        type=int,
        default=None,
        help="Override ntx_exact_n_xi when benchmarking ntx_exact_lij_runtime.",
    )
    parser.add_argument(
        "--ntx-face-response-mode",
        default=DEFAULT_NTX_EXACT_FACE_RESPONSE_MODE,
        choices=["face_local_response", "interpolate_center_response"],
        help=(
            "Exact-runtime NTX face response mode to use for lagged_response benchmarks. "
            "Black-box mode continues to use the normal live face-local path."
        ),
    )
    parser.add_argument(
        "--ntx-radial-batch-size",
        type=int,
        default=DEFAULT_NTX_EXACT_RADIAL_BATCH_SIZE,
        help=(
            "Exact-runtime NTX radial batch size. "
            "With radial-batch-mode=simple, unset/0 keeps lax.map and values >1 use vmap. "
            "With radial-batch-mode=hybrid, values >1 use chunked lax.map+vmap."
        ),
    )
    parser.add_argument(
        "--ntx-radial-batch-mode",
        default=DEFAULT_NTX_EXACT_RADIAL_BATCH_MODE,
        choices=["simple", "lax_map", "vmap", "hybrid"],
        help=(
            "Exact-runtime NTX radial mapper. "
            "simple: current default (lax.map if batch unset, otherwise vmap). "
            "lax_map: always lax.map. "
            "vmap: always full vmap over the provided radii. "
            "hybrid: chunked lax.map over radial batches with vmap inside each chunk."
        ),
    )
    parser.add_argument(
        "--ntx-scan-batch-size",
        type=int,
        default=DEFAULT_NTX_EXACT_SCAN_BATCH_SIZE,
        help=(
            "Exact-runtime NTX coefficient-scan batch size across the energy-grid cases. "
            "Unset or 0 keeps the default full vmap over n_x; values >1 apply NTX-style "
            "chunking to reduce peak memory."
        ),
    )
    parser.add_argument(
        "--ntx-response-anchor-count",
        type=int,
        default=DEFAULT_NTX_EXACT_RESPONSE_ANCHOR_COUNT,
        help=(
            "If set below the full transport radial count, build the lagged exact-runtime NTX "
            "response only on that many radial anchor points and interpolate the reduced "
            "transport-moment response back to the full transport grid."
        ),
    )
    parser.add_argument(
        "--ntx-response-anchor-counts",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Run a sweep over multiple exact-runtime lagged-response anchor counts. "
            "Useful for comparisons such as 7 vs 14 anchors. "
            "When set, each anchor count is benchmarked as a separate configuration."
        ),
    )
    parser.add_argument(
        "--ntx-use-remat",
        action="store_true",
        default=DEFAULT_NTX_EXACT_USE_REMAT,
        help=(
            "Enable jax.checkpoint/rematerialization on the local exact-runtime NTX solve path. "
            "This can reduce peak memory at the cost of extra recomputation."
        ),
    )
    parser.add_argument(
        "--compute-final-state-delta",
        action="store_true",
        help=(
            "Compute max|delta| of final_state versus the first rhs_mode. "
            "Disabled by default because materializing the full final state can be very expensive."
        ),
    )
    parser.add_argument(
        "--debug-lagged-timing",
        action="store_true",
        help=(
            "Print runtime timing callbacks for lagged-response build/evaluation sections. "
            "Useful to separate compile delay from actual lagged NTX work."
        ),
    )
    parser.add_argument(
        "--single-attempt",
        action="store_true",
        help=(
            "Benchmark exactly one solver step attempt by forcing transport_solver.max_steps=1 "
            "and clearing stop_after_accepted_steps."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Override transport_solver.max_steps for the benchmark run. "
            "Ignored when --single-attempt is used."
        ),
    )
    parser.add_argument(
        "--stop-after-accepted-steps",
        type=int,
        default=None,
        help=(
            "Override transport_solver.stop_after_accepted_steps for the benchmark run. "
            "Useful to allow retries and stop after the first accepted step. "
            "Ignored when --single-attempt is used."
        ),
    )
    parser.add_argument(
        "--debug-initial-finiteness",
        action="store_true",
        help=(
            "Before each timed run, probe the initialized transport state and print finiteness "
            "summaries for center fluxes, equation debug components, and rhs0."
        ),
    )
    parser.add_argument(
        "--radau-newton-divergence-mode",
        default="legacy",
        choices=["legacy", "conservative"],
        help="Custom Radau Newton divergence policy. 'conservative' is less aggressive and more Hairer-like.",
    )
    parser.add_argument(
        "--radau-newton-residual-norm",
        default="raw",
        choices=["raw", "rms"],
        help="Residual norm used by the custom Radau Newton convergence test.",
    )
    args = parser.parse_args()

    if args.debug_lagged_timing:
        os.environ["NEOPAX_DEBUG_LAGGED_TIMING"] = "1"
    else:
        os.environ.pop("NEOPAX_DEBUG_LAGGED_TIMING", None)

    config_path = Path(args.config)
    sweep_anchor_counts = args.ntx_response_anchor_counts
    if sweep_anchor_counts is None:
        sweep_anchor_counts = [args.ntx_response_anchor_count]

    mode_results = []
    reference_final_state = None

    print(f"[benchmark] config={config_path}")
    print(f"[benchmark] backend={args.backend}")
    print(f"[benchmark] device={args.device}")
    print(f"[benchmark] requested_flux_model={args.flux_model}")
    print(f"[benchmark] er_init_mode={args.er_init_mode}")
    print(f"[benchmark] ntx_face_response_mode={args.ntx_face_response_mode}")
    print(f"[benchmark] ntx_radial_batch_size={args.ntx_radial_batch_size}")
    print(f"[benchmark] ntx_radial_batch_mode={args.ntx_radial_batch_mode}")
    print(f"[benchmark] ntx_scan_batch_size={args.ntx_scan_batch_size}")
    print(f"[benchmark] ntx_response_anchor_counts={sweep_anchor_counts}")
    print(f"[benchmark] ntx_use_remat={args.ntx_use_remat}")
    print(f"[benchmark] radau_newton_divergence_mode={args.radau_newton_divergence_mode}")
    print(f"[benchmark] radau_newton_residual_norm={args.radau_newton_residual_norm}")
    print(f"[benchmark] compute_final_state_delta={args.compute_final_state_delta}")
    print(f"[benchmark] debug_lagged_timing={args.debug_lagged_timing}")
    print(f"[benchmark] single_attempt={args.single_attempt}")
    print(f"[benchmark] max_steps_override={args.max_steps}")
    print(f"[benchmark] stop_after_accepted_steps_override={args.stop_after_accepted_steps}")
    print(f"[benchmark] debug_initial_finiteness={args.debug_initial_finiteness}")
    print(f"[benchmark] rhs_modes={args.rhs_modes}")

    for sweep_anchor_count in sweep_anchor_counts:
        active_flux_model = None
        for rhs_mode in args.rhs_modes:
            config, active_flux_model = _build_config(
                config_path,
                backend=args.backend,
                device=args.device,
                rhs_mode=rhs_mode,
                flux_model=args.flux_model,
                er_init_mode=args.er_init_mode,
                n_theta=args.ntx_n_theta,
                n_zeta=args.ntx_n_zeta,
                n_xi=args.ntx_n_xi,
                ntx_face_response_mode=args.ntx_face_response_mode,
                ntx_radial_batch_size=args.ntx_radial_batch_size,
                ntx_radial_batch_mode=args.ntx_radial_batch_mode,
                ntx_scan_batch_size=args.ntx_scan_batch_size,
                ntx_response_anchor_count=sweep_anchor_count,
                ntx_use_remat=args.ntx_use_remat,
                radau_newton_divergence_mode=args.radau_newton_divergence_mode,
                radau_newton_residual_norm=args.radau_newton_residual_norm,
                debug_initial_finiteness=args.debug_initial_finiteness,
                single_attempt=args.single_attempt,
                max_steps=args.max_steps,
                stop_after_accepted_steps=args.stop_after_accepted_steps,
            )

            if rhs_mode == args.rhs_modes[0]:
                print(f"[benchmark] active_flux_model={active_flux_model}")
                if active_flux_model == "ntx_exact_lij_runtime":
                    print(
                        "[benchmark] ntx exact runtime resolution:",
                        f"n_theta={config['neoclassical'].get('ntx_exact_n_theta')}",
                        f"n_zeta={config['neoclassical'].get('ntx_exact_n_zeta')}",
                        f"n_xi={config['neoclassical'].get('ntx_exact_n_xi')}",
                        f"radial_batch_size={config['neoclassical'].get('ntx_exact_radial_batch_size', 0)}",
                        f"radial_batch_mode={config['neoclassical'].get('ntx_exact_radial_batch_mode', 'simple')}",
                        f"scan_batch_size={config['neoclassical'].get('ntx_exact_scan_batch_size', 0)}",
                        f"response_anchor_count={config['neoclassical'].get('ntx_exact_response_anchor_count', 0)}",
                        f"use_remat={config['neoclassical'].get('ntx_exact_use_remat', False)}",
                    )
                if active_flux_model == "ntx_exact_lij_runtime":
                    print(
                        "[benchmark] ntx exact runtime face response mode:",
                        config["neoclassical"].get("ntx_exact_face_response_mode", "face_local_response"),
                    )
                    anchor_count = config["neoclassical"].get("ntx_exact_response_anchor_count", 0)
                    if int(anchor_count) > 0:
                        print(
                            "[benchmark] ntx exact runtime lagged response mode:",
                            f"coarse_anchor_interpolated(anchor_count={anchor_count}, derivatives=Er+log_nu_star)",
                        )
                    else:
                        print(
                            "[benchmark] ntx exact runtime lagged response mode:",
                            "full_radius_response",
                        )

            for _ in range(max(0, args.warmup)):
                print(
                    f"[benchmark] warmup compile/run rhs_mode={rhs_mode}"
                    f" anchor_count={sweep_anchor_count}"
                )
                _run_once(config)

            timed_runs = []
            last_result = None
            for _ in range(max(1, args.repeat)):
                if args.debug_initial_finiteness:
                    _print_initial_finiteness_probe(config)
                last_result, wall_seconds = _run_once(config)
                timed_runs.append(wall_seconds)
                if last_result is not None and not args.debug_initial_finiteness:
                    raw = last_result.raw_result if hasattr(last_result, "raw_result") else None
                    if isinstance(raw, dict):
                        if bool(raw.get("last_attempt_nonfinite_stage_residual", False)):
                            print(
                                "[benchmark] detected nonfinite stage residual; "
                                "printing initial RHS finiteness probe for diagnosis"
                            )
                            _print_initial_finiteness_probe(config)

            final_state_delta = None
            if args.compute_final_state_delta and last_result is not None:
                if reference_final_state is None:
                    reference_final_state = _tree_to_host_numpy(last_result.final_state)
                else:
                    final_state_delta = _leaf_max_abs_delta(reference_final_state, _tree_to_host_numpy(last_result.final_state))

            mode_results.append(
                {
                    "rhs_mode": rhs_mode,
                    "flux_model": active_flux_model,
                    "response_anchor_count": sweep_anchor_count,
                    "mean_wall_seconds": sum(timed_runs) / len(timed_runs),
                    "best_wall_seconds": min(timed_runs),
                    "n_steps": None if last_result is None or last_result.n_steps is None else int(last_result.n_steps),
                    "accepted_steps": None if last_result is None else _accepted_count(last_result),
                    "failed": None if last_result is None or last_result.failed is None else bool(last_result.failed),
                    "fail_code": None if last_result is None or last_result.fail_code is None else int(last_result.fail_code),
                    "final_time": None if last_result is None or last_result.final_time is None else float(last_result.final_time),
                    "last_attempt_accepted": None if last_result is None else last_result.raw_result.get("last_attempt_accepted"),
                    "last_attempt_converged": None if last_result is None else last_result.raw_result.get("last_attempt_converged"),
                    "last_attempt_err_norm": None if last_result is None else last_result.raw_result.get("last_attempt_err_norm"),
                    "last_attempt_fail_code": None if last_result is None else last_result.raw_result.get("last_attempt_fail_code"),
                    "final_state_max_abs_delta_vs_first": final_state_delta,
                }
            )

    print()
    print("rhs_mode                 anchors     mean_s     best_s     n_steps  accepted  failed  fail_code  final_t  att_acc  att_conv  att_code     att_err   max|delta|")
    print("-" * 157)
    for row in mode_results:
        final_time_str = str(None if row["final_time"] is None else round(row["final_time"], 6))
        final_delta = row["final_state_max_abs_delta_vs_first"]
        final_delta_str = "None" if final_delta is None else f"{final_delta:.3e}"
        att_err = row["last_attempt_err_norm"]
        att_err_str = "None" if att_err is None else f"{float(att_err):.3e}"
        print(
            f"{row['rhs_mode']:<23}"
            f"{str(row['response_anchor_count']):>9}"
            f"{row['mean_wall_seconds']:>10.3f}"
            f"{row['best_wall_seconds']:>11.3f}"
            f"{str(row['n_steps']):>12}"
            f"{str(row['accepted_steps']):>10}"
            f"{str(row['failed']):>8}"
            f"{str(row['fail_code']):>11}"
            f"{final_time_str:>10}"
            f"{str(row['last_attempt_accepted']):>9}"
            f"{str(row['last_attempt_converged']):>10}"
            f"{str(row['last_attempt_fail_code']):>10}"
            f"{att_err_str:>12}"
            f"{final_delta_str:>14}"
        )


if __name__ == "__main__":
    main()
