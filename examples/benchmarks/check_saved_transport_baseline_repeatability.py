"""Repeat the unperturbed Radau baseline from saved AD/FD physical inputs.

Two sequential GPU processes each run the native adaptive helper once, then
run one jitted callable twice from the same initial carry. No equilibrium,
coefficient scan, root solve, objective evaluation or derivative is computed.
The extra jit is a separate repeatability control, not the production graph.
The parent and saved-array comparison import no JAX.

--trace-ad-fd instead isolates forward setup differences in a failed paired
run. Four independent processes change the model wrapper, AD context metadata,
then checkpoint capture, without sharing a timestep schedule or rerunning AD.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCHEDULE_FIELDS = ("active_mask", "accepted_mask", "err_norms", "attempted_dts",
                   "next_dts", "step_ts", "jacobian_reused")
FORWARD_VARIANTS = ("fd", "fd_fixed_model", "ad_no_checkpoints", "ad")


def read_one(folder, pattern):
    paths = list(Path(folder).glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected one {pattern} in {folder}; found {len(paths)}")
    with np.load(paths[0], allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def differences(reference, current):
    """Report exact bit differences, including signed zero and the first index."""
    rows = []
    for key in sorted(reference.keys() | current.keys()):
        a, b = reference.get(key), current.get(key)
        if a is None or b is None or a.shape != b.shape or a.dtype != b.dtype:
            rows.append(dict(field=key, reason="missing field, shape or dtype mismatch"))
            continue
        if a.tobytes() == b.tobytes():
            continue
        changed = (np.frombuffer(a.tobytes(), dtype=np.uint8)
                   != np.frombuffer(b.tobytes(), dtype=np.uint8))
        changed = changed.reshape(a.size, a.dtype.itemsize).any(axis=1)
        first = int(np.flatnonzero(changed)[0])
        row = dict(field=key, different_entries=int(changed.sum()),
                   first_flat_index=first)
        if key in SCHEDULE_FIELDS:
            row["first_attempt"] = first + 1
        if a.dtype.kind in "bifuc":
            finite = np.isfinite(a) & np.isfinite(b)
            row["finite_masks_match"] = bool(np.array_equal(np.isfinite(a), np.isfinite(b)))
            row["max_abs"] = float(np.max(np.abs(
                a[finite].astype(np.complex128) - b[finite].astype(np.complex128)), initial=0))
        rows.append(row)
    return rows


def numeric_leaves(value, prefix):
    """Host-copy data fields; callable closures are intentionally not traversed."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {key: array for name, child in value.items()
                for key, array in numeric_leaves(child, f"{prefix}.{name}").items()}
    if isinstance(value, (tuple, list)):
        return {key: array for i, child in enumerate(value)
                for key, array in numeric_leaves(child, f"{prefix}[{i}]").items()}
    if isinstance(value, (bool, int, float, complex, str, np.generic)) or hasattr(value, "shape"):
        array = np.asarray(value)
        if array.dtype.kind != "O":
            result = {prefix: array.copy()}
            if hasattr(value, "weak_type"):
                result[prefix + ".__weak_type__"] = np.asarray(value.weak_type)
            return result
    return {}


def ad_reference_initial_er(reference, *, geometry, database, initial_y, er_size):
    """Check the common physical point before taking Er from the AD snapshot."""
    if not {"initial_y", "final_y", "final_time"}.issubset(reference):
        raise ValueError("AD reference must contain initial_y, final_y and final_time.")
    prefixes = ("support['geometry'].", "support['database'].")
    expected = {k: v for k, v in reference.items() if k.startswith(prefixes)}
    current = {k: v for label, obj in (("geometry", geometry), ("database", database))
               for k, v in numeric_leaves(obj, f"support['{label}']").items()
               if not k.endswith(".__weak_type__")}
    rows = differences(expected, current)
    if database is None or not expected or rows:
        for row in rows:
            print("[fd-ad-reference-support]", row, flush=True)
        raise ValueError("AD reference geometry/database arrays differ from the fresh FD runtime.")
    saved, initial_y = np.asarray(reference["initial_y"]), np.asarray(initial_y)
    if (saved.ndim != 1 or saved.shape != initial_y.shape or saved.dtype != initial_y.dtype
            or not 0 < er_size < saved.size or not np.isfinite(saved).all()):
        raise ValueError("AD reference packed initial-state shape, dtype or values are invalid.")
    if differences({"profiles": saved[:-er_size]}, {"profiles": initial_y[:-er_size]}):
        raise ValueError("AD reference initial density/pressure differs from the fresh FD profiles.")
    return saved[-er_size:].copy()


def require_ad_reference_baseline(reference, **current):
    """Reject a different initial/terminal state or time before FD endpoints."""
    keys = ("initial_y", "final_y", "final_time")
    rows = differences({k: np.asarray(reference[k]) for k in keys},
                       {k: np.asarray(current[k]) for k in keys})
    if rows:
        for row in rows:
            print("[fd-ad-reference]", row, flush=True)
        raise ValueError("FD baseline differs from the AD reference; stopped before FD endpoints.")
    print("[fd-ad-reference] initial_y, final_y and final_time match AD exactly", flush=True)


def setup_arrays(components, prepared, execution):
    result = numeric_leaves(prepared.initial_carry, "initial_carry")
    result.update(numeric_leaves(execution, "execution"))
    result.update(numeric_leaves(components["equation_system"].fixed_temperature_profile,
                                "fixed_temperature_profile"))
    return result


def rollout_arrays(rollout):
    count = int(rollout.attempt_count)
    result = {name: np.asarray(getattr(rollout.trace, name))[:count].copy()
              for name in SCHEDULE_FIELDS if getattr(rollout.trace, name) is not None}
    result.update(final_y=np.asarray(rollout.final_carry.y).copy(),
                  final_time=np.asarray(rollout.final_carry.t).copy())
    return result


def capture_fresh_baseline(directory, *, components, prepared, execution, rollout,
                           runtime, state, parameter_name):
    """Copy existing outputs after the ordinary FD baseline has completed."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "setup.npz", **setup_arrays(components, prepared, execution))
    np.savez_compressed(directory / "native.npz", **rollout_arrays(rollout))
    inputs = numeric_leaves(runtime, "runtime")
    inputs.update(numeric_leaves(state, "initial_state"))
    np.savez_compressed(directory / "physical_inputs.npz", **inputs)
    write_json(directory / "metadata.json", dict(parameter_name=parameter_name,
        accepted_count=int(rollout.accepted_count), attempt_count=int(rollout.attempt_count),
        solver_parameters=runtime.solver_parameters,
        versions={name: version(name) for name in ("jax", "jaxlib", "ntx", "vmex", "solvax")},
        sources={name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                 for name, module in tuple(sys.modules.items())
                 if name in {"NEOPAX._orchestrator", "NEOPAX._transport_solvers",
                             "NEOPAX._transport_flux_models", "NEOPAX._transport_equations"}}))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False,
                              default=lambda item: np.asarray(item).tolist()) + "\n")


def raw_initial_state_arrays(reference, initial_y, species_names):
    """Invert the packed layout without applying any solver projections."""
    names, flat = tuple(species_names), np.asarray(initial_y)
    nr = np.asarray(reference["final_state.Er"]).size
    shape = (len(names), nr)
    ions = [i for i, name in enumerate(names) if name != "e"]
    density_size, pressure_size = len(ions) * nr, len(names) * nr
    if (flat.ndim != 1 or flat.size != density_size + pressure_size + nr
            or np.asarray(reference["final_state.pressure"]).shape != shape):
        raise ValueError("Saved packed initial state does not match the species/grid layout.")
    density = np.zeros(shape, dtype=flat.dtype)
    density[ions] = flat[:density_size].reshape(len(ions), nr)
    return dict(density=density,
        pressure=flat[density_size:density_size + pressure_size].reshape(shape).copy(),
        Er=flat[density_size + pressure_size:].copy())


def restore_runtime(config, ad, fd, orch, jnp):
    from NEOPAX._database import Monoenergetic, MonoenergeticRuntimeNTSS1D
    from NEOPAX._geometry_models import VmecBoozer
    from NEOPAX._monoenergetic import monoenergetic_database_kind, normalize_interpolation_mode
    from NEOPAX._state import TransportState
    from NEOPAX._transport_flux_models import NTXRuntimeScanTransportModel

    nc = config["neoclassical"]
    if nc.get("flux_model") != "ntx_scan_runtime":
        raise ValueError("This diagnostic requires the saved ntx_scan_runtime case.")
    mode = normalize_interpolation_mode(nc.get("interpolation_mode"))
    database_type = {"generic": Monoenergetic,
                     "preprocessed_3d_radial_ntss1d": MonoenergeticRuntimeNTSS1D}.get(mode)
    if database_type is None:
        raise ValueError(f"Unsupported saved runtime interpolation: {mode}")

    def fields(prefix):
        return {key[len(prefix):]: jnp.asarray(value)
                for key, value in ad.items() if key.startswith(prefix)}
    geometry = VmecBoozer(**fields("support['geometry']."))
    database = database_type(**fields("support['database']."))
    if monoenergetic_database_kind(database) != mode:
        raise ValueError("Restored database selects a different interpolation rule.")
    print(f"[baseline-repeat] restored database={type(database).__name__} interpolation_mode={mode}", flush=True)
    species, energy = orch._build_species(config), orch._build_energy_grid(config)
    sources = orch.build_source_models_from_config(config, species)
    # Keep the scan-wrapper class and the original interpolation/physics.
    # The tables are already saved: prevent its builder from launching NTX.
    nc.update(prebuild_database=False, preload_channels=False)
    flux = orch._build_flux_model(config, species, energy, geometry, database, source_models=sources)
    neo = flux.neoclassical_model
    if not isinstance(neo, NTXRuntimeScanTransportModel):
        raise TypeError(f"Unexpected neoclassical wrapper: {type(neo)}")
    flux = dataclasses.replace(flux, neoclassical_model=dataclasses.replace(neo, database=database))
    runtime = orch.RuntimeContext(species=species, energy_grid=energy, geometry=geometry,
        database=database, solver_parameters=orch._normalize_solver_config(config),
        models=orch.Models(flux=flux, source=sources))
    if "initial_state.density" in fd:
        state = TransportState(**{name: jnp.asarray(fd[f"initial_state.{name}"])
                                  for name in ("density", "pressure", "Er")})
    else:
        # A failed reference check saves packed initial_y, before fd_baseline
        # is written. The output unpacker applies a density floor, changing
        # the raw zero-He initial condition. Restore raw arrays instead;
        # only the omitted electron density is reconstructed here.
        from NEOPAX._transport_equations import enforce_quasi_neutrality
        arrays = raw_initial_state_arrays(ad, fd["initial_y"], species.names)
        state = TransportState(**{name: jnp.asarray(value) for name, value in arrays.items()})
        if "e" in species.names:
            state = enforce_quasi_neutrality(state, species)
    return runtime, state


def first_attempt_difference(reference, current, fields=SCHEDULE_FIELDS):
    """Locate the first differing scalar attempt, including signed zero."""
    changes = []
    for name in fields:
        a, b = reference[name], current[name]
        if a.ndim != 1 or b.ndim != 1 or a.dtype != b.dtype:
            raise ValueError(f"Incompatible attempt field: {name}")
        count = min(a.size, b.size)
        changed = (np.frombuffer(a[:count].tobytes(), dtype=np.uint8)
                   != np.frombuffer(b[:count].tobytes(), dtype=np.uint8))
        indices = np.flatnonzero(changed.reshape(count, a.dtype.itemsize).any(axis=1))
        if indices.size:
            i = int(indices[0])
            changes.append(dict(attempt=i + 1, field=name, reference=a[i].item(), current=b[i].item()))
        elif a.size != b.size:
            changes.append(dict(attempt=count + 1, field=name, reason="different attempt counts"))
    first = min((row["attempt"] for row in changes), default=None)
    return [row for row in changes if row["attempt"] == first]


@contextlib.contextmanager
def fixed_solver_means():
    """Diagnostic-only fixed-order FP64 means in the solver module."""
    from NEOPAX import _transport_solvers as solvers
    from solvax._reproducible import schur_product
    from unittest.mock import patch
    original = solvers.jnp

    class MeanControl:
        def __init__(self):
            self.sizes = set()

        def __getattr__(self, name):
            return getattr(original, name)

        def mean(self, value, *args, **kwargs):
            if (not args and not kwargs and getattr(value, "ndim", None) == 1 and value.size
                    and value.dtype == np.dtype("float64")):
                self.sizes.add(value.size)
                total = schur_product(original.ones((1, value.size), dtype=value.dtype), value[:, None])[0, 0]
                return total / original.asarray(value.size, dtype=value.dtype)
            return original.mean(value, *args, **kwargs)

    control = MeanControl()
    with patch.object(solvers, "jnp", control):
        yield control


@contextlib.contextmanager
def capture_forward_compiler_ir(directory):
    """Observe original scan compilations; do not re-jit or change options."""
    from jax._src import compiler
    from unittest.mock import patch
    original = compiler.compile_or_get_cached
    programs, largest = [], 0

    def observe(*args, **kwargs):
        nonlocal largest
        executable = original(*args, **kwargs)
        computation = args[1] if len(args) > 1 else kwargs["computation"]
        name = str(computation.operation.attributes["sym_name"])
        if "scan" in name:
            text = str(computation)
            programs.append(dict(name=name, stablehlo_bytes=len(text.encode())))
            if len(text) > largest:
                largest = len(text)
                (directory / "adaptive_stablehlo.mlir").write_text(text)
                hlo = "\n\n".join(module.to_string() for module in executable.hlo_modules())
                (directory / "adaptive_compiled_hlo.txt").write_text(hlo)
        return executable

    try:
        with patch.object(compiler, "compile_or_get_cached", observe):
            yield
    finally:
        write_json(directory / "compiler_programs.json", programs)


class ForwardBaselineCaptured(BaseException):
    """Stop a requested capture even across ordinary failed-trial handlers."""


@contextlib.contextmanager
def capture_original_forward(main, directory):
    """Observe the actual entry point's first adaptive call without rebuilding it."""
    from NEOPAX import _reverse_ad_transport as reverse
    from unittest.mock import patch
    directory.mkdir(parents=True, exist_ok=False)
    original = reverse._radau_adaptive_schedule_rollout

    def observe(execution, carry, **options):
        # Compile/run the original call first, with the same objects and options.
        with capture_forward_compiler_ir(directory):
            rollout = original(execution, carry, **options)
        arrays = rollout_arrays(rollout)
        np.savez_compressed(directory / "native.npz", **arrays)
        np.savez_compressed(directory / "setup.npz",
            **numeric_leaves(carry, "initial_carry"),
            **numeric_leaves(execution, "execution"))
        write_json(directory / "metadata.json", dict(options=options,
            accepted_count=int(rollout.accepted_count), attempt_count=int(rollout.attempt_count),
            failed=bool(rollout.failed), completed=bool(rollout.completed),
            entry_point=main.__code__.co_filename,
            versions={name: version(name) for name in ("jax", "jaxlib", "ntx", "vmex", "solvax")}))
        if bool(rollout.failed) or not bool(rollout.completed) or not np.isfinite(arrays["final_y"]).all():
            raise RuntimeError("Original forward baseline failed; capture retained.")
        print(f"[original-forward] saved {directory}; stopped before backward/FD endpoints", flush=True)
        raise ForwardBaselineCaptured()

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(reverse, "_radau_adaptive_schedule_rollout", observe))
        # FD imported the helper directly; AD's wrapper resolves it in reverse.
        namespace = main.__globals__
        if "_radau_adaptive_schedule_rollout" in namespace:
            if namespace["_radau_adaptive_schedule_rollout"] is not original:
                raise ValueError("Unexpected FD adaptive helper; refusing to change its call.")
            stack.enter_context(patch.dict(namespace, {"_radau_adaptive_schedule_rollout": observe}))
        yield
        raise RuntimeError("Benchmark returned without an original adaptive baseline capture.")


def run_benchmark(main):
    """Opt-in fixed means and observation of the ordinary benchmark entry point."""
    import os
    mode = os.environ.get("NEOPAX_SOLVER_MEAN_REDUCTION", "native")
    directory = os.environ.get("NEOPAX_ORIGINAL_FORWARD_DIR")
    if mode == "native" and not directory:
        return main()
    if mode not in {"native", "fixed"}:
        raise ValueError("NEOPAX_SOLVER_MEAN_REDUCTION must be native or fixed.")
    if mode == "fixed":
        print("[solver-means] fixed FP64 solver reductions enabled", flush=True)
    try:
        with contextlib.ExitStack() as stack:
            control = stack.enter_context(fixed_solver_means()) if mode == "fixed" else None
            if directory:
                stack.enter_context(capture_original_forward(main, Path(directory)))
            result = main()
    except ForwardBaselineCaptured:
        result = None
    if control is None:
        return result
    if not control.sizes:
        raise RuntimeError("Fixed solver means were not traced in the benchmark.")
    record = dict(solver_mean_reduction=mode, fixed_mean_sizes=sorted(control.sizes))
    if directory := os.environ.get("NEOPAX_GRADIENT_AUDIT_DIR"):
        write_json(Path(directory).parent / "solver_mean_reduction.json", record)
    print("[solver-means]", record, flush=True)
    return result


def trace_original_forwards(run, out, limit, mean_reduction):
    """Run only the two real entry points' forward calls, with their live setup."""
    from benchmark_transport_ntss1d_ad_fd import commands, worker, worker_environment
    args = argparse.Namespace(config=run / "config.toml", out_dir=run,
        accepted_step_limit=limit, fd_rel_step=3e-7, fd_abs_step=1e-10)
    tasks = list(commands(args))[:2]
    for label, command, _ in tasks:
        name = "ad" if label == "ad" else "fd"
        folder = out / name
        if folder.exists():
            raise ValueError(f"{folder} already exists; choose a fresh directory.")
        env = worker_environment()
        env.update(NEOPAX_SOLVER_MEAN_REDUCTION=mean_reduction,
                   NEOPAX_ORIGINAL_FORWARD_DIR=str(folder / "forward"))
        worker(command, folder, env)
        metadata = json.loads((folder / "forward/metadata.json").read_text())
        if metadata["accepted_count"] != limit:
            raise ValueError(f"{name} did not complete exactly {limit} accepted steps.")
    a, b = (read_one(out / name / "forward", "native.npz") for name in ("ad", "fd"))
    setup_a, setup_b = (read_one(out / name / "forward", "setup.npz") for name in ("ad", "fd"))
    carry = lambda data: {k: v for k, v in data.items() if k.startswith("initial_carry.")}
    row = dict(comparison="original AD vs original FD", solver_mean_reduction=mean_reduction,
        initial=differences(carry(setup_a), carry(setup_b)), trajectory=differences(a, b),
        first_attempt=first_attempt_difference(a, b),
        first_time_map=first_attempt_difference(a, b, ("next_dts", "attempted_dts", "step_ts")))
    rows = [row]
    print("[original-forward]", row, flush=True)
    for name, folder, pattern in (("ad", "ad/snapshots", "ad_terminal_*.npz"),
                                  ("fd", "fd_n0/snapshots", "fd_reference_baseline_*.npz")):
        saved = read_one(run / folder, pattern)
        current = a if name == "ad" else b
        change = differences({k: saved[k] for k in ("final_y", "final_time")},
                             {k: current[k] for k in ("final_y", "final_time")})
        rows.append(dict(comparison=f"{name} vs archived {name}", differences=change))
        print(f"[original-forward] {name} vs archived {name}: {change}", flush=True)
    write_json(out / "comparison.json", rows)


def trace_forward_worker(run, out, variant, limit, *, capture_ir=False, mean_reduction="native"):
    """Execute an independent ordinary forward baseline from saved inputs."""
    import jax
    import jax.numpy as jnp
    from unittest.mock import patch
    sys.path.insert(0, str(ROOT))
    from NEOPAX import _orchestrator as orch, _reverse_ad_transport as reverse
    from NEOPAX._transport_solvers import _pack_transport_state_arrays
    from NEOPAX._reverse_ad_initial_er import runtime_with_fixed_ntx_database_model
    from benchmark_transport_realtime_geometry_forward_fd import _schedule_rollout
    from benchmark_transport_autodiff_lagged_ntx import _prepare_benchmark_config, _baseline_profile_cfg
    from benchmark_transport_reverse_ad_only import PARAMETER_ORDER, _profile_cfg_scalar_value

    if not jax.config.x64_enabled or jax.default_backend() != "gpu":
        raise RuntimeError("Run on the GPU machine with JAX_ENABLE_X64=true.")
    ad = read_one(run / "ad/snapshots", "ad_terminal_*.npz")
    fd_dir = run / "fd_n0/snapshots"
    fd = read_one(fd_dir, "fd_reference_support_*.npz")
    fd.update(read_one(fd_dir, "fd_reference_baseline_*.npz"))
    physical = lambda data: {k: v for k, v in data.items()
                             if k == "initial_y" or k.startswith(("support['geometry'].", "support['database']."))}
    if differences(physical(ad), physical(fd)):
        raise ValueError("Saved AD/FD physical inputs differ; this control requires an identical physical point.")
    report = json.loads((run / "ad/summary.json").read_text())
    if report["accepted_step_limit"] != limit:
        raise ValueError("Requested step limit differs from the saved AD run.")
    config = _prepare_benchmark_config(run / "config.toml", device=None,
        ntx_exact_derivative_mode="direct", radau_jacobian_reuse_mode="legacy")
    if orch._resolve_er_right_boundary_mode(config, orch._normalize_solver_config(config)) in {
        "ambipolar_edge_root", "floating_ambipolar_edge_node",
    }:
        raise ValueError("This diagnostic must not launch a boundary root solve.")
    with orch._execution_device_context(config), (
        capture_forward_compiler_ir(out) if capture_ir else contextlib.nullcontext()
    ), (fixed_solver_means() if mean_reduction == "fixed" else contextlib.nullcontext()) as means:
        # Use the FD's own saved tables. No geometry, table or root is built.
        runtime, state = restore_runtime(config, {**ad, **fd}, fd, orch, jnp)
        initial, _ = jax.flatten_util.ravel_pytree(_pack_transport_state_arrays(state, runtime.species))
        initial_changes = differences({"initial_y": ad["initial_y"]}, {"initial_y": np.asarray(initial)})
        if initial_changes:
            print("[forward-trace] restored initial-state differences:", initial_changes, flush=True)
            raise ValueError("Restored physical state differs before solver preparation.")
        if variant != "fd":
            runtime = runtime_with_fixed_ntx_database_model(runtime)
        if variant.startswith("ad"):
            captured = {}
            def probe(execution, carry, **options):
                captured["execution"] = execution
                if variant == "ad_no_checkpoints":
                    options["capture_segment_length"] = None
                    # Keep the same scalar Newton-reuse output; change only
                    # the checkpoint-carry capture, not that output contract.
                    result = reverse._radau_adaptive_schedule_rollout(
                        execution, carry, capture_newton_setup=True, **options)
                else:
                    result = reverse._reverse_adaptive_schedule_rollout(execution, carry, **options)
                captured["rollout"] = result
                return result
            profiles = _baseline_profile_cfg(config)
            values = jnp.asarray([_profile_cfg_scalar_value(profiles, name) for name in PARAMETER_ORDER])
            # Only initial-state construction is replaced by the already-saved
            # common state, skipping roots. The production AD preparation,
            # contexts and adaptive solver execute normally; no VJP is run.
            with patch.object(reverse, "initial_state_for_parameter_vector", return_value=state):
                setup = reverse.prepare_reverse_static_setup(values, config=config,
                    runtime=runtime, baseline_state=state, profile_cfg=profiles,
                    initial_er_root_ad="jax_selected_root", accepted_step_limit_override=limit,
                    reverse_segment_length=4, reverse_direct_stage_adjoint=True,
                    reverse_stage_adjoint_solve_mode="block", reverse_rhs_transpose_mode="explicit_database",
                    reverse_segment_start_replay_mode="minimal",
                    reverse_segment_primal_record_mode="reuse_segment_primal_record",
                    reverse_final_objective_cotangent_mode="grouped_vjp",
                    reverse_bootstrap_cotangent_mode="joint_local_vjp_upar_only",
                    reverse_step_bwd_mode="reduced_cotangent_call_boundary",
                    reverse_schedule_artifact_mode="reuse_static_probe", schedule_probe_runner=probe)
            prepared, execution, rollout = setup.prepared_rollout, captured["execution"], captured["rollout"]
            components = {"equation_system": setup.solve_vector_field.__self__}
        else:
            components, prepared, execution, rollout, _ = _schedule_rollout(
                config, runtime, state, accepted_step_limit=limit, capture_newton_setup=True)
        initial = np.asarray(prepared.initial_carry.y)
        if differences({"initial_y": ad["initial_y"]}, {"initial_y": initial}):
            raise ValueError("Restored initial state does not reproduce the saved packed state.")
        arrays = rollout_arrays(rollout)
        if (bool(rollout.failed) or not bool(rollout.completed)
                or int(rollout.accepted_count) != limit or not np.isfinite(arrays["final_y"]).all()):
            raise RuntimeError("Independent baseline did not complete the requested finite prefix.")
        inputs = setup_arrays(components, prepared, execution)
        inputs["initial_rhs"] = np.asarray(prepared.initial_carry.prev_stages).reshape(-1, initial.size)[0].copy()
        np.savez_compressed(out / "setup.npz", **inputs)
        np.savez_compressed(out / "native.npz", **arrays)
        if mean_reduction == "fixed" and not means.sizes:
            raise RuntimeError("Fixed solver means were not traced; the requested control did not execute.")
        write_json(out / "metadata.json", dict(variant=variant, accepted_step_limit=limit,
            compiler_ir_captured=capture_ir,
            solver_mean_reduction=mean_reduction, fixed_mean_sizes=sorted(means.sizes) if means else [],
            versions={name: version(name) for name in ("jax", "jaxlib", "ntx", "vmex", "solvax")},
            sources={name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                     for name, module in tuple(sys.modules.items()) if name.startswith("NEOPAX.")
                     and getattr(module, "__file__", None)}))
        print(f"[forward-trace] {variant}: attempts={int(rollout.attempt_count)} "
              f"final_time={float(arrays['final_time']):.17e}", flush=True)


def compare_forward_variants(run, out, *, mean_reduction="native"):
    rows = []
    for left, right in zip(FORWARD_VARIANTS, FORWARD_VARIANTS[1:]):
        setup_a, setup_b = (read_one(out / name, "setup.npz") for name in (left, right))
        a, b = (read_one(out / name, "native.npz") for name in (left, right))
        carry = lambda data: {k: v for k, v in data.items() if k.startswith("initial_carry.") or k == "initial_rhs"}
        row = dict(comparison=f"{left} vs {right}", solver_mean_reduction=mean_reduction,
                   initial=differences(carry(setup_a), carry(setup_b)),
                   context=differences({k: v for k, v in setup_a.items() if not k.startswith("initial_carry.")},
                                       {k: v for k, v in setup_b.items() if not k.startswith("initial_carry.")}),
                   first_attempt=first_attempt_difference(a, b), trajectory=differences(a, b))
        row["first_time_map"] = first_attempt_difference(a, b, ("next_dts", "attempted_dts", "step_ts"))
        for change in row["first_time_map"]:
            i = change["attempt"] - 1
            change["reference_error"] = str(a["err_norms"][i]) if i < a["err_norms"].size else None
            change["current_error"] = str(b["err_norms"][i]) if i < b["err_norms"].size else None
        rows.append(row)
        print(f"[forward-trace] {row['comparison']}: initial={row['initial']} "
              f"first_attempt={row['first_attempt']} first_time_map={row['first_time_map']}", flush=True)
    for name, folder, pattern in (("ad", "ad/snapshots", "ad_terminal_*.npz"),
                                  ("fd", "fd_n0/snapshots", "fd_reference_baseline_*.npz")):
        saved = read_one(run / folder, pattern)
        current = read_one(out / name, "native.npz")
        changed = differences({k: saved[k] for k in ("final_y", "final_time")},
                              {k: current[k] for k in ("final_y", "final_time")})
        rows.append(dict(comparison=f"{name} vs archived {name}", differences=changed))
        print(f"[forward-trace] {name} vs archived {name}: {changed}", flush=True)
    write_json(out / "comparison.json", rows)
    if mean_reduction == "native":
        print("[forward-trace] Controls locate a cause only if the archived AD and FD trajectories reproduce.", flush=True)
    else:
        print("[forward-trace] Fixed means retain the formula with a fixed reduction kernel; native archive comparisons are references. "
              "Check the independent fixed-control time maps and states before claiming a correction.", flush=True)


def worker(run, out):
    import jax
    import jax.numpy as jnp
    if not jax.config.x64_enabled or jax.default_backend() != "gpu":
        raise RuntimeError("Run on the GPU machine with JAX_ENABLE_X64=true.")
    sys.path.insert(0, str(ROOT))
    from NEOPAX import _orchestrator as orch, _transport_solvers as solvers
    from benchmark_transport_realtime_geometry_forward_fd import _prepare_rollout
    from benchmark_transport_autodiff_lagged_ntx import _prepare_benchmark_config

    ad = read_one(run / "ad_snapshots", "ad_terminal_*.npz")
    fd = read_one(run / "fd_RBC_1_0/snapshots", "fd_baseline_*.npz")
    zbs = read_one(run / "fd_ZBS_1_0/snapshots", "fd_baseline_*.npz")
    physical = lambda data: {k: v for k, v in data.items()
                             if k.startswith(("initial_state.", "support[")) or k == "initial_y"}
    if differences(physical(fd), physical(zbs)):
        raise ValueError("The saved RBC/ZBS physical inputs differ; inspect them before repeating.")
    common = {k: v for k, v in physical(ad).items() if k in physical(fd)}
    if not any(k.startswith("support['database'].") for k in common):
        raise ValueError("Snapshots must contain the saved coefficient tables.")
    if differences(common, {k: fd[k] for k in common}):
        raise ValueError("The saved AD/FD physical inputs differ.")
    report = json.loads((run / "fd_ZBS_1_0/summary.json").read_text())
    config = _prepare_benchmark_config(run / "config.toml", device=None,
        ntx_exact_derivative_mode="direct",
        radau_jacobian_reuse_mode=report["radau_jacobian_reuse_mode"])
    if orch._resolve_er_right_boundary_mode(config, orch._normalize_solver_config(config)) in {
        "ambipolar_edge_root", "floating_ambipolar_edge_node",
    }:
        raise ValueError("This check requires a boundary setup without a new root solve.")
    runtime, state = restore_runtime(config, ad, fd, orch, jnp)
    print("[baseline-repeat] saved state/geometry/tables; preparing ordinary forward solver", flush=True)
    components, prepared, execution = _prepare_rollout(config, runtime, state)
    if differences({"y": ad["initial_y"]}, {"y": np.asarray(prepared.initial_carry.y)}):
        raise ValueError("Restored packed initial state differs from the saved benchmark.")
    np.savez_compressed(out / "setup.npz", **setup_arrays(components, prepared, execution))
    limit = int(report["accepted_step_limit"])
    slots = min(int(components["solver"].max_steps), max(limit * 16, limit + 16))
    if limit <= 0:
        raise ValueError("The saved report must specify a positive accepted-step limit.")

    def solve(carry):
        return solvers._radau_adaptive_schedule_rollout(execution, carry,
            max_total_steps=slots, stop_after_accepted_steps=limit, capture_newton_setup=True)

    metadata = dict(versions={name: version(name) for name in ("jax", "jaxlib", "ntx", "vmex", "solvax")},
        device=str(jax.devices()[0]), accepted_step_limit=limit, attempt_slots=slots,
        solver_parameters=runtime.solver_parameters,
        sources={str(Path(module.__file__).relative_to(ROOT)):
                 hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                 for module in (orch, solvers)}, timings={})
    for name in ("native", "compiled_1", "compiled_2"):
        if name == "compiled_1":
            gc.collect()
            jax.clear_caches()
            compiled = jax.jit(solve)  # Both calls below share this callable and identical arguments.
        start = time.perf_counter()
        result = (solve if name == "native" else compiled)(prepared.initial_carry)
        jax.block_until_ready(result)
        metadata["timings"][name] = time.perf_counter() - start
        count = int(result.attempt_count)
        arrays = rollout_arrays(result)
        if bool(result.failed) or not bool(result.completed) or int(result.accepted_count) != limit:
            raise RuntimeError(f"{name}: incomplete solve, fail_code={int(result.fail_code)}")
        if not np.isfinite(arrays["final_y"]).all():
            raise FloatingPointError(f"{name}: nonfinite endpoint")
        np.savez_compressed(out / f"{name}.npz", **arrays)
        print(f"[baseline-repeat] {name}: attempts={count} accepted={limit} "
              f"final_time={float(arrays['final_time']):.17e} "
              f"elapsed_s={metadata['timings'][name]:.3f}", flush=True)
        del result, arrays
    write_json(out / "metadata.json", metadata)


def compare_saved(run, out):
    rows = []
    def compare(label, a, b):
        changed = differences(a, b)
        rows.append(dict(comparison=label, differing_fields=changed))
        print(f"[baseline-repeat] {label}: differing_fields={len(changed)}", flush=True)
        for row in changed:
            print(" ", json.dumps(row), flush=True)
    load = lambda process, name: read_one(out / process, f"{name}.npz")
    for process in ("process_1", "process_2"):
        native = load(process, "native")
        compare(f"{process}: same compiled callable twice", load(process, "compiled_1"), load(process, "compiled_2"))
        compare(f"{process}: native vs extra jit control", native, load(process, "compiled_1"))
        for parameter in ("RBC_1_0", "ZBS_1_0"):
            old = read_one(run / f"fd_{parameter}/snapshots", "fd_baseline_*.npz")
            report = json.loads((run / f"fd_{parameter}/summary.json").read_text())
            reference = {k: old[k] for k in ("final_y", "final_time")}
            reference.update({k: np.asarray(report["baseline_rollout"][k])
                              for k in ("accepted_mask", "err_norms", "attempted_dts", "next_dts", "step_ts")})
            compare(f"{process}: native vs saved {parameter}", reference, {k: native[k] for k in reference})
    for name in ("setup", "native", "compiled_1"):
        compare(f"independent processes: {name}", load("process_1", name), load("process_2", name))
    write_json(out / "comparison.json", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--trace-ad-fd", action="store_true",
        help="Compare independent AD/FD forward setup paths from a failed paired run, without new physics preparation.")
    parser.add_argument("--trace-original-ad-fd", action="store_true",
        help="Capture the real AD and FD entry points through their forward baselines only, including live preparation.")
    parser.add_argument("--accepted-step-limit", type=int, default=16)
    parser.add_argument("--capture-compiler-ir", action="store_true",
        help="Save original scan compiler IR without changing its inputs, compilation options or JIT nesting.")
    parser.add_argument("--solver-mean-reduction", choices=("native", "fixed"), default="native",
        help="Diagnostic control: fixed-order FP64 one-dimensional means in the solver only; default native.")
    parser.add_argument("--variant", choices=FORWARD_VARIANTS, help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    run, out = args.run_dir.resolve(), args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.trace_original_ad_fd:
        if args.trace_ad_fd or args.accepted_step_limit <= 0:
            parser.error("Choose one trace mode and a positive step limit.")
        trace_original_forwards(run, out, args.accepted_step_limit, args.solver_mean_reduction)
        return
    if args.trace_ad_fd:
        from benchmark_transport_ntss1d_ad_fd import worker_environment
        if args.accepted_step_limit <= 0:
            parser.error("--accepted-step-limit must be positive")
        if args.variant:
            trace_forward_worker(run, out, args.variant, args.accepted_step_limit,
                                 capture_ir=args.capture_compiler_ir, mean_reduction=args.solver_mean_reduction)
            return
        for name in FORWARD_VARIANTS:
            folder = out / name
            if folder.exists():
                parser.error(f"{folder} already exists; choose a fresh output directory.")
            folder.mkdir()
            print(f"[forward-trace] starting {name}; saved physical inputs only; log={folder/'run.log'}", flush=True)
            with (folder / "run.log").open("w") as log:
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                    "--run-dir", str(run), "--out-dir", str(folder), "--trace-ad-fd",
                    "--accepted-step-limit", str(args.accepted_step_limit), "--variant", name,
                    "--solver-mean-reduction", args.solver_mean_reduction,
                    *(["--capture-compiler-ir"] if args.capture_compiler_ir else [])],
                    cwd=ROOT, env=worker_environment(), stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                print("\n".join((folder / "run.log").read_text().splitlines()[-70:]),
                      file=sys.stderr, flush=True)
                parser.error(f"{name} failed; see {folder/'run.log'}")
        compare_forward_variants(run, out, mean_reduction=args.solver_mean_reduction)
        return
    if args.worker:
        worker(run, out)
        return
    for name in ("process_1", "process_2"):
        folder = out / name
        if folder.exists():
            parser.error(f"{folder} already exists; choose a fresh output directory.")
        folder.mkdir()
        print(f"[baseline-repeat] starting {name}; log={folder/'run.log'}", flush=True)
        with (folder / "run.log").open("w") as log:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                "--run-dir", str(run), "--out-dir", str(folder), "--worker"],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            parser.error(f"{name} failed; see {folder/'run.log'}")
    compare_saved(run, out)


if __name__ == "__main__":
    main()
