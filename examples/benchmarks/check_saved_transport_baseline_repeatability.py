"""Repeat the unperturbed Radau baseline from saved AD/FD physical inputs.

Two sequential GPU processes each run the native adaptive helper once, then
run one jitted callable twice from the same initial carry. No equilibrium,
coefficient scan, root solve, objective evaluation or derivative is computed.
The extra jit is a separate repeatability control, not the production graph.
The parent and saved-array comparison import no JAX.
"""
from __future__ import annotations

import argparse
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
    state = TransportState(**{name: jnp.asarray(fd[f"initial_state.{name}"])
                              for name in ("density", "pressure", "Er")})
    return runtime, state


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
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    run, out = args.run_dir.resolve(), args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
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
