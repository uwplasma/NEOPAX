"""Compare generic-interpolation AD/FD baseline database builds on the GPU.

Three independent processes build one AD-route and two FD-route baselines.
No ambipolarity solve, transport integration, perturbation or derivative runs.
The parent and saved-array comparison import no JAX. An archived generic AD
snapshot is an additional reference, not a replacement for the fresh AD build.
--trace-inputs instead repeats one configuration, captures upstream boundaries,
and stops before NTX. --reuse-states also skips equilibrium solves.
"""
from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT/"examples/benchmarks/Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box.toml"
PREFIXES = ("support['geometry'].", "support['database'].", "scan_inputs[", "trace[")
TABLES = {f"support['database'].{name}" for name in ("D11_log", "D13", "D33")}


def baseline_config(config, route):
    config = copy.deepcopy(config)
    nc = config["neoclassical"]
    if nc.get("flux_model") != "ntx_scan_runtime" or nc.get("interpolation_mode", "generic") != "generic":
        raise ValueError("This check requires runtime-built tables with GENERIC interpolation.")
    # Only initialization is skipped; table construction precedes this mode check.
    config.setdefault("general", {})["mode"] = "ambipolarity"
    if route == "ad":
        # These are the production full-transport AD benchmark's scan overrides.
        nc.update(ntx_scan_coefficient_reverse_mode="structured", ntx_scan_record_primal=True)
    elif route != "fd":
        raise ValueError(f"Unknown baseline route: {route}")
    return config


def capture_tree(value):
    """Expose dataclass/static metadata without copying or changing device arrays."""
    if dataclasses.is_dataclass(value):
        return {f.name: capture_tree(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, SimpleNamespace):
        return capture_tree(vars(value))
    if isinstance(value, dict):
        return {k: capture_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(capture_tree(v) for v in value)
    return str(value) if isinstance(value, Path) else value


def load_named_state(path, names, prefix="implicit"):
    with np.load(path, allow_pickle=False) as saved:
        keys = names if all(name in saved for name in names) else tuple(f"{prefix}__{name}" for name in names)
        if any(key not in saved for key in keys):
            raise ValueError(f"Saved state lacks required {prefix} fields: {path}")
        result = {name: saved[key].copy() for name, key in zip(names, keys, strict=True)}
    if any(not np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("Saved equilibrium contains nonfinite values.")
    return result


def trace_inputs(config, geometry_module, booz_api, build_runtime, fields, frozen_state=None):
    """Observe the production input-building path, stopping before NTX construction."""
    captured = {}
    def record(name, value):
        captured[f"{len(captured):02d}_{name}"] = capture_tree(value)

    class InputsReady(Exception):
        pass

    solve = geometry_module._solve_state_for_single_param
    build_geometry = geometry_module._build_neopax_geometry_from_state
    build_scan = geometry_module.build_ntx_runtime_scan_inputs_from_vmec_state
    import_vmec = geometry_module._import_vmec_module
    boozer = booz_api.booz_xform_from_inputs
    calls = 0
    observing = False

    def observed(function, *args, **kwargs):
        nonlocal observing
        previous, observing = observing, True
        try:
            return function(*args, **kwargs)
        finally:
            observing = previous

    def equilibrium(*args, **kwargs):
        result = solve(*args, **kwargs) if frozen_state is None else frozen_state
        record("equilibrium", result)
        return result

    def transform(*args, **kwargs):
        nonlocal calls
        if not observing:
            return boozer(*args, **kwargs)
        index, calls = calls, calls+1
        record(f"boozer_{index}_inputs", kwargs)
        result = boozer(*args, **kwargs)
        record(f"boozer_{index}_outputs", result)
        return result

    def imported(name):
        module = import_vmec(name)
        if not observing:
            return module
        # The volume used by a_b is calculated before the Boozer transform.
        # Capture the actual returned volume/vp, without recalculating either.
        replacements = {}
        for function in ("vmec_force_norms_from_bcovar_dynamic", "energies_and_force_norms"):
            if hasattr(module, function):
                def norms(*args, _function=getattr(module, function), **kwargs):
                    result = _function(*args, **kwargs)
                    record("volume_inputs", dict(volume=result.volume, vp=result.vp))
                    return result
                replacements[function] = norms
        return SimpleNamespace(**(vars(module) | replacements)) if replacements else module

    def geometry(*args, **kwargs):
        record("geometry_equilibrium", args[1])
        result = observed(build_geometry, *args, **kwargs)
        record("geometry", result)
        return result

    def scan(*args, **kwargs):
        channels, surfaces = observed(build_scan, *args, **kwargs)
        record("channels", channels)
        record("surfaces", surfaces)
        record("electric_fields", fields(channels))
        raise InputsReady

    with ExitStack() as stack:
        for target, name, function in (
            (geometry_module, "_solve_state_for_single_param", equilibrium),
            (geometry_module, "_build_neopax_geometry_from_state", geometry),
            (geometry_module, "_import_vmec_module", imported),
            (geometry_module, "build_ntx_runtime_scan_inputs_from_vmec_state", scan),
            (booz_api, "booz_xform_from_inputs", transform),
        ):
            stack.enter_context(patch.object(target, name, function))
        try:
            build_runtime(config)
        except InputsReady:
            return captured
    raise RuntimeError("Input trace did not reach the pre-NTX boundary.")


def first_changed_stage(rows):
    names = {row["leaf"].split("'", 2)[1] for row in rows
        if not row["arrays_match"] and row["leaf"].startswith("trace[")}
    return min(names) if names else "none"


def saved_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files if k.startswith(PREFIXES)}


def compare(reference, actual, name):
    rows = []
    for key in sorted(reference.keys() | actual.keys()):
        a, b = reference.get(key), actual.get(key)
        shape_matches = a is not None and b is not None and a.shape == b.shape
        match = shape_matches and a.dtype == b.dtype and np.array_equal(
            a, b, equal_nan=a.dtype.kind in "fc" and b.dtype.kind in "fc")
        difference = 0. if match else float("inf")
        nonfinite_match = False
        index = ""
        values = dict(reference_value="", actual_value="", reference_hex="", actual_hex="")
        if shape_matches and a.dtype.kind in "biufc" and b.dtype.kind in "biufc":
            finite = np.isfinite(a) & np.isfinite(b)
            nonfinite_match = (np.array_equal(np.isfinite(a), np.isfinite(b))
                and np.array_equal(a[~finite], b[~finite], equal_nan=True))
            dtype = np.result_type(a, b, np.float64)
            d = np.zeros(a.shape, dtype=dtype)
            np.subtract(a.astype(dtype), b.astype(dtype), out=d, where=finite)
            if nonfinite_match:
                difference = float(np.max(np.abs(d), initial=0.))
                index = str(np.unravel_index(np.argmax(np.abs(d)), d.shape)) if d.size else ""
                if d.size:
                    i = np.unravel_index(np.argmax(np.abs(d)), d.shape)
                    for label, value in (("reference", a[i].item()), ("actual", b[i].item())):
                        values[f"{label}_value"] = str(value)
                        values[f"{label}_hex"] = value.hex() if isinstance(value, float) else ""
        rows.append(dict(comparison=name, leaf=key, arrays_match=bool(match),
            shape_matches=shape_matches, nonfinite_values_match=bool(nonfinite_match),
            max_abs_difference=difference, max_abs_index=index, **values))
    print(f"[generic-baseline] {name}: differing leaves={sum(not r['arrays_match'] for r in rows)}", flush=True)
    for row in rows:
        if row["leaf"] in TABLES or not row["arrays_match"]:
            print(f"  {row['leaf']}: max_abs={row['max_abs_difference']:.16e} "
                  f"match={row['arrays_match']} index={row['max_abs_index']}", flush=True)
    return rows


def worker(args):
    # Numerical work is restricted to an explicitly launched GPU worker.
    import jax
    if jax.default_backend() != "gpu" or not jax.config.x64_enabled:
        raise RuntimeError("Run on the large GPU machine with JAX_ENABLE_X64=true; no CPU fallback.")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from benchmark_transport_autodiff_lagged_ntx import _prepare_benchmark_config
    from NEOPAX._orchestrator import build_runtime_context
    from NEOPAX._optimization_diagnostics import save_gradient_audit

    config = _prepare_benchmark_config(args.config, device=None,
        ntx_exact_derivative_mode="direct", radau_jacobian_reuse_mode="legacy")
    config = baseline_config(config, args.worker)
    os.environ["NEOPAX_GRADIENT_AUDIT_DIR"] = str(args.out_dir)
    if args.trace_inputs:
        print(f"[normalization-trace] equilibrium={'saved' if args.reuse_states else 'fresh'}; "
            "capture only; NO NTX coefficient, root or transport solve", flush=True)
    else:
        print(f"[generic-baseline] building {args.worker} baseline; generic interpolation; NO root or transport solve", flush=True)
    with jax.default_device(jax.local_devices(backend="gpu")[0]):
        if args.trace_inputs:
            import jax.numpy as jnp
            from NEOPAX import _geometry_autodiff as geometry_module
            from NEOPAX._transport_flux_models import _build_ntx_field_channels
            frozen = None
            if args.reuse_states:
                from vmex.core.solver import SpectralState
                names = tuple(f.name for f in dataclasses.fields(SpectralState))
                frozen = SpectralState(**jax.tree.map(jnp.asarray,
                    load_named_state(args.reuse_states, names, args.state_prefix)))
            def fields(channels):
                values = _build_ntx_field_channels(jnp.asarray(config['neoclassical']['ntx_scan_rho']),
                    jnp.asarray(config['neoclassical']['ntx_scan_er_tilde']), channels.as_mapping())
                return dict(zip(("Er", "Es", "Er_to_ertilde"), values, strict=True))
            captured = trace_inputs(config, geometry_module, geometry_module._import_booz_xform_jax_api(),
                build_runtime_context, fields, frozen)
            snapshot = save_gradient_audit("normalization_inputs", trace=captured)
            state = next(value for name, value in captured.items() if name.endswith("_geometry_equilibrium"))
            np.savez_compressed(args.out_dir/"state_reference.npz", **jax.device_get(state))
            print(f"[normalization-trace] captured stages={list(captured)}; NO NTX solve", flush=True)
        else:
            runtime, _ = build_runtime_context(config)
            model = getattr(runtime.models.flux, "neoclassical_model", runtime.models.flux)
            if model.database is None or model.interpolation_mode != "generic":
                raise RuntimeError("Expected the materialized generic NTX runtime database.")
            snapshot = save_gradient_audit("generic_database_baseline",
                support={"geometry": runtime.geometry, "database": model.database},
                scan_inputs={"channels": model.channels, "surfaces": model.scan_surfaces})
    nc = config['neoclassical']
    metadata = dict(route=args.worker, interpolation_mode=nc.get('interpolation_mode', 'generic'),
        coefficient_reverse_mode=nc.get('ntx_scan_coefficient_reverse_mode', 'generic'),
        record_scan_primal=nc.get('ntx_scan_record_primal', False),
        trace_inputs=args.trace_inputs, reused_state=str(args.reuse_states) if args.reuse_states else None,
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(), snapshot=Path(snapshot).name,
        vmec_lane=config["geometry"].get("vmec_lane", "forward"),
        vmec_response_mode=config["geometry"].get("vmec_response_mode", "raw_block"), versions={}, sources={})
    if args.trace_inputs:
        metadata['input_sha256'] = hashlib.sha256(Path(config['geometry']['vmec_input_file']).read_bytes()).hexdigest()
        metadata['reused_state_sha256'] = hashlib.sha256(args.reuse_states.read_bytes()).hexdigest() if args.reuse_states else None
    for package in ("jax", "jaxlib", "ntx", "vmex", "solvax", "booz-xform-jax"):
        try:
            metadata["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            metadata["versions"][package] = None
    modules = ("NEOPAX._geometry_autodiff", "NEOPAX._transport_flux_models", "ntx._neopax_scan",
                 "ntx._solver", "ntx._solver_scan", "ntx._neopax_ad", "solvax.direct",
                 "booz_xform_jax.jax_api")
    if args.trace_inputs:
        modules += tuple(sorted(name for name in sys.modules if name.startswith("vmex.")))
    for name in modules:
        module = sys.modules.get(name)
        path = Path(module.__file__) if module is not None else None
        if path is not None and path.is_file():
            metadata["sources"][name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (args.out_dir/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--reference", type=Path, help="Saved GENERIC AD ad_terminal_*.npz snapshot.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--worker", choices=("ad", "fd"), help=argparse.SUPPRESS)
    parser.add_argument("--trace-inputs", action="store_true", help="Capture input boundaries and stop before NTX coefficient solves.")
    parser.add_argument("--reuse-states", type=Path, help="Input trace only: reuse saved states.npz or state_reference.npz; skip equilibrium solves.")
    parser.add_argument("--state-prefix", default="implicit", help="Field prefix for a reused states.npz (default: implicit).")
    args = parser.parse_args()
    args.config, args.out_dir = args.config.resolve(), args.out_dir.resolve()
    if args.reuse_states:
        args.reuse_states = args.reuse_states.resolve()
        if not args.trace_inputs or not args.reuse_states.is_file():
            parser.error("--reuse-states requires --trace-inputs and an existing state file.")
    if args.trace_inputs and args.reference:
        parser.error("--reference is for database comparisons; input traces compare their own three repeats.")
    if args.worker:
        worker(args)
        return
    if args.reference is not None and not args.reference.is_file():
        parser.error(f"Reference snapshot missing: {args.reference}")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("Choose an empty output directory to keep independent baseline builds separate.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir/"config.toml").write_bytes(args.config.read_bytes())
    env = {k: v for k, v in os.environ.items() if k not in {
        "PYTHONPATH", "LD_LIBRARY_PATH", "JAX_PLATFORM_NAME", "JAX_COMPILATION_CACHE_DIR",
        "XLA_FLAGS", "XLA_PYTHON_CLIENT_ALLOCATOR", "NEOPAX_GRADIENT_AUDIT_DIR"}}
    env.update(JAX_PLATFORMS="cuda,cpu", JAX_ENABLE_X64="true", XLA_PYTHON_CLIENT_PREALLOCATE="false",
        JAX_ENABLE_COMPILATION_CACHE="0", NTX_DISABLE_COMPILATION_CACHE="1", VMEX_COMPILATION_CACHE="disabled")
    arrays, rows = {}, []
    reference_controls = None
    runs = (("baseline_1", "fd"), ("baseline_2", "fd"), ("baseline_3", "fd")) if args.trace_inputs else (("ad", "ad"), ("fd_1", "fd"), ("fd_2", "fd"))
    reference_name = runs[0][0]
    for name, route in runs:
        folder = args.out_dir/name
        folder.mkdir()
        command = [sys.executable, str(Path(__file__).resolve()), "--config", str(args.config),
                   "--out-dir", str(folder), "--worker", route]
        if args.trace_inputs:
            command += ["--trace-inputs"]
            if args.reuse_states:
                command += ["--reuse-states", str(args.reuse_states), "--state-prefix", args.state_prefix]
        print(f"[generic-baseline] starting independent {name} process; log={folder/'run.log'}", flush=True)
        with (folder/"run.log").open("w") as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{name} baseline failed; see {folder/'run.log'}")
        if args.trace_inputs:
            metadata = json.loads((folder/'metadata.json').read_text())
            controls = {k: metadata[k] for k in ('config_sha256', 'input_sha256', 'reused_state_sha256', 'versions', 'sources')}
            if reference_controls is not None and controls != reference_controls:
                raise ValueError("Input files, package versions or source hashes differ between repeats.")
            reference_controls = controls
        snapshot = next(folder.glob("normalization_inputs_*.npz" if args.trace_inputs else "generic_database_baseline_*.npz"))
        arrays[name] = saved_arrays(snapshot)
        if not args.trace_inputs and not TABLES.issubset(arrays[name]):
            raise RuntimeError(f"Missing baseline coefficient tables: {snapshot}")
        if name != reference_name:
            comparison = f"{reference_name}_vs_{name}" if args.trace_inputs else f"fresh_ad_vs_{name}"
            compared = compare(arrays[reference_name], arrays[name], comparison)
            rows.extend(compared)
            if args.trace_inputs:
                print(f"[normalization-trace] {name}: first_changed_stage={first_changed_stage(compared)}", flush=True)
    rows.extend(compare(arrays[runs[1][0]], arrays[runs[2][0]], f"{runs[1][0]}_vs_{runs[2][0]}"))
    if args.reference is not None:
        reference = saved_arrays(args.reference)
        if not TABLES.issubset(reference):
            raise RuntimeError("Reference does not contain the saved generic AD coefficient tables.")
        rows.extend(compare(reference, arrays["ad"], "archived_ad_vs_fresh_ad"))
    with (args.out_dir/"comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: f"{v:.16e}" if isinstance(v, float) else v for k, v in row.items()} for row in rows)
    label = "normalization-trace" if args.trace_inputs else "generic-baseline"
    print(f"[{label}] saved {args.out_dir/'comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
