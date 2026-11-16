"""Compare generic-interpolation AD/FD baseline database builds on the GPU.

Three independent processes build one AD-route and two FD-route baselines.
No ambipolarity solve, transport integration, perturbation or derivative runs.
The parent and saved-array comparison import no JAX. An archived generic AD
snapshot is an additional reference, not a replacement for the fresh AD build.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT/"examples/benchmarks/Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box.toml"
PREFIXES = ("support['geometry'].", "support['database'].", "scan_inputs[")
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


def saved_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files if k.startswith(PREFIXES)}


def compare(reference, actual, name):
    rows = []
    for key in sorted(reference):
        a, b = reference[key], actual.get(key)
        shape_matches = b is not None and a.shape == b.shape
        match = shape_matches and a.dtype == b.dtype and np.array_equal(
            a, b, equal_nan=a.dtype.kind in "fc" and b.dtype.kind in "fc")
        difference = 0. if match else float("inf")
        nonfinite_match = False
        index = ""
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
        rows.append(dict(comparison=name, leaf=key, arrays_match=bool(match),
            shape_matches=shape_matches, nonfinite_values_match=bool(nonfinite_match),
            max_abs_difference=difference, max_abs_index=index))
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
    print(f"[generic-baseline] building {args.worker} baseline; generic interpolation; NO root or transport solve", flush=True)
    with jax.default_device(jax.local_devices(backend="gpu")[0]):
        runtime, _ = build_runtime_context(config)
        model = getattr(runtime.models.flux, "neoclassical_model", runtime.models.flux)
        if model.database is None or model.interpolation_mode != "generic":
            raise RuntimeError("Expected the materialized generic NTX runtime database.")
        snapshot = save_gradient_audit("generic_database_baseline",
            support={"geometry": runtime.geometry, "database": model.database},
            scan_inputs={"channels": model.channels, "surfaces": model.scan_surfaces})
    metadata = dict(route=args.worker, interpolation_mode=model.interpolation_mode,
        coefficient_reverse_mode=model.coefficient_reverse_mode, record_scan_primal=model.record_scan_primal,
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(), snapshot=Path(snapshot).name,
        vmec_lane=config["geometry"].get("vmec_lane", "forward"),
        vmec_response_mode=config["geometry"].get("vmec_response_mode", "raw_block"), versions={}, sources={})
    for package in ("jax", "jaxlib", "ntx", "vmex", "solvax"):
        try:
            metadata["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            metadata["versions"][package] = None
    for name in ("NEOPAX._geometry_autodiff", "NEOPAX._transport_flux_models", "ntx._neopax_scan",
                 "ntx._solver", "ntx._solver_scan", "ntx._neopax_ad", "solvax.direct"):
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
    args = parser.parse_args()
    args.config, args.out_dir = args.config.resolve(), args.out_dir.resolve()
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
    for name, route in (("ad", "ad"), ("fd_1", "fd"), ("fd_2", "fd")):
        folder = args.out_dir/name
        folder.mkdir()
        command = [sys.executable, str(Path(__file__).resolve()), "--config", str(args.config),
                   "--out-dir", str(folder), "--worker", route]
        print(f"[generic-baseline] starting independent {name} process; log={folder/'run.log'}", flush=True)
        with (folder/"run.log").open("w") as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{name} baseline failed; see {folder/'run.log'}")
        snapshot = next(folder.glob("generic_database_baseline_*.npz"))
        arrays[name] = saved_arrays(snapshot)
        if not TABLES.issubset(arrays[name]):
            raise RuntimeError(f"Missing baseline coefficient tables: {snapshot}")
        if name != "ad":
            rows.extend(compare(arrays["ad"], arrays[name], f"fresh_ad_vs_{name}"))
    rows.extend(compare(arrays["fd_1"], arrays["fd_2"], "fd_1_vs_fd_2"))
    if args.reference is not None:
        reference = saved_arrays(args.reference)
        if not TABLES.issubset(reference):
            raise RuntimeError("Reference does not contain the saved generic AD coefficient tables.")
        rows.extend(compare(reference, arrays["ad"], "archived_ad_vs_fresh_ad"))
    with (args.out_dir/"comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: f"{v:.16e}" if isinstance(v, float) else v for k, v in row.items()} for row in rows)
    print(f"[generic-baseline] saved {args.out_dir/'comparison.csv'}; all comparisons use GENERIC interpolation", flush=True)


if __name__ == "__main__":
    main()
