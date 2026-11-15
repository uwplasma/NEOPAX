"""Replay only the final geometry contraction from an AD16 compact capture.

No equilibrium solve, NTX coefficient solve, ambipolar root or transport run.
Uses the production raw-block geometry entrypoint and the captured parameters,
state, mask and all folded objective cotangents. Run on the large GPU machine.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import os
from pathlib import Path
import sys
import tomllib

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "examples/benchmarks/Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box.toml"


def named_arrays(arrays, prefix):
    """Read one flat dataclass without accidentally matching a sibling prefix."""
    return {key[len(prefix):]: value for key, value in arrays.items()
            if key.startswith(prefix) and "." not in key[len(prefix):]}


def contraction_rows(captured, tangents, current):
    """Project saved per-leaf tangents on the original bars, using NumPy only."""
    labels = [str(captured[f"parameter_labels[{i}]"]) for i in range(current.shape[1])]
    paths = [str(tangents[f"payload_paths[{i}]"])
             for i in range(sum(key.startswith("payload_paths[") for key in tangents))]
    rows = []
    reconstructed = np.zeros_like(current)
    for leaf, path in enumerate(paths):
        dots = tangents[f"payload_tangents[{leaf}]"]
        for objective in range(current.shape[0]):
            bar = captured[f"payload_bars[{objective}]" + path]
            if dots.shape != (current.shape[1], *bar.shape):
                raise ValueError(f"Tangent/bar shape mismatch: {path}")
            for column, label in enumerate(labels):
                value = float(np.sum(bar * dots[column]))
                reconstructed[objective, column] += value
                rows.append(dict(objective_index=objective, parameter=label, path=path,
                                 contribution=value))
    return rows, reconstructed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.capture, allow_pickle=False) as data:
        captured = {key: data[key].copy() for key in data.files}
    if any(key.startswith("native_vmec_face_coefficient_bars") for key in captured):
        parser.error("This checker expects the scan-only AD16 capture, without native face bars.")
    os.environ["NEOPAX_GRADIENT_AUDIT_DIR"] = str(args.out_dir.resolve())

    # Imports below this boundary may initialize the GPU. Help and saved-array
    # analysis above it do not import JAX or any physics package.
    import jax
    import jax.numpy as jnp
    if jax.default_backend() != "gpu":
        parser.error("Run this contraction check on the large GPU machine; no CPU fallback.")
    if not jax.config.x64_enabled:
        parser.error("JAX_ENABLE_X64=true is required to read this float64 capture.")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from benchmark_transport_reverse_ad_only import _geometry_context_from_config
    from NEOPAX import _geometry_autodiff as geo
    from NEOPAX._geometry_models import VmecBoozer
    from NEOPAX._transport_flux_models import NTXRuntimeScanChannels
    from ntx.geometry import VmecSurface
    from vmex.core.solver import SpectralState

    labels = tuple(str(captured[f"parameter_labels[{i}]"])
                   for i in range(captured["geometry_gradients"].shape[1]))
    specs = tuple((family, int(m), int(n)) for family, m, n in
                  (label.split(":") for label in labels))
    with args.config.open("rb") as stream:
        config = tomllib.load(stream)
    input_path = Path(config["geometry"]["vmec_input_file"])
    config["geometry"]["vmec_input_file"] = str((ROOT / input_path).resolve())
    print(f"[compact-replay] JAX={jax.__version__} device={jax.devices()[0]}; "
          "reconstructing context from input, without solving equilibrium", flush=True)
    context = _geometry_context_from_config(config, labels[0])
    entries = geo.boundary_param_entries(context, specs)
    implicit, params, cfg = geo._current_implicit_params_cfg_for_param_vector(
        context, jnp.zeros(len(specs)), entries, solver_device="default")
    for module in (geo, implicit, geo._import_booz_xform_jax_api()):
        source = Path(module.__file__)
        print(f"[compact-replay] source={source} "
              f"sha256={hashlib.sha256(source.read_bytes()).hexdigest()}", flush=True)
    params = dataclasses.replace(params, **{
        field.name: jnp.asarray(captured["params." + field.name])
        for field in dataclasses.fields(params)})
    state = SpectralState(**{name: jnp.asarray(value)
                             for name, value in named_arrays(captured, "state.").items()})
    mask = SpectralState(**{name: jnp.asarray(value)
                            for name, value in named_arrays(captured, "mask.").items()})
    raw = geo.GeometryRawBlockSolve(implicit, params, cfg, state, mask, entries)
    n_r = captured["payload_bars[0]['geometry'].r_grid"].size
    surface_count = len(config["neoclassical"]["ntx_scan_rho"])
    payload_bars = []
    for row in range(captured["geometry_gradients"].shape[0]):
        prefix = f"payload_bars[{row}]"
        fields = named_arrays(captured, prefix + "['geometry'].")
        fields.update(n_r=np.zeros((), dtype=jax.dtypes.float0),
                      full_grid_indices=np.zeros(n_r, dtype=jax.dtypes.float0))
        geometry = VmecBoozer(**fields)
        channels = NTXRuntimeScanChannels(**named_arrays(captured, prefix + "['channels']."))
        surfaces = []
        for radius in range(surface_count):
            fields = named_arrays(captured, prefix + f"['surfaces'][{radius}].")
            count = fields["b_cos"].size
            fields.update(path=context.input_path, ns=state.R_cos.shape[0],
                          mpol=int(context.static.cfg.mpol), ntor=int(context.static.cfg.ntor),
                          total_mode_count=count, loaded_mode_count=count,
                          nfp=np.zeros((), dtype=jax.dtypes.float0),
                          m=np.zeros(count, dtype=jax.dtypes.float0),
                          n=np.zeros(count, dtype=jax.dtypes.float0))
            surfaces.append(VmecSurface(**fields))
        payload_bars.append(dict(geometry=geometry, channels=channels, surfaces=tuple(surfaces)))
    print("[compact-replay] captured state/parameters/mask and all objective rows; "
          "no equilibrium, NTX, root or transport solve", flush=True)
    current = np.asarray(jax.device_get(
        geo.geometry_payload_pullback_from_param_vector_raw_block_transpose(
            context, jnp.zeros(len(specs)), specs, tuple(payload_bars),
            combined_payload=True, payload_kind="ntx_scan_runtime", n_r=n_r,
            scan_rho=config["neoclassical"]["ntx_scan_rho"], scan_surface_backend="vmec",
            raw_block_solve=raw, progress_label="[compact-replay]")))
    tangent_file = max(args.out_dir.glob("ad_compact_tangents_*.npz"), key=lambda p: p.stat().st_mtime_ns)
    with np.load(tangent_file, allow_pickle=False) as data:
        tangents = {key: data[key].copy() for key in data.files}
    rows, reconstructed = contraction_rows(captured, tangents, current)
    with (args.out_dir / "leaf_contributions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(args.out_dir / "comparison.npz", captured=captured["geometry_gradients"],
                        current=current, reconstructed=reconstructed)
    print("[compact-replay] max saved-leaf/production contraction difference="
          f"{np.max(np.abs(reconstructed-current)):.16e}", flush=True)
    for row in range(current.shape[0]):
        for column, label in enumerate(labels):
            print(f"[compact-replay] row={row} parameter={label} "
                  f"captured={captured['geometry_gradients'][row, column]:.16e} "
                  f"current={current[row, column]:.16e} "
                  f"saved_leaf_projection={reconstructed[row, column]:.16e}", flush=True)
    print(f"[compact-replay] saved tangents and leaf contributions in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
