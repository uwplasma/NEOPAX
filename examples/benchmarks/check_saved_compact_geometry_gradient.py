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
import json
import os
from pathlib import Path
import sys
import tomllib
from unittest.mock import patch

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


def state_tangent_rows(tangents, parameter, current):
    labels = [str(tangents[f'parameter_labels[{i}]'])
              for i in range(sum(k.startswith('parameter_labels[') for k in tangents))]
    column = labels.index(parameter)
    reference = {key[len('state_tangents.'):]: value[column]
                 for key,value in tangents.items() if key.startswith('state_tangents.')}
    if not reference or set(reference) != set(current):
        raise ValueError('Saved/current state tangent fields differ')
    rows = []
    for name,a in reference.items():
        b = np.asarray(current[name])
        if a.shape != b.shape or a.dtype != b.dtype or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f'Invalid state tangent: {name}')
        error = float(np.linalg.norm(a-b))
        norm = float(np.linalg.norm(a))
        rows.append(dict(parameter=parameter,field=name,different_entries=int(np.count_nonzero(a!=b)),
            max_abs_difference=float(np.max(np.abs(a-b),initial=0.)),
            relative_l2_difference=error/norm if norm else (0. if error==0. else float('inf'))))
    return rows,reference


def gradient_direction_rows(captured, parameter, replay_ad, replay_fd, report, labels):
    parameters = [str(captured[f'parameter_labels[{i}]']) for i in range(captured['geometry_gradients'].shape[1])]
    if report['parameter_name'] != parameter or list(labels) != report['objective_labels']:
        raise ValueError('FD report parameter/objective order differs')
    saved = captured['geometry_gradients'][:,parameters.index(parameter)]
    fd = np.asarray(report['gradient_fd'])
    ad,other = np.asarray(replay_ad).reshape(-1),np.asarray(replay_fd).reshape(-1)
    if any(x.shape != saved.shape or not np.isfinite(x).all() for x in (fd,ad,other)) or len(labels) != len(saved):
        raise ValueError('Invalid gradient projection shape/values')
    relative = lambda a,b: abs(a-b)/abs(b) if b else (0. if a==0. else float('inf'))
    return [dict(objective=label,saved_ad=float(saved[i]),replayed_ad_direction=float(ad[i]),
        projected_fd_direction=float(other[i]),original_fd=float(fd[i]),
        tangent_effect=float(other[i]-ad[i]),
        control_relative_difference=relative(ad[i],saved[i]),
        original_relative_difference=relative(saved[i],fd[i]),
        projected_fd_relative_difference=relative(other[i],fd[i])) for i,label in enumerate(labels)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fd-tangent-only", action="store_true",
        help="Compare the original FD VMEX tangent entrypoint with saved AD at fixed state/params/mask; add --fd-report to quantify its cost-gradient effect")
    parser.add_argument("--parameter", default="ZBS:1:0", help="Captured geometry parameter for --fd-tangent-only")
    parser.add_argument("--fd-report",type=Path,
        help="With --fd-tangent-only, also project saved/fresh directions through the recorded 13-row adjoints and compare with this FD summary")
    args = parser.parse_args()
    if args.fd_report is not None and not args.fd_tangent_only:
        parser.error('--fd-report requires --fd-tangent-only')
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
    if args.fd_tangent_only and args.parameter not in labels:
        parser.error('Requested parameter is absent from the compact capture')
    specs = tuple((family, int(m), int(n)) for family, m, n in
                  (label.split(":") for label in ((args.parameter,) if args.fd_tangent_only else labels)))
    if args.fd_tangent_only:
        from benchmark_transport_autodiff_lagged_ntx import _prepare_benchmark_config
        config = _prepare_benchmark_config(args.config,device=None,
            ntx_exact_derivative_mode='direct',radau_jacobian_reuse_mode='legacy')
    else:
        with args.config.open("rb") as stream:
            config = tomllib.load(stream)
    input_path = Path(config["geometry"]["vmec_input_file"])
    config["geometry"]["vmec_input_file"] = str((ROOT / input_path).resolve())
    response_mode = str(captured.get("vmec_response_mode", "raw_block"))
    if args.fd_report is not None and response_mode != 'raw_block':
        parser.error('Weighted FD-direction projection currently requires a raw_block capture')
    config["geometry"]["vmec_response_mode"] = response_mode
    print(f"[compact-replay] JAX={jax.__version__} device={jax.devices()[0]}; "
          "reconstructing context from input, without solving equilibrium", flush=True)
    context = _geometry_context_from_config(config, args.parameter if args.fd_tangent_only else labels[0])
    entries = geo.boundary_param_entries(context, specs)
    implicit, params, cfg = geo._current_implicit_params_cfg_for_param_vector(
        context, jnp.zeros(len(specs)), entries,
        max_iter=config['geometry'].get('vmec_max_iter') if args.fd_tangent_only else None,
        solver_device=str(config['geometry'].get('vmec_implicit_solver_device','default')) if args.fd_tangent_only else "default")
    for module in (geo, implicit, geo._import_booz_xform_jax_api()):
        source = Path(module.__file__)
        print(f"[compact-replay] source={source} "
              f"sha256={hashlib.sha256(source.read_bytes()).hexdigest()}", flush=True)
    if args.fd_tangent_only:
        # Keep the original FD constructor's scalar/weak types. Never silently
        # substitute a different parameter point before comparing tangents.
        for field in dataclasses.fields(params):
            a,b = np.asarray(jax.device_get(getattr(params,field.name))),captured['params.'+field.name]
            if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
                parser.error(f'FD constructor parameters differ from captured AD: {field.name}')
    else:
        params = dataclasses.replace(params, **{
            field.name: jnp.asarray(captured["params." + field.name])
            for field in dataclasses.fields(params)})
    state = SpectralState(**{name: jnp.asarray(value)
                             for name, value in named_arrays(captured, "state.").items()})
    mask = SpectralState(**{name: jnp.asarray(value)
                            for name, value in named_arrays(captured, "mask.").items()})
    if args.fd_tangent_only:
        from benchmark_transport_realtime_geometry_forward_fd import (
            _manual_implicit_forward_state_tangent, _param_unit_tangent_like,
        )
        if _manual_implicit_forward_state_tangent.__globals__['im'] is not implicit:
            parser.error('FD entrypoint and captured-context setup resolved different VMEX implicit modules')
        paths = list(args.capture.parent.glob('ad_compact_tangents_*.npz'))
        if len(paths) != 1:
            parser.error('Expected exactly one saved compact tangent alongside --capture')
        with np.load(paths[0],allow_pickle=False) as data:
            saved_tangents = {key:data[key] for key in data.files}
        print('[fd-tangent] fixed captured state/parameters/mask; original FD tangent entrypoint; no equilibrium/NTX-coefficient/root/transport solve',flush=True)
        current = jax.device_get(jax.block_until_ready(_manual_implicit_forward_state_tangent(
            params=params,param_tangent=_param_unit_tangent_like(params,entries[0]),cfg=cfg,
            x_star=state,dof_mask=mask,vmec_response_mode=response_mode)))
        current = {field.name:np.asarray(getattr(current,field.name)) for field in dataclasses.fields(current)}
        rows,reference = state_tangent_rows(saved_tangents,args.parameter,current)
        args.out_dir.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.out_dir/'state_tangents.npz',parameter=np.asarray(args.parameter),
            **{'fd.'+name:value for name,value in current.items()},
            **{'ad.'+name:value for name,value in reference.items()})
        with (args.out_dir/'state_tangents.csv').open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        for row in rows:
            print(f"[fd-tangent] {row['field']} max_abs={row['max_abs_difference']:.12e} relative_l2={row['relative_l2_difference']:.12e} different_entries={row['different_entries']}",flush=True)
        if args.fd_report is None:
            return
        terminal_paths = list(args.capture.parent.glob('ad_terminal_*.npz'))
        baseline_paths = list((args.fd_report.parent/'snapshots').glob('fd_baseline_*.npz'))
        if len(terminal_paths) != 1 or len(baseline_paths) != 1:
            parser.error('Expected one AD terminal and one FD baseline capture')
        report = json.loads(args.fd_report.read_text())
        with np.load(terminal_paths[0],allow_pickle=False) as terminal, np.load(baseline_paths[0],allow_pickle=False) as baseline:
            objective_labels = tuple(str(terminal[f'labels[{i}]']) for i in range(captured['geometry_gradients'].shape[0]))
            for key in ('initial_y','final_y','final_time','values'):
                a,b = terminal[key],baseline[key]
                if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
                    parser.error(f'AD/FD baseline differs: {key}')
            if list(objective_labels) != report['objective_labels'] or not np.array_equal(terminal['values'],report['objective_values']):
                parser.error('FD report does not match the saved AD point/objectives')
        if report['parameter_name'] != args.parameter:
            parser.error('FD report parameter differs')
        ad_direction = SpectralState(**{name:jnp.asarray(value) for name,value in reference.items()})
        fd_direction = SpectralState(**{name:jnp.asarray(value) for name,value in current.items()})
    raw = geo.GeometryRawBlockSolve(
        implicit, params, cfg, state, mask, entries, vmec_response_mode=response_mode)
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
    def evaluate():
        return np.asarray(jax.device_get(geo.geometry_payload_pullback_from_param_vector_raw_block_transpose(
            context, jnp.zeros(len(specs)), specs, tuple(payload_bars),
            combined_payload=True, payload_kind="ntx_scan_runtime", n_r=n_r,
            n_theta=int(config["neoclassical"]["ntx_scan_n_theta"]),
            n_zeta=int(config["neoclassical"]["ntx_scan_n_zeta"]),
            n_xi=int(config["neoclassical"]["ntx_scan_n_xi"]),
            scan_rho=config["neoclassical"]["ntx_scan_rho"], scan_surface_backend="vmec",
            raw_block_solve=raw, progress_label="[compact-replay]")))
    if args.fd_report is not None:
        # Diagnostic injection at the existing linear-response boundary. The
        # complete production payload contraction and saved bars stay intact.
        with patch.object(implicit,'implicit_state_tangent_raw_block',lambda *a,**kw:ad_direction):
            replay_ad = evaluate()
        with patch.object(implicit,'implicit_state_tangent_raw_block',lambda *a,**kw:fd_direction):
            replay_fd = evaluate()
        rows = gradient_direction_rows(captured,args.parameter,replay_ad,replay_fd,report,objective_labels)
        with (args.out_dir/'gradient_directions.csv').open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        for row in rows:
            print(f"[fd-direction-cost] {row['objective']} saved_AD={row['saved_ad']:.12e} replay_AD={row['replayed_ad_direction']:.12e} projected_FD_direction={row['projected_fd_direction']:.12e} original_FD={row['original_fd']:.12e} control_relative={row['control_relative_difference']:.6e} original_relative={row['original_relative_difference']:.6e} projected_relative={row['projected_fd_relative_difference']:.6e}",flush=True)
        return
    current = evaluate()
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
