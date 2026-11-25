"""Capture the original AD/FD geometry builds at a saved equilibrium.

The opaque VMEX callback returns the recorded state/mask, retaining its normal
placement and caller. Stop before NTX coefficients, roots and transport. Exact
agreement with each archived geometry is required before interpreting the trace.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def saved_equilibrium(capture, state_type):
    """Restore the callback result, rejecting a different physical input."""
    from check_saved_transport_baseline_repeatability import differences
    state = state_type(**{k[6:]: v for k, v in capture.items() if k.startswith('state.')})
    mask = state_type(**{k[5:]: v for k, v in capture.items() if k.startswith('mask.')})
    expected = {k[7:]: v for k, v in capture.items() if k.startswith('params.')}

    def restore(cfg, params, *, refine=True):
        actual = {f.name: np.asarray(getattr(params, f.name)) for f in dataclasses.fields(params)}
        changed = differences(expected, actual)
        if changed:
            raise ValueError(f'Saved equilibrium parameters differ: {changed}')
        restore.calls += 1
        return state, mask
    restore.calls = 0
    return restore


@contextmanager
def capture_geometry_ir(directory, compiler):
    """Save the actual executable; preserve compile options and returned object."""
    directory.mkdir(parents=True, exist_ok=True)
    original = compiler.compile_or_get_cached
    programs = []

    def observe(*args, **kwargs):
        executable = original(*args, **kwargs)
        computation = args[1] if len(args) > 1 else kwargs['computation']
        name = str(computation.operation.attributes['sym_name'])
        if any(word in name.lower() for word in ('booz', 'einsum')):
            prefix = f'program_{len(programs):03d}'
            stable = str(computation)
            (directory/f'{prefix}.mlir').write_text(stable)
            (directory/f'{prefix}.txt').write_text('\n\n'.join(m.to_string() for m in executable.hlo_modules()))
            programs.append(dict(prefix=prefix, name=name, sha256=hashlib.sha256(stable.encode()).hexdigest()))
        return executable

    try:
        with patch.object(compiler, 'compile_or_get_cached', observe):
            yield
    finally:
        (directory/'programs.json').write_text(json.dumps(programs, indent=2))


def aligned_trace(arrays):
    """Match repeated boundaries without relying on route-specific numbering."""
    counts, result, order = {}, {}, []
    for key, value in arrays.items():
        match = re.match(r"trace\['(\d+)_([^']+)'\]", key)
        if not match:
            continue
        number, stage = match.groups()
        if number not in counts:
            counts[number] = sum(s == stage for _, s in order)
            order.append((number, stage))
        result[f'{stage}:{counts[number]}'+key[match.end():]] = value
    return result


def geometry_arrays(arrays):
    return {k[len("support['geometry']."):]: v for k, v in arrays.items()
            if k.startswith("support['geometry'].")}


def worker(args):
    import jax
    import jax.numpy as jnp
    from jax._src import compiler
    if jax.default_backend() != 'gpu' or not jax.config.x64_enabled:
        raise RuntimeError('Run on the large GPU machine with JAX_ENABLE_X64=true.')
    sys.path.insert(0, str(ROOT))
    from NEOPAX import _geometry_autodiff as geo
    from NEOPAX._optimization_diagnostics import save_gradient_audit
    from vmex.core import implicit
    from vmex.core.solver import SpectralState
    from benchmark_transport_ntss1d_ad_fd import commands
    from check_generic_ntx_database_baselines import capture_tree, trace_inputs
    from check_saved_transport_baseline_repeatability import differences, numeric_leaves, read_one
    import tomllib

    run, out = args.run_dir.resolve(), args.out_dir.resolve()
    config = tomllib.loads((run/'config.toml').read_text())
    if config['geometry'].get('vmec_lane') != 'ad' or config['neoclassical'].get('flux_model') != 'ntx_scan_runtime':
        raise ValueError('This capture requires the saved AD-lane runtime-database case.')
    capture = read_one(run/'ad/snapshots', 'ad_compact_geometry_*.npz')
    options = SimpleNamespace(config=run/'config.toml', out_dir=run, accepted_step_limit=16,
                              fd_rel_step=3e-7, fd_abs_step=1e-10)
    command = next(command for name, command, _ in commands(options)
                   if name == ('ad' if args.worker == 'ad' else 'fd_RBC_1_0'))
    def original_entrypoint(_):
        with patch.object(sys, 'argv', command[1:]):
            runpy.run_path(str(ROOT/command[1]), run_name='__main__')
        raise RuntimeError('Original benchmark passed the geometry stopping boundary.')
    nc = config['neoclassical']
    def fields(channels):
        from NEOPAX._transport_flux_models import _build_ntx_field_channels
        values = _build_ntx_field_channels(jnp.asarray(nc['ntx_scan_rho']),
            jnp.asarray(nc['ntx_scan_er_tilde']), channels.as_mapping())
        return dict(zip(('Er', 'Es', 'Er_to_ertilde'), values, strict=True))

    print('[geometry-build] original entrypoint; recorded equilibrium; no NTX/root/transport solve', flush=True)
    restore = saved_equilibrium(capture, SpectralState)
    with patch.object(implicit, '_host_solve_and_mask_impl', restore), \
         patch.object(implicit, '_host_solve', side_effect=AssertionError('Equilibrium solves are disabled in this capture.')), \
         capture_geometry_ir(out/'compiler', compiler):
        trace = trace_inputs(config, geo, geo._import_booz_xform_jax_api(), original_entrypoint,
            fields, projection_module=jnp, trace_vmec_primal=True)
    save_gradient_audit('geometry_build', trace=trace)
    geometry = next(value for name, value in trace.items() if re.fullmatch(r'\d+_geometry', name))
    expected = read_one(run/('ad/snapshots' if args.worker == 'ad' else 'fd_RBC_1_0/snapshots'),
                        'ad_terminal_*.npz' if args.worker == 'ad' else 'fd_reference_support_*.npz')
    actual = {k: v for k, v in numeric_leaves(geometry, "support['geometry']").items()
              if not k.endswith('.__weak_type__')}
    changed = differences(geometry_arrays(expected), geometry_arrays(actual))
    placement = {}
    for stage, value in trace.items():
        if stage.endswith('_geometry_equilibrium'):
            for name, array in capture_tree(value).items():
                placement[name] = {k: str(getattr(array, k, None))
                                   for k in ('committed', 'weak_type', 'sharding', 'format')}
    sources = {str(Path(module.__file__).resolve()): hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
               for module in (geo, implicit, geo._import_booz_xform_jax_api())}
    metadata = dict(archived_geometry_matches=not changed, differences=changed,
        state_placement=placement, restored_callbacks=restore.calls,
        default_device=str(jax.config.jax_default_device), sources=sources,
        versions=dict(jax=jax.__version__), command=command)
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(f'[geometry-build] {args.worker}: archived_geometry_matches={not changed} differences={changed}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True, help='The generic AD16 comparison folder')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--worker', choices=('ad', 'fd'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.run_dir, args.out_dir = args.run_dir.resolve(), args.out_dir.resolve()
    if args.worker:
        return worker(args)
    from benchmark_transport_ntss1d_ad_fd import worker as launch, worker_environment
    from check_saved_transport_baseline_repeatability import differences, read_one
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error('Choose an empty output directory.')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    settings = (args.run_dir/'config.toml').read_text()
    policy = re.search(r'# comparison:.*\bmean=(native|fixed)\b', settings)
    if not policy:
        parser.error('Saved comparison must record its solver mean policy.')
    env = worker_environment()
    env['NEOPAX_SOLVER_MEAN_REDUCTION'] = policy[1]
    for route in ('ad', 'fd'):
        launch([sys.executable, str(Path(__file__).resolve()), '--worker', route,
                '--run-dir', str(args.run_dir.resolve()), '--out-dir', str((args.out_dir/route).resolve())],
               args.out_dir/route, env)
    a, b = [aligned_trace(read_one(args.out_dir/route/'snapshots', 'geometry_build_*.npz'))
            for route in ('ad', 'fd')]
    common = a.keys() & b.keys()
    changed = differences({k: a[k] for k in common}, {k: b[k] for k in common})
    (args.out_dir/'differences.json').write_text(json.dumps(changed, indent=2))
    stages = list(dict.fromkeys(k.split(':', 1)[0] for k in a))
    first = next((stage for stage in stages if any(r['field'].startswith(stage+':') for r in changed)), 'none')
    matched = all(json.loads((args.out_dir/route/'metadata.json').read_text())['archived_geometry_matches']
                  for route in ('ad', 'fd'))
    print(f'[geometry-build] archived_paths_reproduced={matched} first_changed_stage={first}', flush=True)
    if not matched:
        raise SystemExit('Archived geometry did not reproduce; do not attribute its drift to this trace.')


if __name__ == '__main__':
    main()
