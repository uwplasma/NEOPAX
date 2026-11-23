"""Locate NTX baseline drift with one saved surface and one scan point.

The parent reads NumPy snapshots only. Sequential GPU workers compare fresh
preparation and replay one identical prepared system in separate processes.
No equilibrium, geometry reconstruction, root, transport, AD or FD solve runs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def read_one(folder, pattern):
    paths = list(folder.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected one {pattern} in {folder}; found {len(paths)}")
    with np.load(paths[0], allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def scan_inputs(data):
    """Align the terminal dataclass capture and the input-only dict capture."""
    result = {}
    for key, value in data.items():
        match = re.fullmatch(r"trace\['\d+_(channels|surfaces)'\](.*)", key)
        actual = re.fullmatch(r"scan_inputs\['(channels|surfaces)'\](.*)", key)
        if match or actual:
            group, suffix = (match or actual).groups()
            suffix = re.sub(r"\.([A-Za-z_][A-Za-z_0-9]*)", r"['\1']", suffix)
            result[group+suffix] = value
    return result


def changes(a, b):
    return [key for key in sorted(a.keys() | b.keys()) if key not in a or key not in b
            or a[key].shape != b[key].shape or a[key].dtype != b[key].dtype
            or a[key].tobytes() != b[key].tobytes()]


def preflight(args):
    config = tomllib.loads((args.fd_run/'config.toml').read_text())
    fd = read_one(args.fd_run/'fd_ZBS_1_0/snapshots', 'fd_reference_support_*.npz')
    reference = read_one(args.input_reference, 'normalization_inputs_*.npz')
    ad = read_one(args.ad_dir, 'ad_terminal_*.npz')
    expected, actual = scan_inputs(reference), scan_inputs(fd)
    if not actual or changes({key: expected[key] for key in actual if key in expected}, actual):
        raise ValueError('The actual saved NTX input fields differ; inspect them before a GPU solve')
    geometry = {key:value for key,value in ad.items() if key.startswith("support['geometry'].")}
    if not geometry or changes(geometry, {key:fd[key] for key in geometry if key in fd}):
        raise ValueError('Saved AD/FD transport geometries differ')
    metadata = json.loads(args.source_metadata.read_text())
    for name, expected_version in metadata['versions'].items():
        if expected_version is not None and importlib.metadata.version(name) != expected_version:
            raise ValueError(f'Package version changed since the recorded audit: {name}')
    for name, item in metadata['sources'].items():
        if name.startswith(('ntx.', 'solvax.', 'NEOPAX.')):
            path = Path(item['path'])
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError(f'Source changed since the recorded audit: {name}')
    r, n, e = args.point
    shape = fd["support['database'].D13"].shape
    if any(i < 0 or i >= size for i, size in zip(args.point, shape, strict=True)):
        raise ValueError(f'Point {args.point} outside the saved table {shape}')
    return config, fd, reference, ad, actual


def normalized(coefficients, drds, nu):
    return np.asarray((np.log10(max(coefficients[0]*drds**2, 1e-20)),
                       coefficients[2]*drds, coefficients[3]*nu))


def worker(args, data):
    config, fd, reference, ad, inputs = data
    # GPU work exists only behind the explicit worker boundary.
    import jax
    import jax.numpy as jnp
    if jax.default_backend() != 'gpu' or not jax.config.x64_enabled:
        raise RuntimeError('This worker requires a GPU and JAX_ENABLE_X64=true')
    sys.path.insert(0, str(ROOT))
    from ntx.geometry import VmecSurface
    from ntx import GridSpec, prepare_monoenergetic_system
    from ntx._solver_scan import _scan_coefficients_sequential_impl
    from NEOPAX._transport_flux_models import _build_ntx_field_channels

    r, n, e = args.point
    prefix = re.compile(r"trace\['\d+_surfaces'\]\["+str(r)+r"\]\['([^']+)'\]")
    fields = {m[1]: value for key, value in reference.items() if (m := prefix.fullmatch(key))}
    # Match the actual native-surface constructor's Python/static scalar fields.
    scalar_fields = {'requested_psi_n', 'psi_n', 'nfp', 'ns', 'mpol', 'ntor',
                     'total_mode_count', 'loaded_mode_count', 'stellarator_symmetric'}
    values = {key: value.item() if key in scalar_fields else jnp.asarray(value, dtype=value.dtype)
              for key, value in fields.items() if key != 'path'}
    values['path'] = Path(str(fields['path']))
    surface = VmecSurface(**values)
    nc = config['neoclassical']
    grid = GridSpec(n_theta=nc['ntx_scan_n_theta'], n_zeta=nc['ntx_scan_n_zeta'], n_xi=nc['ntx_scan_n_xi'])
    channels = {key[len("channels['"):-2]: jnp.asarray(value, dtype=value.dtype)
                for key, value in inputs.items() if key.startswith("channels['")}
    rho = jnp.asarray(nc['ntx_scan_rho'], dtype=jnp.float64)
    er = jnp.asarray(nc['ntx_scan_er_tilde'], dtype=jnp.float64)
    es = _build_ntx_field_channels(rho, er, channels)[1][r, e]
    nu = jnp.asarray(nc['ntx_scan_nu_v'][n], dtype=jnp.float64)
    prepared = prepare_monoenergetic_system(surface, grid)
    paths, structure = jax.tree_util.tree_flatten_with_path(prepared)
    names = ['prepared'+jax.tree_util.keystr(path) for path, _ in paths]
    leaves = [leaf for _, leaf in paths]
    arrays = {name: np.asarray(jax.device_get(value)) for name, value in zip(names, leaves, strict=True)}
    np.savez_compressed(args.out_dir/'fresh_prepared.npz', **arrays)
    (args.out_dir/'input_types.txt').write_text('\n'.join(
        f"{name}\t{type(value).__name__}\tweak={getattr(value,'weak_type','python')}"
        f"\tdevice={getattr(value,'device','python')}\tcommitted={getattr(value,'committed','python')}"
        for name,value in zip(names,leaves,strict=True))+'\n')
    if args.frozen_prepared is not None:
        frozen = read_one(args.frozen_prepared, 'fresh_prepared.npz')
        if set(frozen) != set(arrays) or any(frozen[key].shape != value.shape or frozen[key].dtype != value.dtype
                                           for key,value in arrays.items()):
            raise ValueError('Prepared tree structure changed')
        def restore(old, value):
            if isinstance(old, (bool, int, float)):
                return value.item()
            return jnp.asarray(value.item()) if getattr(old, 'weak_type', False) else jnp.asarray(value, dtype=value.dtype)
        leaves = [restore(old, frozen[name]) for name, old in zip(names, leaves, strict=True)]
        if changes(frozen, {name:np.asarray(jax.device_get(value)) for name,value in zip(names,leaves,strict=True)}):
            raise ValueError('Prepared replay did not preserve the frozen arrays exactly')
    # Flat arguments avoid the JAX 0.10 ArgInfo/dataclass validation problem in
    # .lower(). The body is the production sequential coefficient-scan body.
    def solve(*flat):
        current = jax.tree_util.tree_unflatten(structure, flat[:-2])
        return _scan_coefficients_sequential_impl(current, flat[-2], flat[-1])
    options = {'xla_gpu_deterministic_ops': True} if args.worker.startswith('controlled') else {}
    print(f'[ntx-point] compiling one point; local compiler options={options}', flush=True)
    compiled = jax.jit(solve, compiler_options=options).lower(*leaves, nu[None], es[None]).compile()
    (args.out_dir/'compiled_hlo.txt').write_text(compiled.as_text())
    result = {'nu': np.asarray(nu), 'Es': np.asarray(es)}
    for repeat in range(2):
        coeff = np.asarray(compiled(*leaves, nu[None], es[None]).block_until_ready())[0]
        result[f'coefficients_{repeat}'] = coeff
        result[f'table_{repeat}'] = normalized(coeff, float(channels['drds'][r]), float(nu))
    result['expected_ad'] = np.asarray([ad["support['database']."+name][r,n,e] for name in ('D11_log','D13','D33')])
    result['expected_fd'] = np.asarray([fd["support['database']."+name][r,n,e] for name in ('D11_log','D13','D33')])
    np.savez_compressed(args.out_dir/'result.npz', **result)
    print(f"[ntx-point] {args.worker} Es={float(es).hex()} raw_D13={result['coefficients_0'][2]:.16e} "
          f"table_D13={result['table_0'][1]:.16e} repeat_matches="
          f"{result['coefficients_0'].tobytes()==result['coefficients_1'].tobytes()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fd-run', type=Path, required=True)
    parser.add_argument('--input-reference', type=Path, required=True)
    parser.add_argument('--ad-dir', type=Path, required=True)
    parser.add_argument('--source-metadata', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--point', default='6:0:7', help='radius:nu:Er table indices')
    parser.add_argument('--deterministic-control', action='store_true',
        help='Also test a kernel-local deterministic compilation; changes no production or environment options')
    parser.add_argument('--worker', choices=('native_1','native_2','controlled_1','controlled_2'), help=argparse.SUPPRESS)
    parser.add_argument('--frozen-prepared', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.point = tuple(int(value) for value in args.point.split(':'))
    if len(args.point) != 3:
        parser.error('--point requires three indices')
    data = preflight(args)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error('Choose an empty output directory')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args, data)
        return
    assert 'jax' not in sys.modules
    common = [sys.executable, str(Path(__file__).resolve()), '--fd-run', str(args.fd_run.resolve()),
        '--input-reference', str(args.input_reference.resolve()), '--ad-dir', str(args.ad_dir.resolve()),
        '--source-metadata', str(args.source_metadata.resolve()), '--point', ':'.join(map(str,args.point))]
    modes = ['native_1','native_2'] + (['controlled_1','controlled_2'] if args.deterministic_control else [])
    results = {}
    env = {key:value for key,value in os.environ.items() if key not in ('PYTHONPATH','JAX_COMPILATION_CACHE_DIR')}
    env.update(JAX_ENABLE_COMPILATION_CACHE='0', NTX_DISABLE_COMPILATION_CACHE='1',
               JAX_ENABLE_X64='true', XLA_PYTHON_CLIENT_PREALLOCATE='false')
    for mode in modes:
        command = [*common, '--out-dir', str((args.out_dir/mode).resolve()), '--worker', mode]
        if mode != 'native_1':
            command += ['--frozen-prepared', str((args.out_dir/'native_1').resolve())]
        print(f'[ntx-point] starting {mode}; one saved surface/point; log={args.out_dir/(mode+".log")}', flush=True)
        with (args.out_dir/(mode+'.log')).open('w') as log:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  cwd=ROOT, env=env, text=True, bufsize=1) as process:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                status = process.wait()
        if status:
            raise RuntimeError(f'{mode} failed; see {args.out_dir/(mode+".log")}')
        results[mode] = read_one(args.out_dir/mode, 'result.npz')
    base = read_one(args.out_dir/'native_1', 'fresh_prepared.npz')
    rows = []
    for mode, result in results.items():
        if changes({key:results['native_1'][key] for key in ('nu','Es')},
                   {key:result[key] for key in ('nu','Es')}):
            raise ValueError('The replay point inputs changed between workers')
        fresh = read_one(args.out_dir/mode, 'fresh_prepared.npz')
        changed = changes(base, fresh)
        print(f'[ntx-point] {mode} fresh_prepared_differences={changed}', flush=True)
        for i, field in enumerate(('D11_log','D13','D33')):
            rows.append(dict(worker=mode, field=field, observed=float(result['table_0'][i]),
                archived_ad=float(result['expected_ad'][i]), archived_fd=float(result['expected_fd'][i]),
                minus_ad=float(result['table_0'][i]-result['expected_ad'][i]),
                minus_fd=float(result['table_0'][i]-result['expected_fd'][i]), prepared_differing_fields=len(changed),
                replay_equals_native_1=result['coefficients_0'].tobytes()==results['native_1']['coefficients_0'].tobytes()))
    with (args.out_dir/'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        if row['field']=='D13':
            print(f"[ntx-point] {row['worker']} D13={row['observed']:.16e} "
                  f"minus_archived_ad={row['minus_ad']:.9e} minus_archived_fd={row['minus_fd']:.9e} "
                  f"replay_equals_native_1={row['replay_equals_native_1']}", flush=True)


if __name__ == '__main__':
    main()
