"""Locate NTX baseline drift with one saved surface and one scan point.

The parent reads NumPy snapshots only. Sequential GPU workers compare fresh
preparation and replay one identical prepared system in separate processes.
No equilibrium, geometry reconstruction, root, transport, AD or FD solve runs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack, nullcontext
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


def read_live_scan(path, shape, point):
    with np.load(path,allow_pickle=False) as saved:
        arrays = {key:saved[key].copy() for key in saved.files}
    nu, es, coefficients = (arrays[key] for key in ('nu','Es','coefficients'))
    if (nu.shape != (shape[1]*shape[2],) or es.shape != nu.shape
        or coefficients.shape != (nu.size,5)
        or not any(key.startswith('prepared.') for key in arrays)):
        raise ValueError('Live scan shape or prepared fields do not match this benchmark')
    selected = point[1]*shape[2]+point[2]
    if not np.isfinite(nu[selected]) or not np.isfinite(es[selected]):
        raise ValueError('Live scan point is nonfinite')
    return arrays, selected


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
    parameter = getattr(args, 'fd_parameter', 'ZBS:1:0').replace(':', '_')
    fd = read_one(args.fd_run/f'fd_{parameter}/snapshots', 'fd_reference_support_*.npz')
    reference = read_one(args.input_reference, getattr(args, 'input_pattern', 'normalization_inputs_*.npz'))
    ad = read_one(args.ad_dir, 'ad_terminal_*.npz')
    expected, actual = scan_inputs(reference), scan_inputs(fd)
    if not actual or changes({key: expected[key] for key in actual if key in expected}, actual):
        raise ValueError('The actual saved NTX input fields differ; inspect them before a GPU solve')
    geometry = {key:value for key,value in ad.items() if key.startswith("support['geometry'].")}
    if not geometry or changes(geometry, {key:fd[key] for key in geometry if key in fd}):
        raise ValueError('Saved AD/FD transport geometries differ')
    metadata = json.loads(args.source_metadata.read_text())
    versions = dict(metadata['versions'])
    forward_metadata = args.fd_run/'ad/forward/metadata.json'
    if forward_metadata.is_file():
        for name, version in json.loads(forward_metadata.read_text())['versions'].items():
            if name in versions and versions[name] != version:
                raise ValueError(f'Conflicting saved package versions: {name}')
            versions[name] = version
    for name, expected_version in versions.items():
        if expected_version is not None and importlib.metadata.version(name) != expected_version:
            raise ValueError(f'Package version changed since the recorded audit: {name}')
    for name, item in metadata['sources'].items():
        if isinstance(item, str):  # Original-entry geometry captures use path -> hash.
            path, digest = Path(name), item
        else:
            path = Path(item['path'])
            digest = item['sha256']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Source changed since the recorded audit: {name}')
    r, n, e = args.point
    shape = fd["support['database'].D13"].shape
    if any(i < 0 or i >= size for i, size in zip(args.point, shape, strict=True)):
        raise ValueError(f'Point {args.point} outside the saved table {shape}')
    if args.live_scan is not None:
        read_live_scan(args.live_scan,shape,args.point)
    return config, fd, reference, ad, actual


def normalized(coefficients, drds, nu):
    return np.asarray((np.log10(max(coefficients[0]*drds**2, 1e-20)),
                       coefficients[2]*drds, coefficients[3]*nu))


def compare_preparations(results, out_dir, *, fail_on_difference=True):
    base = next(iter(results.values()))
    rows = []
    for mode, arrays in results.items():
        changed = changes(base, arrays)
        print(f'[ntx-point] {mode} prepared_differing_fields={changed}',flush=True)
        for key in changed or ['all']:
            delta = 0. if key=='all' else float('nan')
            if key in base and key in arrays and base[key].shape==arrays[key].shape:
                delta = float(np.max(np.abs(base[key].astype(float)-arrays[key].astype(float)),initial=0.))
            rows.append(dict(worker=mode,field=key,bitwise_equal=not changed,max_abs_difference=delta))
            if changed:
                print(f'[ntx-point] {mode} {key} max_abs_difference={delta:.16e}',flush=True)
    with (out_dir/'preparation_comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if fail_on_difference and any(not row['bitwise_equal'] for row in rows):
        raise ValueError('Fresh NTX preparation still differs across the tested contexts; saved arrays retained')


@contextmanager
def capture_production_scans(directory, *, reference=None, point=(6,0,7), compiler_ir=False):
    """Retain actual serial-scan arguments/results; copy only after the build.

    No preparation is repeated and no argument or result is replaced. The
    generic and recorded builders import the serial kernel into different
    modules, so both aliases must be observed.
    """
    import traceback
    from unittest.mock import patch
    import jax

    records = []
    preparation_count = 0
    def compile_context(name):
        if not compiler_ir:
            return nullcontext()
        from jax._src import compiler
        return capture_ntx_ir(Path(directory)/name, compiler)

    def observe_preparation(original):
        def prepare(*args, **kwargs):
            nonlocal preparation_count
            index = preparation_count
            preparation_count += 1
            with compile_context(f'preparation_{index:04d}_compiler'):
                return original(*args, **kwargs)
        return prepare

    def observe(original, owner):
        def scan(prepared, nu, es):
            record = dict(prepared=prepared, nu=nu, Es=es)
            index = len(records)
            records.append((owner, traceback.format_stack(limit=8), record))
            with compile_context(f'surface_{index:04d}_compiler'):
                result = original(prepared, nu, es)
            record['coefficients'] = result
            return result
        return scan
    try:
        with ExitStack() as patches:
            if compiler_ir:
                for owner in ('ntx', 'ntx._solver_scan', 'ntx._neopax_ad'):
                    module = importlib.import_module(owner)
                    patches.enter_context(patch.object(module, 'prepare_monoenergetic_system',
                        observe_preparation(module.prepare_monoenergetic_system)))
            for owner in ('ntx._solver_scan','ntx._neopax_ad'):
                module = importlib.import_module(owner)
                patches.enter_context(patch.object(module,'_scan_coefficients_serial',
                    observe(module._scan_coefficients_serial,owner)))
            yield
    finally:
        directory = Path(directory)
        directory.mkdir(parents=True,exist_ok=True)
        for index,(owner,stack,record) in enumerate(records):
            arrays, types = {}, []
            for name,tree in record.items():
                leaves, structure = jax.tree_util.tree_flatten_with_path(tree)
                types.append(f'{name}.__treedef__\t{structure}')
                for path,value in leaves:
                    key = name+jax.tree_util.keystr(path)
                    arrays[key] = np.asarray(jax.device_get(value))
                    types.append(f'{key}\t{type(value).__name__}\tweak={getattr(value,"weak_type","python")}'
                        f'\tcommitted={getattr(value,"committed","python")}\tdevice={getattr(value,"device","python")}')
            stem = directory/f'surface_{index:04d}'
            np.savez_compressed(stem.with_suffix('.npz'),**arrays)
            stem.with_suffix('.txt').write_text(owner+'\n'+''.join(stack)+'\n'+'\n'.join(types)+'\n')
            print(f'[ntx-live] surface={index} owner={owner} saved={stem}.npz',flush=True)
            if index != point[0]:
                continue
            if 'coefficients' in arrays:
                selected = point[1]*np.unique(arrays['Es']).size+point[2]
                values = arrays['coefficients'][selected]
                print(f'[ntx-live] point={point} nu={float(arrays["nu"][selected]).hex()} '
                    f'Es={float(arrays["Es"][selected]).hex()} raw_coefficients={values.tolist()}',flush=True)
            if reference is not None:
                with np.load(reference,allow_pickle=False) as saved:
                    expected = {key:saved[key] for key in saved.files if key.startswith('prepared')}
                actual = {key:value for key,value in arrays.items() if key.startswith('prepared')}
                compare_preparations({'reference':expected,'live':actual},directory,
                    fail_on_difference=False)
        print(f'[ntx-live] captured {len(records)} actual scan calls; no scan replay',flush=True)
        sources = {name: dict(path=module.__file__, sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())
            for name, module in tuple(sys.modules.items())
            if name.startswith(('ntx.', 'solvax.', 'NEOPAX.')) and getattr(module, '__file__', None)
            and Path(module.__file__).suffix == '.py'}
        (directory/'loaded_sources.json').write_text(json.dumps(sources, indent=2))


@contextmanager
def capture_ntx_ir(directory, compiler):
    """Observe actual preparation/scan compilations without another JIT or options."""
    from unittest.mock import patch
    directory.mkdir(parents=True, exist_ok=True)
    original = compiler.compile_or_get_cached
    programs = []

    def observe(*args, **kwargs):
        executable = original(*args, **kwargs)
        computation = args[1] if len(args) > 1 else kwargs['computation']
        stable = str(computation)
        prefix = f'program_{len(programs):04d}'
        (directory/f'{prefix}.mlir').write_text(stable)
        (directory/f'{prefix}.txt').write_text('\n\n'.join(m.to_string() for m in executable.hlo_modules()))
        programs.append(dict(prefix=prefix, name=str(computation.operation.attributes['sym_name']),
                             sha256=hashlib.sha256(stable.encode()).hexdigest()))
        return executable

    try:
        with patch.object(compiler, 'compile_or_get_cached', observe):
            yield
    finally:
        (directory/'programs.json').write_text(json.dumps(programs, indent=2))


def worker(args, data):
    if getattr(args, 'capture_compiler_ir', False):
        from jax._src import compiler
        with capture_ntx_ir(args.out_dir/'compiler', compiler):
            _worker(args, data)
    else:
        _worker(args, data)


def _worker(args, data):
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
    from ntx._solver_scan import _scan_coefficients_serial
    from NEOPAX._transport_flux_models import _build_ntx_field_channels
    sources = {name: dict(path=module.__file__, sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())
        for name, module in tuple(sys.modules.items())
        if name.startswith(('ntx.', 'solvax.')) and getattr(module, '__file__', None)
        and Path(module.__file__).suffix == '.py'}
    (args.out_dir/'loaded_sources.json').write_text(json.dumps(sources, indent=2))
    geometry_module = importlib.import_module('ntx._geometry')
    geometry_path = Path(geometry_module.__file__)
    provenance = (f"geometry_source={geometry_path} sha256={hashlib.sha256(geometry_path.read_bytes()).hexdigest()} "
                  f"ordered_sum={hasattr(geometry_module,'_sum_fourier_modes')}")
    (args.out_dir/'geometry_source.txt').write_text(provenance+'\n')
    print('[ntx-point] '+provenance,flush=True)

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
    es_row = _build_ntx_field_channels(rho, er, channels)[1][r]
    nu_axis = jnp.asarray(nc['ntx_scan_nu_v'], dtype=jnp.float64)
    nu, es = nu_axis[n], es_row[e]
    scan_nu, scan_es, selected = nu[None], es[None], 0
    if args.scan_layout == 'production':
        nu_grid, es_grid = jnp.meshgrid(nu_axis, es_row, indexing='ij')
        scan_nu, scan_es = nu_grid.ravel(), es_grid.ravel()
        selected = n*int(es_row.size)+e
    live = None
    if args.live_scan is not None:
        live, selected = read_live_scan(args.live_scan,fd["support['database'].D13"].shape,args.point)
        if float(nu).hex()!=float(live['nu'][selected]).hex() or float(es).hex()!=float(live['Es'][selected]).hex():
            raise ValueError('Selected live nu/Es point differs from the saved reference point')
        scan_nu = jnp.asarray(live['nu'],dtype=live['nu'].dtype)
        scan_es = jnp.asarray(live['Es'],dtype=live['Es'].dtype)
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
    if args.prepare_only:
        print(f'[ntx-point] preparation saved: {len(arrays)} fields; no coefficient solve',flush=True)
        return
    if args.frozen_prepared is not None or live is not None:
        frozen = ({key:value for key,value in live.items() if key.startswith('prepared.')}
                  if live is not None else read_one(args.frozen_prepared, 'fresh_prepared.npz'))
        if set(frozen) != set(arrays) or any(frozen[key].shape != value.shape or frozen[key].dtype != value.dtype
                                           for key,value in arrays.items()):
            raise ValueError('Prepared tree structure changed')
        print(f'[ntx-point] fresh_vs_frozen_preparation_differences={changes(frozen,arrays)}',flush=True)
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
    print(f'[ntx-point] kernel={args.scan_kernel} entries={scan_nu.size} captured_scan={live is not None}; local compiler options={options}', flush=True)
    if args.scan_kernel == 'production':
        # Ordinary Python unpacking, followed by the actual production entry.
        # Do not wrap this call in another jit or lower the prepared dataclass.
        def compiled(*flat):
            current = jax.tree_util.tree_unflatten(structure,flat[:-2])
            return _scan_coefficients_serial(current,flat[-2],flat[-1])
    else:
        compiled = jax.jit(solve, compiler_options=options).lower(*leaves, scan_nu, scan_es).compile()
        (args.out_dir/'compiled_hlo.txt').write_text(compiled.as_text())
    result = {'nu': np.asarray(nu), 'Es': np.asarray(es),
              'scan_nu':np.asarray(scan_nu),'scan_Es':np.asarray(scan_es),'selected':np.asarray(selected)}
    for repeat in range(args.repeats):
        scan_coeff = np.asarray(compiled(*leaves, scan_nu, scan_es).block_until_ready())
        result[f'scan_coefficients_{repeat}'] = scan_coeff
        coeff = scan_coeff[selected]
        result[f'coefficients_{repeat}'] = coeff
        result[f'table_{repeat}'] = normalized(coeff, float(channels['drds'][r]), float(nu))
    result['expected_ad'] = np.asarray([ad["support['database']."+name][r,n,e] for name in ('D11_log','D13','D33')])
    result['expected_fd'] = np.asarray([fd["support['database']."+name][r,n,e] for name in ('D11_log','D13','D33')])
    if live is not None:
        result['expected_live_coefficients'] = live['coefficients']
        print(f"[ntx-point] kernel={args.scan_kernel} raw_D13_minus_live="
            f"{result['coefficients_0'][2]-live['coefficients'][selected,2]:.16e} "
            f"all_coefficients_max_abs_difference={np.max(np.abs(result['scan_coefficients_0']-live['coefficients'])):.16e}",flush=True)
    np.savez_compressed(args.out_dir/'result.npz', **result)
    repeat_matches = all(result['scan_coefficients_0'].tobytes()==result[f'scan_coefficients_{i}'].tobytes()
                         for i in range(1,args.repeats)) if args.repeats>1 else 'not repeated'
    print(f"[ntx-point] {args.worker} Es={float(es).hex()} raw_D13={result['coefficients_0'][2]:.16e} "
          f"table_D13={result['table_0'][1]:.16e} repeat_matches={repeat_matches}", flush=True)
    print(f"[ntx-point] minus_archived_ad={result['table_0'][1]-result['expected_ad'][1]:.9e} "
          f"minus_archived_fd={result['table_0'][1]-result['expected_fd'][1]:.9e}",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fd-run', type=Path, required=True)
    parser.add_argument('--fd-parameter', default='ZBS:1:0',
        help='Select the saved fd_PARAMETER baseline, including a failed coefficient build')
    parser.add_argument('--input-reference', type=Path, required=True)
    parser.add_argument('--input-pattern', default='normalization_inputs_*.npz',
        help='Snapshot pattern in --input-reference; geometry_build_*.npz also contains native scan inputs')
    parser.add_argument('--ad-dir', type=Path, required=True)
    parser.add_argument('--source-metadata', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--point', default='6:0:7', help='radius:nu:Er table indices')
    parser.add_argument('--scan-layout',choices=('point','production'),default='point',
        help='Replay one entry or the original full nu/Er scan on this one prepared surface')
    parser.add_argument('--repeats',type=int,default=2,help='Executions of each compiled scan')
    parser.add_argument('--live-scan',type=Path,
        help='Restore prepared fields and the complete nu/Es row from a captured surface_NNNN.npz')
    parser.add_argument('--scan-kernel',choices=('flat','production'),default='flat',
        help='Compile the diagnostic wrapper or call the unmodified production JIT entry directly')
    parser.add_argument('--prepare-only',action='store_true',
        help='Compare fresh preparation in point/production contexts; stop before coefficient solves')
    parser.add_argument('--capture-compiler-ir', action='store_true',
        help='Save actual preparation and scan executables without changing JIT nesting or compiler options')
    parser.add_argument('--deterministic-control', action='store_true',
        help='Also test a kernel-local deterministic compilation; changes no production or environment options')
    parser.add_argument('--worker', choices=('native_1','native_2','controlled_1','controlled_2',
        'point_1','point_2','production_1','production_2'), help=argparse.SUPPRESS)
    parser.add_argument('--frozen-prepared', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.point = tuple(int(value) for value in args.point.split(':'))
    if len(args.point) != 3:
        parser.error('--point requires three indices')
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    if args.prepare_only and (args.frozen_prepared is not None or args.deterministic_control):
        parser.error('--prepare-only requires fresh inputs without a coefficient compilation control')
    if args.prepare_only and (args.live_scan is not None or args.scan_kernel != 'flat'):
        parser.error('--prepare-only cannot restore a live scan or select a coefficient kernel')
    if args.scan_kernel == 'production' and (args.deterministic_control or (args.worker or '').startswith('controlled')):
        parser.error('The production kernel uses its unchanged compilation options')
    data = preflight(args)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error('Choose an empty output directory')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args, data)
        return
    assert 'jax' not in sys.modules
    common = [sys.executable, str(Path(__file__).resolve()), '--fd-run', str(args.fd_run.resolve()),
        '--fd-parameter', args.fd_parameter, '--input-pattern', args.input_pattern,
        '--input-reference', str(args.input_reference.resolve()), '--ad-dir', str(args.ad_dir.resolve()),
        '--source-metadata', str(args.source_metadata.resolve()), '--point', ':'.join(map(str,args.point)),
        '--repeats',str(args.repeats),'--scan-kernel',args.scan_kernel]
    if args.capture_compiler_ir:
        common += ['--capture-compiler-ir']
    if args.live_scan is not None:
        common += ['--live-scan',str(args.live_scan.resolve())]
    modes = (['point_1','point_2','production_1','production_2'] if args.prepare_only else
        ['native_1','native_2'] + (['controlled_1','controlled_2'] if args.deterministic_control else []))
    results = {}
    env = {key:value for key,value in os.environ.items() if key not in ('PYTHONPATH','JAX_COMPILATION_CACHE_DIR')}
    env.update(JAX_ENABLE_COMPILATION_CACHE='0', NTX_DISABLE_COMPILATION_CACHE='1',
               JAX_ENABLE_X64='true', XLA_PYTHON_CLIENT_PREALLOCATE='false')
    for mode in modes:
        layout=('production' if mode.startswith('production') else 'point') if args.prepare_only else args.scan_layout
        command = [*common, '--out-dir', str((args.out_dir/mode).resolve()), '--worker', mode,'--scan-layout',layout]
        if args.prepare_only:
            command += ['--prepare-only']
        elif mode != 'native_1' and args.live_scan is None:
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
        results[mode] = read_one(args.out_dir/mode, 'fresh_prepared.npz' if args.prepare_only else 'result.npz')
    if args.prepare_only:
        compare_preparations(results,args.out_dir)
        return
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
