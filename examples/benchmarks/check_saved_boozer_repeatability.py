"""Repeat the original Boozer API on one exact saved argument set.

The NumPy-only parent launches two fresh sequential CUDA workers. Each calls
the unchanged API repeatedly, retaining its compiler program and comparing
every output byte. No equilibrium, NTX, root, transport, AD or FD solve runs.
"""
import argparse
import dataclasses
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from check_saved_transport_baseline_repeatability import differences


def fields(data, prefix):
    return {key[len(prefix):]: value for key, value in data.items()
            if key.startswith(prefix) and '.' not in key[len(prefix):]}


def read_capture(path):
    with np.load(path, allow_pickle=False) as saved:
        data = {key:saved[key].copy() for key in saved.files}
    groups = {name:fields(data, 'capture.'+name+'.')
              for name in ('inputs','constants','grids','outputs')}
    if not all(groups.values()) or not bool(data.get('capture.jit', False)):
        raise ValueError('Require a completed original jitted Boozer call')
    for key,value in data.items():
        if value.dtype.kind not in 'biufc' or not np.isfinite(value).all():
            raise ValueError(f'Invalid captured numerical leaf: {key}')
    return data, groups


def restore_arguments(data, groups, api, asarray):
    """Restore only recorded values; preserve the production API and its jit."""
    def array(value, path):
        return asarray(value, weak_type=bool(data.get(path+'.__weak_type__', False)))
    constants = {field.name:groups['constants'][field.name].item()
                 for field in dataclasses.fields(api.BoozXformConstants)}
    grids = {field.name:array(groups['grids'][field.name], 'capture.grids.'+field.name)
             for field in dataclasses.fields(api.BoozXformGrids)}
    return dict(inputs=SimpleNamespace(**{key:array(value,'capture.inputs.'+key) for key,value in groups['inputs'].items()}),
        constants=api.BoozXformConstants(**constants), grids=api.BoozXformGrids(**grids),
        surface_indices=array(data['capture.surface_indices'],'capture.surface_indices') if 'capture.surface_indices' in data else None,
        jit=True)


def worker(args, data, groups):
    import jax
    import jax.numpy as jnp
    from jax._src import compiler
    from jax._src.lax.lax import _convert_element_type
    from booz_xform_jax import jax_api as api
    from check_saved_ntx_point_repeatability import capture_ntx_ir
    if jax.default_backend() != 'gpu' or not jax.config.x64_enabled:
        raise RuntimeError('Require CUDA and JAX_ENABLE_X64=true')
    recorded = json.loads((args.capture.parent/'loaded_sources.json').read_text())
    sources = {}
    for name,item in recorded.items():
        if not name.startswith('booz_xform_jax.'):
            continue
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item['sha256']:
            raise ValueError(f'Boozer source changed: {name}')
        sources[name] = dict(path=str(path),sha256=digest)
    if 'booz_xform_jax.jax_api' not in sources:
        raise ValueError('Capture is missing its Boozer API source hash')
    def array(value, *, weak_type):
        # Saved grids can be weakly typed arrays. Preserve that JIT signature
        # as well as the bytes; plain jnp.asarray would silently strengthen it.
        result = _convert_element_type(jnp.asarray(value),value.dtype,weak_type=weak_type)
        if (result.dtype != value.dtype or result.weak_type != weak_type
                or np.asarray(result).tobytes() != value.tobytes()):
            raise ValueError('Restored argument differs in bytes, dtype or weak type')
        return result
    arguments = restore_arguments(data,groups,api,array)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    (args.out_dir/'metadata.json').write_text(json.dumps(dict(jax=jax.__version__,device=str(jax.devices()[0]),
        capture=str(args.capture),capture_sha256=hashlib.sha256(args.capture.read_bytes()).hexdigest(),
        sources=sources,repeats=args.repeats),indent=2))
    print(f'[boozer-repeat] worker={args.worker} JAX={jax.__version__} device={jax.devices()[0]}; '
          'original API, identical arguments, no extra jit or compiler options',flush=True)
    first = None
    first_changed = None
    rows = []
    with capture_ntx_ir(args.out_dir/'compiler',compiler):
        for index in range(args.repeats):
            result = jax.block_until_ready(api.booz_xform_from_inputs(**arguments))
            current = {key:np.asarray(value).copy() for key,value in result.items() if value is not None}
            if first is None:
                first = current
            changed = differences(first,current)
            reference = differences(groups['outputs'],current)
            rows.append(dict(repeat=index,vs_first=changed,vs_capture=reference))
            if changed and first_changed is None:
                first_changed = current
            print(f'[boozer-repeat] worker={args.worker} repeat={index} '
                  f'changed_from_first={len(changed)} changed_from_capture={len(reference)}',flush=True)
    np.savez_compressed(args.out_dir/'first_output.npz',**first)
    if first_changed is not None:
        np.savez_compressed(args.out_dir/'first_changed_output.npz',**first_changed)
    (args.out_dir/'comparison.json').write_text(json.dumps(rows,indent=2))
    programs = json.loads((args.out_dir/'compiler/programs.json').read_text())
    count = sum('booz_xform_jax_impl' in item['name'] for item in programs)
    print(f'[boozer-repeat] worker={args.worker} actual_Boozer_compilations={count} '
          f'variable_repeats={sum(bool(row["vs_first"]) for row in rows)}',flush=True)
    if count != 1:
        raise RuntimeError('Expected one actual Boozer compilation; captures retained')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=20)
    parser.add_argument('--worker',choices=('process_1','process_2'),help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.capture,args.out_dir = args.capture.resolve(),args.out_dir.resolve()
    if args.repeats < 2:
        parser.error('Require at least two repetitions')
    data,groups = read_capture(args.capture)
    if args.worker:
        return worker(args,data,groups)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error('Choose an empty output directory')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    results = {}
    for name in ('process_1','process_2'):
        folder = args.out_dir/name
        folder.mkdir()
        command = [sys.executable,str(Path(__file__).resolve()),'--capture',str(args.capture),
                   '--out-dir',str(folder),'--repeats',str(args.repeats),'--worker',name]
        with (folder/'run.log').open('w') as log:
            with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as process:
                for line in process.stdout:
                    print(line,end='',flush=True);log.write(line);log.flush()
                if process.wait():
                    raise RuntimeError(f'Worker failed; see {folder/"run.log"}')
        with np.load(folder/'first_output.npz',allow_pickle=False) as saved:
            results[name] = {key:saved[key].copy() for key in saved.files}
    rows = differences(results['process_1'],results['process_2'])
    (args.out_dir/'process_comparison.json').write_text(json.dumps(rows,indent=2))
    print(f'[boozer-repeat] independent_process_changed_fields={len(rows)}; saved {args.out_dir}',flush=True)
    for row in rows:
        print('[boozer-repeat] '+str(row),flush=True)


if __name__ == '__main__':
    main()
