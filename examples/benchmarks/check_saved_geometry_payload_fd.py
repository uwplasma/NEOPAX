"""Check the geometry input map against the actual saved AD16 cost gradients.

Reuse the recorded FD/AD VMEC state and tangent. Rebuild geometry and native
scan inputs, stopping before NTX. Contract input secants with the folded AD16
objective bars at four uniform steps. No equilibrium, implicit, coefficient,
ambipolar-root or transport solve, and no new AD sweep.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

from check_captured_ntx_directional_derivative import (
    captured_direction, fields, saved, snapshot_directory,
)
from check_saved_ntx_geometry_endpoints import collect_inputs, require_equal, write_csv

ROOT = Path(__file__).resolve().parents[2]


def payload_arrays(inputs, paths):
    """Resolve only the known geometry/channel/native-surface capture paths."""
    channels = inputs['channels']
    channels = channels.as_mapping() if hasattr(channels,'as_mapping') else channels
    result = {}
    for path in paths:
        match = re.fullmatch(r"\['(geometry|channels)'\]\.([A-Za-z0-9_]+)",path)
        surface = re.fullmatch(r"\['surfaces'\]\[(\d+)\]\.([A-Za-z0-9_]+)",path)
        if match:
            group,name = match.groups()
            value = channels[name] if group == 'channels' else getattr(inputs['geometry'],name)
        elif surface:
            index,name = surface.groups()
            value = getattr(inputs['surfaces'][int(index)],name)
        else:
            raise ValueError('Unsupported captured payload path: '+path)
        array = np.asarray(value)
        if array.dtype.kind not in 'fc' or not np.isfinite(array).all():
            raise ValueError('Nonfinite or non-floating payload: '+path)
        result[path] = array
    return result


def project(capture, direction):
    """Keep every saved row and input path; expose the three input groups."""
    rows,terms = [],{}
    for i in range(capture['geometry_gradients'].shape[0]):
        groups = {name:[] for name in ('geometry','channels','surfaces')}
        for path,value in direction.items():
            bar = capture[f'payload_bars[{i}]'+path]
            if bar.shape != value.shape or not np.isfinite(bar).all() or not np.isfinite(value).all():
                raise ValueError('Invalid payload bar/direction: '+path)
            group = path.split("'",2)[1]
            term = math.fsum(np.real(bar*value).reshape(-1))
            groups[group].append(term)
            terms[i,path] = term
        rows.append({group:math.fsum(values) for group,values in groups.items()})
    return np.asarray([math.fsum(r.values()) for r in rows]),rows,terms


def run_sweep(build_inputs, *, capture, direction, parameter, report, references, out_dir,
              step_scales=(1.,3.,10.,30.)):
    """NumPy-only bookkeeping around the supplied forward input constructor."""
    labels = report['objective_labels']
    params = [str(capture[f'parameter_labels[{i}]']) for i in range(capture['geometry_gradients'].shape[1])]
    reference = capture['geometry_gradients'][:,params.index(parameter)]
    control,ad_groups,_ = project(capture,direction)
    if len(reference) != len(labels) or not np.allclose(control,reference,rtol=1e-12,atol=1e-12):
        raise ValueError('Saved payload bars/tangents do not reconstruct the AD cost gradients')
    h0 = float(report['fd_step'])
    if not np.isfinite(h0) or h0 <= 0:
        raise ValueError('Saved FD step must be finite and positive')
    baseline = build_inputs(0.)
    geometry = lambda inputs:{f.name:np.asarray(getattr(inputs['geometry'],f.name))
                              for f in dataclasses.fields(inputs['geometry'])}
    require_equal(geometry(baseline),references['baseline'],'Recorded baseline geometry')
    base = payload_arrays(baseline,direction)
    np.savez_compressed(out_dir/'payload_baseline.npz',**base)
    rows,field_rows = [],[]
    delta = float(report.get('baseline_value',0.))
    for scale in step_scales:
        h = h0*scale
        print(f'[geometry-only] rebuilding input pair scale={scale:g}; no NTX solves',flush=True)
        minus = build_inputs((delta-h)-delta)
        plus = build_inputs((delta+h)-delta)
        if scale == 1.:
            require_equal(geometry(minus),references['minus'],'Recorded minus geometry')
            require_equal(geometry(plus),references['plus'],'Recorded plus geometry')
            print('[geometry-only] baseline and both original-step endpoint geometries match saved FD byte-for-byte',flush=True)
        a,b = payload_arrays(minus,direction),payload_arrays(plus,direction)
        require_shapes = all(a[k].shape == base[k].shape == b[k].shape
                             and a[k].dtype == base[k].dtype == b[k].dtype for k in direction)
        if not require_shapes:
            raise ValueError('Payload shape/dtype changed at a geometry endpoint')
        if scale == 1.:
            np.savez_compressed(out_dir/'payload_minus_h1.npz',**a)
            np.savez_compressed(out_dir/'payload_plus_h1.npz',**b)
        secant = {k:(b[k]-a[k])/(2*h) for k in direction}
        observed,fd_groups,terms = project(capture,secant)
        for i,label in enumerate(labels):
            gap = float(observed[i]-reference[i])
            relative = abs(gap/reference[i]) if reference[i] else (0. if gap==0. else float('inf'))
            row = dict(objective=label,parameter=parameter,step_scale=scale,fd_step=h,
                saved_ad=float(reference[i]),control_ad=float(control[i]),geometry_map_fd=float(observed[i]),
                signed_difference=gap,relative_difference_to_ad=relative,
                saved_full_fd=float(report['gradient_fd'][i]),
                map_minus_full_fd_at_same_step=float(observed[i]-report['gradient_fd'][i]) if scale==1. else None)
            for group in ad_groups[i]:
                row[group+'_ad'] = ad_groups[i][group]
                row[group+'_fd'] = fd_groups[i][group]
            rows.append(row)
            for path in direction:
                bar = capture[f'payload_bars[{i}]'+path]
                ad = math.fsum(np.real(bar*direction[path]).reshape(-1))
                field_rows.append(dict(objective=label,step_scale=scale,path=path,
                    ad=ad,fd=terms[i,path],signed_difference=terms[i,path]-ad))
            print(f'[geometry-only] scale={scale:g} {label} AD={reference[i]:.12e} FD={observed[i]:.12e} relative_to_ad={relative:.6e}',flush=True)
        write_csv(out_dir/'geometry_map_fd.csv',rows)
        write_csv(out_dir/'geometry_map_fields.csv',field_rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True,help='Completed FD16 run containing the shared AD reference')
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--parameter',choices=('RBC:1:0','ZBS:1:0'),default='ZBS:1:0')
    parser.add_argument('--native-precision',choices=('float64','extended'),default='float64')
    parser.add_argument('--step-scales',default='1,3,10,30',help='Comma-separated positive FD multipliers; use 1 for only the original saved step')
    args = parser.parse_args()
    try:
        scales = tuple(float(value) for value in args.step_scales.split(','))
        if not scales or not all(np.isfinite(value) and value > 0 for value in scales):
            raise ValueError
    except ValueError:
        parser.error('--step-scales must contain positive finite numbers')
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error('Choose an empty output directory')
    snapshots = snapshot_directory(args.run_dir)
    capture = saved(snapshots,'ad_compact_geometry_*.npz')
    tangents = saved(snapshots,'ad_compact_tangents_*.npz')
    terminal = saved(snapshots,'ad_terminal_*.npz')
    folder = args.run_dir/f"fd_{args.parameter.replace(':','_')}"
    report = json.loads((folder/'summary.json').read_text())
    if report['parameter_name'] != args.parameter or report.get('geometry_fd_lane') != 'frozen_linearized':
        parser.error('Requires the selected frozen-linearized geometry FD report')
    labels = [str(terminal[f'labels[{i}]']) for i in range(sum(k.startswith('labels[') for k in terminal))]
    if labels != report['objective_labels']:
        parser.error('Saved AD/FD objective order differs')
    baseline = saved(folder/'snapshots','fd_baseline_*.npz')
    require_equal(baseline,{k:v for k,v in terminal.items() if k in ('initial_y','final_y','final_time','values')
        or k.startswith(("support['geometry'].","support['database']."))},'AD/FD baseline')
    record = saved(folder/'snapshots','fd_geometry_linearization_*.npz')
    if str(record['parameter_label']) != args.parameter:
        parser.error('Recorded FD linearization belongs to another parameter')
    for prefix in ('state.','params.','mask.'):
        require_equal(fields(record,prefix),fields(capture,prefix),'Recorded '+prefix)
    columns = [str(tangents[f'parameter_labels[{i}]']) for i in range(sum(k.startswith('parameter_labels[') for k in tangents))]
    dot = {k:v[columns.index(args.parameter)] for k,v in fields(tangents,'state_tangents.').items()}
    require_equal(fields(record,'state_tangent.'),dot,'Recorded AD/FD state tangent')
    refs = {'baseline':fields(baseline,"support['geometry'].")}
    for sign in ('minus','plus'):
        refs[sign] = fields(saved(folder/'snapshots','fd_*_'+sign+'_*.npz'),"support['geometry'].")

    # GPU initialization only after all saved-array preflight checks.
    import jax
    import jax.numpy as jnp
    if jax.default_backend() != 'gpu' or not jax.config.x64_enabled:
        parser.error('Run on the large GPU machine with JAX_ENABLE_X64=true')
    sys.path.insert(0,str(ROOT))
    from vmex.core.solver import SpectralState
    from NEOPAX import _geometry_autodiff as geo
    import benchmark_transport_realtime_geometry_forward_fd as forward
    from benchmark_transport_autodiff_lagged_ntx import _prepare_benchmark_config
    config = _prepare_benchmark_config(args.run_dir/'config.toml',device=None,
        ntx_exact_derivative_mode='direct',radau_jacobian_reuse_mode='legacy')
    config['geometry']['vmec_input_file'] = str((ROOT/config['geometry']['vmec_input_file']).resolve())
    if str(record['vmec_response_mode']) != config['geometry'].get('vmec_response_mode','raw_block'):
        parser.error('Recorded/configured VMEC response modes differ')
    context = forward._geometry_context_from_config(config,args.parameter)
    bundle = dict(context=context,n_r=int(config['geometry'].get('n_radial',51)),
        state_star=SpectralState(**{k:jnp.asarray(v) for k,v in fields(record,'state.').items()}),
        state_tangent=SpectralState(**{k:jnp.asarray(v) for k,v in dot.items()}),
        native_precision=args.native_precision)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    provenance = [f'JAX={jax.__version__} device={jax.devices()[0]} native_precision={args.native_precision} step_scales={scales}']
    provenance_paths = (Path(geo.__file__),Path(forward.__file__),Path(__file__),args.run_dir/'config.toml',
        next((folder/'snapshots').glob('fd_geometry_linearization_*.npz')),
        next(snapshots.glob('ad_compact_geometry_*.npz')),next(snapshots.glob('ad_compact_tangents_*.npz')))
    if args.native_precision == 'extended':
        provenance_paths += (Path(__file__).with_name('_fd_native_precision.py'),)
    for path in provenance_paths:
        provenance.append(f'{path} sha256={hashlib.sha256(path.read_bytes()).hexdigest()}')
    (args.out_dir/'sources.txt').write_text('\n'.join(provenance)+'\n')
    print('[geometry-only] actual AD16 folded objective bars; no equilibrium/implicit/NTX/root/transport solve',flush=True)
    run_sweep(lambda step:collect_inputs(forward,geo,config,bundle,step),capture=capture,
        direction=captured_direction(tangents,args.parameter),parameter=args.parameter,
        report=report,references=refs,out_dir=args.out_dir,step_scales=scales)


if __name__ == '__main__':
    main()
