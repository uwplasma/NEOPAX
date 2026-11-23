"""Locate a saved geometry-FD table gap without rerunning transport.

Reuse the captured equilibrium and the FD tangent already checked on the GPU.
Repeat the original FD geometry construction, stopping before NTX/root/transport.
On selected radii, compare production full scans with scalar solves of exactly
the same numerical endpoints and a JVP along their numerical input secant.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import importlib
import math
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

from check_captured_ntx_directional_derivative import (
    SURFACE_FIELDS, TABLE_FIELDS, captured_direction, fd_reference, fields, point_indices,
    precision_reference, saved, snapshot_directory, table_interpolation_mode,
)

ROOT = Path(__file__).resolve().parents[2]


def require_equal(actual, reference, prefix):
    for name, value in reference.items():
        if name not in actual:
            raise ValueError(f'Missing {prefix}: {name}')
        a, b = np.asarray(actual[name]), np.asarray(value)
        if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
            raise ValueError(f'{prefix} differs: {name}')


def collect_inputs(forward, geo, config, bundle, step):
    """Use the actual FD entrypoint, interrupting it at its pre-NTX boundary."""
    original = geo.build_ntx_runtime_scan_inputs_from_vmec_state
    captured = {}
    class InputsReady(Exception):
        pass
    def scan(context, state, geometry, **kwargs):
        channels, surfaces = original(context, state, geometry, **kwargs)
        captured.update(geometry=geometry, channels=channels, surfaces=surfaces)
        raise InputsReady
    with patch.object(geo, 'build_ntx_runtime_scan_inputs_from_vmec_state', scan):
        try:
            forward._runtime_for_frozen_linearized_geometry_step(config, bundle, step_scale=step)
        except InputsReady:
            if not step:
                bundle['native_baseline_surfaces'] = captured['surfaces']
            return captured
    raise RuntimeError('Original FD constructor did not reach the pre-NTX boundary')


def gap_rows(point, reference, endpoint_values, h):
    """Signed telescoping budget; every term is in table-derivative units."""
    if not np.isfinite(h) or h <= 0:
        raise ValueError('FD step must be finite and positive')
    required = ('saved_minus', 'saved_plus', 'scan_minus', 'scan_plus',
                'scalar_minus', 'scalar_plus', 'input_secant_jvp', 'control_jvp')
    arrays = [np.asarray(reference), *(np.asarray(endpoint_values[k]) for k in required)]
    if any(a.shape != (3,) or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Expected three finite table coefficients at every boundary')
    secant = lambda route: (endpoint_values[route+'_plus']-endpoint_values[route+'_minus'])/(2*h)
    saved_fd, scan_fd, scalar_fd = (secant(route) for route in ('saved','scan','scalar'))
    upstream = endpoint_values['input_secant_jvp']
    control = endpoint_values['control_jvp']
    rows = []
    for k, field in enumerate(TABLE_FIELDS):
        rows.append(dict(radius_index=point[0], nu_index=point[1], field_index=point[2], field=field,
            reference_jvp=float(reference[k]), input_secant_jvp=float(upstream[k]),
            control_jvp=float(control[k]), audit_control_gap=float(reference[k]-control[k]),
            scalar_fd=float(scalar_fd[k]), scan_fd=float(scan_fd[k]), saved_fd=float(saved_fd[k]),
            upstream_gap=float(control[k]-upstream[k]),
            scalar_fd_gap=float(upstream[k]-scalar_fd[k]),
            scan_context_gap=float(scalar_fd[k]-scan_fd[k]),
            reproduction_gap=float(scan_fd[k]-saved_fd[k]),
            total_gap=float(reference[k]-saved_fd[k]),
            scan_minus_offset=float(endpoint_values['scan_minus'][k]-endpoint_values['saved_minus'][k]),
            scan_plus_offset=float(endpoint_values['scan_plus'][k]-endpoint_values['saved_plus'][k])))
    return rows


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--state-tangents', type=Path, required=True,
        help='state_tangents.npz from the completed --fd-tangent-only check')
    parser.add_argument('--table-audit', type=Path, required=True)
    parser.add_argument('--parameter', choices=('RBC:1:0','ZBS:1:0'), default='ZBS:1:0')
    parser.add_argument('--points', type=point_indices,
        default=point_indices('3:4:3,3:5:7,5:6:7,4:2:6,5:4:2,4:4:3'))
    parser.add_argument('--fd-reference-dir', type=Path)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error('Use a new empty output directory; saved endpoints are never overwritten')
    snapshots = snapshot_directory(args.run_dir)
    terminal = saved(snapshots, 'ad_terminal_*.npz')
    capture = saved(snapshots, 'ad_compact_geometry_*.npz')
    tangents = saved(snapshots, 'ad_compact_tangents_*.npz')
    folder, report = fd_reference(args)
    if report.get('geometry_fd_lane') != 'frozen_linearized':
        parser.error('Saved FD must use the frozen-linearized geometry lane')
    labels = [str(terminal[f'labels[{i}]']) for i in range(sum(k.startswith('labels[') for k in terminal))]
    if report.get('objective_labels') != labels:
        parser.error('Saved FD objective labels differ')
    h = float(report['fd_step'])
    if not np.isfinite(h) or h <= 0:
        parser.error('Saved FD step must be finite and positive')
    baseline = saved(folder/'snapshots', 'fd_baseline_*.npz')
    keys = ('initial_y','final_y','final_time','values')+tuple(k for k in terminal if k.startswith(("support['geometry'].","support['database'].")))
    require_equal(baseline, {k:terminal[k] for k in keys}, 'AD/FD baseline')
    minus, plus = (saved(folder/'snapshots', 'fd_*_'+sign+'_*.npz') for sign in ('minus','plus'))
    _, reference = precision_reference(args.table_audit, terminal, args.parameter, args.points)
    with np.load(args.state_tangents, allow_pickle=False) as archive:
        direction = {k:archive[k].copy() for k in archive.files}
    if str(direction['parameter']) != args.parameter:
        parser.error('State tangent parameter differs')
    column = [str(tangents[f'parameter_labels[{i}]']) for i in range(sum(k.startswith('parameter_labels[') for k in tangents))].index(args.parameter)
    ad_direction = {k:value[column] for k,value in fields(tangents,'state_tangents.').items()}
    require_equal(fields(direction,'ad.'), ad_direction, 'Tangent reference')
    state_fields = fields(capture,'state.')
    fd_direction = fields(direction,'fd.')
    if set(fd_direction) != set(state_fields) or any(fd_direction[k].shape != v.shape or fd_direction[k].dtype != v.dtype or not np.isfinite(fd_direction[k]).all() for k,v in state_fields.items()):
        parser.error('FD state tangent fields/shapes/dtypes/values differ')

    # No JAX import or physics execution above this boundary.
    import jax
    import jax.numpy as jnp
    if jax.default_backend() != 'gpu' or not jax.config.x64_enabled:
        parser.error('Run on the large GPU machine with JAX_ENABLE_X64=true')
    sys.path.insert(0,str(ROOT))
    import ntx
    from vmex.core.solver import SpectralState
    from NEOPAX import _geometry_autodiff as geo
    from NEOPAX._transport_flux_models import D11_POSITIVE_FLOOR, _build_ntx_field_channels
    import benchmark_transport_realtime_geometry_forward_fd as forward
    from benchmark_transport_autodiff_lagged_ntx import _prepare_benchmark_config
    config = _prepare_benchmark_config(args.run_dir/'config.toml',device=None,
        ntx_exact_derivative_mode='direct',radau_jacobian_reuse_mode='legacy')
    config['geometry']['vmec_input_file'] = str((ROOT/config['geometry']['vmec_input_file']).resolve())
    if str(capture.get('vmec_response_mode','raw_block')) != config['geometry'].get('vmec_response_mode','raw_block'):
        parser.error('Captured and configured VMEC response modes differ')
    nc = config['neoclassical']
    table_interpolation_mode(nc)
    if nc.get('ntx_scan_coefficient_reverse_mode','generic') != 'generic' or nc.get('ntx_scan_record_primal',False):
        parser.error('This check repeats the generic coefficient FD scan, without a primal record')
    if nc.get('ntx_scan_surface_backend','vmec') not in ('vmec','vmec_jax'):
        parser.error('Requires the captured native VMEC scan backend')
    context = forward._geometry_context_from_config(config,args.parameter)
    bundle = dict(context=context,state_star=SpectralState(**{k:jnp.asarray(v) for k,v in state_fields.items()}),
        state_tangent=SpectralState(**{k:jnp.asarray(v) for k,v in fd_direction.items()}),
        n_r=int(config['geometry'].get('n_radial',51)))
    args.out_dir.mkdir(parents=True,exist_ok=True)
    manifest = [f'JAX={jax.__version__} device={jax.devices()[0]} h={h:.17e}']
    for name in ('NEOPAX._geometry_autodiff','ntx._solver','ntx._solver_scan','ntx._neopax_scan','ntx._neopax_ad','vmex.core.implicit'):
        module = importlib.import_module(name)
        path = Path(module.__file__)
        manifest.append(f'{name} {path} sha256={hashlib.sha256(path.read_bytes()).hexdigest()}')
    manifest.extend(f'input {path} sha256={hashlib.sha256(path.read_bytes()).hexdigest()}' for path in
        (args.run_dir/'config.toml',args.state_tangents,Path(config['geometry']['vmec_input_file'])))
    (args.out_dir/'sources.txt').write_text('\n'.join(manifest)+'\n')
    print('[geometry-endpoints] no equilibrium/implicit/root/transport solve; reuse saved FD direction',flush=True)
    print('[geometry-endpoints] '+manifest[0],flush=True)
    inputs = {}
    delta = float(report.get('baseline_value',0.))
    for label, step in (('baseline',0.),('minus',(delta-h)-delta),('plus',(delta+h)-delta)):
        item = collect_inputs(forward,geo,config,bundle,step)
        channels = item['channels']
        item['channels'] = channels.as_mapping() if hasattr(channels,'as_mapping') else channels
        item['Es'] = _build_ntx_field_channels(jnp.asarray(nc['ntx_scan_rho']),jnp.asarray(nc['ntx_scan_er_tilde']),item['channels'])[1]
        inputs[label] = item
        archive = {'geometry.'+f.name:np.asarray(getattr(item['geometry'],f.name)) for f in dataclasses.fields(item['geometry'])}
        archive.update({'channels.'+k:np.asarray(v) for k,v in item['channels'].items()})
        for r,surface in enumerate(item['surfaces']):
            for f in dataclasses.fields(surface):
                v = getattr(surface,f.name)
                if v is not None and np.asarray(v).dtype.kind in 'biufc':
                    archive[f'surfaces[{r}].'+f.name] = np.asarray(v)
        np.savez_compressed(args.out_dir/(label+'_inputs.npz'),Es=np.asarray(item['Es']),**archive)
        if label == 'baseline':
            require_equal(fields(archive,'geometry.'),fields(terminal,"support['geometry']."),'Reconstructed baseline geometry')
        print(f'[geometry-endpoints] {label}: original FD inputs captured',flush=True)

    grid = ntx.GridSpec(n_theta=int(nc['ntx_scan_n_theta']),n_zeta=int(nc['ntx_scan_n_zeta']),n_xi=int(nc['ntx_scan_n_xi']))
    template = inputs['baseline']['surfaces'][0]
    names = tuple(f.name for f in dataclasses.fields(template) if getattr(template,f.name) is not None and np.asarray(getattr(template,f.name)).dtype.kind == 'f')
    for item in inputs.values():
        for surface in item['surfaces']:
            require_equal({k:getattr(surface,k) for k in ('m','n','nfp')},{k:getattr(template,k) for k in ('m','n','nfp')},'Native mode metadata')
    def scalar(values,nu):
        surface = dataclasses.replace(template,**dict(zip(names,values[:-2],strict=True)))
        prepared = ntx.prepare_monoenergetic_system(surface,grid)
        result = ntx.solve_prepared_coefficient_vector(prepared,ntx.MonoenergeticCase(nu_hat=nu,epsi_hat=values[-2]))
        return normalize(result[0],result[2],result[3],values[-1],nu)
    def normalize(d11,d13,d33,drds,nu):
        return jnp.stack((jnp.log10(jnp.maximum(D11_POSITIVE_FLOOR,d11*drds**2)),d13*drds,d33*nu))
    primal = jax.jit(scalar)
    secant_jvp = jax.jit(lambda v,d,nu:jax.jvp(lambda x:scalar(x,nu),(v,),(d,))[1])
    def native_numbers(value):
        native = geo._traceable_vmec_surfaces_from_state(context,value,
            s_values=geo._positive_transport_s_values_from_rho(nc['ntx_scan_rho']))
        return tuple(tuple(jnp.asarray(getattr(s,name),dtype=jnp.float64) for name in names) for s in native)
    _,native_dots = jax.jvp(native_numbers,(bundle['state_star'],),
        (SpectralState(**{k:jnp.asarray(v) for k,v in ad_direction.items()}),))
    ad_payload = captured_direction(tangents,args.parameter)
    nu_values = jnp.asarray(nc['ntx_scan_nu_v'],dtype=jnp.float64)
    rows = []
    saved_scans = {}
    for radius in sorted({p[0] for p in args.points}):
        scans = {}
        for label,item in inputs.items():
            nu_grid,es_grid = jnp.meshgrid(nu_values,item['Es'][radius],indexing='ij')
            # Same public scan and full (nu,Er) shape as the original FD build.
            result = ntx.solve_monoenergetic_scan(item['surfaces'][radius],grid,nu_grid,epsi_hat=es_grid)
            scans[label] = np.asarray(jax.block_until_ready(normalize(result['D11'],result['D13'],result['D33'],item['channels']['drds'][radius],nu_grid)))
            saved_scans[label+f'.r{radius}'] = scans[label]
        np.savez_compressed(args.out_dir/'scan_endpoints.npz',**saved_scans)
        for point in (p for p in args.points if p[0] == radius):
            _,n,e = point
            numerical = {label:tuple(jnp.asarray(getattr(item['surfaces'][radius],name),dtype=jnp.float64) for name in names)+(item['Es'][radius,e],item['channels']['drds'][radius]) for label,item in inputs.items()}
            d = tuple((p-m)/(2*h) for m,p in zip(numerical['minus'],numerical['plus'],strict=True))
            local = {label:np.asarray(jax.block_until_ready(primal(v,nu_values[n]))) for label,v in numerical.items()}
            values = {route+'_'+sign:np.asarray([source["support['database']."+field][point] for field in TABLE_FIELDS]) for route in ('saved',) for sign,source in (('minus',minus),('plus',plus))}
            values.update({route+'_'+sign:(scans[sign][:,n,e] if route == 'scan' else local[sign]) for route in ('scan','scalar') for sign in ('minus','plus')})
            values['input_secant_jvp'] = np.asarray(jax.block_until_ready(secant_jvp(numerical['baseline'],d,nu_values[n])))
            c = inputs['baseline']['channels']
            es_dot = nc['ntx_scan_er_tilde'][e]*(ad_payload["['channels'].dr_tildeds"][radius]*c['b00'][radius]+c['dr_tildeds'][radius]*ad_payload["['channels'].b00"][radius])
            control_direction = tuple(jnp.asarray(ad_payload[f"['surfaces'][{radius}].{name}"]) if name in SURFACE_FIELDS else native_dots[radius][i] for i,name in enumerate(names))+(es_dot,jnp.asarray(ad_payload["['channels'].drds"][radius]))
            values['control_jvp'] = np.asarray(jax.block_until_ready(secant_jvp(numerical['baseline'],control_direction,nu_values[n])))
            reference_dot = np.asarray([reference['database_dot.'+field][point] for field in TABLE_FIELDS])
            added = gap_rows(point,reference_dot,values,h)
            for k,row in enumerate(added):
                anchor = terminal["support['database']."+row['field']][point]
                row.update(scan_baseline_offset=float(scans['baseline'][k,n,e]-anchor),scalar_baseline_offset=float(local['baseline'][k]-anchor))
                print(f"[endpoint-gap] node={radius}:{n}:{e} field={row['field']} total={row['total_gap']:.9e} control={row['audit_control_gap']:.9e} upstream={row['upstream_gap']:.9e} scalar_FD={row['scalar_fd_gap']:.9e} scan_context={row['scan_context_gap']:.9e} reproduction={row['reproduction_gap']:.9e}",flush=True)
            rows.extend(added)
            write_csv(args.out_dir/'endpoint_gaps.csv',rows)
    before = saved(snapshots,'ad_before_scan_fold_*.npz')
    if [str(before[f'labels[{i}]']) for i in range(sum(k.startswith('labels[') for k in before))] != labels:
        raise ValueError('Support-adjoint objective labels differ')
    if any(not np.isfinite(before[f"support_bars[{i}]['database'].{r['field']}"][r['radius_index'],r['nu_index'],r['field_index']]) for i in range(len(labels)) for r in rows):
        raise ValueError('Nonfinite selected-node objective cotangent')
    weighted = []
    for i in range(sum(k.startswith('labels[') for k in before)):
        row = dict(objective=str(before[f'labels[{i}]']),selected_nodes=len(args.points))
        for key in ('total_gap','audit_control_gap','upstream_gap','scalar_fd_gap','scan_context_gap','reproduction_gap'):
            row[key] = math.fsum(float(before[f"support_bars[{i}]['database'].{r['field']}"][r['radius_index'],r['nu_index'],r['field_index']])*r[key] for r in rows)
        weighted.append(row)
        print('[endpoint-budget] '+str(row),flush=True)
    write_csv(args.out_dir/'selected_node_budget.csv',weighted)
    print('[geometry-endpoints] complete; selected-node budget only, not a new full gradient comparison',flush=True)


if __name__ == '__main__':
    main()
