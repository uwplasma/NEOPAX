"""Check NTX derivatives at selected or objective-weighted nodes of an AD16 run.

GPU only. Reuses saved VMEC state and numerical surface/channel directions.
No equilibrium, ambipolar root or transport integration. Selected-node mode
compares local FD/JVP/VJP; --audit-table differentiates raw solves at every
active node and compares their weighted sum with the recorded table VJP.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import sys
import tomllib
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SURFACE_FIELDS = ('b_cos', 'jacobian_cos', 'b_sub_theta_cos', 'b_sub_zeta_cos',
                  'b_sup_theta_cos', 'b_sup_zeta_cos', 'b0')
COEFFICIENTS = (('D11', 0), ('D13', 2), ('D33', 3))
TABLE_FIELDS = ('D11_log', 'D13', 'D33')
EXTRA_SURFACE_FIELDS = ('iota', 'psi_a_hat', 'phi_edge', 'r_n', 'r_hat',
                        'dpsi_hat_dr_hat', 'dr_hat_dpsi_hat', 'aminor_p',
                        'transport_psi_scale', 'requested_psi_n', 'psi_n')


def local_fd_samples(evaluate, values, direction, h0, *, jvp_vjp_only=False):
    """Retain ordinary FD scales, or perform no perturbed evaluations at all."""
    if jvp_vjp_only:
        yield None, None
        return
    for scale in (10., 3., 1.):
        h = h0*scale
        minus = evaluate(tuple(x-h*d for x,d in zip(values,direction,strict=True)))
        plus = evaluate(tuple(x+h*d for x,d in zip(values,direction,strict=True)))
        if not np.isfinite(minus).all() or not np.isfinite(plus).all():
            raise FloatingPointError('Nonfinite local coefficient FD endpoint')
        yield h, (plus-minus)/(2*h)


def captured_coefficient_bars(before, point, raw, drds, nu, *, d11_floor=1e-20):
    """Map the actual 13 objective table bars to raw-coefficient RHS columns."""
    labels = [str(before[f'labels[{i}]']) for i in range(sum(k.startswith('labels[') for k in before))]
    if np.shape(raw) != (5,) or not labels or not np.isfinite(raw).all():
        raise ValueError('Invalid raw coefficient vector or captured objective labels')
    bars = np.zeros((len(labels),5),dtype=np.float64)
    for i in range(len(labels)):
        prefix = f"support_bars[{i}]['database']."
        if raw[0]*drds**2 > d11_floor:
            bars[i,0] = before[prefix+'D11_log'][point]/(np.log(10.)*raw[0])
        bars[i,2] = before[prefix+'D13'][point]*drds
        bars[i,3] = before[prefix+'D33'][point]*nu
    if not np.isfinite(bars).all():
        raise ValueError('Nonfinite captured coefficient bars')
    return bars,labels


def saved(directory, pattern):
    paths = list(directory.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f'Expected one {pattern} in {directory}, found {len(paths)}')
    with np.load(paths[0], allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def fields(arrays, prefix):
    return {key[len(prefix):]: value for key, value in arrays.items()
            if key.startswith(prefix) and '.' not in key[len(prefix):]}


def snapshot_directory(run_dir):
    choices = [p for p in (run_dir/'ad_snapshots', run_dir/'ad/snapshots') if p.is_dir()]
    if len(choices) != 1:
        raise ValueError('Expected one AD snapshot directory: ad_snapshots or ad/snapshots')
    return choices[0]


def table_interpolation_mode(nc):
    mode = nc.get('interpolation_mode', 'generic')
    if mode not in ('generic', 'preprocessed_3d_radial_ntss1d'):
        raise ValueError('Table audit requires generic or preprocessed_3d_radial_ntss1d runtime tables')
    return mode


def saved_database_direction(folder, database, h):
    if not np.isfinite(h) or h <= 0:
        raise ValueError('FD step must be finite and positive')
    minus = saved(folder/'snapshots', 'fd_*_minus_*.npz')
    plus = saved(folder/'snapshots', 'fd_*_plus_*.npz')
    directions = {}
    for name, anchor in database.items():
        key = "support['database']."+name
        if key not in minus or key not in plus:
            raise ValueError(f'Missing endpoint support: {key}; use the FD capture with saved support')
        a,b = minus[key],plus[key]
        if any(x.shape != anchor.shape or x.dtype != anchor.dtype or not np.isfinite(x).all() for x in (a,b)):
            raise ValueError(f'Invalid FD database field: {name}')
        directions[name] = (b-a)/(2*h)
        if not np.isfinite(directions[name]).all():
            raise ValueError(f'Nonfinite FD database direction: {name}')
    return directions


def weighted_table_direction(before, prefix, directions):
    terms = []
    for name, direction in directions.items():
        bar = before[prefix+"['database']."+name]
        if bar.shape != direction.shape or not np.isfinite(bar).all() or not np.isfinite(direction).all():
            raise ValueError(f'Invalid table direction/cotangent: {name}')
        terms.append(math.fsum(np.real(bar*direction).reshape(-1)))
    return math.fsum(terms)


def precision_reference(directory, terminal, parameter, points):
    with (directory/'table_comparison.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    labels = [str(terminal[f'labels[{i}]']) for i in range(sum(k.startswith('labels[') for k in terminal))]
    if [r['objective'] for r in rows] != labels or any(r['parameter'] != parameter for r in rows):
        raise ValueError('Precision reference objective/parameter differs')
    raw = saved(directory,'raw_direction.npz')
    database = saved(directory,'database_direction.npz')
    active = {tuple(index) for index in raw['active_points'].tolist()}
    if not set(points).issubset(active):
        raise ValueError('Precision points must belong to the completed weighted audit')
    for key,value in terminal.items():
        if key.startswith("support['database']."):
            field = 'database.'+key.split('.',1)[1]
            if field not in database or value.shape != database[field].shape or value.dtype != database[field].dtype or value.tobytes() != database[field].tobytes():
                raise ValueError(f'Precision reference database differs: {field}')
    return raw,database


def precision_sweep(args, *, terminal, tangents, directions, nc, context, state, channels, surfaces, grid, h0):
    """FD of complete numerical NTX inputs against the already-saved full JVP."""
    import jax
    import jax.numpy as jnp
    import ntx
    from NEOPAX import _geometry_autodiff as geo
    from NEOPAX._transport_flux_models import D11_POSITIVE_FLOOR, _build_ntx_field_channels

    raw,reference = precision_reference(args.precision_from_table_audit,terminal,args.parameter,args.points)
    names = SURFACE_FIELDS+EXTRA_SURFACE_FIELDS
    column = [str(tangents[f'parameter_labels[{i}]']) for i in range(sum(k.startswith('parameter_labels[') for k in tangents))].index(args.parameter)
    state_dot = type(state)(**{name:jnp.asarray(value[column]) for name,value in fields(tangents,'state_tangents.').items()})
    def native_numbers(value):
        native = geo._traceable_vmec_surfaces_from_state(context,value,
            s_values=geo._positive_transport_s_values_from_rho(nc['ntx_scan_rho']))
        return tuple(tuple(jnp.asarray(getattr(surface,name),dtype=jnp.float64) for name in names) for surface in native)
    _,native_dots = jax.jvp(native_numbers,(state,),(state_dot,))
    template = surfaces[0]
    for surface in surfaces:
        if int(surface.nfp) != int(template.nfp) or not np.array_equal(surface.m,template.m) or not np.array_equal(surface.n,template.n):
            raise ValueError('Shared precision kernel requires identical native mode metadata')
    @jax.jit
    def primal(values,nu):
        surface = dataclasses.replace(template,**dict(zip(names,values[:-1],strict=True)))
        prepared = ntx.prepare_monoenergetic_system(surface,grid)
        return ntx.solve_prepared_coefficient_vector(prepared,ntx.MonoenergeticCase(nu_hat=nu,epsi_hat=values[-1]))
    @jax.jit
    def normalized(value,drds,nu):
        return jnp.stack((jnp.log10(jnp.maximum(D11_POSITIVE_FLOOR,value[0]*drds**2)),value[2]*drds,value[3]*nu))
    _,es,_ = _build_ntx_field_channels(jnp.asarray(nc['ntx_scan_rho']),jnp.asarray(nc['ntx_scan_er_tilde']),channels)
    rows = []
    for radius,n,e in args.points:
        surface = surfaces[radius]
        nu = jnp.asarray(nc['ntx_scan_nu_v'][n],dtype=jnp.float64)
        factor = nc['ntx_scan_er_tilde'][e]
        es_dot = factor*(directions["['channels'].dr_tildeds"][radius]*channels['b00'][radius]
                        + channels['dr_tildeds'][radius]*directions["['channels'].b00"][radius])
        values = tuple(jnp.asarray(getattr(surface,name),dtype=jnp.float64) for name in names)+(es[radius,e],)
        direction = tuple(jnp.asarray(directions[f"['surfaces'][{radius}].{name}"]) if name in SURFACE_FIELDS
                          else native_dots[radius][i] for i,name in enumerate(names))+(es_dot,)
        base = np.asarray(jax.block_until_ready(primal(values,nu)))
        anchor = raw['ordinary'][:,radius,n,e]
        if not np.allclose(base[[0,2,3]],anchor,rtol=1e-8,atol=1e-12):
            raise ValueError(f'Precision primal differs from completed audit at {(radius,n,e)}')
        drds = channels['drds'][radius]
        drds_dot = directions["['channels'].drds"][radius]
        for scale in getattr(args,'precision_step_scales',(1.,3.,10.,30.)):
            h = h0*scale
            minus = jax.block_until_ready(primal(tuple(x-h*d for x,d in zip(values,direction,strict=True)),nu))
            plus = jax.block_until_ready(primal(tuple(x+h*d for x,d in zip(values,direction,strict=True)),nu))
            norm_minus = np.asarray(jax.block_until_ready(normalized(minus,drds-h*drds_dot,nu)))
            norm_plus = np.asarray(jax.block_until_ready(normalized(plus,drds+h*drds_dot,nu)))
            raw_fd = (np.asarray(plus)-np.asarray(minus))/(2*h)
            norm_fd = (norm_plus-norm_minus)/(2*h)
            if not all(np.isfinite(value).all() for value in (minus,plus,raw_fd,norm_fd)):
                raise FloatingPointError(f'Nonfinite precision endpoint at {(radius,n,e)}, scale={scale}')
            errors = []
            for k,((_,index),field) in enumerate(zip(COEFFICIENTS,TABLE_FIELDS,strict=True)):
                derivative = float(reference['database_dot.'+field][radius,n,e])
                absolute = abs(float(norm_fd[k])-derivative)
                relative = absolute/abs(derivative) if derivative else (0. if absolute == 0. else None)
                rows.append(dict(radius_index=radius,nu_index=n,field_index=e,field=field,fd_scale=scale,fd_step=h,
                    raw_reference_jvp=float(raw['raw_jvp'][k,radius,n,e]),raw_fd=float(raw_fd[index]),
                    normalized_reference_jvp=derivative,normalized_fd=float(norm_fd[k]),
                    normalized_absolute_difference=absolute,normalized_relative_difference=relative,
                    raw_primal=float(base[index]),saved_raw_primal=float(anchor[k]),
                    normalized_minus=float(norm_minus[k]),normalized_plus=float(norm_plus[k]),
                    d11_floor_minus=bool(float(minus[0])*(float(drds)-h*float(drds_dot))**2 <= D11_POSITIVE_FLOOR),
                    d11_floor_plus=bool(float(plus[0])*(float(drds)+h*float(drds_dot))**2 <= D11_POSITIVE_FLOOR)))
                errors.append(f'{field}={relative:.6e}' if relative is not None else f'{field}_absolute={absolute:.6e}')
            print(f'[ntx-precision] point={radius}:{n}:{e} scale={scale:g} '+ ' '.join(errors),flush=True)
            with (args.out_dir/'precision.csv').open('w',newline='') as stream:
                writer = csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(f'[ntx-precision] saved {len(rows)} rows; complete numerical-input FD, no transport or coefficient VJP/JVP rerun',flush=True)


def fd_reference(args):
    folder = args.fd_reference_dir or args.run_dir/f"fd_{args.parameter.replace(':', '_')}"
    report = json.loads((folder/'summary.json').read_text())
    if report['parameter_name'] != args.parameter:
        raise ValueError('Saved FD report belongs to a different parameter')
    return folder, report


def point_indices(text):
    try:
        points = tuple(tuple(map(int, entry.split(':'))) for entry in text.split(','))
        if any(len(point) != 3 or min(point) < 0 for point in points):
            raise ValueError
    except ValueError as error:
        raise argparse.ArgumentTypeError('Use zero-based radius:nu:field triplets') from error
    if len(set(points)) != len(points):
        raise argparse.ArgumentTypeError('Points must be distinct')
    return points


def captured_direction(tangents, parameter):
    labels = [str(tangents[f'parameter_labels[{i}]'])
              for i in range(sum(key.startswith('parameter_labels[') for key in tangents))]
    column = labels.index(parameter)
    return {str(tangents[f'payload_paths[{i}]']): tangents[f'payload_tangents[{i}]'][column]
            for i in range(sum(key.startswith('payload_paths[') for key in tangents))}


def active_table_points(before, count):
    active = np.zeros(before["support_bars[0]['database'].D11_log"].shape, dtype=bool)
    for row in range(count):
        for field in TABLE_FIELDS:
            bar = before[f"support_bars[{row}]['database'].{field}"]
            if bar.shape != active.shape or not np.isfinite(bar).all():
                raise ValueError('Invalid captured table cotangent')
            active |= bar != 0
    return tuple(map(tuple, np.argwhere(active).tolist()))


def project_saved_support(archive, prefix, direction):
    result = 0.
    for key, bar in archive.items():
        if not key.startswith(prefix) or bar.dtype.kind not in 'fc':
            continue
        path = key[len(prefix):]
        if path.startswith("['database']."):
            continue
        if path not in direction:
            if np.any(bar != 0):
                raise ValueError(f'Missing captured direction for nonzero bar: {key}')
            continue
        if bar.shape != direction[path].shape:
            raise ValueError(f'Tangent/bar shape mismatch: {key}')
        result += float(np.sum(bar*direction[path]))
    return result


def audit_table(args, *, terminal, capture, tangents, directions, nc, context, state, channels, surfaces, grid):
    """Compare the entire captured table transpose against raw-solver JVPs."""
    import jax
    import jax.numpy as jnp
    import ntx
    from NEOPAX import _geometry_autodiff as geo
    from NEOPAX._transport_flux_models import (
        D11_POSITIVE_FLOOR, _build_ntx_field_channels,
        _ntx_runtime_scan_to_neopax_monoenergetic,
    )

    before = saved(snapshot_directory(args.run_dir), 'ad_before_scan_fold_*.npz')
    labels = [str(before[f'labels[{i}]']) for i in range(sum(key.startswith('labels[') for key in before))]
    points = active_table_points(before, len(labels))
    database = fields(terminal, "support['database'].")
    mode = table_interpolation_mode(nc)
    fd_folder,fd = fd_reference(args)
    if fd['objective_labels'] != labels:
        raise ValueError('Captured objective order differs from the saved FD reference')
    fd_baseline = saved(fd_folder/'snapshots','fd_baseline_*.npz')
    baseline_keys = ('initial_y','final_y','final_time')+tuple(
        key for key in terminal if key.startswith(("support['geometry'].","support['database'].")))
    fd_verified = all(key in fd_baseline and terminal[key].shape == fd_baseline[key].shape
        and terminal[key].dtype == fd_baseline[key].dtype
        and terminal[key].tobytes() == fd_baseline[key].tobytes() for key in baseline_keys)
    if (args.fd_reference_dir is not None or args.compare_saved_fd) and not fd_verified:
        raise ValueError('Selected FD reference does not share the saved AD baseline')
    saved_fd_dot = saved_database_direction(fd_folder,database,float(fd['fd_step'])) if args.compare_saved_fd else None
    shape = database['D11_log'].shape
    if shape != (len(nc['ntx_scan_rho']),len(nc['ntx_scan_nu_v']),len(nc['ntx_scan_er_tilde'])):
        raise ValueError('Captured table shape differs from the saved configuration')
    drds = np.asarray(channels['drds'])
    nu_grid = np.asarray(nc['ntx_scan_nu_v'])
    raw = [np.where(database['D11_log'] > np.log10(D11_POSITIVE_FLOOR),
                    10.**database['D11_log']/drds[:,None,None]**2, 0.),
           database['D13']/drds[:,None,None], database['D33']/nu_grid[None,:,None]]
    ordinary = [value.copy() for value in raw]
    forward_primal = [value.copy() for value in raw]
    raw_dot = [np.zeros(shape) for _ in raw]
    names = SURFACE_FIELDS + EXTRA_SURFACE_FIELDS
    column = [str(tangents[f'parameter_labels[{i}]']) for i in range(2)].index(args.parameter)
    state_dot = type(state)(**{name: jnp.asarray(value[column])
                               for name,value in fields(tangents,'state_tangents.').items()})
    # Complete native directions also test fields that had zero captured bars.
    def native_numbers(value):
        native = geo._traceable_vmec_surfaces_from_state(
            context, value, s_values=geo._positive_transport_s_values_from_rho(nc['ntx_scan_rho']))
        return tuple(tuple(jnp.asarray(getattr(surface,name),dtype=jnp.float64) for name in names)
                     for surface in native)
    _, native_dots = jax.jvp(native_numbers, (state,), (state_dot,))
    template = surfaces[0]
    for surface in surfaces:
        if int(surface.nfp) != int(template.nfp) or not np.array_equal(surface.m,template.m) or not np.array_equal(surface.n,template.n):
            raise ValueError('Shared kernel requires identical native mode metadata')

    def coefficient_vector(values, nu):
        surface = dataclasses.replace(template, **dict(zip(names,values[:-1],strict=True)))
        prepared = ntx.prepare_monoenergetic_system(surface,grid)
        return ntx.solve_prepared_coefficient_vector(prepared,ntx.MonoenergeticCase(nu_hat=nu,epsi_hat=values[-1]))
    primal = jax.jit(coefficient_vector)
    forward = jax.jit(lambda values,direction,nu: jax.jvp(
        lambda x:coefficient_vector(x,nu),(values,),(direction,)))
    rho, ertilde = jnp.asarray(nc['ntx_scan_rho']), jnp.asarray(nc['ntx_scan_er_tilde'])
    _, es, _ = _build_ntx_field_channels(rho,ertilde,channels)
    print(f'[captured-table] {len(points)} active cases; one shared raw-solver/JVP kernel; no coefficient VJP rerun',flush=True)
    case_rows=[]
    for step,(radius,n,e) in enumerate(points,1):
        surface=surfaces[radius]
        es_dot=ertilde[e]*(directions["['channels'].dr_tildeds"][radius]*channels['b00'][radius]
                           + channels['dr_tildeds'][radius]*directions["['channels'].b00"][radius])
        values=tuple(jnp.asarray(getattr(surface,name),dtype=jnp.float64) for name in names)+(es[radius,e],)
        direction=tuple(jnp.asarray(directions[f"['surfaces'][{radius}].{name}"])
                        if name in SURFACE_FIELDS else native_dots[radius][i]
                        for i,name in enumerate(names))+(es_dot,)
        nu=jnp.asarray(nu_grid[n])
        base=np.asarray(jax.block_until_ready(primal(values,nu)))
        jvp_base,dot=map(np.asarray,jax.block_until_ready(forward(values,direction,nu)))
        if not all(np.isfinite(value).all() for value in (base,jvp_base,dot)):
            raise FloatingPointError(f'Nonfinite raw coefficient/JVP at {(radius,n,e)}')
        for k,(coefficient,index) in enumerate(COEFFICIENTS):
            ordinary[k][radius,n,e]=base[index]
            forward_primal[k][radius,n,e]=jvp_base[index]
            raw_dot[k][radius,n,e]=dot[index]
            case_rows.append(dict(radius_index=radius,nu_index=n,field_index=e,coefficient=coefficient,
                ordinary_primal=float(base[index]),jvp_primal=float(jvp_base[index]),raw_jvp=float(dot[index])))
        if step == 1 or step % 50 == 0 or step == len(points):
            print(f'[captured-table] cases={step}/{len(points)} last={(radius,n,e)}',flush=True)
            np.savez_compressed(args.out_dir/'raw_direction.npz',active_points=np.asarray(points[:step]),
                ordinary=np.asarray(ordinary),jvp_primal=np.asarray(forward_primal),raw_jvp=np.asarray(raw_dot))

    channel_names=('a_b','b00','drds','dr_tildedr')
    channel_values=tuple(jnp.asarray(channels[name]) for name in channel_names)
    channel_dots=tuple(jnp.asarray(directions[f"['channels'].{name}"]) for name in channel_names)
    database_names=tuple(database)
    def converted(raw_values,channel_values):
        local=dict(zip(channel_names,channel_values,strict=True))
        er=ertilde[None,:]*local['dr_tildedr'][:,None]*local['b00'][:,None]
        scan=SimpleNamespace(rho=rho,nu_v=jnp.asarray(nu_grid),Er=er,drds=local['drds'],
                             D11=raw_values[0],D13=raw_values[1],D33=raw_values[2])
        result=_ntx_runtime_scan_to_neopax_monoenergetic(scan,a_b=local['a_b'],interpolation_mode=mode)
        return tuple(jnp.asarray(getattr(result,name),dtype=jnp.float64) for name in database_names)
    baseline,database_dot=jax.jvp(converted,
        (tuple(map(jnp.asarray,raw)),channel_values),(tuple(map(jnp.asarray,raw_dot)),channel_dots))
    baseline,database_dot=map(lambda values:tuple(map(np.asarray,values)),jax.block_until_ready((baseline,database_dot)))
    if not all(np.isfinite(value).all() for value in database_dot):
        raise FloatingPointError('Nonfinite normalized database direction')
    _, ab_dot=jax.jvp(converted,(tuple(map(jnp.asarray,raw)),channel_values),
        (tuple(jnp.zeros_like(value) for value in raw),
         tuple(dot if name == 'a_b' else jnp.zeros_like(dot)
               for name,dot in zip(channel_names,channel_dots,strict=True))))
    ab_dot=tuple(map(np.asarray,jax.block_until_ready(ab_dot)))
    for name,value in zip(database_names,baseline,strict=True):
        if not np.allclose(value,database[name],rtol=1e-12,atol=1e-12):
            raise ValueError(f'Normalization anchor differs from captured database: {name}')
    ordinary_db=tuple(map(np.asarray,jax.block_until_ready(converted(tuple(map(jnp.asarray,ordinary)),channel_values))))
    jvp_db=tuple(map(np.asarray,jax.block_until_ready(converted(tuple(map(jnp.asarray,forward_primal)),channel_values))))
    np.savez_compressed(args.out_dir/'database_direction.npz',**{
        **{f'database.{name}':value for name,value in database.items()},
        **{f'database_dot.{name}':dot for name,dot in zip(database_names,database_dot,strict=True)}})
    database_direction = dict(zip(database_names,database_dot,strict=True))
    def table_dot(prefix):
        return weighted_table_direction(before,prefix,database_direction)
    rows=[];component_rows=[];radius_rows=[];fd_field_rows=[]
    for i,label in enumerate(labels):
        complete=project_saved_support(capture,f'payload_bars[{i}]',directions)
        direct=project_saved_support(before,f'support_bars[{i}]',directions)
        reference=complete-direct
        actual=table_dot(f'support_bars[{i}]')
        captured_global=0.
        if "['channels'].a_b" in directions:
            captured_global=float(np.sum(capture[f"payload_bars[{i}]['channels'].a_b"]*directions["['channels'].a_b"]))
        forward_global=sum(float(np.sum(before[f"support_bars[{i}]['database']."+name]*dot))
                           for name,dot in zip(database_names,ab_dot,strict=True))
        radius_rows.append(dict(objective=label,parameter=args.parameter,radius='global_a_b',
            captured_table_vjp=captured_global,raw_solver_table_jvp=forward_global,
            absolute_difference=abs(captured_global-forward_global)))
        for radius in range(shape[0]):
            captured_radius=0.
            for path,dot in directions.items():
                bar=capture[f'payload_bars[{i}]'+path]
                if path.startswith(f"['surfaces'][{radius}]."):
                    captured_radius += float(np.sum(bar*dot))
                elif path.startswith("['channels'].") and np.ndim(dot) == 1:
                    captured_radius += float(bar[radius]*dot[radius])
            forward_radius=sum(float(np.sum(before[f"support_bars[{i}]['database']."+name][radius]*(dot-ab_dot[k])[radius]))
                for k,(name,dot) in enumerate(zip(database_names,database_dot,strict=True))
                if name in (*TABLE_FIELDS,'Er_list'))
            radius_rows.append(dict(objective=label,parameter=args.parameter,radius=radius,
                captured_table_vjp=captured_radius,raw_solver_table_jvp=forward_radius,
                absolute_difference=abs(captured_radius-forward_radius)))
        grouped=radius_rows[-(shape[0]+1):]
        if not np.isclose(sum(row['captured_table_vjp'] for row in grouped),reference,rtol=1e-10,atol=1e-10):
            raise ValueError('Incomplete saved table-radius transpose decomposition')
        if not np.isclose(sum(row['raw_solver_table_jvp'] for row in grouped),actual,rtol=1e-10,atol=1e-10):
            raise ValueError('Incomplete forward table-radius decomposition')
        explicit_direct=project_saved_support(before,f"components['objective_explicit'][{i}]",directions)
        explicit_table=table_dot(f"components['objective_explicit'][{i}]")
        explicit=explicit_direct+explicit_table
        rows.append(dict(objective=label,parameter=args.parameter,active_cases=len(points),
            fd_baseline_verified=fd_verified,
            captured_table_vjp=reference,raw_solver_table_jvp=actual,
            absolute_difference=abs(actual-reference),
            relative_difference=abs(actual-reference)/max(abs(reference),np.finfo(float).tiny),
            explicit_geometry_plus_table=explicit,
            fd_explicit_geometry=fd['fixed_final_state_objective_geometry_fd'][i],
            captured_full_gradient=complete,independent_table_full_gradient=direct+actual,
            state_chain_after_explicit=complete-explicit,
            independent_state_chain=direct+actual-explicit,
            fd_state_chain=fd['baseline_geometry_final_state_fd'][i],
            weighted_ordinary_primal_offset=sum(float(np.sum(before[f"support_bars[{i}]['database']."+name]*(value-baseline[k])))
                for k,(name,value) in enumerate(zip(database_names,ordinary_db,strict=True))),
            weighted_jvp_primal_offset=sum(float(np.sum(before[f"support_bars[{i}]['database']."+name]*(value-baseline[k])))
                for k,(name,value) in enumerate(zip(database_names,jvp_db,strict=True)))))
        if saved_fd_dot is not None:
            fd_table = weighted_table_direction(before,f'support_bars[{i}]',saved_fd_dot)
            rows[-1].update(saved_table_fd=fd_table,
                jvp_minus_recorded_table_vjp=actual-reference,
                jvp_minus_saved_table_fd=actual-fd_table)
            for name,dot in database_direction.items():
                bar = before[f"support_bars[{i}]['database']."+name]
                fd_field_rows.append(dict(objective=label,field=name,
                    jvp_contribution=math.fsum(np.real(bar*dot).reshape(-1)),
                    fd_contribution=math.fsum(np.real(bar*saved_fd_dot[name]).reshape(-1))))
        print('[captured-table] '+json.dumps(rows[-1]),flush=True)
        for component in ('initial_cache','initial_direct_rhs','initial_er_root','initial_profile','objective_explicit','transport_rhs'):
            prefix=f"components['{component}'][{i}]"
            component_rows.append(dict(objective=label,parameter=args.parameter,component=component,
                geometry=project_saved_support(before,prefix,directions),table=table_dot(prefix)))
    for name,values in (('table_comparison.csv',rows),('component_directions.csv',component_rows),
                        ('radius_comparison.csv',radius_rows),('case_directions.csv',case_rows),
                        ('table_fd_fields.csv',fd_field_rows)):
        if not values:
            continue
        with (args.out_dir/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader()
            writer.writerows({key:f'{value:.17e}' if isinstance(value,float) else value
                             for key,value in row.items()} for row in values)
    print(f'[captured-table] saved {len(rows)} weighted objective comparisons in {args.out_dir}',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--parameter', choices=('RBC:1:0', 'ZBS:1:0'), default='RBC:1:0')
    selection=parser.add_mutually_exclusive_group()
    selection.add_argument('--points', type=point_indices, default=point_indices('4:4:2,4:4:3,4:4:5'))
    selection.add_argument('--audit-table',action='store_true',
        help='Check all active table nodes against all captured objective cotangents; no local FD or coefficient VJP rerun')
    parser.add_argument('--fd-reference-dir',type=Path,
        help='Use an existing summary.json and snapshots directory without modifying the archived AD/FD run')
    parser.add_argument('--compare-saved-fd',action='store_true',
        help='With --audit-table, also compare against the saved FD endpoint database direction')
    parser.add_argument('--jvp-vjp-only',action='store_true',
        help='At --points, compare forward/reverse derivatives for all captured objective RHS rows without FD perturbations')
    parser.add_argument('--precision-from-table-audit',type=Path,
        help='At --points, reuse a completed weighted audit JVP and test numerical-input FD at 1,3,10,30 times the saved step')
    parser.add_argument('--precision-step-scales',type=float,nargs='+',default=(1.,3.,10.,30.),
        help='With --precision-from-table-audit, select positive FD step multipliers; use 1 for the original saved perturbation only')
    args = parser.parse_args()
    if any(not math.isfinite(scale) or scale <= 0 for scale in args.precision_step_scales):
        parser.error('--precision-step-scales must be finite and positive')
    if args.jvp_vjp_only and (args.audit_table or args.precision_from_table_audit is not None):
        parser.error('--jvp-vjp-only requires selected-point mode without a precision sweep')
    if args.precision_from_table_audit is not None and args.audit_table:
        parser.error('--precision-from-table-audit cannot be combined with --audit-table')
    if args.compare_saved_fd and not args.audit_table:
        parser.error('--compare-saved-fd requires --audit-table')
    snapshots = snapshot_directory(args.run_dir)
    terminal = saved(snapshots, 'ad_terminal_*.npz')
    before = saved(snapshots, 'ad_before_scan_fold_*.npz') if args.jvp_vjp_only else None
    capture = saved(snapshots, 'ad_compact_geometry_*.npz')
    tangents = saved(snapshots, 'ad_compact_tangents_*.npz')
    directions = captured_direction(tangents, args.parameter)
    _, fd = fd_reference(args)
    h0 = float(fd['fd_step'])
    if not np.isfinite(h0) or h0 <= 0:
        parser.error('Saved FD step must be finite and positive')
    with (args.run_dir/'config.toml').open('rb') as stream:
        config = tomllib.load(stream)
    nc = config['neoclassical']
    if nc.get('ntx_scan_surface_backend','vmec') not in {'vmec','vmec_jax'}:
        parser.error('This captured numerical surface check requires the VMEC scan backend')
    if args.audit_table:
        try:
            table_interpolation_mode(nc)
        except ValueError as error:
            parser.error(str(error))
    if args.precision_from_table_audit is not None:
        precision_reference(args.precision_from_table_audit,terminal,args.parameter,args.points)
    for radius, nu_index, field_index in (() if args.audit_table else args.points):
        if (radius >= len(nc['ntx_scan_rho']) or nu_index >= len(nc['ntx_scan_nu_v'])
                or field_index >= len(nc['ntx_scan_er_tilde'])):
            parser.error('Point is outside the captured scan grid')

    # Deferred imports: help and saved-array validation do not initialize JAX.
    import jax
    import jax.numpy as jnp
    if jax.default_backend() != 'gpu' or not jax.config.x64_enabled:
        parser.error('Run on the large GPU machine with JAX_ENABLE_X64=true; no CPU fallback')
    sys.path.insert(0, str(ROOT))
    import ntx
    from ntx import _neopax_ad as native_ad
    from ntx._solver import _prepared_implicit_vjp_primal
    from ntx.geometry import geometry_on_grid
    from vmex.core.solver import SpectralState
    from NEOPAX import _geometry_autodiff as geo
    from NEOPAX._geometry_models import VmecBoozer
    from NEOPAX._transport_flux_models import D11_POSITIVE_FLOOR, _build_ntx_field_channels
    from benchmark_transport_reverse_ad_only import _geometry_context_from_config

    for module in (geo,native_ad,sys.modules[_prepared_implicit_vjp_primal.__module__]):
        path=Path(module.__file__)
        print(f'[captured-ntx] source={path} sha256={hashlib.sha256(path.read_bytes()).hexdigest()}',flush=True)

    config['geometry']['vmec_input_file'] = str((ROOT/config['geometry']['vmec_input_file']).resolve())
    context = _geometry_context_from_config(config, args.parameter)
    state = SpectralState(**{key: jnp.asarray(value)
                             for key, value in fields(capture, 'state.').items()})
    geometry_fields = fields(terminal, "support['geometry'].")
    geometry_fields['n_r'] = int(geometry_fields['n_r'])
    geometry = VmecBoozer(**{key: jnp.asarray(value) if key != 'n_r' else value
                            for key, value in geometry_fields.items()})
    print('[captured-ntx] rebuilding numerical inputs from saved state; no equilibrium solve', flush=True)
    channels, surfaces = geo.build_ntx_runtime_scan_inputs_from_vmec_state(
        context, state, geometry, rho_scan=nc['ntx_scan_rho'], surface_backend='vmec')
    channels = channels.as_mapping() if hasattr(channels, 'as_mapping') else channels
    _, es, _ = _build_ntx_field_channels(
        jnp.asarray(nc['ntx_scan_rho']), jnp.asarray(nc['ntx_scan_er_tilde']), channels)
    grid = ntx.GridSpec(n_theta=int(nc['ntx_scan_n_theta']),
        n_zeta=int(nc['ntx_scan_n_zeta']), n_xi=int(nc['ntx_scan_n_xi']))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    basis = jnp.eye(5, dtype=jnp.float64)[jnp.asarray([0, 2, 3])]
    print(f'[captured-ntx] JAX={jax.__version__} device={jax.devices()[0]} '
          f'parameter={args.parameter} grid=({grid.n_theta},{grid.n_zeta},{grid.n_xi})', flush=True)
    if args.audit_table:
        audit_table(args,terminal=terminal,capture=capture,tangents=tangents,directions=directions,
            nc=nc,context=context,state=state,channels=channels,surfaces=surfaces,grid=grid)
        return
    if args.precision_from_table_audit is not None:
        precision_sweep(args,terminal=terminal,tangents=tangents,directions=directions,
            nc=nc,context=context,state=state,channels=channels,surfaces=surfaces,grid=grid,h0=h0)
        return

    # One compiled shape for every case. VMEC metadata stays in the closure;
    # only flat numerical leaves enter lower/compile (avoids ArgInfo constructors).
    for radius in sorted({point[0] for point in args.points}):
        surface = surfaces[radius]
        def local_surface(values):
            return dataclasses.replace(surface, **dict(zip(SURFACE_FIELDS, values[:7], strict=True)))

        def primal(values, nu):
            prepared = ntx.prepare_monoenergetic_system(local_surface(values), grid)
            return ntx.solve_prepared_coefficient_vector(
                prepared, ntx.MonoenergeticCase(nu_hat=nu, epsi_hat=values[7]))

        @jax.jit
        def forward(values, direction, nu):
            return jax.jvp(lambda x: primal(x, nu), (values,), (direction,))

        @jax.jit
        def adjoint(values, direction, nu, coefficient_bars=basis):
            prepared = ntx.prepare_monoenergetic_system(local_surface(values), grid)
            implicit_primal = _prepared_implicit_vjp_primal(prepared, nu, values[7])[0]
            case_bars, geometry_bars = native_ad.pullback_prepared_database_coefficients_to_vmec_primitive_multi_rhs(
                prepared, ntx.MonoenergeticCase(nu_hat=nu, epsi_hat=values[7]), coefficient_bars)
            _, pullback = jax.vjp(lambda x: geometry_on_grid(local_surface(x), grid), values)
            def contract(index):
                one = jax.tree_util.tree_map(lambda value: value[index], geometry_bars)
                (bars,) = pullback(one)
                return sum(jnp.vdot(bar, dot).real for bar, dot in zip(bars, direction, strict=True)) + case_bars.epsi_hat[index]*direction[7]
            return implicit_primal, jax.vmap(contract)(jnp.arange(coefficient_bars.shape[0]))

        primal_jit = jax.jit(primal)
        for _, nu_index, field_index in (point for point in args.points if point[0] == radius):
            nu = jnp.asarray(nc['ntx_scan_nu_v'][nu_index], dtype=jnp.float64)
            factor = jnp.asarray(nc['ntx_scan_er_tilde'][field_index], dtype=jnp.float64)
            es_dot = factor*(directions["['channels'].dr_tildeds"][radius]*channels['b00'][radius]
                            + channels['dr_tildeds'][radius]*directions["['channels'].b00"][radius])
            values = tuple(jnp.asarray(getattr(surface, key)) for key in SURFACE_FIELDS) + (es[radius, field_index],)
            direction = tuple(jnp.asarray(directions[f"['surfaces'][{radius}].{key}"])
                              for key in SURFACE_FIELDS) + (es_dot,)
            print(f'[captured-ntx] point={radius}:{nu_index}:{field_index}; ordinary primal and raw-solver JVP', flush=True)
            baseline = np.asarray(jax.block_until_ready(primal_jit(values, nu)))
            jvp_primal, tangent = map(np.asarray, jax.block_until_ready(forward(values, direction, nu)))
            drds = float(channels['drds'][radius])
            expected = np.asarray((10.**float(terminal["support['database'].D11_log"][radius,nu_index,field_index])/drds**2,
                float(terminal["support['database'].D13"][radius,nu_index,field_index])/drds,
                float(terminal["support['database'].D33"][radius,nu_index,field_index])/float(nu)))
            actual = baseline[[0,2,3]]
            primal_gap = np.abs(actual-expected)/np.maximum(np.abs(expected), np.finfo(float).tiny)
            print(f'[captured-ntx] saved-table primal relative differences={primal_gap}', flush=True)
            if not np.allclose(actual, expected, rtol=1e-8, atol=1e-12):
                raise ValueError('Reconstructed NTX primal differs from the captured table; stop before comparing derivatives')
            print('[captured-ntx] production database transpose', flush=True)
            coefficient_names = COEFFICIENTS
            if args.jvp_vjp_only:
                weights,labels = captured_coefficient_bars(before,(radius,nu_index,field_index),
                    baseline,drds,float(nu),d11_floor=D11_POSITIVE_FLOOR)
                print(f'[captured-ntx] actual captured objective RHS rows={len(labels)}; no FD perturbations',flush=True)
                result = adjoint(values,direction,nu,jnp.asarray(weights))
                coefficient_names = tuple((label,i) for i,label in enumerate(labels))
            else:
                result = adjoint(values,direction,nu)
            implicit_primal, reverse = map(np.asarray, jax.block_until_ready(result))
            if not all(np.isfinite(value).all() for value in (baseline, jvp_primal, implicit_primal, tangent, reverse)):
                raise FloatingPointError('Nonfinite coefficient or derivative')
            if args.jvp_vjp_only:
                print('[captured-ntx] forward versus pullback raw-primal relative differences='
                      f'{np.abs(baseline[[0,2,3]]-implicit_primal[[0,2,3]])/np.maximum(np.abs(baseline[[0,2,3]]),np.finfo(float).tiny)}',flush=True)
                np.savez_compressed(args.out_dir/f'point_{radius}_{nu_index}_{field_index}.npz',
                    forward_primal=baseline,jvp_primal=jvp_primal,pullback_primal=implicit_primal,
                    raw_solver_jvp=tangent,production_vjp=reverse,coefficient_bars=weights,objective_labels=np.asarray(labels))
            forward_values,implicit_values,jvp_values,tangent_values = baseline,implicit_primal,jvp_primal,tangent
            if args.jvp_vjp_only:
                forward_values,implicit_values,jvp_values,tangent_values = (weights@x for x in
                    (baseline,implicit_primal,jvp_primal,tangent))
            evaluate = lambda x: np.asarray(jax.block_until_ready(primal_jit(x,nu)))
            for h, fd_vector in local_fd_samples(evaluate,values,direction,h0,jvp_vjp_only=args.jvp_vjp_only):
                for channel, (name, index) in enumerate(coefficient_names):
                    fd_value = None if fd_vector is None else float(fd_vector[index])
                    jvp_value, vjp_value = float(tangent_values[index]), float(reverse[channel])
                    relative = abs(vjp_value-jvp_value)/max(abs(jvp_value),np.finfo(float).tiny)
                    row = dict(radius_index=radius, nu_index=nu_index, field_index=field_index,
                        parameter=args.parameter, coefficient=name, fd_step=h, raw_solver_jvp=jvp_value,
                        production_vjp=vjp_value, gradient_fd=fd_value, jvp_vjp_relative_difference=relative,
                        forward_primal=float(forward_values[index]), implicit_pullback_primal=float(implicit_values[index]),
                        raw_jvp_primal=float(jvp_values[index]),
                        raw_jvp_primal_relative_difference=float(abs(forward_values[index]-jvp_values[index])/max(abs(forward_values[index]),np.finfo(float).tiny)),
                        forward_pullback_primal_relative_difference=float(abs(forward_values[index]-implicit_values[index])/max(abs(forward_values[index]),np.finfo(float).tiny)),
                        fd_vjp_absolute_difference=None if fd_value is None else abs(fd_value-vjp_value))
                    if args.jvp_vjp_only:
                        row.update(objective=name,coefficient='objective_weighted_raw_coefficients',rhs_count=len(labels))
                    rows.append(row)
                    print('[captured-ntx] '+json.dumps(row), flush=True)
            with (args.out_dir/'comparison.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f'[captured-ntx] saved {len(rows)} rows in {args.out_dir}', flush=True)


if __name__ == '__main__':
    main()
