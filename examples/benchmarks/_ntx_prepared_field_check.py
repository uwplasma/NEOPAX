"""Compare saved prepared-field secants using the production NTX pullback."""
import csv
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


def contract_field_rows(bars, fields, epsilon_bars, epsilon_dot):
    """Contract each raw-coefficient adjoint row without a new forward solve."""
    epsilon_bars=np.asarray(epsilon_bars)
    terms=[[float(value)*float(epsilon_dot)] for value in epsilon_bars]
    for name,bar in bars.items():
        bar=np.asarray(bar)
        if name not in fields or bar.shape != (len(terms),*np.shape(fields[name])):
            raise ValueError(f'Missing or incompatible prepared field: {name}')
        if not np.isfinite(bar).all() or not np.isfinite(fields[name]).all():
            raise FloatingPointError(f'Nonfinite prepared field contraction: {name}')
        for row,value in enumerate(bar*np.asarray(fields[name])):
            terms[row].append(math.fsum(value.reshape(-1)))
    result=np.asarray([math.fsum(row) for row in terms])
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite prepared-field derivative')
    return result


def check_prepared_fields(directory, reference, *, template, names, grid, values, direction, nu, h, raw_jvp, raw_baseline):
    """One baseline production pullback; use only existing saved FD endpoints."""
    import jax
    import jax.numpy as jnp
    import ntx
    from ntx import _neopax_ad as native_ad
    from ntx._solver import _prepared_implicit_vjp_primal

    def load(label):
        with np.load(reference/f'{label}_stages.npz',allow_pickle=False) as z:
            return {key:z[key].copy() for key in z.files}
    base,minus,plus=(load(label) for label in ('baseline','minus','plus'))
    metadata=reference/'metadata.json'
    if metadata.is_file():
        for module,source in json.loads(metadata.read_text())['sources'].items():
            path=getattr(sys.modules.get(module),'__file__',None)
            if not path or hashlib.sha256(Path(path).read_bytes()).hexdigest()!=source['sha256']:
                raise ValueError(f'Captured NTX stage source differs: {module}')
    if any(float(z['fd_step'])!=h or float(z['nu'])!=float(nu) for z in (base,minus,plus)):
        raise ValueError('Saved stage FD step/collisionality differs')
    if not np.array_equal(base['coefficients'][[0,2,3]],raw_baseline):
        raise ValueError('Saved stage coefficient baseline differs from completed audit')
    for name,value in zip(names+('epsi_hat',),values,strict=True):
        anchor=base['input.'+name]
        value=np.asarray(value)
        if value.dtype!=anchor.dtype or value.shape!=anchor.shape or value.tobytes()!=anchor.tobytes():
            raise ValueError(f'Saved stage input differs: {name}')
    fields=tuple(key[9:] for key in base if key.startswith('geometry.'))
    for z in (base,minus,plus):
        if not np.array_equal(z['coefficients'],z['ordinary_coefficients']):
            raise ValueError('Saved observed coefficients differ from ordinary coefficients')
    def field_map(numbers):
        surface=dataclasses.replace(template,**dict(zip(names,numbers[:-1],strict=True)))
        geometry=ntx.prepare_monoenergetic_system(surface,grid).geometry
        return tuple(jnp.asarray(getattr(geometry,name)) for name in fields)
    primal_fields,dots=jax.block_until_ready(jax.jit(lambda x,d:jax.jvp(field_map,(x,),(d,)))(values,direction))
    exact_dot=dict(zip(fields,map(np.asarray,dots),strict=True))
    mismatches=[name for name,value in zip(fields,map(np.asarray,primal_fields),strict=True)
                if value.tobytes()!=base['geometry.'+name].tobytes()]
    print(f'[ntx-prepared] geometry JVP primal byte mismatches={mismatches}',flush=True)
    @jax.jit
    def pullback(numbers,nu_value,geometry_values,d_theta,d_zeta):
        surface=dataclasses.replace(template,**dict(zip(names,numbers[:-1],strict=True)))
        prepared=ntx.prepare_monoenergetic_system(surface,grid)
        geometry=dataclasses.replace(prepared.geometry,**dict(zip(fields,geometry_values,strict=True)))
        prepared=dataclasses.replace(prepared,geometry=geometry,d_theta=d_theta,d_zeta=d_zeta)
        case=ntx.MonoenergeticCase(nu_hat=nu_value,epsi_hat=numbers[-1])
        implicit_primal=_prepared_implicit_vjp_primal(prepared,nu_value,numbers[-1])[0]
        case_bars,geometry_bars=native_ad.pullback_prepared_database_coefficients_to_vmec_primitive_multi_rhs(
            prepared,case,jnp.eye(5,dtype=jnp.float64)[jnp.asarray([0,2,3])])
        return implicit_primal,case_bars.epsi_hat,{name:jnp.asarray(getattr(geometry_bars,name)) for name in fields}
    implicit,epsilon_bars,bars=jax.block_until_ready(pullback(values,nu,
        tuple(jnp.asarray(base['geometry.'+name]) for name in fields),
        jnp.asarray(base['d_theta']),jnp.asarray(base['d_zeta'])))
    bars={name:np.asarray(value) for name,value in bars.items()}
    field_fd={name:np.asarray((plus['geometry.'+name].astype(np.longdouble)-minus['geometry.'+name].astype(np.longdouble))/(2*np.longdouble(h)),dtype=np.float64) for name in fields}
    epsilon_fd=float((np.longdouble(plus['input.epsi_hat'])-np.longdouble(minus['input.epsi_hat']))/(2*np.longdouble(h)))
    vjp_true=contract_field_rows(bars,exact_dot,epsilon_bars,float(direction[-1]))
    vjp_field_fd=contract_field_rows(bars,field_fd,epsilon_bars,epsilon_fd)
    raw_fd=(plus['coefficients']-minus['coefficients'])/(2*h)
    directory.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(directory/'prepared_field_bars.npz',**{'bar.'+key:value for key,value in bars.items()},
        **{'jvp.'+key:value for key,value in exact_dot.items()},**{'fd.'+key:value for key,value in field_fd.items()},
        epsilon_bar=np.asarray(epsilon_bars),epsilon_dot=np.asarray(direction[-1]),epsilon_fd=epsilon_fd,
        implicit_primal=np.asarray(implicit),saved_forward_primal=base['coefficients'],fd_step=h)
    rows=[]
    for k,(name,index) in enumerate((('D11',0),('D13',2),('D33',3))):
        rows.append(dict(coefficient=name,saved_raw_jvp=float(raw_jvp[k]),production_vjp=float(vjp_true[k]),
            production_vjp_on_saved_field_fd=float(vjp_field_fd[k]),saved_raw_fd=float(raw_fd[index]),
            vjp_minus_jvp=float(vjp_true[k]-raw_jvp[k]),field_fd_minus_jvp=float(vjp_field_fd[k]-raw_jvp[k]),
            saved_fd_minus_field_fd=float(raw_fd[index]-vjp_field_fd[k]),
            saved_forward_primal=float(base['coefficients'][index]),implicit_pullback_primal=float(np.asarray(implicit)[index])))
        print('[ntx-prepared] '+str(rows[-1]),flush=True)
    with (directory/'prepared_field_comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return rows
