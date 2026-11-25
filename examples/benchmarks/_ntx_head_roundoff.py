"""Error analysis of saved elimination operands; no new primal/factorization."""
import csv

import numpy as np


def transpose_action(lu, pivots, rhs):
    """Apply recorded LU factors to a transpose error-functional vector."""
    lu,rhs=np.asarray(lu,dtype=np.longdouble),np.asarray(rhs,dtype=np.longdouble)
    y=np.zeros_like(rhs)
    for i in range(len(pivots)):
        y[i]=(rhs[i]-np.sum(lu[:i,i]*y[:i],dtype=np.longdouble))/lu[i,i]
    z=np.zeros_like(y)
    for i in range(len(pivots)-1,-1,-1):
        z[i]=y[i]-np.sum(lu[i+1:,i]*z[i+1:],dtype=np.longdouble)
    p=np.arange(len(pivots))
    for i,j in enumerate(pivots):
        p[i],p[j]=p[j],p[i]
    result=np.empty_like(z)
    result[p]=z
    return result


def row_budget(state, heads, weights, *, return_application_residual=False):
    """Split the original row-1 residual using its actual saved operands."""
    cast=lambda x:np.asarray(x,dtype=np.longdouble)
    lu,piv,lower,sigma=(state[key] for key in ('head_lu','head_pivots','head_lower','head_sigma'))
    lower,sigma=cast(lower),cast(sigma[...,1])
    modes=cast(state['f3_modes'])
    adjoint=np.zeros_like(weights,dtype=np.longdouble)
    for j in range(2,-1,-1):
        rhs=cast(weights[j]).copy()
        if j<2:
            rhs-=lower[j+1].T@adjoint[j+1]
        adjoint[j]=transpose_action(lu[j],piv[j],rhs)
    # Undo the first elimination row in the adjoint, without solving a primal.
    adjoint[1]-=transpose_action(lu[1],piv[1],cast(heads[0]['u_j']).T@adjoint[0])
    bar=adjoint[1]
    a1,a2=(cast(heads[j]['matrix']) for j in (1,2))
    d,u,x=(cast(heads[1][key]) for key in ('d_j','u_j','x'))
    t,b,next_sigma=(cast(heads[1][key][...,1]) for key in ('solved_sigma','b_j','sigma_next'))
    x_action=x@modes[1]
    q=transpose_action(lu[2],piv[2],u.T@bar)
    dot=lambda a,b:float(np.sum(a*b,dtype=np.longdouble))
    # P^-1 L U action, using only already-recorded factors and error vectors.
    factor=cast(lu[2]);p=np.arange(len(piv[2]))
    for i,j in enumerate(piv[2]):
        p[i],p[j]=p[j],p[i]
    y=x_action-t
    action=(np.tril(factor,-1)+np.eye(len(p),dtype=np.longdouble))@(np.triu(factor)@y)
    factor_action=np.empty_like(action);factor_action[p]=action
    application_residual=factor_action-lower[2]@modes[1]+next_sigma
    result={
        'head1_substitution':dot(bar,a1@modes[1]+lower[1]@modes[0]-sigma[1]),
        'schur_matrix_product':dot(bar,(d-a1)@modes[1]-u@x_action),
        'schur_source_product':dot(bar,sigma[1]-b+u@t),
        'head2_substitution':dot(q,a2@modes[2]+lower[2]@modes[1]-next_sigma),
        'lu_factorization':-dot(q,factor_action-a2@y),
        'lu_application':dot(q,application_residual),
        'original_row1':dot(bar,d@modes[1]+lower[1]@modes[0]+u@modes[2]-b),
    }
    result['decomposition_residual']=sum(result[key] for key in tuple(result)[:6])-result['original_row1']
    residual=a2.T@q-u.T@bar
    result['adjoint_relative_residual']=float(np.sqrt(np.sum(residual**2)/np.sum((u.T@bar)**2)))
    if return_application_residual:
        return result,application_residual
    return result


def report_head_roundoff(directory):
    """Report one captured D13 error budget using NumPy, never JAX."""
    if np.finfo(np.longdouble).eps>=np.finfo(np.float64).eps:
        raise RuntimeError('The saved error budget requires extended host accumulation')
    def load(path):
        with np.load(path,allow_pickle=False) as z:
            return {key:z[key].copy() for key in z.files}
    rows=[];coefficients=[];ordinary=[]
    for phase in ('baseline','minus','plus'):
        state=load(directory/f'{phase}_stages.npz')
        heads=[load(directory/f'{phase}_head_{j}.npz') for j in range(3)]
        g={key[9:]:value.astype(np.longdouble) for key,value in state.items() if key.startswith('geometry.')}
        pref=g['jacobian']/g['volume_prime']*np.longdouble(state['grid_dtheta'])*np.longdouble(state['grid_dzeta'])
        weight=pref*(-2*g['radial_drift_spatial'])/(g['coefficient_psi_scale']*g['b0'])
        weights=np.zeros_like(state['f3_modes'],dtype=np.longdouble)
        weights[0]=(weight*(np.longdouble(2)/3)).ravel(order='F')
        weights[2]=(weight/15).ravel(order='F')
        row=dict(phase=phase,**row_budget(state,heads,weights))
        row['observed_equals_ordinary']=bool(np.array_equal(state['coefficients'],state['ordinary_coefficients']))
        rows.append(row)
        coefficients.append(np.longdouble(state['coefficients'][2]))
        ordinary.append(np.longdouble(state['ordinary_coefficients'][2]))
    with (directory/'head_roundoff.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    h=np.longdouble(state['fd_step'])
    print(f'[ntx-head-roundoff] observed_raw_D13_fd={(coefficients[2]-coefficients[1])/(2*h):.12e} '
          f'ordinary_raw_D13_fd={(ordinary[2]-ordinary[1])/(2*h):.12e}',flush=True)
    for key in rows[0]:
        if key in ('phase','observed_equals_ordinary','adjoint_relative_residual'):
            continue
        contribution=(np.longdouble(rows[2][key])-np.longdouble(rows[1][key]))/(2*h)
        print(f'[ntx-head-roundoff] D13 {key} FD contribution={contribution:.12e}',flush=True)
    print('[ntx-head-roundoff] observed_matches='+str([row['observed_equals_ordinary'] for row in rows])+
          ' adjoint_relative_residuals='+str([row['adjoint_relative_residual'] for row in rows]),flush=True)
    return rows
