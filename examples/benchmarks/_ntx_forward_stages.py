"""Optional observation of an unchanged NTX forward solve at one saved point."""
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np


@contextmanager
def observe_head(directory, phase, retained, *, jax, direct):
    """Retain existing head outputs and observe the actual pre-LU matrices."""
    original_state = direct._block_thomas_truncated_fn_state
    original_factor = direct.lu_factor
    source = Path(direct.__file__).resolve()

    def write_head(index, matrix, operands):
        # Row 1 dominates the captured D13 error budget. Keep its actual
        # operands to separate Schur-product rounding from LU-solve error.
        extra = operands if int(index) == 1 else ({'u_j':operands['u_j']} if int(index) == 0 else {})
        np.savez_compressed(directory/f'{phase[0]}_head_{int(index)}.npz',
            matrix=np.asarray(matrix),**{key:np.asarray(value) for key,value in extra.items()})

    def factor(matrix, *args, **kwargs):
        frame = inspect.currentframe().f_back
        try:
            if frame.f_code.co_name == 'head_step' and Path(frame.f_code.co_filename).resolve() == source:
                operands = {key:frame.f_locals[key] for key in
                    ('d_j','u_j','x','solved_sigma','b_j','sigma_next')}
                jax.debug.callback(write_head,frame.f_locals['j'],matrix,operands,ordered=True)
        finally:
            del frame
        return original_factor(matrix,*args,**kwargs)

    def state(*args, **kwargs):
        result = original_state(*args,**kwargs)
        if retained:
            raise RuntimeError('Expected one retained-head solve per forward evaluation')
        retained.update(head_lu=result[1],head_pivots=result[2],head_sigma=result[3],head_lower=result[4])
        return result

    with patch.object(direct,'_block_thomas_truncated_fn_state',state),patch.object(direct,'lu_factor',factor):
        try:
            yield
        finally:
            retained.clear()


def capture_forward_stages(directory, *, template, names, grid, values, minus, plus, nu, ordinary, fd_step):
    """Call the same public forward solver and save its existing intermediate data.

    Extra outputs/callbacks can change compilation layout. Ordinary-vs-observed
    coefficients are therefore recorded explicitly; equality is not presumed.
    """
    import dataclasses
    import jax
    import jax.numpy as jnp
    import ntx
    from solvax import direct

    directory.mkdir(parents=True,exist_ok=True)
    sources = {}
    for name in ('ntx._solver','ntx._solver_scan','solvax.direct','solvax._reproducible'):
        path = getattr(sys.modules.get(name),'__file__',None)
        if path:
            sources[name] = dict(path=str(path),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    (directory/'metadata.json').write_text(json.dumps(dict(jax=getattr(jax,'__version__',None),
        sources=sources,fd_step=float(fd_step)),indent=2))
    phase,retained = ['baseline'],{}
    geometry_names = ('b','d_b_dtheta','d_b_dzeta','jacobian','b_sub_theta','b_sub_zeta',
        'b_sup_theta','b_sup_zeta','volume_prime','b2_mean','radial_drift_spatial','b0',
        'coefficient_psi_scale','transport_psi_scale')

    @jax.jit
    def observed(numbers,nu_value):
        surface = dataclasses.replace(template,**dict(zip(names,numbers[:-1],strict=True)))
        prepared = ntx.prepare_monoenergetic_system(surface,grid)
        result = ntx.solve_prepared(prepared,ntx.MonoenergeticCase(nu_hat=nu_value,epsi_hat=numbers[-1]))
        if set(retained) != {'head_lu','head_pivots','head_sigma','head_lower'}:
            raise RuntimeError('Configured NTX forward did not expose the expected retained-head state')
        payload = dict(retained)
        retained.clear()  # Keep traced arrays within this trace; no tracer escapes.
        payload.update({f'geometry.{name}':jnp.asarray(getattr(prepared.geometry,name)) for name in geometry_names})
        payload.update(f1_modes=result.f1_modes,f3_modes=result.f3_modes,
            d_theta=prepared.d_theta,d_zeta=prepared.d_zeta,
            grid_dtheta=jnp.asarray(prepared.geometry.grid.dtheta),
            grid_dzeta=jnp.asarray(prepared.geometry.grid.dzeta))
        return jnp.stack((result.D11,result.D31,result.D13,result.D33,result.D33_spitzer)),payload

    with observe_head(directory,phase,retained,jax=jax,direct=direct):
        for label,numbers,reference in zip(('baseline','minus','plus'),(values,minus,plus),ordinary,strict=True):
            phase[0] = label
            coefficients,payload = jax.block_until_ready(observed(numbers,nu))
            jax.effects_barrier()
            coefficients,reference = np.asarray(coefficients),np.asarray(reference)
            inputs = {f'input.{name}':np.asarray(value) for name,value in zip(names+('epsi_hat',),numbers,strict=True)}
            np.savez_compressed(directory/f'{label}_stages.npz',**inputs,
                **{name:np.asarray(value) for name,value in payload.items()},
                coefficients=coefficients,ordinary_coefficients=reference,nu=np.asarray(nu),fd_step=fd_step)
            print(f'[ntx-stages] phase={label} coefficients_equal={np.array_equal(coefficients,reference)} '
                  f'max_abs_difference={np.max(np.abs(coefficients-reference)):.9e} saved={directory}',flush=True)
    from _ntx_head_roundoff import report_head_roundoff
    report_head_roundoff(directory)
    return directory
