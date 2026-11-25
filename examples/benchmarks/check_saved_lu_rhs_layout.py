"""Replay existing LU applications with different RHS batching, without refinement."""
import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import time

import numpy as np

from _ntx_head_roundoff import row_budget


def same_bytes(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def panel_solve(solve, factors, rhs, columns, *, xp, lax):
    """Use the same factors/solve for each panel; preserve every RHS column."""
    if columns < 0 or rhs.ndim != 2:
        raise ValueError('Require nonnegative panel width and a rank-two RHS')
    n, count = rhs.shape
    if count == 0:
        return xp.zeros_like(rhs)
    if columns == 0 or columns >= count:
        return solve(factors, rhs)
    stop = count // columns * columns
    panels = rhs[:, :stop].reshape(n, stop // columns, columns).transpose(1, 0, 2)
    values = lax.map(lambda b: solve(factors, b), panels)
    values = values.transpose(1, 0, 2).reshape(n, stop)
    if stop < count:
        values = xp.concatenate((values, solve(factors, rhs[:, stop:])), axis=1)
    return values


def coefficient_weights(state):
    g = {k[9:]:v.astype(np.longdouble) for k,v in state.items() if k.startswith('geometry.')}
    pref = g['jacobian']/g['volume_prime']*np.longdouble(state['grid_dtheta'])*np.longdouble(state['grid_dzeta'])
    weight = pref*(-2*g['radial_drift_spatial'])/(g['coefficient_psi_scale']*g['b0'])
    weights = np.zeros_like(state['f3_modes'], dtype=np.longdouble)
    weights[0] = (weight*(np.longdouble(2)/3)).ravel(order='F')
    weights[2] = (weight/15).ravel(order='F')
    return weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--columns-per-call', default='0,1,16,64')
    args = parser.parse_args()
    columns = list(dict.fromkeys([0]+[int(n) for n in args.columns_per_call.split(',')]))
    if any(n < 0 for n in columns):
        parser.error('Column counts must be nonnegative; 0 is the native control')
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error('Choose an empty output directory')
    if np.finfo(np.longdouble).eps >= np.finfo(np.float64).eps:
        parser.error('The error analysis requires extended host accumulation')
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import lu_solve

    jax.config.update('jax_enable_x64', True)
    if jax.default_backend() != 'gpu':
        raise RuntimeError('This CUDA arithmetic check requires a GPU')
    versions = {d.metadata['Name']:d.version for d in metadata.distributions()
                if d.metadata.get('Name','').lower().startswith(
                    ('jax','nvidia-cublas','nvidia-cusolver','nvidia-cuda-runtime'))}
    record = dict(versions=versions, device=str(jax.devices()[0]),
                  device_kind=jax.devices()[0].device_kind, input_sha256={})
    print(f'[lu-rhs-layout] JAX={jax.__version__} device={jax.devices()[0]}; '
          'same saved LU factors and right-hand sides',flush=True)

    def load(name):
        path = args.capture/name
        record['input_sha256'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as z:
            return {key:z[key].copy() for key in z.files}

    cases = []
    for phase in ('baseline','minus','plus'):
        state = load(f'{phase}_stages.npz')
        heads = [load(f'{phase}_head_{j}.npz') for j in range(3)]
        if not np.array_equal(state['coefficients'], state['ordinary_coefficients']):
            raise ValueError(f'{phase}: captured forward differs from ordinary execution')
        cases.append((phase, state, heads, coefficient_weights(state)))
    steps = [float(state['fd_step']) for _,state,_,_ in cases]
    if not np.isfinite(steps).all() or steps[0] <= 0 or len(set(steps)) != 1:
        raise ValueError('Captured endpoint steps must agree and be positive')
    (args.out_dir/'metadata.json').write_text(json.dumps(record, indent=2))
    rows = []
    for count in columns:
        def solve(lu, piv, matrix, source):
            kwargs = dict(xp=jnp, lax=jax.lax)
            return (panel_solve(lu_solve, (lu,piv), matrix, count, **kwargs),
                    panel_solve(lu_solve, (lu,piv), source, count, **kwargs))
        function = jax.jit(solve)
        executable = None
        panel_rows = []
        for phase,state,heads,weights in cases:
            values = tuple(jnp.asarray(value) for value in
                           (state['head_lu'][2],state['head_pivots'][2],
                            state['head_lower'][2],heads[1]['sigma_next']))
            if executable is None:
                start = time.perf_counter()
                executable = function.lower(*values).compile()
                compile_s = time.perf_counter()-start
                (args.out_dir/f'columns_{count}_compiled.txt').write_text(executable.as_text())
                print(f'[lu-rhs-layout] columns={count} compile_s={compile_s:.6e} '
                      f'memory={executable.memory_analysis()}', flush=True)
            reference = tuple(map(np.asarray,jax.block_until_ready(executable(*values))))
            times = []
            for _ in range(3):
                start = time.perf_counter()
                repeated = jax.block_until_ready(executable(*values))
                times.append(time.perf_counter()-start)
            if not all(same_bytes(a,b) for a,b in zip(reference,map(np.asarray,repeated))):
                raise ValueError(f'{phase}: repeated fixed-input LU calls differ')
            x,t = reference
            row = dict(columns_per_call=count, phase=phase, median_s=float(np.median(times)),
                       matrix_matches_capture=same_bytes(x,heads[1]['x']),
                       source_matches_capture=same_bytes(t,heads[1]['solved_sigma']),
                       matrix_max_abs_difference=float(np.max(np.abs(x-heads[1]['x']))))
            replay_heads = [dict(head) for head in heads]
            replay_heads[1].update(x=x, solved_sigma=t)
            budget,residual = row_budget(state,replay_heads,weights,return_application_residual=True)
            row['lu_application_functional'] = budget['lu_application']
            row['application_residual_action_l2'] = float(np.sqrt(np.sum(residual**2)))
            panel_rows.append(row)
            np.savez_compressed(args.out_dir/f'columns_{count}_{phase}.npz',matrix_solution=x,source_solution=t)
        rows.extend(panel_rows)
        with (args.out_dir/'comparison.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        contribution = (np.longdouble(panel_rows[2]['lu_application_functional'])-
                        np.longdouble(panel_rows[1]['lu_application_functional']))/(2*np.longdouble(steps[0]))
        matches = [r['matrix_matches_capture'] and r['source_matches_capture'] for r in panel_rows]
        print(f'[lu-rhs-layout] columns={count} capture_matches={matches} '
              f'D13_lu_application_fd_error={contribution:.12e} '
              f'max_abs_functional={max(abs(r["lu_application_functional"]) for r in panel_rows):.6e} '
              f'max_residual_action_l2={max(r["application_residual_action_l2"] for r in panel_rows):.6e} '
              f'median_s={np.median([r["median_s"] for r in panel_rows]):.6e}',flush=True)
        if count == 0 and not all(matches):
            raise ValueError('Native control differs from the captured LU application; stop before comparing layouts')
    print(f'[lu-rhs-layout] saved {args.out_dir}; no factorization, refinement or transport run',flush=True)


if __name__ == '__main__':
    main()
